import ray
import socket
import logging
import numpy as np
from fastapi import FastAPI, HTTPException
from typing import Union, Optional, List, Tuple, Set, Dict
from enum import Enum
from pollux_job import PolluxJob, JobState
from ray.util.placement_group import get_current_placement_group,\
    PlacementGroup
import time
from util import try_import_ray_worker, is_ray_node_resource
import uuid
from alpa.global_env import global_config

RAY_CLUSTER_ADDRESS = "127.0.0.1:6379"
RAY_CLUSTER_NAMESPACE = "Alpa-AdaptDL-Ray-NameSpace"
POLICY_INITIAL_NUM_GPU = 1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SchedulerError(Exception):
    def __init__(self, message=None):
        self.message = message
        super().__init__(message)

class Orchestrator:
    ip_address: str
    jobs: Dict[str, PolluxJob]
    ray_cluster_address: str
    ray_cluster_namespace: str
    allocation_matrix: Dict[str, np.ndarray]
    all_host_num_devices: np.ndarray
    gpus_per_node: int
    jobs_queue: List[str]
    
    
    def __init__(self):
        self.ip_address = None
        self.jobs = {}
        self.ray_cluster_address = RAY_CLUSTER_ADDRESS
        self.ray_cluster_namespace = RAY_CLUSTER_NAMESPACE
        self._ray_init(address=self.ray_cluster_address, namespace=self.ray_cluster_namespace)
        self.allocation_matrix = {}
        self.jobs_queue = []
        
    
    def _ray_init(self, address: str, namespace: str):
        if not ray.is_initialized():
            ray_addr = address if address else "auto"
            ray.init(address=ray_addr,
                     ignore_reinit_error=True,
                     namespace=namespace)
        # Gather host ids
        all_host_info = []
        all_host_ips = []

        for node in ray.nodes():
            for key in node["Resources"]:
                if (is_ray_node_resource(key) and
                        global_config.ray_accelerator_name
                        in node["Resources"]):
                    all_host_info.append(node)
                    all_host_ips.append(key.split("node:")[-1])

        # Gather device info
        all_host_num_devices = []
        for host_info in all_host_info:
            number = host_info["Resources"][global_config.ray_accelerator_name]
            assert number.is_integer()
            all_host_num_devices.append(int(number))
        
        self.all_host_info = all_host_info
        self.all_host_ips = all_host_ips
        self.all_host_num_devices = np.array(all_host_num_devices)
        
        # Assuming that all nodes have equal number of GPUs, as is required by both Alpa and AdaptDL.
        # I.e., designed for clusters with each node having the # of GPUs in the powers of 2.
        self.gpus_per_node = all_host_num_devices[0]
            
    
    def register_job(self):
        job_id = str(uuid.uuid4())
        job = PolluxJob(id=job_id)
        self.jobs[job_id] = job
        return job_id
    
    
    def get_all_jobs(self):
        return self.jobs
    
    
    def replace_all_jobs(self, jobs: Dict[str, PolluxJob]):
        logger.info("Warning! Replacing all the jobs!")
        self.jobs = jobs
        
        
    def initial_request_placement_group(self, job_id: str):
        if job_id in self.allocation_matrix.keys() or self.jobs[job_id].status is not JobState.registered:
            raise SchedulerError("This job is not supposed to request initial resources!")
        
        # Get used and free resources' vectors
        used_resources = np.zeros(self.all_host_num_devices.shape, dtype=int)
        
        for allocation_vector in list(self.allocation_matrix.values()):
            used_resources += allocation_vector

        free_resources = self.all_host_num_devices - used_resources
        
        init_num_gpus = min(POLICY_INITIAL_NUM_GPU, self.gpus_per_node)
        
        allocated_node = None
        
        for i, num_free_gpus in enumerate(free_resources):
            if num_free_gpus >= init_num_gpus:
                allocated_node = i
        
        if allocated_node is None:
            self.jobs_queue.append(job_id)
            self.jobs[job_id].status = JobState.queued
            logger.info("queued")
        else:
            allocation_vector = np.array([init_num_gpus if i == allocated_node else 0 for i in range(self.all_host_num_devices.shape[0])])
            logger.info(f"allocation_vector - {allocation_vector}")
            self.allocation_matrix[job_id] = allocation_vector
            self.jobs[job_id].status = JobState.allocated
    
    def create_placement_group(self, num_hosts,
                            host_num_devices,
                            name,
                            job_id,
                            additional_resources_per_host=None):
        """Creates a placement group if it does not exist.

        If a placement group is already detected (in Tune integration),
        this will be a no-op.

        By default the placement group will be created with `SPREAD` strategy.
        This is optimized for colocating GPUs on different nodes.

        Args:
            num_hosts: the number of hosts to create the placement group for
            host_num_devices: the number of devices on each host
            additional_resources_per_host: additional resources per host

        Returns:
            The placement group
        """
        try:   
            if ray.util.get_placement_group(name) is not None:
                raise SchedulerError("placement group with the given name already exists!")
        except ValueError as e:
            pass
        
        current_placement_group = get_current_placement_group()
        ray_worker = try_import_ray_worker()
        worker = ray_worker.global_worker  # pylint: disable=protected-access
        should_capture_child_tasks_in_placement_group = (
            worker.should_capture_child_tasks_in_placement_group)
        should_create_placement_group = (
            current_placement_group is None or
            not should_capture_child_tasks_in_placement_group)

        if should_create_placement_group:
            # `should_create_placement_group` is always True when using alpa alone.
            # `should_create_placement_group` can be false when integrated with Tune
            additional_resources_per_host = (additional_resources_per_host or {})
            bundles = [{
                "CPU": 1,
                "GPU": host_num_devices[i],
                **additional_resources_per_host
            } for i in range(num_hosts)]

            # Alpa Placement Group: `SPREAD` strategy is required
            # https://docs.ray.io/en/latest/ray-core/placement-group.html#strategy-types
            # Each bundle must be scheduled in a separate node.
            strategy = "SPREAD"

            placement_group = ray.util.placement_group(bundles,
                                                    strategy=strategy,
                                                    name=name or "")
            logger.info("Waiting for placement group to start.")
            timeout = 100
            ready, _ = ray.wait([placement_group.ready()], timeout=timeout)
            if ready:
                logger.info("Placement group has started.")
            else:
                raise TimeoutError(
                    "Placement group creation timed out. Make sure your "
                    "cluster either has enough resources or use an "
                    "autoscaling cluster. If you are running on a cluster, "
                    "make sure you specify an address in `ray.init()`, for example,"
                    ' `ray.init("auto")`. You can also increase the timeout by '
                    "setting the ALPA_PLACEMENT_GROUP_TIMEOUT_S environment "
                    "variable. Current resources available: "
                    f"{ray.available_resources()}, resources requested by "
                    f"the placement group: {placement_group.bundle_specs}")
            self.jobs[job_id].pg_name = name
            return placement_group
        else:
            return current_placement_group
        
        
orchestrator = Orchestrator()