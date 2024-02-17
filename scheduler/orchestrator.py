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
import asyncio

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
                        "GPU"
                        in node["Resources"]):
                    all_host_info.append(node)
                    all_host_ips.append(key.split("node:")[-1])

        # Gather device info
        all_host_num_devices = []
        for host_info in all_host_info:
            number = host_info["Resources"]["GPU"]
            assert number.is_integer()
            all_host_num_devices.append(int(number))
        
        self.all_host_info = all_host_info
        self.all_host_ips = all_host_ips
        self.all_host_num_devices = np.array(all_host_num_devices)
        self.all_node_ids = list(node['NodeID'] for node in all_host_info)
        
        # logger.info(f"all_host_info: {self.all_host_info}")
        # logger.info(f"all_host_ips: {self.all_host_ips}")
        # logger.info(f"all_node_ids: {self.all_node_ids}")
        
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
    
    def get_all_info(self):
        info = {"ip_address": self.ip_address,
                "jobs": self.jobs,
                "ray_cluster_address": self.ray_cluster_address,
                "ray_cluster_namespace": self.ray_cluster_namespace,
                "allocation_matrix": {k: v.tolist() for k, v in self.allocation_matrix.items()},
                "all_host_num_devices": self.all_host_num_devices,
                "gpus_per_node": self.gpus_per_node,
                "jobs_queue": self.jobs_queue,
                "all_host_info": self.all_host_info,
                "all_host_ips": self.all_host_ips,
                "all_host_num_devices": self.all_host_num_devices.tolist(),
                "all_node_ids": self.all_node_ids}
        return info
    
    
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
        
        if allocated_node is None: # TODO: also check if such a placement group is available
            self.jobs_queue.append(job_id)
            self.jobs[job_id].status = JobState.queued
            logger.info("queued")
        else:
            allocation_vector = np.array([init_num_gpus if i == allocated_node else 0 for i in range(self.all_host_num_devices.shape[0])])
            logger.info(f"allocation_vector - {allocation_vector}")
            self.allocation_matrix[job_id] = allocation_vector
            self.jobs[job_id].status = JobState.allocated

    
    async def create_placement_group(self, num_hosts,
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
                    ' `ray.init("auto")`. Current resources available: '
                    f"{ray.available_resources()}, resources requested by "
                    f"the placement group: {placement_group.bundle_specs}")
            self.jobs[job_id].pg_name = name
            pg_table = ray.util.placement_group_table(placement_group)
            logger.info(f"PG table: {pg_table}")
            node_ids = [v for k, v in pg_table['bundles_to_node_id'].items()]
            node_indeces = [self.all_node_ids.index(node_id) for node_id in node_ids]
            gpus_on_nodes = [int(v['GPU']) for k, v in pg_table['bundles'].items()]
            # logger.info(f"node IDs of bundles: {node_ids}")
            # logger.info(f"indeces of node IDs in the allocation matrix: {[self.all_node_ids.index(node_id) for node_id in node_ids]}")
            # logger.info(f"gpus_on_nodes: {gpus_on_nodes}")
            # logger.info(f"allocation matrix before: {self.allocation_matrix}")
            
            self.allocation_matrix[job_id] = np.zeros(self.all_host_num_devices.shape, dtype=int)
            for node_index, num_gpus in zip(node_indeces, gpus_on_nodes):
                self.allocation_matrix[job_id][node_index] = num_gpus
            
            # logger.info(f"allocation matrix after: {self.allocation_matrix}")
            self.jobs[job_id].status = JobState.allocated

            from main import send_message_to_client
            
            await send_message_to_client(job_id=job_id, message="Created placement group for you - websocket message")

            # from util import periodically_send_messages

            # asyncio.create_task(periodically_send_messages(job_id))
            
            return placement_group
        else:
            return current_placement_group


    def update_state(self, job_id: str, pollux_agent):
        if job_id not in self.jobs.keys():
            raise SchedulerError("This job is not found in the scheduler's database")
        self.jobs[job_id].pollux_agent = pollux_agent
        
    
    def release_resources(self, job_id: str):
         if job_id not in self.allocation_matrix.keys():
             raise SchedulerError("This job does not have any allocated resources!")
         placement_group = ray.util.get_placement_group(self.jobs[job_id].pg_name)
         ray.util.remove_placement_group(placement_group)
         self.jobs[job_id].status = JobState.ended
         del self.allocation_matrix[job_id]
         logger.info(f"Resources of job {job_id} released!")
        
        
orchestrator = Orchestrator()