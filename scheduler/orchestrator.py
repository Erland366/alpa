import ray
import socket
import logging
import numpy as np
from fastapi import FastAPI, HTTPException
from typing import Union, Optional, List, Tuple, Set, Dict
from enum import Enum
from pollux_job import PolluxJob, JobState, ResourceReleaseReason
from ray.util.placement_group import get_current_placement_group,\
    PlacementGroup
import time
from util import try_import_ray_worker, is_ray_node_resource
import uuid
import asyncio
from datetime import datetime
from cluster_optimization import list_possible_allocations
from goodput import GoodputFunction
import math
from collections import defaultdict
import copy

RAY_CLUSTER_ADDRESS = "127.0.0.1:6379"
RAY_CLUSTER_NAMESPACE = "Alpa-AdaptDL-Ray-NameSpace"
POLICY_INITIAL_NUM_GPU = 1
FAIRNESS_KNOB = -1

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
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
    job_ids_reallocating_resources: Dict[str, Tuple[int, int]] # { job_id: (num_gpus_per_node, num_nodes) }
    
    
    def __init__(self):
        self.ip_address = None
        self.jobs = {}
        self.ray_cluster_address = RAY_CLUSTER_ADDRESS
        self.ray_cluster_namespace = RAY_CLUSTER_NAMESPACE
        self._ray_init(address=self.ray_cluster_address, namespace=self.ray_cluster_namespace)
        self.allocation_matrix = {}
        self.jobs_queue = []
        self.realloc_requests_once = False # temporary for tests
        self.first_job_arrived = False
        self.job_ids_reallocating_resources = {}
        
    
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

    def get_jsonable_jobs(self):
        jsonable_jobs = {}
        for k, v in self.jobs.items():
            jsonable_job_object = PolluxJob(v.id)
            jsonable_job_object.creation_time = v.creation_time
            jsonable_job_object.status = v.status
            jsonable_job_object.pg_name = v.pg_name
            jsonable_job_object.pollux_agent_jsonable = {}
            jsonable_job_object.pollux_agent_jsonable['iter'] = v.pollux_agent.iter
            jsonable_job_object.pollux_agent_jsonable['total_batch_size'] = v.pollux_agent.total_batch_size
            jsonable_job_object.pollux_agent_jsonable['alloc_vector'] = v.pollux_agent.alloc_vector
            jsonable_job_object.pollux_agent_jsonable['scheduler_enabled'] = v.pollux_agent.scheduler_enabled
            jsonable_job_object.pollux_agent_jsonable['scheduler_address'] = v.pollux_agent.scheduler_address
            jsonable_job_object.pollux_agent_jsonable['job_id'] = v.pollux_agent.job_id
            jsonable_job_object.pollux_agent_jsonable['grad_norm_sqr'] = v.pollux_agent.grad_norm_sqr
            jsonable_job_object.pollux_agent_jsonable['grad_variance'] = v.pollux_agent.grad_variance
            jsonable_job_object.pollux_agent_jsonable['max_batch_size'] = v.pollux_agent.max_batch_size
            jsonable_job_object.pollux_agent_jsonable['local_bsz_bounds'] = v.pollux_agent.local_bsz_bounds
            jsonable_job_object.pollux_agent_jsonable['init_batch_size'] = v.pollux_agent.init_batch_size
            jsonable_job_object.pollux_agent_jsonable['total_overhead_time'] = v.pollux_agent.total_overhead_time

            jsonable_jobs[k] = jsonable_job_object
        return jsonable_jobs
    
    def get_all_jobs(self):
        return self.get_jsonable_jobs()
    
    def get_all_info(self):
        info = {"ip_address": self.ip_address,
                "jobs": self.get_jsonable_jobs(),
                "ray_cluster_address": self.ray_cluster_address,
                "ray_cluster_namespace": self.ray_cluster_namespace,
                "allocation_matrix": {k: v.tolist() for k, v in self.allocation_matrix.items()},
                "all_host_num_devices": self.all_host_num_devices,
                "gpus_per_node": self.gpus_per_node,
                "jobs_queue": self.jobs_queue,
                "all_host_info": self.all_host_info,
                "all_host_ips": self.all_host_ips,
                "all_host_num_devices": self.all_host_num_devices.tolist(),
                "all_node_ids": self.all_node_ids,
                "job_ids_reallocating_resources": self.job_ids_reallocating_resources}
        return info
    
    
    def replace_all_jobs(self, jobs: Dict[str, PolluxJob]):
        logger.info("Warning! Replacing all the jobs!")
        self.jobs = jobs
        
        
    async def initial_request_placement_group(self, job_id: str, name: str):
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
            logger.info(f"queued job {job_id}")
            return None # TODO: instead, implement a queueing mechanism, job should wait for a placement group
        else:
            # allocation_vector = np.array([init_num_gpus if i == allocated_node else 0 for i in range(self.all_host_num_devices.shape[0])])
            # logger.info(f"allocation_vector - {allocation_vector}")
            # self.allocation_matrix[job_id] = allocation_vector
            num_hosts = 1
            host_num_devices = [init_num_gpus] * num_hosts
            # pg_name = RAY_CLUSTER_NAMESPACE + "_pg_" + job_id
            # self.jobs[job_id].status = JobState.allocated
            return await self.create_placement_group(num_hosts, host_num_devices, name, job_id)

    
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
            timeout = 5 # TODO: need to handle queueing, just denying the job right now
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

            if not self.first_job_arrived:
                self.first_job_arrived = True
                asyncio.create_task(self.periodically_reallocate_resources())
            
            return placement_group
        else:
            return current_placement_group


    def update_state(self, job_id: str, pollux_agent):
        if job_id not in self.jobs.keys():
            raise SchedulerError("This job is not found in the scheduler's database")
        self.jobs[job_id].pollux_agent = pollux_agent
        
    
    def release_resources(self, job_id: str, reason: ResourceReleaseReason):
        if job_id not in self.allocation_matrix.keys():
            raise SchedulerError("This job does not have any allocated resources!")
        try:
            placement_group = ray.util.get_placement_group(self.jobs[job_id].pg_name)
            ray.util.remove_placement_group(placement_group)
            if reason is ResourceReleaseReason.reallocation:
                self.jobs[job_id].status = JobState.reallocating
            else:
                self.jobs[job_id].status = JobState.ended
                self.job_ids_reallocating_resources.pop(job_id, None)
            del self.allocation_matrix[job_id]
            logger.info(f"Resources of job {job_id} released!")
        except:
            # del self.allocation_matrix[job_id]
            raise SchedulerError("Placement group not found, job probably called alpa.shutdown().")


    def get_fair_num_gpus(self):
        total_num_gpus = np.sum(self.all_host_num_devices)
        num_running_jobs = len(self.allocation_matrix)

        share = total_num_gpus // num_running_jobs
        if share == 0:
            return 0
        power = 2 ** math.floor(math.log2(share))
        return power


    def optimize_resource_allocation(self) -> Tuple[Dict, Dict]:
        candidate_jobs = {}
        for job_id, job in self.jobs.items():
            if job.status == JobState.allocated and job.pollux_agent is not None \
                and (job.pollux_agent.grad_norm_sqr is not None and job.pollux_agent.grad_variance is not None):
                candidate_jobs[job_id] = job
        
        possible_allocations = list_possible_allocations(self.all_host_num_devices, candidate_jobs)
        print(f"Possible allocations - {possible_allocations}")

        allocation_fitness = defaultdict(float) # each value intitialized with 0.0
        optim_atomic_bszs = copy.deepcopy(possible_allocations)

        if len(candidate_jobs):
            fair_num_gpus = self.get_fair_num_gpus()
            fair_allocation = (fair_num_gpus, 1) # TODO: account for cases when fair_num_gpus > #GPUs per node
            print(f"Fair allocation - {fair_allocation}")

            for i, allocation in enumerate(possible_allocations):
                fitness = float()
                print(f"Computing goodput for allocation {allocation}:")
                for job_id, alloc_config in allocation.items():
                    print(f"Goodput for job {job_id} with allocation config {alloc_config}:")
                    grad_params = (self.jobs[job_id].pollux_agent.grad_norm_sqr, self.jobs[job_id].pollux_agent.grad_variance)
                    goodput_fn = GoodputFunction(grad_params, self.jobs[job_id].pollux_agent.init_batch_size, self.jobs[job_id].pollux_agent)
                    suggest_goodput, atomic_bsz, accum_steps = goodput_fn.optimize(
                        alloc_config[1], alloc_config[0] * alloc_config[1], alloc_config, 
                        max_batch_size=self.jobs[job_id].pollux_agent.max_batch_size,
                        atomic_bsz_range=self.jobs[job_id].pollux_agent.local_bsz_bounds,
                        accumulation=False
                    )

                    optim_atomic_bszs[i][job_id] = atomic_bsz

                    fair_suggest_goodput, fair_atomic_bsz, fair_accum_steps = goodput_fn.optimize(
                        fair_allocation[1], fair_allocation[0] * fair_allocation[1], fair_allocation, 
                        max_batch_size=self.jobs[job_id].pollux_agent.max_batch_size,
                        atomic_bsz_range=self.jobs[job_id].pollux_agent.local_bsz_bounds,
                        accumulation=False
                    )

                    speedup = suggest_goodput / fair_suggest_goodput
                    num_jobs = len(self.allocation_matrix)
                    fitness_component = (1 / num_jobs) * speedup ** FAIRNESS_KNOB

                    print(f"Speedup - {speedup}, fitness_component - {fitness_component}")
                    
                    fitness += fitness_component

                    print(f"Goodput - {suggest_goodput}, local_bsz - {atomic_bsz}, accum_steps - {accum_steps}")
                    print(f"Fair Goodput - {fair_suggest_goodput}, fair local_bsz - {fair_atomic_bsz}, fair accum_steps - {fair_accum_steps}")

                fitness = fitness ** (1 / FAIRNESS_KNOB)
                allocation_fitness[i] = fitness
            
            max_fitness_allocation_index = max(allocation_fitness, key=allocation_fitness.get)
            print(f"Maximum fitness allocation index - {max_fitness_allocation_index}")
            print(f"Maximum fitness allocation - {possible_allocations[max_fitness_allocation_index]}")
            print(f"Config for best allocation - {optim_atomic_bszs[max_fitness_allocation_index]}")

            return possible_allocations[max_fitness_allocation_index], optim_atomic_bszs[max_fitness_allocation_index]

        return None, None

        # for job_id, job in self.candidate_jobs.items():
        #     print(f"Max BS: {job.pollux_agent.max_batch_size}")
        #     print(f"Init BS: {job.pollux_agent.init_batch_size}")
        #     print(f"local bsz bounds: {job.pollux_agent.local_bsz_bounds}")


    async def periodically_reallocate_resources(self):
        logger.info(f"Resource reallocation timer started!")
        await asyncio.sleep(30)
        while True and not self.realloc_requests_once:
            await asyncio.sleep(30)
            try:
                # # Using all jobs for now, allocating 2 GPUs each
                # job_ids_to_reallocate = [j for j in self.jobs.keys() if self.jobs[j].status == JobState.allocated]
                # allocations = (2, 1)

                # for job_id in job_ids_to_reallocate:
                #     self.job_ids_reallocating_resources[job_id] = allocations
                # #

                self.job_ids_reallocating_resources, optim_atomic_bszs = self.optimize_resource_allocation()


                if self.job_ids_reallocating_resources is not None:
                    for job_id in list(self.job_ids_reallocating_resources.keys()):
                        if self.job_ids_reallocating_resources[job_id] == self.get_current_job_allocation(job_id):
                            del self.job_ids_reallocating_resources[job_id]

                    if len(self.job_ids_reallocating_resources):
                        print(f"Doing the following reallocations: {self.job_ids_reallocating_resources}")
                        await self.send_reallocation_notices(self.job_ids_reallocating_resources)

                    while len(self.job_ids_reallocating_resources) > 0:
                        await asyncio.sleep(1)
                logger.info(f"All jobs have their resources reallocated.")
                # self.realloc_requests_once = True
            except Exception as e:
                print(repr(e))
        logger.info(f"Done reallocating resources")


    def get_current_job_allocation(self, job_id: str):
        if job_id not in self.allocation_matrix.keys():
            return None
        num_gpus_per_node = np.max(self.allocation_matrix[job_id]) # assuming that number of GPUs on each node is equal
        num_nodes = np.sum(self.allocation_matrix[job_id] == num_gpus_per_node)
        return (num_gpus_per_node, num_nodes)


    async def send_reallocation_notices(self, job_ids: List[str]): # input list of relevant jobs
        for job_id in job_ids:
            from main import send_message_to_client
            logger.info(f"Sending reallocation notice at {datetime.now().strftime('%H:%M:%S')} to job {job_id}")
            await send_message_to_client(job_id, "reallocation")

    
    async def reallocation_request_placement_group(self, job_id: str, name: str):
        if job_id in self.allocation_matrix.keys() or self.jobs[job_id].status is not JobState.reallocating:
            raise SchedulerError("This job is not supposed to request resource reallocation!")
        
        reallocating_resources = self.job_ids_reallocating_resources[job_id]
        self.job_ids_reallocating_resources.pop(job_id, None) # TODO: ensure that reference to 'reallocating_resources' is unaffected
        while len(self.job_ids_reallocating_resources) > 0:
            await asyncio.sleep(1)

        # Get used and free resources' vectors
        used_resources = np.zeros(self.all_host_num_devices.shape, dtype=int)
        
        for allocation_vector in list(self.allocation_matrix.values()):
            used_resources += allocation_vector

        free_resources = self.all_host_num_devices - used_resources
        
        # init_num_gpus = min(2, self.gpus_per_node)
        num_gpus, num_nodes = reallocating_resources[0], reallocating_resources[1]
        
        # allocated_node = None
        
        # for i, num_free_gpus in enumerate(free_resources):
        #     if num_free_gpus >= init_num_gpus:
        #         allocated_node = i
        
        # if allocated_node is None: # TODO: also check if such a placement group is available
        #     self.jobs_queue.append(job_id)
        #     self.jobs[job_id].status = JobState.queued
        #     logger.info(f"queued job {job_id}")
        #     return None # TODO: instead, implement a queueing mechanism, job should wait for a placement group
        # else:
        # allocation_vector = np.array([init_num_gpus if i == allocated_node else 0 for i in range(self.all_host_num_devices.shape[0])])
        # logger.info(f"allocation_vector - {allocation_vector}")
        # self.allocation_matrix[job_id] = allocation_vector
        host_num_devices = [num_gpus] * num_nodes
        # pg_name = RAY_CLUSTER_NAMESPACE + "_pg_" + job_id
        # self.jobs[job_id].status = JobState.allocated
        return await self.create_placement_group(num_nodes, host_num_devices, name, job_id)

        
        
orchestrator = Orchestrator()