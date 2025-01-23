import numpy as np
from itertools import product
from typing import Union, Optional, List, Tuple, Set, Dict


def generate_power_of_two_combinations(max_gpus):
    return [2**i for i in range(int(np.log2(max_gpus))+1)]


def generate_allocations_for_job(num_nodes, gpus_per_node):
    allocations = []

    for gpus in generate_power_of_two_combinations(gpus_per_node):
        allocations.append((gpus, 1))

    if num_nodes > 1:
        for nodes_used in range(2, num_nodes+1):

            allocations.append((gpus_per_node, nodes_used))

    return allocations

def is_allocation_valid(combination, num_nodes, gpus_per_node):
    node_usage = [0] * num_nodes
    
    for alloc in combination:
        gpus, nodes = alloc
        if nodes == 1:

            allocated = False
            for i in range(num_nodes):
                if node_usage[i] + gpus <= gpus_per_node:
                    node_usage[i] += gpus
                    allocated = True
                    break
            if not allocated:
                return False
        else:

            if sum(node_usage) + gpus * nodes > gpus_per_node * num_nodes:
                return False

            for i in range(nodes):
                node_usage[i] += gpus
                
    return True

def list_possible_allocations(cluster_config, jobs) -> List[Dict]:
    num_nodes = len(cluster_config)
    gpus_per_node = cluster_config[0]

    job_ids = list(jobs.keys())
    all_job_allocations = [list(generate_allocations_for_job(num_nodes, gpus_per_node)) for _ in job_ids]

    valid_configurations = []
    for combination in product(*all_job_allocations):
        if is_allocation_valid(combination, num_nodes, gpus_per_node):

            config_dict = dict(zip(job_ids, combination))
            valid_configurations.append(config_dict)

    return valid_configurations