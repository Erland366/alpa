from alpa.adaptdl.pollux_agent import pollux_agent
from alpa.adaptdl.scaling_rules import ScalingRuleBase
import alpa
from jax.tree_util import tree_flatten, tree_unflatten, PyTreeDef
from typing import Callable, Optional, List, Dict, Any, Union
from alpa.model.model_util import DynamicScale, TrainState
from addict import Dict as AddictDict
import numpy as np
from alpa.adaptdl.scaling_rules import ScalingRuleBase, LinearScale, SqrtScale
import jax
import sys
import logging
import yaml
import os
import datetime
import time


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)

def update_state_on_bs_change(state):
    if pollux_agent.last_state_retrieved_batch_size == pollux_agent.total_batch_size:
        # TODO: also, check if method is PipeShard + auto specifically
        return state
    
    step_val = state.step._value
    
    flattened_opt_state = tree_flatten(state.opt_state)
    for i, leaf in enumerate(flattened_opt_state[0]):
        if isinstance(leaf, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray)):
            flattened_opt_state[0][i] = leaf._value
    unflattened_opt_state = tree_unflatten(flattened_opt_state[1], flattened_opt_state[0])
    
    flattened_params = tree_flatten(state.params)
    for i, leaf in enumerate(flattened_params[0]):
        if isinstance(leaf, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray)):
            flattened_params[0][i] = leaf._value
    unflattened_params = tree_unflatten(flattened_params[1], flattened_params[0])
    
    state = state.replace(step=step_val, opt_state=unflattened_opt_state, params=unflattened_params)
    
    alpa.shutdown()
    alpa.clear_executable_cache()
    alpa.init(cluster='ray')
    
    pollux_agent.last_state_retrieved_batch_size = pollux_agent.total_batch_size
    
    return state


def scale_lr(current_batch_size: int, initial_batch_size: int, base_lr: float) -> float:
    """Plug-in interface for linear learning rate scaling based on batch size."""
    return base_lr * (current_batch_size / initial_batch_size)


def create_scaled_lr_fn(original_lr_fn, initial_batch_size: int, scaling_rule: ScalingRuleBase):
    """Returns a new learning rate function based on the current batch size."""
    def scaled_lr_fn(step: int) -> float:
        current_batch_size = pollux_agent.total_batch_size
        base_lr = original_lr_fn(step)
        scale = current_batch_size / initial_batch_size
        return scaling_rule.scale_lr(scale) * base_lr
    
    return scaled_lr_fn


def reallocate_and_update_state(state):
    if not pollux_agent.reallocation_approaching or not pollux_agent.scheduler_enabled:
        # TODO: also, check if method is PipeShard + auto specifically
        return state
    
    step_val = state.step._value
    
    flattened_opt_state = tree_flatten(state.opt_state)
    for i, leaf in enumerate(flattened_opt_state[0]):
        if isinstance(leaf, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray)):
            flattened_opt_state[0][i] = leaf._value
    unflattened_opt_state = tree_unflatten(flattened_opt_state[1], flattened_opt_state[0])
    
    flattened_params = tree_flatten(state.params)
    for i, leaf in enumerate(flattened_params[0]):
        if isinstance(leaf, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray)):
            flattened_params[0][i] = leaf._value
    unflattened_params = tree_unflatten(flattened_params[1], flattened_params[0])

    replacement_attributes = {'step': step_val, 'opt_state': unflattened_opt_state, 'params': unflattened_params}

    if getattr(state, 'master_copy', None) is not None:
        flattened_master_copy = tree_flatten(state.master_copy)
        for i, leaf in enumerate(flattened_master_copy[0]):
            if isinstance(leaf, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray)):
                flattened_master_copy[0][i] = leaf._value
        unflattened_master_copy = tree_unflatten(flattened_master_copy[1], flattened_master_copy[0])
        replacement_attributes['master_copy'] = unflattened_master_copy

    if getattr(state, 'dynamic_scale', None) is not None:
        fin_steps = state.dynamic_scale.fin_steps
        scale = state.dynamic_scale.scale
        if isinstance(fin_steps, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray)):
            fin_steps = fin_steps._value
        if isinstance(scale, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray)):
            scale = scale._value
        dynamic_scale_materialized = DynamicScale(fin_steps=fin_steps, scale=scale)
        replacement_attributes['dynamic_scale'] = dynamic_scale_materialized

    state = state.replace(**replacement_attributes) 
    
    alpa.shutdown(is_reallocation=True)
    alpa.clear_executable_cache()
    alpa.init(cluster='ray', scheduler_address=pollux_agent.scheduler_address, is_reallocation=True)

    pollux_agent.reallocation_approaching = False
    pollux_agent.update_dataloader_batchsize = True
    
    return state

def fix_regressors(yml_config: AddictDict):
    """
    Set regression coefficients if specified in the config
    """
    if yml_config.pollux_agent.fix_regressors:
        for item in yml_config.pollux_agent.regression_coefficients:
            key = tuple(item.key)  # Convert list in .yml to tuple
            values = item
            pollux_agent.alloc_config_regressor[key].coef_ = np.array([values.coef])
            pollux_agent.alloc_config_regressor[key].intercept_ = values.intercept
        pollux_agent.fix_regressors()

def get_scaled_learning_rate_fn(yml_config: AddictDict, original_learning_rate_fn):
    if not yml_config.training.scale_lr.enabled:
        return original_learning_rate_fn
    if yml_config.training.scale_lr.type == 'sqrt':
        scaling_rule = SqrtScale()
    else:
        scaling_rule = LinearScale()
    
    # TODO: initial batch size should probably be stored separately to avoid newer batch size being set after a checkpoint-restart
    scaled_learning_rate_fn = create_scaled_lr_fn(original_lr_fn=original_learning_rate_fn, initial_batch_size=pollux_agent.total_batch_size,
                                                             scaling_rule=scaling_rule)

    return scaled_learning_rate_fn

def get_parallel_method(yml_config: AddictDict):
    if yml_config.training.parallel_method.method == 'ShardParallel':
        method = alpa.ShardParallel(num_micro_batches=yml_config.training.parallel_method.num_micro_batches if yml_config.training.parallel_method.num_micro_batches != 1 else None)
    elif yml_config.training.parallel_method.method == 'PipeshardParallel':
        stage_option = yml_config.training.parallel_method.parameters.PipeshardParallel.stage_option
        method = alpa.PipeshardParallel(stage_option=stage_option, num_micro_batches=yml_config.training.parallel_method.num_micro_batches)
    elif yml_config.training.parallel_method.method == 'DataParallel':
        method = alpa.DataParallel(num_micro_batches=yml_config.training.parallel_method.num_micro_batches if yml_config.training.parallel_method.num_micro_batches != 1 else None)
    elif yml_config.training.parallel_method.method == '3D':
        method = alpa.get_3d_parallel_method(num_micro_batches=yml_config.training.parallel_method.num_micro_batches,
                                             data_parallel=yml_config.training.parallel_method.parameters._3D.data_parallel,
                                             operator_parallel=yml_config.training.parallel_method.parameters._3D.operator_parallel,
                                             pipeline_parallel=yml_config.training.parallel_method.parameters._3D.pipeline_parallel)
    elif yml_config.training.parallel_method.method == 'DynP':
        dynp_strategies_dict = load_yaml_full_loader(yml_config, yml_config.training.parallel_method.parameters.DynP.dynp_manualstages_yml_path)
        global_cluster = alpa.get_global_cluster()
        host_num_devices = global_cluster.host_num_devices
        devices_per_node, nodes = host_num_devices[0], len(host_num_devices)
        manual_stage_params_dict = find_config(dynp_strategies_dict, nodes, devices_per_node)
        manual_stage_option = alpa.ManualStageOption(
            forward_stage_layer_ids=manual_stage_params_dict["forward_stage_layer_ids"],
            submesh_physical_shapes=manual_stage_params_dict["submesh_physical_shapes"],
            submesh_logical_shapes=manual_stage_params_dict["submesh_logical_shapes"],
            submesh_autosharding_option_dicts=manual_stage_params_dict["submesh_autosharding_option_dicts"],
        )
        method = alpa.PipeshardParallel(stage_option=manual_stage_option, num_micro_batches=yml_config.training.parallel_method.num_micro_batches)
        # TODO: degenerate to 3D if list forward_stage_layer_ids has one element
    elif yml_config.training.parallel_method.method == 'DynPsingle':
        manual_stage_params_dict = load_yaml_full_loader(yml_config, yml_config.training.parallel_method.parameters.DynPsingle.manualstage_yml_path)
        global_cluster = alpa.get_global_cluster()
        host_num_devices = global_cluster.host_num_devices
        devices_per_node, nodes = host_num_devices[0], len(host_num_devices)
        if (nodes, devices_per_node) != (manual_stage_params_dict["nodes"], manual_stage_params_dict["devices_per_node"]):
            raise Exception(f"Number of nodes/devices in DynP YAML ({manual_stage_params_dict['nodes']}, {manual_stage_params_dict['devices_per_node']}) does not match current cluster ({nodes}, {devices_per_node})")
        manual_stage_option = alpa.ManualStageOption(
            forward_stage_layer_ids=manual_stage_params_dict["forward_stage_layer_ids"],
            submesh_physical_shapes=manual_stage_params_dict["submesh_physical_shapes"],
            submesh_logical_shapes=manual_stage_params_dict["submesh_logical_shapes"],
            submesh_autosharding_option_dicts=manual_stage_params_dict["submesh_autosharding_option_dicts"],
        )
        method = alpa.PipeshardParallel(stage_option=manual_stage_option, num_micro_batches=yml_config.training.parallel_method.num_micro_batches)
    else:
        method = alpa.DataParallel()
    
    return method

def do_reallocation(yml_config: AddictDict, p_train_step, variables_dict: dict, gns, state):
    p_train_step.get_last_executable().sync()

    materialized_variables_dict = {}
    for k, v in variables_dict.items():
        if isinstance(v, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray)):
            materialized_variables_dict[k] = v._value
        elif isinstance(v, list):
            materialized_list = []
            for el in v:
                if isinstance(el, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray)):
                    materialized_list.append(el._value)
                else:
                    materialized_list.append(el)
            materialized_variables_dict[k] = materialized_list
        else:
            materialized_variables_dict[k] = v

    if isinstance(pollux_agent.grad_norm_sqr_abstract, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray)) \
            and isinstance(pollux_agent.grad_variance_abstract, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray)):
        pollux_agent.grad_norm_sqr_abstract = pollux_agent.grad_norm_sqr = pollux_agent.grad_norm_sqr_abstract._value.item()
        pollux_agent.grad_variance_abstract = pollux_agent.grad_variance = pollux_agent.grad_variance_abstract._value.item()

    if isinstance(gns.store_grads, list):
        store_grads_materialized = []
        for el in gns.store_grads:
            if isinstance(el, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray)):
                store_grads_materialized.append(el._value)
            else:
                store_grads_materialized.append(el)
        gns.store_grads = store_grads_materialized
    if isinstance(gns.biased_sqr, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray)):
        gns.biased_sqr = gns.biased_sqr._value
    if isinstance(gns.unbias_sqr, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray)):
        gns.unbias_sqr = gns.unbias_sqr._value
    if isinstance(gns.biased_var, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray)):
        gns.biased_var = gns.biased_var._value
    if isinstance(gns.unbias_var, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray)):
        gns.unbias_var = gns.unbias_var._value

    state = reallocate_and_update_state(state)

    # TODO: no need to return materialized_variables_dict if main training loop has `continue` right after this function, because
    # variables_dict is rebuilt at the beginning of every iteration (rethink this?)

    # TODO: when changing how GNS is accessed/computed, do not forget to do the changes here too
    return state, materialized_variables_dict

def dynp_profiling(yml_config: AddictDict):
    if not (yml_config.training.parallel_method.method == "PipeshardParallel" and yml_config.training.parallel_method.parameters.PipeshardParallel.stage_option == "auto"):
        alpa.shutdown()
        raise Exception("DynP profiling is only available for PipeshardParallel with auto stage option")
    if yml_config.profiling.enabled:
        alpa.shutdown()
        raise Exception("Throughput profiling should be DISABLED to collect DynP profiling results")
    if yml_config.training.gns_enabled:
        alpa.shutdown()
        raise Exception("GNS should be disabled to collect DynP profiling results")
    dynp_results = alpa.get_last_dp_result()
    logger.info(f"Retrieved best DynP results: {dynp_results}")
    global_cluster = alpa.get_global_cluster()
    host_num_devices = global_cluster.host_num_devices
    devices_per_node, nodes = host_num_devices[0], len(host_num_devices)
    dynp_dictionary = {
        "devices_per_node": devices_per_node,
        "nodes": nodes,
        "forward_stage_layer_ids": dynp_results[1],
        "submesh_physical_shapes": dynp_results[2],
        "submesh_logical_shapes": dynp_results[3],
        "submesh_autosharding_option_dicts": dynp_results[4],
    }
    os.makedirs(yml_config.dynp_profiling.save_dir, exist_ok=True)
    filename = (
        f"dynp_results_{nodes}"
        f"nodes_{devices_per_node}"
        f"gpus_{yml_config.dataloader.train.init_local_batch_size}"
        f"localbsz_{yml_config.training.parallel_method.num_micro_batches}"
        f"microbatches_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.yml"
    )
    dynp_save_path = os.path.join(yml_config.dynp_profiling.save_dir, filename)
    with open(dynp_save_path, 'w') as f:
        # TODO: better format yaml
        yaml.dump(dynp_dictionary, f, default_flow_style=False)
    logger.info(f"Saved DynP results to {dynp_save_path}")
    alpa.shutdown()
    sys.exit(1)

def load_yaml_full_loader(yml_config: AddictDict, path):
    """
    Load a DynP profiling YAML (dumped with python/tuple tags) and return
    an AddictDict with exactly the same nested types dumped previously:
      - forward_stage_layer_ids as List[List[int]]
      - submesh_physical_shapes as List[Tuple[int, ...]]
      - submesh_logical_shapes as List[Tuple[int, ...]]
      - submesh_autosharding_option_dicts as List[Dict[str, Any]]
    """
    with open(path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    return data

def find_config(configs: List[Dict[str, Any]], nodes: int, devices_per_node: int) -> Dict[str, Any]:
    """Find a specific configuration by nodes and devices_per_node."""
    for item in configs:
        if item['nodes'] == nodes and item['devices_per_node'] == devices_per_node:
            return item['config']
    
    raise ValueError(f"No configuration found for nodes={nodes}, devices_per_node={devices_per_node}")

def get_profiling_setup(
    profiling_enabled: bool, 
    profiling_config: Dict[str, Union[int, bool, Dict]],
    yml_config: Dict[str, str], 
    training_args
):
    # Profiling config
    num_micro_batches = training_args.num_micro_batches
    num_devices = alpa.get_global_num_devices()

    batch_sizes_to_run = []
    if profiling_enabled:
        # Disable wandb
        os.environ["WANDB_MODE"] = "offline"

        # Disable preallocation of JAX
        # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        # os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

        # Check if we should use the custom list
        use_custom_list = profiling_config.get("custom_list", {}).get("use_list", False)
        
        if use_custom_list:
            # Use the provided list of local batch sizes
            local_batch_size_list = profiling_config.get("custom_list", {}).get("local_batch_size_list", [])
            max_batch_size = profiling_config.get("max_batch_size", None)
            
            logger.info(f"Using custom local batch size list: {local_batch_size_list}")
            logger.info(f"Constraints: Divisible by num_devices ({num_devices}), divisible by num_micro_batches ({num_micro_batches}), max_total_bs ({max_batch_size})")
            
            for local_bs in local_batch_size_list:
                total_bs = local_bs * num_devices
                
                # 1. Check max batch size limit
                if max_batch_size is not None and total_bs > (max_batch_size * num_devices):
                    logger.info(f"Total batch size {total_bs} from local batch size {local_bs} exceeds max {max_batch_size * num_devices}. Skipping.")
                    continue
                
                # 2. Check divisibility by number of devices
                if total_bs % num_devices != 0:
                    logger.debug(f"Skipping batch size {total_bs}: Not divisible by num_devices ({num_devices})")
                    continue
                
                # 3. Check divisibility by num_micro_batches
                if num_micro_batches > 0 and total_bs % num_micro_batches != 0:
                    logger.debug(f"Skipping batch size {total_bs}: Not divisible by num_micro_batches ({num_micro_batches})")
                    continue
                elif num_micro_batches <= 0:
                    logger.warning("num_micro_batches is <= 0. Skipping divisibility check.")
                
                # If all checks pass, add it to the list
                logger.info(f"Found valid profile total batch size: {total_bs}, local batch size: {local_bs}")
                batch_sizes_to_run.append(local_bs)
        else:
            # Use the original power-of-2 search approach
            # batch sizes
            min_batch_size = profiling_config.get("min_batch_size", 1)
            max_batch_size = profiling_config.get("max_batch_size", None)

            if min_batch_size <= 0:
                logger.warning("Minimum batch size should be greater than 0. Setting it to 1.")
                min_batch_size = 1

            total_bs = 1
            while total_bs < (min_batch_size * num_devices):
                total_bs *= 2

            logger.info(f"Starting search for profile batch sizes from {total_bs}")
            logger.info(f"Constraints: Divisible by num_devices ({num_devices}), divisible by num_micro_batches ({num_micro_batches}), max_total_bs ({max_batch_size})")

            while True:
                # 1. Check max batch size limit
                if max_batch_size is not None and total_bs > (max_batch_size * num_devices):
                    logger.info(f"Current batch size {total_bs} exceeds max {max_batch_size * num_devices}. Stopping search.")
                    break

                # 2. Check divisibility by number of devices
                if total_bs % num_devices != 0:
                    logger.debug(f"Skipping batch size {total_bs}: Not divisible by num_devices ({num_devices})")
                    total_bs *= 2
                    continue

                # 3. Check divisibility by num_micro_batches
                # The total batch size per step must be divisible by num_micro_batches
                # for gradient accumulation logic.
                if num_micro_batches > 0 and total_bs % num_micro_batches != 0:
                     logger.debug(f"Skipping batch size {total_bs}: Not divisible by num_micro_batches ({num_micro_batches})")
                     total_bs *= 2
                     continue
                elif num_micro_batches <= 0:
                     logger.warning("num_micro_batches is <= 0. Skipping divisibility check.")

                # If all checks pass, add it to the list
                logger.info(f"Found valid profile batch size: {total_bs}")
                batch_sizes_to_run.append(int(total_bs / num_devices))

                # Move to the next power of 2
                total_bs *= 2

                # Safety break for extremely large numbers if max_total_bs is None
                if total_bs > 2 ** 20: # Arbitrary large limit (~1 million)
                    logger.warning("Reached very large batch size during profiling search without max_total_bs. Stopping.")
                    break

        if not batch_sizes_to_run:
            logger.error("No valid batch sizes found for profiling based on the constraints!")
            return # Exit if no batch sizes are viable
    else:
        initial_local_batch_size = yml_config.dataloader.train.init_local_batch_size
        initial_total_batch_size = initial_local_batch_size * num_devices
        batch_sizes_to_run = [initial_total_batch_size]
        logger.info(f"Using initial batch size: {initial_total_batch_size} for profiling.")

    return batch_sizes_to_run

def run_profile(
    p_train_step: Callable,
    state: Any,
    batch: Any,
    variables_dict: Dict[str, Any],
    # CSV
    i_run: int,
    epoch: int,
    current_local_batch_size: int,
    yml_config: Dict[str, str],
    rng=None
):
    print("Running profiling...")

    # warmup
    compil_time = None
    compil_start = time.time()
    for _ in range(yml_config.profiling.warmup_steps):
        if rng:
            dropout_rng, rng = jax.random.split(rng)
            state, train_metric = p_train_step(state, batch, dropout_rng, variables_dict)
        else:
            state, train_metric = p_train_step(state, batch, variables_dict)
        if compil_time is None:
            p_train_step.get_last_executable().sync()
            compil_time = time.time() - compil_start
    p_train_step.get_last_executable().sync()
    avg_cost = []
    time_start = time.time()
    for _ in range(yml_config.profiling.profile_steps):
        if rng:
            dropout_rng, rng = jax.random.split(rng)
            state, train_metric = p_train_step(state, batch, dropout_rng, variables_dict)
        else:
            state, train_metric = p_train_step(state, batch, variables_dict)
    p_train_step.get_last_executable().sync()
    avg_cost = (time.time() - time_start) / yml_config.profiling.profile_steps
    avg_cost = np.mean(np.array(avg_cost))
    if isinstance(p_train_step.method, alpa.PipeshardParallel):
        mem_gb = p_train_step.get_last_executable().get_stage_allocation_size()
        mem_gb = sum(mem_gb) / (1024**3)
    else:
        mem_gb = p_train_step.get_last_executable().get_total_allocation_size() / (1024**3)
    logger.info(f"Average cost: {avg_cost}")
    logger.info(f"Memory usage: {mem_gb} GB")

    global_cluster = alpa.get_global_cluster()
    host_num_devices = global_cluster.host_num_devices
    devices_per_node, nodes = host_num_devices[0], len(host_num_devices)
    filename = (
        f"csv_results_{nodes}"
        f"nodes_{devices_per_node}"
        f"gpus_{yml_config.training.parallel_method.num_micro_batches}"
        f"microbatches.csv"
    )
    os.makedirs(yml_config.profiling.csv_dir, exist_ok=True)
    csv_save_path = os.path.join(yml_config.profiling.csv_dir, filename)

    if jax.process_index() == 0:
        with open(csv_save_path, "a") as f:
            if os.stat(csv_save_path).st_size == 0:
                f.write("run,epoch,batch_size,avg_cost,mem_gb,compil_time,model_name,strategy,dp,pp,tp\n")
            f.write(f"{i_run},{epoch},{current_local_batch_size},{avg_cost},{mem_gb},{compil_time},"
                    f"{yml_config.model_name_or_path},{yml_config.training.parallel_method.method},"
                    f"{yml_config.training.parallel_method.parameters._3D.data_parallel},{yml_config.training.parallel_method.parameters._3D.operator_parallel},"
                    f"{yml_config.training.parallel_method.parameters._3D.operator_parallel}\n")

    if avg_cost == float("inf"):
        sys.exit(1)

    for _ in range(10):
        import gc
        gc.collect()

    return state

def execute_profiling_trials(batch_sizes_to_run, p_train_step, state, batch, variables_dict, epoch, yml_config, rng=None):
    current_local_batch_size = batch_sizes_to_run.pop(0)
    for i_run in range(yml_config.profiling.get("repeat_profile_steps", 1)):
        state = run_profile(
            p_train_step=p_train_step,
            state=state,
            batch=batch,
            variables_dict=variables_dict,
            i_run=i_run,
            epoch=epoch,
            current_local_batch_size=current_local_batch_size,
            yml_config=yml_config,
            rng=rng
        )
    
    p_train_step.get_last_executable().sync()
    pollux_agent.update_dataloader_batchsize = True
    if len(batch_sizes_to_run) == 0:
        print(f"Finished profiling.")
        alpa.shutdown()
        sys.exit(1)
    pollux_agent.force_dataloader_localbatchsize = batch_sizes_to_run[0]
    return state