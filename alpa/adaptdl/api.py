from alpa.adaptdl.pollux_agent import pollux_agent
from alpa.adaptdl.scaling_rules import ScalingRuleBase
import alpa
from jax.tree_util import tree_flatten, tree_unflatten, PyTreeDef
from typing import Callable, Optional
from alpa.model.model_util import DynamicScale, TrainState
from addict import Dict as AddictDict
import numpy as np
from alpa.adaptdl.scaling_rules import ScalingRuleBase, LinearScale, SqrtScale
import sys
import logging
import yaml
import os
import datetime


logger = logging.getLogger(__name__)

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
    dynp_save_path = os.path.join(yml_config.dynp_profiling.save_dir, f"dynp_results_{nodes}nodes_{devices_per_node}gpus_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.yml")
    with open(dynp_save_path, 'w') as f:
        yaml.dump(dynp_dictionary, f, default_flow_style=False)
    logger.info(f"Saved DynP results to {dynp_save_path}")
    alpa.shutdown()
    sys.exit(1)