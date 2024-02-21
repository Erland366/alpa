from alpa.adaptdl.pollux_agent import pollux_agent
from alpa.adaptdl.scaling_rules import ScalingRuleBase
import alpa
from jax.tree_util import tree_flatten, tree_unflatten, PyTreeDef
from typing import Callable, Optional

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
    
    state = state.replace(step=step_val, opt_state=unflattened_opt_state, params=unflattened_params)
    
    alpa.shutdown(is_reallocation=True)
    alpa.clear_executable_cache()
    alpa.init(cluster='ray', scheduler_address=pollux_agent.scheduler_address, is_reallocation=True)

    pollux_agent.reallocation_approaching = False
    
    return state