import jax
import jax.numpy as jnp
import time
import alpa

from jax.tree_util import tree_flatten, tree_unflatten
from jax.lib import xla_bridge

def extract_values_with_key_p(structure):
    flat_structure, tree_def = tree_flatten(structure)
    #flattened_gradients = [node._value for node in flat_structure]
    flattened_gradients = [node for node in flat_structure]
    #jax_tree = jax.tree_util.tree_map(lambda x: x if isinstance(x, jnp.ndarray) else jnp.asarray(x), flattened_gradients)
    #print(f'flattened_gradients: {flattened_gradients}')
    return (flattened_gradients)

def normsqr_groups_2(grads, pinvs):
    def inner_norm(group, pinv_group):
        return jnp.sum(jnp.sum((group / pinv_group) ** 2, axis=-1))

    normsqr_list = [
        inner_norm(group, pinv_group) 
        for group, pinv_group in zip(grads, pinvs)
        ]
    
    return jnp.sum(jnp.array(normsqr_list))

def normsqr_groups(grads, pinvs):
    normsqr_list = [jnp.sum(jnp.square(g/p)) for g, p in zip(grads, pinvs)]
    return jnp.sum(jnp.array(normsqr_list))

def compute_gradsnorms(gradients, preconditioners, num_replicas, accum_scale, accum_count=False):
    local_sqr_val = 0.0

    #for grad, preconditioner in zip(gradients, preconditioners):
    #    local_sqr_val += jnp.sum((grad/preconditioner)**2)

    for grad, preconditioner in zip(gradients, preconditioners):
        local_sqr_val += jnp.sum((grad/preconditioner - jnp.mean(grad/preconditioner))**2)
            
    grads_normsqr = normsqr_groups(gradients, preconditioners)
    return local_sqr_val, grads_normsqr




