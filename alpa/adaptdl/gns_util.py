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

def compute_gradsnorms(gradients, preconditioners):
    local_sqr_val = 0.0

    #for grad, preconditioner in zip(gradients, preconditioners):
    #    local_sqr_val += jnp.sum((grad/preconditioner)**2)

    for grad, preconditioner in zip(gradients, preconditioners):
        local_sqr_val += jnp.sum((grad/preconditioner - jnp.mean(grad/preconditioner))**2)
            
    grads_normsqr = normsqr_groups(gradients, preconditioners)
    return local_sqr_val, grads_normsqr

def running_gradient(running_grd, running_grd_sqr, grads_flat, itr, beta=0.9):
    """
    This function comptes the noise and scale for gradient noise scale
    grads_flat is the flattened gradients of shape (N,1) where N is the total number of params
    """
    # Update the gradient and squared gradient
    running_grd = (beta * running_grd + (1 - beta)*grads_flat) / (1 - beta**(itr + 1))
    running_grd_sqr = (beta * running_grd_sqr + (1 - beta)*jnp.square(grads_flat)) / (1 - beta**(itr + 1))
    # compute the sum of variances with the formula V[X] = E[X^2] - E[X]^2
    noise = jnp.sum(running_grd_sqr) - jnp.sum(jnp.square(running_grd))
    scale = jnp.sum(jnp.square(grads_flat))
    return running_grd, running_grd_sqr, noise, scale


# Welford's Algorithm to compute SE
def update_variance(current_variance, new_value, step):
    delta = new_value - current_variance['mean']
    current_variance['n'] = step
    current_variance['mean'] += delta / current_variance['n']
    delta2 = new_value - current_variance['mean']
    current_variance['M2'] += delta * delta2
    return current_variance
    #return delta

def compute_variance(variance_stats):
    #if variance_stats['n'] < 2:
    #    return 0  # Insufficient data for variance estimation
    return variance_stats['M2'] / (variance_stats['n'] - 1)

def init_dict(state):
    params_flat, tree_def = tree_flatten(state.params)
    #params_keys = [str(i) for i in range(len(params_flat))]
    params_keys = tuple(str(i) for i in range(len(params_flat)))
    params_dict = {key: param for key, param in zip(params_keys, params_flat)}

    variance_stats_flat = {name: {'n': 0, 'mean': 0, 'M2': 0} for name in params_keys}
    
    return params_keys, params_dict, variance_stats_flat

def compute_grad_flat_mean(grd):
    return jnp.mean(jnp.array(grd))


# Different Method
def flatten_gradients(grads):
    flat_gradients, _ = tree_flatten(jax.tree_map(lambda x: x.ravel(), grads))
    return jnp.concatenate(flat_gradients)

def running_gradient_2(running_grd, grad, itr, beta=0.9):
    return (beta * running_grd + (1 - beta)*grad) / (1 - beta**(itr + 1))


def init_running_gradients(grads):
    flat_grds = flatten_gradients(grads)
    running_grd = jnp.zeros_like(flat_grds)
    running_grd_sqr = jnp.zeros_like(flat_grds)
    return running_grd, running_grd_sqr
    


