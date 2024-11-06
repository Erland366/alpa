import jax
import jax.numpy as jnp
import numpy as np
import time
import alpa

from jax.tree_util import tree_flatten, tree_unflatten
from jax.lib import xla_bridge
from alpa.device_mesh import get_global_physical_mesh
from jax.core import ShapedArray
from jax.interpreters import pxla


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


def compute_gradient_noise_scale(prev_grads, new_grads,
                                  preconditioner, 
                                  biased_sqr, unbias_sqr, biased_var, unbias_var,
                                  count, scale, theta
                                 ):
        
    grads_normsqr = normsqr_groups(new_grads, preconditioner)
    local_sqr = (normsqr_groups(prev_grads, preconditioner)
                             + grads_normsqr) / 2
    avg_grads = average_groups(new_grads, prev_grads)
    total_sqr = normsqr_groups(avg_grads, preconditioner)
    grad_sqr = (count * total_sqr - local_sqr) / (count - 1) 
    grad_var = (local_sqr - total_sqr) * scale / (count - 1)
    biased_sqr, unbias_sqr, grad_sqr = update_avg(grad_sqr, theta, biased_sqr, unbias_sqr)
    biased_var, unbias_var, grad_var = update_avg(grad_var, theta, biased_var, unbias_var)
    return grad_sqr, grad_var, biased_sqr, unbias_sqr, biased_var, unbias_var


# wrong but does not require storing previous iteration's gradients
def compute_gradient_noise_scale_running(grads, prev_normsqr, preconditioner, biased_sqr, unbias_sqr, 
                                        biased_var, unbias_var, count, scale, theta):
    # grads_normsqr = normsqr_groups(grads, preconditioner)
    flat_grads = jax.tree_util.tree_leaves(grads)
    flat_grads = [jnp.ravel(g) for g in flat_grads]
    flat_grads = jnp.concatenate(flat_grads)
    
    grads_normsqr = jnp.sum(flat_grads**2)
    local_sqr = (prev_normsqr + grads_normsqr) / 2
    # total_sqr = (prev_normsqr + grads_normsqr) / 2
    total_sqr = 1/4 * prev_normsqr + 3/4 * grads_normsqr
    # grad_sqr = (count * total_sqr - local_sqr) / (count - 1) 
    # grad_var = (local_sqr - total_sqr) * scale / (count - 1)
    grad_sqr = count * total_sqr - local_sqr
    grad_var = local_sqr - total_sqr
    # biased_sqr, unbias_sqr, grad_sqr = update_avg(grad_sqr, theta, biased_sqr, unbias_sqr)
    # biased_var, unbias_var, grad_var = update_avg(grad_var, theta, biased_var, unbias_var)
    return grads_normsqr, grad_sqr, grad_var, biased_sqr, unbias_sqr, biased_var, unbias_var


def compute_gradient_noise_scale_nowarning_OOM(prev_grads, new_grads,
                                 preconditioner, 
                                 biased_sqr, unbias_sqr, biased_var, unbias_var,
                                 count, scale, theta):
    
    def flatten_and_concat(nested):
        flat = jax.tree_util.tree_leaves(jax.tree_map(jnp.ravel, nested))
        return jnp.concatenate(flat)
    
    def normsqr_groups(grads, pinvs):
        flat_grads = flatten_and_concat(grads)
        flat_pinvs = flatten_and_concat(pinvs)
        return jnp.sum(jnp.square(flat_grads / flat_pinvs))
    
    def average_groups(grads1, grads2):
        flat_grads1 = flatten_and_concat(grads1)
        flat_grads2 = flatten_and_concat(grads2)
        return (flat_grads1 + flat_grads2) / 2
    
    grads_normsqr = normsqr_groups(new_grads, preconditioner)
    local_sqr = (normsqr_groups(prev_grads, preconditioner) + grads_normsqr) / 2
    
    avg_grads = average_groups(new_grads, prev_grads)
    total_sqr = jnp.sum(jnp.square(avg_grads / flatten_and_concat(preconditioner)))
    
    grad_sqr = (count * total_sqr - local_sqr) / (count - 1) 
    grad_var = (local_sqr - total_sqr) * scale / (count - 1)
    
    biased_sqr, unbias_sqr, grad_sqr = update_avg(grad_sqr, theta, biased_sqr, unbias_sqr)
    biased_var, unbias_var, grad_var = update_avg(grad_var, theta, biased_var, unbias_var)
    
    return grad_sqr, grad_var, biased_sqr, unbias_sqr, biased_var, unbias_var

def simple_gns_estimate(grads, prev_norm_sq, prev_var, beta=0.9):
    flat_grads = jax.tree_util.tree_leaves(grads)
    flat_grads = [jnp.ravel(g) for g in flat_grads]
    flat_grads = jnp.concatenate(flat_grads)
    
    norm_sq = jnp.sum(flat_grads**2)
    mean = jnp.mean(flat_grads)
    var = jnp.mean((flat_grads - mean)**2)
    
    norm_sq = beta * prev_norm_sq + (1 - beta) * norm_sq
    var = beta * prev_var + (1 - beta) * var
    var *= 1e9 # var seems to be out of norm_sq's scale by a large margin (1e9 for OPT)
    
    return norm_sq, var

# wrong
def improved_gns_estimate(grads, prev_mean_grad, prev_norm_sq, prev_var, beta=0.9, epsilon=1e-8):
    flat_grads = jax.tree_util.tree_leaves(grads)
    flat_grads = [jnp.ravel(g) for g in flat_grads]
    flat_grads = jnp.concatenate(flat_grads)
    
    # Compute current statistics
    norm_sq = jnp.sum(flat_grads**2)
    mean_grad = jnp.mean(flat_grads)
    
    # Estimate the "signal" component
    signal = jnp.sum((mean_grad - prev_mean_grad)**2)
    
    # Estimate the "noise" component
    noise = jnp.mean((flat_grads - mean_grad)**2)
    
    # Update running averages
    norm_sq = beta * prev_norm_sq + (1 - beta) * norm_sq
    var = beta * prev_var + (1 - beta) * noise
    mean_grad_new = beta * prev_mean_grad + (1 - beta) * mean_grad
    
    # Adjust norm_sq and var to better reflect signal and noise
    adjusted_norm_sq = norm_sq * (signal / (signal + noise + epsilon))
    adjusted_var = var * (noise / (signal + noise + epsilon))
    
    return adjusted_norm_sq, adjusted_var, mean_grad_new


def compute_gradient_noise_scale_no_ewma(prev_grads, new_grads,
                                  preconditioner, 
                                  biased_sqr, unbias_sqr, biased_var, unbias_var,
                                  count, scale, theta
                                 ):
        
    grads_normsqr = normsqr_groups(new_grads, preconditioner)
    local_sqr = (normsqr_groups(prev_grads, preconditioner)
                             + grads_normsqr) / 2
    avg_grads = average_groups(new_grads, prev_grads)
    total_sqr = normsqr_groups(avg_grads, preconditioner)
    grad_sqr = (count * total_sqr - local_sqr) / (count - 1) 
    grad_var = (local_sqr - total_sqr) * scale / (count - 1)
    # biased_sqr, unbias_sqr, grad_sqr = update_avg(grad_sqr, theta, biased_sqr, unbias_sqr)
    # biased_var, unbias_var, grad_var = update_avg(grad_var, theta, biased_var, unbias_var)
    return grad_sqr, grad_var, biased_sqr, unbias_sqr, biased_var, unbias_var

def compute_gradsnorms(gradients, preconditioners):
    local_sqr_val = 0.0

    #for grad, preconditioner in zip(gradients, preconditioners):
    #    local_sqr_val += jnp.sum((grad/preconditioner)**2)

    for grad, preconditioner in zip(gradients, preconditioners):
        local_sqr_val += jnp.sum((grad/preconditioner - jnp.mean(grad/preconditioner))**2)
            
    grads_normsqr = normsqr_groups(gradients, preconditioners)
    return local_sqr_val, grads_normsqr

def average_groups(grads1, grads2):
    ret = []
    for group1, group2 in zip(grads1, grads2):
        ret.append((group1 + group2) / 2)
    return ret

# def average_groups(grads1, grads2):
    # return jax.tree_map(lambda g1, g2: (g1 + g2) / 2, grads1, grads2)

# def update_avg(value, factor, biased, unbias):
#     # epsilon = 1e-8
#     biased = factor * biased + (1.0 - factor) * value
#     unbias = factor * unbias + (1.0 - factor)

#     # value = biased / (unbias + epsilon)
#     value = biased / unbias
    
#     return biased, unbias, value

def update_avg(value, factor, biased, unbias):
    new_biased = factor * biased + (1.0 - factor) * value
    new_unbias = factor * unbias + (1.0 - factor)
    
    biased = jnp.where(jnp.isnan(new_biased), biased, new_biased)
    unbias = jnp.where(jnp.isnan(new_unbias), unbias, new_unbias)

    value = biased / jnp.maximum(unbias, 1e-8)
    value = jnp.where(jnp.isnan(value), biased, value)

    return biased, unbias, value

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
    return jnp.reshape(jnp.concatenate(flat_gradients), (-1,1))

def running_gradient_2(running_grd, grad, itr, beta=0.9):
    return (beta * running_grd + (1 - beta)*grad) / (1 - beta**(itr + 1))


def init_running_gradients(grads):
    flat_grds = flatten_gradients(grads)
    running_grd = jnp.zeros_like(flat_grds)
    running_grd_sqr = jnp.zeros_like(flat_grds)
    return running_grd, running_grd_sqr


def run_grads(shape, store_grads, n_batch, running_noise, running_scale, beta, iteration):
    assert len(store_grads) == n_batch
    stored_grads = jnp.concatenate(store_grads, axis=1) 
    acc_grads = jnp.mean(stored_grads, axis=1)
    #g_small = jnp.mean(jnp.sum(stored_grads**2, axis=0))
    g_small = jnp.sum((stored_grads[:,-1] ** 2)) # mean or sum ?
    #g_big = jnp.sum(acc_grads ** 2)
    g_big = jnp.sum(acc_grads ** 2)    # mean or sum ?
    #b_small, b_big = batch.shape[0], batch.shape[0] * n_batch
    b_small, b_big = shape, shape * n_batch
    noise = (b_big * g_big - b_small * g_small) / (b_big - b_small)
    scale = (g_small - g_big) / ((1 / b_small) - (1 / b_big))
        
    lin_comb_scale = beta * running_scale + (1 - beta) * scale
    lin_comb_noise = beta * running_noise + (1 - beta) * noise

    running_scale, scale = lin_comb_scale, lin_comb_scale / (1-beta**(iteration+1))
    running_noise, noise = lin_comb_noise, lin_comb_noise / (1-beta**(iteration+1))

    noise_scale = (scale / noise)
        
    return noise, scale, noise_scale, running_noise, running_scale
    

def init_distributed_scalar():
    # Get the physical mesh that Alpa will use
    physical_mesh = get_global_physical_mesh(create_if_not_exist=True) # TODO: double-check if this is not causing problems with reallocation etc.
    
    # Create aval for a scalar
    aval = ShapedArray((), jnp.float32)
    
    # Create a replicated sharding spec for the scalar
    # This ensures the scalar is replicated across all devices
    mesh_shape = physical_mesh.shape  # e.g., (num_hosts, num_devices_per_host)
    sharding_spec = pxla.ShardingSpec(
        sharding=(),  # empty tuple for scalar
        mesh_mapping=(pxla.Replicated(np.prod(mesh_shape)),)  # replicate across all devices
    )
    
    # Initialize the scalar value
    scalar = jnp.array(0., dtype=jnp.float32)
    
    # Convert to distributed array
    distributed_arrays = physical_mesh.shard_args_to_arrays(
        [aval],  # list of abstract values
        [pxla.spec_to_indices(aval.shape, sharding_spec)],  # list of indices
        [sharding_spec],  # list of sharding specs
        [scalar]  # list of concrete values
    )
    
    return distributed_arrays[0]  # return the first (and only) array


def init_distributed_zeros_like(x):
    physical_mesh = get_global_physical_mesh(create_if_not_exist=True)
    mesh_shape = physical_mesh.shape

    def to_distributed_array(arr):
        # Create abstract value matching input array's shape and dtype
        aval = ShapedArray(arr.shape, arr.dtype)
        
        # Create a replicated sharding spec
        # For arrays (unlike scalars), we need to handle the actual shape
        sharding = tuple(pxla.NoSharding() for _ in arr.shape)
        mesh_mapping = (pxla.Replicated(np.prod(mesh_shape)),)
        sharding_spec = pxla.ShardingSpec(
            sharding=sharding,
            mesh_mapping=mesh_mapping
        )
        
        # Create zeros array with matching shape/dtype
        zeros = jnp.zeros_like(arr)
        
        # Convert to distributed array
        distributed_arrays = physical_mesh.shard_args_to_arrays(
            [aval],
            [pxla.spec_to_indices(aval.shape, sharding_spec)],
            [sharding_spec],
            [zeros]
        )
        return distributed_arrays[0]
    
    # Apply the conversion to the entire pytree
    return jax.tree_util.tree_map(to_distributed_array, x)

