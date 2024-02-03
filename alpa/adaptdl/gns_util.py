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

def average_groups_2(grads1, grads2):
    ret = []
    for group1, group2 in zip(grads1, grads2):
        ret.append([])
        for g1, g2 in zip(group1, group2):
            if g1 is None:
                ret[-1].append(g2)
            elif g2 is None:
                ret[-1].append(g1)
            else:
                ret[-1].append((g1 + g2) / 2)
    return ret

def update_avg(value, factor, biased, unbias):
    biased = factor * biased + (1.0 - factor) * value
    unbias = factor * unbias + (1.0 - factor)

    value = biased / unbias
    
    return biased, unbias, value

#def compute_pgns_values_2(store_grads, preconditioner, count=2, scale=1, smoothing=0.9):
#    grads_normsqr = normsqr_groups(store_grads[1], preconditioner)
#    local_sqr = (normsqr_groups(store_grads[0], preconditioner)
#                             + grads_normsqr) / 2
#    avg_grads = average_groups(store_grads[1], store_grads[0])
#    total_sqr = normsqr_groups(avg_grads, preconditioner)
#    grad_sqr = (count * total_sqr - local_sqr) / (count - 1)
#    grad_var = (local_sqr - total_sqr) * scale / (count - 1)
#    theta = smoothing ** scale
#    grad_sqr = update_avg(grad_sqr, theta)
#    grad_var = update_avg(grad_var, theta)
#    return grad_sqr, grad_var

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
    


