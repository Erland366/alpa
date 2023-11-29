import jax
import jax.numpy as jnp
import time

from jax.tree_util import tree_flatten, tree_unflatten
from jax.lib import xla_bridge

jax.config.update('jax_platform_name', 'gpu')  # Set the platform to GPU
print(jax.devices())

def flatten(x):
    """Flatten a nested JAX structure."""
    leaves, tree_structure = jax.tree_flatten(x)
    jax_leaves = [jnp.array(leaf) for leaf in leaves]
    flat_params = jnp.concatenate([jnp.ravel(leaf) for leaf in jax_leaves])
    return flat_params, tree_structure


def extract_values_with_key(structure):
    flat_structure, tree_def = tree_flatten(structure)
    #flattened_gradients = [node._value for node in flat_structure]
    flattened_gradients = [node for node in flat_structure]
    
    #jax_tree = jax.tree_util.tree_map(lambda x: x if isinstance(x, jnp.ndarray) else jnp.asarray(x), flattened_gradients)
    return (flattened_gradients)

def _average_groups(prev_grads, grads):
    ret = []
    for prev_grp, grp in zip(prev_grads, grads):
        ret.append([])
        for p_g, g in zip(prev_grp, grp):
            if p_g is None:
                ret[-1].append(g)
            elif g is None:
                ret[-1].append(p_g)
            else:
                ret[-1].append((p_g + g) / 2)
    return ret

#@jax.jit
def update_total_local_sqr(local_sqr, grad, preconditioner):
    local_sqr += jnp.sum((grad._value / preconditioner._value) ** 2)
    return local_sqr

#@jax.jit
def update_local_sqr(g, pinv):
    return jnp.sum((g/pinv)**2)

#@jax.jit
def sum_normsqr(normsqr):
    return jnp.sum(normsqr)


def _normsqr_groups(grads, pinvs):
    ret = []
    for group, pinv_group in zip(grads, pinvs):
        print('start computing normsqr - loop inside _normsqr_groups')
        normsqr = [jnp.sum((g/pinv)**2) for g, pinv in zip(group, pinv_group) if g is not None]
        #normsqr = [update_local_sqr(g, pinv) for g, pinv in zip(group, pinv_group) if g is not None]
        #ret.append(jnp.sum(jnp.array(normsqr)) if normsqr else 0.0)
        ret.append(sum_normsqr(jnp.array(normsqr)) if normsqr else 0.0)
        print('end computing normsqr - loop inside _normsqr_groups')
    print('now return jnp.sum(jnp.array(ret))')
    #print(f'ret: {jnp.sum(jnp.array(ret))}')
    return jnp.sum(jnp.array(ret))

def _normsqr_groups_2(grads, pinvs):
    #jax_device = jax.local_devices(backend='gpu')[0]
    #print("jax devices",jax.device_count())
    print(xla_bridge.get_backend().platform)
    normsqr_list = [update_local_sqr(g, pinv) for group, pinv_group in zip(grads, pinvs) for g, pinv in zip(group, pinv_group) if g is not None]

    return sum_normsqr(jnp.array(normsqr_list))

def compute_gradsnorms(gradients, preconditioners, num_replicas, accum_scale, accum_count=False):
        local_sqr_val = 0.0
        print('start computing local_sqr_val')
        for grad, preconditioner in zip(gradients, preconditioners):
            local_sqr_val += jnp.sum((grad/preconditioner)**2)
        print('end computing local_sqr_val')
        print('start computing grads_normsqr')
        grads_normsqr = _normsqr_groups(gradients, preconditioners)
        print('end computing grads_normsqr')
        #count = num_replicas * accum_count if accum_count is not None else num_replicas
        #scale = accum_scale * accum_count if accum_count is not None else accum_scale
        count = num_replicas
        scale = accum_scale

        return local_sqr_val, grads_normsqr, count, scale


class GradientNoiseScale():
    def __init__(self, 
                 mp_scaler, 
                 state, 
                 num_workers, 
                 init_batch_size, 
                 accum_scale=None, 
                 smoothing=0.999) -> None:
        self._prev_grads = None
        self._local_sqr = 0.0
        self.state = state #pgns_gradients unflattened
        #print('state is saved on: ', jax.tree_map(lambda x: x.device_buffer.device(), self.state))
        self._mp_scaler = mp_scaler
        self._num_replicas = num_workers
        self._accum_scale = accum_scale or self._num_replicas
        self._accum_count = None
        self._init_batch_size = init_batch_size
        self._smoothing = smoothing

        # initialization of pgns state
        self.pgns = {
            'progress'  : 0.0,
            'prev_scale': 0.0,
            'sqr_avg'   : jnp.ones(1),
            'var_avg'   : jnp.zeros(1),
            'biased'    : False,
        }

    @property
    def _state(self):
        return self.pgns
    
    def sqr_avg(self):
        return jnp.float32(jnp.sum(jnp.maximum(self.pgns["sqr_avg"], 0.0)))

    def var_avg(self):
        return jnp.float32(jnp.sum(jnp.maximum(self.pgns["var_avg"], 1e-6)))

    def get_progress(self):
        return self.pgns["progress"]
    
    def set_progress(self, progress):
        self.pgns["progress"] = progress

    def gain(self, scale):
        var = self.var_avg()
        norm = self.sqr_avg()
        return (var + norm) / (var / scale + norm)
    
    #@jax.jit
    def set_preconditioner(self, grads):
        def ones_like(x):
            return jnp.ones_like(x)

        pinv = jax.tree_util.tree_map(ones_like, grads._value)
        return pinv
   
    def update_state(self,state):
        self.state = state

    def _reset_avg(self, param_name):
        self._state.pop(param_name + "_biased", None)
        self._state.pop(param_name + "_unbias", None)
    
    def _update_avg(self, param_name, value, factor):
        biased = self._state.get(param_name + "_biased", 0.0)
        unbias = self._state.get(param_name + "_unbias", 0.0)
        biased = factor * biased + (1.0 - factor) * value
        unbias = factor * unbias + (1.0 - factor)
        self._state[param_name + "_biased"] = biased
        self._state[param_name + "_unbias"] = unbias
        self._state[param_name] = biased / unbias

    
    def compute_pgns(self, preconditioners):
        #gradients = extract_values_with_key(self.state.params)
        #print("gradients", gradients)
        print('------------------------------------------------------------------------')
        time1 = time.time()
        #for grad in gradients:
        #print(self.state)
        for grad, preconditioner in zip (self.state, preconditioners):
            #print('****************************************************')
            #grad = jax.device_put(grad, jax_device)
            #preconditioner = self.set_preconditioner(grad)
            #preconditioner = jax.device_put(preconditioner, jax_device)
            #####self._local_sqr += jnp.sum((grad/preconditioner)**2)
            self._local_sqr = update_total_local_sqr(self._local_sqr, grad, preconditioner)
        time1 = time.time() - time1
        print(f'time1: {time1}')

        #preconditioners = self.set_preconditioner(gradients)
        #preconditioners = self.set_preconditioner(self.state)
        #preconditioners = jax.device_put(preconditioners, jax_device)

        #self._local_sqr = [update_total_local_sqr(self._local_sqr, grad, preconditioner) for grad, preconditioner in zip(gradients, preconditioners)]
        #self._local_sqr = jnp.array(self._local_sqr)[0]

        time2 = time.time()
        #grads_normsqr = _normsqr_groups(gradients, preconditioners)
        grads_normsqr = _normsqr_groups(self.state, preconditioners)  
        time2 = time.time() - time2  
        print(f'time2: {time2}')
           
        count = self._num_replicas * self._accum_count if self._accum_count is not None else self._num_replicas
        scale = self._accum_scale * self._accum_count if self._accum_count is not None else self._accum_scale
        if count > 1:
            local_sqr = self._local_sqr / count
            total_sqr = grads_normsqr
            if self._state['biased']:
                self._reset_avg("sqr_avg")
                self._reset_avg("var_avg")
            self._state['biased'] = False
            self._prev_grads = None
        else:
            if self._prev_grads is not None:
                local_sqr = (_normsqr_groups(self._prev_grads, preconditioners) + grads_normsqr) / 2
                #avg_grads = _average_groups(gradients, self._prev_grads)
                avg_grads = _average_groups(self.state, self._prev_grads)
                total_sqr = _normsqr_groups(avg_grads, preconditioners)
                count = 2
                scale = 2 * self._accum_scale
            self._state["biased"] = True
            self._prev_grads = [[g.clone() if g is not None else None for g in group] for group in gradients]
        if count > 1:
            grad_sqr = (count * total_sqr - local_sqr) / (count -1)
            grad_var = (local_sqr - total_sqr) * scale / (count -1)
            print(f'grad_sqr: {grad_sqr}')
            print(f'grad_var: {grad_var}')
            theta = self._smoothing * scale
            self._update_avg('sqr_avg', grad_sqr, theta)
            self._update_avg('var_avg', grad_var, theta)
            #print(f'self._state: {self._state}')
        self.set_progress(self.get_progress() + self.gain(scale))
        print(f'pgns: {self._state}')
        






def main():
    pass

if __name__ == "__main__":
    main()