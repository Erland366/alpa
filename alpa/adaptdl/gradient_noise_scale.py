import jax
import jax.numpy as jnp
import time
import alpa

from jax.tree_util import tree_flatten, tree_unflatten
from jax.lib import xla_bridge

class GradientNoiseScale():
    def __init__(self, mp_scaler=None, 
                 state=None, 
                 num_workers=None, 
                 init_batch_size=None, 
                 accum_scale=None, 
                 smoothing=0.999) -> None:
        # initialization of pgns state
        self.pgns = {
            'progress'          : 0.0,
            'prev_scale'        : 0.0,
            'local_sqr_val'     : 0.0,
            'pgns_grads_norms'  : 0.0,
            'sqr_avg'           : jnp.ones(1),
            'var_avg'           : jnp.zeros(1),
            'biased'            : False,
        }
        self._prev_grads = None
        self.localsqr = 0.0
        self.state = state          # pgns_gradients unflattened
        self.mp_scaler = mp_scaler
        self.num_replicas = num_workers
        self.accum_scale = accum_scale
        self.accum_count = None
        self.init_batch_size = init_batch_size
        self.smoothing = smoothing
        self.store_grads = []
        #self.precondionners = []
        self.running_noise = 0.0
        self.running_scale = 0.0
        self.noise = None
        self.scale = None
        self.noise_scale = None
        self.biased_sqr = 0.0 
        self.unbias_sqr = 0.0 
        self.biased_var = 0.0 
        self.unbias_var = 0.0
        

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
    
    def set_preconditioner(self, grads):
        def ones_like(x):
            return jnp.ones_like(x)

        pinv = jax.tree_util.tree_map(ones_like, grads._value)
        return pinv
   
    def update_state(self, state):
        self.state                      = state
        #self.pgns['local_sqr_val']      = local_sqr_val._value
        #self.pgns['pgns_grads_norms']   = pgns_grads_norms._value

    def _reset_avg(self, param_name):
        self._state.pop(param_name + "_biased", None)
        self._state.pop(param_name + "_unbias", None)
    
    def _update_avg_2(self, param_name, value, factor):
        biased = self._state.get(param_name + "_biased", 0.0)
        unbias = self._state.get(param_name + "_unbias", 0.0)
        biased = factor * biased + (1.0 - factor) * value
        unbias = factor * unbias + (1.0 - factor)
        self._state[param_name + "_biased"] = biased
        self._state[param_name + "_unbias"] = unbias
        self._state[param_name] = biased / unbias

    def _update_avg(self, param_name, value):
        self._state[param_name] = value

    def compute_pgns_p(self, grads_normsqr, local_sqr_):  

        #self._local_sqr = local_sqr_._value
        self.localsqr = local_sqr_
        count = self.num_replicas * self.accum_count if self.accum_count is not None else self.num_replicas
        scale = self.accum_scale * self.accum_count if self.accum_count is not None else self.accum_scale
        
        #local_sqr = local_sqr_._value / count
        #total_sqr = grads_normsqr._value
        #local_sqr = local_sqr_ / count
        grad_var = local_sqr_ 
        grad_sqr = grads_normsqr
        #total_sqr = grads_normsqr
        if self._state['biased']:
            self._reset_avg("sqr_avg")
            self._reset_avg("var_avg")
        self._state['biased'] = False
        self._prev_grads = None
        
        #grad_sqr = (count * total_sqr - local_sqr) / (count -1)
        #grad_var = (local_sqr - total_sqr) * scale / (count -1)
        #theta = self.smoothing * scale
        theta = self.smoothing
        self._update_avg('sqr_avg', grad_sqr, theta)
        self._update_avg('var_avg', grad_var, theta)
        self.set_progress(self.get_progress() + self.gain(scale))
        return self._state
    
    def compute_pgns(self, grad_sqr, grad_var):
        self._update_avg('sqr_avg', grad_sqr)
        self._update_avg('var_avg', grad_var)
        #return self.pgns['local_sqr_val'] / self.pgns['pgns_grads_norms']
        return self._state
    
gns = GradientNoiseScale()
   