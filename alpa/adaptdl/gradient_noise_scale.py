import jax
import jax.numpy as jnp
#import time
import alpa

#from jax.tree_util import tree_flatten, tree_unflatten
#from jax.lib import xla_bridge
from alpa.device_mesh import DistributedArray

class GradientNoiseScale():
    def __init__(self, mp_scaler=None, 
                 state=None, 
                 num_workers=None, 
                 init_batch_size=None, 
                 accum_scale=None 
                 ) -> None:
        # initialization of pgns state
        self.state = state          # pgns_gradients unflattened
        self.mp_scaler = mp_scaler
        #self.num_replicas = num_workers
        self.num_workers = num_workers
        self.accum_scale = accum_scale
        self.accum_count = 1
        self.init_batch_size = init_batch_size
        self.store_grads = jax.tree_util.tree_map(jnp.zeros_like, state)         #previous Gradient
        #self.running_noise = 0.0
        #self.running_scale = 0.0
        # self.noise = 0.0
        # self.scale = 0.0
        # self.noise_scale = 0.0
        # self.biased_sqr = 0.0 
        # self.unbias_sqr = 0.0 
        # self.biased_var = 0.0 
        # self.unbias_var = 0.0
        self.noise = jnp.array(0.)
        self.scale = jnp.array(0.)
        self.noise_scale = jnp.array(0.)
        self.biased_sqr = jnp.array(0.)
        self.unbias_sqr = jnp.array(0.)
        self.biased_var = jnp.array(0.)
        self.unbias_var = jnp.array(0.)
    
    def set_preconditioner(self, grads):
        def ones_like(x):
            return jnp.ones_like(x)

        pinv = jax.tree_util.tree_map(ones_like, grads._value)
        return pinv
   
    # def update_state(self, state, grad_sqr, grad_var, biased_sqr, unbias_sqr, biased_var, unbias_var, gradients):
    #     self.state          = state
    #     self.noise          = grad_sqr
    #     self.scale          = grad_var
    #     self.biased_sqr     = biased_sqr 
    #     self.unbias_sqr     = unbias_sqr 
    #     self.biased_var     = biased_var 
    #     self.unbias_var     = unbias_var
    #     self.store_grads    = gradients

    def update_state(self, state, grad_sqr, grad_var):
        self.state          = state
        self.noise          = grad_sqr
        self.scale          = grad_var

    def initialize_gns(self, state, init_bsz, num_workers, accum_scale):
        self.state = state
        self.init_batch_size = init_bsz
        self.num_workers = num_workers
        self.accum_scale = accum_scale

gns = GradientNoiseScale()
   