import jax
import jax.numpy as jnp
import alpa

class GradientNoiseScale():
    def __init__(self, mp_scaler=None, 
                 state=None, 
                 num_workers=None, 
                 init_batch_size=None, 
                 ) -> None:
        self.state = state
        self.mp_scaler = mp_scaler
        self.num_workers = num_workers
        self.accum_count = 1
        self.init_batch_size = init_batch_size
        self.store_grads = jnp.array(0.)
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

    def update_state(self, state, train_metric: dict):
        self.state          = state
        self.noise          = train_metric["grad_sqr"]
        self.scale          = train_metric["grad_var"]
        self.biased_sqr     = train_metric["biased_sqr"]
        self.unbias_sqr     = train_metric["unbias_sqr"]
        self.biased_var     = train_metric["biased_var"]
        self.unbias_var     = train_metric["unbias_var"]
        self.store_grads    = train_metric["gradients"]

    def initialize_gns(self, state, init_bsz, num_workers, store_grads, count, scale, theta):
        self.state = state
        self.init_batch_size = init_bsz
        self.num_workers = num_workers
        self.store_grads = store_grads
        self.gns_count = count
        self.gns_scale = scale
        self.gns_theta = theta

    def construct_gns_dict(self):
        gns_dict = {
            'gns_store_grads': self.store_grads, 
            'gns_biased_sqr': self.biased_sqr, 
            'gns_unbias_sqr': self.unbias_sqr, 
            'gns_biased_var': self.biased_var, 
            'gns_unbias_var': self.unbias_var, 
            'count': self.gns_count, 
            'scale': self.gns_scale, 
            'theta': self.gns_theta
            }
        
        return gns_dict
        

gns = GradientNoiseScale()
   