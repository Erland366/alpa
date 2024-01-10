import collections
from typing import Any
import numpy as np
import jax
import jax.numpy as jnp
import logging



from alpa.adaptdl.pollux_agent import pollux_agent

GradParams = collections.namedtuple("GradParams", ["sqr", "var"])


LOGGER = logging.getLogger("vit logger")
LOGGER.setLevel(logging.INFO)
handler = logging.FileHandler('/home/ubuntu/alpa-adaptdl/examples/ViT/sevit.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
   
class GoodputFunction(object):
    def __init__(self, 
                 grad_params, 
                 init_batch_size) -> None:
        self._grad_params = GradParams(*grad_params)
        self._init_batch_size = init_batch_size

    def __call__(self, num_nodes, 
                 num_replicas, atomic_bsz, accum_steps):
        return self.evaluate(num_nodes, num_replicas, atomic_bsz, accum_steps)
    
    def evaluate(self, num_nodes, num_replicas, atomic_bsz, accum_steps):
        global LOGGER
        batch_size = num_replicas * atomic_bsz * (accum_steps + 1)
        assert np.all(self._init_batch_size <= batch_size)
        bsz_logs = atomic_bsz
        selogs = self.efficiency(batch_size)
        throughput_logs = jnp.ravel(self.throughtput(batch_size))
        try:
            LOGGER.info(f"atomic_bsz {bsz_logs}: SE = {selogs}, Throughput: {throughput_logs}")
        except Exception as e:
            print(f'cannot write to selogs.log: {e}')
        return self.efficiency(batch_size) * jnp.ravel(self.throughtput(batch_size))
    
    def efficiency_2(self, batch_size):
        grad_sqr = self._grad_params.sqr
        grad_var = self._grad_params.var
        scale = batch_size / self._init_batch_size
        denom = grad_var / scale + grad_sqr
        gain = jnp.where(denom > 0, (grad_var + grad_sqr) / denom, 1.0)
        return gain / scale
    
    def efficiency(self, batch_size):
        grad_sqr = self._grad_params.sqr
        grad_var = self._grad_params.var
        pgns = grad_var / grad_sqr
        se = (pgns + self._init_batch_size) / (pgns + batch_size)
        return se
    
    def throughtput(self, batchsizes):
        throughtput = pollux_agent.predict_throughput(batch_sizes=batchsizes)
        return throughtput
    
    def optimize(self, num_nodes, num_replicas, max_batch_size=None, 
                 atomic_bsz_range=None, accumulation=None):
        if max_batch_size is None:
            max_batch_size = self._init_batch_size
        assert self._init_batch_size <= max_batch_size
        atomic_bsz_range = atomic_bsz_range or (None, None)
        min_atomic_bsz = atomic_bsz_range[0] or 1
        max_atomic_bsz = atomic_bsz_range[1] or max_batch_size
        min_batch_size = jnp.maximum(self._init_batch_size, min_atomic_bsz * num_replicas)
        batch_size = jnp.geomspace(min_batch_size, max_batch_size)
        local_bsz = batch_size / num_replicas
        eps = 1e-8
        if accumulation:
            pass
        else:
            accum_steps = jnp.zeros_like(local_bsz)
            print(f'accum_steps: {accum_steps}')
            # TODO: akhmed: num_replicas was 1, I set it to -1 to test adaptive BS on a single GPU
            atomic_bsz = jnp.where(
                num_replicas == -1,
                self._init_batch_size,
                jnp.ceil(local_bsz - eps)
            )
        atomic_bsz = jnp.maximum(min_atomic_bsz, atomic_bsz)
        atomic_bsz = jnp.minimum(max_atomic_bsz, atomic_bsz)
        print(f'atomic_bsz: {atomic_bsz}')
        goodput = self.evaluate(num_nodes=num_nodes, 
                                    num_replicas=num_replicas, 
                                    atomic_bsz=atomic_bsz, 
                                    accum_steps=accum_steps)
        print(f'goodput: {goodput}')
        indices = jnp.argmax(goodput, axis=0)
        print(f'indices: {indices}')
        goodput_elem = goodput[indices]
        print(f'best goodput: {goodput_elem}')
        atomic_bsz = atomic_bsz[indices]
        accum_steps = accum_steps[indices]

        return goodput_elem, atomic_bsz, accum_steps