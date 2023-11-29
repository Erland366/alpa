import collections
from typing import Any
import numpy as np
import jax
import jax.numpy as jnp

GradParams = collections.namedtuple("GradParams", ["sqr", "var"])

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
        batch_size = num_replicas * atomic_bsz * (accum_steps + 1)
        assert np.all(self._init_batch_size <= batch_size)
        return self.efficiency(batch_size)
    
    def efficiency(self, batch_size):
        grad_sqr = self._grad_params.sqr
        grad_var = self._grad_params.var
        scale = batch_size / self._init_batch_size
        denom = grad_var / scale + grad_sqr
        gain = jnp.where(denom > 0, (grad_var + grad_sqr) / denom, 1.0)
        return gain / scale
    
    def optimize(self, num_nodes, num_replicas, max_batch_size=None, 
                 atomic_bsz_range=None, accumulation=None):
        if max_batch_size is None:
            max_batch_size = self._init_batch_size
        assert self._init_batch_size <= max_batch_size
        atomic_bsz_range = atomic_bsz_range or (None, None)
        min_atomic_bsz = atomic_bsz_range[0] or 1
        max_atomic_bsz = atomic_bsz_range[1] or max_batch_size
        print(f'min_atomic_bsz: {min_atomic_bsz}')
        print(f'max_atomic_bsz: {max_atomic_bsz}')
        print(f'num_replicas: {num_replicas}')
        min_batch_size = jnp.maximum(self._init_batch_size, min_atomic_bsz * num_replicas)
        print(f'min_batch_size: {min_batch_size}')
        print(f'self._init_batch_size: {self._init_batch_size}')
        batch_size = jnp.geomspace(min_batch_size, max_batch_size)
        print(f'batch_size: {batch_size}')
        local_bsz = batch_size / num_replicas
        print(f'local_bsz: {local_bsz}')
        eps = 1e-8
        if accumulation:
            pass
        else:
            accum_steps = jnp.zeros_like(local_bsz)
            print(f'accum_steps: {accum_steps}')
            atomic_bsz = jnp.where(
                num_replicas == 1,
                self._init_batch_size,
                jnp.ceil(local_bsz - eps)
            )
            print(f'atomic_bsz: {atomic_bsz}')
        atomic_bsz = jnp.maximum(min_atomic_bsz, atomic_bsz)
        atomic_bsz = jnp.minimum(max_atomic_bsz, atomic_bsz)
        print(f'atomic_bsz after constraint: {atomic_bsz}')
        goodput = self.evaluate(num_nodes=num_nodes, 
                                    num_replicas=num_replicas, 
                                    atomic_bsz=atomic_bsz, 
                                    accum_steps=accum_steps)
        print(f'goodputs-----: {goodput}')
        indices = jnp.argmax(goodput, axis=0)
        print(f'indices: {indices}')
        goodput_elem = goodput[indices]
        print(f'best goodput: {goodput_elem}')
        atomic_bsz = atomic_bsz[indices]
        accum_steps = accum_steps[indices]

        return goodput_elem, atomic_bsz, accum_steps