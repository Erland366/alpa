import collections
from typing import Any
import numpy as np
import jax
import jax.numpy as jnp
import logging
import datetime

from alpa.adaptdl.pollux_agent import PolluxAgent
from pollux_job import PolluxJob, JobState

GradParams = collections.namedtuple("GradParams", ["scale", "noise"])


LOGGER = logging.getLogger("vit logger")
LOGGER.setLevel(logging.INFO)
handler = logging.FileHandler('sevit.log')
# handler = logging.FileHandler(f"/home/haifatl/Documents/alpa/alpa-adaptdl-feb11/alpa-adaptdl/examples/huggingface/transformers/examples/flax/BERT/logs/Bert_goodput_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
#handler = logging.FileHandler(f"/home/haifatl/Documents/alpa/alpa-adaptdl-feb11/alpa-adaptdl/examples/ViT/logs/ViT_goodput_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
# LOGGER.info('testing vit logger')
   
class GoodputFunction(object):
    def __init__(self, 
                 grad_params, 
                 init_batch_size,
                 pollux_agent: PolluxAgent) -> None:
        self._grad_params = GradParams(*grad_params)
        self._init_batch_size = init_batch_size
        self._pollux_agent = pollux_agent

    def __call__(self, num_nodes, 
                 num_workers, alloc_config, atomic_bsz, accum_steps):
        return self.evaluate(num_nodes, num_workers, alloc_config, atomic_bsz, accum_steps)
    
    def evaluate(self, num_nodes, num_workers, alloc_config, atomic_bsz, accum_steps):
        global LOGGER
        batch_size = num_workers * atomic_bsz * (accum_steps + 1)
        assert np.all(self._init_batch_size <= batch_size)
        bsz_logs = atomic_bsz
        selogs = self.efficiency(batch_size)
        throughput_logs = jnp.ravel(self.throughput(batch_size, alloc_config))
        # try:
        #     LOGGER.info(f"atomic_bsz {bsz_logs}: SE = {selogs}, Throughput: {throughput_logs}, Goodput: {selogs * throughput_logs}")
        # except Exception as e:
        #     print(f'cannot write to selogs.log: {e}')
        #return self.efficiency(batch_size) * jnp.ravel(self.throughput(batch_size))
        return selogs * throughput_logs
    
    def efficiency_2(self, batch_size):
        grad_scale = self._grad_params.scale
        grad_noise = self._grad_params.noise
        pgns = grad_scale / grad_noise
        # LOGGER.info(f'noise: {grad_noise}')
        # LOGGER.info(f'scale: {grad_scale}')
        # LOGGER.info(f'pgns: {pgns}')
        se = (pgns + self._init_batch_size) / (pgns + batch_size)
        return se
    
    def efficiency(self, batch_size):
        grad_sqr = jnp.maximum(self._grad_params.scale, 0)
        grad_var = jnp.maximum(self._grad_params.noise, 1e-6)

        scale = batch_size / self._init_batch_size
        denom = grad_var / scale + grad_sqr
        gain = np.where(denom > 0, (grad_var + grad_sqr) / denom, 1.0)
        # LOGGER.info(f'Efficiency & GNS Computation')
        # LOGGER.info(f'grad_sqr: {grad_sqr}')
        # LOGGER.info(f'grad_var: {grad_var}')
        # LOGGER.info(f'scale: {scale}')
        # LOGGER.info(f'gain (GNS): {gain}')
        return gain / scale
    
    def throughput(self, batchsizes, alloc_config):
        throughput = self._pollux_agent.predict_throughput(batch_sizes=batchsizes, alloc_config=alloc_config)
        return throughput
    
    def optimize(self, num_nodes, num_workers, alloc_config, max_batch_size=None, 
                 atomic_bsz_range=None, accumulation=None):
        if max_batch_size is None:
            max_batch_size = self._init_batch_size
        assert self._init_batch_size <= max_batch_size
        atomic_bsz_range = atomic_bsz_range or (None, None)
        min_atomic_bsz = atomic_bsz_range[0] or 1
        max_atomic_bsz = atomic_bsz_range[1] or max_batch_size
        min_batch_size = jnp.maximum(self._init_batch_size, min_atomic_bsz * num_workers)
        batch_size = jnp.geomspace(min_batch_size, max_batch_size)
        local_bsz = batch_size / num_workers
        eps = 1e-8
        if accumulation:
            pass
        else:
            accum_steps = jnp.zeros_like(local_bsz)
            # TODO: akhmed: num_replicas was 1, I set it to -1 to test adaptive BS on a single GPU
            atomic_bsz = jnp.where(
                num_workers == -1,
                self._init_batch_size,
                jnp.ceil(local_bsz - eps)
            )
        atomic_bsz = jnp.maximum(min_atomic_bsz, atomic_bsz)
        atomic_bsz = jnp.minimum(max_atomic_bsz, atomic_bsz)
        #print(f'atomic_bsz: {atomic_bsz}')
        goodput = self.evaluate(num_nodes=num_nodes, 
                                    num_workers=num_workers,
                                    alloc_config=alloc_config,
                                    atomic_bsz=atomic_bsz,
                                    accum_steps=accum_steps)
        #print(f'goodput: {goodput}')
        indices = jnp.argmax(goodput, axis=0)
        # LOGGER.info(f'indices: {indices}')
        goodput_elem = goodput[indices]
        # LOGGER.info(f'best goodput: {goodput_elem}')
        atomic_bsz = atomic_bsz[indices]
        # LOGGER.info(f'best atomic_bsz: {atomic_bsz}')
        accum_steps = accum_steps[indices]

        return goodput_elem, atomic_bsz, accum_steps