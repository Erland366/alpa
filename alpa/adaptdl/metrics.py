import jax
import jax.numpy as jnp

import jax
import jax.numpy as jnp

from alpa.adaptdl.goodput import GoodputFunction

_GRAD_PARAM_DICT = None

def update_grad_params(grad_norm_sqr, grad_variance):
    global _GRAD_PARAM_DICT
    _GRAD_PARAM_DICT = jnp.asarray([grad_norm_sqr, grad_variance])
    #print(f'_GRAD_PARAM_DICT: {_GRAD_PARAM_DICT}')
    #grad_params = sum(_GRAD_PARAM_DICT.values())
    grad_params = _GRAD_PARAM_DICT
    print(f'grad_params: {grad_params}')
    _metric_state().grad_params = (grad_params[0], grad_params[1])
    print(_metric_state().grad_params)


def set_batch_size(init_batch_size, 
                   max_batch_size, 
                   local_bsz_bounds, 
                   gradient_accumulation):
    state = _metric_state()
    state.init_batch_size = init_batch_size
    state.max_batch_size = max_batch_size
    state.local_bsz_bounds = local_bsz_bounds
    state.gradient_accumulation = gradient_accumulation

def get_goodput_fn():
    state = _metric_state()
    if state.grad_params is None:
        return None
    return GoodputFunction(state.grad_params, state.init_batch_size)

def get_progress():
    return _metric_state().progress

def update_progress(progress):
    _metric_state().progress = progress

class _MetricsState():
    def __init__(self) -> None:
        self.grad_params = None
        self.init_batch_size = None
        self.max_batch_size = None
        self.local_bsz_bounds = None
        self.gradient_accumulation = False
        self.progress = 0.0

def _metric_state():
    global _METRICS_STATE
    if _METRICS_STATE is None:
        _METRICS_STATE = _MetricsState()
    return _METRICS_STATE

_METRICS_STATE = None