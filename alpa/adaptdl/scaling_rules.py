import functools
import math
import numpy as np
import warnings

from types import MethodType

class ScalingRuleBase:
    def scale_lr(self, scale):
        raise NotImplementedError
    

class LinearScale(ScalingRuleBase):

    def scale_lr(self, scale):
        return scale


class SqrtScale(ScalingRuleBase):

    def scale_lr(self, scale):
        return math.sqrt(scale)


class AdaScale(ScalingRuleBase):
    """
    Implements the AdaScale_ algorithm for scaling the learning rate for
    distributed and large batch size training.

    .. _AdaScale: https://proceedings.icml.cc/static/paper_files/icml/2020/4682-Supplemental.pdf
    """  # noqa: E501

    def scale_lr(self, scale):
        """Calculate factors to be applied to lr for each parameter group."""
        var = self.adp.gns.raw_var_avg
        sqr = self.adp.gns.raw_sqr_avg
        var = np.maximum(var, 1e-6)
        sqr = np.maximum(sqr,  0.0)
        return (var + sqr) / (var / scale + sqr)


class AdamScale(AdaScale):
    """
    Implements the variant of AdaScale_ that supports Adam, AdamW and RMSProp
    """

    def scale_lr(self, scale,  power=0.5):
        return np.power(super().scale_lr(scale=scale), power)