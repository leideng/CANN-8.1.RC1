#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
util_attr_constant
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm


# 'pylint: disable=too-few-public-methods
class OpAttr:
    """
    OpAttr, define the attr base info
    """

    def __init__(self, attr_idx, attr_name, attr_type, attr_default_value=None):
        self.attr_idx = attr_idx
        self.attr_name = attr_name
        self.attr_type = attr_type
        self.attr_value = attr_default_value


def get_attr_by_cls(attr_value, attr_cls, target_dtype):
    """
    get the attr

    Parameters
    ----------
    attr_value: value of attr or None
    attr_cls: OpAttr
    target_dtype: the dtype used for calculation in tvm

    Returns
    -------
    attr_var
    """
    if attr_value is None:
        attr_sting_lower = attr_cls.attr_type.lower()
        attr_dtype = {"src_dtype": attr_sting_lower, "index": attr_cls.attr_idx}
        attr_var = tbe.var_attr(attr_cls.attr_name, dtype=target_dtype, addition=attr_dtype)
    else:
        attr_var = tvm.const(attr_value, target_dtype)
    return attr_var


# begin to define the attr for op
class SyncBatchNormGatherStatsWithCountsAttrInfo:
    """
    define SyncBatchNormGatherStatsWithCounts attr info
    """
    ATTR_MOMENTUM = OpAttr(0, "momentum", "Float", 0.1)
    ATTR_EPSILON = OpAttr(1, "epsilon", "Float", 0.001)


class HardtanhGradAttrInfo:
    """
    define HardtanhGrad attr info
    """
    ATTR_MIN_VAL = OpAttr(0, "min_val", "Float", -1.0)
    ATTR_MAX_VAL = OpAttr(1, "max_val", "Float", 1.0)


class LayerNormAttrInfo:
    """
    define LayerNorm attr info
    """
    ATTR_NORM_AXIS = OpAttr(0, "begin_norm_axis", "Int", 0)
    ATTR_PARAMS_AXIS = OpAttr(1, "begin_params_axis", "Int", 0)
    ATTR_EPSILON = OpAttr(2, "epsilon", "Float", 0.0000001)


class MaskedScaleAttrInfo:
    """
    define MaskedScale attr info
    """
    ATTR_VALUE = OpAttr(0, "value", "Float", 1.0)


class SoftmaxV2AttrInfo:
    """
    define SoftmaxV2 attr info
    """
    ATTR_AXES = OpAttr(0, "axes", "ListInt", [-1])


class ShrinkAttrInfo:
    """
    define Shrink attr info
    """
    ATTR_LAMBD = OpAttr(0, "lambd", "Float", 0.5)
    ATTR_BIAS = OpAttr(1, "bias", "Float", 0.0)


class Relu6DAttrInfo:
    """
    define Relu6D attr info
    """
    ATTR_SCALE = OpAttr(0, "scale", "Float", 1.0)


class RenormAttrInfo:
    """
    define Renorm attr info
    """
    ATTR_P = OpAttr(0, "p", "Float")
    ATTR_DIM = OpAttr(1, "dim", "Int")
    ATTR_MAXNORM = OpAttr(2, "maxnorm", "Float")


class SwishAttrInfo:
    """
    define Swish attr info
    """
    ATTR_SCALE = OpAttr(0, "scale", "Float", 1.0)


class SwishGradAttrInfo:
    """
    define SwishGrad attr info
    """
    ATTR_SCALE = OpAttr(0, "scale", "Float", 1.0)


class ThresholdAttrInfo:
    """
    define Threshold attr info
    """
    ATTR_THRESHOLD = OpAttr(0, "threshold", "Float", 0.0)


class HardShrinkAttrInfo:
    """
    define HardShrink attr info
    """
    ATTR_LAMBD = OpAttr(0, "lambd", "Float", 0.5)


class HardShrinkGradAttrInfo:
    """
    define HardShrinkGrad attr info
    """
    ATTR_LAMBD = OpAttr(0, "lambd", "Float", 0.5)


class IsCloseAttrInfo:
    """
    define IsClose attr info
    """
    ATTR_RTOL = OpAttr(0, "rtol", "Float", 1e-5)
    ATTR_ATOL = OpAttr(1, "atol", "Float", 1e-8)


class HardSigmoidAttrInfo:
    """
    define HardSigmoid attr info
    """
    ATTR_ALPHA = OpAttr(0, "alpha", "Float", 0.16666666)
    ATTR_BETA = OpAttr(1, "beta", "Float", 0.5)


class HardSigmoidGradAttrInfo:
    """
    define HardSigmoidGrad attr info
    """
    ATTR_ALPHA = OpAttr(0, "alpha", "Float", 0.16666666)
    ATTR_BETA = OpAttr(1, "beta", "Float", 0.5)


class EluGradV2AttrInfo:
    """
    define EluGradV2 attr info
    """
    ATTR_ALPHA = OpAttr(0, "alpha", "Float", 1.0)
    ATTR_SCALE = OpAttr(1, "scale", "Float", 1.0)
    ATTR_INPUT_SCALE = OpAttr(2, "input_scale", "Float", 1.0)


class SparseApplyRMSPropDAttrInfo:
    """
    define SparseApplyRMSPropD attr info
    """
    ATTR_RHO = OpAttr(0, "rho", "Float", 0.0000001)
    ATTR_MOM = OpAttr(1, "momentum", "Float", 0.0000001)
    ATTR_EPS = OpAttr(2, "epsilon", "Float", 0.0000001)


class SmoothL1LossV2AttrInfo:
    """
    define SmoothL1LossV2 attr info
    """
    ATTR_SIGMA = OpAttr(0, "sigma", "Float", 1.0)


class SmoothL1LossGradV2AttrInfo:
    """
    define SmoothL1LossGradV2 attr info
    """
    ATTR_SIGMA = OpAttr(0, "sigma", "Float", 1.0)


class CeluV2AttrInfo:
    """
    define CeluV2 attr info
    """
    ATTR_ALPHA = OpAttr(0, "alpha", "Float", 1.0)


class SoftplusV2AttrInfo:
    """
    define SoftplusV2 attr info
    """
    ATTR_BETA = OpAttr(0, "beta", "Float", 1.0)
    ATTR_THRESHOLD = OpAttr(1, "threshold", "Float", 20.0)


class SoftShrinkInfo:
    """
    define SoftShrink attr info
    """
    ATTR_LAMBD = OpAttr(0, "lambd", "Float", 0.5)


class SoftShrinkGradInfo:
    """
    define SoftShrinkGrad attr info
    """
    ATTR_LAMBD = OpAttr(0, "lambd", "Float", 0.5)


class SoftplusV2GradAttrInfo:
    """
    define SoftplusV2Grad attr info
    """
    ATTR_BETA = OpAttr(0, "beta", "Float", 1.0)
    ATTR_THRESHOLD = OpAttr(1, "threshold", "Float", 20.0)
