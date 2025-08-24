#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2022 Huawei Technologies Co., Ltd
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
dynamic fused_bias_leaky_relu
"""
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_attr_common import OpAttr
from impl.util.util_attr_common import get_attr_by_cls


class FusedBiasLeakyReluAttrInfo:
    """
    define attr info
    """
    ATTR_SLOPE = OpAttr(0, "negative_slope", "Float")
    ATTR_SCALE = OpAttr(1, "scale", "Float")




@register_operator_compute("FusedBiasLeakyRelu", op_mode="dynamic", support_fusion=True)
def fused_bias_leaky_relu_compute(x, bias, negative_slope=0.2, scale=2**0.5, kernel_name="fused_bias_leaky_relu"):
    """
    compute for fused_bias_leaky_relu
    """
    
    dtype = x.dtype
    tmp = tbe.vadd(x, bias)
    negative_slope_scalar = get_attr_by_cls(negative_slope, FusedBiasLeakyReluAttrInfo.ATTR_SLOPE, dtype)
    res = tbe.vlrelu(tmp, negative_slope_scalar)
    scale_scalar = get_attr_by_cls(scale, FusedBiasLeakyReluAttrInfo.ATTR_SCALE, dtype)
    res = tbe.vmuls(res, scale_scalar)

    return res


@register_operator("FusedBiasLeakyRelu")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, 
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def fused_bias_leaky_relu(x, bias, y, negative_slope=0.2, scale=2**0.5, kernel_name="fused_bias_leaky_relu"):
    """fuse_bias_leaky_relu op for input tensor

       f(x)= scale*(x+bias)(x+bias>=0) or scale*negative_slope*(x+bias)(x+bias<0) equal to
       f(x)= scale*negative_slope*(x+bias)

    Parameters
    ----------
    x : TVM tensor
        input tensor has shape and dtype attributes
    bias : TVM tensor
        input tensor has shape and dtype attributes
    y : dict
        dict with keys(shape and dtype) of output

    negative_slope : float or int
        allow non-zero slope for negative inputs to speed up optimization. Default to 0.2
    
    scale : float or int
        A scalar to adjust the variance of the feture map. Default to 2**0.5

    kernel_name : str
        cce kernel name, default value is "fused_bias_leaky_relu"

    Returns
    ------
    None
    """

    # check input tensor shape
    x_dtype = x.get("dtype").lower()
    bias_dtype = bias.get("dtype").lower()

    # check input tensor data_type
    check_list = ["float16", "float32", "int32", "int8"]
    para_check.check_dtype(x_dtype, check_list, param_name="x")
    para_check.check_dtype(bias_dtype, check_list, param_name="bias")
    if x_dtype != bias_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal("fused_bias_leaky_relu", "x", "bias",
                                                            str(x_dtype), str(bias_dtype))

    ins = classify([x, bias], OpPatternMode.ELEWISE)
    schedules, tensors = [], []

    for (_x, _bias) in ins:
        with tbe.compute():
            x_shape, bias_shape = shape_util.variable_shape([_x, _bias])
            input_data_x = tvm.placeholder(x_shape, name="input_data_x",
                                           dtype=x_dtype)
            input_data_bias = tvm.placeholder(bias_shape, name="input_data_bias",
                                           dtype=bias_dtype)

            res = fused_bias_leaky_relu_compute(input_data_x, input_data_bias, negative_slope, scale, kernel_name)
            tensors.append([input_data_x, input_data_bias, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {
        "name": kernel_name,
        "tensor_list": tensors
    }

    tbe.build(schedules, config)
