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
soft_margin_loss
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=unused-argument,too-many-locals
@register_operator_compute("SoftMarginLoss", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def soft_margin_loss_compute(input_x, input_y, output_z, axis, reduction='mean', kernel_name="soft_margin_loss"):
    """calculating data

    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    input y : TVM tensor
        the placeholder of input_y
    axis : list
        the axis for reduction
    output_z : dict
        dict of output_z, include keys(shape and dtype)
    reduction : str
        reduction type, default value is "mean"
    kernel_name : str
        kernel name, default value is "soft_margin_loss"

    Returns
    -------
    output tensor
    """

    result_type = input_x.dtype.lower()
    trans_dtype = result_type
    if ((result_type == "float16" or result_type == "bfloat16") and
        tbe_platform.api_check_support("te.lang.cce.vexp", "float32")):
        input_x = tbe.cast_to(input_x, "float32")
        trans_dtype = "float32"
    if ((input_y.dtype.lower() == "float16" or input_y.dtype.lower() == "bfloat16") and
        tbe_platform.api_check_support("te.lang.cce.vexp", "float32")):
        input_y = tbe.cast_to(input_y, "float32")
        trans_dtype = "float32"

    x_mul = tbe.vmuls(input_x, tvm.const(-1, dtype=trans_dtype))
    x_y_mul = tbe.vmul(x_mul, input_y)
    res_exp = tbe.vexp(x_y_mul)
    res_add = tbe.vadds(res_exp, tvm.const(1, dtype=trans_dtype))
    result = tbe.vlog(res_add)
    
    if reduction == 'sum':
        result = tbe.reduce_sum(result, axis["value"], False)
    elif reduction == 'mean':
        result = tbe.reduce_mean(result, axis["value"], False)
            
    if result_type != trans_dtype:
        result = tbe.cast_to(result, result_type)
    return result


# 'pylint: disable=unused-argument,too-many-locals
@register_operator("SoftMarginLoss")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_STR,
                            para_check.KERNEL_NAME)
def soft_margin_loss(input_x, input_y, output_z, reduction='mean', kernel_name="soft_margin_loss"):
    """calculating data

    Parameters
    ----------
    input_x : dict
        shape and dtype of input_predict
    input_y: dict
        shape and dtype of input label
    output_z : dict
     if reduction is none, shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "soft_margin_loss"

    Returns
    -------
    None
    """
    shape_x = input_x.get("shape")
    dtype_x = input_x.get("dtype").lower()

    shape_y = input_y.get("shape")
    dtype_y = input_y.get("dtype").lower()

    #reduction value must be in reduction_list
    reduction_list = ("mean", "none", "sum")
    para_check.check_dtype(reduction.lower(), reduction_list, param_name="reduction")

    # dtype of input must be float16, float32, bfloat16
    check_tuple = ("float16", "float32", "bfloat16")
    para_check.check_shape(shape_x)
    para_check.check_shape(shape_y)

    if -1 not in shape_x and -2 not in shape_x and -1 not in shape_y and -2 not in shape_y:
        if len(shape_x) != len(shape_y):
            raise RuntimeError("input_x shape ", shape_x, " must be same as input_y shape ", shape_y)
        else:
            for dim, _ in enumerate(shape_x):
                if shape_x[dim] != shape_y[dim]:
                    raise RuntimeError("input_x shape ", shape_x, " must be same as input_y shape ", shape_y)

    para_check.check_dtype(dtype_x, check_tuple)
    para_check.check_dtype(dtype_y, check_tuple)

    para_check.check_kernel_name(kernel_name)
    
    axis = []
    for i, _ in enumerate(shape_x):
        axis.append(i)
    input_axis = {"shape": [len(axis), ], "value": axis, "rel_pos_to_reduce": "axis"}
    schedules, tensors = [], []
    if reduction != "none":
        ins = classify([input_x, input_y, input_axis], OpPatternMode.REDUCE, {"keepdims": False})
        for (_x_shape, _y_shape, _axis) in ins:
            with tbe.compute():
                x_shape, y_shape, _ = shape_util.variable_shape([_x_shape, _y_shape, _axis], op_mode="reduce")
                data_input1 = tvm.placeholder(x_shape, name="data_input1", dtype=dtype_x)
                data_input2 = tvm.placeholder(y_shape, name="data_input2", dtype=dtype_y)
                res = soft_margin_loss_compute(data_input1, data_input2, output_z, _axis, reduction, kernel_name)
                tensors.append([data_input1, data_input2, res])
            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)

    else:
        ins = classify([input_x, input_y], OpPatternMode.ELEWISE)
        for (_x_shape, _y_shape) in ins:
            with tbe.compute():
                x_shape, y_shape = shape_util.variable_shape([_x_shape, _y_shape])
                data_input1 = tvm.placeholder(x_shape, name="data_input1", dtype=dtype_x)
                data_input2 = tvm.placeholder(y_shape, name="data_input2", dtype=dtype_y)
                res = soft_margin_loss_compute(data_input1, data_input2, output_z, [], reduction, kernel_name)
                tensors.append([data_input1, data_input2, res])
            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)