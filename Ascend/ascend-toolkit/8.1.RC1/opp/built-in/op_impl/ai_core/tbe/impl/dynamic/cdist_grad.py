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
cdist_grad
"""

import math
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


#Power calculation
def tensor_pow(data, p):
    """
    tensor_pow
    """
    log_value = tbe.vlog(data)
    return tbe.vexp(tbe.vmuls(log_value, p))


#Compute the less than two norm
def less_than_two_norm_compute(p, sign, dtype, input_grad, diff_abs, nz_cdist, input_cdist, zero):
    power = tvm.const(p - 1, dtype=dtype)

    num = tbe.vmul(sign, tensor_pow(diff_abs, power))
    numerator = tbe.vmul(num, input_grad)
    denominator = tensor_pow(nz_cdist, power)

    res = tbe.vdiv(numerator, denominator)
    res = tbe.vcmpsel(input_cdist, zero, 'eq', slhs=zero, srhs=res)
    res = tbe.vcmpsel(diff_abs, zero, 'eq', slhs=zero, srhs=res)

    return res


#Compute the two norm
def two_norm_comupte(input_grad, diff, nz_cdist, input_cdist, zero):
    numerator = tbe.vmul(input_grad, diff)
    res = tbe.vdiv(numerator, nz_cdist)
    res = tbe.vcmpsel(input_cdist, zero, 'eq', slhs=zero, srhs=res)

    return res


#Compute the general p norm
def general_p_norm_compute(p, dtype, diff, diff_abs, input_grad, nz_cdist, input_cdist, zero):
    power1 = tvm.const(p - 1, dtype=dtype)
    power2 = tvm.const(p - 2, dtype=dtype)

    num = tbe.vmul(diff, tensor_pow(diff_abs, power2))
    numerator = tbe.vmul(num, input_grad)
    denominator = tensor_pow(nz_cdist, power1)

    res = tbe.vdiv(numerator, denominator)
    res = tbe.vcmpsel(input_cdist, zero, 'eq', slhs=zero, srhs=res)

    return res


#Compute the inf norm
def inf_norm_compute(input_cdist, diff_abs, input_grad, sign, zero, one):
    mask = tbe.vcmpsel(input_cdist, diff_abs, 'gt', slhs=zero,
                           srhs=one)
    grad_sign = tbe.vmul(input_grad, sign)
    res = tbe.vmul(grad_sign, mask)

    return res


# 'pylint: disable=unused-argument,too-many-arguments,too-many-locals
@register_operator_compute("CdistGrad", op_mode="dynamic", support_fusion=True)
def cdist_grad_compute(input_grad, input_x1, input_x2, input_cdist, axis, p,
                       kernel_name="cdist_grad"):
    """
    calculating data

    Parameters
    ----------
    input_grad : TVM tensor
        the placeholder of input_grad
    input_x1 : TVM tensor
        the placeholder of input_x1
    input_x2 : TVM tensor
        the placeholder of input_x2
    input_cdist : TVM tensor
        the placeholder of input_cdist
    p : float
        norm number
    axis: int
        the axis to reduce
    kernel_name : str
        kernel name, default value is "cdist"

    Returns
    -------
    :return: TVM tensor
        result tensor
    """
    dtype = input_x1.dtype

    if dtype == 'float16' and \
            tbe_platform.api_check_support("te.lang.cce.vlog", "float32") and \
            tbe_platform.api_check_support("te.lang.cce.vexp", "float32"):
        input_x1 = tbe.cast_to(input_x1, 'float32')
        input_x2 = tbe.cast_to(input_x2, 'float32')
        input_grad = tbe.cast_to(input_grad, 'float32')
        input_cdist = tbe.cast_to(input_cdist, 'float32')

    diff = tbe.vsub(input_x1, input_x2)
    diff_abs = tbe.vabs(diff)

    zero = tvm.const(0, dtype=dtype)
    one = tvm.const(1, dtype=dtype)
    n_one = tvm.const(-1, dtype=dtype)

    nz_cdist = tbe.vcmpsel(input_cdist, zero, 'ne', srhs=one)
    sign = tbe.vcmpsel(diff, zero, 'gt', slhs=one, srhs=n_one)
    sign = tbe.vcmpsel(diff, zero, 'eq', slhs=zero, srhs=sign)

    z_1 = tbe.vmuls(diff, zero)
    z_2 = tbe.vmul(z_1, input_grad)
    z_3 = tbe.vmul(z_2, input_cdist)

    if math.isclose(p, 0.0):
        # If p is 0, the result is a zero tensor.
        pass
    elif math.isclose(p, 1.0):
        res = tbe.vmul(input_grad, sign)
    elif 0 < p < 2.0:
        res = less_than_two_norm_compute(p, sign, dtype, input_grad, diff_abs, nz_cdist, input_cdist, zero)
    elif math.isclose(p, 2.0):
        res = two_norm_comupte(input_grad, diff, nz_cdist, input_cdist, zero)
    elif math.isclose(p, -1): # Note: -1 here means inf
        res = inf_norm_compute(input_cdist, diff_abs, input_grad, sign, zero, one)
    else:
        res = general_p_norm_compute(p, dtype, diff, diff_abs, input_grad, nz_cdist, input_cdist, zero)

    if math.isclose(p, 0.0):
        ret = tbe.reduce_sum(z_3, axis=axis["value"])
    else:
        ret = tbe.reduce_sum(tbe.vadd(res, z_3), axis=axis["value"])

    if dtype == 'float16':
        return tbe.cast_to(ret, 'float16')
    return ret


# 'pylint: disable=unused-argument,too-many-arguments,too-many-locals
@register_operator("CdistGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def cdist_grad(input_grad, input_x1, input_x2, input_cdist, output_y, p,
               kernel_name="cdist_grad"):
    """
    calculating data

    Parameters
    ----------
    :param input_grad: dict
        shape and dtype of grad
    :param input_x1: dict
        shape and dtype of input1
    :param input_x2: dict
        shape and dtype of input2
    :param input_cdist: dict
        shape and dtype of dist
    :param output_y: dict
        shape and dtype of output, should be same shape and type as input
    :param p: float
        norm number
    :param kernel_name: str
        kernel name, default value is "cdist"

    Returns
    -------
    :return: None
    """
    # To improve performance, the "broadcast" moves to the adaptation layer.
    # Therefore, the input_x1.shape and input_x2.shape are equal.
    para_check.check_kernel_name(kernel_name)

    grad_shape = list(shape_util.scalar2tensor_one(input_grad.get("shape")))
    grad_dtype = input_grad.get("dtype")
    cdist_shape = list(shape_util.scalar2tensor_one(input_cdist.get("shape")))
    cdist_dtype = input_cdist.get("dtype")
    x1_shape = list(shape_util.scalar2tensor_one(input_x1.get("shape")))
    x1_dtype = input_x1.get("dtype")
    x2_shape = list(shape_util.scalar2tensor_one(input_x2.get("shape")))
    x2_dtype = input_x2.get("dtype")

    if len(x1_shape) != len(x2_shape):
        raise RuntimeError(
            "The shape of input data must be the same.")

    if len(x1_shape)-1 > 3 or len(x1_shape)-1 < 2:
        raise RuntimeError(
            "The number of dim of input data must equal to 2 or 3: got %d." %
            (len(x1_shape)-1))

    for pos, _ in enumerate(x1_shape):
        if x1_shape[pos] != x2_shape[pos]:
            raise RuntimeError(
                "The shape of x1 and x2 must be equal: got %d and %d in %d dim." %
                (x1_shape[pos], x2_shape[pos], pos))

    input_grad["rel_pos_to_reduce"] = "before"
    input_x1["rel_pos_to_reduce"] = "before"
    input_x2["rel_pos_to_reduce"] = "before"
    input_cdist["rel_pos_to_reduce"] = "before"

    axis = [-2]
    input_axis = {"shape": [len(axis), ], "value": axis, "rel_pos_to_reduce": "axis"}

    # initialize data
    schedules = []
    ins = classify([input_grad, input_x1, input_x2, input_cdist, input_axis], OpPatternMode.REDUCE,
        {"keepdims": False})
    tensors = []

    for (_input_grad, _input_x1, _input_x2, _input_cdist, _input_axis) in ins:
        with tbe.compute():
            shape_grad, shape_x1, shape_x2, shape_cdist = shape_util.variable_shape(
                [_input_grad, _input_x1, _input_x2, _input_cdist, _input_axis], op_mode="reduce"
            )[:4]
            x1_data = tvm.placeholder(shape_x1, name="data_x1", dtype=x1_dtype)
            x2_data = tvm.placeholder(shape_x2, name="data_x2", dtype=x2_dtype)
            grad_data = tvm.placeholder(shape_grad, name="data_grad",
                                        dtype=grad_dtype)
            cdist_data = tvm.placeholder(shape_cdist, name="data_cdist",
                                        dtype=cdist_dtype)
            res = cdist_grad_compute(grad_data, x1_data, x2_data, cdist_data, _input_axis, p)
            tensors.append([grad_data, x1_data, x2_data, cdist_data, res])
        with tvm.target.cce():
            schedule = tbe.auto_schedule(res)
        schedules.append(schedule)

    config = {"name": kernel_name,
            "tensor_list": tensors}
    tbe.build(schedules, config)
