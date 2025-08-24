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
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util


#Power calculation
def tensor_pow(data, p):
    """
    tensor_pow
    """
    log_value = tbe.vlog(data)
    return tbe.vexp(tbe.vmuls(log_value, p))


# 'pylint: disable=unused-argument,too-many-arguments,too-many-locals
@register_operator_compute("cdist_grad", op_mode="static", support_fusion=True)
def cdist_grad_compute(input_grad, input_x1, input_x2, input_cdist, p,
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
    kernel_name : str
        kernel name, default value is "cdist"

    Returns
    -------
    :return: TVM tensor
        result tensor
    """
    dtype = input_x1.dtype

    if dtype == 'float16':
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

    if p == 0.0:
        # If p is 0, the result is a zero tensor.
        pass
    elif p == 1.0:
        res = tbe.vmul(input_grad, sign)
    elif p < 2.0:
        power = tvm.const(p - 1, dtype=dtype)

        num = tbe.vmul(sign, tensor_pow(diff_abs, power))
        numerator = tbe.vmul(num, input_grad)
        denominator = tensor_pow(nz_cdist, power)

        res = tbe.vdiv(numerator, denominator)
        res = tbe.vcmpsel(input_cdist, zero, 'eq', slhs=zero, srhs=res)
    elif p == 2.0:
        numerator = tbe.vmul(input_grad, diff)
        res = tbe.vdiv(numerator, nz_cdist)
        res = tbe.vcmpsel(input_cdist, zero, 'eq', slhs=zero, srhs=res)
    elif p == float("inf"):
        mask = tbe.vcmpsel(input_cdist, diff_abs, 'gt', slhs=zero,
                           srhs=one)
        grad_sign = tbe.vmul(input_grad, sign)
        res = tbe.vmul(grad_sign, mask)
    else:
        power1 = tvm.const(p - 1, dtype=dtype)
        power2 = tvm.const(p - 2, dtype=dtype)

        num = tbe.vmul(diff, tensor_pow(diff_abs, power2))
        numerator = tbe.vmul(num, input_grad)
        denominator = tensor_pow(nz_cdist, power1)

        res = tbe.vdiv(numerator, denominator)
        res = tbe.vcmpsel(input_cdist, zero, 'eq', slhs=zero, srhs=res)

    if p == 0:
        ret = tbe.sum(z_3, axis=-2)
    else:
        ret = tbe.sum(tbe.vadd(res, z_3), axis=-2)

    if dtype == 'float16':
        return tbe.cast_to(ret, 'float16')
    return ret


# 'pylint: disable=unused-argument,too-many-arguments,too-many-locals
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

    grad_shape = shape_util.scalar2tensor_one(input_grad.get("shape"))
    grad_dtype = input_grad.get("dtype")
    cdist_shape = shape_util.scalar2tensor_one(input_cdist.get("shape"))
    cdist_dtype = input_cdist.get("dtype")
    x1_shape = shape_util.scalar2tensor_one(input_x1.get("shape"))
    x1_dtype = input_x1.get("dtype")
    x2_shape = shape_util.scalar2tensor_one(input_x2.get("shape"))
    x2_dtype = input_x2.get("dtype")

    # check shape
    x1_shape = list(x1_shape)
    x2_shape = list(x2_shape)
    grad_shape = list(grad_shape)
    cdist_shape = list(cdist_shape)

    if len(x1_shape) != len(x2_shape):
        raise RuntimeError(
            "The shape of input data must be the same.")

    if len(x1_shape)-1 > 3 or len(x1_shape)-1 < 2:
        raise RuntimeError(
            "The number of dim of input data must equal to 2 or 3: got %d." %
            (len(x1_shape)-1))

    for idx, (dim_x1, dim_x2) in enumerate(zip(x1_shape, x2_shape)):
        if dim_x1 != dim_x2:
            raise RuntimeError(
                "The shape of x1 and x2 must be equal: got %d and %d in %d dim." %
                (dim_x1, dim_x2, idx))

    # initialize data
    x1_data = tvm.placeholder(x1_shape, name="data_x1", dtype=x1_dtype)
    x2_data = tvm.placeholder(x2_shape, name="data_x2", dtype=x2_dtype)
    grad_data = tvm.placeholder(grad_shape, name="data_grad",
                                dtype=grad_dtype)
    cdist_data = tvm.placeholder(cdist_shape, name="data_cdist",
                                 dtype=cdist_dtype)
    res = cdist_grad_compute(grad_data, x1_data, x2_data, cdist_data, p)

    with tvm.target.cce():
        schedule = auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [grad_data, x1_data, x2_data, cdist_data, res]}

    build(schedule, config)
