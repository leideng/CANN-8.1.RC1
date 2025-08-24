#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2019 Huawei Technologies Co., Ltd
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
softmax_grad_ext
"""
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute
from impl import constant_util as constant
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.dynamic.softmax_grad_ext import op_select_format as softmax_grad_ext_op_select_format


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=locally-disabled,too-many-locals,unused-variable
# 'pylint: disable=locally-disabled,invalid-name,unidiomatic-typecheck
# 'pylint: disable=locally-disabled,too-many-branches
# 'pylint: disable=unnecessary-comprehension
def op_select_format(grad, x1, x2, y, axis, keepdims,
                     kernel_name="softmax_grad_ext"):
    """select format dynamically"""
    return softmax_grad_ext_op_select_format(grad, x1, x2, y, axis, keepdims, kernel_name)


def _check_nz_rule(grad, x1, x2, axis):
    # 'pylint: disable = unused-variable
    shape_grad = shape_util.scalar2tensor_one(grad.get("shape"))
    # 'pylint: disable = unused-variable
    shape_x1 = shape_util.scalar2tensor_one(x1.get("shape"))
    shape_x2 = shape_util.scalar2tensor_one(x2.get("shape"))

    ori_shape = shape_util.scalar2tensor_one(grad.get("ori_shape"))

    format_grad = grad.get("format")
    format_x1 = x1.get("format")
    format_x2 = x2.get("format")

    format_target = [["FRACTAL_NZ", "FRACTAL_NZ", "ND"],
                     ["FRACTAL_NZ", "FRACTAL_NZ", "NCHW"],
                     ["FRACTAL_NZ", "FRACTAL_NZ", "NHWC"]]
    format_list = [format_grad, format_x1, format_x2]

    if format_list not in format_target:
        error_manager_vector.raise_err_specific_reson("softmax_grad_ext", "Combination of format is illegal in nz+nd")

    if not(len(shape_x2) == 1 and shape_x2[0] == 1):
        error_manager_vector.raise_err_specific_reson("softmax_grad_ext", "the last input tensor should be scalar")

    forward = list(i for i in range(len(ori_shape)))
    back_forwad = [i-len(ori_shape) for i in range(len(ori_shape))]
    if type(axis) in [list, tuple]:
        axis = list(axis)
        flag_0 = False
        flag_1 = False
        axis_new = []
        for value in axis:
            if value in [forward[-1], back_forwad[-1]]:
                flag_0 = True
            elif value in [forward[-2], back_forwad[-2]]:
                flag_1 = True
            else:
                axis_new.append(value)

        if flag_0:
            axis_new.append(-1)
            axis_new.append(-4)
        if flag_1:
            axis_new.append(-2)
            axis_new.append(-3)
    else:
        if axis >= 0:
            if axis == forward[-1]:
                axis_new = [-1, -4]
            elif axis == forward[-2]:
                axis_new = [-2, -3]
            else:
                axis_new = axis

        else:
            if axis == back_forwad[-1]:
                axis_new = [-1, -4]
            elif axis == back_forwad[-2]:
                axis_new = [-2, -3]
            else:
                axis_new = axis

    return axis_new


def shape_broadcast(data_1, data_2):
    """broadcast the two input

    Parameters
    ----------
    data_1: TVM tensor
        the placeholder of first input data
    data_2: TVM tensor
        the placeholder of second input data
    output_z: dict
        shape and dtype of output, should be broadcast shape and type as input

    Returns
    -------
    res : output of the data's divide
    """
    shape_x = shape_util.shape_to_list(data_1.shape)
    shape_y = shape_util.shape_to_list(data_2.shape)
    if shape_x != shape_y:
        shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x, shape_y,
                                                                  param_name_input1="data_1",
                                                                  param_name_input2="data_2")
        data_1 = _broadcast_nz(data_1, shape_max)
        data_2 = _broadcast_nz(data_2, shape_max)

    return data_1, data_2


def _broadcast_nz(tensor, shape):
    broadcast_axes = []
    src_shape = shape_util.shape_to_list(tensor.shape)
    for i, _ in enumerate(shape):
        if shape[i] != src_shape[i]:
            broadcast_axes.append(i)
    if len(broadcast_axes) == 2 and \
            broadcast_axes[1] - broadcast_axes[0] != 1 and \
            broadcast_axes[1] + 1 == len(shape):
        temp_shape = src_shape[:-1] + [shape[-1]]
        tensor = tbe.broadcast(tensor, temp_shape)
    tensor = tbe.broadcast(tensor, shape)
    return tensor


@register_operator_compute("softmax_grad_ext", op_mode="static", support_fusion=True)
# 'pylint: disable = unused-argument
def softmax_grad_ext_compute(data_grad, data_x1, data_x2,
                             y, axis, keepdims,
                             kernel_name="softmax_grad_ext"):
    """apply one adam calculation function

    Parameters
    ----------
    data_grad: TVM tensor
         the input tensor of mul and sub
    data_x1: TVM tensor
         the input tensor of mul and mul_1
    data_x2: TVM tensor
         the input tensor of mul_1
    y: dict
         the output tensor of mul_grad
    axis: int, list, tuple
        the axis for reduce.
    keepdims: bool
        if true, retains reduced dimensions with length 1.
    kernel_name : str
        kernel name, default value is "softmax_grad_ext"

    Returns
    -------
    output tensor
    """

    # mul
    data_grad, data_x1 = shape_broadcast(data_grad, data_x1)
    mul_result = tbe.vmul(data_grad, data_x1)

    # sum
    dtype = mul_result.dtype
    if dtype == "float16":
        mul_result = tbe.cast_to(mul_result, "float32")
    sum_result = tbe.sum(mul_result, axis=axis, keepdims=keepdims)
    if dtype == "float16":
        sum_result = tbe.cast_to(sum_result, "float16")

    # sub
    data_grad, sum_result = shape_broadcast(data_grad, sum_result)
    sub_result = tbe.vsub(data_grad, sum_result)

    # mul_1
    data_x1, data_x2 = shape_broadcast(data_x1, data_x2)
    mul_1_result = tbe.vmul(data_x1, data_x2)

    # mul_grad
    sub_result, mul_1_result = shape_broadcast(sub_result, mul_1_result)
    mul_grad_result = tbe.vmul(sub_result, mul_1_result)
    res = [mul_grad_result]

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            (para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_LIST_INT),
                            para_check.REQUIRED_ATTR_BOOL, para_check.KERNEL_NAME)
def softmax_grad_ext(grad, x1, x2, y, axis, keepdims,
                     kernel_name="softmax_grad_ext"):
    """function: softmax_grad_ext

    Parameters
    ----------
    grad: dict
         the input tensor of mul and sub
    x1: dict
         the input tensor of mul and mul_1
    x2: dict
         the input tensor of mul_1
    y: dict
         the output tensor of mul_grad
    axis: int, list, tuple
        the axis for reduce.
    keepdims: bool
        if true, retains reduced dimensions with length 1.
    kernel_name : str
        kernel name, default value is "softmax_grad_ext"

    Returns
    -------
    None
    """
    shape_grad = shape_util.scalar2tensor_one(grad.get("shape"))
    shape_x1 = shape_util.scalar2tensor_one(x1.get("shape"))
    shape_x2 = shape_util.scalar2tensor_one(x2.get("shape"))

    dtype_grad = grad.get("dtype").lower()
    dtype_x1 = x1.get("dtype").lower()
    dtype_x2 = x2.get("dtype").lower()

    if grad.get("format") == "FRACTAL_NZ":
        axis = _check_nz_rule(grad, x1, x2, axis)

    shape_grad, shape_x1, shape_max_mul = \
        shape_util.broadcast_shapes(shape_grad, shape_grad, param_name_input1="grad",
                                    param_name_input2="grad")
    shape_x1, shape_x2, shape_max_mul1 = \
        shape_util.broadcast_shapes(shape_x1, shape_x2, param_name_input1="x1",
                                    param_name_input2="x2")

    data_grad = tvm.placeholder(shape_grad,
                                name="data_grad",
                                dtype=dtype_grad)
    data_x1 = tvm.placeholder(shape_x1,
                              name="data_x1",
                              dtype=dtype_x1)
    data_x2 = tvm.placeholder(shape_x2,
                              name="data_x2",
                              dtype=dtype_x2)

    res = softmax_grad_ext_compute(data_grad, data_x1, data_x2,
                                   y, axis, keepdims, kernel_name)

    inputlist = [data_grad, data_x1, data_x2]

    with tvm.target.cce():
        sch = auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": list(inputlist) + list(res)}

    build(sch, config)
