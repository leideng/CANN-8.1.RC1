#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2020 Huawei Technologies Co., Ltd
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
lamb_apply_weight_assign
"""
import uuid

from tbe.dsl.api import auto_schedule
from tbe.dsl.api import build
import te.lang.cce as tbe
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from .greater import greater_compute


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=locally-disabled,too-many-locals,unused-variable
# 'pylint: disable=relative-beyond-top-level
def select_compute(condition, data_x=None, data_y=None):
    """
    select data from data_x or data_y according to the condition.
    :param condition: TVM tensor
    :param data_x: TVM tensor
    :param data_y: TVM tensor
    :return: select results
    """
    select_a = tbe.vmul(condition, data_x)
    neg_condition = tbe.vmuls(condition, -1)
    select_b = tbe.vadds(neg_condition, 1)
    select_b = tbe.vmul(select_b, data_y)
    res = tbe.vadd(select_a, select_b)
    return res


def shape_broadcast(data_1, data_2):
    """
    broadcast the two input

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
        shape_x, shape_y, shape_max = \
            shape_util.broadcast_shapes(shape_x, shape_y, param_name_input1="data_1", param_name_input2="data_2")
        data_1 = tbe.broadcast(data_1, shape_max)
        data_2 = tbe.broadcast(data_2, shape_max)

    return data_1, data_2


def compute_ratio(w_norm, g_norm):
    """
    compute :`ratio = array_ops.where(math_ops.greater(w_norm, 0)`, `array_ops.where(math_ops.greater(g_norm, 0)`,
    `(w_norm / g_norm), 1.0), 1.0)`
    :param w_norm:
    :param g_norm:
    :return: ratio
    """
    g_norm_shape = shape_util.shape_to_list(g_norm.shape)
    dtype = 'float32'
    data_zero = tbe.broadcast(tvm.const(0, dtype), g_norm_shape, dtype)
    scalar_zero = tvm.const(0, "float32")
    greater_g_norm_zero = greater_compute(g_norm, data_zero, scalar_zero)
    greater_w_norm_zero = greater_compute(w_norm, data_zero, scalar_zero)

    w_norm_g_norm = tbe.vdiv(w_norm, g_norm)
    data_one = tbe.broadcast(tvm.const(1, dtype), g_norm_shape, dtype)
    select_1 = select_compute(greater_g_norm_zero, w_norm_g_norm, data_one, )

    # compute ratio
    ratio = select_compute(greater_w_norm_zero, select_1, data_one, )

    return ratio


@register_operator_compute("lamb_apply_weight_assign", op_mode="static", support_fusion=True)
def lamb_apply_weight_assign_compute(w_norm, g_norm, input_lr, update, input_param, output_param,
                                     kernel_name="lamb_apply_weight_assign"):
    """
    apply one lamb calculation function

    Parameters
    ----------
    w_norm: TVM tensor
         the input tensor of w_norm
    g_norm: TVM tensor
         the input tensor of g_norm
    input_lr: TVM tensor
         the input tensor of input_lr
    update: TVM tensor
         the input tensor of update
    input_param: TVM tensor
         the input tensor of input_param
    kernel_name : str
        kernel name, default value is "lamb_apply_weight_assign"

    Returns
    -------
    output tensor
    """
    # compute ratio

    ratio = compute_ratio(w_norm, g_norm)

    update, input_lr = shape_broadcast(update, input_lr)
    update_with_lr = tbe.vmul(update, input_lr)
    ratio, update_with_lr = shape_broadcast(ratio, update_with_lr)
    ratio_update_with_lr = tbe.vmul(ratio, update_with_lr)

    ratio_update_with_lr, input_param = shape_broadcast(ratio_update_with_lr, input_param)
    next_param = tbe.vsub(input_param, ratio_update_with_lr)

    res = [next_param]

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def lamb_apply_weight_assign(w_norm, g_norm, inputlr,
                             update, input_param, out_param, kernel_name="lamb_apply_weight_assign"):
    """
    function: For bert fuse

    Parameters
    ----------
    w_norm: dict
         the dict of input of w_norm, and dtype supports 'float16', 'float32'
    g_norm: dict
         the dict of input of g_norm, and dtype supports 'float16', 'float32'
    inputlr: dict
         the dict of input of inputlr, and dtype supports 'float16', 'float32'
    update: dict
         the dict of input of update, and dtype supports 'float16', 'float32'
    input_param: dict
         the dict of input of input_param, and dtype supports 'float16', 'float32'
    out_param: dict
         the dict of input of input_param, and dtype supports 'float16', 'float32'
    kernel_name: str
        cce kernel name, default value is lamb_apply_weight_assign

    Returns
    -------
    None
    """
    shape_w_norm = shape_util.scalar2tensor_one(w_norm.get("shape"))
    shape_g_norm = shape_util.scalar2tensor_one(g_norm.get("shape"))
    shape_inputlr = shape_util.scalar2tensor_one(inputlr.get("shape"))
    shape_update = shape_util.scalar2tensor_one(update.get("shape"))
    shape_input_param = shape_util.scalar2tensor_one(input_param.get("shape"))

    dtype_w_norm = w_norm.get("dtype").lower()
    dtype_g_norm = g_norm.get("dtype").lower()
    dtype_inputlr = inputlr.get("dtype").lower()
    dtype_update = update.get("dtype").lower()
    dtype_input_param = input_param.get("dtype").lower()

    shape_g_norm, shape_input_param, shape_max_mul1 = \
        shape_util.broadcast_shapes(shape_g_norm, shape_input_param, param_name_input1="g_norm",
                                    param_name_input2="input_param")
    shape_inputlr, shape_input_param, shape_max_mul1 = \
        shape_util.broadcast_shapes(shape_inputlr, shape_input_param, param_name_input1="inputlr",
                                    param_name_input2="input_param")
    shape_update, shape_input_param, shape_max_mul1 = \
        shape_util.broadcast_shapes(shape_update, shape_input_param, param_name_input1="update",
                                    param_name_input2="input_param")
    shape_w_norm, shape_input_param, shape_max_mul1 = \
        shape_util.broadcast_shapes(shape_w_norm, shape_input_param, param_name_input1="w_norm",
                                    param_name_input2="input_param")

    if kernel_name == "lamb_apply_weight_assign":
        kernel_name += \
            str(uuid.uuid4()) + str(uuid.uuid4()) + str(uuid.uuid4()) + str(uuid.uuid4())
        kernel_name = kernel_name.replace('-', 'Z')
    w_norm = tvm.placeholder(shape_w_norm,
                             name="w_norm__" + kernel_name,
                             dtype=dtype_w_norm)
    g_norm = tvm.placeholder(shape_g_norm,
                             name="g_norm__" + kernel_name,
                             dtype=dtype_g_norm)
    input_lr = tvm.placeholder(shape_inputlr,
                               name="input_lr__" + kernel_name,
                               dtype=dtype_inputlr)
    update = tvm.placeholder(shape_update,
                             name="update__" + kernel_name,
                             dtype=dtype_update)
    input_param = tvm.placeholder(shape_input_param,
                                  name="input_param__" + kernel_name,
                                  dtype=dtype_input_param)

    res = lamb_apply_weight_assign_compute(w_norm, g_norm, input_lr, update, input_param, out_param, kernel_name)

    inputlist = [w_norm, g_norm, input_lr, update, input_param]

    with tvm.target.cce():
        sch = auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": list(inputlist) + list(res),
              "dummy_placeholder": True}

    build(sch, config)
