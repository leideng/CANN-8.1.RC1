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
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util_soc_common import is_v200


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=locally-disabled,too-many-locals,unused-variable
# 'pylint: disable=relative-beyond-top-level
class Constant:
    """
    The class for constant
    """
    # min float32 value
    MIN_FP32 = 2 ** (-126)
    # min float16 value
    MIN_FP16 = 2 ** (-24)
    NUM_FIVE = 5
    NUM_ZERO = 0


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
            shape_util.broadcast_shapes(data_1.shape,
                                        data_2.shape,
                                        param_name_input1="data_1",
                                        param_name_input2="data_2")
        data_1 = tbe.broadcast(data_1, shape_max)
        data_2 = tbe.broadcast(data_2, shape_max)
    return data_1, data_2


def _greater_compare(data, shape, dtype, data_min):
    """
    if x is greater than y, then return 1, else return 0.

    Parameters:
    ----------
    data : tuple
        two input data
    shape : list or tuple
        shape of input data
    dtype : str
        source data type, support float16,float32,int32,int8,uint8
    data_min : tvm.const
        the minimal data according to dtype

    Returns
    -------
    the compare result
    """
    data_zero = tbe.broadcast(tvm.const(0, dtype), shape, dtype)

    data_one = tbe.broadcast(tvm.const(1, dtype), shape, dtype)

    res_sub = tbe.vsub(data[1], data[0])
    # to amend sub zero result
    res_sub_zero = tbe.vadd(res_sub, data_min)
    res_min = tbe.vmin(res_sub_zero, data_min)
    res_max = tbe.vmax(res_min, data_zero)

    if dtype == "float32":
        # max num of float32 is 2**126
        # but cce can only support 2**62, so use 62/62/2 to adaptor 126
        max_support_fp32 = tvm.const(2 ** 62, dtype=dtype)
        res_mul1 = tbe.vmuls(res_max, max_support_fp32)
        res_mul2 = tbe.vmuls(res_mul1, max_support_fp32)
        res_mul = tbe.vmuls(res_mul2, tvm.const(2 ** 2, dtype=dtype))
    else:
        # max num of float16 is 2**24
        # but cce can only support 2**12, so use 12/12 to adaptor 24
        max_support_fp16 = tvm.const(2 ** 12, dtype=dtype)
        res_mul1 = tbe.vmuls(res_max, max_support_fp16)
        res_mul = tbe.vmuls(res_mul1, max_support_fp16)

    res = tbe.vsub(data_one, res_mul)
    return res


def greater_compute(data_x, data_y, shape, dtype):
    """
    if x is greater than y, then return 1, else return 0.

    Parameters:
    ----------
    x : Tensor
        input data_x
    y : Tensor
        input data_y
    z : dict
        shape and dtype of output data_z
    kernel_name : str
        cce kernel name, default value is "greater"

    Returns
    -------
    the result
    """

    if dtype == "float32":
        # minimun num of float32 2**(-126)
        data_min = tbe.broadcast(tvm.const(Constant.MIN_FP32, dtype=dtype),
                                 shape, dtype)
    else:
        data_min = tbe.broadcast(tvm.const(Constant.MIN_FP16, dtype=dtype),
                                 shape, dtype)

    return _greater_compare((data_x, data_y), shape, dtype, data_min)


def compute_ratio_vcmp(w_norm, g_norm, data_zero, data_one, w_norm_g_norm):
    """
    compute :`ratio = array_ops.where(math_ops.greater(w_norm, 0)`, `array_ops.where(math_ops.greater(g_norm, 0)`,
    `(w_norm / g_norm), 1.0), 1.0) by vcmpsel interface`
    :param w_norm:
    :param g_norm:
    :param w_norm:
    :param g_norm:
    :param g_norm:
    :return: ratio
    """
    greater_g_norm_zero = tbe.vcmpsel(g_norm, data_zero, 'gt', w_norm_g_norm, data_one)
    ratio = tbe.vcmpsel(w_norm, data_zero, 'gt', greater_g_norm_zero, data_one)
    return ratio


def compute_ratio(w_norm, g_norm):
    """
    compute :`ratio = array_ops.where(math_ops.greater(w_norm, 0)`, `array_ops.where(math_ops.greater(g_norm, 0)`,
    `(w_norm / g_norm), 1.0), 1.0)`
    :param w_norm:
    :param g_norm:
    :return: ratio
    """
    g_norm_shape = shape_util.shape_to_list(g_norm.shape)
    dtype = w_norm.dtype

    shape_x = shape_util.shape_to_list(w_norm.shape)
    shape_x, shape_y, shape = shape_util.broadcast_shapes(shape_x, g_norm_shape,
                                                          param_name_input1="x",
                                                          param_name_input2="y")
    w_norm = tbe.broadcast(w_norm, shape)
    g_norm = tbe.broadcast(g_norm, shape)
    data_zero = tbe.broadcast(tvm.const(0, dtype), shape, dtype)
    data_one = tbe.broadcast(tvm.const(1, dtype), shape, dtype)
    g_norm_tmp = tbe.vadd(g_norm, data_one)
    g_norm_plus = tbe.vcmpsel(g_norm, data_zero, 'eq', g_norm_tmp, g_norm)
    w_norm_g_norm = tbe.vdiv(w_norm, g_norm_plus)
    if is_v200():
        return compute_ratio_vcmp(w_norm, g_norm, data_zero, data_one, w_norm_g_norm)

    greater_g_norm_zero = greater_compute(g_norm, data_zero, shape, dtype)
    greater_w_norm_zero = greater_compute(w_norm, data_zero, shape, dtype)
    select_1 = select_compute(greater_g_norm_zero, w_norm_g_norm, data_one)

    # compute ratio
    ratio = select_compute(greater_w_norm_zero, select_1, data_one)
    return ratio


@register_operator_compute("LambApplyWeightAssign", op_mode="dynamic", support_fusion=False)
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

    ratio = compute_ratio(w_norm, g_norm)  # w_norm g_norm yidui

    update, input_lr = shape_broadcast(update, input_lr)
    update_with_lr = tbe.vmul(update, input_lr)
    ratio, update_with_lr = shape_broadcast(ratio, update_with_lr)
    ratio_update_with_lr = tbe.vmul(ratio, update_with_lr)

    ratio_update_with_lr, input_param = shape_broadcast(ratio_update_with_lr, input_param)
    next_param = tbe.vsub(input_param, ratio_update_with_lr)

    return next_param


@register_operator("LambApplyWeightAssign")
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
    dtype_w_norm = w_norm.get("dtype").lower()
    data_dict = {"w_norm": 0, "g_norm": 1, "inputlr": 2,
                 "update": 3, "input_param": 4}
    data_inputs = [None] * Constant.NUM_FIVE
    dynamic_inputs = [w_norm, g_norm, inputlr, update, input_param]
    ins = classify(dynamic_inputs, OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    data_names = ["w_norm", "g_norm", "inputlr", "update", "input_param"]
    for _dinputs in ins:
        with tbe.compute():
            shape_dinputs = shape_util.variable_shape(_dinputs)
            idx = Constant.NUM_ZERO
            for shape_dinput in shape_dinputs:
                data_inputs[idx] = tvm.placeholder(shape_dinput,
                                                   name=data_names[idx],
                                                   dtype=dtype_w_norm)
                idx += 1

            res = lamb_apply_weight_assign_compute(data_inputs[data_dict.get("w_norm")],
                                                   data_inputs[data_dict.get("g_norm")],
                                                   data_inputs[data_dict.get("inputlr")],
                                                   data_inputs[data_dict.get("update")],
                                                   data_inputs[data_dict.get("input_param")], out_param, kernel_name)
            tensors.append(data_inputs + [res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors,
              "dummy_placeholder": True}
    tbe.build(schedules, config)
