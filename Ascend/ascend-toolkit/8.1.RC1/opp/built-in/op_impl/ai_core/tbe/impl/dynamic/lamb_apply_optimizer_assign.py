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
lamb_apply_optimizer_assign
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


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=locally-disabled,too-many-locals,unused-variable
# 'pylint: disable=too-many-statements
class Constant:
    """
    The class for constant
    """
    NUM = 12
    NUM_ZERO = 0


def shape_broadcast(data_1, data_2):
    """
    broadcast the two input

    Parameters
    ----------
    data_1: TVM tensor
        the placeholder of first input data
    data_2: TVM tensor
        the placeholder of second input data

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


def pow_compute(input_x, input_y):
    """
    compute result of pow when data_x is more than 0,
    use exp(y * ln(x)).
    """
    log_value = tbe.vlog(input_x)
    mul_value = tbe.vmul(input_y, log_value)
    res = tbe.vexp(mul_value)
    return res


@register_operator_compute("LambApplyOptimizerAssign", op_mode="dynamic", support_fusion=False)
def lamb_apply_optimizer_assign_compute(grad, input_v, input_m, input_param,
                                        beta_1, one_minus_beta_1,
                                        beta_2, one_minus_beta_2, epsilon,
                                        steps, do_use_weight, weight_decay_rate,
                                        update, v_out, m_out, kernel_name="lamb_apply_optimizer_assign"):
    """
    apply one lamb calculation function

    Parameters
    ----------
    grad: TVM tensor
         the input tensor of grad
    input_v: TVM tensor
         the input tensor of input_v
    input_m: TVM tensor
         the input tensor of input_m
    input_param: TVM tensor
         the input tensor of input_param
    beta_1: TVM tensor
         the input tensor of beta_1
    one_minus_beta_1: TVM tensor
         the input tensor of one_minus_beta_1
    beta_2: TVM tensor
         the input tensor of beta_2
    one_minus_beta_2: TVM tensor
         the input tensor of one_minus_beta_2
    epsilon: TVM tensor
         the input tensor of epsilon
    steps: TVM tensor
         the input tensor of steps
    update: TVM tensor
         the output tensor of update
    kernel_name : str
        kernel name, default value is "lamb_apply_optimizer_assign"

    Returns
    -------
    output tensor
    """
    # compute next_v
    square_grad = tbe.vmul(grad, grad)
    # mul_3
    square_grad, one_minus_beta_2 = shape_broadcast(square_grad, one_minus_beta_2)
    mul_3_result = tbe.vmul(square_grad, one_minus_beta_2)
    # mul_2
    input_v, beta_2 = shape_broadcast(input_v, beta_2)
    mul_2_result = tbe.vmul(input_v, beta_2)
    # compute: `next_v = (tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,tf.square(grad)))`
    mul_2_result, mul_3_result = shape_broadcast(mul_2_result, mul_3_result)
    next_v = tbe.vadd(mul_2_result, mul_3_result)

    # compute next_m
    input_m, beta_1 = shape_broadcast(input_m, beta_1)
    mul_0_result = tbe.vmul(input_m, beta_1)
    # mul_1
    grad, one_minus_beta_1 = shape_broadcast(grad, one_minus_beta_1)
    mul_1_result = tbe.vmul(grad, one_minus_beta_1)
    # compute: `next_m = (tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))`
    mul_0_result, mul_1_result = shape_broadcast(mul_0_result, mul_1_result)
    next_m = tbe.vadd(mul_0_result, mul_1_result)

    # compute: `beta1_correction = (1 - self.beta_1 ** steps)`
    beta_1, steps = shape_broadcast(beta_1, steps)
    beta_1_steps = pow_compute(beta_1, steps)
    neg_beta_1_step = tbe.vmuls(beta_1_steps, -1)
    beta1_correction = tbe.vadds(neg_beta_1_step, 1)

    # compute: `beta2_correction = (1 - self.beta_1 ** steps)`
    beta_2, steps = shape_broadcast(beta_2, steps)
    beta_2_steps = pow_compute(beta_2, steps)
    neg_beta_2_step = tbe.vmuls(beta_2_steps, -1)
    beta2_correction = tbe.vadds(neg_beta_2_step, 1)

    # compute: `next_m_unbiased = next_m / beta1_correction`
    next_m, beta1_correction = shape_broadcast(next_m, beta1_correction)
    next_m_unbiased = tbe.vdiv(next_m, beta1_correction)
    # compute: `next_v_unbiased = next_v / beta2_correction`
    next_v, beta2_correction = shape_broadcast(next_v, beta2_correction)
    next_v = tbe.vmuls(next_v, tvm.const(1, dtype=next_v.dtype))
    next_v_unbiased = tbe.vdiv(next_v, beta2_correction)

    # compute update
    sqrt_next_v = tbe.vsqrt(next_v_unbiased)
    # add_2
    epsilon, sqrt_next_v = shape_broadcast(epsilon, sqrt_next_v)
    add_2_result = tbe.vadd(sqrt_next_v, epsilon)
    # compute: `update = next_m / (tf.sqrt(next_v) + self.epsilon)`
    next_m_unbiased, add_2_result = shape_broadcast(next_m_unbiased, add_2_result)
    update = tbe.vdiv(next_m_unbiased, add_2_result)

    # compute do_use_weight_decay
    input_param, weight_decay_rate = shape_broadcast(input_param, weight_decay_rate)
    do_use_weight_mul = tbe.vmul(input_param, weight_decay_rate)
    do_use_weight_mul, do_use_weight = shape_broadcast(do_use_weight_mul, do_use_weight)
    do_use_weight_decay = tbe.vmul(do_use_weight_mul, do_use_weight)
    do_use_weight_decay, update = shape_broadcast(do_use_weight_decay, update)
    update = tbe.vadd(do_use_weight_decay, update)
    res = [update, next_v, next_m]
    return res


@register_operator("LambApplyOptimizerAssign")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def lamb_apply_optimizer_assign(grad, inputv, inputm, input_param,
                                beta_1, one_minus_beta_1, beta_2, one_minus_beta_2, epsilon, steps, do_use_weight,
                                weight_decay_rate, update, v_out, m_out, kernel_name="lamb_apply_optimizer_assign"):
    """
    function: For bert fuse

    Parameters
    ----------
    grad: dict
         the dict of input of grad, and dtype supports 'float16', 'float32'
    inputv: dict
         the dict of input of inputv, and dtype supports 'float16', 'float32'
    inputm: dict
         the dict of input of inputm, and dtype supports 'float16', 'float32'
    input_param: dict
         the dict of input of input_param, and dtype supports 'float16', 'float32'
    beta_1: dict
         the dict of input of beta_1, and dtype supports 'float16', 'float32'
    one_minus_beta_1: dict
         the dict of input of one_minus_beta_1, and dtype supports 'float16', 'float32'
    beta_2: dict
         the dict of input of beta_2, and dtype supports 'float16', 'float32'
    one_minus_beta_2: dict
         the dict of input of one_minus_beta_2, and dtype supports 'float16', 'float32'
    epsilon: dict
         the dict of input of epsilon, and dtype supports 'float16', 'float32'
    steps: dict
         the dict of input of steps, and dtype supports 'float16', 'float32'
    update: dict
         the dict of output of update, and dtype supports 'float16', 'float32'
    v_out: dict
         the dict of output of update, and dtype supports 'float16', 'float32'
    m_out: dict
         the dict of output of update, and dtype supports 'float16', 'float32'
    kernel_name: str
        cce kernel name, default value is lamb_apply_optimizer_assign

    Returns
    -------
    None
    """
    dtype_grad = grad.get("dtype").lower()
    data_dict = {"grad": 0, "inputv": 1, "inputm": 2, "input_param": 3,
                 "beta_1": 4, "one_minus_beta_1": 5, "beta_2": 6,
                 "one_minus_beta_2": 7, "epsilon": 8, "steps": 9, "do_use_weight": 10, "weight_decay_rate": 11}
    data_inputs = [None] * Constant.NUM
    dynamic_inputs = [grad, inputv, inputm, input_param, beta_1, one_minus_beta_1, beta_2, one_minus_beta_2, epsilon,
                      steps, do_use_weight, weight_decay_rate]
    ins = classify(dynamic_inputs, OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    data_names = ["grad", "inputv", "inputm", "input_param", "beta_1", "one_minus_beta_1", "beta_2", "one_minus_beta_2",
                  "epsilon", "steps", "do_use_weight", "weight_decay_rate"]
    for _dinputs in ins:
        with tbe.compute():
            shape_dinputs = shape_util.variable_shape(_dinputs)
            idx = Constant.NUM_ZERO
            for shape_dinput in shape_dinputs:
                data_inputs[idx] = tvm.placeholder(shape_dinput,
                                                   name=data_names[idx],
                                                   dtype=dtype_grad)
                idx += 1

            res = lamb_apply_optimizer_assign_compute(data_inputs[data_dict.get("grad")],
                                                      data_inputs[data_dict.get("inputv")],
                                                      data_inputs[data_dict.get("inputm")],
                                                      data_inputs[data_dict.get("input_param")],
                                                      data_inputs[data_dict.get("beta_1")],
                                                      data_inputs[data_dict.get("one_minus_beta_1")],
                                                      data_inputs[data_dict.get("beta_2")],
                                                      data_inputs[data_dict.get("one_minus_beta_2")],
                                                      data_inputs[data_dict.get("epsilon")],
                                                      data_inputs[data_dict.get("steps")],
                                                      data_inputs[data_dict.get("do_use_weight")],
                                                      data_inputs[data_dict.get("weight_decay_rate")],
                                                      update, v_out, m_out, kernel_name)
            tensors.append(data_inputs + list(res))
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors,
              "dummy_placeholder": True}
    tbe.build(schedules, config)
