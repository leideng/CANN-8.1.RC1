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
import uuid

from impl.util.platform_adapter import build
from impl.util.platform_adapter import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=locally-disabled,too-many-locals,unused-variable
# 'pylint: disable=too-many-statements

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


def pow_compute(input_x, input_y):
    """
    compute result of pow when data_x is more than 0,
    use exp(y * ln(x)).
    """
    log_value = tbe.vlog(input_x)
    mul_value = tbe.vmul(input_y, log_value)
    res = tbe.vexp(mul_value)

    return res


@register_operator_compute("lamb_apply_optimizer_assign", op_mode="static", support_fusion=True)
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
    shape_grad = shape_util.scalar2tensor_one(grad.get("shape"))
    shape_inputv = shape_util.scalar2tensor_one(inputv.get("shape"))
    shape_inputm = shape_util.scalar2tensor_one(inputm.get("shape"))
    shape_input_param = shape_util.scalar2tensor_one(input_param.get("shape"))
    shape_beta_1 = shape_util.scalar2tensor_one(beta_1.get("shape"))
    shape_one_minus_beta_1 = shape_util.scalar2tensor_one(one_minus_beta_1.get("shape"))
    shape_beta_2 = shape_util.scalar2tensor_one(beta_2.get("shape"))
    shape_one_minus_beta_2 = shape_util.scalar2tensor_one(one_minus_beta_2.get("shape"))
    shape_epsilon = shape_util.scalar2tensor_one(epsilon.get("shape"))
    shape_steps = shape_util.scalar2tensor_one(steps.get("shape"))
    shape_do_use_weight = shape_util.scalar2tensor_one(do_use_weight.get("shape"))
    shape_weight_decay_rate = shape_util.scalar2tensor_one(weight_decay_rate.get("shape"))

    dtype_grad = grad.get("dtype").lower()
    dtype_inputv = inputv.get("dtype").lower()
    dtype_inputm = inputm.get("dtype").lower()
    dtype_input_param = input_param.get("dtype").lower()
    dtype_beta_1 = beta_1.get("dtype").lower()
    dtype_one_minus_beta_1 = one_minus_beta_1.get("dtype").lower()
    dtype_beta_2 = beta_2.get("dtype").lower()
    dtype_one_minus_beta_2 = one_minus_beta_2.get("dtype").lower()
    dtype_epsilon = epsilon.get("dtype").lower()
    dtype_steps = steps.get("dtype").lower()
    dtype_do_use_weight = do_use_weight.get("dtype").lower()
    dtype_weight_decay_rate = weight_decay_rate.get("dtype").lower()

    shape_inputm, shape_beta_1, shape_max_mul0 = \
        shape_util.broadcast_shapes(shape_inputm, shape_beta_1, param_name_input1="inputm", param_name_input2="beta_1")
    shape_grad, shape_one_minus_beta_1, shape_max_mul1 = \
        shape_util.broadcast_shapes(shape_grad, shape_one_minus_beta_1, param_name_input1="grad",
                                    param_name_input2="one_minus_beta_1")
    shape_inputv, shape_beta_2, shape_max_mul2 = \
        shape_util.broadcast_shapes(shape_inputv, shape_beta_2, param_name_input1="inputv", param_name_input2="beta_2")
    shape_grad, shape_one_minus_beta_2, shape_max_mul3 = \
        shape_util.broadcast_shapes(shape_grad, shape_one_minus_beta_2, param_name_input1="grad",
                                    param_name_input2="one_minus_beta_2")
    shape_inputv, shape_epsilon, shape_max_add2 = \
        shape_util.broadcast_shapes(shape_inputv, shape_epsilon, param_name_input1="inputv",
                                    param_name_input2="epsilon")
    shape_inputv, shape_steps, shape_max_add2 = \
        shape_util.broadcast_shapes(shape_inputv, shape_steps, param_name_input1="inputv", param_name_input2="epsilon")
    shape_inputv, shape_do_use_weight, shape_max_add2 = \
        shape_util.broadcast_shapes(shape_inputv, shape_do_use_weight, param_name_input1="inputv",
                                    param_name_input2="epsilon")
    shape_inputv, shape_weight_decay_rate, shape_max_add2 = \
        shape_util.broadcast_shapes(shape_inputv, shape_weight_decay_rate, param_name_input1="inputv",
                                    param_name_input2="epsilon")

    if kernel_name == "lamb_apply_optimizer_assign":
        kernel_name += \
            str(uuid.uuid4()) + str(uuid.uuid4()) + str(uuid.uuid4()) + str(uuid.uuid4())
        kernel_name = kernel_name.replace('-', 'Z')
    grad = tvm.placeholder(shape_grad,
                           name="grad__" + kernel_name,
                           dtype=dtype_grad)
    input_v = tvm.placeholder(shape_inputv,
                              name="input_v__" + kernel_name,
                              dtype=dtype_inputv)
    input_m = tvm.placeholder(shape_inputm,
                              name="input_m__" + kernel_name,
                              dtype=dtype_inputm)
    input_param = tvm.placeholder(shape_input_param,
                                  name="input_param__" + kernel_name,
                                  dtype=dtype_input_param)
    beta_1 = tvm.placeholder(shape_beta_1,
                             name="beta_1__" + kernel_name,
                             dtype=dtype_beta_1)
    one_minus_beta_1 = tvm.placeholder(shape_one_minus_beta_1,
                                       name="one_minus_beta_1__" + kernel_name,
                                       dtype=dtype_one_minus_beta_1)
    beta_2 = tvm.placeholder(shape_beta_2,
                             name="beta_2__" + kernel_name,
                             dtype=dtype_beta_2)
    one_minus_beta_2 = tvm.placeholder(shape_one_minus_beta_2,
                                       name="one_minus_beta_2__" + kernel_name,
                                       dtype=dtype_one_minus_beta_2)
    epsilon = tvm.placeholder(shape_epsilon,
                              name="epsilon__" + kernel_name,
                              dtype=dtype_epsilon)
    steps = tvm.placeholder(shape_steps,
                            name="steps__" + kernel_name,
                            dtype=dtype_steps)
    do_use_weight = tvm.placeholder(shape_do_use_weight,
                                    name="do_use_weight__" + kernel_name,
                                    dtype=dtype_do_use_weight)
    weight_decay_rate = tvm.placeholder(shape_weight_decay_rate,
                                        name="weight_decay_rate__" + kernel_name,
                                        dtype=dtype_weight_decay_rate)

    res = lamb_apply_optimizer_assign_compute(grad, input_v, input_m,
                                              input_param, beta_1,
                                              one_minus_beta_1, beta_2,
                                              one_minus_beta_2, epsilon,
                                              steps, do_use_weight, weight_decay_rate,
                                              update, v_out, m_out, kernel_name)

    inputlist = [grad, input_v, input_m, input_param, beta_1, one_minus_beta_1,
                 beta_2, one_minus_beta_2, epsilon, steps, do_use_weight, weight_decay_rate]

    with tvm.target.cce():
        sch = auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": list(inputlist) + list(res),
              "dummy_placeholder": True}

    build(sch, config)
