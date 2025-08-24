"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

hard_sigmoid_grad
"""
import te.lang.cce as tbe
from impl.util.platform_adapter import build
from impl.util.platform_adapter import auto_schedule
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=unused-argument,too-many-arguments,too-many-locals
@register_operator_compute("hard_sigmoid_grad", op_mode="static", support_fusion=True)
def hard_sigmoid_grad_compute(grads, input_x, y, alpha, kernel_name="hard_sigmoid_grad"):
    """
    calculating data

    Parameters
    ----------
    grads : TVM tensor
        the placeholder of grads
    input_x : TVM tensor
        the placeholder of input_x
    y : dict
        dict of output_y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "hard_sigmoid_grad"

    Returns
    -------
    output tensor
    """
    dtypex = input_x.dtype
    shape = input_x.shape
    resultalpha = tbe.vmuls(grads, alpha)
    zero_tensor_x = tbe.broadcast(tvm.const(0, dtypex), shape)
    result = tbe.vcmpsel(tbe.vabs(input_x), 3.0, 'lt', resultalpha, zero_tensor_x)
    return result


# 'pylint: disable=unused-argument
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def hard_sigmoid_grad(grads, input_x, y, alpha=0.16666666, beta=0.5, kernel_name="hard_sigmoid_grad"):
    """
    calculating data

    Parameters
    ----------
    grads : dict
        shape and dtype of input
    input_x : dict
        shape and dtype of input
    y : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "hard_sigmoid_grad"

    Returns
    -------
    None
    :param kernel_name: kernel name, default value is "hard_sigmoid_grad"
    :param beta: An optional float. Defaults to 0.5.
    :param grads: A tensor. Must be one of the following types: float16, float32.
    :param input_x: A tensor. Must be one of the following types: float16, float32.
    :param alpha: An optional float. Defaults to 0.16666666.
    """

    shape_grad = grads.get("shape")
    shape_x = input_x.get("shape")

    para_check.check_shape_rule(shape_x)
    para_check.check_shape_rule(shape_grad)
    para_check.check_shape_size(shape_x)
    para_check.check_shape_size(shape_grad)
    if list(shape_x) != list(shape_grad):
        raise RuntimeError("grads and input_x must have the same shape.")

    check_list = ("float16", "float32")
    grad_dtype = grads.get("dtype").lower()
    input_dtype = input_x.get("dtype").lower()
    para_check.check_dtype_rule(input_dtype, check_list)
    para_check.check_dtype_rule(grad_dtype, check_list)
    para_check.check_kernel_name(kernel_name)
    data_input_grad = tvm.placeholder(shape_grad, name="data_grads", dtype=grad_dtype)
    data_input_x = tvm.placeholder(shape_x, name="data_input_x", dtype=input_dtype)
    res = hard_sigmoid_grad_compute(data_input_grad, data_input_x, y, alpha, kernel_name)

    with tvm.target.cce():
        schedule = auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input_grad, data_input_x, res]}

    build(schedule, config)
