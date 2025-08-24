"""
Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

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

from ..util.platform_adapter import tbe
from ..util.platform_adapter import para_check
from ..util.platform_adapter import shape_util
from ..util.platform_adapter import tvm
from ..util.platform_adapter import register_operator_compute
from ..util.platform_adapter import classify
from ..util.platform_adapter import OpPatternMode
from ..util.platform_adapter import register_operator
from ..util.util_attr_common import HardSigmoidGradAttrInfo
from ..util.util_attr_common import get_attr_by_cls


# 'pylint: disable=unused-argument,too-many-arguments,too-many-locals
@register_operator_compute("HardSigmoidGrad", op_mode="dynamic", support_fusion=True)
def hard_sigmoid_grad_compute(grads, input_x, y, alpha=0.16666666, beta=0.5, kernel_name="hard_sigmoid_grad"):
    """
    x_grad = 0 if x > 3 or x < -3
    x_grad = alpha * grads if x > -3 and x < 3

    Parameters
    ----------
    grads : TVM tensor
        the placeholder of grads
    input_x : TVM tensor
        the placeholder of input_x
    y : dict
        dict of output_y, include keys(shape and dtype)
    alpha : float
        slope of the func
    beta : float
        offset of the func, specifically, beta does not used in the calculation.
    kernel_name : str
        kernel name, default value is "hard_sigmoid_grad"

    Returns
    -------
    output tensor
    """

    input_dtype = input_x.dtype.lower()
    if input_dtype in ("bfloat16",):
        input_x = tbe.cast_to(input_x, "float32")
        grads = tbe.cast_to(grads, "float32")

    abs_bound_value = 3.0
    dtypex = input_x.dtype
    shape = input_x.shape
    alpha_scalar = get_attr_by_cls(alpha, HardSigmoidGradAttrInfo.ATTR_ALPHA, dtypex)
    result_alpha = tbe.vmuls(grads, alpha_scalar)
    zero_tensor_x = tbe.broadcast(tvm.const(0, dtypex), shape)
    result = tbe.vcmpsel(tbe.vabs(input_x), abs_bound_value, 'lt', result_alpha, zero_tensor_x)
    if input_dtype in ("bfloat16",):
        result = tbe.round(result, "bfloat16")
    return result


# 'pylint: disable=unused-argument
@register_operator("HardSigmoidGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def hard_sigmoid_grad(grads, input_x, y, alpha=0.16666666, beta=0.5, kernel_name="hard_sigmoid_grad"):
    """
    formula:
    x_grad = 0 if x > 3 or x < -3
    x_grad = alpha * grads if x > -3 and x < 3
    
    Parameters
    ----------
    grads : dict
        shape and dtype of input
    input_x : dict
        shape and dtype of input
    y : dict
        shape and dtype of output, should be same shape and type as input_x
    alpha : float
        slope of the func
    beta : float
        offset of the func, specifically, beta does not used in the calculation.
    kernel_name : str
        kernel name, default value is "hard_sigmoid_grad"

    Returns
    -------
    None
    """
    shape_grad = grads.get("shape")
    shape_x = input_x.get("shape")

    para_check.check_shape(shape_x)
    para_check.check_shape(shape_grad)

    # do para check
    check_list = ("bfloat16", "float16", "float32")
    grad_dtype = grads.get("dtype").lower()
    input_dtype = input_x.get("dtype").lower()
    para_check.check_dtype_rule(input_dtype, check_list)
    para_check.check_dtype_rule(grad_dtype, check_list)
    para_check.check_kernel_name(kernel_name)

    # go into compute
    schedules, tensors = [], []
    ins = classify([grads, input_x], OpPatternMode.ELEWISE)

    for (_grad, _x) in ins:
        with tbe.compute():
            grad_shape, x_shape = shape_util.variable_shape([_grad, _x])
            grad_input = tvm.placeholder(grad_shape, dtype=grad_dtype, name="grad_input")
            data_input = tvm.placeholder(x_shape, dtype=input_dtype, name="data_input")
            res = hard_sigmoid_grad_compute(grads=grad_input, input_x=data_input,
             y=y, alpha=alpha, kernel_name=kernel_name)
            tensors.append([grad_input, data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {
        "name": kernel_name,
        "tensor_list": tensors
    }
    tbe.build(schedules, config)
