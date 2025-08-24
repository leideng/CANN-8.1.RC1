"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

fast_gelu grad
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import OpImplMode
from impl.util.util_common import check_op_impl_mode
from impl.util.util_soc_common import after_v200
from impl.util import util_compute


def fast_gelu_grad_v2(input_x):
    """
    res' = sig * (1 - beta_x*(sig - 1))
    sig = sigmoid(beta_x)
    beta_x = 1.702*x
    """
    const_one = tbe.broadcast(tvm.const(1.0, dtype=input_x.dtype), input_x.shape)
    const_attr = tvm.const(-1.702, dtype=input_x.dtype)
    const_neg_one = tvm.const(-1.0, dtype=input_x.dtype)
    mul_x = tbe.vmuls(input_x, const_attr)

    sig_mul_x = tbe.vexp(mul_x)
    sig_mul_x = tbe.vadd(sig_mul_x, const_one)
    sig_mul_x = tbe.vdiv(const_one, sig_mul_x)

    result_temp = tbe.vadds(sig_mul_x, const_neg_one)
    result_temp = tbe.vmul(mul_x, result_temp)
    result_temp = tbe.vadd(result_temp, const_one)
    result_temp = tbe.vmul(result_temp, sig_mul_x)

    return result_temp


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,no-member
# 'pylint: disable=too-many-locals,invalid-name
@register_operator_compute("FastGeluGrad", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def fast_gelu_grad_compute(input_dy,
                           input_x,
                           output_z,
                           kernel_name="fast_gelu_grad",
                           impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """
    algorithm: fast_gelu_grad
    calculating: dy*res'
    res' = div_up/div_down
    div_up = e^(-1.702x) + 1.702xe^(-1.702x) + e^(1.702(x-|x|))
    div_down = (e^(-1.702x)+1)^2

    Parameters
    ----------
    input_dy : dict
        shape and dtype of dy input, only support float16, float32
    input_x : dict
        shape and dtype of x input, only support float16, float32
    output_z: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is fast_gelu_grad

    Returns
    -------
    A TVM tensor same as input placeholders.
    """
    const_1_value = 1
    attr = 1.702
    dtype = input_x.dtype
    attr_opp = 0 - attr
    check_support_flag = False
    if not (tbe_platform.api_check_support("tbe.dsl.vexp", "float32")) and \
            dtype == "float32":
        check_support_flag = True
        dtype = "float16"
        input_x = tbe.cast_to(input_x, dtype)
        input_dy = tbe.cast_to(input_dy, dtype)
    if after_v200():
        result_temp = fast_gelu_grad_v2(input_x)
    else:
        const1 = tvm.const(-1.702, dtype)
        const2 = tvm.const(1.702, dtype)
        const3 = tvm.const(1.0, dtype)
        # e^(-1.702x)
        abs_x = tbe.vabs(input_x)
        mul_abs_x = tbe.vmuls(abs_x, const1)
        exp_x = tbe.vexp(mul_abs_x)

        # 1.702xe^(-1.702x)
        add_2 = tbe.vmul(input_x, exp_x)
        add_2 = tbe.vmuls(add_2, const2)

        # e^(1.702(x-|x|))
        pn_x = tbe.vsub(input_x, abs_x)
        mul_pn_x = tbe.vmuls(pn_x, const2)
        exp_pn_x = tbe.vexp(mul_pn_x)

        #  e^(-1.702x) + 1.702xe^(-1.702x) + e^(1.702(x-|x|))
        div_up = tbe.vadd(exp_x, add_2)
        div_up = tbe.vadd(div_up, exp_pn_x)

        # (e^(-1.702x)+1)^2
        div_down_i = tbe.vadds(exp_x, const3)
        divdown = tbe.vmul(div_down_i, div_down_i)

        div_down_rec = tbe.vrec(divdown, impl_mode)
        result_temp = tbe.vmul(div_up, div_down_rec)

    if util_compute.check_batchmatmul_fuse(input_dy):
        batch_shape = shape_util.shape_to_list(input_dy.op.attrs["batch_shape"])
        unfold_shape = batch_shape + shape_util.shape_to_list(input_dy.shape)[-4:]
        result_temp = tbe.broadcast(result_temp, unfold_shape)
        result_temp = util_compute.batchmatmul_elem_reshape(input_dy, result_temp, batch_shape, "fast_gelu_grad")
    elif util_compute.check_batchmatmul_fuse(input_x):
        batch_shape = shape_util.shape_to_list(input_x.op.attrs["batch_shape"])
        unfold_shape = batch_shape + shape_util.shape_to_list(input_x.shape)[-4:]
        input_dy = tbe.broadcast(input_dy, unfold_shape)
        input_dy = util_compute.batchmatmul_elem_reshape(result_temp, input_dy, batch_shape, "fast_gelu_grad")

    result = tbe.vmul(input_dy, result_temp)
    if check_support_flag:
        result = tbe.cast_to(result, "float32")

    return result


@register_operator("FastGeluGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME, para_check.OPTION_ATTR_STR)
def fast_gelu_grad(input_dy, input_x, output_z, kernel_name="fast_gelu_grad", impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """
    algorithm: fast_gelu_grad
    calculating: dy*res'
    res' = div_up/div_down
    div_up = e^(-1.702x) + 1.702xe^(-1.702x) + e^(1.702(x-|x|))
    div_down = (e^(-1.702x)+1)^2

    Parameters
    ----------
    input_dy : dict
        shape and dtype of dy input, only support float16, float32
    input_x : dict
        shape and dtype of x input, only support float16, float32
    output_z: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is fast_gelu_grad

    Returns
    -------
    none.
    """
    check_op_impl_mode(impl_mode, [OpImplMode.HIGH_PERFORMANCE, OpImplMode.HIGH_PRECISION], kernel_name)

    shape_dy = input_dy.get("shape")
    shape_x = input_x.get("shape")

    para_check.check_shape(shape_dy, param_name="input_dy")
    para_check.check_shape(shape_x, param_name="input_x")
    input_dtype = input_dy.get("dtype").lower()
    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(input_dtype, check_list, param_name="input_dy")

    ins = classify([input_dy, input_x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_input_dy, _input_x) in ins:
        with tbe.compute():
            dy_shape, x_shape = shape_util.variable_shape([_input_dy, _input_x])

            tensor_dy = tvm.placeholder(dy_shape, input_dtype, "tensor_dy")
            tensor_x = tvm.placeholder(x_shape, input_dtype, "tensor_x")
            res = fast_gelu_grad_compute(tensor_dy, tensor_x, output_z, kernel_name, impl_mode)
            tensors.append([tensor_dy, tensor_x, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
