"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

reciprocal
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.reciprocal import op_select_format as reciprocal_op_select_format
from impl.util.platform_adapter import OpTbeImplMode


def op_select_format(input_x, output_y, kernel_name="reciprocal"):
    """
    Get support format according to input_x
    """
    return reciprocal_op_select_format(input_x, output_y, kernel_name)


# 'pylint: disable=redefined-builtin,unused-argument,too-many-locals
@register_operator_compute("Reciprocal", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def reciprocal_compute(input_x, output_y, kernel_name="reciprocal"):
    """
    reciprocal_compute
    calculating data's reciprocal,y= 1 / x

    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input data
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is reciprocal

    Returns
    -------
    res: TVM tensor
    """
    if tbe_platform.api_check_support("tbe.dsl.vdiv",
                                      "float32"):
        dtype = input_x.dtype
        if dtype == "float16":
            input_x = tbe.cast_to(input_x, "float32")
        data_one = tbe.broadcast(tvm.const(1, "float32"), input_x.shape)
        res = tbe.vdiv(data_one, input_x)
        if dtype == "float16":
            res = tbe.cast_to(res, "float16")
    else:
        res = tbe.vrec(input_x, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)

    return res


@register_operator("Reciprocal")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def reciprocal(input_x, output_y, kernel_name="reciprocal"):
    """
    algorithm: reciprocal

    calculating data's reciprocal,y= 1 / x

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support bfloat16, float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is reciprocal

    Returns
    -------
    None
    """
    dtype = input_x.get("dtype")
    check_list = ("bfloat16", "float16", "float32")
    input_dtype = dtype.lower()
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")
    schedules, tensors = [], []
    ins = classify([input_x], OpPatternMode.ELEWISE)
    for (_input_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_input_x])
            data_input = tvm.placeholder(x_shape[0], dtype=input_dtype,
                                         name="data_input")
            res = reciprocal_compute(data_input, output_y, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name,
              "print_ir": False,
              "tensor_list": tensors
              }
    tbe.build(schedules, config)
