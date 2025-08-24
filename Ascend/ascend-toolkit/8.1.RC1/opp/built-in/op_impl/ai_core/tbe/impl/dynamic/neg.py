"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

neg
"""
from __future__ import absolute_import

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util import util_common


# 'pylint: disable=locally-disabled,unused-argument,redefined-argument-from-local
@register_operator_compute("Neg", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def neg_compute(input_x, output_y, kernel_name="neg"):
    """
    compute neg

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        data of output.
    kernel_name: str
        kernel name, default value is "neg"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    dtype = input_x.dtype.lower()

    if dtype == "int32" or dtype == "int64":
        data_tmp = tbe.broadcast(-1, input_x.shape, input_x.dtype)
        res = tbe.vmul(input_x, data_tmp)
    else:
        if dtype == "int8":
            input_x = tbe.cast_to(input_x, "float16")
        res = tbe.vmuls(input_x, -1.0)

    if dtype == "int8":
        res = util_common.uint8_int8_overflow_proc(res, dtype)

    return res


@register_operator("Neg")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def neg(input_x, output_y, kernel_name="neg"):
    """
    Computes numerical negative value element-wise, y = -x.

    Parameters
    ----------
    input_x: dict
        shape and dtype of input, only support bfloat16, float16, float32, int32, int8, int64
    output_y: dict
        shape and dtype of output, should be same type as input
    kernel_name: str
        kernel name, default value is "neg"

    Returns
    -------
    None
    """
    dtype_input = input_x.get("dtype").lower()
    check_list = ("bfloat16", "float16", "float32", "int32", "int8", "int64")
    para_check.check_dtype(dtype_input, check_list, param_name="input_x")

    ins = classify([input_x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (input_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([input_x])
            data_input = tvm.placeholder(x_shape[0], name="data_input",
                                         dtype=dtype_input)
            res = neg_compute(data_input, output_y, kernel_name)

            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
