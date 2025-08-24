#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
addcmul
"""
import te.lang.cce as tbe
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json


# 'pylint: disable=invalid-name,too-many-arguments,too-many-locals,unused-argument
def op_select_format(input_data, x1, x2, value, y, kernel_name="addcmul"):
    """
    op_select_format
    """
    dtype_list = ["float16", "float32", "int32", "bfloat16"]
    dtype_len = len(dtype_list)

    support_format = ["ND"]
    support_dtype = []

    input_data_shape = input_data.get("ori_shape")
    x1_shape = x1.get("ori_shape")
    x2_shape = x2.get("ori_shape")
    input_data_shape = list(shape_util.scalar2tensor_one(input_data_shape))
    x1_shape = list(shape_util.scalar2tensor_one(x1_shape))
    x2_shape = list(shape_util.scalar2tensor_one(x2_shape))

    if input_data_shape == x1_shape == x2_shape:
        support_format.append("FRACTAL_Z")
        support_format.append("FRACTAL_NZ")
        support_format.append("NC1HWC0")

    for dtype in dtype_list:
        support_dtype.extend([dtype] * len(support_format))

    support_format = support_format * dtype_len
    last_format = ["ND"] * len(support_format)

    input0 = gen_param(classify="input0", name="input_data",
                       datatype=",".join(support_dtype),
                       format=",".join(support_format))
    input1 = gen_param(classify="input1", name="x1",
                       datatype=",".join(support_dtype),
                       format=",".join(support_format))
    input2 = gen_param(classify="input2", name="x2",
                       datatype=",".join(support_dtype),
                       format=",".join(support_format))
    input3 = gen_param(classify="input3", name="value",
                       datatype=",".join(support_dtype),
                       format=",".join(last_format))
    output0 = gen_param(classify="output0", name="y",
                        datatype=",".join(support_dtype),
                        format=",".join(support_format))

    param_list = [input0, input1, input2, input3, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def check_op_dtype(dtype_input, dtype_x1, dtype_x2, dtype_value):
    """
    :param dtype_input: str
    :param dtype_x1: str
    :param dtype_x2: str
    :param dtype_value: str
    :return: none
    """
    check_list = ["float16", "float32", "int32", "bfloat16"]

    para_check.check_dtype(dtype_input, check_list)
    para_check.check_dtype(dtype_x1, check_list)
    para_check.check_dtype(dtype_x2, check_list)
    para_check.check_dtype(dtype_value, check_list)

    if dtype_input != dtype_x1 or dtype_input != dtype_x2:
        raise RuntimeError("the dtype of input_data, x1, x2 must be same")

    if dtype_input != dtype_value:
        raise RuntimeError("the dtype of input_data, value must be same")


@register_operator_compute("addcmul", op_mode="static", support_fusion=True, support_bfp16=True)
def addcmul_compute(input_data, x1, x2, value, shape_max, y, kernel_name="addcmul"):
    """
    calculating data's addcmul, y = input_data + value * (x1 * x2)
    :param input_data: TVM tensor
    :param x1: TVM tensor
    :param x2: TVM tensor
    :param value: TVM tensor
    :param shape_max: list
    :param y: dict
    :param kernel_name: str
    :return: TVM tensor
    """
    input_dtype = input_data.dtype.lower()

    input_data = tbe.broadcast(input_data, shape_max)
    x1 = tbe.broadcast(x1, shape_max)
    x2 = tbe.broadcast(x2, shape_max)
    value = tbe.broadcast(value, shape_max)

    vmul_val = tbe.vmul(x1, x2)
    vmul_val2 = tbe.vmul(vmul_val, value)
    res = tbe.vadd(input_data, vmul_val2)

    if res.dtype.lower() != input_dtype:
        res = tbe.cast_to(res, input_dtype)
    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def addcmul(input_data, x1, x2, value, y, kernel_name="addcmul"):
    """
    algorithm: addcmul
    calculating data's addcmul, y = input_data + value * (x1 * x2)

    Parameters
    ----------
    input_data : dict
        shape and dtype of first input, only support float16, float32, int32, bfloat16
    x1 : dict
        shape and dtype of second input, only support float16, float32, int32, bfloat16
    x2 : dict
        shape and dtype of third input, only support float16, float32, int32, bfloat16
    value: dict
        shape and dtype of value, only support float16, float32, int32, bfloat16
    y: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name : str
        cce kernel name, default value is addcmul

    Returns
    -------
    None
    """
    shape_input = input_data.get("shape")
    shape_x1 = x1.get("shape")
    shape_x2 = x2.get("shape")
    shape_value = value.get("shape")
    dtype_input = input_data.get("dtype").lower()
    dtype_x1 = x1.get("dtype").lower()
    dtype_x2 = x2.get("dtype").lower()
    dtype_value = value.get("dtype").lower()

    shape_input = shape_util.scalar2tensor_one(shape_input)
    shape_x1 = shape_util.scalar2tensor_one(shape_x1)
    shape_x2 = shape_util.scalar2tensor_one(shape_x2)
    shape_value = shape_util.scalar2tensor_one(shape_value)
    para_check.check_shape(shape_input)
    para_check.check_shape(shape_x1)
    para_check.check_shape(shape_x2)

    para_check.check_kernel_name(kernel_name)

    check_op_dtype(dtype_input, dtype_x1, dtype_x2, dtype_value)

    if not para_check.is_scalar(shape_value):
        raise RuntimeError("value should be 0D or 1D tensor")

    shape_x1, shape_x2, shape_max1 = shape_util.broadcast_shapes(shape_x1, shape_x2)
    shape_input, _, shape_max = shape_util.broadcast_shapes(shape_input, shape_max1)
    shape_x1, _, _ = shape_util.broadcast_shapes(shape_x1, shape_max)
    shape_x2, _, _ = shape_util.broadcast_shapes(shape_x2, shape_max)
    shape_value, _, _ = shape_util.broadcast_shapes(shape_value, shape_max)
    para_check.check_shape_size(shape_max)

    data_input = tvm.placeholder(shape_input, dtype=dtype_input, name="data_input")
    data_x1 = tvm.placeholder(shape_x1, dtype=dtype_x1, name="data_x1")
    data_x2 = tvm.placeholder(shape_x2, dtype=dtype_x2, name="data_x2")
    data_value = tvm.placeholder(shape_value, dtype=dtype_value, name="data_value")
    res = addcmul_compute(data_input, data_x1, data_x2, data_value, shape_max, y, kernel_name="addcmul")

    with tvm.target.cce():
        schedule = auto_schedule(res)

    tensor_list = [data_input, data_x1, data_x2, data_value, res]

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}

    build(schedule, config)
