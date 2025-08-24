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
tensor_equal
"""

from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import shape_util
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm

NUM_ONE = 1.0
NUM_ZERO = 0.0

# define a scalar exponent, value is -126,minimun num of float32 exponent
SCALAR_MIN_EXP_FP32 = -126
# define a scalar exponent, value is 50
SCALAR_MUL_EXP_FP32 = 50
# define a scalar exponent, value is 26
SCALAR_MUL2_EXP_FP32 = 26
# define a scalar exponent, value is -24,minimun num of float16 exponent
SCALAR_MIN_EXP_FP16 = -24
# define a scalar exponent, value is 12
SCALAR_MUL_EXP_FP16 = 12
# define a scalar, minimun num of float32 2^SCALAR_MIN_EXP_FP32
SCALAR_MIN_FP32 = 2 ** SCALAR_MIN_EXP_FP32
# define a scalar, value is 2^SCALAR_MUL_EXP_FP32
SCALAR_MUL_FP32 = 2 ** SCALAR_MUL_EXP_FP32
# define a scalar, value is 2^SCALAR_MUL2_EXP_FP32
SCALAR_MUL2_FP32 = 2 ** SCALAR_MUL2_EXP_FP32
# define a scalar, minimun num of float16 2^SCALAR_MIN_EXP_FP16
SCALAR_MIN_FP16 = 2 ** SCALAR_MIN_EXP_FP16
# define a scalar, value is 2^SCALAR_MUL_EXP_FP16
SCALAR_MUL_FP16 = 2 ** SCALAR_MUL_EXP_FP16
# define a scalar, value is 1
SCALAR_ONE = 1


# 'pylint: disable=too-many-locals,unused-argument
@register_operator_compute("tensor_equal", op_mode="static", support_fusion=True)
def tensor_equal_compute_use_sub(input_x, input_y, output_y, kernel_name="tensor_equal"):
    '''
    True if two tensors have the same size and elements, False otherwise
    :param input_x: TVM tensor
            input tenser x
    :param input_y: TVM tensor
            input tensor y
    :param kernel_name: str
            kernel name, default value is "tensor_equal"
    :return:output_z
            output tensor with True or False
    '''
    dtype_x = input_x.dtype
    dtype_y = input_y.dtype
    shape_x = shape_util.shape_to_list(input_x.shape)
    shape_y = shape_util.shape_to_list(input_y.shape)

    x_axis_list = []
    for i in range(len(shape_x)):
        x_axis_list.append(i)

    if shape_x != shape_y or dtype_x != dtype_y:
        # general False result and return
        scalar_zero = tvm.const(NUM_ZERO, dtype=dtype_x)
        zero_res = tbe.vmuls(input_x, scalar_zero)
        zero_res = tbe.cast_to(zero_res, "int8", True)
        res = tbe.reduce_min(zero_res, x_axis_list, False)
        res = tbe.cast_to(res, "int8", True)
        return res

    if dtype_x == "float32":
        scalar_min = tvm.const(SCALAR_MIN_FP32, dtype="float32")
        scalar_mul = tvm.const(SCALAR_MUL_FP32, dtype="float32")
        scalar_mul1 = tvm.const(SCALAR_MUL2_FP32, dtype="float32")
        scalar_one = tvm.const(-1*SCALAR_ONE, dtype="float32")
    else:
        scalar_min = tvm.const(SCALAR_MIN_FP16, dtype="float16")
        scalar_mul = tvm.const(SCALAR_MUL_FP16, dtype="float16")
        scalar_one = tvm.const(-1*SCALAR_ONE, dtype="float16")
    if dtype_x in ("int8", "uint8"):
        input_x = tbe.cast_to(input_x, "float16")
        input_y = tbe.cast_to(input_y, "float16")

    res_vsub = tbe.vsub(input_x, input_y)
    res_vabs = tbe.vabs(res_vsub)
    res_min = tbe.vmins(res_vabs, scalar_min)
    res_vmul = tbe.vmuls(res_min, scalar_mul)
    res_vmul1 = tbe.vmuls(res_vmul, scalar_mul)

    if dtype_x == "float32":
        res_vmul2 = tbe.vmuls(res_vmul1, scalar_mul1)
        res_vsub1 = tbe.vadds(res_vmul2, scalar_one)
        res_vabs1 = tbe.vabs(res_vsub1)
    else:
        res_vsub1 = tbe.vadds(res_vmul1, scalar_one)
        res_vabs1 = tbe.vabs(res_vsub1)

    res = tbe.cast_to(res_vabs1, "int8", True)
    res = tbe.reduce_min(res, x_axis_list, True)
    res = tbe.cast_to(res, "int8", True)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def tensor_equal(input_x, input_y, output_z, kernel_name="tensor_equal"):
    '''
    True if two tensors have the same size and elements, False otherwise

    :param input_x: dict
                input tenser x
    :param input_y: dict
                input tensor y
    :param kernel_name: str
                  kernel name, default value is "tensor_equal"
    :return: none
    '''

    shape_x = input_x.get("shape")
    dtype_x = input_x.get("dtype")
    shape_y = input_y.get("shape")
    dtype_y = input_y.get("dtype")

    para_check.check_shape(shape_x)
    para_check.check_shape(shape_y)

    check_list = ("float16", "float32", "int32", "int8", "uint8")
    para_check.check_dtype(dtype_x, check_list)
    para_check.check_dtype(dtype_y, check_list)

    shape_x = list(shape_x)
    shape_x, _ = shape_util.refine_shape_axes(shape_x, [])
    data_input_x = tvm.placeholder(shape_x, name="data_input_x", dtype=dtype_x)
    shape_y, _ = shape_util.refine_shape_axes(shape_y, [])
    data_input_y = tvm.placeholder(shape_y, name="data_input_y", dtype=dtype_y)

    # use vsub method compute equal result
    res = tensor_equal_compute_use_sub(data_input_x, data_input_y, output_z, kernel_name)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input_x, data_input_y, res],
              "bool_storage_as_1bit": False}

    tbe.cce_build_code(schedule, config)
