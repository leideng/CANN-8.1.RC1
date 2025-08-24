#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019 Huawei Technologies Co., Ltd
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
relu

  Op_description :
    Algrithm: relu(x) = max(x, 0)

    # relu(
    #   x,
    #   y,
    #   kernel_name='relu_cce')

  Supportive_dtype_format :
    ['float16', 'float32', 'int8', 'int32']
    ['NCHW', 'NHWC', 'NC1HWC0']
  Constraint :
    [1] All : shape size limit is 2147483648.

"""
import tbe.dsl as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util

# const value
CONST_ZERO = 0


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
@register_operator_compute("Relu", op_mode="static", support_fusion=True)
@tbe_platform.fusion_manager.fusion_manager.register("relu")
def relu_compute(x, y, kernel_name="relu"):
    """
    Algrithm : relu(x) = max(x, 0)

    Parameters
    ----------
    x: the placeholder of data input

    y : the dict of output

    kernel_name : cce kernel name

    Returns
    -------
    res : result of relu
    """

    inp_dtype = x.dtype
    shape = x.shape
    compatible_dtype = x.dtype

    if inp_dtype == 'int8' and tbe_platform.api_check_support('tbe.dsl.cast_to', 's82f16'):
        x = tbe.cast_to(x, 'float16')
        compatible_dtype = 'float16'
    if tbe_platform.api_check_support('tbe.dsl.vrelu', compatible_dtype):
        data_res = tbe.vrelu(x)
    else:
        tensor_zero = tbe.broadcast(tvm.const(CONST_ZERO, compatible_dtype), shape)
        data_res = tbe.vmax(x, tensor_zero)

    data_res = tbe.cast_to(data_res, inp_dtype)

    return data_res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def relu(x, y, kernel_name="relu"):
    """
    Algrithm: relu(x) = max(x, 0)

    Parameters
    ----------
    Algorithm: relu

    Parameters:

    x: the dict of input data, support float16, float32, int8, int32

    y: the dict of output

    kernel_name: cce kernel name, default value is "relu".

    Returns
    -------
    None
    """

    shape = x.get("shape")
    dtype = x.get("dtype")

    para_check.check_shape(shape, param_name="x")
    shape, _ = shape_util.refine_shape_axes(shape, [])

    check_list = ("float16", "float32", "int8", "int32")
    para_check.check_dtype(dtype, check_list, param_name="x")

    dtype = dtype.lower()
    input_data = tvm.placeholder(shape, dtype, "input_data")

    with tvm.target.cce():
        res = relu_compute(input_data, y, kernel_name)
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [input_data, res],
              "print_ir": False,
             }

    tbe.build(sch, config)
