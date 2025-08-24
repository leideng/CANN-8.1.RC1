#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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
ger
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator_compute
from impl import constant_util


# 'pylint: disable=invalid-name, unused-argument
@register_operator_compute("ger", op_mode="static", support_fusion=True)
def ger_compute(data_x1, data_x2, y, kernel_name="ger"):
    """
    algorithm: ger

    Parameters
    ----------
    data_x1: TVM tensor
        the placeholder of input x
    data_x2: TVM tensor
        the placeholder of input x2
    y: dict
        shape and dtype of output
    kernel_name: str
        cce kernel name, default value is "ger"

    Returns
    -------
    res : output of the datas' ger
    """
    shape_x1 = shape_util.shape_to_list(data_x1.shape)
    shape_x2 = shape_util.shape_to_list(data_x2.shape)
    shape_common = [shape_x1[0], shape_x2[1]]

    broa_x1 = tbe.broadcast(data_x1, shape_common)
    broa_x2 = tbe.broadcast(data_x2, shape_common)
    res = tbe.vmul(broa_x1, broa_x2)
    return res


# 'pylint: disable=invalid-name, unused-argument, too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def ger(x1, x2, y, kernel_name="ger"):
    """
    calculate the outer product of x1 and x2. If x1 is a vector of size n and
    x2 is a vector of size m, then y must be a matrix of size (n*m)

    Parameters
    ----------
    x1 : dict
        shape and dtype of first input, only support float16, float32
    x2 : dict
        shape and dtype of second input, only support float16, float32
    y: dict
        shape and dtype of output
    kernel_name : str
        cce kernel name, default value is "ger"

    Returns
    -------
    None
    """
    # obtain operator information
    shape_x1 = x1.get("shape")
    shape_x2 = x2.get("shape")
    data_type_x1 = x1.get("dtype").lower()
    data_type_x2 = x2.get("dtype").lower()
    check_tuple = ("float16", "float32")

    # operator check
    para_check.check_kernel_name(kernel_name)
    para_check.check_shape_rule(shape_x1)
    para_check.check_shape_rule(shape_x2)
    para_check.check_shape_size(shape_x1, constant_util.SHAPE_SIZE_LIMIT - 1)
    para_check.check_shape_size(shape_x2, constant_util.SHAPE_SIZE_LIMIT - 1)
    para_check.check_dtype_rule(data_type_x1, check_tuple)
    para_check.check_dtype_rule(data_type_x2, check_tuple)

    # tensor placeholder
    shape_broa_x1 = [shape_x1[0], 1]
    shape_broa_x2 = [1, shape_x2[0]]
    data_x1 = tvm.placeholder(shape_broa_x1, name="data_x1", dtype=data_type_x1)
    data_x2 = tvm.placeholder(shape_broa_x2, name="data_x2", dtype=data_type_x2)

    # ger compute function
    res = ger_compute(data_x1, data_x2, y, kernel_name)

    # auto schedule
    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    # compile configuration
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data_x1, data_x2, res]}
    tbe.build(schedule, config)
