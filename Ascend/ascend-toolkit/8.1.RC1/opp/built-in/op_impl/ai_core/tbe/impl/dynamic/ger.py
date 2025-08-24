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
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_common import is_unknown_rank_input


# 'pylint: disable=invalid-name, unused-argument
@register_operator_compute("Ger", op_mode="dynamic", support_fusion=False)
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
    shape_x1, shape_x2, shape_broadcast = shape_util.unify_broadcast_shapes([data_x1.shape, data_x2.shape])

    broa_x1 = tbe.broadcast(data_x1, shape_broadcast)
    broa_x2 = tbe.broadcast(data_x2, shape_broadcast)
    res = tbe.vmul(broa_x1, broa_x2)
    return res


# 'pylint: disable=invalid-name, unused-argument,too-many-locals
@register_operator("Ger")
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
    # operator parameter dtype check
    data_type_x1 = x1.get("dtype").lower()
    data_type_x2 = x2.get("dtype").lower()
    check_tuple = ("float16", "float32")
    para_check.check_dtype(data_type_x1, check_tuple, param_name="x1")
    para_check.check_dtype(data_type_x2, check_tuple, param_name="x2")

    if is_unknown_rank_input([x1, x2]):
        x1, x2 = [x1, x1] if is_unknown_rank_input([x1]) else [x2, x2]
    else:
        shape_x1 = list(x1.get("shape"))
        shape_x2 = list(x2.get("shape"))
        range_x1 = list(x1.get("range"))
        range_x2 = list(x2.get("range"))
        shape_x1_new = shape_x1 + [1]
        range_x1_new = range_x1 + [(1, 1)]
        shape_x2_new = [1] + shape_x2
        range_x2_new = [(1, 1)] + range_x2

        x1["shape"] = shape_x1_new
        x2["shape"] = shape_x2_new
        x1["range"] = range_x1_new
        x2["range"] = range_x2_new
    # operator check
    para_check.check_kernel_name(kernel_name)

    ins = classify([x1, x2], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_x1, _x2) in ins:
        with tbe.compute():
            shape_x1, shape_x2 = shape_util.variable_shape([_x1, _x2])
            data_x1 = tvm.placeholder(shape_x1, name="data_x1", dtype=data_type_x1)
            data_x2 = tvm.placeholder(shape_x2, name="data_x2", dtype=data_type_x2)
            res = ger_compute(data_x1, data_x2, y, kernel_name)
            tensors.append((data_x1, data_x2, res))
        with tvm.target.cce():
            schedule = tbe.auto_schedule(res)
        schedules.append(schedule)

    config = {"name": kernel_name,
              "tensor_list": tensors,
              "need_build": False,
              }
    tbe.build(schedules, config)
