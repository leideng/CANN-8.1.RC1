#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (C) Huawei Technologies Co., Ltd 2022-2022. All rights reserved.
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
sort_v2
"""
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator


# 'pylint: disable=too-few-public-methods
@register_operator("SortV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def sort_v2(x, y, axis=-1, descending=False, kernel_name="sort_v2"):
    """
    Function: Sorts the elements of the input tensor along a given dimension in ascending order by value.

    Init base parameters
    Parameters
    ----------
    input(x): dict
        data of input
    output(y): dict
        data of output
    dim(axis): int
    descending: bool
    kernel_name: str
        the name of the operator
    ----------
    """
    dtype = x.get("dtype")
    out_dtype = y.get("dtype")
    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(dtype, check_list, param_name="x")
    para_check.check_dtype(out_dtype, check_list, param_name="y")
    x_shape = x.get("shape")
    if axis is None:
        axis = -1
    if x_shape[axis] == 1:
        raise RuntimeError("Data of sort axis is 1, SortV2 doesn't support.")

    ins = classify([x, axis], OpPatternMode.SORT, {"op_mode": "sort"})
    schedules, tensors = [], []
    for (_x, ) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_x], "sort")
            x_input = tvm.placeholder(x_shape, name="data_input", dtype=x["dtype"])
            direction = "descend" if descending else "ascend"
            if x["dtype"] == "bfloat16":
                x_input_fp32 = tbe.cast_to(x_input, "float32")
                value = tbe.sort(x_input_fp32, sort_axis=-1, direction=direction, return_type="value", need_cast=True)
            else:
                value = tbe.sort(x_input, sort_axis=-1, direction=direction, return_type="value")
            tensors.append([x_input, value])
        with tvm.target.cce():
            sch = tbe.auto_schedule(value)
        schedules.append(sch)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
