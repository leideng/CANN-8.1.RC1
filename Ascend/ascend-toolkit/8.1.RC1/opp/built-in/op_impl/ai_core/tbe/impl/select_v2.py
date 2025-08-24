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
select_v2
"""
import functools

from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import shape_util
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals,invalid-name
# 'pylint: disable=too-many-locals, invalid-name, unused-argument
# 'pylint: disable=unused-variable
@register_operator_compute("select", op_mode="static", support_fusion=True)
def select_v2_compute(condition, x1, x2, y, kernel_name="select_v2"):
    """compute for select_v2

    Parameters
    ----------
    condition: TVM tensor
        the placeholder of input condition
    x1: TVM tensor
        the placeholder of input x1
    x2: TVM tensor
        the placeholder of input x2
    y: dict
        dict of y
    kernel_name: str
        cce kernel name, default value is "select_v2"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    num_dtype = x1.dtype
    condition_dtype = condition.dtype
    x1 = tbe.cast_to(x1, "float32")
    x2 = tbe.cast_to(x2, "float32")
    condition = tbe.cast_to(condition, "float32")
    shape_x1list = shape_util.shape_to_list(x1.shape)
    shape_x2list = shape_util.shape_to_list(x2.shape)
    con_shapelist = shape_util.shape_to_list(condition.shape)
    shape_x1list, con_shapelist, shape_max_x1 = shape_util.produce_shapes(shape_x1list, con_shapelist)
    shape_x2list, shape_max_x1, shape_max = shape_util.produce_shapes(shape_x2list, shape_max_x1)
    x1 = tbe.broadcast(x1, shape_max)
    x2 = tbe.broadcast(x2, shape_max)
    condition = tbe.broadcast(condition, shape_max)

    ones = tbe.broadcast(tvm.const(1, dtype="float32"), shape_max, output_dtype="float32")

    res = tbe.vcmpsel(condition, rhs=ones, operation='eq', slhs=x1, srhs=x2)
    res = tbe.cast_to(res, num_dtype)
    return res


@para_check.check_input_type(dict, dict, dict, dict, str)
def select_v2(condition, x1, x2, y, kernel_name="select_v2"):
    """Selects elements from `x1` or `x2`, depending on `condition`.

    Parameters
    ----------
    condition: dict
        dict of condition, include keys(shape and dtype),
        only support bool
    x1: dict
        dict of x1, only support float16, float32, int32, int8, uint8
    x2: dict
        dict of x2, only support float16, float32, int32, int8, uint8
    y: dict
        dict of output
    kernel_name: str
        cce kernel name, default value is "select"

    Returns
    -------
    None
    """
    shape_x1 = x1.get("shape")
    dtype_x1 = x1.get("dtype")
    shape_x2 = x2.get("shape")
    dtype_x2 = x2.get("dtype")
    bool_dtype = condition.get("dtype")
    con_shape = condition.get("shape")

    shape_x1, con_shape, shape_max_x1 = shape_util.produce_shapes(shape_x1, con_shape)
    shape_x2, con_shape, shape_max_x2 = shape_util.produce_shapes(shape_x2, con_shape)

    if shape_x1[-1] == 1 and shape_x2[-1] == 1 and con_shape[-1] == 1 \
            and shape_max_x1[-1] == 1:
        shape_x1 = shape_x1 if len(shape_x1) == 1 else shape_x1[:-1]
        shape_x2 = shape_x2 if len(shape_x2) == 1 else shape_x2[:-1]
        con_shape = con_shape if len(con_shape) == 1 else con_shape[:-1]

    para_check.check_kernel_name(kernel_name)
    para_check.check_shape_rule(shape_x1)
    para_check.check_tensor_shape_size(shape_x1)

    if shape_x1 == shape_x2 == con_shape:
        shape_x1 = (functools.reduce(lambda x, y: x * y, shape_x1[:]),)
        shape_x2 = (functools.reduce(lambda x, y: x * y, shape_x2[:]),)
        con_shape = (functools.reduce(lambda x, y: x * y, con_shape[:]),)

    dtype_x1 = dtype_x1.lower()
    dtype_x2 = dtype_x2.lower()
    check_list = ("float16", "float32", "int32", "int8", "uint8")
    para_check.check_dtype_rule(dtype_x1, check_list)
    if dtype_x1 != dtype_x2:
        error_manager_vector.raise_err_inputs_dtype_not_equal("select_v2", "x1", "x2",
                                                              str(dtype_x1), str(dtype_x2))

    bool_dtype = bool_dtype.lower()
    bool_check_list = ("bool", "int8", "uint8")
    para_check.check_dtype_rule(bool_dtype, bool_check_list)

    condition = tvm.placeholder(con_shape, name="condition", dtype=bool_dtype)
    input_then = tvm.placeholder(shape_x1, name="input_then", dtype=dtype_x1)
    input_else = tvm.placeholder(shape_x2, name="input_else", dtype=dtype_x2)

    with tvm.target.cce():
        res = select_v2_compute(condition, input_then, input_else, y, kernel_name)
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [condition, input_then, input_else, res]}
    tbe.cce_build_code(sch, config)
