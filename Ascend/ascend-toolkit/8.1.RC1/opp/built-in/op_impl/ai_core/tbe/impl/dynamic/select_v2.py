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
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import error_manager_vector as error_manager
from impl.util.util_soc_common import is_v220
from impl.util.util_soc_common import after_v200


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals,invalid-name
# 'pylint: disable=too-many-locals, invalid-name, unused-argument
# 'pylint: disable=unused-variable
@register_operator_compute("SelectV2", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def select_v2_compute(condition, x1, x2, y, kernel_name="select_v2"):
    """
    compute for select_v2

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
    if (condition.dtype, x1.dtype, x2.dtype) in (("bool", "int64", "int64"), ("int8", "int64", "int64")):
        _, _, shape_max1 = shape_util.broadcast_shapes(condition.shape, x1.shape)
        _, _, shape_max = shape_util.broadcast_shapes(x2.shape, shape_max1)
        condition = tbe.cast_to(condition, "float16")
        condition = tbe.broadcast(condition, shape_max)
        condition = tbe.cast_to(condition, "bool")
        x1 = tbe.broadcast(x1, shape_max)
        x2 = tbe.broadcast(x2, shape_max)

        return tbe.vsel(condition, x1, x2)

    ori_dtype = x1.dtype
    trans_dtype = "float16"
    if ori_dtype in ("float32", "int32") and tbe_platform.api_check_support("tbe.dsl.vcmpsel", "float32"):
        trans_dtype = "float32"
    if ori_dtype != trans_dtype:
        x1 = tbe.cast_to(x1, trans_dtype)
        x2 = tbe.cast_to(x2, trans_dtype)
    condition = tbe.cast_to(condition, trans_dtype)

    shape_x1list, con_shapelist, shape_max_x1 = shape_util.broadcast_shapes(x1.shape, condition.shape)
    shape_x2list, shape_max_x1, shape_max = shape_util.broadcast_shapes(x2.shape, shape_max_x1)
    x1 = tbe.broadcast(x1, shape_max)
    x2 = tbe.broadcast(x2, shape_max)
    condition = tbe.broadcast(condition, shape_max)
    ones = tbe.broadcast(tvm.const(1, dtype=trans_dtype), shape_max, output_dtype=trans_dtype)
    if is_v220():
        mask = tbe.vcmp(condition, ones, "eq", "bit")
        res = tbe.vsel(mask, x1, x2)
    else:
        res = tbe.vcmpsel(condition, rhs=ones, operation='eq', slhs=x1, srhs=x2)
    if ori_dtype != trans_dtype:
        res = tbe.cast_to(res, ori_dtype)
    return res


@register_operator("SelectV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def select_v2(condition, x1, x2, y, kernel_name="select_v2"):
    """
      Selects elements from `x1` or `x2`, depending on `condition`.

      Parameters
      ----------
      condition: dict
          dict of condition, include keys(shape and dtype),
          only support bool
      x1: dict
          dict of x1, only support bfloat16, float16, float32, int32, int8, uint8
      x2: dict
          dict of x2, only support bfloat16, float16, float32, int32, int8, uint8
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

    dtype_x1 = dtype_x1.lower()
    dtype_x2 = dtype_x2.lower()
    check_list = ("bfloat16", "float16", "float32", "int32", "int8", "uint8", "int64")
    para_check.check_dtype_rule(dtype_x1, check_list)
    if dtype_x1 != dtype_x2:
        error_detail = "Dtype of tensor x1 and x2 must be equal"
        error_manager.raise_err_two_input_shape_invalid(kernel_name, "dtype_x1", "dtype_x2", error_detail)

    bool_dtype = bool_dtype.lower()
    bool_check_list = ("bool", "int8", "uint8")
    para_check.check_dtype_rule(bool_dtype, bool_check_list)

    bool_dtype = "int8" if bool_dtype == "bool" else bool_dtype

    ins = classify([condition, x1, x2], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_condition, _x1, _x2) in ins:
        with tbe.compute():
            con_shape, shape_x1, shape_x2 = shape_util.variable_shape([_condition, _x1, _x2])
            if after_v200() and dtype_x1 == "int32":
                dtype_x1 = "float32"
                dtype_x2 = "float32"

            condition_tensor = tvm.placeholder(con_shape, name="condition_tensor", dtype=bool_dtype)
            input_then_tensor = tvm.placeholder(shape_x1, name="input_then_tensor", dtype=dtype_x1)
            input_else_tensor = tvm.placeholder(shape_x2, name="input_else_tensor", dtype=dtype_x2)

            res = select_v2_compute(condition_tensor, input_then_tensor, input_else_tensor, y, kernel_name)
            tensors.append([condition_tensor, input_then_tensor, input_else_tensor, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
