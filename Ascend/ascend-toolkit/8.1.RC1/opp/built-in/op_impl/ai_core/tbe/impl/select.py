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
select
"""
from impl.util.platform_adapter import register_operator_compute
import tbe.dsl as tbe_dsl
import te.lang.cce as tbe
from impl.util.util_soc_common import is_v310
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import error_manager_vector
from impl.dynamic.select import op_select_format as static_op_select_format


# 'pylint: disable=unused-argument,invalid-name,too-many-arguments
def check_supported(condition, x1, x2, y, kernel_name="select"):
    """
    static shape do not support int64
    """
    x1_dtype = x1.get("dtype").lower()
    if x1_dtype in ("int64",):
        return False, "int64 not supported."

    return True, ""


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals,invalid-name
# 'pylint: disable=locally-disabled,too-many-statements,too-many-branches
# 'pylint: disable=get-dict-value-exception
def op_select_format(condition, x1, x2, y, kernel_name="select"):
    """1.when all input(condition, x1, x2) have the same ori_shape, ori_format,
       and the format is in ["NCHW", "NHWC", "HWCN"] or ["NDHWC", "DHWCN", "NCDHW"],
       the Op Select can support ND, FRACTAL_NZ, NC1HWC0 and FRACTAL_Z.

        for example:
        conditon : Tensor (shape=(16, 16, 16, 16), "NCHW")
        x1 : Tensor of (shape=(16, 16, 16, 16), "NCHW")
        x2 : Tensor of (shape=(16, 16, 16, 16), "NCHW")
        the Op Select can process with NC1HWC0:
        conditon : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")
        x1 : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")
        x2 : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")

    2.when all input(x1, x2) have the same ori_shape, ori_format, and the
       format is in ["NCHW", "NHWC", "HWCN"] or ["NDHWC", "DHWCN", "NCDHW"],
       and conditon is a scaler. The Op Select can support ND, FRACTAL_NZ,
       NC1HWC0 and FRACTAL_Z.

        for example:
        conditon : Tensor of (shape=(2), "NCHW")
        x1 : Tensor of (shape=(16, 16, 16, 16), "NCHW")
        x2 : Tensor of (shape=(16, 16, 16, 16), "NCHW")
        the Op Select can process with NC1HWC0:
        conditon : Tensor of (shape=(2), "NCHW")
        x1 : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")
        x2 : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")
    """
    return static_op_select_format(condition, x1, x2, y, kernel_name)


# 'pylint: disable=too-many-locals, invalid-name, unused-argument
@register_operator_compute("select", op_mode="static", support_fusion=True)
def select_compute(condition, x1, x2, y, kernel_name="select"):
    """compute for select

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
        cce kernel name, default value is "select"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    shape = shape_util.shape_to_list(x1.shape)
    con_shape = shape_util.shape_to_list(condition.shape)
    input_x_dtype = x1.dtype
    if input_x_dtype ==  "bfloat16" and not tbe_platform.api_check_support("tbe.dsl.vadd", "bfloat16") and \
            tbe_platform.intrinsic_check_support("Intrinsic_vconv", "bf162f32"):
        x1 = tbe.cast_to(x1, "float32")
        x2 = tbe.cast_to(x2, "float32")
    num_dtype = x1.dtype

    if (num_dtype in ("float32", "int32")) and \
            (not tbe_platform.api_check_support("tbe.dsl.vsel", "float32")):
        if num_dtype == "int32":
            condition = tbe.ceil(condition)
        else:
            condition = tbe.cast_to(condition, num_dtype)
        condition = tbe.broadcast(condition, shape)
        ones = tbe.broadcast(tvm.const(1, dtype=num_dtype), shape, output_dtype=num_dtype)
        condition_opp = tbe.vsub(ones, condition)
        temp_x = tbe.vmul(x1, condition)
        temp_y = tbe.vmul(x2, condition_opp)
        res = tbe.vadd(temp_x, temp_y)
        return res

    if num_dtype in ("int8", "uint8", "int32"):
        if tbe_platform.api_check_support("tbe.dsl.vsel", "float32"):
            x1_dtype = "float32"
            ones = tbe.broadcast(tvm.const(1, dtype="float32"), shape, output_dtype="float32")
            x1 = tbe.cast_to(x1, "float32")
            x2 = tbe.cast_to(x2, "float32")
        else:
            x1_dtype = "float16"
            ones = tbe.broadcast(tvm.const(1, dtype="float16"), shape, output_dtype="float16")
            x1 = tbe.cast_to(x1, "float16")
            x2 = tbe.cast_to(x2, "float16")
    else:
        x1_dtype = num_dtype
        ones = tbe.broadcast(tvm.const(1, dtype=num_dtype), shape, output_dtype=num_dtype)
    if list(con_shape) == list(shape):
        if not is_v310():
            res = tbe.vsel(condition, x1, x2)
        else:
            condition = tbe.cast_to(condition, x1_dtype)
            condition_rev = tbe.vsub(ones, condition)
            temp_x = tbe.vmul(x1, condition)
            temp_y = tbe.vmul(x2, condition_rev)
            res = tbe.vadd(temp_x, temp_y)
    else:
        condition = tbe.cast_to(condition, x1_dtype)
        condition = tbe.broadcast(condition, shape)
        res = tbe.vcmpsel(condition, rhs=ones, operation='eq', slhs=x1, srhs=x2)
    if num_dtype in ("int8", "uint8", "int32"):
        res = tbe.cast_to(res, num_dtype)
    if input_x_dtype == "bfloat16" and res.dtype != input_x_dtype and res.dtype.lower() == "float32":
        res = tbe_dsl.round(res, input_x_dtype)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def select(condition, x1, x2, y, kernel_name="select"):
    """Selects elements from `x1` or `x2`, depending on `condition`.

    Parameters
    ----------
    condition: dict
        dict of condition, include keys(shape and dtype),
        only support int8,int32
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
    dtype_x1 = x1.get("dtype").lower()
    shape_x2 = x2.get("shape")
    dtype_x2 = x2.get("dtype").lower()
    con_shape = condition.get("shape")
    bool_dtype = condition.get("dtype").lower()
    if bool_dtype == "bool":
        bool_dtype = "int8"
    para_check.check_shape(shape_x1, param_name="x1")
    check_list = ("bfloat16", "float16", "float32", "int32", "int8", "uint8")
    para_check.check_dtype(dtype_x1, check_list, param_name="x1")

    if shape_x1 != shape_x2:
        error_detail = "Shape of tensor x1 and x2 must be equal!"
        error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "x1", \
                                                               "x2", error_detail)

    if dtype_x1 != dtype_x2:
        error_detail = "Dtype of tensor x1 and x2 must be equal!"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "x1", \
                                                               "x2", error_detail)

    x_len = len(shape_x1)
    con_shape = list(con_shape)
    if len(con_shape) == 1 and x_len != 1:
        if 1 != con_shape[0] != shape_x1[0]:
            error_detail = "Shape of tensor condition and x1 dim[0] must be equal!"
            error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "condition", \
                                                                   "x1", error_detail)
        while x_len > len(con_shape):
            con_shape += [1]
    else:
        if list(con_shape) != list(shape_x1):
            error_detail = "Shape of tensor condition and x1 must be equal!"
            error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "condition", \
                                                                   "x1", error_detail)

    con_shape, shape_x1 = shape_util.refine_shapes_for_broadcast(con_shape, shape_x1)

    flag_cloud = tbe_platform.api_check_support("tbe.dsl.vsel", "float32")
    flag_dtype = dtype_x1 in ("float32", "int32")
    if (list(con_shape) != list(shape_x1)) or \
            ((not flag_cloud) and flag_dtype):
        condition = tvm.placeholder(con_shape, name="condition", dtype=bool_dtype)
    else:
        condition = tvm.placeholder(con_shape, name="condition", dtype="bool")
    input_x1 = tvm.placeholder(shape_x1, name="input_x1", dtype=dtype_x1)
    input_x2 = tvm.placeholder(shape_x1, name="input_x2", dtype=dtype_x2)

    with tvm.target.cce():
        res = select_compute(condition, input_x1, input_x2, y, kernel_name)
        sch = tbe.auto_schedule(res)

    if list(con_shape) == list(shape_x1):
        config = {"name": kernel_name,
                  "tensor_list": [condition, input_x1, input_x2, res],
                  "bool_storage_as_1bit": False}
    else:
        config = {"name": kernel_name,
                  "tensor_list": [condition, input_x1, input_x2, res]}
    tbe.build(sch, config)
