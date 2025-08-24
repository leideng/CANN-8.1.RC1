#!/usr/bin/python
# -*- coding: utf-8 -*-
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
assign_sub
"""
import functools
import operator
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import error_manager_vector


def _check_param(shape_var, shape_value, dtype_var, dtype_value, kernel_name):
    """
    check the parameters including ref_shape, value_shape, dtype and kernel_name

    Parameters
    ----------
    shape_var: list or tuple
        shape of input_var
    shape_value: list or tuple
        shape of value_tensor
    dtype: str
        the data type
    kernel_name: str
        cce kernel name, default value is "assign_sub"

    Returns
    -------
    None
    """
    # check whether the shape is right
    para_check.check_shape(shape_var, param_name="var")
    para_check.check_shape(shape_value, param_name="value")
    if not operator.eq(shape_var, shape_value):
        error_detail = "Shape of var and value should be same"
        error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "var", "value", error_detail)

    # check whether dtypes are fp16, fp32, int8, uint8, int32
    # and whether they are the same
    check_list = ("float16", "float32", "int8", "uint8", "int32")
    para_check.check_dtype(dtype_var, check_list, param_name="var")
    para_check.check_dtype(dtype_value, check_list, param_name="value")
    if dtype_var != dtype_value:
        error_detail = "Dtype of var and value should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "var", "value", error_detail)


# 'pylint: disable=locally-disabled,too-many-arguments
# 'pylint: disable=unused-argument,unnecessary-lambda
@register_operator_compute("assign_sub", op_mode="static", support_fusion=True)
def _assign_sub_compute(tensor_var, tensor_value, out, kernel_name="assign_sub"):
    """
    assign_sub compute function

    Parameters
    ----------
    tensor_var : tvm.tensor
        tensor of var
    tensor_value : tvm.tensor
        tensor of value
    out : dict
        dict of out.
    kernel_name : str
        cce kernel name, default value is "assign_sub"

    Returns
    -------
    res : tvm.tensor
        tensor of result
    """
    res = tbe.vsub(tensor_var, tensor_value)
    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def assign_sub(var, value, out, kernel_name="assign_sub"):
    """
    Update var by subtracting value from it.

    Parameters:
    ----------
    var : dict
        dict of input_var, include shape and dtype,
        dtype support int8, uint8, int32, float16, float32

    value : dict
        dict of input_value, include shape and dtype,
        dtype support int8, uint8, int32, float16, float32.
        Must have the same shape and dtype as input_var

    out : dict
        dict of out

    kernel_name : str
        cce kernel name, default value is "assign_sub"

    Returns
    -------
    None
    """
    # get the shape and dtype
    shape_var = var.get("shape")
    shape_value = value.get("shape")
    dtype_var = var.get("dtype").lower()
    dtype_value = value.get("dtype").lower()
    _check_param(shape_var, shape_value, dtype_var, dtype_value, kernel_name)

    shape = [functools.reduce(lambda x, y: x * y, shape_var[:])]
    tensor_var = tvm.placeholder(shape, dtype=dtype_var, name="tensor_var")
    tensor_value = tvm.placeholder(shape, dtype=dtype_value, name="tensor_value")
    res = _assign_sub_compute(tensor_var, tensor_value, out, kernel_name="assign_sub")

    with tvm.target.cce():
        sch = auto_schedule(res)
    config = {"name": kernel_name, "tensor_list": [tensor_var, tensor_value, res]}
    build(sch, config)
