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
data_compare
"""
import operator
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=too-few-public-methods,not-use-list-comprehension
class Constant:
    """
    Constant
    """
    NUM_ONE = 1.0
    NUM_ZERO = 0.0


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=too-many-locals
@register_operator_compute("data_compare", op_mode="static", support_fusion=True, support_bfp16=True)
def data_compare_compute(input_x, input_y, output_num, atol, rtol, kernel_name="data_compare"):
    """
    algorithm: data_compare

    calculating abs(x-y) > atol + rtol * abs(y)

    Parameters
    ----------
    input_x : the placeholders of input data
    input_y : the placeholders of input data
    atol: default 1e-5
    rtol: default 1e-3
    output_num: shape and dtype of output
    kernel_name: cce kernel name, default value is "data_compare"
    Returns
    -------
    the function of _data_compare_compute
    """

    input_dtype = input_x.dtype
    shape = input_x.shape
    shape_list = []
    for i in range(len(shape)):
        shape_list.append(i)

    res_vsub = tbe.vsub(input_x, input_y)
    res_vabs = tbe.vabs(res_vsub)

    atol_scaler = tvm.const(atol, input_dtype)
    rtol_scaler = tvm.const(rtol, input_dtype)

    y_vabs = tbe.vabs(input_y)

    res_muls = tbe.vmuls(y_vabs, rtol_scaler)
    res_vadds = tbe.vadds(res_muls, atol_scaler)

    res_cmp = tbe.vcmp(res_vabs, res_vadds, 'gt')

    zero_scaler = tvm.const(Constant.NUM_ZERO, "float16")
    one_scaler = tvm.const(Constant.NUM_ONE, "float16")
    res_sel = tbe.vsel(res_cmp, one_scaler, zero_scaler)

    if tbe_platform.api_check_support("tbe.dsl.reduce_sum", "float32"):
        res_sel = tbe.cast_to(res_sel, "float32")
    else:
        res_sel = tbe.cast_to(res_sel, "float16")

    res_num = tbe.sum(res_sel, axis=shape_list)
    res_num = tbe.cast_to(res_num, "float32")

    return res_num


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def data_compare(input_x, input_y, output_num, atol=1e-5, rtol=1e-3, kernel_name="data_compare"):
    """
    abs(x-y) > atol + rtol * abs(y)
    Parameters
    ----------
    input_x : dict, include shape and dtype, support fp16 and fp32
        shape of tensors, assume src_shape equals dst_shape

    input_y : dict, include shape and dtype, support fp16 and fp32
        shape of tensors, assume src_shape equals dst_shape

    output_num : dict, include shape and dtype, reserve

    atol: default 1e-5
    rtol: default 0.001

    kernel_name : str
        cce kernel name, default value is "data_compare"

    Returns
    ------
    None
    """

    shape_x = input_x.get("shape")
    shape_y = input_y.get("shape")
    in_dtype = input_x.get("dtype")
    in_y_dtype = input_y.get("dtype")

    if atol < 0:
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "atol", \
                                                           ">= 0", atol)
    if rtol < 0:
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "rtol", \
                                                           ">= 0", rtol)

    # check shape
    if not operator.eq(shape_x, shape_y):
        error_detail = "shape of input_x and input_y should be same"
        error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "input_x", \
                                                               "input_y", error_detail)
    para_check.check_shape(shape_x, param_name="input_x")
    shape_x, _ = shape_util.refine_shape_axes(shape_x, [])
    shape_y, _ = shape_util.refine_shape_axes(shape_y, [])

    # check input tensor data_type
    check_list = ("bfloat16", "float16", "float32", "int8", "uint8", "int32")

    para_check.check_dtype(in_dtype, check_list, param_name="input_x")
    para_check.check_dtype(in_y_dtype, check_list, param_name="input_y")
    in_dtype = input_x.get("dtype").lower()
    in_y_dtype = input_y.get("dtype").lower()
    if not operator.eq(in_dtype, in_y_dtype):
        error_detail = "dtype of input_x and input_y should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "input_x", \
                                                               "input_y", error_detail)

    in_data_x = tvm.placeholder(shape_x, name="shape_x", dtype=in_dtype)
    in_data_y = tvm.placeholder(shape_y, name="shape_y", dtype=in_dtype)

    res = data_compare_compute(in_data_x, in_data_y, output_num, atol, rtol, kernel_name)

    with tvm.target.cce():
        auto_sch = auto_schedule(res)

    tensor_list = [in_data_x, in_data_y, res]

    config = {"name": kernel_name,
              "tensor_list": tensor_list,
              "bool_storage_as_1bit": False}
    build(auto_sch, config)
