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
fused_minimum_or_maximum_grad
"""

from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util


# 'pylint: disable=arguments-out-of-order
def _compare_value_int32(x_data, y_data, shape_dz):
    """
    The input data type of this function only support int32;
    The return value of the function: if x_data >= y_data return 1; else return 0.
    """
    min_value_int = tvm.const(1, dtype="int32")
    data_zero_int = tvm.const(0, dtype="int32")
    min_value_tensor = tbe.broadcast(min_value_int, shape_dz)
    data_zero_int_tensor = tbe.broadcast(data_zero_int, shape_dz)
    sub_xy = tbe.vsub(x_data, y_data)
    add_min = tbe.vadd(sub_xy, min_value_tensor)
    vmax_zero = tbe.vmax(add_min, data_zero_int_tensor)
    result = tbe.vmin(vmax_zero, min_value_tensor)

    return result


# 'pylint: disable = locally-disabled,too-many-locals
def _compare_value_float(x_data, y_data):
    """
    The input data type of the function only support float;
    The return value of the function: if x_data >= y_data return 1; else return 0.
    """
    # The smallest positive subnormal number of float32 is 2**(-126)
    min_value = tvm.const(2**(-126), dtype="float32")
    # `(2**(-126))*(2**(62))*(2**(62))*(2**(2)) = 1`
    # so `min_value*max_value*max_value*max_value_1 = 1`
    max_value = tvm.const(2**(62), dtype="float32")
    max_value_1 = tvm.const(2**(2), dtype="float32")

    data_zero = tbe.vmuls(x_data, 0)
    min_value_tensor = tbe.vadds(data_zero, min_value)
    max_value_tensor = tbe.vadds(data_zero, max_value)
    max_value_1_tensor = tbe.vadds(data_zero, max_value_1)
    sub_xy = tbe.vsub(x_data, y_data)
    add_min_value = tbe.vadds(sub_xy, min_value)
    vmax_zero = tbe.vmax(add_min_value, data_zero)
    vmin_min_value = tbe.vmin(vmax_zero, min_value_tensor)
    vmul_max_value = tbe.vmul(vmin_min_value, max_value_tensor)
    vmul_max_value_1 = tbe.vmul(vmul_max_value, max_value_tensor)
    result = tbe.vmul(vmul_max_value_1, max_value_1_tensor)

    return result


def _compare_value(x_data, y_data, dtype, shape_dz):
    """
    The input data type of the function only support float and int32;
    The return value of the function: if x_data >= y_data return 1; else return 0.
    """
    dtype = dtype.lower()
    if dtype == "int32":
        compare_value_data = _compare_value_int32(x_data, y_data, shape_dz)
    else:
        compare_value_data = _compare_value_float(x_data, y_data)

    return compare_value_data


def _calculate_result_le(x_data, y_data, dz_data, dtype, shape_dz):
    """
    The input data type of the function only support float int32 dtype;
    The return value of the function: if y_data >= x_data : result_dx = dz_data, result_dy = 0;
    else result_dx = 0,result_dx = dz_data.
    """
    minus_one = tvm.const(-1, dtype="int32")

    minus_one_tensor = tbe.broadcast(minus_one, shape_dz)
    # `if y_data >= x_data ; datax_select_le = 1; else datax_select_le =0;`
    datax_select_le = _compare_value(y_data, x_data, dtype, shape_dz)
    result_dx = tbe.vmul(dz_data, datax_select_le)
    select_reverse = tbe.vadd(datax_select_le, minus_one_tensor)
    select_dy = tbe.vmul(select_reverse, minus_one_tensor)
    result_dy = tbe.vmul(dz_data, select_dy)

    return result_dx, result_dy


def _calculate_result_ge(x_data, y_data, dz_data, dtype, shape_dz):
    """
    The input data type of the function only support float int32 dtype;
    The return value of the function: if x_data >= y_data : result_dx = dz_data, result_dy = 0;
    else result_dx = 0,result_dx = dz_data.
    """
    minus_one = tvm.const(-1, "int32")

    minus_one_tensor = tbe.broadcast(minus_one, shape_dz)
    # `if x_data >= y_data ; datax_select_ge = 1; else datax_select_ge =0;`
    datax_select_ge = _compare_value(x_data, y_data, dtype, shape_dz)
    result_dx = tbe.vmul(dz_data, datax_select_ge)
    select_reverse = tbe.vadd(datax_select_ge, minus_one_tensor)
    select_dy = tbe.vmul(select_reverse, minus_one_tensor)
    result_dy = tbe.vmul(dz_data, select_dy)

    return result_dx, result_dy


def _reduce_result(shape_x, shape_y, shape_dz, result_dx, result_dy):
    """
    If the shapes of the two input data are not equal,
    we need to call this function to do reduce operation.
    """
    if list(shape_x) != list(shape_dz):
        reduce_axis = []
        for i, shape_x_i in enumerate(shape_x):
            if shape_x_i == 1:
                reduce_axis.append(i)
        result_dx = tbe.sum(result_dx, axis=reduce_axis, keepdims=None)

    if list(shape_y) != list(shape_dz):
        reduce_axis = []
        for i, shape_y_i in enumerate(shape_y):
            if shape_y_i == 1:
                reduce_axis.append(i)
        result_dy = tbe.sum(result_dy, axis=reduce_axis, keepdims=None)

    return result_dx, result_dy


# 'pylint: disable = locally-disabled,invalid-name,too-many-arguments,unused-argument,no-member
@register_operator_compute("fused_minimum_or_maximum_grad_cce", op_mode="static", support_fusion=True)
def fused_minimum_or_maximum_grad_compute(placeholders, shape_x, shape_y, shape_dz, cmp_type,
                                          dtype,
                                          kernel_name="cce_fused_minimum_or_maximum_grad",
                                          need_build=False, need_print=False):
    """
    algorithm:
    calculating minimum or maximum_grad of the two input data

    Parameters
    ----------
    placeholders:TVM tensor.
        The tensor of inputs data
    shape_x: list or tuple.
        shape of data_inputx
    shape_y: list or tuple.
        shape of data_inputy
    shape_dz: list or tuple.
        shape of data_inputdz
    cmp_type: str
        LessEqual or GreatEqual
    dtype: str
        the data type, assume src_dtype equals dst_dtype,
        only support float16, float32, int32
    kernel_name: str
        cce kernel name, default value is "cce_fused_minimum_or_maximum_grad"
    need_build: bool
        if need to build CCEC kernel, default value is False
    need_print: bool
        if need to print the ir, default value is False

    Returns:
    -------
    results of minimum or maximum_grad of the two input data.
    """
    dz_data, inputx_data, inputy_data = placeholders
    if dtype == "float16":
        inputx_data = tbe.cast_to(inputx_data, "float32")
        inputy_data = tbe.cast_to(inputy_data, "float32")
        dz_data = tbe.cast_to(dz_data, "float32")
    inputx_data = tbe.broadcast(inputx_data, shape_dz)
    inputy_data = tbe.broadcast(inputy_data, shape_dz)

    if cmp_type == "LE":
        result_dx, result_dy = _calculate_result_le(inputx_data, inputy_data,
                                                    dz_data, dtype, shape_dz)
    if cmp_type == "GE":
        result_dx, result_dy = _calculate_result_ge(inputx_data, inputy_data,
                                                    dz_data, dtype, shape_dz)
    if list(shape_x) != list(shape_dz) or list(shape_y) != list(shape_dz):
        result_dx, result_dy = _reduce_result(shape_x, shape_y, shape_dz, result_dx, result_dy)

    if dtype == "float16":
        result_dx = tbe.cast_to(result_dx, "float16")
        result_dy = tbe.cast_to(result_dy, "float16")
    outs = [result_dx, result_dy]

    return outs


@para_check.check_input_type((list, tuple), (list, tuple), (list, tuple), bool, bool, str,
                             str, str, bool, bool)
def fused_minimum_or_maximum_grad_cce(shape_dz, shape_x, shape_y, grad_x=True, grad_y=True,
                                      cmp_type="LE", dtype="float32",
                                      kernel_name="cce_fused_minimum_or_maximum_grad",
                                      need_build=False, need_print=False):
    """
    algorithm:
    calculating minimum or maximum_grad of the two input data

    Parameters
    ----------
    shape_dz: list or tuple.
        shape of data_inputdz
    shape_x: list or tuple.
        shape of data_inputx
    shape_y: list or tuple.
        shape of data_inputy
    grad_x: bool
        if grad_x is true,output need return dx
    grad_y: bool
        if grad_y is true,output need return dy
    cmp_type: str
        LessEqual or GreatEqual
    dtype: str
        the data type, assume src_dtype equals dst_dtype,
        only support float16, float32, int32
    kernel_name: str
        cce kernel name, default value is "cce_fused_minimum_or_maximum_grad"
    need_build: bool
        if need to build CCEC kernel, default value is False
    need_print: bool
        if need to print the ir, default value is False

    Returns:
    -------
    none.
    """
    para_check.check_kernel_name(kernel_name)
    para_check.check_shape_rule(shape_x)
    para_check.check_shape_rule(shape_y)
    shape_x, shape_y, shape_max = shape_util.produce_shapes(shape_x, shape_y)
    para_check.check_shape_rule(shape_max)
    para_check.check_shape_size(shape_max)
    if list(shape_dz) != list(shape_max):
        raise RuntimeError("fused_minimum_or_maximum_grad_cce shape_dz != shape_max")

    dtype = dtype.lower()
    if dtype not in ["float16", "float32", "int32"]:
        raise RuntimeError("fused_minimum_or_maximum_grad_cce only support"
                           " float16, float32, int32")

    if (grad_x, grad_y) == (False, False):
        raise RuntimeError("grad_x and grad_x at least one is true")

    placeholders = []
    placeholders.append(tvm.placeholder(shape_dz, name="input_dz", dtype=dtype))
    placeholders.append(tvm.placeholder(shape_x, name="input_x", dtype=dtype))
    placeholders.append(tvm.placeholder(shape_y, name="input_y", dtype=dtype))

    outs = fused_minimum_or_maximum_grad_compute(placeholders, shape_x, shape_y,
                                                 shape_dz, cmp_type, dtype)

    with tvm.target.cce():
        if (grad_x, grad_y) == (True, False):
            sch = auto_schedule(outs[0])
            outs = [outs[0]]
        if (grad_x, grad_y) == (False, True):
            sch = auto_schedule(outs[1])
            outs = [outs[1]]
        if (grad_x, grad_y) == (True, True):
            sch = auto_schedule(outs)

    config = {"print_ir": need_print,
              "need_build": need_build,
              "name": kernel_name,
              "tensor_list": placeholders + outs}

    build(sch, config)
