#!/usr/bin/env python
# coding: utf-8
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
strided read operator
"""
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector


STRIDED_READ_TAG = "strided_read"


# 'pylint: disable=invalid-name,unused-argument,unused-variable
def check_params(x, y, axis):
    """
    check the parameters including x, y, axis.
    """
    if len(x.get("shape")) != 5:
        error_manager_vector.raise_err_specific_reson("strided_read", "x's length must be 5 \
                                                      while length is{}.".format(len(x.get("shape"))))
    if len(y.get("shape")) != 5:
        error_manager_vector.raise_err_specific_reson("strided_read", "y's length must be 5 \
                                                      while length is{}.".format(len(y.get("shape"))))
    if x.get("dtype") not in ("float16", "int8"):
        error_manager_vector.raise_err_input_dtype_not_supported("strided_read", "x", "float16 or int8",
                                                                 x.get("dtype"))
    if y.get("dtype") not in ("float16", "int8"):
        error_manager_vector.raise_err_input_dtype_not_supported("strided_read", "y", "float16 or int8",
                                                                 y.get("dtype"))
    if x.get("format") != "NC1HWC0":
        error_manager_vector.raise_err_input_format_invalid("strided_read", "x", "NC1HWC0", x.get("format"))
    if y.get("format") != "NC1HWC0":
        error_manager_vector.raise_err_input_format_invalid("strided_read", "y", "NC1HWC0", y.get("format"))
    if x.get("dtype") != y.get("dtype"):
        error_manager_vector.raise_err_inputs_dtype_not_equal("strided_read", "x", "y",
                                                              x.get("dtype"), y.get("dtype"))
    if axis != 1:
        error_manager_vector.raise_err_input_value_invalid("strided_read", "axis", "1", str(axis))


@register_operator_compute("strided_read", op_mode="static", support_fusion=True)
def strided_read_compute(x, y, axis, stride, kernel_name='strided_read'):
    """
    read data from tensor by stride.

    Parameters:
    ----------
    x: placeholder of input tesnor.

    y: dict of output tensor.

    axis: which axis to read data by stride.

    stride: data read stride.

    kernel_name: cce kernel name, default value is "strided_read".

    Returns:
    ----------
    output_y: result tensor.
    """
    output_y = tvm.compute(
        y.get("shape"),
        lambda batch_idx, c1_idx, h_idx, w_idx, c0_idx:
        x[batch_idx, c1_idx, h_idx, w_idx, c0_idx],
        name="strided_read",
        tag=STRIDED_READ_TAG,
        attrs=x.op.attrs)

    return output_y


@para_check.check_input_type(dict, dict, int, int, str)
def strided_read(x, y, axis, stride, kernel_name='strided_read'):
    """
    read data from tensor by stride.

    Parameters:
    ----------
    x: dict of input.

    y: dict of output.

    axis: which axis to read data by stride.

    stride: data read stride.

    kernel_name: cce kernel name, default value is "strided_read".

    Returns:
    -------
    None
    """

    check_params(x, y, axis)
    shape_x = x.get("shape")
    dtype_x = x.get("dtype")

    input_x = tvm.placeholder(shape_x, name="input_x", dtype=dtype_x)
    res = strided_read_compute(input_x, y, axis, stride, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
