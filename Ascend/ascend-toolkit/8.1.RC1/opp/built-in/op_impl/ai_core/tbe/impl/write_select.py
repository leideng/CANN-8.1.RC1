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
write_select
"""
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm

WRITE_SELECT_TAG = "write_select"
PARA_LIST_LEN = 5
NAME_INDEX = [0]


# 'pylint: disable=locally-disabled,unnecessary-lambda,unused-argument
@register_operator_compute("write_select", op_mode="static", support_fusion=True)
def write_select_compute(input_tensor, output_x, kernel_name="write_select"):
    """
    calculating data

    Parameters
    ----------
    input_tensor : TVM tensor
        the input tensor
    output_x : dict
        dict of output_x, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "write_select"

    Returns
    -------
    output tensor
    """
    input_shape = input_tensor.shape
    valid_shape = output_x.get("valid_shape")

    if len(valid_shape) != PARA_LIST_LEN:
        error_manager_vector.raise_err_specific_reson("write_select", "the len of \
                                                      valid shape should be 5")

    _, _, h_valid, w_valid, c0_valid = valid_shape

    compute_name = "res_write_select" + "_" + str(NAME_INDEX[0])
    NAME_INDEX[0] += 1
    res = tvm.compute(input_shape, lambda *indice: input_tensor(*indice),
                      name=compute_name,
                      attrs={"HWC0": h_valid*w_valid*c0_valid},
                      tag=WRITE_SELECT_TAG)

    return res


# 'pylint: disable=locally-disabled,too-many-locals,unexpected-keyword-arg
@para_check.check_input_type(dict, dict, str)
def write_select(input_x, output_x, kernel_name="write_select"):
    """
    Write data with offset

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_x : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "write_select"

    Returns
    -------
    None
    """
    input_shape = input_x.get("shape")
    input_dtype = input_x.get("dtype").lower()
    valid_shape = output_x.get("valid_shape")

    para_check.check_shape_rule(input_shape)
    para_check.check_shape_rule(valid_shape)
    para_check.check_tensor_shape_size(input_shape)
    para_check.check_tensor_shape_size(valid_shape)
    para_check.check_kernel_name(kernel_name)

    if tbe_platform.get_soc_spec("SOC_VERSION") == tbe_platform.HI3796CV300ESAIC:
        check_list = ["int32", "float16", "int8", "int16"]
    else:
        check_list = ["int32", "float16", "float32", "int8"]

    if input_dtype not in check_list:
        error_manager_vector.raise_err_input_dtype_not_supported("write_select", "input_x",
                                                                 ", ".join(check_list), input_dtype)

    if len(valid_shape) != PARA_LIST_LEN:
        error_manager_vector.raise_err_specific_reson("write_select", "the len of \
                                                      valid shape should be 5")

    dst_out_flag = "DDR"
    if "dst_out_flag" in output_x:
        dst_out_flag = output_x.get("dst_out_flag")

    input_tensor_ph = tvm.placeholder(input_shape,
                                      name="input_tensor_ph",
                                      dtype=input_dtype,
                                      attrs={"valid_shape": valid_shape,
                                             "dst_out_flag": dst_out_flag})

    input_tensor = tvm.compute(input_shape,
                               lambda *indice: input_tensor_ph(*indice),
                               name="input_tensor")
    res = write_select_compute(input_tensor, output_x, kernel_name=kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [input_tensor_ph, res]}
    tbe.build(sch, config)
