#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
threshold_v2
"""

import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check


def threshold_v2_compute(x, threshold, value, kernel_name="threshold_v2"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        input tensor of x
    threshold : TVM tensor
        input tensor of threshold
    value : TVM tensor
        input tensor of value
    y : dict
        dict of y
    kernel_name : str
        kenel name, default value is "thershold_v2"

    Returns
    -------
    res: TVM tensor
        the result of threshold_v2_compute
    """
    inp_dtype = x.dtype
    compatible_dtype = x.dtype
    shape = x.shape
    # 'pylint: disable=unused-variable
    kernel_name_var = kernel_name

    if inp_dtype in ("int8", "uint8", "int32"):
        if tbe_platform.api_check_support("tbe.dsl.vcmpsel", "float32"):
            compatible_dtype = "float32"
        else:
            compatible_dtype = "float16"

        x = tbe.cast_to(x, compatible_dtype)
        threshold = tbe.cast_to(threshold, compatible_dtype)
        if value is not None:
            value = tbe.cast_to(value, compatible_dtype)

    threshold = tbe.broadcast(threshold, shape)
    if value is None:
        value = tbe.broadcast(tvm.const(0, compatible_dtype), shape)
    else:
        value = tbe.broadcast(value, shape)

    data_res = tbe.vcmpsel(x, threshold, 'gt', x, value)
    if inp_dtype != compatible_dtype:
        data_res = tbe.cast_to(data_res, inp_dtype)

    return data_res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
# 'pylint: disable=unused-argument,too-many-locals
def threshold_v2(x, threshold, value, y, kernel_name="threshold_v2"):
    """
    Thresholds each element of the input Tensor
    y = (x > threshold) ? x : value

    Parameters
    ----------
    x : dict
        shape and dtype of input
    threshold : dict
        shape and dtype of the value to threshold at
    value : dict
        shape and dtype of the value to replace with, default value is 0
    y : dict
        shape and dtype of output, should be the same shape and dtype as input
    kernel_name : str
        kernel name, default value is "threshold_v2"

    Returns
    -------
    output tensor
    """
    dtype_x = x.get("dtype").lower()
    check_list = ("float16", "float32", "int8", "int32", "uint8")
    para_check.check_dtype(dtype_x, check_list)
    para_check.check_shape(threshold.get("shape"), max_dim=1, max_rank=1, param_name="threshold")

    # Add dimensions to shape for spliting
    output_shape = y.get("shape")
    threshold_shape = threshold.get("shape")
    threshold_shape = tuple([1] * (len(output_shape) - len(threshold_shape))) + tuple(threshold_shape)

    tensor_list = []
    data_x = tvm.placeholder(x.get("shape"), dtype=x.get("dtype"), name="data_x")
    data_threshold = tvm.placeholder(threshold_shape, dtype=dtype_x, name="data_threshold")
    tensor_list.append(data_x)
    tensor_list.append(data_threshold)

    data_value = None
    if value is not None:
        para_check.check_shape(value.get("shape"), max_dim=1, max_rank=1, param_name="value")
        data_value = tvm.placeholder(threshold_shape, dtype=dtype_x, name="data_value")
        tensor_list.append(data_value)

    res = threshold_v2_compute(data_x, data_threshold, data_value, kernel_name)
    tensor_list.append(res)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": tensor_list}

    tbe.build(schedule, config)
