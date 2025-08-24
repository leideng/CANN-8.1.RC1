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
static reduce_log_sum_exp
"""


from te.utils import shape_util
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import OpImplMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import platform_info
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


class Constant:
    """
    The class for constant

    Parameters:
    ----------
    fp16_x_sub: constant
        When compatible_dtype is float16, input minus fp16_x_sub
    -------
    """
    fp16_x_sub = 9


# 'pylint: disable=unused-argument,invalid-name,redefined-argument-from-local,too-many-arguments
@register_operator_compute("ReduceLogSumExp", op_mode="static", support_fusion=True)
def reduce_log_sum_exp_compute(x, axes, y, keep_dims=False,
                               kernel_name="reduce_log_sum_exp",
                               impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """
    reduce_log_sum_exp compute

    Parameters:
    ----------
    x: TVM tensor
        input tensor.
    axes: int, list, tuple or NONETYPE
        the axes for reduce.
    y: dict
        the dict of output tensor.
    keep_dims: bool or NONETYPE
        if true, retains reduced dimensions with length 1.
    impl_mode: string
        "high_performance" mode or "high_precision" mode
    kernel_name: str
        cce kernel name, default value is "reduce_log_sum_exp".

    Returns
    -------
    res: TVM tensor
        output tensor, has the same type as input tensor.
    """
    x_dtype = x.dtype
    compatible_dtype = x.dtype
    if platform_info.api_check_support("tbe.dsl.vexp", "float32"):
        compatible_dtype = "float32"
    else:
        compatible_dtype = "float16"
    if x_dtype != compatible_dtype:
        x = tbe.cast_to(x, compatible_dtype)
    if compatible_dtype == "float16" and impl_mode == "high_precision":
        x = tbe.vadds(x, tvm.const(-Constant.fp16_x_sub, x.dtype))
    res_exp = tbe.vexp(x)
    reduced = tbe.reduce_sum(res_exp,
                             axes,
                             keep_dims)
    if reduced.dtype != compatible_dtype:
        reduced = tbe.cast_to(reduced, compatible_dtype)
    res = tbe.vlog(reduced)
    if compatible_dtype == "float16" and impl_mode == "high_precision":
        res = tbe.vadds(res, tvm.const(Constant.fp16_x_sub, res.dtype))
    if x_dtype != compatible_dtype:
        res = tbe.cast_to(res, x_dtype)
    return res


# 'pylint: disable=too-many-locals,invalid-name,too-many-arguments
@register_operator("ReduceLogSumExp")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def reduce_log_sum_exp(x, axes, y, keep_dims=False,
                       kernel_name="reduce_log_sum_exp",
                       impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """reduce a tensor on a certain axes based on sum.

    Parameters:
    ----------
    x: dict
        the dict of input tensor.
    axes: dict
        the axes for reduce.
    y: dict
        the dict of output tensor.
    keep_dims: bool or NONETYPE
        if true, retains reduced dimensions with length 1.
    impl_mode: string
        "high_performance" mode or "high_precision" mode
    kernel_name: str
        cce kernel name, default value is "reduce_log_sum_exp".

    Returns
    -------
    None
    """
    shape = x.get("shape")
    dtype = x.get("dtype")
    dtype_lower = dtype.lower()
    check_list = ("float16", "float32")

    para_check.check_shape(shape, param_name="x")
    para_check.check_dtype(dtype_lower, check_list, param_name="x")

    dtype_axes = axes.get("dtype")
    shape_axes = axes.get("shape")
    dtype_lower_axes = dtype_axes.lower()
    check_list_axes = ("int32", "int64")
    para_check.check_dtype(dtype_lower_axes, check_list_axes, param_name="axes")
    axes["rel_pos_to_reduce"] = "axis"

    if "const_value" in axes.keys():
        axes["value"] = list(axes["const_value"])

    data_input = tvm.placeholder(shape, name="data_input_" + kernel_name, dtype=dtype_lower)
    data_input_axes = tvm.placeholder(shape_axes, name="data_input_axes" , dtype=dtype_lower_axes)
    res = reduce_log_sum_exp_compute(data_input, axes.get("value"), y, keep_dims, kernel_name, impl_mode)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_input, data_input_axes, res]}
    tbe.build(sch, config)
