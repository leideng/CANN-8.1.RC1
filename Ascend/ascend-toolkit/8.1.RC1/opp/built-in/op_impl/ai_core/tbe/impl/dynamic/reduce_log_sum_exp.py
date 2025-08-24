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
dynamic reduce_log_sum_exp
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpImplMode
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import platform_info
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_context


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
@register_operator_compute("ReduceLogSumExp", op_mode="dynamic", support_fusion=True)
def reduce_log_sum_exp_compute(x, axes, y, keep_dims=False,
                               kernel_name="reduce_log_sum",
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
    if compatible_dtype == "float16":
        x = tbe.vadds(x, tvm.const(-Constant.fp16_x_sub, x.dtype))
    res_exp = tbe.vexp(x)
    reduced = tbe.reduce_sum(res_exp,
                             axes,
                             keep_dims)
    if reduced.dtype != compatible_dtype:
        reduced = tbe.cast_to(reduced, compatible_dtype)
    res = tbe.vlog(reduced, impl_mode=impl_mode)
    if compatible_dtype == "float16":
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
    axes : dict
        shape and dtype of input
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
    check_list_x = ("float16", "float32")
    shape_x = x.get("shape")
    dtype_lower_x = x.get("dtype").lower()
    para_check.check_dtype(dtype_lower_x, check_list_x,
                           param_name="x")
    para_check.check_shape(shape_x, param_name="x")
    x["rel_pos_to_reduce"] = "before"

    check_list_axes = ("int32", "int64")
    dtype_lower_axes = axes.get("dtype").lower()
    para_check.check_dtype(dtype_lower_axes, check_list_axes,
                           param_name="axes")
    axes["rel_pos_to_reduce"] = "axis"

    tbe_context.get_context().add_compile_info("axes_idx", 1)
    if "const_value" in axes.keys():
        axes["value"] = list(axes["const_value"])

    schedules = []
    ins = classify([x, axes], OpPatternMode.REDUCE, {"keepdims": keep_dims is True})
    tensors = []

    for (_x, _axes) in ins:
        with tbe.compute():
            shape_x, shape_axes = shape_util.variable_shape([_x, _axes], op_mode="reduce")
            data_input_x = tvm.placeholder(shape_x, name="data_input_x",
                                           dtype=dtype_lower_x)
            data_input_axes = tvm.placeholder(shape_axes, name="data_input_axes",
                                              dtype=dtype_lower_axes)
            axes_d = shape_util.axis_check(len(shape_x), _axes.get("value"))
            res = reduce_log_sum_exp_compute(data_input_x, axes_d, y, keep_dims, impl_mode=impl_mode)
            tensors.append([data_input_x, data_input_axes, res])

        with tvm.target.cce():
            schedule = tbe.auto_schedule(res)
        schedules.append(schedule)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
