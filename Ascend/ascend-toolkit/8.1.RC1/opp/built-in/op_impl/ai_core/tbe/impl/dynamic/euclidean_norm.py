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
dynamic euclidean_norm
"""
# 'pylint: disable=W0613,W0401
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_context


# 'pylint: disable=unused-argument,invalid-name,redefined-argument-from-local,too-many-arguments
@register_operator_compute("EuclideanNorm", op_mode="dynamic", support_fusion=True)
def euclidean_norm_compute(x, axes, y, keep_dims=False, kernel_name="euclidean_norm"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of input_x
    axes: TVM tensor, defaults to "None"
        the dimensions to reduce
    y : TVM tensor
        the placeholder of output_x
    keep_dims: optional bool, defaults to "False"
        if true, retains reduced dimensions with length 1
    kernel_name : str
        kernel name, default value is "euclidean_norm"

    Returns
    -------
    res: TVM tensor
        the calculation results
    """
    dtype = x.dtype.lower()
    if dtype != "float32":
        x = tbe.cast_to(x, "float32")

    res_mul = tbe.vmul(x, x)
    res_sum = tbe.reduce_sum(res_mul, axes, keep_dims)
    res = tbe.vsqrt(res_sum)

    if res.dtype != dtype:
        res = tbe.cast_to(res, dtype)
    return res


# 'pylint: disable=too-many-locals,invalid-name,too-many-arguments
@register_operator("EuclideanNorm")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def euclidean_norm(x, axes, y, keep_dims=False, kernel_name="euclidean_norm"):

    """
    calculating data

    Parameters
    ----------
    x : dict
        shape and dtype of input
    axes : dict, defaults to "None"
        the first axes to reduce, may be negative to index from the end
    y : dict
        shape and dtype of output, should be same format and type as input
    keepdims : bool, defaults to "False"
        if true, retains reduced dimensions with length 1

    Returns
    -------
    None
    """
    input_check_list = ["float16", "float32", "int32"]
    input_dtype = x.get("dtype").lower()
    para_check.check_dtype(input_dtype, input_check_list, param_name="x")
    x["rel_pos_to_reduce"] = "before"

    input_shape = x.get("shape")
    para_check.check_shape(input_shape, param_name="x")

    check_list_axes = ("int32", "int64")
    dtype_lower_axes = axes.get("dtype").lower()
    para_check.check_dtype(dtype_lower_axes,
                           check_list_axes,
                           param_name="axes")
    axes["rel_pos_to_reduce"] = "axis"

    tbe_context.get_context().add_compile_info("axes_idx", 1)
    if "const_value" in axes.keys():
        axes["value"] = list(axes["const_value"])

    if isinstance(axes, int):
        axes = [axes]
    if axes is None or len(axes) == 0:
        axes = list(range(len(input_shape)))

    schedules = []
    ins = classify([x, axes], OpPatternMode.REDUCE, {"keepdims": keep_dims is True})
    tensors = []

    for (_x, _axes) in ins:
        with tbe.compute():
            shape_x, shape_axes = shape_util.variable_shape([_x, _axes], op_mode="reduce")
            data_x = tvm.placeholder(shape_x, name="data_x",
                                           dtype=input_dtype)
            data_axes = tvm.placeholder(shape_axes, name="data_axes",
                                              dtype=dtype_lower_axes)
            axes_d = shape_util.axis_check(len(shape_x), _axes.get("value"))
            res = euclidean_norm_compute(data_x, axes_d, y, keep_dims, kernel_name)
            tensors.append([data_x, data_axes, res])

        with tvm.target.cce():
            schedule = tbe.auto_schedule(res)
        schedules.append(schedule)

    # build
    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
