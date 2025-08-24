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
dynamic reduce_mean_with_count
"""
import collections

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import OpImplMode
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=too-many-branches,too-many-arguments,too-many-locals
# 'pylint: disable=unused-argument,invalid-name
@register_operator_compute("ReduceMeanWithCount", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def reduce_mean_with_count_compute(x,
                                   count,
                                   count_sum,
                                   y,
                                   axes,
                                   keep_dims=None,
                                   kernel_name="reduce_mean_with_count",
                                   impl_mode=OpImplMode.HIGH_PERFORMANCE,
                                   is_5hdc=False):
    """reduce_mean_with_count compute

    Parameters:
    ----------
    x: TVM tensor
        input tensor.
    count : TVM tensor
        weight of input
    count_sum : TVM tensor
        number of inputs
    y: TVM tensor
        the dict of output tensor.
    axes: int, list, tuple or NoneType
        the axes for reduce.
    keep_dims: bool or NoneType
        if true, retains reduced dimensions with length 1.
    kernel_name: str
        cce kernel name, default value is "reduce_mean_with_count".

    Returns
    -------
    res: TVM tensor
        output tensor.
    """

    dtype = x.dtype
    if dtype == "float32":
        calc_dtype = "float32"
    elif dtype == "float16":
        cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
        if (not tbe_platform.api_check_support("te.lang.cce.sum", "float32") and
            not tbe_platform.api_check_support("te.lang.cce.vdiv", "float32")):
            calc_dtype = "float16"
        elif cce_product == "Ascend310" and impl_mode == OpImplMode.HIGH_PERFORMANCE:
            calc_dtype = "float16"
        else:
            calc_dtype = "float32"
    else:
        # int8 and uint8
        calc_dtype = "float16"

    if dtype != calc_dtype:
        data_x_tmp = tbe.cast_to(x, calc_dtype)
        data_count_tmp = tbe.cast_to(count, calc_dtype)
        data_count_sum_tmp = tbe.cast_to(count_sum, calc_dtype)
    else:
        data_x_tmp = x
        data_count_tmp = count
        data_count_sum_tmp = count_sum

    data_x_tmp_count = tbe.vmul(data_x_tmp, data_count_tmp)
    data_x_tmp_mean = tbe.vdiv(data_x_tmp_count, data_count_sum_tmp)
    res = tbe.reduce_sum(data_x_tmp_mean, axis=axes, keepdims=keep_dims)

    if dtype != calc_dtype:
        if dtype in ("int8", "uint8"):
            res = tbe.cast_to(res, dtype, False)
        else:
            res = tbe.cast_to(res, dtype)

    return res


@register_operator("ReduceMeanWithCount")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def reduce_mean_with_count(x, count, count_sum, y, axes=None, keep_dims=False,
                           kernel_name="reduce_mean_with_count", impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """
    Reduce a tensor on a certa in axes based on mean.

    Parameters:
    ----------
    x : dict
        shape and dtype of input
    count : dict
        weight of input
    count_sum : dict
        number of inputs
    y: dict
        shape and dtype of output
    axes : int, list, tuple, NoneType
        The dimensions to reduce. If None (the default), reduces all dimensions.
        Must be in the range [-rank(input_tensor), rank(input_tensor)).
    keep_dims : bool, NoneType
        if true, retains reduced dimensions with length 1,
        default value is None.
    kernel_name : str
        cce kernel name, default value is reduce_mean_with_count

    Returns
    -------
    None
    """
    dtype_x = x["dtype"]
    dtype_x_lower = dtype_x.lower()
    dtype_count = count["dtype"]
    dtype_count_lower = dtype_count.lower()
    dtype_count_sum = count_sum["dtype"]
    dtype_count_sum_lower = dtype_count_sum.lower()

    check_list = ("float16", "float32", "int8", "uint8", "bfloat16")
    para_check.check_dtype(dtype_x_lower, check_list)
    x["rel_pos_to_reduce"] = "before"
    para_check.check_dtype(dtype_count_lower, check_list)
    count["rel_pos_to_reduce"] = "before"
    para_check.check_dtype(dtype_count_sum_lower, check_list)
    count_sum["rel_pos_to_reduce"] = "before"

    if axes is None:
        input_axis = {"shape": [-1], "rel_pos_to_reduce": "axis"}
    else:
        shape_x = x["shape"]
        dims = len(shape_x)
        if axes == []:
            axes = list(range(dims))
        else:
            axes = list(axes)
        axes = shape_util.axis_check(dims, axes)
        input_axis = {"shape": [len(axes), ], "value": axes, "rel_pos_to_reduce": "axis"}

    schedules = []
    tensors = []
    ins = classify([x, count, count_sum, input_axis], OpPatternMode.REDUCE,
                   {"keepdims": keep_dims is True})
    for (_x, _count, _count_sum, _axes) in ins:
        with tbe.compute():
            # not support 5HD
            is_5hdc = False
            [shape_x, shape_count, shape_count_sum] = \
                shape_util.variable_shape([_x, _count, _count_sum, _axes], op_mode="reduce")[:3]
            data_x = tvm.placeholder(shape_x, name="data_x", dtype=dtype_x_lower)
            data_count = tvm.placeholder(shape_count, name="data_count", dtype=dtype_count_lower)
            data_count_sum = tvm.placeholder(shape_count_sum, name="data_count_sum", dtype=dtype_count_sum_lower)
            res = reduce_mean_with_count_compute(data_x, data_count, data_count_sum, y, _axes.get("value"),
                                                 keep_dims, impl_mode=impl_mode, is_5hdc=is_5hdc)
            tensors.append([data_x, data_count, data_count_sum, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
