# Copyright 2021 Huawei Technologies Co., Ltd
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
reduce_std_v2_update
"""
import math
import operator as op
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import OpPatternMode


# 'pylint: disable=invalid-name,too-many-locals,unused-argument,too-many-arguments,too-many-branches
def reduce_std_check_dim(axis_dim, shape_x, dim):
    """
    reduce_std_check_dim
    """
    dim_number = len(shape_x)

    for i in dim:
        if ((i < 0) and ((i + dim_number) in axis_dim)) or (i in axis_dim):
            continue
        axis_dim.append(int((i + dim_number) % dim_number))
    return axis_dim


@register_operator_compute("ReduceStdV2Update", op_mode="dynamic", support_fusion=True)
def reduce_std_v2_update_compute(x, mean, dim, if_std, unbiased, keepdim, correction,
                                 kernel_name="reduce_std_v2_update"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of X
    mean : TVM tensor
        the mean of X
    dim : intlist
        dimension to calculate
    if_std : bool
        control whether the output is standard deviation or variance, default value is False
    unbiased : bool
        control Bessel deviation, default value is True
    keepdim : bool
        hold dimension or not, default value is False
    correction: int
        if unbiased is true, Bessel's correction will be used, default value is 1
    kernel_name: str
        kernel name

    Returns
    -------
    output TVM tensors
    """
    x_type = x.dtype.lower()

    if x_type == "float16" or x_type == "bfloat16":
        x = tbe.cast_to(x, "float32")
        mean = tbe.cast_to(mean, "float32")

    shape_x = shape_util.shape_to_list(x.shape)

    axis_dim = []
    axis_dim = reduce_std_check_dim(axis_dim, shape_x, dim)

    reduce_ele = 1.0
    for i in axis_dim:
        reduce_ele *= shape_x[i]
    dtype = x.dtype

    x_sub = tbe.vsub(x, mean)
    var_mul = tbe.vmul(x_sub, x_sub)

    if unbiased:
        if isinstance(reduce_ele, float):
            cof_unbiased = 1.0
            if reduce_ele > correction:
                cof_unbiased = (reduce_ele - correction)**(-1)
                cof_unbiased = tvm.const(cof_unbiased, dtype=dtype)
        else:
            cof_unbiased = tbe.var("cof_unbiased", dtype=dtype)
            if dtype == "float16" or dtype == "bfloat16":
                tbe.var("cof_empty", dtype=dtype)
            tbe_context.get_context().add_compile_info("reduce_mean_cof_dtype", dtype)
        var_muls = tbe.vmuls(var_mul, cof_unbiased)
    else:
        if isinstance(reduce_ele, float):
            cof = reduce_ele ** (-1)
            cof = tvm.const(cof, dtype=dtype)
        else:
            cof = tbe.var("cof", dtype=dtype)
            if dtype == "float16" or dtype == "bfloat16":
                tbe.var("cof_empty", dtype=dtype)
            tbe_context.get_context().add_compile_info("reduce_mean_cof_dtype", dtype)
        var_muls = tbe.vmuls(var_mul, cof)

    var = tbe.reduce_sum(var_muls, axis=dim, keepdims=keepdim)

    if if_std:
        std = tbe.vsqrt(var, impl_mode="high_precision")
        if x_type == "bfloat16":
            std = tbe.round(std, "bfloat16")
        elif x_type == "float16":
            std = tbe.cast_to(std, "float16")
        return std
    if x_type == "bfloat16":
        var = tbe.round(var, "bfloat16")
    elif x_type == "float16":
        var = tbe.cast_to(var, "float16")
    return var


@register_operator("ReduceStdV2Update")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def reduce_std_v2_update(x, mean, output_var, dim, if_std=False, unbiased=True, keepdim=False,
                         correction=1, kernel_name="reduce_std_v2_update"):
    """
    calculating data

    Parameters
    ----------
    x: dict
        input tensor
    mean: dict
        mean value of input tensor
    output_var: dict
        output, variance or standard deviation
    dim: list[int]
        dimension to calculate
    if_std : bool
        control whether the output is standard deviation or variance, default value is False
    unbiased: bool
        control Bessel deviation, default value is True
    keepdims: bool
        hold dimension or not, default value is False
    correction: int
        if unbiased is true, Bessel's correction will be used, default value is 1
    kernel_name: str
        cce kernel name, default value is reduce_std_v2_update

    Returns
    -------
    None
    """
    check_list = ("float16", "float32", "bfloat16")

    shape_x = x.get("shape")
    dtype_x = x.get("dtype").lower()
    
    para_check.check_shape(shape_x, param_name="x")
    para_check.check_dtype(dtype_x, check_list, param_name="x")

    mean["rel_pos_to_reduce"] = "before"
    x["rel_pos_to_reduce"] = "before"
    if dim is None:
        input_axis = {"shape": [-1], "rel_pos_to_reduce": "axis"}
    else:
        dim = list(dim)
        dim = shape_util.axis_check(len(shape_x), dim)
        input_axis = {"shape": [len(dim), ], "value": dim, "rel_pos_to_reduce": "axis"}

    schedules, tensors = [], []
    ins = classify([x, mean, input_axis], OpPatternMode.REDUCE, {"keepdims": keepdim is True})

    for(_input_x, _mean, _axes) in ins:
        with tbe.compute():
            x_var_new, mean_var_new = shape_util.variable_shape([_input_x, _mean, _axes],
                                                                op_mode="reduce")[0:2]
            data_x = tvm.placeholder(x_var_new, name="data_x", dtype=dtype_x)
            data_mean = tvm.placeholder(mean_var_new, name="data_mean", dtype=dtype_x)
            res = reduce_std_v2_update_compute(data_x, data_mean, _axes.get("value"),
                                               if_std, unbiased, keepdim, correction, kernel_name)
            tensors.append([data_x, data_mean, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)