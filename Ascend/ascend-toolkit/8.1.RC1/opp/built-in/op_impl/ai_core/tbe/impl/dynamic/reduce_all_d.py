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
dynamic reduce_all_d
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import tbe_platform


# 'pylint: disable=unused-argument,invalid-name
# 'pylint: disable=redefined-argument-from-local
@register_operator_compute("ReduceAllD", op_mode="dynamic", support_fusion=True)
def reduce_all_d_compute(x, y, axes=None, keepdims=None, kernel_name="reduce_all_d"):
    """
    reduce_all_d compute

    Parameters:
    ----------
    x: TVM tensor
        input tensor.
    y: dict
        the dict of output tensor.
    axes: int, list, tuple or NONETYPE
        the axes for reduce.
    keepdims: bool or NONETYPE
        if true, retains reduced dimensions with length 1.
    kernel_name: str
        cce kernel name, default value is "reduce_all_d".

    Returns
    -------
    res: TVM tensor
         output tensor, has the same shape and type as input tensor.
    """
    shape = shape_util.shape_to_list(x.shape)
    shape_len = len(shape)
    if not axes:
        axes = range(shape_len)
    if hasattr(axes, 'index'):
        axes = list(axes)
    dtype = x.dtype

    zero_tensor_flg = 0 in x.shape
    reduce_all_support = tbe_platform.api_check_support('tbe.dsl.reduce_all', dtype)
    if zero_tensor_flg and reduce_all_support:
        res = tbe.reduce_all(x, axes, keepdims)
        return res

    data_fp16 = tbe.cast_to(x, "float16")
    data_abs = tbe.vabs(data_fp16)
    res_any = tbe.reduce_min(data_abs, axis=axes, keepdims=keepdims)
    res = tbe.cast_to(res_any, dtype)

    return res


# 'pylint: disable=too-many-locals,invalid-name
@register_operator("ReduceAllD")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def reduce_all_d(x, y, axes=None, keep_dims=None, kernel_name="reduce_all_d"):
    """
    reduce a tensor on a certain axes based on max.

    Parameters
    ----------
    x : dict
        shape and dtype of input
    y : dict
        shape and dtype of output, should be same shape and type as input
    axes: list
        the first axes to reduce,may be negative to index from the end
        (e.g., -1 for the last axes).
        axes may be int or list(e.g. [1,2])
    keep_dims: bool
        if true, retains reduced dimensions with length 1,
        default value is None
    kernel_name : str
        kernel name, default value is "reduce_all_d"

    Returns
    -------
    None
    """

    dtype = x.get("dtype")
    dtype_lower = dtype.lower()
    check_list = ("int8",)
    para_check.check_dtype(dtype_lower, check_list)
    x["rel_pos_to_reduce"] = "before"

    shape = x.get("shape")
    shape_len = len(shape)
    if not axes:
        axes = range(shape_len)
    if hasattr(axes, 'index'):
        axes = list(axes)
    axes = shape_util.axis_check(shape_len, axes)
    input_axis = {"shape": [len(axes), ], "value": axes, "rel_pos_to_reduce": "axis"}

    schedules = []
    tensors = []
    ins = classify([x, input_axis], OpPatternMode.REDUCE, {
        "keepdims": keep_dims is True,
        "ignore_fractal_format": False
    })

    for (x, axes) in ins:
        with tbe.compute():
            shape_var_new = shape_util.variable_shape([x, axes], op_mode="reduce")[0]
            data_input = tvm.placeholder(shape_var_new, name="data_input", dtype=dtype_lower)
            res = reduce_all_d_compute(data_input, y, axes.get("value"), keep_dims, kernel_name)
            tensors.append([data_input, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
