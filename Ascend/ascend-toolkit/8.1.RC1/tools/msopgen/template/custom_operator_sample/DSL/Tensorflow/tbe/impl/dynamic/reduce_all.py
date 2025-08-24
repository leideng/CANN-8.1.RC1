# Copyright 2020 Huawei Technologies Co., Ltd
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
reduce all
"""
from ..util.platform_adapter import tbe
from ..util.platform_adapter import tvm
from ..util.platform_adapter import classify
from ..util.platform_adapter import OpPatternMode
from ..util.platform_adapter import shape_util
from ..util.platform_adapter import para_check
from ..util.platform_adapter import register_operator


# 'pylint: disable=unused-argument,invalid-name
def reduce_all_compute(x, axes, y, keepdims=None, kernel_name="reduce_all"):
    """
    reduce_all compute

    Parameters:
    ----------
    x: TVM tensor
        input tensor.
    axes: int, list, tuple or NONETYPE
        the axes for reduce.
    y: dict
        the dict of output tensor.
    keepdims: bool or NONETYPE
        if true, retains reduced dimensions with length 1.
    kernel_name: str
        cce kernel name, default value is "reduce_all".

    Returns
    -------
    res: TVM tensor
        output tensor, has the same type as input tensor.
    """
    dtype = x.dtype
    data_fp16 = tbe.cast_to(x, "float16")
    data_abs = tbe.vabs(data_fp16)
    res_any = tbe.reduce_min(data_abs, axis=axes, keepdims=keepdims)
    res = tbe.cast_to(res_any, dtype)

    return res


# 'pylint: disable=too-many-locals,invalid-name
@register_operator("ReduceAll")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                 para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def reduce_all(x, axes, y, keepdims=False, kernel_name="reduce_all"):
    """reduce a tensor on a certain axes based on all.

    Parameters:
    ----------
    x: dict
        the dict of input tensor.
    axes: dict
        the axes for reduce.
    y: dict
        the dict of output tensor.
    keepdims: bool or NONETYPE
        if true, retains reduced dimensions with length 1.
    kernel_name: str
        cce kernel name, default value is "reduce_all".

    Returns
    -------
    None
    """

    dtype_x = x["dtype"]
    dtype_lower_x = dtype_x.lower()
    check_list_x = ("int8",)
    para_check.check_dtype(dtype_lower_x, check_list_x, param_name="x")
    x["rel_pos_to_reduce"] = "before"

    dtype_axes = axes["dtype"]
    dtype_lower_axes = dtype_axes.lower()
    check_list_axes = ("int32", "int64")
    para_check.check_dtype(dtype_lower_axes, check_list_axes, param_name="axes")
    axes["rel_pos_to_reduce"] = "axis"

    schedules = []
    ins = classify([x, axes], OpPatternMode.REDUCE, {"keepdims": keepdims is True})
    tensors = []

    for (_x, _axes) in ins:
        with tbe.compute():
            shape_x, shape_axes = shape_util.variable_shape([_x, _axes], op_mode="reduce")
            data_input_x = tvm.placeholder(shape_x, name="data_input_x",
                                           dtype=dtype_lower_x)
            data_input_axes = tvm.placeholder(shape_axes, name="data_input_axes",
                                              dtype=dtype_lower_axes)
            axes_d = shape_util.axis_check(len(shape_x), _axes.get("value"))
            res = reduce_all_compute(data_input_x, axes_d, y, keepdims)
            tensors.append([data_input_x, data_input_axes, res])

        with tvm.target.cce():
            schedule = tbe.auto_schedule(res)
        schedules.append(schedule)

    # build
    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
