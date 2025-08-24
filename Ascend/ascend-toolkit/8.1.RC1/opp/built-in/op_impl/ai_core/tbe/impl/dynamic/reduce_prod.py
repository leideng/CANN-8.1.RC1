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
dynamic reduce prod
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_context
from impl.util.util_soc_common import after_v200
from impl.util import util_common


# 'pylint: disable=unused-argument,invalid-name
@register_operator_compute("ReduceProd", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def reduce_prod_compute(x, axes, y, keepdims=None, kernel_name="reduce_prod"):
    """
    reduce_prod compute

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
        cce kernel name, default value is "reduce_prod".

    Returns
    -------
    res: TVM tensor
        output tensor, has the same type as input tensor.
    """
    dtype = x.dtype
    if after_v200() and dtype in ("int8", "uint8"):
        x = tbe.cast_to(x, "int32")
        res_prod = tbe.reduce_prod(x, axis=axes, keepdims=keepdims)
        res = util_common.reduce_int_cast_to_b8(res_prod, dtype)
        return res

    if dtype in ("int8", "uint8"):
        x = tbe.cast_to(x, "float16")
    res_prod = tbe.reduce_prod(x, axis=axes, keepdims=keepdims)
    res = tbe.cast_to(res_prod, dtype)

    return res


# 'pylint: disable=too-many-locals,invalid-name
@register_operator("ReduceProd")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def reduce_prod(x, axes, y, keep_dims=False, kernel_name="reduce_prod"):
    """reduce a tensor on a certain axes based on prod.

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
    kernel_name: str
        cce kernel name, default value is "reduce_prod".

    Returns
    -------
    None
    """
    keepdims = False if keep_dims is None else keep_dims
    dtype_x = x.get("dtype")
    dtype_lower_x = dtype_x.lower()
    check_list_x = ("bfloat16", "float16", "float32", "int8", "uint8", "int64")
    para_check.check_dtype(dtype_lower_x, check_list_x, param_name="x")
    x["rel_pos_to_reduce"] = "before"

    dtype_axes = axes.get("dtype")
    dtype_lower_axes = dtype_axes.lower()
    check_list_axes = ("int32", "int64")
    para_check.check_dtype(dtype_lower_axes, check_list_axes, param_name="axes")
    axes["rel_pos_to_reduce"] = "axis"

    tbe_context.get_context().add_compile_info("axes_idx", 1)
    if "const_value" in axes.keys():
        axes["value"] = list(axes["const_value"])

    schedules = []
    ins = classify([x, axes], OpPatternMode.REDUCE, {"keepdims": keepdims is True})
    tensors = []

    for (_x, _axes) in ins:
        with tbe.compute():
            shape_x, shape_axes = shape_util.variable_shape([_x, _axes], op_mode="reduce")
            data_input_x = tvm.placeholder(shape_x, name="data_input_x", dtype=dtype_lower_x)
            data_input_axes = tvm.placeholder(shape_axes, name="data_input_axes", dtype=dtype_lower_axes)
            axes_d = shape_util.axis_check(len(shape_x), _axes.get("value"))
            res = reduce_prod_compute(data_input_x, axes_d, y, keepdims, kernel_name)
            tensors.append([data_input_x, data_input_axes, res])

        with tvm.target.cce():
            schedule = tbe.auto_schedule(res)
        schedules.append(schedule)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
