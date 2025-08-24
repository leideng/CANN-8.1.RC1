# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
reduce_nansum
"""
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


@register_operator_compute("ReduceNansum", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def reduce_nansum_compute(x, axes, y, keepdims, kernel_name="reduce_nansum"):
    """
    reduce_nansum compute

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
        cce kernel name, default value is "reduce_nansum".

    Returns
    -------
    res: TVM tensor
        output tensor, has the same type as input tensor.
    """
    xshape = x.shape
    dtype = x.dtype
    cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")

    if cce_product not in ("Ascend310", ) and dtype == "float16" and \
            tbe_platform.api_check_support("te.lang.cce.sum", "float32"):
        x = tbe.cast_to(x, "float32")
    xdtype = x.dtype
    zeros = tbe.broadcast(tvm.const(0.0, dtype=xdtype), xshape, xdtype)
    # NaN is not equal to NaN, so if the value x is not equal itself,then it is NaN. We can replace it with 0
    no_nan_x = tbe.vcmpsel(x, x, "eq", x, zeros)
    res = tbe.reduce_sum(no_nan_x, axis=axes, keepdims=keepdims)
    res = tbe.cast_to(res, dtype)
    return res


@register_operator("ReduceNansum")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def reduce_nansum(x, axes, y, keep_dims=False, kernel_name="reduce_nansum"):
    """
    reduce a tensor on a certain axes based on sum, treating Not a Numbers(NaNs) as zero.

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
        cce kernel name, default value is "reduce_nansum".

    Returns
    -------
    None
    """
    keepdims = False if keep_dims is None else keep_dims
    dtype_x = x.get("dtype")
    dtype_lower_x = dtype_x.lower()
    check_list_x = ("bfloat16", "float16", "float32")
    para_check.check_dtype(dtype_lower_x, check_list_x, param_name="x")
    x["rel_pos_to_reduce"] = "before"

    dtype_axes = axes.get("dtype")
    dtype_lower_axes = dtype_axes.lower()
    check_list_axes = ("int32", "int64")
    para_check.check_dtype(dtype_lower_axes, check_list_axes)
    axes["rel_pos_to_reduce"] = "axis"

    tbe_context.get_context().add_compile_info("axes_idx", 1)
    if "const_value" in axes.keys():
        axes["value"] = list(axes["const_value"])

    schedules = []
    tensors = []
    ins = classify([x, axes], OpPatternMode.REDUCE, {"keepdims": keepdims is True})

    for (x, axes) in ins:
        with tbe.compute():
            shape_x, shape_axes = shape_util.variable_shape([x, axes], op_mode="reduce")
            data_input_x = tvm.placeholder(shape_x, name="data_input_x", dtype=dtype_lower_x)
            data_input_axes = tvm.placeholder(shape_axes, name="data_input_axes", dtype=dtype_lower_axes)
            axes_d = shape_util.axis_check(len(shape_x), axes.get("value"))
            res = reduce_nansum_compute(data_input_x, axes_d, y, keepdims, kernel_name)
            tensors.append([data_input_x, data_input_axes, res])

        with tvm.target.cce():
            schedule = tbe.auto_schedule(res)
        schedules.append(schedule)

    #build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)

