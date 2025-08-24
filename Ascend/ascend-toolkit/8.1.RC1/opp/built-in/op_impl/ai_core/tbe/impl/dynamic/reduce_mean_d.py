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
dynamic reduce_mean_d
"""
from tbe.common.context import op_context
from impl.dynamic.reduce_mean import get_calc_dtype
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import OpImplMode
from impl.util.util_compute import get_cof
from impl.util import util_soc_common
from impl.util import util_common


def is_vir_type():
    """
    get vir_type
    """
    context = op_context.get_context()
    if context is None:
        return False

    vir_type = context.get_addition("virtual_type")
    is_vir = vir_type == "1"

    return is_vir


# 'pylint: disable=too-many-branches,too-many-arguments,too-many-locals
# 'pylint: disable=unused-argument,invalid-name
def check_supported(input_x,
                    output_y,
                    axes=None,
                    keep_dims=None,
                    noop_with_empty_axes=False,
                    kernel_name="reduce_mean_d",
                    impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """
    Reduce True or False.

    Parameters:
    ----------
    input_x : dict
        shape and dtype of input
    output_y: dict
        shape and dtype of output
    axes : int, list, tuple, NoneType
        The dimensions to reduce. If None (the default), reduces all dimensions.
        Must be in the range [-rank(input_tensor), rank(input_tensor)).
    keep_dims : bool, NoneType
        if true, retains reduced dimensions with length 1,
        default value is None.
    noop_with_empty_axes : bool, NoneType
        useless attr to avoid dynamic reduce_mean_d compile error.
    kernel_name : str
        cce kernel name, default value is reduce_mean_d
    impl_mode: HIGH_PERFORMANCE or HIGH_PRECISION
    Returns
    -------
    None
    """
    if is_vir_type():
        return False, "Not supported in virtualizetion scebarrios"

    if util_soc_common.after_v200():
        return True, "supported after_v200"

    shape_x = input_x.get("shape")
    x_format = input_x.get("format")
    dtype = input_x.get("dtype")
    is_unknown_shape = util_common.is_unknown([input_x, output_y])
    bfp16_support = tbe_platform.intrinsic_check_support("Intrinsic_vconv", "bf162f32")
    if bfp16_support:
        if dtype not in ("bfloat16", "float16", "float32"):
            return False, "dtype is not support"
    else:
        if dtype not in ("float16", "float32"):
            return False, "dtype is not support"
    if axes is None:
        axes = range(len(shape_x))
    else:
        axes = list(axes)
    if not is_unknown_shape and x_format == "NC1HWC0" and axes == [1, 4]:
        return False, "when static 5hd not support"
    return True, ""


def get_axes(axes, shape_len, noop_with_empty_axes):
    """
    preprocess axes
    """
    if axes is None or (len(axes) == 0 and not noop_with_empty_axes):
        axes = range(shape_len)
    if hasattr(axes, 'index'):
        axes = list(axes)
    axes = shape_util.axis_check(shape_len, axes)
    return axes


def support_fusion_condition():
    """
    check ub fusion support
    """
    inputs = tbe_context.op_context.get_context().get_op_info()[0].inputs
    if tbe_context.get_context().get_op_mode() == "static" and inputs[0].get('format') != 'NC1HWC0':
        return True
    return False


@register_operator_compute("ReduceMeanD", op_mode="dynamic",
                            support_fusion=support_fusion_condition, support_bfp16=True)
def reduce_mean_d_compute(x,
                          y,
                          axes=None,
                          keepdims=None,
                          noop_with_empty_axes=False,
                          kernel_name="reduce_mean_d",
                          impl_mode=OpImplMode.HIGH_PERFORMANCE,
                          attrs=None):
    """reduce_mean_d compute

    Parameters:
    ----------
    x: TVM tensor
        input tensor.
    y: dict
        the dict of output tensor.
    axes: int, list, tuple or NoneType
        the axes for reduce.
    keepdims: bool or NoneType
        if true, retains reduced dimensions with length 1.
    kernel_name: str
        cce kernel name, default value is "reduce_mean_d".

    Returns
    -------
    res: TVM tensor
        output tensor, has the same shape and type as input tensor.
    """
    shape_x = shape_util.shape_to_list(x.shape)
    dtype_x = x.dtype
    shape_len = len(shape_x)
    axes = get_axes(axes, shape_len, noop_with_empty_axes)
    reduce_elts = get_cof(axes, shape_x)
    calc_dtype = get_calc_dtype(dtype_x, impl_mode)

    if dtype_x != calc_dtype:
        data_input_tmp = tbe.cast_to(x, calc_dtype)
    else:
        data_input_tmp = x

    if isinstance(reduce_elts, float):
        if reduce_elts == 0:
            res = tbe.reduce_mean(data_input_tmp, axis=axes, keepdims=keepdims)
            if dtype_x != calc_dtype:
                if dtype_x in ("int8", "uint8"):
                    res = tbe.cast_to(res, dtype_x, False)
                else:
                    res = tbe.cast_to(res, dtype_x)
            return res
        
        cof = reduce_elts**(-1)
        if attrs is not None and attrs.get("format") == "NC1HWC0":
            ori_format = attrs.get("ori_format")
            axis_new = attrs.get("new_axis")
            ori_shape = attrs.get("ori_shape")
            if ori_format == "NHWC" and list(axis_new) == [1, 4] and len(ori_shape) == 4:
                cof = ori_shape[-1]**(-1)
        cof = tvm.const(cof, dtype=calc_dtype)
    else:
        cof = tbe.var("cof", dtype=calc_dtype)
        if calc_dtype == "float16":
            tbe.var("cof_empty", dtype=calc_dtype)
        tbe_context.get_context().add_compile_info("reduce_mean_cof_dtype", calc_dtype)

    data_input_tmp = tbe.vmuls(data_input_tmp, cof)
    res = tbe.reduce_sum(data_input_tmp, axis=axes, keepdims=keepdims)

    if dtype_x != calc_dtype:
        if dtype_x in ("int8", "uint8"):
            res = tbe.cast_to(res, dtype_x, False)
        else:
            res = tbe.cast_to(res, dtype_x)

    return res


@register_operator("ReduceMeanD")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def reduce_mean_d(input_x,
                  output_y,
                  axes=None,
                  keep_dims=None,
                  noop_with_empty_axes=False,
                  kernel_name="reduce_mean_d",
                  impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """
    Reduce a tensor on a certa in axes based on mean.

    Parameters:
    ----------
    input_x : dict
        shape and dtype of input
    output_y: dict
        shape and dtype of output
    axes : int, list, tuple, NoneType
        The dimensions to reduce. If None (the default), reduces all dimensions.
        Must be in the range [-rank(input_tensor), rank(input_tensor)).
    keep_dims : bool, NoneType
        if true, retains reduced dimensions with length 1,
        default value is None.
    noop_with_empty_axes : bool, NoneType
        useless attr to avoid dynamic reduce_mean_d compile error.
    kernel_name : str
        cce kernel name, default value is reduce_mean_d

    Returns
    -------
    None
    """
    dtype = input_x.get("dtype")
    dtype_lower = dtype.lower()
    check_list = ("bfloat16", "float16", "float32", "int8", "uint8")
    para_check.check_dtype(dtype_lower, check_list)
    input_x["rel_pos_to_reduce"] = "before"
    shape = input_x.get("shape")
    ori_shape = input_x.get("ori_shape")
    cur_format = input_x.get("format")
    ori_format = input_x.get("ori_format")
    shape_len = len(shape)
    if axes is None or (len(axes) == 0 and not noop_with_empty_axes):
        axes = range(shape_len)
    if hasattr(axes, 'index'):
        axes = list(axes)
    axes = shape_util.axis_check(shape_len, axes)
    input_axis = {"shape": [len(axes), ], "value": axes, "rel_pos_to_reduce": "axis"}
    attr = {"shape": shape, "ori_shape": ori_shape, "format": cur_format, "ori_format": ori_format, "new_axis": axes}
    schedules = []
    tensors = []
    ins = classify([input_x, input_axis], OpPatternMode.REDUCE, {
        "keepdims": keep_dims is True,
        "ignore_fractal_format": False
    })
    for (_input_x, _axes) in ins:
        with tbe.compute():
            shape_var_new = shape_util.variable_shape([_input_x, _axes], op_mode="reduce")[0]
            data_input = tvm.placeholder(shape_var_new, name="data_input", dtype=dtype_lower)
            res = reduce_mean_d_compute(data_input,
                                        output_y,
                                        _axes.get("value"),
                                        keep_dims,
                                        noop_with_empty_axes=noop_with_empty_axes,
                                        kernel_name=kernel_name,
                                        impl_mode=impl_mode,
                                        attrs = attr)
            tensors.append([data_input, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
