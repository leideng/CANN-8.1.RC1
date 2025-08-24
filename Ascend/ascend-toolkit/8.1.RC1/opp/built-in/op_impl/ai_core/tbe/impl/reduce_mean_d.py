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
reduce_mean_d
"""
# 'pylint: disable=too-many-arguments,too-many-locals,global-statement
import collections
import te.lang.cce as tbe
import te.platform as tbe_platform
from te.utils import para_check
from te.utils import shape_util
from tbe import tvm
from impl.dynamic.reduce_mean_d import is_vir_type
from impl.util import util_soc_common


SHAPE_SIZE_LIMIT = 100000000  # shape limit for tf_reduce_mean

NoneType = type(None)
ori_shape = [[0], [0]]
ori_format = ["NCHW", "NCHW"]


def check_supported(input_x,
                    output_y,
                    axes,
                    keep_dims=None,
                    noop_with_empty_axes=False,
                    kernel_name="reduce_mean_d",
                    impl_mode=None):
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

    dtype = input_x.get("dtype")
    if dtype not in ("float16", "float32"):
        return False, "dtype is not support"
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


def get_cof(axes, shape):
    """
    get cof
    """
    reduce_elts = 1.0
    if isinstance(axes, collections.abc.Iterable):
        for i in axes:
            reduce_elts *= shape[i]
    else:
        reduce_elts = shape[axes]
    cof = reduce_elts**(-1)

    if ori_format[0] == 'NHWC' and ori_format[1] == 'NC1HWC0' and len(axes) == 2 \
            and axes == [1, 4] and len(ori_shape[0]) == 4:
        cof = ori_shape[0][-1]**(-1)
    return cof


def preprocess_input(impl_mode, dtype, data_input_tmp, is_5hdc, is_nz_nd):
    """
    preprocess input and get has_improve_precision
    """
    has_improve_precision = False
    cce_product = tbe_platform.cce_conf.get_soc_spec("SHORT_SOC_VERSION")
    if impl_mode is None:
        if cce_product in ("Ascend310",):
            impl_mode = "high_performance"
        else:
            impl_mode = "high_precision"

    if cce_product not in ("Ascend310",) and dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support(
                "te.lang.cce.sum", "float32") and not (is_5hdc or is_nz_nd) and impl_mode == "high_precision":
        data_input_tmp = tbe.cast_to(data_input_tmp, "float32")
        has_improve_precision = True
    elif cce_product in ("Ascend310",) and dtype == "float16" \
            and tbe_platform.cce_conf.api_check_support("te.lang.cce.sum",
                                                        "float32") \
            and not (is_5hdc or is_nz_nd) and impl_mode != "high_performance":
        data_input_tmp = tbe.cast_to(data_input_tmp, "float32")
        has_improve_precision = True
    return has_improve_precision, data_input_tmp


# 'pylint: disable=locally-disabled,unused-argument,invalid-name
@tbe_platform.fusion_manager.fusion_manager.register("reduce_mean_d")
def reduce_mean_d_compute(x,
                          y,
                          axes,
                          keepdims,
                          noop_with_empty_axes,
                          kernel_name="reduce_mean_d",
                          impl_mode=None,
                          is_5hdc=False,
                          is_nz_nd=False):
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
    is_nz_nd: bool
    is_5hdc: bool
    Returns
    -------
    res: TVM tensor
        output tensor, has the same shape and type as input tensor.
    """
    shape = shape_util.shape_to_list(x.shape)
    shape_len = len(shape)
    axes = get_axes(axes, shape_len, noop_with_empty_axes)

    cof = get_cof(axes, shape)

    dtype = x.dtype
    data_input_tmp = x

    has_improve_precision = False
    has_improve_precision, data_input_tmp = preprocess_input(impl_mode, dtype, data_input_tmp, is_5hdc, is_nz_nd)
    data_input_tmp = tbe.vmuls(data_input_tmp, cof)
    res = tbe.sum(data_input_tmp, axis=axes, keepdims=keepdims)

    if has_improve_precision:
        res = tbe.cast_to(res, dtype)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            (para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_LIST_INT),
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def reduce_mean_d(input_x,
                  output_y,
                  axes,
                  keep_dims=None,
                  noop_with_empty_axes=False,
                  kernel_name="reduce_mean_d",
                  impl_mode=None):
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
    noop_with_empty_axes : bool, default is True
        if true, no op, just like tf,
        if false, reduce all dims when x's shape is [].
    kernel_name : str
        cce kernel name, default value is reduce_mean_d

    Returns
    -------
    None
    """
    global ori_shape
    global ori_format
    shape = input_x.get("shape")
    format_x = input_x.get("format")
    format_y = output_y.get("format")
    format_ori_y = output_y.get("ori_format")
    if isinstance(axes, int):
        axes = [axes]

    para_check.check_shape(shape, param_name="input_x")
    check_list = ["float16", "float32"]
    shape_len = len(shape)

    if axes is None or (len(axes) == 0 and not noop_with_empty_axes) :
        axes = range(shape_len)

    if hasattr(axes, 'index'):
        axes = list(axes)

    inp_dtype = input_x.get("dtype").lower()
    para_check.check_dtype(inp_dtype, check_list, param_name="input_x")

    axes = shape_util.axis_check(shape_len, axes)

    # Shape should not be modified in 5HD mode
    # 5HD Special param for 5hd schedule
    is_nz_nd = False
    if format_x == "FRACTAL_NZ" and format_y == format_ori_y:
        is_nz_nd = True
    is_5hdc = para_check.check_and_init_5hdc_reduce_support(input_x, axes)
    if not is_5hdc:
        shape, axes = shape_util.shape_refine(list(shape), axes)
        if len(axes) != 0:
            shape, axes = shape_util.simplify_axis_shape(shape, axes)
    ori_shape = [input_x["ori_shape"], input_x["shape"]]
    ori_format = [input_x["ori_format"], input_x["format"]]
    data_input = tvm.placeholder(shape, name="data_input", dtype=inp_dtype)
    res = reduce_mean_d_compute(data_input,
                                output_y,
                                axes,
                                keep_dims,
                                noop_with_empty_axes,
                                impl_mode=impl_mode,
                                is_5hdc=is_5hdc,
                                is_nz_nd=is_nz_nd)
    if is_5hdc:
        res.ori_shape = input_x["ori_shape"]
        res.ori_format = input_x["ori_format"]

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    config = {"print_ir": False, "name": kernel_name, "tensor_list": [data_input, res]}
    tbe.cce_build_code(sch, config)
