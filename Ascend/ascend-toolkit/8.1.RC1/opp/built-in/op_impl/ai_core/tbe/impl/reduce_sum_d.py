# Copyright 2019 Huawei Technologies Co., Ltd
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
reduce_sum_d
"""
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm

# define the type of None
NONETYPE = type(None)


# 'pylint: disable=locally-disabled,unused-argument,invalid-name
# 'pylint: disable=too-many-arguments
@register_operator_compute("reduce_sum_d", op_mode="static", support_fusion=True)
def reduce_sum_d_compute(x, y, axis, keepdims, kernel_name="reduce_sum_d", is_5hdc=False, is_nz_nd=False):
    """redusce_sum_d compute

    Parameters:
    ----------
    x: TVM tensor
        input tensor.
    y: dict
        the dict of output tensor.
    axis: int, list, tuple or NONETYPE
        the axis for reduce.
    keepdims: bool or NONETYPE
        if true, retains reduced dimensions with length 1.
    kernel_name: str
        cce kernel name, default value is "reduce_sum_d".
    is_5hdc: bool
    is_nz_nd: bool
    Returns
    -------
    res: TVM tensor
        output tensor, has the same shape and type as input tensor.
    """
    shape = shape_util.shape_to_list(x.shape)
    shape_len = len(shape)
    if not axis:
        axis = range(shape_len)
    if hasattr(axis, 'index'):
        axis = list(axis)

    dtype = x.dtype
    cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")

    if cce_product not in ("Ascend310",) and dtype == "float16" and \
            tbe_platform.api_check_support("tbe.dsl.reduce_sum", "float32") and \
            not (is_5hdc or is_nz_nd):
        x = tbe.cast_to(x, "float32")
    res_sum = tbe.sum(x, axis=axis, keepdims=keepdims)
    res = tbe.cast_to(res_sum, dtype)

    return res


# 'pylint: disable=locally-disabled,too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def reduce_sum_d(x, y, axis, keep_dims=None, kernel_name="reduce_sum_d"):
    """reduce a tensor on a certain axis based on sum.

    Parameters:
    ----------
    x: dict
        the dict of input tensor.
    y: dict
        the dict of output tensor.
    axis: int, list, tuple or NONETYPE
        the axis for reduce.
    keep_dims: bool or NONETYPE
        if true, retains reduced dimensions with length 1.
    kernel_name: str
        cce kernel name, default value is "reduce_sum_d".

    Returns
    -------
    None
    """
    shape = x.get("shape")
    dtype = x.get("dtype")
    format_x = x.get("format")
    format_y = y.get("format")
    format_ori_y = y.get("ori_format")
    dtype_lower = dtype.lower()
    check_list = ("float16", "float32")

    para_check.check_shape(shape, param_name="x")
    para_check.check_dtype(dtype_lower, check_list, param_name="x")

    axis_d = []
    shape_len = len(shape)
    if not axis:
        for i, _ in enumerate(shape):
            axis_d.append(i)
    else:
        axis_d = list(axis)
    axis_d = shape_util.axis_check(shape_len, axis_d)
    # 5HD Special param for 5hd schedule
    is_nz_nd = False
    if format_x == "FRACTAL_NZ" and format_ori_y == format_y:
        is_nz_nd = True
    is_5hdc = para_check.check_and_init_5hdc_reduce_support(x, axis)

    if not keep_dims and not is_5hdc:
        shape, axis_d = shape_util.shape_refine(list(shape), axis_d, keep_dims)
        shape, axis_d = shape_util.simplify_axis_shape(shape, axis_d)

    data_input = tvm.placeholder(shape, name="data_input_" + kernel_name, dtype=dtype_lower)
    res = reduce_sum_d_compute(data_input, y, axis_d, keep_dims, is_5hdc=is_5hdc, is_nz_nd=is_nz_nd)
    if is_5hdc:
        res.ori_shape = x["ori_shape"]
        res.ori_format = x["ori_format"]

    with tvm.target.cce():
        sch = auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_input, res]}
    build(sch, config)
