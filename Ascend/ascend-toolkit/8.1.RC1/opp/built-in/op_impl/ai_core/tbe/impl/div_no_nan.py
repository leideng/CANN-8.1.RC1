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
div_no_nan
"""
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals
@register_operator_compute("div_no_nan", op_mode="static", support_fusion=True)
def div_no_nan_compute(input_x, input_y, output_z, kernel_name="div_no_nan"):
    """
    div_no_nan_compute
    Returns 0 if the denominator is zero, else, like Div.
    ----------
    input_x: TVM tensor
        the placeholder of input tensor x
    input_y: TVM tensor
        the placeholder of input tensor y
    output_z: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        cce kernel name

    Returns
    -------
    res: TVM tensor
        the result of div_no_nan_compute
    """
    dtype = input_x.dtype
    shape_x = shape_util.shape_to_list(input_x.shape)
    shape_y = shape_util.shape_to_list(input_y.shape)
    shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x, shape_y,
                                                              param_name_input1="input_x",
                                                              param_name_input2="input_y")

    int_list = ("int32", "int8", "uint8")
    if dtype in int_list:
        input_x = tbe.cast_to(input_x, "float32")
        input_y = tbe.cast_to(input_y, "float32")

    if dtype in ("float16",):
        help_min = tvm.const(2**(-24), "float16")
        help_rec_one = tvm.const(2**12, "float16")
        help_rec_sec = tvm.const(2**12, "float16")
        neg_one = tvm.const(-1, "float16")
    else:
        help_min = tvm.const(2**(-126), "float32")
        help_rec_one = tvm.const(2**38, "float32")
        help_rec_sec = tvm.const(2**44, "float32")
        neg_one = tvm.const(-1, "float32")

    y_cmp = tbe.vabs(input_y)
    if tbe_platform.api_check_support("tbe.dsl.vmins", "float32"):
        y_index_help_1 = tbe.vmins(y_cmp, help_min)
    else:
        cmp_help = tbe.broadcast(help_min, shape_y)
        y_index_help_1 = tbe.vmin(y_cmp, cmp_help)
    y_index_help_2 = tbe.vmuls(y_index_help_1, help_rec_one)
    y_index = tbe.vmuls(y_index_help_2, help_rec_sec)
    if dtype not in ("float16",):
        y_index = tbe.vmuls(y_index, help_rec_sec)

    data_x_broadcast = tbe.broadcast(input_x, shape_max)
    data_y_broadcast = tbe.broadcast(input_y, shape_max)
    index_y_broadcast = tbe.broadcast(y_index, shape_max)
    neg_index = tbe.vadds(index_y_broadcast, neg_one)
    data_y_broadcast = tbe.vadd(data_y_broadcast, neg_index)
    res_vdiv = tbe.vdiv(data_x_broadcast, data_y_broadcast)
    res = tbe.vmul(res_vdiv, index_y_broadcast)

    if dtype in int_list:
        res = tbe.floor(res)
        res = tbe.cast_to(res, dtype)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def div_no_nan(input_x, input_y, output_z, kernel_name="div_no_nan"):
    """
    algorithm: div_no_nan_cce
    Returns 0 if the denominator is zero, else, like Div.
    Supports broadcasting.

    Parameters
    ----------
    input_x: dict
        dict with keys(shape and dtype) of input_x
    input_y: dict
        dict with keys(shape and dtype) of input_y
    output_z: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        cce kernel name, default value is "div_no_nan"

    Returns
    -------
    None
    """
    shape_x = input_x.get("shape")
    shape_y = input_y.get("shape")
    dtype = input_x.get("dtype")

    for shape in (shape_x, shape_y):
        para_check.check_shape(shape, param_name="input_x")
    shape_x, shape_y, _ = shape_util.broadcast_shapes(shape_x, shape_y,
                                                      param_name_input1="input_x",
                                                      param_name_input2="input_y")
    input_dtype = dtype.lower()
    para_check.check_dtype(input_dtype, ("float16", "float32",
                              "int32", "int8", "uint8"), param_name="input_x")
    reshape_x, reshape_y = shape_util.refine_shapes_for_broadcast(shape_x, shape_y)
    data_x = tvm.placeholder(reshape_x, name="data_x", dtype=input_dtype)
    data_y = tvm.placeholder(reshape_y, name="data_y", dtype=input_dtype)

    res = div_no_nan_compute(data_x, data_y, output_z, kernel_name)
    with tvm.target.cce():
        sch = auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_x, data_y, res]}
    build(sch, config)
