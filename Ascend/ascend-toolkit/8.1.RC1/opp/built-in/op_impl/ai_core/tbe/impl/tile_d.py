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
tile_d
"""
from impl.dynamic.tile_d import get_op_support_info as static_get_op_support_info
from impl.dynamic.tile_d import op_select_format as static_op_select_format
from impl.dynamic.tile_d import update_mutiples_with_format
from impl.util import util_common
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import shape_util
import tbe.dsl as tbe_dsl
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm


# 'pylint: disable = unused-argument
def get_op_support_info(input_x, output_x, multiples, kernel_name="tile_d"):
    """
    get_op_support_info
    """
    return static_get_op_support_info(input_x, output_x, multiples, kernel_name)


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,too-many-locals
# 'pylint: disable=locally-disabled,too-many-branches,too-many-statements
def op_select_format(input_x, output_x, multiples, kernel_name="tile_d"):
    """TileD: to do boradcast with multiples

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_x: dict
        dict of output.
    multiples : list or tuple.
        Number of the axis replicates.
    kernel_name : str
        kernel name, default value is "tile_d".

    Returns
    -------
    param_dynamic_in_json
    """
    return static_op_select_format(input_x, output_x, multiples, kernel_name="tile_d")


@register_operator_compute("tile_d", op_mode="static", support_fusion=True)
def tile_d_compute(data, output_x, multiples, kernel_name="tile_d"):
    """TVM calculation process, used for fusion operation.

    Parameters
    ----------
    data: list of placeholders.
        Input data.
    output_x: dict.
        dict of output.
    multiples : list or tuple.
        Number of the axis replicates.
    kernel_name : str.
        Cce kernel name, default value is "tile_d".

    Returns
    -------
    res
    """
    src_dtype = data.dtype.lower()
    shape = shape_util.shape_to_list(data.shape)
    out_shape = []
    for shape_i, multiples_i in zip(shape, multiples):
        out_shape_i = shape_i*multiples_i
        out_shape.append(out_shape_i)
    if src_dtype == "int8" or src_dtype == "uint8":
        data = tbe.cast_to(data, "float16")
    elif src_dtype == "bfloat16" and not tbe_platform.api_check_support("tbe.dsl.vadd", "bfloat16") and \
            tbe_platform.intrinsic_check_support("Intrinsic_vconv", "bf162f32"):
        data = tbe.cast_to(data, "float32")
    res = tbe.broadcast(data, out_shape)
    if src_dtype == "int8" or src_dtype == "uint8":
        res = tbe.cast_to(res, src_dtype)
    elif src_dtype == "bfloat16" and res.dtype != src_dtype and res.dtype.lower() == "float32":
        res = tbe_dsl.round(res, src_dtype)

    return res


# 'pylint: disable=too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_LIST_INT,
                            para_check.KERNEL_NAME)
def tile_d(input_x, output_x, multiples, kernel_name="tile_d"):
    """algorithm: tile.
    The tile in tensorflow can multiple the shape of the given tensor.
    For example, tiling [a b c d] by [2] produces [a b c d a b c d].
    The tile op in TBE is different from tf.tile, tile of TBE use broadcast
    api, and only support that at least an axis in shape is 1.The '1' axis
    is to be multipled.
    For example, if shape = [51, 1] and multiples = [1, 77], after computation,
    the output shape will be [51, 77].
    Abnormal condition:
    1. The length of shape must be equal to or less than the shape of multiples.
    2. The type of kernel_name is not string.
    3. The shape is neither list nor tuple.
    4. The dtype is not float32, float16, or int32.
    5. All of the axises of the multiples is 1.

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_x: dict
        dict of output.
    multiples : list or tuple.
        Number of the axis replicates.
    kernel_name : str.
        kernel name, default value is "tile_d".

    Returns
    -------
    None
    """
    input_x = util_common.update_shape_base_other_format(input_x)
    shape = input_x.get("shape")
    dtype = input_x.get("dtype").lower()
    para_check.check_shape(shape, param_name="input_x")
    para_check.check_shape(multiples, param_name="multiples")
    para_check.check_dtype(dtype.lower(), ("bfloat16", "float16", "float32", "int32", "int8", "bool", "uint8"),
                           param_name="input_x")
    if dtype == "bool":
        dtype = "int8"
    shape = list(shape)
    multiples = list(multiples)
    input_format = input_x.get("format")
    output_format = output_x.get("format")
    if input_format in ("NCHW", "NHWC") and output_format in ("NC1HWC0",):
        # branch: 4D tile to 5HD ((N, 1, 1, 1) to (N, C1, H, W, C0)) and output C is 16 align
        # change input shape from (N, 1, 1, 1) to (N, 1, 1, 1, 1)
        shape = shape + [1]
        if input_format == "NCHW":
            # change multiples from (1, C, H, W) to (1, C1, H, W, C0)
            multiples = [multiples[0], multiples[1] // 16, multiples[2], multiples[3], 16]
        else:
            # change multiples from (1, H, W, C) to (1, C1, H, W, C0)
            multiples = [multiples[0], multiples[3] // 16, multiples[1], multiples[2], 16]
    elif input_format in ("FRACTAL_Z", "FRACTAL_Z_3D", "NDC1HWC0", "NC1HWC0", "FRACTAL_NZ"):
        input_ori_format = input_x.get("ori_format")
        multiples = update_mutiples_with_format(input_format, input_ori_format, multiples)

    if len(shape) > len(multiples):
        multiples = [1] * (len(shape) - len(multiples)) + list(multiples)

    if len(shape) < len(multiples):
        len_error = len(multiples) - len(shape)
        shape = [1]*len_error + shape

    out_shape = []
    for shape_i, multiples_i in zip(shape, multiples):
        out_shape_i = shape_i*multiples_i
        out_shape.append(out_shape_i)
    para_check.check_shape(out_shape, param_name="output_x")

    shape_adapt = []
    multiples_adapt = []
    for i, shape_i in enumerate(shape):
        multiples_i = multiples[i]
        if multiples_i != 1 and shape_i != 1:
            shape_adapt.append(1)
            multiples_adapt.append(multiples_i)
            multiples_i = 1
        shape_adapt.append(shape_i)
        multiples_adapt.append(multiples_i)

    shape = shape_adapt
    multiples = multiples_adapt

    for shape_i, multiples_i in zip(shape, multiples):
        if not (shape_i == 1 or multiples_i == 1):
            error_detail = "In tile of TBE, any axis of either shape or multiples have to be 1"
            error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "input_x", \
                                                               "multiples", error_detail)

    axis_not_multiple = 0
    for multiples_i in multiples:
        if multiples_i == 1:
            axis_not_multiple += 1
    if axis_not_multiple == len(multiples):
        error_detail = "In tile of TBE, the axis of multiples can't all be 1"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "multiples", error_detail)

    data = tvm.placeholder(shape, name="data", dtype=dtype.lower())

    res = tile_d_compute(data, output_x, multiples, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data, res]}

    tbe.build(sch, config)
