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
reciprocal
"""
import functools

from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util import util_select_op_base
from impl.util import util_common

SHAPE_SIZE_LIMIT = 2147483648  # shape limit


# 'pylint: disable=redefined-builtin,unused-argument
# 'pylint: disable=too-many-locals,too-many-branches
def op_select_format(input_x, output_y, kernel_name="reciprocal"):
    """
    Get support format according to input_x
    """
    shape = input_x.get("ori_shape")
    shape_len = len(shape)
    format = input_x.get("ori_format")
    format_4d_list = ["NCHW", "NHWC", "HWCN"]
    format_5d_list = ["NDHWC", "DHWCN", "NCDHW"]
    bfp16_support = tbe_platform.intrinsic_check_support("Intrinsic_vconv", "bf162f32")
    if not bfp16_support:
        dtype_list = ["float", "float16"]
        support_format = "ND,ND,NCHW,NCHW,NHWC,NHWC,HWCN,HWCN"
    else:
        dtype_list = ["bfloat16", "float", "float16"]
        support_format = "ND,ND,ND,NCHW,NCHW,NCHW,NHWC,NHWC,NHWC,HWCN,HWCN,HWCN"
    support_format = support_format.split(',')

    if (len(shape) == 5 and format in format_5d_list):
        c_dim = shape[format.index("C")]
        n_dim = shape[format.index("N")]

    # NDC1HWC0 FRACTAL_Z_3D
    if len(shape) == 5 and format in ['NDHWC']:
        if c_dim % 16 == 0:
            support_format.append("NDC1HWC0")
    if len(shape) == 5 and format in format_5d_list:
        if c_dim % 16 == 0 and n_dim % 16 == 0:
            support_format.append("FRACTAL_Z_3D")

    # whether support format NC1HWC0 FRACTAL_Z C1HWNCoC0
    if shape_len == 4 and format in format_4d_list:
        if format == "NCHW":
            n_dim = shape[0]
            c_dim = shape[1]
        if format == "NHWC":
            n_dim = shape[0]
            c_dim = shape[3]
        if format == "HWCN":
            n_dim = shape[3]
            c_dim = shape[2]
        # whether support format NC1HWC0
        if c_dim % 16 == 0:
            support_format.append("NC1HWC0")
        # whether support format FRACTAL_Z and C1HWNCoC0
        if n_dim % 16 == 0 and c_dim % 16 == 0:
            support_format.append("C1HWNCoC0")
        if util_common.is_support_fractal_z_input(input_x) and n_dim % 16 == 0 and \
                c_dim % 16 == 0:
            support_format.append("FRACTAL_Z")

    dtype_total = []
    for dtype in dtype_list:
        dtype_total = dtype_total + [dtype] * len(support_format)
    format_list = support_format * len(dtype_list)
    format_list_input = format_list
    format_list_output = format_list
    input0 = util_select_op_base.gen_param(classify="input0",
                                           name="x",
                                           datatype=",".join(dtype_total),
                                           format=",".join(format_list_input))
    output0 = util_select_op_base.gen_param(classify="output0",
                                            name="y",
                                            datatype=",".join(dtype_total),
                                            format=",".join(format_list_output))

    param_list = [input0, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


@register_operator_compute("reciprocal", op_mode="static", support_fusion=True)
def reciprocal_compute(input_x, output_y, kernel_name="reciprocal"):
    """
    reciprocal_compute
    """
    if tbe_platform.api_check_support("tbe.dsl.vdiv", "float32"):
        dtype = input_x.dtype
        shape = shape_util.shape_to_list(input_x.shape)
        if dtype == "float16":
            input_x = tbe.cast_to(input_x, "float32")
        data_one = tbe.broadcast(tvm.const(1, "float32"), shape)
        res = tbe.vdiv(data_one, input_x)
        if dtype == "float16":
            res = tbe.cast_to(res, "float16")
    else:
        res = tbe.vrec(input_x)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def reciprocal(input_x, output_y, kernel_name="reciprocal"):
    """
    algorithm: reciprocal

    calculating data's reciprocal,y= 1 / x

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is reciprocal

    Returns
    -------
    None
    """
    shape = shape_util.scalar2tensor_one(input_x.get("shape"))
    para_check.check_shape(shape, param_name="input_x")

    check_list = ["float16", "float32"]
    inp_dtype = input_x.get("dtype").lower()
    para_check.check_dtype(inp_dtype, check_list, param_name="input_x")

    shape = shape_util.shape_refine(shape)
    fuseshape = [1]
    fuseshape[0] = functools.reduce(lambda x, y: x * y, shape)
    data = tvm.placeholder(fuseshape, name="data", dtype=inp_dtype)

    res = reciprocal_compute(data, output_y, kernel_name)

    with tvm.target.cce():
        sch = auto_schedule(res)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": [data, res]}

    build(sch, config)
