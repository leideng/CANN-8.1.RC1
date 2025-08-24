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
square_sum_v1
"""
from impl.dynamic.square_sum_v1 import get_new_format_axis
from impl.dynamic.square_sum_v1 import get_op_support_info as dynamic_get_op_support_info
from impl.dynamic.square_sum_v1 import op_select_format as dynamic_op_select_format
from impl.util.platform_adapter import OpImplMode
from impl.util.util_common import check_op_impl_mode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm


# 'pylint: disable = unused-argument
def get_op_support_info(input_x, output1, axis, keep_dims=True, kernel_name="square_sum_v1"):
    """
    get_op_support_info
    """

    return dynamic_get_op_support_info(input_x, output1, axis, keep_dims, kernel_name="square_sum_v1")


def op_select_format(input_x, output1, axis, keep_dims, kernel_name="square_sum_v1"):
    """
    select format dynamically
    op_select_format support desc:
    1. input_format always support 'ND'
    2. when ori_format is 'HWCN', input_format support 'FRACTAL_Z' or 'FRACTAL_NZ' in compile_static process
        for example:
            ori:
                input_x              shape = [5,5,16,16]           format = 'HWCN'
                output1              shape = []                    format = 'ND'
            format transformer:
                input_x              shape = [25,1,16,16]          format = 'FRACTAL_Z'
                output1              shape = []                    format = 'ND'
            ---------------------------------------------------------------------------
            ori:
                input_x              shape = [16,16]               format = 'ND'
                output1              shape = []                    format = 'ND'
            format transformer:
                input_x              shape = [1,1,16,16]          format = 'FRACTAL_NZ'
                output1              shape = []                    format = 'ND'

    """

    return dynamic_op_select_format(input_x, output1, axis, keep_dims, kernel_name="square_sum_v1")


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
# 'pylint: disable=locally-disabled,redefined-builtin,too-many-locals,unused-variable
@register_operator_compute("square_sum_v1", op_mode="static", support_fusion=True)
def square_sum_v1_compute(input_x, output1, axis, keep_dims, kernel_name="square_sum_v1", impl_mode=None):
    """
    calculating data

    Parameters
    ----------
    Input and output of fusion graph

    Returns
    -------
    output tensor
    """
    ori_dtype = input_x.dtype

    if impl_mode == OpImplMode.HIGH_PRECISION:
        input_x = tbe.cast_to(input_x, "float32")
    square_res = tbe.vmul(input_x, input_x)

    if square_res.dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.reduce_sum", "float32"):
        square_res = tbe.cast_to(square_res, "float32")

    sum_res = tbe.sum(square_res, axis=axis, keepdims=keep_dims)

    res = tbe.cast_to(sum_res, ori_dtype)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def square_sum_v1(input_x, output1, axis, keep_dims=True, kernel_name="square_sum_v1",
                  impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """
    calculating data

    Parameters
    ----------
    Input and output of fusion graph

    Returns
    -------
    None
    """
    check_op_impl_mode(impl_mode, [OpImplMode.HIGH_PERFORMANCE, OpImplMode.HIGH_PRECISION], kernel_name)

    shape = input_x.get("shape")
    dtype = input_x.get("dtype")
    ori_shape = input_x.get("ori_shape")
    x_format = input_x.get("format")
    x_ori_format = input_x.get("ori_format")
    input_dtype = dtype.lower()

    axis_d = []

    if not axis:
        for i, _ in enumerate(shape):
            axis_d.append(i)
    elif x_format in ["FRACTAL_NZ", "FRACTAL_Z"]:
        axis_d = get_new_format_axis(ori_shape, axis, x_format, x_ori_format)
    else:
        axis_d = axis

    para_check.check_shape(shape, param_name="input_x")

    data_input = tvm.placeholder(shape, name="data_input", dtype=input_dtype)

    res = square_sum_v1_compute(data_input, output1, axis_d, keep_dims, kernel_name, impl_mode)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_input, res]}

    tbe.cce_build_code(sch, config)
