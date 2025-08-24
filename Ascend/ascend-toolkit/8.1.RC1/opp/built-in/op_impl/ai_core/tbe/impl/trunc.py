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
trunc
"""

import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from tbe.common.utils.shape_util import shape_to_list

CONST_ZERO = 0.0
SHAPE_SIZE_LIMIT = 2147483648


# 'pylint: disable=unused-argument,invalid-name
@register_operator_compute("trunc", op_mode="static", support_fusion=True)
def trunc_compute(input_x, output_y, kernel_name="trunc"):
    """
    calculating the value of x1 OR x2 element-wise

    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    output_y : TVM tensor
        the dict of output_y
    kernel_name : str
        kernel name, default value is "trunc"

    Returns
    -------
    result res
    """

    input_data_type = input_x.dtype
    tensor_zero_dtype = input_data_type
    shape_x1 = shape_to_list(input_x.shape)
    para_check.check_shape(shape_x1, 0, SHAPE_SIZE_LIMIT)
    shape = input_x.shape
    tensor_zero = tbe.broadcast(tvm.const(CONST_ZERO, tensor_zero_dtype), shape)
    data_res1 = tbe.vmax(input_x, tensor_zero)
    data_res2 = tbe.vmin(input_x, tensor_zero)
    data_res1 = tbe.floor(data_res1)
    data_res2 = tbe.ceil(data_res2)
    res = tbe.vadd(data_res1, data_res2)
    res = tbe.cast_to(res, input_data_type)
    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_STR)
def trunc(input_x, output_y, kernel_name="trunc"):
    """
    calculating the value of x1 OR x2 element-wise

    Parameters
    ----------
    input_x : the dict of input_x
         include shape and dtype.

    output_y : the dict of output_y
         include shape and dtype.

    kernel_name : str
        kernel name, default value is "trunc"

    Returns
    -------
    None
    """

    shape = input_x.get("shape")
    dtype = input_x.get("dtype")
    input_dtype = dtype.lower()

    #Check whether the dtype of the input parameter is supported.
    check_tuple = ("float16", "float32", "int8", "int32", "uint8")
    para_check.check_dtype_rule(input_dtype, check_tuple)

    para_check.check_shape_rule(shape)
    para_check.check_shape(shape)
    para_check.check_kernel_name(kernel_name)

    data_input = tvm.placeholder(shape, name="data_input", dtype=input_dtype)
    res = trunc_compute(data_input, output_y, kernel_name)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input, res]}

    tbe.cce_build_code(schedule, config)
