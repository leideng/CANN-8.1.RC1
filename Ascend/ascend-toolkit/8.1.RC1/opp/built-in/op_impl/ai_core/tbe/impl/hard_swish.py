# Copyright 2021 Huawei Technologies Co., Ltd
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
hard_swish
`f(x) = min(max(0,x+3), 6) * x / 6`
"""
import functools
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import build
from impl.util.platform_adapter import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class Constants:
    """
    The class for constant.
    """
    CONST_THREE = 3
    CONST_SIX = 6
    CONST_ONE_IN_SIX = 1 / 6


# 'pylint: disable=unused-argument
@register_operator_compute("hard_swish", op_mode="static", support_fusion=True)
def hard_swish_compute(input_x, output_y, kernel_name="hard_swish"):
    """
    compute of hard_swish

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of first input data
    output_y: dict
        shape and dtype of output,should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is "hard_swish"

    Returns
    -------
    compute result of hard_swish
    """
    dtype = input_x.dtype
    if dtype == "float16":
        input_x_fp32 = tbe.cast_to(input_x, "float32")
        input_add3 = tbe.vadds(input_x_fp32, tvm.const(Constants.CONST_THREE, "float32"))
        max_res = tbe.vmaxs(input_add3, tvm.const(0, "float32"))
        relu6_res = tbe.vmins(max_res, tvm.const(Constants.CONST_SIX, "float32"))
        relu6_res_ov6 = tbe.vmuls(relu6_res, tvm.const(Constants.CONST_ONE_IN_SIX, "float32"))
        res_fp32 = tbe.vmul(input_x_fp32, relu6_res_ov6)
        return tbe.cast_to(res_fp32, "float16")
    else:
        input_add3 = tbe.vadds(input_x, tvm.const(Constants.CONST_THREE, "float32"))
        max_res = tbe.vmaxs(input_add3, tvm.const(0, "float32"))
        relu6_res = tbe.vmins(max_res, tvm.const(Constants.CONST_SIX, "float32"))
        relu6_res_ov6 = tbe.vmuls(relu6_res, tvm.const(Constants.CONST_ONE_IN_SIX, "float32"))
        return tbe.vmul(input_x, relu6_res_ov6)


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def hard_swish(input_x, output_y, kernel_name="hard_swish"):
    """
       f(x)= 0(x <= -3)
       f(x)= x(x >= 3)
       f(x)= x(x+3)/6(otherwise)
    Parameters
    ----------
    input_x : dict
        shape and dtype of input_x
    output_y : dict
        shape and dtype of output_y, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is "hard_swish"

    Returns
    ------
    None
    """
    input_shape = shape_util.scalar2tensor_one(input_x.get("shape"))
    input_dtype = input_x.get("dtype").lower()
    para_check.check_shape(input_shape, param_name="input_x")
    # check input tensor data_type
    check_list = ("float16", "float32")
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")

    input_shape = [functools.reduce(lambda x, y: x * y, input_shape[:])]
    input_data = tvm.placeholder(input_shape, name="input_data", dtype=input_dtype)
    final_res = hard_swish_compute(input_data, output_y, kernel_name="hard_swish")

    with tvm.target.cce():
        auto_sch = auto_schedule(final_res)

    config = {"name": kernel_name, "tensor_list": (input_data, final_res)}
    build(auto_sch, config)
