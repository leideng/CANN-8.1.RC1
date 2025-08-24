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
relu6
`f(x) = min(max(0,x), 6)`
"""
import functools

from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tvm
import te.lang.cce as tbe
from impl.util.platform_adapter import build
from impl.util.platform_adapter import auto_schedule
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check


# 'pylint: disable=unused-argument
@register_operator_compute("relu6", op_mode="static", support_fusion=True)
def relu6_compute(input_x, output_y, kernel_name="relu6"):
    """
    compute of relu6

    Parameters
    ----------
    input_data: TVM tensor
        the placeholder of first input data
    output_y: dict
        shape and dtype of output,should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is "relu6"

    Returns
    -------
    compute result of relu6
    """
    tmp_res = tbe.vmaxs(input_x, tvm.const(0, input_x.dtype))
    final_res = tbe.vmins(tmp_res, tvm.const(6, input_x.dtype))

    return final_res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def relu6(input_x, output_y, kernel_name="relu6"):
    """
       f(x)= 6(x >= 6)
       f(x)= 0(x <= 0)
       f(x)= x(0<x<6)

    Parameters
    ----------
    input_x : dict
        shape and dtype of input_x
    output_y : dict
        shape and dtype of output_y, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is "relu6"

    Returns
    ------
    None
    """
    input_shape = shape_util.scalar2tensor_one(input_x.get("shape"))
    input_dtype = input_x.get("dtype").lower()
    para_check.check_shape(input_shape, param_name="input_x")

    vmaxs_support = tbe_platform.api_check_support("tbe.dsl.vmaxs", "float32")
    if not vmaxs_support:
        para_check.check_dtype(input_dtype, ("int32", "float16"), param_name="input_x")

    # check input tensor data_type
    check_list = ("int32", "float16", "float32")
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")

    input_shape = [functools.reduce(lambda x, y: x * y, input_shape[:])]
    input_data = tvm.placeholder(input_shape, name="input_data", dtype=input_dtype)
    final_res = relu6_compute(input_data, output_y, kernel_name="relu6")

    with tvm.target.cce():
        auto_sch = auto_schedule(final_res)

    config = {"name": kernel_name, "tensor_list": (input_data, final_res)}
    build(auto_sch, config)
