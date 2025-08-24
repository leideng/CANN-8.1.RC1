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
cosh
"""
import functools

from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    Constant
    """
    # define a scaler, value is -1
    SCALER_NEGATIVE_ONE = -1
    # define a scaler, value is 0.5
    SCALER_ZERO_POINT_FIVE = 0.5
    # define a scaler, value is 2
    SCALAR_TWO = 2


# 'pylint: disable=locally-disabled,unused-argument
@register_operator_compute("cosh", op_mode="static", support_fusion=True)
def cosh_compute(input_x, output_cosh, kernel_name="cosh"):
    """
    algorithm: cosh
    calculating data's cosh, y = (e^(x)+e^(-x))/2

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_cosh: TVM tensor
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "cosh"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    dtype = input_x.dtype
    shape = input_x.shape
    has_improve_precision = False
    if dtype != "float32" and \
            tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        input_x = tbe.cast_to(input_x, "float32")
        dtype = "float32"
        has_improve_precision = True

    data_mul = tbe.vmuls(input_x, tvm.const(Constant.SCALER_NEGATIVE_ONE, dtype))
    data_exp = tbe.vexp(data_mul)
    data_exp_x = tbe.vmuls(data_exp, tvm.const(Constant.SCALER_ZERO_POINT_FIVE, dtype))

    tensor_two = tbe.broadcast(tvm.const(Constant.SCALAR_TWO, dtype), shape)
    data_ln2 = tbe.vlog(tensor_two)
    data_neg_ln2 = tbe.vmuls(data_ln2, tvm.const(Constant.SCALER_NEGATIVE_ONE, dtype))
    data_x = tbe.vadd(input_x, data_neg_ln2)
    data_exp_data = tbe.vexp(data_x)

    res = tbe.vadd(data_exp_x, data_exp_data)

    if has_improve_precision:
        res = tbe.cast_to(res, "float16")

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def cosh(input_x, output_cosh, kernel_name="cosh"):
    """
    algorithm: cosh
    calculating data's cosh, y = (e^(2x)+e^(-x))/2

    Parameters
    ----------
    input_x: dict
        shape and dtype of input, only support float16, float32
    output_cosh: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "cosh"

    Returns
    -------
    None
    """
    shape = input_x.get("shape")
    dtype = input_x.get("dtype")
    para_check.check_shape(shape, param_name="input_x")
    check_list = ("float16", "float32")
    input_dtype = dtype.lower()
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")
    reshape_input = (functools.reduce(lambda x, y: x * y, shape[:]),)
    data_input = tvm.placeholder(reshape_input,
                                 name="data_input", dtype=input_dtype)
    res = cosh_compute(data_input, output_cosh, kernel_name)

    with tvm.target.cce():
        sch = auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input, res]}
    build(sch, config)
