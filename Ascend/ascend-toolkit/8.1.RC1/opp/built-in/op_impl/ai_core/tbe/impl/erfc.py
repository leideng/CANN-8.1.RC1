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
erfc
"""
import functools

from tbe.dsl.api import auto_schedule
from tbe.dsl.api import build
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    SCALER_ONE = 1
    SCALER_P = 0.47047
    SCALER_NEGATIVE_ONE = -1
    SCALER_B = -0.0958798
    SCALER_A = 0.3480242
    SCALER_FP16_MIN = 2**(-15)
    SCALER_C = 0.7478556
    SCALER_FP16_MAX = 32768


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals
@register_operator_compute("erfc", op_mode="static", support_fusion=True)
def erfc_compute(input_x, output_y, kernel_name="erfc"):
    """
    compute erfc

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        he dict of output_data, include keys(shape and dtype)
    kernel_name: str
        kernel name, default value is "erfc"

    Returns
    -------
    erfc_result: TVM tensor
        the =result of compute
    """

    dtype = input_x.dtype
    shape = shape_util.shape_to_list(input_x.shape)

    const_one = tvm.const(Constant.SCALER_ONE, dtype="float32")
    const_negative_one = tvm.const(Constant.SCALER_NEGATIVE_ONE, dtype="float32")
    const_p = tvm.const(Constant.SCALER_P, dtype="float32")
    const_a = tvm.const(Constant.SCALER_A, dtype="float32")
    const_b = tvm.const(Constant.SCALER_B, dtype="float32")
    const_c = tvm.const(Constant.SCALER_C, dtype="float32")
    fp16_max = tvm.const(Constant.SCALER_FP16_MAX, dtype=dtype)
    fp16_min = tvm.const(Constant.SCALER_FP16_MIN, dtype=dtype)

    if dtype == "float16":
        input_x = tbe.cast_to(input_x, "float32")

    data_sign_vmuls = tbe.vmuls(input_x, fp16_max)
    data_sign_abs = tbe.vabs(data_sign_vmuls)
    data_vadds = tbe.vadds(data_sign_abs, fp16_min)
    data_sign_div = tbe.vdiv(data_sign_vmuls, data_vadds)
    data_round = tbe.round(data_sign_div)
    tensor_sign = tbe.cast_to(data_round, dtype)

    tensor_one = tbe.broadcast(const_one, shape, "float32")
    tensor_abs = tbe.vabs(input_x)
    erfc_t_vmuls = tbe.vmuls(tensor_abs, const_p)
    erfc_t_vadds = tbe.vadds(erfc_t_vmuls, const_one)
    erfc_data_t = tbe.vdiv(tensor_one, erfc_t_vadds)

    erfc_abs_square = tbe.vmul(tensor_abs, tensor_abs)
    erfc_data_vmuls = tbe.vmuls(erfc_abs_square, const_negative_one)
    erfc_data_exp = tbe.vexp(erfc_data_vmuls)

    erfc_data_t_square = tbe.vmul(erfc_data_t, erfc_data_t)
    erfc_data_t_cube = tbe.vmul(erfc_data_t, erfc_data_t_square)

    erfc_t_vmuls = tbe.vmuls(erfc_data_t, const_a)
    erfc_t_square_vmuls = tbe.vmuls(erfc_data_t_square, const_b)
    erfc_t_cube_vmuls = tbe.vmuls(erfc_data_t_cube, const_c)

    erfc_square_vadd = tbe.vadd(erfc_t_vmuls, erfc_t_square_vmuls)
    erfc_cube_vadd_ = tbe.vadd(erfc_square_vadd, erfc_t_cube_vmuls)
    erfc_cube_vmuls = tbe.vmuls(erfc_cube_vadd_, const_negative_one)
    erfc_exp_vmul = tbe.vmul(erfc_cube_vmuls, erfc_data_exp)
    erfc_exp_vadds = tbe.vadds(erfc_exp_vmul, const_one)
    erfc_sign_vmul = tbe.vmul(tensor_sign, erfc_exp_vadds)
    erfc_sign_vmuls = tbe.vmuls(erfc_sign_vmul, const_negative_one)
    erfc_result = tbe.vadds(erfc_sign_vmuls, const_one)

    if dtype == "float16":
        erfc_result = tbe.cast_to(erfc_result, dtype)
    return erfc_result


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def erfc(input_x, output_y, kernel_name="erfc"):
    """
    algorithm: erfc
    Computes the Gauss error function of `x` element-wise

    Parameters
    ----------
    input_x: dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "erfc"

    Returns
    -------
    None
    """
    shape_input = input_x.get("shape")
    dtype_input = input_x.get("dtype")

    para_check.check_shape(shape_input, param_name="input_x")

    dtype_input = dtype_input.lower()
    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_input, check_list, param_name="input_x")

    shape_input = shape_util.shape_refine(shape_input)
    reshape_input = (functools.reduce(lambda x, y: x * y, shape_input[:]),)
    data_input = tvm.placeholder(reshape_input, name="data_input",
                                 dtype=dtype_input)

    erfc_result = erfc_compute(data_input, output_y, kernel_name)

    with tvm.target.cce():
        sch = auto_schedule(erfc_result)

    config = {"name": kernel_name,
              "tensor_list": [data_input, erfc_result]}

    build(sch, config)
