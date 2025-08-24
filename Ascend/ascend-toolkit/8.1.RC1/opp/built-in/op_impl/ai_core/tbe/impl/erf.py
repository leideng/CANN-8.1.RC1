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
erf
"""
import functools

from impl.util.platform_adapter import tvm
import te.lang.cce as tbe
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    SCALER_ONE = 1
    SCALER_NEGATIVE_ONE = -1
    SCALER_P = 0.47047
    SCALER_A = 0.3480242
    SCALER_B = -0.0958798
    SCALER_C = 0.7478556
    SCALER_FP16_MAX = 32768
    SCALER_FP16_MIN = 2**(-15)


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals
@register_operator_compute("erf", op_mode="static", support_fusion=True)
def erf_compute(input_x, output_y, kernel_name="erf"):
    """
    compute erf

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        he dict of output_data, include keys(shape and dtype)
    kernel_name: str
        kernel name, default value is "erf"

    Returns
    -------
    erf_result: TVM tensor
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

    data_vmuls = tbe.vmuls(input_x, fp16_max)
    data_abs = tbe.vabs(data_vmuls)
    data_vadds = tbe.vadds(data_abs, fp16_min)
    data_div = tbe.vdiv(data_vmuls, data_vadds)
    data_round = tbe.round(data_div)
    tensor_sign = tbe.cast_to(data_round, dtype)

    tensor_one = tbe.broadcast(const_one, shape, "float32")
    tensor_abs = tbe.vabs(input_x)
    erf_t_vmuls = tbe.vmuls(tensor_abs, const_p)
    erf_t_vadds = tbe.vadds(erf_t_vmuls, const_one)
    erf_data_t = tbe.vdiv(tensor_one, erf_t_vadds)

    erf_abs_square = tbe.vmul(tensor_abs, tensor_abs)
    erf_data_vmuls = tbe.vmuls(erf_abs_square, const_negative_one)
    erf_data_exp = tbe.vexp(erf_data_vmuls)

    erf_data_t_square = tbe.vmul(erf_data_t, erf_data_t)
    erf_data_t_cube = tbe.vmul(erf_data_t, erf_data_t_square)

    erf_t_vmuls = tbe.vmuls(erf_data_t, const_a)
    erf_t_square_vmuls = tbe.vmuls(erf_data_t_square, const_b)
    erf_t_cube_vmuls = tbe.vmuls(erf_data_t_cube, const_c)

    erf_square_vadd = tbe.vadd(erf_t_vmuls, erf_t_square_vmuls)
    erf_cube_vadd_ = tbe.vadd(erf_square_vadd, erf_t_cube_vmuls)
    erf_cube_vmuls = tbe.vmuls(erf_cube_vadd_, const_negative_one)
    erf_exp_vmul = tbe.vmul(erf_cube_vmuls, erf_data_exp)
    erf_exp_vadds = tbe.vadds(erf_exp_vmul, const_one)
    erf_result = tbe.vmul(tensor_sign, erf_exp_vadds)

    if dtype == "float16":
        erf_result = tbe.cast_to(erf_result, dtype)
    return erf_result


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def erf(input_x, output_y, kernel_name="erf"):
    """
    algorithm: erf
    Computes the Gauss error function of `x` element-wise

    Parameters
    ----------
    input_x: dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "erf"

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

    erf_result = erf_compute(data_input, output_y, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(erf_result)

    config = {"name": kernel_name,
              "tensor_list": [data_input, erf_result]}

    tbe.cce_build_code(sch, config)
