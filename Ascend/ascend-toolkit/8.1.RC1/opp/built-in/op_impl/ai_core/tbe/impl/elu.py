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
elu
  Op_description :
    do element-wise elu operation.

    # elu(
    #   x,
    #   y,
    #   kernel_name='cce_elu')

  Supportive_dtype_format :
    ["float16", "float32"]
    ['ND', 'NCHW', 'NHWC', 'NC1HWC0']

  Constraint :
    [1] All : shape size limit is 2147483648

"""
import functools

from tbe.dsl.api import auto_schedule
from tbe.dsl.api import build
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import error_manager_vector


def _elu_computer_performance(data, alpha, scale, input_scale, dtype):
    """
    computer performance
    """
    scalar_one_neg = tvm.const(-1.0, dtype)

    negative_data = tbe.vmuls(data, scalar_one_neg)
    negative_data = tbe.vrelu(negative_data)
    negative_data = tbe.vmuls(negative_data, tvm.const(-input_scale, dtype))
    positive_data = tbe.vrelu(data)
    exp_res = tbe.vexp(negative_data)
    exp_res = tbe.vadds(exp_res, scalar_one_neg)

    res = tbe.vmuls(exp_res, tvm.const(alpha, dtype))
    res = tbe.vadd(positive_data, res)
    res = tbe.vmuls(res, tvm.const(scale, dtype))

    return res


def _elu_computer_precision(data, alpha, scale, input_scale, dtype):
    """
    computer precision
    """
    scalar_zero = tvm.const(0.0, dtype)
    negative_data = tbe.vmins(data, scalar_zero)
    negative_data = tbe.vmuls(negative_data, tvm.const(input_scale, dtype))
    positive_data = tbe.vmaxs(data, scalar_zero)

    exp_res = tbe.vexp(negative_data)
    exp_res = tbe.vadds(exp_res, tvm.const(-1.0, dtype))

    res = tbe.vaxpy(exp_res, positive_data, tvm.const(alpha, dtype))
    res = tbe.vmuls(res, tvm.const(scale, dtype))

    return res


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
@register_operator_compute("elu", op_mode="static", support_fusion=True)
def elu_compute(x, y, alpha, scale, input_scale, kernel_name="elu"):
    """
    do element-wise elu compute
    f(x) = max(min(scale*alpha(e^(input_scale*x) - 1), 0), scale*x),  in cloud scene, for all inputs
    f(x) = max(min(scale*alpha(e^(input_scale*x) - 1), 0), scale*x),  in mini scene, for x <= TAYLOR_THRESHOLD or x >= 0
    f(x) = fifth taylor computer,    in mini scene, for TAYLOR_THRESHOLD < x < 0

    Parameters:
    ----------
    x: the placeholder of data input

    alpha: float, coefficient when input tensor is less than zero
    scale: float, coefficient of input data
    input_scale: float, coefficient when input tensor is less than zero

    y: the dict of output

    kernel_name : cce kernel name, default value is "elu"

    Returns : A Tensor. Has the same type as data_input.
    -------
    """

    data = x
    dtype = data.dtype

    has_improve_precision = False
    if tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        has_improve_precision = True
    if dtype.lower() == "float16" and has_improve_precision:
        data = tbe.cast_to(data, "float32")
        cvt_dtype = "float32"
    else:
        cvt_dtype = dtype

    if has_improve_precision:
        res = _elu_computer_precision(data, alpha, scale, input_scale, cvt_dtype)
    else:
        res = _elu_computer_performance(data, alpha, scale, input_scale, cvt_dtype)

    if dtype.lower() == "float16" and has_improve_precision:
        res = tbe.cast_to(res, "float16")

    return res


# 'pylint: disable=invalid-name
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def elu(x, y, alpha=1.0, scale = 1.0, input_scale = 1.0, kernel_name="elu"):
    """
    do element-wise elu operation

    Parameters:
    ----------
    x: the dict of input, only support float16, float32

    alpha: float, coefficient when input tensor is less than zero
    scale: float, coefficient of input data
    input_scale: float, coefficient when input tensor is less than zero

    output_res : the dict of output

    kernel_name : cce kernel name, default value is "elu"

    Returns
    -------
    None
    """

    shape_input = x.get("shape")
    dtype_input = x.get("dtype")
    input_dtype = dtype_input.lower()

    para_check.check_shape(shape_input, param_name="x")

    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_input, check_list, param_name="x")

    if not tbe_platform.api_check_support("tbe.dsl.reduce_sum", "float32") and dtype_input == "float32":
        error_manager_vector.raise_err_input_dtype_not_supported("elu", "x", "float16", dtype_input)
    fuseshape = [1]
    fuseshape[0] = functools.reduce(lambda x, y: x*y, shape_input)
    data_input = tvm.placeholder(fuseshape, name="data_input", dtype=input_dtype)

    res = elu_compute(data_input, y, alpha, scale, input_scale, kernel_name)

    with tvm.target.cce():
        auto_sch = auto_schedule(res)

    config = {"name": kernel_name,
              "print_ir": False,
              "tensor_list": [data_input, res],
              "bool_storage_as_1bit": False}
    build(auto_sch, config)
