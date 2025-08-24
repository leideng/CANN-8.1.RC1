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
dynamic cosh
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util import util_soc_common


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # `define a scaler , value = -1`
    SCALER_NEGATIVE_ONE = -1


def cosh_compute_v1(input_x, dtype, dtype_):
    # `define a scaler , value = -1.5`
    scaler_negative_one_point_five = -1.5
    # `define a scaler , value = 0.5`
    scaler_zero_point_five = 0.5

    has_improve_precision = False
    if dtype != "float32" and tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        input_x = tbe.cast_to(input_x, "float32")
        dtype = "float32"
        has_improve_precision = True

    if dtype == "float32" and not tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        input_x = tbe.cast_to(input_x, "float16")
        dtype = "float16"
        has_improve_precision = True
    x = tbe.vabs(input_x)
    x1 = tbe.vmuls(x, tvm.const(scaler_zero_point_five, dtype))
    x2 = tbe.vmuls(x, tvm.const(scaler_negative_one_point_five, dtype))
    e1 = tbe.vexp(x1)
    e2 = tbe.vexp(x2)
    x3 = tbe.vadd(e1, e2)
    x4 = tbe.vmuls(x3, tvm.const(scaler_zero_point_five, dtype))
    res = tbe.vmul(x4, e1)

    if has_improve_precision:
        res = tbe.cast_to(res, dtype_)

    return res


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals,invalid-name
@register_operator_compute("Cosh", op_mode="dynamic", support_fusion=True, support_bfp16=True)
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
    dtype_ = input_x.dtype
    if util_soc_common.after_v200():
        return cosh_compute_v1(input_x, dtype, dtype_)
    # `define a scaler , value = 0.5`
    scaler_zero_point_five = 0.5
    # `define a scaler , value = 2`
    scalar_two = 2
    shape = input_x.shape
    has_improve_precision = False
    if dtype != "float32" and tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        input_x = tbe.cast_to(input_x, "float32")
        dtype = "float32"
        has_improve_precision = True

    if dtype == "float32" and not tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        input_x = tbe.cast_to(input_x, "float16")
        dtype = "float16"
        has_improve_precision = True
    
    data_mul = tbe.vmuls(input_x, tvm.const(Constant.SCALER_NEGATIVE_ONE, dtype))
    data_exp = tbe.vexp(data_mul)
    data_exp_x = tbe.vmuls(data_exp, tvm.const(scaler_zero_point_five, dtype))

    tensor_two = tbe.broadcast(tvm.const(scalar_two, dtype), shape)
    data_ln2 = tbe.vlog(tensor_two)
    data_neg_ln2 = tbe.vmuls(data_ln2, tvm.const(Constant.SCALER_NEGATIVE_ONE, dtype))
    data_x = tbe.vadd(input_x, data_neg_ln2)
    data_exp_data = tbe.vexp(data_x)

    res = tbe.vadd(data_exp_x, data_exp_data)

    if has_improve_precision:
        res = tbe.cast_to(res, dtype_)

    return res


@register_operator("Cosh")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def cosh(input_x, output_cosh, kernel_name="cosh"):
    """
    algorithm: cosh
    calculating data's cosh, y = (e^(2x)+e^(-x))/2

    Parameters
    ----------
    input_x: dict
        shape and dtype of input, only support bfloat16, float16, float32
    output_cosh: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "cosh"

    Returns
    --------
    None
    """
    input_dtype = input_x.get("dtype").lower()
    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")

    schedules, tensors = [], []
    ins = classify([input_x], OpPatternMode.ELEWISE)
    for (_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_x])
            data_input = tvm.placeholder(x_shape[0], dtype=input_dtype,
                                         name="data_input")
            res = cosh_compute(data_input, output_cosh, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors,
              "bool_storage_as_1bit": False}
    tbe.build(schedules, config)
