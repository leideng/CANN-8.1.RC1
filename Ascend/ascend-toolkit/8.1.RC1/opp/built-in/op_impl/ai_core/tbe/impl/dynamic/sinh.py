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
dynamic sinh
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


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant.
    """
    SCALER_NEGATIVE_ONE = -1
    SCALER_ZERO_POINT_FIVE = 0.5
    SCALAR_TWO = 2
    SCALAR_ZERO_0199 = 0.0001998459335617813754003
    SCALAR_ZERO_0833 = 0.00833308538698833
    SCALAR_ZERO_166 = 0.16666668254541
    SCALAR_ONE = 1.0
    SCALAR_NEGATIVE_15 = -1.5
    SCALAR_ZERO_48 = 0.48
    SCALAR_ZERO = 0.0


def sinh_compute_v1(input_data):
    dtype = input_data.dtype
    input_type = input_data.dtype
    has_improve_precision = False
    if dtype.lower() == "float16":
        input_data = tbe.cast_to(input_data, "float32")
        dtype = input_data.dtype.lower()
        has_improve_precision = True

    data = tbe.vabs(input_data)
    data_square = tbe.vmul(data, data)
    data_s = tbe.vmuls(data_square, tvm.const(Constant.SCALAR_ZERO_0199, dtype))
    data_s = tbe.vadds(data_s, tvm.const(Constant.SCALAR_ZERO_0833, dtype))
    data_s = tbe.vmul(data_s, data_square)
    data_s = tbe.vadds(data_s, tvm.const(Constant.SCALAR_ZERO_166, dtype))
    data_s = tbe.vmul(data_s, data_square)
    data_s = tbe.vadds(data_s, tvm.const(Constant.SCALAR_ONE, dtype))
    data_s = tbe.vmul(data_s, data)

    data_e1 = tbe.vmuls(data, tvm.const(Constant.SCALER_ZERO_POINT_FIVE, dtype))
    data_e1 = tbe.vexp(data_e1)

    data_e2 = tbe.vmuls(data, tvm.const(Constant.SCALAR_NEGATIVE_15, dtype))
    data_e2 = tbe.vexp(data_e2)
    data_s1 = tbe.vsub(data_e1, data_e2)
    data_s1 = tbe.vmuls(data_s1, tvm.const(Constant.SCALER_ZERO_POINT_FIVE, dtype))
    data_s1 = tbe.vmul(data_s1, data_e1)

    mask1 = tbe.vcmp(data, tvm.const(Constant.SCALAR_ZERO_48, dtype), 'lt', 'bit')
    data_s = tbe.vsel(mask1, data_s, data_s1)

    neg_data_s = tbe.vmuls(data_s, tvm.const(Constant.SCALER_NEGATIVE_ONE, dtype))
    mask2 = tbe.vcmp(input_data, tvm.const(Constant.SCALAR_ZERO, dtype), 'ge', 'bit')
    res = tbe.vsel(mask2, data_s, neg_data_s)
    if has_improve_precision:
        res = tbe.cast_to(res, input_type)
    return res


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals
@register_operator_compute("Sinh", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def sinh_compute(input_data, output_data, kernel_name="sinh"):
    """
    algorithm: sinh
    calculating data's sinh = exp(x - ln(2)) - exp(-x - ln(2))

    Parameters
    ----------
    input_data: TVM tensor
        data of input.
    output_data: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "sinh"

    Returns
    -------
    res: TVM tensor
        the res of sinh
    """
    if util_soc_common.after_v200():
        return sinh_compute_v1(input_data)

    dtype = input_data.dtype
    dtype_copy = input_data.dtype
    shape = input_data.shape

    # in order to get the precise calcuate result
    has_improve_precision = False
    if dtype.lower() == "float16" and tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        input_data = tbe.cast_to(input_data, "float32")
        dtype = "float32"
        has_improve_precision = True

    if dtype.lower() == "float32" and not tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        input_data = tbe.cast_to(input_data, "float16")
        dtype = "float16"
        has_improve_precision = True

    tensor_two = tbe.broadcast(tvm.const(Constant.SCALAR_TWO, dtype), shape)
    data_ln2 = tbe.vlog(tensor_two)
    data_neg_ln2 = tbe.vmuls(data_ln2, tvm.const(Constant.SCALER_NEGATIVE_ONE, dtype))

    data_mul = tbe.vmuls(input_data, tvm.const(Constant.SCALER_NEGATIVE_ONE, dtype))
    data_tmp = tbe.vadd(data_mul, data_neg_ln2)
    data_exp = tbe.vexp(data_tmp)

    data_x = tbe.vadd(input_data, data_neg_ln2)
    data_exp_data = tbe.vexp(data_x)

    res = tbe.vsub(data_exp_data, data_exp)

    # cast the dtype to float16
    if has_improve_precision:
        res = tbe.cast_to(res, dtype_copy)

    return res


@register_operator("Sinh")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def sinh(input_data, output_data, kernel_name="sinh"):
    """
    algorithm: sinh
    calculating data's sinh = (exp(x) - exp(-x)) / 2

    Parameters
    ----------
    input_data: dict
        shape and dtype of input, only support float16, float32, bfloat16
    output_data: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "sinh"

    Returns
    -------
    None
    """
    check_list = ("bfloat16", "float16", "float32")
    dtype_input = input_data.get("dtype")
    input_dtype = dtype_input.lower()
    para_check.check_dtype(input_dtype, check_list, param_name="input_data")

    schedules, tensors = [], []
    ins = classify([input_data], OpPatternMode.ELEWISE)
    for (_input_data,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_input_data])
            data_input = tvm.placeholder(x_shape[0], dtype=input_dtype,
                                         name="data_input")
            res = sinh_compute(data_input, output_data, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors,
              "bool_storage_as_1bit": False}
    tbe.build(schedules, config)
