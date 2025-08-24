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
rsqrt
"""

from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpTbeImplMode


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # const value
    CONST_ONE = 1.0


@register_operator_compute("Rsqrt", op_mode="dynamic", support_fusion=True, support_bfp16=True)
# 'pylint: disable=unused-argument,too-many-locals,invalid-name
def rsqrt_compute(x, y, kernel_name="rsqrt_cce"):
    """
    Algrithm : rsqrt(x) = 1 / sqrt(x)  where x > 0

    Parameters
    ----------
    x: the placeholder of data input

    y : the dict of output

    kernel_name : cce kernel name

    Returns
    -------
    res : result of rsqrt
    """

    inp_dtype = x.dtype

    if inp_dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.vadd", "float32"):
        x = tbe.cast_to(x, "float32")

    data_res = _compute(x)

    if inp_dtype == "float16":
        data_res = tbe.cast_to(data_res, "float16")

    return data_res


def _compute(data_input):
    """
    Algrithm: rsqrt(x) = 1 / sqrt(x)

    Parameters
    ----------
    data_input: the placeholder of data input

    Returns
    -------
    data_res :  return of rsqrt
    """

    inp_shape = data_input.shape
    data_sqrt = tbe.vsqrt(data_input, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)
    tesor_one = tbe.broadcast(tvm.const(Constant.CONST_ONE, data_input.dtype), inp_shape)
    result = tbe.vdiv(tesor_one, data_sqrt)

    return result


# 'pylint: disable=unused-argument,too-many-locals,invalid-name
@register_operator("Rsqrt")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def rsqrt(x, y, kernel_name="rsqrt_cce"):
    """
    Algrithm: rsqrt(x) = 1 / sqrt(x)  where x > 0

    Parameters
    ----------
    x: dict
       dict of input, include shape and dtype,
       support float16, bfloat16, float32

    y: dict
       shape and dtype of output, should be same shape and type as input

    kernel_name: str
        cce kernel name, default value is "rsqrt_cce".

    Returns
    -------
    None
    """

    dtype = x.get("dtype").lower()
    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(dtype, check_list, param_name="x")

    ins = classify([x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []

    for (_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_x])[0]

            input_data = tvm.placeholder(x_shape, name="input_data",
                                         dtype=dtype)
            res = rsqrt_compute(input_data, y, kernel_name)

            tensors.append([input_data, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
