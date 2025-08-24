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
atanh
"""
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    the class for constant.
    """
    CONST_HALF = 0.5
    CONST_ONE = 1
    CONST_NEG_ONE = -1


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
@register_operator_compute("atanh", op_mode="static", support_fusion=True)
def atanh_compute(x, y, kernel_name="atanh"):
    """
    Algrithm : atanh(x) = 0.5 * log((1 + x) / (1 - x)) if abs(x) < 1

    Parameters
    ----------
    x: the placeholder of data input

    y : the dict of output

    kernel_name : cce kernel name

    Returns
    -------
    res : result of atanh
    """

    inp_dtype = x.dtype
    shape = x.shape

    if inp_dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.vadd", "float32"):
        x = tbe.cast_to(x, "float32")

    data_res = _compute(x, shape)

    if inp_dtype == "float16":
        data_res = tbe.cast_to(data_res, "float16")
    else:
        data_res = tbe.cast_to(data_res, "float32")

    return data_res


def _compute(data_input, shape):
    """
    Algrithm: atanh(x) = 0.5*log((1+x)/(1-x))

    Parameters
    ----------
    data_input: the placeholder of data input

    shape: the shape of data_input

    Returns
    -------
    data_res :  return of atanh
    """

    data_1_sum_x = tbe.vadds(data_input, tvm.const(Constant.CONST_ONE, data_input.dtype))
    data_sub_x = tbe.vmuls(data_input, tvm.const(Constant.CONST_NEG_ONE, data_input.dtype))
    data_1_sub_x = tbe.vadds(data_sub_x, tvm.const(Constant.CONST_ONE, data_input.dtype))
    data_x_mul = tbe.vdiv(data_1_sum_x, data_1_sub_x)
    data_x_log = tbe.vlog(data_x_mul, 1)
    data_res = tbe.vmuls(data_x_log, tvm.const(Constant.CONST_HALF, data_input.dtype))

    return data_res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def atanh(x, y, kernel_name="atanh"):
    """
    Algrithm: atanh(x) = atanh

    Parameters
    ----------
    Algorithm: atanh

    Parameters:

    x: the dict of input data, only support float16, float32.

    y: the dict of output

    kernel_name: cce kernel name, default value is "atanh".

    Returns
    -------
    None
    """

    shape = x.get("shape")
    dtype = x.get("dtype")

    para_check.check_shape(shape, param_name="x")
    shape, _ = shape_util.refine_shape_axes(shape, [])

    check_list = ("float16", "float32")
    para_check.check_dtype(dtype, check_list, param_name="x")

    dtype = dtype.lower()
    input_data = tvm.placeholder(shape, dtype, "input_data")

    with tvm.target.cce():
        res = atanh_compute(input_data, y, kernel_name)
        sch = auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [input_data, res],
              "print_ir": False,
              "bool_storage_as_1bit": False
             }

    build(sch, config)
