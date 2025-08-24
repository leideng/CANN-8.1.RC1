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
cos
"""
import functools

import te.lang.cce as tbe
import te.platform as tbe_platform
from tbe import tvm
from te.utils import para_check
from te.utils import shape_util
from impl.dynamic.cos import op_select_format as cos_op_select_format


# 'pylint: disable=unused-argument
def op_select_format(input_x, output_y, kernel_name="cos"):
    """
    select format dynamically\n
    op_select_format support desc:
      1. support ND format at any scenarios
      2. support NC1HWC0/NDC1HWC0 format when C is aligned(16)
          > for example:
          > x : Tensor of (ori_shape=(15, 15, 15, 16), ori_fromat="NHWC") support NC1HWC0
      3. support FRACTAL_Z_3D/FRACTAL_Z format when both C and N are aligned(16)
          > for example:
          > x : Tensor of (ori_shape=(32, 15, 15, 16), ori_fromat="NHWC") support FRACTAL_Z
      4. support FRACTAL_N format when both the last dim and the renultimate dim are aligned(16)
          > for example:
          > x : Tensor of (ori_shape=(13, 13, 32, 16), ori_fromat="NHWC") support FRACTAL_N
    """
    return cos_op_select_format(input_x, output_y, kernel_name)


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    Constant
    """
    # 2pi, the cycle of cosin
    TWO_PI = 2*3.14159265358979


# 'pylint: disable=locally-disabled, unused-argument
@tbe_platform.fusion_manager.fusion_manager.register("cos")
def cos_compute(input_x, output_y, kernel_name="cos"):
    """
    algorithm: cos
    calculating data's cos x = 1 - x^2/2! + x^4/4! + ... + (-1)^k*x^2k/(2k)!

    Parameters
    ----------
    input_x : TVM tensor
              data of input
    output_y: dict
              shape and dtype of output, should be same shape and type as input
    kernel_name: str
              kernel name, default value is "cos"

    Returns
    -------
    res : TVM tensor
          the result of cos
    """

    dtype = input_x.dtype
    shape = shape_util.shape_to_list(input_x.shape)

    # cast to type float32 when type is float16
    has_improve_precision = False
    if dtype.lower() == "float16" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vmul", "float32"):
        input_x = tbe.cast_to(input_x, "float32")
        dtype = "float32"
        has_improve_precision = True

    # round the input
    round_fp16 = tbe.round(tbe.vmuls(input_x, 1.0 / Constant.TWO_PI))
    round_fp32 = tbe.cast_to(round_fp16, dtype)
    input_x_round = tbe.vsub(input_x, tbe.vmuls(round_fp32, Constant.TWO_PI))

    # the initial value one
    const_res = tvm.const(1.0, dtype=dtype)
    res = tbe.broadcast(const_res, shape)
    # compute the rank 2
    input_x_power = tbe.vmul(input_x_round, input_x_round)
    iter_value = tbe.vmuls(input_x_power, -1.0/2.0)
    res = tbe.vadd(res, iter_value)
    # compute the rank 4~14
    iter_list = (4, 6, 8, 10, 12, 14)
    for i in iter_list:
        iter_value = tbe.vmuls(tbe.vmul(input_x_power, iter_value), -1.0/(i*(i-1)))
        res = tbe.vadd(res, iter_value)

    # cast the dtype to float16
    if has_improve_precision:
        res = tbe.cast_to(res, "float16")

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def cos(input_x, output_y, kernel_name="cos"):
    """
    algorithm: cos
    calculating data's cos x = 1 - x^2/2! + x^4/4! + ... + (-1)^k*x^2k/(2k)!

    Parameters
    ----------
    input_x : dict
              shape and dtype of input, only support float16, float32
    output_y: dict
              shape and dtype of output, should be same shape and type as input
    kernel_name : str
              kernel name, default value is "cos"

    Returns
    -------
    None
    """
    shape_input = input_x.get("shape")
    dtype_input = input_x.get("dtype").lower()

    para_check.check_shape(shape_input, param_name="input_x")
    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_input, check_list, param_name="input_x")

    reshape_input = (functools.reduce(lambda x, y: x * y, shape_input[:]),)
    data_input = tvm.placeholder(reshape_input,
                                 name="data_input", dtype=dtype_input)
    res = cos_compute(data_input, output_y, kernel_name)
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input, res]}
    tbe.cce_build_code(sch, config)
