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
rsqrt_grad
"""
from impl.util import util_common
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    SCALAR = -0.5


# 'pylint: disable=locally-disabled,unused-argument
@register_operator_compute("RsqrtGrad", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def rsqrt_grad_compute(input_y, input_dy, output_z, kernel_name="rsqrt_grad"):
    """
    compute for rsqrt_grad

    Parameters
    ----------
    input_y: TVM tensor
        the placeholder of input_y
    input_dy: TVM tensor
        the placeholder of input_dy
    output_z: dict
        dict info of output_z
    kernel_name: str
        cce kernel name, default value is "rsqrt_grad"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    dtype_input_y = input_y.dtype
    rsqrt_const = tvm.const(Constant.SCALAR, dtype=dtype_input_y)
    if dtype_input_y in ("int8", "float16"):
        rsqrt_const = tvm.const(Constant.SCALAR, dtype="float32")
        input_y = tbe.cast_to(input_y, "float32")
        input_dy = tbe.cast_to(input_dy, "float32")
    res_vmul = tbe.vmul(input_y, input_y)
    res_vmul1 = tbe.vmul(res_vmul, input_y)
    res_vmul2 = tbe.vmul(res_vmul1, input_dy)
    res = tbe.vmuls(res_vmul2, rsqrt_const)
    if dtype_input_y in ("int32", "float16"):
        res = tbe.cast_to(res, dtype_input_y, f1628IntegerFlag=True)

    if dtype_input_y in ("int8",):
        res_int32 = tbe.cast_to(res, "int32")
        const_ff = tvm.const(255, "int32")
        const_ff = tbe.broadcast(const_ff, res.shape)
        data_and = tbe.vand(res_int32, const_ff)
        res_fp32_mask = tbe.cast_to(data_and, "float32")
        res = util_common.uint8_int8_overflow_proc(res_fp32_mask, "int8")

    return res


# 'pylint: disable=too-many-locals
@register_operator("RsqrtGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def rsqrt_grad(input_y, input_dy, output_z, kernel_name="rsqrt_grad"):
    """
    calculate the backpropagation of rsqrt operation
    rsqrt: y = 1 / sqrt (x)
    rsqrt_grad: -1/2 * y**3 *dy

    Parameters
    ----------
    input_y: dict
        dict of input_y, include keys(shape and dtype)
    input_dy: dict
        dict of input_dy, include keys(shape and dtype)
    output_z: dict
        dict of  output
    kernel_name: str
        cce kernel name, default value is "rsqrt_grad"

    Returns
    -------
    None
    """
    dtype_input_y = input_y.get("dtype")
    dtype_input_dy = input_dy.get("dtype")
    check_list = ("bfloat16", "float16", "float32", "int32", "int8")
    dtype_input_y = dtype_input_y.lower()
    para_check.check_dtype(dtype_input_y, check_list, param_name="input_y")
    dtype_input_dy = dtype_input_dy.lower()
    para_check.check_dtype(dtype_input_dy, check_list, param_name="input_dy")
    schedules, tensors = [], []
    ins = classify([input_y, input_dy], OpPatternMode.ELEWISE)
    for (data1, data2) in ins:
        with tbe.compute():
            shape_y, shape_dy = shape_util.variable_shape([data1, data2])
            data_input_y = tvm.placeholder(shape_y,
                                           name="data_input_y",
                                           dtype=dtype_input_y)
            data_input_dy = tvm.placeholder(shape_dy,
                                            name="data_input_dy",
                                            dtype=dtype_input_dy)
            res = rsqrt_grad_compute(data_input_y, data_input_dy, output_z, kernel_name)
            tensors.append([data_input_y, data_input_dy, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
