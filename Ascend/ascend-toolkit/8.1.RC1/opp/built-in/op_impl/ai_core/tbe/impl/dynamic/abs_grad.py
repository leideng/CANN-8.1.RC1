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
abs_grad

Op_description :
Computes gradients for abs operation

# abs_grad(
#   y,
#   dy,
#   z,
#   kernel_name="cce_abs_grad")

Supportive_dtype_format :
['bfloat16', 'float16', 'float32']
['ALL']

Constraint :
[1] All : 'y' and 'dy' must have the same type and shape.
[2] All : shape size limit is 2147483648.
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_soc_common import after_v200
from impl.dynamic.sign import sign_compute


# 'pylint: disable=unused-argument,too-many-locals,invalid-name
@register_operator_compute("AbsGrad", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def abs_grad_compute(y, dy, z, kernel_name="abs_grad"):
    """
    do abs_grad compute
    Parameters:
    ----------------
    y: input tensor y
    dy: input tensor dy
    z: output dict
    kernel_name: cce kernel name, default value is "abs_grad"
    return: data_dy * sign(data_y)
    ----------------
    """

    dtype = dy.dtype

    if after_v200():
        scalar_zero = tvm.const(0, dtype)
        mask = tbe.vcmp(y, y, "eq", "bit")
        y_no_nan = tbe.vsel(mask, y, scalar_zero)
        sign_res = sign_compute(y_no_nan, z)
        data1_res = tbe.vmul(sign_res, dy)
        return data1_res

    if dtype == "float16":
        fp_max = tvm.const(2 ** 15, dtype)
        fp_min = tvm.const(2 ** (-15), dtype)
    else:
        fp_max = tvm.const(2 ** 62, dtype)
        fp_min = tvm.const(2 ** (-126), dtype)
    new_data = tbe.vmuls(y, fp_max)
    abs_data = tbe.vabs(new_data)
    denominator = tbe.vadds(abs_data, fp_min)
    res = tbe.vdiv(new_data, denominator)
    res = tbe.round(res)
    res = tbe.cast_to(res, dtype)
    data1_res = tbe.vmul(res, dy)
    return data1_res


# 'pylint: disable=invalid-name
@register_operator("AbsGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def abs_grad(y, dy, z, kernel_name="abs_grad"):
    """
    do element-wise abs_grad operation between two input tensors

    Parameters:
    ----------
    y : dict of y, include shape and dtype, dtype support bfloat16, float16, float32

    dy : dict of dy, include shape and dtype, dtype support bfloat16, float16, float32

    z : dict of z, include shape and dtype, dtype support bfloat16, float16, float32

    kernel_name : cce kernel name, default value is "abs_grad"
    -------
    """

    # get the shape and dtype for input_1,input_2
    dtype_y = y.get("dtype").lower()
    dtype_dy = dy.get("dtype").lower()
    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(dtype_y, check_list, param_name="y")
    para_check.check_dtype(dtype_dy, check_list, param_name="dy")

    ins = classify([y, dy], OpPatternMode.ELEWISE)
    schedules, tensors = [], []

    for (input_y, input_dy) in ins:
        with tbe.compute():
            shape_y, shape_dy = shape_util.variable_shape([input_y, input_dy])
            data_y = tvm.placeholder(shape_y, name="data_y", dtype=dtype_y)
            data_dy = tvm.placeholder(shape_dy, name="data_dy", dtype=dtype_dy)
            res = abs_grad_compute(data_y, data_dy, z, kernel_name)
            tensors.append([data_y, data_dy, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
