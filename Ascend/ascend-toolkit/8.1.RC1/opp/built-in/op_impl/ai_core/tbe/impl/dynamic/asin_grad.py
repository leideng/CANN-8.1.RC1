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
asin_grad

Op_description :
Computes gradients for Asin operation

# asin_grad(
#   y,
#   dy,
#   z,
#   kernel_name="cce_asin_grad")

Supportive_dtype_format :
['bfloat16', 'float16', 'float32']
['ALL']

Constraint :
[1] All : 'y' and 'dy' must have the same type and shape.
[2] All : shape size limit is 2147483648.
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import OpTbeImplMode


# 'pylint: disable=unused-argument,invalid-name,too-many-locals
@register_operator_compute("AsinGrad", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def asin_grad_compute(y, dy, z, kernel_name="asin_grad"):
    """
    do element-wise asin_grad compute

    Parameters:
    ----------
    y : the placeholders of input y

    dy : the placeholders of input dy

    z : output dict

    kernel_name : cce kernel name, default value is "cce_asin_grad"

    return : dy * (1 / sqrt(1 - y^2))
    -------
    """

    # scalar in asin_grad and Newton's equation
    num_minus_one = -1
    num_one = 1
    dtype = y.dtype
    if dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.vadd", "float32"):
        y = tbe.cast_to(y, "float32")
        dy = tbe.cast_to(dy, "float32")

    # `step 1: calculate num_to_vrsqrt = 1 - y^2`
    data = tbe.vmul(y, y)
    data = tbe.vmuls(data, tvm.const(num_minus_one, y.dtype))
    num_to_vrsqrt = tbe.vadds(data, tvm.const(num_one, y.dtype))

    # step 2: calculate dy * (1 / sqrt(1 - y^2))
    vsqrt_res = tbe.vsqrt(num_to_vrsqrt, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)
    res = tbe.vdiv(dy, vsqrt_res)

    if dtype == "float16":
        res = tbe.cast_to(res, "float16")

    return res


@register_operator("AsinGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def asin_grad(y, dy, z, kernel_name="asin_grad"):
    """
    do element-wise asin_grad operation between two input tensors

    Parameters:
    ----------
    y : dict of y, include shape and dtype, dtype support bfloat16, float16, float32

    dy : dict of dy, include shape and dtype, dtype support bfloat16, float16, float32

    z : dict of output

    kernel_name : cce kernel name, default value is "asin_grad"

    Returns
    -------
    None
    """

    # get the dtype
    dtype_y = y.get("dtype").lower()
    dtype_dy = dy.get("dtype").lower()

    # check whether dtypes are fp16,fp32 and whether they are the same
    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(dtype_y, check_list, param_name="y")
    para_check.check_dtype(dtype_dy, check_list, param_name="dy")

    if dtype_y != dtype_dy:
        error_manager_vector.raise_err_inputs_dtype_not_equal('asin_grad', 'dtype_y', 'dtype_dy',
                                                              str(dtype_y), str(dtype_dy))

    # get 2 input tensors: data_y, data_dy
    ins = classify([y, dy], OpPatternMode.ELEWISE)
    schedules, tensors = [], []

    for (_y, _dy) in ins:
        with tbe.compute():
            shape_y, shape_dy = shape_util.variable_shape([_y, _dy])
            data_y = tvm.placeholder(shape_y, name="data_y", dtype=dtype_y)
            data_dy = tvm.placeholder(shape_dy, name="data_dy", dtype=dtype_y)
            res = asin_grad_compute(data_y, data_dy, z, kernel_name)

            tensors.append([data_y, data_dy, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
