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
asinh_grad

Op_description :
Computes gradients for Asinh operation

# asinh_grad(
#   y,
#   dy,
#   z,
#   kernel_name="cce_asinh_grad")

Supportive_dtype_format :
['bfloat16' ,'float16', 'float32']
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
from impl.dynamic.cosh import cosh_compute


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # scalar in asinh_grad
    NUM_TWO = 2
    NUM_ONE = 1


def _cosh_taylor_compute(data):
    """
    Calculate cosh  = 1 + x^2( 1/2! + x^2( 1/4! + x^2/6!))

    Parameters:
    ----------
    data : the placeholder of data input

    Returns
    -------
    A Tensor represents cosh(data). Has the same type as data.
    """

    taylor_second = 0.5
    taylor_fourth = 1 / 24.0
    taylor_sixth = 1 / 720.0
    # x^2 / 6!
    pow_2 = tbe.vmul(data, data)
    pow_2_div = tbe.vmuls(pow_2, tvm.const(taylor_sixth, data.dtype))

    # 1/4! + x^2 / 6!
    pow_2_plus = tbe.vadds(pow_2_div, tvm.const(taylor_fourth, data.dtype))

    # 1/2! + x^2( 1/4! + x^2/6!)
    pow_4 = tbe.vmul(pow_2_plus, pow_2)
    pow_4_plus = tbe.vadds(pow_4, tvm.const(taylor_second, data.dtype))

    # 1 + x^2( 1/2! + x^2( 1/4! + x^2/6!))
    pow_6 = tbe.vmul(pow_4_plus, pow_2)
    res = tbe.vadds(pow_6, tvm.const(Constant.NUM_ONE, data.dtype))

    return res


def _cosh_repeat(data):
    """
    Calculate f(2x) = 2f(x)^2 -1

    Parameters:
    ----------
    data : the placeholder of data input

    Returns
    -------
    A Tensor represents f(2x). Has the same type as data.
    """

    num_minus_one = -1
    data_square = tbe.vmul(data, data)
    data_mul = tbe.vmuls(data_square, tvm.const(Constant.NUM_TWO, data.dtype))
    res = tbe.vadds(data_mul, tvm.const(num_minus_one, data.dtype))

    return res


# 'pylint: disable=unused-argument,invalid-name,too-many-locals
@register_operator_compute("AsinhGrad", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def asinh_grad_compute(y, dy, output_res, kernel_name="cce_asinh_grad"):
    """
    do element-wise asinh_grad compute

    Parameters:
    ----------
    y : the placeholders of input y

    dy : the placeholders of input dy

    output_res : output dict

    kernel_name : cce kernel name, default value is "cce_asinh_grad"

    Return :
    -------
    dy * (1/cosh(y))
    """

    num_repeat = 0.125
    dtype = y.dtype
    if dtype == "float16" and \
            tbe_platform.api_check_support("tbe.dsl.vadd", "float32"):
        y = tbe.cast_to(y, "float32")
        dy = tbe.cast_to(dy, "float32")

    if tbe_platform.api_check_support('tbe.dsl.vexp', 'float32'):
        res = cosh_compute(y, output_res)
        res = tbe.vdiv(dy, res)
    else:
        # use taylor's method for high accuracy result
        y = tbe.vmuls(y, tvm.const(num_repeat, y.dtype))
        cosh_value_0 = _cosh_taylor_compute(y)
        # repeat 3 times
        cosh_value_1 = _cosh_repeat(cosh_value_0)
        cosh_value_2 = _cosh_repeat(cosh_value_1)
        cosh_value = _cosh_repeat(cosh_value_2)
        res = tbe.vrec(cosh_value, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)
        res = tbe.vmul(res, dy)

    if dtype == "float16":
        res = tbe.cast_to(res, "float16")

    return res


@register_operator("AsinhGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def asinh_grad(y, dy, z, kernel_name="cce_asinh_grad"):
    """
    do element-wise asinh_grad operation between two input tensors

    Parameters:
    ----------
    y : dict of y, include shape and dtype, dtype support bfloat16, float16, float32

    dy : dict of dy, include shape and dtype, dtype support bfloat16, float16, float32

    z : dict of output

    kernel_name : cce kernel name, default value is "cce_asinh_grad"

    Returns
    -------
    None
    """

    # get the shape and dtype
    shape_y = y.get("shape")
    shape_dy = dy.get("shape")
    dtype_y = y.get("dtype").lower()
    dtype_dy = dy.get("dtype").lower()

    # kernel name check: should be unique

    # check whether the shape is right
    para_check.check_shape(shape_y, param_name="y")
    para_check.check_shape(shape_dy, param_name="dy")

    # check whether dtypes are fp16,fp32 and whether they are the same
    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(dtype_y, check_list, param_name="y")
    para_check.check_dtype(dtype_dy, check_list, param_name="dy")
    if dtype_y != dtype_dy:
        error_manager_vector.raise_err_inputs_dtype_not_equal('asinh_grad', 'dtype_y', 'dtype_dy',
                                                              str(dtype_y), str(dtype_dy))

    ins = classify([y, dy], OpPatternMode.ELEWISE)
    schedules, tensors = [], []

    for (input_y, input_dy) in ins:
        with tbe.compute():
            shape_y, shape_dy = shape_util.variable_shape([input_y, input_dy])
            data_y = tvm.placeholder(shape_y, name="data_y", dtype=dtype_y)
            data_dy = tvm.placeholder(shape_dy, name="data_dy", dtype=dtype_dy)
            res = asinh_grad_compute(data_y, data_dy, z, kernel_name)

            tensors.append([data_y, data_dy, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
