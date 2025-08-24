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
pow
"""
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=invalid-name
def _is_equal_zero(x):
    """
    if x==0,then return 1,else return 0.
    """
    dtype = x.dtype
    shape = x.shape
    data_one = tbe.broadcast(tvm.const(1, dtype), shape, dtype)
    abs_x = tbe.vabs(x)
    if dtype == "float32":
        data_min = tbe.broadcast(tvm.const(2**(-126), dtype=dtype), shape, dtype)
        abs_x_min = tbe.vmin(abs_x, data_min)
        zero_mul_val = tbe.vmuls(abs_x_min, tvm.const(2**62, dtype=dtype))
        zero_mul = tbe.vmuls(zero_mul_val, tvm.const(2**62, dtype=dtype))
        zero_res = tbe.vmuls(zero_mul, tvm.const(2**2, dtype=dtype))
    else:
        data_min = tbe.broadcast(tvm.const(2**(-24), dtype=dtype), shape, dtype)
        abs_x_min = tbe.vmin(abs_x, data_min)
        zero_mul_val = tbe.vmuls(abs_x_min, tvm.const(2**12, dtype=dtype))
        zero_res = tbe.vmuls(zero_mul_val, tvm.const(2**12, dtype=dtype))
    zero_index = tbe.vsub(data_one, zero_res)

    return zero_index


def _less_compute(input_x, input_y):
    """
    if x is less than y, then return 1, else return 0.
    """
    dtype = input_x.dtype
    shape = input_x.shape

    data_min = tbe.broadcast(tvm.const(2**(-126), dtype=dtype), shape, dtype)
    data_zero = tbe.broadcast(tvm.const(0, dtype), shape, dtype)
    res_sub = tbe.vsub(input_y, input_x)
    res_min = tbe.vmin(res_sub, data_min)
    res_max = tbe.vmax(res_min, data_zero)

    # max num of float32 is 2**126
    # but cce can only support 2**62, so use 62/62/2 to adaptor 126
    res_mul_val = tbe.vmuls(res_max, tvm.const(2**62, dtype=dtype))
    res_mul = tbe.vmuls(res_mul_val, tvm.const(2**62, dtype=dtype))
    res = tbe.vmuls(res_mul, tvm.const(2**2, dtype=dtype))

    return res


def _less_compute_fp16(input_x, input_y):
    """
    if x is less than y, then return 1, else return 0.
    """
    dtype = input_x.dtype
    shape = input_x.shape

    data_min = tbe.broadcast(tvm.const(2**(-24), dtype=dtype), shape, dtype)
    data_zero = tbe.broadcast(tvm.const(0, dtype), shape, dtype)
    res_sub = tbe.vsub(input_y, input_x)
    res_min = tbe.vmin(res_sub, data_min)
    res_max = tbe.vmax(res_min, data_zero)

    # max num of float32 is 2**24
    # but cce can only support 2**24, so use 12/12 to adaptor 24
    res_mul_val = tbe.vmuls(res_max, tvm.const(2**12, dtype=dtype))
    res_mul = tbe.vmuls(res_mul_val, tvm.const(2**12, dtype=dtype))
    res = tbe.vmuls(res_mul, tvm.const(1, dtype=dtype))

    return res


def _positive_compute(input_x, input_y):
    """
    compute result of pow when data_x is more than 0,
    use exp(y * ln(x)).
    """
    input_x = tbe.vabs(input_x)
    log_value = tbe.vlog(input_x)
    mul_value = tbe.vmul(input_y, log_value)
    res = tbe.vexp(mul_value)

    return res


def _negative_compute(input_x, input_y):
    """
    compute result of pow when data_x is less than 0,
    use [-2 * (|y| % 2) + 1] * exp(y * ln|x|)
    """
    dtype = input_x.dtype
    shape = input_x.shape
    abs_value = tbe.vabs(input_y)

    if not tbe_platform.api_check_support("te.lang.cce.vmod", "float32"):
        dtype = "float16"
        abs_value = tbe.cast_to(abs_value, "float16")

    data_two = tbe.broadcast(tvm.const(2, dtype), shape, dtype)
    mod_value = tbe.vmod(abs_value, data_two)
    mul_value = tbe.vmuls(mod_value, tvm.const(-2, dtype))
    add_value = tbe.vadds(mul_value, tvm.const(1, dtype))

    if tbe_platform.api_check_support("te.lang.cce.vexp", "float32"):
        add_value = tbe.cast_to(add_value, "float32")

    abs_data_x = tbe.vabs(input_x)
    log_value = tbe.vlog(abs_data_x)
    mul_value = tbe.vmul(input_y, log_value)
    exp_value = tbe.vexp(mul_value)
    res = tbe.vmul(add_value, exp_value)

    return res


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals
@register_operator_compute("pow", op_mode="static", support_fusion=True)
def pow_compute(input_x, input_y, output_z, kernel_name="pow"):
    """
    pow compute
    calculating data pow, res =x ^ y,
    x > 0: use exp(y*ln(x))
    x < 0: use [-2*(|y|%2)+1]*exp(y*ln|x|)
    x = 0: 0^0=1 & 0^y=0

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input_x
    input_y: TVM tensor
        the placeholder of input_y
    output_z: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        kernel name, default value is "pow"

    Returns
    -------
    res: TVM tensor
        the result of pow compute
    """
    x_dtype = input_x.dtype.lower()
    shape_x = shape_util.shape_to_list(input_x.shape)
    shape_y = shape_util.shape_to_list(input_y.shape)
    list_shape = shape_util.broadcast_shapes(shape_x, shape_y, param_name_input1="input_x", param_name_input2="input_y")

    has_improve_precision = False
    data_x_cast = input_x
    data_y_cast = input_y
    cast_dtype = "float16"
    if tbe_platform.api_check_support("tbe.dsl.vdiv", "float32"):
        data_x_cast = tbe.cast_to(input_x, "float32")
        data_y_cast = tbe.cast_to(input_y, "float32")
        has_improve_precision = True
        cast_dtype = "float32"

    x = tbe.broadcast(data_x_cast, list_shape[2])
    y = tbe.broadcast(data_y_cast, list_shape[2])

    data_zero = tbe.broadcast(tvm.const(0, cast_dtype), list_shape[2], cast_dtype)
    if has_improve_precision:
        data_x_negative = _less_compute(x, data_zero)
    else:
        data_x_negative = _less_compute_fp16(x, data_zero)

    # compute result of pow when x is more than 0
    data_one = tbe.broadcast(tvm.const(1, cast_dtype), list_shape[2], cast_dtype)
    x_zero_index = _is_equal_zero(x)
    y_zero_index = _is_equal_zero(y)
    x = tbe.vadd(x, x_zero_index)
    positive = _positive_compute(x, y)
    sub_one = tbe.vsub(data_one, data_x_negative)
    sub_one_except_zero = tbe.vsub(sub_one, x_zero_index)
    res_positive = tbe.vmul(positive, sub_one_except_zero)

    # compute result of pow when data_x is less than 0
    val_negative = _negative_compute(x, y)
    res_tmp_negative = tbe.vmul(val_negative, data_x_negative)
    # compute result of pow when data_x is equal 0
    zero_index = tbe.vmul(x_zero_index, y_zero_index)

    tmp_res = tbe.vadd(res_positive, res_tmp_negative)
    res = tbe.vadd(tmp_res, zero_index)
    if x_dtype == "int32":
        res = tbe.round(res)
    else:
        res = tbe.cast_to(res, x_dtype)

    return res


# 'pylint: disable=locally-disabled,redefined-builtin
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def pow(input_x, input_y, output_z, kernel_name="pow"):
    """
    algorithm: pow
    calculating data pow, res =x ** y

    Parameters
    ----------
    input_x: dict
        dict with keys(shape and dtype) of input_x
    input_y: dict
        dict with keys(shape and dtype) of input_y
    output_z: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        kernel name, default value is "pow"

    Returns
    -------
    None
    """
    shape_x = input_x.get("shape")
    shape_y = input_y.get("shape")
    if len(shape_x) == 0:
        shape_x = (1, )
    if len(shape_y) == 0:
        shape_y = (1, )
    para_check.check_shape(shape_x, param_name="input_x")
    para_check.check_shape(shape_y, param_name="input_y")
    list_shape = shape_util.broadcast_shapes(shape_x, shape_y, param_name_input1="input_x", param_name_input2="input_y")

    input_x_dtype = input_x.get("dtype").lower()
    input_y_dtype = input_y.get("dtype").lower()
    if input_x_dtype != input_y_dtype:
        error_detail = "Dtype of input_x and input_y must be the same."
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "input_x", \
                                                               "input_y", error_detail)
    check_list = ("float16", "float32", "int8", "uint8", "int32")
    para_check.check_dtype(input_x_dtype, check_list, param_name="input_x")

    x_shape, y_shape = shape_util.refine_shapes_for_broadcast(list_shape[0], list_shape[1])
    datax = tvm.placeholder(x_shape, dtype=input_x_dtype, name="data_x")
    datay = tvm.placeholder(y_shape, dtype=input_y_dtype, name="data_y")
    res = pow_compute(datax, datay, output_z, kernel_name="pow")

    with tvm.target.cce():
        sch = auto_schedule(res)
    config = {"name": kernel_name, "tensor_list": [datax, datay, res], "bool_storage_as_1bit": False}
    build(sch, config)