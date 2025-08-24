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
pow
"""
from impl.util import util_common
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_soc_common import after_v200
from impl.util.util_compute import only_static_support


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
        data_min = tbe.broadcast(tvm.const(2 ** (-126), dtype=dtype),
                                 shape, dtype)
        abs_x_min = tbe.vmin(abs_x, data_min)
        zero_mul_val = tbe.vmuls(abs_x_min, tvm.const(2 ** 62, dtype=dtype))
        zero_mul = tbe.vmuls(zero_mul_val, tvm.const(2 ** 62, dtype=dtype))
        zero_res = tbe.vmuls(zero_mul, tvm.const(2 ** 2, dtype=dtype))
    else:
        data_min = tbe.broadcast(tvm.const(2 ** (-24), dtype=dtype),
                                 shape, dtype)
        abs_x_min = tbe.vmin(abs_x, data_min)
        zero_mul_val = tbe.vmuls(abs_x_min, tvm.const(2 ** 12, dtype=dtype))
        zero_res = tbe.vmuls(zero_mul_val, tvm.const(2 ** 12, dtype=dtype))
    zero_index = tbe.vsub(data_one, zero_res)

    return zero_index


def _less_compute(input_x, input_y):
    """
    if x is less than y, then return 1, else return 0.
    """
    dtype = input_x.dtype
    shape = input_x.shape

    data_min = tbe.broadcast(tvm.const(2 ** (-126), dtype=dtype),
                             shape, dtype)
    data_zero = tbe.broadcast(tvm.const(0, dtype), shape, dtype)
    res_sub = tbe.vsub(input_y, input_x)
    res_min = tbe.vmin(res_sub, data_min)
    res_max = tbe.vmax(res_min, data_zero)

    # max num of float32 is 2**126
    # but cce can only support 2**62, so use 62/62/2 to adaptor 126
    res_mul_val = tbe.vmuls(res_max, tvm.const(2 ** 62, dtype=dtype))
    res_mul = tbe.vmuls(res_mul_val, tvm.const(2 ** 62, dtype=dtype))
    res = tbe.vmuls(res_mul, tvm.const(2 ** 2, dtype=dtype))

    return res


def _less_compute_fp16(input_x, input_y):
    """
    if x is less than y, then return 1, else return 0.
    """
    dtype = input_x.dtype
    shape = input_x.shape

    data_min = tbe.broadcast(tvm.const(2 ** (-24), dtype=dtype),
                             shape, dtype)
    data_zero = tbe.broadcast(tvm.const(0, dtype), shape, dtype)
    res_sub = tbe.vsub(input_y, input_x)
    res_min = tbe.vmin(res_sub, data_min)
    res_max = tbe.vmax(res_min, data_zero)

    # max num of float32 is 2**24
    # but cce can only support 2**24, so use 12/12 to adaptor 24
    res_mul_val = tbe.vmuls(res_max, tvm.const(2 ** 12, dtype=dtype))
    res_mul = tbe.vmuls(res_mul_val, tvm.const(2 ** 12, dtype=dtype))
    res = tbe.vmuls(res_mul, tvm.const(1, dtype=dtype))

    return res


def _positive_compute(input_x, input_y):
    """
    compute result of pow when data_x is more than 0,
    use exp(y * ln(x)).
    """
    dtype = input_x.dtype
    input_x = tbe.vabs(input_x)
    if not tbe_platform.api_check_support("tbe.dsl.vlog",
                                          dtype):
        input_x = tbe.cast_to(input_x, "float16")
        input_y = tbe.cast_to(input_y, "float16")

    log_value = tbe.vlog(input_x)

    mul_value = tbe.vmul(input_y, log_value)
    res = tbe.vexp(mul_value)

    if not tbe_platform.api_check_support("tbe.dsl.vlog",
                                          dtype):
        res = tbe.cast_to(res, dtype)

    return res


# 'pylint: disable=too-many-locals
def _negative_compute(input_x, input_y):
    """
    compute result of pow when data_x is less than 0,
    use [-2 * (|y| % 2) + 1] * exp(y * ln|x|)
    """
    input_dtype = input_x.dtype
    dtype = input_x.dtype
    shape = input_x.shape
    abs_value = tbe.vabs(input_y)

    vmod_support = tbe_platform.api_check_support("tbe.dsl.vmod", "float32")
    vlog_support = tbe_platform.api_check_support("tbe.dsl.vlog", input_dtype)

    if not vmod_support or not vlog_support:
        dtype = "float16"
        input_x = tbe.cast_to(input_x, "float16")
        input_y = tbe.cast_to(input_y, "float16")
        abs_value = tbe.cast_to(abs_value, "float16")

    data_two = tbe.broadcast(tvm.const(2, dtype), shape, dtype)
    mod_value = tbe.vmod(abs_value, data_two)
    mul_value = tbe.vmuls(mod_value, tvm.const(-2, dtype))
    add_value = tbe.vadds(mul_value, tvm.const(1, dtype))

    abs_data_x = tbe.vabs(input_x)
    log_value = tbe.vlog(abs_data_x)

    mul_value = tbe.vmul(input_y, log_value)

    if tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        add_value = tbe.cast_to(add_value, "float32")
        mul_value = tbe.cast_to(mul_value, "float32")

    exp_value = tbe.vexp(mul_value)
    res = tbe.vmul(add_value, exp_value)

    if res.dtype != input_dtype:
        res = tbe.cast_to(res, input_dtype)
    return res


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals
def _powf(x, y, xdtype, xshape):
    """
    the signbit of x is 1 and y is int, returns negtive_compute;
    |y| is 0.0, returns 1.0;
    x is 1.0, returns 1.0;
    x is -1.0 and |y| is inf, returns 1.0;
    x is finite and x lt zero, y is finite and y is not int, returns nan;
    the rest returns positive_compute;
    """
    first_res = _positive_compute(x, y)
    second_res = _negative_compute(x, y)

    tensor_zero = tbe.broadcast(tvm.const(0.0, xdtype), xshape)
    tensor_one = tbe.broadcast(tvm.const(1.0, xdtype), xshape)
    tensor_ne_one = tbe.broadcast(tvm.const(-1.0, xdtype), xshape)
    neg_inf = tvm.const(float("-inf"), xdtype)
    pos_inf = tvm.const(float("inf"), xdtype)
    const_nan = tvm.const(float("nan"), xdtype)

    x_lt_zero_mask = tbe.vcmp(x, tensor_zero, "lt", "bit")
    x_lt_zero = tbe.vsel(x_lt_zero_mask, tensor_one, tensor_zero)
    x_signbit = tbe.vsignbit(x)

    floor_y = tbe.floor(y, "float32")
    y_is_not_int_mask = tbe.vcmp(floor_y, y, "ne", "bit")
    y_is_int = tbe.vsel(y_is_not_int_mask, tensor_zero, tensor_one)

    abs_y = tbe.vabs(y)
    abs_y_inf = tbe.vcmp(abs_y, pos_inf, "eq", "bit")
    y_is_int_not_inf = tbe.vsel(abs_y_inf, tensor_zero, y_is_int)

    second_index = tbe.vmul(x_signbit, y_is_int_not_inf)
    res_temp_mask = tbe.vcmp(second_index, tensor_one, "eq", "bit")
    res_temp = tbe.vsel(res_temp_mask, second_res, first_res)

    x_ne_inf_mask = tbe.vcmp(x, neg_inf, "eq", "bit")
    x_lt_zero_gt_ne_inf = tbe.vsel(x_ne_inf_mask, tensor_zero, x_lt_zero)
    y_is_not_int = tbe.vsub(tensor_one, y_is_int_not_inf)
    
    y_is_float_and_not_inf = tbe.vsel(abs_y_inf, tensor_zero, y_is_not_int)
    fouth_index = tbe.vmul(x_lt_zero_gt_ne_inf, y_is_float_and_not_inf)
    res_temp_mask = tbe.vcmp(fouth_index, tensor_one, "eq", "bit")
    res_temp = tbe.vsel(res_temp_mask, const_nan, res_temp)

    x_ne_one_mask = tbe.vcmp(x, tensor_ne_one, "eq", "bit")
    x_ne_one = tbe.vsel(x_ne_one_mask, tensor_one, tensor_zero)
    x_ne_one_abs_y_inf = tbe.vsel(abs_y_inf, x_ne_one, tensor_zero)
    res_temp_mask = tbe.vcmp(x_ne_one_abs_y_inf, tensor_one, "eq", "bit")
    res_temp = tbe.vsel(res_temp_mask, tensor_one, res_temp)

    res_temp_mask = tbe.vcmp(y, tensor_zero, "eq", "bit")
    res_temp = tbe.vsel(res_temp_mask, tensor_one, res_temp)
    res_mask = tbe.vcmp(x, tensor_one, "eq", "bit")
    res = tbe.vsel(res_mask, tensor_one, res_temp)

    return res


def pow_compute_b32(input_x, input_y, shape_broadcast):
    """
    power compute with powi instruction
    """
    x = tbe.broadcast(input_x, shape_broadcast)
    y = tbe.broadcast(input_y, shape_broadcast)
    res = tbe.powi(x, y)

    return res


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals
@register_operator_compute("Pow", op_mode="dynamic", support_fusion=only_static_support, support_bfp16=True)
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
    shape_list = shape_util.broadcast_shapes(input_x.shape, input_y.shape, param_name_input1="input_x",
                                             param_name_input2="input_y")

    has_improve_precision = False
    x_cast = input_x
    y_cast = input_y
    cast_dtype = "float16"
    cur_support = tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ('Ascend310P', 'Ascend910') or after_v200()
    if cur_support and x_dtype in ("int8", "uint8") and tbe_platform.api_check_support("tbe.dsl.powi", "int32"):
        input_x = tbe.cast_to(input_x, "int32")
        input_y = tbe.cast_to(input_y, "int32")
        res_int32 = pow_compute_b32(input_x, input_y, shape_list[2])
        res = util_common.int_cast_to_b8(res_int32, x_dtype)

        return res

    if x_dtype == "int32" and tbe_platform.api_check_support("tbe.dsl.powi", x_dtype):
        return pow_compute_b32(input_x, input_y, shape_list[2])
    if tbe_platform.api_check_support("tbe.dsl.vdiv", "float32"):
        x_cast = tbe.cast_to(input_x, "float32")
        y_cast = tbe.cast_to(input_y, "float32")
        has_improve_precision = True
        cast_dtype = "float32"

    x = tbe.broadcast(x_cast, shape_list[2])
    y = tbe.broadcast(y_cast, shape_list[2])

    if cur_support and tbe_platform.api_check_support("tbe.dsl.round", "f322f32"):
        res = _powf(x, y, cast_dtype, shape_list[2])
    else:
        data_zero = tbe.broadcast(tvm.const(0, cast_dtype), shape_list[2], cast_dtype)
        data_one = tbe.broadcast(tvm.const(1, cast_dtype), shape_list[2], cast_dtype)
        data_threshold = tbe.broadcast(tvm.const(0.00000001, cast_dtype), shape_list[2])
        if has_improve_precision:
            data_x_negative = _less_compute(x, data_zero)
        else:
            data_x_negative = _less_compute_fp16(x, data_zero)

        # compute result of pow when x is more than 0
        data_one = tbe.broadcast(tvm.const(1, cast_dtype), shape_list[2], cast_dtype)
        zero_index_x = _is_equal_zero(x)
        zero_index_y = _is_equal_zero(y)
        x = tbe.vadd(x, zero_index_x)
        res_val_positive = _positive_compute(x, y)
        sub_one_val = tbe.vsub(data_one, data_x_negative)
        sub_one_val_except_zero = tbe.vsub(sub_one_val, zero_index_x)
        res_positive = tbe.vmul(res_val_positive, sub_one_val_except_zero)

        # compute result of pow when x is less than 0
        res_tmp_negative = _negative_compute(x, y)
        res_negative = tbe.vmul(res_tmp_negative, data_x_negative)
        # compute result of pow when x is equal 0
        zero_index = tbe.vmul(zero_index_x, zero_index_y)

        tmp_res = tbe.vadd(res_positive, res_negative)
        res = tbe.vadd(tmp_res, zero_index)

    if x_dtype == "int32":
        res = tbe.round(res)
    else:
        res = tbe.cast_to(res, x_dtype)

    return res


# 'pylint: disable=locally-disabled,redefined-builtin
@register_operator("Pow")
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
    shapex = input_x.get("shape")
    shapey = input_y.get("shape")

    input_x_dtype = input_x.get("dtype").lower()
    input_y_dtype = input_y.get("dtype").lower()
    if input_x_dtype != input_y_dtype:
        error_detail = "Dtype of input_x and input_y must be the same."
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "input_x", \
                                                               "input_y", error_detail)
    check_list = ("bfloat16", "float16", "float32", "int8", "uint8", "int32")
    para_check.check_dtype(input_x_dtype, check_list, param_name="input_x")

    ins = classify([input_x, input_y], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []

    for (_input_x, _input_y) in ins:
        with tbe.compute():
            shapex, shapey = shape_util.variable_shape([_input_x, _input_y])
            shapex, shapey = shape_util.refine_shapes_for_broadcast(shapex, shapey)
            datax = tvm.placeholder(shapex, dtype=input_x_dtype, name="datax")
            datay = tvm.placeholder(shapey, dtype=input_y_dtype, name="datay")
            res = pow_compute(datax, datay, output_z, kernel_name="pow")
            tensors.append([datax, datay, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors,
              "bool_storage_as_1bit": False}
    tbe.build(schedules, config)
