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
dynamic erf
"""

from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util import util_soc_common


def calc_expand_zero(sel_x):
    # expand from zero
    taylor_0 = tvm.const(0.128379151, "float32")
    taylor_1 = tvm.const(-0.376126647, "float32")
    taylor_2 = tvm.const(0.112840049, "float32")
    taylor_3 = tvm.const(-0.0268687736, "float32")
    taylor_4 = tvm.const(0.00521348882, "float32")
    taylor_5 = tvm.const(-0.000821309164, "float32")
    taylor_6 = tvm.const(0.0000848349446, "float32")

    square_x = tbe.vmul(sel_x, sel_x)
    step_1 = tbe.vmuls(square_x, taylor_6)
    step_1 = tbe.vadds(step_1, taylor_5)
    step_2 = tbe.vmul(step_1, square_x)
    step_2 = tbe.vadds(step_2, taylor_4)
    step_3 = tbe.vmul(step_2, square_x)
    step_3 = tbe.vadds(step_3, taylor_3)
    step_4 = tbe.vmul(step_3, square_x)
    step_4 = tbe.vadds(step_4, taylor_2)
    step_5 = tbe.vmul(step_4, square_x)
    step_5 = tbe.vadds(step_5, taylor_1)
    step_6 = tbe.vmul(step_5, square_x)
    step_6 = tbe.vadds(step_6, taylor_0)
    expand_zero = tbe.vmul(step_6, sel_x)
    expand_zero = tbe.vadd(expand_zero, sel_x)
    return expand_zero


def calc_expand_inf(input_x, abs_x):
    const_zero = tvm.const(0.0, "float32")
    const_one = tvm.const(1.0, "float32")
    const_neg_one = tvm.const(-1.0, "float32")
    ln_2 = tvm.const(0.69314718056, "float32")

    # expand from infinity
    coeff_0 = tvm.const(0.000112198715, "float32")
    coeff_1 = tvm.const(-0.00132752524, "float32")
    coeff_2 = tvm.const(0.00839653518, "float32")
    coeff_3 = tvm.const(-0.0402465835, "float32")
    coeff_4 = tvm.const(0.159504309, "float32")
    coeff_5 = tvm.const(0.912917674, "float32")
    coeff_6 = tvm.const(0.629060030, "float32")

    step_1 = tbe.vmuls(abs_x, coeff_0)
    step_1 = tbe.vadds(step_1, coeff_1)
    step_2 = tbe.vmul(step_1, abs_x)
    step_2 = tbe.vadds(step_2, coeff_2)
    step_3 = tbe.vmul(step_2, abs_x)
    step_3 = tbe.vadds(step_3, coeff_3)
    step_4 = tbe.vmul(step_3, abs_x)
    step_4 = tbe.vadds(step_4, coeff_4)
    step_5 = tbe.vmul(step_4, abs_x)
    step_5 = tbe.vadds(step_5, coeff_5)
    step_6 = tbe.vmul(step_5, abs_x)
    step_6 = tbe.vadds(step_6, coeff_6)
    neg_abs = tbe.vmuls(abs_x, const_neg_one)
    step_7 = tbe.vmul(step_6, neg_abs)
    step_7 = tbe.vadd(step_7, neg_abs)
    pow_1 = tbe.vmuls(step_7, ln_2)
    pow_2 = tbe.vexp(pow_1)
    pow_2 = tbe.vmuls(pow_2, const_neg_one)
    pow_3 = tbe.vadds(pow_2, const_one)
    neg_abs_pow = tbe.vabs(pow_3)
    neg_abs_pow = tbe.vmuls(neg_abs_pow, const_neg_one)

    pos_neg_mask = tbe.vcmp(input_x, const_zero, operation='ge', mode='bit')
    expand_inf = tbe.vsel(pos_neg_mask, pow_3, neg_abs_pow)
    return expand_inf


def erf_high_precision_compute(input_x):
    """
    mathematical formula of erf(x):
    |x| <= 1.00295997:
    erf(x) =
        ((((((x^2 * taylor_6 + taylor_5) * x^2 + taylor_4) * x^2 + taylor_3) * x^2 + taylor_2)
            * x^2 + taylor_1) * x^2 + taylor_0) * x

    |x| > 1.00295997:
    erf(x) =
        ((((((coeff_0*|x| + coeff_1)*|x| + coeff_2)*|x| + coeff_3)*|x| +
        coeff_4)*|x| + coeff_5)*|x| + coeff_6)*(-|x|)
    """

    threshold_value = tvm.const(1.00295997, "float32")
    const_zero = tvm.const(0.0, "float32")

    abs_x = tbe.vabs(input_x)
    threshold_mask = tbe.vcmp(abs_x, threshold_value, operation='ge', mode='bit')
    sel_x = tbe.vsel(threshold_mask, const_zero, input_x)

    expand_zero = calc_expand_zero(sel_x)
    expand_inf = calc_expand_inf(input_x, abs_x)

    res = tbe.vsel(threshold_mask, expand_inf, expand_zero)
    return res


def erf_compute_with_simplified_formula(input_x):
    """
    Use the simplified formula to calulate the erf.
    """
    dtype = input_x.dtype
    ori_dtype = input_x.dtype
    if dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.vdiv", "float32"):
        dtype = "float32"
        input_x = tbe.cast_to(input_x, "float32")

    const_pos_threshold = tvm.const(3.92, dtype=dtype)
    const_neg_threshold = tvm.const(-3.92, dtype=dtype)
    input_x = tbe.vmins(input_x, const_pos_threshold)
    input_x = tbe.vmaxs(input_x, const_neg_threshold)

    x2 = tbe.vmul(input_x, input_x)

    numer = tbe.vmuls(x2, tvm.const(0.53443748819e-1, dtype=dtype))
    numer = tbe.vadds(numer, tvm.const(0.75517016694e1, dtype=dtype))
    numer = tbe.vmul(numer, x2)
    numer = tbe.vadds(numer, tvm.const(0.10162808918e3, dtype=dtype))
    numer = tbe.vmul(numer, x2)
    numer = tbe.vadds(numer, tvm.const(0.13938061484e4, dtype=dtype))
    numer = tbe.vmul(numer, x2)
    numer = tbe.vadds(numer, tvm.const(0.50637915060e4, dtype=dtype))
    numer = tbe.vmul(numer, x2)
    numer = tbe.vadds(numer, tvm.const(0.29639384698e5, dtype=dtype))
    numer = tbe.vmul(numer, input_x)

    denom = tbe.vadds(x2, tvm.const(0.31212858877e2, dtype=dtype))
    denom = tbe.vmul(denom, x2)
    denom = tbe.vadds(denom, tvm.const(0.39856963806e3, dtype=dtype))
    denom = tbe.vmul(denom, x2)
    denom = tbe.vadds(denom, tvm.const(0.30231248150e4, dtype=dtype))
    denom = tbe.vmul(denom, x2)
    denom = tbe.vadds(denom, tvm.const(0.13243365831e5, dtype=dtype))
    denom = tbe.vmul(denom, x2)
    denom = tbe.vadds(denom, tvm.const(0.26267224157e5, dtype=dtype))

    res = tbe.vdiv(numer, denom)

    if dtype != ori_dtype:
        res = tbe.cast_to(res, ori_dtype)

    return res


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals,too-many-statements,invalid-name
@register_operator_compute("Erf", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def erf_compute(input_x, output_y, kernel_name="erf"):
    """
    compute erf

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        he dict of output_data, include keys(shape and dtype)
    kernel_name: str
        kernel name, default value is "erf"

    Returns
    -------
    erf_result: TVM tensor
        the =result of compute
    """
    if util_soc_common.after_v200():
        return erf_compute_with_simplified_formula(input_x)

    # `define a scaler, value = 1`
    scalar_one = 1
    # `define a scaler, value = -1`
    scalar_negative_one = -1
    # `define a scaler, value = -0.47047, only used in compute of erf and erfc`
    scalar_p = 0.47047
    # `define a scaler, value = 0.3480242, only used in compute of erf and erfc`
    scalar_a = 0.3480242
    # `define a scaler, value = -0.0958798, only used in compute of erf and erfc`
    scalar_b = -0.0958798
    # `define a scaler, value = 0.7478556, only used in compute of erf and erfc`
    scalar_c = 0.7478556

    dtype = input_x.dtype
    dtype_ = input_x.dtype
    if dtype == "float16":
        dtype = "float32"
        input_x = tbe.cast_to(input_x, "float32")
    const_one = tvm.const(scalar_one, dtype=dtype)
    const_negative_one = tvm.const(scalar_negative_one, dtype=dtype)
    const_p = tvm.const(scalar_p, dtype=dtype)
    const_a = tvm.const(scalar_a, dtype=dtype)
    const_b = tvm.const(scalar_b, dtype=dtype)
    const_c = tvm.const(scalar_c, dtype=dtype)

    tensor_one = tbe.broadcast(const_one, input_x.shape, dtype)
    tensor_abs = tbe.vabs(input_x)
    erf_t_vmuls = tbe.vmuls(tensor_abs, const_p)
    erf_t_vadds = tbe.vadds(erf_t_vmuls, const_one)
    erf_data_t = tbe.vdiv(tensor_one, erf_t_vadds)
    erf_abs_square = tbe.vmul(tensor_abs, tensor_abs)
    erf_data_vmuls = tbe.vmuls(erf_abs_square, const_negative_one)
    if not tbe_platform.api_check_support("tbe.dsl.vexp", "float32") and dtype == "float32":
        data_div_16 = tbe.cast_to(erf_data_vmuls, "float16")
        erf_data_exp_16 = tbe.vexp(data_div_16)
        erf_data_exp = tbe.cast_to(erf_data_exp_16, dtype)
    else:
        erf_data_exp = tbe.vexp(erf_data_vmuls)
    erf_data_t_square = tbe.vmul(erf_data_t, erf_data_t)
    erf_data_t_cube = tbe.vmul(erf_data_t, erf_data_t_square)
    erf_t_vmuls = tbe.vmuls(erf_data_t, const_a)
    erf_t_square_vmuls = tbe.vmuls(erf_data_t_square, const_b)
    erf_t_cube_vmuls = tbe.vmuls(erf_data_t_cube, const_c)
    erf_square_vadd = tbe.vadd(erf_t_vmuls, erf_t_square_vmuls)
    erf_cube_vadd_ = tbe.vadd(erf_square_vadd, erf_t_cube_vmuls)
    erf_cube_vmuls = tbe.vmuls(erf_cube_vadd_, const_negative_one)
    erf_exp_vmul = tbe.vmul(erf_cube_vmuls, erf_data_exp)
    erf_exp_vadds = tbe.vadds(erf_exp_vmul, const_one)

    if util_soc_common.after_v200():
        scalar_zero = 0
        const_zero = tvm.const(scalar_zero, dtype=dtype)
        erf_exp_vadds_neg = tbe.vmuls(erf_exp_vadds, const_negative_one)
        erf_result = tbe.vcmpsel(input_x, const_zero, 'ge', erf_exp_vadds, erf_exp_vadds_neg)
    else:
        # `define a scaler, value = 32768`
        scalar_fp16_max = 32768
        # `define a scaler, value = 2**(-15)`
        scalar_fp16_min = 2 ** (-15)
        fp16_max = tvm.const(scalar_fp16_max, dtype=dtype)
        fp16_min = tvm.const(scalar_fp16_min, dtype=dtype)
        data_vmuls = tbe.vmuls(input_x, fp16_max)
        data_abs = tbe.vabs(data_vmuls)
        data_vadds = tbe.vadds(data_abs, fp16_min)
        data_div = tbe.vdiv(data_vmuls, data_vadds)
        if not tbe_platform.api_check_support("tbe.dsl.round", "float32") and dtype == "float32":
            data_div_16 = tbe.cast_to(data_div, "float16")
            data_round_16 = tbe.round(data_div_16)
            data_round = tbe.cast_to(data_round_16, dtype)
        else:
            data_round = tbe.round(data_div)
        tensor_sign = tbe.cast_to(data_round, dtype)
        erf_result = tbe.vmul(tensor_sign, erf_exp_vadds)

    if dtype != dtype_:
        erf_result = tbe.cast_to(erf_result, dtype_)
    return erf_result


@register_operator("Erf")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def erf(input_x, output_y, kernel_name="erf"):
    """
    algorithm: erf
    Computes the Gauss error function of `x` element-wise

    Parameters
    ----------
    input_x: dict
        shape and dtype of input, only support bfloat16, float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "erf"

    Returns
    -------
    None
    """
    dtype_input = input_x.get("dtype")
    dtype_input = dtype_input.lower()
    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(dtype_input, check_list, param_name="input_x")

    schedules, tensors = [], []
    ins = classify([input_x], OpPatternMode.ELEWISE)
    for (_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_x])
            data_input = tvm.placeholder(x_shape[0], dtype=dtype_input,
                                         name="data_input")
            res = erf_compute(data_input, output_y, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors,
              "bool_storage_as_1bit": False}
    tbe.build(schedules, config)
