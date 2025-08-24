# Copyright 2020 Huawei Technologies Co., Ltd
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
tanh
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import OpImplMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_common import check_op_impl_mode
from impl.util import util_soc_common

NEG_SCALAR_MIN_FP32 = -(2 ** (-126))
SCALAR_MIN_FP32 = 2 ** (-126)
FP32_MIN_V2 = -8.8
FP32_MAX_V2 = 8.8
FP16_MIN_V2 = -4.6
FP16_MAX_V2 = 4.6
DOUBLE_X = 2


FP32_ZERO_055 = 0.55
FP32_ZERO_015 = 0.0157396831
FP32_ZERO_NEG_052 = -0.0523039624
FP32_ZERO_133 = 0.133152977
FP32_ZERO_NEG_333 = -0.333327681


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=too-many-locals,invalid-name,redefined-argument-from-local
def tanh_compute_v4(input_x):
    input_dtype_ori = input_x.dtype.lower()
    if input_dtype_ori == "float16":
        input_x = tbe.cast_to(input_x, "float32")
    input_dtype = input_x.dtype.lower()

    a = tbe.vabs(input_x)
    a2 = tbe.vmul(input_x, input_x)

    tmp_vmul = tbe.vmuls(a2, tvm.const(FP32_ZERO_015, input_dtype))
    tmp_vmul = tbe.vadds(tmp_vmul, tvm.const(FP32_ZERO_NEG_052, input_dtype))
    tmp_vmul = tbe.vmul(tmp_vmul, a2)
    tmp_vmul = tbe.vadds(tmp_vmul, tvm.const(FP32_ZERO_133, input_dtype))
    tmp_vmul = tbe.vmul(tmp_vmul, a2)
    tmp_vmul = tbe.vadds(tmp_vmul, tvm.const(FP32_ZERO_NEG_333, input_dtype))
    tmp_vmul = tbe.vmul(tmp_vmul, a2)
    tmp_vmul = tbe.vadds(tmp_vmul, tvm.const(1, input_dtype))
    s1 = tbe.vmul(tmp_vmul, input_x)

    input_x = tbe.vmins(input_x, tvm.const(20.0, input_dtype))
    e1 = tbe.vexp(tbe.vmuls(input_x, tvm.const(2.0, input_dtype)))
    s2 = tbe.vdiv(tbe.vadds(e1, tvm.const(-1.0, input_dtype)), tbe.vadds(e1, tvm.const(1.0, input_dtype)))

    cmp055 = tbe.vcmp(a, tvm.const(FP32_ZERO_055, input_dtype), "lt")
    res = tbe.vsel(cmp055, s1, s2)

    if input_dtype_ori == "float16":
        res = tbe.cast_to(res, input_dtype_ori)
    return res    


def _sign_compute(input_x):
    """
    computes the sign of the input data
    the dtype of input data is float32
    """
    dtype = input_x.dtype.lower()

    data_min = tvm.const(SCALAR_MIN_FP32, dtype=dtype)
    neg_data_min = tvm.const(NEG_SCALAR_MIN_FP32, dtype=dtype)

    vmax = tbe.vmaxs(input_x, neg_data_min)
    vmin = tbe.vmins(vmax, data_min)

    # max num of float32 is 2**126
    max_support_fp32 = tvm.const(2 ** 62, dtype=dtype)
    res_mul1 = tbe.vmuls(vmin, max_support_fp32)
    res_mul2 = tbe.vmuls(res_mul1, max_support_fp32)
    res = tbe.vmuls(res_mul2, tvm.const(2 ** 2, dtype=dtype))

    return res


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=too-many-locals,invalid-name,redefined-argument-from-local
def tanh_compute_v2(input_x):
    """
    algorithm: tanh
    calculating data's tanh,y= (e^(2x)-1)/(e^(2x)+1)

        For x > 0, to avoid overflow in exp(x), we reformulate the above

          (exp(2x) - 1) / (exp(2x) + 1)
        = (1 - exp(-2x)) / (1 + exp(-2x))

        = sign(x)* ((1 - exp(-2x)) / (1 + exp(-2x)))

        avoid value divide by zero, so abs(x) -> (abs(x) + min_value)


    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data

    Returns
    -------
    res : tvm.tensor
        the result of tanh
    """
    input_dtype = input_x.dtype

    has_improve_precision = False
    const_dtype = input_dtype
    if input_dtype == "float16" and \
            tbe_platform.api_check_support("te.lang.cce.vexp", "float32"):
        input_x = tbe.cast_to(input_x, "float32")
        has_improve_precision = True
        const_dtype = "float32"

    if const_dtype == "float16":
        min_fp_data = tvm.const(FP16_MIN_V2, dtype=const_dtype)
        max_fp_data = tvm.const(FP16_MAX_V2, dtype=const_dtype)
    else:
        min_fp_data = tvm.const(FP32_MIN_V2, dtype=const_dtype)
        max_fp_data = tvm.const(FP32_MAX_V2, dtype=const_dtype)

    # FP32
    input_x = tbe.vmins(input_x, max_fp_data)
    input_x = tbe.vmaxs(input_x, min_fp_data)

    power_val = tbe.vmuls(input_x, tvm.const(DOUBLE_X, const_dtype))
    exp_val = tbe.vexp(power_val)
    number = tbe.vadds(exp_val, tvm.const(-1., const_dtype))
    denom = tbe.vadds(exp_val, tvm.const(1., const_dtype))
    res = tbe.vdiv(number, denom)

    if has_improve_precision:
        res = tbe.cast_to(res, "float16")

    return res


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=too-many-locals,invalid-name,redefined-argument-from-local
@register_operator_compute("Tanh", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def tanh_compute(input_x, output_y, kernel_name="tanh", impl_mode=OpImplMode.HIGH_PRECISION):
    """
    algorithm: tanh
    calculating data's tanh,y= (e^(2x)-1)/(e^(2x)+1)

        For x > 0, to avoid overflow in exp(x), we reformulate the above

          (exp(2x) - 1) / (exp(2x) + 1)
        = (1 - exp(-2x)) / (1 + exp(-2x))

        = sign(x)* ((1 - exp(-2x)) / (1 + exp(-2x)))

        avoid value divide by zero, so abs(x) -> (abs(x) + min_value)


    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is tanh

    Returns
    -------
    res : tvm.tensor
        the result of tanh
    """
    input_dtype = input_x.dtype
    if tbe_platform.intrinsic_check_support("Intrinsic_vtanh", input_dtype):
        res = tbe.vtanh(input_x)

        return res

    cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    ### 910 高性能分支
    if cce_product in ("Ascend910B", "Ascend910_93") and impl_mode == OpImplMode.HIGH_PRECISION:
        return tanh_compute_v4(input_x)

    if util_soc_common.after_v200():
        return tanh_compute_v2(input_x)

    # positive min float32 value
    const_dtype = input_dtype
    min_fp_data = 2 ** (-126)

    # positive min float16 value
    if input_dtype == "float16":
        min_fp_data = 2 ** (-14)

    has_improve_precision = False

    if input_dtype == "float16" and \
            tbe_platform.api_check_support("te.lang.cce.vexp", "float32"):
        input_x = tbe.cast_to(input_x, "float32")
        has_improve_precision = True
        const_dtype = "float32"
        min_fp_data = 2 ** (-126)

    input_abs = tbe.vabs(input_x)
    power_val = tbe.vmuls(input_abs, tvm.const(-2, const_dtype))

    if input_dtype == "float32" and \
            not tbe_platform.api_check_support("te.lang.cce.vexp", "float32"):
        power_val = tbe.cast_to(power_val, "float16")

    exp_val = tbe.vexp(power_val)

    if input_dtype == "float32" and \
            not tbe_platform.api_check_support("te.lang.cce.vexp", "float32"):
        exp_val = tbe.cast_to(exp_val, "float32")

    if util_soc_common.after_v200() and input_x.dtype == "float32":
        # calculate sign
        sign_x = _sign_compute(input_x)
        const_one = tbe.broadcast(tvm.const(1, const_dtype), input_x.shape)
        down_val = tbe.vadds(exp_val, tvm.const(1, const_dtype))
        up_val = tbe.vsub(const_one, exp_val)
        res_div = tbe.vdiv(up_val, down_val)
        res = tbe.vmul(res_div, sign_x)
    else:
        up_val_tmp = tbe.vmul(exp_val, input_x)
        up_val = tbe.vsub(input_x, up_val_tmp)

        input_x_tmp = tbe.vadds(input_abs, min_fp_data)
        down_val_tmp = tbe.vadds(exp_val, tvm.const(1, const_dtype))
        down_val = tbe.vmul(down_val_tmp, input_x_tmp)

        res = tbe.vdiv(up_val, down_val)

    if has_improve_precision:
        res = tbe.cast_to(res, "float16")

    return res


@register_operator("Tanh")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def tanh(input_x, output_y, kernel_name="tanh", impl_mode=OpImplMode.HIGH_PRECISION):
    """
    algorithm: tanh
    calculating data's tanh,y= (e^(2x)-1)/(e^(2x)+1)

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support bfloat16, float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is tanh

    Returns
    -------
    None
    """
    check_op_impl_mode(impl_mode, [OpImplMode.HIGH_PERFORMANCE, OpImplMode.HIGH_PRECISION], kernel_name)

    input_dtype = input_x.get("dtype").lower()

    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")

    ins = classify([input_x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (input_x,) in ins:
        with tbe.compute():
            shape_x = shape_util.variable_shape([input_x])[0]
            data_input = tvm.placeholder(shape_x, name="data_input", dtype=input_dtype)
            res = tanh_compute(data_input, output_y, kernel_name, impl_mode)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
