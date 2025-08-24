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
dynamic sigmoid
"""
import math
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpImplMode
from impl.util.util_common import check_op_impl_mode
from impl.util.util_soc_common import after_v200


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant.
    """
    CONST_FP32_MAX = 3.4e+38
    CONST_FP16_MAX = 65504
    FLOAT_32 = "float32"


# 'pylint: disable=unused-argument,too-many-locals,invalid-name
def sigmoid_high_performance_compute(x, y, kernel_name="sigmoid"):
    """calculating data

    Calculation principle
    ---------------------
    `L(x) = 0.229270815*x - 0.0102459298*x^3 + 0.000207697530*x^5 + 0.5`
    `L(x) = a*x + b*x^3 + c*x^5 + d = x(a + x^2(b + cx^2)) + d`
    `sigmoid = max(0, min(1,L(x)))`

    Parameters
    ----------
    x : TVM tensor
        the placeholder of x
    y : dict
        dict of y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "sigmoid"

    Returns
    -------
    output tensor
    """
    dtype = x.dtype
    mul_support = tbe_platform.api_check_support("te.lang.cce.vmuls", "float32")
    if dtype == "float32" and not mul_support:
        error_manager_vector.raise_err_input_dtype_not_supported(kernel_name, 'x', ("float16",), dtype)

    const_num_c = tvm.const(0.000207697530, dtype=dtype)
    const_num_b = tvm.const(-0.0102459298, dtype=dtype)
    const_num_a = tvm.const(0.229270815, dtype=dtype)
    const_num_d = tvm.const(0.5, dtype=dtype)
    const_num_one = tvm.const(1, dtype=dtype)
    const_num_zero = tvm.const(0, dtype=dtype)
    # `x^2`
    tmp_x2 = tbe.vmul(x, x)
    # `cx^2`
    tmp_cx2 = tbe.vmuls(tmp_x2, const_num_c)
    # `b + cx^2`
    tmp_bsum = tbe.vadds(tmp_cx2, const_num_b)
    # `x^2(b + cx^2)`
    tmop_cx4 = tbe.vmul(tmp_x2, tmp_bsum)
    # `a + x^2(b + cx^2)`
    tmp_asum = tbe.vadds(tmop_cx4, const_num_a)
    # `x(a + x^2(b + cx^2))`
    tmp_cx5 = tbe.vmul(x, tmp_asum)
    # `x(a + x^2(b + cx^2)) + d`
    tmp_d = tbe.vadds(tmp_cx5, const_num_d)

    tmp_min = tbe.vmins(tmp_d, const_num_one)
    tmp_max = tbe.vmaxs(tmp_min, const_num_zero)

    return tmp_max


# 'pylint: disable=unused-argument,too-many-locals,invalid-name
@register_operator_compute("Sigmoid", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def sigmoid_compute(x, y, kernel_name="sigmoid", impl_mode=OpImplMode.HIGH_PRECISION):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of x
    y : dict
        dict of y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "sigmoid"
    impl_mode : str
        impl_mode, default value is "high_precision"

    Returns
    -------
    output tensor
    """
    data_input = x
    dtype = x.dtype
    origin_dtype = dtype
    soc_version = tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION)

    if soc_version in ("Ascend310P",) and impl_mode == "high_performance":
        return sigmoid_high_performance_compute(x, y, kernel_name)

    if tbe_platform.intrinsic_check_support("Intrinsic_vsigmoid", dtype):
        res = tbe.vsigmoid(x)
        return res

    cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    is_support_inf_nan = cce_product in ("Ascend910", "Ascend910B", "Ascend910_93")
    mul_support = tbe_platform.api_check_support("tbe.dsl.vmuls", "float32")
    exp_support = tbe_platform.api_check_support("tbe.dsl.vexp", "float32")
    if dtype == "float32" and not mul_support:
        error_manager_vector.raise_err_input_dtype_not_supported(kernel_name, 'x', ("float16",), dtype)

    if cce_product in ("Ascend910",):
        if dtype == "float32" and exp_support:
            ln_res = -math.log(Constant.CONST_FP32_MAX)
            ln_res = int(ln_res * 10) / 10
        else:
            ln_res = -math.log(Constant.CONST_FP16_MAX)
            ln_res = int(ln_res * 1000) / 1000
        data_input = tbe.vmaxs(data_input, ln_res)

    # only support VV mixed and single, not support CV mixed because can influence knowledge base
    cv_flag = "convolution_" not in data_input.op.tag and "matmul_gemv" not in data_input.op.tag\
              and "matmul_gevm" not in data_input.op.tag and "matmul" not in data_input.op.tag
    if cv_flag and dtype != Constant.FLOAT_32:
        if cce_product in ("Ascend310P",) or (cce_product in ("Ascend910B", "Ascend910_93") and \
        impl_mode == OpImplMode.HIGH_PRECISION):
            dtype = Constant.FLOAT_32
            data_input = tbe.cast_to(data_input, Constant.FLOAT_32)

    const_num_one = tvm.const(1, dtype=dtype)
    const_num_neg_one = tvm.const(-1, dtype=dtype)
    tmp_negative = tbe.vmuls(data_input, const_num_neg_one)
    if dtype == "float32" and not exp_support:
        tmp_negative_16 = tbe.cast_to(tmp_negative, "float16")
        tmp_exp_16 = tbe.vexp(tmp_negative_16)
        tmp_exp = tbe.cast_to(tmp_exp_16, dtype)
    else:
        tmp_exp = tbe.vexp(tmp_negative)
    tmp_sum = tbe.vadds(tmp_exp, const_num_one)
    if dtype == "float32" or is_support_inf_nan:
        inp_shape = tmp_sum.shape
        tensor_one = tbe.broadcast(tvm.const(1, dtype), inp_shape)
        tensor_one.op.attrs["broadcast_flag"] = "brdcast_for_vdiv"
        tmp_rec = tbe.vdiv(tensor_one, tmp_sum)
    else:
        tmp_rec = tbe.vrec(tmp_sum, impl_mode)
    
    if origin_dtype != dtype:
        tmp_rec = tbe.cast_to(tmp_rec, origin_dtype)

    return tmp_rec


@register_operator("Sigmoid")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def sigmoid(x, y, kernel_name="sigmoid", impl_mode=OpImplMode.HIGH_PRECISION):
    """
    calculating data

    Parameters
    ----------
    x : dict
        dict of x, include keys(shape and dtype)
    y : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "sigmoid"
    impl_mode : str
        impl_mode, default value is "high_precision"

    Returns
    -------
    None
    """
    check_op_impl_mode(impl_mode, [OpImplMode.HIGH_PERFORMANCE, OpImplMode.HIGH_PRECISION], kernel_name)

    dtype = x.get("dtype").lower()
    check_list = ("bfloat16", "float16", "float32")
    para_check.check_dtype(dtype, check_list, param_name="x")

    schedules, tensors = [], []
    ins = classify([x], OpPatternMode.ELEWISE)
    for (_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_x])
            data_input = tvm.placeholder(x_shape[0], dtype=dtype, name="dtype")
            res = sigmoid_compute(data_input, y, kernel_name, impl_mode)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {
        "name": kernel_name,
        "tensor_list": tensors,
        "bool_storage_as_1bit": False,
        "build_args": {
            "status_check": False
        }
    }
    tbe.build(schedules, config)
