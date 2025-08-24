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
dynamic erfc
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
from impl.dynamic.erf import erf_compute
from impl.util import util_soc_common


def erfc_compute_v2(input_x):
    min_fp32 = -(2 ** (-149))
    input_dtype = input_x.dtype
    ori_dtype = input_x.dtype
    if tbe_platform.api_check_support("tbe.dsl.vexp", "float32") and input_dtype != "float32":
        input_x = tbe.cast_to(input_x, "float32")
        input_dtype = input_x.dtype
    
    input_x = tbe.vmins(input_x, 11.0)
    input_x = tbe.vmaxs(input_x, -11.0)
    input_x_abs = tbe.vabs(input_x)
    input_x_abs = tbe.vadds(input_x_abs, tvm.const(min_fp32, input_dtype))
    input_x_sign = tbe.vdiv(input_x, input_x_abs)

    num = tbe.vmuls(input_x_abs, tvm.const(0.1735313680e-7, dtype=input_dtype))
    num = tbe.vadds(num, tvm.const(-0.9856738394e-6, dtype=input_dtype))
    num = tbe.vmul(num, input_x_abs)
    num = tbe.vadds(num, tvm.const(0.2517003236e-4, dtype=input_dtype))
    num = tbe.vmul(num, input_x_abs)
    num = tbe.vadds(num, tvm.const(-0.3848015171e-3, dtype=input_dtype))
    num = tbe.vmul(num, input_x_abs)
    num = tbe.vadds(num, tvm.const(0.5681528564e0, dtype=input_dtype))
    num = tbe.vmul(num, input_x_abs)
    num = tbe.vadds(num, tvm.const(0.5245623129e1, dtype=input_dtype))
    num = tbe.vmul(num, input_x_abs)
    num = tbe.vadds(num, tvm.const(0.2107740710e2, dtype=input_dtype))
    num = tbe.vmul(num, input_x_abs)
    num = tbe.vadds(num, tvm.const(0.4212761755e2, dtype=input_dtype))
    num = tbe.vmul(num, input_x_abs)
    num = tbe.vadds(num, tvm.const(0.4380524149e2, dtype=input_dtype))
    
    den = tbe.vadds(input_x_abs, tvm.const(0.9349684299e1, dtype=input_dtype))
    den = tbe.vmul(den, input_x_abs)
    den = tbe.vadds(den, tvm.const(0.3756930664e2, dtype=input_dtype))
    den = tbe.vmul(den, input_x_abs)
    den = tbe.vadds(den, tvm.const(0.8058268949e2, dtype=input_dtype))
    den = tbe.vmul(den, input_x_abs)
    den = tbe.vadds(den, tvm.const(0.9155653738e2, dtype=input_dtype))
    den = tbe.vmul(den, input_x_abs)
    den = tbe.vadds(den, tvm.const(0.4380524152e2, dtype=input_dtype))
    res = tbe.vdiv(num, den)
    
    tmp = tbe.vmul(input_x_abs, input_x_abs)
    tmp = tbe.vmuls(tmp, tvm.const(-1.0, dtype=input_dtype))
    tmp = tbe.vexp(tmp)
    res = tbe.vmul(tmp, res)
    
    res = tbe.vmul(res, input_x_sign)
    tmp_1 = tbe.broadcast(tvm.const(1.0, dtype=input_dtype), input_x.shape)
    tmp = tbe.vsub(tmp_1, input_x_sign)
    erfc_result = tbe.vadd(res, tmp)
    
    if ori_dtype != input_dtype:
        erfc_result = tbe.cast_to(erfc_result, ori_dtype)
        
    return erfc_result
    

# 'pylint: disable=locally-disabled,unused-argument,too-many-locals,too-many-statements,invalid-name
@register_operator_compute("Erfc", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def erfc_compute(input_x, output_y, kernel_name="erfc"):
    """
    compute erfc

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        the dict of output_data, include keys(shape and dtype)
    kernel_name: str
        kernel name, default value is "erfc"

    Returns
    -------
    erfc_result: TVM tensor
        the =result of compute
    """
    cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if util_soc_common.after_v200() or cce_product == "Ascend310P":
        return erfc_compute_v2(input_x)
    
    # `define a scaler, value = 1`
    scaler_one = 1
    # `define a scaler, value = -1`
    scaler_negative_one = -1

    dtype = input_x.dtype

    const_one = tvm.const(scaler_one, dtype=dtype)
    const_negative_one = tvm.const(scaler_negative_one, dtype=dtype)

    erf_result = erf_compute(input_x, output_y)

    erf_result_neg = tbe.vmuls(erf_result, const_negative_one)
    erfc_result = tbe.vadds(erf_result_neg, const_one)

    return erfc_result


@register_operator("Erfc")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def erfc(input_x, output_y, kernel_name="erfc"):
    """
    algorithm: erfc
    Computes the Gauss error function of `x` element-wise

    Parameters
    ----------
    input_x: dict
        shape and dtype of input, only support bfloat16, float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "erfc"

    Returns
    -------
    None
    """
    dtype_input = input_x.get("dtype").lower()
    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(dtype_input, check_list, param_name="input_x")

    schedules, tensors = [], []
    ins = classify([input_x], OpPatternMode.ELEWISE)
    for (_input_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_input_x])
            data_input = tvm.placeholder(x_shape[0], dtype=dtype_input,
                                         name="data_input")
            res = erfc_compute(data_input, output_y, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors,
              "bool_storage_as_1bit": False}
    tbe.build(schedules, config)
