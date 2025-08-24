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
sign
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_platform
from impl.util.util_soc_common import after_v200

NEG_SCALAR_MIN_FP16 = -(2 ** (-24))
NEG_SCALAR_MIN_FP32 = -(2 ** (-126))
SCALAR_MIN_FP16 = 2 ** (-24)
SCALAR_MIN_FP32 = 2 ** (-126)
TYPE_FLOAT16 = "float16"


# 'pylint: disable=unused-argument,redefined-argument-from-local
@register_operator_compute("Sign", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def sign_compute(input_x, output_y, kernel_name="sign"):
    """
    compute for sign
    """

    dtype = input_x.dtype.lower()
    cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if after_v200() and dtype not in ("int32",):
        cmp_gt = tbe.vcmp(input_x, tvm.const(0.0, dtype), "gt")
        cmp_lt = tbe.vcmp(input_x, tvm.const(0.0, dtype), "lt")
        f16_gt = tbe.cast_to(cmp_gt, TYPE_FLOAT16)
        f16_lt = tbe.cast_to(cmp_lt, TYPE_FLOAT16)
        res = tbe.vsub(f16_gt, f16_lt)
        return tbe.cast_to(res, dtype)

    if cce_product in ("Ascend310P", "Ascend910B", "Ascend910_93") and dtype not in ("int32",):
        nan_res_tmp = tbe.vcmp(input_x, input_x, "eq", "bit")
        input_x = tbe.vsel(nan_res_tmp, input_x, tvm.const(0, input_x.dtype))
    if dtype == "float32":
        data_min = tvm.const(SCALAR_MIN_FP32, dtype=dtype)
        neg_data_min = tvm.const(NEG_SCALAR_MIN_FP32, dtype=dtype)
    elif dtype == TYPE_FLOAT16:
        data_min = tvm.const(SCALAR_MIN_FP16, dtype=dtype)
        neg_data_min = tvm.const(NEG_SCALAR_MIN_FP16, dtype=dtype)
    else:
        data_min = tvm.const(1, dtype=dtype)
        neg_data_min = tvm.const(-1, dtype=dtype)

    vmax = tbe.vmaxs(input_x, neg_data_min)
    vmin = tbe.vmins(vmax, data_min)
    if dtype == "float32":
        # max num of float32 is 2**126
        max_support_fp32 = tvm.const(2 ** 62, dtype=dtype)
        res_mul1 = tbe.vmuls(vmin, max_support_fp32)
        res_mul2 = tbe.vmuls(res_mul1, max_support_fp32)
        res = tbe.vmuls(res_mul2, tvm.const(2 ** 2, dtype=dtype))
    elif dtype == TYPE_FLOAT16:
        # max num of float16 is 2**24
        # but cce can only support 2**12, so use 12/12 to adaptor 24
        max_support_fp16 = tvm.const(2 ** 12, dtype=dtype)
        res_mul1 = tbe.vmuls(vmin, max_support_fp16)
        res = tbe.vmuls(res_mul1, max_support_fp16)
    else:
        res = vmin

    return res


@register_operator("Sign")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def sign(input_x, output_y, kernel_name="sign"):
    """
                                 x*32768
    algrithm: sign = round(-------------------------)
                            2 ** (-15) + |x*32768|

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support bfloat16, float16, float32, int32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is sign

    Returns
    -------
    None
    """
    dtype = input_x.get("dtype")
    check_list = ("bfloat16", "float16", "float32", "int32")
    input_dtype = dtype.lower()
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")
    schedules, tensors = [], []
    ins = classify([input_x], OpPatternMode.ELEWISE)
    for (input_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([input_x])
            data_input = tvm.placeholder(x_shape[0], dtype=input_dtype,
                                         name="data_input")
            res = sign_compute(data_input, output_y, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name,
              "tensor_list": tensors,
              "bool_storage_as_1bit": False}
    tbe.build(schedules, config)
