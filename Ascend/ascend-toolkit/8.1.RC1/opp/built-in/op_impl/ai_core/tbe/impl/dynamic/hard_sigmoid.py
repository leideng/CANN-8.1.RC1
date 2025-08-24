# Copyright (C) Huawei Technologies Co., Ltd 2022-2022. All rights reserved.
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
hard_sigmoid
"""
from ..util.platform_adapter import tbe
from ..util.platform_adapter import para_check
from ..util.platform_adapter import shape_util
from ..util.platform_adapter import tbe_platform
from ..util.platform_adapter import tvm
from ..util.platform_adapter import register_operator
from ..util.platform_adapter import register_operator_compute
from ..util.platform_adapter import classify
from ..util.platform_adapter import OpPatternMode
from ..util.util_attr_common import HardSigmoidAttrInfo
from ..util.util_attr_common import get_attr_by_cls


# 'pylint: disable=unused-argument,too-many-locals
@register_operator_compute("HardSigmoid", op_mode="dynamic", support_fusion=True)
def hard_sigmoid_compute(input_x, output_y, alpha, beta, kernel_name="hard_sigmoid"):
    """
    calculating data

    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    output_y : dict
        dict of output_y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "hard_sigmoid"

    Return Result
    -------
    output tensor
    """

    input_dtype = input_x.dtype.lower()
    if input_dtype in ("bfloat16",):
        input_x = tbe.cast_to(input_x, "float32")
    
    dtype = input_x.dtype
    mul_support_fp32 = tbe_platform.api_check_support("te.lang.cce.vmuls", "float32")
    if dtype != "float32" and mul_support_fp32:
        input_x = tbe.cast_to(input_x, "float32")
    elif dtype != "float16" and not mul_support_fp32:
        cast_support_f322f16 = tbe_platform.api_check_support("te.lang.cce.cast_to", "f322f16")
        cast_support_s322f16 = tbe_platform.api_check_support("te.lang.cce.cast_to", "s322f16")
        if cast_support_f322f16 and dtype == "float32" or  cast_support_s322f16 and dtype == "int32":
            input_x = tbe.cast_to(input_x, "float16")
        else:
            raise RuntimeError("Type of input x must be float16")

    alpha_scalar = get_attr_by_cls(alpha, HardSigmoidAttrInfo.ATTR_ALPHA, input_x.dtype)
    beta_scalar = get_attr_by_cls(beta, HardSigmoidAttrInfo.ATTR_BETA, input_x.dtype)
    alpha_x = tbe.vmuls(input_x, alpha_scalar)
    alpha_x_beta = tbe.vadds(alpha_x, beta_scalar)
    result = tbe.vmaxs(alpha_x_beta, tvm.const(0, dtype=dtype))
    result = tbe.vmins(result, tvm.const(1, dtype=dtype))
    if dtype != result.dtype:
        result = tbe.cast_to(result, dtype)
    if input_dtype in ("bfloat16",):
        result = tbe.round(result, "bfloat16")
    return result


@register_operator("HardSigmoid")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def hard_sigmoid(input_x, output_y, alpha=0.16666666, beta=0.5, kernel_name="hard_sigmoid"):
    """
    calculating data

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "hard_sigmoid"

    Returns
    -------
    None
    """
    input_shape = input_x.get("shape")
    input_dtype = input_x.get("dtype").lower()
    para_check.check_shape(input_shape, param_name="input_x")
    check_tuple = ("bfloat16", "int32", "float16", "float32")
    para_check.check_dtype(input_dtype, check_tuple, param_name="input_x")

    schedules, tensors = [], []
    ins = classify([input_x], OpPatternMode.ELEWISE)

    for (_x, ) in ins:
        with tbe.compute():
            x_shape,  = shape_util.variable_shape([_x])
            data_input = tvm.placeholder(x_shape, dtype=input_dtype, name="data_input")
            res = hard_sigmoid_compute(data_input, output_y=output_y, alpha=alpha, beta=beta, kernel_name=kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {
        "name": kernel_name,
        "tensor_list": tensors,
        "bool_storage_as_1bit": False
    }
    tbe.build(schedules, config)
