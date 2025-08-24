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
shrink
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_attr_common import ShrinkAttrInfo
from impl.util.util_attr_common import get_attr_by_cls


# 'pylint: disable=superfluous-parens,unused-argument,too-many-locals
@register_operator_compute("Shrink", op_mode="dynamic", support_fusion=True)
def shrink_compute(input_x, output_y, lambd, bias, kernel_name="Shrink"):
    """
    calculating data

    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    output_y : dict
        dict of output_y, include keys(shape and dtype)
    lambda : float
        calculation threshold
    kernel_name : str
        kernel name, default value is "Shrink"

    Returns
    -------
    output tensor
    """
    ori_dtype = input_x.dtype
    shape = input_x.shape
    product = tbe_platform.api_check_support("tbe.dsl.vcmpsel", "float32")
    if product:
        input_dtype = "float32"
    else:
        input_dtype = "float16"

    if ori_dtype != input_dtype:
        input_x = tbe.cast_to(input_x, input_dtype)
    input_x_abs = tbe.vabs(input_x)
    bias_scalar = get_attr_by_cls(bias, ShrinkAttrInfo.ATTR_BIAS, input_dtype)
    lambd_scalar = get_attr_by_cls(lambd, ShrinkAttrInfo.ATTR_LAMBD, input_dtype)
    lambd_tensor = tbe.broadcast(lambd_scalar, shape)
    bias_tensor = tbe.broadcast(bias_scalar, shape)
    # use vmuls instrucion to avoid unsupported "-x" binary op case
    negative_bias_tensor = tbe.vmuls(bias_tensor, tvm.const(-1, input_dtype))
    zero_tensor = tbe.broadcast(tvm.const(0, input_dtype), shape)
    x_bias_tensor = tbe.vadd(input_x_abs, negative_bias_tensor)
    res1 = tbe.vcmpsel(input_x_abs, lambd_tensor, 'le', zero_tensor, x_bias_tensor)
    x_sign_tensor = tbe.vcmpsel(input_x, zero_tensor, 'le', -1, 1)
    result = tbe.vmul(x_sign_tensor, res1)
    if ori_dtype != input_dtype:
        result = tbe.cast_to(result, ori_dtype)
    return result


# 'pylint: disable=too-many-locals
@register_operator("Shrink")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def shrink(input_x, output_y, lambd=0.5, bias=0.0, kernel_name="Shrink"):
    """
    calculating data

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    lambda : float
        calculation threshold
    kernel_name : str
        kernel name, default value is "Shrink"

    Returns
    -------
    None
    """
    input_shape = input_x.get("shape")
    input_dtype = input_x.get("dtype").lower()

    para_check.check_shape(input_shape, param_name="input_x")
    check_tuple = ("float16", "float32")
    para_check.check_dtype(input_dtype, check_tuple, param_name="input_x")

    ins = classify([input_x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for shrink_input_x, in ins:
        with tbe.compute():
            shape_input_x, = shape_util.variable_shape([shrink_input_x])
            data_input = tvm.placeholder(shape_input_x, name="data_input", dtype=input_dtype)
            res = shrink_compute(data_input, output_y, lambd, bias, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            schedule = tbe.auto_schedule(res)
        schedules.append(schedule)

    config = {"name": kernel_name,
              "tensor_list": tensors}

    tbe.build(schedules, config)
