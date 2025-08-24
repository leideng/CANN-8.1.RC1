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
threshold_v2
"""

from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_platform


# 'pylint: disable=unused-argument,invalid-name
@register_operator_compute("ThresholdV2", op_mode="dynamic", support_fusion=True)
def threshold_v2_compute(x, threshold, value, kernel_name="threshold_v2"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        input tensor of x
    threshold : TVM tensor
        input tensor of threshold
    value : TVM tensor
        input tensor of value
    y : dict
        dict of y
    kernel_name : str
        kenel name, default value is "thershold_v2"

    Returns
    -------
    res: TVM tensor
        the result of threshold_v2_compute
    """
    input_dtype = x.dtype
    compatible_dtype = x.dtype
    shape = x.shape

    if input_dtype in ("int8", "uint8", "int32", "bfloat16"):
        if tbe_platform.api_check_support("tbe.dsl.vcmpsel", "float32") or input_dtype == "bfloat16":
            compatible_dtype = "float32"
        else:
            compatible_dtype = "float16"

        x = tbe.cast_to(x, compatible_dtype)
        threshold = tbe.cast_to(threshold, compatible_dtype)
        if value is not None:
            value = tbe.cast_to(value, compatible_dtype)

    threshold = tbe.broadcast(threshold, shape)
    if value is None:
        value = tbe.broadcast(tvm.const(0, compatible_dtype), shape)
    else:
        value = tbe.broadcast(value, shape)
    
    data_res = None
    if input_dtype == "int64":
        temp_res = tbe.vcmp(x, threshold, operation='le', mode='bool')
        temp_res = tbe.cast_to(temp_res, "bool")
        data_res = tbe.vsel(temp_res, value, x)
        return data_res

    if tbe_platform.api_check_support("tik.vcopy"):
        condition = tbe.vcmp(x, threshold, 'le', 'bit')
        data_res = tbe.vsel(condition, value, x)
    else:
        data_res = tbe.vcmpsel(x, threshold, operation='le', slhs=value, srhs=x)
    
    if input_dtype == "bfloat16":
        data_res = tbe.round(data_res, "bfloat16")
        return data_res

    if input_dtype != compatible_dtype:
        data_res = tbe.cast_to(data_res, input_dtype)

    return data_res


# 'pylint: disable=unused-argument,too-many-locals,invalid-name
@register_operator("ThresholdV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def threshold_v2(x, threshold, value, y, kernel_name="threshold_v2"):
    """
    Thresholds each element of the input Tensor
    y = (x > threshold) ? x : value

    Parameters
    ----------
    x : dict
        shape and dtype of input
    threshold : dict
        shape and dtype of the value to threshold at
    value : dict
        shape and dtype of the value to replace with, default value is 0
    y : dict
        shape and dtype of output, should be the same shape and dtype as input
    kernel_name : str
        kernel name, default value is "threshold_v2"

    Returns
    -------
    output tensor
    """
    dtype_x = x.get("dtype").lower()
    check_list = ("float16", "float32", "int8", "int32", "uint8", "int64", "bfloat16")
    para_check.check_dtype(dtype_x, check_list)

    data_threshold = tvm.placeholder((1,), dtype=dtype_x, name="data_threshold")
    data_value = None
    if value is not None:
        data_value = tvm.placeholder((1,), dtype=dtype_x, name="data_value")

    ins = classify([x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for(_x,) in ins:
        with tbe.compute():
            shape_x = shape_util.variable_shape([_x])[0]
            data_x = tvm.placeholder(shape_x, name="data_x", dtype=dtype_x)
            res = threshold_v2_compute(data_x, data_threshold, data_value, kernel_name)
            if value is not None:
                tensors.append([data_x, data_threshold, data_value, res])
            else:
                tensors.append([data_x, data_threshold, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}

    tbe.build(schedules, config)
