#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
dynamic elu_grad_v2
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_attr_common import EluGradV2AttrInfo
from impl.util.util_attr_common import get_attr_by_cls


# 'pylint: disable=unused-argument,too-many-locals,too-many-arguments,invalid-name
@register_operator_compute("EluGradV2", op_mode="dynamic", support_fusion=True)
def elu_grad_v2_compute(grads, activations, y, alpha, scale, input_scale, is_result, kernel_name="elu_grad_v2"):
    """
    do calculating

    Parameters
    ----------
    grads: TVM tensor
        the placeholder of input grads
    activations: TVM tensor
        the placeholder of input activations
    y: TVM tensor
        shape and dtype of output y
    alpha: scalar parameter
        default value = 1.0
    scale: scalar parameter
        default value = 1.0
    input_scale: scalar parameter
        default value = 1.0
    is_result: bool parameter
        default value = False
    kernel_name: str
        kernel name, default value is "elu_grad_v2"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    input_dtype = grads.dtype
    input_shape = grads.shape
    
    # attr输入，转换成fp32
    scalar_alpha = get_attr_by_cls(alpha, EluGradV2AttrInfo.ATTR_ALPHA, "float32")
    scalar_scale = get_attr_by_cls(scale, EluGradV2AttrInfo.ATTR_SCALE, "float32")
    scalar_input_scale = get_attr_by_cls(input_scale, EluGradV2AttrInfo.ATTR_INPUT_SCALE, "float32")

    # 临时变量
    tensor_scale = tbe.broadcast(scalar_scale, input_shape)
    scalar_zero = tvm.const(0.0, "float32")
    tensor_zeros = tbe.broadcast(scalar_zero, input_shape)

    # 计算使用中间结果
    alpha_t_scale = scalar_alpha * scalar_scale

    # 如果输入数据类型不是fp32，则转换成fp32计算
    if input_dtype != 'float32':
        grads = tbe.cast_to(grads, "float32")
        activations = tbe.cast_to(activations, "float32")

    if is_result:
        temp_vcmp = tbe.vcmp(activations, tensor_zeros, 'le', mode='bool')
        tmp_result1 = tbe.vadds(activations, alpha_t_scale)
        tmp_result2 = tbe.vmuls(grads, scalar_input_scale)
        activations_smaller_than_zero = tbe.vmul(tmp_result2, tmp_result1)
        activations_bigger_than_zero = tbe.vmuls(grads, scalar_scale)
        if tbe_platform.api_check_support("tbe.dsl.vsel", "float32"):
            result = tbe.vsel(temp_vcmp, activations_smaller_than_zero, activations_bigger_than_zero)
        else:
            temp_vcmp = tbe.cast_to(temp_vcmp, "float16")
            activations_smaller_than_zero = tbe.cast_to(activations_smaller_than_zero, "float16")
            activations_bigger_than_zero = tbe.cast_to(activations_bigger_than_zero, "float16")
            result = tbe.vsel(temp_vcmp, activations_smaller_than_zero, activations_bigger_than_zero)
    else:
        temp_vcmp = tbe.vcmp(activations, tensor_zeros, 'le', mode='bool')
        tmp_result1 = tbe.vmuls(activations, scalar_input_scale)
        if tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
            temp_vexp = tbe.vexp(tmp_result1)
        else:
            tmp_result1 = tbe.cast_to(tmp_result1, "float16")
            temp_vexp = tbe.vexp(tmp_result1)
            temp_vexp = tbe.cast_to(temp_vexp, "float32")
        tmp_result2 = tbe.vmuls(grads, scalar_input_scale)
        tmp_result3 = tbe.vmuls(tmp_result2, alpha_t_scale)
        activations_smaller_than_zero = tbe.vmul(tmp_result3, temp_vexp)
        activations_bigger_than_zero = tbe.vmuls(grads, scalar_scale)
        if tbe_platform.api_check_support("tbe.dsl.vsel", "float32"):
            result = tbe.vsel(temp_vcmp, activations_smaller_than_zero, activations_bigger_than_zero)
        else:
            temp_vcmp = tbe.cast_to(temp_vcmp, "float16")
            activations_smaller_than_zero = tbe.cast_to(activations_smaller_than_zero, "float16")
            activations_bigger_than_zero = tbe.cast_to(activations_bigger_than_zero, "float16")
            result = tbe.vsel(temp_vcmp, activations_smaller_than_zero, activations_bigger_than_zero)

    if input_dtype == "bfloat16":
        result = tbe.round(result, "bfloat16")
    elif result.dtype != input_dtype:
        result = tbe.cast_to(result, input_dtype)
    return result


# 'pylint: disable=unused-argument
@register_operator("EluGradV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def elu_grad_v2(grads, activations, y, alpha=1.0, scale=1.0, input_scale=1.0, is_result=False, kernel_name="elu_grad_v2"):
    """
    do element-wise elu_backward operation

    if is_result :
        activations > 0 , y = grads * scale
        activations <= 0 , y = grads * input_scale * (activations + alpha * scale)
    else :
        activations > 0 , y = grads * scale
        activations <= 0 , y = grads * input_scale * alpha * scale * e ^ (activations * input_scale)

    Parameters
    ----------
    grads : dict
        shape and dtype of input grads, only support float16, float32, bfloat16
    activations : dict
        shape and dtype of input activations, only support float16, float32, bfloat16
    y: dict
        the same as input grads
    alpha: float
        default value is 1.0
    scale: float
        default value is 1.0
    input_scale: float
        default value is 1.0
    is_result: bool
        default value is False
    kernel_name : str
        cce kernel name, default value is elu_grad_v2

    Returns
    -------
    None
    """
    dtype_gradient = grads.get("dtype").lower()
    dtype_activation = activations.get("dtype").lower()
    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(dtype_gradient, check_list)
    para_check.check_dtype(dtype_activation, check_list)
    if dtype_gradient != dtype_activation:
        raise RuntimeError("All inputs must be the same dtype.")
    para_check.check_kernel_name(kernel_name)
  
    schedules, tensors = [], []
    ins = classify([grads, activations], OpPatternMode.ELEWISE)

    for (_grads, _activations) in ins:
        with tbe.compute():
            dtype = dtype_gradient
            shape_gradient, shape_activation = shape_util.variable_shape([_grads, _activations])
            data_gradient = tvm.placeholder(shape_gradient, dtype=dtype, name="data_gradient")
            data_activation = tvm.placeholder(shape_activation, dtype=dtype, name="data_activation")
            res = elu_grad_v2_compute(data_gradient, data_activation, y, alpha, scale, input_scale, is_result)
            tensors.append([data_gradient, data_activation, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {
        "name": kernel_name,
        "tensor_list": tensors,
        "bool_storage_as_1bit": False
    }
    tbe.build(schedules, config)
