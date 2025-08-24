# Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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
MaskedSelectV2
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


@register_operator_compute("MaskedSelectV2", op_mode="dynamic", support_fusion=True)
# 'pylint: disable=unused-argument
def masked_select_v2_compute(x, mask, y, kernel_name="masked_select_v2"):
    """
    Multiply input Tensor and  masked boolTensor
    The output of this op may be used in downstream sort task
    Use minimum value to fill in input x according to the position of False 
    in mask tensor to avoid affecting sort

    Parameters
    --------
    x : TVM Tensor
        the placeholder of input x, only support float16, float32
    mask : TVM Tensor
        the placeholder of, only support bool, int8
    y: TVM Tensor
        the placeholder of output y

    Returns
    -------
    res: TVM Tensor
        the result of computation
    """

    target_dtype = x.dtype
    mask = tbe.cast_to(mask, target_dtype)
    x_shape, mask_shape, shape_max = shape_util.broadcast_shapes(x.shape, mask.shape)
    if x_shape != mask_shape:
        x = tbe.broadcast(x, shape_max)
        mask = tbe.broadcast(mask, shape_max)
    tensor_one = tbe.broadcast(tvm.const(1.0, target_dtype), shape_max)
    if target_dtype == "float16":
        tensor_min = tbe.broadcast(tvm.const(-65504, target_dtype), shape_max)
    else:
        tensor_min = tbe.broadcast(tvm.const(-3.4e38, target_dtype), shape_max)
    mask = tbe.vsub(tensor_one, mask)
    mask_negative = tbe.vmul(tensor_min, mask)
    res = tbe.vadd(x, mask_negative)
    return res


@register_operator("MaskedSelectV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def masked_select_v2(x, mask, y, kernel_name="masked_select_v2"):
    """
    Multiply input Tensor and  masked boolTensor
    The output of this op may be used in downstream sort task
    Use minimum value to fill in input x according to the position of False 
    in mask tensor to avoid affecting sort

    Parameters
    --------
    x : dict
        shape and dtype of input x, only support float16, float32
    mask : dict
        shape and dtype of mask, only support bool, int8
    y : dict
        shape and dtype of output y

    Returns
    -------
    None
    """

    dtype_x = x.get("dtype").lower()
    dtype_mask = mask.get("dtype").lower()
    check_list_x = ("float16", "float32")
    check_list_mask = ("bool", "int8")
    para_check.check_dtype(dtype_x, check_list_x)
    para_check.check_dtype(dtype_mask, check_list_mask)
    if dtype_mask == "bool":
        dtype_mask = "int8"
    schedules, tensors = [], []
    ins = classify([x, mask], OpPatternMode.ELEWISE_WITH_BROADCAST)
    for (input_x, input_mask) in ins:
        with tbe.compute():
            shape_x, shape_mask = shape_util.variable_shape([input_x, input_mask])
            if shape_x[-1] == 1 and shape_mask[-1] == 1:
                shape_x = shape_x if len(shape_x) == 1 else shape_x[:-1]
                shape_mask = shape_mask if len(shape_mask) == 1 else shape_mask[:-1]
            data_x = tvm.placeholder(shape_x, name="data_x", dtype=dtype_x)
            data_mask = tvm.placeholder(shape_mask, name="data_mask", dtype=dtype_mask)
            res = masked_select_v2_compute(data_x, data_mask, y, kernel_name)
            tensors.append([data_x, data_mask, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
