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
MaskedSelectV2
"""
import te.lang.cce as tbe
from tbe import tvm
from te.utils import para_check
import te.platform as tbe_platform


@tbe_platform.fusion_manager.fusion_manager.register("masked_select_v2")
def masked_select_v2_compute(x, mask, y, kernel_name="masked_select_v2"):
    """
    Multiply input Tensor and  masked boolTensor
    The output of this op may be used in downstream sort task
    Use minimum value to fill in input x according to the position of False 
    in mask tensor to avoid affecting sort

    Parameters
    --------
    x : dict
        only support float16, float32
    mask : dict
        only support bool

    Returns
    -------
    Tensor
    """

    mask = tbe.cast_to(mask, "float16")
    x_shape = tbe.util.shape_to_list(x.shape)
    mask_shape = tbe.util.shape_to_list(mask.shape)
    if x_shape != mask_shape:
        mask = tbe.broadcast(mask, x_shape)
    target_dtype = x.dtype
    if target_dtype != "float16":
        mask = tbe.cast_to(mask, target_dtype)
        tensor_min = tbe.broadcast(tvm.const(-65504, target_dtype), x_shape)
    else:
        tensor_min = tbe.broadcast(tvm.const(-3.4e38, target_dtype), x_shape)
    tensor_one = tbe.broadcast(tvm.const(1, target_dtype), x_shape)
    mask = tbe.vsub(tensor_one, mask)
    mask_negative = tbe.vmul(tensor_min, mask)
    res = tbe.vadd(x, mask_negative)
    return res


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
        only support float16, float32
    mask : dict
        only support bool, int8

    Returns
    -------
    Tensor
    """

    # get shape and dtype of input tensor
    shape_x = x.get("shape")
    shape_mask = mask.get("shape")
    if shape_x[-1] == 1 and shape_mask[-1] == 1:
        shape_x = shape_x if len(shape_x) == 1 else shape_x[:-1]
        shape_mask = shape_mask if len(shape_mask) == 1 else shape_mask[:-1]

    x_data_type = x.get("dtype").lower()
    mask_data_type = mask.get("dtype").lower()
    if mask_data_type == "bool":
        mask_data_type = "int8"

    data_x = tvm.placeholder(shape_x, name="data_x", dtype=x_data_type)
    data_mask = tvm.placeholder(shape_mask, name="data_mask", dtype=mask_data_type)

    # call masked_select_compute to calculate
    res = masked_select_v2_compute(data_x, data_mask, y, kernel_name)

    # auto schedule
    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    # config of compile
    config = {"name": kernel_name,
              "tensor_list": (data_x, data_mask, res)}
    tbe.cce_build_code(schedule, config)
