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
masked_scale
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.util_attr_common import get_attr_by_cls
from impl.util.util_attr_common import MaskedScaleAttrInfo


# 'pylint: disable=unused-argument
@register_operator_compute("MaskedScale", op_mode="dynamic", support_fusion=True)
def masked_scale_compute(x, mask, y, value=1.0, kernel_name="masked_scale"):
    """
    function: compute of masked_scale
    Parameters
    ----------
    x: tensor
    mask: tensor
        shape must be same with x
    value: scaler
        dtype is float, default value is 1.0
    y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "masked_scale"

    Returns
    -------
    None
    """
    # mask_dst_dtype="float16" case cast_to only support int8->float16
    mask_dtype = mask.dtype
    if mask_dtype == "int8":
        mask = tbe.cast_to(mask, dtype="float16")
        mask = tbe.cast_to(mask, dtype="float32")
    elif mask_dtype == "float16":
        mask = tbe.cast_to(mask, dtype="float32")

    if x.dtype != "float32":
        x = tbe.cast_to(x, dtype="float32")

    res_vmul = tbe.vmul(x, mask)

    data_value = get_attr_by_cls(value, MaskedScaleAttrInfo.ATTR_VALUE, "float32")
    res = tbe.vmuls(res_vmul, data_value)

    if res.dtype != y.get("dtype"):
        res = tbe.cast_to(res, dtype=y.get("dtype"))

    return res


# 'pylint: disable=unused-argument,too-many-locals
@register_operator("MaskedScale")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def masked_scale(x, mask, y, value=1.0, kernel_name="masked_scale"):
    """
    algorithm: masked_scale
    calculating data's reciprocal, y = x * mask * value

    Parameters
    ----------
    x: dict
        shape and dtype of input, only support float16, float32
    mask: dict
        shape and dtype of input, only support int8
    value: scaler
        dtype is float, default value is 1.0
    y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "masked_scale"

    Returns
    -------
    None
    """
    x_dtype = x.get("dtype")

    mask_dtype = mask.get("dtype")

    ins = classify([x, mask], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_x, _mask) in ins:
        with tbe.compute():
            x_shape, mask_shape = shape_util.variable_shape([_x, _mask])
            data_x = tvm.placeholder(x_shape, name="data_x", dtype=x_dtype)
            data_mask = tvm.placeholder(mask_shape, name="data_mask", dtype=mask_dtype)
            res = masked_scale_compute(data_x, data_mask, y, value, kernel_name)
            tensors.append((data_x, data_mask, res))

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
