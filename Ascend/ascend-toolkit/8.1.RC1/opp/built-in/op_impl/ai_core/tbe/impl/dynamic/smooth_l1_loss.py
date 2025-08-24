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
smooth_l1_loss
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import register_operator
from impl.util.util_attr_common import OpAttr
from impl.util.util_attr_common import get_attr_by_cls
from impl.util.util_soc_common import after_v200


class SmoothL1LossAttrInfo:
    """
    define SmoothL1Loss attr info
    """
    ATTR_SIGMA = OpAttr(0, "sigma", "Float", 1.0)


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals
@register_operator_compute("SmoothL1Loss", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def smooth_l1_loss_compute(input_predict, input_label, output_loss, sigma, kernel_name="smooth_l1_loss"):
    """
    calculating data

    Parameters
    ----------
    input_predict : TVM tensor
        the placeholder of input_predict
    input_label : TVM tensor
        the placeholder of input_label
    output_loss : dict
        dict of output_loss, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "smooth_l1_loss"

    Returns
    -------
    output tensor
    """

    input_dtype = input_predict.dtype
    half_const = tvm.const(0.5, dtype=input_dtype)
    half_const_tensor = tbe.broadcast(half_const, input_predict.shape)
    one_const = tvm.const(1.0, dtype=input_dtype)
    one_const_tensor = tbe.broadcast(one_const, input_predict.shape)

    sigma_scalar = get_attr_by_cls(sigma, SmoothL1LossAttrInfo.ATTR_SIGMA, input_dtype)

    input_sub_res = tbe.vsub(input_predict, input_label)

    method_one_res = tbe.vmul(tbe.vmuls(input_sub_res, half_const), input_sub_res)
    method_one_res = tbe.vmuls(method_one_res, 1 / sigma_scalar)
    predict_label_sub_abs = tbe.vabs(input_sub_res)
    method_two_res = tbe.vsub(predict_label_sub_abs, tbe.vmuls(half_const_tensor, sigma_scalar))

    if after_v200():
        less_mask = tbe.vcmp(predict_label_sub_abs, sigma_scalar, 'lt', "bit")
        return tbe.vsel(less_mask, method_one_res, method_two_res)

    if not (tbe_platform.api_check_support("tbe.dsl.vcmpsel", "float32")) and \
            input_dtype == "float32":
        predict_label_sub_abs = tbe.cast_to(predict_label_sub_abs, "float16")
        sigma_scalar = tvm.const(sigma, "float16")
        is_method_one_res = tbe.vcmpsel(predict_label_sub_abs, sigma_scalar, 'lt', 1.0, 0.0)
        is_method_one_res = tbe.cast_to(is_method_one_res, "float32")
    else:
        is_method_one_res = tbe.vcmpsel(predict_label_sub_abs, sigma_scalar, 'lt', 1.0, 0.0)

    is_method_two_res = tbe.vsub(one_const_tensor, is_method_one_res)
    method_one_get_res = tbe.vmul(method_one_res, is_method_one_res)
    method_two_get_res = tbe.vmul(method_two_res, is_method_two_res)
    res = tbe.vadd(method_one_get_res, method_two_get_res)
    return res


@register_operator("SmoothL1Loss")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def smooth_l1_loss(predict, label, loss, sigma=1.0, kernel_name="smooth_l1_loss"):
    """
    calculating data

    Parameters
    ----------
    predict : dict
        shape and dtype of input
    label : dict
        shape and dtype of input
    loss : dict
        shape and dtype of output,
        should be same shape and type as input
    sigma: float
        sigma,default value is 1
    kernel_name : str
        kernel name, default value is "smooth_l1_loss"

    Returns
    -------
    None
    """

    check_list = ("bfloat16", "float16", "float32")
    shape_predict = predict.get("shape")
    dtype_predict = predict.get("dtype")
    input_predict_dtype = dtype_predict.lower()
    para_check.check_dtype(input_predict_dtype, check_list, param_name="predict")

    shape_label = label.get("shape")
    dtype_label = label.get("dtype")
    input_label_dtype = dtype_label.lower()
    dtype_loss = loss.get("dtype").lower()
    para_check.check_dtype(input_label_dtype, check_list, param_name="label")
    para_check.check_dtype(dtype_loss, check_list, param_name="loss")

    para_check.check_shape(shape_predict, param_name="predict")
    para_check.check_shape(shape_label, param_name="label")

    ins = classify([predict, label], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_predict, _label) in ins:
        with tbe.compute():
            predict_shape, label_shape = shape_util.variable_shape([_predict, _label])
            tensor_predict = tvm.placeholder(predict_shape, name="tensor_predict", dtype=input_predict_dtype)
            tensor_label = tvm.placeholder(label_shape, name="tensor_label", dtype=input_label_dtype)

            res = smooth_l1_loss_compute(tensor_predict, tensor_label, loss, sigma, kernel_name)
            tensors.append([tensor_predict, tensor_label, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
