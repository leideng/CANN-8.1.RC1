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
smooth_l1_loss_grad
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_attr_common import OpAttr
from impl.util.util_attr_common import get_attr_by_cls


class SmoothL1LossAttrInfo:
    """
    define SmoothL1Loss attr info
    """
    ATTR_SIGMA = OpAttr(0, "sigma", "Float", 1.0)


# 'pylint: disable=unused-argument,too-many-arguments
@register_operator_compute("SmoothL1LossGrad", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def smooth_l1_loss_grad_compute(predict, label, dout, gradient, sigma, kernel_name):
    """
    calculating data

    Parameters
    ----------
    predict : TVM tensor
        the placeholder of predict
    label : TVM tensor
        the placeholder of label
    dout : TVM tensor
        the placeholder of dout
    gradient : dict
        dict of gradient, include keys(shape and dtype)
    sigma : float
        sigma
    kernel_name : str
        kernel name, default value is "smooth_l1_loss_grad"

    Returns
    -------
    output tensor
    """
    dtype = predict.dtype
    sigma = get_attr_by_cls(sigma, SmoothL1LossAttrInfo.ATTR_SIGMA, dtype)

    out_sub = tbe.vsub(predict, label)
    # `out = sigma if out_sub > sigma`
    out_sub_one = tbe.vmins(out_sub, sigma)
    # `out = -sigma if out_sub < -sigma`
    out_sub_one_neg_one = tbe.vmaxs(out_sub_one, -sigma)
    out_sub_one_neg_one_sigma = tbe.vmuls(out_sub_one_neg_one, 1 / sigma)
    res = tbe.vmul(out_sub_one_neg_one_sigma, dout)

    return res


# 'pylint: disable=too-many-arguments,too-many-locals
@register_operator("SmoothL1LossGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def smooth_l1_loss_grad(predict, label, dout, gradient, sigma=1.0, kernel_name="smooth_l1_loss_grad"):
    """
    calculating data
    smooth = x/sigma        if -sigma < x < sigma
             1              if x > sigma
             -1             if x < -sigma
    out = smooth * dout

    Parameters
    ----------
    predict : dict
        shape and dtype of input
    label : dict
        shape and dtype of input
    dout : dict
        shape and dtype of input
    gradient : dict
        shape and dtype of output, should be same shape and type as predict
    sigma : float
        sigma
    kernel_name : str
        kernel name, default value is "smooth_l1_loss_grad"

    Returns
    -------
    None
    """

    predict_dtype = predict.get("dtype")
    input_dtype = predict_dtype.lower()
    label_dtype = label.get("dtype").lower()
    dout_dtype = dout.get("dtype").lower()

    check_list = ("bfloat16", "float16", "float32")
    para_check.check_dtype(input_dtype, check_list, param_name="predict")
    para_check.check_dtype(label_dtype, check_list, param_name="label")
    para_check.check_dtype(dout_dtype, check_list, param_name="dout")

    ins = classify([predict, label, dout], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_predict, _label, _dout) in ins:
        with tbe.compute():
            shape_predict, shape_label, shape_dout = shape_util.variable_shape([_predict, _label, _dout])
            tensor_predict = tvm.placeholder(shape_predict, name="tensor_predict", dtype=predict_dtype)
            tensor_label = tvm.placeholder(shape_label, name="tensor_label", dtype=label_dtype)
            tensor_dout = tvm.placeholder(shape_dout, name="tensor_dout", dtype=dout_dtype)

            res = smooth_l1_loss_grad_compute(tensor_predict, tensor_label, tensor_dout, gradient, sigma, kernel_name)
            tensors.append([tensor_predict, tensor_label, tensor_dout, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
