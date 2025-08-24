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
mse_loss_grad
"""
import math
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_context


# 'pylint: disable=too-many-arguments,unused-argument,consider-using-in,len-as-condition,too-many-locals
@register_operator_compute("MseLossGrad", op_mode="dynamic", support_fusion=False, support_bfp16=True)
def mse_loss_grad_compute(predict, label, dout, grad, reduction="mean", kernel_name="mse_loss_grad"):
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
    grad : dict
        dict of gradient, include keys(shape and dtype)
    reduction : str
        reduce mode, can be 'mean','sum' or 'none'
    kernel_name : str
        kernel name, default value is "mse_loss_grad"

    Returns
    -------
    output tensor
    """
    ori_dtype = predict.dtype
    trans_dtype = ori_dtype
    if ori_dtype == "float16" and tbe_platform.api_check_support("te.lang.cce.vmul", "float32"):
        predict = tbe.cast_to(predict, "float32")
        label = tbe.cast_to(label, "float32")
        dout = tbe.cast_to(dout, "float32")
        trans_dtype = "float32"

    predict_shape = shape_util.shape_to_list(predict.shape)

    predict_shape, label_shape, dout_shape, max_shape = shape_util.unify_broadcast_shapes(
        [predict.shape, label.shape, dout.shape])

    predict = tbe.broadcast(predict, max_shape)
    label = tbe.broadcast(label, max_shape)
    dout = tbe.broadcast(dout, max_shape)

    calc_dtype = trans_dtype
    if reduction == "mean":
        reduce_elts = 1.0
        for i in predict_shape:
            reduce_elts *= i
        if isinstance(reduce_elts, float):
            cof = 2.0 if math.isclose(reduce_elts, 0.0) else reduce_elts ** (-1) * 2.0
            cof = tvm.const(cof, dtype=calc_dtype)
        else:
            cof = tbe.var("cof", dtype=calc_dtype)
            if calc_dtype == "float16":
                tbe.var("cof_empty", dtype=calc_dtype)
            tbe_context.get_context().add_compile_info("reduce_mean_cof_dtype", calc_dtype)
    else:
        cof = 2.0

    sub_res = tbe.vsub(predict, label)
    norm_grad = tbe.vmuls(sub_res, cof)
    grad_res = tbe.vmul(norm_grad, dout)

    if ori_dtype == "float16":
        grad_res = tbe.cast_to(grad_res, "float16")

    return grad_res


@register_operator("MseLossGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def mse_loss_grad(predict, label, dout, grad, reduction="mean", kernel_name="mse_loss_grad"):
    """
    calculating data

    Parameters
    ----------
    predict : dict
        shape and dtype of input
    label : dict
        shape and dtype of output, should be same shape and type as predict
    dout : dict
        shape and dtype of output, should be same shape and type as predict
    grad : dict
        shape and dtype of output, should be same shape and type as predict
    reduction : str
        reduce mode,can be 'mean','sum' or 'none'
    kernel_name : str
        kernel name, default value is "mse_loss_grad"

    Returns
    -------
    None
    """
    if reduction not in ("mean", "sum", "none"):
        rule_desc = "reduction type should in mean/sum/none"
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule_desc, "reduction", reduction)

    predict_dtype = predict.get("dtype").lower()
    label_dtype = label.get("dtype").lower()
    dout_dtype = dout.get("dtype").lower()

    shape_util.compare_tensor_dict_key(predict, label, "dtype")
    shape_util.compare_tensor_dict_key(predict, dout, "dtype")

    check_list = ("bfloat16", "float16", "float32")
    para_check.check_dtype(predict_dtype, check_list)
    para_check.check_dtype(label_dtype, check_list)
    para_check.check_dtype(dout_dtype, check_list)

    para_check.check_kernel_name(kernel_name)

    ins = classify([predict, label, dout], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_predict, _label, _dout) in ins:
        with tbe.compute():
            shape_predict, shape_label, shape_dout = shape_util.variable_shape([_predict, _label, _dout])
            data_predict = tvm.placeholder(shape_predict, name="data_predict", dtype=predict_dtype)
            data_label = tvm.placeholder(shape_label, name="data_label", dtype=label_dtype)
            data_dout = tvm.placeholder(shape_dout, name="data_dout", dtype=dout_dtype)
            res = mse_loss_grad_compute(data_predict, data_label, data_dout, grad, reduction, kernel_name)
            tensors.append([data_predict, data_label, data_dout, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}

    tbe.build(schedules, config)
