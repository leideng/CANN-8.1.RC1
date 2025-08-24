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

import functools
import te.lang.cce as tbe
from impl.util.platform_adapter import build
from impl.util.platform_adapter import auto_schedule
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=too-many-arguments,unused-argument,consider-using-in,len-as-condition,too-many-locals
@register_operator_compute("mse_loss_grad", op_mode="static", support_fusion=True)
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
    dtype = predict.dtype.lower()
    if dtype == "float16":
        predict = tbe.cast_to(predict, "float32")
        label = tbe.cast_to(label, "float32")

    predict_shape = shape_util.shape_to_list(predict.shape)
    dout_shape = shape_util.shape_to_list(dout.shape)

    if predict_shape != dout_shape:
        dout = tbe.broadcast(dout, predict_shape)

    num = functools.reduce(lambda x, y: x * y, predict_shape)
    norm = 2.0 / num if reduction == "mean" else 2.0

    sub_res = tbe.vsub(predict, label)
    norm_grad = tbe.vmuls(sub_res, norm)
    grad_res = tbe.vmul(norm_grad, dout)
    if dtype == "float16":
        grad_res = tbe.cast_to(grad_res, dtype)

    return grad_res


def get_dout_shape(predict, dout, reduction="mean"):
    """
    get_dout_shape
    """
    predict_shape = predict.get("shape")
    dout_shape = dout.get("shape")

    if reduction == "mean" or reduction == "sum":
        if len(dout_shape) and not para_check.is_scalar(dout_shape):
            raise RuntimeError("when reduction is mean or sum, dout should be 0D or 1D tensor")
        dout_shape = tuple([1] * (len(predict_shape) - len(dout_shape))) + tuple(dout_shape)
    else:
        shape_util.compare_tensor_dict_key(predict, dout, "shape")

    return dout_shape


@para_check.check_op_params(dict, dict, dict, dict, str, str)
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

    predict_shape = predict.get("shape")
    predict_dtype = predict.get("dtype")
    label_shape = label.get("shape")
    input_dtype = predict_dtype.lower()
    label_dtype = label.get("dtype").lower()
    dout_dtype = dout.get("dtype").lower()
    dout_shape = get_dout_shape(predict, dout, reduction)

    shape_util.compare_tensor_dict_key(predict, label, "shape")
    shape_util.compare_tensor_dict_key(predict, label, "dtype")
    shape_util.compare_tensor_dict_key(predict, dout, "dtype")

    check_list = ("float16", "float32")
    para_check.check_dtype(input_dtype, check_list)
    para_check.check_dtype(label_dtype, check_list)
    para_check.check_dtype(dout_dtype, check_list)

    para_check.check_shape(predict_shape)
    para_check.check_shape(label_shape)
    para_check.check_shape(dout_shape)

    para_check.check_kernel_name(kernel_name)

    predict_input = tvm.placeholder(predict_shape, name="predict_input", dtype=input_dtype)
    label_input = tvm.placeholder(label_shape, name="label_input", dtype=input_dtype)
    dout_input = tvm.placeholder(dout_shape, name="dout_input", dtype=input_dtype)

    res = mse_loss_grad_compute(predict_input, label_input, dout_input, grad, reduction, kernel_name)

    with tvm.target.cce():
        schedule = auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [predict_input, label_input, dout_input, res]}

    build(schedule, config)
