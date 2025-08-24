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
smooth_l1_loss_v2
"""
import functools
from tbe.dsl.api import auto_schedule
from tbe.dsl.api import build
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=redefined-builtin,too-many-locals,too-many-arguments
@register_operator_compute("smooth_l1_loss_v2", op_mode="static", support_fusion=True)
def smooth_l1_loss_v2_compute(input_predict,
                              input_label,
                              sigma,
                              reduction):
    """calculating data

    Parameters
    ----------
    input_predict : TVM tensor
       the placeholder of input_predict
    input_label : TVM tensor
       the placeholder of input_label
    output_loss : dict
       dict of output_loss, include keys(shape and dtype)
    reduction: str
       type of result, default value is "mean"
    kernel_name : str
       kernel name, default value is "smooth_l1_loss_v2"

    Returns
    -------
    output tensor
    """
    ori_dtype = input_predict.dtype
    shape = input_predict.shape
    input_dtype = "float32"

    half_const_tensor = tbe.broadcast(tvm.const(0.5, dtype=input_dtype), input_predict.shape)
    one_const_tensor = tbe.broadcast(tvm.const(1.0, dtype=input_dtype), input_predict.shape)

    if ori_dtype == "float16":
        input_predict = tbe.cast_to(input_predict, input_dtype)
        input_label = tbe.cast_to(input_label, input_dtype)

    sigma_scalar = tvm.const(sigma, dtype=input_dtype)
    input_sub_res = tbe.vsub(input_predict, input_label)
    method_one_res = tbe.vmul(tbe.vmul(input_sub_res, half_const_tensor), input_sub_res)
    method_one_res = tbe.vmuls(method_one_res, 1 / sigma_scalar)

    predict_label_sub_abs = tbe.vabs(input_sub_res)
    method_two_res = tbe.vsub(predict_label_sub_abs, tbe.vmuls(half_const_tensor, sigma_scalar))

    is_method_one_res = tbe.vcmpsel(predict_label_sub_abs, sigma_scalar, 'lt', 1.0, 0.0)
    is_method_two_res = tbe.vsub(one_const_tensor, is_method_one_res)
    method_one_get_res = tbe.vmul(method_one_res, is_method_one_res)
    method_two_get_res = tbe.vmul(method_two_res, is_method_two_res)
    res = tbe.vadd(method_one_get_res, method_two_get_res)

    list = []
    if reduction == "sum":
        for i in range(len(shape)):
            list.append(i)
        res = tbe.sum(res, axis=list)
    elif reduction == "mean":
        for i in range(len(shape)):
            list.append(i)
        res = tbe.sum(res, axis=list)

        shape_val = functools.reduce(lambda x, y: x * y, shape)
        scalar = tvm.const(int(shape_val), dtype=input_dtype)
        res = tbe.vmuls(res, 1 / scalar)

    if ori_dtype == "float16":
        res = tbe.cast_to(res, "float16")

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def smooth_l1_loss_v2(predict,
                      label,
                      loss,
                      sigma=1.0,
                      reduction="mean",
                      kernel_name="smooth_l1_loss_v2"):
    """calculating data

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
        sigma, default value is 1
    reduction: str
        type of result, default value is "mean"
    kernel_name : str
        kernel name, default value is "smooth_l1_lossV2"

    Returns
    -------
    None
    """
    para_check.check_kernel_name(kernel_name)
    check_list = ("float16", "float32")

    shape_predict = predict.get("shape")
    dtype_predict = predict.get("dtype").lower()
    para_check.check_dtype(dtype_predict, check_list, param_name="predict")

    shape_label = label.get("shape")
    dtype_label = label.get("dtype").lower()
    para_check.check_dtype(dtype_label, check_list, param_name="label")

    shape_loss = label.get("shape")
    dtype_loss = loss.get("dtype").lower()
    para_check.check_dtype(dtype_loss, check_list, param_name="loss")

    para_check.check_shape(shape_predict, param_name="predict")
    para_check.check_shape(shape_label, param_name="label")
    para_check.check_shape(shape_loss, param_name="loss")

    check_list_reduction = ("none", "mean", "sum")
    reduction_type = reduction.lower()

    para_check.check_dtype(reduction_type, check_list_reduction, param_name="reduction")

    input_predict = tvm.placeholder(
        shape_predict, name="predict", dtype=dtype_predict)
    input_label = tvm.placeholder(
        shape_label, name="label", dtype=dtype_label)

    res = smooth_l1_loss_v2_compute(input_predict, input_label, sigma,
                                    reduction_type)

    with tvm.target.cce():
        sch = auto_schedule(res)

    config = {
        "name": kernel_name,
        "tensor_list": [input_predict, input_label, res]
    }

    build(sch, config)
