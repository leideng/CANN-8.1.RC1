#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019 Huawei Technologies Co., Ltd
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
sigmoid_cross_entropy_with_logits_v2
"""

from tbe.dsl.api import auto_schedule
from tbe.dsl.api import build
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=redefined-builtin,too-many-arguments,too-many-locals,unused-argument
def op_select_format(predict, target, weight, pos_weight, loss, reduction="mean",
                     kernel_name="sigmoid_cross_entropy_with_logits_v2"):
    """op_select_format.

    Parameters
    ----------
    predict : dict
        shape and dtype of predict
    target : dict
        shape and dtype of target
    weight : dict
        shape and dtype of weight
    pos_weight : dict
        shape and dtype of pos_weight
    loss : dict
        shape and dtype of output, should be same shape and type as input
    reduction: str
        default value is "mean"
    kernel_name : str
        kernel name, default value is "sigmoid_cross_entropy_with_logits_v2"

    Returns
    -------
    None
    """
    cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        dtype = ["float16"]
    else:
        dtype = ["float16", "float"]

    dtype_length = len(dtype)
    format_list = ["ND", "NC1HWC0", "NDC1HWC0"]
    dtype = dtype * len(format_list)

    format = []
    for data_format in format_list:
        format = format + [data_format] * dtype_length

    dtype_total = ','.join(dtype)
    format_total = ','.join(format)
    dtype_output = ','.join(["float"] * len(dtype))

    input0 = gen_param(
        classify="input0", name="predict", datatype=dtype_total,
        format=format_total)
    input1 = gen_param(
        classify="input1", name="target", datatype=dtype_total,
        format=format_total)
    input2 = gen_param(
        classify="input2", name="weight", datatype=dtype_total,
        format=format_total)
    input3 = gen_param(
        classify="input3", name="pos_weight", datatype=dtype_total,
        format=format_total)
    output0 = gen_param(
        classify="output0", name="loss", datatype=dtype_output,
        format=format_total)

    param_list = [input0, input1, input2, input3, output0]

    param_dynamic_in_json = get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals
@register_operator_compute("sigmoid_cross_entropy_with_logits_v2", op_mode="static", support_fusion=True)
def sigmoid_cross_entropy_with_logits_v2_compute(predict,
                                                 target,
                                                 weight,
                                                 pos_weight,
                                                 loss,
                                                 reduction,
                                                 kernel_name):
    """calculating data

    Parameters
    ----------
    predict : TVM tensor
        the placeholder of predict
    target : TVM tensor
        the placeholder of target
    weight : TVM tensor
        the placeholder of weight
    pos_weigth : TVM tensor
        the placeholder of pos_weight
    loss : dict
        dict of loss, include keys(shape and dtype)
    reduction: str
        default value is "mean"
    kernel_name : str
        kernel name, default value is "sigmoid_cross_entropy_with_logits_v2"

    Returns
    -------
    output tensor
    """
    predict_dtype = predict.dtype

    is_support_float32 = tbe_platform.api_check_support(
        "tbe.dsl.vmul", "float32")

    if is_support_float32:
        if predict_dtype == "float16":
            predict = tbe.cast_to(predict, "float32")

        if target.dtype == "float16":
            target = tbe.cast_to(target, "float32")

        if weight is not None and weight.dtype == "float16":
            weight = tbe.cast_to(weight, "float32")

        if pos_weight is not None and pos_weight.dtype == "float16":
            pos_weight = tbe.cast_to(pos_weight, "float32")

    shape_predict = shape_util.shape_to_list(predict.shape)
    const_zero = tvm.const(0, dtype=predict.dtype)
    const_one = tvm.const(1, dtype=predict.dtype)
    const_zero_broadcast = tbe.broadcast(const_zero, shape_predict)
    const_one_broadcast = tbe.broadcast(const_one, shape_predict)

    # `info: max(-predict,0)
    reversed_predict = tbe.vsub(const_zero_broadcast, predict)
    max_predict_zero = tbe.vmaxs(reversed_predict, const_zero)

    # `info: max_val=max(-predict,0)`
    # `info: ln(1+exp(-x))=max_val+np.log(np.exp(-max_val)+np.exp(-predict-max_val)))`
    reversed_max_predict_zero = tbe.vsub(const_zero_broadcast, max_predict_zero)
    exp_reversed_max_predict_zero = tbe.vexp(reversed_max_predict_zero)
    sub_reversed_max_predict_zero = tbe.vsub(reversed_max_predict_zero, predict)
    exp_sub_reversed_max_predict_zero = tbe.vexp(sub_reversed_max_predict_zero)
    add_reversed_predict = tbe.vadd(exp_reversed_max_predict_zero, exp_sub_reversed_max_predict_zero)
    log_reversed_predict = tbe.vlog(add_reversed_predict)
    add_max_predict = tbe.vadd(log_reversed_predict, max_predict_zero)

    # `info: (1-target)*predict
    sub_target = tbe.vsub(const_one_broadcast, target)
    mul_predict_target = tbe.vmul(sub_target, predict)

    if pos_weight is not None:
        # `info: log_weight=(pos_weight - 1)*target+1`
        # `info: loss=(1-target)*predict+(log_weight*(max_val+np.log(np.exp(-max_val)+np.exp(-predict-max_val))))`
        pos_weight = tbe.broadcast(pos_weight, shape_predict)
        sub_pos_weight = tbe.vsub(pos_weight, const_one_broadcast)
        mul_pos_weight = tbe.vmul(sub_pos_weight, target)
        add_pos_weight = tbe.vadds(mul_pos_weight, const_one)
        mul_pos_weight_predict = tbe.vmul(add_pos_weight, add_max_predict)
        loss = tbe.vadd(mul_predict_target, mul_pos_weight_predict)
    else:
        # `info: loss=(1-target)*predict+max_val+np.log(np.exp(-max_val)+np.exp(-predict-max_val))`
        loss = tbe.vadd(mul_predict_target, add_max_predict)

    if weight is not None:
        weight = tbe.broadcast(weight, shape_predict)
        loss = tbe.vmul(loss, weight)

    return loss


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_STR,
                            para_check.KERNEL_NAME)
def sigmoid_cross_entropy_with_logits_v2(
        predict, target, weight, pos_weight, loss, reduction="mean",
        kernel_name="sigmoid_cross_entropy_with_logits_v2"):
    """calculating data

    Parameters
    ----------
    predict : dict
        shape and dtype of predict
    target : dict
        shape and dtype of target
    weight : dict
        shape and dtype of weight
    pos_weight : dict
        shape and dtype of pos_weight
    loss : dict
        shape and dtype of output, should be same shape and type as input
    reduction: str
        default value is "mean"
    kernel_name : str
        kernel name, default value is "sigmoid_cross_entropy_with_logits_v2"

    Returns
    -------
    None
    """
    check_list = ("float16", "float32")

    shape_predict = predict.get("shape")
    dtype_predict = predict.get("dtype").lower()
    para_check.check_shape(shape_predict, param_name="predict")
    para_check.check_dtype(dtype_predict, check_list, param_name="predict")

    shape_target = target.get("shape")
    dtype_target = target.get("dtype").lower()
    para_check.check_shape(shape_target, param_name="target")
    para_check.check_dtype(dtype_target, check_list, param_name="target")

    shape_util.compare_tensor_dict_key(predict, target, "shape")

    if reduction not in ("mean", "sum", "none"):
        error_manager_vector.raise_err_input_value_invalid("sigmoid_cross_entropy_with_logits_v2",
                                                           "reduction", "mean, sum, none", str(reduction))

    data_weight = None
    if weight is not None:
        shape_weight = weight.get("shape")
        dtype_weight = weight.get("dtype").lower()
        para_check.check_shape(shape_weight, param_name=weight)
        para_check.check_dtype(dtype_weight, check_list, param_name="weight")
        _, shape_weight, _ = shape_util.broadcast_shapes(shape_predict, shape_weight,
                                                         param_name_input1="predict",
                                                         param_name_input2="weight")
        data_weight = tvm.placeholder(shape_weight,
                                      name="data_weight",
                                      dtype=dtype_weight)

    data_pos_weight = None
    if pos_weight is not None:
        shape_pos_weight = pos_weight.get("shape")
        dtype_pos_weight = pos_weight.get("dtype").lower()
        para_check.check_shape(shape_pos_weight, param_name=weight)
        para_check.check_dtype(dtype_pos_weight, check_list, param_name="pos_weight")
        _, shape_pos_weight, _ = shape_util.broadcast_shapes(shape_predict, shape_pos_weight,
                                                             param_name_input1="predict",
                                                             param_name_input2="pos_weight")
        data_pos_weight = tvm.placeholder(shape_pos_weight,
                                          name="data_pos_weight",
                                          dtype=dtype_pos_weight)

    data_predict = tvm.placeholder(shape_predict,
                                   name="data_predict",
                                   dtype=dtype_predict)
    data_target = tvm.placeholder(shape_target,
                                  name="data_target",
                                  dtype=dtype_target)

    loss = sigmoid_cross_entropy_with_logits_v2_compute(data_predict,
                                                        data_target,
                                                        data_weight,
                                                        data_pos_weight,
                                                        loss,
                                                        reduction,
                                                        kernel_name)
    tensor_list = [data_predict, data_target]
    if data_weight is not None:
        tensor_list.append(data_weight)

    if data_pos_weight is not None:
        tensor_list.append(data_pos_weight)

    tensor_list.append(loss)

    with tvm.target.cce():
        sch = auto_schedule(loss)

    config = {"name": kernel_name,
              "tensor_list": tensor_list}

    build(sch, config)
