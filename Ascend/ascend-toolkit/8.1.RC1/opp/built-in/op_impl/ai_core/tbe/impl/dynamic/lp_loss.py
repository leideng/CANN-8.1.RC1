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
lp_loss
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.reduce_pattern_adapter import ReducePattern
from impl.util.util_common import is_unknown_rank_input


# 'pylint: disable=invalid-name,unused-argument,too-many-arguments,too-many-locals
def l1_loss_compute(predict, label, input_axis, reduction):
    """
    :param predict: TVM tensor
        the placeholder of predict
    :param label: TVM tensor
        the placeholder of label
    :param reduction: str
        reduce mode, can be 'mean','sum' or 'none'
    :return: output tensor
    """
    # float16 cast to float32
    predict_dtype = predict.dtype.lower()
    precision_dtype = "float32"
    if predict_dtype in ("float16", "bfloat16"):
        predict = tbe.cast_to(predict, precision_dtype)
        label = tbe.cast_to(label, precision_dtype)

    # calculate the result of loss = |predict-label|
    res = tbe.vabs(tbe.vsub(predict, label))
    trans_flag = False

    if reduction == "mean":
        if not tbe_platform.api_check_support("te.lang.cce.reduce_mean", "float32"):
            trans_flag = True
            res = tbe.cast_to(res, "float16")
        res = tbe.reduce_mean(res, input_axis["value"], keepdims=False)
    elif reduction == "sum":
        res = tbe.reduce_sum(res, input_axis["value"], keepdims=False)

    if (predict_dtype != precision_dtype and not trans_flag) or (predict_dtype == precision_dtype and trans_flag):
        if predict_dtype in ("bfloat16",):
            res = tbe.round(res, "bfloat16")
        else:
            res = tbe.cast_to(res, predict_dtype)

    return res


@register_operator_compute("LpLoss", op_mode="dynamic", support_fusion=True)
def lp_loss_compute(predict, label, axis, p, reduction):
    """
    :param predict: TVM tensor
        the placeholder of predict
    :param label: TVM tensor
        the placeholder of label
    :param p: int
        decides which loss to compute, now the p only can be 1 to compute l1_loss
    :param reduction: str
        reduce mode,can be 'mean','sum' or 'none'
    :return: output tensor
    """
    res = l1_loss_compute(predict, label, axis, reduction)
    return res


@register_operator("LpLoss")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_INT,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def lp_loss(predict, label, y, p, reduction="mean", kernel_name="lp_loss"):
    """
    :param predict: dict
        shape and dtype of input
    :param label: dict
        shape and dtype of label, should be same shape and type as predict
    :param y: dict
        shape and dtype of y, should be same shape and type as predict
    :param p: int
        decides which loss to compute, now the p only can be 1 to compute l1_loss
    :param reduction: str
        reduce mode, can be 'mean','sum' or 'none', default value is "mean"
    :param kernel_name: str
        kernel name, default value is "lp_loss"
    :return:
        None
    """
    predict_shape = predict.get("shape")
    predict_dtype = predict.get("dtype").lower()
    label_shape = label.get("shape")
    label_dtype = label.get("dtype").lower()

    reduction_list = ("none", "mean", "sum")
    dtype_list = ["bfloat16", "float16", "float32"]

    para_check.check_shape(predict_shape)
    para_check.check_dtype(predict_dtype, dtype_list)
    para_check.check_shape(label_shape)
    para_check.check_dtype(label_dtype, dtype_list)
    shape_util.compare_tensor_dict_key(predict, label, "dtype")
    para_check.check_dtype(reduction, reduction_list)

    schedules = []
    tensors = []

    if reduction == "none":
        ins = classify([predict, label], OpPatternMode.ELEWISE)
        for (_predict, _label) in ins:
            with tbe.compute():
                shape_predict, shape_label = shape_util.variable_shape([_predict, _label])
                predict_data = tvm.placeholder(shape_predict, dtype=predict_dtype, name="predict_data")
                label_data = tvm.placeholder(shape_label, dtype=label_dtype, name="label_data")
                res = lp_loss_compute(predict_data, label_data, None, p, reduction)
                tensors.append([predict_data, label_data, res])

            with tvm.target.cce():
                schedule = tbe.auto_schedule(res)
            schedules.append(schedule)
    else:
        predict["rel_pos_to_reduce"] = "before"
        label["rel_pos_to_reduce"] = "before"
        extra_params = dict()
        extra_params.update(ReducePattern.KEEP_DIMS_FALSE)
        extra_params.update(ReducePattern.REDUCE_MODE_REDUCE_ALL)
        if is_unknown_rank_input(predict):
            input_axis = {"shape": [-1, ], "value": [], "rel_pos_to_reduce": "axis"}
        else:
            axes = list(range(len(predict_shape)))
            input_axis = {"shape": [len(axes), ],
                          "value": axes,
                          "rel_pos_to_reduce": "axis"}

        ins = classify([predict, label, input_axis], OpPatternMode.REDUCE, extra_params)
        for (_predict, _label, _input_axis) in ins:
            with tbe.compute():
                shape_predict, shape_label = shape_util.variable_shape([_predict, _label, _input_axis],
                                                                       op_mode="reduce")[0:2]
                predict_data = tvm.placeholder(shape_predict, dtype=predict_dtype, name="predict_data")
                label_data = tvm.placeholder(shape_label, dtype=label_dtype, name="label_data")
                res = lp_loss_compute(predict_data, label_data, _input_axis, p, reduction)
                tensors.append([predict_data, label_data, res])

            with tvm.target.cce():
                schedule = tbe.auto_schedule(res)
            schedules.append(schedule)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
