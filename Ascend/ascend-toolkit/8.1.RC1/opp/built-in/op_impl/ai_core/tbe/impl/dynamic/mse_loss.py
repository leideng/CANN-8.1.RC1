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
mse_loss
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_context
from impl.util.reduce_pattern_adapter import ReducePattern


# 'pylint: disable=invalid-name,unused-argument,too-many-locals,too-many-arguments
@register_operator_compute("MseLoss", op_mode="dynamic", support_fusion=False)
def mse_loss_compute(predict, label, y, axis, reduction='mean', kernel_name="mse_loss"):
    """
    calculating mse_loss
    :param predict: TVM tensor
                   the output of previous layer
    :param label: TVM tensor
                label
    :param axis: list
                axis
    :param reduction: str
                    reduce configuration parameter: mean/sum/none. Default: mean
    :param kernel_name: str
                    kernel name, default value is "mse_loss"
    :return:y
            when reduction=none:TVM tensor, output tensor
            when reduction=sum/mean, A Scalar
    """

    ori_dtype = predict.dtype
    trans_dtype = ori_dtype
    shape = shape_util.shape_to_list(predict.shape)

    if ori_dtype in ["float16", "bfloat16"]:
        predict = tbe.cast_to(predict, "float32")
        label = tbe.cast_to(label, "float32")
        trans_dtype = "float32"

    # calcu value:(predict_n - label_n)^2
    res = tbe.vsub(predict, label)
    result = tbe.vmul(res, res)

    calc_dtype = trans_dtype
    if reduction == "mean":
        reduce_elts = 1.0
        for i in shape:
            reduce_elts *= i
        if isinstance(reduce_elts, float):
            cof = float("nan") if reduce_elts == 0 else reduce_elts ** (-1)
            cof = tvm.const(cof, dtype=calc_dtype)
        else:
            cof = tbe.var("cof", dtype=calc_dtype)
            if calc_dtype == "float16":
                tbe.var("cof_empty", dtype=calc_dtype)
            tbe_context.get_context().add_compile_info("reduce_mean_cof_dtype", calc_dtype)

    if reduction != 'none':
        result = tbe.reduce_sum(result, axis=axis["value"], keepdims=False)
        if reduction == 'mean':
            result = tbe.vmuls(result, cof)

    if trans_dtype != ori_dtype:
        if ori_dtype == "float16":
            result = tbe.cast_to(result, ori_dtype)
        elif ori_dtype == "bfloat16":
            result = tbe.round(result, ori_dtype)

    return result


@register_operator("MseLoss")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_OUTPUT, para_check.OPTION_ATTR_STR,
                            para_check.KERNEL_NAME)
def mse_loss(predict, label, y, reduction='mean', kernel_name="mse_loss"):
    """
    calculating data
    sum = (predict_n - label_n)^2
    if  reduction == sum: res = sum output a scalal
    if reduction == mean: res == sum/total_number_of_tensor output a scalar
    if reduction == none: res == (predict_n - label_n)^2  output a tensor

    :param predict: dict
                    shape and dtype of tensor predict
    :param label: dict
                    shape and dtype of tensor real label,
                    should be same shape and dtype as predict
    :param y: dict
              shape and dtype of output, loss result after compute
    :param reduction: str
                      Specifies the reduction to apply to the output:'none' | 'mean' | 'sum'
                      Default: 'mean'
                      'none': no reduction will be applied,
                      'mean': the sum of the output will be divided by the number
                            of elements in the output
                      'sum': the output will be summed. Note: size_average and reduce
                           are in the process of being deprecated
                           and in the meantime, specifying either of those
                           two args will override reduction.
    :param kernel_name: str
                      kernel name, default value is "mse_loss"
    :return: none
    """
    predict_dtype = predict.get("dtype").lower()
    predict["rel_pos_to_reduce"] = "before"

    label_dtype = label.get("dtype").lower()
    label["rel_pos_to_reduce"] = "before"

    # check dtype
    dtype_list = ("bfloat16", "float16", "float32")
    para_check.check_dtype(predict_dtype, dtype_list)
    para_check.check_dtype(label_dtype, dtype_list)

    # check kernel_name
    para_check.check_kernel_name(kernel_name)

    if reduction not in ("mean", "sum", "none"):
        raise RuntimeError("reduction type should in mean/sum/none")

    tbe_context.get_context().add_compile_info("reduction", reduction)

    # gen reduce axis input dict
    input_axis = {"shape": [-1], "value": [], "rel_pos_to_reduce": "axis"}

    # gen extra_params for reduce pattern
    extra_params = dict()
    # set KEEP_DIMS flag
    extra_params.update(ReducePattern.KEEP_DIMS_FALSE)
    # set all reduce pattern
    extra_params.update(ReducePattern.REDUCE_MODE_REDUCE_ALL)

    schedules, tensors = [], []

    if reduction != "none":
        ins = classify([predict, label, input_axis], OpPatternMode.REDUCE, extra_params)
        for (_predict_shape, _label_shape, _axis) in ins:
            with tbe.compute():
                predict_shape, label_shape = shape_util.variable_shape([_predict_shape,
                                                                        _label_shape,
                                                                        _axis], op_mode="reduce")[0:2]
                data_predict = tvm.placeholder(predict_shape, name="data_predict",
                                               dtype=predict_dtype)
                data_label = tvm.placeholder(label_shape, name="data_label",
                                             dtype=label_dtype)
                res = mse_loss_compute(data_predict, data_label, y,
                                       _axis, reduction, kernel_name)
                tensors.append([data_predict, data_label, res])
            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)
    else:
        ins = classify([predict, label], OpPatternMode.ELEWISE)
        for (_predict_shape, _label_shape) in ins:
            with tbe.compute():
                predict_shape, label_shape = shape_util.variable_shape([_predict_shape,
                                                                        _label_shape])[0:2]
                data_predict = tvm.placeholder(predict_shape, name="data_predict",
                                               dtype=predict_dtype)
                data_label = tvm.placeholder(label_shape, name="data_label",
                                             dtype=label_dtype)
                res = mse_loss_compute(data_predict, data_label, y,
                                       [], reduction, kernel_name)
                tensors.append([data_predict, data_label, res])
            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)
    config = {"name": kernel_name, "tensor_list":tensors}
    tbe.build(schedules, config)
