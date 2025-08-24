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
mse_loss
"""
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=invalid-name,unused-argument,too-many-locals
@register_operator_compute("mse_loss", op_mode="static", support_fusion=True)
def mse_loss_compute(predict, label, reduction='mean', kernel_name="mse_loss"):
    """
    calculating mse_loss
    :param predict: TVM tensor
                   the output of previous layer
    :param label: TVM tensor
                label
    :param reduction: str
                    reduce configuration parameter: mean/sum/none. Default: mean
    :param kernel_name: str
                    kernel name, default value is "mse_loss"
    :return:y
            when reduction=none:TVM tensor, output tensor
            when reduction=sum/mean, A Scalar
    """
    ori_dtype = predict.dtype
    shape = shape_util.shape_to_list(predict.shape)

    if ori_dtype == "float16" and tbe_platform.api_check_support(
            "te.lang.cce.vmul", "float32"):
        predict = tbe.cast_to(predict, "float32")
        label = tbe.cast_to(label, "float32")

    # get total number of tensor
    reduce_elts = 1.0
    for i in shape:
        reduce_elts *= i
    cof = reduce_elts**(-1)

    # get total axis for reduce
    axis_d = []
    for i, _ in enumerate(shape):
        axis_d.append(i)
    axis_d = shape_util.axis_check(len(shape), axis_d)

    # calcu value:(predict_n - label_n)^2
    res = tbe.vsub(predict, label)
    res_sqr = tbe.vmul(res, res)

    y = 0.0

    if reduction == 'mean':
        # calcu mean
        y = tbe.sum(res_sqr, axis=axis_d, keepdims=False)
        y = tbe.vmuls(y, cof)
    elif reduction == 'sum':
        # calcu sum
        y = tbe.sum(res_sqr, axis=axis_d, keepdims=False)
    elif reduction == 'none':
        y = res_sqr

    if ori_dtype == "float16":
        y = tbe.cast_to(y, "float16")

    return y


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_ATTR_STR,
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

    predict_shape = predict.get("shape")
    predict_dtype = predict.get("dtype")
    predict_dtype_lower = predict_dtype.lower()

    label_shape = label.get("shape")
    label_dtype = label.get("dtype")
    label_dtype_lower = label_dtype.lower()

    # check dtype
    dtype_list = ("float16", "float32")
    para_check.check_dtype(predict_dtype, dtype_list)
    para_check.check_dtype(label_dtype, dtype_list)

    # check shape
    para_check.check_shape(predict_shape)
    para_check.check_shape(label_shape)

    # check kernel_name
    para_check.check_kernel_name(kernel_name)

    predict_size, _ = shape_util.refine_shape_axes(predict_shape, [])
    data_predict = tvm.placeholder(predict_size, dtype=predict_dtype_lower, name="data_predict")

    label_size, _ = shape_util.refine_shape_axes(label_shape, [])
    data_label = tvm.placeholder(label_size, dtype=label_dtype_lower, name="data_label")

    if predict_size != label_size:
        raise RuntimeError("predict tensor size don't match label tensor")
    if reduction not in ("mean", "sum", "none"):
        raise RuntimeError("reduction type should in mean/sum/none")

    res = mse_loss_compute(data_predict, data_label, reduction, kernel_name)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data_predict, data_label, res]}
    tbe.cce_build_code(schedule, config)
