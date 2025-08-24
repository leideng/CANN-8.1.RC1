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
dynamic softmax_cross_entropy_with_logits
"""
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import classify
from impl.util.util_common import is_unknown_rank_input
import te.platform as tp


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant
    """
    # compute needed,scalar -1
    SCALAR_MINUS_ONE = -1
    DIM0_FP16_LOWER = 72
    DIM0_FP16_UPPER = 95
    DIM1_FP16_LOWER = 1009
    DIM1_FP16_UPPER = 10000
    DIM0_FP32_LOWER = 8
    DIM0_FP32_UPPER = 47
    DIM1_FP32_LOWER = 1249
    DIM1_FP32_UPPER = 10000


# 'pylint: disable=unused-argument, unused-variable, too-many-locals
def check_supported(input_features,
                    input_labels,
                    output_loss,
                    output_backprop,
                    kernel_name="softmax_cross_entropy_with_logits",
                    impl_mode="high_performance"):
    shape_features = shape_util.shape_to_list(input_features.get("shape"))
    shape_labels = shape_util.shape_to_list(input_labels.get("shape"))
    dtype_features = input_features.get("dtype").lower()
    # dynamic inputs
    for _, dim_val in enumerate(shape_features):
        if dim_val < 0:
            return True, ""
    for _, dim_val in enumerate(shape_labels):
        if dim_val < 0:
            return True, ""
    # dynamic for bfloat16
    if dtype_features == "bfloat16":
        return True, "goto dynamic operator"
    # static inputs with four dimensions
    if len(shape_features) > 2 or len(shape_labels) > 2:
        return False, "goto static operator"
    # static inputs of dtype float16, special process for multi-core
    if (dtype_features == "float16" and
        Constant.DIM0_FP16_LOWER <= shape_features[0] <= Constant.DIM0_FP16_UPPER and
        Constant.DIM1_FP16_LOWER <= shape_features[-1] <= Constant.DIM1_FP16_UPPER):
        return False, "goto static operator"
    # static inputs of dtype float32, special process for multi-core
    if (dtype_features == "float32" and
        Constant.DIM0_FP32_LOWER <= shape_features[0] <= Constant.DIM0_FP32_UPPER and
        Constant.DIM1_FP32_LOWER <= shape_features[-1] <= Constant.DIM1_FP32_UPPER):
        return False, "goto static operator"
    # static inputs, while the last axis is too large
    if list(shape_features) != list(shape_labels):
        shape_features, shape_labels, shape_broadcast = \
            shape_util.broadcast_shapes(shape_features, shape_labels, param_name_input1="input_features",
                                        param_name_input2="input_labels")
    else:
        shape_broadcast = shape_features
    not_920 = tp.cce_conf.intrinsic_check_support("Intrinsic_data_move_l12ub")
    current_csize_maximum_fp32 = 15360 if not_920 else 11264
    high_perf_csize_maximum_fp32 = 20000 if not_920 else 15000
    if current_csize_maximum_fp32 < shape_broadcast[1] < high_perf_csize_maximum_fp32 and \
            tp.api_check_support("te.lang.cce.vexp", "float32") and not_920:
        return False, "goto static operator"
    # norm static inputs
    return True, ""


# 'pylint: disable=unused-argument,too-many-locals
@register_operator_compute("SoftmaxCrossEntropyWithLogits", op_mode="dynamic", support_fusion=False, support_bfp16=True)
def softmax_cross_entropy_with_logits_compute(
        input_features,
        input_labels,
        output_loss,
        output_backprop,
        kernel_name="softmax_cross_entropy_with_logits",
        impl_mode="high_performance"):
    """
    Computes softmax cross entropy cost.
    softmax = e^(x-max) / ∑(e^(x-max))
    log(softmax) = (x-max) - log(∑e^(x-max))
    cross_entropy = -∑(y * log⁡(softmax))

    Parameters
    # ----------
    input_features: TVM tensor
        input tensor contains shape and dtype attributes.
        source data type support "float16", "float32".
    input_labels: TVM tensor
        input tensor contains shape and dtype attributes.
        Must have the same type as 'input_features'.
    output_loss: dict
        data of output.
        Must have the same type as 'input_features'.
    output_backprop: dict
        data of output.
        Must have the same type as 'input_features'.
    kernel_name: str
        kernel name, default value is "softmax_cross_entropy_with_logits"
    impl_mode: str
        specifying whether cast fp32 to fp16 before compute max.
        "high_precision" or "high_performance", defaults to "high_performance".

    Returns:
    res: TVM tensor
        output tensor. Has the same type as "input_features".
    """
    shape_features = shape_util.shape_to_list(input_features.shape)
    shape_labels = shape_util.shape_to_list(input_labels.shape)
    dtype = input_features.dtype.lower()
    input_features1 = input_features
    input_labels1 = input_labels

    if list(shape_features) != list(shape_labels):
        shape_features, shape_labels, shape_broadcast = \
            shape_util.broadcast_shapes(input_features.shape, input_labels.shape, param_name_input1="input_features",
                                        param_name_input2="input_labels")
        input_features_broad = tbe.broadcast(input_features, shape_broadcast, dtype)
        input_features_broad1 = tbe.broadcast(input_features1, shape_broadcast, dtype)
        input_features = input_features_broad
        input_features1 = input_features_broad1
        input_labels_broad = tbe.broadcast(input_labels, shape_broadcast, dtype)
        input_labels_broad1 = tbe.broadcast(input_labels1, shape_broadcast, dtype)
        input_labels = input_labels_broad
        input_labels1 = input_labels_broad1
    else:
        shape_broadcast = shape_features

    data_max = tbe.reduce_max(input_features, axis=-1, keepdims=True)
    data_max_broadcast = tbe.broadcast(data_max, shape_broadcast)
    has_improve_precision = False
    if dtype == "float16" and tbe_platform.api_check_support("te.lang.cce.vexp", "float32"):
        input_labels_cast_fp32 = tbe.cast_to(input_labels, "float32")
        input_labels = input_labels_cast_fp32

        input_features1_cast_fp32 = tbe.cast_to(input_features1, "float32")
        input_features1 = input_features1_cast_fp32
        input_labels1_cast_fp32 = tbe.cast_to(input_labels1, "float32")
        input_labels1 = input_labels1_cast_fp32

        has_improve_precision = True

    if has_improve_precision:
        data_max_broadcast = tbe.cast_to(data_max_broadcast, "float32")

    data_sub = tbe.vsub(input_features1, data_max_broadcast)
    data_exp = tbe.vexp(data_sub)
    data_sum = tbe.reduce_sum(data_exp, axis=-1, keepdims=True)
    data_sum_broadcast = tbe.broadcast(data_sum, shape_broadcast)
    data_div = tbe.vdiv(data_exp, data_sum_broadcast)
    data_log_tmp = tbe.vlog(data_sum_broadcast)
    data_log = tbe.vsub(data_sub, data_log_tmp)
    data_mul = tbe.vmul(data_log, input_labels)
    data_muls = tbe.vmuls(data_mul, Constant.SCALAR_MINUS_ONE)
    loss = tbe.reduce_sum(data_muls, axis=-1, keepdims=True)
    backprop = tbe.vsub(data_div, input_labels1)

    if has_improve_precision:
        loss = tbe.cast_to(loss, "float16")
        backprop = tbe.cast_to(backprop, "float16")

    res = [loss, backprop]
    return res


@register_operator_compute("SoftmaxCrossEntropyWithLogits", op_mode="dynamic", support_fusion=False, support_bfp16=True)
# 'pylint: disable=unused-argument,too-many-locals
def softmax_cross_entropy_with_logits_nchw_compute(
        input_features,
        input_labels,
        output_loss,
        output_backprop,
        kernel_name="softmax_cross_entropy_with_logits",
        impl_mode="high_performance"):
    """
    Computes softmax cross entropy cost.
    softmax = e^(x-max) / ∑(e^(x-max))
    log(softmax) = (x-max) - log(∑e^(x-max))
    cross_entropy = -∑(y * log⁡(softmax))

    Parameters
    # ----------
    input_features: TVM tensor
        input tensor contains shape and dtype attributes.
        source data type support "bfloat16".
    input_labels: TVM tensor
        input tensor contains shape and dtype attributes.
        Must have the same type as 'input_features'.
    output_loss: dict
        data of output.
        Must have the same type as 'input_features'.
    output_backprop: dict
        data of output.
        Must have the same type as 'input_features'.
    kernel_name: str
        kernel name, default value is "softmax_cross_entropy_with_logits"
    impl_mode: str
        defaults to "high_performance".

    Returns:
    res: TVM tensor
        output tensor. Has the same type as "input_features".
    """
    shape_features = shape_util.shape_to_list(input_features.shape)
    dtype = input_features.dtype.lower()
    shape_labels = shape_util.shape_to_list(input_labels.shape)
    
    if list(shape_features) != list(shape_labels):
        shape_features, shape_labels, shape_broadcast = \
            shape_util.broadcast_shapes(input_features.shape, input_labels.shape, param_name_input1="input_features",
                                        param_name_input2="input_labels")
        input_labels = tbe.broadcast(input_labels, shape_broadcast,
                                     dtype)    
        input_features = tbe.broadcast(input_features, shape_broadcast,
                                       dtype)
    else:
        shape_broadcast = shape_features

    data_max = tbe.reduce_max(input_features, axis=1, keepdims=True)
    data_max_broadcast = tbe.broadcast(data_max, shape_broadcast)

    data_sub = tbe.vsub(input_features, data_max_broadcast)
    data_exp = tbe.vexp(data_sub)
    data_sum = tbe.reduce_sum(data_exp, axis=1, keepdims=True)
    data_sum_broadcast = tbe.broadcast(data_sum, shape_broadcast)
    data_div = tbe.vdiv(data_exp, data_sum_broadcast)
    data_log_tmp = tbe.vlog(data_sum_broadcast)
    data_log = tbe.vsub(data_sub, data_log_tmp)
    data_mul = tbe.vmul(data_log, input_labels)
    data_muls = tbe.vmuls(data_mul, Constant.SCALAR_MINUS_ONE)
    loss = tbe.reduce_sum(data_muls, axis=1, keepdims=True)
    backprop = tbe.vsub(data_div, input_labels)

    res = [loss, backprop]
    return res


def input_broadcast(shape_features, shape_labels, input_features, input_labels):
    if len(shape_features) == 1 and len(shape_labels) == 2:
        if shape_features[0] == shape_labels[1]:
            shape_features = [1, shape_features[0]]
        elif shape_features[0] == shape_labels[0]:
            shape_features = [shape_features[0], 1]
        else:
            error_manager_vector.raise_err_specific_reson("softmax_cross_entropy_with_logits",
                                                "The features inputs can not be broadcasted to 2-dimensional")
        input_features['range'] = [[1, 1], input_features['range'][0]]
        input_features['shape'] = shape_features
    if len(shape_features) == 2 and len(shape_labels) == 1:
        if shape_labels[0] == shape_features[1]:
            shape_labels = [1, shape_labels[0]]
        elif shape_labels[0] == shape_features[0]:
            shape_labels = [shape_labels[0], 1]
        else:
            error_manager_vector.raise_err_specific_reson("softmax_cross_entropy_with_logits",
                                                "The labels inputs can not be broadcasted to 2-dimensional")
        input_labels['range'] = [[1, 1], input_labels['range'][0]]
        input_labels['shape'] = shape_labels

    return shape_features, shape_labels, input_features, input_labels  


# 'pylint: disable=unused-argument,too-many-locals,invalid-name
@register_operator("SoftmaxCrossEntropyWithLogits")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def softmax_cross_entropy_with_logits(
        input_features,
        input_labels,
        output_loss,
        output_backprop,
        kernel_name="softmax_cross_entropy_with_logits",
        impl_mode="high_performance"):
    """
    Computes softmax cross entropy cost.

    Parameters
    ----------
    input_features: dict
        input tensor contains shape and dtype attributes.
        source data type support "float16", "float32".
    input_labels: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'input_features'.
    output_loss: dict
        data of output.
        Must have the same type as 'input_features'.
    output_backprop: dict
        data of output.
        Must have the same type as 'input_features'.
    kernel_name: str
        kernel name, default value is "softmax_cross_entropy_with_logits"
    impl_mode: str
        specifying whether cast fp32 to fp16 before compute max.
        "high_precision" or "high_performance", defaults to "high_performance".

    Returns:
    None
    """
    shape_features = input_features.get("shape")
    shape_labels = input_labels.get("shape")

    shape_features, shape_labels, input_features, input_labels = input_broadcast(shape_features, shape_labels, 
                                                                        input_features, input_labels)
    if is_unknown_rank_input((input_features, input_labels)):
        shape_features = [-1, -1]
        shape_labels = [-1, -1]
        input_features["shape"] = shape_features
        input_labels["shape"] = shape_labels

        range_features = [(1, None), (1, None)]
        range_labels = [(1, None), (1, None)]
        input_features["range"] = range_features
        input_labels["range"] = range_labels

    shape_util.compare_tensor_dict_key(input_features, input_labels, "dtype")

    check_list = ("float16", "bfloat16", "float32")
    input_dtype = input_features.get("dtype").lower()
    para_check.check_dtype(input_dtype, check_list, param_name="input_features")

    shape_features = shape_util.scalar2tensor_one(shape_features)
    shape_labels = shape_util.scalar2tensor_one(shape_labels)

    input_features["shape"] = shape_features
    input_labels["shape"] = shape_labels

    shape_features_nchw_flag = False
    if len(shape_features) == 4:
        extra_params = {}
        shape_features_nchw_flag = True
    else:
        extra_params = {"input_shape_type": [1, 1], "same_input_shape_group": [[]],
                    "runtime_broadcast_axes": {0: [0, 1], 1: [0, 1]}}
    reduce_axis = [1, ]
    ins = classify([input_features, input_labels, reduce_axis], OpPatternMode.NORM, extra_params)

    if len(shape_features) == 1 and len(shape_labels) == 1:
        error_manager_vector.raise_err_specific_reson("softmax_cross_entropy_with_logits",
                                                      "The rank of two inputs can not be 1 at the same time")
    if (len(shape_features) > 2 or len(shape_labels) > 2) and input_dtype != "bfloat16":
        error_manager_vector.raise_err_specific_reson("softmax_cross_entropy_with_logits",
                                                      "features and labels must be either 2-dimensional,"
                                                      "or broadcasted to 2-dimensional")

    schedules, tensors = [], []
    for (x1, x2, _) in ins:
        with tbe.compute():
            shape_features, shape_labels = shape_util.variable_shape([x1, x2], op_mode="norm")
            data_features = tvm.placeholder(shape_features, dtype=input_dtype, name="data_features")
            data_labels = tvm.placeholder(shape_labels, dtype=input_dtype, name="data_labels")
            if shape_features_nchw_flag and input_dtype == "bfloat16":
                res = softmax_cross_entropy_with_logits_nchw_compute(data_features, data_labels, output_loss,
                                            output_backprop, kernel_name, impl_mode)
            else:
                res = softmax_cross_entropy_with_logits_compute(data_features, data_labels, output_loss,
                                                            output_backprop, kernel_name, impl_mode)
            tensor_list = [data_features, data_labels] + list(res)
            tensors.append(tensor_list)
        with tvm.target.cce():
            schedule = tbe.auto_schedule(res[:2])
        schedules.append(schedule)

    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
