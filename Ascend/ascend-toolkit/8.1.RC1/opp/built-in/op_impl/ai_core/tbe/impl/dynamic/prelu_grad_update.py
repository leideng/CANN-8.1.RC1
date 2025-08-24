# Copyright 2023 Huawei Technologies Co., Ltd
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
prelu_grad_update
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import OpPatternMode
from impl.util import util_select_op_base
from impl.util import util_soc_common
from impl.util.util_common import is_unknown_rank_input
import tbe.dsl as dsl


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,too-many-locals,invalid-name
def op_select_format(grads, features, weights, dx, update, kernel_name="prelu_grad_update"):
    """
    Parameters
    ----------
    grads : TVM tensor
        input tensor of grad
    features : TVM tensor
        input tensor of prelu output
    weights : TVM tensor
        input tensor of prelu output
    dx : dict
        dx output dict of prelu_grad_update
    update : dict
        update output dict of prelu_grad_update
    kernel_name : str
        kernel name, default value is "prelu_grad_update"
    Returns
    -------
    None.
    """

    weights_shape = weights.get("ori_shape")
    bf16_support = tbe_platform.intrinsic_check_support("Intrinsic_vconv", "bf162f32")

    format_grads = ["ND", "ND"]
    format_weight = ["ND", "ND"]
    dtype_base = ["float", "float16"]
    valid_format_category = 1
    if bf16_support:
        dtype_base.append("bfloat16")
        format_grads.append("ND")
        format_weight.append("ND")
        
    format_grads = format_grads + ["NCHW"] * len(dtype_base)
    format_weight = format_weight + ["ND"] * len(dtype_base)
    valid_format_category = valid_format_category + 1

    if len(weights_shape) == 1 and weights_shape[0] > 1:
        format_grads = format_grads + ["NC1HWC0"] * len(dtype_base)
        format_weight = format_weight + ["NC1HWC0"] * len(dtype_base)
        valid_format_category = valid_format_category + 1
    
    dtype_base = dtype_base * valid_format_category

    dtype_str = ','.join(dtype_base)
    format_grad_str = ','.join(format_grads)
    format_weight_str = ','.join(format_weight)

    input0 = util_select_op_base.gen_param(classify="input0", name="grads", datatype=dtype_str,
                                           format=format_grad_str, unknownshape_format=format_grad_str)
    input1 = util_select_op_base.gen_param(classify="input1", name="features", datatype=dtype_str,
                                           format=format_grad_str, unknownshape_format=format_grad_str)
    input2 = util_select_op_base.gen_param(classify="input2", name="weights", datatype=dtype_str,
                                           format=format_weight_str, unknownshape_format=format_weight_str)
    output0 = util_select_op_base.gen_param(classify="output0", name="dx", datatype=dtype_str,
                                            format=format_grad_str, unknownshape_format=format_grad_str)
    output1 = util_select_op_base.gen_param(classify="output1", name="update", datatype=dtype_str,
                                            format=format_grad_str, unknownshape_format=format_grad_str)
    param_list = [input0, input1, input2, output0, output1]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def compare_zero_and_select(input_features, input_weights, new_dtype):
    """
    compare_zero_and_select comput

    Parameters
    ----------
    input_features : TVM tensor
        the tensor of input_features
    input_weights : TVM tensor
        the tensor of input_weights
    new_dtype : str
        dtype of input

    Returns
    -------
    result_dx : TVM tensor
        the tensor of result_dx
    result_update : TVM tensor
        the tensor of result_update
    """
    if util_soc_common.after_v200():
        one = tvm.const(1.0, new_dtype)
        zero = tvm.const(0.0, new_dtype)
        ones_tensor = tbe.broadcast(one, input_features.shape, new_dtype)
        pos_mask = tbe.vcmp(input_features, zero, "gt", "bool")
        zero_one_tensor = tbe.vsel(pos_mask, zero, ones_tensor)
        one_zero_tensor = tbe.vsel(pos_mask, ones_tensor, zero)
        result_dx_neg = tbe.vmul(zero_one_tensor, input_weights)
        result_dx = tbe.vadd(one_zero_tensor, result_dx_neg)
        result_update = tbe.vmul(zero_one_tensor, input_features)
        return result_dx, result_update

    help_min = tvm.const(2 ** (-126), "float32")
    help_rec_one = tvm.const(2 ** 38, "float32")
    help_rec_sec = tvm.const(2 ** 44, "float32")

    tmp_min = tbe.vmins(input_features, help_min)
    tmp_max = tbe.vmaxs(tmp_min, tvm.const(0, new_dtype))
    tmp_result = tbe.vmuls(tmp_max, help_rec_one)
    if new_dtype == "float32":
        tmp_result = tbe.vmuls(tmp_result, help_rec_sec)
    tmp_result = tbe.vmuls(tmp_result, help_rec_sec)
    tmp_neg_result = tbe.vadds(tmp_result, tvm.const(-1, new_dtype))
    tmp_neg_result = tbe.vabs(tmp_neg_result)

    result_dx_pos = tmp_result
    result_dx_neg = tbe.vmul(tmp_neg_result, input_weights)

    result_update_neg = tbe.vmul(tmp_neg_result, input_features)
    result_dx = tbe.vadd(result_dx_pos, result_dx_neg)
    result_update = result_update_neg

    return result_dx, result_update


# 'pylint: disable=too-many-arguments,unused-argument,too-many-branches,too-many-statements
@register_operator_compute("PReluGradUpdate", op_mode="dynamic", support_fusion=False)
def prelu_grad_upate_compute(input_grads, input_features, input_weights,
                             output_backprops_dx, output_backprops_update, kernel_name="prelu_grad_update"):
    """
    calculating the backpropagation of prelu operation
    prelu equivalent function: prelu(x) = max(0, input_features)
    + input_weights * min(0, input_features)
    so prelu_grad_update output_backprops:
        output_backprops_dx = input_features > 0
                ? input_gradients : input_weights * input_gradients
        output_backprops_update = input_features > 0
                ? 0 : input_features * input_gradients

    Parameters
    ----------
    input_gradients : TVM tensor
        input tensor of grad
    input_features : TVM tensor
        input tensor of prelu output
    input_weights : TVM tensor
        input tensor of prelu output
    kernel_name : str
        kernel name, default value is "prelu_grad_update"

    Returns
    -------
    output tensor
    """
    dtype = input_grads.dtype
    dtype_cast_list = ["float16", "bfloat16"]
    new_dtype = dtype
    if dtype in dtype_cast_list and tbe_platform.api_check_support("te.lang.cce.broadcast", "float32"):
        input_grads = tbe.cast_to(input_grads, "float32")
        input_features = tbe.cast_to(input_features, "float32")
        input_weights = tbe.cast_to(input_weights, "float32")
        new_dtype = "float32"

    _, _, shape_max = shape_util.broadcast_shapes(input_grads.shape, input_features.shape,
                                                  param_name_input1="input_gradients",
                                                  param_name_input2="input_features")
    data_input_grads = tbe.broadcast(input_grads, shape_max, new_dtype)
    data_input_features = tbe.broadcast(input_features, shape_max, new_dtype)
    data_input_weights = tbe.broadcast(input_weights, shape_max, new_dtype)

    res_dx, res_update = compare_zero_and_select(data_input_features, data_input_weights, new_dtype)
    output_backprops_dx = tbe.vmul(res_dx, data_input_grads)
    output_backprops_update = tbe.vmul(res_update, data_input_grads)

    if dtype in dtype_cast_list and tbe_platform.api_check_support("te.lang.cce.vmuls", "float32"):
        if dtype == "float16":
            output_backprops_dx = tbe.cast_to(output_backprops_dx, dtype)
        elif dtype == "bfloat16":
            output_backprops_dx = dsl.round(output_backprops_dx, dtype)
        output_backprops_dx_tmp = tbe.cast_to(output_backprops_dx, "float32")
    else:
        output_backprops_dx_tmp = output_backprops_dx

    output_backprops_dx_zero = tbe.vmuls(output_backprops_dx_tmp, tvm.const(0, new_dtype))
    output_backprops_update = tbe.vadd(output_backprops_update, output_backprops_dx_zero)

    if dtype == "float16":
        output_backprops_update = tbe.cast_to(output_backprops_update, dtype)
    elif dtype == "bfloat16":
        output_backprops_update = dsl.round(output_backprops_update, dtype)

    return [output_backprops_dx, output_backprops_update]


def broadcast_weight_shape(input_gradients, input_features, input_weights):
    if is_unknown_rank_input([input_gradients, input_features, input_weights]):
        if is_unknown_rank_input([input_gradients]):
            input_features, input_weights = input_gradients, input_gradients
        elif is_unknown_rank_input([input_features]):
            input_gradients, input_weights = input_features, input_features
        else:
            input_gradients, input_features = input_weights, input_weights
    else:
        input_features_shape = input_features.get("shape")
        input_weights_shape = input_weights.get("shape")
        input_features_format = input_features.get("format").upper()
        new_weights_shape = []
        if (len(input_weights_shape) == len(input_features_shape) - 1) and len(input_weights_shape) != 1:
            new_weights_shape = list(input_weights_shape)
            new_weights_shape.insert(0, 1)
        elif len(input_features_shape) == 4:
            new_weights_shape = [1, input_weights_shape[0], 1, 1]
        elif input_features_format == "NC1HWC0" and len(input_weights_shape) == 5:
            new_weights_shape = input_weights_shape
        elif input_features_format == "NC1HWC0" and len(input_weights_shape) == 1 and input_weights_shape[0] != 1:
            weights_c1 = input_weights_shape[0] if input_weights_shape[0] < 0 else (input_weights_shape[0] + 15) // 16
            new_weights_shape = [1, weights_c1, 1, 1, 16]
        else:
            new_weights_shape = [1 for _ in range(len(input_features_shape))]
            new_weights_shape[1] = input_weights_shape[0]

        new_weight_range = []
        for i, _range in enumerate(input_features["range"]):
            _range = (new_weights_shape[i], new_weights_shape[i]) if new_weights_shape[i] != -1 else _range
            new_weight_range.append(_range)
        input_weights["range"] = tuple(new_weight_range)
        input_weights["shape"] = new_weights_shape
    return input_gradients, input_features, input_weights


@register_operator("PReluGradUpdate")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def prelu_grad_update(input_gradients,
                      input_features,
                      input_weights,
                      output_backprops_dx,
                      output_backprops_update,
                      kernel_name="prelu_grad_update"):
    """
    calculating the backpropagation of prelu operation
    prelu equivalent function: prelu(x) = max(0, input_features)
    + input_weights * min(0, input_features)
    so prelu_grad_upate output_backprops:
        output_backprops_dx = input_features > 0
                ? input_gradients : input_weights * input_gradients
        output_backprops_update = input_features > 0
                ? 0 : input_features * input_gradients
    support dtype:float16, float32

    Parameters
    ----------
    input_gradients : dict
        shape and dtype of grad, not support 1D
    input_features : dict
        shape and dtype of input tensor, not support 1D
    input_weights : dict
        shape and dtype of input learning weight
    output_backprops_dx : dict
        shape and dtype of output, should be same shape and type as input_features
    update : dict
        shape and dtype of output, should be same shape and type as input_features
    kernel_name : str
        kernel name, default value is "prelu_grad_update"
    Returns
    -------
    None
    """
    valid_dtypes = ("float16", "float32", "bfloat16")
    input_gradients_dtype = input_gradients.get("dtype").lower()
    input_features_dtype = input_features.get("dtype").lower()
    input_weights_dtype = input_weights.get("dtype").lower()
    shape_util.compare_tensor_dict_key(input_gradients, input_features, "dtype")
    shape_util.compare_tensor_dict_key(input_gradients, input_weights, "dtype")
    para_check.check_dtype(input_gradients_dtype, valid_dtypes, param_name="input_gradients")
    para_check.check_dtype(input_features_dtype, valid_dtypes, param_name="input_features")
    para_check.check_dtype(input_weights_dtype, valid_dtypes, param_name="input_weights")

    input_gradients_shape = input_gradients.get("shape")
    input_features_shape = input_features.get("shape")
    input_weights_shape = input_weights.get("shape")
    para_check.check_shape(input_gradients_shape, param_name="input_gradients")
    para_check.check_shape(input_features_shape, param_name="input_features")
    para_check.check_shape(input_weights_shape, param_name="input_weights")

    valid_formats = ("ND", "NCHW", "NC1HWC0")
    input_gradients_format = input_gradients.get("format").upper()
    input_features_format = input_features.get("format").upper()
    para_check.check_format(input_gradients_format, valid_formats, param_name="input_gradients")
    para_check.check_format(input_features_format, valid_formats, param_name="input_features")

    input_gradients, input_features, input_weights = broadcast_weight_shape(input_gradients,
                                                                            input_features,
                                                                            input_weights)

    ins = classify([input_gradients, input_features, input_weights], OpPatternMode.ELEWISE_WITH_BROADCAST)

    schedule_list, tensor_list = [], []
    for (_grads, _features, _weights) in ins:
        with tbe.compute():
            grads_shape, features_shape, weights_shape = shape_util.variable_shape([_grads, _features, _weights])
            tensor_grads = tvm.placeholder(grads_shape, input_gradients_dtype, "tensor_grads")
            tensor_features = tvm.placeholder(features_shape, input_features_dtype, "tensor_features")
            tensor_weights = tvm.placeholder(weights_shape, input_weights_dtype, "tensor_weights")
            res = prelu_grad_upate_compute(tensor_grads, tensor_features, tensor_weights,
                                           output_backprops_dx, output_backprops_update, kernel_name)
            tensor_list.append([tensor_grads, tensor_features, tensor_weights] + list(res))
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedule_list.append(sch)
    config = {
        "name" : kernel_name,
        "tensor_list" : tensor_list,
        "bool_storage_as_1bit": False
    }
    tbe.build(schedule_list, config)
