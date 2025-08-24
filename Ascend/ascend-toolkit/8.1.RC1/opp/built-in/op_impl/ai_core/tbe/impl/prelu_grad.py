#!/usr/bin/python
# -*- coding: utf-8 -*-
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
prelu_grad
"""
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tvm
from impl.util import util_select_op_base
from impl.util import util_soc_common


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,too-many-locals,invalid-name
def op_select_format(grads, features, weights, dx, da, kernel_name="prelu_grad"):
    """ calculating data

    Parameters
    ----------
    grads : TVM tensor
        input tensor of grad
    features : TVM tensor
        input tensor of prelu output
    weights : TVM tensor
        input tensor of prelu output
    dx : dict
        dx output dict of prelu_grad
    da : dict
        da output dict of prelu_grad
    input_format : str
        input format of grad
    kernel_name : str
        kernel name, default value is "prelu_grad"
    Returns
    -------
    None.
    """

    weights_shape = weights.get("ori_shape")
    dtype_base = ["float"]
    format_grads = ["ND", "NCHW"]
    format_weight = ["ND", "ND"]
    dtype_base_out = ["float", "float"]

    if len(weights_shape) == 1 and weights_shape[0] != 1:
        dtype_base_out = dtype_base_out + dtype_base
        format_grads = format_grads + ["NC1HWC0"] * len(dtype_base)
        format_weight = format_weight + ["NC1HWC0"] * len(dtype_base)

    dtype_str = ','.join(dtype_base_out)
    format_x_str = ','.join(format_grads)
    format_weight_str = ','.join(format_weight)

    input0 = util_select_op_base.gen_param(classify="input0", name="grads", datatype=dtype_str,
                                           format=format_x_str, unknownshape_format=format_x_str)
    input1 = util_select_op_base.gen_param(classify="input1", name="features", datatype=dtype_str,
                                           format=format_x_str, unknownshape_format=format_x_str)
    input2 = util_select_op_base.gen_param(classify="input2", name="weights", datatype=dtype_str,
                                           format=format_weight_str, unknownshape_format=format_weight_str)
    output0 = util_select_op_base.gen_param(classify="output0", name="dx", datatype=dtype_str,
                                            format=format_x_str, unknownshape_format=format_x_str)
    output1 = util_select_op_base.gen_param(classify="output1", name="da", datatype=dtype_str,
                                            format=format_weight_str, unknownshape_format=format_weight_str)
    param_list = [input0, input1, input2, output0, output1]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


# 'pylint: disable=unused-variable
def check_inputs_shape(features_shape, weights_shape, input_format):
    """
    check input para

    Parameters
    ----------
    features_shape : list
        shape of feature_map
    weights_shape : list
        shape of weights
    input_format : str
        str of input

    Returns
    -------
    None
    """
    features_shape = list(features_shape)
    weights_shape = list(weights_shape)
    features_dim = len(features_shape)
    weights_dim = len(weights_shape)

    if features_dim == 1:
        detail = "weight dim only support two values: 1, or the number of channels at input, " \
                 "feature don't support 1D shape, while feature shape is {0}".format(features_shape)
        error_manager_vector.raise_err_input_shape_invalid('prelu_grad', 'feature_shape', detail)

    if input_format == "NC1HWC0" and features_dim == 5 and weights_dim == 5 \
            and (features_shape[1] != weights_shape[1] or features_shape[4] != weights_shape[4]):
        detail = "weight dim only support two values: 1, or the number of channels at input, " \
                 "when feature_dim and weight_dim are 5(NC1HWC0), channel(C1/C0) dim for features and " \
                 "weights must be matched, while feature [C1, C0]:[%d, %d]," \
                 " weight [C1, C0]:[%d, %d]" % (features_shape[1], features_shape[4],
                                                weights_shape[1], weights_shape[4])
        error_manager_vector.raise_err_two_input_shape_invalid('prelu_grad', 'feature', 'weight', detail)
    if input_format == "NC1HWC0" and features_dim == 5 and weights_dim == 1 \
            and features_shape[1] * features_shape[4] != weights_shape[0] and weights_shape[0] != 1:
        detail = "weight dim only support two values: 1, or the number of channels at input, " \
                 "when feature_dim is 5(NC1HWC0), and weight_dim is 1, " \
                 "weight value must be 1 or the number of channel(C1*C0)," \
                 " while feature shape is %s, weight shape is %s" % (features_shape, weights_shape)
        error_manager_vector.raise_err_two_input_shape_invalid('prelu_grad', 'feature', 'weight', detail)
    if input_format == "NC1HWC0" and features_dim == 5 and weights_dim != 5 and weights_dim != 1:
        detail = "weight dim only support two values: 1, or the number of channels at input, " \
                 "when feature_dim is 5(NC1HWC0), weight_dim must be equal to 5(1, C1, 1, 1, C0) " \
                 "or 1(1 or C1*C0), while weight shape is %s" % weights_shape
        error_manager_vector.raise_err_input_shape_invalid('prelu_grad', 'weight', detail)
    if features_dim == 4 and weights_dim == 1 and features_shape[1] != weights_shape[0] and weights_shape[0] != 1:
        detail = "weight dim only support two values: 1, or the number of channels at input, " \
                 "when feature_dim is 4, weight dim must be 1(weight shape is a vector), " \
                 " while feature shape is %s, weight shape is %s" % (features_shape, weights_shape)
        error_manager_vector.raise_err_two_input_shape_invalid('prelu_grad', 'feature', 'weight', detail)
    if features_dim == 4 and weights_dim != 1 and weights_dim != 3:
        detail = "weight dim only support two values: 1, or the number of channels at input, " \
                 "when feature_dim is 4, weight dim must be 1(weight shape is a vector)," \
                 "while feature shape is %s, weight shape is %s" % (features_shape, weights_shape)
        error_manager_vector.raise_err_two_input_shape_invalid('prelu_grad', 'feature', 'weight', detail)
    if input_format == "ND" and features_dim != 1 and weights_dim == 1 \
            and features_shape[1] != weights_shape[0] and weights_shape[0] != 1:
        detail = "weight dim only support two values: 1, or the number of channels at input, " \
                 "When feature_dim is ND(except 1D), weight dim must be 1(weight shape is a vector)," \
                 "channel dim for features and weights' must be matched," \
                 "while feature shape is %s, weight shape is %s" % (features_shape, weights_shape)
        error_manager_vector.raise_err_two_input_shape_invalid('prelu_grad', 'feature', 'weight', detail)
    if input_format == "ND" and features_dim != 1 and weights_dim != 1 and weights_dim != features_dim - 1:
        detail = "weight dim only support two values: 1, or the number of channels at input, " \
                 "When feature_dim is ND(except 1D), weight dim must be 1(weight shape is a vector)," \
                 "channel dim for features and weights' must be matched, " \
                 "while feature shape is %s, weight shape is %s" % (features_shape, weights_shape)
        error_manager_vector.raise_err_two_input_shape_invalid('prelu_grad', 'feature', 'weight', detail)


# 'pylint: disable=too-many-locals
def compare_zero_and_select(input_features, input_weights, shape, dtype):
    """
    compare_zero_and_select comput

    Parameters
    ----------
    input_features : TVM tensor
        the tensor of input_features
    input_weights : TVM tensor
        the tensor of input_weights
    shape: list
        list shape
    dtype : str
        dtype of input

    Returns
    -------
    result_dx : TVM tensor
        the tensor of result_dx
    result_da : TVM tensor
        the tensor of result_da
    """
    shape_input_features = shape_util.shape_to_list(input_features.shape)
    shape_input_weights = shape_util.shape_to_list(input_weights.shape)
    if list(shape_input_features) != list(shape_input_weights):
        input_weights = tbe.broadcast(input_weights, shape, dtype)

    if util_soc_common.after_v200():
        one = tvm.const(1.0, dtype)
        zero = tvm.const(0.0, dtype)
        ones_tensor = tbe.broadcast(one, input_features.shape, dtype)
        pos_mask = tbe.vcmp(input_features, zero, "gt", "bool")
        zero_one_tensor = tbe.vsel(pos_mask, zero, ones_tensor)
        one_zero_tensor = tbe.vsel(pos_mask, ones_tensor, zero)
        result_dx_neg = tbe.vmul(zero_one_tensor, input_weights)
        result_dx = tbe.vadd(one_zero_tensor, result_dx_neg)
        result_update = tbe.vmul(zero_one_tensor, input_features)
        return result_dx, result_update

    # auxiliary number
    help_min = tvm.const(2 ** (-126), "float32")
    help_rec_one = tvm.const(2 ** 38, "float32")
    help_rec_sec = tvm.const(2 ** 44, "float32")

    tmp_min = tbe.vmins(input_features, help_min)
    tmp_max = tbe.vmaxs(tmp_min, tvm.const(0, dtype))
    tmp_result = tbe.vmuls(tmp_max, help_rec_one)
    if dtype == "float32":
        tmp_result = tbe.vmuls(tmp_result, help_rec_sec)
    tmp_result = tbe.vmuls(tmp_result, help_rec_sec)
    tmp_neg_result = tbe.vadds(tmp_result, tvm.const(-1, dtype))
    tmp_neg_result = tbe.vabs(tmp_neg_result)

    result_dx_pos = tmp_result
    result_dx_neg = tbe.vmul(tmp_neg_result, input_weights)

    result_da_neg = tbe.vmul(tmp_neg_result, input_features)
    result_dx = tbe.vadd(result_dx_pos, result_dx_neg)
    result_da = result_da_neg

    return result_dx, result_da


# 'pylint: disable=too-many-arguments,unused-argument,too-many-branches
# 'pylint: disable=too-many-statements
def prelu_grad_compute(input_gradients,
                       input_features,
                       input_weights,
                       output_backprops_dx,
                       output_backprops_da,
                       input_format,
                       is_weight_generalization,
                       kernel_name="prelu_grad"):
    """
    calculating the backpropagation of prelu operation
    prelu equivalent function: prelu(x) = max(0, input_features)
    + input_weights * min(0, input_features)
    so prelu_grad output_backprops:
        output_backprops_dx = input_features > 0
                ? input_gradients : input_weights * input_gradients
        output_backprops_da = input_features > 0
                ? 0 : input_features * input_gradients

    Parameters
    ----------
    input_gradients : TVM tensor
        input tensor of grad
    input_features : TVM tensor
        input tensor of prelu output
    input_weights : TVM tensor
        input tensor of prelu output
    output_backprops_dx : dict
        dx output dict of prelu_grad
    output_backprops_da : dict
        da output dict of prelu_grad
    input_format : str
        input format of grad
    is_weight_generalization : bool
        when weights_dim = grads_dim - 1,is_weight_generalization is Ture
    kernel_name : str
        kernel name, default value is "prelu_grad"

    Returns
    -------
    output tensor
    """
    dtype = input_gradients.dtype
    trans_type = dtype
    shape_input_gradients = shape_util.shape_to_list(input_gradients.shape)
    shape_input_features = shape_util.shape_to_list(input_features.shape)
    shape_input_weights = shape_util.shape_to_list(input_weights.shape)
    shape = shape_input_gradients
    weight_share = False
    if input_format == "NC1HWC0":
        if shape_input_weights[4] == 1:
            weight_share = True
    else:
        if shape_input_weights[1] == 1:
            weight_share = True

    # need cast float16 to float32
    if dtype == "float16" and tbe_platform.api_check_support(
            "te.lang.cce.broadcast", "float32"):
        input_gradients = tbe.cast_to(input_gradients, "float32")
        input_features = tbe.cast_to(input_features, "float32")
        input_weights = tbe.cast_to(input_weights, "float32")
        trans_type = "float32"

    # broadcast in case the input shapes are not same
    if list(shape_input_gradients) != list(shape_input_features):
        shape_input_gradients, shape_input_features, shape = \
            shape_util.broadcast_shapes(
                shape_input_gradients, shape_input_features,
                param_name_input1="input_gradients",
                param_name_input2="input_features")
        input_gradients = tbe.broadcast(input_gradients, shape, trans_type)
        input_features = tbe.broadcast(input_features, shape, trans_type)

    # custom vcmpsel start
    res_dx, res_da = compare_zero_and_select(input_features, input_weights,
                                             shape, trans_type)
    output_backprops_dx = tbe.vmul(res_dx, input_gradients)
    output_backprops_da = tbe.vmul(res_da, input_gradients)
    # custom vcmpsel end

    if dtype == "float16" and tbe_platform.api_check_support("te.lang.cce.vmuls", "float32"):
        output_backprops_dx = tbe.cast_to(output_backprops_dx, dtype)
        output_backprops_dx_tmp = tbe.cast_to(output_backprops_dx, "float32")
    else:
        output_backprops_dx_tmp = output_backprops_dx

    output_backprops_dx_zero = tbe.vmuls(output_backprops_dx_tmp, tvm.const(0, trans_type))
    output_backprops_da = tbe.vadd(output_backprops_da, output_backprops_dx_zero)

    shape_input_da = shape_util.shape_to_list(output_backprops_da.shape)
    axis = list(range(len(shape_input_da)))

    if is_weight_generalization:
        pass
    elif len(shape_input_features) == 4:
        if not weight_share:
            output_backprops_da = tbe.sum(
                output_backprops_da, axis=[0, 2, 3], keepdims=False)
        else:
            output_backprops_da = tbe.sum(
                output_backprops_da, axis=axis, keepdims=False)
    elif len(shape_input_features) == 5 and input_format == "NC1HWC0":
        if not weight_share:
            output_backprops_da = tbe.sum(
                output_backprops_da, axis=[0, 2, 3], keepdims=True)
        else:
            output_backprops_da = tbe.sum(
                output_backprops_da, axis=axis, keepdims=False)
    else:
        if not weight_share:
            axis_nd = axis[0:1] + axis[2:]
            output_backprops_da = tbe.sum(
                output_backprops_da, axis=axis_nd, keepdims=False)
        else:
            output_backprops_da = tbe.sum(
                output_backprops_da, axis=axis, keepdims=False)

    if dtype == "float16":
        output_backprops_da = tbe.cast_to(output_backprops_da, dtype)

    return output_backprops_dx, output_backprops_da


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def prelu_grad(input_gradients,
               input_features,
               input_weights,
               output_backprops_dx,
               output_backprops_da,
               kernel_name="prelu_grad"):
    """
    calculating the backpropagation of prelu operation
    prelu equivalent function: prelu(x) =
    max(0, input_features) + input_weights * min(0, input_features)

    so prelu_grad output_backprops:
        output_backprops_dx = input_features > 0
            ? input_gradients : input_weights * input_gradients
        output_backprops_da = input_features > 0
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
        shape and dtype of output, should be same shape
         and type as input_features
    output_backprops_da : dict
        shape and dtype of output, should be same shape
         and type as input_features
    kernel_name : str
        kernel name, default value is "prelu_grad"

    Returns
    -------
    None
    """
    shape_input_gradients = input_gradients.get("shape")
    dtype_input_gradients = input_gradients.get("dtype")
    input_gradients_dtype = dtype_input_gradients.lower()
    input_format = input_gradients.get("format")

    shape_input_features = input_features.get("shape")
    dtype_input_features = input_features.get("dtype")
    input_features_dtype = dtype_input_features.lower()

    shape_input_weights = input_weights.get("shape")
    dtype_input_weights = input_weights.get("dtype")
    input_weights_dtype = dtype_input_weights.lower()

    # check dtype
    check_list = ("float16", "float32")
    shape_util.compare_tensor_dict_key(input_gradients, input_features, "dtype")
    shape_util.compare_tensor_dict_key(input_gradients, input_weights, "dtype")
    para_check.check_dtype(dtype_input_gradients, check_list, param_name="input_gradients")
    para_check.check_dtype(dtype_input_features, check_list, param_name="input_features")
    para_check.check_dtype(dtype_input_weights, check_list, param_name="input_weights")
    # check shape
    para_check.check_shape(shape_input_gradients, param_name="input_gradients")
    para_check.check_shape(shape_input_features, param_name="input_features")
    para_check.check_shape(shape_input_weights, param_name="input_weights")
    if list(shape_input_gradients) != list(shape_input_features):
        shape_input_gradients, shape_input_features, shape_max = \
            shape_util.broadcast_shapes(shape_input_gradients, shape_input_features,
                                        param_name_input1="input_gradients",
                                        param_name_input2="input_features")
    check_inputs_shape(shape_input_features, shape_input_weights, input_format)

    is_weight_generalization = False
    if (len(shape_input_weights) == len(shape_input_features) - 1) and len(shape_input_weights) != 1:
        shape_input_weights = list(shape_input_weights)
        shape_input_weights.insert(0, 1)
        is_weight_generalization = True
    elif len(shape_input_features) == 4:
        shape_input_weights = [1, shape_input_weights[0], 1, 1]
    elif input_format == "NC1HWC0" and len(shape_input_weights) == 5:
        pass
    elif input_format == "NC1HWC0" and len(shape_input_weights) == 1 \
            and shape_input_weights[0] != 1:
        weights_c1 = (shape_input_weights[0] + 15) // 16
        shape_input_weights = [1, weights_c1, 1, 1, 16]
    else:
        weights_shape = [1 for _ in range(len(shape_input_features))]
        weights_shape[1] = shape_input_weights[0]
        shape_input_weights = weights_shape

    data_input_gradients = tvm.placeholder(
        shape_input_gradients,
        name="data_input_gradients",
        dtype=input_gradients_dtype)
    data_input_features = tvm.placeholder(
        shape_input_features,
        name="data_input_features",
        dtype=input_features_dtype)
    data_input_weights = tvm.placeholder(
        shape_input_weights,
        name="data_input_weights",
        dtype=input_weights_dtype)
    res_dx, res_da = prelu_grad_compute(
        data_input_gradients, data_input_features, data_input_weights,
        output_backprops_dx, output_backprops_da, input_format, is_weight_generalization, kernel_name)
    res = [res_dx, res_da]
    tensor_list = [data_input_gradients, data_input_features, data_input_weights] + list(res)

    with tvm.target.cce():
        sch = auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": tensor_list,
              "bool_storage_as_1bit": False}

    build(sch, config)
