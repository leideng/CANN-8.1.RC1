# Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
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
prelu_grad_reduce
"""
# 'pylint: disable=locally-disabled,invalid-name
# 'pylint: disable=unnecessary-comprehension,global-statement
from impl.util import util_common
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_compute import only_static_support
from impl.util import util_select_op_base
import tbe.dsl as dsl


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,too-many-locals,invalid-name
def op_select_format(grads, features, weights, updates, da, kernel_name="prelu_grad_reduce"):
    """ calculating data
    Parameters
    ----------
    grads : TVM tensor
        input tensor of grad
    features : TVM tensor
        input tensor of prelu output
    weights : TVM tensor
        input tensor of prelu output
    update : TVM tensor
        input tensor of prelu output
    da : dict
        da output dict of prelu_grad
    kernel_name : str
        kernel name, default value is "prelu_grad"
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
    input3 = util_select_op_base.gen_param(classify="input3", name="updates", datatype=dtype_str,
                                           format=format_grad_str, unknownshape_format=format_grad_str)
    output0 = util_select_op_base.gen_param(classify="output0", name="da", datatype=dtype_str,
                                            format=format_weight_str, unknownshape_format=format_weight_str)
    param_list = [input0, input1, input2, input3, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class ReduceParam:
    """
    The class for ReduceParam
    """
    REDUCE_LIST = None


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,too-many-branches
# 'pylint: disable=locally-disabled,too-many-locals
def prelu_grad_reduce_infer_axes(input_gradients, input_weights, input_updates):
    """
    To infer sum operate axis by input_updates
    to keep compute Architecture, so use global parameter send variable
    Parameters:
    ----------
    input_gradients: dict
        shape and dtype of input
    input_weights: dict
        shape and dtype of input
    input_updates: dict
        shape and dtype of input

    Returns
    -------
    g_shape_list. list
    keepdims. bool
    """
    shape_input_da = shape_util.shape_to_list(input_updates.get("shape"))
    shape_input_weights = shape_util.shape_to_list(input_weights.get("shape"))
    input_format = input_gradients.get("format")

    weight_share = False
    keepdims = False
    if len(shape_input_weights) == 1 and shape_input_weights[0] == 1:
        weight_share = True

    axis = list(range(len(shape_input_da)))
    if (len(shape_input_weights) == len(shape_input_da) - 1) and len(shape_input_weights) != 1:
        axis = [0]  
        for i in range(1, len(shape_input_da)):
            if shape_input_da[i] != shape_input_weights[i - 1]:
                axis += [i]
        keepdims = True
    elif len(shape_input_da) == 4:
        if not weight_share:
            axis = [0, 2, 3]
    elif len(shape_input_da) == 5 and input_format == "NC1HWC0":
        if not weight_share:
            axis = [0, 2, 3] 
            keepdims = True
    else:
        if not weight_share:
            axis_nd = axis[0:1] + axis[2:]
            axis = axis_nd

    return axis, keepdims


@register_operator_compute("PReluGradReduce", op_mode="dynamic", support_fusion=True)
def prelu_grad_reduce_compute(input_updates, output_backprops_da, kernel_name="prelu_grad_reduce"):
    """
    Reduce a tensor on last dimension in axis based on sum.

    Parameters:
    ----------
    input_updates: TVM tensor
        the placeholder of input data
    output_backprops_da: dict
        shape and dtype of output
    kernel_name : str
        cce kernel name, default value is prelu_grad_reduce

    Returns
    -------
    TVM tensor da by prelu_grad
    """
    dtype = input_updates.dtype
    res_dtype = output_backprops_da.get("dtype").lower()

    if dtype == "bfloat16":
        input_updates = tbe.cast_to(input_updates, "float32")
    elif dtype == "float16" and tbe_platform.api_check_support("te.lang.cce.sum", "float32"):
        input_updates = tbe.cast_to(input_updates, "float32")

    result = tbe.reduce_sum(input_updates, ReduceParam.REDUCE_LIST)
    result = dsl.round(result, res_dtype) if res_dtype == "bfloat16" else tbe.cast_to(result, res_dtype)        

    return result


@register_operator("PReluGradReduce")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def prelu_grad_reduce(input_gradients,
               input_features,
               input_weights,
               input_updates,
               output_backprops_da,
               kernel_name="prelu_grad_reduce"):
    """
    Reduce a tensor on last dimension in axis based on sum.

    Parameters:
    ----------
    input_gradients : dict
        shape and dtype of input, only support float16, float32, bfloat16
    input_features : dict
        shape and dtype of input, only support float16, float32, bfloat16
    input_weights : dict
        shape and dtype of input, only support float16, float32, bfloat16
    input_updates : dict
        shape and dtype of input, only support float16, float32, bfloat16
    output_backprops_da : dict
        shape and dtype of output, should be same shape and type as input_weights
    kernel_name : str
        cce kernel name, default value is prelu_grad_reduce
    Returns
    -------
    None
    """
    # check shape
    shape_input_gradients = input_gradients.get("shape")
    shape_input_features = input_features.get("shape")
    shape_input_weights = input_weights.get("shape")
    shape_input_updates = input_updates.get("shape")
    
    para_check.check_shape(shape_input_gradients, param_name="input_gradients")
    para_check.check_shape(shape_input_features, param_name="input_features")
    para_check.check_shape(shape_input_weights, param_name="input_weights")
    para_check.check_shape(shape_input_updates, param_name="input_updates")

    # check dtype
    check_list = ("float16", "float32", "bfloat16")
    dtype_input_gradients = input_gradients.get("dtype").lower()
    dtype_input_features = input_features.get("dtype").lower()
    dtype_input_weights = input_weights.get("dtype").lower()
    dtype_input_updates = input_updates.get("dtype").lower()

    para_check.check_dtype(dtype_input_gradients, check_list, param_name="input_gradients")
    para_check.check_dtype(dtype_input_features, check_list, param_name="input_features")
    para_check.check_dtype(dtype_input_weights, check_list, param_name="input_weights")
    para_check.check_dtype(dtype_input_updates, check_list, param_name="input_updates")
    if input_gradients == "bfloat16" and not tbe_platform.api_check_support("te.lang.cce.sum", "float32"):
        raise RuntimeError("bfloat16 is not support at current platform")
    input_gradients_format = input_gradients.get("format")
    vaild_formats = ("ND", "NCHW", "NC1HWC0")
    para_check.check_format(input_gradients_format, vaild_formats, param_name="input_gradients")    

    shape_util.compare_tensor_dict_key(input_gradients, input_features, "dtype")
    shape_util.compare_tensor_dict_key(input_gradients, input_weights, "dtype")
    shape_util.compare_tensor_dict_key(input_gradients, input_updates, "dtype")

    input_updates["rel_pos_to_reduce"] = "before"
    is_unknown_rank = util_common.is_unknown_rank_input(input_updates)
    if is_unknown_rank:
        input_axis = {"shape": [-1], "rel_pos_to_reduce": "axis"}
        keep_dims = False
    else:
        g_shape_list, keep_dims = prelu_grad_reduce_infer_axes(input_gradients, input_weights, input_updates)
        input_axis = {"shape": [len(g_shape_list), ], "value": g_shape_list, "rel_pos_to_reduce": "axis"}
    ins = classify([input_updates, input_axis], OpPatternMode.REDUCE, {"keepdims": keep_dims})
    schedules, tensors = [], []
    for (_da, axes) in ins:
        with tbe.compute():
            shape_da = shape_util.variable_shape([_da, axes], op_mode="reduce")[0]
            tensor_input_updates = tvm.placeholder(shape_da, name="tensor_input_updates", dtype=dtype_input_updates)
            tensor_input_gradients = tvm.placeholder(shape_input_gradients, 
                                                     name="tensor_input_gradients", dtype=dtype_input_gradients)
            tensor_input_features = tvm.placeholder(shape_input_features, 
                                                    name="tensor_input_features", dtype=dtype_input_features)
            tensor_input_weights = tvm.placeholder(shape_input_weights, 
                                                   name="tensor_input_weights", dtype=dtype_input_weights)
            ReduceParam.REDUCE_LIST = shape_util.axis_check(len(shape_da), axes.get("value"))
            res = prelu_grad_reduce_compute(tensor_input_updates, output_backprops_da)
            tensors.append([tensor_input_gradients, tensor_input_features, 
                            tensor_input_weights, tensor_input_updates, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}
    tbe_context.get_context().add_compile_info("is_unknown_rank", is_unknown_rank)
    tbe.build(schedules, config)
