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
binary_cross_entropy
"""
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import OpImplMode
from impl.util.platform_adapter import register_operator_compute
from impl.util import util_select_op_base
from impl.util import util_common


# 'pylint: disable=invalid-name,too-many-arguments
# 'pylint: disable=unused-argument,too-many-locals
def op_select_format(x, y, weight, output,
                     reduction="mean",
                     kernel_name="binary_cross_entropy"):
    """
    1. when input x's ori_shape is 4D and the ori_format in ["NCHW", "NHWC"],
    and the dim C of x's ori_shape can be divisible by 16. The Op
    BinaryCrossEntropy can support ND and NC1HWC0.
    > for example:
    > x : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")
    > y : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")
    > weight : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")
    2. when input x's ori_shape is 5D and the ori_format in ["NDCHW", "NDHWC"],
    and the dim C of x's ori_shape can be divisible by 16. The Op
    BinaryCrossEntropy can support ND and NDC1HWC0.
    > for example:
    > x : Tensor of (shape=(16,1, 1, 16, 16, 16), "NDC1HWC0")
    > y : Tensor of (shape=(16,1, 1, 16, 16, 16), "NDC1HWC0")
    > weight : Tensor of (shape=(16,1, 1, 16, 16, 16), "NDC1HWC0")
    """
    is_support_hd = False
    is_support_fz = False
    support_ori_format = \
        util_common.get_fused_format_str(["N", "D", "H", "W", "C"]) \
        + util_common.get_fused_format_str(["N", "H", "W", "C"])
    input_ori_shape = x.get("ori_shape")
    input_ori_format = x.get("ori_format")
    shape_hd_c0 = 16
    shape_fz_c0 = 16
    shape_fz_n = 16

    if input_ori_format in support_ori_format \
            and len(input_ori_shape) == len(input_ori_format):
        if input_ori_shape[input_ori_format.index("C")] % shape_fz_c0 == 0 \
                and input_ori_shape[input_ori_format.index("N")] % shape_fz_n == 0:
            is_support_fz = True
        elif input_ori_shape[input_ori_format.index("C")] % shape_hd_c0 == 0:
            is_support_hd = True

    if reduction in ("none",):
        is_support_hd = True
        is_support_fz = True

    cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        dtype_base = ["float16"]
    else:
        dtype_base = ["float16", "float"]

    dtype_base_out = dtype_base[:]
    format_base_out = ["ND"] * len(dtype_base)

    if is_support_hd:
        dtype_base_out = dtype_base_out + dtype_base
        other_format = "NDC1HWC0" if len(input_ori_shape) == 5 else "NC1HWC0"
        format_base_out = format_base_out + [other_format] * len(dtype_base)
    if is_support_fz:
        dtype_base_out = dtype_base_out + dtype_base
        other_format = "FRACTAL_Z_3D" if len(input_ori_shape) == 5 else "FRACTAL_Z"
        format_base_out = format_base_out + [other_format] * len(dtype_base)

    dtype_str = ','.join(dtype_base_out)
    format_str = ','.join(format_base_out)

    input0 = util_select_op_base.gen_param(
        classify="input0", name="x", datatype=dtype_str,
        format=format_str)
    input1 = util_select_op_base.gen_param(
        classify="input1", name="y", datatype=dtype_str,
        format=format_str)
    input2 = util_select_op_base.gen_param(
        classify="input2", name="weight", datatype=dtype_str,
        format=format_str)
    output0 = util_select_op_base.gen_param(
        classify="output0", name="output", datatype=dtype_str,
        format=format_str)
    param_list = [input0, input1, input2, output0]

    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


@register_operator_compute("binary_cross_entropy", op_mode="static", support_fusion=True)
def binary_cross_entropy_compute(x, y, weight, output,
                                 reduction, kernel_name):
    """
    calculating binary_cross_entropy

    Parameters
    ----------
    x : TVM tensor
        the output of previous layer
    y : TVM tensor
        label
    weight :
        a manual rescaling weight given to the loss of each batch element.
        If given, has to be a Tensor of size nbatch
    output :
        loss result after compute
    reduction :
        reduce configuration parameter: mean/sum/none. Default: mean
    kernel_name : str
        kernel name, default value is "binary_cross_entropy"

    Returns
    -------
    result : TVM tensor
        output tensor
    """
    ori_dtype = x.dtype
    trans_dtype = ori_dtype
    shape = shape_util.shape_to_list(x.shape)
    if ori_dtype == "float16" and tbe_platform.api_check_support(
            "tbe.dsl.vmul", "float32"):
        x = tbe.cast_to(x, "float32")
        y = tbe.cast_to(y, "float32")
        if weight is not None:
            weight = tbe.cast_to(weight, "float32")
        trans_dtype = "float32"

    const_one = tvm.const(1, trans_dtype)
    const_neg_one = tvm.const(-1, trans_dtype)
    scalar_eps = 1e-12
    # calcu value : y * log(x)
    x = tbe.vmaxs(x, tvm.const(scalar_eps, trans_dtype))
    x_log_tmp = tbe.vlog(x, priority_flag=1)
    data_mul1 = tbe.vmul(x_log_tmp, y)
    # calcu value : (1-y) * log(1-x)
    x_neg_tmp = tbe.vmuls(x, const_neg_one)
    x1_tmp = tbe.vadds(x_neg_tmp, const_one)
    y_neg_tmp = tbe.vmuls(y, const_neg_one)
    y1_tmp = tbe.vadds(y_neg_tmp, const_one)
    x1_tmp = tbe.vmaxs(x1_tmp, tvm.const(scalar_eps, trans_dtype))
    x1_log_tmp = tbe.vlog(x1_tmp, priority_flag=1)
    data_mul2 = tbe.vmul(x1_log_tmp, y1_tmp)
    # calcu value : y * log(x) + (1-y) * log(1-x)
    data_sum = tbe.vadd(data_mul1, data_mul2)
    # calcu value : -(y * log(x) + (1-y) * log(1-x))
    result = tbe.vmuls(data_sum, const_neg_one)

    if weight is not None:
        result = tbe.vmul(result, weight)

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

    if reduction == "mean":
        result = tbe.vmuls(result, cof)
        result = tbe.sum(result, axis=axis_d, keepdims=False)
    elif reduction == "sum":
        result = tbe.sum(result, axis=axis_d, keepdims=False)
    elif reduction == "none":
        pass

    if ori_dtype == "float16":
        result = tbe.cast_to(result, "float16")

    return result


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def binary_cross_entropy(x, y, weight, output,
                         reduction="mean",
                         kernel_name="binary_cross_entropy"):
    """
    calculating data
    res = -w (y ln(x) + (1-y) ln(1-x))
    if reduction == sum:  res = reduce_sum(res)            output a scalar
    if reduction == mean:  res = reduce_sum(res)/data_len  output a scalar
    if reduction == none: res = res   output a tensor

    Parameters
    ----------
    x : dict
        shape and dtype of tensor predict
    y : dict
        shape and dtype of tensor target,
        should be same shape and dtype as predict
    weight : None or TVM tensor
        a manual rescaling weight given to the loss of each batch element.
        If given, has to be a Tensor of size nbatch
    output : dict
        shape and dtype of output, loss result after compute
    reduction : str
        Specifies the reduction to apply to the output:'none' | 'mean' | 'sum'
         Default: 'mean'
        'none': no reduction will be applied,
        'mean': the sum of the output will be divided by the number
                of elements in the output
        'sum': the output will be summed. Note: size_average and reduce
               are in the process of being deprecated
               and in the meantime, specifying either of those
               two args will override reduction.
    kernel_name : str
        kernel name, default value is "binary_cross_entropy"

    Returns
    -------
    None
    """
    predict_shape = x.get("shape")
    predict_dtype = x.get("dtype")
    predict_dtype_lower = predict_dtype.lower()

    target_shape = y.get("shape")
    target_dtype = y.get("dtype")
    target_dtype_lower = target_dtype.lower()

    # check dtype
    dtype_list = ("float16", "float32")
    para_check.check_dtype(predict_dtype, dtype_list, param_name="x")
    para_check.check_dtype(target_dtype, dtype_list, param_name="y")
    shape_util.compare_tensor_dict_key(x, y, "dtype")

    # check shape
    para_check.check_shape(predict_shape, param_name="x")
    para_check.check_shape(target_shape, param_name="y")
    shape_util.compare_tensor_dict_key(x, y, "shape")


    data_weight = None
    if weight is not None:
        weight_shape = weight.get("shape")
        weight_dtype = weight.get("dtype")
        weight_dtype_lower = weight_dtype.lower()
        para_check.check_dtype(weight_dtype, dtype_list, param_name="weight")
        shape_util.compare_tensor_dict_key(x, weight, "dtype")
        para_check.check_shape(weight_shape, param_name="weight")
        shape_size = para_check.check_tensor_shape_size(weight_shape)
        shape_util.compare_tensor_dict_key(x, weight, "shape")
        data_weight = tvm.placeholder([shape_size], name="data_weight",
                                      dtype=weight_dtype_lower)
    shape_size = para_check.check_tensor_shape_size(predict_shape)
    data_predict = tvm.placeholder([shape_size], name="data_predict",
                                   dtype=predict_dtype_lower)
    shape_size = para_check.check_tensor_shape_size(target_shape)
    data_target = tvm.placeholder([shape_size], name="data_target",
                                  dtype=target_dtype_lower)

    if reduction not in ("mean", "sum", "none"):
        rule_desc = "reduction type should in mean/sum/none"
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule_desc,
                                                          "reduction", reduction)

    res = binary_cross_entropy_compute(data_predict, data_target,
                                       data_weight, output,
                                       reduction, kernel_name)

    with tvm.target.cce():
        schedule = auto_schedule(res)

    if weight is None:
        config = {"name": kernel_name,
                  "tensor_list": [data_predict, data_target, res]}
    else:
        config = {"name": kernel_name,
                  "tensor_list": [data_predict, data_target, data_weight, res]}
    build(schedule, config)
