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
binary_cross_entropy
"""
import math
from impl.util.util_common import is_unknown_rank_input
from impl.util import util_select_op_base
from impl.util import util_common
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_context
from impl.util.util_compute import only_static_support
from impl.util.util_compute import get_cof
from impl.util.platform_adapter import OpTbeImplMode


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # eps value
    SCALAR_EPS = 1e-12
    SHAPE_ALIGNED = 16
    THREE_DIMS_SHAPE_LENS = 5
    MIN_VALUE = -100.0


# 'pylint: disable=invalid-name,too-many-arguments
# 'pylint: disable=unused-argument,too-many-locals
def op_select_format(x, y, weight, output,
                     reduction="mean",
                     kernel_name="binary_cross_entropy"):
    """
    select format dynamically
    op_select_format support desc:

    1.when the reduction mode is mean/sum
    1.1 when the dim C of x's ori_shape can be divisible by 16
    The Op can support
    ND + ND + ND = ND,
    NC1HWC0 + NC1HWC0 + NC1HWC0 = ND,
    NDC1HWC0 + NDC1HWC0 + NDC1HWC0 = ND.

        for example:
        inputs:
            x          shape = [2, 16] format = "ND"
            y          shape = [2, 16] format = "ND"
            weight     shape = [2, 16] format = "ND"
        outputs:
            output     shape = [1] format = "ND"

        inputs:
            x          shape = [16, 1, 1, 16, 16, 16] format = "NDC1HWC0"
            y          shape = [16, 1, 1, 16, 16, 16] format = "NDC1HWC0"
            weight     shape = [16, 1, 1, 16, 16, 16] format = "NDC1HWC0"
        outputs:
            output     shape = [1] format = "ND"

    1.2  when the dim C and the dim N of x's ori_shape can be divisible by 16
    The Op can support
    ND + ND + ND = ND,
    NC1HWC0 + NC1HWC0 + NC1HWC0 = ND,
    NDC1HWC0 + NDC1HWC0 + NDC1HWC0 = ND,
    FRACTAL_Z + FRACTAL_Z + FRACTAL_Z = ND,
    FRACTAL_Z_3D + FRACTAL_Z_3D + FRACTAL_Z_3D = ND.

        for example:
        inputs:
            x          shape = [16, 16, 16, 16] format = "FRACTAL_Z"
            y          shape = [16, 16, 16, 16] format = "FRACTAL_Z"
            weight     shape = [16, 16, 16, 16] format = "FRACTAL_Z"
        outputs:
            output     shape = [1] format = "ND"

    2.when the reduction mode is none
    The Op can support
    ND + ND + ND = ND,
    NC1HWC0 + NC1HWC0 + NC1HWC0 = NC1HWC0,
    NDC1HWC0 + NDC1HWC0 + NDC1HWC0 = NDC1HWC0,
    FRACTAL_Z + FRACTAL_Z + FRACTAL_Z = FRACTAL_Z,
    FRACTAL_Z_3D + FRACTAL_Z_3D + FRACTAL_Z_3D = FRACTAL_Z_3D.
    """
    is_support_hd = False
    is_support_fz = False
    support_ori_format = \
        util_common.get_fused_format_str(["N", "D", "H", "W", "C"]) \
        + util_common.get_fused_format_str(["N", "H", "W", "C"])
    input_ori_shape = x.get("ori_shape")
    input_ori_format = x.get("ori_format")
    shape_hd_c0 = Constant.SHAPE_ALIGNED
    shape_fz_c0 = Constant.SHAPE_ALIGNED
    shape_fz_n = Constant.SHAPE_ALIGNED

    if input_ori_format in support_ori_format and len(input_ori_shape) == len(input_ori_format):
        if input_ori_shape[input_ori_format.index("C")] % shape_fz_c0 == 0 \
                and input_ori_shape[input_ori_format.index("N")] % shape_fz_n == 0:
            is_support_fz = True
        elif input_ori_shape[input_ori_format.index("C")] % shape_hd_c0 == 0:
            is_support_hd = True

    if reduction in ("none",):
        is_support_hd = True
        is_support_fz = True

    if not tbe_platform.api_check_support("tik.vadd", "float32"):
        dtype_base = ["float16"]
    else:
        dtype_base = ["float16", "float"]

    dtype_base_out = dtype_base[:]
    format_base_out = ["ND"] * len(dtype_base)

    if is_support_hd:
        dtype_base_out = dtype_base_out + dtype_base
        other_format = "NDC1HWC0" if len(input_ori_shape) == Constant.THREE_DIMS_SHAPE_LENS else "NC1HWC0"
        format_base_out = format_base_out + [other_format] * len(dtype_base)
    if is_support_fz:
        dtype_base_out = dtype_base_out + dtype_base
        other_format = "FRACTAL_Z_3D" if len(input_ori_shape) == Constant.THREE_DIMS_SHAPE_LENS else "FRACTAL_Z"
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


# 'pylint: disable=invalid-name,too-many-arguments,too-many-branches
# 'pylint: disable=unused-argument,too-many-locals,too-many-statements
@register_operator_compute("BinaryCrossEntropy", op_mode="dynamic", support_fusion=only_static_support,
                           support_bfp16=True)
def binary_cross_entropy_compute(x, y, weight, output, axis,
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
    if ori_dtype == "float16" and \
            tbe_platform.api_check_support("tbe.dsl.vmul", "float32") and \
            tbe_platform.api_check_support("tbe.dsl.vlog", "float32"):
        x = tbe.cast_to(x, "float32")
        y = tbe.cast_to(y, "float32")
        if weight is not None:
            weight = tbe.cast_to(weight, "float32")
        trans_dtype = "float32"

    if ori_dtype == "float32" and \
            not tbe_platform.api_check_support("tbe.dsl.vlog", "float32"):
        x = tbe.cast_to(x, "float16")
        y = tbe.cast_to(y, "float16")
        if weight is not None:
            weight = tbe.cast_to(weight, "float16")
        trans_dtype = "float16"

    const_one = tvm.const(1, trans_dtype)
    const_neg_one = tvm.const(-1, trans_dtype)
    # calcu value : y * log(x)
    if not tbe_platform.api_check_support("tik.vcopy"):
        x = tbe.vmaxs(x, tvm.const(Constant.SCALAR_EPS, trans_dtype))
    x_log_tmp = tbe.vlog(x, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)
    if tbe_platform.api_check_support("tik.vcopy"):
        x_log_tmp = tbe.vmaxs(x_log_tmp, tvm.const(Constant.MIN_VALUE, x_log_tmp.dtype))
    data_mul1 = tbe.vmul(x_log_tmp, y)
    # calcu value : (1-y) * log(1-x)
    x_neg_tmp = tbe.vmuls(x, const_neg_one)
    x1_tmp = tbe.vadds(x_neg_tmp, const_one)
    y_neg_tmp = tbe.vmuls(y, const_neg_one)
    y1_tmp = tbe.vadds(y_neg_tmp, const_one)
    if not tbe_platform.api_check_support("tik.vcopy"):
        x1_tmp = tbe.vmaxs(x1_tmp, tvm.const(Constant.SCALAR_EPS, trans_dtype))
    x1_log_tmp = tbe.vlog(x1_tmp, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)
    if tbe_platform.api_check_support("tik.vcopy"):
        x1_log_tmp = tbe.vmaxs(x1_log_tmp, tvm.const(Constant.MIN_VALUE, x1_log_tmp.dtype))
    data_mul2 = tbe.vmul(x1_log_tmp, y1_tmp)
    # calcu value : y * log(x) + (1-y) * log(1-x)
    data_sum = tbe.vadd(data_mul1, data_mul2)
    # calcu value : -(y * log(x) + (1-y) * log(1-x))
    result = tbe.vmuls(data_sum, const_neg_one)

    if weight is not None:
        result = tbe.vmul(result, weight)
    calc_dtype = trans_dtype
    if reduction == "mean":
        reduce_elts = get_cof(axis["value"], shape)
        if isinstance(reduce_elts, float):
            if math.isclose(reduce_elts, 0.0):
                result = tbe.reduce_mean(result, axis=axis["value"], keepdims=False)
                result = tbe.cast_to(result, ori_dtype)
                return result
            else:
                cof = reduce_elts ** (-1)
                cof = tvm.const(cof, dtype=calc_dtype)
        else:
            cof = tbe.var("cof", dtype=calc_dtype)
            if calc_dtype == "float16":
                tbe.var("cof_empty", dtype=calc_dtype)
            tbe_context.get_context().add_compile_info("reduce_mean_cof_dtype", calc_dtype)
        result = tbe.vmuls(result, cof)
        result = tbe.reduce_sum(result, axis=axis["value"], keepdims=False)
    elif reduction == "sum":
        result = tbe.reduce_sum(result, axis=axis["value"], keepdims=False)
    elif reduction == "none":
        pass

    result = tbe.cast_to(result, ori_dtype)

    return result


# 'pylint: disable=too-many-statements
@register_operator("BinaryCrossEntropy")
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
    if reduction not in ("mean", "sum", "none"):
        rule_desc = "reduction type should in mean/sum/none"
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule_desc,
                                                          "reduction", reduction)

    predict_shape = x.get("shape")
    predict_dtype = x.get("dtype")
    x["rel_pos_to_reduce"] = "before"

    target_shape = y.get("shape")
    target_dtype = y.get("dtype")
    y["rel_pos_to_reduce"] = "before"

    dtype_list = ("bfloat16", "float16", "float32")
    para_check.check_dtype(predict_dtype, dtype_list, param_name="x")
    para_check.check_dtype(target_dtype, dtype_list, param_name="y")
    shape_util.compare_tensor_dict_key(x, y, "dtype")

    if weight is not None:
        weight_shape = weight.get("shape")
        weight_dtype = weight.get("dtype")
        para_check.check_dtype(weight_dtype, dtype_list, param_name="weight")
        shape_util.compare_tensor_dict_key(x, weight, "dtype")
        weight["rel_pos_to_reduce"] = "before"

    tbe_context.get_context().add_compile_info("reduction", reduction)

    if is_unknown_rank_input(x):
        input_axis = {"shape": [-1], "rel_pos_to_reduce": "axis"}
    else:
        axis = []
        for i, _ in enumerate(predict_shape):
            axis.append(i)
        input_axis = {"shape": [len(axis), ], "value": axis, "rel_pos_to_reduce": "axis"}

    schedules, tensors = [], []
    if reduction != "none" and weight is not None:
        ins = classify([x, y, weight, input_axis], OpPatternMode.REDUCE, {"keepdims": False})
        for (_predict_shape, _target_shape, _weight_shape, _axis) in ins:
            with tbe.compute():
                predict_shape, target_shape, weight_shape = shape_util.variable_shape([_predict_shape,
                                                                                       _target_shape,
                                                                                       _weight_shape,
                                                                                       _axis], op_mode="reduce")[0:3]
                data_weight = tvm.placeholder(weight_shape, name="data_weight",
                                              dtype=weight_dtype)
                data_predict = tvm.placeholder(predict_shape, name="data_predict",
                                               dtype=predict_dtype)
                data_target = tvm.placeholder(target_shape, name="data_target",
                                              dtype=target_dtype)
                res = binary_cross_entropy_compute(data_predict, data_target,
                                                   data_weight, output, _axis,
                                                   reduction, kernel_name)
                tensors.append([data_predict, data_target, data_weight, res])
            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)
    elif reduction != "none" and weight is None:
        ins = classify([x, y, input_axis], OpPatternMode.REDUCE, {"keepdims": False})
        for (_predict_shape, _target_shape, _axis) in ins:
            with tbe.compute():
                predict_shape, target_shape = shape_util.variable_shape([_predict_shape,
                                                                         _target_shape,
                                                                         _axis], op_mode="reduce")[0:2]
                data_weight = None
                data_predict = tvm.placeholder(predict_shape, name="data_predict",
                                               dtype=predict_dtype)
                data_target = tvm.placeholder(target_shape, name="data_target",
                                              dtype=target_dtype)
                res = binary_cross_entropy_compute(data_predict, data_target,
                                                   data_weight, output, _axis,
                                                   reduction, kernel_name)
                tensors.append([data_predict, data_target, res])
            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)
    elif reduction == "none" and weight is not None:
        ins = classify([x, y, weight], OpPatternMode.ELEWISE)
        for (_predict_shape, _target_shape, _weight_shape) in ins:
            with tbe.compute():
                predict_shape, target_shape, weight_shape = shape_util.variable_shape([_predict_shape,
                                                                                       _target_shape,
                                                                                       _weight_shape])[0:3]
                data_weight = tvm.placeholder(weight_shape, name="data_weight",
                                              dtype=weight_dtype)
                data_predict = tvm.placeholder(predict_shape, name="data_predict",
                                               dtype=predict_dtype)
                data_target = tvm.placeholder(target_shape, name="data_target",
                                              dtype=target_dtype)
                res = binary_cross_entropy_compute(data_predict, data_target,
                                                   data_weight, output, [],
                                                   reduction, kernel_name)
                tensors.append([data_predict, data_target, data_weight, res])
            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)
    elif reduction == "none" and weight is None:
        ins = classify([x, y], OpPatternMode.ELEWISE)
        for (_predict_shape, _target_shape) in ins:
            with tbe.compute():
                predict_shape, target_shape = shape_util.variable_shape([_predict_shape,
                                                                         _target_shape])[0:2]
                data_weight = None
                data_predict = tvm.placeholder(predict_shape, name="data_predict",
                                               dtype=predict_dtype)
                data_target = tvm.placeholder(target_shape, name="data_target",
                                              dtype=target_dtype)
                res = binary_cross_entropy_compute(data_predict, data_target,
                                                   data_weight, output, [],
                                                   reduction, kernel_name)
                tensors.append([data_predict, data_target, res])
            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
