# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
dynamic kl_div_v2
"""
import math
from functools import reduce
from typing import List
from enum import Enum
import tbe.dsl as dsl
from tbe.common import platform as tbe_platform
from impl.util import util_select_op_base
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import OpImplMode
from impl.util.reduce_pattern_adapter import ReducePattern
from impl.util.util_compute import only_static_support


class ReduceMode(Enum):
    """
    The class for reduction
    """
    NONE_MODE = 0
    MEAN_MODE = 1
    BATCHMEAN_MODE = 2
    SUM_MODE = 3


class Constant:
    """
    The class for constant
    """
    CONST_ZERO = 0
    CONST_ONE = 1


def product(lst):
    return reduce(lambda x, y: x * y, lst)


def calc_coeff(tensor_shape: List[int], reduction):
    # none mode, sum mode does not need coeff
    if reduction in ["none", "sum"]:
        return None
    # binary mode
    if -2 in tensor_shape or -1 in tensor_shape or len(tensor_shape) == 0:
        return None
    # static mode
    if reduction == "batchmean":
        batch_size = tensor_shape[0]
        return 1.0 / batch_size
    elif reduction == "mean":
        total = product(tensor_shape)
        return 1.0 / total
    else:
        raise RuntimeError("reduction should be in batchmean, mean, sum, none")
    return None


# 'pylint: disable=too-many-arguments,unused-argument,too-many-locals
def op_select_format(input_x, input_target, output_y, reduction, log_target=False, kernel_name="kl_div_v2"):
    """
    select format dynamically
    op_select_format support desc:

    1.When reduction is "none".

    The output format is the same as the input.

        for example:
        inputs:
            input_x         shape = [16, 16, 16, 16, 16] format = "NC1HWC0"
            input_target      shape = [16, 16, 16, 16, 16] format = "NC1HWC0"
        outputs:
            output_y         shape = [16, 16, 16, 16, 16] format = "NC1HWC0"

    2.In other scenes, all output_y only support ND.

        for example:
        inputs:
            input_x         shape = [16, 16, 16, 16, 16] format = "NC1HWC0"
            input_target      shape = [16, 16, 16, 16, 16] format = "NC1HWC0"
        outputs:
            output_y         shape = [1,] format = "ND"

    Parameters
    ----------
    input_x : dict
        shape and dtype of input_x, dtype only support fp16 and fp32.
    input_target : dict
        shape and dtype of input_target.Shape and dtype must be same as input_x
    output_y : dict
        shape and dtype of output.Dtype must be same as input_x
    reduction: str
        reduction="batchmean" or "sum" or "none".
        "batchmean": the sum of the output will be divided by the batchsize
        "sum": the output will be summed
        "none": no reduction will be applied
    kernel_name : str
        cce kernel name, default value is "kl_div_v2"

    Returns
    ------
    param_dynamic_in_json : dict
    supported format and dtype
    """
    if tbe_platform.intrinsic_check_support("Intrinsic_vconv", "bf162f32"):
        input_format_list = ["ND", "NC1HWC0", "FRACTAL_Z", "HWCN", "FRACTAL_NZ", "C1HWNCoC0", 
                             "ND", "NC1HWC0", "FRACTAL_Z", "HWCN", "FRACTAL_NZ", "C1HWNCoC0",
                             "ND", "NC1HWC0", "FRACTAL_Z", "HWCN", "FRACTAL_NZ", "C1HWNCoC0"]
        dtype_list = ["float", "float", "float", "float", "float", "float",
                      "float16", "float16", "float16", "float16", "float16", "float16",
                      "bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16"]
    else:
        input_format_list = ["ND", "NC1HWC0", "FRACTAL_Z", "HWCN", "FRACTAL_NZ", "C1HWNCoC0",
                            "ND", "NC1HWC0", "FRACTAL_Z", "HWCN", "FRACTAL_NZ", "C1HWNCoC0"]
        dtype_list = ["float", "float", "float", "float", "float", "float",
                      "float16", "float16", "float16", "float16", "float16", "float16"]

    input_format = ",".join(input_format_list)
    if reduction == "none":
        output_format = input_format
    else:
        output_format = ','.join(["ND"] * len(input_format_list))

    dtype = ','.join(dtype_list)
    input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                           datatype=dtype,
                                           format=input_format,
                                           unknownshape_format=input_format)
    input1 = util_select_op_base.gen_param(classify="input1", name="target",
                                           datatype=dtype,
                                           format=input_format,
                                           unknownshape_format=input_format)
    output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                            datatype=dtype,
                                            format=output_format,
                                            unknownshape_format=output_format)
    param_list = [input0, input1, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)
 
    return param_dynamic_in_json


# 'pylint: disable=too-many-arguments,unused-argument,too-many-locals
@register_operator_compute("KLDivV2", op_mode="dynamic", support_fusion=False)
def kl_div_v2_compute(input_x, input_target, log_target, axes, output_y, reduction,
                      coeff=None, kernel_name="kl_div_v2"):
    """
    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    input_target : TVM tensor
        the placeholder of input_target
    output_y : dict
        shape and dtype of output, should be same shape and type as input_x
    reduction: str
        Specifies the reduction to apply to the output:
        reduction="batchmean" or reduction="sum".
        "batchmean": the sum of the output will be divided by the batchsize
        "sum": the output will be summed
    batch_size: int
        Equal to the first dimension value of the input shape.
    kernel_name : str
        cce kernel name, default value is "kl_div_v2"

    Returns
    ------
    compute result of kl_div_v2
    """
    input_x_dtype = input_x.dtype

    # BF16 adapter
    if input_x_dtype in ["bfloat16", "float16"]:
        input_x = tbe.cast_to(input_x, "float32")
    if input_target.dtype in ["bfloat16", "float16"]:
        input_target = tbe.cast_to(input_target, "float32")

    if not log_target:
        target_dtype = input_target.dtype
        compare_one = tbe.vcmp(input_target, tvm.const(Constant.CONST_ZERO, target_dtype), "eq")
        input_target_tmp = tbe.vsel(compare_one, tvm.const(Constant.CONST_ONE, target_dtype), input_target)

        log_input_target = tbe.vlog(input_target_tmp)
        tmp_result = tbe.vsub(log_input_target, input_x)
        loss_pointwise = tbe.vmul(input_target, tmp_result)
    else:
        tmp_target = tbe.vsub(input_target, input_x)
        exp_target = tbe.vexp(input_target)
        loss_pointwise = tbe.vmul(exp_target, tmp_target)

    # process parameter reduction
    if reduction == "batchmean":
        tbe_context.get_context().add_compile_info("reduce_mode", ReduceMode.BATCHMEAN_MODE.value)
        if coeff is None:
            coefficient = tbe.var("coefficient", dtype=loss_pointwise.dtype)
            tbe_context.get_context().add_compile_info("coefficient_is_unknown", True)
        else:
            coefficient = coeff
        output_tmp = tbe.vmuls(loss_pointwise, coefficient)
        final_res = tbe.reduce_sum(output_tmp, axis=axes, keepdims=False)
    elif reduction == "mean":
        tbe_context.get_context().add_compile_info("reduce_mode", ReduceMode.MEAN_MODE.value)
        if coeff is None:
            coefficient = tbe.var("coefficient", dtype=loss_pointwise.dtype)
            tbe_context.get_context().add_compile_info("coefficient_is_unknown", True)
        else:
            coefficient = coeff
        output_tmp = tbe.vmuls(loss_pointwise, coefficient)
        final_res = tbe.reduce_sum(output_tmp, axis=axes, keepdims=False)
    elif reduction == "sum":
        tbe_context.get_context().add_compile_info("reduce_mode", ReduceMode.SUM_MODE.value)
        final_res = tbe.reduce_sum(loss_pointwise, axis=axes, keepdims=False)
    elif reduction == "none":
        tbe_context.get_context().add_compile_info("reduce_mode", ReduceMode.NONE_MODE.value)
        final_res = loss_pointwise
    else:
        error_manager_vector.raise_err_input_value_invalid(kernel_name, 'reduction',
                                                           ("batchmean", "mean", "sum", "none"), reduction)

    if input_x_dtype == "bfloat16":
        final_res = dsl.round(final_res, input_x_dtype)
    elif input_x_dtype == "float16":
        final_res = tbe.cast_to(final_res, input_x_dtype)

    return final_res


def _check_parameter(input_x, input_target):
    """
    Parameters
    ----------
    input_x : dict
        shape and dtype of input_x
    input_target : dict
        shape and dtype of input_target.Shape and dtype must be same as input_x
    Returns
    ------
    None
    """
    # check input tensor data_type
    dtype_x = input_x.get("dtype").lower()
    dtype_target = input_target.get("dtype").lower()
    check_list = ("bfloat16", "float16", "float32")
    para_check.check_dtype(dtype_x, check_list, param_name="input_x")
    if dtype_x != dtype_target:
        error_manager_vector.raise_err_inputs_dtype_not_equal('kl_div_v2', 'input_x', 'input_target', dtype_x,
                                                              dtype_target)


# 'pylint: disable =invalid-name
@register_operator("KLDivV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def kl_div_v2(input_x, input_target, output_y, reduction="mean", log_target=False, kernel_name="kl_div_v2"):
    """
    Calcuate Kullback-Leibler divergence.

    output_pos = input_target * (log(input_target) - input_x)
    output = where(input_target > 0, output_pos, zeros)
    reduced = reduce_sum_all(output)
    if reduction = "batchmean":
        final_res = reduced / input.dim[0]
    else:
        final_res = reduced
    Parameters
    ----------
    input_x : dict
        shape and dtype of input_x, dtype only support fp16 and fp32.
    input_target : dict
        shape and dtype of input_target.Shape and dtype must be same as input_x
    output_y : dict
        shape and dtype of output.Dtype must be same as input_x
    reduction: str
        Specifies the reduction to apply to the output:
        reduction="batchmean/sum/none".
        "batchmean": the sum of the output will be divided by the batchsize
        "sum": the output will be summed
        "none": no reduction will be applied 
    kernel_name : str
        cce kernel name, default value is "kl_div_v2"

    Returns
    ------
    None
    """
    # check input parameter
    _check_parameter(input_x, input_target)

    input_x["rel_pos_to_reduce"] = "before"
    input_target["rel_pos_to_reduce"] = "before"
    input_shape = input_x.get("shape")

    x_dtype = input_x.get("dtype").lower()
    schedules, tensors = [], []

    if reduction in ["sum", "batchmean", "mean"]:
        # calculate coeff
        coeff = calc_coeff(input_shape, reduction)
        # gen reduce axis input dict
        input_axis = {"shape": [-1], "value": [], "rel_pos_to_reduce": "axis"}
        # gen extra_params for reduce pattern
        extra_params = dict()
        # set KEEP_DIMS flag
        extra_params.update(ReducePattern.KEEP_DIMS_FALSE)
        # set all reduce pattern
        extra_params.update(ReducePattern.REDUCE_MODE_REDUCE_ALL)
        ins = classify([input_x, input_target, input_axis], OpPatternMode.REDUCE, extra_params)
        for (x, target, axis) in ins:
            with tbe.compute():
                x_shape, target_shape = shape_util.variable_shape([x, target, axis], op_mode="reduce")[0:2]
                tensor_x = tvm.placeholder(x_shape, x_dtype, "tensor_x")
                tensor_target = tvm.placeholder(target_shape, x_dtype, "tensor_target")
                res = kl_div_v2_compute(tensor_x, tensor_target, log_target, axis.get("value"), output_y, reduction,
                                        coeff, kernel_name)
                tensors.append([tensor_x, tensor_target, res])
            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)
    else:
        ins = classify([input_x, input_target], OpPatternMode.ELEWISE)
        for (x, target) in ins:
            with tbe.compute():
                x_shape, target_shape = shape_util.variable_shape([x, target])[0:2]
                tensor_x = tvm.placeholder(x_shape, x_dtype, "tensor_x")
                tensor_target = tvm.placeholder(target_shape, x_dtype, "tensor_target")
                res = kl_div_v2_compute(tensor_x, tensor_target, log_target, None, output_y, "none", None, kernel_name)
                tensors.append([tensor_x, tensor_target, res])
            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
