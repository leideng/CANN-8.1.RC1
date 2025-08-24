# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
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
dynamic kl_div
"""
import math
from impl.util import util_select_op_base
from impl.util.platform_adapter import tbe_platform
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


# 'pylint: disable=too-many-arguments,unused-argument,too-many-locals
def op_select_format(input_x, input_target, output_y, reduction, kernel_name="kl_div"):
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
        cce kernel name, default value is "kl_div"

    Returns
    ------
    param_dynamic_in_json : dict
    supported format and dtype
    """
    input_format_list = ["ND", "NC1HWC0", "FRACTAL_Z", "HWCN", "FRACTAL_NZ", "C1HWNCoC0", "ND", "NC1HWC0", "FRACTAL_Z",
                         "HWCN", "FRACTAL_NZ", "C1HWNCoC0", "ND", "NC1HWC0", "FRACTAL_Z", "HWCN", "FRACTAL_NZ",
                         "C1HWNCoC0"]
    input_format = ",".join(input_format_list)
    if reduction == "none":
        output_format = input_format
    else:
        output_format = ','.join(["ND"] * len(input_format_list))

    dtype_list = ["float", "float", "float", "float", "float", "float", "float16", "float16", "float16", "float16",
                  "float16", "float16", "bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16"]
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
@register_operator_compute("KLDiv", op_mode="dynamic", support_fusion=False, support_bfp16=True)
def kl_div_compute(input_x, input_target, output_y, axis, reduction, batch_size, kernel_name="kl_div"):
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
        cce kernel name, default value is "kl_div"

    Returns
    ------
    compute result of kl_div
    """
    input_x_dtype = input_x.dtype
    log_support_fp32 = tbe_platform.api_check_support("te.lang.cce.vlog", "float32")

    if log_support_fp32 and input_x_dtype == "float32":
        # for float32, take the maximum value between input_target and 1.18e-38, so that
        # the argument of natural logarithm will be a positive number.
        log_target_max = tbe.vmaxs(input_target, tvm.const(1.18e-38, dtype=input_x_dtype))
        log_target = tbe.vlog(log_target_max, OpImplMode.HIGH_PRECISION)
    else:
        log_target_max = tbe.vmaxs(input_target, tvm.const(1.18e-7, dtype=input_x_dtype))
        log_target = tbe.vlog(log_target_max)

    tmp_result = tbe.vsub(log_target, input_x)

    output_pos = tbe.vmul(input_target, tmp_result)

    # `max(output_pos, 0)`
    target_gt_zero = tbe.vmaxs(input_target, 0)

    if input_x_dtype == "float16":
        # algrithm : `Y = X*1024/(X*1024+ESP_MIN)`
        # for float16, add a small number which value is 1.18e-7, so that the
        # divisor is not equal to 0, and for accuracy, multiply by a number
        # which value is 1024.
        target_gt_zero_fp32 = tbe.cast_to(target_gt_zero, "float32")
        mul_big = tbe.vmuls(target_gt_zero_fp32, 1024)
        add_espmin = tbe.vadds(mul_big, 1.18e-7)
        y_espmin_fp32 = tbe.vdiv(mul_big, add_espmin)
        y_espmin = tbe.cast_to(y_espmin_fp32, "float16")
    elif input_x_dtype == "float32":
        # algrithm : `Y = X/(X+ESP_MIN)`
        # for float32, add a small number which value is 1.18e-38, so that
        # the divisor is not equal to 0.
        add_espmin = tbe.vadds(target_gt_zero, 1.18e-38)
        y_espmin = tbe.vdiv(target_gt_zero, add_espmin)

    output_res = tbe.vmul(y_espmin, output_pos)

    if input_x_dtype == "float32":
        batch_size_dtype = "float32"
    else:
        batch_size_dtype = "float16"

    if batch_size >= 0:
        batch_size = batch_size if math.isclose(batch_size, 0.0) else batch_size ** (-1)
        batch_size = tvm.const(batch_size, dtype=batch_size_dtype)
    else:
        batch_size = tbe.var("cof", dtype=batch_size_dtype)
        if batch_size_dtype == "float16":
            tbe.var("cof_empty", dtype=batch_size_dtype)
        tbe_context.get_context().add_compile_info("reduce_mean_cof_dtype", batch_size_dtype)

    if reduction == "batchmean":
        tbe_context.get_context().add_compile_info("reduce_mode", 1)
        output_res = tbe.vmuls(output_res, batch_size)
        final_res = tbe.reduce_sum(output_res, axis=axis["value"])
    elif reduction == "sum":
        tbe_context.get_context().add_compile_info("reduce_mode", 1)
        final_res = tbe.reduce_sum(output_res, axis=axis["value"])
    elif reduction == "none":
        tbe_context.get_context().add_compile_info("reduce_mode", 0)
        final_res = output_res
    else:
        error_manager_vector.raise_err_input_value_invalid(kernel_name, 'reduction',
                                                           ("batchmean", "sum", "none"), reduction)

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
    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(dtype_x, check_list, param_name="input_x")
    if dtype_x != dtype_target:
        error_manager_vector.raise_err_inputs_dtype_not_equal('kl_div', 'input_x', 'input_target', dtype_x,
                                                              dtype_target)

    if dtype_x == "float32" and not tbe_platform.api_check_support("te.lang.cce.vmul", "float32"):
        error_manager_vector.raise_err_input_dtype_not_supported('kl_div', 'input_x', ('float16',), dtype_x)


# 'pylint: disable =invalid-name
@register_operator("KLDiv")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def kl_div(input_x, input_target, output_y, reduction, kernel_name="kl_div"):
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
        cce kernel name, default value is "kl_div"

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

    if reduction in ["sum", "batchmean"]:
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

                res = kl_div_compute(tensor_x, tensor_target, output_y, axis, reduction, input_shape[0], kernel_name)

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

                res = kl_div_compute(tensor_x, tensor_target, output_y, 0, "none", input_shape[0], kernel_name)

                tensors.append([tensor_x, tensor_target, res])

            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)




    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
