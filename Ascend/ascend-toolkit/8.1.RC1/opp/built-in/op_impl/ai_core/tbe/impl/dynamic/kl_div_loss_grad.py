#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
kl_div_loss_grad
"""
from impl.util.platform_adapter import tbe
from impl.util import util_select_op_base
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpImplMode
from impl.util.util_compute import only_static_support
from impl.util.util_soc_common import after_v200

# 'pylint: disable=redefined-builtin,too-many-locals,consider-using-in
# 'pylint: disable=unused-variable,invalid-name,too-many-arguments,unused-argument
PREVENTS_ZERO_FP32 = 1.18e-38
PREVENTS_ZERO_FP16 = 1.18e-7


def _check_parameter(grad, input, target):
    """
    Parameters
    ----------
    grad : tvm.Tensor
        shape and dtype of grad, only support float16, float32
    input : tvm.Tensor
        shape and dtype of input, only support float16, float32
    target : tvm.Tensor
        shape and dtype of target, should be broadcast shape and type as input

    Returns
    ------
    None
    """
    shape_grad = grad.get("shape")
    para_check.check_shape(shape_grad, param_name="grad")

    # check input tensor data_type
    dtype_grad = grad.get("dtype").lower()
    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(dtype_grad, check_list, param_name="grad")


def op_select_format(input_grad, input_input, input_target, output_y,
                     reduction, log_target=False, kernel_name="KlDivLossGrad"):
    """
    select format dynamically
    op_select_format support desc:

    1.When reduction is "none".

    The output format is the same as the input .

        for example:
        inputs:
            input_grad        shape = [16, 16, 16, 16, 16] format = "NC1HWC0"
            input_input       shape = [16, 16, 16, 16, 16] format = "NC1HWC0"
            input_target      shape = [16, 16, 16, 16, 16] format = "NC1HWC0"
        outputs:
            output_y          shape = [16, 16, 16, 16, 16] format = "NC1HWC0"

    2.In other scenes, The input "grad" only support ND, output_y is the same as the input_input, input_target.

        for example:
        inputs:
            input_grad        shape = [1,] format =  "ND"
            input_input       shape = [16, 16, 16, 16, 16] format = "NC1HWC0"
            input_target      shape = [16, 16, 16, 16, 16] format = "NC1HWC0"
        outputs:
            output_y          shape = [16, 16, 16, 16, 16] format = "NC1HWC0"

    Parameters
    ----------
    input_grad : tvm.Tensor
        shape and dtype of input_x, dtype only support fp16 and fp32.
    input_input : tvm.Tensor
        shape and dtype of input_x, dtype only support fp16 and fp32.
    input_target : tvm.Tensor
        shape and dtype of input_target.Shape and dtype must be same as input_input
    output_y : tvm.Tensor
        shape and dtype of output.Dtype must be same as  input_input
    reduction: str
        reduction="batchmean" or "sum" or "none" or "mean".
        "batchmean": the sum of the output will be divided by the batchsize
        "sum": the output will be summed
        "none": no reduction will be applied
        "mean": the output will be divided by the number of elements in the output
    kernel_name : str
        cce kernel name, default value is "KlDivLossGrad"

    Returns
    ------
    param_dynamic_in_json : tvm.Tensor
    supported format and dtype
    """
    version_info = tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION)
    input_format_list = []
    dtype_list = []
    if version_info == "Ascend910B" or version_info == "Ascend910_93":
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
    output_format = input_format
    if reduction == "none":
        input_format_grad = input_format
    else:
        input_format_grad = ','.join(["ND"] * len(input_format_list))
    dtype = ','.join(dtype_list)
    input0 = util_select_op_base.gen_param(classify="input0", name="grad",
                                           datatype=dtype,
                                           format=input_format_grad,
                                           unknownshape_format=input_format_grad)
    input1 = util_select_op_base.gen_param(classify="input1", name="input",
                                           datatype=dtype,
                                           format=input_format,
                                           unknownshape_format=input_format)
    input2 = util_select_op_base.gen_param(classify="input2", name="target",
                                           datatype=dtype,
                                           format=input_format,
                                           unknownshape_format=input_format)
    output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                            datatype=dtype,
                                            format=output_format,
                                            unknownshape_format=output_format)
    param_list = [input0, input1, input2, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


@register_operator_compute("KlDivLossGrad", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def kl_div_loss_grad_compute(grad,
                             input,
                             target,
                             y,
                             input_shape_ori,
                             reduction,
                             log_target=False,
                             kernel_name="KlDivLossGrad"):
    """
    Backpropagate the input parameter of kl_div to find the gradient,
    y = -grad * target

    Parameters
    --------
    grad : tvm.Tensor
        shape and dtype of grad, only support float16, float32
    input : tvm.Tensor
        shape and dtype of input, only support float16, float32
    target : tvm.Tensor
        shape and dtype of target, should be broadcast shape and type as input
    reduction : str
        Specifies the reduction to apply to the output: 'none' | 'batchmean' | 'sum' | 'mean'.
            'none': no reduction will be applied;
            'batchmean': the sum of the output will be divided by the batchsize;
            'sum': the output will be summed;
            'mean': the output will be divided by the number of elements in the output;
        Default: 'mean'
    log_target : bool
        A flag indicating whether target is passed in the log space.
        It is recommended to pass certain distributions (like softmax)
        in the log space to avoid numerical issues caused by explicit log.
        Default: False

    Returns
    -------
    None
    """
    input_dtype = input.dtype
    grad_shape = grad.shape
    input_shape = input.shape
    target_shape = target.shape
    grad_shape, input_shape, target_shape, shape_max = shape_util.unify_broadcast_shapes(
        [grad_shape, input_shape, target_shape])
    grad = tbe.broadcast(grad, shape_max)
    input = tbe.broadcast(input, shape_max)
    target = tbe.broadcast(target, shape_max)

    if log_target:
        target = tbe.vexp(target)

    res_tmp1 = tbe.vmul(target, grad)
    res_tmp2 = tbe.vmuls(res_tmp1, -1)

    element_total_cof = 1.0
    if int(-1) in input_shape_ori or int(-2) in input_shape_ori:
        element_total_cof = tbe.var("cof", dtype=input_dtype)
    else:
        for i in input_shape_ori:
            element_total_cof *= i
        element_total_cof = element_total_cof ** (-1)
        element_total_cof = tvm.const(element_total_cof, dtype=input_dtype)

    cof_batch_size = input_shape_ori[0] * 1.0
    if int(-1) in input_shape_ori or int(-2) in input_shape_ori:
        cof_batch_size = tbe.var("cof_else", dtype=input_dtype)
    else:
        cof_batch_size = cof_batch_size ** (-1)
        cof_batch_size = tvm.const(cof_batch_size, dtype=input_dtype)

    if input_dtype == "float16":
        tbe.var("cof_empty", dtype=input_dtype)
    tbe_context.get_context().add_compile_info("reduce_mean_cof_dtype", input_dtype)

    if reduction == "mean":
        res_tmp2 = tbe.vmuls(res_tmp2, element_total_cof)
        res = res_tmp2
    elif reduction == "sum" or reduction == "none":
        res = res_tmp2
    elif reduction == "batchmean":
        res_tmp2 = tbe.vmuls(res_tmp2, cof_batch_size)
        res = res_tmp2
    else:
        error_manager_vector.raise_err_input_value_invalid(kernel_name, 'reduction',
                                                           ("batchmean", "sum", "none", "mean"), reduction)
    return res


@register_operator("KlDivLossGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def kl_div_loss_grad(grad, input, target, y, reduction="mean",
                     log_target=False, kernel_name="KlDivLossGrad"):
    """
    algorithm: kl_div_loss_grad


    Parameters
    --------
    grad : tvm.Tensor
        shape and dtype of grad, only support float16, float32, double
    input : tvm.Tensor
        shape and dtype of input, only support float16, float32, double
    target : tvm.Tensor
        shape and dtype of target, should be broadcast shape and type as input
    reduction : str
        Specifies the reduction to apply to the output: 'none' | 'batchmean' | 'sum' | 'mean'.
            'none': no reduction will be applied;
            'batchmean': the sum of the output will be divided by the batchsize;
            'sum': the output will be summed;
            'mean': the output will be divided by the number of elements in the output;
        Default: 'mean'
    log_target : bool
        A flag indicating whether target is passed in the log space.
        It is recommended to pass certain distributions (like softmax)
        in the log space to avoid numerical issues caused by explicit log.
        Default: False
    kernel_name : str
        cce kernel name, default value is add

    Returns
    -------
    None
    """
    _check_parameter(grad, input, target)
    # get the shape and type
    schedules, tensors = [], []
    input_shape_ori = input.get("shape")
    reduction_list = ("none", "batchmean", "sum", "mean")
    ins = classify([grad, input, target], OpPatternMode.ELEWISE_WITH_BROADCAST)
    input_dtype = input.get("dtype")
    grad_dtype = grad.get("dtype")
    target_dtype = target.get("dtype")

    if reduction not in reduction_list:
        raise RuntimeError("The reduction ({}) is not supported".format(reduction))

    for (input_grad, input_input, input_target) in ins:
        with tbe.compute():

            grad_shape, input_shape, target_shape = shape_util.variable_shape([input_grad,
                                                                               input_input, input_target])
            target_data = tvm.placeholder(target_shape, name="target_data", dtype=target_dtype)
            grad_data = tvm.placeholder(grad_shape, name="grad_data", dtype=grad_dtype)
            input_data = tvm.placeholder(input_shape, name="input_data", dtype=input_dtype)

            res = kl_div_loss_grad_compute(grad_data, input_data, target_data,
                                           y, input_shape_ori, reduction, log_target, kernel_name)
            tensors.append([grad_data, input_data, target_data, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
