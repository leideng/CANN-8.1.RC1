#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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
import functools
import te.lang.cce as tbe
from tbe import tvm
from te.utils import para_check
from te.utils import shape_util
import te.platform as tbe_platform


# 'pylint: disable=redefined-builtin,too-many-locals,consider-using-in
# 'pylint: disable=unused-variable,invalid-name,too-many-arguments,unused-argument
@tbe_platform.fusion_manager.fusion_manager.register("KlDivLossGrad")
def kl_div_loss_grad_compute(grad, input, target, reduction="mean", log_target=False):
    """
    Backpropagate the input parameter of kl_div to find the gradient,
    y = -grad * target

    Parameters
    --------
    grad : dict
        shape and dtype of grad, only support float16, float32, double
    input : dict
        shape and dtype of input, only support float16, float32, double
    target : dict
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
    input_shape = shape_util.shape_to_list(input.shape)
    grad_shape = shape_util.shape_to_list(grad.shape)
    if input_shape != grad_shape:
        grad = tbe.broadcast(grad, input_shape)

    input_num = 1.0 / input_shape[0]

    if log_target:
        target = tbe.vexp(target)

    nothing = tbe.vadds(input, 0.0)
    res_tmp1 = tbe.vmul(target, grad)
    res_tmp2 = tbe.vmuls(res_tmp1, -1)

    target_gt_zero = tbe.vmaxs(target, 0)

    # To be the same as the cpu, avoid dividing inf by inf as nan
    target_normalized = tbe.vmins(target_gt_zero, 1)

    if input_dtype == "float16":
        # for float16, add a small number which value is 1.18e-7, so that the
        # divisor is not equal to 0, and for accuracy, multiply by a number
        # which value is 1024.
        mul_big = tbe.vmuls(target_normalized, 1024)
        add_espmin = tbe.vadds(mul_big, 1.18e-7)
        y_espmin = tbe.vdiv(mul_big, add_espmin)
    if input_dtype == "float32":
        # for float32, add a small number which value is 1.18e-38, so that
        # the divisor is not equal to 0.
        add_espmin = tbe.vadds(target_normalized, 1.18e-38)
        y_espmin = tbe.vdiv(target_normalized, add_espmin)

    res_tmp3 = tbe.vmul(y_espmin, res_tmp2)

    if reduction == "mean" or reduction == "batchmean":
        res = tbe.vmuls(res_tmp3, input_num)
    else:
        res = res_tmp3

    return res


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
    grad : dict
        shape and dtype of grad, only support float16, float32, double
    input : dict
        shape and dtype of input, only support float16, float32, double
    target : dict
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

    shape_grad = shape_util.scalar2tensor_one(grad.get("shape"))
    input_shape = shape_util.scalar2tensor_one(input.get("shape"))
    shape_target = shape_util.scalar2tensor_one(target.get("shape"))
    para_check.check_kernel_name(kernel_name)
    para_check.check_shape(shape_grad)
    para_check.check_shape(input_shape)
    para_check.check_shape(shape_target)
    para_check.check_shape_size(shape_grad)
    para_check.check_shape_size(input_shape)
    para_check.check_shape_size(shape_target)

    check_tuple = ("float16", "float32")
    input_dtype = input.get("dtype").lower()
    para_check.check_dtype_rule(input_dtype, check_tuple)

    reduction_list = ("none", "batchmean", "sum", "mean")
    if reduction not in reduction_list:
        raise RuntimeError("The reduction ({}) is not supported".format(reduction))

    if reduction != "none":
        shape_grad = [1] * len(input_shape)
    data_grad = tvm.placeholder(shape_grad, name="data_1", dtype=input_dtype)
    data_input = tvm.placeholder(input_shape, name="data_2", dtype=input_dtype)
    data_target = tvm.placeholder(shape_target, name="data_3", dtype=input_dtype)

    res = kl_div_loss_grad_compute(data_grad, data_input, data_target, reduction, log_target)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": (data_grad, data_input, data_target, res),
              "dummy_placeholder": True}
    tbe.cce_build_code(schedule, config)
