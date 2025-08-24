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
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import error_manager_vector
from impl.util.util_common import is_unknown_rank_input
from tbe.dsl.compute.max_pool_grad_with_argmax import max_pool_grad_with_argmax_3d
from tbe.dsl.compute.max_pool_grad_with_argmax import max_pool_grad_with_argmax


# 'pylint: disable=locally-disabled,too-many-arguments,invalid-name,huawei-too-many-arguments
@register_operator("MaxPoolWithArgmax")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_LIST_INT, para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def max_pool_grad_with_argmax(x, grad, argmax, y, ksize, strides, padding, kernel_name="max_pool_grad_with_argmax"):
    """
    the main function of the maxpoolGradWithArgmax
    Parameters
    ----------
    x: input of maxpool, useless for maxpool gard
    grad: input of maxpoolgard or output of maxpool
    argmax:output of maxpool mask or index
    y: output of maxpoolgard
    ksize: kernel or windows size,minimum length is 4,
           just like [1, poolingWindowH, poolingWindowW, 1]
    strides: stride , minimum length is 4, just like [1, poolingStrideH, poolingStrideW, 1]
    padding: pad mode, just support "SAME" or "VALID"
    kernel_name: kernel_name
    Returns
    -------
    tik_instance: tik_instance
    """

    dtype_x = x.get("dtype").lower()
    dtype_grad = grad.get("dtype").lower()
    dtype_argmax = "uint1" if argmax.get("format").lower() == "nc1hwc0" else argmax.get("dtype").lower()
    dtype_y = y.get("dtype").lower()

    para_check.check_dtype(dtype_x, ["float16", "float32"], param_name="x")
    para_check.check_dtype(dtype_grad, ["float16", "float32"], param_name="grad")
    para_check.check_dtype(dtype_y, ["float16", "float32"], param_name="y")

    transfer_ksize = [ksize[0], ksize[3], ksize[1], ksize[2]] if ksize else None
    transfer_strides = [strides[0], strides[3], strides[1], strides[2]] if strides else None
    input_format = grad.get("format")

    if padding not in ["SAME", "VALID"]:
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "padding", ["SAME", "VALID"], padding)

    extra_params = {"ksize": transfer_ksize,
                    "strides": transfer_strides,
                    "padding_mode": padding,
                    "is_2d": True,
                    "input_format": input_format}
    ins = classify([x, grad, argmax],
                   OpPatternMode.POOLING_GRAD_WITH_ARG, extra_params)

    schedules = []
    tensors = []
    for (_x, _grad, _argmax, _ksize, _strides, _paddings, _format) in ins:
        with tbe.compute():
            shape_x, shape_grad, shape_argmax, ksize, strides, pads = shape_util.variable_shape(
                [_x, _grad, _argmax, _ksize, _strides, _paddings, _format], op_mode=OpPatternMode.POOLING_GRAD_WITH_ARG)
            input_x = tvm.placeholder(shape_x, name="input_x", dtype=dtype_x)
            input_grad = tvm.placeholder(
                shape_grad, name="input_grad", dtype=dtype_grad)
            input_argmax = tvm.placeholder(
                shape_argmax, name="input_argmax", dtype=dtype_argmax)
            param_dict = {"ksize": ksize, "strides": strides, "pads": pads,
                          "input_dtype": dtype_grad, "output_dtype": dtype_y, "input_format": _format.get("format")}
            if _format.get("format") == "NC1HWC0":
                res = max_pool_grad_with_argmax_3d(input_x, input_grad, input_argmax, param_dict)
            else:
                res = max_pool_grad_with_argmax(input_x, input_grad, input_argmax, param_dict)
            tensors.append([input_x, input_grad, input_argmax, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    tbe_context.get_context().add_compile_info("ksize_attr_idx", 0)
    tbe_context.get_context().add_compile_info("strides_attr_idx", 1)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)