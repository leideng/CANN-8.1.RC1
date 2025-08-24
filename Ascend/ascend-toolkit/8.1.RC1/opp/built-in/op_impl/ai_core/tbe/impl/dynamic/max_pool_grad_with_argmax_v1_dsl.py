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


# 'pylint: disable=unused-argument
# 'pylint: disable=invalid-name,too-many-arguments,useless-super-delegation,super-with-arguments
# 'pylint: disable=too-many-locals,too-many-branches,too-many-statements,too-many-return-statements,consider-using-in
def max_pool_grad_with_argmax_v1_dsl(x, grad, argmax, y, ksize, strides, pads,
                                     dilation=(1, 1, 1, 1), ceil_mode=False,
                                     kernel_name="max_pool_grad_with_argmax_v1"):
    dtype_x = x.get("dtype").lower()
    dtype_grad = grad.get("dtype").lower()
    dtype_argmax = "uint1" if argmax.get("format").lower() == "nc1hwc0" else argmax.get("dtype").lower()
    dtype_y = y.get("dtype").lower()

    para_check.check_dtype(dtype_x, ["float16", "float32"], param_name="x")
    para_check.check_dtype(dtype_grad, ["float16", "float32"], param_name="grad")
    para_check.check_dtype(dtype_y, ["float16", "float32"], param_name="y")

    transfer_ksize = [ksize[0], ksize[3], ksize[1], ksize[2]] if ksize else None
    transfer_strides = [strides[0], strides[3], strides[1], strides[2]] if strides else None
    transfer_pads = [pads[0], pads[3], pads[1], pads[2]] if pads else None
    input_format = grad.get("format")

    extra_params = {"ksize": transfer_ksize,
                    "strides": transfer_strides,
                    "pads": transfer_pads,
                    "is_2d": True,
                    "ceil_mode": ceil_mode,
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
    tbe_context.get_context().add_compile_info("pads_attr_idx", 2)
    tbe_context.get_context().add_compile_info("dilations_attr_idx", 4)
    tbe_context.get_context().add_compile_info("ceil_mode_idx", 5)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
