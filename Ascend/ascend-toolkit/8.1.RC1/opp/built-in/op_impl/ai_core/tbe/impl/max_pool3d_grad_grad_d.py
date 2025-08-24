#!/usr/bin/python
# -*- coding: utf-8 -*-
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
max_pool3d_grad_grad_d
"""
# 'pylint: disable=E0401
# 'pylint: disable=unreachable
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from tbe.dsl.compute.pooling3d_max_grad_grad import pooling3d_max_grad_grad
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=too-many-arguments,unused-argument,invalid-name
# 'pylint: disable=too-many-locals,too-many-boolean-expressions
def check_supported(orig_input, orig_output, grad_grad, assist, output,
                    ksize, strides, pads=(0, 0, 0, 0, 0, 0),
                    data_format="NDHWC",
                    kernel_name="max_pool3d_grad_grad_d"):
    """
    Parameters
    ----------
    orig_input : dict, shape and dtype of input_data, format is NDC1HWC0

    orig_output : dict, result of max_pool3d(orig_input, ksize, ...),format is NDC1HWC0

    grad_grad : dict, gradients of gradients, format is NDC1HWC0

    output: dict, shape and dtype of output_data,format is NDC1HWC0

    ksize : list or tuple, the window of max_pool3d_grad_grad_d,
            only support max_pool3d_grad_grad_d in D or H or W

    strides : list or tuple, the stride of max_pool3d window,
              only support max_pool3d_grad_grad_d in D or H or W

    pads : reserved.

    data_format : str, default = "NDHWC"

    kernel_name : cce kernel name, default value is "max_pool3d_grad_grad_d"

    Returns
    -------
    Bool
    """
    ori_format = orig_input.get("ori_format").upper()
    support_list = ["NDHWC"]
    if ori_format not in support_list:
        reason = "MaxPool3dGradGrad op not support ori_format %s \
                  when supported list is %s." % (ori_format, support_list)
        return False, reason
    return True, ""


# 'pylint: disable=too-many-arguments,unused-argument,invalid-name
# 'pylint: disable=too-many-locals,too-many-boolean-expressions
def max_pool3d_grad_grad_d(orig_input, orig_output, grad_grad, assist, output,
                           ksize, strides, pads=(0, 0, 0, 0, 0, 0),
                           data_format="NDHWC",
                           kernel_name="max_pool3d_grad_grad_d"):
    """
    Parameters
    ----------
    orig_input : dict, shape and dtype of input_data, format is NDC1HWC0

    orig_output : dict, result of max_pool3d(orig_input, ksize, ...),format is NDC1HWC0

    grad_grad : dict, gradients of gradients, format is NDC1HWC0

    output: dict, shape and dtype of output_data,format is NDC1HWC0

    ksize : list or tuple, the window of max_pool3d_grad_grad_d,
            only support max_pool3d_grad_grad_d in D or H or W

    strides : list or tuple, the stride of max_pool3d window,
              only support max_pool3d_grad_grad_d in D or H or W

    pads : reserved.

    data_format : str, default = "NDHWC"

    kernel_name : cce kernel name, default value is "max_pool3d_grad_grad_d"

    Returns
    -------
    None
    """
    max_build_round_for_recalc_ub = 8
    if (pads[0] == 0 and pads[1] == 0 and pads[2] == 0 and
            pads[3] == 0 and pads[4] == 0 and pads[5] == 0):
        padding = "VALID"
    else:
        padding = "SAME"

    orig_input_shape = orig_input.get("shape")
    orig_output_shape = orig_output.get("shape")
    grad_grad_shape = grad_grad.get("shape")
    assist_shape = assist.get("shape")

    input_dtype = orig_input.get("dtype")
    output_dtype = orig_output.get("dtype")
    input_dtype = input_dtype.lower()
    output_dtype = output_dtype.lower()

    window_d, window_h, window_w = _get_ksize(ksize, data_format)
    stride_d, stride_h, stride_w = _get_stride(strides, data_format)
    orig_input_tensor = tvm.placeholder(orig_input_shape,
                                        name="orig_data_input",
                                        dtype=input_dtype)
    orig_output_tensor = tvm.placeholder(orig_output_shape,
                                         name="orig_data_output",
                                         dtype=output_dtype)
    grad_grad_tensor = tvm.placeholder(grad_grad_shape,
                                       name="grad_grad",
                                       dtype=input_dtype)
    assist_tensor = tvm.placeholder(assist_shape,
                                    name="assist",
                                    dtype=input_dtype)

    # UB size can not be calculated accurately, so retry 8 times at most
    build_count = 0
    while build_count <= max_build_round_for_recalc_ub:
        res = pooling3d_max_grad_grad(orig_input_tensor,
                                          orig_output_tensor,
                                          grad_grad_tensor,
                                          assist_tensor,
                                          (window_d, window_h, window_w),
                                          (stride_d, stride_h, stride_w),
                                          pads, data_format, padding)
        try:
            with tvm.target.cce():
                # because of attr could be assigned only once, so use attr name in schedule to judge tiling round.
                res.op.attrs["recalc_ub_round_"+str(build_count)] = build_count
                build_count = build_count + 1
                sch = tbe.auto_schedule(res)
                config = {
                    "name": kernel_name,
                    "dummy_placeholder": True,
                    "tensor_list": [orig_input_tensor, orig_output_tensor,
                                    grad_grad_tensor, assist_tensor, res]}
                tbe.cce_build_code(sch, config)
                break
        except tvm.TVMError as e:
            if str(e).find("VMError: Allocation exceed bound of memory tag:local.UB") != -1:
                error_manager_vector.raise_err_specific_reson("VMError: Allocation exceed bound of memory tag:local.UB")
                continue
            raise
            break


def _get_ksize(ksize, data_format):
    if len(ksize) == 1:
        return ksize[0], ksize[0], ksize[0]
    if len(ksize) == 3:
        return ksize[0], ksize[1], ksize[2]
    if data_format == "NDHWC" and len(ksize) == 5:
        return ksize[1], ksize[2], ksize[3]
    if data_format == "NCDHW" and len(ksize) == 5:
        return ksize[2], ksize[3], ksize[4]
    error_manager_vector.raise_err_specific_reson("max_pool3d_grad_grad_d", "Invalid ksize")


def _get_stride(strides, data_format):
    if len(strides) == 1:
        return strides[0], strides[0], strides[0]
    if len(strides) == 3:
        return strides[0], strides[1], strides[2]
    if data_format == "NDHWC" and len(strides) == 5:
        return strides[1], strides[2], strides[3]
    if data_format == "NCDHW" and len(strides) == 5:
        return strides[2], strides[3], strides[4]
    error_manager_vector.raise_err_specific_reson("max_pool3d_grad_grad_d", "Invalid strides")
