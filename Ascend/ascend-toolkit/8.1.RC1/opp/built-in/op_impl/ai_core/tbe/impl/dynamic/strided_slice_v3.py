#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
strided_slice_v3
"""
from __future__ import absolute_import
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tik
from .strided_slice import StridedSlice


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant
    """
    MAX_SIZE = 2 ** 31 - 1


# 'pylint: disable=locally-disabled,too-many-arguments,invalid-name,unused-argument
# 'pylint: disable=unused-argument,too-many-locals,redefined-builtin
@register_operator("StridedSliceV3")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def strided_slice_v3(x, begin, end, axes, strides, y, kernel_name="strided_slice_v3"):
    """
    algorithm: slice
    calculating: this operation extracts a slice of size size
                 from a tensor input
                 starting at the location specified by begin.

    Parameters
    ----------
    x: dict
        contains shape and dtype information of input tensor
    y: dict
        contains shape and dtype information of output tensor
    begin: dict.
        shape and dtype of begin, represents the index of the first value to select.
    end: dict.
        shape and dtype of end, represents the index of the last value to select.
    axes:dict
        axes of begin and end pairs to select
    strides: dict.
        shape and dtype of strides, step length to select.
    kernel_name: str
        cce kernel name, default value is "stride_slice_v3".

    Returns
    -------
    tik instance
    """
    # dynamic strided_slice_v3 does not use axes params.
    input_dtype = x.get("dtype").lower()
    check_list = ("bfloat16", "bool", "float32", "float16", "int8", "int16", "int32",
                  "int64", "uint8", "uint16", "uint32", "uint64")
    para_check.check_dtype(input_dtype, check_list, param_name="x")
    strided_slice_instance = StridedSlice(x, None, 0, 0, 0, 0, 0, kernel_name)
    strided_slice_instance.strided_slice()
    inst = strided_slice_instance.tik_instance
    opt_config = strided_slice_instance.get_opt_config()
    strided_slice_instance.axes_gm = inst.Tensor(strided_slice_instance.dtype, (Constant.MAX_SIZE,),
                                                 name="axes_gm", scope=tik.scope_gm)
    inputs_list = [strided_slice_instance.input_gm, strided_slice_instance.begin_gm, strided_slice_instance.end_gm]
    if strides:
        inputs_list.append(strided_slice_instance.strides_gm)
    if axes:
        inputs_list.append(strided_slice_instance.axes_gm)

    inst.BuildCCE(kernel_name=strided_slice_instance.kernel_name,
                  inputs=inputs_list,
                  outputs=(strided_slice_instance.output_gm,),
                  flowtable=[strided_slice_instance.tiling_param.tiling_gm],
                  config=opt_config,
                  enable_l2=False)

    tbe_context.get_context().add_compile_info("vars", strided_slice_instance.get_vars_info())
    return inst
