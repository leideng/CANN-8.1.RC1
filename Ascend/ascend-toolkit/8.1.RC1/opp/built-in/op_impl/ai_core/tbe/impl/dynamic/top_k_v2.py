#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2022 Huawei Technologies Co., Ltd
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
top_k_v2
"""
# 'pylint: disable=too-many-lines
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import OpPatternMode


# 'pylint: disable=redefined-builtin,too-many-arguments
@register_operator("TopKV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def top_k_v2(input_tensor,
             k_tensor,
             out_tensor,
             out_indices_tensor,
             sorted=True,
             dim=-1,
             largest=True,
             kernel_name='top_k_v2'):
    """
    top_k_v2 interface

    Parameters
    ----------
    input_tensor: dict. input params shape, dtype and range
    k_tensor: dict. k params shape, dtype
    out_tensor: dict. output shape, dtype and range
    out_indices_tensor: dict. output index shape, dtype and range
    sorted : bool. if is sorted
    largest : bool. if is sorted by largest
    kernel_name: kernel name of top_k op
    """
    from impl.util.platform_adapter import tbe_context
    dtype = input_tensor.get("dtype")
    k_dtype = k_tensor.get("dtype")
    out_dtype = out_tensor.get("dtype")
    out_indices_dtype = out_indices_tensor.get("dtype")
    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(dtype, check_list, param_name="input_tensor")
    para_check.check_dtype(k_dtype, ("int32",), param_name="k_tensor")
    para_check.check_dtype(out_dtype, check_list, param_name="out_tensor")
    para_check.check_dtype(out_indices_dtype, ("int32",), param_name="out_indices_tensor")

    if dim is None:
        dim = -1

    ins = classify([input_tensor, dim, k_tensor], OpPatternMode.SORT, {"op_mode": "topkv2"})
    schedules, tensors = [], []
    for (_x, _k) in ins:
        with tbe.compute():
            x_shape, k_var = shape_util.variable_shape([_x, _k], "sort")
            x_input = tvm.placeholder(x_shape, name="data_input", dtype=input_tensor["dtype"])
            k_input = tvm.placeholder([len(k_var)], name="k_input", dtype=k_tensor["dtype"])
            direction = "descend" if largest else "ascend"
            if dtype == "bfloat16":
                x_input_fp32 = tbe.cast_to(x_input, "float32")
                value, indices = tbe.topk(x_input_fp32, k_var[0], sort_axis=-1, direction=direction, return_type="both",
                                      indices_dtype=out_indices_tensor["dtype"], need_cast=True)
            else:
                value, indices = tbe.topk(x_input, k_var[0], sort_axis=-1, direction=direction, return_type="both",
                                      indices_dtype=out_indices_tensor["dtype"])
            tensors.append([x_input, k_input, value, indices])
        with tvm.target.cce():
            tbe_context.get_context().add_compile_info("isTopkV2", True)
            sch = tbe.auto_schedule([value, indices])
        schedules.append(sch)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
