"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

topk_v2
"""
from impl.util.platform_adapter import para_check
from impl.dynamic.top_k_d import top_k_d
from impl.util.platform_adapter import register_operator


# 'pylint: disable=redefined-builtin,too-many-arguments
@register_operator("TopKV2D")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def top_k_v2_d(input_tensor,
               k_tensor,
               indices_tensor,
               out_tensor,
               out_indices_tensor,
               sorted=True,
               dim=-1,
               largest=True,
               kernel_name='top_k'):

    """
    top_k_v2 interface

    Parameters
    ----------
    input_tensor: dict. input params shape, dtype and range
    k_tensor: dict. k params shape, dtype
    indices_tensor: dict. input indices shape, dtype and range
    out_tensor: dict. output shape, dtype and range
    out_indices_tensor: dict. output index shape, dtype and range
    sorted : bool. if is sorted
    largest : bool. if is sorted by largest
    kernel_name: kernel name of top_k op
    """
    k_dtype = k_tensor.get("dtype")
    para_check.check_dtype(k_dtype, ["int32"], "k_tensor")
    top_k_d(input_tensor, indices_tensor, out_tensor, out_indices_tensor, -1, sorted, dim, largest, kernel_name)
