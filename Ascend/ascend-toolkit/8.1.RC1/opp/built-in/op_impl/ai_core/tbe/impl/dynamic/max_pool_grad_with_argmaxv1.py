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
max_pool_grad_with_argmax_v1
"""
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.dynamic.max_pool_grad_with_argmaxv2 import MaxpoolGrad
from impl.dynamic.max_pool_grad_with_argmaxv2 import check_param
from impl.util.util_common import ceil, is_unknown_rank_input
from impl.dynamic.max_pool_grad_with_argmax_v1_dsl import max_pool_grad_with_argmax_v1_dsl


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """

    DT_INT32 = 3

    def __init__(self):
        pass


def is_unknown_attr(attr_list):
    if None in attr_list:
        return True
    return False


# 'pylint: disable=unused-argument
# 'pylint: disable=invalid-name,too-many-arguments,useless-super-delegation,super-with-arguments
# 'pylint: disable=too-many-locals,too-many-branches,too-many-statements,too-many-return-statements,consider-using-in
@register_operator("MaxPoolGradWithArgmaxV1")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def max_pool_grad_with_argmax_v1(x, grad, argmax, y, ksize, strides, pads,
                                 dtype=Constant.DT_INT32,
                                 dilation=(1, 1, 1, 1), ceil_mode=False,
                                 kernel_name="max_pool_grad_with_argmax_v1"):
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
    strides: stride , minimum length is 4, just like
    [1, poolingStrideH, poolingStrideW, 1]
    pads: pad list_int
    kernel_name: kernel_name
    Returns
    -------
    tik_instance: tik_instance
    :param ceil_mode:
    """
    if is_unknown_rank_input((x, grad, argmax)) or is_unknown_attr([ksize, strides, pads]) or \
            grad.get("dtype").lower() == "float32":
        max_pool_grad_with_argmax_v1_dsl(
            x, grad, argmax, y, ksize, strides, pads, dilation, ceil_mode, kernel_name)
    else:
        ori_format = x.get("ori_format")
        check_param(x, argmax, grad, ksize, strides,
                    ori_format, dilation, pads, kernel_name)

        dtype = x.get("dtype").lower()
        maxpoolgrad = MaxpoolGrad(
            dtype, ksize, strides, pads, dilation, ceil_mode, kernel_name)
        return maxpoolgrad.tik_instance_function()
