# /usr/bin/env python
# -*- coding:utf-8 -*-
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
concat_d
"""
from impl.util.platform_adapter import para_check
from impl.concat_v2_d import concat_v2_d
from impl.concat_v2_d import op_select_format as concat_v2_op_select_format
from impl.concat_v2_d import get_op_support_info as concat_v2_get_op_support_info
from impl.concat_v2_d import check_supported as concat_v2_d_check_supported


# 'pylint: disable = unused-argument
# 'pylint: disable=consider-using-in
def get_op_support_info(input_values, output_data, concat_dim, kernel_name="concat"):
    """
    get_op_support_info
    """
    return concat_v2_get_op_support_info(input_values, output_data, concat_dim, kernel_name)


# 'pylint: disable=locally-disabled,unused-argument,too-many-branches
# 'pylint: disable=too-many-locals,too-many-statements,unused-variable
def op_select_format(input_values, output_data, concat_dim,
                     kernel_name="concat"):
    """
    1. When input ori_format is in ["NDCHW", "HWCN", "NCHW"], and
       ori_format indexed by concat_dim is not C or N. When all
       of input's shape is same, and C axis is in [2, 4, 8]. Or
       all of input's shape is not same, C axis of output is
       greater then or equal to 16. The Op ConcatD can support
       NC1HWC0 and NDC1HWC0.
    > for example:
    > x : Tensor of (shape=(16, 16, 16, 16, 16), "NC1HWC0")

    2. When input ori_format is in ["NDCHW", "HWCN", "NCHW"], and
       ori_format indexed by concat_dim is not C. The Op
       ConcatD can support HWCN, NCHW and NDCHW.
    > for example:
    > x : Tensor of (shape=(16, 16, 16, 16), "NCHW")

    3. When length of input is greater then or equal to 2,
    concat_dim is the last dimension or second-to-last index.
    The Op ConcatD can support ND.
    > for example:
    > x : Tensor of (shape=(16, 16, 16, 16), "ND")
    """
    return concat_v2_op_select_format(input_values, output_data, concat_dim, kernel_name)


def check_supported(input_values, output_data, concat_dim, kernel_name="concat"):
    """
    check_supported invoked by framework
    """
    return concat_v2_d_check_supported(input_values, output_data, concat_dim, kernel_name)


@para_check.check_op_params(para_check.DYNAMIC_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_INT, para_check.KERNEL_NAME)
def concat_d(input_values, output_data, concat_dim, kernel_name="concat"):
    """
    algorithm: concat
    Concatenates tensors along one dimension.
    Parameters
    ----------
    input_values : A list of `dict`.dict include keys shape and dtype
    output_data: dict of output_data, dict include keys shape and dtype
    concat_dim : scalar, in the range [-rank(values), rank(values))]
    kernel_name : cce kernel name, default value is "concat"
    Returns
    -------
    None
    """
    # concat_d is the same as concat_v2_d
    # use concat_v2_d to replace
    concat_v2_d(input_values, output_data, concat_dim, kernel_name)
