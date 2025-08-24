"""
Copyright (C) 2020-2022. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

concat_d
"""
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.dynamic.concat_v2_d import concat_v2_d
from impl.concat_v2_d import get_op_support_info as concat_v2_get_op_support_info
from impl.concat_v2_d import op_select_format as concat_v2_op_select_format
from impl.dynamic.concat_v2_d import check_supported as concat_v2_check_supported


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
    return concat_v2_check_supported(input_values, output_data, concat_dim, kernel_name)


# 'pylint: disable=unused-argument
@register_operator_compute("ConcatD", op_mode="dynamic", support_fusion=False)
def concat_d_compute(input_values, output_data, concat_dim, kernel_name="concat"):
    """
    algorithm: concat
    Concatenates tensors along one dimension.

    Parameters
    ----------
    input_values : list of placeholders, all input data
    output_data : dict, dict of output
    concat_dim : scalar, in the range [-rank(values), rank(values))]
    kernel_name : string
        cce kernel name, default value is concat

    Returns
    -------
    res : placeholder and res
    """
    res = tbe.concat(input_values, concat_dim)

    return res


@register_operator("ConcatD")
@para_check.check_op_params(para_check.DYNAMIC_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
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
    return concat_v2_d(input_values, output_data, concat_dim, kernel_name)
