#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2019 Huawei Technologies Co., Ltd
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
trans_data_rnn
"""
from impl import trans_data_common_func as tdc
from impl import trans_data_negative_target_ntc
from impl import trans_data_positive_source_ntc
from impl import trans_data_rnn_negative_target
from impl import trans_data_rnn_positive_source
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check


# 'pylint: disable=locally-disabled,too-many-branches,too-many-arguments,too-many-statements
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_STR,
                            para_check.REQUIRED_ATTR_STR, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def trans_data_rnn(src, dst, src_format, dst_format, input_size, hidden_size, kernel_name='trans_data_rnn'):
    """
    algorithm: format_transfer
    doing format_transfer for various data format
    only support ND to FRACTAL_ZN_RNN/ND_RNN_BIAS and FRACTAL_ZN_RNN/ND_RNN_BIAS to ND

    Parameters
    ----------
    src : dict
        shape and dtype of input
    dst: dict
        shape and dtype of output, should be same shape and type as input
    src_format: str
        source data format, can be NHWC, NCHW, FRACTAL_Zn etc.
    dst_format: str
        target data format, can be NC1HWC0, NCHW, FRACTAL_Zn etc.
    input_size: int
    hidden_size: int
    kernel_name: str
        kernel name, default value is "format_transfer"

    Returns
    -------
    None
    """
    src_format = src_format.upper()
    dst_format = dst_format.upper()
    in_shape = list(src.get("shape"))
    out_shape = list(dst.get("shape"))
    if src_format == "ND" and out_shape[-1] % tdc.C0_16 != 0:
        error_detail = "the shape of out_shape is not satisfied." + str(out_shape[-1])
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "out_shape", error_detail)
    if dst_format == "ND" and in_shape[-1] % tdc.C0_16 != 0:
        error_detail = "the shape of in_shape is not satisfied." + str(in_shape[-1])
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "in_shape", error_detail)

    if src_format == "ND" and dst_format == "ND_RNN_BIAS":
        if len(in_shape) != 1:
            error_detail = "the length of in_shape is not 1."
            error_manager_vector.raise_err_input_shape_invalid(kernel_name, "in_shape", error_detail)
        if in_shape[-1] % hidden_size != 0:
            error_detail = "the in_shape[-1] can't dived by hidden_size."
            error_manager_vector.raise_err_input_shape_invalid(kernel_name, "in_shape", error_detail)
        hidden_cnt = in_shape[-1] // hidden_size
        input_shape = [hidden_cnt, hidden_size, 1, 1]
        output_shape = [hidden_cnt, tdc.ceil_div(hidden_size, tdc.C0_16), 1, 1, tdc.C0_16]
        src["shape"] = input_shape
        dst["shape"] = output_shape
        trans_data_positive_source_ntc.trans_data_positive_source_ntc(src, dst, "NCHW", "NC1HWC0",
                                                                      kernel_name=kernel_name)
    elif src_format == "ND" and dst_format == "FRACTAL_ZN_RNN":
        if len(in_shape) != 2 or len(out_shape) != 4:
            error_detail = "the length of in_shape is not 2 or the length of out_shape is not 4."
            error_manager_vector.raise_err_input_shape_invalid(kernel_name, "in_shape and out_shape", error_detail)
        if in_shape[0] < min(input_size, hidden_size):
            error_detail = "in_shape[0] is less than input_size and hidden_size."
            error_manager_vector.raise_err_input_shape_invalid(kernel_name, "in_shape", error_detail)
        trans_data_rnn_positive_source.trans_data_rnn_positive_source(src, dst, input_size, hidden_size, kernel_name)
    elif src_format == "ND_RNN_BIAS" and dst_format == "ND":
        if len(out_shape) != 1:
            error_detail = "the length of out_shape is not 1."
            error_manager_vector.raise_err_input_shape_invalid(kernel_name, "out_shape", error_detail)
        if out_shape[-1] % hidden_size != 0:
            error_detail = "the out_shape[-1] can't dived by hidden_size."
            error_manager_vector.raise_err_input_shape_invalid(kernel_name, "out_shape", error_detail)
        hidden_cnt = out_shape[-1] // hidden_size
        input_shape = [hidden_cnt, tdc.ceil_div(hidden_size, tdc.C0_16), 1, 1, tdc.C0_16]
        output_shape = [hidden_cnt, hidden_size, 1, 1]
        src["shape"] = input_shape
        dst["shape"] = output_shape
        trans_data_negative_target_ntc.trans_data_negative_target_ntc(src, dst, "NC1HWC0", "NCHW",
                                                                      kernel_name=kernel_name)
    elif src_format == "FRACTAL_ZN_RNN" and dst_format in ("ND", "NHWC"):
        if len(in_shape) != 4 or len(out_shape) != 2:
            error_detail = "the length of in_shape is not 4 or the length of out_shape is not 2."
            error_manager_vector.raise_err_input_shape_invalid(kernel_name, "in_shape and out_shape", error_detail)
        if out_shape[0] < min(input_size, hidden_size):
            error_detail = "out_shape[0] is less than input_size and hidden_size."
            error_manager_vector.raise_err_input_shape_invalid(kernel_name, "out_shape", error_detail)
        trans_data_rnn_negative_target.trans_data_rnn_negative_target(src, dst, input_size, hidden_size, kernel_name)
    else:
        raise RuntimeError("not support this kind of format transfer !")
