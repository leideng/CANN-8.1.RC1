#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
public function for avgpool update dynamic
"""
from __future__ import absolute_import
import math
from tbe import tvm
from tbe.dsl.base.operation import get_te_var
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import error_manager_vector
from tbe.common.utils import log
from tbe.common.utils.op_util import op_util_conv2d
from tbe.common.utils.op_util import op_util_avg_pool_update
from tbe.common.utils.op_util.op_util_conv2d import ceil_div

DYNAMIC_FLAG = -1


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    N_DIM = 0
    C_DIM = 1
    H_DIM = 2
    W_DIM = 3


# 'pylint: disable=unused-variable,too-many-arguments,too-many-locals,too-many-arguments,invalid-name
def get_attr_nchw_format(input_shape, ksize, strides, data_format):
    """
    get attr nchw format
    """
    if data_format not in ("NCHW", "NHWC"):
        error_manager_vector.raise_err_input_format_invalid("AvgPoolUpdate", "input", ["NCHW", "NHWC"], data_format)

    if data_format == "NHWC":
        batch, hi, wi, channel = input_shape
        kn, kh, kw, kc = ksize
        stride_n, stride_h, stride_w, stride_c = strides

        input_shape = [batch, channel, hi, wi]
        ksize = [kn, kc, kh, kw]
        strides = [stride_n, stride_c, stride_h, stride_w]

    return input_shape, ksize, strides


def get_correct_pad(in_pad):
    """
    correct pads when less than zero
    """
    return tvm.max(in_pad, 0)


# 'pylint: disable=unused-variable,too-many-arguments,too-many-locals,too-many-arguments,invalid-name
def calculate_pads(input_shape, output_shape, ksize, strides, padding, pads, ceil_mode):
    """
    calculate pads
    """
    input_h, input_w = input_shape[Constant.H_DIM], input_shape[Constant.W_DIM]
    output_h, output_w = output_shape[Constant.H_DIM], output_shape[Constant.W_DIM]
    k_h, k_w = ksize[Constant.H_DIM], ksize[Constant.W_DIM]
    stride_h, stride_w = strides[Constant.H_DIM], strides[Constant.W_DIM]

    if padding == "SAME":
        pad_row = (output_h - 1) * stride_h + k_h - input_h
        pad_col = (output_w - 1) * stride_w + k_w - input_w
        pad_top = get_correct_pad(pad_row // 2)
        pad_bottom = get_correct_pad(pad_row - pad_top)
        pad_left = get_correct_pad(pad_col // 2)
        pad_right = get_correct_pad(pad_col - pad_left)

        correct_pads = [pad_top, pad_bottom, pad_left, pad_right]
    elif padding == "CALCULATED":
        pad_top, pad_bottom, pad_left, pad_right = pads
        pad_bottom = get_correct_pad((output_h - 1)*stride_h + k_h - input_h - pad_top)
        pad_right = get_correct_pad((output_w - 1)*stride_w + k_w - input_w - pad_left)

        correct_pads = [pad_top, pad_bottom, pad_left, pad_right]
    else:
        if ceil_mode:
            pad_bottom = get_correct_pad((output_h - 1)*stride_h + k_h - input_h)
            pad_right = get_correct_pad((output_w - 1)*stride_w + k_w - input_w)

            correct_pads = [0, pad_bottom, 0, pad_right]
        else:
            correct_pads = [0, 0, 0, 0]

    return correct_pads


def cache_tiling_get_var(para_dict):
    dilation_h_var = get_te_var(op_util_conv2d.TilingDataKey.DILATION_H).get_tvm_var()
    dilation_w_var = get_te_var(op_util_conv2d.TilingDataKey.DILATION_W).get_tvm_var()
    stride_h_var = get_te_var(op_util_conv2d.TilingDataKey.STRIDE_H).get_tvm_var()
    stride_w_var = get_te_var(op_util_conv2d.TilingDataKey.STRIDE_W).get_tvm_var()
    dilations_var = [1, 1, dilation_h_var, dilation_w_var]
    strides_var = [1, 1, stride_h_var, stride_w_var]

    batch_n = get_te_var(op_util_conv2d.TilingDataKey.BATCH_N).get_tvm_var()
    fmap_h = get_te_var(op_util_conv2d.TilingDataKey.FMAP_H).get_tvm_var()
    fmap_w = get_te_var(op_util_conv2d.TilingDataKey.FMAP_W).get_tvm_var()
    h_out = get_te_var(op_util_conv2d.TilingDataKey.HO).get_tvm_var()
    w_out = get_te_var(op_util_conv2d.TilingDataKey.WO).get_tvm_var()

    c_in = get_te_var(op_util_conv2d.TilingDataKey.C_IN).get_tvm_var()
    c_out = get_te_var(op_util_conv2d.TilingDataKey.C_OUT).get_tvm_var()
    k_h = get_te_var(op_util_conv2d.TilingDataKey.K_H).get_tvm_var()
    k_w = get_te_var(op_util_conv2d.TilingDataKey.K_W).get_tvm_var()
    pad_top = get_te_var(op_util_conv2d.TilingDataKey.PAD_TOP).get_tvm_var()
    pad_bottom = get_te_var(op_util_conv2d.TilingDataKey.PAD_BOTTOM).get_tvm_var()
    pad_left = get_te_var(op_util_conv2d.TilingDataKey.PAD_LEFT).get_tvm_var()
    pad_right = get_te_var(op_util_conv2d.TilingDataKey.PAD_RIGHT).get_tvm_var()
    pads = [pad_top, pad_bottom, pad_left, pad_right]

    block_size_k, block_size_n = tbe_platform.CUBE_MKN[para_dict["dtype"]]["mac"][1:3]
    ci1 = ceil_div(c_in, block_size_k)
    co1 = ceil_div(c_out, block_size_k)  # Here we calculate co1 on UB.

    # The output of conv2d on L0C is (batch_n, ceildiv(cout, blk_n), h_out, w_out, blk_n).
    # The output of connv2d on UB/OUT is (batch_n, ceildiv(cout, blk_k), h_out, w_out, blk_k).
    # The input1 of avg_pool_update is the output of conv2d on UB.
    input1_shape_5hd = (batch_n, co1, h_out, w_out, block_size_k)
    input2_shape_5hd = (batch_n, ci1, fmap_h, fmap_w, block_size_k)
    input1_shape = (batch_n, c_out, h_out, w_out)
    input2_shape = (batch_n, c_in, fmap_h, fmap_w)
    ksize = [1, 1, k_h, k_w]

    para_dict.update({"input1_realshape": input1_shape_5hd, "input2_realshape": input2_shape_5hd,
            "input1_shape": input1_shape, "input2_shape": input2_shape, "ksize": ksize,
            "strides": strides_var, "pads": pads, "dynamic_flag": True,
    })
    return para_dict


def cache_tiling_paras_process_avgupdate(para_dict):
    """
    config paras for cachetiling
    """
    def generate_op_var():
        for i in range(op_util_avg_pool_update.TilingDataIdx.TILINGDATA_IDX_END):
            if i in op_util_avg_pool_update.TILINGDATA_KEY_MAP.keys():
                if not operation.get_te_var(op_util_avg_pool_update.TILINGDATA_KEY_MAP.get(i)):
                    operation.var(op_util_avg_pool_update.TILINGDATA_KEY_MAP.get(i))
            else:
                log.error("Tiling key not in Tilingkey Map")

    generate_op_var()
    para_dict = cache_tiling_get_var(para_dict)
    return para_dict


def cache_tiling_static(para_dict):
    x1 = para_dict["input_1"]
    x2 = para_dict["input_2"]
    ksize = para_dict["ksize"]
    strides = para_dict["strides"]
    data_format = para_dict["data_format"]
    padding = para_dict["padding"]
    pads = para_dict["pads"]
    ceil_mode = para_dict["ceil_mode"]

    input1_shape_5hd = x1.get('shape')
    input2_shape_5hd = x2.get('shape')
    x1_shape_nchw, _, _ = get_attr_nchw_format(x1.get('ori_shape'), ksize, strides, data_format)
    x2_shape_nchw, ksize, strides = get_attr_nchw_format(x2.get('ori_shape'), ksize, strides, data_format)
    pads = calculate_pads(x2_shape_nchw, x1_shape_nchw, ksize, strides, padding, pads, ceil_mode)

    para_dict.update({"input1_realshape": input1_shape_5hd, "input2_realshape": input2_shape_5hd,
            "input1_shape": x1_shape_nchw, "input2_shape": x2_shape_nchw, "ksize": ksize,
            "pads": pads, "dynamic_flag": False,
    })
    return para_dict
