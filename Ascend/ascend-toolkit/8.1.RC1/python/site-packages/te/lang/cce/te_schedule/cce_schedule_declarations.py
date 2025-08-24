#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
auto_schedule distribution rules

Add your rules to pattern mapping here so cce auto_schedule can distribute your operators
to corresponding schedule correctly
"""
from enum import Enum


class OpFlags(Enum):
    """Define all tag flags here for compute type data collection"""
    elewise_single_flag = "elewise_single"
    elewise_binary_flag = "elewise_binary"
    elewise_flag = "elewise"
    broadcast_flag = "broadcast"
    reduce_flag = "reduce"
    segment_flag = "segment"
    segment_tensor_flag = "segmentensor"
    inplace_flag = "inplace"
    matmul_flag = "matmul"
    matmul_v2_flag = "matmul_v2"
    convolution_flag = "convolution"
    conv2d_backprop_input_flag = "conv2d_backprop_input"
    conv2d_backprop_filter_flag = "conv2d_backprop_filter"
    conv3d_backprop_input_flag = "conv3d_backprop_input"
    conv3d_backprop_filter_flag = "conv3d_backprop_filter"
    concat_flag = "concat"
    ascend_quant_flag = "quant"
    ascend_dequant_flag = "dequant"
    ascend_anti_quant_flag = "anti_quant"
    strided_read_flag = "strided_read"
    strided_write_flag = "strided_write"
    ascend_dequant_s16_flag = "dequant_s16"
    ascend_requant_flag = "requant"
    ascend_requant_s16_flag = "requant_s16"
    # for max_pool_v200 in max_pool_v200_compute, conv + max_pool fusion
    max_pool_flag = "pooling2d_max_"
    depthwise_conv_flag = "depthwise_conv2d"
    conv3d_flag = "conv3d_"
    pool2d_avg_flag = "pooling2d_avg"
    pool2d_max_flag = "pooling2d_max"
    pool2d_gap_flag = "pooling2d_gap"
    pool2d_gmp_flag = "pooling2d_gmp"
    pool3d_avg_flag = "pooling3d_avg"
    pool3d_max_flag = "pooling3d_max"
    pool3d_gap_flag = "pooling3d_gap"
    pool3d_gmp_flag = "pooling3d_gmp"
    read_select_flag = "read_select"
    write_select_flag = "write_select"
    cast_flag = "elewise_single_cast"
    cmpsel_flag = "elewise_binary_cmpsel"
    cmp_flag = "elewise_binary_vcmpv_eq"
    sel_flag = "elewise_multiple_sel"
    dilation_flag = "dilation"


class OpPatterns(Enum):
    """Define all patterns here for different schedule"""
    BN_UPDATE_GRAD_PATTERN = "bn_update_grad"
    BN_GRAD_REDUCE_PATTERN = "bn_grad_reduce"
    LAYER_NORM_GRAD_PATTERN = "layer_norm_grad"
    L2LOSS_MUL_ADDN_PATTERN = "l2loss_mul_addn"
    ELEMWISE_PATTERN = "ElemWise"
    PURE_BROADCAST_PATTERN = "PureBroadcast"
    REDUCE_PATTERN = "CommReduce"
    SEGMENT_PATTERN = "Segment"
    INPLACE_PATTERN = "Inplace"
    MATMUL_PATTERN = "Matmul"
    MATMUL_V2_PATTERN = "Matmul_v2"
    CONV_PATTERN = "Convolution"
    CONV2D_BACKPROP_INPUT_PATTERN = "Conv2d_backprop_input"
    CONV2D_BACKPROP_FILTER_PATTERN = "Conv2d_backprop_filter"
    CONV3D_BACKPROP_INPUT_PATTERN = "Conv3d_backprop_input"
    CONV3D_BACKPROP_FILTER_PATTERN = "Conv3d_backprop_filter"
    OPAQUE_PATTERN = "Opaque"
    BN_REDUCE_PATTERN = "bn_reduce"
    BN_UPDATE_PATTERN = "bn_update"
    SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_PATTERN = "softmax_cross_entropy_with_logits"
    L2_NORMALIZE_PATTERN = "l2_normalize"
    SOFTMAX_PATTERN = "softmax_pattern"
    L2_LOSS_PATTERN = "l2_loss"
    ASCEND_QUANT_PATTERN = "quant"
    ASCEND_DEQUANT_PATTERN = "dequant"
    ASCEND_ANTI_QUANT_PATTERN = "anti_quant"
    STRIDED_READ_PATTERN = "strided_read"
    STRIDED_WRITE_PATTERN = "strided_write"
    ASCEND_DEQUANT_S16_PATTERN = "dequant_s16"
    ASCEND_REQUANT_PATTERN = "requant"
    ASCEND_REQUANT_S16_PATTERN = "requant_s16"
    MAX_POOL_PATTERN = "MaxPool"
    DEPTHWISECONV_PATTERN = "DepthwiseConvolution"
    CONV3D_PATTERN = "Conv3d"
    POOL2D_PATTERN = "Pool2d"
    POOL3D_PATTERN = "Pool3d"
    READ_SELECT_PATTERN = "read_select"
    WRITE_SELECT_PATTERN = "write_select"
    COSINE_EMBEDDING_LOSS_PATTERN = "cosine_embedding_loss"
    DILATION_PATTERN = "dilation"


class OpSubPatterns(Enum):
    """Define all sub_patterns here for different sub-schedule"""
    REDUCE_ATOMIC_PATTERN = "reduce_atomic"
    REDUCE_5HDC_PATTERN = "reduce_5hdc"
    CMPSEL_PATTERN = "cmpsel"


class OpSpecTypes(Enum):
    """Define all special operators here for different special schedule"""
    RELU_GRAD_V2 = "relu_grad_v2"
    FLOOR_MOD = "floor_mod"
    TRUNCATE_DIV = "truncate_div"
    THRESHOLD_GRAD_V2_D = "threshold_grad_v2_d"
    RELU_V2 = "relu_v2"
    NORMALIZE_SCALE = "normalize_scale"
    ONLY_BROADCAST_TYPE = "only_broadcast_type"
    MVN = "mvn"
    BN_EXT2 = "bn_ext2"
    REDUCE_MEAN_2D_ALIGNED_MID_REDUCE_NO_CAST = "reduce_mean_2d_aligned_mid_reduce_no_cast"
