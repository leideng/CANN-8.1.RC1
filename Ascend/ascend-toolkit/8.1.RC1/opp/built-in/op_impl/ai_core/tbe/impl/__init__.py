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
cce specific declaration and schedules.
"""
from __future__ import absolute_import as _abs

import impl.ignore_deprecation_warning

# tbe base op
from .abs import abs
from .l2_loss import l2_loss
from .reduce_mean_d import reduce_mean_d
from .reduce_std_with_mean import reduce_std_with_mean
from .reduce_prod_d import reduce_prod_d
from .segment_max_d import segment_max_d
from .sign import sign
from .squared_difference import squared_difference
from .ldpc import ldpc
from .iou_3d import iou_3d

# tbe it1
from .selu import selu
from .ones_like import ones_like
from .softmax_grad import softmax_grad
from .relu_grad import relu_grad
from .relu_grad_v2 import relu_grad_v2
from .softplus import softplus
from .softplus_v2 import softplus_v2
from .softsign import softsign
from .inv import inv
from .inv_grad import inv_grad
from .matrix_set_diag_d import matrix_set_diag_d
from .softplus_grad import softplus_grad
from .softplus_v2_grad import softplus_v2_grad
from .reduce_min_d import reduce_min_d
from .reduce_min import reduce_min
from .lin_space_d import lin_space_d
from .matrix_diag_d import matrix_diag_d
from .matrix_diag_part_d import matrix_diag_part_d
from .sinh import sinh
from .truncate_div import truncate_div
from .div import div
from .cosh import cosh
from .clip_by_value import clip_by_value
from .div_no_nan import div_no_nan
from .swish import swish
from .hard_swish import hard_swish
from .swish_grad import swish_grad
from .hard_swish_grad import hard_swish_grad
# tbe it2
from .asinh_grad import asinh_grad
from .acosh_grad import acosh_grad
from .elu import elu
from .elu_grad import elu_grad
from .broadcast_to_d import broadcast_to_d
from .atan_grad import atan_grad
from .asinh import asinh
from .acosh import acosh
from .atan import atan
from .atan2 import atan2

# tbe it3
from .concat_d import concat_d
from .histogram_fixed_width_d import histogram_fixed_width_d
from .invert import invert
from .split_v_d import split_v_d
from .mod import mod
from .softmax_cross_entropy_loss import softmax_cross_entropy_loss
from .softmax_cross_entropy_with_logits import softmax_cross_entropy_with_logits
from .log1p import log1p
from .xlogy import xlogy
from .cos import cos
from .xdivy import xdivy
from .exp import exp
from .expm1 import expm1
from .erf import erf
from .erfc import erfc
from .concat_v2_d import concat_v2_d
from .max_pool import max_pool
from .max_pool3d import max_pool3d
from .max_pool3d_grad_grad_d import max_pool3d_grad_grad_d
from .max_pool_ext2 import max_pool_ext2
from .adaptive_max_pool2d import adaptive_max_pool2d
from .neg import neg
from .batch_norm_ext2 import batch_norm_ext2
from .reciprocal_grad import reciprocal_grad
from .not_equal import not_equal
from .rsqrt_grad import rsqrt_grad
from .batch_norm_grad_ext2 import batch_norm_grad_ext2

# tbe it4
from .round import round
from .bitwise_and import bitwise_and
from .rint import rint
from .floor import floor
from .floor_mod import floor_mod
from .slice_d import slice_d
from .fake_quant_with_min_max_args import fake_quant_with_min_max_args
from .fake_quant_with_min_max_args_gradient import fake_quant_with_min_max_args_gradient
from .pow import pow
from .floor_div import floor_div
from .truncate_mod import truncate_mod
from .sin import sin
from .bitwise_or import bitwise_or
from .tan import tan
from .ceil import ceil
from .range_d import range_d
from .bias_add import bias_add
from .bitwise_xor import bitwise_xor
from .fake_quant_with_min_max_vars_gradient import fake_quant_with_min_max_vars_gradient
from .fake_quant_with_min_max_vars_per_channel_gradient import fake_quant_with_min_max_vars_per_channel_gradient
from .fake_quant_with_min_max_vars_per_channel import fake_quant_with_min_max_vars_per_channel
from .fake_quant_with_min_max_vars import fake_quant_with_min_max_vars

# tbe it5
from .rsqrt import rsqrt

# tbe it6
from .greater import greater
from .batch_to_space_nd_d import batch_to_space_nd_d
from .space_to_batch_nd_d import space_to_batch_nd_d
from .batch_norm import batch_norm
from .batch_norm_grad import batch_norm_grad
from .bn_training_reduce import bn_training_reduce
from .bn_training_update import bn_training_update
from .bn_training_reduce_grad import bn_training_reduce_grad
from .bn_training_update_grad import bn_training_update_grad
from .resize_nearest_neighbor_v2_d import resize_nearest_neighbor_v2_d
from .confusion_matrix import confusion_matrix
from .resize_bilinear_v2_d import resize_bilinear_v2_d
from .zeros_like import zeros_like
from .logical_not import logical_not
from .select import select
from .diag_part_d import diag_part_d
from .reduce_sum_d import reduce_sum_d
from .trans_data_2d import trans_data_2d
from .trans_data_rnn import trans_data_rnn
from .bn_infer_grad import bn_infer_grad
from .bn_training_update_v3 import bn_training_update_v3
from .non_zero_with_value import non_zero_with_value

# tbe it9
from .scatter_max import scatter_max
from .scatter_min import scatter_min
from .scatter_mul import scatter_mul
from .scatter_update import scatter_update
from .scatter_nd_update import scatter_nd_update
from .parallel_concat import parallel_concat
from .sparse_apply_proximal_adagrad_d import sparse_apply_proximal_adagrad_d

# tbe it11
from .sparse_apply_ftrl_d import sparse_apply_ftrl_d
from .sparse_apply_adagrad_d import sparse_apply_adagrad_d
from .sparse_apply_adagrad import sparse_apply_adagrad
from .sparse_apply_ftrl_v2_d import sparse_apply_ftrl_v2_d

# tbe it12
from .apply_adagradv2_d import apply_adagradv2_d
from .apply_keras_momentum_d import apply_keras_momentum_d
from .sparse_apply_adagrad_v2_d import sparse_apply_adagrad_v2_d
from .sparse_apply_rms_prop_d import sparse_apply_rms_prop_d
from .sparse_apply_adadelta_d import sparse_apply_adadelta_d
from .apply_adam_with_amsgrad_d import apply_adam_with_amsgrad_d

# tbe it13
from .add_mat_mat_elements import add_mat_mat_elements
from .select_v2 import select_v2
from .leaky_relu import leaky_relu
from .leaky_relu_grad import leaky_relu_grad
from .max_pool3d_with_argmax import max_pool3d_with_argmax
from .max_pool_with_argmax import max_pool_with_argmax
from .max_pool_grad_with_argmax import max_pool_grad_with_argmax

# resnet50
from .pooling import pooling
from .avg_pool import avg_pool
from .aipp import aipp
from .eltwise import eltwise

# SinglePath
from .log import log
from .sqrt_grad import sqrt_grad
from .sigmoid import sigmoid
from .sigmoid_grad import sigmoid_grad
from .depthwise_weight_4d_2_6d import depthwise_weight_4d_2_6d
from .depthwise_weight_6d_2_4d import depthwise_weight_6d_2_4d
from .apply_rms_prop_d import apply_rms_prop_d

from .conv2d import conv2d
from .conv2d import conv2d_compute

from .conv2d_compress import conv2dcompress

# Mobilenetv2
from .avg_pool_grad_d import avg_pool_grad_d
from .depthwise_conv2d import depthwise_conv2d
from .depthwise_conv2d_backprop_filter_d import depthwise_conv2d_backprop_filter_d
from .depthwise_conv2d_backprop_input_d import depthwise_conv2d_backprop_input_d
from .mul import mul
from .relu6_grad import relu6_grad
from .relu6 import relu6
from .relu6_d import relu6_d

# BERT
from .add import add
from .reduce_all_d import reduce_all_d
from .assign_add import assign_add
from .assign import assign
from .batch_matmul import batch_matmul
from .weight_quant_batchmatmul import weight_quant_batchmatmul
from .bias_add_grad import bias_add_grad
from .cast import cast
from .drop_out_do_mask import drop_out_do_mask
from .fused_minimum_or_maximum_grad import fused_minimum_or_maximum_grad_cce
from .gather_v2_d import gather_v2_d
from .gelu_grad import gelu_grad
from .gelu import gelu
from .fast_gelu_v2 import fast_gelu_v2
from .greater_equal import greater_equal
from .layer_norm_grad import layer_norm_grad
from .layer_norm import layer_norm
from .instance_norm import instance_norm
from .instance_norm_grad import instance_norm_grad
from .in_training_reduce_grad import in_training_reduce_grad
from .in_training_update_grad import in_training_update_grad
from .in_training_update_grad_gamma_beta import in_training_update_grad_gamma_beta
from .less import less
from .log_softmax_grad import log_softmax_grad
from .log_softmax_v2 import log_softmax_v2
from .mat_mul import mat_mul
from .compress_mat_mul import compress_mat_mul
from .real_div import real_div
from .one_hot_d import one_hot_d
from .pad_d import pad_d
from .softmax_v2 import softmax_v2
from .sqrt import sqrt
from .strided_slice_d import strided_slice_d
from .tanh_grad import tanh_grad
from .tanh import tanh
from .mem_set import mem_set
from .tile_d import tile_d
from .tile import tile
from .transpose_d import transpose_d
from .unsorted_segment_sum_d import unsorted_segment_sum_d
from .maximum import maximum
from .minimum import minimum
from .reciprocal import reciprocal
from .add_n import add_n
from .sub import sub
from .square import square
from .confusion_softmax_grad import confusion_softmax_grad
from .confusion_transpose_d import confusion_transpose_d
from .bn_training_update_v2 import bn_training_update_v2
from .roi_align_true import roi_align_true
from .roi_align_grad_pt import roi_align_grad_pt

# hz it1
from .asin_grad import asin_grad
from .acos_grad import acos_grad
from .abs_grad import abs_grad
from .reduce_any_d import reduce_any_d
from .accumulate_nv2 import accumulate_nv2
from .approximate_equal import approximate_equal

# hz it2
from .atanh import atanh
from .relu import relu
from .relu_v2 import relu_v2
from .assign_sub import assign_sub
from .logical_or import logical_or
from .asin import asin
from .acos import acos
from .bessel_i0e import bessel_i0e
from .bessel_i1e import bessel_i1e
from .fill_d import fill_d
from .dequantize import dequantize
from .gather_nd import gather_nd

# hz it2_2
from .apply_adadelta_d import apply_adadelta_d
from .apply_ada_max_d import apply_ada_max_d
from .apply_gradient_descent import apply_gradient_descent
from .apply_ftrl_d import apply_ftrl_d
from .apply_momentum_d import apply_momentum_d
from .apply_add_sign_d import apply_add_sign_d
from .apply_proximal_gradient_descent import apply_proximal_gradient_descent
from .data_format_dim_map import data_format_dim_map
from .apply_adagrad_d import apply_adagrad_d
from .apply_centered_rms_prop_d import apply_centered_rms_prop_d
from .apply_power_sign_d import apply_power_sign_d

# hz it8
from .unpack import unpack
from .xdivy_grad import xdivy_grad
from .sgd import sgd
from .top_k_d import top_k_d
from .xlogy_grad import xlogy_grad
from .strided_slice_assign_d import strided_slice_assign_d
from .in_top_k import in_top_k
from .nms_with_mask import nms_with_mask
from .scatter_non_aliasing_add import scatter_non_aliasing_add
from .max_pool_grad_grad_with_argmax import max_pool_grad_grad_with_argmax
from .max_pool_grad_grad import max_pool_grad_grad
from .extract_image_patches import extract_image_patches
from .extract_volume_patches import extract_volume_patches
from .unsorted_segment_min_d import unsorted_segment_min_d
from .unsorted_segment_prod_d import unsorted_segment_prod_d
from .max_pool_grad import max_pool_grad
from .max_pool3d_grad import max_pool3d_grad

# hz it18
from .unsorted_segment_max_d import unsorted_segment_max_d

# tbe nj
from .diag_d import diag_d
from .less_equal import less_equal
from .equal import equal
from .l2_normalize import l2_normalize
from .l2_normalize_grad import l2_normalize_grad
from .reduce_max_d import reduce_max_d
from .flatten import flatten
from .prelu_grad import prelu_grad
from .prelu import prelu
from .pack import pack
from .read_select import read_select
from .write_select import write_select
from .load_to_l1 import load_to_l1
from .store_to_gm import store_to_gm

from .apply_adam_d import apply_adam_d
from .apply_adagrad_da_d import apply_adagrad_da_d
from .cumsum_d import cumsum_d
from .cumprod_d import cumprod_d
from .space_to_batch_d import space_to_batch_d
from .minimum_grad import minimum_grad
from .logical_and import logical_and
from .inplace_update_d import inplace_update_d
from .apply_ftrl_v2_d import apply_ftrl_v2_d
from .gather import gather
from .concat_offset_d import concat_offset_d
from .scatter_nd_d import scatter_nd_d
from .population_count import population_count
from .inplace_add_d import inplace_add_d
from .inplace_sub_d import inplace_sub_d
from .apply_proximal_adagrad_d import apply_proximal_adagrad_d
from .batch_to_space_d import batch_to_space_d
from .lrn_grad import lrn_grad
from .basic_lstm_cell import basic_lstm_cell
from .basic_lstm_cell_c_state_grad_v2 import basic_lstm_cell_c_state_grad_v2
from .dynamic_lstm import dynamic_lstm
from .dynamic_rnn import dynamic_rnn
from .dynamic_rnn_v2 import dynamic_rnn_v2
from .dynamic_rnn_v3 import dynamic_rnn_v3
from .dynamic_lstm_v2 import dynamic_lstm_v2
from .dynamic_gru import dynamic_gru
from .dynamic_gru_v2 import dynamic_gru_v2
from .square_sum_v1 import square_sum_v1
from .square_sum_v2 import square_sum_v2
from .clip_by_norm_no_div_sum import clip_by_norm_no_div_sum
from .space_to_depth import space_to_depth
from .top_k_pq_distance_merge import top_k_pq_distance_merge
# caffe it2
from .yolo_v3_detection_output_d import yolo_v3_detection_output_d
from .upsample import upsample
from .yolo import yolo
from .proposal_d import proposal_d
from .fully_connection import fully_connection
from .compress_fully_connection import compress_fully_connection
from .roi_pooling import roi_pooling
from .fsr_detection_output import fsr_detection_output
# fused op
from .fused_mul_apply_momentum import fused_mul_apply_momentum
from .fused_mul_apply_keras_momentum import fused_mul_apply_keras_momentum
from .fused_mul_add_n import fused_mul_add_n
from .fused_mul_add_add import fused_mul_add_add
from .fused_mul_addn_l2_loss import fused_mul_addn_l2loss
from .yolo_v2_detection_output_d import yolo_v2_detection_output_d
from .pass_through import pass_through
from .fused_mul_apply_momentum_extern import fused_mul_apply_momentum_extern
from .lars_v2_update import lars_v2_update
from .square_sum_all import square_sum_all
# 2DH1 op
from .clip_boxes_d import clip_boxes_d
from .decode_bbox import decode_bbox
from .to_absolute_bbox import to_absolute_bbox
from .decode_boundaries_target import decode_boundaries_target
from .decode_cornerpoints_target_bg import decode_cornerpoints_target_bg
from .decode_cornerpoints_target_wrt_center_v1 import decode_cornerpoints_target_wrt_center_v1
from .decode_wheels_target import decode_wheels_target
from .fastrcnn_predictions import fastrcnn_predictions
from .rpn_proposals_d import rpn_proposals_d
from .score_filter_pre_sort import score_filter_pre_sort
from .rpn_proposal_post_processing import rpn_proposal_post_processing
from .roi_align import roi_align
# caffe it3
from .ascend_quant import ascend_quant
from .ascend_dequant import ascend_dequant
from .strided_read import strided_read
from .strided_write import strided_write
from .tile_with_axis import tile_with_axis
from .lrn import lrn
from .power import power
from .reduction import reduction
from .normalize_sum import normalize_sum
from .normalize_scale import normalize_scale
from .crop import crop
from .mvn import mvn
from .mvn_v2 import mvn_v2
from .threshold import threshold
from .prior_box_d import prior_box_d
from .scale import scale
from .bnll import bnll
from .shuffle_channel import shuffle_channel
# caffe it4
from .bias import bias
from .arg_max_with_kd import arg_max_with_kd

from .ascend_dequant_s16 import ascend_dequant_s16
from .ascend_requant import ascend_requant
from .ascend_requant_s16 import ascend_requant_s16
from .dilation2d import dilation2d
from .dilation2d_backprop_filter import dilation2d_backprop_filter
from .dilation2d_backprop_input import dilation2d_backprop_input
from .euclidean_norm_d import euclidean_norm_d
from .mul_no_nan import mul_no_nan
from .adds import adds
from .muls import muls
from .fills import fills
from .axpy import axpy
from .cumulativelogsumexp_d import cumulative_logsumexp_d
from .avg_pool3d_d import avg_pool3d_d
from .avg_pool_1d import avg_pool_1d
from .kl_div import kl_div
from .stn_pre import stn_pre
from .stn_compute import stn_compute
# caffe it6
from .prior_box_d_v2 import prior_box_d_v2
from .fill import fill
from .im2col import im2col
from .trans_data import trans_data
from .act_ulq_clamp_max_grad import act_ulq_clamp_max_grad
from .act_ulq_clamp_min_grad import act_ulq_clamp_min_grad
from .acts_ulq_input_grad import acts_ulq_input_grad
from .acts_ulq import acts_ulq
from .ifmr import ifmr

from .update_tensor_desc import update_tensor_desc
from .reduce_mean_variance import reduce_mean_variance
from .fused_bn2_reluv2_conv2d_bn1 import fused_bn2_reluv2_conv2d_bn1
from .celu import celu
from .attention_ln_qkv import attention_ln_qkv
from .attention_qkv_grad_x import attention_qkv_grad_x
from .attention_qkv_grad_w import attention_qkv_grad_w
from .kl_div_loss_grad import kl_div_loss_grad
from .hard_sigmoid_grad import hard_sigmoid_grad
from .layer_norm_beta_gamma_backprop_v2_unify import layer_norm_beta_gamma_backprop_v2_unify

# hdrnet
from .trans_argb import trans_argb
from .ada_cast import ada_cast

from .basic_lstm_inplace_fill_window_cache import basic_lstm_inplace_fill_window_cache
from .basic_gru_inplace_fill_window_cache import basic_gru_inplace_fill_window_cache