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
comput schedule init
"""
from .api import broadcast
from .api import ceil
from .api import floor
from .api import round
from .api import trunc
from .api import round_half_up
from .api import round_to
from .api import cast_to
from .api import concat
from .api import vmuls
from .api import vadds
from .api import vlog
from .api import vexp
from .api import vabs
from .api import vrec
from .api import vrelu
from .api import vnot
from .api import vsqrt
from .api import vrsqrt
from .api import vdiv
from .api import vmul
from .api import vadd
from .api import vsub
from .api import vmin
from .api import vmax
from .api import vor
from .api import vand
from .api import vaxpy
from .api import vmla
from .api import vmadd
from .api import vmaddrelu
from .api import vmaxs
from .api import vmins
from .api import vcmp
from .api import vlogic
from .api import vsel
from .api import vcmpsel
from .api import vmod
from .api import vlrelu
from .api import vaddrelu
from .api import vsubrelu
from .api import sum
from .api import reduce_min
from .api import reduce_max
from .api import reduce_prod
from .api import pooling2d
from .api import pooling3d
from .api import max_pooling3d_grad_grad
from .api import inplace_add
from .api import inplace_sub
from .api import inplace_update
from .api import split
from .api import split_compute_com
from .api import split_schedule_com
from .api import tuple_sum
from .api import unsorted_segment_max
from .api import unsorted_segment_min
from .api import unsorted_segment_sum
from .api import unsorted_segment_mean
from .api import unsorted_segment_prod
from .api import cce_build_code
from .api import auto_schedule
from .api import pooling3d_max_grad_grad

from .te_compute.common import cast_to_round
from .te_compute.common import calculate_one_or_zero
from .te_compute.conv_compute import conv
from .te_compute.depthwise_conv2d_compute import depthwise_conv2d_backprop_filter_d_compute
from .te_compute.depthwise_conv2d_compute import depthwise_conv2d_backprop_input_d_compute
from .te_compute.depthwise_conv2d_compute import depthwise_conv2d_compute
from .te_compute.conv_compute import check_conv_shape
from .te_compute.conv_compute import conv_compress
from .te_compute.conv_compute import is_support_v200
from .te_compute.max_pool2d_3_2_fusion_compute import max_pool_compute
from .te_compute.dim_conv import compute_four2five
from .te_compute.dim_conv import compute_five2four
from .te_compute import util
from .te_compute.mmad_compute import matmul
from .te_compute.mmad_compute import get_matmul_performance_format
from .te_compute.gemm_compute import gemm
from .te_compute.pooling2d_compute import get_caffe_out_size_and_pad
from .te_compute.conv2d_backprop_filter_compute import conv2d_backprop_filter_compute
from .te_compute.conv2d_backprop_input_compute import conv2d_backprop_input_compute
from .te_compute.conv3d_compute import conv3d
from .te_compute.conv3d_backprop_input_compute import conv3d_dx
from .te_compute.conv3d_backprop_filter_compute import conv3d_dw
from .te_compute.util import dsl_check_support

from .te_schedule.cce_schedule import get_op_info

from .te_schedule.cce_schedule import schedule_cce
from te.utils.cce import build
from .te_compute.dilation_compute import dilation_compute
