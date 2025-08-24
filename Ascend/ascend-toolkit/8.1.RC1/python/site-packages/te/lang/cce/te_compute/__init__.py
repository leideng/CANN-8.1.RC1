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
cce compute API:
In order to simplify the  procedure of writing schedule,TE provides a set of
TensorEngine APIs.
Using those API to develop operators, you can use the "Auto_schedule" create
schedule.
"""
# 'pylint: disable=redefined-builtin
from .broadcast_compute import broadcast
from .cast_compute import ceil
from .cast_compute import floor
from .cast_compute import round
from .cast_compute import trunc
from .cast_compute import round_half_up
from .common import round_to
from .common import cast_to
from .common import cast_to_round
from .concat_compute import concat
from .conv_compute import conv
from .conv_compute import check_conv_shape
from .conv_compute import conv_compress
from .conv_compute import is_support_v200
from .dim_conv import compute_four2five
from .dim_conv import compute_five2four
from .elewise_compute import vmuls
from .elewise_compute import vadds
from .elewise_compute import vlog
from .elewise_compute import vexp
from .elewise_compute import vabs
from .elewise_compute import vrec
from .elewise_compute import vrelu
from .elewise_compute import vnot
from .elewise_compute import vsqrt
from .elewise_compute import vrsqrt
from .elewise_compute import vdiv
from .elewise_compute import vmul
from .elewise_compute import vadd
from .elewise_compute import vsub
from .elewise_compute import vmin
from .elewise_compute import vmax
from .elewise_compute import vor
from .elewise_compute import vand
from .elewise_compute import vaxpy
from .elewise_compute import vmla
from .elewise_compute import vmadd
from .elewise_compute import vmaddrelu
from .elewise_compute import vmaxs
from .elewise_compute import vmins
from .elewise_compute import vcmp
from .elewise_compute import vlogic
from .elewise_compute import vsel
from .elewise_compute import vcmpsel
from .elewise_compute import vmod
from .elewise_compute import vlrelu
from .elewise_compute import vaddrelu
from .elewise_compute import vsubrelu
from .reduction_compute import sum
from .reduction_compute import reduce_min
from .reduction_compute import reduce_max
from .reduction_compute import reduce_prod
from .segment_compute import unsorted_segment_max
from .segment_compute import unsorted_segment_min
from .segment_compute import unsorted_segment_sum
from .segment_compute import unsorted_segment_mean
from .segment_compute import unsorted_segment_prod
from .depthwise_conv2d_compute import depthwise_conv2d_backprop_filter_d_compute
from .depthwise_conv2d_compute import depthwise_conv2d_backprop_input_d_compute
from .depthwise_conv2d_compute import depthwise_conv2d_compute
from .inplace_compute import inplace_add
from .inplace_compute import inplace_sub
from .inplace_compute import inplace_update
