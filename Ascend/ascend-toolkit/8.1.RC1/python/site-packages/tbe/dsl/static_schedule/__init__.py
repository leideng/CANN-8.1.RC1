#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
cce schedule
"""
from .cce_schedule import schedule_cce
from .cce_schedule import cce_build_code
from .depthwise_conv2d_schedule import depthwise_conv2d_backprop_filter_d_schedule
from .depthwise_conv2d_schedule import depthwise_conv2d_backprop_input_d_schedule
from .depthwise_conv2d_schedule import depthwise_conv2d_schedule
from .pooling2d_schedule import pooling2d_schedule
from .pooling3d_schedule import pooling3d_schedule
from .pooling3d_max_grad_grad_schedule import pooling3d_max_grad_grad_schedule
from .reduce_mean_mid_reduce_high_performance_schedule import *
from .mmad_transpose_intrin import dma_copy_matmul_transpose
from .reduce_5hdc_intrin import *
