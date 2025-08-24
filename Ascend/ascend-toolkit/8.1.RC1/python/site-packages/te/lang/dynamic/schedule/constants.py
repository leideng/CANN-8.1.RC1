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
auto_schedule template, if user call auto_schedule, this file will choose a
corresponding schedule template for user's compute
"""


class Pattern:
    """
    Built-in Patterns
    """
    OPAQUE = "Opaque"
    CONV2D = "Convolution"
    CONV2D_BACKPROP_INPUT = "Conv2d_backprop_input"
    CONV2D_BACKPROP_FILTER = "Conv2d_backprop_filter"
    MAT_MUL = "Matmul"
    BATCH_MATMUL = "BatchMatmul"
    LAYER_NORM_BETA_GAMMA_BACKPROP = "Layer_norm_beta_gamma_backprop"
    CONV3D = "conv3d"
    CONV3D_BACKPROP_INPUT = "Conv3d_backprop_input"
