# -*- coding: UTF-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
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
_internal_types.py
"""
import enum


class VecGatherCmdType(enum.Enum):
    """
    define some params
    """
    # those cmds share same interface treat as one type
    # params are mask, dst, src, repeat_times, dst_blk_stride, src_blk_stride,
    #            dst_rep_stride, src_rep_stride
    SINGLE = 0
    # params are mask, dst, src0, src1, repeat_times, dst_blk_stride,
    #            src0_blk_stride, src1_blk_stride, dst_rep_stride,
    #            src0_rep_stride, src1_rep_stride
    DBL_TRI = 1
    # params are mask, dst, src, scalar, repeat_times, dst_blk_stride,
    #            src_blk_stride, dst_rep_stride, src_rep_stride
    SCA_DBL_TRI = 2
    # params are mask, round mode, dst, src, repeat_times, dst_blk_stride,
    #            src_blk_stride, dst_rep_stride, src_rep_stride, deqscale
    CONV = 3
    # params are mask, dst, scalar, repeat_times, dst_blk_stride, dst_rep_stride
    INIT = 4
