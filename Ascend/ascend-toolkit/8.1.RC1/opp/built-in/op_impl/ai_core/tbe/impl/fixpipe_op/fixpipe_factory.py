#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
fixpipe factory class
"""
from typing import List
from tbe.tvm import Tensor
from impl.fixpipe_op.fixpipe_matmul import FixpipeMatmul
from impl.fixpipe_op.fixpipe_conv2d import FixpipeConv2d
from impl.fixpipe_op.fixpipe_conv2d_wino import FixpipeConv2dWino
from impl.fixpipe_op.fixpipe_conv3d import FixpipeConv3d
from impl.fixpipe_op.fixpipe_conv2d_backprop_input import FixpipeConv2dBackpropInput
from impl.fixpipe_op.fixpipe_conv3d_backprop_input import FixpipeConv3dBackpropInput
from impl.fixpipe_op.fixpipe_conv2d_backprop_filter import FixpipeConv2dBackpropFilter

FIXPIPE_OP_SUPPORT_MAP = {
    "matmul": FixpipeMatmul,
    "conv2d": FixpipeConv2d,
    "conv2d_wino": FixpipeConv2dWino,
    "conv3d": FixpipeConv3d,
    "conv2d_backprop_input": FixpipeConv2dBackpropInput,
    "conv3d_backprop_input": FixpipeConv3dBackpropInput,
    "conv2d_backprop_filter": FixpipeConv2dBackpropFilter,
}


class FixpipeFactory:
    """
    FixpipeFactory
    """

    @staticmethod
    def get_fixpipe(op_type: str, x: Tensor, x1: (Tensor, None), quant_scale_0: (Tensor, None),
                    relu_weight_0: (Tensor, None), clip_value_0: (Tensor, None), quant_scale_1: (Tensor, None),
                    relu_weight_1: (Tensor, None), clip_value_1: (Tensor, None), anti_quant_scale: (Tensor, None),
                    anti_quant_offset: (Tensor, None), output: dict, fusion_op_list: List[str], unit_list: List[str],
                    eltwise_mode: str):
        if op_type not in FIXPIPE_OP_SUPPORT_MAP.keys():
            raise RuntimeError("[{}] not support fixpipe fusion".format(op_type))
        else:
            fixpipe_cube = FIXPIPE_OP_SUPPORT_MAP.get(op_type)

        fixpipe = fixpipe_cube(op_type, x, x1, quant_scale_0, relu_weight_0, clip_value_0, quant_scale_1, relu_weight_1,
                               clip_value_1, anti_quant_scale, anti_quant_offset, output, fusion_op_list, unit_list,
                               eltwise_mode)
        return fixpipe
