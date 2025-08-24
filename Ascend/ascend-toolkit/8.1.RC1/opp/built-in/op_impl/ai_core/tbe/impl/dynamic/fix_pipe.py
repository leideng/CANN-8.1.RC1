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
dynamic fixpipe
"""
from typing import List
from tbe import tvm
from tbe.tvm import Tensor
from tbe.common.register import register_op_compute
from tbe.common.register import register_operator
from tbe.common.utils import log
from tbe.common.utils import para_check
from tbe.common.utils import shape_util
from tbe.dsl import auto_schedule
from impl.fixpipe_op.fixpipe_factory import FixpipeFactory
from impl.fixpipe_op import fixpipe_util
from impl.fix_pipe import get_op_support_info as get_op_info_func


@register_operator("FixPipe")
@para_check.check_input_type(dict, (dict, para_check.NONE_TYPE), (dict, para_check.NONE_TYPE),
                             (dict, para_check.NONE_TYPE), (dict, para_check.NONE_TYPE),
                             (dict, para_check.NONE_TYPE), (dict, para_check.NONE_TYPE),
                             (dict, para_check.NONE_TYPE), (dict, para_check.NONE_TYPE),
                             (dict, para_check.NONE_TYPE), dict, (list, tuple), (list, tuple),
                             str, str)
def fix_pipe(x1: dict, x2: (dict, None), quant_scale_0: (dict, None), relu_weight_0: (dict, None),
             clip_value_0: (dict, None),
             quant_scale_1: (dict, None), relu_weight_1: (dict, None), clip_value_1: (dict, None),
             anti_quant_scale: (dict, None), anti_quant_offset: (dict, None), output: dict,
             fusion_op_list: (List[str], tuple), unit_list: (List[str], tuple), eltwise_mode: str,
             kernel_name="fixpipe"):
    """
    :param x1:
    :param x2:
    :param quant_scale_0:
    :param relu_weight_0:
    :param clip_value_0:
    :param quant_scale_1:
    :param relu_weight_1:
    :param clip_value_1:
    :param anti_quant_scale:
    :param anti_quant_offset:
    :param output:
    :param fusion_op_list:
    :param unit_list:
    :param eltwise_mode:
    :return:
    """
    _ = x2
    _ = quant_scale_0
    _ = relu_weight_0
    _ = clip_value_0
    _ = quant_scale_1
    _ = relu_weight_1
    _ = clip_value_1
    _ = anti_quant_scale
    _ = anti_quant_offset
    _ = output
    _ = fusion_op_list
    _ = unit_list
    _ = eltwise_mode
    _ = kernel_name

    fixpipe_util.check_fixpipe_support()
    input_x = tvm.placeholder(x1.get("shape"), x1.get("dtype"), name="x1")

    # fake compute for pre_build
    res = tvm.compute(x1.get("shape"),
                      lambda *indice: input_x(*indice),
                      name="fixpipe",
                      tag="fixpipe")

    with tvm.target.cce():
        auto_schedule(res)


def get_op_support_info(x1: dict, x2: (dict, None), quant_scale_0: (dict, None), relu_weight_0: (dict, None),
                        clip_value_0: (dict, None), quant_scale_1: (dict, None), relu_weight_1: (dict, None),
                        clip_value_1: (dict, None), anti_quant_scale: (dict, None), anti_quant_offset: (dict, None),
                        output: dict, fusion_op_list: (List[str], tuple), unit_list: (List[str], tuple),
                        eltwise_mode: str, kernel_name="fixpipe"):
    params_list = [
        x1, x2, quant_scale_0, relu_weight_0, clip_value_0, quant_scale_1, relu_weight_1,
        clip_value_1, anti_quant_scale, anti_quant_offset, output, fusion_op_list, unit_list,
        eltwise_mode, kernel_name
    ]
    res_info = get_op_info_func(*params_list)
    return res_info


@register_op_compute("fix_pipe", op_mode="dynamic", support_fusion=True)
@register_op_compute("FixPipe", op_mode="dynamic", support_fusion=True)
def fixpipe_compute(x1: Tensor, x2: (Tensor, None), quant_scale_0: (Tensor, None),
                    relu_weight_0: (Tensor, None),
                    clip_value_0: (Tensor, None), quant_scale_1: (Tensor, None),
                    relu_weight_1: (Tensor, None),
                    clip_value_1: (Tensor, None), anti_quant_scale: (Tensor, None),
                    anti_quant_offset: (Tensor, None),
                    output: dict, fusion_op_list: List[str],
                    unit_list: List[str], eltwise_mode: str):
    """
    :param x1:
    :param x2:
    :param quant_scale_0:
    :param relu_weight_0:
    :param clip_value_0:
    :param quant_scale_1:
    :param relu_weight_1:
    :param clip_value_1:
    :param anti_quant_scale:
    :param anti_quant_offset:
    :param output:
    :param fusion_op_list:
    :param unit_list:
    :param eltwise_mode:
    :return:
    """

    log.debug("input param for fixpipe: \n x1: {}, \n x2: {}, \n quant_scale_0: {}, \n"
              "relu_weight_0: {},\n clip_value_0: {},\n quant_scale_1: {},\n relu_weight_1: {},\n"
              "clip_value_1: {},\n anti_quant_scale: {},\n anti_quant_offset: {},\n output: {},\n"
              "fusion_op_list: {},\n unit_list: {},\n eltwise_mode: {}\n".format(x1, x2,
                                                                                 quant_scale_0,
                                                                                 relu_weight_0,
                                                                                 clip_value_0,
                                                                                 quant_scale_1,
                                                                                 relu_weight_1,
                                                                                 clip_value_1,
                                                                                 anti_quant_scale,
                                                                                 anti_quant_offset,
                                                                                 output,
                                                                                 fusion_op_list,
                                                                                 unit_list,
                                                                                 eltwise_mode))
    fixpipe_util.check_fixpipe_support()
    op_type = fixpipe_util.get_op_type(x1)
    log.debug("fixpipe fusion for op [{}]".format(op_type))

    fixpipe = FixpipeFactory.get_fixpipe(op_type, x1, x2,
                                         quant_scale_0, relu_weight_0, clip_value_0,
                                         quant_scale_1, relu_weight_1, clip_value_1,
                                         anti_quant_scale, anti_quant_offset, output,
                                         fusion_op_list, unit_list, eltwise_mode)

    res = fixpipe.fixpipe_compute()

    fixpipe_util.set_build_cfg()

    return res
