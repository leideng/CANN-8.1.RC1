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
fixpipe
"""
import json
from typing import List
from tbe import tvm
from tbe.tvm import Tensor
from tbe.dsl import auto_schedule
from tbe.common.utils import log
from tbe.common.utils import para_check
from tbe.common.utils import shape_util
from tbe.common.register import register_op_compute
from impl.fixpipe_op.fixpipe_factory import FixpipeFactory
from impl.fixpipe_op import fixpipe_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import get_current_build_config

N_CONV2D = 0
C1 = 1
H_CONV2D = 2
W_CONV2D = 3
N_CONV3D = 0
D_CONV3D = 1
H_CONV3D = 3
W_CONV3D = 4


@register_op_compute("fix_pipe", op_mode="static", support_fusion=True)
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
    if not get_current_build_config("enable_op_prebuild"):
            error_manager_vector.raise_err_specific_reson(
                "fix_pipe", "fix_pipe does not support single op compilation!"
                )
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
    n_axis = 0
    if output.get("format") in ["NC1HWC0", "NDHWC"]:
        h_axis = 2
        w_axis = 3
    elif output.get("format") == "NHWC":
        h_axis = 1
        w_axis = 2
    elif output.get("format") == "NDC1HWC0":
        h_axis = 3
        w_axis = 4
    elif output.get("format") == "FRACTAL_Z":
        # FRACTAL_Z shape is Cin1HkWk, Cout1, Cout0, Cin0
        ci_axis = 0
        co_axis = 1
    else:
        slice_info = {"_op_slice_info":{"splitMaps": [], "reduceMaps": [], "l1FusionEnable": 2, "minTbeL1Space": 0}}
        return json.dumps(slice_info)

    if output.get("format") == "FRACTAL_Z":
        slice_info = {"_op_slice_info":
                        {"splitMaps": [{"inputList": [],
                                        "outputList": [{"idx": 0, "axis": [ci_axis]}]},  # filter c_in
                                        {"inputList": [],
                                        "outputList": [{"idx": 0, "axis": [co_axis]}]}], # filter c_out
                        "reduceMaps": [],
                        "l1FusionEnable": 2,
                        "minTbeL1Space": 0}}
    else:
        slice_info = {"_op_slice_info":
                        {"splitMaps": [{"inputList": [],
                                        "outputList": [{"idx": 0, "axis": [n_axis]}]},  # fmap batch
                                        {"inputList": [],
                                        "outputList": [{"idx": 0, "axis": [h_axis]}]},  # fmap H
                                        {"inputList": [],
                                        "outputList": [{"idx": 0, "axis": [w_axis]}]}], # fmap W
                        "reduceMaps": [],
                        "l1FusionEnable": 2,
                        "minTbeL1Space": 0}}
        if output.get("format") in ["NDHWC", "NDC1HWC0"]:
            d_axis = 1
            slice_info["_op_slice_info"]["splitMaps"].insert(1, {"inputList": [],
                                                                 "outputList": [{"idx": 0, "axis": [d_axis]}]})

    slice_info = add_input_split_info(slice_info, x1, x2)

    shape_x1 = x1.get("ori_shape")
    shape_x1 = shape_util.scalar2tensor_one(shape_x1)

    # shape is [-2], all axes do not support split
    if list(shape_x1) == [-2]:
        slice_info.get('_op_slice_info').get("splitMaps").clear()
        return json.dumps(slice_info)

    # H/W shape is -1, remove corresponding split info
    format_x1 = x1.get("ori_format")
    if output.get("format") == "FRACTAL_Z":
        # HWC is the same axis in FRACTAL_Z
        unsupported_dynamic_axis = {"H": [0], "W": [0], "C": [0]}
    elif output.get("format") in ["NDHWC", "NDC1HWC0"]:
        unsupported_dynamic_axis = {"D": [1], "H": [3], "W": [4]}
    else:
        unsupported_dynamic_axis = {"H": [2], "W": [3]}
    temp_info = slice_info.get('_op_slice_info').get("splitMaps")
    for name, index in unsupported_dynamic_axis.items():
        if shape_x1[format_x1.find(name)] == -1:
            filtered_info = filter(lambda splits : splits["inputList"][0]["axis"] != index, temp_info)
            temp_info = list(filtered_info)
    # when with channel_wise tensor, the Cin can not split
    with_channel_tensor = False
    for channel_tensor in (quant_scale_0, relu_weight_0, quant_scale_1, relu_weight_1):
        if fixpipe_util.is_vector_input(channel_tensor):
            with_channel_tensor = True
            break
    if output.get("format") == "FRACTAL_Z" and with_channel_tensor:
        filtered_info = filter(lambda splits : splits["inputList"][0]["axis"] != 0, temp_info)
        temp_info = list(filtered_info)

    try:
        slice_info["_op_slice_info"]["splitMaps"] = temp_info
    except KeyError:
        error_detail = "Key(_op_slice_info or splitMaps) not in the dict"
        error_manager_vector.raise_err_specific_user("fixpipe", error_detail)
    return json.dumps(slice_info)


def add_input_split_info(slice_info: dict, x1: dict, x2: (dict, None)):
    if x1.get("format") == "NDC1HWC0":
        x1_batch_info = [{"idx": 0, "axis": [N_CONV3D], "headOverLap": [-1], "tailOverLap": [-1]}]
        x1_d_info = [{"idx": 0, "axis": [D_CONV3D], "headOverLap": [-1], "tailOverLap": [-1]}]
        x1_h_info = [{"idx": 0, "axis": [H_CONV3D], "headOverLap": [-1], "tailOverLap": [-1]}]
        x1_w_info = [{"idx": 0, "axis": [W_CONV3D], "headOverLap": [-1], "tailOverLap": [-1]}]
        slice_info['_op_slice_info']["splitMaps"][0]["inputList"].extend(x1_batch_info)
        slice_info['_op_slice_info']["splitMaps"][1]["inputList"].extend(x1_d_info)
        slice_info['_op_slice_info']["splitMaps"][2]["inputList"].extend(x1_h_info)
        slice_info['_op_slice_info']["splitMaps"][3]["inputList"].extend(x1_w_info)

        if x2:
            x2_batch_info = [{"idx": 1, "axis": [N_CONV3D], "headOverLap": [-1], "tailOverLap": [-1]}]
            x2_d_info = [{"idx": 1, "axis": [D_CONV3D], "headOverLap": [-1], "tailOverLap": [-1]}]
            x2_h_info = [{"idx": 1, "axis": [H_CONV3D], "headOverLap": [-1], "tailOverLap": [-1]}]
            x2_w_info = [{"idx": 1, "axis": [W_CONV3D], "headOverLap": [-1], "tailOverLap": [-1]}]
            slice_info['_op_slice_info']["splitMaps"][0]["inputList"].extend(x2_batch_info)
            slice_info['_op_slice_info']["splitMaps"][1]["inputList"].extend(x2_d_info)
            slice_info['_op_slice_info']["splitMaps"][2]["inputList"].extend(x2_h_info)
            slice_info['_op_slice_info']["splitMaps"][3]["inputList"].extend(x2_w_info)
    if x1.get("format") == "NC1HWC0":
        x1_batch_info = [{"idx": 0, "axis": [N_CONV2D], "headOverLap": [-1], "tailOverLap": [-1]}]
        x1_h_info = [{"idx": 0, "axis": [H_CONV2D], "headOverLap": [-1], "tailOverLap": [-1]}]
        x1_w_info = [{"idx": 0, "axis": [W_CONV2D], "headOverLap": [-1], "tailOverLap": [-1]}]
        slice_info['_op_slice_info']["splitMaps"][0]["inputList"].extend(x1_batch_info)
        slice_info['_op_slice_info']["splitMaps"][1]["inputList"].extend(x1_h_info)
        slice_info['_op_slice_info']["splitMaps"][2]["inputList"].extend(x1_w_info)

        if x2:
            x2_batch_info = [{"idx": 1, "axis": [N_CONV2D], "headOverLap": [-1], "tailOverLap": [-1]}]
            x2_h_info = [{"idx": 1, "axis": [H_CONV2D], "headOverLap": [-1], "tailOverLap": [-1]}]
            x2_w_info = [{"idx": 1, "axis": [W_CONV2D], "headOverLap": [-1], "tailOverLap": [-1]}]
            slice_info['_op_slice_info']["splitMaps"][0]["inputList"].extend(x2_batch_info)
            slice_info['_op_slice_info']["splitMaps"][1]["inputList"].extend(x2_h_info)
            slice_info['_op_slice_info']["splitMaps"][2]["inputList"].extend(x2_w_info)
    if x1.get("format") == "FRACTAL_Z":
        x1_ci_info = [{"idx": 0, "axis": [0], "headOverLap": [-1], "tailOverLap": [-1]}]
        x1_co_info = [{"idx": 0, "axis": [1], "headOverLap": [-1], "tailOverLap": [-1]}]
        slice_info['_op_slice_info']["splitMaps"][0]["inputList"].extend(x1_ci_info)
        slice_info['_op_slice_info']["splitMaps"][1]["inputList"].extend(x1_co_info)
        if x2:
            x2_ci_info = [{"idx": 1, "axis": [0], "headOverLap": [-1], "tailOverLap": [-1]}]
            x2_co_info = [{"idx": 1, "axis": [1], "headOverLap": [-1], "tailOverLap": [-1]}]
            slice_info['_op_slice_info']["splitMaps"][0]["inputList"].extend(x2_ci_info)
            slice_info['_op_slice_info']["splitMaps"][1]["inputList"].extend(x2_co_info)
    return slice_info
