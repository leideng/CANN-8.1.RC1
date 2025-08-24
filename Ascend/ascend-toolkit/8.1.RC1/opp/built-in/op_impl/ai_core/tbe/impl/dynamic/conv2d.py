#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
dynamic conv2d
"""
from __future__ import absolute_import

import math
import re
import json
import copy
import warnings
import tbe.dsl as tbe_base
from tbe import tvm
from tbe.dsl import auto_schedule
from tbe.dsl import build
from tbe.dsl.base.classifier.conv2d_classifier import cube_forward_op_classify
from tbe.dsl.compute.conv_compute import conv
from tbe.common.platform import CUBE_MKN
from tbe.common.register import set_fusion_buildcfg
from tbe.common.register import register_op_compute
from tbe.common.register import register_operator
from tbe.common.register import register_param_generalization
from tbe.common.utils import para_check
from tbe.common.utils import shape_util
from tbe.common.utils import log
from tbe.common.utils.errormgr import error_manager_cube as err_man
from tbe.common.utils.errormgr import error_manager_util
from tbe.common.utils.op_util.op_util_conv2d import is_support_fixpipe
from tbe.common.utils.op_util.op_util_conv2d import check_nosupport_binary_op
from tbe.common.utils.op_util.op_util_conv2d import check_nosupport_binary_dtype
from impl.util import fusion_util
from impl.util import util_conv2d
from impl.util.util_conv2d_dynamic import Conv2dParaProcess
from impl.util.util_conv2d_dynamic import modify_input_range
from impl.util.util_conv2d_dynamic import check_l1_size
from impl.util.util_conv2d_dynamic import check_support_dtype
from impl.util.util_conv2d_dynamic import enable_db_fold
from impl.util.util_conv2d_dynamic import create_fuzz_range
from impl.util.util_conv2d_dynamic import correct_input_range
from impl.util.util_conv2d_dynamic import get_format_attr
from impl.util.util_conv2d_dynamic import check_graph_mode
from impl.util.util_conv2d_dynamic import check_conv2d_range
from impl.util.util_cube_dynamic import BIT_RATIO_DICT
from impl.util.platform_adapter import tbe_platform

NONETYPE = type(None)
H_DIM = 2
W_DIM = 3
SHAPE_LEN = 5
ORI_SHAPE_LEN = 4
DYNAMIC_VALUE = -1


def gen_conv2d_range(inputs):
    """
    fuzz input range
    """
    op_type = "conv2d"
    x_format = inputs.get("ori_format")

    if x_format != "NCHW" and x_format != "NHWC":
        err_man.raise_err_specific_user(op_type, "input fmap format only support NCHW or NHWC.")

    x_range = [[1, -1], [1, -1], [1, -1], [1, -1]]
    return x_range


@register_param_generalization("Conv2D")
def conv2d_generalization(inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
                          groups=1, data_format='NHWC', offset_x=0, kernel_name="conv2d",
                          generalize_config=None):
    """
    conv2d generalization

    Notice
    ------
    run after infershape and before operator compile
    only modify input and output tensors with range

    for use:
        1. te fusion distinguish .o (remove the generalization dim)
        2. pass them to the operator to follow the dynanmic shape process

    Parameters
    ----------
    same to conv2d

    Returns
    -------
    list of params list:
        single item under "keep_rank" mode and multiple under "all_shape"
    """
    if generalize_config is None:
        generalize_config = {"mode": "keep_rank"}
    support_mode = ["keep_rank", "all_shape"]
    generalize_mode = generalize_config.get("mode")
    if generalize_mode not in support_mode:
        err_man.raise_err_specific_user("conv2d", "invalid generalize mode {}, only support {}".format(
            generalize_mode, support_mode))

    log.debug("conv2d generalization mode: %s, inputs: %s", generalize_mode, inputs)
    # unknow_rank inputs ori_shape is [-2], others' shape length is 4
    unknow_rank = len(inputs["ori_shape"]) == 1 and inputs["ori_shape"][0] == -2
    if unknow_rank:
        if generalize_mode == "all_shape":
            result = []
            result.append([inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
                           groups, data_format, offset_x, kernel_name])
            log.debug("conv2d generalization result: %s", result)
            return result
        else:
            err_man.raise_err_specific_user("conv2d", "not support unknow_rank under mode {}".format(generalize_mode))

    if weights.get("format") == "FRACTAL_Z_C04" and generalize_mode == "all_shape":
        result = []
        result.append([inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
                       groups, data_format, offset_x, kernel_name])
        log.debug("conv2d generalization result: %s", result)
        return result

    result = []
    log.debug("conv2d generalization mode: %s, inputs: %s", generalize_mode, inputs)
    if generalize_mode == "keep_rank":  # fuzz build situation
        x_range = gen_conv2d_range(inputs)
        inputs["ori_range"] = x_range
        have_range = {"inputs": inputs, "outputs": outputs}
        for name, tensor in have_range.items():
            # only change shape NHW dim to -1, range is already set at infershape
            valid = isinstance(tensor.get("ori_shape"), (list, tuple)) and len(tensor["ori_shape"]) == ORI_SHAPE_LEN
            if not valid:
                err_man.raise_err_specific_user("conv2d", "invalid {} ori_shape {}, only support {}d".format(
                    name, str(tensor.get("ori_shape")), str(ORI_SHAPE_LEN)))
            tensor["ori_shape"] = [-1, -1, -1, -1]

        weights["ori_range"] = [[1, -1], [1, -1], [1, -1], [1, -1]]
        weights["ori_shape"] = [DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE]

    elif generalize_mode == "all_shape":
        for tensor in [inputs, outputs]:
            if len(tensor.get("ori_shape")) != ORI_SHAPE_LEN or len(tensor.get("shape")) != SHAPE_LEN:
                err_man.raise_err_specific_user("conv2d", "check input shape size fail")
            # Has already guaranteed fp16(ci0 -> 16) / fp32(ci0 -> 8)
            dtype = tensor["dtype"]
            ci0 = CUBE_MKN[dtype]["mac"][1]
            tensor["shape"] = (DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE, ci0)
            tensor["ori_shape"] = (DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE)
            tensor["range"] = ((1, None), (1, None), (1, None), (1, None), (ci0, ci0))
            tensor["ori_range"] = ((1, None), (1, None), (1, None), (1, None))

        if len(weights.get("ori_shape")) != ORI_SHAPE_LEN or len(weights.get("shape")) != ORI_SHAPE_LEN:
            err_man.raise_err_specific_user("conv2d", "check weight shape size fail")
        *_, n0, k0 = weights.get("shape")
        weights["shape"] = (DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE)
        weights["ori_shape"] = (DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE)
        weights["range"] = ((1, None), (1, None), (1, None), (1, None))
        weights["ori_range"] = ((1, None), (1, None), (1, None), (1, None))

        for tensor in [bias, offset_w]:
            if tensor is not None:
                tensor["shape"] = (DYNAMIC_VALUE,)
                tensor["ori_shape"] = (DYNAMIC_VALUE,)
                tensor["range"] = ((1, None),)
                tensor["ori_range"] = ((1, None),)

                if tensor == bias:
                    tensor["format"] = "ALL"

        for attrs in [strides, pads, dilations]:
            if len(attrs) != ORI_SHAPE_LEN:
                err_man.raise_err_specific_user("conv2d", "check attrs shape size fail")

        strides = (DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE)
        pads = (DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE)
        dilations = (DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE)
        groups = None
        offset_x = None
        data_format = "ALL"

    result.append([inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
                   groups, data_format, offset_x, kernel_name])
    log.debug("conv2d generalization result: %s", result)
    return result


def set_default_para():
    """
    set default parameter value
    """
    default_para = {}

    default_para["optim_dict"] = {"c0_optim_flg": False,
                                  "use_v200_c04_flg": False,
                                  "v220_c04_mode": "disabled",
                                  "invalid_data_rm": False}
    default_para["fusion_para"] = {"input_memory_type": 0, "output_memory_type": 0,
                                   "valid_shape": (), "slice_offset": (),
                                   "l1_fusion_type": -1}
    default_para["ori_shape"] = [0, 0, 0, 0]
    return default_para


@register_op_compute("Conv2D", op_mode="dynamic", support_fusion=True)
@para_check.check_input_type(tvm.Tensor, tvm.Tensor, (tvm.Tensor, NONETYPE),
                             (tvm.Tensor, NONETYPE), dict, (tuple, list, NONETYPE), (tuple, list, NONETYPE),
                             (tuple, list, NONETYPE), (int, NONETYPE), str, (int, NONETYPE), str, str, dict)
def conv2d_fusion_compute(inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
                          groups=1, data_format='NHWC', offset_x=0, kernel_name="conv2d",
                          dsl_flag=True, options=None):
    fusion_util.check_fusion_input([inputs])
    fusion_util.check_fusion_input([weights])

    # set fusion build config
    build_cfg = {
        'constant_realize_extent_in_infer_bound': False,
        'enable_branch_eliminator_else_case': False
    }
    set_fusion_buildcfg("Conv2D", build_cfg)
    op_type = "Conv2D"
    if options is not None:
        options["op_option_dict"] = {}

    return _cube_forward_op_compute(
        op_type, inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
        groups, data_format, offset_x, kernel_name, dsl_flag, options)


def _collect_org_tensors(ori_paras):
    """
    get valid tensors
    """
    ori_tensors = {}
    for key, value in ori_paras.items():
        valid_tensor = isinstance(value, dict) \
                       and isinstance(value.get("ori_shape"), (list, tuple)) \
                       and len(value.get("ori_shape")) > 0
        if valid_tensor:
            ori_tensors[key] = value
    return ori_tensors


def _cube_forward_op_compute(op_type, inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
                    groups=1, data_format='NHWC', offset_x=0, kernel_name="conv2d",
                    dsl_flag=True, options=None):
    """
    conv2d compute

    Notice
    ------
    only used by framework combine with IR

    Parameters
    ----------
    inputs: dict with keys(shape and dtype)
        input 4d feature map tensor
    weights: dict with keys(shape and dtype)
        input 4d weight tensor
    bias: dict with keys(shape and dtype) or None
        input bias tensor
    offset_w: keys(shape and dtype) or None
        input offset_w tensor
    outputs: dict with keys(shape and dtype)
        output tensor, dtype must be assigned
    strides: tuple/list of 4 integers
        stride on H/W, format sensitive
    pads: tuple/list of 4 integers
        [pad_top, pad_bottom, pad_left, pad_right]
        when string, it supports "SAME", "VALID"
    dilations: tuple/list of 4 integers
        dilation on H/W, format sensitive
    groups: int
        param for group covolution
    data_format: string
        input data format
    offset_x: int
        offset for fmap

    Returns
    -------
    tvm compute
    """
    if options is None:
        options = dict()
    default_para = set_default_para()
    if not outputs.get("ori_shape"):
        outputs["ori_shape"] = default_para["ori_shape"]
    ori_paras = {
        "op_type" : op_type, "inputs": inputs, "weights": weights, "bias": bias, "offset_w": offset_w,
        "outputs": outputs, "strides": strides, "pads": pads, "dilations": dilations,
        "groups": groups, "data_format": data_format, "offset_x": offset_x,
        "kernel_name": kernel_name, "optim_dict": default_para.get("optim_dict")
    }
    impl_mode = util_conv2d.get_op_precision_mode(op_type)
    conv_para = Conv2dParaProcess(ori_paras)
    paras = conv_para.config_paras(options)

    pad_t, pad_b, pad_l, pad_r = conv_para.pads
    op_res = conv(paras.get("input_tensor"), paras.get("weight_tensor"),
                  {"bias_tensor": paras.get("bias_tensor"),
                   "offset_w_tensor": offset_w,
                   "pad_h": [pad_t, pad_b], "pad_w": [pad_l, pad_r],
                   "stride_h": conv_para.strides[H_DIM], "stride_w": conv_para.strides[W_DIM],
                   "dilate_h": conv_para.dilations[H_DIM], "dilate_w": conv_para.dilations[W_DIM],
                   "filter_h": paras.get("w_shape")[H_DIM],
                   "filter_w": paras.get("w_shape")[W_DIM],
                   "offset_x": offset_x,
                   "res_dtype": outputs.get("dtype", "float16"),
                   "fusion_para": default_para.get("fusion_para"),
                   "kernel_name": kernel_name,
                   "impl_mode": impl_mode,
                   "group": conv_para.groups,
                   "enlarge": paras.get("group_para").get("enlarge"),
                   "c1_opt": paras.get("group_para").get("c1_opt"),
                   "cout1_opt": paras.get("group_para").get("cout1_opt"),
                   "group_opt": paras.get("group_para").get("group_opt"),
                   "a_shape": paras.get("in_shape_nc1hwc0"),
                   "weight_fracz_shape": paras.get("w_shape_frac_z"),
                   "weight_ori_shape_nchw": paras.get("w_shape"),
                   "padding_mode": paras.get("padding_mode"),
                   "pooling_mode": paras.get("pooling_mode"),
                   "correct_range_flag": paras.get("correct_range_flag", False),
                   "new_in_range": paras.get("new_in_range"),
                   "ori_tensors": _collect_org_tensors(ori_paras),
                   "cache_tiling_flag": paras.get("cache_tiling_flag"),
                   "ori_shape_attr": paras.get("ori_shape_attr")},
                  optim_dict=paras.get("optim_dict"),
                  dsl_flag=dsl_flag)

    if conv_para.is_tensor:
        return op_res
    if conv_para.bias is not None:
        return {"op_placeholder": [paras.get("input_tensor"), paras.get("weight_tensor"), paras.get("bias_tensor")],
                "op_res": [op_res]}
    return {"op_placeholder": [paras.get("input_tensor"), paras.get("weight_tensor")], "op_res": [op_res]}


def cube_forward_op(op_type, inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
           groups, data_format, offset_x, kernel_name, op_option_dict):
    # format classifier input
    input_list = [inputs, weights, bias, offset_w]
    dsl_flag = False
    attr_list = [strides, pads, dilations, groups, data_format, offset_x, kernel_name, dsl_flag]
    output_dict = outputs
    option_dict = set_default_para().get("optim_dict")
    option_dict["op_option_dict"] = op_option_dict
    extra_parameters = {}

    check_support_dtype(inputs, weights, bias, outputs)

    ins = (input_list, attr_list, option_dict, output_dict)
    # run classifier to generate inputs to for each compute branch
    classified_input_list = cube_forward_op_classify(op_type, ins, extra_parameters)

    # run compute and schedule for each set of input
    sch_list = []
    tensor_list = []
    for compute_input in classified_input_list:
        input_list, attr_list, option_dict, output_dict = compute_input
        with tbe_base.compute():
            res = _cube_forward_op_compute(op_type, *input_list, output_dict, *attr_list, option_dict)

        with tvm.target.cce():
            sch_list.append(auto_schedule(res.get("op_res")))

        tensor_list.append(res.get("op_placeholder") + res.get("op_res"))

    # build all kernels
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list,
        "build_args": {"constant_realize_extent_in_infer_bound": False,
                       "dummy_placeholder": True,
                       "enable_branch_eliminator_else_case": False,
                       "enable_db_fold": enable_db_fold()}
    }
    build(sch_list, config)


@register_operator("Conv2D")
@para_check.check_input_type(dict, dict, (dict, NONETYPE), (dict, NONETYPE), dict,
                             (tuple, list), (tuple, list), (tuple, list),
                             (int, NONETYPE), str, (int, NONETYPE), str, str)
def conv2d(inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
           groups=1, data_format='NHWC', offset_x=0, kernel_name="conv2d"):
    """
    algorithm: conv2d

    Notice
    ------
    only used by framework combine with IR

    Parameters
    ----------
    inputs: dict with keys(shape and dtype and range)
        input 4d feature map tensor
    weights: dict with keys(shape and dtype)
        input 4d weight tensor
    bias: dict with keys(shape and dtype) or None
        input bias tensor
    offset_w: keys(shape and dtype) or None
        input offset_w tensor
    outputs: dict with keys(shape and dtype)
        output tensor, dtype must be assigned
    strides: tuple/list of 4 integers
        stride on H/W, format sensitive
    pads: tuple/list of 4 integers
        [pad_top, pad_bottom, pad_left, pad_right]
    dilations: tuple/list of 4 integers
        dilation on H/W, format sensitive
    groups: int
        param for group covolution
    data_format: string
        input data format
    offset_x: int
        offset of fmap
    kernel_name: str
        kernel name, default value is "conv2d"

    Returns
    -------
    None
    """
    op_type = "Conv2D"
    op_option_dict = {}
    cube_forward_op(op_type, inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
           groups, data_format, offset_x, kernel_name, op_option_dict)


def get_op_support_info(inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
                        groups=1, data_format='NCHW', offset_x=0, kernel_name="conv2d"):
    """
    algorithm: get_op_support_info

    Notice
    ------
    get the conv2d split

    Parameters
    ----------
    inputs: dict with keys(shape and dtype)
        input 4d feature map tensor
    weights: dict with keys(shape and dtype)
        input 4d weight tensor
    outputs: dict with keys(shape and dtype)
        output tensor, dtype must be assigned
    bias: dict with keys(shape and dtype) or None
        input bias tensor
    offset_w: keys(shape and dtype) or None
        input offset_w tensor
    strides: tuple/list of 4 integers
        stride on H/W, format sensitive
    pads: tuple/list of 4 integers
        [pad_top, pad_bottom, pad_left, pad_right]
    dilations: tuple/list of 4 integers
        dilation on H/W, format sensitive
    groups: int
        param for group covolution
    data_format: string
        input data format
    offset_x: int
        offset of fmap
    kernel_name: str
        kernel name, default value is "conv2d"

    Returns
    -------
    None
    """
    bias_idx = 2
    format_x = inputs.get("format")
    slice_info = util_conv2d.get_op_support_info_static_common(bias, bias_idx, format_x)

    # >>> start: process for dynamic shape
    shape_x = inputs.get("ori_shape")
    shape_x = shape_util.scalar2tensor_one(shape_x)
    # shape is [-2], all axes do not support split
    if list(shape_x) == [-2]:
        slice_info.get("_op_slice_info").get("splitMaps").clear()
    else:
        # H/W shape is -1, remove corresponding split info
        format_fm = inputs.get("ori_format")
        overlap_axis = {"H": [2], "W": [3]}
        temp_info = slice_info.get('_op_slice_info').get("splitMaps")
        for name, index in overlap_axis.items():
            if shape_x[format_fm.find(name)] == -1:
                last_maps = filter(lambda splits: splits["inputList"][0]["axis"] != index, temp_info)
                temp_info = list(last_maps)
        slice_info.get("_op_slice_info")["splitMaps"] = temp_info
    # <<< end: process for dynamic shape
    return json.dumps(slice_info)
