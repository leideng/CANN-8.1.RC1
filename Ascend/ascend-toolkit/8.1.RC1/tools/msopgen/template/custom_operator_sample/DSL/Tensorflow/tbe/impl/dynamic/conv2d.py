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
import copy
import warnings
import tbe.dsl as tbe_base
from tbe import tvm
from tbe.dsl import auto_schedule
from tbe.dsl import build
from tbe.dsl.compute.conv_compute import conv
from tbe.common.register import set_fusion_buildcfg
from tbe.common.register import register_op_compute
from tbe.common.register import register_operator
from tbe.common.register import register_param_generalization
from tbe.common.utils import para_check
from tbe.common.utils.errormgr import error_manager_cube as err_man
from tbe.common.utils.errormgr import error_manager_util
from ..util import fusion_util
from ..util import util_conv2d
from ..util.util_cube_dynamic import Conv2dParaProcess
from ..util.platform_adapter import tbe_platform

NONETYPE = type(None)
H_DIM = 2
W_DIM = 3
SHAPE_LEN = 5
ORI_SHAPE_LEN = 4


def correct_fuzz_build_range(inputs, weights, strides, pads, dilations):
    """
    get input w range within L1 size

    Notice
    ------
    the proper range are not smaller then shape value

    Parameters
    ----------
    same to conv2d

    Returns
    -------
    None
    """
    valid = isinstance(inputs.get("range"), (list, tuple)) and len(inputs["range"]) >= 4
    if not valid:
        return
    proper_range = list(map(list, inputs["range"]))
    pos_h = inputs["ori_format"].find("H")
    pos_w = inputs["ori_format"].find("W")
    pos_range_h = pos_h
    pos_range_w = pos_w
    if len(proper_range) > 4:
        pos_range_h = 2
        pos_range_w = 3
    # -1 shape range is set by user, should not change
    invaild = (None in proper_range[pos_range_h]) \
              or (None in proper_range[pos_range_w]) \
              or inputs["ori_shape"][pos_w] == -1
    if invaild:
        return
    # >>> start: get the max proper w right range, at least same to shape value
    type_byte = int(re.findall(r'\d+', inputs["dtype"])[0]) // 8
    pad_top, pad_bottom, pad_left, pad_right = pads
    filter_h_dilation = (weights["ori_shape"][weights["ori_format"].find("H")] - 1) * dilations[pos_h] + 1
    filter_w_dilation = (weights["ori_shape"][weights["ori_format"].find("W")] - 1) * dilations[pos_w] + 1
    l1_size = tbe_platform.get_soc_spec("L1_SIZE")
    proper_w = proper_range[pos_range_w][1]
    for i in list(range(proper_w, inputs["ori_shape"][pos_w] - 1, -1)):
        if -1 in pads:
            w_out = i + strides[pos_w] - 1 // strides[pos_w]
        else:
            w_out = (i + (pad_left + pad_right) - filter_w_dilation) // strides[pos_w] + 1
        limit_h_out = math.floor(tbe_platform.CUBE_MKN[inputs["dtype"]]['mac'][0] / w_out) + 2
        limit_size = ((limit_h_out - 1) * strides[pos_h] + filter_h_dilation) * i
        limit_size = limit_size * tbe_platform.CUBE_MKN[inputs["dtype"]]['mac'][1] * type_byte
        proper_w = i
        if limit_size < l1_size:
            break
    # <<< end: get the max proper w right range, at least same to shape value
    # >>> start: change input range and ori_range if exists
    if proper_w != proper_range[pos_range_w][1]:
        proper_range[pos_range_w][1] = proper_w
        to_print = "conv2d fuzz build range changed from {} to {}".format(inputs["range"], proper_range)
        inputs["range"] = proper_range
        valid = isinstance(inputs.get("ori_range"), (list, tuple)) and len(inputs["ori_range"]) == 4
        if valid:
            ori_range = list(map(list, inputs["ori_range"]))
            ori_range[pos_w][1] = proper_w
            to_print = "{}, ori_range changed from {} to {}".format(to_print, inputs["ori_range"], proper_range)
            inputs["ori_range"] = ori_range
        warnings.warn(to_print)
    # <<< end: change input range and ori_range if exists


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
    support_mode = ["keep_rank"]
    if generalize_config.get("mode") not in support_mode:
        err_man.raise_err_specific_user("conv2d", "invalid generalize mode {}, only support {}".format(
            str(generalize_config.get("mode")), str(support_mode)))
    result = []
    if generalize_config["mode"] == "keep_rank": # fuzz build situation
        # unknow_rank inputs ori_shape is [-2], others' shape length is 4
        unknow_rank = len(inputs["ori_shape"]) == 1 and inputs["ori_shape"][0] == -2
        if unknow_rank:
            err_man.raise_err_specific_user("conv2d", "not support unknow_rank under mode {}".format(
                generalize_config["mode"]))
        correct_fuzz_build_range(inputs, weights, strides, pads, dilations)
        have_range = {"inputs": inputs, "outputs": outputs}
        support_format = ["NCHW", "NHWC"]
        for name, tensor in have_range.items():
            # modify tesnors have range
            if tensor.get("ori_format") not in support_format:
                err_man.raise_err_specific_user("conv2d", "invalid {} ori_format {}, only support {}".format(
                    name, str(tensor.get("ori_format")), str(support_format)))
            # only change shape NHW dim to -1, range is already set at infershape
            valid = isinstance(tensor.get("ori_shape"), (list, tuple)) and len(tensor["ori_shape"]) == ORI_SHAPE_LEN
            if not valid:
                err_man.raise_err_specific_user("conv2d", "invalid {} ori_shape {}, only support {}d".format(
                        name, str(tensor.get("ori_shape")), str(ORI_SHAPE_LEN)))
            valid = isinstance(tensor.get("shape"), (list, tuple)) and len(tensor["shape"]) == SHAPE_LEN
            if not valid:
                err_man.raise_err_specific_user("conv2d", "invalid {} ori_shape {}, only support {}d".format(
                        name, str(tensor.get("shape")), str(SHAPE_LEN)))
            tensor["ori_shape"] = [-1, tensor["ori_shape"][1], -1, -1] \
               if tensor.get("ori_format") == "NCHW" else [-1, -1, -1, tensor["ori_shape"][3]]
            tensor["shape"] = [-1, tensor["shape"][1], -1, -1, tensor["shape"][4]]
        result.append([inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
                       groups, data_format, offset_x, kernel_name])
    return result



def set_default_para():
    """
    set default parameter value
    """
    default_para = {}
    default_para["res_dtype"] = "float16"
    default_para["optim_dict"] = {"c0_optim_flg": False}
    default_para["fusion_para"] = {"input_memory_type": 0, "output_memory_type": 0,
                                   "valid_shape": (), "slice_offset": (),
                                   "l1_fusion_type": -1}
    default_para["ori_shape"] = [0, 0, 0, 0]
    return default_para


@register_op_compute("Conv2D", op_mode="dynamic", support_fusion=True)
@para_check.check_input_type(tvm.Tensor, tvm.Tensor, (tvm.Tensor, NONETYPE),
                             (tvm.Tensor, NONETYPE), dict, (tuple, list), (tuple, list), (tuple, list),
                             int, str, int, str, str)
def conv2d_fusion_compute(inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
                          groups=1, data_format='NHWC', offset_x=0, kernel_name="conv2d",
                          dsl_flag=True):
    fusion_util.check_fusion_input([inputs])
    fusion_util.check_fusion_input([weights])

    # set fusion build config
    build_cfg = {
        'constant_realize_extent_in_infer_bound': False
    }
    set_fusion_buildcfg("Conv2D", build_cfg)
    return _conv2d_compute(
        inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
        groups, data_format, offset_x, kernel_name, dsl_flag)

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


def _conv2d_compute(inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
                    groups=1, data_format='NHWC', offset_x=0, kernel_name="conv2d",
                    dsl_flag=True):

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

    default_para = set_default_para()
    if not outputs.get("ori_shape"):
        outputs["ori_shape"] = default_para["ori_shape"]
    ori_paras = {
        "inputs": inputs, "weights": weights, "bias": bias, "offset_w": offset_w,
        "outputs": outputs, "strides": strides, "pads": pads, "dilations": dilations,
        "groups": groups, "data_format": data_format, "offset_x": offset_x,
        "kernel_name": kernel_name, "optim_dict": default_para.get("optim_dict"),
    }

    conv_para = Conv2dParaProcess(ori_paras)
    paras = conv_para.config_paras()

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
                   "res_dtype": default_para.get("res_dtype"),
                   "fusion_para": default_para.get("fusion_para"),
                   "kernel_name": kernel_name,
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
                   "ori_tensors": _collect_org_tensors(ori_paras)},
                  optim_dict=default_para.get("optim_dict"),
                  dsl_flag=dsl_flag)

    if conv_para.is_tensor == True:
        return op_res
    if conv_para.bias is not None:
        return {"op_placeholder": [paras.get("input_tensor"), paras.get("weight_tensor"), paras.get("bias_tensor")],
                "op_res": [op_res]}
    return {"op_placeholder": [paras.get("input_tensor"), paras.get("weight_tensor")], "op_res": [op_res]}


@register_operator("Conv2D")
@para_check.check_input_type(dict, dict, (dict, NONETYPE), (dict, NONETYPE), dict,
                             (tuple, list), (tuple, list), (tuple, list),
                             int, str, int, str, str)
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

    with tbe_base.compute():
        res = _conv2d_compute(
            inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
            groups, data_format, offset_x, kernel_name, dsl_flag=False)

    with tvm.target.cce():
        sch = auto_schedule(res.get("op_res"))

    tensor_list = res.get("op_placeholder") + res.get("op_res")
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list,
        "build_args": {"constant_realize_extent_in_infer_bound": False, "dummy_placeholder": True}
    }

    build(sch, config)
