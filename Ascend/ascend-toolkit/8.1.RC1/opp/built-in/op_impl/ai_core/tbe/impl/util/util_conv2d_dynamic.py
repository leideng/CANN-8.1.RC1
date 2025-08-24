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
public function for cube dynamic
"""

from __future__ import absolute_import
import warnings
import math

from impl.util.platform_adapter import error_manager_cube as err_man
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.util_cube_dynamic import CubeParaProcess
from tbe.common.context import get_context
from tbe.common.utils import log
from tbe.common.utils.op_util import op_util_conv2d
from tbe.dsl.base.operation import get_te_var
from tbe.common.utils.op_util.op_util_conv2d import is_support_fixpipe
from tbe.common.utils.op_util.op_util_conv2d import check_nosupport_binary_op
from tbe.common.utils.op_util.op_util_conv2d import check_support_db_fold
from tbe.common.utils.op_util.op_util_conv2d import BINARY_SRC_SHAPE_ATTR, C0_OPTIM_FLG, ConstValue
from tbe.common.utils.op_util.op_util_conv2d import ceil_div

# the bytes length of several dtype
BIT_RATIO_DICT = {"int32": 4, "float32": 4, "float16": 2,
                  "uint8": 1, "int8": 1, "uint4": 0.5, "int4": 0.5}
N_DIM = 0
C_DIM = 1
H_DIM = 2
W_DIM = 3
DYNAMIC_FLAG = -1
SHAPE_LEN = 5
ALL_SHAPE_DYNAMIC = -2
ALL_SHAPE_DYNAMIC_LEN = 1
ORI_SHAPE_LEN = 4
LOAD2D_H_VALUE = 1
LOAD2D_W_VALUE = 1
DYNAMIC_FMAP_W_MIN = 1
DYNAMIC_FMAP_W_MAX = 4096

FMAP_POS = 0
WEIGHT_POS = 1
BIAS_POS = 2
OUTPUT_POS = 3
CONV2D_DYN_SUPPORTED_DTYPE_LIST = [
    # fmap, weight, bias, output
    ["float16", "float16", "float16", "float16"],
    ["float32", "float32", "float32", "float32"],
    ["bfloat16", "bfloat16", "float32", "bfloat16"],
]


def get_format_attr(w_shape, w_format):
    """
    get format attr
    """
    if w_format == "NCHW":
        kh = w_shape[2]
        kw = w_shape[3]
    elif w_format == "NHWC":
        kh = w_shape[1]
        kw = w_shape[2]
    elif w_format == "HWCN":
        kh = w_shape[0]
        kw = w_shape[1]
    else:
        err_man.raise_err_specific_user("Conv2D", "input filter format only support NCHW, NHWC or HWCN.")

    return kh, kw


# 'pylint: disable=too-many-arguments, too-many-locals
def modify_input_range(inputs, in_range_nchw, data_type, idx_h, idx_w, attr_params):
    '''
    check for not bigger than L1
    '''
    strides, hk_dilation, wk_dilation, pads = attr_params
    fmap_w_min = in_range_nchw[idx_w][0]
    fmap_w_max = in_range_nchw[idx_w][1]
    m_bit_ratio = {"float32": 2, "float16": 2, "int8": 1}
    c0 = tbe_platform.CUBE_MKN[data_type]["mac"][1]
    new_in_range_nchw = list(in_range_nchw)
    w_in = inputs.get("ori_shape")[idx_w]

    stride_h = strides[idx_h]
    stride_w = strides[idx_w]
    l1size_limit_upper = tbe_platform.get_soc_spec("L1_SIZE")
    w_left = fmap_w_min
    w_right = fmap_w_max
    current_w = fmap_w_max
    while (w_right - w_left) != 1:
        if -1 in pads:
            w_out = (current_w + stride_w - 1) // stride_w
        else:
            w_out = math.floor((current_w - wk_dilation + pads[2] + pads[3]) / stride_w) + 1
        ho_num = math.floor(tbe_platform.CUBE_MKN[data_type]["mac"][0] / w_out) + 2
        l1_m = ((ho_num - 1) * stride_h + hk_dilation) * current_w
        max_feature_map_l1 = c0 * l1_m * m_bit_ratio[data_type]
        if max_feature_map_l1 > l1size_limit_upper:
            w_right = current_w
        else:
            w_left = current_w
        current_w = w_left + (w_right - w_left)//2

        if w_left == fmap_w_max:
            break
    cor_w_range = (fmap_w_min, w_left)
    if w_in < cor_w_range[0] or w_in > cor_w_range[1]:
        new_in_range_nchw[idx_w] = [w_in, w_in]
    else:
        new_in_range_nchw[idx_w] = cor_w_range
    to_print = "conv2d fmap ori_range changed from {} to {}.".format(in_range_nchw, new_in_range_nchw)
    warnings.warn(to_print)

    return new_in_range_nchw


# 'pylint: disable=too-many-arguments, too-many-locals
def check_l1_size(op_type, inputs, kh_dilate, kw_dilate, strides, pads):
    """
    check exceed l1 buf
    """
    l1_size = tbe_platform.get_soc_spec("L1_SIZE")
    # default type fp16
    type_byte = BIT_RATIO_DICT.get(inputs["dtype"], 2)
    idx_h = inputs.get("ori_format").find("H")
    idx_w = inputs.get("ori_format").find("W")
    w_in = inputs.get("ori_shape")[idx_w]
    pad_top, pad_bottom, pad_left, pad_right = pads
    if DYNAMIC_FLAG in pads:
        w_out = w_in + strides[idx_w] - 1 // strides[idx_w]
    else:
        w_out = (w_in + (pad_left + pad_right) - kw_dilate) // strides[idx_w] + 1
    limit_h_out = math.floor(tbe_platform.CUBE_MKN[inputs["dtype"]]['mac'][0] / w_out) + 2
    hw_size = ((limit_h_out - 1) * strides[idx_h] + kh_dilate) * w_in
    limit_size = hw_size * tbe_platform.CUBE_MKN[inputs["dtype"]]['mac'][1] * type_byte
    if limit_size > l1_size:
        err_man.raise_err_specific_user(op_type, "input range is too large, the mininum tiling may exceed L1buffer.")


def create_fuzz_range(op_type, dim_value, grade_item):
    """
    gen fuzz range
    """
    # dim_value must less than grade max_value
    if dim_value > grade_item[-1]:
        # 'pylint: disable=too-many-function-args
        err_man.raise_err_specific_user(op_type, "input value {} is out the range of {}", dim_value, grade_item[-1])
        err_man.raise_err_attr_range_invalid(op_type, "[1, 4096]", "input shape", str(dim_value))

    for g_value in grade_item:
        if dim_value > g_value:
            low = g_value + 1
        if dim_value <= g_value:
            high = g_value
            break
    return [low, high]


# 'pylint: disable=too-many-arguments
def check_graph_mode(tensor):
    """
    check graph mode or single mode in fuzz compile
    """
    if (DYNAMIC_FLAG in tensor.get("ori_shape") and "ori_range" in tensor.keys()):
        return True
    return False


def check_support_dtype(inputs, weights, bias, outputs):
    if not is_support_fixpipe():
        return

    data_dtype = "dtype"

    inputs_data_type = inputs.get(data_dtype)
    weights_data_type = weights.get(data_dtype)
    bias_data_type = bias.get(data_dtype) if bias is not None else None
    outputs_data_type = outputs.get(data_dtype)

    for supported_dtype in CONV2D_DYN_SUPPORTED_DTYPE_LIST:
        is_valid_dtype = (inputs_data_type == supported_dtype[FMAP_POS]) and \
                         (weights_data_type == supported_dtype[WEIGHT_POS]) and \
                         (outputs_data_type == supported_dtype[OUTPUT_POS]) and \
                         (bias_data_type is None or bias_data_type == supported_dtype[BIAS_POS])
        if is_valid_dtype:
            return

    err_man.raise_err_message_cube("Not support the given input and output data type."
                                   "inputs: {}, weights: {}, bias: {}, outputs:{}."
                                   .format(inputs_data_type, weights_data_type, bias_data_type, outputs_data_type))


def enable_db_fold():
    if check_support_db_fold():
        return True

    return False


def check_input_range(input_range, idx_h, idx_w, kh_dilate, kw_dilate, pads):
    """
    graph mode fuzz, check range[low] for output >= 1
    """
    padt, padd, padl, padr = pads
    if DYNAMIC_FLAG not in pads:
        low_h = kh_dilate - padt - padd
        if input_range[idx_h][0] < low_h:
            log.debug("graph mode: the h_range[0] must be greater than or equal to kernel_h")
            return "lower_limit"

        low_w = kw_dilate - padl - padr
        if input_range[idx_w][0] < low_w:
            log.debug("graph mode: the w_range[0] must be greater than or equal to kernel_w")
            return "lower_limit"
    return ""


# 'pylint: disable=unused-variable
def check_range_l1_size(inputs, kh_dilate, kw_dilate, strides, pads):
    """
    graph mode fuzz, check range[high] exceed l1 buf
    """
    l1_size = tbe_platform.get_soc_spec("L1_SIZE")
    # default type fp16
    type_byte = BIT_RATIO_DICT.get(inputs["dtype"], 2)
    idx_h = inputs.get("ori_format").find("H")
    idx_w = inputs.get("ori_format").find("W")
    w_in = inputs.get("ori_range")[idx_w][1]
    if (w_in is None):
        w_in = 4096
    pad_top, pad_bottom, pad_left, pad_right = pads
    if DYNAMIC_FLAG in pads:
        w_out = (w_in + strides[idx_w] - 1) // strides[idx_w]
    else:
        w_out = (w_in + (pad_left + pad_right) - kw_dilate) // strides[idx_w] + 1
    limit_h_out = math.floor(tbe_platform.CUBE_MKN[inputs["dtype"]]['mac'][0] / w_out) + 2
    hw_size = ((limit_h_out - 1) * strides[idx_h] + kh_dilate) * w_in
    limit_size = hw_size * tbe_platform.CUBE_MKN[inputs["dtype"]]['mac'][1] * type_byte
    if limit_size > l1_size:
        log.debug("input range is too large, the minimum tiling may exceed L1buffer")
        return "upper_limit"
    return ""


def check_range_value(op_type, input_range, idx_h, idx_w):
    """
    check if input range is < 0
    """
    input_range_h_min_invalid = input_range[idx_h][0] is not None and input_range[idx_h][0] <= 0
    input_range_h_max_invalid = input_range[idx_h][1] is not None and input_range[idx_h][1] <= 0
    input_range_w_min_invalid = input_range[idx_w][0] is not None and input_range[idx_w][0] <= 0
    input_range_w_max_invalid = input_range[idx_w][1] is not None and input_range[idx_w][1] <= 0

    if (input_range_h_min_invalid or input_range_h_max_invalid or 
        input_range_w_min_invalid or input_range_w_max_invalid):

        err_man.raise_err_specific_user(op_type, "input invalid range <= 0.")


def check_conv2d_range(inputs, weights, strides, pads, dilations):
    """
    graph mode fuzz, check input range
    """
    op_type = "conv2d"
    input_range = inputs.get("ori_range")
    x_format = inputs.get("ori_format")
    w_shape = weights.get("ori_shape")
    w_format = weights.get("ori_format")

    if x_format == "NCHW":
        idx_h = 2
        idx_w = 3
        dilh = dilations[2]
        dilw = dilations[3]
    elif x_format == "NHWC":
        idx_h = 1
        idx_w = 2
        dilh = dilations[1]
        dilw = dilations[2]
    else:
        err_man.raise_err_specific_user(op_type, "input fmap format only support NCHW or NHWC.")

    check_range_value(op_type, input_range, idx_h, idx_w)

    kh, kw = get_format_attr(w_shape, w_format)
    kh_dilate = dilh * (kh - 1) + 1
    kw_dilate = dilw * (kw - 1) + 1

    low_check = check_input_range(input_range, idx_h, idx_w, kh_dilate, kw_dilate, pads)
    up_check = check_range_l1_size(inputs, kh_dilate, kw_dilate, strides, pads)
    if not up_check and not low_check:
        return []

    type_info = []
    if up_check:
        type_info.append(up_check)
    if low_check:
        type_info.append(low_check)

    check_result = [{"result": "UNSUPPORTED", "reason": {"param_index": [0], "type": type_info}}]
    return check_result


# 'pylint: disable=too-many-arguments
def correct_input_range(op_type, input_range, x_shape, idx_h, idx_w, kh_dilate, kw_dilate, pads):
    """
    correct range[low] for  output >= 1
    """
    padt, padd, padl, padr = pads
    if DYNAMIC_FLAG in pads:  # padding=same mode
        low_h = input_range[idx_h][0]
        low_w = input_range[idx_w][0]
    else:
        low_h = kh_dilate - padt - padd
        if x_shape[idx_h] != DYNAMIC_FLAG and x_shape[idx_h] < low_h:
            err_man.raise_err_specific_user(op_type, "the output h must be greater than or equal to 1")
        if x_shape[idx_h] == DYNAMIC_FLAG and input_range[idx_h][1] < low_h:
            err_man.raise_err_specific_user(op_type, "the h_range[1] must be greater than or equal to kerne_h")
        low_w = kw_dilate - padl - padr
        if x_shape[idx_w] != DYNAMIC_FLAG and x_shape[idx_w] < low_w:
            err_man.raise_err_specific_user(op_type, "the output w must be greater than or equal to 1")
        if x_shape[idx_w] == DYNAMIC_FLAG and input_range[idx_w][1] < low_w:
            err_man.raise_err_specific_user(op_type, "the w_range[1] must be greater than or equal to kerne_w")

    input_range[idx_h][0] = max(input_range[idx_h][0], low_h)
    input_range[idx_w][0] = max(input_range[idx_w][0], low_w)


def align(x_1, x_2):
    """
    align up for inputs
    """

    return ceil_div(x_1, x_2) * x_2


def is_shape_all_dynamic(inputs):
    ori_shape = inputs.get("ori_shape")
    if not isinstance(ori_shape, list) and not isinstance(ori_shape, tuple):
        return False
    if len(ori_shape) is not ALL_SHAPE_DYNAMIC_LEN:
        return False
    return ALL_SHAPE_DYNAMIC in ori_shape


class Conv2dParaProcess(CubeParaProcess):
    """
    class of param check and preprocess for dynamic conv2d and other cube ops
    """

    def __init__(self, paras):
        def conver_tensor2dict(tensor, need_range):
            if tensor is None:
                return None
            dict = {}
            dict["ori_shape"] = []
            for i in tensor.op.attrs['ori_shape']:
                dict["ori_shape"].append(i.value)
            dict["dtype"] = tensor.dtype
            dict["ori_format"] = tensor.op.attrs['ori_format']
            dict["format"] = tensor.op.attrs['format']

            if need_range is True:
                dict["range"] = []
                for one_range in tensor.op.attrs['range']:
                    range_list = []
                    for value in one_range:
                        range_list.append(value.value)
                    dict["range"].append(range_list)
                if operation.get_te_var("batch_n"):
                    dict["range"][N_DIM] = list(operation.get_te_var("batch_n").get_bound())
                if operation.get_te_var("fmap_h"):
                    dict["range"][H_DIM] = list(operation.get_te_var("fmap_h").get_bound())
                if operation.get_te_var("fmap_w"):
                    dict["range"][W_DIM] = list(operation.get_te_var("fmap_w").get_bound())

            return dict

        super().__init__(paras)
        self.op_type = paras.get("op_type", "conv2d").lower()
        if isinstance(paras.get("inputs"), dict):
            self.is_tensor = False
            self.inputs = paras.get("inputs")
            self.weights = paras.get("weights")
            self.bias = paras.get("bias")
            self.dtype = paras.get("inputs").get("dtype")
        else:
            self.is_tensor = True
            fmap_need_range = True
            if paras.get("binary_static_flag"):
                fmap_need_range = False
            self.input_tensor = paras.get("inputs")
            self.weights_tensor = paras.get("weights")
            self.bias_tensor = paras.get("bias")
            self.inputs = conver_tensor2dict(self.input_tensor, fmap_need_range)
            self.weights = conver_tensor2dict(self.weights_tensor, False)
            self.bias = conver_tensor2dict(self.bias_tensor, False)
            self.dtype = self.input_tensor.dtype

        self.outputs = paras.get("outputs")
        self.data_format = paras.get("data_format")
        self.cache_tiling_flag = False
        self.binary_static_flag = paras.get("binary_static_flag")
        self.ori_shape_attr = {}
        self.valid_paras = {
            "n_min": 0,
            "hw_min": 0,
            "hw_max": 4096,
            "valid_format": {"weights": ("NCHW", "NHWC", "HWCN"),
                             "input": ("NCHW", "NHWC"),
                             "output": ("NCHW", "NHWC")},
            "valid_dtype": ("float16", "int8", "int32", "float32")
        }
        self.dim_valid_dict = {
            N_DIM: (self.valid_paras.get("n_min"), None),
            H_DIM: (self.valid_paras.get("hw_min"), self.valid_paras.get("hw_max")),
            W_DIM: (self.valid_paras.get("hw_min"), self.valid_paras.get("hw_max"))
        }

    def check_dynamic_mode(self):
        """
        demo
        """
        # not support fixpipe/avgpool/depthwise/not float16
        binary_support_cond = not check_nosupport_binary_op()
        # input has range or ori_range
        inputs_has_range_cond = self.inputs.get("range") or self.inputs.get("ori_range")
        # input has -1 in shape or ori_shape
        dynamic_shape_cond = \
            self.inputs.get("shape") and DYNAMIC_FLAG in self.inputs.get("shape") and inputs_has_range_cond
        dynamic_orishape_cond = \
            self.inputs.get("ori_shape") and DYNAMIC_FLAG in self.inputs.get("ori_shape") and inputs_has_range_cond
        # binary combined condition list
        condition_list = [
            self.strides == [-1, -1, -1, -1],
            self.strides == (-1, -1, -1, -1),
            is_shape_all_dynamic(self.inputs),
            (dynamic_shape_cond or dynamic_orishape_cond) and binary_support_cond,
            self.binary_static_flag  # dynamic and static integrate flag
        ]
        for cond in condition_list:
            if cond:
                self.cache_tiling_flag = True

    def check_support_valid(self, in_shape, w_shape):
        """
        check whether dynamic shape is supported for cube operation
        """

        super().check_support_valid(in_shape, w_shape)
        if in_shape[C_DIM] == DYNAMIC_FLAG:
            err_man.raise_err_specific_user(
                self.op_type, "dynamic c dimension is not supported yet.")
        if self.paras.get("offset_w"):
            err_man.raise_err_specific_user(
                self.op_type, "offset_w is not supported in dynamic shape yet.")

    # 'pylint: disable=too-many-arguments
    def correct_in_range(self, in_range_nchw, w_shape_nchw):
        """
        correct in range
        """
        # correct in_range when w_range=[1, None] or fuzz_build mode
        fuzz_build = get_context().get_build_type() == "fuzzily_build"
        fmap_w_min = in_range_nchw[W_DIM][0] if fuzz_build else DYNAMIC_FMAP_W_MIN
        fmap_w_max = in_range_nchw[W_DIM][1] if fuzz_build else DYNAMIC_FMAP_W_MAX
        m_bit_ratio = {"float16": 2, "int8": 1}
        c0 = tbe_platform.CUBE_MKN[self.weights["dtype"]]["mac"][1]
        fmap_w_upper = in_range_nchw[W_DIM][1]
        new_in_range_nchw = list(in_range_nchw)

        if not fmap_w_upper or fuzz_build:
            stride_h = self.strides[H_DIM]
            stride_w = self.strides[W_DIM]
            hk_dilation = (w_shape_nchw[H_DIM] - 1) * self.dilations[H_DIM] + 1
            wk_dilation = (w_shape_nchw[W_DIM] - 1) * self.dilations[W_DIM] + 1
            l1size_limit_upper = tbe_platform.get_soc_spec("L1_SIZE")
            w_left = fmap_w_min
            w_right = fmap_w_max
            current_w = fmap_w_max
            while (w_right - w_left) != 1:
                if -1 in self.pads:
                    w_out = (current_w + stride_w - 1) // stride_w
                else:
                    w_out = math.floor((current_w - wk_dilation + self.pads[2] + self.pads[3]) / stride_w) + 1
                ho_num = math.floor(tbe_platform.CUBE_MKN[self.weights["dtype"]]["mac"][0] / w_out) + 2
                l1_m = ((ho_num - 1) * stride_h + hk_dilation) * current_w
                max_feature_map_l1 = c0 * l1_m * m_bit_ratio[self.weights["dtype"]]
                if max_feature_map_l1 > l1size_limit_upper:
                    w_right = current_w
                else:
                    w_left = current_w
                current_w = w_left + (w_right - w_left) // 2

                if w_left == fmap_w_max:
                    break
            cor_w_range = (fmap_w_min, w_left)
            new_in_range_nchw[W_DIM] = cor_w_range
            to_print = "{} fmap ori_range changed from {} to {}.".format(self.op_type, in_range_nchw, new_in_range_nchw)
            warnings.warn(to_print)

        return new_in_range_nchw

    def check_only_cdim_dynamic(self, inputs):
        """
        check cdim_dynamic
        """
        if inputs.get("shape"):
            in_shape_5hd = list(inputs.get("shape"))
        if inputs.get("ori_range"):
            ori_range = list(inputs.get("ori_range"))
        if inputs.get("range"):
            in_range = list(inputs.get("range"))
        if inputs.get("ori_format"):
            ori_format = inputs.get("ori_format")
            n_index = ori_format.find("N")
            c_index = ori_format.find("C")
            h_index = ori_format.find("H")
            w_index = ori_format.find("W")
            if inputs.get("ori_shape"):
                ori_shape = list(inputs.get("ori_shape"))
                if ori_shape == [-2]:
                    return
                if DYNAMIC_FLAG == ori_shape[c_index] and \
                    DYNAMIC_FLAG not in (ori_shape[n_index], ori_shape[h_index], ori_shape[w_index]):
                    in_range[0] = (ori_shape[n_index], ori_shape[n_index])
                    if inputs.get("shape"):
                        in_shape_5hd[n_index] = DYNAMIC_FLAG
                        self.inputs["shape"] = in_shape_5hd
                    if inputs.get("ori_range"):
                        ori_range[n_index] = (ori_range[n_index], ori_range[n_index])
                        self.inputs["ori_range"] = ori_range
                    if inputs.get("ori_range"):
                        ori_shape[n_index] = DYNAMIC_FLAG
                        self.inputs["ori_shape"] = ori_shape
                    if inputs.get("range"):
                        self.inputs["range"] = in_range

    # 'pylint: disable= too-many-locals
    def check_paras(self):
        """
        check original paras
        """
        self.check_only_cdim_dynamic(self.inputs)
        self.check_input_dict(self.inputs, "inputs", True)
        self.check_input_dict(self.weights, "weights", False)
        para_check.check_dtype_rule(self.dtype, self.valid_paras.get("valid_dtype"))
        para_check.check_dtype_rule(self.weights.get("dtype"), self.valid_paras.get("valid_dtype"))
        para_check.check_dtype_rule(self.paras.get("outputs").get("dtype"), self.valid_paras.get("valid_dtype"))
        if self.dtype != self.weights.get("dtype"):
            err_man.raise_err_specific_user(self.op_type, "in_dtype != w_dtype")
        self.check_format(self.data_format, "input")
        self.check_format(self.weights.get("ori_format"), "weights")
        if self.inputs.get("ori_format") != self.data_format:
            err_man.raise_err_specific_user("conv2d", "in_format != data_format")
        para_check.check_kernel_name(self.paras.get("kernel_name"))

        in_shape = list(self.inputs.get("ori_shape"))
        in_range = self.inputs.get("range")
        w_shape = list(self.weights.get("ori_shape"))
        outputs_shape = list(self.outputs.get("ori_shape"))
        self.check_para_dim(w_shape, "weights")
        self.check_para_dim(self.strides, "strides")
        self.check_para_dim(self.dilations, "dilations")
        self.check_para_dim(self.pads, "pads")
        w_shape_nchw = self.get_input_nchw(w_shape, self.weights.get("ori_format"))
        out_shape_nchw = self.get_input_nchw(outputs_shape, self.outputs.get("ori_format"))

        if self.check_unknown_scene(in_shape, out_shape_nchw, w_shape_nchw[N_DIM]):
            in_shape_nchw = [DYNAMIC_FLAG, w_shape_nchw[C_DIM], DYNAMIC_FLAG, DYNAMIC_FLAG]
            in_range_nchw = [(1, None), (w_shape_nchw[C_DIM], w_shape_nchw[C_DIM]), (1, None), (1, None)]
        else:
            self.check_para_dim(in_shape, "in_shape")
            in_shape_nchw, in_range_nchw = self.get_input_nchw(in_shape, self.data_format, in_range)
            if in_shape_nchw[1] == -1:
                in_shape_nchw[1] = w_shape_nchw[1]*self.groups
            self.check_range_valid(in_shape_nchw, in_range_nchw, "fmap", self.data_format)

        self.check_support_valid(in_shape_nchw, w_shape_nchw)
        self.get_attr_nchw(self.data_format)
        cor_in_range_nchw = self.correct_in_range(in_range_nchw, w_shape_nchw)
        y_range, correct_range_flag, new_in_range_nchw = self.get_output_range(w_shape_nchw, cor_in_range_nchw)
        self.check_range_valid(out_shape_nchw, y_range, "output", self.data_format)

        group_para = self.set_group_para(in_shape_nchw, w_shape_nchw, self.dtype)
        in_shape_nchw, w_shape_nchw, in_shape_nc1hwc0, w_shape_frac_z = self._calc_shape(
            in_shape_nchw, w_shape_nchw, new_in_range_nchw, y_range, group_para)
        self.calc_pads(in_shape_nc1hwc0, w_shape_nchw)

        return {"in_shape_nc1hwc0": in_shape_nc1hwc0, "w_shape_frac_z": w_shape_frac_z,
                "w_shape": w_shape_nchw, "group_para": group_para,
                "correct_range_flag": correct_range_flag,
                "new_in_range": new_in_range_nchw, "ori_shape_attr": self.ori_shape_attr}

    def cache_tiling_paras_process(self, options):
        """
        config paras for cachetiling
        """
        def generate_op_var():
            var_dtype = "int32"
            if self.binary_static_flag and not self.is_tensor:
                var_dtype = "int64"
            for i in range(op_util_conv2d.TilingDataIdx.TILINGDATA_IDX_END):
                if i in op_util_conv2d.TILINGDATA_KEY_MAP.keys():
                    if not operation.get_te_var(op_util_conv2d.TILINGDATA_KEY_MAP.get(i)):
                        operation.var(op_util_conv2d.TILINGDATA_KEY_MAP.get(i), dtype=var_dtype)
                else:
                    log.error("Tiling key not in Tilingkey Map")

        def generate_ori_shape_attr():
            if options and options.get(BINARY_SRC_SHAPE_ATTR):
                for key, value in options.get(BINARY_SRC_SHAPE_ATTR).items():
                    self.ori_shape_attr[key] = value

        def get_weight_fz_k1():
            if options and options.get(C0_OPTIM_FLG, False):
                weight_fz_k1 = ceil_div(group_opt * cin1_opt * k_h * k_w, ConstValue.C04_CONST)
            else:
                weight_fz_k1 = group_opt * cin1_opt * k_h * k_w
            return weight_fz_k1

        generate_op_var()
        generate_ori_shape_attr()
        dilation_h_var = get_te_var(op_util_conv2d.TilingDataKey.DILATION_H).get_tvm_var()
        dilation_w_var = get_te_var(op_util_conv2d.TilingDataKey.DILATION_W).get_tvm_var()
        stride_h_var = get_te_var(op_util_conv2d.TilingDataKey.STRIDE_H).get_tvm_var()
        stride_w_var = get_te_var(op_util_conv2d.TilingDataKey.STRIDE_W).get_tvm_var()
        batch_n = get_te_var(op_util_conv2d.TilingDataKey.BATCH_N).get_tvm_var()
        fmap_h = get_te_var(op_util_conv2d.TilingDataKey.FMAP_H).get_tvm_var()
        fmap_w = get_te_var(op_util_conv2d.TilingDataKey.FMAP_W).get_tvm_var()
        c_in = get_te_var(op_util_conv2d.TilingDataKey.C_IN).get_tvm_var()
        c_out = get_te_var(op_util_conv2d.TilingDataKey.C_OUT).get_tvm_var()
        k_h = get_te_var(op_util_conv2d.TilingDataKey.K_H).get_tvm_var()
        k_w = get_te_var(op_util_conv2d.TilingDataKey.K_W).get_tvm_var()
        pad_top = get_te_var(op_util_conv2d.TilingDataKey.PAD_TOP).get_tvm_var()
        pad_bottom = get_te_var(op_util_conv2d.TilingDataKey.PAD_BOTTOM).get_tvm_var()
        pad_left = get_te_var(op_util_conv2d.TilingDataKey.PAD_LEFT).get_tvm_var()
        pad_right = get_te_var(op_util_conv2d.TilingDataKey.PAD_RIGHT).get_tvm_var()
        block_size_k, block_size_n = tbe_platform.CUBE_MKN[self.dtype]["mac"][1:3]
        group_opt = get_te_var(op_util_conv2d.TilingDataKey.GROUP_OPT).get_tvm_var()
        cin1_opt = get_te_var(op_util_conv2d.TilingDataKey.CIN1_OPT).get_tvm_var()
        cout1_opt = get_te_var(op_util_conv2d.TilingDataKey.COUT1_OPT).get_tvm_var()
        enlarge = get_te_var(op_util_conv2d.TilingDataKey.ENLARGE).get_tvm_var()

        if self.binary_static_flag and self.is_tensor:
            weight_format_str = self.paras.get("weights").op.attrs.get("ori_format")
            format_str = self.paras.get("data_format")
            pos_c = format_str.find('C')
            pos_h = format_str.find('H')
            pos_w = format_str.find('W')
            pos_batch = format_str.find('N')
            pos_cout = weight_format_str.find('N')
            pos_kh = weight_format_str.find('H')
            pos_kw = weight_format_str.find('W')
            self.dilations = [1, 1, self.paras.get("dilations")[pos_h], self.paras.get("dilations")[pos_w]]
            self.strides = [1, 1, self.paras.get("strides")[pos_h], self.paras.get("strides")[pos_w]]
            fmap_shape_ori = self.paras.get("inputs").op.attrs.get("ori_shape")
            batch_n = fmap_shape_ori[pos_batch]
            fmap_h = fmap_shape_ori[pos_h]
            fmap_w = fmap_shape_ori[pos_w]
            c_in = fmap_shape_ori[pos_c]
            filter_shape_ori = self.paras.get("weights").op.attrs.get("ori_shape")
            c_out = filter_shape_ori[pos_cout]
            k_h = filter_shape_ori[pos_kh]
            k_w = filter_shape_ori[pos_kw]
            self.pads = self.paras.get("pads")
            ci1 = ceil_div(c_in, block_size_k)
            co1 = ceil_div(c_out, block_size_n)
            input_shape_5hd = (batch_n, ci1, fmap_h, fmap_w, block_size_k)
            w_shape = (c_out, c_in, k_h, k_w)
            # quantconv2d not support group > 1, enlarge and group_opt = 1 too
            enlarge = 1
            group_opt = 1
            cin1_opt = ci1
            cout1_opt = co1
            self.groups = self.paras.get("groups")
        else:
            self.dilations = [1, 1, dilation_h_var, dilation_w_var]
            self.strides = [1, 1, stride_h_var, stride_w_var]
            self.pads = [pad_top, pad_bottom, pad_left, pad_right]
            ci1 = ceil_div(c_in, block_size_k)
            input_shape_5hd = (batch_n, ci1, fmap_h, fmap_w, block_size_k)
            w_shape = (c_out, c_in, k_h, k_w)
            group = get_te_var(op_util_conv2d.TilingDataKey.GROUP).get_tvm_var()
            self.groups = group

        w_shape_fracz = (get_weight_fz_k1(), cout1_opt, block_size_n, block_size_k)
        group_para = {"enlarge": enlarge, "c1_opt": cin1_opt, "cout1_opt": cout1_opt, "group_opt": group_opt}

        return {"in_shape_nc1hwc0": input_shape_5hd, "w_shape_frac_z": w_shape_fracz,
                "w_shape": w_shape, "group_para": group_para,
                "correct_range_flag": False,
                "new_in_range": [(1, None), (1, None), (1, None), (1, None)],
                "ori_shape_attr": self.ori_shape_attr}

    def config_paras(self, options=None):
        """
        config paras and placeholders
        """
        self.check_dynamic_mode()
        if self.cache_tiling_flag:
            log.debug("Begin binary model processing!")
            param = self.cache_tiling_paras_process(options)
        else:
            param = self.check_paras()
        if self.is_tensor is False:
            input_tensor = tvm.placeholder(param.get("in_shape_nc1hwc0"), name="Fmap", dtype=self.dtype)
            if self.op_type == "pooling":
                weight_shape = self.weights.get("shape")
                weight_tensor = tvm.placeholder(weight_shape, name="Filter", dtype=self.dtype)
            else:
                weight_tensor = tvm.placeholder(param.get("w_shape_frac_z"), name="Filter", dtype=self.dtype)
            if self.bias:
                bias_tensor = tvm.placeholder((param.get("w_shape")[N_DIM],), name="bias_tensor",
                                              dtype=self.bias.get("dtype"))
            else:
                bias_tensor = None
        else:
            input_tensor = self.input_tensor
            weight_tensor = self.weights_tensor
            bias_tensor = self.bias_tensor

        optim_dict = self.paras.get("optim_dict")
        if options is not None:
            optim_dict.update(options)

        return {"input_tensor": input_tensor, "weight_tensor": weight_tensor, "bias_tensor": bias_tensor,
                "w_shape": param.get("w_shape"), "in_shape_nc1hwc0": param.get("in_shape_nc1hwc0"),
                "w_shape_frac_z": param.get("w_shape_frac_z"), "group_para": param.get("group_para"),
                "correct_range_flag": param.get("correct_range_flag", False), "new_in_range": param.get("new_in_range"),
                "optim_dict": optim_dict, "cache_tiling_flag": self.cache_tiling_flag,
                "ori_shape_attr": param.get("ori_shape_attr")}

    def _calc_shape(self, in_shape, w_shape, in_range, y_range, group_para):
        """
        calculate shape for mmad
        """

        block_size_k, block_size_n = tbe_platform.CUBE_MKN[self.dtype]['mac'][1:3]
        in_shape[C_DIM] = align(in_shape[C_DIM], block_size_k)
        # filter channel should be equal input channel
        w_shape[C_DIM] = in_shape[C_DIM]

        in_shape_nc1hwc0 = [in_shape[N_DIM], in_shape[C_DIM] // block_size_k,
                            in_shape[H_DIM], in_shape[W_DIM], block_size_k]
        if self.is_tensor is False:
            if in_shape_nc1hwc0[N_DIM] == DYNAMIC_FLAG:
                in_shape_nc1hwc0[N_DIM] = operation.var("batch_n", in_range[N_DIM])
                operation.add_exclude_bound_var(in_shape_nc1hwc0[N_DIM])
            if in_shape_nc1hwc0[H_DIM] == DYNAMIC_FLAG:
                in_shape_nc1hwc0[H_DIM] = operation.var("fmap_h", in_range[H_DIM])
                operation.add_exclude_bound_var(in_shape_nc1hwc0[H_DIM])
                operation.add_exclude_bound_var(operation.var("ho", y_range[H_DIM]))
            if in_shape_nc1hwc0[W_DIM] == DYNAMIC_FLAG:
                in_shape_nc1hwc0[W_DIM] = operation.var("fmap_w", in_range[W_DIM])
                operation.add_exclude_bound_var(in_shape_nc1hwc0[W_DIM])
                operation.add_exclude_bound_var(operation.var("wo", y_range[W_DIM]))
        else:
            if in_shape_nc1hwc0[N_DIM] == DYNAMIC_FLAG:
                in_shape_nc1hwc0[N_DIM] = self.input_tensor.shape[N_DIM]
            if in_shape_nc1hwc0[H_DIM] == DYNAMIC_FLAG:
                in_shape_nc1hwc0[H_DIM] = self.input_tensor.shape[H_DIM]
                if not operation.get_te_var("ho"):
                    operation.add_exclude_bound_var(operation.var("ho", y_range[H_DIM]))
            if in_shape_nc1hwc0[W_DIM] == DYNAMIC_FLAG:
                in_shape_nc1hwc0[W_DIM] = self.input_tensor.shape[W_DIM]
                if not operation.get_te_var("wo"):
                    operation.add_exclude_bound_var(operation.var("wo", y_range[W_DIM]))

        if self.paras.get("optim_dict").get("c0_optim_flg"):
            w_shape_frac_z = (ceil_div(4 * w_shape[H_DIM] * w_shape[W_DIM], block_size_k),
                              math.ceil(w_shape[N_DIM] / block_size_n), block_size_n, block_size_k)
        else:
            w_shape_frac_z = (group_para.get("group_opt") * group_para.get("c1_opt") * w_shape[H_DIM] * w_shape[W_DIM],
                              group_para.get("cout1_opt"), block_size_n, block_size_k)
        return in_shape, w_shape, in_shape_nc1hwc0, w_shape_frac_z
