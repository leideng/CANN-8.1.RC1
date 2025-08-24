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
avg_pool_v2
"""
from typing import Union
import tbe.dsl as tbe_base
from tbe import tvm
from tbe.common.platform.platform_info import get_soc_spec
from tbe.common.utils import log
from tbe.dsl import auto_schedule
from tbe.dsl import build
from tbe.dsl.compute.conv_compute import conv
from tbe.dsl.compute.conv_compute import ConvParam
from impl.util.platform_adapter import error_manager_vector
from impl.util.util_cube_dynamic import Conv2dParaProcess
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import error_manager_cube
from impl.util.util_conv2d_dynamic import check_input_range
from impl.util.util_conv2d_dynamic import check_range_l1_size
from impl.util.platform_adapter import tbe_register
from impl.util.util_conv2d_dynamic import check_range_value
from impl.util.util_conv2d_dynamic import check_graph_mode
from tbe.common.utils.op_util.op_util_conv2d import is_support_fixpipe


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    N_DIM = 0
    C_DIM = 1
    H_DIM = 2
    W_DIM = 3
    NONETYPE = type(None)
    DYNAMIC_FLAG = -1
    # kh, kw must be in [1,255]
    KSIZE_HW_MIN = 1
    KSIZE_HW_MAX = 255
    # kh * kw must be in [1,255]
    AVGV2_KERNEL_SIZE_H_MUL_W = 255
    # padH, padW must be in [0,255]
    PAD_MIN = 0
    PAD_MAX = 255
    # stride must be in [1,63]
    STRIDE_MIN = 1
    STRIDE_MAX = 63
    # avgpoolv2 divisor_override must be in [1, 255]
    AVGV2_DIVISOR_OVERRIDE_MIN = 1
    AVGV2_DIVISOR_OVERRIDE_MAX = 255
    # when cube, kh and kw must be in [1, 20], kh * kw must be in [1, 255]
    KSIZE_MAX = 20
    KSIZE_MULTI_MAX = 255


# 'pylint: disable=locally-disabled,too-many-arguments
# 'pylint: disable=invalid-name,redefined-builtin,too-many-locals,unused-argument,no-else-raise,unnecessary-lambda
def check_supported(x, weight, bias, y, ksize, strides, padding="CALCULATED", pads=(0, 0, 0, 0),
                    data_format="NCHW", global_pooling=False, ceil_mode=False,
                    exclusive=True, offset_x=0, divisor_override=0, kernel_name="avg_pool_v2"):
    """
    Parameters
    ----------
    x : dict, shape and dtype of input_data, only support float16, shape is 4
        dims, format is NCHW

    weight : assist matrix

    bias: dict with keys(shape and dtype) or None
        input bias tensor

    y : dict, shape and dtype of output_data, only support float16

    ksize : list or tuple, the window of avgpooling, only support avgpooling
            in H or W

    strides : list or tuple, the stride of avgpooling window, only support
              avgpooling in H or W

    padding : str, the mode of padding, support VALID, SAME and CALCULATED

    pads : padding value when padding_mode is CALCULATED

    data_format : str, default = "NCHW"

    global_pooling : global pooling or not

    ceil_mode : use ceil or floor to calculate ho and wo when padding_mode is CALCULATED

    exclusive : ignore padding area or not when calculating the average

    kernel_name : cce kernel name, default value is "avg_pool_v2"

    Returns
    -------
    True or False
    """
    if data_format == "NHWC":
        ksize_h = ksize[1]
        ksize_w = ksize[2]
        stride_h = strides[1]
        stride_w = strides[2]
    elif data_format == "NCHW":
        ksize_h = ksize[2]
        ksize_w = ksize[3]
        stride_h = strides[2]
        stride_w = strides[3]
    is_support_kernel = (ksize_h * ksize_w <= Constant.AVGV2_KERNEL_SIZE_H_MUL_W)
    reason_kernel = "the shape is not supported by aicore cube schedule, ksize:%s " % (str(ksize))
    is_strides_valid = stride_h <= Constant.STRIDE_MAX and stride_w <= Constant.STRIDE_MAX
    reason_strides = "the strides are invalid for aicore, stride_h: %s, stride_w: %s" % (str(stride_h), str(stride_w))
    is_divisor_override_valid = divisor_override >= 0 and divisor_override <= Constant.AVGV2_DIVISOR_OVERRIDE_MAX
    reason_divisor_override = "the divisor_overrride is invalid, divisor_override = %s" % (str(divisor_override))
    if not is_support_kernel:
        return False, reason_kernel
    if not is_strides_valid:
        return False, reason_strides
    if not is_divisor_override_valid:
        return False, reason_divisor_override
    return True, ""


# 'pylint: disable=unused-variable,too-many-arguments,too-many-locals,too-many-arguments,invalid-name
def get_attr_nchw_format(input_shape, ksize, strides):
    """
    get attr nchw format
    """
    batch, hi, wi, channel = input_shape
    kn, kh, kw, kc = ksize
    stride_n, stride_h, stride_w, stride_c = strides

    input_shape = [batch, channel, hi, wi]
    ksize = [kn, kc, kh, kw]
    strides = [stride_n, stride_c, stride_h, stride_w]

    return input_shape, ksize, strides


# 'pylint: disable=unused-variable,too-many-arguments,too-many-locals,too-many-arguments,invalid-name
def check_avgpoolv2_params(input_shape, input_type, output_shape, output_type, ksize, strides,
                           padding, pads, divisor_override):
    """
    check params valid
    """
    para_check.check_shape(input_shape)
    para_check.check_shape(output_shape)
    para_check.check_dtype(input_type, ["float16"])
    para_check.check_dtype(output_type, ["float16"])

    if len(ksize) != 4:
        error_manager_vector.raise_err_input_value_invalid("AvgPoolV2", "ksize", "4", len(ksize))
    if len(strides) != 4:
        error_manager_vector.raise_err_input_value_invalid("AvgPoolV2", "strides", "4", len(strides))
    if len(pads) != 4:
        error_manager_vector.raise_err_input_value_invalid("AvgPoolV2", "pads", "4", len(pads))

    if padding not in ("CALCULATED", "VALID", "SAME"):
        error_manager_vector.raise_err_specific_reson("AvgPoolV2", "Padding mode only support CALCULATED, "
                                                      "VALID or SAME!")

    if ksize[Constant.H_DIM] == -1 or ksize[Constant.W_DIM] == -1:
        error_manager_vector.raise_err_specific_reson("AvgPoolV2", "The ksize of the H and W dimensions "
                                                      "must be more than zero")
    if strides[Constant.H_DIM] == -1 or strides[Constant.W_DIM] == -1:
        error_manager_vector.raise_err_specific_reson("AvgPoolV2", "The strides of the H and W dimensions "
                                                      "must be more than zero")

    # pads must be less than ksize
    if pads[0] >= ksize[Constant.H_DIM] or pads[1] >= ksize[Constant.H_DIM]:
        error_manager_vector.raise_err_specific_reson("AvgPoolV2", "Pad_h must be less than kernel_h")
    if pads[2] >= ksize[Constant.W_DIM] or pads[3] >= ksize[Constant.W_DIM]:
        error_manager_vector.raise_err_specific_reson("AvgPoolV2", "Pad_w must be less than kernel_w")

    # The ksize/strides of the N and C dimensions are 1
    if ksize[Constant.N_DIM] != 1 or ksize[Constant.C_DIM] != 1:
        error_manager_vector.raise_err_specific_reson("AvgPoolV2", "The ksize of the N and C dimensions are 1")
    if strides[Constant.N_DIM] != 1 or strides[Constant.C_DIM] != 1:
        error_manager_vector.raise_err_specific_reson("AvgPoolV2", "The strides of the N and C dimensions are 1")

    # The divisor_override must be in [0, 255]; 0 or None is default, not enable divisor_override.
    if not divisor_override and (divisor_override < 0 or divisor_override > Constant.AVGV2_DIVISOR_OVERRIDE_MAX):
        error_manager_vector.raise_err_specific_reson("AvgPoolV2", "The divisor_override must be in [0, 255]")

    if divisor_override == 0:
        log.debug("avg_pool_v2 divisor_override is zero, kernel size will be used as divisor.")


def check_cube_params(input_shape, ksize, strides, pads):
    """
    check params valid
    """
    if input_shape[Constant.C_DIM] == -1:
        error_manager_vector.raise_err_specific_reson("AvgPoolV2", "Don't support dynamic in C dimensions")

    range_value = "".join([str(Constant.KSIZE_HW_MIN), ",", str(Constant.KSIZE_HW_MAX)])
    if ksize[Constant.H_DIM] < Constant.KSIZE_HW_MIN or ksize[Constant.H_DIM] > Constant.KSIZE_HW_MAX:
        error_manager_vector.raise_err_attr_range_invalid("AvgPoolV2", range_value, "kernel_h", \
        str(ksize[Constant.H_DIM]))
    if ksize[Constant.W_DIM] < Constant.KSIZE_HW_MIN or ksize[Constant.W_DIM] > Constant.KSIZE_HW_MAX:
        error_manager_vector.raise_err_attr_range_invalid("AvgPoolV2", range_value, "kernel_h", \
        str(ksize[Constant.W_DIM]))

    range_value = "".join([str(Constant.STRIDE_MIN), ",", str(Constant.STRIDE_MAX)])
    if strides[Constant.H_DIM] < Constant.STRIDE_MIN or strides[Constant.H_DIM] > Constant.STRIDE_MAX:
        error_manager_vector.raise_err_attr_range_invalid("AvgPoolV2", range_value, "stride_h", \
        str(strides[Constant.H_DIM]))
    if strides[Constant.W_DIM] < Constant.STRIDE_MIN or strides[Constant.W_DIM] > Constant.STRIDE_MAX:
        error_manager_vector.raise_err_attr_range_invalid("AvgPoolV2", range_value, "stride_w", \
        str(strides[Constant.W_DIM]))

    range_value = "".join([str(Constant.PAD_MIN), ",", str(Constant.PAD_MAX)])
    if Constant.DYNAMIC_FLAG in pads:
        return
    for pad_value in pads:
        if pad_value < Constant.PAD_MIN or pad_value > Constant.PAD_MAX:
            error_manager_vector.raise_err_attr_range_invalid("AvgPoolV2", range_value, "pads", str(pad_value))


def get_correct_pad(in_pad):
    """
    correct pads when less than zero
    """
    if in_pad < 0:
        out_pad = 0
    else:
        out_pad = in_pad

    return out_pad


# 'pylint: disable=unused-variable,too-many-arguments,too-many-locals,too-many-arguments,invalid-name
def calculate_pads(input_shape, ksize, strides, padding, pads, ceil_mode, hw_dynamic_flag):
    """
    calculate pads
    """
    input_h, input_w = input_shape[Constant.H_DIM], input_shape[Constant.W_DIM]
    k_h, k_w = ksize[Constant.H_DIM], ksize[Constant.W_DIM]
    stride_h, stride_w = strides[Constant.H_DIM], strides[Constant.W_DIM]

    if padding == "SAME":
        if hw_dynamic_flag:
            correct_pads = [-1, -1, -1, -1]
        else:
            output_h = (input_h + stride_h - 1)//stride_h
            output_w = (input_w + stride_w - 1)//stride_w
            pad_row = (output_h - 1)*stride_h + k_h - input_h
            pad_col = (output_w - 1)*stride_w + k_w - input_w
            pad_top = get_correct_pad(pad_row//2)
            pad_bottom = get_correct_pad(pad_row - pad_top)
            pad_left = get_correct_pad(pad_col//2)
            pad_right = get_correct_pad(pad_col - pad_left)

            correct_pads = [pad_top, pad_bottom, pad_left, pad_right]
    elif padding == "CALCULATED":
        if hw_dynamic_flag:
            correct_pads = pads
        else:
            pad_top, pad_bottom, pad_left, pad_right = pads
            if ceil_mode:
                output_h = (input_h - k_h + pad_top + pad_bottom + stride_h - 1)//stride_h + 1
                output_w = (input_w - k_w + pad_left + pad_right + stride_w - 1)//stride_w + 1
            else:
                output_h = (input_h - k_h + pad_top + pad_bottom)//stride_h + 1
                output_w = (input_w - k_w + pad_left + pad_right)//stride_w + 1
            pad_bottom = get_correct_pad((output_h - 1)*stride_h + k_h - input_h - pad_top)
            pad_right = get_correct_pad((output_w - 1)*stride_w + k_w - input_w - pad_left)

            correct_pads = [pad_top, pad_bottom, pad_left, pad_right]
    else:
        if ceil_mode:
            if hw_dynamic_flag:
                correct_pads = [0, 0, 0, 0]
            else:
                output_h = (input_h - k_h + stride_h - 1)//stride_h + 1
                output_w = (input_w - k_w + stride_w - 1)//stride_w + 1
                pad_bottom = get_correct_pad((output_h - 1)*stride_h + k_h - input_h)
                pad_right = get_correct_pad((output_w - 1)*stride_w + k_w - input_w)

                correct_pads = [0, pad_bottom, 0, pad_right]
        else:
            correct_pads = [0, 0, 0, 0]

    return correct_pads


# 'pylint: disable=unused-variable,too-many-arguments,too-many-locals,too-many-arguments,invalid-name
def calculate_pads_expr(in_shape_nc1hwc0, ksize, strides, padding, cor_pads, ceil_mode, hw_dynamic_flag):
    """
    calculate dynamic pads var
    """
    input_h, input_w = in_shape_nc1hwc0[Constant.H_DIM], in_shape_nc1hwc0[Constant.W_DIM]
    k_h, k_w = ksize[Constant.H_DIM], ksize[Constant.W_DIM]
    stride_h, stride_w = strides[Constant.H_DIM], strides[Constant.W_DIM]

    if hw_dynamic_flag:
        # calculate pad expr
        if padding == "SAME":
            output_h = (input_h + stride_h - 1)//stride_h
            output_w = (input_w + stride_w - 1)//stride_w
            pad_row = (output_h - 1)*stride_h + k_h - input_h
            pad_col = (output_w - 1)*stride_w + k_w - input_w
            # `pad_row with tvm expr`
            pad_row = tvm.max(pad_row, 0)
            pad_top = pad_row//2
            pad_bottom = pad_row - pad_top
            # `pad_col with tvm expr`
            pad_col = tvm.max(pad_col, 0)
            pad_left = pad_col//2
            pad_right = pad_col - pad_left
        elif padding == "CALCULATED":
            pad_top, pad_bottom, pad_left, pad_right = cor_pads
            if ceil_mode:
                output_h = (input_h - k_h + pad_top + pad_bottom + stride_h - 1)//stride_h + 1
                output_w = (input_w - k_w + pad_left + pad_right + stride_w - 1)//stride_w + 1
            else:
                output_h = (input_h - k_h + pad_top + pad_bottom)//stride_h + 1
                output_w = (input_w - k_w + pad_left + pad_right)//stride_w + 1
            pad_bottom = (output_h - 1)*stride_h + k_h - input_h - pad_top
            pad_right = (output_w - 1)*stride_w + k_w - input_w - pad_left
            pad_bottom = tvm.max(pad_bottom, 0)
            pad_right = tvm.max(pad_right, 0)
        else:  # VALID
            pad_top, pad_bottom, pad_left, pad_right = cor_pads
            if ceil_mode:
                output_h = (input_h - k_h + stride_h - 1)//stride_h + 1
                output_w = (input_w - k_w + stride_w - 1)//stride_w + 1
                pad_bottom = (output_h - 1)*stride_h + k_h - input_h
                pad_right = (output_w - 1)*stride_w + k_w - input_w
                pad_bottom = tvm.max(pad_bottom, 0)
                pad_right = tvm.max(pad_right, 0)

        cor_pads = [pad_top, pad_bottom, pad_left, pad_right]
        cor_pads = list(map(lambda x: int(x) if (isinstance(x, tvm.tir.IntImm)) else x, cor_pads))
    return cor_pads


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


def check_hw_is_dynamic(input_shape):
    """
    input_shape format NCHW
    """
    if input_shape[Constant.H_DIM] == -1 or input_shape[Constant.W_DIM] == -1:
        return True

    return False


def check_avg_pool_v2_range(x, ksize, strides, padding, pads):
    """
    check if dynamic input range is supported
    """
    op_type = "avg_pool_v2"
    x_format = x.get("ori_format")
    input_range = x.get("ori_range")

    if x_format == "NCHW":
        idx_h = 2
        idx_w = 3
    elif  x_format == "NHWC":
        idx_h = 1
        idx_w = 2
    else:
        error_manager_cube.raise_err_specific_user("avg_pool_v2", "input fmap format only support NCHW or NHWC.")

    check_range_value(op_type, input_range, idx_h, idx_w)

    kh = ksize[idx_h]
    kw = ksize[idx_w]

    if padding == "SAME":
        correct_pads = [-1, -1, -1, -1]
    elif padding == "CALCULATED":
        correct_pads = pads
    else:  # VALID
        correct_pads = [0, 0, 0, 0]
    low_check = check_input_range(input_range, idx_h, idx_w, kh, kw, correct_pads)
    up_check = check_range_l1_size(x, kh, kw, strides, correct_pads)
    if not up_check and not low_check:
        return []

    type_info = []
    if low_check:
        type_info.append(low_check)
    elif up_check:
        type_info.append(up_check)

    check_result = [{"result": "UNSUPPORTED", "reason": {"param_index": [0], "type": type_info}}]
    return check_result


def generalize_tensor_ori_shape_range(tensor):
    op_type = "Avgpool_v2"
    ori_format = tensor.get("ori_format")

    if ori_format not in ["NCHW", "NHWC"]:
        error_manager_cube.raise_err_specific_user(op_type, "input fmap format only support NCHW or NHWC.")

    tensor["ori_shape"] = [Constant.DYNAMIC_FLAG, Constant.DYNAMIC_FLAG, Constant.DYNAMIC_FLAG, Constant.DYNAMIC_FLAG]
    tensor["ori_range"] = [[1, Constant.DYNAMIC_FLAG], [1, Constant.DYNAMIC_FLAG],
                           [1, Constant.DYNAMIC_FLAG], [1, Constant.DYNAMIC_FLAG]]


def get_op_specific_info(x, weight, bias, y, ksize, strides, padding="CALCULATED", pads=(0, 0, 0, 0),
                         data_format="NCHW", global_pooling=False, ceil_mode=False,
                         exclusive=True, offset_x=0, divisor_override=0, kernel_name="avg_pool_v2"):
    """
    get the avgpool prebuild pattern

    """
    if weight is not None:
        return '{"prebuildPattern": "Convolution"}'
    return '{"prebuildPattern": "undefined"}'


def IsCubeAvgpool(x, ksize, strides, global_pooling):
    op_type = "avg_pool_v2"
    x_format = x.get("ori_format")

    if x_format == "NCHW":
        idx_h = 2
        idx_w = 3
    elif  x_format == "NHWC":
        idx_h = 1
        idx_w = 2
    else:
        error_manager_cube.raise_err_specific_user("avg_pool_v2", "input fmap format only support NCHW or NHWC.")

    kh = ksize[idx_h]
    kw = ksize[idx_w]
    strideh = strides[idx_h]
    stridew = strides[idx_w]
    return kh*kw <= Constant.KSIZE_MULTI_MAX and \
           kh >= Constant.KSIZE_HW_MIN and kw >= Constant.KSIZE_HW_MIN and \
           strideh <= Constant.STRIDE_MAX and stridew <= Constant.STRIDE_MAX and \
           strideh >= Constant.STRIDE_MIN and stridew >= Constant.STRIDE_MIN and not global_pooling


@tbe_register.register_param_generalization("AvgPoolV2")
def avg_pool_v2_generalization(x: dict, weight: dict, bias: dict, y: dict, ksize: Union[tuple, list],
                               strides: Union[tuple, list], padding: str = "CALCULATED", pads: tuple = (0, 0, 0, 0),
                               data_format: str = "NCHW", global_pooling: bool = False, ceil_mode: bool = False,
                               exclusive: bool = True, offset_x: int = 0, divisor_override: int = 0,
                               kernel_name: str = "avg_pool_v2", generalize_config: dict = None) -> list:
    """
    avg_pool_v2 generalization

    Notice
    ------
    run after infershape and before operator compile
    only check if dynamic input range is supported

    for use:
        1. te fusion distinguish .o (remove the generalization dim)
        2. pass them to the operator to follow the dynanmic shape process

    Parameters
    ----------
    same to avg_pool_v2

    Returns
    -------
    list of params list if supported
    list of unsupported param index if unsupported
    """
    # check generalize_config
    if generalize_config is None:
        generalize_config = {"mode": "keep_rank"}
    support_mode = ["keep_rank"]
    if generalize_config.get("mode") not in support_mode:
        error_manager_cube.raise_err_specific_user("avg_pool_v2", "invalid generalize mode {}, only support {}".format(
            str(generalize_config.get("mode")), str(support_mode)))

    # unknow_rank inputs ori_shape is [-2], others' shape length is 4
    unknow_rank = len(x["ori_shape"]) == 1 and x["ori_shape"][0] == -2
    if unknow_rank:
        error_manager_cube.raise_err_specific_user("avg_pool_v2", "not support unknow_rank under mode {}".format(
            generalize_config.get("mode")))

    log.debug("avg_pool_v2 generalization inputs: %s", x)
    if check_graph_mode(x):
        if is_support_fixpipe():
            for tensor in [x, weight, y]:
                generalize_tensor_ori_shape_range(tensor)
        else:
            # check if range of inputs is supported or not
            check_result = check_avg_pool_v2_range(x, ksize, strides, padding, pads)
            if check_result:
                log.debug("avg_pool_v2 generalization invalid range, check result: %s", check_result)
                return check_result
    else:
        if IsCubeAvgpool(x, ksize, strides, global_pooling) and is_support_fixpipe():
            for tensor in [x, y]:
                generalize_tensor_ori_shape_range(tensor)

    result = [[x, weight, bias, y, ksize, strides, padding,
              pads, data_format, global_pooling, ceil_mode,
              exclusive, offset_x, divisor_override, kernel_name]]

    log.debug("avg_pool_v2 generalization result: %s", result)
    return result


# 'pylint: disable=unused-variable,too-many-arguments,too-many-locals
# 'pylint: disable=too-many-arguments,invalid-name,too-many-statements
@register_operator("AvgPoolV2")
@para_check.check_input_type(dict, (dict, Constant.NONETYPE), (dict, Constant.NONETYPE), dict, (list, tuple),
                             (list, tuple), str, (list, tuple), str, bool, bool, bool, int, (int, Constant.NONETYPE),
                             str)
def avg_pool_v2(x, weight, bias, y, ksize, strides, padding="CALCULATED", pads=(0, 0, 0, 0),
                data_format="NCHW", global_pooling=False, ceil_mode=False,
                exclusive=True, offset_x=0, divisor_override=0, kernel_name="avg_pool_v2"):
    """
    Parameters
    ----------
    x: dict, shape and dtype of input_data, only support float16, shape is 4
        dims, format is NCHW

    weight: assist matrix

    y: dict, shape and dtype of output_data, only support float16

    ksize: list or tuple, the window of avgpooling, only support avgpooling
            in H or W

    strides: list or tuple, the stride of avgpooling window, only support
              avgpooling in H or W

    padding: str, the mode of padding, support VALID, SAME and CALCULATED

    pads: padding value when padding_mode is CALCULATED

    data_format: str, default = "NCHW"

    global_pooling: global pooling or not

    ceil_mode: use ceil or floor to calculate ho and wo when padding_mode is CALCULATED

    exclusive: ignore padding area or not when calculating the average

    kernel_name: cce kernel name, default value is "avg_pool_v2"

    Returns
    -------
    None
    """
    if global_pooling:
        error_manager_vector.raise_err_specific_reson("AvgPoolV2", "Don't support global_pooling!")

    # check data format
    input_format = x.get("ori_format")
    if input_format not in ("NCHW", "NHWC"):
        error_manager_vector.raise_err_input_format_invalid("AvgPoolV2", "input", ["NCHW", "NHWC"], input_format)

    output_format = y.get("ori_format")
    if output_format not in ("NCHW", "NHWC"):
        error_manager_vector.raise_err_input_format_invalid("AvgPoolV2", "output", ["NCHW", "NHWC"], output_format)

    if data_format not in ("NCHW", "NHWC"):
        error_manager_vector.raise_err_input_format_invalid("AvgPoolV2", "input", ["NCHW", "NHWC"], data_format)

    # check params
    input_shape = x.get("ori_shape")
    input_type = x.get("dtype").lower()
    output_shape = y.get("ori_shape")
    output_type = y.get("dtype").lower()

    # nchw format attr
    if data_format == "NHWC":
        input_shape, ksize, strides = get_attr_nchw_format(input_shape, ksize, strides)

    check_avgpoolv2_params(input_shape, input_type, output_shape, output_type, ksize, strides,
                           padding, pads, divisor_override)

    # add strides compile_info
    tbe_context.get_context().add_compile_info("strides_h", strides[Constant.H_DIM])
    tbe_context.get_context().add_compile_info("strides_w", strides[Constant.W_DIM])

    if weight is None:
        log.info("[%s]: Enter avgpoolv2 vector dynamic branch", kernel_name)
        error_manager_vector.raise_err_specific_reson("AvgPoolV2", "Don't support vector function")

    log.info("[%s]: Enter avgpoolv2 cube dynamic branch", kernel_name)
    dilations = [1, 1, 1, 1]
    hw_dynamic_flag = check_hw_is_dynamic(input_shape)
    # calculate pads
    cor_pads = calculate_pads(input_shape, ksize, strides, padding, pads, ceil_mode, hw_dynamic_flag)

    filter_shape = weight.get("ori_shape")
    # check cube params
    check_cube_params(input_shape, ksize, strides, cor_pads)

    groups = filter_shape[Constant.N_DIM]
    default_para = set_default_para()
    # avgpool2 no support bias, offset_w, offset_x
    bias = None
    offset_w = None
    offset_x = 0
    ori_paras = {
        "inputs": x, "weights": weight, "bias": bias, "offset_w": offset_w,
        "outputs": y, "strides": strides, "pads": cor_pads, "dilations": dilations,
        "groups": groups, "data_format": data_format, "offset_x": offset_x,
        "kernel_name": kernel_name, "optim_dict": default_para.get("optim_dict"),
    }

    with tbe_base.compute():
        conv_para = Conv2dParaProcess(ori_paras)
        paras = conv_para.config_paras()
        in_shape_nc1hwc0 = paras.get("in_shape_nc1hwc0")
        expr_pads = calculate_pads_expr(in_shape_nc1hwc0, ksize, strides, padding, cor_pads, \
                                        ceil_mode, hw_dynamic_flag)

        pad_t, pad_b, pad_l, pad_r = expr_pads
        conv_res = conv(paras.get("input_tensor"), paras.get("weight_tensor"),
                        {"bias_tensor": paras.get("bias_tensor"),
                        "offset_w_tensor": offset_w,
                        "pad_h": [pad_t, pad_b], "pad_w": [pad_l, pad_r],
                        "stride_h": strides[Constant.H_DIM], "stride_w": strides[Constant.W_DIM],
                        "dilate_h": dilations[Constant.H_DIM], "dilate_w": dilations[Constant.W_DIM],
                        "filter_h": paras.get("w_shape")[Constant.H_DIM],
                        "filter_w": paras.get("w_shape")[Constant.W_DIM],
                        "offset_x": offset_x,
                        "res_dtype": output_type,
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
                        "pooling_mode": paras.get("pooling_mode"),
                        "correct_range_flag": paras.get("correct_range_flag", False),
                        "new_in_range": paras.get("new_in_range")},
                        optim_dict=None,
                        dsl_flag=True)

        res_shape = conv_res.shape
        input_h, input_w = paras.get("input_tensor").shape[2:4]
        out_h = ConvParam.h_out
        out_w = ConvParam.w_out

        k_h, k_w = ksize[Constant.H_DIM], ksize[Constant.W_DIM]
        stride_h, stride_w = strides[Constant.H_DIM], strides[Constant.W_DIM]

        if divisor_override >= Constant.AVGV2_DIVISOR_OVERRIDE_MIN and \
            divisor_override <= Constant.AVGV2_DIVISOR_OVERRIDE_MAX:
            c_ub_avg = tvm.compute(res_shape, lambda n, c1, m, c0:
            conv_res(n, c1, m, c0) * tvm.const(1 / (divisor_override)).astype("float16"),
                                    name="c_ub_avg",
                                    tag="elewise_binary_mul")
        else :
            # factor area same when exclusive=False or no padding
            if not exclusive or (padding == "VALID" and not ceil_mode):
                c_ub_avg = tvm.compute(res_shape, lambda n, c1, m, c0:
                                        conv_res(n, c1, m, c0)*tvm.const(1/(k_h*k_w)).astype("float16"),
                                        name="c_ub_avg",
                                        tag="elewise_binary_mul")
            else:
                mean_matrix_shape = res_shape[2:4]
                mean_matrix_avgv2 = tvm.compute(mean_matrix_shape, lambda m, c0:
                                                tvm.max(
                                                    (tvm.min((m // out_w)*stride_h-pad_t+k_h, input_h) -
                                                        tvm.max((m // out_w)*stride_h-pad_t, 0)) *
                                                    (tvm.min((m % out_w)*stride_w-pad_l+k_w, input_w) -
                                                        tvm.max((m % out_w)*stride_w-pad_l, 0)), 1),
                                                name="mean_matrix_avgv2")

                mean_matrix_fp16 = tvm.compute(mean_matrix_shape, lambda m, c0:
                                                mean_matrix_avgv2(m, c0).astype("float16"),
                                                name="mean_matrix_fp16",
                                                tag="elewise_single_cast")

                if get_soc_spec("SHORT_SOC_VERSION") in ("Ascend310",):
                    mean_matrix_rec = tvm.compute(mean_matrix_shape, lambda m, c0:
                                                    1/mean_matrix_fp16(m, c0),
                                                    name="mean_matrix_rec",
                                                    tag="elewise_single_rec")

                    c_ub_avg = tvm.compute(res_shape, lambda n, c1, m, c0:
                                            conv_res(n, c1, m, c0)*mean_matrix_rec(m, c0),
                                            name="c_ub_avg",
                                            tag="elewise_binary_mul")
                else:
                    c_ub_avg = tvm.compute(res_shape, lambda n, c1, m, c0:
                                            tvm.div(conv_res(n, c1, m, c0), mean_matrix_fp16(m, c0)),
                                            name="c_ub_avg",
                                            tag="elewise_binary_div")
                ConvParam.tensor_map["mean_matrix_avgv2"] = mean_matrix_avgv2

    with tvm.target.cce():
        sch = auto_schedule(c_ub_avg)
    tensor_list = [paras.get("input_tensor"), paras.get("weight_tensor"), c_ub_avg]
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list,
        "build_args": {"constant_realize_extent_in_infer_bound": False, "dummy_placeholder": True}
    }
    build(sch, config)
