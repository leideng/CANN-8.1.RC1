#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

conv2d common
provide common function used by conv2d
"""

import math
from collections import deque
from impl.util import util_select_op_base
import tbe
from tbe import tvm
from tbe.common.buildcfg import set_current_build_config
from tbe.common.utils import para_check
from tbe.common.utils import log
from tbe.common.utils.op_util import op_util_conv2d
from tbe.common.utils.errormgr import error_manager_cube as err_man
from tbe.common.platform import get_soc_spec
from tbe.common.platform import CUBE_MKN
from tbe.common.platform import get_cube_mkn
from tbe.common.context import get_context
from te.tvm.buffer_manager import get_buffer_manager
from tbe.common.utils.op_util.op_util_conv2d import check_dynamic
from tbe.common.utils.op_util.op_util_conv2d import check_load3dv2_postk_params_invalid
from tbe.common.utils.op_util.op_util_conv2d import BIT_RATIO_MAP
from tbe.common.utils.op_util.op_util_conv2d import check_range_illegal


PAD_SHAPE_DIM = 2
# fmapH, fmapW must be in [1,4096]
FMAP_HW_MIN = 1
FMAP_W_MAX = 2**32 - 1
FMAP_H_MAX = 100000
DMA_HW_MAX = 2**32 - 1

FMAP_W_MIN_SPLIT_W = 1
FMAP_W_MAX_SPLIT_W = 4294967295

# default filterH, filterW must be in [1,255]
FILTER_HW_MIN = 1
FILTER_HW_MAX = 255

# padH, padW must be in [0,255]
PAD_MIN = 0
PAD_MAX = 255

# stride must be in [1,63]
STRIDE_MIN = 1
STRIDE_MAX = 63

# dilate must be in [1,255]
DILATE_MIN = 1
DILATE_MAX = 255
CONV_SHAPE_DIM = 4

# In v200, small channel case: 4*filter_h*filter_w must be smaller than 65536.
HK_WK_C04_V200 = 65535

# V350 conv2d support range shape
FMAP_HW_RANGE_MIN = 1
FMAP_HW_RANGE_MAX = 4096
KERNEL_HW_RANGE_MIN = 1
KERNEL_HW_RANGE_MAX = 511
STRIDE_HW_RANGE_MIN = 1
STRIDE_HW_RANGE_MAX = 63
DILATE_HW_RANGE_MIN = 1
DILATE_HW_RANGE_MAX = 63
PAD_RANGE_MIN = 0
PAD_RANGE_MAX = 254
COUT_RANGE_MIN = 1
COUT_RANGE_MAX = 2048
CIN_RANGE_MIN = 1
CIN_RANGE_MAX = 2048

# index NCHW value
INDEX_N = 0
INDEX_C = 1
INDEX_H = 2
INDEX_W = 3

# conv2d support nhwc input case dtype list
CONV_ND_IN_DTYPE_LIST = ["float16", "bfloat16", "float32"]

DYNAMIC_VALUE = -1

# index params value
WEIGHT_INDEX = 1
STRIDES_INDEX = 5
PADS_INDEX = 6
DILATIONS_INDEX = 7
DATA_FORMAT_INDEX = 9


def lcm(a_val, b_val):
    return (a_val * b_val) // math.gcd(a_val, b_val)


def _shape_to_list(shape):
    """
    translate tvm.shape to list type in python
    """
    if isinstance(shape, (list, tuple)):
        return shape
    tmp = []
    if shape == "":
        return ()
    for i in shape:
        tmp.append(i.value)
    return tmp


def transform_shape_with_format(src_format: str, to_format: str,
                                ori_shape: list, format_white_list: list) -> list:
    # input format is not expected
    if ((src_format not in format_white_list) or
        (to_format not in format_white_list)):
        return None
    # need not to transform
    if src_format == to_format:
        return list(ori_shape)
    res_shape = [1 for _ in range(len(to_format))]
    for i, _ in enumerate(to_format):
        for j, _ in enumerate(src_format):
            if to_format[i] == src_format[j]:
                res_shape[i] = ori_shape[j]
                break
    return res_shape


def get_kernel_max_value():
    # filterH, filterW must be in [1,511] if v220/v300
    if op_util_conv2d.is_support_fixpipe():
        global FILTER_HW_MAX
        FILTER_HW_MAX = 511


def is_support_v200():
    """
    Check if Ascend610/BS9SX1A/Ascend310P/Hi3796CV300CS version.
    ----------

    Returns
    -------
    True:  Ascend610/BS9SX1A/Ascend310P/Hi3796CV300CS version
    False: Other version
    """
    soc_version = get_soc_spec("SHORT_SOC_VERSION")
    if soc_version in ("Ascend310P", "Ascend610", "BS9SX1A", "Hi3796CV300CS",
                       "SD3403"):
        return True
    return False


def is_support_fixpipe():
    """
    Check if support fixpipe.
    """
    return tbe.common.platform.platform_info.intrinsic_check_support("Intrinsic_fix_pipe_unit_list")


def is_support_v300():
    """
    Check v300 intrinsic support.
    """
    if tbe.common.platform.platform_info.intrinsic_check_support("Intrinsic_fix_pipe_unit_list"):
        return tbe.common.platform.platform_info.intrinsic_check_support(
            "Intrinsic_fix_pipe_unit_list", "post_eltwise")

    return False


def calc_para_from_tensor(inputs,
                          weights,
                          bias,
                          offset_w,
                          strides,
                          pads,
                          dilations,
                          offset_x,
                          groups,
                          kernel_name,
                          data_format="NCHW",
                          options=None):

    shape_w = []
    for i in weights.op.attrs['ori_shape']:
        shape_w.append(i.value)
    shape_fm = []
    multi_conv2d_fusion_flag = False
    input_nd_flag = False
    if is_support_fixpipe() and inputs.op.attrs.get("format") == "NHWC":
        input_nd_flag = True
    if len(inputs.shape) == 5:
        for i in inputs.shape:
            shape_fm.append(i.value)
    elif len(inputs.shape) == 4:
        if input_nd_flag:
            for i in inputs.shape:
                shape_fm.append(i.value)
            c0_val = CUBE_MKN[weights.dtype]['mac'][1]
            ci1_val = op_util_conv2d.ceil_div(shape_fm[3], c0_val)
            shape_fm = [shape_fm[0], ci1_val, shape_fm[1], shape_fm[2], c0_val]
        elif inputs.op.attrs['current_shape']:
            cur_shape = inputs.op.attrs['current_shape']
            if cur_shape[2].value * cur_shape[3].value != inputs.shape[2].value:
                err_man.raise_err_specific_input_shape(
                    "conv2d",
                    "the h*w of current_shape is not equal inputs.shape[3].value"
                )
            multi_conv2d_fusion_flag = True
            for i in inputs.op.attrs['current_shape']:
                shape_fm.append(i.value)
        else:
            err_man.raise_err_specific(
                "conv2d",
                "current_shape not in op.attrs on 4 dimensions tensor")
    else:
        err_man.raise_err_input_params_not_expected(
            "conv2d", "fmap", "4 dimensions or 5 dimensions",
            str(len(inputs.shape)) + " dimensions")

    input_h = shape_fm[2]
    input_w = shape_fm[3]

    format_w = weights.op.attrs['ori_format']
    all_fmt = ["NCHW", "NHWC", "HWCN"]
    if format_w not in all_fmt:
        err_man.raise_err_input_format_invalid("conv2d", "weights",
                                               ["NCHW", "NHWC", "HWCN"],
                                               format_w)

    pos_c = format_w.find('C')
    pos_h = format_w.find('H')
    pos_w = format_w.find('W')
    pos_cout = format_w.find('N')
    weight_h = shape_w[pos_h]
    weight_w = shape_w[pos_w]
    # fix the weight's channel=cin_ori
    shape_c = shape_w[pos_c] * groups
    cout_all = shape_w[pos_cout]

    if len(strides) != 4:
        err_man.raise_err_should_be_4d("conv2d", "strides")
    if len(dilations) != 4:
        err_man.raise_err_should_be_4d("conv2d", "directions")

    pos_h = data_format.find('H')
    pos_w = data_format.find('W')
    strideh = strides[pos_h]
    stridew = strides[pos_w]
    dlt_h = dilations[pos_h]
    dlt_w = dilations[pos_w]

    if len(pads) == 4:
        padh = [pads[0], pads[1]]
        padw = [pads[2], pads[3]]
    else:
        err_man.raise_err_should_be_4d("conv2d", "pads shape")

    strideh = _trans_stride(input_h, weight_h, strideh, padh, dlt_h)
    stridew = _trans_stride(input_w, weight_w, stridew, padw, dlt_w)

    c0_val = CUBE_MKN[weights.dtype]['mac'][1]
    cin_ori = shape_c // groups
    cout_ori = cout_all // groups
    if get_op_enlarge():
        enlarge = get_op_enlarge()
    else:
        enlarge = min(
            lcm(lcm(cin_ori, c0_val) // cin_ori,
                lcm(cout_ori, 16) // cout_ori), groups)
    c1_opt = math.ceil(cin_ori * enlarge / c0_val)
    cout1_opt = math.ceil(cout_ori * enlarge / 16)
    group_opt = math.ceil(groups / enlarge)

    if inputs.op.tag == "aipp_res_convolution":
        fmap_l1_addr_flag = "nothing"
        fmap_l1_valid_size = -1
        slice_offset = (0, 0, 0, 0, 0)
        buffer_manager = get_buffer_manager()
        for remapped_buffer in buffer_manager.get_remapped_buffers():
            remapped_buffer_attr = remapped_buffer.get_buffer_attr()
            if "L1_addr_flag" in remapped_buffer_attr and remapped_buffer_attr[
                    "L1_addr_flag"] != "nothing":
                fmap_l1_addr_flag = remapped_buffer_attr.get(
                    "L1_addr_flag", "nothing")
                fmap_l1_valid_size = remapped_buffer_attr.get(
                    "L1_valid_size", -1)
                slice_offset = remapped_buffer_attr.get(
                    "slice_offset", (0, 0, 0, 0, 0))
                break
    else:
        fmap_l1_addr_flag = inputs.op.attrs["L1_addr_flag"].value if "L1_addr_flag" in inputs.op.attrs else "nothing"
        fmap_l1_valid_size = inputs.op.attrs["L1_valid_size"].value if "L1_valid_size" in inputs.op.attrs else -1
        slice_offset = inputs.op.attrs["slice_offset"] if "slice_offset" in inputs.op.attrs else (0, 0, 0, 0, 0)
    fusion_para = {
        "fmap_l1_addr_flag": fmap_l1_addr_flag,
        "fmap_l1_valid_size": fmap_l1_valid_size,
        "slice_offset": slice_offset
    }

    para_dict = {
        "pad_h": padh,
        "pad_w": padw,
        "stride_h": strideh,
        "stride_w": stridew,
        "dilate_h": dlt_h,
        "dilate_w": dlt_w,
        "offset_x": offset_x,
        "filter_h": weight_h,
        "filter_w": weight_w,
        "bias_tensor": bias,
        "offset_w_tensor": offset_w,
        "fusion_para": fusion_para,
        "kernel_name": kernel_name,
        "group": groups,
        "enlarge": enlarge,
        "c1_opt": c1_opt,
        "cout1_opt": cout1_opt,
        "group_opt": group_opt,
        "a_shape": shape_fm,
        "weight_fracz_shape": _shape_to_list(weights.shape),
        "weight_ori_shape_nchw": [cout_all, shape_c, weight_h, weight_w],
        "multi_conv2d_fusion_flag": multi_conv2d_fusion_flag
    }

    c0_optim_flg = False
    use_v200_c04_flg = False

    if shape_c <= 4 and ("format" in weights.op.attrs and
                         weights.op.attrs['format'] == "FRACTAL_Z_C04"):
        c0_optim_flg = True

        if (weight_h == 1) and (weight_w == 1):
            err_man.raise_err_specific_user(
                "conv2d",
                "weight shape does not support that H and W are both equal to 1 when C0=4."
            )

        if inputs.shape[-1].value == 4 and is_support_v200():
            use_v200_c04_flg = True

    # input nd can not enable with stride_read and lxFusion, check c, h, w
    if input_nd_flag:
        ori_shape = inputs.op.attrs.get("ori_shape")
        cur_shape = inputs.op.attrs.get("current_shape")
        if ori_shape[1:] != cur_shape[1:]:
            err_man.raise_err_specific("conv2d", "nd input case ori_shape and shape not same in C,H,W axis")

    optim_dict = {
        "input_nd_flag": input_nd_flag,
        "c0_optim_flg": c0_optim_flg,
        "use_v200_c04_flg": use_v200_c04_flg,
        "invalid_data_rm": False
    }

    if options is not None:
        optim_dict.update(options)

    return para_dict, optim_dict


def calc_para_v350_from_tensor(inputs,
                               weights,
                               bias,
                               offset_w,
                               strides,
                               pads,
                               dilations,
                               offset_x,
                               groups,
                               kernel_name,
                               data_format="NCHW",
                               options=None):

    shape_w = []
    for i in weights.op.attrs['ori_shape']:
        shape_w.append(i.value)
    shape_fm = []
    if len(inputs.shape) == 5:
        for i in inputs.shape:
            shape_fm.append(i.value)
    else:
        err_man.raise_err_input_params_not_expected(
            "conv2d", "fmap", "5 dimensions for v350",
            str(len(inputs.shape)) + " dimensions")

    input_h = shape_fm[2]
    input_w = shape_fm[3]

    format_w = weights.op.attrs['ori_format']
    all_fmt = ["NCHW", "NHWC", "HWCN"]

    if format_w not in all_fmt:
        err_man.raise_err_input_format_invalid("conv2d", "weights",
                                               ["NCHW", "NHWC", "HWCN"],
                                               format_w)

    pos_c = format_w.find('C')
    pos_h = format_w.find('H')
    pos_w = format_w.find('W')
    pos_cout = format_w.find('N')
    weight_h = shape_w[pos_h]
    weight_w = shape_w[pos_w]
    cin_ori = shape_w[pos_c]
    cout_ori = shape_w[pos_cout]
    fmap_cin_ori = cin_ori * groups

    if groups != 1 and (cout_ori % groups):
        err_man.raise_err_specific_user(
            "conv2d", "groups must equal 1 or divisible by cin and cout, but given" + str(groups))
    if len(strides) != 4:
        err_man.raise_err_should_be_4d("conv2d", "strides")
    if len(dilations) != 4:
        err_man.raise_err_should_be_4d("conv2d", "directions")

    pos_h = data_format.find('H')
    pos_w = data_format.find('W')
    strideh = strides[pos_h]
    stridew = strides[pos_w]
    dlt_h = dilations[pos_h]
    dlt_w = dilations[pos_w]

    if len(pads) == 4:
        padh = [pads[0], pads[1]]
        padw = [pads[2], pads[3]]
    else:
        err_man.raise_err_should_be_4d("conv2d", "pads shape")

    strideh = _trans_stride(input_h, weight_h, strideh, padh, dlt_h)
    stridew = _trans_stride(input_w, weight_w, stridew, padw, dlt_w)

    shape_in = [shape_fm, fmap_cin_ori, input_h, input_w]
    kernel_in = [cout_ori, cin_ori, weight_h, weight_w]
    conv_param = [shape_in, kernel_in, padh, padw, strideh, stridew, dlt_h, dlt_w]
    check_conv2d_support_range(conv_param)

    _, block_size_k_fmap, block_size_n = get_cube_mkn(inputs.dtype)
    _, block_size_k_weight, _ = get_cube_mkn(weights.dtype)

    cin1_data = math.ceil(fmap_cin_ori / block_size_k_fmap)
    cin1_weight = math.ceil(cin_ori / block_size_k_weight)
    cout1 = math.ceil(cout_ori / block_size_n)

    para_dict = {
        "pad_h": padh,
        "pad_w": padw,
        "stride_h": strideh,
        "stride_w": stridew,
        "dilate_h": dlt_h,
        "dilate_w": dlt_w,
        "offset_x": offset_x,
        "weight_h": weight_h,
        "weight_w": weight_w,
        "bias_tensor": bias,
        "offset_w_tensor": offset_w,
        "kernel_name": kernel_name,
        "cin1_data": cin1_data,
        "cin1_weight": cin1_weight,
        "cout1": cout1,
        "groups": groups,
        "a_shape": shape_fm,
        "fmap_cin_ori": fmap_cin_ori,
        "weight_fracz_shape": _shape_to_list(weights.shape),
        "weight_ori_shape_nchw": [cout_ori, cin_ori, weight_h, weight_w],
    }

    optim_dict = {}

    if options is not None:
        optim_dict.update(options)

    return para_dict, optim_dict


def calc_para_from_dict(inputs,
                        weights,
                        strides,
                        pads,
                        dilations,
                        groups,
                        outputs,
                        data_format="NCHW"):
    shape_x = inputs.get("ori_shape")
    shape_x_5hd = inputs.get("shape")
    shape_w = weights.get("ori_shape")

    input_nd_flag = False
    if len(shape_x_5hd) == 4 and inputs.get("format") == "NHWC":
        input_nd_flag = True

    if len(strides) != 4:
        err_man.raise_err_should_be_4d("conv2d", "strides")
    if len(dilations) != 4:
        err_man.raise_err_should_be_4d("conv2d", "dilations")

    if len(pads) == 4:
        padh = [pads[0], pads[1]]
        padw = [pads[2], pads[3]]
    else:
        err_man.raise_err_should_be_4d("conv2d", "pads shape")

    if (not isinstance(shape_x, (tuple, list))) or len(shape_x) != 4:
        err_man.raise_err_should_be_4d("conv2d", "inputs")

    if (not isinstance(shape_w, (tuple, list))) or len(shape_w) != 4:
        err_man.raise_err_should_be_4d("conv2d", "weights")

    format_x = inputs.get("ori_format")
    all_fmt = ["NCHW", "NHWC"]
    if format_x not in all_fmt:
        err_man.raise_err_input_format_invalid("conv2d", "inputs",
                                               ["NCHW", "NHWC"], format_x)

    pos_c = format_x.find('C')
    if len(shape_x_5hd) == 5:
        _, fmap_ci1, _, _, fmap_ci0 = shape_x_5hd
    fmap_c = shape_x[pos_c]

    if is_support_fixpipe() and len(shape_x_5hd) == 5 and fmap_c > fmap_ci1 * fmap_ci0:
        if groups != 1:
            err_man.raise_err_specific(
                "conv2d",
                "strided read is only supported when groups = 1."
                )
        shape_fm = [
            shape_x_5hd[0], fmap_ci1 * fmap_ci0, shape_x_5hd[2], shape_x_5hd[3]
        ]  # [Ni, Ci, Hi, Wi]
    elif input_nd_flag:
        # when input nd, shape trans from nhwc to nchw.
        shape_fm = [
            shape_x_5hd[0], fmap_c, shape_x_5hd[1], shape_x_5hd[2]
        ]  # [Ni, Ci, Hi, Wi]
    else:
        # only c is original value when lxfusion split batch and h.
        shape_fm = [
            shape_x_5hd[0], fmap_c, shape_x_5hd[2], shape_x_5hd[3]
        ]  # [Ni, Ci, Hi, Wi]

    pos_attr_h = data_format.find('H')
    pos_attr_w = data_format.find('W')
    strideh = strides[pos_attr_h]
    stridew = strides[pos_attr_w]
    dlt_h = dilations[pos_attr_h]
    dlt_w = dilations[pos_attr_w]

    format_w = weights.get("ori_format")
    all_fmt = ["NCHW", "NHWC", "HWCN"]
    if format_w not in all_fmt:
        err_man.raise_err_input_format_invalid("conv2d", \
            "weights", ["NCHW", "NHWC", "HWCN"], format_w)
    pos_n = format_w.find('N')
    pos_c = format_w.find('C')
    pos_h = format_w.find('H')
    pos_w = format_w.find('W')
    # fix the weight's channel=cin_ori
    shape_filter = [shape_w[pos_n], shape_fm[1], \
                    shape_w[pos_h], shape_w[pos_w]]

    fusion_para = _conv2d_fusion_para(inputs, outputs)

    input_h = shape_fm[2]
    input_w = shape_fm[3]

    strideh = _trans_stride(input_h, shape_filter[2], strideh, padh, dlt_h)
    stridew = _trans_stride(input_w, shape_filter[3], stridew, padw, dlt_w)

    c0_optim_flg = False
    use_v200_c04_flg = False

    if shape_w[pos_c] <= 4 and weights.get("format") == "FRACTAL_Z_C04":
        c0_optim_flg = True

        if (shape_w[pos_h] == 1) and (shape_w[pos_w] == 1):
            err_man.raise_err_specific_user(
                "conv2d",
                "weight shape does not support that H and W are both equal to 1 when C0=4."
            )
        if inputs.get("format") == "NC1HWC0_C04" and is_support_v200():
            use_v200_c04_flg = True

    # input nd can not enable with stride_read and lxFusion, check c, h, w
    if input_nd_flag:
        ori_shape = inputs.get("ori_shape")
        cur_shape = inputs.get("shape")
        if ori_shape[1:] != cur_shape[1:]:
            err_man.raise_err_specific("conv2d", "nd input case ori_shape and shape not same in C,H,W axis")

    optim_dict = {
        "input_nd_flag": input_nd_flag,
        "c0_optim_flg": c0_optim_flg,
        "use_v200_c04_flg": use_v200_c04_flg,
        "enable_input_4channel": inputs.get("format") == "NC1HWC0_C04"
    }

    return shape_fm, shape_filter, padh, padw, strideh, stridew, dlt_h, dlt_w, optim_dict, fusion_para


@para_check.check_input_type((list, tuple), (list, tuple), (list, int),
                             (list, int), int, int, str, str, str, str, bool,
                             str, int, int, dict, int)
def conv_layer_cce_para_check(shape_in,
                              shape_w,
                              padh,
                              padw,
                              strideh,
                              stridew,
                              in_dtype,
                              w_dtype,
                              res_dtype,
                              offset_w_dtype,
                              bias,
                              kernel_name,
                              dilateh=1,
                              dilatew=1,
                              optim_dict=None,
                              groups=1):
    """

    Parameters
    ----------
    shape_in: shape of feature map

    shape_w: shape of weight

    padh: H direction padding

    padw: W direction padding

    strideh: H direction stride

    stridew: W direction stride

    in_dtype: the feature map data type

    w_dtype: the weight data type

    res_dtype: the result data type

    offset_w_dtype: weight offset data type, default 'int32'

    bias: the tag for bias or not

    kernel_name: cce kernel name

    dilateh: H direction spacing between kernel

    dilatew: W direction spacing between kernel

    optim_dict: optimize feature dict

    Returns
    -------
    None

    """
    para_check.check_kernel_name(kernel_name)
    para_check.check_dtype_rule(offset_w_dtype, ['int32'])
    para_check.check_dtype_rule(in_dtype,
                                ("int4", 'int8', "float16", "bfloat16", "float32"))
    para_check.check_dtype_rule(w_dtype,
                                ("int4", 'int8', "float16", "bfloat16", "float32"))
    para_check.check_dtype_rule(res_dtype,
                                ('int32', "float16", "bfloat16", "float32"))

    if isinstance(padh, list):
        if len(padh) != PAD_SHAPE_DIM:
            err_man.raise_err_specific_user(
                "conv2d", "Dimension must be " + str(PAD_SHAPE_DIM) +
                " when padh is a list.")

    if isinstance(padw, list):
        if len(padw) != PAD_SHAPE_DIM:
            err_man.raise_err_specific_user(
                "conv2d", "Dimension must be " + str(PAD_SHAPE_DIM) +
                " when padw is a list.")

    if optim_dict is None:
        optim_dict = {
            "input_nd_flag": False,
            "c0_optim_flg": False,
            "use_v200_c04_flg": False
        }

    # check Cin of ori_shape
    optim_off = shape_in[1] > 4 or shape_w[1] > 4 or (shape_w[2] == 1
                                                      and shape_w[3] == 1)

    if optim_dict.get("c0_optim_flg") is True:
        if optim_off:
            err_man.raise_err_specific_user(
                "conv2d", "Invalid config for c0=4 optimize feature.")

    block_size_k = CUBE_MKN[in_dtype]['mac'][1]
    block_size_n = CUBE_MKN[w_dtype]['mac'][2]

    shape_in[1] = ((shape_in[1] + block_size_k - 1) // block_size_k)*block_size_k
    shape_w[1] = ((shape_in[1] + block_size_k - 1) // block_size_k)*block_size_k
    shape_w[0] = ((shape_w[0] + block_size_n - 1) // block_size_n)*block_size_n

    if optim_dict["c0_optim_flg"]:
        shape_in[1] = 4
        shape_w[1] = 4

    shape_in[1] = shape_in[1] // groups

    return shape_in, shape_w


@para_check.check_input_type((list, tuple), (list, tuple), (list, int),
                             (list, int), int, int, str, str, str, str, bool,
                             str, int, int, dict, int)
def conv_layer_cce_v350_para_check(shape_in,
                                   shape_w,
                                   padh,
                                   padw,
                                   strideh,
                                   stridew,
                                   in_dtype,
                                   w_dtype,
                                   res_dtype,
                                   offset_w_dtype,
                                   bias,
                                   kernel_name,
                                   dilateh=1,
                                   dilatew=1,
                                   optim_dict=None,
                                   groups=1):
    """

    Parameters
    ----------
    shape_in: shape of feature map

    shape_w: shape of weight

    padh: H direction padding

    padw: W direction padding

    strideh: H direction stride

    stridew: W direction stride

    in_dtype: the feature map data type

    w_dtype: the weight data type

    res_dtype: the result data type

    offset_w_dtype: weight offset data type, default 'int32'

    bias: the tag for bias or not

    kernel_name: cce kernel name

    dilateh: H direction spacing between kernel

    dilatew: W direction spacing between kernel

    optim_dict: optimize feature dict

    Returns
    -------
    None

    """
    para_check.check_kernel_name(kernel_name)
    para_check.check_dtype_rule(offset_w_dtype, ['int32'])
    para_check.check_dtype_rule(in_dtype, ("int16", "int8"))
    para_check.check_dtype_rule(w_dtype, ("int8"))
    para_check.check_dtype_rule(res_dtype, ("int32"))

    if isinstance(padh, list):
        if len(padh) != PAD_SHAPE_DIM:
            err_man.raise_err_specific_user(
                "conv2d", "Dimension must be " + str(PAD_SHAPE_DIM) +
                " when padh is a list.")

    if isinstance(padw, list):
        if len(padw) != PAD_SHAPE_DIM:
            err_man.raise_err_specific_user(
                "conv2d", "Dimension must be " + str(PAD_SHAPE_DIM) +
                " when padw is a list.")

    if groups != 1 and groups != shape_in[1]:
        err_man.raise_err_specific_user(
            "conv2d", "groups must equal 1 or cin, but given" + str(groups))

    conv_param = [shape_in, shape_w, padh, padw, strideh, stridew, dilateh, dilatew]
    check_conv2d_support_range(conv_param)

    _, block_size_k_fmap, block_size_n = get_cube_mkn(in_dtype)
    _, block_size_k_weight, _ = get_cube_mkn(w_dtype)

    shape_in[1] = ((shape_in[1] + block_size_k_fmap - 1) // block_size_k_fmap) * block_size_k_fmap
    shape_w[1] = ((shape_in[1] + block_size_k_weight - 1) // block_size_k_weight) * block_size_k_weight
    shape_w[0] = ((shape_w[0] + block_size_n - 1) // block_size_n) * block_size_n

    return shape_in, shape_w


def conv_layer_cce_shape_calc(shape_in,
                              shape_w,
                              in_dtype,
                              w_dtype,
                              optim_dict,
                              cout1_opt=1,
                              c1_opt=1,
                              group_opt=1,
                              c1in_ori_align=1):
    """
    Parameters
    ----------
    shape_in: shape of feature map

    shape_w: shape of weight

    in_dtype: the feature map data type

    w_dtype: the weight data type

    optim_dict: optimize feature dict

    Returns
    -------
    None
    """
    block_size_k = CUBE_MKN[w_dtype]['mac'][1]
    block_size_n = CUBE_MKN[w_dtype]['mac'][2]

    enable_input_4channel = optim_dict.get("enable_input_4channel")
    fmap_shape_nc1hwc0 = (shape_in[0], c1in_ori_align, shape_in[2],
                          shape_in[3],
                          4 if enable_input_4channel else block_size_k)

    out_channel, _, filter_h, filter_w = shape_w
    if optim_dict["c0_optim_flg"]:
        filter_shape_frac_z = ((4 * filter_h * filter_w + block_size_k - 1) //
                               block_size_k, out_channel // block_size_n,
                               block_size_n, block_size_k)
    else:
        filter_shape_frac_z = (group_opt * c1_opt * filter_h * filter_w,
                               cout1_opt, block_size_n, block_size_k)
    return fmap_shape_nc1hwc0, filter_shape_frac_z


def conv_layer_cce_v350_shape_calc(shape_in,
                                   shape_w,
                                   in_dtype,
                                   w_dtype,
                                   cout1,
                                   cin1_data,
                                   cin1_weight,
                                   groups=1):
    """
    Parameters
    ----------
    shape_in: shape of feature map
    shape_w: shape of weight
    in_dtype: the feature map data type
    w_dtype: the weight data type
    cout1: cout of weight
    cin1_data: cin1 value of fmap
    cin1_weight: cin1 value of weight

    Returns
    -------
    None
    """
    _, block_size_k_fmap, block_size_n = get_cube_mkn(in_dtype)
    _, block_size_k_weight, _ = get_cube_mkn(w_dtype)

    fmap_shape_nc1hwc0 = (shape_in[0], cin1_data, shape_in[2],
                          shape_in[3], block_size_k_fmap)

    out_channel, _, filter_h, filter_w = shape_w
    filter_shape_frac_z = (cin1_weight * filter_h * filter_w, cout1, block_size_n, block_size_k_weight)
    # depthwise -> C1HWC0
    filter_shape_c1hwc0 = (cout1, filter_h, filter_w, block_size_n)
    depthwise_conv2d_flag = (groups == out_channel and groups == shape_in[1])
    filter_shape = filter_shape_c1hwc0 if depthwise_conv2d_flag else filter_shape_frac_z
    return fmap_shape_nc1hwc0, filter_shape


def _trans_stride(input_size, kernel, stride, pad, dlt):
    """
    transform stride

    Notice
    ------
    adapt stride value to hardware request

    Parameters
    ----------
    input_size: int
        feature map H/W size
    kernel: int
        kernel H/W size
    pad: 2D list of int
        pad on H/W side
    strides: int
        stride on H/W
    dlt: int
        dilation on H/W
    Returns
    -------
    new stride
    """
    return 1 if input_size + pad[0] + pad[1] == (kernel - 1) * dlt + 1 else stride


def _conv2d_fusion_para(inputs, outputs):
    """
    get lxfusion params for conv2d
    """
    fmap_l1_addr_flag = inputs.get("L1_addr_flag", "nothing")
    fmap_l1_valid_size = inputs.get("L1_valid_size", -1)
    slice_offset = inputs.get("slice_offset", (0, 0, 0, 0, 0))
    fusion_para = {
        "fmap_l1_addr_flag": fmap_l1_addr_flag,
        "fmap_l1_valid_size": fmap_l1_valid_size,
        "slice_offset": slice_offset
    }

    return fusion_para


def _lcm(num1, num2):
    """
    Obtain the least common multiple of num1 and num2
    """
    tmp = num1 * num2
    while num1 % num2 != 0:
        num1, num2 = num2, (num1 % num2)
    return tmp // num2


def _get_minimum_load_l1(shape_fm,
                         shape_filter,
                         params):
    """
    Obtains the minimum amount of data to be loaded to L1.
    """
    weight, strides, pads, dilations, data_format = params[WEIGHT_INDEX], params[STRIDES_INDEX], \
                                                    params[PADS_INDEX], params[DILATIONS_INDEX], \
                                                    params[DATA_FORMAT_INDEX]
    pos_attr_h = data_format.find('H')
    pos_attr_w = data_format.find('W')
    strideh = strides[pos_attr_h]
    stridew = strides[pos_attr_w]
    dilate_h = dilations[pos_attr_h]
    dilate_w = dilations[pos_attr_w]
    if len(pads) == 4:
        pad_top, pad_bottom, pad_left, pad_right = pads
    else:
        err_man.raise_err_should_be_4d("conv2d", "pads shape")
    strideh = _trans_stride(shape_fm[2], shape_filter[2], strideh,
                            [pad_top, pad_bottom], dilate_h)
    stridew = _trans_stride(shape_fm[3], shape_filter[3], stridew,
                            [pad_left, pad_right], dilate_w)
    filter_h_dilation = (shape_filter[2] - 1) * dilate_h + 1
    filter_w_dilation = (shape_filter[3] - 1) * dilate_w + 1
    w_out = (shape_fm[3] +
             (pad_left + pad_right) - filter_w_dilation) // stridew + 1
    h_out_part = _lcm(w_out, 16) // w_out
    h_part_length = (h_out_part - 1) * strideh + filter_h_dilation
    m_bit_ratio = 2
    if weight.get("dtype") == "int8":
        m_bit_ratio = 1
    minimum_load_l1 = m_bit_ratio * 1 * 4 * h_part_length * shape_fm[3]
    return minimum_load_l1


def use_v200_c04_check(shape_fm, shape_filter, params):
    """
    Check whether to use v200 c0=4 optimization
    """
    use_v200_c04_flg = False
    minimum_load_l1 = _get_minimum_load_l1(shape_fm, shape_filter, params)
    if minimum_load_l1 < get_soc_spec("L1_SIZE"):
        use_v200_c04_flg = True
    return use_v200_c04_flg


def get_dynamic_flag(shape_fm: list) -> bool:
    '''
    get dynamic flag
    '''
    dynamic_flag = (shape_fm == (-2,)) or (shape_fm[0] == -1 and -1 not in shape_fm[1:]) or \
        (shape_fm[2] == -1 and shape_fm[3] == -1 and -1 not in shape_fm[:2])
    return dynamic_flag


def check_soc_and_dtype(op_params):
    """
    simply check op support version and input dtype

    Notice
    ------
    data type only check required tensors

    Parameters
    ----------
    op_params: op name and input tensors
        {
            "conv2d_compress": [inputs, weight_compress, compress_index]
        }

    Returns
    -------
    None
    """
    valid = isinstance(op_params, dict) and len(op_params) > 0
    if not valid:
        err_man.raise_err_message_cube(f"input op_params {op_params} is invalid")
    # "soc_version" is required, should be list and "All" means support all version
    # "data_type" is required, should be list and length be equal to input tensors
    support_info = {
        "conv2d_compress": {
            "soc_version": ["Hi3796CV300CS", "SD3403"],
            "data_type": [
                ["int8", "int8", "int8"],
            ],
        },
    }
    version = get_soc_spec("SHORT_SOC_VERSION")
    for op_name, tensor_list in op_params.items():
        if op_name not in support_info:
            err_man.raise_err_message_cube(f"op_name should be in {list(support_info.keys())}, actual is {op_name}")
        # >>> start: soc version support check
        support_soc = support_info.get(op_name).get("soc_version")
        valid = "All" in support_soc or version in support_soc
        if not valid:
            err_man.raise_err_common(op_name, f"only support {support_soc}", version)
        # <<< end: soc version support check

        # >>> start: data type support check
        type_list = []
        # traverse all required tensor and record in order
        for tensor in tensor_list:
            if isinstance(tensor, dict):
                type_list.append(tensor.get("dtype"))
            elif isinstance(tensor, (tvm.Tensor, tvm.var)):
                type_list.append(tensor.dtype)
        support_dtype = support_info[op_name].get("data_type")
        # confirm whether tensor dtype as expected or not
        valid = type_list in support_dtype
        if not valid:
            err_man.raise_err_common(op_name,
                                     f"input tensor dtype combination should be in {list(support_dtype)}",
                                     type_list)
        # <<< end: data type support check


def get_op_support_info_static_common(bias, bias_idx, format_x):
    """
    algorithm: get_op_support_info_static_common
    Notice
    ------
    get the conv2d common static split info
    Parameters
    ----------
    bias: dict with keys(shape and dtype) or None
    input bias tensor
    format_x: the format of fmap
    Returns
    -------
    slice info
    """
    cout_slice_idx = 3
    if format_x == "NHWC":
        # nz2nd and h,w slice can not enable together, which pass not supported
        slice_info = {"_op_slice_info":
                    {"splitMaps": [{"inputList": [{"idx": 0, "axis": [0], "headOverLap": [-1], "tailOverLap": [-1]}],
                                    "outputList": [{"idx": 0, "axis": [0]}]},
                                    {"inputList": [{"idx": 1, "axis": [1], "headOverLap": [-1], "tailOverLap": [-1]}],
                                    "outputList": [{"idx": 0, "axis": [1]}]}],
                    "reduceMaps": [],
                    "l1FusionEnable": 2,
                    "minTbeL1Space": 0}}
        cout_slice_idx = 1
    else:
        slice_info = {"_op_slice_info":
                    {"splitMaps": [{"inputList": [{"idx": 0, "axis": [0], "headOverLap": [-1], "tailOverLap": [-1]}],
                                    "outputList": [{"idx": 0, "axis": [0]}]},
                                    {"inputList": [{"idx": 0, "axis": [2], "headOverLap": [0], "tailOverLap": [0]}],
                                    "outputList": [{"idx": 0, "axis": [2]}]},
                                    {"inputList": [{"idx": 0, "axis": [3], "headOverLap": [0], "tailOverLap": [0]}],
                                    "outputList": [{"idx": 0, "axis": [3]}]},
                                    {"inputList": [{"idx": 1, "axis": [1], "headOverLap": [-1], "tailOverLap": [-1]}],
                                    "outputList": [{"idx": 0, "axis": [1]}]}],
                    "reduceMaps": [],
                    "l1FusionEnable": 2,
                    "minTbeL1Space": 0}}
    if bias:
        bias_input = [{"idx": bias_idx, "axis": [0], "headOverLap": [-1], "tailOverLap": [-1]}]
        slice_info.get('_op_slice_info').get("splitMaps")[cout_slice_idx].get("inputList").extend(bias_input)

    return slice_info


def feature_support_check(params: list, c0_optim_flag: bool) -> list:
    """
    input: params = [inputs, weights, bias, offset_w, outputs, strides, pads,
                     dilations, groups, data_format, offset_x, kernel_name]
           c0_optim_flag
    output:
        nhwc_support_fp16_flag: check is support float16 and float32 nhwc
        nhwc_support_fp32_flag: check is only support float32 nhwc
        is_support_c04_flg: check is support C04. when dma and split_w is false
    ---------------------------------
    judge if support fmap NHWC format input
    1. fmap ori format = NHWC; 2. c0_optim_flag = Flase; 3. support nd->nz instruction;
    4. support load3d; 5. splitw can handle the case when fmap can not load in L1(no DMA)
    """
    inputs, weights, bias, _, _, strides, pads, dilations, _, data_format, _, _ = params
    get_kernel_max_value()
    fm_ori_format = inputs.get("ori_format")
    fm_dtype = inputs.get("dtype")
    filter_ori_format = weights.get("ori_format")
    bias_shape = [] if bias is None else bias.get("ori_shape")
    dynamic_flag = check_dynamic(inputs.get("ori_shape"), weights.get("ori_shape"), bias_shape)
    if dynamic_flag:
        return [False, False, False]
    pos_h = data_format.find('H')
    pos_w = data_format.find('W')
    strideh = strides[pos_h]
    stridew = strides[pos_w]
    dilateh = dilations[pos_h]
    dilatew = dilations[pos_w]
    pad_top, pad_bottom, pad_left, pad_right = pads
    fm_pos_h = fm_ori_format.find('H')
    fm_pos_w = fm_ori_format.find('W')
    h_in = inputs.get("ori_shape")[fm_pos_h]
    w_in = inputs.get("ori_shape")[fm_pos_w]
    filter_pos_h = filter_ori_format.find('H')
    filter_pos_w = filter_ori_format.find('W')
    h_k = weights.get("ori_shape")[filter_pos_h]
    w_k = weights.get("ori_shape")[filter_pos_w]
    hk_dilation = (h_k - 1) * dilateh + 1
    wk_dilation = (w_k - 1) * dilatew + 1
    h_out = (h_in + pad_top + pad_bottom - hk_dilation) // strideh + 1
    w_out = (w_in + pad_left + pad_right - wk_dilation) // stridew + 1
    w_dtype = weights.get("dtype")
    max_flag = strideh > STRIDE_MAX or stridew > STRIDE_MAX or pad_top > PAD_MAX or pad_bottom > PAD_MAX or \
        pad_left > PAD_MAX or pad_right > PAD_MAX or dilateh > DILATE_MAX or dilatew > DILATE_MAX or \
        h_k > FILTER_HW_MAX or w_k > FILTER_HW_MAX
    if max_flag:
        return [False, False, False]
    # cal the max_feature_map_L1 when splitW enable in milan version
    conv_params = [h_in, w_in, w_out, strideh, stridew, hk_dilation, wk_dilation, w_dtype, c0_optim_flag]
    # do C04 check, when split_w unset C04
    if c0_optim_flag:
        if op_util_conv2d.check_l1_size_invalid(conv_params):
            log.warn("conv2d can not enable both c04 & splitw on current soc version.")
            return [False, False, False]
        return [False, False, True]
    #C04 check over, do nz2nd check
    if fm_dtype not in CONV_ND_IN_DTYPE_LIST or fm_ori_format != "NHWC":
        return [False, False, False]
    if not tbe.common.platform.platform_info.intrinsic_check_support("Intrinsic_data_move_out2l1_nd2nz"):
        return [False, False, False]
    if op_util_conv2d.check_splitw_l1_size_invalid(conv_params):
        return [False, False, False]
    if check_load3dv2_postk_params_invalid(h_k, w_k, "float32"):
        return [False, False, False]
    elif check_load3dv2_postk_params_invalid(h_k, w_k, "float16"):
        return [False, True, False]
    return [True, True, False]


def gen_conv2d_base_param(dtype_dict: dict, format_dict: dict, c0_optim_flag: bool, dynamic_flag: bool) -> list:
    """
    gen conv2d base param
    """
    if not c0_optim_flag and dynamic_flag:
        input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                               datatype=dtype_dict["input0"],
                                               format=format_dict["input0"],
                                               unknownshape_format=format_dict["input0"])
        input1 = util_select_op_base.gen_param(classify="input1", name="filter",
                                               datatype=dtype_dict["input1"],
                                               format=format_dict["input1"],
                                               unknownshape_format=format_dict["input1"],
                                               sub_format="0")
        input2 = util_select_op_base.gen_param(classify="input2", name="bias",
                                               datatype=dtype_dict["input2"],
                                               format=format_dict["input2"])
        input3 = util_select_op_base.gen_param(classify="input3", name="offset_w",
                                               datatype=dtype_dict["input3"],
                                               format=format_dict["input3"])
        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                datatype=dtype_dict["output0"],
                                                format=format_dict["output0"],
                                                unknownshape_format=format_dict["output0"])
    else:
        input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                               datatype=dtype_dict["input0"],
                                               format=format_dict["input0"])
        input1 = util_select_op_base.gen_param(classify="input1", name="filter",
                                               datatype=dtype_dict["input1"],
                                               format=format_dict["input1"],
                                               sub_format="0")
        input2 = util_select_op_base.gen_param(classify="input2", name="bias",
                                               datatype=dtype_dict["input2"],
                                               format=format_dict["input2"])
        input3 = util_select_op_base.gen_param(classify="input3", name="offset_w",
                                               datatype=dtype_dict["input3"],
                                               format=format_dict["input3"])
        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                datatype=dtype_dict["output0"],
                                                format=format_dict["output0"])

    return [input0, input1, input2, input3, output0]


def v220_gen_param(inputs: dict, weights: dict, shape_fm: list, c0_optim_flg: bool, flg_list: list) -> list:
    """
    Gen op info in v220 situation.
    """
    # only dynamic_hw or dynamic_batch is supported by dynamic conv2d.
    nhwc_support_fp16_flag, nhwc_support_fp32_flag, is_support_c04_flg = flg_list
    dynamic_flag = get_dynamic_flag(shape_fm)

    if nhwc_support_fp16_flag:
        dtype_dict = {
            "input0": "float16,float16,bfloat16,bfloat16,float32,float16,float16,bfloat16,bfloat16,float32", # fmap
            "input1": "float16,float16,bfloat16,bfloat16,float32,float16,float16,bfloat16,bfloat16,float32", # weight
            "input2": "float16,float32,float32,float32,float32,float16,float32,float32,float32,float32", # bias
            "input3": "int8,int8,int8,int8,int8,int8,int8,int8,int8,int8", # offset_w
            "output0": "float16,float32,bfloat16,float32,float32,float16,float32,bfloat16,float32,float32", # y
        }
        format_dict = {
            "input0": "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NHWC,NHWC,NHWC,NHWC,NHWC", # fmap
            "input1": "FRACTAL_Z,FRACTAL_Z,FRACTAL_Z,FRACTAL_Z,FRACTAL_Z,FRACTAL_Z,"
                      "FRACTAL_Z,FRACTAL_Z,FRACTAL_Z,FRACTAL_Z", # weight
            "input2": "ND,ND,ND,ND,ND,ND,ND,ND,ND,ND", # bias
            "input3": "ND,ND,ND,ND,ND,ND,ND,ND,ND,ND", # offset_w
            "output0": "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0", # y
        }
    elif nhwc_support_fp32_flag:
        dtype_dict = {
            "input0": "float16,float16,bfloat16,bfloat16,float32,float32", # fmap
            "input1": "float16,float16,bfloat16,bfloat16,float32,float32", # weight
            "input2": "float16,float16,float32,float16,float32,float32", # bias
            "input3": "int8,int8,int8,int8,int8,int8", # offset_w
            "output0": "float16,float32,bfloat16,float32,float32,float32", # y
        }
        format_dict = {
            "input0": "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NHWC", # fmap
            "input1": "FRACTAL_Z,FRACTAL_Z,FRACTAL_Z,FRACTAL_Z,FRACTAL_Z,FRACTAL_Z", # weight
            "input2": "ND,ND,ND,ND,ND,ND", # bias
            "input3": "ND,ND,ND,ND,ND,ND", # offset_w
            "output0": "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0", # y
        }
    elif c0_optim_flg and inputs.get("is_first_layer"):
        # first layer c04 (only set when aipp + conv2d).
        if is_support_c04_flg:
            dtype_dict = {
                "input0": "float16,float16,int8", # fmap
                "input1": "float16,float16,int8", # weight
                "input2": "float16,float32,int32", # bias
                "input3": "int8,int8,int8", # offset_w
                "output0": "float16,float32,int32", # y
            }
            format_dict = {
                "input0": "NC1HWC0_C04,NC1HWC0_C04,NC1HWC0_C04", # fmap
                "input1": "FRACTAL_Z_C04,FRACTAL_Z_C04,FRACTAL_Z_C04", # weight
                "input2": "ND,ND,ND", # bias
                "input3": "ND,ND,ND", # offset_w
                "output0": "NC1HWC0,NC1HWC0,NC1HWC0", # y
            }
        else:
            # when dma or split_w is not support C04
            dtype_dict = {
                "input0": "float16,float16,int8", # fmap
                "input1": "float16,float16,int8", # weight
                "input2": "float16,float32,int32", # bias
                "input3": "int8,int8,int8", # offset_w
                "output0": "float16,float32,int32", # y
            }
            format_dict = {
                "input0": "NC1HWC0,NC1HWC0,NC1HWC0", # fmap
                "input1": "FRACTAL_Z,FRACTAL_Z,FRACTAL_Z", # weight
                "input2": "ND,ND,ND", # bias
                "input3": "ND,ND,ND", # offset_w
                "output0": "NC1HWC0,NC1HWC0,NC1HWC0", # y
            }
    elif c0_optim_flg and not inputs.get("is_first_layer") and is_support_c04_flg:
        # not aipp c04
        # NC1HWC0 + FRACTAL_Z_C04 for enablle_small_channel.
        # NC1HWC0 + FRACTAL_Z for not enablle_small_channel.
        dtype_dict = {
            "input0": "float16,float16,float16,float16,bfloat16,bfloat16,bfloat16,bfloat16,"
                        "float32,int8,int8", # fmap
            "input1": "float16,float16,float16,float16,bfloat16,bfloat16,bfloat16,bfloat16,"
                        "float32,int8,int8", # weight
            "input2": "float16,float16,float32,float32,float32,float32,float32,float32,"
                        "float32,int32,int32", # bias
            "input3": "int8,int8,int8,int8,int8,int8,int8,int8,int8,int8,int8", # offset_w
            "output0": "float16,float16,float32,float32,bfloat16,bfloat16,float32,float32,"
                        "float32,int32,int32", # y
        }
        format_dict = {
            "input0": "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,"
                        "NC1HWC0,NC1HWC0,NC1HWC0", # fmap
            "input1": "FRACTAL_Z_C04,FRACTAL_Z,FRACTAL_Z_C04,FRACTAL_Z,"
                        "FRACTAL_Z_C04,FRACTAL_Z,FRACTAL_Z_C04,FRACTAL_Z,"
                        "FRACTAL_Z,FRACTAL_Z_C04,FRACTAL_Z", # weight
            "input2": "ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND", # bias
            "input3": "ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND", # offset_w
            "output0": "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,"
                        "NC1HWC0,NC1HWC0,NC1HWC0", # y
        }
    elif (shape_fm == (-2,)) or (DYNAMIC_VALUE in shape_fm):
        dtype_dict = {
            "input0": "float16,float32,bfloat16",  # fmap
            "input1": "float16,float32,bfloat16",  # weight
            "input2": "float16,float32,float32",  # bias
            "input3": "int8,int8,int8",  # offset_w
            "output0": "float16,float32,bfloat16",  # y
        }
        format_dict = {
            "input0": "NC1HWC0,NC1HWC0,NC1HWC0", # fmap
            "input1": "FRACTAL_Z,FRACTAL_Z,FRACTAL_Z",  # weight
            "input2": "ND,ND,ND",  # bias
            "input3": "ND,ND,ND",  # offset_w
            "output0": "NC1HWC0,NC1HWC0,NC1HWC0",  # y
        }
    else:
        dtype_dict = {
            "input0": "float16,float16,bfloat16,bfloat16,float32,int8", # fmap
            "input1": "float16,float16,bfloat16,bfloat16,float32,int8", # weight
            "input2": "float16,float32,float32,float32,float32,int32", # bias
            "input3": "int8,int8,int8,int8,int8,int8", # offset_w
            "output0": "float16,float32,bfloat16,float32,float32,int32", # y
        }
        format_dict = {
            "input0": "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0", # fmap
            "input1": "FRACTAL_Z,FRACTAL_Z,FRACTAL_Z,FRACTAL_Z,FRACTAL_Z,FRACTAL_Z", # weight
            "input2": "ND,ND,ND,ND,ND,ND", # bias
            "input3": "ND,ND,ND,ND,ND,ND", # offset_w
            "output0": "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0", # y
        }

    return gen_conv2d_base_param(dtype_dict, format_dict, c0_optim_flg, dynamic_flag)


def v310_gen_param(inputs: dict, weights: dict, shape_fm: list, c0_optim_flag: bool, flg_list: list) -> list:
    """
    Gen op info in v310 situation.
    """
    _, _, is_support_c04_flg, winograd_conv_flag = flg_list
    # only dynamic_hw or dynamic_batch is supported by dynamic conv2d.
    dynamic_flag = get_dynamic_flag(shape_fm)
    if winograd_conv_flag:
        dtype_dict = {
            "input0": "int8", # fmap
            "input1": "int8", # weight
            "input2": "int32", # bias
            "input3": "int8", # offset_w
            "output0": "int32", # y
        }
        format_dict = {
            "input0": "NC1HWC0", # fmap
            "input1": "FRACTAL_Z_WINO", # weight
            "input2": "ND", # bias
            "input3": "ND", # offset_w
            "output0": "NC1HWC0", # y
        }
    elif c0_optim_flag and inputs.get("is_first_layer") and is_support_c04_flg:
        # first layer c04 (only set when aipp + conv2d).
        dtype_dict = {
            "input0": "float16,int8", # fmap
            "input1": "float16,int8", # weight
            "input2": "float16,int32", # bias
            "input3": "int8,int8", # offset_w
            "output0": "float16,int32", # y
        }
        format_dict = {
            "input0": "NC1HWC0_C04,NC1HWC0_C04", # fmap
            "input1": "FRACTAL_Z_C04,FRACTAL_Z_C04", # weight
            "input2": "ND,ND", # bias
            "input3": "ND,ND", # offset_w
            "output0": "NC1HWC0,NC1HWC0", # y
        }
    elif c0_optim_flag and not inputs.get("is_first_layer") and is_support_c04_flg:
        # not aipp c04
        # NC1HWC0 + FRACTAL_Z_C04 for enable_small_channel.
        # NC1HWC0 + FRACTAL_Z for not enable_small_channel.
        dtype_dict = {
            "input0": "float16,float16,int8,int8", # fmap
            "input1": "float16,float16,int8,int8", # weight
            "input2": "float16,float16,int32,int32", # bias
            "input3": "int8,int8,int8,int8", # offset_w
            "output0": "float16,float16,int32,int32", # y
        }
        format_dict = {
            "input0": "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0", # fmap
            "input1": "FRACTAL_Z_C04,FRACTAL_Z,FRACTAL_Z_C04,FRACTAL_Z", # weight
            "input2": "ND,ND,ND,ND", # bias
            "input3": "ND,ND,ND,ND", # offset_w
            "output0": "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0", # y
        }
    else:
        dtype_dict = {
            "input0": "float16,int8", # fmap
            "input1": "float16,int8", # weight
            "input2": "float16,int32", # bias
            "input3": "int8,int8", # offset_w
            "output0": "float16,int32", # y
        }
        format_dict = {
            "input0": "NC1HWC0,NC1HWC0", # fmap
            "input1": "FRACTAL_Z,FRACTAL_Z", # weight
            "input2": "ND,ND", # bias
            "input3": "ND,ND", # offset_w
            "output0": "NC1HWC0,NC1HWC0", # y
        }

    return gen_conv2d_base_param(dtype_dict, format_dict, c0_optim_flag, dynamic_flag)


def is_force_fp32(input_type, weight_type, output_type):
    """
    check fp16/bf16 inputs and fp32 outputs
    """
    if (input_type, weight_type) in (("float16", "float16"), ("bfloat16", "bfloat16")) and output_type == "float32":
        return True
    return False


def check_valid_precision_mode(input_type, weight_type, output_type, impl_mode):
    if input_type == "float32" and weight_type == "float32" and output_type == "float32":
        if impl_mode not in ("enable_hi_float_32_execution", "enable_float_32_execution", ""):
            err_man.raise_err_specific("conv2d",
                "Illegal precision mode {} is captured.".format(impl_mode))


def get_op_precision_mode(op_type):
    """
    Get op precision mode from op_info base on op_type
    default mode is ""
    conv2d only support "high_performance"/"" by now
    """
    context = get_context()
    if context is None:
        return ""
    op_infos = context.get_op_info()
    if not op_infos:
        return ""
    for op_info in op_infos:
        if op_info.op_type == op_type:
            log.debug("{}'s precision mode received from op_info is {}".format(op_type, op_info.precision_mode))
        if op_info.op_type == op_type and \
            op_info.precision_mode in ("enable_hi_float_32_execution", "enable_float_32_execution", ""):
            return op_info.precision_mode
    return ""


def get_op_enlarge():
    """
    The channel merge requirement uses graph fusion to calculate
    new_enlarge parameter to replace the previous enlarge parameter.
    This function attempts to obtain the new_enlarge parameter.
    If new_enlarge parameter is obtained, the previous enlarge parameter cannot be used.
    """
    context = get_context()
    if context is None:
        return ""
    op_infos = context.get_op_info()
    if not op_infos:
        return ""
    for op_info in op_infos:
        if "new_enlarge" in op_info.extra_params.keys():
            log.info("enlarge received from extra params is {},"
                     "will be used".format(op_info.extra_params.get("new_enlarge")))
            return op_info.extra_params.get("new_enlarge")
    return ""


def set_dummy_placeholder():
    """
    set dummy placeholder to False when conv2d fusion compute is called.
    """
    config_current = build_module.current_build_config()
    config_current.set_attr("dummy_placeholder", False)


def search_op(res, op_tag):
    """
    Search certain op according to op_tag.
    """
    tensor_queue = deque()
    tensor_queue.append(res)
    while tensor_queue:
        src_tensor = tensor_queue.popleft()
        tag = src_tensor.op.tag

        if tag in ("convolution_c_col", "convolution_c_col_bias"):
            break

        if op_tag == tag:
            return src_tensor

        if src_tensor.op.input_tensors:
            append_list = list(i for i in src_tensor.op.input_tensors)
            append_list.reverse()
            tensor_queue.extend(append_list)
    return None


def clear_suffix(name):
    """
    Clear the number suffix of op name or tag.
    """
    if not isinstance(name, str):
        err_man.raise_err_specific("conv2d", f"clear_suffix input is {type(name)}, which should be str type")

    name_str_list = name.split("_")

    if not name_str_list:
        return name
    if name_str_list and not name_str_list[-1].isdigit():
        return name
    return "_".join(name_str_list[:-1])


def check_conv2d_support_range(conv_params):
    """
    check shape and attr range for V350, throw exception when out of range
    | Name          | Field   | Scope       |
    | :------------:| :------:| :----------:|
    | Input size    | H       | [1, 4096]   |
    |               | W       | [1, 4096]   |
    | Filter size   | H       | [1, 511]    |
    |               | W       | [1, 511]    |
    | Stride        | H       | [1, 63]     |
    |               | W       | [1, 63]     |
    | Padding       | Top     | [0, 254]    |
    |               | Bottom  | [0, 254]    |
    |               | Left    | [0, 254]    |
    |               | Right   | [0, 254]    |
    | Dilation      | H       | [1, 63]     |
    |               | W       | [1, 63]     |
    Input: list of [fmap_in, kernel_in, pad_h, pad_w, stride_h, stride_w, dilate_h, dilate_w]
    """
    # get shape value
    fmap_in, kernel_in, pad_h, pad_w, stride_h, stride_w, dilate_h, dilate_w = conv_params
    fmap_hw_value = [fmap_in[INDEX_H], fmap_in[INDEX_W]]
    kernel_hw_value = [kernel_in[INDEX_H], kernel_in[INDEX_W]]
    stride_hw_value = [stride_h, stride_w]
    dilate_hw_value = [dilate_h, dilate_w]
    pad_value = [pad_h[0], pad_h[1], pad_w[0], pad_w[1]]

    # check support range of each dim of conv2d
    check_range_illegal(fmap_hw_value, FMAP_HW_RANGE_MIN, FMAP_HW_RANGE_MAX, "famp H/W")
    check_range_illegal(kernel_hw_value, KERNEL_HW_RANGE_MIN, KERNEL_HW_RANGE_MAX, "kernel H/W")
    check_range_illegal(stride_hw_value, STRIDE_HW_RANGE_MIN, STRIDE_HW_RANGE_MAX, "stride H/W")
    check_range_illegal(dilate_hw_value, DILATE_HW_RANGE_MIN, DILATE_HW_RANGE_MAX, "dilate H/W")
    check_range_illegal(pad_value, PAD_RANGE_MIN, PAD_RANGE_MAX, "pad")
    check_range_illegal(fmap_in[INDEX_C], CIN_RANGE_MIN, CIN_RANGE_MAX, "cin")
    check_range_illegal(kernel_in[INDEX_N], COUT_RANGE_MIN, COUT_RANGE_MAX, "cout")
