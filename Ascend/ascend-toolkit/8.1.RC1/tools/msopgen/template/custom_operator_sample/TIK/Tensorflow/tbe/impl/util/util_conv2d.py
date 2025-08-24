#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

conv2d common
provide common function used by conv2d
"""

import math
import json
from tbe import tvm
from tbe.common.utils import para_check
from tbe.common.platform.platform_info import get_soc_spec
from tbe.common.platform import CUBE_MKN
from tbe.common.utils.errormgr import error_manager_cube as err_man

PAD_SHAPE_DIM = 2
# fmapH, fmapW must be in [1,4096]
FMAP_HW_MIN = 1
FMAP_W_MAX = 2**32 - 1
FMAP_H_MAX = 100000
DMA_HW_MAX = 2**32 - 1

FMAP_W_MIN_SPLIT_W = 1
FMAP_W_MAX_SPLIT_W = 4294967295

# filterH, filterW must be in [1,255]
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


def is_support_v220():
    """
    Check if a100 version.

    Returns
    -------
    True: a100 version.
    False: other version.
    """
    soc_version = get_soc_spec("SHORT_SOC_VERSION")
    if soc_version == "a100":
        return True
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
    if len(inputs.shape) == 5:
        for i in inputs.shape:
            shape_fm.append(i.value)
    elif len(inputs.shape) == 4:
        if inputs.op.attrs['current_shape']:
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

    format_w = weights.op.attrs['ori_format'].value
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
        from te.tvm.buffer_manager import get_buffer_manager
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
        fmap_l1_addr_flag = inputs.op.attrs[
            "L1_addr_flag"].value if "L1_addr_flag" in inputs.op.attrs else "nothing"
        fmap_l1_valid_size = inputs.op.attrs[
            "L1_valid_size"].value if "L1_valid_size" in inputs.op.attrs else -1
        slice_offset = inputs.op.attrs[
            "slice_offset"] if "slice_offset" in inputs.op.attrs else (0, 0, 0,
                                                                       0, 0)
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

    is_first_layer = bool(inputs.op.attrs["is_first_layer"].value
                          ) if "is_first_layer" in inputs.op.attrs else False
    is_input4channel = inputs.shape[-1].value == 4

    c0_optim_flg = False
    use_v200_c04_flg = False
    v220_c04_mode = "disabled"
    if shape_c <= 4 and ("format" in weights.op.attrs and
                         weights.op.attrs['format'].value == "FRACTAL_Z_C04"):
        c0_optim_flg = True

        if is_support_v220():
            v220_c04_mode = "first_layer_c04" if is_first_layer else "not_first_layer_c04"
            if (is_first_layer
                    and not is_input4channel) or (is_input4channel
                                                  and not is_first_layer):
                err_man.raise_err_specific_user(
                    "conv2d",
                    "input 4 channel is only accepted in first layer when v220 c0=4!"
                )
        else:
            if (weight_h == 1) and (weight_w == 1):
                err_man.raise_err_specific_user(
                    "conv2d",
                    "weight shape does not support that H and W are both equal to 1 when C0=4."
                )

            if inputs.shape[-1].value == 4 and is_support_v200():
                use_v200_c04_flg = True

    optim_dict = {
        "c0_optim_flg": c0_optim_flg,
        "use_v200_c04_flg": use_v200_c04_flg,
        "v220_c04_mode": v220_c04_mode,
        "invalid_data_rm": False
    }

    if options is not None:
        optim_dict.update(options)

    return para_dict, optim_dict


def calc_para_from_dict(inputs,
                        weights,
                        strides,
                        pads,
                        dilations,
                        outputs,
                        data_format="NCHW"):
    shape_x = inputs.get("ori_shape")
    shape_x_5hd = inputs.get("shape")
    shape_w = weights.get("ori_shape")

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
    # only c is original value when lxfusion split batch and h.
    shape_fm = [
        shape_x_5hd[0], shape_x[pos_c], shape_x_5hd[2], shape_x_5hd[3]
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

    is_first_layer = inputs.get("is_first_layer", False)
    c0_optim_flg = False
    use_v200_c04_flg = False
    v220_c04_mode = "disabled"

    if shape_w[pos_c] <= 4 and weights.get("format") == "FRACTAL_Z_C04":
        c0_optim_flg = True

        if is_support_v220():
            v220_c04_mode = "first_layer_c04" if is_first_layer else "not_first_layer_c04"
        else:
            if (shape_w[pos_h] == 1) and (shape_w[pos_w] == 1):
                err_man.raise_err_specific_user(
                    "conv2d",
                    "weight shape does not support that H and W are both equal to 1 when C0=4."
                )
            if inputs.get("format") == "NC1HWC0_C04" and is_support_v200():
                use_v200_c04_flg = True

    optim_dict = {
        "c0_optim_flg": c0_optim_flg,
        "use_v200_c04_flg": use_v200_c04_flg,
        "v220_c04_mode": v220_c04_mode
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
            "c0_optim_flg": False,
            "use_v200_c04_flg": False,
            "v220_c04_mode": "disabled"
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

    enable_input_4channel = optim_dict.get("use_v200_c04_flg") or optim_dict.get(
        "v220_c04_mode") == "first_layer_c04"
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
    return 1 if input_size + pad[0] + pad[1] == (kernel -
                                                 1) * dlt + 1 else stride


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


def _get_minimum_load_L1(shape_fm,
                         shape_filter,
                         strides,
                         pads,
                         dilations,
                         data_format="NCHW"):
    """
    Obtains the minimum amount of data to be loaded to L1.
    """
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
    minimum_load_L1 = 2 * 1 * 4 * h_part_length * shape_fm[3]
    return minimum_load_L1


def use_v200_c04_check(shape_fm, shape_filter, params):
    """
    Check whether to use v200 c0=4 optimization
    """
    use_v200_c04_flg = False
    strides, pads, dilations, data_format = params[5], params[6], params[
        7], params[9]
    minimum_load_L1 = _get_minimum_load_L1(shape_fm, shape_filter, strides,
                                           pads, dilations, data_format)
    if minimum_load_L1 < get_soc_spec("L1_SIZE"):
        use_v200_c04_flg = True
    return use_v200_c04_flg

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
        support_soc = support_info[op_name]["soc_version"]
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
        support_dtype = support_info[op_name]["data_type"]
        # confirm whether tensor dtype as expected or not
        valid = type_list in support_dtype
        if not valid:
            err_man.raise_err_common(op_name,
                                     f"input tensor dtype combination should be in {list(support_dtype)}",
                                     type_list)
        # <<< end: data type support check


def get_op_support_info_static_common(bias, bias_idx):
    """
    algorithm: get_op_support_info_static_common
    Notice
    ------
    get the conv2d common static split info
    Parameters
    ----------
    bias: dict with keys(shape and dtype) or None
    input bias tensor
    Returns
    -------
    slice info
    """
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
        slice_info['_op_slice_info']["splitMaps"][3]["inputList"].extend(bias_input)

    return slice_info
