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
conv2d_compress
"""
from __future__ import absolute_import
import math
import json
from operator import index
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_cube as err_man_cube
from impl.util.platform_adapter import conv_compress
from impl.util.platform_adapter import WEIGHT_SPARSE_4_2
from impl.util.platform_adapter import WEIGHT_UNZIP
from impl.util.platform_adapter import COMPRESS_ALG_SUPPORT
from impl.util.platform_adapter import CUBE_MKN_IDX_K
from impl.util.platform_adapter import CUBE_MKN_IDX_N
from impl.util import util_conv2d
from tbe.common.utils.op_util import op_util_conv2d


MAX_FITLER_HW = 1024
FZ_SHAPE_DIM = 4
SMALL_CHANNEL_4 = 4


def get_tensor_dtype(tensor):
    """
    get dtype from tensor or dict
    """
    if isinstance(tensor, dict):
        return tensor.get("dtype")
    elif isinstance(tensor, (tvm.Tensor, tvm.var)):
        return tensor.dtype
    else:
        err_man_cube.raise_err_message_cube("input should be dict or tensor.")
        return ""


def get_tensor_shape(tensor):
    """
    get shape from tensor or dict
    """
    if isinstance(tensor, dict):
        return tensor.get("shape")
    elif isinstance(tensor, (tvm.Tensor, tvm.var)):
        return tensor.shape
    else:
        err_man_cube.raise_err_message_cube("input should be dict or tensor.")
        return []


def check_supported(inputs, weight_compress, compress_index, bias, offset_w, outputs, strides, pads, dilations,
                    groups=1, data_format='NHWC', offset_x=0, alg=WEIGHT_UNZIP, kernel_name="conv2dcompress"):
    if groups != 1:
        return False, "groups must be 1"
    pos_c = inputs.get("ori_format").find('C')
    cin_ori = inputs.get("ori_shape")[pos_c]
    if cin_ori <= SMALL_CHANNEL_4:
        return False, "c04 not support"
    wino_params = [inputs, weight_compress, bias, offset_w, outputs, strides, pads,
                   dilations, groups, data_format, offset_x, kernel_name]
    winograd_conv_flag = op_util_conv2d.get_winograd_conv_flag(wino_params)
    if winograd_conv_flag:
        return False, "winograd conv supported"
    return True, ""


def conv2dcompress_support_check(inputs, weight_compress, compress_index, groups, alg):
    """
    compress support check
    """
    if groups != 1:
        err_man_cube.raise_err_three_paras("E62305", "conv2d_compress",
                                           "groups", "1", "groups={}".format(groups))

    if alg not in COMPRESS_ALG_SUPPORT:
        err_man_cube.raise_err_message_cube(
            "only algorithm [{}] are supported, current is {}".format(COMPRESS_ALG_SUPPORT, alg))

    if alg == WEIGHT_UNZIP:
        util_conv2d.check_soc_and_dtype(
            {"conv2d_compress": [inputs, weight_compress, compress_index]})

    if alg == WEIGHT_SPARSE_4_2:
        if not util_conv2d.is_support_fixpipe():
            err_man_cube.raise_err_message_cube("only v200 and v300 soc support {} alg".format(WEIGHT_SPARSE_4_2))

        if get_tensor_dtype(inputs) != "int8" or get_tensor_dtype(weight_compress) != "int8" or \
                get_tensor_dtype(compress_index) != "int8":
            err_man_cube.raise_err_message_cube("input tesnor dtype should be int8.")

        if len(get_tensor_shape(compress_index)) != FZ_SHAPE_DIM:
            err_man_cube.raise_err_message_cube("index tensor shape should be 4 dim.")


@register_operator_compute("conv2dcompress", op_mode="static", support_fusion=True)
def conv2dcompress_compute(inputs, weight_compress, compress_index, bias, offset_w, outputs,
                           strides, pads, dilations, groups=1, data_format='NHWC', offset_x=0,
                           alg=WEIGHT_UNZIP, kernel_name="conv2dcompress", options=None):
    """
    conv2dcompress compute

    Notice
    ------
    only used by framework combine with IR

    Parameters
    ----------
    inputs: tvm placeholder
        input 5hd feature map tensor
    weight_compress: tvm placeholder
        input frac_z compress weight tensor
    compress_index: tvm placeholder
        input ND compress index
    outputs: tvm placeholder
        output tensor, dtype must be assigned
    bias: tvm placeholder or None
        input 1d bias tensor
    offset_w: tvm placeholder or None
        offset_w bias tensor
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
        offset for fmap

    Returns
    -------
    tvm compute
    """
    conv2dcompress_support_check(inputs, weight_compress, compress_index, groups, alg)

    para_dict, optim_dict = util_conv2d.calc_para_from_tensor(
        inputs, weight_compress, bias, offset_w, strides, pads, dilations,
        offset_x, groups, kernel_name, data_format, options)

    if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ("Hi3796CV300ES") and \
            para_dict["filter_h"] * para_dict["filter_w"] > MAX_FITLER_HW:
        err_man_cube.raise_err_specific("conv2dcompress",
                                        "conv2d Min tiling still exceed ub buffer, when open weight unzip")
    para_dict["alg"] = alg
    para_dict["compress_index"] = compress_index
    compress_index_shape = compress_index.shape[0]
    res = conv_compress(inputs, weight_compress, compress_index,
                            compress_index_shape, para_dict, optim_dict)

    return res


@para_check.check_input_type(dict, dict, dict, (dict, para_check.NONE_TYPE), (dict, para_check.NONE_TYPE), dict,
                  (tuple, list), (tuple, list), (tuple, list), int,
                  str, int, str, str)
def conv2dcompress(inputs, weight_compress, compress_index, bias, offset_w, outputs, strides, pads, dilations,
                   groups=1, data_format='NHWC', offset_x=0, alg=WEIGHT_UNZIP, kernel_name="conv2dcompress"):
    """
    algorithm: conv2dcompress

    Notice
    ------
    only used by framework combine with IR

    Parameters
    ----------
    inputs: dict with keys(shape and dtype)
        input 4d feature map tensor
    weight_compress: dict with keys(shape and dtype)
        input 4d weight tensor
    compress_index: dict with keys(shape and dtype)
        input ND compress index tensor
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
        kernel name, default value is "conv2dcompress"

    Returns
    -------
    None
    """
    conv2dcompress_support_check(inputs, weight_compress, compress_index, groups, alg)

    in_dtype = inputs.get("dtype")
    w_dtype = weight_compress.get("dtype")
    res_dtype = outputs.get("dtype")
    shape_index = compress_index.get("ori_shape")
    index_dtype = compress_index.get("dtype")

    shape_fm, shape_filter, padh, padw, strideh, stridew, \
    dlt_h, dlt_w, optim_dict, fusion_para = util_conv2d.calc_para_from_dict(inputs, \
        weight_compress, strides, pads, dilations, groups, outputs, data_format)

    use_bias = True
    if bias is None:
        use_bias = False
    use_offset_w = True
    if offset_w is None:
        use_offset_w = False

    _conv_layer_compress_cce(shape_fm, shape_filter, shape_index, in_dtype,
        w_dtype, index_dtype, res_dtype,
        padh, padw, strideh, stridew, dlt_h, dlt_w,
        offset_x, groups=groups, offset_w=use_offset_w,
        bias=use_bias, optim_dict=optim_dict,
        fusion_para=fusion_para,
        kernel_name=kernel_name, need_build=True,
        need_print=False, alg=alg)


def get_sparse_weight_fz_shape(cout1_opt, c1_opt, group_opt, shape_w, w_dtype):
    """
    calculate sparse 4 to 2 weight fz shape
    """
    block_size_k = tbe_platform.CUBE_MKN[w_dtype]['mac'][CUBE_MKN_IDX_K]
    block_size_n = tbe_platform.CUBE_MKN[w_dtype]['mac'][CUBE_MKN_IDX_N]
    _, _, filter_h, filter_w = shape_w
    # 2: sparse c1_opt is ceil of origin c1_opt/2
    filter_shape_frac_z_sparse = (group_opt * ((c1_opt + 1) // 2) * filter_h * filter_w,
                                  cout1_opt, block_size_n, block_size_k)
    return filter_shape_frac_z_sparse


def get_group_opt_params(shape_in, shape_w, w_dtype, groups):
    """
    calculate group opt params
    """
    c0_val = 16
    if w_dtype == "int8":
        c0_val = 32

    cin_ori = shape_in[1] // groups
    cout_ori = shape_w[0] // groups

    enlarge = min(util_conv2d.lcm(util_conv2d.lcm(cin_ori, c0_val) // cin_ori,
        util_conv2d.lcm(cout_ori, 16) // cout_ori), groups)
    c1_opt = math.ceil(cin_ori * enlarge / c0_val)
    cout1_opt = math.ceil(cout_ori * enlarge / 16)
    group_opt = math.ceil(groups / enlarge)
    c1in_ori_align = math.ceil(cin_ori * groups / c0_val)

    return c1_opt, cout1_opt, group_opt, enlarge, c1in_ori_align


def get_fm_weight_shape(shape_in, shape_w, in_dtype, w_dtype, groups, optim_dict, alg):
    """
    get sparse 4to2 weight fz shape
    """
    c1_opt, cout1_opt, group_opt, _, c1in_ori_align = get_group_opt_params(shape_in, shape_w,
                                                                           w_dtype, groups)

    fmap_shape_nc1hwc0, filter_shape_frac_z = util_conv2d.conv_layer_cce_shape_calc(
        shape_in, shape_w, in_dtype, w_dtype, optim_dict, cout1_opt, c1_opt, group_opt, c1in_ori_align)

    if alg == WEIGHT_SPARSE_4_2:
        filter_shape_frac_z = get_sparse_weight_fz_shape(cout1_opt, c1_opt, group_opt, shape_w, w_dtype)

    return fmap_shape_nc1hwc0, filter_shape_frac_z


def create_compress_index_tensor(filter_shape_frac_z, index_dtype, alg):
    """
    create compress index tensor
    """
    if alg == WEIGHT_UNZIP:
        compress_index_shape = tvm.var("compress_index_shape", dtype="int32")
        compress_index = tvm.placeholder((compress_index_shape,), name='compress_index', dtype=index_dtype)
    else:
        # WEIGHT_SPARSE_4_2, 4: weight's shape is four times weight_index
        compress_index_shape = filter_shape_frac_z[:-1] + (filter_shape_frac_z[-1] // 4,)
        compress_index = tvm.placeholder(compress_index_shape, name='compress_index', dtype=index_dtype)
    return compress_index, compress_index_shape


@para_check.check_input_type((list, tuple), (list, tuple), (list, tuple), str, str,
                  str, str, (list, int), (list, int), int, int,
                  (int, para_check.NONE_TYPE), (int, para_check.NONE_TYPE), int, int, str, bool, bool,
                  dict, (dict, para_check.NONE_TYPE), str, bool, bool)
def _conv_layer_compress_cce(shape_in, shape_w, shape_index, in_dtype,
                             w_dtype, index_dtype, res_dtype, padh, padw,
                             strideh, stridew, dilateh=1, dilatew=1,
                             offset_x=0, groups=1, offset_w_dtype='int32',
                             offset_w=False, bias=False, optim_dict=None,
                             fusion_para=None, kernel_name="cce_conv",
                             need_build=False, need_print=False,
                             alg=WEIGHT_UNZIP):
    """
    Parameters
    ----------
    shape_in: shape of feature map

    shape_w: shape of compress weight

    shape_index: shape of compress index

    in_dtype: the feature map data type

    w_dtype: the compress weight data type

    index_dtype: the index data type

    res_dtype: the result data type

    padh: H direction padding

    padw: W direction padding

    strideh: H direction stride

    stridew: W direction stride

    dilateh: H direction spacing between kernel

    dilatew: W direction spacing between kernel

    offset_x: the offset for fmap

    offset_w_dtype: weight offset data type, default 'int32'

    offset_w: the tag for offset_w or not

    bias: the tag for bias or not

    fusion_para: the config for L2 Fusion
                input_memory_type: feature map from L2/GM, 0 for GM, 2 for L2
                output_memory_type: calculation results are outputs to L2/GM
                valid_shape: valid shape in L1 buffer, NC1HWC0
                slice_offset: the offset of each dimension
                              between valid shape and shape in
    kernel_name: cce kernel name, default value is "cce_conv"

    need_build: if need to build CCEC kernel, default value is False

    need_print: if need to print the ir, default value is False

    Returns
    -------
    wrapped_tensor

    """
    # for pylint, otherwise "Dangerous default value [] as argument"
    if optim_dict is None:
        optim_dict = {"c0_optim_flg": False, "input_nd_flag": False}

    if fusion_para is None:
        fusion_para = {"fmap_l1_addr_flag": 0, "fmap_l1_valid_size": -1, "slice_offset": (0, 0, 0, 0, 0)}
    in_dtype = in_dtype.lower()
    w_dtype = w_dtype.lower()
    index_dtype = index_dtype.lower()
    res_dtype = res_dtype.lower()
    offset_w_dtype = offset_w_dtype.lower()

    mad_dtype = 'float32'
    if w_dtype == 'int8':
        mad_dtype = 'int32'

    shape_in = list(shape_in)
    shape_w = list(shape_w)
    fmap_shape_nhwc = [shape_in[0], shape_in[2], shape_in[3], shape_in[1]]
    weight_ori_shape_nchw = shape_w

    shape_in, shape_w = util_conv2d.conv_layer_cce_para_check(
        shape_in, shape_w, padh, padw, strideh, stridew, in_dtype, w_dtype, res_dtype,
        offset_w_dtype, bias, kernel_name, dilateh, dilatew, optim_dict, groups)

    out_channel, _, filter_h, filter_w = shape_w
    if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ("Hi3796CV300ES") and \
            filter_h * filter_w > MAX_FITLER_HW:
        err_man_cube.raise_err_specific("conv2dcompress",
                                        "conv2d Min tiling still exceed ub buffer, "
                                        "when open weight unzip")
    fmap_shape_nc1hwc0, filter_shape_frac_z = get_fm_weight_shape(shape_in, shape_w, in_dtype,
                                                                  w_dtype, groups, optim_dict, alg)
    tensor_list = []
    fmap_format = "NHWC" if optim_dict.get("input_nd_flag") else "NC1HWC0"
    fmap_shape_input = fmap_shape_nhwc if optim_dict.get("input_nd_flag") else fmap_shape_nc1hwc0
    with tvm.target.cce():
        data = tvm.placeholder(fmap_shape_input, name='Fmap', dtype=in_dtype,
                               attrs={"format": fmap_format, "current_shape": fmap_shape_input})
        tensor_list.append(data)
        weight = tvm.placeholder(filter_shape_frac_z, name='Filter', dtype=w_dtype)
        tensor_list.append(weight)

        compress_index, compress_index_shape = create_compress_index_tensor(
            filter_shape_frac_z, index_dtype, alg)
        tensor_list.append(compress_index)
        bias_tensor = None
        offset_w_tensor = None
        if bias:
            bias_tensor = tvm.placeholder((out_channel,), name='bias_tensor', dtype=res_dtype)
            tensor_list.append(bias_tensor)

        c1_opt, cout1_opt, group_opt, enlarge, _ = get_group_opt_params(shape_in, shape_w, w_dtype, groups)
        conv_res = conv_compress(
            data, weight, compress_index, compress_index_shape,
            {"bias_tensor": bias_tensor, "offset_w_tensor": offset_w_tensor, "pad_h": padh, "pad_w": padw,
             "stride_h": strideh, "stride_w": stridew, "dilate_h": dilateh, "dilate_w": dilatew, "filter_h": filter_h,
             "filter_w": filter_w, "offset_x": offset_x, "res_dtype": res_dtype, "mad_dtype": mad_dtype,
             "fusion_para": fusion_para, "group": groups, "enlarge": enlarge, "c1_opt": c1_opt, "cout1_opt": cout1_opt,
             "group_opt": group_opt, "a_shape": fmap_shape_nc1hwc0, "weight_ori_shape_nchw": weight_ori_shape_nchw,
             "weight_fracz_shape": filter_shape_frac_z,
             "kernel_name": kernel_name, "alg": alg, "compress_index": compress_index},
            optim_dict=optim_dict, dsl_flag=False)
        sch = tbe.auto_schedule(conv_res)
        tensor_list.append(conv_res)

    config = {
        "print_ir": need_print, "need_build": need_build, "name": kernel_name, "tensor_list": tensor_list
    }

    tbe.build(sch, config)


def get_op_support_info(inputs, weight_compress, compress_index, bias, offset_w, outputs, strides, pads, dilations,
                        groups=1, data_format='NHWC', offset_x=0, kernel_name="conv2dcompress"):
    """
    algorithm: get_op_support_info

    Notice
    ------
    get the conv2d compress split info

    Parameters
    ----------
    inputs: dict with keys(shape and dtype)
        input 4d feature map tensor
    weight_compress: dict with keys(shape and dtype)
        input 4d weight tensor
    compress_index: dict with keys(shape and dtype)
        input ND compress index tensor
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
        kernel name, default value is "conv2dcompress"

    Returns
    -------
    None
    """
    bias_idx = 3
    format_x = inputs.get("format")
    slice_info = util_conv2d.get_op_support_info_static_common(bias, bias_idx, format_x)
    return json.dumps(slice_info)
