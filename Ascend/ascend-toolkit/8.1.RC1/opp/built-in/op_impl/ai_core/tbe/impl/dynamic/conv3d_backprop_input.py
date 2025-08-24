#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019 Huawei Technologies Co., Ltd
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
conv3d_backprop_input
"""
import re
import warnings

from impl.util import util_common
from impl.util import util_conv3d
from impl.util import util_cube_dynamic

from impl.util.platform_adapter import classify
from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import error_manager_util
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_register
from impl.util.platform_adapter import tvm

from impl.util.util_cube_dynamic import check_tensor_shape
from impl.util.util_cube_dynamic import check_dynamic_range_lower
from impl.util.util_cube_dynamic import correct_range
from impl.util.util_cube_dynamic import is_empty_tensor_scene

from tbe.common.context import op_context
from tbe.common.utils import const
from tbe.dsl.unify_schedule.conv3d_bp_input_tilingcase import Conv3dBackpropInputBinaryParaProcess

from impl.util.util_cube_dynamic import generalize_shape_and_range


# the dim of shape in conv_backprop must be 5
_CONV_BACKPROP_SHAPE_DIM = 5
# the dim of pads in conv3d_backprop must be 6
_CONV_BACKPROP_PAD_SHAPE_DIM = 6
# the dim of strides in conv_backprop must be 5
_STRIDES_SHAPE_DIM = 5
# the dim of dilations in conv_backprop must be 5
_DILATIONS_SHAPE_DIM = 5
# fmapH, fmapW must be in [1,4096]
_FMAP_HW_MIN = 1
_FMAP_HW_MAX = 4096

# DeDy H,W must be in [1,4096]
_DEDY_HW_MIN = 1
_DEDY_HW_MAX = 4096

# filterH, filterW must be in [1,255]
_FILTER_HW_MIN = 1
_FILTER_HW_MAX = 255
_FILTER_HW_SIZE = 256
_FILTER_D_MAX = 128

# stride must be in [1,63] and h*w not lagger than 256
_STRIDE_HW_MIN = 1
_STRIDE_HW_MAX = 63
_STRIDE_SIZE_MAX = 256
_STRIDE_SIZE_HWD_MAX = 343

# special num
_KHWD_COEFF = 343

# the max num of each axis of shape
_DEFAULT_MAX_SHAPE_NUM = 1000000

# dilation must be in [1,255]
_DILATION_HW_MIN = 1
_DILATION_HW_MAX = 255

# NDHWC or NCDHW
FORMAT_5D_DIMS = 5
# NDC1HWC0
FORMAT_6D_DIMS = 6

# lower range
_LOWER_RANGE = 1
# upper range
_UPPER_RANGE = 4096

# generalize error json
LOWER_LIST = [{"result": "UNSUPPORTED", "reason": {"param_index": [2], "type": ["lower_limit"]}}]
UPPER_LIST = [{"result": "UNSUPPORTED", "reason": {"param_index": [2], "type": ["upper_limit"]}}]
UNSUPPORT_LIST = [{"result": "UNSUPPORTED"}]

# the bytes length of several dtype
_BIT_RATIO_DICT = {"int32": 4, "float32": 4, "float16": 2,
                   "uint8": 1, "int8": 1, "uint4": 0.5, "int4": 0.5}
_DATA_SIZE_MAX = 9223372036854775807
_PADDING_VAILD = [0, 0, 0, 0, 0, 0]
# align with 16 for chips
_C0_SIZE = tbe_platform.C0_SIZE
_DIM_STR = "NDHW"
_DIM_MAP = {"N": [0, 0], "D": [1, 1], "H": [2, 3], "W": [3, 4]}

_DYNAMIC_DIM_VAL = -1
_DYNAMIC_RANK_FLAG = [-2]
_OP_TYPE = "conv3d_backprop_input"
_D_DIM_EXTRA_LEN = 2
# pad_front, pad_tail, pad_top, pad_bottom, pad_left, pad_right
_PADS_FRONT_POS = 0
_PADS_TAIL_POS = 1
_PADS_TOP_POS = 2
_PADS_BOTTOM_POS = 3
_PADS_LEFT_POS = 4
_PADS_RIGHT_POS = 5


def _check_attr_range(attr_name, attr_value, attr_min, attr_max):
    if attr_value < attr_min or attr_value > attr_max:
        error_manager_cube.raise_err_attr_range_invalid(
            'conv3d_backprop_input', "[{},{}]".format(attr_min, attr_max),
            attr_name, str(attr_value))


def _get_ndhwc_shape(ori_format_filters, ori_shape_filters,
                     ori_format_out_backprop, ori_shape_out_backprop,
                     ori_shape_strides, ori_shape_dialtions, range_input,
                     ori_format_res, ori_shape_res):
    def _ncdhw2ndhwc(shape_ncdhw):
        shape_ndhwc = [shape_ncdhw[0], shape_ncdhw[2], shape_ncdhw[3], shape_ncdhw[4], shape_ncdhw[1]]
        return shape_ndhwc

    if ori_format_filters == "DHWCN":
        shape_filters = ori_shape_filters
    elif ori_format_filters == "NDHWC":
        shape_filters = (ori_shape_filters[1],
                         ori_shape_filters[2],
                         ori_shape_filters[3],
                         ori_shape_filters[4],
                         ori_shape_filters[0],
                         )
    elif ori_format_filters == "NCDHW":
        shape_filters = (ori_shape_filters[2],
                         ori_shape_filters[3],
                         ori_shape_filters[4],
                         ori_shape_filters[1],
                         ori_shape_filters[0],
                         )
    else:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'filter',
            'expected_format_list': '[{}, {}, {}]'
                .format('DHWCN', 'NDHWC', 'NCDHW'),
            'format': ori_format_filters
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if ori_format_out_backprop == "NDHWC":
        if list(ori_shape_out_backprop) == _DYNAMIC_RANK_FLAG:
            shape_out_backprop = [-1, -1, -1, -1, shape_filters[-1]]
        else:
            shape_out_backprop = list(ori_shape_out_backprop)
        shape_strides = ori_shape_strides
        shape_dilations = ori_shape_dialtions
    elif ori_format_out_backprop == "NCDHW":
        if list(ori_shape_out_backprop) == _DYNAMIC_RANK_FLAG:
            shape_out_backprop = [-1, -1, -1, -1, shape_filters[-1]]
        else:
            shape_out_backprop = _ncdhw2ndhwc(ori_shape_out_backprop)
        shape_strides = _ncdhw2ndhwc(ori_shape_strides)
        shape_dilations = _ncdhw2ndhwc(ori_shape_dialtions)
    else:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'y_backprop',
            'expected_format_list': '[{}, {}]'.format('NDHWC', 'NCDHW'),
            'format': ori_format_out_backprop
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if ori_format_res == "NDHWC":
        shape_res = ori_shape_res
    elif ori_format_res == "NCDHW":
        shape_res = _ncdhw2ndhwc(ori_shape_res)
        if len(range_input) == FORMAT_5D_DIMS:
            range_input = _ncdhw2ndhwc(range_input)
    else:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'y',
            'expected_format_list': '[{}, {}]'.format('NDHWC', 'NCDHW'),
            'format': ori_format_res
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))
    shape_out_backprop[-1] = shape_filters[-1]

    return shape_filters, shape_out_backprop, shape_strides, shape_dilations, range_input, shape_res


def _check_range(dim_range, range_min=1, range_max=None):
    if dim_range[0] < range_min:
        error_manager_cube.raise_err_specific(
            'conv3d_backprop_input', "the lower bound of range should be larger than {}".format(range_min))
    if not dim_range[1]:
        return
    if (range_max is not None) and (dim_range[1] > range_max):
        error_manager_cube.raise_err_specific(
            'conv3d_backprop_input', "the upper bound of range should be less than {}".format(range_max))
    if dim_range[0] > dim_range[1]:
        error_manager_cube.raise_err_specific(
            'conv3d_backprop_input', "the upper bound of range should be larger than lower bound")


def _check_dynamic_flag(input_size_ndhwc, shape_filters, groups):
    for i in range(4):
        if input_size_ndhwc[i] < -1:
            error_manager_cube.raise_err_specific(
                'conv3d_backprop_input', "Dynamic flag is -1, but dim {} is {}".format(
                    _DIM_STR[i], input_size_ndhwc[i]))
    input_size_ndhwc[-1] = shape_filters[3] * groups if input_size_ndhwc[-1] < 0 else input_size_ndhwc[-1]


def _get_output(x_in, k_size, pads, stride, dilation):
    x_lower, x_upper = x_in
    y_upper = None if x_upper is None else (x_upper + pads[0] + pads[1] - dilation * (k_size - 1) - 1) // stride + 1
    y_lower = None if x_lower is None else (x_lower + pads[0] + pads[1] - dilation * (k_size - 1) - 1) // stride + 1
    return y_lower, y_upper


def _range_to_fix(shape, range_ori):
    # if N/D/H/W dim is not dynamic, change corresponding upper and lower bound in range
    for i in range(4):
        if shape[i] != -1:
            dim = _DIM_STR[i]
            range_idx = _DIM_MAP[dim][1]
            shape_idx = _DIM_MAP[dim][0]
            range_ori[range_idx] = (shape[shape_idx], shape[shape_idx])
    return range_ori


def _modify_dedy(shape, range_ori):
    for i in range(4):
        dim = _DIM_STR[i]
        range_idx = _DIM_MAP[dim][1]
        shape_idx = _DIM_MAP[dim][0]
        if shape[i] == -1:
            shape[shape_idx] = range_ori[range_idx][0]
    return shape


def _range_correction(fmap_range, kernel, pads, stride, dilation, out_shape):
    fmap_range_n, fmap_range_d, fmap_range_c1, fmap_range_h, fmap_range_w, fmap_range_c0 = fmap_range
    _check_range(fmap_range_n)
    _check_range(fmap_range_d)
    _check_range(fmap_range_h, _LOWER_RANGE, _UPPER_RANGE)
    _check_range(fmap_range_w, _LOWER_RANGE, _UPPER_RANGE)
    w_d, w_h, w_w, _, _ = kernel
    _check_attr_range("stride's D", stride[1], _STRIDE_HW_MIN, _STRIDE_HW_MAX)
    _check_attr_range("stride's H", stride[2], _STRIDE_HW_MIN, _STRIDE_HW_MAX)
    _check_attr_range("stride's W", stride[3], _STRIDE_HW_MIN, _STRIDE_HW_MAX)
    if not all(i == 0 for i in pads):
        out_d_upper, out_h_upper, out_w_upper = None, None, None
        out_d_lower = util_common.ceil(fmap_range_d[0], stride[1])
        if fmap_range_d[1]:
            out_d_upper = util_common.ceil(fmap_range_d[1], stride[1])
        out_h_lower = util_common.ceil(fmap_range_h[0], stride[2])
        if fmap_range_h[1]:
            out_h_upper = util_common.ceil(fmap_range_h[1], stride[2])
        out_w_lower = util_common.ceil(fmap_range_w[0], stride[3])
        if fmap_range_w[1]:
            out_w_upper = util_common.ceil(fmap_range_w[1], stride[3])
    else:
        out_d_lower, out_d_upper = _get_output(fmap_range_d, w_d, (pads[0], pads[1]), stride[1], dilation[1])
        if out_d_lower < 1:
            fmap_range_d_lower = min(w_d, fmap_range_d[1]) if fmap_range_d[1] else w_d
            fmap_range_d = (fmap_range_d_lower, fmap_range_d[1])
            out_d_lower, out_d_upper = _get_output(fmap_range_d, w_d, (pads[0], pads[1]), stride[1], dilation[1])

        out_h_lower, out_h_upper = _get_output(fmap_range_h, w_h, (pads[2], pads[3]), stride[2], dilation[2])
        if out_h_lower < 1:
            fmap_range_h_lower = min(w_h, fmap_range_h[1]) if fmap_range_h[1] else w_h
            fmap_range_h = (fmap_range_h_lower, fmap_range_h[1])
            out_h_lower, out_h_upper = _get_output(fmap_range_h, w_h, (pads[2], pads[3]), stride[2], dilation[2])

        out_w_lower, out_w_upper = _get_output(fmap_range_w, w_w, (pads[4], pads[5]), stride[3], dilation[3])
        if out_w_lower < 1:
            fmap_range_w_lower = min(w_w, fmap_range_w[1]) if fmap_range_w[1] else w_w
            fmap_range_w = (fmap_range_w_lower, fmap_range_w[1])
            out_w_lower, out_w_upper = _get_output(fmap_range_w, w_w, (pads[4], pads[5]), stride[3], dilation[3])

    range_dedy = [(fmap_range_n[0], fmap_range_n[1]), (out_d_lower, out_d_upper),
                  (util_common.ceil(out_shape[4], _C0_SIZE), util_common.ceil(out_shape[4], _C0_SIZE)),
                  (out_h_lower, out_h_upper), (out_w_lower, out_w_upper), (_C0_SIZE, _C0_SIZE)]

    range_input = [fmap_range_n, fmap_range_d, fmap_range_c1,
                   fmap_range_h, fmap_range_w, fmap_range_c0]

    return range_dedy, range_input


def _config_placeholder(shape_out_backprop, shape_filters, input_sizes, attrs_list):

    filters_dtype, out_backprop_dtype, range_dedy, range_input, groups, is_fuzzy_build = attrs_list
    _, dy_k0, _ = tbe_platform.CUBE_MKN[out_backprop_dtype]['mac']

    dedy_batch, dedy_depth, dedy_h, dedy_w, dedy_channel = shape_out_backprop
    dedx_batch, dedx_depth, dedx_h, dedx_w, dedx_channel = input_sizes
    filter_depth, filter_h, filter_w, _, filter_batch = shape_filters
    group_dict = util_common.calculate_group(dedx_channel, filter_batch, groups, _C0_SIZE, _C0_SIZE)
    real_g = group_dict.get("real_g")
    cin1_g = group_dict.get("cin1_g")
    cout_g = group_dict.get("cout_g")
    shape_filter_frac = (real_g * filter_depth * cin1_g * filter_h * filter_w,
                         cout_g // _C0_SIZE, _C0_SIZE, _C0_SIZE)
    if not is_fuzzy_build:
        if dedx_batch == -1:
            dedy_batch = operation.var("batch_n", range_input[0])
            operation.add_exclude_bound_var(dedy_batch)
            input_sizes[0] = dedy_batch
        if dedx_depth == -1:
            dx_depth = operation.var("dedx_d", range_input[1])
            dedy_depth = operation.var("dedy_d", range_dedy[1])
            operation.add_exclude_bound_var(dx_depth)
            operation.add_exclude_bound_var(dedy_depth)
            input_sizes[1] = dx_depth
        if dedx_h == -1:
            dx_h = operation.var("dedx_h", range_input[3])
            dedy_h = operation.var("dedy_h", range_dedy[3])
            operation.add_exclude_bound_var(dx_h)
            operation.add_exclude_bound_var(dedy_h)
            input_sizes[2] = dx_h
        if dedx_w == -1:
            dx_w = operation.var("dedx_w", range_input[4])
            dedy_w = operation.var("dedy_w", range_dedy[4])
            operation.add_exclude_bound_var(dx_w)
            operation.add_exclude_bound_var(dedy_w)
            input_sizes[3] = dx_w
    else:
        if dedx_batch == -1:
            dedy_batch = tvm.var("batch_n")
            input_sizes[0] = dedy_batch
        if dedx_depth == -1:
            dx_depth = tvm.var("dedx_d")
            dedy_depth = tvm.var("dedy_d")
            input_sizes[1] = dx_depth
        if dedx_h == -1:
            dx_h = tvm.var("dedx_h")
            dedy_h = tvm.var("dedy_h")
            input_sizes[2] = dx_h
        if dedx_w == -1:
            dx_w = tvm.var("dedx_w")
            dedy_w = tvm.var("dedy_w")
            input_sizes[3] = dx_w

    shape_out_backprop = [dedy_batch, dedy_depth, dedy_h, dedy_w, dedy_channel]
    [dedy_batch, dedy_depth, dedy_h, dedy_w, dedy_channel] = _modify_dedy(shape_out_backprop, range_dedy)
    shape_dedy = (dedy_batch, dedy_depth, util_common.ceil(dedy_channel, dy_k0), dedy_h, dedy_w, dy_k0)

    dx_shape = tvm.placeholder([5], name="input_size", dtype="int32")
    dedy = tvm.placeholder(shape_dedy, name="dedy", dtype=out_backprop_dtype)
    filter_frac = tvm.placeholder(shape_filter_frac, name="filter", dtype=filters_dtype)

    return dx_shape, dedy, filter_frac, input_sizes, shape_out_backprop, group_dict


def _check_pads(pads):
    if isinstance(pads, (tuple, list)) and len(pads) != _CONV_BACKPROP_PAD_SHAPE_DIM:
        error_manager_cube.raise_err_one_para('E62501', 'conv3d_backprop_input', 'pads')

    if isinstance(pads, str) and pads not in ['SAME', 'VALID']:
        error_manager_cube.raise_err_input_params_not_expected(
            'conv3d_backprop_input', 'pads', 'SAME or VALID', str(pads))

    pad_var_flag = False
    if not isinstance(pads, str):
        pad_var_flag = all(i == -1 for i in pads)
        pad_all_positive_flag = all(i >= 0 for i in pads)
        if not pad_var_flag and not pad_all_positive_flag:
            error_manager_cube.raise_err_specific(
                    'conv3d_backprop_input', "pad should be positive")
    return pad_var_flag


@para_check.check_input_type((list, tuple), (list, tuple), (list, tuple),
                             (list, tuple), (str, list, tuple),
                             (list, tuple), str, str, str, str,
                             (list, tuple), (list, tuple), dict)
def check_conv3dbp_input_params(shape_filter,
                                shape_out_backprop,
                                input_sizes, strides, pads, dilations,
                                filter_dtype, out_backprop_dtype,
                                res_dtype, kernel_name, range_input, range_dedy,
                                param_dict=None):
    """
    The params check function of conv3d backprop input

    Parameters
    -------------------------
    shape_filter : The shape of filter
    5-D with shape (depth, height, weight, batch, channels)

    shape_out_backprop : The shape of gradients
    5-D with shape [batch, depth, height, weight,channels]

    input_sizes : The shape of feature map
    5-D with shape [batch, depth, height, weight, channels]

    strides : A list/tuple of ints. The stride of the sliding window

    pads : A list/tuple of ints or str

    dilations : An optional list/tuple of ints

    filter_dtype : The dtype of filter data

    out_backprop_dtype : The dtype of gradients data

    res_dtype : The dtype of result(De/Dx) data

    kernel_name : Cce kernel name

    dynamic_mode : Dynamic mode

    Returns
    -----------------------
    All transformed params
    """
    def _check_64bits_limitation(attr_name, attr_value, dtype):
        bit_ratio = _BIT_RATIO_DICT.get(dtype)
        if attr_value * bit_ratio > _DATA_SIZE_MAX:
            dict_args = {
                'errCode': 'E60020',
                'attr_name': attr_name,
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))

    def _check_ub_limitation(stride_h, stride_w):
        if stride_h == 1 and stride_w == 1:
            return
        fused_num = 0 if param_dict is None else param_dict.get("fused_num", 0)
        if not dedy_w_upper:
            return
        w_value = dedy_w_upper * stride_w

        aub_dedy_size_min = dedy_w_upper * _C0_SIZE * 2
        aub_filling_size_min = w_value * _C0_SIZE * 2
        cub_size_min = _C0_SIZE * _C0_SIZE * _BIT_RATIO_DICT.get(res_dtype)
        ub_size = tbe_platform.get_soc_spec("UB_SIZE")

        if (aub_dedy_size_min * (fused_num + 1) + aub_filling_size_min + cub_size_min) > ub_size:
            dict_args = {
                'errCode': 'E60119'
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))

    def _check_shape_error():

        if not isinstance(fmap_batch, tvm.Var) and dedy_channel != filter_batch:
            error_manager_cube.raise_err_specific(
                'conv3d_backprop_input', "Shape error: Dedy's C must be equal to Filter'N.")

        # check dhw dimension
        if (not isinstance(fmap_h, tvm.Var) and
            not isinstance(fmap_w, tvm.Var) and
            not isinstance(fmap_deep, tvm.Var)):
            pad_head, pad_tail, pad_up, pad_down, pad_left, pad_right = pads
            fmap_h_padding = fmap_h + pad_up + pad_down
            fmap_w_padding = fmap_w + pad_left + pad_right
            fmap_d_padding = fmap_deep + pad_head + pad_tail

            if filter_h_dilation > fmap_h_padding:
                error_manager_cube.raise_err_three_paras('E62507', 'conv3d_backprop_input', 'H',
                                               str(filter_h_dilation), str(fmap_h_padding))
            if filter_w_dilation > fmap_w_padding:
                error_manager_cube.raise_err_three_paras('E62507', 'conv3d_backprop_input', 'W',
                                                str(filter_w_dilation), str(fmap_w_padding))
            if filter_d_dilation > fmap_d_padding:
                error_manager_cube.raise_err_three_paras('E62507', 'conv3d_backprop_input', 'D',
                                               str(filter_d_dilation), str(fmap_d_padding))
            if ((fmap_h - filter_h_dilation + pad_up + pad_down) // stride_h + 1) != dedy_h:
                dict_args = {'errCode': 'E60024'}
                raise RuntimeError(dict_args,
                                   error_manager_util.get_error_message(dict_args))
            if ((fmap_w - filter_w_dilation + pad_left + pad_right) // stride_w + 1) != dedy_w:
                dict_args = {'errCode': 'E60025'}
                raise RuntimeError(dict_args,
                                   error_manager_util.get_error_message(dict_args))
            if ((fmap_deep - filter_d_dilation + pad_head + pad_tail) // stride_d + 1) != dedy_deep:
                dict_args = {'errCode': 'E62508'}
                raise RuntimeError(dict_args,
                                   error_manager_util.get_error_message(dict_args))

    def _check_shape_size():
        if fmap_batch_upper and fmap_d_upper and fmap_h_upper and fmap_w_upper:
            fmap_size = (fmap_batch_upper * util_common.align(fmap_channel, _C0_SIZE) *
                         fmap_d_upper * fmap_h_upper * fmap_w_upper)
            dedy_size = (dedy_batch_upper * dedy_channel * dedy_d_upper *
                         dedy_h_upper * dedy_w_upper)
            _check_64bits_limitation("input", fmap_size, dtype=res_dtype)
            _check_64bits_limitation("out_backprop", dedy_size, dtype=out_backprop_dtype)
        filter_size = util_common.align(filter_batch, _C0_SIZE) * util_common.align(
                      filter_channel, _C0_SIZE) * filter_depth * filter_h * filter_w
        _check_64bits_limitation("filter", filter_size, dtype=filter_dtype)

    # Base check, Mainly required by interface appearance
    # ===========================================================
    # para_check check
    para_check.check_kernel_name(kernel_name)
    para_check.check_shape_rule(shape_filter, _CONV_BACKPROP_SHAPE_DIM,
                                _CONV_BACKPROP_SHAPE_DIM, _DEFAULT_MAX_SHAPE_NUM)
    para_check.check_shape_rule(strides, _STRIDES_SHAPE_DIM, _STRIDES_SHAPE_DIM,
                                _DEFAULT_MAX_SHAPE_NUM)
    para_check.check_shape_rule(dilations, _DILATIONS_SHAPE_DIM,
                                _DILATIONS_SHAPE_DIM, _DEFAULT_MAX_SHAPE_NUM)
    #dilations check
    dilation_n, dilation_d, dilation_h, dilation_w, dilation_c = dilations
    if dilation_d != 1:
        error_manager_cube.raise_err_specific('conv3d_backprop_input', "dilation in D dimension only supports 1.")

    # dtype check
    filter_dtype = filter_dtype.lower()
    out_backprop_dtype = out_backprop_dtype.lower()
    res_dtype = res_dtype.lower()

    # the relation limits between shape
    shape_filter = list(shape_filter)
    shape_out_backprop = list(shape_out_backprop)
    input_sizes = list(input_sizes)
    strides = list(strides)
    fmap_batch, fmap_deep, fmap_h, fmap_w, fmap_channel = input_sizes
    dedy_batch, dedy_deep, dedy_h, dedy_w, dedy_channel = shape_out_backprop
    filter_depth, filter_h, filter_w, filter_channel, filter_batch = shape_filter
    _, stride_d, stride_h, stride_w, _ = strides

    filter_h_dilation = (filter_h - 1) * dilation_h + 1
    filter_w_dilation = (filter_w - 1) * dilation_w + 1
    filter_d_dilation = (filter_depth - 1) * dilation_d + 1

    # pads check
    pad_var_flag = _check_pads(pads)

    # pads compute
    if pads == 'SAME' or pad_var_flag:
        pad_h = util_common.align(fmap_h, stride_h) - stride_h + filter_h_dilation - fmap_h
        pad_h = tvm.max(pad_h, 0) if isinstance(fmap_h, tvm.Var) else max(pad_h, 0)
        pad_up = pad_h // 2
        pad_down = pad_h - pad_up
        pad_w = util_common.align(fmap_w, stride_w) - stride_w + filter_w_dilation - fmap_w
        pad_w = tvm.max(pad_w, 0) if isinstance(fmap_w, tvm.Var) else max(pad_w, 0)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_d = util_common.align(fmap_deep, stride_d) - stride_d + filter_d_dilation - fmap_deep
        pad_d = tvm.max(pad_d, 0) if isinstance(fmap_deep, tvm.Var) else max(pad_d, 0)
        pad_head = pad_d // 2
        pad_tail = pad_d - pad_head
        pads = [pad_head, pad_tail, pad_up, pad_down, pad_left, pad_right]
    elif pads == "VALID":
        pads = _PADDING_VAILD

    pads = list(pads)
    if isinstance(fmap_deep, tvm.Var):
        fmap_d_upper, dedy_d_upper = range_input[1][1], range_dedy[1][1]
    else:
        fmap_d_upper, dedy_d_upper = fmap_deep, dedy_deep
    if isinstance(fmap_h, tvm.Var):
        fmap_h_upper, dedy_h_upper = range_input[3][1], range_dedy[3][1]
        fmap_h_lower, dedy_h_lower = range_input[3][0], range_dedy[3][0]
    else:
        fmap_h_upper, dedy_h_upper = fmap_h, dedy_h
        fmap_h_lower, dedy_h_lower = fmap_h, dedy_h
    if isinstance(fmap_w, tvm.Var):
        fmap_w_upper, dedy_w_upper = range_input[4][1], range_dedy[4][1]
        fmap_w_lower, dedy_w_lower = range_input[4][0], range_dedy[4][0]
    else:
        fmap_w_upper, dedy_w_upper = fmap_w, dedy_w
        fmap_w_lower, dedy_w_lower = fmap_w, dedy_w
    if isinstance(fmap_batch, tvm.Var):
        fmap_batch_upper, dedy_batch_upper = range_input[0][1], range_dedy[0][1]
    else:
        fmap_batch_upper, dedy_batch_upper = fmap_batch, dedy_batch

    if fmap_h_upper != 1 and fmap_w_upper == 1:
        # Chip Design demand fmap_w must larger than 2 when fmap_h == 1
        error_manager_cube.raise_err_one_para(
            'E62006', 'conv3d_backprop_input', 'Chip Design demand input_size_w must >=2 when input_size_h != 1')

    _check_shape_error()
    if dedy_w_upper is not None:
        block_k = tbe_platform.CUBE_MKN.get(out_backprop_dtype).get('mac')[1]
        util_conv3d.check_l1_limitation_dx(dedy_w_upper * stride_w, stride_d, filter_h_dilation,
                                           filter_d_dilation, block_k)

    _check_ub_limitation(stride_h, stride_w)

    # Dedy value limit
    _check_attr_range("Dedy's H after expands", dedy_h_lower * stride_h,
                      _DEDY_HW_MIN, _DEDY_HW_MAX)
    if dedy_h_upper:
        _check_attr_range("Dedy's H after expands", dedy_h_upper * stride_h,
                          _DEDY_HW_MIN, _DEDY_HW_MAX)
    _check_attr_range("Dedy's W after expands", dedy_w_lower * stride_w,
                      _DEDY_HW_MIN, _DEDY_HW_MAX)
    if dedy_w_upper:
        _check_attr_range("Dedy's W after expands", dedy_w_upper * stride_w,
                          _DEDY_HW_MIN, _DEDY_HW_MAX)

    # filter value limit
    _check_attr_range("filter's H", filter_h, _FILTER_HW_MIN, _FILTER_HW_MAX)
    _check_attr_range("filter's W", filter_w, _FILTER_HW_MIN, _FILTER_HW_MAX)
    _check_attr_range("filter's D", filter_depth, _FILTER_HW_MIN, _FILTER_D_MAX)
    _check_attr_range("filter H*W", filter_h * filter_w, _FILTER_HW_MIN,
                      _FILTER_HW_SIZE)
    _check_attr_range("filter H*W*D", filter_h * filter_w * filter_depth,
                      _FILTER_HW_MIN, _KHWD_COEFF)

    # Fmap value limit
    _check_attr_range("Fmap's H", fmap_h_lower, _FMAP_HW_MIN, _FMAP_HW_MAX)
    _check_attr_range("Fmap's W", fmap_w_lower, _FMAP_HW_MIN, _FMAP_HW_MAX)
    if fmap_h_upper:
        _check_attr_range("Fmap's H", fmap_h_upper, _FMAP_HW_MIN, _FMAP_HW_MAX)
    if fmap_w_upper:
        _check_attr_range("Fmap's W", fmap_w_upper, _FMAP_HW_MIN, _FMAP_HW_MAX)
    # stride value limit
    _check_attr_range("stride's H*W",
                      stride_h * stride_w, _STRIDE_HW_MIN, _STRIDE_SIZE_MAX)
    _check_attr_range("stride's H*W*D", stride_h * stride_w * stride_d,
                      _STRIDE_HW_MIN, _STRIDE_SIZE_HWD_MAX)

    # dilation value limit
    _check_attr_range("dilation's N", dilation_n, _DILATION_HW_MIN, _DILATION_HW_MIN)
    _check_attr_range("dilation's C", dilation_c, _DILATION_HW_MIN, _DILATION_HW_MIN)
    _check_attr_range("dilation's H", dilation_h, _DILATION_HW_MIN, _DILATION_HW_MAX)
    _check_attr_range("dilation's W", dilation_w, _DILATION_HW_MIN, _DILATION_HW_MAX)

    # check shape size, 64 bits limitation
    # ===========================================================
    _check_shape_size()

    result = (shape_filter, shape_out_backprop, input_sizes, strides, pads, dilations,
              filter_dtype, out_backprop_dtype, res_dtype, kernel_name)

    return result


def check_empty_tensor(out_backprop, weight, y, strides, pads, dilations=(1, 1, 1, 1, 1)):
    if check_dynamic_range_lower([out_backprop, weight, y]) or is_empty_tensor_scene([out_backprop, weight, y]):
        shape_filters_dhwcn, _, shape_strides, shape_dilations, _, input_sizes = \
            _get_ndhwc_shape(weight.get("ori_format"),
                            weight.get("ori_shape"),
                            out_backprop.get("ori_format"),
                            out_backprop.get("ori_shape"),
                            strides,
                            dilations,
                            y.get("range"),
                            y.get("ori_format"),
                            list(y.get("ori_shape")))
        shape_filters = (shape_filters_dhwcn[4],) + shape_filters_dhwcn[:3] + (shape_filters_dhwcn[3],)
        fmap_format = y.get("ori_format")
        range_input = y.get("range")
        pos_d = fmap_format.find("D")
        pos_h = fmap_format.find("H")
        pos_w = fmap_format.find("W")
        pos_c = fmap_format.find("C")
        pos_n = fmap_format.find("N")
        if len(range_input) == FORMAT_5D_DIMS:
            fmap_range = [range_input[pos_n], range_input[pos_d],
                          range_input[pos_h], range_input[pos_w], range_input[pos_c]]
        elif len(range_input) == FORMAT_6D_DIMS:
            fmap_range = [range_input[0], range_input[1],
                          range_input[3], range_input[4], (input_sizes[4], input_sizes[4])]

        if input_sizes[4] == 0 or 0 in shape_filters[1:]:
            error_manager_cube.raise_err_specific_user("conv3d_backprop_input", "fmap_c, weight_cdhw not support 0")
        check_tensor_shape({"tensor": [out_backprop, weight, y],
                            "value": [-1, 1, -1],
                            "range": [(1, 1), (1, 1), (1, 1)]})
        correct_range(y, fmap_range, shape_filters, shape_strides, shape_dilations, pads, "NDHWC", True)


def check_and_config_para(weight, out_backprop, y, input_size, strides, pads,
                          dilations, groups, data_format, kernel_name, is_fuzzy_build=False):
    """
    algorithm: check_and_config_para

    Parameters
    ----------
    weight: A dict with keys(shape and dtype)
    Input weight tensor

    out_backprop: A dict with keys(shape and dtype)
    Gradients tensor

    y: A dict with keys(shape and dtype)
    Conv3d_backprop_input output tensor, dtype must be assigned

    input_size: dict, will not be used
    input tensor size.

    strides: A tuple/list of 5 integers
    Filter move stride

    pads: A tuple/list of 6 integers: [pad_front, pad_tail, pad_top, pad_bottom, pad_left, pad_right]
    str, "SAME" or "VALID"

    dilations: A tuple/list of 5 integers
    filter expand size of dilated conv3d_backprop_input, default value (1, 1, 1, 1, 1)

    groups: Int of blocked connections from input channels to output channels
    Default value 1

    data_format: The data format of the input and output data
    Default format "NDHWC"

    kernel_name: Str
    Kernel name, default value is "conv3d_backprop_input"

    Returns
    -------
    Shape
    """
    ori_shape_filters = weight.get("ori_shape")
    ori_shape_out_backprop = out_backprop.get("ori_shape")
    # range_input is N,D,C1,H,W,C0
    range_input = y.get("range")
    filters_dtype = weight.get("dtype")
    out_backprop_dtype = out_backprop.get("dtype")
    res_dtype = y.get("dtype")
    ori_format_filters = weight.get("ori_format")
    ori_format_out_backprop = out_backprop.get("ori_format")
    ori_format_res = y.get("ori_format")
    ori_shape_input_size = input_size.get("ori_shape")
    # check -2 scenario
    if len(ori_shape_input_size) == 1:
        input_size_shape, = ori_shape_input_size
        if input_size_shape < 0:
            error_manager_cube.raise_err_specific(
                'conv3d_backprop_input', "prebuild failed, not support input size's shape [-1] and [-2]")

    if not ori_format_res == ori_format_out_backprop == data_format:
        error_manager_cube.raise_err_specific(
            'conv3d_backprop_input', "The data format of out_backprop, input_size and data_format should be same")

    ori_shape_strides = strides
    ori_shape_dilations = dilations

    input_sizes = list(y.get("ori_shape"))

    shape_filters, shape_out_backprop, shape_strides, shape_dilations, range_input, input_sizes = \
        _get_ndhwc_shape(ori_format_filters,
                         ori_shape_filters,
                         ori_format_out_backprop,
                         ori_shape_out_backprop,
                         ori_shape_strides,
                         ori_shape_dilations,
                         range_input,
                         ori_format_res,
                         input_sizes)
    _check_dynamic_flag(input_sizes, shape_filters, groups)
    if len(range_input) == FORMAT_5D_DIMS:
        c1_value = util_common.ceil(input_sizes[-1], _C0_SIZE)
        range_input = [range_input[0], range_input[1], (c1_value, c1_value),
                       range_input[2], range_input[3], (_C0_SIZE, _C0_SIZE)]
    # modify range of non_dynamic dim
    range_input = _range_to_fix(input_sizes, list(range_input))
    # get range_dedy
    range_dedy, range_input = _range_correction(range_input, shape_filters, pads, shape_strides,
                                                shape_dilations, shape_out_backprop)
    if list(ori_shape_out_backprop) == _DYNAMIC_RANK_FLAG:
        range_dedy = [(1, None), (1, None),
                      (util_common.ceil(shape_out_backprop[4], _C0_SIZE),
                       util_common.ceil(shape_out_backprop[4], _C0_SIZE)),
                      (1, None), (1, None), (_C0_SIZE, _C0_SIZE)]
    # get placeholder
    attrs_list = [filters_dtype, out_backprop_dtype, range_dedy, range_input, groups, is_fuzzy_build]
    dx_shape, dedy, filter_frac, input_sizes, shape_out_backprop, group_dict = \
        _config_placeholder(shape_out_backprop, shape_filters, input_sizes, attrs_list)

    res = check_conv3dbp_input_params(shape_filters, shape_out_backprop,
                                      input_sizes, shape_strides, pads,
                                      dilations, filters_dtype,
                                      out_backprop_dtype,
                                      res_dtype, kernel_name, range_input, range_dedy)

    (shape_filter, shape_out_backprop, input_sizes, strides, pads, dilations,
     _, out_backprop_dtype, res_dtype, kernel_name) = res

    return dx_shape, dedy, filter_frac, shape_filter, input_sizes, strides, \
           pads, dilations, res_dtype, kernel_name, group_dict


def _conv3d_backprop_input_compute(filters,
                                   out_backprop,
                                   y_input,
                                   input_size,
                                   strides,
                                   pads,
                                   dilations=(1, 1, 1, 1, 1),
                                   groups=1,
                                   data_format="NDHWC",
                                   kernel_name="conv3d_backprop_input",
                                   build_options: dict = None):
    binary_flag = Conv3dBackpropInputBinaryParaProcess.check_binary_flag(filters)
    if binary_flag:
        ori_paras = {
            "input_size": input_size,
            "filters": filters,
            "out_backprop": out_backprop,
            "y": y_input,
            "strides": strides,
            "pads": pads,
            "dilations": dilations,
            "groups": groups,
            "data_format": data_format,
            "kernel_name": kernel_name,
            "build_options": build_options
        }
        conv3dbp_para = Conv3dBackpropInputBinaryParaProcess(ori_paras)
        conv3dbp_para.check_paras()
        input_size_tensor, filter_tensor, dedy_tensor = conv3dbp_para.define_placeholder()
        res_dtype = conv3dbp_para.y.get("dtype")
        group_dict = conv3dbp_para.attrs.get("group_dict")
        strides_ndhwc = conv3dbp_para.attrs.get("strides_ndhwc")
        pads_res = conv3dbp_para.attrs.get("pads")
        dilations_ndhwc = conv3dbp_para.attrs.get("dilations_ndhwc")
        shape_filter_ncdhw = conv3dbp_para.shape.get("filter_ncdhw")
        input_sizes = conv3dbp_para.shape.get("dedx_ndc1hwc0")
    else:
        check_empty_tensor(out_backprop, filters, y_input, strides, pads, dilations)
        (input_size_tensor, dedy_tensor, filter_tensor, shape_filter, input_sizes, strides_ndhwc, pads_res,
         dilations_ndhwc, res_dtype, kernel_name,
         group_dict) = check_and_config_para(filters, out_backprop, y_input, input_size, strides, pads, dilations,
                                             groups, data_format, kernel_name)
        filter_depth, filter_h, filter_w, filter_channel, filter_batch = shape_filter
        shape_filter_ncdhw = [filter_batch, filter_channel, filter_depth, filter_h, filter_w]

    para_dict = {
        "strides": strides_ndhwc,
        "pads": pads_res,
        "dilations": dilations_ndhwc,
        "res_dtype": res_dtype,
        "kernel_name": kernel_name,
        "group_dict": group_dict,
        "groups": groups,
        "ori_tensors": {"filter": filters,
                        "out_backprop": out_backprop,
                        "y": y_input,
                        "input_size": input_size}
    }

    if build_options is not None:
        para_dict.update(build_options)

    dedx = tbe.conv3d_backprop_input(filter=filter_tensor,
                                     out_backprop=dedy_tensor,
                                     filter_size=shape_filter_ncdhw,
                                     input_size=input_sizes,
                                     para_dict=para_dict)

    context = op_context.get_context()
    if context.get_addition("enable_binary_constant"):
        op_placeholder = [filter_tensor, dedy_tensor]
    else:
        op_placeholder = [input_size_tensor, filter_tensor, dedy_tensor]

    return {'op_placeholder': op_placeholder, 'op_res': [dedx]}


def _get_ndhwc_by_format(obj_format, obj_shape):
    idx_n = obj_format.find('N')
    idx_d = obj_format.find('D')
    idx_h = obj_format.find('H')
    idx_w = obj_format.find('W')
    idx_c = obj_format.find('C')
    return [obj_shape[idx_n],
            obj_shape[idx_d],
            obj_shape[idx_h],
            obj_shape[idx_w],
            obj_shape[idx_c]]


def _get_pad_mode(pads):
    pad_mode = "FIX"
    if not isinstance(pads, str):
        if all(i == -1 for i in pads):
            pad_mode = "VAR"
    elif pads == "SAME":
        pad_mode = "VAR"
    return pad_mode


def _get_filter_dilation(weight: dict, dilation: list, data_format: str) -> int:
    filter_d = weight.get("ori_shape")[weight.get("ori_format").find("D")]
    filter_h = weight.get("ori_shape")[weight.get("ori_format").find("H")]
    filter_w = weight.get("ori_shape")[weight.get("ori_format").find("W")]
    dilation_d = dilation[data_format.find("D")]
    dilation_h = dilation[data_format.find("H")]
    dilation_w = dilation[data_format.find("W")]
    filter_h_dilation = (filter_h - 1) * dilation_h + 1
    filter_d_dilation = (filter_d - 1) * dilation_d + 1
    filter_w_dilation = (filter_w - 1) * dilation_w + 1
    return filter_d_dilation, filter_h_dilation, filter_w_dilation


def _modify_w_range_max(weight: dict, dedy: dict, strides: list, dilation: list, data_format: str,
                        is_dynamic_fuzz_mode: bool = False) -> dict:
    """
    modify w range max value
    """
    d_pos = data_format.find("D")
    w_pos = data_format.find("W")
    out_backprop_date_byte = _BIT_RATIO_DICT.get(dedy.get("dtype"))
    filter_data_byte = _BIT_RATIO_DICT.get(weight.get("dtype"))
    filter_d_dilation, filter_h_dilation, _ = _get_filter_dilation(weight, dilation, data_format)
    d_factor = (filter_d_dilation - _D_DIM_EXTRA_LEN) // strides[d_pos] + _D_DIM_EXTRA_LEN
    block_size_k = tbe_platform.CUBE_MKN[dedy.get("dtype")].get("mac")[1]
    # Using default tiling, b_l1_size is _C0_SIZE * block_size_k * filter_data_byte
    a_l1_size = tbe_platform.get_soc_spec("L1_SIZE") - _C0_SIZE * block_size_k * filter_data_byte
    h_value_max = filter_h_dilation + 1
    w_max = a_l1_size // (h_value_max * d_factor * block_size_k * out_backprop_date_byte) // strides[w_pos]
    exceed_l1_lst = []
    is_single_point = False
    if is_dynamic_fuzz_mode:
        if w_max < dedy.get("ori_range")[w_pos][0]:
            exceed_l1_lst = LOWER_LIST
        elif w_max < dedy.get("ori_range")[w_pos][1]:
            exceed_l1_lst = UPPER_LIST
        return {"exceed_l1_lst": exceed_l1_lst, "w_range_single_point": is_single_point}
    dedy_w = dedy.get("ori_shape")[w_pos]
    w_value = dedy_w * strides[w_pos]
    if w_max < dedy_w:
        if w_value % _C0_SIZE == 0 or _C0_SIZE % w_value == 0:
            h_value_max = filter_h_dilation
            w_max = a_l1_size // (h_value_max * d_factor * block_size_k * out_backprop_date_byte) // strides[w_pos]
            if w_max >= dedy_w:
                # cover a single w point
                dedy.get("ori_range")[w_pos] = (dedy_w, dedy_w)
                is_single_point = True
            else:
                exceed_l1_lst = UNSUPPORT_LIST
                is_single_point = True
        else:
            exceed_l1_lst = UNSUPPORT_LIST
            is_single_point = False
    else:
        w_range_upper = min(w_max, dedy.get("ori_range")[w_pos][1])
        dedy.get("ori_range")[w_pos] = (dedy.get("ori_range")[w_pos][0], w_range_upper)
    return {"exceed_l1_lst": exceed_l1_lst, "w_range_single_point": is_single_point}


def _check_correct_fuzz_input_range(fmap, kernel, pads, stride, dilation, is_dynamic_fuzz_mode):
    fmap_range = fmap.get("ori_range")
    fmap_format = fmap.get("ori_format")
    fmap_shape = fmap.get("ori_shape")
    pos_d = fmap_format.find("D")
    pos_h = fmap_format.find("H")
    pos_w = fmap_format.find("W")
    fmap_range_d = fmap_range[pos_d]
    fmap_range_h = fmap_range[pos_h]
    fmap_range_w = fmap_range[pos_w]
    kernel_shape = kernel.get("ori_shape")
    kernel_format = kernel.get("ori_format")
    w_d = kernel_shape[kernel_format.find("D")]
    w_h = kernel_shape[kernel_format.find("H")]
    w_w = kernel_shape[kernel_format.find("W")]
    correct_range_flag = False
    if all(i == 0 for i in pads):
        out_d_lower, _ = _get_output(fmap_range_d, w_d, (pads[_PADS_FRONT_POS], pads[_PADS_TAIL_POS]),
                                               stride[pos_d], dilation[pos_d])
        if out_d_lower < 1:
            correct_range_flag = True
            fmap_range_d_lower = min(w_d, fmap_range_d[1]) if fmap_range_d[1] else w_d
            fmap_range[pos_d] = (fmap_range_d_lower, fmap_range_d[1])
            warnings.warn("The output calculated based on the lower limit of the input d \
                range is less than 1, and the lower limit of the input d range is corrected \
                as {}".format(fmap_range_d_lower))

        out_h_lower, _ = _get_output(fmap_range_h, w_h, (pads[_PADS_TOP_POS], pads[_PADS_BOTTOM_POS]), stride[pos_h],
                                     dilation[pos_h])
        if out_h_lower < 1:
            correct_range_flag = True
            fmap_range_h_lower = min(w_h, fmap_range_h[1]) if fmap_range_h[1] else w_h
            fmap_range[pos_h] = (fmap_range_h_lower, fmap_range_h[1])
            warnings.warn("The output calculated based on the lower limit of the input h \
                range is less than 1, and the lower limit of the input h range is corrected \
                as {}".format(fmap_range_h_lower))

        out_w_lower, _ = _get_output(fmap_range_w, w_w, (pads[_PADS_LEFT_POS], pads[_PADS_RIGHT_POS]), stride[pos_w],
                                     dilation[pos_w])
        if out_w_lower < 1:
            correct_range_flag = True
            fmap_range_w_lower = min(w_w, fmap_range_w[1]) if fmap_range_w[1] else w_w
            fmap_range[pos_w] = (fmap_range_w_lower, fmap_range_w[1])
            warnings.warn("The output calculated based on the lower limit of the input w \
                range is less than 1, and the lower limit of the input w range is corrected \
                as {}".format(fmap_range_w_lower))
    if correct_range_flag:
        if is_dynamic_fuzz_mode:
            return LOWER_LIST
        if (fmap_range[pos_d][0] > fmap_shape[pos_d] or fmap_range[pos_h][0] > fmap_shape[pos_h] or
            fmap_range[pos_w][0] > fmap_shape[pos_w]):
            return UNSUPPORT_LIST
    return []


@tbe_register.register_param_generalization("Conv3DBackpropInput")
def conv3d_backprop_input_generalization(input_size, filter,
                                         out_backprop, y, strides,
                                         pads, dilations=(1, 1, 1, 1, 1), groups=1,
                                         data_format="NDHWC",
                                         kernel_name="conv3d_backprop_input",
                                         generalize_config=None):
    """
    algorithm: Conv3d_backprop_input generalization

    Notice
    ------
    run after infershape and before operator compile
    only modify input and output tensors with range

    for use:
        1. te fusion distinguish .o (remove the generalization dim)
        2. pass them to the operator to follow the dynamic shape process

    Parameters
    ----------
    input_size: dict, will not be used
    input tensor size.

    filter: A dict with keys(shape and dtype)
    Input weight tensor

    out_backprop: A dict with keys(shape and dtype)
    Gradients tensor

    y: A dict with keys(shape and dtype)
    Conv3d_backprop_input output tensor, dtype must be assigned

    strides: A tuple/list of 5 integers
    Filter move stride

    pads: A tuple/list of 6 integers: [pad_front, pad_tail, pad_top, pad_bottom, pad_left, pad_right]
    str: "SAME" or "VALID"

    dilations: A tuple/list of 5 integers
    filter expand size of dilated conv3d_backprop_input, default value (1, 1, 1, 1, 1)

    groups: Int of blocked connections from input channels to output channels
    Default value 1

    data_format: The data format of the input and output data
    Default format "NDHWC"

    kernel_name: Str
    Kernel name, default value is "conv3d_backprop_input"

    generalize_config: dict
    support keep_rank

    Returns
    -------
    list of params list
    """
    support_mode = ["keep_rank", "all_shape"]
    is_generalize_config = (generalize_config is not None and generalize_config.get("mode") in support_mode)
    if not is_generalize_config:
        return
    result = []
    support_l0c2out_flag = tbe_platform.intrinsic_check_support("Intrinsic_fix_pipe_l0c2out")
    if generalize_config.get("mode") == "keep_rank":  # fuzz build situation
        if support_l0c2out_flag:
            input_size["const_value"] = None
            filter["ori_range"], filter["ori_shape"] = \
                generalize_shape_and_range(filter.get("ori_format"), filter.get("ori_shape"))
            out_backprop["ori_range"], out_backprop["ori_shape"] = \
                generalize_shape_and_range(out_backprop.get("ori_format"), out_backprop.get("ori_shape"))
        else:
            is_dynamic_fuzz_mode = util_conv3d.check_fuzz_dynamic_mode(out_backprop)
            check_result = util_conv3d.check_para_fuzz_compile_3d(out_backprop, y, filter, dilations, strides, pads,
                                                                is_dynamic_fuzz_mode, _OP_TYPE)
            if check_result:
                return check_result
            out_backprop = util_cube_dynamic.gen_conv_shape_range(out_backprop, _OP_TYPE, is_dynamic_fuzz_mode)
            new_pads = util_conv3d.correct_pads(y, out_backprop, filter, strides, pads, is_dynamic_fuzz_mode)
            # check output_d and output_h and output_w
            err_json = _check_correct_fuzz_input_range(out_backprop, filter, new_pads, strides, dilations,
                                                    is_dynamic_fuzz_mode)
            if err_json:
                return err_json
            util_conv3d.get_range(out_backprop)
            # if excced L1 size, narrow the w range.
            l1_size_check_res = _modify_w_range_max(filter, out_backprop, strides, dilations, data_format,
                                                    is_dynamic_fuzz_mode)
            exceed_l1_lst = l1_size_check_res.get("exceed_l1_lst")
            w_range_single_point = l1_size_check_res.get("w_range_single_point")
            if exceed_l1_lst:
                return exceed_l1_lst
            # get dx_range depends on dy_range
            if not is_dynamic_fuzz_mode:
                para_dict = {
                    "strides": _get_ndhwc_by_format(out_backprop["ori_format"], strides),
                    "pads": new_pads,
                    "dilations": _get_ndhwc_by_format(out_backprop["ori_format"], dilations),
                    "kernel_name": kernel_name,
                    "groups": groups,
                    "ori_tensors": {"filter": filter,
                                    "out_backprop": out_backprop,
                                    "y": y,
                                    "input_size": input_size}
                }
                w_pos = data_format.find("W")
                conv3d_backprop = util_cube_dynamic.Conv3dBackpropParaProcess(para_dict, _get_pad_mode(new_pads))
                dy_ori_range = out_backprop.get("ori_range")
                dx_ori_range = conv3d_backprop.get_dx_ori_range(dy_ori_range)
                dx_ori_shape_w = y.get("ori_shape")[w_pos]
                if w_range_single_point:
                    dx_ori_range[w_pos] = (dx_ori_shape_w, dx_ori_shape_w)
                input_size["const_value_range"] = dx_ori_range
                y["ori_range"] = dx_ori_range
                util_conv3d.generalize_input_keep_rank(out_backprop)
                util_conv3d.generalize_input_keep_rank(y)
                input_size["const_value"] = None
            # check attrs and filter
            util_conv3d.get_range(y)
            try:
                check_and_config_para(filter, out_backprop, y, input_size, strides, new_pads,
                                    dilations, groups, data_format, kernel_name, is_fuzzy_build=True)
            except RuntimeError as exc:
                return UNSUPPORT_LIST
            finally:
                pass
        result.append([input_size, filter, out_backprop, y, {"strides": strides}, {"pads": pads},
                    {"dilations": dilations}, {"groups": groups}, {"data_format": data_format}])
    else:
        input_size["ori_format"] = "ND"
        input_size["format"] = "ND"
        filter["range"], filter["shape"] = generalize_shape_and_range(filter.get("format"), filter.get("shape"))
        out_backprop["range"], out_backprop["shape"] = \
            generalize_shape_and_range(out_backprop.get("format"), out_backprop.get("shape"))
        y["range"], y["shape"] = generalize_shape_and_range(y.get("format"), y.get("shape"))
        strides = [-1, -1, -1, -1, -1]
        pads = [-1, -1, -1, -1, -1, -1]
        dilations = [-1, -1, -1, -1, -1]
        groups = -1
        filter["ori_format"] = "NCDHW"
        out_backprop["ori_format"] = "NCDHW"
        y["ori_format"] = "NCDHW"
        data_format = "NCDHW"
        result.append([input_size, filter, out_backprop, y, strides, pads, dilations, groups, data_format])
    return result


@register_operator("Conv3DBackpropInput")
@para_check.check_input_type(dict, dict, dict, dict, (tuple, list), (tuple, list, str), (tuple, list), int, str, str)
def conv3d_backprop_input(input_size,
                          filter,
                          out_backprop,
                          y,
                          strides,
                          pads,
                          dilations=(1, 1, 1, 1, 1),
                          groups=1,
                          data_format="NDHWC",
                          kernel_name="conv3d_backprop_input"):
    """
    algorithm: Conv3d_backprop_input

    Parameters
    ----------
    input_size: dict, will not be used
    input tensor size.

    filter: A dict with keys(shape and dtype)
    Input weight tensor

    out_backprop: A dict with keys(shape and dtype)
    Gradients tensor

    y: A dict with keys(shape and dtype)
    Conv3d_backprop_input output tensor, dtype must be assigned

    strides: A tuple/list of 5 integers
    Filter move stride

    pads: A tuple/list of 6 integers: [pad_front, pad_tail, pad_top, pad_bottom, pad_left, pad_right]
    str, "SAME" or "VALID"

    dilations: A tuple/list of 5 integers
    filter expand size of dilated conv3d_backprop_input, default value (1, 1, 1, 1, 1)

    groups: Int of blocked connections from input channels to output channels
    Default value 1

    data_format: The data format of the input and output data
    Default format "NDHWC"

    kernel_name: Str
    Kernel name, default value is "conv3d_backprop_input"

    Returns
    -------
    None
    """
    input_list = [filter, out_backprop, y, input_size, strides, pads, dilations, groups, data_format, kernel_name]
    context = op_context.get_context()
    extra_params = {
        "single_op": True,
        "split_w": False,
        "binary_flag": Conv3dBackpropInputBinaryParaProcess.check_binary_flag(filter)
    }
    build_args = {
        "constant_realize_extent_in_infer_bound": False
    }

    if context.get_addition("enable_binary_constant"):
        extra_params.update({"need_expand_stride": context.get_addition("need_expand_stride")})
        extra_params.update({const.SD_KD_MODE_KEY: context.get_addition(const.SD_KD_MODE_KEY)})
        extra_params.update({const.DILATION_D_GT_ONE_KEY: context.get_addition(const.DILATION_D_GT_ONE_KEY)})
        input_args = classify(input_list, "conv3d_backprop_input", extra_params)
    else:
        input_args = classify(input_list, "conv3d_backprop_input", extra_params)

    schedules, tensors_list = [], []
    for input_param in input_args:
        tensors = []
        with tbe.compute():
            res = _conv3d_backprop_input_compute(*input_param)
        with tvm.target.cce():
            sch = tbe.auto_schedule(res.get('op_res'))

        schedules.append(sch)
        tensors = res.get('op_placeholder') + res.get('op_res')
        tensors_list.append(tensors)

    config = {'print_ir': False,
              'name': kernel_name,
              'tensor_list': tensors_list,
              'build_args': build_args}

    tbe.build(schedules, config)