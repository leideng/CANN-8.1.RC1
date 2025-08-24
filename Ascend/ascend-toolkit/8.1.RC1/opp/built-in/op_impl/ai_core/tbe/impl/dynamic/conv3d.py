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
dynamic conv3d
"""
from __future__ import absolute_import

import math
import warnings

from impl.util import util_common
from impl.util import util_conv3d
from impl.util import util_select_op_base
from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_register
from impl.util.util_cube_dynamic import check_tensor_shape
from impl.util.util_cube_dynamic import check_dynamic_range_lower
from impl.util.util_cube_dynamic import is_empty_tensor_scene
from impl.util.util_cube_dynamic import correct_range
from impl.util.util_cube_dynamic import check_binary_flag
from impl.util.util_cube_dynamic import generalize_shape_and_range
from tbe.dsl.base.operation import get_op_context
from tbe.dsl.classifier.conv3d_classifier import classify
from tbe.dsl.unify_schedule.conv3d_tilingcase import ATTR_VARS
from tbe.dsl.unify_schedule.conv3d_tilingcase import SHAPE_VARS
from tbe.dsl.unify_schedule.conv3d_tilingcase import TILING_VARS

# [strides_batch, strides_depth, strides_height,
#  strides_width, strides_channel]
STRIDE_LENGTH = 5
DILATION_LENGTH = 5
PADS_LENGTH = 6
# NDHWC or NCDHW
SHAPE_DIMS = 5
FORMAT_5D_DIMS = 5
# NDC1HWC0
FORMAT_6D_DIMS = 6
N_DIM_6D = 0
D_DIM_6D = 1
H_DIM_6D = 3
W_DIM_6D = 4
C0 = 16
HW_MIN = 1
HW_MAX = 4096
PAD_MIN = 0
PAD_MAX = 255
# filterD must be in [1,255]
FILTER_DHW_MIN = 1
FILTER_DHW_MAX = 255
# pad must be in [0,255]
PAD_MIN = 0
PAD_MAX = 255
# stride must be in [1,63]
STRIDE_MIN = 1
STRIDE_MAX = 63

# fmap H and W must be in [1, 4096]
FMAP_HW_MIN = 1
FMAP_HW_MAX = 4096
MAX_SHAPE_NUM = 2 ** 31 - 1

FMAP_TARGET_FORMAT = "NCDHW"
FMAP_FORMAT_WHITE_LIST = ["NCDHW", "NDHWC"]
FILTER_TARGET_FORMAT = "NCDHW"
FILTER_FORMAT_WHITE_LIST = ["NCDHW", "NDHWC", "DHWCN"]
DYNAMIC_FLAG = -1
RANGE_DIM_LEN = 2
L1FUSION_INPUT_CTR = 2
DYNAMIC_RANK_FLAG = [-2]
_OP_TYPE = "conv3d"

_ALIGN_BYTE = 32
_DEFAULT_FP16_SIZE = 2
_DEFAULT_L1_HO_LEN = 2

# generalize error json
LOWER_LIST = [{"result": "UNSUPPORTED", "reason": {"param_index": [0], "type": ["lower_limit"]}}]
UPPER_LIST = [{"result": "UNSUPPORTED", "reason": {"param_index": [0], "type": ["upper_limit"]}}]
UNSUPPORT_LIST = [{"result": "UNSUPPORTED"}]


def get_op_support_info(fmap,
                        weight,
                        bias,
                        offset_w,
                        output,
                        strides,
                        pads,
                        dilations=(1, 1, 1, 1, 1),
                        groups=1,
                        data_format="NDHWC",
                        offset_x=0,
                        kernel_name="conv3d",
                        op_slice_info=""):
    """
    algorithm: get_op_support_info

    Parameters
    ----------
    fmap: A dict with keys(shape and dtype)
    Input 5d feature map tensor

    weight: A dict with keys(shape and dtype)
    Input 5d weight tensor

    bias: A dict with keys(shape and dtype) or None
    Input bias tensor

    offset_w: A dict with keys(shape and dtype) or None
    Input offset_w tensor

    output: A dict with keys(shape and dtype)
    Output tensor, dtype must be assigned

    strides: A tuple/list of 5 integers, format sensitive
    [strides_batch, strides_depth, strides_height, strides_width, strides_channel]

    pads: A tuple/list of 6 integers
    [pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right]

    dilations: A tuple/list of 5 integers
    Dilation on D/H/W, format sensitive, default value is (1, 1, 1, 1, 1)

    groups: Int of blocked connections from input channels to output channels
    Default value is 1

    data_format: The data format of the input and output data
    Default format is "NDHWC"

    offset_x: Int
    Input offset_x value, default value is 0

    kernel_name: Str
    Kernel name, default value is "conv3d"

    op_slice_info: Str
    Default value is ""

    Returns
    -------
    op_cal_info_in_json: A dict with keys(split_maps, reduce_maps, l1_fusion_enable and min_tbe_l1_space)
    """
    def _get_slice_info():
        overlap_d = -1 if (filter_d == 1 and strides_d == 1) else 0
        overlap_h = -1 if (filter_h == 1 and strides_h == 1) else 0
        overlap_w = -1 if (filter_w == 1 and strides_w == 1) else 0

        axis_split_matrix = []
        axis_reduce_list = []
        if fm_format == "NDC1HWC0":
            # cut N
            axis_split_matrix.append([util_select_op_base.SplitInput([0, [0], [-1], [-1]]),
                                      util_select_op_base.SplitOutput([0, [0]])])
            # cut D
            axis_split_matrix.append([util_select_op_base.SplitInput([0, [1], [overlap_d], [overlap_d]]),
                                      util_select_op_base.SplitOutput([0, [1]])])
            # cut H
            axis_split_matrix.append([util_select_op_base.SplitInput([0, [3], [overlap_h], [overlap_h]]),
                                      util_select_op_base.SplitOutput([0, [3]])])
            # cut W
            axis_split_matrix.append([util_select_op_base.SplitInput([0, [4], [overlap_w], [overlap_w]]),
                                      util_select_op_base.SplitOutput([0, [4]])])
            # cut Cout
            if bias:
                axis_split_matrix.append(
                    [util_select_op_base.SplitInput([1, [1], [-1], [-1]], [2, [0], [-1], [-1]]),
                     util_select_op_base.SplitOutput([0, [2]])]
                )
            else:
                axis_split_matrix.append(
                    [util_select_op_base.SplitInput([1, [1], [-1], [-1]]),
                     util_select_op_base.SplitOutput([0, [2]])]
                )
            axis_reduce_list = None
        else:
            axis_split_matrix = None
            axis_reduce_list = None

        return axis_split_matrix, axis_reduce_list

    fm_format = fmap.get("format")
    filter_shape = util_conv3d.transform_shape_with_format(weight.get("ori_format"),
                                                           FILTER_TARGET_FORMAT,
                                                           weight.get("ori_shape"),
                                                           FILTER_FORMAT_WHITE_LIST)
    if filter_shape is None:
        error_manager_cube.raise_err_one_para("E62306", "Conv3d", "filter format should be NDHWC/NCDHW/DHWCN")

    strides_formated = util_conv3d.transform_shape_with_format(fmap.get("ori_format"),
                                                               FMAP_TARGET_FORMAT,
                                                               strides,
                                                               FMAP_FORMAT_WHITE_LIST)
    if strides_formated is None:
        error_manager_cube.raise_err_one_para("E62306", "Conv3d", "data format should be NDHWC or NCDHW")

    _, _, filter_d, filter_h, filter_w = filter_shape
    _, strides_d, strides_h, strides_w, _ = strides_formated

    axis_split_info, axis_reduce_info = _get_slice_info()

    op_cal_info_in_json = util_select_op_base.get_op_cal_info(axis_split_info,
                                                              axis_reduce_info,
                                                              L1FUSION_INPUT_CTR,
                                                              0)
    return op_cal_info_in_json


def _check_groups_validation(fmap_cin, filter_cin, groups):
    """
    algorithm: check the input params of conv3d

    Parameters
    ----------
    fmap_cin: the C channel input of the feature map

    filter_cin: the C channel input of the filter

    groups: The groups for group convolution

    Returns
    -------
    None
    """
    if fmap_cin != filter_cin * groups:
        error_manager_cube.raise_err_scene_equal_limitation("conv3d", 'channel of x',
                                                  'the product of the filter_channel and groups')


def _common_check(shape_filter, stride_dhw):
    _, _, filter_d, filter_h, filter_w = shape_filter
    stride_d, stride_h, stride_w = stride_dhw
    if filter_d < FILTER_DHW_MIN or filter_d > FILTER_DHW_MAX:
        error_manager_cube.raise_err_four_paras('E62003', 'conv3d', 'weight', 'D',
            '[{}, {}]'.format(FILTER_DHW_MIN, FILTER_DHW_MAX), str(filter_d))

    if stride_d < STRIDE_MIN or stride_d > STRIDE_MAX:
        error_manager_cube.raise_err_four_paras('E62003', 'conv3d', 'stride', 'D',
            '[{}, {}]'.format(STRIDE_MIN, STRIDE_MAX), str(stride_d))

    if filter_h < FILTER_DHW_MIN or filter_h > FILTER_DHW_MAX:
        error_manager_cube.raise_err_four_paras('E62003', 'conv3d', 'filter', 'H',
            '[{}, {}]'.format(FILTER_DHW_MIN, FILTER_DHW_MAX), str(filter_h))

    if stride_h < STRIDE_MIN or stride_h > STRIDE_MAX:
        error_manager_cube.raise_err_four_paras('E62003', 'conv3d', 'stride', 'H',
            '[{}, {}]'.format(FILTER_DHW_MIN, FILTER_DHW_MAX), str(stride_h))

    if filter_w < FILTER_DHW_MIN or filter_w > FILTER_DHW_MAX:
        error_manager_cube.raise_err_four_paras('E62003', 'conv3d', 'filter', 'W',
            '[{}, {}]'.format(FILTER_DHW_MIN, FILTER_DHW_MAX), str(filter_w))

    if stride_w < STRIDE_MIN or stride_w > STRIDE_MAX:
        error_manager_cube.raise_err_four_paras('E62003', 'conv3d', 'stride', 'W',
            '[{}, {}]'.format(STRIDE_MIN, STRIDE_MAX), str(stride_w))


def _check_d_dimension(fmap_d, filter_d, pad_d, dilation_d):
    filter_dilated_d = (filter_d - 1) * dilation_d + 1
    if fmap_d != DYNAMIC_FLAG and ((fmap_d + pad_d[0] + pad_d[1]) < filter_dilated_d):
        error_manager_cube.raise_err_three_paras("E62507", "conv3d", "D",
            str(filter_dilated_d), str(fmap_d + pad_d[0] + pad_d[1]))

    if pad_d[0] < PAD_MIN or pad_d[1] < PAD_MIN or pad_d[0] > PAD_MAX or pad_d[1] > PAD_MAX:
        error_manager_cube.raise_err_four_paras('E62003', 'conv3d', 'pad', 'D',
            '[{}, {}]'.format(PAD_MIN, PAD_MAX),
            'pad_d[0] = {}, pad_d[1] = {}'.format(pad_d[0], pad_d[1]))

    if pad_d[0] >= filter_dilated_d or pad_d[1] >= filter_dilated_d:
        error_manager_cube.raise_err_specific_user("conv3d",
            "the depth of pad can not be less than shape_filter's, \
             actual are {} and {}".format(pad_d[0], pad_d[1]))


def _check_h_dimension(fmap_h, filter_h, pad_h, dilation_h):
    filter_dilated_h = (filter_h - 1) * dilation_h + 1
    if pad_h[0] < PAD_MIN or pad_h[1] < PAD_MIN or pad_h[0] > PAD_MAX or pad_h[1] > PAD_MAX:
        error_manager_cube.raise_err_four_paras('E62003', 'conv3d', 'pad', 'H',
            '[{}, {}]'.format(PAD_MIN, PAD_MAX),
            'pad_h[0] = {}, pad_h[1] = {}'.format(pad_h[0], pad_h[1]))

    if fmap_h != DYNAMIC_FLAG and ((fmap_h + pad_h[0] + pad_h[1]) < filter_dilated_h):
        error_manager_cube.raise_err_three_paras("E62507", "conv3d", "H",
            str(filter_dilated_h), str(fmap_h + pad_h[0] + pad_h[1]))

    if pad_h[0] >= filter_dilated_h or pad_h[1] >= filter_dilated_h:
        error_manager_cube.raise_err_specific_user("conv3d",
            "the height of pad can not be less than shape_filter's, \
             actual are {} and {}".format(pad_h[0], pad_h[1]))


def _check_w_dimension(fmap_w, filter_w, pad_w, dilation_w):
    filter_dilated_w = (filter_w - 1) * dilation_w + 1
    if pad_w[0] < PAD_MIN or pad_w[1] < PAD_MIN or pad_w[0] > PAD_MAX or pad_w[1] > PAD_MAX:
        error_manager_cube.raise_err_four_paras('E62003', 'conv3d', 'pad', 'W',
            '[{}, {}]'.format(PAD_MIN, PAD_MAX),
            'pad_w[0] = {}, pad_w[1] = {}'.format(pad_w[0], pad_w[1]))

    if fmap_w != DYNAMIC_FLAG and (filter_dilated_w > (fmap_w + pad_w[0] + pad_w[1])):
        error_manager_cube.raise_err_three_paras("E62507", "conv3d", "W",
            str(filter_dilated_w), str(fmap_w + pad_w[0] + pad_w[1]))

    if pad_w[0] >= filter_dilated_w or pad_w[1] >= filter_dilated_w:
        error_manager_cube.raise_err_specific_user("conv3d",
            "the width of pad can not be less than shape_filter's, \
            actual are {} and {}".format(pad_w[0], pad_w[1]))


def _check_conv3d_shape(shape_fm, shape_filter, pads, stride_dhw, dilation_dhw,
                       fmp_dtype, w_dtype, out_range=None):
    """
    algorithm: check the input params of conv3d

    Parameters
    ----------
    shape_fm: the shape of feature, format is 'NCDHW'.
        a list/tuple of 'int' that has length `== 5`

    shape_filter: the shape of filter, format is 'NCDHW'.
        a list of 'int' that has length `== 5`

    pads: tuple/list of 6 integers
        [pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right]

    stride_dhw: A list of `ints` that has length `== 3`

    dilation_dhw: A list of `ints` that has length `== 3`

    fmp_dtype: the dtype of feature

    w_dtype: the dtype of filter

    out_range: The range of output

    Returns
    -------
    None
    """
    _, _, fmap_d, fmap_h, fmap_w = shape_fm
    filter_n, _, filter_d, filter_h, filter_w = shape_filter

    pad_d = [pads[0], pads[1]]
    pad_h = [pads[2], pads[3]]
    pad_w = [pads[4], pads[5]]

    if -1 not in pad_d:
        _check_d_dimension(fmap_d, filter_d, pad_d, dilation_dhw[0])
    if -1 not in pad_h:
        _check_h_dimension(fmap_h, filter_h, pad_h, dilation_dhw[1])
    if -1 not in pad_w:
        _check_w_dimension(fmap_w, filter_w, pad_w, dilation_dhw[2])

    # C dimension should align 16
    block_size_k = tbe_platform.CUBE_MKN[fmp_dtype]['mac'][1]
    block_size_m = tbe_platform.CUBE_MKN[fmp_dtype]['mac'][0]

    block_size_n = tbe_platform.CUBE_MKN[w_dtype]['mac'][2]
    filter_n = ((filter_n + block_size_n - 1) //
                block_size_n) * block_size_n

    # check for not bigger than L1
    l1_buffer_size = tbe_platform.get_soc_spec("L1_SIZE")
    # when input dtype is float in fuzz, the bit_ratio is same to float16
    m_bit_ratio = {"float16": 2, "int8": 1, "float32": 2}
    point_per_w = out_range[-1][1]
    w_in = block_size_m // point_per_w + 2
    tmp = ((w_in - 1) * stride_dhw[1] + filter_h) * fmap_w
    max_feature_map_l1 = block_size_k * tmp * m_bit_ratio[w_dtype]

    if max_feature_map_l1 > l1_buffer_size:
        error_manager_cube.raise_err_specific_user("conv3d",
            "Input is too large, the minimum tiling may exceed L1_Buffer")


def _get_mad_dtype(w_dtype):
    """
    algorithm: Get the dtype of mad

    Parameters
    ----------
    w_dtype: The dtype of filter

    Returns
    -------
    mad dtype
    """
    mad_dtype = "float32"
    if w_dtype == 'int8':
        mad_dtype = "int32"
    elif tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        mad_dtype = "float16"

    return mad_dtype


def _pos_from_format(ele_format):
    """
    get value from ele_format
    """

    pos_n = ele_format.find('N')
    pos_c = ele_format.find('C')
    pos_d = ele_format.find('D')
    pos_h = ele_format.find('H')
    pos_w = ele_format.find('W')
    return pos_n, pos_c, pos_d, pos_h, pos_w


def _get_shape_ncdhw(shape_in, format_in):
    pos_n, pos_c, pos_d, pos_h, pos_w = _pos_from_format(format_in)
    return [shape_in[pos_n], shape_in[pos_c], shape_in[pos_d], shape_in[pos_h], shape_in[pos_w]]


def _get_fmap_range(in_range, in_shape, in_format):
    if len(in_range) == FORMAT_5D_DIMS:
        # convert NCDHW/NDHWC to NDCHW
        pos_n, pos_c, pos_d, pos_h, pos_w = _pos_from_format(in_format)
        fmap_range = [in_range[pos_n], in_range[pos_d], in_range[pos_c],
                    in_range[pos_h], in_range[pos_w]]
    elif len(in_range) == FORMAT_6D_DIMS:
        # convert NDC1HWC0 to NDCHW
        fmap_range = [in_range[N_DIM_6D], in_range[D_DIM_6D], (in_shape[1], in_shape[1]),
                      in_range[H_DIM_6D], in_range[W_DIM_6D]]
    else:
        error_manager_cube.raise_err_equal_invalid('conv3d', 'range_format', 'in_format')

    return [tuple(r) for r in fmap_range]


def _get_output(x_in, k_size, pads, stride):
    return (x_in + pads[0] + pads[1] - k_size) // stride + 1


def _get_range_both_ends_for_pad_same(axis_range, axis_stride, max_value):
    range_upper = max_value

    range_lower = util_common.ceil(axis_range[0], axis_stride)
    if axis_range[1]:
        range_upper = util_common.ceil(axis_range[1], axis_stride)
    return range_lower, range_upper


def _get_range_both_ends_for_pad_list(axis_range, weight_axis_value, axis_stride, pads, max_value):
    correct_range_flag = False
    fmap_range_axis_lower = axis_range[0]
    out_range_axis = [1, max_value]

    out_range_axis[0] = _get_output(axis_range[0], weight_axis_value, (pads[0], pads[1]), axis_stride)
    if out_range_axis[0] < 1:
        fmap_range_axis_lower = min(weight_axis_value, axis_range[1]) if axis_range[1] else weight_axis_value
        axis_range = (fmap_range_axis_lower, axis_range[1])
        out_range_axis[0] = _get_output(axis_range[0], weight_axis_value, (pads[0], pads[1]), axis_stride)
        correct_range_flag = True
        warnings.warn("The output calculated based on the lower limit of the input axis \
            range is less than 1, and the lower limit of the input axis range is corrected \
            as {}".format(fmap_range_axis_lower))
    if axis_range[1]:
        out_range_axis[1] = _get_output(axis_range[1], weight_axis_value, (pads[0], pads[1]), axis_stride)
    return out_range_axis, fmap_range_axis_lower, correct_range_flag


def _get_out_range(fmap_range, w_shape, pads, strides):
    fmap_range_n, fmap_range_d, fmap_range_c, fmap_range_h, fmap_range_w = fmap_range
    w_n, _, w_d, w_h, w_w = w_shape
    correct_range_flag = False
    y_w_upper = HW_MAX

    if -1 in pads:
        # calculate output range for pad is SAME
        y_d_lower, y_d_upper = _get_range_both_ends_for_pad_same(fmap_range_d, strides[0], None)
        y_h_lower, y_h_upper = _get_range_both_ends_for_pad_same(fmap_range_h, strides[1], HW_MAX)
        y_w_lower, y_w_upper = _get_range_both_ends_for_pad_same(fmap_range_w, strides[2], HW_MAX)

        # the lower limit of w_out is 2
        if y_w_lower < 2:
            lower_new = strides[2] + 1
            fmap_range_w_lower = min(lower_new, fmap_range_w[1]) if fmap_range_w[1] else lower_new
            fmap_range_w = (fmap_range_w_lower, fmap_range_w[1])
            y_w_lower = util_common.ceil(fmap_range_w[0], strides[2])
            correct_range_flag = True
            warnings.warn("The output calculated based on the lower limit of the input w \
                range is less than 2, and the lower limit of the input w range is corrected \
                as {}".format(fmap_range_w_lower))

        pad_check_load2d_flag = True
    else:
        # calcaulate output range for pad is list
        out_range_d, fmap_range_d_lower, correct_range_flag = _get_range_both_ends_for_pad_list(
            fmap_range_d, w_d, strides[0], (pads[0], pads[1]), None)
        y_d_lower, y_d_upper = out_range_d
        fmap_range_d = (fmap_range_d_lower, fmap_range_d[1])
        out_range_h, fmap_range_h_lower, correct_range_flag = _get_range_both_ends_for_pad_list(
            fmap_range_h, w_h, strides[1], (pads[2], pads[3]), HW_MAX)
        y_h_lower, y_h_upper = out_range_h
        fmap_range_h = (fmap_range_h_lower, fmap_range_h[1])
        y_w_lower = _get_output(fmap_range_w[0], w_w, (pads[4], pads[5]), strides[2])
        if y_w_lower < 2:
            lower_new = w_w + strides[2] - (pads[4] + pads[5])
            fmap_range_w_lower = min(lower_new, fmap_range_w[1]) if fmap_range_w[1] else lower_new
            fmap_range_w = (fmap_range_w_lower, fmap_range_w[1])
            y_w_lower = _get_output(fmap_range_w[0], w_w, (pads[4], pads[5]), strides[2])
            correct_range_flag = True
            warnings.warn("The output calculated based on the lower limit of the input w \
                range is less than 2, and the lower limit of the input w range is corrected \
                as {}".format(fmap_range_w_lower))
        if fmap_range_w[1]:
            y_w_upper = _get_output(fmap_range_w[1], w_w, (pads[4], pads[5]), strides[2])

        pad_check_load2d_flag = (sum(pads) == 0)

    if y_d_lower < 1:
        error_manager_cube.raise_err_one_para('E62006', 'conv3d', 'd_out must >= 1')

    load2d_pass_flag =  ((w_d == 1) and (w_h == 1) and (w_w == 1) and
                        pad_check_load2d_flag and
                        (list(strides) == [1, 1, 1]))
    #  Chip Design demand only h_dimesion constraint
    only_fhkh_pass_flag = ((1 <= w_h <= 11) and
                           (strides[1] == 1) and
                           (y_h_lower == 1) and (y_h_upper == 1))

    #  Chip Design demand both h_dimesion and w_dimension constraint
    fhkh_fwkw_pass_flag = ((1 <= w_w <= 11) and (1 <= w_h <= 11) and
                           (strides[1] == 1) and (strides[2] == 1) and
                           (y_h_lower == 1) and (y_w_lower == 1) and
                           (y_h_upper == 1) and (y_w_upper == 1))

    if not (load2d_pass_flag and only_fhkh_pass_flag and fhkh_fwkw_pass_flag):
        if y_w_lower < 2:
            error_manager_cube.raise_err_one_para('E62006', 'conv3d', 'Chip Design demand w_out must >=2')

        if y_h_lower < 1:
            error_manager_cube.raise_err_one_para('E62006', 'conv3d', 'Chip Design demand h_out must >=1')
    out_range = [fmap_range[0], (y_d_lower, y_d_upper), (w_n, w_n),
                 (y_h_lower, y_h_upper), (y_w_lower, y_w_upper)]
    fmap_range = [fmap_range_n, fmap_range_d, fmap_range_c,
                  fmap_range_h, fmap_range_w]

    return out_range, fmap_range, correct_range_flag


def _check_const_dim(dim_value, dim_name):
    if not isinstance(dim_value, int):
        error_manager_cube.raise_err_specific_user("conv3d",
                 "the value of the {} dimension of shape must be int".format(dim_name))
    if dim_value <= 0:
        error_manager_cube.raise_err_specific_user("conv3d",
                 "the value of the {} dimension of shape must be -1 or >0".format(dim_name))


def _check_dynamic_mode(in_shape, w_shape, groups):
    """
    check dynamic mode
    """

    # in_shape format is NCDHW
    c_dim = 1
    fmap_dim_name_lis = ["N", "C", "D", "H", "W"]
    if DYNAMIC_FLAG not in in_shape:
        error_manager_cube.raise_err_specific_user(
            "conv3d", "need at least one dimension is a variable.")
    if DYNAMIC_FLAG in w_shape:
        error_manager_cube.raise_err_specific_user(
            "conv3d", "dynamic weight is not supported yet.")
    if in_shape[c_dim] == DYNAMIC_FLAG:
        in_shape[c_dim] = w_shape[c_dim] * groups
    for index, dim in enumerate(in_shape):
        if dim != DYNAMIC_FLAG:
            _check_const_dim(dim, fmap_dim_name_lis[index])


def _check_variable_range(range_i, mini, maxi=MAX_SHAPE_NUM, name=None):
    """
    check variable range

    """
    if not isinstance(range_i, (tuple, list)):
        error_manager_cube.raise_err_specific_user("conv3d", "type of range must be tuple or list.")
    if len(range_i) != RANGE_DIM_LEN:
        error_manager_cube.raise_err_specific_user("conv3d", "each dimension of range must be 2.")
    if not isinstance(range_i[0], int):
        error_manager_cube.raise_err_specific_user("conv3d", "The lower limit of the range must be Int.")
    if range_i[1] and (not isinstance(range_i[1], int)):
        error_manager_cube.raise_err_specific_user("conv3d", "The upper limit of the range must be Int or None.")
    if range_i[0] < mini or range_i[0] > maxi:
        error_manager_cube.raise_err_attr_range_invalid(
            "conv3d", [mini, maxi], name, range_i[0])
    if range_i[1] and (range_i[1] < mini or range_i[1] > maxi):
        error_manager_cube.raise_err_attr_range_invalid(
            "conv3d", [mini, maxi], name, range_i[1])


def check_empty_tensor(fmap, weight, output, strides, pads, dilation=(1, 1, 1, 1, 1), groups=1):
    if check_dynamic_range_lower([fmap, weight, output]) or is_empty_tensor_scene([fmap, weight, output]):
        in_shape = list(fmap.get("ori_shape"))
        w_shape = list(weight.get("ori_shape"))
        in_format = fmap.get("ori_format")
        w_format = weight.get("ori_format")
        in_range = fmap.get("range")
        strides_ncdhw = _get_shape_ncdhw(strides, in_format)
        dilation_ncdhw = _get_shape_ncdhw(dilation, in_format)
        input_info = {"fmap": (in_format, in_shape), "weight": (w_format, w_shape)}
        fmap_ncdhw, w_shape_ncdhw, _, _ = util_conv3d.format_normalize(input_info, strides, dilation, groups)

        if list(in_shape) == DYNAMIC_RANK_FLAG:
            fmap_range_ncdhw = [(1, None), (w_shape_ncdhw[1] * groups, w_shape_ncdhw[1] * groups),
                                (1, None), (1, None), (1, None)]
        else:
            range_ndchw = _get_fmap_range(in_range, fmap_ncdhw, in_format)
            fmap_range_ncdhw = [range_ndchw[0], range_ndchw[2], range_ndchw[1]] + range_ndchw[3:]

        if fmap_ncdhw[1] == 0 or 0 in w_shape_ncdhw[1:]:
            error_manager_cube.raise_err_specific_user("conv3d", "fmap_c weight_cdhw not support 0")

        check_tensor_shape({"tensor": [fmap, weight, output],
                            "value": [-1, 1, -1],
                            "range": [(1, 1), (1, 1), (1, 1)]})

        if list(fmap.get("ori_shape")) != DYNAMIC_RANK_FLAG:
            correct_range(fmap, fmap_range_ncdhw, w_shape_ncdhw, strides_ncdhw, dilation_ncdhw, pads, "NCDHW")


def _check_input_output_para(fmap, weight, bias, offset_w, output):
    in_shape = list(fmap.get("ori_shape"))
    w_shape = list(weight.get("ori_shape"))
    in_dtype = fmap.get("dtype")
    w_dtype = weight.get("dtype")
    res_dtype = output.get("dtype")
    bias_dtype = None
    in_format = fmap.get("ori_format")
    w_format = weight.get("ori_format")
    in_range = fmap.get("range")

    in_dtype = in_dtype.lower()
    w_dtype = w_dtype.lower()
    res_dtype = res_dtype.lower()

    if list(in_shape) != DYNAMIC_RANK_FLAG:
        if len(in_shape) != SHAPE_DIMS:
            error_manager_cube.raise_err_one_para('E62501', 'conv3d', 'in_shape')

    if in_format not in FMAP_FORMAT_WHITE_LIST:
        error_manager_cube.raise_err_input_format_invalid(
            'conv3d', 'input', FMAP_FORMAT_WHITE_LIST, in_format)

    if w_format not in FILTER_FORMAT_WHITE_LIST:
        error_manager_cube.raise_err_input_format_invalid(
            'conv3d', 'weight', FILTER_FORMAT_WHITE_LIST, w_format)

    if bias:
        util_conv3d.check_bias(bias, res_dtype)
        bias_dtype = bias.get("dtype").lower()
        para_check.check_dtype_rule(bias_dtype, ("float16", "float32"), "bias")

    if offset_w:
        error_manager_cube.raise_err_specific_user(
            'conv3d', "offset_w is not supported yet in dynamic conv3d")

    binary_flag = False
    if check_binary_flag(in_range, w_shape) and (DYNAMIC_FLAG in in_shape or list(in_shape) == DYNAMIC_RANK_FLAG):
        binary_flag = True
        return {"in_format": in_format, "w_format": w_format, "in_shape": in_shape, "w_shape": w_shape,
                "w_dtype": w_dtype, "in_dtype": in_dtype, "res_dtype": res_dtype, "bias_dtype": bias_dtype,
                "in_range": in_range, "binary_flag": binary_flag}

    para_check.check_shape_rule(w_shape, min_dim=SHAPE_DIMS, max_dim=SHAPE_DIMS)

    return {"in_format": in_format, "w_format": w_format, "in_shape": in_shape, "w_shape": w_shape, "w_dtype": w_dtype,
            "in_dtype": in_dtype, "res_dtype": res_dtype, "in_range": in_range, "binary_flag": binary_flag,
            "bias_dtype": bias_dtype}


def _check_attr_para(strides, pads, dilations, groups, binary_flag):

    if len(strides) != STRIDE_LENGTH:
        error_manager_cube.raise_err_specific_user("conv3d", "strides should be 5d list")

    if len(dilations) != DILATION_LENGTH:
        error_manager_cube.raise_err_specific_user("conv3d", "dilations should be 5d list")

    if len(pads) != PADS_LENGTH:
        error_manager_cube.raise_err_one_para('E62501', 'conv3d', 'pads')

    if binary_flag:
        return {"strides": strides, "dilations": dilations, "groups": groups, "pads": pads}

    # check dilations for it1
    if len(set(dilations)) != 1 or dilations[2] != 1:
        error_manager_cube.raise_err_three_paras('E62001', 'Conv3D', str(dilations[2]),
            str(dilations[3]), str(dilations[1]))

    return {"strides": strides, "dilations": dilations, "groups": groups, "pads": pads}


def _config_para(input_output_info, attr_info):
    in_format = input_output_info.get("in_format")
    w_format = input_output_info.get("w_format")
    in_shape = input_output_info.get("in_shape")
    w_shape = input_output_info.get("w_shape")
    strides = attr_info.get("strides")
    dilations = attr_info.get("dilations")
    groups = attr_info.get("groups")
    pads = attr_info.get("pads")
    w_dtype = input_output_info.get("w_dtype")
    in_dtype = input_output_info.get("in_dtype")
    res_dtype = input_output_info.get("res_dtype")
    bias_dtype = input_output_info.get("bias_dtype", res_dtype)
    in_range = input_output_info.get("in_range")

    if input_output_info.get("binary_flag"):
        config_dict = {"in_dtype": in_dtype, "w_dtype": w_dtype, "res_dtype": res_dtype, "bias_dtype": bias_dtype}
        return config_dict

    # shape_fm/shape_filter format is NCDHW
    input_info = {"fmap": (in_format, in_shape), "weight": (w_format, w_shape)}
    shape_fm, shape_filter, stride, dilation = util_conv3d.format_normalize(
        input_info, strides, dilations, groups)
    stride_dhw = stride[2:]
    dilation_dhw = dilation[2:]
    cin0 = tbe_platform.CUBE_MKN[w_dtype]['mac'][1]
    cout0 = tbe_platform.CUBE_MKN[w_dtype]['mac'][2]
    _check_dynamic_mode(shape_fm, shape_filter, groups)
    util_conv3d.check_conv3d_dtype(tbe_platform.intrinsic_check_support(
        "Intrinsic_fix_pipe_l0c2out"), in_dtype, w_dtype, res_dtype)
    # calculate fmap_range
    if list(in_shape) == DYNAMIC_RANK_FLAG:
        fmap_range = [(1, None), (1, None),
                      (shape_filter[1] * groups, shape_filter[1] * groups),
                      (1, None), (1, None)]
    else:
        fmap_range = _get_fmap_range(in_range, shape_fm, in_format)

    # check fmap_range
    _, _, _, h_range, w_range = fmap_range
    _check_variable_range(h_range, HW_MIN, HW_MAX, "fmap_h")
    _check_variable_range(w_range, HW_MIN, HW_MAX, "fmap_w")
    name_lis = ['fmap_batch', 'fmap_d', 'fmap_c']
    for index, dim_range in enumerate(fmap_range[:3]):
        _check_variable_range(dim_range, 1, name=name_lis[index])
    _common_check(shape_filter, stride_dhw)

    # calculate out_range
    out_range, fmap_range, correct_range_flag = _get_out_range(fmap_range, shape_filter, pads,
                                                               stride_dhw)
    _check_groups_validation(shape_fm[1], shape_filter[1], groups)
    # calculate group parameter
    group_dict = util_common.calculate_group(shape_fm[1], shape_filter[0],
                                             groups, cout0, cin0)
    # C dimension 16 aligned
    _check_conv3d_shape(shape_fm, shape_filter, pads,
                        stride_dhw, dilation_dhw, in_dtype,
                        w_dtype, out_range)

    config_dict = {
        "shape_fm": shape_fm,
        "shape_filter": shape_filter,
        "stride_dhw": stride_dhw,
        "dilation_dhw": dilation_dhw,
        "group_dict": group_dict,
        "groups": groups,
        "in_dtype": in_dtype,
        "w_dtype": w_dtype,
        "res_dtype": res_dtype,
        "bias_dtype": bias_dtype,
        "fmap_range": fmap_range,
        "out_range": out_range,
        "correct_range_flag":correct_range_flag,
    }
    return config_dict


def _calc_pads(fmap_shape_ndc1hwc0, shape_filter, stride_dhw, dilation_dhw, pads):
    """
    calculate pads
    """
    _, fmap_d, _, fmap_h, fmap_w, _ = fmap_shape_ndc1hwc0
    _, _, filter_d, filter_h, filter_w = shape_filter
    stride_d, stride_h, stride_w = stride_dhw
    dilation_d, dilation_h, dilation_w = dilation_dhw

    filter_d_dilation = (filter_d - 1) * dilation_d + 1
    filter_h_dilation = (filter_h - 1) * dilation_h + 1
    filter_w_dilation = (filter_w - 1) * dilation_w + 1
    if -1 in pads:
        if list(stride_dhw) == [1, 1, 1] and [filter_d, filter_h, filter_w] == [1, 1, 1]:
            return [0, 0, 0, 0, 0, 0]
        pad_d = \
            util_common.ceil(fmap_d, stride_d) * stride_d - stride_d + filter_d_dilation - fmap_d
        pad_d = tvm.max(pad_d, 0)
        pad_head = pad_d // 2
        pad_tail = pad_d - pad_head
        pad_h = \
            util_common.ceil(fmap_h, stride_h) * stride_h - stride_h + filter_h_dilation - fmap_h
        pad_h = tvm.max(pad_h, 0)
        pad_up = pad_h // 2
        pad_down = pad_h - pad_up
        pad_w = \
            util_common.ceil(fmap_w, stride_w) * stride_w - stride_w + filter_w_dilation - fmap_w
        pad_w = tvm.max(pad_w, 0)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return pad_head, pad_tail, pad_up, pad_down, pad_left, pad_right
    return pads


def _get_input_shape(config_dict):
    shape_fm = config_dict.get('shape_fm')
    shape_filter = config_dict.get('shape_filter')
    group_dict = config_dict.get('group_dict')
    fmp_dtype = config_dict.get('in_dtype')
    w_dtype = config_dict.get('w_dtype')
    # C and Cout align 16
    shape_fm = list(shape_fm)
    fmp_block_k = tbe_platform.CUBE_MKN[fmp_dtype]['mac'][1]
    shape_fm[1] = ((shape_fm[1] + fmp_block_k - 1) // fmp_block_k) * fmp_block_k
    w_block_k = tbe_platform.CUBE_MKN[w_dtype]['mac'][1]
    shape_filter = list(shape_filter)
    shape_filter[1] = ((shape_filter[1] + w_block_k - 1) // w_block_k) * w_block_k
    w_block_n = tbe_platform.CUBE_MKN[w_dtype]['mac'][2]
    shape_filter[0] = ((shape_filter[0] + w_block_n - 1) // w_block_n) * w_block_n

    # convert fmap shape to ndc1hwc0
    batch, cin, fmp_d, fmp_h, fmp_w = shape_fm
    shape_fmp_ndc1hwc0 = [batch, fmp_d, (cin + fmp_block_k - 1) // fmp_block_k, fmp_h, fmp_w,
                          fmp_block_k]
    # convert filter shape to frac_z
    real_g = group_dict["real_g"]
    cin1_g = group_dict["cin1_g"]
    cout_g = group_dict["cout_g"]
    cout_ori = group_dict["cout_ori"]
    _, _, w_d, w_h, w_w = shape_filter
    shape_w_frac_z = (real_g * w_d * cin1_g * w_h * w_w, cout_g // w_block_n,
                        w_block_n, w_block_k)
    return shape_fmp_ndc1hwc0, shape_w_frac_z


def _define_var_non_binary(config_dict):
    '''
    Define vars for non binary mode.
    '''
    fmap_range = config_dict.get('fmap_range')
    out_range = config_dict.get('out_range')
    shape_fmp_ndc1hwc0, shape_w_frac_z = _get_input_shape(config_dict)
    # define var
    if shape_fmp_ndc1hwc0[N_DIM_6D] == DYNAMIC_FLAG:
        shape_fmp_ndc1hwc0[N_DIM_6D] = operation.var("batch_n", fmap_range[N_DIM_6D])
        operation.add_exclude_bound_var(shape_fmp_ndc1hwc0[N_DIM_6D])
    if shape_fmp_ndc1hwc0[D_DIM_6D] == DYNAMIC_FLAG:
        shape_fmp_ndc1hwc0[D_DIM_6D] = operation.var("fmap_d", fmap_range[D_DIM_6D])
        d_out = operation.var("d_out", out_range[D_DIM_6D])
        operation.add_exclude_bound_var(shape_fmp_ndc1hwc0[D_DIM_6D])
        operation.add_exclude_bound_var(d_out)
    if shape_fmp_ndc1hwc0[H_DIM_6D] == DYNAMIC_FLAG:
        shape_fmp_ndc1hwc0[H_DIM_6D] = operation.var("fmap_h", fmap_range[H_DIM_6D])
        h_out = operation.var("h_out", out_range[H_DIM_6D])
        operation.add_exclude_bound_var(shape_fmp_ndc1hwc0[H_DIM_6D])
        operation.add_exclude_bound_var(h_out)
    if shape_fmp_ndc1hwc0[W_DIM_6D] == DYNAMIC_FLAG:
        shape_fmp_ndc1hwc0[W_DIM_6D] = operation.var("fmap_w", fmap_range[W_DIM_6D])
        w_out = operation.var("w_out", out_range[W_DIM_6D])
        operation.add_exclude_bound_var(shape_fmp_ndc1hwc0[W_DIM_6D])
        operation.add_exclude_bound_var(shape_fmp_ndc1hwc0[W_DIM_6D])
    return shape_fmp_ndc1hwc0, shape_w_frac_z


def _define_optional_vars(var_name):
    if operation.get_te_var(var_name) is None:
        return operation.var(var_name)
    return operation.get_te_var(var_name).get_tvm_var()


def _define_binary_mode_vars(fmap_dtype="float16"):
    '''
    Define vars for binary mode.
    '''
    shape_var_map = {}
    attr_var_map = {}
    tiling_var_map = {}

    for var in SHAPE_VARS:
        shape_var_map[var] = _define_optional_vars(var)

    for var in ATTR_VARS:
        attr_var_map[var] = operation.var(var)

    for var in TILING_VARS:
        tiling_var_map[var] = operation.var(var)

    w_block_n = tbe_platform.CUBE_MKN[fmap_dtype]['mac'][2]
    w_block_k = tbe_platform.CUBE_MKN[fmap_dtype]['mac'][1]
    real_g = attr_var_map.get("real_g")
    cin1_g = attr_var_map.get("cin1_g")
    cout1_g = attr_var_map.get("cout1_g")
    mag_factor = attr_var_map.get("mag_factor")
    w_d, w_h, w_w = attr_var_map.get("kernel_d"), attr_var_map.get("kernel_h"), attr_var_map.get("kernel_w")
    c0_size = tbe_platform.CUBE_MKN.get(fmap_dtype).get("mac")[1]

    var_shape_map = {}
    var_shape_map["fmap_ncdhw"] = (shape_var_map.get("batch_n"), shape_var_map.get("fmap_c"),
                                   shape_var_map.get("fmap_d"),
                                   shape_var_map.get("fmap_h"), shape_var_map.get("fmap_w"))
    var_shape_map["output_ncdhw"] = (shape_var_map.get("batch_n"), shape_var_map.get("c_out"),
                                     shape_var_map.get("d_out"),
                                     shape_var_map.get("h_out"), shape_var_map.get("w_out"))
    var_shape_map["dedw_ncdhw"] = (shape_var_map.get("c_out"), shape_var_map.get("fmap_c"),
                                   attr_var_map.get("kernel_d"),
                                   attr_var_map.get("kernel_h"), attr_var_map.get("kernel_w"))
    var_shape_map["fmap_ndc1hwc0"] = (shape_var_map.get("batch_n"), shape_var_map.get("fmap_d"),
                                      attr_var_map.get("fmap_c1"),
                                      shape_var_map.get("fmap_h"), shape_var_map.get("fmap_w"), c0_size)
    var_shape_map["weight_frac_z"] = (real_g * w_d * cin1_g * w_h * w_w, cout1_g, w_block_n, w_block_k)
    var_shape_map["stride_dhw"] = (attr_var_map.get("stride_d"),
                                   attr_var_map.get("stride_h"), attr_var_map.get("stride_w"))
    var_shape_map["pads"] = (attr_var_map.get("padf"), attr_var_map.get("padb"), attr_var_map.get("padu"),
                             attr_var_map.get("padd"), attr_var_map.get("padl"), attr_var_map.get("padr"))
    var_shape_map["dilation_dhw"] = (attr_var_map.get("dilation_d"),
                                     attr_var_map.get("dilation_h"), attr_var_map.get("dilation_w"))
    var_shape_map["group_dict"] = {"real_g": real_g, "mag_factor": mag_factor, "cin1_g": cin1_g,
                                   "cout_g": cout1_g * w_block_n, "cin_ori": shape_var_map.get("fmap_c"),
                                   "cout_ori": shape_var_map.get("c_out")}
    var_shape_map["groups"] = -1
    var_shape_map["correct_range_flag"] = False

    return var_shape_map


def _get_para_dict(ori_tensors, shape_info_dict, type_info, attr_info):
    bias_tensor = ori_tensors.get("bias_tensor")

    para_dict = {
        "dsl_flag": False,
        "bias_tensor": bias_tensor,
        "pads": attr_info.get("pads"),
        "strides": list(shape_info_dict.get("stride_dhw")),
        "dilation_dhw": list(shape_info_dict.get("dilation_dhw")),
        "res_dtype": type_info.get("res_dtype"),
        "mad_dtype": type_info.get("mad_dtype"),
        "kernel_name": attr_info.get("kernel_name"),
        "group_dict": shape_info_dict.get("group_dict"),
        "correct_range_flag": shape_info_dict.get("correct_range_flag"),
        "groups": shape_info_dict.get("groups"),
        "ori_tensors": ori_tensors,
        "binary_flag": attr_info.get("binary_flag"),
    }
    return para_dict


def _conv3d_compute(compute_input_dict):

    """
    algorithm: conv3d

    Parameters
    ----------
    compute_input_dict: A dict with input, attr, option info

    Returns
    -------
    tvm compute
    """
    fmap = compute_input_dict.get("fmap")
    weight = compute_input_dict.get("weight")
    bias = compute_input_dict.get("bias")
    offset_w = compute_input_dict.get("offset_w")
    output = compute_input_dict.get("output")
    strides = compute_input_dict.get("strides")
    pads = compute_input_dict.get("pads")
    dilations = compute_input_dict.get("dilations")
    groups = compute_input_dict.get("groups")
    kernel_name = compute_input_dict.get("kernel_name")
    options = compute_input_dict.get("options")

    check_empty_tensor(fmap, weight, output, strides, pads, dilations, groups)
    # shape_fm/shape_filter format is NCDHW, fmap_range/out_range format is NDCHW
    input_output_info = _check_input_output_para(fmap, weight, bias, offset_w, output)
    attr_info = _check_attr_para(strides, pads, dilations, groups, input_output_info.get("binary_flag"))
    config_dict = _config_para(input_output_info, attr_info)
    w_dtype = config_dict.get('w_dtype')
    mad_dtype = _get_mad_dtype(w_dtype)
    shape_filter = config_dict.get('shape_filter')
    group_dict = config_dict.get('group_dict')
    fmp_dtype = config_dict.get('in_dtype')
    res_dtype = config_dict.get('res_dtype')
    bias_dtype = config_dict.get('bias_dtype', res_dtype)
    fmap_range = fmap.get("range")

    offset_w = None
    pads = list(pads)

    bias_tensor = None
    var_shape_map = {}
    binary_flag = check_binary_flag(fmap_range, weight.get("shape"))
    if not binary_flag:
        shape_fmp_ndc1hwc0, shape_w_frac_z = _define_var_non_binary(config_dict)
        # calculate pads
        stride_dhw, dilation_dhw = config_dict.get('stride_dhw'), config_dict.get('dilation_dhw')
        pads = _calc_pads(shape_fmp_ndc1hwc0, shape_filter, stride_dhw, dilation_dhw, pads)
        cout_ori = group_dict.get("cout_ori")
    else:
        var_shape_map = _define_binary_mode_vars(fmp_dtype)
        shape_fmp_ndc1hwc0, shape_w_frac_z = var_shape_map.get("fmap_ndc1hwc0"), var_shape_map.get("weight_frac_z")
        shape_filter = var_shape_map.get("dedw_ncdhw")
        pads = var_shape_map.get("pads")
        cout_ori = var_shape_map.get("output_ncdhw")[1]

    data = tvm.placeholder(shape_fmp_ndc1hwc0, name='Fmap', dtype=fmp_dtype)
    weight = tvm.placeholder(shape_w_frac_z, name='Filter', dtype=w_dtype)
    ori_tensors = {"fmap": fmap, "weight": weight, "bias": bias, "output": output}
    if bias is not None:
        align_mod = _ALIGN_BYTE // util_common.BIT_RATIO_DICT.get(bias_dtype, _DEFAULT_FP16_SIZE)
        bias_align_shape = (cout_ori + align_mod - 1) // align_mod * align_mod
        bias_tensor = tvm.placeholder((bias_align_shape,), name='bias_tensor', dtype=bias_dtype,
                                        attrs={"ori_shape": [cout_ori, ]})
        ori_tensors["bias_tensor"] = bias_tensor

    type_info = {"res_dtype": res_dtype, "mad_dtype": mad_dtype}
    attr_info = {"kernel_name": kernel_name, "pads": pads, "binary_flag": binary_flag}
    shape_info_dict = var_shape_map if var_shape_map else config_dict
    para_dict = _get_para_dict(ori_tensors, shape_info_dict, type_info, attr_info)
    if options is not None:
        para_dict.update(options)

    conv_res = tbe.conv3d(data, weight, shape_filter, para_dict)

    if bias:
        return {"op_placeholder": [data, weight, bias_tensor], "op_res": [conv_res]}
    return {"op_placeholder": [data, weight], "op_res": [conv_res]}


def _check_l1_size(fmap, weight, pads, strides, dilations, is_dynamic_fuzz_mode):
    """
    check exceed l1 buf
    graph mode fuzz, check range[high]
    single mode fuzz, check shape and modify range[high]
    """
    def _get_l1_size(w_in):
        if DYNAMIC_FLAG in pads:
            w_out = w_in + stride_w - 1 // stride_w
        else:
            w_out = (w_in + (pad_left + pad_right) - filter_dilated_w) // stride_w + 1
        limit_h_out = math.floor(block_size_m / w_out) + _DEFAULT_L1_HO_LEN
        hw_size = ((limit_h_out - 1) * stride_h + filter_dilated_h) * w_in
        return hw_size * block_size_k * util_common.BIT_RATIO_DICT.get(fmap_dtype, _DEFAULT_FP16_SIZE)

    l1_buffer_size = tbe_platform.get_soc_spec("L1_SIZE")
    fmap_dtype = fmap["dtype"]
    block_size_k = tbe_platform.CUBE_MKN[fmap_dtype]['mac'][1]
    block_size_m = tbe_platform.CUBE_MKN[fmap_dtype]['mac'][0]
    idx_w = fmap.get("ori_format").find("W")
    idx_h = fmap.get("ori_format").find("H")
    stride_h = strides[idx_h]
    stride_w = strides[idx_w]
    _, _, _, _, pad_left, pad_right = pads
    dilation_w = dilations[idx_w]
    dilation_h = dilations[idx_h]
    filter_w = weight.get("ori_shape")[weight.get("ori_format").find('W')]
    filter_h = weight.get("ori_shape")[weight.get("ori_format").find('H')]
    filter_dilated_w = (filter_w - 1) * dilation_w + 1
    filter_dilated_h = (filter_h - 1) * dilation_h + 1
    if not is_dynamic_fuzz_mode:
        w_in = fmap.get("ori_shape")[idx_w]
        limit_size = _get_l1_size(w_in)
        if limit_size > l1_buffer_size:
            return util_conv3d.LOWER_LIST
    return []


def _config_compute_input_dict(compute_input):
    '''
    config compute input dict
    '''
    option_lis = []
    # 3 represents 3 members: input_lis, attr_lis, option_lis
    if len(compute_input) == 3:
        input_lis, attr_lis, option_lis = compute_input
    else:
        input_lis, attr_lis = compute_input

    compute_input_dict = {
        "fmap" : input_lis[0],
        "weight" : input_lis[1],
        "bias" : input_lis[2],
        "offset_w" : input_lis[3],
        "output" : input_lis[4],
        "strides" : attr_lis[0],
        "pads" : attr_lis[1],
        "dilations" : attr_lis[2],
        "groups" : attr_lis[3],
        "data_format" : attr_lis[4],
        "kernel_name" : attr_lis[5]
    }
    if option_lis:
        if isinstance(option_lis[0], dict):
            compute_input_dict["options"] = option_lis[0].get("options")
    return compute_input_dict


@tbe_register.register_param_generalization("Conv3D")
def conv3d_generalization(fmap,
                          weight,
                          bias,
                          offset_w,
                          output,
                          strides,
                          pads,
                          dilations=(1, 1, 1, 1, 1),
                          groups=1,
                          data_format="NDHWC",
                          offset_x=0,
                          kernel_name="conv3d",
                          generalize_config=None):
    """
    algorithm: conv3d_generalization

    Notice
    ------
    run after infershape and before operator compile
    only modify input and output tensors with range

    for use:
        1. te fusion distinguish .o (remove the generalization dim)
        2. pass them to the operator to follow the dynanmic shape process

    Parameters
    ----------
    fmap: A dict with keys(shape and dtype)
    Input 5d feature map tensor

    weight: A dict with keys(shape and dtype)
    Input 5d weight tensor

    bias: A dict with keys(shape and dtype) or None
    Input bias tensor

    offset_w: A dict with keys(shape and dtype) or None
    Input offset_w tensor

    output: A dict with keys(shape and dtype)
    Output tensor, dtype must be assigned

    strides: A tuple/list of 5 integers, format sensitive
    [strides_batch, strides_depth, strides_height, strides_width, strides_channel]

    pads: A tuple/list of 6 integers
    [pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right]

    dilations: A tuple/list of 5 integers
    Dilation on D/H/W, format sensitive, default value is (1, 1, 1, 1, 1)

    groups: Int of blocked connections from input channels to output channels
    Default value is 1

    data_format: The data format of the input and output data
    Default format is "NDHWC"

    offset_x: Int
    Input offset_x value, default value is 0

    kernel_name: Str
    Kernel name, default value is "conv3d"

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
    if generalize_config.get("mode") == "keep_rank":  # fuzz build situation
        is_dynamic_fuzz_mode = util_conv3d.check_fuzz_dynamic_mode(fmap)
        check_result = util_conv3d.check_para_fuzz_compile_3d(fmap, output, weight, dilations, strides, pads,
                                                            is_dynamic_fuzz_mode, _OP_TYPE, 0)
        if check_result:
            return check_result
        is_exceed_l1_lst = _check_l1_size(fmap, weight, pads, strides, dilations, is_dynamic_fuzz_mode)
        if is_exceed_l1_lst:
            warnings.warn("Conv3d generalization fuzz build exceed l1 buffer size.")
            return is_exceed_l1_lst
        fmap["ori_range"], fmap["ori_shape"] = generalize_shape_and_range(fmap.get("ori_format"), fmap.get("ori_shape"))
        weight["ori_range"], weight["ori_shape"] = \
            generalize_shape_and_range(weight.get("ori_format"), weight.get("ori_shape"))
        if bias:
            bias["ori_range"], bias["ori_shape"] = \
                generalize_shape_and_range(bias.get("ori_format"), bias.get("ori_shape"))
        try:
            input_output_info = _check_input_output_para(fmap, weight, bias, offset_w, output)
            attr_info = _check_attr_para(strides, pads, dilations, groups, input_output_info.get("binary_flag"))
            _config_para(input_output_info, attr_info)
        except RuntimeError as exc:
            return util_conv3d.UNSUPPORT_LIST
        finally:
            pass
        result.append([fmap, weight, bias, offset_w, output, {"strides": strides},
                    {"pads": pads}, {"dilations": dilations}, {"groups": groups},
                    {"data_format": data_format}, {"offset_x": offset_x}])
    else:
        fmap["range"], fmap["shape"] = generalize_shape_and_range(fmap.get("format"), fmap.get("shape"))
        weight["range"], weight["shape"] = \
            generalize_shape_and_range(weight.get("format"), weight.get("shape"))
        if bias:
            bias["range"], bias["shape"] = generalize_shape_and_range(bias.get("format"), bias.get("shape"))
            bias["ori_format"] = "ND"
            bias["format"] = "ND"
        output["range"], output["shape"] = generalize_shape_and_range(output.get("format"), output.get("shape"))
        strides = [-1, -1, -1, -1, -1]
        pads = [-1, -1, -1, -1, -1, -1]
        dilations = [-1, -1, -1, -1, -1]
        groups = -1
        # change formt to ensure reuse
        fmap["ori_format"] = "NCDHW"
        weight["ori_format"] = "NCDHW"
        output["ori_format"] = "NCDHW"
        data_format = "NCDHW"
        result.append([fmap, weight, bias, offset_w, output, strides, pads, dilations, groups, data_format, offset_x])
    return result


@register_operator("Conv3D")
@para_check.check_op_params(
    para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
    para_check.OPTION_INPUT, para_check.OPTION_INPUT,
    para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_STR,
    para_check.REQUIRED_ATTR_INT, para_check.KERNEL_NAME)
def conv3d(fmap,
           weight,
           bias,
           offset_w,
           output,
           strides,
           pads,
           dilations=(1, 1, 1, 1, 1),
           groups=1,
           data_format="NDHWC",
           offset_x=0,
           kernel_name="conv3d"):
    """
    algorithm: conv3d

    Parameters
    ----------
    fmap: A dict with keys(shape and dtype)
    Input 5d feature map tensor

    weight: A dict with keys(shape and dtype)
    Input 5d weight tensor

    bias: A dict with keys(shape and dtype) or None
    Input bias tensor

    offset_w: A dict with keys(shape and dtype) or None
    Input offset_w tensor

    output: A dict with keys(shape and dtype)
    Output tensor, dtype must be assigned

    strides: A tuple/list of 5 integers, format sensitive
    [strides_batch, strides_depth, strides_height, strides_width, strides_channel]

    pads: A tuple/list of 6 integers
    [pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right]

    dilations: A tuple/list of 5 integers
    Dilation on D/H/W, format sensitive, default value is (1, 1, 1, 1, 1)

    groups: Int of blocked connections from input channels to output channels
    Default value is 1

    data_format: The data format of the input and output data
    Default format is "NDHWC"

    offset_x: Int
    Input offset_x value, default value is 0

    kernel_name: Str
    Kernel name, default value is "conv3d"

    Returns
    -------
    None
    """
    # Whether it is a binary scenario
    fmap_range = fmap.get("range") if fmap.get("range") else fmap.get("ori_range")
    binary_flag = check_binary_flag(fmap_range, weight.get("shape"))

    # Do classify
    input_list = [fmap, weight, bias, offset_w, output]
    attr_list = [strides, pads, dilations, groups, data_format, kernel_name]
    extra_parameters = {"binary_flag": binary_flag}
    ins = (input_list, attr_list)
    classified_ins = classify(ins, extra_parameters)

    # The call is made based on the result of classify
    sch_list = []
    tensor_list = []
    for compute_input in classified_ins:
        compute_input_dict = _config_compute_input_dict(compute_input)
        with tbe.compute():
            res = _conv3d_compute(compute_input_dict)

        with tvm.target.cce():
            sch = tbe.auto_schedule(res.get("op_res"))
        # get real output tensor
        real_out = res.get('op_res')[0]
        tensor_list.append(res.get('op_placeholder') + [real_out])
        sch_list.append(sch)

    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list,
        "build_args": {"constant_realize_extent_in_infer_bound": False}
    }
    if get_op_context().get_addition("is_dynamic_constantization") is True:
        get_op_context().set_op_mode("static")
    tbe.build(sch_list, config)
