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
conv3d
"""
from impl.util import util_select_op_base
from impl.util import util_common
from impl.util.platform_adapter import error_manager_util
from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util import util_conv3d

# [strides_batch, strides_depth, strides_height,
#  strides_width, strides_channel]
_STRIDE_LENGTH = 5

_DILATION_LENGTH = 5
_PADS_LENGTH = 6
# NDHWC or NCDHW
_SHAPE_DIMS = 5
_C0 = 16
_L1FUSION_INPUT_CTR = 2

# filterD must be in [1,255]
_FILTER_DHW_MIN = 1
_FILTER_DHW_MAX = 255
# pad must be in [0,255]
_PAD_MIN = 0
_PAD_MAX = 255
# stride must be in [1,63]
_STRIDE_MIN = 1
_STRIDE_MAX = 63

# fmap H and W must be in [1, 4096]
_FMAP_HW_MIN = 1
_FMAP_HW_MAX = 4096
# dilations must be in [1,255]
_DILATION_MIN = 1
_DILATION_MAX = 255

_FMAP_TARGET_FORMAT = "NCDHW"
_FMAP_FORMAT_WHITE_LIST = ["NCDHW", "NDHWC"]
_FILTER_TARGET_FORMAT = "NCDHW"
_FILTER_FORMAT_WHITE_LIST = ["NCDHW", "NDHWC", "DHWCN"]

_DTYPE_SIZE = {"float16": 2, "float32": 4, "int32": 4}
_ALIGN_BYTE = 32
_DEFAULT_FP16_SIZE = 2


def _transform_shape_with_format(src_format, to_format, ori_shape, format_white_list):
    # input format is not expected
    if ((src_format not in format_white_list) or
        (to_format not in format_white_list)):
        return ""
    # need not to transform
    if src_format == to_format:
        return ori_shape
    res_shape = [1 for _ in range(len(to_format))]
    for i in range(_SHAPE_DIMS):
        for j in range(_SHAPE_DIMS):
            if to_format[i] == src_format[j]:
                res_shape[i] = ori_shape[j]
    return res_shape


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
        overlap_d = -1 if (filter_d - 1) * dilation_d + 1 <= strides_d else 0
        overlap_h = -1 if (filter_h - 1) * dilation_h + 1 <= strides_h else 0
        overlap_w = -1 if (filter_w - 1) * dilation_w + 1 <= strides_w else 0

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
    filter_shape = _transform_shape_with_format(weight.get("ori_format"),
                                                _FILTER_TARGET_FORMAT,
                                                weight.get("ori_shape"),
                                                _FILTER_FORMAT_WHITE_LIST)
    if not filter_shape:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'weight',
            'expected_format_list': ",".join(_FILTER_FORMAT_WHITE_LIST),
            'format': weight.get("ori_format")
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    strides_formated = _transform_shape_with_format(fmap.get("ori_format"),
                                                    _FMAP_TARGET_FORMAT,
                                                    strides,
                                                    _FMAP_FORMAT_WHITE_LIST)
    if not strides_formated:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'strides',
            'expected_format_list': ",".join(_FMAP_FORMAT_WHITE_LIST),
            'format': fmap.get("ori_format")
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    dilations_formated = _transform_shape_with_format(fmap.get("ori_format"),
                                                      _FMAP_TARGET_FORMAT,
                                                      dilations,
                                                      _FMAP_FORMAT_WHITE_LIST)

    _, _, filter_d, filter_h, filter_w = filter_shape
    _, strides_d, strides_h, strides_w, _ = strides_formated
    _, dilation_d, dilation_h, dilation_w, _ = dilations_formated

    axis_split_info, axis_reduce_info = _get_slice_info()

    op_cal_info_in_json = util_select_op_base.get_op_cal_info(axis_split_info,
                                                              axis_reduce_info,
                                                              _L1FUSION_INPUT_CTR,
                                                              0)
    return op_cal_info_in_json


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


def _conv3d_compute(shape_fm,
                    shape_filter,
                    bias,
                    stride_dhw,
                    pads,
                    fmp_dtype,
                    w_dtype,
                    res_dtype,
                    dilation_dhw=None,
                    group_dict=None,
                    offset_x=0,
                    kernel_name='conv3d'):
    """
    algorithm: compute conv3d

    Parameters
    ----------
    shape_fm: The shape of feature,
        A list/tuple of 'int' that has length `== 5`

    shape_filter: The shape of filter, a list of 'int' that has length `== 5`

    bias: A dict with keys(shape and dtype) or None
        An input bias tensor

    stride_dhw: A tuple/list of `ints` that has length `== 3`

    pads: A tuple/list of 6 integers
        [pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right]

    fmp_dtype: The dtype of feature

    w_dtype: The dtype of filter

    res_dtype: The dtype of output

    dilation_dhw: A tuple/list of `ints` that has length `==3`

    group_dict: Dict
        Group convolution related information

    kernel_name: Str
        Kernel name, default value is "conv3d"

    Returns
    -------
    list of tensor
    """
    if dilation_dhw is None:
        dilation_dhw = [1, 1, 1]
    real_g = group_dict["real_g"]
    cin1_g = group_dict["cin1_g"]
    cout_g = group_dict["cout_g"]
    cout_ori = group_dict["cout_ori"]

    # C and Cout align 16
    shape_fm = list(shape_fm)
    fmp_block_k = tbe_platform.CUBE_MKN[fmp_dtype]['mac'][1]
    shape_fm[1] = ((shape_fm[1] + fmp_block_k - 1) // fmp_block_k) * fmp_block_k
    w_block_k = tbe_platform.CUBE_MKN[w_dtype]['mac'][1]
    shape_filter = list(shape_filter)
    shape_filter[1] = ((shape_filter[1] + w_block_k - 1) // w_block_k) * w_block_k
    w_block_n = tbe_platform.CUBE_MKN[w_dtype]['mac'][2]
    shape_filter[0] = ((shape_filter[0] + w_block_n - 1) // w_block_n) * w_block_n

    batch, cin, fmp_d, fmp_h, fmp_w = shape_fm
    fmp_block_k = tbe_platform.CUBE_MKN[fmp_dtype]['mac'][1]
    shape_fmp_ndc1hwc0 = (batch, fmp_d, cin // fmp_block_k, fmp_h, fmp_w, fmp_block_k)

    _, _, w_d, w_h, w_w = shape_filter
    w_block_k = tbe_platform.CUBE_MKN[w_dtype]['mac'][1]
    w_block_n = tbe_platform.CUBE_MKN[w_dtype]['mac'][2]


    shape_w_frac_z = (real_g * w_d * cin1_g * w_h * w_w, cout_g // w_block_n,
                      w_block_n, w_block_k)

    mad_dtype = _get_mad_dtype(w_dtype)

    data = tvm.placeholder(shape_fmp_ndc1hwc0, name='Fmap', dtype=fmp_dtype)
    weight = tvm.placeholder(shape_w_frac_z, name='Filter', dtype=w_dtype)

    bias_tensor = None
    if bias is not None:
        align_mod = _ALIGN_BYTE // _DTYPE_SIZE.get(res_dtype, _DEFAULT_FP16_SIZE)
        bias_align_shape = (cout_ori + align_mod - 1) // align_mod * align_mod
        bias_tensor = tvm.placeholder((bias_align_shape,),
                                      name='bias_tensor',
                                      dtype=res_dtype,
                                      attrs={'ori_shape': [cout_ori, ]})
    para_dict = {
        "dsl_flag": False,
        "bias_tensor": bias_tensor,
        "pads": pads,
        "strides": stride_dhw,
        "res_dtype": res_dtype,
        "mad_dtype": mad_dtype,
        "kernel_name": kernel_name,
        "group_dict": group_dict,
        "dilation_dhw": dilation_dhw,
        "offset_x": offset_x
    }

    conv_res = tbe.conv3d(data, weight, shape_filter, para_dict)

    if bias:
        tensor_list = [data, weight, bias_tensor, conv_res]
    else:
        tensor_list = [data, weight, conv_res]

    return tensor_list


def _check_conv3d_dtype(fmp_dtype, w_dtype, res_dtype):
    """
    algorithm: Check the input dtype of conv3d

    Parameters
    ----------

    fmp_dtype: The dtype of feature

    w_dtype: The dtype of filter

    res_dtype: The dtype of output

    Returns
    -------
    None
    """
    para_check.check_dtype_rule(fmp_dtype, ('float16', 'int8'), "fmap")
    para_check.check_dtype_rule(w_dtype, ('float16', 'int8'), "filter")
    para_check.check_dtype_rule(res_dtype, ('float16', 'float32', 'int32'), "output")


def _format_normalize(fmp_format, w_format, fmp_shape, w_shape, strides,
                      dilations):
    """
    algorithm: unified format

    Parameters
    ----------
    fmp_format: The data format of the input feature

    w_format: The data format of the input filter

    fmp_shape: The shape of feature
        A list/tuple of 'int' that has length `== 5`

    w_shape: The shape of filter, a list of 'int' that has length `== 5`

    strides: A tuple/list of `ints` that has length `== 5`

    dilations: A tuple/list of 5 integers
        Dilation on D/H/W, format sensitive
        Dilations in the batch and depth dimensions must be 1

    Returns
    -------
    shape_fm, shape_filter, stride_dhw, dilation_dhw
    """
    shape_fm = _transform_shape_with_format(fmp_format,
                                            _FMAP_TARGET_FORMAT,
                                            fmp_shape,
                                            _FMAP_FORMAT_WHITE_LIST)
    stride_full = _transform_shape_with_format(fmp_format,
                                               _FMAP_TARGET_FORMAT,
                                               strides,
                                               _FMAP_FORMAT_WHITE_LIST)
    dilation_full = _transform_shape_with_format(fmp_format,
                                                 _FMAP_TARGET_FORMAT,
                                                 dilations,
                                                 _FMAP_FORMAT_WHITE_LIST)

    if not shape_fm or not stride_full or not dilation_full:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'input',
            'expected_format_list': ",".join(_FMAP_FORMAT_WHITE_LIST),
            'format': fmp_format
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    shape_filter = _transform_shape_with_format(w_format,
                                                _FILTER_TARGET_FORMAT,
                                                w_shape,
                                                _FILTER_FORMAT_WHITE_LIST)
    if not shape_filter:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'weight',
            'expected_format_list': ",".join(_FILTER_FORMAT_WHITE_LIST),
            'format': w_format
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    return shape_fm, shape_filter, stride_full[2:], dilation_full[2:]


def _check_input_param(fmp_shape, w_shape, fmp_dtype, w_dtype, res_dtype,
                       fmp_format, w_format, bias, strides, pads, dilations,
                       groups):
    """
    algorithm: Check the input params of conv3d

    Parameters
    ----------
    fmp_shape: The shape of feature
        A list/tuple of 'int' that has length `== 5`

    w_shape: The shape of filter
        A list/tuple of 'int' that has length `== 5`

    fmp_dtype: The dtype of feature

    w_dtype: The dtype of filter

    res_dtype: The dtype of output

    fmp_format: The data format of the input feature

    w_format: The data format of the input filter

    bias: A dict with keys(shape and dtype) or None
        input bias tensor

    strides: A list/tuple of `ints` that has length `== 5`

    pads: A list/tuple of 6 integers
        [pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right]

    dilations: A list/tuple of 5 integers
        Dilation on D/H/W, format sensitive
        Dilations in the batch and depth dimensions must be 1

    groups: int
        Group convolution parameter

    Returns
    -------
    """
    if bias:
        util_conv3d.check_bias(bias, res_dtype)
        bias_dtype = bias.get("dtype").lower()
        para_check.check_dtype_rule(bias_dtype, ("float16", "float32", "int32"), "bias")

    if len(strides) != _STRIDE_LENGTH:
        dict_args = {
            'errCode': 'E60006',
            'param_name': 'strides',
            'expected_length': '5',
            'length': '{}'.format(len(strides))
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))
    if len(dilations) != _DILATION_LENGTH:
        dict_args = {
            'errCode': 'E60006',
            'param_name': 'dilations',
            'expected_length': '5',
            'length': '{}'.format(len(dilations))
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if len(pads) != _PADS_LENGTH:
        error_manager_cube.raise_err_one_para('E62501', 'conv3d', 'pads')

    para_check.check_shape_rule(fmp_shape, min_dim=_SHAPE_DIMS,
                                max_dim=_SHAPE_DIMS)
    para_check.check_shape_rule(w_shape, min_dim=_SHAPE_DIMS,
                                max_dim=_SHAPE_DIMS)

    # normalized format as NCDHW
    shape_fm, shape_filter, stride_dhw, dilation_dhw = _format_normalize(
        fmp_format, w_format, fmp_shape, w_shape, strides, dilations)
    dilation_d, _, _ = dilation_dhw
    if dilation_d != 1:
        dict_args = {
            'errCode': 'E60038',
            'desc': 'dilation in D dimension only supports 1',
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    w_block_k = tbe_platform.CUBE_MKN[w_dtype]['mac'][1]
    group_dict = util_common.calculate_group(shape_fm[1], shape_filter[0], groups, _C0, w_block_k)

    _check_conv3d_dtype(fmp_dtype, w_dtype, res_dtype)

    _check_groups_validation(shape_fm[1], shape_filter[1], groups)

    _check_conv3d_shape(shape_fm, shape_filter, pads, stride_dhw,
                        dilation_dhw, fmp_dtype, w_dtype)

    return shape_fm, shape_filter, stride_dhw, dilation_dhw, group_dict


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
        dict_args = {
            'errCode': 'E60010',
            'channel_of_x': str(fmap_cin),
            'channel_of_filter': str(filter_cin * groups)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))


def _check_conv3d_shape(shape_fm, shape_filter, pads, stride_dhw, dilation_dhw,
                        fmp_dtype, w_dtype):
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

    Returns
    -------
    None
    """
    _, _, fmap_d, fmap_h, fmap_w = shape_fm
    _, _, filter_d, filter_h, filter_w = shape_filter

    pad_d = [pads[0], pads[1]]
    _check_d_dimension(fmap_d, filter_d, pad_d, stride_dhw[0], dilation_dhw[0])

    pad_h = [pads[2], pads[3]]
    _check_h_dimension(fmap_h, filter_h, pad_h, stride_dhw[1], dilation_dhw[1])

    pad_w = [pads[4], pads[5]]
    _check_w_dimension(fmap_w, filter_w, pad_w, stride_dhw[2], dilation_dhw[2])

    # C dimension should align 16
    block_size_k = tbe_platform.CUBE_MKN[fmp_dtype]['mac'][1]
    block_size_m = tbe_platform.CUBE_MKN[fmp_dtype]['mac'][0]

    # calculated by h_i and w_i
    _, dilation_h, dilation_w = dilation_dhw
    filter_dilated_h = (filter_h - 1) * dilation_h + 1
    filter_dilated_w = (filter_w - 1) * dilation_w + 1

    # check for not bigger than L1
    l1_buffer_size = tbe_platform.get_soc_spec("L1_SIZE")
    m_bit_ratio = {"float16": 2, "int8": 1}
    point_per_w = ((fmap_w - filter_dilated_w) +
                   pad_w[0] + pad_w[1]) // stride_dhw[2] + 1
    w_in = block_size_m // point_per_w + 2
    tmp = ((w_in - 1) * stride_dhw[1] + filter_dilated_h) * fmap_w
    max_feature_map_l1 = block_size_k * tmp * m_bit_ratio[w_dtype]

    if max_feature_map_l1 > l1_buffer_size:
        dict_args = {
            'errCode': 'E60026',
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))


def _check_d_dimension(fmap_d, filter_d, pad_d, stride_d, dilation_d):
    filter_dilated_d = (filter_d - 1) * dilation_d + 1
    if filter_d < _FILTER_DHW_MIN or filter_d > _FILTER_DHW_MAX:
        error_manager_cube.raise_err_four_paras('E62003', 'conv3d', 'weight', 'D',
            '[{}, {}]'.format(_FILTER_DHW_MIN, _FILTER_DHW_MAX), str(filter_d))

    if (fmap_d + pad_d[0] + pad_d[1]) < filter_dilated_d:
        dict_args = {
            'errCode': 'E60012',
            'depth_of_x': str(fmap_d + pad_d[0] + pad_d[1]),
            'depth_of_filter': str(filter_dilated_d),
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if pad_d[0] < _PAD_MIN or pad_d[1] < _PAD_MIN or pad_d[0] > _PAD_MAX or pad_d[1] > _PAD_MAX:
        error_manager_cube.raise_err_four_paras('E62003', 'conv3d', 'pad', 'D',
            '[{}, {}]'.format(_PAD_MIN, _PAD_MAX),
            'pad_d[0] = {}, pad_d[1] = {}'.format(pad_d[0], pad_d[1]))

    if pad_d[0] >= filter_dilated_d or pad_d[1] >= filter_dilated_d:
        dict_args = {
            'errCode': 'E60013',
            'depth_of_pad': 'pad_d[0] = {}, pad_d[1] = {}'.format(pad_d[0],
                                                                  pad_d[1]),
            'depth_of_filter': str(filter_dilated_d)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if stride_d < _STRIDE_MIN or stride_d > _STRIDE_MAX:
        error_manager_cube.raise_err_four_paras('E62003', 'conv3d', 'stride', 'D',
            '[{}, {}]'.format(_STRIDE_MIN, _STRIDE_MAX),
            str(stride_d))

    if dilation_d < _DILATION_MIN or dilation_d > _DILATION_MAX:
        error_manager_cube.raise_err_four_paras('E62003', 'conv3d', 'dilation', 'D',
            '[{}, {}]'.format(_DILATION_MIN, _DILATION_MAX),
            str(dilation_d))


def _check_h_dimension(fmap_h, filter_h, pad_h, stride_h, dilation_h):
    filter_dilated_h = (filter_h - 1) * dilation_h + 1
    if fmap_h < _FMAP_HW_MIN or fmap_h > _FMAP_HW_MAX:
        error_manager_cube.raise_err_four_paras('E62003', 'conv3d', 'input', 'H',
            '[{}, {}]'.format(_FMAP_HW_MIN, _FMAP_HW_MAX),
            str(fmap_h))

    if filter_h < _FILTER_DHW_MIN or filter_h > _FILTER_DHW_MAX:
        error_manager_cube.raise_err_four_paras('E62003', 'conv3d', 'filter', 'H',
            '[{}, {}]'.format(_FILTER_DHW_MIN, _FILTER_DHW_MAX),
            str(filter_h))


    def _check_pad_h():
        if pad_h[0] < _PAD_MIN or pad_h[1] < _PAD_MIN or pad_h[0] > _PAD_MAX or pad_h[1] > _PAD_MAX:
            error_manager_cube.raise_err_four_paras('E62003', 'conv3d', 'pad', 'H',
                '[{}, {}]'.format(_PAD_MIN, _PAD_MAX),
                'pad_h[0] = {}, pad_h[1] = {}'.format(pad_h[0], pad_h[1]))

        if pad_h[0] >= filter_dilated_h or pad_h[1] >= filter_dilated_h:
            dict_args = {
                'errCode': 'E60016',
                'h_of_filter': str(filter_dilated_h),
                'h_of_pad': '[pad_h[0]={}, pad_h[1]={}]'.format(pad_h[0], pad_h[1])
            }
            raise RuntimeError(dict_args,
                            error_manager_util.get_error_message(dict_args))

    _check_pad_h()
    if (fmap_h + pad_h[0] + pad_h[1]) < filter_dilated_h:
        # Chip Design demand, Load3D
        dict_args = {
            'errCode': 'E60014',
            'h_of_x': str(fmap_h + pad_h[0] + pad_h[1]),
            'h_of_filter': str(filter_dilated_h)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if stride_h < _STRIDE_MIN or stride_h > _STRIDE_MAX:
        error_manager_cube.raise_err_four_paras('E62003', 'conv3d', 'stride', 'H',
            '[{}, {}]'.format(_STRIDE_MIN, _STRIDE_MAX),
            'stride_h = {}'.format(stride_h))

    if dilation_h < _DILATION_MIN or dilation_h > _DILATION_MAX:
        error_manager_cube.raise_err_four_paras('E62003', 'conv3d', 'dilation', 'H',
            '[{}, {}]'.format(_DILATION_MIN, _DILATION_MAX),
            str(dilation_h))


def _check_w_dimension(fmap_w, filter_w, pad_w, stride_w, dilation_w):
    filter_dilated_w = (filter_w - 1) * dilation_w + 1
    if fmap_w < _FMAP_HW_MIN or fmap_w > _FMAP_HW_MAX:
        error_manager_cube.raise_err_four_paras('E62003', 'conv3d', 'input', 'W',
            '[{}, {}]'.format(_FMAP_HW_MIN, _FMAP_HW_MAX),
            str(fmap_w))

    if filter_w < _FILTER_DHW_MIN or filter_w > _FILTER_DHW_MAX:
        error_manager_cube.raise_err_four_paras('E62003', 'conv3d', 'filter', 'W',
            '[{}, {}]'.format(_FILTER_DHW_MIN, _FILTER_DHW_MAX),
            str(filter_w))


    def _check_pad_w():
        if pad_w[0] < _PAD_MIN or pad_w[1] < _PAD_MIN or pad_w[0] > _PAD_MAX or pad_w[1] > _PAD_MAX:
            error_manager_cube.raise_err_four_paras('E62003', 'conv3d', 'pad', 'W',
                '[{}, {}]'.format(_PAD_MIN, _PAD_MAX),
                'pad_w[0] = {}, pad_w[1] = {}'.format(pad_w[0], pad_w[1]))

        if pad_w[0] >= filter_dilated_w or pad_w[1] >= filter_dilated_w:
            dict_args = {
                'errCode': 'E60017',
                'w_of_filter': str(filter_dilated_w),
                'w_of_pad': '[pad_w[0]={}, pad_w[1]={}]'.format(pad_w[0], pad_w[1])
            }
            raise RuntimeError(dict_args,
                            error_manager_util.get_error_message(dict_args))

    _check_pad_w()
    if filter_dilated_w > (fmap_w + pad_w[0] + pad_w[1]):
        # Chip Design demand, Load3D
        dict_args = {
            'errCode': 'E60015',
            'w_of_x': str(fmap_w + pad_w[0] + pad_w[1]),
            'w_of_filter': str(filter_dilated_w)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if stride_w < _STRIDE_MIN or stride_w > _STRIDE_MAX:
        error_manager_cube.raise_err_four_paras('E62003', 'conv3d', 'stride', 'W',
            '[{}, {}]'.format(_STRIDE_MIN, _STRIDE_MAX),
            str(stride_w))

    if dilation_w < _DILATION_MIN or dilation_w > _DILATION_MAX:
        error_manager_cube.raise_err_four_paras('E62003', 'conv3d', 'dilation', 'W',
            '[{}, {}]'.format(_DILATION_MIN, _DILATION_MAX),
            str(dilation_w))


def _cal_input_param(fmap,
                     weight,
                     bias_tensor,
                     output,
                     strides,
                     pads,
                     dilations,
                     groups,
                     data_format,
                     offset_x=0,
                     kernel_name="conv3d"):
    """
    to calculate fusion param
    """
    shape_fmap = []
    for i in fmap.op.attrs['ori_shape']:
        shape_fmap.append(i.value)

    shape_filter = []
    for i in weight.op.attrs['ori_shape']:
        shape_filter.append(i.value)

    res_dtype = output.get("dtype").lower()
    mad_dtype = _get_mad_dtype(weight.dtype)

    w_format = weight.op.attrs['ori_format']
    # NCDHW
    shape_fmap, shape_filter, stride_dhw, dilation_dhw = _format_normalize(
        data_format, w_format, shape_fmap, shape_filter, strides, dilations)

    _check_groups_validation(shape_fmap[1], shape_filter[1], groups)

    w_block_k = tbe_platform.CUBE_MKN[weight.dtype]['mac'][1]
    group_dict = util_common.calculate_group(shape_fmap[1], shape_filter[0], groups, _C0, w_block_k)

    para_dict = {
        "dsl_flag": True,
        "bias_tensor": bias_tensor,
        "pads": pads,
        "strides": stride_dhw,
        "res_dtype": res_dtype,
        "mad_dtype": mad_dtype,
        "kernel_name": kernel_name,
        "group_dict": group_dict,
        "dilation_dhw": dilation_dhw,
        "offset_x": offset_x
    }

    return para_dict, shape_filter


def check_supported(fmap,
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
    The H and W dimension of dilation should be in range [1, 255]. \n
    The D,H or W dimension of the filter should be in range [1, 255]. \n
    The padding in each dimension should be in range [0, 255]. \n
    The D,H or W dimension of the stride should be in range [1, 63]. \n

    The groups should <= the feature map's and the filter's channel dimension. \n
    Feature map's channel dimension or filter's channel dimension must be divisible by groups. \n
    The channel dimension of the feature map should = filter's channel dimension * groups. \n
    The D,H or W dimension of the feature map after padding should >= the filter's corresponding \
    dimension after dilation. \n
    The padding in each dimension should < the filter's corresponding dimension after dilation. \n

    If the output H dimension is not 1, the output W dimension should >= 2. \n
    The feature map size in L1 buffer should <= the chip's L1 buffer size
    """
    fmp_shape = fmap.get("ori_shape")
    fmp_dtype = fmap.get("dtype")
    fmp_format = data_format
    w_shape = weight.get("ori_shape")
    w_dtype = weight.get("dtype")
    w_format = weight.get("ori_format")
    res_dtype = output.get("dtype")

    fmp_dtype = fmp_dtype.lower()
    w_dtype = w_dtype.lower()
    res_dtype = res_dtype.lower()

    # normalized format as NCDHW
    try:
        _check_input_param(fmp_shape, w_shape, fmp_dtype, w_dtype, res_dtype, fmp_format, w_format, bias,
                           strides, pads, dilations, groups)
        return True, ""
    except Exception as e:
        reason = e.args[1]
        return False, reason
    finally:
        pass


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
    fmp_shape = fmap.get("ori_shape")
    fmp_dtype = fmap.get("dtype")
    fmp_format = data_format
    w_shape = weight.get("ori_shape")
    w_dtype = weight.get("dtype")
    w_format = weight.get("ori_format")
    res_dtype = output.get("dtype")

    fmp_dtype = fmp_dtype.lower()
    w_dtype = w_dtype.lower()
    res_dtype = res_dtype.lower()

    # normalized format as NCDHW
    shape_fm, shape_filter, stride_dhw, dilation_dhw, group_dict = _check_input_param(
        fmp_shape, w_shape, fmp_dtype, w_dtype, res_dtype, fmp_format,
        w_format, bias, strides, pads, dilations, groups)

    pads = list(pads)
    stride_dhw = list(stride_dhw)

    tensor_list = _conv3d_compute(shape_fm,
                                  shape_filter,
                                  bias,
                                  stride_dhw,
                                  pads,
                                  fmp_dtype,
                                  w_dtype,
                                  res_dtype,
                                  dilation_dhw=dilation_dhw,
                                  group_dict=group_dict,
                                  offset_x=offset_x,
                                  kernel_name=kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(tensor_list[-1])

    config = {"name": kernel_name, "tensor_list": tensor_list, "dummy_placeholder": True}
    tbe.build(sch, config)


@tbe_platform.fusion_manager.register("conv3d")
def conv3d_fusion_compute(data,
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

    para_dict, filter_size = _cal_input_param(
        data, weight, bias, output, strides, pads, dilations, groups, data_format, offset_x, kernel_name)

    res = tbe.conv3d(data, weight, filter_size, para_dict)

    return res
