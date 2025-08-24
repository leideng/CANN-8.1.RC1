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
avg_pool3d_grad_d
"""
from impl.conv3d_backprop_input_d import conv3d_backprop_input_fusion_compute
from impl.util import util_conv3d
from impl.util import util_select_op_base
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import error_manager_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm

_BLOCK_SIZE = 16
_C0_SIZE = tbe_platform.C0_SIZE
_UB_FUSED_OP_NUM = 2
_KSIZE_DIM = 5
_STRIDES_DIM = 5
_PADS_DIM = 6
_FMAP_TARGET_FORMAT = "NDHWC"
_GRADS_TARGET_FORMAT = "NDHWC"
_GRADS_FORMAT_WHITE_LIST = ["NCDHW", "NDHWC"]
_DATA_FORMAT_WHITE_LIST = ["NCDHW", "NDHWC"]
_L1FUSION_INPUT_CTR = 2
_L1_FUSION_DISABLE = 0


def get_op_support_info(grads, filter, multiplier, output,
                        orig_input_shape, ksize,
                        strides, pads, ceil_mode=False,
                        count_include_pad=True,
                        divisor_override=0,
                        data_format="NDHWC",
                        kernel_name="avg_pool3d_grad_d",
                        op_slice_info=""):
    """
    algorithm: get_op_support_info

    Parameters
    ----------
    grads : dict, shape and dtype of input_data, only support float16, shape is 5dims, format is NDC1HWC0

    filter : dict, fractal_z_3d layout, float16 dtype

    multiplier : dict, NDC1HWC0 layout, float16 dtype

    output : dict, shape and dtype of output_data, only support float16

    ksize : list or tuple, the window of avg_pool3d, only support avg_pool3d in D or H or W

    strides:list or tuple, the window of avg_pool3d, only support avg_pool3d in D or H or W

    pads : list or tuple, count of padding zero or d, h, w axis

    ceil_mode : when True, will use ceil mode instead of floor in the formula to compute the output shape

    count_include_pad : when True, will include the zero-padding in the averaging calculation

    divisor_override : if specified, it will be used as divisor, otherwise size of the pooling region will be used

    data_format : str, default value is "NDHWC"

    kernel_name : cce kernel name, default value is "avg_pool3d_grad_d"

    op_slice_info: Str, Default value is ""

    Returns
    -------
    op_cal_info_in_json: A dict with keys(split_maps, reduce_maps, l1_fusion_enable and min_tbe_l1_space)
    """
    def _cal_min_l1space():
        block_size = 16
        w_value = ori_shape_grads[3] * strides_formated[3]
        if ori_shape_res[3] > block_size:
            h_value_max = kh + 1
        elif block_size % ori_shape_res[3] == 0:
            h_value_max = kh + block_size // ori_shape_res[3] - 1
        else:
            h_value_max = kh + block_size // ori_shape_res[3] + 1

        a_l1_size = h_value_max * w_value * ((kd - 2) // strides_formated[1] + 2) * block_size * 2
        b_l1_size = kh * kw * kd * block_size * block_size * 2
        return a_l1_size + b_l1_size

    def _get_slice_info():
        overlap_d = -1 if (kd == 1 and strides_formated[1] == 1) else 0
        overlap_h = -1 if (kh == 1 and strides_formated[2] == 1) else 0
        overlap_w = -1 if (kw == 1 and strides_formated[3] == 1) else 0

        # format
        axis_split_matrix = []
        axis_reduce_list = None
        format_grads = grads.get("format")
        if format_grads == "NDC1HWC0" and not global_mode:
            if not multiplier:
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
            else:
                # cut N
                axis_split_matrix.append([util_select_op_base.SplitInput([0, [0], [-1], [-1]], [2, [0], [-1], [-1]]),
                                          util_select_op_base.SplitOutput([0, [0]])])
                # cut D
                axis_split_matrix.append([util_select_op_base.SplitInput([0, [1], [overlap_d], [overlap_d]],
                                                                         [2, [1], [overlap_d], [overlap_d]]),
                                          util_select_op_base.SplitOutput([0, [1]])])
                # cut H
                axis_split_matrix.append([util_select_op_base.SplitInput([0, [3], [overlap_h], [overlap_h]],
                                                                         [2, [3], [overlap_h], [overlap_h]]),
                                          util_select_op_base.SplitOutput([0, [3]])])
                # cut W
                axis_split_matrix.append([util_select_op_base.SplitInput([0, [4], [overlap_w], [overlap_w]],
                                                                         [2, [4], [overlap_w], [overlap_w]]),
                                          util_select_op_base.SplitOutput([0, [4]])])

        else:
            axis_split_matrix = None

        return axis_split_matrix, axis_reduce_list

    ori_shape_grads = util_conv3d.transform_shape_with_format(grads.get("ori_format"),
                                                              _GRADS_TARGET_FORMAT,
                                                              grads.get("ori_shape"),
                                                              _GRADS_FORMAT_WHITE_LIST)
    strides_formated = util_conv3d.transform_shape_with_format(data_format,
                                                               _GRADS_TARGET_FORMAT,
                                                               strides,
                                                               _GRADS_FORMAT_WHITE_LIST)

    ksize_formated = util_conv3d.transform_shape_with_format(data_format,
                                                             _GRADS_TARGET_FORMAT,
                                                             ksize,
                                                             _GRADS_FORMAT_WHITE_LIST)
    if ori_shape_grads is None:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'grads',
            'expected_format_list': ",".join(_GRADS_FORMAT_WHITE_LIST),
            'format': grads.get("ori_format")
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    ori_shape_res = util_conv3d.transform_shape_with_format(data_format,
                                                            _FMAP_TARGET_FORMAT,
                                                            orig_input_shape,
                                                            _DATA_FORMAT_WHITE_LIST)

    if ori_shape_res is None or strides_formated is None or ksize_formated is None:
        dict_args = {
            'errCode': 'E62002',
            'param_name': 'data_format',
            'expected_format_list': ",".join(_DATA_FORMAT_WHITE_LIST),
            'format': data_format
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    _, kd, kh, kw, _ = ksize_formated
    _, grads_d, grads_h, grads_w, _ = ori_shape_grads
    global_mode = (grads_d == 1 and grads_h == 1 and grads_w == 1)
    axis_split_info, axis_reduce_info = _get_slice_info()
    if global_mode:
        l1_fusion_tag = _L1_FUSION_DISABLE
        min_tbe_l1_space = 0
    else:
        l1_fusion_tag = _L1FUSION_INPUT_CTR
        min_tbe_l1_space = _cal_min_l1space()
    op_cal_info_in_json = util_select_op_base.get_op_cal_info(axis_split_info,
                                                              axis_reduce_info,
                                                              l1_fusion_tag,
                                                              min_tbe_l1_space)
    return op_cal_info_in_json


def _check_window_rule(ksize, strides, pads):
    if len(ksize) != _KSIZE_DIM:
        error_manager_vector.raise_err_input_param_range_invalid('avg_pool3d_grad_d',
                                                                 'ksize',
                                                                 _KSIZE_DIM,
                                                                 _KSIZE_DIM,
                                                                 len(ksize))
    if len(strides) != _STRIDES_DIM:
        error_manager_vector.raise_err_input_param_range_invalid('avg_pool3d_grad_d',
                                                                 'strides',
                                                                 _STRIDES_DIM,
                                                                 _STRIDES_DIM,
                                                                 len(strides))
    if len(pads) != _PADS_DIM:
        error_manager_vector.raise_err_input_param_range_invalid('avg_pool3d_grad_d',
                                                                 'pads',
                                                                 _PADS_DIM,
                                                                 _PADS_DIM,
                                                                 len(pads))


def _check_ub_limitation(input_shape, strides):
    w_value = input_shape[3] * strides[3]

    aub_size_min = input_shape[3] * _BLOCK_SIZE * 2
    aub_filling_size_min = w_value * _BLOCK_SIZE * 2
    cub_size_min = _BLOCK_SIZE * _BLOCK_SIZE * 2
    ub_size = tbe_platform.get_soc_spec("UB_SIZE")

    if (aub_size_min * _UB_FUSED_OP_NUM + aub_filling_size_min + cub_size_min) > ub_size:
        dict_args = {
            'errCode': 'E60119'
        }
        raise RuntimeError(dict_args,
                            error_manager_util.get_error_message(dict_args))


def _check_ksize_stride_val(ksize, strides):
    kn, kc = ksize[0], ksize[-1]
    strn, strc = strides[0], strides[-1]
    if kn != 1 or kc != 1:
        error_manager_cube.raise_err_specific('avg_pool3d_grad_d',
                                              "ksize's N dim and C dim only support val equal 1.")
    if strn != 1 or strc != 1:
        error_manager_cube.raise_err_specific('avg_pool3d_grad_d',
                                              "strides's N dim and C dim only support val equal 1.")


def _avg_pool3d_grad_check_rule(input_shape, input_dtype, ksize, strides, pads, kernel_name):
    para_check.check_shape(input_shape)
    para_check.check_dtype(input_dtype, ("float16", ))
    para_check.check_kernel_name(kernel_name)
    _check_window_rule(ksize, strides, pads)


def _correct_pads(input_shape, fmap_shape, ksize, strides, pads):
    _, input_d, input_h, input_w, _ = input_shape
    _, fmap_d, fmap_h, fmap_w, _ = fmap_shape
    _, ksize_d, ksize_h, ksize_w, _ = ksize
    _, stride_d, stride_h, stride_w, _ = strides
    pad_before, pad_after, pad_top, pad_bottom, pad_left, pad_right = pads

    pad_after = max((input_d - 1) * stride_d + ksize_d - fmap_d - pad_before, 0)
    pad_bottom = max((input_h - 1) * stride_h + ksize_h - fmap_h - pad_top, 0)
    pad_right = max((input_w - 1) * stride_w + ksize_w - fmap_w - pad_left, 0)

    return [pad_before, pad_after, pad_top, pad_bottom, pad_left, pad_right]


@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT,
                            para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_STR,
                            para_check.KERNEL_NAME
                            )
def avg_pool3d_grad_d(grads,
                      filter,
                      multiplier,
                      output,
                      orig_input_shape,
                      ksize,
                      strides,
                      pads,
                      ceil_mode=False,
                      count_include_pad=True,
                      divisor_override=0,
                      data_format="NDHWC",
                      kernel_name="avg_pool3d_grad_d"):
    """
    computes average pooling3d backwards gradients.

    Parameters:
    -----------

    grads : dict, shape and dtype of input_data, only support float16, shape is 5dims, format is NDC1HWC0

    filter : dict, fractal_z_3d layout, float16 dtype

    multiplier : dict, NDC1HWC0 layout, float16 dtype

    output : dict, shape and dtype of output_data, only support float16

    ksize : list or tuple, the window of avg_pool3d, only support avg_pool3d in D or H or W

    strides:list or tuple, the window of avg_pool3d, only support avg_pool3d in D or H or W

    pads : list or tuple, count of padding zero or d, h, w axis

    ceil_mode : when True, will use ceil mode instead of floor in the formula to compute the output shape

    count_include_pad : when True, will include the zero-padding in the averaging calculation

    divisor_override : if specified, it will be used as divisor, otherwise size of the pooling region will be used

    data_format : str, default value is "NDHWC"

    kernel_name : cce kernel name, default value is "avg_pool3d_grad_d"

    Returns
    -------
    None
    """

    grads_ori_format = grads.get("ori_format")
    grads_ori_shape = grads.get("ori_shape")
    grads_shape = grads.get("shape")
    grads_dtype = grads.get("dtype").lower()

    _avg_pool3d_grad_check_rule(grads_shape, grads_dtype, ksize, strides, pads, kernel_name)

    strides_formated = util_conv3d.transform_shape_with_exception(data_format,
                                                                  _GRADS_TARGET_FORMAT,
                                                                  strides,
                                                                  _DATA_FORMAT_WHITE_LIST,
                                                                  "strides")

    orig_input_shape_formated = util_conv3d.transform_shape_with_exception(data_format,
                                                                           _FMAP_TARGET_FORMAT,
                                                                           orig_input_shape,
                                                                           _DATA_FORMAT_WHITE_LIST,
                                                                           "ori_input_shape")

    ksize_formated = util_conv3d.transform_shape_with_exception(data_format,
                                                                _GRADS_TARGET_FORMAT,
                                                                ksize,
                                                                _DATA_FORMAT_WHITE_LIST,
                                                                "ksize")

    grads_ori_shape_formated = util_conv3d.transform_shape_with_exception(grads_ori_format,
                                                                          _GRADS_TARGET_FORMAT,
                                                                          grads_ori_shape,
                                                                          _GRADS_FORMAT_WHITE_LIST,
                                                                          "grads")
    _check_ksize_stride_val(ksize_formated, strides_formated)

    _check_ub_limitation(grads_ori_shape_formated, strides_formated)

    if ceil_mode:
        pads = _correct_pads(grads_ori_shape_formated, orig_input_shape_formated,
                             ksize_formated, strides_formated, pads)

    _, kd, kh, kw, _ = ksize_formated
    on, od, oh, ow, oc = orig_input_shape_formated

    grads = tvm.placeholder(grads_shape,
                            name="grads",
                            dtype=grads_dtype,
                            attrs={"ori_shape": grads_ori_shape,
                                   "ori_format": grads_ori_format,
                                   "data_type": "float16"})

    # global mode
    if (kd >= od + pads[0] + pads[1] and
            kh >= oh + pads[2] + pads[3] and
            kw >= ow + pads[4] + pads[5]):
        if (grads_ori_shape[grads_ori_format.find('D')] != 1 or
                grads_ori_shape[grads_ori_format.find('H')] != 1 or
                grads_ori_shape[grads_ori_format.find('W')] != 1):
            error_detail = "when global mode, " \
                           "the d-axis, h-axis and w-axis of input_grad must be 1."
            error_manager_vector.raise_err_input_shape_invalid(kernel_name,
                                                               "grads_ori_shape",
                                                               error_detail)

        kd = min(kd, od + pads[0] + pads[1])
        kh = min(kh, oh + pads[2] + pads[3])
        kw = min(kw, ow + pads[4] + pads[5])

        if divisor_override:
            kernel_size_reciprocal = 1.0 / divisor_override
        elif count_include_pad:
            kernel_size_reciprocal = 1.0 / (kh * kw * kd)
        else:
            kernel_size_reciprocal = 1.0 / (oh * ow * od)

        with tvm.target.cce():
            grad_tmp = tbe.vmuls(tbe.cast_to(grads, "float32"),
                                 kernel_size_reciprocal)
            if grads_dtype == "float16":
                grad_tmp = tbe.cast_to(grad_tmp, "float16")

            output_shape = (on, od, (oc + _C0_SIZE - 1) // _C0_SIZE, oh, ow, _C0_SIZE)

            res = tbe.broadcast(grad_tmp, output_shape)

            tensor_list = [grads, res]

            sch = tbe.auto_schedule(tensor_list[-1])

        config = {"name": kernel_name,
                  "tensor_list": tensor_list}
        tbe.build(sch, config)
        return

    # cube mode
    dilations = (1, 1, 1, 1, 1)
    offset_w = None
    bias = None
    fmap_c = oc
    c1 = grads_shape[2]
    w_ori_shape = (kd, kh, kw, 1, fmap_c)
    filter_frac_z = (c1 * kd * kh * kw, 1, _C0_SIZE, _C0_SIZE)
    filter = tvm.placeholder(filter_frac_z,
                             name="filter",
                             dtype="float16",
                             attrs={"ori_shape": w_ori_shape,
                                    "ori_format": "DHWCN",
                                    "data_type": "float16"})
    if multiplier:
        mul_shape = multiplier.get("shape")
        multiplier = tvm.placeholder(mul_shape,
                                     name="multiplier",
                                     dtype="float16")
        mul_res = tbe.vmul(grads, multiplier)
        mul_res.op.attrs['ori_format'] = grads_ori_format
        mul_res.op.attrs['shape'] = grads_shape
        mul_res.op.attrs['ori_shape'] = grads_ori_shape

        res = conv3d_backprop_input_fusion_compute(filter,
                                                   mul_res,
                                                   output,
                                                   orig_input_shape,
                                                   strides,
                                                   pads,
                                                   dilations,
                                                   groups=fmap_c,
                                                   data_format=data_format,
                                                   kernel_name=kernel_name)
        tensor_list = [grads, filter, multiplier, res]
    else:
        res = conv3d_backprop_input_fusion_compute(filter,
                                                   grads,
                                                   output,
                                                   orig_input_shape,
                                                   strides,
                                                   pads,
                                                   dilations,
                                                   groups=fmap_c,
                                                   data_format=data_format,
                                                   kernel_name=kernel_name)
        tensor_list = [grads, filter, res]

    with tvm.target.cce():
        sch = tbe.auto_schedule(tensor_list[-1])

    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    tbe.build(sch, config)
