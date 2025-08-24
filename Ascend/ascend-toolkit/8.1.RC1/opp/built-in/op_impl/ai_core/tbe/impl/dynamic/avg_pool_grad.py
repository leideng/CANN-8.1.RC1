#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
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
avg_pool_grad
"""
from __future__ import absolute_import

import warnings

from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_register
from impl.util.platform_adapter import tvm
from impl.util.util_conv2d import transform_shape_with_format
from impl.util.util_cube_dynamic import AvgPoolGradParaProcess
from impl.util.util_cube_dynamic import Conv2dTransposeParaProcess
from impl.util.util_cube_dynamic import set_default_para
from impl.util.util_cube_dynamic import modify_w_range_max
from impl.util.util_cube_dynamic import gen_conv_shape_range
from impl.util.util_cube_dynamic import modify_dy_w_range_max_opti
from impl.util.util_cube_dynamic import check_modify_w_range
from impl.dynamic.conv2d_backprop_input import check_empty_tensor

BLOCK_SIZE = tbe_platform.BLOCK_REDUCE

SHAPE_SIZE = 4

# shape's dim of input must be 5
INPUT_DIM = 5

# shape's dim of filter must be 6
FILTER_DIM = 6

# shape's dim of output must be 5
OUTPUT_DIM = 5

UB_FUSED_OP_NUM = 4

NONETYPE = type(None)
H_DIM = 2
W_DIM = 3

UNKNOWN_RANK_SHAPE = [-2]

# fuzzy compile constant
FUZZ_MODE_N_MAX = 2147483647
FUZZ_MODE_HW_MAX = 4096
OP_TYPE = "avg_pool_grad"
LOEWR_LIMIT_FLAG = "lower_limit"
UPPER_LIMIT_FLAG = "upper_limit"
UNSUPPORTED_FUZZ_RES = [{"result": "UNSUPPORTED"}]
LOWER_LIMIT_FUZZ_RES = [{"result": "UNSUPPORTED", "reason": {"param_index": [1], "type": ["lower_limit"]}}]
UPPER_LIMIT_FUZZ_RES = [{"result": "UNSUPPORTED", "reason": {"param_index": [1], "type": ["upper_limit"]}}]


def _check_range(range_str, range_in):
    if len(range_in) != 2:
        error_manager_cube.raise_err_specific_user("dynamic_avg_pool_grad",
                                                   "the range length of " + range_str + " must be equal to 2")
    if range_in[0] < 1:
        error_manager_cube.raise_err_specific_user("dynamic_avg_pool_grad",
                                                   "the range of " + range_str + " should >= 1")
    if range_in[1] is not None and range_in[1] < range_in[0]:
        error_manager_cube.raise_err_specific_user("dynamic_avg_pool_grad",
                                                   "in the range of " + range_str +
                                                   ", the upper limit is not less than the lower limit")


def _check_shape(check_shape, expect_shape, shape_str):
    if len(check_shape) != expect_shape:
        error_manager_cube.raise_err_specific_user("dynamic_avg_pool_grad",
                                                   "dim of " + shape_str + " should be " + str(expect_shape) +
                                                   " but actual is " + str(check_shape))


def _check_param_value_in_range(param_value, expetect_values, param_name):
    if param_value not in expetect_values:
        error_manager_cube.raise_err_input_params_not_expected("dynamic_avg_pool_grad", param_name,
                                                               "/".join(expetect_values), param_value)


def _get_unknown_rank_shape_range(input_grad, out_grad, data_format, channel_out):
    if list(input_grad.get('shape')) == UNKNOWN_RANK_SHAPE or list(input_grad.get('ori_shape')) == UNKNOWN_RANK_SHAPE:
        if data_format == "NCHW":
            input_grad['ori_shape'] = [-1, channel_out, -1, -1]
            input_grad['range'] = [(1, None), (channel_out, channel_out), (1, None), (1, None)]
        else:
            input_grad['ori_shape'] = [-1, -1, -1, channel_out]
            input_grad['range'] = [(1, None), (1, None), (1, None), (channel_out, channel_out)]
        input_grad['shape'] = [-1, (channel_out + BLOCK_SIZE - 1) // BLOCK_SIZE, -1, -1, BLOCK_SIZE]
        if out_grad['ori_format'] == "NCHW":
            out_grad['ori_shape'] = [-1, channel_out, -1, -1]
            out_grad['range'] = [(1, None), (channel_out, channel_out), (1, None), (1, None)]
        else:
            out_grad['ori_shape'] = [-1, -1, -1, channel_out]
            out_grad['range'] = [(1, None), (1, None), (1, None), (channel_out, channel_out)]
        out_grad['shape'] = [-1, (channel_out + BLOCK_SIZE - 1) // BLOCK_SIZE, -1, -1, BLOCK_SIZE]
    return input_grad, out_grad


def _get_unknown_channel_shape_range(input_grad, out_grad, data_format, channel_out):
    if list(input_grad.get('shape'))[1] == -1 or list(input_grad.get('ori_shape'))[data_format.index('C')] == -1:
        input_grad_ori_shape = input_grad.get('ori_shape')
        input_grad_range = input_grad.get('range')
        input_grad_shape = input_grad.get('shape')
        if data_format == "NCHW":
            input_grad['ori_shape'] = [input_grad_ori_shape[0], channel_out,
                                           input_grad_ori_shape[2], input_grad_ori_shape[3]]
            input_grad['range'] = [input_grad_range[0], (channel_out, channel_out),
                                       input_grad_range[2], input_grad_range[3]]
        else:
            input_grad['ori_shape'] = [input_grad_ori_shape[0], input_grad_ori_shape[1],
                                           input_grad_ori_shape[2], channel_out]
            input_grad['range'] = [input_grad_range[0], input_grad_range[1],
                                       input_grad_range[2], (channel_out, channel_out)]
        input_grad['shape'] = [input_grad_shape[0], (channel_out + BLOCK_SIZE - 1) // BLOCK_SIZE,
                               input_grad_shape[2], input_grad_shape[3], BLOCK_SIZE]
        out_grad_ori_shape = out_grad.get('ori_shape')
        out_grad_range = out_grad.get('range')
        out_grad_shape = out_grad.get('shape')

        x_n_idx = out_grad.get('ori_format').index('N')
        x_h_idx = out_grad.get('ori_format').index('H')
        x_w_idx = out_grad.get('ori_format').index('W')
        if out_grad['ori_format'] == "NCHW":
            out_grad['ori_shape'] = [out_grad_ori_shape[x_n_idx], channel_out,
                                        out_grad_ori_shape[x_h_idx], out_grad_ori_shape[x_w_idx]]
            out_grad['range'] = [out_grad_range[x_n_idx], (channel_out, channel_out),
                                    out_grad_range[x_h_idx], out_grad_range[x_w_idx]]
        else:
            out_grad['ori_shape'] = [out_grad_ori_shape[x_n_idx], out_grad_ori_shape[x_h_idx],
                                        out_grad_ori_shape[x_w_idx], channel_out]
            out_grad['range'] = [out_grad_range[x_n_idx], out_grad_range[x_h_idx],
                                 out_grad_range[x_w_idx], (channel_out, channel_out)]
        out_grad['shape'] = [out_grad_shape[0], (channel_out + BLOCK_SIZE - 1) // BLOCK_SIZE,
                             out_grad_shape[2], out_grad_shape[3], BLOCK_SIZE]
    return input_grad, out_grad


def _check_strides(stride_n, stride_c, stride_h, stride_w):
    if stride_h < 1 or stride_w < 1:
        error_manager_cube.raise_err_specific_user("dynamic_avg_pool_grad",
                                                   "the H and W dimensions of strides should >= 1")
    if stride_n != 1 or stride_c != 1:
        error_manager_cube.raise_err_specific_user("dynamic_avg_pool_grad",
                                                   "the N and C dimensions of strides should == 1")


def _get_shape_hw(matrix, shape, range_name, idx):
    if shape[idx] == -1:
        _check_range(range_name, matrix.get('range')[idx])
        range_upper = matrix.get('range')[idx][1]
        shape[idx] = range_upper if range_upper is not None else 1
    return shape


def _check_kernel_shape(k_c, k_h, k_w, ksize_h, ksize_w):
    if k_c != 1:
        error_manager_cube.raise_err_specific_user("dynamic_avg_pool_grad",
                                                   "the c_shape of kernel should be 1")
    if k_h > 255 or k_w > 255:
        error_manager_cube.raise_err_specific_user("dynamic_avg_pool_grad",
                                                   "chip ISA limit kernel_h or kernel_w must less than 255")
    if k_h != ksize_h or k_w != ksize_w:
        error_manager_cube.raise_err_specific_user("dynamic_avg_pool_grad",
                                                   "the h_shape and w_shape of kernel should be equal with ksize")


def _check_ub_limitation(stride_h, stride_w, y_w):
    w_value = y_w * stride_w
    aub_size_min = y_w * BLOCK_SIZE * 2
    aub_filling_size_min = w_value * BLOCK_SIZE * 2
    cub_size_min = BLOCK_SIZE * BLOCK_SIZE * 2
    ub_size = tbe_platform.get_soc_spec("UB_SIZE")
    ub_under_lim_case1 = stride_h == 1 and stride_w == 1 and (aub_size_min * UB_FUSED_OP_NUM + cub_size_min) > ub_size
    ub_under_lim_case2 = ((stride_h > 1 or stride_w > 1) and
                                 (aub_size_min * UB_FUSED_OP_NUM + aub_filling_size_min + cub_size_min) > ub_size)
    if ub_under_lim_case1 or ub_under_lim_case2:
        error_manager_cube.raise_err_specific_user("dynamic_avg_pool_grad",
                                                   "UB's memory space must be enough to support minimum block")


def _avgpoolgrad_check_rule(input_grad, kernel_matrix, out_grad, ksize, strides,
                            padding, data_format, kernel_name):
    _check_param_value_in_range(data_format, ("NCHW", "NHWC"), "data_format")
    shape_k = list(kernel_matrix.get('ori_shape'))
    _check_param_value_in_range(kernel_matrix.get('ori_format'), ("NCHW", "NHWC", "HWCN"), "kernel_matrix's ori_format")
    dim_k_n = kernel_matrix.get('ori_format').index("N")
    dim_k_c = kernel_matrix.get('ori_format').index("C")
    dim_k_h = kernel_matrix.get('ori_format').index("H")
    dim_k_w = kernel_matrix.get('ori_format').index("W")
    x_n_idx = out_grad.get('ori_format').index('N')
    x_c_idx = out_grad.get('ori_format').index('C')
    x_h_idx = out_grad.get('ori_format').index('H')
    x_w_idx = out_grad.get('ori_format').index('W')
    y_n_idx = input_grad.get('ori_format').index('N')
    y_c_idx = input_grad.get('ori_format').index('C')
    y_h_idx = input_grad.get('ori_format').index('H')
    y_w_idx = input_grad.get('ori_format').index('W')
    channel_out = shape_k[dim_k_n]
    if channel_out == -1:
        error_manager_cube.raise_err_specific_user("dynamic_avg_pool_grad",
                                                   "N-dim of kernel should not be dynamic")
    input_grad, out_grad = _get_unknown_rank_shape_range(input_grad, out_grad, data_format, channel_out)
    input_grad, out_grad = _get_unknown_channel_shape_range(input_grad, out_grad, data_format, channel_out)

    input_grad_shape = input_grad.get('shape')
    out_grad_shape = out_grad.get('shape')
    data_dtype = input_grad.get('dtype').lower()
    out_dtype = out_grad.get('dtype').lower()
    input_grad_ori_format = input_grad.get('ori_format')
    _check_shape(input_grad_shape, INPUT_DIM, "input_grad_shape")
    _check_shape(out_grad_shape, OUTPUT_DIM, "out_grad_shape")
    _check_shape(strides, SHAPE_SIZE, "strides")
    _check_shape(ksize, SHAPE_SIZE, "ksize")
    para_check.check_dtype(data_dtype, ('float16',))
    para_check.check_dtype(out_dtype, ('float16',))
    para_check.check_kernel_name(kernel_name)
    if input_grad_ori_format == "NHWC":
        stride_n = strides[0]
        stride_h = strides[1]
        stride_w = strides[2]
        stride_c = strides[3]
        ksize_h = ksize[1]
        ksize_w = ksize[2]
    elif input_grad_ori_format == "NCHW":
        stride_n = strides[0]
        stride_c = strides[1]
        stride_h = strides[2]
        stride_w = strides[3]
        ksize_h = ksize[2]
        ksize_w = ksize[3]
    else:
        error_manager_cube.raise_err_input_params_not_expected("dynamic_avg_pool_grad",
                                                               "input_grad", "NCHW/NHWC", input_grad_ori_format)

    if input_grad_ori_format != data_format:
        error_manager_cube.raise_err_specific_user("dynamic_avg_pool_grad",
                                                   "the ori_format of input_grad must be equal with data_format")

    _check_param_value_in_range(padding, ("SAME", "VALID"), "padding")
    _check_strides(stride_n, stride_c, stride_h, stride_w)

    shape_dx = list(out_grad.get('ori_shape'))
    shape_dy = list(input_grad.get('ori_shape'))
    _check_shape(shape_dx, SHAPE_SIZE, "out_grad's ori_shape")
    _check_shape(shape_dy, SHAPE_SIZE, "input_grad's ori_shape")
    _check_shape(shape_k, SHAPE_SIZE, "kernel_matrix's ori_shape")
    _check_param_value_in_range(out_grad.get('ori_format'), ("NCHW", "NHWC"), "out_grad's ori_format")

    # dynamic_mode h, w is range_max h, w
    k_h = shape_k[dim_k_h]
    k_w = shape_k[dim_k_w]
    y_w = shape_dy[y_w_idx]
    if shape_dy[y_n_idx] != -1 and shape_dy[y_h_idx] != -1 and shape_dy[y_w_idx] != -1:
        error_manager_cube.raise_err_specific_user("dynamic_avg_pool_grad",
                                                   "no dynamic shape found in input")
    shape_dx = _get_shape_hw(out_grad, shape_dx, "out_grad_h", x_h_idx)
    shape_dx = _get_shape_hw(out_grad, shape_dx, "out_grad_w", x_w_idx)
    shape_dy = _get_shape_hw(input_grad, shape_dy, "input_grad_h", y_h_idx)
    shape_dy = _get_shape_hw(input_grad, shape_dy, "input_grad_w", y_w_idx)
    if shape_dy[y_n_idx] == -1:
        _check_range("input_grad_n", input_grad.get('range')[y_n_idx])
    if shape_dx[x_n_idx] == -1:
        _check_range("out_grad_n", out_grad.get('range')[x_n_idx])
    if shape_dx[x_n_idx] != shape_dy[y_n_idx] or shape_dx[x_c_idx] != shape_dy[y_c_idx]:
        error_manager_cube.raise_err_specific_user("dynamic_avg_pool_grad",
                                                   "input must be equal with out on N-dim and C-dim")
    if shape_dx[x_c_idx] != shape_k[dim_k_n]:
        error_manager_cube.raise_err_specific_user("dynamic_avg_pool_grad",
                                                   "N-dim of kernel must be equal with dx on C-dim")

    _check_kernel_shape(shape_k[dim_k_c], k_h, k_w, ksize_h, ksize_w)
    _check_ub_limitation(stride_h, stride_w, y_w)
    return stride_h, stride_w, shape_dy[y_c_idx]


def _check_dynamic_range(grad_range):
    """
    check dynamic range

    Parameters

    ---------
    grad_range: dict, range information.

    Return

    --------
    grad_res: range of grad_range.
    """

    grad_range_new = []
    if grad_range.get('range') is not None:
        for i in range(len(grad_range.get('range'))):
            conver_middle = list(grad_range.get('range')[i])
            conver_middle[0] = max(conver_middle[0], 1)
            grad_final_range = tuple(conver_middle)
            grad_range_new.append(grad_final_range)

        grad_range['range'] = tuple(grad_range_new)

    return grad_range


def _is_fuzzy_input_valid(tensor_dict):
    """
    Validate input for fuzzy compile.
    """
    is_valid = (
        isinstance(tensor_dict.get("ori_shape"), (list, tuple)) and
        len(tensor_dict.get("ori_shape")) == SHAPE_SIZE and
        list(tensor_dict.get("ori_shape")) != UNKNOWN_RANK_SHAPE and
        tensor_dict.get("ori_format") in ("NHWC", "NCHW")
        )
    if is_valid and -1 in tensor_dict.get("ori_shape"):
        is_valid &= len(tensor_dict.get("ori_range")) == SHAPE_SIZE

    return is_valid


def _check_tensor_range(tensor_dict, upper_limit_dict):
    """
    Helper function for tensor range validation.
    """
    data_format = tensor_dict.get("ori_format")
    n_dim, h_dim, w_dim = data_format.find("N"), data_format.find("H"), data_format.find("W")
    n_dim_max, h_dim_max, w_dim_max = \
        upper_limit_dict.get("N"), upper_limit_dict.get("H"), upper_limit_dict.get("W")

    lower_limit_flag = (
        tensor_dict.get("ori_range")[n_dim][0] > n_dim_max or
        tensor_dict.get("ori_range")[h_dim][0] > h_dim_max or
        tensor_dict.get("ori_range")[w_dim][0] > w_dim_max
    )

    if lower_limit_flag:
        warnings.warn("Lower range of N/H/W dim exceeds upper limit, please check.")
        return LOEWR_LIMIT_FLAG

    upper_limit_flag = (
        None in list(zip(*tensor_dict.get("ori_range")))[1] or
        -1 in list(zip(*tensor_dict.get("ori_range")))[1] or
        tensor_dict.get("ori_range")[n_dim][1] > n_dim_max or
        tensor_dict.get("ori_range")[h_dim][1] > h_dim_max or
        tensor_dict.get("ori_range")[w_dim][1] > w_dim_max
    )

    if upper_limit_flag:
        warnings.warn("Upper range of N/H/W dim exceeds upper limit, please check.")
        return UPPER_LIMIT_FLAG

    return ""


def _check_tensor_shape(tensor_dict, upper_limit_dict):
    """
    Helper function for tensor shape check.
    """
    data_format = tensor_dict.get("ori_format")
    n_dim, h_dim, w_dim = data_format.find("N"), data_format.find("H"), data_format.find("W")
    n_dim_max, h_dim_max, w_dim_max = \
        upper_limit_dict.get("N"), upper_limit_dict.get("H"), upper_limit_dict.get("W")
    shape_invalid = lambda shape_val, upper_limit: shape_val >= upper_limit or shape_val <= 0
    return (
        shape_invalid(tensor_dict.get("ori_shape")[n_dim], n_dim_max) or
        shape_invalid(tensor_dict.get("ori_shape")[h_dim], h_dim_max) or
        shape_invalid(tensor_dict.get("ori_shape")[w_dim], w_dim_max)
    )


def _generalize_tensor_shape(tensor_dict):
    """
    Helper function for tensor shape generalization.
    """
    general_shape = [-1] * len(tensor_dict.get("ori_shape"))
    c_dim = tensor_dict.get("ori_format").find("C")
    general_shape[c_dim] = tensor_dict.get("ori_shape")[c_dim]

    return general_shape


def _correct_generalized_range(input_tensor_list, input_attr_list):
    """
    Check and modify generalized range.
    """
    orig_input_shape, input_grad, kernel_matrix, out_grad = input_tensor_list
    strides, padding, dilations, data_format, kernel_name, dynamic_flag = input_attr_list
    param_list = [strides, padding, (1, 1, 1, 1), data_format]
    is_pass_check, dedy_modify = modify_dy_w_range_max_opti(input_grad, kernel_matrix, param_list,
                                                            OP_TYPE, dynamic_flag)
    if not is_pass_check:
        return dedy_modify
    input_grad = dedy_modify

    # if over l1 size then modify dy h/w range
    upper_range_result = modify_w_range_max(out_grad, kernel_matrix, input_grad, strides, dilations, data_format,
                                            OP_TYPE, dynamic_flag)
    dedy_h_range_max = upper_range_result.get("dedy_h_max")
    dedy_w_range_max = upper_range_result.get("w_max")
    is_single_point = upper_range_result.get("is_single_point")

    if upper_range_result.get("is_exceed_l1"):
        return UNSUPPORTED_FUZZ_RES

    # get dx_range depends on dy_range
    dy_range = input_grad["ori_range"]
    ori_data_format = input_grad["ori_format"]
    pos_c = ori_data_format.find('C')
    groups = input_grad["ori_shape"][pos_c]
    pads = [0, 0, 0, 0] if padding == "VALID" else [-1, -1, -1, -1]
    ori_paras = {
        "input_size": orig_input_shape, "x": input_grad, "filters": kernel_matrix, "bias": None, "offset_w": None,
        "y": out_grad, "strides": strides, "pads": pads, "dilations": (1, 1, 1, 1), "groups": groups,
        "data_format": data_format, "output_padding": (0, 0, 0, 0), "offset_x": 0, "kernel_name": kernel_name}
    conv2d_tranpose = Conv2dTransposeParaProcess(ori_paras)
    conv2d_tranpose.get_attr_nchw(data_format)
    filter_shape_nchw = conv2d_tranpose.get_input_nchw(kernel_matrix.get("ori_shape"), kernel_matrix.get("ori_format"))
    _, dy_range_nchw = conv2d_tranpose.get_input_nchw(input_grad.get("ori_shape"), input_grad.get("ori_format"),
                                                      dy_range)

    dy_range_nchw[H_DIM] = [dy_range_nchw[H_DIM][0], min(dedy_h_range_max, dy_range_nchw[H_DIM][1])]
    if is_single_point:
        dy_range_nchw[W_DIM] = [dedy_w_range_max, dedy_w_range_max]
    else:
        dy_range_nchw[W_DIM] = [dy_range_nchw[W_DIM][0], min(dedy_w_range_max, dy_range_nchw[W_DIM][1])]
    if input_grad["ori_shape"][input_grad.get("ori_format").find("W")] > dy_range_nchw[W_DIM][1]:
        warnings.warn(
            OP_TYPE, "{}, invalid input_grad ori_shape {}, w should not larger than {}".format(
                OP_TYPE, str(input_grad.get("shape")), dy_range_nchw[W_DIM][1]))
        return UNSUPPORTED_FUZZ_RES

    _, _, new_dy_range_nchw = conv2d_tranpose.get_input_range(filter_shape_nchw, dy_range_nchw)

    input_grad["ori_range"] = list(input_grad["ori_range"])
    input_grad["ori_range"][input_grad.get("ori_format").find("H")] = new_dy_range_nchw[H_DIM]
    input_grad["ori_range"][input_grad.get("ori_format").find("W")] = new_dy_range_nchw[W_DIM]

    input_grad["ori_shape"] = _generalize_tensor_shape(input_grad)
    out_grad["ori_shape"] = _generalize_tensor_shape(out_grad)

    return ""


def _collect_ori_tensors(ori_paras):
    """
    get valid tensors
    """
    ori_tensors = {}
    for key, value in ori_paras.items():
        valid_tensor = isinstance(value, dict) and \
                       isinstance(value.get("ori_shape"), (list, tuple)) and \
                       len(value.get("ori_shape")) > 0
        if valid_tensor:
            ori_tensors[key] = value
    return ori_tensors


def _avgpoolgrad_compute(input_size, filters, out_backprop, y, strides, pads,
                         dilations=(1, 1, 1, 1), groups=1, data_format='NHWC',
                         kernel_name='cce_avg_pool_grad_dilation'):
    ori_paras = {
        "input_size": input_size, "filters": filters, "out_backprop": out_backprop, "y": y,
        "strides": strides, "pads": pads, "dilations": dilations, "groups": groups, "data_format": data_format,
        "kernel_name": kernel_name, "pooling_mode": "AVG"
    }

    default_para = set_default_para()

    if not input_size.get("ori_shape"):
        ori_paras["input_size"]["ori_shape"] = default_para["input_size"]["ori_shape"]

    conv2dbp_para = AvgPoolGradParaProcess(ori_paras)

    conv2dbp_para.config_paras()

    dedx = tbe.conv2d_backprop_input(filters=conv2dbp_para.tensors.get("filter_tensor"),
                                     out_backprop=conv2dbp_para.tensors.get("dy_tensor"),
                                     filter_sizes=conv2dbp_para.shape.get("filter_shape_nchw"),
                                     input_sizes=conv2dbp_para.shape.get("dx_shape_nchw"),
                                     para_dict={
                                         "strides":
                                         (conv2dbp_para.strides[H_DIM], conv2dbp_para.strides[W_DIM]),
                                         "padding": conv2dbp_para.pads,
                                         "dilations": conv2dbp_para.dilations,
                                         "res_dtype": default_para.get("res_dtype"),
                                         "kernel_name": kernel_name,
                                         "group_dict": conv2dbp_para.attrs.get("group_para"),
                                         "correct_range_flag": conv2dbp_para.attrs.get("correct_range_flag", False),
                                         "pooling_mode": conv2dbp_para.pooling_mode,
                                         "ori_tensors": _collect_ori_tensors(ori_paras),
                                         "op_type": "AvgPoolGrad"
                                     })

    return {'op_placeholder': [conv2dbp_para.tensors.get("input_tensor"),
                               conv2dbp_para.tensors.get("dy_tensor"),
                               conv2dbp_para.tensors.get("filter_tensor")],
            'op_res': [dedx]}


def correct_fuzzy_build_range(input_grad, strides, data_format):
    """
    get input w range with UB size

    Notice
    ------
    the proper range are not smaller than shape value
    """
    pos_h = data_format.find('H')
    pos_w = data_format.find('W')
    proper_range = list(map(list, input_grad.get("ori_range")))
    pos_range_h, pos_range_w = 2, 3  # NC1HWC0
    proper_w = proper_range[pos_range_w][1]
    cube_size_min = BLOCK_SIZE * BLOCK_SIZE * 2
    ub_size = tbe_platform.get_soc_spec("UB_SIZE")
    invalid = (None in proper_range[pos_range_h]) or \
              (None in proper_range[pos_range_w]) or \
              (input_grad["ori_shape"][pos_w] == -1)
    if invalid:
        return
    strides_h = strides[pos_h]
    strides_w = strides[pos_w]
    if strides_h == 1 and strides_w == 1:
        for y_w in list(range(proper_w, input_grad["ori_shape"][pos_w] - 1, -1)):
            aub_size_min = y_w * BLOCK_SIZE * 2
            limit_size = aub_size_min * UB_FUSED_OP_NUM + cube_size_min
            if limit_size < ub_size:
                break
        proper_w = y_w
    elif strides_h > 1 or strides_w > 1:
        for y_w in list(range(proper_w, input_grad["ori_shape"][pos_w] - 1, -1)):
            w_value = y_w * strides_w
            aub_size_min = y_w * BLOCK_SIZE * 2
            aub_filling_size_min = w_value * BLOCK_SIZE * 2
            limit_size = aub_size_min * UB_FUSED_OP_NUM + aub_filling_size_min + cube_size_min
            if limit_size < ub_size:
                break
        proper_w = y_w
    if proper_w != proper_range[pos_range_w][1]:
        proper_range[pos_range_w][1] = proper_w
        input_grad["range"] = proper_range
        valid = isinstance(input_grad["ori_range"], (list, tuple)) and len(input_grad["ori_range"]) == 4
        if valid:
            ori_range = list(map(list, input_grad["ori_range"]))
            ori_range[pos_w][1] = proper_w
            input_grad["ori_range"] = ori_range


@tbe_register.register_param_generalization("AvgPoolGrad")
def avg_pool_grad_generalization(orig_input_shape,
                                 input_grad,
                                 kernel_matrix,
                                 out_grad,
                                 ksize,
                                 strides,
                                 padding,
                                 data_format='NHWC',
                                 kernel_name="cce_avg_pool_grad_dilation",
                                 generalize_config=None):
    """
    computes average pooling backwards gradients.

    Parameters:
    ----------

    orig_input_shape: a dict, forward input shape

    input_grad: a dict, global model support 'NHWC' or 'NCHW'
                and padding valid, common model support 'NHWC'
                and float16

    kernel_matrix: a dict, global model support 'NHWC' or 'NCHW'
                and padding valid, common model support 'NHWC'
                and float16

    out_grad: a dict, global model support 'NHWC' or 'NCHW'
                and padding valid, common model support 'NHWC'
                and float16

    ksize: filter window size, int or 4-D list, support 'NHWC'

    strides: strides over h and w axis, int or 4-D list,
             support 'NHWC' or 'NCHW'

    padding:global model support 'NHWC' or 'NCHW' and padding valid

    data_format: support 'NHWC' or 'NCHW'

    kernel_name : cce kernel name, default value is "cce_avg_pool_grad_dilation"

    generalize_config: dict
        support keep_rank

    Returns
    -------
    params list
    """
    if generalize_config.get("mode") != "keep_rank":
        warnings.warn("Only support keep_rank in generalize_config")
        return

    if not _is_fuzzy_input_valid(input_grad) or not _is_fuzzy_input_valid(out_grad):
        return UNSUPPORTED_FUZZ_RES

    result = []
    dynamic_flag = -1 in input_grad.get("ori_shape")
    upper_limit_dict = {"N": FUZZ_MODE_N_MAX, "H": FUZZ_MODE_HW_MAX, "W": FUZZ_MODE_HW_MAX}
    fuzz_res_dict = {"unsupported": UNSUPPORTED_FUZZ_RES, "lower_limit": LOWER_LIMIT_FUZZ_RES,
                     "upper_limit": UPPER_LIMIT_FUZZ_RES}
    if kernel_matrix is None:
        kernel_shape = list(ksize)
        kernel_shape[data_format.find("N")] = input_grad.get("ori_shape")[data_format.find("C")]
        kernel_matrix = {"ori_shape": kernel_shape, "dtype": "float16", "ori_format": data_format}
    if dynamic_flag:
        dedy_range_status = _check_tensor_range(input_grad, upper_limit_dict)
        if dedy_range_status:
            return [{"result": "UNSUPPORTED", "reason": {"param_index": [1], "type": [dedy_range_status]}}]
        input_list = [out_grad, kernel_matrix, input_grad]
        param_list = [strides, padding, [1, 1, 1, 1], data_format]
        check_res, _, _ = check_modify_w_range(input_list, param_list, OP_TYPE, True)
        if fuzz_res_dict.get(check_res):
            return fuzz_res_dict.get(check_res)
    else:
        dedy_invalid_flag = _check_tensor_shape(input_grad, upper_limit_dict)
        dedx_invalid_flag = _check_tensor_shape(out_grad, upper_limit_dict)
        if dedx_invalid_flag or dedy_invalid_flag:
            return UNSUPPORTED_FUZZ_RES
        # generalize input range
        input_grad = gen_conv_shape_range(input_grad, OP_TYPE, dynamic_flag)
        # correct generalized range
        input_tensor_list = [orig_input_shape, input_grad, kernel_matrix, out_grad]
        input_attr_list = [strides, padding, (1, 1, 1, 1,), data_format, kernel_name, dynamic_flag]
        range_status = _correct_generalized_range(input_tensor_list, input_attr_list)
        if range_status:
            return range_status

    result.append([orig_input_shape, input_grad, kernel_matrix, out_grad, {"strides": strides}, {"padding": padding},
                   {"ksize": ksize}, {"kernel_name": kernel_name}, {"data_format": data_format}])

    return result


def _check_empty_tensor(input_grad, kernel_matrix, out_grad, strides, padding):
    pads = [-1, -1, -1, -1] if padding == "SAME" else [0, 0, 0, 0]
    check_empty_tensor(kernel_matrix, input_grad, out_grad, strides, pads)


@tbe_register.register_operator("AvgPoolGrad")
@para_check.check_input_type(dict, dict, dict, dict, (tuple, list),
                             (tuple, list), (tuple, list, str), str, str)
def avg_pool_grad(orig_input_shape,
                  input_grad,
                  kernel_matrix,
                  out_grad,
                  ksize,
                  strides,
                  padding,
                  data_format='NHWC',
                  kernel_name="cce_avg_pool_grad_dilation"):
    """
    computes average pooling backwards gradients.

    Parameters:
    ----------

    orig_input_shape: a dict, forward input shape

    input_grad: a dict, global model support 'NHWC' or 'NCHW'
                and padding valid, common model support 'NHWC'
                and float16

    kernel_matrix: a dict, global model support 'NHWC' or 'NCHW'
                and padding valid, common model support 'NHWC'
                and float16

    out_grad: a dict, global model support 'NHWC' or 'NCHW'
                and padding valid, common model support 'NHWC'
                and float16

    ksize: filter window size, int or 4-D list, support 'NHWC'

    strides: strides over h and w axis, int or 4-D list,
             support 'NHWC' or 'NCHW'

    padding:global model support 'NHWC' or 'NCHW' and padding valid

    data_format: support 'NHWC' or 'NCHW'

    kernel_name : cce kernel name, default value is "cce_avg_pool_grad_dilation"

    Returns
    -------
    None
    """
    _check_empty_tensor(input_grad, kernel_matrix, out_grad, strides, padding)
    stride_h, stride_w, input_c = _avgpoolgrad_check_rule(input_grad, kernel_matrix, out_grad, ksize, strides,
                                                          padding, data_format, kernel_name)

    if kernel_matrix is not None:
        dilations = (1, 1, 1, 1)
        if data_format == "NCHW":
            strides = [1, 1, stride_h, stride_w]
        elif data_format == "NHWC":
            strides = [1, stride_h, stride_w, 1]
        if padding == "SAME":
            padding = [-1, -1, -1, -1]
        else:
            padding = [0, 0, 0, 0]
        with tbe.compute():
            res = _avgpoolgrad_compute(orig_input_shape, kernel_matrix, input_grad, out_grad, strides, padding,
                                       dilations, groups=input_c, data_format=data_format, kernel_name=kernel_name)
        with tvm.target.cce():
            sch = tbe.auto_schedule(res.get('op_res'))
        tensor_list = res.get('op_placeholder') + res.get('op_res')
        config = {'print_ir': False,
                  'name': kernel_name,
                  'tensor_list': tensor_list,
                  'build_args': {'constant_realize_extent_in_infer_bound': False}}
        tbe.build(sch, config)
    else:
        error_manager_cube.raise_err_input_params_not_expected("dynamic_avg_pool_grad", "filter", "dict", "None")
