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
avg_pool3d
"""
from impl.dynamic import conv3d
from impl.util import util_common
from impl.util import util_conv3d
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import error_manager_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm

_OP_NAME = "AvgPool3D"
_KSIZE_TARGET_FORMAT = "NDHWC"
_STRIDE_TARGET_FORMAT = "NDHWC"
_FMAP_TARGET_FORMAT = "NDHWC"
_PADS_FORMAT = "DHW"

_FORMAT_WHITE_LIST = ["NDHWC", "NCDHW"]
_FILTER_WHITE_LIST = ["DHWCN", "NDHWC"]

_N_DIM_6D = 0
_D_DIM_6D = 1
_H_DIM_6D = 3
_W_DIM_6D = 4

_N_DIM_5D = 0
_D_DIM_5D = 1
_H_DIM_5D = 2
_W_DIM_5D = 3
_C_DIM_5D = 4

_C0_SIZE = 16
_DYNAMIC_FLAG = -1
_FUSED_NUM = 3
_ORI_SHAPE_DIM = 5
_SHAPE_DIM = 6
_PADS_DIM = 6
# lower range
_LOWER_RANGE = 1
# upper range
_UPPER_RANGE = 4096
# stride range
_STRIDE_MIN = 1
_STRIDE_MAX = 63
# ksize range
_KSIZE_MIN = 1
_KSIZE_MAX = 255


def _check_inputs(fmap, kernel, ksize, strides, pads, ceil_mode, data_format):
    def __check_data_fomat(format_name, checked_format, excepted_format_list):
        if checked_format not in excepted_format_list:
            dict_args = {
                'errCode': 'E62002',
                'param_name': format_name,
                'expected_format_list': ",".join(excepted_format_list),
                'format': checked_format
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    def __check_dim(dims_name, dims, range_min, range_max) -> None:
        if (len(dims) > range_max or len(dims) < range_min):
            error_manager_vector.raise_err_input_param_range_invalid(
                _OP_NAME, dims_name, range_min, range_max, len(dims))

    def __check_range(dim_name, dim_range, range_min=1, range_max=None) -> None:
        if dim_range[0] < range_min:
            error_manager_cube.raise_err_specific(
                _OP_NAME + " " + dim_name, "the lower bound of range should be larger than {}".format(range_min))
        if not dim_range[1]:
            return
        if (range_max is not None) and (dim_range[1] > range_max):
            error_manager_cube.raise_err_specific(
                _OP_NAME + " " + dim_name, "the upper bound of range should be less than {}".format(range_max))
        if dim_range[0] > dim_range[1]:
            error_manager_cube.raise_err_specific(
                _OP_NAME + " " + dim_name, "the upper bound of range should be larger than lower bound")

    def __check_attr_range(attr_name, attr_list, attr_min, attr_max) -> None:
        for i in attr_list:
            if ((i < attr_min) or (i > attr_max)):
                error_manager_cube.raise_err_attr_range_invalid(
                    _OP_NAME, "[{},{}]".format(attr_min, attr_max), attr_name, str(attr_list))

    def __check_shape_in_range(attr_name, shape, vrange) -> None:
        length = len(shape)
        for i in range(length):
            if (shape[i] != -1 and (shape[i] != vrange[i][0] or shape[i] != vrange[i][1])):
                error_manager_cube.raise_err_specific(
                    _OP_NAME + " " + attr_name, "shape:{} is not in range:{}".format(shape, vrange))

    def __check_shape_valid(dim_name, dim_val, vmin, vmax) -> None:
        if dim_val != -1 and (((vmin is not None) and dim_val < vmin) or
                ((vmax is not None) and dim_val > vmax)):
            error_manager_cube.raise_err_specific(
                    _OP_NAME + " shape's " + dim_name,
                    "dim val:{} should be in range[{},{}]".format(dim_val, vmin, vmax))

    fmap_shape = fmap.get("ori_shape")
    fmap_range = fmap.get("range")
    fmap_format = fmap.get("ori_format")
    # check input dtype
    para_check.check_dtype_rule(fmap.get('dtype'), ['float16'], "x")
    para_check.check_dtype_rule(kernel.get('dtype'), ['float16'], "filter")
    # not support ceil_mode
    if ceil_mode:
        error_manager_cube.raise_err_specific(_OP_NAME,
                                              "ceil_mode only support false.")
    if len(fmap_shape) == 1 and fmap_shape[0] == -2:
        error_manager_cube.raise_err_specific(_OP_NAME, "input:x not support -2 current.")
    if fmap_shape[fmap.get("ori_format").index('C')] == -1:
        error_manager_cube.raise_err_specific(_OP_NAME, "fmap c dim not support -1 current.")
    # check format
    __check_data_fomat("fmap's format", fmap.get("ori_format"), _FORMAT_WHITE_LIST)
    __check_data_fomat("filter's format", kernel.get("ori_format"), _FILTER_WHITE_LIST)
    __check_data_fomat("data_format", data_format, _FORMAT_WHITE_LIST)
    # check shape and range dims
    __check_dim("fmap's ori_shape", fmap.get("ori_shape"), _ORI_SHAPE_DIM, _ORI_SHAPE_DIM)
    __check_dim("fmap's shape", fmap.get("shape"), _SHAPE_DIM, _SHAPE_DIM)
    __check_dim("grads's range", fmap.get("range"), _ORI_SHAPE_DIM, _SHAPE_DIM)
    __check_dim("strides", strides, _ORI_SHAPE_DIM, _ORI_SHAPE_DIM)
    __check_dim("ksize", ksize, _ORI_SHAPE_DIM, _ORI_SHAPE_DIM)
    __check_dim("pads", pads, _PADS_DIM, _PADS_DIM)
    # check range value
    range_n, range_d, _, range_h, range_w, _ = fmap_range
    __check_range("x N dim range", range_n)
    __check_range("x D dim range", range_d)
    __check_range("x H dim range", range_h, _LOWER_RANGE, _UPPER_RANGE)
    __check_range("x W dim range", range_w, _LOWER_RANGE, _UPPER_RANGE)
    # check attrs shape
    __check_attr_range("strides", strides, _STRIDE_MIN, _STRIDE_MAX)
    __check_attr_range("ksize", ksize, _KSIZE_MIN, _KSIZE_MAX)
    # check inputs shape
    __check_shape_valid("N dim", fmap_shape[fmap_format.index("N")],
                        _LOWER_RANGE, None)
    __check_shape_valid("D dim", fmap_shape[fmap_format.index("D")],
                        _LOWER_RANGE, _UPPER_RANGE)
    __check_shape_valid("H dim", fmap_shape[fmap_format.index("H")],
                        _LOWER_RANGE, _UPPER_RANGE)
    __check_shape_valid("W dim", fmap_shape[fmap_format.index("W")],
                        _LOWER_RANGE, _UPPER_RANGE)
    __check_shape_valid("C dim", fmap_shape[fmap_format.index("C")],
                        _LOWER_RANGE, None)
    __check_shape_in_range("fmap's shape", fmap.get("shape"), fmap_range)
    # check slice window
    for dim_format in _PADS_FORMAT:
        shape_legal_flag = ((fmap_shape[fmap.get("ori_format").index(dim_format)] +
                            pads[2 * _PADS_FORMAT.index(dim_format)] +
                            pads[2 * _PADS_FORMAT.index(dim_format) + 1]) <
                            ksize[_KSIZE_TARGET_FORMAT.index(dim_format)])
        if (fmap_shape[fmap.get("ori_format").index(dim_format)] != -1 and
            pads[2 * _PADS_FORMAT.index(dim_format)] != -1 and
            pads[2 * _PADS_FORMAT.index(dim_format) + 1] != -1 and
            shape_legal_flag):
            error_manager_cube.raise_err_specific(
                _OP_NAME,
                "kernel:{} should smaller than fmap:{} + pads:{} in D,H,W dims.".format(
                    ksize, fmap_shape, pads))


def _update_dynamic_pads(fmap_shape, ksize, strides, pads):
    _, fmap_d, fmap_h, fmap_w, _ = fmap_shape
    _, ksize_d, ksize_h, ksize_w, _ = ksize
    _, stride_d, stride_h, stride_w, _ = strides

    if all(i == _DYNAMIC_FLAG for i in pads):
        pad_h = util_common.align(fmap_h, stride_h) - stride_h + ksize_h - fmap_h
        pad_h = tvm.max(pad_h, 0) if isinstance(fmap_h, tvm.Var) else max(pad_h, 0)
        pad_up = pad_h // 2
        pad_down = pad_h - pad_up
        pad_w = util_common.align(fmap_w, stride_w) - stride_w + ksize_w - fmap_w
        pad_w = tvm.max(pad_w, 0) if isinstance(fmap_w, tvm.Var) else max(pad_w, 0)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_d = util_common.align(fmap_d, stride_d) - stride_d + ksize_d - fmap_d
        pad_d = tvm.max(pad_d, 0) if isinstance(fmap_d, tvm.Var) else max(pad_d, 0)
        pad_head = pad_d // 2
        pad_tail = pad_d - pad_head
        pads = [pad_head, pad_tail, pad_up, pad_down, pad_left, pad_right]
    return list(pads)


def _get_output(fmap, ksize, padf, padb, stride):
    return (fmap + padf + padb - ksize) // stride + 1


def _calc_output_shape(fmap_shape, ksize, strides, pads):
    if not all(i == 0 for i in pads):
        output_d = (fmap_shape[_D_DIM_5D] + strides[_D_DIM_5D] - 1) // strides[_D_DIM_5D]
        output_h = (fmap_shape[_H_DIM_5D] + strides[_H_DIM_5D] - 1) // strides[_H_DIM_5D]
        output_w = (fmap_shape[_W_DIM_5D] + strides[_W_DIM_5D] - 1) // strides[_W_DIM_5D]
    else:
        output_d = (fmap_shape[_D_DIM_5D] - ksize[_D_DIM_5D]) // strides[_D_DIM_5D] + 1
        output_h = (fmap_shape[_H_DIM_5D] - ksize[_H_DIM_5D]) // strides[_H_DIM_5D] + 1
        output_w = (fmap_shape[_W_DIM_5D] - ksize[_W_DIM_5D]) // strides[_W_DIM_5D] + 1
    return [fmap_shape[_N_DIM_5D], output_d, output_h, output_w, fmap_shape[_C_DIM_5D]]


def _range_correction(fmap_range, ksize, strides, pads):
    _, ksize_d, ksize_h, ksize_w, _ = ksize
    _, stride_d, stride_h, stride_w, _ = strides

    fmap_range_n, fmap_range_d, fmap_range_c1, fmap_range_h, fmap_range_w, fmap_range_c0 = fmap_range
    if not all(i == 0 for i in pads):
        out_d_upper, out_h_upper, out_w_upper = None, None, None
        out_d_lower = util_common.ceil(fmap_range_d[0], stride_d)
        out_d_upper = util_common.ceil(fmap_range_d[1], stride_d)
        out_h_lower = util_common.ceil(fmap_range_h[0], stride_h)
        out_h_upper = util_common.ceil(fmap_range_h[1], stride_h)
        out_w_lower = util_common.ceil(fmap_range_w[0], stride_w)
        out_w_upper = util_common.ceil(fmap_range_w[1], stride_w)
    else:
        out_d_lower = _get_output(fmap_range_d[0], ksize_d, pads[0], pads[1], stride_d)
        if out_d_lower < 1:
            fmap_range_d_lower = min(ksize_d, fmap_range_d[1]) if fmap_range_d[1] else ksize_d
            fmap_range_d = (fmap_range_d_lower, fmap_range_d[1])
            out_d_lower = _get_output(fmap_range_d[0], ksize_d, pads[0], pads[1], stride_d)
        out_d_upper = _get_output(fmap_range_d[1], ksize_d, pads[0], pads[1], stride_d)

        out_h_lower = _get_output(fmap_range_h[0], ksize_h, pads[2], pads[3], stride_h)
        if out_h_lower < 1:
            fmap_range_h_lower = min(ksize_h, fmap_range_h[1]) if fmap_range_h[1] else ksize_h
            fmap_range_h = (fmap_range_h_lower, fmap_range_h[1])
            out_h_lower = _get_output(fmap_range_h[0], ksize_h, pads[2], pads[3], stride_h)
        out_h_upper = _get_output(fmap_range_h[1], ksize_h, pads[2], pads[3], stride_h)

        out_w_lower = _get_output(fmap_range_w[0], ksize_w, pads[4], pads[5], stride_w)
        if out_w_lower < 1:
            fmap_range_w_lower = min(ksize_w, fmap_range_w[1]) if fmap_range_w[1] else ksize_w
            fmap_range_w = (fmap_range_w_lower, fmap_range_w[1])
            out_w_lower = _get_output(fmap_range_w[1], ksize_w, pads[4], pads[5], stride_w)
        out_w_upper = _get_output(fmap_range_w[0], ksize_w, pads[4], pads[5], stride_w)
    fmap_range_updated = [fmap_range_n, fmap_range_d, fmap_range_c1, fmap_range_h, fmap_range_w, fmap_range_c0]
    output_range_updated = [(fmap_range_n[0], fmap_range_n[1]), (out_d_lower, out_d_upper), fmap_range_c1,
                            (out_h_lower, out_h_upper), (out_w_lower, out_w_upper), (_C0_SIZE, _C0_SIZE)]
    return fmap_range_updated, output_range_updated


def _init_dynamic_shape_var(fmap_shape, fmap_range, output_range):
    if (fmap_shape[_N_DIM_5D] == _DYNAMIC_FLAG or
            all(i != _DYNAMIC_FLAG for i in fmap_shape)):
        fmap_shape[_N_DIM_5D] = operation.var("batch_n", fmap_range[_N_DIM_6D])
        operation.add_exclude_bound_var(fmap_shape[_N_DIM_5D])
    if fmap_shape[_D_DIM_5D] == _DYNAMIC_FLAG:
        fmap_shape[_D_DIM_5D] = operation.var("fmap_d", fmap_range[_D_DIM_6D])
        d_out = operation.var("d_out", output_range[_D_DIM_6D])
        operation.add_exclude_bound_var(d_out)
        operation.add_exclude_bound_var(fmap_shape[_D_DIM_5D])
    if fmap_shape[_H_DIM_5D] == _DYNAMIC_FLAG:
        fmap_shape[_H_DIM_5D] = operation.var("fmap_h", fmap_range[_H_DIM_6D])
        h_out = operation.var("h_out", output_range[_H_DIM_6D])
        operation.add_exclude_bound_var(h_out)
        operation.add_exclude_bound_var(fmap_shape[_H_DIM_5D])
    if fmap_shape[_W_DIM_5D] == _DYNAMIC_FLAG:
        fmap_shape[_W_DIM_5D] = operation.var("fmap_w", fmap_range[_W_DIM_6D])
        w_out = operation.var("w_out", output_range[_W_DIM_6D])
        operation.add_exclude_bound_var(w_out)
        operation.add_exclude_bound_var(fmap_shape[_W_DIM_5D])
    return fmap_shape


def _trans_range_to_6d(shape_range, ori_format):
    range_5d = util_conv3d.transform_shape_with_format(ori_format,
                                                       _FMAP_TARGET_FORMAT,
                                                       shape_range,
                                                       _FORMAT_WHITE_LIST)
    if range_5d is None or len(range_5d) != _ORI_SHAPE_DIM:
        error_manager_cube.raise_err_specific(_OP_NAME,
            "transform range from {} to {} failed, please check range dims and format.".format(
                ori_format, _FMAP_TARGET_FORMAT))
    range_n, range_d, range_h, range_w, range_c = range_5d
    range_c0 = [_C0_SIZE, _C0_SIZE]
    range_c1 = [(range_c[0] + _C0_SIZE - 1) / _C0_SIZE, (range_c[1] + _C0_SIZE - 1) / _C0_SIZE]

    return [range_n, range_d, range_c1, range_h, range_w, range_c0]


def _check_empty_tensor(x, filters, y, strides, pads):
    def _check_dynamic_range_upper(tensor):
        if tensor.get("range"):
            tensor["range"] = tuple((lower, _UPPER_RANGE) if upper is None else (lower, upper)
                                    for _, (lower, upper) in enumerate(tensor.get("range")))
        if tensor.get("ori_range"):
            tensor["ori_range"] = tuple((lower, _UPPER_RANGE) if upper is None else (lower, upper)
                                    for _, (lower, upper) in enumerate(tensor.get("ori_range")))
    _check_dynamic_range_upper(x)
    conv3d.check_empty_tensor(x, filters, y, strides, pads)


@register_operator("AvgPool3D")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def avg_pool3d(x,
               filter,
               y,
               ksize,
               strides,
               pads,
               ceil_mode=False,
               count_include_pad=True,
               divisor_override=0,
               data_format="NDHWC",
               kernel_name="avg_pool3d"):
    """
    computes average pooling3d.

    Parameters:
    ----------
    x : dict, shape and dtype of input_data, only support float16, shape is 5 dims, format is NDC1HWC0

    filter : dict, fractal_z_3d layout, float16 dtype

    y : dict, shape and dtype of output_data, only support float16

    ksize : list or tuple, the window of avg_pool3d, only support avg_pool3d in D or H or W

    strides : list or tuple, the stride of avg_pool3d window, only support avg_pool3d in D or H or W

    pads : list or tuple, count of padding zero or d,h,w axis

    ceil_mode: when True, will use ceil instead of floor in the formula to compute the output shape

    count_include_pad: when True, will include the zero-padding in the averaging calculation.

    divisor_override: if specified, it will be used as divisor, otherwise size of the pooling region will be used.

    data_format : str, default = "NDHWC"

    kernel_name : cce kernel name, default value is "avg_pool3d_d"

    Returns
    -------
    None

    Notice
    -------
    Only support global model currently.
    """
    _check_empty_tensor(x, filter, y, strides, pads)
    fmap_ori_format = x.get("ori_format")
    fmap_ori_shape = x.get("ori_shape")
    fmap_dtype = x.get("dtype")
    fmap_range = x.get("range")

    if len(fmap_range) == _ORI_SHAPE_DIM:
        fmap_range = _trans_range_to_6d(fmap_range, fmap_ori_format)

    ksize = util_conv3d.transform_shape_with_format(data_format, _KSIZE_TARGET_FORMAT,
                                                             ksize, _FORMAT_WHITE_LIST)
    strides = util_conv3d.transform_shape_with_format(data_format, _STRIDE_TARGET_FORMAT,
                                                               strides, _FORMAT_WHITE_LIST)
    fmap_ori_shape_formated = util_conv3d.transform_shape_with_format(fmap_ori_format,
                                                                      _FMAP_TARGET_FORMAT,
                                                                      fmap_ori_shape,
                                                                      _FORMAT_WHITE_LIST)
    _check_inputs(x, filter, ksize, strides, pads, ceil_mode, data_format)

    fmap_range_updated, output_range_updated = _range_correction(fmap_range, ksize, strides, pads)
    dync_fmap_ori_shape = _init_dynamic_shape_var(fmap_ori_shape_formated, fmap_range_updated, output_range_updated)
    dync_output_ori_shape = _calc_output_shape(fmap_ori_shape_formated, ksize, strides, pads)
    dync_pads = _update_dynamic_pads(dync_fmap_ori_shape, ksize, strides, pads)

    dync_fmap_shape = (dync_fmap_ori_shape[0], dync_fmap_ori_shape[1],
                       util_common.ceil(dync_fmap_ori_shape[4], _C0_SIZE),
                       dync_fmap_ori_shape[2], dync_fmap_ori_shape[3], _C0_SIZE)
    dync_output_shape = (dync_output_ori_shape[0], dync_output_ori_shape[1],
                         util_common.ceil(dync_output_ori_shape[4], _C0_SIZE),
                         dync_output_ori_shape[2], dync_output_ori_shape[3], _C0_SIZE)
    group_dict = util_common.calculate_group(fmap_ori_shape_formated[4],
                                             fmap_ori_shape_formated[4],
                                             fmap_ori_shape_formated[4],
                                             _C0_SIZE,
                                             _C0_SIZE)
    _, kd, kh, kw, _ = ksize
    _, stride_d, stride_h, stride_w, _ = strides

    filter_fracz = (dync_fmap_shape[2] * kd * kh * kw, 1, _C0_SIZE, _C0_SIZE)
    align_filter_shape = (util_common.align(fmap_ori_shape_formated[4], _C0_SIZE),
                          16,
                          kd, kh, kw)

    _, fmap_d, fmap_h, fmap_w, _ = dync_fmap_ori_shape
    mul_n, mul_d, mul_c1, mul_h, mul_w, mul_c0 = dync_output_shape
    mul_shape = (mul_n * mul_d, mul_c1, mul_h * mul_w, mul_c0)
    fmap_plh = tvm.placeholder(dync_fmap_shape, name="fmap", dtype=fmap_dtype)
    filter_plh = tvm.placeholder(filter_fracz, name="filter", dtype="float16")
    if not count_include_pad and divisor_override == 0:
        with tbe.compute():
            para_dict = {
                "dsl_flag": True,
                "bias_tensor": None,
                "pads": dync_pads,
                "strides": [strides[_D_DIM_5D], strides[_H_DIM_5D], strides[_W_DIM_5D]],
                "dilation_dhw": [1, 1, 1],
                "res_dtype": "float16",
                "mad_dtype": "float32",
                "kernel_name": kernel_name,
                "group_dict": group_dict,
                "correct_range_flag": True,
                "fused_num": _FUSED_NUM,
            }
            conv_res = tbe.conv3d(fmap_plh, filter_plh, align_filter_shape, para_dict)
            mean_lambda = lambda n, c1, hw, c0: ((tvm.min((n % mul_d) * stride_d + kd, dync_pads[0] + fmap_d) -
                                                  tvm.max(dync_pads[0], (n % mul_d) * stride_d)) *
                                                 (tvm.min((hw // mul_w) * stride_h + kh, dync_pads[2] + fmap_h) -
                                                  tvm.max(dync_pads[2], (hw // mul_w) * stride_h)) *
                                                 (tvm.min((hw % mul_w) * stride_w + kw, dync_pads[4] + fmap_w) -
                                                  tvm.max(dync_pads[4], (hw % mul_w) * stride_w))).astype("int")
            mean_matrix_init = tvm.compute(
                mul_shape, mean_lambda, name="mean_matrix_init", tag="mean_matrix_init"
            )
            mean_matrix_fp16 = tvm.compute(mul_shape, lambda *index:
                                    mean_matrix_init(*index).astype("float16"),
                                    name="mean_matrix_fp16")
            mul_lambda = lambda n, c1, hw, c0: conv_res[n][c1][hw][c0] / mean_matrix_fp16[n][c1][hw][c0]
            mul_res = tvm.compute(mul_shape, mul_lambda,
                        name="mean_matrix_mul", tag="mean_matrix_mul")
        with tvm.target.cce():
            sch = tbe.auto_schedule(mul_res)
        tensor_list = [fmap_plh, filter_plh, mul_res]
    else:
        para_dict = {
            "dsl_flag": False,
            "bias_tensor": None,
            "pads": dync_pads,
            "strides": [strides[_D_DIM_5D], strides[_H_DIM_5D], strides[_W_DIM_5D]],
            "dilation_dhw": [1, 1, 1],
            "res_dtype": "float16",
            "mad_dtype": "float32",
            "kernel_name": kernel_name,
            "group_dict": group_dict,
            "correct_range_flag": True,
            "fused_num": 0,
        }
        with tbe.compute():
            conv_res = tbe.conv3d(fmap_plh, filter_plh, align_filter_shape, para_dict)
        with tvm.target.cce():
            sch = tbe.auto_schedule(conv_res)
        tensor_list = [fmap_plh, filter_plh, conv_res]
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list,
        "dummy_placeholder": True,
        "build_args": {"constant_realize_extent_in_infer_bound": False}
    }

    tbe.build(sch, config)
