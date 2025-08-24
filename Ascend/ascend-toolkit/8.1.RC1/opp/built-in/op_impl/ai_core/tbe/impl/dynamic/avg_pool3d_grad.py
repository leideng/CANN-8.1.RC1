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
avg_pool3d_grad
"""
from impl.util import util_common
from impl.dynamic.conv3d_backprop_input import check_conv3dbp_input_params
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import error_manager_util
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_register
from impl.util.platform_adapter import tvm
from impl.dynamic.conv3d_backprop_input import check_empty_tensor

_C0_SIZE = tbe_platform.C0_SIZE
_UB_FUSED_OP_NUM = 3
_STRIDES_DIM_SIZE = 5
_KSIZE_DIM_SIZE = 5
_PADS_DIM_SIZE = 6
_ORI_SHAPE_DIM_SIZE = 5
_SHAPE_DIM_SIZE = 6
_FMAP_TARGET_FORMAT = "NDHWC"
_GRADS_TARGET_FORMAT = "NDHWC"
_GRADS_FORMAT_WHITE_LIST = ["NCDHW", "NDHWC"]
_DATA_FORMAT_WHITE_LIST = ["NCDHW", "NDHWC"]
_FILTER_TARGET_FORMAT = "DHWCN"
_FILTER_FORMAT_WHITE_LIST = ["DHWCN"]
_KSIZE_FORMAT = "NDHWC"
_STRIDES_FORMAT = "NDHWC"
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
# dynamic mode
LOWER_STR = [{"result": "UNSUPPORTED", "reason": {"param_index": [0], "type": ["lower_limit"]}}]
DYNAMIC_MODE = -1
UNLIMIT_RANGE = [1, -1]


def _check_inputs(grads, weight, output, output_range, ksize, strides, pads, ceil_mode,
                  count_include_pad, divisor_override, data_format):
    def __check_data_fomat(format_name, checked_format, excepted_format_list) -> None:
        if checked_format not in excepted_format_list:
            dict_args = {
                'errCode': 'E62002',
                'param_name': format_name,
                'expected_format_list': ",".join(excepted_format_list),
                'format': checked_format
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    def __check_params_dims(ksize, strides, pads) -> None:
        if len(ksize) != _KSIZE_DIM_SIZE:
            error_manager_vector.raise_err_input_param_range_invalid('avg_pool3d_grad',
                                                                    'ksize',
                                                                    _KSIZE_DIM_SIZE,
                                                                    _KSIZE_DIM_SIZE,
                                                                    len(ksize))
        if len(strides) != _STRIDES_DIM_SIZE:
            error_manager_vector.raise_err_input_param_range_invalid('avg_pool3d_grad',
                                                                    'strides',
                                                                    _STRIDES_DIM_SIZE,
                                                                    _STRIDES_DIM_SIZE,
                                                                    len(strides))
        if len(pads) != _PADS_DIM_SIZE:
            error_manager_vector.raise_err_input_param_range_invalid('avg_pool3d_grad',
                                                                    'pads',
                                                                    _PADS_DIM_SIZE,
                                                                    _PADS_DIM_SIZE,
                                                                    len(pads))

    def __check_dim_range(dims_name, dims, range_min, range_max) -> None:
        if (len(dims) > range_max or len(dims) < range_min):
            error_manager_vector.raise_err_input_param_range_invalid('avg_pool3d_grad',
                                                                    dims_name,
                                                                    range_min,
                                                                    range_max,
                                                                    len(dims))

    def __check_range(dim_name, dim_range, range_min=1, range_max=None) -> None:
        if dim_range[0] < range_min:
            error_manager_cube.raise_err_specific(
                "avg_pool3d_grad's " + dim_name,
                "the lower bound of range should be larger than {}".format(range_min))
        if not dim_range[1]:
            return
        if (range_max is not None) and (dim_range[1] > range_max):
            error_manager_cube.raise_err_specific(
                "avg_pool3d_grad's " + dim_name,
                "the upper bound of range should be less than {}".format(range_max))
        if dim_range[0] > dim_range[1]:
            error_manager_cube.raise_err_specific(
                "avg_pool3d_grad's " + dim_name,
                "the upper bound of range should be larger than lower bound")

    def __check_attr_range(attr_name, attr_value, attr_min, attr_max) -> None:
        if ((attr_min and attr_value < attr_min) or
                (attr_max and  attr_value > attr_max)):
            error_manager_cube.raise_err_attr_range_invalid(
                'avg_pool3d_grad', "-1 or [{},{}]".format(attr_min, attr_max),
                attr_name, str(attr_value))

    def __check_support(ceil_mode, count_include_pad, divisor_override) -> None:
        if ceil_mode:
            error_manager_cube.raise_err_specific('avg_pool3d_grad',
                                                  "ceil_mode only support false.")
        if count_include_pad:
            error_manager_cube.raise_err_specific('avg_pool3d_grad',
                                                  "count_include_pad only support false.")
        if divisor_override != 0:
            error_manager_cube.raise_err_specific('avg_pool3d_grad',
                                                  "divisor_override only support 0.")

    grads_shape = grads.get("ori_shape")
    grads_format = grads.get("ori_format")
    output_shape = output.get("ori_shape")
    output_format = data_format
    # dynamic only support ceil_mode=False count_include=False divisor_override=0
    __check_support(ceil_mode, count_include_pad, divisor_override)
    if len(grads_shape) == 1 and grads_shape[0] == -2:
        error_manager_cube.raise_err_specific('avg_pool3d_grad',
                                              "grads not support -2 current.")
    if len(output_shape) == 1 and output_shape[0] == -2:
        error_manager_cube.raise_err_specific('avg_pool3d_grad',
                                              "ori_input_shape not support -2 current.")
    # check inputs' format
    __check_data_fomat("grads's format", grads.get("ori_format"), _GRADS_FORMAT_WHITE_LIST)
    __check_data_fomat("filter's format", weight.get("ori_format"), _FILTER_FORMAT_WHITE_LIST)
    __check_data_fomat("attribute data_format", data_format, _DATA_FORMAT_WHITE_LIST)
    # check inputs' shape and range dims
    __check_params_dims(ksize, strides, pads)
    __check_dim_range("grads's ori_shape", grads.get("ori_shape"), _ORI_SHAPE_DIM_SIZE, _ORI_SHAPE_DIM_SIZE)
    __check_dim_range("grads's shape", grads.get("shape"), _SHAPE_DIM_SIZE, _SHAPE_DIM_SIZE)
    __check_dim_range("output's ori_shape", output.get("ori_shape"), _ORI_SHAPE_DIM_SIZE, _ORI_SHAPE_DIM_SIZE)
    __check_dim_range("output's shape", output.get("shape"), _SHAPE_DIM_SIZE, _SHAPE_DIM_SIZE)
    __check_dim_range("grads's range", grads.get("range"), _ORI_SHAPE_DIM_SIZE, _SHAPE_DIM_SIZE)
    __check_dim_range("output's range", output_range, _ORI_SHAPE_DIM_SIZE, _SHAPE_DIM_SIZE)
    # check range lower and upper
    range_n, range_d, _, range_h, range_w, _ = output_range
    __check_range("ori_shape_input N dim range", range_n)
    __check_range("ori_shape_input D dim range", range_d)
    __check_range("ori_shape_input H dim range", range_h, _LOWER_RANGE, _UPPER_RANGE)
    __check_range("ori_shape_input W dim range", range_w, _LOWER_RANGE, _UPPER_RANGE)
    # check inputs' shape val
    for st in strides:
        __check_attr_range("stride's val", st, _STRIDE_MIN, _STRIDE_MAX)
    for size in ksize:
        __check_attr_range("ksize's val", size, _KSIZE_MIN, _KSIZE_MAX)

    if grads_shape[grads_format.index('N')] != -1:
        __check_attr_range("grads n_dim",
                           grads_shape[grads_format.index('N')],
                           _LOWER_RANGE, None)
    if grads_shape[grads_format.index('D')] != -1:
        __check_attr_range("grads d_dim",
                           grads_shape[grads_format.index('D')],
                           _LOWER_RANGE, _UPPER_RANGE)
    if grads_shape[grads_format.index('H')] != -1:
        __check_attr_range("grads h_dim",
                           grads_shape[grads_format.index('H')],
                           _LOWER_RANGE, _UPPER_RANGE)
    if grads_shape[grads_format.index('W')] != -1:
        __check_attr_range("grads w_dim",
                           grads_shape[grads_format.index('W')],
                           _LOWER_RANGE, _UPPER_RANGE)
    __check_attr_range("grads c_dim",
                       grads_shape[grads_format.index('C')],
                       _LOWER_RANGE, None)

    if output_shape[output_format.index('N')] != -1:
        __check_attr_range("output n_dim",
                           output_shape[output_format.index('N')],
                           _LOWER_RANGE, None)
    if output_shape[output_format.index('D')] != -1:
        __check_attr_range("output d_dim",
                           output_shape[output_format.index('D')],
                           _LOWER_RANGE, _UPPER_RANGE)
    if output_shape[output_format.index('H')] != -1:
        __check_attr_range("output h_dim",
                           output_shape[output_format.index('H')],
                           _LOWER_RANGE, _UPPER_RANGE)
    if output_shape[output_format.index('W')] != -1:
        __check_attr_range("output w_dim",
                           output_shape[output_format.index('W')],
                           _LOWER_RANGE, _UPPER_RANGE)
    __check_attr_range("output c_dim",
                       output_shape[output_format.index('C')],
                       _LOWER_RANGE, None)


def _transform_shape_with_format(src_format, to_format, ori_shape):
    # need not to transform
    if src_format == to_format:
        return list(ori_shape)
    res_shape = [1 for _ in range(_ORI_SHAPE_DIM_SIZE)]
    for i in range(_ORI_SHAPE_DIM_SIZE):
        for j in range(_ORI_SHAPE_DIM_SIZE):
            if to_format[i] == src_format[j]:
                res_shape[i] = ori_shape[j]
                break
    return res_shape


def _get_output(fmap, ksize, padf, padb, stride):
    return (fmap + padf + padb - ksize) // stride + 1


def _range_correction(fmap_range, kernel, pads, strides, out_shape):
    w_d, w_h, w_w, _, _ = kernel
    _, strd, strh, strw, _ = strides
    fmap_range_n, fmap_range_d, fmap_range_c1, fmap_range_h, fmap_range_w, fmap_range_c0 = fmap_range
    if not all(i == 0 for i in pads):
        out_d_upper, out_h_upper, out_w_upper = None, None, None
        out_d_lower = util_common.ceil(fmap_range_d[0], strd)
        out_d_upper = util_common.ceil(fmap_range_d[1], strd)
        out_h_lower = util_common.ceil(fmap_range_h[0], strh)
        out_h_upper = util_common.ceil(fmap_range_h[1], strh)
        out_w_lower = util_common.ceil(fmap_range_w[0], strw)
        out_w_upper = util_common.ceil(fmap_range_w[1], strw)
    else:
        out_d_lower = _get_output(fmap_range_d[0], w_d, pads[0], pads[1], strd)
        if out_d_lower < 1:
            fmap_range_d_lower = min(w_d, fmap_range_d[1]) if fmap_range_d[1] else w_d
            fmap_range_d = (fmap_range_d_lower, fmap_range_d[1])
            out_d_lower = _get_output(fmap_range_d[0], w_d, pads[0], pads[1], strd)
        out_d_upper = _get_output(fmap_range_d[1], w_d, pads[0], pads[1], strd)

        out_h_lower = _get_output(fmap_range_h[0], w_h, pads[2], pads[3], strh)
        if out_h_lower < 1:
            fmap_range_h_lower = min(w_h, fmap_range_h[1]) if fmap_range_h[1] else w_h
            fmap_range_h = (fmap_range_h_lower, fmap_range_h[1])
            out_h_lower = _get_output(fmap_range_h[0], w_h, pads[2], pads[3], strh)
        out_h_upper = _get_output(fmap_range_h[1], w_h, pads[2], pads[3], strh)

        out_w_lower = _get_output(fmap_range_w[0], w_w, pads[4], pads[5], strw)
        if out_w_lower < 1:
            fmap_range_w_lower = min(w_w, fmap_range_w[1]) if fmap_range_w[1] else w_w
            fmap_range_w = (fmap_range_w_lower, fmap_range_w[1])
            out_w_lower = _get_output(fmap_range_w[0], w_w, pads[4], pads[5], strw)
        out_w_upper = _get_output(fmap_range_w[1], w_w, pads[4], pads[5], strw)

    range_dedy = [(fmap_range_n[0], fmap_range_n[1]), (out_d_lower, out_d_upper),
                  (util_common.ceil(out_shape[4], _C0_SIZE), util_common.ceil(out_shape[4], _C0_SIZE)),
                  (out_h_lower, out_h_upper), (out_w_lower, out_w_upper), (_C0_SIZE, _C0_SIZE)]

    range_input = [fmap_range_n, fmap_range_d, fmap_range_c1,
                   fmap_range_h, fmap_range_w, fmap_range_c0]

    return range_dedy, range_input


def _shape_correction(grads_shape, orig_input_shape, padding, ksize, strides, shape_format):
    for i, _ in enumerate(grads_shape):
        if grads_shape[i] == -1 and orig_input_shape[i] != -1:
            dim = shape_format[i]
            if dim in ('N' or 'C'):
                grads_shape[i] = orig_input_shape[i]
            elif padding == "SAME":
                grads_shape[i] = ((orig_input_shape[i] + strides[_STRIDES_FORMAT.find(dim)] - 1) /
                                  strides[_STRIDES_FORMAT.find(dim)])
            else:
                grads_shape[i] = ((orig_input_shape[i] - ksize[_KSIZE_FORMAT.find(dim)]) /
                                  strides[_STRIDES_FORMAT.find(dim)]) + 1
    return grads_shape


def _init_dynamic_shape_var(shape_out_backprop, input_sizes, range_dedy, range_input):
    dedy_batch, dedy_depth, dedy_h, dedy_w, dedy_channel = shape_out_backprop

    if input_sizes[0] == -1 or all(i != -1 for i in input_sizes):
        dedy_batch = operation.var("batch_n", range_input[0])
        operation.add_exclude_bound_var(dedy_batch)
        input_sizes[0] = dedy_batch
    if input_sizes[1] == -1:
        dx_depth = operation.var("dedx_d", range_input[1])
        dedy_depth = operation.var("dedy_d", range_dedy[1])
        operation.add_exclude_bound_var(dx_depth)
        operation.add_exclude_bound_var(dedy_depth)
        input_sizes[1] = dx_depth
    if input_sizes[2] == -1:
        dx_h = operation.var("dedx_h", range_input[3])
        dedy_h = operation.var("dedy_h", range_dedy[3])
        operation.add_exclude_bound_var(dx_h)
        operation.add_exclude_bound_var(dedy_h)
        input_sizes[2] = dx_h
    if input_sizes[3] == -1:
        dx_w = operation.var("dedx_w", range_input[4])
        dedy_w = operation.var("dedy_w", range_dedy[4])
        operation.add_exclude_bound_var(dx_w)
        operation.add_exclude_bound_var(dedy_w)
        input_sizes[3] = dx_w

    shape_out_backprop = (dedy_batch, dedy_depth, dedy_h, dedy_w, dedy_channel)

    return shape_out_backprop, input_sizes


def _update_dync_pads(dync_input_size, filter_shape, strides, pads):
    _, fmap_d, fmap_h, fmap_w, _ = dync_input_size
    filter_d, filter_h, filter_w, _, _ = filter_shape
    _, stride_d, stride_h, stride_w, _ = strides

    if all(i == -1 for i in pads):
        pad_h = util_common.align(fmap_h, stride_h) - stride_h + filter_h - fmap_h
        pad_h = tvm.max(pad_h, 0) if isinstance(fmap_h, tvm.Var) else max(pad_h, 0)
        pad_up = pad_h // 2
        pad_down = pad_h - pad_up
        pad_w = util_common.align(fmap_w, stride_w) - stride_w + filter_w - fmap_w
        pad_w = tvm.max(pad_w, 0) if isinstance(fmap_w, tvm.Var) else max(pad_w, 0)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_d = util_common.align(fmap_d, stride_d) - stride_d + filter_d - fmap_d
        pad_d = tvm.max(pad_d, 0) if isinstance(fmap_d, tvm.Var) else max(pad_d, 0)
        pad_head = pad_d // 2
        pad_tail = pad_d - pad_head
        pads = [pad_head, pad_tail, pad_up, pad_down, pad_left, pad_right]
    return list(pads)


def _trans_range_to_6d(fmap_range, fmap_ori_format):
    fmap_range_n, fmap_range_d, fmap_range_h, fmap_range_w, fmap_range_c = _transform_shape_with_format(
        fmap_ori_format, _FMAP_TARGET_FORMAT, fmap_range)
    fmap_range_c0 = [_C0_SIZE, _C0_SIZE]
    fmap_range_c1 = [(fmap_range_c[0] + _C0_SIZE - 1) / _C0_SIZE, (fmap_range_c[1] + _C0_SIZE - 1) / _C0_SIZE]

    return [fmap_range_n, fmap_range_d, fmap_range_c1, fmap_range_h, fmap_range_w, fmap_range_c0]


def _fuzz_shape_range(tenor_dict):
    """
    fuzz the shape and range for input and output
    """
    ori_shape = [DYNAMIC_MODE] * _ORI_SHAPE_DIM_SIZE
    ori_range = [UNLIMIT_RANGE] * _ORI_SHAPE_DIM_SIZE
    c_index = tenor_dict.get("ori_format").find("C")
    c_dim = tenor_dict.get("ori_shape")[c_index]
    ori_shape[c_index] = c_dim
    ori_range[c_index] = [c_dim, c_dim]
    tenor_dict["ori_shape"] = ori_shape
    tenor_dict["shape"] = ori_shape
    tenor_dict["ori_range"] = ori_range
    tenor_dict["range"] = ori_range

    return tenor_dict


@tbe_register.register_param_generalization("AvgPool3DGrad")
def avg_pool3d_grad_generalization(orig_input_shape,
                                   grads,
                                   filter,
                                   output,
                                   ksize,
                                   strides,
                                   pads,
                                   ceil_mode=False,
                                   count_include_pad=True,
                                   divisor_override=0,
                                   data_format="NDHWC",
                                   kernel_name="avg_pool3d_grad",
                                   generalize_config=None):
    if generalize_config.get("mode") == "keep_rank":
        # dynamic mode not supported ceil_mode,  count_include_pad, and divisor_override
        if ceil_mode or count_include_pad or divisor_override != 0:
            return LOWER_STR
        grads = _fuzz_shape_range(grads)
        # handle the first input orig_input_shape
        orig_input_shape["const_value"] = None
        orig_input_shape["shape"] = [DYNAMIC_MODE]
        orig_input_shape["ori_shape"] = [DYNAMIC_MODE]
        orig_input_shape["range"] = [UNLIMIT_RANGE]
        orig_input_shape["ori_range"] = [UNLIMIT_RANGE]

        return [[orig_input_shape, grads, filter, output, ksize, strides, pads,
                 ceil_mode, count_include_pad, divisor_override, data_format]]
    return


@register_operator("AvgPool3DGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_STR,
                            para_check.KERNEL_NAME
                            )
def avg_pool3d_grad(orig_input_shape,
                    grads,
                    filter,
                    output,
                    ksize,
                    strides,
                    pads,
                    ceil_mode=False,
                    count_include_pad=True,
                    divisor_override=0,
                    data_format="NDHWC",
                    kernel_name="avg_pool3d_grad"):
    """
    computes average pooling3d backwards gradients.

    Parameters:
    -----------
    orig_input_shape: dict, shape and dtype of dedx, only support int32, shape is 1dims, format is ND

    grads : dict, shape and dtype of input_data, only support float16, shape is 5dims, format is NDC1HWC0

    filter : dict, fractal_z_3d layout, float16 dtype

    output : dict, shape and dtype of output_data, only support float16

    ksize : list or tuple, the window of avg_pool3d, only support avg_pool3d in D or H or W

    strides: list or tuple, the window of avg_pool3d, only support avg_pool3d in D or H or W

    pads : list or tuple, count of padding zero or d, h, w axis

    ceil_mode : when True, will use ceil mode instead of floor in the formula to compute the output shape

    count_include_pad : when True, will include the zero-padding in the averaging calculation

    divisor_override : if specified, it will be used as divisor, otherwise size of the pooling region will be used

    data_format : str, default value is "NDHWC"

    kernel_name : cce kernel name, default value is "avg_pool3d_grad"

    Returns
    -------
    None
    """
    check_empty_tensor(grads, filter, output, strides, pads)
    grads_ori_format = grads.get("ori_format")
    grads_ori_shape = grads.get("ori_shape")
    grads_dtype = grads.get("dtype").lower()
    fmap_ori_shape = output.get("ori_shape")
    fmap_dtype = output.get("dtype")
    filter_ori_shape = filter.get("ori_shape")
    fitler_ori_format = filter.get("ori_format")
    filter_dtype = filter.get("dtype").lower()
    fmap_range = output.get("range")

    if len(fmap_range) == _ORI_SHAPE_DIM_SIZE:
        fmap_range = _trans_range_to_6d(fmap_range, data_format)

    _check_inputs(grads, filter, output, fmap_range, ksize, strides, pads, ceil_mode,
                  count_include_pad, divisor_override, data_format)

    fmap_ori_shape_formated = _transform_shape_with_format(data_format,
                                                           _FMAP_TARGET_FORMAT,
                                                           fmap_ori_shape)

    grads_ori_shape_formated = _transform_shape_with_format(grads_ori_format,
                                                            _GRADS_TARGET_FORMAT,
                                                            grads_ori_shape)

    filter_ori_shape_formated = _transform_shape_with_format(fitler_ori_format,
                                                             _FILTER_TARGET_FORMAT,
                                                             filter_ori_shape)

    ksize_formated = _transform_shape_with_format(data_format,
                                                  _KSIZE_FORMAT,
                                                  ksize)

    strides_formated = _transform_shape_with_format(data_format,
                                                    _STRIDES_FORMAT,
                                                    strides)

    padding = "VALID" if all(i == 0 for i in pads) else "SAME"
    # correct -1 in dedy shape by dedx shape
    grads_ori_shape_formated = _shape_correction(grads_ori_shape_formated,
                                                 fmap_ori_shape_formated,
                                                 padding,
                                                 ksize_formated,
                                                 strides_formated,
                                                 _GRADS_TARGET_FORMAT)
    # correct dedy range by fmap range
    range_dedy, range_input = _range_correction(fmap_range, filter_ori_shape_formated,
                                                pads, strides_formated, fmap_ori_shape_formated)
    # init dynamic shape tbe var
    dync_grads_ori_shape, dync_ori_input_size = _init_dynamic_shape_var(grads_ori_shape_formated,
                                                                        fmap_ori_shape_formated,
                                                                        range_dedy,
                                                                        range_input)

    dync_grads_shape = (dync_grads_ori_shape[0], dync_grads_ori_shape[1],
                        util_common.ceil(dync_grads_ori_shape[4], _C0_SIZE),
                        dync_grads_ori_shape[2], dync_grads_ori_shape[3], _C0_SIZE)
    dync_input_size = (dync_ori_input_size[0], dync_ori_input_size[1],
                       util_common.ceil(dync_ori_input_size[4], _C0_SIZE),
                       dync_ori_input_size[2], dync_ori_input_size[3], _C0_SIZE)
    dync_pads = _update_dync_pads(dync_ori_input_size, filter_ori_shape_formated, strides_formated, pads)

    ori_input_shape_plh = tvm.placeholder([5], name="ori_input_shape", dtype="int32")
    grads = tvm.placeholder(dync_grads_shape, name="dedy", dtype=grads_dtype,
                            attrs={"ori_shape": dync_grads_ori_shape,
                                   "ori_format": _GRADS_TARGET_FORMAT,
                                   "data_type": "float16"}
                            )
    output = tvm.placeholder(dync_input_size, name="output", dtype=fmap_dtype,
                             attrs={"ori_shape": dync_ori_input_size,
                                    "ori_format": _FMAP_TARGET_FORMAT,
                                    "data_type": "float16"}
                            )

    _, kd, kh, kw, _ = ksize_formated
    _, strd, strh, strw, _ = strides_formated
    w_ori_shape = (kd, kh, kw, 1, dync_ori_input_size[4])
    filter_frac_z = (dync_grads_shape[2] * kd * kh * kw, 1, _C0_SIZE, _C0_SIZE)
    filter = tvm.placeholder(filter_frac_z,
                             name="filter",
                             dtype="float16",
                             attrs={"ori_shape": w_ori_shape,
                                    "ori_format": "DHWCN",
                                    "data_type": "float16"}
                            )

    mean_matrix = tvm.compute(
        dync_grads_shape,
        lambda n, d, c1, h, w, c0:(
            (tvm.min(d * strd + kd, dync_pads[0] + dync_ori_input_size[1]) - tvm.max(dync_pads[0], d * strd)) *
            (tvm.min(h * strh + kh, dync_pads[2] + dync_ori_input_size[2]) - tvm.max(dync_pads[2], h * strh)) *
            (tvm.min(w * strw + kw, dync_pads[4] + dync_ori_input_size[3]) -
             tvm.max(dync_pads[4], w * strw))).astype("int"),
        name="mean_matrix_init")

    mean_matrix_fp16 = tvm.compute(dync_grads_shape, lambda *index:
                                   mean_matrix(*index).astype("float16"),
                                   name="mean_matrix_fp16")
    mul_res = tvm.compute(dync_grads_shape, lambda n, d, c1, h, w, c0:
                    grads[n][d][c1][h][w][c0] / mean_matrix_fp16[n][d][c1][h][w][c0],
                    name="mean_matrix_mul", tag="mean_matrix_mul",
                    attrs={"ori_shape": dync_grads_ori_shape,
                           "ori_format": _GRADS_TARGET_FORMAT,
                           "data_type": "float16"})

    dilations = (1, 1, 1, 1, 1)
    group_dict = util_common.calculate_group(fmap_ori_shape_formated[4],
                                             filter_ori_shape_formated[4],
                                             fmap_ori_shape_formated[4],
                                             _C0_SIZE,
                                             _C0_SIZE)

    para_dict = {
        "strides": strides_formated,
        "pads": dync_pads,
        "dilations": dilations,
        "res_dtype": fmap_dtype,
        "kernel_name": kernel_name,
        "group_dict": group_dict,
        "fused_num": _UB_FUSED_OP_NUM
    }

    check_conv3dbp_input_params(filter_ori_shape_formated, dync_grads_ori_shape,
                                dync_ori_input_size, strides_formated, pads,
                                dilations, filter_dtype, grads_dtype,
                                fmap_dtype, kernel_name,
                                range_input, range_dedy,
                                para_dict)

    with tbe.compute():
        dedx = tbe.conv3d_backprop_input(
                    filter=filter,
                    out_backprop=mul_res,
                    filter_size=[filter_ori_shape_formated[4], 1, kd, kh, kw],
                    input_size=dync_ori_input_size,
                    para_dict=para_dict
        )
    tensor_list = [ori_input_shape_plh, grads, filter, dedx]

    with tvm.target.cce():
        sch = tbe.auto_schedule(dedx)

    config = {"name": kernel_name,
              "tensor_list": tensor_list,
              "dummy_placeholder": True,
              "build_args": {"constant_realize_extent_in_infer_bound": True}}
    tbe.build(sch, config)
