#!/usr/bin/python
# -*- coding: utf-8 -*-
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
avg_pool3d_d
"""
from impl.conv3d import conv3d_fusion_compute
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_build
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class of Constant
    """
    MAX_BLOCK_NUM = 65535


# 'pylint: disable = unused-argument,redefined-builtin,too-many-arguments,too-many-boolean-expressions,too-many-locals
def get_op_support_info(x,
                        filter,
                        multiplier,
                        y,
                        ksize,
                        strides,
                        pads,
                        ceil_mode=False,
                        count_include_pad=True,
                        divisor_override=0,
                        data_format="NDHWC",
                        kernel_name="avg_pool3d_d"):
    """
    get avg_pool_3d slice info
    """
    format_x = x.get("format")
    shape_x = x.get("shape")
    if len(ksize) == 1:
        ksize_d = ksize[0]
        ksize_h = ksize[0]
    elif len(ksize) == 3:
        ksize_d = ksize[0]
        ksize_h = ksize[1]
    elif len(ksize) == 5:
        ksize_d = ksize[1]
        ksize_h = ksize[2]
        if data_format == "NCDHW":
            ksize_d = ksize[2]
            ksize_h = ksize[3]

    if format_x == "NDC1HWC0":
        input_d = shape_x[1]
        input_h = shape_x[3]
        if input_d == ksize_d and input_h == ksize_h:
            axis_split_matrix = [[SplitInput([0, [0], [-1], [-1]]),
                                  SplitOutput([0, [0]])]]
        elif input_d == ksize_d and input_h != ksize_h:
            axis_split_matrix = [[SplitInput([0, [0], [-1], [-1]]),
                                  SplitOutput([0, [0]])],
                                 [SplitInput([0, [3], [0], [0]]),
                                  SplitOutput([0, [3]])]]
        elif input_d != ksize_d and input_h != ksize_h:
            axis_split_matrix = [[SplitInput([0, [0], [-1], [-1]]),
                                  SplitOutput([0, [0]])],
                                 [SplitInput([0, [1], [0], [0]]),
                                  SplitOutput([0, [1]])],
                                 [SplitInput([0, [3], [0], [0]]),
                                  SplitOutput([0, [3]])]]
        elif input_d != ksize_d and input_h == ksize_h:
            axis_split_matrix = [[SplitInput([0, [0], [-1], [-1]]),
                                  SplitOutput([0, [0]])],
                                 [SplitInput([0, [1], [0], [0]]),
                                  SplitOutput([0, [1]])]]
        else:
            axis_split_matrix = None
        axis_reduce_list = None
    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)

    return op_cal_info_in_json


def _check_window_rule(ksize, strides, pads):
    """
    avg_pool3d_check_window_rule

    Parameters
    ----------
    ksize: kernel size
    strides: stride

    Returns
    -------
    None
    """
    if len(ksize) != 5:
        error_manager_vector.raise_err_input_param_range_invalid('avg_pool3d', 'ksize', 5, 5, len(ksize))

    if len(strides) != 5:
        error_manager_vector.raise_err_input_param_range_invalid('avg_pool3d', 'strides', 5, 5, len(strides))

    if len(pads) != 6:
        error_manager_vector.raise_err_input_param_range_invalid('avg_pool3d', 'pads', 6, 6, len(pads))


# 'pylint: disable = too-many-boolean-expressions
def _check_global_rule(input_shape, ksize, pads):
    ksize_h = ksize[2]
    ksize_w = ksize[3]
    fmap_h = input_shape[3]
    fmap_w = input_shape[4]
    if ksize_h != fmap_h or ksize_w != fmap_w:
        error_manager_vector.raise_err_check_params_rules('avg_pool3d', 'only support slide on D dimension now',
                                                          'ksize', ksize)

    if pads[0] != 0 or pads[1] != 0 or pads[2] != 0 or pads[3] != 0 or pads[4] != 0 or pads[5] != 0:
        error_manager_vector.raise_err_pad_mode_invalid('avg_pool3d', 'valid', 'other')


# 'pylint: disable=too-many-arguments,unused-argument,invalid-name
def _avg_pool3d_check_rule(input_shape, input_dtype, ksize, strides, pads, kernel_name):
    """
    _avg_pool3d_check_rule

    Parameters
    ----------
    input_shape: input shape
    input_dtype: output dtype
    ksize: kernel size
    strides: strides
    pads: zero paddings on both sides
    data_format: must be "NDHWC"
    kernel_name: kernel name

    Returns
    -------
    None
    """
    para_check.check_shape(input_shape)
    para_check.check_dtype(input_dtype, ("float16", ))
    para_check.check_kernel_name(kernel_name)
    _check_window_rule(ksize, strides, pads)


# 'pylint: disable=too-many-locals,too-many-arguments,unused-argument,invalid-name
def avg_pool3d_compute(x,
                       filter,
                       multiplier,
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
    avg_pool3d compute

    Parameters
    ----------
    x: input tensor dict
    filter: filter tensor dict
    multiplier: multiplier tensor dict
    y: output tensor dict
    ksize: kernel size
    strides: strides
    padding: padding mode, str
    data_format: must be "NDHWC"
    kernel_name: kernel name

    Returns
    -------
    output tensor
    """
    shape = x.shape

    ksize_d = ksize[0]
    stride_d = strides[0]
    size_kernel = (ksize[0] * ksize[1] * ksize[2])
    if divisor_override != 0:
        size_kernel = divisor_override

    cast_type = "float16"
    if tbe_platform.intrinsic_check_support("Intrinsic_vadd", "float32"):
        cast_type = "float32"

    # copy gm to ub
    tensor_in_ub = tvm.compute(shape, lambda *i: x[i], name="tensor_in_ub")

    tensor_in_ub_cast = tvm.compute(shape, lambda *i: tensor_in_ub(*i).astype(cast_type), name="tensor_in_ub_cast")

    d_axis = tvm.reduce_axis((0, ksize_d), "d_sum")
    hw_axis = tvm.reduce_axis((0, shape[3]), "hw_sum")
    origin_d = shape[1]
    reduced_d = 1 + (origin_d - ksize_d) // stride_d
    shape_d_hw = (shape[0], reduced_d, shape[2], 1, shape[4])
    tensor_d_hw = tvm.compute(shape_d_hw,
                              lambda n, d, c1, hw, c0: tvm.sum(
                                  tensor_in_ub_cast[n, d * stride_d + d_axis, c1, hw_axis, c0], axis=[d_axis, hw_axis]),
                              name="tensor_d_hw")

    tensor_a = tvm.compute(
        shape_d_hw,
        lambda n, d, c1, hw, c0: tensor_d_hw[n, d, c1, hw, c0] * tvm.const(1.0 / size_kernel, dtype=cast_type),
        name="tensor_a")

    res_cast = tvm.compute(shape_d_hw, lambda *i: tensor_a(*i).astype("float16"), name="res_cast")

    res = tvm.compute(shape_d_hw, lambda *i: res_cast[i], name='res')
    return res


# 'pylint: disable=too-many-return-values
def _tiling_param(shape, ksize, strides, core_num):
    d = shape[1]
    hw = shape[3]
    c0 = shape[4]
    ksize_d = ksize[0]
    stride_d = strides[0]
    d_out = 1 + (d - ksize_d) // stride_d

    total_ub_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    factor_c1 = 1
    # 5 for five tensors in ub, 4 for a fp32 data four bytes
    if hw * 5 * c0 * 4 > total_ub_bytes:
        # split reduce hw axis
        factor_dout = 1
        factor_rd = 1
        factor_rhw = total_ub_bytes // (5 * c0 * 4)
    elif ksize_d * hw * 5 * c0 * 4 > total_ub_bytes:
        # split reduce d axis
        factor_dout = 1
        factor_rd = total_ub_bytes // (hw * 5 * c0 * 4)
        factor_rhw = hw
    elif d_out * (stride_d + ksize_d) * hw * c0 * 5 * 4 > total_ub_bytes:
        # split d_out axis
        factor_dout = max(1, total_ub_bytes // ((stride_d + ksize_d) * hw * c0 * 5 * 4))
        factor_rd = ksize_d
        factor_rhw = hw
    else:
        # do not split any axis
        factor_dout = d_out
        factor_rd = ksize_d
        factor_rhw = hw

    return factor_c1, factor_dout, factor_rd, factor_rhw


# 'pylint: disable=too-many-statements
def _avg_pool3d_schedule(res, sch, ksize, strides):
    res_cast = res.op.input_tensors[0]
    tensor_a = res_cast.op.input_tensors[0]
    tensor_d_hw = tensor_a.op.input_tensors[0]
    tensor_in_ub_cast = tensor_d_hw.op.input_tensors[0]
    tensor_in_ub = tensor_in_ub_cast.op.input_tensors[0]

    input_shape = [int(i) for i in tensor_in_ub.shape]
    # set scope
    sch[tensor_in_ub].set_scope(tbe_platform.scope_ubuf)
    sch[tensor_in_ub_cast].set_scope(tbe_platform.scope_ubuf)
    sch[tensor_d_hw].set_scope(tbe_platform.scope_ubuf)
    sch[tensor_a].set_scope(tbe_platform.scope_ubuf)
    sch[res_cast].set_scope(tbe_platform.scope_ubuf)

    core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

    ax_res_n = res.op.axis[0]
    ax_res_do = res.op.axis[1]
    ax_res_c1 = res.op.axis[2]
    ax_res_hw = res.op.axis[3]
    ax_res_c0 = res.op.axis[4]

    ax_dhw_n = tensor_d_hw.op.axis[0]
    ax_dhw_do = tensor_d_hw.op.axis[1]
    ax_dhw_c1 = tensor_d_hw.op.axis[2]
    ax_dhw_hw = tensor_d_hw.op.axis[3]
    ax_dhw_c0 = tensor_d_hw.op.axis[4]
    ax_dhw_rd = tensor_d_hw.op.reduce_axis[0]
    ax_dhw_rhw = tensor_d_hw.op.reduce_axis[1]

    factor_c1, factor_dout, factor_reduce_d, factor_reduce_hw = _tiling_param(input_shape, ksize, strides, core_num)

    reduce_hw_o, reduce_hw_i = sch[tensor_d_hw].split(ax_dhw_rhw, factor=factor_reduce_hw)
    reduce_d_o, reduce_d_i = sch[tensor_d_hw].split(ax_dhw_rd, factor=factor_reduce_d)
    dhw_do_o, dhw_do_i = sch[tensor_d_hw].split(ax_dhw_do, factor=factor_dout)

    sch[tensor_d_hw].reorder(ax_dhw_n, ax_dhw_c1, ax_dhw_hw, dhw_do_o, reduce_d_o, reduce_hw_o, dhw_do_i, reduce_d_i,
                             reduce_hw_i, ax_dhw_c0)

    ax_res_c1_o, ax_res_c1_i = sch[res].split(ax_res_c1, factor=factor_c1)
    ax_res_do_o, ax_res_do_i = sch[res].split(ax_res_do, factor=factor_dout)

    sch[res].reorder(ax_res_n, ax_res_c1_o, ax_res_do_o, ax_res_c1_i, ax_res_do_i, ax_res_hw, ax_res_c0)

    ax_fused = sch[res].fuse(ax_res_n, ax_res_c1_o)

    ax_fused_o = ax_fused
    output_shape = [int(i) for i in res.shape]
    if (output_shape[0] * ((output_shape[2] + factor_c1 - 1) // factor_c1)) > Constant.MAX_BLOCK_NUM:
        ax_fused_o, _ = sch[res].split(ax_fused, nparts=core_num)

    block = tvm.thread_axis("blockIdx.x")
    sch[res].bind(ax_fused_o, block)

    sch[tensor_in_ub].compute_at(sch[tensor_d_hw], reduce_hw_o)
    sch[tensor_in_ub_cast].compute_at(sch[tensor_d_hw], reduce_hw_o)
    sch[res_cast].compute_at(sch[res], ax_res_do_o)
    sch[tensor_a].compute_at(sch[res], ax_res_do_o)
    sch[tensor_d_hw].compute_at(sch[res], ax_res_do_o)

    if tensor_in_ub.dtype != tensor_in_ub_cast.dtype:
        sch[tensor_in_ub_cast].emit_insn(sch[tensor_in_ub_cast].op.axis[0], tbe_platform.CAST)
        sch[res_cast].emit_insn(sch[res_cast].op.axis[0], tbe_platform.CAST)
    sch[tensor_in_ub].emit_insn(sch[tensor_in_ub].op.axis[0], tbe_platform.DMA_COPY)
    sch[tensor_d_hw].emit_insn(dhw_do_i, tbe_platform.REDUCE_SUM)
    sch[tensor_a].emit_insn(sch[tensor_a].op.axis[0], tbe_platform.MUL)
    sch[res].emit_insn(ax_res_c1_i, tbe_platform.DMA_COPY)


def check_vector_impl(input_shape, ksize, pads):
    """
    check vector impl shape ksize pads
    """
    for pad in pads:
        if pad != 0:
            return False
    input_h = input_shape[3]
    input_w = input_shape[4]
    ksize_h = ksize[1]
    ksize_w = ksize[2]

    if ksize_h != input_h:
        return False
    if ksize_w != input_w:
        return False
    return True


def correct_pads(input_shape, ksize, strides, pads):
    """
    calculate correct pads
    """
    input_d = input_shape[1]
    input_h = input_shape[3]
    input_w = input_shape[4]
    ksize_d = ksize[0]
    ksize_h = ksize[1]
    ksize_w = ksize[2]
    stride_d = strides[0]
    stride_h = strides[1]
    stride_w = strides[2]
    pad_before, pad_after, pad_top, pad_bottom, pad_left, pad_right = pads
    do = (input_d - ksize_d + pad_before + pad_after + stride_d - 1) // stride_d + 1
    ho = (input_h - ksize_h + pad_top + pad_bottom + stride_h - 1) // stride_h + 1
    wo = (input_w - ksize_w + pad_left + pad_right + stride_w - 1) // stride_w + 1

    if (do - 1) * stride_d >= input_d + pads[0]:
        do = do - 1
    if (ho - 1) * stride_h >= input_h + pads[2]:
        ho = ho - 1
    if (wo - 1) * stride_w >= input_w + pads[4]:
        wo = wo - 1
    if do > 1:
        pad_after = max((do - 1) * stride_d + ksize_d - input_d - pad_before, 0)
    if ho > 1:
        pad_bottom = max((ho - 1) * stride_h + ksize_h - input_h - pad_top, 0)
    if wo > 1:
        pad_right = max((wo - 1) * stride_w + ksize_w - input_w - pad_left, 0)

    return [pad_before, pad_after, pad_top, pad_bottom, pad_left, pad_right]


def _transform_shape_with_format(ori_format, shape):
    idx_d = ori_format.find('D')
    idx_h = ori_format.find('H')
    idx_w = ori_format.find('W')
    shape_all = [1, 1, 1, 1, 1]
    if len(shape) not in (1, 3, 5):
        error_manager_vector.raise_err_check_params_rules('avg_pool3d', 'ksize or strides size error', 'shape', shape)
    if len(shape) == 1:
        shape_dhw = (shape[0], shape[0], shape[0])
        shape_all[idx_d] = shape[0]
        shape_all[idx_h] = shape[0]
        shape_all[idx_w] = shape[0]
    elif len(shape) == 3:
        shape_dhw = shape
        shape_all[idx_d] = shape[0]
        shape_all[idx_h] = shape[1]
        shape_all[idx_w] = shape[2]
    elif len(shape) == 5:
        shape_dhw = (shape[idx_d], shape[idx_h], shape[idx_w])
        shape_all = shape
    return tuple(shape_all), shape_dhw


# 'pylint: disable=too-many-arguments,unused-argument,invalid-name
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def avg_pool3d_d(x,
                 filter,
                 multiplier,
                 y,
                 ksize,
                 strides,
                 pads,
                 ceil_mode=False,
                 count_include_pad=True,
                 divisor_override=0,
                 data_format="NDHWC",
                 kernel_name="avg_pool3d_d"):
    """
    Parameters
    ----------
    x : dict, shape and dtype of input_data,
    only support float16, shape is 5 dims, format is NDC1HWC0

    filter : dict, fractal_z_3d layout, float16 dtype

    multiplier : dict, NDC1HWC0 layout, float16 dtype

    y : dict, shape and dtype of output_data, only support float16

    ksize : list or tuple, the window of avg_pool3d,
    only support avg_pool3d in D or H or W

    strides : list or tuple, the stride of avg_pool3d window,
    only support avg_pool3d in D or H or W

    pads : list or tuple, count of padding zero or d,h,w axis

    ceil_mode: when True, will use ceil instead of floor in the formula to
    compute the output shape

    count_include_pad: when True, will include the zero-padding in the
    averaging calculation.

    divisor_override: if specified, it will be used as divisor, otherwise size
    of the pooling region will be used.

    data_format : str, default = "NDHWC"

    kernel_name : cce kernel name, default value is "avg_pool3d_d"

    Returns
    -------
    None

    Notice
    -------
    Only support global model currently.
    """
    input_ori_format = x.get("ori_format")
    input_shape = x.get("shape")
    input_dtype = x.get("dtype")
    input_dtype = input_dtype.lower()

    ksize, ksize_dhw = _transform_shape_with_format(input_ori_format, ksize)
    strides, strides_dhw = _transform_shape_with_format(input_ori_format, strides)

    _avg_pool3d_check_rule(input_shape, input_dtype, ksize, strides, pads, kernel_name)

    if ceil_mode:
        pads = correct_pads(input_shape, ksize_dhw, strides_dhw, pads)

    if check_vector_impl(input_shape, ksize_dhw, pads):
        # create tensor_in
        input_shape = (input_shape[0], input_shape[1], input_shape[2], input_shape[3] * input_shape[4], input_shape[5])
        tensor_in = tvm.placeholder(input_shape, name="tensor_in", dtype=input_dtype)

        res = avg_pool3d_compute(tensor_in, filter, multiplier, y, ksize_dhw, strides_dhw, pads, ceil_mode,
                                 count_include_pad, divisor_override, data_format, kernel_name)

        # schedule
        sch = tvm.create_schedule(res.op)

        _avg_pool3d_schedule(res, sch, ksize_dhw, strides_dhw)

        with tbe_build.build_config():
            tvm.build(sch, [tensor_in, res], "cce", name=kernel_name)
    else:
        dilations = (1, 1, 1, 1, 1)
        offset_w = None
        bias = None
        data_format = input_ori_format
        fmap = tvm.placeholder(input_shape, name="fmap", dtype=input_dtype, attrs={"ori_shape": x.get("ori_shape")})
        fmap_c = x.get('ori_shape')[data_format.find('C')]
        groups = fmap_c
        kd = ksize_dhw[0]
        kh = ksize_dhw[1]
        kw = ksize_dhw[2]
        w_ori_shape = (kd, kh, kw, 1, fmap_c)
        c1 = x.get('shape')[2]
        filter_frac_z = (c1 * kd * kh * kw, 1, 16, 16)
        filter = tvm.placeholder(filter_frac_z, name="filter", dtype="float16",
                                 attrs={"ori_shape": w_ori_shape, 'ori_format': 'DHWCN'})
        conv_res = conv3d_fusion_compute(fmap, filter, bias, offset_w, y, strides, pads, dilations, groups=groups,
                                         data_format=data_format, kernel_name=kernel_name)
        tensor_list = [fmap, filter, conv_res]
        if multiplier:
            mul_n, mul_d, mul_c1, mul_h, mul_w, mul_c0 = multiplier.get('shape')
            mul_shape = (mul_n * mul_d, mul_c1, mul_h * mul_w, mul_c0)
            multiplier = tvm.placeholder(mul_shape, name="multiplier", dtype="float16")
            res = tbe.vmul(conv_res, multiplier)
            tensor_list = [fmap, filter, multiplier, res]

        with tvm.target.cce():
            sch = tbe.auto_schedule(tensor_list[-1])
        config = {"name": kernel_name, "tensor_list": tensor_list}
        tbe.build(sch, config)
