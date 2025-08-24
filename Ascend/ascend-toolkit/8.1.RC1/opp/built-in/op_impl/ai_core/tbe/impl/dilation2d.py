#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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
dilation2d
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tik
from impl.add import add
from impl import common_util


# 'pylint: disable=too-few-public-methods,not-use-list-comprehension
class Constant:
    """
    The class for constant
    """
    MIN_VAL = -65534.0


# 'pylint: disable=invalid-name,unused-argument,too-many-arguments,too-many-locals,too-many-branches
# 'pylint: disable=redefined-builtin,too-many-lines,too-many-instance-attributes,too-many-statements
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_LIST_INT, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def dilation2d(x,
               filter,
               y,
               strides,
               rates,
               padding_mode="SAME",
               pads=(0, 0, 0, 0),
               ceil_mode=False,
               data_format="NHWC",
               kernel_name="dilation2d"):
    """
    Returns the grayscale dilation of x and filter tensors

    Parameters
    ----------
    x: dict, dict of x, include keys(shape and dtype)
    filter: dict, dict of filter
    y: dict, dict of output
    strides: list or tuple, the stride of sliding window, only support in H or W
    rates: list or tuple, the input stride for atrous morphological dilation, only support in H or W
    padding_mode: str, the mode of padding, support padding and not padding
    pads: list or tuple, the fill value of input
    ceil_mode: bool, use ceil or floor to calculate ho and wo when padding_mode is CALCULATED
    data_format: str, default = "NHWC"
    kernel_name: str, cce kernel name, default value is "dilation2d"

    Returns
    -------
    tik_instance: tik_instance
    """
    x_format = x.get("format")
    x_ori_format = x.get("ori_format")
    x_shape = x.get("shape")
    x_dtype = x.get("dtype").lower()
    filter_format = filter.get("format")
    filter_shape = filter.get("shape")
    check_list = ["float16"]

    para_check.check_dtype(x_dtype, check_list, param_name="x")
    para_check.check_format(x_format, "NC1HWC0", param_name="x")
    para_check.check_shape(x_shape, min_rank=5, max_rank=5, param_name="x")
    para_check.check_format(filter_format, "NC1HWC0", param_name="filter")
    para_check.check_shape(filter_shape, min_rank=5, max_rank=5, param_name="filter")
    if padding_mode not in ("SAME", "VALID", "CALCULATED"):
        error_manager_vector.raise_err_pad_mode_invalid(kernel_name, "SAME,VALID,CALCULATED", padding_mode)
    if len(strides) != 4:
        error_manager_vector.raise_err_check_params_rules(kernel_name, "the length of strides should be 4", "strides",
                                                          strides)
    if len(rates) != 4:
        error_manager_vector.raise_err_check_params_rules(kernel_name, "the length of rates should be 4", "rates",
                                                          rates)
    if len(pads) != 4:
        error_manager_vector.raise_err_check_params_rules(kernel_name, "the length of pads should be 4", "pads", pads)
    input_h = x_shape[2]
    input_w = x_shape[3]
    filter_h = filter_shape[2]
    filter_w = filter_shape[3]
    if x_ori_format == "NHWC":
        stride_n, stride_h, stride_w, stride_c = strides
        rate_n, rate_h, rate_w, rate_c = rates
    else:
        stride_n, stride_c, stride_h, stride_w = strides
        rate_n, rate_c, rate_h, rate_w = rates
    window_h = (filter_h - 1) * rate_h + 1
    window_w = (filter_w - 1) * rate_w + 1

    if stride_n != 1 or stride_c != 1:
        error_manager_vector.raise_err_check_params_rules(kernel_name, "n,c of strides should be 1", "strides", strides)
    if rate_n != 1 or rate_c != 1:
        error_manager_vector.raise_err_check_params_rules(kernel_name, "n,c of rates should be 1", "rates", rates)

    if list(pads) == [0, 0, 0, 0] and window_h == input_h and window_w == input_w:
        if padding_mode == "CALCULATED":
            padding_mode = "VALID"
        if padding_mode == "SAME" and stride_h == input_h and stride_w == input_w:
            padding_mode = "VALID"

    input_params = {
        "x": x,
        "filter": filter,
        "y": y,
        "stride_h": stride_h,
        "stride_w": stride_w,
        "rate_h": rate_h,
        "rate_w": rate_w,
        "window_h": window_h,
        "window_w": window_w,
        "padding_mode": padding_mode,
        "pads": pads,
        "ceil_mode": ceil_mode,
        "data_format": data_format,
        "kernel_name": kernel_name
    }
    cal_pads = _cal_pads(input_params)
    input_params["pads"] = cal_pads.get("cal_pads")
    if filter_h == 1 and filter_w == 1 and stride_w == 1 and stride_h == 1:
        add(x, filter, y, kernel_name)
        return None
    dilation = Dilation2D(input_params)
    dilation.dilation_compute()
    dilation.instance.BuildCCE(kernel_name=kernel_name,
                               inputs=[dilation.x_gm, dilation.filter_gm],
                               outputs=[dilation.y_gm])
    return dilation.instance


def check_supported(x,
                    filter,
                    y,
                    strides,
                    rates,
                    padding_mode="SAME",
                    pads=(0, 0, 0, 0),
                    ceil_mode=False,
                    data_format="NHWC",
                    kernel_name="dilation2d"):
    """
    verify the types and params of dilation2d supported by tbe
    """
    x_shape = x.get("shape")
    x_format = x.get("format")
    filter_shape = filter.get("shape")

    if data_format == "NHWC":
        stride_h = strides[1]
        stride_w = strides[2]
        rate_h = rates[1]
        rate_w = rates[2]
    elif data_format == "NCHW":
        stride_h = strides[2]
        stride_w = strides[3]
        rate_h = rates[2]
        rate_w = rates[3]
    else:
        reason = "data_format[%s] is not supported by aicore" % data_format
        return False, reason

    if x_format == "NHWC":
        filter_h = filter_shape[0]
        filter_w = filter_shape[1]
        x_h = x_shape[1]
        x_w = x_shape[2]
    elif x_format == "NCHW":
        filter_h = filter_shape[1]
        filter_w = filter_shape[2]
        x_h = x_shape[2]
        x_w = x_shape[3]
    elif x_format == "NC1HWC0":
        filter_h = filter_shape[2]
        filter_w = filter_shape[3]
        x_h = x_shape[2]
        x_w = x_shape[3]
    else:
        reason = "x_format[%s] is not supported by aicore" % x_format
        return False, reason

    window_h = (filter_h - 1) * rate_h + 1
    window_w = (filter_w - 1) * rate_w + 1

    if window_w < 1 or window_w > 255 or window_h < 1 or window_h > 255 or window_h * window_w > 512:
        reason = "size of window is not supported, window_w is %s, window_h is %s" % (
            str(window_w), str(window_h))
        return False, reason
    if stride_h < 1 or stride_h > 255 or stride_w < 1 or stride_w > 255:
        reason = "size of stride is not supported, stride_h is %s, stride_w is %s" % (
            str(stride_h), str(stride_w))
        return False, reason
    if window_w > x_w or window_h > x_h:
        reason = "window_w gt x_w or window_h gt x_h,window_w is %s, window_h is %s, x_shape is %s" \
                 % (str(window_w), str(window_h), str(x_shape))
        return False, reason
    return True, ""


def _cal_pads(params):
    """
    calculate pad values
    """
    input_h = params.get("x").get("shape")[2]
    input_w = params.get("x").get("shape")[3]
    stride_h = params.get("stride_h")
    stride_w = params.get("stride_w")
    window_h = params.get("window_h")
    window_w = params.get("window_w")
    if params.get("padding_mode") == "SAME":
        out_h = (input_h + stride_h - 1) // stride_h
        out_w = (input_w + stride_w - 1) // stride_w
        pad_row = (out_h - 1) * stride_h + window_h - input_h
        pad_col = (out_w - 1) * stride_w + window_w - input_w
        pad_top = pad_row // 2
        pad_bottom = pad_row - pad_top
        pad_left = pad_col // 2
        pad_right = pad_col - pad_left

        pad_top_new = 0 if pad_top < 0 else pad_top
        pad_bottom_new = 0 if pad_bottom < 0 else pad_bottom
        pad_left_new = 0 if pad_left < 0 else pad_left
        pad_right_new = 0 if pad_right < 0 else pad_right

        cal_pads = (pad_top_new, pad_bottom_new, pad_left_new, pad_right_new)
    elif params.get("padding_mode") == "CALCULATED":
        pad_top, pad_bottom, pad_left, pad_right = params.get("pads")
        if params.get("ceil_mode"):
            out_h = (input_h - window_h + pad_top + pad_bottom + stride_h - 1) // stride_h + 1
            out_w = (input_w - window_w + pad_left + pad_right + stride_w - 1) // stride_w + 1
        else:
            out_h = (input_h - window_h + pad_top + pad_bottom) // stride_h + 1
            out_w = (input_w - window_w + pad_left + pad_right) // stride_w + 1
        pad_bottom = (out_h - 1) * stride_h + window_h - input_h - pad_top
        pad_right = (out_w - 1) * stride_w + window_w - input_w - pad_left

        pad_bottom_new = 0 if pad_bottom < 0 else pad_bottom
        pad_right_new = 0 if pad_right < 0 else pad_right

        cal_pads = (pad_top, pad_bottom_new, pad_left, pad_right_new)
    else:
        cal_pads = (0, 0, 0, 0)

    if cal_pads[0] >= window_h or cal_pads[1] >= window_h or cal_pads[2] >= window_w or cal_pads[3] >= window_w:
        error_manager_vector.raise_err_specific_reson(params.get("kernel_name"), "the pad value is valid")
    return {"cal_pads": cal_pads}


def _get_shape_size(shape):
    """
    get the total size of shape
    """
    total = 1
    for i in shape:
        total = total * i
    return total


def _get_index_num(shape, start, end, step, threshold):
    """
    get the index num
    """
    val = 1
    index = end - step
    for i in range(start, end, step):
        val = val * shape[i]
        if val >= threshold:
            index = i
            break
    return index, _get_shape_size(shape[:index + 1])


def _get_product_of_each_dim(shape, dims):
    """
    get product of each dim
    """
    product_list = [1] * len(shape)
    for i in range(dims):
        j = i + 1
        while j < dims:
            product_list[i] = product_list[i] * shape[j]
            j = j + 1
    return product_list


def _is_immediate(val):
    """
    check is immediate number
    """
    if isinstance(val, (tik.Expr, tik.Scalar)):
        return False
    return True


class Dilation2DBase:
    """
    use to store dilation2d base parameters
    """

    def __init__(self, input_params):
        """
        init shape and format information
        """
        self.input_params = input_params
        self.instance = tik.Tik(tik.Dprofile())
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.dtype = input_params.get("x").get("dtype").lower()
        self.dsize = common_util.get_data_size(self.dtype)
        self.block_size = 32 // self.dsize

        self.x_shape = input_params.get("x").get("shape")
        self.batch = self.x_shape[0]
        self.c1 = self.x_shape[1]
        self.h_in = self.x_shape[2]
        self.w_in = self.x_shape[3]
        self.c0 = self.x_shape[4]

        self.filter_shape = input_params.get("filter").get("shape")
        self.filter_h = self.filter_shape[2]
        self.filter_w = self.filter_shape[3]

        self.y_shape = input_params.get("y").get("shape")
        self.h_out = self.y_shape[2]
        self.w_out = self.y_shape[3]

        self.stride_h = input_params.get("stride_h")
        self.stride_w = input_params.get("stride_w")
        self.rate_h = input_params.get("rate_h")
        self.rate_w = input_params.get("rate_w")
        self.window_h = input_params.get("window_h")
        self.window_w = input_params.get("window_w")
        self.pad_top, self.pad_bottom, self.pad_left, self.pad_right = input_params.get("pads")

        self.offset_list = _get_product_of_each_dim(self.y_shape[:-1], len(self.y_shape[:-1]))
        self.repeat_max = self.instance.Scalar("uint32", name="repeat_max")
        self.repeat_max.set_as(255)

    def get_input_need_size(self, shape):
        """
        calculate the needed input size
        """
        n_factor, c1_factor, ho_factor, wo_factor = shape
        h = (ho_factor - 1) * self.stride_h + self.window_h
        w = (wo_factor - 1) * self.stride_w + self.window_w
        if ho_factor == self.h_out:
            ho_factor = max(self.h_in, h)
        else:
            ho_factor = h
        if wo_factor == self.w_out:
            wo_factor = max(self.w_in, w)
        else:
            wo_factor = w
        return n_factor * c1_factor * ho_factor * wo_factor * self.c0

    def get_expand_need_size(self, shape):
        """
        calculate the needed expand size
        """
        n_factor, c1_factor, ho_factor, wo_factor = shape
        size = n_factor * c1_factor * ho_factor * wo_factor * self.c0
        return 2 * size * self.filter_w

    def get_filter_need_size(self, ub_size):
        """
        calculate the needed filter size
        """
        filter_all = self.c1 * self.filter_h * self.filter_w * self.c0
        if ub_size // 3 > filter_all:
            return filter_all, True
        return self.filter_h * self.filter_w * self.c0, False

    def get_mid_ho_wo_range(self):
        """
        get the range without pad
        """
        h_start = 0
        h_end = 0
        w_start = 0
        w_end = 0
        for i in range(self.h_out):
            h_beg = i * self.stride_h - self.pad_top
            if h_beg >= 0:
                h_start = i
                break
        for j in range(self.h_out - 1, h_start - 1, -1):
            h_beg = j * self.stride_h - self.pad_top + self.window_h - 1
            if 0 <= h_beg < self.h_in:
                h_end = j
                break
        for i in range(self.w_out):
            w_beg = i * self.stride_w - self.pad_left
            if w_beg >= 0:
                w_start = i
                break
        for j in range(self.w_out - 1, w_start - 1, -1):
            w_beg = j * self.stride_w - self.pad_left + self.window_w - 1
            if 0 <= w_beg < self.w_in:
                w_end = j
                break
        if h_end >= h_start:
            h_end = h_end + 1
        else:
            h_end = h_start
        if w_end >= w_start:
            w_end = w_end + 1
        else:
            w_end = w_start
        return_list = [h_start, h_end, w_start, w_end]
        return return_list

    def get_offset(self, index, ub_index):
        """
        get the offset position of input
        """
        index_list = []
        for j in range(len(self.offset_list)):
            mod = index
            for k in range(j):
                mod %= self.offset_list[k]
            mod = mod // self.offset_list[j]
            if j <= ub_index:
                scalar = self.instance.Scalar("uint32", name="offset_scalar")
                scalar.set_as(mod)
                index_list.append(scalar)
        return index_list

    def vector_dup(self, start_index, ub_buf, size, val):
        """
        vector_dup function, set ub_buf to 0
        """
        one_cnt = 128
        repeat = size // one_cnt
        remainder = size % one_cnt
        loop_repeat = repeat // 255
        loop_remainder = repeat % 255
        if _is_immediate(size):
            if loop_repeat > 0:
                with self.instance.for_range(0, loop_repeat) as l_i:
                    offset_repeat = l_i * one_cnt * 255
                    self.instance.vec_dup(one_cnt, ub_buf[start_index + offset_repeat], val, 255, 8)
            if loop_remainder > 0:
                offset_remainder = loop_repeat * one_cnt * 255
                self.instance.vec_dup(one_cnt, ub_buf[start_index + offset_remainder], val, loop_remainder, 8)
            if remainder > 0:
                offset = loop_repeat * 255 * one_cnt + loop_remainder * one_cnt
                self.instance.vec_dup(remainder, ub_buf[start_index + offset], val, 1, 8)
        else:
            loop_remainder_s = self.instance.Scalar("uint32", name="dup_loop_remainder")
            mask = self.instance.Scalar("uint32", name="dup_mask")
            loop_remainder_s.set_as(loop_remainder)
            mask.set_as(remainder)
            with self.instance.for_range(0, loop_repeat) as l_i:
                offset_repeat = l_i * one_cnt * 255
                self.instance.vec_dup(one_cnt, ub_buf[start_index + offset_repeat], val, self.repeat_max, 8)
            with self.instance.if_scope(loop_remainder > 0):
                offset_remainder = loop_repeat * one_cnt * 255
                self.instance.vec_dup(one_cnt, ub_buf[start_index + offset_remainder], val, loop_remainder_s, 8)
            with self.instance.if_scope(remainder > 0):
                offset = loop_repeat * 255 * one_cnt + loop_remainder * one_cnt
                self.instance.vec_dup(mask, ub_buf[start_index + offset], val, 1, 8)

    def vector_add(self, start_list, ub_list, stride_list, mask_info):
        """
        vector_add function
        """
        repeat, mask_all = mask_info
        dst_start, src0_start, src1_start = start_list
        dst_rep_stride, src0_rep_stride, src1_rep_stride, _, _, _ = stride_list
        src_stride = 1 if self.filter_w == 1 else self.rate_w
        mask_max = min(128 // self.c0, 255 // src_stride) * self.c0
        repeat_loop = repeat // 255
        repeat_tail = repeat % 255
        mask_loop = mask_all // mask_max
        mask_tail = mask_all % mask_max
        if _is_immediate(repeat):
            if repeat_loop > 0:
                with self.instance.for_range(0, repeat_loop) as l_i:
                    l_dst_start = dst_start + l_i * 255 * dst_rep_stride * self.c0
                    l_src0_start = src0_start + l_i * 255 * src0_rep_stride * self.c0
                    l_src1_start = src1_start + l_i * 255 * src1_rep_stride * self.c0
                    self.mask_add([255, mask_loop, mask_tail, mask_max], ub_list,
                                  [l_dst_start, l_src0_start, l_src1_start], stride_list)
            if repeat_tail > 0:
                t_dst_start = dst_start + repeat_loop * 255 * dst_rep_stride * self.c0
                t_src0_start = src0_start + repeat_loop * 255 * src0_rep_stride * self.c0
                t_src1_start = src1_start + repeat_loop * 255 * src1_rep_stride * self.c0
                self.mask_add([repeat_tail, mask_loop, mask_tail, mask_max], ub_list,
                              [t_dst_start, t_src0_start, t_src1_start], stride_list)
        else:
            repeat_tail_s = self.instance.Scalar("uint32", name="add_repeat_tail")
            repeat_tail_s.set_as(repeat_tail)
            with self.instance.for_range(0, repeat_loop) as l_i:
                l_dst_start = dst_start + l_i * 255 * dst_rep_stride * self.c0
                l_src0_start = src0_start + l_i * 255 * src0_rep_stride * self.c0
                l_src1_start = src1_start + l_i * 255 * src1_rep_stride * self.c0
                self.mask_add([self.repeat_max, mask_loop, mask_tail, mask_max], ub_list,
                              [l_dst_start, l_src0_start, l_src1_start], stride_list)
            with self.instance.if_scope(repeat_tail > 0):
                t_dst_start = dst_start + repeat_loop * 255 * dst_rep_stride * self.c0
                t_src0_start = src0_start + repeat_loop * 255 * src0_rep_stride * self.c0
                t_src1_start = src1_start + repeat_loop * 255 * src1_rep_stride * self.c0
                self.mask_add([repeat_tail_s, mask_loop, mask_tail, mask_max], ub_list,
                              [t_dst_start, t_src0_start, t_src1_start], stride_list)

    def mask_add(self, mask_list, ub_list, start_list, stride_list):
        """
        handle the scenario where block stride > 128
        """
        dst_ub, src0_ub, src1_ub = ub_list
        repeat, mask_loop, mask_tail, mask = mask_list
        dst_start, src0_start, src1_start = start_list
        dst_rep_stride, src0_rep_stride, src1_rep_stride, dst_blk_stride, src0_blk_stride, src1_blk_stride = stride_list
        if _is_immediate(mask_loop):
            if mask_loop > 0:
                with self.instance.for_range(0, mask_loop) as l_i:
                    dst_index = dst_start + l_i * mask * dst_blk_stride
                    src0_index = src0_start + l_i * mask * src0_blk_stride
                    src1_index = src1_start + l_i * mask * src1_blk_stride
                    self.instance.vadd(mask, dst_ub[dst_index], src0_ub[src0_index], src1_ub[src1_index], repeat,
                                       dst_blk_stride, src0_blk_stride, src1_blk_stride, dst_rep_stride,
                                       src0_rep_stride, src1_rep_stride)
            if mask_tail > 0:
                dst_index = dst_start + mask_loop * mask * dst_blk_stride
                src0_index = src0_start + mask_loop * mask * src0_blk_stride
                src1_index = src1_start + mask_loop * mask * src1_blk_stride
                self.instance.vadd(mask_tail, dst_ub[dst_index], src0_ub[src0_index], src1_ub[src1_index], repeat,
                                   dst_blk_stride, src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride,
                                   src1_rep_stride)
        else:
            with self.instance.for_range(0, mask_loop) as l_i:
                dst_index = dst_start + l_i * mask * dst_blk_stride
                src0_index = src0_start + l_i * mask * src0_blk_stride
                src1_index = src1_start + l_i * mask * src1_blk_stride
                self.instance.vadd(mask, dst_ub[dst_index], src0_ub[src0_index], src1_ub[src1_index], repeat,
                                   dst_blk_stride, src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride,
                                   src1_rep_stride)
            with self.instance.if_scope(mask_tail > 0):
                dst_index = dst_start + mask_loop * mask * dst_blk_stride
                src0_index = src0_start + mask_loop * mask * src0_blk_stride
                src1_index = src1_start + mask_loop * mask * src1_blk_stride
                self.instance.vadd(mask_tail, dst_ub[dst_index], src0_ub[src0_index], src1_ub[src1_index], repeat,
                                   dst_blk_stride, src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride,
                                   src1_rep_stride)

    def vector_max(self, mask_info, ub_list, start_list, stride_list):
        """
        vector_max function
        """
        size, mask_num = mask_info
        one_cnt = mask_num * self.c0
        repeat = size // one_cnt
        remainder = size % one_cnt
        loop_repeat = repeat // 255
        loop_remainder = repeat % 255
        dst_ub, src0_ub, src1_ub = ub_list
        dst_start, src0_start, src1_start = start_list
        dst_rep_stride, src0_rep_stride, src1_rep_stride, dst_blk_stride, src0_blk_stride, src1_blk_stride = stride_list
        if _is_immediate(size):
            if loop_repeat > 0:
                with self.instance.for_range(0, loop_repeat) as l_i:
                    dst_index = dst_start + l_i * 255 * dst_rep_stride * self.c0
                    src0_index = src0_start + l_i * 255 * src0_rep_stride * self.c0
                    src1_index = src1_start + l_i * 255 * src1_rep_stride * self.c0
                    self.instance.vmax(one_cnt, dst_ub[dst_index], src0_ub[src0_index], src1_ub[src1_index], 255,
                                       dst_blk_stride, src0_blk_stride, src1_blk_stride, dst_rep_stride,
                                       src0_rep_stride, src1_rep_stride)
            if loop_remainder > 0:
                dst_index = dst_start + loop_repeat * 255 * dst_rep_stride * self.c0
                src0_index = src0_start + loop_repeat * 255 * src0_rep_stride * self.c0
                src1_index = src1_start + loop_repeat * 255 * src1_rep_stride * self.c0
                self.instance.vmax(one_cnt, dst_ub[dst_index], src0_ub[src0_index], src1_ub[src1_index], loop_remainder,
                                   dst_blk_stride, src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride,
                                   src1_rep_stride)
            if remainder > 0:
                dst_index = dst_start + repeat * dst_rep_stride * self.c0
                src0_index = src0_start + repeat * src0_rep_stride * self.c0
                src1_index = src1_start + repeat * src1_rep_stride * self.c0
                self.instance.vmax(remainder, dst_ub[dst_index], src0_ub[src0_index], src1_ub[src1_index], 1,
                                   dst_blk_stride, src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride,
                                   src1_rep_stride)
        else:
            loop_remainder_s = self.instance.Scalar("uint32", name="max_loop_remainder")
            mask = self.instance.Scalar("uint32", name="max_mask")
            loop_remainder_s.set_as(loop_remainder)
            mask.set_as(remainder)
            with self.instance.for_range(0, loop_repeat) as l_i:
                dst_index = dst_start + l_i * 255 * dst_rep_stride * self.c0
                src0_index = src0_start + l_i * 255 * src0_rep_stride * self.c0
                src1_index = src1_start + l_i * 255 * src1_rep_stride * self.c0
                self.instance.vmax(one_cnt, dst_ub[dst_index], src0_ub[src0_index], src1_ub[src1_index],
                                   self.repeat_max, dst_blk_stride, src0_blk_stride, src1_blk_stride, dst_rep_stride,
                                   src0_rep_stride, src1_rep_stride)
            with self.instance.if_scope(loop_remainder > 0):
                dst_index = dst_start + loop_repeat * 255 * dst_rep_stride * self.c0
                src0_index = src0_start + loop_repeat * 255 * src0_rep_stride * self.c0
                src1_index = src1_start + loop_repeat * 255 * src1_rep_stride * self.c0
                self.instance.vmax(one_cnt, dst_ub[dst_index], src0_ub[src0_index], src1_ub[src1_index],
                                   loop_remainder_s, dst_blk_stride, src0_blk_stride, src1_blk_stride, dst_rep_stride,
                                   src0_rep_stride, src1_rep_stride)
            with self.instance.if_scope(remainder > 0):
                dst_index = dst_start + repeat * dst_rep_stride * self.c0
                src0_index = src0_start + repeat * src0_rep_stride * self.c0
                src1_index = src1_start + repeat * src1_rep_stride * self.c0
                self.instance.vmax(mask, dst_ub[dst_index], src0_ub[src0_index], src1_ub[src1_index], 1, dst_blk_stride,
                                   src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride, src1_rep_stride)


class Dilation2D(Dilation2DBase):
    """
    use to store dilation2d compute parameters
    """

    def __init__(self, input_params):
        """
        init gm and offset information
        """
        super(Dilation2D, self).__init__(input_params)

        x_size = _get_shape_size(self.x_shape)
        filter_size = _get_shape_size(self.filter_shape)
        y_size = _get_shape_size(self.y_shape)
        self.x_gm = self.instance.Tensor(self.dtype, (x_size,), name="x_gm", scope=tik.scope_gm)
        self.filter_gm = self.instance.Tensor(self.dtype, (filter_size,), name="filter_gm", scope=tik.scope_gm)
        self.y_gm = self.instance.Tensor(self.dtype, (y_size,), name="y_gm", scope=tik.scope_gm)
        self.x_offset_list = _get_product_of_each_dim(self.x_shape, len(self.x_shape))
        self.filter_offset_list = _get_product_of_each_dim(self.filter_shape, len(self.filter_shape))
        self.y_offset_list = _get_product_of_each_dim(self.y_shape, len(self.y_shape))

        [self.ho_start, self.ho_end, self.wo_start, self.wo_end] = self.get_mid_ho_wo_range()
        self.tiling_params = {}
        pad_flag = self.pad_left != 0 or self.pad_right != 0
        move_flag = False
        if pad_flag and self.w_in >= 16:
            move_flag = True
        self.move_by_row = move_flag
        self.pad_w = self.pad_left + self.pad_right + self.w_in

    def do_tiling(self, tiling_shape, ub_size):
        """
        get tiling information
        """
        self.tiling_params["tiling_shape"] = tiling_shape
        block_index, block_len = _get_index_num(tiling_shape, 0, len(tiling_shape) - 2, 1, self.core_num)
        block_num = self.core_num
        if block_len < block_num:
            block_num = block_len
        self.tiling_params["block_num"] = block_num
        self.tiling_params["block_index"] = block_index
        self.tiling_params["block_cycle"] = block_len // block_num
        self.tiling_params["block_tail"] = block_len % block_num
        self.tiling_params["block_element"] = _get_shape_size(tiling_shape[block_index + 1:len(tiling_shape) - 1])

        mod = 1 if block_len % block_num > 0 else 0
        ub_tiling_shape = [block_len // block_num + mod]
        ub_tiling_shape.extend(tiling_shape[block_index + 1:len(tiling_shape) - 1])
        flag = False
        is_all = False
        ub_factor = 1
        ub_index = len(tiling_shape) - 2
        ub_size_info = {
            "input": self.window_h * self.window_w * self.c0,
            "filter": self.filter_h * self.filter_w * self.c0,
            "expand": self.filter_h * self.filter_w * self.c0
        }
        for i, elem in enumerate(ub_tiling_shape):
            [find, t_factor, size_info, is_all] = self.try_tiling([i, elem], block_index, tiling_shape, ub_size)
            if find:
                flag = True
                ub_index = block_index + i
                ub_factor = t_factor
                ub_size_info = size_info
                break
        self.tiling_params["is_filter_all"] = is_all
        self.tiling_params["ub_index"] = ub_index
        self.tiling_params["ub_factor"] = ub_factor
        self.tiling_params["ub_size_info"] = ub_size_info
        elem_shape = tiling_shape[ub_index + 1:len(tiling_shape) - 1]
        self.tiling_params["ub_num"] = ub_factor * _get_shape_size(elem_shape)
        return flag

    def try_tiling(self, index_info, block_index, tiling_shape, ub_size):
        """
        try to do tiling
        """
        index, element = index_info
        find = False
        t_factor = 1
        is_all = False
        input = self.window_h * self.window_w * self.c0
        filter = self.filter_h * self.filter_w * self.c0
        expand = self.filter_h * self.filter_w * self.c0
        for t_factor in range(element, 0, -1):
            tmp_shape = [1] * (block_index + index)
            tmp_shape.extend([t_factor])
            tmp_shape.extend(tiling_shape[block_index + index + 1:len(tiling_shape) - 1])
            expand = self.get_expand_need_size(tmp_shape)
            filter, is_all = self.get_filter_need_size(ub_size)
            input = self.get_input_need_size(tmp_shape)
            size = expand + filter + input
            if size <= ub_size:
                find = True
                break
        info = {"input": input, "filter": filter, "expand": expand}
        return_list = [find, t_factor, info, is_all]
        return return_list

    def dilation_compute(self):
        """
        dilation2d compute function
        """
        ub_size = (self.ub_size - 4 * 1024) // self.dsize
        flag = self.do_tiling(self.y_shape, ub_size // 2)
        self.tiling_params["thread_num"] = 1
        if self.tiling_params["ub_index"] == 3:
            flag = False

        if flag:
            size = self.tiling_params.get("block_cycle") * self.tiling_params.get("block_element")
            db_num = size // self.tiling_params["ub_num"]
            self.tiling_params["thread_num"] = 2 if db_num >= 2 else 1

        if not flag or self.tiling_params["thread_num"] == 1:
            flag_1 = self.do_tiling(self.y_shape, ub_size)
            if flag_1:
                self.tiling_params["thread_num"] = 1
            else:
                error_manager_vector.raise_err_specific_reson("dilation2d",
                                                              "can not find tiling, filter or rates is too big")

        block_num = self.tiling_params["block_num"]
        block_cycle = self.tiling_params["block_cycle"]
        block_element = self.tiling_params["block_element"]
        block_tail = self.tiling_params["block_tail"]
        ub_num = self.tiling_params["ub_num"]
        thread_num = self.tiling_params["thread_num"]

        with self.instance.for_range(0, block_num, block_num=block_num) as block_id:
            each_cycle = self.instance.Scalar("uint32", name="each_cycle")
            offset = self.instance.Scalar("uint32", name="offset")
            each_cycle.set_as(block_cycle * block_element)
            if block_tail > 0:
                with self.instance.if_scope(block_id < block_tail):
                    each_cycle.set_as((block_cycle + 1) * block_element)
            offset.set_as(block_id * each_cycle)
            if block_tail > 0:
                with self.instance.if_scope(block_id >= block_tail):
                    offset.set_as((block_id * block_cycle + block_tail) * block_element)

            ub_loop = each_cycle // ub_num
            ub_tail = each_cycle % ub_num
            filter_ub = self.prepare_filter()
            if self.tiling_params["ub_index"] == 3:
                loop = each_cycle // self.w_out
                ub_loop = self.w_out // ub_num
                ub_tail = self.w_out % ub_num
                with self.instance.for_range(0, loop, thread_num=thread_num) as loop_id:
                    with self.instance.for_range(0, ub_loop) as u_id:
                        self.do_compute(ub_num, offset + loop_id * self.w_out + u_id * ub_num, filter_ub)
                    with self.instance.if_scope(ub_tail > 0):
                        self.do_compute(ub_tail, offset + loop_id * self.w_out + ub_loop * ub_num, filter_ub)
            else:
                with self.instance.for_range(0, ub_loop, thread_num=thread_num) as loop_id:
                    self.do_compute(ub_num, offset + loop_id * ub_num, filter_ub)
                with self.instance.if_scope(ub_tail > 0):
                    self.do_compute(ub_tail, offset + ub_loop * ub_num, filter_ub)

    def prepare_filter(self):
        """
        prepare the filter ub tensor
        """
        filter_size = _get_shape_size(self.filter_shape)
        if self.tiling_params["is_filter_all"]:
            filter_ub = self.instance.Tensor(self.dtype, (filter_size,),
                                             name="filter_ub",
                                             scope=tbe_platform.scope_ubuf)
            self.instance.data_move(filter_ub, self.filter_gm, 0, 1, filter_size // self.block_size, 0, 0)
            return filter_ub
        return None

    def do_compute(self, num, offset, filter_ub):
        """
        describes the calculation of add and max
        """
        ub_index = self.tiling_params["ub_index"]
        if filter_ub is None:
            filter_size = self.tiling_params["ub_size_info"].get("filter")
            filter_ub = self.instance.Tensor(self.dtype, (filter_size,),
                                             name="filter_ub",
                                             scope=tbe_platform.scope_ubuf)
        x_size = self.tiling_params["ub_size_info"].get("input")
        x_ub = self.instance.Tensor(self.dtype, (x_size,), name="x_ub", scope=tbe_platform.scope_ubuf)
        expand_size = self.tiling_params["ub_size_info"].get("expand")
        expand_ub = self.instance.Tensor(self.dtype, (expand_size,), name="expand_ub", scope=tbe_platform.scope_ubuf)
        ub_list = [expand_ub, x_ub, filter_ub]
        out_offset = offset * self.c0
        offset_list = self.get_offset(offset, ub_index)

        if ub_index == 0:
            n_num = num // self.c1 // self.h_out // self.w_out
            self.cut_n(ub_list, offset_list, n_num, out_offset)
        elif ub_index == 1:
            c1_num = num // self.h_out // self.w_out
            self.cut_c1(ub_list, offset_list, c1_num, out_offset)
        elif ub_index == 2:
            h_num = num // self.w_out
            with self.instance.if_scope(h_num + offset_list[2] > self.h_out):
                h_num_1 = self.h_out - offset_list[2]
                h_num_2 = h_num - h_num_1
                self.cut_h(ub_list, offset_list, h_num_1, out_offset)

                offset_list_2 = self.get_offset(offset + h_num_1 * self.w_out, ub_index)
                out_offset_2 = (offset + h_num_1 * self.w_out) * self.c0
                self.cut_h(ub_list, offset_list_2, h_num_2, out_offset_2)
            with self.instance.else_scope():
                self.cut_h(ub_list, offset_list, h_num, out_offset)
        else:
            self.cut_w(ub_list, offset_list, num, out_offset)

    def move_data(self, x_ub, ub_offset, x_index, n_burst):
        """
        move data from gm to ub
        """
        if self.move_by_row:
            with self.instance.if_scope(tik.any(n_burst > 4095, self.w_in > 65535)):
                with self.instance.for_range(0, n_burst) as h_i:
                    self.instance.data_move(x_ub[ub_offset + h_i * self.pad_w * self.c0],
                                            self.x_gm[x_index + h_i * self.w_in * self.c0], 0, 1, self.w_in, 0, 0)
            with self.instance.else_scope():
                self.instance.data_move(x_ub[ub_offset], self.x_gm[x_index], 0, n_burst, self.w_in, 0,
                                        self.pad_w - self.w_in)
        else:
            ub_index = ub_offset - self.pad_left * self.c0
            self.instance.data_move(x_ub[ub_index], self.x_gm[x_index], 0, 1, n_burst * self.w_in, 0, 0)

    def expand_row(self, ub_list, h_in, start_list):
        """
        expand data by filter row, reduce row and reduce col
        """
        expand_ub, _, _ = ub_list
        expand_start, x_start, filter_start = start_list
        x_stride = 1 if self.filter_w == 1 else self.rate_w
        h_size = self.w_out * self.filter_w * self.c0
        w_size = self.filter_w * self.c0
        rep_stride_list = [self.filter_w, self.stride_w, 0]
        blk_stride_list = [1, x_stride, 1]

        with self.instance.if_scope(tik.all(h_in >= 0, h_in < self.h_in)):
            if self.move_by_row:
                self.vector_add(start_list, ub_list, rep_stride_list + blk_stride_list,
                                [self.w_out, self.filter_w * self.c0])
                if self.wo_start > 0:
                    for wo_i in range(0, self.wo_start):
                        fw_i = (self.pad_left - wo_i * self.stride_w + self.rate_w - 1) // self.rate_w
                        l_expand = expand_start + wo_i * w_size
                        self.vector_dup(l_expand, expand_ub, fw_i * self.c0, Constant.MIN_VAL)
                if self.w_out - self.wo_end > 0:
                    for wo_i in range(self.wo_end, self.w_out):
                        w_beg = wo_i * self.stride_w - self.pad_left
                        end = (self.w_in - w_beg + self.rate_w - 1) // self.rate_w
                        num = self.filter_w - end
                        r_expand = expand_start + wo_i * w_size + end * self.c0
                        self.vector_dup(r_expand, expand_ub, num * self.c0, Constant.MIN_VAL)
            else:
                if self.wo_start > 0:
                    left_size = self.wo_start * w_size
                    self.vector_dup(expand_start, expand_ub, left_size, Constant.MIN_VAL)
                if self.w_out - self.wo_end > 0:
                    right_size = (self.w_out - self.wo_end) * w_size
                    start_index = expand_start + self.wo_end * w_size
                    self.vector_dup(start_index, expand_ub, right_size, Constant.MIN_VAL)

                if self.wo_start > 0:
                    for wo_i in range(0, self.wo_start):
                        w_beg = wo_i * self.stride_w - self.pad_left
                        fw_i = (self.pad_left - wo_i * self.stride_w + self.rate_w - 1) // self.rate_w
                        num = self.filter_w - fw_i
                        l_x = x_start + (w_beg + fw_i * self.rate_w) * self.c0
                        l_expand = expand_start + wo_i * w_size + fw_i * self.c0
                        l_filter = filter_start + fw_i * self.c0
                        self.vector_add([l_expand, l_x, l_filter], ub_list, rep_stride_list + blk_stride_list,
                                        [1, num * self.c0])
                if self.wo_end - self.wo_start > 0:
                    mid_x = x_start + (self.wo_start * self.stride_w - self.pad_left) * self.c0
                    mid_expand = expand_start + self.wo_start * w_size
                    mid_filter = filter_start
                    self.vector_add([mid_expand, mid_x, mid_filter], ub_list, rep_stride_list + blk_stride_list,
                                    [self.wo_end - self.wo_start, self.filter_w * self.c0])
                if self.w_out - self.wo_end > 0:
                    for wo_i in range(self.wo_end, self.w_out):
                        w_beg = wo_i * self.stride_w - self.pad_left
                        end = (self.w_in - w_beg + self.rate_w - 1) // self.rate_w
                        r_x = x_start + w_beg * self.c0
                        r_expand = expand_start + wo_i * w_size
                        r_filter = filter_start
                        self.vector_add([r_expand, r_x, r_filter], ub_list, rep_stride_list + blk_stride_list,
                                        [1, end * self.c0])
        with self.instance.else_scope():
            self.vector_dup(expand_start, expand_ub, h_size, Constant.MIN_VAL)

    def reduce_h(self, fh_index, fh_size, expand_ub):
        """
        maximize data by row, MAX(row1,row2)
        """
        if self.filter_h > 1:
            rep_stride_list = [8, 8, 8]
            blk_stride_list = [1, 1, 1]
            ub_list = [expand_ub, expand_ub, expand_ub]
            with self.instance.if_scope(fh_index > 0):
                self.vector_max([fh_size, 8], ub_list, [(fh_index % 2) * fh_size, 0, fh_size],
                                rep_stride_list + blk_stride_list)

    def reduce_w(self, fh_size, fw_size, expand_ub):
        """
        maximize data by col, MAX(col1,col2)
        """
        start = ((self.filter_h - 1) % 2) * fh_size
        reduce_index = start
        if self.filter_w > 1:
            num = min(8, 255 // self.filter_w)
            rep_stride_list = [num, num * self.filter_w, num * self.filter_w]
            blk_stride_list = [1, self.filter_w, self.filter_w]
            ub_list = [expand_ub, expand_ub, expand_ub]
            max_start = (self.filter_h % 2) * fh_size
            index_list = [max_start, max_start + fw_size, start]
            self.vector_max([fw_size, num], ub_list, [index_list[0], start, start + self.c0],
                            rep_stride_list + blk_stride_list)
            rep_stride_list_2 = [num, num, num * self.filter_w]
            blk_stride_list_2 = [1, 1, self.filter_w]
            with self.instance.for_range(2, self.filter_w) as fw:
                with self.instance.if_scope(fw % 2 == 0):
                    self.vector_max([fw_size, num], ub_list, [index_list[1], index_list[0], start + self.c0 * fw],
                                    rep_stride_list_2 + blk_stride_list_2)
                with self.instance.else_scope():
                    self.vector_max([fw_size, num], ub_list, [index_list[0], index_list[1], start + self.c0 * fw],
                                    rep_stride_list_2 + blk_stride_list_2)
            reduce_index = index_list[self.filter_w % 2]
        return reduce_index

    def cut_n(self, ub_list, offset_list, n_num, out_offset):
        """
        tiling cut n scenario
        """
        expand_ub, x_ub, filter_ub = ub_list
        n_offset = offset_list[0]
        fh_size = n_num * self.c1 * self.h_out * self.w_out * self.filter_w * self.c0
        fw_size = n_num * self.c1 * self.h_out * self.w_out * self.c0
        n_size = self.c1 * self.h_out * self.w_out * self.filter_w * self.c0
        c1_size = self.h_out * self.w_out * self.filter_w * self.c0
        h_size = self.w_out * self.filter_w * self.c0

        ub_offset = self.pad_left * self.c0
        x_index = n_offset * self.x_offset_list[0]
        w_in = self.pad_w if self.move_by_row else self.w_in
        self.move_data(x_ub, ub_offset, x_index, n_num * self.c1 * self.h_in)

        with self.instance.for_range(0, self.filter_h) as fh_i:
            with self.instance.for_range(0, n_num * self.c1 * self.h_out) as id:
                n_i = id // (self.c1 * self.h_out)
                c1_i = id % (self.c1 * self.h_out) // self.h_out
                ho_i = id % (self.c1 * self.h_out) % self.h_out
                h_in = ho_i * self.stride_h - self.pad_top + fh_i * self.rate_h
                expand_start = (fh_i % 2) * fh_size + n_i * n_size + c1_i * c1_size + ho_i * h_size
                x_start = n_i * self.c1 * self.h_in * w_in * self.c0 + \
                          c1_i * self.h_in * w_in * self.c0 + \
                          h_in * w_in * self.c0
                if not self.tiling_params["is_filter_all"]:
                    gm_index = c1_i * self.filter_offset_list[1] + fh_i * self.filter_offset_list[2]
                    self.instance.data_move(filter_ub, self.filter_gm[gm_index], 0, 1,
                                            self.filter_offset_list[2] // self.block_size, 0, 0)
                    filter_start = 0
                else:
                    filter_start = c1_i * self.filter_offset_list[1] + fh_i * self.filter_offset_list[2]
                self.expand_row(ub_list, h_in, [expand_start, x_start, filter_start])
            self.reduce_h(fh_i, fh_size, expand_ub)
        reduce_index = self.reduce_w(fh_size, fw_size, expand_ub)
        self.instance.data_move(self.y_gm[out_offset], expand_ub[reduce_index], 0, 1, fw_size // self.block_size, 0, 0)

    def cut_c1(self, ub_list, offset_list, c1_num, out_offset):
        """
        tiling cut c1 scenario
        """
        expand_ub, x_ub, filter_ub = ub_list
        n_offset, c1_offset = offset_list
        fh_size = c1_num * self.h_out * self.w_out * self.filter_w * self.c0
        fw_size = c1_num * self.h_out * self.w_out * self.c0
        c1_size = self.h_out * self.w_out * self.filter_w * self.c0
        h_size = self.w_out * self.filter_w * self.c0

        ub_offset = self.pad_left * self.c0
        x_index = n_offset * self.x_offset_list[0] + c1_offset * self.x_offset_list[1]
        w_in = self.pad_w if self.move_by_row else self.w_in
        self.move_data(x_ub, ub_offset, x_index, c1_num * self.h_in)

        with self.instance.for_range(0, self.filter_h) as fh_i:
            with self.instance.for_range(0, c1_num * self.h_out) as id:
                c1_i = id // self.h_out
                ho_i = id % self.h_out
                h_in = ho_i * self.stride_h - self.pad_top + fh_i * self.rate_h
                expand_start = (fh_i % 2) * fh_size + c1_i * c1_size + ho_i * h_size
                x_start = c1_i * self.h_in * w_in * self.c0 + \
                          h_in * w_in * self.c0
                if not self.tiling_params["is_filter_all"]:
                    gm_index = ((c1_offset + c1_i) % self.c1) * self.filter_offset_list[1] + fh_i * \
                               self.filter_offset_list[2]
                    self.instance.data_move(filter_ub, self.filter_gm[gm_index], 0, 1,
                                            self.filter_offset_list[2] // self.block_size, 0, 0)
                    filter_start = 0
                else:
                    filter_start = ((c1_offset + c1_i) % self.c1) * self.filter_offset_list[1] + fh_i * \
                                   self.filter_offset_list[2]
                self.expand_row(ub_list, h_in, [expand_start, x_start, filter_start])
            self.reduce_h(fh_i, fh_size, expand_ub)
        reduce_index = self.reduce_w(fh_size, fw_size, expand_ub)

        self.instance.data_move(self.y_gm[out_offset], expand_ub[reduce_index], 0, 1, fw_size // self.block_size, 0, 0)

    def cut_h(self, ub_list, offset_list, h_num, out_offset):
        """
        tiling cut h scenario
        """
        expand_ub, x_ub, filter_ub = ub_list
        n_offset, c1_offset, h_offset = offset_list
        fh_size = h_num * self.w_out * self.filter_w * self.c0
        fw_size = h_num * self.w_out * self.c0
        h_size = self.w_out * self.filter_w * self.c0
        h_beg = self.instance.Scalar("int32", name="h_beg")
        h_beg.set_as(h_offset * self.stride_h - self.pad_top)
        x_h_offset = self.instance.Scalar("int32", name="x_h_offset")
        x_h_offset.set_as(0)
        with self.instance.if_scope(h_beg < 0):
            h_beg.set_as(0)
            x_h_offset.set_as(self.pad_top - h_offset * self.stride_h)
        h_all = (h_num - 1) * self.stride_h + self.window_h
        h_len = self.instance.Scalar("int32", name="h_len")
        h_len.set_as(h_all - x_h_offset)
        h_max = self.h_in - h_beg
        with self.instance.if_scope(h_len > h_max):
            h_len.set_as(h_max)

        w_in = self.pad_w if self.move_by_row else self.w_in
        x_index = n_offset * self.x_offset_list[0] + c1_offset * self.x_offset_list[1] + h_beg * self.x_offset_list[2]
        ub_offset = x_h_offset * w_in * self.c0 + self.pad_left * self.c0
        self.move_data(x_ub, ub_offset, x_index, h_len)
        if not self.tiling_params["is_filter_all"]:
            self.instance.data_move(filter_ub, self.filter_gm[c1_offset * self.filter_offset_list[1]], 0, 1,
                                    self.filter_offset_list[1] // self.block_size, 0, 0)

        with self.instance.for_range(0, self.filter_h) as fh_i:
            with self.instance.for_range(0, h_num) as h_i:
                ho_i = h_offset + h_i
                h_in = ho_i * self.stride_h - self.pad_top + fh_i * self.rate_h
                expand_start = (fh_i % 2) * fh_size + h_i * h_size
                x_start = (h_i * self.stride_h + fh_i * self.rate_h) * w_in * self.c0
                if self.tiling_params["is_filter_all"]:
                    filter_start = c1_offset * self.filter_offset_list[1] + fh_i * self.filter_offset_list[2]
                else:
                    filter_start = fh_i * self.filter_offset_list[2]
                self.expand_row(ub_list, h_in, [expand_start, x_start, filter_start])
            self.reduce_h(fh_i, fh_size, expand_ub)
        reduce_index = self.reduce_w(fh_size, fw_size, expand_ub)

        self.instance.data_move(self.y_gm[out_offset], expand_ub[reduce_index], 0, 1, fw_size // self.block_size, 0, 0)

    def cut_w(self, ub_list, offset_list, w_num, out_offset):
        """
        tiling cut w scenario
        """
        expand_ub, x_ub, filter_ub = ub_list
        n_offset, c1_offset, h_offset, w_offset = offset_list
        fh_size = w_num * self.filter_w * self.c0
        fw_size = w_num * self.c0
        w_size = self.filter_w * self.c0
        x_stride = 1 if self.filter_w == 1 else self.rate_w
        rep_stride_list = [self.filter_w, self.stride_w, 0]
        blk_stride_list = [1, x_stride, 1]
        h_beg = self.instance.Scalar("int32", name="h_beg")
        h_beg.set_as(h_offset * self.stride_h - self.pad_top)
        x_h_offset = self.instance.Scalar("int32", name="x_h_offset")
        x_h_offset.set_as(0)
        w_beg = self.instance.Scalar("int32", name="w_beg")
        w_beg.set_as(w_offset * self.stride_w - self.pad_left)
        x_w_offset = self.instance.Scalar("int32", name="x_w_offset")
        x_w_offset.set_as(0)

        with self.instance.if_scope(h_beg < 0):
            h_beg.set_as(0)
            x_h_offset.set_as(self.pad_top - h_offset * self.stride_h)
        with self.instance.if_scope(w_beg < 0):
            w_beg.set_as(0)
            x_w_offset.set_as(self.pad_left - w_offset * self.stride_w)

        h_all = self.window_h
        h_len = self.instance.Scalar("int32", name="h_len")
        h_len.set_as(h_all - x_h_offset)
        h_max = self.h_in - h_beg
        with self.instance.if_scope(h_len > h_max):
            h_len.set_as(h_max)

        w_all = (w_num - 1) * self.stride_w + self.window_w
        w_max = self.w_in - w_beg
        w_len = self.instance.Scalar("int32", name="w_len")
        w_len.set_as(w_all - x_w_offset)
        with self.instance.if_scope(w_len > w_max):
            w_len.set_as(w_max)

        x_index = n_offset * self.x_offset_list[0] + c1_offset * self.x_offset_list[1] + \
                  h_beg * self.x_offset_list[2] + w_beg * self.x_offset_list[3]
        x_ub_index = (x_h_offset * w_all + x_w_offset) * self.c0
        if self.w_in >= 65535:
            with self.instance.for_range(0, h_len) as h_i:
                self.instance.data_move(x_ub[x_ub_index + h_i * w_all * self.c0],
                                        self.x_gm[x_index + h_i * self.w_in * self.c0], 0, 1, w_len, 0, 0)
        else:
            self.instance.data_move(x_ub[x_ub_index], self.x_gm[x_index], 0, h_len, w_len, self.w_in - w_len,
                                    w_all - w_len)
        if not self.tiling_params["is_filter_all"]:
            self.instance.data_move(filter_ub, self.filter_gm[c1_offset * self.filter_offset_list[1]], 0, 1,
                                    self.filter_offset_list[1] // self.block_size, 0, 0)

        left_start = self.instance.Scalar("int32", name="left_start")
        left_end = self.instance.Scalar("int32", name="left_end")
        mid_start = self.instance.Scalar("int32", name="mid_start")
        mid_end = self.instance.Scalar("int32", name="mid_end")
        right_start = self.instance.Scalar("int32", name="right_start")
        right_end = self.instance.Scalar("int32", name="right_end")
        left_start.set_as(w_offset)
        left_end.set_as(w_offset)
        mid_start.set_as(w_offset)
        mid_end.set_as(w_offset + w_num)
        right_start.set_as(w_offset + w_num)
        right_end.set_as(w_offset + w_num)
        if self.pad_left > 0:
            with self.instance.if_scope(w_offset < self.wo_start):
                with self.instance.if_scope(w_offset + w_num < self.wo_start):
                    left_end.set_as(w_offset + w_num)
                    mid_start.set_as(w_offset + w_num)
                with self.instance.else_scope():
                    left_end.set_as(self.wo_start)
                    mid_start.set_as(self.wo_start)
        if self.pad_right > 0:
            with self.instance.if_scope(w_offset + w_num >= self.wo_end):
                with self.instance.if_scope(w_offset < self.wo_end):
                    right_start.set_as(self.wo_end)
                    mid_end.set_as(self.wo_end)
                with self.instance.else_scope():
                    right_start.set_as(w_offset)
                    mid_end.set_as(w_offset)

        with self.instance.for_range(0, self.filter_h) as fh_i:
            h_in = h_offset * self.stride_h - self.pad_top + fh_i * self.rate_h
            expand_start = (fh_i % 2) * fh_size
            x_start = fh_i * self.rate_h * w_all * self.c0
            if self.tiling_params["is_filter_all"]:
                filter_start = c1_offset * self.filter_offset_list[1] + fh_i * self.filter_offset_list[2]
            else:
                filter_start = fh_i * self.filter_offset_list[2]

            with self.instance.if_scope(tik.all(h_in >= 0, h_in < self.h_in)):
                self.vector_add([expand_start, x_start, filter_start], ub_list, rep_stride_list + blk_stride_list,
                                [w_num, self.filter_w * self.c0])
                if self.pad_left > 0:
                    with self.instance.if_scope(left_end > left_start):
                        with self.instance.for_range(left_start, left_end) as wo_i:
                            fw_i = (self.pad_left - wo_i * self.stride_w + self.rate_w - 1) // self.rate_w
                            l_expand = expand_start + (wo_i - w_offset) * w_size
                            self.vector_dup(l_expand, expand_ub, fw_i * self.c0, Constant.MIN_VAL)
                if self.pad_right > 0:
                    with self.instance.if_scope(right_end > right_start):
                        with self.instance.for_range(right_start, right_end) as wo_i:
                            end = (self.w_in - (wo_i * self.stride_w - self.pad_left) + self.rate_w - 1) // self.rate_w
                            num = self.filter_w - end
                            r_expand = expand_start + (wo_i - w_offset) * w_size + end * self.c0
                            self.vector_dup(r_expand, expand_ub, num * self.c0, Constant.MIN_VAL)
            with self.instance.else_scope():
                self.vector_dup(expand_start, expand_ub, w_num * w_size, Constant.MIN_VAL)
            self.reduce_h(fh_i, fh_size, expand_ub)
        reduce_index = self.reduce_w(fh_size, fw_size, expand_ub)
        self.instance.data_move(self.y_gm[out_offset], expand_ub[reduce_index], 0, 1, fw_size // self.block_size, 0, 0)
