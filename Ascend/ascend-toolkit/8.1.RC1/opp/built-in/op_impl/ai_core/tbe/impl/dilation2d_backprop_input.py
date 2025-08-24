#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
dilation2d backprop input
"""
from impl import common_util
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=too-few-public-methods,not-use-list-comprehension
class Constant:
    """
    The class for constant
    """
    # 'define a scalar, value = -(2**32 - 1)
    MIN_VAL = -3402823424.0


# 'pylint: disable=invalid-name,unused-argument,too-many-arguments,too-many-locals,unused-variable
# 'pylint: disable=redefined-builtin,too-many-lines,too-many-instance-attributes
# 'pylint: disable=too-many-statements,too-many-public-methods
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_LIST_INT, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_STR,
                            para_check.KERNEL_NAME)
def dilation2d_backprop_input(x,
                              filter,
                              out_backprop,
                              y,
                              strides,
                              rates,
                              padding_mode="SAME",
                              pads=(0, 0, 0, 0),
                              ceil_mode=False,
                              data_format="NHWC",
                              kernel_name="dilation2d_backprop_input"):
    """
    Returns the grayscale dilation of x and filter tensors

    Parameters
    ------------
    x: dict, dict of x, include keys(shape and dtype)
    filter: dict, dict of filter
    out_backprop: dict, dict of out_backprop
    y: dict, dict of output
    strides: list or tuple, the strides of sliding window, only support in H or W
    rates: list or tuple, the input strides for atrous morphological dilation, only support in H or W
    padding_mode: str, the mode of padding, support padding and not padding
    pads: list or tuple, the fill value of input
    ceil_mode: bool, use ceil or floor to calculate ho and wo while padding_mode is CALCULATED
    data_format: str, default = "NHWC"
    kernel_name: str, cce kernel name, default value is "dilation2d_backprop_input"

    Returns
    -------
    tik_instance: tik_instance
    """
    check_list = ["float32"]
    x_dtype = x.get("dtype").lower()
    x_format = x.get("format")
    x_shape = x.get("shape")
    x_ori_format = x.get("ori_format")
    filter_format = filter.get("format")
    filter_shape = filter.get("shape")

    para_check.check_format(x_format, "NC1HWC0", param_name="x")
    para_check.check_dtype(x_dtype, check_list, param_name="x")
    para_check.check_shape(x_shape, min_rank=5, max_rank=5, param_name="x")
    para_check.check_format(filter_format, "NC1HWC0", param_name="filter")
    para_check.check_shape(filter_shape, min_rank=5, max_rank=5, param_name="filter")

    if padding_mode not in ("SAME", "VALID", "CALCULATED"):
        error_manager_vector.raise_err_pad_mode_invalid(kernel_name, "SAME, VALID, CALCULATED", padding_mode)

    if len(strides) != 4:
        error_manager_vector.raise_err_check_params_rules(kernel_name, "the length of strides should be 4", "strides",
                                                          strides)
    if len(rates) != 4:
        error_manager_vector.raise_err_check_params_rules(kernel_name, "the length of rates should be 4", "rates",
                                                          rates)
    if len(pads) != 4:
        error_manager_vector.raise_err_check_params_rules(kernel_name, "the length of pads should be 4", "pads", pads)

    filter_h = filter_shape[2]
    filter_w = filter_shape[3]

    input_h = x_shape[2]
    input_w = x_shape[3]

    if x_ori_format == "NHWC":
        stride_n, stride_h, stride_w, stride_c = strides
        rate_n, rate_h, rate_w, rate_c = rates
    else:
        stride_n, stride_c, stride_h, stride_w = strides
        rate_n, rate_c, rate_h, rate_w = rates

    window_h = (filter_h - 1) * rate_h + 1
    window_w = (filter_w - 1) * rate_w + 1

    if stride_n != 1 or stride_c != 1:
        error_manager_vector.raise_err_check_params_rules(kernel_name, "n, c of strides should be 1", "strides",
                                                          strides)

    if rate_n != 1 or rate_c != 1:
        error_manager_vector.raise_err_check_params_rules(kernel_name, "n, c of rates should be 1", "rates", rates)

    if list(pads) == [0, 0, 0, 0] and window_h == input_h and window_w == input_w:
        if padding_mode == "CALCULATED":
            padding_mode = "VALID"
        if padding_mode == "SAME" and stride_h == input_h and stride_w == input_w:
            padding_mode = "VALID"

    input_params = {
        "x": x,
        "filter": filter,
        "out_backprop": out_backprop,
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
    input_params["pads"] = cal_pads
    obj = Dilation2D(input_params)
    obj.dilation_compute()
    obj.instance.BuildCCE(kernel_name=kernel_name,
                          inputs=[obj.x_gm, obj.filter_gm, obj.out_backprop_gm],
                          outputs=[obj.y_gm])
    return obj.instance


def check_supported(x,
                    filter,
                    out_backprop,
                    y,
                    strides,
                    rates,
                    padding_mode="SAME",
                    pads=(0, 0, 0, 0),
                    ceil_mode=False,
                    data_format="NHWC",
                    kernel_name="dilation2d_backprop_input"):
    """
    verify the types and params of dilation2d_backprop_input supported by tbe
    """
    x_shape = x.get("ori_shape")
    x_format = x.get("ori_format")
    filter_shape = filter.get("ori_shape")

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
        return False

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
        reason  = "window_w gt x_w or window_h gt x_h,window_w is %s, window_h is %s, x_shape is %s" \
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
    return cal_pads


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
    if isinstance(val, (tik.tik_lib.tik_expr.Expr, tik.api.tik_scalar.Scalar)):
        return False
    return True


class Dilation2DBase:
    """
    use to store dilation2d_bakcprop_input base parameters
    """

    def __init__(self, input_params):
        """
        init shape and format information
        """
        self.instance = tik.Tik(tik.Dprofile())
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.input_params = input_params
        self.x_dtype = input_params.get("x").get("dtype").lower()
        self.x_shape = input_params.get("x").get("shape")
        self.x_dsize = common_util.get_data_size(self.x_dtype)
        self.x_block_size = 32 // self.x_dsize
        self.batch = self.x_shape[0]
        self.c1 = self.x_shape[1]
        self.h_in = self.x_shape[2]
        self.w_in = self.x_shape[3]
        self.c0 = self.x_shape[4]

        self.filter_shape = input_params.get("filter").get("shape")
        self.filter_h = self.filter_shape[2]
        self.filter_w = self.filter_shape[3]

        self.out_backprop_shape = input_params.get("out_backprop").get("shape")
        self.out_backprop_h = self.out_backprop_shape[2]
        self.out_backprop_w = self.out_backprop_shape[3]

        self.y_dtype = input_params.get("y").get("dtype").lower()
        self.y_shape = input_params.get("y").get("shape")
        self.y_dsize = common_util.get_data_size(self.y_dtype)
        self.y_block_size = 32 // self.y_dsize

        self.rate_h = input_params.get("rate_h")
        self.rate_w = input_params.get("rate_w")
        self.stride_h = input_params.get("stride_h")
        self.stride_w = input_params.get("stride_w")
        self.window_h = input_params.get("window_h")
        self.window_w = input_params.get("window_w")
        self.pad_top, self.pad_bottom, self.pad_left, self.pad_right = input_params.get("pads")

        self.offset_list = _get_product_of_each_dim(self.out_backprop_shape[:-1], len(self.out_backprop_shape[:-1]))
        self.repeat_max = self.instance.Scalar("uint32", name="repeat_max")
        self.repeat_max.set_as(255)

    def get_input_need_size(self, shape):
        """
        calculate the needed input size
        """
        n_factor, c1_factor, ho_factor, wo_factor = shape
        h = (ho_factor - 1) * self.stride_h + self.window_h
        w = (wo_factor - 1) * self.stride_w + self.window_w
        if ho_factor == self.out_backprop_h:
            ho_factor = max(self.h_in, h)
        else:
            ho_factor = h
        if wo_factor == self.out_backprop_w:
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

        return size

    def get_filter_need_size(self, ub_size):
        """
        calculate the needed filter size
        """
        filter_all = self.c1 * self.filter_h * self.filter_w * self.c0
        if ub_size // 3 > filter_all:
            return filter_all, True
        return self.filter_h * self.filter_w * self.c0, False

    def get_out_backprop_size(self, shape):
        """
        calculate the needed out_backprop size
        """
        n_factor, c1_factor, ho_factor, wo_factor = shape
        size = n_factor * c1_factor * ho_factor * wo_factor * self.c0

        return size

    def get_mask_ub_size(self, shape):
        """
        calculate the needed mask_ub size
        """
        n_factor, c1_factor, ho_factor, wo_factor = shape
        align_num = (n_factor * c1_factor * ho_factor * wo_factor // 4 + 1) * 4
        size = (self.filter_w * self.filter_h) * (align_num // 4 * 32)

        return size

    def get_update_matrix_size(self, shape):
        """
        calculate the needed update_matrix size
        """
        n_factor, c1_factor, ho_factor, wo_factor = shape
        size = (self.filter_h * self.filter_w * n_factor * c1_factor * ho_factor * wo_factor * self.c0 // 64 + 1) * 64

        return size

    def get_max_data_size(self, shape):
        """
        calculate the needed max_data size
        """
        n_factor, c1_factor, ho_factor, wo_factor = shape
        size = (n_factor * c1_factor * ho_factor * wo_factor * self.c0 // 64 + 1) * 64

        return size

    def get_expand_ub_b_padding_size(self, shape):
        """
        calculate the needed expand_ub_b_padding size
        """
        n_factor, c1_factor, ho_factor, wo_factor = shape
        size = (n_factor * c1_factor * ho_factor * wo_factor * self.c0 // 64 + 1) * 64

        return size

    def get_out_backprop_ub_padding_size(self, shape):
        """
        calculate the needed backprop_ub_padding size
        """
        n_factor, c1_factor, ho_factor, wo_factor = shape
        size = (n_factor * c1_factor * ho_factor * wo_factor * self.c0 // 64 + 1) * 64

        return size

    def get_val_size(self, shape):
        """
        get_val_size
        """
        n_factor, c1_factor, ho_factor, wo_factor = shape
        ge_cmp_ret_size = 4
        nt_ge_cmp_ret_size = 4
        sel_zero_ub_size = (n_factor * c1_factor * ho_factor * wo_factor * self.c0 // 64 + 1) * 64
        size = ge_cmp_ret_size + nt_ge_cmp_ret_size + sel_zero_ub_size

        return size

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

    def vector_conv(self, start_index_list, ub_list, size):
        """
        vector_conv function, trans fp16 to fp32
        """
        one_cnt = 64
        repeat = size // one_cnt
        remainder = size % one_cnt
        loop_repeat = repeat // 255
        loop_remainder = repeat % 255
        dst_ub, src_ub = ub_list
        dst_index, src_index = start_index_list
        if _is_immediate(size):
            if loop_repeat > 0:
                with self.instance.for_range(0, loop_repeat) as l_i:
                    offset_repeat = l_i * one_cnt * 255
                    self.instance.vconv(one_cnt, "", dst_ub[dst_index + offset_repeat],
                                        src_ub[src_index + offset_repeat], 255, 1, 1, 8, 4)
            if loop_remainder > 0:
                offset_remainder = loop_repeat * one_cnt * 255
                self.instance.vconv(one_cnt, "", dst_ub[dst_index + offset_remainder],
                                    src_ub[src_index + offset_remainder], loop_remainder, 1, 1, 8, 4)
            if remainder > 0:
                offset = loop_repeat * 255 * one_cnt + loop_remainder * one_cnt
                self.instance.vconv(remainder, "", dst_ub[dst_index + offset], src_ub[src_index + offset], 1, 1, 1, 8,
                                    4)
        else:
            loop_remainder_s = self.instance.Scalar("uint32", name="dup_loop_remainder")
            mask = self.instance.Scalar("uint32", name="dup_mask")
            loop_remainder_s.set_as(loop_remainder)
            mask.set_as(remainder)
            with self.instance.for_range(0, loop_repeat) as l_i:
                offset_repeat = l_i * one_cnt * 255
                self.instance.vconv(one_cnt, "", dst_ub[dst_index + offset_repeat], src_ub[src_index + offset_repeat],
                                    self.repeat_max, 1, 1, 8, 4)
            with self.instance.if_scope(loop_remainder > 0):
                offset_remainder = loop_repeat * one_cnt * 255
                self.instance.vconv(one_cnt, "", dst_ub[dst_index + offset_remainder],
                                    src_ub[src_index + offset_remainder], loop_remainder_s, 1, 1, 8, 4)
            with self.instance.if_scope(remainder > 0):
                offset = loop_repeat * 255 * one_cnt + loop_remainder * one_cnt
                self.instance.vconv(mask, "", dst_ub[dst_index + offset], src_ub[src_index + offset], 1, 1, 1, 8, 4)

    def vector_dup(self, start_index, ub_buf, size, val):
        """
        vector_dup function, set ub_buf to 0
        """

        one_cnt = 64
        repeat = size // one_cnt
        remainder = size % one_cnt
        loop_repeat = repeat // 255
        loop_remainder = repeat % 255
        end = self.instance.Scalar("uint32", name="end")

        if _is_immediate(size):
            if loop_repeat > 0:
                with self.instance.for_range(0, loop_repeat) as l_i:
                    offset_repeat = l_i * one_cnt * 255
                    end.set_as(offset_repeat)
                    self.instance.vec_dup(one_cnt, ub_buf[start_index + end], val, 255, 8)
            if loop_remainder > 0:
                offset_remainder = loop_repeat * one_cnt * 255
                end.set_as(offset_remainder)
                self.instance.vec_dup(one_cnt, ub_buf[start_index + end], val, loop_remainder, 8)
            if remainder > 0:
                offset = loop_repeat * 255 * one_cnt + loop_remainder * one_cnt
                end.set_as(offset)
                self.instance.vec_dup(remainder, ub_buf[start_index + end], val, 1, 8)
        else:
            loop_remainder_s = self.instance.Scalar("uint32", name="dup_loop_remainder")
            mask = self.instance.Scalar("uint32", name="dup_mask")
            loop_remainder_s.set_as(loop_remainder)
            mask.set_as(remainder)
            with self.instance.for_range(0, loop_repeat) as l_i:
                offset_repeat = l_i * one_cnt * 255
                end.set_as(offset_repeat)
                self.instance.vec_dup(one_cnt, ub_buf[start_index + end], val, self.repeat_max, 8)
            with self.instance.if_scope(loop_remainder > 0):
                offset_remainder = loop_repeat * one_cnt * 255
                end.set_as(offset_remainder)
                self.instance.vec_dup(one_cnt, ub_buf[start_index + end], val, loop_remainder_s, 8)
            with self.instance.if_scope(remainder > 0):
                offset = loop_repeat * 255 * one_cnt + loop_remainder * one_cnt
                end.set_as(offset)
                self.instance.vec_dup(mask, ub_buf[start_index + end], val, 1, 8)

    def vector_add(self, start_list, ub_list, size, rep_stride_list, blk_stride_list, mask_all):
        """
        vector_add function
        """
        num = mask_all // self.c0
        mask = num * 8

        dst_start, src0_start, src1_start = start_list
        dst_ub, src0_ub, src1_ub = ub_list
        dst_blk_stride, src0_blk_stride, src1_blk_stride = blk_stride_list
        dst_rep_stride, src0_rep_stride, src1_rep_stride = rep_stride_list
        repeat = size // num
        remainder = size % num
        loop_repeat = repeat // 255
        loop_remainder = repeat % 255

        if _is_immediate(repeat):
            if loop_repeat > 0:
                with self.instance.for_range(0, loop_repeat) as l_i:
                    dst_index = dst_start + l_i * 255 * dst_rep_stride * self.c0
                    src0_index = src0_start + l_i * 255 * src0_rep_stride * self.c0
                    src1_index = src1_start + l_i * 255 * src1_rep_stride * self.c0
                    self.instance.vadd(mask, dst_ub[dst_index], src0_ub[src0_index], src1_ub[src1_index], 255,
                                       dst_blk_stride * 2, src0_blk_stride * 2, src1_blk_stride * 2, dst_rep_stride * 2,
                                       src0_rep_stride * 2, src1_rep_stride * 2)
            if loop_remainder > 0:
                dst_index = dst_start + loop_repeat * 255 * dst_rep_stride * self.c0
                src0_index = src0_start + loop_repeat * 255 * src0_rep_stride * self.c0
                src1_index = src1_start + loop_repeat * 255 * src1_rep_stride * self.c0
                self.instance.vadd(mask, dst_ub[dst_index], src0_ub[src0_index], src1_ub[src1_index], loop_remainder,
                                   dst_blk_stride * 2, src0_blk_stride * 2, src1_blk_stride * 2, dst_rep_stride * 2,
                                   src0_rep_stride * 2, src1_rep_stride * 2)
            if remainder > 0:
                dst_index = dst_start + repeat * dst_rep_stride * self.c0
                src0_index = src0_start + repeat * src0_rep_stride * self.c0
                src1_index = src1_start + repeat * src1_rep_stride * self.c0
                self.instance.vadd(remainder * 8, dst_ub[dst_index], src0_ub[src0_index], src1_ub[src1_index], 1,
                                   dst_blk_stride * 2, src0_blk_stride * 2, src1_blk_stride * 2, dst_rep_stride * 2,
                                   src0_rep_stride * 2, src1_rep_stride * 2)
        else:
            with self.instance.for_range(0, loop_repeat) as l_i:
                dst_index = dst_start + l_i * 255 * dst_rep_stride * self.c0
                src0_index = src0_start + l_i * 255 * src0_rep_stride * self.c0
                src1_index = src1_start + l_i * 255 * src1_rep_stride * self.c0
                self.instance.vadd(mask, dst_ub[dst_index], src0_ub[src0_index], src1_ub[src1_index], self.repeat_max,
                                   dst_blk_stride * 2, src0_blk_stride * 2, src1_blk_stride * 2, dst_rep_stride * 2,
                                   src0_rep_stride * 2, src1_rep_stride * 2)

            loop_remainder_s = self.instance.Scalar("uint32", name="loop_remainder_s")
            loop_remainder_s.set_as(loop_remainder)
            with self.instance.if_scope(loop_remainder > 0):
                dst_index = dst_start + loop_repeat * 255 * dst_rep_stride * self.c0
                src0_index = src0_start + loop_repeat * 255 * src0_rep_stride * self.c0
                src1_index = src1_start + loop_repeat * 255 * src1_rep_stride * self.c0
                self.instance.vadd(mask, dst_ub[dst_index], src0_ub[src0_index], src1_ub[src1_index], loop_remainder_s,
                                   dst_blk_stride * 2, src0_blk_stride * 2, src1_blk_stride * 2, dst_rep_stride * 2,
                                   src0_rep_stride * 2, src1_rep_stride * 2)
            remainder_s = self.instance.Scalar("uint32", name="remainder_s")
            remainder_s.set_as(remainder * 8)
            with self.instance.if_scope(loop_remainder > 0):
                dst_index = dst_start + repeat * dst_rep_stride * self.c0
                src0_index = src0_start + repeat * src0_rep_stride * self.c0
                src1_index = src1_start + repeat * src1_rep_stride * self.c0
                self.instance.vadd(remainder_s, dst_ub[dst_index], src0_ub[src0_index], src1_ub[src1_index], 1,
                                   dst_blk_stride * 2, src0_blk_stride * 2, src1_blk_stride * 2, dst_rep_stride * 2,
                                   src0_rep_stride * 2, src1_rep_stride * 2)


class Dilation2D(Dilation2DBase):
    """
    use to store dilation2d_backprop_input compute parameters
    """

    def __init__(self, input_params):
        """
        init gm and offset information
        """
        super(Dilation2D, self).__init__(input_params)

        x_size = _get_shape_size(self.x_shape)
        filter_size = _get_shape_size(self.filter_shape)
        out_backprop_size = _get_shape_size(self.out_backprop_shape)
        y_size = _get_shape_size(self.y_shape)
        self.x_gm = self.instance.Tensor(self.x_dtype, (x_size,), name="x_gm", scope=tik.scope_gm)
        self.filter_gm = self.instance.Tensor(self.x_dtype, (filter_size,), name="filter_gm", scope=tik.scope_gm)
        self.out_backprop_gm = self.instance.Tensor(self.x_dtype, (out_backprop_size,),
                                                    name="out_backprop_gm",
                                                    scope=tik.scope_gm)
        self.y_gm = self.instance.Tensor(self.x_dtype, (self.h_in, y_size // self.h_in // 16, 16),
                                         name="y_gm",
                                         scope=tik.scope_gm,
                                         is_atomic_add=True)
        self.x_offset_list = _get_product_of_each_dim(self.x_shape, len(self.x_shape))
        self.filter_offset_list = _get_product_of_each_dim(self.filter_shape, len(self.filter_shape))
        self.tiling_params = {}
        self.pad_w = self.pad_left + self.pad_right + self.w_in
        self.need_vconv = ((self.x_dtype == "float16") and (self.y_dtype == "float32"))

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
            "expand": self.filter_h * self.filter_w * self.c0,
            "out_backprop": self.c0
        }

        for i, elem in enumerate(ub_tiling_shape):
            [find, t_factor, size_info, is_all] = self.try_tiling(i, elem, block_index, tiling_shape, ub_size)
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

    def try_tiling(self, index, element, block_index, tiling_shape, ub_size):
        """
        try to do tiling
        """
        find = False
        t_factor = 1
        is_all = False
        input = self.window_h * self.window_w * self.c0
        filter = self.filter_h * self.filter_w * self.c0
        expand = self.filter_h * self.filter_w * self.c0
        out_backprop = self.c0

        for t_factor in range(element, 0, -1):
            tmp_shape = [1] * (block_index + index)
            tmp_shape.extend([t_factor])
            tmp_shape.extend(tiling_shape[block_index + index + 1:len(tiling_shape) - 1])

            expand = self.get_expand_need_size(tmp_shape)
            filter, is_all = self.get_filter_need_size(ub_size)
            input = self.get_input_need_size(tmp_shape)
            out_backprop = self.get_out_backprop_size(tmp_shape)

            mask_ub = self.get_mask_ub_size(tmp_shape)
            update_matrix = self.get_update_matrix_size(tmp_shape)
            max_data = self.get_max_data_size(tmp_shape)
            expand_ub_b_padding = self.get_expand_ub_b_padding_size(tmp_shape)
            out_backprop_padding = self.get_out_backprop_ub_padding_size(tmp_shape)
            val_size = self.get_val_size(tmp_shape)

            if self.need_vconv:
                size = expand + filter + input + input // 2 + filter // 2 + \
                       out_backprop + mask_ub + update_matrix + max_data + \
                       expand_ub_b_padding + out_backprop + val_size
            else:
                size = expand + filter + input + out_backprop + \
                       mask_ub + update_matrix + max_data + \
                       expand_ub_b_padding + out_backprop_padding + val_size

            if size <= ub_size:
                find = True
                break

        info = {
            "input": input,
            "filter": filter,
            "expand": expand,
            "out_backprop": out_backprop,
            "mask_ub": mask_ub,
            "update_matrix": update_matrix,
            "max_data": max_data,
            "expand_ub_b_padding": expand_ub_b_padding,
            "out_backprop_padding": out_backprop_padding,
            "val_size": val_size
        }
        return_list = [find, t_factor, info, is_all]

        return return_list

    def dilation_compute(self):
        """
        dilation2d_backprop_input compute function
        """
        ub_size = (self.ub_size - 4 * 1024) // self.x_dsize
        flag = self.do_tiling(self.out_backprop_shape, ub_size // 2)
        self.tiling_params["thread_num"] = 1
        if self.tiling_params["ub_index"] == 3:
            flag = False

        if flag:
            size = self.tiling_params["block_cycle"] * self.tiling_params["block_element"]
            db_num = size // self.tiling_params["ub_num"]
            self.tiling_params["thread_num"] = 2 if db_num >= 2 else 1

        if not flag or self.tiling_params["thread_num"] == 1:
            flag_1 = self.do_tiling(self.out_backprop_shape, ub_size)
            if flag_1:
                self.tiling_params["thread_num"] = 1
            else:
                error_manager_vector.raise_err_specific_reson("dilation2d_backprop_input",
                                                              "can not find tiling, filter or rates is too big")

        block_num = self.tiling_params["block_num"]
        block_cycle = self.tiling_params["block_cycle"]
        block_element = self.tiling_params["block_element"]
        block_tail = self.tiling_params["block_tail"]
        ub_num = self.tiling_params["ub_num"]
        thread_num = self.tiling_params["thread_num"]

        with self.instance.for_range(0, block_num, block_num=block_num) as block_idx:
            each_cycle = self.instance.Scalar("uint32", name="each_cycle")
            offset = self.instance.Scalar("uint32", name="offset")
            if block_tail == 0:
                each_cycle.set_as(block_cycle * block_element)
                offset.set_as(block_idx * each_cycle)
            else:
                with self.instance.if_scope(block_idx < block_tail):
                    each_cycle.set_as((block_cycle + 1) * block_element)
                    offset.set_as(block_idx * each_cycle)
                with self.instance.else_scope():
                    each_cycle.set_as(block_cycle * block_element)
                    offset.set_as((block_idx * block_cycle + block_tail) * block_element)

            filter_ub = self.prepare_filter()
            if self.tiling_params["ub_index"] == 3:
                loop = each_cycle // self.out_backprop_w
                ub_loop = self.out_backprop_w // ub_num
                ub_tail = self.out_backprop_w % ub_num
                with self.instance.for_range(0, loop, thread_num=thread_num) as loop_idx:
                    with self.instance.for_range(0, ub_loop) as u_idx:
                        self.do_compute(ub_num, offset + loop_idx * self.out_backprop_w + u_idx * ub_num, filter_ub)
                    if ub_tail != 0:
                        self.do_compute(ub_tail, offset + loop_idx * self.out_backprop_w + ub_loop * ub_num, filter_ub)
            else:
                ub_loop = each_cycle // ub_num
                ub_tail = each_cycle % ub_num
                with self.instance.for_range(0, ub_loop, thread_num=thread_num) as loop_idx:
                    self.do_compute(ub_num, offset + loop_idx * ub_num, filter_ub)
                with self.instance.if_scope(ub_tail > 0):
                    self.do_compute(ub_tail, offset + ub_loop * ub_num, filter_ub)

    def move_filter_data(self, filter_size, filter_ub, gm_index):
        """
        move_filter_data
        """
        if self.need_vconv:
            filter_ub_fp16 = self.instance.Tensor(self.x_dtype, (filter_size,),
                                                  name="filter_ub_fp16",
                                                  scope=tbe_platform.scope_ubuf)
            self.instance.data_move(filter_ub_fp16, self.filter_gm[gm_index], 0, 1, filter_size // self.x_block_size, 0,
                                    0)
            self.vector_conv([0, 0], [filter_ub, filter_ub_fp16], filter_size)
        else:
            self.instance.data_move(filter_ub, self.filter_gm[gm_index], 0, 1, filter_size // self.y_block_size, 0, 0)

    def move_out_backprop_data(self, size, out_backprop_ub, gm_index):
        """
        move_out_backprop_data
        """
        if self.need_vconv:
            out_backprop_size = self.tiling_params["ub_size_info"].get("out_backprop")
            out_backprop_fp16 = self.instance.Tensor(self.x_dtype, (out_backprop_size,),
                                                     name="out_backprop_fp16",
                                                     scope=tbe_platform.scope_ubuf)
            self.instance.data_move(out_backprop_fp16, self.out_backprop_gm[gm_index], 0, 1, size // self.x_block_size,
                                    0, 0)
            self.vector_conv([0, 0], [out_backprop_ub, out_backprop_fp16], size)
        else:
            self.instance.data_move(out_backprop_ub, self.out_backprop_gm[gm_index], 0, 1, size // self.y_block_size, 0,
                                    0)

    def move_data(self, x_ub, ub_offset, x_index, n_burst):
        """
        move data from gm to ub
        """
        if self.need_vconv:
            x_size = self.tiling_params["ub_size_info"].get("input")
            x_ub_fp16 = self.instance.Tensor(self.x_dtype, (x_size,), name="x_ub_fp16", scope=tbe_platform.scope_ubuf)
            with self.instance.if_scope(tik.any(n_burst > 4095, self.w_in > 65535)):
                with self.instance.for_range(0, n_burst) as h_i:
                    self.instance.data_move(x_ub_fp16[ub_offset + h_i * self.pad_w * self.c0],
                                            self.x_gm[x_index + h_i * self.w_in * self.c0], 0, 1,
                                            self.w_in * self.c0 // self.x_block_size, 0, 0)
            with self.instance.else_scope():
                self.instance.data_move(x_ub_fp16[ub_offset], self.x_gm[x_index], 0, n_burst,
                                        self.w_in * self.c0 // self.x_block_size, 0,
                                        (self.pad_w - self.w_in) * self.c0 // self.x_block_size)
            self.vector_conv([0, 0], [x_ub, x_ub_fp16], x_size)
        else:
            with self.instance.if_scope(tik.any(n_burst > 4095, self.w_in > 65535)):
                with self.instance.for_range(0, n_burst) as h_i:
                    self.instance.data_move(x_ub[ub_offset + h_i * self.pad_w * self.c0],
                                            self.x_gm[x_index + h_i * self.w_in * self.c0], 0, 1,
                                            self.w_in * self.c0 // self.y_block_size, 0, 0)
            with self.instance.else_scope():
                self.instance.data_move(x_ub[ub_offset], self.x_gm[x_index], 0, n_burst,
                                        self.w_in * self.c0 // self.y_block_size, 0,
                                        (self.pad_w - self.w_in) * self.c0 // self.y_block_size)

    def prepare_filter(self):
        """
        prepare the filter ub tensor
        """
        filter_size = _get_shape_size(self.filter_shape)
        if self.tiling_params["is_filter_all"]:
            filter_ub = self.instance.Tensor(self.y_dtype, (filter_size,),
                                             name="filter_ub",
                                             scope=tbe_platform.scope_ubuf)
            self.move_filter_data(filter_size, filter_ub, 0)
            return filter_ub
        return None

    def do_compute(self, num, offset, filter_ub):
        """
        describes the calculation of add and max
        """
        ub_index = self.tiling_params["ub_index"]
        if filter_ub is None:
            filter_size = self.tiling_params["ub_size_info"].get("filter")
            filter_ub = self.instance.Tensor(self.y_dtype, (filter_size,),
                                             name="filter_ub",
                                             scope=tbe_platform.scope_ubuf)

        x_size = self.tiling_params["ub_size_info"].get("input")
        x_ub = self.instance.Tensor(self.y_dtype, (x_size,), name="x_ub", scope=tbe_platform.scope_ubuf)

        expand_size = self.tiling_params["ub_size_info"].get("expand")
        expand_ub = self.instance.Tensor(self.y_dtype, (expand_size,), name="expand_ub", scope=tbe_platform.scope_ubuf)

        out_backprop_size = self.tiling_params["ub_size_info"].get("out_backprop")
        out_backprop_ub = self.instance.Tensor(self.y_dtype, (out_backprop_size,),
                                               name="out_backprop_ub",
                                               scope=tbe_platform.scope_ubuf)

        mask_ub_size = self.tiling_params["ub_size_info"].get("mask_ub")
        mask_ub = self.instance.Tensor("uint16", (mask_ub_size,), name="mask_ub", scope=tbe_platform.scope_ubuf)

        # update_matrix_ub vector dup zero
        update_matrix_size = self.tiling_params["ub_size_info"].get("update_matrix")
        update_matrix_ub = self.instance.Tensor(self.x_dtype, (update_matrix_size,),
                                                name="update_matrix_ub",
                                                scope=tbe_platform.scope_ubuf)
        self.vector_dup(0, update_matrix_ub, update_matrix_size, 0)

        # max_data_ub vector dup zero
        max_data_size = self.tiling_params["ub_size_info"].get("max_data")
        max_data_ub = self.instance.Tensor(self.x_dtype, (max_data_size,), name="max_data_ub", scope=tik.scope_ubuf)
        self.vector_dup(0, max_data_ub, max_data_size, 0)
        # sel_zero_ub vector dup zero
        sel_zero_ub = self.instance.Tensor(self.x_dtype, (max_data_size,), name="sel_zero_ub", scope=tik.scope_ubuf)
        self.vector_dup(0, sel_zero_ub, max_data_size, 0)

        ub_list = [expand_ub, x_ub, filter_ub, out_backprop_ub, mask_ub, update_matrix_ub, max_data_ub, sel_zero_ub]
        out_offset = offset * self.c0
        offset_list = self.get_offset(offset, ub_index)

        if ub_index == 0:
            n_num = num // self.c1 // self.out_backprop_h // self.out_backprop_w
            self.cut_n(ub_list, offset_list, n_num, out_offset)
        elif ub_index == 1:
            c1_num = num // self.out_backprop_h // self.out_backprop_w
            self.cut_c1(ub_list, offset_list, c1_num, out_offset)
        elif ub_index == 2:
            h_num = num // self.out_backprop_w
            with self.instance.if_scope(h_num + offset_list[2] > self.out_backprop_h):
                h_num_1 = self.out_backprop_h - offset_list[2]
                h_num_2 = h_num - h_num_1
                self.cut_h(ub_list, offset_list, h_num_1, out_offset)

                offset_list_2 = self.get_offset(offset + h_num_1 * self.out_backprop_w, ub_index)
                out_offset_2 = (offset + h_num_1 * self.out_backprop_w) * self.c0
                self.cut_h(ub_list, offset_list_2, h_num_2, out_offset_2)
            with self.instance.else_scope():
                self.cut_h(ub_list, offset_list, h_num, out_offset)
        else:
            self.cut_w(ub_list, offset_list, num, out_offset)

    def expand_row(self, ub_list, h_in, start_list, fw_i):
        """
        expand data by filter row, reduce row and reduce col
        """
        expand_ub, _, _ = ub_list
        expand_start, x_start, filter_start = start_list
        h_size = self.out_backprop_w * self.c0

        num = min(8 // self.stride_w, 255 // 2 // self.stride_w)
        rep_num = 1 if num < 0 else num
        rep_stride_list = [rep_num, self.stride_w * rep_num, 0]
        blk_stride_list = [1, self.stride_w, 0]

        with self.instance.if_scope(tik.all(h_in >= 0, h_in < self.h_in)):
            self.vector_add(start_list, ub_list, self.out_backprop_w, rep_stride_list, blk_stride_list, num * self.c0)
            self.vector_add([expand_start + 8, x_start + 8, filter_start + 8], ub_list, self.out_backprop_w,
                            rep_stride_list, blk_stride_list, num * self.c0)
            if self.pad_left > 0:
                num = (self.pad_left - fw_i * self.rate_w + self.stride_w - 1) // self.stride_w
                self.vector_dup(expand_start, expand_ub, num * self.c0, Constant.MIN_VAL)
            if self.pad_right > 0:
                end = (self.pad_left - fw_i * self.rate_w + self.w_in + self.stride_w - 1) // self.stride_w
                num = self.out_backprop_w - end
                self.vector_dup(expand_start + end * self.c0, expand_ub, num * self.c0, Constant.MIN_VAL)
        with self.instance.else_scope():
            self.vector_dup(expand_start, expand_ub, h_size, Constant.MIN_VAL)

    # cut_n: branch1 left up area
    def cut_n_update_left_up_area(self, update_matrix_ub, n_offset):
        """
        cut_n_update_left_up_area
        """
        self.instance.set_atomic_add(1)
        c1_stride = self.c1
        cut_size = self.tiling_params["ub_size_info"].get("expand")

        # w way slide range
        w_pad_left_steps = 1 if self.pad_left == 1 else self.pad_left // self.stride_w
        w_pad_left_steps = \
            self.pad_left // self.stride_w + 1 if self.pad_left % self.stride_w != 0 else w_pad_left_steps

        # h way slide range
        h_pad_left_steps = 1 if self.pad_top == 1 else self.pad_top // self.stride_h
        h_pad_left_steps = self.pad_top // self.stride_h + 1 if self.pad_top % self.stride_h != 0 else h_pad_left_steps

        w_elem_start_offset = 1 if self.pad_left == 1 else self.pad_left // self.rate_w
        w_elem_start_offset = \
            self.pad_left // self.rate_w + 1 if self.pad_top % self.rate_w != 0 else w_elem_start_offset

        h_elem_start_offset = 1 if self.pad_top == 1 else self.pad_top // self.rate_h
        h_elem_start_offset = \
            self.pad_top // self.rate_h + 1 if self.pad_top % self.rate_h != 0 else h_elem_start_offset

        # calcute w & h elements
        w_elem_range = self.filter_w - w_elem_start_offset
        h_elem_range = self.filter_h - h_elem_start_offset

        # if slide, each line contain steps
        line_strides = self.w_in // self.stride_w if self.w_in % self.stride_w == 0 else self.w_in // self.stride_w + 1

        with self.instance.for_range(0, c1_stride) as c1_cnt:
            with self.instance.for_range(0, h_pad_left_steps) as h_step_i:
                with self.instance.for_range(0, w_pad_left_steps) as w_step_j:
                    h_range = h_elem_range + h_step_i * self.stride_h // self.rate_h
                    w_range = w_elem_range + w_step_j * self.stride_w // self.rate_w
                    with self.instance.for_range(0, h_range) as fh_i:
                        with self.instance.for_range(0, w_range) as fw_j:
                            gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                            gm_c1_offset = c1_cnt * self.w_in * self.h_in * self.c0
                            ub_c1_offset = c1_cnt * (cut_size // c1_stride)
                            gm_update_idx = fh_i * self.rate_h * self.w_in + fw_j * self.rate_w
                            ub_update_idx = ((h_elem_start_offset - h_step_i * self.stride_h + fh_i) * self.filter_w +
                                             w_elem_start_offset - w_step_j * self.stride_w + fw_j) * cut_size + \
                                            (h_step_i * line_strides + w_step_j) * self.c0
                            self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                                    update_matrix_ub[ub_c1_offset + ub_update_idx], 0, 1, 2, 0, 0)

        self.instance.set_atomic_add(0)

    # cut_n: branch2 right up area
    def cut_n_update_right_up_area(self, update_matrix_ub, n_offset):
        """
        cut_n_update_right_up_area
        """
        self.instance.set_atomic_add(1)
        c1_stride = self.c1
        cut_size = self.tiling_params["ub_size_info"].get("expand")
        dilation_window_w = (self.filter_w - 1) * self.rate_w + 1

        # w way slide range
        w_pad_right_steps = 1 if self.pad_right == 1 else self.pad_right // self.stride_w
        w_pad_right_steps = \
            self.pad_right // self.stride_w + 1 if self.pad_right % self.stride_w != 0 else w_pad_right_steps

        # h way slide range
        h_pad_right_steps = 1 if self.pad_top == 1 else self.pad_top // self.stride_h
        h_pad_right_steps = \
            self.pad_top // self.stride_h + 1 if self.pad_top % self.stride_h != 0 else h_pad_right_steps

        # right up area step offset
        w_total_step = (self.w_in + self.pad_left + self.pad_right - dilation_window_w) // self.stride_w + 1
        right_up_begin_step = w_total_step - w_pad_right_steps

        w_elem_start_offset = 0
        h_elem_start_offset = 1 if self.pad_top == 1 else self.pad_top // self.rate_h
        h_elem_start_offset = \
            self.pad_top // self.rate_h + 1 if self.pad_top % self.rate_h != 0 else h_elem_start_offset

        # calculate w & h elements
        outer_elem = right_up_begin_step * self.stride_w + dilation_window_w - (self.pad_left + self.w_in)
        outer_elem_no_dilation = 1 if outer_elem == 1 else outer_elem // self.rate_w
        outer_elem_no_dilation = \
            outer_elem // self.rate_w + 1 if outer_elem % self.rate_w != 0 else outer_elem_no_dilation

        w_elem_range = self.filter_w - outer_elem_no_dilation
        h_elem_range = self.filter_h - h_elem_start_offset

        # if stride, each line contain steps
        line_strides = self.w_in // self.stride_w if self.w_in % self.stride_w == 0 else self.w_in // self.stride_w + 1

        with self.instance.for_range(0, c1_stride) as c1_cnt:
            with self.instance.for_range(0, h_pad_right_steps) as h_step_i:
                with self.instance.for_range(0, w_pad_right_steps) as w_step_j:
                    h_range = h_elem_range + h_step_i * self.stride_h // self.rate_h
                    w_range = w_elem_range - w_step_j * self.stride_w // self.rate_w
                    with self.instance.for_range(0, h_range) as fh_i:
                        with self.instance.for_range(0, w_range) as fw_j:
                            gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                            gm_c1_offset = c1_cnt * self.w_in * self.h_in * self.c0
                            ub_c1_offset = c1_cnt * (cut_size // c1_stride)
                            gm_update_idx = (fh_i * self.rate_h) * self.w_in + \
                                            (right_up_begin_step * self.stride_w - self.pad_left) + w_step_j * \
                                            self.stride_w + fw_j * self.rate_w
                            ub_update_idx = ((h_elem_start_offset - h_step_i * self.stride_h + fh_i) * self.filter_w +
                                             w_elem_start_offset + fw_j) * cut_size + \
                                            (h_step_i * line_strides + right_up_begin_step + w_step_j) * self.c0
                            self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                                    update_matrix_ub[ub_c1_offset + ub_update_idx], 0, 1, 2, 0, 0)

        self.instance.set_atomic_add(0)

    # cut_n: branch3 center up area
    def cut_n_update_center_up_area(self, update_matrix_ub, n_offset):
        """
        cut_n_update_center_up_area
        """
        self.instance.set_atomic_add(1)
        c1_stride = self.c1
        cut_size = self.tiling_params["ub_size_info"].get("expand")
        dilation_window_w = (self.filter_w - 1) * self.rate_w + 1

        # h way slide range
        h_upcenter_steps = 1 if self.pad_top == 1 else self.pad_top // self.stride_h
        h_upcenter_steps = self.pad_top // self.stride_h + 1 if self.pad_top % self.stride_h != 0 else h_upcenter_steps

        # w way slide range
        w_left_pad_step = 1 if self.pad_left == 1 else self.pad_left // self.stride_w
        w_left_pad_step = self.pad_left // self.stride_w + 1 if self.pad_left % self.stride_w != 0 else w_left_pad_step

        w_right_pad_step = 1 if self.pad_right == 1 else self.pad_right // self.stride_w
        w_right_pad_step = \
            self.pad_right // self.stride_w + 1 if self.pad_right % self.stride_w != 0 else w_right_pad_step

        w_total_step = (self.w_in + self.pad_left + self.pad_right - dilation_window_w) // self.stride_w + 1
        w_upcenter_steps = w_total_step - w_left_pad_step - w_right_pad_step

        w_elem_start_offset = 0
        h_elem_start_offset = 1 if self.pad_top == 1 else self.pad_top // self.rate_h
        h_elem_start_offset = \
            self.pad_top // self.rate_h + 1 if self.pad_top % self.rate_h != 0 else h_elem_start_offset

        w_elem_range = self.filter_w
        h_elem_range = self.filter_h - h_elem_start_offset

        line_strides = self.w_in // self.stride_w if self.w_in % self.stride_w == 0 else self.w_in // self.stride_w + 1
        center_up_begin_step = w_left_pad_step

        with self.instance.for_range(0, c1_stride) as c1_cnt:
            with self.instance.for_range(0, h_upcenter_steps) as h_step_i:
                with self.instance.for_range(0, w_upcenter_steps) as w_step_j:
                    h_range = h_elem_range + h_step_i * self.stride_h // self.rate_h
                    w_range = w_elem_range
                    with self.instance.for_range(0, h_range) as fh_i:
                        with self.instance.for_range(0, w_range) as fw_j:
                            gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                            gm_c1_offset = c1_cnt * self.w_in * self.h_in * self.c0
                            ub_c1_offset = c1_cnt * (cut_size // c1_stride)
                            gm_update_idx = fh_i * self.rate_h * self.w_in + \
                                             w_step_j * self.stride_w + fw_j * self.rate_w
                            ub_update_idx = ((h_elem_start_offset - h_step_i * self.stride_h + fh_i) * self.filter_w +
                                             w_elem_start_offset + fw_j) * cut_size + \
                                            (center_up_begin_step + h_step_i * line_strides + w_step_j) * self.c0
                            self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                                    update_matrix_ub[ub_c1_offset + ub_update_idx], 0, 1, 2, 0, 0)

        self.instance.set_atomic_add(0)

    # cut_n: branch4 left center area
    def cut_n_update_left_center_area(self, update_matrix_ub, n_offset):
        """
        cut_n_update_left_center_area
        """
        self.instance.set_atomic_add(1)
        c1_stride = self.c1
        cut_size = self.tiling_params["ub_size_info"].get("expand")
        dilation_window_h = (self.filter_h - 1) * self.rate_h + 1
        dilation_window_w = (self.filter_w - 1) * self.rate_w + 1

        w_leftcenter_steps = 1 if self.pad_left == 1 else self.pad_left // self.stride_w
        w_leftcenter_steps = \
            self.pad_left // self.stride_w + 1 if self.pad_left % self.stride_w != 0 else w_leftcenter_steps

        h_top_steps = 1 if self.pad_top == 1 else self.pad_top // self.stride_h
        h_top_steps = self.pad_top // self.stride_h + 1 if self.pad_top % self.stride_h != 0 else h_top_steps

        h_bottom_steps = 1 if self.pad_bottom == 1 else self.pad_bottom // self.stride_h
        h_bottom_steps = \
            self.pad_bottom // self.stride_h + 1 if self.pad_bottom % self.stride_h != 0 else h_bottom_steps

        h_total_steps = (self.h_in + self.pad_top + self.pad_bottom - dilation_window_h) // self.stride_h + 1
        h_leftcenter_steps = h_total_steps - h_top_steps - h_bottom_steps

        h_elem_start_offset = 0
        w_elem_start_offset = 1 if self.pad_left == 1 else self.pad_left // self.rate_w
        w_elem_start_offset = \
            self.pad_left // self.rate_w + 1 if self.pad_left % self.rate_w != 0 else w_elem_start_offset

        w_elem_range = self.filter_w - w_elem_start_offset
        h_elem_range = self.filter_h

        left_center_begin_step = h_top_steps * (
            (self.pad_left + self.pad_right + self.w_in - dilation_window_w) // self.stride_w + 1)
        line_strides = self.w_in // self.stride_w if self.w_in % self.stride_w == 0 else self.w_in // self.stride_w + 1

        with self.instance.for_range(0, c1_stride) as c1_cnt:
            with self.instance.for_range(0, h_leftcenter_steps) as h_step_i:
                with self.instance.for_range(0, w_leftcenter_steps) as w_step_j:
                    h_range = h_elem_range
                    w_range = w_elem_range + w_step_j * self.stride_w // self.rate_w
                    with self.instance.for_range(0, h_range) as fh_i:
                        with self.instance.for_range(0, w_range) as fw_j:
                            gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                            gm_c1_offset = c1_cnt * self.w_in * self.h_in * self.c0
                            ub_c1_offset = c1_cnt * (cut_size // c1_stride)
                            gm_update_idx = (h_step_i * self.stride_h +
                                             fh_i * self.rate_h) * self.w_in + fw_j * self.rate_w
                            ub_update_idx = ((h_elem_start_offset + fh_i) * self.filter_w + w_elem_start_offset -
                                             w_step_j * self.stride_w + fw_j) * cut_size + \
                                            (left_center_begin_step + h_step_i * line_strides + w_step_j) * self.c0
                            self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                                    update_matrix_ub[ub_c1_offset + ub_update_idx], 0, 1, 2, 0, 0)

        self.instance.set_atomic_add(0)

    # cut_n: branch 5 right center area
    def cut_n_update_right_center_area(self, update_matrix_ub, n_offset):
        """
        cut_n_update_right_center_area
        """
        self.instance.set_atomic_add(1)
        cut_size = self.tiling_params["ub_size_info"].get("expand")
        c1_stride = self.c1
        dilation_window_h = (self.filter_h - 1) * self.rate_h + 1
        dilation_window_w = (self.filter_w - 1) * self.rate_w + 1

        w_rightcenter_steps = 1 if self.pad_right == 1 else self.pad_right // self.stride_w
        w_rightcenter_steps = \
            self.pad_right // self.stride_w + 1 if self.pad_right % self.stride_w != 0 else w_rightcenter_steps

        h_top_steps = 1 if self.pad_top == 1 else self.pad_top // self.stride_h
        h_top_steps = self.pad_top // self.stride_h + 1 if self.pad_top % self.stride_h != 0 else h_top_steps

        h_bottom_steps = 1 if self.pad_bottom == 1 else self.pad_bottom // self.stride_h
        h_bottom_steps = \
            self.pad_bottom // self.stride_h + 1 if self.pad_bottom % self.stride_h != 0 else h_bottom_steps

        h_total_steps = (self.h_in + self.pad_top + self.pad_bottom - dilation_window_h) // self.stride_h + 1
        h_rightcenter_steps = h_total_steps - h_top_steps - h_bottom_steps

        w_elem_start_offset = 0
        h_elem_start_offset = 0

        w_total_step = (self.w_in + self.pad_left + self.pad_right - dilation_window_w) // self.stride_w + 1
        right_center_begin_step = h_top_steps * w_total_step + w_total_step - w_rightcenter_steps

        # calculate w & h elements
        outer_elem = (w_total_step - w_rightcenter_steps) * self.stride_w + dilation_window_w - (self.pad_left +
                                                                                                 self.w_in)
        outer_elem_no_dilation = 1 if outer_elem == 1 else outer_elem // self.rate_w
        outer_elem_no_dilation = \
            outer_elem // self.rate_w + 1 if outer_elem % self.rate_w != 0 else outer_elem_no_dilation

        w_elem_range = self.filter_w - outer_elem_no_dilation
        h_elem_range = self.filter_h
        line_strides = self.w_in // self.stride_w if self.w_in % self.stride_w == 0 else self.w_in // self.stride_w + 1

        with self.instance.for_range(0, c1_stride) as c1_cnt:
            with self.instance.for_range(0, h_rightcenter_steps) as h_step_i:
                with self.instance.for_range(0, w_rightcenter_steps) as w_step_j:
                    h_range = h_elem_range
                    w_range = w_elem_range - w_step_j * self.stride_w // self.rate_w
                    with self.instance.for_range(0, h_range) as fh_i:
                        with self.instance.for_range(0, w_range) as fw_j:
                            gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                            gm_c1_offset = c1_cnt * self.w_in * self.h_in * self.c0
                            ub_c1_offset = c1_cnt * (cut_size // c1_stride)
                            gm_update_idx = (h_step_i * self.stride_h + fh_i * self.rate_h) * self.w_in + \
                                            (w_total_step - w_rightcenter_steps) * self.stride_w - self.pad_left + \
                                            w_step_j * self.stride_w + fw_j * self.rate_w
                            ub_update_idx = ((h_elem_start_offset + fh_i) * self.filter_w + w_elem_start_offset +
                                             fw_j) * cut_size + \
                                            (right_center_begin_step + h_step_i * line_strides + w_step_j) * self.c0
                            self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                                    update_matrix_ub[ub_c1_offset + ub_update_idx], 0, 1, 2, 0, 0)

        self.instance.set_atomic_add(0)

    # cut_n: branch6 left down area
    def cut_n_update_left_down_area(self, update_matrix_ub, n_offset):
        """
        cut_n_update_left_down_area
        """
        self.instance.set_atomic_add(1)
        cut_size = self.tiling_params["ub_size_info"].get("expand")
        c1_stride = self.c1
        dilation_window_h = (self.filter_h - 1) * self.rate_h + 1

        w_pad_left_steps = 1 if self.pad_left == 1 else self.pad_left // self.stride_w
        w_pad_left_steps = \
            self.pad_left // self.stride_w + 1 if self.pad_left % self.stride_w != 0 else w_pad_left_steps

        h_pad_left_steps = 1 if self.pad_bottom == 1 else self.pad_bottom // self.stride_h
        h_pad_left_steps = \
            self.pad_bottom // self.stride_h + 1 if self.pad_bottom % self.stride_h != 0 else h_pad_left_steps

        h_total_step = (self.pad_top + self.h_in + self.pad_bottom - dilation_window_h) // self.stride_h + 1
        left_down_begin_step = h_total_step - h_pad_left_steps

        h_elem_start_offset = 0
        w_elem_start_offset = 1 if self.pad_left == 1 else self.pad_left // self.rate_w
        w_elem_start_offset = \
            self.pad_left // self.rate_w + 1 if self.pad_top % self.rate_w != 0 else w_elem_start_offset

        # calculate w & h elements
        h_outer_elem = (h_total_step - h_pad_left_steps) * self.stride_h + dilation_window_h - (self.pad_top +
                                                                                                self.h_in)
        h_outer_elem_no_dilation = 1 if h_outer_elem == 1 else h_outer_elem // self.rate_h
        h_outer_elem_no_dilation = \
            h_outer_elem // self.rate_h + 1 if h_outer_elem % self.rate_h != 0 else h_outer_elem_no_dilation
        w_elem_range = self.filter_w - w_elem_start_offset
        h_elem_range = self.filter_h - h_outer_elem_no_dilation

        line_strides = self.w_in // self.stride_w if self.w_in % self.stride_w == 0 else self.w_in // self.stride_w + 1

        with self.instance.for_range(0, c1_stride) as c1_cnt:
            with self.instance.for_range(0, h_pad_left_steps) as h_step_i:
                with self.instance.for_range(0, w_pad_left_steps) as w_step_j:
                    h_range = h_elem_range - h_step_i * self.stride_h // self.rate_h
                    w_range = w_elem_range + w_step_j * self.stride_w // self.rate_w
                    with self.instance.for_range(0, h_range) as fh_i:
                        with self.instance.for_range(0, w_range) as fw_j:
                            gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                            gm_c1_offset = c1_cnt * self.w_in * self.h_in * self.c0
                            ub_c1_offset = c1_cnt * (cut_size // c1_stride)
                            gm_update_idx = (left_down_begin_step * self.stride_h - self.pad_top + h_step_i *
                                             self.stride_h + fh_i * self.rate_h) * self.w_in + fw_j * self.rate_w
                            ub_update_idx = ((h_elem_start_offset + fh_i) * self.filter_w + w_elem_start_offset -
                                             w_step_j * self.stride_w + fw_j) * cut_size + \
                                            ((left_down_begin_step + h_step_i) * line_strides + w_step_j) * self.c0
                            self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                                    update_matrix_ub[ub_c1_offset + ub_update_idx], 0, 1, 2, 0, 0)

        self.instance.set_atomic_add(0)

    # cut_n: branch7 right down area
    def cut_n_update_right_down_area(self, update_matrix_ub, n_offset):
        """
        cut_n_update_right_down_area
        """
        self.instance.set_atomic_add(1)
        cut_size = self.tiling_params["ub_size_info"].get("expand")
        c1_stride = self.c1

        dilation_window_h = (self.filter_h - 1) * self.rate_h + 1
        dilation_window_w = (self.filter_w - 1) * self.rate_w + 1

        h_down_right_steps = 1 if self.pad_bottom == 1 else self.pad_bottom // self.stride_h
        h_down_right_steps = \
            self.pad_bottom // self.stride_h + 1 if self.pad_bottom % self.stride_h != 0 else h_down_right_steps

        w_down_right_steps = 1 if self.pad_right == 1 else self.pad_right // self.stride_w
        w_down_right_steps = \
            self.pad_right // self.stride_w + 1 if self.pad_right % self.stride_w != 0 else w_down_right_steps

        w_elem_start_offset = 0
        h_elem_start_offset = 0

        h_total_step = (self.pad_top + self.h_in + self.pad_bottom - dilation_window_h) // self.stride_h + 1
        w_total_step = (self.pad_left + self.w_in + self.pad_right - dilation_window_w) // self.stride_w + 1

        # calculate w & h elements
        w_outer_elem = (w_total_step - w_down_right_steps) * self.stride_w + dilation_window_w - (self.pad_left +
                                                                                                  self.w_in)
        w_outer_elem_no_dilation = 1 if w_outer_elem == 1 else w_outer_elem // self.rate_w
        w_outer_elem_no_dilation = \
            w_outer_elem // self.rate_w + 1 if w_outer_elem % self.rate_w != 0 else w_outer_elem_no_dilation

        h_outer_elem = (h_total_step - h_down_right_steps) * self.stride_h + dilation_window_h - (self.pad_top +
                                                                                                  self.h_in)
        h_outer_elem_no_dilation = 1 if h_outer_elem == 1 else h_outer_elem // self.rate_h
        h_outer_elem_no_dilation = \
            h_outer_elem // self.rate_h + 1 if h_outer_elem % self.rate_h else h_outer_elem_no_dilation

        w_elem_range = self.filter_w - w_outer_elem_no_dilation
        h_elem_range = self.filter_h - h_outer_elem_no_dilation

        line_strides = self.w_in // self.stride_w if self.w_in % self.stride_w == 0 else self.w_in // self.stride_w + 1
        right_down_begin_step = (h_total_step - h_down_right_steps) * line_strides + \
                                 w_total_step - w_down_right_steps

        with self.instance.for_range(0, c1_stride) as c1_cnt:
            with self.instance.for_range(0, h_down_right_steps) as h_step_i:
                with self.instance.for_range(0, w_down_right_steps) as w_step_j:
                    h_range = h_elem_range - h_step_i * self.stride_h // self.rate_h
                    w_range = w_elem_range - w_step_j * self.stride_w // self.rate_w
                    with self.instance.for_range(0, h_range) as fh_i:
                        with self.instance.for_range(0, w_range) as fw_j:
                            gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                            gm_c1_offset = c1_cnt * self.w_in * self.h_in * self.c0
                            ub_c1_offset = c1_cnt * (cut_size // c1_stride)
                            gm_update_idx = ((h_total_step - h_down_right_steps) * self.stride_h - self.pad_top +
                                             h_step_i * self.stride_h + fh_i * self.rate_h) * self.w_in + \
                                            (w_total_step - w_down_right_steps) * self.stride_w - self.pad_left + \
                                            w_step_j * self.stride_w + fw_j * self.rate_w
                            ub_update_idx = ((h_elem_start_offset + fh_i) * self.filter_w + w_elem_start_offset +
                                             fw_j) * cut_size + \
                                            (right_down_begin_step + h_step_i * line_strides + w_step_j) * self.c0
                            self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                                    update_matrix_ub[ub_c1_offset + ub_update_idx], 0, 1, 2, 0, 0)

        self.instance.set_atomic_add(0)

    # cut_n: branch8 center down area
    def cut_n_update_center_down_area(self, update_matrix_ub, n_offset):
        """
        cut_n_update_center_down_area
        """
        self.instance.set_atomic_add(1)
        cut_size = self.tiling_params["ub_size_info"].get("expand")
        c1_stride = self.c1

        dilation_window_h = (self.filter_h - 1) * self.rate_h + 1
        dilation_window_w = (self.filter_w - 1) * self.rate_w + 1

        h_down_center_steps = 1 if self.pad_bottom == 1 else self.pad_bottom // self.stride_h
        h_down_center_steps = \
            self.pad_bottom // self.stride_h + 1 if self.pad_bottom % self.stride_h != 0 else h_down_center_steps

        w_left_pad_step = 1 if self.pad_left == 1 else self.pad_left // self.stride_w
        w_left_pad_step = self.pad_left // self.stride_w + 1 if self.pad_left % self.stride_w != 0 else w_left_pad_step

        w_right_pad_step = 1 if self.pad_right == 1 else self.pad_right // self.stride_w
        w_right_pad_step = \
            self.pad_right // self.stride_w + 1 if self.pad_right % self.stride_w != 0 else w_right_pad_step

        w_total_step = (self.w_in + self.pad_left + self.pad_right - dilation_window_w) // self.stride_w + 1
        w_down_center_steps = w_total_step - w_left_pad_step - w_right_pad_step

        w_elem_start_offset = 0
        h_elem_start_offset = 0

        h_total_step = (self.h_in + self.pad_top + self.pad_bottom - dilation_window_h) // self.stride_h + 1

        # calculate w & h elements
        h_outer_elem = (h_total_step - h_down_center_steps) * self.stride_h + dilation_window_h - (self.pad_top +
                                                                                                   self.h_in)
        h_outer_elem_no_dilation = 1 if h_outer_elem == 1 else h_outer_elem // self.rate_h
        h_outer_elem_no_dilation = \
            h_outer_elem // self.rate_h + 1 if h_outer_elem % self.rate_h else h_outer_elem_no_dilation

        w_elem_range = self.filter_w
        h_elem_range = self.filter_h - h_outer_elem_no_dilation

        line_strides = self.w_in // self.stride_w if self.w_in % self.stride_w == 0 else self.w_in // self.stride_w + 1
        center_down_begin_step = (h_total_step - h_down_center_steps) * line_strides + w_left_pad_step

        with self.instance.for_range(0, c1_stride) as c1_cnt:
            with self.instance.for_range(0, h_down_center_steps) as h_step_i:
                with self.instance.for_range(0, w_down_center_steps) as w_step_j:
                    h_range = h_elem_range - h_step_i * self.stride_h // self.rate_h
                    w_range = w_elem_range
                    with self.instance.for_range(0, h_range) as fh_i:
                        with self.instance.for_range(0, w_range) as fw_j:
                            gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                            gm_c1_offset = c1_cnt * self.w_in * self.h_in * self.c0
                            ub_c1_offset = c1_cnt * (cut_size // c1_stride)
                            gm_update_idx = ((h_total_step - h_down_center_steps) * self.stride_h - self.pad_top +
                                             h_step_i * self.stride_h + fh_i * self.rate_h) * self.w_in + w_step_j * \
                                            self.stride_w + fw_j * self.rate_w
                            ub_update_idx = ((h_elem_start_offset + fh_i) * self.filter_w + w_elem_start_offset +
                                             fw_j) * cut_size + \
                                            (center_down_begin_step + h_step_i * line_strides + w_step_j) * self.c0
                            self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                                    update_matrix_ub[ub_c1_offset + ub_update_idx], 0, 1, 2, 0, 0)
        self.instance.set_atomic_add(0)

    # cut_n: branch 9 no padding area
    def cut_n_update_no_padding_area(self, update_matrix_ub, n_offset):
        """
        cut_n_update_no_padding_area
        """
        self.instance.set_atomic_add(1)

        cut_size = self.tiling_params["ub_size_info"].get("expand")
        c1_stride = self.c1

        dilation_window_h = (self.filter_h - 1) * self.rate_h + 1
        dilation_window_w = (self.filter_w - 1) * self.rate_w + 1

        # calcualte w step range
        w_left_pad_step = 1 if self.pad_left == 1 else self.pad_left // self.stride_w
        w_left_pad_step = self.pad_left // self.stride_w + 1 if self.pad_left % self.stride_w != 0 else w_left_pad_step

        w_right_pad_step = 1 if self.pad_right == 1 else self.pad_right // self.stride_w
        w_right_pad_step = \
            self.pad_right // self.stride_w + 1 if self.pad_right % self.stride_w != 0 else w_right_pad_step

        w_total_step = (self.w_in + self.pad_left + self.pad_right - dilation_window_w) // self.stride_w + 1
        w_step_range = w_total_step - w_left_pad_step - w_right_pad_step

        # calculate h step range
        h_top_steps = 1 if self.pad_top == 1 else self.pad_top // self.stride_h
        h_top_steps = self.pad_top // self.stride_h + 1 if self.pad_top % self.stride_h != 0 else h_top_steps

        h_bottom_steps = 1 if self.pad_bottom == 1 else self.pad_bottom // self.stride_h
        h_bottom_steps = \
            self.pad_bottom // self.stride_h + 1 if self.pad_bottom % self.stride_h != 0 else h_bottom_steps

        h_total_steps = (self.h_in + self.pad_top + self.pad_bottom - dilation_window_h) // self.stride_h + 1
        h_step_range = h_total_steps - h_top_steps - h_bottom_steps

        line_strides = self.w_in // self.stride_w if self.w_in % self.stride_w == 0 else self.w_in // self.stride_w + 1
        begin_step = h_top_steps * line_strides + w_left_pad_step

        with self.instance.for_range(0, c1_stride) as c1_cnt:
            with self.instance.for_range(0, h_step_range) as h_step_i:
                with self.instance.for_range(0, w_step_range) as w_step_j:
                    with self.instance.for_range(0, self.filter_h) as fh_i:
                        with self.instance.for_range(0, self.filter_w) as fw_j:
                            gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                            gm_c1_offset = c1_cnt * self.w_in * self.h_in * self.c0
                            ub_c1_offset = c1_cnt * (cut_size // c1_stride)
                            gm_update_idx = (h_step_i * self.stride_h + fh_i * self.rate_h) * self.w_in + \
                                             w_step_j * self.stride_w + fw_j * self.rate_w
                            ub_update_idx = (fh_i * self.filter_w + fw_j) * cut_size + \
                                            (begin_step + h_step_i * line_strides + w_step_j) * self.c0
                            self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                                    update_matrix_ub[ub_c1_offset + ub_update_idx], 0, 1, 2, 0, 0)

        self.instance.set_atomic_add(0)

    # cut_c1: branch1 left up area
    def cut_c1_update_left_up_area(self, update_matrix_ub, n_offset, c1_offset):
        """
        cut_c1_update_left_up_area
        """
        self.instance.set_atomic_add(1)
        c1_stride = self.tiling_params["ub_factor"]
        cut_size = self.tiling_params["ub_size_info"].get("expand")

        # w way slide range
        w_pad_left_steps = 1 if self.pad_left == 1 else self.pad_left // self.stride_w
        w_pad_left_steps = \
            self.pad_left // self.stride_w + 1 if self.pad_left % self.stride_w != 0 else w_pad_left_steps

        # h way slide range
        h_pad_left_steps = 1 if self.pad_top == 1 else self.pad_top // self.stride_h
        h_pad_left_steps = self.pad_top // self.stride_h + 1 if self.pad_top % self.stride_h != 0 else h_pad_left_steps

        w_elem_start_offset = 1 if self.pad_left == 1 else self.pad_left // self.rate_w
        w_elem_start_offset = \
            self.pad_left // self.rate_w + 1 if self.pad_top % self.rate_w != 0 else w_elem_start_offset

        h_elem_start_offset = 1 if self.pad_top == 1 else self.pad_top // self.rate_h
        h_elem_start_offset = \
            self.pad_top // self.rate_h + 1 if self.pad_top % self.rate_h != 0 else h_elem_start_offset

        # calcute w & h elements
        w_elem_range = self.filter_w - w_elem_start_offset
        h_elem_range = self.filter_h - h_elem_start_offset

        # if slide, each line contain steps
        line_strides = self.w_in // self.stride_w if self.w_in % self.stride_w == 0 else self.w_in // self.stride_w + 1

        with self.instance.for_range(0, c1_stride) as c1_cnt:
            with self.instance.for_range(0, h_pad_left_steps) as h_step_i:
                with self.instance.for_range(0, w_pad_left_steps) as w_step_j:
                    h_range = h_elem_range + h_step_i * self.stride_h // self.rate_h
                    w_range = w_elem_range + w_step_j * self.stride_w // self.rate_w
                    with self.instance.for_range(0, h_range) as fh_i:
                        with self.instance.for_range(0, w_range) as fw_j:
                            gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                            gm_c1_offset = (c1_offset + c1_cnt) * self.w_in * self.h_in * self.c0
                            gm_update_idx = fh_i * self.rate_h * self.w_in + fw_j * self.rate_w
                            ub_update_idx = ((h_elem_start_offset - h_step_i * self.stride_h + fh_i) * self.filter_w +
                                             w_elem_start_offset - w_step_j * self.stride_w + fw_j) * cut_size + \
                                            c1_cnt * (cut_size // c1_stride) + (h_step_i * line_strides + w_step_j) * \
                                            self.c0
                            self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                                    update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)

        self.instance.set_atomic_add(0)

    # cut_c1: branch2 right up area
    def cut_c1_update_right_up_area(self, update_matrix_ub, n_offset, c1_offset):
        """
        cut_c1_update_right_up_area
        """
        self.instance.set_atomic_add(1)

        c1_stride = self.tiling_params["ub_factor"]
        cut_size = self.tiling_params["ub_size_info"].get("expand")
        dilation_window_w = (self.filter_w - 1) * self.rate_w + 1

        # w way slide range
        w_pad_right_steps = 1 if self.pad_right == 1 else self.pad_right // self.stride_w
        w_pad_right_steps = \
            self.pad_right // self.stride_w + 1 if self.pad_right % self.stride_w != 0 else w_pad_right_steps

        # h way slide range
        h_pad_right_steps = 1 if self.pad_top == 1 else self.pad_top // self.stride_h
        h_pad_right_steps = \
            self.pad_top // self.stride_h + 1 if self.pad_top % self.stride_h != 0 else h_pad_right_steps

        # right up area step offset
        w_total_step = (self.w_in + self.pad_left + self.pad_right - dilation_window_w) // self.stride_w + 1
        right_up_begin_step = w_total_step - w_pad_right_steps

        w_elem_start_offset = 0
        h_elem_start_offset = 1 if self.pad_top == 1 else self.pad_top // self.rate_h
        h_elem_start_offset = \
            self.pad_top // self.rate_h + 1 if self.pad_top % self.rate_h != 0 else h_elem_start_offset

        # calculate w & h elements
        outer_elem = right_up_begin_step * self.stride_w + dilation_window_w - (self.pad_left + self.w_in)
        outer_elem_no_dilation = 1 if outer_elem == 1 else outer_elem // self.rate_w
        outer_elem_no_dilation = \
            outer_elem // self.rate_w + 1 if outer_elem % self.rate_w != 0 else outer_elem_no_dilation

        w_elem_range = self.filter_w - outer_elem_no_dilation
        h_elem_range = self.filter_h - h_elem_start_offset

        # if stride, each line contain steps
        line_strides = self.w_in // self.stride_w if self.w_in % self.stride_w == 0 else self.w_in // self.stride_w + 1

        with self.instance.for_range(0, c1_stride) as c1_cnt:
            with self.instance.for_range(0, h_pad_right_steps) as h_step_i:
                with self.instance.for_range(0, w_pad_right_steps) as w_step_j:
                    h_range = h_elem_range + h_step_i * self.stride_h // self.rate_h
                    w_range = w_elem_range - w_step_j * self.stride_w // self.rate_w
                    with self.instance.for_range(0, h_range) as fh_i:
                        with self.instance.for_range(0, w_range) as fw_j:
                            gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                            gm_c1_offset = (c1_offset + c1_cnt) * self.w_in * self.h_in * self.c0
                            gm_update_idx = (fh_i * self.rate_h) * self.w_in + \
                                            (right_up_begin_step * self.stride_w - self.pad_left) + w_step_j * \
                                            self.stride_w + fw_j * self.rate_w
                            ub_update_idx = ((h_elem_start_offset - h_step_i * self.stride_h + fh_i) * self.filter_w +
                                             w_elem_start_offset + fw_j) * cut_size + c1_cnt * \
                                            (cut_size // c1_stride) + (h_step_i * line_strides + right_up_begin_step +
                                                                       w_step_j) * self.c0
                            self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                                    update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)

        self.instance.set_atomic_add(0)

    # cut_c1: branch3 center up area
    def cut_c1_update_center_up_area(self, update_matrix_ub, n_offset, c1_offset):
        """
        cut_c1_update_center_up_area
        """
        self.instance.set_atomic_add(1)

        c1_stride = self.tiling_params["ub_factor"]
        cut_size = self.tiling_params["ub_size_info"].get("expand")
        dilation_window_w = (self.filter_w - 1) * self.rate_w + 1

        # h way slide range
        h_upcenter_steps = 1 if self.pad_top == 1 else self.pad_top // self.stride_h
        h_upcenter_steps = self.pad_top // self.stride_h + 1 if self.pad_top % self.stride_h != 0 else h_upcenter_steps

        # w way slide range
        w_left_pad_step = 1 if self.pad_left == 1 else self.pad_left // self.stride_w
        w_left_pad_step = self.pad_left // self.stride_w + 1 if self.pad_left % self.stride_w != 0 else w_left_pad_step

        w_right_pad_step = 1 if self.pad_right == 1 else self.pad_right // self.stride_w
        w_right_pad_step = \
            self.pad_right // self.stride_w + 1 if self.pad_right % self.stride_w != 0 else w_right_pad_step

        w_total_step = (self.w_in + self.pad_left + self.pad_right - dilation_window_w) // self.stride_w + 1
        w_upcenter_steps = w_total_step - w_left_pad_step - w_right_pad_step

        w_elem_start_offset = 0
        h_elem_start_offset = 1 if self.pad_top == 1 else self.pad_top // self.rate_h
        h_elem_start_offset = \
            self.pad_top // self.rate_h + 1 if self.pad_top % self.rate_h != 0 else h_elem_start_offset

        w_elem_range = self.filter_w
        h_elem_range = self.filter_h - h_elem_start_offset

        line_strides = self.w_in // self.stride_w if self.w_in % self.stride_w == 0 else self.w_in // self.stride_w + 1
        center_up_begin_step = w_left_pad_step

        with self.instance.for_range(0, c1_stride) as c1_cnt:
            with self.instance.for_range(0, h_upcenter_steps) as h_step_i:
                with self.instance.for_range(0, w_upcenter_steps) as w_step_j:
                    h_range = h_elem_range + h_step_i * self.stride_h // self.rate_h
                    w_range = w_elem_range
                    with self.instance.for_range(0, h_range) as fh_i:
                        with self.instance.for_range(0, w_range) as fw_j:
                            gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                            gm_c1_offset = (c1_offset + c1_cnt) * self.w_in * self.h_in * self.c0
                            gm_update_idx = fh_i * self.rate_h * self.w_in + \
                                             w_step_j * self.stride_w + fw_j * self.rate_w
                            ub_update_idx = ((h_elem_start_offset - h_step_i * self.stride_h + fh_i) * self.filter_w +
                                             w_elem_start_offset + fw_j) * cut_size + c1_cnt * \
                                            (cut_size // c1_stride) + (center_up_begin_step + h_step_i *
                                                                       line_strides + w_step_j) * self.c0
                            self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                                    update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)

        self.instance.set_atomic_add(0)

    # cut_c1: branch4 left center area
    def cut_c1_update_left_center_area(self, update_matrix_ub, n_offset, c1_offset):
        """
        cut_c1_update_left_center_area
        """
        self.instance.set_atomic_add(1)
        c1_stride = self.tiling_params["ub_factor"]
        cut_size = self.tiling_params["ub_size_info"].get("expand")
        dilation_window_h = (self.filter_h - 1) * self.rate_h + 1
        dilation_window_w = (self.filter_w - 1) * self.rate_w + 1

        w_leftcenter_steps = 1 if self.pad_left == 1 else self.pad_left // self.stride_w
        w_leftcenter_steps = \
            self.pad_left // self.stride_w + 1 if self.pad_left % self.stride_w != 0 else w_leftcenter_steps

        h_top_steps = 1 if self.pad_top == 1 else self.pad_top // self.stride_h
        h_top_steps = self.pad_top // self.stride_h + 1 if self.pad_top % self.stride_h != 0 else h_top_steps

        h_bottom_steps = 1 if self.pad_bottom == 1 else self.pad_bottom // self.stride_h
        h_bottom_steps = \
            self.pad_bottom // self.stride_h + 1 if self.pad_bottom % self.stride_h != 0 else h_bottom_steps

        h_total_steps = (self.h_in + self.pad_top + self.pad_bottom - dilation_window_h) // self.stride_h + 1
        h_leftcenter_steps = h_total_steps - h_top_steps - h_bottom_steps

        h_elem_start_offset = 0
        w_elem_start_offset = 1 if self.pad_left == 1 else self.pad_left // self.rate_w
        w_elem_start_offset = \
            self.pad_left // self.rate_w + 1 if self.pad_left % self.rate_w != 0 else w_elem_start_offset

        w_elem_range = self.filter_w - w_elem_start_offset
        h_elem_range = self.filter_h

        left_center_begin_step = h_top_steps * (
            (self.pad_left + self.pad_right + self.w_in - dilation_window_w) // self.stride_w + 1)
        line_strides = self.w_in // self.stride_w if self.w_in % self.stride_w == 0 else self.w_in // self.stride_w + 1

        with self.instance.for_range(0, c1_stride) as c1_cnt:
            with self.instance.for_range(0, h_leftcenter_steps) as h_step_i:
                with self.instance.for_range(0, w_leftcenter_steps) as w_step_j:
                    h_range = h_elem_range
                    w_range = w_elem_range + w_step_j * self.stride_w // self.rate_w
                    with self.instance.for_range(0, h_range) as fh_i:
                        with self.instance.for_range(0, w_range) as fw_j:
                            gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                            gm_c1_offset = (c1_offset + c1_cnt) * self.w_in * self.h_in * self.c0
                            gm_update_idx = (h_step_i * self.stride_h +
                                             fh_i * self.rate_h) * self.w_in + fw_j * self.rate_w
                            ub_update_idx = ((h_elem_start_offset + fh_i) * self.filter_w + w_elem_start_offset -
                                             w_step_j * self.stride_w + fw_j) * cut_size + c1_cnt * \
                                            (cut_size // c1_stride) + (left_center_begin_step + h_step_i *
                                                                       line_strides + w_step_j) * self.c0
                            self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                                    update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)

        self.instance.set_atomic_add(0)

    # cut_c1: branch5 right center area
    def cut_c1_update_right_center_area(self, update_matrix_ub, n_offset, c1_offset):
        """
        cut_c1_update_right_center_area
        """
        self.instance.set_atomic_add(1)
        cut_size = self.tiling_params["ub_size_info"].get("expand")
        c1_stride = self.tiling_params["ub_factor"]
        dilation_window_h = (self.filter_h - 1) * self.rate_h + 1
        dilation_window_w = (self.filter_w - 1) * self.rate_w + 1

        w_rightcenter_steps = 1 if self.pad_right == 1 else self.pad_right // self.stride_w
        w_rightcenter_steps = \
            self.pad_right // self.stride_w + 1 if self.pad_right % self.stride_w != 0 else w_rightcenter_steps

        h_top_steps = 1 if self.pad_top == 1 else self.pad_top // self.stride_h
        h_top_steps = self.pad_top // self.stride_h + 1 if self.pad_top % self.stride_h != 0 else h_top_steps

        h_bottom_steps = 1 if self.pad_bottom == 1 else self.pad_bottom // self.stride_h
        h_bottom_steps = \
            self.pad_bottom // self.stride_h + 1 if self.pad_bottom % self.stride_h != 0 else h_bottom_steps

        h_total_steps = (self.h_in + self.pad_top + self.pad_bottom - dilation_window_h) // self.stride_h + 1
        h_rightcenter_steps = h_total_steps - h_top_steps - h_bottom_steps

        w_elem_start_offset = 0
        h_elem_start_offset = 0

        w_total_step = (self.w_in + self.pad_left + self.pad_right - dilation_window_w) // self.stride_w + 1
        right_center_begin_step = h_top_steps * w_total_step + w_total_step - w_rightcenter_steps

        # calculate w & h elements
        outer_elem = (w_total_step - w_rightcenter_steps) * self.stride_w + dilation_window_w - (self.pad_left +
                                                                                                 self.w_in)
        outer_elem_no_dilation = 1 if outer_elem == 1 else outer_elem // self.rate_w
        outer_elem_no_dilation = \
            outer_elem // self.rate_w + 1 if outer_elem % self.rate_w != 0 else outer_elem_no_dilation

        w_elem_range = self.filter_w - outer_elem_no_dilation
        h_elem_range = self.filter_h
        line_strides = self.w_in // self.stride_w if self.w_in % self.stride_w == 0 else self.w_in // self.stride_w + 1

        with self.instance.for_range(0, c1_stride) as c1_cnt:
            with self.instance.for_range(0, h_rightcenter_steps) as h_step_i:
                with self.instance.for_range(0, w_rightcenter_steps) as w_step_j:
                    h_range = h_elem_range
                    w_range = w_elem_range - w_step_j * self.stride_w // self.rate_w
                    with self.instance.for_range(0, h_range) as fh_i:
                        with self.instance.for_range(0, w_range) as fw_j:
                            gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                            gm_c1_offset = (c1_offset + c1_cnt) * self.w_in * self.h_in * self.c0
                            gm_update_idx = (h_step_i * self.stride_h + fh_i * self.rate_h) * self.w_in + \
                                            (w_total_step - w_rightcenter_steps) * self.stride_w - self.pad_left + \
                                            w_step_j * self.stride_w + fw_j * self.rate_w
                            ub_update_idx = ((h_elem_start_offset + fh_i) * self.filter_w + w_elem_start_offset +
                                             fw_j) * cut_size + c1_cnt * (cut_size // c1_stride) + \
                                            (right_center_begin_step + h_step_i * line_strides + w_step_j) * self.c0
                            self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                                    update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)

        self.instance.set_atomic_add(0)

    # cut_c1: branch6 left down area
    def cut_c1_update_left_down_area(self, update_matrix_ub, n_offset, c1_offset):
        """
        cut_c1_update_left_down_area
        """
        self.instance.set_atomic_add(1)
        cut_size = self.tiling_params["ub_size_info"].get("expand")
        c1_stride = self.tiling_params["ub_factor"]
        dilation_window_h = (self.filter_h - 1) * self.rate_h + 1

        w_pad_left_steps = 1 if self.pad_left == 1 else self.pad_left // self.stride_w
        w_pad_left_steps = \
            self.pad_left // self.stride_w + 1 if self.pad_left % self.stride_w != 0 else w_pad_left_steps

        h_pad_left_steps = 1 if self.pad_bottom == 1 else self.pad_bottom // self.stride_h
        h_pad_left_steps = \
            self.pad_bottom // self.stride_h + 1 if self.pad_bottom % self.stride_h != 0 else h_pad_left_steps

        h_total_step = (self.pad_top + self.h_in + self.pad_bottom - dilation_window_h) // self.stride_h + 1
        left_down_begin_step = h_total_step - h_pad_left_steps

        h_elem_start_offset = 0
        w_elem_start_offset = 1 if self.pad_left == 1 else self.pad_left // self.rate_w
        w_elem_start_offset = \
            self.pad_left // self.rate_w + 1 if self.pad_top % self.rate_w != 0 else w_elem_start_offset

        # calculate w & h elements
        h_outer_elem = (h_total_step - h_pad_left_steps) * self.stride_h + dilation_window_h - (self.pad_top +
                                                                                                self.h_in)
        h_outer_elem_no_dilation = 1 if h_outer_elem == 1 else h_outer_elem // self.rate_h
        h_outer_elem_no_dilation = \
            h_outer_elem // self.rate_h + 1 if h_outer_elem % self.rate_h != 0 else h_outer_elem_no_dilation
        w_elem_range = self.filter_w - w_elem_start_offset
        h_elem_range = self.filter_h - h_outer_elem_no_dilation

        line_strides = self.w_in // self.stride_w if self.w_in % self.stride_w == 0 else self.w_in // self.stride_w + 1

        with self.instance.for_range(0, c1_stride) as c1_cnt:
            with self.instance.for_range(0, h_pad_left_steps) as h_step_i:
                with self.instance.for_range(0, w_pad_left_steps) as w_step_j:
                    h_range = h_elem_range - h_step_i * self.stride_h // self.rate_h
                    w_range = w_elem_range + w_step_j * self.stride_w // self.rate_w
                    with self.instance.for_range(0, h_range) as fh_i:
                        with self.instance.for_range(0, w_range) as fw_j:
                            gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                            gm_c1_offset = (c1_offset + c1_cnt) * self.w_in * self.h_in * self.c0
                            gm_update_idx = (left_down_begin_step * self.stride_h - self.pad_top + h_step_i *
                                             self.stride_h + fh_i * self.rate_h) * self.w_in + fw_j * self.rate_w
                            ub_update_idx = ((h_elem_start_offset + fh_i) * self.filter_w + w_elem_start_offset -
                                             w_step_j * self.stride_w + fw_j) * cut_size + c1_cnt * \
                                            (cut_size // c1_stride) + \
                                            ((left_down_begin_step + h_step_i) * line_strides + w_step_j) * self.c0
                            self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                                    update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)
        self.instance.set_atomic_add(0)

    # cut_c1: branch7 right down area
    def cut_c1_update_right_down_area(self, update_matrix_ub, n_offset, c1_offset):
        """
        cut_c1_update_right_down_area
        """
        self.instance.set_atomic_add(1)
        cut_size = self.tiling_params["ub_size_info"].get("expand")
        c1_stride = self.tiling_params["ub_factor"]

        dilation_window_h = (self.filter_h - 1) * self.rate_h + 1
        dilation_window_w = (self.filter_w - 1) * self.rate_w + 1

        h_down_right_steps = 1 if self.pad_bottom == 1 else self.pad_bottom // self.stride_h
        h_down_right_steps = \
            self.pad_bottom // self.stride_h + 1 if self.pad_bottom % self.stride_h != 0 else h_down_right_steps

        w_down_right_steps = 1 if self.pad_right == 1 else self.pad_right // self.stride_w
        w_down_right_steps = \
            self.pad_right // self.stride_w + 1 if self.pad_right % self.stride_w != 0 else w_down_right_steps

        w_elem_start_offset = 0
        h_elem_start_offset = 0

        h_total_step = (self.pad_top + self.h_in + self.pad_bottom - dilation_window_h) // self.stride_h + 1
        w_total_step = (self.pad_left + self.w_in + self.pad_right - dilation_window_w) // self.stride_w + 1

        # calculate w & h elements
        w_outer_elem = (w_total_step - w_down_right_steps) * self.stride_w + dilation_window_w - (self.pad_left +
                                                                                                  self.w_in)
        w_outer_elem_no_dilation = 1 if w_outer_elem == 1 else w_outer_elem // self.rate_w
        w_outer_elem_no_dilation = \
            w_outer_elem // self.rate_w + 1 if w_outer_elem % self.rate_w != 0 else w_outer_elem_no_dilation

        h_outer_elem = (h_total_step - h_down_right_steps) * self.stride_h + dilation_window_h - (self.pad_top +
                                                                                                  self.h_in)
        h_outer_elem_no_dilation = 1 if h_outer_elem == 1 else h_outer_elem // self.rate_h
        h_outer_elem_no_dilation = \
            h_outer_elem // self.rate_h + 1 if h_outer_elem % self.rate_h else h_outer_elem_no_dilation

        w_elem_range = self.filter_w - w_outer_elem_no_dilation
        h_elem_range = self.filter_h - h_outer_elem_no_dilation

        line_strides = self.w_in // self.stride_w if self.w_in % self.stride_w == 0 else self.w_in // self.stride_w + 1
        right_down_begin_step = (h_total_step - h_down_right_steps) * line_strides + \
                                 w_total_step - w_down_right_steps

        with self.instance.for_range(0, c1_stride) as c1_cnt:
            with self.instance.for_range(0, h_down_right_steps) as h_step_i:
                with self.instance.for_range(0, w_down_right_steps) as w_step_j:
                    h_range = h_elem_range - h_step_i * self.stride_h // self.rate_h
                    w_range = w_elem_range - w_step_j * self.stride_w // self.rate_w
                    with self.instance.for_range(0, h_range) as fh_i:
                        with self.instance.for_range(0, w_range) as fw_j:
                            gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                            gm_c1_offset = (c1_offset + c1_cnt) * self.w_in * self.h_in * self.c0
                            gm_update_idx = ((h_total_step - h_down_right_steps) * self.stride_h - self.pad_top +
                                             h_step_i * self.stride_h + fh_i * self.rate_h) * self.w_in + \
                                            (w_total_step - w_down_right_steps) * self.stride_w - self.pad_left + \
                                            w_step_j * self.stride_w + fw_j * self.rate_w
                            ub_update_idx = ((h_elem_start_offset + fh_i) * self.filter_w + w_elem_start_offset +
                                             fw_j) * cut_size + c1_cnt * (cut_size // c1_stride) + \
                                            (right_down_begin_step + h_step_i * line_strides + w_step_j) * self.c0
                            self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                                    update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)
        self.instance.set_atomic_add(0)

    # cut_c1: branch8 center down area
    def cut_c1_update_center_down_area(self, update_matrix_ub, n_offset, c1_offset):
        """
        cut_c1_update_center_down_area
        """
        self.instance.set_atomic_add(1)
        cut_size = self.tiling_params["ub_size_info"].get("expand")
        c1_stride = self.tiling_params["ub_factor"]

        dilation_window_h = (self.filter_h - 1) * self.rate_h + 1
        dilation_window_w = (self.filter_w - 1) * self.rate_w + 1

        h_down_center_steps = 1 if self.pad_bottom == 1 else self.pad_bottom // self.stride_h
        h_down_center_steps = \
            self.pad_bottom // self.stride_h + 1 if self.pad_bottom % self.stride_h != 0 else h_down_center_steps

        w_left_pad_step = 1 if self.pad_left == 1 else self.pad_left // self.stride_w
        w_left_pad_step = self.pad_left // self.stride_w + 1 if self.pad_left % self.stride_w != 0 else w_left_pad_step

        w_right_pad_step = 1 if self.pad_right == 1 else self.pad_right // self.stride_w
        w_right_pad_step = \
            self.pad_right // self.stride_w + 1 if self.pad_right % self.stride_w != 0 else w_right_pad_step

        w_total_step = (self.w_in + self.pad_left + self.pad_right - dilation_window_w) // self.stride_w + 1
        w_down_center_steps = w_total_step - w_left_pad_step - w_right_pad_step

        w_elem_start_offset = 0
        h_elem_start_offset = 0

        h_total_step = (self.h_in + self.pad_top + self.pad_bottom - dilation_window_h) // self.stride_h + 1

        # calculate w & h elements
        h_outer_elem = (h_total_step - h_down_center_steps) * self.stride_h + dilation_window_h - (self.pad_top +
                                                                                                   self.h_in)
        h_outer_elem_no_dilation = 1 if h_outer_elem == 1 else h_outer_elem // self.rate_h
        h_outer_elem_no_dilation = \
            h_outer_elem // self.rate_h + 1 if h_outer_elem % self.rate_h else h_outer_elem_no_dilation

        w_elem_range = self.filter_w
        h_elem_range = self.filter_h - h_outer_elem_no_dilation

        line_strides = self.w_in // self.stride_w if self.w_in % self.stride_w == 0 else self.w_in // self.stride_w + 1
        center_down_begin_step = (h_total_step - h_down_center_steps) * line_strides + w_left_pad_step

        with self.instance.for_range(0, c1_stride) as c1_cnt:
            with self.instance.for_range(0, h_down_center_steps) as h_step_i:
                with self.instance.for_range(0, w_down_center_steps) as w_step_j:
                    h_range = h_elem_range - h_step_i * self.stride_h // self.rate_h
                    w_range = w_elem_range
                    with self.instance.for_range(0, h_range) as fh_i:
                        with self.instance.for_range(0, w_range) as fw_j:
                            gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                            gm_c1_offset = (c1_offset + c1_cnt) * self.w_in * self.h_in * self.c0
                            gm_update_idx = ((h_total_step - h_down_center_steps) * self.stride_h - self.pad_top +
                                             h_step_i * self.stride_h + fh_i * self.rate_h) * self.w_in + w_step_j * \
                                            self.stride_w + fw_j * self.rate_w
                            ub_update_idx = ((h_elem_start_offset + fh_i) * self.filter_w + w_elem_start_offset +
                                             fw_j) * cut_size + c1_cnt * (cut_size // c1_stride) + \
                                            (center_down_begin_step + h_step_i * line_strides + w_step_j) * self.c0
                            self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                                    update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)
        self.instance.set_atomic_add(0)

    # cut_c1: branch9 no padding down area
    def cut_c1_update_no_padding_area(self, update_matrix_ub, n_offset, c1_offset):
        """
        cut_c1_update_no_padding_area
        """
        self.instance.set_atomic_add(1)

        cut_size = self.tiling_params["ub_size_info"].get("expand")
        c1_stride = self.tiling_params["ub_factor"]

        dilation_window_h = (self.filter_h - 1) * self.rate_h + 1
        dilation_window_w = (self.filter_w - 1) * self.rate_w + 1

        # calcualte w step range
        w_left_pad_step = 1 if self.pad_left == 1 else self.pad_left // self.stride_w
        w_left_pad_step = self.pad_left // self.stride_w + 1 if self.pad_left % self.stride_w != 0 else w_left_pad_step

        w_right_pad_step = 1 if self.pad_right == 1 else self.pad_right // self.stride_w
        w_right_pad_step = \
            self.pad_right // self.stride_w + 1 if self.pad_right % self.stride_w != 0 else w_right_pad_step

        w_total_step = (self.w_in + self.pad_left + self.pad_right - dilation_window_w) // self.stride_w + 1
        w_step_range = w_total_step - w_left_pad_step - w_right_pad_step

        # calculate h step range
        h_top_steps = 1 if self.pad_top == 1 else self.pad_top // self.stride_h
        h_top_steps = self.pad_top // self.stride_h + 1 if self.pad_top % self.stride_h != 0 else h_top_steps

        h_bottom_steps = 1 if self.pad_bottom == 1 else self.pad_bottom // self.stride_h
        h_bottom_steps = \
            self.pad_bottom // self.stride_h + 1 if self.pad_bottom % self.stride_h != 0 else h_bottom_steps

        h_total_steps = (self.h_in + self.pad_top + self.pad_bottom - dilation_window_h) // self.stride_h + 1
        h_step_range = h_total_steps - h_top_steps - h_bottom_steps

        line_strides = self.w_in // self.stride_w if self.w_in % self.stride_w == 0 else self.w_in // self.stride_w + 1
        begin_step = h_top_steps * line_strides + w_left_pad_step

        with self.instance.for_range(0, c1_stride) as c1_cnt:
            with self.instance.for_range(0, h_step_range) as h_step_i:
                with self.instance.for_range(0, w_step_range) as w_step_j:
                    with self.instance.for_range(0, self.filter_h) as fh_i:
                        with self.instance.for_range(0, self.filter_w) as fw_j:
                            gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                            gm_c1_offset = (c1_offset + c1_cnt) * self.w_in * self.h_in * self.c0
                            gm_update_idx = (h_step_i * self.stride_h + fh_i * self.rate_h) * self.w_in + \
                                             w_step_j * self.stride_w + fw_j * self.rate_w
                            ub_update_idx = (fh_i * self.filter_w + fw_j) * cut_size + \
                                            c1_cnt * (cut_size // c1_stride) + (begin_step + h_step_i * line_strides +
                                                                                w_step_j) * self.c0
                            self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                                    update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)
        self.instance.set_atomic_add(0)

    def cut_n_update_gm_pad_mode(self, update_matrix_ub, n_offset):
        """
        cut_n_update_gm_pad_mode
        """
        self.cut_n_update_left_up_area(update_matrix_ub, n_offset)

        self.cut_n_update_right_up_area(update_matrix_ub, n_offset)

        self.cut_n_update_center_up_area(update_matrix_ub, n_offset)

        self.cut_n_update_left_center_area(update_matrix_ub, n_offset)

        self.cut_n_update_right_center_area(update_matrix_ub, n_offset)

        self.cut_n_update_left_down_area(update_matrix_ub, n_offset)

        self.cut_n_update_right_down_area(update_matrix_ub, n_offset)

        self.cut_n_update_center_down_area(update_matrix_ub, n_offset)

        self.cut_n_update_no_padding_area(update_matrix_ub, n_offset)

    def cut_c1_update_gm_pad_mode(self, update_matrix_ub, n_offset, c1_offset):
        """
        cut_c1_update_gm_pad_mode
        """
        self.cut_c1_update_left_up_area(update_matrix_ub, n_offset, c1_offset)

        self.cut_c1_update_right_up_area(update_matrix_ub, n_offset, c1_offset)

        self.cut_c1_update_center_up_area(update_matrix_ub, n_offset, c1_offset)

        self.cut_c1_update_left_center_area(update_matrix_ub, n_offset, c1_offset)

        self.cut_c1_update_right_center_area(update_matrix_ub, n_offset, c1_offset)

        self.cut_c1_update_left_down_area(update_matrix_ub, n_offset, c1_offset)

        self.cut_c1_update_right_down_area(update_matrix_ub, n_offset, c1_offset)

        self.cut_c1_update_center_down_area(update_matrix_ub, n_offset, c1_offset)

        self.cut_c1_update_no_padding_area(update_matrix_ub, n_offset, c1_offset)

    def cut_n_update_valid_mode(self, update_matrix_ub, n_offset):
        """
        cut_n_update_valid_mode
        """
        self.instance.set_atomic_add(1)
        cut_size = self.tiling_params["ub_size_info"].get("expand")
        c1_stride = self.c1

        dilation_window_h = (self.filter_h - 1) * self.rate_h + 1
        dilation_window_w = (self.filter_w - 1) * self.rate_w + 1
        w_step_range = (self.w_in - dilation_window_w) // self.stride_w + 1
        h_step_range = (self.h_in - dilation_window_h) // self.stride_h + 1

        with self.instance.for_range(0, c1_stride) as c1_cnt:
            with self.instance.for_range(0, h_step_range) as h_step_i:
                with self.instance.for_range(0, w_step_range) as w_step_j:
                    with self.instance.for_range(0, self.filter_h) as fh_i:
                        with self.instance.for_range(0, self.filter_w) as fw_j:
                            gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                            gm_c1_offset = c1_cnt * self.w_in * self.h_in * self.c0
                            ub_c1_offset = c1_cnt * (cut_size // c1_stride)
                            gm_update_idx = (h_step_i * self.stride_h + fh_i * self.rate_h) * self.w_in + \
                                             w_step_j * self.stride_w + fw_j * self.rate_w
                            ub_update_idx = (fh_i * self.filter_w + fw_j) * cut_size + \
                                            (h_step_i * w_step_range + w_step_j) * self.c0
                            self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                                    update_matrix_ub[ub_c1_offset + ub_update_idx], 0, 1, 2, 0, 0)
        self.instance.set_atomic_add(0)

    def cut_c1_update_valid_mode(self, update_matrix_ub, n_offset, c1_offset):
        """
        cut_c1_update_valid_mode
        """
        self.instance.set_atomic_add(1)
        cut_size = self.tiling_params["ub_size_info"].get("expand")
        c1_stride = self.tiling_params["ub_factor"]

        dilation_window_h = (self.filter_h - 1) * self.rate_h + 1
        dilation_window_w = (self.filter_w - 1) * self.rate_w + 1
        w_step_range = (self.w_in - dilation_window_w) // self.stride_w + 1
        h_step_range = (self.h_in - dilation_window_h) // self.stride_h + 1

        with self.instance.for_range(0, c1_stride) as c1_cnt:
            with self.instance.for_range(0, h_step_range) as h_step_i:
                with self.instance.for_range(0, w_step_range) as w_step_j:
                    with self.instance.for_range(0, self.filter_h) as fh_i:
                        with self.instance.for_range(0, self.filter_w) as fw_j:
                            gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                            gm_c1_offset = (c1_offset + c1_cnt) * self.w_in * self.h_in * self.c0
                            gm_update_idx = (h_step_i * self.stride_h + fh_i * self.rate_h) * self.w_in + \
                                             w_step_j * self.stride_w + fw_j * self.rate_w
                            ub_update_idx = (fh_i * self.filter_w + fw_j) * cut_size + c1_cnt * \
                                            (cut_size // c1_stride) + (h_step_i * w_step_range + w_step_j) * self.c0
                            self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                                    update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)

        self.instance.set_atomic_add(0)

    def cut_n(self, ub_list, offset_list, n_num, out_offset):
        """
        tiling cut n scenario
        """
        expand_ub, x_ub, filter_ub, out_backprop_ub, mask_ub, update_matrix_ub, max_data_ub, sel_zero_ub = ub_list
        n_offset = offset_list[0]  # factor list of every dim
        n_size = self.c1 * self.out_backprop_h * self.out_backprop_w * self.c0
        c1_size = self.out_backprop_h * self.out_backprop_w * self.c0
        h_size = self.out_backprop_w * self.c0

        ub_offset = self.pad_left * self.c0
        x_index = n_offset * self.x_offset_list[0]
        w_in = self.pad_w
        self.move_data(x_ub, ub_offset, x_index, n_num * self.c1 * self.h_in)
        self.move_out_backprop_data(n_num * self.c1 * self.out_backprop_h * self.out_backprop_w * self.c0,
                                    out_backprop_ub, out_offset)

        with self.instance.for_range(0, self.filter_h) as fh_i:
            with self.instance.for_range(0, self.filter_w) as fw_i:
                with self.instance.for_range(0, n_num * self.c1 * self.out_backprop_h) as id:
                    n_i = id // (self.c1 * self.out_backprop_h)
                    c1_i = id % (self.c1 * self.out_backprop_h) // self.out_backprop_h
                    ho_i = id % (self.c1 * self.out_backprop_h) % self.out_backprop_h
                    h_in = ho_i * self.stride_h - self.pad_top + fh_i * self.rate_h

                    expand_start = n_i * n_size + c1_i * c1_size + ho_i * h_size
                    x_start = n_i * self.c1 * self.h_in * w_in * self.c0 + \
                              c1_i * self.h_in * w_in * self.c0 + \
                              h_in * w_in * self.c0 + \
                              fw_i * self.rate_w * self.c0
                    if not self.tiling_params["is_filter_all"]:
                        gm_index = c1_i * self.filter_offset_list[1] + \
                                   fh_i * self.filter_offset_list[2] + \
                                   fw_i * self.filter_offset_list[3]
                        self.move_filter_data(self.filter_offset_list[3], filter_ub, gm_index)
                        filter_start = 0
                    else:
                        filter_start = c1_i * self.filter_offset_list[1] + \
                                       fh_i * self.filter_offset_list[2] + \
                                       fw_i * self.filter_offset_list[3]
                    self.expand_row([expand_ub, x_ub, filter_ub], h_in, [expand_start, x_start, filter_start], fw_i)

                ub_list = [mask_ub, expand_ub, max_data_ub]
                mask_ub, max_data_ub = self.search_max_position(ub_list, fh_i, fw_i)

        ub_list = [mask_ub, sel_zero_ub, out_backprop_ub, update_matrix_ub]
        update_matrix_ub = self.update_max_pos_n(ub_list)

        if self.input_params["padding_mode"] == "SAME":
            self.cut_n_update_gm_pad_mode(update_matrix_ub, n_offset)
        else:
            self.cut_n_update_valid_mode(update_matrix_ub, n_offset)

    def cut_c1(self, ub_list, offset_list, c1_num, out_offset):
        """
        tiling cut c1 scenario
        """
        expand_ub, x_ub, filter_ub, out_backprop_ub, mask_ub, update_matrix_ub, max_data_ub, sel_zero_ub = ub_list
        n_offset, c1_offset = offset_list
        c1_size = self.out_backprop_h * self.out_backprop_w * self.c0
        h_size = self.out_backprop_w * self.c0

        ub_offset = self.pad_left * self.c0
        x_index = n_offset * self.x_offset_list[0] + c1_offset * self.x_offset_list[1]
        w_in = self.pad_w
        self.move_data(x_ub, ub_offset, x_index, c1_num * self.h_in)
        self.move_out_backprop_data(c1_num * self.out_backprop_h * self.out_backprop_w * self.c0, out_backprop_ub,
                                    out_offset)

        with self.instance.for_range(0, self.filter_h) as fh_i:
            with self.instance.for_range(0, self.filter_w) as fw_i:
                with self.instance.for_range(0, c1_num * self.out_backprop_h) as id:
                    c1_i = id // self.out_backprop_h
                    ho_i = id % self.out_backprop_h
                    h_in = ho_i * self.stride_h - self.pad_top + fh_i * self.rate_h

                    expand_start = c1_i * c1_size + ho_i * h_size
                    x_start = c1_i * self.h_in * w_in * self.c0 + \
                              h_in * w_in * self.c0 + \
                              fw_i * self.rate_w * self.c0

                    if not self.tiling_params["is_filter_all"]:
                        gm_index = ((c1_offset + c1_i) % self.c1) * self.filter_offset_list[1] + \
                                   fh_i * self.filter_offset_list[2] + \
                                   fw_i * self.filter_offset_list[3]
                        self.move_filter_data(self.filter_offset_list[3], filter_ub, gm_index)
                        filter_start = 0
                    else:
                        filter_start = ((c1_offset + c1_i) % self.c1) * self.filter_offset_list[1] + \
                                       fh_i * self.filter_offset_list[2] + \
                                       fw_i * self.filter_offset_list[3]
                    self.expand_row([expand_ub, x_ub, filter_ub], h_in, [expand_start, x_start, filter_start], fw_i)
                ub_list = [mask_ub, expand_ub, max_data_ub]
                mask_ub, max_data_ub = self.search_max_position(ub_list, fh_i, fw_i)

        ub_list = [mask_ub, sel_zero_ub, out_backprop_ub, update_matrix_ub]
        update_matrix_ub = self.update_max_pos_c1(ub_list)

        if self.input_params["padding_mode"] == "SAME":
            self.cut_c1_update_gm_pad_mode(update_matrix_ub, n_offset, c1_offset)
        else:
            self.cut_c1_update_valid_mode(update_matrix_ub, n_offset, c1_offset)

    def search_max_position(self, ub_list, h_step, w_step):
        """
        search_max_position
        """
        cut_size = self.tiling_params["ub_size_info"].get("expand")
        one_cnt = 64
        mask_size = (cut_size // self.c0 // 4 + 1) * 4
        mask_align_pad_size = mask_size // 4 * 32

        pad_size = (cut_size // 64 + 1) * 64
        repeat = pad_size // one_cnt
        loop_repeat = repeat // 255
        loop_remainder = repeat % 255

        mask_ub, expand_ub, max_data_ub = ub_list
        cur_step = h_step * self.filter_w + w_step

        ge_cmp_ret = self.instance.Tensor("uint16", (4,), name="gt_cmp_ret", scope=tik.scope_ubuf)
        nt_ge_cmp_ret = self.instance.Tensor("uint16", (4,), name="nt_gt_cmp_ret", scope=tik.scope_ubuf)

        expand_ub_padding = self.instance.Tensor(self.x_dtype, (pad_size // self.c0, self.c0),
                                                 name="expand_ub_padding",
                                                 scope=tbe_platform.scope_ubuf)
        self.vector_dup(0, expand_ub_padding, pad_size, 0)

        if loop_repeat > 0:
            self.instance.data_move(expand_ub_padding, expand_ub, 0, 1, cut_size // self.y_block_size, 0, 0)

            with self.instance.for_range(0, loop_repeat) as l_i:
                offset_repeat = l_i * one_cnt * 255
                mask_offset = l_i * 255 * (one_cnt // 16 // 4 * 32)

                with self.instance.if_scope(cur_step == 0):
                    self.instance.data_move(max_data_ub, expand_ub, 0, 1, cut_size // self.y_block_size, 0, 0)

                    with self.instance.for_range(0, 255) as i:
                        self.instance.vcmpv_ge(mask_ub[mask_offset + 32 * i], max_data_ub[offset_repeat + i * one_cnt],
                                               max_data_ub[offset_repeat + i * one_cnt], 1, 1, 1, 8, 8)

                with self.instance.else_scope():
                    with self.instance.for_range(0, 255) as i:
                        self.instance.vcmpv_ge(ge_cmp_ret, max_data_ub[offset_repeat + i * one_cnt],
                                               expand_ub_padding[offset_repeat + i * one_cnt], 1, 1, 1, 8, 8)
                        self.instance.vcmpv_lt(nt_ge_cmp_ret, max_data_ub[offset_repeat + i * one_cnt],
                                               expand_ub_padding[offset_repeat + i * one_cnt], 1, 1, 1, 8, 8)
                        self.instance.vec_max(64, max_data_ub[offset_repeat + i * one_cnt],
                                              max_data_ub[offset_repeat + i * one_cnt],
                                              expand_ub_padding[offset_repeat + i * one_cnt], 1, 8, 8, 8)

                        with self.instance.for_range(0, cur_step) as j:
                            self.instance.vec_and(4, mask_ub[j * mask_align_pad_size + mask_offset + i * 32],
                                                  mask_ub[j * mask_align_pad_size + mask_offset + i * 32], ge_cmp_ret,
                                                  1, 8, 8, 8)
                        self.instance.data_move(mask_ub[cur_step * mask_align_pad_size + mask_offset + i * 32],
                                                nt_ge_cmp_ret, 0, 1, 1, 0, 0)

        if loop_remainder > 0:
            offset_remainder = loop_repeat * one_cnt * 255
            repeats = (pad_size - offset_remainder) // one_cnt
            self.instance.data_move(expand_ub_padding, expand_ub, 0, 1, cut_size // self.y_block_size, 0, 0)

            with self.instance.if_scope(cur_step == 0):
                self.instance.data_move(max_data_ub, expand_ub[offset_remainder], 0, 1,
                                        (cut_size - offset_remainder) // self.y_block_size, 0, 0)

                with self.instance.for_range(0, repeats) as i:
                    self.instance.vcmpv_ge(mask_ub[i * 32], max_data_ub[i * one_cnt], max_data_ub[i * one_cnt], 1, 1, 1,
                                           8, 8)

            with self.instance.else_scope():
                with self.instance.for_range(0, repeats) as i:
                    self.instance.vcmpv_ge(ge_cmp_ret, max_data_ub[i * one_cnt], expand_ub_padding[i * one_cnt], 1, 1,
                                           1, 8, 8)
                    self.instance.vcmpv_lt(nt_ge_cmp_ret, max_data_ub[i * one_cnt], expand_ub_padding[i * one_cnt], 1,
                                           1, 1, 8, 8)
                    self.instance.vec_max(64, max_data_ub[i * one_cnt], max_data_ub[i * one_cnt],
                                          expand_ub_padding[i * one_cnt], 1, 8, 8, 8)

                    with self.instance.for_range(0, cur_step) as j:
                        self.instance.vec_and(4, mask_ub[j * mask_align_pad_size + i * 32],
                                              mask_ub[j * mask_align_pad_size + i * 32], ge_cmp_ret, 1, 8, 8, 8)
                    self.instance.data_move(mask_ub[cur_step * mask_align_pad_size + i * 32], nt_ge_cmp_ret, 0, 1, 1, 0,
                                            0)

        return mask_ub, max_data_ub

    def update_max_pos_n(self, ub_list):
        """
        update_max_pos_n
        """
        mask_ub, sel_zero_ub, out_backprop_ub, update_matrix = ub_list
        cut_size = self.tiling_params["ub_size_info"].get("expand")
        pad_size = (cut_size // 64 + 1) * 64

        one_cnt = 64
        repeats = pad_size // one_cnt

        mask_size = (cut_size // self.c0 // 4 + 1) * 4
        mask_align_pad_size = mask_size // 4 * 32

        out_backprop_ub_padding = self.instance.Tensor(self.x_dtype, (pad_size // self.c0, self.c0),
                                                       name="out_backprop_ub_padding",
                                                       scope=tbe_platform.scope_ubuf)
        self.vector_dup(0, out_backprop_ub_padding, pad_size, 0)
        self.instance.data_move(out_backprop_ub_padding, out_backprop_ub, 0, 1, cut_size // self.y_block_size, 0, 0)

        # generate update matrix
        with self.instance.for_range(0, self.filter_w * self.filter_h) as i:
            with self.instance.for_range(0, repeats) as j:
                self.instance.vec_sel(64, 0, update_matrix[i * (cut_size) + j * one_cnt],
                                      mask_ub[i * mask_align_pad_size + j * 32], out_backprop_ub_padding[j * one_cnt],
                                      sel_zero_ub, 1, 8, 8, 8)

        return update_matrix

    def update_max_pos_c1(self, ub_list):
        """
        update_max_pos_c1
        """
        mask_ub, sel_zero_ub, out_backprop_ub, update_matrix = ub_list
        cut_size = self.tiling_params["ub_size_info"].get("expand")
        pad_size = (cut_size // 64 + 1) * 64

        one_cnt = 64
        repeats = pad_size // one_cnt

        mask_size = (cut_size // self.c0 // 4 + 1) * 4
        mask_align_pad_size = mask_size // 4 * 32

        out_backprop_ub_padding = self.instance.Tensor(self.x_dtype, (pad_size // self.c0, self.c0),
                                                       name="out_backprop_ub_padding",
                                                       scope=tbe_platform.scope_ubuf)
        self.vector_dup(0, out_backprop_ub_padding, pad_size, 0)
        self.instance.data_move(out_backprop_ub_padding, out_backprop_ub, 0, 1, cut_size // self.y_block_size, 0, 0)

        # generate update matrix
        with self.instance.for_range(0, self.filter_w * self.filter_h) as i:
            with self.instance.for_range(0, repeats) as j:
                self.instance.vec_sel(64, 0, update_matrix[i * cut_size + j * one_cnt],
                                      mask_ub[i * mask_align_pad_size + j * 32], out_backprop_ub_padding[j * one_cnt],
                                      sel_zero_ub, 1, 8, 8, 8)

        return update_matrix

    def update_max_pos_h(self, ub_list, h_num):
        """
        update_max_pos_h
        """
        mask_ub, sel_zero_ub, out_backprop_ub, update_matrix = ub_list
        cut_size = self.tiling_params["ub_size_info"].get("expand")
        pad_size = (cut_size // 64 + 1) * 64

        one_cnt = 64
        repeats = pad_size // one_cnt

        mask_size = (cut_size // self.c0 // 4 + 1) * 4
        mask_align_pad_size = mask_size // 4 * 32

        out_backprop_ub_padding = self.instance.Tensor(self.x_dtype, (pad_size // self.c0, self.c0),
                                                       name="out_backprop_ub_padding",
                                                       scope=tbe_platform.scope_ubuf)
        self.vector_dup(0, out_backprop_ub_padding, pad_size, 0)
        self.instance.data_move(out_backprop_ub_padding, out_backprop_ub, 0, 1, cut_size // self.y_block_size, 0, 0)

        # generate update matrix
        with self.instance.for_range(0, self.filter_w * self.filter_h) as i:
            with self.instance.for_range(0, repeats) as j:
                self.instance.vec_sel(64, 0, update_matrix[i * cut_size + j * one_cnt],
                                      mask_ub[i * mask_align_pad_size + j * 32], out_backprop_ub_padding[j * one_cnt],
                                      sel_zero_ub, 1, 8, 8, 8)

        return update_matrix

    def update_max_pos_w(self, ub_list, w_num):
        """
        update_max_pos_w
        """
        mask_ub, sel_zero_ub, out_backprop_ub, update_matrix = ub_list
        cut_size = self.tiling_params["ub_size_info"].get("expand")
        pad_size = (cut_size // 64 + 1) * 64

        one_cnt = 64
        repeats = pad_size // one_cnt

        mask_size = (cut_size // self.c0 // 4 + 1) * 4
        mask_align_pad_size = mask_size // 4 * 32

        w_num_scalar = self.instance.Scalar("int32", name="w_num_scalar")
        w_num_scalar.set_as(w_num)

        out_backprop_ub_padding = self.instance.Tensor(self.x_dtype, (pad_size // self.c0, self.c0),
                                                       name="out_backprop_ub_padding",
                                                       scope=tbe_platform.scope_ubuf)
        self.vector_dup(0, out_backprop_ub_padding, pad_size, 0)
        self.instance.data_move(out_backprop_ub_padding, out_backprop_ub, 0, 1, cut_size // self.y_block_size, 0, 0)

        # generate update matrix
        with self.instance.for_range(0, self.filter_w * self.filter_h) as i:
            with self.instance.for_range(0, repeats) as j:
                self.instance.vec_sel(64, 0, update_matrix[i * cut_size + j * one_cnt],
                                      mask_ub[i * mask_align_pad_size + j * 32], out_backprop_ub_padding[j * one_cnt],
                                      sel_zero_ub, 1, 8, 8, 8)

        return update_matrix

    # cut_h: branch1 left up area
    def cut_h_update_left_up_area(self, update_matrix_ub, n_offset, c1_offset, h_pos, h_step_i):
        """
        cut_h_update_left_up_area
        """
        self.instance.set_atomic_add(1)
        if self.pad_left == 1:
            w_step_range = 1
        elif self.pad_left % self.stride_w != 0:
            w_step_range = self.pad_left // self.stride_w + 1
        else:
            w_step_range = self.pad_left // self.stride_w

        cut_size = self.tiling_params["ub_size_info"].get("expand")

        w_elem_start_offset = self.pad_left // self.rate_w
        h_elem_start_offset = self.pad_top // self.rate_h

        w_elem_range = self.filter_w - self.pad_left // self.rate_w
        h_elem_range = self.filter_h - self.pad_top // self.rate_h + h_pos // self.rate_h

        if self.w_in % self.stride_w == 0:
            line_steps = self.w_in // self.stride_w
        else:
            line_steps = self.w_in // self.stride_w + 1

        with self.instance.for_range(0, w_step_range) as w_step_i:
            h_range = h_elem_range
            w_range = w_elem_range + (w_step_i * self.stride_w) // self.rate_w
            with self.instance.for_range(0, h_range) as fh_i:
                with self.instance.for_range(0, w_range) as fw_j:
                    gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                    gm_c1_offset = c1_offset * self.w_in * self.h_in * self.c0
                    w_start_offset = w_step_i * self.stride_w % self.rate_w
                    h_start_offset = h_pos % self.rate_h
                    gm_update_idx = (h_start_offset +
                                     fh_i * self.rate_h) * self.w_in + w_start_offset + fw_j * self.rate_w
                    ub_update_idx = ((h_elem_start_offset + fh_i - h_pos // self.rate_h) * self.filter_w +
                                     w_elem_start_offset - (w_step_i * self.stride_w) // self.rate_w + fw_j) * \
                                    cut_size + (h_step_i * line_steps + w_step_i) * self.c0
                    self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                            update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)
        self.instance.set_atomic_add(0)

    # cut_h: branch2 right up area
    def cut_h_update_right_up_area(self, update_matrix_ub, n_offset, c1_offset, h_pos, h_step_i):
        """
        cut_h_update_right_up_area
        """
        self.instance.set_atomic_add(1)

        cut_size = self.tiling_params["ub_size_info"].get("expand")

        if self.pad_right == 1:
            w_right_pad_step = 1
        elif self.pad_right % self.stride_w != 0:
            w_right_pad_step = self.pad_right // self.stride_w + 1
        else:
            w_right_pad_step = self.pad_right // self.stride_w

        w_elem_start_offset = 0
        h_elem_start_offset = self.pad_top // self.rate_h

        w_elem_range = self.filter_w - 1
        h_elem_range = self.filter_h - self.pad_top // self.rate_h + h_pos // self.rate_h

        dilation_window_w = (self.filter_w - 1) * self.rate_w + 1

        w_total_step = (self.w_in + self.pad_left + self.pad_right - dilation_window_w) // self.stride_w + 1
        right_up_begin_step = w_total_step - w_right_pad_step

        if self.w_in % self.stride_w == 0:
            line_steps = self.w_in // self.stride_w
        else:
            line_steps = self.w_in // self.stride_w + 1

        with self.instance.for_range(0, w_right_pad_step) as w_step_i:
            h_range = h_elem_range
            w_range = w_elem_range - w_step_i * self.stride_w // self.rate_w
            with self.instance.for_range(0, h_range) as fh_i:
                with self.instance.for_range(0, w_range) as fw_j:
                    gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                    gm_c1_offset = c1_offset * self.w_in * self.h_in * self.c0
                    h_start_offset = h_pos % self.rate_h
                    gm_update_idx = (h_start_offset + fh_i * self.rate_h) * self.w_in + \
                                    (right_up_begin_step * self.stride_w - self.pad_left) + w_step_i * \
                                    self.stride_w + fw_j * self.rate_w
                    ub_update_idx = ((h_elem_start_offset + fh_i - h_pos // self.rate_h) * self.filter_w +
                                     w_elem_start_offset + fw_j) * cut_size + \
                                    (right_up_begin_step + h_step_i * line_steps + w_step_i) * self.c0
                    self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                            update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)
        self.instance.set_atomic_add(0)

    # cut_h: branch3 up center area
    def cut_h_update_up_center_area(self, update_matrix_ub, n_offset, c1_offset, h_pos, h_step_i):
        """
        cut_h_update_up_center_area
        """
        self.instance.set_atomic_add(1)

        cut_size = self.tiling_params["ub_size_info"].get("expand")
        dilation_window_w = (self.filter_w - 1) * self.rate_w + 1

        if self.pad_left == 1:
            w_left_pad_step = 1
        elif self.pad_left % self.stride_w != 0:
            w_left_pad_step = self.pad_left // self.stride_w + 1
        else:
            w_left_pad_step = self.pad_left // self.stride_w

        if self.pad_right == 1:
            w_right_pad_step = 1
        elif self.pad_right % self.stride_w != 0:
            w_right_pad_step = self.pad_right // self.stride_w + 1
        else:
            w_right_pad_step = self.pad_right // self.stride_w

        w_total_step = (self.w_in + self.pad_left + self.pad_right - dilation_window_w) // self.stride_w + 1
        w_step_range = w_total_step - w_left_pad_step - w_right_pad_step

        w_elem_start_offset = 0
        h_elem_start_offset = self.pad_top // self.rate_h

        w_elem_range = self.filter_w
        h_elem_range = self.filter_h - self.pad_top // self.rate_h + h_pos // self.rate_h

        center_up_begin_step = w_left_pad_step

        if self.w_in % self.stride_w == 0:
            line_steps = self.w_in // self.stride_w
        else:
            line_steps = self.w_in // self.stride_w + 1

        with self.instance.for_range(0, w_step_range) as w_step_i:
            h_range = h_elem_range
            w_range = w_elem_range
            with self.instance.for_range(0, h_range) as fh_i:
                with self.instance.for_range(0, w_range) as fw_j:
                    gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                    gm_c1_offset = c1_offset * self.w_in * self.h_in * self.c0
                    h_start_offset = h_pos % self.rate_h
                    w_start_offset = w_left_pad_step * self.stride_w - self.pad_left
                    gm_update_idx = (h_start_offset + fh_i * self.rate_h) * self.w_in + \
                                   w_start_offset + w_step_i * self.stride_w + fw_j * self.rate_w
                    ub_update_idx = ((h_elem_start_offset + fh_i - h_pos // self.rate_h) * self.filter_w +
                                     w_elem_start_offset + fw_j) * cut_size + \
                                    (center_up_begin_step + h_step_i * line_steps + w_step_i) * self.c0
                    self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                            update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)
        self.instance.set_atomic_add(0)

    # cut_h: branch4 down left area
    def cut_h_update_down_left_area(self, update_matrix_ub, n_offset, c1_offset, h_pos, h_step_i):
        """
        cut_h_update_down_left_area
        """
        self.instance.set_atomic_add(1)

        cut_size = self.tiling_params["ub_size_info"].get("expand")

        dilation_window_h = (self.filter_h - 1) * self.rate_h + 1
        if self.pad_left == 1:
            w_step_range = 1
        elif self.pad_left % self.stride_w != 0:
            w_step_range = self.pad_left // self.stride_w + 1
        else:
            w_step_range = self.pad_left // self.stride_w

        w_elem_start_offset = self.pad_left // self.rate_w
        h_elem_start_offset = 0

        w_elem_range = self.filter_w - self.pad_left // self.rate_w
        h_elem_range = self.filter_h - (h_pos - self.pad_top + dilation_window_h - self.h_in) // self.rate_h

        if self.w_in % self.stride_w == 0:
            line_steps = self.w_in // self.stride_w
        else:
            line_steps = self.w_in // self.stride_w + 1

        with self.instance.for_range(0, w_step_range) as w_step_i:
            h_range = h_elem_range
            w_range = w_elem_range + w_step_i * self.stride_w // self.rate_w
            with self.instance.for_range(0, h_range) as fh_i:
                with self.instance.for_range(0, w_range) as fw_j:
                    gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                    gm_c1_offset = c1_offset * self.w_in * self.h_in * self.c0
                    w_start_offset = w_step_i * self.stride_w % self.rate_w
                    gm_update_idx = (h_pos - self.pad_top +
                                     fh_i * self.rate_h) * self.w_in + w_start_offset + fw_j * self.rate_w
                    ub_update_idx = ((h_elem_start_offset + fh_i) * self.filter_w + w_elem_start_offset -
                                     (w_step_i * self.stride_w) // self.rate_w + fw_j) * cut_size + \
                                     (h_step_i * line_steps + w_step_i) * self.c0
                    self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                            update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)
        self.instance.set_atomic_add(0)

    # cut_h: branch5 down right area
    def cut_h_update_down_right_area(self, update_matrix_ub, n_offset, c1_offset, h_pos, h_step_i):
        """
        cut_h_update_down_right_area
        """
        self.instance.set_atomic_add(1)

        dilation_window_h = (self.filter_h - 1) * self.rate_h + 1
        dilation_window_w = (self.filter_w - 1) * self.rate_w + 1

        cut_size = self.tiling_params["ub_size_info"].get("expand")

        if self.pad_right == 1:
            w_right_pad_step = 1
        elif self.pad_right % self.stride_w != 0:
            w_right_pad_step = self.pad_right // self.stride_w + 1
        else:
            w_right_pad_step = self.pad_right // self.stride_w

        w_elem_start_offset = 0
        h_elem_start_offset = 0

        w_elem_range = self.filter_w - 1
        h_elem_range = self.filter_h - (h_pos - self.pad_top + dilation_window_h - self.h_in) // self.rate_h

        w_total_step = (self.w_in + self.pad_left + self.pad_right - dilation_window_w) // self.stride_w + 1
        right_up_begin_step = w_total_step - w_right_pad_step

        if self.w_in % self.stride_w == 0:
            line_steps = self.w_in // self.stride_w
        else:
            line_steps = self.w_in // self.stride_w + 1

        with self.instance.for_range(0, w_right_pad_step) as w_step_i:
            h_range = h_elem_range
            w_range = w_elem_range - w_step_i * self.stride_w // self.rate_w
            with self.instance.for_range(0, h_range) as fh_i:
                with self.instance.for_range(0, w_range) as fw_j:
                    gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                    gm_c1_offset = c1_offset * self.w_in * self.h_in * self.c0
                    gm_update_idx = (h_pos - self.pad_top + fh_i * self.rate_h) * self.w_in + \
                                    (right_up_begin_step * self.stride_w - self.pad_left) + w_step_i * \
                                    self.stride_w + fw_j * self.rate_w
                    ub_update_idx = ((h_elem_start_offset + fh_i) * self.filter_w + w_elem_start_offset + fw_j) * \
                                    cut_size + (h_step_i * line_steps + right_up_begin_step + w_step_i) * self.c0

                    self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                            update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)

        self.instance.set_atomic_add(0)

    # cut_h: branch6 down center area
    def cut_h_update_down_center_area(self, update_matrix_ub, n_offset, c1_offset, h_pos, h_step_i):
        """
        cut_h_update_down_center_area
        """
        self.instance.set_atomic_add(1)
        dilation_window_h = (self.filter_h - 1) * self.rate_h + 1
        dilation_window_w = (self.filter_w - 1) * self.rate_w + 1

        cut_size = self.tiling_params["ub_size_info"].get("expand")

        if self.pad_left == 1:
            w_left_pad_step = 1
        elif self.pad_left % self.stride_w != 0:
            w_left_pad_step = self.pad_left // self.stride_w + 1
        else:
            w_left_pad_step = self.pad_left // self.stride_w

        if self.pad_right == 1:
            w_right_pad_step = 1
        elif self.pad_right % self.stride_w != 0:
            w_right_pad_step = self.pad_right // self.stride_w + 1
        else:
            w_right_pad_step = self.pad_right // self.stride_w

        w_total_step = (self.w_in + self.pad_left + self.pad_right - dilation_window_w) // self.stride_w + 1
        w_step_range = w_total_step - w_left_pad_step - w_right_pad_step

        w_elem_start_offset = 0
        h_elem_start_offset = 0

        w_elem_range = self.filter_w
        h_elem_range = self.filter_h - (h_pos - self.pad_top + dilation_window_h - self.h_in) // self.rate_h

        center_down_begin_step = w_left_pad_step

        if self.w_in % self.stride_w == 0:
            line_steps = self.w_in // self.stride_w
        else:
            line_steps = self.w_in // self.stride_w + 1

        with self.instance.for_range(0, w_step_range) as w_step_i:
            h_range = h_elem_range
            w_range = w_elem_range
            with self.instance.for_range(0, h_range) as fh_i:
                with self.instance.for_range(0, w_range) as fw_j:
                    gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                    gm_c1_offset = c1_offset * self.w_in * self.h_in * self.c0
                    w_start_offset = center_down_begin_step * self.stride_w - self.pad_left
                    gm_update_idx = ((h_pos - self.pad_top) + fh_i * self.rate_h) * self.w_in + \
                                    w_start_offset + w_step_i * self.stride_w + fw_j * self.rate_w
                    ub_update_idx = ((h_elem_start_offset + fh_i) * self.filter_w + w_elem_start_offset + fw_j) * \
                                    cut_size + (center_down_begin_step + h_step_i * line_steps + w_step_i) * self.c0
                    self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                            update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)

        self.instance.set_atomic_add(0)

    # cut_h: branch7 left center area
    def cut_h_update_left_center_area(self, update_matrix_ub, n_offset, c1_offset, h_pos, h_step_i):
        """
        cut_h_update_left_center_area
        """
        self.instance.set_atomic_add(1)

        if self.pad_left == 1:
            w_step_range = 1
        elif self.pad_left % self.stride_w != 0:
            w_step_range = self.pad_left // self.stride_w + 1
        else:
            w_step_range = self.pad_left // self.stride_w

        cut_size = self.tiling_params["ub_size_info"].get("expand")

        w_elem_range = self.filter_w - self.pad_left // self.rate_w
        h_elem_range = self.filter_h

        w_elem_start_offset = self.pad_left // self.rate_w
        h_elem_start_offset = 0

        if self.w_in % self.stride_w == 0:
            line_steps = self.w_in // self.stride_w
        else:
            line_steps = self.w_in // self.stride_w + 1

        with self.instance.for_range(0, w_step_range) as w_step_i:
            h_range = h_elem_range
            w_range = w_elem_range + w_step_i * self.stride_w // self.rate_w
            with self.instance.for_range(0, h_range) as fh_i:
                with self.instance.for_range(0, w_range) as fw_j:
                    gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                    gm_c1_offset = c1_offset * self.w_in * self.h_in * self.c0
                    w_start_offset = w_step_i * self.stride_w % self.rate_w
                    gm_update_idx = (h_pos - self.pad_top +
                                     fh_i * self.rate_h) * self.w_in + w_start_offset + fw_j * self.rate_w
                    ub_update_idx = ((h_elem_start_offset + fh_i) * self.filter_w + w_elem_start_offset + fw_j -
                                     (w_step_i * self.stride_w) // self.rate_w) * cut_size + \
                                    (h_step_i * line_steps + w_step_i) * self.c0
                    self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                            update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)
        self.instance.set_atomic_add(0)

    # cut_h: branch8 right center area
    def cut_h_update_right_center_area(self, update_matrix_ub, n_offset, c1_offset, h_pos, h_step_i):
        """
        cut_h_update_right_center_area
        """
        self.instance.set_atomic_add(1)
        dilation_window_w = (self.filter_w - 1) * self.rate_w + 1

        if self.pad_right == 1:
            w_right_pad_step = 1
        elif self.pad_right % self.stride_w != 0:
            w_right_pad_step = self.pad_right // self.stride_w + 1
        else:
            w_right_pad_step = self.pad_right // self.stride_w

        cut_size = self.tiling_params["ub_size_info"].get("expand")

        w_elem_start_offset = 0
        h_elem_start_offset = 0

        w_elem_range = self.filter_w - 1
        h_elem_range = self.filter_h

        w_total_step = (self.w_in + self.pad_left + self.pad_right - dilation_window_w) // self.stride_w + 1
        right_center_begin_step = w_total_step - w_right_pad_step

        if self.w_in % self.stride_w == 0:
            line_steps = self.w_in // self.stride_w
        else:
            line_steps = self.w_in // self.stride_w + 1

        with self.instance.for_range(0, w_right_pad_step) as w_step_i:
            h_range = h_elem_range
            w_range = w_elem_range - w_step_i * self.stride_w // self.rate_w
            with self.instance.for_range(0, h_range) as fh_i:
                with self.instance.for_range(0, w_range) as fw_j:
                    gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                    gm_c1_offset = c1_offset * self.w_in * self.h_in * self.c0
                    gm_update_idx = (h_pos - self.pad_top + fh_i * self.rate_h) * self.w_in + \
                                    (right_center_begin_step * self.stride_w - self.pad_left) + fw_j * self.rate_w + \
                                    w_step_i * self.stride_w
                    ub_update_idx = ((h_elem_start_offset + fh_i) * self.filter_w + w_elem_start_offset + fw_j) * \
                                    cut_size + (right_center_begin_step + h_step_i * line_steps + w_step_i) * self.c0
                    self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                            update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)
        self.instance.set_atomic_add(0)

    # cut_h: branch9 no padding area
    def cut_h_update_no_padding_area(self, update_matrix_ub, n_offset, c1_offset, h_pos, h_step_i):
        """
        cut_h_update_no_padding_area
        """
        self.instance.set_atomic_add(1)
        cut_size = self.tiling_params["ub_size_info"].get("expand")
        dilation_window_w = (self.filter_w - 1) * self.rate_w + 1

        if self.pad_left == 1:
            w_left_pad_step = 1
        elif self.pad_left % self.stride_w != 0:
            w_left_pad_step = self.pad_left // self.stride_w + 1
        else:
            w_left_pad_step = self.pad_left // self.stride_w

        if self.pad_right == 1:
            w_right_pad_step = 1
        elif self.pad_right % self.stride_w != 0:
            w_right_pad_step = self.pad_right // self.stride_w + 1
        else:
            w_right_pad_step = self.pad_right // self.stride_w

        w_total_step = (self.w_in + self.pad_left + self.pad_right - dilation_window_w) // self.stride_w + 1
        w_step_range = w_total_step - w_left_pad_step - w_right_pad_step

        begin_step = w_left_pad_step

        if self.w_in % self.stride_w == 0:
            line_steps = self.w_in // self.stride_w
        else:
            line_steps = self.w_in // self.stride_w + 1

        with self.instance.for_range(0, w_step_range) as w_step_i:
            with self.instance.for_range(0, self.filter_h) as fh_i:
                with self.instance.for_range(0, self.filter_w) as fw_j:
                    gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                    gm_c1_offset = c1_offset * self.w_in * self.h_in * self.c0
                    w_start_offset = w_left_pad_step * self.stride_w - self.pad_left
                    gm_update_idx = (h_pos - self.pad_top + fh_i * self.rate_h) * self.w_in + \
                                     w_start_offset + w_step_i * self.stride_w + fw_j * self.rate_w
                    ub_update_idx = (fh_i * self.filter_w + fw_j) * cut_size + \
                                    (begin_step + h_step_i * line_steps + w_step_i) * self.c0
                    self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                            update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)
        self.instance.set_atomic_add(0)

    def cut_h_update_valid_mode(self, update_matrix_ub, n_offset, c1_offset, h_pos, h_step_i):
        """
        cut_h_update_valid_mode
        """
        self.instance.set_atomic_add(1)
        dilation_window_w = (self.filter_w - 1) * self.rate_w + 1
        w_step_range = (self.w_in - dilation_window_w) // self.stride_w + 1
        cut_size = self.tiling_params["ub_size_info"].get("expand")

        with self.instance.for_range(0, w_step_range) as w_step_i:
            with self.instance.for_range(0, self.filter_h) as fh_i:
                with self.instance.for_range(0, self.filter_w) as fw_j:
                    gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                    gm_c1_offset = c1_offset * self.w_in * self.h_in * self.c0
                    gm_update_idx = (h_pos + fh_i * self.rate_h) * self.w_in + \
                                    w_step_i * self.stride_w + fw_j * self.rate_w
                    ub_update_idx = (fh_i * self.filter_w + fw_j) * cut_size + \
                                    (h_step_i * w_step_range + w_step_i) * self.c0
                    self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                            update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)
        self.instance.set_atomic_add(0)

    def cut_h_update_gm_pad_mode(self, update_matrix_ub, n_offset, c1_offset, h_offset, h_num):
        """
        cut_h_update_gm_pad_mode
        """
        dilation_window_h = (self.filter_h - 1) * self.rate_h + 1

        h_pos = self.instance.Scalar("int32", name="h_pos")
        h_pos_up_boundary = self.instance.Scalar("int32", name="h_pos_up_boundary")

        if self.pad_bottom == 1:
            down_area_step = 1
        elif self.pad_bottom % self.stride_h != 0:
            down_area_step = self.pad_bottom // self.stride_h + 1
        else:
            down_area_step = self.pad_bottom // self.stride_h

        h_total_step = (self.h_in + self.pad_top + self.pad_bottom - dilation_window_h) // self.stride_h + 1
        bottom_begin_step = h_total_step - down_area_step
        bottom_area_start_pos = bottom_begin_step * self.stride_h

        with self.instance.for_range(0, h_num) as h_step_i:
            h_pos.set_as((h_offset + h_step_i) * self.stride_h)
            h_pos_up_boundary.set_as((h_offset + h_step_i) * self.stride_h - self.pad_top)

            with self.instance.if_scope(h_pos_up_boundary < 0):
                # cut_h: branch1 left up area
                self.cut_h_update_left_up_area(update_matrix_ub, n_offset, c1_offset, h_pos, h_step_i)

                # cut_h: branch2 right up area
                self.cut_h_update_right_up_area(update_matrix_ub, n_offset, c1_offset, h_pos, h_step_i)

                # cut_h: branch3 up center area
                self.cut_h_update_up_center_area(update_matrix_ub, n_offset, c1_offset, h_pos, h_step_i)

            with self.instance.if_scope(h_pos >= bottom_area_start_pos):
                # cut_h: branch4 down left area
                self.cut_h_update_down_left_area(update_matrix_ub, n_offset, c1_offset, h_pos, h_step_i)

                # cut_h: branch5 down right area
                self.cut_h_update_down_right_area(update_matrix_ub, n_offset, c1_offset, h_pos, h_step_i)

                # cut_h: branch6 down center area
                self.cut_h_update_down_center_area(update_matrix_ub, n_offset, c1_offset, h_pos, h_step_i)

            with self.instance.if_scope(tik.all(h_pos_up_boundary >= 0, h_pos < bottom_area_start_pos)):
                # cut_h: branch7 left center area
                self.cut_h_update_left_center_area(update_matrix_ub, n_offset, c1_offset, h_pos, h_step_i)

                # cut_h: branch8 right center area
                self.cut_h_update_right_center_area(update_matrix_ub, n_offset, c1_offset, h_pos, h_step_i)

                # cut_h: branch9 no padding area
                self.cut_h_update_no_padding_area(update_matrix_ub, n_offset, c1_offset, h_pos, h_step_i)

    # cut_w: branch1 left and center up area
    def cut_w_update_left_and_center_up_area(self, update_matrix_ub, n_offset, c1_offset, h_pos, w_offset, w_num):
        """
        cut_w_update_left_and_center_up_area
        """
        self.instance.set_atomic_add(1)

        # (1) update left up area
        w_left_pad_steps = 1 if self.pad_left == 1 else self.pad_left // self.stride_w
        w_left_pad_steps = \
            self.pad_left // self.stride_w + 1 if self.pad_left % self.stride_w != 0 else w_left_pad_steps

        cut_size = self.tiling_params["ub_size_info"].get("expand")

        w_elem_start_offset = self.pad_left // self.rate_w - w_offset * self.stride_w // self.rate_w
        h_elem_start_offset = self.pad_top // self.rate_h

        w_elem_range = self.filter_w - w_elem_start_offset
        h_elem_range = self.filter_h - self.pad_top // self.rate_h + h_pos // self.rate_h

        with self.instance.for_range(0, w_left_pad_steps) as w_step_i:
            h_range = h_elem_range
            w_range = w_elem_range + (w_step_i * self.stride_w) // self.rate_w
            with self.instance.for_range(0, h_range) as fh_i:
                with self.instance.for_range(0, w_range) as fw_j:
                    gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                    gm_c1_offset = c1_offset * self.w_in * self.h_in * self.c0
                    w_start_offset = w_offset * self.stride_w + w_step_i * self.stride_w % self.rate_w
                    h_start_offset = h_pos % self.rate_h
                    gm_update_idx = (h_start_offset +
                                     fh_i * self.rate_h) * self.w_in + w_start_offset + fw_j * self.rate_w
                    ub_update_idx = ((h_elem_start_offset + fh_i - h_pos // self.rate_h) * self.filter_w +
                                     w_elem_start_offset - (w_step_i * self.stride_w) // self.rate_w + fw_j) * \
                                    cut_size + w_step_i * self.c0
                    self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                            update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)

        # (2) update center up area
        left_center_up_range = self.instance.Scalar("int32", name="left_center_up_range")
        left_center_up_range.set_as(w_num - w_left_pad_steps)

        w_center_up_elem_start_offset = 0
        h_center_up_elem_start_offset = self.pad_top // self.rate_h

        w_center_up_elem_range = self.filter_w
        h_center_up_elem_range = self.filter_h - self.pad_top // self.rate_h + h_pos // self.rate_h

        center_up_begin_step = w_left_pad_steps

        with self.instance.if_scope(left_center_up_range > 0):
            with self.instance.for_range(0, left_center_up_range) as w_step_i:
                with self.instance.for_range(0, h_center_up_elem_range) as fh_i:
                    with self.instance.for_range(0, w_center_up_elem_range) as fw_j:
                        gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                        gm_c1_offset = c1_offset * self.w_in * self.h_in * self.c0
                        h_center_start_offset = h_pos % self.rate_h
                        w_center_start_offset = w_left_pad_steps * self.stride_w - self.pad_left
                        gm_update_idx = (h_center_start_offset + fh_i * self.rate_h) * self.w_in + \
                                         w_center_start_offset + w_step_i * self.stride_w + fw_j * self.rate_w
                        ub_update_idx = ((h_center_up_elem_start_offset + fh_i - h_pos // self.rate_h) *
                                         self.filter_w + w_center_up_elem_start_offset + fw_j) * cut_size + \
                                        (center_up_begin_step + w_step_i) * self.c0
                        self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                                update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)

        self.instance.set_atomic_add(0)

    # cut_w: branch2 right and center up area
    def cut_w_update_right_and_center_up_area(self, update_matrix_ub, n_offset, c1_offset, h_pos, w_offset, w_num):
        """
        cut_w_update_right_and_center_up_area
        """
        self.instance.set_atomic_add(1)

        # (1) update right up  area
        w_right_pad_steps = 1 if self.pad_right == 1 else self.pad_right // self.stride_w
        w_right_pad_steps = \
            self.pad_right // self.stride_w + 1 if self.pad_right % self.stride_w != 0 else w_right_pad_steps

        cut_size = self.tiling_params["ub_size_info"].get("expand")
        dilation_window_w = (self.filter_w - 1) * self.rate_w + 1

        # right up area step offset
        w_total_step = (self.w_in + self.pad_left + self.pad_right - dilation_window_w) // self.stride_w + 1
        right_up_begin_step = w_total_step - w_right_pad_steps
        cut_w_ub_begin_step = w_num - w_right_pad_steps

        w_right_elem_start_offset = 0
        h_right_elem_start_offset = 1 if self.pad_top == 1 else self.pad_top // self.rate_h
        h_right_elem_start_offset = \
            self.pad_top // self.rate_h + 1 if self.pad_top % self.rate_h != 0 else h_right_elem_start_offset

        # calculate w & h elements
        outer_elem = right_up_begin_step * self.stride_w + dilation_window_w - (self.pad_left + self.w_in)
        outer_elem_no_dilation = 1 if outer_elem == 1 else outer_elem // self.rate_w
        outer_elem_no_dilation = \
            outer_elem // self.rate_w + 1 if outer_elem % self.rate_w != 0 else outer_elem_no_dilation

        w_right_elem_range = self.filter_w - outer_elem_no_dilation
        h_right_elem_range = self.filter_h - h_right_elem_start_offset + h_pos // self.rate_h

        with self.instance.for_range(0, w_right_pad_steps) as w_step_i:
            h_range = h_right_elem_range
            w_range = w_right_elem_range - w_step_i * self.stride_w // self.rate_w
            with self.instance.for_range(0, h_range) as fh_i:
                with self.instance.for_range(0, w_range) as fw_j:
                    gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                    gm_c1_offset = c1_offset * self.w_in * self.h_in * self.c0
                    h_start_offset = h_pos % self.rate_h
                    gm_update_idx = (h_start_offset + fh_i * self.rate_h) * self.w_in + \
                                    (right_up_begin_step * self.stride_w - self.pad_left) + w_step_i * self.stride_w + \
                                    fw_j * self.rate_w
                    ub_update_idx = ((h_right_elem_start_offset + fh_i - h_pos // self.rate_h) * self.filter_w +
                                     w_right_elem_start_offset + fw_j) * cut_size + (cut_w_ub_begin_step + w_step_i) * \
                                    self.c0
                    self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                            update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)

        # update center up area
        right_center_up_range = self.instance.Scalar("int32", name="right_center_up_range")
        right_center_up_range.set_as(w_num - w_right_pad_steps)

        w_right_center_up_elem_start_offset = 0
        h_right_center_up_elem_start_offset = 1 if self.pad_top == 1 else self.pad_top // self.rate_h
        h_right_center_up_elem_start_offset = \
            self.pad_top // self.rate_h + 1 if self.pad_top % self.rate_h != 0 else h_right_center_up_elem_start_offset

        w_right_center_up_elem_range = self.filter_w
        h_right_center_up_elem_range = self.filter_h - self.pad_top // self.rate_h + h_pos // self.rate_h

        ub_right_center_up_begin_step = 0

        with self.instance.if_scope(right_center_up_range > 0):
            with self.instance.for_range(0, right_center_up_range) as w_step_i:
                with self.instance.for_range(0, h_right_center_up_elem_range) as fh_i:
                    with self.instance.for_range(0, w_right_center_up_elem_range) as fw_j:
                        gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                        gm_c1_offset = c1_offset * self.w_in * self.h_in * self.c0
                        h_right_center_start_offset = h_pos % self.rate_h
                        w_right_center_start_offset = w_offset * self.stride_w - self.pad_left
                        gm_update_idx = (h_right_center_start_offset + fh_i * self.rate_h) * self.w_in + \
                                         w_right_center_start_offset + w_step_i * self.stride_w + fw_j * self.rate_w
                        ub_update_idx = ((h_right_center_up_elem_start_offset + fh_i - h_pos // self.rate_h) *
                                         self.filter_w + w_right_center_up_elem_start_offset + fw_j) * cut_size + \
                                        (ub_right_center_up_begin_step + w_step_i) * self.c0
                        self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                                update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)

        self.instance.set_atomic_add(0)

    # cut_w: branch3 center up area
    def cut_w_update_center_up_area(self, update_matrix_ub, n_offset, c1_offset, h_pos, w_offset, w_num):
        """
        cut_w_update_center_up_area
        """
        self.instance.set_atomic_add(1)
        cut_size = self.tiling_params["ub_size_info"].get("expand")
        center_up_range = self.instance.Scalar("int32", name="center_up_range")
        center_up_range.set_as(w_num)

        w_center_up_elem_start_offset = 0
        h_center_up_elem_start_offset = 1 if self.pad_top == 1 else self.pad_top // self.rate_h
        h_center_up_elem_start_offset = \
            self.pad_top // self.rate_h + 1 if self.pad_top % self.rate_h != 0 else h_center_up_elem_start_offset

        w_center_up_elem_range = self.filter_w
        h_center_up_elem_range = self.filter_h - self.pad_top // self.rate_h + h_pos // self.rate_h

        ub_center_up_begin_step = 0

        with self.instance.for_range(0, center_up_range) as w_step_i:
            with self.instance.for_range(0, h_center_up_elem_range) as fh_i:
                with self.instance.for_range(0, w_center_up_elem_range) as fw_j:
                    gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                    gm_c1_offset = c1_offset * self.w_in * self.h_in * self.c0
                    h_right_center_start_offset = h_pos % self.rate_h
                    w_right_center_start_offset = w_offset * self.stride_w - self.pad_left
                    gm_update_idx = (h_right_center_start_offset + fh_i * self.rate_h) * self.w_in + \
                                     w_right_center_start_offset + w_step_i * self.stride_w + fw_j * self.rate_w
                    ub_update_idx = ((h_center_up_elem_start_offset + fh_i - h_pos // self.rate_h) * self.filter_w +
                                     w_center_up_elem_start_offset + fw_j) * cut_size + \
                                    (ub_center_up_begin_step + w_step_i) * self.c0
                    self.instance.data_move(
                        self.y_gm[gm_n_offset + gm_c1_offset + w_offset * self.stride_w + gm_update_idx * self.c0],
                        update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)
        self.instance.set_atomic_add(0)

    # cut_w: branch4 left and center middle area
    def cut_w_update_left_and_center_middle_area(self, update_matrix_ub, n_offset, c1_offset, h_pos, w_offset, w_num):
        """
        cut_w_update_left_and_center_middle_area
        """
        self.instance.set_atomic_add(1)

        # (1) update left middle area
        if self.pad_left == 1:
            w_left_pad_steps = 1
        elif self.pad_left % self.stride_w != 0:
            w_left_pad_steps = self.pad_left // self.stride_w + 1
        else:
            w_left_pad_steps = self.pad_left // self.stride_w

        cut_size = self.tiling_params["ub_size_info"].get("expand")

        w_left_elem_start_offset = 1 if self.pad_left == 1 else self.pad_left // self.rate_w
        w_left_elem_start_offset = \
            self.pad_left // self.rate_w + 1 if self.pad_top % self.rate_w != 0 else w_left_elem_start_offset
        h_left_elem_start_offset = 0

        w_left_elem_range = self.filter_w - w_left_elem_start_offset
        h_left_elem_range = self.filter_h

        with self.instance.for_range(0, w_left_pad_steps) as w_step_i:
            h_range = h_left_elem_range
            w_range = w_left_elem_range + w_step_i * self.stride_w // self.rate_w
            with self.instance.for_range(0, h_range) as fh_i:
                with self.instance.for_range(0, w_range) as fw_j:
                    gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                    gm_c1_offset = c1_offset * self.w_in * self.h_in * self.c0
                    w_start_offset = w_step_i * self.stride_w % self.rate_w
                    gm_update_idx = (h_pos - self.pad_top +
                                     fh_i * self.rate_h) * self.w_in + w_start_offset + fw_j * self.rate_w
                    ub_update_idx = (
                        (h_left_elem_start_offset + fh_i) * self.filter_w + w_left_elem_start_offset + fw_j -
                        (w_step_i * self.stride_w) // self.rate_w) * cut_size + w_step_i * self.c0
                    self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                            update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)

        # (2) update center middle area
        left_center_middle_range = self.instance.Scalar("int32", name="left_center_middle_range")
        left_center_middle_range.set_as(w_num - w_left_pad_steps)

        begin_step = w_left_pad_steps

        with self.instance.if_scope(left_center_middle_range > 0):
            with self.instance.for_range(0, left_center_middle_range) as w_step_i:
                with self.instance.for_range(0, self.filter_h) as fh_i:
                    with self.instance.for_range(0, self.filter_w) as fw_j:
                        gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                        gm_c1_offset = c1_offset * self.w_in * self.h_in * self.c0
                        w_start_offset = w_left_pad_steps * self.stride_w - self.pad_left
                        gm_update_idx = (h_pos - self.pad_top + fh_i * self.rate_h) * self.w_in + \
                                         w_start_offset + w_step_i * self.stride_w + fw_j * self.rate_w
                        ub_update_idx = (fh_i * self.filter_w + fw_j) * cut_size + \
                                        (begin_step + w_step_i) * self.c0
                        self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                                update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)
        self.instance.set_atomic_add(0)

    # cut_w: branch5 right and center middle area
    def cut_w_update_right_and_center_middle_area(self, update_matrix_ub, n_offset, c1_offset, h_pos, w_offset, w_num):
        """
        cut_w_update_right_and_center_middle_area
        """
        self.instance.set_atomic_add(1)

        # (1) update right middle area
        if self.pad_right == 1:
            w_right_pad_steps = 1
        elif self.pad_right % self.stride_w != 0:
            w_right_pad_steps = self.pad_right // self.stride_w + 1
        else:
            w_right_pad_steps = self.pad_right // self.stride_w

        dilation_window_w = (self.filter_w - 1) * self.rate_w + 1
        cut_size = self.tiling_params["ub_size_info"].get("expand")

        # right up area step offset
        w_total_step = (self.w_in + self.pad_left + self.pad_right - dilation_window_w) // self.stride_w + 1
        right_center_begin_step = w_total_step - w_right_pad_steps
        cut_w_ub_begin_step = w_num - w_right_pad_steps

        w_right_elem_start_offset = 0
        h_right_elem_start_offset = 0

        # calculate w & h elements
        outer_elem = right_center_begin_step * self.stride_w + dilation_window_w - (self.pad_left + self.w_in)
        outer_elem_no_dilation = 1 if outer_elem == 1 else outer_elem // self.rate_w
        outer_elem_no_dilation = \
            outer_elem // self.rate_w + 1 if outer_elem % self.rate_w != 0 else outer_elem_no_dilation

        w_right_elem_range = self.filter_w - outer_elem_no_dilation
        h_right_elem_range = self.filter_h

        with self.instance.for_range(0, w_right_pad_steps) as w_step_i:
            h_range = h_right_elem_range
            w_range = w_right_elem_range - w_step_i * self.stride_w // self.rate_w
            with self.instance.for_range(0, h_range) as fh_i:
                with self.instance.for_range(0, w_range) as fw_j:
                    gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                    gm_c1_offset = c1_offset * self.w_in * self.h_in * self.c0
                    gm_update_idx = (h_pos - self.pad_top + fh_i * self.rate_h) * self.w_in + \
                                    (right_center_begin_step * self.stride_w - self.pad_left) + fw_j * \
                                    self.rate_w + w_step_i * self.stride_w
                    ub_update_idx = ((h_right_elem_start_offset + fh_i) * self.filter_w + w_right_elem_start_offset +
                                     fw_j) * cut_size + (cut_w_ub_begin_step + w_step_i) * self.c0
                    self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                            update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)

        # (2) update center middle arae
        right_center_middle_range = self.instance.Scalar("int32", name="right_center_middle_range")
        right_center_middle_range.set_as(w_num - w_right_pad_steps)
        ub_begin_step = 0

        with self.instance.if_scope(right_center_middle_range > 0):
            with self.instance.for_range(0, right_center_middle_range) as w_step_i:
                with self.instance.for_range(0, self.filter_h) as fh_i:
                    with self.instance.for_range(0, self.filter_w) as fw_j:
                        gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                        gm_c1_offset = c1_offset * self.w_in * self.h_in * self.c0
                        w_start_offset = w_offset * self.stride_w
                        gm_update_idx = (h_pos - self.pad_top + fh_i * self.rate_h) * self.w_in + \
                                         w_start_offset + w_step_i * self.stride_w + fw_j * self.rate_w
                        ub_update_idx = (fh_i * self.filter_w + fw_j) * cut_size + \
                                        (ub_begin_step + w_step_i) * self.c0
                        self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                                update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)
        self.instance.set_atomic_add(0)

    # cut_w: branch6 center middle area
    def cut_w_update_center_middle_area(self, update_matrix_ub, n_offset, c1_offset, h_pos, w_offset, w_num):
        """
        cut_w_update_center_middle_area
        """
        self.instance.set_atomic_add(1)
        cut_size = self.tiling_params["ub_size_info"].get("expand")
        center_middle_range = self.instance.Scalar("int32", name="center_middle_range")
        center_middle_range.set_as(w_num)
        ub_begin_step = 0

        with self.instance.for_range(0, center_middle_range) as w_step_i:
            with self.instance.for_range(0, self.filter_h) as fh_i:
                with self.instance.for_range(0, self.filter_w) as fw_j:
                    gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                    gm_c1_offset = c1_offset * self.w_in * self.h_in * self.c0
                    w_start_offset = w_offset * self.stride_w
                    gm_update_idx = (h_pos - self.pad_top + fh_i * self.rate_h) * self.w_in + \
                                     w_start_offset + w_step_i * self.stride_w + fw_j * self.rate_w
                    ub_update_idx = (fh_i * self.filter_w + fw_j) * cut_size + \
                                    (ub_begin_step + w_step_i) * self.c0
                    self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                            update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)

        self.instance.set_atomic_add(0)

    # cut_w: branch7 left and center bottom area
    def cut_w_update_left_and_center_bottom_area(self, update_matrix_ub, n_offset, c1_offset, h_pos, w_offset, w_num):
        """
        cut_w_update_center_middle_area
        """
        self.instance.set_atomic_add(1)
        # (1) update left and bottom area
        cut_size = self.tiling_params["ub_size_info"].get("expand")
        dilation_window_h = (self.filter_h - 1) * self.rate_h + 1

        w_left_pad_steps = 1 if self.pad_left == 1 else self.pad_left // self.stride_w
        w_left_pad_steps = \
            self.pad_left // self.stride_w + 1 if self.pad_left % self.stride_w != 0 else w_left_pad_steps

        h_left_pad_steps = 1 if self.pad_bottom == 1 else self.pad_bottom // self.stride_h
        h_left_pad_steps = \
            self.pad_bottom // self.stride_h + 1 if self.pad_bottom % self.stride_h != 0 else h_left_pad_steps

        w_left_elem_start_offset = 1 if self.pad_left == 1 else self.pad_left // self.rate_w
        w_left_elem_start_offset = \
            self.pad_left // self.rate_w + 1 if self.pad_top % self.rate_w != 0 else w_left_elem_start_offset
        h_left_elem_start_offset = 0

        h_total_step = (self.pad_top + self.h_in + self.pad_bottom - dilation_window_h) // self.stride_h + 1
        h_outer_elem = (h_total_step - h_left_pad_steps) * self.stride_h + dilation_window_h - (self.pad_top +
                                                                                                self.h_in)
        h_outer_elem_no_dilation = 1 if h_outer_elem == 1 else h_outer_elem // self.rate_h
        h_outer_elem_no_dilation = \
            h_outer_elem // self.rate_h + 1 if h_outer_elem % self.rate_h != 0 else h_outer_elem_no_dilation

        w_left_elem_range = self.filter_w - w_left_elem_start_offset
        h_left_elem_range = self.filter_h - h_outer_elem_no_dilation

        with self.instance.for_range(0, w_left_pad_steps) as w_step_i:
            h_range = h_left_elem_range
            w_range = w_left_elem_range + w_step_i * self.stride_w // self.rate_w
            with self.instance.for_range(0, h_range) as fh_i:
                with self.instance.for_range(0, w_range) as fw_j:
                    gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                    gm_c1_offset = c1_offset * self.w_in * self.h_in * self.c0
                    w_start_offset = w_step_i * self.stride_w % self.rate_w
                    gm_update_idx = (h_pos - self.pad_top + fh_i * self.rate_h) * self.w_in + w_start_offset + \
                                    fw_j * self.rate_w
                    ub_update_idx = ((h_left_elem_start_offset + fh_i) * self.filter_w + w_left_elem_start_offset -
                                     (w_step_i * self.stride_w) // self.rate_w + fw_j) * cut_size + w_step_i * self.c0
                    self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                            update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)

        # (2) update center and bottom area
        left_center_bottom_range = self.instance.Scalar("int32", name="left_center_bottom_range")
        left_center_bottom_range.set_as(w_num - w_left_pad_steps)

        w_center_bottom_elem_start_offset = 0
        h_center_bottom_elem_start_offset = 0

        w_center_bottom_elem_range = self.filter_w
        h_center_bottom_elem_range = self.filter_h - (h_pos - self.pad_top + dilation_window_h -
                                                      self.h_in) // self.rate_h
        ub_center_begin_step = w_left_pad_steps

        with self.instance.if_scope(left_center_bottom_range > 0):
            with self.instance.for_range(0, left_center_bottom_range) as w_step_i:
                with self.instance.for_range(0, h_center_bottom_elem_range) as fh_i:
                    with self.instance.for_range(0, w_center_bottom_elem_range) as fw_j:
                        gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                        gm_c1_offset = c1_offset * self.w_in * self.h_in * self.c0
                        w_start_offset = w_left_pad_steps * self.stride_w - self.pad_left
                        gm_update_idx = ((h_pos - self.pad_top) + fh_i * self.rate_h) * self.w_in + \
                                         w_start_offset + w_step_i * self.stride_w + fw_j * self.rate_w
                        ub_update_idx = ((h_center_bottom_elem_start_offset +  fh_i) * self.filter_w +
                                         w_center_bottom_elem_start_offset + fw_j) * cut_size + \
                                        (ub_center_begin_step + w_step_i) * self.c0
                        self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                                update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)

        self.instance.set_atomic_add(0)

    # cut_w: branch8 right and center bottom area
    def cut_w_update_right_and_center_bottom_area(self, update_matrix_ub, n_offset, c1_offset, h_pos, w_offset, w_num):
        """
        cut_w_update_right_and_center_bottom_area
        """
        self.instance.set_atomic_add(1)
        cut_size = self.tiling_params["ub_size_info"].get("expand")
        dilation_window_h = (self.filter_h - 1) * self.rate_h + 1
        dilation_window_w = (self.filter_w - 1) * self.rate_w + 1

        w_right_pad_steps = 1 if self.pad_right == 1 else self.pad_right // self.stride_w
        w_right_pad_steps = \
            self.pad_right // self.stride_w + 1 if self.pad_right % self.stride_w != 0 else w_right_pad_steps

        w_total_step = (self.pad_left + self.w_in + self.pad_right - dilation_window_w) // self.stride_w + 1
        right_up_begin_step = w_total_step - w_right_pad_steps

        # calculate w & h elements
        w_outer_elem = (w_total_step - w_right_pad_steps) * self.stride_w + dilation_window_w - (self.pad_left +
                                                                                                 self.w_in)
        w_outer_elem_no_dilation = 1 if w_outer_elem == 1 else w_outer_elem // self.rate_w
        w_outer_elem_no_dilation = \
            w_outer_elem // self.rate_w + 1 if w_outer_elem % self.rate_w != 0 else w_outer_elem_no_dilation

        w_right_elem_range = self.filter_w - w_outer_elem_no_dilation
        h_right_elem_range = self.filter_h - (h_pos - self.pad_top + dilation_window_h - self.h_in) // self.rate_h
        ub_right_down_step = w_num - w_right_pad_steps

        with self.instance.for_range(0, w_right_pad_steps) as w_step_i:
            h_range = h_right_elem_range
            w_range = w_right_elem_range - w_step_i * self.stride_w // self.rate_w
            with self.instance.for_range(0, h_range) as fh_i:
                with self.instance.for_range(0, w_range) as fw_j:
                    gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                    gm_c1_offset = c1_offset * self.w_in * self.h_in * self.c0
                    gm_update_idx = (h_pos - self.pad_top + fh_i * self.rate_h) * self.w_in + \
                                    (right_up_begin_step * self.stride_w - self.pad_left) + w_step_i * \
                                    self.stride_w + fw_j * self.rate_w
                    ub_update_idx = (fh_i * self.filter_w + fw_j) * cut_size + (ub_right_down_step + w_step_i) * self.c0
                    self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                            update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)

        # (2) update center and bottom area
        right_center_bottom_range = self.instance.Scalar("int32", name="right_center_bottom_range")
        right_center_bottom_range.set_as(w_num - w_right_pad_steps)

        w_center_bottom_elem_range = self.filter_w
        h_center_bottom_elem_range = self.filter_h - (h_pos - self.pad_top + dilation_window_h -
                                                      self.h_in) // self.rate_h

        with self.instance.if_scope(right_center_bottom_range > 0):
            with self.instance.for_range(0, right_center_bottom_range) as w_step_i:
                with self.instance.for_range(0, h_center_bottom_elem_range) as fh_i:
                    with self.instance.for_range(0, w_center_bottom_elem_range) as fw_j:
                        gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                        gm_c1_offset = c1_offset * self.w_in * self.h_in * self.c0
                        w_start_offset = w_offset * self.stride_w
                        gm_update_idx = ((h_pos - self.pad_top) + fh_i * self.rate_h) * self.w_in + \
                                         w_start_offset + w_step_i * self.stride_w + fw_j * self.rate_w
                        ub_update_idx = (fh_i * self.filter_w + fw_j) * cut_size + w_step_i * self.c0
                        self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                                update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)

        self.instance.set_atomic_add(0)

    # cut_w: branch9 center bottom area
    def cut_w_update_center_bottom_area(self, update_matrix_ub, n_offset, c1_offset, h_pos, w_offset, w_num):
        """
        cut_w_update_center_bottom_area
        """
        self.instance.set_atomic_add(1)
        cut_size = self.tiling_params["ub_size_info"].get("expand")
        center_bottom_range = self.instance.Scalar("int32", name="center_bottom_range")
        center_bottom_range.set_as(w_num)

        dilation_window_h = (self.filter_h - 1) * self.rate_h + 1
        w_center_bottom_elem_range = self.filter_w
        h_center_bottom_elem_range = self.filter_h - (h_pos - self.pad_top + dilation_window_h -
                                                      self.h_in) // self.rate_h

        with self.instance.for_range(0, center_bottom_range) as w_step_i:
            with self.instance.for_range(0, h_center_bottom_elem_range) as fh_i:
                with self.instance.for_range(0, w_center_bottom_elem_range) as fw_j:
                    gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                    gm_c1_offset = c1_offset * self.w_in * self.h_in * self.c0
                    w_start_offset = w_offset * self.stride_w
                    gm_update_idx = ((h_pos - self.pad_top) + fh_i * self.rate_h) * self.w_in + \
                                     w_start_offset + w_step_i * self.stride_w + fw_j * self.rate_w
                    ub_update_idx = (fh_i * self.filter_w + fw_j) * cut_size + w_step_i * self.c0
                    self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                            update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)

        self.instance.set_atomic_add(0)

    def cut_w_update_gm_pad_mode(self, update_matrix_ub, n_offset, c1_offset, h_offset, w_offset, w_num):
        """
        cut_w_update_gm_pad_mode
        """
        dilation_window_h = (self.filter_h - 1) * self.rate_h + 1
        dilation_window_w = (self.filter_w - 1) * self.rate_w + 1

        h_pos = self.instance.Scalar("int32", name="h_pos")
        h_pos.set_as(h_offset * self.stride_h)

        h_pos_up_boundary = self.instance.Scalar("int32", name="h_pos_up_boundary")
        h_pos_up_boundary.set_as(h_offset * self.stride_h - self.pad_top)

        w_pos_left_boundary = self.instance.Scalar("int32", name="w_pos_left_boundary")
        w_pos_left_boundary.set_as(self.pad_left - w_offset * self.stride_w)

        w_len = self.instance.Scalar("int32", name="w_len")
        w_len.set_as((w_offset + w_num - 1) * self.stride_w - self.pad_left + self.window_w - 1)

        if self.pad_bottom == 1:
            down_area_step = 1
        elif self.pad_bottom % self.stride_h != 0:
            down_area_step = self.pad_bottom // self.stride_h + 1
        else:
            down_area_step = self.pad_bottom // self.stride_h

        h_total_step = (self.h_in + self.pad_top + self.pad_bottom - dilation_window_h) // self.stride_h + 1
        bottom_begin_step = h_total_step - down_area_step
        bottom_area_start_pos = bottom_begin_step * self.stride_h

        # update upper area
        with self.instance.if_scope(h_pos_up_boundary < 0):
            # left up and center up area
            with self.instance.if_scope(w_pos_left_boundary > 0):
                self.cut_w_update_left_and_center_up_area(update_matrix_ub, n_offset, c1_offset, h_pos, w_offset, w_num)

            # right up and center up area
            with self.instance.if_scope(w_len >= self.w_in):
                self.cut_w_update_right_and_center_up_area(update_matrix_ub, n_offset, c1_offset, h_pos, w_offset,
                                                           w_num)

            # center up area
            with self.instance.if_scope(tik.all(self.pad_left - w_offset * self.stride_w < 0, w_len < self.w_in)):
                self.cut_w_update_center_up_area(update_matrix_ub, n_offset, c1_offset, h_pos, w_offset, w_num)

        # update middle area
        with self.instance.if_scope(tik.all(h_pos_up_boundary >= 0, h_pos < bottom_area_start_pos)):
            # left up and center middle area
            with self.instance.if_scope(w_pos_left_boundary > 0):
                self.cut_w_update_left_and_center_middle_area(update_matrix_ub, n_offset, c1_offset, h_pos, w_offset,
                                                              w_num)

            # right up and center middle area
            with self.instance.if_scope(w_len >= self.w_in):
                self.cut_w_update_right_and_center_middle_area(update_matrix_ub, n_offset, c1_offset, h_pos, w_offset,
                                                               w_num)

            # center middle area
            with self.instance.if_scope(tik.all(self.pad_left - w_offset * self.stride_w < 0, w_len < self.w_in)):
                self.cut_w_update_center_middle_area(update_matrix_ub, n_offset, c1_offset, h_pos, w_offset, w_num)

        # update bottom area
        with self.instance.if_scope(h_pos >= bottom_area_start_pos):
            # left bottom and center middle area
            with self.instance.if_scope(w_pos_left_boundary > 0):
                self.cut_w_update_left_and_center_bottom_area(update_matrix_ub, n_offset, c1_offset, h_pos, w_offset,
                                                              w_num)

            # right bottom and center middle area
            with self.instance.if_scope(w_len >= self.w_in):
                self.cut_w_update_right_and_center_bottom_area(update_matrix_ub, n_offset, c1_offset, h_pos, w_offset,
                                                               w_num)

            # center bottom area
            with self.instance.if_scope(tik.all(self.pad_left - w_offset * self.stride_w < 0, w_len < self.w_in)):
                self.cut_w_update_center_bottom_area(update_matrix_ub, n_offset, c1_offset, h_pos, w_offset, w_num)

    def cut_w_update_valid_mode(self, update_matrix_ub, n_offset, c1_offset, h_pos, w_offset, w_num):
        """
        cut_w_update_valid_mode
        """
        self.instance.set_atomic_add(1)
        w_steps = self.instance.Scalar("int32", name="w_steps")
        w_steps.set_as(w_num)
        cut_size = self.tiling_params["ub_size_info"].get("expand")

        with self.instance.for_range(0, w_steps) as w_step_i:
            with self.instance.for_range(0, self.filter_h) as fh_i:
                with self.instance.for_range(0, self.filter_w) as fw_j:
                    gm_n_offset = n_offset * self.c1 * self.w_in * self.h_in * self.c0
                    gm_c1_offset = c1_offset * self.w_in * self.h_in * self.c0
                    gm_update_idx = (h_pos + fh_i * self.rate_h) * self.w_in + \
                                     w_offset * self.stride_w + w_step_i * self.stride_w + fw_j * self.rate_w
                    ub_update_idx = (fh_i * self.filter_w + fw_j) * cut_size + w_step_i * self.c0
                    self.instance.data_move(self.y_gm[gm_n_offset + gm_c1_offset + gm_update_idx * self.c0],
                                            update_matrix_ub[ub_update_idx], 0, 1, 2, 0, 0)

        self.instance.set_atomic_add(0)

    def cut_h(self, ub_list, offset_list, h_num, out_offset):
        """
        tiling cut h scenario
        """
        expand_ub, x_ub, filter_ub, out_backprop_ub, mask_ub, update_matrix_ub, max_data_ub, sel_zero_ub = ub_list
        n_offset, c1_offset, h_offset = offset_list

        h_size = self.out_backprop_w * self.c0
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

        w_in = self.pad_w
        x_index = n_offset * self.x_offset_list[0] + c1_offset * self.x_offset_list[1] + h_beg * self.x_offset_list[2]
        ub_offset = x_h_offset * w_in * self.c0 + self.pad_left * self.c0
        self.move_data(x_ub, ub_offset, x_index, h_len)
        self.move_out_backprop_data(h_num * self.out_backprop_w * self.c0, out_backprop_ub, out_offset)

        if not self.tiling_params["is_filter_all"]:
            gm_index = c1_offset * self.filter_offset_list[1]
            self.move_filter_data(self.filter_offset_list[1], filter_ub, gm_index)

        with self.instance.for_range(0, self.filter_h) as fh_i:
            with self.instance.for_range(0, self.filter_w) as fw_i:
                with self.instance.for_range(0, h_num) as h_i:
                    ho_i = h_offset + h_i
                    h_in = ho_i * self.stride_h - self.pad_top + fh_i * self.rate_h

                    expand_start = h_i * h_size
                    x_start = (h_i * self.stride_h + fh_i * self.rate_h) * w_in * self.c0 + \
                              fw_i * self.rate_w * self.c0

                    if self.tiling_params["is_filter_all"]:
                        filter_start = c1_offset * self.filter_offset_list[1] + \
                                       fh_i * self.filter_offset_list[2] + \
                                       fw_i * self.filter_offset_list[3]
                    else:
                        filter_start = fh_i * self.filter_offset_list[2] + \
                                       fw_i * self.filter_offset_list[3]

                    self.expand_row([expand_ub, x_ub, filter_ub], h_in, [expand_start, x_start, filter_start], fw_i)

                ub_list = [mask_ub, expand_ub, max_data_ub]
                mask_ub, max_data_ub = self.search_max_position(ub_list, fh_i, fw_i)

        ub_list = [mask_ub, sel_zero_ub, out_backprop_ub, update_matrix_ub]
        update_matrix_ub = self.update_max_pos_h(ub_list, h_num)

        if self.input_params["padding_mode"] == "SAME":
            self.cut_h_update_gm_pad_mode(update_matrix_ub, n_offset, c1_offset, h_offset, h_num)
        else:
            h_pos = self.instance.Scalar("int32", name="h_pos")
            with self.instance.for_range(0, h_num) as h_step_i:
                h_pos.set_as((h_offset + h_step_i) * self.stride_h)
                self.cut_h_update_valid_mode(update_matrix_ub, n_offset, c1_offset, h_pos, h_step_i)

    def expand_row_w(self, ub_list, h_in, start_list, fw_i, x_w_offset, w_num, w_offset):
        """
        expand data by filter row, reduce row and reduce col
        """
        expand_ub, _, _ = ub_list
        expand_start, x_start, filter_start = start_list
        h_size = w_num * self.c0

        num = min(8 // self.stride_w, 255 // 2 // self.stride_w)
        rep_num = 1 if num < 0 else num
        rep_stride_list = [rep_num, self.stride_w * rep_num, 0]
        blk_stride_list = [1, self.stride_w, 0]

        with self.instance.if_scope(tik.all(h_in >= 0, h_in < self.h_in)):
            self.vector_add(start_list, ub_list, w_num, rep_stride_list, blk_stride_list, num * self.c0)
            self.vector_add([expand_start + 8, x_start + 8, filter_start + 8], ub_list, w_num, rep_stride_list,
                            blk_stride_list, num * self.c0)
            with self.instance.if_scope(self.pad_left - w_offset * self.stride_w > 0):
                num = (x_w_offset - fw_i * self.rate_w + self.stride_w - 1) // self.stride_w
                self.vector_dup(expand_start, expand_ub, num * self.c0, Constant.MIN_VAL)

            w_len = (w_offset + w_num - 1) * self.stride_w - self.pad_left + self.window_w - 1

            num_ = self.instance.Scalar("int32", name="num_")
            num_.set_as(0)
            with self.instance.if_scope(w_len >= self.w_in):
                end = (self.pad_left + self.w_in + x_w_offset - fw_i * self.rate_w - w_offset * self.stride_w +
                       self.stride_w - 1) // self.stride_w
                num_.set_as(w_num - end)
                with self.instance.if_scope(num_ < 0):
                    num_.set_as(0)
                self.vector_dup(expand_start + end * self.c0, expand_ub, num_ * self.c0, Constant.MIN_VAL)

        with self.instance.else_scope():
            self.vector_dup(expand_start, expand_ub, h_size, Constant.MIN_VAL)

    def cut_w(self, ub_list, offset_list, w_num, out_offset):
        """
        tiling cut w scenario
        """
        expand_ub, x_ub, filter_ub, out_backprop_ub, mask_ub, update_matrix_ub, max_data_ub, sel_zero_ub = ub_list
        n_offset, c1_offset, h_offset, w_offset = offset_list
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
        if self.need_vconv:
            x_size = self.tiling_params["ub_size_info"].get("input")
            x_ub_fp16 = self.instance.Tensor(self.x_dtype, (x_size,), name="x_ub_fp16", scope=tbe_platform.scope_ubuf)
            if self.w_in >= 65535:
                with self.instance.for_range(0, h_len) as h_i:
                    self.instance.data_move(x_ub_fp16[x_ub_index + h_i * w_all * self.c0],
                                            self.x_gm[x_index + h_i * self.w_in * self.c0], 0, 1,
                                            w_len * self.c0 // self.x_block_size, 0, 0)
            else:
                self.instance.data_move(x_ub_fp16[x_ub_index], self.x_gm[x_index], 0, h_len,
                                        w_len * self.c0 // self.x_block_size,
                                        (self.w_in - w_len) * self.c0 // self.x_block_size,
                                        (w_all - w_len) * self.c0 // self.x_block_size)
            self.vector_conv([0, 0], [x_ub, x_ub_fp16], x_size)
        else:
            if self.w_in >= 65535:
                with self.instance.for_range(0, h_len) as h_i:
                    self.instance.data_move(x_ub[x_ub_index + h_i * w_all * self.c0],
                                            self.x_gm[x_index + h_i * self.w_in * self.c0], 0, 1,
                                            w_len * self.c0 // self.y_block_size, 0, 0)
            else:
                self.instance.data_move(x_ub[x_ub_index], self.x_gm[x_index], 0, h_len,
                                        w_len * self.c0 // self.y_block_size,
                                        (self.w_in - w_len) * self.c0 // self.y_block_size,
                                        (w_all - w_len) * self.c0 // self.y_block_size)

        self.move_out_backprop_data(w_num * self.c0, out_backprop_ub, out_offset)
        if not self.tiling_params["is_filter_all"]:
            gm_index = c1_offset * self.filter_offset_list[1]
            self.move_filter_data(self.filter_offset_list[1], filter_ub, gm_index)

        with self.instance.for_range(0, self.filter_h) as fh_i:
            with self.instance.for_range(0, self.filter_w) as fw_i:
                h_in = h_offset * self.stride_h - self.pad_top + fh_i * self.rate_h
                expand_start = 0
                x_start = fh_i * self.rate_h * w_all * self.c0 + \
                          fw_i * self.rate_w * self.c0

                if self.tiling_params["is_filter_all"]:
                    filter_start = c1_offset * self.filter_offset_list[1] + \
                                   fh_i * self.filter_offset_list[2] + \
                                   fw_i * self.filter_offset_list[3]
                else:
                    filter_start = fh_i * self.filter_offset_list[2] + \
                                   fw_i * self.filter_offset_list[3]

                self.expand_row_w([expand_ub, x_ub, filter_ub], h_in, [expand_start, x_start, filter_start], fw_i,
                                  x_w_offset, w_num, w_offset)

                ub_list = [mask_ub, expand_ub, max_data_ub]
                mask_ub, max_data_ub = self.search_max_position(ub_list, fh_i, fw_i)

        ub_list = [mask_ub, sel_zero_ub, out_backprop_ub, update_matrix_ub]
        update_matrix_ub = self.update_max_pos_w(ub_list, w_num)

        if self.input_params["padding_mode"] == "SAME":
            self.cut_w_update_gm_pad_mode(update_matrix_ub, n_offset, c1_offset, h_offset, w_offset, w_num)
        else:
            h_pos = self.instance.Scalar("int32", name="h_pos")
            h_pos.set_as(h_offset * self.stride_h)
            self.cut_w_update_valid_mode(update_matrix_ub, n_offset, c1_offset, h_pos, w_offset, w_num)
