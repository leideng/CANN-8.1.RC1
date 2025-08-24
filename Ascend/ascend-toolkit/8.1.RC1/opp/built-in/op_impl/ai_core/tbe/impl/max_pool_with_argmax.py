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
max_pool_with_argmax
"""
import functools
import math

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl import max_pool_with_argmax_resnet50 as resnet50
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info
from impl.load3d_common_func import img2col

# size of fp16 is 2 byte
SIZE_FP16 = 2
# min value of fp16
MIN_VALUE_FP16 = -65504.0
MAX_VALUE_FP16 = 65535
SCALAR_C0 = 16
SCALAR_MAX = 240
SCALAR_255 = 255
# define dilation size
DILATION = 1
# parameters for vector instruct
MASK = 128
REPEAT_2 = 2
DSTSTRIDEM0 = 1
SRC0STRIDEM0 = 1
SRC1STRIDEM0 = 1
DSTSTRIDEM1 = 8
SRC0STRIDEM1 = 8
SRC1STRIDEM1 = 8
# get available ub size
UB_SIZE = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
# get available l1 size
L1_SIZE = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)


# 'pylint: disable=unused-argument,unused-variable,too-many-arguments,too-many-locals
def check_supported(input_x, out_y, output_argmax, ksize, strides,
                    padding, kernel_name="max_pool_with_argmax"):
    """
    check whether ai_core is supported
    """
    if ksize[1] * ksize[2] > SCALAR_255:
        reason = "the ksize is too big, ksize[1] * ksize[2]=%s" % ksize[1] * ksize[2]
        return False, reason
    input_format = input_x.get("ori_format").upper()
    input_shape = input_x.get("ori_shape")
    if input_format == "NHWC":
        in_size_h = input_shape[1]
        in_size_w = input_shape[2]
    else:
        in_size_h = input_shape[2]
        in_size_w = input_shape[3]
    window_h = ksize[1]
    window_w = ksize[2]
    stride_h = strides[1]
    stride_w = strides[2]

    # caculate output size in VALID mode
    if padding == "VALID":
        # `Hout = ceil(Hi - Fh + 1, Sh), Wout = ceil(Wi - Fw + 1, Sw)`
        out_size_h = (in_size_h - window_h + 1 + (stride_h - 1)) // stride_h
        out_size_w = (in_size_w - window_w + 1 + (stride_w - 1)) // stride_w
    if padding == "SAME":
        # `Hout = ceil(Hi, Sh), Wout = ceil(Wi, Sw)`
        out_size_h = (in_size_h + stride_h - 1) // stride_h
        out_size_w = (in_size_w + stride_w - 1) // stride_w
    if out_size_w == 1 and out_size_h > 1:
        reason = "out_size_w is not supported by schedule when out_size_w=1 and out_size_h>1, out_size_h:%s"\
                  % out_size_h
        return False, reason

    return True, ""


# 'pylint: disable=unused-argument,too-many-arguments
# 'pylint: disable=too-many-lines,unused-variable
def get_op_support_info(input_x, output_y, output_argmax, ksize, strides,
                        padding, kernel_name="max_pool_with_argmax"):
    """
    return: split info of max_pool_with_argmax
    """
    format_x = input_x.get("format").upper()
    if format_x == "NC1HWC0":
        axis_split_matrix = [
            [SplitInput([0, [0], [-1], [-1]]), SplitOutput([0, [0]], [1, [0]])],
            [SplitInput([0, [1], [-1], [-1]]), SplitOutput([0, [1]], [1, [1]])]
        ]
        axis_reduce_list = None

    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


def _ceil_div(value, factor):
    """
    caculate ceil value of div

    Parameters
    ----------
    value: dtype of int or float
        original value
    factor: dtype of int or float
        dividend value

    Returns
    -------
    value: dtype of int or float
    """
    return math.ceil(value / factor)


def _check_param(input_x, ksize, strides, padding, kernel_name):
    """
    check parameters, if one is invalid, then raise error

    Parameters
    ----------
    input_x: dict
        shape and datatype
    ksize: list or tuple
        the size of the window
    strides: list or tuple
        the stride of the sliding window
    padding: str
        value from `SAME`, `VALID`
    kernel_name: str

    Returns
    -------
    None
    """
    input_shape = input_x.get("shape")
    input_dtype = input_x.get("dtype").lower()

    para_check.check_shape(input_shape, param_name="input_x")
    para_check.check_dtype(input_dtype, ("float16",), param_name="input_x")

    # the format of input_x must be NC1HWC0
    if len(input_shape) != 5:
        expected_value = "equal to 5"
        real_value = "not equal to 5"
        error_manager_vector.raise_err_input_value_invalid(kernel_name,
                                                           "length of input_shape",
                                                           expected_value, real_value)
    # get shape info of feature map in NC1HWC0 format
    in_size_h = input_shape[2]
    in_size_w = input_shape[3]
    c_block_size = input_shape[4]

    if c_block_size != SCALAR_C0:
        error_manager_vector.raise_err_input_value_invalid(kernel_name,
                                                           "input_shape[4]",
                                                           SCALAR_C0, c_block_size)

    if len(ksize) != 4:
        expected_value = "equal to 4"
        real_value = "not equal to 4"
        error_manager_vector.raise_err_input_value_invalid(kernel_name,
                                                           "length of ksize",
                                                           expected_value, real_value)

    if ksize[0] != 1 or ksize[3] != 1:
        expected_value = "equal to 1"
        real_value = "not equal to 1"
        error_manager_vector.raise_err_input_value_invalid(kernel_name,
                                                           "ksize[0] and ksize[3]",
                                                           expected_value, real_value)
    if len(strides) != 4:
        expected_value = "equal to 4"
        real_value = "not equal to 4"
        error_manager_vector.raise_err_input_value_invalid(kernel_name,
                                                           "length of strides",
                                                           expected_value, real_value)

    if strides[0] != 1 or strides[3] != 1:
        expected_value = "equal to 1"
        real_value = "not equal to 1"
        error_manager_vector.raise_err_input_value_invalid(kernel_name,
                                                           "strides[0] and strides[3]",
                                                           expected_value, real_value)

    if ksize[1] * ksize[2] > SCALAR_255:
        expected_value = "smaller than or equal to 255"
        real_value = "greater than 255"
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "ksize[1] * ksize[2]",
                                                           expected_value, real_value)

    if padding not in ("SAME", "VALID"):
        expected_value = "SAME or VALID"
        real_value = padding
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "padding",
                                                           expected_value, real_value)


# 'pylint: disable=too-many-arguments,unused-argument,too-many-lines
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_STR,
                            para_check.KERNEL_NAME)
def max_pool_with_argmax(input_x, output_y, output_argmax, ksize, strides,
                         padding, kernel_name="max_pool_with_argmax"):
    """
    Performs max pooling on the input and outputs both max values and indices.

    Parameters
    ----------
    input_x: dict
        shape and datatype
    output_y: dict
        The max pooled output tensor.
    output_argmax: dict
        the max values chosen for each output.
    ksize: list or tuple
        The size of the window for each dimension of the input tensor.
    strides: list or tuple
        The stride of the sliding window for each dimension of the input tensor.
    padding: str
        The type of padding algorithm to use.
    kernel_name: str
        kernel_name, default value is 'max_pool_with_argmax'

    Returns
    -------
    max_pool_reslut: reslut of maxpool
    """
    _check_param(input_x, ksize, strides, padding, kernel_name)
    if resnet50.is_max_pool_with_argmax_param(input_x, ksize, strides, padding):
        return resnet50.max_pool_with_argmax(input_x, ksize, strides, padding, kernel_name)
    max_pool_reslut = MaxPoolWithargmax(input_x, ksize, strides, padding)

    return max_pool_reslut.tik_instance_function(kernel_name)


# 'pylint: disable=too-many-instance-attributes,too-few-public-methods
class MaxPoolWithargmax():
    """
       Function: use to finish MaxPoolWithargmax main functions
       Modify : 2019-10-16
    """

    def __init__(self, input_x, ksize, strides, padding):
        """
        init MaxPoolWithargmax parameters

        Parameters
        ----------
        input_x: dict
            shape and datatype
        ksize: list or tuple
            The size of the window for each dimension of the input tensor.
        strides: list or tuple
            The stride of the sliding window of the input tensor.
        padding: str
            The type of padding algorithm to use.

        Returns
        -------
        None
        """
        self.input_shape = input_x.get("shape")
        self.input_dtype = input_x.get("dtype").lower()
        self.tik_instance = tik.Tik()

        self.ksize = ksize
        self.strides = strides
        self.padding = padding
        self.batch_size = self.input_shape[0]
        self.c1_size = self.input_shape[1]
        self.in_size_h = self.input_shape[2]
        self.in_size_w = self.input_shape[3]
        self.c_block_size = self.input_shape[4]

        self.window_h = self.ksize[1]
        self.window_w = self.ksize[2]
        self.stride_h = self.strides[1]
        self.stride_w = self.strides[2]
        self.nc1 = self.batch_size * self.c1_size
        # scalar for load3d
        self.scalar_source_h = self.tik_instance.Scalar(dtype="int64")
        self.scalar_source_w = self.tik_instance.Scalar(dtype="int64")

        # caculate pad and output size
        self.pad, self.out_size_h, self.out_size_w = self._calc_out_size_and_pad()
        # output_shape
        self.fmap_img2col_h = self.out_size_h * self.out_size_w
        self.fmap_img2col_w = self.window_h * self.window_w
        self.fmap_img2col_h_num = _ceil_div(self.fmap_img2col_h, self.c_block_size)
        mask_tmp = self.fmap_img2col_h_num * SCALAR_C0 - self.fmap_img2col_h
        self.mask_zero = 2 ** SCALAR_C0 - 2**(SCALAR_C0 - mask_tmp)

        if self.input_dtype == "float16":
            self.pad_value = MIN_VALUE_FP16
        # famp is NC1HWC0 format
        fmap_gm_shape = (self.batch_size, self.c1_size, self.in_size_h, self.in_size_w, self.c_block_size)

        output_gm_shape = (self.batch_size, self.c1_size, self.out_size_h, self.out_size_w, self.c_block_size)
        output_mask_gm_shape = (self.batch_size, self.c1_size, self.fmap_img2col_w,
                                (self.fmap_img2col_h_num + 1) * self.c_block_size)
        # input and output
        self.input_fmap_gm = self.tik_instance.Tensor(self.input_dtype, fmap_gm_shape, name="input_fmap_gm",
                                                      scope=tik.scope_gm)
        self.output_max_gm = self.tik_instance.Tensor(self.input_dtype, output_gm_shape, name="output_max_gm",
                                                      scope=tik.scope_gm)
        self.output_mask_gm = self.tik_instance.Tensor("uint16", output_mask_gm_shape, name="output_mask_gm",
                                                       scope=tik.scope_gm, is_atomic_add=True)

        self.is_support_load3d = tbe_platform.api_check_support("tik.load3dv1")

    # 'pylint: disable=too-many-locals,too-many-function-args,too-many-branches,too-many-statements
    def tik_instance_function(self, kernel_name):
        """
        tik_instance_function

        Parameters
        ----------
        kernel_name: str
            kernel_name

        Returns
        -------
        tik_instance
        """
        # caculate if need cutH or cutW
        # caculate block number
        core_counts = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        need_cut_h, need_cut_h_w, need_cut = self._check_if_need_cut_h_w()
        cut_h_size, cut_stride, cut_h_num = self._calc_cut_h_size_fun(need_cut)
        # cut_h_size must be smaller than in_size_h
        if need_cut_h or need_cut:
            flag_cut_h = False
            out_size_cut_h = (cut_h_size - self.window_h + self.stride_h) // self.stride_h
            fmap_img2col_cut_h = self.out_size_w * out_size_cut_h
            if (fmap_img2col_cut_h % SCALAR_C0) == 0:
                flag_cut_h = True
                nc1_cuth = self.nc1 * cut_h_num
            else:
                nc1_cuth = self.nc1

            if need_cut_h_w:
                cut_w_size, cut_w_stride, cut_w_num = self._calc_cut_w_size_fun()
                cut_w_tail = self.in_size_w + self.pad[0] - cut_w_stride * (cut_w_num - 1)
                out_size_tail_w = (cut_w_tail - self.window_w + self.stride_w + self.pad[1]) // self.stride_w
                if (out_size_tail_w % SCALAR_C0) == 0 and cut_w_size > 0:
                    flag_cut_h = True
                    nc1_cuth = self.nc1 * cut_h_num
                else:
                    nc1_cuth = self.nc1
                    flag_cut_h = False

            nc1_cuth_size = nc1_cuth // core_counts + (1 if nc1_cuth % core_counts > 0 else 0)
            if (nc1_cuth % core_counts == 0) or (nc1_cuth % nc1_cuth_size == 0):
                is_same_percore = 0
            else:
                is_same_percore = 1

            block_dim = nc1_cuth // nc1_cuth_size + (0 if nc1_cuth // core_counts == 0 else is_same_percore)
            with self.tik_instance.for_range(0, block_dim, block_num=block_dim) as block_index:
                with self.tik_instance.if_scope(block_index != block_dim - 1):
                    with self.tik_instance.for_range(0, nc1_cuth_size) as nc1_cuth_index:
                        # size of ub is not enough, need cutH
                        if need_cut_h_w:
                            self._fun_need_cut_h_w(block_index, nc1_cuth_index, cut_h_size, cut_stride,
                                                   cut_h_num, nc1_cuth_size, flag_cut_h)
                        else:
                            self._fun_only_cut_h(block_index, nc1_cuth_index, cut_h_size,
                                                 cut_stride, cut_h_num, nc1_cuth_size, flag_cut_h)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, nc1_cuth - (block_dim - 1) * nc1_cuth_size) as nc1_cuth_index:
                        # size of ub is not enough, need cutH
                        if need_cut_h_w:
                            self._fun_need_cut_h_w(block_index, nc1_cuth_index, cut_h_size, cut_stride,
                                                   cut_h_num, nc1_cuth_size, flag_cut_h)
                        else:
                            self._fun_only_cut_h(block_index, nc1_cuth_index, cut_h_size,
                                                 cut_stride, cut_h_num, nc1_cuth_size, flag_cut_h)
        # no need cut
        else:
            nc1_size = self.nc1 // core_counts + (1 if self.nc1 % core_counts > 0 else 0)
            if (self.nc1 % core_counts == 0) or (self.nc1 % nc1_size == 0):
                is_same_percore = 0
            else:
                is_same_percore = 1

            block_dim = self.nc1 // nc1_size + (0 if self.nc1 // core_counts == 0 else is_same_percore)

            with self.tik_instance.for_range(0, block_dim, block_num=block_dim) as block_index:
                with self.tik_instance.if_scope(block_index != block_dim - 1):
                    with self.tik_instance.for_range(0, nc1_size) as nc1_index:
                        self._fun_no_cut(block_index, nc1_index, nc1_size)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, self.nc1 - (block_dim - 1) * nc1_size) as nc1_index:
                        self._fun_no_cut(block_index, nc1_index, nc1_size)

        self.tik_instance.BuildCCE(kernel_name=kernel_name, inputs=(self.input_fmap_gm),
                                   outputs=(self.output_max_gm, self.output_mask_gm))
        return self.tik_instance

    def _check_if_need_cut_h_w(self):
        """
        funtion check if need cutH or cutW

        Parameters
        ----------
            none

        Returns
        -------
        need_cut_h: bool
        need_cut_h_w: bool
        need_cut: bool

        """
        need_cut_h = False
        need_cut_h_w = False
        need_cut = False
        if self.is_support_load3d:
            ub_size_used_max = self.out_size_h * self.out_size_w * SCALAR_C0 * self.window_h * self.window_w * SIZE_FP16
            ub_size_cut_h_max = self.out_size_w * SCALAR_C0 * self.window_h * self.window_w * SIZE_FP16

            if ub_size_used_max > (UB_SIZE / 2):
                need_cut_h = True

            if ub_size_cut_h_max > (UB_SIZE / 2):
                need_cut_h_w = True

            if self.window_h * self.in_size_w * self.c_block_size * SIZE_FP16 > L1_SIZE:
                expected_value = "smaller than supported value"
                real_value = "greater than supported value"
                error_manager_vector.raise_err_input_value_invalid("max_pool_with_argmax",
                                                                   "ksize or input shape",
                                                                   expected_value, real_value)

            if not need_cut_h:
                if self.in_size_h * self.in_size_w * self.c_block_size * SIZE_FP16 > L1_SIZE:
                    need_cut = True
        else:
            ub_input_size = (self.in_size_w + self.pad[0] + self.pad[1]) * \
                            (self.in_size_h + self.pad[2] + self.pad[3]) * SCALAR_C0 * SIZE_FP16
            fmap_size = self.out_size_h * self.out_size_w * SCALAR_C0 * self.window_h * self.window_w * SIZE_FP16
            ub_size_cut_h = (self.in_size_w + self.pad[0] + self.pad[1]) * self.window_h * SCALAR_C0 * SIZE_FP16
            fmap_w_size = self.out_size_w * SCALAR_C0 * self.window_h * self.window_w * SIZE_FP16

            if ub_input_size + fmap_size * 2 > UB_SIZE:
                need_cut_h = True

            if ub_size_cut_h + fmap_w_size * 2 > UB_SIZE:
                need_cut_h_w = True

            if self.window_h * self.in_size_w * self.c_block_size * SIZE_FP16 * 3 > UB_SIZE:
                expected_value = "smaller than supported value"
                real_value = "greater than supported value"
                error_manager_vector.raise_err_input_value_invalid("max_pool_with_argmax",
                                                                   "ksize or input shape",
                                                                   expected_value, real_value)


        return need_cut_h, need_cut_h_w, need_cut

    def _vector_dup(self, src, src_start, shape, dup_reg):
        vector_fp16_size = 128
        max_vector_repeat_time = 255

        ele_num = functools.reduce(lambda x, y: x * y, shape)
        total_repeat_time = ele_num // vector_fp16_size
        remain_ele = ele_num % vector_fp16_size
        mask_value = vector_fp16_size
        repeat_max_time = total_repeat_time // max_vector_repeat_time
        remain_repeat_time = total_repeat_time % max_vector_repeat_time

        with self.tik_instance.for_range(0, repeat_max_time) as loop:
            self.tik_instance.vector_dup(mask_value, src[src_start + loop * max_vector_repeat_time * mask_value],
                                         dup_reg, max_vector_repeat_time, 1, 8)

        if remain_repeat_time > 0:
            self.tik_instance.vector_dup(mask_value,
                                         src[src_start + repeat_max_time * max_vector_repeat_time * mask_value],
                                         dup_reg, remain_repeat_time, 1, 8)

        if remain_ele > 0:
            self.tik_instance.vector_dup(remain_ele,
                                         src[src_start + repeat_max_time * max_vector_repeat_time * mask_value +
                                             remain_repeat_time * mask_value],
                                         dup_reg, 1, 1, 8)

    # 'pylint: disable=too-many-locals
    def _fun_no_cut(self, block_index, nc1_index, nc1_size):
        """
        funtion while no need cut H

        Parameters
        ----------
        block_index: index of block
        nc1_index: index of nc1

        Returns
        -------
        none
        """

        fmap_img2col_shape_ub = (self.fmap_img2col_h_num * SCALAR_C0, self.window_h, self.window_w, self.c_block_size)
        fmap_img2col_ub = self.tik_instance.Tensor(self.input_dtype, fmap_img2col_shape_ub, name="fmap_img2col_ub",
                                                   scope=tik.scope_ubuf)
        mask_shape_ub = (self.window_h, self.window_w, self.fmap_img2col_h_num, self.c_block_size)
        mask_ub = self.tik_instance.Tensor("uint16", mask_shape_ub, name="mask_ub", scope=tik.scope_ubuf)
        data_x_max = self.tik_instance.Tensor("float16", (self.fmap_img2col_h_num, SCALAR_C0, SCALAR_C0),
                                              name="data_x_max", scope=tik.scope_ubuf)
        if self.is_support_load3d:
            # copy input fmap from gm to l1
            fmap_l1_shape = (self.in_size_h, self.in_size_w, self.c_block_size)
            input_fmap_l1 = self.tik_instance.Tensor(self.input_dtype, fmap_l1_shape, name="input_fmap_l1",
                                                     scope=tik.scope_cbuf)
            gm_l1_burst_len = int(self.in_size_h * self.in_size_w * self.c_block_size // SCALAR_C0)
            self.tik_instance.data_move(input_fmap_l1, self.input_fmap_gm[(block_index * nc1_size + nc1_index) *
                                                                          self.in_size_h * self.in_size_w *
                                                                          self.c_block_size],
                                        0, 1, gm_l1_burst_len, 0, 0)
        else:
            padding_w_size = self.in_size_w + self.pad[0] + self.pad[1]
            padding_h_size = self.in_size_h + self.pad[2] + self.pad[3]
            ori_ub_shape = (padding_h_size, padding_w_size, self.c_block_size)
            ori_ub_input = self.tik_instance.Tensor(self.input_dtype, ori_ub_shape, name="ori_ub_input",
                                                    scope=tik.scope_ubuf)
            ori_ub_scalar = self.tik_instance.Scalar(dtype="int32")
            ori_ub_scalar.set_as(padding_w_size * padding_h_size * SCALAR_C0)
            self._vector_dup(ori_ub_input, 0, ori_ub_shape, self.pad_value)

            if sum(self.pad) == 0:
                gm_ub_burst_len = int(self.in_size_h * self.in_size_w * self.c_block_size // SCALAR_C0)
                self.tik_instance.data_move(ori_ub_input, self.input_fmap_gm[(block_index * nc1_size + nc1_index) *
                                                                              self.in_size_h * self.in_size_w *
                                                                              self.c_block_size],
                                        0, 1, gm_ub_burst_len, 0, 0)
            else:
                with self.tik_instance.for_range(0, self.in_size_h) as pad_index:
                    self.tik_instance.data_move(ori_ub_input[(padding_w_size *
                                                              self.pad[2] + self.pad[0] + pad_index *
                                                              padding_w_size) * SCALAR_C0],
                                                self.input_fmap_gm[(block_index * nc1_size + nc1_index) *
                                                                    self.in_size_h * self.in_size_w *
                                                                    self.c_block_size + pad_index * self.in_size_w *
                                                                    self.c_block_size],
                                                0, 1, self.in_size_w, 0, 0)
                self._vector_dup(fmap_img2col_ub, 0, fmap_img2col_ub.shape, self.pad_value)

        with self.tik_instance.for_range(0, self.fmap_img2col_h_num) as h_index:
            if self.is_support_load3d:
                source_h = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) /
                            (SCALAR_C0 * self.fmap_img2col_w)) / self.out_size_w) * \
                            self.stride_h - self.pad[2]
                source_w = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) /
                            (SCALAR_C0 * self.fmap_img2col_w)) % self.out_size_w) * \
                            self.stride_w - self.pad[0]

                self.scalar_source_h.set_as(source_h)
                self.scalar_source_w.set_as(source_w)
                self.tik_instance.load3dv1(fmap_img2col_ub[h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w],
                                           input_fmap_l1[0], self.pad, self.in_size_h, self.in_size_w, 0, 0, 0,
                                           self.scalar_source_w, self.scalar_source_h, self.stride_w, self.stride_h,
                                           self.window_w, self.window_h, 1, 1, 1, 0, self.fmap_img2col_w,
                                           0, self.pad_value)
            else:
                source_h = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) /
                            (SCALAR_C0 * self.fmap_img2col_w)) / self.out_size_w) * \
                            self.stride_h
                source_w = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) /
                            (SCALAR_C0 * self.fmap_img2col_w)) % self.out_size_w) * \
                            self.stride_w
                self.scalar_source_h.set_as(source_h)
                self.scalar_source_w.set_as(source_w)

                ori_ub_scalar = self.tik_instance.Scalar(dtype="int32")
                ori_ub_scalar.set_as(padding_h_size * padding_w_size * SCALAR_C0)
                self._img2col(h_index, fmap_img2col_ub, ori_ub_input, padding_w_size, ori_ub_scalar)

        if self.fmap_img2col_w != 1:
            self._calc_max_and_mask(self.fmap_img2col_h_num, fmap_img2col_ub, data_x_max, mask_ub)
            # move max output to gm
            self.tik_instance.data_move(self.output_max_gm[(block_index * nc1_size + nc1_index) *
                                                           self.out_size_h * self.out_size_w * self.c_block_size],
                                        data_x_max[0], 0, 1, self.fmap_img2col_h, 0, 0)
            self._remove_repeated_fun(mask_ub)
        else:
            # move max output to gm
            self.tik_instance.data_move(self.output_max_gm[(block_index * nc1_size + nc1_index) *
                                                           self.out_size_h * self.out_size_w * self.c_block_size],
                                        fmap_img2col_ub[0], 0, 1, self.fmap_img2col_h, 0, 0)
            self._dup_mask_fun(mask_ub, mask_shape_ub)

        with self.tik_instance.for_range(0, self.fmap_img2col_w) as w_index:
            offset_output_mask = (block_index * nc1_size + nc1_index) * \
                                 (self.fmap_img2col_h_num + 1) * self.fmap_img2col_w * self.c_block_size
            if self.mask_zero != 0 and self.fmap_img2col_w != 1:
                self.tik_instance.vector_dup([0, self.mask_zero], mask_ub[
                    w_index * self.fmap_img2col_h_num * self.c_block_size +
                    self.fmap_img2col_h_num * SCALAR_C0 - SCALAR_C0], 0, 1, 1, 8)

            self.tik_instance.data_move(self.output_mask_gm[offset_output_mask + w_index *
                                                            (self.fmap_img2col_h_num + 1) * SCALAR_C0],
                                        mask_ub[w_index * self.fmap_img2col_h_num * self.c_block_size],
                                        0, 1, self.fmap_img2col_h_num, 0, 0)

    # 'pylint: disable=too-many-locals,too-many-statements
    def _calc_only_cut_h(self, cut_h_index, cut_h_size, cut_stride, cut_h_num,
                         fmap_img2col_ub, fmap_img2col_cut_h,
                         mask_shape_ub, nc1_num):
        """
        calc only cut H

        Parameters
        ----------
        cut_h_index: index of cuth
        cut_h_size: size of cuth
        cut_stride: stride of cuth
        cut_h_num: number of cuth
        fmap_img2col_ub: fmag in ub
        fmap_img2col_cut_h: fmag cutH
        mask_shape_ub: shape of mask
        nc1_num: num of n*c1

        Returns
        -------
        none
        """
        if self.is_support_load3d:
            fmap_l1_shape = (cut_h_size, self.in_size_w, self.c_block_size)
            # fmag in l1
            input_fmap_l1 = self.tik_instance.Tensor(self.input_dtype, fmap_l1_shape, name="input_fmap_l1",
                                                     scope=tik.scope_cbuf)

        fmap_img2col_cut_h_num = _ceil_div(fmap_img2col_cut_h, SCALAR_C0)
        mask_ub = self.tik_instance.Tensor("uint16", mask_shape_ub, name="mask_ub", scope=tik.scope_ubuf)
        data_x_max = self.tik_instance.Tensor("float16", (
            fmap_img2col_cut_h_num, SCALAR_C0, SCALAR_C0), name="data_x_max",
                                              scope=tik.scope_ubuf)
        len_tmp = self.tik_instance.Scalar(dtype="int32")
        len_tmp1 = self.tik_instance.Scalar(dtype="int32")
        pad_top = self.tik_instance.Scalar(dtype="int32")
        pad_bottom = self.tik_instance.Scalar(dtype="int32")
        gm_l1_burst_len_1 = self.tik_instance.Scalar(dtype="int32")
        gm_tem = self.tik_instance.Scalar(dtype="int32")
        with self.tik_instance.if_scope(cut_h_index != 0):
            with self.tik_instance.if_scope(cut_h_index != (cut_h_num - 1)):
                len_tmp.set_as(cut_h_size)
                # copy input fmap from gm to l1
                with self.tik_instance.if_scope(cut_h_size >= (self.in_size_h +
                                                               self.pad[2] - cut_stride * cut_h_index)):
                    len_tmp.set_as(self.in_size_h + self.pad[2] - cut_stride * cut_h_index)
                len_tmp1.set_as(len_tmp)
                with self.tik_instance.if_scope(len_tmp >= (cut_h_size - self.pad[2] +
                                                            cut_stride * cut_h_index)):
                    len_tmp1.set_as(cut_h_size - self.pad[2] + cut_stride * cut_h_index)
                gm_l1_burst_len_1.set_as(len_tmp1 * self.in_size_w)
                pad_top.set_as(0)
                with self.tik_instance.if_scope(self.pad[2] - cut_stride * cut_h_index > 0):
                    pad_top.set_as(self.pad[2] - cut_stride * cut_h_index)
                pad_bottom.set_as(0)
                with self.tik_instance.if_scope(cut_stride * cut_h_index + cut_h_size -
                                                self.pad[2] - self.in_size_h > 0):
                    pad_bottom.set_as(cut_stride * cut_h_index + cut_h_size - self.pad[2] - self.in_size_h)
                gm_tem.set_as(0)
                with self.tik_instance.if_scope(cut_h_index * cut_stride - self.pad[2] > 0):
                    gm_tem.set_as(cut_h_index * cut_stride - self.pad[2])

                if not self.is_support_load3d:
                    padding_w_size = self.in_size_w + self.pad[0] + self.pad[1]
                    ori_ub_shape = (cut_h_size, padding_w_size, self.c_block_size)
                    ori_ub_input = self.tik_instance.Tensor(self.input_dtype, ori_ub_shape, name="ori_ub_input",
                                                            scope=tik.scope_ubuf)
                    self._vector_dup(ori_ub_input, 0, ori_ub_shape, self.pad_value)

                    with self.tik_instance.for_range(0, len_tmp1) as pad_index:
                        self.tik_instance.data_move(ori_ub_input[(pad_top * padding_w_size + self.pad[0] + pad_index *
                                                                  padding_w_size) * SCALAR_C0],
                                                    self.input_fmap_gm[nc1_num * self.in_size_h * self.in_size_w *
                                                                       self.c_block_size + gm_tem * self.in_size_w *
                                                                       self.c_block_size + pad_index * self.in_size_w *
                                                                       self.c_block_size],
                                                    0, 1, self.in_size_w, 0, 0)
                    self._vector_dup(fmap_img2col_ub, 0, fmap_img2col_ub.shape, self.pad_value)
                else:
                    self.tik_instance.data_move(input_fmap_l1,
                            self.input_fmap_gm[nc1_num * self.in_size_h * self.in_size_w *
                                               self.c_block_size + gm_tem * self.in_size_w *
                                               self.c_block_size],
                            0, 1, gm_l1_burst_len_1, 0, 0)

                with self.tik_instance.for_range(0, fmap_img2col_cut_h_num) as h_index:
                    if self.is_support_load3d:
                        source_h = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) //
                                    (SCALAR_C0 * self.fmap_img2col_w)) //
                                    self.out_size_w) * self.stride_h - pad_top
                        source_w = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) //
                                    (SCALAR_C0 * self.fmap_img2col_w)) % self.out_size_w) * self.stride_w - self.pad[0]
                        self.scalar_source_h.set_as(source_h)
                        self.scalar_source_w.set_as(source_w)
                        self.tik_instance.load3dv1(
                            fmap_img2col_ub[h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w], input_fmap_l1[0],
                            (self.pad[0], self.pad[1], pad_top, pad_bottom), cut_h_size - pad_top - pad_bottom,
                            self.in_size_w, 0, 0, 0, self.scalar_source_w, self.scalar_source_h, self.stride_w,
                            self.stride_h, self.window_w, self.window_h, 1, 1, 1, 0, self.fmap_img2col_w, 0,
                            self.pad_value)
                    else:
                        source_h = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) //
                                    (SCALAR_C0 * self.fmap_img2col_w)) //
                                    self.out_size_w) * self.stride_h
                        source_w = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) //
                                    (SCALAR_C0 * self.fmap_img2col_w)) % self.out_size_w) * self.stride_w
                        self.scalar_source_h.set_as(source_h)
                        self.scalar_source_w.set_as(source_w)
                        ori_ub_scalar = self.tik_instance.Scalar(dtype="int32")
                        ori_ub_scalar.set_as(cut_h_size * padding_w_size * SCALAR_C0)
                        self._img2col(h_index, fmap_img2col_ub, ori_ub_input, padding_w_size, ori_ub_scalar)

                if self.fmap_img2col_w != 1:
                    self._calc_max_and_mask(fmap_img2col_cut_h_num, fmap_img2col_ub, data_x_max, mask_ub)
                    # move max output to gm
                    gm_max_burst_len = int(fmap_img2col_cut_h)
                    self.tik_instance.data_move(self.output_max_gm[nc1_num * self.out_size_h *
                                                                   self.out_size_w * self.c_block_size +
                                                                   cut_h_index * fmap_img2col_cut_h *
                                                                   self.c_block_size],
                                                data_x_max[0], 0, 1, gm_max_burst_len, 0, 0)
                    self._remove_repeated_fun(mask_ub, fmap_img2col_cut_h)
                else:
                    # move max output to gm
                    gm_max_burst_len = int(fmap_img2col_cut_h)
                    self.tik_instance.data_move(self.output_max_gm[nc1_num * self.out_size_h *
                                                                   self.out_size_w * self.c_block_size +
                                                                   cut_h_index * fmap_img2col_cut_h *
                                                                   self.c_block_size],
                                                fmap_img2col_ub[0], 0, 1, gm_max_burst_len, 0, 0)
                    self._dup_mask_fun(mask_ub, mask_shape_ub)
                with self.tik_instance.for_range(0, self.fmap_img2col_w) as w_index:
                    offset_output_mask = nc1_num * (self.fmap_img2col_h_num + 1) * \
                                         self.fmap_img2col_w * self.c_block_size + cut_h_index * fmap_img2col_cut_h
                    self.tik_instance.data_move(self.output_mask_gm[offset_output_mask + w_index *
                                                                    (self.fmap_img2col_h_num + 1) * SCALAR_C0],
                                                mask_ub[w_index * fmap_img2col_cut_h_num * self.c_block_size],
                                                0, 1, fmap_img2col_cut_h_num, 0, 0)
            with self.tik_instance.else_scope():
                cut_h_tail = self.in_size_h + self.pad[2] - cut_stride * (cut_h_num - 1)
                if cut_h_tail > cut_h_size:
                    cut_h_tail = cut_h_size
                out_size_h_tail = (cut_h_tail - self.window_h + self.stride_h + self.pad[3]) // self.stride_h
                fmap_img2col_h_tail = self.out_size_w * out_size_h_tail
                fmap_img2col_h_tail_num = _ceil_div(fmap_img2col_h_tail, SCALAR_C0)

                if not self.is_support_load3d:
                    padding_w_size = self.in_size_w + self.pad[0] + self.pad[1]
                    ori_ub_shape = (cut_h_tail+self.pad[3], padding_w_size, self.c_block_size)
                    ori_ub_input = self.tik_instance.Tensor(self.input_dtype, ori_ub_shape, name="ori_ub_input",
                                                            scope=tik.scope_ubuf)
                    self._vector_dup(ori_ub_input, 0, ori_ub_shape, self.pad_value)

                    if self.pad[0] == 0 and self.pad[1] == 0:
                        gm_ub_burst_len = int(cut_h_tail * self.in_size_w)
                        self.tik_instance.data_move(ori_ub_input,
                                                    self.input_fmap_gm[nc1_num * self.in_size_h * self.in_size_w *
                                                                       self.c_block_size + (cut_h_index * cut_stride -
                                                                                            self.pad[2]) *
                                                                       self.in_size_w * self.c_block_size],
                                                    0, 1, gm_ub_burst_len, 0, 0)
                    else:
                        with self.tik_instance.for_range(0, cut_h_tail) as pad_index:
                            self.tik_instance.data_move(ori_ub_input[(self.pad[0] + pad_index *
                                                                      padding_w_size) * SCALAR_C0],
                                                        self.input_fmap_gm[nc1_num * self.in_size_h * self.in_size_w *
                                                                           self.c_block_size + (cut_h_index *
                                                                           cut_stride - self.pad[2]) * self.in_size_w *
                                                                           self.c_block_size + pad_index *
                                                                           self.in_size_w * self.c_block_size],
                                                        0, 1, self.in_size_w, 0, 0)
                    self._vector_dup(fmap_img2col_ub, 0, fmap_img2col_ub.shape, self.pad_value)
                else:
                    # copy input fmap from gm to l1
                    gm_l1_burst_len = int(cut_h_tail * self.in_size_w)
                    self.tik_instance.data_move(input_fmap_l1,
                                                self.input_fmap_gm[nc1_num * self.in_size_h * self.in_size_w *
                                                                   self.c_block_size + (cut_h_index * cut_stride -
                                                                                        self.pad[2]) *
                                                                   self.in_size_w * self.c_block_size],
                                                0, 1, gm_l1_burst_len, 0, 0)

                with self.tik_instance.for_range(0, fmap_img2col_h_tail_num) as h_index:
                    if self.is_support_load3d:
                        source_h = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) /
                                    (SCALAR_C0 * self.fmap_img2col_w)) /
                                    self.out_size_w) * self.stride_h
                        source_w = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) /
                                    (SCALAR_C0 * self.fmap_img2col_w)) % self.out_size_w) * self.stride_w - self.pad[0]
                        self.scalar_source_h.set_as(source_h)
                        self.scalar_source_w.set_as(source_w)

                        self.tik_instance.load3dv1(
                            fmap_img2col_ub[h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w], input_fmap_l1[0],
                            (self.pad[0], self.pad[1], 0, self.pad[3]), cut_h_tail, self.in_size_w, 0, 0, 0,
                            self.scalar_source_w, self.scalar_source_h, self.stride_w, self.stride_h, self.window_w,
                            self.window_h, 1, 1, 1, 0, self.fmap_img2col_w, 0, self.pad_value)
                    else:
                        source_h = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) /
                                    (SCALAR_C0 * self.fmap_img2col_w)) /
                                    self.out_size_w) * self.stride_h
                        source_w = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) /
                                    (SCALAR_C0 * self.fmap_img2col_w)) % self.out_size_w) * self.stride_w
                        self.scalar_source_h.set_as(source_h)
                        self.scalar_source_w.set_as(source_w)

                        ori_ub_scalar = self.tik_instance.Scalar(dtype="int32")
                        ori_ub_scalar.set_as(ori_ub_shape[0] * padding_w_size * SCALAR_C0)
                        self._img2col(h_index, fmap_img2col_ub, ori_ub_input, padding_w_size, ori_ub_scalar)

                if self.fmap_img2col_w != 1:
                    self._calc_max_and_mask(fmap_img2col_h_tail_num, fmap_img2col_ub, data_x_max, mask_ub,
                                            fmap_img2col_cut_h_num)
                    # move max output to gm
                    gm_max_burst_len = int(fmap_img2col_h_tail)
                    self.tik_instance.data_move(
                        self.output_max_gm[
                            nc1_num * self.out_size_h *
                            self.out_size_w * self.c_block_size +
                            cut_h_index * fmap_img2col_cut_h *
                            self.c_block_size],
                        data_x_max[0], 0, 1, gm_max_burst_len, 0, 0)
                    self._remove_repeated_fun(mask_ub, fmap_img2col_h_tail, 0,
                                              0, fmap_img2col_cut_h)
                else:
                    # move max output to gm
                    gm_max_burst_len = int(fmap_img2col_h_tail)
                    self.tik_instance.data_move(
                        self.output_max_gm[
                            nc1_num * self.out_size_h *
                            self.out_size_w * self.c_block_size +
                            cut_h_index * fmap_img2col_cut_h *
                            self.c_block_size],
                        fmap_img2col_ub[0], 0, 1, gm_max_burst_len, 0, 0)
                    self._dup_mask_fun(mask_ub, mask_shape_ub)
                mask_cut = fmap_img2col_h_tail_num * SCALAR_C0 - fmap_img2col_h_tail
                mask_zero_cut = 2 ** SCALAR_C0 - 2**(SCALAR_C0 - mask_cut)
                with self.tik_instance.for_range(
                        0, self.fmap_img2col_w) as w_index:
                    offset_output_mask = \
                        nc1_num * (self.fmap_img2col_h_num + 1) * \
                        self.fmap_img2col_w * self.c_block_size + \
                        cut_h_index * fmap_img2col_cut_h
                    if mask_zero_cut != 0 and self.fmap_img2col_w != 1:
                        self.tik_instance.vector_dup([
                            0, mask_zero_cut], mask_ub[
                            w_index * fmap_img2col_cut_h_num * self.c_block_size +
                            fmap_img2col_h_tail_num * SCALAR_C0 - SCALAR_C0], 0, 1, 1, 8)

                    self.tik_instance.data_move(self.output_mask_gm[offset_output_mask + w_index *
                                                                    (self.fmap_img2col_h_num + 1) * SCALAR_C0],
                                                mask_ub[w_index * fmap_img2col_cut_h_num * self.c_block_size],
                                                0, 1, fmap_img2col_h_tail_num, 0, 0)
        with self.tik_instance.else_scope():
            if not self.is_support_load3d:
                padding_w_size = self.in_size_w + self.pad[0] + self.pad[1]
                ori_ub_shape = (cut_h_size, padding_w_size, self.c_block_size)
                ori_ub_input = self.tik_instance.Tensor(self.input_dtype, ori_ub_shape, name="ori_ub_input",
                                                        scope=tik.scope_ubuf)
                self._vector_dup(ori_ub_input, 0, ori_ub_shape, self.pad_value)

                if sum(self.pad) == 0:
                    gm_ub_burst_len = int((cut_h_size - self.pad[2]) * self.in_size_w)
                    self.tik_instance.data_move(ori_ub_input,
                                                self.input_fmap_gm[nc1_num * self.in_size_h * self.in_size_w *
                                                                   self.c_block_size],
                                                0, 1, gm_ub_burst_len, 0, 0)
                else:
                    with self.tik_instance.for_range(0, (cut_h_size-self.pad[2])) as pad_index:
                        self.tik_instance.data_move(ori_ub_input[(padding_w_size * self.pad[2] +
                                                                  self.pad[0] + pad_index *
                                                                  padding_w_size) * SCALAR_C0],
                                                    self.input_fmap_gm[nc1_num * self.in_size_h * self.in_size_w *
                                                                       self.c_block_size + pad_index * self.in_size_w *
                                                                       self.c_block_size],
                                                    0, 1, self.in_size_w, 0, 0)
                self._vector_dup(fmap_img2col_ub, 0, fmap_img2col_ub.shape, self.pad_value)
            else:
                # copy input fmap from gm to l1
                gm_l1_burst_len = int((cut_h_size - self.pad[2]) * self.in_size_w)
                self.tik_instance.data_move(input_fmap_l1, self.input_fmap_gm[nc1_num * self.in_size_h *
                                                                              self.in_size_w *
                                                                              self.c_block_size],
                                            0, 1, gm_l1_burst_len, 0, 0)

            with self.tik_instance.for_range(0, fmap_img2col_cut_h_num) as h_index:
                if self.is_support_load3d:
                    source_h = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) /
                               (SCALAR_C0 * self.fmap_img2col_w)) / self.out_size_w) * self.stride_h - self.pad[2]
                    source_w = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) /
                               (SCALAR_C0 * self.fmap_img2col_w)) % self.out_size_w) * self.stride_w - self.pad[0]
                    self.scalar_source_h.set_as(source_h)
                    self.scalar_source_w.set_as(source_w)
                    self.tik_instance.load3dv1(
                        fmap_img2col_ub[h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w], input_fmap_l1[0],
                        (self.pad[0], self.pad[1], self.pad[2], 0), (cut_h_size - self.pad[2]), self.in_size_w,
                        0, 0, 0, self.scalar_source_w, self.scalar_source_h, self.stride_w, self.stride_h,
                        self.window_w, self.window_h, 1, 1, 1, 0, self.fmap_img2col_w, 0, self.pad_value)
                else:
                    source_h = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) /
                               (SCALAR_C0 * self.fmap_img2col_w)) / self.out_size_w) * self.stride_h
                    source_w = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) /
                               (SCALAR_C0 * self.fmap_img2col_w)) % self.out_size_w) * self.stride_w
                    self.scalar_source_h.set_as(source_h)
                    self.scalar_source_w.set_as(source_w)

                    ori_ub_scalar = self.tik_instance.Scalar(dtype="int32")
                    ori_ub_scalar.set_as(cut_h_size * padding_w_size * SCALAR_C0)
                    self._img2col(h_index, fmap_img2col_ub, ori_ub_input, padding_w_size, ori_ub_scalar)

            if self.fmap_img2col_w != 1:
                self._calc_max_and_mask(fmap_img2col_cut_h_num, fmap_img2col_ub, data_x_max, mask_ub)
                # move max output to gm
                gm_max_burst_len = int(fmap_img2col_cut_h)
                self.tik_instance.data_move(self.output_max_gm[nc1_num * self.out_size_h * self.out_size_w *
                                                               self.c_block_size],
                                            data_x_max[0], 0, 1, gm_max_burst_len, 0, 0)
                self._remove_repeated_fun(mask_ub, fmap_img2col_cut_h)
            else:
                # move max output to gm
                gm_max_burst_len = int(fmap_img2col_cut_h)
                self.tik_instance.data_move(self.output_max_gm[nc1_num * self.out_size_h * self.out_size_w *
                                                               self.c_block_size],
                                            fmap_img2col_ub[0], 0, 1, gm_max_burst_len, 0, 0)
                self._dup_mask_fun(mask_ub, mask_shape_ub)
            with self.tik_instance.for_range(0, self.fmap_img2col_w) as w_index:
                offset_output_mask = nc1_num * (self.fmap_img2col_h_num + 1) * self.fmap_img2col_w * \
                                     self.c_block_size + cut_h_index * fmap_img2col_cut_h
                self.tik_instance.data_move(self.output_mask_gm[offset_output_mask + w_index *
                                                                (self.fmap_img2col_h_num + 1) * SCALAR_C0],
                                            mask_ub[w_index * fmap_img2col_cut_h_num * self.c_block_size],
                                            0, 1, fmap_img2col_cut_h_num, 0, 0)

    # 'pylint: disable=too-many-locals,too-many-statements
    def _fun_only_cut_h(self, block_index, nc1_cuth_index, cut_h_size, cut_stride, cut_h_num,
                        nc1_cuth_size, flag_cut_h):
        """
        funtion only cut H

        Parameters
        ----------
        block_index: index of block
        nc1_cuth_index: index of nc1_cuth
        cut_h_size: size of cuth
        cut_stride: stride of cuth
        cut_h_num: number of cuth
        nc1_cuth_size: size of nc1_cuth
        flag_cut_h: bool

        Returns
        -------
        none
        """
        out_size_cut_h = (cut_h_size - self.window_h + self.stride_h) // self.stride_h
        fmap_img2col_cut_h = self.out_size_w * out_size_cut_h
        fmap_img2col_cut_h_num = _ceil_div(fmap_img2col_cut_h, SCALAR_C0)
        fmap_img2col_shape_ub = (fmap_img2col_cut_h_num * SCALAR_C0, self.window_h, self.window_w, self.c_block_size)
        fmap_img2col_ub = self.tik_instance.Tensor(self.input_dtype, fmap_img2col_shape_ub,
                                                   name="fmap_img2col_ub", scope=tik.scope_ubuf)
        mask_shape_ub = (self.window_h, self.window_w, fmap_img2col_cut_h_num, self.c_block_size)
        if flag_cut_h:
            cut_h_index = (block_index * nc1_cuth_size + nc1_cuth_index) % cut_h_num
            nc1_num = (block_index * nc1_cuth_size + nc1_cuth_index) // cut_h_num
            self._calc_only_cut_h(cut_h_index, cut_h_size, cut_stride, cut_h_num,
                                  fmap_img2col_ub, fmap_img2col_cut_h, mask_shape_ub, nc1_num)
        else:
            nc1_num = block_index * nc1_cuth_size + nc1_cuth_index
            with self.tik_instance.for_range(0, cut_h_num) as cut_h_index:
                self._calc_only_cut_h(cut_h_index, cut_h_size, cut_stride, cut_h_num,
                                      fmap_img2col_ub, fmap_img2col_cut_h, mask_shape_ub, nc1_num)

    # 'pylint: disable=too-many-statements,too-many-branches
    def _calc_need_cut_h_w(self, nc1_num, cut_h_size, cut_h_num, cut_h_index, cut_stride):
        """
        funtion need cut H and W while l1 not enough

        Parameters
        ----------
        nc1_num: num of n*c1
        cut_h_size: size of cuth
        cut_stride: stride of cuth
        cut_h_num: number of cuth
        cut_h_index: index of cuth

        Returns
        -------
        none
        """
        cut_w_size, cut_w_stride, cut_w_num = self._calc_cut_w_size_fun()
        if self.is_support_load3d:
            fmap_l1_shape = (cut_h_size, self.in_size_w, self.c_block_size)
            input_fmap_l1 = self.tik_instance.Tensor(self.input_dtype, fmap_l1_shape, name="input_fmap_l1",
                                                     scope=tik.scope_cbuf)

        with self.tik_instance.for_range(0, cut_w_num) as cut_w_index:
            out_size_cut_h = (cut_h_size - self.window_h + self.stride_h) // self.stride_h
            fmap_img2col_cut_h = self.out_size_w * out_size_cut_h
            out_size_cut_w = (cut_w_size - self.window_w + self.stride_w) // self.stride_w
            fmap_img2col_cut_w = out_size_cut_w
            fmap_img2col_cut_w_num = _ceil_div(fmap_img2col_cut_w, SCALAR_C0)
            fmap_img2col_shape_ub = (fmap_img2col_cut_w_num * SCALAR_C0, self.window_h,
                                     self.window_w, self.c_block_size)
            fmap_img2col_ub = self.tik_instance.Tensor(self.input_dtype, fmap_img2col_shape_ub,
                                                       name="fmap_img2col_ub", scope=tik.scope_ubuf)
            mask_shape_ub = (self.window_h, self.window_w, fmap_img2col_cut_w_num, self.c_block_size)
            mask_ub = self.tik_instance.Tensor("uint16", mask_shape_ub, name="mask_ub", scope=tik.scope_ubuf)
            data_x_max = self.tik_instance.Tensor("float16", (fmap_img2col_cut_w_num, SCALAR_C0, SCALAR_C0),
                                                  name="data_x_max", scope=tik.scope_ubuf)
            len_tmp = self.tik_instance.Scalar(dtype="int32")
            len_tmp1 = self.tik_instance.Scalar(dtype="int32")
            pad_top = self.tik_instance.Scalar(dtype="int32")
            pad_bottom = self.tik_instance.Scalar(dtype="int32")
            gm_l1_burst_len_1 = self.tik_instance.Scalar(dtype="int32")
            gm_tem = self.tik_instance.Scalar(dtype="int32")
            with self.tik_instance.if_scope(cut_h_index != 0):
                with self.tik_instance.if_scope(cut_h_index != (cut_h_num - 1)):
                    # copy input fmap from gm to l1
                    len_tmp.set_as(cut_h_size)
                    with self.tik_instance.if_scope(cut_h_size >= (self.in_size_h +
                                                                   self.pad[2] - cut_stride * cut_h_index)):
                        len_tmp.set_as(self.in_size_h + self.pad[2] - cut_stride * cut_h_index)
                    len_tmp1.set_as(len_tmp)
                    with self.tik_instance.if_scope(len_tmp >= (cut_h_size - self.pad[2] +
                                                                cut_stride * cut_h_index)):
                        len_tmp1.set_as(cut_h_size - self.pad[2] + cut_stride * cut_h_index)
                    gm_l1_burst_len_1.set_as(len_tmp1 * self.in_size_w)
                    pad_top.set_as(0)
                    with self.tik_instance.if_scope(self.pad[2] - cut_stride * cut_h_index > 0):
                        pad_top.set_as(self.pad[2] - cut_stride * cut_h_index)
                    pad_bottom.set_as(0)
                    with self.tik_instance.if_scope(cut_stride * cut_h_index + cut_h_size -
                                                    self.pad[2] - self.in_size_h > 0):
                        pad_bottom.set_as(cut_stride * cut_h_index + cut_h_size - self.pad[2] - self.in_size_h)
                    gm_tem.set_as(0)
                    with self.tik_instance.if_scope(cut_h_index * cut_stride - self.pad[2] > 0):
                        gm_tem.set_as(cut_h_index * cut_stride - self.pad[2])
                    if self.is_support_load3d:
                        self.tik_instance.data_move(input_fmap_l1,
                                                    self.input_fmap_gm[nc1_num * self.in_size_h *
                                                                       self.in_size_w * self.c_block_size +
                                                                       gm_tem * self.in_size_w * self.c_block_size],
                                                    0, 1, gm_l1_burst_len_1, 0, 0)
                    else:
                        pad_w = self.in_size_w + self.pad[0] + self.pad[1]
                        ori_ub_shape = (cut_h_size, pad_w, self.c_block_size)
                        ori_ub_input = self.tik_instance.Tensor(self.input_dtype, ori_ub_shape,
                                                                name="ori_ub_input",
                                                                scope=tik.scope_ubuf)
                        self._vector_dup(ori_ub_input, 0, ori_ub_shape, self.pad_value)

                        with self.tik_instance.for_range(0, len_tmp1) as row_index:
                            self.tik_instance.data_move(ori_ub_input[(pad_top * pad_w + self.pad[0] + row_index *
                                                                      pad_w) * self.c_block_size],
                                                        self.input_fmap_gm[nc1_num * self.in_size_h *
                                                                           self.in_size_w * self.c_block_size +
                                                                           gm_tem * self.in_size_w * self.c_block_size +
                                                                           row_index * self.in_size_w *
                                                                           self.c_block_size],
                                                        0, 1, self.in_size_w, 0, 0)

                    with self.tik_instance.if_scope(cut_w_index != 0):
                        with self.tik_instance.if_scope(cut_w_index != (cut_w_num - 1)):
                            if not self.is_support_load3d:
                                self._vector_dup(fmap_img2col_ub, 0, fmap_img2col_ub.shape, self.pad_value)
                            with self.tik_instance. for_range(0, fmap_img2col_cut_w_num) as h_index:
                                if self.is_support_load3d:
                                    source_h = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) //
                                                (SCALAR_C0 * self.fmap_img2col_w)) //
                                                self.out_size_w) * self.stride_h - pad_top
                                    source_w = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) /
                                                (SCALAR_C0 * self.fmap_img2col_w)) % out_size_cut_w) * self.stride_w + \
                                                cut_w_stride * cut_w_index - self.pad[0]
                                    self.scalar_source_h.set_as(source_h)
                                    self.scalar_source_w.set_as(source_w)

                                    self.tik_instance.load3dv1(fmap_img2col_ub[h_index * SCALAR_C0 * SCALAR_C0 *
                                                                               self.fmap_img2col_w],
                                                               input_fmap_l1[0],
                                                               (self.pad[0], self.pad[1], pad_top, pad_bottom),
                                                               cut_h_size - pad_top - pad_bottom, self.in_size_w,
                                                               0, 0, 0, self.scalar_source_w,
                                                               self.scalar_source_h, self.stride_w, self.stride_h,
                                                               self.window_w, self.window_h, 1, 1, 1, 0,
                                                               self.fmap_img2col_w, 0, self.pad_value)
                                else:
                                    source_h = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) //
                                                (SCALAR_C0 * self.fmap_img2col_w)) //
                                                self.out_size_w) * self.stride_h
                                    source_w = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) /
                                                (SCALAR_C0 * self.fmap_img2col_w)) % out_size_cut_w) * self.stride_w + \
                                                cut_w_stride * cut_w_index
                                    self.scalar_source_h.set_as(source_h)
                                    self.scalar_source_w.set_as(source_w)

                                    ori_ub_scalar = self.tik_instance.Scalar(dtype="int32")
                                    ori_ub_scalar.set_as(cut_h_size * pad_w * SCALAR_C0)
                                    self._img2col(h_index, fmap_img2col_ub, ori_ub_input, pad_w, ori_ub_scalar)

                            if self.fmap_img2col_w != 1:
                                self._calc_max_and_mask(fmap_img2col_cut_w_num, fmap_img2col_ub, data_x_max, mask_ub)
                                # move max output to gm
                                gm_max_burst_len = int(fmap_img2col_cut_w)
                                self.tik_instance.data_move(self.output_max_gm[nc1_num * self.out_size_h *
                                                                               self.out_size_w * self.c_block_size +
                                                                               cut_h_index * self.out_size_w *
                                                                               self.c_block_size + cut_w_index *
                                                                               fmap_img2col_cut_w * self.c_block_size],
                                                            data_x_max[0], 0, 1, gm_max_burst_len, 0, 0)
                                self._remove_repeated_fun(mask_ub, fmap_img2col_cut_h, fmap_img2col_cut_w)
                            else:
                                # move max output to gm
                                gm_max_burst_len = int(fmap_img2col_cut_w)
                                self.tik_instance.data_move(self.output_max_gm[nc1_num * self.out_size_h *
                                                                               self.out_size_w * self.c_block_size +
                                                                               cut_h_index * self.out_size_w *
                                                                               self.c_block_size + cut_w_index *
                                                                               fmap_img2col_cut_w * self.c_block_size],
                                                            fmap_img2col_ub[0], 0, 1, gm_max_burst_len, 0, 0)
                                self._dup_mask_fun(mask_ub, mask_shape_ub)
                            with self.tik_instance. for_range(0, self.fmap_img2col_w) as w_index:
                                offset_output_mask = nc1_num * (self.fmap_img2col_h_num + 1) * self.fmap_img2col_w * \
                                                     self.c_block_size + cut_h_index * fmap_img2col_cut_h + \
                                                     cut_w_index * fmap_img2col_cut_w
                                self.tik_instance.data_move(self.output_mask_gm[offset_output_mask + w_index *
                                                                                (self.fmap_img2col_h_num + 1) *
                                                                                SCALAR_C0],
                                                            mask_ub[w_index * fmap_img2col_cut_w_num *
                                                                    self.c_block_size],
                                                            0, 1, fmap_img2col_cut_w_num, 0, 0)
                        with self.tik_instance.else_scope():
                            cut_w_tail = self.in_size_w + self.pad[0] - cut_w_stride * (cut_w_num - 1)
                            if cut_w_tail > cut_w_size:
                                cut_w_tail = cut_w_size
                            out_size_tail_w = (cut_w_tail - self.window_w + self.stride_w + self.pad[1]) // \
                                              self.stride_w
                            fmap_img2col_tail_w = out_size_tail_w
                            fmap_img2col_tail_w_num = _ceil_div(fmap_img2col_tail_w, SCALAR_C0)

                            if not self.is_support_load3d:
                                self._vector_dup(fmap_img2col_ub, 0, fmap_img2col_ub.shape, self.pad_value)
                            with self.tik_instance.for_range(0, fmap_img2col_tail_w_num) as h_index:
                                if self.is_support_load3d:
                                    source_h = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) //
                                               (SCALAR_C0 * self.fmap_img2col_w)) //
                                               self.out_size_w) * self.stride_h - pad_top
                                    source_w = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) /
                                               (SCALAR_C0 * self.fmap_img2col_w)) % out_size_tail_w) * \
                                               self.stride_w + cut_w_stride * cut_w_index - self.pad[0]
                                    self.scalar_source_h.set_as(source_h)
                                    self.scalar_source_w.set_as(source_w)

                                    self.tik_instance.load3dv1(fmap_img2col_ub[h_index * SCALAR_C0 * SCALAR_C0 *
                                                                               self.fmap_img2col_w],
                                                               input_fmap_l1[0],
                                                               (self.pad[0], self.pad[1], pad_top, pad_bottom),
                                                               cut_h_size - pad_top - pad_bottom, self.in_size_w,
                                                               0, 0, 0, self.scalar_source_w,
                                                               self.scalar_source_h, self.stride_w, self.stride_h,
                                                               self.window_w, self.window_h, 1, 1, 1, 0,
                                                               self.fmap_img2col_w, 0, self.pad_value)
                                else:
                                    source_h = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) //
                                               (SCALAR_C0 * self.fmap_img2col_w)) //
                                               self.out_size_w) * self.stride_h
                                    source_w = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) /
                                               (SCALAR_C0 * self.fmap_img2col_w)) % out_size_tail_w) * \
                                               self.stride_w + cut_w_stride * cut_w_index
                                    self.scalar_source_h.set_as(source_h)
                                    self.scalar_source_w.set_as(source_w)

                                    ori_ub_scalar = self.tik_instance.Scalar(dtype="int32")
                                    ori_ub_scalar.set_as(cut_h_size * pad_w * SCALAR_C0)
                                    self._img2col(h_index, fmap_img2col_ub, ori_ub_input, pad_w, ori_ub_scalar)

                            if self.fmap_img2col_w != 1:
                                self._calc_max_and_mask(fmap_img2col_tail_w_num, fmap_img2col_ub,
                                                        data_x_max, mask_ub, fmap_img2col_cut_w_num)
                                # move max output to gm
                                gm_max_burst_len = int(fmap_img2col_tail_w)
                                self.tik_instance.data_move(self.output_max_gm[nc1_num * self.out_size_h *
                                                                               self.out_size_w * self.c_block_size
                                                                               + cut_h_index * self.out_size_w *
                                                                               self.c_block_size + cut_w_index *
                                                                               fmap_img2col_cut_w * self.c_block_size],
                                                            data_x_max[0], 0, 1, gm_max_burst_len, 0, 0)
                                self._remove_repeated_fun(mask_ub, fmap_img2col_cut_h, fmap_img2col_tail_w,
                                                          fmap_img2col_cut_w)
                            else:
                                self.tik_instance.data_move(self.output_max_gm[nc1_num * self.out_size_h *
                                                                               self.out_size_w * self.c_block_size +
                                                                               cut_h_index * self.out_size_w *
                                                                               self.c_block_size + cut_w_index *
                                                                               fmap_img2col_cut_w * self.c_block_size],
                                                            fmap_img2col_ub[0], 0, 1, gm_max_burst_len, 0, 0)
                                self._dup_mask_fun(mask_ub, mask_shape_ub)
                            with self.tik_instance.for_range(0, self.fmap_img2col_w) as w_index:
                                offset_output_mask = nc1_num * (self.fmap_img2col_h_num + 1) * self.fmap_img2col_w * \
                                                     self.c_block_size + cut_h_index * fmap_img2col_cut_h + \
                                                     cut_w_index * fmap_img2col_cut_w
                                self.tik_instance.data_move(self.output_mask_gm[offset_output_mask + w_index *
                                                                                (self.fmap_img2col_h_num + 1) *
                                                                                SCALAR_C0],
                                                            mask_ub[w_index * fmap_img2col_cut_w_num *
                                                                    self.c_block_size],
                                                            0, 1, fmap_img2col_tail_w_num, 0, 0)
                    with self.tik_instance.else_scope():
                        if not self.is_support_load3d:
                            self._vector_dup(fmap_img2col_ub, 0, fmap_img2col_ub.shape, self.pad_value)
                        with self.tik_instance.for_range(0, fmap_img2col_cut_w_num) as h_index:
                            if self.is_support_load3d:
                                source_h = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w)
                                            // (SCALAR_C0 * self.fmap_img2col_w)) //
                                            self.out_size_w) * self.stride_h - pad_top
                                source_w = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) /
                                            (SCALAR_C0 * self.fmap_img2col_w)) % out_size_cut_w) * \
                                        self.stride_w - self.pad[0]
                                self.scalar_source_h.set_as(source_h)
                                self.scalar_source_w.set_as(source_w)

                                self.tik_instance.load3dv1(
                                    fmap_img2col_ub[h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w],
                                    input_fmap_l1[0], (self.pad[0], self.pad[1], pad_top, pad_bottom),
                                    cut_h_size - pad_top - pad_bottom, self.in_size_w, 0, 0, 0,
                                    self.scalar_source_w, self.scalar_source_h, self.stride_w, self.stride_h,
                                    self.window_w, self.window_h, 1, 1, 1, 0, self.fmap_img2col_w, 0, self.pad_value)
                            else:
                                source_h = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w)
                                            // (SCALAR_C0 * self.fmap_img2col_w)) //
                                            self.out_size_w) * self.stride_h
                                source_w = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) /
                                            (SCALAR_C0 * self.fmap_img2col_w)) % out_size_cut_w) * \
                                        self.stride_w
                                self.scalar_source_h.set_as(source_h)
                                self.scalar_source_w.set_as(source_w)

                                ori_ub_scalar = self.tik_instance.Scalar(dtype="int32")
                                ori_ub_scalar.set_as(cut_h_size * pad_w * SCALAR_C0)
                                self._img2col(h_index, fmap_img2col_ub, ori_ub_input, pad_w, ori_ub_scalar)

                        if self.fmap_img2col_w != 1:
                            self._calc_max_and_mask(fmap_img2col_cut_w_num, fmap_img2col_ub, data_x_max, mask_ub)
                            # move max output to gm
                            gm_max_burst_len = int(fmap_img2col_cut_w)
                            self.tik_instance.data_move(self.output_max_gm[nc1_num * self.out_size_h *
                                                                           self.out_size_w * self.c_block_size +
                                                                           cut_h_index * self.out_size_w *
                                                                           self.c_block_size],
                                                        data_x_max[0], 0, 1, gm_max_burst_len, 0, 0)
                            self._remove_repeated_fun(mask_ub, fmap_img2col_cut_h, fmap_img2col_cut_w)
                        else:
                            # move max output to gm
                            gm_max_burst_len = int(fmap_img2col_cut_w)
                            self.tik_instance.data_move(self.output_max_gm[nc1_num * self.out_size_h *
                                                                           self.out_size_w * self.c_block_size +
                                                                           cut_h_index * self.out_size_w *
                                                                           self.c_block_size],
                                                        fmap_img2col_ub[0], 0, 1, gm_max_burst_len, 0, 0)
                            self._dup_mask_fun(mask_ub, mask_shape_ub)
                        with self.tik_instance.for_range(0, self.fmap_img2col_w) as w_index:
                            offset_output_mask = nc1_num * (self.fmap_img2col_h_num + 1) * self.fmap_img2col_w * \
                                                 self.c_block_size + cut_h_index * fmap_img2col_cut_h
                            self.tik_instance.data_move(self.output_mask_gm[offset_output_mask + w_index *
                                                                            (self.fmap_img2col_h_num + 1) * SCALAR_C0],
                                                        mask_ub[w_index * fmap_img2col_cut_w_num *
                                                                self.c_block_size],
                                                        0, 1, fmap_img2col_cut_w_num, 0, 0)
                with self.tik_instance.else_scope():
                    # copy input fmap from gm to l1
                    if self.in_size_h - cut_stride * (cut_h_num - 1) + self.pad[2] <= cut_h_size:
                        gm_l1_burst_len = int((self.in_size_h - cut_stride * (cut_h_num - 1) + self.pad[2]) *
                                              self.in_size_w * self.c_block_size // SCALAR_C0)
                        bottom = cut_h_size - (self.in_size_h - cut_stride * (cut_h_num - 1) + self.pad[2])
                    else:
                        gm_l1_burst_len = int(cut_h_size * self.in_size_w * self.c_block_size // SCALAR_C0)
                        bottom = 0

                    if self.is_support_load3d:
                        self.tik_instance.data_move(input_fmap_l1, self.input_fmap_gm[nc1_num * self.in_size_h *
                                                                                      self.in_size_w *
                                                                                      self.c_block_size +
                                                                                      (cut_h_index * cut_stride -
                                                                                      self.pad[2]) * self.in_size_w *
                                                                                      self.c_block_size],
                                                    0, 1, gm_l1_burst_len, 0, 0)
                    else:
                        pad_w = self.in_size_w + self.pad[0] + self.pad[1]
                        ori_ub_shape = (cut_h_size, pad_w, self.c_block_size)
                        ori_ub_input = self.tik_instance.Tensor(self.input_dtype, ori_ub_shape,
                                                                name="ori_ub_input",
                                                                scope=tik.scope_ubuf)
                        self._vector_dup(ori_ub_input, 0, ori_ub_shape, self.pad_value)

                        with self.tik_instance.for_range(0, (cut_h_size-bottom)) as row_index:
                            self.tik_instance.data_move(ori_ub_input[(self.pad[0] + row_index *
                                                                      pad_w) * self.c_block_size],
                                                        self.input_fmap_gm[nc1_num * self.in_size_h *
                                                                           self.in_size_w * self.c_block_size +
                                                                           (cut_h_index * cut_stride -
                                                                           self.pad[2]) * self.in_size_w *
                                                                           self.c_block_size + row_index *
                                                                           self.in_size_w * self.c_block_size],
                                                        0, 1, self.in_size_w, 0, 0)

                    with self.tik_instance.if_scope(cut_w_index != 0):
                        with self.tik_instance.if_scope(cut_w_index != (cut_w_num - 1)):
                            if not self.is_support_load3d:
                                self._vector_dup(fmap_img2col_ub, 0, fmap_img2col_ub.shape, self.pad_value)
                            with self.tik_instance.for_range(0, fmap_img2col_cut_w_num) as h_index:
                                if self.is_support_load3d:
                                    source_h = 0
                                    source_w = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) /
                                                (SCALAR_C0 * self.fmap_img2col_w)) % out_size_cut_w) * self.stride_w + \
                                            cut_w_stride * cut_w_index - self.pad[0]
                                    self.scalar_source_h.set_as(source_h)
                                    self.scalar_source_w.set_as(source_w)

                                    self.tik_instance.load3dv1(fmap_img2col_ub[h_index * SCALAR_C0 * SCALAR_C0 *
                                                                               self.fmap_img2col_w],
                                                               input_fmap_l1[0],
                                                               (self.pad[0], self.pad[1], 0, self.pad[3]),
                                                               (cut_h_size - self.pad[3]), self.in_size_w, 0, 0, 0,
                                                               self.scalar_source_w, self.scalar_source_h,
                                                               self.stride_w, self.stride_h, self.window_w,
                                                               self.window_h, 1, 1, 1, 0, self.fmap_img2col_w,
                                                               0, self.pad_value)
                                else:
                                    source_h = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w)
                                                // (SCALAR_C0 * self.fmap_img2col_w)) //
                                                self.out_size_w) * self.stride_h
                                    source_w = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) /
                                               (SCALAR_C0 * self.fmap_img2col_w)) % out_size_cut_w) * self.stride_w + \
                                               cut_w_stride * cut_w_index
                                    self.scalar_source_h.set_as(source_h)
                                    self.scalar_source_w.set_as(source_w)

                                    ori_ub_scalar = self.tik_instance.Scalar(dtype="int32")
                                    ori_ub_scalar.set_as(cut_h_size * pad_w * SCALAR_C0)
                                    self._img2col(h_index, fmap_img2col_ub, ori_ub_input, pad_w, ori_ub_scalar)

                            if self.fmap_img2col_w != 1:
                                self._calc_max_and_mask(fmap_img2col_cut_w_num, fmap_img2col_ub, data_x_max, mask_ub)
                                # move max output to gm
                                gm_max_burst_len = int(fmap_img2col_cut_w)
                                self.tik_instance.data_move(self.output_max_gm[nc1_num * self.out_size_h *
                                                                               self.out_size_w * self.c_block_size +
                                                                               cut_h_index * out_size_cut_h *
                                                                               self.out_size_w * self.c_block_size +
                                                                               cut_w_index * fmap_img2col_cut_w *
                                                                               self.c_block_size],
                                                            data_x_max[0], 0, 1, gm_max_burst_len, 0, 0)
                                self._remove_repeated_fun(mask_ub, fmap_img2col_cut_h, fmap_img2col_cut_w)
                            else:
                                # move max output to gm
                                gm_max_burst_len = int(fmap_img2col_cut_w)
                                self.tik_instance.data_move(self.output_max_gm[nc1_num * self.out_size_h *
                                                                               self.out_size_w * self.c_block_size +
                                                                               cut_h_index * out_size_cut_h *
                                                                               self.out_size_w * self.c_block_size +
                                                                               cut_w_index * fmap_img2col_cut_w *
                                                                               self.c_block_size],
                                                            fmap_img2col_ub[0], 0, 1, gm_max_burst_len, 0, 0)
                                self._dup_mask_fun(mask_ub, mask_shape_ub)
                            with self.tik_instance.for_range(0, self.fmap_img2col_w) as w_index:
                                offset_output_mask = nc1_num * (self.fmap_img2col_h_num + 1) * self.fmap_img2col_w * \
                                                     self.c_block_size + cut_h_index * fmap_img2col_cut_h + \
                                                     cut_w_index * fmap_img2col_cut_w
                                self.tik_instance.data_move(self.output_mask_gm[offset_output_mask + w_index *
                                                                                (self.fmap_img2col_h_num + 1) *
                                                                                SCALAR_C0],
                                                            mask_ub[w_index * fmap_img2col_cut_w_num
                                                                    * self.c_block_size],
                                                            0, 1, fmap_img2col_cut_w_num, 0, 0)
                        with self.tik_instance.else_scope():
                            cut_w_tail = self.in_size_w + self.pad[0] - cut_w_stride * (cut_w_num - 1)
                            if cut_w_tail > cut_w_size:
                                cut_w_tail = cut_w_size
                            out_size_tail_w = (cut_w_tail - self.window_w + self.stride_w + self.pad[1]) // \
                                              self.stride_w
                            fmap_img2col_tail_w = out_size_tail_w
                            fmap_img2col_tail_w_num = _ceil_div(fmap_img2col_tail_w, SCALAR_C0)

                            if not self.is_support_load3d:
                                self._vector_dup(fmap_img2col_ub, 0, fmap_img2col_ub.shape, self.pad_value)
                            with self.tik_instance.for_range(0, fmap_img2col_tail_w_num) as h_index:
                                if self.is_support_load3d:
                                    source_h = 0
                                    source_w = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) /
                                                (SCALAR_C0 * self.fmap_img2col_w)) % out_size_tail_w) * \
                                                self.stride_w + cut_w_stride * cut_w_index - self.pad[0]
                                    self.scalar_source_h.set_as(source_h)
                                    self.scalar_source_w.set_as(source_w)

                                    self.tik_instance.load3dv1(
                                        fmap_img2col_ub[h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w],
                                        input_fmap_l1[0], (self.pad[0], self.pad[1], 0, self.pad[3]),
                                        (cut_h_size - self.pad[3]), self.in_size_w, 0, 0, 0,
                                        self.scalar_source_w, self.scalar_source_h,
                                        self.stride_w, self.stride_h, self.window_w, self.window_h,
                                        1, 1, 1, 0, self.fmap_img2col_w, 0, self.pad_value)
                                else:
                                    source_h = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w)
                                                // (SCALAR_C0 * self.fmap_img2col_w)) //
                                                self.out_size_w) * self.stride_h
                                    source_w = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) /
                                                (SCALAR_C0 * self.fmap_img2col_w)) % out_size_tail_w) * \
                                                self.stride_w + cut_w_stride * cut_w_index
                                    self.scalar_source_h.set_as(source_h)
                                    self.scalar_source_w.set_as(source_w)

                                    ori_ub_scalar = self.tik_instance.Scalar(dtype="int32")
                                    ori_ub_scalar.set_as(cut_h_size * pad_w * SCALAR_C0)
                                    self._img2col(h_index, fmap_img2col_ub, ori_ub_input, pad_w, ori_ub_scalar)

                            if self.fmap_img2col_w != 1:
                                self._calc_max_and_mask(fmap_img2col_tail_w_num, fmap_img2col_ub, data_x_max, mask_ub,
                                                        fmap_img2col_cut_w_num)
                                # move max output to gm
                                gm_max_burst_len = int(fmap_img2col_tail_w)
                                self.tik_instance.data_move(self.output_max_gm[nc1_num * self.out_size_h *
                                                                               self.out_size_w * self.c_block_size +
                                                                               cut_h_index * out_size_cut_h *
                                                                               self.out_size_w * self.c_block_size +
                                                                               cut_w_index * fmap_img2col_cut_w *
                                                                               self.c_block_size],
                                                            data_x_max[0], 0, 1, gm_max_burst_len, 0, 0)
                                self._remove_repeated_fun(mask_ub, fmap_img2col_cut_h, fmap_img2col_tail_w,
                                                          fmap_img2col_cut_w)
                            else:
                                # move max output to gm
                                gm_max_burst_len = int(fmap_img2col_tail_w)
                                self.tik_instance.data_move(self.output_max_gm[nc1_num * self.out_size_h *
                                                                               self.out_size_w * self.c_block_size +
                                                                               cut_h_index * out_size_cut_h *
                                                                               self.out_size_w * self.c_block_size +
                                                                               cut_w_index * fmap_img2col_cut_w *
                                                                               self.c_block_size],
                                                            fmap_img2col_ub[0], 0, 1, gm_max_burst_len, 0, 0)
                                self._dup_mask_fun(mask_ub, mask_shape_ub)
                            mask_cut_w = fmap_img2col_tail_w_num * SCALAR_C0 - fmap_img2col_tail_w
                            mask_zero_w = 2 ** SCALAR_C0 - 2**(SCALAR_C0 - mask_cut_w)
                            with self.tik_instance.for_range(0, self.fmap_img2col_w) as w_index:
                                offset_output_mask = nc1_num * (self.fmap_img2col_h_num + 1) * self.fmap_img2col_w * \
                                                     self.c_block_size + cut_h_index * fmap_img2col_cut_h + \
                                                     cut_w_index * fmap_img2col_cut_w
                                if mask_zero_w != 0 and self.fmap_img2col_w != 1:
                                    self.tik_instance.vector_dup([0, mask_zero_w],
                                                                 mask_ub[w_index * fmap_img2col_cut_w_num *
                                                                         self.c_block_size + fmap_img2col_tail_w_num *
                                                                         SCALAR_C0 - SCALAR_C0],
                                                                 0, 1, 1, 8)

                                self.tik_instance.data_move(self.output_mask_gm[offset_output_mask + w_index *
                                                                                (self.fmap_img2col_h_num + 1) *
                                                                                SCALAR_C0],
                                                            mask_ub[w_index * fmap_img2col_cut_w_num *
                                                                    self.c_block_size],
                                                            0, 1, fmap_img2col_tail_w_num, 0, 0)
                    with self.tik_instance.else_scope():
                        if not self.is_support_load3d:
                            self._vector_dup(fmap_img2col_ub, 0, fmap_img2col_ub.shape, self.pad_value)
                        with self.tik_instance.for_range(0, fmap_img2col_cut_w_num) as h_index:
                            if self.is_support_load3d:
                                source_h = 0
                                source_w = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) /
                                            (SCALAR_C0 * self.fmap_img2col_w)) % out_size_cut_w) * \
                                            self.stride_w - self.pad[0]
                                self.scalar_source_h.set_as(source_h)
                                self.scalar_source_w.set_as(source_w)

                                self.tik_instance.load3dv1(fmap_img2col_ub[h_index * SCALAR_C0 * SCALAR_C0 *
                                                                           self.fmap_img2col_w],
                                                           input_fmap_l1[0], (self.pad[0], self.pad[1], 0, self.pad[3]),
                                                           (cut_h_size - self.pad[3]), self.in_size_w, 0, 0, 0,
                                                           self.scalar_source_w, self.scalar_source_h,
                                                           self.stride_w, self.stride_h, self.window_w, self.window_h,
                                                           1, 1, 1, 0, self.fmap_img2col_w, 0, self.pad_value)
                            else:
                                source_h = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w)
                                                // (SCALAR_C0 * self.fmap_img2col_w)) //
                                                self.out_size_w) * self.stride_h
                                source_w = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) /
                                            (SCALAR_C0 * self.fmap_img2col_w)) % out_size_cut_w) * \
                                            self.stride_w
                                self.scalar_source_h.set_as(source_h)
                                self.scalar_source_w.set_as(source_w)

                                ori_ub_scalar = self.tik_instance.Scalar(dtype="int32")
                                ori_ub_scalar.set_as(cut_h_size * pad_w * SCALAR_C0)
                                self._img2col(h_index, fmap_img2col_ub, ori_ub_input, pad_w, ori_ub_scalar)

                        if self.fmap_img2col_w != 1:
                            self._calc_max_and_mask(fmap_img2col_cut_w_num, fmap_img2col_ub, data_x_max, mask_ub)
                            # move max output to gm
                            gm_max_burst_len = int(fmap_img2col_cut_w)
                            self.tik_instance.data_move(self.output_max_gm[nc1_num * self.out_size_h *
                                                                           self.out_size_w * self.c_block_size +
                                                                           cut_h_index * self.out_size_w *
                                                                           self.c_block_size],
                                                        data_x_max[0], 0, 1, gm_max_burst_len, 0, 0)
                            self._remove_repeated_fun(mask_ub, fmap_img2col_cut_h, fmap_img2col_cut_w)
                        else:
                            # move max output to gm
                            gm_max_burst_len = int(fmap_img2col_cut_w)
                            self.tik_instance.data_move(self.output_max_gm[nc1_num * self.out_size_h *
                                                                           self.out_size_w * self.c_block_size +
                                                                           cut_h_index * self.out_size_w *
                                                                           self.c_block_size],
                                                        fmap_img2col_ub[0], 0, 1, gm_max_burst_len, 0, 0)
                            self._dup_mask_fun(mask_ub, mask_shape_ub)
                        with self.tik_instance.for_range(0, self.fmap_img2col_w) as w_index:
                            offset_output_mask = nc1_num * (self.fmap_img2col_h_num + 1) * \
                                                 self.fmap_img2col_w * self.c_block_size + \
                                                 cut_h_index * fmap_img2col_cut_h
                            self.tik_instance.data_move(self.output_mask_gm[offset_output_mask + w_index *
                                                                            (self.fmap_img2col_h_num + 1) * SCALAR_C0],
                                                        mask_ub[w_index * fmap_img2col_cut_w_num * self.c_block_size],
                                                        0, 1, fmap_img2col_cut_w_num, 0, 0)
            with self.tik_instance.else_scope():
                if self.is_support_load3d:
                    # copy input fmap from gm to l1
                    gm_l1_burst_len = int((cut_h_size - self.pad[2]) *
                                           self.in_size_w * self.c_block_size // SCALAR_C0)
                    self.tik_instance.data_move(input_fmap_l1,
                                                self.input_fmap_gm[nc1_num * self.in_size_h * self.in_size_w *
                                                                   self.c_block_size],
                                                0, 1, gm_l1_burst_len, 0, 0)
                else:
                    pad_w = self.in_size_w + self.pad[0] + self.pad[1]
                    ori_ub_shape = (cut_h_size, pad_w, self.c_block_size)
                    ori_ub_input = self.tik_instance.Tensor(self.input_dtype, ori_ub_shape,
                                                            name="ori_ub_input",
                                                            scope=tik.scope_ubuf)
                    self._vector_dup(ori_ub_input, 0, ori_ub_shape, self.pad_value)

                    with self.tik_instance.for_range(0, (cut_h_size-self.pad[2])) as row_index:
                        self.tik_instance.data_move(ori_ub_input[(self.pad[2] * pad_w + self.pad[0] + row_index *
                                                                  pad_w) * self.c_block_size],
                                                    self.input_fmap_gm[nc1_num * self.in_size_h * self.in_size_w *
                                                                       self.c_block_size + row_index * self.in_size_w *
                                                                       self.c_block_size],
                                                    0, 1, self.in_size_w, 0, 0)

                with self.tik_instance.if_scope(cut_w_index != 0):
                    with self.tik_instance.if_scope(cut_w_index != (cut_w_num - 1)):
                        if not self.is_support_load3d:
                            self._vector_dup(fmap_img2col_ub, 0, fmap_img2col_ub.shape, self.pad_value)
                        with self.tik_instance.for_range(0, fmap_img2col_cut_w_num) as h_index:
                            if self.is_support_load3d:
                                source_h = -self.pad[2]
                                source_w = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) /
                                            (SCALAR_C0 * self.fmap_img2col_w)) % out_size_cut_w) * \
                                        self.stride_w + cut_w_stride * cut_w_index - self.pad[0]
                                self.scalar_source_h.set_as(source_h)
                                self.scalar_source_w.set_as(source_w)

                                self.tik_instance.load3dv1(
                                    fmap_img2col_ub[h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w],
                                    input_fmap_l1[0], (self.pad[0], self.pad[1], self.pad[2], 0),
                                    (cut_h_size - self.pad[2]), self.in_size_w, 0, 0, 0,
                                    self.scalar_source_w, self.scalar_source_h, self.stride_w, self.stride_h,
                                    self.window_w, self.window_h, 1, 1, 1, 0, self.fmap_img2col_w, 0, self.pad_value)
                            else:
                                source_h = 0
                                source_w = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) /
                                            (SCALAR_C0 * self.fmap_img2col_w)) % out_size_cut_w) * \
                                        self.stride_w + cut_w_stride * cut_w_index
                                self.scalar_source_h.set_as(source_h)
                                self.scalar_source_w.set_as(source_w)

                                ori_ub_scalar = self.tik_instance.Scalar(dtype="int32")
                                ori_ub_scalar.set_as(cut_h_size * pad_w * SCALAR_C0)
                                self._img2col(h_index, fmap_img2col_ub, ori_ub_input, pad_w, ori_ub_scalar)

                        if self.fmap_img2col_w != 1:
                            self._calc_max_and_mask(fmap_img2col_cut_w_num, fmap_img2col_ub, data_x_max, mask_ub)
                            # move max output to gm
                            gm_max_burst_len = int(fmap_img2col_cut_w)
                            self.tik_instance.data_move(self.output_max_gm[nc1_num * self.out_size_h *
                                                                           self.out_size_w * self.c_block_size +
                                                                           cut_w_index * fmap_img2col_cut_w *
                                                                           self.c_block_size],
                                                        data_x_max[0], 0, 1, gm_max_burst_len, 0, 0)
                            self._remove_repeated_fun(mask_ub, fmap_img2col_cut_h, fmap_img2col_cut_w)
                        else:
                            # move max output to gm
                            gm_max_burst_len = int(fmap_img2col_cut_w)
                            self.tik_instance.data_move(self.output_max_gm[nc1_num * self.out_size_h *
                                                                           self.out_size_w * self.c_block_size +
                                                                           cut_w_index * fmap_img2col_cut_w *
                                                                           self.c_block_size],
                                                        fmap_img2col_ub[0], 0, 1, gm_max_burst_len, 0, 0)
                            self._dup_mask_fun(mask_ub, mask_shape_ub)
                        with self.tik_instance.for_range(0, self.fmap_img2col_w) as w_index:
                            offset_output_mask = nc1_num * (self.fmap_img2col_h_num + 1) * self.fmap_img2col_w * \
                                                 self.c_block_size + cut_h_index * fmap_img2col_cut_h + cut_w_index * \
                                                 fmap_img2col_cut_w
                            self.tik_instance.data_move(self.output_mask_gm[offset_output_mask + w_index *
                                                                            (self.fmap_img2col_h_num + 1) * SCALAR_C0],
                                                        mask_ub[w_index * fmap_img2col_cut_w_num * self.c_block_size],
                                                        0, 1, fmap_img2col_cut_w_num, 0, 0)
                    with self.tik_instance.else_scope():
                        cut_w_tail = self.in_size_w + self.pad[0] - cut_w_stride * (cut_w_num - 1)
                        if cut_w_tail > cut_w_size:
                            cut_w_tail = cut_w_size
                        out_size_tail_w = (cut_w_tail - self.window_w + self.stride_w + self.pad[1]) // self.stride_w
                        fmap_img2col_tail_w = out_size_tail_w
                        fmap_img2col_tail_w_num = _ceil_div(fmap_img2col_tail_w, SCALAR_C0)

                        if not self.is_support_load3d:
                            self._vector_dup(fmap_img2col_ub, 0, fmap_img2col_ub.shape, self.pad_value)
                        with self.tik_instance.for_range(0, fmap_img2col_tail_w_num) as h_index:
                            if self.is_support_load3d:
                                source_h = -self.pad[2]
                                source_w = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) /
                                            (SCALAR_C0 * self.fmap_img2col_w)) % out_size_tail_w) * \
                                            self.stride_w + cut_w_stride * cut_w_index - self.pad[0]
                                self.scalar_source_h.set_as(source_h)
                                self.scalar_source_w.set_as(source_w)

                                self.tik_instance.load3dv1(fmap_img2col_ub[h_index * SCALAR_C0 * SCALAR_C0 *
                                                                           self.fmap_img2col_w],
                                                           input_fmap_l1[0], (self.pad[0], self.pad[1], self.pad[2], 0),
                                                           (cut_h_size - self.pad[2]), self.in_size_w, 0, 0, 0,
                                                           self.scalar_source_w, self.scalar_source_h,
                                                           self.stride_w, self.stride_h, self.window_w, self.window_h,
                                                           1, 1, 1, 0, self.fmap_img2col_w, 0, self.pad_value)
                            else:
                                source_h = 0
                                source_w = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) /
                                            (SCALAR_C0 * self.fmap_img2col_w)) % out_size_tail_w) * \
                                            self.stride_w + cut_w_stride * cut_w_index
                                self.scalar_source_h.set_as(source_h)
                                self.scalar_source_w.set_as(source_w)

                                ori_ub_scalar = self.tik_instance.Scalar(dtype="int32")
                                ori_ub_scalar.set_as(cut_h_size * pad_w * SCALAR_C0)
                                self._img2col(h_index, fmap_img2col_ub, ori_ub_input, pad_w, ori_ub_scalar)

                        if self.fmap_img2col_w != 1:
                            self._calc_max_and_mask(fmap_img2col_tail_w_num, fmap_img2col_ub, data_x_max,
                                                    mask_ub, fmap_img2col_cut_w_num)
                            # move max output to gm
                            gm_max_burst_len = int(fmap_img2col_tail_w)
                            self.tik_instance.data_move(self.output_max_gm[nc1_num * self.out_size_h *
                                                                           self.out_size_w * self.c_block_size +
                                                                           cut_w_index * fmap_img2col_cut_w *
                                                                           self.c_block_size],
                                                        data_x_max[0], 0, 1, gm_max_burst_len, 0, 0)
                            self._remove_repeated_fun(mask_ub, fmap_img2col_cut_h, fmap_img2col_tail_w,
                                                      fmap_img2col_cut_w)
                        else:
                            # move max output to gm
                            gm_max_burst_len = int(fmap_img2col_tail_w)
                            self.tik_instance.data_move(self.output_max_gm[nc1_num * self.out_size_h *
                                                                           self.out_size_w * self.c_block_size +
                                                                           cut_w_index * fmap_img2col_cut_w *
                                                                           self.c_block_size],
                                                        fmap_img2col_ub[0], 0, 1, gm_max_burst_len, 0, 0)
                            self._dup_mask_fun(mask_ub, mask_shape_ub)
                        mask_cut_w = fmap_img2col_tail_w_num * SCALAR_C0 - fmap_img2col_tail_w
                        mask_zero_w = 2 ** SCALAR_C0 - 2**(SCALAR_C0 - mask_cut_w)
                        with self.tik_instance.for_range(0, self.fmap_img2col_w) as w_index:
                            offset_output_mask = nc1_num * (self.fmap_img2col_h_num + 1) * self.fmap_img2col_w * \
                                                 self.c_block_size + cut_h_index * fmap_img2col_cut_h + \
                                                 cut_w_index * fmap_img2col_cut_w
                            if mask_zero_w != 0 and self.fmap_img2col_w != 1 and cut_h_num == 1:
                                self.tik_instance.vector_dup([0, mask_zero_w],
                                                             mask_ub[w_index * fmap_img2col_cut_w_num *
                                                                     self.c_block_size + fmap_img2col_tail_w_num *
                                                                     SCALAR_C0 - SCALAR_C0],
                                                             0, 1, 1, 8)

                            self.tik_instance.data_move(self.output_mask_gm[offset_output_mask + w_index *
                                                                            (self.fmap_img2col_h_num + 1) * SCALAR_C0],
                                                        mask_ub[w_index * fmap_img2col_cut_w_num * self.c_block_size],
                                                        0, 1, fmap_img2col_tail_w_num, 0, 0)
                with self.tik_instance.else_scope():
                    if not self.is_support_load3d:
                        self._vector_dup(fmap_img2col_ub, 0, fmap_img2col_ub.shape, self.pad_value)
                    with self.tik_instance.for_range(0, fmap_img2col_cut_w_num) as h_index:
                        if self.is_support_load3d:
                            source_h = -self.pad[2]
                            source_w = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) /
                                        (SCALAR_C0 * self.fmap_img2col_w)) % out_size_cut_w) * \
                                        self.stride_w - self.pad[0]
                            self.scalar_source_h.set_as(source_h)
                            self.scalar_source_w.set_as(source_w)

                            self.tik_instance.load3dv1(fmap_img2col_ub[h_index * SCALAR_C0 * SCALAR_C0 *
                                                                       self.fmap_img2col_w],
                                                       input_fmap_l1[0], (self.pad[0], self.pad[1], self.pad[2], 0),
                                                       (cut_h_size - self.pad[2]), self.in_size_w, 0, 0, 0,
                                                       self.scalar_source_w, self.scalar_source_h,
                                                       self.stride_w, self.stride_h, self.window_w, self.window_h,
                                                       1, 1, 1, 0, self.fmap_img2col_w, 0, self.pad_value)
                        else:
                            source_h = 0
                            source_w = (((h_index * SCALAR_C0 * SCALAR_C0 * self.fmap_img2col_w) /
                                        (SCALAR_C0 * self.fmap_img2col_w)) % out_size_cut_w) * self.stride_w
                            self.scalar_source_h.set_as(source_h)
                            self.scalar_source_w.set_as(source_w)

                            ori_ub_scalar = self.tik_instance.Scalar(dtype="int32")
                            ori_ub_scalar.set_as(cut_h_size * pad_w * SCALAR_C0)
                            self._img2col(h_index, fmap_img2col_ub, ori_ub_input, pad_w, ori_ub_scalar)

                    if self.fmap_img2col_w != 1:
                        self._calc_max_and_mask(fmap_img2col_cut_w_num, fmap_img2col_ub, data_x_max, mask_ub)
                        # move max output to gm
                        gm_max_burst_len = int(fmap_img2col_cut_w)
                        self.tik_instance.data_move(self.output_max_gm[nc1_num * self.out_size_h * self.out_size_w *
                                                                       self.c_block_size],
                                                    data_x_max[0], 0, 1, gm_max_burst_len, 0, 0)
                        self._remove_repeated_fun(mask_ub, fmap_img2col_cut_h, fmap_img2col_cut_w)
                    else:
                        # move max output to gm
                        gm_max_burst_len = int(fmap_img2col_cut_w)
                        self.tik_instance.data_move(self.output_max_gm[nc1_num * self.out_size_h * self.out_size_w *
                                                                       self.c_block_size],
                                                    fmap_img2col_ub[0], 0, 1, gm_max_burst_len, 0, 0)
                        self._dup_mask_fun(mask_ub, mask_shape_ub)
                    with self.tik_instance.for_range(0, self.fmap_img2col_w) as w_index:
                        offset_output_mask = nc1_num * (self.fmap_img2col_h_num + 1) * \
                                             self.fmap_img2col_w * self.c_block_size + \
                                             cut_h_index * fmap_img2col_cut_h
                        self.tik_instance.data_move(self.output_mask_gm[offset_output_mask + w_index *
                                                                        (self.fmap_img2col_h_num + 1) * SCALAR_C0],
                                                    mask_ub[w_index * fmap_img2col_cut_w_num * self.c_block_size],
                                                    0, 1, fmap_img2col_cut_w_num, 0, 0)

    # 'pylint: disable=too-many-statements,too-many-branches
    def _fun_need_cut_h_w(self, block_index, nc1_cuth_index, cut_h_size,
                          cut_stride, cut_h_num, nc1_cuth_size, flag_cut_h):
        """
        funtion need cut H and W while l1 not enough

        Parameters
        ----------
        block_index: index of block
        nc1_index: index of nc1
        cut_h_size: size of cuth
        cut_stride: stride of cuth
        cut_h_num: number of cuth
        flag_cut_h:bool

        Returns
        -------
        none
        """
        if flag_cut_h:
            cut_h_index = (block_index * nc1_cuth_size + nc1_cuth_index) % cut_h_num
            nc1_num = (block_index * nc1_cuth_size + nc1_cuth_index) // cut_h_num
            self._calc_need_cut_h_w(nc1_num, cut_h_size, cut_h_num, cut_h_index, cut_stride)
        else:
            nc1_num = block_index * nc1_cuth_size + nc1_cuth_index
            with self.tik_instance.for_range(0, cut_h_num) as cut_h_index:
                self._calc_need_cut_h_w(nc1_num, cut_h_size, cut_h_num, cut_h_index, cut_stride)

    def _calc_out_size_and_pad(self):
        """
        caculate output size and padding size

        Parameters
        ----------
        none

        Returns
        -------
        pad: include pad_t, pad_b, pad_l, pad_r
        out_size_h: out_size in h direction
        out_size_w: out_size in w direction
        """
        # pad_l, pad_r, pad_t, pad_b is for pad on the left, right, top, bottom
        pad_l, pad_r, pad_t, pad_b = 0, 0, 0, 0

        if self.padding == "SAME":
            # `Hout = ceil(Hi, Sh), Wout = ceil(Wi, Sw)`
            out_size_h = (self.in_size_h + self.stride_h - 1) // self.stride_h
            out_size_w = (self.in_size_w + self.stride_w - 1) // self.stride_w

            # get total pad rows or pad columns
            pad_rows = (out_size_h - 1) * self.stride_h + ((self.window_h - 1) * DILATION + 1) - self.in_size_h
            pad_cols = (out_size_w - 1) * self.stride_w + ((self.window_w - 1) * DILATION + 1) - self.in_size_w

            # pad_rows and pad_columns is odd or even number
            if pad_rows % 2 == 0:
                pad_t = pad_rows // 2
                pad_b = pad_rows // 2
            else:
                pad_t = pad_rows // 2
                pad_b = pad_rows - pad_t

            if pad_cols % 2 == 0:
                pad_l = pad_cols // 2
                pad_r = pad_cols // 2
            else:
                pad_l = pad_cols // 2
                pad_r = pad_cols - pad_l

            if pad_t < 0:
                pad_t = 0

            if pad_b < 0:
                pad_b = 0

            if pad_l < 0:
                pad_l = 0

            if pad_r < 0:
                pad_r = 0

        # caculate output size in VALID mode
        if self.padding == "VALID":
            # `Hout = ceil(Hi - Fh + 1, Sh), Wout = ceil(Wi - Fw + 1, Sw)`
            out_size_h = (self.in_size_h - self.window_h + 1 + (self.stride_h - 1)) // self.stride_h
            out_size_w = (self.in_size_w - self.window_w + 1 + (self.stride_w - 1)) // self.stride_w
        pad = (pad_l, pad_r, pad_t, pad_b)

        return pad, out_size_h, out_size_w

    # 'pylint: disable=too-many-branches
    def _calc_cut_h_size_fun(self, need_cut=False):
        """
        caculate cut_h size

        Parameters
        ----------
        need_cut :bool
            if need cut

        Returns
        -------
        cut_h_size: cut size
        cut_stride: cut stride
        fh_loop: loop number
        """
        img2col_w = self.window_h * self.window_w * SCALAR_C0
        if self.is_support_load3d:
            img2col_h = UB_SIZE / 2 / (img2col_w * 2 + (32 * 5))
        else:
            img2col_h = UB_SIZE / 2 / 2 / (img2col_w * 2 + (32 * 5))
        if self.window_h >= self.stride_h:
            cut_h_size = ((img2col_h // (((self.in_size_w + self.pad[0] + self.pad[1])) // self.stride_w + 1)) - 1) * \
                         self.stride_h + self.window_h - self.stride_h
            if cut_h_size < self.window_h or cut_h_size >= self.in_size_h:
                cut_h_size = self.window_h
            cut_stride = cut_h_size - (self.window_h - self.stride_h)
        else:
            cut_h_size = ((img2col_h // (((self.in_size_w + self.pad[0] + self.pad[1])) // self.stride_w + 1)) - 1) * \
                         self.stride_h
            if cut_h_size < self.window_h:
                cut_h_size = self.window_h
                cut_stride = self.stride_h
            else:
                cut_stride = cut_h_size

        if cut_h_size >= cut_stride:
            fh_loop = _ceil_div(((self.in_size_h + self.pad[2] + self.pad[3]) - cut_h_size), cut_stride) + 1
        else:
            if (self.in_size_h + self.pad[2] + self.pad[3]) % cut_stride == 0:
                fh_loop = (self.in_size_h + self.pad[2] + self.pad[3]) // cut_stride
            else:
                fh_loop = _ceil_div((self.in_size_h + self.pad[2] + self.pad[3]), cut_stride)

        if cut_h_size * self.in_size_w * self.c_block_size * 2 > L1_SIZE:
            need_cut = True

        if need_cut:
            cut_h_size = self.window_h
            cut_stride = self.stride_h
            if cut_h_size >= cut_stride:
                fh_loop = _ceil_div(((self.in_size_h + self.pad[2] + self.pad[3]) - cut_h_size), cut_stride) + 1
            else:
                if (self.in_size_h + self.pad[2] + self.pad[3]) % cut_stride == 0:
                    fh_loop = (self.in_size_h + self.pad[2] + self.pad[3]) // cut_stride
                else:
                    fh_loop = _ceil_div((self.in_size_h + self.pad[2] + self.pad[3]), cut_stride)

        if self.padding == "VALID":
            if cut_h_size == self.window_h:
                if cut_h_size >= cut_stride:
                    fh_loop = (self.in_size_h - cut_h_size + cut_stride) // cut_stride
                else:
                    fh_loop = self.in_size_h // cut_stride
                    if (self.in_size_h - cut_stride * fh_loop) >= self.window_h:
                        fh_loop = fh_loop + 1
            else:
                if cut_h_size >= cut_stride:
                    fh_loop = (self.in_size_h - cut_h_size + cut_stride) // cut_stride
                else:
                    fh_loop = self.in_size_h // cut_stride
                if (self.in_size_h - cut_stride * fh_loop) >= self.window_h:
                    fh_loop = fh_loop + 1

        return int(cut_h_size), int(cut_stride), int(fh_loop)

    def _calc_cut_w_size_fun(self):
        """
        caculate cut_w size

        Parameters
        ----------
        none

        Returns
        -------
        cut_w_size: cut size
        cut_w_stride: cut stride
        fw_loop: loop number
        """
        img2col_w = self.window_h * self.window_w * SCALAR_C0
        if self.is_support_load3d:
            img2col_h = UB_SIZE / 2 / (img2col_w * 2 + (32 * 5))
        else:
            img2col_h = UB_SIZE / 2 / 2 / (img2col_w * 2 + (32 * 5))
        if self.window_w >= self.stride_w:
            cut_w_size = (img2col_h // 1 - 1) * self.stride_w + self.window_w - self.stride_w
            cut_w_stride = cut_w_size - (self.window_w - self.stride_w)
        else:
            cut_w_size = (img2col_h // 1 - 1) * self.stride_w
            cut_w_stride = cut_w_size

        if cut_w_size < self.window_w:
            expected_value = "smaller than supported value"
            real_value = "greater than supported value"
            error_manager_vector.raise_err_input_value_invalid("max_pool_with_argmax",
                                                               "ksize or input shape",
                                                               expected_value, real_value)

        if cut_w_size >= cut_w_stride:
            fw_loop = _ceil_div(((self.in_size_w + self.pad[0] + self.pad[1]) - cut_w_size), cut_w_stride) + 1
        else:
            if (self.in_size_w + self.pad[0] + self.pad[1]) % cut_w_stride == 0:
                fw_loop = (self.in_size_w + self.pad[0] + self.pad[1]) // cut_w_stride
            else:
                fw_loop = _ceil_div((self.in_size_w + self.pad[0] + self.pad[1]), cut_w_stride)

        if self.padding == "VALID":
            if cut_w_size == self.window_w:
                if cut_w_size >= cut_w_stride:
                    fw_loop = (self.in_size_w - cut_w_size + cut_w_stride) // cut_w_stride
                else:
                    fw_loop = self.in_size_w // cut_w_size
            else:
                if cut_w_size >= cut_w_stride:
                    fw_loop = (self.in_size_w - cut_w_size + cut_w_stride) // cut_w_stride
                else:
                    fw_loop = self.in_size_w // cut_w_stride
                if (self.in_size_w - cut_w_stride * fw_loop) >= self.window_w:
                    fw_loop = fw_loop + 1

        return int(cut_w_size), int(cut_w_stride), int(fw_loop)

    def _calc_max_fun(self, data_input, data_input_ub, index_w, index_h):
        """
        caculate max of data_input

        Parameters
        ----------
        data_input: input data
        data_input_ub: input data in ub
        index_w: input size in w direction
        index_h: input size in h direction

        Returns
        -------
        data_input: output tensor
        """
        self.tik_instance.vmax(MASK, data_input[index_h * SCALAR_C0 * SCALAR_C0],
                               data_input[index_h * SCALAR_C0 * SCALAR_C0],
                               data_input_ub[index_w * SCALAR_C0 * SCALAR_C0 + index_h *
                                             self.fmap_img2col_w * SCALAR_C0 * SCALAR_C0],
                               REPEAT_2, DSTSTRIDEM0, SRC0STRIDEM0, SRC1STRIDEM0, DSTSTRIDEM1,
                               SRC0STRIDEM1, SRC1STRIDEM1)
        return data_input

    def _calc_mask_fun(self, data_input_max, data_input_ub, index_w, index_h,
                       fmap_h_num, mask_ub):
        """
        caculate mask of data_input_max

        Parameters
        ----------
        data_input_max: max value in input data
        data_input_ub: input data in ub
        index_w: index of w
        index_h: index of h
        fmap_h_num: num of fmap in h
        mask_ub: mask in ub

        Returns
        -------
        mask_ub: mask in ub
        """
        self.tik_instance.vcmpv_eq(mask_ub[index_w * fmap_h_num * SCALAR_C0 + index_h * SCALAR_C0],
                                   data_input_ub[index_w * SCALAR_C0 * SCALAR_C0 + index_h *
                                                 self.fmap_img2col_w * SCALAR_C0 * SCALAR_C0],
                                   data_input_max[index_h * SCALAR_C0 * SCALAR_C0], REPEAT_2,
                                   SRC0STRIDEM0, SRC1STRIDEM0, SRC0STRIDEM1,
                                   SRC1STRIDEM1)
        return mask_ub

    def _calc_max_and_mask(self, fmap_h_num, fmap_img2col_ub, data_x_max, mask_ub, fmap_img2col_cut_w_num=0,
                           fmap_h_tail_num=0):
        """
        caculate max and mask of data_input

        Parameters
        ----------
        fmap_h_num: num of fmap_img2col_h
        fmap_img2col_ub: fmap in ub
        data_x_max: max value in input data
        mask_ub: mask in ub
        fmap_img2col_cut_w_num: cut number of w
        fmap_h_tail_num: num of h tail

        Returns
        -------
        data_input_ub: output tensor
        """
        scalar_repeat_times = int(fmap_h_num * 2)
        repeat_times = _ceil_div(scalar_repeat_times, SCALAR_255)
        # dup 8*blocks init 1 into a buffer:
        if scalar_repeat_times > SCALAR_255:
            with self.tik_instance.for_range(0, repeat_times) as repeat_index:
                with self.tik_instance.if_scope(repeat_index != (repeat_times - 1)):
                    self.tik_instance.vector_dup(MASK, data_x_max[repeat_index * SCALAR_255 * MASK],
                                                 MIN_VALUE_FP16, SCALAR_255, DSTSTRIDEM0, SRC0STRIDEM1)
                with self.tik_instance.else_scope():
                    self.tik_instance.vector_dup(MASK, data_x_max[repeat_index * SCALAR_255 * MASK],
                                                 MIN_VALUE_FP16, (scalar_repeat_times - repeat_index * SCALAR_255),
                                                 DSTSTRIDEM0, SRC0STRIDEM1)
        else:
            self.tik_instance.vector_dup(MASK, data_x_max, MIN_VALUE_FP16, scalar_repeat_times, DSTSTRIDEM0,
                                         SRC0STRIDEM1)
        # do max
        with self.tik_instance.for_range(0, self.fmap_img2col_w) as index_w:
            with self.tik_instance.for_range(0, fmap_h_num) as index_h:
                # the first 128
                data_x_max = self._calc_max_fun(data_x_max, fmap_img2col_ub, index_w, index_h)
        # do mask
        with self.tik_instance.for_range(0, self.fmap_img2col_w) as index_w:
            with self.tik_instance.for_range(0, fmap_h_num) as index_h:
                if fmap_img2col_cut_w_num == 0:
                    if fmap_h_tail_num == 0:
                        mask_ub = self._calc_mask_fun(data_x_max, fmap_img2col_ub, index_w, index_h,
                                                      fmap_h_num, mask_ub)
                    else:
                        mask_ub = self._calc_mask_fun(data_x_max, fmap_img2col_ub, index_w, index_h,
                                                      fmap_h_tail_num, mask_ub)
                else:
                    mask_ub = self._calc_mask_fun(data_x_max, fmap_img2col_ub, index_w, index_h,
                                                  fmap_img2col_cut_w_num, mask_ub)

    def _remove_repeated_fun(self, mask_ub, fmap_img2col_cut_h=0, fmap_img2col_cut_w=0, fmap_img2col_tail_w=0,
                             fmap_img2col_tail_h=0):
        """
        caculate max and mask of data_input

        Parameters
        ----------
        fmap_img2col_h: size of fmap_img2col_h
        mask_ub: mask in ub
        fmap_img2col_cut_h: size of fmap_img2col_cut_h
        fmap_img2col_cut_w: size of fmap_img2col_cut_w
        fmap_img2col_tail_w: size of fmap_img2col_tail_w
        fmap_img2col_tail_h: size of tail_h

        Returns
        -------
        data_input_ub: output tensor
        """
        if fmap_img2col_cut_h != 0:
            if fmap_img2col_cut_w != 0:
                fmap_img2col_h_num = _ceil_div(fmap_img2col_cut_w, SCALAR_C0)
            else:
                fmap_img2col_h_num = _ceil_div(fmap_img2col_cut_h, SCALAR_C0)
        else:
            fmap_img2col_h_num = _ceil_div(self.fmap_img2col_h, SCALAR_C0)

        mask_or_shape_ub = (fmap_img2col_h_num, SCALAR_C0)
        mask_or = self.tik_instance.Tensor(
            "uint16", mask_or_shape_ub, name="mask_or", scope=tik.scope_ubuf)
        mask_not = self.tik_instance.Tensor(
            "uint16", mask_or_shape_ub, name="mask_not", scope=tik.scope_ubuf)
        with self.tik_instance.for_range(0, self.fmap_img2col_w) as index_w:
            with self.tik_instance.if_scope(index_w > 0):
                if fmap_img2col_tail_w == 0:
                    if fmap_img2col_tail_h == 0:
                        self.tik_instance.vor(SCALAR_C0, mask_or[0],
                                              mask_ub[index_w * fmap_img2col_h_num * SCALAR_C0],
                                              mask_or[0], fmap_img2col_h_num, DSTSTRIDEM0, SRC0STRIDEM0,
                                              SRC1STRIDEM0, DSTSTRIDEM0, SRC0STRIDEM0, SRC1STRIDEM0)
                        self.tik_instance.vand(SCALAR_C0, mask_ub[index_w * fmap_img2col_h_num * SCALAR_C0],
                                               mask_not[0], mask_ub[index_w * fmap_img2col_h_num * SCALAR_C0],
                                               fmap_img2col_h_num, DSTSTRIDEM0, SRC0STRIDEM0,
                                               SRC1STRIDEM0, DSTSTRIDEM0, SRC0STRIDEM0, SRC1STRIDEM0)
                    else:
                        fmap_img2col_tail_num = _ceil_div(fmap_img2col_tail_h, SCALAR_C0)
                        self.tik_instance.vor(SCALAR_C0, mask_or[0],
                                              mask_ub[index_w * fmap_img2col_tail_num * SCALAR_C0],
                                              mask_or[0], fmap_img2col_h_num, DSTSTRIDEM0, SRC0STRIDEM0,
                                              SRC1STRIDEM0, DSTSTRIDEM0, SRC0STRIDEM0, SRC1STRIDEM0)
                        self.tik_instance.vand(SCALAR_C0,
                                               mask_ub[index_w * fmap_img2col_tail_num * SCALAR_C0],
                                               mask_not[0],
                                               mask_ub[index_w * fmap_img2col_tail_num * SCALAR_C0],
                                               fmap_img2col_h_num, DSTSTRIDEM0, SRC0STRIDEM0, SRC1STRIDEM0,
                                               DSTSTRIDEM0, SRC0STRIDEM0, SRC1STRIDEM0)
                else:
                    fmap_img2col_tail_num = _ceil_div(fmap_img2col_tail_w, SCALAR_C0)
                    self.tik_instance.vor(SCALAR_C0, mask_or[0],
                                          mask_ub[index_w * fmap_img2col_tail_num * SCALAR_C0], mask_or[0],
                                          fmap_img2col_h_num, DSTSTRIDEM0, SRC0STRIDEM0,
                                          SRC1STRIDEM0, DSTSTRIDEM0, SRC0STRIDEM0, SRC1STRIDEM0)
                    self.tik_instance.vand(SCALAR_C0,
                                           mask_ub[index_w * fmap_img2col_tail_num * SCALAR_C0],
                                           mask_not[0], mask_ub[index_w * fmap_img2col_tail_num * SCALAR_C0],
                                           fmap_img2col_h_num, DSTSTRIDEM0, SRC0STRIDEM0,
                                           SRC1STRIDEM0, DSTSTRIDEM0, SRC0STRIDEM0, SRC1STRIDEM0)
                self.tik_instance.vnot(SCALAR_C0, mask_not[0], mask_or[0], fmap_img2col_h_num, SRC0STRIDEM0,
                                       SRC1STRIDEM0, SRC0STRIDEM0, SRC1STRIDEM0)
            with self.tik_instance.else_scope():
                self.tik_instance.vnot(SCALAR_C0, mask_not[0], mask_ub[0], fmap_img2col_h_num, SRC0STRIDEM0,
                                       SRC1STRIDEM0, SRC0STRIDEM0, SRC1STRIDEM0)
                self.tik_instance.data_move(mask_or[0], mask_ub[0], 0, 1, fmap_img2col_h_num, 0, 0)

    def _dup_mask_fun(self, mask_ub, mask_shape_ub):
        """
         caculate max and mask of data_input

         Parameters
         ----------
         mask_ub: mask in ub
         mask_shape_ub: shape of mask_ub

         Returns
         -------
         none
         """
        scalar_repeat_times = mask_shape_ub[2]
        repeat_times = _ceil_div(scalar_repeat_times, SCALAR_MAX)
        # dup 8*blocks init 1 into a buffer:
        if scalar_repeat_times > SCALAR_MAX:
            with self.tik_instance.for_range(0, repeat_times) as repeat_index:
                with self.tik_instance.if_scope(repeat_index != (repeat_times - 1)):
                    self.tik_instance.vector_dup(MASK, mask_ub[repeat_index * SCALAR_MAX * SCALAR_C0],
                                                 MAX_VALUE_FP16, SCALAR_MAX // DSTSTRIDEM1, DSTSTRIDEM0, SRC0STRIDEM1)
                with self.tik_instance.else_scope():
                    self.tik_instance.vector_dup(SCALAR_C0, mask_ub[repeat_index * SCALAR_MAX * SCALAR_C0],
                                                 MAX_VALUE_FP16, (scalar_repeat_times - repeat_index * SCALAR_MAX),
                                                 DSTSTRIDEM0, DSTSTRIDEM0)
        else:
            self.tik_instance.vector_dup(SCALAR_C0, mask_ub, MAX_VALUE_FP16, scalar_repeat_times,
                                         DSTSTRIDEM0, DSTSTRIDEM0)

    def _img2col(self, h_index, fmap_img2col_ub,
                 ori_ub_input, w_size, ori_ub_scalar):
        repeat = self.window_w * SCALAR_C0 // MASK
        remain = self.window_w * SCALAR_C0 % MASK

        with self.tik_instance.for_range(0, self.window_h) as h_rep:
            with self.tik_instance.for_range(0, SCALAR_C0) as b_rep:
                if repeat > 0:
                    with self.tik_instance.for_range(0, repeat) as idx:
                        with self.tik_instance.if_scope(
                            (self.scalar_source_w + self.stride_w * b_rep) // self.stride_w >= self.out_size_w):
                            times = (self.scalar_source_w + self.stride_w * b_rep) // self.stride_w // self.out_size_w
                            source_h_new = self.scalar_source_h + self.stride_h * times + h_rep
                            source_w_new = ((self.scalar_source_w + self.stride_w * b_rep) // self.stride_w) % \
                                             self.out_size_w * self.stride_w + idx * MASK // SCALAR_C0

                            with self.tik_instance.if_scope((source_h_new * w_size + source_w_new) * SCALAR_C0 <
                                                                ori_ub_scalar):
                                self.tik_instance.vadds(MASK, fmap_img2col_ub[h_index * SCALAR_C0 * SCALAR_C0 *
                                                                              self.fmap_img2col_w +
                                                                              h_rep * self.window_w *
                                                                              SCALAR_C0 * SCALAR_C0 + b_rep *
                                                                              SCALAR_C0 + idx * MASK * SCALAR_C0],
                                                        ori_ub_input[(source_h_new * w_size +
                                                                      source_w_new) * SCALAR_C0],
                                                        self.tik_instance.Scalar(dtype="float16", init_value=0.0),
                                                        1, 16, 1, 0, 0)

                        with self.tik_instance.else_scope():
                            self.tik_instance.vadds(MASK, fmap_img2col_ub[h_index * SCALAR_C0 * SCALAR_C0 *
                                                                          self.fmap_img2col_w +
                                                                          h_rep * self.window_w *
                                                                          SCALAR_C0 * SCALAR_C0 + b_rep *
                                                                          SCALAR_C0 + idx * MASK * SCALAR_C0],
                                                        ori_ub_input[(self.scalar_source_h * w_size +
                                                                      self.scalar_source_w + w_size * h_rep +
                                                                      self.stride_w * b_rep) *
                                                                      SCALAR_C0 + idx * MASK],
                                                        self.tik_instance.Scalar(dtype="float16", init_value=0.0),
                                                        1, 16, 1, 0, 0)

                if remain > 0:
                    with self.tik_instance.if_scope(
                        (self.scalar_source_w + self.stride_w * b_rep) // self.stride_w >= self.out_size_w):
                        times = (self.scalar_source_w + self.stride_w * b_rep) // self.stride_w // self.out_size_w
                        source_h_new = self.scalar_source_h + self.stride_h * times + h_rep
                        source_w_new = ((self.scalar_source_w + self.stride_w * b_rep) // self.stride_w) % \
                                        self.out_size_w * self.stride_w + repeat * MASK // SCALAR_C0

                        with self.tik_instance.if_scope((source_h_new * w_size + source_w_new) * SCALAR_C0 <
                                                         ori_ub_scalar):
                                self.tik_instance.vadds(remain, fmap_img2col_ub[h_index * SCALAR_C0 * SCALAR_C0 *
                                                                                self.fmap_img2col_w +
                                                                                h_rep * self.window_w *
                                                                                SCALAR_C0 * SCALAR_C0 + b_rep *
                                                                                SCALAR_C0 + repeat *
                                                                                MASK * SCALAR_C0],
                                                        ori_ub_input[(source_h_new * w_size +
                                                                      source_w_new) * SCALAR_C0],
                                                        self.tik_instance.Scalar(dtype="float16", init_value=0.0),
                                                        1, 16, 1, 0, 0)

                    with self.tik_instance.else_scope():
                        self.tik_instance.vadds(remain, fmap_img2col_ub[h_index * SCALAR_C0 * SCALAR_C0 *
                                                                        self.fmap_img2col_w +
                                                                        h_rep * self.window_w *
                                                                        SCALAR_C0 * SCALAR_C0 + b_rep *
                                                                        SCALAR_C0 + repeat *
                                                                        MASK * SCALAR_C0],
                                                    ori_ub_input[(self.scalar_source_h * w_size +
                                                                  self.scalar_source_w + w_size * h_rep +
                                                                  self.stride_w * b_rep) *
                                                                  SCALAR_C0 + repeat * MASK],
                                                    self.tik_instance.Scalar(dtype="float16", init_value=0.0),
                                                    1, 16, 1, 0, 0)
