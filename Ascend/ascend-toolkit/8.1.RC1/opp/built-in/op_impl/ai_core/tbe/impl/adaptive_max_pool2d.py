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
adaptive max pool2d
"""
import math
import functools
import te.platform as tbe_platform
from te import tik
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import OPTION_OUTPUT
from te.utils.op_utils import REQUIRED_ATTR_LIST_INT
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import check_op_params
from te.utils.error_manager import error_manager_vector
from tbe.common.utils import para_check
from tbe.common.platform import get_bit_len


# 'pylint: disable=too-few-public-methods,not-use-list-comprehension
class Constant:
    """
    Constant in this class
    """
    BITS_OF_BYTE = 8
    BLOCK_BYTES = 32
    # reserved ub size
    UB_RESERVED_SIZE = 8 * 1024
    # parameters for vector instruct
    MASK = 128
    # repeat limit
    REPEAT_LIMIT = 255
    STRIDE_LIMIT = 255
    # min fp16
    MIN_FP16 = -65400.0
    C_ZERO = 16
    TIK_MOVE_STRIDE_MAX = 65535
    UB_DIVIDE_NUM = 2


# 'pylint: disable=unused-argument,unused-variable,too-many-locals
def check_supported(x, y, argmax, output_size, kernel_name='adaptive_max_pool2d'):
    """
    check whether ai_core is supported
    """
    input_format = x.get("ori_format").upper()
    input_shape = x.get("ori_shape")
    input_dtype = x.get("dtype")
    if input_format == "NHWC":
        in_size = input_shape[1:3]
    else:
        in_size = input_shape[2:4]

    k_info_h, k_info_w = k_info(in_size, output_size)
    k_max_h = max(k_info_h[2])
    k_max_w = max(k_info_w[2])
    dtype_bytes = get_bit_len(input_dtype) // Constant.BITS_OF_BYTE
    ub_cell_available = (tbe_platform.get_soc_spec(
        tbe_platform.UB_SIZE) - 8 * 1024) // Constant.BLOCK_BYTES // dtype_bytes // Constant.UB_DIVIDE_NUM
    if k_max_h * k_max_w > ub_cell_available:
        reason = "size is too large to compute in ub,k_max_h is %s, k_max_w is %s" % (k_max_h, k_max_w)
        return False, reason
    return True, ""


def ceil_div(num, factor):
    """up div
    """
    return (num + factor - 1) // factor


def ceil_align(num, factor):
    """up align
    """
    return ceil_div(num, factor) * factor


def start_index(a, b, c):
    """calculate start index
    """
    return math.floor(float(a * c) / b)


def end_index(a, b, c):
    """calculate end index
    """
    return math.ceil((float(a + 1) * c) / b)


def k_info(input_size, output_size):
    """calculate window start, end, size of w and h
    """
    input_h = input_size[0]
    input_w = input_size[1]
    output_h = output_size[0]
    output_w = output_size[1]
    k_info_h = [[0] * output_h for _ in range(3)]
    k_info_w = [[0] * output_w for _ in range(3)]
    for oh in range(0, output_h):
        k_info_h[0][oh] = start_index(oh, output_h, input_h)
        k_info_h[1][oh] = end_index(oh, output_h, input_h)
        k_info_h[2][oh] = k_info_h[1][oh] - k_info_h[0][oh]
    for ow in range(0, output_w):
        k_info_w[0][ow] = start_index(ow, output_w, input_w)
        k_info_w[1][ow] = end_index(ow, output_w, input_w)
        k_info_w[2][ow] = k_info_w[1][ow] - k_info_w[0][ow]
    return k_info_h, k_info_w


# 'pylint: disable=too-many-public-methods
class AdaptiveMaxPool2d:
    """
    Applies a 2D adaptive max pooling over an input signal composed of several input planes.

    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    """
    # 'pylint: disable=too-many-arguments,too-many-statements
    def __init__(self, x, y, argmax, output_size, kernel_name='adaptive_max_pool2d'):
        """
        Init DiagPart parameters.

        Parameters
        ----------
        x: dict
            the dict of input tensor.
        y: dict
            the dict of output tensor.
        output_size:list
            the output h and w, 2 dimension required
        kernel_name: str
            cce kernel name.

        Returns
        -------
        None
        """
        # params about npu
        self.tik_instance = tik.Tik()
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_availble = tbe_platform.get_soc_spec(
            tbe_platform.UB_SIZE) - Constant.UB_RESERVED_SIZE
        # params run on cpu
        self.x_shape = x.get("shape")
        self.x_dtype = x.get("dtype").lower()
        self.x_format = x.get("format").upper()
        self.y_shape = y.get("shape")
        self.y_dtype = y.get("dtype").lower()
        self.argmax_dtype = argmax.get("dtype").lower()
        self.output_size = output_size
        self.kernel_name = kernel_name
        self.dtype = self.x_dtype

        self.check_param()

        self.n = self.x_shape[0]
        self.c1 = self.x_shape[1]
        self.input_h = self.x_shape[2]
        self.input_w = self.x_shape[3]
        self.c0 = self.x_shape[4]
        self.output_h = self.y_shape[2]
        self.output_w = self.y_shape[3]

        self.k_info_size = (self.output_h + self.output_w) * (
                get_bit_len('int32') // Constant.BITS_OF_BYTE)

        # start, end, size, offset
        self.k_info_h = [[0] * self.output_h for _ in range(3)]
        self.k_info_w = [[0] * self.output_w for _ in range(3)]
        # max_h, max_w, second max_h, second max_w
        self.k_max = [0, 0]
        self.k_max_second = [0, 0]
        # mode: 0, only one size; 1, two size contain
        self.k_mode = [0, 0]
        self.cal_k_info()
        self.adaptive_mode = 0

        self.dtype_bytes = get_bit_len(self.x_dtype) // Constant.BITS_OF_BYTE
        self.ub_max_num = self.ub_availble // self.dtype_bytes
        self.ub_sec_eles = self.ub_max_num // Constant.UB_DIVIDE_NUM
        self.elem_per_block = Constant.BLOCK_BYTES // self.dtype_bytes
        x_size = int(functools.reduce(lambda _x, _y: _x * _y, self.x_shape))
        y_size = int(functools.reduce(lambda _x, _y: _x * _y, self.y_shape))

        self.total_cells = self.n * self.c1 * self.output_h * self.output_w
        self.trans_cells = self.n * self.c1 * self.output_h
        self.ele_per_cell = self.c0
        self.block_per_cell = self.ele_per_cell // self.elem_per_block
        self.block_per_wi = self.input_w * self.block_per_cell
        self.block_per_wo = self.output_w * self.block_per_cell
        self.ub_max_cells = self.ub_max_num // (self.ele_per_cell * self.dtype_bytes)
        self.ub_sec_cells = self.ub_max_cells // Constant.UB_DIVIDE_NUM

        self.act_core_num = 0
        self.one_core_unit = 0
        self.last_core_unit = 0
        self.total_unit = 0
        self.unit = 0
        self.max_unit = 0
        self.one_core_loop_num = 0
        self.one_core_loop_left = 0
        self.last_core_loop_num = 0
        self.last_core_loop_left = 0

        self.cal_tiling_info()

        # variables run on npu
        self.input_gm = self.tik_instance.Tensor(
            self.x_dtype, (x_size,), name="input_gm", scope=tik.scope_gm)
        self.output_gm = self.tik_instance.Tensor(
            self.y_dtype, (y_size,), name="output_gm", scope=tik.scope_gm)
        self.argmax_gm = self.tik_instance.Tensor(
            self.argmax_dtype, (y_size,), name="argmax_gm", scope=tik.scope_gm)

        self.k_start_h = self.tik_instance.Tensor(
            "int32", (self.output_h,), name="k_start_h", scope=tik.scope_ubuf)
        self.k_end_h = self.tik_instance.Tensor(
            "int32", (self.output_h,), name="k_end_h", scope=tik.scope_ubuf)
        self.k_size_h = self.tik_instance.Tensor(
            "int32", (self.output_h,), name="k_size_h", scope=tik.scope_ubuf)
        self.k_start_w = self.tik_instance.Tensor(
            "int32", (self.output_w,), name="k_start_w", scope=tik.scope_ubuf)
        self.k_end_w = self.tik_instance.Tensor(
            "int32", (self.output_w,), name="k_end_w", scope=tik.scope_ubuf)
        self.k_size_w = self.tik_instance.Tensor(
            "int32", (self.output_w,), name="k_size_w", scope=tik.scope_ubuf)
        self.k_max_h = self.k_max[0]
        self.k_max_w = self.k_max[1]
        self.k_mode_h = self.k_mode[1]
        self.k_mode_w = self.k_mode[0]
        self.offset_h = self.tik_instance.Tensor("int32", (self.output_h,), name="offset_h", scope=tik.scope_ubuf)
        self.offset_w = self.tik_instance.Tensor("int32", (self.output_w,), name="offset_w", scope=tik.scope_ubuf)
        self.set_k_info()
        self.init_ub_tensor()

    def init_ub_tensor(self):
        """Init ub tensor
        """
        if self.adaptive_mode == 1:
            self.ub_tensor = self.tik_instance.Tensor(
                self.dtype, (self.ub_max_num,), name="ub_a", scope=tik.scope_ubuf)
        else:
            self.ub_a = self.tik_instance.Tensor(
                self.dtype, (self.ub_sec_eles,), name="ub_a", scope=tik.scope_ubuf)
            self.ub_b = self.tik_instance.Tensor(
                self.dtype, (self.ub_sec_eles,), name="ub_b", scope=tik.scope_ubuf)

    def cal_k_info(self):
        """calculate window start, end, size
        """
        self.k_info_h, self.k_info_w = k_info([self.input_h, self.input_w], [self.output_h, self.output_w])

        k_h = list(set(self.k_info_h[2]))
        k_w = list(set(self.k_info_w[2]))
        self.k_max = [max(k_h), max(k_w)]
        self.k_max_second = [min(k_h), min(k_w)]
        self.k_mode = [len(k_h) - 1, len(k_w) - 1]

    def set_k_info(self):
        """set window info on ub tensor
        """
        k_sca1 = self.tik_instance.Scalar("int32", name="k_sca1")
        k_sca2 = self.tik_instance.Scalar("int32", name="k_sca2")
        for oh in range(0, self.output_h):
            k_sca1.set_as(self.k_info_h[0][oh])
            k_sca2.set_as(self.k_info_h[1][oh])
            self.k_start_h[oh].set_as(k_sca1)
            self.k_end_h[oh].set_as(k_sca2)
            self.k_size_h[oh].set_as(k_sca2 - k_sca1)
        for ow in range(0, self.output_w):
            k_sca1.set_as(self.k_info_w[0][ow])
            k_sca2.set_as(self.k_info_w[1][ow])
            self.k_start_w[ow].set_as(k_sca1)
            self.k_end_w[ow].set_as(k_sca2)
            self.k_size_w[ow].set_as(k_sca2 - k_sca1)

    def cal_core_num(self, total_unit, core_num):
        """cal actual used core number
        """
        self.one_core_unit = ceil_div(total_unit, core_num)
        self.act_core_num = total_unit // self.one_core_unit
        if total_unit % self.one_core_unit != 0:
            self.act_core_num += 1
        self.last_core_unit = total_unit - (self.act_core_num - 1) * self.one_core_unit

    def cal_tiling_info(self):
        """assign core task
        """
        # unit: cell, copy only, for scenes: input size == output size
        if self.output_h == self.input_h and self.output_w == self.input_w:
            self.adaptive_mode = 1
            self.total_unit = self.total_cells
            self.cal_core_num(self.total_unit, self.core_num)
            self.max_unit = self.ub_max_cells
            self.one_core_loop_num = self.one_core_unit // self.max_unit
            self.one_core_loop_left = self.one_core_unit % self.max_unit
            self.last_core_loop_num = self.last_core_unit // self.max_unit
            self.last_core_loop_left = self.last_core_unit % self.max_unit

        # unit: tape, output_w*k_max_h*k_max_w*C0, for scenes: small output_w and window
        elif self.output_w * self.k_max[0] * self.k_max[1] <= self.ub_sec_cells:
            self.adaptive_mode = 2
            self.total_unit = self.n * self.c1 * self.output_h
            self.cal_core_num(self.total_unit, self.core_num)
            self.unit = self.output_w * self.k_max[0] * self.k_max[1]
            self.max_unit = self.ub_sec_cells // self.unit
            self.one_core_loop_num = self.one_core_unit // self.max_unit
            self.one_core_loop_left = self.one_core_unit % self.max_unit
            self.last_core_loop_num = self.last_core_unit // self.max_unit
            self.last_core_loop_left = self.last_core_unit % self.max_unit

        # unit: window, k_max_h*k_max_w*C0, , for scenes: big window
        elif self.k_max[0] * self.k_max[1] <= self.ub_sec_cells:
            self.adaptive_mode = 3
            self.total_unit = self.n * self.c1 * self.output_h * self.output_w
            self.cal_core_num(self.total_unit, self.core_num)
            self.unit = self.k_max[0] * self.k_max[1]
            self.max_unit = self.ub_sec_cells // self.unit
            self.one_core_loop_num = self.one_core_unit // self.max_unit
            self.one_core_loop_left = self.one_core_unit % self.max_unit
            self.last_core_loop_num = self.last_core_unit // self.max_unit
            self.last_core_loop_left = self.last_core_unit % self.max_unit
        else:
            error_detail = "the product {} of the ksize [{}, {}] is larger than the ub assigned size {}, " \
                           "please check in check_supported".format(
                self.k_max[0] * self.k_max[1], self.k_max[0], self.k_max[1], self.ub_sec_cells)
            error_manager_vector.raise_err_input_shape_invalid(self.kernel_name, "x", error_detail)

    def adaptive_max_pool2d_compute(self):
        """Run
        """
        with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as core_idx:
            with self.tik_instance.if_scope(core_idx < self.act_core_num):
                # front cores
                with self.tik_instance.if_scope(core_idx < self.act_core_num - 1):
                    if self.adaptive_mode == 1:
                        self.copy_only(core_idx, self.one_core_loop_num, self.one_core_loop_left)
                    if self.adaptive_mode == 2:
                        self.tiling_unit_output_w(core_idx, self.one_core_loop_num, self.one_core_loop_left)
                    if self.adaptive_mode == 3:
                        self.tiling_w_dim_core_nc(core_idx, self.one_core_loop_num, self.one_core_loop_left)
                # last core
                with self.tik_instance.else_scope():
                    if self.adaptive_mode == 1:
                        self.copy_only(core_idx, self.last_core_loop_num, self.last_core_loop_left)
                    if self.adaptive_mode == 2:
                        self.tiling_unit_output_w(core_idx, self.last_core_loop_num, self.last_core_loop_left)
                    if self.adaptive_mode == 3:
                        self.tiling_w_dim_core_nc(core_idx, self.last_core_loop_num, self.last_core_loop_left)

    def tiling_w_dim_core_nc(self, core_idx, loop_num, loop_left):
        """Tiling w dim when core num at nc1
        """
        if loop_num >= 2:
            thread_num = 2
        else:
            thread_num = 1
        with self.tik_instance.for_range(0, loop_num, thread_num=thread_num) as loop_idx:
            self.tiling_w_dim_core_nc_process(core_idx, loop_idx, self.max_unit)
        if loop_left > 0:
            self.tiling_w_dim_core_nc_process(core_idx, loop_num, loop_left)

    # 'pylint: disable=too-many-locals
    def tiling_w_dim_core_nc_process(self, core_idx, loop_idx, unit_num):
        """Tiling w dim process when core num at nc1
        """
        # dup all
        size = unit_num * self.k_max_h * self.k_max_w * Constant.C_ZERO
        repeat_times = size // Constant.MASK
        repeat_left = size % Constant.MASK
        if repeat_times > 0:
            self.tik_instance.vector_dup(Constant.MASK, self.ub_a, Constant.MIN_FP16, repeat_times, 1, 8)
        if repeat_left > 0:
            offset = repeat_times * Constant.MASK
            self.tik_instance.vector_dup(repeat_left, self.ub_a[offset], Constant.MIN_FP16, 1, 1, 8)

        # move in
        with self.tik_instance.for_range(0, unit_num) as unit_idx:
            unit_id = core_idx * self.one_core_unit + loop_idx * self.max_unit + unit_idx
            # The plane is output_h * output_w * c0
            plane_id = unit_id // (self.output_h * self.output_w)
            hw_idx = unit_id % (self.output_h * self.output_w)
            h_idx = hw_idx // self.output_w
            w_idx = hw_idx % self.output_w

            k_start_h = self.tik_instance.Scalar("int32", name="k_start_h")
            k_start_w = self.tik_instance.Scalar("int32", name="k_start_w")
            k_size_h = self.tik_instance.Scalar("int32", name="k_size_h")
            k_size_w = self.tik_instance.Scalar("int32", name="k_size_w")

            k_start_h.set_as(self.k_start_h[h_idx])
            k_start_w.set_as(self.k_start_w[w_idx])
            k_size_h.set_as(self.k_size_h[h_idx])
            k_size_w.set_as(self.k_size_w[w_idx])

            in_offset = \
                (plane_id * self.input_h * self.input_w + k_start_h * self.input_w + k_start_w) * Constant.C_ZERO
            ub_offset = unit_idx * self.k_max_w * Constant.C_ZERO
            self.move_gm_to_ub_one_window(self.ub_a, in_offset, ub_offset, h_idx, w_idx, unit_num)

        # reduce max width
        if self.k_max_w == 1:
            self.reduce_max_repeat_width_ksize_one_width(self.ub_a, self.ub_b, self.k_max_h, unit_num * Constant.C_ZERO)
        else:
            self.reduce_max_repeat_width_ksize_more_width(
                self.ub_a, self.ub_b, self.k_max_h, unit_num * Constant.C_ZERO)
        # reduce max height
        if self.k_max_h == 1:
            self.reduce_max_repeat_width_ksize_one_height(self.ub_b, self.ub_a, 1, unit_num * Constant.C_ZERO)
        else:
            self.reduce_max_repeat_width_ksize_more_height(self.ub_b, self.ub_a, 1, unit_num * Constant.C_ZERO)
        # move out
        offset_out = (core_idx * self.one_core_unit + loop_idx * self.max_unit) * Constant.C_ZERO
        self.tik_instance.data_move(self.output_gm[offset_out], self.ub_a, 0, 1, unit_num, 0, 0)

    # 'pylint: disable=too-many-arguments
    def move_gm_to_ub_one_window(self, ub_dst, base_offset, ub_offset, h_idx, w_idx, win_num):
        """move gm to ub
        """
        k_start_w = self.tik_instance.Scalar("int32", name="k_start_w")
        k_size_h = self.tik_instance.Scalar("int32", name="k_size_h")
        k_size_w = self.tik_instance.Scalar("int32", name="k_size_w")
        k_size_h.set_as(self.k_size_h[h_idx])
        k_start_w.set_as(self.k_start_w[w_idx])
        k_size_w.set_as(self.k_size_w[w_idx])

        src_gap = self.input_w - k_size_w
        dst_gap = win_num * self.k_max_w - k_size_w
        with self.tik_instance.if_scope(tik.all(dst_gap <= Constant.TIK_MOVE_STRIDE_MAX,
                                                src_gap <= Constant.TIK_MOVE_STRIDE_MAX)):
            self.tik_instance.data_move(ub_dst[ub_offset], self.input_gm[base_offset], 0,
                                        k_size_h, k_size_w,
                                        src_gap, dst_gap)
        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, k_size_h) as h_loop:
                src_offset = base_offset + h_loop * self.input_w * Constant.C_ZERO
                dst_offset = ub_offset + h_loop * (win_num * self.k_max_w) * Constant.C_ZERO
                self.tik_instance.data_move(ub_dst[dst_offset], self.input_gm[src_offset], 0,
                                            1, k_size_w,
                                            0, 0)

    def tiling_unit_output_w(self, core_idx, loop_num, loop_left):
        """Tiling instance when core num at nc1h
        unit w mode
        """
        if loop_num >= 2:
            thread_num = 2
        else:
            thread_num = 1
        with self.tik_instance.for_range(0, loop_num, thread_num=thread_num) as loop_idx:
            self.tiling_unit_output_w_process(self.ub_a, self.ub_b, core_idx, loop_idx, self.max_unit)
        if loop_left > 0:
            self.tiling_unit_output_w_process(self.ub_a, self.ub_b, core_idx, loop_num, loop_left)

    # 'pylint: disable=too-many-locals,too-many-arguments
    def tiling_unit_output_w_process(self, ub_x, ub_y, core_idx, loop_idx, unit_num):
        """Tiling unit output_w
        """
        # move in
        # get output idx
        loop_plane_start_idx = (core_idx * self.one_core_unit + loop_idx * self.max_unit) // self.output_h
        loop_unit_start_h_idx = (core_idx * self.one_core_unit + loop_idx * self.max_unit) % self.output_h
        loop_plane_end_idx = (core_idx * self.one_core_unit + loop_idx * self.max_unit +
                              unit_num - 1) // self.output_h
        loop_unit_end_h_idx = (core_idx * self.one_core_unit + loop_idx * self.max_unit +
                               unit_num - 1) % self.output_h
        # # calc input idx
        loop_k_start_h = self.tik_instance.Scalar("int32", name="loop_k_start_h")
        loop_k_start_h.set_as(self.k_start_h[loop_unit_start_h_idx])
        loop_k_end_h = self.tik_instance.Scalar("int32", name="loop_k_end_h")
        loop_k_end_h.set_as(self.k_end_h[loop_unit_end_h_idx])
        in_offset = loop_plane_start_idx * self.input_h * self.input_w * Constant.C_ZERO + (
                loop_k_start_h * self.input_w) * Constant.C_ZERO
        burst_len = (loop_plane_end_idx - loop_plane_start_idx) * self.input_h + loop_k_end_h - loop_k_start_h
        self.tik_instance.data_move(ub_y, self.input_gm[in_offset], 0,
                                    1, burst_len * self.input_w,
                                    0, 0)

        # dup all
        size = unit_num * self.k_max_h * self.output_w * self.k_max_w * Constant.C_ZERO
        repeat_times = size // Constant.MASK
        repeat_left = size % Constant.MASK
        if repeat_times > 0:
            self.tik_instance.vector_dup(Constant.MASK, ub_x, Constant.MIN_FP16, repeat_times, 1, 8)
        if repeat_left > 0:
            offset = repeat_times * Constant.MASK
            self.tik_instance.vector_dup(repeat_left, ub_x[offset], Constant.MIN_FP16, 1, 1, 8)

        # rearrange: vector dup and move ub 2 ub
        with self.tik_instance.for_range(0, unit_num) as unit_idx:
            unit_total_id = core_idx * self.one_core_unit + loop_idx * self.max_unit + unit_idx
            plane_idx = unit_total_id // self.output_h
            h_idx = unit_total_id % self.output_h
            k_start_h = self.tik_instance.Scalar("int32", name="k_start_h")
            k_start_h.set_as(self.k_start_h[h_idx])
            src_offset = ((plane_idx - loop_plane_start_idx) * self.input_h + k_start_h - loop_k_start_h
                          ) * self.input_w * Constant.C_ZERO
            dst_offset_h = (unit_idx * self.output_w * self.k_max_h * self.k_max_w) * Constant.C_ZERO
            with self.tik_instance.for_range(0, self.output_w) as w_idx:
                self.move_ub_to_ub_one_window_before_pool(ub_x, ub_y, src_offset, dst_offset_h, h_idx, w_idx)

        # # reduce max width
        if self.k_max_w == 1:
            self.reduce_max_repeat_width_ksize_one_width(ub_x, ub_y, unit_num * self.k_max_h,
                                                         self.output_w * Constant.C_ZERO)
        else:
            self.reduce_max_repeat_width_ksize_more_width(ub_x, ub_y, unit_num * self.k_max_h,
                                                          self.output_w * Constant.C_ZERO)
        # reduce max height
        if self.k_max_h == 1:
            self.reduce_max_repeat_width_ksize_one_height(ub_y, ub_x, unit_num, self.output_w * Constant.C_ZERO)
        else:
            self.reduce_max_repeat_width_ksize_more_height(ub_y, ub_x, unit_num, self.output_w * Constant.C_ZERO)
        # # move out
        offset_out = (core_idx * self.one_core_unit + loop_idx * self.max_unit) * self.output_w * Constant.C_ZERO
        self.tik_instance.data_move(self.output_gm[offset_out], ub_x, 0, 1, unit_num * self.output_w, 0, 0)

    # 'pylint: disable=too-many-locals,too-many-arguments
    def move_ub_to_ub_one_window_before_pool(self, ub_dst, ub_src, src_base_offset, dst_base_offset, h_idx, w_idx):
        """rearrange data
        """
        k_start_h = self.tik_instance.Scalar("int32", name="k_start_h")
        k_start_w = self.tik_instance.Scalar("int32", name="k_start_w")
        k_size_h = self.tik_instance.Scalar("int32", name="k_size_h")
        k_size_w = self.tik_instance.Scalar("int32", name="k_size_w")

        k_start_h.set_as(self.k_start_h[h_idx])
        k_start_w.set_as(self.k_start_w[w_idx])
        k_size_h.set_as(self.k_size_h[h_idx])
        k_size_w.set_as(self.k_size_w[w_idx])

        src_offset = src_base_offset + k_start_w * Constant.C_ZERO
        src_gap = self.input_w - k_size_w
        dst_offset = dst_base_offset + w_idx * self.k_max_w * Constant.C_ZERO
        dst_gap = self.output_w * self.k_max_w - k_size_w

        with self.tik_instance.if_scope(dst_gap <= Constant.TIK_MOVE_STRIDE_MAX):
            self.tik_instance.data_move(ub_dst[dst_offset], ub_src[src_offset], 0,
                                        k_size_h, k_size_w,
                                        src_gap, dst_gap)
        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, k_size_h) as h_loop:
                loop_src_offset = src_offset + h_loop * self.input_w * Constant.C_ZERO
                loop_dst_offset = dst_offset + h_loop * (self.output_w * k_size_w) * Constant.C_ZERO
                self.tik_instance.data_move(ub_dst[loop_dst_offset], ub_src[loop_src_offset], 0,
                                            1, k_size_w,
                                            0, 0)

    def reduce_max_repeat_width_ksize_one_width(self, ub_x, ub_y, repeat_ph, size_ow):
        """Reduce max width with repeat width when ksize equal to one
        """
        self.reduce_max_repeat_width(ub_y, ub_x, ub_x, repeat_ph * size_ow, self.k_max_w, self.k_max_w)

    def reduce_max_repeat_width_ksize_more_width(self, ub_x, ub_y, repeat_ph, size_ow):
        """Reduce max width with repeat width when ksize not equal to one
        """
        self.reduce_max_repeat_width(
            ub_y, ub_x, ub_x[Constant.C_ZERO:], repeat_ph * size_ow, self.k_max_w, self.k_max_w)
        with self.tik_instance.for_range(0, self.k_max_w - 2) as idx:
            offset_w = (idx + 2) * Constant.C_ZERO
            self.reduce_max_repeat_width(ub_y, ub_x[offset_w:], ub_y, repeat_ph * size_ow, self.k_max_w, 1)

    def reduce_max_repeat_width_ksize_one_height(self, ub_y, ub_z, repeat_oh, size_ow):
        """Reduce max height with repeat width when ksize equal to one
        """
        self.reduce_max_repeat_width(ub_z, ub_y, ub_y, repeat_oh * size_ow, 1, 1)

    def reduce_max_repeat_width_ksize_more_height(self, ub_y, ub_z, repeat_oh, size_ow):
        """Reduce max height with repeat width when ksize not equal to one
        """
        with self.tik_instance.for_range(0, repeat_oh) as o_idx:
            offset_dst = o_idx * size_ow
            offset_src0 = o_idx * self.k_max_h * size_ow
            offset_src1 = offset_src0 + size_ow
            self.reduce_max_repeat_width(ub_z[offset_dst:], ub_y[offset_src0:], ub_y[offset_src1:], size_ow, 1, 1)
            with self.tik_instance.for_range(0, self.k_max_h - 2) as idx:
                offset_h = offset_src0 + (idx + 2) * size_ow
                self.reduce_max_repeat_width(ub_z[offset_dst:], ub_y[offset_h:], ub_z[offset_dst:], size_ow, 1, 1)

    # 'pylint: disable=too-many-locals,too-many-arguments
    def reduce_max_repeat_width(self, dst, src0, src1, size, src0_blk=1, src1_blk=1):
        """reduce max for width and height with repeat width, src0_blk/src1_blk is strides_w or one
        """
        # strides_w less and equal to 31
        if max(src0_blk, src1_blk) * 8 <= Constant.STRIDE_LIMIT:
            if size > 0:
                size_loop = size // Constant.MASK
                size_left = size % Constant.MASK
                repeat_loop = size_loop // Constant.REPEAT_LIMIT
                repeat_left = size_loop % Constant.REPEAT_LIMIT
                if repeat_left > 0:
                    repeat_offset = repeat_loop * Constant.REPEAT_LIMIT * Constant.MASK
                    repeat_offset_src0 = repeat_loop * Constant.REPEAT_LIMIT * src0_blk * Constant.MASK
                    repeat_offset_src1 = repeat_loop * Constant.REPEAT_LIMIT * src1_blk * Constant.MASK
                    self.tik_instance.vmax(Constant.MASK,
                                           dst[repeat_offset], src0[repeat_offset_src0], src1[repeat_offset_src1],
                                           repeat_left,
                                           1, src0_blk, src1_blk, 8, src0_blk * 8, src1_blk * 8)
                if size_left > 0:
                    size_offset = size_loop * Constant.MASK
                    size_offset_src0 = size_loop * src0_blk * Constant.MASK
                    size_offset_src1 = size_loop * src1_blk * Constant.MASK
                    self.tik_instance.vmax(size_left,
                                           dst[size_offset], src0[size_offset_src0], src1[size_offset_src1],
                                           1,
                                           1, src0_blk, src1_blk, 8, src0_blk * 8, src1_blk * 8)
        # strides_w greater to 31
        else:
            if size > 0:
                size_loop = size // Constant.C_ZERO
                with self.tik_instance.for_range(0, size_loop) as size_loop_idx:
                    size_offset = size_loop_idx * Constant.C_ZERO
                    size_offset_src0 = size_loop_idx * src0_blk * Constant.C_ZERO
                    size_offset_src1 = size_loop_idx * src1_blk * Constant.C_ZERO
                    self.tik_instance.vmax(Constant.C_ZERO,
                                           dst[size_offset], src0[size_offset_src0], src1[size_offset_src1],
                                           1,
                                           1, 1, 1, 8, 8, 8)

    def copy_only(self, core_idx, loop_num, loop_left):
        """Only execute move in and move out
        """
        # max unit of 'n*c1*h*w' can move to ub
        with self.tik_instance.for_range(0, loop_num) as loop_idx:
            self.copy_only_process(core_idx, loop_idx, self.max_unit)
        if loop_left > 0:
            self.copy_only_process(core_idx, loop_num, loop_left)

    def copy_only_process(self, core_idx, loop_idx, block_num):
        """Only execute move in and move out
        """
        offset = (core_idx * self.one_core_unit + loop_idx * self.max_unit) * Constant.C_ZERO
        self.tik_instance.data_move(self.ub_tensor, self.input_gm[offset], 0, 1, block_num, 0, 0)
        self.tik_instance.data_move(self.output_gm[offset], self.ub_tensor, 0, 1, block_num, 0, 0)

    def check_param(self):
        """
        Check if the shape of input tensor is compatible with output tensor.

        Parameters:
        ----------
        None.(Get from class member.)

        Returns:
        -------
        None.
        Error will report when exception happened.
        """

        check_list = ("float16",)
        para_check.check_dtype(self.x_dtype, check_list, param_name="x")
        para_check.check_format(self.x_format, ("NC1HWC0",), param_name="x")
        para_check.check_shape(self.x_shape, param_name="x")
        if not len(self.y_shape) == len(self.x_shape) == 5:
            error_detail = "the rank of Output y and input x must be 5, " \
                           "but input x shape is: {}, output y shape is: {}. ".format(
                self.x_shape, self.y_shape)
            error_manager_vector.raise_err_input_shape_invalid(self.kernel_name, "y", error_detail)
        if len(self.output_size) not in (2,):
            error_detail = "output_size must be length 2, " \
                           "but given: {}. ".format(self.output_size)
            error_manager_vector.raise_err_input_shape_invalid(self.kernel_name, "output_size", error_detail)
        if list(self.y_shape) != list(self.x_shape[:2]) + list(self.output_size) + [self.x_shape[4]]:
            error_detail = "the Output y planes must be the same with the input x planes, " \
                           "but input x shape is: {}, output y shape is: {}, output_size is: {}.".format(
                self.x_shape, self.y_shape, self.output_size)
            error_manager_vector.raise_err_input_shape_invalid(self.kernel_name, "x", error_detail)

    def tik_instance_function(self):
        """
        the entry of adaptive_max_pool2d calculation

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.adaptive_max_pool2d_compute()
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.input_gm],
            outputs=[self.output_gm, self.argmax_gm],
            config=opt_config)


# 'pylint: disable=too-many-locals
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, OPTION_OUTPUT, REQUIRED_ATTR_LIST_INT, KERNEL_NAME)
def adaptive_max_pool2d(x, y, argmax, output_size, kernel_name="adaptive_max_pool2d"):
    """
    Applies a 2D adaptive max pooling over an input signal composed of several input planes.

    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.

    Parameters
    ----------
    x: dict
        Dict of input, include keys(shape and dtype)
    y: dict
        Dict of output, Has the same type as input.
    argmax: dict
        Dict of output index, Has the same shape as output.
        the indices along with the outputs in the input plane, Reserved.
    output_size: list of 2 int
        Attributes of input
        the target output size of the image of the form H x W.
        Can be a tuple (H, W) or a single H for a square image H x H.
        H and W can be either a ``int``, or ``None`` which means the size will
        be the same as that of the input.
    kernel_name: str
        cce kernel name, default value is "adaptive_max_pool2d"

    Returns
    -------
    Examples:
    ----------
    # target output size of 5x7
    AdaptiveMaxPool2d((1, 64, 8, 9), (5,7))
    # target output size of 7x7 (square)
    AdaptiveMaxPool2d((1, 64, 10, 9), 7)
    # target output size of 10x7
    AdaptiveMaxPool2d((1, 64, 10, 9), (None, 7))
    -------
    None
    """
    adaptive_max_pool2d_inst = AdaptiveMaxPool2d(x, y, argmax, output_size, kernel_name)
    adaptive_max_pool2d_inst.tik_instance_function()
    return adaptive_max_pool2d_inst.tik_instance
