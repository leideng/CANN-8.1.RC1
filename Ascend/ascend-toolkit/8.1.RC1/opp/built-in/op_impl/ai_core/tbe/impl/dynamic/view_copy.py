#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
view_copy
"""

from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from tbe.common.platform.platform_info import get_soc_spec


# pylint: disable=too-few-public-methods,invalid-name,unused-variable
class Constant:
    DTYPE_INT32 = "int32"
    TILING_SCALAR_DTYPE = "int64"
    TILING_PARAMS_NUM = 40
    BYTE_PER_DATA_8 = 8
    BYTE_PER_DATA_4 = 4
    BYTE_PER_DATA_2 = 2
    BYTE_PER_DATA_1 = 1
    MAX_INT32 = 2 ** 31 - 1
    BLOCK = 32
    TILING_MODE_BATCH_SET_AS = 1
    TILING_MODE_BATCH_DATA_MOVE = 2
    TILING_MODE_MIX = 3
    TILING_MODE_SINGLE_DATA_MOVE = 4
    TILING_MODE_SINGLE_SET_AS = 5
    TILING_MODE_TRANS = 6
    TILING_MODE_CONV = 7
    TILING_MODE_PARTIAL_TRANS = 8
    TILING_MODE_THREE_DIMS_BATCH_DATA_MOVE = 9
    TILING_MODE_THREE_DIMS_BATCH_DATA_MOVE_WITH_TAIL = 10
    TILING_MODE_FOUR_DIMS_BATCH_DATA_MOVE = 11
    TILING_MODE_FOUR_DIMS_BATCH_DATA_MOVE_WITH_TAIL = 12
    LAST_DIM_LIMIT = 32
    CONV_LEN = 16
    UB_USED = 400
    STRIDE_LIMIT = 65535
    MODE_STRIDE = 20
    MODE_STRIDE_FOUR_DIMS = 40
    MAX_NBURST = 4095
    ONCE_MOVE_PAD = 2
    SMALL_UB_SIZE = 196608


def _get_byte_per_data(dtype):
    if dtype in {"uint8", "int8", "bool"}:
        return Constant.BYTE_PER_DATA_1
    if dtype in {"float32", "int32", "uint32"}:
        return Constant.BYTE_PER_DATA_4
    if dtype in {"int16", "float16", "uint16"}:
        return Constant.BYTE_PER_DATA_2
    else:
        return Constant.BYTE_PER_DATA_8


# 'pylint: disable=too-many-locals, too-many-arguments
def check_supported(dst, dst_size, dst_stride, dst_storage_offset, src, src_size, src_stride, src_storage_offset,
                    result, kernel_name="view_copy"):
    """
        check the op support situation.
    """
    dst_shape = dst.get("shape")
    src_shape = src.get("shape")
    dst_size_value = dst_size.get("const_value")
    dst_stride_value = dst_stride.get("const_value")
    src_size_value = src_size.get("const_value")
    dst_dtype = dst.get("dtype").lower()
    if dst_dtype == "bfloat16":
        dst_dtype = "float16"
    byte_per_data = _get_byte_per_data(dst_dtype)
    data_per_block = Constant.BLOCK / byte_per_data
    if int(-1) in dst_shape or int(-2) in dst_shape:
        return "Unknown"

    if src_shape != src_size_value or src_size_value != dst_size_value:
        return False
    
    dst_size_len = len(dst_size_value)
    # To: generate more generalized branches.
    isize = 1
    istride = 1
    new_dst_size = []
    for i in range(dst_size_len - 1, -1, -1):
        if (istride * isize != dst_stride_value[i]):
            new_dst_size.insert(0, isize)
            isize = dst_size_value[i]
        else:
            isize *= dst_size_value[i]
    new_dst_size.insert(0, isize)
    if dst_size_len == 3 and dst_size_value[1] == dst_stride_value[2] and \
        dst_stride_value[1] == 1 and dst_size[2] == 2 and \
        dst_size_value[1] * dst_size_value[2] * 2 * byte_per_data < Constant.SMALL_UB_SIZE and \
        (dst_size_value[1] * dst_size_value[2] % data_per_block == 0 or \
        dst_stride_value[0] - dst_size_value[1] * dst_size_value[2] >= data_per_block):
        return True
        
    if dst_size_len == 3 and dst_size_value[0] == 1 and dst_stride_value[0] == 1 and \
        dst_stride_value[1] == 1 and (dst_size_value[1] % (data_per_block) == 0 or \
        dst_stride_value[2] - dst_size_value[1] >= data_per_block) and \
        dst_size_value[1] * (dst_size_value[2] + 1) * byte_per_data < Constant.SMALL_UB_SIZE:
        return True

    if 4 >= len(new_dst_size) >= 3:
        if new_dst_size[-1] >= Constant.LAST_DIM_LIMIT:
            return True
    
    # calculate dst_size_stride
    dst_size_stride = []
    tmp = 1
    for i in range(dst_size_len - 1, -1, -1):
        dst_size_stride.append(tmp)
        tmp = tmp * dst_size_value[i]
    dst_size_stride.reverse()

    diff = -1
    diff_index = -1
    for i in range(dst_size_len - 1, -1, -1):
        if dst_stride_value[i] != dst_size_stride[i]:
            diff = dst_stride_value[i] - dst_size_stride[i]
            diff_index = i
            break

    if diff == -1:
        return False
    
    for i in range(diff_index - 1, -1, -1):
        diff = diff * dst_size_value[i + 1]
        if diff != dst_stride_value[i] - dst_size_stride[i]:
            return False

    return True


class ViewCopy():
    '''ViewCopy'''
    # pylint: disable=too-few-public-methods,invalid-name,unused-variable,too-many-arguments
    def __init__(self, dst, dst_size, dst_stride, dst_offset, src, src_size, src_stride, src_offset):
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.dtype = dst.get("dtype").lower()
        if self.dtype == "bool":
            self.dtype = "int8"
        if self.dtype == "bfloat16":
            self.dtype = "float16"
        self.dtype_cons = dst_size.get("dtype").lower()

        # func: for task allocation
        self.avail_aicore_num = get_soc_spec("CORE_NUM")
        self.available_ub_size = get_soc_spec("UB_SIZE")

        self.dst_gm = self.tik_instance.Tensor(
            self.dtype, (Constant.MAX_INT32,),
            name="dst",
            scope=tik.scope_gm)
        self.src_gm = self.tik_instance.Tensor(
            self.dtype, (Constant.MAX_INT32,),
            name="src",
            scope=tik.scope_gm)
        self.result_gm = self.tik_instance.Tensor(
            self.dtype, (Constant.MAX_INT32,),
            name="result",
            scope=tik.scope_gm)
        
        self.dst_size_gm = self.tik_instance.Tensor(
            self.dtype_cons, (32,),
            name="dst_size",
            scope=tik.scope_gm)
        self.src_size_gm = self.tik_instance.Tensor(
            self.dtype_cons, (32,),
            name="src_size",
            scope=tik.scope_gm)

        self.dst_stride_gm = self.tik_instance.Tensor(
            self.dtype_cons, (32,),
            name="dst_stride",
            scope=tik.scope_gm)
        self.src_stride_gm = self.tik_instance.Tensor(
            self.dtype_cons, (32,),
            name="src_stride",
            scope=tik.scope_gm)

        self.dst_offset_gm = self.tik_instance.Tensor(
            self.dtype_cons, (32,),
            name="dst_storage_offset",
            scope=tik.scope_gm)
        self.src_offset_gm = self.tik_instance.Tensor(
            self.dtype_cons, (32,),
            name="src_storage_offset",
            scope=tik.scope_gm)
        
        self.tiling_gm = self.tik_instance.Tensor(
            Constant.TILING_SCALAR_DTYPE,
            (Constant.TILING_PARAMS_NUM,),
            name="tiling_gm",
            scope=tik.scope_gm)
        
        self.tiling_mode = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "tiling_mode")
        self.move_per_task = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "move_per_task")
        self.total_per_task = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "total_per_task")
        self.dst_offset = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "dst_offset")
        self.data_align = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "data_align")
        self.task_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "task_num")
        self.batch_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "batch_num")
        self.batch_size = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "batch_size")
        self.src_move_size = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "src_move_size")
        self.dst_move_size = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "dst_move_size")
        self.tail_align = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "tail_align")
        self.tail_move_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "tail_move_num")
        self.move_rep_time_src = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "move_rep_time_src")
        self.move_rep_time_src_tail = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "move_rep_time_src_tail")
        self.move_rep_time_dst = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "move_rep_time_dst")
        self.move_rep_time_dst_tail = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "move_rep_time_dst_tail")
        self.move_burst = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "move_burst")
        self.move_stride = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "move_stride")
        self.left_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "left_num")
        self.ub_offset = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "ub_offset")
        self.ub_offset_tail = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "ub_offset_tail")
        self.once_move_in = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "once_move_in")
        self.once_move_out = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "once_move_out")
        self.rep_move_burst = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "rep_move_burst")
        self.rep_times = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "rep_times")
        self.dst_move_size_rep = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "dst_move_size_rep")
        self.src_move_size_rep = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "src_move_size_rep")
        self.move_stride_src = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "move_stride_src")
        self.move_stride_dst = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "move_stride_dst")
        self.move_stride_tail_src = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "move_stride_tail_src")
        self.move_stride_tail_dst = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "move_stride_tail_dst")
        self.offset_src = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "offset_src")
        self.offset_dst = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "offset_dst")
        self.tiling_core_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "tiling_core_num")
        self.dim_four = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "dim_four")
        self.dim_four_src_stride = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "dim_four_src_stride")
        self.dim_four_dst_stride = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "dim_four_dst_stride")
        self.dim_three = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "dim_three")
        self.dim_three_src_stride = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "dim_three_src_stride")
        self.dim_three_dst_stride = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "dim_three_dst_stride")
        self.byte_gap_dst = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "byte_gap_dst")
        self.byte_gap_dst_tail = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "byte_gap_dst_tail")
        self.byte_gap_src = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "byte_gap_src")
        self.byte_gap_src_tail = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "byte_gap_src_tail")
        self.move_burst_byte = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "move_burst_byte")
        self.byte_per_data = _get_byte_per_data(self.dtype)
        self.min_point_per_block = Constant.BLOCK // self.byte_per_data
        self.support_data_move_pad = tbe_platform.api_check_support("tik.data_move_pad", "int8") or \
                                     tbe_platform.api_check_support("tik.data_move_pad", self.dtype)

    def check_data_move_pad(self):
        tiling_mode_mix = tik.any(self.tiling_mode == Constant.TILING_MODE_MIX,
                                  self.tiling_mode == Constant.TILING_MODE_MIX
                                                      + Constant.MODE_STRIDE,
                                  self.tiling_mode == Constant.TILING_MODE_MIX
                                                      + Constant.MODE_STRIDE_FOUR_DIMS)
        if self.support_data_move_pad:
            with self.tik_instance.if_scope(tik.all(self.src_move_size_rep > 1,
                                                    tiling_mode_mix)):
                self.task_num.set_as(self.move_stride_src)

    def data_move_pad_bit64_2_int8(self, dst, src, nburst, burst, dst_gap, src_gap,
                                   right_padding=0, left_padding=0, padding_value=None):
        if tbe_platform.api_check_support("tik.data_move_pad", self.dtype):
            self.tik_instance.data_move_pad(dst, src, nburst, burst, dst_gap,
                                            src_gap, right_padding, left_padding, padding_value)
        else:
            dst_int8 = dst.reinterpret_cast_to("int8")
            src_int8 = src.reinterpret_cast_to("int8")
            self.tik_instance.data_move_pad(dst_int8, src_int8, nburst, burst, dst_gap,
                                            src_gap, right_padding, left_padding, padding_value)

    def get_tiling_data(self):
        tiling_ub = self.tik_instance.Tensor(
            Constant.TILING_SCALAR_DTYPE,
            (Constant.TILING_PARAMS_NUM,),
            tik.scope_ubuf,
            "tiling_ub"
        )
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, 10, 0, 0)
        self.tiling_mode.set_as(tiling_ub[0])
        self.move_per_task.set_as(tiling_ub[1])
        self.total_per_task.set_as(tiling_ub[2])
        self.dst_offset.set_as(tiling_ub[3])
        self.data_align.set_as(tiling_ub[4])
        self.task_num.set_as(tiling_ub[5])
        self.batch_num.set_as(tiling_ub[6])
        self.batch_size.set_as(tiling_ub[7])
        self.src_move_size.set_as(tiling_ub[8])
        self.dst_move_size.set_as(tiling_ub[9])
        self.tail_align.set_as(tiling_ub[10])
        self.tail_move_num.set_as(tiling_ub[11])
        self.move_rep_time_src.set_as(tiling_ub[12])
        self.move_rep_time_src_tail.set_as(tiling_ub[13])
        self.move_rep_time_dst.set_as(tiling_ub[14])
        self.move_rep_time_dst_tail.set_as(tiling_ub[15])
        self.move_burst.set_as(tiling_ub[16])
        self.move_stride.set_as(tiling_ub[17])
        self.left_num.set_as(tiling_ub[18])
        self.ub_offset.set_as(tiling_ub[19])
        self.ub_offset_tail.set_as(tiling_ub[20])
        self.once_move_in.set_as(tiling_ub[21])
        self.once_move_out.set_as(tiling_ub[22])
        self.rep_move_burst.set_as(tiling_ub[23])
        self.rep_times.set_as(tiling_ub[24])
        self.dst_move_size_rep.set_as(tiling_ub[25])
        self.src_move_size_rep.set_as(tiling_ub[26])
        self.move_stride_src.set_as(tiling_ub[27]) 
        self.move_stride_dst.set_as(tiling_ub[28])
        self.move_stride_tail_src.set_as(tiling_ub[29])
        self.move_stride_tail_dst.set_as(tiling_ub[30])
        self.offset_src.set_as(tiling_ub[31])
        self.offset_dst.set_as(tiling_ub[32])
        self.tiling_core_num.set_as(tiling_ub[33])
        self.dim_four.set_as(tiling_ub[34])
        self.dim_four_src_stride.set_as(tiling_ub[35])
        self.dim_four_dst_stride.set_as(tiling_ub[36])
        self.dim_three.set_as(tiling_ub[37])
        self.dim_three_src_stride.set_as(tiling_ub[38])
        self.dim_three_dst_stride.set_as(tiling_ub[39])
        self.byte_gap_dst.set_as(self.offset_dst * self.byte_per_data - self.move_rep_time_src * Constant.BLOCK)
        self.byte_gap_dst_tail.set_as(self.offset_dst * self.byte_per_data - Constant.BLOCK)
        self.byte_gap_src.set_as(self.offset_src * self.byte_per_data - self.move_rep_time_src * Constant.BLOCK)
        self.byte_gap_src_tail.set_as(self.offset_src * self.byte_per_data - Constant.BLOCK)
        self.move_burst_byte.set_as(self.move_rep_time_src * Constant.BLOCK)
        self.check_data_move_pad()

    def view_copy_process_tiling_mode_single_data_move(self, task_id, src_ub,
                                                       src_offset_start=0, dst_offset_start=0):
        with self.tik_instance.for_range(0, self.batch_num) as i:
            src_offset = task_id * self.src_move_size + i * self.batch_size + src_offset_start
            dst_offset = task_id * self.dst_move_size + i * self.batch_size + self.dst_offset + dst_offset_start
            with self.tik_instance.if_scope(tik.any(i != self.batch_num - 1, self.tail_move_num == 0)):
                self.tik_instance.data_move(src_ub, self.src_gm[src_offset], 0, 1, self.move_rep_time_src, 0, 0)
                self.tik_instance.data_move(self.result_gm[dst_offset], src_ub, 0, 1, self.move_rep_time_src, 0, 0)
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(src_ub, self.src_gm[src_offset], 0, 1, self.move_rep_time_src_tail, 0, 0)
                self.tik_instance.data_move(self.result_gm[dst_offset], src_ub, 0, 1, self.move_rep_time_src_tail, 0, 0)
    
    def view_copy_process_tiling_mode_three_dims_batch_data_move(self, task_id, src_ub,
                                                                 src_offset_start=0, dst_offset_start=0):
        src_offset = self.tik_instance.Scalar("int64", init_value=task_id * self.src_move_size + src_offset_start)
        dst_offset = self.tik_instance.Scalar("int64", init_value=task_id * self.dst_move_size + dst_offset_start)

        with self.tik_instance.for_range(0, self.rep_times - 1) as rep:
            with self.tik_instance.if_scope(self.once_move_in == 1):
                self.tik_instance.data_move(src_ub, self.src_gm[src_offset], 0, self.rep_move_burst,
                                            self.move_rep_time_src, self.move_stride_src, 0)
            with self.tik_instance.if_scope(self.once_move_in == 0):
                with self.tik_instance.for_range(0, self.rep_move_burst) as i:
                    self.tik_instance.data_move(src_ub[i * self.ub_offset],
                                                self.src_gm[src_offset + i * self.offset_src],
                                                0, 1, self.move_rep_time_src, 0, 0)
            with self.tik_instance.if_scope(self.once_move_in == Constant.ONCE_MOVE_PAD):
                if self.support_data_move_pad:
                    self.data_move_pad_bit64_2_int8(src_ub, self.src_gm[src_offset], self.rep_move_burst,
                                                    self.move_burst_byte, 0, self.byte_gap_src)
                else:
                    with self.tik_instance.for_range(0, self.rep_move_burst) as i:
                        self.tik_instance.data_move(src_ub[i * self.ub_offset],
                                                    self.src_gm[src_offset + i * self.offset_src],
                                                    0, 1, self.move_rep_time_src, 0, 0)
            with self.tik_instance.if_scope(self.once_move_out == 1):
                self.tik_instance.data_move(self.result_gm[dst_offset], src_ub, 0, self.rep_move_burst,
                                            self.move_rep_time_src, 0, self.move_stride_dst)
            with self.tik_instance.if_scope(self.once_move_out == 0):
                with self.tik_instance.for_range(0, self.rep_move_burst) as i:
                    self.tik_instance.data_move(self.result_gm[dst_offset + i * self.offset_dst],
                                                src_ub[i * self.ub_offset],
                                                0, 1, self.move_rep_time_src, 0, 0)
            with self.tik_instance.if_scope(self.once_move_out == Constant.ONCE_MOVE_PAD):
                if self.support_data_move_pad:
                    self.data_move_pad_bit64_2_int8(self.result_gm[dst_offset], src_ub, self.rep_move_burst,
                                                    self.move_burst_byte, self.byte_gap_dst,
                                                    0)
                else:
                    with self.tik_instance.for_range(0, self.rep_move_burst) as i:
                        self.tik_instance.data_move(self.result_gm[dst_offset + i * self.offset_dst],
                                                    src_ub[i * self.ub_offset],
                                                    0, 1, self.move_rep_time_src, 0, 0)
            src_offset.set_as(self.src_move_size_rep + src_offset)
            dst_offset.set_as(self.dst_move_size_rep + dst_offset)
        with self.tik_instance.if_scope(self.once_move_in == 1):
            self.tik_instance.data_move(src_ub, self.src_gm[src_offset], 0, self.move_burst,
                                        self.move_rep_time_src, self.move_stride_src, 0)
        with self.tik_instance.if_scope(self.once_move_in == 0):
            with self.tik_instance.for_range(0, self.move_burst) as i:
                self.tik_instance.data_move(src_ub[i * self.ub_offset], self.src_gm[src_offset + i * self.offset_src],
                                            0, 1, self.move_rep_time_src, 0, 0)
        with self.tik_instance.if_scope(self.once_move_in == Constant.ONCE_MOVE_PAD):
            if self.support_data_move_pad:
                self.data_move_pad_bit64_2_int8(src_ub, self.src_gm[src_offset], self.move_burst,
                                                self.move_burst_byte, 0, self.byte_gap_src)
            else:
                with self.tik_instance.for_range(0, self.move_burst) as i:
                    self.tik_instance.data_move(src_ub[i * self.ub_offset],
                                                self.src_gm[src_offset + i * self.offset_src],
                                                0, 1, self.move_rep_time_src, 0, 0)
        with self.tik_instance.if_scope(self.once_move_out == 1):
            self.tik_instance.data_move(self.result_gm[dst_offset], src_ub, 0, self.move_burst,
                                        self.move_rep_time_src, 0, self.move_stride_dst)
        with self.tik_instance.if_scope(self.once_move_out == 0):
            with self.tik_instance.for_range(0, self.move_burst) as i:
                self.tik_instance.data_move(self.result_gm[dst_offset + i * self.offset_dst],
                                            src_ub[i * self.ub_offset],
                                            0, 1, self.move_rep_time_src, 0, 0)
        with self.tik_instance.if_scope(self.once_move_out == Constant.ONCE_MOVE_PAD):
            if self.support_data_move_pad:
                self.data_move_pad_bit64_2_int8(self.result_gm[dst_offset], src_ub, self.move_burst,
                                                self.move_burst_byte, self.byte_gap_dst,
                                                0)
            else:
                with self.tik_instance.for_range(0, self.move_burst) as i:
                    self.tik_instance.data_move(self.result_gm[dst_offset + i * self.offset_dst],
                                                src_ub[i * self.ub_offset],
                                                0, 1, self.move_rep_time_src, 0, 0)

    def view_copy_process_tiling_mode_three_dims_batch_data_move_with_tail(self, task_id, src_ub,
                                                                           src_offset_start=0, dst_offset_start=0):
        src_offset_tail = self.tik_instance.Scalar("int64", init_value=task_id * self.src_move_size +
                                                                       src_offset_start + self.left_num)
        dst_offset_tail = self.tik_instance.Scalar("int64", init_value=task_id * self.dst_move_size +
                                                                       dst_offset_start + self.left_num)
        with self.tik_instance.for_range(0, self.tail_move_num - 1) as rep:
            with self.tik_instance.if_scope(self.once_move_in == 1):
                self.tik_instance.data_move(src_ub, self.src_gm[src_offset_tail], 0, self.move_rep_time_src_tail,
                                            1, self.move_stride_tail_src, 0)
            with self.tik_instance.if_scope(self.once_move_in == 0):
                with self.tik_instance.for_range(0, self.move_rep_time_src_tail) as i:
                    self.tik_instance.data_move(src_ub[i * self.ub_offset_tail],
                                                self.src_gm[src_offset_tail + i * self.offset_src],
                                                0, 1, 1, 0, 0)
            with self.tik_instance.if_scope(self.once_move_in == Constant.ONCE_MOVE_PAD):
                if self.support_data_move_pad:
                    self.data_move_pad_bit64_2_int8(src_ub, self.src_gm[src_offset_tail], self.move_rep_time_src_tail,
                                                    Constant.BLOCK, 0, self.byte_gap_src_tail)
                else:
                    with self.tik_instance.for_range(0, self.move_rep_time_src_tail) as i:
                        self.tik_instance.data_move(src_ub[i * self.ub_offset_tail],
                                                    self.src_gm[src_offset_tail + i * self.offset_src],
                                                    0, 1, 1, 0, 0)
            with self.tik_instance.if_scope(self.once_move_out == 1):
                self.tik_instance.data_move(self.result_gm[dst_offset_tail], src_ub, 0, self.move_rep_time_src_tail,
                                            1, 0, self.move_stride_tail_dst)
            with self.tik_instance.if_scope(self.once_move_out == 0):
                with self.tik_instance.for_range(0, self.move_rep_time_src_tail) as i:
                    self.tik_instance.data_move(self.result_gm[dst_offset_tail + i * self.offset_dst],
                                                src_ub[i * self.ub_offset_tail],
                                                0, 1, 1, 0, 0)
            with self.tik_instance.if_scope(self.once_move_out == Constant.ONCE_MOVE_PAD):
                if self.support_data_move_pad:
                    self.data_move_pad_bit64_2_int8(self.result_gm[dst_offset_tail], src_ub,
                                                    self.move_rep_time_src_tail,
                                                    Constant.BLOCK, self.byte_gap_dst_tail, 0)
                else:                
                    with self.tik_instance.for_range(0, self.move_rep_time_src_tail) as i:
                        self.tik_instance.data_move(self.result_gm[dst_offset_tail + i * self.offset_dst],
                                                    src_ub[i * self.ub_offset_tail],
                                                    0, 1, 1, 0, 0)        
            src_offset_tail.set_as(src_offset_tail + self.offset_src * self.move_rep_time_src_tail)
            dst_offset_tail.set_as(dst_offset_tail + self.offset_dst * self.move_rep_time_src_tail)
 
        with self.tik_instance.if_scope(self.once_move_in == 1):
            self.tik_instance.data_move(src_ub, self.src_gm[src_offset_tail], 0, self.tail_align,
                                        1, self.move_stride_tail_src, 0)
        with self.tik_instance.if_scope(self.once_move_in == 0):
            with self.tik_instance.for_range(0, self.tail_align) as i:
                self.tik_instance.data_move(src_ub[i * self.ub_offset_tail],
                                            self.src_gm[src_offset_tail + i * self.offset_src],
                                            0, 1, 1, 0, 0)
        with self.tik_instance.if_scope(self.once_move_in == Constant.ONCE_MOVE_PAD):
            if self.support_data_move_pad:
                self.data_move_pad_bit64_2_int8(src_ub, self.src_gm[src_offset_tail], self.tail_align,
                                                Constant.BLOCK, 0, self.byte_gap_src_tail)
            else:
                with self.tik_instance.for_range(0, self.tail_align) as i:
                    self.tik_instance.data_move(src_ub[i * self.ub_offset_tail],
                                                self.src_gm[src_offset_tail + i * self.offset_src],
                                                0, 1, 1, 0, 0)
        with self.tik_instance.if_scope(self.once_move_out == 1):
            self.tik_instance.data_move(self.result_gm[dst_offset_tail], src_ub, 0, self.tail_align,
                                        1, 0, self.move_stride_tail_dst)
        with self.tik_instance.if_scope(self.once_move_out == 0):
            with self.tik_instance.for_range(0, self.tail_align) as i:
                self.tik_instance.data_move(self.result_gm[dst_offset_tail + i * self.offset_dst],
                                            src_ub[i * self.ub_offset_tail],
                                            0, 1, 1, 0, 0)
        with self.tik_instance.if_scope(self.once_move_out == Constant.ONCE_MOVE_PAD):
            if self.support_data_move_pad:
                self.data_move_pad_bit64_2_int8(self.result_gm[dst_offset_tail], src_ub, self.tail_align,
                                                Constant.BLOCK, self.byte_gap_dst_tail, 0)
            else:
                with self.tik_instance.for_range(0, self.tail_align) as i:
                    self.tik_instance.data_move(self.result_gm[dst_offset_tail + i * self.offset_dst],
                                                src_ub[i * self.ub_offset_tail],
                                                0, 1, 1, 0, 0)
        self.view_copy_process_tiling_mode_three_dims_batch_data_move(task_id,
                                                                      src_ub, src_offset_start, dst_offset_start)
        
    def view_copy_process_tiling_mode_batch_data_move(self, task_id, src_ub, move_rep_time, align, rep_times):
        src_offset = task_id * self.src_move_size
        self.tik_instance.data_move(src_ub, self.src_gm[src_offset], 0, 1, move_rep_time, 0, 0)
        dst_offset = task_id * self.dst_move_size + self.dst_offset
        with self.tik_instance.if_scope(rep_times != 0):
            with self.tik_instance.for_range(0, rep_times) as i:
                dst_offset = task_id * self.dst_move_size + self.dst_offset + \
                    i * self.total_per_task * Constant.MAX_NBURST
                self.tik_instance.data_move(self.result_gm[dst_offset], src_ub[i * self.batch_num], 0,
                                            Constant.MAX_NBURST, self.move_burst, 0, self.move_stride)
        dst_offset = task_id * self.dst_move_size + self.dst_offset + \
            rep_times * self.total_per_task * Constant.MAX_NBURST
        self.tik_instance.data_move(self.result_gm[dst_offset],
                                    src_ub[rep_times * self.batch_num], 0, align, self.move_burst, 0, self.move_stride)

    def view_copy_process_tiling_mode_batch_set_as(self, task_id, src_ub, dst_ub):
        src_offset = task_id * self.src_move_size
        self.tik_instance.data_move(src_ub, self.src_gm[src_offset], 0, 1, self.move_rep_time_src, 0, 0)
        dst_offset = task_id * self.dst_move_size + self.dst_offset
        self.tik_instance.data_move(dst_ub, self.dst_gm[dst_offset], 0, 1, self.move_rep_time_dst, 0, 0)
        
        offset_total_per_task = self.tik_instance.Scalar("int64", init_value=0)
        offset_move_per_task = self.tik_instance.Scalar("int64", init_value=0)
        with self.tik_instance.for_range(0, self.data_align) as i:
            offset_total_per_task.set_as(i * self.total_per_task)
            offset_move_per_task.set_as(i * self.move_per_task)
            with self.tik_instance.for_range(0, self.move_per_task) as j:
                dst_ub_offset = offset_total_per_task + j
                src_ub_offset = offset_move_per_task + j
                dst_ub[dst_ub_offset].set_as(src_ub[src_ub_offset])

        self.tik_instance.data_move(self.result_gm[dst_offset], dst_ub, 0, 1, self.move_rep_time_dst, 0, 0)

    def view_copy_process_tiling_mode_batch_set_as_tail(self, task_id, src_ub, dst_ub):
        src_offset = task_id * self.src_move_size
        if self.support_data_move_pad:
            self.data_move_pad_bit64_2_int8(src_ub, self.src_gm[src_offset], 1,
                                            ((self.tail_align) * self.move_per_task) * self.byte_per_data,
                                            0, 0)
        else:
            self.tik_instance.data_move(src_ub, self.src_gm[src_offset], 0, 1, self.move_rep_time_src_tail, 0, 0)
        dst_offset = task_id * self.dst_move_size + self.dst_offset
        if self.support_data_move_pad:
            self.data_move_pad_bit64_2_int8(dst_ub, self.dst_gm[dst_offset], 1,
                                            ((self.tail_align - 1) * self.total_per_task +
                                             self.move_per_task) * self.byte_per_data, 0, 0)
        else:
            self.tik_instance.data_move(dst_ub, self.dst_gm[dst_offset], 0, 1, self.move_rep_time_dst_tail, 0, 0)
        offset_total_per_task = self.tik_instance.Scalar("int64", init_value=0)
        offset_move_per_task = self.tik_instance.Scalar("int64", init_value=0)
        with self.tik_instance.for_range(0, self.tail_align) as i:
            offset_total_per_task.set_as(i * self.total_per_task)
            offset_move_per_task.set_as(i * self.move_per_task)
            with self.tik_instance.for_range(0, self.move_per_task) as j:
                dst_ub_offset = offset_total_per_task + j
                src_ub_offset = offset_move_per_task + j
                dst_ub[dst_ub_offset].set_as(src_ub[src_ub_offset])
        if self.support_data_move_pad:
            self.data_move_pad_bit64_2_int8(self.result_gm[dst_offset], dst_ub, 1,
                                            ((self.tail_align - 1) * self.total_per_task +
                                             self.move_per_task) * self.byte_per_data, 0, 0)
        else:
            self.tik_instance.data_move(self.result_gm[dst_offset], dst_ub, 0, 1, self.move_rep_time_dst_tail, 0, 0)

    def view_copy_process_tiling_mode_single_set_as(self, task_id, src_ub, dst_ub):
        dst_offset = self.tik_instance.Scalar("int64", init_value=self.total_per_task)
        dst_offset.set_as(dst_offset * task_id + self.dst_offset)
        src_offset = self.tik_instance.Scalar("int64", init_value=self.move_per_task)
        src_offset.set_as(src_offset * task_id)
        if self.support_data_move_pad:
            self.data_move_pad_bit64_2_int8(src_ub, self.src_gm[src_offset], 1,
                                            self.move_per_task * self.byte_per_data, 0, 0)
            self.data_move_pad_bit64_2_int8(self.dst_gm[dst_offset], src_ub, 1,
                                            self.move_per_task * self.byte_per_data, 0, 0)
        else:
            self.tik_instance.data_move(src_ub, self.src_gm[src_offset], 0, 1, 1, 0, 0)
            self.tik_instance.data_move(dst_ub, self.dst_gm[dst_offset], 0, 1, 1, 0, 0)
            with self.tik_instance.for_range(0, self.move_per_task) as i:
                dst_ub[i].set_as(src_ub[i])
            
            self.tik_instance.data_move(self.dst_gm[dst_offset], dst_ub, 0, 1, 1, 0, 0)

    def view_copy_process_tiling_mode_mix_set_as(self, task_id, src_ub, dst_ub, src_offset_start=0, dst_offset_start=0):
        with self.tik_instance.for_range(0, self.batch_num) as i:
            with self.tik_instance.if_scope(tik.any(i != self.batch_num - 1, self.tail_move_num == 0)):
                src_offset = task_id * self.move_per_task + \
                             i * self.batch_size + src_offset_start
                self.tik_instance.data_move(src_ub, self.src_gm[src_offset], 0, 1, self.move_rep_time_src, 0, 0)
                dst_offset = self.dst_offset + task_id * self.total_per_task + \
                             i * self.batch_size + dst_offset_start
                self.tik_instance.data_move(self.result_gm[dst_offset], src_ub, 0, 1, self.move_rep_time_src, 0, 0)
            with self.tik_instance.else_scope():
                src_offset = task_id * self.move_per_task + i * self.batch_size + src_offset_start
                self.tik_instance.data_move(src_ub, self.src_gm[src_offset],
                                            0, 1, self.move_rep_time_src_tail, 0, 0)
                
                move_offset = dst_offset_start + self.dst_offset + \
                            task_id * self.total_per_task + i * self.batch_size + \
                            self.move_burst * self.min_point_per_block - (self.min_point_per_block - self.left_num)
                self.tik_instance.data_move(dst_ub, self.dst_gm[move_offset], 0, 1, 1, 0, 0)
                with self.tik_instance.for_range(0, self.left_num) as j:
                    dst_ub[self.min_point_per_block - self.left_num + \
                           j].set_as(src_ub[self.move_burst * self.min_point_per_block + j])
                
                self.tik_instance.data_move(self.result_gm[move_offset], dst_ub, 0, 1, 1, 0, 0)
                
                dst_offset = self.dst_offset + task_id * self.total_per_task + \
                             i * self.batch_size + dst_offset_start
                self.tik_instance.data_move(self.result_gm[dst_offset], src_ub, 0, 1, self.move_burst, 0, 0)

    def view_copy_process_tiling_mode_mix_pad(self, task_id, src_ub, dst_ub, src_offset_start=0, dst_offset_start=0):
        with self.tik_instance.for_range(0, self.batch_num) as i:
            with self.tik_instance.if_scope(tik.any(i != self.batch_num - 1, self.tail_move_num == 0)):
                src_offset = task_id * self.move_per_task * self.src_move_size_rep + \
                             i * self.batch_size + src_offset_start
                self.tik_instance.data_move(src_ub, self.src_gm[src_offset], 0, 1, self.move_rep_time_src, 0, 0)
                dst_offset = self.dst_offset + task_id * self.total_per_task * self.src_move_size_rep + \
                             i * self.batch_size + dst_offset_start
                self.tik_instance.data_move(self.result_gm[dst_offset], src_ub, 0, 1, self.move_rep_time_src, 0, 0)
            with self.tik_instance.else_scope():
                src_offset = task_id * self.move_per_task * self.src_move_size_rep + \
                             i * self.batch_size + src_offset_start
                dst_offset = self.dst_offset + task_id * self.total_per_task * self.src_move_size_rep + \
                             i * self.batch_size + dst_offset_start
                with self.tik_instance.if_scope(self.task_num == task_id + 1):
                    self.data_move_pad_bit64_2_int8(src_ub, self.src_gm[src_offset],
                                                    self.move_stride_tail_src, self.move_rep_time_dst, 0, 0)
                    self.data_move_pad_bit64_2_int8(self.result_gm[dst_offset], src_ub,
                                                    self.move_stride_tail_src, self.move_rep_time_dst,
                                                    self.move_stride_dst, 0)
                with self.tik_instance.else_scope():
                    self.data_move_pad_bit64_2_int8(src_ub, self.src_gm[src_offset],
                                                    self.src_move_size_rep, self.move_rep_time_dst, 0, 0)
                    self.data_move_pad_bit64_2_int8(self.result_gm[dst_offset], src_ub, self.src_move_size_rep,
                                                    self.move_rep_time_dst, self.move_stride_dst, 0)

    def view_copy_process_tiling_mode_mix(self, task_id, src_ub, dst_ub, src_offset_start=0, dst_offset_start=0):

        if self.support_data_move_pad:
            self.view_copy_process_tiling_mode_mix_pad(task_id, src_ub, dst_ub, src_offset_start, dst_offset_start)
        else:
            self.view_copy_process_tiling_mode_mix_set_as(task_id, src_ub, dst_ub, src_offset_start, dst_offset_start)

    def view_copy_process_tiling_mode_trans(self, task_id, src_ub, dst_ub):
        dst_offset = task_id * self.dst_move_size + self.dst_offset
        self.tik_instance.data_move(dst_ub, self.dst_gm[dst_offset], 0, 1, self.move_rep_time_dst, 0, 0)
        src_offset = task_id * self.src_move_size
        self.tik_instance.data_move(src_ub, self.src_gm[src_offset], 0, 1, self.move_rep_time_dst, 0, 0)

        with self.tik_instance.for_range(0, self.move_per_task) as i:
            with self.tik_instance.for_range(0, self.total_per_task) as j:
                dst_ub[i * self.total_per_task + j].set_as(src_ub[self.move_per_task * j + i])
        
        self.tik_instance.data_move(self.result_gm[dst_offset], dst_ub, 0, 1, self.move_rep_time_dst, 0, 0)
    
    def view_copy_conv_process(self, task_id, src_ub, dst_ub, move_num):
        for i in range(Constant.CONV_LEN):
            dst_offset = task_id * self.data_align + i * self.batch_size + self.dst_offset
            dst_ub_offset = i * self.data_align
            self.tik_instance.data_move(dst_ub[dst_ub_offset], self.dst_gm[dst_offset],
                                        0, 1, self.move_rep_time_dst, 0, 0)
            src_offset = task_id * self.batch_num + i * self.move_burst
            src_ub_offset = i * self.batch_num
            self.tik_instance.data_move(src_ub[src_ub_offset], self.src_gm[src_offset],
                                        0, 1, self.move_rep_time_src, 0, 0)
                                        
        with self.tik_instance.new_stmt_scope():
            dst_ub_conv = self.tik_instance.Tensor(self.dtype, [self.dst_move_size],
                                                name="dst_ub_conv", scope=tik.scope_ubuf)
            dst_conv_list = [dst_ub_conv[i * self.min_point_per_block] for i in range(Constant.CONV_LEN)]
            dst_list = [dst_ub[i * self.data_align] for i in range(Constant.CONV_LEN)]
            self.tik_instance.vnchwconv(False, False, dst_conv_list, dst_list,
                                        self.move_rep_time_dst, self.min_point_per_block, 1)

            src_ub_conv = self.tik_instance.Tensor(self.dtype, [self.src_move_size],
                                                name="src_ub_conv", scope=tik.scope_ubuf)
            src_conv_list = [src_ub_conv[i * self.min_point_per_block] for i in range(Constant.CONV_LEN)]
            src_list = [src_ub[i * self.batch_num] for i in range(Constant.CONV_LEN)]
            self.tik_instance.vnchwconv(False, False, src_conv_list, src_list,
                                        self.move_rep_time_src, self.min_point_per_block, 1)

            move_diff = self.total_per_task - self.move_per_task
            self.tik_instance.data_move(dst_ub_conv, src_ub_conv, 0, move_num, self.move_per_task, 0, move_diff)
            self.tik_instance.vnchwconv(False, False, dst_list, dst_conv_list,
                                        self.move_rep_time_dst, 1, self.min_point_per_block)

    def view_copy_process_tiling_mode_conv(self, task_id, src_ub, dst_ub):
        self.view_copy_conv_process(task_id, src_ub, dst_ub, self.left_num)

        for i in range(Constant.CONV_LEN):
            dst_offset = task_id * self.data_align + i * self.batch_size + self.dst_offset
            dst_ub_offset = i * self.data_align
            self.tik_instance.data_move(self.dst_gm[dst_offset], dst_ub[dst_ub_offset], 0, 1,
                                        self.move_rep_time_dst, 0, 0)

    def view_copy_process_tiling_mode_conv_tail(self, task_id, src_ub, dst_ub):
        self.view_copy_conv_process(task_id, src_ub, dst_ub, self.left_num)

        for i in range(Constant.CONV_LEN):
            dst_offset = task_id * self.data_align + i * self.batch_size + self.dst_offset
            dst_ub_offset = i * self.data_align
            self.tik_instance.data_move(self.dst_gm[dst_offset], dst_ub[dst_ub_offset], 0, 1,
                                        self.move_rep_time_dst, 0, 0)

        self.view_copy_conv_process(task_id + 1, src_ub, dst_ub, self.tail_move_num)

        with self.tik_instance.if_scope(self.move_rep_time_dst_tail > 0):
            for i in range(Constant.CONV_LEN):
                dst_offset = (task_id + 1) * self.data_align + i * self.batch_size + self.dst_offset
                dst_ub_offset = i * self.data_align
                self.tik_instance.data_move(self.dst_gm[dst_offset], dst_ub[dst_ub_offset], 0, 1,
                                            self.move_rep_time_dst_tail, 0, 0)

        with self.tik_instance.if_scope(self.move_stride > 0):
            for i in range(Constant.CONV_LEN):
                dst_temp_ub = self.tik_instance.Tensor(self.dtype, [self.min_point_per_block],
                                                        name="dst_temp_ub", scope=tik.scope_ubuf)
                gm_offset = (i + 1) * self.batch_size - self.min_point_per_block + self.dst_offset
                self.tik_instance.data_move(dst_temp_ub, self.dst_gm[gm_offset], 0, 1, 1, 0, 0)
                diff = self.min_point_per_block - self.move_stride
                with self.tik_instance.for_range(0, self.move_stride) as j:
                    dst_temp_ub[j + diff].set_as(dst_ub[j + i * self.data_align])
                self.tik_instance.data_move(self.dst_gm[gm_offset], dst_temp_ub, 0, 1, 1, 0, 0)

    def run_core(self, task_id):
        with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_SINGLE_DATA_MOVE):
            src_ub = self.tik_instance.Tensor(self.dtype, [self.batch_size],
                                              name="src_ub", scope=tik.scope_ubuf)
            self.view_copy_process_tiling_mode_single_data_move(task_id, src_ub)
        with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_BATCH_DATA_MOVE):
            src_ub = self.tik_instance.Tensor(self.dtype, [self.src_move_size],
                                              name="src_ub", scope=tik.scope_ubuf)
            
            with self.tik_instance.if_scope(tik.any(task_id != self.task_num - 1, self.tail_align == 0)):
                self.view_copy_process_tiling_mode_batch_data_move(task_id, src_ub,
                                                                   self.move_rep_time_src,
                                                                   self.data_align, self.move_rep_time_dst)
            with self.tik_instance.else_scope():
                self.view_copy_process_tiling_mode_batch_data_move(task_id, src_ub,
                                                                   self.move_rep_time_src_tail,
                                                                   self.tail_align, self.move_rep_time_dst_tail)

        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_SINGLE_SET_AS):
                src_ub = self.tik_instance.Tensor(self.dtype, [self.src_move_size],
                                                name="src_ub", scope=tik.scope_ubuf)
                dst_ub = self.tik_instance.Tensor(self.dtype, [self.dst_move_size],
                                                name="dst_ub", scope=tik.scope_ubuf)
                self.view_copy_process_tiling_mode_single_set_as(task_id, src_ub, dst_ub)
            
            with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_MIX):
                src_ub = self.tik_instance.Tensor(self.dtype, [self.batch_size],
                                                name="src_ub", scope=tik.scope_ubuf)
                dst_ub = self.tik_instance.Tensor(self.dtype, [self.dst_move_size],
                                                name="dst_ub", scope=tik.scope_ubuf)

                self.view_copy_process_tiling_mode_mix(task_id, src_ub, dst_ub)
            
            with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_TRANS):
                src_ub = self.tik_instance.Tensor(self.dtype, [self.data_align],
                                                name="src_ub", scope=tik.scope_ubuf)
                dst_ub = self.tik_instance.Tensor(self.dtype, [self.data_align],
                                                name="dst_ub", scope=tik.scope_ubuf)
                
                self.view_copy_process_tiling_mode_trans(task_id, src_ub, dst_ub)
            with self.tik_instance.else_scope():
                if self.dtype == "float16":
                    with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_CONV):
                        src_ub = self.tik_instance.Tensor(self.dtype, [self.src_move_size],
                                                        name="src_ub", scope=tik.scope_ubuf)
                        dst_ub = self.tik_instance.Tensor(self.dtype, [self.dst_move_size],
                                                        name="dst_ub", scope=tik.scope_ubuf)

                        with self.tik_instance.if_scope(tik.any(task_id != self.task_num - 1, self.tail_align == 0)):
                            self.view_copy_process_tiling_mode_conv(task_id, src_ub, dst_ub)
                        with self.tik_instance.else_scope():
                            self.view_copy_process_tiling_mode_conv_tail(task_id, src_ub, dst_ub)
                
                with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_PARTIAL_TRANS):
                    src_ub = self.tik_instance.Tensor(self.dtype, [self.src_move_size],
                                                      name="src_ub", scope=tik.scope_ubuf)
                    if self.support_data_move_pad:
                        self.data_move_pad_bit64_2_int8(src_ub, self.src_gm, 1,
                                                        self.src_move_size * self.byte_per_data, 0, 0)
                    else:
                        self.tik_instance.data_move(src_ub, self.src_gm, 0, 1, self.move_rep_time_src, 0, 0)
                    dst_ub = self.tik_instance.Tensor(self.dtype, [self.dst_move_size],
                                                    name="dst_ub", scope=tik.scope_ubuf)
                    dst_offset = self.dst_offset + task_id * self.total_per_task
                    if self.support_data_move_pad:
                        with self.tik_instance.for_range(0, self.move_per_task) as i:
                            dst_ub[i].set_as(src_ub[task_id + i * self.task_num])
                        self.data_move_pad_bit64_2_int8(self.dst_gm[dst_offset], dst_ub, 1,
                                                        self.move_per_task * self.byte_per_data, 0, 0)
                    else:
                        self.tik_instance.data_move(dst_ub, self.dst_gm[dst_offset], 0, 1, self.move_rep_time_dst, 0, 0)
                        with self.tik_instance.for_range(0, self.move_per_task) as i:
                            dst_ub[i].set_as(src_ub[task_id + i * self.task_num])
                        
                        self.tik_instance.data_move(self.dst_gm[dst_offset], dst_ub, 0, 1, self.move_rep_time_dst, 0, 0)
                with self.tik_instance.if_scope(self.tiling_mode ==
                                                (Constant.TILING_MODE_SINGLE_DATA_MOVE + \
                                                 Constant.MODE_STRIDE_FOUR_DIMS)):
                    src_ub = self.tik_instance.Tensor(self.dtype, [self.batch_size],
                                                    name="src_ub", scope=tik.scope_ubuf)            
                    with self.tik_instance.for_range(0, self.dim_four) as index4:
                        with self.tik_instance.for_range(0, self.dim_three) as index3:
                            src_offset_start = index4 * self.dim_four_src_stride + index3 * self.dim_three_src_stride
                            dst_offset_start = index4 * self.dim_four_dst_stride + index3 * self.dim_three_dst_stride
                            self.view_copy_process_tiling_mode_single_data_move(task_id, src_ub,
                                                                                src_offset_start, dst_offset_start)        
                with self.tik_instance.if_scope(self.tiling_mode ==
                                                (Constant.TILING_MODE_SINGLE_DATA_MOVE + Constant.MODE_STRIDE)):
                    src_ub = self.tik_instance.Tensor(self.dtype, [self.batch_size],
                                                    name="src_ub", scope=tik.scope_ubuf)            
                    with self.tik_instance.for_range(0, self.dim_three) as index3:
                        src_offset_start = index3 * self.dim_three_src_stride
                        dst_offset_start = index3 * self.dim_three_dst_stride
                        self.view_copy_process_tiling_mode_single_data_move(task_id, src_ub,
                                                                            src_offset_start, dst_offset_start)    
                with self.tik_instance.if_scope(self.tiling_mode ==
                                                (Constant.TILING_MODE_MIX + Constant.MODE_STRIDE_FOUR_DIMS)):
                    src_ub = self.tik_instance.Tensor(self.dtype, [self.batch_size],
                                                    name="src_ub", scope=tik.scope_ubuf)
                    dst_ub = self.tik_instance.Tensor(self.dtype, [self.dst_move_size],
                                                    name="dst_ub", scope=tik.scope_ubuf)
                    with self.tik_instance.for_range(0, self.dim_four) as index4:
                        with self.tik_instance.for_range(0, self.dim_three) as index3:
                            src_offset_start = index4 * self.dim_four_src_stride + \
                                index3 * self.dim_three_src_stride
                            dst_offset_start = index4 * self.dim_four_dst_stride + \
                                index3 * self.dim_three_dst_stride
                            self.view_copy_process_tiling_mode_mix(task_id, src_ub, dst_ub,
                                                                   src_offset_start, dst_offset_start)
                with self.tik_instance.else_scope():
                    with self.tik_instance.if_scope(self.tiling_mode ==
                                                    (Constant.TILING_MODE_MIX + Constant.MODE_STRIDE)):
                        src_ub = self.tik_instance.Tensor(self.dtype, [self.batch_size],
                                                        name="src_ub", scope=tik.scope_ubuf)
                        dst_ub = self.tik_instance.Tensor(self.dtype, [self.dst_move_size],
                                                        name="dst_ub", scope=tik.scope_ubuf)
                        with self.tik_instance.for_range(0, self.dim_three) as index3:
                            src_offset_start = index3 * self.dim_three_src_stride
                            dst_offset_start = index3 * self.dim_three_dst_stride
                            self.view_copy_process_tiling_mode_mix(task_id, src_ub, dst_ub, 
                                                                   src_offset_start, dst_offset_start)
                    with self.tik_instance.if_scope(self.tiling_mode == 
                                                    Constant.TILING_MODE_THREE_DIMS_BATCH_DATA_MOVE):
                        src_ub = self.tik_instance.Tensor(self.dtype, [self.batch_size],
                                                        name="src_ub", scope=tik.scope_ubuf)
                        self.view_copy_process_tiling_mode_three_dims_batch_data_move(task_id, src_ub, 
                                                                                    0,
                                                                                    self.dst_offset)
                    with self.tik_instance.if_scope(self.tiling_mode == 
                                                    Constant.TILING_MODE_THREE_DIMS_BATCH_DATA_MOVE_WITH_TAIL):
                        src_ub = self.tik_instance.Tensor(self.dtype, [self.batch_size],
                                                        name="src_ub", scope=tik.scope_ubuf)
                        self.view_copy_process_tiling_mode_three_dims_batch_data_move_with_tail(task_id,
                                                                                                src_ub,
                                                                                                0,
                                                                                                self.dst_offset)
                    with self.tik_instance.if_scope(self.tiling_mode == 
                                                    Constant.TILING_MODE_FOUR_DIMS_BATCH_DATA_MOVE):
                        src_ub = self.tik_instance.Tensor(self.dtype, [self.batch_size],
                                                        name="src_ub", scope=tik.scope_ubuf)
                        with self.tik_instance.for_range(0, self.batch_num) as i:
                            self.view_copy_process_tiling_mode_three_dims_batch_data_move(i, src_ub, 
                                                                                        task_id *
                                                                                        self.move_per_task,
                                                                                        self.dst_offset +
                                                                                        task_id *
                                                                                        self.total_per_task)
                    with self.tik_instance.if_scope(self.tiling_mode == 
                                                    Constant.TILING_MODE_FOUR_DIMS_BATCH_DATA_MOVE_WITH_TAIL):
                        src_ub = self.tik_instance.Tensor(self.dtype, [self.batch_size],
                                                        name="src_ub", scope=tik.scope_ubuf)
                        with self.tik_instance.for_range(0, self.batch_num) as i:
                            self.view_copy_process_tiling_mode_three_dims_batch_data_move_with_tail(i, src_ub, 
                                                                                                    task_id *
                                                                                                    self.move_per_task,
                                                                                                    self.dst_offset +
                                                                                                    task_id *
                                                                                                    self.total_per_task)

    def run_core_batch_set_as(self, task_id):
        src_ub = self.tik_instance.Tensor(self.dtype, [self.src_move_size],
                                            name="src_ub", scope=tik.scope_ubuf)
        dst_ub = self.tik_instance.Tensor(self.dtype, [self.dst_move_size],
                                            name="dst_ub", scope=tik.scope_ubuf)
        
        with self.tik_instance.if_scope(tik.any(task_id != self.task_num - 1)):
            self.view_copy_process_tiling_mode_batch_set_as(task_id, src_ub, dst_ub)
        with self.tik_instance.else_scope():
            self.view_copy_process_tiling_mode_batch_set_as_tail(task_id, src_ub, dst_ub)

    def run_tik(self, kernel_name):
        self.get_tiling_data()

        batch_core_num = self.task_num // self.tiling_core_num
        batch_core_tail = self.task_num % self.tiling_core_num

        with self.tik_instance.for_range(0, self.tiling_core_num, block_num = self.tiling_core_num) as i:
            with self.tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_BATCH_SET_AS):
                with self.tik_instance.for_range(0, batch_core_num) as j:
                    self.run_core_batch_set_as(i + j * self.tiling_core_num)  
                with self.tik_instance.if_scope(i < batch_core_tail):
                    self.run_core_batch_set_as(batch_core_num * self.tiling_core_num + i)
            with self.tik_instance.else_scope():
                with self.tik_instance.for_range(0, batch_core_num) as j:
                    self.run_core(i + j * self.tiling_core_num)
                with self.tik_instance.if_scope(i < batch_core_tail):
                    self.run_core(batch_core_num * self.tiling_core_num + i)
    
        tbe_context.get_context().add_compile_info(
            "vars", {
                "full_core_num": self.avail_aicore_num,
                "ub_size": self.available_ub_size
            }
        )

        self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=[self.dst_gm, self.dst_size_gm, self.dst_stride_gm, self.dst_offset_gm,
                                   self.src_gm, self.src_size_gm, self.src_stride_gm, self.src_offset_gm],
                                   outputs=[self.result_gm],
                                   flowtable = [self.tiling_gm])

        return self.tik_instance


# pylint: disable=too-many-arguments
@register_operator("ViewCopy")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, 
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def view_copy(dst, dst_size, dst_stride, dst_storage_offset, src, src_size, src_stride, src_storage_offset, 
              result, kernel_name="view_copy"):
    view_copy_obj = ViewCopy(dst, dst_size, dst_stride, dst_storage_offset, 
                             src, src_size, src_stride, src_storage_offset)
    res = view_copy_obj.run_tik(kernel_name)

    return res