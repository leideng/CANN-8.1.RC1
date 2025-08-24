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
trace
"""

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant
    """
    TILING_ARG_NUM = 5
    MAX_SHAPE_SIZE = 2 ** 32 - 1
    TILING_MODE_1 = 1
    TILING_MODE_2 = 2
    MASK = 64
    MAX_REPEAT_NUM = 255
    BF16_DTYPE = "bfloat16"


def get_bytes_len(dtype):
    """
    Parameters
    ----------
    dtype: input dtype

    Returns
    -------
    dtype Btypes
    """

    index = 0
    for i in dtype:
        if i.isdigit():
            break
        index += 1
    return int(dtype[index:]) // 8


def ceil_value(value, factor):
    """
    if not divide exactly then plus 1

    Parameters
    ----------
    value: input number
    factor: factor

    Returns
    -------
    ceil value
    """

    return (value + factor - 1) // factor


class TikTrace():
    """
    trace init
    """

    def __init__(self, input_data, kernel_name):
        self.tik_instance = tik.Tik()
        self.input_data_shape = input_data.get("shape")
        self.kernel_name = kernel_name

        self.ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.block_byte_size = 32

        self.input_dtype = input_data.get("dtype")
        self.input_dtype_bytes_size = get_bytes_len(self.input_dtype)
        self.data_each_block = (self.block_byte_size
            // self.input_dtype_bytes_size)
        self.input_x_gm = self.tik_instance.Tensor(self.input_dtype,
            (Constant.MAX_SHAPE_SIZE,), name="input_x_gm", scope=tik.scope_gm)
        self.output_y_gm = self.tik_instance.Tensor(self.input_dtype,
            (1,), name="output_y_gm", scope=tik.scope_gm, is_atomic_add=True)
        self.dtype_fp32_bytes_size = get_bytes_len("float32")

        # tiling params
        self.tiling_dtype = "int64"
        self.tiling_block_num = ceil_value(get_bytes_len(self.tiling_dtype)
            * Constant.TILING_ARG_NUM, self.block_byte_size)
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype,
            (Constant.TILING_ARG_NUM,), name="tiling_gm", scope=tik.scope_gm)

        self.tiling_ub = None
        self.input_h = None
        self.input_w = None
        self.tiling_mode = None
        self.need_core_num = None
        self.core_num_var = None
        self.metrix_rank = None
        self.aicore_num = None
        self.aicore_output_gm = None
        self.data_num_each_core = None
        self.aicore_comp_ub = None
        self.metrix_sum_ub = None
        self.aicore_proc_cnt = None
        self.caculate_data_dtype = None
        self.caculate_each_block = None
        self.sync_workspace = None

    def _get_tiling_args(self):
        """
        get runtime params from tiling data

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.tiling_ub = self.tik_instance.Tensor(self.tiling_dtype,
            (self.tiling_block_num * Constant.TILING_ARG_NUM,), name="tiling_ub",
            scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1,
            self.tiling_block_num, 0, 0)

        self.input_h = self.tik_instance.Scalar(dtype=self.tiling_dtype,
            name="input_h")
        self.input_w = self.tik_instance.Scalar(dtype=self.tiling_dtype,
            name="input_w")
        self.tiling_mode = self.tik_instance.Scalar(dtype=self.tiling_dtype,
            name="tiling_mode")
        self.need_core_num = self.tik_instance.Scalar(dtype=self.tiling_dtype,
            name="need_core_num")
        self.core_num_var = self.tik_instance.Scalar(dtype=self.tiling_dtype,
            name="core_num_var")

        self.input_h.set_as(self.tiling_ub[0])
        self.input_w.set_as(self.tiling_ub[1])
        self.tiling_mode.set_as(self.tiling_ub[2])
        self.need_core_num.set_as(self.tiling_ub[3])
        self.core_num_var.set_as(self.tiling_ub[4])

    def _init_process_args(self):
        """
        get process params from input

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.metrix_rank = self.tik_instance.Scalar(dtype=self.tiling_dtype,
            name="metrix_rank")
        self.tik_instance.scalar_min(self.metrix_rank, self.input_h, self.input_w)

        self.aicore_num = self.tik_instance.Scalar(dtype="int32",
            name="aicore_num", init_value=self.need_core_num)
        
        # fp16 or bf16 should be converted to fp32 to avoid precision loss
        if self.input_dtype == "float16" or self.input_dtype == Constant.BF16_DTYPE:
            self.caculate_data_dtype = "float32"
            self.caculate_each_block = (self.block_byte_size // self.dtype_fp32_bytes_size)
        else:
            self.caculate_data_dtype = self.input_dtype
            self.caculate_each_block = self.data_each_block

        # Define temporary gm space in multi-core processing
        self.aicore_output_gm = self.tik_instance.Tensor(self.caculate_data_dtype,
            shape=(self.core_num, self.caculate_each_block),
            name="aicore_output_gm", scope=tik.scope_gm, is_workspace=True, is_atomic_add=True)
        self.data_num_each_core = self.tik_instance.Scalar(dtype="int32",
            name="data_num_each_core")
        self.data_num_each_core.set_as(ceil_value(self.metrix_rank,
            self.aicore_num))
        # The maximum number of blocks processed by aicore at a time
        if self.input_dtype == "float16" or self.input_dtype == Constant.BF16_DTYPE:
            self.aicore_proc_cnt = 2047
        else:
            self.aicore_proc_cnt = 4095
        
        if self.input_dtype == Constant.BF16_DTYPE:
            self.core_num_var.set_as(self.need_core_num)
            self.sync_workspace = self.tik_instance.Tensor('int64', (self.core_num * 4, ),
                                                            tik.scope_gm,
                                                            'sync_workspace',
                                                            is_workspace=True,
                                                            is_atomic_add=True)

# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
    def trace_computer(self):
        """
        main process of trace dynamic shape

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self._get_tiling_args()
        self._init_process_args()

        with self.tik_instance.for_range(0, self.core_num_var,
            block_num=self.core_num_var) as index:
            with self.tik_instance.if_scope(index < self.need_core_num):
                zero_scalar = self.tik_instance.Scalar(self.caculate_data_dtype, init_value=0)
                self.metrix_sum_ub = self.tik_instance.Tensor(self.caculate_data_dtype,
                    shape=(self.caculate_each_block, 1), name="metrix_sum_ub",
                    scope=tik.scope_ubuf)
                self.tik_instance.vec_dup(self.caculate_each_block, self.metrix_sum_ub, zero_scalar, 1, 0)

                move_offset = index * self.data_num_each_core
                process_limit = self.tik_instance.Scalar(dtype="int32",
                    name="process_limit",
                    init_value=move_offset + self.data_num_each_core)
                with self.tik_instance.if_scope(process_limit <= self.metrix_rank):
                    self._trace_computer_each_core(index, move_offset, self.data_num_each_core)
                with self.tik_instance.else_scope():
                    tail_cnt = self.tik_instance.Scalar(
                        dtype="int32", name="tail_cnt",
                        init_value=self.metrix_rank - ((self.aicore_num - 1) * self.data_num_each_core))
                    self._trace_computer_each_core(index, move_offset, tail_cnt)
                
                if tbe_platform.api_check_support("tik.set_atomic_add", "float32") and \
                                                self.input_dtype != Constant.BF16_DTYPE:
                    self._trace_computer_all_core()
                elif self.input_dtype == Constant.BF16_DTYPE:
                    self._trace_data_move_for_bf16(index)
                else:
                    self.tik_instance.data_move(self.aicore_output_gm[index, 0],
                                        self.metrix_sum_ub, 0, 1, 1, 0, 0)

        if not tbe_platform.api_check_support("tik.set_atomic_add", "float32") or \
                                          self.input_dtype == Constant.BF16_DTYPE:
            self._trace_computer_all_core_for_310_or_bf16()

        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        tbe_context.get_context().add_compile_info("vars", {
            "core_num": self.core_num
        })
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
            inputs=[self.input_x_gm], outputs=[self.output_y_gm],
            flowtable=[self.tiling_gm],
            config=opt_config)
        return self.tik_instance

    def data_conv(self, dst, src, offsets, mask=64, mode="ceil", num=0, dst_stride=8, src_stride=8):
        """
        tik conv
        """
        dst_offset = self.tik_instance.Scalar("int32", init_value=offsets[0])
        src_offset = self.tik_instance.Scalar("int32", init_value=offsets[1])

        tensor_size = num
        loop = tensor_size // (mask * Constant.MAX_REPEAT_NUM)

        dst_gap = dst_stride * 32 // get_bytes_len(dst.dtype)
        src_gap = src_stride * 32 // get_bytes_len(src.dtype)

        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst_offset = dst_offset + index * Constant.MAX_REPEAT_NUM * dst_gap
                tmp_src_offset = src_offset + index * Constant.MAX_REPEAT_NUM * src_gap
                self.tik_instance.vec_conv(mask, mode, dst[tmp_dst_offset], src[tmp_src_offset],
                                           Constant.MAX_REPEAT_NUM, dst_stride, src_stride)

            dst_offset.set_as(dst_offset + loop * dst_gap * Constant.MAX_REPEAT_NUM)
            src_offset.set_as(src_offset + loop * src_gap * Constant.MAX_REPEAT_NUM)

        repeat_time_last = (tensor_size % (mask * Constant.MAX_REPEAT_NUM)) // mask

        with self.tik_instance.if_scope(repeat_time_last > 0):
            self.tik_instance.vec_conv(mask, mode, dst[dst_offset], src[src_offset], repeat_time_last,
                                       dst_stride, src_stride)
            dst_offset.set_as(dst_offset + repeat_time_last * dst_gap)
            src_offset.set_as(src_offset + repeat_time_last * src_gap)

        last_num = tensor_size % mask
        with self.tik_instance.if_scope(last_num > 0):
            self.tik_instance.vec_conv(last_num, mode, dst[dst_offset], src[src_offset], 1, dst_stride, src_stride)

    def _trace_computer_each_core(self, index, move_offset, proc_cnt):
        """
        Calculate the matrix data in each ai core

        Parameters
        ----------
        move_offset: The beginning index of the input data
        proc_cnt: The count of the input data

        Returns
        -------
        Sum of matrix data
        """

        origin_move_offset = move_offset
        loop_time = proc_cnt // self.aicore_proc_cnt
        with self.tik_instance.if_scope(loop_time > 0):
            with self.tik_instance.for_range(0, loop_time) as loop_index:
                move_offset += loop_index * self.aicore_proc_cnt
                self._trace_computer_each_matrix(move_offset,
                    self.aicore_proc_cnt)
            move_offset = origin_move_offset + loop_time * self.aicore_proc_cnt
        last_cnt = proc_cnt % self.aicore_proc_cnt
        with self.tik_instance.if_scope(last_cnt > 0):
            self._trace_computer_each_matrix(move_offset, last_cnt)

    def _trace_tiling_proc_big(self, move_offset, proc_cnt):
        """
        When the mode is tiling_mode_2 or tiling_mode_3,
        Process the bigger inputs

        Parameters
        ----------
        move_offset: The beginning index of the input data
        proc_cnt: The count of the input data

        Returns
        -------
        Sum of matrix data
        """

        comp_line_num = proc_cnt - 1
        with self.tik_instance.for_range(0, comp_line_num) as i:
            diag_idx = self.tik_instance.Scalar(dtype="int64", name="diag_idx",
                init_value=((i + move_offset) * (self.input_w + 1)))
            self.tik_instance.data_move(self.aicore_comp_ub[i, 0],
                self.input_x_gm[diag_idx], 0, 1, 1, 0, 0)

        last_data_idx = self.tik_instance.Scalar(
                dtype="int64", name="last_data_idx",
                init_value=(comp_line_num + move_offset) * (self.input_w + 1))

        with self.tik_instance.if_scope(last_data_idx == 0):
            self.tik_instance.data_move(self.aicore_comp_ub[comp_line_num, 0], 
                                        self.input_x_gm[last_data_idx], 0, 1, 1, 0, 0)
        with self.tik_instance.else_scope():
            last_data_idx = last_data_idx - self.data_each_block + 1
            self.tik_instance.data_move(
                self.aicore_comp_ub[comp_line_num, 0], self.input_x_gm[last_data_idx], 0, 1, 1, 0, 0)
            self.aicore_comp_ub[comp_line_num, 0].set_as(
                self.aicore_comp_ub[comp_line_num, self.data_each_block - 1])

    def _trace_tiling_proc_small(self, move_offset, proc_cnt):
        """
        When the mode is tiling_mode_1, Process the smaller inputs

        Parameters
        ----------
        move_offset: The beginning index of the input data
        proc_cnt: The count of the input data

        Returns
        -------
        Sum of matrix data
        """

        block_num = ceil_value(proc_cnt * self.input_w, self.data_each_block)
        tmp_ub = self.tik_instance.Tensor(self.input_dtype,
            shape=(block_num * self.data_each_block, 1), name="tmp_ub",
            scope=tik.scope_ubuf)
        burst_len = proc_cnt * self.input_w // self.data_each_block
        with self.tik_instance.if_scope(burst_len > 0):
            move_start_idx = self.tik_instance.Scalar(dtype="int64",
                name="move_start_idx", init_value=move_offset * self.input_w)
            self.tik_instance.data_move(tmp_ub,
                self.input_x_gm[move_start_idx], 0, 1, burst_len, 0, 0)
            last_num = proc_cnt * self.input_w % self.data_each_block

            with self.tik_instance.if_scope(last_num > 0):
                ub_base_offset = burst_len * self.data_each_block
                gm_base_offset = (move_offset * self.input_w
                    + burst_len * self.data_each_block)
                gm_back_offset = last_num - self.data_each_block
                gm_last_start_idx = gm_base_offset + gm_back_offset
                self.tik_instance.data_move(tmp_ub[ub_base_offset],
                    self.input_x_gm[gm_last_start_idx], 0, 1, 1, 0, 0)
                with self.tik_instance.for_range(0, last_num) as i:
                    tmp_ub[ub_base_offset + i, 0].set_as(
                        tmp_ub[ub_base_offset - gm_back_offset + i, 0])
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(tmp_ub,
                self.input_x_gm[move_offset], 0, 1, 1, 0, 0)

        with self.tik_instance.for_range(0, proc_cnt) as i:
            diag_idx = self.tik_instance.Scalar(dtype="int64",
                name="diag_idx", init_value=move_offset +
                i * (self.input_w + 1))
            self.aicore_comp_ub[i, 0].set_as(tmp_ub[diag_idx])

    def _trace_computer_each_matrix(self, move_offset, proc_cnt):
        """
        Calculate the part matrix data in each ai core

        Parameters
        ----------
        move_offset: The beginning index of the input data
        proc_cnt: The count of the input data

        Returns
        -------
        Sum of part matrix data
        """
        # aicore_comp_ub is used to store data moved from gm to ubuf
        self.aicore_comp_ub = self.tik_instance.Tensor(self.input_dtype,
            shape=(proc_cnt, self.data_each_block),
            name="aicore_comp_ub", scope=tik.scope_ubuf)

        with self.tik_instance.if_scope(self.tiling_mode ==
            Constant.TILING_MODE_1):
            self._trace_tiling_proc_small(move_offset, proc_cnt)
        with self.tik_instance.if_scope(self.tiling_mode ==
            Constant.TILING_MODE_2):
            self._trace_tiling_proc_big(move_offset, proc_cnt)
        
        work_tensor_ub = self.tik_instance.Tensor(self.caculate_data_dtype,
            shape=(proc_cnt,), name="work_tensor_ub", scope=tik.scope_ubuf)
        zero_scalar = self.tik_instance.Scalar(dtype=self.caculate_data_dtype,
            init_value=0)
        add_tensor_a = self.tik_instance.Tensor(self.caculate_data_dtype,
            shape=(self.caculate_each_block, 1), name="add_tensor_a",
            scope=tik.scope_ubuf)
        self.tik_instance.vec_dup(self.caculate_each_block, add_tensor_a,
                zero_scalar, 1, 0)
        if self.input_dtype == "float16" or self.input_dtype == Constant.BF16_DTYPE:
            aicore_comp_ub_fp32 = self.tik_instance.Tensor(self.caculate_data_dtype,
                shape=(proc_cnt, self.caculate_each_block),
                name="aicore_comp_ub_fp32", scope=tik.scope_ubuf)

            self.data_conv(aicore_comp_ub_fp32, self.aicore_comp_ub, [0, 0], mask=self.caculate_each_block,
                           mode="", num=proc_cnt * self.caculate_each_block, dst_stride=1, src_stride=1)
            self.tik_instance.vec_reduce_add(1, add_tensor_a, aicore_comp_ub_fp32,
                work_tensor_ub, proc_cnt, 1)
            self.tik_instance.vec_add(1, self.metrix_sum_ub, add_tensor_a,
                self.metrix_sum_ub, 1, 0, 0, 0)
            
        else:
            self.tik_instance.vec_reduce_add(1, add_tensor_a, self.aicore_comp_ub,
                work_tensor_ub, proc_cnt, 1)
            self.tik_instance.vec_add(1, self.metrix_sum_ub, add_tensor_a,
                self.metrix_sum_ub, 1, 0, 0, 0)

    def _trace_computer_all_core_for_310_or_bf16(self):
        """
        Sum the output of each core

        Parameters
        ----------
        None

        Returns
        -------
        sum data
        """
        aicore_all_input = self.tik_instance.Tensor(self.caculate_data_dtype,
            shape=(self.aicore_num, self.caculate_each_block),
            name="aicore_all_input", scope=tik.scope_ubuf)
        self.tik_instance.data_move(aicore_all_input,
            self.aicore_output_gm, 0, 1, self.aicore_num, 0, 0)
        work_tensor_ub = self.tik_instance.Tensor(self.caculate_data_dtype,
            shape=(self.aicore_num, self.caculate_each_block),
            name="work_tensor_ub", scope=tik.scope_ubuf)
        sum_tensor = self.tik_instance.Tensor(self.caculate_data_dtype,
            shape=(self.caculate_each_block, 1), name="sum_tensor",
            scope=tik.scope_ubuf)
        self.tik_instance.vec_reduce_add(1, sum_tensor, aicore_all_input,
            work_tensor_ub, self.aicore_num, 1)

        if self.input_dtype == "float16":
            sum_tensor_out = self.tik_instance.Tensor(self.input_dtype,
                shape=(self.data_each_block, 1), name="sum_tensor_out",
                scope=tik.scope_ubuf)

            self.data_conv(sum_tensor_out, sum_tensor, [0, 0], mode="",
                           num=self.data_each_block, dst_stride=4, src_stride=8)
            self.tik_instance.data_move(self.output_y_gm[0], sum_tensor_out, 0, 1, 1, 0, 0)

        elif self.input_dtype == Constant.BF16_DTYPE:
            sum_tensor_out = self.tik_instance.Tensor(self.input_dtype,
                        shape=(self.data_each_block, 1), name="sum_tensor_out",
                        scope=tik.scope_ubuf)
            self.tik_instance.vec_conv(1, 'round', sum_tensor_out, sum_tensor, 1, 4, 8)
            self.tik_instance.data_move_pad(self.output_y_gm, sum_tensor_out, 1, 2, 0, 0)

        else:
            self.tik_instance.data_move(self.output_y_gm[0], sum_tensor, 0, 1, 1, 0, 0)

    def _trace_data_move_for_bf16(self, index):
        self.tik_instance.data_move(self.aicore_output_gm[index, 0],
                                        self.metrix_sum_ub, 0, 1, 1, 0, 0)
        with self.tik_instance.if_scope(self.core_num_var > 1):
            self.tik_instance.block_barrier(self.sync_workspace)

    def _trace_computer_all_core(self):
        if self.input_dtype == "float16":
            output_y_ub = self.tik_instance.Tensor(self.input_dtype,
                (1,), name="output_y_ub", scope=tik.scope_ubuf)
            output_y_ub_fp32 = self.tik_instance.Tensor(self.caculate_data_dtype,
                (1,), name="output_y_ub_fp32", scope=tik.scope_ubuf)
            self.tik_instance.set_atomic_add(1)
            self.tik_instance.data_move(self.aicore_output_gm[0, 0],
                self.metrix_sum_ub, 0, 1, 1, 0, 0)
            self.tik_instance.set_atomic_add(0)

            self.tik_instance.data_move(output_y_ub_fp32, self.aicore_output_gm, 0, 1, 1, 0, 0)
            self.tik_instance.vec_conv(1, 'none', output_y_ub, output_y_ub_fp32, 1, 4, 8)
            self.tik_instance.data_move(self.output_y_gm[0], output_y_ub, 0, 1, 1, 0, 0)
        else:
            self.tik_instance.set_atomic_add(1)
            self.tik_instance.data_move(self.output_y_gm[0],
                self.metrix_sum_ub, 0, 1, 1, 0, 0)
            self.tik_instance.set_atomic_add(0)


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
@register_operator("Trace")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                    para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def trace(input_x, output_y, kernel_name="trace"):
    """
    Operation for trace.

    Parameters
    ----------
    input_data: 2D metrix of input, include shape and dtype, dtype support float16, float
    kernel_name: cce kernel name, default value is trace

    Returns
    -------
    tik_instance
    """

    trace_instance = TikTrace(input_x, kernel_name)
    tik_instance = trace_instance.trace_computer()
    return tik_instance