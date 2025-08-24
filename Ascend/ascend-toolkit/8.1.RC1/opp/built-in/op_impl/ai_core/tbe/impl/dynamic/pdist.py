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
pdist
"""

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_platform


# 'pylint: disable=too-few-public-methods,invalid-name,unused-variable
class Constant:
    """The class for constant.
    """
    BLOCK_BITE_SIZE = 32
    WORK_TENSOR_SIZE = 256
    TOTAL_SCALAR_SIZE = 17
    TEMP_SUM_TENSOR_BYTES = 32
    CONV_NUM_ONE_REPEAT = 64
    FP16_BYTE_SIZE = 2
    FP32_BYTE_SIZE = 4
    NUM_FP32_EACH_BLOCK = 8
    MAX_NUM_ONE_REPEAT = 64

    MAX_INT32 = 2 ** 31 - 1
    TILING_SCALAR_DTYPE = "int32"
    TILING_PARAMS_NUM = 8
    NUM_EACH_BURST = 8

    ROWS_IDX = 0
    COLS_IDX = 1
    COMPUTE_NUM_IDX = 2
    NUM_EACH_CORE_IDX = 3
    NUM_EACH_LOOP_IDX = 4
    UB_TENSOR_EACH_LOOP_IDX = 5
    CORE_NUM_VAR_IDX = 6
    P_IDX = 7


def _get_ceil_int(int1, int2):
    ceil_int = (int1 + int2 - 1) // int2
    return ceil_int


@register_operator("pdist")
# 'pylint: disable=too-many-instance-attributes,too-many-lines
class Pdist():
    """Function: use to finish Pdist main functions
    """

    # 'pylint: disable=too-many-statements,too-many-arguments
    def __init__(self, input_x, output_y, p, kernel_name="pdist"):
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.kernel_name = kernel_name
        self.ai_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

        self.dtype_x = input_x.get("dtype").lower()
        self.dtype_y = output_y.get("dtype").lower()
        # check param
        self.check_param()

        self.rows = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="rows")
        self.cols = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="cols")
        self.compute_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="compute_num")
        self.num_each_core = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="num_each_core")
        self.num_each_loop = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="num_each_loop")
        self.ub_tensor_each_loop = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, name="ub_tensor_each_loop")
        self.core_num_var = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, 
                                                     name="core_num_var", 
                                                     init_value=self.ai_core_num)
        self.p = self.tik_instance.Scalar("float32", name="p")

        # request gm
        self.input_x_gm = self.tik_instance.Tensor(self.dtype_x, [Constant.MAX_INT32],
                                                   name="input_x_gm", scope=tik.scope_gm)
        self.output_y_gm = self.tik_instance.Tensor(self.dtype_y, [Constant.MAX_INT32],
                                                    name="output_y_gm", scope=tik.scope_gm)
        self.tiling_gm = self.tik_instance.Tensor(Constant.TILING_SCALAR_DTYPE, [Constant.TILING_PARAMS_NUM],
                                                  name="tiling_gm", scope=tik.scope_gm)

        # get from tiling
        self.get_tiling_args()

        # get size of UB and compute the number of elements can be stored in a block
        self.ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.dtype_bytes_size = tbe_platform.get_bit_len(self.dtype_x) // 8
        self.data_each_block = Constant.BLOCK_BITE_SIZE // self.dtype_bytes_size

        # compute the tensor size except src1_ub and src2_ub and src_temp_fp16_tensor
        output_tensor_size = self.data_each_block

        others_tensor_size_bytes = Constant.TEMP_SUM_TENSOR_BYTES + \
                                   (Constant.WORK_TENSOR_SIZE + output_tensor_size + Constant.TOTAL_SCALAR_SIZE) * \
                                   Constant.FP32_BYTE_SIZE

        if self.dtype_x == "float16":
            dst_sum_fp16_tensor_size = self.data_each_block
            others_tensor_size_bytes = others_tensor_size_bytes + dst_sum_fp16_tensor_size * Constant.FP16_BYTE_SIZE

        if others_tensor_size_bytes > self.ub_size_bytes:
            raise RuntimeError("Out of Unified buffer")

        self.ub_size_bytes = self.ub_size_bytes - others_tensor_size_bytes

    def get_tiling_args(self):
        """get tiling args from tiling_ub
        """
        tiling_ub = self.tik_instance.Tensor(Constant.TILING_SCALAR_DTYPE, [Constant.TILING_PARAMS_NUM, ],
                                             name="tiling_ub", scope=tik.scope_ubuf)
        burst_val = _get_ceil_int(Constant.TILING_PARAMS_NUM, Constant.NUM_EACH_BURST)
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, burst_val, 0, 0)
        self.rows.set_as(tiling_ub[Constant.ROWS_IDX])
        self.cols.set_as(tiling_ub[Constant.COLS_IDX])
        self.compute_num.set_as(tiling_ub[Constant.COMPUTE_NUM_IDX])
        self.num_each_core.set_as(tiling_ub[Constant.NUM_EACH_CORE_IDX])
        self.num_each_loop.set_as(tiling_ub[Constant.NUM_EACH_LOOP_IDX])
        self.ub_tensor_each_loop.set_as(tiling_ub[Constant.UB_TENSOR_EACH_LOOP_IDX])
        self.core_num_var.set_as(tiling_ub[Constant.CORE_NUM_VAR_IDX])
        self.p.set_as(tiling_ub[Constant.P_IDX])

    # 'pylint: disable=too-many-locals,too-many-branches,too-many-lines
    def pdist_compute(self):
        # blocks processed by each core
        num_block_each_core = self.compute_num // self.core_num_var // self.data_each_block
        # last nums processed by each core
        last_nums = self.compute_num % (self.data_each_block * self.core_num_var)
        # last blocks processed by each_core
        last_nums_blocks = last_nums // self.data_each_block
        # last nums processed by none full block
        last_nums_none_full_block = last_nums % self.data_each_block

        with self.tik_instance.if_scope(num_block_each_core > 0):
            with self.tik_instance.for_range(0, self.core_num_var, block_num=self.core_num_var) as core_id:
                self.init_ub_tensor_and_scalar()
                with self.tik_instance.for_range(0, num_block_each_core) as block_num_id:
                    output_y_gm_index = core_id * self.data_each_block * num_block_each_core + \
                                        block_num_id * self.data_each_block
                    self.out_index_float.set_as(output_y_gm_index)
                    with self.tik_instance.for_range(0, self.data_each_block) as k:    
                        self.get_i_j_from_index(output_y_gm_index, k)
                        self.pdist_compute_each_core(self.i_int * self.num_each_core,
                                                     self.j_int * self.num_each_core, k)
                    with self.tik_instance.if_scope(self.p > 0):
                        self.pdist_sum_process(self.data_each_block, 0, 1)
                    # to improve the accuracy, convert fp16 to fp32, convert it back to fp16 after calculation
                    if self.dtype_y == "float16":
                        self.pdist_convert(self.dst_sum_fp16_tensor, self.dst_sum_tensor, self.data_each_block,
                                           "float32")
                        self.tik_instance.data_move(self.output_y_gm[output_y_gm_index], self.dst_sum_fp16_tensor,
                                                    0, 1, 1, 0, 0)
                    else:
                        self.tik_instance.data_move(self.output_y_gm[output_y_gm_index], self.dst_sum_tensor,
                                                    0, 1, 1, 0, 0)

        with self.tik_instance.if_scope(last_nums > 0):
            with self.tik_instance.for_range(0, last_nums_blocks) as last_block_id:
                self.init_ub_tensor_and_scalar()
                output_y_gm_index = self.core_num_var * num_block_each_core * self.data_each_block + \
                                    last_block_id * self.data_each_block
                self.out_index_float.set_as(output_y_gm_index)
                with self.tik_instance.for_range(0, self.data_each_block) as k:
                    self.get_i_j_from_index(output_y_gm_index, k)
                    self.pdist_compute_each_core(self.i_int * self.num_each_core,
                                                 self.j_int * self.num_each_core, k)
                with self.tik_instance.if_scope(self.p > 0):
                    self.pdist_sum_process(self.data_each_block, 0, 1)
                # to improve the accuracy, convert fp16 to fp32, convert it back to fp16 after calculation
                if self.dtype_y == "float16":
                    self.pdist_convert(self.dst_sum_fp16_tensor, self.dst_sum_tensor, self.data_each_block, "float32")
                    self.tik_instance.data_move(self.output_y_gm[output_y_gm_index], self.dst_sum_fp16_tensor, 
                                                0, 1, 1, 0, 0)
                else:
                    self.tik_instance.data_move(self.output_y_gm[output_y_gm_index], self.dst_sum_tensor, 
                                                0, 1, 1, 0, 0)
        
        with self.tik_instance.if_scope(last_nums_none_full_block > 0):
            with self.tik_instance.new_stmt_scope():
                self.init_ub_tensor_and_scalar()
                with self.tik_instance.for_range(0, last_nums_none_full_block) as k:
                    output_y_gm_index = self.core_num_var * num_block_each_core * self.data_each_block + \
                                        last_nums_blocks * self.data_each_block
                    self.out_index_float.set_as(output_y_gm_index)
                    self.get_i_j_from_index(output_y_gm_index, k)
                    self.pdist_compute_each_core(self.i_int * self.num_each_core,
                                                 self.j_int * self.num_each_core, k)
                with self.tik_instance.if_scope(self.p > 0):
                    self.pdist_sum_process(self.data_each_block, 0, 1)
                # to improve the accuracy, convert fp16 to fp32, convert it back to fp16 after calculation
                if self.dtype_y == "float16":
                    self.pdist_convert(self.dst_sum_fp16_tensor, self.dst_sum_tensor, self.data_each_block,
                                    "float32")
                    self.tik_instance.data_move(self.output_y_gm[output_y_gm_index], self.dst_sum_fp16_tensor,
                                                0, 1, 1, 0, 0)
                else:
                    self.tik_instance.data_move(self.output_y_gm[output_y_gm_index], self.dst_sum_tensor,
                                                0, 1, 1, 0, 0)
        
        tbe_context.get_context().add_compile_info(
            "vars", {
                "ub_size": self.ub_size_bytes,
                "core_num": self.ai_core_num
            })

    def get_i_j_from_index(self, index_id, k):
        """
        infers what i and j are from the value of index_id
        we iterate over tuples of (i, j, index_id + k)
        where i is the first vector from the input, j is the second, and index_id + k is the result index
        """
        i_minus_half = self.rows_float - .5
        expr_squared = i_minus_half * i_minus_half - 2 * self.out_index_float - 2 * k
        self.squared_sqrt.set_as(expr_squared)
        self.tik_instance.scalar_sqrt(self.squared_sqrt, self.squared_sqrt)
        expr_i = i_minus_half - self.squared_sqrt
        # i round down
        self.i_int.set_as(expr_i)
        expr_j = index_id + k - self.rows * self.i_int + self.i_int * \
                 (self.i_int + 1) / 2 + self.i_int + 1
        with self.tik_instance.if_scope(self.i_int >= expr_j):
            self.i_int.set_as(expr_i - 1)
            expr_j = index_id + k - self.rows * self.i_int + self.i_int * \
                     (self.i_int + 1) / 2 + self.i_int + 1
        # convert float32->int32
        self.j_int.set_as(expr_j)

    # compute on each core
    def pdist_compute_each_core(self, src_offset1, src_offset2, index_id):
        loop_times = self.num_each_core // self.num_each_loop
        num_last_loop = self.num_each_core - self.num_each_loop * loop_times
        self.dst_sum_tensor[index_id].set_as(self.scalar_zero_loop)

        start_double_buffer = True
        with self.tik_instance.if_scope(loop_times < 2):
            start_double_buffer = False
        if start_double_buffer:
            with self.tik_instance.for_range(0, loop_times, thread_num=2) as loop:
                move_src_offset1 = src_offset1 + loop * self.num_each_loop
                move_src_offset2 = src_offset2 + loop * self.num_each_loop
                self.pdist_compute_each_loop(move_src_offset1, move_src_offset2, self.num_each_loop, index_id)
        else:
            with self.tik_instance.for_range(0, loop_times) as loop:
                move_src_offset1 = src_offset1 + loop * self.num_each_loop
                move_src_offset2 = src_offset2 + loop * self.num_each_loop
                self.pdist_compute_each_loop(move_src_offset1, move_src_offset2, self.num_each_loop, index_id)

        with self.tik_instance.if_scope(num_last_loop > 0):
            move_src_offset1 = src_offset1 + loop_times * self.num_each_loop
            move_src_offset2 = src_offset2 + loop_times * self.num_each_loop
            self.pdist_compute_each_loop(move_src_offset1, move_src_offset2, num_last_loop, index_id)

    # compute on each loop
    def pdist_compute_each_loop(self, src_offset1, src_offset2, move_num, index_id):
        burst_len = _get_ceil_int(move_num, self.data_each_block)

        if self.dtype_x == "float16":
            self.tik_instance.data_move(self.src_temp_fp16_tensor, self.input_x_gm[src_offset1], 0, 1, burst_len, 0, 0)
            # fp16->fp32
            self.pdist_convert(self.src1_ub, self.src_temp_fp16_tensor, move_num, "float16")

            self.tik_instance.data_move(self.src_temp_fp16_tensor, self.input_x_gm[src_offset2], 0, 1, burst_len, 0, 0)
            # fp16->fp32
            self.pdist_convert(self.src2_ub, self.src_temp_fp16_tensor, move_num, "float16")
        else:
            self.tik_instance.data_move(self.src1_ub, self.input_x_gm[src_offset1], 0, 1, burst_len, 0, 0)
            self.tik_instance.data_move(self.src2_ub, self.input_x_gm[src_offset2], 0, 1, burst_len, 0, 0)

        compute_loop = move_num // Constant.MAX_NUM_ONE_REPEAT // 255
        with self.tik_instance.if_scope(compute_loop > 0):
            with self.tik_instance.for_range(0, compute_loop) as index:
                index_offset = index * Constant.MAX_NUM_ONE_REPEAT * 255
                src_offset1 = index_offset
                src_offset2 = index_offset
                count_num = 255 * Constant.MAX_NUM_ONE_REPEAT
                self.pdist_process(Constant.MAX_NUM_ONE_REPEAT, src_offset1, src_offset2, 255, count_num, index_id)

        last_loop = move_num % (Constant.MAX_NUM_ONE_REPEAT * 255) // Constant.MAX_NUM_ONE_REPEAT
        with self.tik_instance.if_scope(last_loop > 0):
            index_offset = compute_loop * Constant.MAX_NUM_ONE_REPEAT * 255
            src_offset1 = index_offset
            src_offset2 = index_offset
            count_num = last_loop * Constant.MAX_NUM_ONE_REPEAT
            self.pdist_process(Constant.MAX_NUM_ONE_REPEAT, src_offset1, src_offset2, last_loop, count_num, index_id)

        compute_mask = move_num % Constant.MAX_NUM_ONE_REPEAT
        with self.tik_instance.if_scope(compute_mask > 0):
            index_offset = move_num // Constant.MAX_NUM_ONE_REPEAT * Constant.MAX_NUM_ONE_REPEAT
            src_offset1 = index_offset
            src_offset2 = index_offset
            count_num = compute_mask
            self.pdist_process(compute_mask, src_offset1, src_offset2, 1, count_num, index_id)

    # the specific calculation process
    # 'pylint: disable = unused-argument,redefined-builtin,too-many-arguments
    def pdist_process(self, mask, src_addr1, src_addr2, repeat_times, count_num, index_id):
        with self.tik_instance.if_scope(self.p == 0.0):
            self.tik_instance.vec_sub(mask, self.src1_ub[src_addr1], self.src1_ub[src_addr1], self.src2_ub[src_addr2],
                                      repeat_times, 8, 8, 8)
            with self.tik_instance.for_range(0, count_num) as k:
                with self.tik_instance.if_scope(self.src1_ub[src_addr1 + k] == 0.0):
                    self.src1_ub[src_addr1 + k].set_as(self.scalar_zero_loop)
                with self.tik_instance.else_scope():
                    self.src1_ub[src_addr1 + k].set_as(self.scalar_one_loop)
            self.tik_instance.vec_reduce_add(mask, self.temp_sum_tensor, self.src1_ub[src_addr1], self.work_tensor,
                                             repeat_times, 8)
        with self.tik_instance.else_scope():
            self.tik_instance.vec_sub(mask, self.src1_ub[src_addr1], self.src1_ub[src_addr1], self.src2_ub[src_addr2],
                                      repeat_times, 8, 8, 8)
            self.tik_instance.vec_abs(mask, self.src1_ub[src_addr1], self.src1_ub[src_addr1], repeat_times, 8, 8)
            self.tik_instance.vec_ln(mask, self.src1_ub[src_addr1], self.src1_ub[src_addr1], repeat_times, 8, 8)
            self.tik_instance.vec_muls(mask, self.src1_ub[src_addr1], self.src1_ub[src_addr1], self.p,
                                       repeat_times, 8, 8)
            self.tik_instance.vec_exp(mask, self.src1_ub[src_addr1], self.src1_ub[src_addr1], repeat_times, 8, 8)
            self.tik_instance.vec_reduce_add(mask, self.temp_sum_tensor, self.src1_ub[src_addr1], self.work_tensor,
                                             repeat_times, 8)
        scalar_sum_each_core = self.tik_instance.Scalar(dtype="float32",
                                                        name="scalar_sum_each_core",
                                                        init_value=self.dst_sum_tensor[index_id])
        self.tik_instance.vec_adds(1, self.temp_sum_tensor, self.temp_sum_tensor, scalar_sum_each_core, 1, 8, 8)
        self.dst_sum_tensor[index_id].set_as(self.temp_sum_tensor[0])

    # while p > 0
    def pdist_sum_process(self, compute_mask, dst_addr, repeat_times):
        self.tik_instance.vec_ln(compute_mask, self.dst_sum_tensor[dst_addr], self.dst_sum_tensor[dst_addr],
                                 repeat_times, 8, 8)
        self.tik_instance.vec_muls(compute_mask, self.dst_sum_tensor[dst_addr], self.dst_sum_tensor[dst_addr],
                                   self.p_reciprocal, repeat_times, 8, 8)
        self.tik_instance.vec_exp(compute_mask, self.dst_sum_tensor[dst_addr], self.dst_sum_tensor[dst_addr],
                                  repeat_times, 8, 8)

    # convert Accuracy
    def pdist_convert(self, dst_tensor, src_tensor, num, src_type):
        if src_type == "float16":
            dst_rep_stride = 8
            src_rep_stride = 4
        else:
            dst_rep_stride = 4
            src_rep_stride = 8

        compute_loop = num // Constant.CONV_NUM_ONE_REPEAT // 255
        with self.tik_instance.if_scope(compute_loop > 0):
            with self.tik_instance.for_range(0, compute_loop) as index:
                dst_offset = index * Constant.CONV_NUM_ONE_REPEAT * 255
                self.tik_instance.vec_conv(Constant.CONV_NUM_ONE_REPEAT, 'none',
                                           dst_tensor[dst_offset], src_tensor[dst_offset],
                                           255, dst_rep_stride, src_rep_stride)
        last_loop = num % (Constant.CONV_NUM_ONE_REPEAT * 255) // Constant.CONV_NUM_ONE_REPEAT
        with self.tik_instance.if_scope(last_loop > 0):
            dst_offset = compute_loop * Constant.CONV_NUM_ONE_REPEAT * 255
            self.tik_instance.vec_conv(Constant.CONV_NUM_ONE_REPEAT, 'none',
                                       dst_tensor[dst_offset], src_tensor[dst_offset],
                                       last_loop, dst_rep_stride, src_rep_stride)
        compute_mask = num % Constant.CONV_NUM_ONE_REPEAT
        with self.tik_instance.if_scope(compute_mask > 0):
            dst_offset = num // Constant.CONV_NUM_ONE_REPEAT * Constant.CONV_NUM_ONE_REPEAT
            self.tik_instance.vec_conv(compute_mask, 'none',
                                       dst_tensor[dst_offset], src_tensor[dst_offset],
                                       1, dst_rep_stride, src_rep_stride)

    def run_tik(self):
        # cal tik_instance according to mode
        self.pdist_compute()
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, 
                                   inputs=[self.input_x_gm], 
                                   outputs=[self.output_y_gm],
                                   flowtable=[self.tiling_gm])
        return self.tik_instance

    # init tensor and scalar on ub
    # 'pylint: disable=W0201
    def init_ub_tensor_and_scalar(self):
        self.src1_ub = self.tik_instance.Tensor("float32", (self.ub_tensor_each_loop, ),
                                                name="src1_ub",
                                                scope=tik.scope_ubuf)
        self.src2_ub = self.tik_instance.Tensor("float32", (self.ub_tensor_each_loop, ),
                                                name="src2_ub",
                                                scope=tik.scope_ubuf)
        self.work_tensor = self.tik_instance.Tensor("float32", (256, ), name="work_tensor", scope=tik.scope_ubuf)
        self.temp_sum_tensor = self.tik_instance.Tensor("float32", (Constant.NUM_FP32_EACH_BLOCK, ),
                                                        name="temp_sum_tensor",
                                                        scope=tik.scope_ubuf)
        self.dst_sum_tensor = self.tik_instance.Tensor("float32", (self.data_each_block, ),
                                                       name="dst_sum_tensor",
                                                       scope=tik.scope_ubuf)

        self.scalar_zero_loop = self.tik_instance.Scalar(dtype="float32", name="scalar_zero_loop", init_value=0.0)
        self.scalar_one_loop = self.tik_instance.Scalar(dtype="float32", name="scalar_one_loop", init_value=1.0)
        self.squared_sqrt = self.tik_instance.Scalar(dtype="float32", name="squared_sqrt")
        self.i_int = self.tik_instance.Scalar(dtype="int32", name="i_int")
        self.j_int = self.tik_instance.Scalar(dtype="int32", name="j_int")
        self.rows_float = self.tik_instance.Scalar(dtype="float32", name="rows_float", init_value=self.rows)
        self.out_index_float = self.tik_instance.Scalar(dtype="float32", name="out_index_float")
        self.p_reciprocal = self.tik_instance.Scalar("float32", name="p_reciprocal", init_value=1.0 / self.p)

        if self.dtype_y == "float16":
            self.dst_sum_fp16_tensor = self.tik_instance.Tensor("float16", (self.data_each_block, ),
                                                                name="dst_sum_fp16_tensor",
                                                                scope=tik.scope_ubuf)
            self.src_temp_fp16_tensor = self.tik_instance.Tensor("float16", (self.ub_tensor_each_loop, ),
                                                                 name="src_temp_fp16_tensor",
                                                                 scope=tik.scope_ubuf)

    # check the param of input
    def check_param(self):
        para_check.check_kernel_name(self.kernel_name)
        check_tuple = ("float16", "float32")
        para_check.check_dtype_rule(self.dtype_x, check_tuple)


# 'pylint: disable=invalid-name,too-many-locals,too-many-arguments,unused-argument
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def pdist(input_x, output_y, p, kernel_name="pdist"):
    """
    calculating data

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, if shape of input_x is should be
    kernel_name : str
        kernel name, default value is "pdist"

    Returns
    -------
    None
    """
    res = Pdist(input_x, output_y, p, kernel_name)
    return res.run_tik()
