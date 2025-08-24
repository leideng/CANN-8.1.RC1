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

import math
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform

BLOCK_BITE_SIZE = 32
WORK_TENSOR_SIZE = 256
TOTAL_SCALAR_SIZE = 8
TEMP_SUM_TENSOR_BYTES = 32
CONV_NUM_ONE_REPEAT = 64


class Pdist():
    def __init__(self, input_x, output_y, p, kernel_name="pdist"):
        self.shape_x = input_x.get("shape")
        self.dtype_x = input_x.get("dtype").lower()
        self.shape_y = output_y.get("shape")
        self.dtype_y = output_y.get("dtype").lower()
        self.p = p
        self.kernel_name = kernel_name

        # check param
        self.check_param()

        self.rows = self.shape_x[0]
        self.cols = self.shape_x[1]
        self.compute_num = int(self.rows * (self.rows - 1) / 2)
        self.num_each_core = self.cols

        self.tik_instance = tik.Tik()
        self.ai_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

        # get size of UB and compute the number of elements can be stored in a block
        self.ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.dtype_bytes_size = tbe_platform.get_bit_len(self.dtype_x) // 8
        self.data_each_block = BLOCK_BITE_SIZE // self.dtype_bytes_size

        self.fp16_bytes = 2
        self.fp32_bytes = 4
        self.num_fp32_each_block = 8
        self.max_num_one_repeat = 64

        # compute the tensor size except src1_ub and src2_ub and src_temp_fp16_tensor
        output_tensor_size = self.data_each_block

        others_tensor_size_bytes = TEMP_SUM_TENSOR_BYTES + \
                                   (WORK_TENSOR_SIZE + output_tensor_size + TOTAL_SCALAR_SIZE) * self.fp32_bytes

        if self.dtype_x == "float16":
            dst_sum_fp16_tensor_size = self.data_each_block
            others_tensor_size_bytes = others_tensor_size_bytes + dst_sum_fp16_tensor_size * self.fp16_bytes

        if others_tensor_size_bytes > self.ub_size_bytes:
            raise RuntimeError("Out of Unified buffer")

        self.ub_size_bytes = self.ub_size_bytes - others_tensor_size_bytes
        num_each_block = math.ceil(self.num_each_core / self.num_fp32_each_block) * self.num_fp32_each_block

        if self.dtype_x == "float16":
            each_core_size_bytes = num_each_block * (2 * self.fp32_bytes + self.fp16_bytes)
        else:
            each_core_size_bytes = num_each_block * 2 * self.fp32_bytes

        if each_core_size_bytes < self.ub_size_bytes:
            self.num_each_loop = self.num_each_core
            self.ub_tensor_each_loop = num_each_block
        else:
            # compute the number of elements of each row can be stored on UB and keep 32-bit alignment
            if self.dtype_x == "float16":
                self.num_each_loop = self.ub_size_bytes // (
                    2 * self.fp32_bytes + self.fp16_bytes) // self.num_fp32_each_block * self.num_fp32_each_block
            else:
                self.num_each_loop = self.ub_size_bytes // 2 // self.fp32_bytes // self.num_fp32_each_block * \
                                     self.num_fp32_each_block
            self.ub_tensor_each_loop = self.num_each_loop
        # request gm
        self.input_x_gm = self.tik_instance.Tensor(self.dtype_x, self.shape_x, name="input_x_gm", scope=tik.scope_gm)
        self.output_y_gm = self.tik_instance.Tensor(self.dtype_y, self.shape_y, name="output_y_gm", scope=tik.scope_gm)

    @register_operator_compute("pdist", op_mode="static", support_fusion=True)
    def pdist_compute(self):
        # blocks processed by each core
        num_block_each_core = self.compute_num // self.ai_core_num // self.data_each_block
        # last nums processed by each core
        last_nums = self.compute_num % (self.data_each_block * self.ai_core_num)
        # last blocks processed by each_core
        last_nums_blocks = last_nums // self.data_each_block
        # last nums processed by none full block
        last_nums_none_full_block = last_nums % self.data_each_block

        if num_block_each_core > 0:
            with self.tik_instance.for_range(0, self.ai_core_num, block_num=self.ai_core_num) as core_id:
                self.init_ub_tensor_and_scalar()
                with self.tik_instance.for_range(0, num_block_each_core) as block_num_id:
                    output_y_gm_index = core_id * self.data_each_block * num_block_each_core + \
                                        block_num_id * self.data_each_block
                    self.out_index_float.set_as(output_y_gm_index)
                    with self.tik_instance.for_range(0, self.data_each_block) as k:    
                        self.get_i_j_from_index(output_y_gm_index, k)
                        self.pdist_compute_each_core(self.i_int * self.num_each_core,
                                                     self.j_int * self.num_each_core, k)
                    if self.p > 0:
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

        if last_nums > 0:
            with self.tik_instance.for_range(0, last_nums_blocks) as last_block_id:
                self.init_ub_tensor_and_scalar()
                output_y_gm_index = self.ai_core_num * num_block_each_core * self.data_each_block + \
                                    last_block_id * self.data_each_block
                self.out_index_float.set_as(output_y_gm_index)
                with self.tik_instance.for_range(0, self.data_each_block) as k:
                    self.get_i_j_from_index(output_y_gm_index, k)
                    self.pdist_compute_each_core(self.i_int * self.num_each_core,
                                                 self.j_int * self.num_each_core, k)
                if self.p > 0:
                    self.pdist_sum_process(self.data_each_block, 0, 1)
                # to improve the accuracy, convert fp16 to fp32, convert it back to fp16 after calculation
                if self.dtype_y == "float16":
                    self.pdist_convert(self.dst_sum_fp16_tensor, self.dst_sum_tensor, self.data_each_block, "float32")
                    self.tik_instance.data_move(self.output_y_gm[output_y_gm_index], self.dst_sum_fp16_tensor, 
                                                0, 1, 1, 0, 0)
                else:
                    self.tik_instance.data_move(self.output_y_gm[output_y_gm_index], self.dst_sum_tensor, 
                                                0, 1, 1, 0, 0)
        
        if last_nums_none_full_block > 0:
            with self.tik_instance.new_stmt_scope():
                self.init_ub_tensor_and_scalar()
                with self.tik_instance.for_range(0, last_nums_none_full_block) as k:
                    output_y_gm_index = self.ai_core_num * num_block_each_core * self.data_each_block + \
                                        last_nums_blocks * self.data_each_block
                    self.out_index_float.set_as(output_y_gm_index)
                    self.get_i_j_from_index(output_y_gm_index, k)
                    self.pdist_compute_each_core(self.i_int * self.num_each_core,
                                                 self.j_int * self.num_each_core, k)
                if self.p > 0:
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

    def get_i_j_from_index(self, index_id, k):
        """
        infers what i and j are from the value of index_id
        we iterate over tuples of (i, j, index_id + k)
        where i is the first vector from the input, j is the second, and index_id + k is the result index
        """
        i_minus_half = self.rows - .5
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
        if loop_times < 2:
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

        if num_last_loop > 0:
            move_src_offset1 = src_offset1 + loop_times * self.num_each_loop
            move_src_offset2 = src_offset2 + loop_times * self.num_each_loop
            self.pdist_compute_each_loop(move_src_offset1, move_src_offset2, num_last_loop, index_id)

    # compute on each loop
    def pdist_compute_each_loop(self, src_offset1, src_offset2, move_num, index_id):
        burst_len = math.ceil(move_num / self.data_each_block)

        if self.dtype_x == "float16":
            self.tik_instance.data_move(self.src_temp_fp16_tensor, self.input_x_gm[src_offset1], 0, 1, burst_len, 0, 0)
            #fp16->fp32
            self.pdist_convert(self.src1_ub, self.src_temp_fp16_tensor, move_num, "float16")

            self.tik_instance.data_move(self.src_temp_fp16_tensor, self.input_x_gm[src_offset2], 0, 1, burst_len, 0, 0)
            #fp16->fp32
            self.pdist_convert(self.src2_ub, self.src_temp_fp16_tensor, move_num, "float16")
        else:
            self.tik_instance.data_move(self.src1_ub, self.input_x_gm[src_offset1], 0, 1, burst_len, 0, 0)
            self.tik_instance.data_move(self.src2_ub, self.input_x_gm[src_offset2], 0, 1, burst_len, 0, 0)

        compute_loop = move_num // self.max_num_one_repeat // 255
        if compute_loop > 0:
            with self.tik_instance.for_range(0, compute_loop) as index:
                index_offset = index * self.max_num_one_repeat * 255
                src_offset1 = index_offset
                src_offset2 = index_offset
                count_num = 255 * self.max_num_one_repeat
                self.pdist_process(self.max_num_one_repeat, src_offset1, src_offset2, 255, count_num, index_id)

        last_loop = move_num % (self.max_num_one_repeat * 255) // self.max_num_one_repeat
        if last_loop > 0:
            index_offset = compute_loop * self.max_num_one_repeat * 255
            src_offset1 = index_offset
            src_offset2 = index_offset
            count_num = last_loop * self.max_num_one_repeat
            self.pdist_process(self.max_num_one_repeat, src_offset1, src_offset2, last_loop, count_num, index_id)

        compute_mask = move_num % self.max_num_one_repeat
        if compute_mask > 0:
            index_offset = move_num // self.max_num_one_repeat * self.max_num_one_repeat
            src_offset1 = index_offset
            src_offset2 = index_offset
            count_num = compute_mask
            self.pdist_process(compute_mask, src_offset1, src_offset2, 1, count_num, index_id)

    # the specific calculation process
    # 'pylint: disable = unused-argument,redefined-builtin,too-many-arguments
    def pdist_process(self, mask, src_addr1, src_addr2, repeat_times, count_num, index_id):
        if self.p == 0.0:
            self.tik_instance.vec_sub(mask, self.src1_ub[src_addr1], self.src1_ub[src_addr1], self.src2_ub[src_addr2],
                                      repeat_times, 8, 8, 8)
            with self.tik_instance.for_range(0, count_num) as k:
                with self.tik_instance.if_scope(self.src1_ub[src_addr1 + k] == 0.0):
                    self.src1_ub[src_addr1 + k].set_as(self.scalar_zero_loop)
                with self.tik_instance.else_scope():
                    self.src1_ub[src_addr1 + k].set_as(self.scalar_one_loop)
            self.tik_instance.vec_reduce_add(mask, self.temp_sum_tensor, self.src1_ub[src_addr1], self.work_tensor,
                                             repeat_times, 8)
        else:
            self.tik_instance.vec_sub(mask, self.src1_ub[src_addr1], self.src1_ub[src_addr1], self.src2_ub[src_addr2],
                                      repeat_times, 8, 8, 8)
            self.tik_instance.vec_abs(mask, self.src1_ub[src_addr1], self.src1_ub[src_addr1], repeat_times, 8, 8)
            self.tik_instance.vec_ln(mask, self.src1_ub[src_addr1], self.src1_ub[src_addr1], repeat_times, 8, 8)
            self.tik_instance.vec_muls(mask, self.src1_ub[src_addr1], self.src1_ub[src_addr1], self.scalar_p,
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
                                   self.scalar_p_reciprocal, repeat_times, 8, 8)
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

        compute_loop = num // CONV_NUM_ONE_REPEAT // 255
        if compute_loop > 0:
            with self.tik_instance.for_range(0, compute_loop) as index:
                dst_offset = index * CONV_NUM_ONE_REPEAT * 255
                self.tik_instance.vec_conv(CONV_NUM_ONE_REPEAT, 'none', dst_tensor[dst_offset], src_tensor[dst_offset],
                                           255, dst_rep_stride, src_rep_stride)
        last_loop = num % (CONV_NUM_ONE_REPEAT * 255) // CONV_NUM_ONE_REPEAT
        if last_loop > 0:
            dst_offset = compute_loop * CONV_NUM_ONE_REPEAT * 255
            self.tik_instance.vec_conv(CONV_NUM_ONE_REPEAT, 'none', dst_tensor[dst_offset], src_tensor[dst_offset],
                                       last_loop, dst_rep_stride, src_rep_stride)
        compute_mask = num % CONV_NUM_ONE_REPEAT
        if compute_mask > 0:
            dst_offset = num // CONV_NUM_ONE_REPEAT * CONV_NUM_ONE_REPEAT
            self.tik_instance.vec_conv(compute_mask, 'none', dst_tensor[dst_offset], src_tensor[dst_offset], 1,
                                       dst_rep_stride, src_rep_stride)

    def run_tik(self):
        # cal tik_instance according to mode
        self.pdist_compute()
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[self.input_x_gm], outputs=[self.output_y_gm])
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
        self.temp_sum_tensor = self.tik_instance.Tensor("float32", (self.num_fp32_each_block, ),
                                                        name="temp_sum_tensor",
                                                        scope=tik.scope_ubuf)
        self.dst_sum_tensor = self.tik_instance.Tensor("float32", (self.data_each_block, ),
                                                       name="dst_sum_tensor",
                                                       scope=tik.scope_ubuf)

        self.scalar_zero_loop = self.tik_instance.Scalar(dtype="float32", name="scalar_zero_loop", init_value=0.0)
        self.squared_sqrt = self.tik_instance.Scalar(dtype="float32", name="squared_sqrt")
        self.i_int = self.tik_instance.Scalar(dtype="int32", name="i_int")
        self.j_int = self.tik_instance.Scalar(dtype="int32", name="j_int")
        self.out_index_float = self.tik_instance.Scalar(dtype="float32", name="out_index_float")

        if self.p == 0:
            self.scalar_one_loop = self.tik_instance.Scalar(dtype="float32", name="scalar_one_loop", init_value=1.0)
        else:
            self.scalar_p = self.tik_instance.Scalar("float32", name="scalar_p", init_value=self.p)
            self.scalar_p_reciprocal = self.tik_instance.Scalar("float32",
                                                                name="scalar_p_reciprocal",
                                                                init_value=1.0 / self.p)

        if self.dtype_y == "float16":
            self.dst_sum_fp16_tensor = self.tik_instance.Tensor("float16", (self.data_each_block, ),
                                                                name="dst_sum_fp16_tensor",
                                                                scope=tik.scope_ubuf)
            self.src_temp_fp16_tensor = self.tik_instance.Tensor("float16", (self.ub_tensor_each_loop, ),
                                                                 name="src_temp_fp16_tensor",
                                                                 scope=tik.scope_ubuf)

    # check the param of input
    def check_param(self):
        para_check.check_shape_rule(self.shape_x)
        para_check.check_shape(self.shape_x)
        para_check.check_kernel_name(self.kernel_name)
        check_tuple = ("float16", "float32")
        para_check.check_dtype_rule(self.dtype_x, check_tuple)
        shape_x_len = len(self.shape_x)
        if self.shape_x[0] < 2:
            raise RuntimeError("rows of input tensor should > 1 while the rows is {0}".format(self.shape_x[0]))
        if shape_x_len != 2:
            raise RuntimeError("dim of input tensor only support 2D while the dim is {0} D".format(shape_x_len))
        if self.p < 0:
            raise RuntimeError("only support p >= 0 while p is {0}".format(self.p))


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
