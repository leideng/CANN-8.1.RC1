#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2023. All rights reserved.
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
chamfer_distance
"""
from math import ceil
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator


class Constant:
    """
    The class for constant
    """
    MAX_INT64 = 2**64 - 1
    TILING_ARG_NUM = 16
    BLOCK_BYTE_SIZE = 32
    NUM_256 = 256
    INT_32_size = 4
    NUM_16 = 16
    FP16_MAX = 0x7C00
    FP32_MAX = 0x7F800000


class ChamferDistance():
    """
    Function: use to finish ChamferDistance main functions
    """

    def __init__(self, xyz1, xyz2, dist1, dist2, idx1, idx2, kernel_name):
        self.tik_instance = tik.Tik()
        self.xyz_dtype = xyz1.get("dtype").lower()
        self.dist_dtype = dist1.get("dtype").lower()
        self.idx_dtype = idx1.get("dtype").lower()
        para_check.check_dtype(self.xyz_dtype, ("float16", "float32"), param_name="xyz")
        para_check.check_dtype(self.dist_dtype, ("float16", "float32"), param_name="dist")
        para_check.check_dtype(self.idx_dtype, ("int32"), param_name="idx")

        self.kernel_name = kernel_name

        self.dtype_size = tbe_platform.get_bit_len(self.xyz_dtype) // 8
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.data_len_one_block = Constant.BLOCK_BYTE_SIZE // self.dtype_size
        self.int32_len_one_block = Constant.BLOCK_BYTE_SIZE // Constant.INT_32_size
        # reserved ub size 20KB
        reserved_ub_size = 20 * 1024
        self.ub_availble = self.ub - reserved_ub_size

        self.actual_core_used = self.tik_instance.Scalar("int32", name="actual_core_used")
        self.batch = self.tik_instance.Scalar("int32", name="batch")
        self.n = self.tik_instance.Scalar("int32", name="n")
        self.task_num_each_core = self.tik_instance.Scalar("int32", name="task_num_each_core")
        self.task_num_last_core = self.tik_instance.Scalar("int32", name="task_num_last_core")
        self.tensor_size = self.tik_instance.Scalar("int32", name="tensor_size")
        self.tiling_gm = None
        self.xyz1_gm = None
        self.xyz2_gm = None
        self.dist1_gm = None
        self.dist2_gm = None
        self.idx1_gm = None
        self.idx2_gm = None
        self.src1_x_ub = None
        self.src1_y_ub = None
        self.dist_ub = None
        self.idx_ub = None
        self.src2_x_ub = None
        self.src2_y_ub = None
        self.work_tensor_ub = None
        
    def _tiling_args(self):
        """
        Get runtime tiling parameters from tiling
        """
        self.tiling_gm = self.tik_instance.Tensor("int32", (Constant.TILING_ARG_NUM,), 
                                                  name="tiling_gm", scope=tik.scope_gm)
        with self.tik_instance.new_stmt_scope():
            tiling_ub = self.tik_instance.Tensor("int32", (Constant.TILING_ARG_NUM,), 
                                                 name="tiling_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, 2, 0, 0)
            #   tiling int32 scalar
            self.actual_core_used.set_as(tiling_ub[0])
            self.batch.set_as(tiling_ub[2])
            self.n.set_as(tiling_ub[3])
            self.task_num_each_core.set_as(tiling_ub[4])
            self.task_num_last_core.set_as(tiling_ub[5])
            self.tensor_size.set_as(tiling_ub[6])

    def init_ub_tensor(self):
        self.src1_x_ub = self.tik_instance.Tensor(self.xyz_dtype, (Constant.NUM_256,), 
                                                  name="src1_x_ub", scope=tik.scope_ubuf)
        self.src1_y_ub = self.tik_instance.Tensor(self.xyz_dtype, (Constant.NUM_256,), 
                                                  name="src1_y_ub", scope=tik.scope_ubuf)
        self.dist_ub = self.tik_instance.Tensor(self.dist_dtype, (Constant.NUM_256,), 
                                                name="dist_ub", scope=tik.scope_ubuf)
        self.idx_ub = self.tik_instance.Tensor("int32", (Constant.NUM_256,), 
                                               name="idx_ub", scope=tik.scope_ubuf,)
        self.src2_x_ub = self.tik_instance.Tensor(self.xyz_dtype, (self.tensor_size,), 
                                                  name="src2_x_ub", scope=tik.scope_ubuf)
        self.src2_y_ub = self.tik_instance.Tensor(self.xyz_dtype, (self.tensor_size,), 
                                                  name="src2_y_ub", scope=tik.scope_ubuf)
        #create the work_tensor_ub
        mask = 256 // self.dtype_size
        it1_output_count = 1024 * 2
        it2_align_start = self._get_ceil_int(it1_output_count, self.data_len_one_block) * self.data_len_one_block
        it2_output_count = self._get_ceil_int(it1_output_count, mask) * 2
        it3_align_start = self._get_ceil_int(it2_output_count, self.data_len_one_block) * self.data_len_one_block 
        it3_output_count = self._get_ceil_int(it2_output_count, mask) * 2
        it4_align_start = self._get_ceil_int(it3_output_count, self.data_len_one_block) * self.data_len_one_block
        it4_output_count = self._get_ceil_int(it3_output_count, mask) * 2
        final_work_tensor_need_size = it2_align_start + it3_align_start + it4_align_start + it4_output_count
        self.work_tensor_ub = self.tik_instance.Tensor(self.xyz_dtype, (final_work_tensor_need_size,), 
                                                       tik.scope_ubuf, 'work_tensor_ub')

    def init_gm_tensor(self):
        """
        init_gm_tensor
        """
        self.xyz1_gm = self.tik_instance.Tensor(
            self.xyz_dtype, (Constant.MAX_INT64,), name="xyz1_gm", scope=tik.scope_gm)
        self.xyz2_gm = self.tik_instance.Tensor(
            self.xyz_dtype, (Constant.MAX_INT64,), name="xyz2_gm", scope=tik.scope_gm)
        self.dist1_gm = self.tik_instance.Tensor(
            self.dist_dtype, (Constant.MAX_INT64,), name="dist1_gm", scope=tik.scope_gm, is_atomic_add=True)
        self.dist2_gm = self.tik_instance.Tensor(
            self.dist_dtype, (Constant.MAX_INT64,), name="dist2_gm", scope=tik.scope_gm, is_atomic_add=True)
        self.idx1_gm = self.tik_instance.Tensor(
            self.idx_dtype, (Constant.MAX_INT64,), name="idx1_gm", scope=tik.scope_gm, is_atomic_add=True)
        self.idx2_gm = self.tik_instance.Tensor(
            self.idx_dtype, (Constant.MAX_INT64,), name="idx2_gm", scope=tik.scope_gm, is_atomic_add=True)
        
    def chamfer_distance_compute(self):
        """
        ChamferDistance compute
        """
        self._tiling_args()
        self.init_gm_tensor()
        with self.tik_instance.for_range(0, self.actual_core_used, block_num=self.actual_core_used) as core_index:
            self.init_ub_tensor()
            with self.tik_instance.if_scope(core_index < (self.actual_core_used - 1)):
                self.compute_each_core(core_index, self.task_num_each_core)
            with self.tik_instance.else_scope():
                self.compute_each_core(core_index, self.task_num_last_core)
        self._add_compile_info()
        opt_config = {"out_of_bound_sync_check": True}
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                inputs=[self.xyz1_gm, self.xyz2_gm],
                                flowtable=[self.tiling_gm],
                                outputs=[self.dist1_gm, self.dist2_gm, self.idx1_gm, self.idx2_gm],
                                config=opt_config)
        return self.tik_instance

    def _add_compile_info(self):
        """
        Add compile info
        """
        tbe_context.get_context().add_compile_info("vars", {
            "_ub_size": self.ub_availble,
            "_core_num": self.core_num,
            "_block_size": self.data_len_one_block
        })

    def compute_each_core(self, core_index, task_num):
        # judge it have the tail
        self.calculate_chamfer_distance(self.xyz1_gm, self.xyz2_gm, core_index,
                                        task_num, self.dist1_gm, self.idx1_gm)
        self.calculate_chamfer_distance(self.xyz2_gm, self.xyz1_gm, core_index,
                                        task_num, self.dist2_gm, self.idx2_gm)
            
    def calculate_chamfer_distance(self, src1, src2, core_index, 
                                   task_num, dist, idx):
        current_core_offset = core_index * self.task_num_each_core
        repeat_times = self._get_ceil_int(task_num, Constant.NUM_256)
        src1_tail = task_num - (repeat_times - 1) * Constant.NUM_256

        burst_len_src1 = Constant.NUM_256 // self.data_len_one_block
        burst_len_src1_tail = src1_tail // self.data_len_one_block
        src1_tail_len = burst_len_src1_tail * self.data_len_one_block
        src1_tail_offset = src1_tail % self.data_len_one_block
        with self.tik_instance.for_range(0, repeat_times - 1) as repeat_idx:
            current_src1_offset_x = current_core_offset + repeat_idx * Constant.NUM_256
            current_src1_offset_y = current_core_offset + repeat_idx * Constant.NUM_256 + self.n * self.batch
            # move to ub
            self.tik_instance.data_move(self.src1_x_ub, src1[current_src1_offset_x], 0, 1, burst_len_src1, 0, 0)
            self.tik_instance.data_move(self.src1_y_ub, src1[current_src1_offset_y], 0, 1, burst_len_src1, 0, 0)
            # to calculate 
            self.calculate_chamfer(current_src1_offset_x, current_src1_offset_y, Constant.NUM_256, src2)
            dst_id = current_core_offset + repeat_idx * Constant.NUM_256
            # move to gm
            self.tik_instance.data_move(dist[dst_id], self.dist_ub, 0, 1, burst_len_src1, 0, 0)
            self.tik_instance.data_move(idx[dst_id], self.idx_ub, 0, 1, 
                                        Constant.NUM_256 // self.int32_len_one_block, 0, 0)
        # TO SOLVE THE TAIL
        dst_id_offset = current_core_offset + (repeat_times - 1) * Constant.NUM_256
        current_src1_offset_x = current_core_offset + (repeat_times - 1) * Constant.NUM_256
        current_src1_offset_y = current_core_offset + (repeat_times - 1) * Constant.NUM_256 + self.n * self.batch
        with self.tik_instance.if_scope(burst_len_src1_tail >= 1):
            # move to ub
            self.tik_instance.data_move(self.src1_x_ub, src1[current_src1_offset_x], 0, 1, burst_len_src1_tail, 0, 0)
            self.tik_instance.data_move(self.src1_y_ub, src1[current_src1_offset_y], 0, 1, burst_len_src1_tail, 0, 0)
            # to calculate
            self.calculate_chamfer(current_src1_offset_x, current_src1_offset_y, src1_tail_len, src2)
            # move to gm
            self.tik_instance.data_move(dist[dst_id_offset], self.dist_ub, 0, 1, burst_len_src1_tail, 0, 0)
            self.tik_instance.data_move(idx[dst_id_offset], self.idx_ub, 0, 1, 
                                        src1_tail_len // self.int32_len_one_block, 0, 0)
        # TO SOLVE THE TAIL that smaller than one block(sometimes maybe not to do)
        current_dst_id_offset = current_core_offset + (repeat_times - 1) * Constant.NUM_256 + src1_tail_len
        current_src1_tail_offset_x = current_core_offset + (repeat_times - 1) * Constant.NUM_256 + src1_tail_len
        current_src1_tail_offset_y = (current_core_offset + (repeat_times - 1) * Constant.NUM_256 
                                     + src1_tail_len + self.n * self.batch)
        with self.tik_instance.if_scope(src1_tail_offset >= 1):
            # move to ub
            self.tik_instance.data_move(self.src1_x_ub, src1[current_src1_tail_offset_x], 0, 1, 1, 0, 0)
            self.tik_instance.data_move(self.src1_y_ub, src1[current_src1_tail_offset_y], 0, 1, 1, 0, 0)
            # to calculate
            self.calculate_chamfer(current_src1_tail_offset_x, current_src1_tail_offset_y, src1_tail_offset, src2)
            # move to gm
            self.tik_instance.data_move(dist[current_dst_id_offset], self.dist_ub, 0, 1, 1, 0, 0)
            self.tik_instance.data_move(idx[current_dst_id_offset], self.idx_ub, 0, 1, 
                                        (src1_tail_offset - 1) // 8 + 1, 0, 0)

    def calculate_chamfer(self, current_src_offset_x, current_src_offset_y, src1_len, src):
        src1_x = self.tik_instance.Scalar(self.xyz_dtype, name="src1_x")
        src1_y = self.tik_instance.Scalar(self.xyz_dtype, name="src1_y")
        result_final_dst = self.tik_instance.Scalar(self.xyz_dtype, name="result_final_dst")
        result_final_dst_fp32 = self.tik_instance.Scalar("float32", name="result_final_dst_fp32")
        result_final_idx = self.tik_instance.Scalar("int32", name="result_final_idx")
        mask = 256 // self.dtype_size
        src2_repeat_times = self.n // mask
        loop = src2_repeat_times // 128
        src2_repeats_tail = src2_repeat_times % 128
        src2_tail = self.n % mask
        tail_brust = self._get_ceil_int(src2_tail, self.data_len_one_block)
        #calculate the min dist
        with self.tik_instance.for_range(0, src1_len) as src1_idx:
            batch_id = (current_src_offset_x + src1_idx) // self.n
            src2_offset_x = batch_id * self.n
            src2_offset_y = batch_id * self.n + self.n * self.batch
            src1_x.set_as(self.src1_x_ub[src1_idx])
            src1_y.set_as(self.src1_y_ub[src1_idx])
            if self.xyz_dtype == "float16":
                result_final_dst.set_as(Constant.FP16_MAX)
                result_final_dst_fp32.set_as(Constant.FP32_MAX)
            else:
                result_final_dst.set_as(Constant.FP32_MAX)
                result_final_dst_fp32.set_as(Constant.FP32_MAX)
            with self.tik_instance.if_scope(loop >= 1):
                with self.tik_instance.for_range(0, loop) as loop_idx:
                    current_src2_offset_x = src2_offset_x + loop_idx * mask * 128
                    current_src2_offset_y = src2_offset_y + loop_idx * mask * 128
                    self._reduce_min_(src, src1_x, src1_y, current_src2_offset_x, current_src2_offset_y,
                                    1024, mask, 128, 8, loop_idx * mask * 128,
                                    result_final_dst, result_final_dst_fp32, result_final_idx)
            with self.tik_instance.if_scope(src2_repeats_tail >= 1):
                current_src2_offset_x = src2_offset_x + loop * mask * 128
                current_src2_offset_y = src2_offset_y + loop * mask * 128
                self._reduce_min_(src, src1_x, src1_y, current_src2_offset_x, current_src2_offset_y,
                                src2_repeats_tail * 8, mask, src2_repeats_tail, 8, loop * mask * 128,
                                result_final_dst, result_final_dst_fp32, result_final_idx)
            with self.tik_instance.if_scope(src2_tail >= 1):
                current_src2_offset_x = src2_offset_x + loop * 128 * mask + src2_repeats_tail * mask
                current_src2_offset_y = src2_offset_y + loop * 128 * mask + src2_repeats_tail * mask
                self._reduce_min_(src, src1_x, src1_y, current_src2_offset_x, current_src2_offset_y,
                                tail_brust, src2_tail, tail_brust, 0, loop * 128 * mask + src2_repeats_tail * mask,
                                result_final_dst, result_final_dst_fp32, result_final_idx)
            self.dist_ub[src1_idx].set_as(result_final_dst)
            self.idx_ub[src1_idx].set_as(result_final_idx)

    def _reduce_min_(self, src, src1_x, src1_y, current_src2_offset_x, current_src2_offset_y, 
                    brust_times, mask, reduce_min_brust, reduce_stride, src2_offset, result_final_dst,
                    result_final_dst_fp32, result_final_idx):
        result = self.tik_instance.Tensor(self.xyz_dtype, (16,), name="result", scope=tik.scope_ubuf)
        result_dst = self.tik_instance.Scalar(self.xyz_dtype, name="result_dst")
        result_dst_fp32 = self.tik_instance.Scalar("float32", name="result_dst_fp32")
        result_idx_int32 = self.tik_instance.Scalar("int32", name="result_idx_int32")
        if self.xyz_dtype == "float16":
            result_idx = self.tik_instance.Scalar("uint16", name="result_idx")
        else:
            result_idx = self.tik_instance.Scalar("uint32", name="result_idx")
        self.tik_instance.data_move(self.src2_x_ub, src[current_src2_offset_x], 0, 1, brust_times, 0, 0)
        self.tik_instance.data_move(self.src2_y_ub, src[current_src2_offset_y], 0, 1, brust_times, 0, 0)
        self.tik_instance.h_sub(self.src2_x_ub, src1_x, self.src2_x_ub)
        self.tik_instance.h_sub(self.src2_y_ub, src1_y, self.src2_y_ub)
        self.tik_instance.h_mul(self.src2_x_ub, self.src2_x_ub, self.src2_x_ub)
        self.tik_instance.h_mul(self.src2_y_ub, self.src2_y_ub, self.src2_y_ub)
        self.tik_instance.h_add(self.src2_x_ub, self.src2_x_ub, self.src2_y_ub)
        self.tik_instance.vec_reduce_min(mask, result, self.src2_x_ub, self.work_tensor_ub, 
                                         reduce_min_brust, reduce_stride, cal_index=True)

        result_dst.set_as(result[0])
        result_dst_fp32.set_as(result_dst)
        with self.tik_instance.if_scope(result_dst_fp32 < result_final_dst_fp32):
            result_final_dst_fp32.set_as(result_dst_fp32)
            result_final_dst.set_as(result[0])
            if self.xyz_dtype == "float16":
                result = result.reinterpret_cast_to("uint16")
                result_idx.set_as(result[1])
                result_idx_int32.set_as(result_idx)
                result_idx_int32 = result_idx_int32
                result_idx_int32 = result_idx_int32 + src2_offset
                result_final_idx.set_as(result_idx_int32)
            else:
                result = result.reinterpret_cast_to("uint32")
                result_idx.set_as(result[1])
                result_idx_int32.set_as(result_idx)
                result_idx_int32 = result_idx_int32
                result_idx_int32 = result_idx_int32 + src2_offset
                result_final_idx.set_as(result_idx_int32)
        
    def _get_ceil_int(self, int1, int2):
        """
        Get ceil
        """
        result = self.tik_instance.Scalar("int32", name="result")
        with self.tik_instance.if_scope(int1 % int2 == 0):
            result.set_as(int1 // int2)
        with self.tik_instance.else_scope():
            result.set_as(int1 // int2 + 1)
        return result

    

@register_operator("ChamferDistance")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def chamfer_distance(xyz1, xyz2, dist1, dist2, idx1, idx2, kernel_name="chamfer_distance"):
    """

    """
    obj = ChamferDistance(xyz1, xyz2, dist1, dist2, idx1, idx2, kernel_name)
    return obj.chamfer_distance_compute()





