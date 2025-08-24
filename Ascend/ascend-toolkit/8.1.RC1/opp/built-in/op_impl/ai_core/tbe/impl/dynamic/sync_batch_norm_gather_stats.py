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
sync_batch_norm_gather_stats
"""

from impl.util.platform_adapter import tik
import tbe.common.platform as tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import para_check
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json

MAX_INT32 = 2 ** 31 - 1
TILING_NUM = 32
MASK = 64


# 'pylint: disable=unused-argument,too-many-locals
def op_select_format(total_sum, total_square_sum, sample_count, running_mean, running_var, batch_mean, batch_invstd,
                     running_mean_update, running_var_update, momentum=0.1, eps=1e-5,
                     kernel_name="sync_batch_norm_gather_stats"):
    """
    op_select format func for dynamic format
    """

    dtype_list = ["float32", "float16"]
    format_list = ["ND", "ND"]
    count_list = ["int32", "int32"]

    input0 = gen_param(classify="input0", name="total_sum", datatype=",".join(dtype_list), 
                       format=",".join(format_list), unknownshape_format=",".join(format_list))
    input1 = gen_param(classify="input1", name="total_square_sum", datatype=",".join(dtype_list),
                       format=",".join(format_list), unknownshape_format=",".join(format_list))
    input2 = gen_param(classify="input2", name="sample_count", datatype=",".join(count_list),
                       format=",".join(format_list), unknownshape_format=",".join(format_list))
    input3 = gen_param(classify="input3", name="mean", datatype=",".join(dtype_list),
                       format=",".join(format_list), unknownshape_format=",".join(format_list))
    input4 = gen_param(classify="input4", name="variance", datatype=",".join(dtype_list),
                       format=",".join(format_list), unknownshape_format=",".join(format_list))
    output0 = gen_param(classify="output0", name="batch_mean", datatype=",".join(dtype_list),
                        format=",".join(format_list), unknownshape_format=",".join(format_list))
    output1 = gen_param(classify="output1", name="batch_invstd", datatype=",".join(dtype_list),
                        format=",".join(format_list), unknownshape_format=",".join(format_list))
    output2 = gen_param(classify="output2", name="mean", datatype=",".join(dtype_list),
                        format=",".join(format_list), unknownshape_format=",".join(format_list))
    output3 = gen_param(classify="output3", name="variance", datatype=",".join(dtype_list),
                        format=",".join(format_list), unknownshape_format=",".join(format_list))

    param_dynamic_in_json = get_dynamic_param_in_json([input0, input1, input2, input3, input4,
                                                       output0, output1, output2, output3])

    return param_dynamic_in_json


class SyncBatchNormGatherStats(object):
    def __init__(self, total_sum, total_square_sum, sample_count, running_mean, running_var, batch_mean, batch_invstd,
                 running_mean_update, running_var_update, momentum=0.1, eps=1e-5,
                 kernel_name="sync_batch_norm_gather_stats"):
        self.tik_instance = tik.Tik()
        self.sum_dtype = total_sum.get("dtype")
        self.count_dtype = sample_count.get("dtype")
        self.kernel_name = kernel_name

        self.is_fp16 = self.sum_dtype == "float16"
        self.fp32 = "float32"
        self.c0 = 16
        self.block_byte_size = 32
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.dtype_byte_size = self.get_dtype_size(self.fp32)
        self.data_each_block = self.block_byte_size // self.dtype_byte_size

        self.max_mask = 64
        self.ub_n = 256
        self.iter_num = 8
        self.use_num = self.ub_n * self.c0
        self.tmp_ub = None
        # tiling_data
        self.block_num = None
        self.world_size = None
        self.c = None
        self.avg_c = None
        self.last_c = None
        self.momentum = None
        self.eps = None

        self.sum_gm = self.tik_instance.Tensor(self.sum_dtype, [MAX_INT32], scope=tik.scope_gm, name="sum_gm")
        self.square_sum_gm = self.tik_instance.Tensor(self.sum_dtype, [MAX_INT32], scope=tik.scope_gm,
                                                      name="square_sum_gm")
        self.count_gm = self.tik_instance.Tensor(self.count_dtype, [MAX_INT32], scope=tik.scope_gm, name="count_gm")
        self.running_mean_gm = self.tik_instance.Tensor(self.sum_dtype, [MAX_INT32], scope=tik.scope_gm,
                                                        name="running_mean_gm")
        self.running_var_gm = self.tik_instance.Tensor(self.sum_dtype, [MAX_INT32], scope=tik.scope_gm,
                                                       name="running_var_gm")
        self.batch_mean_gm = self.tik_instance.Tensor(self.sum_dtype, [MAX_INT32], scope=tik.scope_gm,
                                                      name="batch_mean_gm")
        self.batch_invstd_gm = self.tik_instance.Tensor(self.sum_dtype, [MAX_INT32], scope=tik.scope_gm,
                                                        name="batch_invstd_gm")
        self.mean_update_gm = self.tik_instance.Tensor(self.sum_dtype, [MAX_INT32], scope=tik.scope_gm,
                                                       name="mean_update_gm")
        self.var_update_gm = self.tik_instance.Tensor(self.sum_dtype, [MAX_INT32], scope=tik.scope_gm,
                                                      name="var_update_gm")
        self.tiling_gm = self.tik_instance.Tensor("int32", [TILING_NUM], scope=tik.scope_gm, name="tiling_gm")

    def get_dtype_size(self, dtype):
        """
        get byte size of dtype
        """
        dtype_dict = {"float32": 4, "uint8": 1, "int32": 4, "float16": 2, "int64": 8}
        return dtype_dict.get(dtype)

    def get_tiling_params(self):
        self.block_num = self.tik_instance.Scalar("int32")
        self.world_size = self.tik_instance.Scalar("int32")
        self.c = self.tik_instance.Scalar("int32")
        self.avg_c = self.tik_instance.Scalar("int32")
        self.last_c = self.tik_instance.Scalar("int32")
        self.momentum = self.tik_instance.Scalar(self.fp32)
        self.eps = self.tik_instance.Scalar(self.fp32)

        with self.tik_instance.new_stmt_scope():
            tiling_ub = self.tik_instance.Tensor("int32", shape=(TILING_NUM,), scope=tik.scope_ubuf, name="tiling_ub")
            self.data_move(tiling_ub, self.tiling_gm, num=TILING_NUM)

            self.block_num.set_as(tiling_ub[0])
            self.world_size.set_as(tiling_ub[1])
            self.c.set_as(tiling_ub[2])
            self.avg_c.set_as(tiling_ub[3])
            self.last_c.set_as(tiling_ub[4])
            self.momentum.set_as(tiling_ub[5])
            self.eps.set_as(tiling_ub[6])

    def compute(self):
        with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as block_idx:
            self.get_tiling_params()
            c_num = self.tik_instance.Scalar("int32")
            self.tmp_ub = self.tik_instance.Tensor("float16", [self.ub_n, self.c0], scope=tik.scope_ubuf,
                                                   name="conv_ub")
            with self.tik_instance.if_scope(block_idx < self.block_num):
                with self.tik_instance.if_scope(block_idx < self.block_num - 1):
                    c_num.set_as(self.avg_c)
                with self.tik_instance.if_scope(block_idx == self.block_num - 1):
                    c_num.set_as(self.last_c)

                self.compute_per_core(block_idx, c_num)
        
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        tbe_context.get_context().add_compile_info("vars", {"core_num": self.core_num})
        inputs = [self.sum_gm, self.square_sum_gm, self.count_gm, self.running_mean_gm, self.running_var_gm]
        outputs = [self.batch_mean_gm, self.batch_invstd_gm, self.mean_update_gm, self.var_update_gm]

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=inputs, outputs=outputs,
                                   flowtable=[self.tiling_gm], config=opt_config)
        return self.tik_instance

    def compute_per_core(self, block_idx, c_num):
        count_num = self.tik_instance.Scalar("float32", init_value=0)
        inv_count_num = self.tik_instance.Scalar("float32", init_value=0)
        inv_count_num_unbias = self.tik_instance.Scalar("float32", init_value=0)
        base_c = self.tik_instance.Scalar("int32", init_value=0)
        offset = self.tik_instance.Scalar("int32", init_value=0)
        loop_offset = self.tik_instance.Scalar("int32", init_value=0)

        with self.tik_instance.new_stmt_scope():
            self.compute_count(count_num)
        inv_count_num.set_as(1 / count_num)
        inv_count_num_unbias.set_as(count_num / (count_num - 1))

        sum_ub = self.tik_instance.Tensor(self.fp32, [self.ub_n, self.c0], scope=tik.scope_ubuf, name="sum_ub")
        square_sum_ub = self.tik_instance.Tensor(self.fp32, [self.ub_n, self.c0], scope=tik.scope_ubuf,
                                                 name="square_sum_ub")

        sum_all_ub = self.tik_instance.Tensor(self.fp32, [self.ub_n, self.c0], scope=tik.scope_ubuf, name="sum_all_ub")
        square_sum_all_ub = self.tik_instance.Tensor(self.fp32, [self.ub_n, self.c0], scope=tik.scope_ubuf,
                                                     name="square_sum_all_ub")

        var_res_ub = self.tik_instance.Tensor(self.fp32, [self.ub_n, self.c0], scope=tik.scope_ubuf, name="var_res_ub")
        mean_2_ub = self.tik_instance.Tensor(self.fp32, [self.ub_n, self.c0], scope=tik.scope_ubuf, name="mean_2_ub")
        invstd_ub = self.tik_instance.Tensor(self.fp32, [self.ub_n, self.c0], scope=tik.scope_ubuf, name="invstd_ub")

        loop = self.tik_instance.Scalar("int32")
        tail = self.tik_instance.Scalar("int32")
        loop.set_as(c_num // self.use_num)
        tail.set_as(c_num % self.use_num)

        base_c.set_as(block_idx * self.avg_c)
        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as loop_idx:
                ub_list = [sum_ub, square_sum_ub, sum_all_ub, square_sum_all_ub, mean_2_ub, var_res_ub, invstd_ub]
                self.clear_ub_list(ub_list, num=self.use_num)
                loop_offset.set_as(base_c + loop_idx * self.use_num)

                self.loop_compute(ub_list, inv_count_num, inv_count_num_unbias, offset, loop_offset, self.use_num)

        with self.tik_instance.if_scope(tail > 0):
            ub_list = [sum_ub, square_sum_ub, sum_all_ub, square_sum_all_ub, mean_2_ub, var_res_ub, invstd_ub]
            self.clear_ub_list(ub_list, num=tail)
            loop_offset.set_as(base_c + loop * self.use_num)

            self.loop_compute(ub_list, inv_count_num, inv_count_num_unbias, offset, loop_offset, tail)
    
    # 'pylint: disable=unused-argument,too-many-locals,too-many-arguments
    def loop_compute(self, ub_list, inv_count_num, inv_count_num_unbias, offset, loop_offset, num):
        sum_ub, square_sum_ub, sum_all_ub, square_sum_all_ub, mean_2_ub, var_res_ub, invstd_ub = ub_list
        with self.tik_instance.for_range(0, self.world_size) as world_idx:
            offset.set_as(world_idx * self.c + loop_offset)
            self.data_move(sum_ub, self.sum_gm[offset], num=num, need_conv=True)
            self.data_move(square_sum_ub, self.square_sum_gm[offset], num=num, need_conv=True)
            self.data_add(sum_all_ub, sum_all_ub, sum_ub, [0, 0, 0], num=num)
            self.data_add(square_sum_all_ub, square_sum_all_ub, square_sum_ub, [0, 0, 0], num=num)

        self.data_muls(sum_all_ub, sum_all_ub, inv_count_num, [0, 0], num=num)
        self.data_muls(square_sum_all_ub, square_sum_all_ub, inv_count_num, [0, 0], num=num)

        self.data_mul(mean_2_ub, sum_all_ub, sum_all_ub, [0, 0, 0], num=num)
        self.data_sub(var_res_ub, square_sum_all_ub, mean_2_ub, [0, 0, 0], num=num)
        self.compute_invstd(var_res_ub, invstd_ub, num=num)
        self.back_clear(invstd_ub, num=num)

        self.data_move(self.batch_mean_gm[loop_offset], sum_all_ub, num=num, need_conv=True, out=True)
        self.data_move(self.batch_invstd_gm[loop_offset], invstd_ub, num=num,
                       need_conv=True, out=True)

        update_mean_ub = self.tik_instance.Tensor(self.fp32, [self.ub_n, self.c0], scope=tik.scope_ubuf, 
                                                  name="update_mean_ub")
        update_var_ub = self.tik_instance.Tensor(self.fp32, [self.ub_n, self.c0], scope=tik.scope_ubuf, 
                                                  name="update_var_ub")
        self.dup_value(update_mean_ub, num=self.use_num)
        self.dup_value(update_var_ub, num=self.use_num)

        self.data_muls(var_res_ub, var_res_ub, inv_count_num_unbias, [0, 0], num=num)
        self.update_mean_and_var(sum_all_ub, var_res_ub, sum_ub, square_sum_ub, update_mean_ub, update_var_ub,
                                 loop_offset, num=num)
        
        self.data_move(self.mean_update_gm[loop_offset], update_mean_ub, num=num, need_conv=True, out=True)
        self.data_move(self.var_update_gm[loop_offset], update_var_ub, num=num, need_conv=True, out=True)
        
    def compute_count(self, count_num):
        loop = self.tik_instance.Scalar("int32")
        tail = self.tik_instance.Scalar("int32")
        tail_align = self.tik_instance.Scalar("int32")
        back_zero = self.tik_instance.Scalar("int32")
        tmp_scalar = self.tik_instance.Scalar(self.fp32)

        loop.set_as(self.world_size // self.use_num)
        tail.set_as(self.world_size % self.use_num)
        tail_align.set_as((tail + self.c0 - 1) // self.c0 * self.c0)
        back_zero.set_as(tail_align - tail)
        cum_ub = self.tik_instance.Tensor(self.count_dtype, [self.ub_n, self.c0], scope=tik.scope_ubuf, name="cum_ub")
        cum_ub_fp = self.tik_instance.Tensor(self.fp32, [self.c0], scope=tik.scope_ubuf, name="cum_ub_fp")
        sum_ub = self.tik_instance.Tensor(self.fp32, [self.c0], scope=tik.scope_ubuf, name="sum_ub")
        self.dup_value(sum_ub, self.c0)

        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as loop_idx:
                self.data_move(cum_ub, self.count_gm[loop_idx * self.use_num], num=self.use_num)
                self.data_sum(cum_ub, num=self.use_num, iter_num=self.iter_num)
                self.data_conv(cum_ub_fp, cum_ub, [0, 0], mode="", num=self.c0)
                self.tik_instance.vcadd(16, sum_ub, cum_ub_fp, 1, 1, 1, 1)
                tmp_scalar.set_as(sum_ub[0])
                count_num.set_as(count_num + tmp_scalar)

        with self.tik_instance.if_scope(tail > 0):
            self.dup_value(cum_ub, self.use_num)
            self.data_move(cum_ub, self.count_gm[loop * self.use_num], num=tail)
            with self.tik_instance.if_scope(back_zero > 0):
                with self.tik_instance.for_range(0, back_zero) as idx:
                    cum_ub[tail_align - 1 - idx].set_as(0)
            self.data_sum(cum_ub, num=self.use_num, iter_num=self.iter_num)
            self.data_conv(cum_ub_fp, cum_ub, [0, 0], mode="", num=self.c0)
            self.tik_instance.vcadd(16, sum_ub, cum_ub_fp, 1, 1, 1, 1)
            tmp_scalar.set_as(sum_ub[0])
            count_num.set_as(count_num + tmp_scalar)

    def compute_invstd(self, var_res_ub, invstd_ub, num):
        work_tensor = self.tik_instance.Tensor(self.fp32, [192], scope=tik.scope_ubuf, name="work_tensor")
        var_eps_ub = self.tik_instance.Tensor(self.fp32, [self.ub_n, self.c0], scope=tik.scope_ubuf, 
                                              name="var_eps_ub")
        loop = self.tik_instance.Scalar("int32")
        tail = self.tik_instance.Scalar("int32")
        loop.set_as(num // self.max_mask)
        tail.set_as(num % self.max_mask)

        self.data_adds(var_eps_ub, var_res_ub, self.eps, [0, 0], num=num)
        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as loop_idx:
                self.tik_instance.vec_rsqrt_high_preci(self.max_mask, invstd_ub[loop_idx * self.max_mask],
                                                       var_eps_ub[loop_idx * self.max_mask], work_tensor, 1, 1, 1)

        with self.tik_instance.if_scope(tail > 0):
            self.tik_instance.vec_rsqrt_high_preci(tail, invstd_ub[loop * self.max_mask],
                                                   var_eps_ub[loop * self.max_mask], work_tensor, 1, 1, 1)
    
    def update_mean_and_var(self, mean_res_ub, var_res_ub, running_mean_ub, runing_var_ub,
                            update_mean_ub, update_var_ub, offset, num):
        self.data_move(running_mean_ub, self.running_mean_gm[offset], num=num, need_conv=True)
        self.data_move(runing_var_ub, self.running_var_gm[offset], num=num, need_conv=True)

        self.data_muls(running_mean_ub, running_mean_ub, 1 - self.momentum, [0, 0], num=num)
        self.data_muls(mean_res_ub, mean_res_ub, self.momentum, [0, 0], num=num)
        self.data_add(update_mean_ub, running_mean_ub, mean_res_ub, [0, 0, 0], num=num)

        self.data_muls(runing_var_ub, runing_var_ub, 1 - self.momentum, [0, 0], num=num)
        self.data_muls(var_res_ub, var_res_ub, self.momentum, [0, 0], num=num)
        self.data_add(update_var_ub, runing_var_ub, var_res_ub, [0, 0, 0], num=num)

    def clear_ub_list(self, ub_list, num):
        for ub_tensor in ub_list:
            self.dup_value(ub_tensor, num)

    def back_clear(self, ub_tensor, num):
        align_num = self.tik_instance.Scalar("int32")
        back_num = self.tik_instance.Scalar("int32")
        align_num.set_as((num + self.data_each_block - 1) // self.data_each_block * self.data_each_block)
        back_num.set_as(align_num - num)
        with self.tik_instance.if_scope(back_num > 0):
            with self.tik_instance.for_range(0, back_num) as back_id:
                ub_tensor[num + back_id].set_as(0)

    def data_move(self, dst, src, num, src_stride=0, dst_stride=0, need_conv=False, out=False):
        """
        move data
        """
        sid = 0
        nburst = 1
        if self.is_fp16 and need_conv:
            dtype_byte_size = self.get_dtype_size("float16")
            data_each_block = self.block_byte_size // dtype_byte_size
            burst_len = (num + data_each_block - 1) // data_each_block
            if not out:
                self.tik_instance.data_move(self.tmp_ub, src, sid, nburst, burst_len, src_stride=src_stride,
                                            dst_stride=dst_stride)
                self.data_conv(dst, self.tmp_ub, [0, 0], mode="", num=num, dst_stride=8, src_stride=4)
            else:
                self.data_conv(self.tmp_ub, src, [0, 0], mode="", num=num, dst_stride=4, src_stride=8)
                self.tik_instance.data_move(dst, self.tmp_ub, sid, nburst, burst_len, src_stride=src_stride,
                                            dst_stride=dst_stride)
        else:
            dtype_byte_size = self.get_dtype_size(dst.dtype)
            data_each_block = self.block_byte_size // dtype_byte_size
            burst_len = (num + data_each_block - 1) // data_each_block
            self.tik_instance.data_move(dst, src, sid, nburst, burst_len, src_stride=src_stride, dst_stride=dst_stride)

    def dup_value(self, dst, num, dup_value=0, offset=0):
        """
        dup value to ub
        """
        offset = self.tik_instance.Scalar("int32", init_value=offset)
        dtype_byte_size = self.get_dtype_size(dst.dtype)
        mask = 256 // dtype_byte_size
        stride = mask // self.data_each_block

        loop = num // (mask * 255)
        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as index:
                tmp_offset = offset + index * mask * 255
                self.tik_instance.vec_dup(mask, dst[tmp_offset], dup_value, 255, stride)
            offset.set_as(offset + loop * mask * 255)

        repeat_time = (num % (mask * 255)) // mask
        with self.tik_instance.if_scope(repeat_time > 0):
            self.tik_instance.vec_dup(mask, dst[offset], dup_value, repeat_time, stride)
            offset.set_as(offset + repeat_time * mask)

        last_num = num % mask
        with self.tik_instance.if_scope(last_num > 0):
            self.tik_instance.vec_dup(last_num, dst[offset], dup_value, 1, stride)

    def data_sum(self, src, num, iter_num):
        """
        sum data
        """
        for _ in range(iter_num):
            num = num // 2
            if num // self.max_mask > 0:
                mask = self.max_mask
                repeat_time = num // self.max_mask
            else:
                mask = num
                repeat_time = 1

            src_stride = mask // self.data_each_block
            self.tik_instance.vec_add(mask, src, src[num], src, repeat_time, 0, src_stride, 0)

    def single_operator_template(self, op_obj, dst, src, scalar, offsets, num=0, dst_stride=8, src_stride=8):
        """
        tik api template
        """
        dst_offset = self.tik_instance.Scalar("int32", init_value=offsets[0])
        src_offset = self.tik_instance.Scalar("int32", init_value=offsets[1])
        vector_mask_max = 256 // self.dtype_byte_size

        tensor_size = num
        loop = tensor_size // (vector_mask_max * 255)

        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst_offset = dst_offset + index * vector_mask_max * 255
                tmp_src_offset = src_offset + index * vector_mask_max * 255
                op_obj(vector_mask_max, dst[tmp_dst_offset], src[tmp_src_offset], scalar, 255,
                       dst_stride, src_stride)

            dst_offset.set_as(dst_offset + loop * vector_mask_max * 255)
            src_offset.set_as(src_offset + loop * vector_mask_max * 255)

        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max

        with self.tik_instance.if_scope(repeat_time > 0):
            op_obj(vector_mask_max, dst[dst_offset], src[src_offset], scalar, repeat_time, dst_stride, src_stride)
            dst_offset.set_as(dst_offset + repeat_time * vector_mask_max)
            src_offset.set_as(src_offset + repeat_time * vector_mask_max)

        last_num = tensor_size % vector_mask_max
        with self.tik_instance.if_scope(last_num > 0):
            op_obj(last_num, dst[dst_offset], src[src_offset], scalar, 1, dst_stride, src_stride)

    def double_operator_template(self, op_obj, dst, src0, src1, offsets, num=0, dst_stride=8, src0_stride=8,
                                 src1_stride=8):
        """
        tik api template
        """
        dst_offset = self.tik_instance.Scalar("int32", init_value=offsets[0])
        src0_offset = self.tik_instance.Scalar("int32", init_value=offsets[1])
        src1_offset = self.tik_instance.Scalar("int32", init_value=offsets[2])
        vector_mask_max = 256 // self.dtype_byte_size

        tensor_size = num
        loop = tensor_size // (vector_mask_max * 255)

        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst_offset = dst_offset + index * vector_mask_max * 255
                tmp_src0_offset = src0_offset + index * vector_mask_max * 255
                tmp_src1_offset = src1_offset + index * vector_mask_max * 255
                op_obj(vector_mask_max, dst[tmp_dst_offset], src0[tmp_src0_offset], src1[tmp_src1_offset], 255,
                       dst_stride, src0_stride, src1_stride)

            dst_offset.set_as(dst_offset + loop * vector_mask_max * 255)
            src0_offset.set_as(src0_offset + loop * vector_mask_max * 255)
            src1_offset.set_as(src1_offset + loop * vector_mask_max * 255)

        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max

        with self.tik_instance.if_scope(repeat_time > 0):
            op_obj(vector_mask_max, dst[dst_offset], src0[src0_offset], src1[src1_offset], repeat_time, dst_stride,
                   src0_stride, src1_stride)
            dst_offset.set_as(dst_offset + repeat_time * vector_mask_max)
            src0_offset.set_as(src0_offset + repeat_time * vector_mask_max)
            src1_offset.set_as(src1_offset + repeat_time * vector_mask_max)

        last_num = tensor_size % vector_mask_max
        with self.tik_instance.if_scope(last_num > 0):
            op_obj(last_num, dst[dst_offset], src0[src0_offset], src1[src1_offset], 1, dst_stride, src0_stride,
                   src1_stride)

    def data_conv(self, dst, src, offsets, mode="ceil", num=0, dst_stride=8, src_stride=8):
        """
        tik conv
        """
        dst_offset = self.tik_instance.Scalar("int32", init_value=offsets[0])
        src_offset = self.tik_instance.Scalar("int32", init_value=offsets[1])
        vector_mask_max = 64

        tensor_size = num
        loop = tensor_size // (vector_mask_max * 255)

        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst_offset = dst_offset + index * vector_mask_max * 255
                tmp_src_offset = src_offset + index * vector_mask_max * 255
                self.tik_instance.vec_conv(vector_mask_max, mode, dst[tmp_dst_offset], src[tmp_src_offset], 255,
                                           dst_stride, src_stride)

            dst_offset.set_as(dst_offset + loop * vector_mask_max * 255)
            src_offset.set_as(src_offset + loop * vector_mask_max * 255)

        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max

        with self.tik_instance.if_scope(repeat_time > 0):
            self.tik_instance.vec_conv(vector_mask_max, mode, dst[dst_offset], src[src_offset], repeat_time,
                                       dst_stride, src_stride)
            dst_offset.set_as(dst_offset + repeat_time * vector_mask_max)
            src_offset.set_as(src_offset + repeat_time * vector_mask_max)

        last_num = tensor_size % vector_mask_max
        with self.tik_instance.if_scope(last_num > 0):
            self.tik_instance.vec_conv(last_num, mode, dst[dst_offset], src[src_offset], 1, dst_stride, src_stride)
    
    def data_adds(self, dst, src, scalar, offsets, num=0, dst_stride=8, src_stride=8):
        """
        tik adds
        """
        self.single_operator_template(self.tik_instance.vec_adds, dst, src, scalar, offsets, num, dst_stride,
                                      src_stride)
    
    # 'pylint: disable=unused-argument,too-many-locals
    def data_add(self, dst, src0, src1, offsets, num=0, dst_stride=8, src0_stride=8, src1_stride=8):
        """
        tik add
        """
        self.double_operator_template(self.tik_instance.vec_add, dst, src0, src1, offsets, num, dst_stride, src0_stride,
                                      src1_stride)
    
    # 'pylint: disable=unused-argument,too-many-locals
    def data_mul(self, dst, src0, src1, offsets, num=0, dst_stride=8, src0_stride=8, src1_stride=8):
        """
        tik mul
        """
        self.double_operator_template(self.tik_instance.vec_mul, dst, src0, src1, offsets, num, dst_stride, src0_stride,
                                      src1_stride)
    
    # 'pylint: disable=unused-argument,too-many-locals
    def data_muls(self, dst, src, scalar, offsets, num=0, dst_stride=8, src_stride=8):
        """
        tik muls
        """
        self.single_operator_template(self.tik_instance.vec_muls, dst, src, scalar, offsets, num, dst_stride,
                                      src_stride)
    
    # 'pylint: disable=unused-argument,too-many-locals
    def data_sub(self, dst, src0, src1, offsets, num=0, dst_stride=8, src0_stride=8, src1_stride=8):
        """
        tik sub
        """
        self.double_operator_template(self.tik_instance.vec_sub, dst, src0, src1, offsets, num, dst_stride, src0_stride,
                                      src1_stride)


def check_param(total_sum, total_square_sum, sample_count, running_mean, running_var):
    sum_dtype = total_sum.get("dtype")
    total_sum_dtype = total_square_sum.get("dtype")
    count_dtype = sample_count.get("dtype")
    mean_dtype = running_mean.get("dtype")
    var_dtype = running_var.get("dtype")

    if total_sum_dtype != sum_dtype or mean_dtype != sum_dtype or var_dtype != sum_dtype:
        raise RuntimeError("input_dtype should be same except count")

    if sum_dtype not in ["float16", "float32"]:
        raise RuntimeError("sum_dtype only supports float16 or float32")

    if count_dtype != "int32":
        raise RuntimeError("count dtype must be int32")


# 'pylint: disable=unused-argument, too-many-locals, too-many-lines
@register_operator("SyncBatchNormGatherStats")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def sync_batch_norm_gather_stats(total_sum, total_square_sum, sample_count, running_mean, running_var, batch_mean,
                                 batch_invstd, running_mean_update, running_var_update, momentum=0.1, eps=1e-5,
                                 kernel_name="sync_batch_norm_gather_stats"):
    check_param(total_sum, total_square_sum, sample_count, running_mean, running_var)
    instance = SyncBatchNormGatherStats(total_sum, total_square_sum, sample_count, running_mean, running_var,
                                        batch_mean, batch_invstd, running_mean_update, running_var_update,
                                        momentum=momentum, eps=eps, kernel_name=kernel_name)
    return instance.compute()
