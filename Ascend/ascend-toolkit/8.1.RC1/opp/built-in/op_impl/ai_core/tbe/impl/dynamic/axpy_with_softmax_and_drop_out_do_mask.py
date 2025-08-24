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
axpy_with_softmax_and_drop_out_do_mask
""" 
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context


# 'pylint: disable=too-few-public-methods
class Constant:
    # LINE_THRESHOLD：
    # if the UB size is larger than 200k, two rows of data can be processed
    # if the UB size is smaller than 200k, one rows of data can be processed
    LINE_THRESHOLD = 204800 
    BLOCK = 16
    BLOCK_SQUARE = 256
    VEC_MASK = 128
    VEC_MASK_FP32 = 64
    VEC_MASK_FP16 = 128
    VEC_DUMP_SHAPE = 128
    FP16_NUM_PER_BLOCK = 16
    FP32_NUM_PER_BLOCK = 8
    MAX_REPEAT = 255
    LEN = 2
    LIMIT_LEN = 512
    TILING_PARAMS_NUM = 16
    MAX_INT32 = 2 ** 31 - 1
    SHAPE_W2 = 16
    SHAPE_H2 = 16


# 'pylint: disable=invalid-name,too-many-statements,too-many-locals,too-many-arguments,unused-argument
# 'pylint: disable=too-many-instance-attributes,too-many-public-methods,too-many-lines
class AxpyWithSoftmaxAndDropOutDoMask():
    """
    Create: 2023-02-11
    Modify: 2023-02-14
    """
    def __init__(self, x1, x2, mask, y1, y2, alpha, input_keep_prob, axis=-1, 
                 kernel_name="axpy_with_softmax_and_drop_out_do_mask"):
        self.tensor_dtype = x1.get("dtype").lower()
        self.mask_dtype = mask.get("dtype").lower()

        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        if tbe_platform.api_check_support("tik.vcopy"):
            self.sel_mode = 1
        else:
            self.sel_mode = 0
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.kernel_name = kernel_name

        self.x1_gm = self.tik_instance.Tensor(self.tensor_dtype, (Constant.MAX_INT32,), name="x1_gm",
                                              scope=tik.scope_gm)
        self.x2_gm = self.tik_instance.Tensor(self.tensor_dtype, (Constant.MAX_INT32,), name="x2_gm",
                                              scope=tik.scope_gm)
        self.mask_gm = self.tik_instance.Tensor(self.mask_dtype, (Constant.MAX_INT32,), name="mask_gm",
                                                scope=tik.scope_gm)
        self.y1_gm = self.tik_instance.Tensor(self.tensor_dtype, (Constant.MAX_INT32,), name="y1_gm",
                                              scope=tik.scope_gm)
        self.y2_gm = self.tik_instance.Tensor(self.tensor_dtype, (Constant.MAX_INT32,), name="y2_gm",
                                              scope=tik.scope_gm)
                                              
        self.dup_value = self.tik_instance.Scalar(init_value=0, dtype="uint16")
        self.line = None
        self.ranges = None
        if self.tensor_dtype == "float16":
            self.dup_sel_0 = self.tik_instance.Tensor(self.tensor_dtype, (128,), scope=tik.scope_ubuf, name="dup_sel_0")
            self.tik_instance.vec_dup(128, self.dup_sel_0, 0, 1, 0)
        else:
            self.dup_sel_0 = self.tik_instance.Tensor(self.tensor_dtype, (64,), scope=tik.scope_ubuf, name="dup_sel_0")
            self.tik_instance.vec_dup(64, self.dup_sel_0, 0, 1, 0)

        self.tiling_gm = self.tik_instance.Tensor("int32", (Constant.TILING_PARAMS_NUM,), name="tiling_gm",
                                                  scope=tik.scope_gm)
        self.get_tiling_args()

    def get_tiling_args(self):
        """get_tiling_args"""
        tiling_ub = self.tik_instance.Tensor("int32", (Constant.TILING_PARAMS_NUM,),
                                             name='tiling_ub', scope=tik.scope_ubuf)
        self.batch_per_core = self.tik_instance.Scalar("int32", name="batch_per_core")
        self.batch_small_per_core = self.tik_instance.Scalar("int32", name="batch_small_per_core")
        self.batch_large_core_num = self.tik_instance.Scalar("int32", name="batch_large_core_num")
        self.ele_per_batch = self.tik_instance.Scalar("int32", name="ele_per_batch")
        self.w_dim = self.tik_instance.Scalar("int32", name="w_dim")
        self.dim_n = self.tik_instance.Scalar("int32", name="dim_n")
        self.dim_c = self.tik_instance.Scalar("int32", name="dim_c")
        self.dim_w1 = self.tik_instance.Scalar("int32", name="dim_w1")
        self.dim_h1 = self.tik_instance.Scalar("int32", name="dim_h1")
        self.cnt = self.tik_instance.Scalar("int32", name="cnt")
        self.remain = self.tik_instance.Scalar("int32", name="remain")
        self.alpha = self.tik_instance.Scalar("float32", name="alpha")
        self.input_keep_prob = self.tik_instance.Scalar("float32", name="input_keep_prob")
        self.mask_is_bit = self.tik_instance.Scalar("int32", name="mask_is_bit")

        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, 2, 0, 0)

        self.batch_per_core.set_as(tiling_ub[0])
        self.batch_small_per_core.set_as(tiling_ub[1])
        self.batch_large_core_num.set_as(tiling_ub[2])
        self.ele_per_batch.set_as(tiling_ub[3])
        self.w_dim.set_as(tiling_ub[4])
        self.dim_n.set_as(tiling_ub[5])
        self.dim_c.set_as(tiling_ub[6])
        self.dim_w1.set_as(tiling_ub[7])
        self.dim_h1.set_as(tiling_ub[8])
        self.cnt.set_as(tiling_ub[9])
        self.remain.set_as(tiling_ub[10])
        self.alpha.set_as(tiling_ub[11])
        self.input_keep_prob.set_as(tiling_ub[12])
        self.mask_is_bit.set_as(tiling_ub[13])

    def axpy_with_softmax_and_drop_out_do_mask_compute(self):
        with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as blockid:
            with self.tik_instance.if_scope(self.w_dim == Constant.LIMIT_LEN):
                self.axpy_with_softmax_and_drop_out_do_mask_w_large(blockid)
            with self.tik_instance.else_scope():
                self.axpy_with_softmax_and_drop_out_do_mask_w_small(blockid)
    
        tbe_context.get_context().add_compile_info(
            "vars", {
                "core_num": self.core_num
            })
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.x1_gm, self.x2_gm, self.mask_gm],
                                   outputs=[self.y1_gm, self.y2_gm],
                                   flowtable=[self.tiling_gm])
        return self.tik_instance

    def axpy_with_softmax_and_drop_out_do_mask_w_large(self, blockid):
        w1_len = self.tik_instance.Scalar("int32", name="w1_len", init_value=self.dim_w1)
        h1_len = self.tik_instance.Scalar("int32", name="h1_len", init_value=self.dim_h1)
        core_offset = self.tik_instance.Scalar("int32", name="core_offset")
        core_process_num = self.tik_instance.Scalar("int32", name="core_process_num",
                                                    init_value=w1_len * h1_len * 16 * 16)
        
        block_count_truth = self.tik_instance.Scalar("int32", name='block_count_truth')
        core_offset = self.tik_instance.Scalar("int32", name="core_offset")
        move_offset = self.tik_instance.Scalar("int32", name="move_offset")

        with self.tik_instance.if_scope(blockid < self.batch_large_core_num):
            block_count_truth.set_as(self.batch_per_core)
            core_offset.set_as(blockid * self.batch_per_core)
        with self.tik_instance.else_scope():
            block_count_truth.set_as(self.batch_small_per_core)
            core_offset.set_as(self.batch_large_core_num * self.batch_per_core + \
                               (blockid - self.batch_large_core_num) * self.batch_small_per_core)
        with self.tik_instance.for_range(0, block_count_truth) as i:
            with self.tik_instance.for_range(0, w1_len // 2) as j:
                ub_1 = self.tik_instance.Tensor(self.tensor_dtype, (h1_len, 32, 16), scope=tik.scope_ubuf, name="ub_1")
                ub_2 = self.tik_instance.Tensor(self.tensor_dtype, (h1_len, 32, 16), scope=tik.scope_ubuf, name="ub_2")
                
                move_offset.set_as((core_offset + i) * core_process_num + j * 512)
                if self.tensor_dtype == "float32":
                    self.large_fp32_compute(move_offset, ub_1, ub_2, w1_len, h1_len)
                else:
                    self.large_fp16_compute(move_offset, ub_1, ub_2, w1_len, h1_len)

    def large_fp16_compute(self, move_offset, ub_1, ub_2, w1_len, h1_len):
        # gm->ub
        self.tik_instance.data_move(ub_1[0], self.x1_gm[move_offset], 0, w1_len, 32, h1_len * 16 - 32, 0)
        self.tik_instance.data_move(ub_2[0], self.x2_gm[move_offset], 0, w1_len, 32, h1_len * 16 - 32, 0)

        # axpyv2
        alpha_fp16 = self.tik_instance.Scalar("float16", name="alpha_fp16")
        self.tik_instance.scalar_conv("", alpha_fp16, self.alpha)
        self.tik_instance.vmuls(Constant.VEC_MASK, ub_2, ub_2, alpha_fp16, h1_len * 4, 1, 1, 8, 8)
        self.tik_instance.vadd(Constant.VEC_MASK, ub_1, ub_1, ub_2, h1_len * 4, 1, 1, 1, 8, 8, 8)

        ub_broadcast = self.tik_instance.Tensor("uint16", (32 * 16,), scope=tik.scope_ubuf,
                                                name="ub_broadcast")
        with self.tik_instance.new_stmt_scope():
            ub_reducemax = self.tik_instance.Tensor("float16", (32,), scope=tik.scope_ubuf, 
                                                    name="ub_reducemax")
            ub_dup = self.tik_instance.Tensor("uint16", (128,), scope=tik.scope_ubuf,
                                                name="ub_dup")
            ub_cast = self.tik_instance.Tensor("float32", (h1_len, 32, 16), scope=tik.scope_ubuf,
                                                name="ub_cast")
            ub_reduceadd = self.tik_instance.Tensor("float32", (32,), scope=tik.scope_ubuf,
                                                    name="ub_reduceadd")
            ub_reduceadd_high_preci = self.tik_instance.Tensor("float32", (32,), scope=tik.scope_ubuf,
                                                                name="ub_reduceadd_high_preci")
            work_tensor_ub = self.tik_instance.Tensor("float32", (64,), scope=tik.scope_ubuf,
                                                        name="work_tensor_ub")
            ub_reduceadd_fp16 = self.tik_instance.Tensor("float16", (32,), scope=tik.scope_ubuf,
                                                            name="ub_reduceadd_fp16")
            # get max element
            self.get_max_ele(ub_1, ub_2, ub_reducemax, h1_len)

            # ub_reducemax broadcast 32 -> (32, 16)
            ub_broadcast_fp16 = self.broadcast(ub_dup, ub_reducemax, ub_broadcast)

            # x - x_max
            with self.tik_instance.for_range(0, 4) as idx:
                self.tik_instance.vsub(Constant.VEC_MASK, ub_2[idx * 128], ub_1[idx * 128],
                                        ub_broadcast_fp16[idx * 128], 32, 1, 1, 1, h1_len, h1_len, 0)

            # exp
            self.exp(ub_2, ub_1, ub_cast)

            # sum exp
            self.sum_exp(ub_cast, ub_reduceadd, h1_len)

            # 1 / sum_exp
            self.tik_instance.vec_rec_high_preci(32, ub_reduceadd_high_preci[0], ub_reduceadd[0],
                                                    work_tensor_ub[0:], 1, 4, 4)
            self.tik_instance.vconv(32, "", ub_reduceadd_fp16[0], ub_reduceadd_high_preci[0], 1, 1, 1, 0, 0)

            # ub_reduceadd_fp16 broadcast 32 -> (32, 16)
            ub_broadcast_fp16 = self.broadcast(ub_dup, ub_reduceadd_fp16, ub_broadcast)

        ub_temp = self.tik_instance.Tensor("float16", (h1_len, 32, 16), scope=tik.scope_ubuf,
                                            name="ub_temp")
        # calculate exp * (1 / sum_exp)
        with self.tik_instance.for_range(0, 4) as idx:
            self.tik_instance.vmul(Constant.VEC_MASK, ub_temp[idx * 128], ub_1[idx * 128],
                                    ub_broadcast_fp16[idx * 128], 32, 1, 1, 1, h1_len, h1_len, 0)

        # dropoutdomaskv3
        with self.tik_instance.if_scope(self.input_keep_prob == 0):
            self.tik_instance.vmuls(Constant.VEC_MASK, ub_2[0], ub_temp[0],
                                    self.tik_instance.Scalar(init_value=0.0, dtype="float16"),
                                    h1_len * 4, 1, 1, 8, 8)
        with self.tik_instance.else_scope():
            self.tik_instance.vmuls(Constant.VEC_MASK, ub_2[0], ub_temp[0],
                                    self.tik_instance.Scalar(init_value=1 / self.input_keep_prob,
                                    dtype="float16"), h1_len * 4, 1, 1, 8, 8)

        with self.tik_instance.if_scope(self.mask_is_bit == 1):
            ub_mask_0 = self.tik_instance.Tensor("uint8", (32*32*2,), scope=tik.scope_ubuf, name="ub_mask_0")
            self.tik_instance.data_move(ub_mask_0, self.mask_gm[move_offset // 8], 0, 32, 2, 32 - 2, 0)
            if self.sel_mode == 0:
                ub_mask_1 = self.tik_instance.Tensor("uint8", (32*32*2,), scope=tik.scope_ubuf, name="ub_mask_1")
                self.tik_instance.data_move(ub_mask_1, self.mask_gm[move_offset // 8 + 16], 0, 32, 2, 32 - 2, 0)
                for mask_idx in range(0, 64):
                    self.tik_instance.vec_sel(128, 0, ub_1[mask_idx*256], ub_mask_0[mask_idx*32], ub_2[mask_idx*256],
                                              self.dup_sel_0, 1, 8, 8, 0)
                    self.tik_instance.vec_sel(128, 0, ub_1[mask_idx*256 + 128], ub_mask_1[mask_idx*32],
                                              ub_2[mask_idx*256 + 128], self.dup_sel_0, 1, 8, 8, 0)
            else:
                self.tik_instance.vec_sel(Constant.VEC_MASK, 1, ub_1, ub_mask_0, ub_2, 0, 128, 8, 8)
        with self.tik_instance.else_scope():
            ub_mask = self.tik_instance.Tensor("uint8", (h1_len, 32, 16), scope=tik.scope_ubuf,
                                                name="ub_mask")
            ub_mask_fp16 = self.tik_instance.Tensor("float16", (h1_len, 32, 16), scope=tik.scope_ubuf,
                                                    name="ub_mask_fp16")
            self.tik_instance.data_move(ub_mask[0], self.mask_gm[move_offset], 0, w1_len, 16, h1_len * 8 - 16, 0)
            self.tik_instance.vconv(Constant.VEC_MASK, "", ub_mask_fp16[0], ub_mask[0], h1_len * 4, 1, 1, 8, 4)
            self.tik_instance.vmul(Constant.VEC_MASK, ub_1[0], ub_mask_fp16[0], ub_2[0], h1_len * 4,
                                    1, 1, 1, 8, 8, 8)

        # ub -> gm
        self.tik_instance.data_move(self.y1_gm[move_offset], ub_temp[0], 0, w1_len, 32, 0, h1_len * 16 - 32)
        self.tik_instance.data_move(self.y2_gm[move_offset], ub_1[0], 0, w1_len, 32, 0, h1_len * 16 - 32)
    
    def large_fp32_compute(self, move_offset, ub_1, ub_2, w1_len, h1_len):
        # gm->ub
        self.tik_instance.data_move(ub_1[0], self.x1_gm[move_offset], 0, w1_len, 64, h1_len * 32 - 64, 0)
        self.tik_instance.data_move(ub_2[0], self.x2_gm[move_offset], 0, w1_len, 64, h1_len * 32 - 64, 0)

        # axpyv2
        self.tik_instance.vmuls(Constant.VEC_MASK_FP32, ub_2, ub_2, self.alpha, Constant.MAX_REPEAT, 1, 1, 8, 8)
        self.tik_instance.vmuls(Constant.VEC_MASK_FP32, ub_2[Constant.VEC_MASK_FP32 * Constant.MAX_REPEAT], \
                                ub_2[Constant.VEC_MASK_FP32 * Constant.MAX_REPEAT], self.alpha, 1, 1, 1, 8, 8)
        self.tik_instance.vadd(Constant.VEC_MASK_FP32, ub_1, ub_1, ub_2, Constant.MAX_REPEAT, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(Constant.VEC_MASK_FP32, ub_1[Constant.VEC_MASK_FP32 * Constant.MAX_REPEAT], \
                               ub_1[Constant.VEC_MASK_FP32 * Constant.MAX_REPEAT], \
                               ub_2[Constant.VEC_MASK_FP32 * Constant.MAX_REPEAT], 1, 1, 1, 1, 8, 8, 8)

        with self.tik_instance.new_stmt_scope():
            ub_reducemax = self.tik_instance.Tensor("float32", (32,), scope=tik.scope_ubuf, 
                                                    name="ub_reducemax")
            ub_reduceadd = self.tik_instance.Tensor("float32", (32,), scope=tik.scope_ubuf,
                                                    name="ub_reduceadd")
            ub_reduceadd_high_preci = self.tik_instance.Tensor("float32", (32,), scope=tik.scope_ubuf,
                                                                name="ub_reduceadd_high_preci")
            work_tensor_ub = self.tik_instance.Tensor("float32", (64,), scope=tik.scope_ubuf,
                                                        name="work_tensor_ub")

            # get max element
            self.get_max_ele_fp32(ub_1, ub_2, ub_reducemax, h1_len)

            # x - x_max
            max_value = self.tik_instance.Scalar("float32", name="max_value")
            with self.tik_instance.for_range(0, 32) as idx:
                max_value.set_as(ub_reducemax[idx])
                self.tik_instance.vadds(16, ub_2[idx * 16], ub_1[idx * 16], max_value * (-1), 32, 1, 1, 64, 64)

            # exp
            self.tik_instance.vexp(Constant.VEC_MASK_FP32, ub_2[0], ub_2[0], Constant.MAX_REPEAT, 1, 1, 8, 8)
            self.tik_instance.vexp(Constant.VEC_MASK_FP32, ub_2[Constant.VEC_MASK_FP32 * Constant.MAX_REPEAT],
                                   ub_2[Constant.VEC_MASK_FP32 * Constant.MAX_REPEAT], 1, 1, 1, 8, 8)

            # sum exp
            self.sum_exp_fp32(ub_2, ub_1, ub_reduceadd, h1_len)

            # 1 / sum_exp
            self.tik_instance.vec_rec_high_preci(32, ub_reduceadd_high_preci[0], ub_reduceadd[0],
                                                    work_tensor_ub[0:], 1, 4, 4)

            # calculate exp * (1 / sum_exp)
            vmuls_value = self.tik_instance.Scalar("float32", name="vmuls_value")
            with self.tik_instance.for_range(0, 32) as idx:
                vmuls_value.set_as(ub_reduceadd_high_preci[idx])
                self.tik_instance.vmuls(16, ub_1[idx * 16], ub_2[idx * 16], vmuls_value, 32, 1, 1, 64, 64)
        
        # ub_1 -> y1_gm
        self.tik_instance.data_move(self.y1_gm[move_offset], ub_1[0], 0, w1_len, 64, 0, h1_len * 32 - 64)

        # dropoutdomaskv3
        with self.tik_instance.if_scope(self.input_keep_prob == 0):
            self.tik_instance.vmuls(Constant.VEC_MASK_FP32, ub_2[0], ub_1[0], 0, Constant.MAX_REPEAT, 1, 1, 8, 8)
            self.tik_instance.vmuls(Constant.VEC_MASK_FP32, ub_2[Constant.VEC_MASK_FP32 * Constant.MAX_REPEAT],
                                    ub_1[Constant.VEC_MASK_FP32 * Constant.MAX_REPEAT], 0, 1, 1, 1, 8, 8)
        with self.tik_instance.else_scope():
            self.tik_instance.vmuls(Constant.VEC_MASK_FP32, ub_2[0], ub_1[0],
                                    self.tik_instance.Scalar(init_value=1 / self.input_keep_prob, dtype="float32"),
                                    Constant.MAX_REPEAT, 1, 1, 8, 8)
            self.tik_instance.vmuls(Constant.VEC_MASK_FP32, ub_2[Constant.VEC_MASK_FP32 * Constant.MAX_REPEAT],
                                    ub_1[Constant.VEC_MASK_FP32 * Constant.MAX_REPEAT],
                                    self.tik_instance.Scalar(init_value=1 / self.input_keep_prob, dtype="float32"),
                                    1, 1, 1, 8, 8)

        with self.tik_instance.if_scope(self.mask_is_bit == 1):
            ub_mask_0 = self.tik_instance.Tensor("uint8", (32*32*2,), scope=tik.scope_ubuf, name="ub_mask_0")
            self.tik_instance.data_move(ub_mask_0, self.mask_gm[move_offset // 8], 0, 32, 2, 32 - 2, 0)
            if self.sel_mode == 0:
                ub_mask_1 = self.tik_instance.Tensor("uint8", (32*32*2,), scope=tik.scope_ubuf, name="ub_mask_1")
                ub_mask_2 = self.tik_instance.Tensor("uint8", (32*32*2,), scope=tik.scope_ubuf, name="ub_mask_2")
                ub_mask_3 = self.tik_instance.Tensor("uint8", (32*32*2,), scope=tik.scope_ubuf, name="ub_mask_3")
                self.tik_instance.data_move(ub_mask_1, self.mask_gm[move_offset // 8 + 8], 0, 32, 2, 32 - 2, 0)
                self.tik_instance.data_move(ub_mask_2, self.mask_gm[move_offset // 8 + 16], 0, 32, 2, 32 - 2, 0)
                self.tik_instance.data_move(ub_mask_3, self.mask_gm[move_offset // 8 + 24], 0, 32, 2, 32 - 2, 0)
                for mask_idx in range(0, 64):
                    self.tik_instance.vec_sel(64, 0, ub_2[mask_idx*256], ub_mask_0[mask_idx*32], ub_2[mask_idx*256],
                                              self.dup_sel_0, 1, 8, 8, 0)
                    self.tik_instance.vec_sel(64, 0, ub_2[mask_idx*256 + 64], ub_mask_1[mask_idx*32],
                                              ub_2[mask_idx*256 + 64], self.dup_sel_0, 1, 8, 8, 0)
                    self.tik_instance.vec_sel(64, 0, ub_2[mask_idx*256 + 128], ub_mask_2[mask_idx*32],
                                              ub_2[mask_idx*256 + 128], self.dup_sel_0, 1, 8, 8, 0)
                    self.tik_instance.vec_sel(64, 0, ub_2[mask_idx*256 + 192], ub_mask_3[mask_idx*32],
                                              ub_2[mask_idx*256 + 192], self.dup_sel_0, 1, 8, 8, 0)
            else:
                self.tik_instance.vec_sel(64, 2, ub_2, ub_mask_0, ub_2, self.dup_sel_0, 128, 8, 8, 0)
                self.tik_instance.vec_sel(64, 2, ub_2[128*64], ub_mask_0[128*8], ub_2[128*64], self.dup_sel_0,
                                          128, 8, 8, 0)
        with self.tik_instance.else_scope():    
            # ub_1 store the mask of type float32
            ub_mask = self.tik_instance.Tensor("uint8", (h1_len, 32, 16), scope=tik.scope_ubuf, name="ub_mask")
            ub_mask_fp16 = self.tik_instance.Tensor("float16", (h1_len, 32, 16), scope=tik.scope_ubuf,
                                                    name="ub_mask_fp16")
            self.tik_instance.data_move(ub_mask[0], self.mask_gm[move_offset], 0, w1_len, 16, h1_len * 8 - 16, 0)
            self.tik_instance.vconv(Constant.VEC_MASK, "", ub_mask_fp16[0], ub_mask[0], h1_len * 4, 1, 1, 8, 4)
            self.tik_instance.vconv(Constant.VEC_MASK_FP32, "", ub_1[0], ub_mask_fp16[0], Constant.MAX_REPEAT,
                                    1, 1, 8, 4)
            self.tik_instance.vconv(Constant.VEC_MASK_FP32, "", ub_1[Constant.VEC_MASK_FP32 * Constant.MAX_REPEAT],
                                    ub_mask_fp16[Constant.VEC_MASK_FP32 * Constant.MAX_REPEAT], 1, 1, 1, 8, 4)
            self.tik_instance.vmul(Constant.VEC_MASK_FP32, ub_2[0], ub_2[0], ub_1[0], Constant.MAX_REPEAT,
                                   1, 1, 1, 8, 8, 8)
            self.tik_instance.vmul(Constant.VEC_MASK_FP32, ub_2[Constant.VEC_MASK_FP32 * Constant.MAX_REPEAT],
                                   ub_2[Constant.VEC_MASK_FP32 * Constant.MAX_REPEAT],
                                   ub_1[Constant.VEC_MASK_FP32 * Constant.MAX_REPEAT],
                                   1, 1, 1, 1, 8, 8, 8)

        # ub_2 -> y2_gm
        self.tik_instance.data_move(self.y2_gm[move_offset], ub_2[0], 0, w1_len, 64, 0, h1_len * 32 - 64)

    def axpy_with_softmax_and_drop_out_do_mask_w_small(self, core_index):
        self.line = 2 if self.ub_size > Constant.LINE_THRESHOLD else 1
        self.ranges = self.tik_instance.Scalar("int32", name="ranges", init_value=self.dim_w1 // self.line)
        offset = self.tik_instance.Scalar("int32", name="offset")
        core_offset = self.tik_instance.Scalar("int32", name="core_offset")
        batch_per_core_truth = self.tik_instance.Scalar("int32", name="batch_per_core_truth")

        with self.tik_instance.if_scope(core_index < self.batch_large_core_num):
            batch_per_core_truth.set_as(self.batch_per_core)
            core_offset.set_as(core_index * self. batch_per_core * self.ele_per_batch)
        with self.tik_instance.else_scope():
            batch_per_core_truth.set_as(self.batch_small_per_core)
            core_offset.set_as(self.batch_large_core_num * self.batch_per_core * self.ele_per_batch + \
                               (core_index - self.batch_large_core_num) * self.batch_small_per_core * \
                               self.ele_per_batch)

        with self.tik_instance.for_range(0, batch_per_core_truth) as index:
            with self.tik_instance.for_range(0, self.ranges) as i:
                offset.set_as(core_offset + index * self.ele_per_batch + i * self.line * Constant.BLOCK_SQUARE)
                if self.tensor_dtype == "float32":
                    self.small_fp32_compute(offset, i)
                else:
                    self.small_fp16_compute(offset, i)

    def small_fp16_compute(self, offset, i):
        shape = (self.dim_w1, self.line * Constant.BLOCK, Constant.SHAPE_W2)
        ub_1 = self.tik_instance.Tensor("float16", shape, scope=tik.scope_ubuf, name="ub_1")
        ub_2 = self.tik_instance.Tensor("float16", shape, scope=tik.scope_ubuf, name="ub_2")
        ub_cast = self.tik_instance.Tensor("float32", shape, scope=tik.scope_ubuf, name="ub_cast")
        ub_3 = self.tik_instance.Tensor("float32", shape, scope=tik.scope_ubuf, name="ub_3")
        ub_4 = self.tik_instance.Tensor("float32", shape, scope=tik.scope_ubuf, name="ub_4")

        ub_reducemax = self.tik_instance.Tensor("float16", (self.line * Constant.BLOCK,),
                                                scope=tik.scope_ubuf, name="ub_reducemax")
        ub_reduceadd = self.tik_instance.Tensor("float32", (self.line * Constant.BLOCK,),
                                                scope=tik.scope_ubuf, name="ub_reduceadd")
        ub_dup = self.tik_instance.Tensor("uint16", (Constant.VEC_DUMP_SHAPE,), scope=tik.scope_ubuf, name="ub_dup")
        ub_dup_fp32 = self.tik_instance.Tensor("float32", (32, 16), scope=tik.scope_ubuf, name="ub_dup_fp32")
        ub_broadcast = self.tik_instance.Tensor("uint16", (self.line * Constant.BLOCK_SQUARE,), 
                                                scope=tik.scope_ubuf, name="ub_broadcast")

        self.data_move_in(offset, ub_1, ub_2)
        alpha_fp16 = self.tik_instance.Scalar("float16", name="alpha_fp16")
        self.tik_instance.scalar_conv("", alpha_fp16, self.alpha)
        self.tik_instance.vec_axpy(Constant.VEC_MASK_FP16, ub_1, ub_2, alpha_fp16, self.w_dim * \
                                   self.line // (Constant.VEC_MASK_FP16 // Constant.FP16_NUM_PER_BLOCK), 8, 8)
        self.reduce_max_and_sub(ub_1, ub_2, ub_reducemax, ub_broadcast, ub_dup)
        self.conv_and_exp(ub_2, ub_cast, i)
        self.reduce_sum_and_div(ub_cast, ub_reduceadd, ub_3, ub_4, ub_dup_fp32, ub_1)
        self.mul_with_dropout(ub_2, ub_3, ub_4, offset)
        self.move_data_out(ub_1, ub_2, offset)
    
    def small_fp32_compute(self, offset, i):
        shape = (self.dim_w1, self.line * Constant.BLOCK, Constant.SHAPE_W2)
        ub_1 = self.tik_instance.Tensor("float32", shape, scope=tik.scope_ubuf, name="ub_1")
        ub_2 = self.tik_instance.Tensor("float32", shape, scope=tik.scope_ubuf, name="ub_2")
        ub_3 = self.tik_instance.Tensor("float32", shape, scope=tik.scope_ubuf, name="ub_3")
        ub_reducemax = self.tik_instance.Tensor("float32", (self.line * Constant.BLOCK,),
                                                scope=tik.scope_ubuf, name="ub_reducemax")
        ub_reduceadd = self.tik_instance.Tensor("float32", (self.line * Constant.BLOCK,),
                                                scope=tik.scope_ubuf, name="ub_reduceadd")
        ub_dup_fp32 = self.tik_instance.Tensor("float32", (32, 16), scope=tik.scope_ubuf, name="ub_dup_fp32")

        self.tik_instance.data_move(ub_1, self.x1_gm[offset], 0, self.dim_w1, Constant.BLOCK * self.line * 2,
                                    (self.dim_w1 - self.line) * Constant.BLOCK * 2, 0)
        self.tik_instance.data_move(ub_2, self.x2_gm[offset], 0, self.dim_w1, Constant.BLOCK * self.line * 2,
                                    (self.dim_w1 - self.line) * Constant.BLOCK * 2, 0)

        self.tik_instance.vec_axpy(Constant.VEC_MASK_FP32, ub_1, ub_2, self.alpha, self.w_dim * self.line * \
                                   Constant.BLOCK // Constant.VEC_MASK_FP32, 8, 8)
        self.reduce_max_and_sub_fp32(ub_1, ub_2, ub_reducemax)
        self.conv_and_exp_fp32(ub_2, i)
        self.reduce_sum_and_div_fp32(ub_2, ub_reduceadd, ub_3, ub_dup_fp32)
        self.tik_instance.data_move(self.y1_gm[offset], ub_2, 0, self.dim_w1, Constant.BLOCK * self.line * 2, 0,
                                    (self.dim_h1 - self.line) * Constant.BLOCK * 2)
        self.tik_instance.vmuls(Constant.VEC_MASK_FP32, ub_2, ub_2,
                                self.tik_instance.Scalar(init_value=1 / self.input_keep_prob, dtype="float32"),
                                self.line * Constant.BLOCK * self.w_dim // Constant.VEC_MASK_FP32, 1, 1, 8, 8)

        with self.tik_instance.if_scope(self.mask_is_bit == 1):
            ub_mask_0 = self.tik_instance.Tensor("uint8",
                                                 (self.dim_w1 * self.line * Constant.BLOCK * Constant.SHAPE_W2 // 8,),
                                                 scope=tik.scope_ubuf, name="ub_mask_0")
            self.tik_instance.data_move(ub_mask_0, self.mask_gm[offset // 8], 0, self.dim_w1, self.line,
                                        (self.dim_w1 - self.line), 0)
            if self.sel_mode == 0:
                ub_mask_1 = self.tik_instance.Tensor("uint8",
                            (self.dim_w1 * self.line * Constant.BLOCK * Constant.SHAPE_W2 // 8,),
                            scope=tik.scope_ubuf, name="ub_mask_1")
                ub_mask_2 = self.tik_instance.Tensor("uint8",
                            (self.dim_w1 * self.line * Constant.BLOCK * Constant.SHAPE_W2 // 8,),
                            scope=tik.scope_ubuf, name="ub_mask_2")
                ub_mask_3 = self.tik_instance.Tensor("uint8",
                            (self.dim_w1 * self.line * Constant.BLOCK * Constant.SHAPE_W2 // 8,),
                            scope=tik.scope_ubuf, name="ub_mask_3")
                self.tik_instance.data_move(ub_mask_1, self.mask_gm[offset // 8 + 8], 0, self.dim_w1, self.line,
                                           (self.dim_w1 - self.line), 0)
                self.tik_instance.data_move(ub_mask_2, self.mask_gm[offset // 8 + 16], 0, self.dim_w1, self.line,
                                            (self.dim_w1 - self.line), 0)
                self.tik_instance.data_move(ub_mask_3, self.mask_gm[offset // 8 + 24], 0, self.dim_w1, self.line,
                                            (self.dim_w1 - self.line), 0)
                with self.tik_instance.for_range(0, self.dim_w1 * self.line) as mask_idx:
                    self.tik_instance.vec_sel(64, 0, ub_1[mask_idx*256], ub_mask_0[mask_idx*32], ub_2[mask_idx*256],
                                              self.dup_sel_0, 1, 8, 8, 0)
                    self.tik_instance.vec_sel(64, 0, ub_1[mask_idx*256 + 64], ub_mask_1[mask_idx*32],
                                              ub_2[mask_idx*256 + 64], self.dup_sel_0, 1, 8, 8, 0)
                    self.tik_instance.vec_sel(64, 0, ub_1[mask_idx*256 + 128], ub_mask_2[mask_idx*32],
                                              ub_2[mask_idx*256 + 128], self.dup_sel_0, 1, 8, 8, 0)
                    self.tik_instance.vec_sel(64, 0, ub_1[mask_idx*256 + 192], ub_mask_3[mask_idx*32],
                                              ub_2[mask_idx*256 + 192], self.dup_sel_0, 1, 8, 8, 0)
            else:
                self.tik_instance.vec_sel(64, 1, ub_1, ub_mask_0, ub_2, 0, self.dim_w1 * self.line * 4, 8, 8, 0)
        with self.tik_instance.else_scope():
            ub_mask = self.tik_instance.Tensor("uint8", shape, scope=tik.scope_ubuf, name="ub_mask")
            ub_mask_fp16 = self.tik_instance.Tensor("float16", shape, scope=tik.scope_ubuf, name="ub_mask_fp16")
            self.tik_instance.data_move(ub_mask, self.mask_gm[offset], 0, self.dim_w1,
                                        Constant.BLOCK * self.line // Constant.LEN,
                                        (self.dim_w1 - self.line) * Constant.BLOCK // Constant.LEN, 0)
            self.tik_instance.vconv(Constant.VEC_MASK, "", ub_mask_fp16, ub_mask,
                                    self.line * Constant.BLOCK * self.w_dim // Constant.VEC_MASK,
                                    1, 1, 8, 4)
            self.tik_instance.vconv(Constant.VEC_MASK_FP32, "", ub_1, ub_mask_fp16, self.line * Constant.BLOCK * \
                                    self.w_dim // Constant.VEC_MASK_FP32, 1, 1, 8, 4)
            self.tik_instance.vmul(Constant.VEC_MASK_FP32, ub_1, ub_1, ub_2, self.line * Constant.BLOCK * \
                                self.w_dim // Constant.VEC_MASK_FP32, 1, 1, 1, 8, 8, 8)
        self.tik_instance.data_move(self.y2_gm[offset], ub_1, 0, self.dim_w1, Constant.BLOCK * self.line * 2, 0,
                                        (self.dim_h1 - self.line) * Constant.BLOCK * 2)

    def get_max_ele(self, ub_1, ub_2, ub_reducemax, h1_len):
        self.tik_instance.vmax(Constant.VEC_MASK, ub_2[0], ub_1[0], ub_1[h1_len * 32 * 8], h1_len * 2, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmax(Constant.VEC_MASK, ub_2[0], ub_2[0], ub_2[h1_len * 32 * 4], h1_len, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmax(Constant.VEC_MASK, ub_2[0], ub_2[0], ub_2[h1_len * 32 * 2], 16, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmax(Constant.VEC_MASK, ub_2[0], ub_2[0], ub_2[h1_len * 32], 8, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmax(Constant.VEC_MASK, ub_2[0], ub_2[0], ub_2[h1_len * 16], 4, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vcgmax(Constant.VEC_MASK, ub_reducemax[0], ub_2[0], 4, 1, 1, 8)
    
    def get_max_ele_fp32(self, ub_1, ub_2, ub_reducemax, h1_len):
        self.tik_instance.vmax(Constant.VEC_MASK_FP32, ub_2[0], ub_1[0], ub_1[h1_len * 32 * 8], h1_len * 4,
                               1, 1, 1, 8, 8, 8)
        self.tik_instance.vmax(Constant.VEC_MASK_FP32, ub_2[0], ub_2[0], ub_2[h1_len * 32 * 4], h1_len * 2,
                               1, 1, 1, 8, 8, 8)
        self.tik_instance.vmax(Constant.VEC_MASK_FP32, ub_2[0], ub_2[0], ub_2[h1_len * 32 * 2], h1_len,
                               1, 1, 1, 8, 8, 8)
        self.tik_instance.vmax(Constant.VEC_MASK_FP32, ub_2[0], ub_2[0], ub_2[h1_len * 32], 16, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmax(Constant.VEC_MASK_FP32, ub_2[0], ub_2[0], ub_2[h1_len * 16], 8, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmax(8, ub_2[0], ub_2[0], ub_2[8], h1_len, 1, 1, 1, 1, 2, 2)
        self.tik_instance.vcgmax(Constant.VEC_MASK_FP32, ub_reducemax[0], ub_2[0], 4, 1, 2, 8)

    def broadcast(self, ub_dup, ub_need_broadcast, ub_broadcast):
        self.tik_instance.vector_dup(Constant.VEC_MASK, ub_dup[0], self.dup_value, 1, 1, 8)
        ub_need_broadcast_int16 = ub_need_broadcast.reinterpret_cast_to("uint16")
        self.tik_instance.vor(16, ub_broadcast[0], ub_need_broadcast_int16[0], ub_dup[0], 16, 1, 1, 0, 1, 0, 0)
        self.tik_instance.vor(16, ub_broadcast[256], ub_need_broadcast_int16[16], ub_dup[0], 16, 1, 1, 0, 1, 0, 0)

        self.tik_instance.vtranspose(ub_broadcast[0], ub_broadcast[0])
        self.tik_instance.vtranspose(ub_broadcast[256], ub_broadcast[256])
        ub_broadcast_fp16 = ub_broadcast.reinterpret_cast_to("float16")
        return ub_broadcast_fp16

    def exp(self, ub_2, ub_3, ub_cast):
        self.tik_instance.vconv(Constant.VEC_MASK_FP32, "", ub_cast[0], ub_2[0], Constant.MAX_REPEAT, 1, 1, 8, 4)
        self.tik_instance.vconv(Constant.VEC_MASK_FP32, "", ub_cast[Constant.VEC_MASK_FP32 * Constant.MAX_REPEAT],
                                ub_2[Constant.VEC_MASK_FP32 * Constant.MAX_REPEAT], 1, 1, 1, 8, 4)
        self.tik_instance.vexp(Constant.VEC_MASK_FP32, ub_cast[0], ub_cast[0], Constant.MAX_REPEAT, 1, 1, 8, 8)
        self.tik_instance.vexp(Constant.VEC_MASK_FP32, ub_cast[Constant.VEC_MASK_FP32 * Constant.MAX_REPEAT],
                               ub_cast[Constant.VEC_MASK_FP32 * Constant.MAX_REPEAT], 1, 1, 1, 8, 8)
        self.tik_instance.vconv(Constant.VEC_MASK_FP32, "", ub_3[0], ub_cast[0], Constant.MAX_REPEAT, 1, 1, 4, 8)
        self.tik_instance.vconv(Constant.VEC_MASK_FP32, "", ub_3[Constant.VEC_MASK_FP32 * Constant.MAX_REPEAT],
                                ub_cast[Constant.VEC_MASK_FP32 * Constant.MAX_REPEAT], 1, 1, 1, 4, 8)
 
    def sum_exp(self, ub_cast, ub_reduceadd, h1_len):
        self.tik_instance.vadd(Constant.VEC_MASK_FP32, ub_cast[0], ub_cast[0], ub_cast[h1_len * 32 * 8],
                               h1_len * 4, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(Constant.VEC_MASK_FP32, ub_cast[0], ub_cast[0], ub_cast[h1_len * 32 * 4],
                               h1_len * 2, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(Constant.VEC_MASK_FP32, ub_cast[0], ub_cast[0], ub_cast[h1_len * 32 * 2],
                               32, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(Constant.VEC_MASK_FP32, ub_cast[0], ub_cast[0], ub_cast[h1_len * 32],
                               16, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(Constant.VEC_MASK_FP32, ub_cast[0], ub_cast[0], ub_cast[h1_len * 16],
                               8, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vcadd(16, ub_reduceadd[0], ub_cast[0], 32, 1, 1, 2)
    
    def sum_exp_fp32(self, ub_2, ub_1, ub_reduceadd, h1_len):
        self.tik_instance.vadd(Constant.VEC_MASK_FP32, ub_1[0], ub_2[0], ub_2[h1_len * 32 * 8],
                               h1_len * 4, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(Constant.VEC_MASK_FP32, ub_1[0], ub_1[0], ub_1[h1_len * 32 * 4],
                               h1_len * 2, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(Constant.VEC_MASK_FP32, ub_1[0], ub_1[0], ub_1[h1_len * 32 * 2],
                               32, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(Constant.VEC_MASK_FP32, ub_1[0], ub_1[0], ub_1[h1_len * 32],
                               16, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(Constant.VEC_MASK_FP32, ub_1[0], ub_1[0], ub_1[h1_len * 16],
                               8, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vcadd(16, ub_reduceadd[0], ub_1[0], 32, 1, 1, 2)
    
    def data_move_in(self, offset, ub_1, ub_2):
        """
        data_move_in
        """
        self.tik_instance.data_move(ub_1, self.x1_gm[offset], 0, self.dim_w1, Constant.BLOCK * self.line,
                                    (self.dim_w1 - self.line) * Constant.BLOCK, 0)
        self.tik_instance.data_move(ub_2, self.x2_gm[offset], 0, self.dim_w1, Constant.BLOCK * self.line,
                                    (self.dim_w1 - self.line) * Constant.BLOCK, 0)

    def reduce_max_and_sub(self, ub_1, ub_2, ub_reducemax, ub_broadcast, ub_dup):
        """
        reduce_max_and_sub
        """
        time = self.tik_instance.Scalar("int32", name='time', init_value=Constant.LEN)
        self.tik_instance.vmax(Constant.VEC_MASK_FP16, ub_2, ub_1,
                               ub_1[self.w_dim * Constant.BLOCK * self.line // time],
                               self.w_dim * Constant.BLOCK * self.line // time // Constant.VEC_MASK_FP16,
                               1, 1, 1, 8, 8, 8)
        with self.tik_instance.for_range(1, self.cnt) as j:
            time.set_as(time * Constant.LEN)
            self.tik_instance.vmax(Constant.VEC_MASK_FP16, ub_2, ub_2,
                                   ub_2[self.w_dim * Constant.BLOCK * self.line // time],
                                   self.w_dim * Constant.BLOCK * self.line // time // Constant.VEC_MASK_FP16,
                                   1, 1, 1, 8, 8, 8)
        with self.tik_instance.if_scope(self.remain > 0):
            with self.tik_instance.for_range(1, self.remain + 1) as j:
                self.tik_instance.vmax(Constant.VEC_MASK_FP16,
                                       ub_2[Constant.BLOCK_SQUARE * self.line * (self.remain - j)],
                                       ub_2[Constant.BLOCK_SQUARE * self.line * (self.remain - j)],
                                       ub_2[Constant.BLOCK_SQUARE * self.line * (self.remain - j + 1)],
                                       Constant.BLOCK_SQUARE * self.line // Constant.VEC_MASK_FP16,
                                       1, 1, 1, 8, 8, 8)
        self.tik_instance.vcgmax(Constant.VEC_MASK_FP16, ub_reducemax, ub_2, Constant.LEN * self.line, 1, 1, 8)

        self.tik_instance.vector_dup(Constant.VEC_DUMP_SHAPE, ub_dup, self.dup_value, 1, 1, 8)
        ub_reducemax_int16 = ub_reducemax.reinterpret_cast_to("uint16")
        with self.tik_instance.for_range(0, self.line) as j:
            self.tik_instance.vor(Constant.BLOCK, ub_broadcast[Constant.BLOCK_SQUARE * j], 
                            ub_reducemax_int16[Constant.BLOCK * j], ub_dup, Constant.BLOCK, 1, 1, 0, 1, 0, 0)
        with self.tik_instance.for_range(0, self.line) as j:
            self.tik_instance.vtranspose(ub_broadcast[Constant.BLOCK_SQUARE * j],
                                         ub_broadcast[Constant.BLOCK_SQUARE * j])
        ub_broadcast_fp16 = ub_broadcast.reinterpret_cast_to("float16")

        with self.tik_instance.for_range(0, self.line * Constant.BLOCK_SQUARE // Constant.VEC_MASK) as idx:
            self.tik_instance.vsub(Constant.VEC_MASK, ub_2[idx * Constant.VEC_MASK], ub_1[idx * Constant.VEC_MASK],
                                   ub_broadcast_fp16[idx * Constant.VEC_MASK], self.dim_w1, 1, 1, 1,
                                   self.line * Constant.BLOCK, self.line * Constant.BLOCK, 0)

    def reduce_max_and_sub_fp32(self, ub_1, ub_2, ub_reducemax):
        """
        reduce_max_and_sub_fp32
        """
        time = self.tik_instance.Scalar("int32", name='time', init_value=Constant.LEN)
        self.tik_instance.vmax(Constant.VEC_MASK_FP32, ub_2, ub_1,
                               ub_1[self.w_dim * Constant.BLOCK * self.line // time],
                               self.w_dim * Constant.BLOCK * self.line // time // Constant.VEC_MASK_FP32,
                               1, 1, 1, 8, 8, 8)
        with self.tik_instance.for_range(1, self.cnt) as j:
            time.set_as(time * Constant.LEN)
            self.tik_instance.vmax(Constant.VEC_MASK_FP32, ub_2, ub_2,
                                   ub_2[self.w_dim * Constant.BLOCK * self.line // time],
                                   self.w_dim * Constant.BLOCK * self.line // time // Constant.VEC_MASK_FP32,
                                   1, 1, 1, 8, 8, 8)
        with self.tik_instance.if_scope(self.remain > 0):
            with self.tik_instance.for_range(1, self.remain + 1) as j:
                self.tik_instance.vmax(Constant.VEC_MASK_FP32,
                                       ub_2[Constant.BLOCK_SQUARE * self.line * (self.remain - j)],
                                       ub_2[Constant.BLOCK_SQUARE * self.line * (self.remain - j)],
                                       ub_2[Constant.BLOCK_SQUARE * self.line * (self.remain - j + 1)],
                                       Constant.BLOCK_SQUARE * self.line // Constant.VEC_MASK_FP32,
                                       1, 1, 1, 8, 8, 8)
        self.tik_instance.vmax(Constant.FP32_NUM_PER_BLOCK, ub_2[0], ub_2[0], ub_2[Constant.FP32_NUM_PER_BLOCK],
                               self.line * Constant.BLOCK, 1, 1, 1, 1, 2, 2)
        self.tik_instance.vcgmax(Constant.VEC_MASK_FP32, ub_reducemax, ub_2, Constant.LEN * self.line, 1, 2, 8)

        # x - x_max
        max_value = self.tik_instance.Scalar("float32", name="max_value")
        with self.tik_instance.for_range(0, self.line * Constant.BLOCK) as idx:
            max_value.set_as(ub_reducemax[idx])
            self.tik_instance.vadds(Constant.SHAPE_W2, ub_2[idx * 16], ub_1[idx * 16], max_value * (-1), self.dim_w1,
                                    1, 1, self.line * Constant.BLOCK * 2, self.line * Constant.BLOCK * 2)

    def conv_and_exp(self, ub_2, ub_cast, i):
        """
        conv_and_exp
        """
        repeat_time = self.tik_instance.Scalar("int32", name='repeat_time', init_value=self.line * Constant.BLOCK * \
                                               self.w_dim // Constant.VEC_MASK_FP32)
        cnt = self.tik_instance.Scalar("int32", name='cnt', init_value=repeat_time // Constant.MAX_REPEAT)
        remain = self.tik_instance.Scalar("int32", name='remain', init_value=repeat_time % Constant.MAX_REPEAT)
        with self.tik_instance.if_scope(cnt > 0):
            self.tik_instance.vconv(Constant.VEC_MASK_FP32, "",
                                    ub_cast[Constant.MAX_REPEAT * Constant.VEC_MASK_FP32 * i],
                                    ub_2[Constant.MAX_REPEAT * Constant.VEC_MASK_FP32 * i],
                                    Constant.MAX_REPEAT, 1, 1, 8, 4)
            self.tik_instance.vconv(Constant.VEC_MASK_FP32, "",
                                    ub_cast[Constant.MAX_REPEAT * Constant.VEC_MASK_FP32 * cnt],
                                    ub_2[Constant.MAX_REPEAT * Constant.VEC_MASK_FP32 * cnt],
                                    remain, 1, 1, 8, 4)

            self.tik_instance.vexp(Constant.VEC_MASK_FP32, ub_cast[Constant.MAX_REPEAT * Constant.VEC_MASK_FP32 * i],
                                   ub_cast[Constant.MAX_REPEAT * Constant.VEC_MASK_FP32 * i], Constant.MAX_REPEAT,
                                   1, 1, 8, 8)
            self.tik_instance.vexp(Constant.VEC_MASK_FP32, ub_cast[Constant.MAX_REPEAT * Constant.VEC_MASK_FP32 * cnt],
                                   ub_cast[Constant.MAX_REPEAT * Constant.VEC_MASK_FP32 * cnt], remain, 1, 1, 8, 8)
        with self.tik_instance.else_scope():
            self.tik_instance.vconv(Constant.VEC_MASK_FP32, "", ub_cast, ub_2, remain, 1, 1, 8, 4)
            self.tik_instance.vexp(Constant.VEC_MASK_FP32, ub_cast, ub_cast, remain, 1, 1, 8, 8)
        
    def conv_and_exp_fp32(self, ub_2, i):
        """
        conv_and_exp_fp32
        """
        repeat_time = self.tik_instance.Scalar("int32", name='repeat_time', init_value=self.line * Constant.BLOCK * \
                                               self.w_dim // Constant.VEC_MASK_FP32)
        cnt = self.tik_instance.Scalar("int32", name='cnt', init_value=repeat_time // Constant.MAX_REPEAT)
        remain = self.tik_instance.Scalar("int32", name='remain', init_value=repeat_time % Constant.MAX_REPEAT)
        with self.tik_instance.if_scope(cnt > 0):
            self.tik_instance.vexp(Constant.VEC_MASK_FP32, ub_2[Constant.MAX_REPEAT * Constant.VEC_MASK_FP32 * i],
                                   ub_2[Constant.MAX_REPEAT * Constant.VEC_MASK_FP32 * i], Constant.MAX_REPEAT,
                                   1, 1, 8, 8)
            self.tik_instance.vexp(Constant.VEC_MASK_FP32, ub_2[Constant.MAX_REPEAT * Constant.VEC_MASK_FP32 * cnt],
                                   ub_2[Constant.MAX_REPEAT * Constant.VEC_MASK_FP32 * cnt], remain, 1, 1, 8, 8)
        with self.tik_instance.else_scope():
            self.tik_instance.vexp(Constant.VEC_MASK_FP32, ub_2, ub_2, remain, 1, 1, 8, 8)
  
    def reduce_sum_and_div(self, ub_cast, ub_reduceadd, ub_3, ub_4, ub_dup_fp32, ub_1):
        """
        reduce_sum_and_div
        """
        self.tik_instance.vmuls(Constant.VEC_MASK_FP32, ub_3, ub_cast,
                                self.tik_instance.Scalar(init_value=1, dtype="float32"),
                                self.line * Constant.BLOCK * self.w_dim // Constant.VEC_MASK_FP32,
                                1, 1, 8, 8)

        time = self.tik_instance.Scalar("int32", name='time', init_value=1)
        # reduce_add
        with self.tik_instance.for_range(0, self.cnt) as j:
            time.set_as(time * Constant.LEN)
            self.tik_instance.vadd(Constant.VEC_MASK_FP32, ub_cast, ub_cast,
                                   ub_cast[self.w_dim * Constant.BLOCK * self.line // time],
                                   self.w_dim * Constant.BLOCK * self.line // time // Constant.VEC_MASK_FP32, 
                                   1, 1, 1, 8, 8, 8)
        with self.tik_instance.if_scope(self.remain > 0):
            with self.tik_instance.for_range(1, self.remain + 1) as j:
                self.tik_instance.vadd(Constant.VEC_MASK_FP32,
                                       ub_cast[Constant.BLOCK_SQUARE * self.line * (self.remain - j)],
                                       ub_cast[Constant.BLOCK_SQUARE * self.line * (self.remain - j)],
                                       ub_cast[Constant.BLOCK_SQUARE * self.line * (self.remain - j + 1)],
                                       Constant.BLOCK_SQUARE * self.line // Constant.VEC_MASK_FP32,
                                       1, 1, 1, 8, 8, 8)
        self.tik_instance.vcadd(Constant.BLOCK, ub_reduceadd, ub_cast, Constant.BLOCK * self.line, 1, 1, 2)

        # vrec
        self.tik_instance.vrec(self.line * Constant.BLOCK, ub_reduceadd, ub_reduceadd, 1, 1, 1, 0, 0)

        with self.tik_instance.for_range(0, self.line * Constant.SHAPE_H2 / 8) as j:
            with self.tik_instance.for_range(0, 8) as k:
                self.tik_instance.vector_dup(16, ub_dup_fp32[j * 128 + 16 * k],
                                             self.tik_instance.Scalar(init_value=ub_reduceadd[j * 8 + k],
                                             dtype="float32"), 1, 1, 8)

        with self.tik_instance.for_range(0, self.line * Constant.BLOCK_SQUARE // Constant.VEC_MASK_FP32) as idx:
            self.tik_instance.vmul(Constant.VEC_MASK_FP32, ub_4[idx * Constant.VEC_MASK_FP32],
                                   ub_3[idx * Constant.VEC_MASK_FP32], ub_dup_fp32[idx * Constant.VEC_MASK_FP32],
                                   self.dim_w1, 1, 1, 1, self.line * Constant.BLOCK * 2,
                                   self.line * Constant.BLOCK * 2, 0)

        self.tik_instance.vconv(Constant.VEC_MASK_FP32, "", ub_1, ub_4,
                                self.w_dim * self.line // (Constant.VEC_MASK_FP32 // Constant.FP16_NUM_PER_BLOCK),
                                1, 1, 4, 8)
    
    def reduce_sum_and_div_fp32(self, ub_2, ub_reduceadd, ub_3, ub_dup_fp32):
        """
        reduce_sum_and_div_fp32
        """
        self.tik_instance.vmuls(Constant.VEC_MASK_FP32, ub_3, ub_2,
                                self.tik_instance.Scalar(init_value=1, dtype="float32"),
                                self.line * Constant.BLOCK * self.w_dim // Constant.VEC_MASK_FP32,
                                1, 1, 8, 8)

        time = self.tik_instance.Scalar("int32", name='time', init_value=1)
        # reduce_add
        with self.tik_instance.for_range(0, self.cnt) as j:
            time.set_as(time * Constant.LEN)
            self.tik_instance.vadd(Constant.VEC_MASK_FP32, ub_2, ub_2,
                                   ub_2[self.w_dim * Constant.BLOCK * self.line // time],
                                   self.w_dim * Constant.BLOCK * self.line // time // Constant.VEC_MASK_FP32, 
                                   1, 1, 1, 8, 8, 8)
        with self.tik_instance.if_scope(self.remain > 0):
            with self.tik_instance.for_range(1, self.remain + 1) as j:
                self.tik_instance.vadd(Constant.VEC_MASK_FP32,
                                       ub_2[Constant.BLOCK_SQUARE * self.line * (self.remain - j)],
                                       ub_2[Constant.BLOCK_SQUARE * self.line * (self.remain - j)],
                                       ub_2[Constant.BLOCK_SQUARE * self.line * (self.remain - j + 1)],
                                       Constant.BLOCK_SQUARE * self.line // Constant.VEC_MASK_FP32,
                                       1, 1, 1, 8, 8, 8)
        self.tik_instance.vcadd(Constant.BLOCK, ub_reduceadd, ub_2, Constant.BLOCK * self.line, 1, 1, 2)

        with self.tik_instance.new_stmt_scope():
            mask_len = self.line * Constant.BLOCK
            wk_size_unit = ((mask_len + Constant.FP32_NUM_PER_BLOCK - 1) // Constant.FP32_NUM_PER_BLOCK) * \
                           Constant.FP32_NUM_PER_BLOCK
            work_tensor_ub = self.tik_instance.Tensor("float32", (2*wk_size_unit,), name="work_tensor_ub", \
                                                      scope=tik.scope_ubuf)
            ub_reduceadd_res = self.tik_instance.Tensor("float32", (2*wk_size_unit,), name="work_tensor_ub", \
                                                        scope=tik.scope_ubuf)
            self.tik_instance.vec_rec_high_preci(self.line * Constant.BLOCK, ub_reduceadd_res, ub_reduceadd, \
                                                 work_tensor_ub[0:], 1, 8, 8)

            with self.tik_instance.for_range(0, self.line * Constant.SHAPE_H2 / 8) as j:
                with self.tik_instance.for_range(0, 8) as k:
                    self.tik_instance.vector_dup(16, ub_dup_fp32[j * 128 + 16 * k],
                                                self.tik_instance.Scalar(init_value=ub_reduceadd_res[j * 8 + k],
                                                dtype="float32"), 1, 1, 8)

        with self.tik_instance.for_range(0, self.line * Constant.BLOCK_SQUARE // Constant.VEC_MASK_FP32) as idx:
            self.tik_instance.vmul(Constant.VEC_MASK_FP32, ub_2[idx * Constant.VEC_MASK_FP32],
                                   ub_3[idx * Constant.VEC_MASK_FP32], ub_dup_fp32[idx * Constant.VEC_MASK_FP32],
                                   self.dim_w1, 1, 1, 1, self.line * Constant.BLOCK * 2,
                                   self.line * Constant.BLOCK * 2, 0)
 
    def mul_with_dropout(self, ub_2, ub_3, ub_4, offset):
        """
        mul_with_dropout
        """
        # vmuls and vmul
        self.tik_instance.vmuls(Constant.VEC_MASK_FP32, ub_3, ub_4,
                                self.tik_instance.Scalar(init_value=1 / self.input_keep_prob, dtype="float32"),
                                self.line * Constant.BLOCK * self.w_dim // Constant.VEC_MASK_FP32, 1, 1, 8, 8)
        self.tik_instance.vconv(Constant.VEC_MASK_FP32, "", ub_2, ub_3,
                                self.w_dim * self.line // (Constant.VEC_MASK_FP32 // Constant.FP16_NUM_PER_BLOCK),
                                1, 1, 4, 8)
        with self.tik_instance.if_scope(self.mask_is_bit == 1):
            ub_mask_0 = self.tik_instance.Tensor("uint8",
                                                 (self.dim_w1 * self.line * Constant.BLOCK * Constant.SHAPE_W2 // 8,),
                                                 scope=tik.scope_ubuf, name="ub_mask_0")
            self.tik_instance.data_move(ub_mask_0, self.mask_gm[offset // 8], 0, self.dim_w1, self.line,
                                        (self.dim_w1 - self.line), 0)
            if self.sel_mode == 0:
                ub_mask_1 = self.tik_instance.Tensor("uint8",
                            (self.dim_w1 * self.line * Constant.BLOCK * Constant.SHAPE_W2 // 8,),
                            scope=tik.scope_ubuf, name="ub_mask_1")
                self.tik_instance.data_move(ub_mask_1, self.mask_gm[offset // 8 + 16], 0, self.dim_w1, self.line,
                                            (self.dim_w1 - self.line), 0)
                with self.tik_instance.for_range(0, self.dim_w1 * self.line) as mask_idx:
                    self.tik_instance.vec_sel(128, 0, ub_2[mask_idx*256], ub_mask_0[mask_idx*32], ub_2[mask_idx*256],
                                              self.dup_sel_0, 1, 8, 8, 0)
                    self.tik_instance.vec_sel(128, 0, ub_2[mask_idx*256 + 128], ub_mask_1[mask_idx*32],
                                              ub_2[mask_idx*256 + 128], self.dup_sel_0, 1, 8, 8, 0)
            else:
                self.tik_instance.vec_sel(128, 1, ub_2, ub_mask_0, ub_2, 0, self.dim_w1 * self.line * 2, 8, 8, 0)
        with self.tik_instance.else_scope():
            shape = (self.dim_w1, self.line * Constant.BLOCK, Constant.SHAPE_W2)
            ub_mask = self.tik_instance.Tensor("uint8", shape, scope=tik.scope_ubuf, name="ub_mask")
            ub_mask_fp16 = self.tik_instance.Tensor("float16", shape, scope=tik.scope_ubuf, name="ub_mask_fp16")
            self.tik_instance.data_move(ub_mask, self.mask_gm[offset], 0, self.dim_w1,
                                        Constant.BLOCK * self.line // Constant.LEN,
                                        (self.dim_w1 - self.line) * Constant.BLOCK // Constant.LEN, 0)
            self.tik_instance.vconv(Constant.VEC_MASK, "", ub_mask_fp16, ub_mask,
                                    self.line * Constant.BLOCK * self.w_dim // Constant.VEC_MASK,
                                    1, 1, 8, 4)
            self.tik_instance.vmul(Constant.VEC_MASK, ub_2, ub_mask_fp16, ub_2, 
                                   self.line * Constant.BLOCK * self.w_dim // Constant.VEC_MASK,
                                   1, 1, 1, 8, 8, 8)
    
    def move_data_out(self, ub_1, ub_2, offset):
        """
        move_data_out
        """
        self.tik_instance.data_move(self.y1_gm[offset], ub_1, 0, self.dim_w1, Constant.BLOCK * self.line, 0,
                                    (self.dim_h1 - self.line) * Constant.BLOCK)
        self.tik_instance.data_move(self.y2_gm[offset], ub_2, 0, self.dim_w1, Constant.BLOCK * self.line, 0,
                                    (self.dim_h1 - self.line) * Constant.BLOCK)


# 'pylint: disable=unused-argument,too-many-arguments
# 'pylint: disable=too-many-locals,too-many-statements
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_LIST_INT,
                            para_check.KERNEL_NAME)
def axpy_with_softmax_and_drop_out_do_mask(x1, x2, mask, y1, y2, alpha, input_keep_prob, axis=-1,
                                           kernel_name="axpy_with_softmax_and_drop_out_do_mask"):
    """
    calculating data

    Parameters
    ----------
    x1 : dict
        input x1 shape and dtype of input
    x2 : dict
        input x2 shape and dtype of input
    mask : dict
        input mask shape and dtype of input
    y1 : dict
        output y1 shape and dtype, should be same shape and type as input
    y2 : dict
        output y2 shape and dtype, should be same shape and type as input
    alpha : int
        A attribute used to scale tensor. The type is float
    input_keep_prob : int
        A attribute used to judge which units should be keep
    axis : int
        A list of int. The dimension softmax would be performed on. default is -1
    kernel_name : str
        kernel name, default value is "axpy_with_softmax_and_drop_out_do_mask"

    Returns
    -------
    None
    """
    op_instance = AxpyWithSoftmaxAndDropOutDoMask(x1, x2, mask, y1, y2, alpha, input_keep_prob, axis=-1,
                                                  kernel_name=kernel_name)
    tik_instance = op_instance.axpy_with_softmax_and_drop_out_do_mask_compute()
    return tik_instance