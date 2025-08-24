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
ldpc
"""

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform as cce


class Constant:
    """
    The class for constant
    """
    RESERVE_SIZE = 16 * 1024
    BLOCK = 16


class LDPC:
    """
    Function: store LDPC parameters  and compute LDPC
    """
    def __init__(self, x, indices, y_en, y_cn, lmn0, kernel_name="ldpc"):
        """
        init the LDPC parameters
        """
        self.dtype_x = x.get("dtype")
        self.x_shape = x["shape"]
        self.indices_shape = indices["shape"]
        self.dtype_indices = indices.get("dtype")
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.kernel_name = kernel_name
        block_bite_size = 32
        # Get the size of UB space in Bytes
        self.ub_size_bytes = cce.get_soc_spec(cce.UB_SIZE)
        self.aicore_num = cce.get_soc_spec(cce.CORE_NUM)
        self.dtype_bytes_size_x = cce.get_bit_len(self.dtype_x) // Constant.BLOCK
        self.x_each_block = block_bite_size // self.dtype_bytes_size_x
        self.dtype_bytes_size_indices = cce.get_bit_len(self.dtype_indices) // Constant.BLOCK
        self.indices_each_block = block_bite_size // self.dtype_bytes_size_indices

        self.vector_mask_max_x = Constant.BLOCK * self.x_each_block
        self.ele_not_last_core = self.tik_instance.Scalar(self.dtype_indices, name='ele_not_last_core')
        self.ele_last_core = self.tik_instance.Scalar(self.dtype_indices, name='ele_last_core')
        self.ranges = self.tik_instance.Scalar("int32", name='ranges')
        self.sync_workspace = self.tik_instance.Tensor('int64', (4 * self.aicore_num,), tik.scope_gm,
                                                       'sync_workspace', is_workspace=True, is_atomic_add=True)
        self.sync_workspace1 = self.tik_instance.Tensor('int64', (4 * self.aicore_num,), tik.scope_gm,
                                                       'sync_workspace1', is_workspace=True, is_atomic_add=True)
        self.x = None
        self.indices = None
        self.y_en = None
        self.y_cn =  None
        self.lmn0 =  None

    @staticmethod
    def get_dtype_size(dtype):
        """
        get byte size of dtype
        """
        dtype_dict = {"float32": 4, "uint8": 1, "int32": 4, "float16": 2}
        return dtype_dict.get(dtype)

    def ldpc_compute(self):
        """
        compute ldpc
        """
        self.gm_for_data()
        self.cal_en()
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[self.x, self.indices],
                                   outputs=[self.y_en, self.y_cn, self.lmn0])
        return self.tik_instance

    def gm_for_data(self):
        """
        gm_for_data
        """
        self.x = self.tik_instance.Tensor(self.dtype_x, self.x_shape, name="x", scope=tik.scope_gm)
        self.indices = self.tik_instance.Tensor(self.dtype_indices, self.indices_shape, name="indices",
                                                scope=tik.scope_gm)
        self.y_en = self.tik_instance.Tensor(self.dtype_x, self.x_shape, name="y_en", scope=tik.scope_gm)
        self.y_cn = self.tik_instance.Tensor("uint8", (128, 1280), name="y_cn", scope=tik.scope_gm)
        lmn0_size = 4718592
        self.lmn0 = self.tik_instance.Tensor(self.dtype_x,
                                               [lmn0_size, ],
                                               name="lmn0",
                                               scope=tik.scope_gm, is_atomic_add=True)

    def cal_en(self):
        """
        cal_en
        """
        size = 512
        ele_per_core = size // self.aicore_num
        core_used = size // ele_per_core
        with self.tik_instance.for_range(0, core_used, block_num=core_used) as core_index:
            self.cal_en_and_cn_core(core_index)


    def ub_for_data(self):
        """
        ub_for_data
        """
        tik_inst = self.tik_instance
        shape_index = (16, 6)
        shape_x = (16, 6, 128)
        shape_middle = (16, 128)
        shape_uint = (128,)
        index_ub = tik_inst.Tensor("int32", shape_index, scope=tik.scope_ubuf, name="index_ub")
        temp_ub = tik_inst.Tensor("float16", shape_x, scope=tik.scope_ubuf, name="temp_ub")
        lmn0_ub = tik_inst.Tensor("float16", shape_x, scope=tik.scope_ubuf, name="lmn0_ub")
        negative_one_ub = tik_inst.Tensor("float16", shape_uint, scope=tik.scope_ubuf, name="negative_one_ub")
        positive_one_ub = tik_inst.Tensor("float16", shape_uint, scope=tik.scope_ubuf, name="positive_one_ub")
        zero_ub = tik_inst.Tensor("float16", shape_uint, scope=tik.scope_ubuf, name="positive_one_ub")
        mid_mul_ub = tik_inst.Tensor("float16", shape_middle, tik.scope_ubuf, "zero_ub")
        abs_tmp_ub = tik_inst.Tensor("float16", shape_x, tik.scope_ubuf, "abs_tmp_ub")
        cmp_tensor_ub = tik_inst.Tensor("float16", shape_x, tik.scope_ubuf, "cmp_tensor_ub")
        new_mul_tensor = tik_inst.Tensor("float16", shape_x, tik.scope_ubuf, "new_mul_tensor")
        min_res_tensor = tik_inst.Tensor("float16", shape_x, tik.scope_ubuf, "min_res_tensor")
        lis = [index_ub, temp_ub, lmn0_ub, negative_one_ub, positive_one_ub, zero_ub, mid_mul_ub, abs_tmp_ub,
               cmp_tensor_ub, new_mul_tensor, min_res_tensor]
        return lis


    def sign_fun_over_four(self, mid_mul_ub, cmp_tensor_ub, new_mul_tensor):
        """
        sign_fun_over_four
        """
        ranges = 24
        tik_instance = self.tik_instance
        tik_instance.vec_mul(128, mid_mul_ub, cmp_tensor_ub[1 * 128], cmp_tensor_ub[2 * 128], ranges // 6, 8, 48, 48)
        tik_instance.vec_mul(128, mid_mul_ub, cmp_tensor_ub[3 * 128], mid_mul_ub, ranges // 6, 8, 48, 8)
        tik_instance.vec_mul(128, mid_mul_ub, cmp_tensor_ub[4 * 128], mid_mul_ub, ranges // 6, 8, 48, 8)
        tik_instance.vec_mul(128, new_mul_tensor[0 * 128], cmp_tensor_ub[5 * 128], mid_mul_ub, ranges // 6, 48, 48, 8)

        tik_instance.vec_mul(128, mid_mul_ub, cmp_tensor_ub[0 * 128], cmp_tensor_ub[2 * 128], ranges // 6, 8, 48, 48)
        tik_instance.vec_mul(128, mid_mul_ub, cmp_tensor_ub[3 * 128], mid_mul_ub, ranges // 6, 8, 48, 8)
        tik_instance.vec_mul(128, mid_mul_ub, cmp_tensor_ub[4 * 128], mid_mul_ub, ranges // 6, 8, 48, 8)
        tik_instance.vec_mul(128, new_mul_tensor[1 * 128], cmp_tensor_ub[5 * 128], mid_mul_ub, ranges // 6, 48, 48, 8)

        tik_instance.vec_mul(128, mid_mul_ub, cmp_tensor_ub[0 * 128], cmp_tensor_ub[1 * 128], ranges // 6, 8, 48, 48)
        tik_instance.vec_mul(128, mid_mul_ub, cmp_tensor_ub[3 * 128], mid_mul_ub, ranges // 6, 8, 48, 8)
        tik_instance.vec_mul(128, mid_mul_ub, cmp_tensor_ub[4 * 128], mid_mul_ub, ranges // 6, 8, 48, 8)
        tik_instance.vec_mul(128, new_mul_tensor[2 * 128], cmp_tensor_ub[5 * 128], mid_mul_ub, ranges // 6, 48, 48, 8)

        tik_instance.vec_mul(128, mid_mul_ub, cmp_tensor_ub[0 * 128], cmp_tensor_ub[1 * 128], ranges // 6, 8, 48, 48)
        tik_instance.vec_mul(128, mid_mul_ub, cmp_tensor_ub[2 * 128], mid_mul_ub, ranges // 6, 8, 48, 8)
        tik_instance.vec_mul(128, mid_mul_ub, cmp_tensor_ub[4 * 128], mid_mul_ub, ranges // 6, 8, 48, 8)
        tik_instance.vec_mul(128, new_mul_tensor[3 * 128], cmp_tensor_ub[5 * 128], mid_mul_ub, ranges // 6, 48, 48, 8)

        tik_instance.vec_mul(128, mid_mul_ub, cmp_tensor_ub[0 * 128], cmp_tensor_ub[1 * 128], ranges // 6, 8, 48, 48)
        tik_instance.vec_mul(128, mid_mul_ub, cmp_tensor_ub[2 * 128], mid_mul_ub, ranges // 6, 8, 48, 8)
        tik_instance.vec_mul(128, mid_mul_ub, cmp_tensor_ub[3 * 128], mid_mul_ub, ranges // 6, 8, 48, 8)
        tik_instance.vec_mul(128, new_mul_tensor[4 * 128], cmp_tensor_ub[5 * 128], mid_mul_ub, ranges // 6, 48, 48, 8)

        tik_instance.vec_mul(128, mid_mul_ub, cmp_tensor_ub[0 * 128], cmp_tensor_ub[1 * 128], ranges // 6, 8, 48, 48)
        tik_instance.vec_mul(128, mid_mul_ub, cmp_tensor_ub[2 * 128], mid_mul_ub, ranges // 6, 8, 48, 8)
        tik_instance.vec_mul(128, mid_mul_ub, cmp_tensor_ub[3 * 128], mid_mul_ub, ranges // 6, 8, 48, 8)
        tik_instance.vec_mul(128, new_mul_tensor[5 * 128], cmp_tensor_ub[4 * 128], mid_mul_ub, ranges // 6, 48, 48, 8)

    def min_fun_over_four(self, mid_mul_ub, abs_tmp_ub, min_res_tensor):
        """
        min_fun_over_four
        """
        tik_instance = self.tik_instance
        ranges = 24
        tik_instance.vec_min(128, mid_mul_ub, abs_tmp_ub[1 * 128], abs_tmp_ub[2 * 128], ranges // 6, 8, 48, 48)
        tik_instance.vec_min(128, mid_mul_ub, abs_tmp_ub[3 * 128], mid_mul_ub, ranges // 6, 8, 48, 8)
        tik_instance.vec_min(128, mid_mul_ub, abs_tmp_ub[4 * 128], mid_mul_ub, ranges // 6, 8, 48, 8)
        tik_instance.vec_min(128, min_res_tensor[0 * 128], abs_tmp_ub[5 * 128], mid_mul_ub, ranges // 6, 48, 48, 8)

        tik_instance.vec_min(128, mid_mul_ub, abs_tmp_ub[0 * 128], abs_tmp_ub[2 * 128], ranges // 6, 8, 48, 48)
        tik_instance.vec_min(128, mid_mul_ub, abs_tmp_ub[3 * 128], mid_mul_ub, ranges // 6, 8, 48, 8)
        tik_instance.vec_min(128, mid_mul_ub, abs_tmp_ub[4 * 128], mid_mul_ub, ranges // 6, 8, 48, 8)
        tik_instance.vec_min(128, min_res_tensor[1 * 128], abs_tmp_ub[5 * 128], mid_mul_ub, ranges // 6, 48, 48, 8)

        tik_instance.vec_min(128, mid_mul_ub, abs_tmp_ub[0 * 128], abs_tmp_ub[1 * 128], ranges // 6, 8, 48, 48)
        tik_instance.vec_min(128, mid_mul_ub, abs_tmp_ub[3 * 128], mid_mul_ub, ranges // 6, 8, 48, 8)
        tik_instance.vec_min(128, mid_mul_ub, abs_tmp_ub[4 * 128], mid_mul_ub, ranges // 6, 8, 48, 8)
        tik_instance.vec_min(128, min_res_tensor[2 * 128], abs_tmp_ub[5 * 128], mid_mul_ub, ranges // 6, 48, 48, 8)

        tik_instance.vec_min(128, mid_mul_ub, abs_tmp_ub[0 * 128], abs_tmp_ub[1 * 128], ranges // 6, 8, 48, 48)
        tik_instance.vec_min(128, mid_mul_ub, abs_tmp_ub[2 * 128], mid_mul_ub, ranges // 6, 8, 48, 8)
        tik_instance.vec_min(128, mid_mul_ub, abs_tmp_ub[4 * 128], mid_mul_ub, ranges // 6, 8, 48, 8)
        tik_instance.vec_min(128, min_res_tensor[3 * 128], abs_tmp_ub[5 * 128], mid_mul_ub, ranges // 6, 48, 48, 8)

        tik_instance.vec_min(128, mid_mul_ub, abs_tmp_ub[0 * 128], abs_tmp_ub[1 * 128], ranges // 6, 8, 48, 48)
        tik_instance.vec_min(128, mid_mul_ub, abs_tmp_ub[2 * 128], mid_mul_ub, ranges // 6, 8, 48, 8)
        tik_instance.vec_min(128, mid_mul_ub, abs_tmp_ub[3 * 128], mid_mul_ub, ranges // 6, 8, 48, 8)
        tik_instance.vec_min(128, min_res_tensor[4 * 128], abs_tmp_ub[5 * 128], mid_mul_ub, ranges // 6, 48, 48, 8)

        tik_instance.vec_min(128, mid_mul_ub, abs_tmp_ub[0 * 128], abs_tmp_ub[1 * 128], ranges // 6, 8, 48, 48)
        tik_instance.vec_min(128, mid_mul_ub, abs_tmp_ub[2 * 128], mid_mul_ub, ranges // 6, 8, 48, 8)
        tik_instance.vec_min(128, mid_mul_ub, abs_tmp_ub[3 * 128], mid_mul_ub, ranges // 6, 8, 48, 8)
        tik_instance.vec_min(128, min_res_tensor[5 * 128], abs_tmp_ub[4 * 128], mid_mul_ub, ranges // 6, 48, 48, 8)

    def sign_and_min_less_four(self, new_mul_tensor, cmp_tensor_ub, abs_tmp_ub, temp_ub, min_res_tensor):
        """
        sign_and_min_less_four
        """
        tik_instance = self.tik_instance
        ranges = 24
        tik_instance.vec_mul(128, new_mul_tensor[0 * 128], cmp_tensor_ub[1 * 128],
                             cmp_tensor_ub[2 * 128], ranges // 6, 48, 48, 48)
        tik_instance.vec_mul(128, new_mul_tensor[1 * 128], cmp_tensor_ub[0 * 128],
                             cmp_tensor_ub[2 * 128], ranges // 6, 48, 48, 48)
        tik_instance.vec_mul(128, new_mul_tensor[2 * 128], cmp_tensor_ub[0 * 128],
                             cmp_tensor_ub[1 * 128], ranges // 6, 48, 48, 48)
        tik_instance.vabs(128, abs_tmp_ub, temp_ub, ranges, 1, 1, 8, 8)
        tik_instance.vec_min(128, min_res_tensor[0 * 128], abs_tmp_ub[1 * 128],
                             abs_tmp_ub[2 * 128], ranges // 6, 48, 48, 48)
        tik_instance.vec_min(128, min_res_tensor[1 * 128], abs_tmp_ub[0 * 128],
                             abs_tmp_ub[2 * 128], ranges // 6, 48, 48, 48)
        tik_instance.vec_min(128, min_res_tensor[2 * 128], abs_tmp_ub[0 * 128],
                             abs_tmp_ub[1 * 128], ranges // 6, 48, 48, 48)

    def cal_en_with_iter(self, tik_instance, ranges, core_index):
        """
        cal_en_with_iter
        """
        index_ub, temp_ub, lmn0_ub, negative_one_ub, positive_one_ub, zero_ub, mid_mul_ub, \
        abs_tmp_ub, cmp_tensor_ub, new_mul_tensor, min_res_tensor = self.ub_for_data()
        k = tik_instance.Scalar("int32", name="k")
        self.dup_value(negative_one_ub, 128, -1)
        self.dup_value(positive_one_ub, 128, 1)
        self.dup_value(zero_ub, 128, 0)
        with tik_instance.for_range(0, 20) as _:
            with tik_instance.for_range(0, 12) as j:
                with tik_instance.for_range(0, 4) as ind:
                    tik_instance.data_move(index_ub, self.indices[(j * 512 + core_index * 16 + ind * 4) * 6],
                                           0, 1, 3, 0, 0)
                    with tik_instance.for_range(0, ranges) as index:
                        k.set_as(index_ub[index])
                        with tik_instance.if_scope(k != -1):
                            tik_instance.data_move(temp_ub[index * 128], self.y_en[k * 128], 0, 1, 8, 0, 0)
                    with tik_instance.if_scope(j < 4):
                        tik_instance.data_move(lmn0_ub, self.lmn0[(j * 512 + core_index * 16 + ind * ranges
                                                                   // 6) * 6 * 128], 0, ranges // 6, 24, 24, 24)
                    with tik_instance.else_scope():
                        tik_instance.data_move(lmn0_ub, self.lmn0[(j * 512 + core_index * 16 + ind * ranges
                                                                   // 6) * 6 * 128], 0, 1, ranges * 8, 0, 0)
                    tik_instance.vec_sub(128, temp_ub, temp_ub, lmn0_ub, ranges, 8, 8, 8)
                    with tik_instance.for_range(0, ranges) as i:
                        cmp_mask = tik_instance.vcmp_ge(128, temp_ub[i * 128], zero_ub, 1, 1)
                        tik_instance.vsel(128, 0, cmp_tensor_ub[i * 128], cmp_mask, positive_one_ub,
                                          negative_one_ub, 1, 1, 1, 1, 8, 8, 8)
                    with tik_instance.if_scope(j < 4):
                        self.sign_and_min_less_four(new_mul_tensor, cmp_tensor_ub, abs_tmp_ub, temp_ub,
                                                    min_res_tensor)
                    with tik_instance.else_scope():
                        self.sign_fun_over_four(mid_mul_ub, cmp_tensor_ub, new_mul_tensor)
                        tik_instance.vabs(128, abs_tmp_ub, temp_ub, ranges, 1, 1, 8, 8)
                        self.min_fun_over_four(mid_mul_ub, abs_tmp_ub, min_res_tensor)
                    tik_instance.vec_mul(128, min_res_tensor, min_res_tensor, new_mul_tensor, ranges, 8, 8, 8)
                    tik_instance.vmuls(128, min_res_tensor, min_res_tensor,
                                       tik_instance.Scalar(init_value=0.8, dtype="float16"), ranges, 1, 1, 8, 8)
                    tik_instance.vec_add(128, new_mul_tensor, temp_ub, min_res_tensor, ranges, 8, 8, 8)

                    with tik_instance.for_range(0, ranges) as index:
                        k.set_as(index_ub[index])
                        with tik_instance.if_scope(k != -1):
                            tik_instance.data_move(self.y_en[k * 128], new_mul_tensor[index * 128], 0, 1, 8, 0, 0)
                    with tik_instance.if_scope(j < 4):
                        tik_instance.data_move(self.lmn0[(j * 512 + core_index * 16 + ind * ranges
                                                          // 6) * 6 * 128], min_res_tensor, 0, ranges // 6, 24, 24, 24)
                    with tik_instance.else_scope():
                        tik_instance.data_move(self.lmn0[(j * 512 + core_index * 16 + ind * ranges
                                                          // 6) * 6 * 128], min_res_tensor, 0, 1, ranges * 8, 0, 0)
                tik_instance.block_barrier(self.sync_workspace)

    def cal_cn(self, tik_instance, core_index):
        """
        cal_cn
        """
        shape_x = (320, 128)
        res_ub = tik_instance.Tensor("uint8", (128, 40), scope=tik.scope_ubuf, name="res_ub")
        with tik_instance.new_stmt_scope():
            x_ub = tik_instance.Tensor("float16", shape_x, scope=tik.scope_ubuf, name="x_ub")
            x_conv_ub = tik_instance.Tensor("float16", (128, 320), scope=tik.scope_ubuf, name="x_conv_ub")
            zero_ub = tik_instance.Tensor("float16", (128, 320), scope=tik.scope_ubuf, name="zero_ub")
            self.dup_value(zero_ub, 128 * 320, 0)
            time_2 = 20
            time_3 = 8
            tik_instance.data_move(x_ub, self.y_en[core_index * 320 * 128], 0, 1, 2560, 0, 0)
            with tik_instance.for_range(0, time_2) as i:
                with tik_instance.for_range(0, time_3) as j:
                    src_list = []
                    dst_list = []
                    for k in range(Constant.BLOCK):
                        src_list.append(x_ub[time_3 * Constant.BLOCK * Constant.BLOCK * i +
                                             Constant.BLOCK * j + time_3 * Constant.BLOCK * k])
                        dst_list.append(x_conv_ub[time_2 * Constant.BLOCK * Constant.BLOCK * j
                                      + Constant.BLOCK * i + time_2 * Constant.BLOCK * k])
                    tik_instance.vnchwconv(True, False, dst_list, src_list, 1, 0, 0)
            tik_instance.vec_cmpv_gt(res_ub, x_conv_ub, zero_ub, 254, 8, 8)
            tik_instance.vec_cmpv_gt(res_ub[254 * 16], x_conv_ub[254 * 128], zero_ub, 66, 8, 8)
        mid_ub = tik_instance.Tensor("uint8", (128, 32), scope=tik.scope_ubuf, name="mid_ub")
        first_ub = tik_instance.Tensor("uint8", (128, 32), scope=tik.scope_ubuf, name="first_ub")
        with tik_instance.for_range(0, 128) as i:
            with tik_instance.for_range(0, 32) as j:
                first_ub[i * 32 + j].set_as(res_ub[i * 40 + j])
        with tik_instance.for_range(0, 128) as i:
            with tik_instance.for_range(0, 32) as j:
                mid_ub[i * 32 + j].set_as(res_ub[i * 40 + 8 + j])
        tik_instance.data_move(self.y_cn[core_index * 40], first_ub, 0, 128, 1, 0, 39)
        tik_instance.data_move(self.y_cn[core_index * 40 + 8], mid_ub, 0, 128, 1, 0, 39)

    def dup_value(self, dst, num, dup_value=0, offset=0):
        """
        dup value to ub
        """
        dtype_byte_size = LDPC.get_dtype_size(dst.dtype)
        mask = 256 // dtype_byte_size
        stride = 8
        loop = num // (mask * 255)
        if loop > 0:
            with self.tik_instance.for_range(0, loop) as index:
                tmp_offset = offset + index * mask * 255
                self.tik_instance.vec_dup(mask, dst[tmp_offset], dup_value, 255, stride)
            offset += loop * mask * 255

        repeat_time = (num % (mask * 255)) // mask
        if repeat_time > 0:
            self.tik_instance.vec_dup(mask, dst[offset], dup_value, repeat_time, stride)
            offset += repeat_time * mask
        last_num = num % mask
        if last_num > 0:
            self.tik_instance.vec_dup(last_num, dst[offset], dup_value, 1, stride)

    def cal_en_and_cn_core(self, core_index):
        """
        cal_en_and_cn_core
        """
        ranges = 24
        tik_instance = self.tik_instance
        with tik_instance.new_stmt_scope():
            fill_ub = tik_instance.Tensor("float16", (320, 128), scope=tik.scope_ubuf, name="fill_ub")
            tik_instance.data_move(fill_ub, self.x[core_index * 320 * 128], 0, 1, 2560, 0, 0)
            tik_instance.data_move(self.y_en[core_index * 320 * 128], fill_ub, 0, 1, 2560, 0, 0)
            tik_instance.block_barrier(self.sync_workspace1)
        with tik_instance.new_stmt_scope():
            self.cal_en_with_iter(tik_instance, ranges, core_index)
        self.cal_cn(tik_instance, core_index)


def ldpc(x, indices, y_en, y_cn, lmn0, kernel_name="ldpc"):
    """
    the main function of ldpc
    """
    ldpc_instance = LDPC(x, indices, y_en, y_cn, lmn0, kernel_name)
    tik_instance = ldpc_instance.ldpc_compute()
    return tik_instance
