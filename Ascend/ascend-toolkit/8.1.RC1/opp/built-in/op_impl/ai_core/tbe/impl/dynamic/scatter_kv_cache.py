#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2023 Huawei Technologies Co., Ltd
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
"""ascend custom op: scatter_kv_cache by tik for op Scatter"""

import functools
from tbe import tik
import tbe.common.platform as tbe_platform


class ScatterKvCacheDynImpl():
    def __init__(self, obj):
        # pass
        self.element_obj = obj

        self.tik_inst = self.element_obj.tik_instance

        self.out_burst_len = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='out_burst_len')
        self.each_core_bs_num = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='each_core_bs_num')
        self.core_num = self.element_obj.used_aicore_num
        self.last_core_bs_num = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='last_core_bs_num')
        self.update_axis_shape = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='update_axis_shape')
        self.index_burst_len = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='index_burst_len')
        self.update_burst_len = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='update_burst_len')
        self.src_bs_stride = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='src_bs_stride')
        self.dst_bs_stride = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='dst_bs_stride')
        self.index_elements = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='index_elements')
        self.num_head = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='num_head')
        self.size_per_head = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='size_per_head')
        self.data_axis_shape = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='data_axis_shape')
        self.scatter_stride = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='scatter_stride')
        self.num_one_block = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='num_one_block')
        self.inner_loop_ele = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='inner_loop_ele')
        self.inner_loop_times = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='inner_loop_times')
        self.inner_loop_tail = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='inner_loop_tail')
        self.indices_shape_rank = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='indices_shape_rank')
        self.update_dim_0 = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='update_dim_0')
        self.update_dim_1 = self.tik_inst.Scalar(self.element_obj.tiling_param_dtype, name='update_dim_1')

        tiling_ub = self.element_obj.tiling_ub
        self.out_burst_len.set_as(tiling_ub[2])
        self.each_core_bs_num.set_as(tiling_ub[3])
        self.last_core_bs_num.set_as(tiling_ub[4])
        self.update_axis_shape.set_as(tiling_ub[5])
        self.index_burst_len.set_as(tiling_ub[6])
        self.update_burst_len.set_as(tiling_ub[7])
        self.src_bs_stride.set_as(tiling_ub[8])
        self.dst_bs_stride.set_as(tiling_ub[9])
        self.index_elements.set_as(tiling_ub[10])
        self.num_head.set_as(tiling_ub[11])
        self.size_per_head.set_as(tiling_ub[12])
        self.data_axis_shape.set_as(tiling_ub[13])
        self.scatter_stride.set_as(tiling_ub[14])
        self.num_one_block.set_as(tiling_ub[15])
        self.inner_loop_ele.set_as(tiling_ub[16])
        self.inner_loop_times.set_as(tiling_ub[17])
        self.inner_loop_tail.set_as(tiling_ub[18])
        self.indices_shape_rank.set_as(tiling_ub[19])
        self.update_dim_0.set_as(tiling_ub[20])
        self.update_dim_1.set_as(tiling_ub[21])

        self.valid_idx = self.tik_inst.Scalar(dtype="int64", name="valid_idx")
        self.bs_idx = self.tik_inst.Scalar(dtype="int64", name="bs_idx")
        self.dst_offset = self.tik_inst.Scalar(dtype="int64", name="dst_offset")
        self.data_dtype_size = self.get_dtype_size(self.element_obj.dtype_data)
        self.index_dtype_size = self.get_dtype_size(self.element_obj.dtype_indices)

    @staticmethod
    def get_dtype_size(dtype):
        """
        get byte size of dtype
        """
        dtype_dict = {"float32": 4, "float16": 2, "int8": 1, "int32": 4, "int64": 8, "bfloat16": 2}
        return dtype_dict.get(dtype)

    def transpose_b16(self, src_shape, src_ub, dst_ub, src_rep_stride, dst_rep_stride):
        src_list = [src_ub[src_shape[1] * i] for i in range(16)]
        dst_list = [dst_ub[src_shape[0] * i] for i in range(16)]
        repeat_time = src_shape[0] * src_shape[1] // 256
        self.tik_inst.vnchwconv(False, True, dst_list, src_list, repeat_time, dst_rep_stride, src_rep_stride)

    def transpose_b8(self, src_shape, src_ub, dst_ub, src_rep_stride, dst_rep_stride):
        shape0 = src_shape[0]
        shape1 = src_shape[1]

        repeat_time = shape0 * shape1 // (16 * 16) // 4

        src_list = [src_ub[shape1 * i] for i in range(16)]
        dst_list = [dst_ub[shape0 * i] for i in range(16)]
        self.tik_inst.vnchwconv(False, False, dst_list, src_list, repeat_time, dst_rep_stride, src_rep_stride)

        src_list = [src_ub[shape1 * i] for i in range(16)]
        dst_list = [dst_ub[shape0 * (i + 16)] for i in range(16)]
        self.tik_inst.vnchwconv(False, True, dst_list, src_list, repeat_time, dst_rep_stride, src_rep_stride)

        src_list = [src_ub[shape1 * (i + 16)] for i in range(16)]
        dst_list = [dst_ub[shape0 * i] for i in range(16)]
        self.tik_inst.vnchwconv(True, False, dst_list, src_list, repeat_time, dst_rep_stride, src_rep_stride)

        src_list = [src_ub[shape1 * (i + 16)] for i in range(16)]
        dst_list = [dst_ub[shape0 * (i + 16)] for i in range(16)]
        self.tik_inst.vnchwconv(True, True, dst_list, src_list, repeat_time, dst_rep_stride, src_rep_stride)

    def transpose_b32(self, src_shape, src_ub, dst_ub):
        shape0 = src_shape[0]
        with self.tik_inst.for_range(0, shape0 // 16) as repeat_idx:
            src_list = [src_ub[8 * i + repeat_idx * 128] for i in range(16)]
            dst_list = [0] * 16
            for i in range(8):
                dst_list[2 * i] = dst_ub[shape0 * i + repeat_idx * 16]
                dst_list[2 * i + 1] = dst_ub[shape0 * i + 8 + repeat_idx * 16]
            self.tik_inst.vnchwconv(False, False, dst_list, src_list, 1, 0, 0)

    def transpose_b32_bak(self, src_shape, src_ub, dst_ub):
        shape1 = src_shape[1]
        with self.tik_inst.for_range(0, shape1 // 16) as repeat_idx:
            dst_list = [0] * 16
            src_list = [src_ub[shape1 * i + repeat_idx * 16] for i in range(8)] +\
                       [src_ub[shape1 * i + 8 + repeat_idx * 16] for i in range(8)]
            for i in range(8):
                dst_list[2 * i] = dst_ub[8 * i + repeat_idx * 128]
                dst_list[2 * i + 1] = dst_ub[8 * i + 64 + repeat_idx * 128]
            self.tik_inst.vnchwconv(False, False, dst_list, src_list, 1, 0, 0)

    def compute_each_core_transpose_in(self, src_shape_dim0, data_ub, transpose_ub):
        if self.element_obj.dtype_data in ("int8",):
            self.transpose_b8((src_shape_dim0, self.num_one_block), data_ub, transpose_ub, 32, 1)
        elif self.element_obj.dtype_data in ("float16", "bfloat16"):
            self.transpose_b16((src_shape_dim0, self.num_one_block), data_ub, transpose_ub, 16, 1)
        elif self.element_obj.dtype_data in ("float32",):
            self.transpose_b32((src_shape_dim0, self.num_one_block), data_ub, transpose_ub)

    def compute_each_core_transpose_out(self, src_shape_dim1, data_ub, transpose_ub):
        if self.element_obj.dtype_data in ("int8",):
            self.transpose_b8((self.num_one_block, src_shape_dim1), transpose_ub, data_ub, 1, 32)
        elif self.element_obj.dtype_data in ("float16", "bfloat16"):
            self.transpose_b16((self.num_one_block, src_shape_dim1), transpose_ub, data_ub, 1, 16)
        elif self.element_obj.dtype_data in ("float32",):
            self.transpose_b32_bak((self.num_one_block, src_shape_dim1), transpose_ub, data_ub)

    def move_in_indices(self, index_ub):
        if tbe_platform.api_check_support("tik.data_move_pad"):
            index_gm_b8 = self.element_obj.indices_gm.reinterpret_cast_to("int8")
            index_ub_b8 = index_ub.reinterpret_cast_to("int8")
            self.tik_inst.data_move_pad(index_ub_b8, index_gm_b8, 1, self.index_elements * self.index_dtype_size, 0, 0)
        else:
            self.tik_inst.data_move(index_ub, self.element_obj.indices_gm, 0, 1, self.index_burst_len, 0, 0)

    def compute_each_core(self, core_idx, core_bs_num):
        index_ub = self.tik_inst.Tensor(self.element_obj.dtype_indices, (self.index_elements,), name="index_ub",
                                        scope=tik.scope_ubuf)
        self.move_in_indices(index_ub)

        updates_ub_elements = core_bs_num * self.src_bs_stride
        updates_ub = self.tik_inst.Tensor(self.element_obj.dtype_data, (updates_ub_elements,), name="updates_ub",
                                          scope=tik.scope_ubuf)
        updates_offset = core_idx * self.each_core_bs_num * self.src_bs_stride
        if tbe_platform.api_check_support("tik.data_move_pad"):
            self.tik_inst.data_move_pad(updates_ub, self.element_obj.updates_gm[updates_offset], 1,
                                        updates_ub_elements * self.data_dtype_size, 0, 0)
        else:
            self.tik_inst.data_move(updates_ub, self.element_obj.updates_gm[updates_offset], 0, 1,
                                    (updates_ub_elements * self.data_dtype_size + 31) // 32, 0, 0)

        with self.tik_inst.for_range(0, core_bs_num) as each_core_bs_idx:
            src_offset = each_core_bs_idx * self.src_bs_stride
            self._set_dst_offset_neg2(core_idx, each_core_bs_idx, index_ub)
            if tbe_platform.api_check_support("tik.data_move_pad"):
                self.tik_inst.data_move_pad(self.element_obj.result_gm[self.dst_offset], updates_ub[src_offset], 1,
                                            self.src_bs_stride * self.data_dtype_size, 0, 0)
            else:
                self.tik_inst.data_move(self.element_obj.result_gm[self.dst_offset], updates_ub[src_offset], 0, 1,
                                        self.out_burst_len, 0, 0)

    def compute_each_core_transpose(self, core_idx, core_bs_num):
        updates_ub_elements = core_bs_num * self.src_bs_stride
        index_ub = self.tik_inst.Tensor(self.element_obj.dtype_indices, (self.index_elements,), name="index_ub",
                                        scope=tik.scope_ubuf)
        updates_ub = self.tik_inst.Tensor(self.element_obj.dtype_data, (updates_ub_elements,), name="updates_ub",
                                          scope=tik.scope_ubuf)
        data_ub = self.tik_inst.Tensor(self.element_obj.dtype_data, (self.size_per_head * self.num_one_block,),
                                       name="data_ub", scope=tik.scope_ubuf)
        transpose_ub = self.tik_inst.Tensor(self.element_obj.dtype_data, (self.size_per_head * self.num_one_block,),
                                            name="transpose_ub", scope=tik.scope_ubuf)
        self.move_in_indices(index_ub)

        updates_offset = core_idx * self.each_core_bs_num * self.src_bs_stride
        if tbe_platform.api_check_support("tik.data_move_pad"):
            self.tik_inst.data_move_pad(updates_ub, self.element_obj.updates_gm[updates_offset], 1,
                                        updates_ub_elements * self.data_dtype_size, 0, 0)
        else:
            self.tik_inst.data_move(updates_ub, self.element_obj.updates_gm[updates_offset], 0, 1,
                                    (updates_ub_elements * self.data_dtype_size + 31) // 32, 0, 0)

        with self.tik_inst.for_range(0, core_bs_num) as each_core_bs_idx:
            self._set_dst_offset_neg1(core_idx, each_core_bs_idx, index_ub)
            if tbe_platform.api_check_support("tik.data_move_pad"):
                self.tik_inst.data_move_pad(data_ub, self.element_obj.data_gm[self.dst_offset], self.size_per_head,
                                            self.data_dtype_size, 0, (self.data_axis_shape - 1) * self.data_dtype_size)
            else:
                self.tik_inst.data_move(data_ub, self.element_obj.data_gm[self.dst_offset], 0, self.size_per_head, 1,
                                        self.scatter_stride, 0)
            self.compute_each_core_transpose_in(self.size_per_head, data_ub, transpose_ub)
            self.tik_inst.data_move(transpose_ub, updates_ub[each_core_bs_idx * self.src_bs_stride], 0, 1,
                                    self.out_burst_len, 0, 0)
            self.compute_each_core_transpose_out(self.size_per_head, data_ub, transpose_ub)
            if tbe_platform.api_check_support("tik.data_move_pad"):
                self.tik_inst.data_move_pad(self.element_obj.data_gm[self.dst_offset], data_ub, self.size_per_head,
                                            self.data_dtype_size, (self.data_axis_shape - 1) * self.data_dtype_size, 0)
            else:
                self.tik_inst.data_move(self.element_obj.data_gm[self.dst_offset], data_ub, 0, self.size_per_head, 1, 0,
                                        self.scatter_stride)

    def compute_each_core_large_batch(self, core_idx, core_bs_num):
        index_ub = self.tik_inst.Tensor(self.element_obj.dtype_indices, (self.index_elements,), name="index_ub",
                                        scope=tik.scope_ubuf)
        updates_ub = self.tik_inst.Tensor(self.element_obj.dtype_data, (self.src_bs_stride,), name="updates_ub",
                                          scope=tik.scope_ubuf)
        self.move_in_indices(index_ub)
        
        with self.tik_inst.for_range(0, core_bs_num) as each_core_bs_idx:
            src_offset = (core_idx * self.each_core_bs_num + each_core_bs_idx) * self.src_bs_stride
            self._set_dst_offset_neg2(core_idx, each_core_bs_idx, index_ub)
            if tbe_platform.api_check_support("tik.data_move_pad"):
                update_len = self.src_bs_stride * self.data_dtype_size
                self.tik_inst.data_move_pad(updates_ub, self.element_obj.updates_gm[src_offset], 1, update_len, 0, 0)    
                self.tik_inst.data_move_pad(self.element_obj.result_gm[self.dst_offset], updates_ub, 1, update_len, 0,
                                            0)
            else:
                update_len = (self.src_bs_stride * self.data_dtype_size + 31) // 32
                self.tik_inst.data_move(updates_ub, self.element_obj.updates_gm[src_offset], 0, 1, update_len, 0, 0)
                self.tik_inst.data_move(self.element_obj.result_gm[self.dst_offset], updates_ub, 0, 1, update_len, 0, 0)

    def compute_each_core_transpose_large_batch(self, core_idx, core_bs_num):
        index_ub = self.tik_inst.Tensor(self.element_obj.dtype_indices, (self.index_elements,), name="index_ub",
                                        scope=tik.scope_ubuf)
        updates_ub = self.tik_inst.Tensor(self.element_obj.dtype_data, (self.src_bs_stride,), name="updates_ub",
                                          scope=tik.scope_ubuf)
        data_ub = self.tik_inst.Tensor(self.element_obj.dtype_data, (self.size_per_head * self.num_one_block,),
                                       name="data_ub", scope=tik.scope_ubuf)
        transpose_ub = self.tik_inst.Tensor(self.element_obj.dtype_data, (self.size_per_head * self.num_one_block,),
                                            name="transpose_ub", scope=tik.scope_ubuf)
        self.move_in_indices(index_ub)

        with self.tik_inst.for_range(0, core_bs_num) as each_core_bs_idx:
            src_offset = (core_idx * self.each_core_bs_num + each_core_bs_idx) * self.src_bs_stride
            self._set_dst_offset_neg1(core_idx, each_core_bs_idx, index_ub)
            if tbe_platform.api_check_support("tik.data_move_pad"):
                self.tik_inst.data_move_pad(updates_ub, self.element_obj.updates_gm[src_offset], 1,
                                            self.src_bs_stride * self.data_dtype_size, 0, 0)
                self.tik_inst.data_move_pad(data_ub, self.element_obj.data_gm[self.dst_offset], self.size_per_head,
                                            self.data_dtype_size, 0, (self.data_axis_shape - 1) * self.data_dtype_size)
            else:
                self.tik_inst.data_move(updates_ub, self.element_obj.updates_gm[src_offset], 0, 1,
                                        (self.src_bs_stride * self.data_dtype_size + 31) // 32, 0, 0)
                self.tik_inst.data_move(data_ub, self.element_obj.data_gm[self.dst_offset], 0, self.size_per_head, 1,
                                        self.scatter_stride, 0)
            self.compute_each_core_transpose_in(self.size_per_head, data_ub, transpose_ub)
            self.tik_inst.data_move(transpose_ub, updates_ub, 0, 1, self.out_burst_len, 0, 0)
            self.compute_each_core_transpose_out(self.size_per_head, data_ub, transpose_ub)
            if tbe_platform.api_check_support("tik.data_move_pad"):
                self.tik_inst.data_move_pad(self.element_obj.data_gm[self.dst_offset], data_ub, self.size_per_head,
                                            self.data_dtype_size, (self.data_axis_shape - 1) * self.data_dtype_size, 0)
            else:
                self.tik_inst.data_move(self.element_obj.data_gm[self.dst_offset], data_ub, 0, self.size_per_head, 1, 0,
                                        self.scatter_stride)

    def compute_each_core_large_ele(self, core_idx, core_bs_num):
        index_ub = self.tik_inst.Tensor(self.element_obj.dtype_indices, (self.index_elements,), name="index_ub",
                                        scope=tik.scope_ubuf)
        updates_ub = self.tik_inst.Tensor(self.element_obj.dtype_data, (self.inner_loop_ele,), name="updates_ub",
                                          scope=tik.scope_ubuf)
        self.move_in_indices(index_ub)

        with self.tik_inst.for_range(0, core_bs_num) as each_core_bs_idx:
            self._set_dst_offset_neg2(core_idx, each_core_bs_idx, index_ub)
            with self.tik_inst.for_range(0, self.inner_loop_times) as inner_loop_idx:
                src_offset = (core_idx * self.each_core_bs_num + each_core_bs_idx) * self.src_bs_stride + \
                             inner_loop_idx * self.inner_loop_ele
                dst_offset_new = self.dst_offset + inner_loop_idx * self.inner_loop_ele
                if tbe_platform.api_check_support("tik.data_move_pad"):
                    self.tik_inst.data_move_pad(updates_ub, self.element_obj.updates_gm[src_offset], 1, 
                                                self.inner_loop_ele * self.data_dtype_size, 0, 0)
                    self.tik_inst.data_move_pad(self.element_obj.result_gm[dst_offset_new], updates_ub, 1,
                                                self.inner_loop_ele * self.data_dtype_size, 0, 0)
                else:
                    self.tik_inst.data_move(updates_ub, self.element_obj.updates_gm[src_offset], 0, 1,
                                            (self.inner_loop_ele * self.data_dtype_size + 31) // 32, 0, 0)
                    self.tik_inst.data_move(self.element_obj.result_gm[dst_offset_new], updates_ub, 0, 1,
                                            (self.inner_loop_ele * self.data_dtype_size + 31) // 32, 0, 0)
            with self.tik_inst.if_scope(self.inner_loop_tail != 0):
                src_offset = (core_idx * self.each_core_bs_num + each_core_bs_idx) * self.src_bs_stride + \
                             self.inner_loop_times * self.inner_loop_ele
                dst_offset_new = self.dst_offset + self.inner_loop_times * self.inner_loop_ele
                if tbe_platform.api_check_support("tik.data_move_pad"):
                    self.tik_inst.data_move_pad(updates_ub, self.element_obj.updates_gm[src_offset], 1,
                                                self.inner_loop_tail * self.data_dtype_size, 0, 0)
                    self.tik_inst.data_move_pad(self.element_obj.result_gm[dst_offset_new], updates_ub, 1,
                                                self.inner_loop_tail * self.data_dtype_size, 0, 0)
                else:
                    self.tik_inst.data_move(updates_ub, self.element_obj.updates_gm[src_offset], 0, 1,
                                            (self.inner_loop_tail * self.data_dtype_size + 31) // 32, 0, 0)
                    self.tik_inst.data_move(self.element_obj.result_gm[dst_offset_new], updates_ub, 0, 1,
                                            (self.inner_loop_tail * self.data_dtype_size + 31) // 32, 0, 0)

    def compute_each_core_transpose_large_ele(self, core_idx, core_bs_num):
        index_ub = self.tik_inst.Tensor(self.element_obj.dtype_indices, (self.index_elements,), name="index_ub",
                                        scope=tik.scope_ubuf)
        updates_ub = self.tik_inst.Tensor(self.element_obj.dtype_data, (self.inner_loop_ele,), name="updates_ub",
                                          scope=tik.scope_ubuf)
        data_ub = self.tik_inst.Tensor(self.element_obj.dtype_data, (self.inner_loop_ele * self.num_one_block,),
                                       name="data_ub", scope=tik.scope_ubuf)
        transpose_ub = self.tik_inst.Tensor(self.element_obj.dtype_data, (self.inner_loop_ele * self.num_one_block,),
                                            name="transpose_ub", scope=tik.scope_ubuf)
        self.move_in_indices(index_ub)

        with self.tik_inst.for_range(0, core_bs_num) as each_core_bs_idx:
            with self.tik_inst.for_range(0, self.inner_loop_times) as inner_loop_idx:
                src_offset = (core_idx * self.each_core_bs_num + each_core_bs_idx) * self.src_bs_stride + \
                             inner_loop_idx * self.inner_loop_ele
                self._set_dst_offset_neg1(core_idx, each_core_bs_idx, index_ub)
                dst_offset_new = self.dst_offset + inner_loop_idx * self.inner_loop_ele * self.data_axis_shape
                if tbe_platform.api_check_support("tik.data_move_pad"):
                    self.tik_inst.data_move_pad(updates_ub, self.element_obj.updates_gm[src_offset], 1,
                                                self.inner_loop_ele * self.data_dtype_size, 0, 0)
                    self.tik_inst.data_move_pad(data_ub, self.element_obj.data_gm[dst_offset_new],
                                                self.inner_loop_ele, self.data_dtype_size,
                                                0, (self.data_axis_shape - 1) * self.data_dtype_size)
                else:
                    self.tik_inst.data_move(updates_ub, self.element_obj.updates_gm[src_offset], 0, 1,
                                            (self.inner_loop_ele * self.data_dtype_size + 31) // 32, 0, 0)
                    self.tik_inst.data_move(data_ub, self.element_obj.data_gm[dst_offset_new], 0, self.inner_loop_ele,
                                            1, self.scatter_stride, 0)
                self.compute_each_core_transpose_in(self.inner_loop_ele, data_ub, transpose_ub)
                self.tik_inst.data_move(transpose_ub, updates_ub, 0, 1,
                                        (self.inner_loop_ele * self.data_dtype_size + 31) // 32, 0, 0)
                self.compute_each_core_transpose_out(self.inner_loop_ele, data_ub, transpose_ub)
                if tbe_platform.api_check_support("tik.data_move_pad"):
                    self.tik_inst.data_move_pad(self.element_obj.data_gm[dst_offset_new], data_ub,
                                                self.inner_loop_ele, self.data_dtype_size,
                                                (self.data_axis_shape - 1) * self.data_dtype_size, 0)
                else:
                    self.tik_inst.data_move(self.element_obj.data_gm[dst_offset_new], data_ub, 0, self.inner_loop_ele,
                                            1, 0, self.scatter_stride)
            with self.tik_inst.if_scope(self.inner_loop_tail != 0):
                src_offset = (core_idx * self.each_core_bs_num + each_core_bs_idx) * self.src_bs_stride + \
                             self.inner_loop_times * self.inner_loop_ele
                self._set_dst_offset_neg1(core_idx, each_core_bs_idx, index_ub)
                dst_offset_new = self.dst_offset + self.inner_loop_times * self.inner_loop_ele * self.data_axis_shape
                if tbe_platform.api_check_support("tik.data_move_pad"):
                    self.tik_inst.data_move_pad(updates_ub, self.element_obj.updates_gm[src_offset], 1,
                                                self.inner_loop_tail * self.data_dtype_size, 0, 0)
                    self.tik_inst.data_move_pad(data_ub, self.element_obj.data_gm[dst_offset_new],
                                                self.inner_loop_tail, self.data_dtype_size, 0,
                                                (self.data_axis_shape - 1) * self.data_dtype_size)
                else:
                    self.tik_inst.data_move(updates_ub, self.element_obj.updates_gm[src_offset], 0, 1,
                                            (self.inner_loop_tail * self.data_dtype_size + 31) // 32, 0, 0)
                    self.tik_inst.data_move(data_ub, self.element_obj.data_gm[dst_offset_new], 0, self.inner_loop_tail,
                                            1, self.scatter_stride, 0)
                self.compute_each_core_transpose_in(self.inner_loop_tail, data_ub, transpose_ub)
                self.tik_inst.data_move(transpose_ub, updates_ub, 0, 1,
                                        (self.inner_loop_tail * self.data_dtype_size + 31) // 32, 0, 0)
                self.compute_each_core_transpose_out(self.inner_loop_tail, data_ub, transpose_ub)
                if tbe_platform.api_check_support("tik.data_move_pad"):
                    self.tik_inst.data_move_pad(self.element_obj.data_gm[dst_offset_new], data_ub,
                                                self.inner_loop_tail, self.data_dtype_size,
                                                (self.data_axis_shape - 1) * self.data_dtype_size, 0)
                else:
                    self.tik_inst.data_move(self.element_obj.data_gm[dst_offset_new], data_ub, 0, self.inner_loop_tail,
                                            1, 0, self.scatter_stride)

    def compute_each_core_neg_1_continous(self, core_idx, core_bs_num):
        index_ub = self.tik_inst.Tensor(self.element_obj.dtype_indices, (self.index_elements,), name="index_ub",
                                        scope=tik.scope_ubuf)
        updates_ub = self.tik_inst.Tensor(self.element_obj.dtype_data, (self.inner_loop_ele,), name="updates_ub",
                                          scope=tik.scope_ubuf)
        self.move_in_indices(index_ub)

        with self.tik_inst.for_range(0, core_bs_num) as each_core_bs_idx:
            self._set_dst_offset_neg1(core_idx, each_core_bs_idx, index_ub)
            with self.tik_inst.for_range(0, self.size_per_head) as axis_idx:
                with self.tik_inst.for_range(0, self.inner_loop_times) as inner_loop_idx:
                    src_offset = (core_idx * self.each_core_bs_num + each_core_bs_idx) * self.src_bs_stride + \
                                 inner_loop_idx * self.inner_loop_ele + axis_idx * self.update_axis_shape
                    dst_offset_new = self.dst_offset + inner_loop_idx * self.inner_loop_ele + \
                                     axis_idx * self.data_axis_shape
                    if tbe_platform.api_check_support("tik.data_move_pad"):
                        self.tik_inst.data_move_pad(updates_ub, self.element_obj.updates_gm[src_offset], 1,
                                                    self.inner_loop_ele * self.data_dtype_size, 0, 0)
                        self.tik_inst.data_move_pad(self.element_obj.data_gm[dst_offset_new], updates_ub, 1,
                                                    self.inner_loop_ele * self.data_dtype_size, 0, 0)
                    else:
                        self.tik_inst.data_move(updates_ub, self.element_obj.updates_gm[src_offset], 0, 1,
                                                self.inner_loop_ele * self.data_dtype_size // 32, 0, 0)  
                        self.tik_inst.data_move(self.element_obj.data_gm[dst_offset_new], updates_ub, 0, 1,
                                                self.inner_loop_ele * self.data_dtype_size // 32, 0, 0)
                        src_back_offset = src_offset + self.inner_loop_ele - self.num_one_block
                        dst_back_offset = dst_offset_new + self.inner_loop_ele - self.num_one_block
                        self.tik_inst.data_move(updates_ub, self.element_obj.updates_gm[src_back_offset], 0, 1, 1, 0, 0)
                        self.tik_inst.data_move(self.element_obj.data_gm[dst_back_offset], updates_ub, 0, 1, 1, 0, 0)
                with self.tik_inst.if_scope(self.inner_loop_tail != 0):
                    src_offset = (core_idx * self.each_core_bs_num + each_core_bs_idx) * self.src_bs_stride + \
                                 self.inner_loop_times * self.inner_loop_ele + axis_idx * self.update_axis_shape
                    dst_offset_new = self.dst_offset + self.inner_loop_times * self.inner_loop_ele + \
                                     axis_idx * self.data_axis_shape
                    if tbe_platform.api_check_support("tik.data_move_pad"):
                        self.tik_inst.data_move_pad(updates_ub, self.element_obj.updates_gm[src_offset], 1,
                                                    self.inner_loop_tail * self.data_dtype_size, 0, 0)
                        self.tik_inst.data_move_pad(self.element_obj.data_gm[dst_offset_new], updates_ub, 1,
                                                    self.inner_loop_tail * self.data_dtype_size, 0, 0)
                    else:
                        self.tik_inst.data_move(updates_ub, self.element_obj.updates_gm[src_offset], 0, 1,
                                                self.inner_loop_tail * self.data_dtype_size // 32, 0, 0)                     
                        self.tik_inst.data_move(self.element_obj.data_gm[dst_offset_new], updates_ub, 0, 1,
                                                self.inner_loop_tail * self.data_dtype_size // 32, 0, 0)
                        src_back_offset = src_offset + self.inner_loop_tail - self.num_one_block
                        dst_back_offset = dst_offset_new + self.inner_loop_tail - self.num_one_block
                        self.tik_inst.data_move(updates_ub, self.element_obj.updates_gm[src_back_offset], 0, 1, 1, 0, 0)
                        self.tik_inst.data_move(self.element_obj.data_gm[dst_back_offset], updates_ub,
                                                0, 1, 1, 0, 0)

    def compute_each_core_neg_1_less_than_block(self, core_idx, core_bs_num):
        index_ub = self.tik_inst.Tensor(self.element_obj.dtype_indices, (self.index_elements,), name="index_ub",
                                        scope=tik.scope_ubuf)
        updates_ub = self.tik_inst.Tensor(self.element_obj.dtype_data, (self.num_one_block,), name="updates_ub",
                                          scope=tik.scope_ubuf)
        data_ub = self.tik_inst.Tensor(self.element_obj.dtype_data, (self.num_one_block,), name="data_ub",
                                       scope=tik.scope_ubuf)
        self.move_in_indices(index_ub)

        with self.tik_inst.for_range(0, core_bs_num) as each_core_bs_idx:
            self._set_dst_offset_neg1(core_idx, each_core_bs_idx, index_ub)
            with self.tik_inst.for_range(0, self.size_per_head) as axis_idx:
                src_offset = (core_idx * self.each_core_bs_num + each_core_bs_idx) * self.src_bs_stride + \
                             axis_idx * self.update_axis_shape
                dst_offset_new = self.dst_offset + axis_idx * self.data_axis_shape
                if tbe_platform.api_check_support("tik.data_move_pad"):
                    self.tik_inst.data_move_pad(updates_ub, self.element_obj.updates_gm[src_offset], 1,
                                                self.update_axis_shape * self.data_dtype_size, 0, 0)
                    self.tik_inst.data_move_pad(self.element_obj.data_gm[dst_offset_new], updates_ub, 1,
                                                self.update_axis_shape * self.data_dtype_size, 0, 0)
                else:
                    self.tik_inst.data_move(data_ub, self.element_obj.data_gm[dst_offset_new], 0, 1, 1, 0, 0)
                    self.tik_inst.data_move(updates_ub, self.element_obj.updates_gm[src_offset], 0, 1, 1, 0, 0)
                    with self.tik_inst.for_range(0, self.update_axis_shape) as update_idx:
                        data_ub[update_idx] = updates_ub[update_idx]
                    self.tik_inst.data_move(self.element_obj.data_gm[dst_offset_new], data_ub, 0, 1, 1, 0, 0)

    def compute_each_core_less_batch(self, core_idx, core_bs_num):
        index_ub = self.tik_inst.Tensor(self.element_obj.dtype_indices, (self.index_elements,), name="index_ub",
                                        scope=tik.scope_ubuf)
        updates_ub = self.tik_inst.Tensor(self.element_obj.dtype_data, (self.src_bs_stride,), name="updates_ub",
                                          scope=tik.scope_ubuf)
        index = self.tik_inst.Scalar(dtype="int64", name="index")
        self.move_in_indices(index_ub)

        with self.tik_inst.for_range(0, core_bs_num) as each_core_bs_idx:
            cycle_id = core_idx * self.each_core_bs_num + each_core_bs_idx
            batch_id = cycle_id // (self.num_head * self.update_axis_shape)
            second_batch_id = cycle_id // self.update_axis_shape % self.num_head
            index.set_as(index_ub[batch_id])
            offsets = cycle_id % self.update_axis_shape
            dst_batch_offsets = index + (batch_id * self.num_head + second_batch_id) * self.data_axis_shape + offsets
            src_offset = cycle_id * self.src_bs_stride
            self.dst_offset.set_as(dst_batch_offsets * self.src_bs_stride)
            update_len = self.src_bs_stride * self.data_dtype_size // 32
            self.tik_inst.data_move(updates_ub, self.element_obj.updates_gm[src_offset], 0, 1, update_len, 0, 0)
            self.tik_inst.data_move(self.element_obj.result_gm[self.dst_offset], updates_ub, 0, 1, update_len, 0, 0)
            
    def compute_each_core_one_batch(self, core_idx, core_bs_num):
        index_ub = self.tik_inst.Tensor(self.element_obj.dtype_indices, (self.index_elements,), name="index_ub",
                                        scope=tik.scope_ubuf)
        updates_ub = self.tik_inst.Tensor(self.element_obj.dtype_data, (self.each_core_bs_num * self.src_bs_stride,),
                                          name="updates_ub", scope=tik.scope_ubuf)
        index = self.tik_inst.Scalar(dtype="int64", name="index")
        self.move_in_indices(index_ub)
        index.set_as(index_ub[0])

        src_offset = core_idx * self.each_core_bs_num * self.src_bs_stride
        self.dst_offset.set_as((index + core_idx * self.each_core_bs_num) * self.src_bs_stride)
        update_len = core_bs_num * self.src_bs_stride * self.data_dtype_size // 32
        self.tik_inst.data_move(updates_ub, self.element_obj.updates_gm[src_offset], 0, 1, update_len, 0, 0)
        self.tik_inst.data_move(self.element_obj.result_gm[self.dst_offset], updates_ub, 0, 1, update_len, 0, 0)

    def compute_each_core_large_continous(self, core_idx):
        index_ub = self.tik_inst.Tensor(self.element_obj.dtype_indices, (self.index_elements,), name="index_ub",
                                        scope=tik.scope_ubuf)
        updates_ub = self.tik_inst.Tensor(self.element_obj.dtype_data, (self.inner_loop_ele,), name="updates_ub",
                                          scope=tik.scope_ubuf)
        self.move_in_indices(index_ub)

        dst_fir_bs_stide = self.num_head * self.dst_bs_stride
        update_len = self.inner_loop_ele * self.data_dtype_size // 32

        with self.tik_inst.for_range(0, self.update_dim_0) as update_dim0_idx:
            with self.tik_inst.for_range(0, self.update_dim_1) as update_dim1_idx:
                with self.tik_inst.for_range(0, self.inner_loop_times) as loop_bs_idx:
                    src_offset = update_dim0_idx * self.update_dim_1 * self.update_axis_shape * self.src_bs_stride + \
                                 update_dim1_idx * self.update_axis_shape * self.src_bs_stride + \
                                 (core_idx * self.each_core_bs_num) * self.src_bs_stride + \
                                 loop_bs_idx * self.inner_loop_ele
                    self.tik_inst.data_move(updates_ub, self.element_obj.updates_gm[src_offset], 0, 1, update_len, 0, 0)
                    self._set_dst_offset_neg110(core_idx, self.each_core_bs_num, index_ub, update_dim0_idx,
                                                update_dim1_idx, dst_fir_bs_stide)
                    dst_offset_new = self.dst_offset + loop_bs_idx * self.inner_loop_ele
                    self.tik_inst.data_move(self.element_obj.result_gm[dst_offset_new], updates_ub, 0, 1, update_len, 0,
                                            0)
                with self.tik_inst.if_scope(self.inner_loop_tail != 0):
                    update_len = self.inner_loop_tail * self.data_dtype_size // 32
                    src_offset = update_dim0_idx * self.update_dim_1 * self.update_axis_shape * self.src_bs_stride + \
                                 update_dim1_idx * self.update_axis_shape * self.src_bs_stride + \
                                 (core_idx * self.each_core_bs_num) * self.src_bs_stride + \
                                 self.inner_loop_times * self.inner_loop_ele
                    self.tik_inst.data_move(updates_ub, self.element_obj.updates_gm[src_offset], 0, 1, update_len, 0, 0)
                    self._set_dst_offset_neg110(core_idx, self.each_core_bs_num, index_ub, update_dim0_idx,
                                                update_dim1_idx, dst_fir_bs_stide)
                    dst_offset_new = self.dst_offset + self.inner_loop_times * self.inner_loop_ele
                    self.tik_inst.data_move(self.element_obj.result_gm[dst_offset_new], updates_ub, 0, 1, update_len, 0,
                                            0)

    def compute_each_core_large_batch_pad_move_out(self, updates_ub, data_ub, src_offset):
        if tbe_platform.api_check_support("tik.data_move_pad"):
            update_len = self.src_bs_stride * self.data_dtype_size
            self.tik_inst.data_move_pad(updates_ub, self.element_obj.updates_gm[src_offset], 1, update_len, 0, 0)
            self.tik_inst.data_move_pad(self.element_obj.data_gm[self.dst_offset], updates_ub, 1, update_len, 0, 0)
        else:
            with self.tik_inst.if_scope(self.src_bs_stride >= self.num_one_block):
                update_len = self.src_bs_stride * self.data_dtype_size // 32
                self.tik_inst.data_move(updates_ub, self.element_obj.updates_gm[src_offset], 0, 1, update_len, 0, 0)
                self.tik_inst.data_move(self.element_obj.data_gm[self.dst_offset], updates_ub, 0, 1, update_len, 0, 0)
                src_back_offset = src_offset + self.src_bs_stride - self.num_one_block
                dst_back_offset = self.dst_offset + self.src_bs_stride - self.num_one_block
                self.tik_inst.data_move(updates_ub, self.element_obj.updates_gm[src_back_offset], 0, 1, 1, 0, 0)
                self.tik_inst.data_move(self.element_obj.data_gm[dst_back_offset], updates_ub, 0, 1, 1, 0, 0)
            with self.tik_inst.else_scope():
                self.tik_inst.data_move(data_ub, self.element_obj.data_gm[self.dst_offset], 0, 1, 1, 0, 0)
                self.tik_inst.data_move(updates_ub, self.element_obj.updates_gm[src_offset], 0, 1, 1, 0, 0)
                with self.tik_inst.for_range(0, self.src_bs_stride) as update_idx:
                    data_ub[update_idx] = updates_ub[update_idx]
                self.tik_inst.data_move(self.element_obj.data_gm[self.dst_offset], data_ub, 0, 1, 1, 0, 0)

    def compute_each_core_large_batch_pad(self, core_idx, core_bs_num):
        index_ub = self.tik_inst.Tensor(self.element_obj.dtype_indices, (self.index_elements,), name="index_ub",
                                        scope=tik.scope_ubuf)
        updates_ub = self.tik_inst.Tensor(self.element_obj.dtype_data, (self.src_bs_stride,), name="updates_ub",
                                          scope=tik.scope_ubuf)
        data_ub = self.tik_inst.Tensor(self.element_obj.dtype_data, (self.num_one_block,), name="data_ub",
                                       scope=tik.scope_ubuf)
        self.move_in_indices(index_ub)

        with self.tik_inst.for_range(0, core_bs_num) as each_core_bs_idx:
            src_offset = (core_idx * self.each_core_bs_num + each_core_bs_idx) * self.src_bs_stride
            self._set_dst_offset_neg2(core_idx, each_core_bs_idx, index_ub)
            self.compute_each_core_large_batch_pad_move_out(updates_ub, data_ub, src_offset)

    def compute_each_core_large_ele_pad_move_out(self, updates_ub, src_offset):
        if tbe_platform.api_check_support("tik.data_move_pad"):
            with self.tik_inst.for_range(0, self.inner_loop_times) as inner_loop_idx:
                add_offset = inner_loop_idx * self.inner_loop_ele
                update_len = self.inner_loop_ele * self.data_dtype_size
                self.tik_inst.data_move_pad(updates_ub, self.element_obj.updates_gm[src_offset + add_offset], 1,
                                            update_len, 0, 0)
                self.tik_inst.data_move_pad(self.element_obj.data_gm[self.dst_offset + add_offset], updates_ub, 1,
                                            update_len, 0, 0)
            with self.tik_inst.if_scope(self.inner_loop_tail != 0):
                add_offset = self.inner_loop_times * self.inner_loop_ele
                update_len = self.inner_loop_tail * self.data_dtype_size
                self.tik_inst.data_move_pad(updates_ub, self.element_obj.updates_gm[src_offset + add_offset], 1,
                                            update_len, 0, 0)
                self.tik_inst.data_move_pad(self.element_obj.data_gm[self.dst_offset + add_offset], updates_ub, 1,
                                            update_len, 0, 0)
        else:
            with self.tik_inst.for_range(0, self.inner_loop_times) as inner_loop_idx:
                add_offset = inner_loop_idx * self.inner_loop_ele
                update_len = self.inner_loop_ele * self.data_dtype_size // 32
                self.tik_inst.data_move(updates_ub, self.element_obj.updates_gm[src_offset + add_offset], 0, 1,
                                        update_len, 0, 0)
                self.tik_inst.data_move(self.element_obj.data_gm[self.dst_offset + add_offset], updates_ub, 0, 1,
                                        update_len, 0, 0)
                src_back_offset = src_offset + add_offset + self.inner_loop_ele - self.num_one_block
                dst_back_offset = self.dst_offset + add_offset + self.inner_loop_ele - self.num_one_block
                self.tik_inst.data_move(updates_ub, self.element_obj.updates_gm[src_back_offset], 0, 1, 1, 0, 0)
                self.tik_inst.data_move(self.element_obj.data_gm[dst_back_offset], updates_ub, 0, 1, 1, 0, 0)
            with self.tik_inst.if_scope(self.inner_loop_tail != 0):
                add_offset = self.inner_loop_times * self.inner_loop_ele
                with self.tik_inst.if_scope(self.inner_loop_tail >= self.num_one_block):
                    update_len = self.inner_loop_tail * self.data_dtype_size // 32
                    self.tik_inst.data_move(updates_ub, self.element_obj.updates_gm[src_offset + add_offset], 0, 1,
                                            update_len, 0, 0)
                    self.tik_inst.data_move(self.element_obj.data_gm[self.dst_offset + add_offset], updates_ub, 0, 1,
                                            update_len, 0, 0)
                src_back_offset = src_offset + add_offset + self.inner_loop_tail - self.num_one_block
                dst_back_offset = self.dst_offset + add_offset + self.inner_loop_tail - self.num_one_block
                self.tik_inst.data_move(updates_ub, self.element_obj.updates_gm[src_back_offset], 0, 1, 1, 0, 0)
                self.tik_inst.data_move(self.element_obj.data_gm[dst_back_offset], updates_ub, 0, 1, 1, 0, 0)

    def compute_each_core_large_ele_pad(self, core_idx, core_bs_num):
        index_ub = self.tik_inst.Tensor(self.element_obj.dtype_indices, (self.index_elements,), name="index_ub",
                                        scope=tik.scope_ubuf)
        updates_ub = self.tik_inst.Tensor(self.element_obj.dtype_data, (self.inner_loop_ele,), name="updates_ub",
                                          scope=tik.scope_ubuf)
        self.move_in_indices(index_ub)

        with self.tik_inst.for_range(0, core_bs_num) as each_core_bs_idx:
            src_offset = (core_idx * self.each_core_bs_num + each_core_bs_idx) * self.src_bs_stride
            self._set_dst_offset_neg2(core_idx, each_core_bs_idx, index_ub)
            self.compute_each_core_large_ele_pad_move_out(updates_ub, src_offset)

    def compute(self, core_index):
        with self.tik_inst.if_scope(self.element_obj.tiling_mode == 100):
            with self.tik_inst.if_scope(core_index < self.core_num - 1):
                self.compute_each_core(core_idx=core_index, core_bs_num=self.each_core_bs_num)
            with self.tik_inst.else_scope():
                self.compute_each_core(core_idx=core_index, core_bs_num=self.last_core_bs_num)
        with self.tik_inst.elif_scope(self.element_obj.tiling_mode == 101):
            with self.tik_inst.if_scope(core_index < self.core_num - 1):
                self.compute_each_core_transpose(core_idx=core_index, core_bs_num=self.each_core_bs_num)
            with self.tik_inst.else_scope():
                self.compute_each_core_transpose(core_idx=core_index, core_bs_num=self.last_core_bs_num)
        with self.tik_inst.elif_scope(self.element_obj.tiling_mode == 102):
            with self.tik_inst.if_scope(core_index < self.core_num - 1):
                self.compute_each_core_large_batch(core_idx=core_index, core_bs_num=self.each_core_bs_num)
            with self.tik_inst.else_scope():
                self.compute_each_core_large_batch(core_idx=core_index, core_bs_num=self.last_core_bs_num)
        with self.tik_inst.elif_scope(self.element_obj.tiling_mode == 103):
            with self.tik_inst.if_scope(core_index < self.core_num - 1):
                self.compute_each_core_transpose_large_batch(core_idx=core_index, core_bs_num=self.each_core_bs_num)
            with self.tik_inst.else_scope():
                self.compute_each_core_transpose_large_batch(core_idx=core_index, core_bs_num=self.last_core_bs_num)
        with self.tik_inst.elif_scope(self.element_obj.tiling_mode == 104):
            with self.tik_inst.if_scope(core_index < self.core_num - 1):
                self.compute_each_core_large_ele(core_idx=core_index, core_bs_num=self.each_core_bs_num)
            with self.tik_inst.else_scope():
                self.compute_each_core_large_ele(core_idx=core_index, core_bs_num=self.last_core_bs_num)
        with self.tik_inst.elif_scope(self.element_obj.tiling_mode == 105):
            with self.tik_inst.if_scope(core_index < self.core_num - 1):
                self.compute_each_core_transpose_large_ele(core_idx=core_index, core_bs_num=self.each_core_bs_num)
            with self.tik_inst.else_scope():
                self.compute_each_core_transpose_large_ele(core_idx=core_index, core_bs_num=self.last_core_bs_num)
        with self.tik_inst.elif_scope(self.element_obj.tiling_mode == 106):
            with self.tik_inst.if_scope(core_index < self.core_num - 1):
                self.compute_each_core_neg_1_continous(core_idx=core_index, core_bs_num=self.each_core_bs_num)
            with self.tik_inst.else_scope():
                self.compute_each_core_neg_1_continous(core_idx=core_index, core_bs_num=self.last_core_bs_num)
        with self.tik_inst.elif_scope(self.element_obj.tiling_mode == 107):
            with self.tik_inst.if_scope(core_index < self.core_num - 1):
                self.compute_each_core_neg_1_less_than_block(core_idx=core_index, core_bs_num=self.each_core_bs_num)
            with self.tik_inst.else_scope():
                self.compute_each_core_neg_1_less_than_block(core_idx=core_index, core_bs_num=self.last_core_bs_num)
        with self.tik_inst.elif_scope(self.element_obj.tiling_mode == 108):
            with self.tik_inst.if_scope(core_index < self.core_num - 1):
                self.compute_each_core_less_batch(core_idx=core_index, core_bs_num=self.each_core_bs_num)
            with self.tik_inst.else_scope():
                self.compute_each_core_less_batch(core_idx=core_index, core_bs_num=self.last_core_bs_num)
        with self.tik_inst.elif_scope(self.element_obj.tiling_mode == 109):
            with self.tik_inst.if_scope(core_index < self.core_num - 1):
                self.compute_each_core_one_batch(core_idx=core_index, core_bs_num=self.each_core_bs_num)
            with self.tik_inst.else_scope():
                self.compute_each_core_one_batch(core_idx=core_index, core_bs_num=self.last_core_bs_num)
        with self.tik_inst.elif_scope(self.element_obj.tiling_mode == 110):
            self.compute_each_core_large_continous(core_idx=core_index)
        with self.tik_inst.elif_scope(self.element_obj.tiling_mode == 111):
            with self.tik_inst.if_scope(core_index < self.core_num - 1):
                self.compute_each_core_large_batch_pad(core_idx=core_index, core_bs_num=self.each_core_bs_num)
            with self.tik_inst.else_scope():
                self.compute_each_core_large_batch_pad(core_idx=core_index, core_bs_num=self.last_core_bs_num)
        with self.tik_inst.elif_scope(self.element_obj.tiling_mode == 112):
            with self.tik_inst.if_scope(core_index < self.core_num - 1):
                self.compute_each_core_large_ele_pad(core_idx=core_index, core_bs_num=self.each_core_bs_num)
            with self.tik_inst.else_scope():
                self.compute_each_core_large_ele_pad(core_idx=core_index, core_bs_num=self.last_core_bs_num)

    def _set_dst_offset_neg1(self, core_idx, each_core_bs_idx, index_ub):
        with self.tik_inst.if_scope(self.indices_shape_rank == 2):
            index_idx = (core_idx * self.each_core_bs_num + each_core_bs_idx) // self.num_head
            self.bs_idx.set_as(index_ub[2 * index_idx])
            self.valid_idx.set_as(index_ub[2 * index_idx + 1])
            actual_bs_idx = self.bs_idx * self.num_head + \
                            (core_idx * self.each_core_bs_num + each_core_bs_idx) % self.num_head
            self.dst_offset.set_as(actual_bs_idx * self.dst_bs_stride + self.valid_idx)
        with self.tik_inst.else_scope():
            index_idx = core_idx * self.each_core_bs_num + each_core_bs_idx
            self.bs_idx.set_as(index_idx)
            self.valid_idx.set_as(index_ub[index_idx // self.num_head])
            self.dst_offset.set_as(self.bs_idx * self.dst_bs_stride + self.valid_idx)

    def _set_dst_offset_neg2(self, core_idx, each_core_bs_idx, index_ub):
        with self.tik_inst.if_scope(self.indices_shape_rank == 2):
            index_idx = (core_idx * self.each_core_bs_num + each_core_bs_idx) // self.num_head
            self.bs_idx.set_as(index_ub[2 * index_idx])
            self.valid_idx.set_as(index_ub[2 * index_idx + 1])
            actual_bs_idx = self.bs_idx * self.num_head + \
                            (core_idx * self.each_core_bs_num + each_core_bs_idx) % self.num_head
            self.dst_offset.set_as(actual_bs_idx * self.dst_bs_stride + self.valid_idx * self.size_per_head)
        with self.tik_inst.else_scope():
            index_idx = core_idx * self.each_core_bs_num + each_core_bs_idx
            self.bs_idx.set_as(index_idx)
            self.valid_idx.set_as(index_ub[index_idx // self.num_head])
            self.dst_offset.set_as(self.bs_idx * self.dst_bs_stride + self.valid_idx * self.size_per_head)

    def _set_dst_offset_neg110(self, core_idx, each_core_bs, index_ub, update_dim0_idx, update_dim1_idx,
                               dst_fir_bs_stide):
        index_idx = update_dim0_idx
        with self.tik_inst.if_scope(self.indices_shape_rank == 2):
            self.bs_idx.set_as(index_ub[2 * index_idx])
            self.valid_idx.set_as(index_ub[2 * index_idx + 1])
            actual_bs_idx = self.bs_idx * dst_fir_bs_stide + update_dim1_idx * self.dst_bs_stride
            self.dst_offset.set_as(actual_bs_idx + (self.valid_idx + each_core_bs * core_idx) * self.src_bs_stride)
        with self.tik_inst.else_scope():
            self.bs_idx.set_as(index_ub[index_idx])
            actual_bs_idx = index_idx * dst_fir_bs_stide + update_dim1_idx * self.dst_bs_stride
            self.dst_offset.set_as(actual_bs_idx + (self.bs_idx + each_core_bs * core_idx) * self.src_bs_stride)
