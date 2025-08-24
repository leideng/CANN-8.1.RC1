#!/usr/bin/env python
# coding: utf-8
# Copyright 2021 Huawei Technologies Co., Ltd
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
top_k_large
"""
import te.platform as tbe_platform
from impl.ascend import AContainer
from impl.merge_sort import CommonMethod
from impl.merge_sort import MergeSort as MergeSortV1
from impl.merge_sort_v2 import BaseConstant
from impl.merge_sort_v2 import MergeSort as MergeSortV2
from impl import constant_util as constant


def _prod(values):
    """
    Prod the input values by multiply.
    """
    res = 1
    for value in values:
        res *= value
    return res


class Base:
    def __init__(self, input_shape, indices_shape, out_shape, k_num, input_dtype,
                 input_indices_dtype, out_indices_dtype, largest, kernel_name):
        self.input_shape = input_shape
        self.index_shape = indices_shape
        self.out_shape = out_shape
        self.rows = _prod(input_shape[:-1])
        self.cols = input_shape[-1]
        self.index_num = indices_shape[0]
        self.k_num = k_num
        self.dtype = input_dtype
        self.index_dtype = input_indices_dtype
        self.out_index_dtype = out_indices_dtype
        self.kernel_name = kernel_name
        self.largest = largest
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        AContainer.reset_instance()
        self.cont = AContainer.get_instance()
        self.tik = self.cont.tik
        self.tik_instance = self.cont.tinst
        self.method = CommonMethod(self.cont)

        self.input_data = self.tik_instance.Tensor(self.dtype, self.input_shape,
                                                   name="input_data", scope=self.tik.scope_gm)
        self.input_index = self.tik_instance.Tensor(self.index_dtype, self.index_shape,
                                                    name="input_index", scope=self.tik.scope_gm)
        self.output_data = self.tik_instance.Tensor(self.dtype, self.out_shape,
                                                    name="output_data", scope=self.tik.scope_gm)
        self.output_index = self.tik_instance.Tensor(self.out_index_dtype, self.out_shape,
                                                     name="output_index", scope=self.tik.scope_gm)

    def _get_aicore_num(self):
        loop_num = 1
        ai_core_num = self.rows
        if ai_core_num > self.core_num:
            ai_core_num = self.core_num
            loop_num = self.method.ceil_div(self.rows, ai_core_num)
        return loop_num, ai_core_num

    def _emit_vmuls(self, dst, src, cnt):
        """
        emit vmuls
        """
        repeat = cnt // constant.MASK128
        repeat_remain = cnt % constant.MASK128
        times = (repeat + constant.MAX_REPEAT_TIMES - 1) // constant.MAX_REPEAT_TIMES
        if repeat > 0:
            with self.tik_instance.for_range(0, times, name="vmuls_i0") as i:
                src0_scalar = self.tik_instance.Scalar(dtype="int64", name="src0_scalar",
                                                       init_value=repeat - i * constant.MAX_REPEAT_TIMES)
                src1_scalar = self.tik_instance.Scalar(dtype="int64",
                                                       name="src1_scalar",
                                                       init_value=constant.MAX_REPEAT_TIMES)
                times_len = self.tik_instance.Scalar(dtype="int64", name="times_len")
                self.tik_instance.scalar_min(times_len, src0_scalar, src1_scalar)
                self.tik_instance.vmuls(constant.MASK128,
                                        dst[i * constant.MASK128 * constant.MAX_REPEAT_TIMES],
                                        src[i * constant.MASK128 * constant.MAX_REPEAT_TIMES],
                                        -1, times_len, 1, 1, 8, 8)
        if repeat_remain > 0:
            self.tik_instance.vmuls(repeat_remain, dst[repeat * constant.MASK128],
                                    src[repeat * constant.MASK128], -1, 1, 1, 1, 8, 8)


class SegmentSort(Base):
    def __init__(self, input_shape, indices_shape, out_shape, k_num, input_dtype,
                 input_indices_dtype, out_indices_dtype, largest, kernel_name):
        super(SegmentSort, self).__init__(input_shape, indices_shape, out_shape, k_num, input_dtype,
                                          input_indices_dtype, out_indices_dtype, largest, kernel_name)
        self.ub_size = self.cont.const_ub_max_byte
        self.pro_data_num = self.cont.const_proposal_data_num
        self.pro_repeat_num = self.cont.const_proposal_repeat_num
        self.fp16_ne_inf = -(2 ** 16 - 1.0)
        self.merge_sort = MergeSortV1(self.cont, self.dtype, self.ub_size)
        self.int_data_size, self.int_block_data_num, self.int_repeat_data_num = \
            self.method.get_type_const(self.out_index_dtype)
        self.data_size, self.block_data_num, self.repeat_data_num = self.method.get_type_const(self.dtype)
        self.ub_pro_num_max, self.ub_sort_num, self.each_loop_index_num = \
            self.merge_sort.get_pro_num_info(self.index_num)
        self.align_cols = max(self.cols + self.pro_repeat_num, self.ub_pro_num_max)

        k_num_align = self.method.get_align_num(self.k_num + self.pro_repeat_num, self.pro_repeat_num)
        if k_num_align <= self.ub_sort_num:
            parts = self.method.ceil_div(self.align_cols, self.ub_pro_num_max)
            new_cols = min(parts * k_num_align, self.ub_pro_num_max)
            self.temp_proposal_1 = self.tik_instance.Tensor(self.dtype, (self.rows, new_cols, self.pro_data_num),
                                                            name="temp_proposal_1", scope=self.tik.scope_gm,
                                                            is_workspace=True)
            self.temp_proposal_2 = self.tik_instance.Tensor(self.dtype, (self.rows, new_cols, self.pro_data_num),
                                                            name="temp_proposal_2", scope=self.tik.scope_gm,
                                                            is_workspace=True)
        else:
            self.temp_proposal_1 = self.tik_instance.Tensor(self.dtype, (self.rows, self.align_cols, self.pro_data_num),
                                                            name="temp_proposal_1", scope=self.tik.scope_gm,
                                                            is_workspace=True)
            self.temp_proposal_2 = self.tik_instance.Tensor(self.dtype, (self.rows, self.align_cols, self.pro_data_num),
                                                            name="temp_proposal_2", scope=self.tik.scope_gm,
                                                            is_workspace=True)

    def mode_compute(self):
        loop_num, ai_core_num = self._get_aicore_num()
        multi_core = True
        if self.k_num < self.block_data_num:
            ai_core_num = 1
            loop_num = self.rows
            multi_core = False
        with self.tik_instance.for_range(0, ai_core_num, block_num=ai_core_num) as core_idx:
            with self.tik_instance.for_range(0, loop_num) as loop_idx:
                batch_idx = core_idx * loop_num + loop_idx
                with self.tik_instance.if_scope(batch_idx < self.rows):
                    self._mode_compute_each_core(batch_idx, multi_core)

        self.tik_instance.BuildCCE(inputs=[self.input_data, self.input_index],
                                   outputs=[self.output_data, self.output_index],
                                   kernel_name=self.kernel_name)

    def _mode_compute_each_core(self, batch_idx, multi_core):
        index_l1 = self._index_move_in()
        if self.cols <= self.ub_pro_num_max or self.k_num <= self.ub_sort_num:
            self.merge_sort.get_top_proposal(self.temp_proposal_1, batch_idx, self.cols, self.k_num,
                                             self._get_proposal_ub, (index_l1, batch_idx))
        else:
            self.merge_sort.get_top_proposal_large(self.temp_proposal_1, self.temp_proposal_2,
                                                   batch_idx, self.cols, self.k_num,
                                                   self._get_proposal_ub, (index_l1, batch_idx))

        with self.tik_instance.new_stmt_scope():
            each_loop_data_num = self._get_loop_data_num()
            loop_time, last_loop_data_num = self.method.get_loop_info(self.k_num, each_loop_data_num)
            with self.tik_instance.for_range(0, loop_time) as loop_idx:
                data_index_start = loop_idx * each_loop_data_num
                with self.tik_instance.if_scope(loop_idx != loop_time - 1):
                    self._result_move_out_each_loop(batch_idx, data_index_start, each_loop_data_num, multi_core)
                with self.tik_instance.else_scope():
                    self._result_move_out_each_loop(batch_idx, data_index_start, last_loop_data_num, multi_core)

    def _index_move_in(self):
        index_l1 = self.tik_instance.Tensor(self.index_dtype, (self.index_num,),
                                            name="index_l1", scope=self.tik.scope_cbuf)
        block_num = self.method.ceil_div(self.index_num, self.block_data_num)
        self.tik_instance.data_move(index_l1, self.input_index, 0, 1, block_num, 0, 0)
        return index_l1

    def _get_proposal_ub(self, boxes_num, start_index, index_l1, batch_idx):
        proposal_num = self.method.get_align_num(boxes_num, self.pro_repeat_num)
        ub_proposal_1 = self.tik_instance.Tensor(self.dtype, (proposal_num, self.pro_data_num),
                                                 name="ub_proposal_1", scope=self.tik.scope_ubuf)
        with self.tik_instance.new_stmt_scope():
            self._init_index_channel(ub_proposal_1, index_l1, start_index, proposal_num)
            self._init_score_channel(ub_proposal_1, start_index, boxes_num, proposal_num, batch_idx)

        ub_proposal_2 = self.tik_instance.Tensor(self.dtype, (proposal_num, self.pro_data_num),
                                                 name="ub_proposal_2", scope=self.tik.scope_ubuf)
        return ub_proposal_1, ub_proposal_2

    def _init_index_channel(self, ub_proposal_1, index_l1, boxes_index, proposal_num):
        """
        algorithm: init index
            ub_proposal_1[:, 0] = index % each_loop_index_num
            ub_proposal_1[:, 1] = index // each_loop_index_num % each_loop_index_num
            ub_proposal_1[:, 2] = index // (each_loop_index_num * each_loop_index_num)
        """
        index_channel_0, index_channel_1, index_channel_2 = 0, 1, 2
        index_shape = (self.ub_pro_num_max,)
        index_ub_0 = self.tik_instance.Tensor(self.index_dtype, index_shape, self.tik.scope_ubuf, "index_ub_0")
        index_ub_1 = self.tik_instance.Tensor(self.index_dtype, index_shape, self.tik.scope_ubuf, "index_ub_1")
        index_ub_2 = self.tik_instance.Tensor(self.index_dtype, index_shape, self.tik.scope_ubuf, "index_ub_2")

        loop_time = self.ub_pro_num_max // self.each_loop_index_num
        index_block_num = self.each_loop_index_num // self.block_data_num
        self.tik_instance.data_move(index_ub_0, index_l1, 0, 1, index_block_num, 0, 0)

        with self.tik_instance.for_range(1, loop_time) as loop_index:
            index_ub_0_index = loop_index * self.each_loop_index_num
            self.tik_instance.data_move(index_ub_0[index_ub_0_index], index_ub_0, 0, 1, index_block_num, 0, 0)

        loop_time_align = self.method.get_align_num(loop_time, self.block_data_num)
        index_fp16_ub = self.tik_instance.Tensor(self.index_dtype, (2, loop_time_align),
                                                 self.tik.scope_ubuf, "index_fp16_ub")
        pow_index = self.each_loop_index_num * self.each_loop_index_num
        index_1_s = self.tik_instance.Scalar(dtype="int32")
        index_2_s = self.tik_instance.Scalar(dtype="int32")
        index_1_s.set_as((boxes_index // self.each_loop_index_num) % self.each_loop_index_num)
        index_2_s.set_as(boxes_index // pow_index)
        index_1_block_num = loop_time_align // self.block_data_num
        index_2_block_num = 1

        self.tik_instance.data_move(index_fp16_ub[0, 0], self.input_index[index_1_s], 0, 1, index_1_block_num, 0, 0)
        self.tik_instance.data_move(index_fp16_ub[1, 0], self.input_index[index_2_s], 0, 1, index_2_block_num, 0, 0)

        index_fp16_scalar = self.tik_instance.Scalar(dtype=self.index_dtype)
        index_fp16_scalar.set_as(index_fp16_ub[1, 0])
        self.method.vector_dup(index_ub_2, index_fp16_scalar)

        with self.tik_instance.for_range(0, loop_time) as loop_index:
            index_fp16_scalar.set_as(index_fp16_ub[0, loop_index])
            self.method.vector_dup(index_ub_1, index_fp16_scalar,
                                   self.each_loop_index_num,
                                   self.each_loop_index_num * loop_index)

        self.method.vector_concat(ub_proposal_1, index_ub_0, index_channel_0, proposal_num)
        self.method.vector_concat(ub_proposal_1, index_ub_1, index_channel_1, proposal_num)
        self.method.vector_concat(ub_proposal_1, index_ub_2, index_channel_2, proposal_num)

    def _init_score_channel(self, ub_proposal_1, boxes_index, boxes_num, proposal_num, batch_idx):
        score_shape = (self.ub_pro_num_max,)
        score_ub = self.tik_instance.Tensor(self.dtype, score_shape, self.tik.scope_ubuf, "score_ub")
        score_label_0, score_label_1 = 3, 4
        score_block_num = self.method.ceil_div(boxes_num, self.block_data_num)
        input_offset = batch_idx * self.cols + boxes_index
        self.tik_instance.data_move(score_ub, self.input_data[input_offset], 0, 1, score_block_num, 0, 0)
        if not self.largest:
            self._emit_vmuls(score_ub, score_ub, boxes_num)
        mask_h, mask_l, index_last = self.method.get_mask(boxes_num, self.repeat_data_num, self.pro_repeat_num)
        if mask_h != 0 or mask_l != 0:
            self.tik_instance.vector_dup([mask_h, mask_l], score_ub[index_last], self.fp16_ne_inf, 1, 1, 8)
        self.method.vector_concat(ub_proposal_1, score_ub, score_label_0, proposal_num)
        self.method.vector_concat(ub_proposal_1, score_ub, score_label_1, proposal_num)

    def _get_loop_data_num(self):
        each_data_size = self.data_size * (4 + self.pro_data_num) + self.int_data_size * 3
        data_num = self.ub_size // each_data_size
        data_num_align = self.method.get_align_num(data_num, self.int_repeat_data_num, False)
        return data_num_align

    def _result_move_out_each_loop(self, batch_idx, data_index_start, data_num, multi_core):
        """
        algorithm: get result_data, result_index
            result_data = gm_tensor[0, :, 3]
            result_index = gm_tensor[0, :, 0] + gm_tensor[0, :, 0] * each_loop_index_num
                           + gm_tensor[0, :, 0] * each_loop_index_num * each_loop_index_num
        """
        if multi_core and data_num < self.block_data_num:
            data_index_start = data_index_start - self.block_data_num + data_num
            data_num = self.block_data_num
        proposal_num = self.method.get_align_num(data_num, self.pro_repeat_num)
        data_num_align = self.method.get_align_num(data_num, self.int_repeat_data_num)
        data_shape = (data_num_align, )
        index_ub_0 = self.tik_instance.Tensor(self.index_dtype, data_shape, self.tik.scope_ubuf, "index_ub_0")
        index_ub_1 = self.tik_instance.Tensor(self.index_dtype, data_shape, self.tik.scope_ubuf, "index_ub_1")
        index_ub_2 = self.tik_instance.Tensor(self.index_dtype, data_shape, self.tik.scope_ubuf, "index_ub_2")
        score_ub = self.tik_instance.Tensor(self.index_dtype, data_shape, self.tik.scope_ubuf, "score_ub")

        index_int_ub_0 = self.tik_instance.Tensor(self.out_index_dtype, data_shape,
                                                  self.tik.scope_ubuf, "index_int_ub_0")
        index_int_ub_1 = self.tik_instance.Tensor(self.out_index_dtype, data_shape,
                                                  self.tik.scope_ubuf, "index_int_ub_1")
        index_int_ub_2 = self.tik_instance.Tensor(self.out_index_dtype, data_shape,
                                                  self.tik.scope_ubuf, "index_int_ub_2")

        proposal_shape = (proposal_num, self.pro_data_num)
        proposal_ub = self.tik_instance.Tensor(self.dtype, proposal_shape, self.tik.scope_ubuf, "proposal_ub")
        block_num_move_in = self.method.ceil_div(data_num * self.pro_data_num, self.block_data_num)

        self.tik_instance.data_move(proposal_ub, self.temp_proposal_1[batch_idx, data_index_start, 0],
                                    0, 1, block_num_move_in, 0, 0)
        self.method.vector_extract(index_ub_0, proposal_ub, 0, proposal_num)
        self.method.vector_extract(index_ub_1, proposal_ub, 1, proposal_num)
        self.method.vector_extract(index_ub_2, proposal_ub, 2, proposal_num)
        self.method.vector_extract(score_ub, proposal_ub, 3, proposal_num)

        if not self.largest:
            self._emit_vmuls(score_ub, score_ub, data_num)

        mask = self.int_repeat_data_num
        repeat_num = data_num_align // self.int_repeat_data_num

        self.tik_instance.vconv(mask, "round", index_int_ub_0, index_ub_0, repeat_num, 1, 1, 8, 4)
        self._get_int_index(index_int_ub_0, index_int_ub_1, index_ub_1, index_int_ub_2, mask, repeat_num,
                            self.each_loop_index_num)
        mul_num = self.each_loop_index_num * self.each_loop_index_num
        self._get_int_index(index_int_ub_0, index_int_ub_1, index_ub_2, index_int_ub_2, mask, repeat_num, mul_num)

        out_offset = batch_idx * self.k_num + data_index_start
        if not multi_core:
            data_block_num = self.method.ceil_div(data_num, self.block_data_num)
            index_block_num = self.method.ceil_div(data_num, self.int_block_data_num)
            self.tik_instance.data_move(self.output_data[out_offset], score_ub, 0, 1, data_block_num, 0, 0)
            self.tik_instance.data_move(self.output_index[out_offset], index_int_ub_0, 0, 1, index_block_num, 0, 0)
        else:
            data_block_num = data_num // self.block_data_num
            index_block_num = data_num // self.int_block_data_num
            if data_block_num > 0:
                self.tik_instance.data_move(self.output_data[out_offset], score_ub, 0, 1, data_block_num, 0, 0)
            if index_block_num > 0:
                self.tik_instance.data_move(self.output_index[out_offset], index_int_ub_0, 0, 1, index_block_num, 0, 0)

            if data_num % self.block_data_num != 0:
                for i in range(self.block_data_num):
                    score_ub[i] = score_ub[data_num - self.block_data_num + i]
                out_data_offset = out_offset + data_num - self.block_data_num
                self.tik_instance.data_move(self.output_data[out_data_offset], score_ub, 0, 1, 1, 0, 0)
            if data_num % self.int_block_data_num != 0:
                for i in range(self.int_block_data_num):
                    index_int_ub_0[i] = index_int_ub_0[data_num - self.int_block_data_num + i]
                out_index_offset = out_offset + data_num - self.int_block_data_num
                self.tik_instance.data_move(self.output_index[out_index_offset], index_int_ub_0, 0, 1, 1, 0, 0)

    def _get_int_index(self, result_index, index_int_ub, index_ub, index_int_ub_temp, mask, repeat_num, mul_num):
        self.tik_instance.vconv(mask, "round", index_int_ub, index_ub, repeat_num, 1, 1, 8, 4)
        self.tik_instance.vector_dup(mask, index_int_ub_temp, mul_num, repeat_num, 1, 8)
        self.tik_instance.vmul(mask, index_int_ub, index_int_ub, index_int_ub_temp, repeat_num, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(mask, result_index, result_index, index_int_ub, repeat_num, 1, 1, 1, 8, 8, 8)


class SegmentSortV2(Base):
    def __init__(self, input_shape, indices_shape, out_shape, k_num, input_dtype,
                 input_indices_dtype, out_indices_dtype, largest, kernel_name):
        super(SegmentSortV2, self).__init__(input_shape, indices_shape, out_shape, k_num, input_dtype,
                                            input_indices_dtype, out_indices_dtype, largest, kernel_name)
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.const_value = BaseConstant(self.dtype, "uint32", self.ub_size, kernel_name)
        self.merge_sort = MergeSortV2(self.tik, self.tik_instance, self.const_value)
        self.element_num_proposal = self.const_value.element_num_proposal
        self.pro_data_num = 4 if self.dtype == "float16" else 2
        self.ub_pro_num_max = self.const_value.proposal_num_max_ub
        self.align_cols = max(self.ub_pro_num_max, self.cols + self.const_value.proposal_num_repeat)

        self.temp_proposal_1 = self.tik_instance.Tensor(self.dtype, (self.rows, self.align_cols, self.pro_data_num),
                                                        name="temp_proposal_1", scope=self.tik.scope_gm,
                                                        is_workspace=True)
        self.temp_proposal_2 = self.tik_instance.Tensor(self.dtype, (self.rows, self.align_cols, self.pro_data_num),
                                                        name="temp_proposal_2", scope=self.tik.scope_gm,
                                                        is_workspace=True)

    def mode_compute(self):
        loop_num, ai_core_num = self._get_aicore_num()
        multi_core = True
        if self.k_num < self.const_value.score_num_block:
            ai_core_num = 1
            loop_num = self.rows
            multi_core = False
        with self.tik_instance.for_range(0, ai_core_num, block_num=ai_core_num) as core_idx:
            with self.tik_instance.for_range(0, loop_num) as loop_idx:
                batch_idx = core_idx * loop_num + loop_idx
                with self.tik_instance.if_scope(batch_idx < self.rows):
                    self._mode_compute_each_core(batch_idx, multi_core)

        self.tik_instance.BuildCCE(inputs=[self.input_data, self.input_index],
                                   outputs=[self.output_data, self.output_index],
                                   kernel_name=self.kernel_name)

    def _mode_compute_each_core(self, batch_idx, multi_core):
        fun_args = {
            "batch_index": batch_idx
        }

        self.merge_sort.merge_sort_start(self.temp_proposal_1, self.temp_proposal_2,
                                         batch_idx, batch_idx, self.k_num, self.cols,
                                         self._get_score_index_tensor, fun_args)

        each_loop_data_num = self._get_loop_data_num()
        loop_time, last_loop_data_num = self.method.get_loop_info(self.k_num, each_loop_data_num)
        with self.tik_instance.for_range(0, loop_time) as loop_idx:
            data_index_start = loop_idx * each_loop_data_num
            with self.tik_instance.if_scope(loop_idx != loop_time - 1):
                self._result_move_out_each_loop(batch_idx, data_index_start, each_loop_data_num, multi_core)
            with self.tik_instance.else_scope():
                self._result_move_out_each_loop(batch_idx, data_index_start, last_loop_data_num, multi_core)

    def _get_score_index_tensor(self, fun_args):
        batch_index = fun_args.get("batch_index")
        score_num_loop = fun_args.get("score_num_loop")
        proposal_num_loop = fun_args.get("proposal_num_loop")
        score_index = fun_args.get("score_index_loop")

        tensor_shape_ub = (proposal_num_loop,)
        index_ub = self.tik_instance.Tensor(self.out_index_dtype, tensor_shape_ub, self.tik.scope_ubuf, "index_ub")
        self._init_index_channel(index_ub, score_index, proposal_num_loop)

        score_ub = self.tik_instance.Tensor(self.dtype, tensor_shape_ub, self.tik.scope_ubuf, "score_ub")
        self._init_score_channel(score_ub, batch_index, score_index, score_num_loop)
        index_ub = index_ub.reinterpret_cast_to("uint32")
        return score_ub, index_ub

    def _init_index_channel(self, index_ub, score_index, proposal_num):
        mask = self.const_value.index_num_repeat
        if proposal_num <= mask:
            block_num = proposal_num // self.const_value.index_num_block
            self.tik_instance.data_move(index_ub, self.input_index, 0, 1, block_num, 0, 0)
            self.tik_instance.vadds(proposal_num, index_ub, index_ub, score_index, 1, 1, 1, 8, 8)
        else:
            index_num = self.method.get_align_num(min(self.index_num, proposal_num), mask, False)
            block_num = index_num // self.const_value.index_num_block
            repeat_num = index_num // mask
            self.tik_instance.data_move(index_ub, self.input_index, 0, 1, block_num, 0, 0)
            self.tik_instance.vadds(mask, index_ub, index_ub, score_index, repeat_num, 1, 1, 8, 8)
            while index_num < proposal_num:
                if proposal_num >= index_num * 2:
                    repeat_num = index_num // mask
                    self.tik_instance.vadds(mask, index_ub[index_num], index_ub, index_num, repeat_num, 1, 1, 8, 8)
                else:
                    proposal_num_not_add = proposal_num - index_num
                    repeat_num = proposal_num_not_add // mask
                    if repeat_num > 0:
                        self.tik_instance.vadds(mask, index_ub[index_num], index_ub, index_num, repeat_num, 1, 1, 8, 8)
                    proposal_num_last = proposal_num_not_add - repeat_num * mask
                    if proposal_num_last > 0:
                        start_index = repeat_num * mask + index_num
                        self.tik_instance.vadds(proposal_num_last, index_ub[start_index],
                                                index_ub, start_index, 1, 1, 1, 8, 8)
                index_num *= 2

    def _init_score_channel(self, score_ub, batch_index, score_index, score_num):
        block_num = self.method.ceil_div(score_num, self.const_value.score_num_block)
        input_offset = batch_index * self.cols + score_index
        self.tik_instance.data_move(score_ub, self.input_data[input_offset], 0, 1, block_num, 0, 0)
        if not self.largest:
            self._emit_vmuls(score_ub, score_ub, score_num)
        align_num = self.const_value.proposal_num_repeat
        mask_h, mask_l, index_last = self.method.get_mask(score_num, align_num, align_num)
        if mask_h != 0 or mask_l != 0:
            self.tik_instance.vector_dup([mask_h, mask_l], score_ub[index_last], self.const_value.neg_inf, 1, 1, 8)

    def _get_loop_data_num(self):
        each_data_size = self.const_value.proposal_size + self.const_value.score_size + self.const_value.index_size
        data_num = self.ub_size // each_data_size
        max_data_num = (self.const_value.repeat_num_max_cmd * self.const_value.score_num_repeat
                        // self.element_num_proposal)
        data_num = min(max_data_num, data_num)
        data_num_align = self.method.get_align_num(data_num, self.const_value.score_num_repeat, False)
        return data_num_align

    def _result_move_out_each_loop(self, batch_idx, proposal_index_start, proposal_num, multi_core):
        if multi_core and proposal_num < self.const_value.score_num_block:
            proposal_index_start = proposal_index_start - self.const_value.score_num_block + proposal_num
            proposal_num = self.const_value.score_num_block
        proposal_num_align = self.method.get_align_num(proposal_num, self.const_value.score_num_repeat)
        element_num_move_in = proposal_num * self.element_num_proposal
        element_num_align = proposal_num_align * self.element_num_proposal

        proposal_shape_ub = (element_num_align,)
        proposal_ub = self.tik_instance.Tensor(self.dtype, proposal_shape_ub, self.tik.scope_ubuf, "proposal_ub")
        block_num_move_in = self.method.ceil_div(element_num_move_in, self.const_value.score_num_block)
        self.tik_instance.data_move(proposal_ub, self.temp_proposal_1[batch_idx, proposal_index_start, 0],
                                    0, 1, block_num_move_in, 0, 0)

        score_shape = (proposal_num_align,)
        index_type = "uint32"
        score_ub = self.tik_instance.Tensor(self.dtype, score_shape, self.tik.scope_ubuf, "score_ub")
        index_ub = self.tik_instance.Tensor(index_type, score_shape, self.tik.scope_ubuf, "index_ub")
        if self.dtype == "float16":
            score_src1_pattern = 3
        else:
            score_src1_pattern = 1
        index_src1_pattern = 2
        repeat_time = element_num_align // self.const_value.score_num_repeat
        self.tik_instance.vreduce(self.const_value.score_num_repeat, score_ub, proposal_ub,
                                  score_src1_pattern, repeat_time, 1, 8, 0)
        index_proposal_ub = proposal_ub.reinterpret_cast_to(index_type)
        self.tik_instance.vreduce(self.const_value.index_num_repeat, index_ub, index_proposal_ub,
                                  index_src1_pattern, repeat_time, 1, 8, 0)
        index_ub = index_ub.reinterpret_cast_to(self.out_index_dtype)

        if not self.largest:
            self._emit_vmuls(score_ub, score_ub, proposal_num)

        out_offset = batch_idx * self.k_num + proposal_index_start
        if not multi_core:
            score_block_num = self.method.ceil_div(proposal_num, self.const_value.score_num_block)
            index_block_num = self.method.ceil_div(proposal_num, self.const_value.index_num_block)
            self.tik_instance.data_move(self.output_data[out_offset], score_ub, 0, 1, score_block_num, 0, 0)
            self.tik_instance.data_move(self.output_index[out_offset], index_ub, 0, 1, index_block_num, 0, 0)
        else:
            score_block_num = proposal_num // self.const_value.score_num_block
            index_block_num = proposal_num // self.const_value.index_num_block
            if score_block_num > 0:
                self.tik_instance.data_move(self.output_data[out_offset], score_ub, 0, 1, score_block_num, 0, 0)
            if index_block_num > 0:
                self.tik_instance.data_move(self.output_index[out_offset], index_ub, 0, 1, index_block_num, 0, 0)

            if proposal_num % self.const_value.score_num_block != 0:
                for i in range(self.const_value.score_num_block):
                    score_ub[i] = score_ub[proposal_num - self.const_value.score_num_block + i]
                out_data_offset = out_offset + proposal_num - self.const_value.score_num_block
                self.tik_instance.data_move(self.output_data[out_data_offset], score_ub, 0, 1, 1, 0, 0)
            if proposal_num % self.const_value.index_num_block != 0:
                for i in range(self.const_value.index_num_block):
                    index_ub[i] = index_ub[proposal_num - self.const_value.index_num_block + i]
                out_index_offset = out_offset + proposal_num - self.const_value.index_num_block
                self.tik_instance.data_move(self.output_index[out_index_offset], index_ub, 0, 1, 1, 0, 0)


def top_k_large(input_shape, indices_shape, out_shape, k, input_dtype,
                input_indices_dtype, out_indices_dtype, largest, kernel_name):
    if tbe_platform.api_check_support("tik.vbitsort32"):
        obj = SegmentSortV2(input_shape, indices_shape, out_shape, k, input_dtype, input_indices_dtype,
                            out_indices_dtype, largest, kernel_name)
    else:
        obj = SegmentSort(input_shape, indices_shape, out_shape, k, input_dtype, input_indices_dtype,
                          out_indices_dtype, largest, kernel_name)
    obj.mode_compute()
