#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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
batch_multi_class_non_max_suppression_norm_class
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.util_common import is_vector_core


def ceil_div(dividend, divisor):
    return (dividend + divisor - 1) // divisor


def get_align_num(data_num, align_num):
    return ceil_div(data_num, align_num) * align_num


def get_loop_info(all_data_num, each_loop_num):
    loop_times = ceil_div(all_data_num, each_loop_num)
    last_loop_num = all_data_num - each_loop_num * (loop_times - 1)
    return loop_times, last_loop_num


class TikComputeInterface:

    def __init__(self, tik_inst):
        self.tik_inst = tik_inst
        self.element_proposal = 8
        self.max_repeat = 255

    def start_tik_compute(self, data_num, mask, func, args=None, begin_index=0):
        """
        start_tik_compute
        """
        repeat_num = data_num // mask
        last_num = data_num - repeat_num * mask
        loop_times, last_repeat = get_loop_info(repeat_num, self.max_repeat)
        for loop_index in range(loop_times):
            start_index = loop_index * self.max_repeat * mask + begin_index
            if loop_index != (loop_times - 1):
                func(mask, start_index, self.max_repeat, args)
            else:
                func(mask, start_index, last_repeat, args)
        if last_num != 0:
            start_index = repeat_num * mask + begin_index
            func(last_num, start_index, 1, args)

    # 'pylint: disable = unused-argument
    def tik_vconcat(self, mask, start_index, repeat_num, args):
        """
        tik_vconcat
        """
        dst, src, mode_number, dst_index, src_index = args
        dst_flatten = dst.reshape((dst.size,))
        src_flatten = src.reshape((src.size,))
        self.tik_inst.vconcat(dst_flatten[dst_index + start_index * self.element_proposal],
                              src_flatten[src_index + start_index],
                              repeat_num, mode_number)

    # 'pylint: disable = unused-argument
    def tik_vextract(self, mask, start_index, repeat_num, args):
        """
        tik_vextract
        """
        dst, src, mode_number, dst_index, src_index = args
        dst_flatten = dst.reshape((dst.size,))
        src_flatten = src.reshape((src.size,))
        self.tik_inst.vextract(dst_flatten[src_index + start_index],
                               src_flatten[dst_index + start_index * self.element_proposal],
                               repeat_num, mode_number)

    def tik_dup(self, mask, start_index, repeat_num, args):
        """
        tik_dup
        """
        dst, dup_data = args
        dst_flatten = dst.reshape((dst.size,))
        self.tik_inst.vector_dup(mask, dst_flatten[start_index], dup_data, repeat_num, 1, 8)

    def tik_vmuls(self, mask, start_index, repeat_num, args):
        """
        tik_vmuls
        """
        dst, src0, scalar, dst_index, src0_index = args
        dst_flatten = dst.reshape((dst.size,))
        src0_flatten = src0.reshape((src0.size,))
        self.tik_inst.vmuls(mask, dst_flatten[dst_index + start_index], src0_flatten[src0_index + start_index], scalar,
                            repeat_num, 1, 1, 8, 8)

    def tik_vadds(self, mask, start_index, repeat_num, args):
        """
        tik_vadds
        """
        dst, src0, scalar, dst_index, src0_index = args
        dst_flatten = dst.reshape((dst.size,))
        src0_flatten = src0.reshape((src0.size,))
        self.tik_inst.vadds(mask, dst_flatten[dst_index + start_index],
                            src0_flatten[src0_index + start_index],
                            scalar,
                            repeat_num, 1, 1, 8, 8)

    # 'pylint: disable = unused-argument
    # 'pylint: disable=too-many-arguments,too-many-locals
    def tik_vrpsort16(self, mask, start_index, repeat_num, args):
        """
        tik_vrpsort16
        """
        dst, src = args
        dst_flatten = dst.reshape((dst.size,))
        src_flatten = src.reshape((src.size,))
        self.tik_inst.vrpsort16(dst_flatten[start_index * self.element_proposal],
                                src_flatten[start_index * self.element_proposal],
                                repeat_num)


class BatchMultiClassNonMaxSuppressionNormClass:

    def __init__(self, batch_num, class_num, boxes_num, score_threshold, iou_threshold, max_size_per_class,
                 max_total_size, image_size, kernel_name):
        self.batch_num = batch_num
        self.class_num = class_num
        self.boxes_num = boxes_num
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.max_size_per_class = max_size_per_class
        self.max_total_size = max_total_size
        self.image_size = image_size
        self.kernel_name = kernel_name

        self.tik_inst = tik.Tik()
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

        self.tik_func = TikComputeInterface(self.tik_inst)
        self.merge_sort = MergeSort(self.tik_inst)
        self.get_boxes_num = GetBoxesNum(self.tik_inst, self.tik_func)
        self.nms = NonMaximumSuppression(self.tik_inst, self.tik_func)

        self.max_sort_size = 4096
        self.proposal_num_max_ub = 8192
        self.data_dtype = "float16"
        self.num_block_data = 16
        self.num_repeat_data = 128
        self.num_repeat_proposal = 16
        self.element_proposal = 8
        self.pro_num_block = 32 // (8 * 2)
        self.merge_channel_num = 4
        input_boxes_shape = (batch_num, 4, boxes_num)
        input_score_shape = (batch_num, class_num, boxes_num)
        output_boxes_shape = (batch_num, 4, max_total_size)
        output_score_shape = (batch_num, max_total_size)
        self.output_num_dtype = "int32"
        output_num_shape = (batch_num, 8)
        self.input_boxes = self.tik_inst.Tensor(self.data_dtype, input_boxes_shape, tik.scope_gm, "input_boxes")
        self.input_score = self.tik_inst.Tensor(self.data_dtype, input_score_shape, tik.scope_gm, "input_score")

        self.output_boxes = self.tik_inst.Tensor(self.data_dtype, output_boxes_shape, tik.scope_gm, "output_boxes")
        self.output_score = self.tik_inst.Tensor(self.data_dtype, output_score_shape, tik.scope_gm, "output_score")
        self.output_label = self.tik_inst.Tensor(self.data_dtype, output_score_shape, tik.scope_gm, "output_label")
        self.output_num = self.tik_inst.Tensor(self.output_num_dtype, output_num_shape, tik.scope_gm, "output_num")
        self.is_vector_core = is_vector_core()
        if self.is_vector_core:
            proposal_l1_shape = (self.batch_num, self.class_num * self.max_size_per_class, self.element_proposal)
            self.temp_work_space = self.tik_inst.Tensor(self.data_dtype, proposal_l1_shape, tik.scope_gm,
                                                        "temp_work_space", is_workspace=True)

    def start(self):
        """
        start
        """
        each_core_batch_num = ceil_div(self.batch_num, self.core_num)
        self.core_num, last_core_batch_num = get_loop_info(self.batch_num, each_core_batch_num)
        with self.tik_inst.for_range(0, self.core_num, block_num=self.core_num) as core_index:
            batch_start_index_core = core_index * each_core_batch_num
            with self.tik_inst.if_scope(core_index < self.core_num - 1):
                self.start_each_core(batch_start_index_core, each_core_batch_num)
            with self.tik_inst.else_scope():
                self.start_each_core(batch_start_index_core, last_core_batch_num)
        self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                               inputs=[self.input_boxes, self.input_score],
                               outputs=[self.output_boxes, self.output_score, self.output_label, self.output_num])

    def start_each_core(self, batch_start_index_core, batch_num):
        """
        start_each_core
        """
        with self.tik_inst.for_range(0, batch_num) as batch_index:
            batch_start_index_loop = batch_start_index_core + batch_index
            self.start_each_batch(batch_start_index_loop)

    def start_each_batch(self, batch_index):
        """
        start_each_batch
        """
        proposal_l1_shape = (1, self.class_num * self.max_size_per_class, self.element_proposal)
        if self.is_vector_core:
            proposal_l1 = self.temp_work_space
            batch_index_l1 = batch_index
        else:
            proposal_l1 = self.tik_inst.Tensor(self.data_dtype, proposal_l1_shape, tik.scope_cbuf, "proposal_l1")
            batch_index_l1 = 0

        self.sort_each_class(proposal_l1, batch_index_l1, batch_index)

        self.sort_all_class(proposal_l1, batch_index_l1)

        pro_num = self.tik_inst.Scalar("int32", "pro_num")
        self.get_boxes_num.get_boxes_num(proposal_l1, batch_index_l1, pro_num, self.max_sort_size, self.score_threshold)

        select_pro_ub, select_num_scalar = self.nms.non_maximum_suppression(proposal_l1, batch_index_l1, pro_num,
                                                                            self.max_total_size, self.iou_threshold,
                                                                            self.image_size)

        max_sort_size_align = get_align_num(self.max_total_size, self.num_repeat_proposal)
        vextract_repeat = max_sort_size_align // self.num_repeat_proposal
        result_boxes = self.tik_inst.Tensor(self.data_dtype, (4, max_sort_size_align), tik.scope_ubuf, "result_boxes")
        result_score = self.tik_inst.Tensor(self.data_dtype, (max_sort_size_align,), tik.scope_ubuf, "result_score")
        result_label = self.tik_inst.Tensor(self.data_dtype, (max_sort_size_align,), tik.scope_ubuf, "result_label")
        result_num = self.tik_inst.Tensor(self.output_num_dtype, (8,), tik.scope_ubuf, "result_num")

        self.tik_inst.vextract(result_boxes[0, 0], select_pro_ub, vextract_repeat, 0)
        self.tik_inst.vextract(result_boxes[1, 0], select_pro_ub, vextract_repeat, 1)
        self.tik_inst.vextract(result_boxes[2, 0], select_pro_ub, vextract_repeat, 2)
        self.tik_inst.vextract(result_boxes[3, 0], select_pro_ub, vextract_repeat, 3)

        self.tik_inst.vextract(result_score, select_pro_ub, vextract_repeat, 4)
        self.tik_inst.vextract(result_label, select_pro_ub, vextract_repeat, 5)
        self.tik_inst.vector_dup(8, result_num, 0, 1, 1, 8)
        result_num[0].set_as(select_num_scalar)

        boxes_block = 4 * max_sort_size_align // self.num_block_data
        score_block = max_sort_size_align // self.num_block_data
        self.tik_inst.data_move(self.output_boxes[batch_index:, :, :], result_boxes, 0, 1, boxes_block, 0, 0)
        self.tik_inst.data_move(self.output_score[batch_index:, :], result_score, 0, 1, score_block, 0, 0)
        self.tik_inst.data_move(self.output_label[batch_index:, :], result_label, 0, 1, score_block, 0, 0)
        self.tik_inst.data_move(self.output_num[batch_index:, :], result_num, 0, 1, 1, 0, 0)

    def init_proposal_ub_loop(self, proposal_ub_0, batch_index, class_start_index, class_num):
        """
        init_proposal_ub_loop
        """
        with self.tik_inst.new_stmt_scope():
            self.init_proposal_boxes(proposal_ub_0, batch_index, class_num)
        with self.tik_inst.new_stmt_scope():
            self.init_proposal_scores(proposal_ub_0, batch_index, class_start_index, class_num)
        with self.tik_inst.new_stmt_scope():
            self.init_proposal_label(proposal_ub_0, class_start_index, class_num)
        return proposal_ub_0

    def init_proposal_boxes(self, proposal_ub, batch_index, class_num):
        """
        init_proposal_boxes
        """
        boxes_shape = (4, self.boxes_num)
        boxes_ub = self.tik_inst.Tensor(self.data_dtype, boxes_shape, tik.scope_ubuf, "boxes_ub")
        boxes_move_block_num = 4 * self.boxes_num // self.num_block_data
        self.tik_inst.data_move(boxes_ub, self.input_boxes[batch_index:, :, :], 0, 1, boxes_move_block_num, 0, 0)

        for i in range(4):
            self.tik_func.start_tik_compute(self.boxes_num, self.num_repeat_proposal, self.tik_func.tik_vconcat,
                                            args=(proposal_ub, boxes_ub, i, 0, i * self.boxes_num))
        proposal_move_block_num = self.element_proposal * self.boxes_num // self.num_block_data
        for i in range(1, class_num):
            self.tik_inst.data_move(proposal_ub[i * self.boxes_num:, :], proposal_ub, 0, 1, proposal_move_block_num, 0,
                                    0)

    def init_proposal_scores(self, proposal_ub, batch_index, class_index, class_num):
        """
        init_proposal_scores
        """
        score_shape = (class_num, self.boxes_num)
        score_ub = self.tik_inst.Tensor(self.data_dtype, score_shape, tik.scope_ubuf, "score_ub")
        score_move_block_num = class_num * self.boxes_num // self.num_block_data
        self.tik_inst.data_move(score_ub, self.input_score[batch_index:, class_index:, :],
                                0, 1, score_move_block_num, 0, 0)
        self.tik_func.start_tik_compute(class_num * self.boxes_num, self.num_repeat_proposal,
                                        self.tik_func.tik_vconcat,
                                        args=(proposal_ub, score_ub, 4, 0, 0))

    def init_proposal_label(self, proposal_ub, class_index, class_num):
        """
        init_proposal_label
        """
        class_scalar_int = self.tik_inst.Scalar("int32", "class_scalar_int")
        class_scalar_fp16 = self.tik_inst.Scalar(self.data_dtype, "class_scalar_fp16")
        class_scalar_int.set_as(class_index)

        temp_shape = (16,)
        label_ub_int32 = self.tik_inst.Tensor("int32", temp_shape, tik.scope_ubuf, "label_ub_int32")
        label_shape = (class_num, self.boxes_num)
        label_ub_fp16 = self.tik_inst.Tensor(self.data_dtype, label_shape, tik.scope_ubuf, "label_ub_fp16")
        self.tik_inst.vector_dup(16, label_ub_int32, class_scalar_int, 1, 1, 0)
        self.tik_inst.vconv(16, 'none', label_ub_fp16, label_ub_int32, 1, 1, 1, 0, 0, deqscale=1.0)
        class_scalar_fp16.set_as(label_ub_fp16[0])

        self.tik_func.start_tik_compute(self.boxes_num, self.num_repeat_data, self.tik_func.tik_dup,
                                        (label_ub_fp16, class_scalar_fp16))
        for i in range(1, class_num):
            self.tik_func.start_tik_compute(self.boxes_num, self.num_repeat_data, self.tik_func.tik_vadds,
                                            (label_ub_fp16, label_ub_fp16, i, i * self.boxes_num, 0))
        self.tik_func.start_tik_compute(class_num * self.boxes_num, self.num_repeat_proposal,
                                        self.tik_func.tik_vconcat,
                                        args=(proposal_ub, label_ub_fp16, 5, 0, 0))

    def sort_each_class(self, proposal_l1, batch_index_l1, batch_index):
        """
        sort_each_class
        """
        each_loop_class_num = self.proposal_num_max_ub // self.boxes_num
        loop_times, last_loop_class_num = get_loop_info(self.class_num, each_loop_class_num)
        with self.tik_inst.for_range(0, loop_times) as loop_index:
            class_start_index = loop_index * each_loop_class_num
            with self.tik_inst.if_scope(loop_index < loop_times - 1):
                self.sort_each_class_each_loop(proposal_l1, batch_index_l1, batch_index,
                                               class_start_index, each_loop_class_num)
            with self.tik_inst.else_scope():
                self.sort_each_class_each_loop(proposal_l1, batch_index_l1, batch_index,
                                               class_start_index, last_loop_class_num)

    def sort_each_class_each_loop(self, proposal_l1, batch_index_l1, batch_index,
                                  class_start_index, each_loop_class_num):
        proposal_ub_shape = (self.proposal_num_max_ub, self.element_proposal)
        proposal_ub_0 = self.tik_inst.Tensor(self.data_dtype, proposal_ub_shape, tik.scope_ubuf, "proposal_ub_0")
        proposal_ub_0 = self.init_proposal_ub_loop(proposal_ub_0, batch_index,
                                                   class_start_index, each_loop_class_num)

        proposal_ub_1 = self.tik_inst.Tensor(self.data_dtype, proposal_ub_shape, tik.scope_ubuf, "proposal_ub_1")
        self.tik_func.start_tik_compute(each_loop_class_num * self.boxes_num, self.num_repeat_proposal,
                                        self.tik_func.tik_vrpsort16,
                                        args=(proposal_ub_1, proposal_ub_0))
        proposal_ub_0, proposal_ub_1 = self.merge_sort.merge_sort(proposal_ub_0, proposal_ub_1,
                                                                  self.boxes_num * each_loop_class_num,
                                                                  self.num_repeat_proposal, self.boxes_num)
        nburst = each_loop_class_num
        burst = self.max_size_per_class // self.pro_num_block
        src_stride = (self.boxes_num - self.max_size_per_class) // self.pro_num_block
        self.tik_inst.data_move(proposal_l1[batch_index_l1:, class_start_index * self.max_size_per_class:, :],
                                proposal_ub_1, 0, nburst, burst, src_stride, 0)

    def sort_all_class(self, proposal_l1, batch_index_l1):
        """
        sort_all_class
        """
        each_class_boxes_num = self.max_size_per_class
        each_loop_class_num = self.proposal_num_max_ub // each_class_boxes_num
        loop_times, last_loop_class_num = get_loop_info(self.class_num, each_loop_class_num)
        next_loop_boxes_num = self.max_sort_size

        while True:
            with self.tik_inst.for_range(0, loop_times) as loop_index:
                start_index = loop_index * each_loop_class_num * each_class_boxes_num
                output_index = loop_index * next_loop_boxes_num
                proposal_ub_shape = (self.proposal_num_max_ub, self.element_proposal)
                proposal_ub_0 = self.tik_inst.Tensor(self.data_dtype, proposal_ub_shape, tik.scope_ubuf,
                                                     "proposal_ub_0")
                proposal_ub_1 = self.tik_inst.Tensor(self.data_dtype, proposal_ub_shape, tik.scope_ubuf,
                                                     "proposal_ub_1")
                with self.tik_inst.if_scope(loop_index < loop_times - 1):
                    self.sort_all_class_loop(proposal_l1, batch_index_l1, proposal_ub_1, proposal_ub_0, start_index,
                                             output_index,
                                             each_loop_class_num, each_class_boxes_num, next_loop_boxes_num)
                with self.tik_inst.else_scope():
                    self.sort_all_class_loop(proposal_l1, batch_index_l1, proposal_ub_1, proposal_ub_0, start_index,
                                             output_index,
                                             last_loop_class_num, each_class_boxes_num, next_loop_boxes_num)

            if loop_times == 1:
                break
            each_class_boxes_num = next_loop_boxes_num
            next_loop_boxes_num = self.max_sort_size
            each_loop_class_num = self.proposal_num_max_ub // each_class_boxes_num
            loop_times, last_loop_class_num = get_loop_info(loop_times, each_loop_class_num)

    def sort_all_class_loop(self, proposal_l1, batch_index_l1, proposal_ub_1, proposal_ub_0, start_index, output_index,
                            class_num,
                            each_class_pro_num, next_loop_boxes_num):
        """
        sort_all_class_loop
        """
        total_pro_num = class_num * each_class_pro_num
        block_num = total_pro_num // self.pro_num_block
        self.tik_inst.data_move(proposal_ub_0, proposal_l1[batch_index_l1:, start_index:, :], 0, 1, block_num, 0, 0)
        proposal_ub_1, proposal_ub_0 = self.merge_sort.merge_sort(proposal_ub_1, proposal_ub_0, total_pro_num,
                                                                  each_class_pro_num, total_pro_num)
        block_num_out = next_loop_boxes_num // self.pro_num_block
        self.tik_inst.data_move(proposal_l1[batch_index_l1:, output_index:, :], proposal_ub_0,
                                0, 1, block_num_out, 0, 0)


class MergeSort:

    def __init__(self, tik_inst):
        self.tik_inst = tik_inst
        self._valid_bit_all = {1: 1, 2: 3, 3: 7, 4: 15}
        self._merge_channel_num = 4
        self._element_count_max_num = 4096
        self._element_split_num = self._element_count_max_num - 16
        self._pro_num_block = 32 // (8 * 2)

    def merge_sort(self, proposal_next, proposal_sorted, total_num, sorted_num, split_num):
        """
        merge_sort
        """
        proposal_next, proposal_sorted, sorted_num = self.merge_sort_repeat(proposal_next, proposal_sorted,
                                                                            total_num, sorted_num, split_num)
        valid_not_merge, element_count_last = get_loop_info(split_num, sorted_num)
        if valid_not_merge > 1:
            element_list = [sorted_num for _ in range(valid_not_merge)]
            element_list[-1] = element_count_last
            proposal_next, proposal_sorted = self.merge_sort_last(proposal_next, proposal_sorted,
                                                                  element_list, total_num, split_num)
        return proposal_next, proposal_sorted

    def merge_sort_repeat(self, proposal_next, proposal_sorted, total_num, sorted_num, split_num):
        """
        merge_sort_repeat
        """
        repeat_times = total_num // (sorted_num * self._merge_channel_num)
        next_sorted_num = sorted_num * self._merge_channel_num
        while repeat_times > 0 and next_sorted_num <= split_num and sorted_num < self._element_count_max_num:
            src_list = [proposal_sorted[index_i * sorted_num, 0] for index_i in range(self._merge_channel_num)]
            element_count_list = [sorted_num for _ in range(self._merge_channel_num)]
            self.tik_inst.vmrgsort4(proposal_next, src_list, element_count_list, False, 15, repeat_times)

            merged_proposal_num = sorted_num * self._merge_channel_num * repeat_times
            proposal_not_merge_num = total_num - merged_proposal_num
            valid_not_merge, element_count_last = get_loop_info(proposal_not_merge_num, sorted_num)
            if valid_not_merge == 1:
                self.tik_inst.data_move(
                    proposal_next[merged_proposal_num:, :],
                    proposal_sorted[merged_proposal_num:, :], 0, 1,
                    proposal_not_merge_num // self._pro_num_block, 0, 0)
            elif valid_not_merge > 1:
                src_list = [proposal_sorted[0] for _ in range(self._merge_channel_num)]
                element_count_list = [0 for _ in range(self._merge_channel_num)]
                for index_i in range(valid_not_merge):
                    index_add = index_i * sorted_num
                    src_list[index_i] = proposal_sorted[merged_proposal_num + index_add, 0]
                    element_count_list[index_i] = sorted_num
                element_count_list[valid_not_merge - 1] = element_count_last
                self.tik_inst.vmrgsort4(proposal_next[merged_proposal_num, 0], src_list,
                                        element_count_list, False, self._valid_bit_all.get(valid_not_merge), 1)
            sorted_num = next_sorted_num
            next_sorted_num = sorted_num * self._merge_channel_num
            repeat_times = total_num // (sorted_num * 4)
            proposal_sorted, proposal_next = proposal_next, proposal_sorted
        return proposal_next, proposal_sorted, sorted_num

    def merge_sort_last(self, proposal_next, proposal_sorted, element_list, total_num, split_num):
        """
        merge_sort_last
        """
        group_num = total_num // split_num
        for group_index in range(group_num):
            index_start = group_index * split_num
            index_start_all = [index_start] * self._merge_channel_num
            element_count_list = [0] * self._merge_channel_num
            valid_num = 0
            for element_count in element_list:
                if element_count >= self._element_count_max_num:
                    index_start_all[valid_num] = index_start
                    element_count_list[valid_num] = self._element_split_num
                    valid_num += 1
                    index_start += self._element_split_num
                    last_num = element_count - self._element_split_num
                    index_start_all[valid_num] = index_start

                    element_count_list[valid_num] = last_num
                    index_start += last_num
                    valid_num += 1
                else:
                    index_start_all[valid_num] = index_start
                    element_count_list[valid_num] = element_count
                    valid_num += 1
                    index_start += element_count
            src_list = [proposal_sorted[index_temp, 0] for index_temp in index_start_all]
            self.tik_inst.vmrgsort4(proposal_next[group_index * split_num:, :], src_list, element_count_list,
                                    False, self._valid_bit_all.get(valid_num), 1)
        proposal_sorted, proposal_next = proposal_next, proposal_sorted
        return proposal_next, proposal_sorted


class GetBoxesNum:

    def __init__(self, tik_inst, tik_func):
        self.tik_inst = tik_inst
        self.tik_func = tik_func

        self.fp32_dtype = "float32"
        self.out_dtype = "float16"

        self.element_num_block_fp32 = 8
        self.element_num_repeat_fp32 = 64

        self.element_num_block_out = 16
        self.element_num_repeat_out = 128

        self.element_proposal = 8
        self.proposal_num_repeat = 16
        self.pro_num_block = 32 // (8 * 2)

    def get_boxes_num(self, pro_out, batch_index_l1, pro_num, boxes_num, score_threshold, neg_inf=-65504.0,
                      pos_inf=65504.0):
        """
        get_boxes_num
        """
        if score_threshold <= neg_inf:
            pro_num.set_as(boxes_num)
        elif score_threshold >= pos_inf:
            pro_num.set_as(0)
        else:
            with self.tik_inst.new_stmt_scope():
                self._get_num_max(pro_out, batch_index_l1, pro_num, boxes_num, score_threshold)
        return pro_num

    def _init_score_ub(self, pro_out, batch_index_l1, score_ub, boxes_num):
        with self.tik_inst.new_stmt_scope():
            proposal_ub = self.tik_inst.Tensor(self.out_dtype, (boxes_num, self.element_proposal),
                                               tik.scope_ubuf, "proposal_ub")
            score_move_block_num = boxes_num // self.pro_num_block
            self.tik_inst.data_move(proposal_ub, pro_out[batch_index_l1:, :, :], 0, 1, score_move_block_num, 0, 0)
            self.tik_func.start_tik_compute(boxes_num, self.proposal_num_repeat,
                                            self.tik_func.tik_vextract,
                                            args=(score_ub, proposal_ub, 4, 0, 0))

    def _get_num_max(self, pro_out, batch_index_l1, pro_num, boxes_num, score_threshold):
        score_ub = self.tik_inst.Tensor(self.out_dtype, (boxes_num,),
                                        tik.scope_ubuf, "score_ub")
        self._init_score_ub(pro_out, batch_index_l1, score_ub, boxes_num)

        num_sum_ub_fp32 = self.tik_inst.Tensor(self.fp32_dtype, (self.element_num_block_fp32,),
                                               tik.scope_ubuf, "num_sum_ub_fp32")
        self.tik_func.start_tik_compute(self.element_num_block_fp32, self.element_num_repeat_fp32,
                                        self.tik_func.tik_dup, (num_sum_ub_fp32, 0.0))
        num_sum_ub_int32 = self.tik_inst.Tensor("int32", (self.element_num_block_fp32,),
                                                tik.scope_ubuf, "num_sum_ub_int32")

        cmp_shape = (self.element_num_repeat_out,)
        score_threshold_ub = self.tik_inst.Tensor(
            self.out_dtype, cmp_shape,
            tik.scope_ubuf, "score_threshold_ub")
        one_ub = self.tik_inst.Tensor(
            self.out_dtype, cmp_shape, tik.scope_ubuf, "one_ub")
        zero_ub = self.tik_inst.Tensor(
            self.out_dtype, cmp_shape, tik.scope_ubuf, "zero_ub")
        self.tik_func.start_tik_compute(self.element_num_repeat_out, self.element_num_repeat_out,
                                        self.tik_func.tik_dup, (score_threshold_ub, score_threshold))
        self.tik_func.start_tik_compute(self.element_num_repeat_out, self.element_num_repeat_out,
                                        self.tik_func.tik_dup, (one_ub, 1.0))
        self.tik_func.start_tik_compute(self.element_num_repeat_out, self.element_num_repeat_out,
                                        self.tik_func.tik_dup, (zero_ub, 0.0))

        cmp_num = boxes_num // 8
        cmp_result_ub = self.tik_inst.Tensor("uint8", (cmp_num,), tik.scope_ubuf, "cmp_result_ub")

        repeat_num_fp32 = boxes_num // self.element_num_repeat_fp32
        work_tensor_fp32 = self.tik_inst.Tensor(self.fp32_dtype, (repeat_num_fp32,), tik.scope_ubuf, "work_tensor_fp32")
        cmp_result = self.tik_inst.Tensor(self.fp32_dtype, (boxes_num,), tik.scope_ubuf, "cmp_result")

        mask = self.element_num_repeat_out
        repeat_num_out = boxes_num // mask
        self.tik_inst.vec_cmpv_gt(cmp_result_ub, score_ub, score_threshold_ub, repeat_num_out, 8, 0)
        self.tik_inst.vec_sel(mask, 2, score_ub, cmp_result_ub, one_ub, zero_ub, repeat_num_out, 8, 0, 0)

        mask_fp32 = self.element_num_repeat_fp32
        repeat_num_fp32 = boxes_num // mask_fp32
        self.tik_inst.vconv(mask_fp32, "", cmp_result, score_ub, repeat_num_fp32, 1, 1, 8, 4)
        self.tik_inst.vec_reduce_add(mask_fp32, num_sum_ub_fp32, cmp_result, work_tensor_fp32, repeat_num_fp32, 8)
        self.tik_inst.vconv(1, "round", num_sum_ub_int32, num_sum_ub_fp32, 1, 1, 1, 8, 8)
        pro_num.set_as(num_sum_ub_int32[0])


class NonMaximumSuppression:

    def __init__(self, tik_inst, tik_func):
        self.tik_inst = tik_inst
        self.out_dtype = "float16"
        self.ub_dtype = "float32"
        self.element_num_pro = 8
        self.pro_num_repeat = 16
        self.tik_func = tik_func

        self.element_num_block_ub = 8
        self.element_num_repeat_ub = 64

        self.element_num_block_out = 16
        self.element_num_repeat_out = 128

        self.element_proposal = 8
        self.proposal_num_repeat = 16
        self.pro_num_block = 32 // (8 * 2)

        self._uint_type = "uint16"
        self._uint_size = 2

    def non_maximum_suppression(self, pro_out, batch_index_l1, boxes_num,
                                max_output_size, iou_threshold,
                                image_size=(0.0, 0.0), offset=0.0):
        """
        non_maximum_suppression
        """
        select_num_scalar = self.tik_inst.Scalar(dtype="int32", init_value=0)
        max_output_size_format = get_align_num(max_output_size, self.pro_num_repeat)
        select_pro_result = self.tik_inst.Tensor(
            self.out_dtype, (max_output_size_format, self.element_num_pro),
            tik.scope_ubuf, "select_pro_result")

        each_loop_boxes_num = 512
        threshold = iou_threshold / (1 + iou_threshold)
        with self.tik_inst.new_stmt_scope():
            select_pro_ub = self.tik_inst.Tensor(
                self.ub_dtype, (max_output_size_format, self.element_num_pro),
                tik.scope_ubuf, "select_pro_ub")
            mask = self.element_num_repeat_ub
            repeat_times = max_output_size_format * self.element_num_pro // self.element_num_repeat_ub
            self.tik_inst.vector_dup(mask, select_pro_ub, 0.0, repeat_times, 1, 8)
            self.tik_inst.set_rpn_offset(offset, self.ub_dtype)
            self._nms_start(pro_out, batch_index_l1, select_pro_ub, select_num_scalar,
                            boxes_num, max_output_size, threshold, each_loop_boxes_num, image_size, offset)
            self.tik_inst.set_rpn_offset(1.0, self.ub_dtype)
            mask_fp32 = self.element_num_repeat_ub
            repeat_num_fp32 = max_output_size_format * self.element_num_pro // mask_fp32
            self.tik_inst.vconv(mask_fp32, "", select_pro_result, select_pro_ub, repeat_num_fp32, 1, 1, 4, 8)
        return select_pro_result, select_num_scalar

    def _init_scalar(self):
        select_num_temp_scalar = \
            self.tik_inst.Scalar(dtype="int32", init_value=0)
        select_num_format_scalar = \
            self.tik_inst.Scalar(dtype="int32", init_value=0)
        pro_num_batch_scalar = \
            self.tik_inst.Scalar(dtype="int32", init_value=0)
        pro_repeat_scalar = \
            self.tik_inst.Scalar(dtype="int32", init_value=0)
        repeat_scalar = self.tik_inst.Scalar(dtype="int32", init_value=0)
        zero_scalar = self.tik_inst.Scalar("uint16", init_value=0)
        temp_data_scalar = self.tik_inst.Scalar(self.ub_dtype, init_value=0)
        return {"select_num_temp_scalar": select_num_temp_scalar,
                "select_num_format_scalar": select_num_format_scalar,
                "pro_num_batch_scalar": pro_num_batch_scalar,
                "pro_repeat_scalar": pro_repeat_scalar,
                "repeat_scalar": repeat_scalar,
                "zero_scalar": zero_scalar,
                "temp_data_scalar": temp_data_scalar}

    def _init_tensor(self, pro_num, max_output_size_format):
        pro_ub = self.tik_inst.Tensor(
            self.ub_dtype, (pro_num, self.element_num_pro),
            tik.scope_ubuf, "pro_ub")
        loop_pro_area_ub = self.tik_inst.Tensor(
            self.ub_dtype, (pro_num, self.element_num_pro),
            tik.scope_ubuf, "loop_pro_area_ub")
        select_pro_area_ub = self.tik_inst.Tensor(
            self.ub_dtype, (pro_num, self.element_num_pro),
            tik.scope_ubuf, "select_pro_area_ub")

        pro_num_all = max_output_size_format + pro_num
        area_shape = (pro_num_all, self.pro_num_repeat)
        iou_area_ub = self.tik_inst.Tensor(
            self.ub_dtype, area_shape, tik.scope_ubuf, "iou_area_ub")
        add_area_ub = self.tik_inst.Tensor(
            self.ub_dtype, area_shape, tik.scope_ubuf, "add_area_ub")

        area_cmp_gt_ub = self.tik_inst.Tensor(
            "uint16", (pro_num_all,), tik.scope_ubuf, "area_cmp_gt_ub")
        loop_sup_vec_ub = self.tik_inst.Tensor(
            "uint16", (pro_num,), tik.scope_ubuf, "loop_sup_vec_ub")
        select_sup_vec_ub = self.tik_inst.Tensor(
            "uint16", (max_output_size_format,),
            tik.scope_ubuf, "select_sup_vec_ub")
        self.tik_func.start_tik_compute(select_sup_vec_ub.size, self.element_num_repeat_out, self.tik_func.tik_dup,
                                        (select_sup_vec_ub, 1))
        self.tik_func.start_tik_compute(loop_sup_vec_ub.size, self.element_num_repeat_out, self.tik_func.tik_dup,
                                        (loop_sup_vec_ub, 1))
        return {"pro_ub": pro_ub,
                "loop_pro_area_ub": loop_pro_area_ub,
                "select_pro_area_ub": select_pro_area_ub,
                "iou_area_ub": iou_area_ub,
                "add_area_ub": add_area_ub,
                "area_cmp_gt_ub": area_cmp_gt_ub,
                "loop_sup_vec_ub": loop_sup_vec_ub,
                "select_sup_vec_ub": select_sup_vec_ub}

    def _init_trans_ub(self, pro_num, max_output_size_format):
        pro_ub_trans = self.tik_inst.Tensor(
            self.ub_dtype, (pro_num, self.element_num_pro),
            tik.scope_ubuf, "pro_ub_trans")
        select_pro_ub_trans = self.tik_inst.Tensor(
            self.ub_dtype, (max_output_size_format, self.element_num_pro),
            tik.scope_ubuf, "select_pro_ub_trans")
        return pro_ub_trans, select_pro_ub_trans

    def _nms_start(self, pro_out, batch_index_l1, select_pro_ub, select_num_scalar,
                   boxes_num, max_output_size, threshold,
                   each_loop_boxes_num, image_size, offset):
        max_output_size_format = get_align_num(max_output_size, self.pro_num_repeat)
        scalar_all = self._init_scalar()
        loop_times, last_loop_boxes_num = get_loop_info(boxes_num, each_loop_boxes_num)
        tensor_all = self._init_tensor(each_loop_boxes_num, max_output_size_format)
        trans_tensor_all = self._init_trans_ub(each_loop_boxes_num, max_output_size_format)

        with self.tik_inst.for_range(0, loop_times) as loop_index:
            with self.tik_inst.if_scope(select_num_scalar < max_output_size):
                start_index = loop_index * each_loop_boxes_num
                with self.tik_inst.if_scope(loop_index != loop_times - 1):
                    self._nms_each_loop(pro_out, batch_index_l1, select_pro_ub, select_num_scalar,
                                        tensor_all, trans_tensor_all, scalar_all,
                                        max_output_size, threshold, image_size, offset,
                                        start_index, each_loop_boxes_num, each_loop_boxes_num)
                with self.tik_inst.else_scope():
                    self._nms_each_loop(pro_out, batch_index_l1, select_pro_ub, select_num_scalar,
                                        tensor_all, trans_tensor_all, scalar_all,
                                        max_output_size, threshold, image_size, offset,
                                        start_index, last_loop_boxes_num, each_loop_boxes_num)
        return select_pro_ub, select_num_scalar

    # 'pylint: disable = unused-argument
    # 'pylint: disable=too-many-arguments,too-many-locals
    def _init_pro_ub(self, pro_out, batch_index_l1, pro_ub_trans, pro_ub, start_index, boxes_num, shape_pro_num,
                     image_size, offset):
        element_num = shape_pro_num * self.element_num_pro
        mask_ub = self.element_num_repeat_ub
        repeat_ub = element_num // mask_ub
        self.tik_inst.vector_dup(mask_ub, pro_ub, 0.0, repeat_ub, 1, 8)
        self.tik_inst.vector_dup(mask_ub, pro_ub_trans, 0.0, repeat_ub, 1, 8)
        with self.tik_inst.new_stmt_scope():
            pro_ub_out = self.tik_inst.Tensor(self.out_dtype, (shape_pro_num, self.element_num_pro),
                                              tik.scope_ubuf, "pro_ub")
            mask_out = self.element_num_repeat_out
            repeat_times_out = element_num // mask_out
            self.tik_inst.vector_dup(mask_out, pro_ub_out, 0.0, repeat_times_out, 1, 8)
            move_in_block_num = ceil_div(boxes_num, self.pro_num_block)
            self.tik_inst.data_move(pro_ub_out, pro_out[batch_index_l1:, start_index:, :],
                                    0, 1, move_in_block_num, 0, 0)
            self.tik_inst.vconv(mask_ub, "", pro_ub, pro_ub_out, repeat_ub, 1, 1, 8, 4)
        with self.tik_inst.if_scope(boxes_num < shape_pro_num):
            self.tik_inst.vector_dup(self.element_num_pro, pro_ub[boxes_num, 0], 0.0, 1, 1, 8)
        with self.tik_inst.new_stmt_scope():
            repeat_extract = shape_pro_num // self.pro_num_repeat
            repeat_trans = shape_pro_num // mask_ub
            trans_shape = (shape_pro_num,)
            x1_ub = self.tik_inst.Tensor(self.ub_dtype, trans_shape, tik.scope_ubuf, "x1_ub")
            y1_ub = self.tik_inst.Tensor(self.ub_dtype, trans_shape, tik.scope_ubuf, "y1_ub")
            x2_ub = self.tik_inst.Tensor(self.ub_dtype, trans_shape, tik.scope_ubuf, "x2_ub")
            y2_ub = self.tik_inst.Tensor(self.ub_dtype, trans_shape, tik.scope_ubuf, "y2_ub")
            score_ub = self.tik_inst.Tensor(self.ub_dtype, trans_shape, tik.scope_ubuf, "score_ub")
            label_ub = self.tik_inst.Tensor(self.ub_dtype, trans_shape, tik.scope_ubuf, "label_ub")

            x_add = self.tik_inst.Tensor(self.ub_dtype, trans_shape, tik.scope_ubuf, "x_add")
            y_add = self.tik_inst.Tensor(self.ub_dtype, trans_shape, tik.scope_ubuf, "y_add")

            self.tik_inst.vextract(x1_ub, pro_ub, repeat_extract, 0)
            self.tik_inst.vextract(y1_ub, pro_ub, repeat_extract, 1)
            self.tik_inst.vextract(x2_ub, pro_ub, repeat_extract, 2)
            self.tik_inst.vextract(y2_ub, pro_ub, repeat_extract, 3)
            self.tik_inst.vextract(score_ub, pro_ub, repeat_extract, 4)
            self.tik_inst.vextract(label_ub, pro_ub, repeat_extract, 5)

            self.tik_inst.vmuls(mask_ub, x_add, label_ub, image_size[0], repeat_trans, 1, 1, 8, 8)
            self.tik_inst.vmuls(mask_ub, y_add, label_ub, image_size[1], repeat_trans, 1, 1, 8, 8)
            self.tik_inst.vadd(mask_ub, x1_ub, x1_ub, x_add, repeat_trans, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vadd(mask_ub, y1_ub, y1_ub, y_add, repeat_trans, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vadd(mask_ub, x2_ub, x2_ub, x_add, repeat_trans, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vadd(mask_ub, y2_ub, y2_ub, y_add, repeat_trans, 1, 1, 1, 8, 8, 8)

            self.tik_inst.vconcat(pro_ub_trans, x1_ub, repeat_extract, 0)
            self.tik_inst.vconcat(pro_ub_trans, y1_ub, repeat_extract, 1)
            self.tik_inst.vconcat(pro_ub_trans, x2_ub, repeat_extract, 2)
            self.tik_inst.vconcat(pro_ub_trans, y2_ub, repeat_extract, 3)
            self.tik_inst.vconcat(pro_ub_trans, score_ub, repeat_extract, 4)
            self.tik_inst.vconcat(pro_ub_trans, label_ub, repeat_extract, 5)
        return pro_ub, pro_ub_trans

    def _nms_each_loop(self, pro_out, batch_index_l1, select_pro_ub, select_num_scalar,
                       tensor_all, trans_tensor_all, scalar_all,
                       max_output_size, threshold, image_size, offset,
                       start_index, boxes_num, shape_pro_num):
        (select_num_temp_scalar, select_num_format_scalar, pro_num_batch_scalar, pro_repeat_scalar,
         repeat_scalar, zero_scalar, temp_data_scalar) = \
            (scalar_all.get("select_num_temp_scalar"), scalar_all.get("select_num_format_scalar"),
             scalar_all.get("pro_num_batch_scalar"), scalar_all.get("pro_repeat_scalar"),
             scalar_all.get("repeat_scalar"), scalar_all.get("zero_scalar"),
             scalar_all.get("temp_data_scalar"))
        (pro_ub, loop_pro_area_ub, select_pro_area_ub, iou_area_ub, add_area_ub,
         area_cmp_gt_ub, loop_sup_vec_ub, select_sup_vec_ub) = \
            (tensor_all.get("pro_ub"), tensor_all.get("loop_pro_area_ub"),
             tensor_all.get("select_pro_area_ub"), tensor_all.get("iou_area_ub"),
             tensor_all.get("add_area_ub"), tensor_all.get("area_cmp_gt_ub"),
             tensor_all.get("loop_sup_vec_ub"), tensor_all.get("select_sup_vec_ub"))

        pro_ub_trans, select_pro_ub_trans = trans_tensor_all

        pro_ub, pro_ub_trans = self._init_pro_ub(pro_out, batch_index_l1, pro_ub_trans, pro_ub, start_index, boxes_num,
                                                 shape_pro_num,
                                                 image_size, offset)
        self.tik_func.start_tik_compute(loop_sup_vec_ub.size, self.element_num_repeat_out, self.tik_func.tik_dup,
                                        (loop_sup_vec_ub, 1))
        select_num_temp_scalar.set_as(select_num_scalar)
        pro_repeat_scalar.set_as(ceil_div(select_num_scalar, self.pro_num_repeat))
        select_num_format_scalar.set_as(pro_repeat_scalar * self.pro_num_repeat)
        pro_num_batch_scalar.set_as(select_num_format_scalar)
        repeat_num = ceil_div(boxes_num, self.pro_num_repeat)
        self.tik_inst.vrpac(loop_pro_area_ub, pro_ub_trans, repeat_num)
        self.tik_inst.vrpac(select_pro_area_ub, select_pro_ub_trans, pro_repeat_scalar)

        with self.tik_inst.for_range(0, repeat_num) as index_i:
            with self.tik_inst.if_scope(select_num_temp_scalar < max_output_size):
                self._nms_each_batch(tensor_all, trans_tensor_all,
                                     scalar_all, threshold, index_i)
        self._update_result_each_loop(select_pro_ub, tensor_all,
                                      trans_tensor_all, select_num_scalar,
                                      zero_scalar, boxes_num,
                                      max_output_size)

    def _nms_each_batch(self, tensor_all, trans_tensor_all,
                        scalar_all, threshold, index_i):
        (pro_ub, loop_pro_area_ub, select_pro_area_ub, iou_area_ub, add_area_ub,
         area_cmp_gt_ub, loop_sup_vec_ub, select_sup_vec_ub) = \
            (tensor_all.get("pro_ub"), tensor_all.get("loop_pro_area_ub"),
             tensor_all.get("select_pro_area_ub"), tensor_all.get("iou_area_ub"),
             tensor_all.get("add_area_ub"), tensor_all.get("area_cmp_gt_ub"),
             tensor_all.get("loop_sup_vec_ub"), tensor_all.get("select_sup_vec_ub"))
        pro_ub_trans, select_pro_ub_trans = trans_tensor_all
        (select_num_temp_scalar, select_num_format_scalar, pro_num_batch_scalar, pro_repeat_scalar,
         repeat_scalar, zero_scalar, temp_data_scalar) = \
            (scalar_all.get("select_num_temp_scalar"), scalar_all.get("select_num_format_scalar"),
             scalar_all.get("pro_num_batch_scalar"), scalar_all.get("pro_repeat_scalar"),
             scalar_all.get("repeat_scalar"), scalar_all.get("zero_scalar"),
             scalar_all.get("temp_data_scalar"))

        pro_num_batch_scalar.set_as(pro_num_batch_scalar +
                                    self.pro_num_repeat)
        start_index = index_i * self.pro_num_repeat

        self.tik_inst.viou(iou_area_ub, select_pro_ub_trans,
                           pro_ub_trans[start_index, 0], pro_repeat_scalar)
        self.tik_inst.viou(iou_area_ub[select_num_format_scalar, 0],
                           pro_ub_trans, pro_ub_trans[start_index, 0],
                           index_i + 1)

        self.tik_inst.vaadd(add_area_ub, select_pro_area_ub,
                            loop_pro_area_ub[start_index, ],
                            pro_repeat_scalar)
        self.tik_inst.vaadd(add_area_ub[select_num_format_scalar, 0],
                            loop_pro_area_ub,
                            loop_pro_area_ub[start_index], index_i + 1)

        repeat_scalar.set_as(pro_num_batch_scalar //
                             (self.element_num_repeat_ub // self.pro_num_repeat))
        self.tik_inst.vmuls(self.element_num_repeat_ub, add_area_ub, add_area_ub,
                            threshold, repeat_scalar, 1, 1, 8, 8)
        self.tik_inst.vcmpv_gt(area_cmp_gt_ub, iou_area_ub[0, 0],
                               add_area_ub[0, 0], repeat_scalar, 1, 1, 8, 8)
        rpn_cor_ir = self.tik_inst.set_rpn_cor_ir(0)
        rpn_cor_ir = self.tik_inst.rpn_cor(area_cmp_gt_ub[0],
                                           select_sup_vec_ub[0], 1, 1,
                                           pro_repeat_scalar)
        rpn_cor_ir = self.tik_inst.rpn_cor(
            area_cmp_gt_ub[select_num_format_scalar],
            loop_sup_vec_ub[0], 1, 1, index_i)
        self.tik_inst.rpn_cor_diag(
            loop_sup_vec_ub[start_index],
            area_cmp_gt_ub[pro_num_batch_scalar - self.pro_num_repeat],
            rpn_cor_ir)
        with self.tik_inst.for_range(0, self.pro_num_repeat) as index_j:
            with self.tik_inst.if_scope(
                    loop_sup_vec_ub[start_index + index_j] == 0):
                select_num_temp_scalar.set_as(select_num_temp_scalar + 1)

    def _update_result_each_loop(self, select_pro_ub, tensor_all,
                                 trans_tensor_all, select_num_scalar,
                                 zero_scalar, boxes_num,
                                 max_output_size):
        (pro_ub, loop_pro_area_ub, select_pro_area_ub, iou_area_ub, add_area_ub, area_cmp_gt_ub, loop_sup_vec_ub,
         select_sup_vec_ub) = \
            (tensor_all.get("pro_ub"), tensor_all.get("loop_pro_area_ub"),
             tensor_all.get("select_pro_area_ub"), tensor_all.get("iou_area_ub"),
             tensor_all.get("add_area_ub"), tensor_all.get("area_cmp_gt_ub"),
             tensor_all.get("loop_sup_vec_ub"), tensor_all.get("select_sup_vec_ub"))
        pro_ub_trans, select_pro_ub_trans = trans_tensor_all
        with self.tik_inst.for_range(0, boxes_num) as index_i:
            with self.tik_inst.if_scope(select_num_scalar < max_output_size):
                with self.tik_inst.if_scope(loop_sup_vec_ub[index_i] == 0):
                    self._update_result_each_boxes(
                        pro_ub, pro_ub_trans,
                        select_pro_ub, select_pro_ub_trans,
                        select_num_scalar, index_i)
                    select_sup_vec_ub[select_num_scalar].set_as(zero_scalar)
                    select_num_scalar.set_as(select_num_scalar + 1)

    def _update_result_each_boxes(self, pro_ub, pro_ub_trans,
                                  select_pro_ub, select_pro_ub_trans,
                                  select_num_scalar, index_i):
        self.tik_inst.data_move(
            select_pro_ub_trans[select_num_scalar:, :],
            pro_ub_trans[index_i:, :], 0, 1, 1, 0, 0)
        self.tik_inst.data_move(
            select_pro_ub[select_num_scalar:, :],
            pro_ub[index_i:, :], 0, 1, 1, 0, 0)


# 'pylint: disable = unused-argument
# 'pylint: disable=too-many-arguments,too-many-locals
def norm_class_support(boxes, scores, clip_window, num_valid_boxes,
                       nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num,
                       score_threshold, iou_threshold, max_size_per_class,
                       max_total_size, change_coordinate_frame, transpose_box,
                       image_size, kernel_name, impl_mode):
    if impl_mode != "norm_class":
        return False
    elif not tbe_platform.api_check_support("tik.viou", "float32"):
        return False
    elif not (clip_window is None and num_valid_boxes is None):
        return False
    elif len(image_size) != 2:
        return False
    return True


# 'pylint: disable = unused-argument
# 'pylint: disable=too-many-arguments,too-many-locals
def batch_multi_class_non_max_suppression_norm_class(boxes, scores, clip_window, num_valid_boxes,
                                                     nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num,
                                                     score_threshold, iou_threshold, max_size_per_class,
                                                     max_total_size, change_coordinate_frame, transpose_box,
                                                     image_size,
                                                     kernel_name,
                                                     impl_mode):
    """
    do non_max_suppression for multi batch and multi class
    Parameters:
    ----------
    boxes : dict.
        shape, dtype of boxes, a 4D Tensor of type float16 with shape (batch, num_anchors, num_classes, 4).
        "batch" indicates the batch size of image,
        and "num_anchors" indicates num of boxes, and "num_classes" indicates classes of detect.
        and the value "4" refers to "x0", "x1", "y0", and "y1".
    scores : dict.
        shape, dtype of scores
        a 3D Tensor of type float16 with shape (batch, num_anchors, num_classes).
    clip_window : dict.
        shape, dtype of scores
        a 2D Tensor of type float16 with shape (batch, 4).
        4" refers to "anchor_x0", "anchor_x1", "anchor_y0", and "anchor_y1".
    num_valid_boxes : dict.
        A 1D Tensor of type int32 with shape (batch,).
        specifying valid boxes number for each batch
    nmsed_boxes : dict.
        A 3D Tensor of type float16 with shape (batch, max_total_size, 4).
        specifying the output nms boxes per batch
    nmsed_scores : dict.
        A 3D Tensor of type float16 with shape (batch, max_total_size).
        specifying the output nms score per batch
    nmsed_classes : dict.
        A 3D Tensor of type float16 with shape (batch, max_total_size).
        specifying the output nms class per batch
    nmsed_num : dict.
        A 1D Tensor of type int32 with shape (batch,),
        specifying the valid num of nmsed_boxes
    score_threshold : float.
        A required attribute of type float32, specifying the score filter iou iou_threshold.
    iou_threshold : float.
        A required attribute of type float32, specifying the nms iou iou_threshold
    max_size_per_class : int.
        A required attribute of type int, specifying the nms output num per class.
    max_total_size : int.
        A required attribute of type int, specifying the the nms output num per batch.
    change_coordinate_frame : bool.
        A required attribute of type bool, whether to normalize coordinates after clipping.
    transpose_box : bool.
        A required attribute of type bool, whether inserted transpose before this op
    image_size:
        A optional attribute of type ListInt, the size of the image.
    kernel_name : str.
        cce kernel name, default value is "batch_multi_class_non_max_suppression"
    impl_mode: str.
        high_precision or high_performance for inference, default value is "high_performance".
        no need to add into ops_info file.

    Returns
    -------
    tik_instance
    """
    boxes_shape = boxes.get("shape")
    scores_shape = scores.get("shape")
    if len(scores_shape) == 2 and len(boxes_shape) == 3:
        batch_num = scores_shape[0]
        class_num = 1
        boxes_num = scores_shape[1]
    else:
        batch_num, class_num, boxes_num = scores_shape
    obj = BatchMultiClassNonMaxSuppressionNormClass(batch_num, class_num, boxes_num, score_threshold, iou_threshold,
                                                    max_size_per_class,
                                                    max_total_size, image_size, kernel_name)
    obj.start()
