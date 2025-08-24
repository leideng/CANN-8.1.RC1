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
merge_sort
"""
from impl.ascend import TensorOperatorParam
from impl.ascend import VecCmd
from impl.ascend import VecExecutor


# 'pylint: disable=useless-object-inheritance
class CommonMethod(object):
    """
    CommonMethod
    """

    def __init__(self, container):
        self._cont = container
        self._tik_inst = self._cont.tinst
        self._block_size = self._cont.const_block_byte
        self._repeat_time_max = self._cont.const_vector_proc_max_rpt
        self._pro_data_num = self._cont.const_proposal_data_num
        self._pro_repeat_num = self._cont.const_proposal_repeat_num

    @staticmethod
    def ceil_div(dividend, divisor):
        """
        ceil div
        """
        result = (dividend + divisor - 1) // divisor
        return result

    def get_type_const(self, data_type):
        """
        get type const
        """
        data_size = self._cont.const_dtype_byte.get(data_type)
        block_data_num = self._cont.get_vec_proc_num_per_cmd_blk(data_type)
        repeat_data_num = self._cont.get_vec_proc_num_per_cmd(data_type)
        return data_size, block_data_num, repeat_data_num

    def get_block_pro_num(self, data_type):
        """
        get block proposal num
        """
        pro_block_num = (self._block_size // (self._pro_data_num * self._cont.const_dtype_byte.get(data_type)))
        return pro_block_num

    def get_mask(self, data_num, format_num_floor, format_num_ceil):
        """
        algorithm: count TIk command mask
        Parameters
        ----------
        data_num: int
        format_num_floor: int, align num
        format_num_ceil: int, align num
        Returns
        -------
        mask_h: int
        mask_l: int
        data_num_floor: int
        """
        mask_split = 64
        data_num_floor = self.get_align_num(data_num, format_num_floor, False)
        data_num_ceil = self.get_align_num(data_num, format_num_ceil, True)
        mask_h, mask_l = 0, 0
        index_start = data_num - data_num_floor
        index_end = data_num_ceil - data_num_floor
        for index_l in range(index_start, min(index_end, mask_split)):
            mask_l += 2 ** index_l
        for index_h in range(max(mask_split, index_start), index_end):
            mask_h += 2 ** (index_h - mask_split)
        return mask_h, mask_l, data_num_floor

    def get_loop_info(self, all_data_num, each_loop_num):
        """
        get loop info
        """
        loop_times = self.ceil_div(all_data_num, each_loop_num)
        last_loop_num = all_data_num - each_loop_num * (loop_times - 1)
        return loop_times, last_loop_num

    # 'pylint: disable=no-self-use
    @staticmethod
    def vector_dup(tensor, num, data_num=None, offset=0):
        """
        vector dup
        """
        if data_num is None:
            data_num = tensor.size
        buf_dup_all = {"tensor_dst": TensorOperatorParam(tensor, data_num, offset)}
        cmd_dup_tensor = [VecCmd(cmd_name="vector_dup", dst_name="tensor_dst", scalar=num)]

        VecExecutor.exec_vec_cmd(buf_dup_all, cmd_dup_tensor, "tensor_dst")

    def vector_concat(self, dst, src, mode_number, pro_num):
        """
        vector concat
        """
        repeat_num_max = self._repeat_time_max * self._pro_repeat_num
        repeat_times = pro_num // self._pro_repeat_num
        loop_times, last_loop_repeat = self.get_loop_info(
            repeat_times, self._repeat_time_max)
        for loop_index in range(loop_times):
            boxes_index = loop_index * repeat_num_max
            if loop_index != loop_times - 1:
                self._tik_inst.vconcat(dst[boxes_index, 0], src[boxes_index],
                                       self._repeat_time_max, mode_number)
            else:
                self._tik_inst.vconcat(dst[boxes_index, 0], src[boxes_index],
                                       last_loop_repeat, mode_number)

    def vector_extract(self, dst, src, mode_number, pro_num):
        """
        vector extract
        """
        repeat_num_max = self._repeat_time_max * self._pro_repeat_num
        repeat_times = pro_num // self._pro_repeat_num
        loop_times, last_loop_repeat = self.get_loop_info(
            repeat_times, self._repeat_time_max)
        for loop_index in range(loop_times):
            boxes_index = loop_index * repeat_num_max
            if loop_index != loop_times - 1:
                self._tik_inst.vextract(dst[boxes_index], src[boxes_index, 0],
                                        self._repeat_time_max, mode_number)
            else:
                self._tik_inst.vextract(dst[boxes_index], src[boxes_index, 0],
                                        last_loop_repeat, mode_number)

    def get_align_num(self, input_num, align_num, ceil=True):
        """
        get align num
        """
        if ceil:
            result = self.ceil_div(input_num, align_num) * align_num
        else:
            result = input_num // align_num * align_num
        return result

    def emit_vmuls(self, dst, src, cnt):
        """
        emit vmuls
        """
        MASK_FP16 = 128
        MAX_REPEAT_TIMES = 255
        repeat = cnt // MASK_FP16
        repeat_remain = cnt % MASK_FP16
        times = (repeat + MAX_REPEAT_TIMES - 1) // MAX_REPEAT_TIMES
        if repeat > 0:
            with self._tik_inst.for_range(0, times, name="vmuls_i0") as i:
                src0_scalar = self._tik_inst.Scalar(dtype="int64", name="src0_scalar",
                                                    init_value=repeat - i * MAX_REPEAT_TIMES)
                src1_scalar = self._tik_inst.Scalar(dtype="int64", name="src1_scalar", init_value=MAX_REPEAT_TIMES)
                times_len = self._tik_inst.Scalar(dtype="int64", name="times_len")
                self._tik_inst.scalar_min(times_len, src0_scalar, src1_scalar)
                self._tik_inst.vmuls(MASK_FP16, dst[i * MASK_FP16 * MAX_REPEAT_TIMES],
                                     src[i * MASK_FP16 * MAX_REPEAT_TIMES], -1, times_len, 1, 1, 8, 8)
        if repeat_remain > 0:
            self._tik_inst.vmuls(repeat_remain, dst[repeat * MASK_FP16], src[repeat * MASK_FP16], -1, 1, 1, 1, 8, 8)


# 'pylint: disable=useless-object-inheritance
class MergeSort(object):
    """
    MergeSort
    """

    def __init__(self, container, data_type, ub_size=None):
        self._cont = container
        self._tik = self._cont.tik
        self._tik_inst = self._cont.tinst
        self._data_type = data_type
        self._ub_size = self._cont.const_ub_max_byte if ub_size is None else ub_size
        self._method = CommonMethod(self._cont)
        # tik const value
        self._pro_data_num = self._cont.const_proposal_data_num
        self._pro_repeat_num = self._cont.const_proposal_repeat_num
        self._repeat_time_max = self._cont.const_vector_proc_max_rpt
        self._data_size, self._block_data_num, self._repeat_data_num = self._method.get_type_const(self._data_type)
        self._block_pro_num = self._method.get_block_pro_num(self._data_type)
        # merge data
        self._valid_bit_all = {1: 1, 2: 3, 3: 7, 4: 15}
        self._merge_channel_num = 4
        self._element_count_max_num = 4096
        self._element_split_num = self._element_count_max_num - self._pro_repeat_num
        self._ub_pro_num_max, self._ub_sort_num = self._get_pro_num_info()
        self._int_type = "int64"

    def _get_pro_num_info(self):
        each_pro_size = self._pro_data_num * self._data_size * 2
        ub_pro_num_max = self._ub_size // each_pro_size
        align_num = self._pro_repeat_num * self._merge_channel_num
        ub_pro_num_max = min(ub_pro_num_max, self._element_split_num * 2)
        ub_pro_num_max = self._method.get_align_num(ub_pro_num_max, align_num, False)
        ub_sort_num = ub_pro_num_max // 2
        return ub_pro_num_max, ub_sort_num

    @property
    def merge_channel_num(self):
        """
        merge channel num
        """
        return self._merge_channel_num

    def get_pro_num_info(self, fp16_index_max=2048):
        """
        get pro num info
        """
        each_loop_index_num = self._ub_sort_num
        while each_loop_index_num > fp16_index_max:
            each_loop_index_num //= 2
        return self._ub_pro_num_max, self._ub_sort_num, each_loop_index_num

    # 'pylint: disable=too-many-arguments,too-many-locals
    def get_top_proposal(self, result_proposal, batch_index, boxes_num, sort_num, get_proposal_ub, args):
        """
        algorithm: sort proposal
        Parameters
        ----------
        result_proposal: A gm Tensor. shape: (n, proposal, 8)
        batch_index: int/Scalar(int)
        boxes_num: int
        sort_num: int
        get_proposal_ub: fun to get ub_proposal_1, ub_proposal_2
        args: param of get_proposal_ub
        Returns
        -------
        """
        sort_num_align = self._method.get_align_num(sort_num, self._pro_repeat_num)
        proposal_num = self._method.get_align_num(boxes_num, self._pro_repeat_num)
        sort_num_align = min(sort_num_align, proposal_num)
        if boxes_num <= self._ub_pro_num_max:
            self._get_top_proposal_l1(
                result_proposal, batch_index, 0, boxes_num, proposal_num, 0,
                sort_num_align, get_proposal_ub, args)
        elif sort_num_align <= self._ub_sort_num:
            self._get_top_proposal_l1(
                result_proposal, batch_index, 0, self._ub_pro_num_max,
                self._ub_pro_num_max, 0, sort_num_align,
                get_proposal_ub, args)
            self._sort_other_proposal(
                result_proposal, batch_index, boxes_num, sort_num_align,
                get_proposal_ub, args)
        else:
            raise RuntimeError("merge sort not support")

    # 'pylint: disable=too-many-arguments,too-many-locals
    def _get_top_proposal_l1(self, result_proposal, batch_index, result_index,
                             boxes_num, proposal_num, start_index,
                             sort_num_align, get_proposal_ub, args):
        with self._tik_inst.new_stmt_scope():
            ub_proposal_1, ub_proposal_2 = get_proposal_ub(
                boxes_num, start_index, *args)
            ub_proposal_1, ub_proposal_2 = self._merge_sort_element(
                ub_proposal_1, ub_proposal_2, proposal_num, 1)
            block_move_out = self._method.ceil_div(
                sort_num_align, self._block_pro_num)
            self._tik_inst.data_move(
                result_proposal[batch_index, result_index, 0], ub_proposal_1,
                0, 1, block_move_out, 0, 0)

    def _merge_sort_element(self, ub_proposal_1, ub_proposal_2, proposal_num, element_count=1):
        # sort 16 proposal
        if self._method.ceil_div(proposal_num, element_count) == 1:
            return ub_proposal_1, ub_proposal_2
        if element_count == 1:
            ub_proposal_1, ub_proposal_2 = self._sort(ub_proposal_1, ub_proposal_2, proposal_num)
            element_count = 16
        repeat_times = proposal_num // (element_count * self._merge_channel_num)
        # merge element_count count;
        ub_proposal_1, ub_proposal_2, element_count = \
            self._merge_sort_channel_all(ub_proposal_1, ub_proposal_2, repeat_times, element_count, proposal_num)
        valid_not_merge, element_count_last = self._method.get_loop_info(proposal_num, element_count)
        if valid_not_merge > 1:
            element_count_list = list((element_count for _ in range(valid_not_merge)))
            element_count_list[-1] = element_count_last
            ub_proposal_1, ub_proposal_2 = self._merge_proposal_element_list(ub_proposal_1,
                                                                             ub_proposal_2,
                                                                             element_count_list)
        return ub_proposal_1, ub_proposal_2

    def _sort(self, ub_proposal_1, ub_proposal_2, proposal_num):
        repeat_data_max = (self._pro_repeat_num * self._repeat_time_max)
        repeat_times = proposal_num // self._pro_repeat_num
        loop_times, last_loop_repeat = self._method.get_loop_info(
            repeat_times, self._repeat_time_max)
        for index_i in range(loop_times):
            proposal_index = index_i * repeat_data_max
            if index_i != loop_times - 1:
                self._tik_inst.vrpsort16(ub_proposal_2[proposal_index, 0],
                                         ub_proposal_1[proposal_index, 0],
                                         self._repeat_time_max)
            else:
                self._tik_inst.vrpsort16(ub_proposal_2[proposal_index, 0],
                                         ub_proposal_1[proposal_index, 0],
                                         last_loop_repeat)
        ub_proposal_1, ub_proposal_2 = ub_proposal_2, ub_proposal_1
        return ub_proposal_1, ub_proposal_2

    def _merge_proposal_element_list(self, ub_proposal_1, ub_proposal_2, element_list):
        """
        Based on element_list sort ub_proposal
        """
        element_count_list = [0] * self._merge_channel_num
        index_start_all = [0] * self._merge_channel_num
        index_start = 0
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
        src_list = list((ub_proposal_1[index_temp, 0] for index_temp in
                         index_start_all))
        self._tik_inst.vmrgsort4(ub_proposal_2, src_list, element_count_list,
                                 False, self._valid_bit_all.get(valid_num), 1)
        ub_proposal_1, ub_proposal_2 = ub_proposal_2, ub_proposal_1
        return ub_proposal_1, ub_proposal_2

    # 'pylint: disable=too-many-arguments,too-many-locals
    def _merge_sort_channel_all(self, ub_proposal_1, ub_proposal_2,
                                repeat_times, element_count, proposal_num):
        """
        merge sort
        """
        while repeat_times > 0 and element_count < self._element_count_max_num:
            # sort align proposal
            src_list = list((ub_proposal_1[index_i * element_count, 0] for index_i in range(self._merge_channel_num)))
            element_count_list = list((element_count for _ in range(self._merge_channel_num)))
            self._tik_inst.vmrgsort4(ub_proposal_2, src_list, element_count_list,
                                     False, 15, repeat_times)
            # sort last proposal
            merged_proposal_num = element_count * self._merge_channel_num * repeat_times
            proposal_not_merge_num = proposal_num - merged_proposal_num
            valid_not_merge, element_count_last = self._method.get_loop_info(proposal_not_merge_num, element_count)
            if valid_not_merge == 1:
                self._tik_inst.data_move(
                    ub_proposal_2[merged_proposal_num, 0],
                    ub_proposal_1[merged_proposal_num, 0], 0, 1,
                    proposal_not_merge_num // self._block_pro_num, 0, 0)
            elif valid_not_merge > 1:
                src_list = list((ub_proposal_1[0] for _ in range(self._merge_channel_num)))
                element_count_list = list((0 for _ in range(self._merge_channel_num)))
                for index_i in range(valid_not_merge):
                    index_add = index_i * element_count
                    src_list[index_i] = ub_proposal_1[merged_proposal_num + index_add, 0]
                    element_count_list[index_i] = element_count
                element_count_list[valid_not_merge - 1] = element_count_last
                self._tik_inst.vmrgsort4(ub_proposal_2[merged_proposal_num, 0], src_list,
                                         element_count_list, False, self._valid_bit_all.get(valid_not_merge), 1)
            element_count *= 4
            repeat_times = proposal_num // (element_count * 4)
            ub_proposal_1, ub_proposal_2 = ub_proposal_2, ub_proposal_1
        return ub_proposal_1, ub_proposal_2, element_count

    # 'pylint: disable=too-many-arguments,too-many-locals
    def _sort_other_proposal(self, result_proposal, batch_index, boxes_num,
                             sort_num_align, get_proposal_ub, args):
        last_boxes_num = boxes_num - self._ub_pro_num_max
        each_ub_merge_times = self._ub_pro_num_max // sort_num_align - 1
        each_loop_boxes_num = each_ub_merge_times * self._ub_pro_num_max
        loop_times, last_loop_boxes_num = self._method.get_loop_info(
            last_boxes_num, each_loop_boxes_num)
        with self._tik_inst.for_range(0, loop_times) as loop_index:
            start_index_gm = self._ub_pro_num_max + loop_index * each_loop_boxes_num
            with self._tik_inst.if_scope(loop_index != loop_times - 1):
                self._get_merge_sort_out(
                    result_proposal, batch_index, each_loop_boxes_num,
                    start_index_gm, sort_num_align, get_proposal_ub, args)
            with self._tik_inst.else_scope():
                self._get_merge_sort_out(
                    result_proposal, batch_index, last_loop_boxes_num,
                    start_index_gm, sort_num_align, get_proposal_ub, args)

    # 'pylint: disable=too-many-arguments,too-many-locals
    def _get_merge_sort_out(self, result_proposal, batch_index, boxes_num,
                            start_index_gm, sort_num_align,
                            get_proposal_ub, args):
        each_merge_sort_num = sort_num_align
        each_merge_boxes_num = self._ub_pro_num_max
        merge_times, last_merge_boxes_num = self._method.get_loop_info(boxes_num, each_merge_boxes_num)
        last_merge_proposal_num = self._method.get_align_num(last_merge_boxes_num, self._pro_repeat_num)
        sort_num_align_last = min(last_merge_proposal_num, sort_num_align)

        with self._tik_inst.for_range(0, merge_times) as merge_index:
            with self._tik_inst.new_stmt_scope():
                index_gm = start_index_gm + merge_index * each_merge_boxes_num
                index_out = sort_num_align + merge_index * each_merge_sort_num
                with self._tik_inst.if_scope(merge_index != merge_times - 1):
                    self._get_top_proposal_l1(result_proposal, batch_index, index_out,
                                              each_merge_boxes_num, each_merge_boxes_num,
                                              index_gm, sort_num_align, get_proposal_ub, args)
                with self._tik_inst.else_scope():
                    self._get_top_proposal_l1(result_proposal, batch_index, index_out,
                                              last_merge_boxes_num, last_merge_proposal_num,
                                              index_gm, sort_num_align_last, get_proposal_ub, args)
        sort_num_all = sort_num_align * merge_times + sort_num_align_last
        self._merge_sort_out_mode(result_proposal, batch_index, sort_num_all, sort_num_align)

    # 'pylint: disable=too-many-arguments,too-many-locals
    def _merge_sort_out_mode(self, result_proposal, batch_index, sort_num_all, sort_num_align):
        block_move_in = sort_num_all // self._block_pro_num
        ub_proposal_1 = self._tik_inst.Tensor(self._data_type, (self._ub_pro_num_max, self._pro_data_num),
                                              self._tik.scope_ubuf, "ub_proposal_1_ms")
        ub_proposal_2 = self._tik_inst.Tensor(self._data_type, (self._ub_pro_num_max, self._pro_data_num),
                                              self._tik.scope_ubuf, "ub_proposal_2_ms")
        self._tik_inst.data_move(ub_proposal_1, result_proposal[batch_index, 0, 0],
                                 0, 1, block_move_in, 0, 0)
        ub_proposal_1, ub_proposal_2 = self._merge_sort_element(ub_proposal_1, ub_proposal_2,
                                                                sort_num_all, sort_num_align)
        block_move_out = sort_num_align // self._block_pro_num
        self._tik_inst.data_move(result_proposal[batch_index, 0, 0], ub_proposal_1, 0, 1, block_move_out, 0, 0)

    # 'pylint: disable=too-many-arguments,too-many-locals
    def get_top_proposal_large(self, dst_gm, src_gm, batch_index, boxes_num,
                               sort_num, get_proposal_ub, args):
        """
        algorithm: sort proposal
        Parameters
        ----------
        dst_gm: A gm Tensor. shape: (n, proposal + tail_num, 8) tail_num >= 2.
        src_gm: A gm Tensor. shape: (n, proposal + tail_num, 8) tail_num >= 2
        batch_index: int/Scalar(int)
        boxes_num: int
        sort_num: int
        get_proposal_ub:  to get ub_proposal_1, ub_proposal_2
        args: param of get_proposal_ub
        Returns
        -------
        """
        sort_num_align = self._method.get_align_num(
            sort_num, self._pro_repeat_num)
        proposal_num = self._method.get_align_num(
            boxes_num, self._pro_repeat_num)
        sort_num_align = min(sort_num_align, proposal_num)
        if (boxes_num <= self._ub_pro_num_max or
                sort_num_align <= self._ub_sort_num):
            raise RuntimeError("merge sort not support")
        loop_times, last_loop_boxes_num = self._method.get_loop_info(
            boxes_num, self._ub_pro_num_max)
        last_loop_proposal_num = self._method.get_align_num(
            last_loop_boxes_num, self._pro_repeat_num)
        each_loop_ori_num = self._ub_pro_num_max
        each_loop_sort_num = min(self._ub_pro_num_max, sort_num_align)
        last_loop_sort_num = min(each_loop_sort_num, last_loop_proposal_num)
        src_gm_sorted_num = list((each_loop_sort_num for _ in range(loop_times)))
        src_gm_sorted_num[-1] = last_loop_sort_num
        loop_info_all = self.large_get_loop_info(sort_num_align, src_gm_sorted_num)
        if len(loop_info_all) % 2 == 1:
            dst_gm, src_gm = src_gm, dst_gm
        with self._tik_inst.for_range(0, loop_times) as loop_index:
            result_index = loop_index * each_loop_sort_num
            start_index = loop_index * each_loop_ori_num
            with self._tik_inst.if_scope(loop_index != loop_times - 1):
                self._get_top_proposal_l1(
                    dst_gm, batch_index, result_index,
                    self._ub_pro_num_max, self._ub_pro_num_max,
                    start_index, each_loop_sort_num, get_proposal_ub, args)
            with self._tik_inst.else_scope():
                self._get_top_proposal_l1(
                    dst_gm, batch_index, result_index,
                    last_loop_boxes_num, last_loop_proposal_num,
                    start_index, last_loop_sort_num, get_proposal_ub, args)

        self._merge_sort_large_num(
            dst_gm, src_gm, batch_index, sort_num_align, loop_info_all)

    # 'pylint: disable=too-many-arguments,too-many-locals
    def _merge_sort_large_num(self, dst_gm, src_gm, batch_index, sort_num_align, loop_info_all):
        for loop_info in loop_info_all:
            _, c_num_next, sorted_num, next_sorted_num, src_gm_rem_list, last_src_gm_rem_list = loop_info
            dst_gm, src_gm = src_gm, dst_gm
            with self._tik_inst.for_range(0, c_num_next) as merge_index:
                result_index = merge_index * next_sorted_num
                start_index = merge_index * sorted_num * self._merge_channel_num
                with self._tik_inst.if_scope(merge_index != c_num_next - 1):
                    self.merge_sort_gm_loop(
                        dst_gm, src_gm, batch_index,
                        result_index, start_index, src_gm_rem_list,
                        sort_num_align, sorted_num)
                with self._tik_inst.else_scope():
                    self.merge_sort_gm_loop(
                        dst_gm, src_gm, batch_index,
                        result_index, start_index, last_src_gm_rem_list,
                        sort_num_align, sorted_num)

    # 'pylint: disable=too-many-arguments,too-many-locals
    def large_get_loop_info(self, sort_num_align, src_gm_sorted_num):
        """
        get recursion info
        """
        loop_info_all = []
        while len(src_gm_sorted_num) > 1:
            c_num = len(src_gm_sorted_num)
            sorted_num = src_gm_sorted_num[0]
            c_num_next, last_c_num = self._method.get_loop_info(
                c_num, self._merge_channel_num)

            src_gm_rem_list = [sorted_num] * self._merge_channel_num
            last_c_index = c_num - last_c_num
            last_src_gm_rem_list = src_gm_sorted_num[last_c_index:]

            next_sorted_num = min(sum(src_gm_rem_list), sort_num_align)
            last_next_sorted_num = \
                min(sum(last_src_gm_rem_list), sort_num_align)

            loop_info_all.append((c_num, c_num_next, sorted_num,
                                  next_sorted_num, src_gm_rem_list,
                                  last_src_gm_rem_list))
            src_gm_sorted_num = [next_sorted_num] * c_num_next
            src_gm_sorted_num[-1] = last_next_sorted_num
        return loop_info_all

    # 'pylint: disable=too-many-arguments,too-many-locals
    def merge_sort_gm_loop(self, dst_gm, src_gm, batch_index,
                           result_index, start_index, src_gm_rem_list,
                           sort_num_align, sorted_num):
        """
        algorithm: merge sort once
        Parameters
        ----------
        dst_gm: A gm Tensor. shape: (n, proposal + tail_num, 8) tail_num >= 2.
        src_gm: A gm Tensor. shape: (n, proposal + tail_num, 8) tail_num >= 2
        batch_index: int/Scalar(int)
        result_index: int
        start_index: int
        src_gm_rem_list: gm proposal sorted index
        sort_num_align: result sort num
        sorted_num: sorted num
        Returns
        -------
        """
        if len(src_gm_rem_list) == 1:
            pro_num = sum(src_gm_rem_list)
            self._large_src_to_dst(
                dst_gm, src_gm, batch_index, result_index, start_index,
                pro_num)
        else:
            with self._tik_inst.new_stmt_scope():
                self._merge_sort_gm_channels(
                    dst_gm, src_gm, batch_index,
                    result_index, start_index, src_gm_rem_list,
                    sort_num_align, sorted_num)

    # 'pylint: disable=too-many-arguments,too-many-locals
    def _large_src_to_dst(self, dst_gm, src_gm, batch_index, result_index,
                          start_index, pro_num):
        """
        src to dst
        """
        each_loop_pro_num = self._ub_pro_num_max * 2
        with self._tik_inst.new_stmt_scope():
            ub_pro = self._tik_inst.Tensor(
                self._data_type, (each_loop_pro_num, self._pro_data_num),
                self._tik.scope_ubuf, "ub_pro_1_ms")
            loop_times, last_loop_pro_num = self._method.get_loop_info(
                pro_num, each_loop_pro_num)
            each_loop_block_num = each_loop_pro_num // self._block_pro_num
            last_loop_block_num = last_loop_pro_num // self._block_pro_num
            with self._tik_inst.for_range(0, loop_times) as loop_index:
                start_index_loop = start_index + loop_index * each_loop_pro_num
                result_index_loop = (result_index +
                                     loop_index * each_loop_pro_num)
                with self._tik_inst.if_scope(loop_index != loop_times - 1):
                    self._tik_inst.data_move(
                        ub_pro, src_gm[batch_index, start_index_loop, 0],
                        0, 1, each_loop_block_num, 0, 0)
                    self._tik_inst.data_move(
                        dst_gm[batch_index, result_index_loop, 0], ub_pro,
                        0, 1, each_loop_block_num, 0, 0)
                with self._tik_inst.else_scope():
                    self._tik_inst.data_move(
                        ub_pro, src_gm[batch_index, start_index_loop, 0],
                        0, 1, last_loop_block_num, 0, 0)
                    self._tik_inst.data_move(
                        dst_gm[batch_index, result_index_loop, 0], ub_pro,
                        0, 1, last_loop_block_num, 0, 0)

    # 'pylint: disable=too-many-arguments,too-many-locals
    def _merge_sort_gm_channels(self, dst_gm, src_gm, batch_index,
                                result_index_start_, start_index_,
                                src_gm_rem_list, sort_num_align, sorted_num):
        """
        Based info, merge sort each channel
        """
        list_num = len(src_gm_rem_list)
        src_ub = self._tik_inst.Tensor(
            self._data_type, (self._ub_pro_num_max, self._pro_data_num),
            self._tik.scope_ubuf, "src_ub_ms")
        dst_ub = self._tik_inst.Tensor(
            self._data_type, (self._ub_pro_num_max, self._pro_data_num),
            self._tik.scope_ubuf, "dst_ub_ms")
        ub_data_num = self._ub_pro_num_max // self._merge_channel_num
        (valid_bit_, list_index_, block_num_, selected_num_sum_, selected_num_, rem_num_all_, result_index_) = \
            self._large_init_scalar(src_gm_rem_list, sort_num_align)

        (src_gm_idx_list_, src_gm_rem_list_, src_ub_list_, src_ub_num_list_, slot_map_list_, dst_ub_num_list_) = \
            self._large_init_scalar_list(src_ub, ub_data_num, src_gm_rem_list, start_index_, sorted_num)
        sort_num_align = min(sort_num_align, sum(src_gm_rem_list))
        loop_times = sort_num_align // ub_data_num + self._merge_channel_num
        with self._tik_inst.for_range(0, loop_times):
            with self._tik_inst.if_scope(selected_num_sum_ < sort_num_align):
                valid_bit_, slot_map_list_, src_ub_list_ = \
                    self._large_gm_to_ub(
                        src_gm, batch_index,
                        (list_num, ub_data_num, valid_bit_, list_index_, block_num_),
                        (src_gm_idx_list_, src_gm_rem_list_, src_ub_list_, src_ub_num_list_, slot_map_list_))
                with self._tik_inst.if_scope(valid_bit_ > 0):
                    with self._tik_inst.if_scope(valid_bit_ > 1):
                        self._tik_inst.vmrgsort4(
                            dst_ub, src_ub_list_, src_ub_num_list_,
                            True, valid_bit_, 1, dst_ub_num_list_)
                        self._large_update_scalar(
                            (list_num, selected_num_, valid_bit_),
                            (src_gm_idx_list_, src_gm_rem_list_, dst_ub_num_list_, slot_map_list_))
                        self._large_ub_to_gm(
                            dst_gm, dst_ub, batch_index,
                            (result_index_start_, result_index_, selected_num_,
                             selected_num_sum_, block_num_, rem_num_all_))
                    with self._tik_inst.else_scope():
                        self._large_update_scalar(
                            (list_num, selected_num_, valid_bit_),
                            (src_gm_idx_list_, src_gm_rem_list_, src_ub_num_list_, slot_map_list_))
                        self._large_ub_to_gm(
                            dst_gm, src_ub_list_[0], batch_index,
                            (result_index_start_, result_index_, selected_num_,
                             selected_num_sum_, block_num_, rem_num_all_))

    def _large_init_scalar(self, src_gm_rem_list, sort_num_align):
        """
        init scalar
        """
        valid_bit_ = self._tik_inst.Scalar(self._int_type, init_value=0)
        list_index_ = self._tik_inst.Scalar(self._int_type, init_value=0)
        block_num_ = self._tik_inst.Scalar(self._int_type, init_value=0)
        selected_num_sum_ = self._tik_inst.Scalar(self._int_type, init_value=0)
        selected_num_ = self._tik_inst.Scalar(self._int_type, init_value=0)
        rem_num_all = min(sum(src_gm_rem_list), sort_num_align)
        rem_num_all_ = self._tik_inst.Scalar(self._int_type, init_value=rem_num_all)
        result_index_ = self._tik_inst.Scalar(self._int_type, init_value=0)
        return valid_bit_, list_index_, block_num_, selected_num_sum_, selected_num_, rem_num_all_, result_index_

    # 'pylint: disable=too-many-arguments,too-many-locals
    def _large_init_scalar_list(self, src_ub, ub_data_num, src_gm_rem_list,
                                start_index_, sorted_num):
        """
        init scalar list
        """
        list_num = len(src_gm_rem_list)
        src_gm_idx_list_ = list((self._tik_inst.Scalar(
            self._int_type, init_value=start_index_ + slot_idx * sorted_num)
            for slot_idx in range(list_num)))
        src_gm_rem_list_ = list((self._tik_inst.Scalar(
            self._int_type, init_value=rem_num)
            for rem_num in src_gm_rem_list))
        src_ub_list_ = list((
            src_ub[list_idx * ub_data_num, 0]
            for list_idx in range(self._merge_channel_num)))
        src_ub_num_list_ = list((
            self._tik_inst.Scalar(self._int_type, init_value=0)
            for _ in range(self._merge_channel_num)))
        slot_map_list_ = list((
            self._tik_inst.Scalar(self._int_type, init_value=0)
            for _ in range(self._merge_channel_num)))
        dst_ub_num_list_ = list((
            self._tik_inst.Scalar(self._int_type, init_value=0)
            for _ in range(self._merge_channel_num)))
        return (src_gm_idx_list_, src_gm_rem_list_, src_ub_list_,
                src_ub_num_list_, slot_map_list_, dst_ub_num_list_)

    def _tik_min(self, num_result, num_1, num_2):
        with self._tik_inst.if_scope(num_1 < num_2):
            num_result.set_as(num_1)
        with self._tik_inst.else_scope():
            num_result.set_as(num_2)

    def _get_pro_block_num(self, pro_num):
        if self._block_pro_num == 1:
            return pro_num
        return (pro_num + (self._block_pro_num - 1)) // self._block_pro_num

    # 'pylint: disable=too-many-arguments,too-many-locals
    def _large_gm_to_ub(self, src_gm, batch_index,
                        const_all, const_list_all):
        """
        based on scalar, gm -> ub
        """
        list_num, ub_data_num, valid_bit_, list_index_, block_num_ = const_all
        (src_gm_idx_list_, src_gm_rem_list_, src_ub_list_, src_ub_num_list_,
         slot_map_list_) = const_list_all
        valid_bit_.set_as(0)
        list_index_.set_as(0)
        for list_idx in range(list_num):
            with self._tik_inst.if_scope(src_gm_rem_list_[list_idx] > 0):
                slot_map_list_[list_idx].set_as(list_index_)
                self._large_gm_to_ub_move(
                    src_gm, batch_index, list_num, ub_data_num, valid_bit_, list_index_, block_num_,
                    src_gm_idx_list_, src_gm_rem_list_, src_ub_list_, src_ub_num_list_, list_idx)
                list_index_.set_as(list_index_ + 1)
        return valid_bit_, slot_map_list_, src_ub_list_

    # 'pylint: disable=too-many-arguments,too-many-locals
    def _large_gm_to_ub_move(self, src_gm, batch_index, list_num, ub_data_num, valid_bit_, list_index_, block_num_,
                             src_gm_idx_list_, src_gm_rem_list_, src_ub_list_, src_ub_num_list_, list_idx):
        """
        based on scalar, gm -> ub
        """
        for slot_idx in range(list_num):
            with self._tik_inst.if_scope(slot_idx == list_index_):
                self._tik_min(
                    src_ub_num_list_[slot_idx],
                    ub_data_num,
                    src_gm_rem_list_[list_idx])
                valid_bit_.set_as(valid_bit_ + 2 ** slot_idx)
                block_num_.set_as(
                    self._get_pro_block_num(
                        src_ub_num_list_[slot_idx]))
                self._tik_inst.data_move(
                    src_ub_list_[slot_idx],
                    src_gm[batch_index, src_gm_idx_list_[list_idx], 0],
                    0, 1, block_num_, 0, 0)

    def _large_update_scalar(self, const_all, const_list_all):
        """
        updata scalar
        """
        list_num, selected_num_, valid_bit_ = const_all
        src_gm_idx_list_, src_gm_rem_list_, dst_ub_num_list_, slot_map_list_ = const_list_all
        selected_num_.set_as(0)
        for slot_idx in range(list_num):
            with self._tik_inst.if_scope(valid_bit_ & (0x01 << slot_idx)):
                selected_num_.set_as(selected_num_ + dst_ub_num_list_[slot_idx])
                self._large_update_scalar_start(
                    list_num, src_gm_idx_list_, src_gm_rem_list_,
                    dst_ub_num_list_, slot_map_list_, slot_idx)

    # 'pylint: disable=too-many-arguments,too-many-locals
    def _large_update_scalar_start(self, list_num, src_gm_idx_list_, src_gm_rem_list_,
                                   dst_ub_num_list_, slot_map_list_, slot_idx):
        """
        updata scalar
        """
        for list_idx in range(list_num):
            with self._tik_inst.if_scope(slot_map_list_[list_idx] == slot_idx):
                src_gm_idx_list_[list_idx].set_as(src_gm_idx_list_[list_idx] + dst_ub_num_list_[slot_idx])
                src_gm_rem_list_[list_idx].set_as(src_gm_rem_list_[list_idx] - dst_ub_num_list_[slot_idx])

    def _large_ub_to_gm(self, dst_gm, dst_ub, batch_index, const_all):
        """
        based on scalar, ub -> gm
        """
        (result_index_start_, result_index_, selected_num_, selected_num_sum_,
         block_num_, rem_num_all_) = const_all
        self._tik_min(selected_num_, selected_num_, rem_num_all_)
        block_num_.set_as(self._get_pro_block_num(selected_num_))

        result_index_.set_as(result_index_start_ + selected_num_sum_)
        with self._tik_inst.if_scope(block_num_ > 0):
            self._tik_inst.data_move(
                dst_gm[batch_index, result_index_, 0], dst_ub,
                0, 1, block_num_, 0, 0)
        selected_num_sum_.set_as(selected_num_sum_ + selected_num_)
        rem_num_all_.set_as(rem_num_all_ - selected_num_)
