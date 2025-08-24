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
segment_sort_v2
"""
from impl.merge_sort_v2 import MergeSort, BaseConstant, ceil_div, get_align_num, emit_vmuls
from impl.util.platform_adapter import PlatformApi
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import error_manager_vector


def get_mask(data_num, format_num_floor, format_num_ceil):
    """
    Get mask and start index of the tik command.
    Parameters
    ----------
    data_num: int.
    format_num_floor: int.
    format_num_ceil: int.

    Returns
    -------
    mask_h:
        int. The first mask of the command.
    mask_l:
        int. The second mask of the command.
    data_num_floor:
        int. The start index of the command.
    """
    mask_split = 64
    data_num_floor = get_align_num(data_num, format_num_floor, False)
    data_num_ceil = get_align_num(data_num, format_num_ceil, True)
    mask_h, mask_l = 0, 0
    index_start = data_num - data_num_floor
    index_end = data_num_ceil - data_num_floor
    for index_l in range(index_start, min(index_end, mask_split)):
        mask_l += 2 ** index_l
    for index_h in range(max(mask_split, index_start), index_end):
        mask_h += 2 ** (index_h - mask_split)
    return mask_h, mask_l, data_num_floor


class SegmentSort:
    """
    define SegmentSort
    """

    def __init__(self, score_shape, index_num, score_type, index_type, k_num,
                 proposal_shape_result, largest, kernel_name):
        self.score_num = score_shape[-1]
        self.index_num = index_num
        self.k_num = k_num
        self.score_type = score_type
        self.index_type = index_type
        self.kernel_name = kernel_name
        self.largest = largest

        self.tik = tik
        self.tik_inst = self.tik.Tik()
        self.ub_size = PlatformApi.get_soc_spec(PlatformApi.UB_SIZE)
        self.const_value = BaseConstant(self.score_type, "uint32", self.ub_size, kernel_name)
        self.merge_sort = MergeSort(self.tik, self.tik_inst, self.const_value)

        self.proposal_num_tail = self.const_value.proposal_num_repeat
        self.element_num_proposal = self.const_value.element_num_proposal
        self.ai_core_use_num = proposal_shape_result[0]
        self.proposal_num_channel = proposal_shape_result[1] - self.proposal_num_tail
        self.channel_num = ceil_div(self.score_num, self.proposal_num_channel)

        self.input_score = self.tik_inst.Tensor(self.score_type, score_shape, self.tik.scope_gm, "input_score")
        self.input_index = self.tik_inst.Tensor(self.index_type, (self.index_num,), self.tik.scope_gm, "input_index")
        self.temp_proposal = self.tik_inst.Tensor(self.score_type, proposal_shape_result,
                                                  self.tik.scope_gm, "temp_proposal",
                                                  is_workspace=True)
        self.output_proposal = self.tik_inst.Tensor(self.score_type, proposal_shape_result,
                                                    self.tik.scope_gm, "output_proposal")

    def mode_compute(self):
        """main compute"""
        each_channel_score_num = self.proposal_num_channel
        last_channel_score_num = self.score_num - each_channel_score_num * (self.channel_num - 1)
        with self.tik_inst.for_range(0, self.ai_core_use_num, block_num=self.ai_core_use_num) as core_index:
            with self.tik_inst.if_scope(core_index < self.channel_num):
                score_index_start = each_channel_score_num * core_index
                with self.tik_inst.if_scope(core_index != self.channel_num - 1):
                    self._mode_compute_each_core(core_index, score_index_start, each_channel_score_num)
                with self.tik_inst.else_scope():
                    self._mode_compute_each_core(core_index, score_index_start, last_channel_score_num)
            with self.tik_inst.else_scope():
                self._set_tail(core_index, 0)
        inputs_all = [self.input_score, self.input_index]
        outputs_all = [self.output_proposal]
        self.tik_inst.BuildCCE(
            inputs=inputs_all,
            outputs=outputs_all,
            kernel_name=self.kernel_name)

    def _mode_compute_each_core(self, core_index, score_index_start, score_num):
        fun_args = {
            "score_index_start": score_index_start
        }

        self.merge_sort.merge_sort_start(self.output_proposal, self.temp_proposal,
                                         core_index, core_index,
                                         self.k_num, score_num,
                                         self.get_score_index_tensor, fun_args)
        self._set_tail(core_index, score_num)

    def get_score_index_tensor(self, fun_args):
        score_index_start = fun_args.get("score_index_start")
        score_num_loop = fun_args.get("score_num_loop")
        proposal_num_loop = fun_args.get("proposal_num_loop")
        score_index_loop = fun_args.get("score_index_loop")
        score_index = score_index_start + score_index_loop

        tensor_shape_ub = (proposal_num_loop, )
        index_ub = self.tik_inst.Tensor(self.index_type, tensor_shape_ub, self.tik.scope_ubuf, "index_ub")
        self._init_index_channel(index_ub, score_index, proposal_num_loop)

        score_ub = self.tik_inst.Tensor(self.score_type, tensor_shape_ub, self.tik.scope_ubuf, "score_ub")
        self._init_score_channel(score_ub, score_index, score_num_loop)
        index_ub = index_ub.reinterpret_cast_to("uint32")
        return score_ub, index_ub

    def _init_index_channel(self, index_ub, score_index, proposal_num):
        mask = self.const_value.index_num_repeat
        if proposal_num <= mask:
            block_num = proposal_num // self.const_value.index_num_block
            self.tik_inst.data_move(index_ub, self.input_index, 0, 1, block_num, 0, 0)
            self.tik_inst.vadds(proposal_num, index_ub, index_ub, score_index, 1, 1, 1, 8, 8)
        else:
            index_num = get_align_num(min(self.index_num, proposal_num), mask, False)
            block_num = index_num // self.const_value.index_num_block
            repeat_num = index_num // mask
            self.tik_inst.data_move(index_ub, self.input_index, 0, 1, block_num, 0, 0)
            self.tik_inst.vadds(mask, index_ub, index_ub, score_index, repeat_num, 1, 1, 8, 8)
            while index_num < proposal_num:
                if proposal_num >= index_num * 2:
                    repeat_num = index_num // mask
                    self.tik_inst.vadds(mask, index_ub[index_num], index_ub, index_num, repeat_num, 1, 1, 8, 8)
                else:
                    proposal_num_not_add = proposal_num - index_num
                    repeat_num = proposal_num_not_add // mask
                    if repeat_num > 0:
                        self.tik_inst.vadds(mask, index_ub[index_num], index_ub, index_num, repeat_num, 1, 1, 8, 8)
                    proposal_num_last = proposal_num_not_add - repeat_num * mask
                    if proposal_num_last > 0:
                        start_index = repeat_num * mask + index_num
                        self.tik_inst.vadds(proposal_num_last, index_ub[start_index],
                                            index_ub, start_index, 1, 1, 1, 8, 8)
                index_num *= 2

    def _init_score_channel(self, score_ub, score_index, score_num):
        block_num = ceil_div(score_num, self.const_value.score_num_block)
        self.tik_inst.data_move(score_ub, self.input_score[score_index], 0, 1, block_num, 0, 0)
        if not self.largest:
            emit_vmuls(self.tik_inst, score_ub, score_ub, score_num, self.score_type)
        align_num = self.const_value.proposal_num_repeat
        mask_h, mask_l, index_last = get_mask(score_num, align_num, align_num)
        if mask_h != 0 or mask_l != 0:
            self.tik_inst.vector_dup([mask_h, mask_l], score_ub[index_last], self.const_value.neg_inf, 1, 1, 8)

    def _set_tail(self, core_index, sorted_num):
        element_num_tail = self.proposal_num_tail * self.element_num_proposal
        element_num_tail_align = get_align_num(element_num_tail, self.const_value.score_num_repeat)
        block_num = element_num_tail // self.const_value.score_num_block
        repeat_num = element_num_tail_align // self.const_value.score_num_repeat

        ne_inf_ub = self.tik_inst.Tensor(self.score_type, (element_num_tail_align, ), self.tik.scope_ubuf, "ne_inf_ub")
        self.tik_inst.vector_dup(self.const_value.score_num_repeat, ne_inf_ub, self.const_value.neg_inf,
                                 repeat_num, 1, 8)
        result_index = min(sorted_num, self.k_num)
        self.tik_inst.data_move(self.output_proposal[core_index, result_index, 0], ne_inf_ub, 0, 1, block_num, 0, 0)


def check_param(input_score, input_index, k_num, kernel_name):
    """checking input params"""
    score_type = input_score.get("dtype").lower()
    index_type = input_index.get("dtype").lower()
    score_type_supported = ("float16", "float32")
    index_type_supported = ("int32",)
    assert score_type in score_type_supported, \
        error_manager_vector.raise_err_input_dtype_not_supported(kernel_name,
                                                                 "score",
                                                                 str(score_type_supported),
                                                                 score_type)
    assert index_type in index_type_supported, \
        error_manager_vector.raise_err_input_dtype_not_supported(kernel_name,
                                                                 "index",
                                                                 str(index_type_supported),
                                                                 index_type)
    score_shape = input_score.get("shape")
    score_excepted_shape = "len of input shape must be 1"
    rows = 1
    for i in range(len(score_shape) - 1):
        rows *= int(score_shape[i])
    assert rows == 1, \
        error_manager_vector.raise_err_input_value_invalid(kernel_name,
                                                           "input_score",
                                                           score_excepted_shape,
                                                           score_shape)
    index_shape = input_index.get("shape")
    index_excepted_shape = "len of input shape must be 1 and must be divisible by 64"
    assert len(index_shape) == 1 and index_shape[0] % 64 == 0, \
        error_manager_vector.raise_err_input_value_invalid(kernel_name,
                                                           "input_index",
                                                           index_excepted_shape,
                                                           index_shape)
    assert isinstance(k_num, int) and k_num > 0, \
        error_manager_vector.raise_err_input_value_invalid(kernel_name,
                                                           "k",
                                                           "must be positive integer",
                                                           k_num)


def segment_sort_v2(input_score, input_index, output_proposal, k_num, largest, kernel_name="SegmentSort"):
    """
    algorithm: Segment merge sort on multiple core
    Parameters
    ----------
    input_score:
        A Tensor. Data to be sorted. Support float16, float32
    input_index :
        A Tensor. Range(0, index). Support int32.
    output_proposal:
        A Tensor. Datatype and format is same as input_data. Proposal sorted for each channel.
    k_num: int
        Number to be sorted.
    largest: An optional bool
        Controls whether to return largest or smallest elements. Defaults to true.
        If "True", the "k" largest elements are returned in descending order.
        If "False", the "k" smallest elements are returned in ascending order.
    kernel_name : str
        cce kernel name, default value is SegmentSort
    Returns
    -------
    None
    """
    check_param(input_score, input_index, k_num, kernel_name)
    score_shape = input_score.get("shape")
    index_num = input_index.get("shape")[0]
    score_type = input_score.get("dtype").lower()
    index_type = input_index.get("dtype").lower()
    proposal_shape_result = output_proposal.get("shape")
    obj = SegmentSort(score_shape, index_num, score_type, index_type, k_num,
                      proposal_shape_result, largest, kernel_name)
    obj.mode_compute()
