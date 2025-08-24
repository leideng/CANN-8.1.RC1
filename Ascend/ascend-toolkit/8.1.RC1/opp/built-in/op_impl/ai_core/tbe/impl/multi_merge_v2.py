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
multi_merge
"""
from impl.merge_sort_v2 import ProposalMergeGM, BaseConstant, ceil_div, get_align_num
from impl.util.platform_adapter import PlatformApi
from impl.util.platform_adapter import tik
from impl.single_merge_v2 import single_merge_v2


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class MultiMerge:
    """
    MultiMerge
    """

    def __init__(self, input_shape, output_shape, k_num, score_type, kernel_name):
        self.k_num = k_num
        self.score_type = score_type
        self.kernel_name = kernel_name

        self.tik = tik
        self.tik_inst = self.tik.Tik()
        self.ub_size = PlatformApi.get_soc_spec(PlatformApi.UB_SIZE)
        self.const_value = BaseConstant(self.score_type, kernel_name=kernel_name)
        self.proposal_merge_gm = ProposalMergeGM(self.tik, self.tik_inst, self.const_value)

        self.proposal_num_tail = self.const_value.proposal_num_repeat
        self.element_num_proposal = self.const_value.element_num_proposal
        self.channel_num_src = input_shape[0]
        self.channel_num_merge = min(self.const_value.channel_num_max_sort, self.channel_num_src)
        self.channel_num_dst = self.channel_num_src // self.channel_num_merge
        self.sort_num_src = input_shape[1]
        self.sort_num_dst = output_shape[1] - self.proposal_num_tail
        self.ai_core_use_num = output_shape[0]

        input_proposal_shape = (self.channel_num_dst,
                                self.sort_num_src * self.channel_num_merge,
                                self.element_num_proposal)

        self.input_proposal = self.tik_inst.Tensor(self.score_type, input_proposal_shape,
                                                   self.tik.scope_gm, "input_proposal")
        self.output_proposal = self.tik_inst.Tensor(self.score_type, output_shape,
                                                    self.tik.scope_gm, "output_proposal")
        self.output_index = self.tik_inst.Tensor("int32", (1,), self.tik.scope_gm, "output_index")

    def mode_compute(self):
        """
        compute function
        """
        with self.tik_inst.for_range(0, self.ai_core_use_num, block_num=self.ai_core_use_num) as core_index:
            with self.tik_inst.if_scope(core_index < self.channel_num_dst):
                self._mode_compute_each_core(core_index)
            with self.tik_inst.else_scope():
                self._set_tail(core_index, 0)
        outputs_all = [self.output_proposal, self.output_index]
        inputs_all = [self.input_proposal]
        self.tik_inst.BuildCCE(
            outputs=outputs_all,
            inputs=inputs_all,
            kernel_name=self.kernel_name)

    def _mode_compute_each_core(self, core_index):
        src_gm_rem_list = tuple(self.sort_num_src for _ in range(self.channel_num_merge))
        self.proposal_merge_gm.proposal_merge_gm_start(self.output_proposal, self.input_proposal,
                                                       core_index, core_index,
                                                       0, 0,
                                                       self.sort_num_dst, src_gm_rem_list)
        self._set_tail(core_index, self.sort_num_dst)

    def _set_tail(self, core_index, sorted_num):
        element_num_tail = self.proposal_num_tail * self.element_num_proposal
        element_num_tail_align = get_align_num(element_num_tail, self.const_value.score_num_repeat)
        block_num = element_num_tail // self.const_value.score_num_block
        repeat_num = element_num_tail_align // self.const_value.score_num_repeat

        ne_inf_ub = self.tik_inst.Tensor(self.score_type, (element_num_tail_align,), self.tik.scope_ubuf, "ne_inf_ub")
        self.tik_inst.vector_dup(self.const_value.score_num_repeat, ne_inf_ub, self.const_value.neg_inf,
                                 repeat_num, 1, 8)
        result_index = min(sorted_num, self.k_num)
        self.tik_inst.data_move(self.output_proposal[core_index, result_index, 0], ne_inf_ub, 0, 1, block_num, 0, 0)


# 'pylint: disable=unused-argument
def multi_merge_v2(input_proposal, output_proposal, output_index, k_num, include_index, largest, kernel_name):
    """
    algorithm: merge and sort on single core
    Parameters
    ----------
    input_proposal:
        A Tensor. Proposal sorted for each channel. Support float16
    output_proposal :
        A Tensor. Datatype and format is same as input_data. Data sorted.
    k_num: int
        Number to be sorted.
    largest: An optional bool
        Controls whether to return largest or smallest elements. Defaults to true.
        If "True", the "k" largest elements are returned in descending order.
        If "False", the "k" smallest elements are returned in ascending order.
    kernel_name : str
        cce kernel name, default value is top_k_3
    Returns
    -------
    None
    """
    if include_index:
        single_merge_v2(input_proposal, output_proposal, output_index, k_num, include_index, largest, kernel_name)
    else:
        input_shape = input_proposal.get("shape")
        input_dtype = input_proposal.get("dtype").lower()
        proposal_shape_result = output_proposal.get("shape")
        obj = MultiMerge(input_shape, proposal_shape_result, k_num, input_dtype, kernel_name)
        obj.mode_compute()
