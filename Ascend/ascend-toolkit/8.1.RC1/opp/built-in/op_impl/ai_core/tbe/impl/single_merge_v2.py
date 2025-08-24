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
single_merge
"""
from impl.merge_sort_v2 import ProposalMergeGM, BaseConstant, ceil_div, get_align_num, get_loop_info, emit_vmuls
from impl.util.platform_adapter import PlatformApi
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check


# 'pylint: disable=too-many-arguments,too-many-locals,too-few-public-methods
class SingleMerge:
    """method to merge and sort on single core"""

    def __init__(self, input_shape, out_shape, k_num, score_type, largest, kernel_name):
        self.k_num = k_num
        self.score_type = score_type
        self.kernel_name = kernel_name
        self.largest = largest

        self.tik = tik
        self.tik_inst = self.tik.Tik()
        self.ub_size = PlatformApi.get_soc_spec(PlatformApi.UB_SIZE)
        self.const_value = BaseConstant(self.score_type, kernel_name=kernel_name)
        self.proposal_merge_gm = ProposalMergeGM(self.tik, self.tik_inst, self.const_value)

        self.proposal_num_repeat = self.const_value.proposal_num_repeat
        self.proposal_num_tail = self.const_value.proposal_num_repeat
        self.element_num_proposal = self.const_value.element_num_proposal
        self.channel_num_merge = input_shape[0]
        self.channel_num_src = input_shape[0]
        self.channel_num_dst = 1
        self.sort_num_src = input_shape[1]
        self.sort_num_dst = self._get_sort_num_dst()

        input_proposal_shape = (self.channel_num_dst,
                                self.sort_num_src * self.channel_num_merge,
                                self.element_num_proposal)
        output_proposal_shape = (self.channel_num_dst,
                                 self.sort_num_dst + self.proposal_num_tail,
                                 self.element_num_proposal)

        self.input_proposal = self.tik_inst.Tensor(self.score_type, input_proposal_shape,
                                                   self.tik.scope_gm, "input_proposal")
        self.temp_proposal = self.tik_inst.Tensor(self.score_type, output_proposal_shape,
                                                  self.tik.scope_gm, "temp_proposal",
                                                  is_workspace=True)

        self.index_type = "int32"
        self.output_data = self.tik_inst.Tensor(self.score_type, out_shape,
                                                self.tik.scope_gm, "output_data")
        self.output_index = self.tik_inst.Tensor(self.index_type, out_shape,
                                                 self.tik.scope_gm, "output_index")

    def _get_sort_num_dst(self):
        """
        count result sort num
        """
        k_num_align = get_align_num(self.k_num, self.proposal_num_repeat)
        proposal_num_dst = self.sort_num_src * self.channel_num_merge
        proposal_num_dst = min(proposal_num_dst, k_num_align)
        return proposal_num_dst

    def mode_compute(self):
        """main compute mode"""
        self._mode_compute_each_core()
        inputs_all = [self.input_proposal]
        outputs_all = [self.output_data, self.output_index]
        self.tik_inst.BuildCCE(
            inputs=inputs_all,
            outputs=outputs_all,
            kernel_name=self.kernel_name)

    def _mode_compute_each_core(self):
        """
        compute each core
        """
        src_gm_rem_list = tuple(self.sort_num_src for _ in range(self.channel_num_merge))
        self.proposal_merge_gm.proposal_merge_gm_start(self.temp_proposal, self.input_proposal,
                                                       0, 0, 0, 0,
                                                       self.sort_num_dst, src_gm_rem_list)

        each_loop_data_num = self._get_loop_data_num()
        loop_time, last_loop_data_num = get_loop_info(self.k_num, each_loop_data_num)
        with self.tik_inst.for_range(0, loop_time) as loop_index:
            data_index_start = loop_index * each_loop_data_num
            with self.tik_inst.if_scope(loop_index != loop_time - 1):
                self._result_move_out_each_loop(data_index_start, each_loop_data_num)
            with self.tik_inst.else_scope():
                self._result_move_out_each_loop(data_index_start, last_loop_data_num)

    def _get_loop_data_num(self):
        """
        Calculate the amount of data processed in each loop
        """
        each_data_size = self.const_value.proposal_size + self.const_value.score_size + self.const_value.index_size
        data_num = self.ub_size // each_data_size
        max_data_num = (self.const_value.repeat_num_max_cmd * self.const_value.score_num_repeat
                        // self.element_num_proposal)
        data_num = min(max_data_num, data_num)
        data_num_align = get_align_num(data_num, self.const_value.score_num_repeat, False)
        return data_num_align

    def _result_move_out_each_loop(self, proposal_index_start, proposal_num):
        """
        move out result
        """
        proposal_num_align = get_align_num(proposal_num, self.const_value.score_num_repeat)
        element_num_move_in = proposal_num * self.element_num_proposal
        element_num_align = proposal_num_align * self.element_num_proposal

        proposal_shape_ub = (element_num_align, )
        proposal_ub = self.tik_inst.Tensor(self.score_type, proposal_shape_ub, self.tik.scope_ubuf, "proposal_ub")
        block_num_move_in = ceil_div(element_num_move_in, self.const_value.score_num_block)
        self.tik_inst.data_move(proposal_ub, self.temp_proposal[0, proposal_index_start, 0],
                                0, 1, block_num_move_in, 0, 0)

        score_shape = (proposal_num_align,)
        index_type = "uint32"
        score_ub = self.tik_inst.Tensor(self.score_type, score_shape, self.tik.scope_ubuf, "score_ub")
        index_ub = self.tik_inst.Tensor(index_type, score_shape, self.tik.scope_ubuf, "score_ub")
        if self.score_type == "float16":
            score_src1_pattern = 3
        else:
            score_src1_pattern = 1
        index_src1_pattern = 2
        repeat_time = element_num_align // self.const_value.score_num_repeat
        self.tik_inst.vreduce(self.const_value.score_num_repeat, score_ub, proposal_ub,
                              score_src1_pattern, repeat_time, 1, 8, 0)
        index_proposal_ub = proposal_ub.reinterpret_cast_to(index_type)
        self.tik_inst.vreduce(self.const_value.index_num_repeat, index_ub, index_proposal_ub,
                              index_src1_pattern, repeat_time, 1, 8, 0)
        index_ub = index_ub.reinterpret_cast_to(self.index_type)

        if not self.largest:
            emit_vmuls(self.tik_inst, score_ub, score_ub, proposal_num, self.score_type)
        score_block_num = ceil_div(proposal_num, self.const_value.score_num_block)
        index_block_num = ceil_div(proposal_num, self.const_value.index_num_block)
        self.tik_inst.data_move(self.output_data[proposal_index_start], score_ub, 0, 1, score_block_num, 0, 0)
        self.tik_inst.data_move(self.output_index[proposal_index_start], index_ub, 0, 1, index_block_num, 0, 0)


# 'pylint: disable=unused-argument
def single_merge_v2(input_proposal, output_proposal, output_index, k_num, include_index, largest, kernel_name):
    """algorithm: merge and sort on single core
    Parameters
    ----------
    input_proposal:
        A Tensor. Proposal sorted for each channel. Support float16
    output_data :
        A Tensor. Datatype and format is same as input_data. Data sorted.
    output_index:
        A Tensor. int32. Data index.
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
    input_shape = input_proposal.get("shape")
    input_dtype = input_proposal.get("dtype").lower()
    input_format = input_proposal.get("format")
    para_check.check_shape(input_shape, param_name="input_proposal")
    para_check.check_format(input_format)
    out_shape = output_proposal.get("shape")
    para_check.check_shape(out_shape, param_name="output_proposal")

    obj = SingleMerge(input_shape, out_shape, k_num, input_dtype, largest, kernel_name)
    obj.mode_compute()
