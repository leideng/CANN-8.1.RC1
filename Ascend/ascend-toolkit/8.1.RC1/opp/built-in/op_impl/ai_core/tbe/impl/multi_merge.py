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
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector

from impl.ascend import AContainer
from impl.merge_sort import CommonMethod
from impl.merge_sort import MergeSort
from impl.util import util_select_op_base
from impl.single_merge import single_merge
from impl.util.platform_adapter import PlatformApi
from impl.multi_merge_v2 import multi_merge_v2
from impl.merge_sort_v2 import check_soc_version_support


# 'pylint: disable=unused-argument
def op_select_format(input_proposal, output_proposal, output_index, k_num,
                     include_index=False, kernel_name="MultiMerge"):
    """
    select format dynamically
    """
    soc_version = PlatformApi.get_soc_spec(PlatformApi.SHORT_SOC_VERSION)
    if check_soc_version_support(soc_version, ("Ascend910B",)):
        input0 = util_select_op_base.gen_param(classify="input0", name="input_proposal",
                                               datatype="float16,float",
                                               format="ND,ND")
        output0 = util_select_op_base.gen_param(classify="output0", name="output_proposal",
                                                datatype="float16,float",
                                                format="ND,ND")
        output1 = util_select_op_base.gen_param(classify="output1", name="output_index",
                                                datatype="int32,int32",
                                                format="ND,ND")
    else:
        input0 = util_select_op_base.gen_param(classify="input0", name="input_proposal",
                                               datatype="float16",
                                               format="ND")
        output0 = util_select_op_base.gen_param(classify="output0", name="output_proposal",
                                                datatype="float16",
                                                format="ND")
        output1 = util_select_op_base.gen_param(classify="output1", name="output_index",
                                                datatype="int32",
                                                format="ND")
    param_list = [input0, output0, output1]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class MultiMerge:
    """
    MultiMerge
    """
    # 'pylint: disable=too-many-arguments
    def __init__(self, input_shape, output_shape, k_num, data_type, kernel_name, cont):
        self.sorted_num = input_shape[1]
        self.data_type = data_type
        self.kernel_name = kernel_name

        self.cont = cont
        self.tik = self.cont.tik
        self.tik_inst = self.cont.tinst
        self.ub_size = 253952
        self.pro_data_num = self.cont.const_proposal_data_num
        self.pro_repeat_num = self.cont.const_proposal_repeat_num

        self.method = CommonMethod(self.cont)
        self.data_size, self.block_data_num, self.repeat_data_num = self.method.get_type_const(self.data_type)
        self.merge_sort = MergeSort(self.cont, self.data_type, self.ub_size)

        self.tail_proposal_num = self.pro_repeat_num
        self.merge_channel = min(self.merge_sort.merge_channel_num, input_shape[0])
        self.fp16_ne_inf = -(2**16 - 1.0)

        self.result_shape = output_shape
        self.ai_core_use = output_shape[0]
        self.channel_num = input_shape[0] // self.merge_channel
        self.k_num = output_shape[1] - self.pro_repeat_num

        input_proposal_shape = (self.channel_num, input_shape[1] * self.merge_channel, self.pro_data_num)
        self.input_proposal = self.tik_inst.Tensor(self.data_type, input_proposal_shape,
                                                   self.tik.scope_gm, "input_proposal")
        self.output_proposal = self.tik_inst.Tensor(self.data_type, self.result_shape,
                                                    self.tik.scope_gm, "output_proposal")
        self.output_index = self.tik_inst.Tensor("int32", (1, ), self.tik.scope_gm, "output_index")

    def mode_compute(self):
        """
        compute function
        """
        with self.tik_inst.for_range(0, self.ai_core_use, block_num=self.ai_core_use) as core_index:
            with self.tik_inst.if_scope(core_index < self.channel_num):
                self._mode_compute_each_core(core_index)
            with self.tik_inst.else_scope():
                self._set_tail(core_index, 0)
        inputs_all = [self.input_proposal]
        outputs_all = [self.output_proposal, self.output_index]
        self.tik_inst.BuildCCE(
            inputs=inputs_all,
            outputs=outputs_all,
            kernel_name=self.kernel_name)

    def _mode_compute_each_core(self, core_index):
        src_gm_rem_list = [self.sorted_num for _ in range(self.merge_channel)]
        self.merge_sort.merge_sort_gm_loop(self.output_proposal, self.input_proposal,
                                           core_index, 0, 0, src_gm_rem_list, self.k_num, self.sorted_num)
        self._set_tail(core_index, self.k_num)

    def _set_tail(self, core_index, data_num):
        tail_proposal_num = self.pro_repeat_num
        tail_num = tail_proposal_num * self.pro_data_num
        block_num = self.method.ceil_div(tail_num, self.block_data_num)
        ne_inf_ub = self.tik_inst.Tensor(self.data_type, (tail_num,), self.tik.scope_ubuf, "ne_inf_ub")
        self.method.vector_dup(ne_inf_ub, self.fp16_ne_inf)
        result_index = min(data_num, self.k_num)
        self.tik_inst.data_move(self.output_proposal[core_index, result_index, 0], ne_inf_ub, 0, 1, block_num, 0, 0)


def check_params(input_proposal, kernel_name):
    """
    check params
    """
    input_shape = input_proposal.get("shape")
    if len(input_shape) != 3 or input_shape[0] % 4 != 0 or input_shape[2] != 8:
        error_manager_vector.raise_err_input_value_invalid(
            kernel_name, "shape of input_proposal", "support", "not support")


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
# 'pylint: disable=unused-argument
def multi_merge(input_proposal, output_proposal, output_index, k_num, include_index=False, largest=True,
                kernel_name="MultiMerge"):
    """
    algorithm: merge and sort on single core
    Parameters
    ----------
    input_proposal:
        A Tensor. Proposal sorted for each channel. Support float16
    output_proposal :
        A Tensor. Datatype and format is same as input_data. Data sorted.
    output_index:
        A Tensor. if include_index is true, output index.
    k_num: int
        Number to be sorted.
    largest: An optional bool
        Controls whether to return largest or smallest elements. Defaults to true.
        If "True", the "k" largest elements are returned in descending order.
        If "False", the "k" smallest elements are returned in ascending order.
    include_index: bool
        include_index is false,output proposal. include_index is true, output data and index
    kernel_name : str
        cce kernel name, default value is MultiMerge
    Returns
    -------
    None
    """
    soc_version = PlatformApi.get_soc_spec(PlatformApi.SHORT_SOC_VERSION)
    if check_soc_version_support(soc_version, ("Ascend910B",)):
        multi_merge_v2(input_proposal, output_proposal, output_index, k_num, include_index, largest, kernel_name)
    elif include_index:
        single_merge(input_proposal, output_proposal, output_index, k_num, largest, kernel_name)
    else:
        proposal_shape_result = output_proposal.get("shape")
        input_shape = input_proposal.get("shape")
        input_dtype = input_proposal.get("dtype").lower()
        input_format = input_proposal.get("format")
        check_list = ("float16",)
        para_check.check_dtype(input_dtype, check_list, param_name="input_proposal")
        para_check.check_shape(input_shape, param_name="input_proposal")
        para_check.check_format(input_format)
        check_params(input_proposal, kernel_name)
        AContainer.reset_instance()
        cont = AContainer.get_instance()
        obj = MultiMerge(input_shape, proposal_shape_result, k_num, input_dtype, kernel_name, cont)
        obj.mode_compute()
