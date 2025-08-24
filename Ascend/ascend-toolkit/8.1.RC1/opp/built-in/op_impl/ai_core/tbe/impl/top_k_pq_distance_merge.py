#!/usr/bin/env python
# coding: utf-8
# Copyright 2020 Huawei Technologies Co., Ltd
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
top_k_pq_distance_merge
"""
import math

from impl import common_util
from impl import constant_util as constant
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik

PROPOSAL_NUM = 8
MIN_VAL = -65504
MASK_FP32 = 64


# 'pylint: disable=too-many-arguments
# 'pylint: disable=too-few-public-methods
class TopKPQDistanceMerge(object):
    """class for top_k_pq_distance_merge"""

    def __init__(self, sorted_distance, pq_ivf, pq_index, topk_distance, topk_ivf, topk_index, k, kernel_name):
        self.kernel_name = kernel_name
        self.sorted_distance_dtype = sorted_distance.get("dtype").lower()
        self.sorted_distance_shape = sorted_distance.get("shape")
        self.pq_ivf_dtype = pq_ivf.get("dtype").lower()
        self.pq_ivf_shape = pq_ivf.get("shape")
        self.pq_index_dtype = pq_index.get("dtype").lower()
        self.pq_index_shape = pq_index.get("shape")

        self.topk_distance_dtype = topk_distance.get("dtype").lower()
        self.topk_distance_shape = topk_distance.get("shape")
        self.topk_ivf_dtype = topk_ivf.get("dtype").lower()
        self.topk_ivf_shape = topk_ivf.get("shape")
        self.topk_index_dtype = topk_index.get("dtype").lower()
        self.topk_index_shape = topk_index.get("shape")

        self.topk = k
        self.element_num = self.topk
        self._check_input_param()
        self.tik_instance = tik.Tik()
        self.aicore_num = 1

        # get vector computer parameters
        dtype_size_distance = common_util.get_data_size(self.sorted_distance_dtype)
        dtype_size_pd_index_ivf = common_util.get_data_size(self.topk_ivf_dtype)
        distance_block_num = constant.BLOCK_SIZE // dtype_size_distance
        index_ivf_block_num = constant.BLOCK_SIZE // dtype_size_pd_index_ivf

        # align with 16
        self.handle_num_align_16 = math.ceil(self.element_num / distance_block_num) * distance_block_num
        self.distance_burst_len = self.handle_num_align_16 // distance_block_num
        self.index_ivf_burst_len = self.handle_num_align_16 // index_ivf_block_num
        self._init_gm_tensor()

    def _check_input_param(self):
        """
        Check the input parameter
        -------
        None

        returns:
        None
        """
        check_list = ("float16")
        para_check.check_dtype(self.sorted_distance_dtype, check_list, param_name="sorted_distance")
        para_check.check_dtype(self.topk_distance_dtype, check_list, param_name="topk_distance")
        dtype_list = (self.pq_ivf_dtype, self.pq_index_dtype, self.topk_ivf_dtype, self.topk_index_dtype)
        for dtype in dtype_list:
            para_check.check_dtype(dtype, ("int32"), param_name="ivf_index")
        para_check.check_shape(self.sorted_distance_shape, param_name="sorted_distance")
        para_check.check_shape(self.topk_distance_shape, param_name="topk_distance")
        para_check.check_shape(self.pq_ivf_shape, param_name="pq_ivf")
        para_check.check_shape(self.pq_index_shape, param_name="pq_index")
        para_check.check_shape(self.topk_ivf_shape, param_name="topk_ivf")
        para_check.check_shape(self.topk_index_shape, param_name="topk_index")

        para_check.check_kernel_name(self.kernel_name)
        if self.topk > 1024:
            raise RuntimeError("attr k value is no greater than 1024.")
        if self.sorted_distance_shape[0] > 2048:
            raise RuntimeError("sorted_distance_shape is no greater than 2048.")
        if self.sorted_distance_shape[0] != 2 * self.topk:
            raise RuntimeError("sorted_distance_shape is not 2 * k.")

    def _init_gm_tensor(self):
        """
        Declare tensor on gm
        parameters
        -----
        none

        return:
        None
        """
        self.gm_sorted_distance = self.tik_instance.Tensor(self.sorted_distance_dtype, self.sorted_distance_shape,
                                                           name="sorted_distance_gm", scope=tbe_platform.scope_gm)
        self.gm_pq_ivf = self.tik_instance.Tensor(self.pq_ivf_dtype, self.pq_ivf_shape,
                                                  name="pq_ivf_gm", scope=tbe_platform.scope_gm)
        self.gm_pq_index = self.tik_instance.Tensor(self.pq_index_dtype, self.pq_index_shape,
                                                    name="pq_index_gm", scope=tbe_platform.scope_gm)
        self.gm_topk_distance = self.tik_instance.Tensor(self.topk_distance_dtype, self.topk_distance_shape,
                                                         name="top_distance_gm", scope=tbe_platform.scope_gm)
        self.gm_topk_ivf = self.tik_instance.Tensor(self.topk_ivf_dtype, self.topk_ivf_shape,
                                                    name="topk_ivf_gm", scope=tbe_platform.scope_gm)
        self.gm_topk_index = self.tik_instance.Tensor(self.topk_index_dtype, self.topk_index_shape,
                                                      name="ptopk_index_gm", scope=tbe_platform.scope_gm)

    # 'pylint: disable=too-many-locals
    def topk_pq_distance_merge_compute(self):
        sorted_distance_ub = self.tik_instance.Tensor(self.sorted_distance_dtype, (4 * self.handle_num_align_16,),
                                                      name="sorted_distance_ub", scope=tbe_platform.scope_ubuf)
        sorted_distance_ub_conv = self.tik_instance.Tensor("float32", (4 * self.handle_num_align_16,),
                                                           name="sorted_distance_ub_conv",
                                                           scope=tbe_platform.scope_ubuf)
        # ivf,index keep the float32 for performance,int32 dtype size the same as float32
        pq_ivf_ub = self.tik_instance.Tensor("float32", (2 * self.handle_num_align_16,),
                                             name="pq_ivf_ub", scope=tbe_platform.scope_ubuf)
        pq_index_ub = self.tik_instance.Tensor("float32", (2 * self.handle_num_align_16,),
                                               name="pq_index_ub", scope=tbe_platform.scope_ubuf)
        # data move to ub
        if self.element_num == self.handle_num_align_16:
            self.tik_instance.data_move(sorted_distance_ub, self.gm_sorted_distance, 0, 1, 2 * self.distance_burst_len,
                                        0, 0)
            self.tik_instance.data_move(pq_ivf_ub, self.gm_pq_ivf, 0, 1, 2 * self.index_ivf_burst_len, 0, 0)
            self.tik_instance.data_move(pq_index_ub, self.gm_pq_index, 0, 1, 2 * self.index_ivf_burst_len, 0, 0)
        else:
            self.tik_instance.data_move(sorted_distance_ub, self.gm_sorted_distance, 0, 1, self.distance_burst_len, 0,
                                        0)
            self.tik_instance.data_move(sorted_distance_ub[self.handle_num_align_16],
                                        self.gm_sorted_distance[self.element_num],
                                        0, 1, self.distance_burst_len, 0, 0)
            self.tik_instance.data_move(pq_ivf_ub, self.gm_pq_ivf, 0, 1, self.index_ivf_burst_len, 0, 0)
            self.tik_instance.data_move(pq_ivf_ub[self.handle_num_align_16],
                                        self.gm_pq_ivf[self.element_num],
                                        0, 1, self.index_ivf_burst_len, 0, 0)
            self.tik_instance.data_move(pq_index_ub, self.gm_pq_index, 0, 1, self.index_ivf_burst_len, 0, 0)
            self.tik_instance.data_move(pq_index_ub[self.handle_num_align_16],
                                        self.gm_pq_index[self.element_num],
                                        0, 1, self.index_ivf_burst_len, 0, 0)

        # fp16 -> fp32
        conv_repeat_times = 4 * self.handle_num_align_16 // MASK_FP32
        self.tik_instance.vec_conv(MASK_FP32, 'none', sorted_distance_ub_conv, sorted_distance_ub, \
                                   conv_repeat_times, 8, 4)
        min_fp = self.tik_instance.Scalar('float32', init_value=MIN_VAL)
        # only need set sorted_distance padding, ivf and index no need process
        with self.tik_instance.for_range(0, self.handle_num_align_16 - self.element_num) as i:
            sorted_distance_ub_conv[self.element_num + i].set_as(min_fp)
            sorted_distance_ub_conv[self.handle_num_align_16 + self.element_num + i].set_as(min_fp)

        vconcat_proposal_ub = self.tik_instance.Tensor("float32", (2 * self.handle_num_align_16 * PROPOSAL_NUM,),
                                                       name="vconcat_proposal_ub", scope=tbe_platform.scope_ubuf)
        vconcat_repeats_num = 2 * self.handle_num_align_16 // 16
        self.tik_instance.vconcat(vconcat_proposal_ub, sorted_distance_ub_conv, vconcat_repeats_num, 4)
        self.tik_instance.vconcat(vconcat_proposal_ub, pq_ivf_ub, vconcat_repeats_num, 0)
        self.tik_instance.vconcat(vconcat_proposal_ub, pq_index_ub, vconcat_repeats_num, 1)

        src_list = [vconcat_proposal_ub[0],
                    vconcat_proposal_ub[self.handle_num_align_16 * PROPOSAL_NUM],
                    vconcat_proposal_ub[0],
                    vconcat_proposal_ub[0]]
        src_list_lengths = [self.handle_num_align_16, self.handle_num_align_16, 0, 0]

        vconcat_proposal_dest_ub = self.tik_instance.Tensor("float32",
                                                            (2 * self.handle_num_align_16 * PROPOSAL_NUM,),
                                                            name="vconcat_proposal_dest_ub",
                                                            scope=tbe_platform.scope_ubuf)

        self.tik_instance.vmrgsort4(vconcat_proposal_dest_ub, src_list, src_list_lengths,
                                    if_exhausted_suspension=False,
                                    valid_bit="0011", repeat_times=1)
        vextract_repeats_num = self.handle_num_align_16 // 16
        self.tik_instance.vextract(sorted_distance_ub_conv, vconcat_proposal_dest_ub, vextract_repeats_num, 4)
        self.tik_instance.vextract(pq_ivf_ub, vconcat_proposal_dest_ub, vextract_repeats_num, 0)
        self.tik_instance.vextract(pq_index_ub, vconcat_proposal_dest_ub, vextract_repeats_num, 1)
        conv_repeat_times = 4 * self.handle_num_align_16 // MASK_FP32

        # fp32->fp16
        self.tik_instance.vec_conv(MASK_FP32, 'none', sorted_distance_ub, sorted_distance_ub_conv, \
                                   conv_repeat_times, 4, 8)

        # data move out
        self.tik_instance.data_move(self.gm_topk_distance, sorted_distance_ub, 0, 1, self.distance_burst_len, 0, 0)
        self.tik_instance.data_move(self.gm_topk_ivf, pq_ivf_ub, 0, 1, self.index_ivf_burst_len, 0, 0)
        self.tik_instance.data_move(self.gm_topk_index, pq_index_ub, 0, 1, self.index_ivf_burst_len, 0, 0)

        input_list = [self.gm_sorted_distance, self.gm_pq_ivf, self.gm_pq_index]
        output_list = [self.gm_topk_distance, self.gm_topk_ivf, self.gm_topk_index]

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=input_list, outputs=output_list)
        return self.tik_instance


# 'pylint: disable=too-many-arguments
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_INT, para_check.KERNEL_NAME)
def top_k_pq_distance_merge(sorted_distance, pq_ivf, pq_index, topk_distance,
                            topk_ivf, topk_index, k, kernel_name="top_k_pq_distance_merge"):
    """
    Fuction: top_k_pq_distance_merge

    Parameters
    ----------
    input(sorted_distance): dict
          data of input sorted_distance
    input(pq_ivf): dict
          data of input pq_ivf
    input(pq_index): dict
          data of input pq_index
    output(topk_distance): dict
          data of output topk_distance
    output(topk_ivf): dict
          data of output topk_ivf
    output(topk_index): dict
          data of output topk_index
    k: int
          top k value
    kernel_name: str
           the name of operator
    ----------
    """
    topk_pq_distance_merge = TopKPQDistanceMerge(sorted_distance, pq_ivf, pq_index, topk_distance, topk_ivf, topk_index,
                                                 k, kernel_name)
    return topk_pq_distance_merge.topk_pq_distance_merge_compute()
