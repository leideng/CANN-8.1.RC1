#!/usr/bin/env python
# coding: utf-8
# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
non_max_suppression
"""
import math
import functools
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.batch_multi_class_nms_topk import sort_within_ub
from impl.batch_multi_class_nms_topk import sort_with_ub
from impl.non_max_suppression_v7_without_proposal import non_max_suppression_v7_without_proposal
from impl.non_max_suppression_v7_supports_fp32 import non_max_suppression_v7_supports_fp32

# 'pylint: disable=too-many-lines,too-many-instance-attributes,too-many-statements
# 'pylint: disable=too-many-arguments,unused-argument,too-many-locals,too-many-branches
# process 128 proposals at a time
BURST_PROPOSAL_NUM = 128
# RPN compute 16 proposals per iteration
RPN_PROPOSAL_NUM = 16
# output ub limit
OUTPUT_UB_LIMIT = 51200
# coordinate scaling
COORDINATE_SCALING = 0.001


class NonMaxSuppression():
    """
    Function: use to store NonMaxSuppression base parameters
    Modify : 2020-12-01
    """

    def __init__(self,
                 boxes,
                 scores,
                 max_output_size,
                 iou_thresh,
                 score_thresh,
                 index_id,
                 center_point_box,
                 max_boxes_size
                 ):
        """
        Init NonMaxSuppression base parameters

        Returns
        -------
        None
        """
        # define general var
        self.tik_instance = tik.Tik()
        self.aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.input_gm_list = []
        self.output_gm_list = []
        # for soc
        soc_version = tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION)
        if soc_version in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
            self.is_lhisi = True
        else:
            self.is_lhisi = False
        boxes_shape = list(boxes.get("shape"))
        self.box_type = boxes.get("dtype")
        scores_shape = list(scores.get("shape"))
        self.score_type = scores.get("dtype")
        self.boxes_shape = boxes_shape
        self.scores_shape = scores_shape

        if max_output_size is None:
            self.has_max_out = False
        else:
            self.has_max_out = True
            self.max_output_size_shape = list(max_output_size.get("shape"))
        self.max_output_class = 0
        self.max_total_size = 0

        if iou_thresh is None:
            self.has_iou_thresh = False
        else:
            self.has_iou_thresh = True
            self.iou_threshold_shape = list(iou_thresh.get("shape"))
        self.iou_thresh = self.tik_instance.Scalar(dtype="float16")
        self.iou_thresh.set_as(0)

        if score_thresh is None:
            self.has_score_thresh = False
        else:
            self.has_score_thresh = True
            self.score_threshold_shape = list(score_thresh.get("shape"))
        self.score_thresh = self.tik_instance.Scalar(dtype="float16")
        self.score_thresh.set_as(0)
        self.center_point_box = center_point_box
        self.index_id_shape = list(index_id.get("shape"))
        self.batch, self.classes, self.boxes_num = self.scores_shape
        # init input gm
        self.init_tik_input_mem()
        self.max_total_size = self.get_max_output_size(max_boxes_size)
        self.max_output_class = self.get_max_class_size(max_boxes_size)

        # get init value of iou_thresh,score_thresh,max_total_size
        if self.has_iou_thresh:
            self.iou_thresh = self.get_iou_threshold()
        if self.has_score_thresh:
            self.score_thresh = self.get_score_threshold()
        # calcu output shape
        self.selected_indices_shape = [self.max_total_size, 3]

        # for topk
        self.ub_max_topk = None
        self.ub_topk_index = None
        self.gm_nms_result = None
        self.l1_nms_result_zero = None
        self.workspace_proposal_gm = None
        self.workspace_proposal_index_gm = None
        self.workspace_second_nms_gm = None
        self.workspace_second_nms_index_gm = None
        self.l1_score_valid = None
        self.l1_nms_area = None
        self.l1_nms_sup = None
        self.gm_temp = None
        self.proposal_topk_k = ceil_div(2000, 16) * 16
        self.topk_loop_time = 0
        self.topk_loop_tail = 0
        self.single_loop = True
        if self.boxes_num > self.proposal_topk_k:
            self.single_loop = False
            self.topk_loop_time = self.boxes_num // self.proposal_topk_k
            self.topk_loop_tail = self.boxes_num % self.proposal_topk_k
        self.topk_loop_time_reg = self.tik_instance.Scalar(dtype="int32")
        self.topk_loop_time_reg.set_as(self.topk_loop_time)

        # whether user set_rpn_offset, mini do not support it
        self.is_need_rpn_offset = False
        self.is_second_nms = True

        # for nms function param calc
        self.max_selected_nms_num_in_ub = ceil_div(self.max_output_class, RPN_PROPOSAL_NUM) * RPN_PROPOSAL_NUM
        # record the output nms num for one class
        self.selected_proposals_cnt = self.tik_instance.Scalar(dtype="uint16")
        # record the proposal burst num for one loop, value = 128 or self.proposal_topk_k % 128
        self.handling_proposals_cnt = self.tik_instance.Scalar(dtype="uint16")
        # init a scalar value = 0
        self.zero_scalar = self.tik_instance.Scalar(dtype="uint16")
        self.zero_scalar.set_as(0)
        # init a scalar value = 1
        self.one_scalar = self.tik_instance.Scalar(dtype="uint16")
        self.one_scalar.set_as(1)
        # init a fp16 scalar for output class
        self.nms_class_idx = self.tik_instance.Scalar(dtype="float16")
        self.nms_class_idx.set_as(0)

        # init 1 valid num scalar
        self.valid_num_value = self.tik_instance.Scalar(dtype="int32")

    def get_max_output_size(self, max_box_size):
        """
        get max output_size of boxes

        Returns
        -------
        max_total_size
        """
        max_total_size_gm = self.input_gm_list[2]
        with self.tik_instance.new_stmt_scope():
            max_total_size_temp = self.tik_instance.Tensor("int32", (16,), name="max_total_size_temp",
                                                           scope=tik.scope_ubuf)
            self.tik_instance.data_move(max_total_size_temp[0], max_total_size_gm[0], 0, 1, 1, 0, 0)

        self.max_total_size = max_box_size
        return self.max_total_size

    def get_max_class_size(self, max_box_size):
        """
        get max class_size of boxes
        Returns
        -------
        max_output_class
        """
        self.max_output_class = ceil_div(
            max_box_size, self.batch * self.classes)
        return self.max_output_class

    def get_iou_threshold(self):
        """
        get iou_threshold
        Returns
        -------
        iou_thresh
        """
        # `iou_threshold = iou_thres_gm[0]/(iou_thres_gm[0] + 1)`
        iou_thres_gm = self.input_gm_list[3]
        with self.tik_instance.new_stmt_scope():
            if self.is_lhisi:
                iou_threshold_um = self.tik_instance.Tensor("float16", (16,), name="iou_threshold_um",
                                                            scope=tik.scope_ubuf)
                self.tik_instance.data_move(iou_threshold_um[0], iou_thres_gm[0], 0, 1, 1, 0, 0)
                iou_threshold_ub = self.tik_instance.Tensor("float16", (16,), name="iou_threshold_ub",
                                                            scope=tik.scope_ubuf)
                self.tik_instance.vector_dup(16, iou_threshold_ub, 1.0, 1, 1, 1)
                iou_threshold_ub[0].set_as(iou_threshold_um[0])
            else:
                iou_threshold_um = self.tik_instance.Tensor("float32", (16,), name="iou_threshold_um",
                                                            scope=tik.scope_ubuf)
                self.tik_instance.data_move(iou_threshold_um[0], iou_thres_gm[0], 0, 1, 1, 0, 0)
                iou_threshold_temp = self.tik_instance.Tensor("float32", (16,), name="iou_threshold_temp",
                                                              scope=tik.scope_ubuf)

                self.tik_instance.vector_dup(16, iou_threshold_temp, 1.0, 1, 1, 1)
                iou_threshold_temp[0].set_as(iou_threshold_um[0])

                iou_threshold_ub = self.tik_instance.Tensor("float16", (16,), name="iou_threshold_ub",
                                                            scope=tik.scope_ubuf)
                self.tik_instance.vec_conv(16, "none", iou_threshold_ub, iou_threshold_temp, 1, 1, 2)

            ub_one = self.tik_instance.Tensor("float16", (16,), name="ub_one", scope=tik.scope_ubuf)
            self.tik_instance.vector_dup(16, ub_one, 1.0, 1, 1, 1)
            add_ub = self.tik_instance.Tensor("float16", (16,), name="add_ub", scope=tik.scope_ubuf)
            self.tik_instance.vector_dup(16, add_ub, 1.0, 1, 1, 1)
            self.tik_instance.vadd(16, add_ub, iou_threshold_ub, ub_one, 1, 1, 1, 1, 1, 1, 1)
            self.tik_instance.vrec(16, ub_one, add_ub, 1, 1, 1, 1, 1)
            self.tik_instance.vmul(16, add_ub, iou_threshold_ub, ub_one, 1, 1, 1, 1, 1, 1, 1)
            self.iou_thresh.set_as(add_ub[0])
            return self.iou_thresh

    def get_score_threshold(self):
        """
        get score_threshold
        Returns
        -------
        score_thresh
        """
        scores_thres_gm = self.input_gm_list[4]
        with self.tik_instance.new_stmt_scope():
            if self.is_lhisi:
                score_threshold_temp = self.tik_instance.Tensor("float16", (16,), name="score_threshold_temp",
                                                                scope=tik.scope_ubuf)
                self.tik_instance.data_move(score_threshold_temp[0], scores_thres_gm[0], 0, 1, 1, 0, 0)
                self.score_thresh.set_as(score_threshold_temp[0])
            else:
                score_threshold_temp = self.tik_instance.Tensor("float32", (16,), name="score_threshold_temp",
                                                                scope=tik.scope_ubuf)
                self.tik_instance.data_move(score_threshold_temp[0], scores_thres_gm[0], 0, 1, 1, 0, 0)
                score_threshold_ub = self.tik_instance.Tensor("float16", (16,), name="score_threshold_ub",
                                                              scope=tik.scope_ubuf)
                self.tik_instance.vec_conv(16, "none", score_threshold_ub, score_threshold_temp, 1, 1, 2)
                self.score_thresh.set_as(score_threshold_ub[0])
            return self.score_thresh

    def init_tik_input_mem(self):
        """
        init tik gm mem
        """
        # init gm input
        boxes_gm = self.tik_instance.Tensor("float16", self.boxes_shape, name="boxes_gm", scope=tik.scope_gm)
        scores_gm = self.tik_instance.Tensor("float16", self.scores_shape, name="scores_gm", scope=tik.scope_gm)
        index_id_gm = self.tik_instance.Tensor("float16", self.index_id_shape, name="index_id_gm", scope=tik.scope_gm)
        max_output_size_gm = None
        iou_threshold_gm = None
        score_threshold_gm = None
        if self.has_max_out:
            max_output_size_gm = self.tik_instance.Tensor("int32", self.max_output_size_shape,
                                                          name="max_output_size_gm",
                                                          scope=tik.scope_gm)
        if self.has_iou_thresh:
            if self.is_lhisi:
                iou_threshold_gm = self.tik_instance.Tensor("float16", self.iou_threshold_shape,
                                                            name="iou_threshold_gm",
                                                            scope=tik.scope_gm)
            else:
                iou_threshold_gm = self.tik_instance.Tensor("float32", self.iou_threshold_shape,
                                                            name="iou_threshold_gm",
                                                            scope=tik.scope_gm)
        if self.has_score_thresh:
            if self.is_lhisi:
                score_threshold_gm = self.tik_instance.Tensor("float16", self.score_threshold_shape,
                                                              name="score_threshold_gm",
                                                              scope=tik.scope_gm)
            else:
                score_threshold_gm = self.tik_instance.Tensor("float32", self.score_threshold_shape,
                                                              name="score_threshold_gm",
                                                              scope=tik.scope_gm)
            self.input_gm_list = [boxes_gm, scores_gm, max_output_size_gm, iou_threshold_gm,
                                  score_threshold_gm, index_id_gm]
        else:
            self.input_gm_list = [boxes_gm, scores_gm, max_output_size_gm, iou_threshold_gm, index_id_gm]

    def init_tik_output_mem(self):
        """
        init tik gm mem
        """
        # init gm output
        selected_indices_gm = self.tik_instance.Tensor("int32", self.selected_indices_shape,
                                                       name="selected_indices_gm", scope=tik.scope_gm)
        dup_len = total_num(self.selected_indices_shape)

        selected_indices_ub_loop_time = dup_len // OUTPUT_UB_LIMIT
        selected_indices_ub_loop_tail = dup_len % OUTPUT_UB_LIMIT
        with self.tik_instance.for_range(0, selected_indices_ub_loop_time) as i:
            with self.tik_instance.new_stmt_scope():
                selected_indices_ub = self.tik_instance.Tensor("int32", (OUTPUT_UB_LIMIT,),
                                                               name="selected_indices_ub",
                                                               scope=tik.scope_ubuf)
                tik_func_vector_int32(self.tik_instance, selected_indices_ub, -1, OUTPUT_UB_LIMIT)
                brust_len = ceil_div(OUTPUT_UB_LIMIT, 8)
                self.tik_instance.data_move(selected_indices_gm[i*OUTPUT_UB_LIMIT:], selected_indices_ub,
                                            0, 1, brust_len, 0, 0)
        if selected_indices_ub_loop_tail != 0:
            with self.tik_instance.new_stmt_scope():
                selected_indices_ub = self.tik_instance.Tensor("int32", (selected_indices_ub_loop_tail,),
                                                               name="selected_indices_ub", scope=tik.scope_ubuf)
                tik_func_vector_int32(self.tik_instance, selected_indices_ub, -1, selected_indices_ub_loop_tail)
                brust_len = ceil_div(selected_indices_ub_loop_tail, 8)
                self.tik_instance.data_move(selected_indices_gm[selected_indices_ub_loop_time*OUTPUT_UB_LIMIT:],
                                            selected_indices_ub, 0, 1, brust_len, 0, 0)

        self.output_gm_list = [selected_indices_gm]
        # init l1 buff for save multi class nms result, size = [classes, self.max_selected_nms_num_in_ub, 8]
        self.gm_nms_result = self.tik_instance.Tensor("float16", (self.classes, self.max_selected_nms_num_in_ub, 8),
                                                      name="gm_nms_result", scope=tik.scope_gm, is_workspace=True)

        if self.is_second_nms:
            # init l1 buff for save multi class nms area, size = [self.max_selected_nms_num_in_ub]
            self.l1_nms_area = self.tik_instance.Tensor("float16", (self.max_selected_nms_num_in_ub,),
                                                        name="l1_nms_area_tmp", scope=tik.scope_cbuf)
            # init l1 buff for save multi class nms sup, size = [self.max_selected_nms_num_in_ub]
            self.l1_nms_sup = self.tik_instance.Tensor("uint16", (self.max_selected_nms_num_in_ub,),
                                                       name="l1_nms_sup_tmp", scope=tik.scope_cbuf)
        # zero data in l1
        self.l1_nms_result_zero = self.tik_instance.Tensor("float16", (self.max_selected_nms_num_in_ub, 8),
                                                           name="l1_nms_result_zero", scope=tik.scope_cbuf)
        with self.tik_instance.new_stmt_scope():
            ub_nms_result = self.tik_instance.Tensor("float16", (self.max_selected_nms_num_in_ub, 8),
                                                     name="ub_nms_result", scope=tik.scope_ubuf)

            tik_func_vector(self.tik_instance, ub_nms_result, -1,
                            self.max_selected_nms_num_in_ub * 8)
            loop_burst_len = (self.max_selected_nms_num_in_ub * 8) // 16
            self.tik_instance.data_move(self.l1_nms_result_zero,
                                        ub_nms_result, 0, 1, loop_burst_len, 0, 0)

        # workspace
        self.workspace_proposal_gm = self.tik_instance.Tensor("float16",
                                                              [self.aicore_num,
                                                               total_num(self.gm_nms_result.shape) + 128],
                                                              name="workspace_proposal_gm",
                                                              scope=tik.scope_gm, is_workspace=True)
        self.workspace_proposal_index_gm = self.tik_instance.Tensor("float16",
                                                                    [self.aicore_num,
                                                                     total_num(self.gm_nms_result.shape) + 128],
                                                                    name="workspace_proposal_index_gm",
                                                                    scope=tik.scope_gm, is_workspace=True)
        # workspace for second nms
        if self.is_second_nms:
            self.workspace_second_nms_gm = self.tik_instance.Tensor("float16",
                                                                    [self.aicore_num, self.boxes_num * 8],
                                                                    name="workspace_second_nms_gm",
                                                                    scope=tik.scope_gm, is_workspace=True)

            self.workspace_second_nms_index_gm = self.tik_instance.Tensor("float16",
                                                                          [self.aicore_num, self.boxes_num * 8],
                                                                          name="workspace_second_nms_index_gm",
                                                                          scope=tik.scope_gm, is_workspace=True)
        self.gm_temp = self.tik_instance.Tensor("float16", (16,), name="gm_temp", scope=tik.scope_gm,
                                                is_workspace=True)

    def init_tik_ub_mem_for_topk(self):
        """
        init_tik_ub_mem_for_topk
        """
        # init one ub for topk output
        self.ub_max_topk = self.tik_instance.Tensor("float16", (self.proposal_topk_k, 8),
                                                    name="ub_max_topk", scope=tik.scope_ubuf)
        self.ub_topk_index = self.tik_instance.Tensor("float16", (self.proposal_topk_k, 8),
                                                      name="ub_topk_index", scope=tik.scope_ubuf)

    def init_tik_ub_mem_for_nms(self):
        """
        init_tik_ub_mem_for_nms
        """
        ub_selected_proposals = self.tik_instance.Tensor("float16", [self.max_selected_nms_num_in_ub, 8],
                                                         name="ub_selected_proposals", scope=tik.scope_ubuf)
        ub_selected_index_proposals = self.tik_instance.Tensor("float16", [self.max_selected_nms_num_in_ub, 8],
                                                               name="ub_selected_index_proposals", scope=tik.scope_ubuf)
        ub_selected_area = self.tik_instance.Tensor("float16", [self.max_selected_nms_num_in_ub],
                                                    name="ub_selected_area", scope=tik.scope_ubuf)
        ub_sup_vec = self.tik_instance.Tensor("uint16", [self.max_selected_nms_num_in_ub], name="ub_sup_vec",
                                              scope=tik.scope_ubuf)

        topk_out_num = self.proposal_topk_k
        if self.boxes_num < self.proposal_topk_k:
            topk_out_num = self.boxes_num
        nms_var_dict = {
            # topk_out_info mean : nms input info
            "topk_out_ub": self.ub_max_topk,
            "topk_index_out_ub": self.ub_topk_index,
            "topk_out_num": topk_out_num,
            # selected proposal info
            "selected_proposal_ub": ub_selected_proposals,
            "selected_index_ub": ub_selected_index_proposals,
            "selected_area_ub": ub_selected_area,
            "sup_vec_ub": ub_sup_vec,
            # scalar reg info
            "zero_scalar": self.zero_scalar,
            "one_scalar": self.one_scalar,
            "selected_proposals_cnt": self.selected_proposals_cnt,
            "handling_proposals_cnt": self.handling_proposals_cnt,
            # nms output info
            "output_num": self.max_output_class
        }

        return nms_var_dict

    def get_core_schedule(self):
        """
        get_core_schedule
        """
        if self.max_total_size < 16:
            self.aicore_num = 1
        batch_per_core = ceil_div(self.batch, self.aicore_num)
        core_used = ceil_div(self.batch, batch_per_core)
        batch_last_core = self.batch - (core_used - 1) * batch_per_core
        self.aicore_num = core_used

        return core_used, batch_per_core, batch_last_core

    def get_tik_instance(self):
        """
        get_tik_instance
        """
        return self.tik_instance

    def build_tik_instance(self, kernel_name_value):
        """
        build_tik_instance
        """
        self.tik_instance.BuildCCE(kernel_name=kernel_name_value,
                                   inputs=self.input_gm_list,
                                   outputs=self.output_gm_list,
                                   output_files_path=None,
                                   enable_l2=False)

        return self.tik_instance


def total_num(shape):
    """total_num"""
    shape_total_num = functools.reduce(lambda a, b: a * b, shape)
    return shape_total_num


def tik_func_vconcat(tik_instance, proposals_ub, _ub, trans_repeat, mode):
    """tik_func_vconcat"""
    tik_instance.vconcat(proposals_ub, _ub, trans_repeat, mode)


def ceil_div(value, factor):
    """
    Compute the smallest integer value that is greater than
    or equal to value/factor
    """
    result = (value + (factor - 1)) // factor
    return result


def tik_func_add_sub_mul(tik_instance, v_func,
                         out_dst, src0, src1, copy_num,
                         dst_blk, src0_blk, src1_blk,
                         dst_rep, src0_rep, src1_rep):
    """tik add sub and mul"""
    repeat_time = copy_num // 128
    repeat_tail = copy_num % 128
    tik_fun = None

    if v_func == "vadd":
        tik_fun = tik_instance.vadd

    if v_func == "vsub":
        tik_fun = tik_instance.vsub

    if v_func == "vmul":
        tik_fun = tik_instance.vmul

    if repeat_time > 0:
        tik_fun(128, out_dst, src0[0], src1[0], repeat_time,
                dst_blk, src0_blk, src1_blk,
                dst_rep, src0_rep, src1_rep)

    if repeat_tail > 0:
        offset = repeat_time * 128
        tik_fun(repeat_tail, out_dst[offset], src0[offset], src1[offset], 1,
                dst_blk, src0_blk, src1_blk,
                dst_rep, src0_rep, src1_rep)


def tik_func_trans_to_proposals(tik_instance, proposals_ub, boxes_ub_list, score_ub, proposal_num):
    """tik_func_trans_to_proposals"""

    x1_ub, y1_ub, x2_ub, y2_ub = boxes_ub_list

    trans_repeat = ceil_div(proposal_num, 16)
    # concat x1 to proposals_ub
    tik_func_vconcat(tik_instance, proposals_ub, x1_ub, trans_repeat, 1)
    # concat y1 to proposals_ub
    tik_func_vconcat(tik_instance, proposals_ub, y1_ub, trans_repeat, 0)
    # concat x2 to proposals_ub
    tik_func_vconcat(tik_instance, proposals_ub, x2_ub, trans_repeat, 3)
    # concat y2 to proposals_ub
    tik_func_vconcat(tik_instance, proposals_ub, y2_ub, trans_repeat, 2)
    # concat scores to proposals_ub
    tik_func_vconcat(tik_instance, proposals_ub, score_ub, trans_repeat, 4)


def tik_func_trans_to_proposals_center_box(tik_instance, proposals_ub, boxes_ub, score_ub, proposal_num):
    """ tik_func_trans_to_proposals"""
    lens = ceil_div(proposal_num, 16) * 16
    trans_repeat = ceil_div(proposal_num, 16)
    box_len = ceil_div(lens, 16)

    x1_ub = tik_instance.Tensor("float16", [lens], name="x1_ub", scope=tik.scope_ubuf)
    y1_ub = tik_instance.Tensor("float16", [lens], name="y1_ub", scope=tik.scope_ubuf)
    x2_ub = tik_instance.Tensor("float16", [lens], name="x2_ub", scope=tik.scope_ubuf)
    y2_ub = tik_instance.Tensor("float16", [lens], name="y2_ub", scope=tik.scope_ubuf)

    tik_instance.data_move(x1_ub, boxes_ub[0, 0], 0, 1, box_len, 0, 0)
    tik_instance.data_move(y1_ub, boxes_ub[1, 0], 0, 1, box_len, 0, 0)
    tik_instance.data_move(x2_ub, boxes_ub[2, 0], 0, 1, box_len, 0, 0)
    tik_instance.data_move(y2_ub, boxes_ub[3, 0], 0, 1, box_len, 0, 0)

    half_x2_ub = tik_instance.Tensor("float16", [lens], name="half_x2_ub", scope=tik.scope_ubuf)
    half_y2_ub = tik_instance.Tensor("float16", [lens], name="half_y2_ub", scope=tik.scope_ubuf)
    half_ub = tik_instance.Tensor("float16", [lens], name="half_ub", scope=tik.scope_ubuf)

    tik_func_vector(tik_instance, half_ub, 0.5, lens)

    tik_func_add_sub_mul(tik_instance, "vmul", half_x2_ub, x2_ub, half_ub, lens, 1, 1, 1, 8, 8, 8)
    tik_func_add_sub_mul(tik_instance, "vmul", half_y2_ub, y2_ub, half_ub, lens, 1, 1, 1, 8, 8, 8)
    tik_func_add_sub_mul(tik_instance, "vsub", x2_ub, x1_ub, half_x2_ub, lens, 1, 1, 1, 8, 8, 8)
    tik_func_add_sub_mul(tik_instance, "vsub", y2_ub, y1_ub, half_y2_ub, lens, 1, 1, 1, 8, 8, 8)
    # concat x1 to proposals_ub
    tik_func_vconcat(tik_instance, proposals_ub, x2_ub, trans_repeat, 0)
    # concat y1 to proposals_ub
    tik_func_vconcat(tik_instance, proposals_ub, y2_ub, trans_repeat, 1)

    tik_func_add_sub_mul(tik_instance, "vadd", x2_ub, x1_ub, half_x2_ub, lens, 1, 1, 1, 8, 8, 8)
    tik_func_add_sub_mul(tik_instance, "vadd", y2_ub, y1_ub, half_y2_ub, lens, 1, 1, 1, 8, 8, 8)

    # concat x2 to proposals_ub
    tik_func_vconcat(tik_instance, proposals_ub, x2_ub, trans_repeat, 2)
    # concat y2 to proposals_ub
    tik_func_vconcat(tik_instance, proposals_ub, y2_ub, trans_repeat, 3)
    # concat scores to proposals_ub
    tik_func_vconcat(tik_instance, proposals_ub, score_ub, trans_repeat, 4)


def tik_func_trans_index_proposals(tik_instance, proposals_ub, index_ub_list, score_ub, proposal_num):
    """tik_func_trans_to_proposals"""
    # for index id
    batch_id_ub, class_id_ub, index_id_ub, index_tail_ub = index_ub_list
    trans_repeat = ceil_div(proposal_num, 16)
    # concat batch id to proposals_ub
    tik_func_vconcat(tik_instance, proposals_ub, batch_id_ub, trans_repeat, 0)
    # concat class id to proposals_ub
    tik_func_vconcat(tik_instance, proposals_ub, class_id_ub, trans_repeat, 1)
    # concat index_id to proposals_ub
    tik_func_vconcat(tik_instance, proposals_ub, index_id_ub, trans_repeat, 2)
    # concat index_tail to proposals_ub
    tik_func_vconcat(tik_instance, proposals_ub, index_tail_ub, trans_repeat, 3)
    # concat scores to proposals_ub
    tik_func_vconcat(tik_instance, proposals_ub, score_ub, trans_repeat, 4)


def tik_func_vector(tik_instance, _ub, value, dup_len):
    """
    tik_func_vector

    Parameters:
    ----------
    tik_instance : tik_instance.
        tik_instance
    _ub: ub
        vcetor ub
    value: value
        vcetor value
    dup_len: int
        vcetor data len

    Returns
    -------
    None
    """
    repeat = dup_len // 128
    repeat_tail = dup_len % 128
    offset = 0
    while repeat > 255:
        tik_instance.vector_dup(128, _ub[offset], value, 255, 1, 8)
        repeat = repeat - 255
        offset = offset + 128 * 255
    if repeat > 0:
        tik_instance.vector_dup(128, _ub[offset], value, repeat, 1, 8)
        offset = offset + 128 * repeat
    if repeat_tail > 0:
        tik_instance.vector_dup(repeat_tail, _ub[offset], value, 1, 1, 8)


def tik_func_vector_int32(tik_instance, _ub, value, dup_len):
    """
    tik_func_vector

    Parameters:
    ----------
    tik_instance : tik_instance.
        tik_instance
    _ub: ub
        vcetor ub
    value: value
        vcetor value
    dup_len: int
        vcetor data len

    Returns
    -------
    None
    """
    repeat = dup_len // 64
    repeat_tail = dup_len % 64
    offset = 0
    while repeat > 255:
        tik_instance.vector_dup(64, _ub[offset], value, 255, 1, 8)
        repeat = repeat - 255
        offset = offset + 64 * 255
    if repeat > 0:
        tik_instance.vector_dup(64, _ub[offset], value, repeat, 1, 8)
        offset = offset + 64 * repeat
    if repeat_tail > 0:
        tik_instance.vector_dup(repeat_tail, _ub[offset], value, 1, 1, 8)


def clip_window_compute(tik_instance, input_gm_list, input_ub_list, gm_offset, copy_num, data_each_block=16):
    """clip_window_compute"""
    input_boxes_gm = input_gm_list[0]
    input_sorces_gm = input_gm_list[1]
    input_boxes_ub = input_ub_list[0]
    input_scorces_ub = input_ub_list[1]
    input_index_gm = input_gm_list[-1]
    input_index_ub = input_ub_list[-1]

    # move data to boxes
    batch_box_shape = [copy_num, input_boxes_gm.shape[2]]
    box_total_size = total_num(batch_box_shape)

    index_shape = [copy_num, input_index_gm.shape[3]]
    index_total_size = total_num(index_shape)

    tmp_batch_box_ub = tik_instance.Tensor("float16", batch_box_shape, name="tmp_batch_box_ub", scope=tik.scope_ubuf)
    tmp_index_ub = tik_instance.Tensor("float16", index_shape, name="tmp_index_ub", scope=tik.scope_ubuf)

    real_box_offset = gm_offset[0] * input_boxes_gm.shape[1] * input_boxes_gm.shape[2] + gm_offset[3] * 4
    tik_func_data_move(tik_instance, tmp_batch_box_ub, input_boxes_gm, box_total_size, real_box_offset)

    real_index_offset = (gm_offset[0] * input_index_gm.shape[1] + gm_offset[1]) * input_index_gm.shape[2] * \
        input_index_gm.shape[3] + gm_offset[3] * 4

    tik_func_data_move(tik_instance, tmp_index_ub, input_index_gm, index_total_size, real_index_offset)

    real_score_offset = (gm_offset[0] * input_sorces_gm.shape[1] + gm_offset[1]) * \
        input_sorces_gm.shape[2] + gm_offset[3]
    tik_func_data_move(tik_instance, input_scorces_ub, input_sorces_gm, copy_num, real_score_offset)

    # for index id
    with tik_instance.for_range(0, 4) as i:
        with tik_instance.for_range(0, copy_num) as j:
            input_index_ub[i, j].set_as(tmp_index_ub[j, i])
    # for boxes
    with tik_instance.for_range(0, 4) as i:
        with tik_instance.for_range(0, copy_num) as j:
            input_boxes_ub[i, j].set_as(tmp_batch_box_ub[j, i])


def tik_func_data_move(tik_instance, input_x_ub, input_x_gm, move_num, offset):
    """move data from gm to ub"""
    ub_tensor_size = 4000
    loop_time = move_num // ub_tensor_size
    data_each_block = 16

    with tik_instance.for_range(0, loop_time) as loop_index:
        move_offset = loop_index * ub_tensor_size
        gm_offset = loop_index * ub_tensor_size + offset
        burse_len = ub_tensor_size // data_each_block
        tik_instance.data_move(input_x_ub[move_offset], input_x_gm[gm_offset], 0, 1, burse_len, 0, 0)

    last_num = move_num % ub_tensor_size
    if last_num > 0:
        tail_offset = loop_time * ub_tensor_size
        gm_tail_offset = loop_time * ub_tensor_size + offset
        burse_len = math.ceil(last_num / data_each_block)
        tik_instance.data_move(input_x_ub[tail_offset], input_x_gm[gm_tail_offset], 0, 1, burse_len, 0, 0)


def get_sorted_proposal_compute(tik_instance, output_ub, output_index_ub, input_gm_list, gm_offset, copy_num,
                                sorted_num, center_box, rpn_enble=False):
    """
    get_sorted_proposal_compute
    main function do copy boxes/scores, clip_window, change_coordinate, trans_to_proposals and sort

    Parameters:
    ----------
    tik_instance : tik_instance.
        tik_instance
    output_ub : ub
        output ub, save the sorted proposal list
    input_gm_list : list
        input gm list
    gm_offset : list
        input gm offset for score
    copy_num: int
        the process boxes num one time for copy
    sorted_num: int
        the sort boxes num one time for sort
    clip_window_value_list: list
        the window scalar list
    l1_valid_mask: cbuf
        num valid mask
    reduce_flag: bool
        whether reduce all box to avoid iou/vaadd overflows
    rpn_enble: bool
        whether support rpn

    Returns
    -------
    None
    """

    with tik_instance.new_stmt_scope():
        # apply ub for boxes copy_gm_to_ub
        ub_tmp_boxes = tik_instance.Tensor("float16", [4, sorted_num], name="copy_ub_tmp_boxes", scope=tik.scope_ubuf)
        # apply ub for score copy_gm_to_ub
        ub_tmp_score = tik_instance.Tensor("float16", [1, sorted_num], name="copy_ub_tmp_score", scope=tik.scope_ubuf)
        # apply ub for score copy_gm_to_ub
        ub_tmp_index = tik_instance.Tensor("float16", [4, sorted_num], name="ub_tmp_index", scope=tik.scope_ubuf)

        # step 1- copy boxes to ub with copy_num
        # step 2- copy scores to ub with copy_num
        # step 3- clip boxes and update scores
        input_ub_list = [ub_tmp_boxes, ub_tmp_score, ub_tmp_index]
        with tik_instance.new_stmt_scope():
            clip_window_compute(tik_instance, input_gm_list, input_ub_list, gm_offset, copy_num)
        # step 5- change_coordinate_frame if len(clip_window_value_list) == 6. will do change_coordinate_frame
        with tik_instance.new_stmt_scope():
            change_coordinate_frame_compute(tik_instance, ub_tmp_boxes, copy_num)
        if not rpn_enble:
            # x2,y2 sub 1 for iou RPN_offset
            tik_func_vadds(tik_instance, ub_tmp_boxes[2, :], ub_tmp_boxes[2, :], -1.0, copy_num)
            tik_func_vadds(tik_instance, ub_tmp_boxes[3, :], ub_tmp_boxes[3, :], -1.0, copy_num)
        # step 6- trans to proposal
        boxes_list = [ub_tmp_boxes[0, 0], ub_tmp_boxes[1, 0], ub_tmp_boxes[2, 0], ub_tmp_boxes[3, 0]]
        index_list = [ub_tmp_index[0, 0], ub_tmp_index[1, 0], ub_tmp_index[2, 0], ub_tmp_index[3, 0]]

        # vecter_dup the tail score = 0
        if copy_num % 16 != 0:
            dup_mask = int("0" * 48 + "1" * (16 - (copy_num % 16)) + "0" * (copy_num % 16), 2)
            tik_instance.vector_dup([0, dup_mask], ub_tmp_score[(copy_num // 16) * 16], 0.0, 1, 1, 8)
        if center_box == 0:
            tik_func_trans_to_proposals(tik_instance, output_ub, boxes_list, ub_tmp_score, copy_num)
        else:
            tik_func_trans_to_proposals_center_box(tik_instance, output_ub, ub_tmp_boxes, ub_tmp_score, copy_num)

        tik_func_trans_index_proposals(tik_instance, output_index_ub, index_list, ub_tmp_score, copy_num)

    # step 5- sort within ub to output_ub with sorted_num
    sort_within_ub(tik_instance, output_ub, ceil_div(copy_num, 16) * 16)
    if ceil_div(copy_num, 16) * 16 != sorted_num:
        dup_len = (sorted_num - ceil_div(copy_num, 16) * 16)
        offset = ceil_div(copy_num, 16) * 16 * 8
        tik_func_vector(tik_instance, output_ub[offset:], 0.0, dup_len * 8)

    sort_within_ub(tik_instance, output_index_ub, ceil_div(copy_num, 16) * 16)
    if ceil_div(copy_num, 16) * 16 != sorted_num:
        dup_len = (sorted_num - ceil_div(copy_num, 16) * 16)
        offset = ceil_div(copy_num, 16) * 16 * 8
        tik_func_vector(tik_instance, output_index_ub[offset:], 0.0, dup_len * 8)


def ub_offset(input_ub):
    """
    get ub offset
    when ub.shape is 1D tensor offset = 0
    when ub.shape is not 1D tensor change offset = 1D
    ex:
        ub.shape = [2,2,2]
        ub1 = ub[1,:,:]
        ub_offset(ub1) = 2*2 = 4 for ub
    """
    ub_shape = input_ub.shape
    if len(ub_shape) in (0, 1):
        return 0

    return input_ub.offset


def tik_func_vadds(tik_instance, dst_ub, src_ub, value, do_len):
    """tik_func_vadds"""
    repeat = do_len // 128
    repeat_tail = do_len % 128
    offset = ub_offset(src_ub)
    while repeat > 255:
        tik_instance.vadds(128, dst_ub[offset], src_ub[offset], value, 255, 1, 1, 8, 8)
        repeat = repeat - 255
        offset = offset + 128 * 255
    if repeat > 0:
        tik_instance.vadds(128, dst_ub[offset], src_ub[offset], value, repeat, 1, 1, 8, 8)
        offset = offset + 128 * repeat
    if repeat_tail > 0:
        tik_instance.vadds(repeat_tail, dst_ub[offset], src_ub[offset], value, 1, 1, 1, 8, 8)


def tik_func_vmuls(tik_instance, dst_ub, src_ub, value, do_len):
    """tik_func_vmuls"""
    repeat = do_len // 128
    repeat_tail = do_len % 128
    offset = dst_ub.offset
    while repeat > 255:
        tik_instance.vmuls(128, dst_ub[offset], src_ub[offset], value, 255, 1, 1, 8, 8)
        repeat = repeat - 255
        offset = offset + 128 * 255
    if repeat > 0:
        tik_instance.vmuls(128, dst_ub[offset], src_ub[offset], value, repeat, 1, 1, 8, 8)
        offset = offset + 128 * repeat
    if repeat_tail > 0:
        tik_instance.vmuls(repeat_tail, dst_ub[offset], src_ub[offset], value, 1, 1, 1, 8, 8)


def change_coordinate_frame_compute(tik_instance, ub_tmp_boxes, do_num):
    """change_coordinate_frame_compute"""
    h_scale_scale = tik_instance.Scalar(dtype="float16", init_value=COORDINATE_SCALING)
    w_scale_scale = tik_instance.Scalar(dtype="float16", init_value=COORDINATE_SCALING)
    tik_func_vmuls(tik_instance, ub_tmp_boxes[0, :], ub_tmp_boxes[0, :], h_scale_scale, do_num)
    tik_func_vmuls(tik_instance, ub_tmp_boxes[1, :], ub_tmp_boxes[1, :], w_scale_scale, do_num)
    tik_func_vmuls(tik_instance, ub_tmp_boxes[2, :], ub_tmp_boxes[2, :], h_scale_scale, do_num)
    tik_func_vmuls(tik_instance, ub_tmp_boxes[3, :], ub_tmp_boxes[3, :], w_scale_scale, do_num)


def filter_score_compute(tik_instance, index_ub, score_ub, proposal_num, score_thresh):
    """
    filter_score_compute, is score is less score_thresh,
    change score = -1
    """
    with tik_instance.new_stmt_scope():
        index_scalar = tik_instance.Scalar(dtype="int32", init_value=-1)
        score_ub_mask = tik_instance.Tensor(score_ub.dtype, score_ub.shape, name="score_ub_mask", scope=tik.scope_ubuf)
        score_thresh_ub = tik_instance.Tensor(score_ub.dtype, score_ub.shape,
                                              name="score_thresh_ub", scope=tik.scope_ubuf)
        score_ub_mask_int32 = tik_instance.Tensor("int32", score_ub.shape,
                                                  name="score_ub_mask_int32", scope=tik.scope_ubuf)
        tik_func_vector(tik_instance, score_thresh_ub, score_thresh, proposal_num)
        tik_func_add_sub_mul(tik_instance, "vsub", score_ub_mask, score_ub,
                             score_thresh_ub, proposal_num, 1, 1, 1, 8, 8, 8)
        apply_lens = ceil_div(score_ub.shape[0], 16)
        tik_instance.vec_conv(16, "ceil", score_ub_mask_int32, score_ub_mask, apply_lens, 2, 1)

        with tik_instance.for_range(0, proposal_num) as i:
            with tik_instance.if_scope(score_ub_mask_int32[i] < 1):
                with tik_instance.if_scope(index_ub[i, 2] != -1):
                    index_ub[i, 0].set_as(index_scalar)
                    index_ub[i, 1].set_as(index_scalar)
                    index_ub[i, 2].set_as(index_scalar)


def do_nms_compute(tik_instance, nms_var_dict, thresh):
    """
    Compute output boxes after non-maximum suppression.

    Parameters:
    ----------
    tik_instance : tik_instance.
        tik_instance
    nms_var_dict: dict
        the input par for nms, keys as follows:
            topk_out_num: the num proposal to do nms
            output_num: total output nms proposals
            topk_out_ub: the sorted proposals ub
            selected_proposal_ub: the selected_proposal_ub, save selected proposal
            selected_area_ub: the selected_area_ub, save selected proposal area
            sup_vec_ub: sup_vec_ub
            handling_proposals_cnt: a uint16 scalar
            selected_proposals_cnt: a uint16 scalar, specifying the selected proposal num
            zero_scalar : a uint16 scalar, value = 0
    thresh: float
        iou thresh for nms
    """
    total_input_proposal_num = nms_var_dict.get("topk_out_num")
    total_output_proposal_num = nms_var_dict.get("output_num")
    ub_max_topk = nms_var_dict.get("topk_out_ub")
    ub_index_topk = nms_var_dict.get("topk_index_out_ub")

    ub_selected_proposals = nms_var_dict.get("selected_proposal_ub")
    ub_selected_index_proposals = nms_var_dict.get("selected_index_ub")
    ub_selected_area = nms_var_dict.get("selected_area_ub")
    ub_sup_vec = nms_var_dict.get("sup_vec_ub")
    handling_proposals_cnt = nms_var_dict.get("handling_proposals_cnt")

    selected_proposals_cnt = nms_var_dict.get("selected_proposals_cnt")
    zero_scalar = nms_var_dict.get("zero_scalar")
    # variables
    left_proposal_cnt = tik_instance.Scalar(dtype="uint16")
    left_proposal_cnt.set_as(total_input_proposal_num)
    # store the whole
    # change with burst
    temp_proposals_ub = tik_instance.Tensor("float16", [BURST_PROPOSAL_NUM, 8],
                                            name="temp_proposals_ub", scope=tik.scope_ubuf)
    temp_index_proposals_ub = tik_instance.Tensor("float16", [BURST_PROPOSAL_NUM, 8],
                                                  name="temp_index_proposals_ub", scope=tik.scope_ubuf)
    tik_instance.vector_dup(128, temp_proposals_ub[0], 0, 8, 1, 8)
    temp_area_ub = tik_instance.Tensor("float16", [BURST_PROPOSAL_NUM], name="temp_area_ub", scope=tik.scope_ubuf)
    temp_iou_ub = tik_instance.Tensor(
        "float16",
        [ceil_div(total_output_proposal_num, RPN_PROPOSAL_NUM) * RPN_PROPOSAL_NUM + 128, 16],
        name="temp_iou_ub", scope=tik.scope_ubuf)
    temp_join_ub = tik_instance.Tensor(
        "float16",
        [ceil_div(total_output_proposal_num, RPN_PROPOSAL_NUM) * RPN_PROPOSAL_NUM + 128, 16],
        name="temp_join_ub", scope=tik.scope_ubuf)
    temp_sup_matrix_ub = tik_instance.Tensor(
        "uint16",
        [ceil_div(total_output_proposal_num, RPN_PROPOSAL_NUM) * RPN_PROPOSAL_NUM + 128],
        name="temp_sup_matrix_ub", scope=tik.scope_ubuf)
    temp_sup_vec_ub = tik_instance.Tensor("uint16", [BURST_PROPOSAL_NUM], name="temp_sup_vec_ub", scope=tik.scope_ubuf)

    # main body
    nms_flag = tik_instance.Scalar(dtype="uint16")
    nms_flag.set_as(0)
    with tik_instance.for_range(0, ceil_div(total_input_proposal_num, BURST_PROPOSAL_NUM)) as burst_index:
        # update counter
        with tik_instance.if_scope(left_proposal_cnt < BURST_PROPOSAL_NUM):
            handling_proposals_cnt.set_as(left_proposal_cnt)
        with tik_instance.else_scope():
            handling_proposals_cnt.set_as(BURST_PROPOSAL_NUM)

        handling_ceil = tik_instance.Scalar(dtype="uint16")
        handling_ceil.set_as(ceil_div(handling_proposals_cnt, 16))
        selected_ceil = tik_instance.Scalar(dtype="uint16")
        selected_ceil.set_as(ceil_div(selected_proposals_cnt, 16))
        # clear temp_sup_vec_ub
        tik_instance.vector_dup(128, temp_sup_vec_ub[0], 1, temp_sup_vec_ub.shape[0] // BURST_PROPOSAL_NUM, 1, 8)
        start_index = burst_index * BURST_PROPOSAL_NUM * 8

        tik_instance.data_move(temp_proposals_ub, ub_max_topk[start_index], 0, 1, 1024 // 16, 0, 0)
        tik_instance.data_move(temp_index_proposals_ub, ub_index_topk[start_index], 0, 1, 1024 // 16, 0, 0)

        # calculate the area of reduced-proposal
        tik_instance.vrpac(temp_area_ub[0], temp_proposals_ub[0], handling_ceil)
        # start to update iou and or area from the first 16 proposal
        # and get suppression vector 16 by 16 proposal
        length = tik_instance.Scalar(dtype="uint16")
        length.set_as(selected_ceil * 16)
        with tik_instance.if_scope(selected_proposals_cnt < total_output_proposal_num):
            with tik_instance.new_stmt_scope():
                with tik_instance.for_range(0, handling_ceil) as i:
                    length.set_as(length + 16)
                    # calculate intersection of tempReducedProposals
                    # and selReducedProposals
                    tik_instance.viou(temp_iou_ub[0, 0], ub_selected_proposals[0],
                                      temp_proposals_ub[i * 16 * 8], selected_ceil)
                    # calculate intersection of tempReducedProposals and
                    # `tempReducedProposals(include itself)`
                    tik_instance.viou(temp_iou_ub[selected_ceil * 16, 0],
                                      temp_proposals_ub[0], temp_proposals_ub[i * 16 * 8], i + 1)
                    # calculate join of tempReducedProposals
                    # and selReducedProposals
                    tik_instance.vaadd(temp_join_ub[0, 0], ub_selected_area[0], temp_area_ub[i * 16],
                                       selected_ceil)
                    # calculate intersection of tempReducedProposals and
                    # `tempReducedProposals(include itself)`
                    tik_instance.vaadd(temp_join_ub[selected_ceil * 16, 0],
                                       temp_area_ub, temp_area_ub[i * 16], i + 1)
                    # calculate join*(thresh/(1+thresh))

                    tik_instance.vmuls(128, temp_join_ub[0, 0], temp_join_ub[0, 0], thresh,
                                       ceil_div(length, 8), 1, 1, 8, 8)
                    # compare and generate suppression matrix
                    tik_instance.vcmpv_gt(temp_sup_matrix_ub[0], temp_iou_ub[0, 0], temp_join_ub[0, 0],
                                          ceil_div(length, 8), 1, 1, 8, 8)
                    # generate suppression vector
                    # clear rpn_cor_ir
                    rpn_cor_ir = tik_instance.set_rpn_cor_ir(0)
                    # non-diagonal
                    rpn_cor_ir = tik_instance.rpn_cor(temp_sup_matrix_ub[0], ub_sup_vec[0], 1, 1, selected_ceil)
                    with tik_instance.if_scope(i > 0):
                        rpn_cor_ir = tik_instance.rpn_cor(temp_sup_matrix_ub[selected_ceil * 16],
                                                          temp_sup_vec_ub[0], 1, 1, i)
                    # diagonal
                    tik_instance.rpn_cor_diag(temp_sup_vec_ub[i * 16], temp_sup_matrix_ub[length - 16], rpn_cor_ir)

                # find & mov unsuppressed proposals
                with tik_instance.for_range(0, handling_proposals_cnt) as i:
                    with tik_instance.if_scope(selected_proposals_cnt < total_output_proposal_num):
                        nms_flag.set_as(temp_sup_vec_ub[i])
                        with tik_instance.if_scope(nms_flag == 0):
                            ub_selected_proposals_uint64 = ub_selected_proposals.reinterpret_cast_to("uint64")
                            temp_proposals_ub_uint64 = temp_proposals_ub.reinterpret_cast_to("uint64")
                            ub_selected_index_proposals_uint64 = ub_selected_index_proposals.reinterpret_cast_to(
                                "uint64")
                            temp_index_proposals_ub_uint64 = temp_index_proposals_ub.reinterpret_cast_to("uint64")

                            ub_selected_proposals_uint64[selected_proposals_cnt * 2 + 0] = \
                                temp_proposals_ub_uint64[i * 2 + 0]
                            ub_selected_proposals_uint64[selected_proposals_cnt * 2 + 1] = \
                                temp_proposals_ub_uint64[i * 2 + 1]

                            ub_selected_index_proposals_uint64[selected_proposals_cnt * 2 + 0] = \
                                temp_index_proposals_ub_uint64[i * 2 + 0]
                            ub_selected_index_proposals_uint64[selected_proposals_cnt * 2 + 1] = \
                                temp_index_proposals_ub_uint64[i * 2 + 1]

                            ub_selected_area[selected_proposals_cnt] = temp_area_ub[i]
                            # update sup_vec_ub
                            ub_sup_vec[selected_proposals_cnt].set_as(zero_scalar)
                            # update counter
                            selected_proposals_cnt.set_as(selected_proposals_cnt + 1)

            left_proposal_cnt.set_as(left_proposal_cnt - handling_proposals_cnt)


def tik_func_sort_with_ub(tik_instance, src_ub_list, dst_ub_list, sorted_num, whether_save_proposal=None):
    """
    sort two sorted proposals list:
        get the top sorted_num proposals from src_ub_list
        and copy top sorted_num to output_ub
        and if need, copy low sorted_num to l1

    Parameters:
    ----------
    tik_instance : tik_instance.
        tik_instance
    src_ub_list : list
        the proposal list, each list have been sorted
    dst_ub_list : list
        result ub, copy top sorted_num to output_ub
    sorted_num : int
        the proposal num of proposal list
    whether_save_proposal: gm
        whether copy low sorted_num to l1

    Returns
    -------
    None
    """
    list_len = len(src_ub_list)
    with tik_instance.new_stmt_scope():
        # apply second top k proposal ub
        ub_dst_sort_with_ub = tik_instance.Tensor("float16", [list_len * sorted_num * 8],
                                                  name="ub_dst_sort_with_ub", scope=tik.scope_ubuf)
        sort_with_ub(tik_instance, src_ub_list, ub_dst_sort_with_ub, sorted_num)
        loop_burst_len = (sorted_num * 8) // 16
        tik_instance.data_move(dst_ub_list[0], ub_dst_sort_with_ub, 0, 1, loop_burst_len, 0, 0)
        if whether_save_proposal is not None:
            tik_instance.data_move(whether_save_proposal,
                                   ub_dst_sort_with_ub[sorted_num * 8:], 0, 1, loop_burst_len, 0, 0)


# 'pylint: disable=too-many-locals, too-many-branches
def nms_for_single_class(batch_idx, class_idx, nms, core_idx):
    """
    main func to get nms for each class,
    and copy result to l1 to concat
    """
    # get tik instance
    tik_instance = nms.get_tik_instance()
    global_cnt = tik_instance.Scalar(dtype="int32", init_value=0)

    # get first top_k proposals to ub_max_topk
    nms.init_tik_ub_mem_for_topk()
    topk_out_ub = nms.ub_max_topk
    index_out_ub = nms.ub_topk_index
    gm_offset = [batch_idx, class_idx, 0, 0]
    sorted_k = nms.proposal_topk_k
    # get first top 4096 high score boxes and do nms
    if nms.single_loop:
        get_sorted_proposal_compute(tik_instance, topk_out_ub, index_out_ub, nms.input_gm_list, gm_offset,
                                    nms.boxes_num, ceil_div(nms.boxes_num, 16) * 16, nms.center_point_box)
    else:
        get_sorted_proposal_compute(tik_instance, topk_out_ub, index_out_ub, nms.input_gm_list, gm_offset,
                                    sorted_k, sorted_k, nms.center_point_box)
        # do topk k proposal loop to get final top proposal_topk_k proposals to ub_max_topk
        with tik_instance.new_stmt_scope():
            # apply second top k proposal ub
            ub_tmp_topk = tik_instance.Tensor("float16", topk_out_ub.shape, name="ub_tmp_topk", scope=tik.scope_ubuf)
            ub_tmp_topk_index = tik_instance.Tensor("float16", index_out_ub.shape,
                                                    name="ub_tmp_topk_index", scope=tik.scope_ubuf)
            if nms.topk_loop_time > 1:
                with tik_instance.for_range(1, nms.topk_loop_time) as _top_k_idx:
                    gm_offset = [batch_idx, class_idx, 0, _top_k_idx * nms.proposal_topk_k]
                    if nms.is_second_nms:
                        workspace_offset = (_top_k_idx - 1) * sorted_k * 8 + core_idx * (nms.boxes_num * 8)
                        workspace_for_save_proposal = nms.workspace_second_nms_gm[workspace_offset]
                        workspace_for_save_index_proposal = nms.workspace_second_nms_index_gm[workspace_offset]
                    else:
                        workspace_for_save_proposal = None
                        workspace_for_save_index_proposal = None
                    # get tmp sorted proposal to ub_tmp_topk
                    get_sorted_proposal_compute(tik_instance, ub_tmp_topk, ub_tmp_topk_index, nms.input_gm_list,
                                                gm_offset, sorted_k, sorted_k, nms.center_point_box)
                    # sorted two proposals to one proposal list output the top sorted_k
                    tik_func_sort_with_ub(tik_instance, [ub_tmp_topk, topk_out_ub],
                                          [topk_out_ub, ub_tmp_topk], sorted_k,
                                          workspace_for_save_proposal)
                    tik_func_sort_with_ub(tik_instance, [ub_tmp_topk_index, index_out_ub],
                                          [index_out_ub, ub_tmp_topk_index], sorted_k,
                                          workspace_for_save_index_proposal)

            if nms.topk_loop_tail != 0:
                gm_offset = [batch_idx, class_idx, 0, nms.topk_loop_time * nms.proposal_topk_k]
                if nms.is_second_nms:
                    workspace_offset = (nms.topk_loop_time - 1) * sorted_k * 8 + core_idx * nms.boxes_num * 8
                    workspace_for_save_proposal = nms.workspace_second_nms_gm[workspace_offset]
                    workspace_for_save_index_proposal = nms.workspace_second_nms_index_gm[workspace_offset]
                else:
                    workspace_for_save_proposal = None
                    workspace_for_save_index_proposal = None
                # get tmp sorted proposal to ub_tmp_topk
                get_sorted_proposal_compute(tik_instance, ub_tmp_topk, ub_tmp_topk_index, nms.input_gm_list,
                                            gm_offset, nms.topk_loop_tail, sorted_k, nms.center_point_box)
                # sorted two proposals to one proposal list output the top sorted_k
                tik_func_sort_with_ub(tik_instance, [ub_tmp_topk, topk_out_ub],
                                      [topk_out_ub, ub_tmp_topk], sorted_k,
                                      workspace_for_save_proposal)
                tik_func_sort_with_ub(tik_instance, [ub_tmp_topk_index, index_out_ub],
                                      [index_out_ub, ub_tmp_topk_index], sorted_k,
                                      workspace_for_save_index_proposal)

    # do nms use topk output to get nms proposals per class
    # and move result to l1
    with tik_instance.new_stmt_scope():
        nms_var = nms.init_tik_ub_mem_for_nms()
        nmsed_result_ub = nms_var.get("selected_proposal_ub")
        nmsed_result_index_ub = nms_var.get("selected_index_ub")
        nmsed_result_area = nms_var.get("selected_area_ub")
        nmsed_result_sup = nms_var.get("sup_vec_ub")
        # init all sup_vec to 1, mean: no select proposal
        tik_func_vector(tik_instance, nmsed_result_sup, 1, nms.max_selected_nms_num_in_ub)
        # init select nms proposal = 0
        l1_buffer = nms.l1_nms_result_zero
        l1_offset = [class_idx, 0, 0]
        loop_burst_len = (nms.max_selected_nms_num_in_ub * 8) // 16
        tik_instance.data_move(nmsed_result_ub, l1_buffer, 0, 1, loop_burst_len, 0, 0)
        tik_instance.data_move(nmsed_result_index_ub, l1_buffer, 0, 1, loop_burst_len, 0, 0)
        tik_instance.data_move(nms.gm_nms_result[l1_offset], nmsed_result_index_ub, 0, 1, loop_burst_len, 0, 0)
        # init select nms area = 0
        loop_burst_len = nms.max_selected_nms_num_in_ub // 16
        tik_instance.data_move(nmsed_result_area, l1_buffer, 0, 1, loop_burst_len, 0, 0)

        global_cnt.set_as(nms.selected_proposals_cnt)
        with tik_instance.new_stmt_scope():
            do_nms_compute(tik_instance, nms_var, nms.get_iou_threshold())

        # copy one class nms result to l1
        burst_lens_gm = ceil_div(nms.selected_proposals_cnt * 8, 16)
        tik_instance.data_move(nms.gm_nms_result[l1_offset], nmsed_result_index_ub, 0, 1, burst_lens_gm, 0, 0)
        if nms.is_second_nms:
            loop_burst_len = nms.max_selected_nms_num_in_ub // 16
            tik_instance.data_move(nms.l1_nms_area, nmsed_result_area, 0, 1, loop_burst_len, 0, 0)
            tik_instance.data_move(nms.l1_nms_sup, nms_var.get("sup_vec_ub"), 0, 1, loop_burst_len, 0, 0)

    # If the select nms output num of the first top 4096 highest score boxes is less the output need
    # and the impl_mode is high_precision
    # Do nms again from the tail boxes 4096 boxes by 4096 boxes
    if nms.topk_loop_tail == 0:
        tool_loop = nms.topk_loop_time
    else:
        tool_loop = nms.topk_loop_time + 1
    if nms.is_second_nms and tool_loop >= 3:
        # if not to output num
        with tik_instance.for_range(1, tool_loop) as _top_n_idx:
            top_n_num_tail = tool_loop - _top_n_idx - 1
            with tik_instance.if_scope(nms.selected_proposals_cnt < nms.max_total_size):
                # copy a sorted proposals to topk_out_ub
                loop_burst_len = ceil_div(sorted_k * 8, 16)
                tik_instance.data_move(topk_out_ub, nms.workspace_second_nms_gm[core_idx * nms.boxes_num * 8],
                                       0, 1, loop_burst_len, 0, 0)
                tik_instance.data_move(index_out_ub, nms.workspace_second_nms_index_gm[core_idx * nms.boxes_num * 8],
                                       0, 1, loop_burst_len, 0, 0)
                # apply second top k proposal ub
                with tik_instance.new_stmt_scope():
                    ub_tmp_topk = tik_instance.Tensor("float16", topk_out_ub.shape,
                                                      name="ub_tmp_topk", scope=tik.scope_ubuf)
                    ub_tmp_topk_index = tik_instance.Tensor("float16", index_out_ub.shape,
                                                            name="ub_tmp_topk_index", scope=tik.scope_ubuf)
                    with tik_instance.for_range(0, top_n_num_tail) as _top_n_tail_idx:
                        workspace_proposal_offset = sorted_k * 8 + _top_n_tail_idx * sorted_k * 8 + \
                            core_idx * nms.boxes_num * 8
                        tik_instance.data_move(ub_tmp_topk, nms.workspace_second_nms_gm[workspace_proposal_offset],
                                               0, 1, loop_burst_len, 0, 0)
                        tik_instance.data_move(ub_tmp_topk_index,
                                               nms.workspace_second_nms_index_gm[workspace_proposal_offset],
                                               0, 1, loop_burst_len, 0, 0)
                        workspace_offset = _top_n_tail_idx * sorted_k * 8 + core_idx * nms.boxes_num * 8
                        workspace_for_save_proposal = nms.workspace_second_nms_gm[workspace_offset]
                        workspace_for_save_index_proposal = nms.workspace_second_nms_index_gm[workspace_offset]
                        # sorted two proposals to one proposal list output the top sorted_k
                        tik_func_sort_with_ub(tik_instance, [ub_tmp_topk, topk_out_ub],
                                              [topk_out_ub, ub_tmp_topk], sorted_k,
                                              workspace_for_save_proposal)
                        tik_func_sort_with_ub(tik_instance, [ub_tmp_topk_index, index_out_ub],
                                              [index_out_ub, ub_tmp_topk_index], sorted_k,
                                              workspace_for_save_index_proposal)

                # do nms use topk output to get nms proposals per class
                # and move result to gm
                with tik_instance.new_stmt_scope():
                    nms_var = nms.init_tik_ub_mem_for_nms()
                    nmsed_result_ub = nms_var.get("selected_proposal_ub")
                    nmsed_result_index_ub = nms_var.get("selected_index_ub")
                    nmsed_result_area = nms_var.get("selected_area_ub")
                    nmsed_result_sup = nms_var.get("sup_vec_ub")
                    # init all sup_vec to 1, mean: no select proposal
                    tik_func_vector(tik_instance, nmsed_result_sup, 1, nms.max_selected_nms_num_in_ub)
                    # init select nms proposal = 0
                    l1_buffer = nms.l1_nms_result_zero
                    loop_burst_len = (nms.max_selected_nms_num_in_ub * 8) // 16
                    tik_instance.data_move(nmsed_result_ub, l1_buffer, 0, 1, loop_burst_len, 0, 0)
                    tik_instance.data_move(nmsed_result_index_ub, l1_buffer, 0, 1, loop_burst_len, 0, 0)
                    # init select nms area = 0
                    loop_burst_len = nms.max_selected_nms_num_in_ub // 16
                    tik_instance.data_move(nmsed_result_area, l1_buffer, 0, 1, loop_burst_len, 0, 0)
                    global_cnt.set_as(nms.selected_proposals_cnt)

                    with tik_instance.new_stmt_scope():
                        do_nms_compute(tik_instance, nms_var, nms.get_iou_threshold())

                    # copy one class nms result to l1
                    burst_lens_gm = ((nms.selected_proposals_cnt - global_cnt) * 8) // 16
                    class_offset = nms.max_selected_nms_num_in_ub * 8 * class_idx
                    with tik_instance.if_scope(burst_lens_gm > 0):
                        with tik_instance.if_scope(global_cnt % 2 == 1):
                            tik_instance.data_move(nms.gm_temp, nmsed_result_index_ub[(global_cnt - 1) * 8],
                                                   0, 1, 1, 0, 0)
                            ub_temp = tik_instance.Tensor("float16", (16,), name="ub_temp", scope=tik.scope_ubuf)
                            tik_instance.data_move(ub_temp, nms.gm_temp[8], 0, 1, 1, 0, 0)
                            tik_instance.data_move(nms.gm_nms_result[global_cnt * 8 + class_offset],
                                                   ub_temp, 0, 1, 1, 0, 0)
                            tik_instance.data_move(nms.gm_nms_result[(global_cnt + 1) * 8 + class_offset],
                                                   nmsed_result_index_ub[(global_cnt + 1) * 8],
                                                   0, 1, burst_lens_gm, 0, 0)
                        with tik_instance.else_scope():
                            tik_instance.data_move(nms.gm_nms_result[global_cnt * 8 + class_offset],
                                                   nmsed_result_index_ub[global_cnt * 8],
                                                   0, 1, burst_lens_gm, 0, 0)
                    loop_burst_len = nms.max_selected_nms_num_in_ub // 16
                    tik_instance.data_move(nms.l1_nms_area, nmsed_result_area, 0, 1, loop_burst_len, 0, 0)
                    tik_instance.data_move(nms.l1_nms_sup, nmsed_result_sup, 0, 1, loop_burst_len, 0, 0)


def index_revert(tik_instance, tensor_a, tensor_b, tensor_thousands, lens):
    """
    caculate real index
    res = a * 1000 + b
    """
    with tik_instance.new_stmt_scope():
        input_temp = tik_instance.Tensor("int32", (lens,), name="input_temp", scope=tik.scope_ubuf)
        tik_func_add_sub_mul_int32(tik_instance, "vmul", input_temp, tensor_a, tensor_thousands, lens, 1, 1, 1, 8, 8, 8)
        tik_func_add_sub_mul_int32(tik_instance, "vadd", tensor_a, input_temp, tensor_b, lens, 1, 1, 1, 8, 8, 8)


def tik_func_index_recover(tik_instance, ub_result_boxes, ub_index_temp_int32, proposal_num):
    """recover index from four index id"""
    index_id_head = tik_instance.Tensor("float16", [proposal_num], name="index_id_head", scope=tik.scope_ubuf)
    index_id_tail = tik_instance.Tensor("float16", [proposal_num], name="index_id_tail", scope=tik.scope_ubuf)
    input_b_ub_thous_int32 = tik_instance.Tensor(
        "int32", [proposal_num], name="input_b_ub_thous_int32", scope=tik.scope_ubuf)
    input_a_ub_int32 = tik_instance.Tensor(
        "int32", [proposal_num], name="input_a_ub_int32", scope=tik.scope_ubuf)
    input_b_ub_int32 = tik_instance.Tensor(
        "int32", [proposal_num], name="input_b_ub_int32", scope=tik.scope_ubuf)
    tik_func_vector_int32(tik_instance, input_b_ub_thous_int32, 1000, proposal_num)
    with tik_instance.for_range(0, proposal_num) as i:
        index_id_head[i] = ub_result_boxes[i * 8 + 2]
        index_id_tail[i] = ub_result_boxes[i * 8 + 3]
    apply_lens = ceil_div(proposal_num, 16)
    tik_instance.vec_conv(16, "round", input_a_ub_int32, index_id_head, apply_lens, 2, 1)
    tik_instance.vec_conv(16, "round", input_b_ub_int32, index_id_tail, apply_lens, 2, 1)
    index_revert(tik_instance, input_a_ub_int32, input_b_ub_int32, input_b_ub_thous_int32, proposal_num)
    with tik_instance.for_range(0, proposal_num) as i:
        ub_index_temp_int32[i * 3 + 2] = input_a_ub_int32[i]


def tik_func_add_sub_mul_int32(tik_instance, v_func,
                               out_dst, src0, src1, copy_num,
                               dst_blk, src0_blk, src1_blk,
                               dst_rep, src0_rep, src1_rep):
    """tik add sub and mul"""
    repeat_time = copy_num // 64
    repeat_tail = copy_num % 64
    tik_fun = None

    if v_func == "vadd":
        tik_fun = tik_instance.vadd

    if v_func == "vsub":
        tik_fun = tik_instance.vsub

    if v_func == "vmul":
        tik_fun = tik_instance.vmul

    if repeat_time > 0:
        tik_fun(64, out_dst, src0[0], src1[0], repeat_time, dst_blk, src0_blk, src1_blk, dst_rep, src0_rep, src1_rep)

    if repeat_tail > 0:
        offset = repeat_time * 64
        tik_fun(repeat_tail, out_dst[offset], src0[offset], src1[offset],
                1, dst_blk, src0_blk, src1_blk, dst_rep, src0_rep, src1_rep)


def batch_multi_class_nms_copy_out(tik_instance, nms, ub_result_boxes,
                                   output_batch_offset, _class_idx):
    """batch_multi_class_nms_copy_out"""
    proposal_num = ub_result_boxes.shape[0]
    apply_men_len = ceil_div(nms.max_output_class * 3, 8)
    loop_len = ceil_div(proposal_num, 16)

    # ub index temp
    ub_index_temp = tik_instance.Tensor("float16", [proposal_num, 3], name="ub_index_temp", scope=tik.scope_ubuf)
    ub_index_temp_int32 = tik_instance.Tensor("int32", [proposal_num, 3], name="ub_index_temp_int32",
                                              scope=tik.scope_ubuf)
    with tik_instance.for_range(0, proposal_num) as i:
        with tik_instance.for_range(0, 3) as j:
            ub_index_temp[i * 3 + j] = ub_result_boxes[i * 8 + j]

    with tik_instance.for_range(0, loop_len) as i:
        offset_conv = i * 48
        tik_instance.vec_conv(48, "round", ub_index_temp_int32[offset_conv], ub_index_temp[offset_conv], 1, 6, 3)
    with tik_instance.new_stmt_scope():
        # recover index for four index to three index
        tik_func_index_recover(tik_instance, ub_result_boxes, ub_index_temp_int32, proposal_num)

    with tik_instance.new_stmt_scope():
        # process scores
        ub_out_scores = tik_instance.Tensor("float16", [proposal_num], name="ub_out_scores", scope=tik.scope_ubuf)
        with tik_instance.for_range(0, proposal_num) as i:
            ub_out_scores[i] = ub_result_boxes[i * 8 + 4]

        filter_score_compute(tik_instance, ub_index_temp_int32, ub_out_scores, proposal_num, nms.score_thresh)
    if nms.classes * nms.max_selected_nms_num_in_ub < 2000:
        with tik_instance.for_range(0, nms.classes) as i:
            tik_instance.data_move(nms.output_gm_list[0][output_batch_offset+nms.max_output_class*i*3],
                                   ub_index_temp_int32[nms.max_selected_nms_num_in_ub*i*3],
                                   0, 1, apply_men_len, 0, 0)
    else:
        tik_instance.data_move(nms.output_gm_list[0][output_batch_offset+nms.max_output_class*_class_idx*3],
                               ub_index_temp_int32,
                               0, 1, apply_men_len, 0, 0)


def batch_multi_class_nms_output(tik_instance, core_idx, _batch_idx, nms):
    """
    do batch_multi_class_nms_output

    Parameters:
    ----------
    tik_instance : tik_instance.
    _batch_idx : int.
        the process batch
    nms : class.
        all par for nms

    Returns
    -------
    None
    """
    result_total = total_num(nms.gm_nms_result.shape)
    # get score batch offset
    output_batch_offset = _batch_idx * nms.classes * nms.max_output_class * 3

    if nms.classes * nms.max_selected_nms_num_in_ub < 2000:
        # when all output is less nms.proposal_topk_k
        # only use topk with ub for output proposal
        with tik_instance.new_stmt_scope():
            ub_result_boxes = tik_instance.Tensor("float16", [result_total // 8, 8],
                                                  name="ub_result_boxes", scope=tik.scope_ubuf)
            l1_buffer = nms.gm_nms_result
            l1_offset = [0, 0, 0]
            loop_burst_len = result_total // 16
            tik_instance.data_move(ub_result_boxes, l1_buffer[l1_offset], 0, 1, loop_burst_len, 0, 0)

            with tik_instance.new_stmt_scope():
                batch_multi_class_nms_copy_out(tik_instance, nms, ub_result_boxes, output_batch_offset, 0)
    else:
        copy_loop = nms.classes
        with tik_instance.new_stmt_scope():
            ub_result_boxes = tik_instance.Tensor("float16", [nms.max_selected_nms_num_in_ub, 8],
                                                  name="ub_result_boxes", scope=tik.scope_ubuf)
            l1_buffer = nms.gm_nms_result
            with tik_instance.for_range(0, copy_loop) as _class_idx:
                l1_offset = _class_idx * nms.max_selected_nms_num_in_ub * 8
                loop_burst_len = nms.max_selected_nms_num_in_ub * 8 // 16
                tik_instance.data_move(ub_result_boxes, l1_buffer[l1_offset], 0, 1, loop_burst_len, 0, 0)
                with tik_instance.new_stmt_scope():
                    batch_multi_class_nms_copy_out(tik_instance, nms, ub_result_boxes, output_batch_offset, _class_idx)


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.OPTION_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def non_max_suppression_v7(boxes, scores, max_output_size,
                           iou_threshold, score_threshold, index_id,
                           selected_indices, center_point_box=0, max_boxes_size=0,
                           kernel_name="non_max_suppression_v7"):
    """
    do non_max_suppression for multi batch and multi class
    step 1- clip boxes use clip_window, when the area of boxes after clip, change the score = 0
    step 2- filter score, when the score is less score_threshold, change the score = 0
    step 3- filter valid num use num_valid_boxes
    step 4- trans the box and score to proposal
    step 5- sort the input proposals and get 4094 proposals
    step 6- do nms for each class in each batch use top 4094 proposals
    step 7- concat all class nms result in each batch
    step 8- sort the proposals and output the max_total_size box/class/score

    Parameters:
    ----------
    boxes : dict.
        shape, dtype of boxes
        An input tensor with shape [num_batches, spatial_dimension, 4].
        The single box data format is indicated by center_point_box.

    scores : dict.
        shape, dtype of scores
        An input tensor with shape [num_batches, num_classes, spatial_dimension]

    max_output_size : int.
        A option attribute of type int, specifying the nms output num per class.

    iou_threshold : float.
        A required attribute of type float32, specifying the nms iou iou_threshold

    score_threshold : float.
        A required attribute of type float32, specifying the score filter iou iou_threshold.

    index_id: dict.
        A input tensor with shape [num_batches,num_classes,spatial_dimension,3]
        the last dim representing (batch_id,class_id,index_id).

    center_point_box : int.
        Integer indicate the format of the box data.
        The default is 0. 0 - the box data is supplied as [y1, x1, y2, x2]
        where (y1, x1) and (y2, x2) are the coordinates of any diagonal pair of box
        corners and the coordinates can be provided as normalized
        (i.e., lying in the interval [0, 1]) or absolute.
        Mostly used for TF models. 1 - the box data is supplied
        as [x_center, y_center, width, height]. Mostly used for Pytorch models.

    max_boxes_size : int.
        An optional attribute integer representing the real maximum 
        number of boxes to be selected by non max suppression .

    selected_indices : dict.
        selected indices from the boxes tensor. [num_selected_indices, 3],
        the selected index format is [batch_index, class_index, box_index].

    kernel_name : str.
        cce kernel name, default value is "non_max_suppression"

    Returns
    -------
    tik_instance
    """
    if tbe_platform.api_check_support("tik.vcopy"):
        return non_max_suppression_v7_without_proposal(boxes, scores, max_output_size,
                                                       iou_threshold, score_threshold, index_id,
                                                       center_point_box, max_boxes_size, kernel_name)

    if tbe_platform.api_check_support("tik.scalar_conv"):
        return non_max_suppression_v7_supports_fp32(boxes, scores, max_output_size,
                                                    iou_threshold, score_threshold, index_id,
                                                    center_point_box, max_boxes_size, kernel_name)

    nms = NonMaxSuppression(boxes, scores, max_output_size,
                            iou_threshold, score_threshold, index_id,
                            center_point_box, max_boxes_size)

    # init ub
    core_used, batch_per_core, batch_last_core = nms.get_core_schedule()
    class_num = nms.classes
    nms.init_tik_output_mem()
    tik_instance = nms.get_tik_instance()

    def _run_one_core(_real_batch_idx, _real_core_idx):

        with tik_instance.for_range(0, class_num) as _class_idx:
            # for each class, init `selected_proposals_cnt = 0`
            nms.selected_proposals_cnt.set_as(0)
            with tik_instance.new_stmt_scope():
                nms_for_single_class(_real_batch_idx, _class_idx, nms, _real_core_idx)

        # process all class output result is in l1_nms_result, will process output
        # step 1 sort all select proposal with boxes
        # step 2 sort all select proposal with classes score
        with tik_instance.new_stmt_scope():
            batch_multi_class_nms_output(tik_instance, _real_core_idx, _real_batch_idx, nms)

    # do nms with multi cores
    with tik_instance.for_range(0, core_used, block_num=core_used) as _core_idx:
        if batch_per_core == batch_last_core or core_used == 1:
            with tik_instance.for_range(0, batch_per_core) as _batch_idx:
                real_batch_idx = _core_idx * batch_per_core + _batch_idx
                _run_one_core(real_batch_idx, _core_idx)
        else:
            with tik_instance.if_scope(_core_idx < core_used - 1):
                with tik_instance.for_range(0, batch_per_core) as _batch_idx:
                    real_batch_idx = _core_idx * batch_per_core + _batch_idx
                    _run_one_core(real_batch_idx, _core_idx)
            with tik_instance.else_scope():
                with tik_instance.for_range(0, batch_last_core) as _batch_idx:
                    real_batch_idx = _core_idx * batch_per_core + _batch_idx
                    _run_one_core(real_batch_idx, _core_idx)

    return nms.build_tik_instance(kernel_name)
