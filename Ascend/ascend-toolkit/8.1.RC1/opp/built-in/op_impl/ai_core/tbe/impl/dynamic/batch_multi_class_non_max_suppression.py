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
batch_multi_class_non_max_suppression
"""

from functools import reduce
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import OpImplMode
from impl.dynamic.batch_multi_class_nms_topk import sort_within_ub
from impl.dynamic.batch_multi_class_nms_topk import sort_within_ub_scalar
from impl.dynamic.batch_multi_class_nms_topk import sort_with_ub
from impl.dynamic.batch_multi_class_non_max_suppression_new import BMCNMS
from impl.util.platform_adapter import tbe_context
from impl.util.util_common import check_op_impl_mode
from tbe.common.platform import get_bit_len


# 'pylint: disable=too-many-lines,too-many-instance-attributes,too-many-statements
# 'pylint: disable=too-many-arguments,unused-argument,too-many-locals,too-many-branches
# 'pylint: disable=invalid-name
# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # scaling factor
    DOWN_FACTOR = 0.10
    # process 128 proposals at a time
    BURST_PROPOSAL_NUM = 128
    # RPN compute 16 proposals per iteration
    RPN_PROPOSAL_NUM = 16
    # define the positive min value in fp16
    MIN_SCALAR_FP16 = 2 ** (-24)
    # define a value 0f fp16
    TMP_SCALAR_FP16 = 2 ** 12
    MAX_INT32 = 2 ** 31 - 1
    MAX_ClASS = 20
    SCALAR_TENSOR_SIZE = 32
    TILING_ARG_NUM = 64
    PROPOSAL_TOP_K = 3968
    WORK_SIZE = 327808


class BatchMultiClassNonMaxSuppression():
    """
    Function: use to store BatchMultiClassNonMaxSuppression base parameters
    Modify : 2022-3-15
    """

    def __init__(self,
                 boxes,
                 scores,
                 num_valid_boxes,
                 clip_window,
                 score_thresh,
                 iou_thresh,
                 max_size_per_class,
                 max_total_size,
                 change_coordinate_frame,
                 impl_mode):
        """
        Init BatchMultiClassNonMaxSuppression base parameters

        Returns
        -------
        None
        """
        self.boxes_type = boxes.get("dtype")
        self.scores_type = scores.get("dtype")
        # when input have no class dim, will extend 1 for input shape
        if clip_window is None:
            self.need_clip_window = False
            self.clip_window_shape = None
        else:
            self.need_clip_window = True

        if num_valid_boxes is None:
            self.need_valid_num = False
            self.valid_num_shape = None
        else:
            self.need_valid_num = True

        self.tik_instance = tik.Tik()
        self.score_thresh = self.tik_instance.Scalar(dtype="float32")
        self.iou_thresh = self.tik_instance.Scalar(dtype="float32")
        self.max_size_per_class = self.tik_instance.Scalar(dtype="int32")
        self.max_selected_nms_num_in_ub = self.tik_instance.Scalar(dtype="int32")
        self.max_total_size = self.tik_instance.Scalar(dtype="int32")
        self.change_coordinate_frame = change_coordinate_frame

        # whether down the boxes to avoid fp16 overflow
        self.down_flag = True
        self.is_second_nms = False
        if impl_mode == OpImplMode.HIGH_PRECISION:
            self.is_second_nms = True

        self.aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.input_gm_list = []
        self.output_gm_list = []

        # for topk
        self.proposal_topk_k = self.ub_size // 4 // 16
        self.proposal_topk_k = min(self.proposal_topk_k, 255 * 16)

        self.topk_loop_time = self.tik_instance.Scalar(dtype="int32")
        self.topk_loop_tail = self.tik_instance.Scalar(dtype="int32")
        self.tiling_dtype = 'int32'
        self.is_need_rpn_offset = False

        # the scalar use for nms main cal
        self.ub_max_topk = None
        self.l1_nms_result = None
        self.l1_nms_result_zero = None
        self.workspace_proposal_gm = None
        self.workspace_second_nms_gm = None
        self.l1_score_valid = None
        self.l1_nms_area = None
        self.l1_nms_sup = None
        self.cal_mode = self.tik_instance.Scalar(dtype='int32')
        self.core_used = self.tik_instance.Scalar(dtype='int32')
        self.batch_per_core = self.tik_instance.Scalar(dtype='int32')
        self.batch_last_core = self.tik_instance.Scalar(dtype='int32')
        self.batch = self.tik_instance.Scalar(dtype='int32')
        self.classes = self.tik_instance.Scalar(dtype='int32')
        self.boxes_num = self.tik_instance.Scalar(dtype='int32')
        self.classes_box = self.tik_instance.Scalar(dtype='int32')

        # record the output nms num for one class
        self.selected_proposals_cnt = self.tik_instance.Scalar(dtype="uint16")
        # record the proposal burst num for one loop, value = 128 or
        # `self.proposal_topk_k % 128`
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
        # init 4 clip to windows scalar
        if self.need_clip_window:
            if self.change_coordinate_frame:
                self.down_flag = False
                self.clip_window_value_list = [self.tik_instance.Scalar(dtype="float16") for _ in range(6)]
            else:
                self.clip_window_value_list = [self.tik_instance.Scalar(dtype="float16") for _ in range(4)]
        else:
            self.clip_window_value_list = None
        # init 1 valid num scalar
        self.valid_num_value = self.tik_instance.Scalar(dtype="int32")
        self.dtype_bytes_size_tiling = get_bit_len(self.tiling_dtype) // 8
        block_bite_size = 32
        self.tiling_each_block = block_bite_size // self.dtype_bytes_size_tiling

        self.down_scalar_list = None
        # init down scalar
        if self.down_flag:
            self.down_scalar_list = [
                self.tik_instance.Scalar(
                    dtype="float16") for _ in range(2)]
            self.down_scalar_list[0].set_as(Constant.DOWN_FACTOR)
            self.down_scalar_list[1].set_as(1 / Constant.DOWN_FACTOR)
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype, (Constant.TILING_ARG_NUM,), name='tiling_gm',
                                                  scope=tik.scope_gm)

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
                                   flowtable=[self.tiling_gm],
                                   output_files_path=None,
                                   enable_l2=False)
        second_nms = 0
        if self.is_second_nms:
            second_nms = 1
        tbe_context.get_context().add_compile_info('vars',
                               {'aicore_num': self.aicore_num,
                                'proposal_topk_k': self.proposal_topk_k,
                                "dtype_byte": 2, "is_old": 1,
                                "is_second_nms": second_nms})

        return self.tik_instance

    def init_tik_mem(self, nms):
        """
        init tik gm mem
        """
        # init gm input
        max_boxes_num = 10000
        half_l1_ub_size = 512 * 1024
        boxes_gm = self.tik_instance.Tensor(
            "float16", (Constant.MAX_INT32,), name="boxes_gm", scope=tik.scope_gm)
        scores_gm = self.tik_instance.Tensor(
            "float16", (Constant.MAX_INT32,), name="scores_gm", scope=tik.scope_gm)

        clip_window_gm = None
        valid_num_gm = None
        if self.need_clip_window:
            clip_window_gm = self.tik_instance.Tensor(
                "float16", (Constant.MAX_INT32,), name="clip_window_gm", scope=tik.scope_gm)
        if self.need_valid_num:
            valid_num_gm = self.tik_instance.Tensor(
                "int32", (Constant.MAX_INT32,), name="valid_num_gm", scope=tik.scope_gm)
        if self.need_valid_num and self.need_clip_window:
            self.input_gm_list = [
                boxes_gm,
                scores_gm,
                clip_window_gm,
                valid_num_gm]
        elif self.need_clip_window:
            self.input_gm_list = [boxes_gm, scores_gm, clip_window_gm]
        elif self.need_valid_num:
            self.input_gm_list = [boxes_gm, scores_gm, valid_num_gm]
        else:
            self.input_gm_list = [boxes_gm, scores_gm]

        # init gm output
        nmsed_boxes_gm = self.tik_instance.Tensor(
            "float16", (Constant.MAX_INT32,), name="nmsed_boxes_gm", scope=tik.scope_gm)
        nmsed_scores_gm = self.tik_instance.Tensor(
            "float16", (Constant.MAX_INT32,), name="nmsed_scores_gm", scope=tik.scope_gm)
        nmsed_classes_gm = self.tik_instance.Tensor(
            "float16", (Constant.MAX_INT32,), name="nmsed_classes_gm", scope=tik.scope_gm)
        nmsed_num_gm = self.tik_instance.Tensor(
            "int32", (Constant.MAX_INT32,), name="nmsed_num_gm", scope=tik.scope_gm)
        self.output_gm_list = [
            nmsed_boxes_gm,
            nmsed_scores_gm,
            nmsed_classes_gm,
            nmsed_num_gm]

        # init l1 buff for save multi class nms result, size = [classes,
        # self.max_selected_nms_num_in_ub, 8]
        self.l1_nms_result = self.tik_instance.Tensor(
            "float16", (nms.classes, nms.max_selected_nms_num_in_ub, 8),
            name="l1_nms_result", scope=tik.scope_cbuf)

        if self.is_second_nms:
            # init l1 buff for save multi class nms area, size =
            # `[self.max_selected_nms_num_in_ub]`
            self.l1_nms_area = self.tik_instance.Tensor("float16", (nms.max_selected_nms_num_in_ub, ),
                name="l1_nms_area_tmp", scope=tik.scope_cbuf)
            # init l1 buff for save multi class nms sup, size =
            # `[self.max_selected_nms_num_in_ub]`
            self.l1_nms_sup = self.tik_instance.Tensor("uint16", (nms.max_selected_nms_num_in_ub, ),
                                                       name="l1_nms_sup_tmp", scope=tik.scope_cbuf)

        # zero data in l1
        self.l1_nms_result_zero = self.tik_instance.Tensor("float16", (nms.max_selected_nms_num_in_ub, 8),
                                                           name="l1_nms_result_zero", scope=tik.scope_cbuf)
        with self.tik_instance.new_stmt_scope():
            ub_nms_result = self.tik_instance.Tensor("float16", (nms.max_selected_nms_num_in_ub, 8),
                name="ub_nms_result", scope=tik.scope_ubuf)
            tik_func_vector_scalar(self.tik_instance, ub_nms_result, 0, nms.max_selected_nms_num_in_ub * 8)
            loop_burst_len = (nms.max_selected_nms_num_in_ub * 8) // 16
            self.tik_instance.data_move(self.l1_nms_result_zero, ub_nms_result, 0, 1, loop_burst_len, 0, 0)
        # workspace
        self.workspace_proposal_gm = self.tik_instance.Tensor("float16", [self.aicore_num,
                                                                          Constant.WORK_SIZE],
                                                              name="workspace_proposal_gm", scope=tik.scope_gm,
                                                              is_workspace=True)
        # workspace for second nms
        if self.is_second_nms:
            self.workspace_second_nms_gm = self.tik_instance.Tensor("float16", [self.aicore_num, max_boxes_num * 8],
                name="workspace_second_nms_gm", scope=tik.scope_gm, is_workspace=True)
        if self.need_valid_num:
            self.l1_score_valid = self.tik_instance.Tensor("float16", (ceil_div(nms.boxes_num, 16) * 16,),
                                                           name="l1_score_valid", scope=tik.scope_cbuf)

    def init_tik_ub_mem_for_nms(self):
        """
        init_tik_ub_mem_for_nms
        """
        ub_selected_proposals = self.tik_instance.Tensor(
            "float16", [self.max_selected_nms_num_in_ub, 8], name="ub_selected_proposals", scope=tik.scope_ubuf)
        ub_selected_area = self.tik_instance.Tensor(
            "float16", [self.max_selected_nms_num_in_ub], name="ub_selected_area", scope=tik.scope_ubuf)
        ub_sup_vec = self.tik_instance.Tensor(
            "uint16", [self.max_selected_nms_num_in_ub], name="ub_sup_vec", scope=tik.scope_ubuf)

        # when is_need_rpn_offset set rpn offset for vaadd and viou
        # else x2/y2 will do vadds -1 before nms and do vadds 1 after nms
        tiling_ub = self.tik_instance.Tensor(self.tiling_dtype, (Constant.TILING_ARG_NUM,),
                                             name='tiling_ub', scope=tik.scope_ubuf)
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, Constant.SCALAR_TENSOR_SIZE // \
        self.tiling_each_block, 0, 0)

        self.boxes_num.set_as(tiling_ub[6])

        if self.is_need_rpn_offset:
            self.tik_instance.set_rpn_offset(0.0)

        topk_out_num = self.tik_instance.Scalar(
            dtype='int32', name='topk_out_num')
        topk_out_num.set_as(self.proposal_topk_k)
        with self.tik_instance.if_scope(self.boxes_num < self.proposal_topk_k):
            topk_out_num.set_as(self.boxes_num)
        nms_var_dict = {
            # topk_out_info mean : nms input info
            "topk_out_ub": self.ub_max_topk,
            "topk_out_num": topk_out_num,
            # selected proposal info
            "selected_proposal_ub": ub_selected_proposals,
            "selected_area_ub": ub_selected_area,
            "sup_vec_ub": ub_sup_vec,
            # scalar reg info
            "zero_scalar": self.zero_scalar,
            "one_scalar": self.one_scalar,
            "selected_proposals_cnt": self.selected_proposals_cnt,
            "handling_proposals_cnt": self.handling_proposals_cnt,
            # nms output info
            "output_num": self.max_size_per_class
        }

        return nms_var_dict

    def init_tik_ub_mem_for_topk(self):
        """
        init_tik_ub_mem_for_topk
        """
        # init one ub for topk output
        self.ub_max_topk = self.tik_instance.Tensor(
            "float16", (self.proposal_topk_k, 8), name="ub_max_topk", scope=tik.scope_ubuf)

    def get_tiling_args(self):
        """
        get_tiling_args
        """
        with self.tik_instance.new_stmt_scope():
            tiling_ub = self.tik_instance.Tensor(
                self.tiling_dtype, (Constant.TILING_ARG_NUM,), name='tiling_ub', scope=tik.scope_ubuf)
            self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, Constant.SCALAR_TENSOR_SIZE // \
                self.tiling_each_block, 0, 0)

            self.cal_mode.set_as(tiling_ub[0])
            self.core_used.set_as(tiling_ub[1])
            self.batch_per_core.set_as(tiling_ub[2])
            self.batch_last_core.set_as(tiling_ub[3])
            self.batch.set_as(tiling_ub[4])
            self.classes.set_as(tiling_ub[5])
            self.boxes_num.set_as(tiling_ub[6])
            self.topk_loop_time.set_as(tiling_ub[7])
            self.topk_loop_tail.set_as(tiling_ub[8])
            self.classes_box.set_as(tiling_ub[9])
            self.score_thresh.set_as(tiling_ub[10])
            self.iou_thresh.set_as(tiling_ub[11])
            self.iou_thresh.set_as(self.iou_thresh / (1 + self.iou_thresh))
            self.max_size_per_class.set_as(tiling_ub[12])
            self.max_selected_nms_num_in_ub.set_as(
                ((self.max_size_per_class - 1) // Constant.RPN_PROPOSAL_NUM + 1) * Constant.RPN_PROPOSAL_NUM)
            self.max_total_size.set_as(tiling_ub[13])


def total_num(shape):
    """
    total_num
    """
    shape_total_num = reduce(lambda a, b: a * b, shape)
    return shape_total_num


def read_valid_num_compute(tik_instance, input_window_gm, offset, scalar):
    """
    read_valid_num_compute
    """
    with tik_instance.new_stmt_scope():
        input_window_ub = tik_instance.Tensor(
            input_window_gm.dtype,
            (8,),
            name="input_window_ub",
            scope=tik.scope_ubuf)
        tik_instance.data_move(input_window_ub,
                               input_window_gm[offset],
                               0, 1, 1, 0, 0)
        scalar.set_as(input_window_ub[0])


def gen_valid_num_compute(tik_instance, l1_output, input_len, scalar):
    """
    gen_valid_num_compute
    """
    with tik_instance.new_stmt_scope():
        input_window_ub = tik_instance.Tensor(
            l1_output.dtype,
            (ceil_div(input_len, 16) * 16,),
            name="input_window_ub",
            scope=tik.scope_ubuf)
        tik_func_vector_scalar(tik_instance, input_window_ub, 0.0, ceil_div(input_len, 16) * 16)
        with tik_instance.if_scope(scalar // 128 > 0):
            tik_instance.vector_dup(128, input_window_ub, 1.0, scalar // 128, 1, 8)
        with tik_instance.if_scope(scalar % 128 > 0):
            tik_instance.vector_dup(scalar % 128, input_window_ub[(scalar // 128) * 128], 1.0, 1, 1, 8)
        tik_instance.data_move(l1_output, input_window_ub, 0, 1, ceil_div(input_len, 16), 0, 0)


def valid_num_compute(tik_instance, l1_valid_mask, ub_tmp_score, copy_num):
    """
    valid_num_compute
    """
    if l1_valid_mask is None:
        return
    input_window_ub = tik_instance.Tensor(l1_valid_mask.dtype, l1_valid_mask.shape, name="input_window_ub",
                                          scope=tik.scope_ubuf)
    tik_instance.data_move(input_window_ub, l1_valid_mask,
                           0, 1, l1_valid_mask.shape[0] // 16, 0, 0)
    valid_num_compute_scalar(tik_instance, "vmul", ub_tmp_score, input_window_ub, ub_tmp_score,
                     ((copy_num - 1) // 16 + 1) * 16)


def valid_num_compute_scalar(
        tik_instance,
        l1_valid_mask,
        ub_tmp_score,
        copy_num):
    """
    valid_num_compute_scalar
    """
    if l1_valid_mask is None:
        return
    input_window_ub = tik_instance.Tensor(
        l1_valid_mask.dtype,
        l1_valid_mask.shape,
        name="input_window_ub",
        scope=tik.scope_ubuf)
    tik_instance.data_move(input_window_ub,
                           l1_valid_mask,
                           0, 1, l1_valid_mask.shape[0] // 16, 0, 0)
    tik_func_vcomple_scalar(
        tik_instance,
        "vmul",
        ub_tmp_score,
        input_window_ub,
        ub_tmp_score,
        ((copy_num - 1) // 16 + 1) * 16)


def change_coordinate_frame_compute(
        tik_instance,
        clip_window_value_list,
        ub_tmp_boxes,
        do_num):
    """
    change_coordinate_frame_compute
    """
    if clip_window_value_list is None:
        # no need to do clip_window and change_coordinate_frame
        return
    is_need_change_coordinate_frame = False
    if len(clip_window_value_list) == 6:
        is_need_change_coordinate_frame = True

    if is_need_change_coordinate_frame:
        h_scale_scale = clip_window_value_list[4]
        w_scale_scale = clip_window_value_list[5]
        tik_func_vmuls(
            tik_instance, ub_tmp_boxes[0, :], ub_tmp_boxes[0, :], h_scale_scale, do_num)
        tik_func_vmuls(
            tik_instance, ub_tmp_boxes[1, :], ub_tmp_boxes[1, :], w_scale_scale, do_num)
        tik_func_vmuls(
            tik_instance, ub_tmp_boxes[2, :], ub_tmp_boxes[2, :], h_scale_scale, do_num)
        tik_func_vmuls(
            tik_instance, ub_tmp_boxes[3, :], ub_tmp_boxes[3, :], w_scale_scale, do_num)


def change_coordinate_frame_compute_scalar(
        tik_instance,
        clip_window_value_list,
        ub_tmp_boxes,
        do_num):
    """
    change_coordinate_frame_compute_scalar
    """
    if clip_window_value_list is None:
        # no need to do clip_window and change_coordinate_frame
        return
    is_need_change_coordinate_frame = False
    if len(clip_window_value_list) == 6:
        is_need_change_coordinate_frame = True

    if is_need_change_coordinate_frame:
        h_scale_scale = clip_window_value_list[4]
        w_scale_scale = clip_window_value_list[5]
        tik_func_vmuls_scalar(
            tik_instance, ub_tmp_boxes[0, :], ub_tmp_boxes[0, :], h_scale_scale, do_num)
        tik_func_vmuls_scalar(
            tik_instance, ub_tmp_boxes[1, :], ub_tmp_boxes[1, :], w_scale_scale, do_num)
        tik_func_vmuls_scalar(
            tik_instance, ub_tmp_boxes[2, :], ub_tmp_boxes[2, :], h_scale_scale, do_num)
        tik_func_vmuls_scalar(
            tik_instance, ub_tmp_boxes[3, :], ub_tmp_boxes[3, :], w_scale_scale, do_num)


def clip_window_compute(
        tik_instance,
        nms,
        input_gm_list,
        input_ub_list,
        gm_offset,
        scalar_window,
        copy_num,
        data_each_block=16):
    """
    clip_window_compute
    """
    nms.get_tiling_args()
    input_num_boxes = input_ub_list[0].shape[1]

    dtype = input_ub_list[0].dtype
    win_y_min = tik_instance.Tensor(
        dtype,
        (data_each_block,),
        name="win_y_min",
        scope=tik.scope_ubuf)

    win_x_min = tik_instance.Tensor(
        dtype,
        (data_each_block,),
        name="win_x_min",
        scope=tik.scope_ubuf)

    win_y_max = tik_instance.Tensor(
        dtype,
        (data_each_block,),
        name="win_y_man",
        scope=tik.scope_ubuf)

    win_x_max = tik_instance.Tensor(
        dtype,
        (data_each_block,),
        name="win_x_max",
        scope=tik.scope_ubuf)

    zero_tensor = tik_instance.Tensor(
        dtype,
        (data_each_block,),
        name="zero",
        scope=tik.scope_ubuf)
    # min float16 value

    min_fp16 = tik_instance.Tensor(
        dtype,
        (data_each_block,),
        name="min_fp16",
        scope=tik.scope_ubuf)

    concst24 = tik_instance.Tensor(
        dtype,
        (data_each_block,),
        name="concst24",
        scope=tik.scope_ubuf)

    sub1 = tik_instance.Tensor(
        dtype,
        (input_num_boxes,),
        name="sub1",
        scope=tik.scope_ubuf)

    sub2 = tik_instance.Tensor(
        dtype,
        (input_num_boxes,),
        name="sub2",
        scope=tik.scope_ubuf)

    nburst = 1

    input_boxes_gm = input_gm_list[0]
    input_sorces_gm = input_gm_list[1]
    input_boxes_ub = input_ub_list[0]
    input_scorces_ub = input_ub_list[1]

    burse_len = tik_instance.Scalar(dtype='int32', name='burst_len')
    burse_len.set_as((copy_num - 1) // data_each_block + 1)
    class_offset = tik_instance.Scalar(dtype='int32')
    for i in range(4):
        with tik_instance.if_scope(nms.classes_box == 1):
            class_offset.set_as(0)
        with tik_instance.else_scope():
            class_offset.set_as(gm_offset[1])
        tik_instance.data_move(input_boxes_ub[i, 0],
                               input_boxes_gm[gm_offset[0] * nms.classes_box * 4 *
                                              nms.boxes_num + class_offset * 4 * nms.boxes_num + i * nms.boxes_num +
                                              gm_offset[3]], 0, nburst, burse_len, 0, 0)

    tik_instance.data_move(input_scorces_ub,
                           input_sorces_gm[gm_offset[0] * nms.classes * nms.boxes_num + nms.boxes_num *
                                           gm_offset[1] + gm_offset[3]], 0, nburst, burse_len, 0, 0)
    if scalar_window is None:
        # no need to do clip to window, return directly
        return
    index_win_y_min = scalar_window[0]
    index_win_x_min = scalar_window[1]
    index_win_y_max = scalar_window[2]
    index_win_x_max = scalar_window[3]
    tik_instance.vector_dup(data_each_block, win_y_min, index_win_y_min, 1, 1, 1)
    tik_instance.vector_dup(data_each_block, win_x_min, index_win_x_min, 1, 1, 1)
    tik_instance.vector_dup(data_each_block, win_y_max, index_win_y_max, 1, 1, 1)
    tik_instance.vector_dup(data_each_block, win_x_max, index_win_x_max, 1, 1, 1)

    y_min_input = input_boxes_ub[0 * input_num_boxes:]
    x_min_input = input_boxes_ub[1 * input_num_boxes:]
    y_max_input = input_boxes_ub[2 * input_num_boxes:]
    x_max_input = input_boxes_ub[3 * input_num_boxes:]
    y_min_out = input_boxes_ub[0 * input_num_boxes:]
    x_min_out = input_boxes_ub[1 * input_num_boxes:]
    y_max_out = input_boxes_ub[2 * input_num_boxes:]
    x_max_out = input_boxes_ub[3 * input_num_boxes:]

    def tik_func_vmin_vmax(tik_instance, vmin_or_max, out_dst, src0, src1, copy_num,
                           dst_blk, src0_blk, src1_blk, dst_rep, src0_rep, src1_rep):
        """
        tik_func_vmin_vmax
        """
        repeat_time = tik_instance.Scalar(dtype='int32', name='repeat_time')
        repeat_time.set_as(copy_num // 128)
        repeat_tail = tik_instance.Scalar(dtype='int32', name='repeat_tail')
        repeat_tail.set_as(copy_num % 128)
        tik_fun = None
        if vmin_or_max == "vmin":
            tik_fun = tik_instance.vmin

        if vmin_or_max == "vmax":
            tik_fun = tik_instance.vmax

        if vmin_or_max == "vsub":
            tik_fun = tik_instance.vsub

        if vmin_or_max == "vmul":
            tik_fun = tik_instance.vmul

        with tik_instance.if_scope(repeat_time > 0):
            tik_fun(128, out_dst, src0[0], src1[0],
                    repeat_time,
                    dst_blk, src0_blk, src1_blk,
                    dst_rep, src0_rep, src1_rep)

        with tik_instance.if_scope(repeat_tail > 0):
            offset = tik_instance.Scalar(dtype='int32', name='offset')
            offset.set_as(repeat_time * 128)
            tik_fun(repeat_tail, out_dst[offset], src0[offset], src1[0], 1, dst_blk, src0_blk, src1_blk,
                    dst_rep, src0_rep, src1_rep)

    tik_func_vmin_vmax(tik_instance, "vmin", y_min_out, y_min_input, win_y_max,
                       copy_num, 1, 1, 0, 8, 8, 0)
    tik_func_vmin_vmax(tik_instance, "vmin", y_max_out, y_max_input, win_y_max,
                       copy_num, 1, 1, 0, 8, 8, 0)
    tik_func_vmin_vmax(tik_instance, "vmin", x_min_out, x_min_input, win_x_max,
                       copy_num, 1, 1, 0, 8, 8, 0)
    tik_func_vmin_vmax(tik_instance, "vmin", x_max_out, x_max_input, win_x_max,
                       copy_num, 1, 1, 0, 8, 8, 0)

    tik_func_vmin_vmax(tik_instance, "vmax", y_min_out, y_min_out, win_y_min,
                       copy_num, 1, 1, 0, 8, 8, 0)
    tik_func_vmin_vmax(tik_instance, "vmax", y_max_out, y_max_out, win_y_min,
                       copy_num, 1, 1, 0, 8, 8, 0)
    tik_func_vmin_vmax(tik_instance, "vmax", x_min_out, x_min_out, win_x_min,
                       copy_num, 1, 1, 0, 8, 8, 0)
    tik_func_vmin_vmax(tik_instance, "vmax", x_max_out, x_max_out, win_x_min,
                       copy_num, 1, 1, 0, 8, 8, 0)

    tik_func_vmin_vmax(tik_instance, "vsub", sub1, y_max_out, y_min_out,
                       copy_num, 1, 1, 1, 8, 8, 8)
    tik_func_vmin_vmax(tik_instance, "vsub", sub2, x_max_out, x_min_out,
                       copy_num, 1, 1, 1, 8, 8, 8)
    tik_func_vmin_vmax(tik_instance, "vmul", sub1, sub1, sub2,
                       copy_num, 1, 1, 1, 8, 8, 8)

    tik_func_vector(tik_instance, zero_tensor, 0.0, 16)
    tik_func_vector(tik_instance, min_fp16, Constant.MIN_SCALAR_FP16, 16)
    tik_func_vector(tik_instance, concst24, Constant.TMP_SCALAR_FP16, 16)

    valid_num_compute_scalar(tik_instance, "vmin", sub1, sub1, min_fp16,
                     copy_num, 1, 1, 0, 8, 8, 0)
    valid_num_compute_scalar(tik_instance, "vmax", sub1, sub1, zero_tensor,
                     copy_num, 1, 1, 0, 8, 8, 0)
    valid_num_compute_scalar(tik_instance, "vmul", sub1, sub1, concst24,
                     copy_num, 1, 1, 0, 8, 8, 0)
    valid_num_compute_scalar(tik_instance, "vmul", sub1, sub1, concst24,
                     copy_num, 1, 1, 0, 8, 8, 0)

    # modify score = 0 when area <= 0
    tik_func_vmin_vmax(tik_instance, "vmul", input_scorces_ub,
                       input_scorces_ub, sub1,
                       copy_num, 1, 1, 1, 8, 8, 8)


def read_window_compute(
        tik_instance,
        input_window_gm,
        offset,
        scalar_list,
        down_scalar_list,
        change_coordinate_frame=False):
    """
    read_window_compute
    """
    with tik_instance.new_stmt_scope():
        input_window_ub = tik_instance.Tensor(input_window_gm.dtype, (16 * 2,), name="input_window_ub",
                                              scope=tik.scope_ubuf)
        tik_instance.data_move(input_window_ub, input_window_gm[offset],
                               0, 1, 1, 0, 0)

        [index_win_y_min, index_win_x_min, index_win_y_max, index_win_x_max] = [scalar_list[0],
                                                                                scalar_list[1], scalar_list[2],
                                                                                scalar_list[3]]
        index_win_y_min.set_as(input_window_ub[0])
        index_win_x_min.set_as(input_window_ub[1])
        index_win_y_max.set_as(input_window_ub[2])
        index_win_x_max.set_as(input_window_ub[3])

        if down_scalar_list is not None:
            input_window_ub_int32 = tik_instance.Tensor("int32", (16,),
                name="input_window_ub_int32", scope=tik.scope_ubuf)
            tik_instance.vconv(4, "round", input_window_ub_int32, input_window_ub, 1, 1, 1, 8, 4)
            max_h = tik_instance.Scalar(dtype="int32")
            max_w = tik_instance.Scalar(dtype="int32")
            max_h.set_as(input_window_ub_int32[2])
            max_w.set_as(input_window_ub_int32[3])
            with tik_instance.if_scope(max_h * max_w < 200 * 200):
                down_scalar_list[0].set_as(1.0)
                down_scalar_list[1].set_as(1.0)

        if change_coordinate_frame:
            [_, _, _, _, scale_h, scale_w] = scalar_list
            last_offset = offset + 2
            offset = last_offset
            tik_instance.data_move(input_window_ub[16], input_window_gm[offset], 0, 1, 1, 0, 0)
            valid_num_compute_scalar(tik_instance, "vsub", input_window_ub, input_window_ub[16], input_window_ub, 16)
            # do rec in mini
            tik_instance.vrec(2, input_window_ub[16], input_window_ub, 1, 1, 1, 8, 8)
            tik_fuc_vrec_newton(tik_instance, input_window_ub[16], input_window_ub, 2)
            scale_h.set_as(input_window_ub[16])
            scale_w.set_as(input_window_ub[17])


def ceil_div(value, factor):
    """
    Compute the smallest integer value that is greater than
    or equal to value/factor
    """
    result = (value + (factor - 1)) // factor
    return result


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


def tik_fuc_vrec_newton(
        tik_instance,
        vrec_ub,
        origin_ub,
        do_len,
        newton_iteration=2,
        block_num=16):
    """
    tik_fuc_vrec_newton
    """
    with tik_instance.new_stmt_scope():
        vrec_newton_1 = tik_instance.Tensor(
            vrec_ub.dtype, (((do_len + block_num - 1) // block_num) * block_num,),
            name="vrec_newton_1", scope=tik.scope_ubuf)
        vrec_newton_2 = tik_instance.Tensor(
            vrec_ub.dtype, (((do_len + block_num - 1) // block_num) * block_num,),
            name="vrec_newton_2", scope=tik.scope_ubuf)

        def _one_newton():
            """_one_newton"""
            tik_instance.vmul(2, vrec_newton_1, vrec_ub, origin_ub, 1, 1, 1, 1, 8, 8, 8)
            tik_instance.vmuls(2, vrec_newton_2, vrec_newton_1, -1, 1, 1, 1, 8, 8)
            tik_instance.vadds(2, vrec_newton_1, vrec_newton_2, 2, 1, 1, 1, 8, 8)
            tik_instance.vmul(2, vrec_ub, vrec_newton_1, vrec_ub, 1, 1, 1, 1, 8, 8, 8)

        for _ in range(newton_iteration):
            _one_newton()


def tik_func_vcomple(tik_instance, function, out_dst, src0, src1, copy_num,
                     dst_blk=1, src0_blk=1, src1_blk=1, dst_rep=8, src0_rep=8,
                     src1_rep=8):
    """
    tik_func_vcomple
    """
    do_dtype = out_dst.dtype
    if do_dtype in ("float16",):
        block_num = 16
    else:
        block_num = 8
    vector_num = block_num * 8
    repeat_time = copy_num // vector_num
    repeat_tail = copy_num % vector_num
    tik_fun = None
    ori_offset_dst = ub_offset(out_dst)
    ori_offset_src0 = ub_offset(src0)
    ori_offset_src1 = ub_offset(src1)
    if function == "vmin":
        tik_fun = tik_instance.vmin
    elif function == "vmax":
        tik_fun = tik_instance.vmax
    elif function == "vmul":
        tik_fun = tik_instance.vmul
    elif function == "vadd":
        tik_fun = tik_instance.vadd
    elif function == "vsub":
        tik_fun = tik_instance.vsub

    while repeat_time > 255:
        tik_fun(vector_num, out_dst[ori_offset_dst], src0[ori_offset_src0], src1[ori_offset_src1], 255,
                dst_blk, src0_blk, src1_blk, dst_rep, src0_rep, src1_rep)
        repeat_time = repeat_time - 255
        ori_offset_dst = ori_offset_dst + 255 * block_num * dst_rep
        ori_offset_src0 = ori_offset_src0 + 255 * block_num * src0_rep
        ori_offset_src1 = ori_offset_src1 + 255 * block_num * src1_rep

    if repeat_time > 0:
        tik_fun(vector_num, out_dst[ori_offset_dst], src0[ori_offset_src0], src1[ori_offset_src1],
                repeat_time, dst_blk, src0_blk, src1_blk, dst_rep, src0_rep, src1_rep)
        ori_offset_dst = ori_offset_dst + repeat_time * block_num * dst_rep
        ori_offset_src0 = ori_offset_src0 + repeat_time * block_num * src0_rep
        ori_offset_src1 = ori_offset_src1 + repeat_time * block_num * src1_rep

    if repeat_tail > 0:
        tik_fun(repeat_tail, out_dst[ori_offset_dst], src0[ori_offset_src0], src1[ori_offset_src1],
                1, dst_blk, src0_blk, src1_blk, dst_rep, src0_rep, src1_rep)


# 'pylint: disable=unused-variable
def tik_func_vcomple_scalar(
        tik_instance,
        function,
        out_dst,
        src0,
        src1,
        copy_num,
        dst_blk=1,
        src0_blk=1,
        src1_blk=1,
        dst_rep=8,
        src0_rep=8,
        src1_rep=8):
    """
    tik_func_vcomple_scalar
    """
    do_dtype = out_dst.dtype
    if do_dtype in ("float16",):
        block_num = 16
    else:
        block_num = 8
    vector_num = block_num * 8
    repeat_time = tik_instance.Scalar(dtype='int32', name='repeat_time')
    repeat_time.set_as(copy_num // vector_num)
    repeat_tail = tik_instance.Scalar(dtype='int32', name='repeat_tail')
    repeat_tail.set_as(copy_num % vector_num)
    tik_fun = None
    ori_offset_dst = tik_instance.Scalar(dtype="int64")
    ori_offset_src0 = tik_instance.Scalar(dtype="int64")
    ori_offset_src1 = tik_instance.Scalar(dtype="int64")
    ori_offset_dst.set_as(ub_offset(out_dst))
    ori_offset_src0.set_as(ub_offset(src0))
    ori_offset_src1.set_as(ub_offset(src1))
    if function == "vmin":
        tik_fun = tik_instance.vmin
    elif function == "vmax":
        tik_fun = tik_instance.vmax
    elif function == "vmul":
        tik_fun = tik_instance.vmul
    elif function == "vadd":
        tik_fun = tik_instance.vadd
    elif function == "vsub":
        tik_fun = tik_instance.vsub

    with tik_instance.if_scope(repeat_time // 255 > 0):
        with tik_instance.for_range(0, repeat_time // 255) as i:
            tik_fun(vector_num, out_dst[ori_offset_dst], src0[ori_offset_src0], src1[ori_offset_src1],
                    255, dst_blk, src0_blk, src1_blk, dst_rep, src0_rep, src1_rep)
            ori_offset_dst.set_as(ori_offset_dst + 255 * block_num * dst_rep)
            ori_offset_src0.set_as(ori_offset_src0 + 255 * block_num * src0_rep)
            ori_offset_src1.set_as(ori_offset_src1 + 255 * block_num * src1_rep)

    with tik_instance.if_scope(repeat_time > 0):
        tik_fun(vector_num, out_dst[ori_offset_dst], src0[ori_offset_src0], src1[ori_offset_src1],
                repeat_time, dst_blk, src0_blk, src1_blk, dst_rep, src0_rep, src1_rep)
        ori_offset_dst.set_as(ori_offset_dst + repeat_time * block_num * dst_rep)
        ori_offset_src0.set_as(ori_offset_src0 + repeat_time * block_num * src0_rep)
        ori_offset_src1.set_as(ori_offset_src1 + repeat_time * block_num * src1_rep)

    with tik_instance.if_scope(repeat_tail > 0):
        tik_fun(repeat_tail, out_dst[ori_offset_dst], src0[ori_offset_src0], src1[ori_offset_src1],
                1, dst_blk, src0_blk, src1_blk, dst_rep, src0_rep, src1_rep)


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
    ub_selected_proposals = nms_var_dict.get("selected_proposal_ub")
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
    temp_proposals_ub = tik_instance.Tensor(
        "float16", [Constant.BURST_PROPOSAL_NUM, 8], name="temp_proposals_ub", scope=tik.scope_ubuf)
    tik_instance.vector_dup(128, temp_proposals_ub[0], 0, 8, 1, 8)
    temp_area_ub = tik_instance.Tensor("float16", [Constant.BURST_PROPOSAL_NUM], name="temp_area_ub", \
    scope=tik.scope_ubuf)
    temp_iou_ub = tik_instance.Tensor("float16", [((total_output_proposal_num - 1) // Constant.RPN_PROPOSAL_NUM + 1) \
                                      * Constant.RPN_PROPOSAL_NUM + 128, 16], name="temp_iou_ub",
                                      scope=tik.scope_ubuf)
    temp_join_ub = tik_instance.Tensor("float16", [((total_output_proposal_num - 1) // Constant.RPN_PROPOSAL_NUM + 1)
                                                   * Constant.RPN_PROPOSAL_NUM + 128, 16],
                                       name="temp_join_ub", scope=tik.scope_ubuf)
    temp_sup_matrix_ub = tik_instance.Tensor("uint16", [((total_output_proposal_num - 1) // Constant.RPN_PROPOSAL_NUM
                                             + 1) * Constant.RPN_PROPOSAL_NUM + 128], name="temp_sup_matrix_ub",
                                             scope=tik.scope_ubuf)
    temp_sup_vec_ub = tik_instance.Tensor("uint16", [Constant.BURST_PROPOSAL_NUM], name="temp_sup_vec_ub",
                                          scope=tik.scope_ubuf)
    # main body
    nms_flag = tik_instance.Scalar(dtype="uint16")
    nms_flag.set_as(0)
    with tik_instance.for_range(0, ((total_input_proposal_num - 1) // Constant.BURST_PROPOSAL_NUM + 1)) as burst_index:
        # update counter
        with tik_instance.if_scope(left_proposal_cnt < Constant.BURST_PROPOSAL_NUM):
            handling_proposals_cnt.set_as(left_proposal_cnt)
        with tik_instance.else_scope():
            handling_proposals_cnt.set_as(Constant.BURST_PROPOSAL_NUM)

        handling_ceil = tik_instance.Scalar(dtype="uint16")
        handling_ceil.set_as(ceil_div(handling_proposals_cnt, 16))
        selected_ceil = tik_instance.Scalar(dtype="uint16")
        selected_ceil.set_as(ceil_div(selected_proposals_cnt, 16))
        # clear temp_sup_vec_ub
        tik_instance.vector_dup(128, temp_sup_vec_ub[0], 1, temp_sup_vec_ub.shape[0] // \
        Constant.BURST_PROPOSAL_NUM, 1, 8)
        temp_proposals_ub = ub_max_topk[burst_index * Constant.BURST_PROPOSAL_NUM * 8:]
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
                    tik_instance.vaadd(temp_join_ub[0, 0], ub_selected_area[0], temp_area_ub[i * 16], selected_ceil)
                    # calculate intersection of tempReducedProposals and
                    # `tempReducedProposals(include itself)`
                    tik_instance.vaadd(temp_join_ub[selected_ceil * 16, 0],
                                       temp_area_ub, temp_area_ub[i * 16], i + 1)
                    # calculate join*(thresh / (1+thresh))

                    iou_thresh_fp16 = tik_instance.Scalar(dtype='float16')
                    tik_instance.scalar_conv("", iou_thresh_fp16, thresh)
                    tik_instance.vmuls(128, temp_join_ub[0, 0], temp_join_ub[0, 0], iou_thresh_fp16,
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
                            ub_selected_proposals_uint64[selected_proposals_cnt *
                                                         2 + 0] = temp_proposals_ub_uint64[i * 2 + 0]
                            ub_selected_proposals_uint64[selected_proposals_cnt *
                                                         2 + 1] = temp_proposals_ub_uint64[i * 2 + 1]

                            ub_selected_area[selected_proposals_cnt] = temp_area_ub[i]
                            # update sup_vec_ub
                            ub_sup_vec[selected_proposals_cnt].set_as(zero_scalar)
                            # update counter
                            selected_proposals_cnt.set_as(selected_proposals_cnt + 1)
            left_proposal_cnt.set_as(left_proposal_cnt - handling_proposals_cnt)


def tik_func_vconcat(tik_instance, proposals_ub, _ub, trans_repeat, mode):
    """
    tik_func_vconcat
    """
    tik_instance.vconcat(proposals_ub, _ub, trans_repeat, mode)


def tik_func_vextract(tik_instance, proposals_ub, _ub, trans_repeat, mode):
    """
    tik_func_vextract
    """
    tik_instance.vextract(_ub, proposals_ub, trans_repeat, mode)


def tik_func_vadds(tik_instance, dst_ub, src_ub, value, do_len):
    """
    tik_func_vadds
    """
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


# 'pylint: disable=unused-variable
def tik_func_vadds_scalar(tik_instance, dst_ub, src_ub, value, do_len):
    """
    tik_func_vadds_scalar
    """
    repeat = tik_instance.Scalar(dtype='int32', name='repeat')
    repeat.set_as(do_len // 128)
    repeat_tail = tik_instance.Scalar(dtype='int32', name='repeat_tail')
    repeat_tail.set_as(do_len % 128)
    offset = tik_instance.Scalar(dtype='int32')
    offset.set_as(ub_offset(src_ub))
    with tik_instance.if_scope(repeat // 255 > 0):
        with tik_instance.for_range(0, repeat // 255) as i:
            tik_instance.vadds(128, dst_ub[offset], src_ub[offset], value,
                               255, 1, 1, 8, 8)
            offset.set_as(offset + 128 * 255)
    with tik_instance.if_scope(repeat > 0):
        tik_instance.vadds(128, dst_ub[offset], src_ub[offset], value,
                           repeat, 1, 1, 8, 8)
        offset.set_as(offset + 128 * repeat)
    with tik_instance.if_scope(repeat_tail > 0):
        tik_instance.vadds(repeat_tail, dst_ub[offset], src_ub[offset], value,
                           1, 1, 1, 8, 8)


def tik_func_vmuls(tik_instance, dst_ub, src_ub, value, do_len):
    """
    tik_func_vmuls
    """
    repeat = do_len // 128
    repeat_tail = do_len % 128
    offset = dst_ub.offset
    while repeat > 255:
        tik_instance.vmuls(128, dst_ub[offset], src_ub[offset], value,
                           255, 1, 1, 8, 8)
        repeat = repeat - 255
        offset = offset + 128 * 255
    if repeat > 0:
        tik_instance.vmuls(128, dst_ub[offset], src_ub[offset], value,
                           repeat, 1, 1, 8, 8)
        offset = offset + 128 * repeat
    if repeat_tail > 0:
        tik_instance.vmuls(repeat_tail, dst_ub[offset], src_ub[offset], value,
                           1, 1, 1, 8, 8)


def tik_func_vmuls_scalar(tik_instance, dst_ub, src_ub, value, do_len):
    """
    tik_func_vmuls_scalar
    """
    repeat = tik_instance.Scalar(dtype='int32', name='repeat')
    repeat.set_as(do_len // 128)
    repeat_tail = tik_instance.Scalar(dtype='int32', name='repeat_tail')
    repeat_tail.set_as(do_len % 128)
    offset = tik_instance.Scalar(dtype='int32')
    offset.set_as(dst_ub.offset)
    with tik_instance.if_scope(repeat // 255 > 0):
        with tik_instance.for_range(0, repeat // 255) as i:
            tik_instance.vmuls(128, dst_ub[offset], src_ub[offset], value,
                               255, 1, 1, 8, 8)
            offset.set_as(offset + 128 * 255)
    with tik_instance.if_scope(repeat > 0):
        tik_instance.vmuls(128, dst_ub[offset], src_ub[offset], value,
                           repeat, 1, 1, 8, 8)
        offset.set_as(offset + 128 * repeat)
    with tik_instance.if_scope(repeat_tail > 0):
        tik_instance.vmuls(repeat_tail, dst_ub[offset], src_ub[offset], value,
                           1, 1, 1, 8, 8)


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


def tik_func_vector_scalar(tik_instance, _ub, value, dup_len):
    """
    tik_func_vector_scalar
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
    repeat = tik_instance.Scalar(dtype='int32')
    repeat.set_as(dup_len // 128)
    repeat_tail = tik_instance.Scalar(dtype='int32')
    repeat_tail.set_as(dup_len % 128)
    offset = tik_instance.Scalar(dtype='int32', init_value=0)
    with tik_instance.if_scope(repeat // 255 > 0):
        with tik_instance.for_range(0, repeat // 255) as i:
            tik_instance.vector_dup(128, _ub[offset], value, 255, 1, 8)
            offset.set_as(offset + 128 * 255)
    with tik_instance.if_scope(repeat > 0):
        tik_instance.vector_dup(128, _ub[offset], value, repeat, 1, 8)
        offset.set_as(offset + 128 * repeat)
    with tik_instance.if_scope(repeat_tail > 0):
        tik_instance.vector_dup(repeat_tail, _ub[offset], value, 1, 1, 8)


def tik_func_trans_to_proposals(
        tik_instance,
        proposals_ub,
        boxes_ub_list,
        score_ub,
        proposal_num):
    """
    tik_func_trans_to_proposals
    """
    x1_ub, y1_ub, x2_ub, y2_ub = boxes_ub_list
    trans_repeat = ceil_div(proposal_num, 16)
    # concat x1 to proposals_ub
    tik_func_vconcat(tik_instance, proposals_ub, x1_ub, trans_repeat, 0)
    # concat y1 to proposals_ub
    tik_func_vconcat(tik_instance, proposals_ub, y1_ub, trans_repeat, 1)
    # concat x2 to proposals_ub
    tik_func_vconcat(tik_instance, proposals_ub, x2_ub, trans_repeat, 2)
    # concat y2 to proposals_ub
    tik_func_vconcat(tik_instance, proposals_ub, y2_ub, trans_repeat, 3)
    # concat scores to proposals_ub
    tik_func_vconcat(tik_instance, proposals_ub, score_ub, trans_repeat, 4)


def get_sorted_proposal_compute(
        tik_instance,
        nms,
        output_ub,
        input_gm_list,
        gm_offset,
        copy_num,
        sorted_num,
        clip_window_value_list,
        l1_valid_mask,
        reduce_scalar=None,
        rpn_enble=False):
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
        ub_tmp_boxes = tik_instance.Tensor("float16", [4, sorted_num],
                                           name="copy_ub_tmp_boxes", scope=tik.scope_ubuf)
        # apply ub for score copy_gm_to_ub
        ub_tmp_score = tik_instance.Tensor("float16", [1, sorted_num],
                                           name="copy_ub_tmp_score", scope=tik.scope_ubuf)

        # step 1- copy boxes to ub with copy_num
        # step 2- copy scores to ub with copy_num
        # step 3- clip boxes and update scores
        input_ub_list = [ub_tmp_boxes, ub_tmp_score]
        with tik_instance.new_stmt_scope():
            clip_window_compute(tik_instance, nms, input_gm_list, input_ub_list,
                gm_offset, clip_window_value_list, copy_num)
        # DOWN_FACTOR
        if reduce_scalar is not None:
            tik_func_vmuls(
                tik_instance, ub_tmp_boxes[0, :], ub_tmp_boxes[0, :], reduce_scalar[0], copy_num)
            tik_func_vmuls(
                tik_instance, ub_tmp_boxes[1, :], ub_tmp_boxes[1, :], reduce_scalar[0], copy_num)
            tik_func_vmuls(
                tik_instance, ub_tmp_boxes[2, :], ub_tmp_boxes[2, :], reduce_scalar[0], copy_num)
            tik_func_vmuls(
                tik_instance, ub_tmp_boxes[3, :], ub_tmp_boxes[3, :], reduce_scalar[0], copy_num)

        # step 4- filter valid num
        with tik_instance.new_stmt_scope():
            valid_num_compute(tik_instance, l1_valid_mask, ub_tmp_score, copy_num)

        # step 5- change_coordinate_frame if len(clip_window_value_list) == 6.
        # will do change_coordinate_frame
        with tik_instance.new_stmt_scope():
            change_coordinate_frame_compute(tik_instance, clip_window_value_list, ub_tmp_boxes, copy_num)

        if not rpn_enble:
            # x2  y2 sub 1 for iou RPN_offset
            tik_func_vadds(tik_instance, ub_tmp_boxes[2, :], ub_tmp_boxes[2, :], -1.0, copy_num)
            tik_func_vadds(tik_instance, ub_tmp_boxes[3, :], ub_tmp_boxes[3, :], -1.0, copy_num)

        # step 6- trans to proposal
        boxes_list = [ub_tmp_boxes[0, 0], ub_tmp_boxes[1, 0],
                      ub_tmp_boxes[2, 0], ub_tmp_boxes[3, 0]]

        # vecter_dup the tail score = 0
        if copy_num % 16 != 0:
            dup_mask = int("0" * 48 + "1" * (16 - (copy_num % 16)) + "0" * (copy_num % 16), 2)
            tik_instance.vector_dup([0, dup_mask], ub_tmp_score[(copy_num // 16) * 16], 0.0, 1, 1, 8)

        tik_func_trans_to_proposals(tik_instance, output_ub, boxes_list, ub_tmp_score, copy_num)

    # step 5- sort within ub to output_ub with sorted_num
    sort_within_ub(tik_instance, output_ub, ceil_div(copy_num, 16) * 16)
    if ceil_div(copy_num, 16) * 16 != sorted_num:
        dup_len = (sorted_num - ceil_div(copy_num, 16) * 16)
        offset = ceil_div(copy_num, 16) * 16 * 8
        tik_func_vector(tik_instance, output_ub[offset:], 0.0, dup_len * 8)


def get_sorted_proposal_compute_scalar(
        tik_instance,
        nms,
        output_ub,
        input_gm_list,
        gm_offset,
        copy_num,
        sorted_num,
        clip_window_value_list,
        l1_valid_mask,
        reduce_scalar=None,
        rpn_enble=False):
    """
    get_sorted_proposal_compute_scalar
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
        ub_tmp_boxes = tik_instance.Tensor(
            "float16", [4, sorted_num], name="ub_tmp_boxes", scope=tik.scope_ubuf)
        # apply ub for score copy_gm_to_ub
        ub_tmp_score = tik_instance.Tensor(
            "float16", [1, sorted_num], name="ub_tmp_score", scope=tik.scope_ubuf)

        # step 1- copy boxes to ub with copy_num
        # step 2- copy scores to ub with copy_num
        # step 3- clip boxes and update scores
        input_ub_list = [ub_tmp_boxes, ub_tmp_score]
        with tik_instance.new_stmt_scope():
            clip_window_compute(tik_instance, nms, input_gm_list, input_ub_list,
                                gm_offset, clip_window_value_list, copy_num)
        # DOWN_FACTOR
        if reduce_scalar is not None:
            tik_func_vmuls_scalar(
                tik_instance, ub_tmp_boxes[0, :], ub_tmp_boxes[0, :], reduce_scalar[0], copy_num)
            tik_func_vmuls_scalar(
                tik_instance, ub_tmp_boxes[1, :], ub_tmp_boxes[1, :], reduce_scalar[0], copy_num)
            tik_func_vmuls_scalar(
                tik_instance, ub_tmp_boxes[2, :], ub_tmp_boxes[2, :], reduce_scalar[0], copy_num)
            tik_func_vmuls_scalar(
                tik_instance, ub_tmp_boxes[3, :], ub_tmp_boxes[3, :], reduce_scalar[0], copy_num)

        # step 4- filter valid num
        with tik_instance.new_stmt_scope():
            valid_num_compute_scalar(tik_instance, l1_valid_mask, ub_tmp_score, copy_num)

        # step 5- change_coordinate_frame if len(clip_window_value_list) == 6.
        # will do change_coordinate_frame
        with tik_instance.new_stmt_scope():
            change_coordinate_frame_compute_scalar(tik_instance, clip_window_value_list, ub_tmp_boxes, copy_num)

        if not rpn_enble:
            # x2  y2 sub 1 for iou RPN_offset
            tik_func_vadds_scalar(tik_instance, ub_tmp_boxes[2, :], ub_tmp_boxes[2, :], -1.0, copy_num)
            tik_func_vadds_scalar(tik_instance, ub_tmp_boxes[3, :], ub_tmp_boxes[3, :], -1.0, copy_num)

        # step 6- trans to proposal
        boxes_list = [ub_tmp_boxes[0, 0], ub_tmp_boxes[1, 0],
                      ub_tmp_boxes[2, 0], ub_tmp_boxes[3, 0]]

        # vecter_dup the tail score = 0
        with tik_instance.if_scope(copy_num % 16 != 0):
            pre = tik_instance.Scalar(dtype='int32', init_value=0)
            dup_mask = tik_instance.Scalar(dtype='int32', init_value=1)
            with tik_instance.for_range(0, 16) as i:
                with tik_instance.if_scope(i >= (copy_num % 16)):
                    pre.set_as(pre + dup_mask)
                dup_mask.set_as(dup_mask * 2)
            zero_scalar = tik_instance.Scalar(dtype='int32', init_value=0)
            tik_instance.vector_dup(
                [zero_scalar, pre], ub_tmp_score[(copy_num // 16) * 16], 0.0, 1, 1, 8)

        tik_func_trans_to_proposals(tik_instance, output_ub, boxes_list, ub_tmp_score, copy_num)

    # step 5- sort within ub to output_ub with sorted_num
    level = tik_instance.Scalar(dtype='int32', init_value=1)
    region = tik_instance.Scalar(dtype='int32')
    region.set_as((((copy_num - 1) // 16 + 1) * 16 + 15) // 16)
    with tik_instance.if_scope(region > 1):
        with tik_instance.for_range(0, Constant.RPN_PROPOSAL_NUM) as i:
            with tik_instance.if_scope(region > 1):
                level.set_as(level + 1)
                region.set_as((region + 3) // 4)
    sort_within_ub_scalar(tik_instance, output_ub, ((copy_num - 1) // 16 + 1) * 16, level)
    with tik_instance.if_scope(((copy_num - 1) // 16 + 1) * 16 != sorted_num):
        dup_len = tik_instance.Scalar(dtype='int32')
        dup_len.set_as(sorted_num - ((copy_num - 1) // 16 + 1) * 16)
        offset = tik_instance.Scalar(dtype='int32')
        offset.set_as(((copy_num - 1) // 16 + 1) * 16 * 8)
        tik_func_vector_scalar(tik_instance, output_ub[offset:], 0.0, dup_len * 8)


def tik_func_sort_with_ub(
        tik_instance,
        src_ub_list,
        dst_ub_list,
        sorted_num,
        whether_save_proposal=None):
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
            tik_instance.data_move(whether_save_proposal, ub_dst_sort_with_ub[sorted_num * 8:],
                                   0, 1, loop_burst_len, 0, 0)


def filter_score_compute(
        tik_instance,
        score_ub,
        score_valid_num_ub,
        scores_valid_mask,
        score_num,
        score_thresh):
    """
    filter_score_compute, is score is less score_thresh, change score = 0
    """
    with tik_instance.new_stmt_scope():
        tmp_ub_for_vmax = tik_instance.Tensor(score_ub.dtype, [16], name="tmp_ub_for_vmax", scope=tik.scope_ubuf)
        tmp_ub_for_vmin = tik_instance.Tensor(score_ub.dtype, [16], name="tmp_ub_for_vmin", scope=tik.scope_ubuf)
        score_ub_mask = tik_instance.Tensor(score_ub.dtype, score_ub.shape,
                                            name="score_ub_mask", scope=tik.scope_ubuf)
        score_thresh_fp16 = tik_instance.Scalar(dtype='float16')
        score_thresh.set_as(-1 * score_thresh)
        tik_instance.scalar_conv("", score_thresh_fp16, score_thresh)
        tik_func_vadds_scalar(tik_instance, scores_valid_mask, score_ub, score_thresh_fp16, score_num)
        tik_func_vector(tik_instance, tmp_ub_for_vmax, 0.0, 16)
        tik_func_vector(tik_instance, tmp_ub_for_vmin, Constant.MIN_SCALAR_FP16, 16)
        tik_func_vcomple_scalar(tik_instance, "vmin", scores_valid_mask, scores_valid_mask, tmp_ub_for_vmin,
                                score_num, 1, 1, 0, 8, 8, 0)
        tik_func_vcomple_scalar(tik_instance, "vmax", scores_valid_mask, scores_valid_mask, tmp_ub_for_vmax, score_num,
                                1, 1, 0, 8, 8, 0)
        tik_func_vmuls_scalar(tik_instance, scores_valid_mask, scores_valid_mask, Constant.TMP_SCALAR_FP16, score_num)
        tik_func_vmuls_scalar(tik_instance, scores_valid_mask, scores_valid_mask, Constant.TMP_SCALAR_FP16, score_num)
        tik_func_vcomple_scalar(tik_instance, "vmul", score_ub, scores_valid_mask, score_ub, score_num, 1,
                                1, 1, 8, 8, 8)
        tik_func_vmuls_scalar(tik_instance, score_ub_mask, scores_valid_mask, 1, score_num)
        repeat_loop = score_num // 128
        repeat_tail = score_num % 128
        with tik_instance.if_scope(repeat_loop > 1):
            tik_func_vcomple_scalar(tik_instance, "vadd", score_ub_mask, score_ub_mask[128:],
                             score_ub_mask, (repeat_loop - 1) * 128, 1, 1, 1, 0, 8, 0)
        with tik_instance.if_scope(tik.all(repeat_tail != 1, repeat_loop > 0)):
            tik_func_vcomple_scalar(tik_instance, "vadd", score_ub_mask, score_ub_mask[128 * repeat_loop:],
                             score_ub_mask, repeat_tail, 1, 1, 1, 8, 8, 8)
        with tik_instance.if_scope(repeat_loop > 0):
            tik_instance.vcadd(128, score_ub_mask, score_ub_mask, 1, 1, 1, 8)
        with tik_instance.else_scope():
            tik_instance.vcadd(repeat_tail, score_ub_mask, score_ub_mask, 1, 1, 1, 8)

        tik_instance.vconv(8, "round", score_valid_num_ub, score_ub_mask, 1, 1, 1, 8, 4)


def nms_for_single_class(batch_idx, class_idx, nms, core_idx):
    """
    main func to get nms for each class,
    and copy result to l1 to concat
    """
    # get tik instance
    tik_instance = nms.get_tik_instance()
    # get first top_k proposals to ub_max_topk
    nms.init_tik_ub_mem_for_topk()
    nms.get_tiling_args()
    topk_out_ub = nms.ub_max_topk
    clip_window_value_list = nms.clip_window_value_list
    gm_offset = [batch_idx, class_idx, 0, 0]
    sorted_k = nms.proposal_topk_k

    # valid num info
    l1_valid = nms.l1_score_valid

    # get first top 4096 high score boxes and do nms
    with tik_instance.if_scope(nms.boxes_num <= nms.proposal_topk_k):
        get_sorted_proposal_compute_scalar(tik_instance, nms, topk_out_ub, nms.input_gm_list, gm_offset, nms.boxes_num,
                                           ((nms.boxes_num - 1) // 16 + 1) * 16,
                                           clip_window_value_list, l1_valid, reduce_scalar=nms.down_scalar_list)
    with tik_instance.else_scope():
        get_sorted_proposal_compute(tik_instance, nms, topk_out_ub, nms.input_gm_list, gm_offset, sorted_k, sorted_k,
                                    clip_window_value_list, l1_valid, reduce_scalar=nms.down_scalar_list)

        # do topk k proposal loop to get final top proposal_topk_k proposals to
        # ub_max_topk
        with tik_instance.new_stmt_scope():
            # apply second top k proposal ub
            ub_tmp_topk = tik_instance.Tensor("float16", topk_out_ub.shape, name="ub_tmp_topk", scope=tik.scope_ubuf)
            with tik_instance.if_scope(nms.topk_loop_time > 1):
                with tik_instance.for_range(1, nms.topk_loop_time) as _top_k_idx:
                    gm_offset = [batch_idx, class_idx, 0, _top_k_idx * nms.proposal_topk_k]
                    if nms.is_second_nms:
                        workspace_offset = tik_instance.Scalar(dtype='int32', name='workspace_offset')
                        workspace_offset.set_as((_top_k_idx - 1) * sorted_k * 8 + core_idx * (nms.boxes_num * 8))
                        workspace_for_save_proposal = nms.workspace_second_nms_gm[workspace_offset]
                    else:
                        workspace_for_save_proposal = None
                    # get tmp sorted proposal to ub_tmp_topk
                    get_sorted_proposal_compute(tik_instance, nms, ub_tmp_topk, nms.input_gm_list, gm_offset, sorted_k,
                                                sorted_k, clip_window_value_list, l1_valid,
                                                reduce_scalar=nms.down_scalar_list)
                    # sorted two proposals to one proposal list output the top
                    # sorted_k
                    tik_func_sort_with_ub(tik_instance, [ub_tmp_topk, topk_out_ub],
                                          [topk_out_ub, ub_tmp_topk], sorted_k, workspace_for_save_proposal)

            with tik_instance.if_scope(nms.topk_loop_tail != 0):
                gm_offset = [batch_idx, class_idx, 0, nms.topk_loop_time * nms.proposal_topk_k]
                if nms.is_second_nms:
                    workspace_offset = tik_instance.Scalar(dtype='int32', name='workspace_offset')
                    workspace_offset.set_as((nms.topk_loop_time - 1) * sorted_k * 8 + core_idx * nms.boxes_num * 8)
                    workspace_for_save_proposal = nms.workspace_second_nms_gm[workspace_offset]
                else:
                    workspace_for_save_proposal = None
                # get tmp sorted proposal to ub_tmp_topk
                get_sorted_proposal_compute_scalar(tik_instance, nms, ub_tmp_topk, nms.input_gm_list, gm_offset,
                                                   nms.topk_loop_tail, sorted_k, clip_window_value_list, l1_valid,
                                                   reduce_scalar=nms.down_scalar_list)
                # sorted two proposals to one proposal list output the top
                # sorted_k
                tik_func_sort_with_ub(tik_instance, [ub_tmp_topk, topk_out_ub],
                                      [topk_out_ub, ub_tmp_topk], sorted_k,
                                      workspace_for_save_proposal)

    # do nms use topk output to get nms proposals per class
    # and move result to l1
    with tik_instance.new_stmt_scope():
        nms_var = nms.init_tik_ub_mem_for_nms()
        nmsed_result_ub = nms_var.get("selected_proposal_ub")
        nmsed_result_area = nms_var.get("selected_area_ub")
        nmsed_result_sup = nms_var.get("sup_vec_ub")
        # init all sup_vec to 1, mean: no select proposal
        tik_func_vector_scalar(tik_instance, nmsed_result_sup, 1, nms.max_selected_nms_num_in_ub)
        # init select nms proposal = 0
        l1_buffer = nms.l1_nms_result_zero
        loop_burst_len = (nms.max_selected_nms_num_in_ub * 8) // 16
        tik_instance.data_move(nmsed_result_ub, l1_buffer,
                               0, 1, loop_burst_len, 0, 0)
        # init select nms area = 0
        loop_burst_len = nms.max_selected_nms_num_in_ub // 16
        tik_instance.data_move(nmsed_result_area, l1_buffer,
                               0, 1, loop_burst_len, 0, 0)
        with tik_instance.new_stmt_scope():
            do_nms_compute(tik_instance, nms_var, nms.iou_thresh)
        # copy one class nms result to l1
        l1_buffer = nms.l1_nms_result
        l1_offset = [class_idx, 0, 0]
        loop_burst_len = (nms.max_selected_nms_num_in_ub * 8) // 16
        tik_instance.data_move(l1_buffer[l1_offset], nmsed_result_ub,
                               0, 1, loop_burst_len, 0, 0)
        if nms.is_second_nms:
            loop_burst_len = nms.max_selected_nms_num_in_ub // 16
            tik_instance.data_move(nms.l1_nms_area, nmsed_result_area,
                                   0, 1, loop_burst_len, 0, 0)
            tik_instance.data_move(nms.l1_nms_sup, nms_var.get("sup_vec_ub"),
                                   0, 1, loop_burst_len, 0, 0)

    # if the select nms output num of the first top 4096 highest score boxes is less the output need
    # and the impl_mode is high_precision
    # will do nms again from the tail boxes 4096 boxes by 4096 boxes
    tool_loop = tik_instance.Scalar(dtype='int32')
    with tik_instance.if_scope(nms.topk_loop_tail == 0):
        tool_loop.set_as(nms.topk_loop_time)
    with tik_instance.else_scope():
        tool_loop.set_as(nms.topk_loop_time + 1)
    if nms.is_second_nms:
        nms.get_tiling_args()
        with tik_instance.if_scope(tool_loop >= 3):
            # if not to output num
            with tik_instance.for_range(1, tool_loop - 1) as _top_n_idx:
                top_n_num_tail = tik_instance.Scalar(dtype='int32')
                top_n_num_tail.set_as(tool_loop - _top_n_idx - 1)
                with tik_instance.if_scope(nms.selected_proposals_cnt < nms.max_total_size):
                    # copy a sorted proposals to topk_out_ub
                    loop_burst_len = ceil_div(sorted_k * 8, 16)
                    tik_instance.data_move(topk_out_ub, nms.workspace_second_nms_gm[core_idx * nms.boxes_num * 8],
                                           0, 1, loop_burst_len, 0, 0)
                    # apply second top k proposal ub
                    with tik_instance.new_stmt_scope():
                        ub_tmp_topk = tik_instance.Tensor("float16", topk_out_ub.shape,
                                                          name="ub_tmp_topk", scope=tik.scope_ubuf)
                        with tik_instance.for_range(0, top_n_num_tail) as _top_n_tail_idx:
                            workspace_proposal_offset = tik_instance.Scalar(dtype='int32',
                                                                            name='workspace_proposal_offset')
                            workspace_proposal_offset.set_as(
                                sorted_k * 8 + _top_n_tail_idx * sorted_k * 8 + core_idx * nms.boxes_num * 8)
                            tik_instance.data_move(ub_tmp_topk, nms.workspace_second_nms_gm[workspace_proposal_offset],
                                                   0, 1, loop_burst_len, 0, 0)
                            workspace_offset = tik_instance.Scalar(dtype='int32', name='workspace_offset')
                            workspace_offset.set_as(_top_n_tail_idx * sorted_k * 8 + core_idx * nms.boxes_num * 8)
                            workspace_for_save_proposal = nms.workspace_second_nms_gm[workspace_offset]
                            # sorted two proposals to one proposal list output
                            # the top sorted_k
                            tik_func_sort_with_ub(tik_instance, [ub_tmp_topk, topk_out_ub],
                                                  [topk_out_ub, ub_tmp_topk], sorted_k, workspace_for_save_proposal)
                    # do nms use topk output to get nms proposals per class
                    # and move result to l1
                    with tik_instance.new_stmt_scope():
                        nms_var = nms.init_tik_ub_mem_for_nms()
                        nmsed_result_ub = nms_var.get("selected_proposal_ub")
                        nmsed_result_area = nms_var.get("selected_area_ub")
                        nmsed_result_sup = nms_var.get("sup_vec_ub")

                        # copy l1 tmp data to ub
                        l1_buffer = nms.l1_nms_result
                        l1_offset = [class_idx, 0, 0]
                        loop_burst_len = (nms.max_selected_nms_num_in_ub * 8) // 16
                        # copy the selected proposal/area/sup_ub from L1 to UB
                        tik_instance.data_move(nmsed_result_ub, l1_buffer[l1_offset], 0, 1, loop_burst_len, 0, 0)
                        loop_burst_len = nms.max_selected_nms_num_in_ub // 16
                        tik_instance.data_move(nmsed_result_area, nms.l1_nms_area, 0, 1, loop_burst_len, 0, 0)
                        tik_instance.data_move(nmsed_result_sup, nms.l1_nms_sup, 0, 1, loop_burst_len, 0, 0)

                        with tik_instance.new_stmt_scope():
                            do_nms_compute(tik_instance, nms_var, nms.iou_thresh)
                        # copy one class nms result to l1
                        l1_buffer = nms.l1_nms_result
                        l1_offset = [class_idx, 0, 0]
                        loop_burst_len = (nms.max_selected_nms_num_in_ub * 8) // 16
                        tik_instance.data_move(l1_buffer[l1_offset], nmsed_result_ub, 0, 1, loop_burst_len, 0, 0)
                        loop_burst_len = nms.max_selected_nms_num_in_ub // 16
                        tik_instance.data_move(nms.l1_nms_area, nmsed_result_area, 0, 1, loop_burst_len, 0, 0)
                        tik_instance.data_move(nms.l1_nms_sup, nmsed_result_sup, 0, 1, loop_burst_len, 0, 0)


def get_class_tensor(
        tik_instance,
        class_ub,
        class_num,
        len_per_class,
        start_class=0.0):
    """
    get class tensor
    """
    tik_func_vector_scalar(tik_instance, class_ub, start_class, len_per_class)
    with tik_instance.for_range(1, class_num) as _class_idx:
        dst_offset = _class_idx * len_per_class
        src_offset = (_class_idx - 1) * len_per_class
        _repeat_time = len_per_class // 128
        _repeat_tail = len_per_class % 128
        with tik_instance.if_scope(_repeat_time != 0):
            tik_instance.vadds(128, class_ub[dst_offset], class_ub[src_offset], 1.0, _repeat_time, 1, 1, 8, 8)
            dst_offset = 128 * _repeat_time + dst_offset
            src_offset = 128 * _repeat_time + src_offset
        with tik_instance.if_scope(_repeat_tail != 0):
            tik_instance.vadds(_repeat_tail, class_ub[dst_offset], class_ub[src_offset], 1.0, 1, 1, 1, 8, 8)


def copy_tail_data(
        tik_instance,
        gm_dst_info,
        ub_src_info,
        gm_workspace_info,
        copy_len):
    """
    copy_tail_data when output is not align, will use workspace to align force
    """
    gm_dst, gm_dst_offset = gm_dst_info
    ub_src, ub_src_offset = ub_src_info
    gm_workspace, gm_workspace_offset = gm_workspace_info
    data_type = ub_src.dtype
    if data_type in ("float32", "int32"):
        block_num = 8
    else:
        block_num = 16
    copy_nbust_len = copy_len // block_num
    copy_tail_offset = copy_len % block_num
    tik_instance.data_move(gm_dst[gm_dst_offset], ub_src[ub_src_offset], 0, 1, copy_nbust_len, 0, 0)
    tik_instance.data_move(gm_workspace[gm_workspace_offset], ub_src[ub_src_offset + (copy_nbust_len - 1) * block_num],
                           0, 1, 2, 0, 0)
    tik_instance.data_move(ub_src[ub_src_offset], gm_workspace[gm_workspace_offset + copy_tail_offset],
                           0, 1, 1, 0, 0)
    tik_instance.data_move(gm_dst[gm_dst_offset + copy_tail_offset + (copy_nbust_len - 1) *
                                  block_num], ub_src[ub_src_offset], 0, 1, 1, 0, 0)


def batch_multi_class_nms_copy_out(
        tik_instance,
        nms,
        ub_result_boxes,
        ub_result_boxes_class,
        output_batch_offset,
        workspace_core_offset):
    """
    batch_multi_class_nms_copy_out
    """
    workspace_flag = tik_instance.Scalar("int32")
    workspace_flag.set_as(0)
    with tik_instance.if_scope(tik.all(nms.core_used > 1, nms.max_total_size % 16 != 0)):
        workspace_flag.set_as(1)

    workspace = nms.workspace_proposal_gm
    down_scalar = None
    if nms.down_flag:
        down_scalar = nms.down_scalar_list[1]
    loop_burst_len = tik_instance.Scalar(dtype='int32', name='loop_burst_len')
    loop_burst_len.set_as((nms.max_total_size - 1) // 16 + 1)
    # ceil_div
    apply_men_len = (nms.max_total_size - 1) // 16 + 1
    less_flag = tik_instance.Scalar(dtype='int32', name='less_flag')
    less_flag.set_as(0)
    with tik_instance.if_scope(nms.max_selected_nms_num_in_ub * nms.classes < nms.max_total_size):
        less_flag.set_as(1)
        loop_burst_len.set_as((nms.max_selected_nms_num_in_ub * nms.classes - 1) // 16 + 1)
    score_thresh = nms.score_thresh
    _batch = output_batch_offset // nms.max_total_size
    ub_scores_valid_mask = tik_instance.Tensor("float16", [apply_men_len * 16],
                                               name="ub_scores_valid_mask", scope=tik.scope_ubuf)
    # process scores
    with tik_instance.new_stmt_scope():
        # scores
        ub_out_scores = tik_instance.Tensor("float16", [apply_men_len * 16],
                                            name="ub_out_scores", scope=tik.scope_ubuf)
        ub_out_scores_valid = tik_instance.Tensor("int32", [16], name="ub_out_scores_valid", scope=tik.scope_ubuf)
        with tik_instance.if_scope(less_flag > 0):
            tik_func_vector_scalar(tik_instance, ub_out_scores, 0, apply_men_len * 16)
        tik_func_vextract(tik_instance, ub_result_boxes_class, ub_out_scores, loop_burst_len, 3)
        filter_score_compute(tik_instance, ub_out_scores, ub_out_scores_valid, ub_scores_valid_mask,
                             nms.max_total_size, score_thresh)
        with tik_instance.if_scope(workspace_flag == 0):
            tik_instance.data_move(nms.output_gm_list[1][output_batch_offset], ub_out_scores, 0, 1, apply_men_len, 0, 0)
        with tik_instance.else_scope():
            copy_tail_data(tik_instance, [nms.output_gm_list[1], output_batch_offset],
                           [ub_out_scores, 0], [workspace, workspace_core_offset], nms.max_total_size)

        tik_instance.data_move(nms.output_gm_list[3][_batch * 8], ub_out_scores_valid, 0, 1, 1, 0, 0)
        # x1
        ub_out_box_x1 = tik_instance.Tensor("float16", [apply_men_len * 16],
                                            name="ub_out_box_x1", scope=tik.scope_ubuf)
        tik_func_vextract(tik_instance, ub_result_boxes, ub_out_box_x1, loop_burst_len, 0)
        tik_func_vcomple_scalar(tik_instance, "vmul", ub_out_box_x1, ub_scores_valid_mask, ub_out_box_x1,
                                apply_men_len * 16)
        if nms.down_flag:
            tik_func_vmuls_scalar(tik_instance, ub_out_box_x1, ub_out_box_x1, down_scalar, nms.max_total_size)
        # y1
        ub_out_box_y1 = tik_instance.Tensor("float16", [apply_men_len * 16],
                                            name="ub_out_box_y1", scope=tik.scope_ubuf)
        tik_func_vextract(tik_instance, ub_result_boxes, ub_out_box_y1, loop_burst_len, 1)
        tik_func_vcomple_scalar(tik_instance, "vmul", ub_out_box_y1, ub_scores_valid_mask, ub_out_box_y1,
                                apply_men_len * 16)
        # DOWN_FACTOR
        if nms.down_flag:
            tik_func_vmuls_scalar(tik_instance, ub_out_box_y1, ub_out_box_y1, down_scalar, nms.max_total_size)
        tik_instance.data_move(nms.output_gm_list[0][output_batch_offset * 4], ub_out_box_x1,
                               0, 1, apply_men_len, 0, 0)
        tik_instance.data_move(nms.output_gm_list[0][output_batch_offset * 4 + nms.max_total_size],
                               ub_out_box_y1, 0, 1, apply_men_len, 0, 0)

        # x2
        ub_out_box_x2 = tik_instance.Tensor("float16", [apply_men_len * 16],
                                            name="ub_out_box_x2", scope=tik.scope_ubuf)
        tik_func_vextract(tik_instance, ub_result_boxes, ub_out_box_x2, loop_burst_len, 2)

        if not nms.is_need_rpn_offset:
            tik_func_vadds_scalar(tik_instance, ub_out_box_x2, ub_out_box_x2, 1.0, nms.max_total_size)

        if nms.down_flag:
            tik_func_vmuls_scalar(tik_instance, ub_out_box_x2, ub_out_box_x2, down_scalar, nms.max_total_size)
        tik_func_vcomple_scalar(tik_instance, "vmul", ub_out_box_x2, ub_scores_valid_mask, ub_out_box_x2,
                                apply_men_len * 16)
        tik_instance.data_move(nms.output_gm_list[0][output_batch_offset * 4 +
                                                     nms.max_total_size * 2], ub_out_box_x2, 0, 1, apply_men_len, 0, 0)

        # y2
        ub_out_box_y2 = tik_instance.Tensor("float16", [apply_men_len * 16],
                                            name="ub_out_box_y2", scope=tik.scope_ubuf)
        tik_func_vextract(tik_instance, ub_result_boxes, ub_out_box_y2, loop_burst_len, 3)

        if not nms.is_need_rpn_offset:
            tik_func_vadds_scalar(tik_instance, ub_out_box_y2, ub_out_box_y2, 1.0, nms.max_total_size)

        if nms.down_flag:
            tik_func_vmuls_scalar(tik_instance, ub_out_box_y2, ub_out_box_y2, down_scalar, nms.max_total_size)
        tik_func_vcomple_scalar(tik_instance, "vmul", ub_out_box_y2, ub_scores_valid_mask, ub_out_box_y2,
                                apply_men_len * 16)
        with tik_instance.if_scope(workspace_flag == 0):
            tik_instance.data_move(nms.output_gm_list[0][output_batch_offset * 4 + nms.max_total_size * 3],
                                   ub_out_box_y2, 0, 1, apply_men_len, 0, 0)
        with tik_instance.else_scope():
            copy_tail_data(tik_instance, [nms.output_gm_list[0], output_batch_offset * 4 + nms.max_total_size * 3],
                           [ub_out_box_y2, 0], [workspace, workspace_core_offset], nms.max_total_size)
        # class
        ub_out_class = tik_instance.Tensor("float16", [apply_men_len * 16],
                                           name="ub_out_class", scope=tik.scope_ubuf)
        tik_func_vextract(tik_instance, ub_result_boxes_class, ub_out_class, loop_burst_len, 0)

        with tik_instance.if_scope(workspace_flag == 0):
            tik_instance.data_move(nms.output_gm_list[2][output_batch_offset], ub_out_class, 0, 1, apply_men_len, 0, 0)
        with tik_instance.else_scope():
            copy_tail_data(tik_instance, [nms.output_gm_list[2], output_batch_offset],
                           [ub_out_class, 0], [workspace, workspace_core_offset], nms.max_total_size)


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
    # get score batch offset
    nms.get_tiling_args()
    result_total = total_num(nms.l1_nms_result.shape)
    output_batch_offset = _batch_idx * nms.max_total_size
    workspace = nms.workspace_proposal_gm
    workspace_offset = tik_instance.Scalar(dtype='int32', name='workspace_offset')
    workspace_offset.set_as(core_idx * (nms.classes * nms.max_selected_nms_num_in_ub * 8 + 128))
    with tik_instance.if_scope(nms.classes * nms.max_selected_nms_num_in_ub < nms.proposal_topk_k):
        # when all output is less nms.proposal_topk_k
        # only use topk with ub for output proposal
        with tik_instance.new_stmt_scope():
            ub_result_boxes = tik_instance.Tensor("float16", [result_total // 8, 8],
                                                  name="ub_result_boxes", scope=tik.scope_ubuf)
            ub_result_boxes_class = tik_instance.Tensor("float16", [result_total // 8, 8],
                                                        name="ub_result_boxes_class", scope=tik.scope_ubuf)
            l1_buffer = nms.l1_nms_result
            l1_offset = [0, 0, 0]
            loop_burst_len = result_total // 16
            tik_instance.data_move(ub_result_boxes, l1_buffer[l1_offset],
                                   0, 1, loop_burst_len, 0, 0)
            tik_instance.data_move(ub_result_boxes_class, l1_buffer[l1_offset],
                                   0, 1, loop_burst_len, 0, 0)
            with tik_instance.new_stmt_scope():
                ub_class_all = tik_instance.Tensor("float16", [nms.max_selected_nms_num_in_ub * nms.classes],
                                                   name="ub_class_all", scope=tik.scope_ubuf)
                get_class_tensor(tik_instance, ub_class_all, nms.classes, nms.max_selected_nms_num_in_ub)

                trans_repeat = tik_instance.Scalar(dtype='int32', name='trans_repeat')
                trans_repeat.set_as((nms.max_selected_nms_num_in_ub * nms.classes - 1) // 16 + 1)
                tik_func_vconcat(tik_instance, ub_result_boxes_class, ub_class_all, trans_repeat, 1)
                tik_instance.data_move(workspace[workspace_offset], ub_result_boxes_class, 0, 1, loop_burst_len, 0, 0)
                tik_instance.data_move(ub_result_boxes_class, workspace[workspace_offset + 1],
                                       0, 1, loop_burst_len, 0, 0)
                tik_func_vextract(tik_instance, ub_result_boxes_class, ub_class_all, trans_repeat, 3)
                tik_func_vconcat(tik_instance, ub_result_boxes_class, ub_class_all, trans_repeat, 4)

            with tik_instance.if_scope(nms.classes != 1):
                level = tik_instance.Scalar(dtype='int32', init_value=1)
                region = tik_instance.Scalar(dtype='int32')
                region.set_as((result_total // 8 + 15) // 16)
                with tik_instance.if_scope(region > 1):
                    with tik_instance.for_range(0, Constant.RPN_PROPOSAL_NUM) as i:
                        with tik_instance.if_scope(region > 1):
                            level.set_as(level + 1)
                            region.set_as((region + 3) // 4)
                sort_within_ub_scalar(tik_instance, ub_result_boxes_class, result_total // 8, level)
                region.set_as((result_total // 8 + 15) // 16)
                level.set_as(1)
                with tik_instance.if_scope(region > 1):
                    with tik_instance.for_range(0, Constant.RPN_PROPOSAL_NUM) as i:
                        with tik_instance.if_scope(region > 1):
                            level.set_as(level + 1)
                            region.set_as((region + 3) // 4)
                sort_within_ub_scalar(tik_instance, ub_result_boxes, result_total // 8, level)

            with tik_instance.new_stmt_scope():
                batch_multi_class_nms_copy_out(tik_instance, nms, ub_result_boxes, ub_result_boxes_class,
                                               output_batch_offset, workspace_offset)
    with tik_instance.else_scope():
        l1_buffer = nms.l1_nms_result
        copy_classes_num_int = tik_instance.Scalar(dtype='int32')
        copy_classes_num_half = tik_instance.Scalar(dtype='float16')
        copy_classes_num_fp32 = tik_instance.Scalar(dtype='float32')
        copy_classes_num_int.set_as(nms.proposal_topk_k // nms.max_selected_nms_num_in_ub // 2)
        tik_instance.scalar_conv("", copy_classes_num_fp32, copy_classes_num_int)
        tik_instance.scalar_conv("", copy_classes_num_half, copy_classes_num_fp32)

        copy_loop = tik_instance.Scalar(dtype='int32', name='copy_loop')
        copy_loop.set_as(nms.classes // copy_classes_num_int)
        copy_tail = tik_instance.Scalar(dtype='int32', name='copy_tail')
        copy_tail.set_as(nms.classes % copy_classes_num_int)
        tmp_output_proposal_num = ((nms.max_total_size - 1) // 16 + 1) * 16
        ub_out_result = tik_instance.Tensor("float16", [tmp_output_proposal_num, 8],
                                            name="ub_out_result", scope=tik.scope_ubuf)
        ub_out_result_class = tik_instance.Tensor("float16", [tmp_output_proposal_num, 8],
                                                  name="ub_out_result_class", scope=tik.scope_ubuf)
        tik_func_vector_scalar(tik_instance, ub_out_result, 0.0, tmp_output_proposal_num * 8)
        tik_func_vector_scalar(tik_instance, ub_out_result_class, 0.0, tmp_output_proposal_num * 8)
        with tik_instance.new_stmt_scope():
            ub_result_boxes = tik_instance.Tensor("float16", [copy_classes_num_int * nms.max_selected_nms_num_in_ub,
                                                  8], name="ub_result_boxes", scope=tik.scope_ubuf)
            ub_result_boxes_class = tik_instance.Tensor("float16", [copy_classes_num_int
                                                                    * nms.max_selected_nms_num_in_ub,
                                                                    8], name="ub_result_boxes_class",
                                                        scope=tik.scope_ubuf)
            ub_class_all = tik_instance.Tensor("float16", [nms.max_selected_nms_num_in_ub * copy_classes_num_int],
                                               name="ub_class_all", scope=tik.scope_ubuf)
            get_class_tensor(tik_instance, ub_class_all, copy_classes_num_int, nms.max_selected_nms_num_in_ub,
                             -1 * copy_classes_num_int)

            # 'pylint: disable=unused-variable
            def _do_copy_and_vconcat_class(_l1_offset, _loop_burst_len,):
                """
                _do_copy_and_vconcat_class
                """
                tik_instance.data_move(ub_result_boxes, l1_buffer[_l1_offset], 0, 1, _loop_burst_len, 0, 0)
                tik_instance.data_move(ub_result_boxes_class, l1_buffer[_l1_offset], 0, 1, _loop_burst_len, 0, 0)
                # get copy_classes_num sort
                tik_func_vadds_scalar(tik_instance, ub_class_all, ub_class_all, copy_classes_num_half,
                               nms.max_selected_nms_num_in_ub * copy_classes_num_int)
                _trans_repeat = ceil_div(nms.max_selected_nms_num_in_ub * copy_classes_num_int, 16)
                tik_func_vconcat(tik_instance, ub_result_boxes_class, ub_class_all, _trans_repeat, 1)
                tik_instance.data_move(workspace[workspace_offset], ub_result_boxes_class, 0, 1, _loop_burst_len, 0, 0)
                tik_instance.data_move(ub_result_boxes_class, workspace[workspace_offset + 1], 0, 1, _loop_burst_len,
                                       0, 0)
                with tik_instance.new_stmt_scope():
                    ub_class_tmp = tik_instance.Tensor("float16", [nms.max_selected_nms_num_in_ub *
                                                                    copy_classes_num_int],
                                                       name="ub_class_tmp", scope=tik.scope_ubuf)
                    tik_func_vextract(tik_instance, ub_result_boxes_class, ub_class_tmp, _trans_repeat, 3)
                    tik_func_vconcat(tik_instance, ub_result_boxes_class, ub_class_tmp, _trans_repeat, 4)

            with tik_instance.for_range(0, copy_loop) as _class_idx:
                l1_offset = [_class_idx * copy_classes_num_int, 0, 0]
                loop_burst_len = copy_classes_num_int * nms.max_selected_nms_num_in_ub * 8 // 16
                _do_copy_and_vconcat_class(l1_offset, loop_burst_len)
                level = tik_instance.Scalar(dtype='int32', init_value=1)
                region = tik_instance.Scalar(dtype='int32')
                region.set_as((copy_classes_num_int * nms.max_selected_nms_num_in_ub + 15) // 16)
                with tik_instance.if_scope(region > 1):
                    with tik_instance.for_range(0, Constant.RPN_PROPOSAL_NUM) as i:
                        with tik_instance.if_scope(region > 1):
                            level.set_as(level + 1)
                            region.set_as((region + 3) // 4)
                sort_within_ub_scalar(tik_instance, ub_result_boxes, copy_classes_num_int *
                                      nms.max_selected_nms_num_in_ub, level)
                sort_within_ub_scalar(tik_instance, ub_result_boxes_class, copy_classes_num_int *
                                      nms.max_selected_nms_num_in_ub, level)
                tik_func_sort_with_ub(tik_instance, [ub_out_result, ub_result_boxes], [ub_out_result, ub_result_boxes],
                                      tmp_output_proposal_num)
                tik_func_sort_with_ub(tik_instance, [ub_out_result_class, ub_result_boxes_class],
                                      [ub_out_result_class, ub_result_boxes_class], tmp_output_proposal_num)

            with tik_instance.if_scope(copy_tail != 0):
                l1_offset = [copy_loop * copy_classes_num_int, 0, 0]
                loop_burst_len = tik_instance.Scalar(dtype='int32', name='loop_burst_len')
                loop_burst_len.set_as(copy_tail * nms.max_selected_nms_num_in_ub * 8 // 16)
                _do_copy_and_vconcat_class(l1_offset, loop_burst_len)
                level = tik_instance.Scalar(dtype='int32', init_value=1)
                region = tik_instance.Scalar(dtype='int32')
                region.set_as((copy_tail * nms.max_selected_nms_num_in_ub + 15) // 16)
                with tik_instance.if_scope(region > 1):
                    with tik_instance.for_range(0, Constant.RPN_PROPOSAL_NUM) as i:
                        with tik_instance.if_scope(region > 1):
                            level.set_as(level + 1)
                            region.set_as((region + 3) // 4)
                sort_within_ub_scalar(tik_instance, ub_result_boxes, copy_tail * nms.max_selected_nms_num_in_ub, level)
                sort_within_ub_scalar(tik_instance, ub_result_boxes_class, copy_tail * nms.max_selected_nms_num_in_ub,
                                      level)
                with tik_instance.if_scope(copy_tail * nms.max_selected_nms_num_in_ub < tmp_output_proposal_num):
                    dup_len = tik_instance.Scalar(dtype='int32', name='dup_len')
                    dup_len.set_as(tmp_output_proposal_num - copy_tail * nms.max_selected_nms_num_in_ub)
                    dup_offset = tik_instance.Scalar(dtype='int32', name='dup_offset')
                    dup_offset.set_as(copy_tail * nms.max_selected_nms_num_in_ub)
                    tik_func_vector_scalar(tik_instance, ub_result_boxes[dup_offset:], 0.0, dup_len * 8)
                    tik_func_vector_scalar(tik_instance, ub_result_boxes_class[dup_offset:], 0.0, dup_len * 8)
                tik_func_sort_with_ub(tik_instance, [ub_out_result, ub_result_boxes], [ub_out_result, ub_result_boxes],
                                      tmp_output_proposal_num)
                tik_func_sort_with_ub(tik_instance, [ub_out_result_class, ub_result_boxes_class],
                                      [ub_out_result_class, ub_result_boxes_class], tmp_output_proposal_num)
        with tik_instance.new_stmt_scope():
            batch_multi_class_nms_copy_out(tik_instance, nms, ub_out_result, ub_out_result_class,
                                           output_batch_offset, workspace_offset)


@register_operator('BatchMultiClassNonMaxSuppression')
def batch_multi_class_non_max_suppression(boxes, scores, clip_window, num_valid_boxes, nmsed_boxes, nmsed_scores,
        nmsed_classes, nmsed_num, score_threshold, iou_threshold, max_size_per_class, max_total_size,
        change_coordinate_frame, transpose_box, image_size=(),
        kernel_name="batch_multi_class_non_max_suppression", impl_mode=OpImplMode.HIGH_PERFORMANCE):
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
    kernel_name : str.
        cce kernel name, default value is "batch_multi_class_non_max_suppression"
    impl_mode: str.
        high_precision or high_performance for inference, default value is "high_performance".
        no need to add into ops_info file.

    Returns
    -------
    tik_instance
    """
    check_op_impl_mode(impl_mode, [OpImplMode.HIGH_PERFORMANCE, OpImplMode.HIGH_PRECISION], kernel_name)

    if tbe_platform.api_check_support("tik.vcopy"):
        obj = BMCNMS(boxes, scores, clip_window, num_valid_boxes, score_threshold,
                     iou_threshold, max_size_per_class, max_total_size,
                     change_coordinate_frame, transpose_box, kernel_name)
        return obj.bmcnms_compute()
    tiling_mode_1 = 1
    nms = BatchMultiClassNonMaxSuppression(boxes, scores, num_valid_boxes, clip_window, score_threshold, iou_threshold,
                                           max_size_per_class, max_total_size, change_coordinate_frame, impl_mode)
    # init ub
    nms.get_tiling_args()
    nms.init_tik_mem(nms)
    tik_instance = nms.get_tik_instance()
    with tik_instance.if_scope(nms.max_total_size < 16):
        nms.core_used.set_as(1)

    def _run_one_core(_real_batch_idx, _real_core_idx):
        """
        _run_one_core
        """
        if nms.need_clip_window:
            read_window_compute(tik_instance, nms.input_gm_list[2], _real_batch_idx * 4, nms.clip_window_value_list,
                                nms.down_scalar_list, nms.change_coordinate_frame)

        if nms.need_valid_num:
            read_valid_num_compute(tik_instance, nms.input_gm_list[-1], [_real_batch_idx], nms.valid_num_value)
            gen_valid_num_compute(tik_instance, nms.l1_score_valid, nms.boxes_num, nms.valid_num_value)

        with tik_instance.for_range(0, nms.classes) as _class_idx:
            # `for each class, init selected_proposals_cnt = 0`
            nms.selected_proposals_cnt.set_as(0)
            with tik_instance.new_stmt_scope():
                nms_for_single_class(_real_batch_idx, _class_idx, nms, _real_core_idx)

        # process all class output result is in l1_nms_result, will process output
        # step 1 sort all select proposal with boxes
        # step 2 sort all select proposal with classes score
        with tik_instance.new_stmt_scope():
            batch_multi_class_nms_output(tik_instance, _real_core_idx, _real_batch_idx, nms)

    # do nms with multi cores
    cook = tik_instance.Scalar(dtype='int32')
    real_batch_idx = tik_instance.Scalar(dtype='int32')
    with tik_instance.for_range(0, nms.aicore_num, block_num=nms.aicore_num) as _core_idx:
        nms.get_tiling_args()
        with tik_instance.if_scope(_core_idx == (nms.core_used - 1)):
            cook.set_as(nms.batch_last_core)
        with tik_instance.else_scope():
            cook.set_as(nms.batch_per_core)
        with tik_instance.if_scope(_core_idx < nms.core_used):
            with tik_instance.if_scope(nms.cal_mode == tiling_mode_1):
                with tik_instance.for_range(0, cook) as _batch_idx:
                    real_batch_idx.set_as(_core_idx * nms.batch_per_core + _batch_idx)
                    _run_one_core(real_batch_idx, _core_idx)
    return nms.build_tik_instance(kernel_name)
