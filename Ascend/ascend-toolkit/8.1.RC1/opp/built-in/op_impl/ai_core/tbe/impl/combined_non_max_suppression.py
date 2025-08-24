##!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
combined_non_max_suppression
"""
import functools
from impl import common_util
from impl import constant_util
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util import util_tik_comm_func
from impl.batch_multi_class_nms_topk import sort_within_ub
from impl.batch_multi_class_non_max_suppression import nms_for_single_class
from impl.batch_multi_class_non_max_suppression import tik_func_sort_with_ub
from impl.batch_multi_class_non_max_suppression import filter_score_compute
from impl.util.platform_adapter import tbe_platform
from impl.util.util_tik_comm_func import tik_func_vmins
from impl.util.util_tik_comm_func import tik_func_vmaxs
from impl.util.util_tik_comm_func import sort_score_idx_by_desc
from impl.util.util_tik_comm_func import init_index
from impl.util.util_tik_comm_func import gm2ub_for_vsort32
from impl.util.util_common import get_mask_rep_stride
from impl.util.util_common import check_op_impl_mode
from impl.util.platform_adapter import OpImplMode


# 'pylint: disable=too-many-lines,too-many-instance-attributes,too-many-statements
# 'pylint: disable=too-many-arguments,unused-argument,too-many-locals,too-many-branches
# 'pylint: disable=invalid-name


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    the class for constant
    """
    # scaling factor
    DOWN_FACTOR = 0.10
    # RPN compute 16 proposals per iteration
    RPN_PROPOSAL_NUM = 16
    # define the positive min value in fp16
    MIN_SCALAR_FP16 = 2 ** (-24)
    # define a fp16 value = 2**12
    TMP_SCALAR_FP16 = 2 ** 12
    # define repeat elements every time for vsrot32
    REPEAT_ELE = 32
    # every loop process 4096 units
    PER_LOOP_UNIT = 4096
    # every loop process 4096 units during batch sorting scores_idx
    BATCH_LOOP_UNIT = 1024
    # location elements, [x1, y1, x2, y2]
    FOUR_DIRECTION = 4
    # b16 elements num of every block also uesed as b16 elements num of mask
    BLOCK_ELE = 16
    # b32 elements num of every block
    BLOCK_ELE_B32 = 8
    # different size of different dtype
    INT8_SIZE = 1
    FP16_SIZE = 2
    FP32_SIZE = 4
    # the socres_index contains four/two elements(fp16/fp32) also marked as the class num processed every cycle
    UNIT_ELE = 4
    UNIT_ELE_FP32 = 2

    REPEAT_TIMES_MAX = 255
    # 0b0001 0001 0001 0001 is equals to type 3
    PATTERN_TYPE = 3
    # 0b0101 0001 0101 0101 get the odd-bit data
    PATTERN_TYPE_FP32 = 1

    # PIECES
    FOUR_PIECE = 4
    TWO_PIECE = 2


# 'pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-statements,too-many-locals
class CombinedNonMaxSuppression:
    """
    Function: use to store CombinedNonMaxSuppression base parameters
    Modify : 2021-2-19
    """
    def __init__(self,
                 boxes,
                 scores,
                 input_scalar_list,
                 score_thresh,
                 iou_thresh,
                 max_size_per_class,
                 max_total_size,
                 impl_mode):
        """
        Init CombinedNonMaxSuppression base parameters

        Returns
        -------
        None
        """
        boxes_shape = list(boxes.get("shape"))
        self.boxes_type = boxes.get("dtype")
        scores_shape = list(scores.get("shape"))
        # when input have no class dim, will extend 1 for input shape
        if len(scores_shape) == 2 and len(boxes_shape) == 3:
            self.boxes_shape = [boxes_shape[0], 1, boxes_shape[1], boxes_shape[2]]
            self.scores_shape = [scores_shape[0], 1, scores_shape[1]]
        else:
            self.boxes_shape = boxes_shape
            self.scores_shape = scores_shape
        self.input_scalar_list = input_scalar_list

        self.need_clip_window = False
        self.clip_window_shape = None

        self.need_valid_num = False
        self.valid_num_shape = None

        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh / (1 + iou_thresh)
        self.max_size_per_class = max_size_per_class
        self.max_total_size = max_total_size
        self.change_coordinate_frame = False

        para_check.check_shape(self.boxes_shape, min_rank=4, max_rank=4, param_name="boxes")
        para_check.check_shape(self.scores_shape, min_rank=3, max_rank=3, param_name="scores")
        # parsing input
        _, self.boxes_classes, _, _ = self.boxes_shape
        self.batch, self.classes, self.boxes_num = self.scores_shape
        if self.classes == self.boxes_classes and self.boxes_classes == 1:
            if self.max_size_per_class > self.max_total_size:
                self.max_size_per_class = self.max_total_size
        check_par(self.max_size_per_class, self.max_total_size, self.classes, self.boxes_type)
        # whether down the boxes to avoid fp16 overflow
        self.down_flag = False
        self.is_second_nms = False
        if impl_mode == "high_precision":
            self.is_second_nms = True

        self.tik_instance = tik.Tik()
        self.aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.input_gm_list = []
        self.output_gm_list = []

        # calcu output shape
        self.nmsed_boxes_shape = [self.batch, 4, self.max_total_size]
        self.nmsed_scores_shape = [self.batch, self.max_total_size]
        self.nmsed_classes_shape = [self.batch, self.max_total_size]
        self.nmsed_num_shape = [self.batch, 8]

        # for topk
        self.ub_max_topk = None
        self.l1_nms_result = None
        self.l1_nms_result_zero = None
        self.workspace_proposal_gm = None
        self.workspace_second_nms_gm = None
        self.l1_score_valid = None
        self.l1_nms_area = None
        self.l1_nms_sup = None
        self.proposal_topk_k = self.ub_size // 4 // 16
        self.proposal_topk_k = min(self.proposal_topk_k, 255 * 16)
        self.topk_loop_time = 0
        self.topk_loop_tail = 0
        self.single_loop = True
        if self.boxes_num > self.proposal_topk_k:
            self.single_loop = False
            self.topk_loop_time = self.boxes_num // self.proposal_topk_k
            self.topk_loop_tail = self.boxes_num % self.proposal_topk_k
        self.topk_loop_time_reg = self.tik_instance.Scalar(dtype="int32")
        self.topk_loop_time_reg.set_as(self.topk_loop_time)
        self.topk_loop_time_tail = self.tik_instance.Scalar(dtype="int32")
        self.topk_loop_time_tail.set_as(self.topk_loop_tail)

        # whether user set_rpn_offset, mini do not support it
        self.is_need_rpn_offset = False

        # for nms function param calc
        self.max_selected_nms_num_in_ub = \
            ceil_div(max_size_per_class, Constant.RPN_PROPOSAL_NUM) * Constant.RPN_PROPOSAL_NUM
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
        # init 4 clip to windows scalar
        if self.need_clip_window:
            if self.change_coordinate_frame:
                self.down_flag = False
                self.clip_window_value_list = [self.tik_instance.Scalar(dtype="float16")] * 6
            else:
                self.clip_window_value_list = [self.tik_instance.Scalar(dtype="float16")] * 4
        else:
            self.clip_window_value_list = None
        # init 1 valid num scalar
        self.valid_num_value = self.tik_instance.Scalar(dtype="int32")

        self.down_scalar_list = None
        # init down scalar
        if self.down_flag:
            self.down_scalar_list = [self.tik_instance.Scalar(dtype="float16")] * 2
            self.down_scalar_list[0].set_as(Constant.DOWN_FACTOR)
            self.down_scalar_list[1].set_as(1 / Constant.DOWN_FACTOR)

    def get_tik_instance(self):
        """
        get_tik_instance
        """
        return self.tik_instance

    # 'pylint: disable=unused-argument
    @staticmethod
    def get_l1_core_idx(core_idx):
        """
        get l1 core idx
        """
        return 0

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

    def init_tik_mem(self):
        """
        init tik gm mem
        """
        # init gm input
        boxes_gm = self.tik_instance.Tensor("float16", self.boxes_shape, name="boxes_gm", scope=tik.scope_gm)
        scores_gm = self.tik_instance.Tensor("float16", self.scores_shape, name="scores_gm", scope=tik.scope_gm)

        self.input_gm_list = get_input_gm_list(self.tik_instance, self.need_valid_num, self.valid_num_shape,
                                               self.need_clip_window, self.clip_window_shape, boxes_gm,
                                               scores_gm, self.input_scalar_list)
        # init gm output
        nmsed_boxes_gm = self.tik_instance.Tensor("float16", self.nmsed_boxes_shape,
                                                  name="nmsed_boxes_gm", scope=tik.scope_gm)
        nmsed_scores_gm = self.tik_instance.Tensor("float16", self.nmsed_scores_shape,
                                                   name="nmsed_scores_gm", scope=tik.scope_gm)
        nmsed_classes_gm = self.tik_instance.Tensor("float16", self.nmsed_classes_shape,
                                                    name="nmsed_classes_gm", scope=tik.scope_gm)
        nmsed_num_gm = self.tik_instance.Tensor("int32", self.nmsed_num_shape,
                                                name="nmsed_num_gm", scope=tik.scope_gm)
        self.output_gm_list = [nmsed_boxes_gm, nmsed_scores_gm, nmsed_classes_gm, nmsed_num_gm]

        # init l1 buff for save multi class nms result, size = [classes, self.max_selected_nms_num_in_ub, 8]
        self.l1_nms_result = self.tik_instance.Tensor("float16", (1, self.classes, self.max_selected_nms_num_in_ub, 8),
                                                      name="l1_nms_result", scope=tik.scope_cbuf)

        if self.is_second_nms:
            # init l1 buff for save multi class nms area, size = [self.max_selected_nms_num_in_ub]
            self.l1_nms_area = self.tik_instance.Tensor("float16", (self.max_selected_nms_num_in_ub,),
                                                        name="l1_nms_area_tmp", scope=tik.scope_cbuf)
            # init l1 buff for save multi class nms sup, size = [self.max_selected_nms_num_in_ub]
            self.l1_nms_sup = self.tik_instance.Tensor("uint16", (self.max_selected_nms_num_in_ub,),
                                                       name="l1_nms_sup_tmp", scope=tik.scope_cbuf)

        # zero data in l1
        self.l1_nms_result_zero = \
            self.tik_instance.Tensor("float16", (1, self.max_selected_nms_num_in_ub, 8),
                                     name="l1_nms_result_zero", scope=tik.scope_cbuf)
        with self.tik_instance.new_stmt_scope():
            ub_nms_result = self.tik_instance.Tensor("float16", (self.max_selected_nms_num_in_ub, 8),
                                                     name="ub_nms_result", scope=tik.scope_ubuf)
            util_tik_comm_func.tik_func_vector(self.tik_instance, ub_nms_result, 0,
                                               self.max_selected_nms_num_in_ub * 8)
            loop_burst_len = (self.max_selected_nms_num_in_ub * 8) // 16
            self.tik_instance.data_move(self.l1_nms_result_zero,
                                        ub_nms_result, 0, 1, loop_burst_len, 0, 0)
        # workspace
        self.workspace_proposal_gm = self.tik_instance.Tensor("float16",
                                                              [self.aicore_num,
                                                               total_num(self.l1_nms_result.shape[1:]) + 128],
                                                              name="workspace_proposal_gm",
                                                              scope=tik.scope_gm, is_workspace=True)
        # workspace for second nms
        if self.is_second_nms:
            self.workspace_second_nms_gm = self.tik_instance.Tensor("float16",
                                                                    [self.aicore_num,
                                                                     self.boxes_num * 8],
                                                                    name="workspace_second_nms_gm",
                                                                    scope=tik.scope_gm, is_workspace=True)

    def init_tik_ub_mem_for_nms(self):
        """
        init_tik_ub_mem_for_nms
        """
        ub_selected_proposals = self.tik_instance.Tensor("float16", [self.max_selected_nms_num_in_ub, 8],
                                                         name="ub_selected_proposals", scope=tik.scope_ubuf)
        ub_selected_area = self.tik_instance.Tensor("float16", [self.max_selected_nms_num_in_ub],
                                                    name="ub_selected_area", scope=tik.scope_ubuf)
        ub_sup_vec = self.tik_instance.Tensor("uint16", [self.max_selected_nms_num_in_ub], name="ub_sup_vec",
                                              scope=tik.scope_ubuf)

        # when is_need_rpn_offset set rpn offset for vaadd and viou
        # else x2/y2 will do vadds -1 before nms and do vadds 1 after nms
        if self.is_need_rpn_offset:
            self.tik_instance.set_rpn_offset(0.0)

        topk_out_num = self.proposal_topk_k
        if self.boxes_num < self.proposal_topk_k:
            topk_out_num = self.boxes_num
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
        self.ub_max_topk = self.tik_instance.Tensor("float16", (self.proposal_topk_k, 8),
                                                    name="ub_max_topk", scope=tik.scope_ubuf)

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


def total_num(shape):
    """
    the return object is total num
    """
    shape_total_num = functools.reduce(lambda a, b: a * b, shape)
    return shape_total_num


def ceil_div(value, factor):
    """
    Compute the smallest integer value that is greater than
    or equal to value/factor
    """
    result = (value + (factor - 1)) // factor
    return result


# 'pylint: disable=too-many-instance-attributes,too-many-arguments
def get_input_gm_list(tik_instance, need_valid_num, valid_num_shape, need_clip_window,
                      clip_window_shape, boxes_gm, scores_gm, input_scalar_list):
    """
    to get input gm list for both CNMS and CombinedNonMaxSuppression classes
    """
    clip_window_gm = None
    valid_num_gm = None
    if need_valid_num:
        valid_num_gm = tik_instance.Tensor("int32", valid_num_shape, name="valid_num_gm", scope=tik.scope_gm)
    if need_clip_window:
        clip_window_gm = tik_instance.Tensor("float16", clip_window_shape, name="clip_window_gm", scope=tik.scope_gm)
    if need_valid_num and need_clip_window:
        input_gm_list = [boxes_gm, scores_gm, clip_window_gm, valid_num_gm]
    elif need_clip_window:
        input_gm_list = [boxes_gm, scores_gm, clip_window_gm]
    elif need_valid_num:
        input_gm_list = [boxes_gm, scores_gm, valid_num_gm]
    else:
        input_gm_list = [boxes_gm, scores_gm]

    for idx, input_scalar in enumerate(input_scalar_list):
        scalar_dtype = input_scalar.get("dtype")
        scalar_tensor = tik_instance.Tensor(scalar_dtype, [1], name="input_scalar" + str(idx), scope=tik.scope_gm)
        input_gm_list.append(scalar_tensor)

    return input_gm_list


def check_par(max_size_per_class, max_total_size, classes, boxes_type):
    """
    check_par
    """

    def _error_code_002_check(param_name, value_range, value):
        """_error_code_002_check"""
        if value < value_range[0] or value > value_range[1]:
            error_info = {
                'errCode': para_check.OP_ERROR_CODE_002,
                'param_name': param_name,
                'min_value': value_range[0],
                'max_value': value_range[1],
                'real_value': value
            }
            error_manager_vector.raise_err_specific_reson("CombinedNonMaxSuppression",
                                                          "the parameter[{param_name}] should be in"
                                                          " the range of [{min_value}, {max_value}],"
                                                          " but actually is [{real_value}].".format(**error_info))

    _error_code_002_check("max_size_per_class", [1, 1000], max_size_per_class)
    _error_code_002_check("max_total_size", [1, 1000], max_total_size)
    _error_code_002_check("classes num from input scores shape", [1, 200], classes)

    para_check.check_dtype(boxes_type, ("float16", "float32"), param_name="boxes")


def get_class_tensor(tik_instance, class_ub, class_num, len_per_class, start_class=0.0):
    """
    get class tensor
    """
    util_tik_comm_func.tik_func_vector(tik_instance, class_ub, start_class, len_per_class)
    with tik_instance.for_range(1, class_num) as _class_idx:
        dst_offset = _class_idx * len_per_class
        src_offset = (_class_idx - 1) * len_per_class
        _repeat_time = len_per_class // 128
        _repeat_tail = len_per_class % 128
        if _repeat_time != 0:
            tik_instance.vadds(128, class_ub[dst_offset], class_ub[src_offset], 1.0,
                               _repeat_time, 1, 1, 8, 8)
            dst_offset = 128 * _repeat_time + dst_offset
            src_offset = 128 * _repeat_time + src_offset
        if _repeat_tail != 0:
            tik_instance.vadds(_repeat_tail, class_ub[dst_offset], class_ub[src_offset], 1.0,
                               1, 1, 1, 8, 8)


def copy_tail_data(tik_instance, gm_dst_info, ub_src_info, gm_workspace_info, copy_len):
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
    tik_instance.data_move(gm_workspace[gm_workspace_offset],
                           ub_src[ub_src_offset + (copy_nbust_len - 1) * block_num],
                           0, 1, 2, 0, 0)
    tik_instance.data_move(ub_src[ub_src_offset], gm_workspace[gm_workspace_offset + copy_tail_offset],
                           0, 1, 1, 0, 0)
    tik_instance.data_move(gm_dst[gm_dst_offset + copy_tail_offset + (copy_nbust_len - 1) * block_num],
                           ub_src[ub_src_offset], 0, 1, 1, 0, 0)


def clip_boxes_compute(tik_instance, clip_ub, clip_value, clip_num, clip_flag=True):
    """
    clip_boxes with value
    """
    if not clip_flag:
        return
    with tik_instance.new_stmt_scope():
        clip_min_ub = tik_instance.Tensor(clip_ub.dtype, [16], name="clip_min_ub", scope=tik.scope_ubuf)
        clip_max_ub = tik_instance.Tensor(clip_ub.dtype, [16], name="clip_max_ub", scope=tik.scope_ubuf)
        util_tik_comm_func.tik_func_vector(tik_instance, clip_min_ub, clip_value[0], 16)
        util_tik_comm_func.tik_func_vector(tik_instance, clip_max_ub, clip_value[1], 16)
        util_tik_comm_func.tik_func_vcomple(tik_instance, "vmax", clip_ub, clip_ub, clip_min_ub,
                                            clip_num, 1, 1, 0, 8, 8, 0)
        util_tik_comm_func.tik_func_vcomple(tik_instance, "vmin", clip_ub, clip_ub, clip_max_ub,
                                            clip_num, 1, 1, 0, 8, 8, 0)


# 'pylint: disable=too-many-branches
def batch_multi_class_nms_copy_out(tik_instance, nms, ub_result_boxes, ub_result_boxes_class,
                                   output_batch_offset, workspace_core_offset, clip_flag=False):
    """
    batch_multi_class_nms_copy_out
    """
    clip_value = [0.0, 1.0]
    core_used = nms.aicore_num
    workspace_flag = False
    if (core_used > 1) and (nms.max_total_size % 16 != 0):
        workspace_flag = True

    workspace = nms.workspace_proposal_gm
    down_scalar = None
    if nms.down_flag:
        down_scalar = nms.down_scalar_list[1]
    loop_burst_len = ceil_div(nms.max_total_size, 16)
    apply_men_len = ceil_div(nms.max_total_size, 16)
    less_flag = False
    if nms.max_selected_nms_num_in_ub * nms.classes < nms.max_total_size:
        less_flag = True
        loop_burst_len = ceil_div(nms.max_selected_nms_num_in_ub * nms.classes, 16)
    score_thresh = nms.score_thresh
    _batch = output_batch_offset // nms.max_total_size
    ub_scores_valid_mask = tik_instance.Tensor("float16", [apply_men_len * 16],
                                               name="ub_scores_valid_mask", scope=tik.scope_ubuf)
    # process scores
    with tik_instance.new_stmt_scope():
        # scores
        ub_out_scores = tik_instance.Tensor("float16", [apply_men_len * 16],
                                            name="ub_out_scores", scope=tik.scope_ubuf)
        ub_out_scores_valid = tik_instance.Tensor("int32", [16], name="ub_out_scores_valid",
                                                  scope=tik.scope_ubuf)
        if less_flag:
            util_tik_comm_func.tik_func_vector(tik_instance, ub_out_scores, 0, apply_men_len * 16)
        util_tik_comm_func.tik_func_vextract(tik_instance, ub_result_boxes_class, ub_out_scores, loop_burst_len, 3)
        filter_score_compute(tik_instance, ub_out_scores, ub_out_scores_valid, ub_scores_valid_mask,
                             nms.max_total_size, score_thresh)
        if not workspace_flag:
            tik_instance.data_move(nms.output_gm_list[1][output_batch_offset], ub_out_scores,
                                   0, 1, apply_men_len, 0, 0)
        else:
            copy_tail_data(tik_instance,
                           [nms.output_gm_list[1], output_batch_offset],
                           [ub_out_scores, 0],
                           [workspace, workspace_core_offset],
                           nms.max_total_size)

        tik_instance.data_move(nms.output_gm_list[3][_batch * 8], ub_out_scores_valid,
                               0, 1, 1, 0, 0)
        # x1
        ub_out_box_x1 = tik_instance.Tensor("float16", [apply_men_len * 16],
                                            name="ub_out_box_x1", scope=tik.scope_ubuf)
        util_tik_comm_func.tik_func_vextract(tik_instance, ub_result_boxes, ub_out_box_x1, loop_burst_len, 0)
        util_tik_comm_func.tik_func_vcomple(tik_instance, "vmul", ub_out_box_x1, ub_scores_valid_mask, ub_out_box_x1,
                                            apply_men_len * 16)
        if nms.down_flag:
            util_tik_comm_func.tik_func_vmuls(tik_instance, ub_out_box_x1, ub_out_box_x1,
                                              down_scalar, nms.max_total_size)
        # y1
        ub_out_box_y1 = tik_instance.Tensor("float16", [apply_men_len * 16],
                                            name="ub_out_box_y1", scope=tik.scope_ubuf)
        util_tik_comm_func.tik_func_vextract(tik_instance, ub_result_boxes, ub_out_box_y1, loop_burst_len, 1)
        util_tik_comm_func.tik_func_vcomple(tik_instance, "vmul", ub_out_box_y1, ub_scores_valid_mask, ub_out_box_y1,
                                            apply_men_len * 16)
        if nms.down_flag:
            util_tik_comm_func.tik_func_vmuls(tik_instance, ub_out_box_y1, ub_out_box_y1,
                                              down_scalar, nms.max_total_size)
        clip_boxes_compute(tik_instance, ub_out_box_x1, clip_value, nms.max_total_size, clip_flag)
        clip_boxes_compute(tik_instance, ub_out_box_y1, clip_value, nms.max_total_size, clip_flag)
        tik_instance.data_move(nms.output_gm_list[0][output_batch_offset * 4], ub_out_box_x1,
                               0, 1, apply_men_len, 0, 0)
        tik_instance.data_move(nms.output_gm_list[0][output_batch_offset * 4 + nms.max_total_size],
                               ub_out_box_y1, 0, 1, apply_men_len, 0, 0)

        # x2
        ub_out_box_x2 = tik_instance.Tensor("float16", [apply_men_len * 16],
                                            name="ub_out_box_x2", scope=tik.scope_ubuf)
        util_tik_comm_func.tik_func_vextract(tik_instance, ub_result_boxes, ub_out_box_x2, loop_burst_len, 2)

        if not nms.is_need_rpn_offset:
            util_tik_comm_func.tik_func_vadds(tik_instance, ub_out_box_x2, ub_out_box_x2, 1.0, nms.max_total_size)

        if nms.down_flag:
            util_tik_comm_func.tik_func_vmuls(tik_instance, ub_out_box_x2, ub_out_box_x2,
                                              down_scalar, nms.max_total_size)
        util_tik_comm_func.tik_func_vcomple(tik_instance, "vmul", ub_out_box_x2, ub_scores_valid_mask, ub_out_box_x2,
                                            apply_men_len * 16)
        clip_boxes_compute(tik_instance, ub_out_box_x2, clip_value, nms.max_total_size, clip_flag)
        tik_instance.data_move(nms.output_gm_list[0][output_batch_offset * 4 + nms.max_total_size * 2],
                               ub_out_box_x2, 0, 1, apply_men_len, 0, 0)

        # y2
        ub_out_box_y2 = tik_instance.Tensor("float16", [apply_men_len * 16],
                                            name="ub_out_box_y2", scope=tik.scope_ubuf)
        util_tik_comm_func.tik_func_vextract(tik_instance, ub_result_boxes, ub_out_box_y2, loop_burst_len, 3)

        if not nms.is_need_rpn_offset:
            util_tik_comm_func.tik_func_vadds(tik_instance, ub_out_box_y2, ub_out_box_y2, 1.0, nms.max_total_size)

        if nms.down_flag:
            util_tik_comm_func.tik_func_vmuls(tik_instance, ub_out_box_y2, ub_out_box_y2,
                                              down_scalar, nms.max_total_size)
        util_tik_comm_func.tik_func_vcomple(tik_instance, "vmul", ub_out_box_y2, ub_scores_valid_mask, ub_out_box_y2,
                                            apply_men_len * 16)
        clip_boxes_compute(tik_instance, ub_out_box_y2, clip_value, nms.max_total_size, clip_flag)
        if not workspace_flag:
            tik_instance.data_move(nms.output_gm_list[0][output_batch_offset * 4 + nms.max_total_size * 3],
                                   ub_out_box_y2, 0, 1, apply_men_len, 0, 0)
        else:
            copy_tail_data(tik_instance,
                           [nms.output_gm_list[0], output_batch_offset * 4 + nms.max_total_size * 3],
                           [ub_out_box_y2, 0],
                           [workspace, workspace_core_offset],
                           nms.max_total_size)
        # class
        ub_out_class = tik_instance.Tensor("float16", [apply_men_len * 16],
                                           name="ub_out_class", scope=tik.scope_ubuf)
        util_tik_comm_func.tik_func_vextract(tik_instance, ub_result_boxes_class, ub_out_class, loop_burst_len, 0)
        if not workspace_flag:
            tik_instance.data_move(nms.output_gm_list[2][output_batch_offset], ub_out_class,
                                   0, 1, apply_men_len, 0, 0)
        else:
            copy_tail_data(tik_instance,
                           [nms.output_gm_list[2], output_batch_offset],
                           [ub_out_class, 0],
                           [workspace, workspace_core_offset],
                           nms.max_total_size)


def batch_multi_class_nms_output(tik_instance, core_idx, _batch_idx, nms, clip_flag):
    """
    do batch_multi_class_nms_output

    Parameters:
    ----------
    tik_instance : tik_instance.
    _batch_idx : int.
        the process batch
    nms : class.
        all par for nms
    clip_flag: bool:
        whether clip the boxes by value (0, 1)
    Returns
    -------
    None
    """
    result_total = total_num(nms.l1_nms_result.shape[1:])
    class_num = nms.classes
    # get score batch offset
    output_batch_offset = _batch_idx * nms.max_total_size
    workspace = nms.workspace_proposal_gm
    workspace_offset = core_idx * nms.workspace_proposal_gm.shape[-1]
    if nms.classes * nms.max_selected_nms_num_in_ub < nms.proposal_topk_k:
        # when all output is less nms.proposal_topk_k
        # only use topk with ub for output proposal
        with tik_instance.new_stmt_scope():
            ub_result_boxes = tik_instance.Tensor("float16", [result_total // 8, 8],
                                                  name="ub_result_boxes", scope=tik.scope_ubuf)
            ub_result_boxes_class = tik_instance.Tensor("float16", [result_total // 8, 8],
                                                        name="ub_result_boxes_class", scope=tik.scope_ubuf)
            l1_buffer = nms.l1_nms_result
            l1_offset = [nms.get_l1_core_idx(core_idx), 0, 0, 0]
            loop_burst_len = result_total // 16
            tik_instance.data_move(ub_result_boxes, l1_buffer[l1_offset],
                                   0, 1, loop_burst_len, 0, 0)
            tik_instance.data_move(ub_result_boxes_class, l1_buffer[l1_offset],
                                   0, 1, loop_burst_len, 0, 0)
            with tik_instance.new_stmt_scope():
                ub_class_all = tik_instance.Tensor("float16", [nms.max_selected_nms_num_in_ub * nms.classes],
                                                   name="ub_class_all", scope=tik.scope_ubuf)
                get_class_tensor(tik_instance, ub_class_all, class_num, nms.max_selected_nms_num_in_ub)

                trans_repeat = ceil_div(nms.max_selected_nms_num_in_ub * nms.classes, 16)
                util_tik_comm_func.tik_func_vconcat(tik_instance, ub_result_boxes_class, ub_class_all, trans_repeat, 1)
                tik_instance.data_move(workspace[workspace_offset], ub_result_boxes_class,
                                       0, 1, loop_burst_len, 0, 0)
                tik_instance.data_move(ub_result_boxes_class, workspace[workspace_offset + 1],
                                       0, 1, loop_burst_len, 0, 0)
                util_tik_comm_func.tik_func_vextract(tik_instance, ub_result_boxes_class,
                                                     ub_class_all, trans_repeat, 3)
                util_tik_comm_func.tik_func_vconcat(tik_instance, ub_result_boxes_class, ub_class_all, trans_repeat, 4)

            if nms.classes != 1:
                sort_within_ub(tik_instance, ub_result_boxes_class, result_total // 8)
                sort_within_ub(tik_instance, ub_result_boxes, result_total // 8)

            with tik_instance.new_stmt_scope():
                batch_multi_class_nms_copy_out(tik_instance, nms, ub_result_boxes,
                                               ub_result_boxes_class, output_batch_offset,
                                               workspace_offset, clip_flag)
    else:
        l1_buffer = nms.l1_nms_result
        copy_classes_num = nms.proposal_topk_k // nms.max_selected_nms_num_in_ub // 2
        copy_loop = nms.classes // copy_classes_num
        copy_tail = nms.classes % copy_classes_num
        tmp_output_proposal_num = ceil_div(nms.max_total_size, 16) * 16
        ub_out_result = tik_instance.Tensor("float16", [tmp_output_proposal_num, 8],
                                            name="ub_out_result", scope=tik.scope_ubuf)
        ub_out_result_class = tik_instance.Tensor("float16", [tmp_output_proposal_num, 8],
                                                  name="ub_out_result_class", scope=tik.scope_ubuf)
        util_tik_comm_func.tik_func_vector(tik_instance, ub_out_result, 0.0, tmp_output_proposal_num * 8)
        util_tik_comm_func.tik_func_vector(tik_instance, ub_out_result_class, 0.0, tmp_output_proposal_num * 8)
        with tik_instance.new_stmt_scope():
            ub_result_boxes = tik_instance.Tensor("float16", [copy_classes_num * nms.max_selected_nms_num_in_ub, 8],
                                                  name="ub_result_boxes", scope=tik.scope_ubuf)
            ub_result_boxes_class = tik_instance.Tensor("float16", [copy_classes_num * nms.max_selected_nms_num_in_ub,
                                                                    8],
                                                        name="ub_result_boxes_class", scope=tik.scope_ubuf)
            ub_class_all = tik_instance.Tensor("float16", [nms.max_selected_nms_num_in_ub * copy_classes_num],
                                               name="ub_class_all", scope=tik.scope_ubuf)
            get_class_tensor(tik_instance, ub_class_all, copy_classes_num,
                             nms.max_selected_nms_num_in_ub, copy_classes_num * -1)

            def _do_copy_and_vconcat_class(_l1_offset, _loop_burst_len):
                tik_instance.data_move(ub_result_boxes, l1_buffer[_l1_offset],
                                       0, 1, _loop_burst_len, 0, 0)
                tik_instance.data_move(ub_result_boxes_class, l1_buffer[_l1_offset],
                                       0, 1, _loop_burst_len, 0, 0)
                # get copy_classes_num sort
                util_tik_comm_func.tik_func_vadds(tik_instance, ub_class_all, ub_class_all, copy_classes_num * 1.0,
                                                  nms.max_selected_nms_num_in_ub * copy_classes_num)
                _trans_repeat = ceil_div(nms.max_selected_nms_num_in_ub * copy_classes_num, 16)
                util_tik_comm_func.tik_func_vconcat(tik_instance, ub_result_boxes_class,
                                                    ub_class_all, _trans_repeat, 1)
                tik_instance.data_move(workspace[workspace_offset], ub_result_boxes_class,
                                       0, 1, _loop_burst_len, 0, 0)
                tik_instance.data_move(ub_result_boxes_class, workspace[workspace_offset + 1],
                                       0, 1, _loop_burst_len, 0, 0)
                with tik_instance.new_stmt_scope():
                    ub_class_tmp = tik_instance.Tensor("float16", [nms.max_selected_nms_num_in_ub * copy_classes_num],
                                                       name="ub_class_tmp", scope=tik.scope_ubuf)
                    util_tik_comm_func.tik_func_vextract(tik_instance, ub_result_boxes_class,
                                                         ub_class_tmp, _trans_repeat, 3)
                    util_tik_comm_func.tik_func_vconcat(tik_instance, ub_result_boxes_class,
                                                        ub_class_tmp, _trans_repeat, 4)

            with tik_instance.for_range(0, copy_loop) as _class_idx:
                l1_offset = [nms.get_l1_core_idx(core_idx), _class_idx * copy_classes_num, 0, 0]
                loop_burst_len = copy_classes_num * nms.max_selected_nms_num_in_ub * 8 // 16
                _do_copy_and_vconcat_class(l1_offset, loop_burst_len)
                sort_within_ub(tik_instance, ub_result_boxes, copy_classes_num * nms.max_selected_nms_num_in_ub)
                sort_within_ub(tik_instance, ub_result_boxes_class, copy_classes_num * nms.max_selected_nms_num_in_ub)
                tik_func_sort_with_ub(tik_instance, [ub_out_result, ub_result_boxes],
                                      [ub_out_result, ub_result_boxes], tmp_output_proposal_num)
                tik_func_sort_with_ub(tik_instance, [ub_out_result_class, ub_result_boxes_class],
                                      [ub_out_result_class, ub_result_boxes_class], tmp_output_proposal_num)

            if copy_tail != 0:
                l1_offset = [nms.get_l1_core_idx(core_idx), copy_loop * copy_classes_num, 0, 0]
                loop_burst_len = copy_tail * nms.max_selected_nms_num_in_ub * 8 // 16
                _do_copy_and_vconcat_class(l1_offset, loop_burst_len)
                sort_within_ub(tik_instance, ub_result_boxes, copy_tail * nms.max_selected_nms_num_in_ub)
                sort_within_ub(tik_instance, ub_result_boxes_class, copy_tail * nms.max_selected_nms_num_in_ub)
                if copy_tail * nms.max_selected_nms_num_in_ub < tmp_output_proposal_num:
                    dup_len = tmp_output_proposal_num - copy_tail * nms.max_selected_nms_num_in_ub
                    dup_offset = copy_tail * nms.max_selected_nms_num_in_ub
                    util_tik_comm_func.tik_func_vector(tik_instance, ub_result_boxes[dup_offset:], 0.0, dup_len * 8)
                    util_tik_comm_func.tik_func_vector(tik_instance, ub_result_boxes_class[dup_offset:],
                                                       0.0, dup_len * 8)
                tik_func_sort_with_ub(tik_instance, [ub_out_result, ub_result_boxes],
                                      [ub_out_result, ub_result_boxes], tmp_output_proposal_num)
                tik_func_sort_with_ub(tik_instance, [ub_out_result_class, ub_result_boxes_class],
                                      [ub_out_result_class, ub_result_boxes_class], tmp_output_proposal_num)
        with tik_instance.new_stmt_scope():
            batch_multi_class_nms_copy_out(tik_instance, nms, ub_out_result, ub_out_result_class,
                                           output_batch_offset, workspace_offset, clip_flag)


# 'pylint: disable=unused-argument
def check_supported(boxes, scores, max_output_size_per_class,
                    max_total_size, iou_threshold, score_threshold,
                    nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections,
                    pad_per_class, clip_boxes,
                    kernel_name="combined_non_max_suppression"):
    """
    check_supported: check whether the aicore support this case

    if the valid_detections_shape shape len = 2, do in aicore
    """
    valid_detections_shape = valid_detections.get("ori_shape")

    if len(valid_detections_shape) == 2:
        return True, ""
    reason = "if the valid_detections_shape shape len != 2, not supported by aicore"
    return False, reason


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_BOOL, para_check.KERNEL_NAME)
def combined_non_max_suppression(boxes, scores, max_output_size_per_class,
                                 max_total_size, iou_threshold, score_threshold,
                                 nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections,
                                 pad_per_class, clip_boxes,
                                 kernel_name="combined_non_max_suppression",
                                 impl_mode="high_performance"):
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
        shape, dtype of boxes, a 4D Tensor of type float16 with shape (batch, num_classes, num_anchors, 4).
        "batch" indicates the batch size of image,
        and "num_anchors" indicates num of boxes, and "num_classes" indicates classes of detect.
        and the value "4" refers to "x0", "y0", "x1", and "y1".
    scores : dict.
        shape, dtype of scores
        a 3D Tensor of type float16 with shape (batch, num_anchors, num_classes).
    max_output_size_per_class : dict.
        A required scalar of type int, specifying the nms output num per class.
    max_total_size : dict.
        A required scalar of type int, specifying the the nms output num per batch.
    iou_threshold : dict.
        A required scalar of type float32, specifying the nms iou iou_threshold
    score_threshold : dict.
        A required scalar of type float32, specifying the score filter iou iou_threshold.
    nmsed_boxes : dict.
        A 3D Tensor of type float16 with shape (batch, max_total_size, 4).
        specifying the output nms boxes per batch
    nmsed_scores : dict.
        A 3D Tensor of type float16 with shape (batch, max_total_size).
        specifying the output nms score per batch
    nmsed_classes : dict.
        A 3D Tensor of type float16 with shape (batch, max_total_size).
        specifying the output nms class per batch
    valid_detections : dict.
        A 1D Tensor of type int32 with shape (batch,),
        specifying the valid num of nmsed_boxes
    pad_per_class : bool.
        A required attribute of type bool, whether to pad result to max_total_size.
    clip_boxes : bool.
        A required attribute of type bool, whether clip the output boxes by [0, 1]
    kernel_name : str.
        cce kernel name, default value is "combined_non_max_suppression"
    impl_mode: str.
        high_precision or high_performance for inference, default value is "high_performance".
        no need to add into ops_info file.

    Returns
    -------
    tik_instance
    """
    check_op_impl_mode(impl_mode, [OpImplMode.HIGH_PERFORMANCE, OpImplMode.HIGH_PRECISION], kernel_name)

    # new branch compute flow
    if tbe_platform.api_check_support("tik.vcopy"):
        obj = CNMS(boxes, scores,
                   [max_output_size_per_class, max_total_size,
                    iou_threshold, score_threshold],
                   score_threshold.get("const_value")[0],
                   iou_threshold.get("const_value")[0],
                   max_output_size_per_class.get("const_value")[0],
                   max_total_size.get("const_value")[0],
                   kernel_name)
        return obj.cnms_compute()

    nms = CombinedNonMaxSuppression(boxes, scores,
                                    [max_output_size_per_class, max_total_size,
                                     iou_threshold, score_threshold],
                                    score_threshold.get("const_value")[0],
                                    iou_threshold.get("const_value")[0],
                                    max_output_size_per_class.get("const_value")[0],
                                    max_total_size.get("const_value")[0], impl_mode)
    # init ub
    core_used, batch_per_core, batch_last_core = nms.get_core_schedule()
    class_num = nms.classes
    nms.init_tik_mem()
    tik_instance = nms.get_tik_instance()

    def _run_one_core(_real_batch_idx, _real_core_idx):
        with tik_instance.for_range(0, class_num) as _class_idx:
            nms.selected_proposals_cnt.set_as(0)
            with tik_instance.new_stmt_scope():
                nms_for_single_class(_real_batch_idx, _class_idx, nms, _real_core_idx)

        # process all class output result is in l1_nms_result, will process output
        # step 1 sort all select proposal with boxes
        # step 2 sort all select proposal with classes score
        with tik_instance.new_stmt_scope():
            batch_multi_class_nms_output(tik_instance, _real_core_idx, _real_batch_idx, nms, clip_boxes)

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


class CNMS:
    """
    a brand new compute flow, temporarily support fp16
    step 1- initialize, get core_used by batches
    step 2- sorted the scores and get the corresponding index
    step 3- select the indexes based on the score_thresh with dichotomous ordering
    step 4- according to the selected indexes, move the top 4096 scores boxes for iou selection
    step 5- do nms for each class in each batch use top 4094 proposals
    step 6- sorted the scores of every batches to get the top max_size boxes
    step 7- move the data out to gm
    """

    # 'pylint: disable=too-many-arguments
    def __init__(self,
                 boxes,
                 scores,
                 input_scalar_list,
                 score_thresh,
                 iou_thresh,
                 max_size_per_class,
                 max_total_size,
                 kernel_name):
        """
        Init CombinedNonMaxSuppression base parameters

        Returns
        -------
        None
        """
        box_shape = list(boxes.get("shape"))
        self.box_dtype = boxes.get("dtype")
        score_shape = list(scores.get("shape"))
        # when input have no class dim, will extend 1 for input shape
        if len(score_shape) == 2 and len(box_shape) == 3:
            self.boxes_shape = [box_shape[0], 1, box_shape[1], box_shape[2]]
            self.scores_shape = [score_shape[0], 1, score_shape[1]]
        else:
            self.boxes_shape = box_shape
            self.scores_shape = score_shape

        self.input_scalar_list = input_scalar_list
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh / (1 + iou_thresh)
        self.max_size_per_class = max_size_per_class
        self.max_total_size = max_total_size
        self.kernel_name = kernel_name

        self.need_valid_num = False
        self.valid_num_shape = None
        self.need_clip_window = False
        self.clip_window_shape = None
        self.change_coordinate_frame = False

        para_check.check_shape(self.boxes_shape, min_rank=4, max_rank=4, param_name="boxes")
        para_check.check_shape(self.scores_shape, min_rank=3, max_rank=3, param_name="scores")
        # parsing input
        self.batch, self.classes, self.boxes_num = self.scores_shape
        _, self.boxes_classes, _, _ = self.boxes_shape
        if self.classes == self.boxes_classes and self.boxes_classes == 1:
            if self.max_size_per_class > self.max_total_size:
                self.max_size_per_class = self.max_total_size
        check_par(self.max_size_per_class, self.max_total_size, self.classes, self.box_dtype)
        # whether down the boxes to avoid fp16 overflow
        self.down_flag = False

        # tiling attrs
        self.tik_instance = tik.Tik()
        self.aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)

        self.core_used_b = 1
        self.batch_per_core = 1
        self.batch_last_core = 1

        self.loop_num = 1
        self.last_loop_size = 0
        self._init_loop_uint = Constant.PER_LOOP_UNIT
        self._unit_ele = Constant.UNIT_ELE

        dtype_size = common_util.get_data_size(self.box_dtype)
        self._block_ele = constant_util.BLOCK_SIZE // dtype_size

        self.input_gm_list = []
        self.output_gm_list = []

        # calcu output shape
        self.nmsed_boxes_shape = [self.batch, Constant.FOUR_DIRECTION, self.max_total_size]
        self.nmsed_scores_shape = [self.batch, self.max_total_size]
        self.nmsed_classes_shape = [self.batch, self.max_total_size]
        self.nmsed_num_shape = [self.batch, Constant.BLOCK_ELE_B32]
        self.level = 1
        self.fp_min = -65504
        self.pattern_type = Constant.PATTERN_TYPE

        if self.box_dtype == "float32":
            self._init_loop_uint = Constant.PER_LOOP_UNIT // 2
            self._unit_ele = Constant.UNIT_ELE_FP32
            self.level = 0
            self.fp_min = -3.4e+38
            self.pattern_type = Constant.PATTERN_TYPE_FP32

        self._per_loop_uint = self._init_loop_uint
        # optimize the compute data lens
        if self.boxes_num < self._init_loop_uint // Constant.FOUR_PIECE:
            self._per_loop_uint = self._init_loop_uint // Constant.FOUR_PIECE
        elif self.boxes_num < self._init_loop_uint // Constant.TWO_PIECE:
            self._per_loop_uint = self._init_loop_uint // Constant.TWO_PIECE


        idx_size = ceil_div(self.boxes_num, self._per_loop_uint) * self._per_loop_uint
        idx_init = [i for i in range(idx_size)]

        self.idx_gm = self.tik_instance.Tensor("uint32",
                                               [idx_size, ],
                                               name="idx_gm",
                                               scope=tik.scope_gm, init_value=idx_init)

        # workspace
        self.workspace_score_idx = self.tik_instance.Tensor(self.box_dtype,
                                                            [self.batch,
                                                             self.boxes_classes,
                                                             ceil_div(self.boxes_num * 4, 32) * 32],
                                                            name="workspace_score_idx",
                                                            scope=tik.scope_gm, is_workspace=True)
        self.workspace_boxes = self.tik_instance.Tensor(self.box_dtype,
                                                        [self.batch,
                                                         self.boxes_classes, 4, self.max_size_per_class * 2],
                                                        name="workspace_boxes",
                                                        scope=tik.scope_gm, is_workspace=True)

        self.workspace_scores = self.tik_instance.Tensor(self.box_dtype,
                                                         [self.batch,
                                                          self.boxes_classes, self.max_size_per_class * 2],
                                                         name="workspace_scores",
                                                         scope=tik.scope_gm, is_workspace=True)

    def get_core_schedule(self):
        """
        get_core_schedule for batch sorting compute
        """
        if self.max_total_size < 16:
            self.aicore_num = 1
        self.batch_per_core = ceil_div(self.batch, self.aicore_num)
        self.core_used_b = ceil_div(self.batch, self.batch_per_core)
        self.batch_last_core = self.batch - (self.core_used_b - 1) * self.batch_per_core

    def cnms_compute(self):
        """
        compute of combined_non_max_suppression

        Parameters
        ----------
        None

        Returns
        -------
        compile info
        """
        # init gm input
        boxes_gm = self.tik_instance.Tensor(self.box_dtype, self.boxes_shape, name="boxes_gm", scope=tik.scope_gm)
        scores_gm = self.tik_instance.Tensor(self.box_dtype, self.scores_shape, name="scores_gm", scope=tik.scope_gm)

        # init gm output
        nmsed_boxes_gm = self.tik_instance.Tensor(self.box_dtype, self.nmsed_boxes_shape,
                                                  name="nmsed_boxes_gm", scope=tik.scope_gm)
        nmsed_scores_gm = self.tik_instance.Tensor(self.box_dtype, self.nmsed_scores_shape,
                                                   name="nmsed_scores_gm", scope=tik.scope_gm)
        nmsed_classes_gm = self.tik_instance.Tensor(self.box_dtype, self.nmsed_classes_shape,
                                                    name="nmsed_classes_gm", scope=tik.scope_gm)
        nmsed_num_gm = self.tik_instance.Tensor("int32", self.nmsed_num_shape,
                                                name="nmsed_num_gm", scope=tik.scope_gm)
        self.output_gm_list = [nmsed_boxes_gm, nmsed_scores_gm, nmsed_classes_gm, nmsed_num_gm]

        self.input_gm_list = get_input_gm_list(self.tik_instance, self.need_valid_num, self.valid_num_shape,
                                               self.need_clip_window, self.clip_window_shape, boxes_gm,
                                               scores_gm, self.input_scalar_list)

        self.class_nms_compute(boxes_gm, scores_gm)

        opt_config = {
            "enable_const_fold": True
        }

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=self.input_gm_list,
                                   outputs=self.output_gm_list,
                                   output_files_path=None,
                                   enable_l2=False, config=opt_config)
        return self.tik_instance

    def class_nms_compute(self, boxes, scores):
        """
        main compute cycle

        Parameters
        ----------
        boxes : tensor
            input location data in gm
        scores : tensor
            input scores data in gm

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        self.get_core_schedule()
        # do nms with multi cores
        with tik_instance.for_range(0, self.core_used_b, block_num=self.core_used_b) as core_idx:
            if self.core_used_b == 1 or self.batch_per_core == self.batch_last_core:
                with tik_instance.for_range(0, self.batch_last_core) as batch_idx:
                    real_batch_idx = core_idx * self.batch_last_core + batch_idx
                    with tik_instance.for_range(0, self.classes) as class_idx:
                        with tik_instance.new_stmt_scope():
                            self.process_nms_mode(real_batch_idx, class_idx, boxes, scores)
                    with tik_instance.if_scope(self.classes != 1):
                        self.sort_class_per_batch(real_batch_idx)
            else:
                with tik_instance.if_scope(core_idx < self.core_used_b - 1):
                    with tik_instance.for_range(0, self.batch_per_core) as batch_idx:
                        real_batch_idx = core_idx * self.batch_per_core + batch_idx
                        with tik_instance.for_range(0, self.classes) as class_idx:
                            with tik_instance.new_stmt_scope():
                                self.process_nms_mode(real_batch_idx, class_idx, boxes, scores)
                        with tik_instance.if_scope(self.classes != 1):
                            self.sort_class_per_batch(real_batch_idx)

                with tik_instance.else_scope():
                    with tik_instance.for_range(0, self.batch_last_core) as batch_idx:
                        real_batch_idx = core_idx * self.batch_per_core + batch_idx
                        with tik_instance.for_range(0, self.classes) as class_idx:
                            with tik_instance.new_stmt_scope():
                                self.process_nms_mode(real_batch_idx, class_idx, boxes, scores)
                        with tik_instance.if_scope(self.classes != 1):
                            self.sort_class_per_batch(real_batch_idx)

    def process_nms_mode(self, real_batch_idx, class_idx, boxes, scores):
        """
        deal with class nms compute

        Parameters
        ----------
        real_batch_idx : int
            batch index
        class_idx : int
            class index
        boxes : tensor
            input location data in gm
        scores: tensor
            input scores data in gm

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance

        # temp set as 4096, can be optimized later
        shape_aligned = self._per_loop_uint
        eff_size = tik_instance.Scalar(dtype="uint32", name="eff_size")
        eff_lens = tik_instance.Scalar(dtype="uint32", name="eff_lens", init_value=self._per_loop_uint)
        pre_eff_lens = tik_instance.Scalar(dtype="uint32", name="pre_eff_lens", init_value=0)
        x1_ub = tik_instance.Tensor(self.box_dtype, [shape_aligned, ], name="x1_ub", scope=tik.scope_ubuf)
        x2_ub = tik_instance.Tensor(self.box_dtype, [shape_aligned, ], name="x2_ub", scope=tik.scope_ubuf)
        y1_ub = tik_instance.Tensor(self.box_dtype, [shape_aligned, ], name="y1_ub", scope=tik.scope_ubuf)
        y2_ub = tik_instance.Tensor(self.box_dtype, [shape_aligned, ], name="y2_ub", scope=tik.scope_ubuf)
        scores_ub = tik_instance.Tensor(self.box_dtype, [shape_aligned, ], name="scores_ub", scope=tik.scope_ubuf)
        
        with tik_instance.new_stmt_scope():
            scores_idx_out = tik_instance.Tensor(self.box_dtype, [self._per_loop_uint * self._unit_ele * 2, ], 
                                                    name="scores_idx_out", scope=tik.scope_ubuf)
            # first round, process top 4096 or 2048 units
            self.gen_score_index(real_batch_idx, class_idx, scores, scores_idx_out)
            self.get_eff_size_by_threshold(scores_idx_out, eff_size, shape_aligned, gate_value=self.score_thresh)
            with tik_instance.if_scope(eff_size > self.boxes_num):
                eff_size.set_as(self.boxes_num)

            self.get_boxes_after_score_thresh(x1_ub, x2_ub, y1_ub, y2_ub, scores_ub, real_batch_idx, class_idx, boxes,
                                            scores_idx_out, eff_size)
            self.iou_selection(x1_ub, x2_ub, y1_ub, y2_ub, scores_ub, eff_lens)
            pre_eff_lens.set_as(eff_lens)

        with tik_instance.if_scope(
                tik.any(eff_lens >= self.max_size_per_class, self._per_loop_uint >= self.boxes_num)):
            if self.classes == 1:
                self.sort_single_class_per_batch(real_batch_idx, eff_lens, x1_ub, x2_ub, y1_ub, y2_ub, scores_ub)
            else:
                self.store_data(x1_ub, x2_ub, y1_ub, y2_ub, scores_ub, real_batch_idx, class_idx)

        with tik_instance.else_scope():
            # to process the second round
            with tik_instance.new_stmt_scope():
                scores_idx_out = tik_instance.Tensor(self.box_dtype, [self._per_loop_uint * self._unit_ele * 2, ], 
                                        name="scores_idx_out", scope=tik.scope_ubuf)
                self.sort_second_round_data(real_batch_idx, class_idx, scores_idx_out)
                self.get_eff_size_by_threshold(scores_idx_out, eff_size, shape_aligned, gate_value=self.score_thresh)

                with tik_instance.if_scope(tik.all(self.boxes_num > self._per_loop_uint,
                                                   eff_size > (self.boxes_num - self._per_loop_uint))):
                    eff_size.set_as(self.boxes_num - self._per_loop_uint)

                eff_lens.set_as(self._per_loop_uint)
                with tik_instance.if_scope(eff_size > 0):
                    self.get_boxes_after_score_thresh(x1_ub, x2_ub, y1_ub, y2_ub, scores_ub, real_batch_idx, class_idx,
                                                      boxes, scores_idx_out, eff_size, pre_eff_lens)

                    self.iou_selection(x1_ub, x2_ub, y1_ub, y2_ub, scores_ub, eff_lens)
            if self.classes == 1:
                self.sort_single_class_per_batch(real_batch_idx, eff_lens, x1_ub, x2_ub, y1_ub, y2_ub, scores_ub)
            else:
                self.store_data(x1_ub, x2_ub, y1_ub, y2_ub, scores_ub, real_batch_idx, class_idx)

    def sort_second_round_data(self, real_batch_idx, class_idx, dst):
        """
        when second round, get top 4096 scores and index from workspace

        Parameters
        ----------
        real_batch_idx : int
            batch index
        class_idx : int
            class index
        dst : tensor
            scores_idx_out

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        per_loop_ele = self._per_loop_uint
        score_idx_lens = per_loop_ele * self._unit_ele
        # as the first cycle takes 4096 units
        left_size = self.boxes_num - self._per_loop_uint
        loop_num = left_size // per_loop_ele
        tail = left_size - loop_num * per_loop_ele

        self.init_tensor(dst, score_idx_lens * 2, self.fp_min)

        with tik_instance.new_stmt_scope():
            # define the tmp tensor, as 32 bytes aligned required
            scores_idx_ub = tik_instance.Tensor(self.box_dtype, [score_idx_lens * 2, ],
                                                name="scores_idx_ub", scope=tik.scope_ubuf)
            if loop_num > 0:
                # the first 4096 units
                self.init_tensor(scores_idx_ub, score_idx_lens * 2)
                burst_lens_base = score_idx_lens // self._block_ele
                tik_instance.data_move(dst, self.workspace_score_idx[real_batch_idx, class_idx, 0], 0, 1,
                                       burst_lens_base, 0, 0)

                with tik_instance.for_range(1, loop_num) as loop_idx:
                    # set value for index
                    tik_instance.data_move(scores_idx_ub,
                                           self.workspace_score_idx[
                                               real_batch_idx, class_idx, score_idx_lens * loop_idx],
                                           0, 1, burst_lens_base, 0, 0)
                    tik_instance.data_move(scores_idx_ub[score_idx_lens], dst, 0, 1, burst_lens_base, 0, 0)
                    sort_score_idx_by_desc(tik_instance, scores_idx_ub, dst,
                                              score_idx_lens * 2, self.level, dtype=self.box_dtype)

                with tik_instance.if_scope(tail > 0):
                    # init scores_ub & scores_idx_ub in order to clear the pre data
                    self.init_tensor(scores_idx_ub, score_idx_lens)

                    burst_lens = (tail * self._unit_ele) // self._block_ele
                    tail_left = (tail * self._unit_ele) % self._block_ele
                    if burst_lens > 0:
                        tik_instance.data_move(scores_idx_ub, self.workspace_score_idx[
                            real_batch_idx, class_idx, left_size - tail], 0, 1, burst_lens, 0, 0)
                    with tik_instance.for_range(0, tail_left) as _idx:
                        scores_idx_ub[burst_lens * self._block_ele + _idx].set_as(
                            self.workspace_score_idx[real_batch_idx, class_idx, left_size - tail_left + _idx])

                    tik_instance.data_move(scores_idx_ub[score_idx_lens], dst, 0, 1, burst_lens_base, 0, 0)
                    self.init_tensor(dst, score_idx_lens * 2, self.fp_min)
                    sort_score_idx_by_desc(tik_instance, scores_idx_ub, dst,
                                              score_idx_lens * 2, self.level, dtype=self.box_dtype)

            else:
                self.init_tensor(scores_idx_ub, score_idx_lens * 2)
                burst_lens = left_size * self._unit_ele // self._block_ele
                tail_left = left_size * self._unit_ele % self._block_ele
                if burst_lens > 0:
                    tik_instance.data_move(scores_idx_ub, self.workspace_score_idx[real_batch_idx, class_idx, 0], 0, 1,
                                           burst_lens, 0, 0)
                with tik_instance.for_range(0, tail_left) as _idx:
                    scores_idx_ub[burst_lens * self._block_ele + _idx].set_as(
                        self.workspace_score_idx[real_batch_idx, class_idx, left_size - tail_left + _idx])
                sort_score_idx_by_desc(tik_instance, scores_idx_ub, dst, score_idx_lens, self.level, 
                                       dtype=self.box_dtype)

    def init_tensor(self, src, size, init_value=0):
        """
        initialize the input tensor, set as init value

        Parameters
        ----------
        src: tensor
            target tensor in ub
        size: int
            data size, default as 4096
        init_value: int
            initial value

        Returns
        -------
        None
        """
        vector_mask, rep_stride = get_mask_rep_stride(src)

        max_lens = Constant.REPEAT_TIMES_MAX * vector_mask
        loop_num = size // max_lens
        tail = size % max_lens
        repeat_times = tail // vector_mask
        tail_aligned = tail % vector_mask

        tik_instance = self.tik_instance

        off = tik_instance.Scalar("uint32")
        with tik_instance.for_range(0, loop_num) as idx:
            off.set_as(vector_mask * Constant.REPEAT_TIMES_MAX * idx)
            tik_instance.vec_dup(vector_mask, src[off], init_value, Constant.REPEAT_TIMES_MAX, rep_stride)
        if tail != 0 and repeat_times > 0:
            offset = size - tail
            tik_instance.vec_dup(vector_mask, src[offset], init_value, repeat_times, rep_stride)
        if tail_aligned != 0:
            with tik_instance.for_range(0, tail_aligned) as i:
                src[size - tail_aligned + i].set_as(init_value)

    def gen_score_index(self, batch_idx, class_idx, score_gm, scores_idx_out):
        """
        construct the tensor(score_index) for vsort32 and vmrgsort command
        get top 4096 scores and index, others stored in workspace

        Parameters
        ----------
        batch_idx : int
            batch index
        class_idx : int
            class index
        score_gm : tensor
            input scores data in gm
        scores_idx_out : tensor
            scores_idx_out

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance

        per_loop_ele = self._per_loop_uint
        score_idx_lens = per_loop_ele * self._unit_ele
        burst_lens_idx = score_idx_lens // self._block_ele

        loop_num = self.boxes_num // per_loop_ele
        tail = self.boxes_num - loop_num * per_loop_ele
        # repeat_times for every vsort32 cycle
        repeat_times = per_loop_ele // Constant.REPEAT_ELE

        self.init_tensor(scores_idx_out, score_idx_lens * 2, self.fp_min)
        with tik_instance.new_stmt_scope():
            # define the tmp tensor, as 32 bytes aligned required
            index = tik_instance.Tensor("uint32", [per_loop_ele, ], name="idx_ub", scope=tik.scope_ubuf)
            init_index(tik_instance, self.idx_gm, index, 0, per_loop_ele, dtype=self.box_dtype)
            scores_ub = tik_instance.Tensor(self.box_dtype, [per_loop_ele, ], name="scores_ub", scope=tik.scope_ubuf)
            scores_idx_ub = tik_instance.Tensor(self.box_dtype, [score_idx_lens * 2, ],
                                                name="scores_idx_ub", scope=tik.scope_ubuf)

            if loop_num > 0:
                # the first 4096 units
                burst_lens_base = per_loop_ele // self._block_ele
                tik_instance.data_move(scores_ub, score_gm[batch_idx, class_idx, 0], 0, 1, burst_lens_base, 0, 0)
                tik_instance.vsort32(scores_idx_out, scores_ub, index, repeat_times)

                with tik_instance.for_range(1, loop_num) as loop_idx:
                    # set value for index
                    init_index(tik_instance, self.idx_gm, index, loop_idx * per_loop_ele, per_loop_ele,
                                     dtype=self.box_dtype)

                    gm2ub_for_vsort32(tik_instance, score_gm, [batch_idx, class_idx, per_loop_ele * loop_idx],
                                      scores_ub, per_loop_ele)

                    tik_instance.vsort32(scores_idx_ub, scores_ub, index, repeat_times)
                    tik_instance.data_move(scores_idx_ub[score_idx_lens], scores_idx_out, 0, 1, burst_lens_idx, 0, 0)
                    sort_score_idx_by_desc(tik_instance, scores_idx_ub, scores_idx_out, score_idx_lens * 2, 
                                            self.level, dtype=self.box_dtype)

                    # move last 4096 scores_index uints to workspace
                    tik_instance.data_move(
                        self.workspace_score_idx[batch_idx, class_idx, score_idx_lens * (loop_idx - 1)],
                        scores_idx_out[score_idx_lens], 0, 1, burst_lens_idx, 0, 0)

                with tik_instance.if_scope(tail > 0):
                    init_index(tik_instance, self.idx_gm, index, loop_num * per_loop_ele, per_loop_ele,
                                     dtype=self.box_dtype)
                    # init scores_ub & scores_idx_ub in order to clear the pre data
                    self.init_tensor(scores_ub, per_loop_ele, self.fp_min)
                    self.init_tensor(scores_idx_ub, score_idx_lens * 2)

                    gm2ub_for_vsort32(tik_instance, score_gm, [batch_idx, class_idx, self.boxes_num - tail], scores_ub,
                                      tail)

                    tik_instance.vsort32(scores_idx_ub, scores_ub, index, repeat_times)
                    tik_instance.data_move(scores_idx_ub[score_idx_lens], scores_idx_out, 0, 1, burst_lens_idx, 0, 0)
                    self.init_tensor(scores_idx_out, score_idx_lens * 2, self.fp_min)
                    sort_score_idx_by_desc(tik_instance, scores_idx_ub, scores_idx_out, score_idx_lens * 2,
                                            self.level, dtype=self.box_dtype)

                    # move last 4096 scores_idx in to workspace
                    tik_instance.data_move(
                        self.workspace_score_idx[batch_idx, class_idx, score_idx_lens * (loop_num - 1)],
                        scores_idx_out[score_idx_lens], 0, 1, burst_lens_idx, 0, 0)

            else:
                # init tensor
                self.init_tensor(scores_ub, per_loop_ele, self.fp_min)
                gm2ub_for_vsort32(tik_instance, score_gm, [batch_idx, class_idx, 0], scores_ub, tail)
                tik_instance.vsort32(scores_idx_ub, scores_ub, index, repeat_times)
                sort_score_idx_by_desc(tik_instance, scores_idx_ub, scores_idx_out, score_idx_lens,
                                        self.level, dtype=self.box_dtype)

    def get_eff_size_by_threshold(self, scores_index, eff_size, shape_size, gate_value=0):
        """
        compute of index of effective scores based on the gate_value

        Parameters
        ----------
        scores_index : tensor
            scores_index in ub
        eff_size : scalar
            effective data size
        shape_size : int
            shape size of scores, must be 16 aligned
        gate_value : int
            threshold

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        shape = (shape_size,)
        mask_shape = (shape_size // self._block_ele,)

        if gate_value == 0:
            eff_size.set_as(shape_size)
            return
        else:
            with tik_instance.new_stmt_scope():
                scores_tmp = tik_instance.Tensor(self.box_dtype, shape, name="scores_tmp", scope=tik.scope_ubuf)
                scores_thresh = tik_instance.Tensor(self.box_dtype, shape, name="scores_thresh", scope=tik.scope_ubuf)
                # gen scores_thresh tensor
                self.init_tensor(scores_thresh, shape_size, gate_value)

                mask_dtype = "uint16" if self.box_dtype == "float16" else "uint32"
                tmp_mask = tik_instance.Tensor(mask_dtype, mask_shape, name="tmp_mask", scope=tik.scope_ubuf)

                # move scores data from scores_index to scores_tmp
                mask, _ = get_mask_rep_stride(scores_thresh)
                repeat_times = shape_size * self._unit_ele // mask
                tik_instance.vreducev2(None, scores_tmp, scores_index, self.pattern_type, repeat_times, 1, 8, 0)

                # gen mask and then get the effective data lens
                self.gen_mask(scores_thresh, scores_tmp, tmp_mask, shape_size)
                tik_instance.vreducev2(shape_size, scores_thresh, scores_tmp, tmp_mask, 1, 1, 8, 1,
                                       rsvd_scalar=eff_size, mask_mode="counter")

    # 'pylint: disable=too-many-arguments
    def get_boxes_after_score_thresh(self, xx1, xx2, yy1, yy2, scores_ub, batch_idx, class_idx, boxes,
                                     scores_index, size=4096, offset=0):
        """
        move boxes_gm to boxes_ub according to the sorting index

        Parameters
        ----------
        xx1 : tensor
            x1 data in ub
        xx2 : tensor
            x2 data in ub
        yy1 : tensor
            y1 data in ub
        yy2 : tensor
            y2 data in ub
        scores_ub : tensor
            scores data in ub
        batch_idx : int
            batch index
        class_idx : int
            class index
        boxes : tensor
            input location data in gm
        scores_index : tensor
            scores_index in ub
        size : int/Scalar
            valid num default as 4096
        offset : int/Scalar
            pre valid boxes num default as 0
        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        idx_aligned = self._per_loop_uint
        with tik_instance.if_scope(offset == 0):
            self.init_tensor(xx1, idx_aligned)
            self.init_tensor(xx2, idx_aligned)
            self.init_tensor(yy1, idx_aligned)
            self.init_tensor(yy2, idx_aligned)
            self.init_tensor(scores_ub, idx_aligned)

        with tik_instance.if_scope(tik.all(offset > 0, size + offset > self._per_loop_uint)):
            size.set_as(self._per_loop_uint - offset)

        # define the location index, the distance from the begin of class_begin
        lo_index = tik_instance.Scalar("uint32")

        # boxes is set as [4 * 4096], then the x1/x2/y1/y2 is stored in [0/1/2/3, 4096]
        if self.box_dtype == "float16":
            with tik_instance.for_range(0, size) as idx:
                scores_index_offset = idx * self._unit_ele
                lo_index.set_as(
                    scores_index[scores_index_offset + 2:scores_index_offset + 4].reinterpret_cast_to("uint32"))
                xx1[offset + idx].set_as(boxes[batch_idx, class_idx, 0, lo_index])
                yy1[offset + idx].set_as(boxes[batch_idx, class_idx, 1, lo_index])
                xx2[offset + idx].set_as(boxes[batch_idx, class_idx, 2, lo_index])
                yy2[offset + idx].set_as(boxes[batch_idx, class_idx, 3, lo_index])
                scores_ub[offset + idx].set_as(scores_index[scores_index_offset])
        elif self.box_dtype == "float32":
            with tik_instance.for_range(0, size) as idx:
                scores_index_offset = idx * self._unit_ele
                lo_index.set_as(scores_index[scores_index_offset + 1].reinterpret_cast_to("uint32"))
                xx1[offset + idx].set_as(boxes[batch_idx, class_idx, 0, lo_index])
                yy1[offset + idx].set_as(boxes[batch_idx, class_idx, 1, lo_index])
                xx2[offset + idx].set_as(boxes[batch_idx, class_idx, 2, lo_index])
                yy2[offset + idx].set_as(boxes[batch_idx, class_idx, 3, lo_index])
                scores_ub[offset + idx].set_as(scores_index[scores_index_offset])

    # 'pylint: disable=too-many-arguments
    def iou_selection(self, xx1, xx2, yy1, yy2, scores, eff_lens):
        """
        calculate the overlap of multi boxes, sieve out target boxes with  iou_thresh

        Parameters
        ----------
        xx1 : tensor
            x1 data in ub
        xx2 : tensor
            x2 data in ub
        yy1 : tensor
            y1 data in ub
        yy2 : tensor
            y2 data in ub
        scores : tensor
            scores data in ub
        eff_lens : Scalar
            effect data lens

        Returns
        -------
        eff_lens : int
            valid boxes num
        """
        tik_instance = self.tik_instance
        shape_aligned = self._per_loop_uint

        with tik_instance.new_stmt_scope():
            single_area = tik_instance.Tensor(self.box_dtype, [shape_aligned, ], name="single_area",
                                              scope=tik.scope_ubuf)
            iou = tik_instance.Tensor(self.box_dtype, [shape_aligned, ], name="iou",
                                      scope=tik.scope_ubuf)
            mask_shape_lens = self._per_loop_uint // self._block_ele
            mask_dtype = "uint16" if self.box_dtype == "float16" else "uint32"
            mask_ub = tik_instance.Tensor(mask_dtype, [mask_shape_lens, ], name="mask_ub",
                                              scope=tik.scope_ubuf)

            self.init_tensor(iou, shape_aligned)
            self.init_tensor(mask_ub, mask_shape_lens)

            # get area of every window
            self.get_rectangle_area(xx1, xx2, yy1, yy2, single_area)

            # calculate the iou, end up when the output windows is more than max_size_per_class
            overlap = tik_instance.Tensor(self.box_dtype, [shape_aligned, ], name="overlap", scope=tik.scope_ubuf)
            # define tmp tensor for following use, to reduce the cycle of apply/release memory
            tmp1 = tik_instance.Tensor(self.box_dtype, [shape_aligned, ], name="tmp1", scope=tik.scope_ubuf)
            tmp2 = tik_instance.Tensor(self.box_dtype, [shape_aligned, ], name="tmp2", scope=tik.scope_ubuf)

            with tik_instance.for_range(0, self.max_size_per_class) as idx:
                with tik_instance.if_scope(idx < eff_lens):
                    # get overlap of windows_idx and the followings
                    self.get_overlap(xx1, xx2, yy1, yy2, overlap, tmp1, tmp2, idx)
                    # get overlap of windows_idx and the followings
                    self.cal_iou(single_area, iou, tmp2, idx, self._per_loop_uint)
                    self.gen_mask(overlap, iou, mask_ub, self._per_loop_uint)
                    # to keep points and lines
                    self.update_mask(scores, mask_ub, mask_dtype)
                    self.update_input(xx1, xx2, yy1, yy2, scores, eff_lens, tmp1, tmp2, mask_ub, single_area)
                with tik_instance.else_scope():
                    tik_instance.tik_break()

    def update_mask(self, scores, mask_ub, mask_dtype):
        """
        update the mask in order to keep points and lines

        Parameters
        ----------
        scores : tensor
            scores data in ub
        mask_ub : tensor
            mask pattern

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        with tik_instance.new_stmt_scope():
            zero_scores = tik_instance.Tensor(self.box_dtype, [self._per_loop_uint, ], name="zero_scores",
                                              scope=tik.scope_ubuf)
            mask_shape_lens = self._per_loop_uint // self._block_ele
            tmp_mask0 = tik_instance.Tensor(mask_dtype, [mask_shape_lens, ], name="tmp_mask0", scope=tik.scope_ubuf)
            tmp_mask1 = tik_instance.Tensor(mask_dtype, [mask_shape_lens, ], name="tmp_mask1", scope=tik.scope_ubuf)
            self.init_tensor(zero_scores, self._per_loop_uint)

            vector_mask, _ = get_mask_rep_stride(scores)
            repeat_times = self._per_loop_uint // vector_mask
            tik_instance.vec_cmpv_ne(tmp_mask0, scores, zero_scores, repeat_times, 8, 8)

            vector_mask_vand, _ = get_mask_rep_stride(tmp_mask0)
            repeat_times_vand = mask_shape_lens // vector_mask_vand
            if repeat_times_vand > 0:
                tik_instance.vand(vector_mask_vand, tmp_mask1.reinterpret_cast_to("uint16"), 
                                    tmp_mask0.reinterpret_cast_to("uint16"),
                                    mask_ub.reinterpret_cast_to("uint16"), repeat_times_vand, 1, 1, 1, 8, 8, 8)
            else:
                tik_instance.vand(mask_shape_lens, tmp_mask1.reinterpret_cast_to("uint16"), 
                                    tmp_mask0.reinterpret_cast_to("uint16"),
                                     mask_ub.reinterpret_cast_to("uint16"), 1, 1, 1, 1, 8, 8, 8)

            burst_lens = mask_shape_lens // self._block_ele
            tik_instance.data_move(mask_ub, tmp_mask1, 0, 1, burst_lens, 0, 0)

    # 'pylint: disable=huawei-too-many-arguments
    def update_input(self, xx1, xx2, yy1, yy2, scores, size, tmp1, tmp2, cmpmask_ub, single_area=None):
        """
        update the location and scores according to cmpmask

        Parameters
        ----------
        xx1 : tensor
            x1 data in ub
        xx2 : tensor
            x2 data in ub
        yy1 : tensor
            y1 data in ub
        yy2 : tensor
            y2 data in ub
        scores : tensor
            scores data in ub
        single_area : tensor
            boxes area data in ub
        size : scalar
            data size
        tmp1: tmp tensor
        tmp2: tmp tensor
        cmpmask_ub : tensor
            mask pattern

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        mask = self._per_loop_uint
        burst_lens = self._per_loop_uint // self._block_ele
        self.init_tensor(tmp1, self._per_loop_uint)
        self.init_tensor(tmp2, self._per_loop_uint)

        tik_instance.vreducev2(mask, tmp1, xx1, cmpmask_ub, 1, 1, 8, 1, rsvd_scalar=size, mask_mode="counter")
        tik_instance.data_move(xx1, tmp1, 0, 1, burst_lens, 0, 0)

        tik_instance.vreducev2(mask, tmp2, xx2, cmpmask_ub, 1, 1, 8, 1, mask_mode="counter")
        tik_instance.data_move(xx2, tmp2, 0, 1, burst_lens, 0, 0)

        tik_instance.vreducev2(mask, tmp1, yy1, cmpmask_ub, 1, 1, 8, 1, mask_mode="counter")
        tik_instance.data_move(yy1, tmp1, 0, 1, burst_lens, 0, 0)

        tik_instance.vreducev2(mask, tmp2, yy2, cmpmask_ub, 1, 1, 8, 1, mask_mode="counter")
        tik_instance.data_move(yy2, tmp2, 0, 1, burst_lens, 0, 0)

        tik_instance.vreducev2(mask, tmp1, scores, cmpmask_ub, 1, 1, 8, 1, mask_mode="counter")
        tik_instance.data_move(scores, tmp1, 0, 1, burst_lens, 0, 0)

        tik_instance.vreducev2(mask, tmp2, single_area, cmpmask_ub, 1, 1, 8, 1, mask_mode="counter")
        tik_instance.data_move(single_area, tmp2, 0, 1, burst_lens, 0, 0)

    def get_rectangle_area(self, xx1, xx2, yy1, yy2, dst):
        """
        area = (x2-x1) * (y2-y1), this is vector computing
        area can be reused in loops

        Parameters
        ----------
        xx1 : tensor
            x1 data in ub
        xx2 : tensor
            x2 data in ub
        yy1 : tensor
            y1 data in ub
        yy2 : tensor
            y2 data in ub
        dst : tensor
            rectangle_area data in ub

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        shape_aligned = self._per_loop_uint

        with tik_instance.new_stmt_scope():
            tmp1 = tik_instance.Tensor(self.box_dtype, [shape_aligned, ], name="tmp1", scope=tik.scope_ubuf)
            tmp2 = tik_instance.Tensor(self.box_dtype, [shape_aligned, ], name="y_diff", scope=tik.scope_ubuf)

            mask, _ = get_mask_rep_stride(xx2)
            repeat_times = shape_aligned // mask

            tik_instance.vsub(mask, tmp1, xx2, xx1, repeat_times, 1, 1, 1, 8, 8, 8)
            tik_instance.vsub(mask, tmp2, yy2, yy1, repeat_times, 1, 1, 1, 8, 8, 8)

            tik_instance.vmul(mask, dst, tmp1, tmp2, repeat_times, 1, 1, 1, 8, 8, 8)

    # 'pylint: disable=huawei-too-many-arguments
    def get_overlap(self, xx1, xx2, yy1, yy2, overlap, tmp, yyy1, offset):
        """
        get overlap area of x1 and the following others, the pre units mask the overlap 0

        Parameters
        ----------
        xx1 : tensor
            x1 data in ub
        xx2 : tensor
            x2 data in ub
        yy1 : tensor
            y1 data in ub
        yy2 : tensor
            y2 data in ub
        overlap : tensor
            overlap data in ub
        tmp1 : tensor
            tmp tensor
        yyy1 : tensor
            tmp tensor
        offset : scalar
            location index

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance

        shape_aligned = self._per_loop_uint

        with tik_instance.new_stmt_scope():
            tmp1 = tik_instance.Tensor(self.box_dtype, [shape_aligned, ], name="tmp1", scope=tik.scope_ubuf)

            x1 = tik_instance.Scalar(self.box_dtype, init_value=xx1[offset])
            x2 = tik_instance.Scalar(self.box_dtype, init_value=xx2[offset])
            y1 = tik_instance.Scalar(self.box_dtype, init_value=yy1[offset])
            y2 = tik_instance.Scalar(self.box_dtype, init_value=yy2[offset])

            # `tmp = max(xx1[i], xx1[1:]), overlap=min(xx2[i], xx2[1:])
            tik_func_vmaxs(tik_instance, tmp, xx1, x1, shape_aligned, dst_blk=1, src_blk=1, dst_rep=8, src_rep=8)
            tik_func_vmins(tik_instance, overlap, xx2, x2, shape_aligned, dst_blk=1, src_blk=1, dst_rep=8, src_rep=8)
            mask, _ = get_mask_rep_stride(xx1)
            repeat_times = shape_aligned // mask
            # `w = max(0, xx2-xx1+offset), offset=0 here, sorted in tmp1`
            tik_instance.vsub(mask, tmp, overlap, tmp, repeat_times, 1, 1, 1, 8, 8, 8)
            tik_func_vmaxs(tik_instance, tmp1, tmp, 0, shape_aligned, dst_blk=1, src_blk=1, dst_rep=8, src_rep=8)

            # `yyy1 = max(yy1[i], yy1[1:]), overlap = min(yy2[i], yy2[1:])`
            tik_func_vmaxs(tik_instance, yyy1, yy1, y1, shape_aligned, dst_blk=1, src_blk=1, dst_rep=8, src_rep=8)
            tik_func_vmins(tik_instance, overlap, yy2, y2, shape_aligned, dst_blk=1, src_blk=1, dst_rep=8, src_rep=8)

            # `h = max(0, yy2 - yy1 + offset), offset=0 here, sorted in tmp`
            tik_instance.vsub(mask, yyy1, overlap, yyy1, repeat_times, 1, 1, 1, 8, 8, 8)
            tik_func_vmaxs(tik_instance, tmp, yyy1, 0, shape_aligned, dst_blk=1, src_blk=1, dst_rep=8, src_rep=8)

            tik_instance.vmul(mask, overlap, tmp1, tmp, repeat_times, 1, 1, 1, 8, 8, 8)

            # the overlap of the fixed boxes and itself default as 0
            overlap[offset].set_as(0)

    def cal_iou(self, src0, dst, tmp, offset, size):
        """
        to calculate the related areas based on iou_thresh

        Parameters
        ----------
        src0 : tensor
            area of every window
        dst : tensor
            iou data
        tmp : tensor
            tmp tensor in ub
        offset : int
            the start window offset from the beginning
        size : int
            valid num

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance

        # calculate the sum of area1 and area2
        area1 = tik_instance.Scalar(self.box_dtype, init_value=src0[offset])

        mask, _ = get_mask_rep_stride(src0)
        repeat_times = size // mask
        tik_instance.vadds(mask, tmp, src0, area1, repeat_times, 1, 1, 8, 8)
        tik_instance.vmuls(mask, dst, tmp, self.iou_thresh, repeat_times, 1, 1, 8, 8)

    def gen_mask(self, overlap, iou, mask, size):
        """
        gen mask

        Parameters
        ----------
        overlap : tensor
            overlap data in ub
        iou : tensor
            iou data in ub
        mask : tensor
            mask tensor
        size: total size of proposals

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance

        vector_mask, _ = get_mask_rep_stride(overlap)
        per_loop_num = Constant.REPEAT_TIMES_MAX * vector_mask
        loops = size // per_loop_num
        offset = tik_instance.Scalar("int32", init_value=0)

        # step1: max. mask * max. repeat  * loops times
        if loops > 0:
            with tik_instance.for_range(0, loops) as idx:
                # vec_cmpv_lt deal with 255 * 128 fp16 elements once
                tik_instance.vec_cmpv_le(mask[offset], overlap[offset], iou[offset], Constant.REPEAT_TIMES_MAX, 8, 8)
                offset.set_as(per_loop_num * (idx + 1))

        # step3: last num
        repeat_times = (size % per_loop_num) // vector_mask
        if repeat_times > 0:
            tik_instance.vec_cmpv_le(mask[offset], overlap[offset], iou[offset], repeat_times, 8, 8)

    def store_data(self, xx1, xx2, yy1, yy2, scores, batch_idx, class_idx, offset=0):
        """
        sort data in workspace for batch sorting

        Parameters
        ----------
        xx1 : tensor
            x1 data in ub
        xx2 : tensor
            x2 data in ub
        yy1 : tensor
            y1 data in ub
        yy2 : tensor
            y2 data in ub
        scores : tensor
            scores data in ub
        batch_idx : int
            batch index
        class_idx : int
            class index
        offset : int
            offset from the beginning

        Returns
        -------
        None
        """

        tik_instance = self.tik_instance

        size = self.max_size_per_class
        burst_lens = size // self._block_ele
        tail = size % self._block_ele
        # move ub data to workspace
        if burst_lens > 0:
            tik_instance.data_move(self.workspace_boxes[batch_idx, class_idx, 0, offset], xx1, 0, 1, burst_lens, 0, 0)
            tik_instance.data_move(self.workspace_boxes[batch_idx, class_idx, 1, offset], yy1, 0, 1, burst_lens, 0, 0)
            tik_instance.data_move(self.workspace_boxes[batch_idx, class_idx, 2, offset], xx2, 0, 1, burst_lens, 0, 0)
            tik_instance.data_move(self.workspace_boxes[batch_idx, class_idx, 3, offset], yy2, 0, 1, burst_lens, 0, 0)
            tik_instance.data_move(self.workspace_scores[batch_idx, class_idx, offset], scores, 0, 1, burst_lens, 0, 0)

        if tail > 0:
            block_ub0 = tik_instance.Tensor(self.box_dtype, (self._block_ele,), name="block_ub0", 
                                            scope=tik.scope_ubuf)
            block_ub1 = tik_instance.Tensor(self.box_dtype, (self._block_ele,), name="block_ub1", 
                                            scope=tik.scope_ubuf)

            last_offset = offset + max(0, size - self._block_ele)
            with tik_instance.for_range(0, self._block_ele) as idx0:
                block_ub0[idx0].set_as(xx1[last_offset + idx0])
            tik_instance.data_move(self.workspace_boxes[batch_idx, class_idx, 0, last_offset], block_ub0, 0, 1, 1, 0, 0)
            with tik_instance.for_range(0, self._block_ele) as idx1:
                block_ub1[idx1].set_as(yy1[last_offset + idx1])
            tik_instance.data_move(self.workspace_boxes[batch_idx, class_idx, 1, last_offset], block_ub1, 0, 1, 1, 0, 0)
            with tik_instance.for_range(0, self._block_ele) as idx2:
                block_ub0[idx2].set_as(xx2[last_offset + idx2])
            tik_instance.data_move(self.workspace_boxes[batch_idx, class_idx, 2, last_offset], block_ub0, 0, 1, 1, 0, 0)
            with tik_instance.for_range(0, self._block_ele) as idx3:
                block_ub1[idx3].set_as(yy2[last_offset + idx3])
            tik_instance.data_move(self.workspace_boxes[batch_idx, class_idx, 3, last_offset], block_ub1, 0, 1, 1, 0, 0)
            with tik_instance.for_range(0, self._block_ele) as idx4:
                block_ub0[idx4].set_as(scores[last_offset + idx4])
            tik_instance.data_move(self.workspace_scores[batch_idx, class_idx, last_offset], block_ub0, 0, 1, 1, 0, 0)

    def get_batch_scores_idx(self, src, real_batch_idx):
        """
        sort data in workspace for batch sorting

        Parameters
        ----------
        src : tensor
            scores data in workspace
        real_batch_idx : int
            batch index

        Returns
        -------
        score_idx_out : tensor
            score_index in ub
        """
        tik_instance = self.tik_instance
        mask, _ = get_mask_rep_stride(src)
        size = Constant.BATCH_LOOP_UNIT
        size_out = Constant.BATCH_LOOP_UNIT * self._unit_ele

        score_idx_out = tik_instance.Tensor(self.box_dtype, [size_out * self._unit_ele * 2, ], name="score_idx_out",
                                            scope=tik.scope_ubuf)

        # when the class_num * max_size_per_class is less than 4096, the data can be processed in one loop in ub
        # otherwise the sorting should be cycled
        if self.classes * self.max_size_per_class < self._per_loop_uint:
            with tik_instance.new_stmt_scope():
                index = tik_instance.Tensor("uint32", [size, ], name="index", scope=tik.scope_ubuf)
                score_tmp = tik_instance.Tensor(self.box_dtype, [size, ], name="score_tmp", scope=tik.scope_ubuf)
                score_idx_sort = tik_instance.Tensor(self.box_dtype, [size * self._unit_ele, ], name="score_idx_sort",
                                                     scope=tik.scope_ubuf)

                init_index(tik_instance, self.idx_gm, index, 0, size, dtype=self.box_dtype)
                self.init_tensor(score_tmp, size, self.fp_min)
                self.init_tensor(score_idx_sort, size * self._unit_ele, self.fp_min)
                self.init_tensor(score_idx_out, size_out * self._unit_ele * 2, self.fp_min)

                # move scores in workspcae to score_tmp
                burst_lens = self.max_size_per_class // self._block_ele
                tail = self.max_size_per_class % self._block_ele
                uint_lens = ceil_div(self.max_size_per_class, self._block_ele) * self._block_ele
                with tik_instance.for_range(0, self.classes) as i:
                    with tik_instance.if_scope(burst_lens > 0):
                        tik_instance.data_move(score_tmp[uint_lens * i], src[real_batch_idx, i, 0], 0, 1,
                                               burst_lens, 0, 0)
                    with tik_instance.for_range(0, tail) as _idx:
                        score_tmp[uint_lens * i + self.max_size_per_class - tail + _idx].set_as(
                            src[real_batch_idx, i, self.max_size_per_class - tail + _idx])
                repeat_times = size // Constant.REPEAT_ELE
                tik_instance.vsort32(score_idx_sort, score_tmp, index, repeat_times)
                do_lens = size * self._unit_ele
                sort_score_idx_by_desc(tik_instance, score_idx_sort, score_idx_out, do_lens, self.level,
                                       dtype=self.box_dtype)

        else:
            self.gen_batch_score_index(real_batch_idx, src, score_idx_out)

        return score_idx_out

    def gen_batch_score_index(self, batch_idx, src, scores_idx_out):
        """
        construct the tensor(score_index) for vsort32 and vmrgsort command
        get top max_total_size scores and index

        Parameters
        ----------
        batch_idx : int
            batch index
        src : tensor
            input scores data in workspace
        scores_idx_out : tensor
            scores_idx_out

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance

        mask, _ = get_mask_rep_stride(src)
        uint_lens = ceil_div(self.max_size_per_class, self._block_ele) * self._block_ele
        # to accelerate the process, every cycle deal with data of 4 classes
        per_loop_ele = Constant.BATCH_LOOP_UNIT * self._unit_ele
        score_idx_lens = per_loop_ele * self._unit_ele
        loop_num = ceil_div(self.classes, self._unit_ele)

        with tik_instance.new_stmt_scope():
            index = tik_instance.Tensor("uint32", [per_loop_ele, ], name="index", scope=tik.scope_ubuf)
            init_index(tik_instance, self.idx_gm, index, 0, per_loop_ele, dtype=self.box_dtype)

            scores_ub = tik_instance.Tensor(self.box_dtype, [per_loop_ele, ], name="scores_ub", scope=tik.scope_ubuf)
            scores_idx_ub = tik_instance.Tensor(self.box_dtype, [score_idx_lens * 2, ],
                                                name="scores_idx_ub", scope=tik.scope_ubuf)
            self.init_tensor(scores_ub, per_loop_ele, self.fp_min)
            self.init_tensor(scores_idx_ub, score_idx_lens * 2, 0)

            repeat_times = ceil_div(per_loop_ele, Constant.REPEAT_ELE)

            if loop_num > 1:
                burst_lens = score_idx_lens // self._block_ele
                # the first part units
                with tik_instance.for_range(0, self._unit_ele) as class_idx:
                    gm2ub_for_vsort32(tik_instance, src, [batch_idx, class_idx, 0], scores_ub[uint_lens * class_idx],
                                      self.max_size_per_class)

                tik_instance.vsort32(scores_idx_out, scores_ub, index, repeat_times)

                with tik_instance.for_range(1, loop_num) as loop_idx:
                    # set value for index
                    with tik_instance.if_scope(loop_idx == loop_num - 1):
                        # init scores_ub & scores_idx_ub in order to clear the pre data
                        self.init_tensor(scores_ub, per_loop_ele, self.fp_min)

                    init_index(tik_instance, self.idx_gm, index, loop_idx * per_loop_ele, per_loop_ele, 
                                    dtype=self.box_dtype)
                    with tik_instance.for_range(0, self._unit_ele) as class_idx:
                        real_class_idx = loop_idx * self._unit_ele + class_idx
                        with tik_instance.if_scope(real_class_idx < self.classes):
                            gm2ub_for_vsort32(tik_instance, src, [batch_idx, real_class_idx, 0],
                                              scores_ub[uint_lens * real_class_idx], self.max_size_per_class)
                    tik_instance.vsort32(scores_idx_ub, scores_ub, index, repeat_times)

                    tik_instance.data_move(scores_idx_ub[score_idx_lens], scores_idx_out, 0, 1, burst_lens, 0, 0)

                    do_lens = score_idx_lens * 2
                    sort_score_idx_by_desc(tik_instance, scores_idx_ub, scores_idx_out,
                                            do_lens, self.level, dtype=self.box_dtype)

            else:
                with tik_instance.for_range(0, self._unit_ele) as class_idx:
                    gm2ub_for_vsort32(tik_instance, src, [batch_idx, class_idx, 0], scores_ub[uint_lens * class_idx],
                                      self.max_size_per_class)
                tik_instance.vsort32(scores_idx_ub, scores_ub, index, repeat_times)
                sort_score_idx_by_desc(tik_instance, scores_idx_ub, scores_idx_out, score_idx_lens,
                                        self.level, dtype=self.box_dtype)

    def sort_single_class_per_batch(self, batch_idx, data_lens, xx1, xx2, yy1, yy2, scores_ub):
        """
        batch(single class) sorting then move data out

        Parameters
        ----------
        batch_idx : int
            batch index
        data_lens : Scalar
            valid detection boxes num
        xx1 : tensor
            x1 data in ub
        xx2 : tensor
            x2 data in ub
        yy1 : tensor
            y1 data in ub
        yy2 : tensor
            y2 data in ub
        scores_ub : tensor
            scores data in ub

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        class_size = max(self.max_total_size, self._block_ele)
        classes_out = tik_instance.Tensor(self.box_dtype, [class_size, ], name="classes_out", scope=tik.scope_ubuf)
        valid_detection = tik_instance.Tensor("int32", [Constant.BLOCK_ELE_B32, ], name="valid_detection",
                                              scope=tik.scope_ubuf)
        self.init_tensor(classes_out, class_size)
        self.init_tensor(valid_detection, Constant.BLOCK_ELE_B32, 0)
        
        with tik_instance.if_scope(data_lens > self.max_total_size):
            data_lens.set_as(self.max_total_size)

        with tik_instance.for_range(0, data_lens) as idx:
            classes_out[idx].set_as(0)

        valid_detection[0].set_as(data_lens)

        self.move_data_out(xx1, xx2, yy1, yy2, scores_ub, classes_out, valid_detection, batch_idx)

    def sort_class_per_batch(self, batch_idx):
        """
        batch sorting then move data out

        Parameters
        ----------
        batch_idx : int
            batch index

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        mask, _ = get_mask_rep_stride(self.workspace_scores)

        date_lens = ceil_div(self.max_total_size, mask) * mask
        init_shape = (date_lens,)

        eff_size = tik_instance.Scalar("uint32")

        x1 = tik_instance.Tensor(self.box_dtype, init_shape, name="x1", scope=tik.scope_ubuf)
        x2 = tik_instance.Tensor(self.box_dtype, init_shape, name="x2", scope=tik.scope_ubuf)
        y1 = tik_instance.Tensor(self.box_dtype, init_shape, name="y1", scope=tik.scope_ubuf)
        y2 = tik_instance.Tensor(self.box_dtype, init_shape, name="y2", scope=tik.scope_ubuf)
        scores_ub = tik_instance.Tensor(self.box_dtype, init_shape, name="scores_ub", scope=tik.scope_ubuf)
        classes_out = tik_instance.Tensor(self.box_dtype, init_shape, name="classes_out", scope=tik.scope_ubuf)
        classes_tmp = tik_instance.Tensor("uint32", init_shape, name="classes_tmp", scope=tik.scope_ubuf)
        valid_detection = tik_instance.Tensor("int32", [Constant.BLOCK_ELE_B32, ], name="valid_detection",
                                              scope=tik.scope_ubuf)

        # set default value to x1/x2/y1/y2/scores/class_out/valid_detection
        self.init_tensor(x1, date_lens, 0)
        self.init_tensor(x2, date_lens, 0)
        self.init_tensor(y1, date_lens, 0)
        self.init_tensor(y2, date_lens, 0)
        self.init_tensor(scores_ub, date_lens)
        self.init_tensor(classes_out, date_lens)
        self.init_tensor(classes_tmp, date_lens)
        self.init_tensor(valid_detection, Constant.BLOCK_ELE_B32)

        scores_idx = self.get_batch_scores_idx(self.workspace_scores, batch_idx)
        self.get_eff_size_by_threshold(scores_idx, eff_size, date_lens, self.score_thresh)

        with tik_instance.if_scope(eff_size > self.max_total_size):
            eff_size.set_as(self.max_total_size)

        valid_detection[0].set_as(eff_size)

        # define the location index, the distance from the begin of class_begin
        tmp = tik_instance.Scalar("uint32")
        lo_index = tik_instance.Scalar("uint32")
        class_idx = tik_instance.Scalar("uint32")

        # for example: boxes is set as [4 * 4096], then the x1/y1/x2/y2 is stored in [0/1/2/3, 4096]
        # for every class, data(max_total_size) is stored in scores_idx(uint 16 aligned)
        uint_lens = Constant.BATCH_LOOP_UNIT * self._block_ele
        with tik_instance.new_stmt_scope():
            if self.box_dtype == "float16":
                with tik_instance.for_range(0, eff_size) as idx:
                    scores_index_offset = idx * self._unit_ele
                    tmp.set_as(
                        scores_idx[scores_index_offset + 2:scores_index_offset + 4].reinterpret_cast_to("uint32"))
                    class_idx.set_as(tmp // uint_lens)
                    lo_index.set_as(tmp % uint_lens)
                    x1[idx].set_as(self.workspace_boxes[batch_idx, class_idx, 0, lo_index])
                    y1[idx].set_as(self.workspace_boxes[batch_idx, class_idx, 1, lo_index])
                    x2[idx].set_as(self.workspace_boxes[batch_idx, class_idx, 2, lo_index])
                    y2[idx].set_as(self.workspace_boxes[batch_idx, class_idx, 3, lo_index])
                    classes_tmp[idx].set_as(class_idx)
                    scores_ub[idx].set_as(scores_idx[idx * self._unit_ele])
            elif self.box_dtype == "float32":
                with tik_instance.for_range(0, eff_size) as idx:
                    scores_index_offset = idx * self._unit_ele
                    tmp.set_as(
                        scores_idx[scores_index_offset + 1].reinterpret_cast_to("uint32"))
                    class_idx.set_as(tmp // uint_lens)
                    lo_index.set_as(tmp % uint_lens)
                    x1[idx].set_as(self.workspace_boxes[batch_idx, class_idx, 0, lo_index])
                    y1[idx].set_as(self.workspace_boxes[batch_idx, class_idx, 1, lo_index])
                    x2[idx].set_as(self.workspace_boxes[batch_idx, class_idx, 2, lo_index])
                    y2[idx].set_as(self.workspace_boxes[batch_idx, class_idx, 3, lo_index])
                    classes_tmp[idx].set_as(class_idx)
                    scores_ub[idx].set_as(scores_idx[idx * self._unit_ele])

            # conv the dtype of classes_tmp to int32, then move to classes_out, meanwhile the dtype is changed to fp16
            # meanwhil the dtype is changed to fp16//fp32
            data_b = classes_tmp.reinterpret_cast_to("int32")
            mask, _ = get_mask_rep_stride(data_b)
            repeat_times = date_lens // mask
            if self.box_dtype == "float16":
                tik_instance.vec_conv(mask, "none", classes_out, data_b, repeat_times, 4, 8, deqscale=1.0)
            else:
                tik_instance.vec_conv(mask, "none", classes_out, data_b, repeat_times, 4, 8)


        self.move_data_out(x1, x2, y1, y2, scores_ub, classes_out, valid_detection, batch_idx)

    def move_data_out(self, x1, x2, y1, y2, scores_ub, classes_out, valid_detection, batch_idx):
        """
        sort data in workspace for batch sorting

        Parameters
        ----------
        x1 : tensor
            x1 data in ub
        x2 : tensor
            x2 data in ub
        y1 : tensor
            y1 data in ub
        y2 : tensor
            y2 data in ub
        scores_ub : tensor
            scores data in ub
        classes_out : tensor
            classes data in ub
        valid_detection : tensor
            valid_detection data in ub
        batch_idx : int
            batch_index

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        [nmsed_boxes_gm, nmsed_scores_gm, nmsed_classes_gm, nmsed_num_gm] = self.output_gm_list

        data_size = self.max_total_size
        burst_lens = data_size // self._block_ele

        with tik_instance.if_scope(burst_lens > 0):
            tik_instance.data_move(nmsed_boxes_gm[batch_idx, 0, 0], x1, 0, 1, burst_lens, 0, 0)
            tik_instance.data_move(nmsed_boxes_gm[batch_idx, 1, 0], y1, 0, 1, burst_lens, 0, 0)
            tik_instance.data_move(nmsed_boxes_gm[batch_idx, 2, 0], x2, 0, 1, burst_lens, 0, 0)
            tik_instance.data_move(nmsed_boxes_gm[batch_idx, 3, 0], y2, 0, 1, burst_lens, 0, 0)
            tik_instance.data_move(nmsed_scores_gm[batch_idx, 0], scores_ub, 0, 1, burst_lens, 0, 0)
            tik_instance.data_move(nmsed_classes_gm[batch_idx, 0], classes_out, 0, 1, burst_lens, 0, 0)

        block_ub0 = tik_instance.Tensor(self.box_dtype, (self._block_ele,), name="block_ub0", scope=tik.scope_ubuf)
        block_ub1 = tik_instance.Tensor(self.box_dtype, (self._block_ele,), name="block_ub1", scope=tik.scope_ubuf)

        # move tail data to gm
        last_offset = max(0, data_size - self._block_ele)
        with tik_instance.for_range(0, self._block_ele) as idx0:
            block_ub0[idx0].set_as(x1[last_offset + idx0])
        tik_instance.data_move(nmsed_boxes_gm[batch_idx, 0, last_offset], block_ub0, 0, 1, 1, 0, 0)
        with tik_instance.for_range(0, self._block_ele) as idx1:
            block_ub1[idx1].set_as(y1[last_offset + idx1])
        tik_instance.data_move(nmsed_boxes_gm[batch_idx, 1, last_offset], block_ub1, 0, 1, 1, 0, 0)
        with tik_instance.for_range(0, self._block_ele) as idx2:
            block_ub0[idx2].set_as(x2[last_offset + idx2])
        tik_instance.data_move(nmsed_boxes_gm[batch_idx, 2, last_offset], block_ub0, 0, 1, 1, 0, 0)
        with tik_instance.for_range(0, self._block_ele) as idx3:
            block_ub1[idx3].set_as(y2[last_offset + idx3])
        tik_instance.data_move(nmsed_boxes_gm[batch_idx, 3, last_offset], block_ub1, 0, 1, 1, 0, 0)

        with tik_instance.for_range(0, self._block_ele) as idx4:
            block_ub0[idx4].set_as(scores_ub[last_offset + idx4])
        tik_instance.data_move(nmsed_scores_gm[batch_idx, last_offset], block_ub0, 0, 1, 1, 0, 0)

        with tik_instance.for_range(0, self._block_ele) as idx5:
            block_ub1[idx5].set_as(classes_out[last_offset + idx5])
        tik_instance.data_move(nmsed_classes_gm[batch_idx, last_offset], block_ub1, 0, 1, 1, 0, 0)

        tik_instance.data_move(nmsed_num_gm[batch_idx, 0], valid_detection, 0, 1, 1, 0, 0)
