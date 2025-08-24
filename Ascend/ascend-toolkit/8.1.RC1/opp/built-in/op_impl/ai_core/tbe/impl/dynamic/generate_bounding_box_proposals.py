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
generate_bounding_box_proposals
"""
import math

from impl.dynamic.batch_multi_class_nms_topk import sort_within_ub
from impl.dynamic.batch_multi_class_nms_topk import sort_within_ub_scalar
from impl.dynamic.batch_multi_class_non_max_suppression import ceil_div
from impl.dynamic.batch_multi_class_non_max_suppression import do_nms_compute
from impl.dynamic.batch_multi_class_non_max_suppression import tik_func_sort_with_ub
from impl.dynamic.batch_multi_class_non_max_suppression import tik_func_trans_to_proposals
from impl.dynamic.batch_multi_class_non_max_suppression import tik_func_vadds
from impl.dynamic.batch_multi_class_non_max_suppression import tik_func_vadds_scalar
from impl.dynamic.batch_multi_class_non_max_suppression import tik_func_vcomple
from impl.dynamic.batch_multi_class_non_max_suppression import tik_func_vector
from impl.dynamic.batch_multi_class_non_max_suppression import tik_func_vector_scalar
from impl.dynamic.batch_multi_class_non_max_suppression import tik_func_vmuls
from impl.dynamic.batch_multi_class_non_max_suppression import total_num
from impl.dynamic.batch_multi_class_non_max_suppression import ub_offset
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik


class Constant:
    """
    The class for constant
    """
    BLOCK_BYTE_SIZE = 32
    MAX_REPEAT_NUM_EACH_CORE = 64
    IMAGE_NUM_ONE_REPEAT = 64
    DATA_NUM_ONE_BLOCK = 16
    MAX_NUM_MOVE_BOX = 255 * 16
    DATA_MEMSIZE_INVALID = 64
    MAX_INT32_VALUE = 2 ** 31 - 1
    RESERVED_UB = 0
    DTYPE_FP16 = "float16"
    DTYPE_FP32 = "float32"
    DTYPE_INT32 = "int32"
    DTYPE_UINT16 = "uint16"
    # tiling params dtype
    TILING_PARAM_DTYPE = "int32"
    # tiling params num
    TILING_PARAMS_NUM = 8
    # process 128 once
    MAX_CMP_NUM = 128
    # RPN compute 16 proposals per iteration
    RPN_PROPOSAL_NUM = 16

    # box dim
    BOX_DIM_NUM = 4
    # int32 byte
    BYTE_INT32 = 4
    # float32 byte
    BYTE_FP32 = 4
    # float16 byte
    BYTE_FP16 = 2
    # max repeat times
    MAX_REPEAT_TIMES = 255
    # full mask for fp32
    MASK_FP32 = 64
    # full mask for int32
    MASK_INT32 = 64

    SELECT_KEY_MODE_ANCHOR_BIG = 0
    SELECT_KEY_MODE_ANCHOR_SMALL = 1


def filter_greater_than(tik_instance, score_to_filter, cmp_ub, cmp_value, score_size):
    """
    function: cmp_ub >= cmp_value ? score_to_filter : 0
    @param [in] tik_instance: tik handle
    @param [in] score_to_filter: ub to filter
    @param [in] cmp_ub: ub to cmp
    @param [in] cmp_value: value to cmp
    @param [in] score_size: compute length
    --------
    output: score_to_filter
    """

    thresh_ub = tik_instance.Tensor(Constant.DTYPE_FP16, [score_size], name="thresh_ub", scope=tik.scope_ubuf)

    left_data = score_size % Constant.MAX_CMP_NUM
    repeat_time = score_size // Constant.MAX_CMP_NUM

    for iter_t in range(repeat_time):
        # use for mask less than 128 only
        tik_instance.vector_dup(128, thresh_ub[iter_t * 128], cmp_value, 1, 1, 8)
        cmp_mask = tik_instance.vcmp_ge(128, cmp_ub[iter_t * 128], thresh_ub[iter_t * 128], 1, 1)  # cmpmask
        zeros_ub = tik_instance.Tensor(Constant.DTYPE_FP16, [128], name="zeros_ub", scope=tik.scope_ubuf)
        tik_instance.vector_dup(128, zeros_ub[0], 0, 1, 1, 8)  # use for mask less than 128 only
        tik_instance.vsel(128, 0, score_to_filter[iter_t * 128], cmp_mask, score_to_filter[iter_t * 128], zeros_ub[0],
                          1, 1, 1, 1, 1, 1)

    if left_data > 0:
        tik_instance.vector_dup(left_data, thresh_ub[score_size - left_data], cmp_value, 1, 1, 8)
        cmp_mask = tik_instance.vcmp_ge(left_data, cmp_ub[score_size - left_data], thresh_ub[score_size - left_data], 1,
                                        1)  # cmpmask
        zeros_ub = tik_instance.Tensor(Constant.DTYPE_FP16, [left_data], name="zeros_ub", scope=tik.scope_ubuf)
        tik_instance.vector_dup(left_data, zeros_ub[0], 0, 1, 1, 8)  # use for mask less than 128 only
        tik_instance.vsel(left_data, 0, score_to_filter[score_size - left_data], cmp_mask,
                          score_to_filter[score_size - left_data], zeros_ub[0], 1, 1, 1, 1, 1, 1)


def filter_greater_than_scalar(tik_instance, score_to_filter, cmp_ub, cmp_value, score_size):
    """
    function: cmp_ub >= cmp_value ? score_to_filter : 0
    @param [in] tik_instance: tik handle
    @param [in] score_to_filter: ub to filter
    @param [in] cmp_ub: ub to cmp
    @param [in] cmp_value: value to cmp
    @param [in] score_size: compute length
    --------
    output: score_to_filter
    """

    thresh_ub = tik_instance.Tensor(Constant.DTYPE_FP16, [score_size], name="thresh_ub", scope=tik.scope_ubuf)

    left_data = tik_instance.Scalar(dtype=Constant.DTYPE_INT32)
    left_data.set_as(score_size % Constant.MAX_CMP_NUM)
    repeat_time = tik_instance.Scalar(dtype=Constant.DTYPE_INT32)
    repeat_time.set_as(score_size // Constant.MAX_CMP_NUM)

    with tik_instance.for_range(0, repeat_time) as iter_t:
        # use for mask less than 128 only
        tik_instance.vector_dup(128, thresh_ub[iter_t * 128], cmp_value, 1, 1, 8)
        cmp_mask = tik_instance.vcmp_ge(128, cmp_ub[iter_t * 128], thresh_ub[iter_t * 128], 1, 1)  # cmpmask
        zeros_ub = tik_instance.Tensor(Constant.DTYPE_FP16, [128], name="zeros_ub", scope=tik.scope_ubuf)
        tik_instance.vector_dup(128, zeros_ub[0], 0, 1, 1, 8)  # use for mask less than 128 only
        tik_instance.vsel(128, 0, score_to_filter[iter_t * 128], cmp_mask, score_to_filter[iter_t * 128], zeros_ub[0],
                          1, 1, 1, 1, 1, 1)

    with tik_instance.if_scope(left_data > 0):
        tik_instance.vector_dup(left_data, thresh_ub[score_size - left_data], cmp_value, 1, 1, 8)
        cmp_mask = tik_instance.vcmp_ge(left_data, cmp_ub[score_size - left_data], thresh_ub[score_size - left_data], 1,
                                        1)  # cmpmask
        zeros_ub = tik_instance.Tensor(Constant.DTYPE_FP16, [left_data], name="zeros_ub", scope=tik.scope_ubuf)
        tik_instance.vector_dup(left_data, zeros_ub[0], 0, 1, 1, 8)  # use for mask less than 128 only
        tik_instance.vsel(left_data, 0, score_to_filter[score_size - left_data], cmp_mask,
                          score_to_filter[score_size - left_data], zeros_ub[0], 1, 1, 1, 1, 1, 1)


def tik_func_vexp(tik_instance, dst_ub, src_ub, do_len):
    """
    tik_func_vexp
    """
    repeat = do_len // 128
    repeat_tail = do_len % 128
    src_offset = ub_offset(src_ub)
    dst_offset = ub_offset(dst_ub)
    while repeat > Constant.MAX_REPEAT_TIMES:
        tik_instance.vexp(128, dst_ub[dst_offset], src_ub[src_offset], 255, 1, 1, 8, 8)
        repeat = repeat - Constant.MAX_REPEAT_TIMES
        src_offset = src_offset + 128 * Constant.MAX_REPEAT_TIMES
        dst_offset = dst_offset + 128 * Constant.MAX_REPEAT_TIMES
    if repeat > 0:
        tik_instance.vexp(128, dst_ub[dst_offset], src_ub[src_offset], repeat, 1, 1, 8, 8)
        src_offset = src_offset + 128 * repeat
        dst_offset = dst_offset + 128 * repeat
    if repeat_tail > 0:
        tik_instance.vexp(repeat_tail, dst_ub[dst_offset], src_ub[src_offset], 1, 1, 1, 8, 8)


def tik_func_vexp_scalar(tik_instance, dst_ub, src_ub, do_len):
    """
    tik_func_vexp_scalar
    """
    repeat = tik_instance.Scalar(dtype=Constant.DTYPE_INT32)
    repeat.set_as(do_len // 128)
    repeat_tail = tik_instance.Scalar(dtype=Constant.DTYPE_INT32)
    repeat_tail.set_as(do_len % 128)
    src_offset = tik_instance.Scalar(dtype=Constant.DTYPE_INT32)
    src_offset.set_as(ub_offset(src_ub))
    dst_offset = tik_instance.Scalar(dtype=Constant.DTYPE_INT32)
    dst_offset.set_as(ub_offset(dst_ub))
    with tik_instance.if_scope(repeat // Constant.MAX_REPEAT_TIMES > 0):
        with tik_instance.for_range(0, repeat // Constant.MAX_REPEAT_TIMES) as i:
            tik_instance.vexp(128, dst_ub[dst_offset], src_ub[src_offset], 255, 1, 1, 8, 8)
            repeat.set_as(repeat - Constant.MAX_REPEAT_TIMES)
            src_offset.set_as(src_offset + 128 * Constant.MAX_REPEAT_TIMES)
            dst_offset.set_as(dst_offset + 128 * Constant.MAX_REPEAT_TIMES)
    with tik_instance.if_scope(repeat > 0):
        tik_instance.vexp(128, dst_ub[dst_offset], src_ub[src_offset], repeat, 1, 1, 8, 8)
        src_offset.set_as(src_offset + 128 * repeat)
        dst_offset.set_as(dst_offset + 128 * repeat)
    with tik_instance.if_scope(repeat_tail > 0):
        tik_instance.vexp(repeat_tail, dst_ub[dst_offset], src_ub[src_offset], 1, 1, 1, 8, 8)


def tik_func_vmuls_scalar(tik_instance, dst_ub, src_ub, value, do_len):
    """
    tik_func_vmuls_scalar
    """
    repeat = tik_instance.Scalar(dtype=Constant.DTYPE_INT32)
    repeat.set_as(do_len // 128)
    repeat_tail = tik_instance.Scalar(dtype=Constant.DTYPE_INT32)
    repeat_tail.set_as(do_len % 128)
    src_offset = tik_instance.Scalar(dtype=Constant.DTYPE_INT32)
    src_offset.set_as(ub_offset(src_ub))
    dst_offset = tik_instance.Scalar(dtype=Constant.DTYPE_INT32)
    dst_offset.set_as(ub_offset(dst_ub))
    with tik_instance.if_scope(repeat // Constant.MAX_REPEAT_TIMES > 0):
        with tik_instance.for_range(0, repeat // Constant.MAX_REPEAT_TIMES) as i:
            tik_instance.vmuls(128, dst_ub[dst_offset], src_ub[src_offset], value, 255, 1, 1, 8, 8)
            repeat.set_as(repeat - Constant.MAX_REPEAT_TIMES)
            src_offset.set_as(src_offset + 128 * Constant.MAX_REPEAT_TIMES)
            dst_offset.set_as(dst_offset + 128 * Constant.MAX_REPEAT_TIMES)
    with tik_instance.if_scope(repeat > 0):
        tik_instance.vmuls(128, dst_ub[dst_offset], src_ub[src_offset], value, repeat, 1, 1, 8, 8)
        src_offset.set_as(src_offset + 128 * repeat)
        dst_offset.set_as(dst_offset + 128 * repeat)
    with tik_instance.if_scope(repeat_tail > 0):
        tik_instance.vmuls(repeat_tail, dst_ub[dst_offset], src_ub[src_offset], value, 1, 1, 1, 8, 8)


def tik_func_vcomple_scalar(tik_instance, function, out_dst, src0, src1, copy_num, dst_blk=1, src0_blk=1,
                            src1_blk=1, dst_rep=8, src0_rep=8, src1_rep=8):
    """
    tik_func_vcomple_scalar
    """
    do_dtype = out_dst.dtype
    if do_dtype in ("float16",):
        block_num = Constant.BLOCK_BYTE_SIZE // Constant.BYTE_FP16
    else:
        block_num = Constant.BLOCK_BYTE_SIZE // Constant.BYTE_FP32
    vector_num = block_num * 8
    repeat_time = tik_instance.Scalar(dtype='int32', name='repeat_time')
    repeat_time.set_as(copy_num // vector_num)
    repeat_tail = tik_instance.Scalar(dtype='int32', name='repeat_tail')
    repeat_tail.set_as(copy_num % vector_num)
    tik_fun = None
    ori_offset_dst = tik_instance.Scalar(dtype='int32', name='ori_offset_dst')
    ori_offset_dst.set_as(ub_offset(out_dst))
    ori_offset_src0 = tik_instance.Scalar(dtype='int32', name='ori_offset_src0')
    ori_offset_src0.set_as(ub_offset(src0))
    ori_offset_src1 = tik_instance.Scalar(dtype='int32', name='ori_offset_src1')
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

    with tik_instance.if_scope(repeat_time // Constant.MAX_REPEAT_TIMES > 0):
        with tik_instance.for_range(0, repeat_time // Constant.MAX_REPEAT_TIMES) as i:
            tik_fun(vector_num, out_dst[ori_offset_dst], src0[ori_offset_src0], src1[ori_offset_src1],
                    255, dst_blk, src0_blk, src1_blk, dst_rep, src0_rep, src1_rep)
            repeat_time.set_as(repeat_time - Constant.MAX_REPEAT_TIMES)
            ori_offset_dst.set_as(ori_offset_dst + Constant.MAX_REPEAT_TIMES * block_num * dst_rep)
            ori_offset_src0.set_as(ori_offset_src0 + Constant.MAX_REPEAT_TIMES * block_num * src0_rep)
            ori_offset_src1.set_as(ori_offset_src1 + Constant.MAX_REPEAT_TIMES * block_num * src1_rep)

    with tik_instance.if_scope(repeat_time > 0):
        tik_fun(vector_num, out_dst[ori_offset_dst], src0[ori_offset_src0], src1[ori_offset_src1],
                repeat_time, dst_blk, src0_blk, src1_blk, dst_rep, src0_rep, src1_rep)
        ori_offset_dst.set_as(ori_offset_dst + repeat_time * block_num * dst_rep)
        ori_offset_src0.set_as(ori_offset_src0 + repeat_time * block_num * src0_rep)
        ori_offset_src1.set_as(ori_offset_src1 + repeat_time * block_num * src1_rep)

    with tik_instance.if_scope(repeat_tail > 0):
        tik_fun(repeat_tail, out_dst[ori_offset_dst], src0[ori_offset_src0], src1[ori_offset_src1],
                1, dst_blk, src0_blk, src1_blk, dst_rep, src0_rep, src1_rep)


class GenerateBoundingBoxProposalsCompute:
    """
    generate_bounding_box_proposals
    """

    def __init__(self, scores, bbox_deltas, image_info, anchors, nms_threshold, pre_nms_topn, min_size, rois,
                 rois_probabilities, post_nms_topn, kernel_name):
        self.tik_instance = tik.Tik()
        self.kernel_name = kernel_name
        self.dtype_scores = scores.get("dtype").lower()
        self.data_scores = self.tik_instance.Tensor(self.dtype_scores, (Constant.MAX_INT32_VALUE,),
                                                    name="data_scores", scope=tik.scope_gm)
        self.data_bbox_deltas = self.tik_instance.Tensor(self.dtype_scores, (Constant.MAX_INT32_VALUE,),
                                                         name="data_bbox_deltas", scope=tik.scope_gm)
        self.data_image_info = self.tik_instance.Tensor(self.dtype_scores, (Constant.MAX_INT32_VALUE,),
                                                        name="data_image_info", scope=tik.scope_gm)
        self.data_anchors = self.tik_instance.Tensor(self.dtype_scores, (Constant.MAX_INT32_VALUE,),
                                                     name="data_anchors", scope=tik.scope_gm)
        self.data_nms_threshold = self.tik_instance.Tensor(self.dtype_scores, (Constant.MAX_INT32_VALUE,),
                                                           name="data_nms_threshold", scope=tik.scope_gm)
        self.data_pre_nms_topn = self.tik_instance.Tensor(Constant.DTYPE_INT32, (Constant.MAX_INT32_VALUE,),
                                                          name="data_pre_nms_topn", scope=tik.scope_gm)
        self.data_min_size = self.tik_instance.Tensor(Constant.DTYPE_FP32, (Constant.MAX_INT32_VALUE,),
                                                      name="data_min_size", scope=tik.scope_gm)
        self.data_rois = self.tik_instance.Tensor(self.dtype_scores, (Constant.MAX_INT32_VALUE,),
                                                  name="data_rois", scope=tik.scope_gm)
        self.data_rois_probabilities = self.tik_instance.Tensor(self.dtype_scores, (Constant.MAX_INT32_VALUE,),
                                                                name="data_rois_probabilities", scope=tik.scope_gm)
        self.ai_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - Constant.RESERVED_UB
        # upper bound of encoded width and height
        self.bbox_xform_clip = math.log(1000.0 / 16.0)
        self.proposal_topk_k = self.ub_size // Constant.BOX_DIM_NUM // Constant.DATA_NUM_ONE_BLOCK
        self.proposal_topk_k = min(self.proposal_topk_k, Constant.MAX_NUM_MOVE_BOX)
        self.max_selected_nms_num_in_ub = self.tik_instance.Scalar(Constant.DTYPE_INT32,
                                                                   name="max_selected_nms_num_in_ub", init_value=0)
        self.post_nms_topn = post_nms_topn
        self.is_need_rpn_offset = False
        self.tiling_ub = None
        self.l1_nms_sup = None
        self.ub_max_topk = None
        self.l1_nms_area = None
        self.l1_nms_result = None
        self.l1_nms_result_zero = None
        self.workspace_proposal_gm = None
        self.workspace_second_nms_gm = None
        self.tiling_gm = self.tik_instance.Tensor(Constant.TILING_PARAM_DTYPE, (Constant.TILING_PARAMS_NUM,),
                                                  name="tiling_gm", scope=tik.scope_gm)
        self.max_size_topn = self.tik_instance.Scalar(dtype=Constant.DTYPE_INT32, name="max_size_topn",
                                                      init_value=post_nms_topn)
        self.need_core_num = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="need_core_num")
        self.image_num_each_core = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="image_offset_current_core")
        self.image_num_rest = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="image_num_last_core")
        self.image_height = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="image_height")
        self.image_width = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="image_width")
        self.num_anchors = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="num_anchors")
        self.nms_threshold = self.tik_instance.Scalar(self.dtype_scores, name="nms_threshold")
        self.pre_nms_topn = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="pre_nms_topn")
        self.min_size = self.tik_instance.Scalar(Constant.DTYPE_FP32, name="min_size")
        self.box_num_one_image = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="box_num_one_image")
        # record the output nms num
        self.selected_proposals_cnt = self.tik_instance.Scalar(dtype=Constant.DTYPE_UINT16, init_value=0)
        # record the proposal burst num for one loop, value = 128 or `self.proposal_topk_k % 128`
        self.handling_proposals_cnt = self.tik_instance.Scalar(dtype=Constant.DTYPE_UINT16, init_value=0)
        max_boxes_num = 10000
        self.workspace_proposal_gm = self.tik_instance.Tensor(
            self.dtype_scores, [self.ai_core_num, max_boxes_num + 128], name="workspace_proposal_gm",
            scope=tik.scope_gm, is_workspace=True)
        self.workspace_second_nms_gm = self.tik_instance.Tensor(
            self.dtype_scores, [self.ai_core_num, max_boxes_num + 128], name="workspace_second_nms_gm",
            scope=tik.scope_gm, is_workspace=True)

    def generate_bounding_box_proposals_compute(self):
        """
        The tik implementation of operator generate_bounding_box_proposals
        """
        self._init_tiling_param()
        self._get_const_input()
        self._init_tik_mem()
        with self.tik_instance.for_range(0, self.ai_core_num, block_num=self.ai_core_num) as core_index:
            with self.tik_instance.if_scope(core_index < self.need_core_num):
                self._run_one_core(core_index)
        tbe_context.get_context().add_compile_info("vars", {"ub_size": self.ub_size, "core_num": self.ai_core_num})
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.data_scores, self.data_bbox_deltas, self.data_image_info,
                                           self.data_anchors, self.data_nms_threshold, self.data_pre_nms_topn,
                                           self.data_min_size),
                                   outputs=(self.data_rois, self.data_rois_probabilities), flowtable=(self.tiling_gm,),
                                   output_files_path=None, enable_l2=False)

    def _get_tiling_params(self):
        """
        get runtime tiling parameters from tiling
        """
        # read tiling int64 scalar
        self.need_core_num.set_as(self.tiling_ub[0])
        self.image_num_each_core.set_as(self.tiling_ub[1])
        self.image_num_rest.set_as(self.tiling_ub[2])
        self.image_height.set_as(self.tiling_ub[3])
        self.image_width.set_as(self.tiling_ub[4])
        self.num_anchors.set_as(self.tiling_ub[5])

    def _init_tiling_param(self):
        """
        _init_tiling_param
        """
        with self.tik_instance.new_stmt_scope():
            self.tiling_ub = self.tik_instance.Tensor(Constant.TILING_PARAM_DTYPE, (Constant.TILING_PARAMS_NUM,),
                                                      name="tiling_ub", scope=tik.scope_ubuf)
            # mov tiling params from gm to ub
            self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1,
                                        Constant.TILING_PARAMS_NUM * Constant.BYTE_INT32 // Constant.BLOCK_BYTE_SIZE,
                                        0, 0)
            self._get_tiling_params()

    def _get_const_input(self):
        """
        _get_const_input
        """
        with self.tik_instance.new_stmt_scope():
            ub_nms_threshold = self.tik_instance.Tensor(self.dtype_scores, (16,), name="ub_nms_threshold",
                                                        scope=tik.scope_ubuf)
            self.tik_instance.data_move(ub_nms_threshold, self.data_nms_threshold, 0, 1, 1, 0, 0)
            self.nms_threshold.set_as(ub_nms_threshold[0])
            ub_min_size = self.tik_instance.Tensor(Constant.DTYPE_FP32, (8,), name="ub_min_size", scope=tik.scope_ubuf)
            self.tik_instance.data_move(ub_min_size, self.data_min_size, 0, 1, 1, 0, 0)
            self.min_size.set_as(ub_min_size[0])
            ub_pre_nms_topn = self.tik_instance.Tensor(Constant.DTYPE_INT32, (8,), name="ub_pre_nms_topn",
                                                       scope=tik.scope_ubuf)
            self.tik_instance.data_move(ub_pre_nms_topn, self.data_pre_nms_topn, 0, 1, 1, 0, 0)
            self.pre_nms_topn.set_as(ub_pre_nms_topn[0])
            with self.tik_instance.if_scope(self.max_size_topn > self.pre_nms_topn):
                self.max_size_topn.set_as(self.pre_nms_topn)

    def _init_tik_mem(self):
        """
        _init_tik_mem
        """
        # for nms function param calc
        self.box_num_one_image.set_as(self.image_width * self.image_height * self.num_anchors)
        self.max_selected_nms_num_in_ub.set_as(
            ceil_div(self.max_size_topn, Constant.RPN_PROPOSAL_NUM) * Constant.RPN_PROPOSAL_NUM)
        self.l1_nms_result = self.tik_instance.Tensor(
            self.dtype_scores, (self.max_selected_nms_num_in_ub, 8), name="l1_nms_result", scope=tik.scope_cbuf)

        # init l1 buff for save nms area, size = `[self.max_selected_nms_num_in_ub]`
        self.l1_nms_area = self.tik_instance.Tensor(self.dtype_scores, (self.max_selected_nms_num_in_ub,),
                                                    name="l1_nms_area_tmp", scope=tik.scope_cbuf)
        # init l1 buff for save nms sup, size = `[self.max_selected_nms_num_in_ub]`
        self.l1_nms_sup = self.tik_instance.Tensor(Constant.DTYPE_UINT16, (self.max_selected_nms_num_in_ub,),
                                                   name="l1_nms_sup_tmp", scope=tik.scope_cbuf)
        # zero data in l1
        self.l1_nms_result_zero = self.tik_instance.Tensor(self.dtype_scores, (self.max_selected_nms_num_in_ub, 8),
                                                           name="l1_nms_result_zero", scope=tik.scope_cbuf)
        with self.tik_instance.new_stmt_scope():
            ub_nms_result = self.tik_instance.Tensor(self.dtype_scores, (self.max_selected_nms_num_in_ub, 8),
                                                     name="ub_nms_result", scope=tik.scope_ubuf)
            tik_func_vector_scalar(self.tik_instance, ub_nms_result, 0, self.max_selected_nms_num_in_ub * 8)
            loop_burst_len = (self.max_selected_nms_num_in_ub * 8) // 16
            self.tik_instance.data_move(self.l1_nms_result_zero, ub_nms_result, 0, 1, loop_burst_len, 0, 0)

    def _init_tik_ub_mem_for_topk(self):
        """
        init_tik_ub_mem_for_topk
        """
        # init one ub for topk output
        self.ub_max_topk = self.tik_instance.Tensor(
            self.dtype_scores, (self.proposal_topk_k, 8), name="ub_max_topk", scope=tik.scope_ubuf)

    def _init_tik_ub_mem_for_nms(self):
        """
        init_tik_ub_mem_for_nms
        """
        ub_selected_proposals = self.tik_instance.Tensor(
            self.dtype_scores, [self.max_selected_nms_num_in_ub, 8], name="ub_selected_proposals",
            scope=tik.scope_ubuf)
        ub_selected_area = self.tik_instance.Tensor(
            self.dtype_scores, [self.max_selected_nms_num_in_ub], name="ub_selected_area", scope=tik.scope_ubuf)
        ub_sup_vec = self.tik_instance.Tensor(
            Constant.DTYPE_UINT16, [self.max_selected_nms_num_in_ub], name="ub_sup_vec", scope=tik.scope_ubuf)
        if self.is_need_rpn_offset:
            self.tik_instance.set_rpn_offset(1.0)
        topk_out_num = self.tik_instance.Scalar(dtype=Constant.DTYPE_INT32, name='topk_out_num')
        topk_out_num.set_as(self.proposal_topk_k)
        # init a scalar value = 0
        zero_scalar = self.tik_instance.Scalar(dtype=Constant.DTYPE_UINT16, init_value=0)
        one_scalar = self.tik_instance.Scalar(dtype=Constant.DTYPE_UINT16, init_value=1)
        with self.tik_instance.if_scope(self.box_num_one_image < self.proposal_topk_k):
            topk_out_num.set_as(self.box_num_one_image)
        nms_var_dict = {
            # topk_out_info mean : nms input info
            "topk_out_ub": self.ub_max_topk,
            "topk_out_num": topk_out_num,
            # selected proposal info
            "selected_proposal_ub": ub_selected_proposals,
            "selected_area_ub": ub_selected_area,
            "sup_vec_ub": ub_sup_vec,
            # scalar reg info
            "zero_scalar": zero_scalar,
            "one_scalar": one_scalar,
            "selected_proposals_cnt": self.selected_proposals_cnt,
            "handling_proposals_cnt": self.handling_proposals_cnt,
            # nms output info
            "output_num": self.max_size_topn
        }

        return nms_var_dict

    def _mov_data_gm2ub(self, input_ub_list, gm_offset, score_size):
        """
        _mov_data_gm2ub
        """
        ub_boxes = input_ub_list[0]
        ub_scores = input_ub_list[1]
        ub_bbox_delta = input_ub_list[2]
        ub_image_info = input_ub_list[3]
        image_offset = gm_offset[0]
        anchor_offset = gm_offset[1]
        box_front = gm_offset[2]
        burst_len = self.tik_instance.Scalar(Constant.DTYPE_INT32, name='burst_len')
        burst_len.set_as((score_size - 1) // Constant.DATA_NUM_ONE_BLOCK + 1)
        for i in range(Constant.BOX_DIM_NUM):
            self.tik_instance.data_move(ub_boxes[i, 0], self.data_anchors[self.box_num_one_image * i + anchor_offset],
                                        0, 1, burst_len, 0, 0)
            self.tik_instance.data_move(
                ub_bbox_delta[i, 0], self.data_bbox_deltas[box_front * 4 + self.box_num_one_image * i + anchor_offset],
                0, 1, burst_len, 0, 0)
        self.tik_instance.data_move(ub_scores, self.data_scores[box_front + anchor_offset], 0, 1, burst_len, 0, 0)
        self.tik_instance.data_move(ub_image_info, self.data_image_info[image_offset * 5], 0, 1, 1, 0, 0)

    def _get_predict_axis(self, pred_ub_list, func_list, compute_num):
        """
        _get_predict_axis
        """
        ub_box_ctr = self.tik_instance.Tensor(self.dtype_scores, [compute_num], name="ub_box_ctr", scope=tik.scope_ubuf)
        func_list[1](self.tik_instance, "vsub", pred_ub_list[1], pred_ub_list[1], pred_ub_list[0], compute_num)
        func_list[2](self.tik_instance, ub_box_ctr, pred_ub_list[1], 0.5, compute_num)
        ub_tmp_pred = self.tik_instance.Tensor(self.dtype_scores, [compute_num], name="ub_tmp_pred",
                                               scope=tik.scope_ubuf)
        func_list[1](self.tik_instance, "vadd", ub_tmp_pred, pred_ub_list[0], ub_box_ctr, compute_num)
        func_list[1](self.tik_instance, "vmul", pred_ub_list[2], pred_ub_list[1], pred_ub_list[2], compute_num)
        func_list[1](self.tik_instance, "vadd", pred_ub_list[2], ub_tmp_pred, pred_ub_list[2], compute_num)
        func_list[3](self.tik_instance, pred_ub_list[3], pred_ub_list[3], compute_num)
        func_list[1](self.tik_instance, "vmul", ub_tmp_pred, pred_ub_list[1], pred_ub_list[3], compute_num)
        func_list[2](self.tik_instance, ub_tmp_pred, ub_tmp_pred, 0.5, compute_num)
        func_list[1](self.tik_instance, "vsub", pred_ub_list[0], pred_ub_list[2], ub_tmp_pred, compute_num)
        func_list[1](self.tik_instance, "vadd", pred_ub_list[1], pred_ub_list[2], ub_tmp_pred, compute_num)

    def _prenms(self, input_ub_list, score_size, func_list):
        """
        _prenms
        """
        ub_boxes = input_ub_list[0]
        ub_scores = input_ub_list[1]
        ub_bbox_delta = input_ub_list[2]
        ub_image_info = input_ub_list[3]
        ub_box_y1, ub_box_x1, ub_box_y2, ub_box_x2 = [ub_boxes[0, :], ub_boxes[1, :], ub_boxes[2, :], ub_boxes[3, :]]
        ub_delta_y, ub_delta_x, ub_delta_h, ub_delta_w = [ub_bbox_delta[0, :], ub_bbox_delta[1, :], ub_bbox_delta[2, :],
                                                          ub_bbox_delta[3, :]]
        ub_tmp = self.tik_instance.Tensor(self.dtype_scores, [score_size], name="ub_tmp", scope=tik.scope_ubuf)
        func_list[0](self.tik_instance, ub_tmp, self.bbox_xform_clip, score_size)
        func_list[1](self.tik_instance, "vmin", ub_delta_h, ub_delta_h, ub_tmp, score_size, 1, 1, 0, 8, 8, 0)
        func_list[1](self.tik_instance, "vmin", ub_delta_w, ub_delta_w, ub_tmp, score_size, 1, 1, 0, 8, 8, 0)
        with self.tik_instance.new_stmt_scope():
            pred_ub_x_list = [ub_box_x1, ub_box_x2, ub_delta_x, ub_delta_w]
            self._get_predict_axis(pred_ub_x_list, func_list, score_size)
        with self.tik_instance.new_stmt_scope():
            pred_ub_y_list = [ub_box_y1, ub_box_y2, ub_delta_y, ub_delta_h]
            self._get_predict_axis(pred_ub_y_list, func_list, score_size)

        image_height = self.tik_instance.Scalar(dtype=self.dtype_scores)
        image_height.set_as(ub_image_info[0])
        image_width = self.tik_instance.Scalar(dtype=self.dtype_scores)
        image_width.set_as(ub_image_info[1])
        image_scale = self.tik_instance.Scalar(dtype=self.dtype_scores)
        image_scale.set_as(ub_image_info[2])
        min_size_scaled = self.tik_instance.Scalar(dtype=self.dtype_scores)
        min_size_scaled.set_as(image_scale * self.min_size)
        func_list[0](self.tik_instance, ub_tmp, image_width, score_size)
        func_list[1](self.tik_instance, "vmin", ub_box_x1, ub_box_x1, ub_tmp, score_size, 1, 1, 0, 8, 8, 0)
        func_list[1](self.tik_instance, "vmin", ub_box_x2, ub_box_x2, ub_tmp, score_size, 1, 1, 0, 8, 8, 0)
        func_list[0](self.tik_instance, ub_tmp, image_height, score_size)
        func_list[1](self.tik_instance, "vmin", ub_box_y1, ub_box_y1, ub_tmp, score_size, 1, 1, 0, 8, 8, 0)
        func_list[1](self.tik_instance, "vmin", ub_box_y2, ub_box_y2, ub_tmp, score_size, 1, 1, 0, 8, 8, 0)
        func_list[0](self.tik_instance, ub_tmp, 0, score_size)
        func_list[1](self.tik_instance, "vmax", ub_box_x1, ub_box_x1, ub_tmp, score_size, 1, 1, 0, 8, 8, 0)
        func_list[1](self.tik_instance, "vmax", ub_box_y1, ub_box_y1, ub_tmp, score_size, 1, 1, 0, 8, 8, 0)
        func_list[1](self.tik_instance, "vmax", ub_box_x2, ub_box_x2, ub_tmp, score_size, 1, 1, 0, 8, 8, 0)
        func_list[1](self.tik_instance, "vmax", ub_box_y2, ub_box_y2, ub_tmp, score_size, 1, 1, 0, 8, 8, 0)
        func_list[1](self.tik_instance, "vsub", ub_delta_x, ub_box_x2, ub_box_x1, score_size)
        func_list[1](self.tik_instance, "vsub", ub_delta_y, ub_box_y2, ub_box_y1, score_size)
        func_list[1](self.tik_instance, "vmin", ub_tmp, ub_delta_x, ub_delta_y, score_size, 1, 1, 0, 8, 8, 0)
        with self.tik_instance.new_stmt_scope():
            func_list[4](self.tik_instance, ub_scores, ub_tmp, min_size_scaled, score_size)

    def _copy_nms_result_to_l1(self, nmsed_result_ub, nmsed_result_area, nmsed_result_sup):
        """
        _copy_nms_result_to_l1
        """
        # copy nms result to l1
        l1_buffer = self.l1_nms_result
        l1_offset = [0, 0]
        loop_burst_len = (self.max_selected_nms_num_in_ub * 8) // Constant.DATA_NUM_ONE_BLOCK
        self.tik_instance.data_move(l1_buffer[l1_offset], nmsed_result_ub, 0, 1, loop_burst_len, 0, 0)
        loop_burst_len = self.max_selected_nms_num_in_ub // Constant.DATA_NUM_ONE_BLOCK
        self.tik_instance.data_move(self.l1_nms_area, nmsed_result_area, 0, 1, loop_burst_len, 0, 0)
        self.tik_instance.data_move(self.l1_nms_sup, nmsed_result_sup, 0, 1, loop_burst_len, 0, 0)

    def _do_nms_first_time(self):
        """
        _do_nms_first_time
        """
        with self.tik_instance.new_stmt_scope():
            nms_var = self._init_tik_ub_mem_for_nms()
            nmsed_result_ub = nms_var.get("selected_proposal_ub")
            nmsed_result_area = nms_var.get("selected_area_ub")
            nmsed_result_sup = nms_var.get("sup_vec_ub")
            # init all sup_vec to 1, mean: no select proposal
            tik_func_vector_scalar(self.tik_instance, nmsed_result_sup, 1, self.max_selected_nms_num_in_ub)
            # init select nms proposal = 0
            l1_buffer = self.l1_nms_result_zero
            loop_burst_len = (self.max_selected_nms_num_in_ub * 8) // Constant.DATA_NUM_ONE_BLOCK
            self.tik_instance.data_move(nmsed_result_ub, l1_buffer, 0, 1, loop_burst_len, 0, 0)
            # init select nms area = 0
            loop_burst_len = self.max_selected_nms_num_in_ub // Constant.DATA_NUM_ONE_BLOCK
            self.tik_instance.data_move(nmsed_result_area, l1_buffer, 0, 1, loop_burst_len, 0, 0)
            with self.tik_instance.new_stmt_scope():
                do_nms_compute(self.tik_instance, nms_var, self.nms_threshold)
            self._copy_nms_result_to_l1(nmsed_result_ub, nmsed_result_area, nmsed_result_sup)

    def _do_nms_continue(self):
        """
        _do_nms_continue
        """
        with self.tik_instance.new_stmt_scope():
            nms_var = self._init_tik_ub_mem_for_nms()
            nmsed_result_ub = nms_var.get("selected_proposal_ub")
            nmsed_result_area = nms_var.get("selected_area_ub")
            nmsed_result_sup = nms_var.get("sup_vec_ub")

            # copy l1 tmp data to ub
            l1_buffer = self.l1_nms_result
            l1_offset = [0, 0]
            loop_burst_len = (self.max_selected_nms_num_in_ub * 8) // Constant.DATA_NUM_ONE_BLOCK
            # copy the selected proposal/area/sup_ub from L1 to UB
            self.tik_instance.data_move(nmsed_result_ub, l1_buffer[l1_offset], 0, 1, loop_burst_len, 0, 0)
            loop_burst_len = self.max_selected_nms_num_in_ub // Constant.DATA_NUM_ONE_BLOCK
            self.tik_instance.data_move(nmsed_result_area, self.l1_nms_area, 0, 1, loop_burst_len, 0, 0)
            self.tik_instance.data_move(nmsed_result_sup, self.l1_nms_sup, 0, 1, loop_burst_len, 0, 0)

            with self.tik_instance.new_stmt_scope():
                do_nms_compute(self.tik_instance, nms_var, self.nms_threshold)
            self._copy_nms_result_to_l1(nmsed_result_ub, nmsed_result_area, nmsed_result_sup)

    def _nms_handle(self, box_mov_times, box_left, box_num_front, sorted_k, ub_proposal):
        """
        _nms_handle
        """
        self._do_nms_first_time()
        tool_loop = self.tik_instance.Scalar(dtype=Constant.DTYPE_INT32, init_value=0)
        with self.tik_instance.if_scope(box_left == 0):
            tool_loop.set_as(box_mov_times)
        with self.tik_instance.else_scope():
            tool_loop.set_as(box_mov_times + 1)
        with self.tik_instance.if_scope(tool_loop >= 3):
            # if not to output num
            with self.tik_instance.for_range(1, tool_loop - 1) as _top_n_idx:
                top_n_num_tail = self.tik_instance.Scalar(dtype=Constant.DTYPE_INT32)
                top_n_num_tail.set_as(tool_loop - _top_n_idx - 1)
                with self.tik_instance.if_scope(self.selected_proposals_cnt < self.max_size_topn):
                    # copy a sorted proposals to topk_out_ub
                    loop_burst_len = ceil_div(sorted_k * 8, Constant.DATA_NUM_ONE_BLOCK)
                    self.tik_instance.data_move(
                        ub_proposal, self.workspace_second_nms_gm[box_num_front * 8], 0, 1, loop_burst_len, 0, 0)
                    # apply second top k proposal ub
                    with self.tik_instance.new_stmt_scope():
                        ub_tmp_topk = self.tik_instance.Tensor(
                            self.dtype_scores, ub_proposal.shape, name="ub_tmp_topk", scope=tik.scope_ubuf)
                        with self.tik_instance.for_range(0, top_n_num_tail) as _top_n_tail_idx:
                            workspace_proposal_offset = self.tik_instance.Scalar(
                                dtype=Constant.DTYPE_INT32, name='workspace_proposal_offset')
                            workspace_proposal_offset.set_as(
                                sorted_k * 8 + _top_n_tail_idx * sorted_k * 8 + box_num_front * 8)
                            self.tik_instance.data_move(
                                ub_tmp_topk, self.workspace_second_nms_gm[workspace_proposal_offset],
                                0, 1, loop_burst_len, 0, 0)
                            workspace_offset = self.tik_instance.Scalar(dtype=Constant.DTYPE_INT32,
                                                                        name='workspace_offset')
                            workspace_offset.set_as(_top_n_tail_idx * sorted_k * 8 + box_num_front * 8)
                            workspace_for_save_proposal = self.workspace_second_nms_gm[workspace_offset]
                            # sorted two proposals to one proposal list output the top sorted_k
                            tik_func_sort_with_ub(self.tik_instance, [ub_tmp_topk, ub_proposal],
                                                  [ub_proposal, ub_tmp_topk], sorted_k, workspace_for_save_proposal)
                    # do nms use topk output to get nms proposals and move result to l1
                    self._do_nms_continue()

    def _get_sorted_proposal(self, ub_proposal, score_size, sorted_size, gm_offset):
        """
        _get_sorted_proposal
        """
        score_size_align = ceil_div(score_size, Constant.DATA_NUM_ONE_BLOCK) * Constant.DATA_NUM_ONE_BLOCK
        with self.tik_instance.new_stmt_scope():
            ub_scores = self.tik_instance.Tensor(self.dtype_scores, [1, score_size_align], name="ub_scores",
                                                 scope=tik.scope_ubuf)
            ub_boxes = self.tik_instance.Tensor(self.dtype_scores, [4, score_size_align], name="ub_boxes",
                                                scope=tik.scope_ubuf)
            ub_bbox_delta = self.tik_instance.Tensor(self.dtype_scores, [4, score_size_align], name="ub_bbox_delta",
                                                     scope=tik.scope_ubuf)
            ub_image_info = self.tik_instance.Tensor(self.dtype_scores, [1, score_size_align], name="ub_image_info",
                                                     scope=tik.scope_ubuf)
            input_ub_list = [ub_boxes, ub_scores, ub_bbox_delta, ub_image_info]
            self._mov_data_gm2ub(input_ub_list, gm_offset, score_size)

            # set the tail data = 0
            last_block_data_num = score_size % Constant.DATA_NUM_ONE_BLOCK
            if last_block_data_num != 0:
                erase_data_num = Constant.DATA_NUM_ONE_BLOCK - last_block_data_num
                for index_tail in range(erase_data_num):
                    ub_scores[score_size + index_tail] = 0
                    for i in range(Constant.BOX_DIM_NUM):
                        ub_boxes[i, score_size + index_tail] = 0
                        ub_bbox_delta[i, score_size + index_tail] = 0

            # pre-nms
            with self.tik_instance.new_stmt_scope():
                func_list = [tik_func_vector, tik_func_vcomple, tik_func_vmuls, tik_func_vexp, filter_greater_than]
                self._prenms(input_ub_list, score_size, func_list)

            if self.is_need_rpn_offset:
                tik_func_vadds(self.tik_instance, ub_boxes[2, :], ub_boxes[2, :], -1.0, score_size)
                tik_func_vadds(self.tik_instance, ub_boxes[3, :], ub_boxes[3, :], -1.0, score_size)

            # trans to proposal
            boxes_list = [ub_boxes[1, :], ub_boxes[0, :], ub_boxes[3, :], ub_boxes[2, :]]
            tik_func_trans_to_proposals(self.tik_instance, ub_proposal, boxes_list, ub_scores, score_size)

            # sort in ub
        sort_within_ub(self.tik_instance, ub_proposal, score_size_align)
        if score_size_align != sorted_size:
            dup_len = (sorted_size - score_size_align)
            offset = score_size_align * 8
            tik_func_vector_scalar(self.tik_instance, ub_proposal[offset:], 0.0, dup_len * 8)

    def _get_sorted_proposal_scalar(self, ub_proposal, score_size, sorted_size, gm_offset):
        """
        _get_sorted_proposal_scalar
        """
        score_size_align = ceil_div(score_size, Constant.DATA_NUM_ONE_BLOCK) * Constant.DATA_NUM_ONE_BLOCK
        with self.tik_instance.new_stmt_scope():
            ub_scores = self.tik_instance.Tensor(self.dtype_scores, [1, score_size_align], name="ub_scores",
                                                 scope=tik.scope_ubuf)
            ub_boxes = self.tik_instance.Tensor(self.dtype_scores, [4, score_size_align], name="ub_boxes",
                                                scope=tik.scope_ubuf)
            ub_bbox_delta = self.tik_instance.Tensor(self.dtype_scores, [4, score_size_align], name="ub_bbox_delta",
                                                     scope=tik.scope_ubuf)
            ub_image_info = self.tik_instance.Tensor(self.dtype_scores, [1, score_size_align], name="ub_image_info",
                                                     scope=tik.scope_ubuf)
            input_ub_list = [ub_boxes, ub_scores, ub_bbox_delta, ub_image_info]
            self._mov_data_gm2ub(input_ub_list, gm_offset, score_size)

            # set the tail data = 0
            last_block_data_num = self.tik_instance.Scalar(dtype=Constant.DTYPE_INT32, init_value=0)
            last_block_data_num.set_as(score_size % Constant.DATA_NUM_ONE_BLOCK)
            with self.tik_instance.if_scope(last_block_data_num != 0):
                erase_data_num = Constant.DATA_NUM_ONE_BLOCK - last_block_data_num
                with self.tik_instance.for_range(0, erase_data_num) as index_tail:
                    ub_scores[score_size + index_tail].set_as(0)
                    with self.tik_instance.for_range(0, Constant.BOX_DIM_NUM) as i:
                        ub_boxes[i, score_size + index_tail].set_as(0)
                        ub_bbox_delta[i, score_size + index_tail].set_as(0)

            # pre-nms
            with self.tik_instance.new_stmt_scope():
                func_list = [tik_func_vector_scalar, tik_func_vcomple_scalar, tik_func_vmuls_scalar,
                             tik_func_vexp_scalar, filter_greater_than_scalar]
                self._prenms(input_ub_list, score_size, func_list)

            with self.tik_instance.if_scope(self.is_need_rpn_offset):
                tik_func_vadds_scalar(self.tik_instance, ub_boxes[2, :], ub_boxes[2, :], -1.0, score_size)
                tik_func_vadds_scalar(self.tik_instance, ub_boxes[3, :], ub_boxes[3, :], -1.0, score_size)

            # trans to proposal
            boxes_list = [ub_boxes[1, :], ub_boxes[0, :], ub_boxes[3, :], ub_boxes[2, :]]
            tik_func_trans_to_proposals(self.tik_instance, ub_proposal, boxes_list, ub_scores, score_size)

        # sort in ub
        level = self.tik_instance.Scalar(dtype=Constant.DTYPE_INT32, init_value=1)
        region = self.tik_instance.Scalar(dtype=Constant.DTYPE_INT32)
        region.set_as((((score_size - 1) // 16 + 1) * 16 + 15) // 16)
        with self.tik_instance.if_scope(region > 1):
            with self.tik_instance.for_range(0, Constant.RPN_PROPOSAL_NUM) as i:
                with self.tik_instance.if_scope(region > 1):
                    level.set_as(level + 1)
                    region.set_as((region + 3) // 4)
        sort_within_ub_scalar(self.tik_instance, ub_proposal, score_size_align, level)
        with self.tik_instance.if_scope(((score_size - 1) // 16 + 1) * 16 != sorted_size):
            dup_len = self.tik_instance.Scalar(dtype=Constant.DTYPE_INT32)
            dup_len.set_as(sorted_size - ((score_size - 1) // 16 + 1) * 16)
            offset = self.tik_instance.Scalar(dtype=Constant.DTYPE_INT32)
            offset.set_as(((score_size - 1) // 16 + 1) * 16 * 8)
            tik_func_vector_scalar(self.tik_instance, ub_proposal[offset:], 0.0, dup_len * 8)

    def _run_image_loop(self, image_offset):
        """
        _run_image_loop
        """
        self._init_tik_ub_mem_for_topk()
        ub_proposal = self.ub_max_topk
        box_num_front = image_offset * self.box_num_one_image
        box_mov_times = self.box_num_one_image // self.proposal_topk_k
        box_left = self.box_num_one_image % self.proposal_topk_k
        sorted_k = self.proposal_topk_k
        self.selected_proposals_cnt.set_as(0)
        anchor_offset = self.tik_instance.Scalar(dtype=Constant.DTYPE_INT32, name='anchor_offset', init_value=0)

        # total boxes per batch less than 4096，sort all and do nms, then output all
        with self.tik_instance.if_scope(self.box_num_one_image <= self.proposal_topk_k):
            gm_offset = [image_offset, anchor_offset, box_num_front]
            self._get_sorted_proposal_scalar(ub_proposal, self.box_num_one_image, sorted_k, gm_offset)
        # total boxes per batch greater than 4096，depart as 4096，sort each part and get the top 4096 to do nms，if the
        # output less than pre_nms_topn，get the less top 4096 to do nms
        with self.tik_instance.else_scope():
            with self.tik_instance.new_stmt_scope():
                score_size = self.proposal_topk_k
                gm_offset = [image_offset, anchor_offset, box_num_front]
                self._get_sorted_proposal(ub_proposal, score_size, sorted_k, gm_offset)
                tmp_ub_proposal = self.tik_instance.Tensor(
                    self.dtype_scores, (self.proposal_topk_k, 8), name="tmp_ub_proposal", scope=tik.scope_ubuf)
                workspace_offset = self.tik_instance.Scalar(dtype=Constant.DTYPE_INT32, name='workspace_offset')
                with self.tik_instance.if_scope(box_mov_times > 1):
                    with self.tik_instance.for_range(1, box_mov_times) as index:
                        anchor_offset.set_as(index * self.proposal_topk_k)
                        gm_offset = [image_offset, anchor_offset, box_num_front]
                        self._get_sorted_proposal(tmp_ub_proposal, score_size, sorted_k, gm_offset)
                        workspace_offset.set_as((index - 1) * score_size * 8 + box_num_front * 8)
                        # sort two sorted proposals，put less scored 4096 to workspace
                        tik_func_sort_with_ub(self.tik_instance, [tmp_ub_proposal, ub_proposal],
                                              [ub_proposal, tmp_ub_proposal], sorted_k,
                                              self.workspace_second_nms_gm[workspace_offset])
                with self.tik_instance.if_scope(box_left > 0):
                    score_size = box_left
                    anchor_offset.set_as(box_mov_times * self.proposal_topk_k)
                    gm_offset = [image_offset, anchor_offset, box_num_front]
                    self._get_sorted_proposal_scalar(tmp_ub_proposal, score_size, sorted_k, gm_offset)
                    workspace_offset.set_as((box_mov_times - 1) * score_size * 8 + box_num_front * 8)
                    tik_func_sort_with_ub(self.tik_instance, [tmp_ub_proposal, ub_proposal],
                                          [ub_proposal, tmp_ub_proposal], sorted_k,
                                          self.workspace_second_nms_gm[workspace_offset])
        # do nms use topk output to get nms proposals per batch and move result to l1
        self._nms_handle(box_mov_times, box_left, box_num_front, sorted_k, ub_proposal)

    def _batch_nms_copy_out(self, ub_result_boxes, output_batch_offset, result_total):
        """
        batch_nms_copy_out
        """
        loop_burst_len = self.tik_instance.Scalar(dtype=Constant.DTYPE_INT32, name='loop_burst_len')
        loop_burst_len.set_as(ceil_div(result_total // 8, Constant.DATA_NUM_ONE_BLOCK))
        # process scores
        with self.tik_instance.new_stmt_scope():
            ub_out_boxes = self.tik_instance.Tensor(self.dtype_scores, [4, loop_burst_len * 16], name="ub_boxes",
                                                    scope=tik.scope_ubuf)
            # y1
            self.tik_instance.vextract(ub_out_boxes[1, 0], ub_result_boxes, loop_burst_len, 0)
            # x1
            self.tik_instance.vextract(ub_out_boxes[0, 0], ub_result_boxes, loop_burst_len, 1)
            # y2
            self.tik_instance.vextract(ub_out_boxes[3, 0], ub_result_boxes, loop_burst_len, 2)
            # x2
            self.tik_instance.vextract(ub_out_boxes[2, 0], ub_result_boxes, loop_burst_len, 3)

            with self.tik_instance.if_scope(self.is_need_rpn_offset):
                tik_func_vadds_scalar(self.tik_instance, ub_out_boxes[2, :], ub_out_boxes[2, :], 1.0, result_total // 8)
                tik_func_vadds_scalar(self.tik_instance, ub_out_boxes[3, :], ub_out_boxes[3, :], 1.0, result_total // 8)

            # score
            ub_out_box_scores = self.tik_instance.Tensor(self.dtype_scores, [loop_burst_len * 16],
                                                         name="ub_out_box_scores", scope=tik.scope_ubuf)
            self.tik_instance.vextract(ub_out_box_scores, ub_result_boxes, loop_burst_len, 4)
            # set the tail data = 0
            last_block_data_num = self.tik_instance.Scalar(dtype=Constant.DTYPE_INT32, init_value=0)
            last_block_data_num.set_as((result_total // 8) % Constant.DATA_NUM_ONE_BLOCK)
            with self.tik_instance.if_scope(last_block_data_num != 0):
                erase_data_num = Constant.DATA_NUM_ONE_BLOCK - last_block_data_num
                with self.tik_instance.for_range(0, erase_data_num) as index_tail:
                    ub_out_box_scores[result_total // 8 + index_tail].set_as(0)
                    with self.tik_instance.for_range(0, Constant.BOX_DIM_NUM) as i:
                        ub_out_boxes[i, result_total // 8 + index_tail].set_as(0)
            self.tik_instance.data_move(self.data_rois_probabilities[output_batch_offset], ub_out_box_scores,
                                        0, 1, loop_burst_len, 0, 0)
            for i in range(Constant.BOX_DIM_NUM):
                self.tik_instance.data_move(self.data_rois[output_batch_offset * 4 + i * self.post_nms_topn],
                                            ub_out_boxes[i, 0], 0, 1, loop_burst_len, 0, 0)

    def _copy_nms_output(self, image_offset):
        """
        _copy_nms_output
        """
        result_total = self.tik_instance.Scalar(dtype=Constant.DTYPE_INT32, name='result_total')
        result_total.set_as(total_num(self.l1_nms_result.shape) // 8 * 8)
        output_batch_offset = image_offset * self.post_nms_topn
        with self.tik_instance.if_scope(result_total // 8 > self.post_nms_topn):
            result_total.set_as(self.post_nms_topn * 8)
        result_total.set_as(ceil_div(result_total // 8, 16) * 16 * 8)
        with self.tik_instance.new_stmt_scope():
            loop_burst_len = self.tik_instance.Scalar(dtype=Constant.DTYPE_INT32, name='loop_burst_len')
            loop_burst_len.set_as(ceil_div(result_total, Constant.DATA_NUM_ONE_BLOCK))
            ub_result_boxes = self.tik_instance.Tensor(self.dtype_scores, [loop_burst_len * 16],
                                                       name="ub_result_boxes", scope=tik.scope_ubuf)
            tik_func_vector_scalar(self.tik_instance, ub_result_boxes, 0.0, loop_burst_len * 16)
            l1_buffer = self.l1_nms_result
            l1_offset = [0, 0]
            self.tik_instance.data_move(ub_result_boxes, l1_buffer[l1_offset], 0, 1, loop_burst_len, 0, 0)
            with self.tik_instance.new_stmt_scope():
                self._batch_nms_copy_out(ub_result_boxes, output_batch_offset, result_total)

    def _run_one_core(self, core_index):
        """
        _run_one_core
        """
        actual_image_num_each_core = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="actual_image_num_each_core",
                                                              init_value=0)
        actual_image_num_each_core.set_as(self.image_num_each_core)
        with self.tik_instance.if_scope(tik.all(core_index < self.image_num_rest, self.image_num_rest != 0)):
            actual_image_num_each_core.set_as(self.image_num_each_core + 1)

        image_num_front_core = self.tik_instance.Scalar(Constant.DTYPE_INT32, name="image_num_front_core",
                                                        init_value=0)
        with self.tik_instance.if_scope(actual_image_num_each_core > self.image_num_each_core):
            image_num_front_core.set_as(actual_image_num_each_core * core_index)
        with self.tik_instance.elif_scope(actual_image_num_each_core == self.image_num_each_core):
            image_num_front_core.set_as(self.image_num_rest + core_index * self.image_num_each_core)
        with self.tik_instance.for_range(0, actual_image_num_each_core) as index_image:
            image_offset = image_num_front_core + index_image
            with self.tik_instance.new_stmt_scope():
                self._run_image_loop(image_offset)
            with self.tik_instance.new_stmt_scope():
                self._copy_nms_output(image_offset)


@register_operator("GenerateBoundingBoxProposals")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def generate_bounding_box_proposals(scores, bbox_deltas, image_info, anchors, nms_threshold, pre_nms_topn, min_size,
                                    rois, rois_probabilities, post_nms_topn=300,
                                    kernel_name="generate_bounding_box_proposals"):
    """
    calculating data

    Parameters
    ----------
    scores : dict
        shape and dtype of input, the length of shape must be 4, dtype only support float16.
    bbox_deltas : dict
        shape and dtype of input, the length of shape must be 4, dtype only support float16.
    image_info : dict
        shape and dtype of input, the length of shape must be 2, dtype only support float16.
    anchors : dict
        shape and dtype of input, the length of shape must be 2, dtype only support float16.
    nms_threshold : dict
        shape and dtype of input, it is a scalar, dtype only support float16.
    pre_nms_topn : dict
        shape and dtype of input, it is a scalar, dtype only support int32.
    min_size : dict
        shape and dtype of input, it is a scalar, dtype only support float32.
    rois : dict
        shape and dtype of output, the length of shape must be 3, dtype only support float16.
    rois_probabilities : dict
        shape and dtype of output, the length of shape must be 3, dtype only support float16.
    post_nms_topn : int32
        shape and dtype of input, it is a optional attribute, dtype only support int32.
    kernel_name : str
        kernel name, default value is "generate_bounding_box_proposals"

    Returns
    -------
    None
    """
    score_dtype = scores.get("dtype").lower()
    bbox_deltas_dtype = scores.get("dtype").lower()
    image_info_dtype = scores.get("dtype").lower()
    anchors_dtype = scores.get("dtype").lower()
    para_check.check_dtype(score_dtype, (Constant.DTYPE_FP16,), param_name="scores")
    para_check.check_dtype(bbox_deltas_dtype, (Constant.DTYPE_FP16,), param_name="bbox_deltas")
    para_check.check_dtype(image_info_dtype, (Constant.DTYPE_FP16,), param_name="image_info")
    para_check.check_dtype(anchors_dtype, (Constant.DTYPE_FP16,), param_name="anchors")
    obj = GenerateBoundingBoxProposalsCompute(
        scores, bbox_deltas, image_info, anchors, nms_threshold, pre_nms_topn, min_size, rois, rois_probabilities,
        post_nms_topn, kernel_name)
    obj.generate_bounding_box_proposals_compute()
