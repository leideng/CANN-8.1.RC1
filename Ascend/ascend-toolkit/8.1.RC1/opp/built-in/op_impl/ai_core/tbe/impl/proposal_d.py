#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2019 Huawei Technologies Co., Ltd
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
proposal_d
"""
# 'pylint: disable=R0902
# 'pylint: disable=R0903
# 'pylint: disable=R0913
# 'pylint: disable=R0914
# 'pylint: disable=W0613
# 'pylint: disable=too-many-branches


from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tik
from impl import decoded_bbox
from impl import nms
from impl import topk
from impl.util import util_select_op_base
from impl.util.util_common import get_mask_rep_stride
from impl import cnms_yolov3_ver as cnms_yolo


def check_soc_version():
    """
    Check devices whether is rpn supported
    each of these devices only needs coordinate data and scores data
    """

    is_rpn_supported = True
    if tbe_platform.api_check_support("tik.vcopy"):
        is_rpn_supported = False
    return is_rpn_supported


# 'pylint: disable = unused-argument
def get_op_support_info(cls_prob_dic,
                        bbox_delta_dic,
                        im_info_dic,
                        rpn_bbox_dic,
                        rois_dic,
                        actual_rois_num_dic,
                        feat_stride,
                        base_size,
                        min_size,
                        ratio,
                        scale,
                        pre_nms_topn,
                        post_nms_topn,
                        iou_threshold,
                        output_actual_rois_num,
                        kernel_name="cce_proposal"):
    """
    calculate get_op_support_info
    """
    op_cal_info_in_json = util_select_op_base.get_op_cal_info(None, None, 0, 0)
    return op_cal_info_in_json


def _get_dtype_size(input_dtype):
    """
    :param input_dtype:
    :return:
    """
    if input_dtype == "float16":
        size = 2
    else:
        size = 4

    return size


def _get_ub_size():
    """
    :return:
    """
    ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    if ub_size <= 0:
        error_manager_vector.raise_err_specific_reson("proposal_d", "The value of the UB_SIZE is illegal!")
    return ub_size


def _filte_device_core(batch):
    """
    :param batch:
    :return:
    """
    device_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    if device_core_num <= 0:
        error_manager_vector.raise_err_specific_reson("proposal_d", "The value of the CORE_NUM is illegal!")
    if batch >= device_core_num:
        batch_factor = batch // device_core_num
        batch_factor_tail = batch - batch_factor * device_core_num
    else:
        batch_factor = batch
        batch_factor_tail = 0
        device_core_num = 1

    return device_core_num, batch_factor, batch_factor_tail


def _call_topk_sort(tik_instance, input_data, output):
    """
    :param tik_instance:
    :param input_data:
    :param output:
    :return:
    """
    score_threshold = 0
    k = input_data[0]
    regions_orig = input_data[1]
    mem_swap = input_data[2]
    proposal_num = input_data[3]

    batch_id = output[0]
    regions_sorted = output[1]
    proposal_actual_num = output[2]

    topk_input = {
        "proposal_num": proposal_num,
        "k": k,
        "score_threshold": score_threshold,
        "regions_orig": regions_orig,
        "mem_swap": mem_swap,
    }

    topk_out = {
        "batch_id": batch_id,
        "regions_sorted": regions_sorted,
        "proposal_actual_num": proposal_actual_num,
    }

    topk.tik_topk(tik_instance, topk_input, topk_out)


def ceil_div(value, factor):
    """
    Compute the smallest integer value that is greater than
    or equal to value/factor
    """
    result = (value + (factor - 1)) // factor
    return result


class Constant:
    """
    the class for constant
    """
    # define repeat elements every time for vsrot32
    REPEAT_ELE = 32
    # every loop process 4096 units
    PER_LOOP_UNIT = 4096
    # location elements, [x1, y1, x2, y2]
    FOUR_DIRECTION = 4
    # b16 elements num of every block also uesed as b16 elements num of mask
    BLOCK_ELE = 16
    # the socres_index contains four elements also marked as the class num processed every cycle
    UNIT_ELE = 4
    REPEAT_TIMES_MAX = 255
    # 0b0001 0001 0001 0001 is equals to type 3
    PATTERN_TYPE = 3
    DATALEN_4K = 4096
    DATALEN_2K = 2048
    DATALEN_1K = 1024
    DATALEN_128 = 128
    VECTOR_BLOCK_SIZE = 256
    REPEAT_STRIDE_EIGHT = 8
    MAX_MASK = 256
    INT32_MASK = 64
    MAX_BOXES_NUM = 35000


class InitProposalProcess:
    """
    Init Proposal Process
    """
    def __init__(self, input_data):
        """
        :param input_data:
        """
        self.tik_instance = input_data[0]
        feature_dic = input_data[1]
        self.im_info_dic = input_data[2]
        self.min_box_size = input_data[3]
        self.pre_nms_topn = input_data[4]
        self.post_nms_topn = input_data[5]
        self.nms_threshold = input_data[6]
        self.output_actual_rois_num = input_data[7]

        self.input_shape = feature_dic.get('shape')
        input_dtype = feature_dic.get('dtype')
        self.input_dtype = input_dtype

        channel = self.input_shape[1]
        height = self.input_shape[2]
        width = self.input_shape[3]
        num_anchor = channel // 4
        num = (num_anchor * height * width + 127) // 128
        self.num = num


class ProposalProcess(InitProposalProcess):
    """
    Proposal Process
    """
    def __init__(self, input_data):
        """
        :param input_data:
        """
        super(ProposalProcess, self).__init__(input_data)
        feature_dic = input_data[1]
        input_dtype = feature_dic.get('dtype')
        input_shape = feature_dic.get('shape')
        batch, channel, height, width = input_shape
        num_anchor = channel // 4
        num = (num_anchor * height * width + 127) // 128
        self.batch = batch
        
        if input_dtype == "float16":
            dsize = 2
        else:
            dsize = 4
        
        self.mask_num = Constant.MAX_MASK // dsize

        self.is_rpn_supported = check_soc_version()

        self.cls_prob = self.tik_instance.Tensor(input_dtype, (batch, channel // 2, height, width),
                                                 name="cls_prob",
                                                 scope=tbe_platform.scope_gm)
        self.bbox_delta = self.tik_instance.Tensor(input_dtype,
                                                   self.input_shape,
                                                   name="bbox_delta",
                                                   scope=tbe_platform.scope_gm)
        self.rpn_bbox = self.tik_instance.Tensor(input_dtype,
                                                 self.input_shape,
                                                 name="rpn_bbox",
                                                 scope=tbe_platform.scope_gm)

        self.im_info = self.tik_instance.Tensor(input_dtype,
                                                self.im_info_dic.get('shape'),
                                                name="im_info",
                                                scope=tbe_platform.scope_gm)

        size = _get_dtype_size(self.input_dtype)
        burst = ((num * 128 - num_anchor * height * width) * 8 * size + 31) // 32
        tail = burst * 32 // (8 * size) - (num * 128 - num_anchor * height * width)
        self.output_region_proposal = \
            self.tik_instance.Tensor(input_dtype,
                                     (batch, num * 128 + tail, 8),
                                     name="output_region_proposal",
                                     is_workspace=True,
                                     scope=tbe_platform.scope_gm)

        self.mem_swap = self.tik_instance.Tensor(input_dtype, (batch, num * 128 + tail, 8),
                                                 name="mem_swap",
                                                 is_workspace=True,
                                                 scope=tbe_platform.scope_gm)

        self.topk_output_proposal = \
            self.tik_instance.Tensor(input_dtype,
                                     (batch,
                                      ((self.pre_nms_topn + 15) // 16) * 16 + 4, 8),
                                     name="topk_output_proposal",
                                     is_workspace=True,
                                     scope=tbe_platform.scope_gm)

        self.temp_proposal_out = \
            self.tik_instance.Tensor(input_dtype,
                                     (batch, ((self.post_nms_topn + 15) // 16) * 16, 8),
                                     name="temp_proposal_out",
                                     is_workspace=True,
                                     scope=tbe_platform.scope_gm)

        self.rois = self.tik_instance.Tensor(input_dtype, (batch, 5, ((self.post_nms_topn + 15) // 16) * 16),
                                             name="rois",
                                             scope=tbe_platform.scope_gm)
        if self.output_actual_rois_num == 1:
            self.actual_rois_num = self.tik_instance.Tensor("int32", (batch, 8),
                                                            name="actual_rois_num",
                                                            scope=tbe_platform.scope_gm)
        else:
            self.actual_rois_num = self.tik_instance.Tensor("int32", (batch, 8),
                                                            name="actual_rois_num",
                                                            is_workspace=True,
                                                            scope=tbe_platform.scope_gm)

        self.boxes_num = self.num * 128 + tail
        
        self.nmsed_boxes_gm = self.tik_instance.Tensor(input_dtype, (self.batch, 4, self.boxes_num),
                                                   name="nmsed_boxes_gm",
                                                   is_workspace=True,
                                                   scope=tbe_platform.scope_gm)
        self.nmsed_scores_gm = self.tik_instance.Tensor(input_dtype, (self.batch, self.boxes_num),
                                                    name="nmsed_scores_gm",
                                                    is_workspace=True,
                                                    scope=tbe_platform.scope_gm)
        self.nmsed_classes_gm = self.tik_instance.Tensor(input_dtype, (self.batch, self.boxes_num),
                                                     name="nmsed_classes_gm",
                                                     is_workspace=True,
                                                     scope=tbe_platform.scope_gm)
        self.nmsed_num_gm = self.tik_instance.Tensor("int32", (self.batch, 8),
                                                 name="nmsed_num_gm",
                                                 is_workspace=True,
                                                 is_atomic_add=True,
                                                 scope=tbe_platform.scope_gm)
        
        self.x1y1x2y2_gm = self.tik_instance.Tensor(input_dtype, (4, self.boxes_num),
                                                     name="x1y1x2y2_gm",
                                                     is_workspace=True,
                                                     scope=tbe_platform.scope_gm)
        self.scores_gm = self.tik_instance.Tensor(input_dtype, (self.boxes_num, ),
                                                     name="scores_gm",
                                                     is_workspace=True,
                                                     scope=tbe_platform.scope_gm)

        if not self.is_rpn_supported:
            idx_size = ceil_div(self.boxes_num, Constant.DATALEN_4K) * Constant.DATALEN_4K
            idx_init = [i for i in range(idx_size)]
            self.idx_gm = self.tik_instance.Tensor("uint32",
                                            [idx_size, ],
                                            name="idx_gm",
                                            scope=tbe_platform.scope_gm,
                                            init_value=idx_init)
        self.workspace_ub_list = [None, None, None, None, None]  # xx1, yy1, xx2, yy2, scores
        
        out_box_len = ceil_div(self.post_nms_topn, Constant.BLOCK_ELE) * Constant.BLOCK_ELE
        self.out_box_len = out_box_len
        
        self.valid_detection = self.tik_instance.Scalar("int32", "valid_detection", 0)

    def init_tail_zero(self, batch_id, size):
        """
        :param batch_id:
        :param size:
        :return:
        """
        if self.input_dtype == "float16":
            ratio = 1
        else:
            ratio = 2

        channel = self.input_shape[1]
        height = self.input_shape[2]
        width = self.input_shape[3]

        num_anchor = channel // 4
        num = (num_anchor * height * width + 127) // 128

        tik_instance = self.tik_instance

        if num * 128 > num_anchor * height * width:
            with tik_instance.if_scope(True):
                burst = ((num * 128 - num_anchor * height * width) * 8 * size + 31) // 32

                tmp_ub = tik_instance.Tensor(self.input_dtype, (128, 8), name="tmp_ub", scope=tbe_platform.scope_ubuf)
                tik_instance.vector_dup(128 // ratio, tmp_ub, 0, 8 * ratio, 1, 8)
                tik_instance.data_move(self.output_region_proposal[batch_id, num_anchor * height * width, 0], tmp_ub, 0,
                                       1, burst, 0, 0)

    def data_move(self, dst, src, length):
        """
        move data beteen gm and ub
        :param dst: memory space in UB or GM, if in GM, src must in UB
        :param src: memory space in UB or GM, if in GM, dst must in UB
        :param length: Data length which is Aligned by 32 Bytes
        :return:
        """
        burst_len = length // Constant.BLOCK_ELE
        with self.tik_instance.if_scope(burst_len > 0):
            self.tik_instance.data_move(dst, src, 0, 1, burst_len, 0, 0)

    def pre_topk_selection_class(self):
        """
        topk selection for each class
        :return:
        """

        if self.pre_nms_topn < Constant.PER_LOOP_UNIT:
            shape_aligned = ceil_div(Constant.PER_LOOP_UNIT, self.mask_num) * self.mask_num
            
            eff_size = self.tik_instance.Scalar(dtype="uint32", name="eff_size", init_value=shape_aligned)
            
            scores_idx_out = self.tik_instance.Tensor("float16", [Constant.PER_LOOP_UNIT * Constant.UNIT_ELE * 2, ],
                                                    name="scores_idx_out", scope=tbe_platform.scope_ubuf)
        else:
            shape_aligned = ceil_div(self.pre_nms_topn // 2, self.mask_num) * self.mask_num * 2
            
            eff_size = self.tik_instance.Scalar(dtype="uint32", name="eff_size", init_value=shape_aligned)
            
            scores_idx_out = self.tik_instance.Tensor("float16", [shape_aligned * Constant.UNIT_ELE * 2, ],
                                                    name="scores_idx_out", scope=tbe_platform.scope_ubuf)

        cnms_yolo.gen_score_index_v300(self.tik_instance, [self.scores_gm, scores_idx_out],
                                       self.boxes_num, self.idx_gm, shape_aligned)
        
        x1_ub = self.tik_instance.Tensor("float16", [shape_aligned, ], name="x1_ub", scope=tbe_platform.scope_ubuf)
        x2_ub = self.tik_instance.Tensor("float16", [shape_aligned, ], name="x2_ub", scope=tbe_platform.scope_ubuf)
        y1_ub = self.tik_instance.Tensor("float16", [shape_aligned, ], name="y1_ub", scope=tbe_platform.scope_ubuf)
        y2_ub = self.tik_instance.Tensor("float16", [shape_aligned, ], name="y2_ub", scope=tbe_platform.scope_ubuf)
        scores_ub = self.tik_instance.Tensor("float16", [shape_aligned, ],
                                            name="scores_ub", scope=tbe_platform.scope_ubuf)
        cnms_yolo.init_tensor(self.tik_instance, x1_ub, shape_aligned)
        cnms_yolo.init_tensor(self.tik_instance, y1_ub, shape_aligned)
        cnms_yolo.init_tensor(self.tik_instance, x2_ub, shape_aligned)
        cnms_yolo.init_tensor(self.tik_instance, y2_ub, shape_aligned)
        cnms_yolo.init_tensor(self.tik_instance, scores_ub, shape_aligned)
        
        proposal_data = [x1_ub, x2_ub, y1_ub, y2_ub, scores_ub]
        
        if tbe_platform.api_check_support("tik.vgather") and self.pre_nms_topn < Constant.PER_LOOP_UNIT \
                                                        and self.boxes_num < Constant.MAX_BOXES_NUM:
            self.get_boxes_after_score_thresh(proposal_data,
                                                [self.x1y1x2y2_gm, self.boxes_num,
                                                scores_idx_out, shape_aligned], eff_size)
        else:
            cnms_yolo.get_boxes_after_score_thresh_v300(self.tik_instance, proposal_data,
                                                        [self.x1y1x2y2_gm, self.boxes_num,
                                                        scores_idx_out, shape_aligned], eff_size)
        
        self.workspace_ub_list[0] = x1_ub
        self.workspace_ub_list[1] = y1_ub
        self.workspace_ub_list[2] = x2_ub
        self.workspace_ub_list[3] = y2_ub
        self.workspace_ub_list[4] = scores_ub

    def get_boxes_after_score_thresh(self, proposal_info, calc_info, size=4096):
        """
        move boxes_gm to boxes_ub according to the sorting index

        """
        xx1, xx2, yy1, yy2, scores_ub = proposal_info
        boxes, boxes_num, scores_index, pre_loop_unit = calc_info

        cnms_yolo.init_tensor(self.tik_instance, xx1, pre_loop_unit)
        cnms_yolo.init_tensor(self.tik_instance, xx2, pre_loop_unit)
        cnms_yolo.init_tensor(self.tik_instance, yy1, pre_loop_unit)
        cnms_yolo.init_tensor(self.tik_instance, yy2, pre_loop_unit)
        cnms_yolo.init_tensor(self.tik_instance, scores_ub, pre_loop_unit)

        lo_index = self.tik_instance.Scalar("int32", init_value = 2)
        topk_index_ub = self.tik_instance.Tensor("int32", [pre_loop_unit, ], name="topk_index_ub",
                                                 scope=tbe_platform.scope_ubuf)
        topk_box_ub = self.tik_instance.Tensor(self.input_dtype, [self.boxes_num, ], name="topk_box_ub",
                                               scope=tbe_platform.scope_ubuf)
        cnms_yolo.init_tensor(self.tik_instance, topk_index_ub, pre_loop_unit)
        with self.tik_instance.for_range(0, size) as idx:
            scores_index_offset = idx * Constant.UNIT_ELE
            topk_index_ub[idx].set_as(
                scores_index[scores_index_offset + 2:scores_index_offset + 4].reinterpret_cast_to("uint32"))
            scores_ub[idx].set_as(scores_index[scores_index_offset])
        mul_loop = pre_loop_unit // Constant.INT32_MASK
        self.tik_instance.vmuls(Constant.INT32_MASK, topk_index_ub, topk_index_ub, lo_index, mul_loop, 1, 1, 8, 8)
        burst_len = size // Constant.DATALEN_128

        cnms_yolo.init_tensor(self.tik_instance, topk_box_ub, self.boxes_num)
        self.data_move(topk_box_ub, boxes[0 * boxes_num], self.boxes_num)
        self.tik_instance.vgather(Constant.DATALEN_128, xx1, topk_box_ub, topk_index_ub, burst_len,
                                  8, 0, 0, mask_mode="normal")

        cnms_yolo.init_tensor(self.tik_instance, topk_box_ub, self.boxes_num)
        self.data_move(topk_box_ub, boxes[1 * boxes_num], self.boxes_num)
        self.tik_instance.vgather(Constant.DATALEN_128, yy1, topk_box_ub, topk_index_ub, burst_len,
                                  8, 0, 0, mask_mode="normal")

        cnms_yolo.init_tensor(self.tik_instance, topk_box_ub, self.boxes_num)
        self.data_move(topk_box_ub, boxes[2 * boxes_num], self.boxes_num)
        self.tik_instance.vgather(Constant.DATALEN_128, xx2, topk_box_ub, topk_index_ub, burst_len,
                                  8, 0, 0, mask_mode="normal")

        cnms_yolo.init_tensor(self.tik_instance, topk_box_ub, self.boxes_num)
        self.data_move(topk_box_ub, boxes[3 * boxes_num], self.boxes_num)
        self.tik_instance.vgather(Constant.DATALEN_128, yy2, topk_box_ub, topk_index_ub, burst_len,
                                  8, 0, 0, mask_mode="normal")

        cnms_yolo.exchange_coordinate(self.tik_instance, [xx1, xx2, yy1, yy2], pre_loop_unit)

    def ssd_iou_selection(self, x1_ub, y1_ub, x2_ub, y2_ub, scores_ub, eff_lens):
        """
        execute iou selection,
        :param x1_ub:
        :param y1_ub:
        :param x2_ub:
        :param y2_ub:
        :param scores_ub:
        :param eff_lens:
        :return:
        """
        shape_aligned = ceil_div(self.pre_nms_topn, self.mask_num) * self.mask_num
        mask, _ = get_mask_rep_stride(x1_ub)

        # iou Selection for only topk data for per class
        single_area = self.tik_instance.Tensor(self.input_dtype, [shape_aligned, ], name="single_area",
                                           scope=tbe_platform.scope_ubuf)
        # get area of every windows
        cnms_yolo.get_rectangle_area(self.tik_instance, [x1_ub, x2_ub, y1_ub, y2_ub], single_area, shape_aligned)

        iou = self.tik_instance.Tensor(self.input_dtype, [shape_aligned, ], name="iou",
                                   scope=tbe_platform.scope_ubuf)
        cnms_yolo.init_tensor(self.tik_instance, iou, shape_aligned)
        # calculate the iou, exit when the output windows is more than eff_lens
        overlap = self.tik_instance.Tensor(self.input_dtype, [shape_aligned, ], name="overlap",
                                       scope=tbe_platform.scope_ubuf)
        # define tmp tensor for following use, to reduce the cycle of apply/release memory
        tmp1 = self.tik_instance.Tensor(self.input_dtype, [shape_aligned, ], name="tmp1", scope=tbe_platform.scope_ubuf)
        tmp2 = self.tik_instance.Tensor(self.input_dtype, [shape_aligned, ], name="tmp2", scope=tbe_platform.scope_ubuf)
        mask_shape_lens = shape_aligned // Constant.BLOCK_ELE
        mask_uint16 = self.tik_instance.Tensor("uint16", [mask_shape_lens, ], name="mask_uint16",
                                           scope=tbe_platform.scope_ubuf)
        iou_thresh = self.nms_threshold / (1 + self.nms_threshold)

        # calculate ioues for every windows
        with self.tik_instance.for_range(0, self.pre_nms_topn) as idx:
            with self.tik_instance.if_scope(idx < eff_lens):
                cnms_yolo.get_overlap(self.tik_instance, [x1_ub, x2_ub, y1_ub, y2_ub],
                                      [overlap, tmp1, tmp2], idx, shape_aligned)
                _aligned_length = ceil_div(eff_lens, mask) * mask
                cnms_yolo.cal_iou(self.tik_instance, [single_area, iou, tmp2], idx, _aligned_length, iou_thresh)
                cnms_yolo.gen_mask(self.tik_instance, overlap, iou, mask_uint16, size=shape_aligned)
                cnms_yolo.update_input_v300(self.tik_instance,
                                            [x1_ub, x2_ub, y1_ub, y2_ub, scores_ub], single_area,
                                            eff_lens, [tmp1, tmp2, mask_uint16, shape_aligned])
    
    def post_topk_selection_class(self, eff_lens):
        """
        post topk selection, if post_nms_topn > 0, set eff_lens to post_nms_topn
        :param eff_lens:
        :return:
        """
        if self.post_nms_topn > 0:
            eff_lens.set_as(self.post_nms_topn)
    
    def partial_init_tensor(self, dst, size, start, init_value=0):
        """
        init party memory of tensor
        :param dst: ub memory
        :param size: tensor size
        :param start: start init address
        :param init_value:
        :return:
        """
        vector_mask, rep_stride = get_mask_rep_stride(dst)
        aligned_start = ceil_div(start, 32) * 32
        length = size - aligned_start
        max_lens = Constant.REPEAT_TIMES_MAX * vector_mask
        loop_num = length // max_lens
        tail = length % max_lens
        repeat_times = tail // vector_mask
        tail_aligned = tail % vector_mask

        with self.tik_instance.for_range(start, aligned_start) as idx:
            dst[idx].set_as(init_value)

        off = self.tik_instance.Scalar("uint32")
        with self.tik_instance.for_range(0, loop_num) as idx:
            off.set_as(vector_mask * Constant.REPEAT_TIMES_MAX * idx)
            self.tik_instance.vec_dup(vector_mask,
                                  dst[aligned_start + off],
                                  init_value,
                                  Constant.REPEAT_TIMES_MAX,
                                  rep_stride)
        with self.tik_instance.if_scope(tik.all(tail != 0, repeat_times > 0)):
            offset = length - tail
            self.tik_instance.vec_dup(vector_mask,
                                  dst[aligned_start + offset],
                                  init_value,
                                  repeat_times,
                                  rep_stride)
        with self.tik_instance.if_scope(tail_aligned != 0):
            with self.tik_instance.for_range(0, tail_aligned) as idx:
                dst[aligned_start + length - tail_aligned + idx].set_as(init_value)
    
    def cnms_calcation_class(self, batch_idx, class_idx, bbox_out_list):
        """
        execute cnms calculation for per classes
        :param batch_idx:
        :param class_idx:
        :param bbox_out_list:
        :return:
        """
        shape_aligned = ceil_div(self.pre_nms_topn, self.mask_num) * self.mask_num

        x1_ub = self.workspace_ub_list[0]
        y1_ub = self.workspace_ub_list[1]
        x2_ub = self.workspace_ub_list[2]
        y2_ub = self.workspace_ub_list[3]
        scores_ub = self.workspace_ub_list[4]

        # select by scores_threshold
        eff_lens = self.tik_instance.Scalar("uint32", "eff_lens", 0)
        eff_lens.set_as(shape_aligned)

        with self.tik_instance.if_scope(eff_lens > 0):
            
            repeats = self.tik_instance.Scalar("uint32", "repeats", 0)
            repeats.set_as(shape_aligned // self.mask_num)
            
            self.tik_instance.vmuls(self.mask_num , x1_ub, x1_ub, 0.01, repeats, 1, 1,
                                   Constant.REPEAT_STRIDE_EIGHT, Constant.REPEAT_STRIDE_EIGHT)
            self.tik_instance.vmuls(self.mask_num , y1_ub, y1_ub, 0.01, repeats, 1, 1,
                                   Constant.REPEAT_STRIDE_EIGHT, Constant.REPEAT_STRIDE_EIGHT)
            self.tik_instance.vmuls(self.mask_num , x2_ub, x2_ub, 0.01, repeats, 1, 1,
                                   Constant.REPEAT_STRIDE_EIGHT, Constant.REPEAT_STRIDE_EIGHT)
            self.tik_instance.vmuls(self.mask_num , y2_ub, y2_ub, 0.01, repeats, 1, 1,
                                   Constant.REPEAT_STRIDE_EIGHT, Constant.REPEAT_STRIDE_EIGHT)
            # do iou selection
            self.ssd_iou_selection(x1_ub, y1_ub, x2_ub, y2_ub, scores_ub, eff_lens)
            # do post topk
            self.post_topk_selection_class(eff_lens)
            self.tik_instance.vmuls(self.mask_num, x1_ub, x1_ub, 100, repeats, 1, 1,
                                   Constant.REPEAT_STRIDE_EIGHT, Constant.REPEAT_STRIDE_EIGHT)
            self.tik_instance.vmuls(self.mask_num, y1_ub, y1_ub, 100, repeats, 1, 1,
                                   Constant.REPEAT_STRIDE_EIGHT, Constant.REPEAT_STRIDE_EIGHT)
            self.tik_instance.vmuls(self.mask_num, x2_ub, x2_ub, 100, repeats, 1, 1,
                                   Constant.REPEAT_STRIDE_EIGHT, Constant.REPEAT_STRIDE_EIGHT)
            self.tik_instance.vmuls(self.mask_num, y2_ub, y2_ub, 100, repeats, 1, 1,
                                   Constant.REPEAT_STRIDE_EIGHT, Constant.REPEAT_STRIDE_EIGHT)
            # store data
            with self.tik_instance.if_scope(eff_lens > 0):
                self.store_cnms_data_per_class(batch_idx, class_idx,
                                               x1_ub, x2_ub, y1_ub, y2_ub, scores_ub,
                                               bbox_out_list, eff_lens)

    def store_cnms_data_per_class(self, batch_idx, class_idx,
                                  xx1, xx2, yy1, yy2, scores_ub, bbox_out_list, eff_lens):
        """
        save data which select by cnms
        :param batch_idx:
        :param class_idx:
        :param xx1:
        :param xx2:
        :param yy1:
        :param yy2:
        :param scores_ub:
        :param bbox_out_list:
        :param eff_lens:
        :return:
        """
        boxes_out, scores_out, class_out, box_num_out = bbox_out_list

        box_outnum_ub = self.tik_instance.Tensor("int32",
                                             (self.batch, 8),
                                             name="box_outnum_ub",
                                             scope=tbe_platform.scope_ubuf)
        cnms_yolo.init_tensor(self.tik_instance, box_outnum_ub, 8, 0)
        box_outnum_ub[batch_idx, 0].set_as(eff_lens)
        self.tik_instance.data_move(box_num_out, box_outnum_ub, 0, 1, self.batch, 0, 0)

        repeat_times = ceil_div(eff_lens, Constant.BLOCK_ELE)
        with self.tik_instance.if_scope(repeat_times > 0):
            self.tik_instance.data_move(boxes_out[batch_idx, 0, self.valid_detection], xx1, 0, 1, repeat_times, 0, 0)
            self.tik_instance.data_move(boxes_out[batch_idx, 1, self.valid_detection], yy1, 0, 1, repeat_times, 0, 0)
            self.tik_instance.data_move(boxes_out[batch_idx, 2, self.valid_detection], xx2, 0, 1, repeat_times, 0, 0)
            self.tik_instance.data_move(boxes_out[batch_idx, 3, self.valid_detection], yy2, 0, 1, repeat_times, 0, 0)
            
        self.valid_detection.set_as(box_outnum_ub[batch_idx, 0])
    
    def store_cnms_data_output(self, batch_idx, bbox_out_list):
        """
        store cnms output data into gm with SSD out formate
        :param batch_idx:
        :param bbox_out_list:
        :return:
        """
        boxes_out, scores_out, classes_out, box_num_out = bbox_out_list
        nms_len = self.tik_instance.Scalar("int32", "nms_len", init_value=0)
        nms_len.set_as(box_num_out[batch_idx, 0])

        with self.tik_instance.if_scope(nms_len > 0):
            with self.tik_instance.if_scope(nms_len > self.out_box_len):
                nms_len.set_as(self.out_box_len)
            box_outnum_ub = self.tik_instance.Tensor("int32",
                                                 (1, 8),
                                                 name="box_outnum_ub",
                                                 scope=tbe_platform.scope_ubuf)
            cnms_yolo.init_tensor(self.tik_instance, box_outnum_ub, 8)
            box_outnum_ub[0, 0].set_as(nms_len)
            # the size of box_outnum_ub is 32 Byte(1 Block)
            self.tik_instance.data_move(self.actual_rois_num[batch_idx, 0], box_outnum_ub, 0, 1, 1, 0, 0)

        with self.tik_instance.new_stmt_scope():
            out_box_ub = self.tik_instance.Tensor(self.input_dtype,
                                                 (self.batch, 5, self.out_box_len),
                                                 name="out_box_ub",
                                                 scope=tbe_platform.scope_ubuf)
            cnms_yolo.init_tensor(self.tik_instance, out_box_ub, 5 * self.out_box_len)
            repeats_out = (nms_len + 15) // 16 * 16
            self.tik_instance.data_move(out_box_ub[batch_idx, 1, 0], boxes_out[batch_idx, 0, 0],
                                        0, 1, repeats_out, 0, 0)
            self.tik_instance.data_move(out_box_ub[batch_idx, 2, 0], boxes_out[batch_idx, 1, 0],
                                        0, 1, repeats_out, 0, 0)
            self.tik_instance.data_move(out_box_ub[batch_idx, 3, 0], boxes_out[batch_idx, 2, 0],
                                        0, 1, repeats_out, 0, 0)
            self.tik_instance.data_move(out_box_ub[batch_idx, 4, 0], boxes_out[batch_idx, 3, 0],
                                        0, 1, repeats_out, 0, 0)

            burst_times = (5 * self.out_box_len) // Constant.BLOCK_ELE
            with self.tik_instance.if_scope(burst_times > 0):
                self.tik_instance.data_move(self.rois, out_box_ub, 0, 1, burst_times, 0, 0)
    
    def cce_proposal(self, kernel_name="proposal"):
        """
        :param kernel_name:
        :return:
        """
        device_core_num, batch_factor, batch_factor_tail = \
            _filte_device_core(self.input_shape[0])

        size = _get_dtype_size(self.input_dtype)

        with self.tik_instance.for_range(0, device_core_num, block_num=device_core_num) as block_id:

            ub_size = _get_ub_size()
            one_core_process_object = \
                decoded_bbox.OneCoreProcess((self.tik_instance,
                                             self.min_box_size,
                                             self.input_dtype, size,
                                             self.input_shape,
                                             device_core_num, batch_factor,
                                             ub_size, self.x1y1x2y2_gm, self.scores_gm))

            batch = self.tik_instance.Scalar("int32", "batch", 0)
            output_gm_list = [self.nmsed_boxes_gm[batch, :, :],
                              self.nmsed_scores_gm[batch, :],
                              self.nmsed_classes_gm[batch, :],
                              self.nmsed_num_gm[batch, :]]
            
            with self.tik_instance.for_range(0, batch_factor) as batch_index:
                batch_id = block_id * batch_factor + batch_index

                one_core_process_object.one_core_process_decode_bbox(batch_id, self.cls_prob, self.bbox_delta,
                                                                     self.rpn_bbox, self.im_info,
                                                                     self.output_region_proposal)

                self.init_tail_zero(batch_id, size)

                topk_output_actual_proposal_num = \
                    self.tik_instance.Scalar(dtype="int32")
                
                if self.is_rpn_supported:
                    _call_topk_sort(self.tik_instance,
                                    (self.pre_nms_topn, self.output_region_proposal, self.mem_swap, self.num * 128),
                                    (batch_id, self.topk_output_proposal, topk_output_actual_proposal_num))

                    input_offset = batch_id * (((self.pre_nms_topn + 15) // 16) * 16 + 4) * 8
                    nms.cce_nms((self.input_dtype, ub_size, self.nms_threshold, batch_id, self.pre_nms_topn,
                                self.post_nms_topn, input_offset, self.im_info, self.tik_instance),
                                self.temp_proposal_out, self.topk_output_proposal, topk_output_actual_proposal_num,
                                self.actual_rois_num, self.rois)
                else:
                    self.pre_topk_selection_class()
                    
                    self.cnms_calcation_class(batch_id, 0, output_gm_list)
                    
                    self.store_cnms_data_output(batch_id, output_gm_list)

            with self.tik_instance.if_scope(block_id < batch_factor_tail):
                batch_id = batch_factor * device_core_num + block_id

                one_core_process_object.one_core_process_decode_bbox(batch_id, self.cls_prob, self.bbox_delta,
                                                                     self.rpn_bbox, self.im_info,
                                                                     self.output_region_proposal)

                self.init_tail_zero(batch_id, size)

                topk_output_actual_proposal_num = \
                    self.tik_instance.Scalar(dtype="int32")

                if self.is_rpn_supported:
                    _call_topk_sort(self.tik_instance,
                                    (self.pre_nms_topn, self.output_region_proposal, self.mem_swap, self.num * 128),
                                    (batch_id, self.topk_output_proposal, topk_output_actual_proposal_num))

                    input_offset = batch_id * (((self.pre_nms_topn + 15) // 16) * 16 + 4) * 8
                    nms.cce_nms((self.input_dtype, ub_size, self.nms_threshold, batch_id, self.pre_nms_topn,
                                self.post_nms_topn, input_offset, self.im_info, self.tik_instance),
                                self.temp_proposal_out, self.topk_output_proposal, topk_output_actual_proposal_num,
                                self.actual_rois_num, self.rois)
                else:
                    self.pre_topk_selection_class()
                    
                    self.cnms_calcation_class(batch_id, 0, output_gm_list)
                    
                    self.store_cnms_data_output(batch_id, output_gm_list)
        if self.output_actual_rois_num == 1:
            self.tik_instance.BuildCCE(kernel_name,
                                       inputs=[self.cls_prob, self.bbox_delta, self.im_info, self.rpn_bbox],
                                       outputs=[self.rois, self.actual_rois_num])
        else:
            self.tik_instance.BuildCCE(kernel_name,
                                       inputs=[self.cls_prob, self.bbox_delta, self.im_info, self.rpn_bbox],
                                       outputs=[self.rois])

        return self.tik_instance


def _check_datatype(dtype):
    """
    :param dtype:
    :return:
    """
    if not tbe_platform.api_check_support("tik.vrelu", "float32"):
        para_check.check_dtype(dtype.lower(), ["float16"], param_name="rpn_bbox_dic")
    else:
        para_check.check_dtype(dtype.lower(), ["float16", "float32"], param_name="rpn_bbox_dic")


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_LIST_FLOAT, para_check.OPTION_ATTR_LIST_FLOAT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def proposal_d(cls_prob_dic,
               bbox_delta_dic,
               im_info_dic,
               rpn_bbox_dic,
               rois_dic,
               actual_rois_num_dic,
               feat_stride,
               base_size,
               min_size,
               ratio,
               scale,
               pre_nms_topn,
               post_nms_topn,
               iou_threshold,
               output_actual_rois_num,
               kernel_name="cce_proposal"):
    """
    :param cls_prob_dic:
    :param bbox_delta_dic:
    :param im_info_dic:
    :param rpn_bbox_dic:
    :param rois_dic:
    :param actual_rois_num_dic:
    :param feat_stride:
    :param base_size:
    :param min_size:
    :param ratio:
    :param scale:
    :param pre_nms_topn:
    :param post_nms_topn:
    :param iou_threshold:
    :param output_actual_rois_num:
    :param kernel_name:
    :return:
    """
    input_dtype = rpn_bbox_dic.get('dtype')
    input_shape = rpn_bbox_dic.get('shape')
    para_check.check_shape(input_shape, min_rank=4, param_name="rpn_bbox_dic")
    channel = input_shape[1]

    feature_dic = {"shape": input_shape, "dtype": input_dtype}

    tik_instance = tik.Tik(tik.Dprofile())
    tik_name = tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION)
    _check_datatype(input_dtype)
    if actual_rois_num_dic is not None and bool(actual_rois_num_dic):
        output_actual_rois_num = 1
    else:
        output_actual_rois_num = 0

    if min_size <= 0:
        error_manager_vector.raise_err_input_param_range_invalid("proposal_d", "min_size", "0", "inf", str(min_size))

    min_box_size = [min_size, min_size]

    for ratio_value in ratio:
        if ratio_value <= 0:
            error_manager_vector.raise_err_input_param_range_invalid("proposal_d", "ratio_value", "0", "inf",
                                                                     str(ratio_value))

    for scale_value in scale:
        if scale_value <= 0:
            error_manager_vector.raise_err_input_param_range_invalid("proposal_d", "scale_value", "0", "inf",
                                                                     str(scale_value))

    if feat_stride <= 0 or base_size <= 0 or \
            pre_nms_topn <= 0 or post_nms_topn <= 0:
        error_manager_vector.raise_err_input_value_invalid("proposal_d", "feat_stride, base_size, pre_nms_topn \
                                                           and post_nms_topn", "greater than 0", "{}, {}, {} \
                                                           and {}".format(feat_stride, base_size, pre_nms_topn, \
                                                           post_nms_topn))

    if pre_nms_topn > 6000 or post_nms_topn > 6000:
        error_manager_vector.raise_err_input_param_range_invalid("proposal_d", "pre_nms_topn or post_nms_topn", "0",
                                                                 "6000",
                                                                 str(pre_nms_topn) + ", " + str(post_nms_topn))

    if tik_name in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403") and \
            (pre_nms_topn > 3000 or post_nms_topn > 3000):
        rule_desc = "pre_nms_topn and post_nms_topn must be <=3000 in HISI"
        param_value = ','.join((str(pre_nms_topn), str(post_nms_topn)))
        error_manager_vector.raise_err_check_params_rules("proposal_d", rule_desc, "pre_nms_topn or post_nms_topn",
                                                          param_value)

    if channel % 4 != 0:
        error_manager_vector.raise_err_input_value_invalid("proposal_d", "channel", "multiples of 16", channel)

    if iou_threshold <= 0 or iou_threshold >= 1:
        error_manager_vector.raise_err_input_param_range_invalid("proposal_d", "iou_threshold", "0", "1",
                                                                 str(iou_threshold))

    proposal_result = ProposalProcess((tik_instance, feature_dic, im_info_dic, min_box_size, pre_nms_topn,
                                       post_nms_topn, iou_threshold, output_actual_rois_num))

    return proposal_result.cce_proposal(kernel_name)
