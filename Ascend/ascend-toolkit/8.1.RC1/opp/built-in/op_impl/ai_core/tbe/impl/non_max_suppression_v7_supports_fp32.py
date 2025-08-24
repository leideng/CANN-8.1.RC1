#!/usr/bin/env python
# coding: utf-8
# Copyright 2023 Huawei Technologies Co., Ltd
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
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.batch_multi_class_nms_topk import sort_within_ub
from impl.batch_multi_class_nms_topk import sort_with_ub
from impl.non_max_suppression_v7_without_proposal import non_max_suppression_v7_without_proposal


class Constant:
    TOP_K = 2048
    DATA_EACH_BLOCK_16 = 16
    DATA_EACH_BLOCK_32 = 8
    COORDINATE_SCALING = 0.01
    MASK_16 = 128
    MASK_32 = 64
    RPN_OFFSET = -1
    RPN_PROPOSAL_NUM = 16
    BURST_PROPOSAL_NUM = 128
    TENSOR_MAX_NUM = 63
    BYTE_PER_DATA_16 = 2
    BYTE_PER_DATA_32 = 4
    BYTE_PER_DATA_64 = 8
    PROPOSAL_LEN = 8
    INDEX_NUM = 3
    INDEX_SCORE = 4
    INDEX_MINUS_ONE = -1
    ONE_THOUSAND = 1000
    MAX_REPEAT = 255
    BOX_LOC = 4


def ceil_div(value, factor):
    """
    Compute the smallest integer value that is greater than
    or equal to value/factor
    """
    result = (value + (factor - 1)) // factor
    return result


class NonMaxSuppressionSupportsFP32():
    """
    Function: use to store NonMaxSuppression base parameters
    """
    # pylint: disable=too-many-arguments
    def __init__(self,
                 boxes,
                 scores,
                 max_output_size,
                 iou_threshold,
                 score_threshold,
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
        # init tik instance.
        self.tik_instance = tik.Tik()
        # get soc's aicore_num and ub_size.
        self.aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)

        # get boxes shape [num_batches, spatial_dimension, 4] and dtype(FP16 or FP32).
        self.boxes_shape = list(boxes.get("shape"))
        # get boxes/scores dtype.
        self.dtype = boxes.get("dtype")
        # get scores shape [num_batches, num_classes, spatial_dimension].
        self.scores_shape = list(scores.get("shape"))
        # get index_id shape [num_batches, num_classes, spatial_dimension, 3] and dtype(FP16).
        self.index_id_shape = list(index_id.get("shape"))

        # get attr.
        self.center_point_box = center_point_box
        self.max_boxes_size = max_boxes_size

        # init input gm.
        self.init_input_gm(max_output_size, iou_threshold, score_threshold)
        # init output gm.
        self.init_output_gm()

        # get batch, classes, boxes_num
        self.batch, self.classes, self.boxes_num = self.scores_shape
        self.all_classes = self.batch * self.classes

        # get max_class_size.
        self.max_class_size = ceil_div(max_boxes_size, self.all_classes)

        if self.dtype == "float16":
            self.byte_per_data = Constant.BYTE_PER_DATA_16
            self.data_each_block = Constant.DATA_EACH_BLOCK_16
            self.mask = Constant.MASK_16
            self.burst_proposal_num = Constant.MASK_16
        else:
            self.byte_per_data = Constant.BYTE_PER_DATA_32
            self.data_each_block = Constant.DATA_EACH_BLOCK_32
            self.mask = Constant.MASK_32
            self.burst_proposal_num = Constant.MASK_32

        # init iou_thresh scalar dtype as same as boxes.
        self.iou_thresh = self.tik_instance.Scalar(self.dtype, init_value=0.0)
        self.iou_thresh_plus_one = self.tik_instance.Scalar(self.dtype, init_value=0.0)
        if iou_threshold:
            self.get_iou_threshold()
        
        # init score_thresh scalar dtype as same as boxes.
        self.score_thresh = self.tik_instance.Scalar(dtype="float32", init_value=0.0)
        if score_threshold:
            self.get_score_threshold()
        
        self.proposal_topk = self.get_topk()
        self.single_loop = True
        self.topk_loop_time = 1
        self.topk_loop_tail = 0
        if self.boxes_num > self.proposal_topk:
            self.single_loop = False
            self.topk_loop_time = self.boxes_num // self.proposal_topk
            self.topk_loop_tail = self.boxes_num % self.proposal_topk

        # store every topk result.
        self.workspace_boxes_gm = self.tik_instance.Tensor(self.dtype, 
                                                           [self.aicore_num, self.boxes_num * Constant.PROPOSAL_LEN],
                                                           name="workspace_boxes_gm",
                                                           scope=tik.scope_gm,
                                                           is_workspace=True)
        self.workspace_index_gm = self.tik_instance.Tensor(self.dtype, 
                                                           [self.aicore_num, self.boxes_num * Constant.PROPOSAL_LEN],
                                                           name="workspace_index_gm",
                                                           scope=tik.scope_gm,
                                                           is_workspace=True)

        self.max_selected_nms_num_in_ub = ceil_div(self.max_class_size, Constant.RPN_PROPOSAL_NUM) \
                                          * Constant.RPN_PROPOSAL_NUM

        # gm_nms_result is a workspace store all the selected indices from all classes.
        self.gm_nms_result = self.tik_instance.Tensor(self.dtype, 
                                                      [self.all_classes, self.max_selected_nms_num_in_ub,
                                                       Constant.PROPOSAL_LEN],
                                                      name="gm_nms_result", scope=tik.scope_gm, is_workspace=True)
        
        # selected_proposals_cnt for every class, init to 0.
        self.selected_proposals_cnt = self.tik_instance.Scalar(dtype="uint16", init_value=0)

        # get and update the core used.
        self.class_per_core = ceil_div(self.all_classes, self.aicore_num)
        if Constant.INDEX_NUM * self.max_class_size < self.data_each_block:
            self.aicore_num = self.core_used = 1
            self.class_per_core = self.class_core_tail = self.all_classes
        else:
            self.core_used = ceil_div(self.all_classes, self.class_per_core)
            self.class_core_tail = self.all_classes - (self.core_used - 1) * self.class_per_core
            self.aicore_num = self.core_used

        self.nms_temp_zero = self.tik_instance.Tensor(self.dtype,
                                                      [self.max_selected_nms_num_in_ub, Constant.PROPOSAL_LEN],
                                                      name="nms_temp_zero", scope=tik.scope_gm, is_workspace=True)
        self.nms_temp_minus_one = self.tik_instance.Tensor(self.dtype, 
                                                           [self.max_selected_nms_num_in_ub, Constant.PROPOSAL_LEN],
                                                           name="nms_temp_minus_one", scope=tik.scope_gm,
                                                           is_workspace=True)
        
        with self.tik_instance.new_stmt_scope():
            temp_zero_ub = self.tik_instance.Tensor(self.dtype,
                                                    [self.max_selected_nms_num_in_ub, Constant.PROPOSAL_LEN],
                                                    name="temp_zero_ub", scope=tik.scope_ubuf)
            temp_minus_one_ub = self.tik_instance.Tensor(self.dtype, 
                                                         [self.max_selected_nms_num_in_ub, Constant.PROPOSAL_LEN],
                                                         name="temp_minus_one_ub", scope=tik.scope_ubuf)
            self.tik_func_vector(temp_zero_ub, 0, self.max_selected_nms_num_in_ub * Constant.PROPOSAL_LEN, self.mask)
            self.tik_func_vector(temp_minus_one_ub, -1,
                                 self.max_selected_nms_num_in_ub * Constant.PROPOSAL_LEN, self.mask)
            repeat_time = ceil_div(self.max_selected_nms_num_in_ub * Constant.PROPOSAL_LEN, self.data_each_block)
            self.tik_instance.data_move(self.nms_temp_zero, temp_zero_ub, 0, 1, repeat_time, 0, 0)
            self.tik_instance.data_move(self.nms_temp_minus_one, temp_minus_one_ub, 0, 1, repeat_time, 0, 0)
    
    def get_topk(self):
        """
        calculate max topk num according to ub_size.
        """
        topk = Constant.TOP_K if self.dtype == "float16" else Constant.TOP_K // 2
        topk_max_num = self.ub_size // self.byte_per_data // Constant.TENSOR_MAX_NUM
        if topk_max_num < topk:
            return topk_max_num // Constant.BURST_PROPOSAL_NUM * Constant.BURST_PROPOSAL_NUM
        else:
            return topk

    def get_iou_threshold(self):
        '''
        get iou_threshold and iou_threshold + 1, the same dtype as boxes and scores.
        '''
        with self.tik_instance.new_stmt_scope():
            iou_thres_gm = self.input_gm_list[3]
            iou_thres_ub = self.tik_instance.Tensor("float32", [Constant.DATA_EACH_BLOCK_32], 
                                                    name="iou_thres_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(iou_thres_ub, iou_thres_gm, 0, 1, 1, 0, 0)
            if self.dtype == "float32":
                self.iou_thresh.set_as(iou_thres_ub[0])
                self.iou_thresh_plus_one.set_as(self.iou_thresh + 1.0)
            else:
                iou_temp = self.tik_instance.Scalar("float32", init_value=iou_thres_ub[0])
                iou_temp_plus_one = self.tik_instance.Scalar("float32", init_value=iou_temp + 1.0)
                self.tik_instance.scalar_conv("none", self.iou_thresh, iou_temp)
                self.tik_instance.scalar_conv("none", self.iou_thresh_plus_one, iou_temp_plus_one)
            
    def init_input_gm(self, max_output_size, iou_threshold, score_threshold):
        """
        init input gm.
        """
        boxes_gm = self.tik_instance.Tensor(self.dtype, self.boxes_shape, name="boxes_gm", scope=tik.scope_gm)
        scores_gm = self.tik_instance.Tensor(self.dtype, self.scores_shape, name="scores_gm", scope=tik.scope_gm)
        index_id_gm = self.tik_instance.Tensor("float16", self.index_id_shape, name="index_id_gm", scope=tik.scope_gm)

        if max_output_size is None:
            max_output_size_gm = None
        else:
            max_output_size_shape = list(max_output_size.get("shape"))
            max_output_size_gm = self.tik_instance.Tensor("int32",
                                                          max_output_size_shape,
                                                          name="max_output_size_gm",
                                                          scope=tik.scope_gm)
        
        if iou_threshold is None:
            iou_threshold_gm = None
        else:
            iou_threshold_shape = list(iou_threshold.get("shape"))
            iou_threshold_gm = self.tik_instance.Tensor("float32",
                                                        iou_threshold_shape,
                                                        name="iou_threshold_gm",
                                                        scope=tik.scope_gm)
        
        if score_threshold is None:
            score_threshold_gm = None
        else:
            score_threshold_shape = list(score_threshold.get("shape"))
            score_threshold_gm = self.tik_instance.Tensor("float32",
                                                          score_threshold_shape,
                                                          name="score_threshold_gm",
                                                          scope=tik.scope_gm)
        
        self.input_gm_list = [boxes_gm, scores_gm, max_output_size_gm, iou_threshold_gm,
                              score_threshold_gm, index_id_gm]
    
    def init_output_gm(self):
        """
        init output gm.
        """
        selected_indices_shape = [self.max_boxes_size, Constant.INDEX_NUM]
        self.selected_indices_gm = self.tik_instance.Tensor("int32",
                                                            selected_indices_shape,
                                                            name="selected_indices_gm",
                                                            scope=tik.scope_gm)

    def get_score_threshold(self):
        """
        get score thresh.
        """
        with self.tik_instance.new_stmt_scope():
            score_thres_gm = self.input_gm_list[4]
            score_thres_ub = self.tik_instance.Tensor("float32", [Constant.DATA_EACH_BLOCK_32], 
                                                      name="score_thres_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(score_thres_ub, score_thres_gm, 0, 1, 1, 0, 0)
            self.score_thresh.set_as(score_thres_ub[0])
    
    def run_tik(self, kernel_name_value):
        """
        divide task into different cores.
        """
        with self.tik_instance.for_range(0, self.core_used, block_num=self.core_used) as _core_idx:
            if self.class_per_core == self.class_core_tail or self.core_used == 1:
                with self.tik_instance.for_range(0, self.class_per_core) as _class_idx:
                    self.nms_for_single_class(self.class_per_core * _core_idx + _class_idx, _core_idx)
            else:
                with self.tik_instance.if_scope(_core_idx < self.core_used - 1):
                    with self.tik_instance.for_range(0, self.class_per_core) as _class_idx:
                        self.nms_for_single_class(self.class_per_core * _core_idx + _class_idx, _core_idx)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, self.class_core_tail) as _class_idx:
                        self.nms_for_single_class(self.class_per_core * _core_idx + _class_idx, _core_idx)
            
            # move out result from gm_nms_result[self.all_classes, self.max_selected_nms_num_in_ub, 8] every core.
            self.core_multi_class_output(_core_idx)

        self.tik_instance.BuildCCE(kernel_name=kernel_name_value,
                                   inputs=self.input_gm_list,
                                   outputs=self.selected_indices_gm)
        
        return self.tik_instance
    
    def core_multi_class_output(self, core_id):
        """
        move ub from each core to gm.
        """
        class_num = self.tik_instance.Scalar("int32", init_value=self.class_per_core)
        is_last_core = self.tik_instance.Scalar("int32", init_value=0)
        with self.tik_instance.if_scope(core_id == self.core_used - 1):
            class_num.set_as(self.class_core_tail)
            is_last_core.set_as(1)

        apply_len_tail = (self.max_class_size * Constant.INDEX_NUM) % Constant.DATA_EACH_BLOCK_32
        if apply_len_tail == 0:
            no_tail = True
        else:
            no_tail = False
        
        with self.tik_instance.for_range(0, class_num) as _class_id:
            class_offset = (core_id * self.class_per_core + _class_id) * \
                           self.max_selected_nms_num_in_ub * Constant.PROPOSAL_LEN
            ub_result_boxes = self.tik_instance.Tensor(self.dtype,
                                                       [self.max_selected_nms_num_in_ub, Constant.PROPOSAL_LEN],
                                                       name="ub_result_boxes", scope=tik.scope_ubuf)
            loop_burst_len = (self.max_selected_nms_num_in_ub * Constant.PROPOSAL_LEN) // self.data_each_block
            self.tik_instance.data_move(ub_result_boxes, self.gm_nms_result[class_offset], 0, 1, loop_burst_len, 0, 0)
            
            ub_index_temp_int32 = self.tik_instance.Tensor("int32",
                                                           [self.max_selected_nms_num_in_ub, Constant.INDEX_NUM],
                                                           name="ub_index_temp_int32", scope=tik.scope_ubuf)
            
            with self.tik_instance.new_stmt_scope():
                ub_index_temp = self.tik_instance.Tensor(self.dtype,
                                                         [self.max_selected_nms_num_in_ub, Constant.INDEX_NUM],
                                                         name="ub_index_temp", scope=tik.scope_ubuf)

                with self.tik_instance.for_range(0, self.max_selected_nms_num_in_ub) as i:
                    with self.tik_instance.for_range(0, 2) as j:
                        ub_index_temp[i * Constant.INDEX_NUM + j] = ub_result_boxes[i * Constant.PROPOSAL_LEN + j]
                
                loop_len = ceil_div(self.max_selected_nms_num_in_ub, Constant.RPN_PROPOSAL_NUM)
                with self.tik_instance.for_range(0, loop_len) as i:
                    mask = Constant.INDEX_NUM * Constant.RPN_PROPOSAL_NUM
                    offset_conv = i * mask
                    if self.dtype == "float16":
                        src_block = Constant.INDEX_NUM
                    else:
                        src_block = Constant.INDEX_NUM * 2
                    self.tik_instance.vec_conv(mask, "round", ub_index_temp_int32[offset_conv],
                                            ub_index_temp[offset_conv], 1, 6, src_block)
            
            with self.tik_instance.new_stmt_scope():
                self.tik_func_index_recover(ub_result_boxes, ub_index_temp_int32)
            
            with self.tik_instance.new_stmt_scope():
                ub_out_scores = self.tik_instance.Tensor(self.dtype, [self.max_selected_nms_num_in_ub],
                                                         name="ub_out_scores", scope=tik.scope_ubuf)
                with self.tik_instance.for_range(0, self.max_selected_nms_num_in_ub) as i:
                    ub_out_scores[i] = ub_result_boxes[i * Constant.PROPOSAL_LEN + Constant.INDEX_SCORE]
                
                self.filter_score_compute(ub_index_temp_int32, ub_out_scores)

            gm_offset = (core_id * self.class_per_core + _class_id) * self.max_class_size * Constant.INDEX_NUM
            with self.tik_instance.if_scope(tik.any(no_tail, _class_id < class_num - 1, is_last_core)):
                # move out directly.
                apply_gm_len = ceil_div(self.max_class_size * Constant.INDEX_NUM, Constant.PROPOSAL_LEN)
                self.tik_instance.data_move(self.selected_indices_gm[gm_offset], ub_index_temp_int32,
                                            0, 1, apply_gm_len, 0, 0)
            with self.tik_instance.else_scope():
                apply_len = (self.max_class_size * Constant.INDEX_NUM) // Constant.DATA_EACH_BLOCK_32
                with self.tik_instance.if_scope(apply_len > 0):
                    self.tik_instance.data_move(self.selected_indices_gm[gm_offset], ub_index_temp_int32,
                                                0, 1, apply_len, 0, 0)
                    temp_block_ub = self.tik_instance.Tensor("int32", [Constant.DATA_EACH_BLOCK_32], 
                                                             name="temp_block_ub", scope=tik.scope_ubuf)
                    start_offset = apply_len * Constant.DATA_EACH_BLOCK_32 - \
                                   (Constant.DATA_EACH_BLOCK_32 - apply_len_tail)
                    for i in range(Constant.DATA_EACH_BLOCK_32):
                        temp_block_ub[i] = ub_index_temp_int32[start_offset + i]
                    self.tik_instance.data_move(self.selected_indices_gm[gm_offset + start_offset], temp_block_ub,
                                                0, 1, 1, 0, 0)                     
    
    def filter_score_compute(self, index_ub, ub_out_scores):
        """
        filter the score less than score thresh.
        """
        ub_score_scalar = self.tik_instance.Scalar("float32", init_value=0.0)
        if self.dtype == "float16":
            ub_score_fp32 = self.tik_instance.Tensor("float32", [self.max_selected_nms_num_in_ub],
                                                    name="ub_score_fp32", scope=tik.scope_ubuf)
            self.vec_conv_fp16_to_fp32(ub_out_scores, ub_score_fp32, self.max_selected_nms_num_in_ub)

        with self.tik_instance.for_range(0, self.max_selected_nms_num_in_ub) as i:
            if self.dtype == "float16":
                ub_score_scalar.set_as(ub_score_fp32[i])
            else:
                ub_score_scalar.set_as(ub_out_scores[i])
            
            with self.tik_instance.if_scope(ub_score_scalar <= self.score_thresh):
                index_ub[i, 0].set_as(Constant.INDEX_MINUS_ONE)
                index_ub[i, 1].set_as(Constant.INDEX_MINUS_ONE)
                index_ub[i, 2].set_as(Constant.INDEX_MINUS_ONE)

    def tik_func_index_recover(self, ub_result_boxes, ub_index_temp_int32):
        """
        recover index_id = index_id1 * 1000 + index_id2
        """
        with self.tik_instance.new_stmt_scope():
            index_id_head = self.tik_instance.Tensor(self.dtype, [self.max_selected_nms_num_in_ub],
                                                    name="index_id_head", scope=tik.scope_ubuf)
            index_id_tail = self.tik_instance.Tensor(self.dtype, [self.max_selected_nms_num_in_ub],
                                                    name="index_id_tail", scope=tik.scope_ubuf)
            thousand_ub = self.tik_instance.Tensor("int32", [self.max_selected_nms_num_in_ub],
                                                name="thousand_ub", scope=tik.scope_ubuf)
            input_a_ub_int32 = self.tik_instance.Tensor("int32", [self.max_selected_nms_num_in_ub],
                                                        name="input_a_ub_int32", scope=tik.scope_ubuf)
            input_b_ub_int32 = self.tik_instance.Tensor("int32", [self.max_selected_nms_num_in_ub],
                                                        name="input_b_ub_int32", scope=tik.scope_ubuf)
            self.tik_func_vector(thousand_ub, Constant.ONE_THOUSAND, self.max_selected_nms_num_in_ub, Constant.MASK_32)
            with self.tik_instance.for_range(0, self.max_selected_nms_num_in_ub) as i:
                index_id_head[i] = ub_result_boxes[i * Constant.PROPOSAL_LEN + 2]
                index_id_tail[i] = ub_result_boxes[i * Constant.PROPOSAL_LEN + 3]
            
            apply_lens = ceil_div(self.max_selected_nms_num_in_ub, Constant.RPN_PROPOSAL_NUM)
            if self.dtype == "float16":
                src_block = 1
            else:
                src_block = 2
            self.tik_instance.vec_conv(Constant.RPN_PROPOSAL_NUM, "round", input_a_ub_int32, index_id_head,
                                       apply_lens, 2, src_block)
            self.tik_instance.vec_conv(Constant.RPN_PROPOSAL_NUM, "round", input_b_ub_int32, index_id_tail,
                                       apply_lens, 2, src_block)                
            
            with self.tik_instance.new_stmt_scope():
                temp_ub = self.tik_instance.Tensor("int32", [self.max_selected_nms_num_in_ub],
                                                name="temp_ub", scope=tik.scope_ubuf)
                self.tik_func_tensors_calculate("vmul", temp_ub, input_a_ub_int32, thousand_ub,
                                                self.max_selected_nms_num_in_ub, Constant.MASK_32)
                self.tik_func_tensors_calculate("vadd", input_a_ub_int32, temp_ub, input_b_ub_int32,
                                                self.max_selected_nms_num_in_ub, Constant.MASK_32)
            
            with self.tik_instance.for_range(0, self.max_selected_nms_num_in_ub) as i:
                ub_index_temp_int32[i * Constant.INDEX_NUM + 2] = input_a_ub_int32[i]

    def do_topk(self, task_id, core_id, ub_max_topk, ub_topk_index):
        """
        get score sorted.
        """
        # compute batch_id and class_id according to task_id
        batch_id = task_id // self.classes
        class_id = task_id % self.classes

        gm_offset = [batch_id, class_id, 0]

        if self.single_loop:
            self.get_sorted_proposal_compute(ub_max_topk, ub_topk_index, gm_offset, self.boxes_num,
                ceil_div(self.boxes_num, Constant.RPN_PROPOSAL_NUM) * Constant.RPN_PROPOSAL_NUM,
                self.center_point_box)
        else:
            self.get_sorted_proposal_compute(ub_max_topk, ub_topk_index, gm_offset,
                                             self.proposal_topk, self.proposal_topk, self.center_point_box)
        
            # do topk k proposal loop to get final top proposal_topk_k proposals to ub_max_topk
            with self.tik_instance.new_stmt_scope():
                ub_tmp_topk = self.tik_instance.Tensor(self.dtype, (self.proposal_topk, Constant.PROPOSAL_LEN), 
                                                    name="ub_tmp_topk", scope=tik.scope_ubuf)
                ub_tmp_topk_index = self.tik_instance.Tensor(self.dtype, (self.proposal_topk, Constant.PROPOSAL_LEN),
                                                        name="ub_tmp_topk_index", scope=tik.scope_ubuf)
                if self.topk_loop_time > 1:
                    with self.tik_instance.for_range(1, self.topk_loop_time) as _top_k_idx:
                        gm_offset = [batch_id, class_id, _top_k_idx * self.proposal_topk]
                        workspace_offset = (_top_k_idx - 1) * self.proposal_topk * Constant.PROPOSAL_LEN + \
                                           core_id * self.boxes_num * Constant.PROPOSAL_LEN
                        workspace_for_save_proposal = self.workspace_boxes_gm[workspace_offset]
                        workspace_for_save_index_proposal = self.workspace_index_gm[workspace_offset]

                        # get tmp sorted proposal to ub_tmp_topk.
                        self.get_sorted_proposal_compute(ub_tmp_topk, ub_tmp_topk_index, gm_offset,
                                                        self.proposal_topk, self.proposal_topk, self.center_point_box)
                        
                        self.tik_func_sort_with_ub([ub_tmp_topk, ub_max_topk], [ub_max_topk, ub_tmp_topk],
                                                   self.proposal_topk, workspace_for_save_proposal)
                        self.tik_func_sort_with_ub([ub_tmp_topk_index, ub_topk_index],
                                                   [ub_topk_index, ub_tmp_topk_index],
                                                   self.proposal_topk, workspace_for_save_index_proposal)
                
                if self.topk_loop_tail != 0:
                    gm_offset = [batch_id, class_id, self.topk_loop_time * self.proposal_topk]
                    workspace_offset = (self.topk_loop_time - 1) * self.proposal_topk * Constant.PROPOSAL_LEN + \
                                       core_id * self.boxes_num * Constant.PROPOSAL_LEN
                    workspace_for_save_proposal = self.workspace_boxes_gm[workspace_offset]
                    workspace_for_save_index_proposal = self.workspace_index_gm[workspace_offset]

                    # get tmp sorted proposal to ub_tmp_topk.
                    self.get_sorted_proposal_compute(ub_tmp_topk, ub_tmp_topk_index, gm_offset,
                                                    self.topk_loop_tail, self.proposal_topk, self.center_point_box)
                    
                    self.tik_func_sort_with_ub([ub_tmp_topk, ub_max_topk], [ub_max_topk, ub_tmp_topk],
                                               self.proposal_topk, workspace_for_save_proposal)
                    self.tik_func_sort_with_ub([ub_tmp_topk_index, ub_topk_index], [ub_topk_index, ub_tmp_topk_index],
                                               self.proposal_topk, workspace_for_save_index_proposal)

    def do_nms(self, task_id, core_id, ub_max_topk, ub_topk_index):
        """
        get boxes id from sorted topk proposals.
        """
        ub_selected_proposals = self.tik_instance.Tensor(self.dtype, 
                                                         [self.max_selected_nms_num_in_ub, Constant.PROPOSAL_LEN],
                                                         name="ub_selected_proposals", scope=tik.scope_ubuf)
        ub_selected_index_proposals = self.tik_instance.Tensor(self.dtype, 
                                                               [self.max_selected_nms_num_in_ub, Constant.PROPOSAL_LEN],
                                                               name="ub_selected_index_proposals",
                                                               scope=tik.scope_ubuf)
        ub_selected_area = self.tik_instance.Tensor(self.dtype, [self.max_selected_nms_num_in_ub],
                                                    name="ub_selected_area", scope=tik.scope_ubuf)
        ub_sup_vec = self.tik_instance.Tensor("uint16", [self.max_selected_nms_num_in_ub],
                                              name="ub_sup_vec", scope=tik.scope_ubuf)
        
        # init selected_nms_proposals for one class to 0.
        self.selected_proposals_cnt.set_as(0)
        
        # init all sup_vec to 1, mean: no select proposal
        self.tik_func_vector(ub_sup_vec, 1, self.max_selected_nms_num_in_ub, Constant.MASK_16)

        nms_offset = task_id * self.max_selected_nms_num_in_ub * Constant.PROPOSAL_LEN
        loop_burst_len = (self.max_selected_nms_num_in_ub * Constant.PROPOSAL_LEN) // self.data_each_block
        self.tik_instance.data_move(ub_selected_proposals, self.nms_temp_zero, 0, 1, loop_burst_len, 0, 0)
        self.tik_instance.data_move(ub_selected_index_proposals, self.nms_temp_minus_one,
                                    0, 1, loop_burst_len, 0, 0)
        self.tik_instance.data_move(self.gm_nms_result[nms_offset], ub_selected_index_proposals,
                                    0, 1, loop_burst_len, 0, 0)

        # init select nms area = 0
        loop_burst_len = self.max_selected_nms_num_in_ub // self.data_each_block
        self.tik_instance.data_move(ub_selected_area, self.nms_temp_zero, 0, 1, loop_burst_len, 0, 0)

        with self.tik_instance.new_stmt_scope():
            self.do_nms_compute(ub_max_topk, ub_topk_index, ub_selected_proposals, ub_selected_index_proposals, 
                                ub_selected_area, ub_sup_vec)

        burst_lens_gm = ceil_div(self.selected_proposals_cnt * Constant.PROPOSAL_LEN, self.data_each_block)
        self.tik_instance.data_move(self.gm_nms_result[nms_offset], ub_selected_index_proposals,
                                    0, 1, burst_lens_gm, 0, 0)

        with self.tik_instance.if_scope(self.selected_proposals_cnt < self.max_class_size):
            # if the selected nms output num of the first topk highest score boxes is less the output need
            # do nms again from the tail boxes topk boxes by topk boxes
            self.second_nms_compute(core_id, ub_max_topk, ub_topk_index, ub_selected_proposals, 
                                    ub_selected_index_proposals, ub_selected_area, ub_sup_vec)

        burst_lens_gm = ceil_div(self.selected_proposals_cnt * Constant.PROPOSAL_LEN, self.data_each_block)
        self.tik_instance.data_move(self.gm_nms_result[nms_offset], ub_selected_index_proposals,
                                    0, 1, burst_lens_gm, 0, 0)

    def second_nms_compute(self, core_id, ub_max_topk, ub_topk_index, ub_selected_proposals, 
                           ub_selected_index_proposals, ub_selected_area, ub_sup_vec):
        """
        do second nms when selected nms is less the output need.
        """
        if self.topk_loop_tail == 0:
            tool_loop = self.topk_loop_time
        else:
            tool_loop = self.topk_loop_time + 1
        if tool_loop >= 2:
            with self.tik_instance.for_range(1, tool_loop) as _top_n_idx:
                top_n_num_tail = tool_loop - _top_n_idx - 1
                with self.tik_instance.if_scope(self.selected_proposals_cnt < self.max_class_size):
                    # copy sorted proposals to ub_max_topk
                    loop_burst_len = ceil_div(self.proposal_topk * Constant.PROPOSAL_LEN, self.data_each_block)
                    wk_offset = core_id * self.boxes_num * Constant.PROPOSAL_LEN
                    self.tik_instance.data_move(ub_max_topk, self.workspace_boxes_gm[wk_offset],
                                                0, 1, loop_burst_len, 0, 0)
                    self.tik_instance.data_move(ub_topk_index, self.workspace_index_gm[wk_offset],
                                                0, 1, loop_burst_len, 0, 0)
                    
                    # apply second topk proposals.
                    with self.tik_instance.if_scope(top_n_num_tail > 0):
                        ub_tmp_topk = self.tik_instance.Tensor(self.dtype,
                                                               (self.proposal_topk, Constant.PROPOSAL_LEN),
                                                               name="ub_tmp_topk", scope=tik.scope_ubuf)
                        ub_tmp_topk_index = self.tik_instance.Tensor(self.dtype, 
                                                                     (self.proposal_topk, Constant.PROPOSAL_LEN),
                                                                     name="ub_tmp_topk_index",
                                                                     scope=tik.scope_ubuf)
                        with self.tik_instance.for_range(0, top_n_num_tail) as _top_n_tail_idx:
                            workspace_proposal_offset = self.proposal_topk * Constant.PROPOSAL_LEN + \
                                _top_n_tail_idx * self.proposal_topk * Constant.PROPOSAL_LEN + \
                                core_id * self.boxes_num * Constant.PROPOSAL_LEN
                            self.tik_instance.data_move(ub_tmp_topk,
                                                        self.workspace_boxes_gm[workspace_proposal_offset],
                                                        0, 1, loop_burst_len, 0, 0)
                            self.tik_instance.data_move(ub_tmp_topk_index,
                                                        self.workspace_index_gm[workspace_proposal_offset],
                                                        0, 1, loop_burst_len, 0, 0)
                            workspace_offset = _top_n_tail_idx * self.proposal_topk * Constant.PROPOSAL_LEN + \
                                                core_id * self.boxes_num * Constant.PROPOSAL_LEN
                            workspace_for_save_proposal = self.workspace_boxes_gm[workspace_offset]
                            workspace_for_save_index_proposal = self.workspace_index_gm[workspace_offset]
                            self.tik_func_sort_with_ub([ub_tmp_topk, ub_max_topk], [ub_max_topk, ub_tmp_topk],
                                                        self.proposal_topk, workspace_for_save_proposal)
                            self.tik_func_sort_with_ub([ub_tmp_topk_index, ub_topk_index],
                                                        [ub_topk_index, ub_tmp_topk_index],
                                                        self.proposal_topk, workspace_for_save_index_proposal)

                    # do nms use topk output to get nms proposals per class
                    # and move result to gm
                    with self.tik_instance.new_stmt_scope():
                        self.do_nms_compute(ub_max_topk, ub_topk_index, ub_selected_proposals, 
                                            ub_selected_index_proposals, ub_selected_area, ub_sup_vec)

    # pylint: disable=too-many-arguments
    def nms_for_single_class(self, task_id, core_id):
        """
        do topk and then do nms.
        """
        # 1.get first topk proposals
        ub_max_topk = self.tik_instance.Tensor(self.dtype, (self.proposal_topk, Constant.PROPOSAL_LEN),
                                               name="ub_max_topk", scope=tik.scope_ubuf)
        ub_topk_index = self.tik_instance.Tensor(self.dtype, (self.proposal_topk, Constant.PROPOSAL_LEN),
                                                 name="ub_topk_index", scope=tik.scope_ubuf)

        with self.tik_instance.new_stmt_scope():
            self.do_topk(task_id, core_id, ub_max_topk, ub_topk_index)

        with self.tik_instance.new_stmt_scope():
            self.do_nms(task_id, core_id, ub_max_topk, ub_topk_index)

    def do_nms_compute(self, ub_max_topk, ub_topk_index, ub_selected_proposals, ub_selected_index_proposals, 
                       ub_selected_area, ub_sup_vec):
        """
        compute nms for TOPK.
        """
        total_input_proposal_num = self.proposal_topk
        if self.boxes_num  < self.proposal_topk:
            total_input_proposal_num = self.boxes_num

        # init handling proposals and left proposals.
        handling_proposals_cnt = self.tik_instance.Scalar(dtype="uint16", init_value=0)
        left_proposal_cnt = self.tik_instance.Scalar(dtype="uint16", init_value=total_input_proposal_num)

        # init ub.
        temp_sup_vec_ub = self.tik_instance.Tensor("uint16", [self.burst_proposal_num],
                                                   name = "temp_sup_vec_ub", scope=tik.scope_ubuf)
        temp_proposals_ub = self.tik_instance.Tensor(self.dtype, [self.burst_proposal_num, Constant.PROPOSAL_LEN],
                                                     name="temp_proposals_ub", scope=tik.scope_ubuf)
        temp_index_proposals_ub = self.tik_instance.Tensor(self.dtype, [self.burst_proposal_num, Constant.PROPOSAL_LEN],
                                                           name="temp_index_proposals_ub", scope=tik.scope_ubuf)
        temp_area_ub = self.tik_instance.Tensor(self.dtype, [self.burst_proposal_num],
                                                name="temp_area_ub", scope=tik.scope_ubuf)
        temp_iou_ub = self.tik_instance.Tensor(self.dtype, 
            [self.max_selected_nms_num_in_ub + self.burst_proposal_num, Constant.RPN_PROPOSAL_NUM],
            name="temp_iou_ub",
            scope=tik.scope_ubuf)
        temp_join_ub = self.tik_instance.Tensor(self.dtype, 
            [self.max_selected_nms_num_in_ub + self.burst_proposal_num, Constant.RPN_PROPOSAL_NUM],
            name="temp_join_ub",
            scope=tik.scope_ubuf)
        temp_sup_matrix_ub = self.tik_instance.Tensor("uint16",
            [self.max_selected_nms_num_in_ub + self.burst_proposal_num],
            name="temp_sup_matrix_ub",
            scope=tik.scope_ubuf)
        
        ub_temp_list = [temp_sup_vec_ub, temp_proposals_ub, temp_index_proposals_ub, temp_area_ub,
                        ub_selected_proposals, ub_selected_index_proposals, ub_selected_area, ub_sup_vec]
        
        # one time process proposals.
        burst_lens = ceil_div(total_input_proposal_num, self.burst_proposal_num)
        with self.tik_instance.for_range(0, burst_lens) as burst_index:
            # update handling proposals for this burst.
            with self.tik_instance.if_scope(left_proposal_cnt < self.burst_proposal_num):
                handling_proposals_cnt.set_as(left_proposal_cnt)
            with self.tik_instance.else_scope():
                handling_proposals_cnt.set_as(self.burst_proposal_num)
            
            handling_ceil = self.tik_instance.Scalar(dtype="uint16", init_value=0)
            handling_ceil.set_as(ceil_div(handling_proposals_cnt, Constant.RPN_PROPOSAL_NUM))
            selected_ceil = self.tik_instance.Scalar(dtype="uint16", init_value=0)
            selected_ceil.set_as(ceil_div(self.selected_proposals_cnt, Constant.RPN_PROPOSAL_NUM))

            # clear temp_sup_vec_ub to all 1.
            self.tik_instance.vector_dup(self.burst_proposal_num, temp_sup_vec_ub,
                                         1, 1, 1, 8)
            start_index = burst_index * self.burst_proposal_num * Constant.PROPOSAL_LEN
            move_times = self.burst_proposal_num * Constant.PROPOSAL_LEN // self.data_each_block
            self.tik_instance.data_move(temp_proposals_ub, ub_max_topk[start_index], 0, 1, move_times, 0, 0)
            self.tik_instance.data_move(temp_index_proposals_ub, ub_topk_index[start_index],
                                        0, 1, move_times, 0, 0)
            
            # calculate the area of proposals of this burst.
            self.tik_instance.vrpac(temp_area_ub, temp_proposals_ub, handling_ceil)

            length = self.tik_instance.Scalar(dtype="uint16", init_value=0)
            length.set_as(selected_ceil * Constant.RPN_PROPOSAL_NUM)
            with self.tik_instance.if_scope(self.selected_proposals_cnt < self.max_class_size):
                with self.tik_instance.for_range(0, handling_ceil) as i:
                    length.set_as(length + Constant.RPN_PROPOSAL_NUM)
                    # calculate intersection between selected_proposals and handling proposals.
                    self.tik_instance.viou(temp_iou_ub, ub_selected_proposals,
                                           temp_proposals_ub[i * Constant.RPN_PROPOSAL_NUM * Constant.PROPOSAL_LEN],
                                           selected_ceil)
                    self.tik_instance.viou(temp_iou_ub[selected_ceil * Constant.RPN_PROPOSAL_NUM, 0], temp_proposals_ub,
                                           temp_proposals_ub[i * Constant.RPN_PROPOSAL_NUM * Constant.PROPOSAL_LEN],
                                           i + 1)
                    # calculate join between selected_proposals and handling proposals.
                    self.tik_instance.vaadd(temp_join_ub, ub_selected_area,
                                            temp_area_ub[i * Constant.RPN_PROPOSAL_NUM], selected_ceil)
                    self.tik_instance.vaadd(temp_join_ub[selected_ceil * Constant.RPN_PROPOSAL_NUM, 0], temp_area_ub,
                                            temp_area_ub[i * Constant.RPN_PROPOSAL_NUM], i + 1)
                    # calculate join * (thresh/(1+thresh))
                    self.join_mul_iou(temp_join_ub, length)
                    # compare ange generate suppression matrix
                    self.tik_instance.vcmpv_gt(temp_sup_matrix_ub, temp_iou_ub, temp_join_ub,
                                               ceil_div(length * 2, self.data_each_block), 1, 1, 8, 8)

                    # generate suppression vector.
                    # clear rpn_cor_ir
                    rpn_cor_ir = self.tik_instance.set_rpn_cor_ir(0)
                    rpn_cor_ir = self.tik_instance.rpn_cor(temp_sup_matrix_ub, ub_sup_vec, 1, 1, selected_ceil)
                    with self.tik_instance.if_scope(i > 0):
                        rpn_cor_ir = self.tik_instance.rpn_cor(
                            temp_sup_matrix_ub[selected_ceil * Constant.RPN_PROPOSAL_NUM],
                            temp_sup_vec_ub, 1, 1, i)
                    self.tik_instance.rpn_cor_diag(temp_sup_vec_ub[i * Constant.RPN_PROPOSAL_NUM], 
                                                   temp_sup_matrix_ub[length - Constant.RPN_PROPOSAL_NUM], rpn_cor_ir)

                # find & move unsuppressed proposals.
                self.move_unsuppressed_proposals(handling_proposals_cnt, ub_temp_list)

                left_proposal_cnt.set_as(left_proposal_cnt - handling_proposals_cnt)

    def move_unsuppressed_proposals(self, handling_proposals_cnt, ub_temp_list):
        """
        find & move unsuppressed proposals.
        """
        # init nms_flag.
        nms_flag = self.tik_instance.Scalar(dtype="uint16", init_value=0)
        temp_sup_vec_ub = ub_temp_list[0]
        temp_proposals_ub = ub_temp_list[1]
        temp_index_proposals_ub = ub_temp_list[2]
        temp_area_ub = ub_temp_list[3]
        ub_selected_proposals = ub_temp_list[4]
        ub_selected_index_proposals = ub_temp_list[5]
        ub_selected_area = ub_temp_list[6]
        ub_sup_vec = ub_temp_list[7]
        
        with self.tik_instance.for_range(0, handling_proposals_cnt) as i:
            with self.tik_instance.if_scope(self.selected_proposals_cnt < self.max_class_size):
                nms_flag.set_as(temp_sup_vec_ub[i])
                with self.tik_instance.if_scope(nms_flag == 0):
                    ub_selected_proposals_int64 = ub_selected_proposals.reinterpret_cast_to("int64")
                    temp_proposals_ub_int64 = temp_proposals_ub.reinterpret_cast_to("int64")
                    ub_selected_index_proposals_int64 = ub_selected_index_proposals.reinterpret_cast_to("int64")
                    temp_index_proposals_ub_int64 = temp_index_proposals_ub.reinterpret_cast_to("int64")
                    if self.dtype == "float16":
                        lens = Constant.PROPOSAL_LEN // (Constant.BYTE_PER_DATA_64 // Constant.BYTE_PER_DATA_16)
                    else:
                        lens = Constant.PROPOSAL_LEN // (Constant.BYTE_PER_DATA_64 // Constant.BYTE_PER_DATA_32)
                    for j in range(lens):
                        ub_selected_proposals_int64[self.selected_proposals_cnt * lens + j] = \
                            temp_proposals_ub_int64[i * lens + j]
                        ub_selected_index_proposals_int64[self.selected_proposals_cnt * lens + j] = \
                            temp_index_proposals_ub_int64[i * lens + j]
                    ub_selected_area[self.selected_proposals_cnt] = temp_area_ub[i]
                    # update sup_vec_ub
                    ub_sup_vec[self.selected_proposals_cnt].set_as(0)
                    # update counter
                    self.selected_proposals_cnt.set_as(self.selected_proposals_cnt + 1)

    def join_mul_iou(self, temp_join_ub, length):
        """
        calculate join muls iou.
        """
        with self.tik_instance.new_stmt_scope():
            self.tik_instance.vmuls(self.mask, temp_join_ub, temp_join_ub, self.iou_thresh,
                                    ceil_div(length * 2, self.data_each_block), 1, 1, 8, 8)
                        
            iou_plus_one_ub = self.tik_instance.Tensor(self.dtype, [length, Constant.RPN_PROPOSAL_NUM],
                                                    name="iou_plus_one_ub", scope=tik.scope_ubuf)
            repeat = (length * Constant.RPN_PROPOSAL_NUM) // self.mask
            repeat_tail = (length * Constant.RPN_PROPOSAL_NUM) % self.mask
            offset = 0
            with self.tik_instance.if_scope(repeat > 0):
                self.tik_instance.vector_dup(self.mask, iou_plus_one_ub, self.iou_thresh_plus_one, repeat, 1, 8)
                offset = offset + self.mask * repeat
            with self.tik_instance.if_scope(repeat_tail > 0):
                self.tik_instance.vector_dup(repeat_tail, iou_plus_one_ub[offset], self.iou_thresh_plus_one, 1, 1, 8)
            offset = 0
            with self.tik_instance.if_scope(repeat > 0):
                self.tik_instance.vdiv(self.mask, temp_join_ub, temp_join_ub, iou_plus_one_ub, repeat,
                                        1, 1, 1, 8, 8, 8)
                offset = offset + self.mask * repeat
            with self.tik_instance.if_scope(repeat_tail > 0):
                self.tik_instance.vdiv(repeat_tail, temp_join_ub[offset], temp_join_ub[offset], iou_plus_one_ub,
                                    1, 1, 1, 1, 8, 8, 8)

    def tik_func_sort_with_ub(self, src_ub_list, dst_ub_list, sorted_num, workspace_for_proposal):
        """
        sort with ub.
        """
        with self.tik_instance.new_stmt_scope():
            list_len = len(src_ub_list)
            ub_dst_sort_with_ub = self.tik_instance.Tensor(self.dtype, [list_len * sorted_num * Constant.PROPOSAL_LEN],
                                                        name="ub_dst_sort_with_ub",
                                                        scope=tik.scope_ubuf)
            sort_with_ub(self.tik_instance, src_ub_list, ub_dst_sort_with_ub, sorted_num)
            loop_burst_len = (sorted_num * Constant.PROPOSAL_LEN) // self.data_each_block
            self.tik_instance.data_move(dst_ub_list[0], ub_dst_sort_with_ub, 0, 1, loop_burst_len, 0, 0)
            self.tik_instance.data_move(workspace_for_proposal,
                                        ub_dst_sort_with_ub[sorted_num * Constant.PROPOSAL_LEN:],
                                        0, 1, loop_burst_len, 0, 0)
    
    def get_sorted_proposal_compute(self, output_ub, output_index_ub, gm_offset, copy_num, sorted_num, center_box):
        """
        get sorted proposals.
        """
        with self.tik_instance.new_stmt_scope():
            # apply ub for boxes, score, index
            ub_tmp_boxes = self.tik_instance.Tensor(self.dtype, [Constant.BOX_LOC, sorted_num],
                                                    name="copy_ub_tmp_boxes", scope=tik.scope_ubuf)
            ub_tmp_score = self.tik_instance.Tensor(self.dtype, [1, sorted_num],
                                                    name="copy_ub_tmp_score", scope=tik.scope_ubuf)
            ub_tmp_index = self.tik_instance.Tensor(self.dtype, [Constant.BOX_LOC, sorted_num],
                                                    name="copy_ub_tmp_index", scope=tik.scope_ubuf)
            input_ub_list = [ub_tmp_boxes, ub_tmp_score, ub_tmp_index]
            
            # update boxes, score and index ub.
            self.clip_window_compute(input_ub_list, copy_num, gm_offset)

            # scaling box data to avoid overflow.
            if self.dtype == "float16":
                self.change_coordinate_frame_compute(ub_tmp_boxes, copy_num, sorted_num)

            # add rpn_offset.
            self.tik_func_tensor_scal_cacul("vadds", ub_tmp_boxes, ub_tmp_boxes, Constant.RPN_OFFSET,
                                            copy_num, sorted_num * 2)
            self.tik_func_tensor_scal_cacul("vadds", ub_tmp_boxes, ub_tmp_boxes, Constant.RPN_OFFSET,
                                            copy_num, sorted_num * 3)
            
            boxes_list = [ub_tmp_boxes[0], ub_tmp_boxes[sorted_num], 
                          ub_tmp_boxes[sorted_num * 2], ub_tmp_boxes[sorted_num * 3]]
            index_list = [ub_tmp_index[0], ub_tmp_index[sorted_num],
                          ub_tmp_index[sorted_num * 2], ub_tmp_index[sorted_num * 3]]

            # fullfill the tail of copy_num to 0.0.
            if copy_num % Constant.RPN_PROPOSAL_NUM != 0:
                dup_mask = int("0" * (Constant.MASK_32 - Constant.RPN_PROPOSAL_NUM) + \
                               "1" * (Constant.RPN_PROPOSAL_NUM - (copy_num % Constant.RPN_PROPOSAL_NUM)) + \
                               "0" * (copy_num % Constant.RPN_PROPOSAL_NUM), 2)
                self.tik_instance.vector_dup([0, dup_mask],
                    ub_tmp_score[(copy_num // Constant.RPN_PROPOSAL_NUM) * Constant.RPN_PROPOSAL_NUM],
                    0.0, 1, 1, 8)

            # trans to proposals.
            if center_box == 0:
                self.tik_func_trans_to_proposals(output_ub, boxes_list, ub_tmp_score, copy_num)
            else:
                self.tik_func_trans_to_proposals_center_box(output_ub, ub_tmp_boxes, ub_tmp_score, copy_num, sorted_num)
            
            self.tik_func_trans_index_proposals(output_index_ub, index_list, ub_tmp_score, copy_num)

            # sort within ub.
            sort_within_ub(self.tik_instance, output_ub,
                           ceil_div(copy_num, Constant.RPN_PROPOSAL_NUM) * Constant.RPN_PROPOSAL_NUM)
            sort_within_ub(self.tik_instance, output_index_ub, 
                           ceil_div(copy_num, Constant.RPN_PROPOSAL_NUM) * Constant.RPN_PROPOSAL_NUM)
            if ceil_div(copy_num, Constant.RPN_PROPOSAL_NUM) * Constant.RPN_PROPOSAL_NUM != sorted_num:
                dup_len = sorted_num - ceil_div(copy_num, Constant.RPN_PROPOSAL_NUM) * Constant.RPN_PROPOSAL_NUM
                offset = ceil_div(copy_num, Constant.RPN_PROPOSAL_NUM) * \
                         Constant.RPN_PROPOSAL_NUM * Constant.PROPOSAL_LEN
                self.tik_func_vector(output_ub[offset:], 0.0, dup_len * Constant.PROPOSAL_LEN, self.mask)
                self.tik_func_vector(output_index_ub[offset:], 0.0, dup_len * Constant.PROPOSAL_LEN, self.mask)

    def tik_func_trans_index_proposals(self, proposals_ub, index_ub_list, score_ub, proposal_num):
        """
        vconcat index into proposals.
        """
        batch_id_ub, class_id_ub, index_id_ub, index_tail_ub = index_ub_list
        trans_repeat = ceil_div(proposal_num, Constant.RPN_PROPOSAL_NUM)

        # concat batch id, class id, index id, index tail to proposals.
        self.tik_instance.vconcat(proposals_ub, batch_id_ub, trans_repeat, 0)
        self.tik_instance.vconcat(proposals_ub, class_id_ub, trans_repeat, 1)
        self.tik_instance.vconcat(proposals_ub, index_id_ub, trans_repeat, 2)
        self.tik_instance.vconcat(proposals_ub, index_tail_ub, trans_repeat, 3)

        # concat scores to proposals.
        self.tik_instance.vconcat(proposals_ub, score_ub, trans_repeat, 4)
    
    def tik_func_trans_to_proposals(self, proposals_ub, boxes_ub_list, score_ub, proposal_num):
        """
        vconcat boxes into proposals when center_box equals 0.
        """
        # trans boxes data [x1, y1, x2, y2] to proposals.
        x1_ub, y1_ub, x2_ub, y2_ub = boxes_ub_list
        trans_repeat = ceil_div(proposal_num, Constant.RPN_PROPOSAL_NUM)

        # concat x1 to proposals_ub
        self.tik_instance.vconcat(proposals_ub, x1_ub, trans_repeat, 1)
        # concat y1 to proposals_ub
        self.tik_instance.vconcat(proposals_ub, y1_ub, trans_repeat, 0)
        # concat x2 to proposals_ub
        self.tik_instance.vconcat(proposals_ub, x2_ub, trans_repeat, 3)
        # concat y2 to proposals_ub
        self.tik_instance.vconcat(proposals_ub, y2_ub, trans_repeat, 2)
        # concat scores to proposals_ub
        self.tik_instance.vconcat(proposals_ub, score_ub, trans_repeat, 4)

    def tik_func_trans_to_proposals_center_box(self, proposals_ub, boxes_ub, score_ub, proposal_num, sorted_num):
        """
        vconcat boxes into proposals when center_box equals 1.
        """
        with self.tik_instance.new_stmt_scope():
            # trans boxes data [x1, y1, w, h] to proposals.
            lens = ceil_div(proposal_num, Constant.RPN_PROPOSAL_NUM) * Constant.RPN_PROPOSAL_NUM
            trans_repeat = ceil_div(proposal_num, Constant.RPN_PROPOSAL_NUM)
            move_times = lens // self.data_each_block

            x1_ub = self.tik_instance.Tensor(self.dtype, [lens], name = "x1_ub", scope=tik.scope_ubuf)
            y1_ub = self.tik_instance.Tensor(self.dtype, [lens], name = "y1_ub", scope=tik.scope_ubuf)
            x2_ub = self.tik_instance.Tensor(self.dtype, [lens], name = "x2_ub", scope=tik.scope_ubuf)
            y2_ub = self.tik_instance.Tensor(self.dtype, [lens], name = "y2_ub", scope=tik.scope_ubuf)

            self.tik_instance.data_move(x1_ub, boxes_ub[0, 0], 0, 1, move_times, 0, 0)
            self.tik_instance.data_move(y1_ub, boxes_ub[sorted_num], 0, 1, move_times, 0, 0)
            self.tik_instance.data_move(x2_ub, boxes_ub[sorted_num * 2], 0, 1, move_times, 0, 0)
            self.tik_instance.data_move(y2_ub, boxes_ub[sorted_num * 3], 0, 1, move_times, 0, 0)

            half_x2_ub = self.tik_instance.Tensor(self.dtype, [lens], name="half_x2_ub", scope=tik.scope_ubuf)
            half_y2_ub = self.tik_instance.Tensor(self.dtype, [lens], name="half_y2_ub", scope=tik.scope_ubuf)
            half_ub = self.tik_instance.Tensor(self.dtype, [lens], name="half_ub", scope=tik.scope_ubuf)

            self.tik_func_vector(half_ub, 0.5, lens, self.mask)
            self.tik_func_tensors_calculate("vmul", half_x2_ub, x2_ub, half_ub, lens, self.mask)
            self.tik_func_tensors_calculate("vmul", half_y2_ub, y2_ub, half_ub, lens, self.mask)
            self.tik_func_tensors_calculate("vsub", x2_ub, x1_ub, half_x2_ub, lens, self.mask)
            self.tik_func_tensors_calculate("vsub", y2_ub, y1_ub, half_y2_ub, lens, self.mask)

            # concat x1, y1 to proposals.
            self.tik_instance.vconcat(proposals_ub, x2_ub, trans_repeat, 0)
            self.tik_instance.vconcat(proposals_ub, y2_ub, trans_repeat, 1)

            self.tik_func_tensors_calculate("vadd", x2_ub, x1_ub, half_x2_ub, lens, self.mask)
            self.tik_func_tensors_calculate("vadd", y2_ub, y1_ub, half_y2_ub, lens, self.mask)

            # concat x2, y2 to proposals.
            self.tik_instance.vconcat(proposals_ub, x2_ub, trans_repeat, 2)
            self.tik_instance.vconcat(proposals_ub, y2_ub, trans_repeat, 3)

            # concat scores to proposals.
            self.tik_instance.vconcat(proposals_ub, score_ub, trans_repeat, 4)

    def tik_func_tensors_calculate(self, v_func, out_dst, src0, src1, lens, mask):
        """
        tik add sub and mul
        """
        repeat_time = lens // mask
        repeat_tail = lens % mask
        tik_fun = None

        if v_func == "vadd":
            tik_fun = self.tik_instance.vadd
        if v_func == "vsub":
            tik_fun = self.tik_instance.vsub
        if v_func == "vmul":
            tik_fun = self.tik_instance.vmul
        
        if repeat_time > 0:
            tik_fun(mask, out_dst, src0, src1, repeat_time, 1, 1, 1, 8, 8, 8)
        if repeat_tail > 0:
            offset = repeat_time * mask
            tik_fun(repeat_tail, out_dst[offset], src0[offset], src1[offset], 1, 1, 1, 1, 8, 8, 8)
        
    def tik_func_vector(self, _ub, value, dup_len, mask):
        """
        duplicate value to tensor.
        """
        repeat = dup_len // mask
        repeat_tail = dup_len % mask

        offset = 0
        while repeat > Constant.MAX_REPEAT:
            self.tik_instance.vector_dup(mask, _ub[offset], value, Constant.MAX_REPEAT, 1, 8)
            repeat = repeat - Constant.MAX_REPEAT
            offset = offset + mask * Constant.MAX_REPEAT
        if repeat > 0:
            self.tik_instance.vector_dup(mask, _ub[offset], value, repeat, 1, 8)
            offset = offset + mask * repeat
        if repeat_tail > 0:
            self.tik_instance.vector_dup(repeat_tail, _ub[offset], value, 1, 1, 8)
    
    def change_coordinate_frame_compute(self, ub_tmp_boxes, do_num, sorted_num):
        """change_coordinate_frame_compute"""
        scaling = Constant.COORDINATE_SCALING
        self.tik_func_tensor_scal_cacul("vmuls", ub_tmp_boxes, ub_tmp_boxes, scaling, do_num, 0)
        self.tik_func_tensor_scal_cacul("vmuls", ub_tmp_boxes, ub_tmp_boxes, scaling, do_num, sorted_num)
        self.tik_func_tensor_scal_cacul("vmuls", ub_tmp_boxes, ub_tmp_boxes, scaling, do_num, sorted_num * 2)
        self.tik_func_tensor_scal_cacul("vmuls", ub_tmp_boxes, ub_tmp_boxes, scaling, do_num, sorted_num * 3)

    def tik_func_tensor_scal_cacul(self, v_func, dst_ub, src_ub, value, do_len, ub_offset):
        """dst_ub equals src_ub adds scalar"""
        repeat = do_len // self.mask
        repeat_tail = do_len % self.mask
        offset = ub_offset

        if v_func == "vadds":
            tik_fun = self.tik_instance.vadds
        if v_func == "vmuls":
            tik_fun = self.tik_instance.vmuls

        if repeat > 0:
            tik_fun(self.mask, dst_ub[offset], src_ub[offset], value, repeat, 1, 1, 8, 8)
            offset = offset + self.mask * repeat
        if repeat_tail > 0:
            tik_fun(repeat_tail, dst_ub[offset], src_ub[offset], value, 1, 1, 1, 8, 8)
    
    def clip_window_compute(self, input_ub_list, copy_num, gm_offset):
        """
        transpose inputs.
        """
        with self.tik_instance.new_stmt_scope():
            # init input ub and gm.
            input_boxes_ub = input_ub_list[0]
            input_scores_ub = input_ub_list[1]
            input_index_ub = input_ub_list[-1]
            input_boxes_gm = self.input_gm_list[0]
            input_scores_gm = self.input_gm_list[1]
            input_index_gm = self.input_gm_list[-1]

            # move boxes data from gm to ub.
            move_box_shape = [copy_num, Constant.BOX_LOC]
            move_box_total_size = copy_num * Constant.BOX_LOC

            move_index_shape = [copy_num, Constant.BOX_LOC]
            move_index_total_size = copy_num * Constant.BOX_LOC

            tmp_box_ub = self.tik_instance.Tensor(self.dtype, move_box_shape,
                                                name="tmp_box_ub", scope=tik.scope_ubuf)
            tmp_index_ub = self.tik_instance.Tensor("float16", move_index_shape,
                                                    name="tmp_index_ub", scope=tik.scope_ubuf)

            box_offset = gm_offset[0] * self.boxes_num * Constant.BOX_LOC + gm_offset[2] * Constant.BOX_LOC
            self.tik_func_data_move(tmp_box_ub, input_boxes_gm, move_box_total_size, box_offset, self.dtype)

            index_offset = (gm_offset[0] * self.classes + gm_offset[1]) * self.boxes_num * Constant.BOX_LOC + \
                           gm_offset[2] * Constant.BOX_LOC
            self.tik_func_data_move(tmp_index_ub, input_index_gm, move_index_total_size, index_offset, "float16")

            score_offset = (gm_offset[0] * self.classes + gm_offset[1]) * self.boxes_num + gm_offset[2]
            self.tik_func_data_move(input_scores_ub, input_scores_gm, copy_num, score_offset, self.dtype)

            if self.dtype == "float32":
                # conv index to FP32
                tmp_index_conv_ub = self.tik_instance.Tensor("float32", move_index_shape,
                                                             name="tmp_index_conv_ub", scope=tik.scope_ubuf)
                self.vec_conv_fp16_to_fp32(tmp_index_ub, tmp_index_conv_ub, move_index_total_size)
 
                # transpose index data from [n, 4] to [4, n]
                with self.tik_instance.for_range(0, Constant.BOX_LOC) as i:
                    with self.tik_instance.for_range(0, copy_num) as j:
                        input_index_ub[i, j].set_as(tmp_index_conv_ub[j, i])
            else:
                # transpose index data from [n, 4] to [4, n]
                with self.tik_instance.for_range(0, Constant.BOX_LOC) as i:
                    with self.tik_instance.for_range(0, copy_num) as j:
                        input_index_ub[i, j].set_as(tmp_index_ub[j, i])

            # transpose box data from [n, 4] to [4, n]
            with self.tik_instance.for_range(0, Constant.BOX_LOC) as i:
                with self.tik_instance.for_range(0, copy_num) as j:
                    input_boxes_ub[i, j].set_as(tmp_box_ub[j, i])
    
    def vec_conv_fp16_to_fp32(self, src_ub, dst_ub, num):
        """
        conv from fp16 to fp32.
        """
        mask = Constant.DATA_EACH_BLOCK_16
        repeat = num // mask
        repeat_tail = num % mask

        offset = 0
        while repeat > Constant.MAX_REPEAT:
            self.tik_instance.vec_conv(mask, "none", dst_ub[offset], src_ub[offset], Constant.MAX_REPEAT, 2, 1)
            repeat = repeat - Constant.MAX_REPEAT
            offset = offset + mask * Constant.MAX_REPEAT
        if repeat > 0:
            self.tik_instance.vec_conv(mask, "none", dst_ub[offset], src_ub[offset], repeat, 2, 1)
            offset = offset + mask * repeat
        if repeat_tail > 0:
            self.tik_instance.vec_conv(repeat_tail, "none", dst_ub[offset], src_ub[offset], 1, 2, 1)

    def tik_func_data_move(self, x_ub, x_gm, move_num, offset, dtype):
        """move data from gm to ub."""
        if dtype == "float16":
            data_each_block = Constant.DATA_EACH_BLOCK_16
        else:
            data_each_block = Constant.DATA_EACH_BLOCK_32
        burst_len = ceil_div(move_num, data_each_block)
        self.tik_instance.data_move(x_ub, x_gm[offset], 0, 1, burst_len, 0, 0)


# pylint: disable=too-many-arguments
def non_max_suppression_v7_supports_fp32(boxes, scores, max_output_size,
                                         iou_threshold, score_threshold, index_id,
                                         center_point_box, max_boxes_size, kernel_name):
    # init nms.
    nms = NonMaxSuppressionSupportsFP32(boxes, scores, max_output_size,
                                        iou_threshold, score_threshold, index_id,
                                        center_point_box, max_boxes_size)

    res = nms.run_tik(kernel_name)

    return res

    
