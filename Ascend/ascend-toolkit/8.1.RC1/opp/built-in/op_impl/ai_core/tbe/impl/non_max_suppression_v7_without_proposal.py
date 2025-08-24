# Copyright 2023 Huawei Technologies Co., Ltd
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
non_max_suppression_v7
"""
import functools
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform

# 'pylint: disable=too-many-lines,too-many-instance-attributes,too-many-statements
# 'pylint: disable=too-many-arguments,unused-argument,too-many-locals,too-many-branches


class Constant:
    """
    Class for const values
    """
    # Each loop process 4096 units for fp16 and fp32
    FP16_PER_LOOP_UNIT = 4096
    FP32_PER_LOOP_UNIT = 2048
    # Byte of 8bit, 16bit, 32bit and 64bit dtype
    B8_SIZE = 1
    B16_SIZE = 2
    B32_SIZE = 4
    B64_SIZE = 8
    # Minimum and maxmum value of fp16
    FP16_MINS = -65504
    FP16_MAXS = 65535
    # Maximum value of uint32
    UINT32_MAXS = 4294967295
    # Size of one block
    BLOCK_SIZE = 32
    # Elements num per block of 8bit, 16bit, 32bit and 64bit dtype
    B8_NUM_PER_BLOCK = 32
    B16_NUM_PER_BLOCK = 16
    B32_NUM_PER_BLOCK = 8
    B64_NUM_PER_BLOCK = 4
    # Max elements num of 16bit and 32bit in ub
    B16_UB_LIMIT = 40960
    B32_UB_LIMIT = 20480
    # Mum of processed 8bit, 16bit, 32bit and 64bit dtype elements each iteration of vector calculation
    B8_VEC_MASK = 256
    B16_VEC_MASK = 128
    B32_VEC_MASK = 64
    B64_VEC_MASK = 32
    # Element num of fp16 and fp32 in sorting structure
    FP16_UNIT_ELE = 4
    FP32_UNIT_ELE = 2
    # Num of elements passed by vsort32 each repeat
    VSORT_ELE_REPEAT = 32
    # Max repeat times of tik commands
    REPEAT_TIMES_MAX = 255
    # The length of list proccessed by tik command vmrgsort each loop
    LIST_LEN_VMRGSORT = 4
    # The src1_pattern of tik command vreducev2
    VREDUCEV2_PATTERN_ONE = 1
    VREDUCEV2_PATTERN_THREE = 3
    # The bit of 16bit and 32bit
    B16_BIT = 16
    B32_BIT = 32
    # Scaler
    SCALAR_HALF = 0.5
    SCALAR_NEG_ONE = -1
    SCALAR_THOUSAND = 1000
    # coordinate scaling
    COORDINATE_SCALING = 0.001
    # Aligned shape of mask
    FP16_ALIGNED_SHAPE = 264
    FP32_ALIGNED_SHAPE = 66
    # Max nms output num per loop
    FP16_MAX_OUTPUT_NUM = 7500
    FP32_MAX_OUTPUT_NUM = 3750


class NonMaxSupperssionV7WithoutProposal():
    """
    A brand new compute flow for next generation chips
    """

    def __init__(self, boxes, scores, max_output_size, iou_threshold,
                 score_threshold, index_id, center_point_box=0, max_boxes_size=0):
        """
        Init
        """
        self.tik_instance = tik.Tik()
        self.aicore_num = tik.Dprofile().get_aicore_num()

        # Init properties
        self.boxes_shape = list(boxes.get("shape"))
        self.input_dtype = boxes.get("dtype").lower()
        self.scores_shape = list(scores.get("shape"))
        self.index_id_shape = list(index_id.get("shape"))
        self.center_point_box = center_point_box
        self.max_boxes_size = max_boxes_size
        self.input_gm_list = []
        self.output_gm_list = []

        # Get shape of output
        self.selected_indices_shape = [self.max_boxes_size, 3]
        self.batch_size, self.classes_num, self.boxes_num = self.scores_shape
        self.is_fp16 = False
        self.per_loop_unit = Constant.FP32_PER_LOOP_UNIT
        if self.input_dtype == "float16":
            self.is_fp16 = True
            self.per_loop_unit = Constant.FP16_PER_LOOP_UNIT

        # Init optional inputs
        self.score_threshold = self.tik_instance.Scalar(dtype=self.input_dtype)
        self.score_threshold.set_as(0)
        self.iou_threshold = self.tik_instance.Scalar(dtype=self.input_dtype)
        self.iou_threshold.set_as(0)
        if max_output_size is None:
            self.has_max_output_size = False
        else:
            self.has_max_output_size = True
            self.max_output_size_shape = list(max_output_size.get("shape"))
        if iou_threshold is None:
            self.has_iou_threshold = False
        else:
            self.has_iou_threshold = True
            self.iou_threshold_shape = list(iou_threshold.get("shape"))
        if score_threshold is None:
            self.has_score_threshold = False
        else:
            self.has_score_threshold = True
            self.score_threshold_shape = list(score_threshold.get("shape"))

        # Init value of max_output_size_per_class
        self.max_output_size_per_class = ceil_div(max_boxes_size, self.batch_size * self.classes_num)

        # Get parallel schedule and parameters
        self.core_used = None
        self.num_per_core = None
        self.num_last_core = None
        self.get_core_schedule()

        # Init input gm
        self.idx_gm = None
        self.init_tik_input_gm()

        # Init output memories
        self.init_tik_output_gm()

        # Init value of iou_threshold, score_threshold
        if self.has_iou_threshold:
            self.iou_threshold = self.get_iou_threshold()
        if self.has_score_threshold:
            self.score_threshold = self.get_score_threshold()

        # Define workspace
        self.result_idx_ws = None
        self.selected_indices_ws = None
        self.score_idx_ws = None
        self.selected_x1_ws = None
        self.selected_y1_ws = None
        self.selected_x2_ws = None
        self.selected_y2_ws = None
        self.selected_area_ws = None
        self.result_idx_aligned_shape = None
        self.score_idx_aligned_shape = None
        self.selected_boxes_aligned_shape = None
        self.init_tik_workspace_mem()

        # Patameters of calculation
        self.max_output_num = Constant.FP32_MAX_OUTPUT_NUM
        if self.is_fp16:
            self.max_output_num = Constant.FP16_MAX_OUTPUT_NUM

    def get_core_schedule(self):
        """
        Get parallel scheme
        """
        self.num_per_core = ceil_div(self.batch_size * self.classes_num, self.aicore_num)
        self.core_used = ceil_div(self.batch_size * self.classes_num, self.num_per_core)
        self.num_last_core = self.batch_size * self.classes_num - (self.core_used - 1) * self.num_per_core

    def init_tik_input_gm(self):
        """
        Init input gm
        """
        boxes_gm = self.tik_instance.Tensor(self.input_dtype, self.boxes_shape,
                                            name="boxes_gm", scope=tik.scope_gm)
        scores_gm = self.tik_instance.Tensor(self.input_dtype, self.scores_shape,
                                             name="scores_gm", scope=tik.scope_gm)
        index_id_gm = self.tik_instance.Tensor("float16", self.index_id_shape,
                                               name="index_id_gm", scope=tik.scope_gm)
        max_output_size_gm = None
        iou_threshold_gm = None
        score_threshold_gm = None
        if self.has_max_output_size:
            max_output_size_gm = self.tik_instance.Tensor("int32", self.max_output_size_shape,
                                                          name="max_output_size_gm", scope=tik.scope_gm)
        if self.has_iou_threshold:
            iou_threshold_gm = self.tik_instance.Tensor("float32", self.iou_threshold_shape,
                                                        name="iou_threshold_gm", scope=tik.scope_gm)
        if self.has_score_threshold:
            score_threshold_gm = self.tik_instance.Tensor("float32", self.score_threshold_shape,
                                                          name="score_threshold_gm", scope=tik.scope_gm)
            self.input_gm_list = [boxes_gm, scores_gm, max_output_size_gm,
                                  iou_threshold_gm, score_threshold_gm, index_id_gm]
        else:
            self.input_gm_list = [boxes_gm, scores_gm, max_output_size_gm,
                                  iou_threshold_gm, index_id_gm]

        # Init idx gm tensor for following sorting
        idx_size = ceil_div(self.boxes_num, self.per_loop_unit) * self.per_loop_unit
        idx_init = [i for i in range(idx_size)]
        self.idx_gm = self.tik_instance.Tensor("uint32", [idx_size, ], name="idx_gm",
                                               scope=tik.scope_gm, init_value=idx_init)

    def init_tik_output_gm(self):
        """
        Init the output gm with -1
        """
        selected_indices_gm = self.tik_instance.Tensor("int32", self.selected_indices_shape,
                                                       name="selected_indices_gm", scope=tik.scope_gm)
        dup_len = total_num(self.selected_indices_shape)
        vec_dup_gm(self.tik_instance, selected_indices_gm, "int32", dup_len, 0,
                   Constant.B32_NUM_PER_BLOCK, Constant.B32_UB_LIMIT, Constant.SCALAR_NEG_ONE)
        self.output_gm_list.append(selected_indices_gm)

    def init_tik_workspace_mem(self):
        """
        Init workspace gm tensor
        """
        self.result_idx_aligned_shape = ceil_div(self.max_output_size_per_class, Constant.B32_NUM_PER_BLOCK) * \
            Constant.B32_NUM_PER_BLOCK + Constant.B32_NUM_PER_BLOCK
        result_idx_ws_shape = [self.core_used, self.num_per_core, self.result_idx_aligned_shape]
        self.result_idx_ws = self.tik_instance.Tensor("uint32", result_idx_ws_shape,
                                                      name="result_idx_ws", scope=tik.scope_gm, is_workspace=True)
        vec_dup_gm(self.tik_instance, self.result_idx_ws, "uint32", total_num(result_idx_ws_shape), 0,
                   Constant.B32_NUM_PER_BLOCK, Constant.B32_UB_LIMIT, Constant.SCALAR_NEG_ONE)

        self.selected_indices_aligned_shape = self.num_per_core * self.max_output_size_per_class * 3 +\
            Constant.B32_NUM_PER_BLOCK
        selected_indices_ws_shape = [self.core_used, self.selected_indices_aligned_shape]
        self.selected_indices_ws = self.tik_instance.Tensor("int32", selected_indices_ws_shape,
                                                            name="selected_indices_ws", scope=tik.scope_gm,
                                                            is_workspace=True)
        vec_dup_gm(self.tik_instance, self.selected_indices_ws, "uint32", total_num(selected_indices_ws_shape), 0,
                   Constant.B32_NUM_PER_BLOCK, Constant.B32_UB_LIMIT, Constant.SCALAR_NEG_ONE)

        self.score_idx_aligned_shape = (ceil_div(self.boxes_num, self.per_loop_unit) * self.per_loop_unit +
                                        Constant.B32_NUM_PER_BLOCK) * 2
        score_idx_ws_shape = [self.core_used * self.score_idx_aligned_shape]
        self.selected_boxes_aligned_shape = ceil_div(self.max_output_size_per_class, Constant.B32_NUM_PER_BLOCK) * \
            Constant.B32_NUM_PER_BLOCK + Constant.B32_NUM_PER_BLOCK
        output_limit = Constant.B32_UB_LIMIT
        num_block = Constant.B32_NUM_PER_BLOCK
        if self.is_fp16:
            self.score_idx_aligned_shape = (ceil_div(self.boxes_num, self.per_loop_unit) * self.per_loop_unit +
                                            Constant.B16_NUM_PER_BLOCK) * 4
            score_idx_ws_shape = [self.core_used * self.score_idx_aligned_shape]
            self.selected_boxes_aligned_shape = ceil_div(self.max_output_size_per_class, Constant.B16_NUM_PER_BLOCK) * \
                Constant.B16_NUM_PER_BLOCK + Constant.B16_NUM_PER_BLOCK
            output_limit = Constant.B16_UB_LIMIT
            num_block = Constant.B16_NUM_PER_BLOCK

        selected_boxes_ws_shape = [self.core_used, self.selected_boxes_aligned_shape]

        self.score_idx_ws = self.tik_instance.Tensor(self.input_dtype, score_idx_ws_shape,
                                                     name="score_idx_ws", scope=tik.scope_gm, is_workspace=True)
        self.selected_x1_ws = self.tik_instance.Tensor(self.input_dtype, selected_boxes_ws_shape,
                                                       name="selected_x1_ws", scope=tik.scope_gm, is_workspace=True)
        self.selected_y1_ws = self.tik_instance.Tensor(self.input_dtype, selected_boxes_ws_shape,
                                                       name="selected_y1_ws", scope=tik.scope_gm, is_workspace=True)
        self.selected_x2_ws = self.tik_instance.Tensor(self.input_dtype, selected_boxes_ws_shape,
                                                       name="selected_x2_ws", scope=tik.scope_gm, is_workspace=True)
        self.selected_y2_ws = self.tik_instance.Tensor(self.input_dtype, selected_boxes_ws_shape,
                                                       name="selected_y2_ws", scope=tik.scope_gm, is_workspace=True)
        self.selected_area_ws = self.tik_instance.Tensor(self.input_dtype, selected_boxes_ws_shape,
                                                         name="selected_area_ws", scope=tik.scope_gm, is_workspace=True)

    def clear_workspace(self, core_idx):
        """
        Clear workspace for using again
        """
        score_idx_dup_len = self.score_idx_aligned_shape
        score_idx_offset = core_idx * self.score_idx_aligned_shape
        selected_boxes_dup_len = self.selected_boxes_aligned_shape
        selected_boxes_offset = core_idx * self.selected_boxes_aligned_shape
        output_limit = Constant.B32_UB_LIMIT
        num_block = Constant.B32_NUM_PER_BLOCK
        if self.is_fp16:
            output_limit = Constant.B16_UB_LIMIT
            num_block = Constant.B16_NUM_PER_BLOCK

        vec_dup_gm(self.tik_instance, self.score_idx_ws, self.input_dtype, score_idx_dup_len,
                   score_idx_offset, num_block, output_limit, Constant.FP16_MINS)
        vec_dup_gm(self.tik_instance, self.selected_x1_ws, self.input_dtype, selected_boxes_dup_len,
                   selected_boxes_offset, num_block, output_limit)
        vec_dup_gm(self.tik_instance, self.selected_y1_ws, self.input_dtype, selected_boxes_dup_len,
                   selected_boxes_offset, num_block, output_limit)
        vec_dup_gm(self.tik_instance, self.selected_x2_ws, self.input_dtype, selected_boxes_dup_len,
                   selected_boxes_offset, num_block, output_limit)
        vec_dup_gm(self.tik_instance, self.selected_y2_ws, self.input_dtype, selected_boxes_dup_len,
                   selected_boxes_offset, num_block, output_limit)
        vec_dup_gm(self.tik_instance, self.selected_area_ws, self.input_dtype, selected_boxes_dup_len,
                   selected_boxes_offset, num_block, output_limit)

    def get_iou_threshold(self):
        """
        Calculate iou_threshold for following nms and convert the dtype of threshold according the input dtype
        iou_threshold = iou_threshold_gm[0] / (iou_threshold_gm[0] + 1)
        """
        iou_threshold_gm = self.input_gm_list[3]
        with self.tik_instance.new_stmt_scope():
            if self.is_fp16:
                iou_threshold_fp32_ub = self.tik_instance.Tensor("float32", (Constant.B16_NUM_PER_BLOCK,),
                                                                 name="iou_threshold_fp32_ub", scope=tik.scope_ubuf)
                iou_threshold_fp32_ub[0].set_as(iou_threshold_gm[0])
                iou_threshold_ub = self.tik_instance.Tensor(self.input_dtype, (Constant.B16_NUM_PER_BLOCK,),
                                                            name="iou_threshold_ub", scope=tik.scope_ubuf)
                self.tik_instance.vec_conv(Constant.B32_NUM_PER_BLOCK, "none", iou_threshold_ub,
                                           iou_threshold_fp32_ub, 1, 1, 2)
                times_of_vec_rec = 4
            else:
                iou_threshold_ub = self.tik_instance.Tensor(self.input_dtype, (Constant.B16_NUM_PER_BLOCK,),
                                                            name="iou_threshold_ub", scope=tik.scope_ubuf)
                iou_threshold_ub[0].set_as(iou_threshold_gm[0])
                times_of_vec_rec = 2

            # Calclate the patameters for Tik.vec_rec_high_preci
            repeat_times = 1
            mask_len, rep_stride, block_len = get_mask_rep_stride(iou_threshold_ub)
            src_extent_size = (repeat_times - 1) * rep_stride * block_len + mask_len
            wk_size_unit = ((src_extent_size + block_len - 1) // block_len) * block_len
            wk_size = times_of_vec_rec * repeat_times * wk_size_unit

            # Calculate the threshold by iou_threshold_gm[0] / (iou_threshold_gm[0] + 1)
            ub_one = self.tik_instance.Tensor(self.input_dtype, (mask_len,), name="ub_one", scope=tik.scope_ubuf)
            self.tik_instance.vector_dup(mask_len, ub_one, 1.0, repeat_times, 1, 1)
            add_ub = self.tik_instance.Tensor(self.input_dtype, (mask_len,), name="add_ub", scope=tik.scope_ubuf)
            self.tik_instance.vector_dup(mask_len, add_ub, 1.0, repeat_times, 1, 1)
            self.tik_instance.vadd(Constant.B16_NUM_PER_BLOCK, add_ub, iou_threshold_ub, ub_one, 1, 1, 1, 1, 1, 1, 1)
            work_tensor_ub = self.tik_instance.Tensor("float32", (wk_size,),
                                                      name="work_tensor_ub", scope=tik.scope_ubuf)
            self.tik_instance.vec_rec_high_preci(mask_len, ub_one, add_ub, work_tensor_ub[0:],
                                                 repeat_times, rep_stride, rep_stride)
            self.tik_instance.vmul(Constant.B16_NUM_PER_BLOCK, add_ub, iou_threshold_ub, ub_one, 1, 1, 1, 1, 1, 1, 1)
            self.iou_threshold.set_as(add_ub[0])
            return self.iou_threshold

    def get_score_threshold(self):
        """
        Get score threshold and convert the dtype of threshold according the input dtype
        """
        scores_threshold_gm = self.input_gm_list[4]
        with self.tik_instance.new_stmt_scope():
            if self.is_fp16:
                scores_threshold_fp32_ub = self.tik_instance.Tensor("float32", (Constant.B16_NUM_PER_BLOCK,),
                                                                    name="iou_threshold_fp32_ub", scope=tik.scope_ubuf)
                scores_threshold_fp32_ub[0].set_as(scores_threshold_gm[0])
                scores_threshold_fp16_ub = self.tik_instance.Tensor("float16", (Constant.B16_NUM_PER_BLOCK,),
                                                                    name="iou_threshold_ub", scope=tik.scope_ubuf)
                self.tik_instance.vec_conv(Constant.B16_NUM_PER_BLOCK, "none",
                                           scores_threshold_fp16_ub, scores_threshold_fp32_ub, 1, 1, 2)
                self.score_threshold.set_as(scores_threshold_fp16_ub[0])
            else:
                scores_threshold_fp32_ub = self.tik_instance.Tensor("float32", (Constant.B32_NUM_PER_BLOCK,),
                                                                    name="iou_threshold_ub", scope=tik.scope_ubuf)
                scores_threshold_fp32_ub[0].set_as(scores_threshold_gm[0])
                self.score_threshold.set_as(scores_threshold_fp32_ub[0])
            return self.score_threshold

    def build_tik_instance(self, kernel_name):
        """
        Build cce
        """
        self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=self.input_gm_list,
                                   outputs=self.output_gm_list,
                                   output_files_path=None,
                                   enable_l2=False)

        return self.tik_instance

    def do_compute(self):
        """
        Main function that controls parallel computing
        """
        with self.tik_instance.for_range(0, self.core_used, block_num=self.core_used) as core_idx:
            if self.num_per_core == self.num_last_core or self.core_used == 1:
                self.run_one_core(self.num_per_core, core_idx)
            else:
                with self.tik_instance.if_scope(core_idx < self.core_used - 1):
                    self.run_one_core(self.num_per_core, core_idx)
                with self.tik_instance.else_scope():
                    self.run_one_core(self.num_last_core, core_idx)
        self.move_result_to_gm(self.output_gm_list[0])

    def run_one_core(self, num_on_curr_core, core_idx):
        """
        Computation on each core
        """
        with self.tik_instance.for_range(0, num_on_curr_core) as idx_on_curr_core:
            curr_idx = core_idx * self.num_per_core + idx_on_curr_core
            batch_idx = curr_idx // self.classes_num
            class_idx = curr_idx % self.classes_num
            self.nms_for_single_class(batch_idx, class_idx, idx_on_curr_core, core_idx)
        with self.tik_instance.new_stmt_scope():
            self.batch_multi_class_nms_output(num_on_curr_core, core_idx)

    def nms_for_single_class(self, batch_idx, class_idx, idx_on_curr_core, core_idx):
        """
        Computation for single class
        """
        self.clear_workspace(core_idx)
        selected_box_cnt = self.tik_instance.Scalar(dtype="uint32", init_value=0)
        unit_per_block = Constant.B32_NUM_PER_BLOCK
        ele_unit = Constant.FP32_UNIT_ELE
        if self.is_fp16:
            unit_per_block = Constant.B16_NUM_PER_BLOCK
            ele_unit = Constant.FP16_UNIT_ELE
        eff_size = self.tik_instance.Scalar(dtype="uint32", name="eff_size", init_value=0)
        eff_lens = self.tik_instance.Scalar(dtype="uint32", name="eff_lens", init_value=0)

        # Do nms for the first top per_loop_unit elements
        with self.tik_instance.new_stmt_scope():
            x1_ub = self.tik_instance.Tensor(self.input_dtype, [self.per_loop_unit, ],
                                             name="x1_ub", scope=tik.scope_ubuf)
            x2_ub = self.tik_instance.Tensor(self.input_dtype, [self.per_loop_unit, ],
                                             name="x2_ub", scope=tik.scope_ubuf)
            y1_ub = self.tik_instance.Tensor(self.input_dtype, [self.per_loop_unit, ],
                                             name="y1_ub", scope=tik.scope_ubuf)
            y2_ub = self.tik_instance.Tensor(self.input_dtype, [self.per_loop_unit, ],
                                             name="y2_ub", scope=tik.scope_ubuf)
            scores_idx_ub = self.tik_instance.Tensor(self.input_dtype, [self.per_loop_unit * ele_unit * 2, ],
                                                     name="scores_idx_ub", scope=tik.scope_ubuf)

            # Sort boxes by scores and suppression the boxes whoes score less than score threshold
            self.gen_score_index(core_idx, self.per_loop_unit, batch_idx, class_idx,
                                 self.input_gm_list[1], scores_idx_ub)
            self.get_eff_size(scores_idx_ub, eff_size, self.per_loop_unit)
            scores_ub = self.tik_instance.Tensor(self.input_dtype, [self.per_loop_unit, ],
                                                 name="scores_ub", scope=tik.scope_ubuf)
            index_ub = self.tik_instance.Tensor("uint32", [self.per_loop_unit, ], name="index_ub", scope=tik.scope_ubuf)
            self.get_boxes_after_score_thresh(self.per_loop_unit, x1_ub, x2_ub, y1_ub, y2_ub, scores_ub,
                                              index_ub, batch_idx, self.input_gm_list[0], scores_idx_ub, eff_size)
            self.iou_selection(core_idx, self.per_loop_unit, x1_ub, x2_ub, y1_ub, y2_ub, scores_ub,
                               index_ub, eff_lens, eff_size, self.per_loop_unit, selected_box_cnt)
            with self.tik_instance.if_scope(eff_size > self.max_output_size_per_class):
                eff_size.set_as(self.max_output_size_per_class)
            self.store_coordinate(core_idx, x1_ub, x2_ub, y1_ub, y2_ub, eff_size)
            self.store_result(core_idx, idx_on_curr_core, index_ub, eff_size)
            selected_box_cnt.set_as(eff_size)

        # Do nms for following elements
        with self.tik_instance.if_scope(tik.all(selected_box_cnt < self.max_output_size_per_class,
                                                self.per_loop_unit < self.boxes_num)):
            with self.tik_instance.new_stmt_scope():
                shape_aligned = self.cal_align(self.per_loop_unit+1)
                align_shape = self.per_loop_unit + 1 + shape_aligned
                sort_offset = core_idx * self.score_idx_aligned_shape
                x1_ub = self.tik_instance.Tensor(self.input_dtype, [align_shape, ], name="x1_ub", scope=tik.scope_ubuf)
                x2_ub = self.tik_instance.Tensor(self.input_dtype, [align_shape, ], name="x2_ub", scope=tik.scope_ubuf)
                y1_ub = self.tik_instance.Tensor(self.input_dtype, [align_shape, ], name="y1_ub", scope=tik.scope_ubuf)
                y2_ub = self.tik_instance.Tensor(self.input_dtype, [align_shape, ], name="y2_ub", scope=tik.scope_ubuf)

                cfg = [unit_per_block, ele_unit, core_idx]
                loop_num = (self.boxes_num - self.per_loop_unit) // self.per_loop_unit
                loop_tail = (self.boxes_num - self.per_loop_unit) % self.per_loop_unit
                scores_idx_ub = self.tik_instance.Tensor(self.input_dtype, [self.per_loop_unit * ele_unit * 2, ],
                                                         name="scores_idx_ub", scope=tik.scope_ubuf)
                if loop_num > 0:
                    with self.tik_instance.for_range(0, loop_num) as loop_idx:
                        with self.tik_instance.if_scope(selected_box_cnt < self.max_output_size_per_class):
                            eff_lens.set_as(selected_box_cnt)
                            self.sort_following_data(cfg, loop_idx, scores_idx_ub)
                            self.get_eff_size(scores_idx_ub, eff_size, self.per_loop_unit)

                            # Do nms with each selected boxes, so offset = 1
                            scores_ub = self.tik_instance.Tensor(self.input_dtype, [align_shape, ],
                                                                 name="scores_ub", scope=tik.scope_ubuf)
                            index_ub = self.tik_instance.Tensor("uint32", [align_shape, ],
                                                                name="index_ub", scope=tik.scope_ubuf)

                            self.get_boxes_after_score_thresh(align_shape, x1_ub, x2_ub, y1_ub, y2_ub, scores_ub,
                                                              index_ub, batch_idx, self.input_gm_list[0],
                                                              scores_idx_ub, eff_size, 1)
                            self.do_nms_with_selected_box(core_idx, x1_ub, x2_ub, y1_ub, y2_ub, scores_ub, index_ub,
                                                          eff_size, self.per_loop_unit, 1, shape_aligned,
                                                          selected_box_cnt)
                            with self.tik_instance.if_scope(eff_size > 1):
                                self.iou_selection(core_idx, self.per_loop_unit, x1_ub, x2_ub, y1_ub, y2_ub, scores_ub,
                                                   index_ub, eff_lens, eff_size, self.per_loop_unit,
                                                   selected_box_cnt, 1, shape_aligned)
                                with self.tik_instance.if_scope(eff_size > 1):
                                    self.store_follow_coordinate(core_idx, x1_ub, x2_ub, y1_ub, y2_ub,
                                                                 eff_size, selected_box_cnt)
                                    self.store_follow_result(core_idx, idx_on_curr_core, index_ub,
                                                             eff_size, selected_box_cnt)
                                    selected_box_cnt.set_as(selected_box_cnt + eff_size - 1)

                    if loop_tail > 0:
                        with self.tik_instance.if_scope(selected_box_cnt < self.max_output_size_per_class):
                            eff_lens.set_as(selected_box_cnt)
                            tik_func_vector_dup(self.tik_instance, scores_idx_ub,
                                                self.per_loop_unit * ele_unit * 2, Constant.FP16_MINS)
                            gm2ub(self.tik_instance, scores_idx_ub, self.score_idx_ws,
                                  loop_tail * ele_unit, 0, sort_offset)
                            self.get_eff_size(scores_idx_ub, eff_size, self.per_loop_unit)
                            scores_ub = self.tik_instance.Tensor(self.input_dtype, [align_shape, ],
                                                                 name="scores_ub", scope=tik.scope_ubuf)
                            index_ub = self.tik_instance.Tensor("uint32", [align_shape, ],
                                                                name="index_ub", scope=tik.scope_ubuf)
                            self.get_boxes_after_score_thresh(align_shape, x1_ub, x2_ub, y1_ub, y2_ub, scores_ub,
                                                              index_ub, batch_idx, self.input_gm_list[0], scores_idx_ub,
                                                              eff_size, 1)

                            self.do_nms_with_selected_box(core_idx, x1_ub, x2_ub, y1_ub, y2_ub, scores_ub, index_ub,
                                                          eff_size, self.per_loop_unit, 1, shape_aligned,
                                                          selected_box_cnt)
                            with self.tik_instance.if_scope(eff_size > 1):
                                self.iou_selection(core_idx, self.per_loop_unit, x1_ub, x2_ub, y1_ub, y2_ub, scores_ub,
                                                   index_ub, eff_lens, eff_size, self.per_loop_unit,
                                                   selected_box_cnt, 1, shape_aligned)

                                with self.tik_instance.if_scope(eff_size > 1):
                                    self.store_follow_result(core_idx, idx_on_curr_core, index_ub,
                                                             eff_size, selected_box_cnt)
                                    selected_box_cnt.set_as(selected_box_cnt + eff_size - 1)

                else:
                    if loop_tail > 0:
                        with self.tik_instance.if_scope(selected_box_cnt < self.max_output_size_per_class):
                            eff_lens.set_as(selected_box_cnt)
                            tik_func_vector_dup(self.tik_instance, scores_idx_ub, self.per_loop_unit * ele_unit * 2,
                                                Constant.FP16_MINS)
                            gm2ub(self.tik_instance, scores_idx_ub, self.score_idx_ws, loop_tail * ele_unit,
                                  0, sort_offset)
                            self.get_eff_size(scores_idx_ub, eff_size, self.per_loop_unit)
                            scores_ub = self.tik_instance.Tensor(self.input_dtype, [align_shape, ],
                                                                 name="scores_ub", scope=tik.scope_ubuf)
                            index_ub = self.tik_instance.Tensor("uint32", [align_shape, ],
                                                                name="index_ub", scope=tik.scope_ubuf)
                            self.get_boxes_after_score_thresh(align_shape, x1_ub, x2_ub, y1_ub, y2_ub, scores_ub,
                                                              index_ub, batch_idx, self.input_gm_list[0],
                                                              scores_idx_ub, eff_size, 1)
                            self.do_nms_with_selected_box(core_idx, x1_ub, x2_ub, y1_ub, y2_ub, scores_ub,
                                                          index_ub, eff_size, self.per_loop_unit, 1,
                                                          shape_aligned, selected_box_cnt)
                            with self.tik_instance.if_scope(eff_size > 1):
                                self.iou_selection(core_idx, self.per_loop_unit, x1_ub, x2_ub, y1_ub, y2_ub, scores_ub,
                                                   index_ub, eff_lens, eff_size, loop_tail, selected_box_cnt,
                                                   1, shape_aligned)
                                with self.tik_instance.if_scope(eff_size > 1):
                                    self.store_follow_result(core_idx, idx_on_curr_core, index_ub,
                                                             eff_size, selected_box_cnt)
                                    selected_box_cnt.set_as(selected_box_cnt + eff_size - 1)

    def gen_score_index(self, core_idx, per_loop_unit, batch_idx, class_idx, scores_gm, scores_idx_ub):
        """
        Construct combined tensor by sort commands
        Get topk scores and idx, and move rest to workspace
        """
        score_idx_lens, burst_len, block_ele, loop_num, tail, repeat_times, unit_ele = self.cal_topk_params(
            per_loop_unit)
        out_offset = self.tik_instance.Scalar("uint32", init_value=core_idx * self.score_idx_aligned_shape)

        with self.tik_instance.new_stmt_scope():
            index_ub = self.tik_instance.Tensor("uint32", [per_loop_unit, ], name="idx_ub", scope=tik.scope_ubuf)
            init_index(self.tik_instance, self.idx_gm, index_ub, 0, per_loop_unit)
            scores_ub = self.tik_instance.Tensor(self.input_dtype, [per_loop_unit, ],
                                                 name="scores_ub", scope=tik.scope_ubuf)
            scores_idx_temp = self.tik_instance.Tensor(self.input_dtype, [score_idx_lens * 2, ],
                                                       name="scores_idx_temp", scope=tik.scope_ubuf)

            if loop_num > 0:
                # Sort the first per_loop_unit elements
                burst_lens_base = per_loop_unit // block_ele
                self.tik_instance.data_move(scores_ub, scores_gm[batch_idx, class_idx, 0], 0, 1, burst_lens_base, 0, 0)
                gm2ub_for_sort(self.tik_instance, scores_ub, scores_gm, [batch_idx, class_idx, 0], score_idx_lens)
                self.tik_instance.vsort32(scores_idx_ub, scores_ub, index_ub, repeat_times)

                # Sort and merge following elements and move smaller parts to workspace
                with self.tik_instance.for_range(1, loop_num) as loop_idx:
                    init_index(self.tik_instance, self.idx_gm, index_ub, loop_idx * per_loop_unit, per_loop_unit)
                    gm2ub_for_sort(self.tik_instance, scores_ub, scores_gm,
                                   [batch_idx, class_idx, loop_idx * per_loop_unit], score_idx_lens)
                    self.tik_instance.vsort32(scores_idx_temp, scores_ub, index_ub, repeat_times)
                    self.tik_instance.data_move(scores_idx_temp[score_idx_lens], scores_idx_ub, 0, 1, burst_len, 0, 0)
                    cur_sort_score_idx(self.tik_instance, scores_idx_temp, scores_idx_ub, score_idx_lens * 2, unit_ele)
                    self.tik_instance.data_move(self.score_idx_ws[out_offset],
                                                scores_idx_ub[score_idx_lens], 0, 1, burst_len, 0, 0)
                    out_offset.set_as(out_offset + score_idx_lens)

                with self.tik_instance.if_scope(tail > 0):
                    init_index(self.tik_instance, self.idx_gm, index_ub, loop_num * per_loop_unit, per_loop_unit)
                    # Init the scores_ub to avoid pre tail data interfering with the sorting
                    tik_func_vector_dup(self.tik_instance, scores_ub, per_loop_unit, Constant.FP16_MINS)
                    gm2ub_for_sort(self.tik_instance, scores_ub, scores_gm,
                                   [batch_idx, class_idx, self.boxes_num - tail], tail)
                    self.tik_instance.vsort32(scores_idx_temp, scores_ub, index_ub, repeat_times)
                    self.tik_instance.data_move(scores_idx_temp[score_idx_lens], scores_idx_ub, 0, 1, burst_len, 0, 0)
                    cur_sort_score_idx(self.tik_instance, scores_idx_temp, scores_idx_ub, score_idx_lens * 2, unit_ele)
                    self.tik_instance.data_move(self.score_idx_ws[out_offset],
                                                scores_idx_ub[score_idx_lens], 0, 1, burst_len, 0, 0)

            else:
                # Sort top per_loop_unit elements (in this branch box_num less than per_loop_unit)
                tik_func_vector_dup(self.tik_instance, scores_ub, per_loop_unit, Constant.FP16_MINS)
                gm2ub_for_sort(self.tik_instance, scores_ub, scores_gm, [batch_idx, class_idx, 0], tail)
                self.tik_instance.vsort32(scores_idx_temp, scores_ub, index_ub, repeat_times)
                cur_sort_score_idx(self.tik_instance, scores_idx_temp, scores_idx_ub, score_idx_lens, unit_ele)

    def sort_following_data(self, cfg, sort_idx, scores_idx_follow_ub):
        """
        Move score_idx in from workspace and get topk data
        """
        unit_per_block, ele_unit, core_idx = cfg
        score_idx_lens = self.per_loop_unit * ele_unit
        burst_len = score_idx_lens // unit_per_block
        loop_num = (self.boxes_num - self.per_loop_unit * sort_idx) // self.per_loop_unit
        loop_tail = (self.boxes_num - self.per_loop_unit * sort_idx) % self.per_loop_unit
        tik_func_vector_dup(self.tik_instance, scores_idx_follow_ub, score_idx_lens * 2, Constant.FP16_MINS)

        in_offset = self.tik_instance.Scalar("uint32", init_value=core_idx * self.score_idx_aligned_shape)
        out_offset = self.tik_instance.Scalar("uint32", init_value=core_idx * self.score_idx_aligned_shape)
        gm2ub_with_data_move_pad(self.tik_instance, scores_idx_follow_ub, 
                                 self.score_idx_ws, score_idx_lens, 0, in_offset)
        in_offset.set_as(in_offset + score_idx_lens)

        with self.tik_instance.new_stmt_scope():
            scores_idx_temp = self.tik_instance.Tensor(self.input_dtype, [score_idx_lens * 2, ],
                                                       name="scores_idx_temp", scope=tik.scope_ubuf)

            with self.tik_instance.for_range(1, loop_num) as loop_idx:
                gm2ub_with_data_move_pad(self.tik_instance, scores_idx_temp, 
                                         self.score_idx_ws, score_idx_lens, 0, in_offset)
                self.tik_instance.data_move(scores_idx_temp[score_idx_lens],
                                            scores_idx_follow_ub, 0, 1, burst_len, 0, 0)
                cur_sort_score_idx(self.tik_instance, scores_idx_temp, scores_idx_follow_ub,
                                   score_idx_lens * 2, ele_unit)
                self.tik_instance.data_move(self.score_idx_ws[out_offset], scores_idx_follow_ub[score_idx_lens],
                                            0, 1, burst_len, 0, 0)
                in_offset.set_as(in_offset + score_idx_lens)
                out_offset.set_as(out_offset + score_idx_lens)

            with self.tik_instance.if_scope(loop_tail > 0):
                tik_func_vector_dup(self.tik_instance, scores_idx_temp, score_idx_lens * 2)
                gm2ub_with_data_move_pad(self.tik_instance, scores_idx_temp, 
                                         self.score_idx_ws, loop_tail * ele_unit, 0, in_offset)
                self.tik_instance.data_move(scores_idx_temp[score_idx_lens],
                                            scores_idx_follow_ub, 0, 1, burst_len, 0, 0)
                tik_func_vector_dup(self.tik_instance, scores_idx_follow_ub, score_idx_lens * 2, Constant.FP16_MINS)
                cur_sort_score_idx(self.tik_instance, scores_idx_temp,
                                   scores_idx_follow_ub, score_idx_lens * 2, ele_unit)
                self.tik_instance.data_move(self.score_idx_ws[out_offset],
                                            scores_idx_follow_ub[score_idx_lens], 0, 1, burst_len, 0, 0)

    def cal_topk_params(self, per_loop_unit):
        """
        Calculate the parameters of topk computation
        """
        unit_ele = Constant.FP32_UNIT_ELE
        block_ele = Constant.B32_NUM_PER_BLOCK
        if self.is_fp16:
            unit_ele = Constant.FP16_UNIT_ELE
            block_ele = Constant.B16_NUM_PER_BLOCK

        score_idx_lens = per_loop_unit * unit_ele
        burst_len = score_idx_lens // block_ele
        loop_num = self.boxes_num // per_loop_unit
        tail = self.boxes_num - loop_num * per_loop_unit
        repeat_times = per_loop_unit // Constant.VSORT_ELE_REPEAT
        res = [score_idx_lens, burst_len, block_ele, loop_num, tail, repeat_times, unit_ele]
        return res

    def cal_align(self, per_loop_unit):
        """
        Calculate aligned shape for some commands that need shape alighed
        """
        mask_per_block = Constant.B16_NUM_PER_BLOCK
        if self.is_fp16:
            mask_per_block = Constant.B32_NUM_PER_BLOCK
        mask_shape = per_loop_unit // mask_per_block
        mask_shape_tail = per_loop_unit % mask_per_block
        per_loop_unit_align = per_loop_unit
        if mask_shape_tail > 0:
            if self.is_fp16:
                mask_shape = Constant.FP16_ALIGNED_SHAPE
                per_loop_unit_align = mask_shape * Constant.B16_BIT
            else:
                mask_shape = Constant.FP32_ALIGNED_SHAPE
                per_loop_unit_align = mask_shape * Constant.B32_BIT
        shape_aligned = per_loop_unit_align - per_loop_unit
        return shape_aligned

    def get_eff_size(self, scores_idx_ub, eff_size, shape_size):
        """
        Calculate the effective size of current score by compare with score_threshold
        """
        shape = (shape_size,)
        mask, rep_stride, _ = get_mask_rep_stride(scores_idx_ub)
        mask_shape = (shape_size // Constant.B32_BIT,)
        mask_dtype = "uint32"
        if self.is_fp16:
            mask_dtype = "uint16"
            mask_shape = (shape_size // Constant.B16_BIT,)

        with self.tik_instance.new_stmt_scope():
            scores_tmp = self.tik_instance.Tensor(self.input_dtype, shape, name="scores_tmp", scope=tik.scope_ubuf)
            scores_thresh = self.tik_instance.Tensor(self.input_dtype, shape,
                                                     name="scores_thresh", scope=tik.scope_ubuf)
            tik_func_vector_dup(self.tik_instance, scores_thresh, shape_size, self.score_threshold)
            mask_uint = self.tik_instance.Tensor(mask_dtype, mask_shape, name="mask_uint", scope=tik.scope_ubuf)

            # Move scores data from scores_index to scores_tmp
            score_idx_len_per_unit = Constant.FP32_UNIT_ELE
            pattern = Constant.VREDUCEV2_PATTERN_ONE
            if self.is_fp16:
                score_idx_len_per_unit = Constant.FP16_UNIT_ELE
                pattern = Constant.VREDUCEV2_PATTERN_THREE
            repeat_times = shape_size * score_idx_len_per_unit // mask
            self.tik_instance.vreducev2(None, scores_tmp, scores_idx_ub, pattern, repeat_times, 1, rep_stride, 0)
            self.gen_mask(shape_size, scores_thresh, scores_tmp, mask_uint)
            self.tik_instance.vreducev2(shape_size, scores_thresh, scores_tmp, mask_uint, 1, 1, rep_stride, 1,
                                        rsvd_scalar=eff_size, mask_mode="counter")

    def gen_mask(self, size, overlap, iou, mask):
        """
        Generate mask by tik.vec_cmpv_lt
        """
        vector_mask, rep_stride, _ = get_mask_rep_stride(overlap)
        per_loop_num = Constant.REPEAT_TIMES_MAX * vector_mask
        loops = size // per_loop_num
        offset = 0

        if loops > 0:
            with self.tik_instance.for_range(0, loops) as _:
                self.tik_instance.vec_cmpv_lt(mask[offset], overlap[offset], iou[offset],
                                              Constant.REPEAT_TIMES_MAX, rep_stride, rep_stride)
                offset += per_loop_num

        repeat_times = ceil_div((size % per_loop_num), vector_mask)
        if repeat_times > 0:
            self.tik_instance.vec_cmpv_lt(mask[offset // Constant.B32_BIT], overlap[offset],
                                          iou[offset], repeat_times, rep_stride, rep_stride)

    def get_boxes_after_score_thresh(self, per_loop_unit, x1, x2, y1, y2, scores_ub, index_ub,
                                     batch_idx, boxes_gm, scores_idx_ub, eff_size, offset=0):
        """
        Move boxes coordinates from boxes_gm to x y ub according to 
        eff_size and center_point_box
        """
        tik_func_vector_dup(self.tik_instance, x1, per_loop_unit)
        tik_func_vector_dup(self.tik_instance, x2, per_loop_unit)
        tik_func_vector_dup(self.tik_instance, y1, per_loop_unit)
        tik_func_vector_dup(self.tik_instance, y2, per_loop_unit)
        tik_func_vector_dup(self.tik_instance, scores_ub, per_loop_unit, Constant.FP16_MINS)
        tik_func_vector_dup(self.tik_instance, index_ub, per_loop_unit, Constant.SCALAR_NEG_ONE)

        loc_index = self.tik_instance.Scalar("uint32")
        with self.tik_instance.for_range(0, eff_size) as idx:
            if self.is_fp16:
                scores_index_offset = idx * Constant.FP16_UNIT_ELE
                loc_index.set_as(scores_idx_ub[scores_index_offset +
                                 2: scores_index_offset+4].reinterpret_cast_to("uint32"))
            else:
                scores_index_offset = idx * Constant.FP32_UNIT_ELE
                loc_index.set_as(scores_idx_ub[scores_index_offset + 1].reinterpret_cast_to("uint32"))
            x1[offset + idx].set_as(boxes_gm[batch_idx, loc_index, 1])
            y1[offset + idx].set_as(boxes_gm[batch_idx, loc_index, 0])
            x2[offset + idx].set_as(boxes_gm[batch_idx, loc_index, 3])
            y2[offset + idx].set_as(boxes_gm[batch_idx, loc_index, 2])
            scores_ub[offset + idx].set_as(scores_idx_ub[scores_index_offset])
            index_ub[offset + idx].set_as(loc_index)

        # Change coordinate frame by scaling
        self.change_coordinate_frame_compute(x1, x2, y1, y2, per_loop_unit)

        if self.center_point_box == 1:
            # x1 is y_center, y1 is x_center, x2 is height and y2 is width currently
            with self.tik_instance.new_stmt_scope():
                half_w_ub = self.tik_instance.Tensor(self.input_dtype, [per_loop_unit],
                                                     name="half_w_ub", scope=tik.scope_ubuf)
                half_h_ub = self.tik_instance.Tensor(self.input_dtype, [per_loop_unit],
                                                     name="half_h_ub", scope=tik.scope_ubuf)
                half_ub = self.tik_instance.Tensor(self.input_dtype, [per_loop_unit],
                                                   name="half_ub", scope=tik.scope_ubuf)
                tik_func_vector_dup(self.tik_instance, half_ub, per_loop_unit, Constant.SCALAR_HALF)
                tik_func_add_sub_mul(self.tik_instance, "vmul", half_w_ub, y2, half_ub, per_loop_unit)
                tik_func_add_sub_mul(self.tik_instance, "vmul", half_h_ub, x2, half_ub, per_loop_unit)
                tik_func_add_sub_mul(self.tik_instance, "vadd", x2, y1, half_w_ub, per_loop_unit)
                tik_func_add_sub_mul(self.tik_instance, "vadd", y2, x1, half_h_ub, per_loop_unit)
                ub2ub(self.tik_instance, half_ub, x1, per_loop_unit)
                tik_func_add_sub_mul(self.tik_instance, "vsub", x1, y1, half_w_ub, per_loop_unit)
                tik_func_add_sub_mul(self.tik_instance, "vsub", y1, half_ub, half_h_ub, per_loop_unit)

    def change_coordinate_frame_compute(self, x1, x2, y1, y2, per_loop_unit):
        """
        Scale the coordinates
        """
        with self.tik_instance.new_stmt_scope():
            scale_value = self.tik_instance.Scalar(self.input_dtype, init_value=Constant.COORDINATE_SCALING)
            tik_func_vadds_vmuls(self.tik_instance, "vmuls", x1, x1, scale_value, per_loop_unit)
            tik_func_vadds_vmuls(self.tik_instance, "vmuls", x2, x2, scale_value, per_loop_unit)
            tik_func_vadds_vmuls(self.tik_instance, "vmuls", y1, y1, scale_value, per_loop_unit)
            tik_func_vadds_vmuls(self.tik_instance, "vmuls", y2, y2, scale_value, per_loop_unit)

    def do_nms_with_selected_box(self, core_idx, x1_ub, x2_ub, y1_ub, y2_ub, scores_ub, index_ub,
                                 eff_size, per_loop_unit, offset, shape_aligned, selected_box_cnt):
        """
        Do nms between following boxes and selected boxes
        """
        with self.tik_instance.new_stmt_scope():
            align_shape = per_loop_unit + offset + shape_aligned
            iou = self.tik_instance.Tensor(self.input_dtype, [align_shape, ], name="iou", scope=tik.scope_ubuf)
            mask_shape = ceil_div(align_shape, Constant.B32_BIT)
            mask_dtype = "uint32"
            if self.is_fp16:
                mask_dtype = "uint16"
                mask_shape = ceil_div(align_shape, Constant.B16_BIT)
            mask_uint = self.tik_instance.Tensor(mask_dtype, [mask_shape, ], name="mask_uint", scope=tik.scope_ubuf)
            tik_func_vector_dup(self.tik_instance, iou, align_shape)
            tik_func_vector_dup(self.tik_instance, mask_uint, mask_shape)

            single_area = self.tik_instance.Tensor(self.input_dtype, [align_shape, ],
                                                   name="single_area", scope=tik.scope_ubuf)
            tik_func_vector_dup(self.tik_instance, single_area, align_shape)
            self.get_rectangle_area(align_shape, x1_ub, x2_ub, y1_ub, y2_ub, single_area)
            overlap = self.tik_instance.Tensor(self.input_dtype, [align_shape, ], name="overlap", scope=tik.scope_ubuf)
            tmp = self.tik_instance.Tensor(self.input_dtype, [align_shape, ], name="tmp", scope=tik.scope_ubuf)

            with self.tik_instance.for_range(0, selected_box_cnt) as selected_idx:
                with self.tik_instance.if_scope(eff_size > 1):
                    self.get_selected_boxes(core_idx, x1_ub, x2_ub, y1_ub, y2_ub, single_area, selected_idx)
                    self.iou_selection_single(align_shape, x1_ub, x2_ub, y1_ub, y2_ub, scores_ub, index_ub,
                                              iou, single_area, mask_uint, overlap, tmp, eff_size)

    def get_selected_boxes(self, core_idx, x1_ub, x2_ub, y1_ub, y2_ub, single_area, selected_idx):
        """
        Get selected box coordinate and area with changing coordinate frame compute
        """
        x1_ub[0].set_as(self.selected_x1_ws[core_idx * self.selected_boxes_aligned_shape + selected_idx])
        x2_ub[0].set_as(self.selected_x2_ws[core_idx * self.selected_boxes_aligned_shape + selected_idx])
        y1_ub[0].set_as(self.selected_y1_ws[core_idx * self.selected_boxes_aligned_shape + selected_idx])
        y2_ub[0].set_as(self.selected_y2_ws[core_idx * self.selected_boxes_aligned_shape + selected_idx])
        single_area[0].set_as(self.selected_area_ws[core_idx * self.selected_boxes_aligned_shape + selected_idx])

    def iou_selection(self, core_idx, per_loop_unit, x1, x2, y1, y2, scores_ub, index_ub, eff_lens, eff_size,
                      loop_num, selected_box_cnt, offset=0, shape_aligned=0):
        """
        Calculate the overlap and iou of boxes, screen out boxes whose iou is greater than iou_threshold
        """
        with self.tik_instance.new_stmt_scope():
            # Init iou and mask
            align_shape = per_loop_unit+offset+shape_aligned
            iou = self.tik_instance.Tensor(self.input_dtype, [align_shape, ], name="iou", scope=tik.scope_ubuf)
            mask_shape = (align_shape) // Constant.B32_BIT
            mask_dtype = "uint32"
            if self.is_fp16:
                mask_dtype = "uint16"
                mask_shape = (align_shape) // Constant.B16_BIT
            mask_uint = self.tik_instance.Tensor(mask_dtype, [mask_shape, ], name="mask_uint", scope=tik.scope_ubuf)
            tik_func_vector_dup(self.tik_instance, iou, align_shape)
            tik_func_vector_dup(self.tik_instance, mask_uint, mask_shape)

            # Calculate the area of boxes
            single_area = self.tik_instance.Tensor(self.input_dtype, [align_shape, ],
                                                   name="single_area", scope=tik.scope_ubuf)
            tik_func_vector_dup(self.tik_instance, single_area, align_shape)
            self.get_rectangle_area(align_shape, x1, x2, y1, y2, single_area)

            # Define overlap and tmps for iou_selection
            overlap = self.tik_instance.Tensor(self.input_dtype, [align_shape, ], name="overlap", scope=tik.scope_ubuf)
            tik_func_vector_dup(self.tik_instance, overlap, align_shape)
            tmp = self.tik_instance.Tensor(self.input_dtype, [align_shape, ], name="tmp", scope=tik.scope_ubuf)
            tik_func_vector_dup(self.tik_instance, tmp, align_shape)

            # Calculate the overlap and iou of idx box with the following boxes's for updating boxes and indexes
            with self.tik_instance.for_range(offset, loop_num) as idx:
                with self.tik_instance.if_scope(tik.all(idx < eff_size, eff_lens < self.max_output_size_per_class)):
                    self.get_overlap(align_shape, x1, x2, y1, y2, overlap, tmp, idx)
                    self.cal_iou(iou, single_area, tmp, idx, align_shape)
                    self.gen_mask(align_shape, overlap, iou, mask_uint)
                    self.update_input(align_shape, x1, x2, y1, y2, scores_ub, index_ub,
                                      single_area, eff_size, tmp, mask_uint)
                    eff_lens.set_as(eff_lens + 1)

            if shape_aligned == 0:
                self.store_selected_area(core_idx, single_area, eff_size)
            else:
                self.store_follow_selected_area(core_idx, single_area, eff_size, selected_box_cnt)

    def iou_selection_single(self, per_loop_unit, x1, x2, y1, y2, scores_ub, index_ub, iou,
                             single_area, mask_uint, overlap, tmp, eff_size):
        """
        Select box with selected boxes accroding to iou_threshold
        """
        self.get_overlap(per_loop_unit, x1, x2, y1, y2, overlap, tmp, 0)
        self.cal_iou(iou, single_area, tmp, 0, per_loop_unit)
        self.gen_mask(per_loop_unit, overlap, iou, mask_uint)
        self.update_input(per_loop_unit, x1, x2, y1, y2, scores_ub, index_ub, single_area, eff_size, tmp, mask_uint)

    def get_rectangle_area(self, per_loop_unit, x1, x2, y1, y2, single_area):
        """
        Calculate the areas of boxes by (x2 - x1) * (y2 - y1)
        """
        with self.tik_instance.new_stmt_scope():
            x_diff = self.tik_instance.Tensor(self.input_dtype, [per_loop_unit, ], name="x_diff", scope=tik.scope_ubuf)
            y_diff = self.tik_instance.Tensor(self.input_dtype, [per_loop_unit, ], name="y_diff", scope=tik.scope_ubuf)
            tik_func_add_sub_mul(self.tik_instance, "vsub", x_diff, x2, x1, per_loop_unit)
            tik_func_add_sub_mul(self.tik_instance, "vsub", y_diff, y2, y1, per_loop_unit)
            tik_func_add_sub_mul(self.tik_instance, "vmul", single_area, x_diff, y_diff, per_loop_unit)

    def get_overlap(self, per_loop_unit, xx1, xx2, yy1, yy2, overlap, tmp, idx):
        """
        Get overlap area of x1 and the following boxes, the overlap of itself is 0
        """
        with self.tik_instance.new_stmt_scope():
            x1 = self.tik_instance.Scalar(self.input_dtype, init_value=xx1[idx])
            x2 = self.tik_instance.Scalar(self.input_dtype, init_value=xx2[idx])
            y1 = self.tik_instance.Scalar(self.input_dtype, init_value=yy1[idx])
            y2 = self.tik_instance.Scalar(self.input_dtype, init_value=yy2[idx])
            tmp1 = self.tik_instance.Tensor(self.input_dtype, [per_loop_unit, ], name="tmp1", scope=tik.scope_ubuf)

            # Calculate max(min(x2, xx2) - max(x1, xx1), 0)
            tik_func_vmaxs_vmins(self.tik_instance, "vmaxs", tmp1, xx1, x1, per_loop_unit)
            tik_func_vmaxs_vmins(self.tik_instance, "vmins", overlap, xx2, x2, per_loop_unit)
            tik_func_add_sub_mul(self.tik_instance, "vsub", tmp1, overlap, tmp1, per_loop_unit)
            tik_func_vmaxs_vmins(self.tik_instance, "vmaxs", tmp1, tmp1, 0, per_loop_unit)

            # Calculate max(min(y2, yy2) - max(y1, yy1), 0)
            tik_func_vmaxs_vmins(self.tik_instance, "vmaxs", tmp, yy1, y1, per_loop_unit)
            tik_func_vmaxs_vmins(self.tik_instance, "vmins", overlap, yy2, y2, per_loop_unit)
            tik_func_add_sub_mul(self.tik_instance, "vsub", tmp, overlap, tmp, per_loop_unit)
            tik_func_vmaxs_vmins(self.tik_instance, "vmaxs", tmp, tmp, 0, per_loop_unit)

            # Calculate overlap areas
            tik_func_vector_dup(self.tik_instance, overlap, per_loop_unit)
            tik_func_add_sub_mul(self.tik_instance, "vmul", overlap, tmp1, tmp, per_loop_unit)

            # The overlap of the fixed boxes and itself default as 0
            overlap[idx].set_as(0)

    def cal_iou(self, iou, single_area, tmp, idx, per_loop_unit):
        """"
        Calculate part of iou accrodding to
            result = (single_area + fixed_area) * iou_threshold
            * union area = sum of two boxes' areas sub ovrelap area
        """
        fixed_area = self.tik_instance.Scalar(self.input_dtype, init_value=single_area[idx])
        tik_func_vadds_vmuls(self.tik_instance, "vadds", tmp, single_area, fixed_area, per_loop_unit)
        tik_func_vector_dup(self.tik_instance, iou, per_loop_unit)
        tik_func_vadds_vmuls(self.tik_instance, "vmuls", iou, tmp, self.iou_threshold, per_loop_unit)

    def update_input(self, undate_num, x1, x2, y1, y2, scores_ub, inedx_ub, single_area, eff_size, tmp, mask):
        """
        Update(suppression) inputs by mask
        """
        tik_func_vector_dup(self.tik_instance, tmp, undate_num)

        self.tik_instance.vreducev2(undate_num, tmp, x1, mask, 1, 1, 8, 1, rsvd_scalar=eff_size, mask_mode="counter")
        tik_func_vector_dup(self.tik_instance, x1, undate_num)
        ub2ub(self.tik_instance, x1, tmp, eff_size)

        self.tik_instance.vreducev2(undate_num, tmp, x2, mask, 1, 1, 8, 1, mask_mode="counter")
        tik_func_vector_dup(self.tik_instance, x2, undate_num)
        ub2ub(self.tik_instance, x2, tmp, eff_size)

        self.tik_instance.vreducev2(undate_num, tmp, y1, mask, 1, 1, 8, 1, mask_mode="counter")
        tik_func_vector_dup(self.tik_instance, y1, undate_num)
        ub2ub(self.tik_instance, y1, tmp, eff_size)

        self.tik_instance.vreducev2(undate_num, tmp, y2, mask, 1, 1, 8, 1, mask_mode="counter")
        tik_func_vector_dup(self.tik_instance, y2, undate_num)
        ub2ub(self.tik_instance, y2, tmp, eff_size)

        self.tik_instance.vreducev2(undate_num, tmp, scores_ub, mask, 1, 1, 8, 1, mask_mode="counter")
        tik_func_vector_dup(self.tik_instance, scores_ub, undate_num)
        ub2ub(self.tik_instance, scores_ub, tmp, eff_size)

        self.tik_instance.vreducev2(undate_num, tmp, single_area, mask, 1, 1, 8, 1, mask_mode="counter")
        tik_func_vector_dup(self.tik_instance, single_area, undate_num)
        ub2ub(self.tik_instance, single_area, tmp, eff_size)

        with self.tik_instance.new_stmt_scope():
            tmp_uint = self.tik_instance.Tensor("uint32", [undate_num, ], name="tmp_uint", scope=tik.scope_ubuf)
            if self.is_fp16:
                mask_unit_32 = mask.reinterpret_cast_to("uint32")
                self.tik_instance.vreducev2(undate_num, tmp_uint, inedx_ub,
                                            mask_unit_32, 1, 1, 8, 1, mask_mode="counter")
            else:
                self.tik_instance.vreducev2(undate_num, tmp_uint, inedx_ub, mask, 1, 1, 8, 1, mask_mode="counter")
            tik_func_vector_dup(self.tik_instance, inedx_ub, undate_num, Constant.SCALAR_NEG_ONE)
            ub2ub(self.tik_instance, inedx_ub, tmp_uint, eff_size)

    def store_selected_area(self, core_idx, area, eff_size):
        """
        Move selected area coordinates to workspace for following calculation
        """
        _, _, ele_block = get_mask_rep_stride(area)
        burst_len = ceil_div(eff_size, ele_block)
        ws_offset = core_idx * self.selected_boxes_aligned_shape
        self.tik_instance.data_move(self.selected_area_ws[ws_offset], area, 0, 1, burst_len, 0, 0)

    def store_follow_selected_area(self, core_idx, area, eff_size, offset=0):
        """
        Move selected area coordinates to workspace for following calculation
        """
        _, _, ele_block = get_mask_rep_stride(area)
        with self.tik_instance.if_scope(eff_size > self.max_output_size_per_class+1-offset):
            eff_size.set_as(self.max_output_size_per_class+1-offset)
        burst_len = ceil_div(eff_size, ele_block)
        ws_offset = core_idx * self.selected_boxes_aligned_shape + offset - 1
        area[0].set_as(self.selected_area_ws[ws_offset])
        self.tik_instance.data_move(self.selected_area_ws[ws_offset], area, 0, 1, burst_len, 0, 0)

    def store_coordinate(self, core_idx, x1, x2, y1, y2, eff_size):
        """
        Move selected boxes coordinates to workspace for following calculation
        """
        _, _, ele_block = get_mask_rep_stride(x1)
        burst_len = ceil_div(eff_size, ele_block)
        ws_offset = core_idx * self.selected_boxes_aligned_shape
        self.tik_instance.data_move(self.selected_x1_ws[ws_offset], x1, 0, 1, burst_len, 0, 0)
        self.tik_instance.data_move(self.selected_x2_ws[ws_offset], x2, 0, 1, burst_len, 0, 0)
        self.tik_instance.data_move(self.selected_y1_ws[ws_offset], y1, 0, 1, burst_len, 0, 0)
        self.tik_instance.data_move(self.selected_y2_ws[ws_offset], y2, 0, 1, burst_len, 0, 0)

    def store_follow_coordinate(self, core_idx, x1, x2, y1, y2, eff_size, offset):
        """
        Move following selected boxes coordinates to workspace for following loops
        """
        _, _, ele_block = get_mask_rep_stride(x1)
        with self.tik_instance.if_scope(eff_size > self.max_output_size_per_class+1-offset):
            eff_size.set_as(self.max_output_size_per_class+1-offset)
        burst_len = ceil_div(eff_size, ele_block)
        ws_offset = core_idx * self.selected_boxes_aligned_shape + offset - 1
        x1[0].set_as(self.selected_x1_ws[ws_offset])
        x2[0].set_as(self.selected_x2_ws[ws_offset])
        y1[0].set_as(self.selected_y1_ws[ws_offset])
        y2[0].set_as(self.selected_y2_ws[ws_offset])
        self.tik_instance.data_move(self.selected_x1_ws[ws_offset], x1, 0, 1, burst_len, 0, 0)
        self.tik_instance.data_move(self.selected_x2_ws[ws_offset], x2, 0, 1, burst_len, 0, 0)
        self.tik_instance.data_move(self.selected_y1_ws[ws_offset], y1, 0, 1, burst_len, 0, 0)
        self.tik_instance.data_move(self.selected_y2_ws[ws_offset], y2, 0, 1, burst_len, 0, 0)

    def store_result(self, core_idx, idx, index_ub, eff_size):
        """
        Store the nmsed results to workspace
        """
        burst_len = ceil_div(eff_size, Constant.B32_NUM_PER_BLOCK)
        ws_offset = core_idx * self.num_per_core * self.result_idx_aligned_shape + idx * self.result_idx_aligned_shape
        self.tik_instance.data_move(self.result_idx_ws[ws_offset], index_ub, 0, 1, burst_len, 0, 0)

    def store_follow_result(self, core_idx, idx, index_ub, eff_size, offset):
        """
        Store the following nmsed results to workspace
        """
        with self.tik_instance.if_scope(eff_size > self.max_output_size_per_class+1-offset):
            eff_size.set_as(self.max_output_size_per_class+1-offset)
        burst_len = ceil_div(eff_size, Constant.B32_NUM_PER_BLOCK)
        ws_offset = core_idx * self.num_per_core * self.result_idx_aligned_shape + idx * \
            self.result_idx_aligned_shape + offset - 1
        index_ub[0].set_as(self.result_idx_ws[ws_offset])
        self.tik_instance.data_move(self.result_idx_ws[ws_offset], index_ub, 0, 1, burst_len, 0, 0)

    def batch_multi_class_nms_output(self, num_on_curr_core, core_idx):
        """
        Main function to move results to final gm output
        """
        offset = self.num_per_core * core_idx
        with self.tik_instance.for_range(0, num_on_curr_core) as idx_on_curr_core:
            curr_idx = offset + idx_on_curr_core
            batch_idx = curr_idx // self.classes_num
            class_idx = curr_idx % self.classes_num
            with self.tik_instance.new_stmt_scope():
                self.nms_output_index(core_idx, self.max_output_size_per_class, batch_idx, class_idx, idx_on_curr_core)

    def nms_output_index(self, core_idx, output_num, batch_idx, class_idx, idx_on_curr_core):
        """
        Recover and move index to output gm
        """
        shape_aligned = ceil_div(output_num * 3, 16) * 16
        shape_aligned_tail = ceil_div(output_num, 16) * 16
        selected_indices_ub = self.tik_instance.Tensor("int32", [shape_aligned],
                                                       name="selected_indices_ub", scope=tik.scope_ubuf)
        with self.tik_instance.new_stmt_scope():
            nmsed_index_ub = self.tik_instance.Tensor("uint32", [output_num, ],
                                                      name="nmsed_index_ub", scope=tik.scope_ubuf)
            offset = core_idx * self.num_per_core * self.result_idx_aligned_shape + idx_on_curr_core * \
                self.result_idx_aligned_shape
            gm2ub(self.tik_instance, nmsed_index_ub, self.result_idx_ws, output_num, 0, offset)

            selected_indices_fp16_ub = self.tik_instance.Tensor("float16", [shape_aligned],
                                                                name="selected_indices_fp16_ub", scope=tik.scope_ubuf)
            tik_func_vector_dup(self.tik_instance, selected_indices_fp16_ub, shape_aligned, Constant.SCALAR_NEG_ONE)
            selected_indices_head_ub = self.tik_instance.Tensor("float16", [shape_aligned_tail],
                                                                name="selected_indices_head_ub", scope=tik.scope_ubuf)
            selected_indices_tail_ub = self.tik_instance.Tensor("float16", [shape_aligned_tail],
                                                                name="selected_indices_tail_ub", scope=tik.scope_ubuf)

            index_id_offset = (batch_idx * self.classes_num + class_idx) * self.boxes_num * 4
            self.move_indices_from_gm(output_num, nmsed_index_ub, self.input_gm_list[-1], selected_indices_fp16_ub,
                                      selected_indices_head_ub, selected_indices_tail_ub, index_id_offset)
            self.recover_indices(selected_indices_ub, selected_indices_fp16_ub, selected_indices_head_ub,
                                 selected_indices_tail_ub, output_num, shape_aligned, shape_aligned_tail)
        self.move_result_to_ws(selected_indices_ub, output_num, core_idx, batch_idx, class_idx, idx_on_curr_core)

    def move_indices_from_gm(self, output_num, index_ub, index_gm,
                             selected_indices_fp16_ub, head, tail, index_id_offset):
        """
        Move indeices from gm to ub by nmsed indexes
        """
        with self.tik_instance.new_stmt_scope():
            neg_one = self.tik_instance.Scalar("float16", init_value=-1.0)
            scalar_zero = self.tik_instance.Scalar("float16", init_value=0)
            index = self.tik_instance.Scalar("uint32")

            with self.tik_instance.for_range(0, output_num) as idx:
                index.set_as(index_ub[idx])
                with self.tik_instance.if_scope(index < Constant.UINT32_MAXS):
                    offset = index_id_offset + index * 4
                    selected_indices_fp16_ub[idx*3].set_as(index_gm[offset])
                    selected_indices_fp16_ub[idx*3+1].set_as(index_gm[offset+1])
                    head[idx].set_as(index_gm[offset+2])
                    tail[idx].set_as(index_gm[offset+3])

                # 0 * 1000 + (-1) = -1
                with self.tik_instance.else_scope():
                    selected_indices_fp16_ub[idx*3].set_as(neg_one)
                    selected_indices_fp16_ub[idx*3+1].set_as(neg_one)
                    head[idx].set_as(scalar_zero)
                    tail[idx].set_as(neg_one)

    def recover_indices(self, selected_indices, selected_indices_fp16, head, tail,
                        output_num, shape_aligned, shape_aligned_tail):
        """
        Recover indices from shape [4] to shape [3] and from fp16 to int32
        """
        loop_len = ceil_div(output_num*3, 16)
        loop_len_tail = ceil_div(output_num, 16)

        with self.tik_instance.new_stmt_scope():
            # Deal with the last index with   head = head * 1000 + tail
            head_int32_ub = self.tik_instance.Tensor("int32", [shape_aligned_tail],
                                                     name="head_int32_ub", scope=tik.scope_ubuf)
            with self.tik_instance.new_stmt_scope():
                tail_int32_ub = self.tik_instance.Tensor("int32", [shape_aligned_tail],
                                                         name="tail_int32_ub", scope=tik.scope_ubuf)
                self.tik_instance.vec_conv(16, "round", head_int32_ub, head, loop_len_tail, 2, 1)
                self.tik_instance.vec_conv(16, "round", tail_int32_ub, tail, loop_len_tail, 2, 1)
                thousand_ub = self.tik_instance.Tensor("int32", [shape_aligned_tail],
                                                       name="thousand_ub", scope=tik.scope_ubuf)
                input_temp = self.tik_instance.Tensor("int32", (shape_aligned_tail,),
                                                      name="input_temp", scope=tik.scope_ubuf)
                tik_func_vector_dup(self.tik_instance, thousand_ub, output_num, Constant.SCALAR_THOUSAND)
                tik_func_add_sub_mul(self.tik_instance, "vmul", input_temp, head_int32_ub, thousand_ub, output_num)
                tik_func_add_sub_mul(self.tik_instance, "vadd", head_int32_ub, input_temp, tail_int32_ub, output_num)
            
            # Note: vec_conv instruction limitation repeat_times need to be in [0, 255]
            loop_len_new = loop_len // 255
            loop_len_new_tail = loop_len % 255
            with self.tik_instance.for_range(0, loop_len_new) as i:
                self.tik_instance.vec_conv(16, "round", selected_indices[i * 255 * 16],
                                           selected_indices_fp16[i * 255 * 16], 255, 2, 1)
            
            if loop_len_new_tail != 0:
                self.tik_instance.vec_conv(16, "round", selected_indices[loop_len_new * 255 * 16],
                                           selected_indices_fp16[loop_len_new * 255 * 16], loop_len_new_tail, 2, 1)

            with self.tik_instance.for_range(0, output_num) as i:
                selected_indices[i * 3 + 2] = head_int32_ub[i]

    def move_result_to_ws(self, selected_indices_ub, output_num, core_idx, batch_idx, class_idx, idx_on_curr_core):
        """
        Move results from ub to ws to avoid data stampede
        """
        offset = core_idx * self.selected_indices_aligned_shape + idx_on_curr_core * self.max_output_size_per_class * 3
        burst_len = ceil_div(output_num*3, Constant.B32_NUM_PER_BLOCK)
        self.tik_instance.data_move(self.selected_indices_ws[offset], selected_indices_ub, 0, 1, burst_len, 0, 0)

    def move_result_to_gm(self, output_gm):
        """
        Move results from ws to output gm
        """
        burst_len_per_core = ceil_div(self.num_per_core * self.max_output_size_per_class * 3,
                                      Constant.B32_NUM_PER_BLOCK)
        burst_len_last_core = ceil_div(self.num_last_core * self.max_output_size_per_class * 3,
                                       Constant.B32_NUM_PER_BLOCK)
        move_temp = self.tik_instance.Tensor("int32", [self.selected_indices_aligned_shape],
                                             name="move_temp", scope=tik.scope_ubuf)
        offset_ws = self.tik_instance.Scalar("uint32", init_value=0)
        offset_gm = self.tik_instance.Scalar("uint32", init_value=0)

        with self.tik_instance.for_range(0, self.core_used) as core_idx:
            gm2ub(self.tik_instance, move_temp, self.selected_indices_ws,
                  self.selected_indices_aligned_shape, 0, offset_ws)
            with self.tik_instance.if_scope(core_idx == self.core_used - 1):
                self.tik_instance.data_move(output_gm[offset_gm], move_temp, 0, 1, burst_len_last_core, 0, 0)
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(output_gm[offset_gm], move_temp, 0, 1, burst_len_per_core, 0, 0)
            offset_ws.set_as(offset_ws + self.selected_indices_aligned_shape)
            offset_gm.set_as(offset_gm + self.num_per_core * self.max_output_size_per_class * 3)


def cur_sort_score_idx(tik_instance, scores_idx_temp, scores_idx_ub, sort_num, unit_ele, level=1):
    """
    Sort and merge score_index data with vmrgsort recursivelygm2ub
    """
    # Elements num of 4 queues
    compute_lens = Constant.VSORT_ELE_REPEAT * Constant.LIST_LEN_VMRGSORT ** level
    # Elements num of each queue
    unit_lens = compute_lens // Constant.LIST_LEN_VMRGSORT
    # Data length of each queue
    data_lens = unit_lens * unit_ele
    # Data length of 4 queues
    whole_lens = compute_lens * unit_ele
    loop_num = sort_num // whole_lens
    tail = sort_num % whole_lens
    offset = tik_instance.Scalar("uint32", init_value=0)

    if loop_num > 0:
        with tik_instance.for_range(0, loop_num) as idx:
            src_list = [scores_idx_temp[offset + data_lens * 3:offset + data_lens * 4],
                        scores_idx_temp[offset + data_lens * 2:offset + data_lens * 3],
                        scores_idx_temp[offset + data_lens:offset + data_lens * 2],
                        scores_idx_temp[offset:offset + data_lens]
                        ]
            count_list = [unit_lens, unit_lens, unit_lens, unit_lens]
            tik_instance.vmrgsort(scores_idx_ub[whole_lens * idx], src_list, count_list, False, 1)
            offset.set_as(offset + whole_lens)

    if tail != 0:
        tail_mode = tail // data_lens
        if tail_mode == 1:
            ub2ub(tik_instance, scores_idx_ub[offset], scores_idx_temp[offset], tail)
        if tail_mode == 2:
            src_list = [scores_idx_temp[offset + data_lens:offset + data_lens * 2],
                        scores_idx_temp[offset:offset + data_lens]]
            count_list = [unit_lens, unit_lens]
            tik_instance.vmrgsort(scores_idx_ub[offset], src_list, count_list, False, 1)
        else:
            src_list = [scores_idx_temp[offset + data_lens * 2:offset + data_lens * 3],
                        scores_idx_temp[offset + data_lens:offset + data_lens * 2],
                        scores_idx_temp[offset:offset + data_lens]
                        ]
            count_list = [unit_lens, unit_lens, unit_lens]
            tik_instance.vmrgsort(scores_idx_ub[offset], src_list, count_list, False, 1)

    if whole_lens >= sort_num:
        # The two inputs of this func will exchange position on each iter
        # A conditional statement is required to ensure output the correct input
        if level % 2 == 0:
            ub2ub(tik_instance, scores_idx_temp, scores_idx_ub, sort_num)
        return None
    level += 1

    return cur_sort_score_idx(tik_instance, scores_idx_ub, scores_idx_temp, sort_num, unit_ele, level)


def tik_func_vector_dup(tik_instance, dst, dup_len, value=0):
    """
    Duplicate value to dst with dup_len
    """
    mask, rep_stride, _ = get_mask_rep_stride(dst)
    repeat = dup_len // mask
    repeat_tail = dup_len % mask
    offset = 0
    loop_num = repeat // Constant.REPEAT_TIMES_MAX

    with tik_instance.if_scope(loop_num > 0):
        with tik_instance.for_range(0, loop_num) as _:
            tik_instance.vector_dup(mask, dst[offset], value, Constant.REPEAT_TIMES_MAX, 1, rep_stride)
            repeat = repeat - Constant.REPEAT_TIMES_MAX

    offset = Constant.REPEAT_TIMES_MAX * loop_num * mask
    repeat = repeat % Constant.REPEAT_TIMES_MAX
    with tik_instance.if_scope(repeat > 0):
        tik_instance.vector_dup(mask, dst[offset], value, repeat, 1, rep_stride)
    offset = offset + mask * repeat
    with tik_instance.if_scope(repeat_tail > 0):
        tik_instance.vector_dup(repeat_tail, dst[offset], value, 1, 1, rep_stride)


def vec_dup_gm(tik_instance, dst, dtype, dup_len, offset, num_block, output_limit, value=0):
    """
    Duplicate value to gm tensor
    """
    loop_time = dup_len // output_limit
    loop_tail = dup_len % output_limit
    if loop_time > 0:
        with tik_instance.new_stmt_scope():
            dup_ub = tik_instance.Tensor(dtype, (output_limit,), name="dup_ub", scope=tik.scope_ubuf)
            tik_func_vector_dup(tik_instance, dup_ub, output_limit, value)
            brust_len = ceil_div(output_limit, num_block)
            with tik_instance.for_range(0, loop_time) as i:
                tik_instance.data_move(dst[offset+i*output_limit], dup_ub, 0, 1, brust_len, 0, 0)
    if loop_tail != 0:
        with tik_instance.new_stmt_scope():
            dup_ub = tik_instance.Tensor(dtype, (loop_tail,), name="dup_ub", scope=tik.scope_ubuf)
            tik_func_vector_dup(tik_instance, dup_ub, loop_tail, value)
            brust_len = ceil_div(loop_tail, num_block)
            tik_instance.data_move(dst[offset+loop_time*output_limit], dup_ub, 0, 1, brust_len, 0, 0)


def get_mask_rep_stride(src):
    """
    Calculate mask and repeat stride for vector commands
    """
    if src.dtype in ["float16", "int16", "uint16"]:
        mask = Constant.B16_VEC_MASK
        block_ele = Constant.B16_NUM_PER_BLOCK
    elif src.dtype in ["float32", "int32", "uint32"]:
        mask = Constant.B32_VEC_MASK
        block_ele = Constant.B32_NUM_PER_BLOCK
    elif src.dtype in ["int8"]:
        mask = Constant.B8_VEC_MASK
        block_ele = Constant.B8_NUM_PER_BLOCK
    elif src.dtype in ["int64", "uint64"]:
        mask = Constant.B64_VEC_MASK
        block_ele = Constant.B64_NUM_PER_BLOCK
    else:
        raise RuntimeError("Incorrect dtype of src tensor.")
    rep_stride = mask // block_ele
    return mask, rep_stride, block_ele


def init_index(tik_instance, idx_gm, index, offset, index_num):
    """
    Move gm index data to ub index for tik commond vsort32
    """
    burst_lens = index_num // Constant.B32_NUM_PER_BLOCK
    tik_instance.data_move(index, idx_gm[offset], 0, 1, burst_lens, 0, 0)


def gm2ub_for_sort(tik_instance, dst, src, idx_list, sort_num):
    """
    Move data from gm to ub
    """
    _, _, block_ele = get_mask_rep_stride(src)
    burst_lens = sort_num // block_ele
    tail_num = sort_num % block_ele
    batch_idx, class_idx, box_idx = idx_list
    with tik_instance.if_scope(burst_lens > 0):
        tik_instance.data_move(dst, src[batch_idx, class_idx, box_idx], 0, 1, burst_lens, 0, 0)
        box_idx = box_idx + burst_lens * block_ele
    with tik_instance.for_range(0, tail_num) as idx:
        dst[burst_lens * block_ele + idx].set_as(src[batch_idx, class_idx, box_idx + idx])


def gm2ub(tik_instance, dst, src, data_num, dst_offset=0, src_offset=0):
    """
    Move data from gm to ub
    """
    _, _, block_ele = get_mask_rep_stride(src)
    burst_lens = data_num // block_ele
    tail_num = data_num % block_ele
    with tik_instance.if_scope(burst_lens > 0):
        tik_instance.data_move(dst[dst_offset], src[src_offset], 0, 1, burst_lens, 0, 0)
        src_offset += burst_lens * block_ele
        dst_offset += burst_lens * block_ele
    with tik_instance.for_range(0, tail_num) as idx:
        dst[dst_offset + idx].set_as(src[src_offset + idx])


# pylint: disable=too-many-arguments,huawei-too-many-arguments
def gm2ub_with_data_move_pad(tik_instance, dst, src, data_num, dst_offset=0, src_offset=0):
    """
    Move data from gm to ub with data_move_pad
    """
    _, _, block_ele = get_mask_rep_stride(src)
    burst_lens = data_num // block_ele
    tail_num = data_num % block_ele
    if dst.dtype == "float16":
        byte = Constant.B16_SIZE
    else:
        byte = Constant.B32_SIZE
    with tik_instance.if_scope(burst_lens > 0):
        tik_instance.data_move(dst[dst_offset], src[src_offset], 0, 1, burst_lens, 0, 0)
        src_offset += burst_lens * block_ele
        dst_offset += burst_lens * block_ele
    with tik_instance.if_scope(tail_num > 0):
        burst_lens = tail_num * byte
        tik_instance.data_move_pad(dst[dst_offset], src[src_offset], 1, burst_lens, 0, 0)


def ub2ub(tik_instance, dst, src, count, tail_overlap=False):
    """
    Move data from ub to ub
    """
    _, _, block_ele = get_mask_rep_stride(src)
    if tail_overlap:
        burst = ceil_div(count, block_ele)
        tik_instance.data_move(dst, src, 0, 1, burst, 0, 0)
    else:
        burst = count // block_ele
        with tik_instance.if_scope(burst != 0):
            tik_instance.data_move(dst, src, 0, 1, burst, 0, 0)
        new_index = block_ele * burst
        with tik_instance.for_range(new_index, count) as index:
            dst[index] = src[index]


def tik_func_add_sub_mul(tik_instance, func, dst, src0, src1, cal_num):
    """
    Do mul, add and sub computation by tik commands
    """
    mask, rep_stride, _ = get_mask_rep_stride(dst)
    repeat_time = cal_num // mask
    repeat_tail = cal_num % mask
    tik_fun = None
    if func == "vadd":
        tik_fun = tik_instance.vadd
    if func == "vsub":
        tik_fun = tik_instance.vsub
    if func == "vmul":
        tik_fun = tik_instance.vmul
    with tik_instance.if_scope(repeat_time > 0):
        tik_fun(mask, dst, src0[0], src1[0], repeat_time, 1, 1, 1, rep_stride, rep_stride, rep_stride)
    with tik_instance.if_scope(repeat_tail > 0):
        offset = repeat_time * mask
        tik_fun(repeat_tail, dst[offset], src0[offset], src1[offset], 1, 1, 1, 1, rep_stride, rep_stride, rep_stride)


def tik_func_vadds_vmuls(tik_instance, func, dst, src, scalar, cal_num):
    """
    Do vadds and vmuls computation by tik commands
    """
    mask, rep_stride, _ = get_mask_rep_stride(dst)
    repeat_time = cal_num // mask
    repeat_tail = cal_num % mask
    tik_fun = None
    if func == "vadds":
        tik_fun = tik_instance.vadds
    if func == "vmuls":
        tik_fun = tik_instance.vmuls
    with tik_instance.if_scope(repeat_time > 0):
        tik_fun(mask, dst, src[0], scalar, repeat_time, 1, 1, rep_stride, rep_stride)
    with tik_instance.if_scope(repeat_tail > 0):
        offset = repeat_time * mask
        tik_fun(repeat_tail, dst[offset], src[offset], scalar, 1, 1, 1, rep_stride, rep_stride)


def tik_func_vmaxs_vmins(tik_instance, func, dst, src, scalar, cal_num):
    """
    Do vmaxs and vmins computation by tik commands
    """
    mask, rep_stride, _ = get_mask_rep_stride(dst)
    repeat_time = cal_num // mask
    repeat_tail = cal_num % mask
    tik_fun = None
    if func == "vmaxs":
        tik_fun = tik_instance.vmaxs
    if func == "vmins":
        tik_fun = tik_instance.vmins
    if repeat_time > 0:
        tik_fun(mask, dst, src[0], scalar, repeat_time, 1, 1, rep_stride, rep_stride)
    if repeat_tail > 0:
        offset = repeat_time * mask
        tik_fun(repeat_tail, dst[offset], src[offset], scalar, 1, 1, 1, rep_stride, rep_stride)


def ceil_div(value, factor):
    """
    Calculate the smallest integer value  greater than or equal to value/factor
    """
    result = (value + (factor - 1)) // factor
    return result


def total_num(shape):
    """
    Compute the total num of elements in tht input shape
    """
    shape_total_num = functools.reduce(lambda a, b: a * b, shape)
    return shape_total_num


def non_max_suppression_v7_without_proposal(boxes, scores, max_output_size, iou_threshold, score_threshold, index_id,
                                            center_point_box, max_boxes_size, kernel_name):
    """
    Main entrance of implementation solution for next generation chips
    """
    # Instantiation
    nms = NonMaxSupperssionV7WithoutProposal(boxes, scores, max_output_size, iou_threshold,
                                             score_threshold, index_id, center_point_box, max_boxes_size)

    # Start computing
    nms.do_compute()

    return nms.build_tik_instance(kernel_name)
