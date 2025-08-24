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
yolo_boxes_encode
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from tbe.common.platform import get_bit_len


# the number of bits per byte
THREAD_NUM = 2
# the number of data contained in each coordinate box
DEFAULT_NBURST = 1
# The Maximum number of float16 data can store in UB with pingpong (256 * 15)
MAX_UB_ELEMENT_NUMBER_FP16 = 5120
# The Maximum number of float32 data can store in UB with pingpong(256 * 15)
MAX_UB_ELEMENT_NUMBER_FP32 = 5120
# the number of blocks included in each repeat with float16
BLOCK_NUMBER_FP16 = 32
# the number of blocks included in each repeat with float32
BLOCK_NUMBER_FP32 = 64
# one block size takes up 32b
BLOCK_SIZE = 32
# min value
CLAMP_MIN_FP16_VALUE = 1e-6
# max value
CLAMP_MAX_FP16_VALUE = 65504.0
# data type of fp16
FP16 = "float16"
# data type of fp32
FP32 = "float32"
# data type of int32
INT32 = "int32"
# number of element of fp16 and fp32 data type in one mask
VEC_MASK = {FP16: 128, FP32: 64, INT32: 64}
# repeat times
REP_TIMES = {FP16: 2, FP32: 1}
# repeat stride
REP_STRIDE = {FP16: 4, FP32: 8}
# calc mode
CALC_MODE = {"high_precision": 1, "high_performance": 2}
# the number of last dim of input
LAST_DIM = 4


# 'pylint: disable=too-many-instance-attributes
class YoloBoxesEncode():
    """
    Funtion: use to store BoundingBoxEncode base parameters
    """

    def _input_param_check(self):
        if (self.anchor_box_shape[0] != self.ground_truth_shape[0] or
                self.anchor_box_shape[1] != self.ground_truth_shape[1]):
            error_info = {'errCode': 'E80000',
                          'param_value1': self.anchor_box_shape[0],
                          'param_value2': self.ground_truth_shape[0],
                          'param_value3': self.anchor_box_shape[1],
                          'param_value4': self.ground_truth_shape[1],
                          'op_name': 'YoloBoxesEncode'}
            raise RuntimeError(error_info, "In op[%s], the dim-0 of input bboxes[%s] and gt_bboxes[%s] must be equal."
                                           "the dim-1 of input bboxes[%s] and gt_bboxes[%s] must be equal."
                               % (error_info['op_name'], error_info['param_value1'], error_info['param_value2'],
                                  error_info['param_value3'], error_info['param_value4']))

        if self.anchor_box_shape[1] != 4:
            error_info = {'errCode': 'E80001',
                          'param_value1': self.anchor_box_shape[1],
                          'op_name': 'YoloBoxesEncode'}
            raise RuntimeError(error_info, "In op[%s], the dim-1 of input bboxes[%s] must be equal to 4."
                               % (error_info['op_name'], error_info['param_value1']))

        if self.anchor_box_dtype != FP16 and self.anchor_box_dtype != FP32:
            error_info = {'errCode': 'E80002',
                          'param_value1': self.anchor_box_dtype,
                          'op_name': 'YoloBoxesEncode'}
            raise RuntimeError(error_info, "In op[%s], the dim-1 of input datatype[%s] only support fp16 or fp32."
                               % (error_info['op_name'], error_info['param_value1']))

        if self.anchor_box_shape[0] > MAX_UB_ELEMENT_NUMBER_FP32 * 4:
            error_info = {'errCode': 'E80003',
                          'param_value1': self.anchor_box_shape[0],
                          'op_name': 'YoloBoxesEncode'}
            raise RuntimeError(error_info, "In op[%s], the dim-0 of input length [%s] must less than 20480."
                               % (error_info['op_name'], error_info['param_value1']))

        if self.mode != "high_precision" and self.mode != "high_performance":
            error_info = {'errCode': 'E80004',
                          'param_value1': self.mode,
                          'op_name': 'YoloBoxesEncode'}
            raise RuntimeError(error_info, "In op[%s], the calc mode [%s] must be high_precision or high_performance."
                               % (error_info['op_name'], error_info['param_value1']))

    # 'pylint: disable=too-many-arguments
    def __init__(self, anchor_box, ground_truth_box, stride, encode_boxes, mode, kernel_name):
        self.init_tik_inst()
        self.anchor_box_shape = anchor_box.get("shape")
        self.anchor_box_dtype = anchor_box.get("dtype").lower()
        self.ground_truth_shape = ground_truth_box.get("shape")
        self.ground_truth_dtype = ground_truth_box.get("dtype").lower()
        self.stride_shape = stride.get("shape")
        self.stirde_dtype = stride.get("dtype").lower()
        self.encode_boxes = encode_boxes
        self.mode = mode
        self.kernel_name = kernel_name
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self._input_param_check()

        self.data_dtype_bytes_size = tbe_platform.get_bit_len(self.anchor_box_dtype) // 8
        self.data_num_in_each_block = BLOCK_SIZE // self.data_dtype_bytes_size
        self.ub_max_size = MAX_UB_ELEMENT_NUMBER_FP32
        self.each_core_start_addr, self.each_core_calc_num = self.get_core_param()
        self.init_gm_tensor()

        self.loop_cycle = self.get_loop_cycle()
        self.start_block_addr, self.block_number = self.get_loop_param()
        self.repeat_times = self.get_repeat_cycle()

    def init_tik_inst(self):
        """init_tik_inst

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.tik_inst = tik.Tik()
        self.support_div = tbe_platform.api_check_support("tik.vdiv", "float32")

    def data_move_mte2_function(self, loop_input, block_number):
        """data_move_mte2_function

        Parameters
        ----------
        loop_input : int
            loop index
        block_number: int
            block_number

        Returns
        -------
        result : list
            [anchor_box_ub, ground_truth_in_ub]
        """
        anchor_box_ub = self.tik_inst.Tensor(
            self.anchor_box_dtype, (self.ub_max_size // 4, 4),
            name="anchor_box_ub",
            scope=tbe_platform.scope_ubuf)
        self.tik_inst.data_move(anchor_box_ub, self.anchor_box_in[loop_input], 0,
                                DEFAULT_NBURST, block_number, 0, 0)

        ground_truth_in_ub = self.tik_inst.Tensor(
            self.ground_truth_dtype, (self.ub_max_size // 4, 4),
            name="ground_truth_in_ub",
            scope=tbe_platform.scope_ubuf)
        self.tik_inst.data_move(ground_truth_in_ub,
                                self.ground_truth_in[loop_input], 0,
                                DEFAULT_NBURST, block_number, 0, 0)

        stride_ub = self.tik_inst.Tensor(
            self.stirde_dtype, (self.ub_max_size // 4,), name="stride_ub", scope=tbe_platform.scope_ubuf)
        if self.ground_truth_dtype == FP32:
            self.tik_inst.data_move(stride_ub,
                                    self.stride_in[loop_input // 4], 0,
                                    DEFAULT_NBURST, max(block_number // 4, 1), 0, 0)
        else:
            self.tik_inst.data_move(stride_ub,
                                    self.stride_in[loop_input // 4], 0,
                                    DEFAULT_NBURST, max(block_number // 2, 1), 0, 0)
        src_ub_list = [anchor_box_ub, ground_truth_in_ub, stride_ub]
        return src_ub_list

    def data_move_mte3_function(self, loop_input, block_num, delta_dst_ub):
        """data_move_mte3_function

        Parameters
        ----------
        loop_input : int
            loop index
        block_num: int
            block_number
        delta_dst_ub : addr
            delta_dst_ub

        Returns
        -------
        None
        """
        self.tik_inst.data_move(self.encode_out[loop_input], delta_dst_ub,
                                0, DEFAULT_NBURST, block_num, 0, 0)

    def get_repeat_cycle(self):
        """data_move_mte2_function

        Parameters
        ----------
        None

        Returns
        -------
        result : int
            repeat_times
        """
        each_repeat_block_number = BLOCK_NUMBER_FP16
        if self.anchor_box_dtype == "float32":
            each_repeat_block_number = BLOCK_NUMBER_FP32
        if self.block_number < each_repeat_block_number:
            repeat_times = 1
        elif self.block_number % each_repeat_block_number == 0:
            repeat_times = self.block_number // each_repeat_block_number
        else:
            repeat_times = self.block_number // each_repeat_block_number + 1
        return repeat_times

    def get_core_param(self):
        """
        calculate data in number, each core start address
        """
        data_in_number = self.anchor_box_shape[0] * self.anchor_box_shape[1]
        if data_in_number < self.ub_max_size:
            each_core_start_address = 0
            each_core_calc_num = (data_in_number + 15) // 16 * 16
        else:
            each_core_start_address = (data_in_number // (self.core_num * LAST_DIM * self.data_num_in_each_block)) * \
                                      LAST_DIM * self.data_num_in_each_block

            # check input data number can equal divivde to (32 core * 4 point)
            if data_in_number % (self.core_num * LAST_DIM * self.data_num_in_each_block) == 0:
                # check input data number is equal to block
                if each_core_start_address % (self.data_num_in_each_block * LAST_DIM) == 0:
                    each_core_calc_num = each_core_start_address
                else:
                    each_core_calc_num = (each_core_start_address // (self.data_num_in_each_block * LAST_DIM) + 1) * \
                                         (self.data_num_in_each_block * LAST_DIM)
            else:
                each_core_calc_num = data_in_number - each_core_start_address * (self.core_num - 1)
                if each_core_calc_num % (self.data_num_in_each_block * LAST_DIM) != 0:
                    each_core_calc_num = (each_core_calc_num // (self.data_num_in_each_block * LAST_DIM) + 1) * \
                                         (self.data_num_in_each_block * LAST_DIM)
        return each_core_start_address, each_core_calc_num

    def tik_inst_function(self):
        """tik_inst_function

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with self.tik_inst.for_range(0, self.core_num, block_num=self.core_num) as block_id:
            self.calculation_process(block_id)
        self.tik_inst.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.anchor_box_in, self.ground_truth_in, self.stride_in],
            outputs=[self.encode_out])

    def init_gm_tensor(self):
        """init_gm_tensor

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        gm_shape_size = self.each_core_start_addr * (self.core_num - 1) + self.each_core_calc_num
        self.anchor_box_in = self.tik_inst.Tensor(
            self.anchor_box_dtype, (gm_shape_size // 4, 4),
            name="anchor_boxes",
            scope=tbe_platform.scope_gm)
        self.ground_truth_in = self.tik_inst.Tensor(
            self.ground_truth_dtype, (gm_shape_size // 4, 4),
            name="gt_bboxes", scope=tbe_platform.scope_gm)
        self.stride_in = self.tik_inst.Tensor(
            self.stirde_dtype, (gm_shape_size // 4,),
            name="stride", scope=tbe_platform.scope_gm)
        self.encode_out = self.tik_inst.Tensor(
            self.anchor_box_dtype, (gm_shape_size // 4, 4),
            name="encoded_bboxes", scope=tbe_platform.scope_gm)

    def get_loop_cycle(self):
        """get_loop_cycle

        Parameters
        ----------
        None

        Returns
        -------
        result : int
            loop_cycle
        """
        if self.each_core_calc_num % self.ub_max_size == 0:
            loop_cycle = int(self.each_core_calc_num // self.ub_max_size)
        else:
            loop_cycle = int(self.each_core_calc_num // self.ub_max_size + 1)

        return loop_cycle

    def get_loop_param(self):
        """get_loop_param

        Parameters
        ----------
        None

        Returns
        -------
        result : list
            [start_block_addr, block_number]
        """
        block_number = self.each_core_calc_num // self.data_num_in_each_block
        if block_number == 0:
            block_number = 1
        start_block_addr = block_number // self.loop_cycle

        if self.loop_cycle > 1:
            if block_number % self.loop_cycle != 0:
                block_number_loop = block_number - start_block_addr * (
                        self.loop_cycle - 1)
                while block_number * self.loop_cycle < block_number_loop or \
                        block_number_loop * 16 > MAX_UB_ELEMENT_NUMBER_FP16:
                    self.loop_cycle += 1
                    start_block_addr = block_number // self.loop_cycle
                    block_number_loop = block_number - start_block_addr * (
                            self.loop_cycle - 1)
                block_number = block_number_loop
            else:
                block_number = start_block_addr
        return start_block_addr, block_number

    def calculation_process(self, block_id):
        """get_loop_param

        Parameters
        ----------
        block_id : int
            block_id

        Returns
        -------
        None
        """
        if self.loop_cycle == 1:
            loop_input = block_id * self.each_core_start_addr
            str_ub_list = self.data_move_mte2_function(loop_input, self.block_number)
            dst_ub = self.bounding_box_encode_compute(str_ub_list, self.repeat_times)
            self.data_move_mte3_function(loop_input, self.block_number, dst_ub)
        else:
            loop_input = block_id * self.each_core_start_addr
            with self.tik_inst.for_range(0, self.loop_cycle, thread_num=THREAD_NUM) as cycle:
                loop_input = loop_input + cycle * self.start_block_addr * self.data_num_in_each_block
                str_ub_list = self.data_move_mte2_function(loop_input, self.block_number)
                dst_ub = self.bounding_box_encode_compute(str_ub_list, self.repeat_times)
                self.data_move_mte3_function(loop_input, self.block_number, dst_ub)

    def _get_ground_truth_position(self, ground_truth_box_ub, repeat_times):
        ground_truth_box_pos_ub = self.tik_inst.Tensor(
            FP32, (self.ub_max_size,), name="ground_truth_box_pos_ub", scope=tbe_platform.scope_ubuf)
        with self.tik_inst.for_range(0, 2) as iter_i:
            ground_truth_box_tmp_ub_0 = ground_truth_box_pos_ub[(0 + 8 * iter_i):]
            ground_truth_box_tmp_ub_1 = ground_truth_box_pos_ub[(16 + 8 * iter_i):]
            ground_truth_box_tmp_ub_2 = ground_truth_box_pos_ub[(32 + 8 * iter_i):]
            ground_truth_box_tmp_ub_3 = ground_truth_box_pos_ub[(48 + 8 * iter_i):]

            ground_truth_box_dst_ub_0 = ground_truth_box_ub[(0 + 8 * iter_i):]
            ground_truth_box_dst_ub_1 = ground_truth_box_ub[(16 + 8 * iter_i):]
            ground_truth_box_dst_ub_2 = ground_truth_box_ub[(32 + 8 * iter_i):]
            ground_truth_box_dst_ub_3 = ground_truth_box_ub[(48 + 8 * iter_i):]

            # Calculate ground truth center x
            self.tik_inst.vadd(VEC_MASK[FP32], ground_truth_box_tmp_ub_0, ground_truth_box_dst_ub_0,
                               ground_truth_box_dst_ub_2, repeat_times, REP_STRIDE[FP32], REP_STRIDE[FP32],
                               REP_STRIDE[FP32], 64, 64, 64)
            self.tik_inst.vmuls(VEC_MASK[FP32], ground_truth_box_tmp_ub_0, ground_truth_box_tmp_ub_0, 0.5,
                                repeat_times, REP_STRIDE[FP32], REP_STRIDE[FP32], 64, 64)

            # Calculate ground truth center y
            self.tik_inst.vadd(VEC_MASK[FP32], ground_truth_box_tmp_ub_1, ground_truth_box_dst_ub_1,
                               ground_truth_box_dst_ub_3, repeat_times, REP_STRIDE[FP32], REP_STRIDE[FP32],
                               REP_STRIDE[FP32], 64, 64, 64)
            self.tik_inst.vmuls(VEC_MASK[FP32], ground_truth_box_tmp_ub_1, ground_truth_box_tmp_ub_1,
                                0.5, repeat_times, REP_STRIDE[FP32], REP_STRIDE[FP32], 64, 64)

            # Calculate ground truth width
            self.tik_inst.vsub(VEC_MASK[FP32], ground_truth_box_tmp_ub_2, ground_truth_box_dst_ub_2,
                               ground_truth_box_dst_ub_0, repeat_times, REP_STRIDE[FP32], REP_STRIDE[FP32],
                               REP_STRIDE[FP32], 64, 64, 64)

            # Calculate ground truth height
            self.tik_inst.vsub(VEC_MASK[FP32], ground_truth_box_tmp_ub_3, ground_truth_box_dst_ub_3,
                               ground_truth_box_dst_ub_1, repeat_times, REP_STRIDE[FP32], REP_STRIDE[FP32],
                               REP_STRIDE[FP32], 64, 64, 64)

        return ground_truth_box_pos_ub

    def _get_anchor_boxes_position(self, anchor_box_dst_ub, repeat_times):
        anchor_box_pos_ub = self.tik_inst.Tensor(FP32, (self.ub_max_size,),
                                                 name="anchor_box_pos_ub", scope=tbe_platform.scope_ubuf)
        with self.tik_inst.for_range(0, 2) as iter_i:
            anchor_box_pos_ub_0 = anchor_box_pos_ub[(0 + 8 * iter_i):]
            anchor_box_pos_ub_1 = anchor_box_pos_ub[(16 + 8 * iter_i):]
            anchor_box_pos_ub_2 = anchor_box_pos_ub[(32 + 8 * iter_i):]
            anchor_box_pos_ub_3 = anchor_box_pos_ub[(48 + 8 * iter_i):]

            anchor_box_dst_ub_0 = anchor_box_dst_ub[(0 + 8 * iter_i):]
            anchor_box_dst_ub_1 = anchor_box_dst_ub[(16 + 8 * iter_i):]
            anchor_box_dst_ub_2 = anchor_box_dst_ub[(32 + 8 * iter_i):]
            anchor_box_dst_ub_3 = anchor_box_dst_ub[(48 + 8 * iter_i):]

            # calc anchor box center x
            self.tik_inst.vadd(VEC_MASK[FP32], anchor_box_pos_ub_0, anchor_box_dst_ub_0,
                               anchor_box_dst_ub_2, repeat_times, REP_STRIDE[FP32], REP_STRIDE[FP32],
                               REP_STRIDE[FP32], 64, 64, 64)
            self.tik_inst.vmuls(VEC_MASK[FP32], anchor_box_pos_ub_0, anchor_box_pos_ub_0, 0.5, repeat_times,
                                REP_STRIDE[FP32], REP_STRIDE[FP32], 64, 64)

            # calc anchor box center y
            self.tik_inst.vadd(VEC_MASK[FP32], anchor_box_pos_ub_1, anchor_box_dst_ub_1,
                               anchor_box_dst_ub_3, repeat_times, REP_STRIDE[FP32], REP_STRIDE[FP32],
                               REP_STRIDE[FP32], 64, 64, 64)
            self.tik_inst.vmuls(VEC_MASK[FP32], anchor_box_pos_ub_1, anchor_box_pos_ub_1, 0.5, repeat_times,
                                REP_STRIDE[FP32], REP_STRIDE[FP32], 64, 64)

            # calc anchor box width
            self.tik_inst.vsub(VEC_MASK[FP32], anchor_box_pos_ub_2, anchor_box_dst_ub_2,
                               anchor_box_dst_ub_0, repeat_times, REP_STRIDE[FP32], REP_STRIDE[FP32],
                               REP_STRIDE[FP32], 64, 64, 64)

            # calc anchor box height
            self.tik_inst.vsub(VEC_MASK[FP32], anchor_box_pos_ub_3, anchor_box_dst_ub_3,
                               anchor_box_dst_ub_1, repeat_times, REP_STRIDE[FP32], REP_STRIDE[FP32],
                               REP_STRIDE[FP32], 64, 64, 64)

        return anchor_box_pos_ub

    def _get_target_wh_info(self, src_0_ub, src_1_ub, target_info_ub, min_value, max_value, repeat_times, idx):
        if self.support_div is True:
            self.tik_inst.vdiv(VEC_MASK[FP32], target_info_ub[idx], src_0_ub[idx], src_1_ub[idx], repeat_times,
                               REP_STRIDE[FP32], REP_STRIDE[FP32], REP_STRIDE[FP32], 64, 64, 64)
        else:
            self.tik_inst.vrec(VEC_MASK[FP32], target_info_ub[idx], src_1_ub[idx], repeat_times, REP_STRIDE[FP32],
                               REP_STRIDE[FP32], 64, 64)
            self.tik_inst.vmul(VEC_MASK[FP32], target_info_ub[idx], src_0_ub[idx], target_info_ub[idx], repeat_times,
                               REP_STRIDE[FP32], REP_STRIDE[FP32], REP_STRIDE[FP32], 64, 64, 64)
        self._clamp_by_min_max_value(target_info_ub, min_value, max_value, repeat_times, idx)
        self.tik_inst.vln(VEC_MASK[FP32], target_info_ub[idx], target_info_ub[idx],
                          repeat_times, REP_STRIDE[FP32], REP_STRIDE[FP32], 64, 64)

    def _clamp_by_min_max_value(self, src_ub, min_value, max_value, repeat_times, idx):
        with self.tik_inst.new_stmt_scope():
            compare_value_ub = self.tik_inst.Tensor(FP32, (self.ub_max_size,),
                                                    name="compare_value_ub", scope=tbe_platform.scope_ubuf)
            self.tik_inst.vector_dup(VEC_MASK[FP32], compare_value_ub, min_value,
                                     self.ub_max_size // VEC_MASK[FP32], 1, 8)
            self.tik_inst.vmax(VEC_MASK[FP32], src_ub[idx], src_ub[idx], compare_value_ub,
                               repeat_times, REP_STRIDE[FP32], REP_STRIDE[FP32], REP_STRIDE[FP32], 64, 64, 64)
            self.tik_inst.vector_dup(VEC_MASK[FP32], compare_value_ub, max_value,
                                     self.ub_max_size // VEC_MASK[FP32], 1, 8)
            self.tik_inst.vmin(VEC_MASK[FP32], src_ub[idx], src_ub[idx], compare_value_ub,
                               repeat_times, REP_STRIDE[FP32], REP_STRIDE[FP32], REP_STRIDE[FP32], 64, 64, 64)

    def _get_target_center_xy_info(self, src_0_ub, src_1_ub, stride_ub, center_info_ub,
                                   min_value, max_value, repeat_times, idx):
        with self.tik_inst.new_stmt_scope():
            self.tik_inst.vsub(VEC_MASK[FP32], center_info_ub[idx], src_0_ub[idx], src_1_ub[idx], repeat_times,
                               REP_STRIDE[FP32], REP_STRIDE[FP32], REP_STRIDE[FP32], 64, 64, 64)
            if self.support_div is True:
                self.tik_inst.vdiv(VEC_MASK[FP32], center_info_ub[idx], center_info_ub[idx], stride_ub[idx],
                                   repeat_times, REP_STRIDE[FP32], REP_STRIDE[FP32], REP_STRIDE[FP32], 64, 64, 64)
            else:
                stride_fp32_ub = self.tik_inst.Tensor(FP32, (self.ub_max_size,),
                                                      name="stride_fp32_ub", scope=tbe_platform.scope_ubuf)
                self.tik_inst.vrec(VEC_MASK[FP32], stride_fp32_ub, stride_ub, repeat_times, REP_STRIDE[FP32],
                                   REP_STRIDE[FP32], 64, 64)
                self.tik_inst.vmul(VEC_MASK[FP32], center_info_ub[idx], center_info_ub[idx], stride_fp32_ub[idx],
                                   repeat_times, REP_STRIDE[FP32], REP_STRIDE[FP32], REP_STRIDE[FP32], 64, 64, 64)

            self.tik_inst.vadds(VEC_MASK[FP32], center_info_ub[idx], center_info_ub[idx], 0.5, repeat_times,
                                REP_STRIDE[FP32], REP_STRIDE[FP32], 64, 64)
            self._clamp_by_min_max_value(center_info_ub, min_value, max_value, repeat_times, idx)

    def _get_transpose_ub(self, src_ub, dst_ub, repeat_times, data_type):
        with self.tik_inst.new_stmt_scope():
            convert_box_ub = self.tik_inst.Tensor(FP16, (self.ub_max_size,),
                                                  name="convert_box_ub", scope=tbe_platform.scope_ubuf)
            convert_box_fp16_ub = self.tik_inst.Tensor(FP16, (self.ub_max_size,),
                                                       name="convert_box_fp16_ub", scope=tbe_platform.scope_ubuf)
            # convert float32 to float16
            if data_type == "float32":
                self.tik_inst.vconv(VEC_MASK[FP32], 'none', convert_box_ub, src_ub, repeat_times * 8, 1, 1, 4, 8)
            else:
                convert_box_ub = src_ub

            # transverse input data use vnchwconv instruction
            src_list = [convert_box_ub[16 * i] for i in range(16)]
            dst_list = [convert_box_fp16_ub[16 * i] for i in range(16)]

            # transform anchor_box and ground truth box
            self.tik_inst.vnchwconv(True, True, dst_list, src_list, repeat_times * 2, 16, 16)

            self.tik_inst.vconv(VEC_MASK[FP32], 'none', dst_ub, convert_box_fp16_ub, repeat_times * 8, 1, 1, 8, 4)

    def _get_transpose_small_data_fp32_ub(self, src_ub, dst_ub):
        transpose_num = 256
        align_num = 16
        iter_block_num = (self.each_core_calc_num + transpose_num - 1) // transpose_num
        with self.tik_inst.for_range(0, iter_block_num) as block_id:
            src_ub_ptr = src_ub[transpose_num * block_id]
            dst_ub_ptr = dst_ub[transpose_num * block_id]
            with self.tik_inst.for_range(0, transpose_num) as num_id:
                row_id = num_id % align_num
                col_id = num_id // align_num
                dst_ub_ptr[row_id * align_num + col_id] = src_ub_ptr[num_id]

    def _get_transpose_small_data_fp16_ub(self, src_ub, dst_ub):
        with self.tik_inst.new_stmt_scope():
            src_fp32_ub = self.tik_inst.Tensor(FP32, (self.ub_max_size,),
                                               name="src_fp32_ub", scope=tbe_platform.scope_ubuf)
            self.tik_inst.vconv(VEC_MASK[FP32], 'none', src_fp32_ub, src_ub, self.ub_max_size // VEC_MASK[FP32],
                                1, 1, 8, 4)
            self._get_transpose_small_data_fp32_ub(src_fp32_ub, dst_ub)

    def _generate_stride_fp32_ub(self, stride_ub, box_stride_dst_ub, repeat_times):
        with self.tik_inst.new_stmt_scope():
            stride_fp32_ub = self.tik_inst.Tensor(FP32, (self.ub_max_size,),
                                                  name="stride_fp32_ub", scope=tbe_platform.scope_ubuf)
            self.tik_inst.vector_dup(VEC_MASK[FP32], stride_fp32_ub, 1.0, self.ub_max_size // VEC_MASK[FP32],
                                     1, 8)
            box_stride_fp16_ub = self.tik_inst.Tensor(FP16, (self.ub_max_size // 4,),
                                                      name="box_stride_fp16_ub", scope=tbe_platform.scope_ubuf)
            self.tik_inst.vector_dup(VEC_MASK[FP16], box_stride_fp16_ub, 1.0, self.ub_max_size // 4 // 128, 1, 8)
            self.tik_inst.vconv(VEC_MASK[INT32], 'none', box_stride_fp16_ub, stride_ub, repeat_times * 2, 1, 1,
                                REP_STRIDE[FP16], REP_STRIDE[FP32], deqscale=1.0)

            box_stride_fp32_ub = self.tik_inst.Tensor(FP32, (self.ub_max_size // 4,),
                                                      name="box_stride_fp32_ub", scope=tbe_platform.scope_ubuf)
            self.tik_inst.vconv(VEC_MASK[FP32], 'none', box_stride_fp32_ub, box_stride_fp16_ub, repeat_times * 2,
                                1, 1, REP_STRIDE[FP32], REP_STRIDE[FP16])
            with self.tik_inst.for_range(0, self.each_core_calc_num // 4) as iter_i:
                stride_fp32_ub[4 * iter_i] = box_stride_fp32_ub[iter_i]
                stride_fp32_ub[4 * iter_i + 1] = box_stride_fp32_ub[iter_i]
                stride_fp32_ub[4 * iter_i + 2] = box_stride_fp32_ub[iter_i]
                stride_fp32_ub[4 * iter_i + 3] = box_stride_fp32_ub[iter_i]

            self._get_transpose_small_data_fp32_ub(stride_fp32_ub, box_stride_dst_ub)

    def get_encode_out(self, input_ub_list, encode_out_ub, repeat_times):
        anchor_box_src_ub, ground_truth_box_src_ub, stride_ub = input_ub_list[0:3]
        anchor_box_dst_ub = self.tik_inst.Tensor(FP32, (self.ub_max_size,),
                                                 name="anchor_box_dst_ub", scope=tbe_platform.scope_ubuf)
        ground_truth_box_dst_ub = self.tik_inst.Tensor(FP32, (self.ub_max_size,),
                                                       name="ground_truth_box_dst_ub", scope=tbe_platform.scope_ubuf)
        box_stride_dst_ub = self.tik_inst.Tensor(FP32, (self.ub_max_size,),
                                                 name="box_stride_dst_ub", scope=tbe_platform.scope_ubuf)
        if self.mode == "high_precision":
            if self.anchor_box_dtype == FP32:
                self._get_transpose_small_data_fp32_ub(ground_truth_box_src_ub, ground_truth_box_dst_ub)
                self._get_transpose_small_data_fp32_ub(anchor_box_src_ub, anchor_box_dst_ub)
            else:
                self._get_transpose_small_data_fp16_ub(ground_truth_box_src_ub, ground_truth_box_dst_ub)
                self._get_transpose_small_data_fp16_ub(anchor_box_src_ub, anchor_box_dst_ub)
        else:
            self._get_transpose_ub(ground_truth_box_src_ub, ground_truth_box_dst_ub, repeat_times,
                                   self.ground_truth_dtype)
            self._get_transpose_ub(anchor_box_src_ub, anchor_box_dst_ub, repeat_times, self.anchor_box_dtype)

        self._generate_stride_fp32_ub(stride_ub, box_stride_dst_ub, repeat_times)

        ground_truth_box_ub = self._get_ground_truth_position(ground_truth_box_dst_ub, repeat_times)
        anchor_box_pos_ub = self._get_anchor_boxes_position(anchor_box_dst_ub, repeat_times)

        self._get_target_wh_info(ground_truth_box_ub, anchor_box_pos_ub, encode_out_ub,
                                 CLAMP_MIN_FP16_VALUE, CLAMP_MAX_FP16_VALUE, repeat_times, 32)
        self._get_target_wh_info(ground_truth_box_ub, anchor_box_pos_ub, encode_out_ub,
                                 CLAMP_MIN_FP16_VALUE, CLAMP_MAX_FP16_VALUE, repeat_times, 40)
        self._get_target_wh_info(ground_truth_box_ub, anchor_box_pos_ub, encode_out_ub,
                                 CLAMP_MIN_FP16_VALUE, CLAMP_MAX_FP16_VALUE, repeat_times, 48)
        self._get_target_wh_info(ground_truth_box_ub, anchor_box_pos_ub, encode_out_ub,
                                 CLAMP_MIN_FP16_VALUE, CLAMP_MAX_FP16_VALUE, repeat_times, 56)

        self._get_target_center_xy_info(ground_truth_box_ub, anchor_box_pos_ub, box_stride_dst_ub,
                                        encode_out_ub, CLAMP_MIN_FP16_VALUE, 1.0 - CLAMP_MIN_FP16_VALUE,
                                        repeat_times, 0)
        self._get_target_center_xy_info(ground_truth_box_ub, anchor_box_pos_ub, box_stride_dst_ub,
                                        encode_out_ub, CLAMP_MIN_FP16_VALUE, 1.0 - CLAMP_MIN_FP16_VALUE,
                                        repeat_times, 8)
        self._get_target_center_xy_info(ground_truth_box_ub, anchor_box_pos_ub, box_stride_dst_ub,
                                        encode_out_ub, CLAMP_MIN_FP16_VALUE, 1.0 - CLAMP_MIN_FP16_VALUE,
                                        repeat_times, 16)
        self._get_target_center_xy_info(ground_truth_box_ub, anchor_box_pos_ub, box_stride_dst_ub,
                                        encode_out_ub, CLAMP_MIN_FP16_VALUE, 1.0 - CLAMP_MIN_FP16_VALUE,
                                        repeat_times, 24)

    # 'pylint: disable=too-many-locals,too-many-statements,too-many-branches
    def bounding_box_encode_compute(self, input_ub_list, repeat_times):
        """
        use tbe_platform instruction to calculate result bounding_box_encode_compute
        Parameters
        ----------
        :param input_ub_list:
        repeat_times : int
            repeat_times
        Returns
        -------
        delta_out_ub : TVM tensor
        """
        encode_out_ub = self.tik_inst.Tensor(FP32, (self.ub_max_size,), name="encode_out_ub",
                                             scope=tbe_platform.scope_ubuf)

        with self.tik_inst.new_stmt_scope():
            self.get_encode_out(input_ub_list, encode_out_ub, repeat_times)
        # transverse output data back
        delta_out_ub = self.tik_inst.Tensor(FP32, (self.ub_max_size,), name="delta_out_ub",
                                            scope=tbe_platform.scope_ubuf)
        if self.mode == "high_precision":
            self._get_transpose_small_data_fp32_ub(encode_out_ub, delta_out_ub)
        else:
            self._get_transpose_ub(encode_out_ub, delta_out_ub, repeat_times, FP32)
        if self.anchor_box_dtype == FP16:
            delta_out_fp16_ub = self.tik_inst.Tensor("float16", (self.ub_max_size,),
                                                     name="delta_out_fp16_ub", scope=tbe_platform.scope_ubuf)
            self.tik_inst.vconv(VEC_MASK[FP32], 'none', delta_out_fp16_ub,
                                delta_out_ub, repeat_times * 8, 1, 1, REP_STRIDE[FP16], REP_STRIDE[FP32])
            return delta_out_fp16_ub
        return delta_out_ub


# 'pylint: disable=too-many-arguments
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def yolo_boxes_encode(anchor_boxes, gt_bboxes, stride, encoded_bboxes, performance_mode="high_precision",
                      kernel_name="yolo_boxes_encode"):
    """
    algorithm: bounding_box_encode

    Parameters
    ----------
    anchor_boxes : dict
        shape and dtype of input
    gt_bboxes : dict
        shape and dtype of input
    stride : dict
        shape and dtype of input
    encoded_bboxes: dict
        shape and dtype of output
    mode: attr
        attribute, 1-high_precise, 2-high_performance
    kernel_name : str
        kernel name, default value is "yolo_boxes_encode"

    Returns
    -------
    None
    """
    bounding_box_encode_ = YoloBoxesEncode(anchor_boxes, gt_bboxes, stride, encoded_bboxes, performance_mode,
                                           kernel_name)
    bounding_box_encode_.tik_inst_function()

    return bounding_box_encode_.tik_inst
