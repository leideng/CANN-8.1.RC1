#!/usr/bin/python
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
grid_assign_positive.py
"""

from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector

from impl.ascend import AContainer
from impl.ascend import TensorOperatorParam
from impl.ascend import VecCmd
from impl.ascend import VecExecutor
from impl.util.util_tik_comm_func import ceil_div


class GridAssignPositive:
    """
    The class for GridAssignPositive.
    """

    # 'pylint: disable=too-many-arguments
    def __init__(self, k_num, n_num, pos_iou_thr, min_pos_iou, gt_max_assign_all,
                 data_type, flag_type, int_type, kernel_name, cont):
        self.k_num = k_num
        self.n_num = n_num
        self.pos_iou_thr = pos_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.data_type = data_type
        self.flag_type = flag_type
        self.flag_byte = 8
        self.int_type = int_type
        self.kernel_name = kernel_name

        self.cont = cont
        self.tik = self.cont.tik
        self.tik_inst = self.cont.tinst
        self.ai_core_use = self.cont.const_aicore_num

        self.data_size, self.data_block_data_num, self.data_repeat_data_num = self.get_type_const(self.data_type)
        self.int_size, self.int_block_data_num, self.int_repeat_data_num = self.get_type_const(self.int_type)
        self.flag_size, self.flag_block_data_num, self.flag_repeat_data_num = self.get_type_const(self.flag_type)

        flag_shape = (self.n_num,)
        gt_flag_shape = (self.k_num,)
        overlaps_shape = (self.k_num, self.n_num)
        self.assigned_gt_inds = self.tik_inst.Tensor(
            self.data_type, flag_shape, self.tik.scope_gm, "assigned_gt_inds")
        self.overlaps = self.tik_inst.Tensor(
            self.data_type, overlaps_shape, self.tik.scope_gm, "overlaps")
        self.box_responsible_flags = self.tik_inst.Tensor(
            self.flag_type, flag_shape, self.tik.scope_gm, "box_responsible_flags")

        self.max_overlaps = self.tik_inst.Tensor(
            self.data_type, flag_shape, self.tik.scope_gm, "max_overlaps")
        self.argmax_overlaps = self.tik_inst.Tensor(
            self.int_type, flag_shape, self.tik.scope_gm, "argmax_overlaps")

        self.gt_max_overlaps = self.tik_inst.Tensor(
            self.data_type, gt_flag_shape, self.tik.scope_gm, "gt_max_overlaps")
        self.gt_argmax_overlaps = self.tik_inst.Tensor(
            self.int_type, gt_flag_shape, self.tik.scope_gm, "gt_argmax_overlaps")

        self.num_gts = self.tik_inst.Tensor(self.int_type, (1,), self.tik.scope_gm, "num_gts")
        self.assigned_gt_inds_pos = self.tik_inst.Tensor(
            self.data_type, flag_shape, self.tik.scope_gm, "assigned_gt_inds_pos")

    @staticmethod
    def get_loop_info(all_data_num, each_loop_num):
        """
        The function is get loop info.
        """
        loop_times = ceil_div(all_data_num, each_loop_num)
        last_loop_num = all_data_num - each_loop_num * (loop_times - 1)
        return loop_times, last_loop_num

    @staticmethod
    def get_align_num(input_num, align_num, ceil=True):
        """
        The function is get align num.
        """
        if ceil:
            result = ceil_div(input_num, align_num) * align_num
        else:
            result = input_num // align_num * align_num
        return result

    def get_type_const(self, data_type):
        """
        The function is get type const.
        """
        data_size = self.cont.const_dtype_byte.get(data_type)
        block_data_num = self.cont.get_vec_proc_num_per_cmd_blk(data_type)
        repeat_data_num = self.cont.get_vec_proc_num_per_cmd(data_type)
        return data_size, block_data_num, repeat_data_num

    def mode_compute(self):
        """
        The function is mode compute.
        """
        if self.gt_max_assign_all:
            each_core_n_num = ceil_div(self.n_num, self.ai_core_use)
            each_core_n_num = self.get_align_num(each_core_n_num, self.data_repeat_data_num)
            self.ai_core_use, last_core_n_num = self.get_loop_info(self.n_num, each_core_n_num)
            with self.tik_inst.for_range(0, self.ai_core_use, block_num=self.ai_core_use) as core_index:
                n_index_start = each_core_n_num * core_index
                with self.tik_inst.if_scope(core_index != self.ai_core_use - 1):
                    self._mode_1_compute_each_core(n_index_start, each_core_n_num)
                with self.tik_inst.else_scope():
                    self._mode_1_compute_each_core(n_index_start, last_core_n_num)
        else:
            self._mode_0_compute_each_core()
        inputs_all = [self.assigned_gt_inds, self.overlaps, self.box_responsible_flags,
                      self.max_overlaps, self.argmax_overlaps, self.gt_max_overlaps, self.gt_argmax_overlaps,
                      self.num_gts]
        outputs_all = [self.assigned_gt_inds_pos]
        self.tik_inst.BuildCCE(
            inputs=inputs_all,
            outputs=outputs_all,
            kernel_name=self.kernel_name)

    def _mode_1_compute_each_core(self, n_index_start, n_num):
        each_loop_n_num = self.data_repeat_data_num
        loop_times, last_loop_n_num = self.get_loop_info(n_num, each_loop_n_num)
        gt_min_pos_iou_ub = self._get_gt_min_pos_iou_ub()
        num_gts_scalar = self._init_num_gts()
        with self.tik_inst.for_range(0, loop_times) as loop_index:
            n_index_start_loop = each_loop_n_num * loop_index + n_index_start
            with self.tik_inst.if_scope(loop_index != loop_times - 1):
                self._mode_1_compute_each_loop(gt_min_pos_iou_ub, num_gts_scalar, n_index_start_loop, each_loop_n_num)
            with self.tik_inst.else_scope():
                self._mode_1_compute_each_loop(gt_min_pos_iou_ub, num_gts_scalar, n_index_start_loop, last_loop_n_num)

    def _mode_1_compute_each_loop(self, gt_min_pos_iou_ub, num_gts_scalar, n_index_start, n_num):
        n_num_align = self.data_repeat_data_num
        assigned_gt_inds_ub, flag_ub = self._assigned_gt_inds_pos_iou_each_loop(n_index_start, n_num)
        gt_max_overlaps_flag_scalar = self.tik_inst.Scalar(self.int_type)
        with self.tik_inst.for_range(0, num_gts_scalar) as k_index:
            gt_max_overlaps_flag_scalar.set_as(gt_min_pos_iou_ub[k_index])
            with self.tik_inst.if_scope(gt_max_overlaps_flag_scalar == 1):
                self._mode_1_assign_positive(assigned_gt_inds_ub, flag_ub, k_index, n_index_start, n_num, n_num_align)
        block_num = ceil_div(n_num, self.data_block_data_num)
        self.tik_inst.data_move(self.assigned_gt_inds_pos[n_index_start], assigned_gt_inds_ub, 0, 1, block_num, 0, 0)

    # 'pylint: disable=too-many-locals,too-many-arguments
    def _mode_1_assign_positive(self, assigned_gt_inds_ub, flag_ub, k_index, n_index_start, n_num, n_num_align):
        data_shape = (n_num_align,)
        gt_max_overlaps_scalar = self.tik_inst.Scalar(self.data_type)
        overlaps_ub = self.tik_inst.Tensor(self.data_type, data_shape, self.tik.scope_ubuf, "overlaps_ub")
        gt_max_overlaps_ub = self.tik_inst.Tensor(self.data_type, data_shape, self.tik.scope_ubuf, "gt_max_overlaps_ub")
        block_num = ceil_div(n_num, self.data_block_data_num)
        self.tik_inst.data_move(overlaps_ub, self.overlaps[k_index, n_index_start], 0, 1, block_num, 0, 0)
        self.tik_inst.data_move(gt_max_overlaps_ub, self.gt_max_overlaps[k_index], 0, 1, 1, 0, 0)
        gt_max_overlaps_scalar.set_as(gt_max_overlaps_ub[0])
        self.tik_inst.vector_dup(n_num_align, gt_max_overlaps_ub, gt_max_overlaps_scalar, 1, 1, 8)

        zero_tensor_ub = self._vector_dup_data(data_shape, self.data_type, "zero_tensor_ub", 0)
        one_tensor_ub = self._vector_dup_data(data_shape, self.data_type, "one_tensor_ub", 1)
        gt_max_overlaps_mask_ub = self.tik_inst.Tensor(self.flag_type, (self.flag_block_data_num,),
                                                       self.tik.scope_ubuf, "gt_max_overlaps_mask_ub")
        self.tik_inst.vcmpv_eq(gt_max_overlaps_mask_ub, overlaps_ub, gt_max_overlaps_ub, 1, 1, 1, 8, 8)
        self.tik_inst.vec_sel(n_num, 0, overlaps_ub, gt_max_overlaps_mask_ub,
                              one_tensor_ub, zero_tensor_ub, 1, 8, 8, 8)
        self.tik_inst.vmul(n_num, overlaps_ub, overlaps_ub, flag_ub, 1, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vcmpv_eq(gt_max_overlaps_mask_ub, overlaps_ub, one_tensor_ub, 1, 1, 1, 8, 8)

        index_int_ub = self.tik_inst.Tensor(self.int_type, data_shape, self.tik.scope_ubuf, "index_int_ub")
        int_repeat_num = ceil_div(n_num_align, self.int_repeat_data_num)
        self.tik_inst.vector_dup(self.int_repeat_data_num, index_int_ub, k_index + 1, int_repeat_num, 1, 8)
        index_fp_ub = self.tik_inst.Tensor(self.data_type, data_shape, self.tik.scope_ubuf, "index_fp_ub")
        self._vconv_data(n_num_align, 0, (index_int_ub, index_fp_ub))
        self.tik_inst.vec_sel(n_num, 0, assigned_gt_inds_ub, gt_max_overlaps_mask_ub, index_fp_ub, assigned_gt_inds_ub,
                              1, 8, 8, 8)

    def _mode_0_compute_each_core(self):
        num_gts_scalar = self._init_num_gts()
        self._mode_0_assigned_gt_inds_pos_iou()
        self._mode_0_assigned_positive(num_gts_scalar)

    def _mode_0_assigned_gt_inds_pos_iou(self):
        """
        algorithm:
            `pos_inds = (max_overlaps > pos_iou_thr & box_responsible_flags)`
            `assigned_gt_inds_pos[pos_inds] = argmax_overlaps[pos_inds] + 1`
        """
        each_loop_n_num = self.data_repeat_data_num
        loop_times, last_loop_n_num = self.get_loop_info(self.n_num, each_loop_n_num)
        with self.tik_inst.for_range(0, loop_times) as loop_index:
            n_index_start_loop = each_loop_n_num * loop_index
            with self.tik_inst.if_scope(loop_index != loop_times - 1):
                self._mode_0_assigned_gt_inds_pos_iou_each_loop(n_index_start_loop, each_loop_n_num)
            with self.tik_inst.else_scope():
                self._mode_0_assigned_gt_inds_pos_iou_each_loop(n_index_start_loop, last_loop_n_num)

    def _mode_0_assigned_gt_inds_pos_iou_each_loop(self, n_index_start, n_num):
        assigned_gt_inds_ub, _ = self._assigned_gt_inds_pos_iou_each_loop(n_index_start, n_num)
        data_block_num = ceil_div(n_num, self.data_block_data_num)
        self.tik_inst.data_move(self.assigned_gt_inds_pos[n_index_start], assigned_gt_inds_ub, 0, 1, data_block_num, 0,
                                0)

    def _mode_0_assigned_positive(self, num_gts_scalar):
        gt_min_pos_iou_ub = self._get_gt_min_pos_iou_ub()
        gt_max_overlaps_flag_scalar = self.tik_inst.Scalar(self.int_type)
        with self.tik_inst.for_range(0, num_gts_scalar) as k_index:
            gt_max_overlaps_flag_scalar.set_as(gt_min_pos_iou_ub[k_index])
            with self.tik_inst.if_scope(gt_max_overlaps_flag_scalar == 1):
                self._mode0_assigned_positive_each_k(k_index)

    def _mode0_assigned_positive_each_k(self, k_index):
        """
        algorithm:
            `if gt_max_overlaps[k_index] > min_pos_iou and box_responsible_flag:`
                `assigned_gt_inds_pos[gt_argmax_overlaps[i]] = i + 1`
        """
        gt_argmax_overlaps_scalar = self.tik_inst.Scalar(self.int_type)
        gt_argmax_overlaps_ub = self.tik_inst.Tensor(self.int_type, (self.int_block_data_num,),
                                                     self.tik.scope_ubuf, "gt_argmax_overlaps_ub")
        self.tik_inst.data_move(gt_argmax_overlaps_ub, self.gt_argmax_overlaps[k_index], 0, 1, 1, 0, 0)
        gt_argmax_overlaps_scalar.set_as(gt_argmax_overlaps_ub[0])

        fp16_type = "float16"
        fp16_block_data_num = self.cont.get_vec_proc_num_per_cmd_blk(fp16_type)
        box_responsible_flags_ub = self.tik_inst.Tensor(
            self.flag_type, (self.flag_block_data_num,), self.tik.scope_ubuf, "box_responsible_flags_ub")
        box_responsible_flags_float_ub = self.tik_inst.Tensor(
            fp16_type, (fp16_block_data_num,), self.tik.scope_ubuf, "box_responsible_flags_float_ub")
        box_responsible_flags_int_ub = self.tik_inst.Tensor(
            self.int_type, (self.int_block_data_num,), self.tik.scope_ubuf, "box_responsible_flags_int_ub")
        self.tik_inst.data_move(box_responsible_flags_ub, self.box_responsible_flags[gt_argmax_overlaps_scalar], 0, 1,
                                1,
                                0, 0)
        self._vconv_data(1, 3, (box_responsible_flags_ub, box_responsible_flags_float_ub, box_responsible_flags_int_ub))
        box_responsible_flags_scalar = self.tik_inst.Scalar(self.int_type)
        box_responsible_flags_scalar.set_as(box_responsible_flags_int_ub[0])
        with self.tik_inst.if_scope(box_responsible_flags_scalar == 1):
            assigned_gt_inds_ub = self.tik_inst.Tensor(self.data_type, (self.data_block_data_num,),
                                                       self.tik.scope_ubuf, "assigned_gt_inds_ub")
            self.tik_inst.data_move(assigned_gt_inds_ub, self.assigned_gt_inds_pos[gt_argmax_overlaps_scalar], 0, 1, 1,
                                    0,
                                    0)
            box_responsible_flags_int_ub[0].set_as(k_index + 1)
            self._vconv_data(1, 0, (box_responsible_flags_int_ub, assigned_gt_inds_ub))
            self.tik_inst.data_move(self.assigned_gt_inds_pos[gt_argmax_overlaps_scalar], assigned_gt_inds_ub, 0, 1, 1,
                                    0,
                                    0)

    def _get_gt_min_pos_iou_ub(self):
        """
        algorithm: if gt_max_ovelasps > min_pos_iou
        Returns
        -------
            gt_min_pos_iou_ub:
                `Tensor. int_type. shape(k_num, ). gt_min_pos_iou_ub = 1 if (gt_max_ovelasps > min_pos_iou) else 0`
        """
        k_num_align = self.get_align_num(self.k_num, self.data_repeat_data_num)
        gt_min_pos_iou_ub = self.tik_inst.Tensor(self.int_type, (k_num_align,),
                                                 self.tik.scope_ubuf, "gt_min_pos_iou_ub")
        each_loop_k_num = self.data_repeat_data_num
        loop_times, last_loop_k_num = self.get_loop_info(self.k_num, each_loop_k_num)
        with self.tik_inst.for_range(0, loop_times) as loop_index:
            k_index_start_loop = each_loop_k_num * loop_index
            with self.tik_inst.if_scope(loop_index != loop_times - 1):
                self._get_gt_min_pos_iou_ub_each_loop(gt_min_pos_iou_ub, k_index_start_loop, each_loop_k_num)
            with self.tik_inst.else_scope():
                self._get_gt_min_pos_iou_ub_each_loop(gt_min_pos_iou_ub, k_index_start_loop, last_loop_k_num)
        return gt_min_pos_iou_ub

    def _get_gt_min_pos_iou_ub_each_loop(self, gt_min_pos_iou_int_ub, k_index_start, k_num):
        """
        algorithm:
            `k_index_start: k_index_start+k_num`
            `gt_min_pos_iou_int_ub = 1 if (gt_max_ovelasps > min_pos_iou) else 0`
        """
        k_num_align = self.data_repeat_data_num
        data_shape = (k_num_align,)
        gt_min_pos_iou_ub = self.tik_inst.Tensor(self.data_type, data_shape,
                                                 self.tik.scope_ubuf, "gt_min_pos_iou_ub")
        gt_max_overlaps_ub = self.tik_inst.Tensor(self.data_type, data_shape,
                                                  self.tik.scope_ubuf, "gt_max_overlaps_ub")
        min_pos_iou_ub = self._vector_dup_data(data_shape, self.data_type, "min_pos_iou_ub", self.min_pos_iou)
        zero_tensor_ub = self._vector_dup_data(data_shape, self.data_type, "zero_tensor_ub", 0)
        one_tensor_ub = self._vector_dup_data(data_shape, self.data_type, "one_tensor_ub", 1)
        block_num = ceil_div(k_num, self.data_block_data_num)
        self.tik_inst.data_move(gt_max_overlaps_ub, self.gt_max_overlaps[k_index_start], 0, 1, block_num, 0, 0)

        min_pos_iou_mask_ub = self.tik_inst.Tensor(self.flag_type, (self.flag_block_data_num,),
                                                   self.tik.scope_ubuf, "min_pos_iou_mask_ub")
        self.tik_inst.vcmpv_gt(min_pos_iou_mask_ub, gt_max_overlaps_ub, min_pos_iou_ub, 1, 1, 1, 8, 8)
        self.tik_inst.vec_sel(k_num, 0, gt_min_pos_iou_ub, min_pos_iou_mask_ub,
                              one_tensor_ub, zero_tensor_ub, 1, 8, 8, 8)
        self._vconv_data(k_num, 1, (gt_min_pos_iou_ub, gt_min_pos_iou_int_ub), (0, k_index_start))

    def _init_num_gts(self):
        """
        algorithm:
            num_gts: Tensor(int) -> Scalar(int)
        """
        num_gts_scalar = self.tik_inst.Scalar(self.int_type)
        with self.tik_inst.new_stmt_scope():
            num_gts_ub = self.tik_inst.Tensor(self.int_type, (self.int_block_data_num,),
                                              self.tik.scope_ubuf, "num_gts_ub")
            self.tik_inst.data_move(num_gts_ub, self.num_gts, 0, 1, 1, 0, 0)
            num_gts_scalar.set_as(num_gts_ub[0])
            with self.tik_inst.if_scope(num_gts_scalar > self.k_num):
                num_gts_scalar.set_as(self.k_num)
        return num_gts_scalar

    def _assigned_gt_inds_pos_iou_each_loop(self, n_index_start, n_num):
        """
        algorithm:
            `pos_inds = (max_overlaps > pos_iou_thr & box_responsible_flags)`
            `assigned_gt_inds_ub_new[pos_inds] = argmax_overlaps[pos_inds] + 1`
            `assigned_gt_inds_ub_new = assigned_gt_inds_ub_new.astype(data_type)`
            `flag_ub = box_responsible_flags.astype(data_type)`
        """
        n_num_align = self.data_repeat_data_num
        data_shape = (n_num_align,)
        assigned_gt_inds_ub_new = self.tik_inst.Tensor(self.data_type, data_shape,
                                                       self.tik.scope_ubuf, "assigned_gt_inds_ub_new")
        flag_ub = self.tik_inst.Tensor(self.data_type, data_shape, self.tik.scope_ubuf, "flag_ub")
        with self.tik_inst.new_stmt_scope():
            zero_tensor_ub = self._vector_dup_data(data_shape, self.data_type, "zero_tensor_ub", 0)
            one_tensor_ub = self._vector_dup_data(data_shape, self.data_type, "one_tensor_ub", 1)
            self._get_flag_ub(flag_ub, n_index_start, n_num, zero_tensor_ub, one_tensor_ub)
            pos_iou_thr_ub = self.tik_inst.Tensor(self.data_type, (n_num_align,),
                                                  self.tik.scope_ubuf, "pos_iou_thr_ub")
            self._get_pos_iou_thr_ub(pos_iou_thr_ub, n_index_start, n_num, zero_tensor_ub, one_tensor_ub)
            assigned_gt_inds_ub, argmax_overlaps_ub = self._init_inds_argmax(n_index_start, n_num, n_num_align)

            mask = self.data_repeat_data_num
            repeat_num = 1
            self.tik_inst.vmul(mask, pos_iou_thr_ub, flag_ub, pos_iou_thr_ub, repeat_num, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vadds(mask, argmax_overlaps_ub, argmax_overlaps_ub, 1, repeat_num, 1, 1, 8, 8)
            pos_inds_mask_ub = self.tik_inst.Tensor(self.flag_type, (self.flag_block_data_num,),
                                                    self.tik.scope_ubuf, "pos_inds_mask_ub")
            self.tik_inst.vcmpv_gt(pos_inds_mask_ub, pos_iou_thr_ub, zero_tensor_ub, repeat_num, 1, 1, 8, 8)
            self.tik_inst.vec_sel(mask, 0, assigned_gt_inds_ub_new, pos_inds_mask_ub,
                                  argmax_overlaps_ub, assigned_gt_inds_ub, repeat_num, 8, 8, 8)
        return assigned_gt_inds_ub_new, flag_ub

    # 'pylint: disable=too-many-arguments
    def _get_flag_ub(self, flag_ub, n_index_start, n_num, zero_tensor_ub, one_tensor_ub):
        """
        algorithm:
            box_responsible_flags(uint8) -> box_responsible_flags(float16)
            flag_ub = 1 if box_responsible_flags != 0 else 0
        """
        fp16_type = "float16"
        with self.tik_inst.new_stmt_scope():
            fp16_num = self.cont.get_vec_proc_num_per_cmd(fp16_type)
            data_shape = (fp16_num,)
            box_responsible_flags_ub = self.tik_inst.Tensor(
                self.flag_type, data_shape, self.tik.scope_ubuf, "box_responsible_flags_ub")
            box_responsible_flags_fp16_ub = self.tik_inst.Tensor(
                fp16_type, data_shape, self.tik.scope_ubuf, "box_responsible_flags_fp16_ub")
            flag_false_ub = self._vector_dup_data(data_shape, fp16_type, "flag_false_ub", 0)

            block_num = ceil_div(n_num, self.flag_block_data_num)
            self.tik_inst.data_move(box_responsible_flags_ub, self.box_responsible_flags[n_index_start],
                                    0, 1, block_num, 0, 0)
            self._vconv_data(n_num, 2, (box_responsible_flags_ub, box_responsible_flags_fp16_ub))

            flag_mask_ub = self.tik_inst.Tensor(self.flag_type, (self.flag_block_data_num,),
                                                self.tik.scope_ubuf, "flag_mask_ub")
            self.tik_inst.vcmpv_ne(flag_mask_ub, box_responsible_flags_fp16_ub, flag_false_ub, 1, 1, 1, 8, 8)
            self.tik_inst.vec_sel(n_num, 0, flag_ub, flag_mask_ub, one_tensor_ub, zero_tensor_ub, 1, 8, 8, 8)

    # 'pylint: disable=too-many-arguments
    def _get_pos_iou_thr_ub(self, gt_pos_iou_thr_ub, n_index_start, n_num, zero_tensor_ub, one_tensor_ub):
        """
        algorithm:
            `gt_pos_iou_thr_ub = 1 if (max_overlaps > pos_iou_thr) else 0`
        """
        data_shape = gt_pos_iou_thr_ub.shape
        with self.tik_inst.new_stmt_scope():
            max_overlaps_ub = self.tik_inst.Tensor(self.data_type, data_shape,
                                                   self.tik.scope_ubuf, "max_overlaps_ub")
            pos_iou_thr_ub = self._vector_dup_data(data_shape, self.data_type, "pos_iou_thr_ub", self.pos_iou_thr)
            block_num = ceil_div(n_num, self.data_block_data_num)
            self.tik_inst.data_move(max_overlaps_ub, self.max_overlaps[n_index_start], 0, 1, block_num, 0, 0)

            pos_iou_thr_mask_ub = self.tik_inst.Tensor(self.flag_type, (self.flag_block_data_num,),
                                                       self.tik.scope_ubuf, "pos_iou_thr_mask_ub")
            self.tik_inst.vcmpv_gt(pos_iou_thr_mask_ub, max_overlaps_ub, pos_iou_thr_ub, 1, 1, 1, 8, 8)
            self.tik_inst.vec_sel(n_num, 0, gt_pos_iou_thr_ub, pos_iou_thr_mask_ub,
                                  one_tensor_ub, zero_tensor_ub, 1, 8, 8, 8)

    def _init_inds_argmax(self, n_index_start, n_num, n_num_align):
        """
        algorithm:
            assigned_gt_inds_ub move in. argmax_overlaps_ub move in, vconv data_type.
        """
        data_shape = (n_num_align,)
        assigned_gt_inds_ub = self.tik_inst.Tensor(self.data_type, data_shape,
                                                   self.tik.scope_ubuf, "assigned_gt_inds_ub")
        argmax_overlaps_ub = self.tik_inst.Tensor(self.data_type, data_shape,
                                                  self.tik.scope_ubuf, "argmax_overlaps_ub")
        data_block_num = ceil_div(n_num, self.data_block_data_num)
        int_block_num = ceil_div(n_num, self.int_block_data_num)
        self.tik_inst.data_move(assigned_gt_inds_ub, self.assigned_gt_inds[n_index_start], 0, 1, data_block_num, 0, 0)
        with self.tik_inst.new_stmt_scope():
            argmax_overlaps_int_ub = self.tik_inst.Tensor(self.int_type, data_shape,
                                                          self.tik.scope_ubuf, "argmax_overlaps_int_ub")
            self.tik_inst.data_move(argmax_overlaps_int_ub, self.argmax_overlaps[n_index_start], 0, 1, int_block_num, 0,
                                    0)
            self._vconv_data(n_num_align, 0, (argmax_overlaps_int_ub, argmax_overlaps_ub))
        return assigned_gt_inds_ub, argmax_overlaps_ub

    def _vector_dup_data(self, data_shape, data_type, name, data_val):
        tensor_ub = self.tik_inst.Tensor(data_type, data_shape, self.tik.scope_ubuf, name)
        self.tik_inst.vector_dup(data_shape[0], tensor_ub, data_val, 1, 1, 8)
        return tensor_ub

    # 'pylint: disable=too-many-locals
    def _vconv_data(self, data_num, mode, tensor_all, offset_all=None):
        if offset_all is None:
            offset_all = (0,) * len(tensor_all)
        buf_all = {}
        drive_buf_index_all = {0: 0, 1: 1, 2: 1, 3: 2}
        drive_buf_index = drive_buf_index_all[mode]
        num_per_cmd = self.cont.get_vec_proc_num_per_cmd(tensor_all[drive_buf_index].dtype)
        for index_tensor, (tensor, tensor_offset) in enumerate(zip(tensor_all, offset_all)):
            buf_all["tensor_{}".format(index_tensor)] = TensorOperatorParam(tensor, data_num, tensor_offset,
                                                                            num_per_cmd=num_per_cmd)
        if self.data_type == "float16":
            deqscale = 1.0
        else:
            deqscale = None
        cmd_vconv_all = {
            0: [VecCmd(cmd_name="vconv", dst_name="tensor_1", src0_name="tensor_0", round_mode="", deqscale=deqscale)],
            1: [VecCmd(cmd_name="vconv", dst_name="tensor_1", src0_name="tensor_0", round_mode="round")],
            2: [VecCmd(cmd_name="vconv", dst_name="tensor_1", src0_name="tensor_0", round_mode="")],
            3: [VecCmd(cmd_name="vconv", dst_name="tensor_1", src0_name="tensor_0", round_mode=""),
                VecCmd(cmd_name="vconv", dst_name="tensor_2", src0_name="tensor_1", round_mode="round")]
        }
        drive_buf_name = "tensor_{}".format(drive_buf_index)
        cmd_vconv = cmd_vconv_all[mode]
        VecExecutor.exec_vec_cmd(buf_all, cmd_vconv, drive_buf_name)


# 'pylint: disable=too-many-arguments,too-many-locals
def check_params(assigned_gt_inds, overlaps, box_responsible_flags,
                 max_overlaps, argmax_overlaps,
                 gt_max_overlaps, gt_argmax_overlaps,
                 num_gts, assigned_gt_inds_pos, kernel_name):
    """
    check params
    """
    input_shape = overlaps.get("shape")
    input_dtype = overlaps.get("dtype").lower()
    if len(input_shape) != 2:
        error_manager_vector.raise_err_input_value_invalid(
            kernel_name, "len of overlaps shape", "equal 2", str(len(input_shape)))
    if input_dtype not in ("float16", "float32"):
        error_manager_vector.raise_err_input_value_invalid(
            kernel_name, "dtype of overlaps", "float32 or float16", input_dtype)
    k_num, n_num = input_shape
    if k_num >= 256:
        error_manager_vector.raise_err_input_value_invalid(
            kernel_name, "overlaps.shape[0]", "less than or equal to 256", "greater than 256")
    shape_0 = (n_num, )
    shape_1 = (k_num, )
    shape_2 = (1, )
    int_type = "int32"
    flat_type = "uint8"
    param_list = (assigned_gt_inds, box_responsible_flags, max_overlaps, argmax_overlaps,
                  gt_max_overlaps, gt_argmax_overlaps, num_gts, assigned_gt_inds_pos)
    shape_list = (shape_0, shape_0, shape_0, shape_0, shape_1, shape_1, shape_2, shape_0)
    dtype_list = (input_dtype, flat_type, input_dtype, int_type, input_dtype, int_type, int_type, input_dtype)
    param_name_list = ("assigned_gt_inds", "box_responsible_flags", "max_overlaps", "argmax_overlaps",
                       "gt_max_overlaps", "gt_argmax_overlaps", "num_gts", "assigned_gt_inds_pos")
    for param, dtype, shape, param_name in zip(param_list, dtype_list, shape_list, param_name_list):
        param_shape = param.get("shape")
        param_dtype = param.get("dtype").lower()
        if param_shape != shape:
            error_manager_vector.raise_err_input_value_invalid(
                kernel_name, "shape of {}".format(param_name), str(shape), str(param_shape))
        if param_dtype != dtype:
            error_manager_vector.raise_err_input_value_invalid(
                kernel_name, "dtype of {}".format(param_name), dtype, param_dtype)


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_FLOAT, para_check.REQUIRED_ATTR_FLOAT,
                            para_check.REQUIRED_ATTR_BOOL, para_check.KERNEL_NAME)
# 'pylint: disable=unused-argument,too-many-arguments,too-many-locals
def grid_assign_positive(assigned_gt_inds, overlaps, box_responsible_flags,
                         max_overlaps, argmax_overlaps,
                         gt_max_overlaps, gt_argmax_overlaps,
                         num_gts, assigned_gt_inds_pos,
                         pos_iou_thr, min_pos_iou, gt_max_assign_all,
                         kernel_name="GridAssignPositive"):
    """
    algorithm: assign positive bboxes
    Parameters
    ----------
    assigned_gt_inds:
        A Tensor. Support float16/float32. shape (n, )
    overlaps :
        A Tensor. Datatype is same as assigned_gt_inds. IOU between gt_bboxes and bboxes. shape(k, n)
    box_responsible_flags:
        A Tensor. Support uint8. Flag to indicate whether box is responsible.
    max_overlaps :
        A Tensor. Datatype is same as assigned_gt_inds. overlaps.max(axis=0).
    argmax_overlaps :
        A Tensor. Support int32. overlaps.argmax(axis=0).
    gt_max_overlaps :
        A Tensor. Datatype is same as assigned_gt_inds. overlaps.max(axis=1).
    gt_argmax_overlaps :
        A Tensor. Support int32. overlaps.argmax(axis=1).
    num_gts :
        A Tensor. Support int32. real k. shape (1, )
    assigned_gt_inds_pos :
        A Tensor. Support float16/float32. shape (n, ).
    pos_iou_thr:
        float. IOU threshold for positive bboxes.
    min_pos_iou:
        float. minimum iou for a bbox to be considered as a positive bbox
    gt_max_assign_all:
        bool. whether to assign all bboxes with the same highest overlap with some gt to that gt.
    kernel_name : str
        cce kernel name, default value is top_k_3
    Returns
    -------
    None
    """
    check_params(assigned_gt_inds, overlaps, box_responsible_flags,
                 max_overlaps, argmax_overlaps,
                 gt_max_overlaps, gt_argmax_overlaps,
                 num_gts, assigned_gt_inds_pos, kernel_name)
    AContainer.reset_instance()
    cont = AContainer.get_instance()
    k_num, n_num = overlaps.get("shape")
    data_type = assigned_gt_inds.get("dtype").lower()
    flag_type = box_responsible_flags.get("dtype").lower()
    int_type = argmax_overlaps.get("dtype").lower()
    obj = GridAssignPositive(k_num, n_num, pos_iou_thr, min_pos_iou, gt_max_assign_all,
                             data_type, flag_type, int_type, kernel_name, cont)
    obj.mode_compute()
