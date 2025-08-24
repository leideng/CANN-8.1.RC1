#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2022 Huawei Technologies Co., Ltd
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
common_iou
"""

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_context
from tbe.common.platform.platform_info import get_soc_spec


# pylint: disable=too-few-public-methods,invalid-name,unused-variable
class Constant:
    """
    The class for constant.
    """
    MASK_BLOCK_32 = 64
    MASK_BLOCK_16 = 128
    BLOCK_32 = 8
    BLOCK_16 = 16
    BYTE_PER_DATA_32 = 4
    BYTE_PER_DATA_16 = 2
    CONST_PI_BY_FOUR = 0.78539816
    CONST_PI_BY_EIGHT = 0.39269908
    CONST_ITERTOR = 6
    CONST_INERTOR2 = 4
    TAN_PI_BY_EIGHT = 0.41421356
    NEG_TAN_PI_BY_EIGHT = -0.41421356
    TAYLOR = (1.0, -1.0 / 3, 1.0 / 5, -1.0 / 7, 1.0 / 9, -1.0 / 11, 1.0 / 13)
    MAX_INT32 = 2 ** 31 - 1
    TILING_SCALAR_DTYPE = "int64"
    TILING_PARAMS_NUM = 12


def _apply_mem(tik_instance, dtype, shape, name, scope=tik.scope_ubuf):
    """apply mem fuc

    Parameters
    ----------
    tik_instance: tik_instance
        tik_instance
    dtype: str
        ub dtype
    shape: list
        ub shape
    name: str
        ub name
    scope: scope
        scope_ubuf or scope_gm
    Returns
    -------
    Tensor: Tensor
    """
    return tik_instance.Tensor(dtype, shape, name=name, scope=scope)


class CommonIoU(object):
    """CommonIoU"""

    def __init__(self, bboxes, gtboxes, trans, is_cross, mode):
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.trans = trans
        self.is_cross = is_cross
        self.mode = mode.lower()
        self.dtype = bboxes.get("dtype").lower()
        self.product = tbe_platform.api_check_support("tik.vdiv", "float32")

        # func: for task allocation
        self.avail_aicore_num = get_soc_spec("CORE_NUM")
        self.available_ub_size = get_soc_spec("UB_SIZE")

        self.tiling_gm = self.tik_instance.Tensor(
            Constant.TILING_SCALAR_DTYPE,
            (Constant.TILING_PARAMS_NUM,),
            name="tiling_gm",
            scope=tik.scope_gm)
        self.bboxes_gm = self.tik_instance.Tensor(self.dtype, (Constant.MAX_INT32,),
                                                  name="bboxes_gm", scope=tik.scope_gm)
        self.gtboxes_gm = self.tik_instance.Tensor(self.dtype, (Constant.MAX_INT32,),
                                                   name="gtboxes_gm", scope=tik.scope_gm)
        self.overlap_gm = self.tik_instance.Tensor(
            self.dtype, (Constant.MAX_INT32,), 
            name="overlap_gm", 
            scope=tik.scope_gm)
        
        self.all_num_align = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "all_num_align")
        self.core_num_var = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "core_num_var")
        self.batch_num_per_aicore = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "batch_num_per_aicore")
        self.batch_tail = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "batch_tail")
        self.all_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "all_num")
        self.data_align = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "data_align")
        self.mov_rep_time = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "mov_rep_time")
        self.dup_rep_time = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "dup_rep_time")
        self.task_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "task_num")

        # init attr in objext
        self.bboxes_x0 = self.bboxes_x1 = self.bboxes_y0 = self.bboxes_y1 = None
        self.gtboxes_x0 = self.gtboxes_x1 = self.gtboxes_y0 = self.gtboxes_y1 = None
        self.inter_area_x0 = self.inter_area_x1 = self.inter_area_y0 = self.inter_area_y1 = None
        self.outer_area_x0 = self.outer_area_x1 = self.outer_area_y0 = self.outer_area_y1 = None
        
        self.mask = Constant.MASK_BLOCK_16 if self.dtype == "float16" else Constant.MASK_BLOCK_32
        self.eliments_per_block = Constant.BLOCK_16 if self.dtype == "float16" else Constant.BLOCK_32
        self.byte_per_data = Constant.BYTE_PER_DATA_16 if self.dtype == "float16" else Constant.BYTE_PER_DATA_32
    
     # pylint: disable=too-many-locals,too-many-branches,too-many-lines,too-many-statements,too-many-arguments
    def common_iou_process(self):
        """do process and scedule
           main function

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.core_num_var, block_num = self.core_num_var) as i:
            with self.tik_instance.for_range(0, self.batch_num_per_aicore) as j:
                self._run_segment(i + j * self.core_num_var)
            with self.tik_instance.if_scope(i < self.batch_tail):
                self._run_segment(self.batch_num_per_aicore * self.core_num_var + i)
    
    def get_tiling_data(self):
        tiling_ub = self.tik_instance.Tensor(
            Constant.TILING_SCALAR_DTYPE,
            (Constant.TILING_PARAMS_NUM,),
            tik.scope_ubuf,
            "tiling_ub"
        )
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, 3, 0, 0)
        self.all_num.set_as(tiling_ub[0])
        self.data_align.set_as(tiling_ub[1])
        self.mov_rep_time.set_as(tiling_ub[2])
        self.dup_rep_time.set_as(tiling_ub[3])
        self.task_num.set_as(tiling_ub[4])
        self.all_num_align.set_as(tiling_ub[5])
        self.core_num_var.set_as(tiling_ub[6])
        self.batch_num_per_aicore.set_as(tiling_ub[7])
        self.batch_tail.set_as(tiling_ub[8])

    def run_tik(self, kernel_name):
        self.get_tiling_data()
        self.common_iou_process()

        tbe_context.get_context().add_compile_info(
            "vars", {
                "full_core_num": self.avail_aicore_num,
                "ub_size": self.available_ub_size,
                "product": self.product
            }
        )

        self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=[self.bboxes_gm, self.gtboxes_gm],
                                   outputs=[self.overlap_gm],
                                   flowtable = [self.tiling_gm])

        return self.tik_instance
    
    def data_move_in(self, task_idx, data_align, mov_rep_time):
        with self.tik_instance.new_stmt_scope(disable_sync=True):
            self.tik_instance.data_move(self.gtboxes_x0, self.gtboxes_gm[task_idx * data_align], 
                                        0, 1, mov_rep_time, 0, 0)
            self.tik_instance.data_move(self.gtboxes_y0, self.gtboxes_gm[self.all_num + task_idx * data_align],
                                        0, 1, mov_rep_time, 0, 0)
            self.tik_instance.data_move(self.gtboxes_x1, self.gtboxes_gm[self.all_num * 2 + task_idx * data_align],
                                        0, 1, mov_rep_time, 0, 0)
            self.tik_instance.data_move(self.gtboxes_y1, self.gtboxes_gm[self.all_num * 3 + task_idx * data_align],
                                        0, 1, mov_rep_time, 0, 0)

            self.tik_instance.data_move(self.bboxes_x0, self.bboxes_gm[task_idx * data_align], 
                                        0, 1, mov_rep_time, 0, 0)
            self.tik_instance.data_move(self.bboxes_y0, self.bboxes_gm[self.all_num + task_idx * data_align],
                                        0, 1, mov_rep_time, 0, 0)
            self.tik_instance.data_move(self.bboxes_x1, self.bboxes_gm[self.all_num * 2 + task_idx * data_align],
                                        0, 1, mov_rep_time, 0, 0)
            self.tik_instance.data_move(self.bboxes_y1, self.bboxes_gm[self.all_num * 3 + task_idx * data_align],
                                        0, 1, mov_rep_time, 0, 0)
    
    def data_move_in_and_trans(self, task_idx, mask, dup_rep_time, data_align, mov_rep_time):
        boxes_xy = _apply_mem(self.tik_instance, self.dtype, [data_align], "boxes_xy")
        boxes_wh = _apply_mem(self.tik_instance, self.dtype, [data_align], "boxes_wh")

        self.tik_instance.data_move(boxes_xy, self.gtboxes_gm[task_idx * data_align], 
                                    0, 1, mov_rep_time, 0, 0)
        self.tik_instance.data_move(boxes_wh, self.gtboxes_gm[task_idx * data_align + self.all_num * 2], 
                                    0, 1, mov_rep_time, 0, 0)

        self.tik_instance.vmuls(mask, boxes_wh, boxes_wh, 0.5, dup_rep_time, 1, 1, 8, 8)
        self.tik_instance.vsub(mask, self.gtboxes_x0, boxes_xy, boxes_wh, dup_rep_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(mask, self.gtboxes_x1, boxes_xy, boxes_wh, dup_rep_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.data_move(boxes_xy, self.gtboxes_gm[task_idx * data_align + self.all_num], 
                                    0, 1, mov_rep_time, 0, 0)
        self.tik_instance.data_move(boxes_wh, self.gtboxes_gm[task_idx * data_align + self.all_num * 3], 
                                    0, 1, mov_rep_time, 0, 0)
        self.tik_instance.vmuls(mask, boxes_wh, boxes_wh, 0.5, dup_rep_time, 1, 1, 8, 8)
        self.tik_instance.vsub(mask, self.gtboxes_y0, boxes_xy, boxes_wh, dup_rep_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(mask, self.gtboxes_y1, boxes_xy, boxes_wh, dup_rep_time, 1, 1, 1, 8, 8, 8)

        self.tik_instance.data_move(boxes_xy, self.bboxes_gm[task_idx * data_align], 
                                    0, 1, mov_rep_time, 0, 0)
        self.tik_instance.data_move(boxes_wh, self.bboxes_gm[task_idx * data_align + self.all_num * 2], 
                                    0, 1, mov_rep_time, 0, 0)
        self.tik_instance.vmuls(mask, boxes_wh, boxes_wh, 0.5, dup_rep_time, 1, 1, 8, 8)
        self.tik_instance.vsub(mask, self.bboxes_x0, boxes_xy, boxes_wh, dup_rep_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(mask, self.bboxes_x1, boxes_xy, boxes_wh, dup_rep_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.data_move(boxes_xy, self.bboxes_gm[task_idx * data_align + self.all_num], 
                                    0, 1, mov_rep_time, 0, 0)
        self.tik_instance.data_move(boxes_wh, self.bboxes_gm[task_idx * data_align + self.all_num * 3], 
                                    0, 1, mov_rep_time, 0, 0)
        self.tik_instance.vmuls(mask, boxes_wh, boxes_wh, 0.5, dup_rep_time, 1, 1, 8, 8)
        self.tik_instance.vsub(mask, self.bboxes_y0, boxes_xy, boxes_wh, dup_rep_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(mask, self.bboxes_y1, boxes_xy, boxes_wh, dup_rep_time, 1, 1, 1, 8, 8, 8)
    
    def get_inter_outer_area(self):
        self.tik_instance.h_max(self.inter_area_x0, self.bboxes_x0, self.gtboxes_x0)
        self.tik_instance.h_max(self.inter_area_y0, self.bboxes_y0, self.gtboxes_y0)
        self.tik_instance.h_min(self.inter_area_x1, self.bboxes_x1, self.gtboxes_x1)
        self.tik_instance.h_min(self.inter_area_y1, self.bboxes_y1, self.gtboxes_y1)

        self.tik_instance.h_min(self.outer_area_x0, self.bboxes_x0, self.gtboxes_x0)
        self.tik_instance.h_min(self.outer_area_y0, self.bboxes_y0, self.gtboxes_y0)
        self.tik_instance.h_max(self.outer_area_x1, self.bboxes_x1, self.gtboxes_x1)
        self.tik_instance.h_max(self.outer_area_y1, self.bboxes_y1, self.gtboxes_y1)
    
    def calcu_area(self, area_ub, inter_mode=False, gt_mode=False):
        if inter_mode:
            x0_ub = self.inter_area_x0
            x1_ub = self.inter_area_x1
            y0_ub = self.inter_area_y0
            y1_ub = self.inter_area_y1
        elif gt_mode:
            x0_ub = self.gtboxes_x0
            x1_ub = self.gtboxes_x1
            y0_ub = self.gtboxes_y0
            y1_ub = self.gtboxes_y1
        else:
            x0_ub = self.bboxes_x0
            x1_ub = self.bboxes_x1
            y0_ub = self.bboxes_y0
            y1_ub = self.bboxes_y1
        # cala x1 - x0
        area_y1_y0 = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "area_y1_y0")
        self.tik_instance.vsub(self.mask, area_y1_y0, y1_ub, y0_ub, self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(self.mask, area_ub, x1_ub, x0_ub, self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        if inter_mode:
            zero_ub = _apply_mem(self.tik_instance, self.dtype, [self.eliments_per_block], "zero_ub")
            self.tik_instance.vector_dup(self.eliments_per_block, zero_ub, 0.0, 1, 1, 8)
            self.tik_instance.vmax(self.mask, area_ub, zero_ub, area_ub, self.dup_rep_time, 1, 0, 1, 8, 0, 8)
            self.tik_instance.vmax(self.mask, area_y1_y0, zero_ub, area_y1_y0, 
                                   self.dup_rep_time, 1, 0, 1, 8, 0, 8)
        else:
            self.tik_instance.vadds(self.mask, area_ub, area_ub, 1e-16, self.dup_rep_time, 1, 1, 8, 8)
            self.tik_instance.vadds(self.mask, area_y1_y0, area_y1_y0, 1e-16, self.dup_rep_time, 1, 1, 8, 8)

        self.tik_instance.vmul(self.mask, area_ub, area_y1_y0, area_ub, self.dup_rep_time, 1, 1, 1, 8, 8, 8)

    def calcu_out_square(self, out_square):
        area_y1_y0 = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "area_y1_y0")
        self.tik_instance.vsub(self.mask, out_square, self.outer_area_x1, self.outer_area_x0,
                               self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(self.mask, area_y1_y0, self.outer_area_y1, self.outer_area_y0,
                               self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(self.mask, out_square, out_square, out_square, 
                               self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(self.mask, area_y1_y0, area_y1_y0, area_y1_y0, 
                               self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(self.mask, out_square, out_square, area_y1_y0, 
                               self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadds(self.mask, out_square, out_square, 1e-16, self.dup_rep_time, 1, 1, 8, 8)
    
    def calcu_in_square(self, in_square):
        sum_y1_y0 = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "sum_y1_y0")

        self.tik_instance.vadd(self.mask, sum_y1_y0, self.bboxes_y0, self.bboxes_y1, 
                               self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(self.mask, sum_y1_y0, sum_y1_y0, self.gtboxes_y0, 
                               self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(self.mask, sum_y1_y0, sum_y1_y0, self.gtboxes_y1, 
                               self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(self.mask, sum_y1_y0, sum_y1_y0, sum_y1_y0, 
                               self.dup_rep_time, 1, 1, 1, 8, 8, 8)

        self.tik_instance.vadd(self.mask, in_square, self.bboxes_x0, self.bboxes_x1, 
                               self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(self.mask, in_square, in_square, self.gtboxes_x0, 
                               self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(self.mask, in_square, in_square, self.gtboxes_x1, 
                               self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(self.mask, in_square, in_square, in_square, 
                               self.dup_rep_time, 1, 1, 1, 8, 8, 8)

        self.tik_instance.vadd(self.mask, in_square, in_square, sum_y1_y0, 
                               self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmuls(self.mask, in_square, in_square, 0.25, 
                                self.dup_rep_time, 1, 1, 8, 8)
    
    def _rev_div(self, x1_ub, x2_ub, y_ub):
        div_rec_1 = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "div_rec_1")
        div_rec_2 = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "div_rec_2")

        self.tik_instance.vrec(self.mask, div_rec_1, x1_ub, self.dup_rep_time, 1, 1, 8, 8)
        self.tik_instance.vmul(self.mask, div_rec_2, div_rec_1, x1_ub, self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmuls(self.mask, div_rec_2, div_rec_2, -1, self.dup_rep_time, 1, 1, 8, 8)
        self.tik_instance.vadds(self.mask, div_rec_2, div_rec_2, 2, self.dup_rep_time, 1, 1, 8, 8)
        self.tik_instance.vmul(self.mask, div_rec_2, div_rec_2, div_rec_1, 
                               self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(self.mask, div_rec_1, div_rec_2, x1_ub, self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmuls(self.mask, div_rec_1, div_rec_1, -1, self.dup_rep_time, 1, 1, 8, 8)
        self.tik_instance.vadds(self.mask, div_rec_1, div_rec_1, 2, self.dup_rep_time, 1, 1, 8, 8)
        self.tik_instance.vmul(self.mask, div_rec_1, div_rec_1, div_rec_2, 
                               self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(self.mask, y_ub, div_rec_1, x2_ub, self.dup_rep_time, 1, 1, 1, 8, 8, 8)

    def _apply_all_ub(self, one_loop_shape):
        self.bboxes_x0 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "bboxes_x0")
        self.gtboxes_x0 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "gtboxes_x0")
        self.bboxes_x1 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "bboxes_x1")
        self.gtboxes_x1 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "gtboxes_x1")
        self.inter_area_x0 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "inter_area_x0")
        self.outer_area_x0 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "outer_area_x0")
        self.inter_area_x1 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "inter_area_x1")
        self.outer_area_x1 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "outer_area_x1")
        self.bboxes_y0 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "bboxes_y0")
        self.gtboxes_y0 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "gtboxes_y0")
        self.bboxes_y1 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "bboxes_y1")
        self.gtboxes_y1 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "gtboxes_y1")
        self.inter_area_y0 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "inter_area_y0")
        self.outer_area_y0 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "outer_area_y0")
        self.inter_area_y1 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "inter_area_y1")
        self.outer_area_y1 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "outer_area_y1") 
