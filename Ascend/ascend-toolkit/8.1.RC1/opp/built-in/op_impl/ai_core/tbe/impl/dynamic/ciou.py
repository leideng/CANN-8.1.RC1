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
ciou
"""

import math
from re import T
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
from impl.dynamic.common_iou import Constant
from impl.dynamic.common_iou import CommonIoU


def _apply_mem(tik_instance, dtype, shape, name, scope=tik.scope_ubuf):
    return tik_instance.Tensor(dtype, shape, name=name, scope=scope)


# pylint: disable=too-many-instance-attributes,too-many-lines,
class CIoU(CommonIoU):
    """Function: use to finish Iou main functions
    """

    # pylint: disable=too-many-statements,too-many-arguments,too-many-arguments
    def __init__(self, bboxes, gtboxes, trans, is_cross, mode, atan_sub_flag):
        super().__init__(bboxes, gtboxes, trans, is_cross, mode)

        self.atan_sub_gm = self.tik_instance.Tensor(
            self.dtype, (Constant.MAX_INT32,), 
            name="atan_sub_gm", 
            scope=tik.scope_gm)
    
    def run_tik(self, kernel_name):
        self.get_tiling_data()
        self.common_iou_process()

        tbe_context.get_context().add_compile_info(
            "vars", {
                "full_core_num": self.avail_aicore_num,
                "ub_size": self.available_ub_size
            }
        )

        self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=[self.bboxes_gm, self.gtboxes_gm],
                                   outputs=[self.overlap_gm, self.atan_sub_gm],
                                   flowtable = [self.tiling_gm])

        return self.tik_instance

    def calcu_area(self, area_ub, w_h_ub=None, inter_mode=False, gt_mode=False):
        if gt_mode:
            x0_ub = self.gtboxes_x0
            x1_ub = self.gtboxes_x1
            y0_ub = self.gtboxes_y0
            y1_ub = self.gtboxes_y1
        elif inter_mode:
            x0_ub = self.inter_area_x0
            x1_ub = self.inter_area_x1
            y0_ub = self.inter_area_y0
            y1_ub = self.inter_area_y1
        else:
            x0_ub = self.bboxes_x0
            x1_ub = self.bboxes_x1
            y0_ub = self.bboxes_y0
            y1_ub = self.bboxes_y1
        
        area_y1_y0 = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "area_y1_y0")
        # cala x1 - x0
        self.tik_instance.vsub(self.mask, area_ub, x1_ub, x0_ub, self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(self.mask, area_y1_y0, y1_ub, y0_ub, self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadds(self.mask, area_y1_y0, area_y1_y0, 1e-9, self.dup_rep_time, 1, 1, 8, 8)

        max_w_h_ub = _apply_mem(self.tik_instance, self.dtype, [self.eliments_per_block], "max_w_h_ub")
        self.tik_instance.vector_dup(self.eliments_per_block, max_w_h_ub, 10000.0, 1, 1, 8)
        zero_ub = _apply_mem(self.tik_instance, self.dtype, [self.eliments_per_block], "zero_ub")
        self.tik_instance.vector_dup(self.eliments_per_block, zero_ub, 0.0, 1, 1, 8)

        if inter_mode:
            self.tik_instance.vmax(self.mask, area_ub, zero_ub, area_ub, self.dup_rep_time, 1, 0, 1, 8, 0, 8)
            self.tik_instance.vmax(self.mask, area_y1_y0, zero_ub, area_y1_y0, 
                                   self.dup_rep_time, 1, 0, 1, 8, 0, 8)
        else:
            if self.product is True:
                self.tik_instance.vdiv(self.mask, w_h_ub, area_ub, area_y1_y0, 
                                       self.dup_rep_time, 1, 1, 1, 8, 8, 8)
            else:
                self._rev_div(area_y1_y0, area_ub, w_h_ub)
            self.tik_instance.vmin(self.mask, w_h_ub, max_w_h_ub, w_h_ub, self.dup_rep_time, 1, 0, 1, 8, 0, 8)
            self.tik_instance.vadds(self.mask, area_ub, area_ub, 1e-16, self.dup_rep_time, 1, 1, 8, 8)
            self.tik_instance.vadds(self.mask, area_y1_y0, area_y1_y0, 1e-16, self.dup_rep_time, 1, 1, 8, 8)

        self.tik_instance.vmul(self.mask, area_ub, area_y1_y0, area_ub, self.dup_rep_time, 1, 1, 1, 8, 8, 8)
    
    def calcu_atan(self, mask, repeat_time, x_ub, atan_ub, atan_quarter_pi_ub):
        x_add_one = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "x_add_one")
        x_sub_one = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "x_sub_one")

        self.tik_instance.vadds(mask, x_sub_one, x_ub, -1, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vadds(mask, x_add_one, x_ub, 1, repeat_time, 1, 1, 8, 8)
        if self.product is True:
            self.tik_instance.vdiv(mask, x_sub_one, x_sub_one, x_add_one, repeat_time, 1, 1, 1, 8, 8, 8)
        else:
            self._rev_div(x_add_one, x_sub_one, x_sub_one)
        self.tik_instance.vabs(mask, x_sub_one, x_sub_one, repeat_time, 1, 1, 8, 8)
        self.do_taylor(mask, repeat_time, x_ub, atan_ub)
        self.do_taylor(mask, repeat_time, x_sub_one, atan_quarter_pi_ub)
        self.tik_instance.vadds(mask, atan_quarter_pi_ub, atan_quarter_pi_ub, Constant.CONST_PI_BY_FOUR,
                                repeat_time, 1, 1, 8, 8)
        self.tik_instance.h_min(atan_ub, atan_ub, atan_quarter_pi_ub)
    
    def do_taylor(self, mask, repeat_time, data_x, atan_res_ub):
        atan_temp_w_h = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "atan_temp_w_h")
        denominator_data = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "denominator_data")
        square_x = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "square_x")

        self.tik_instance.vmul(mask, square_x, data_x, data_x, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmuls(mask, atan_res_ub, atan_res_ub, 0.0, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vadds(mask, atan_res_ub, atan_res_ub,
                                Constant.TAYLOR[Constant.CONST_INERTOR2], repeat_time, 1, 1, 8, 8)
        for i in reversed(range(Constant.CONST_INERTOR2)):
            self.tik_instance.vmul(mask, atan_res_ub, atan_res_ub, square_x,
                                   repeat_time, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vadds(mask, atan_res_ub, atan_res_ub, Constant.TAYLOR[i],
                                    repeat_time, 1, 1, 8, 8)
        self.tik_instance.vmul(mask, atan_res_ub, atan_res_ub, data_x,
                               repeat_time, 1, 1, 1, 8, 8, 8)

        self.tik_instance.vmuls(mask, denominator_data, data_x, Constant.TAN_PI_BY_EIGHT, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vadds(mask, denominator_data, denominator_data, 1.0, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vadds(mask, square_x, data_x, Constant.NEG_TAN_PI_BY_EIGHT,
                                repeat_time, 1, 1, 8, 8)
        if self.product is True:
            self.tik_instance.vdiv(mask, data_x, square_x, denominator_data, repeat_time, 1, 1, 1, 8, 8, 8)
        else:
            self._rev_div(denominator_data, square_x, data_x)
        self.tik_instance.vabs(mask, data_x, data_x, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vmul(mask, square_x, data_x, data_x, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmuls(mask, atan_temp_w_h, atan_temp_w_h, 0.0, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vadds(mask, atan_temp_w_h, atan_temp_w_h,
                                Constant.TAYLOR[Constant.CONST_ITERTOR], repeat_time, 1, 1, 8, 8)
        for i in reversed(range(Constant.CONST_ITERTOR)):
            self.tik_instance.vmul(mask, atan_temp_w_h, atan_temp_w_h, square_x,
                                   repeat_time, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vadds(mask, atan_temp_w_h, atan_temp_w_h, Constant.TAYLOR[i],
                                    repeat_time, 1, 1, 8, 8)
        self.tik_instance.vmul(mask, atan_temp_w_h, atan_temp_w_h, data_x,
                               repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadds(mask, atan_temp_w_h, atan_temp_w_h, Constant.CONST_PI_BY_EIGHT,
                                repeat_time, 1, 1, 8, 8)

        self.tik_instance.h_min(atan_res_ub, atan_res_ub, atan_temp_w_h)
        
        # pylint: disable=too-many-locals,too-many-branches,too-many-lines
    def _run_segment(self, task_idx):
        """
        do a segment of bbox compute

        Parameters
        ----------
        run_bb_point_segment : int
            bbox segment len
        gm_offset : int
            gm offset

        Returns
        -------
        None
        """

        self._apply_all_ub(self.data_align)

        # copy gm to ub
        if self.trans:
            self.data_move_in_and_trans(task_idx, self.mask, self.dup_rep_time, self.data_align, self.mov_rep_time)
        else:
            self.data_move_in(task_idx, self.data_align, self.mov_rep_time)

        bboxes_w_h = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "bboxes_w_h")
        gtboxes_w_h = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "gtboxes_w_h")
        inter_area_ub = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "inter_area_ub")
        gtboxes_area_ub = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "gtboxes_area_ub")
        bboxes_area_ub = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "bboxes_area_ub")
        v = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "v")
        gtboxes_atan_w_h_quarter_pi = _apply_mem(self.tik_instance, self.dtype, [self.data_align],
                                                      "gtboxes_atan_w_h_quarter_pi")
        atan_w_h_quarter_pi = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "atan_w_h_quarter_pi")
        gtboxes_atan_w_h = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "gtboxes_atan_w_h")
        atan_w_h = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "atan_w_h")
        in_square = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "in_square")
        out_square = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "out_square")

        # calcu bboxes area
        self.calcu_area(bboxes_area_ub, bboxes_w_h)

        # calcu gtboxes area
        self.calcu_area(gtboxes_area_ub, gtboxes_w_h, gt_mode=True)

        # vmin vmax: get inter x0 x1 y0 y1, outer x0 x1 y0 y1
        self.get_inter_outer_area()

        # calcu inter area
        self.calcu_area(inter_area_ub, inter_mode=True)

        self.calcu_out_square(out_square)
        self.calcu_in_square(in_square)

        # calcu atan
        self.calcu_atan(self.mask, self.dup_rep_time, bboxes_w_h, atan_w_h, atan_w_h_quarter_pi)
        self.calcu_atan(self.mask, self.dup_rep_time, gtboxes_w_h, gtboxes_atan_w_h,
                        gtboxes_atan_w_h_quarter_pi)
        self.tik_instance.vsub(self.mask, atan_w_h, gtboxes_atan_w_h, atan_w_h,
                               self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(self.mask, v, atan_w_h, atan_w_h, self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmuls(self.mask, v, v, 4 / math.pi ** 2, self.dup_rep_time, 1, 1, 8, 8)

        out_ub = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "out_ub")
        if self.mode == "iou":
            self.tik_instance.vadd(self.mask, out_ub, bboxes_area_ub,
                                   gtboxes_area_ub, self.dup_rep_time, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vsub(self.mask, out_ub, out_ub,
                                   inter_area_ub, self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        elif self.mode == "iof":
            self.tik_instance.data_move(out_ub, gtboxes_area_ub,
                                        0, 1, self.mov_rep_time, 0, 0)

        if self.product is True:
            self.tik_instance.vdiv(self.mask, out_ub, inter_area_ub, out_ub, 
                                   self.dup_rep_time, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vdiv(self.mask, out_square, in_square,
                                   out_square, self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        else:
            # for mini
            self._rev_div(out_ub, inter_area_ub, out_ub)
            self._rev_div(out_square, in_square, out_square)

        alpha_ub = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "alpha_ub")
        self.tik_instance.vsub(self.mask, alpha_ub, v, out_ub, self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadds(self.mask, alpha_ub, alpha_ub, 1 + 1e-16, self.dup_rep_time, 1, 1, 8, 8)
        if self.product is True:
            self.tik_instance.vdiv(self.mask, alpha_ub, v, alpha_ub, self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        else:
            self._rev_div(alpha_ub, v, alpha_ub)
        self.tik_instance.vmul(self.mask, alpha_ub, v, alpha_ub, self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(self.mask, out_ub, out_ub, out_square, 
                               self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(self.mask, out_ub, out_ub, alpha_ub, self.dup_rep_time, 1, 1, 1, 8, 8, 8)

        move_times = self.mov_rep_time
        with self.tik_instance.if_scope(task_idx == (self.task_num - 1)):
            data_align_tail = self.all_num - (self.task_num - 1) * self.data_align
            move_times = (data_align_tail + self.eliments_per_block - 1) // self.eliments_per_block
        self.tik_instance.data_move(self.overlap_gm[self.data_align * task_idx], out_ub, 
                                    0, 1, move_times, 0, 0)
        self.tik_instance.data_move(self.atan_sub_gm[self.data_align * task_idx], atan_w_h, 0, 1, 
                                    move_times, 0, 0)


# pylint: disable=too-many-arguments
@register_operator("CIoU")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def ciou(bboxes, gtboxes, overlap, atan_sub,
         trans=False, is_cross=True, mode="iou", atan_sub_flag=False, kernel_name="ciou"):
    """
    calculating data

    Parameters
    ----------
    bboxes : dict
        shape and dtype of bboxes, the coordinates of bbox
        shape must be [4, n]
    gtboxes : dict
        shape and dtype of gtboxes, the coordinates of bbox
        shape must be [4, m]
    overlap : dict
        shape and dtype of overlap
        result shape is [m, n] or [1, n]
    trans : bool
        transform from xywh to xyxy or not
    is_cross : bool
        if true: m must be equal to n, shape of overlap is [m, n]
        if false: shape of overlap is [1, n]
    mode : str
        ('iou','iof')
        iou : the output is inter_area / total_area
        iof : the output is inter_area / gtboxes_area
    atan_sub_flag : bool
        if true: Output atan_sub
        if false: Do not output atan_sub
    kernel_name : str
        kernel name, default value is "ciou"

    Returns
    -------
    None
    """
   
    ciou_obj = CIoU(bboxes, gtboxes, trans, is_cross, mode, atan_sub_flag)
    res = ciou_obj.run_tik(kernel_name)

    return res
