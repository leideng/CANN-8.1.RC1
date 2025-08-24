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
diou
"""

from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from tbe.common.platform.platform_info import get_soc_spec
from impl.dynamic.common_iou import Constant
from impl.dynamic.common_iou import CommonIoU


def _apply_mem(tik_instance, dtype, shape, name, scope=tik.scope_ubuf):
    return tik_instance.Tensor(dtype, shape, name=name, scope=scope)


class DIoU(CommonIoU):
    """Function: use to finish Iou main functions
    """

    # pylint: disable=too-many-statements,too-many-arguments
    def __init__(self, bboxes, gtboxes, trans, is_cross, mode):
        super().__init__(bboxes, gtboxes, trans, is_cross, mode)

    def _run_segment(self, task_idx):
        """
        do a segment of bbox compute
        """
        self._apply_all_ub(self.data_align)

        if not self.trans:
            self.data_move_in(task_idx, self.data_align, self.mov_rep_time)
        else:
            self.data_move_in_and_trans(task_idx, self.mask, self.dup_rep_time, self.data_align, self.mov_rep_time)

        gtboxes_area_ub = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "gtboxes_area_ub")
        bboxes_area_ub = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "bboxes_area_ub")
        inter_area_ub = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "inter_area_ub")
        out_ub = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "out_ub")
        in_square = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "in_square")
        out_square = _apply_mem(self.tik_instance, self.dtype, [self.data_align], "out_square")

        # calcu bboxes area
        self.calcu_area(bboxes_area_ub)

        # calcu gtboxes area
        self.calcu_area(gtboxes_area_ub, gt_mode=True)

        # vmin vmax: get inter x0 x1 y0 y1, outer x0 x1 y0 y1
        self.get_inter_outer_area()

        # calcu inter area
        self.calcu_area(inter_area_ub, inter_mode=True)

        self.calcu_out_square(out_square)
        self.calcu_in_square(in_square)

        if self.mode == "iou":
            self.tik_instance.vadd(self.mask, out_ub, bboxes_area_ub,
                                   gtboxes_area_ub, self.dup_rep_time, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vsub(self.mask, out_ub, out_ub, inter_area_ub, 
                                   self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        elif self.mode == "iof":
            self.tik_instance.data_move(out_ub, gtboxes_area_ub, 0, 1, self.mov_rep_time, 0, 0)

        if self.product is True:
            self.tik_instance.vdiv(self.mask, out_ub, inter_area_ub, 
                                   out_ub, self.dup_rep_time, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vdiv(self.mask, out_square, in_square,
                                   out_square, self.dup_rep_time, 1, 1, 1, 8, 8, 8)
        else:
            # for mini
            self._rev_div(out_ub, inter_area_ub, out_ub)
            self._rev_div(out_square, in_square, out_square)

        self.tik_instance.vsub(self.mask, out_ub, out_ub, out_square, 
                               self.dup_rep_time, 1, 1, 1, 8, 8, 8)

        move_times = self.mov_rep_time
        with self.tik_instance.if_scope(task_idx == (self.task_num - 1)):
            data_align_tail = self.all_num - (self.task_num - 1) * self.data_align
            move_times = (data_align_tail + self.eliments_per_block - 1) // self.eliments_per_block
        self.tik_instance.data_move(self.overlap_gm[self.data_align * task_idx], 
                                    out_ub, 0, 1, move_times, 0, 0)


# pylint: disable=too-many-arguments
@register_operator("DIoU")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def diou(bboxes, gtboxes, overlap, trans=False, is_cross=True, mode="iou", kernel_name="diou"):
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
    kernel_name : str
        kernel name, default value is "diou"

    Returns
    -------
    None
    """
    
    diou_obj = DIoU(bboxes, gtboxes, trans, is_cross, mode)
    res = diou_obj.run_tik(kernel_name)

    return res
