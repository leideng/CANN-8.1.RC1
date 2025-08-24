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
giou_grad
"""

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.common_iou_grad import Constant
from impl.common_iou_grad import CommonIoUGrad


# 'pylint: disable=too-many-statements,too-many-arguments
class GIoUGrad(CommonIoUGrad):
    """GIoUGrad"""

    # 'pylint: disable=too-many-statements,too-many-arguments
    def __init__(self, dy, bboxes, trans, kernel_name):
        """__init__"""
        super().__init__(dy, bboxes, trans, kernel_name)
        # func: apply only for the giougrad
        self.enclose = None
        self.denclose = None
        self.cw = None
        self.ch = None

    def compute_core(self, task_idx):
        """giou_grad_compute_compute_core"""
        # func: init all unit
        self.init_date()
        self.cw = self.tik_instance.Tensor("float32", [self.data_align], name="cw", scope=tik.scope_ubuf)
        self.ch = self.tik_instance.Tensor("float32", [self.data_align], name="ch", scope=tik.scope_ubuf)
        self.enclose = self.tik_instance.Tensor("float32", [self.data_align], name="enclose", scope=tik.scope_ubuf)
        self.denclose = self.tik_instance.Tensor("float32", [self.data_align], name="denclose", scope=tik.scope_ubuf)
        # func: get b1 and b2
        self.move_in(task_idx)

        # func: compute for inter/union
        self.update_forward_part()

        # func: compute for enclose
        self.tik_instance.h_mul(self.enclose, self.cw, self.ch)
        self.tik_instance.h_add(self.enclose, self.enclose, Constant.EPS)

        # func: compute for dinter/dunion/denclose
        self.update_backward_part()

        # func: compute for dbboxes/dgtboxes in inter
        self.inter_part()

        # func: compute for dbboxes/dgtboxes in union
        self.union_part()

        # func: compute for dbboxes/dgtboxes in enclose
        self.enclose_part()

        # func: resite res for attr_trans
        self.update_dboxes(task_idx)

    def update_backward_part(self):
        """update_backward_part"""
        # `for dunion, dunion = (1 / enclose - inter / (union ** 2)) * dy`
        self.tik_instance.h_div(self.tmp_a, self.dy_ub, self.enclose)
        self.tik_instance.h_div(self.tmp_b, self.inter, self.area_union)
        self.tik_instance.h_div(self.tmp_c, self.tmp_b, self.area_union)
        self.tik_instance.h_mul(self.tmp_d, self.dy_ub, self.tmp_c)
        self.tik_instance.h_sub(self.dunion, self.tmp_a, self.tmp_d)

        # `for dinter, dinter = 1 / union * dy - dunion`
        self.tik_instance.h_div(self.dinter, self.dy_ub, self.area_union)
        self.tik_instance.h_sub(self.dinter, self.dinter, self.dunion)

        # `for denclose, denclose = -(union / (enclose ** 2)) * dy`
        self.tik_instance.h_div(self.tmp_a, self.area_union, self.enclose)
        self.tik_instance.h_div(self.tmp_b, self.tmp_a, self.enclose)
        self.tik_instance.h_mul(self.tmp_c, self.dy_ub, self.tmp_b)
        self.tik_instance.h_sub(self.denclose, self.tmp_zero, self.tmp_c)

    def enclose_part(self):
        """enclose_part"""
        self.tik_instance.h_mul(self.dxlen, self.denclose, self.ch)
        self.tik_instance.h_mul(self.dylen, self.denclose, self.cw)

        with self.tik_instance.for_range(0, self.data_align // Constant.MASK_BLOCK) as idx:
            # `for enclose part : max(b1_x2, b2_x2)`
            self.tik_instance.vec_cmpv_gt(self.mask_1, self.b_box.x2[Constant.MASK_BLOCK * idx],
                                          self.g_box.x2[Constant.MASK_BLOCK * idx], 1,
                                          Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_a[Constant.MASK_BLOCK * idx], self.mask_1,
                                      self.dxlen[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_b[Constant.MASK_BLOCK * idx], self.mask_1,
                                      self.tmp_zero, self.dxlen[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)
            # `for enclose part : max(b1_y2, b2_y2)`
            self.tik_instance.vec_cmpv_gt(self.mask_2, self.b_box.y2[Constant.MASK_BLOCK * idx],
                                          self.g_box.y2[Constant.MASK_BLOCK * idx], 1,
                                          Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_c[Constant.MASK_BLOCK * idx], self.mask_2,
                                      self.dylen[Constant.MASK_BLOCK * idx], self.tmp_zero, 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_d[Constant.MASK_BLOCK * idx], self.mask_2,
                                      self.tmp_zero, self.dylen[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)

        self.tik_instance.h_add(self.b_box.dx2, self.tmp_a, self.b_box.dx2)
        self.tik_instance.h_add(self.g_box.dx2, self.tmp_b, self.g_box.dx2)
        self.tik_instance.h_add(self.b_box.dy2, self.tmp_c, self.b_box.dy2)
        self.tik_instance.h_add(self.g_box.dy2, self.tmp_d, self.g_box.dy2)

        self.tik_instance.h_sub(self.dxlen, self.tmp_zero, self.dxlen)
        self.tik_instance.h_sub(self.dylen, self.tmp_zero, self.dylen)

        with self.tik_instance.for_range(0, self.data_align // Constant.MASK_BLOCK) as idx:
            # `for enclose part : min(b1_x1, b2_x1)`
            self.tik_instance.vec_cmpv_lt(self.mask_1, self.b_box.x1[Constant.MASK_BLOCK * idx],
                                          self.g_box.x1[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                          Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_a[Constant.MASK_BLOCK * idx], self.mask_1,
                                      self.dxlen[Constant.MASK_BLOCK * idx], self.tmp_zero, 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_b[Constant.MASK_BLOCK * idx], self.mask_1,
                                      self.tmp_zero, self.dxlen[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)
            # `for enclose part : min(b1_y1, b2_y1)`
            self.tik_instance.vec_cmpv_lt(self.mask_2, self.b_box.y1[Constant.MASK_BLOCK * idx],
                                          self.g_box.y1[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                          Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_c[Constant.MASK_BLOCK * idx], self.mask_2,
                                      self.dylen[Constant.MASK_BLOCK * idx], self.tmp_zero, 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_d[Constant.MASK_BLOCK * idx], self.mask_2,
                                      self.tmp_zero, self.dylen[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)

        self.tik_instance.h_add(self.b_box.dx1, self.tmp_a, self.b_box.dx1)
        self.tik_instance.h_add(self.g_box.dx1, self.tmp_b, self.g_box.dx1)
        self.tik_instance.h_add(self.b_box.dy1, self.tmp_c, self.b_box.dy1)
        self.tik_instance.h_add(self.g_box.dy1, self.tmp_d, self.g_box.dy1)


# 'pylint: disable=invalid-name,too-many-locals,too-many-arguments,unused-argument
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def giou_grad(dy, bboxes, gtboxes, dbboxes, dgtboxes, trans=False, is_cross=True, mode="iou",
              kernel_name="giou_grad"):
    """
    calculating data

    Parameters
    ----------
    Inputs:
    dy : dict
        data of grad increment, shape must be [n].
        source data type, support "float32"
    bboxes : dict
        shape and dtype of bboxes, the coordinates of bbox
        shape must be [4, n]
        [x1, y1, x2, y2] or [x, y, w, h]
    gtboxes : dict
        shape and dtype of gtboxes, the coordinates of gtbox
        shape must be [4, m]
        [x1, y1, x2, y2] or [x, y, w, h]

    Outputs:
    dbboxes : dict
        shape and dtype of dbboxes, the coordinates of dbbox
        shape must be [4, n]
        [x1, y1, x2, y2] or [x, y, w, h]
    dgtboxes : dict
        shape and dtype of dgtboxes, the coordinates of dgtbox
        shape must be [4, m]
        [x1, y1, x2, y2] or [x, y, w, h]

    Attributes:
    trans : bool
        true for 'xywh', false for 'xyxy'
    is_cross : bool
        if false: m must be equal to n
    mode :  str
        ('iou','iof')
        iou : the output is inter_area / total_area
        iof : the output is inter_area / gtboxes_area
    kernel_name : str
        kernel name, default value is "giou_grad"
    Returns
    -------
    None
    """
    op_obj = GIoUGrad(dy, bboxes, trans, kernel_name)

    return op_obj.compute()
