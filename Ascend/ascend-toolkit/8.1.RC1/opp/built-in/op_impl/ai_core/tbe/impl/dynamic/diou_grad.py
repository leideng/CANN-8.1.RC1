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
diou_grad
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.dynamic.common_iou_grad import Constant
from impl.dynamic.common_iou_grad import CommonIoUGrad
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context


class DIoUGrad(CommonIoUGrad):
    """DIouGrad"""

    # 'pylint: disable=too-many-statements,too-many-arguments
    def __init__(self, dy, bboxes, trans, kernel_name):
        """__init__"""
        super().__init__(dy, bboxes, trans, kernel_name)

        # func: apply for the calculation cache of diou_grad
        self.rho_x = None
        self.rho_y = None
        self.rho2 = None
        self.cw = None
        self.ch = None
        self.c2 = None
        self.drho2 = None
        self.dc2 = None

    def compute_core(self, task_idx):
        """diou_grad_compute_compute_core"""
        # func: init all unit
        self.init_date()
        self.cw = self.tik_instance.Tensor("float32", [self.data_align], name="cw", scope=tik.scope_ubuf)
        self.ch = self.tik_instance.Tensor("float32", [self.data_align], name="ch", scope=tik.scope_ubuf)
        self.rho_x = self.tik_instance.Tensor("float32", [self.data_align], name="rho_x", scope=tik.scope_ubuf)
        self.rho_y = self.tik_instance.Tensor("float32", [self.data_align], name="rho_y", scope=tik.scope_ubuf)
        self.rho2 = self.tik_instance.Tensor("float32", [self.data_align], name="rho2", scope=tik.scope_ubuf)
        self.c2 = self.tik_instance.Tensor("float32", [self.data_align], name="c2", scope=tik.scope_ubuf)
        self.drho2 = self.tik_instance.Tensor("float32", [self.data_align], name="drho2", scope=tik.scope_ubuf)
        self.dc2 = self.tik_instance.Tensor("float32", [self.data_align], name="dc2", scope=tik.scope_ubuf)

        # func: get b1 and b2
        self.move_in(task_idx)

        # func: compute for inter/union/cw/ch
        self.update_forward_part()

        # func: `for rho2 = (b2x - b1x)**2 + (b2y - b1y)**2`
        self.tik_instance.h_sub(self.rho_x, self.g_box.x, self.b_box.x)
        self.tik_instance.h_mul(self.tmp_a, self.rho_x, self.rho_x)
        self.tik_instance.h_sub(self.rho_y, self.g_box.y, self.b_box.y)

        self.tik_instance.h_mul(self.tmp_b, self.rho_y, self.rho_y)
        self.tik_instance.h_add(self.rho2, self.tmp_a, self.tmp_b)

        # func: `for c2 = cw**2 + ch**2`
        self.tik_instance.h_mul(self.tmp_a, self.cw, self.cw)
        self.tik_instance.h_mul(self.tmp_b, self.ch, self.ch)
        self.tik_instance.h_add(self.c2, self.tmp_a, self.tmp_b)
        self.tik_instance.h_add(self.c2, self.c2, Constant.EPS)

        # func: compute for dinter/dunion/drho2/dc2
        self.update_backward_part()

        # func: compute for dbboxes/dgtboxes in inter
        self.inter_part()

        # func: compute for dbboxes/dgtboxes in union
        self.union_part()

        # func: compute for dbboxes/dgtboxes in rho2
        self.rho2_part()

        # func: compute for dbboxes/dgtboxes in c2
        self.c2_part()

        # func: resite res for attr_trans
        self.update_dboxes(task_idx)

    def update_backward_part(self):
        """update_dpart"""
        # `for dunion, dunion = (- inter / (union ** 2)) * dy`
        self.tik_instance.h_div(self.tmp_a, self.dy_ub, self.unionarea)
        self.tik_instance.h_div(self.tmp_b, self.tmp_a, self.unionarea)
        self.tik_instance.h_mul(self.tmp_c, self.inter, self.tmp_b)
        self.tik_instance.h_sub(self.dunion, self.tmp_zero, self.tmp_c)

        # `for dinter, dinter = 1 / union * dy - dunion`
        self.tik_instance.h_div(self.dinter, self.dy_ub, self.unionarea)
        self.tik_instance.h_sub(self.dinter, self.dinter, self.dunion)

        # `for rho2, drho2 = -(1 / c2)) * dy`
        self.tik_instance.h_div(self.tmp_a, self.dy_ub, self.c2)
        self.tik_instance.h_sub(self.drho2, self.tmp_zero, self.tmp_a)

        # `for c2, dc2 = rho2 / c2**2 * dy`
        self.tik_instance.h_div(self.tmp_b, self.tmp_a, self.c2)
        self.tik_instance.h_mul(self.dc2, self.tmp_b, self.rho2)

    def rho2_part(self):
        """rho2_part"""
        self.tik_instance.h_mul(self.tmp_a, self.rho_x, self.drho2)
        self.tik_instance.h_mul(self.tmp_b, self.rho_y, self.drho2)

        self.tik_instance.h_sub(self.b_box.dx2, self.b_box.dx2, self.tmp_a)
        self.tik_instance.h_sub(self.b_box.dx1, self.b_box.dx1, self.tmp_a)

        self.tik_instance.h_sub(self.b_box.dy2, self.b_box.dy2, self.tmp_b)
        self.tik_instance.h_sub(self.b_box.dy1, self.b_box.dy1, self.tmp_b)

        self.tik_instance.h_add(self.g_box.dx2, self.g_box.dx2, self.tmp_a)
        self.tik_instance.h_add(self.g_box.dx1, self.g_box.dx1, self.tmp_a)

        self.tik_instance.h_add(self.g_box.dy2, self.g_box.dy2, self.tmp_b)
        self.tik_instance.h_add(self.g_box.dy1, self.g_box.dy1, self.tmp_b)

    def c2_part(self):
        """c2_part"""
        tmp = self.tik_instance.Tensor("float32", [self.data_align], name="tmp", scope=tik.scope_ubuf)
        self.tik_instance.h_mul(tmp, self.cw, self.dc2)
        self.tik_instance.h_add(tmp, tmp, tmp)

        with self.tik_instance.for_range(0, self.data_align // Constant.MASK_BLOCK) as idx:
            # `for enclose part : max(b1_x2, b2_x2)`
            self.tik_instance.vec_cmpv_gt(self.mask_1, self.b_box.x2[Constant.MASK_BLOCK * idx],
                                          self.g_box.x2[Constant.MASK_BLOCK * idx], 1,
                                          Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_a[Constant.MASK_BLOCK * idx], self.mask_1,
                                      tmp[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_b[Constant.MASK_BLOCK * idx], self.mask_1,
                                      self.tmp_zero, tmp[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)

            # `for enclose part : min(b1_x1, b2_x1)`
            self.tik_instance.vec_cmpv_lt(self.mask_2, self.b_box.x1[Constant.MASK_BLOCK * idx],
                                          self.g_box.x1[Constant.MASK_BLOCK * idx], 1,
                                          Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_c[Constant.MASK_BLOCK * idx], self.mask_2,
                                      tmp[Constant.MASK_BLOCK * idx], self.tmp_zero, 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_d[Constant.MASK_BLOCK * idx], self.mask_2,
                                      self.tmp_zero, tmp[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)

        self.tik_instance.h_add(self.b_box.dx2, self.tmp_a, self.b_box.dx2)
        self.tik_instance.h_add(self.g_box.dx2, self.tmp_b, self.g_box.dx2)
        self.tik_instance.h_sub(self.b_box.dx1, self.b_box.dx1, self.tmp_c)
        self.tik_instance.h_sub(self.g_box.dx1, self.g_box.dx1, self.tmp_d)

        self.tik_instance.h_mul(tmp, self.ch, self.dc2)
        self.tik_instance.h_add(tmp, tmp, tmp)

        with self.tik_instance.for_range(0, self.data_align // Constant.MASK_BLOCK) as idx:
            # `for enclose part : max(b1_y2, b2_y2)`
            self.tik_instance.vec_cmpv_gt(self.mask_1, self.b_box.y2[Constant.MASK_BLOCK * idx],
                                          self.g_box.y2[Constant.MASK_BLOCK * idx], 1,
                                          Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_a[Constant.MASK_BLOCK * idx], self.mask_1,
                                      tmp[Constant.MASK_BLOCK * idx], self.tmp_zero, 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_b[Constant.MASK_BLOCK * idx], self.mask_1,
                                      self.tmp_zero, tmp[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)
            # `for enclose part : min(b1_y1, b2_y1)`
            self.tik_instance.vec_cmpv_lt(self.mask_2, self.b_box.y1[Constant.MASK_BLOCK * idx],
                                          self.g_box.y1[Constant.MASK_BLOCK * idx],
                                          1, Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_c[Constant.MASK_BLOCK * idx], self.mask_2,
                                      tmp[Constant.MASK_BLOCK * idx], self.tmp_zero, 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_d[Constant.MASK_BLOCK * idx], self.mask_2,
                                      self.tmp_zero, tmp[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)

        self.tik_instance.h_add(self.b_box.dy2, self.tmp_a, self.b_box.dy2)
        self.tik_instance.h_add(self.g_box.dy2, self.tmp_b, self.g_box.dy2)
        self.tik_instance.h_sub(self.b_box.dy1, self.b_box.dy1, self.tmp_c)
        self.tik_instance.h_sub(self.g_box.dy1, self.g_box.dy1, self.tmp_d)


# 'pylint: disable=invalid-name,too-many-locals,too-many-arguments,unused-argument
@register_operator("DIoUGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def diou_grad(dy, bboxes, gtboxes, dbboxes, dgtboxes, trans=False, is_cross=True, mode="iou",
              kernel_name="diou_grad"):
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
        kernel name, default value is "diou_grad"
    Returns
    -------
    None
    """
    op_obj = DIoUGrad(dy, bboxes, trans, kernel_name)

    return op_obj.compute()
