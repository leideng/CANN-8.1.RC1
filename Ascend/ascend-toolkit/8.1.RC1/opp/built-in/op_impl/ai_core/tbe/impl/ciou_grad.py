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
ciou_grad
"""

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.common_iou_grad import Constant
from impl.diou_grad import DIoUGrad


class CIoUGrad(DIoUGrad):
    """CIoUGrad"""

    # 'pylint: disable=too-many-statements,too-many-arguments
    def __init__(self, dy, bboxes, trans, kernel_name):
        """__init__"""
        super().__init__(dy, bboxes, trans, kernel_name)
        # func: apply only for the ciougrad
        self.atan_sub_ub = None
        self.alpha_ub = None
        self.v_ub = None
        self.dv_ub = None
        self.rate_b1 = None
        self.delta_b1 = None
        self.rate_b2 = None
        self.delta_b2 = None

        # func: apply for the input/output tensors
        self.atan_sub = self.tik_instance.Tensor(self.dtype, [self.all_num], name="atan_sub", scope=tik.scope_gm)

    def compute(self):
        """ciou_grad_compute"""
        with self.tik_instance.for_range(0, self.used_aicore_num, block_num=self.used_aicore_num) as i:
            with self.tik_instance.for_range(0, self.batch_num_per_aicore) as j:
                self.compute_core(i + j * self.used_aicore_num)
            with self.tik_instance.if_scope(i < self.batch_tail):
                self.compute_core(self.batch_num_per_aicore * self.used_aicore_num + i)

        if self.move_flag:
            self.move_out()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.dy, self.bboxes, self.gtboxes, self.atan_sub],
                                   outputs=[self.dbboxes, self.dgtboxes])

        return self.tik_instance

    def compute_core(self, task_idx):
        """ciou_grad_compute_compute_core"""
        # func: init all unit
        self.init_ciou_date()
        # func: get b1 and b2
        self.move_in(task_idx)
        # func: compute for ciou
        self.update_ciou_forward(task_idx)
        # func: compute for ciou common grad
        self.update_ciou_backward()
        # func: compute for dbboxes/dgtboxes in inter
        self.inter_part()
        # func: compute for dbboxes/dgtboxes in union
        self.union_part()
        # func: compute for dbboxes/dgtboxes in rho2
        self.rho2_part()
        # func: compute for dbboxes/dgtboxes in c2
        self.c2_part()
        # func: compute for dbboxes/dgtboxes in v
        self.v_part()
        # func: resite res for attr_trans
        self.update_dboxes(task_idx)

    def init_ciou_date(self):
        """init_ciou_date"""
        self.init_date()
        self.cw = self.tik_instance.Tensor("float32", [self.data_align], name="cw", scope=tik.scope_ubuf)
        self.ch = self.tik_instance.Tensor("float32", [self.data_align], name="ch", scope=tik.scope_ubuf)
        self.rho_x = self.tik_instance.Tensor("float32", [self.data_align], name="rho_x", scope=tik.scope_ubuf)
        self.rho_y = self.tik_instance.Tensor("float32", [self.data_align], name="rho_y", scope=tik.scope_ubuf)
        self.rho2 = self.tik_instance.Tensor("float32", [self.data_align], name="rho2", scope=tik.scope_ubuf)
        self.c2 = self.tik_instance.Tensor("float32", [self.data_align], name="c2", scope=tik.scope_ubuf)
        self.drho2 = self.tik_instance.Tensor("float32", [self.data_align], name="drho2", scope=tik.scope_ubuf)
        self.dc2 = self.tik_instance.Tensor("float32", [self.data_align], name="dc2", scope=tik.scope_ubuf)

        self.rate_b1 = self.tik_instance.Tensor("float32", [self.data_align], name="rate_b1", scope=tik.scope_ubuf)
        self.delta_b1 = self.tik_instance.Tensor("float32", [self.data_align], name="delta_b1",
                                                 scope=tik.scope_ubuf)
        self.rate_b2 = self.tik_instance.Tensor("float32", [self.data_align], name="rate_b2", scope=tik.scope_ubuf)
        self.delta_b2 = self.tik_instance.Tensor("float32", [self.data_align], name="delta_b2",
                                                 scope=tik.scope_ubuf)

        self.alpha_ub = self.tik_instance.Tensor("float32", [self.data_align], name="alpha_ub",
                                                 scope=tik.scope_ubuf)
        self.atan_sub_ub = self.tik_instance.Tensor("float32", [self.data_align], name="atan_sub_ub",
                                                    scope=tik.scope_ubuf)
        self.v_ub = self.tik_instance.Tensor("float32", [self.data_align], name="v_ub", scope=tik.scope_ubuf)
        self.dv_ub = self.tik_instance.Tensor("float32", [self.data_align], name="dv_ub", scope=tik.scope_ubuf)

    def update_ciou_forward(self, task_idx):
        """update_ciou_forward"""
        # func: compute for  (1.0 + (w1 / h1) ** 2)
        self.tik_instance.h_div(self.rate_b1, self.b_box.w, self.b_box.h)
        self.tik_instance.h_mul(self.delta_b1, self.rate_b1, self.rate_b1)
        self.tik_instance.h_add(self.delta_b1, self.delta_b1, 1.0)

        self.tik_instance.h_div(self.rate_b2, self.g_box.w, self.g_box.h)
        self.tik_instance.h_mul(self.delta_b2, self.rate_b2, self.rate_b2)
        self.tik_instance.h_add(self.delta_b2, self.delta_b2, 1.0)
        # func: compute for inter/union/cw/ch
        self.update_forward_part()

        # func: `for c2 = cw**2 + ch**2`
        self.tik_instance.h_mul(self.tmp_a, self.cw, self.cw)
        self.tik_instance.h_mul(self.tmp_b, self.ch, self.ch)
        self.tik_instance.h_add(self.c2, self.tmp_a, self.tmp_b)

        # func: `for rho2 = (b2x - b1x)**2 + (b2y - b1y)**2`
        self.tik_instance.h_sub(self.rho_x, self.g_box.x, self.b_box.x)
        self.tik_instance.h_mul(self.tmp_a, self.rho_x, self.rho_x)
        self.tik_instance.h_sub(self.rho_y, self.g_box.y, self.b_box.y)
        self.tik_instance.h_mul(self.tmp_b, self.rho_y, self.rho_y)
        self.tik_instance.h_add(self.rho2, self.tmp_a, self.tmp_b)

        # func: `for v = 4 / pi ** 2 * atan_sub**2 & 4 / pi ** 2 = 0.405285`
        self.tik_instance.data_move(self.atan_sub_ub, self.atan_sub[task_idx * self.data_align], 0, 1,
                                    self.mov_rep_time, 0, 0)

        self.tik_instance.h_mul(self.tmp_a, self.atan_sub_ub, self.atan_sub_ub)
        self.tik_instance.h_mul(self.v_ub, self.tmp_a, 0.405285)

        # func: `for alpha = v / (1 + v - iou)`
        self.tik_instance.h_add(self.tmp_a, self.v_ub, 1.0)
        self.tik_instance.h_add(self.tmp_a, self.tmp_a, Constant.EPS)
        self.tik_instance.h_div(self.tmp_b, self.inter, self.area_union)

        self.tik_instance.h_sub(self.tmp_c, self.tmp_a, self.tmp_b)
        self.tik_instance.h_div(self.alpha_ub, self.v_ub, self.tmp_c)

    def update_ciou_backward(self):
        # func: compute for dinter/dunion/drho2/dc2
        self.update_backward_part()
        # func: compute for dv * 8 * atan_sub / math.pi ** 2 /(1.0 + (w1 / h1) ** 2)
        self.tik_instance.h_mul(self.tmp_a, self.dy_ub, self.alpha_ub)
        self.tik_instance.h_sub(self.dv_ub, self.tmp_zero, self.tmp_a)
        self.tik_instance.h_add(self.atan_sub_ub, self.atan_sub_ub, Constant.EPS)
        self.tik_instance.h_div(self.tmp_a, self.v_ub, self.atan_sub_ub)
        self.tik_instance.h_mul(self.tmp_b, self.tmp_a, 2.0)
        self.tik_instance.h_mul(self.tmp_c, self.tmp_b, self.dv_ub)
        self.tik_instance.h_div(self.delta_b1, self.tmp_c, self.delta_b1)
        self.tik_instance.h_div(self.delta_b2, self.tmp_c, self.delta_b2)

    def v_part(self):
        """v_part"""
        # for b1
        self.tik_instance.h_div(self.delta_b1, self.delta_b1, self.b_box.h)

        self.tik_instance.h_add(self.b_box.dx1, self.b_box.dx1, self.delta_b1)
        self.tik_instance.h_sub(self.b_box.dx2, self.b_box.dx2, self.delta_b1)

        self.tik_instance.h_mul(self.delta_b1, self.delta_b1, self.rate_b1)
        self.tik_instance.h_sub(self.b_box.dy1, self.b_box.dy1, self.delta_b1)

        self.tik_instance.h_add(self.b_box.dy2, self.b_box.dy2, self.delta_b1)
        # for b2
        self.tik_instance.h_div(self.delta_b2, self.delta_b2, self.g_box.h)

        self.tik_instance.h_sub(self.g_box.dx1, self.g_box.dx1, self.delta_b2)
        self.tik_instance.h_add(self.g_box.dx2, self.g_box.dx2, self.delta_b2)

        self.tik_instance.h_mul(self.delta_b2, self.delta_b2, self.rate_b2)

        self.tik_instance.h_add(self.g_box.dy1, self.g_box.dy1, self.delta_b2)
        self.tik_instance.h_sub(self.g_box.dy2, self.g_box.dy2, self.delta_b2)


# 'pylint: disable=invalid-name,too-many-locals,too-many-arguments,unused-argument
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_STR,
                            para_check.KERNEL_NAME)
def ciou_grad(dy, bboxes, gtboxes, atan_sub, dbboxes, dgtboxes, trans=False, is_cross=True, mode="iou",
              kernel_name="ciou_grad"):
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
    atan_sub : dict
        data of grad increment, shape must be [n].
        source data type, support "float32"

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
        kernel name, default value is "ciou_grad"
    Returns
    -------
    None
    """
    op_obj = CIoUGrad(dy, bboxes, trans, kernel_name)

    return op_obj.compute()
