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
common_iou_grad
"""

from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from tbe.common.platform.platform_info import get_soc_spec


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant.
    """
    # for 32Byte align
    BLOCK = 8
    # vector mask in fp32
    MASK_BLOCK = 64
    # vector stride 256Byte
    REP_STRIDE = 8
    # box info class
    BOX_LOC = 4
    # half for div 2
    HALF = 0.5
    # eps to avoid division by zero
    EPS = 1e-9
    # ub object num for ciougrad
    UB_NUM = 60
    # byte per object
    BYTE_PER_DATA = 4
    MAX_INT32 = 2 ** 31 - 1
    TILING_SCALAR_DTYPE = "int64"
    TILING_PARAMS_NUM = 12


class Boxes(object):
    """Boxes"""

    # 'pylint: disable=too-many-statements,too-many-arguments
    def __init__(self, tik_instance, data_align, dup_rep_time):
        """__init__"""
        self.x = None
        self.y = None
        self.w = None
        self.h = None

        self.x1 = None
        self.y1 = None
        self.x2 = None
        self.y2 = None

        self.dx1 = None
        self.dx2 = None
        self.dy1 = None
        self.dy2 = None

        self.data_align = data_align
        self.tik_instance = tik_instance
        self.dup_rep_time = dup_rep_time

    def init_data(self):
        # func: create for the calculation cache
        self.x = self.tik_instance.Tensor("float32", [self.data_align], name="x", scope=tik.scope_ubuf)
        self.y = self.tik_instance.Tensor("float32", [self.data_align], name="y", scope=tik.scope_ubuf)
        self.w = self.tik_instance.Tensor("float32", [self.data_align], name="w", scope=tik.scope_ubuf)
        self.h = self.tik_instance.Tensor("float32", [self.data_align], name="h", scope=tik.scope_ubuf)

        self.x1 = self.tik_instance.Tensor("float32", [self.data_align], name="x1", scope=tik.scope_ubuf)
        self.x2 = self.tik_instance.Tensor("float32", [self.data_align], name="x2", scope=tik.scope_ubuf)
        self.y1 = self.tik_instance.Tensor("float32", [self.data_align], name="y1", scope=tik.scope_ubuf)
        self.y2 = self.tik_instance.Tensor("float32", [self.data_align], name="y2", scope=tik.scope_ubuf)

        self.dx1 = self.tik_instance.Tensor("float32", [self.data_align], name="dx1", scope=tik.scope_ubuf)
        self.dx2 = self.tik_instance.Tensor("float32", [self.data_align], name="dx2", scope=tik.scope_ubuf)
        self.dy1 = self.tik_instance.Tensor("float32", [self.data_align], name="dy1", scope=tik.scope_ubuf)
        self.dy2 = self.tik_instance.Tensor("float32", [self.data_align], name="dy2", scope=tik.scope_ubuf)

        self.tik_instance.vector_dup(Constant.MASK_BLOCK, self.dx1, 0, self.dup_rep_time, 1, Constant.REP_STRIDE)
        self.tik_instance.vector_dup(Constant.MASK_BLOCK, self.dx2, 0, self.dup_rep_time, 1, Constant.REP_STRIDE)
        self.tik_instance.vector_dup(Constant.MASK_BLOCK, self.dy1, 0, self.dup_rep_time, 1, Constant.REP_STRIDE)
        self.tik_instance.vector_dup(Constant.MASK_BLOCK, self.dy2, 0, self.dup_rep_time, 1, Constant.REP_STRIDE)


class CommonIoUGrad(object):
    """CommonIoUGrad"""

    # 'pylint: disable=too-many-statements,too-many-arguments
    def __init__(self, dy, bboxes, trans, kernel_name):
        """__init__"""
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.kernel_name = kernel_name
        self.trans = trans
        self.dtype = self.paras_check(dy, bboxes)

        # func: for task allocation
        self.avail_aicore_num = get_soc_spec("CORE_NUM")
        self.available_ub_size = get_soc_spec("UB_SIZE")

        self.dy = self.tik_instance.Tensor(self.dtype, [Constant.MAX_INT32], name="dy", scope=tik.scope_gm)
        self.bboxes = self.tik_instance.Tensor(self.dtype, [Constant.MAX_INT32], name="bboxes",
                                               scope=tik.scope_gm)
        self.gtboxes = self.tik_instance.Tensor(self.dtype, [Constant.MAX_INT32], name="gtboxes",
                                                scope=tik.scope_gm)

        self.dbboxes = self.tik_instance.Tensor(self.dtype, [Constant.MAX_INT32], name="dbboxes",
                                                scope=tik.scope_gm)
        self.dgtboxes = self.tik_instance.Tensor(self.dtype, [Constant.MAX_INT32], name="dgtboxes",
                                                 scope=tik.scope_gm)
        
        self.dbboxes_ = self.tik_instance.Tensor(self.dtype, [Constant.MAX_INT32],
                                                 name="dbboxes_", scope=tik.scope_gm, is_workspace=True)
        self.dgtboxes_ = self.tik_instance.Tensor(self.dtype, [Constant.MAX_INT32],
                                                  name="dgtboxes_", scope=tik.scope_gm, is_workspace=True)
        
        self.tiling_gm = self.tik_instance.Tensor(
            Constant.TILING_SCALAR_DTYPE,
            [Constant.TILING_PARAMS_NUM],
            name="tiling_gm",
            scope=tik.scope_gm
        )

        self.batch_tail = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "batch_tail")
        self.batch_num_per_aicore = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "batch_num_per_aicore")
        self.core_num_var = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "core_num_var")
        self.move_flag = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "move_flag")
        self.all_num_align = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "all_num_align")
        self.task_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "task_num")
        self.dup_rep_time = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "dup_rep_time")
        self.mov_rep_time = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "mov_rep_time")
        self.data_align = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "data_align")
        self.all_num = self.tik_instance.Scalar(Constant.TILING_SCALAR_DTYPE, "all_num")

        self.ub_segment_max = self.available_ub_size // (Constant.UB_NUM * Constant.BYTE_PER_DATA)

        self.b_box = Boxes(self.tik_instance, self.ub_segment_max, self.dup_rep_time)
        self.g_box = Boxes(self.tik_instance, self.ub_segment_max, self.dup_rep_time)

        # func: apply for the calculation cache of inter/union
        self.inter = None
        self.unionarea = None

        # func: apply for the calculation cache of mask: record the result of compare api
        self.mask_1 = None
        self.mask_2 = None

        # func: apply for the calculation cache of dy_ub
        self.dy_ub = None

        # func: apply for the calculation cache of xlen (xmin-xmax) ylen (ymin-ymax)
        self.xlen = None
        self.ylen = None

        # func: apply for the calculation cache of inter1 = np.maximum(xlen, 0) inter2 = np.maximum(ylen, 0)
        self.inter_x = None
        self.inter_y = None

        # func: apply for the calculation cache of dxlen/dylen
        self.dxlen = None
        self.dylen = None

        # func: apply for the calculation cache of zero
        self.tmp_zero = None

        # func: apply for the calculation cache of dinter/dunion
        self.dinter = None
        self.dunion = None

        # func: apply for the calculation cache of temp obj
        self.tmp_a = None
        self.tmp_b = None
        self.tmp_c = None
        self.tmp_d = None

    def paras_check(self, dy, bboxes):
        """paras_check"""
       
        dtype_dy = dy.get("dtype").lower()
        para_check.check_dtype_rule(dtype_dy, ("float32"))

        dtype_bboxes = bboxes.get("dtype").lower()
        para_check.check_dtype_rule(dtype_bboxes, ("float32"))

        para_check.check_kernel_name(self.kernel_name)

        return dtype_dy
        
    def get_tiling_data(self):
        tiling_ub = self.tik_instance.Tensor(
            Constant.TILING_SCALAR_DTYPE,
            (Constant.TILING_PARAMS_NUM,),
            tik.scope_ubuf,
            "tiling_ub"
        )

        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, 3, 0, 0)
        self.batch_tail.set_as(tiling_ub[9])
        self.batch_num_per_aicore.set_as(tiling_ub[8])
        self.core_num_var.set_as(tiling_ub[7])
        self.move_flag.set_as(tiling_ub[6])
        self.all_num_align.set_as(tiling_ub[5])
        self.task_num.set_as(tiling_ub[4])
        self.dup_rep_time.set_as(tiling_ub[3])
        self.mov_rep_time.set_as(tiling_ub[2])
        self.data_align.set_as(tiling_ub[1])
        self.all_num.set_as(tiling_ub[0])

    def compute(self):
        """iou_grad_compute"""
        self.get_tiling_data()

        with self.tik_instance.for_range(0, self.core_num_var, block_num=self.core_num_var) as i:
            with self.tik_instance.for_range(0, self.batch_num_per_aicore) as j:
                self.compute_core(i + j * self.core_num_var)
            with self.tik_instance.if_scope(i < self.batch_tail):
                self.compute_core(self.batch_num_per_aicore * self.core_num_var + i)

        with self.tik_instance.if_scope(self.move_flag == 1):
            self.move_out()

        tbe_context.get_context().add_compile_info(
            "vars", {
                "full_core_num": self.avail_aicore_num,
                "ub_size": self.available_ub_size
            }
        )

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.dy, self.bboxes, self.gtboxes],
                                   outputs=[self.dbboxes, self.dgtboxes],
                                   flowtable=[self.tiling_gm])

        return self.tik_instance

    def compute_core(self, task_idx):
        """iou_grad_compute_compute_core"""
        # func: init all unit
        self.init_date()

        # func: get b1 and b2
        self.move_in(task_idx)

        # func: compute for inter/union
        self.update_forward_part()

        # func: compute for dinter/dunion
        self.update_backward_part()

        # func: compute for dbboxes/dgtboxes in inter
        self.inter_part()

        # func: compute for dbboxes/dgtboxes in union
        self.union_part()

        # func: resite res for attr_trans
        self.update_dboxes(task_idx)

    def init_date(self):
        """init_date"""
        self.b_box.init_data()
        self.g_box.init_data()
        # func: create for the calculation cache of inter/union
        self.inter = self.tik_instance.Tensor("float32", [self.data_align], name="inter", scope=tik.scope_ubuf)
        self.unionarea = self.tik_instance.Tensor("float32", [self.data_align],
                                                  name="unionarea", scope=tik.scope_ubuf)

        # func: create for the calculation cache of mask: record the result of compare api
        self.mask_1 = self.tik_instance.Tensor("uint16", [self.data_align // 16], name="mask_1", scope=tik.scope_ubuf)
        self.mask_2 = self.tik_instance.Tensor("uint16", [self.data_align // 16], name="mask_2", scope=tik.scope_ubuf)

        # func: ainitpply for the calculation cache of dy_ub
        self.dy_ub = self.tik_instance.Tensor("float32", [self.data_align], name="targets_ub", scope=tik.scope_ubuf)

        # func: init for the calculation cache of xlen (xmin-xmax) ylen (ymin-ymax)
        self.xlen = self.tik_instance.Tensor("float32", [self.data_align], name="xlen", scope=tik.scope_ubuf)
        self.ylen = self.tik_instance.Tensor("float32", [self.data_align], name="ylen", scope=tik.scope_ubuf)

        # func: init for the calculation cache of inter1 = np.maximum(xlen, 0) inter2 = np.maximum(ylen, 0)
        self.inter_x = self.tik_instance.Tensor("float32", [self.data_align], name="inter_x", scope=tik.scope_ubuf)
        self.inter_y = self.tik_instance.Tensor("float32", [self.data_align], name="inter_y", scope=tik.scope_ubuf)

        # func: init for the calculation cache of dxlen/dylen
        self.dxlen = self.tik_instance.Tensor("float32", [self.data_align], name="dxlen", scope=tik.scope_ubuf)
        self.dylen = self.tik_instance.Tensor("float32", [self.data_align], name="dylen", scope=tik.scope_ubuf)

        # func: init for the calculation cache of zero
        self.tmp_zero = self.tik_instance.Tensor("float32", [self.data_align],
                                                 name="tmp_zero", scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(Constant.MASK_BLOCK, self.tmp_zero, 0.0, self.dup_rep_time, 1, Constant.REP_STRIDE)

        # func: init for the calculation cache of dinter/dunion/denclose
        self.dinter = self.tik_instance.Tensor("float32", [self.data_align], name="dinter", scope=tik.scope_ubuf)
        self.dunion = self.tik_instance.Tensor("float32", [self.data_align], name="dunion", scope=tik.scope_ubuf)

        # func: init for the calculation cache of temp obj
        self.tmp_a = self.tik_instance.Tensor("float32", [self.data_align], name="tmp_a", scope=tik.scope_ubuf)
        self.tmp_b = self.tik_instance.Tensor("float32", [self.data_align], name="tmp_b", scope=tik.scope_ubuf)
        self.tmp_c = self.tik_instance.Tensor("float32", [self.data_align], name="tmp_c", scope=tik.scope_ubuf)
        self.tmp_d = self.tik_instance.Tensor("float32", [self.data_align], name="tmp_d", scope=tik.scope_ubuf)

    def move_in(self, task_idx):
        """move_in"""
        # func: for dy
        self.tik_instance.data_move(self.dy_ub, self.dy[task_idx * self.data_align], 0, 1, self.mov_rep_time, 0, 0)

        # func: xyhw trans to xyxy
        if self.trans:
            self.trans_true(task_idx)
        else:
            self.trans_false(task_idx)

        # func: choose the positive one
        with self.tik_instance.for_range(0, self.data_align // Constant.MASK_BLOCK) as idx:
            self.tik_instance.vec_cmpv_gt(self.mask_1, self.b_box.h[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                          Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0,
                                      self.tmp_a[Constant.MASK_BLOCK * idx],
                                      self.mask_1, self.b_box.h[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)

            self.tik_instance.vec_cmpv_gt(self.mask_2, self.g_box.h[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                          Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0,
                                      self.tmp_b[Constant.MASK_BLOCK * idx],
                                      self.mask_2, self.g_box.h[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)

            self.tik_instance.vec_cmpv_gt(self.mask_1, self.b_box.w[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                          Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0,
                                      self.tmp_c[Constant.MASK_BLOCK * idx],
                                      self.mask_1, self.b_box.w[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)

            self.tik_instance.vec_cmpv_gt(self.mask_2, self.g_box.w[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                          Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0,
                                      self.tmp_d[Constant.MASK_BLOCK * idx],
                                      self.mask_2, self.g_box.w[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)

        # func: add eps
        self.tik_instance.h_add(self.b_box.h, self.tmp_a, Constant.EPS)
        self.tik_instance.h_add(self.g_box.h, self.tmp_b, Constant.EPS)
        self.tik_instance.h_add(self.b_box.w, self.tmp_c, Constant.EPS)
        self.tik_instance.h_add(self.g_box.w, self.tmp_d, Constant.EPS)

    def trans_true(self, task_idx):
        # func: for bboxes
        self.tik_instance.data_move(self.b_box.x, self.bboxes[task_idx * self.data_align], 0, 1,
                                    self.mov_rep_time, 0, 0)
        self.tik_instance.data_move(self.b_box.y, self.bboxes[self.all_num + task_idx * self.data_align], 0, 1,
                                    self.mov_rep_time, 0, 0)
        self.tik_instance.data_move(self.b_box.w, self.bboxes[self.all_num * 2 + task_idx * self.data_align], 0, 1,
                                    self.mov_rep_time, 0, 0)
        self.tik_instance.data_move(self.b_box.h, self.bboxes[self.all_num * 3 + task_idx * self.data_align], 0, 1,
                                    self.mov_rep_time, 0, 0)
        # func: for gtboxes
        self.tik_instance.data_move(self.g_box.x, self.gtboxes[task_idx * self.data_align], 0, 1,
                                    self.mov_rep_time, 0, 0)
        self.tik_instance.data_move(self.g_box.y, self.gtboxes[self.all_num + task_idx * self.data_align], 0, 1,
                                    self.mov_rep_time, 0, 0)
        self.tik_instance.data_move(self.g_box.w, self.gtboxes[self.all_num * 2 + task_idx * self.data_align], 0, 1,
                                    self.mov_rep_time, 0, 0)
        self.tik_instance.data_move(self.g_box.h, self.gtboxes[self.all_num * 3 + task_idx * self.data_align], 0, 1,
                                    self.mov_rep_time, 0, 0)

        self.tik_instance.h_mul(self.tmp_a, self.b_box.w, Constant.HALF)
        self.tik_instance.h_mul(self.tmp_b, self.b_box.h, Constant.HALF)

        self.tik_instance.h_sub(self.b_box.x1, self.b_box.x, self.tmp_a)
        self.tik_instance.h_add(self.b_box.x2, self.b_box.x, self.tmp_a)
        self.tik_instance.h_sub(self.b_box.y1, self.b_box.y, self.tmp_b)
        self.tik_instance.h_add(self.b_box.y2, self.b_box.y, self.tmp_b)

        self.tik_instance.h_mul(self.tmp_c, self.g_box.w, Constant.HALF)
        self.tik_instance.h_mul(self.tmp_d, self.g_box.h, Constant.HALF)

        self.tik_instance.h_sub(self.g_box.x1, self.g_box.x, self.tmp_c)
        self.tik_instance.h_add(self.g_box.x2, self.g_box.x, self.tmp_c)
        self.tik_instance.h_sub(self.g_box.y1, self.g_box.y, self.tmp_d)
        self.tik_instance.h_add(self.g_box.y2, self.g_box.y, self.tmp_d)

    def trans_false(self, task_idx):
        # func: for bboxes
        self.tik_instance.data_move(self.b_box.x1, self.bboxes[task_idx * self.data_align], 0, 1,
                                    self.mov_rep_time, 0, 0)
        self.tik_instance.data_move(self.b_box.y1, self.bboxes[self.all_num + task_idx * self.data_align], 0, 1,
                                    self.mov_rep_time, 0, 0)
        self.tik_instance.data_move(self.b_box.x2, self.bboxes[self.all_num * 2 + task_idx * self.data_align],
                                    0, 1, self.mov_rep_time, 0, 0)
        self.tik_instance.data_move(self.b_box.y2, self.bboxes[self.all_num * 3 + task_idx * self.data_align],
                                    0, 1, self.mov_rep_time, 0, 0)

        self.tik_instance.h_sub(self.b_box.w, self.b_box.x2, self.b_box.x1)
        self.tik_instance.h_sub(self.b_box.h, self.b_box.y2, self.b_box.y1)

        # func: for gtboxes
        self.tik_instance.data_move(self.g_box.x1, self.gtboxes[task_idx * self.data_align], 0, 1,
                                    self.mov_rep_time, 0, 0)
        self.tik_instance.data_move(self.g_box.y1, self.gtboxes[self.all_num + task_idx * self.data_align], 0,
                                    1, self.mov_rep_time, 0, 0)
        self.tik_instance.data_move(self.g_box.x2, self.gtboxes[self.all_num * 2 + task_idx * self.data_align],
                                    0, 1, self.mov_rep_time, 0, 0)
        self.tik_instance.data_move(self.g_box.y2, self.gtboxes[self.all_num * 3 + task_idx * self.data_align],
                                    0, 1, self.mov_rep_time, 0, 0)

        self.tik_instance.h_sub(self.g_box.w, self.g_box.x2, self.g_box.x1)
        self.tik_instance.h_sub(self.g_box.h, self.g_box.y2, self.g_box.y1)

        self.tik_instance.h_mul(self.tmp_a, self.b_box.w, Constant.HALF)
        self.tik_instance.h_mul(self.tmp_b, self.b_box.h, Constant.HALF)

        self.tik_instance.h_sub(self.b_box.x, self.b_box.x2, self.tmp_a)
        self.tik_instance.h_sub(self.b_box.y, self.b_box.y2, self.tmp_b)

        self.tik_instance.h_mul(self.tmp_c, self.g_box.w, Constant.HALF)
        self.tik_instance.h_mul(self.tmp_d, self.g_box.h, Constant.HALF)

        self.tik_instance.h_sub(self.g_box.x, self.g_box.x2, self.tmp_c)
        self.tik_instance.h_sub(self.g_box.y, self.g_box.y2, self.tmp_d)

    def update_forward_part(self):
        """update_forward_part"""
        b1_area = self.tik_instance.Tensor("float32", [self.data_align], name="b1_area", scope=tik.scope_ubuf)
        b2_area = self.tik_instance.Tensor("float32", [self.data_align], name="b2_area", scope=tik.scope_ubuf)

        # func: `for inter:  max(min(b1x2. b2x2) - max(b1x1, b2x1), 0) * max(min(b1y2. b2y2) - max(b1y1, b2y1), 0)`
        self.tik_instance.h_max(self.tmp_a, self.b_box.x1, self.g_box.x1)
        self.tik_instance.h_max(self.tmp_c, self.b_box.y1, self.g_box.y1)

        self.tik_instance.h_min(self.tmp_b, self.b_box.x2, self.g_box.x2)
        self.tik_instance.h_min(self.tmp_d, self.b_box.y2, self.g_box.y2)

        self.tik_instance.h_sub(self.xlen, self.tmp_b, self.tmp_a)
        self.tik_instance.h_sub(self.ylen, self.tmp_d, self.tmp_c)

        # func: choose the positive one
        with self.tik_instance.for_range(0, self.data_align // Constant.MASK_BLOCK) as idx:
            self.tik_instance.vec_cmpv_gt(self.mask_1, self.xlen[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                          Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0,
                                      self.inter_x[Constant.MASK_BLOCK * idx],
                                      self.mask_1, self.xlen[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)

            self.tik_instance.vec_cmpv_gt(self.mask_2, self.ylen[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                          Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0,
                                      self.inter_y[Constant.MASK_BLOCK * idx],
                                      self.mask_2, self.ylen[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)

        self.tik_instance.h_mul(self.inter, self.inter_x, self.inter_y)

        # func: `for union: b1_area * b2_area - inter`
        self.tik_instance.h_mul(b1_area, self.b_box.w, self.b_box.h)
        self.tik_instance.h_mul(b2_area, self.g_box.w, self.g_box.h)

        self.tik_instance.h_add(self.unionarea, b1_area, b2_area)
        self.tik_instance.h_sub(self.unionarea, self.unionarea, self.inter)

        # func: `for cw: (max(b1x2. b2x2) - min(b1x1, b2x1)) ch  (max(b1y2. b2y2) - min(b1y1, b2y1))`
        self.tik_instance.h_max(self.tmp_a, self.b_box.x2, self.g_box.x2)
        self.tik_instance.h_min(self.tmp_b, self.b_box.x1, self.g_box.x1)

        self.tik_instance.h_max(self.tmp_c, self.b_box.y2, self.g_box.y2)
        self.tik_instance.h_min(self.tmp_d, self.b_box.y1, self.g_box.y1)

        self.tik_instance.h_sub(self.cw, self.tmp_a, self.tmp_b)
        self.tik_instance.h_sub(self.ch, self.tmp_c, self.tmp_d)

        # func: choose the positive one
        with self.tik_instance.for_range(0, self.data_align // Constant.MASK_BLOCK) as idx:
            self.tik_instance.vec_cmpv_gt(self.mask_1, self.cw[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                          Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0,
                                      self.tmp_a[Constant.MASK_BLOCK * idx],
                                      self.mask_1, self.cw[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)

            self.tik_instance.vec_cmpv_gt(self.mask_2, self.ch[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                          Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0,
                                      self.tmp_b[Constant.MASK_BLOCK * idx],
                                      self.mask_2, self.ch[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)

        # func: add eps
        self.tik_instance.h_add(self.cw, self.tmp_a, Constant.EPS)
        self.tik_instance.h_add(self.ch, self.tmp_b, Constant.EPS)

    def update_backward_part(self):
        """update_backward_part"""
        # `for dunion, dunion = (- inter / (union ** 2)) * dy`
        self.tik_instance.h_div(self.tmp_a, self.dy_ub, self.unionarea)
        self.tik_instance.h_div(self.tmp_b, self.tmp_a, self.unionarea)
        self.tik_instance.h_mul(self.tmp_c, self.inter, self.tmp_b)
        self.tik_instance.h_sub(self.dunion, self.tmp_zero, self.tmp_c)

        # `for dinter, dinter = 1 / union * dy - dunion`
        self.tik_instance.h_div(self.dinter, self.dy_ub, self.unionarea)
        self.tik_instance.h_sub(self.dinter, self.dinter, self.dunion)

    def inter_part(self):
        """inter_part"""
        self.tik_instance.h_mul(self.dxlen, self.dinter, self.inter_y)
        self.tik_instance.h_mul(self.dylen, self.dinter, self.inter_x)

        with self.tik_instance.for_range(0, self.data_align // Constant.MASK_BLOCK) as idx:
            self.tik_instance.vec_cmpv_lt(self.mask_1, self.b_box.x2[Constant.MASK_BLOCK * idx],
                                          self.g_box.x2[Constant.MASK_BLOCK * idx], 1,
                                          Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_cmpv_ge(self.mask_2, self.xlen[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                          Constant.REP_STRIDE, Constant.REP_STRIDE)

            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_a[Constant.MASK_BLOCK * idx], self.mask_1,
                                      self.dxlen[Constant.MASK_BLOCK * idx], self.tmp_zero,
                                      1, Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)

            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_a[Constant.MASK_BLOCK * idx], self.mask_2,
                                      self.tmp_a[Constant.MASK_BLOCK * idx], self.tmp_zero,
                                      1, Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)

            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_b[Constant.MASK_BLOCK * idx], self.mask_1,
                                      self.tmp_zero, self.dxlen[Constant.MASK_BLOCK * idx],
                                      1, Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)

            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_b[Constant.MASK_BLOCK * idx], self.mask_2,
                                      self.tmp_b[Constant.MASK_BLOCK * idx], self.tmp_zero,
                                      1, Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)

            self.tik_instance.vec_cmpv_lt(self.mask_1, self.b_box.y2[Constant.MASK_BLOCK * idx],
                                          self.g_box.y2[Constant.MASK_BLOCK * idx], 1,
                                          Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_cmpv_ge(self.mask_2, self.ylen[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                          Constant.REP_STRIDE, Constant.REP_STRIDE)

            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_c[Constant.MASK_BLOCK * idx], self.mask_1,
                                      self.dylen[Constant.MASK_BLOCK * idx], self.tmp_zero,
                                      1, Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)

            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_c[Constant.MASK_BLOCK * idx], self.mask_2,
                                      self.tmp_c[Constant.MASK_BLOCK * idx], self.tmp_zero,
                                      1, Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)

            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_d[Constant.MASK_BLOCK * idx], self.mask_1,
                                      self.tmp_zero, self.dylen[Constant.MASK_BLOCK * idx],
                                      1, Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)

            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_d[Constant.MASK_BLOCK * idx], self.mask_2,
                                      self.tmp_d[Constant.MASK_BLOCK * idx], self.tmp_zero,
                                      1, Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)

        self.tik_instance.h_add(self.b_box.dx2, self.tmp_a, self.b_box.dx2)
        self.tik_instance.h_add(self.g_box.dx2, self.tmp_b, self.g_box.dx2)
        self.tik_instance.h_add(self.b_box.dy2, self.tmp_c, self.b_box.dy2)
        self.tik_instance.h_add(self.g_box.dy2, self.tmp_d, self.g_box.dy2)

        self.tik_instance.h_sub(self.dxlen, self.tmp_zero, self.dxlen)
        self.tik_instance.h_sub(self.dylen, self.tmp_zero, self.dylen)

        with self.tik_instance.for_range(0, self.data_align // Constant.MASK_BLOCK) as idx:
            self.tik_instance.vec_cmpv_gt(self.mask_1, self.b_box.x1[Constant.MASK_BLOCK * idx],
                                          self.g_box.x1[Constant.MASK_BLOCK * idx], 1,
                                          Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_cmpv_ge(self.mask_2, self.xlen[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                          Constant.REP_STRIDE, Constant.REP_STRIDE)

            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_a[Constant.MASK_BLOCK * idx], self.mask_1,
                                      self.dxlen[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)

            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_a[Constant.MASK_BLOCK * idx], self.mask_2,
                                      self.tmp_a[Constant.MASK_BLOCK * idx], self.tmp_zero, 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)

            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_b[Constant.MASK_BLOCK * idx], self.mask_1,
                                      self.tmp_zero, self.dxlen[Constant.MASK_BLOCK * idx],
                                      Constant.MASK_BLOCK // Constant.MASK_BLOCK,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)

            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_b[Constant.MASK_BLOCK * idx], self.mask_2,
                                      self.tmp_b[Constant.MASK_BLOCK * idx], self.tmp_zero,
                                      Constant.MASK_BLOCK // Constant.MASK_BLOCK, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)

            self.tik_instance.vec_cmpv_gt(self.mask_1, self.b_box.y1[Constant.MASK_BLOCK * idx],
                                          self.g_box.y1[Constant.MASK_BLOCK * idx], 1,
                                          Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_cmpv_ge(self.mask_2, self.ylen[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                          Constant.REP_STRIDE, Constant.REP_STRIDE)

            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_c[Constant.MASK_BLOCK * idx], self.mask_1,
                                      self.dylen[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)

            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_c[Constant.MASK_BLOCK * idx], self.mask_2,
                                      self.tmp_c[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)

            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_d[Constant.MASK_BLOCK * idx], self.mask_1,
                                      self.tmp_zero, self.dylen[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)

            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_d[Constant.MASK_BLOCK * idx], self.mask_2,
                                      self.tmp_d[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)

        self.tik_instance.h_add(self.b_box.dx1, self.tmp_a, self.b_box.dx1)
        self.tik_instance.h_add(self.g_box.dx1, self.tmp_b, self.g_box.dx1)
        self.tik_instance.h_add(self.b_box.dy1, self.tmp_c, self.b_box.dy1)
        self.tik_instance.h_add(self.g_box.dy1, self.tmp_d, self.g_box.dy1)

    def union_part(self):
        """union_part"""
        # b_box
        self.tik_instance.h_mul(self.tmp_a, self.b_box.w, self.dunion)
        self.tik_instance.h_mul(self.tmp_b, self.b_box.h, self.dunion)
        # `for union part : b1x2-b1x1`
        self.tik_instance.h_add(self.b_box.dx2, self.b_box.dx2, self.tmp_b)
        self.tik_instance.h_sub(self.b_box.dx1, self.b_box.dx1, self.tmp_b)
        # `for union part : b1y2-b1y1`
        self.tik_instance.h_add(self.b_box.dy2, self.b_box.dy2, self.tmp_a)
        self.tik_instance.h_sub(self.b_box.dy1, self.b_box.dy1, self.tmp_a)

        # g_box
        self.tik_instance.h_mul(self.tmp_c, self.g_box.w, self.dunion)
        self.tik_instance.h_mul(self.tmp_d, self.g_box.h, self.dunion)
        # `for union part : b2x2-b2x1`
        self.tik_instance.h_add(self.g_box.dx2, self.g_box.dx2, self.tmp_d)
        self.tik_instance.h_sub(self.g_box.dx1, self.g_box.dx1, self.tmp_d)
        # `for union part : b2y2-b2y1`
        self.tik_instance.h_add(self.g_box.dy2, self.g_box.dy2, self.tmp_c)
        self.tik_instance.h_sub(self.g_box.dy1, self.g_box.dy1, self.tmp_c)

    def update_dboxes(self, task_idx):
        """update_dboxes"""
        if self.trans:
            self.update_dboxes_trans_true(task_idx)
        else:
            self.update_dboxes_trans_false(task_idx)

    def update_dboxes_trans_true(self, task_idx):
        """update_dboxes_trans_true"""
        # for b1x b1y b2x b2y
        self.tik_instance.h_add(self.tmp_a, self.b_box.dx1, self.b_box.dx2)
        self.tik_instance.h_add(self.tmp_b, self.b_box.dy1, self.b_box.dy2)
        self.tik_instance.h_add(self.tmp_c, self.g_box.dx1, self.g_box.dx2)
        self.tik_instance.h_add(self.tmp_d, self.g_box.dy1, self.g_box.dy2)

        with self.tik_instance.if_scope(self.move_flag == 1):
            self.tik_instance.data_move(self.dbboxes_[task_idx * self.data_align], self.tmp_a, 0, 1,
                                        self.mov_rep_time, 0, 0)
            self.tik_instance.data_move(self.dbboxes_[self.all_num_align + task_idx * self.data_align],
                                        self.tmp_b, 0, 1, self.mov_rep_time, 0, 0)
            self.tik_instance.data_move(self.dgtboxes_[task_idx * self.data_align], self.tmp_c, 0, 1,
                                        self.mov_rep_time, 0, 0)
            self.tik_instance.data_move(self.dgtboxes_[self.all_num_align + task_idx * self.data_align],
                                        self.tmp_d, 0, 1, self.mov_rep_time, 0, 0)

        with self.tik_instance.else_scope():
            self.tik_instance.data_move(self.dbboxes[task_idx * self.data_align], self.tmp_a, 0, 1,
                                        self.mov_rep_time, 0, 0)
            self.tik_instance.data_move(self.dbboxes[self.all_num + task_idx * self.data_align], self.tmp_b, 0,
                                        1, self.mov_rep_time, 0, 0)
            self.tik_instance.data_move(self.dgtboxes[task_idx * self.data_align], self.tmp_c, 0, 1,
                                        self.mov_rep_time, 0, 0)
            self.tik_instance.data_move(self.dgtboxes[self.all_num + task_idx * self.data_align],
                                        self.tmp_d, 0, 1, self.mov_rep_time, 0, 0)

        # for b1w b1h b2w b2h
        self.tik_instance.h_sub(self.tmp_a, self.b_box.dx2, self.b_box.dx1)
        self.tik_instance.h_mul(self.tmp_a, self.tmp_a, Constant.HALF)

        self.tik_instance.h_sub(self.tmp_b, self.b_box.dy2, self.b_box.dy1)
        self.tik_instance.h_mul(self.tmp_b, self.tmp_b, Constant.HALF)

        self.tik_instance.h_sub(self.tmp_c, self.g_box.dx2, self.g_box.dx1)
        self.tik_instance.h_mul(self.tmp_c, self.tmp_c, Constant.HALF)

        self.tik_instance.h_sub(self.tmp_d, self.g_box.dy2, self.g_box.dy1)
        self.tik_instance.h_mul(self.tmp_d, self.tmp_d, Constant.HALF)

        with self.tik_instance.if_scope(self.move_flag == 1):
            self.tik_instance.data_move(self.dbboxes_[self.all_num_align * 2 + task_idx * self.data_align],
                                        self.tmp_a, 0, 1, self.mov_rep_time, 0, 0)
            self.tik_instance.data_move(self.dbboxes_[self.all_num_align * 3 + task_idx * self.data_align],
                                        self.tmp_b, 0, 1, self.mov_rep_time, 0, 0)
            self.tik_instance.data_move(self.dgtboxes_[self.all_num_align * 2 + task_idx * self.data_align],
                                        self.tmp_c, 0, 1, self.mov_rep_time, 0, 0)
            self.tik_instance.data_move(self.dgtboxes_[self.all_num_align * 3 + task_idx * self.data_align],
                                        self.tmp_d, 0, 1, self.mov_rep_time, 0, 0)
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(self.dbboxes[self.all_num * 2 + task_idx * self.data_align],
                                        self.tmp_a, 0, 1, self.mov_rep_time, 0, 0)
            self.tik_instance.data_move(self.dbboxes[self.all_num * 3 + task_idx * self.data_align],
                                        self.tmp_b, 0, 1, self.mov_rep_time, 0, 0)
            self.tik_instance.data_move(self.dgtboxes[self.all_num * 2 + task_idx * self.data_align],
                                        self.tmp_c, 0, 1, self.mov_rep_time, 0, 0)
            self.tik_instance.data_move(self.dgtboxes[self.all_num * 3 + task_idx * self.data_align],
                                        self.tmp_d, 0, 1, self.mov_rep_time, 0, 0)

    def update_dboxes_trans_false(self, task_idx):
        """update_dboxes_trans_true"""
        with self.tik_instance.if_scope(self.move_flag == 1):
            self.tik_instance.data_move(self.dbboxes_[task_idx * self.data_align], self.b_box.dx1, 0, 1,
                                        self.mov_rep_time, 0, 0)
            self.tik_instance.data_move(self.dbboxes_[self.all_num_align + task_idx * self.data_align],
                                        self.b_box.dy1, 0, 1, self.mov_rep_time, 0, 0)
            self.tik_instance.data_move(self.dbboxes_[self.all_num_align * 2 + task_idx * self.data_align],
                                        self.b_box.dx2, 0, 1, self.mov_rep_time, 0, 0)
            self.tik_instance.data_move(self.dbboxes_[self.all_num_align * 3 + task_idx * self.data_align],
                                        self.b_box.dy2, 0, 1, self.mov_rep_time, 0, 0)

            self.tik_instance.data_move(self.dgtboxes_[task_idx * self.data_align], self.g_box.dx1, 0, 1,
                                        self.mov_rep_time, 0, 0)
            self.tik_instance.data_move(self.dgtboxes_[self.all_num_align + task_idx * self.data_align],
                                        self.g_box.dy1, 0, 1, self.mov_rep_time, 0, 0)
            self.tik_instance.data_move(self.dgtboxes_[self.all_num_align * 2 + task_idx * self.data_align],
                                        self.g_box.dx2, 0, 1, self.mov_rep_time, 0, 0)
            self.tik_instance.data_move(self.dgtboxes_[self.all_num_align * 3 + task_idx * self.data_align],
                                        self.g_box.dy2, 0, 1, self.mov_rep_time, 0, 0)
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(self.dbboxes[task_idx * self.data_align], self.b_box.dx1, 0, 1,
                                        self.mov_rep_time, 0, 0)
            self.tik_instance.data_move(self.dbboxes[self.all_num + task_idx * self.data_align],
                                        self.b_box.dy1, 0, 1, self.mov_rep_time, 0, 0)
            self.tik_instance.data_move(self.dbboxes[self.all_num * 2 + task_idx * self.data_align],
                                        self.b_box.dx2, 0, 1, self.mov_rep_time, 0, 0)
            self.tik_instance.data_move(self.dbboxes[self.all_num * 3 + task_idx * self.data_align],
                                        self.b_box.dy2, 0, 1, self.mov_rep_time, 0, 0)

            self.tik_instance.data_move(self.dgtboxes[task_idx * self.data_align], self.g_box.dx1, 0, 1,
                                        self.mov_rep_time, 0, 0)
            self.tik_instance.data_move(self.dgtboxes[self.all_num + task_idx * self.data_align],
                                        self.g_box.dy1, 0, 1, self.mov_rep_time, 0, 0)
            self.tik_instance.data_move(self.dgtboxes[self.all_num * 2 + task_idx * self.data_align],
                                        self.g_box.dx2, 0, 1, self.mov_rep_time, 0, 0)
            self.tik_instance.data_move(self.dgtboxes[self.all_num * 3 + task_idx * self.data_align],
                                        self.g_box.dy2, 0, 1, self.mov_rep_time, 0, 0)

    def move_out(self):
        """move_out"""
        with self.tik_instance.if_scope((self.available_ub_size // Constant.BYTE_PER_DATA) >= 2 * self.all_num_align):
            dbboxes_tmp = self.tik_instance.Tensor("float32", [self.all_num_align], name="dbboxes_tmp",
                                                   scope=tik.scope_ubuf)
            gtboxes_tmp = self.tik_instance.Tensor("float32", [self.all_num_align], name="gtboxes_tmp",
                                                   scope=tik.scope_ubuf)
            # func: Address fallback for dbboxes
            with self.tik_instance.for_range(0, Constant.BOX_LOC) as idx:
                self.tik_instance.data_move(dbboxes_tmp, self.dbboxes_[idx * self.all_num_align],
                                            0, 1, self.all_num_align // Constant.BLOCK, 0, 0)
                self.tik_instance.data_move(self.dbboxes[idx * self.all_num], dbboxes_tmp, 0, 1,
                                            self.all_num_align // Constant.BLOCK, 0, 0)

            # func: Address fallback for dgtboxes
            with self.tik_instance.for_range(0, Constant.BOX_LOC) as idx:
                self.tik_instance.data_move(gtboxes_tmp, self.dgtboxes_[idx * self.all_num_align], 0, 1,
                                            self.all_num_align // Constant.BLOCK, 0, 0)
                self.tik_instance.data_move(self.dgtboxes[idx * self.all_num], gtboxes_tmp, 0, 1,
                                            self.all_num_align // Constant.BLOCK, 0, 0)
        with self.tik_instance.else_scope():
            dbboxes_tmp = self.tik_instance.Tensor("float32", [self.data_align], name="dbboxes_tmp",
                                                   scope=tik.scope_ubuf)
            gtboxes_tmp = self.tik_instance.Tensor("float32", [self.data_align], name="gtboxes_tmp",
                                                   scope=tik.scope_ubuf)
            # func: Address fallback for dbboxes
            with self.tik_instance.for_range(0, Constant.BOX_LOC) as idx:
                with self.tik_instance.for_range(0, self.task_num) as task_idx:
                    self.tik_instance.data_move(dbboxes_tmp,
                                                self.dbboxes_[idx * self.all_num_align + task_idx * self.data_align],
                                                0, 1, self.data_align // Constant.BLOCK, 0, 0)
                    self.tik_instance.data_move(self.dbboxes[idx * self.all_num + task_idx * self.data_align],
                                                dbboxes_tmp, 0, 1, self.data_align // Constant.BLOCK, 0, 0)

            # func: Address fallback for dgtboxes
            with self.tik_instance.for_range(0, Constant.BOX_LOC) as idx:
                with self.tik_instance.for_range(0, self.task_num) as task_idx:
                    self.tik_instance.data_move(gtboxes_tmp,
                                                self.dgtboxes_[idx * self.all_num_align + task_idx * self.data_align],
                                                0, 1, self.data_align // Constant.BLOCK, 0, 0)
                    self.tik_instance.data_move(self.dgtboxes[idx * self.all_num + task_idx * self.data_align],
                                                gtboxes_tmp, 0, 1, self.data_align // Constant.BLOCK, 0, 0)
