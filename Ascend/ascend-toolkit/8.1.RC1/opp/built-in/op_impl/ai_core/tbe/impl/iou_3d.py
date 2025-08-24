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
iou_3d
"""

from tbe.tvm.topi.cce import util
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import register_operator
from tbe.common.platform.platform_info import api_check_support


class Constant(object):
    """
    The class for constant.
    """
    # min val in float16
    MIN_VAL = -65504
    # float32 data'nums in 32B
    BLOCK = 8
    # float16 data'nums in 32B
    BATCH = 16
    # the minimum decimal of fp32
    FP32_MIN_VAL = -3.4e38
    # the batch size of int32
    INT32_BATCH = 32
    # idx tag for {x, y, z, w, h, d, theta}
    X_IDX = 0
    Y_IDX = 1
    Z_IDX = 2
    W_IDX = 3
    H_IDX = 4
    D_IDX = 5
    T_IDX = 6
    # nums of box info
    INFOS = 7
    # val's idx in proposal
    # val's idx in proposal
    VAL_IDX = 4
    # limit of k's size of query_boxes
    K_LIMIT = 2000
    # to avoid denominator zero
    EPSILON = 1e-6
    UNIT = 19


# 'pylint: disable=locally-disabled, unused-argument, invalid-name
class Iou3D:
    """
    The class for Iou3D.
    """

    # 'pylint:disable=too-many-arguments, disable=too-many-statements
    def __init__(self, boxes, query_boxes, iou, kernel_name):
        """
        class init
        """
        self.tik_instance = tik.Tik(tik.Dprofile())
        check_res = Iou3D.paras_check(boxes, query_boxes, iou, kernel_name)
        self.batch, self.n, self.k, self.dtype = check_res[0], check_res[1], check_res[2], check_res[3]
        self.kernel_name, self.task_num = kernel_name, self.n
        while self.task_num % 2 == 0 and self.task_num >= 64:
            self.task_num //= 2
        self.b1_batch = self.n // self.task_num
        if self.b1_batch >= Constant.BLOCK:
            self.b1_repeats = (self.b1_batch + Constant.BLOCK - 1) // Constant.BLOCK
        else:
            self.task_num, self.b1_batch, self.b1_repeats = self.n, 1, 1
        self.k_align = (self.k + Constant.BLOCK - 1 + self.b1_batch) // Constant.BLOCK * Constant.BLOCK
        self.repeats = self.k_align // Constant.BLOCK
        self.available_aicore_num = tik.Dprofile().get_aicore_num()
        self.used_aicore_num = self.available_aicore_num if self.task_num > self.available_aicore_num \
            else self.task_num
        self.batch_num_per_aicore = self.task_num // self.used_aicore_num
        self.batch_tail = self.task_num % self.used_aicore_num
        self.boxes_gm = self.tik_instance.Tensor(self.dtype, [self.batch, Constant.INFOS, self.n], name="boxes_gm",
                                                 scope=tik.scope_gm)
        self.query_boxes_gm = self.tik_instance.Tensor(self.dtype, [self.batch, Constant.INFOS, self.k],
                                                       name="query_boxes_gm", scope=tik.scope_gm)
        self.iou_gm = self.tik_instance.Tensor(self.dtype, [self.batch, self.n, self.k], name="iou_gm",
                                                    scope=tik.scope_gm, is_atomic_add=True)
        lis = [None] * Constant.UNIT
        self.is_old_version = True if not api_check_support("tik.vreducev2") else False
        if self.is_old_version:
            self.changed_shape = Constant.BATCH
        else:
            self.changed_shape = Constant.INT32_BATCH
        self.ori_idx_uint32_ub = None
        self.min_val = None
        self.value, self.b1_x1, self.b1_y1, self.b2_x1, self.b2_y1, self.b1_x2, self.b1_y2, self.b2_x2, self.b2_y2,\
            self.b1_x3, self.b1_y3, self.AD_x, self.AD_y, self.AP_x, self.AP_y, self.BC_x, self.BC_y, self.BD_x,\
            self.BD_y = lis

    @staticmethod
    def paras_check(boxes, query_boxes, overlaps, kernel_name):
        """
        Check parameters
        """
        util.check_kernel_name(kernel_name)
        shape_boxes = boxes.get("shape")
        dtype_boxes = boxes.get("dtype").lower()
        util.check_shape_rule(shape_boxes)
        util.check_dtype_rule(dtype_boxes, "float32")
        shape_query_boxes = query_boxes.get("shape")
        dtype_query_boxes = query_boxes.get("dtype").lower()
        util.check_shape_rule(shape_query_boxes)
        util.check_dtype_rule(dtype_query_boxes, "float32")

        shape_overlaps = overlaps.get("shape")
        dtype_overlaps = overlaps.get("dtype").lower()
        util.check_shape_rule(shape_overlaps)
        util.check_dtype_rule(dtype_overlaps, "float32")
        if shape_query_boxes[2] != shape_overlaps[2]:
            raise RuntimeError("Shape unmatch in query_boxes nums")
        if shape_boxes[1] != Constant.INFOS:
            raise RuntimeError("Shape of boxes should be [-1, 7,-1].")
        if shape_query_boxes[1] != Constant.INFOS:
            raise RuntimeError("Shape of query_boxes should be [-1, 7, -1].")
        if shape_query_boxes[2] > Constant.K_LIMIT:
            raise RuntimeError("K's value is over 2000.")
        return [shape_boxes[0], shape_overlaps[1], shape_overlaps[2], dtype_boxes]

    def get_area_of_triangle(self, idx_tmp, idx_current_tmp, corners_ub):
        """
        Calculating triangle area based on vertex coordinates.
        """
        self.b1_x2.set_as(corners_ub[idx_tmp])
        self.b1_y2.set_as(corners_ub[idx_tmp + Constant.BLOCK])
        self.b1_x3.set_as(corners_ub[idx_current_tmp])
        self.b1_y3.set_as(corners_ub[idx_current_tmp + Constant.BLOCK])

        self.value.set_as(
            self.b1_x1 * (self.b1_y2 - self.b1_y3) + self.b1_x2 * (self.b1_y3 - self.b1_y1) + self.b1_x3 * (
                    self.b1_y1 - self.b1_y2))

    def sum_area_of_triangles(self, ori_idx_fp16_ub, x_tensor_ub, y_tensor_ub, slope_tensor_ub,
                              add_tensor_ub, abs_tensor_ub, work_tensor_ub,
                              clockwise_idx_int32_ub, corners_ub, corners_num):
        """
        Calculate the sum of the areas of the triangles
        """
        if self.is_old_version:
            val_fp16_ub = self.tik_instance.Tensor("float16", [self.changed_shape], name="val_fp16_ub",
                                                   scope=tik.scope_ubuf)
            
            idx_fp16_ub = self.tik_instance.Tensor("float16", [Constant.BATCH], name="idx_fp16_ub",
                                                   scope=tik.scope_ubuf)
            proposal_ub = self.tik_instance.Tensor("float16", [2, Constant.BATCH, Constant.BLOCK],
                                               name="proposal_ub", scope=tik.scope_ubuf)
        
        idx_int32_ub = self.tik_instance.Tensor("int32", [Constant.BATCH], name="idx_int32_ub",
                                                scope=tik.scope_ubuf)
        
        min_val = self.tik_instance.Scalar('float16', init_value=Constant.MIN_VAL)
        idx_right = self.tik_instance.Scalar("int32")
        idx_left = self.tik_instance.Scalar("int32")
        self.tik_instance.vec_reduce_add(corners_num, add_tensor_ub, corners_ub, work_tensor_ub, 1,
                                         1)
        self.b1_x1.set_as(add_tensor_ub[0])
        self.b1_x1.set_as(self.b1_x1 / corners_num)
        self.tik_instance.vec_reduce_add(corners_num, add_tensor_ub, corners_ub[Constant.BLOCK],
                                         work_tensor_ub, 1, 1)
        self.b1_y1.set_as(add_tensor_ub[0])
        self.b1_y1.set_as(self.b1_y1 / corners_num)

        self.tik_instance.data_move(x_tensor_ub, corners_ub, 0, 1, 1, 0, 0)
        self.tik_instance.data_move(y_tensor_ub, corners_ub[Constant.BLOCK], 0, 1, 1, 0, 0)

        self.tik_instance.h_sub(x_tensor_ub, x_tensor_ub, self.b1_x1)
        self.tik_instance.h_sub(y_tensor_ub, y_tensor_ub, self.b1_y1)
        self.tik_instance.h_div(slope_tensor_ub, y_tensor_ub, x_tensor_ub)
        
        if self.is_old_version:
            self.tik_instance.h_cast(val_fp16_ub, slope_tensor_ub, "none")
            self.min_val = self.tik_instance.Scalar('float16', init_value=Constant.MIN_VAL)
            with self.tik_instance.for_range(corners_num, Constant.BATCH) as idx:
                val_fp16_ub[idx].set_as(self.min_val)
            self.tik_instance.vconcat(proposal_ub, val_fp16_ub, 1, Constant.VAL_IDX)
            self.tik_instance.vconcat(proposal_ub, ori_idx_fp16_ub, 1, 0)

            # Sort slopes in descending order
            self.tik_instance.vrpsort16(proposal_ub[Constant.BATCH * Constant.BLOCK], proposal_ub, 1)
            self.tik_instance.vextract(idx_fp16_ub, proposal_ub[Constant.BATCH * Constant.BLOCK], 1, 0)
            self.tik_instance.h_cast(idx_int32_ub, idx_fp16_ub, "round")

        else:
            self.min_val = self.tik_instance.Scalar('float32', init_value=Constant.FP32_MIN_VAL)
            with self.tik_instance.for_range(corners_num, Constant.INT32_BATCH) as idx:
                slope_tensor_ub[idx].set_as(self.min_val)
            res_ub_tmp = self.tik_instance.Tensor("float32", [Constant.INT32_BATCH, 2], name="res_ub_tmp",
                                                  scope=tik.scope_ubuf)
            self.tik_instance.vsort32(res_ub_tmp, slope_tensor_ub, self.ori_idx_uint32_ub, 1)
            uint32_ub = res_ub_tmp.reinterpret_cast_to("uint32")
            tmp = self.tik_instance.Scalar("uint32")
            with self.tik_instance.for_range(0, corners_num) as i2:
                tmp.set_as(uint32_ub[2 * i2 + 1])
                idx_int32_ub[i2].set_as(tmp)

        idx_left.set_as(0)
        idx_right.set_as(0)
        self.cal_area(idx_int32_ub, x_tensor_ub, y_tensor_ub, add_tensor_ub, abs_tensor_ub,
                      work_tensor_ub, clockwise_idx_int32_ub, idx_right, idx_left, corners_ub, corners_num)

    def cal_area(self, idx_int32_ub, x_tensor_ub, y_tensor_ub, add_tensor_ub, abs_tensor_ub, work_tensor_ub,
                 clockwise_idx_int32_ub, idx_right, idx_left, corners_ub, corners_num):
        """
        Calculate the area of a triangle
        """
        idx_current_tmp = self.tik_instance.Scalar("int32")
        b1_x = self.tik_instance.Scalar(self.dtype)
        b1_y = self.tik_instance.Scalar(self.dtype)
        with self.tik_instance.for_range(0, corners_num) as idx:
            idx_current_tmp.set_as(idx_int32_ub[idx])
            b1_x.set_as(x_tensor_ub[idx_current_tmp])
            with self.tik_instance.if_scope(b1_x < 0):
                clockwise_idx_int32_ub[idx_left].set_as(idx_current_tmp)
                idx_left.set_as(idx_left + 1)
            with self.tik_instance.elif_scope(b1_x > 0):
                clockwise_idx_int32_ub[idx_right + Constant.BLOCK].set_as(idx_current_tmp)
                idx_right.set_as(idx_right + 1)
            with self.tik_instance.else_scope():
                b1_y.set_as(y_tensor_ub[idx_current_tmp])
                with self.tik_instance.if_scope(b1_y < 0):
                    clockwise_idx_int32_ub[idx_left].set_as(idx_current_tmp)
                    idx_left.set_as(idx_left + 1)
                with self.tik_instance.else_scope():
                    clockwise_idx_int32_ub[idx_right + Constant.BLOCK].set_as(idx_current_tmp)
                    idx_right.set_as(idx_right + 1)

        idx_tmp = self.tik_instance.Scalar("int32")
        idx_tmp.set_as(clockwise_idx_int32_ub[0])
        with self.tik_instance.for_range(1, idx_left) as l_idx:
            idx_current_tmp.set_as(clockwise_idx_int32_ub[l_idx])
            self.get_area_of_triangle(idx_tmp, idx_current_tmp, corners_ub)
            add_tensor_ub[l_idx].set_as(self.value)
            idx_tmp.set_as(idx_current_tmp)
        with self.tik_instance.for_range(0, idx_right) as r_idx:
            idx_current_tmp.set_as(clockwise_idx_int32_ub[r_idx + Constant.BLOCK])
            self.get_area_of_triangle(idx_tmp, idx_current_tmp, corners_ub)
            add_tensor_ub[r_idx + idx_left].set_as(self.value)
            idx_tmp.set_as(idx_current_tmp)
        idx_current_tmp.set_as(clockwise_idx_int32_ub[0])
        self.get_area_of_triangle(idx_tmp, idx_current_tmp, corners_ub)
        add_tensor_ub[0].set_as(self.value)
        self.tik_instance.h_abs(abs_tensor_ub, add_tensor_ub)
        self.tik_instance.vec_reduce_add(corners_num, add_tensor_ub, abs_tensor_ub, work_tensor_ub,
                                         1, 1)
        self.value.set_as(add_tensor_ub[0])

    def record_intersection_point_core(self, corners_ub, corners_num):
        """
        Each kernel comes up to record the intersection of two cubes
        """
        AC_x = self.tik_instance.Scalar(self.dtype)
        AC_y = self.tik_instance.Scalar(self.dtype)
        direct_AC_AD = self.tik_instance.Scalar(self.dtype)
        direct_BC_BD = self.tik_instance.Scalar(self.dtype)
        direct_CA_CB = self.tik_instance.Scalar(self.dtype)
        direct_DA_DB = self.tik_instance.Scalar(self.dtype)
        tmp_1 = self.tik_instance.Scalar(self.dtype)
        tmp_2 = self.tik_instance.Scalar(self.dtype)
        b1_x1_x2 = self.tik_instance.Scalar(self.dtype)
        b1_y1_y2 = self.tik_instance.Scalar(self.dtype)
        b2_x1_x2 = self.tik_instance.Scalar(self.dtype)
        b2_y1_y2 = self.tik_instance.Scalar(self.dtype)
        denominator = self.tik_instance.Scalar(self.dtype)
        numerator_x = self.tik_instance.Scalar(self.dtype)
        numerator_y = self.tik_instance.Scalar(self.dtype)
        AC_x.set_as(self.b2_x1 - self.b1_x1)
        AC_y.set_as(self.b2_y1 - self.b1_y1)
        self.AD_x.set_as(self.b2_x2 - self.b1_x1)
        self.AD_y.set_as(self.b2_y2 - self.b1_y1)
        self.BC_x.set_as(self.b2_x1 - self.b1_x2)
        self.BC_y.set_as(self.b2_y1 - self.b1_y2)
        self.BD_x.set_as(self.b2_x2 - self.b1_x2)
        self.BD_y.set_as(self.b2_y2 - self.b1_y2)

        # Check for intersection between two edges
        direct_AC_AD.set_as(AC_x * self.AD_y - AC_y * self.AD_x)
        direct_BC_BD.set_as(self.BC_x * self.BD_y - self.BC_y * self.BD_x)
        with self.tik_instance.if_scope(direct_AC_AD * direct_BC_BD < 0):
            direct_CA_CB.set_as(AC_x * self.BC_y - AC_y * self.BC_x)
            direct_DA_DB.set_as(self.AD_x * self.BD_y - self.AD_y * self.BD_x)
            with self.tik_instance.if_scope(direct_CA_CB * direct_DA_DB < 0):
                tmp_1.set_as(self.b1_x1 * self.b1_y2 - self.b1_y1 * self.b1_x2)
                tmp_2.set_as(self.b2_x1 * self.b2_y2 - self.b2_y1 * self.b2_x2)
                b1_x1_x2.set_as(self.b1_x1 - self.b1_x2)
                b1_y1_y2.set_as(self.b1_y1 - self.b1_y2)
                b2_x1_x2.set_as(self.b2_x1 - self.b2_x2)
                b2_y1_y2.set_as(self.b2_y1 - self.b2_y2)
                denominator.set_as(b2_x1_x2 * b1_y1_y2 - b1_x1_x2 * b2_y1_y2)
                numerator_x.set_as(b1_x1_x2 * tmp_2 - tmp_1 * b2_x1_x2)
                numerator_y.set_as(b1_y1_y2 * tmp_2 - tmp_1 * b2_y1_y2)
                corners_ub[corners_num].set_as(numerator_x / denominator)
                corners_ub[corners_num + Constant.BLOCK].set_as(numerator_y / denominator)
                corners_num.set_as(corners_num + 1)

    def record_intersection_point_compute(self, x1_of_boxes_ub, x2_of_boxes_ub, x3_of_boxes_ub, x4_of_boxes_ub,
                                          y1_of_boxes_ub, y2_of_boxes_ub, y3_of_boxes_ub, y4_of_boxes_ub,
                                          corners_ub, corners_num, b1_offset):
        """
        The specific process of calculating the intersection point
        """
        self.b1_x2.set_as(x2_of_boxes_ub[b1_offset])
        self.b1_y2.set_as(y2_of_boxes_ub[b1_offset])
        self.record_intersection_point_core(corners_ub, corners_num)
        self.b1_x1.set_as(x3_of_boxes_ub[b1_offset])
        self.b1_y1.set_as(y3_of_boxes_ub[b1_offset])
        self.record_intersection_point_core(corners_ub, corners_num)
        self.b1_x2.set_as(x4_of_boxes_ub[b1_offset])
        self.b1_y2.set_as(y4_of_boxes_ub[b1_offset])
        self.record_intersection_point_core(corners_ub, corners_num)
        self.b1_x1.set_as(x1_of_boxes_ub[b1_offset])
        self.b1_y1.set_as(y1_of_boxes_ub[b1_offset])
        self.record_intersection_point_core(corners_ub, corners_num)

    def record_intersection_point(self, b2_idx, x1_of_boxes_ub, x2_of_boxes_ub, x3_of_boxes_ub, x4_of_boxes_ub,
                                  y1_of_boxes_ub, y2_of_boxes_ub, y3_of_boxes_ub,
                                  y4_of_boxes_ub, corners_ub, corners_num, b1_offset):
        """
        Calling function for intersection calculation
        """
        # Calculate the intersection
        self.record_intersection_point_core(corners_ub, corners_num)
        self.b1_x1.set_as(x3_of_boxes_ub[b1_offset])
        self.b1_y1.set_as(y3_of_boxes_ub[b1_offset])
        self.record_intersection_point_core(corners_ub, corners_num)
        self.b1_x2.set_as(x4_of_boxes_ub[b1_offset])
        self.b1_y2.set_as(y4_of_boxes_ub[b1_offset])
        self.record_intersection_point_core(corners_ub, corners_num)
        self.b1_x1.set_as(x1_of_boxes_ub[b1_offset])
        self.b1_y1.set_as(y1_of_boxes_ub[b1_offset])
        self.record_intersection_point_core(corners_ub, corners_num)

        self.b2_x1.set_as(x3_of_boxes_ub[b2_idx])
        self.b2_y1.set_as(y3_of_boxes_ub[b2_idx])
        self.record_intersection_point_compute(x1_of_boxes_ub, x2_of_boxes_ub, x3_of_boxes_ub, x4_of_boxes_ub,
                                               y1_of_boxes_ub, y2_of_boxes_ub, y3_of_boxes_ub, y4_of_boxes_ub,
                                               corners_ub, corners_num, b1_offset)
        self.b2_x2.set_as(x4_of_boxes_ub[b2_idx])
        self.b2_y2.set_as(y4_of_boxes_ub[b2_idx])
        self.record_intersection_point_compute(x1_of_boxes_ub, x2_of_boxes_ub, x3_of_boxes_ub, x4_of_boxes_ub,
                                               y1_of_boxes_ub, y2_of_boxes_ub, y3_of_boxes_ub, y4_of_boxes_ub,
                                               corners_ub, corners_num, b1_offset)
        self.b2_x1.set_as(x1_of_boxes_ub[b2_idx])
        self.b2_y1.set_as(y1_of_boxes_ub[b2_idx])
        self.record_intersection_point_compute(x1_of_boxes_ub, x2_of_boxes_ub, x3_of_boxes_ub, x4_of_boxes_ub,
                                               y1_of_boxes_ub, y2_of_boxes_ub, y3_of_boxes_ub, y4_of_boxes_ub,
                                               corners_ub, corners_num, b1_offset)

    def record_vertex_point(self, b2_idx, x1_of_boxes_ub, x2_of_boxes_ub, x3_of_boxes_ub, x4_of_boxes_ub,
                            y1_of_boxes_ub, y2_of_boxes_ub, y3_of_boxes_ub, y4_of_boxes_ub, corners_ub,
                            corners_num, b1_offset):
        """
        Compute the intersection of convex sets
        """
        corners_num.set_as(0)
        b2_x3 = self.tik_instance.Scalar(self.dtype)
        b2_y3 = self.tik_instance.Scalar(self.dtype)
        b1_x4 = self.tik_instance.Scalar(self.dtype)
        b1_y4 = self.tik_instance.Scalar(self.dtype)
        b2_x4 = self.tik_instance.Scalar(self.dtype)
        b2_y4 = self.tik_instance.Scalar(self.dtype)
        AB_x = self.tik_instance.Scalar(self.dtype)
        AB_y = self.tik_instance.Scalar(self.dtype)
        AP_AB = self.tik_instance.Scalar(self.dtype)
        AB_AB = self.tik_instance.Scalar(self.dtype)
        AD_AD = self.tik_instance.Scalar(self.dtype)
        AP_AD = self.tik_instance.Scalar(self.dtype)
        # func: b1 for input boxes & b2 for input query_boxes
        self.b1_x1.set_as(x1_of_boxes_ub[b1_offset])
        self.b1_x2.set_as(x2_of_boxes_ub[b1_offset])
        self.b1_x3.set_as(x3_of_boxes_ub[b1_offset])
        b1_x4.set_as(x4_of_boxes_ub[b1_offset])
        self.b2_x1.set_as(x1_of_boxes_ub[b2_idx])
        self.b2_x2.set_as(x2_of_boxes_ub[b2_idx])
        b2_x3.set_as(x3_of_boxes_ub[b2_idx])
        b2_x4.set_as(x4_of_boxes_ub[b2_idx])

        self.b1_y1.set_as(y1_of_boxes_ub[b1_offset])
        self.b1_y2.set_as(y2_of_boxes_ub[b1_offset])
        self.b1_y3.set_as(y3_of_boxes_ub[b1_offset])
        b1_y4.set_as(y4_of_boxes_ub[b1_offset])
        self.b2_y1.set_as(y1_of_boxes_ub[b2_idx])
        self.b2_y2.set_as(y2_of_boxes_ub[b2_idx])
        b2_y3.set_as(y3_of_boxes_ub[b2_idx])
        b2_y4.set_as(y4_of_boxes_ub[b2_idx])
        self.check_first_rectangle(b1_x4, b1_y4, b2_x4, b2_y4, AB_x, AB_y, AP_AB, AB_AB, AD_AD, AP_AD,
                                   corners_ub, corners_num)
        self.check_second_rectangle(b2_x3, b2_y3, b1_x4, b1_y4, b2_x4, b2_y4, AB_x, AB_y, AP_AB, AB_AB, AD_AD,
                                    AP_AD, corners_ub, corners_num)

    def check_first_rectangle(self, b1_x4, b1_y4, b2_x4, b2_y4, AB_x, AB_y, AP_AB, AB_AB, AD_AD, AP_AD,
                              corners_ub, corners_num):
        """
        Check the vertices of the first rectangle
        """
        # Check if the vertices of the first rectangular box are inside the convex set
        AB_x.set_as(self.b2_x2 - self.b2_x1)
        AB_y.set_as(self.b2_y2 - self.b2_y1)
        self.AD_x.set_as(b2_x4 - self.b2_x1)
        self.AD_y.set_as(b2_y4 - self.b2_y1)
        AB_AB.set_as(AB_x * AB_x + AB_y * AB_y)
        AD_AD.set_as(self.AD_x * self.AD_x + self.AD_y * self.AD_y)

        self.AP_x.set_as(self.b1_x1 - self.b2_x1)
        self.AP_y.set_as(self.b1_y1 - self.b2_y1)
        AP_AB.set_as(self.AP_x * AB_x + self.AP_y * AB_y)
        with self.tik_instance.if_scope(tik.all(AP_AB >= 0, AP_AB <= AB_AB)):
            AP_AD.set_as(self.AP_x * self.AD_x + self.AP_y * self.AD_y)
            with self.tik_instance.if_scope(tik.all(AP_AD >= 0, AP_AD <= AD_AD)):
                corners_ub[corners_num].set_as(self.b1_x1)
                corners_ub[corners_num + Constant.BLOCK].set_as(self.b1_y1)
                corners_num.set_as(corners_num + 1)

        self.AP_x.set_as(self.b1_x2 - self.b2_x1)
        self.AP_y.set_as(self.b1_y2 - self.b2_y1)
        AP_AB.set_as(self.AP_x * AB_x + self.AP_y * AB_y)
        with self.tik_instance.if_scope(tik.all(AP_AB >= 0, AP_AB <= AB_AB)):
            AP_AD.set_as(self.AP_x * self.AD_x + self.AP_y * self.AD_y)
            with self.tik_instance.if_scope(tik.all(AP_AD >= 0, AP_AD <= AD_AD)):
                corners_ub[corners_num].set_as(self.b1_x2)
                corners_ub[corners_num + Constant.BLOCK].set_as(self.b1_y2)
                corners_num.set_as(corners_num + 1)

        self.AP_x.set_as(self.b1_x3 - self.b2_x1)
        self.AP_y.set_as(self.b1_y3 - self.b2_y1)
        AP_AB.set_as(self.AP_x * AB_x + self.AP_y * AB_y)
        with self.tik_instance.if_scope(tik.all(AP_AB >= 0, AP_AB <= AB_AB)):
            AP_AD.set_as(self.AP_x * self.AD_x + self.AP_y * self.AD_y)
            with self.tik_instance.if_scope(tik.all(AP_AD >= 0, AP_AD <= AD_AD)):
                corners_ub[corners_num].set_as(self.b1_x3)
                corners_ub[corners_num + Constant.BLOCK].set_as(self.b1_y3)
                corners_num.set_as(corners_num + 1)

        self.AP_x.set_as(b1_x4 - self.b2_x1)
        self.AP_y.set_as(b1_y4 - self.b2_y1)
        AP_AB.set_as(self.AP_x * AB_x + self.AP_y * AB_y)
        with self.tik_instance.if_scope(tik.all(AP_AB >= 0, AP_AB <= AB_AB)):
            AP_AD.set_as(self.AP_x * self.AD_x + self.AP_y * self.AD_y)
            with self.tik_instance.if_scope(tik.all(AP_AD >= 0, AP_AD <= AD_AD)):
                corners_ub[corners_num].set_as(b1_x4)
                corners_ub[corners_num + Constant.BLOCK].set_as(b1_y4)
                corners_num.set_as(corners_num + 1)

    def check_second_rectangle(self, b2_x3, b2_y3, b1_x4, b1_y4, b2_x4, b2_y4, AB_x, AB_y, AP_AB,
                               AB_AB, AD_AD, AP_AD, corners_ub, corners_num):
        """
        Check the vertices of the second rectangle
        """
        AB_x.set_as(self.b1_x2 - self.b1_x1)
        AB_y.set_as(self.b1_y2 - self.b1_y1)
        self.AD_x.set_as(b1_x4 - self.b1_x1)
        self.AD_y.set_as(b1_y4 - self.b1_y1)
        AB_AB.set_as(AB_x * AB_x + AB_y * AB_y)
        AD_AD.set_as(self.AD_x * self.AD_x + self.AD_y * self.AD_y)

        # Check if the vertices of the second rectangular box are inside the convex set
        self.AP_x.set_as(self.b2_x1 - self.b1_x1)
        self.AP_y.set_as(self.b2_y1 - self.b1_y1)
        AP_AB.set_as(self.AP_x * AB_x + self.AP_y * AB_y)
        with self.tik_instance.if_scope(tik.all(AP_AB >= 0, AP_AB <= AB_AB)):
            AP_AD.set_as(self.AP_x * self.AD_x + self.AP_y * self.AD_y)
            with self.tik_instance.if_scope(tik.all(AP_AD >= 0, AP_AD <= AD_AD)):
                corners_ub[corners_num].set_as(self.b2_x1)
                corners_ub[corners_num + Constant.BLOCK].set_as(self.b2_y1)
                corners_num.set_as(corners_num + 1)

        self.AP_x.set_as(self.b2_x2 - self.b1_x1)
        self.AP_y.set_as(self.b2_y2 - self.b1_y1)
        AP_AB.set_as(self.AP_x * AB_x + self.AP_y * AB_y)
        with self.tik_instance.if_scope(tik.all(AP_AB >= 0, AP_AB <= AB_AB)):
            AP_AD.set_as(self.AP_x * self.AD_x + self.AP_y * self.AD_y)
            with self.tik_instance.if_scope(tik.all(AP_AD >= 0, AP_AD <= AD_AD)):
                corners_ub[corners_num].set_as(self.b2_x2)
                corners_ub[corners_num + Constant.BLOCK].set_as(self.b2_y2)
                corners_num.set_as(corners_num + 1)

        self.AP_x.set_as(b2_x3 - self.b1_x1)
        self.AP_y.set_as(b2_y3 - self.b1_y1)
        AP_AB.set_as(self.AP_x * AB_x + self.AP_y * AB_y)
        with self.tik_instance.if_scope(tik.all(AP_AB >= 0, AP_AB <= AB_AB)):
            AP_AD.set_as(self.AP_x * self.AD_x + self.AP_y * self.AD_y)
            with self.tik_instance.if_scope(tik.all(AP_AD >= 0, AP_AD <= AD_AD)):
                corners_ub[corners_num].set_as(b2_x3)
                corners_ub[corners_num + Constant.BLOCK].set_as(b2_y3)
                corners_num.set_as(corners_num + 1)

        self.AP_x.set_as(b2_x4 - self.b1_x1)
        self.AP_y.set_as(b2_y4 - self.b1_y1)
        AP_AB.set_as(self.AP_x * AB_x + self.AP_y * AB_y)
        with self.tik_instance.if_scope(tik.all(AP_AB >= 0, AP_AB <= AB_AB)):
            AP_AD.set_as(self.AP_x * self.AD_x + self.AP_y * self.AD_y)
            with self.tik_instance.if_scope(tik.all(AP_AD >= 0, AP_AD <= AD_AD)):
                corners_ub[corners_num].set_as(b2_x4)
                corners_ub[corners_num + Constant.BLOCK].set_as(b2_y4)
                corners_num.set_as(corners_num + 1)

    def get_effective_depth(self, task_idx, current_batch, d_of_boxes_ub, z_sub_d_boxes_ub,
                            z_add_d_boxes_ub, tmp_tensor_ub):
        """
        Get the effective depth of the intersecting volume
        """
        half = self.tik_instance.Scalar(self.dtype, init_value=0.5)
        z_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="z_of_boxes_ub",
                                                 scope=tik.scope_ubuf)
        half_d_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="half_d_of_boxes_ub",
                                                      scope=tik.scope_ubuf)
        self.tik_instance.data_move(
            d_of_boxes_ub, self.query_boxes_gm[self.k * (Constant.D_IDX + current_batch * Constant.INFOS)],
            0, 1, self.repeats, 0, 0)
        self.tik_instance.data_move(
            z_of_boxes_ub, self.query_boxes_gm[self.k * (Constant.Z_IDX + current_batch * Constant.INFOS)],
            0, 1, self.repeats, 0, 0)
        if self.b1_batch == 1:
            self.tik_instance.data_move(
                tmp_tensor_ub,
                self.boxes_gm[
                    self.n * Constant.D_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
            d_of_boxes_ub[self.k_align - self.b1_batch].set_as(tmp_tensor_ub[0])
            self.tik_instance.data_move(
                tmp_tensor_ub,
                self.boxes_gm[
                    self.n * Constant.Z_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
            z_of_boxes_ub[self.k_align - self.b1_batch].set_as(tmp_tensor_ub[0])
        else:
            self.tik_instance.data_move(
                d_of_boxes_ub[self.k_align - self.b1_batch],
                self.boxes_gm[
                    self.n * Constant.D_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
            self.tik_instance.data_move(
                z_of_boxes_ub[self.k_align - self.b1_batch],
                self.boxes_gm[
                    self.n * Constant.Z_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
        self.tik_instance.h_mul(half_d_of_boxes_ub, d_of_boxes_ub, half)
        self.tik_instance.h_sub(z_sub_d_boxes_ub, z_of_boxes_ub, half_d_of_boxes_ub)
        self.tik_instance.h_add(z_add_d_boxes_ub, z_of_boxes_ub, half_d_of_boxes_ub)

    def trans_boxes(self, task_idx, current_batch, w_of_boxes_ub, h_of_boxes_ub, half_w_of_boxes_ub,
                    half_h_of_boxes_ub, cos_t_of_boxes_ub, sin_t_of_boxes_ub, x1_of_boxes_ub, x2_of_boxes_ub,
                    x3_of_boxes_ub, x4_of_boxes_ub, y1_of_boxes_ub, y2_of_boxes_ub,
                    y3_of_boxes_ub, y4_of_boxes_ub, tmp_tensor_ub):
        """
        Calculate the coordinates of the rotated box
        """
        # theta
        x_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="x_of_boxes_ub",
                                                 scope=tik.scope_ubuf)
        y_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="y_of_boxes_ub",
                                                 scope=tik.scope_ubuf)
        t_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="t_of_boxes_ub",
                                                 scope=tik.scope_ubuf)
        radian_t_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="radian_t_of_boxes_ub",
                                                        scope=tik.scope_ubuf)
        self.tik_instance.data_move(
            t_of_boxes_ub, self.query_boxes_gm[self.k * (Constant.T_IDX + current_batch * Constant.INFOS)], 0,
            1, self.repeats, 0, 0)
        if self.b1_batch == 1:
            self.tik_instance.data_move(
                tmp_tensor_ub,
                self.boxes_gm[
                    self.n * Constant.T_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
            t_of_boxes_ub[self.k_align - self.b1_batch].set_as(tmp_tensor_ub[0])
        else:
            self.tik_instance.data_move(
                t_of_boxes_ub[self.k_align - self.b1_batch],
                self.boxes_gm[
                    self.n * Constant.T_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)

        radian = self.tik_instance.Scalar(self.dtype, init_value=1)
        self.tik_instance.h_mul(radian_t_of_boxes_ub, t_of_boxes_ub, radian)
        self.tik_instance.h_sin(sin_t_of_boxes_ub, radian_t_of_boxes_ub)
        self.tik_instance.h_cos(cos_t_of_boxes_ub, radian_t_of_boxes_ub)
        self.data_move_with_w_and_h(current_batch, task_idx, w_of_boxes_ub, h_of_boxes_ub, half_w_of_boxes_ub,
                                    half_h_of_boxes_ub, tmp_tensor_ub)
        self.data_move_with_x_and_y(current_batch, task_idx, x_of_boxes_ub, y_of_boxes_ub, tmp_tensor_ub)
        self.cal_coordinate_with_rotate(x_of_boxes_ub, y_of_boxes_ub, half_w_of_boxes_ub, half_h_of_boxes_ub,
                                        cos_t_of_boxes_ub, sin_t_of_boxes_ub, x1_of_boxes_ub, x2_of_boxes_ub,
                                        x3_of_boxes_ub, x4_of_boxes_ub, y1_of_boxes_ub, y2_of_boxes_ub, y3_of_boxes_ub,
                                        y4_of_boxes_ub)

    def data_move_with_w_and_h(self, current_batch, task_idx, w_of_boxes_ub, h_of_boxes_ub, half_w_of_boxes_ub,
                               half_h_of_boxes_ub, tmp_tensor_ub):
        """
        Move the width and height of the cuboid
        """
        half = self.tik_instance.Scalar(self.dtype, init_value=0.5)
        self.tik_instance.data_move(
            w_of_boxes_ub, self.query_boxes_gm[self.k * (Constant.W_IDX + current_batch * Constant.INFOS)],
            0, 1, self.repeats, 0, 0)
        self.tik_instance.data_move(
            h_of_boxes_ub, self.query_boxes_gm[self.k * (Constant.H_IDX + current_batch * Constant.INFOS)],
            0, 1, self.repeats, 0, 0)

        if self.b1_batch == 1:
            self.tik_instance.data_move(
                tmp_tensor_ub,
                self.boxes_gm[
                    self.n * Constant.W_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
            w_of_boxes_ub[self.k_align - self.b1_batch].set_as(tmp_tensor_ub[0])
            self.tik_instance.data_move(
                tmp_tensor_ub,
                self.boxes_gm[
                    self.n * Constant.H_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
            h_of_boxes_ub[self.k_align - self.b1_batch].set_as(tmp_tensor_ub[0])
        else:
            self.tik_instance.data_move(
                w_of_boxes_ub[self.k_align - self.b1_batch],
                self.boxes_gm[
                    self.n * Constant.W_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
            self.tik_instance.data_move(
                h_of_boxes_ub[self.k_align - self.b1_batch],
                self.boxes_gm[
                    self.n * Constant.H_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
        self.tik_instance.h_mul(half_w_of_boxes_ub, w_of_boxes_ub, half)
        self.tik_instance.h_mul(half_h_of_boxes_ub, h_of_boxes_ub, half)

    def data_move_with_x_and_y(self, current_batch, task_idx, x_of_boxes_ub, y_of_boxes_ub, tmp_tensor_ub):
        """
        Move the x and y coordinates of the center point of the cuboid
        """
        self.tik_instance.data_move(
            x_of_boxes_ub, self.query_boxes_gm[self.k * (Constant.X_IDX + current_batch * Constant.INFOS)],
            0, 1, self.repeats, 0, 0)
        self.tik_instance.data_move(
            y_of_boxes_ub, self.query_boxes_gm[self.k * (Constant.Y_IDX + current_batch * Constant.INFOS)],
            0, 1, self.repeats, 0, 0)

        if self.b1_batch == 1:
            self.tik_instance.data_move(tmp_tensor_ub, self.boxes_gm[
                    self.n * Constant.X_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
            x_of_boxes_ub[self.k_align - self.b1_batch].set_as(tmp_tensor_ub[0])
            self.tik_instance.data_move(tmp_tensor_ub, self.boxes_gm[
                    self.n * Constant.Y_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
            y_of_boxes_ub[self.k_align - self.b1_batch].set_as(tmp_tensor_ub[0])
        else:
            self.tik_instance.data_move(x_of_boxes_ub[self.k_align - self.b1_batch], self.boxes_gm[
                    self.n * Constant.X_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
            self.tik_instance.data_move(y_of_boxes_ub[self.k_align - self.b1_batch], self.boxes_gm[
                    self.n * Constant.Y_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)

    def cal_coordinate_with_rotate(self, x_of_boxes_ub, y_of_boxes_ub, half_w_of_boxes_ub, half_h_of_boxes_ub,
                                   cos_t_of_boxes_ub, sin_t_of_boxes_ub, x1_of_boxes_ub, x2_of_boxes_ub, x3_of_boxes_ub,
                                   x4_of_boxes_ub, y1_of_boxes_ub, y2_of_boxes_ub, y3_of_boxes_ub, y4_of_boxes_ub):
        """
        Specifically calculate the coordinates after the cuboid is rotated
        """
        half_w_cos_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align],
                                                          name="half_w_cos_of_boxes_ub", scope=tik.scope_ubuf)
        half_w_sin_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align],
                                                          name="half_w_sin_of_boxes_ub", scope=tik.scope_ubuf)
        half_h_cos_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align],
                                                          name="half_h_cos_of_boxes_ub", scope=tik.scope_ubuf)
        half_h_sin_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align],
                                                          name="half_h_sin_of_boxes_ub", scope=tik.scope_ubuf)
        x_sub_w_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="x_sub_w_of_boxes_ub",
                                                       scope=tik.scope_ubuf)
        y_sub_w_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="y_sub_w_of_boxes_ub",
                                                       scope=tik.scope_ubuf)
        x_add_w_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="x_add_w_of_boxes_ub",
                                                       scope=tik.scope_ubuf)
        y_add_w_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="y_add_w_of_boxes_ub",
                                                       scope=tik.scope_ubuf)
        self.tik_instance.h_mul(half_w_cos_of_boxes_ub, cos_t_of_boxes_ub, half_w_of_boxes_ub)
        self.tik_instance.h_mul(half_w_sin_of_boxes_ub, sin_t_of_boxes_ub, half_w_of_boxes_ub)
        self.tik_instance.h_mul(half_h_cos_of_boxes_ub, cos_t_of_boxes_ub, half_h_of_boxes_ub)
        self.tik_instance.h_mul(half_h_sin_of_boxes_ub, sin_t_of_boxes_ub, half_h_of_boxes_ub)

        self.tik_instance.h_sub(x_sub_w_of_boxes_ub, x_of_boxes_ub, half_w_cos_of_boxes_ub)
        self.tik_instance.h_sub(y_sub_w_of_boxes_ub, y_of_boxes_ub, half_w_sin_of_boxes_ub)
        self.tik_instance.h_add(x_add_w_of_boxes_ub, x_of_boxes_ub, half_w_cos_of_boxes_ub)
        self.tik_instance.h_add(y_add_w_of_boxes_ub, y_of_boxes_ub, half_w_sin_of_boxes_ub)

        self.tik_instance.h_sub(x1_of_boxes_ub, x_sub_w_of_boxes_ub, half_h_sin_of_boxes_ub)
        self.tik_instance.h_add(y1_of_boxes_ub, y_sub_w_of_boxes_ub, half_h_cos_of_boxes_ub)

        self.tik_instance.h_sub(x2_of_boxes_ub, x_add_w_of_boxes_ub, half_h_sin_of_boxes_ub)
        self.tik_instance.h_add(y2_of_boxes_ub, y_add_w_of_boxes_ub, half_h_cos_of_boxes_ub)

        self.tik_instance.h_add(x3_of_boxes_ub, x_add_w_of_boxes_ub, half_h_sin_of_boxes_ub)
        self.tik_instance.h_sub(y3_of_boxes_ub, y_add_w_of_boxes_ub, half_h_cos_of_boxes_ub)

        self.tik_instance.h_add(x4_of_boxes_ub, x_sub_w_of_boxes_ub, half_h_sin_of_boxes_ub)
        self.tik_instance.h_sub(y4_of_boxes_ub, y_sub_w_of_boxes_ub, half_h_cos_of_boxes_ub)

    # 'pylint:disable=too-many-arguments, disable=too-many-statements
    def data_init(self):
        """
        Data initialization overall function
        """
        self.value = self.tik_instance.Scalar(self.dtype)
        self.b1_x1 = self.tik_instance.Scalar(self.dtype)
        self.b1_y1 = self.tik_instance.Scalar(self.dtype)
        self.b2_x1 = self.tik_instance.Scalar(self.dtype)
        self.b2_y1 = self.tik_instance.Scalar(self.dtype)
        self.b1_x2 = self.tik_instance.Scalar(self.dtype)
        self.b1_y2 = self.tik_instance.Scalar(self.dtype)
        self.b2_x2 = self.tik_instance.Scalar(self.dtype)
        self.b2_y2 = self.tik_instance.Scalar(self.dtype)
        self.b1_x3 = self.tik_instance.Scalar(self.dtype)
        self.b1_y3 = self.tik_instance.Scalar(self.dtype)
        self.AD_x = self.tik_instance.Scalar(self.dtype)
        self.AD_y = self.tik_instance.Scalar(self.dtype)
        self.AP_x = self.tik_instance.Scalar(self.dtype)
        self.AP_y = self.tik_instance.Scalar(self.dtype)
        self.BC_x = self.tik_instance.Scalar(self.dtype)
        self.BC_y = self.tik_instance.Scalar(self.dtype)
        self.BD_x = self.tik_instance.Scalar(self.dtype)
        self.BD_y = self.tik_instance.Scalar(self.dtype)

    def init_ub_local_variable(self):
        """
        init ub local variable
        """
        if self.is_old_version:
            ori_idx_fp16_ub = self.tik_instance.Tensor("float16", [Constant.BATCH], name="ori_idx_fp16_ub",
                                                   scope=tik.scope_ubuf)
        else:
            ori_idx_fp16_ub = None
        overlap_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="overlap_ub",
                                              scope=tik.scope_ubuf)
        w_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="w_of_boxes_ub",
                                                 scope=tik.scope_ubuf)
        h_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="h_of_boxes_ub",
                                                 scope=tik.scope_ubuf)
        half_w_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="half_w_of_boxes_ub",
                                                      scope=tik.scope_ubuf)
        half_h_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="half_h_of_boxes_ub",
                                                      scope=tik.scope_ubuf)
        cos_t_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="cos_t_of_boxes_ub",
                                                     scope=tik.scope_ubuf)
        sin_t_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="sin_t_of_boxes_ub",
                                                     scope=tik.scope_ubuf)
        x1_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="x1_of_boxes_ub",
                                                  scope=tik.scope_ubuf)
        x2_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="x2_of_boxes_ub",
                                                  scope=tik.scope_ubuf)
        x3_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="x3_of_boxes_ub",
                                                  scope=tik.scope_ubuf)
        x4_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="x4_of_boxes_ub",
                                                  scope=tik.scope_ubuf)
        y1_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="y1_of_boxes_ub",
                                                  scope=tik.scope_ubuf)
        y2_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="y2_of_boxes_ub",
                                                  scope=tik.scope_ubuf)
        y3_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="y3_of_boxes_ub",
                                                  scope=tik.scope_ubuf)
        y4_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="y4_of_boxes_ub",
                                                  scope=tik.scope_ubuf)
        x_tensor_ub = self.tik_instance.Tensor(self.dtype, [self.changed_shape], name="x_tensor_ub",
                                               scope=tik.scope_ubuf)
        y_tensor_ub = self.tik_instance.Tensor(self.dtype, [self.changed_shape], name="y_tensor_ub",
                                               scope=tik.scope_ubuf)
        slope_tensor_ub = self.tik_instance.Tensor(self.dtype, [self.changed_shape], name="slope_tensor_ub",
                                                   scope=tik.scope_ubuf)
        add_tensor_ub = self.tik_instance.Tensor(self.dtype, [Constant.BLOCK], name="add_tensor_ub",
                                                 scope=tik.scope_ubuf)
        abs_tensor_ub = self.tik_instance.Tensor(self.dtype, [Constant.BLOCK], name="abs_tensor_ub",
                                                 scope=tik.scope_ubuf)
        lis = [ori_idx_fp16_ub, overlap_ub, w_of_boxes_ub, h_of_boxes_ub, half_w_of_boxes_ub, half_h_of_boxes_ub,
               cos_t_of_boxes_ub, sin_t_of_boxes_ub, x1_of_boxes_ub, x2_of_boxes_ub, x3_of_boxes_ub, x4_of_boxes_ub,
               y1_of_boxes_ub, y2_of_boxes_ub, y3_of_boxes_ub, y4_of_boxes_ub, x_tensor_ub, y_tensor_ub,
               slope_tensor_ub, add_tensor_ub, abs_tensor_ub]
        return lis

    def init_scalar_and_ub_local_variable(self):
        """
        init_scalar_local_variable
        """
        d_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="d_of_boxes_ub",
                                                 scope=tik.scope_ubuf)
        work_tensor_ub = self.tik_instance.Tensor(self.dtype, [Constant.BLOCK], name="work_tensor_ub",
                                                  scope=tik.scope_ubuf)
        clockwise_idx_int32_ub = self.tik_instance.Tensor("int32", [Constant.BATCH],
                                                          name="clockwise_idx_int32_ub", scope=tik.scope_ubuf)
        tmp_tensor_ub = self.tik_instance.Tensor(self.dtype, [Constant.BLOCK], name="tmp_tensor_ub",
                                                 scope=tik.scope_ubuf)
        inter_volume = self.tik_instance.Scalar(self.dtype, init_value=0)
        b1_volume = self.tik_instance.Scalar(self.dtype)
        b2_volume = self.tik_instance.Scalar(self.dtype)
        b1_min = self.tik_instance.Scalar(self.dtype)
        b2_min = self.tik_instance.Scalar(self.dtype)
        b1_max = self.tik_instance.Scalar(self.dtype)
        b2_max = self.tik_instance.Scalar(self.dtype)
        idx_fp32 = self.tik_instance.Scalar(self.dtype, init_value=0)
        h_value = self.tik_instance.Scalar(self.dtype)
        valid_box_num = self.tik_instance.Scalar('int32')
        mov_repeats = self.tik_instance.Scalar('int32')
        max_of_min = self.tik_instance.Scalar(self.dtype)
        z_sub_d_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="z_sub_h_boxes_ub",
                                                    scope=tik.scope_ubuf)
        z_add_d_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="z_add_d_boxes_ub",
                                                    scope=tik.scope_ubuf)
        volume_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="volume_of_boxes_ub",
                                                      scope=tik.scope_ubuf)
        d_value = self.tik_instance.Scalar(self.dtype)
        corners_ub = self.tik_instance.Tensor(self.dtype, [Constant.BATCH], name="corners_ub",
                                              scope=tik.scope_ubuf)
        corners_num = self.tik_instance.Scalar("int32")
        b1_offset = self.tik_instance.Scalar("int32")
        min_of_max = self.tik_instance.Scalar(self.dtype)
        real_d = self.tik_instance.Scalar(self.dtype)
        zero = self.tik_instance.Scalar(self.dtype, init_value=0)
        lis = [d_of_boxes_ub, work_tensor_ub, clockwise_idx_int32_ub, tmp_tensor_ub, inter_volume, b1_volume, b2_volume,
               b1_min, b2_min, b1_max, b2_max, idx_fp32, h_value, valid_box_num, mov_repeats, max_of_min,
               z_sub_d_boxes_ub, z_add_d_boxes_ub, z_add_d_boxes_ub, volume_of_boxes_ub, d_value, corners_ub,
               corners_num, b1_offset, min_of_max, real_d, zero]
        return lis

    def compute_core(self, task_idx):
        """
        core computing
        """
        self.data_init()
        ori_idx_fp16_ub, overlap_ub, w_of_boxes_ub, h_of_boxes_ub, half_w_of_boxes_ub, half_h_of_boxes_ub,\
        cos_t_of_boxes_ub, sin_t_of_boxes_ub, x1_of_boxes_ub, x2_of_boxes_ub, x3_of_boxes_ub, x4_of_boxes_ub,\
        y1_of_boxes_ub, y2_of_boxes_ub, y3_of_boxes_ub, y4_of_boxes_ub, x_tensor_ub, y_tensor_ub,\
        slope_tensor_ub, add_tensor_ub, abs_tensor_ub = self.init_ub_local_variable()
        d_of_boxes_ub, work_tensor_ub, clockwise_idx_int32_ub, tmp_tensor_ub, inter_volume, b1_volume, b2_volume,\
        b1_min, b2_min, b1_max, b2_max, idx_fp32, h_value, valid_box_num, mov_repeats, max_of_min,\
        z_sub_d_boxes_ub, z_add_d_boxes_ub, z_add_d_boxes_ub, volume_of_boxes_ub, d_value, corners_ub,\
        corners_num, b1_offset, min_of_max, real_d, zero = self.init_scalar_and_ub_local_variable()
        if self.is_old_version:
            with self.tik_instance.for_range(0, Constant.BLOCK) as i:
                ori_idx_fp16_ub[i].set_as(idx_fp32)
                idx_fp32.set_as(idx_fp32 + 1)
        else:
            self.ori_idx_uint32_ub = self.tik_instance.Tensor("uint32", [Constant.INT32_BATCH],
                                                              name="ori_idx_uint32_ub",
                                                              scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, Constant.INT32_BATCH) as i:
                self.ori_idx_uint32_ub[i].set_as(i)
        
        with self.tik_instance.for_range(0, self.batch) as current_batch:
            self.trans_boxes(task_idx, current_batch, w_of_boxes_ub, h_of_boxes_ub, half_w_of_boxes_ub,
                             half_h_of_boxes_ub, cos_t_of_boxes_ub, sin_t_of_boxes_ub, x1_of_boxes_ub, x2_of_boxes_ub,
                             x3_of_boxes_ub, x4_of_boxes_ub, y1_of_boxes_ub, y2_of_boxes_ub,
                             y3_of_boxes_ub, y4_of_boxes_ub, tmp_tensor_ub)
            self.get_effective_depth(task_idx, current_batch, d_of_boxes_ub, z_sub_d_boxes_ub,
                                     z_add_d_boxes_ub, tmp_tensor_ub)
            valid_box_num.set_as(0)
            self.tik_instance.h_mul(volume_of_boxes_ub, h_of_boxes_ub, w_of_boxes_ub)
            self.tik_instance.h_mul(volume_of_boxes_ub, volume_of_boxes_ub, d_of_boxes_ub)
            # record the valid query_boxes's num
            w_value = self.tik_instance.Scalar(self.dtype)
            with self.tik_instance.for_range(0, self.k) as idx:
                w_value.set_as(w_of_boxes_ub[idx])
                h_value.set_as(h_of_boxes_ub[idx])
                d_value.set_as(d_of_boxes_ub[idx])
                with self.tik_instance.if_scope(w_value * h_value * d_value > 0):
                    valid_box_num.set_as(valid_box_num + 1)
            mov_repeats.set_as((valid_box_num + Constant.BLOCK - 1) // Constant.BLOCK)
            lis = [b1_volume, b2_volume, b1_min, b1_max, b2_min, b2_max, inter_volume, overlap_ub,
                   x1_of_boxes_ub, x2_of_boxes_ub, x3_of_boxes_ub, x4_of_boxes_ub, y1_of_boxes_ub, y2_of_boxes_ub,
                   y3_of_boxes_ub, y4_of_boxes_ub, x_tensor_ub, y_tensor_ub, slope_tensor_ub, add_tensor_ub,
                   abs_tensor_ub, work_tensor_ub, clockwise_idx_int32_ub, max_of_min, corners_ub, b1_offset, min_of_max,
                   real_d, zero]
            self.main_compute_per_core(lis, task_idx, current_batch, ori_idx_fp16_ub, valid_box_num, mov_repeats,
                                       z_sub_d_boxes_ub, z_add_d_boxes_ub, volume_of_boxes_ub, corners_num)

    # 'pylint:disable=too-many-arguments, disable=too-many-statements
    def main_compute_per_core(self, lis, task_idx, current_batch, ori_idx_fp16_ub, valid_box_num, mov_repeats,
                              z_sub_d_boxes_ub, z_add_d_boxes_ub, volume_of_boxes_ub, corners_num):
        """
        Core Specific Computing
        """
        b1_volume, b2_volume, b1_min, b1_max, b2_min, b2_max, inter_volume, overlap_ub, x1_of_boxes_ub,\
            x2_of_boxes_ub, x3_of_boxes_ub, x4_of_boxes_ub, y1_of_boxes_ub, y2_of_boxes_ub, y3_of_boxes_ub,\
            y4_of_boxes_ub, x_tensor_ub, y_tensor_ub, slope_tensor_ub, add_tensor_ub, abs_tensor_ub, work_tensor_ub,\
            clockwise_idx_int32_ub, max_of_min, corners_ub, b1_offset, min_of_max, real_d, zero = lis
        with self.tik_instance.for_range(0, self.b1_batch) as b1_idx:
            self.tik_instance.vec_dup(Constant.BLOCK, overlap_ub, 0, mov_repeats, 1)
            b1_offset.set_as(self.k_align - self.b1_batch + b1_idx)
            b1_volume.set_as(volume_of_boxes_ub[b1_offset])
            b1_min.set_as(z_sub_d_boxes_ub[b1_offset])
            b1_max.set_as(z_add_d_boxes_ub[b1_offset])
            with self.tik_instance.for_range(0, valid_box_num) as b2_idx:
                self.record_vertex_point(b2_idx, x1_of_boxes_ub, x2_of_boxes_ub, x3_of_boxes_ub, x4_of_boxes_ub,
                                         y1_of_boxes_ub, y2_of_boxes_ub, y3_of_boxes_ub, y4_of_boxes_ub,
                                         corners_ub, corners_num, b1_offset)
                self.record_intersection_point(b2_idx, x1_of_boxes_ub, x2_of_boxes_ub, x3_of_boxes_ub, x4_of_boxes_ub,
                    y1_of_boxes_ub, y2_of_boxes_ub, y3_of_boxes_ub, y4_of_boxes_ub, corners_ub, corners_num, b1_offset)
                b2_volume.set_as(volume_of_boxes_ub[b2_idx])
                b2_min.set_as(z_sub_d_boxes_ub[b2_idx])
                b2_max.set_as(z_add_d_boxes_ub[b2_idx])
                with self.tik_instance.if_scope(b2_min > b1_min):
                    max_of_min.set_as(b2_min)
                with self.tik_instance.else_scope():
                    max_of_min.set_as(b1_min)
                with self.tik_instance.if_scope(b2_max > b1_max):
                    min_of_max.set_as(b1_max)
                with self.tik_instance.else_scope():
                    min_of_max.set_as(b2_max)
                with self.tik_instance.if_scope(min_of_max - max_of_min > zero):
                    real_d.set_as(min_of_max - max_of_min)
                with self.tik_instance.else_scope():
                    real_d.set_as(zero)
                with self.tik_instance.if_scope(corners_num == 3):
                    self.b1_x1.set_as(corners_ub[0])
                    self.b1_y1.set_as(corners_ub[Constant.BLOCK])
                    self.get_area_of_triangle(1, 2, corners_ub)
                    with self.tik_instance.if_scope(self.value > 0):
                        inter_volume.set_as(self.value / 2)
                    with self.tik_instance.else_scope():
                        inter_volume.set_as(-1 * self.value / 2)
                with self.tik_instance.if_scope(corners_num > 3):
                    self.sum_area_of_triangles(ori_idx_fp16_ub, x_tensor_ub, y_tensor_ub, slope_tensor_ub,
                        add_tensor_ub, abs_tensor_ub, work_tensor_ub, clockwise_idx_int32_ub, corners_ub, corners_num)
                    inter_volume.set_as(self.value / 2)
                with self.tik_instance.if_scope(corners_num == 0):
                    inter_volume.set_as(0)
                inter_volume.set_as(real_d * inter_volume)
                with self.tik_instance.if_scope(b1_volume + b2_volume - inter_volume > 0):
                    overlap_ub[b2_idx].set_as(inter_volume / (b1_volume + b2_volume - inter_volume + Constant.EPSILON))
            self.tik_instance.data_move(self.iou_gm[self.k * (task_idx * self.b1_batch + b1_idx + current_batch *
                                                              self.n)], overlap_ub, 0, 1, mov_repeats, 0, 0)

    def compute(self):
        """
        Calculate the total interface
        """
        self.tik_instance.set_atomic_add(1)
        with self.tik_instance.for_range(0, self.used_aicore_num, block_num=self.used_aicore_num) as i:
            with self.tik_instance.for_range(0, self.batch_num_per_aicore) as j:
                self.compute_core(i + j * self.used_aicore_num)
            with self.tik_instance.if_scope(i < self.batch_tail):
                self.compute_core(self.batch_num_per_aicore * self.used_aicore_num + i)
        self.tik_instance.set_atomic_add(0)
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.boxes_gm, self.query_boxes_gm],
                                   outputs=[self.iou_gm])
        return self.tik_instance


# 'pylint:disable=too-many-arguments, disable=too-many-statements
@register_operator("Iou3D")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def iou_3d(boxes, query_boxes, iou, kernel_name="iou_3d"):
    """
    Function: compute the 3d iou.
    Modify : 2022-05-31

    Init base parameters
    Parameters
    ----------
    boxes: dict
        data of input
    query_boxes: dict
        data of input
    iou: dict
        data of output
    kernel_name: str
        the name of the operator
    ----------
    """
    op_obj = Iou3D(boxes, query_boxes, iou, kernel_name)
    return op_obj.compute()
