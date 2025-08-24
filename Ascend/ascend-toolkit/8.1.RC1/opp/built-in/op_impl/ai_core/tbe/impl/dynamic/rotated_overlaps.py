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
rotated_overlaps
"""
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tik
from tbe.common.platform.platform_info import api_check_support
from tbe.common.platform.platform_info import get_soc_spec


class Constant(object):
    """
    The class for constant.
    """
    # min val in float
    FP16_MIN_VAL = -65504
    FP32_MIN_VAL = -3.4e38
    # float32 data'nums in 32B
    BLOCK = 8
    # float16 data'nums in 32B
    BATCH = 16
    INT32_BATCH = 32
    # nums of vertices of rectangle
    CORNERS = 4
    # idx tag
    X_IDX = 0
    Y_IDX = 1
    W_IDX = 2
    H_IDX = 3
    T_IDX = 4
    X1_IDX = 0
    Y1_IDX = 1
    X2_IDX = 2
    Y2_IDX = 3
    # nums of box info
    INFOS = 5
    # val's idx in proposal
    VAL_IDX = 4
    # coefficient of angle to radian
    COEF = 0.01745329252
    # limit of k's size of query_boxes
    K_LIMIT = 2000
    # to avoid denominator zero
    EPSILON = 1e-6
    ALPHA = 1e-5
    MAX_INT32 = 2 ** 31 - 1
    TILING_SCALAR_DTYPE = "int32"
    TILING_PARAMS_NUM = 16
    TENSOR_UNIT = 43
    SCALAR_UNIT = 62


# 'pylint: disable=locally-disabled, unused-argument, invalid-name
class RotatedOverlaps(object):
    """
    The class for RotatedOverlaps.
    """

    # 'pylint:disable=too-many-arguments, disable=too-many-statements
    def __init__(self, boxes, trans, kernel_name):
        """
        class init
        """
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.trans = trans
        self.dtype = boxes.get("dtype").lower()
        self.kernel_name = kernel_name
        self.is_old_version = True if not api_check_support("tik.vreducev2") else False
        self.batch = self.tik_instance.Scalar("int32")
        self.n = self.tik_instance.Scalar("int32")
        self.k = self.tik_instance.Scalar("int32")
        self.task_num = self.tik_instance.Scalar("int32")
        self.b1_batch = self.tik_instance.Scalar("int32")
        self.b1_repeats = self.tik_instance.Scalar("int32")
        self.k_align = self.tik_instance.Scalar("int32")
        self.repeats = self.tik_instance.Scalar("int32")
        self.used_aicore_num = self.tik_instance.Scalar("int32")
        self.batch_num_per_aicore = self.tik_instance.Scalar("int32")
        self.batch_tail = self.tik_instance.Scalar("int32")
        # func: for task allocation
        self.avail_aicore_num = get_soc_spec("CORE_NUM")
        self.available_ub_size = get_soc_spec("UB_SIZE")

        self.boxes_gm = self.tik_instance.Tensor(self.dtype, (Constant.MAX_INT32,),
                                                 name="boxes_gm", scope=tik.scope_gm)
        self.query_boxes_gm = self.tik_instance.Tensor(self.dtype, (Constant.MAX_INT32,),
                                                       name="query_boxes_gm", scope=tik.scope_gm)
        self.overlaps_gm = self.tik_instance.Tensor(self.dtype, (Constant.MAX_INT32,), name="overlaps_gm",
                                                    scope=tik.scope_gm, is_atomic_add=True)
        self.tiling_gm = self.tik_instance.Tensor(Constant.TILING_SCALAR_DTYPE, (Constant.TILING_PARAMS_NUM,),
                                                  name="tiling_gm", scope=tik.scope_gm)

        self.v_threshold = self.tik_instance.Scalar("float32", init_value=0)
        self.e_threshold = self.tik_instance.Scalar("float32", init_value=0)
        # Tensor
        tensor_lis = [None] * Constant.TENSOR_UNIT
        self.idx_fp16_ub, self.ori_idx_fp16_ub, self.ori_idx_uint32_ub, \
        self.box_ub, self.overlap_ub, \
        self.corners_ub, self.val_fp16_ub, self.idx_int32_ub, self.proposal_ub, \
        self.x_of_boxes_ub, self.y_of_boxes_ub, self.w_of_boxes_ub, self.h_of_boxes_ub, \
        self.half_w_of_boxes_ub, self.half_h_of_boxes_ub, \
        self.t_of_boxes_ub, self.radian_t_of_boxes_ub, self.cos_t_of_boxes_ub, self.sin_t_of_boxes_ub, \
        self.half_w_cos_of_boxes_ub, self.half_w_sin_of_boxes_ub, \
        self.half_h_cos_of_boxes_ub, self.half_h_sin_of_boxes_ub, \
        self.x_sub_w_of_boxes_ub, self.y_sub_w_of_boxes_ub, self.x_add_w_of_boxes_ub, self.y_add_w_of_boxes_ub, \
        self.x1_of_boxes_ub, self.x2_of_boxes_ub, self.x3_of_boxes_ub, self.x4_of_boxes_ub, \
        self.y1_of_boxes_ub, self.y2_of_boxes_ub, self.y3_of_boxes_ub, self.y4_of_boxes_ub, \
        self.x_tensor_ub, self.y_tensor_ub, \
        self.slope_tensor_ub, self.add_tensor_ub, self.abs_tensor_ub, self.tmp_tensor_ub, \
        self.work_tensor_ub, self.clockwise_idx_int32_ub = tensor_lis

        # Scalar
        scalar_lis = [None] * Constant.SCALAR_UNIT
        self.idx_fp32, \
        self.min_val, self.half, self.radian, self.value, self.w_value, self.h_value, \
        self.valid_box_num, self.mov_repeats, self.corners_num, \
        self.idx_right, self.idx_left, self.b1_offset, \
        self.b1_x, self.b1_y, self.b2_x, self.b2_y, \
        self.b1_x1, self.b1_y1, self.b2_x1, self.b2_y1, \
        self.b1_x2, self.b1_y2, self.b2_x2, self.b2_y2, \
        self.b1_x3, self.b1_y3, self.b2_x3, self.b2_y3, \
        self.b1_x4, self.b1_y4, self.b2_x4, self.b2_y4, \
        self.AB_x, self.AB_y, self.AC_x, self.AC_y, self.AD_x, self.AD_y, self.AP_x, self.AP_y, \
        self.BC_x, self.BC_y, self.BD_x, self.BD_y, \
        self.AB_AB, self.AD_AD, self.AP_AB, self.AP_AD, \
        self.direct_AC_AD, self.direct_BC_BD, self.direct_CA_CB, self.direct_DA_DB, \
        self.tmp_1, self.tmp_2, \
        self.b1_x1_x2, self.b1_y1_y2, self.b2_x1_x2, self.b2_y1_y2, \
        self.denominator, self.numerator_x, self.numerator_y = scalar_lis

    def get_area_of_triangle(self, idx_tmp, idx_current_tmp):
        """
        Calculating triangle area based on vertex coordinates.
        """
        self.b1_x2.set_as(self.corners_ub[idx_tmp])
        self.b1_y2.set_as(self.corners_ub[idx_tmp + Constant.BLOCK])
        self.b1_x3.set_as(self.corners_ub[idx_current_tmp])
        self.b1_y3.set_as(self.corners_ub[idx_current_tmp + Constant.BLOCK])

        self.value.set_as(
            self.b1_x1 * (self.b1_y2 - self.b1_y3) + self.b1_x2 * (self.b1_y3 - self.b1_y1) + self.b1_x3 * (
                    self.b1_y1 - self.b1_y2))

    # 'pylint:disable=too-many-arguments, disable=too-many-statements
    def set_triangles_data(self):
        """
        set_triangles_data
        """
        self.tik_instance.vec_reduce_add(self.corners_num, self.add_tensor_ub, self.corners_ub,
                                         self.work_tensor_ub, 1, 1)
        self.b1_x1.set_as(self.add_tensor_ub[0])
        self.b1_x1.set_as(self.b1_x1 / self.corners_num)
        self.tik_instance.vec_reduce_add(self.corners_num, self.add_tensor_ub, self.corners_ub[Constant.BLOCK],
                                         self.work_tensor_ub, 1, 1)
        self.b1_y1.set_as(self.add_tensor_ub[0])
        self.b1_y1.set_as(self.b1_y1 / self.corners_num)

        self.tik_instance.data_move(self.x_tensor_ub, self.corners_ub, 0, 1, 1, 0, 0)
        self.tik_instance.data_move(self.y_tensor_ub, self.corners_ub[Constant.BLOCK], 0, 1, 1, 0, 0)

        self.tik_instance.h_sub(self.x_tensor_ub, self.x_tensor_ub, self.b1_x1)
        self.tik_instance.h_sub(self.y_tensor_ub, self.y_tensor_ub, self.b1_y1)
        self.tik_instance.h_div(self.slope_tensor_ub, self.y_tensor_ub, self.x_tensor_ub)

        if self.is_old_version:
            self.tik_instance.h_cast(self.val_fp16_ub, self.slope_tensor_ub, "none")
            self.min_val = self.tik_instance.Scalar('float16', init_value=Constant.FP16_MIN_VAL)
            with self.tik_instance.for_range(self.corners_num, Constant.BATCH) as idx:
                self.val_fp16_ub[idx].set_as(self.min_val)
            self.tik_instance.vconcat(self.proposal_ub, self.val_fp16_ub, 1, Constant.VAL_IDX)
            self.tik_instance.vconcat(self.proposal_ub, self.ori_idx_fp16_ub, 1, 0)

            # Sort slopes in descending order
            self.tik_instance.vrpsort16(self.proposal_ub[Constant.BATCH * Constant.BLOCK], self.proposal_ub, 1)
            self.tik_instance.vextract(self.idx_fp16_ub, self.proposal_ub[Constant.BATCH * Constant.BLOCK], 1, 0)
            self.tik_instance.h_cast(self.idx_int32_ub, self.idx_fp16_ub, "round")

        else:
            self.min_val = self.tik_instance.Scalar('float32', init_value=Constant.FP32_MIN_VAL)
            with self.tik_instance.for_range(self.corners_num, Constant.INT32_BATCH) as idx:
                self.slope_tensor_ub[idx].set_as(self.min_val)

            res_ub_tmp = self.tik_instance.Tensor("float32", [Constant.INT32_BATCH, 2], name="res_ub_tmp",
                                                  scope=tik.scope_ubuf)

            self.tik_instance.vsort32(res_ub_tmp, self.slope_tensor_ub, self.ori_idx_uint32_ub, 1)

            uint32_ub = res_ub_tmp.reinterpret_cast_to("uint32")
            tmp = self.tik_instance.Scalar("uint32")
            with self.tik_instance.for_range(0, self.corners_num) as i2:
                tmp.set_as(uint32_ub[2 * i2 + 1])
                self.idx_int32_ub[i2].set_as(tmp)

    # 'pylint:disable=too-many-arguments, disable=too-many-statements
    def sum_area_of_triangles(self):
        """
        sum_area_of_triangles
        """
        idx_tmp = self.tik_instance.Scalar("int32")
        idx_current_tmp = self.tik_instance.Scalar("int32")
        self.idx_left.set_as(0)
        self.idx_right.set_as(0)
        with self.tik_instance.for_range(0, self.corners_num) as idx:
            idx_current_tmp.set_as(self.idx_int32_ub[idx])
            self.b1_x.set_as(self.x_tensor_ub[idx_current_tmp])

            with self.tik_instance.if_scope(self.b1_x < 0):
                self.clockwise_idx_int32_ub[self.idx_left].set_as(idx_current_tmp)
                self.idx_left.set_as(self.idx_left + 1)
            with self.tik_instance.elif_scope(self.b1_x > 0):
                self.clockwise_idx_int32_ub[self.idx_right + Constant.BLOCK].set_as(idx_current_tmp)
                self.idx_right.set_as(self.idx_right + 1)
            with self.tik_instance.else_scope():
                self.b1_y.set_as(self.y_tensor_ub[idx_current_tmp])
                with self.tik_instance.if_scope(self.b1_y < 0):
                    self.clockwise_idx_int32_ub[self.idx_left].set_as(idx_current_tmp)
                    self.idx_left.set_as(self.idx_left + 1)
                with self.tik_instance.else_scope():
                    self.clockwise_idx_int32_ub[self.idx_right + Constant.BLOCK].set_as(idx_current_tmp)
                    self.idx_right.set_as(self.idx_right + 1)

        idx_tmp.set_as(self.clockwise_idx_int32_ub[0])
        with self.tik_instance.for_range(1, self.idx_left) as l_idx:
            idx_current_tmp.set_as(self.clockwise_idx_int32_ub[l_idx])
            self.get_area_of_triangle(idx_tmp, idx_current_tmp)
            self.add_tensor_ub[l_idx].set_as(self.value)
            idx_tmp.set_as(idx_current_tmp)
        with self.tik_instance.for_range(0, self.idx_right) as r_idx:
            idx_current_tmp.set_as(self.clockwise_idx_int32_ub[r_idx + Constant.BLOCK])
            self.get_area_of_triangle(idx_tmp, idx_current_tmp)
            self.add_tensor_ub[r_idx + self.idx_left].set_as(self.value)
            idx_tmp.set_as(idx_current_tmp)

        idx_current_tmp.set_as(self.clockwise_idx_int32_ub[0])
        self.get_area_of_triangle(idx_tmp, idx_current_tmp)
        self.add_tensor_ub[0].set_as(self.value)

        self.tik_instance.h_abs(self.abs_tensor_ub, self.add_tensor_ub)
        self.tik_instance.vec_reduce_add(self.corners_num, self.add_tensor_ub, self.abs_tensor_ub, self.work_tensor_ub,
                                         1, 1)

        self.value.set_as(self.add_tensor_ub[0])

    def record_intersection_point_core(self):
        """
        record_intersection_point_core
        """
        self.AC_x.set_as(self.b2_x1 - self.b1_x1)
        self.AC_y.set_as(self.b2_y1 - self.b1_y1)
        self.AD_x.set_as(self.b2_x2 - self.b1_x1)
        self.AD_y.set_as(self.b2_y2 - self.b1_y1)
        self.BC_x.set_as(self.b2_x1 - self.b1_x2)
        self.BC_y.set_as(self.b2_y1 - self.b1_y2)
        self.BD_x.set_as(self.b2_x2 - self.b1_x2)
        self.BD_y.set_as(self.b2_y2 - self.b1_y2)

        self.direct_AC_AD.set_as(self.AC_x * self.AD_y - self.AC_y * self.AD_x)
        self.direct_BC_BD.set_as(self.BC_x * self.BD_y - self.BC_y * self.BD_x)
        with self.tik_instance.if_scope(self.direct_AC_AD * self.direct_BC_BD < self.e_threshold):
            self.direct_CA_CB.set_as(self.AC_x * self.BC_y - self.AC_y * self.BC_x)
            self.direct_DA_DB.set_as(self.AD_x * self.BD_y - self.AD_y * self.BD_x)
            with self.tik_instance.if_scope(self.direct_CA_CB * self.direct_DA_DB < self.e_threshold):
                self.tmp_1.set_as(self.b1_x1 * self.b1_y2 - self.b1_y1 * self.b1_x2)
                self.tmp_2.set_as(self.b2_x1 * self.b2_y2 - self.b2_y1 * self.b2_x2)
                self.b1_x1_x2.set_as(self.b1_x1 - self.b1_x2)
                self.b1_y1_y2.set_as(self.b1_y1 - self.b1_y2)
                self.b2_x1_x2.set_as(self.b2_x1 - self.b2_x2)
                self.b2_y1_y2.set_as(self.b2_y1 - self.b2_y2)

                self.denominator.set_as(self.b2_x1_x2 * self.b1_y1_y2 - self.b1_x1_x2 * self.b2_y1_y2)
                self.numerator_x.set_as(self.b1_x1_x2 * self.tmp_2 - self.tmp_1 * self.b2_x1_x2)
                self.numerator_y.set_as(self.b1_y1_y2 * self.tmp_2 - self.tmp_1 * self.b2_y1_y2)

                self.corners_ub[self.corners_num].set_as(self.numerator_x / self.denominator)
                self.corners_ub[self.corners_num + Constant.BLOCK].set_as(self.numerator_y / self.denominator)
                self.corners_num.set_as(self.corners_num + 1)

    # 'pylint:disable=too-many-arguments, disable=too-many-statements
    def record_intersection_point_compute(self):
        self.b1_x2.set_as(self.x2_of_boxes_ub[self.b1_offset])
        self.b1_y2.set_as(self.y2_of_boxes_ub[self.b1_offset])
        self.record_intersection_point_core()
        self.b1_x1.set_as(self.x3_of_boxes_ub[self.b1_offset])
        self.b1_y1.set_as(self.y3_of_boxes_ub[self.b1_offset])
        self.record_intersection_point_core()
        self.b1_x2.set_as(self.x4_of_boxes_ub[self.b1_offset])
        self.b1_y2.set_as(self.y4_of_boxes_ub[self.b1_offset])
        self.record_intersection_point_core()
        self.b1_x1.set_as(self.x1_of_boxes_ub[self.b1_offset])
        self.b1_y1.set_as(self.y1_of_boxes_ub[self.b1_offset])
        self.record_intersection_point_core()

    # 'pylint:disable=too-many-arguments, disable=too-many-statements
    def record_intersection_point(self, b2_idx):
        """
        record_intersection_point
        """
        # part1
        self.record_intersection_point_core()
        self.b1_x1.set_as(self.x3_of_boxes_ub[self.b1_offset])
        self.b1_y1.set_as(self.y3_of_boxes_ub[self.b1_offset])
        self.record_intersection_point_core()
        self.b1_x2.set_as(self.x4_of_boxes_ub[self.b1_offset])
        self.b1_y2.set_as(self.y4_of_boxes_ub[self.b1_offset])
        self.record_intersection_point_core()
        self.b1_x1.set_as(self.x1_of_boxes_ub[self.b1_offset])
        self.b1_y1.set_as(self.y1_of_boxes_ub[self.b1_offset])
        self.record_intersection_point_core()

        # part2 AB->BC == A->C
        self.b2_x1.set_as(self.x3_of_boxes_ub[b2_idx])
        self.b2_y1.set_as(self.y3_of_boxes_ub[b2_idx])

        self.record_intersection_point_compute()

        # part3 BC->CD == B->D
        self.b2_x2.set_as(self.x4_of_boxes_ub[b2_idx])
        self.b2_y2.set_as(self.y4_of_boxes_ub[b2_idx])

        self.record_intersection_point_compute()

        # part4 CD->DA == C->A
        self.b2_x1.set_as(self.x1_of_boxes_ub[b2_idx])
        self.b2_y1.set_as(self.y1_of_boxes_ub[b2_idx])

        self.record_intersection_point_compute()

    # 'pylint:disable=too-many-arguments, disable=too-many-statements
    def record_vertex_point(self, b2_idx):
        """
        record_vertex_point
        """
        self.corners_num.set_as(0)

        # func: b1 for input boxes & b2 for input query_boxes
        self.b1_x1.set_as(self.x1_of_boxes_ub[self.b1_offset])
        self.b1_x2.set_as(self.x2_of_boxes_ub[self.b1_offset])
        self.b1_x3.set_as(self.x3_of_boxes_ub[self.b1_offset])
        self.b1_x4.set_as(self.x4_of_boxes_ub[self.b1_offset])

        self.b2_x1.set_as(self.x1_of_boxes_ub[b2_idx])
        self.b2_x2.set_as(self.x2_of_boxes_ub[b2_idx])
        self.b2_x3.set_as(self.x3_of_boxes_ub[b2_idx])
        self.b2_x4.set_as(self.x4_of_boxes_ub[b2_idx])

        self.b1_y1.set_as(self.y1_of_boxes_ub[self.b1_offset])
        self.b1_y2.set_as(self.y2_of_boxes_ub[self.b1_offset])
        self.b1_y3.set_as(self.y3_of_boxes_ub[self.b1_offset])
        self.b1_y4.set_as(self.y4_of_boxes_ub[self.b1_offset])

        self.b2_y1.set_as(self.y1_of_boxes_ub[b2_idx])
        self.b2_y2.set_as(self.y2_of_boxes_ub[b2_idx])
        self.b2_y3.set_as(self.y3_of_boxes_ub[b2_idx])
        self.b2_y4.set_as(self.y4_of_boxes_ub[b2_idx])

        # check b1
        self.AB_x.set_as(self.b2_x2 - self.b2_x1)
        self.AB_y.set_as(self.b2_y2 - self.b2_y1)
        self.AD_x.set_as(self.b2_x4 - self.b2_x1)
        self.AD_y.set_as(self.b2_y4 - self.b2_y1)

        self.AB_AB.set_as(self.AB_x * self.AB_x + self.AB_y * self.AB_y)
        self.AD_AD.set_as(self.AD_x * self.AD_x + self.AD_y * self.AD_y)

        self.AP_x.set_as(self.b1_x1 - self.b2_x1)
        self.AP_y.set_as(self.b1_y1 - self.b2_y1)
        self.AP_AB.set_as(self.AP_x * self.AB_x + self.AP_y * self.AB_y)
        with self.tik_instance.if_scope(tik.all(self.AP_AB + Constant.ALPHA >= self.v_threshold,
                                                self.AP_AB + self.v_threshold <= self.AB_AB + Constant.ALPHA)):
            self.AP_AD.set_as(self.AP_x * self.AD_x + self.AP_y * self.AD_y)
            with self.tik_instance.if_scope(tik.all(self.AP_AD + Constant.ALPHA >= self.v_threshold,
                                                    self.AP_AD + self.v_threshold <= self.AD_AD + Constant.ALPHA)):
                self.corners_ub[self.corners_num].set_as(self.b1_x1)
                self.corners_ub[self.corners_num + Constant.BLOCK].set_as(self.b1_y1)
                self.corners_num.set_as(self.corners_num + 1)

        self.AP_x.set_as(self.b1_x2 - self.b2_x1)
        self.AP_y.set_as(self.b1_y2 - self.b2_y1)
        self.AP_AB.set_as(self.AP_x * self.AB_x + self.AP_y * self.AB_y)
        with self.tik_instance.if_scope(tik.all(self.AP_AB + Constant.ALPHA >= self.v_threshold,
                                                self.AP_AB + self.v_threshold <= self.AB_AB + Constant.ALPHA)):
            self.AP_AD.set_as(self.AP_x * self.AD_x + self.AP_y * self.AD_y)
            with self.tik_instance.if_scope(tik.all(self.AP_AD + Constant.ALPHA >= self.v_threshold,
                                                    self.AP_AD + self.v_threshold <= self.AD_AD + Constant.ALPHA)):
                self.corners_ub[self.corners_num].set_as(self.b1_x2)
                self.corners_ub[self.corners_num + Constant.BLOCK].set_as(self.b1_y2)
                self.corners_num.set_as(self.corners_num + 1)

        self.AP_x.set_as(self.b1_x3 - self.b2_x1)
        self.AP_y.set_as(self.b1_y3 - self.b2_y1)
        self.AP_AB.set_as(self.AP_x * self.AB_x + self.AP_y * self.AB_y)
        with self.tik_instance.if_scope(tik.all(self.AP_AB + Constant.ALPHA >= self.v_threshold,
                                                self.AP_AB + self.v_threshold <= self.AB_AB + Constant.ALPHA)):
            self.AP_AD.set_as(self.AP_x * self.AD_x + self.AP_y * self.AD_y)
            with self.tik_instance.if_scope(tik.all(self.AP_AD + Constant.ALPHA >= self.v_threshold,
                                                    self.AP_AD + self.v_threshold <= self.AD_AD + Constant.ALPHA)):
                self.corners_ub[self.corners_num].set_as(self.b1_x3)
                self.corners_ub[self.corners_num + Constant.BLOCK].set_as(self.b1_y3)
                self.corners_num.set_as(self.corners_num + 1)

        self.AP_x.set_as(self.b1_x4 - self.b2_x1)
        self.AP_y.set_as(self.b1_y4 - self.b2_y1)
        self.AP_AB.set_as(self.AP_x * self.AB_x + self.AP_y * self.AB_y)
        with self.tik_instance.if_scope(tik.all(self.AP_AB + Constant.ALPHA >= self.v_threshold,
                                                self.AP_AB + self.v_threshold <= self.AB_AB + Constant.ALPHA)):
            self.AP_AD.set_as(self.AP_x * self.AD_x + self.AP_y * self.AD_y)
            with self.tik_instance.if_scope(tik.all(self.AP_AD + Constant.ALPHA >= self.v_threshold,
                                                    self.AP_AD + self.v_threshold <= self.AD_AD + Constant.ALPHA)):
                self.corners_ub[self.corners_num].set_as(self.b1_x4)
                self.corners_ub[self.corners_num + Constant.BLOCK].set_as(self.b1_y4)
                self.corners_num.set_as(self.corners_num + 1)

        # check b2
        self.AB_x.set_as(self.b1_x2 - self.b1_x1)
        self.AB_y.set_as(self.b1_y2 - self.b1_y1)
        self.AD_x.set_as(self.b1_x4 - self.b1_x1)
        self.AD_y.set_as(self.b1_y4 - self.b1_y1)

        self.AB_AB.set_as(self.AB_x * self.AB_x + self.AB_y * self.AB_y)
        self.AD_AD.set_as(self.AD_x * self.AD_x + self.AD_y * self.AD_y)

        self.AP_x.set_as(self.b2_x1 - self.b1_x1)
        self.AP_y.set_as(self.b2_y1 - self.b1_y1)
        self.AP_AB.set_as(self.AP_x * self.AB_x + self.AP_y * self.AB_y)
        with self.tik_instance.if_scope(tik.all(self.AP_AB + Constant.ALPHA >= self.v_threshold,
                                                self.AP_AB + self.v_threshold <= self.AB_AB + Constant.ALPHA)):
            self.AP_AD.set_as(self.AP_x * self.AD_x + self.AP_y * self.AD_y)
            with self.tik_instance.if_scope(tik.all(self.AP_AD + Constant.ALPHA >= self.v_threshold,
                                                    self.AP_AD + self.v_threshold <= self.AD_AD + Constant.ALPHA)):
                self.corners_ub[self.corners_num].set_as(self.b2_x1)
                self.corners_ub[self.corners_num + Constant.BLOCK].set_as(self.b2_y1)
                self.corners_num.set_as(self.corners_num + 1)

        self.AP_x.set_as(self.b2_x2 - self.b1_x1)
        self.AP_y.set_as(self.b2_y2 - self.b1_y1)
        self.AP_AB.set_as(self.AP_x * self.AB_x + self.AP_y * self.AB_y)
        with self.tik_instance.if_scope(tik.all(self.AP_AB + Constant.ALPHA >= self.v_threshold,
                                                self.AP_AB + self.v_threshold <= self.AB_AB + Constant.ALPHA)):
            self.AP_AD.set_as(self.AP_x * self.AD_x + self.AP_y * self.AD_y)
            with self.tik_instance.if_scope(tik.all(self.AP_AD + Constant.ALPHA >= self.v_threshold,
                                                    self.AP_AD + self.v_threshold <= self.AD_AD + Constant.ALPHA)):
                self.corners_ub[self.corners_num].set_as(self.b2_x2)
                self.corners_ub[self.corners_num + Constant.BLOCK].set_as(self.b2_y2)
                self.corners_num.set_as(self.corners_num + 1)

        self.AP_x.set_as(self.b2_x3 - self.b1_x1)
        self.AP_y.set_as(self.b2_y3 - self.b1_y1)
        self.AP_AB.set_as(self.AP_x * self.AB_x + self.AP_y * self.AB_y)
        with self.tik_instance.if_scope(tik.all(self.AP_AB + Constant.ALPHA >= self.v_threshold,
                                                self.AP_AB + self.v_threshold <= self.AB_AB + Constant.ALPHA)):
            self.AP_AD.set_as(self.AP_x * self.AD_x + self.AP_y * self.AD_y)
            with self.tik_instance.if_scope(tik.all(self.AP_AD + Constant.ALPHA >= self.v_threshold,
                                                    self.AP_AD + self.v_threshold <= self.AD_AD + Constant.ALPHA)):
                self.corners_ub[self.corners_num].set_as(self.b2_x3)
                self.corners_ub[self.corners_num + Constant.BLOCK].set_as(self.b2_y3)
                self.corners_num.set_as(self.corners_num + 1)

        self.AP_x.set_as(self.b2_x4 - self.b1_x1)
        self.AP_y.set_as(self.b2_y4 - self.b1_y1)
        self.AP_AB.set_as(self.AP_x * self.AB_x + self.AP_y * self.AB_y)
        with self.tik_instance.if_scope(tik.all(self.AP_AB + Constant.ALPHA >= self.v_threshold,
                                                self.AP_AB + self.v_threshold <= self.AB_AB + Constant.ALPHA)):
            self.AP_AD.set_as(self.AP_x * self.AD_x + self.AP_y * self.AD_y)
            with self.tik_instance.if_scope(tik.all(self.AP_AD + Constant.ALPHA >= self.v_threshold,
                                                    self.AP_AD + self.v_threshold <= self.AD_AD + Constant.ALPHA)):
                self.corners_ub[self.corners_num].set_as(self.b2_x4)
                self.corners_ub[self.corners_num + Constant.BLOCK].set_as(self.b2_y4)
                self.corners_num.set_as(self.corners_num + 1)

    # 'pylint:disable=too-many-arguments, disable=too-many-statements
    def move_theta(self, task_idx, current_batch):
        """
        move_theta
        """
        # theta
        self.tik_instance.data_move(
            self.t_of_boxes_ub, self.query_boxes_gm[self.k * (Constant.T_IDX + current_batch * Constant.INFOS)], 0,
            1, self.repeats, 0, 0)
        with self.tik_instance.if_scope(self.b1_batch == 1):
            self.tik_instance.data_move(
                self.tmp_tensor_ub,
                self.boxes_gm[
                    self.n * Constant.T_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
            self.t_of_boxes_ub[self.k_align - self.b1_batch].set_as(self.tmp_tensor_ub[0])
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(
                self.t_of_boxes_ub[self.k_align - self.b1_batch],
                self.boxes_gm[
                    self.n * Constant.T_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)

        self.tik_instance.h_mul(self.radian_t_of_boxes_ub, self.t_of_boxes_ub, self.radian)
        self.tik_instance.h_sin(self.sin_t_of_boxes_ub, self.radian_t_of_boxes_ub)
        self.tik_instance.h_cos(self.cos_t_of_boxes_ub, self.radian_t_of_boxes_ub)

    # 'pylint:disable=too-many-arguments, disable=too-many-statements
    def trans_boxes_xyxyt(self, task_idx, current_batch):
        """
        trans_boxes_xyxyt
        """
        # x & w
        self.tik_instance.data_move(
            self.x1_of_boxes_ub,
            self.query_boxes_gm[self.k * (Constant.X1_IDX + current_batch * Constant.INFOS)],
            0, 1, self.repeats, 0, 0)
        self.tik_instance.data_move(
            self.x2_of_boxes_ub,
            self.query_boxes_gm[self.k * (Constant.X2_IDX + current_batch * Constant.INFOS)],
            0, 1, self.repeats, 0, 0)

        with self.tik_instance.if_scope(self.b1_batch == 1):
            self.tik_instance.data_move(
                self.tmp_tensor_ub,
                self.boxes_gm[
                    self.n * Constant.X1_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
            self.x1_of_boxes_ub[self.k_align - self.b1_batch].set_as(self.tmp_tensor_ub[0])
            self.tik_instance.data_move(
                self.tmp_tensor_ub,
                self.boxes_gm[
                    self.n * Constant.X2_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
            self.x2_of_boxes_ub[self.k_align - self.b1_batch].set_as(self.tmp_tensor_ub[0])
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(
                self.x1_of_boxes_ub[self.k_align - self.b1_batch],
                self.boxes_gm[
                    self.n * Constant.X1_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
            self.tik_instance.data_move(
                self.x2_of_boxes_ub[self.k_align - self.b1_batch],
                self.boxes_gm[
                    self.n * Constant.X2_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)

        self.tik_instance.h_sub(self.w_of_boxes_ub, self.x2_of_boxes_ub, self.x1_of_boxes_ub)
        self.tik_instance.h_mul(self.half_w_of_boxes_ub, self.w_of_boxes_ub, self.half)
        self.tik_instance.h_add(self.x_of_boxes_ub, self.x1_of_boxes_ub, self.half_w_of_boxes_ub)
        # y & h
        self.tik_instance.data_move(
            self.y1_of_boxes_ub,
            self.query_boxes_gm[self.k * (Constant.Y1_IDX + current_batch * Constant.INFOS)],
            0, 1, self.repeats, 0, 0)
        self.tik_instance.data_move(
            self.y2_of_boxes_ub,
            self.query_boxes_gm[self.k * (Constant.Y2_IDX + current_batch * Constant.INFOS)],
            0, 1, self.repeats, 0, 0)

        with self.tik_instance.if_scope(self.b1_batch == 1):
            self.tik_instance.data_move(
                self.tmp_tensor_ub,
                self.boxes_gm[
                    self.n * Constant.Y1_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
            self.y1_of_boxes_ub[self.k_align - self.b1_batch].set_as(self.tmp_tensor_ub[0])
            self.tik_instance.data_move(
                self.tmp_tensor_ub,
                self.boxes_gm[
                    self.n * Constant.Y2_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
            self.y2_of_boxes_ub[self.k_align - self.b1_batch].set_as(self.tmp_tensor_ub[0])
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(
                self.y1_of_boxes_ub[self.k_align - self.b1_batch],
                self.boxes_gm[
                    self.n * Constant.Y1_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
            self.tik_instance.data_move(
                self.y2_of_boxes_ub[self.k_align - self.b1_batch],
                self.boxes_gm[
                    self.n * Constant.Y2_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)

        self.tik_instance.h_sub(self.h_of_boxes_ub, self.y2_of_boxes_ub, self.y1_of_boxes_ub)
        self.tik_instance.h_mul(self.half_h_of_boxes_ub, self.h_of_boxes_ub, self.half)
        self.tik_instance.h_add(self.y_of_boxes_ub, self.y1_of_boxes_ub, self.half_h_of_boxes_ub)

    # 'pylint:disable=too-many-arguments, disable=too-many-statements
    def trans_boxes_xywht(self, task_idx, current_batch):
        """
        trans_boxes_xywht
        """
        # w * h
        self.tik_instance.data_move(
            self.w_of_boxes_ub, self.query_boxes_gm[self.k * (Constant.W_IDX + current_batch * Constant.INFOS)],
            0, 1, self.repeats, 0, 0)
        self.tik_instance.data_move(
            self.h_of_boxes_ub, self.query_boxes_gm[self.k * (Constant.H_IDX + current_batch * Constant.INFOS)],
            0, 1, self.repeats, 0, 0)

        with self.tik_instance.if_scope(self.b1_batch == 1):
            self.tik_instance.data_move(
                self.tmp_tensor_ub,
                self.boxes_gm[
                    self.n * Constant.W_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
            self.w_of_boxes_ub[self.k_align - self.b1_batch].set_as(self.tmp_tensor_ub[0])
            self.tik_instance.data_move(
                self.tmp_tensor_ub,
                self.boxes_gm[
                    self.n * Constant.H_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
            self.h_of_boxes_ub[self.k_align - self.b1_batch].set_as(self.tmp_tensor_ub[0])
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(
                self.w_of_boxes_ub[self.k_align - self.b1_batch],
                self.boxes_gm[
                    self.n * Constant.W_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
            self.tik_instance.data_move(
                self.h_of_boxes_ub[self.k_align - self.b1_batch],
                self.boxes_gm[
                    self.n * Constant.H_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
        self.tik_instance.h_mul(self.half_w_of_boxes_ub, self.w_of_boxes_ub, self.half)
        self.tik_instance.h_mul(self.half_h_of_boxes_ub, self.h_of_boxes_ub, self.half)
        # x * y
        self.tik_instance.data_move(
            self.x_of_boxes_ub, self.query_boxes_gm[self.k * (Constant.X_IDX + current_batch * Constant.INFOS)],
            0, 1, self.repeats, 0, 0)
        self.tik_instance.data_move(
            self.y_of_boxes_ub, self.query_boxes_gm[self.k * (Constant.Y_IDX + current_batch * Constant.INFOS)],
            0, 1, self.repeats, 0, 0)

        with self.tik_instance.if_scope(self.b1_batch == 1):
            self.tik_instance.data_move(
                self.tmp_tensor_ub,
                self.boxes_gm[
                    self.n * Constant.X_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
            self.x_of_boxes_ub[self.k_align - self.b1_batch].set_as(self.tmp_tensor_ub[0])
            self.tik_instance.data_move(
                self.tmp_tensor_ub,
                self.boxes_gm[
                    self.n * Constant.Y_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
            self.y_of_boxes_ub[self.k_align - self.b1_batch].set_as(self.tmp_tensor_ub[0])
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(
                self.x_of_boxes_ub[self.k_align - self.b1_batch],
                self.boxes_gm[
                    self.n * Constant.X_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)
            self.tik_instance.data_move(
                self.y_of_boxes_ub[self.k_align - self.b1_batch],
                self.boxes_gm[
                    self.n * Constant.Y_IDX + self.b1_batch * task_idx + current_batch * self.n * Constant.INFOS],
                0, 1, self.b1_repeats, 0, 0)

    # 'pylint:disable=too-many-arguments, disable=too-many-statements
    def cal_boxes(self):
        """
        cal_boxes
        """
        self.tik_instance.h_mul(self.half_w_cos_of_boxes_ub, self.cos_t_of_boxes_ub, self.half_w_of_boxes_ub)
        self.tik_instance.h_mul(self.half_w_sin_of_boxes_ub, self.sin_t_of_boxes_ub, self.half_w_of_boxes_ub)
        self.tik_instance.h_mul(self.half_h_cos_of_boxes_ub, self.cos_t_of_boxes_ub, self.half_h_of_boxes_ub)
        self.tik_instance.h_mul(self.half_h_sin_of_boxes_ub, self.sin_t_of_boxes_ub, self.half_h_of_boxes_ub)

        self.tik_instance.h_sub(self.x_sub_w_of_boxes_ub, self.x_of_boxes_ub, self.half_w_cos_of_boxes_ub)
        self.tik_instance.h_sub(self.y_sub_w_of_boxes_ub, self.y_of_boxes_ub, self.half_w_sin_of_boxes_ub)
        self.tik_instance.h_add(self.x_add_w_of_boxes_ub, self.x_of_boxes_ub, self.half_w_cos_of_boxes_ub)
        self.tik_instance.h_add(self.y_add_w_of_boxes_ub, self.y_of_boxes_ub, self.half_w_sin_of_boxes_ub)

        self.tik_instance.h_sub(self.x1_of_boxes_ub, self.x_sub_w_of_boxes_ub, self.half_h_sin_of_boxes_ub)
        self.tik_instance.h_add(self.y1_of_boxes_ub, self.y_sub_w_of_boxes_ub, self.half_h_cos_of_boxes_ub)

        self.tik_instance.h_sub(self.x2_of_boxes_ub, self.x_add_w_of_boxes_ub, self.half_h_sin_of_boxes_ub)
        self.tik_instance.h_add(self.y2_of_boxes_ub, self.y_add_w_of_boxes_ub, self.half_h_cos_of_boxes_ub)

        self.tik_instance.h_add(self.x3_of_boxes_ub, self.x_add_w_of_boxes_ub, self.half_h_sin_of_boxes_ub)
        self.tik_instance.h_sub(self.y3_of_boxes_ub, self.y_add_w_of_boxes_ub, self.half_h_cos_of_boxes_ub)

        self.tik_instance.h_add(self.x4_of_boxes_ub, self.x_sub_w_of_boxes_ub, self.half_h_sin_of_boxes_ub)
        self.tik_instance.h_sub(self.y4_of_boxes_ub, self.y_sub_w_of_boxes_ub, self.half_h_cos_of_boxes_ub)

    # 'pylint:disable=too-many-arguments, disable=too-many-statements
    def init_tensor(self):
        """
        init_tensor
        """
        # Tensor
        self.box_ub = self.tik_instance.Tensor(self.dtype, [Constant.BLOCK], name="box_ub", scope=tik.scope_ubuf)
        self.overlap_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="overlap_ub", scope=tik.scope_ubuf)

        self.x_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="x_of_boxes_ub",
                                                      scope=tik.scope_ubuf)
        self.y_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="y_of_boxes_ub",
                                                      scope=tik.scope_ubuf)
        self.w_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="w_of_boxes_ub",
                                                      scope=tik.scope_ubuf)
        self.h_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="h_of_boxes_ub",
                                                      scope=tik.scope_ubuf)
        self.half_w_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="half_w_of_boxes_ub",
                                                           scope=tik.scope_ubuf)
        self.half_h_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="half_h_of_boxes_ub",
                                                           scope=tik.scope_ubuf)

        self.t_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="t_of_boxes_ub",
                                                      scope=tik.scope_ubuf)
        self.radian_t_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="radian_t_of_boxes_ub",
                                                             scope=tik.scope_ubuf)
        self.cos_t_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="cos_t_of_boxes_ub",
                                                          scope=tik.scope_ubuf)
        self.sin_t_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="sin_t_of_boxes_ub",
                                                          scope=tik.scope_ubuf)

        self.half_w_cos_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align],
                                                               name="half_w_cos_of_boxes_ub", scope=tik.scope_ubuf)
        self.half_w_sin_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align],
                                                               name="half_w_sin_of_boxes_ub", scope=tik.scope_ubuf)
        self.half_h_cos_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align],
                                                               name="half_h_cos_of_boxes_ub", scope=tik.scope_ubuf)
        self.half_h_sin_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align],
                                                               name="half_h_sin_of_boxes_ub", scope=tik.scope_ubuf)

        self.x_sub_w_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="x_sub_w_of_boxes_ub",
                                                            scope=tik.scope_ubuf)
        self.y_sub_w_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="y_sub_w_of_boxes_ub",
                                                            scope=tik.scope_ubuf)
        self.x_add_w_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="x_add_w_of_boxes_ub",
                                                            scope=tik.scope_ubuf)
        self.y_add_w_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="y_add_w_of_boxes_ub",
                                                            scope=tik.scope_ubuf)

        self.x1_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="x1_of_boxes_ub",
                                                       scope=tik.scope_ubuf)
        self.x2_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="x2_of_boxes_ub",
                                                       scope=tik.scope_ubuf)
        self.x3_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="x3_of_boxes_ub",
                                                       scope=tik.scope_ubuf)
        self.x4_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="x4_of_boxes_ub",
                                                       scope=tik.scope_ubuf)
        self.y1_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="y1_of_boxes_ub",
                                                       scope=tik.scope_ubuf)
        self.y2_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="y2_of_boxes_ub",
                                                       scope=tik.scope_ubuf)
        self.y3_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="y3_of_boxes_ub",
                                                       scope=tik.scope_ubuf)
        self.y4_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="y4_of_boxes_ub",
                                                       scope=tik.scope_ubuf)

        self.add_tensor_ub = self.tik_instance.Tensor(self.dtype, [Constant.BLOCK], name="add_tensor_ub",
                                                      scope=tik.scope_ubuf)
        self.abs_tensor_ub = self.tik_instance.Tensor(self.dtype, [Constant.BLOCK], name="abs_tensor_ub",
                                                      scope=tik.scope_ubuf)
        self.tmp_tensor_ub = self.tik_instance.Tensor(self.dtype, [Constant.BLOCK], name="tmp_tensor_ub",
                                                      scope=tik.scope_ubuf)
        self.work_tensor_ub = self.tik_instance.Tensor(self.dtype, [Constant.BLOCK], name="work_tensor_ub",
                                                       scope=tik.scope_ubuf)

        self.corners_ub = self.tik_instance.Tensor(self.dtype, [Constant.BATCH], name="corners_ub",
                                                   scope=tik.scope_ubuf)
        self.clockwise_idx_int32_ub = self.tik_instance.Tensor("int32", [Constant.BATCH], name="clockwise_idx_int32_ub",
                                                               scope=tik.scope_ubuf)
        self.idx_int32_ub = self.tik_instance.Tensor("int32", [Constant.BATCH], name="idx_int32_ub",
                                                     scope=tik.scope_ubuf)
        if self.is_old_version:
            self.val_fp16_ub = self.tik_instance.Tensor("float16", [Constant.BATCH], name="val_fp16_ub",
                                                        scope=tik.scope_ubuf)

            self.proposal_ub = self.tik_instance.Tensor("float16", [2, Constant.BATCH, Constant.BLOCK],
                                                        name="proposal_ub",
                                                        scope=tik.scope_ubuf)

            self.x_tensor_ub = self.tik_instance.Tensor(self.dtype, [Constant.BATCH], name="x_tensor_ub",
                                                        scope=tik.scope_ubuf)
            self.y_tensor_ub = self.tik_instance.Tensor(self.dtype, [Constant.BATCH], name="y_tensor_ub",
                                                        scope=tik.scope_ubuf)

            self.slope_tensor_ub = self.tik_instance.Tensor(self.dtype, [Constant.BATCH], name="slope_tensor_ub",
                                                            scope=tik.scope_ubuf)

            self.idx_fp16_ub = self.tik_instance.Tensor("float16", [Constant.BATCH], name="idx_fp16_ub",
                                                        scope=tik.scope_ubuf)

            self.ori_idx_fp16_ub = self.tik_instance.Tensor("float16", [Constant.BATCH], name="ori_idx_fp16_ub",
                                                            scope=tik.scope_ubuf)
        else:
            self.x_tensor_ub = self.tik_instance.Tensor(self.dtype, [Constant.INT32_BATCH], name="x_tensor_ub",
                                                        scope=tik.scope_ubuf)
            self.y_tensor_ub = self.tik_instance.Tensor(self.dtype, [Constant.INT32_BATCH], name="y_tensor_ub",
                                                        scope=tik.scope_ubuf)

            self.slope_tensor_ub = self.tik_instance.Tensor(self.dtype, [Constant.INT32_BATCH], name="slope_tensor_ub",
                                                            scope=tik.scope_ubuf)

            self.ori_idx_uint32_ub = self.tik_instance.Tensor("uint32", [Constant.INT32_BATCH],
                                                              name="ori_idx_uint32_ub",
                                                              scope=tik.scope_ubuf)

    # 'pylint:disable=too-many-arguments, disable=too-many-statements            
    def init_scalar(self):
        """
        init_scalar
        """
        self.idx_fp32 = self.tik_instance.Scalar(self.dtype, init_value=0)
        self.half = self.tik_instance.Scalar(self.dtype, init_value=0.5)
        self.radian = self.tik_instance.Scalar(self.dtype, init_value=Constant.COEF)
        self.value = self.tik_instance.Scalar(self.dtype)
        self.w_value = self.tik_instance.Scalar(self.dtype)
        self.h_value = self.tik_instance.Scalar(self.dtype)

        self.valid_box_num = self.tik_instance.Scalar('int32')
        self.mov_repeats = self.tik_instance.Scalar('int32')
        self.corners_num = self.tik_instance.Scalar("int32")

        self.idx_right = self.tik_instance.Scalar("int32")
        self.idx_left = self.tik_instance.Scalar("int32")
        self.b1_offset = self.tik_instance.Scalar("int32")

        self.b1_x = self.tik_instance.Scalar(self.dtype)
        self.b1_y = self.tik_instance.Scalar(self.dtype)
        self.b2_x = self.tik_instance.Scalar(self.dtype)
        self.b2_y = self.tik_instance.Scalar(self.dtype)

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
        self.b2_x3 = self.tik_instance.Scalar(self.dtype)
        self.b2_y3 = self.tik_instance.Scalar(self.dtype)

        self.b1_x4 = self.tik_instance.Scalar(self.dtype)
        self.b1_y4 = self.tik_instance.Scalar(self.dtype)
        self.b2_x4 = self.tik_instance.Scalar(self.dtype)
        self.b2_y4 = self.tik_instance.Scalar(self.dtype)

        self.b1_x1_x2 = self.tik_instance.Scalar(self.dtype)
        self.b1_y1_y2 = self.tik_instance.Scalar(self.dtype)
        self.b2_x1_x2 = self.tik_instance.Scalar(self.dtype)
        self.b2_y1_y2 = self.tik_instance.Scalar(self.dtype)

        self.AB_x = self.tik_instance.Scalar(self.dtype)
        self.AB_y = self.tik_instance.Scalar(self.dtype)
        self.AC_x = self.tik_instance.Scalar(self.dtype)
        self.AC_y = self.tik_instance.Scalar(self.dtype)
        self.AD_x = self.tik_instance.Scalar(self.dtype)
        self.AD_y = self.tik_instance.Scalar(self.dtype)
        self.AP_x = self.tik_instance.Scalar(self.dtype)
        self.AP_y = self.tik_instance.Scalar(self.dtype)

        self.AB_AB = self.tik_instance.Scalar(self.dtype)
        self.AD_AD = self.tik_instance.Scalar(self.dtype)
        self.AP_AB = self.tik_instance.Scalar(self.dtype)
        self.AP_AD = self.tik_instance.Scalar(self.dtype)

        self.BC_x = self.tik_instance.Scalar(self.dtype)
        self.BC_y = self.tik_instance.Scalar(self.dtype)
        self.BD_x = self.tik_instance.Scalar(self.dtype)
        self.BD_y = self.tik_instance.Scalar(self.dtype)

        self.direct_AC_AD = self.tik_instance.Scalar(self.dtype)
        self.direct_BC_BD = self.tik_instance.Scalar(self.dtype)
        self.direct_CA_CB = self.tik_instance.Scalar(self.dtype)
        self.direct_DA_DB = self.tik_instance.Scalar(self.dtype)

        self.tmp_1 = self.tik_instance.Scalar(self.dtype)
        self.tmp_2 = self.tik_instance.Scalar(self.dtype)
        self.denominator = self.tik_instance.Scalar(self.dtype)
        self.numerator_x = self.tik_instance.Scalar(self.dtype)
        self.numerator_y = self.tik_instance.Scalar(self.dtype)

    # 'pylint:disable=too-many-arguments, disable=too-many-statements
    def compute_core(self, task_idx):
        """
        single task
        """
        self.init_tensor()
        self.init_scalar()
        if self.is_old_version:
            with self.tik_instance.for_range(0, Constant.BLOCK) as i:
                self.ori_idx_fp16_ub[i].set_as(self.idx_fp32)
                self.idx_fp32.set_as(self.idx_fp32 + 1)
        else:
            with self.tik_instance.for_range(0, Constant.INT32_BATCH) as i:
                self.ori_idx_uint32_ub[i].set_as(i)

        with self.tik_instance.for_range(0, self.batch) as current_batch:
            self.move_theta(task_idx, current_batch)

            if self.trans:
                self.trans_boxes_xyxyt(task_idx, current_batch)
            else:
                self.trans_boxes_xywht(task_idx, current_batch)
            self.cal_boxes()

            self.valid_box_num.set_as(0)
            # record the valid query_boxes's num
            with self.tik_instance.for_range(0, self.k) as idx:
                self.w_value.set_as(self.w_of_boxes_ub[idx])
                self.h_value.set_as(self.h_of_boxes_ub[idx])
                with self.tik_instance.if_scope(self.w_value * self.h_value > 0):
                    self.valid_box_num.set_as(self.valid_box_num + 1)
            self.mov_repeats.set_as((self.valid_box_num + Constant.BLOCK - 1) // Constant.BLOCK)

            with self.tik_instance.for_range(0, self.b1_batch) as b1_idx:
                self.tik_instance.vec_dup(Constant.BLOCK, self.overlap_ub, 0, self.mov_repeats, 1)
                self.b1_offset.set_as(self.k_align - self.b1_batch + b1_idx)
                with self.tik_instance.for_range(0, self.valid_box_num) as b2_idx:
                    self.record_vertex_point(b2_idx)
                    self.record_intersection_point(b2_idx)
                    with self.tik_instance.if_scope(self.corners_num == 3):
                        self.b1_x1.set_as(self.corners_ub[0])
                        self.b1_y1.set_as(self.corners_ub[Constant.BLOCK])
                        self.get_area_of_triangle(1, 2)
                        with self.tik_instance.if_scope(self.value > 0):
                            self.overlap_ub[b2_idx].set_as(self.value / 2)
                        with self.tik_instance.else_scope():
                            self.overlap_ub[b2_idx].set_as(-1 * self.value / 2)
                    with self.tik_instance.if_scope(self.corners_num > 3):
                        self.set_triangles_data()
                        self.sum_area_of_triangles()
                        self.overlap_ub[b2_idx].set_as(self.value / 2)

                self.tik_instance.data_move(
                    self.overlaps_gm[self.k * (task_idx * self.b1_batch + b1_idx + current_batch * self.n)],
                    self.overlap_ub, 0, 1, self.mov_repeats, 0, 0)

    # 'pylint:disable=too-many-arguments, disable=too-many-statements
    def get_tiling_data(self):
        tiling_ub = self.tik_instance.Tensor(
            Constant.TILING_SCALAR_DTYPE,
            (Constant.TILING_PARAMS_NUM,),
            tik.scope_ubuf,
            "tiling_ub"
        )
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, 2, 0, 0)
        self.batch.set_as(tiling_ub[0])
        self.n.set_as(tiling_ub[1])
        self.k.set_as(tiling_ub[2])
        self.task_num.set_as(tiling_ub[3])
        self.b1_batch.set_as(tiling_ub[4])
        self.b1_repeats.set_as(tiling_ub[5])
        self.k_align.set_as(tiling_ub[6])
        self.repeats.set_as(tiling_ub[7])
        self.used_aicore_num.set_as(tiling_ub[8])
        self.batch_num_per_aicore.set_as(tiling_ub[9])
        self.batch_tail.set_as(tiling_ub[10])

    # 'pylint:disable=too-many-arguments, disable=too-many-statements
    def compute(self):
        """
        task fix
        """
        self.get_tiling_data()
        self.tik_instance.set_atomic_add(1)
        with self.tik_instance.for_range(0, self.used_aicore_num, block_num=self.used_aicore_num) as i:
            with self.tik_instance.for_range(0, self.batch_num_per_aicore) as j:
                self.compute_core(i + j * self.used_aicore_num)
            with self.tik_instance.if_scope(i < self.batch_tail):
                self.compute_core(self.batch_num_per_aicore * self.used_aicore_num + i)
        self.tik_instance.set_atomic_add(0)

        tbe_context.get_context().add_compile_info(
            "vars", {
                "full_core_num": self.avail_aicore_num,
                "ub_size": self.available_ub_size,
            }
        )

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.boxes_gm, self.query_boxes_gm],
                                   outputs=[self.overlaps_gm],
                                   flowtable=[self.tiling_gm])

        return self.tik_instance


# 'pylint:disable=too-many-arguments, disable=too-many-statements
@register_operator_compute("rotated_overlaps", op_mode="dynamic", support_fusion=True)
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def rotated_overlaps(boxes, query_boxes, overlaps, trans=False, kernel_name="rotated_overlaps"):
    """
    Function: compute the rotated boxes's overlaps.
    Modify : 2023-06-11

    Init base parameters
    Parameters
    ----------
    input(boxes): dict
        data of input
    input(query_boxes): dict
        data of input
    output(overlaps): dict
        data of output

    Attributes:
    trans : bool
        true for 'xyxyt', false for 'xywht'

    kernel_name: str
        the name of the operator
    ----------
    """
    op_obj = RotatedOverlaps(boxes, trans, kernel_name)

    return op_obj.compute()
