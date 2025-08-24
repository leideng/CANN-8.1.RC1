# Copyright 2023 Huawei Technologies Co., Ltd
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
rotated_iou
"""

from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tik
from impl.rotated_overlaps import Constant
from impl.rotated_overlaps import RotatedOverlaps


# 'pylint: disable=locally-disabled,unused-argument,invalid-name
class RotatedIou(RotatedOverlaps):
    """
    The class for RotatedIou.
    """
    # 'pylint:disable=too-many-arguments, disable=too-many-statements
    def __init__(self, boxes, query_boxes, iou,
                 trans, mode, is_cross, v_threshold, e_threshold, kernel_name):
        """
        class init
        """
        RotatedOverlaps.__init__(self, boxes, query_boxes, iou, trans, kernel_name)
        self.v_threshold = v_threshold
        self.e_threshold = e_threshold

        self.area_of_boxes_ub = None

    # 'pylint:disable=too-many-arguments, disable=too-many-statements
    def compute_core(self, task_idx):
        """
        single task
        """
        self.data_init()
        self.area_of_boxes_ub = self.tik_instance.Tensor(self.dtype, [self.k_align], name="area_of_boxes_ub",
                                                         scope=tik.scope_ubuf)
        b1_area = self.tik_instance.Scalar(self.dtype)
        b2_area = self.tik_instance.Scalar(self.dtype)
        overlap = self.tik_instance.Scalar(self.dtype)
        if self.is_old_version:
            with self.tik_instance.for_range(0, Constant.BLOCK) as i:
                self.ori_idx_fp16_ub[i].set_as(self.idx_fp32)
                self.idx_fp32.set_as(self.idx_fp32 + 1)
        else:
            with self.tik_instance.for_range(0, Constant.INT32_BATCH) as i:
                self.ori_idx_uint32_ub[i].set_as(i)

        with self.tik_instance.for_range(0, self.batch) as current_batch:
            self.trans_boxes(task_idx, current_batch)
            self.valid_box_num.set_as(0)
            self.tik_instance.h_mul(self.area_of_boxes_ub, self.h_of_boxes_ub, self.w_of_boxes_ub)
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
                b1_area.set_as(self.area_of_boxes_ub[self.b1_offset])
                with self.tik_instance.for_range(0, self.valid_box_num) as b2_idx:
                    self.record_vertex_point(b2_idx)
                    self.record_intersection_point(b2_idx)
                    b2_area.set_as(self.area_of_boxes_ub[b2_idx])
                    with self.tik_instance.if_scope(self.corners_num == 3):
                        self.b1_x1.set_as(self.corners_ub[0])
                        self.b1_y1.set_as(self.corners_ub[Constant.BLOCK])
                        self.get_area_of_triangle(1, 2)
                        with self.tik_instance.if_scope(self.value > 0):
                            overlap.set_as(self.value / 2)
                        with self.tik_instance.else_scope():
                            overlap.set_as(-1 * self.value / 2)
                        with self.tik_instance.if_scope(b1_area + b2_area - overlap > 0):
                            self.overlap_ub[b2_idx].set_as(
                                overlap / (b1_area + b2_area - overlap + Constant.EPSILON))
                    with self.tik_instance.if_scope(self.corners_num > 3):
                        self.sum_area_of_triangles(b2_idx)
                        overlap.set_as(self.value / 2)
                        with self.tik_instance.if_scope(b1_area + b2_area - overlap > 0):
                            self.overlap_ub[b2_idx].set_as(
                                overlap / (b1_area + b2_area - overlap + Constant.EPSILON))
                self.tik_instance.data_move(
                    self.overlaps_gm[self.k * (task_idx * self.b1_batch + b1_idx + current_batch * self.n)],
                    self.overlap_ub, 0, 1, self.mov_repeats, 0, 0)


# 'pylint:disable=too-many-arguments, disable=too-many-statements
@register_operator_compute("rotated_iou", op_mode="static", support_fusion=True)
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def rotated_iou(boxes, query_boxes, iou, 
                trans=False, mode="iou", is_cross=True, v_threshold=0, e_threshold=0,
                kernel_name="rotated_iou"):
    """
    Function: compute the rotated boxes's iou.
    Modify : 2021-12-01

    Init base parameters
    Parameters
    ----------
    input(boxes): dict
        data of input
    input(query_boxes): dict
        data of input
    output(iou): dict
        data of output

    Attributes:
    trans : bool
        true for 'xyxyt', false for 'xywht'
    mode: string
        with the value range of ['iou', 'iof'], only support 'iou' now.
    is_cross: bool
        cross calculation when it is True, and one-to-one calculation when it is False.

    kernel_name: str
        the name of the operator
    ----------
    """
    op_obj = RotatedIou(boxes, query_boxes, iou,
                        trans, mode, is_cross, v_threshold, e_threshold, kernel_name)

    return op_obj.compute()
