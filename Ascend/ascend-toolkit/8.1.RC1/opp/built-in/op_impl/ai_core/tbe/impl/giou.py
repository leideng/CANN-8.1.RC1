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
giou
"""

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from tbe.common.platform.platform_info import get_soc_spec


# pylint: disable=too-few-public-methods,invalid-name,unused-variable
class Constant:
    """
    The class for constant.
    """
    FP16_ELIMENTS_BLOCK = 16
    FP32_ELIMENTS_BLOCK = 8
    GTBOX_SEGMENT = 2048
    BBOX_SEGMENT = 2048
    UB_FOR_TIK = 16384
    UB_NUM = 50
    MASK = 256


def _apply_mem(tik_instance, dtype, shape, name, scope=tik.scope_ubuf):
    return tik_instance.Tensor(dtype, shape, name=name, scope=scope)


def _get_ceil_int(int1, int2):
    ceil_int = (int1 + int2 - 1) // int2
    return ceil_int


def GetMaxSegment(ub_size, byte_per_data):
    ub_size = ub_size - Constant.UB_FOR_TIK
    max_align = ub_size // (Constant.UB_NUM * byte_per_data)
    avail_align = (max_align // Constant.MASK) * Constant.MASK

    return avail_align


# pylint: disable=too-many-instance-attributes,too-many-lines
class GIoU():
    """Function: use to finish Iou main functions
    """

    # pylint: disable=too-many-statements,too-many-arguments
    def __init__(self, bboxes, gtboxes, trans, is_cross, mode):
        self.bboxes_shape = bboxes.get("shape")
        self.bboxes_dtype = bboxes.get("dtype").lower()
        self.gtboxes_shape = gtboxes.get("shape")
        self.gtboxes_dtype = gtboxes.get("dtype").lower()
        self.boxes_num = self.gtboxes_shape[1]
        self.dtype = self.bboxes_dtype
        self.trans = trans
        self.is_cross = is_cross
        self.mode = mode.lower()
        self.tik_instance = tik.Tik()
        self.product = tbe_platform.api_check_support("tik.vdiv", "float32")
        self.available_ub_size = get_soc_spec("UB_SIZE")

        # input and output tensor in gm
        self.giou_shape = [1, self.boxes_num]
        self.bboxes_gm = self.tik_instance.Tensor(self.bboxes_dtype, self.bboxes_shape,
                                                  name="bboxes_gm", scope=tik.scope_gm)
        self.gtboxes_gm = self.tik_instance.Tensor(self.gtboxes_dtype, self.gtboxes_shape,
                                                   name="gtboxes_gm", scope=tik.scope_gm)
        self.giou_gm = self.tik_instance.Tensor(self.bboxes_dtype, self.giou_shape, name="giou_gm", scope=tik.scope_gm)

        # init attr in objext
        self.bboxes_x0 = self.bboxes_x1 = self.bboxes_y0 = self.bboxes_y1 = None
        self.gtboxes_x0 = self.gtboxes_x1 = self.gtboxes_y0 = self.gtboxes_y1 = None
        self.inter_area_x0 = self.inter_area_x1 = self.inter_area_y0 = self.inter_area_y1 = None
        self.outer_area_x0 = self.outer_area_x1 = self.outer_area_y0 = self.outer_area_y1 = None
        block_parm_dict = {"float16": Constant.FP16_ELIMENTS_BLOCK, "float32": Constant.FP32_ELIMENTS_BLOCK}
        self.eliments_per_block = block_parm_dict.get(self.bboxes_dtype)
        if self.bboxes_dtype == "float32":
            self.bb_ub_segment = Constant.BBOX_SEGMENT // 2
            self.ub_max_segment = GetMaxSegment(self.available_ub_size, 4)
        else:
            self.bb_ub_segment = Constant.BBOX_SEGMENT
            self.ub_max_segment = GetMaxSegment(self.available_ub_size, 2)
        
        if self.bb_ub_segment > self.ub_max_segment:
            self.bb_ub_segment = self.ub_max_segment
        self.max_eliments = block_parm_dict.get(self.bboxes_dtype) * 8

    # pylint: disable=too-many-locals,too-many-branches,too-many-lines,too-many-statements
    def giou_process(self):
        """do process and scedule
           main function

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        task_num = _get_ceil_int(self.boxes_num, self.bb_ub_segment)
        bb_ub_segment_tail = self.boxes_num % self.bb_ub_segment
        if not bb_ub_segment_tail:
            bb_ub_segment_tail = self.bb_ub_segment

        repeat_time_max = self.bb_ub_segment // self.max_eliments

        with self.tik_instance.for_range(0, task_num, block_num=task_num) as _task_id:
            dst_gm_offset = self.bb_ub_segment * _task_id
            with self.tik_instance.if_scope(_task_id < task_num - 1):
                self._run_segment(self.max_eliments, repeat_time_max, self.bb_ub_segment, dst_gm_offset)
            with self.tik_instance.else_scope():
                self._run_segment(self.max_eliments, repeat_time_max, bb_ub_segment_tail, dst_gm_offset)

    def run_tik(self, kernel_name):
        self.giou_process()
        self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=[self.bboxes_gm, self.gtboxes_gm],
                                   outputs=[self.giou_gm])
        return self.tik_instance

    def data_move_in_and_trans(self, mask, repeat_time, one_loop_shape, gm_offset, nbust):
        boxes_xy = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "boxes_xy")
        boxes_wh = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "boxes_wh")

        self.tik_instance.data_move(boxes_xy, self.gtboxes_gm[gm_offset], 0, 1, nbust, 0, 0)
        self.tik_instance.data_move(boxes_wh, self.gtboxes_gm[gm_offset + self.bboxes_shape[1] * 2], 0, 1, nbust, 0, 0)
        self.tik_instance.vmuls(mask, boxes_wh, boxes_wh, 0.5, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vsub(mask, self.gtboxes_x0, boxes_xy, boxes_wh, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(mask, self.gtboxes_x1, boxes_xy, boxes_wh, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.data_move(boxes_xy, self.gtboxes_gm[gm_offset + self.bboxes_shape[1]], 0, 1, nbust, 0, 0)
        self.tik_instance.data_move(boxes_wh, self.gtboxes_gm[gm_offset + self.bboxes_shape[1] * 3], 0, 1, nbust, 0, 0)
        self.tik_instance.vmuls(mask, boxes_wh, boxes_wh, 0.5, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vsub(mask, self.gtboxes_y0, boxes_xy, boxes_wh, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(mask, self.gtboxes_y1, boxes_xy, boxes_wh, repeat_time, 1, 1, 1, 8, 8, 8)

        self.tik_instance.data_move(boxes_xy, self.bboxes_gm[gm_offset], 0, 1, nbust, 0, 0)
        self.tik_instance.data_move(boxes_wh, self.bboxes_gm[gm_offset + self.bboxes_shape[1] * 2], 0, 1, nbust, 0, 0)
        self.tik_instance.vmuls(mask, boxes_wh, boxes_wh, 0.5, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vsub(mask, self.bboxes_x0, boxes_xy, boxes_wh, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(mask, self.bboxes_x1, boxes_xy, boxes_wh, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.data_move(boxes_xy, self.bboxes_gm[gm_offset + self.bboxes_shape[1]], 0, 1, nbust, 0, 0)
        self.tik_instance.data_move(boxes_wh, self.bboxes_gm[gm_offset + self.bboxes_shape[1] * 3], 0, 1, nbust, 0, 0)
        self.tik_instance.vmuls(mask, boxes_wh, boxes_wh, 0.5, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vsub(mask, self.bboxes_y0, boxes_xy, boxes_wh, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(mask, self.bboxes_y1, boxes_xy, boxes_wh, repeat_time, 1, 1, 1, 8, 8, 8)

    def data_move_in(self, gm_offset, nbust):
        self.tik_instance.data_move(self.gtboxes_x0, self.gtboxes_gm[gm_offset], 0, 1, nbust, 0, 0)
        self.tik_instance.data_move(self.gtboxes_y0, self.gtboxes_gm[gm_offset + self.bboxes_shape[1]],
                                    0, 1, nbust, 0, 0)
        self.tik_instance.data_move(self.gtboxes_x1, self.gtboxes_gm[gm_offset + self.bboxes_shape[1] * 2],
                                    0, 1, nbust, 0, 0)
        self.tik_instance.data_move(self.gtboxes_y1, self.gtboxes_gm[gm_offset + self.bboxes_shape[1] * 3],
                                    0, 1, nbust, 0, 0)

        self.tik_instance.data_move(self.bboxes_x0, self.bboxes_gm[gm_offset], 0, 1, nbust, 0, 0)
        self.tik_instance.data_move(self.bboxes_y0, self.bboxes_gm[gm_offset + self.bboxes_shape[1]],
                                    0, 1, nbust, 0, 0)
        self.tik_instance.data_move(self.bboxes_x1, self.bboxes_gm[gm_offset + self.bboxes_shape[1] * 2],
                                    0, 1, nbust, 0, 0)
        self.tik_instance.data_move(self.bboxes_y1, self.bboxes_gm[gm_offset + self.bboxes_shape[1] * 3],
                                    0, 1, nbust, 0, 0)

    def get_inter_outer_area(self):
        self.tik_instance.h_max(self.inter_area_x0, self.bboxes_x0, self.gtboxes_x0)
        self.tik_instance.h_max(self.inter_area_y0, self.bboxes_y0, self.gtboxes_y0)
        self.tik_instance.h_min(self.inter_area_x1, self.bboxes_x1, self.gtboxes_x1)
        self.tik_instance.h_min(self.inter_area_y1, self.bboxes_y1, self.gtboxes_y1)

        self.tik_instance.h_min(self.outer_area_x0, self.bboxes_x0, self.gtboxes_x0)
        self.tik_instance.h_min(self.outer_area_y0, self.bboxes_y0, self.gtboxes_y0)
        self.tik_instance.h_max(self.outer_area_x1, self.bboxes_x1, self.gtboxes_x1)
        self.tik_instance.h_max(self.outer_area_y1, self.bboxes_y1, self.gtboxes_y1)

    # pylint: disable=too-many-arguments
    def calcu_area(self, mask, repeat_time, area_ub, one_loop_shape, inter_mode=False, outer_mode=False, gt_mode=False):
        if inter_mode:
            x0_ub = self.inter_area_x0
            x1_ub = self.inter_area_x1
            y0_ub = self.inter_area_y0
            y1_ub = self.inter_area_y1
        elif outer_mode:
            x0_ub = self.outer_area_x0
            x1_ub = self.outer_area_x1
            y0_ub = self.outer_area_y0
            y1_ub = self.outer_area_y1
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
        area_y1_y0 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "area_y1_y0")
        self.tik_instance.vsub(mask, area_y1_y0, y1_ub, y0_ub, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(mask, area_ub, x1_ub, x0_ub, repeat_time, 1, 1, 1, 8, 8, 8)
        if inter_mode:
            zero_ub = _apply_mem(self.tik_instance, self.dtype, [self.eliments_per_block], "zero_ub")
            self.tik_instance.vector_dup(self.eliments_per_block, zero_ub, 0.0, 1, 1, 8)
            self.tik_instance.vmax(mask, area_ub, zero_ub, area_ub, repeat_time, 1, 0, 1, 8, 0, 8)
            self.tik_instance.vmax(mask, area_y1_y0, zero_ub, area_y1_y0, repeat_time, 1, 0, 1, 8, 0, 8)
        else:
            self.tik_instance.vadds(mask, area_ub, area_ub, 1e-16, repeat_time, 1, 1, 8, 8)
            self.tik_instance.vadds(mask, area_y1_y0, area_y1_y0, 1e-16, repeat_time, 1, 1, 8, 8)

        self.tik_instance.vmul(mask, area_ub, area_y1_y0, area_ub, repeat_time, 1, 1, 1, 8, 8, 8)

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
        
    # pylint: disable=too-many-locals,too-many-branches,too-many-lines
    def _run_segment(self, mask, repeat_time, ub_segment, gm_offset):
        """
        do a segment of bbox compute
        """
        one_loop_shape = mask * repeat_time
        self._apply_all_ub(one_loop_shape)
        nbust = _get_ceil_int(one_loop_shape, self.eliments_per_block)

        # copy gm to ub
        if not self.trans:
            self.data_move_in(gm_offset, nbust)
        else:
            self.data_move_in_and_trans(mask, repeat_time, one_loop_shape, gm_offset, nbust)

        gtboxes_area_ub = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "gtboxes_area_ub")
        bboxes_area_ub = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "bboxes_area_ub")
        inter_area_ub = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "inter_area_ub")
        outer_area_ub = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "outer_area_ub")
        out_ub = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "out_ub")
        other_ub = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "other_ub")

        # calcu bboxes area
        self.calcu_area(mask, repeat_time, bboxes_area_ub, one_loop_shape)

        # calcu gtboxes area
        self.calcu_area(mask, repeat_time, gtboxes_area_ub, one_loop_shape, gt_mode=True)

        # vmin vmax: get inter x0 x1 y0 y1, outer x0 x1 y0 y1
        self.get_inter_outer_area()

        # calcu inter area
        self.calcu_area(mask, repeat_time, inter_area_ub, one_loop_shape, inter_mode=True)

        # calcu outer area
        self.calcu_area(mask, repeat_time, outer_area_ub, one_loop_shape, outer_mode=True)

        if self.mode == "iou":
            self.tik_instance.vadd(mask, out_ub, bboxes_area_ub,
                                   gtboxes_area_ub, repeat_time, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vsub(mask, out_ub, out_ub, inter_area_ub, repeat_time, 1, 1, 1, 8, 8, 8)
        elif self.mode == "iof":
            self.tik_instance.data_move(out_ub, gtboxes_area_ub, 0, 1, nbust, 0, 0)

        self.tik_instance.vsub(mask, other_ub, outer_area_ub, out_ub,
                               repeat_time, 1, 1, 1, 8, 8, 8)

        if self.product is True:
            self.tik_instance.vdiv(mask, out_ub, inter_area_ub, out_ub, repeat_time, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vdiv(mask, outer_area_ub, other_ub,
                                   outer_area_ub, repeat_time, 1, 1, 1, 8, 8, 8)
        else:
            # for mini
            self._rev_div(mask, repeat_time, out_ub, inter_area_ub, out_ub, one_loop_shape)
            self._rev_div(mask, repeat_time, outer_area_ub, other_ub, outer_area_ub, one_loop_shape)

        self.tik_instance.vsub(mask, out_ub, out_ub, outer_area_ub, repeat_time, 1, 1, 1, 8, 8, 8)

        move_time = _get_ceil_int(ub_segment, self.eliments_per_block)
        self.tik_instance.data_move(self.giou_gm[gm_offset], out_ub, 0, 1, move_time, 0, 0)

    def _rev_div(self, mask, repeat_time, x1_ub, x2_ub, y_ub, one_loop_shape):
        div_rec_1 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "div_rec_1")
        div_rec_2 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "div_rec_2")

        self.tik_instance.vrec(mask, div_rec_1, x1_ub, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vmul(mask, div_rec_2, div_rec_1, x1_ub, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmuls(mask, div_rec_2, div_rec_2, -1, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vadds(mask, div_rec_2, div_rec_2, 2, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vmul(mask, div_rec_2, div_rec_2, div_rec_1, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, div_rec_1, div_rec_2, x1_ub, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmuls(mask, div_rec_1, div_rec_1, -1, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vadds(mask, div_rec_1, div_rec_1, 2, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vmul(mask, div_rec_1, div_rec_1, div_rec_2, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, y_ub, div_rec_1, x2_ub, repeat_time, 1, 1, 1, 8, 8, 8)


def _box_shape_check(input_name, shape):
    shape_len = len(shape)
    if shape_len != 2:
        error_detail = "the shape len should be 2"
        error_manager_vector.raise_err_input_shape_invalid("giou", input_name, error_detail)
    first_shape_dim = shape[0]
    if first_shape_dim != 4:
        error_detail = "the shape should be [4, n]"
        error_manager_vector.raise_err_input_shape_invalid("giou", input_name, error_detail)


# pylint: disable=too-many-arguments
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def giou(bboxes, gtboxes, overlap, trans=False, is_cross=True, mode="iou", kernel_name="giou"):
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
        kernel name, default value is "giou"

    Returns
    -------
    None
    """
    bboxes_shape = bboxes.get("shape")
    gtboxes_shape = gtboxes.get("shape")

    # check whether mode is valid
    check_list = ("iou", "iof")
    if mode not in check_list:
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "mode", "iou,iof", mode)

    _box_shape_check("bboxes", bboxes_shape)
    _box_shape_check("gtboxes", gtboxes_shape)
    bboxes_dtype = bboxes.get("dtype").lower()
    shape_util.compare_tensor_dict_key(bboxes, gtboxes, "dtype")
    check_list = ("float16", "float32")
    para_check.check_dtype(bboxes_dtype, check_list, param_name="bboxes")

    giou_obj = GIoU(bboxes, gtboxes, trans, is_cross, mode)
    res = giou_obj.run_tik(kernel_name)

    return res
