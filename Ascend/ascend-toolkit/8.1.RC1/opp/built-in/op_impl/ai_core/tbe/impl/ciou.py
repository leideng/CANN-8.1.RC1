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
from impl.util.util_select_op_base import get_op_cal_info
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput


# pylint: disable=too-few-public-methods,invalid-name,unused-variable
class Constant:
    """
    The class for constant.
    """
    FP16_ELIMENTS_BLOCK = 16
    FP32_ELIMENTS_BLOCK = 8
    GTBOX_SEGMENT = 1024
    BBOX_SEGMENT = 1024
    CONST_PI_BY_FOUR = 0.78539816
    CONST_PI_BY_EIGHT = 0.39269908
    CONST_ITERTOR = 6
    CONST_INERTOR2 = 4
    TAN_PI_BY_EIGHT = 0.41421356
    NEG_TAN_PI_BY_EIGHT = -0.41421356
    TAYLOR = (1.0, -1.0 / 3, 1.0 / 5, -1.0 / 7, 1.0 / 9, -1.0 / 11, 1.0 / 13)


def _apply_mem(tik_instance, dtype, shape, name, scope=tik.scope_ubuf):
    """apply mem fuc

    Parameters
    ----------
    tik_instance: tik_instance
        tik_instance
    dtype: str
        ub dtype
    shape: list
        ub shape
    name: str
        ub name
    scope: scope
        scope_ubuf or scope_gm
    Returns
    -------
    Tensor: Tensor
    """
    return tik_instance.Tensor(dtype, shape, name=name, scope=scope)


def _get_ceil_int(int1, int2):
    """Get Ceil Int

    Parameters
    ----------
    int1: int
        input int 1
    int2: int
        input int 2

    Returns
    -------
    ceil_int: int
    """
    ceil_int = (int1 + int2 - 1) // int2
    return ceil_int


# pylint: disable=too-many-instance-attributes,too-many-lines
class CIoU():
    """Function: use to finish Iou main functions
    """

    # pylint: disable=too-many-statements,too-many-arguments
    def __init__(self, bboxes, gtboxes, trans, is_cross, mode, atan_sub_flag):
        self.bboxes_shape = bboxes.get("shape")
        self.bboxes_dtype = bboxes.get("dtype").lower()
        self.gtboxes_shape = gtboxes.get("shape")
        self.gtboxes_dtype = gtboxes.get("dtype").lower()
        self.boxes_num = self.gtboxes_shape[1]
        self.dtype = self.bboxes_dtype
        self.trans = trans
        self.mode = mode.lower()
        self.tik_instance = tik.Tik()
        self.core_num = tik.Dprofile().get_aicore_num()
        self.product = tbe_platform.api_check_support("tik.vdiv", "float32")
        # input and output tensor in gm
        self.ciou_shape = [1, self.boxes_num]
        self.bboxes_gm = self.tik_instance.Tensor(self.bboxes_dtype, self.bboxes_shape,
                                                  name="bboxes_gm", scope=tik.scope_gm)
        self.gtboxes_gm = self.tik_instance.Tensor(self.gtboxes_dtype, self.gtboxes_shape,
                                                   name="gtboxes_gm", scope=tik.scope_gm)
        self.ciou_gm = self.tik_instance.Tensor(self.bboxes_dtype, self.ciou_shape, name="ciou_gm", scope=tik.scope_gm)
        self.atan_sub_gm = self.tik_instance.Tensor(self.bboxes_dtype, self.ciou_shape,
                                                    name="atan_sub_gm", scope=tik.scope_gm)

        # init attr in objext
        self.point_per_core = self.core_tail_num = self.bb_ub_segment = 0
        self.bboxes_x0 = self.bboxes_x1 = self.bboxes_y0 = self.bboxes_y1 = None
        self.gtboxes_x0 = self.gtboxes_x1 = self.gtboxes_y0 = self.gtboxes_y1 = None
        self.inter_area_x0 = self.inter_area_x1 = self.inter_area_y0 = self.inter_area_y1 = None
        self.outer_area_x0 = self.outer_area_x1 = self.outer_area_y0 = self.outer_area_y1 = None
        self.in_square = self.out_square = self.bboxes_w_h = self.gtboxes_w_h = None
        self.x_sub_one = self.x_add_one = self.square_x = self.div_rec_1 = self.div_rec_2 = None
        self.atan_w_h_quarter_pi = self.gtboxes_atan_w_h_quarter_pi = self.denominator_data = None
        self.atan_w_h = self.gtboxes_atan_w_h = self.atan_temp_w_h = self.max_w_h_ub = self.v = self.alpha_ub = None
        self.area_y1_y0 = self.sum_y1_y0 = self.gtboxes_area_ub = self.out_ub = None
        self.other_ub = self.bboxes_area_ub = self.inter_area_ub = self.zero_ub = None
        block_parm_dict = {"float16": Constant.FP16_ELIMENTS_BLOCK, "float32": Constant.FP32_ELIMENTS_BLOCK}
        self.min_point_per_core = block_parm_dict.get(self.bboxes_dtype)
        self.eliments_per_block = block_parm_dict.get(self.bboxes_dtype)
        if self.bboxes_dtype == "float16":
            self.bb_ub_segment = Constant.BBOX_SEGMENT
        else:
            self.bb_ub_segment = Constant.BBOX_SEGMENT // 2
        self.max_eliments = block_parm_dict.get(self.bboxes_dtype) * 8

    # pylint: disable=too-many-locals,too-many-branches,too-many-lines,too-many-statements
    def ciou_process(self):
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
        if task_num == 1:
            self.bb_ub_segment = _get_ceil_int(self.boxes_num, self.max_eliments) * self.max_eliments
        repeat_time_max = self.bb_ub_segment // self.max_eliments

        with self.tik_instance.for_range(0, task_num, block_num=task_num) as _task_id:
            dst_gm_offset = self.bb_ub_segment * _task_id
            self._run_segment(self.max_eliments, repeat_time_max, dst_gm_offset)

    def run_tik(self, kernel_name):
        self.ciou_process()
        self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=[self.bboxes_gm, self.gtboxes_gm],
                                   outputs=[self.ciou_gm, self.atan_sub_gm])
        return self.tik_instance

    def data_move_in_and_trans(self, mask, repeat_time, one_loop_shape, gm_offset, nbust):
        boxes_xy = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "boxes_xy")
        boxes_wh = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "boxes_wh")
        self.tik_instance.data_move(boxes_xy, self.bboxes_gm[gm_offset], 0, 1, nbust, 0, 0)
        self.tik_instance.data_move(boxes_wh, self.bboxes_gm[gm_offset + self.bboxes_shape[1] * 2], 0, 1, nbust, 0, 0)
        self.tik_instance.vmuls(mask, boxes_wh, boxes_wh, 0.5, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vsub(mask, self.bboxes_x0, boxes_xy, boxes_wh, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(mask, self.bboxes_x1, boxes_xy, boxes_wh, repeat_time, 1, 1, 1, 8, 8, 8)

        self.tik_instance.data_move(boxes_xy, self.gtboxes_gm[gm_offset], 0, 1, nbust, 0, 0)
        self.tik_instance.data_move(boxes_wh, self.gtboxes_gm[gm_offset + self.bboxes_shape[1] * 2], 0, 1, nbust, 0, 0)
        self.tik_instance.vmuls(mask, boxes_wh, boxes_wh, 0.5, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vsub(mask, self.gtboxes_x0, boxes_xy, boxes_wh, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(mask, self.gtboxes_x1, boxes_xy, boxes_wh, repeat_time, 1, 1, 1, 8, 8, 8)

        self.tik_instance.data_move(boxes_xy, self.bboxes_gm[gm_offset + self.bboxes_shape[1]], 0, 1, nbust, 0, 0)
        self.tik_instance.data_move(boxes_wh, self.bboxes_gm[gm_offset + self.bboxes_shape[1] * 3], 0, 1, nbust, 0, 0)
        self.tik_instance.vmuls(mask, boxes_wh, boxes_wh, 0.5, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vsub(mask, self.bboxes_y0, boxes_xy, boxes_wh, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(mask, self.bboxes_y1, boxes_xy, boxes_wh, repeat_time, 1, 1, 1, 8, 8, 8)

        self.tik_instance.data_move(boxes_xy, self.gtboxes_gm[gm_offset + self.bboxes_shape[1]], 0, 1, nbust, 0, 0)
        self.tik_instance.data_move(boxes_wh, self.gtboxes_gm[gm_offset + self.bboxes_shape[1] * 3], 0, 1, nbust, 0, 0)
        self.tik_instance.vmuls(mask, boxes_wh, boxes_wh, 0.5, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vsub(mask, self.gtboxes_y0, boxes_xy, boxes_wh, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(mask, self.gtboxes_y1, boxes_xy, boxes_wh, repeat_time, 1, 1, 1, 8, 8, 8)

    def data_move_in(self, gm_offset, nbust):
        self.tik_instance.data_move(self.bboxes_x0, self.bboxes_gm[gm_offset], 0, 1, nbust, 0, 0)
        self.tik_instance.data_move(self.bboxes_y0, self.bboxes_gm[gm_offset + self.bboxes_shape[1]],
                                    0, 1, nbust, 0, 0)
        self.tik_instance.data_move(self.bboxes_x1, self.bboxes_gm[gm_offset + self.bboxes_shape[1] * 2],
                                    0, 1, nbust, 0, 0)
        self.tik_instance.data_move(self.bboxes_y1, self.bboxes_gm[gm_offset + self.bboxes_shape[1] * 3],
                                    0, 1, nbust, 0, 0)

        self.tik_instance.data_move(self.gtboxes_x0, self.gtboxes_gm[gm_offset], 0, 1, nbust, 0, 0)
        self.tik_instance.data_move(self.gtboxes_y0, self.gtboxes_gm[gm_offset + self.bboxes_shape[1]],
                                    0, 1, nbust, 0, 0)
        self.tik_instance.data_move(self.gtboxes_x1, self.gtboxes_gm[gm_offset + self.bboxes_shape[1] * 2],
                                    0, 1, nbust, 0, 0)
        self.tik_instance.data_move(self.gtboxes_y1, self.gtboxes_gm[gm_offset + self.bboxes_shape[1] * 3],
                                    0, 1, nbust, 0, 0)

    def get_inter_outer_area(self, mask, repeat_time):
        self.tik_instance.h_min(self.outer_area_x0, self.bboxes_x0, self.gtboxes_x0)
        self.tik_instance.h_min(self.outer_area_y0, self.bboxes_y0, self.gtboxes_y0)
        self.tik_instance.h_max(self.outer_area_x1, self.bboxes_x1, self.gtboxes_x1)
        self.tik_instance.h_max(self.outer_area_y1, self.bboxes_y1, self.gtboxes_y1)

        self.tik_instance.h_max(self.inter_area_x0, self.bboxes_x0, self.gtboxes_x0)
        self.tik_instance.h_max(self.inter_area_y0, self.bboxes_y0, self.gtboxes_y0)
        self.tik_instance.h_min(self.inter_area_x1, self.bboxes_x1, self.gtboxes_x1)
        self.tik_instance.h_min(self.inter_area_y1, self.bboxes_y1, self.gtboxes_y1)

    # pylint: disable=too-many-arguments
    def calcu_area(self, mask, repeat_time, area_ub, w_h_ub=None, inter_mode=False, gt_mode=False):
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
        # cala x1 - x0
        self.tik_instance.vsub(mask, area_ub, x1_ub, x0_ub, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(mask, self.area_y1_y0, y1_ub, y0_ub, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadds(mask, self.area_y1_y0, self.area_y1_y0, 1e-9, repeat_time, 1, 1, 8, 8)
        if inter_mode:
            self.tik_instance.vmax(mask, area_ub, self.zero_ub, area_ub, repeat_time, 1, 0, 1, 8, 0, 8)
            self.tik_instance.vmax(mask, self.area_y1_y0, self.zero_ub, self.area_y1_y0, repeat_time, 1, 0, 1, 8, 0, 8)
        else:
            if self.product is True:
                self.tik_instance.vdiv(mask, w_h_ub, area_ub, self.area_y1_y0, repeat_time, 1, 1, 1, 8, 8, 8)
            else:
                self._rev_div(mask, repeat_time, self.area_y1_y0, area_ub, w_h_ub)
            self.tik_instance.vmin(mask, w_h_ub, self.max_w_h_ub, w_h_ub, repeat_time, 1, 0, 1, 8, 0, 8)
            self.tik_instance.vadds(mask, area_ub, area_ub, 1e-16, repeat_time, 1, 1, 8, 8)
            self.tik_instance.vadds(mask, self.area_y1_y0, self.area_y1_y0, 1e-16, repeat_time, 1, 1, 8, 8)

        self.tik_instance.vmul(mask, area_ub, self.area_y1_y0, area_ub, repeat_time, 1, 1, 1, 8, 8, 8)

    def calcu_in_square(self, mask, repeat_time):
        self.tik_instance.vadd(mask, self.in_square, self.bboxes_x0, self.bboxes_x1, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(mask, self.in_square, self.in_square, self.gtboxes_x0, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(mask, self.in_square, self.in_square, self.gtboxes_x1, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, self.in_square, self.in_square, self.in_square, repeat_time, 1, 1, 1, 8, 8, 8)

        self.tik_instance.vadd(mask, self.sum_y1_y0, self.bboxes_y0, self.bboxes_y1, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(mask, self.sum_y1_y0, self.sum_y1_y0, self.gtboxes_y0, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(mask, self.sum_y1_y0, self.sum_y1_y0, self.gtboxes_y1, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, self.sum_y1_y0, self.sum_y1_y0, self.sum_y1_y0, repeat_time, 1, 1, 1, 8, 8, 8)

        self.tik_instance.vadd(mask, self.in_square, self.in_square, self.sum_y1_y0, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmuls(mask, self.in_square, self.in_square, 0.25, repeat_time, 1, 1, 8, 8)

    def calcu_out_square(self, mask, repeat_time):
        self.tik_instance.vsub(mask, self.out_square, self.outer_area_x1, self.outer_area_x0,
                               repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(mask, self.area_y1_y0, self.outer_area_y1, self.outer_area_y0,
                               repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, self.out_square, self.out_square, self.out_square,
                               repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, self.area_y1_y0, self.area_y1_y0, self.area_y1_y0,
                               repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(mask, self.out_square, self.out_square, self.area_y1_y0,
                               repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadds(mask, self.out_square, self.out_square, 1e-16, repeat_time, 1, 1, 8, 8)

    def calcu_atan(self, mask, one_loop_shape, x_ub, atan_ub, atan_quarter_pi_ub):
        repeat_time = _get_ceil_int(one_loop_shape, mask)
        self.tik_instance.vadds(mask, self.x_sub_one, x_ub, -1, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vadds(mask, self.x_add_one, x_ub, 1, repeat_time, 1, 1, 8, 8)
        if self.product is True:
            self.tik_instance.vdiv(mask, self.x_sub_one, self.x_sub_one, self.x_add_one, repeat_time, 1, 1, 1, 8, 8, 8)
        else:
            self._rev_div(mask, repeat_time, self.x_add_one, self.x_sub_one, self.x_sub_one)
        self.tik_instance.vabs(mask, self.x_sub_one, self.x_sub_one, repeat_time, 1, 1, 8, 8)
        self.do_taylor(mask, repeat_time, x_ub, atan_ub)
        self.do_taylor(mask, repeat_time, self.x_sub_one, atan_quarter_pi_ub)
        self.tik_instance.vadds(mask, atan_quarter_pi_ub, atan_quarter_pi_ub, Constant.CONST_PI_BY_FOUR,
                                repeat_time, 1, 1, 8, 8)
        self.tik_instance.h_min(atan_ub, atan_ub, atan_quarter_pi_ub)

    def do_taylor(self, mask, repeat_time, data_x, atan_res_ub):
        self.tik_instance.vmul(mask, self.square_x, data_x, data_x, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmuls(mask, atan_res_ub, atan_res_ub, 0.0, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vadds(mask, atan_res_ub, atan_res_ub,
                                Constant.TAYLOR[Constant.CONST_INERTOR2], repeat_time, 1, 1, 8, 8)
        for i in reversed(range(Constant.CONST_INERTOR2)):
            self.tik_instance.vmul(mask, atan_res_ub, atan_res_ub, self.square_x,
                                   repeat_time, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vadds(mask, atan_res_ub, atan_res_ub, Constant.TAYLOR[i],
                                    repeat_time, 1, 1, 8, 8)
        self.tik_instance.vmul(mask, atan_res_ub, atan_res_ub, data_x,
                               repeat_time, 1, 1, 1, 8, 8, 8)

        self.tik_instance.vmuls(mask, self.denominator_data, data_x, Constant.TAN_PI_BY_EIGHT, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vadds(mask, self.denominator_data, self.denominator_data, 1.0, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vadds(mask, self.square_x, data_x, Constant.NEG_TAN_PI_BY_EIGHT,
                                repeat_time, 1, 1, 8, 8)
        if self.product is True:
            self.tik_instance.vdiv(mask, data_x, self.square_x, self.denominator_data, repeat_time, 1, 1, 1, 8, 8, 8)
        else:
            self._rev_div(mask, repeat_time, self.denominator_data, self.square_x, data_x)
        self.tik_instance.vabs(mask, data_x, data_x, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vmul(mask, self.square_x, data_x, data_x, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmuls(mask, self.atan_temp_w_h, self.atan_temp_w_h, 0.0, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vadds(mask, self.atan_temp_w_h, self.atan_temp_w_h,
                                Constant.TAYLOR[Constant.CONST_ITERTOR], repeat_time, 1, 1, 8, 8)
        for i in reversed(range(Constant.CONST_ITERTOR)):
            self.tik_instance.vmul(mask, self.atan_temp_w_h, self.atan_temp_w_h, self.square_x,
                                   repeat_time, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vadds(mask, self.atan_temp_w_h, self.atan_temp_w_h, Constant.TAYLOR[i],
                                    repeat_time, 1, 1, 8, 8)
        self.tik_instance.vmul(mask, self.atan_temp_w_h, self.atan_temp_w_h, data_x,
                               repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadds(mask, self.atan_temp_w_h, self.atan_temp_w_h, Constant.CONST_PI_BY_EIGHT,
                                repeat_time, 1, 1, 8, 8)

        self.tik_instance.h_min(atan_res_ub, atan_res_ub, self.atan_temp_w_h)

    def _apply_all_ub(self, one_loop_shape):
        self.bboxes_x0 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "bboxes_x0")
        self.gtboxes_x0 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "gtboxes_x0")
        self.inter_area_x0 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "inter_area_x0")
        self.outer_area_x0 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "outer_area_x0")
        self.bboxes_x1 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "bboxes_x1")
        self.gtboxes_x1 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "gtboxes_x1")
        self.inter_area_x1 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "inter_area_x1")
        self.outer_area_x1 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "outer_area_x1")
        self.bboxes_y0 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "bboxes_y0")
        self.gtboxes_y0 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "gtboxes_y0")
        self.inter_area_y0 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "inter_area_y0")
        self.outer_area_y0 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "outer_area_y0")
        self.bboxes_y1 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "bboxes_y1")
        self.gtboxes_y1 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "gtboxes_y1")
        self.inter_area_y1 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "inter_area_y1")
        self.outer_area_y1 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "outer_area_y1")
        self.area_y1_y0 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "area_y1_y0")
        self.sum_y1_y0 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "sum_y1_y0")
        self.in_square = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "in_square")
        self.out_square = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "out_square")
        self.other_ub = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "other_ub")
        self.gtboxes_area_ub = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "gtboxes_area_ub")
        self.bboxes_area_ub = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "bboxes_area_ub")
        self.inter_area_ub = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "inter_area_ub")
        self.bboxes_w_h = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "bboxes_w_h")
        self.gtboxes_w_h = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "gtboxes_w_h")
        self.zero_ub = _apply_mem(self.tik_instance, self.dtype, [self.eliments_per_block], "zero_ub")
        self.tik_instance.vector_dup(self.eliments_per_block, self.zero_ub, 0.0, 1, 1, 8)
        self.max_w_h_ub = _apply_mem(self.tik_instance, self.dtype, [self.eliments_per_block], "max_w_h_ub")
        self.tik_instance.vector_dup(self.eliments_per_block, self.max_w_h_ub, 10000.0, 1, 1, 8)
        self.x_sub_one = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "x_sub_one")
        self.x_add_one = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "x_add_one")
        self.square_x = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "square_x")
        self.atan_w_h = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "atan_w_h")
        self.gtboxes_atan_w_h = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "gtboxes_atan_w_h")
        self.atan_w_h_quarter_pi = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "atan_w_h_quarter_pi")
        self.gtboxes_atan_w_h_quarter_pi = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape],
                                                      "gtboxes_atan_w_h_quarter_pi")
        self.denominator_data = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "denominator_data")
        self.atan_temp_w_h = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "atan_temp_w_h")
        self.v = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "v")
        self.alpha_ub = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "alpha_ub")
        self.div_rec_1 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "div_rec_1")
        self.div_rec_2 = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "div_rec_2")
        self.out_ub = _apply_mem(self.tik_instance, self.dtype, [one_loop_shape], "out_ub")

    # pylint: disable=too-many-locals,too-many-branches,too-many-lines
    def _run_segment(self, mask, repeat_time, gm_offset):
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
        one_loop_shape = mask * repeat_time
        self._apply_all_ub(one_loop_shape)
        # copy gm to ub
        nbust = _get_ceil_int(one_loop_shape, self.eliments_per_block)
        if self.trans:
            self.data_move_in_and_trans(mask, repeat_time, one_loop_shape, gm_offset, nbust)
        else:
            self.data_move_in(gm_offset, nbust)

        # calcu bboxes area
        self.calcu_area(mask, repeat_time, self.bboxes_area_ub, self.bboxes_w_h)

        # calcu gtboxes area
        self.calcu_area(mask, repeat_time, self.gtboxes_area_ub, self.gtboxes_w_h, gt_mode=True)

        # vmin vmax: get inter x0 x1 y0 y1, outer x0 x1 y0 y1
        self.get_inter_outer_area(mask, repeat_time)

        # calcu inter area
        self.calcu_area(mask, repeat_time, self.inter_area_ub, inter_mode=True)

        self.calcu_out_square(mask, repeat_time)
        self.calcu_in_square(mask, repeat_time)

        # calcu atan
        self.calcu_atan(mask, one_loop_shape, self.bboxes_w_h, self.atan_w_h, self.atan_w_h_quarter_pi)
        self.calcu_atan(mask, one_loop_shape, self.gtboxes_w_h, self.gtboxes_atan_w_h,
                        self.gtboxes_atan_w_h_quarter_pi)
        self.tik_instance.vsub(mask, self.atan_w_h, self.gtboxes_atan_w_h, self.atan_w_h,
                               repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, self.v, self.atan_w_h, self.atan_w_h, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmuls(mask, self.v, self.v, 4 / math.pi ** 2, repeat_time, 1, 1, 8, 8)

        if self.mode == "iou":
            self.tik_instance.vadd(mask, self.out_ub, self.bboxes_area_ub,
                                   self.gtboxes_area_ub, repeat_time, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vsub(mask, self.out_ub, self.out_ub,
                                   self.inter_area_ub, repeat_time, 1, 1, 1, 8, 8, 8)
        elif self.mode == "iof":
            self.tik_instance.data_move(self.out_ub, self.gtboxes_area_ub,
                                        0, 1, (nbust - 1) // 4 + 1, 0, 0)

        if self.product is True:
            self.tik_instance.vdiv(mask, self.out_ub, self.inter_area_ub, self.out_ub, repeat_time, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vdiv(mask, self.out_square, self.in_square,
                                   self.out_square, repeat_time, 1, 1, 1, 8, 8, 8)
        else:
            # for mini
            self._rev_div(mask, repeat_time, self.out_ub, self.inter_area_ub, self.out_ub)
            self._rev_div(mask, repeat_time, self.out_square, self.in_square, self.out_square)

        self.tik_instance.vsub(mask, self.alpha_ub, self.v, self.out_ub, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadds(mask, self.alpha_ub, self.alpha_ub, 1 + 1e-16, repeat_time, 1, 1, 8, 8)
        if self.product is True:
            self.tik_instance.vdiv(mask, self.alpha_ub, self.v, self.alpha_ub, repeat_time, 1, 1, 1, 8, 8, 8)
        else:
            self._rev_div(mask, repeat_time, self.alpha_ub, self.v, self.alpha_ub)
        self.tik_instance.vmul(mask, self.alpha_ub, self.v, self.alpha_ub, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(mask, self.out_ub, self.out_ub, self.out_square, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(mask, self.out_ub, self.out_ub, self.alpha_ub, repeat_time, 1, 1, 1, 8, 8, 8)

        self.tik_instance.data_move(self.ciou_gm[gm_offset], self.out_ub, 0, 1, nbust, 0, 0)
        self.tik_instance.data_move(self.atan_sub_gm[gm_offset], self.atan_w_h, 0, 1, nbust, 0, 0)

    def _rev_div(self, mask, repeat_time, x1_ub, x2_ub, y_ub):
        self.tik_instance.vrec(mask, self.div_rec_1, x1_ub, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vmul(mask, self.div_rec_2, self.div_rec_1, x1_ub, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmuls(mask, self.div_rec_2, self.div_rec_2, -1, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vadds(mask, self.div_rec_2, self.div_rec_2, 2, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vmul(mask, self.div_rec_2, self.div_rec_2, self.div_rec_1,
                               repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, self.div_rec_1, self.div_rec_2, x1_ub, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmuls(mask, self.div_rec_1, self.div_rec_1, -1, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vadds(mask, self.div_rec_1, self.div_rec_1, 2, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vmul(mask, self.div_rec_1, self.div_rec_1, self.div_rec_2,
                               repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, y_ub, self.div_rec_1, x2_ub, repeat_time, 1, 1, 1, 8, 8, 8)


def _box_shape_check(input_name, shape):
    """
    box_shape_check

    Parameters
    ----------
    input_name : str
        input name
    shape : tuple
        shape of input name

    Returns
    -------
    None
    """
    shape_len = len(shape)
    if shape_len != 2:
        error_detail = "the shape len should be 2"
        error_manager_vector.raise_err_input_shape_invalid("ciou", input_name, error_detail)
    first_shape_dim = shape[0]
    if first_shape_dim != 4:
        error_detail = "the shape should be [4, n]"
        error_manager_vector.raise_err_input_shape_invalid("ciou", input_name, error_detail)


# pylint: disable=too-many-arguments
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
    bboxes_shape = bboxes.get("shape")
    gtboxes_shape = gtboxes.get("shape")

    # atan_sub_flag only support True currently, lock atan_sub_flag to true
    atan_sub_flag = True

    if is_cross:
        raise RuntimeError("is_cross only support False currently.")
    if not atan_sub_flag:
        raise RuntimeError("atan_sub_flag only support True currently.")
    if not is_cross and bboxes_shape != gtboxes_shape:
        raise RuntimeError("Shape of bboxes don't match shape of gtboxes.")

    _box_shape_check("bboxes", bboxes_shape)
    _box_shape_check("gtboxes", gtboxes_shape)
    bboxes_dtype = bboxes.get("dtype").lower()
    shape_util.compare_tensor_dict_key(bboxes, gtboxes, "dtype")
    check_list = ("float16", "float32")
    para_check.check_dtype(bboxes_dtype, check_list, param_name="bboxes")

    # check whether mode is valid
    check_list = ("iou", "iof")
    if mode not in check_list:
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "mode", "iou,iof", mode)
    ciou_obj = CIoU(bboxes, gtboxes, trans, is_cross, mode, atan_sub_flag)
    res = ciou_obj.run_tik(kernel_name)

    return res
