#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
rotated_box_decode
"""

from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik


class Constant:
    """
    Constant Num
    """

    def __init__(self):
        pass

    # reserved ub size
    RESERVED_UB_SIZE = 8 * 1024
    # double buffer
    THREAD_NUM = 2
    # bytes of one block
    BLOCK_SIZE = 32
    # the params number of each bounding box
    BBOX_NUM = 5
    # define the PI
    PI = 3.14159265358979
    # define the expansion order of Tan series
    TAN_EXPANSION_ORDER = 5
    # define the number of times using the tan2x formula
    TAN_2X_TIMES = 6
    # define the factors for tan
    FACTORS = [1 / 3, 2 / 15, 17 / 315, 62 / 2835, 1382 / 155925]
    # const one
    CONST_POS_ONE = 1.0
    # pi / 4
    CONST_PI_BY_FOUR = 0.78539816339744830961566084581988
    # pi / 8
    CONST_PI_BY_EIGHT = 0.39269908169872415480783042290994
    # taylor itertor
    CONST_ITERTOR = 6
    # taylor itertor 2
    CONST_ITERTOR2 = 4
    # tan(pi / 8)
    TAN_PI_BY_EIGHT = 0.4142135623730950
    # taylor factor
    TAYLOR = (1.0, -1.0 / 3, 1.0 / 5, -1.0 / 7, 1.0 / 9, -1.0 / 11, 1.0 / 13)


# 'pylint: disable=too-many-lines,invalid-name,too-many-arguments,consider-using-in
# 'pylint: disable=too-many-branches,too-many-instance-attributes,too-many-locals
# 'pylint: disable=too-many-statements,no-self-use,too-few-public-methods
# 'pylint: disable=unused-argument
class RotatedBoxDecode():
    """
    Function: use to finish RotatedBoxDecode main functions
    """

    def __init__(self, anchor_box, deltas, y, weight, kernel_name):
        """
        init RotatedBoxDecode parameters
        ---------------------------------------------------------------
        :param anchor_box: dict of shape and dtype for input anchor box
        :param deltas: dict of shape and dtype for input delta box
        :param y: dict of shape and dtype for output decode box
        :param weight: the weight for decode bounding box
        :param kernel_name: the kernel's name
        :return: None
        """
        self.init_tik_inst()
        self.anchor_shape = anchor_box.get("shape")
        self.anchor_dtype = anchor_box.get("dtype").lower()
        self.delta_shape = deltas.get("shape")
        self.delta_dtype = deltas.get("dtype").lower()
        self.decode_shape = y.get("shape")
        self.decode_dtype = y.get("dtype").lower()
        self.weight = weight

        self.kernel_name = kernel_name
        self.batch = self.decode_shape[0]
        self.bbox_num = self.decode_shape[1]
        self.bbox_len = self.decode_shape[2]
        self.dtype = "float32"
        self.mask = 64

        self.loop_num = None
        self.loop_left = None
        self.repeat_time = None
        self.repeat_left = None
        self.act_core_num = None
        self.core_ele = None
        self.one_core_ele = None
        self.last_core_ele = None
        self.core_overlap = None
        self.anchor_ub = None
        self.anchor_ub_f16 = None
        self.decode_ub = None
        self.decode_ub_f16 = None
        self.delta_ub = None
        self.delta_ub_f16 = None
        self.anchor_pos_ub = None
        self.delta_pos_ub = None

        self.tiling_param_caculate()
        self.init_gm_tensor()

    def init_tik_inst(self):
        """init_tik_inst
        """
        self.tik_inst = tik.Tik()
        self.support_div = tbe_platform.api_check_support("tik.vdiv", "float32")

    def CalCoreNum(self, total_ele, core_num):
        """
        :parameter total_ele: int
        :parameter core_num: int
        :returns: None
        """
        if self.delta_dtype == "float16":
            total_ele = total_ele // 2
        self.one_core_ele = (total_ele + core_num - 1) // core_num
        self.act_core_num = total_ele // self.one_core_ele
        if total_ele % self.one_core_ele != 0:
            self.act_core_num = self.act_core_num + 1
        self.last_core_ele = total_ele - (self.act_core_num - 1) * self.one_core_ele

        if self.delta_dtype == "float16":
            self.one_core_ele = self.one_core_ele * 2
            self.last_core_ele = self.last_core_ele * 2

    def tiling_param_caculate(self):
        """tiling_param_caculate
        """
        self.dtype_size = tbe_platform.get_bit_len(self.dtype) // 8
        self.dtype_delta_size = tbe_platform.get_bit_len(self.delta_dtype) // 8
        self.num_each_block = Constant.BLOCK_SIZE // self.dtype_size
        self.num_each_block_f16 = Constant.BLOCK_SIZE // self.dtype_delta_size
        self.align_unit = Constant.BBOX_NUM * self.num_each_block
        self.align_unit_f16 = Constant.BBOX_NUM * self.num_each_block_f16
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_ele = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) -
                       Constant.RESERVED_UB_SIZE) // self.dtype_size // self.align_unit

        if self.delta_dtype == "float16":
            self.ub_split = 16
            if self.bbox_num * self.bbox_len % self.align_unit_f16 != 0:
                total_unit = self.bbox_num * self.bbox_len // self.align_unit_f16 + 1
            else:
                total_unit = self.bbox_num * self.bbox_len // self.align_unit_f16
            if self.bbox_len > 16:
                self.overlap = total_unit * self.align_unit_f16 // self.bbox_num - self.bbox_len
            else:
                self.overlap = 0
            total_unit = total_unit * 2
            self.max_ele = self.ub_ele * 2 // self.ub_split
        else:
            self.ub_split = 10
            if self.bbox_num * self.bbox_len % self.align_unit != 0:
                total_unit = self.bbox_num * self.bbox_len // self.align_unit + 1
            else:
                total_unit = self.bbox_num * self.bbox_len // self.align_unit
            if self.bbox_len > 8:
                self.overlap = total_unit * self.align_unit // self.bbox_num - self.bbox_len
            else:
                self.overlap = 0
            self.max_ele = self.ub_ele // self.ub_split

        self.CalCoreNum(total_unit, self.core_num)
        self.tensor_f16_len = (self.max_ele * self.align_unit // self.align_unit_f16) * self.align_unit_f16
        self.one_core_loop_num = self.one_core_ele // self.max_ele
        self.one_core_loop_left = self.one_core_ele % self.max_ele
        self.last_core_loop_num = self.last_core_ele // self.max_ele
        self.last_core_loop_left = self.last_core_ele % self.max_ele

    def init_gm_tensor(self):
        """init_gm_tensor
        """
        self.anchor_gm = self.tik_inst.Tensor(self.anchor_dtype,
                                              (self.batch, self.bbox_num, self.bbox_len),
                                              name="anchor_gm",
                                              scope=tik.scope_gm)
        self.delta_gm = self.tik_inst.Tensor(self.delta_dtype,
                                             (self.batch, self.bbox_num, self.bbox_len),
                                             name="delta_gm",
                                             scope=tik.scope_gm)
        self.decode_gm = self.tik_inst.Tensor(self.decode_dtype,
                                              (self.batch, self.bbox_num, self.bbox_len),
                                              name="decode_gm",
                                              scope=tik.scope_gm)

    def init_ub_tensor(self):
        """init_ub_tensor
        """
        self.anchor_ub = self.tik_inst.Tensor(self.dtype,
                                              (Constant.BBOX_NUM,
                                               self.max_ele * self.align_unit // Constant.BBOX_NUM),
                                              name="anchor_ub", scope=tik.scope_ubuf)

        self.delta_ub = self.tik_inst.Tensor(self.dtype,
                                             (Constant.BBOX_NUM,
                                              self.max_ele * self.align_unit // Constant.BBOX_NUM),
                                             name="delta_ub", scope=tik.scope_ubuf)

        self.decode_ub = self.tik_inst.Tensor(self.dtype,
                                              (Constant.BBOX_NUM,
                                               self.max_ele * self.align_unit // Constant.BBOX_NUM),
                                              name="decode_ub", scope=tik.scope_ubuf)

        self.anchor_pos_ub = self.tik_inst.Tensor(self.dtype,
                                                  (Constant.BBOX_NUM,
                                                   self.max_ele * self.align_unit // Constant.BBOX_NUM),
                                                  name="anchor_pos_ub", scope=tik.scope_ubuf)

        self.delta_pos_ub = self.tik_inst.Tensor(self.dtype,
                                                 (Constant.BBOX_NUM,
                                                  self.max_ele * self.align_unit // Constant.BBOX_NUM),
                                                 name="delta_pos_ub", scope=tik.scope_ubuf)

        if self.delta_dtype == "float16":
            self.anchor_ub_f16 = self.tik_inst.Tensor(self.anchor_dtype,
                                                      (Constant.BBOX_NUM,
                                                       self.tensor_f16_len // Constant.BBOX_NUM),
                                                      name="anchor_ub_f16", scope=tik.scope_ubuf)

            self.delta_ub_f16 = self.tik_inst.Tensor(self.delta_dtype,
                                                     (Constant.BBOX_NUM,
                                                      self.tensor_f16_len // Constant.BBOX_NUM),
                                                     name="delta_ub_f16", scope=tik.scope_ubuf)

            self.decode_ub_f16 = self.tik_inst.Tensor(self.decode_dtype,
                                                      (Constant.BBOX_NUM,
                                                       self.tensor_f16_len // Constant.BBOX_NUM),
                                                      name="decode_ub_f16", scope=tik.scope_ubuf)

    def init_ub_scalar(self):
        """init_ub_tensor
        """
        self.core_ele = self.tik_inst.Scalar("int32", name="core_ele")
        self.loop_num = self.tik_inst.Scalar("int32", name="loop_num")
        self.loop_left = self.tik_inst.Scalar("int32", name="loop_left")
        self.repeat_time = self.tik_inst.Scalar("int32", name="repeat_time")
        self.repeat_left = self.tik_inst.Scalar("int32", name="repeat_left")
        self.core_overlap = self.tik_inst.Scalar("int32", name="core_overlap")

    def move_in_gm_data(self, repeat_time, repeat_left, repeat_offset, batch_idx, offset, ub_ele, overlap):
        """
        :param repeat_time: int, the repeat time for tik instruction
        :param repeat_left: int, the repeat left for tik instrucion
        :param repeat_offset: int, the repeat offset for tik instruction
        :param batch_idx: the batch index
        :param offset: the offset for gm
        :param ub_ele: the data lenght
        :param overlap: the overlap for 32B
        :return: None
        """
        if self.delta_dtype == "float16":
            delta_ub = self.delta_ub_f16
            anchor_ub = self.anchor_ub_f16
            ub_ele = ub_ele // 2
            num_each_block = self.num_each_block_f16
        else:
            delta_ub = self.delta_ub
            anchor_ub = self.anchor_ub
            num_each_block = self.num_each_block

        with self.tik_inst.for_range(0, Constant.BBOX_NUM) as bbox_idx:
            with self.tik_inst.if_scope(overlap > 0):
                with self.tik_inst.if_scope(ub_ele - 1 > 0):
                    self.tik_inst.data_move(anchor_ub[bbox_idx, :],
                                            self.anchor_gm[batch_idx, bbox_idx, offset:],
                                            0, 1, ub_ele - 1, 0, 0)
                    self.tik_inst.data_move(delta_ub[bbox_idx, :],
                                            self.delta_gm[batch_idx, bbox_idx, offset:],
                                            0, 1, ub_ele - 1, 0, 0)

                align_offset = offset + (ub_ele - 1) * num_each_block - overlap
                ub_offset = (ub_ele - 1) * num_each_block

                self.tik_inst.data_move(anchor_ub[bbox_idx, ub_offset:],
                                        self.anchor_gm[batch_idx, bbox_idx, align_offset:],
                                        0, 1, 1, 0, 0)
                self.tik_inst.data_move(delta_ub[bbox_idx, ub_offset:],
                                        self.delta_gm[batch_idx, bbox_idx, align_offset:],
                                        0, 1, 1, 0, 0)

            with self.tik_inst.else_scope():
                self.tik_inst.data_move(anchor_ub[bbox_idx, :],
                                        self.anchor_gm[batch_idx, bbox_idx, offset:],
                                        0, 1, ub_ele, 0, 0)
                self.tik_inst.data_move(delta_ub[bbox_idx, :],
                                        self.delta_gm[batch_idx, bbox_idx, offset:],
                                        0, 1, ub_ele, 0, 0)
        if self.delta_dtype == "float16":
            with self.tik_inst.for_range(0, Constant.BBOX_NUM) as bbox_idx:
                with self.tik_inst.if_scope(repeat_time > 0):
                    self.tik_inst.vconv(self.mask, 'none', self.anchor_ub[bbox_idx, :], anchor_ub[bbox_idx, :],
                                        repeat_time, 1, 1, 8, 4)

                    self.tik_inst.vconv(self.mask, 'none', self.delta_ub[bbox_idx, :], delta_ub[bbox_idx, :],
                                        repeat_time, 1, 1, 8, 4)

                with self.tik_inst.if_scope(repeat_left > 0):
                    self.tik_inst.vconv(repeat_left, 'none', self.anchor_ub[bbox_idx, repeat_offset:],
                                        anchor_ub[bbox_idx, repeat_offset:], 1, 1, 1, 8, 4)

                    self.tik_inst.vconv(repeat_left, 'none', self.delta_ub[bbox_idx, repeat_offset:],
                                        delta_ub[bbox_idx, repeat_offset:], 1, 1, 1, 8, 4)

    def move_out_ub_data(self, repeat_time, repeat_left, repeat_offset, batch_idx, offset, ub_ele, overlap):
        """
        :param repeat_time: int, the repeat time for tik instruction
        :param repeat_left: int, the repeat left for tik instrucion
        :param repeat_offset:  int, the repeat offset for tik instruction
        :param batch_idx: the batch index
        :param offset: the offset for gm
        :param ub_ele: the data lenght
        :param overlap: the overlap for 32B
        :return: None
        """
        if self.decode_dtype == "float16":
            ub_ele = ub_ele // 2
            decode_ub = self.decode_ub_f16
            num_each_block = self.num_each_block_f16
            with self.tik_inst.for_range(0, Constant.BBOX_NUM) as bbox_idx:
                with self.tik_inst.if_scope(repeat_time > 0):
                    self.tik_inst.vconv(self.mask, "none", self.decode_ub_f16[bbox_idx, :],
                                        self.decode_ub[bbox_idx, :], repeat_time, 1, 1, 4, 8)
                with self.tik_inst.if_scope(repeat_left > 0):
                    self.tik_inst.vconv(repeat_left, "none", self.decode_ub_f16[bbox_idx, repeat_offset:],
                                        self.decode_ub[bbox_idx, repeat_offset:], 1, 1, 1, 4, 8)
        else:
            decode_ub = self.decode_ub
            num_each_block = self.num_each_block

        with self.tik_inst.for_range(0, Constant.BBOX_NUM) as bbox_idx:
            with self.tik_inst.if_scope(overlap > 0):
                with self.tik_inst.if_scope(ub_ele - 1 > 0):
                    self.tik_inst.data_move(self.decode_gm[batch_idx, bbox_idx, offset:],
                                            decode_ub[bbox_idx, :],
                                            0, 1, ub_ele - 1, 0, 0)
                align_offset = offset + (ub_ele - 1) * num_each_block - overlap
                ub_offset = (ub_ele - 1) * num_each_block

                self.tik_inst.data_move(self.decode_gm[batch_idx, bbox_idx, align_offset:],
                                        decode_ub[bbox_idx, ub_offset:],
                                        0, 1, 1, 0, 0)

            with self.tik_inst.else_scope():
                self.tik_inst.data_move(self.decode_gm[batch_idx, bbox_idx, offset:],
                                        decode_ub[bbox_idx, :],
                                        0, 1, ub_ele, 0, 0)

    def corner_to_center(self, box_corner, box_pos, repeat_time, repeat_left, offset):
        """
        :param box_corner: tensor, the input corner format box.
        :param box_pos: tensor, the output center format box.
        :param repeat_time: int, the repeat time for tik instruction
        :param repeat_left: int, the repeat left for tik instrucion
        :param offset: int, the repeat offset for tik instruction
        :return: None
        """
        corner_lx = box_corner[0, :]
        corner_ly = box_corner[1, :]
        corner_rx = box_corner[2, :]
        corner_ry = box_corner[3, :]
        corner_angle = box_corner[4, :]

        center_x = box_pos[0, :]
        center_y = box_pos[1, :]
        box_w = box_pos[2, :]
        box_h = box_pos[3, :]
        center_angle = box_pos[4, :]

        with self.tik_inst.if_scope(repeat_time > 0):
            # calc box width
            self.tik_inst.vsub(self.mask, box_w, corner_rx,
                               corner_lx, repeat_time, 1, 1, 1, 8, 8, 8)

            # clamp box width
            self.tik_inst.vector_dup(self.mask, corner_rx, 1.0, 1, 1, 8)
            self.tik_inst.vmax(self.mask, box_w, box_w, corner_rx, repeat_time, 1, 1, 1, 8, 8, 0)

            # calc box height
            self.tik_inst.vsub(self.mask, box_h, corner_ry,
                               corner_ly, repeat_time, 1, 1, 1, 8, 8, 8)

            # clamp box height
            self.tik_inst.vmax(self.mask, box_h, box_h, corner_rx, repeat_time, 1, 1, 1, 8, 8, 0)

            # calc box center x
            self.tik_inst.vmuls(self.mask, center_x, box_w, 0.5, repeat_time, 1, 1, 8, 8)
            self.tik_inst.vadd(self.mask, center_x, corner_lx,
                               center_x, repeat_time, 1, 1, 1, 8, 8, 8)

            # calc box center y
            self.tik_inst.vmuls(self.mask, center_y, box_h, 0.5, repeat_time, 1, 1, 8, 8)
            self.tik_inst.vadd(self.mask, center_y, corner_ly,
                               center_y, repeat_time, 1, 1, 1, 8, 8, 8)

            # calc box angle
            self.tik_inst.data_move(center_angle, corner_angle, 0, 1,
                                    self.mask * repeat_time // self.num_each_block, 1, 1)

        with self.tik_inst.if_scope(repeat_left > 0):
            # calc box width
            self.tik_inst.vsub(repeat_left, box_w[0, offset:], corner_rx[0, offset:],
                               corner_lx[0, offset:], 1, 1, 1, 1, 8, 8, 8)

            # clamp box width
            self.tik_inst.vector_dup(repeat_left, corner_rx[0, offset:], 1.0, 1, 1, 8)
            self.tik_inst.vmax(repeat_left, box_w[0, offset:], box_w[0, offset:],
                               corner_rx[0, offset:], 1, 1, 1, 1, 8, 8, 0)

            # calc box height
            self.tik_inst.vsub(repeat_left, box_h[0, offset:], corner_ry[0, offset:],
                               corner_ly[0, offset:], 1, 1, 1, 1, 8, 8, 8)

            # clamp box height
            self.tik_inst.vmax(repeat_left, box_h[0, offset:], box_h[0, offset:],
                               corner_rx[0, offset:], 1, 1, 1, 1, 8, 8, 0)

            # calc box center x
            self.tik_inst.vmuls(repeat_left, center_x[0, offset:], box_w[0, offset:], 0.5, 1, 1, 1, 8, 8)
            self.tik_inst.vadd(repeat_left, center_x[0, offset:], corner_lx[0, offset:],
                               center_x[0, offset:], 1, 1, 1, 1, 8, 8, 8)

            # calc box center y
            self.tik_inst.vmuls(repeat_left, center_y[0, offset:], box_h[0, offset:], 0.5, 1, 1, 1, 8, 8)
            self.tik_inst.vadd(repeat_left, center_y[0, offset:], corner_ly[0, offset:],
                               center_y[0, offset:], 1, 1, 1, 1, 8, 8, 8)

            # calc box angle
            self.tik_inst.data_move(center_angle[0, offset:], corner_angle[0, offset:],
                                    0, 1, repeat_left // self.num_each_block, 1, 1)

    def delta_scalar(self, delta, wx, wy, ww, wh, wt, repeat_time, repeat_left, offset):
        """
        :param delta: dict of shape and dtype for input delta box
        :param wx: weight of x
        :param wy: weight of y
        :param ww: weight of w
        :param wh: weight of h
        :param wt: weight of t
        :param repeat_time: int, the repeat time for tik instruction
        :param repeat_left: int, the repeat left for tik instrucion
        :param offset: int, the repeat offset for tik instruction
        :return: None
        """
        delta_x = delta[0, :]
        delta_y = delta[1, :]
        delta_w = delta[2, :]
        delta_h = delta[3, :]
        delta_t = delta[4, :]

        with self.tik_inst.if_scope(repeat_time > 0):
            self.tik_inst.vmuls(self.mask, delta_x, delta_x, 1 / wx, repeat_time, 1, 1, 8, 8)
            self.tik_inst.vmuls(self.mask, delta_y, delta_y, 1 / wy, repeat_time, 1, 1, 8, 8)
            self.tik_inst.vmuls(self.mask, delta_w, delta_w, 1 / ww, repeat_time, 1, 1, 8, 8)
            self.tik_inst.vmuls(self.mask, delta_h, delta_h, 1 / wh, repeat_time, 1, 1, 8, 8)
            self.tik_inst.vmuls(self.mask, delta_t, delta_t, 1 / wt, repeat_time, 1, 1, 8, 8)

        with self.tik_inst.if_scope(repeat_left > 0):
            self.tik_inst.vmuls(repeat_left, delta_x[0, offset:], delta_x[0, offset:], 1 / wx, 1, 1, 1, 8, 8)
            self.tik_inst.vmuls(repeat_left, delta_y[0, offset:], delta_y[0, offset:], 1 / wy, 1, 1, 1, 8, 8)
            self.tik_inst.vmuls(repeat_left, delta_w[0, offset:], delta_w[0, offset:], 1 / ww, 1, 1, 1, 8, 8)
            self.tik_inst.vmuls(repeat_left, delta_h[0, offset:], delta_h[0, offset:], 1 / wh, 1, 1, 1, 8, 8)
            self.tik_inst.vmuls(repeat_left, delta_t[0, offset:], delta_t[0, offset:], 1 / wt, 1, 1, 1, 8, 8)

    def get_target_center(self, src_0_ub, src_1_ub, src_2_ub, center_info_ub,
                          repeat_time, repeat_left, offset):
        """
        :param src_0_ub: input tensor 0
        :param src_1_ub: input tensor 1
        :param src_2_ub: input tensor 2
        :param center_info_ub: result tensor
        :param repeat_time: int, the repeat time for tik instruction
        :param repeat_left: int, the repeat left for tik instrucion
        :param offset: int, the repeat offset for tik instruction
        :return: None
        """
        with self.tik_inst.new_stmt_scope():
            with self.tik_inst.if_scope(repeat_time > 0):
                self.tik_inst.vmul(self.mask, center_info_ub, src_0_ub, src_2_ub,
                                   repeat_time, 1, 1, 1, 8, 8, 8)
                self.tik_inst.vadd(self.mask, center_info_ub, src_1_ub,
                                   center_info_ub, repeat_time, 1, 1, 1, 8, 8, 8)

            with self.tik_inst.if_scope(repeat_left > 0):
                self.tik_inst.vmul(repeat_left, center_info_ub[0, offset:], src_0_ub[0, offset:],
                                   src_2_ub[0, offset:], 1, 1, 1, 1, 8, 8, 8)
                self.tik_inst.vadd(repeat_left, center_info_ub[0, offset:], src_1_ub[0, offset:],
                                   center_info_ub[0, offset:], 1, 1, 1, 1, 8, 8, 8)

    def get_target_wh(self, src_0_ub, src_1_ub, decode_ub, repeat_time, repeat_left, offset):
        """
        :param src_0_ub: input tensor 0
        :param src_1_ub: input tensor 1
        :param decode_ub: result decode tensor
        :param repeat_time: int, the repeat time for tik instruction
        :param repeat_left: int, the repeat left for tik instrucion
        :param offset: int, the repeat offset for tik instruction
        :return: None
        """
        with self.tik_inst.if_scope(repeat_time > 0):
            self.tik_inst.vec_exp(self.mask, decode_ub, src_0_ub, repeat_time, 8, 8)
            self.tik_inst.vmul(self.mask, decode_ub, decode_ub, src_1_ub,
                               repeat_time, 1, 1, 1, 8, 8, 8)

        with self.tik_inst.if_scope(repeat_left > 0):
            self.tik_inst.vec_exp(repeat_left, decode_ub[0, offset:], src_0_ub[0, offset:], 1, 8, 8)
            self.tik_inst.vmul(repeat_left, decode_ub[0, offset:], decode_ub[0, offset:],
                               src_1_ub[0, offset:], 1, 1, 1, 1, 8, 8, 8)

    def compute_tan(self, mask, repeat_time, ub_a, ub_b, ub_c):
        """
        :param mask: int, the mask for tik instruction
        :param repeat_time: int, the repeat time for tik instruction
        :param offset: int, the repeat offset for tik instruction
        :param ub_a: temp tensor a
        :param ub_b: temp tensor b
        :param ub_c: temp tensor c
        :return: None
        """
        input_x = ub_a
        # round input angle
        round_pi = ub_b
        self.tik_inst.vmuls(mask, round_pi, input_x, 1.0 / Constant.PI, repeat_time, 1, 1, 8, 8)
        temp_round = ub_c.reinterpret_cast_to("int32")
        self.tik_inst.vconv(mask, "round", temp_round, round_pi, repeat_time, 1, 1, 8, 8)
        self.tik_inst.vconv(mask, "", round_pi, temp_round, repeat_time, 1, 1, 8, 8)
        round_x = ub_c.reinterpret_cast_to(self.dtype)
        self.tik_inst.vmuls(mask, round_x, round_pi, Constant.PI, repeat_time, 1, 1, 8, 8)
        self.tik_inst.vsub(mask, round_x, input_x, round_x, repeat_time, 1, 1, 1, 8, 8, 8)

        x_divide = ub_a
        self.tik_inst.vmuls(mask, x_divide, round_x, 1.0 / (2.0 ** Constant.TAN_2X_TIMES), repeat_time, 1, 1, 8, 8)
        x_power = ub_b
        self.tik_inst.vmul(mask, x_power, x_divide, x_divide, repeat_time, 1, 1, 1, 8, 8, 8)

        iter_value = ub_c
        self.tik_inst.data_move(iter_value, x_divide, 0, 1, mask * repeat_time // self.num_each_block, 0, 0)

        res = ub_a
        for i, _ in enumerate(range(Constant.TAN_EXPANSION_ORDER)):
            self.tik_inst.vmul(mask, iter_value, x_power, iter_value, repeat_time, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vmuls(mask, iter_value, iter_value, Constant.FACTORS[i], repeat_time, 1, 1, 8, 8)
            self.tik_inst.vadd(mask, res, res, iter_value, repeat_time, 1, 1, 1, 8, 8, 8)
        times = Constant.TAN_2X_TIMES
        res_denominator = ub_c
        tanx_square = ub_b
        while times != 0:
            self.tik_inst.vmuls(mask, res_denominator, res, 2.0, repeat_time, 1, 1, 8, 8)
            self.tik_inst.vmul(mask, tanx_square, res, res, repeat_time, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vmuls(mask, tanx_square, tanx_square, -1.0, repeat_time, 1, 1, 8, 8)
            self.tik_inst.vadds(mask, tanx_square, tanx_square, 1.0, repeat_time, 1, 1, 8, 8)
            self.tik_inst.vdiv(mask, res, res_denominator, tanx_square, repeat_time, 1, 1, 1, 8, 8, 8)
            times = times - 1

    def do_taylor(self, mask, repeat_time, input_data, res_ub, ub_b, ub_c):
        """
        :param mask: int, the mask for tik instruction
        :param repeat_time: int, the repeat time for tik instruction
        :param input_data: tensor, input data for taylor
        :param res_ub: tensor, result tensor
        :param ub_b: tensor, temp tensor b
        :param ub_c: tensor, temp tensor c
        :return: None
        """
        tensor_offset = ub_b
        self.tik_inst.vec_dup(mask, tensor_offset, Constant.TAN_PI_BY_EIGHT, repeat_time, 8)
        denominator_data = ub_c
        self.tik_inst.vmuls(mask, denominator_data, input_data, Constant.TAN_PI_BY_EIGHT, repeat_time, 1, 1, 8, 8)
        self.tik_inst.vadds(mask, denominator_data, denominator_data, Constant.CONST_POS_ONE, repeat_time, 1, 1, 8, 8)
        molecule = ub_b
        self.tik_inst.vsub(mask, molecule, input_data, tensor_offset, repeat_time, 1, 1, 1, 8, 8, 8)
        data = ub_b
        self.tik_inst.vdiv(mask, data, molecule, denominator_data, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vabs(mask, data, data, repeat_time, 1, 1, 8, 8)

        square_data = ub_c
        self.tik_inst.vmul(mask, square_data, data, data, repeat_time, 1, 1, 1, 8, 8, 8)
        res = res_ub
        self.tik_inst.vec_dup(mask, res, Constant.TAYLOR[Constant.CONST_ITERTOR], repeat_time, 8)
        for i in reversed(range(Constant.CONST_ITERTOR)):
            self.tik_inst.vmul(mask, res, res, square_data, repeat_time, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vadds(mask, res, res, Constant.TAYLOR[i], repeat_time, 1, 1, 8, 8)
        self.tik_inst.vmul(mask, res, res, data, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadds(mask, res, res, Constant.CONST_PI_BY_EIGHT, repeat_time, 1, 1, 8, 8)

        self.tik_inst.vmul(mask, square_data, input_data, input_data, repeat_time, 1, 1, 1, 8, 8, 8)
        res2 = ub_b
        self.tik_inst.vec_dup(mask, res2, Constant.TAYLOR[Constant.CONST_ITERTOR2], repeat_time, 8)
        for i in reversed(range(Constant.CONST_ITERTOR2)):
            self.tik_inst.vmul(mask, res2, res2, square_data, repeat_time, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vadds(mask, res2, res2, Constant.TAYLOR[i], repeat_time, 1, 1, 8, 8)
        self.tik_inst.vmul(mask, res2, res2, input_data, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vmin(mask, res, res, res2, repeat_time, 1, 1, 1, 8, 8, 8)

    def compute_atan(self, mask, repeat_time, ub_a, ub_b, ub_c, ub_d, ub_e, ub_f):
        """
        :param mask: int, the mask for tik instruction
        :param repeat_time: int, the repeat time for tik instruction
        :param ub_a: temp tensor a
        :param ub_b: temp tensor b
        :param ub_c: temp tensor c
        :param ub_d: temp tensor d
        :param ub_e: temp tensor e
        :param ub_f: temp tensor f
        :return: None
        """
        x = ub_a
        abs_data = ub_e
        self.tik_inst.vabs(mask, abs_data, x, repeat_time, 1, 1, 8, 8)

        tensor_one = ub_b
        self.tik_inst.vec_dup(mask, tensor_one, Constant.CONST_POS_ONE, repeat_time, 8)

        abs_data_sub_one = ub_c
        abs_data_add_one = ub_d
        self.tik_inst.vsub(mask, abs_data_sub_one, abs_data, tensor_one, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(mask, abs_data_add_one, abs_data, tensor_one, repeat_time, 1, 1, 1, 8, 8, 8)
        abs_data2 = ub_d
        self.tik_inst.vdiv(mask, abs_data2, abs_data_sub_one, abs_data_add_one, repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vabs(mask, abs_data2, abs_data2, repeat_time, 1, 1, 8, 8)

        res_mt_one = ub_f
        self.do_taylor(mask, repeat_time, abs_data2, res_mt_one, ub_b, ub_c)
        res = ub_d
        self.do_taylor(mask, repeat_time, abs_data, res, ub_b, ub_c)

        self.tik_inst.vadds(mask, res_mt_one, res_mt_one, Constant.CONST_PI_BY_FOUR, repeat_time, 1, 1, 8, 8)
        self.tik_inst.vmin(mask, res, res, res_mt_one, repeat_time, 1, 1, 1, 8, 8, 8)

        new_data = ub_a
        self.tik_inst.vmuls(mask, new_data, x, 2 ** 62, repeat_time, 1, 1, 8, 8)
        abs_data = ub_b
        self.tik_inst.vabs(mask, abs_data, new_data, repeat_time, 1, 1, 8, 8)
        denominator = ub_b
        self.tik_inst.vadds(mask, denominator, abs_data, 2 ** (-62), repeat_time, 1, 1, 8, 8)
        sign_mask = ub_b
        self.tik_inst.vdiv(mask, sign_mask, new_data, denominator, repeat_time, 1, 1, 1, 8, 8, 8)
        res_out = ub_a
        self.tik_inst.vmul(mask, res_out, res, sign_mask, repeat_time, 1, 1, 1, 8, 8, 8)

    def get_target_angle(self, delta_angle, anchor_angle, decode_angle, repeat_time, repeat_left, offset):
        """
        :param delta_angle: delta angle tensor
        :param anchor_angle: anchor angle tensor
        :param decode_angle: decode anelg tensor
        :param repeat_time: int, the repeat time for tik instruction
        :param repeat_left: int, the repeat left for tik instrucion
        :param offset: int, the repeat offset for tik instruction
        :return: None
        """
        with self.tik_inst.if_scope(repeat_time > 0):
            self.tik_inst.vmuls(self.mask, anchor_angle, anchor_angle, Constant.PI / 180.0, repeat_time, 1, 1, 8, 8)
            self.compute_tan(self.mask, repeat_time, anchor_angle, self.delta_pos_ub[0, :], self.delta_pos_ub[1, :])
            self.tik_inst.vadd(self.mask, decode_angle, anchor_angle,
                               delta_angle, repeat_time, 1, 1, 1, 8, 8, 8)
            self.compute_atan(self.mask, repeat_time, decode_angle, self.delta_pos_ub[0, :], self.delta_pos_ub[1, :],
                              self.delta_pos_ub[2, :], self.delta_pos_ub[3, :], self.delta_pos_ub[4, :])
            self.tik_inst.vmuls(self.mask, decode_angle, decode_angle, 180.0 / Constant.PI, repeat_time, 1, 1, 8, 8)

        with self.tik_inst.if_scope(repeat_left > 0):
            self.tik_inst.vmuls(repeat_left, anchor_angle[0, offset:], anchor_angle[0, offset:],
                                Constant.PI / 180.0, 1, 1, 1, 8, 8)
            self.compute_tan(repeat_left, 1, anchor_angle[0, offset:], self.delta_pos_ub[0, offset:],
                             self.delta_pos_ub[1, offset:])
            self.tik_inst.vadd(repeat_left, decode_angle[0, offset:], anchor_angle[0, offset:],
                               delta_angle[0, offset:], 1, 1, 1, 1, 8, 8, 8)
            self.compute_atan(repeat_left, 1, decode_angle[0, offset:], self.delta_pos_ub[0, offset:],
                              self.delta_pos_ub[1, offset:], self.delta_pos_ub[2, offset:],
                              self.delta_pos_ub[3, offset:], self.delta_pos_ub[4, offset:])
            self.tik_inst.vmuls(repeat_left, decode_angle[0, offset:], decode_angle[0, offset:],
                                180.0 / Constant.PI, 1, 1, 1, 8, 8)

    def center_to_corner(self, box_center, box_pos, repeat_time, repeat_left, offset):
        """
        :param box_center: tensor, the input center format box
        :param box_pos: tensor, the ouput corner format box
        :param repeat_time: int, the repeat time for tik instruction
        :param repeat_left: int, the repeat left for tik instrucion
        :param offset: int, the repeat offset for tik instruction
        :return: None
        """
        center_x = box_center[0, :]
        center_y = box_center[1, :]
        box_w = box_center[2, :]
        box_h = box_center[3, :]

        corner_lx = box_pos[0, :]
        corner_ly = box_pos[1, :]
        corner_rx = box_pos[2, :]
        corner_ry = box_pos[3, :]

        with self.tik_inst.if_scope(repeat_time > 0):
            self.tik_inst.vmuls(self.mask, corner_rx, box_w, 0.5, repeat_time, 1, 1, 8, 8)
            self.tik_inst.vsub(self.mask, corner_lx, center_x,
                               corner_rx, repeat_time, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vadd(self.mask, corner_rx, center_x,
                               corner_rx, repeat_time, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vmuls(self.mask, corner_ry, box_h, 0.5, repeat_time, 1, 1, 8, 8)
            self.tik_inst.vsub(self.mask, corner_ly, center_y,
                               corner_ry, repeat_time, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vadd(self.mask, corner_ry, center_y,
                               corner_ry, repeat_time, 1, 1, 1, 8, 8, 8)
            self.tik_inst.data_move(center_x, corner_lx, 0, 1, self.mask * repeat_time // self.num_each_block, 1, 1)
            self.tik_inst.data_move(center_y, corner_ly, 0, 1, self.mask * repeat_time // self.num_each_block, 1, 1)
            self.tik_inst.data_move(box_w, corner_rx, 0, 1, self.mask * repeat_time // self.num_each_block, 1, 1)
            self.tik_inst.data_move(box_h, corner_ry, 0, 1, self.mask * repeat_time // self.num_each_block, 1, 1)

        with self.tik_inst.if_scope(repeat_left > 0):
            self.tik_inst.vmuls(repeat_left, corner_rx[0, offset:], box_w[0, offset:],
                                0.5, 1, 1, 1, 8, 8)
            self.tik_inst.vsub(repeat_left, corner_lx[0, offset:], center_x[0, offset:],
                               corner_rx[0, offset:], 1, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vadd(repeat_left, corner_rx[0, offset:], center_x[0, offset:],
                               corner_rx[0, offset:], 1, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vmuls(repeat_left, corner_ry[0, offset:], box_h[0, offset:],
                                0.5, 1, 1, 1, 8, 8)
            self.tik_inst.vsub(repeat_left, corner_ly[0, offset:], center_y[0, offset:],
                               corner_ry[0, offset:], 1, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vadd(repeat_left, corner_ry[0, offset:], center_y[0, offset:],
                               corner_ry[0, offset:], 1, 1, 1, 1, 8, 8, 8)
            self.tik_inst.data_move(center_x[0, offset:], corner_lx[0, offset:],
                                    0, 1, repeat_left // self.num_each_block, 1, 1)
            self.tik_inst.data_move(center_y[0, offset:], corner_ly[0, offset:],
                                    0, 1, repeat_left // self.num_each_block, 1, 1)
            self.tik_inst.data_move(box_w[0, offset:], corner_rx[0, offset:],
                                    0, 1, repeat_left // self.num_each_block, 1, 1)
            self.tik_inst.data_move(box_h[0, offset:], corner_ry[0, offset:],
                                    0, 1, repeat_left // self.num_each_block, 1, 1)

    def rotated_box_decode_compute(self, repeat_time, repeat_left, offset):
        """
        :param repeat_time: int, the repeat time for tik instruction
        :param repeat_left: int, the repeat left for tik instrucion
        :param offset: int, the repeat offset for tik instruction
        :return: None
        """
        self.corner_to_center(self.anchor_ub, self.anchor_pos_ub, repeat_time, repeat_left, offset)

        self.delta_scalar(self.delta_ub, self.weight[0], self.weight[1], self.weight[2],
                          self.weight[3], self.weight[4], repeat_time, repeat_left, offset)

        self.get_target_center(self.delta_ub[0, :], self.anchor_pos_ub[0, :],
                               self.anchor_pos_ub[2, :], self.decode_ub[0, :],
                               repeat_time, repeat_left, offset)

        self.get_target_center(self.delta_ub[1, :], self.anchor_pos_ub[1, :],
                               self.anchor_pos_ub[3, :], self.decode_ub[1, :],
                               repeat_time, repeat_left, offset)

        self.get_target_wh(self.delta_ub[2, :], self.anchor_pos_ub[2, :],
                           self.decode_ub[2, :], repeat_time, repeat_left, offset)

        self.get_target_wh(self.delta_ub[3, :], self.anchor_pos_ub[3, :],
                           self.decode_ub[3, :], repeat_time, repeat_left, offset)

        self.get_target_angle(self.delta_ub[4, :], self.anchor_pos_ub[4, :],
                              self.decode_ub[4, :], repeat_time, repeat_left, offset)

        self.center_to_corner(self.decode_ub, self.delta_pos_ub, repeat_time, repeat_left, offset)

    def calculation_process(self, core_idx, batch_idx, loop_num, loop_left, overlap):
        """
        :param core_idx: the core index
        :param batch_idx: the batch index
        :param loop_num: the loop num
        :param loop_left: the loop left
        :param overlap: the overlap
        :return: None
        """
        base_offset = core_idx * self.one_core_ele * self.align_unit
        with self.tik_inst.for_range(0, loop_num, thread_num=Constant.THREAD_NUM) as cyc_idx:
            self.repeat_time.set_as(self.max_ele * self.num_each_block // self.mask)
            self.repeat_left.set_as(self.max_ele * self.num_each_block % self.mask)
            offset = base_offset + cyc_idx * self.max_ele * self.align_unit
            self.move_in_gm_data(self.repeat_time, self.repeat_left, self.repeat_time * self.mask,
                                 batch_idx, offset // Constant.BBOX_NUM, self.max_ele, 0)
            self.rotated_box_decode_compute(self.repeat_time, self.repeat_left, self.repeat_time * self.mask)
            self.move_out_ub_data(self.repeat_time, self.repeat_left, self.repeat_time * self.mask,
                                  batch_idx, offset // Constant.BBOX_NUM, self.max_ele, 0)

        with self.tik_inst.if_scope(loop_left > 0):
            self.repeat_time.set_as(loop_left * self.num_each_block // self.mask)
            self.repeat_left.set_as(loop_left * self.num_each_block % self.mask)
            offset = base_offset + loop_num * self.max_ele * self.align_unit
            self.move_in_gm_data(self.repeat_time, self.repeat_left, self.repeat_time * self.mask,
                                 batch_idx, offset // Constant.BBOX_NUM, loop_left, overlap)
            self.rotated_box_decode_compute(self.repeat_time, self.repeat_left, self.repeat_time * self.mask)
            self.move_out_ub_data(self.repeat_time, self.repeat_left, self.repeat_time * self.mask,
                                  batch_idx, offset // Constant.BBOX_NUM, loop_left, overlap)

    def decode_compute_tiling(self):
        """decode_compute_tiling
        """
        with self.tik_inst.for_range(0, self.core_num, block_num=self.core_num) as core_idx:
            with self.tik_inst.if_scope(core_idx < self.act_core_num):
                self.init_ub_scalar()
                self.init_ub_tensor()
                with self.tik_inst.if_scope(core_idx == self.act_core_num - 1):
                    self.core_overlap.set_as(self.overlap)
                with self.tik_inst.else_scope():
                    self.core_overlap.set_as(0)
                with self.tik_inst.if_scope(core_idx < self.act_core_num - 1):
                    self.core_ele.set_as(self.one_core_ele)
                    self.loop_num.set_as(self.one_core_loop_num)
                    self.loop_left.set_as(self.one_core_loop_left)
                with self.tik_inst.if_scope(core_idx == self.act_core_num - 1):
                    self.core_ele.set_as(self.last_core_ele)
                    self.loop_num.set_as(self.last_core_loop_num)
                    self.loop_left.set_as(self.last_core_loop_left)
                with self.tik_inst.for_range(0, self.batch) as batch_idx:
                    self.calculation_process(core_idx, batch_idx, self.loop_num,
                                             self.loop_left, self.core_overlap)

    def tik_inst_function(self):
        """tik_inst_function
        """
        self.decode_compute_tiling()
        self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                               inputs=[self.anchor_gm, self.delta_gm],
                               outputs=[self.decode_gm])


def _check_param(anchor_box, deltas, y, weight):
    """
    check parameters, if one is invalid, then raise error
    -----------------------
    :param anchor_box: dict of shape and dtype of input anchor bbox
    :param deltas: dict of shape and dtype of input delta bbox
    :param y: dict of shape and dtype of output decode bbox
    :param weight: the weight for decode bounding box
    :return: None
    """
    anchor_shape = anchor_box.get("shape")
    anchor_dtype = anchor_box.get("dtype").lower()
    delta_shape = deltas.get("shape")
    delta_dtype = deltas.get("dtype").lower()
    decode_shape = y.get("shape")
    decode_dtype = y.get("dtype").lower()

    if anchor_dtype != delta_dtype or anchor_dtype != decode_dtype or delta_dtype != decode_dtype:
        raise RuntimeError("anchor dtype, delta dtype and decode dtype must be same.")

    if anchor_shape != delta_shape or anchor_shape != decode_shape or delta_shape != decode_shape:
        raise RuntimeError("anchor shape, delta shape and decode shape must be same.")

    if len(anchor_shape) != 3:
        raise RuntimeError("data dim must be 3.")

    if len(weight) != 5:
        raise RuntimeError("weight dim must be 5.")

    if anchor_shape[0] <= 0:
        raise RuntimeError("the data shape[0] must be greater than 0.")

    if anchor_shape[1] != 5:
        raise RuntimeError("the data shape[1] must be 5.")

    if anchor_shape[2] <= 0:
        raise RuntimeError("the data shape[2] must be greater than 0.")

def rotated_box_decode(anchor_box, deltas, y, weight, kernel_name="rotated_box_decode"):
    """
    implementation of rotated_box_decode and return the tik instance
    ----------------------------------------------------------------
    :param anchor_box: dict of shape and dtype of input anchor bbox
    :param deltas: dict of shape and dtype of input delta bbox
    :param y: dict of shape and dtype of output decode bbox
    :param weight: the weight for decode bounding box
    :param kernel_name: the kernel's name
    :return: tik instance
    """
    _check_param(anchor_box, deltas, y, weight)
    obj = RotatedBoxDecode(anchor_box, deltas, y, weight, kernel_name)
    obj.tik_inst_function()

    return obj.tik_inst
