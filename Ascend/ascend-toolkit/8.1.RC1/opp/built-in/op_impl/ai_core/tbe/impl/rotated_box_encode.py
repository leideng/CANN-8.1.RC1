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
rotated_box_encode
"""

from tbe.common.platform import get_bit_len
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


# 'pylint: disable=too-many-lines,invalid-name,too-many-arguments,consider-using-in
# 'pylint: disable=too-many-branches,too-many-instance-attributes,too-many-locals
# 'pylint: disable=too-many-statements,no-self-use,too-few-public-methods
# 'pylint: disable=unused-argument
class RotatedBoxEncode():
    """
    Function: use to finish RotatedBoxEncode main functions
    """

    def __init__(self, anchor_box, gt_box, y, weight, kernel_name):
        """
        init RotatedBoxEncode parameters
        ---------------------------------------------------------------
        :param anchor_box: dict of shape and dtype for input anchor box
        :param gt_box: dict of shape and dtype for input gt box
        :param y: dict of shape and dtype for output encode box
        :param weight: the weight for encode bounding box
        :param kernel_name: the kernel's name
        :return: None
        """
        self.init_tik_inst()
        self.anchor_shape = anchor_box.get("shape")
        self.anchor_dtype = anchor_box.get("dtype").lower()
        self.gt_shape = gt_box.get("shape")
        self.gt_dtype = gt_box.get("dtype").lower()
        self.encode_shape = y.get("shape")
        self.encode_dtype = y.get("dtype").lower()
        self.weight = weight

        self.kernel_name = kernel_name
        self.batch = self.encode_shape[0]
        self.bbox_num = self.encode_shape[1]
        self.bbox_len = self.encode_shape[2]
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
        self.encode_ub = None
        self.encode_ub_f16 = None
        self.gt_ub = None
        self.gt_ub_f16 = None
        self.anchor_pos_ub = None
        self.gt_pos_ub = None

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
        if self.gt_dtype == "float16":
            total_ele = total_ele // 2
        self.one_core_ele = (total_ele + core_num - 1) // core_num
        self.act_core_num = total_ele // self.one_core_ele
        if total_ele % self.one_core_ele != 0:
            self.act_core_num = self.act_core_num + 1
        self.last_core_ele = total_ele - (self.act_core_num - 1) * self.one_core_ele

        if self.gt_dtype == "float16":
            self.one_core_ele = self.one_core_ele * 2
            self.last_core_ele = self.last_core_ele * 2

    def tiling_param_caculate(self):
        """tiling_param_caculate
        """
        self.dtype_size = get_bit_len(self.dtype) // 8
        self.dtype_gt_size = get_bit_len(self.gt_dtype) // 8
        self.num_each_block = Constant.BLOCK_SIZE // self.dtype_size
        self.num_each_block_f16 = Constant.BLOCK_SIZE // self.dtype_gt_size
        self.align_unit = Constant.BBOX_NUM * self.num_each_block
        self.align_unit_f16 = Constant.BBOX_NUM * self.num_each_block_f16
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_ele = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) -
                       Constant.RESERVED_UB_SIZE) // self.dtype_size // self.align_unit

        if self.gt_dtype == "float16":
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
        self.gt_gm = self.tik_inst.Tensor(self.gt_dtype,
                                          (self.batch, self.bbox_num, self.bbox_len),
                                          name="gt_gm",
                                          scope=tik.scope_gm)
        self.encode_gm = self.tik_inst.Tensor(self.encode_dtype,
                                              (self.batch, self.bbox_num, self.bbox_len),
                                              name="encode_gm",
                                              scope=tik.scope_gm)

    def init_ub_tensor(self):
        """init_ub_tensor
        """
        self.anchor_ub = self.tik_inst.Tensor(self.dtype,
                                              (Constant.BBOX_NUM,
                                               self.max_ele * self.align_unit // Constant.BBOX_NUM),
                                              name="anchor_ub", scope=tik.scope_ubuf)

        self.gt_ub = self.tik_inst.Tensor(self.dtype,
                                          (Constant.BBOX_NUM,
                                           self.max_ele * self.align_unit // Constant.BBOX_NUM),
                                          name="gt_ub", scope=tik.scope_ubuf)

        self.encode_ub = self.tik_inst.Tensor(self.dtype,
                                              (Constant.BBOX_NUM,
                                               self.max_ele * self.align_unit // Constant.BBOX_NUM),
                                              name="encode_ub", scope=tik.scope_ubuf)

        self.anchor_pos_ub = self.tik_inst.Tensor(self.dtype,
                                                  (Constant.BBOX_NUM,
                                                   self.max_ele * self.align_unit // Constant.BBOX_NUM),
                                                  name="anchor_pos_ub", scope=tik.scope_ubuf)

        self.gt_pos_ub = self.tik_inst.Tensor(self.dtype,
                                              (Constant.BBOX_NUM,
                                               self.max_ele * self.align_unit // Constant.BBOX_NUM),
                                              name="gt_pos_ub", scope=tik.scope_ubuf)

        if self.gt_dtype == "float16":
            self.anchor_ub_f16 = self.tik_inst.Tensor(self.anchor_dtype,
                                                      (Constant.BBOX_NUM,
                                                       self.tensor_f16_len // Constant.BBOX_NUM),
                                                      name="anchor_ub_f16", scope=tik.scope_ubuf)

            self.gt_ub_f16 = self.tik_inst.Tensor(self.gt_dtype,
                                                  (Constant.BBOX_NUM,
                                                   self.tensor_f16_len // Constant.BBOX_NUM),
                                                  name="gt_ub_f16", scope=tik.scope_ubuf)

            self.encode_ub_f16 = self.tik_inst.Tensor(self.encode_dtype,
                                                      (Constant.BBOX_NUM,
                                                       self.tensor_f16_len // Constant.BBOX_NUM),
                                                      name="encode_ub_f16", scope=tik.scope_ubuf)

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
        if self.gt_dtype == "float16":
            gt_ub = self.gt_ub_f16
            anchor_ub = self.anchor_ub_f16
            ub_ele = ub_ele // 2
            num_each_block = self.num_each_block_f16
        else:
            gt_ub = self.gt_ub
            anchor_ub = self.anchor_ub
            num_each_block = self.num_each_block

        with self.tik_inst.for_range(0, Constant.BBOX_NUM) as bbox_idx:
            with self.tik_inst.if_scope(overlap > 0):
                with self.tik_inst.if_scope(ub_ele - 1 > 0):
                    self.tik_inst.data_move(anchor_ub[bbox_idx, :],
                                            self.anchor_gm[batch_idx, bbox_idx, offset:],
                                            0, 1, ub_ele - 1, 0, 0)
                    self.tik_inst.data_move(gt_ub[bbox_idx, :],
                                            self.gt_gm[batch_idx, bbox_idx, offset:],
                                            0, 1, ub_ele - 1, 0, 0)

                align_offset = offset + (ub_ele - 1) * num_each_block - overlap
                ub_offset = (ub_ele - 1) * num_each_block

                self.tik_inst.data_move(anchor_ub[bbox_idx, ub_offset:],
                                        self.anchor_gm[batch_idx, bbox_idx, align_offset:],
                                        0, 1, 1, 0, 0)
                self.tik_inst.data_move(gt_ub[bbox_idx, ub_offset:],
                                        self.gt_gm[batch_idx, bbox_idx, align_offset:],
                                        0, 1, 1, 0, 0)

            with self.tik_inst.else_scope():
                self.tik_inst.data_move(anchor_ub[bbox_idx, :],
                                        self.anchor_gm[batch_idx, bbox_idx, offset:],
                                        0, 1, ub_ele, 0, 0)
                self.tik_inst.data_move(gt_ub[bbox_idx, :],
                                        self.gt_gm[batch_idx, bbox_idx, offset:],
                                        0, 1, ub_ele, 0, 0)
        if self.gt_dtype == "float16":
            with self.tik_inst.for_range(0, Constant.BBOX_NUM) as bbox_idx:
                with self.tik_inst.if_scope(repeat_time > 0):
                    self.tik_inst.vconv(self.mask, 'none', self.anchor_ub[bbox_idx, :], anchor_ub[bbox_idx, :],
                                        repeat_time, 1, 1, 8, 4)

                    self.tik_inst.vconv(self.mask, 'none', self.gt_ub[bbox_idx, :], gt_ub[bbox_idx, :],
                                        repeat_time, 1, 1, 8, 4)

                with self.tik_inst.if_scope(repeat_left > 0):
                    self.tik_inst.vconv(repeat_left, 'none', self.anchor_ub[bbox_idx, repeat_offset:],
                                        anchor_ub[bbox_idx, repeat_offset:], 1, 1, 1, 8, 4)

                    self.tik_inst.vconv(repeat_left, 'none', self.gt_ub[bbox_idx, repeat_offset:],
                                        gt_ub[bbox_idx, repeat_offset:], 1, 1, 1, 8, 4)

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
        if self.encode_dtype == "float16":
            ub_ele = ub_ele // 2
            encode_ub = self.encode_ub_f16
            num_each_block = self.num_each_block_f16
            with self.tik_inst.for_range(0, Constant.BBOX_NUM) as bbox_idx:
                with self.tik_inst.if_scope(repeat_time > 0):
                    self.tik_inst.vconv(self.mask, "none", self.encode_ub_f16[bbox_idx, :],
                                        self.encode_ub[bbox_idx, :], repeat_time, 1, 1, 4, 8)
                with self.tik_inst.if_scope(repeat_left > 0):
                    self.tik_inst.vconv(repeat_left, "none", self.encode_ub_f16[bbox_idx, repeat_offset:],
                                        self.encode_ub[bbox_idx, repeat_offset:], 1, 1, 1, 4, 8)
        else:
            encode_ub = self.encode_ub
            num_each_block = self.num_each_block

        with self.tik_inst.for_range(0, Constant.BBOX_NUM) as bbox_idx:
            with self.tik_inst.if_scope(overlap > 0):
                with self.tik_inst.if_scope(ub_ele - 1 > 0):
                    self.tik_inst.data_move(self.encode_gm[batch_idx, bbox_idx, offset:],
                                            encode_ub[bbox_idx, :],
                                            0, 1, ub_ele - 1, 0, 0)
                align_offset = offset + (ub_ele - 1) * num_each_block - overlap
                ub_offset = (ub_ele - 1) * num_each_block

                self.tik_inst.data_move(self.encode_gm[batch_idx, bbox_idx, align_offset:],
                                        encode_ub[bbox_idx, ub_offset:],
                                        0, 1, 1, 0, 0)

            with self.tik_inst.else_scope():
                self.tik_inst.data_move(self.encode_gm[batch_idx, bbox_idx, offset:],
                                        encode_ub[bbox_idx, :],
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

    def get_target_center(self, src_0_ub, src_1_ub, src_2_ub, weight, center_info_ub,
                          repeat_time, repeat_left, offset):
        """
        :param src_0_ub: input tensor 0
        :param src_1_ub: input tensor 1
        :param src_2_ub: input tensor 2
        :param weight: the weight for center
        :param center_info_ub: result tensor
        :param repeat_time: int, the repeat time for tik instruction
        :param repeat_left: int, the repeat left for tik instrucion
        :param offset: int, the repeat offset for tik instruction
        :return: None
        """
        with self.tik_inst.new_stmt_scope():
            with self.tik_inst.if_scope(repeat_time > 0):
                self.tik_inst.vsub(self.mask, center_info_ub, src_0_ub, src_1_ub, repeat_time, 1, 1, 1, 8, 8, 8)
                if self.support_div is True:
                    self.tik_inst.vdiv(self.mask, center_info_ub, center_info_ub, src_2_ub,
                                       repeat_time, 1, 1, 1, 8, 8, 8)
                else:
                    self.tik_inst.vrec(self.mask, src_2_ub, src_2_ub, repeat_time, 1, 1, 8, 8)
                    self.tik_inst.vmul(self.mask, center_info_ub, center_info_ub, src_2_ub,
                                       repeat_time, 1, 1, 1, 8, 8, 8)
                self.tik_inst.vmuls(self.mask, center_info_ub, center_info_ub, weight, repeat_time, 1, 1, 8, 8)

            with self.tik_inst.if_scope(repeat_left > 0):
                self.tik_inst.vsub(repeat_left, center_info_ub[0, offset:], src_0_ub[0, offset:],
                                   src_1_ub[0, offset:], 1, 1, 1, 1, 8, 8, 8)
                if self.support_div is True:
                    self.tik_inst.vdiv(repeat_left, center_info_ub[0, offset:], center_info_ub[0, offset:],
                                       src_2_ub[0, offset:], 1, 1, 1, 1, 8, 8, 8)
                else:
                    self.tik_inst.vrec(repeat_left, src_2_ub[0, offset:], src_2_ub[0, offset:], 1, 1, 1, 8, 8)
                    self.tik_inst.vmul(repeat_left, center_info_ub[0, offset:], center_info_ub[0, offset:],
                                       src_2_ub[0, offset:], 1, 1, 1, 1, 8, 8, 8)
                self.tik_inst.vmuls(repeat_left, center_info_ub[0, offset:], center_info_ub[0, offset:],
                                    weight, 1, 1, 1, 8, 8)

    def get_target_wh(self, src_0_ub, src_1_ub, encode_ub, weight, repeat_time, repeat_left, offset):
        """
        :param src_0_ub: input tensor 0
        :param src_1_ub: input tensor 1
        :param encode_ub: result encode tensor
        :param weight: the weight for w or h
        :param repeat_time: int, the repeat time for tik instruction
        :param repeat_left: int, the repeat left for tik instrucion
        :param offset: int, the repeat offset for tik instruction
        :return: None
        """
        with self.tik_inst.if_scope(repeat_time > 0):
            if self.support_div is True:
                self.tik_inst.vdiv(self.mask, encode_ub, src_0_ub, src_1_ub, repeat_time, 1, 1, 1, 8, 8, 8)
            else:
                self.tik_inst.vrec(self.mask, encode_ub, src_1_ub, repeat_time, 1, 1, 8, 8)
                self.tik_inst.vmul(self.mask, encode_ub, src_0_ub, encode_ub, repeat_time, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vln(self.mask, encode_ub, encode_ub, repeat_time, 1, 1, 8, 8)
            self.tik_inst.vmuls(self.mask, encode_ub, encode_ub, weight, repeat_time, 1, 1, 8, 8)

        with self.tik_inst.if_scope(repeat_left > 0):
            if self.support_div is True:
                self.tik_inst.vdiv(repeat_left, encode_ub[0, offset:], src_0_ub[0, offset:], src_1_ub[0, offset:],
                                   1, 1, 1, 1, 8, 8, 8)
            else:
                self.tik_inst.vrec(repeat_left, encode_ub[0, offset:], src_1_ub[0, offset:], 1, 1, 1, 8, 8)
                self.tik_inst.vmul(repeat_left, encode_ub[0, offset:], src_0_ub[0, offset:], encode_ub[0, offset:],
                                   1, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vln(repeat_left, encode_ub[0, offset:], encode_ub[0, offset:], 1, 1, 1, 8, 8)
            self.tik_inst.vmuls(repeat_left, encode_ub[0, offset:], encode_ub[0, offset:], weight, 1, 1, 1, 8, 8)

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

    def get_target_angle(self, gt_angle, anchor_angle, encode_angle, weight, repeat_time, repeat_left, offset):
        """
        :param gt_angle: gt angle tensor
        :param anchor_angle: anchor angle tensor
        :param encode_angle: encode anelg tensor
        :param weight: the weight for angle
        :param repeat_time: int, the repeat time for tik instruction
        :param repeat_left: int, the repeat left for tik instrucion
        :param offset: int, the repeat offset for tik instruction
        :return: None
        """
        with self.tik_inst.if_scope(repeat_time > 0):
            self.tik_inst.vmuls(self.mask, gt_angle, gt_angle, Constant.PI / 180.0, repeat_time, 1, 1, 8, 8)
            self.compute_tan(self.mask, repeat_time, gt_angle, self.gt_pos_ub[0, :], self.gt_pos_ub[1, :])
            self.tik_inst.vmuls(self.mask, anchor_angle, anchor_angle, Constant.PI / 180.0, repeat_time, 1, 1, 8, 8)
            self.compute_tan(self.mask, repeat_time, anchor_angle, self.anchor_pos_ub[0, :], self.anchor_pos_ub[1, :])

            self.tik_inst.vsub(self.mask, encode_angle, gt_angle, anchor_angle, repeat_time, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vmuls(self.mask, encode_angle, encode_angle, weight, repeat_time, 1, 1, 8, 8)

        with self.tik_inst.if_scope(repeat_left > 0):
            self.tik_inst.vmuls(repeat_left, gt_angle[0, offset:], gt_angle[0, offset:],
                                Constant.PI / 180.0, 1, 1, 1, 8, 8)
            self.compute_tan(repeat_left, 1, gt_angle[0, offset:],
                             self.gt_pos_ub[0, offset:], self.gt_pos_ub[1, offset:])
            self.tik_inst.vmuls(repeat_left, anchor_angle[0, offset:], anchor_angle[0, offset:],
                                Constant.PI / 180.0, 1, 1, 1, 8, 8)
            self.compute_tan(repeat_left, 1, anchor_angle[0, offset:],
                             self.anchor_pos_ub[0, offset:], self.anchor_pos_ub[1, offset:])
            self.tik_inst.vsub(repeat_left, encode_angle[0, offset:], gt_angle[0, offset:], anchor_angle[0, offset:],
                               1, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vmuls(repeat_left, encode_angle[0, offset:], encode_angle[0, offset:], weight, 1, 1, 1, 8, 8)

    def rotated_box_encode_compute(self, repeat_time, repeat_left, offset):
        """
        :param repeat_time: int, the repeat time for tik instruction
        :param repeat_left: int, the repeat left for tik instrucion
        :param offset: int, the repeat offset for tik instruction
        :return: None
        """
        self.corner_to_center(self.anchor_ub, self.anchor_pos_ub, repeat_time, repeat_left, offset)
        self.corner_to_center(self.gt_ub, self.gt_pos_ub, repeat_time, repeat_left, offset)

        self.get_target_center(self.gt_pos_ub[0, :], self.anchor_pos_ub[0, :],
                               self.anchor_pos_ub[2, :], self.weight[0],
                               self.encode_ub[0, :], repeat_time, repeat_left, offset)

        self.get_target_center(self.gt_pos_ub[1, :], self.anchor_pos_ub[1, :],
                               self.anchor_pos_ub[3, :], self.weight[1],
                               self.encode_ub[1, :], repeat_time, repeat_left, offset)

        self.get_target_wh(self.gt_pos_ub[2, :], self.anchor_pos_ub[2, :],
                           self.encode_ub[2, :], self.weight[2],
                           repeat_time, repeat_left, offset)

        self.get_target_wh(self.gt_pos_ub[3, :], self.anchor_pos_ub[3, :],
                           self.encode_ub[3, :], self.weight[3],
                           repeat_time, repeat_left, offset)

        self.get_target_angle(self.gt_pos_ub[4, :], self.anchor_pos_ub[4, :],
                              self.encode_ub[4, :], self.weight[4],
                              repeat_time, repeat_left, offset)

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
            self.rotated_box_encode_compute(self.repeat_time, self.repeat_left, self.repeat_time * self.mask)
            self.move_out_ub_data(self.repeat_time, self.repeat_left, self.repeat_time * self.mask,
                                  batch_idx, offset // Constant.BBOX_NUM, self.max_ele, 0)

        with self.tik_inst.if_scope(loop_left > 0):
            self.repeat_time.set_as(loop_left * self.num_each_block // self.mask)
            self.repeat_left.set_as(loop_left * self.num_each_block % self.mask)
            offset = base_offset + loop_num * self.max_ele * self.align_unit
            self.move_in_gm_data(self.repeat_time, self.repeat_left, self.repeat_time * self.mask,
                                 batch_idx, offset // Constant.BBOX_NUM, loop_left, overlap)
            self.rotated_box_encode_compute(self.repeat_time, self.repeat_left, self.repeat_time * self.mask)
            self.move_out_ub_data(self.repeat_time, self.repeat_left, self.repeat_time * self.mask,
                                  batch_idx, offset // Constant.BBOX_NUM, loop_left, overlap)

    def encode_compute_tiling(self):
        """encode_compute_tiling
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
        self.encode_compute_tiling()
        self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                               inputs=[self.anchor_gm, self.gt_gm],
                               outputs=[self.encode_gm])


def _check_param(anchor_box, gt_box, y, weight):
    """
    check parameters, if one is invalid, then raise error
    -----------------------
    :param anchor_box: dict of shape and dtype of input anchor bbox
    :param gt_box: dict of shape and dtype of input gt bbox
    :param y: dict of shape and dtype of output encode bbox
    :param weight: the weight for encode bounding box
    :return: None
    """
    anchor_shape = anchor_box.get("shape")
    anchor_dtype = anchor_box.get("dtype").lower()
    gt_shape = gt_box.get("shape")
    gt_dtype = gt_box.get("dtype").lower()
    encode_shape = y.get("shape")
    encode_dtype = y.get("dtype").lower()

    if anchor_dtype != gt_dtype or anchor_dtype != encode_dtype or gt_dtype != encode_dtype:
        raise RuntimeError("anchor dtype, gt dtype and encode dtype must be same.")

    if anchor_shape != gt_shape or anchor_shape != encode_shape or gt_shape != encode_shape:
        raise RuntimeError("anchor shape, gt shape and encode shape must be same.")

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


def rotated_box_encode(anchor_box, gt_box, y, weight, kernel_name="rotated_box_encode"):
    """
    implementation of rotated_box_encode and return the tik instance
    ----------------------------------------------------------------
    :param anchor_box: dict of shape and dtype of input anchor bbox
    :param gt_box: dict of shape and dtype of input gt bbox
    :param y: dict of shape and dtype of output encode bbox
    :param weight: the weight for encode bounding box
    :param kernel_name: the kernel's name
    :return: tik instance
    """
    _check_param(anchor_box, gt_box, y, weight)
    obj = RotatedBoxEncode(anchor_box, gt_box, y, weight, kernel_name)
    obj.tik_inst_function()

    return obj.tik_inst
