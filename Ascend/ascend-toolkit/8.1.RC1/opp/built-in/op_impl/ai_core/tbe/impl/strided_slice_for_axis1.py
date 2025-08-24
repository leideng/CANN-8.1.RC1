#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (C) Huawei Technologies Co., Ltd 2023-2023. All rights reserved.
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
strided_slice_for_axis1
"""
from impl import constant_util as constant
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.util_tik_comm_func import floor_align
from impl.util.util_tik_comm_func import ceil_div


# 'pylint: disable=old-style-class,no-init,too-few-public-methods
class Ssd1Constant:
    UB_SIZE_MIN = 131072
    UB_SIZE_WORK = 65536


# 'pylint: disable=old-style-class,too-many-instance-attributes,too-few-public-methods,too-many-arguments
class SliceWithAxis1:
    def __init__(self, x, y, begin, end, strides=None, begin_mask=0, end_mask=0,
                 ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0, kernel_name="strided_slice_d"):
        """
        """
        self.x = x
        self.y = y

        self.begin = list(begin)
        self.end = list(end)
        if strides is None:
            self.strides = [1] * len(x.get("shape"))
        else:
            self.strides = list(strides)

        self.begin_mask = begin_mask
        self.end_mask = end_mask
        self.ellipsis_mask = ellipsis_mask
        self.new_axis_mask = new_axis_mask
        self.shrink_axis_mask = shrink_axis_mask
        self.kernel_name = kernel_name

        self.x_shape = shape_util.shape_to_list(self.x.get("shape"))
        self.y_shape = shape_util.shape_to_list(self.y.get("shape"))
        self.d_type = self.x.get("dtype").lower()
        self.d_size = tbe_platform.get_bit_len(self.d_type) // constant.DATA_SIZE_EIGHT
        self.hw = None
        self.i_chw = None
        self.o_chw = None

        self.tik_inst = tik.Tik()
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.x_ub_num = Ssd1Constant.UB_SIZE_WORK // self.d_size

        self.x_gm = None
        self.y_gm = None
        self.x_ub = None

        self.core_offset = None
        self.offset_x = None
        self.offset_y = None
        self.tail_offset = None

    # 'pylint: disable=too-many-return-statements
    def check_params(self):
        if not tbe_platform.api_check_support("tik.vcopy"):
            return False
        if self.ub_size < Ssd1Constant.UB_SIZE_MIN:
            return False

        if self.d_type not in ('float16', 'float32'):
            return False
        if any((self.x.get("format") != "ND", len(self.x_shape) != 4, len(self.y_shape) != 4,
                -1 in self.x_shape, -1 in self.y_shape, 0 in self.x_shape, 0 in self.y_shape,
                len(self.begin) != 4, len(self.end) != 4, len(self.strides) != 4)):
            return False

        if any((self.y_shape[0] > 8, self.y_shape[1] < self.core_num, self.y_shape[2] * self.y_shape[3] % 128 != 0)):
            return False
        if any((self.strides != [1, 1, 1, 1], self.begin_mask != 0, self.end_mask != 0, self.ellipsis_mask != 0,
                self.new_axis_mask != 0, self.shrink_axis_mask != 0)):
            return False

        self._refresh_params()
        if any((self.begin[0] != 0, self.begin[2] != 0, self.begin[3] != 0,
                self.end[0] != self.x_shape[0], self.end[2] != self.x_shape[2], self.end[3] != self.x_shape[3])):
            return False
        if not self._check_axis1():
            return False

        return True

    def compute(self):
        self.hw = self.x_shape[2] * self.x_shape[3]
        self.i_chw = self.x_shape[1] * self.hw
        self.o_chw = self.y_shape[1] * self.hw

        self.x_gm = self.tik_inst.Tensor(self.d_type, self.x_shape, name="x_gm", scope=tik.scope_gm)
        self.y_gm = self.tik_inst.Tensor(self.d_type, self.y_shape, name="y_gm", scope=tik.scope_gm)

        self.core_offset = self.tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="core_offset")
        self.offset_x = self.tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="offset_x")
        self.offset_y = self.tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="offset_y")
        self.tail_offset = self.tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="tail_offset")

        with self.tik_inst.for_range(0, self.core_num, block_num=self.core_num) as core_id:
            self.x_ub = self.tik_inst.Tensor(self.d_type, (self.x_ub_num,), name="x_ub", scope=tik.scope_ubuf)

            pre_core_num = self.y_shape[1] % self.core_num
            if pre_core_num == 0:
                pre_core_num = self.core_num
            num_pre_core = ceil_div(self.y_shape[1], self.core_num) * self.hw
            num_post_core = self.y_shape[1] // self.core_num * self.hw

            with self.tik_inst.if_scope(core_id < pre_core_num):
                self.core_offset.set_as(core_id * num_pre_core)
                self._compute_one_core(num_pre_core)
            with self.tik_inst.else_scope():
                self.core_offset.set_as(core_id * num_post_core + pre_core_num * self.hw)
                self._compute_one_core(num_post_core)

        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                               inputs=[self.x_gm],
                               outputs=[self.y_gm],
                               config=opt_config)

    def _compute_one_core(self, cur_num):
        with self.tik_inst.for_range(0, self.x_shape[0]) as i0:
            if cur_num >= self.x_ub_num:
                loop_burst = self.x_ub_num * self.d_size // constant.BLOCK_SIZE
                with self.tik_inst.for_range(0, cur_num // self.x_ub_num) as i1:
                    self.offset_x.set_as(i0 * self.i_chw + self.begin[1] * self.hw + self.core_offset +
                                         i1 * self.x_ub_num)
                    self.offset_y.set_as(i0 * self.o_chw + self.core_offset + i1 * self.x_ub_num)
                    self.tik_inst.data_move(self.x_ub, self.x_gm[self.offset_x], 0, 1, loop_burst, 0, 0)
                    self.tik_inst.data_move(self.y_gm[self.offset_y], self.x_ub, 0, 1, loop_burst, 0, 0)
            if cur_num % self.x_ub_num > 0:
                self.tail_offset.set_as(self.core_offset + floor_align(cur_num, self.x_ub_num))
                self.offset_x.set_as(i0 * self.i_chw + self.begin[1] * self.hw + self.tail_offset)
                self.offset_y.set_as(i0 * self.o_chw + self.tail_offset)

                tail_busrt = cur_num % self.x_ub_num * self.d_size // constant.BLOCK_SIZE
                self.tik_inst.data_move(self.x_ub, self.x_gm[self.offset_x], 0, 1, tail_busrt, 0, 0)
                self.tik_inst.data_move(self.y_gm[self.offset_y], self.x_ub, 0, 1, tail_busrt, 0, 0)

    def _refresh_params(self):
        x_shape = shape_util.shape_to_list(self.x.get("shape"))

        begin_tmp = []
        for i, v in enumerate(self.begin):
            if v > x_shape[i]:
                begin_tmp.append(x_shape[i])
            elif v < 0:
                begin_tmp.append(v + x_shape[i])
            else:
                begin_tmp.append(v)
        self.begin = begin_tmp

        end_tmp = []
        for i, v in enumerate(self.end):
            if v > x_shape[i]:
                end_tmp.append(x_shape[i])
            elif v < 0:
                end_tmp.append(v + x_shape[i])
            else:
                end_tmp.append(v)
        self.end = end_tmp

    def _check_axis1(self):
        if self.begin[1] == 0:
            if 0 < self.end[1] <= self.x_shape[1]:
                return True
        if self.end[1] == self.x_shape[1]:
            if 0 <= self.begin[1] < self.x_shape[1]:
                return True
        return False
