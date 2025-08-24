#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
# =============================================================================
"""
roi_align_grad
"""
from abc import ABCMeta
from functools import partial
from typing import List
from impl import common_util
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.util_tik_comm_func import OpBase, ceil_div


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # ting param num
    TILING_ARG_NUM = 16
    # reserved ub size for temporary variables
    RESERVED_UB_SIZE = 30 * 1024
    # batch size, 64 align for float32 or 128 align for float16
    BATCH_SIZE = 128
    # nBurst max value of `data_move`
    BURST_NUM_MAX = 4095
    # burstLen max value of `data_move`
    BURST_LEN_MAX = 65535
    # stride max value of `data_move`
    BURST_STRIDE_MAX = 65535
    # whether exceed BURST_STRIDE_MAX
    EXCEED_BURST_STRIDE = 1
    # x_rep_stride for instr like vadd
    REPEAT_STRIDE_MAX = 255
    # instr max repeat value
    INSTR_REPEAT_MAX = 255


class ProcessorBase(metaclass=ABCMeta):
    def __init__(self, op_obj: "RoiAlignGrad") -> None:
        self.op = op_obj
        self.tik_inst: tik.Tik = op_obj.tik_instance
        self.support_fp32_vextract = tbe_platform.api_check_support("tik.vextract", "float32")

        self.roi_in_ub = self.tik_inst.Tensor("float32", (5, Constant.BATCH_SIZE),
                                              name="roi_in_ub", scope=tik.scope_ubuf)

        self.cur_roi_start_x = self.tik_inst.Scalar("float32", "cur_roi_start_x")
        self.cur_roi_start_y = self.tik_inst.Scalar("float32", "cur_roi_start_y")
        # c1 offset and c1 count
        self.c1_offset = self.tik_inst.Scalar("int64", "c1_offset")
        self.c1_num = self.tik_inst.Scalar("int64", "c1_num")
        # roi width / height
        self.roi_wh = self.tik_inst.Tensor("float32", (2, Constant.BATCH_SIZE), name="roi_wh", scope=tik.scope_ubuf)
        # grid width / height
        self.grid_w = self.tik_inst.Tensor("float32", (Constant.BATCH_SIZE,), name="grid_w", scope=tik.scope_ubuf)
        self.grid_h = self.tik_inst.Tensor("float32", (Constant.BATCH_SIZE,), name="grid_h", scope=tik.scope_ubuf)
        # current roi grid height / width
        self.cur_roi_grid_h = self.tik_inst.Scalar("float32", "cur_roi_grid_h")
        self.cur_roi_grid_w = self.tik_inst.Scalar("float32", "cur_roi_grid_w")
        # height / width sample count
        self.sample_h = self.tik_inst.Tensor("int32", (Constant.BATCH_SIZE,), name="sample_h", scope=tik.scope_ubuf)
        self.sample_w = self.tik_inst.Tensor("int32", (Constant.BATCH_SIZE,), name="sample_w", scope=tik.scope_ubuf)
        self.sample_h_fp = self.tik_inst.Tensor("float32", (Constant.BATCH_SIZE,),
                                                name="sample_h_fp", scope=tik.scope_ubuf)
        self.sample_w_fp = self.tik_inst.Tensor("float32", (Constant.BATCH_SIZE,),
                                                name="sample_w_fp", scope=tik.scope_ubuf)
        # current roi samples
        self.cur_roi_sample_h = self.tik_inst.Scalar("int32", "cur_roi_sample_h")
        self.cur_roi_sample_w = self.tik_inst.Scalar("int32", "cur_roi_sample_w")
        self.cur_roi_sample_h_fp = self.tik_inst.Scalar("float32", "cur_roi_sample_h_fp")
        self.cur_roi_sample_w_fp = self.tik_inst.Scalar("float32", "cur_roi_sample_w_fp")

        # feature map index specified in ROI
        self.fm_idx = self.tik_inst.Tensor("int32", (Constant.BATCH_SIZE,), name="fm_idx", scope=tik.scope_ubuf)
        self.cur_fm_idx = self.tik_inst.Scalar("int32", "cur_fm_idx")
        # pool height / width index to find elements in y_diff
        self.pool_h_idx = self.tik_inst.Tensor("int32", (Constant.BATCH_SIZE,), name="pool_h_idx", scope=tik.scope_ubuf)
        self.pool_w_idx = self.tik_inst.Tensor("int32", (Constant.BATCH_SIZE,), name="pool_w_idx", scope=tik.scope_ubuf)
        # grid coordinates
        self.x = self.tik_inst.Tensor("float32", (Constant.BATCH_SIZE,), name="x", scope=tik.scope_ubuf)
        self.y = self.tik_inst.Tensor("float32", (Constant.BATCH_SIZE,), name="y", scope=tik.scope_ubuf)
        # 4 points around
        self.x_low = self.tik_inst.Tensor("int32", (Constant.BATCH_SIZE,), name="x_low", scope=tik.scope_ubuf)
        self.y_low = self.tik_inst.Tensor("int32", (Constant.BATCH_SIZE,), name="y_low", scope=tik.scope_ubuf)
        self.x_high = self.tik_inst.Tensor("int32", (Constant.BATCH_SIZE,), name="x_high", scope=tik.scope_ubuf)
        self.y_high = self.tik_inst.Tensor("int32", (Constant.BATCH_SIZE,), name="y_high", scope=tik.scope_ubuf)
        # length of 4 points
        self.lx = self.tik_inst.Tensor("float32", (Constant.BATCH_SIZE,), name="lx", scope=tik.scope_ubuf)
        self.hx = self.tik_inst.Tensor("float32", (Constant.BATCH_SIZE,), name="hx", scope=tik.scope_ubuf)
        self.ly = self.tik_inst.Tensor("float32", (Constant.BATCH_SIZE,), name="ly", scope=tik.scope_ubuf)
        self.hy = self.tik_inst.Tensor("float32", (Constant.BATCH_SIZE,), name="hy", scope=tik.scope_ubuf)
        # weight of 4 points, w1=hy*hx, w2=hy*lx, w3=ly*hx, w4=ly*lx
        self.w1 = self.tik_inst.Tensor("float32", (Constant.BATCH_SIZE,), name="w1", scope=tik.scope_ubuf)
        self.w2 = self.tik_inst.Tensor("float32", (Constant.BATCH_SIZE,), name="w2", scope=tik.scope_ubuf)
        self.w3 = self.tik_inst.Tensor("float32", (Constant.BATCH_SIZE,), name="w3", scope=tik.scope_ubuf)
        self.w4 = self.tik_inst.Tensor("float32", (Constant.BATCH_SIZE,), name="w4", scope=tik.scope_ubuf)

        # help tensors
        # filled with 0.0 in tensor, total count: 8
        self.const_zero_fp = self.tik_inst.Tensor("float32", (8,), name="const_zero_fp", scope=tik.scope_ubuf)
        self.tik_inst.vector_dup(8, self.const_zero_fp, 0, 1, 0, 0)
        # filled with 1.0 in tensor, total count: 8
        self.const_one_fp = self.tik_inst.Tensor("float32", (8,), name="const_one_fp", scope=tik.scope_ubuf)
        self.tik_inst.vector_dup(8, self.const_one_fp, 1, 1, 0, 0)
        # filled with 1 in tensor, total count: 8
        self.const_one_int = self.tik_inst.Tensor("int32", (8,), name="const_one_int", scope=tik.scope_ubuf)
        self.tik_inst.vector_dup(8, self.const_one_int, 1, 1, 0, 0)
        # filled 0...127 in tensor
        self.const_0_127_fp = self.tik_inst.Tensor("float32", (Constant.BATCH_SIZE,),
                                                   name="const_0_127_fp", scope=tik.scope_ubuf)
        # filled with -1.0 in tensor, total count: 8
        self.const_minus_1_fp = self.tik_inst.Tensor("float32", (8,), name="const_minus_1_fp", scope=tik.scope_ubuf)
        # filled with height/weight of feature_map / x_diff shape in tensor, total count: 8
        self.const_fm_h_fp = self.tik_inst.Tensor("float32", (8,), name="const_fm_h_fp", scope=tik.scope_ubuf)
        self.const_fm_w_fp = self.tik_inst.Tensor("float32", (8,), name="const_fm_w_fp", scope=tik.scope_ubuf)
        # filled with height-1/weight-1 of feature_map / x_diff shape in tensor, total count: 8
        self.const_fm_h_minus_1 = self.tik_inst.Tensor("int32", (8,), name="const_fm_h_minus_1", scope=tik.scope_ubuf)
        self.const_fm_w_minus_1 = self.tik_inst.Tensor("int32", (8,), name="const_fm_w_minus_1", scope=tik.scope_ubuf)
        # filled with height-1/weight-1 of feature_map / x_diff shape in tensor, total count: 8
        self.const_fm_h_minus_1_fp = self.tik_inst.Tensor("float32", (8,),
                                                          name="const_fm_h_minus_1_fp", scope=tik.scope_ubuf)
        self.const_fm_w_minus_1_fp = self.tik_inst.Tensor("float32", (8,),
                                                          name="const_fm_w_minus_1_fp", scope=tik.scope_ubuf)

        # temporary fp tensor
        self.tmp_fp = self.tik_inst.Tensor("float32", (2, Constant.BATCH_SIZE), name="tmp_fp", scope=tik.scope_ubuf)

        # help scalars
        self.mask = self.tik_inst.Scalar("int32", "mask")
        self.repeat = self.tik_inst.Scalar("int32", "repeat")

    def run(self) -> None:
        """main procedure"""
        self.prepare()
        roi_loops = ceil_div(self.op.core_rois, Constant.BATCH_SIZE)
        with self.tik_inst.for_range(0, roi_loops) as _roi_loop_idx:
            rois = self._calc_segment(self.op.core_rois, _roi_loop_idx, Constant.BATCH_SIZE)
            self.compute_roi_batch(self.op.core_roi_offset + _roi_loop_idx * Constant.BATCH_SIZE, rois)

    def prepare(self) -> None:
        """prepares"""
        op = self.op
        # clear duty data.
        self.tik_inst.vector_dup(64, self.roi_in_ub, 0.0, (5 * Constant.BATCH_SIZE + 63) // 64, 1, 8)
        with self.tik_inst.if_scope(op.samples > 0):
            self.tik_inst.vector_dup(64, self.sample_h, op.samples, 2, 1, 8)
            self.tik_inst.vector_dup(64, self.sample_w, op.samples, 2, 1, 8)
            self.tik_inst.vconv(64, "", self.sample_h_fp, self.sample_h, 2, 1, 1, 8, 8)
            self.tik_inst.vconv(64, "", self.sample_w_fp, self.sample_w, 2, 1, 1, 8, 8)
        # constant vectors prepare
        self.tik_inst.vector_dup(8, self.const_minus_1_fp, -1.0, 1, 0, 0)
        self.tik_inst.vector_dup(8, self.const_fm_h_fp, op.x_diff_h + 0, 1, 0, 0)
        self.tik_inst.vector_dup(8, self.const_fm_w_fp, op.x_diff_w + 0, 1, 0, 0)
        self.tik_inst.vector_dup(8, self.const_fm_h_minus_1, op.x_diff_h - 1, 1, 0, 0)
        self.tik_inst.vector_dup(8, self.const_fm_w_minus_1, op.x_diff_w - 1, 1, 0, 0)
        self.tik_inst.vector_dup(8, self.const_fm_h_minus_1_fp, op.x_diff_h - 1, 1, 0, 0)
        self.tik_inst.vector_dup(8, self.const_fm_w_minus_1_fp, op.x_diff_w - 1, 1, 0, 0)
        # 0-127 const index
        with self.tik_inst.for_range(0, Constant.BATCH_SIZE) as i:
            self.const_0_127_fp[i] = i

    def compute_roi_batch(self, roi_offset: tik.Scalar, roi_num: tik.Scalar):
        self._move_roi_data_to_ub(roi_num, roi_offset)
        self._calc_instr_mask_repeat(roi_num)
        self._calc_fm_start_end_coordinate()
        self._calc_roi_size(roi_num)
        self._calc_grid_size(roi_num)
        self._convert_fm_idx(roi_num)
        with self.tik_inst.for_range(0, roi_num) as _roi_idx:
            c1_offset, c1_num = self._calc_c1_offset_num(roi_offset + _roi_idx - self.op.core_roi_offset)
            self.compute_one_roi(roi_offset, _roi_idx, c1_offset, c1_num)

    def compute_one_roi(self, roi_offset: tik.Scalar, roi_idx: tik.Scalar, c1_offset: tik.Scalar, c1_num: tik.Scalar):
        self.cur_fm_idx.set_as(self.fm_idx[roi_idx])
        self.cur_roi_start_x.set_as(self.roi_in_ub[1, roi_idx])
        self.cur_roi_start_y.set_as(self.roi_in_ub[2, roi_idx])
        self.cur_roi_grid_h.set_as(self.grid_h[roi_idx])
        self.cur_roi_grid_w.set_as(self.grid_w[roi_idx])
        self.cur_roi_sample_h.set_as(self.sample_h[roi_idx])
        self.cur_roi_sample_w.set_as(self.sample_w[roi_idx])
        self.cur_roi_sample_h_fp.set_as(self.sample_h_fp[roi_idx])
        self.cur_roi_sample_w_fp.set_as(self.sample_w_fp[roi_idx])

        with self.tik_inst.if_scope(tik.all(self.cur_fm_idx >= 0, self.cur_fm_idx < self.op.x_diff_n)):
            x_weight_cache_idx = self.tik_inst.Scalar("int64", "x_weight_cache_idx", init_value=-1)
            grid_h_loops = ceil_div(self.cur_roi_sample_h * self.op.pool_h, Constant.BATCH_SIZE)
            with self.tik_inst.for_range(0, grid_h_loops) as _grid_h_loop_idx:
                grid_h_num = self._calc_segment(self.cur_roi_sample_h * self.op.pool_h,
                                                _grid_h_loop_idx, Constant.BATCH_SIZE)
                self.compute_grid_batch_h(roi_offset, roi_idx, c1_offset, c1_num,
                                          _grid_h_loop_idx * Constant.BATCH_SIZE, grid_h_num,
                                          x_weight_cache_idx)

    # 'pylint: disable=too-many-arguments
    def compute_grid_batch_h(self, roi_offset: tik.Scalar, roi_idx: tik.Scalar,
                             c1_offset: tik.Scalar, c1_num: tik.Scalar,
                             grid_h_idx_offset: tik.Scalar, grid_h_num: tik.Scalar,
                             x_weight_cache_idx: tik.Scalar):
        self._calc_instr_mask_repeat(grid_h_num)
        self._calc_pool_idx_of_grid(self.pool_h_idx, grid_h_num, grid_h_idx_offset, self.cur_roi_sample_h_fp)
        self._calc_bilinear_interpolate_coordinate(self.y, self.y_low, self.y_high, self.ly, self.hy,
                                                   self.cur_roi_sample_h_fp, grid_h_num, grid_h_idx_offset,
                                                   self.cur_roi_grid_h, self.cur_roi_start_y, False)
        grid_w_loops = ceil_div(self.cur_roi_sample_w * self.op.pool_w, Constant.BATCH_SIZE)
        with self.tik_inst.for_range(0, grid_w_loops) as _grid_w_loop_idx:
            grid_w_num = self._calc_segment(self.cur_roi_sample_w * self.op.pool_w,
                                            _grid_w_loop_idx, Constant.BATCH_SIZE)
            self.compute_grid_batch_w(roi_offset, roi_idx, c1_offset, c1_num, grid_h_num,
                                      _grid_w_loop_idx * Constant.BATCH_SIZE, grid_w_num,
                                      x_weight_cache_idx)

    # 'pylint: disable=too-many-arguments
    def compute_grid_batch_w(self, roi_offset: tik.Scalar, roi_idx: tik.Scalar,
                             c1_offset: tik.Scalar, c1_num: tik.Scalar,
                             grid_h_num: tik.Scalar,
                             grid_w_idx_offset: tik.Scalar, grid_w_num: tik.Scalar,
                             x_weight_cache_idx: tik.Scalar):
        op = self.op
        with self.tik_inst.if_scope(x_weight_cache_idx != grid_w_idx_offset):
            x_weight_cache_idx.set_as(grid_w_idx_offset)
            self._calc_instr_mask_repeat(grid_w_num)
            self._calc_pool_idx_of_grid(self.pool_w_idx, grid_w_num, grid_w_idx_offset, self.cur_roi_sample_w_fp)
            self._calc_bilinear_interpolate_coordinate(self.x, self.x_low, self.x_high, self.lx, self.hx,
                                                       self.cur_roi_sample_w_fp, grid_w_num, grid_w_idx_offset,
                                                       self.cur_roi_grid_w, self.cur_roi_start_x, True)
        # c1 batch loop
        c1_loops = ceil_div(c1_num, op.c1_batch_max)
        with self.tik_inst.for_range(0, c1_loops) as _c1_loop_idx:
            inner_c1_num = self._calc_segment(c1_num, _c1_loop_idx, op.c1_batch_max)
            self.compute_gradient_batch_c1(roi_offset, roi_idx, inner_c1_num,
                                           c1_offset + _c1_loop_idx * op.c1_batch_max,
                                           grid_h_num, grid_w_num)

    # 'pylint: disable=too-many-arguments
    def compute_gradient_batch_c1(self, roi_offset: tik.Scalar, roi_idx: tik.Scalar,
                                  c1_num: tik.Scalar, c1_offset: tik.Scalar,
                                  gird_h_num: tik.Scalar, grid_w_num: tik.Scalar):
        pass

    def _calc_instr_mask_repeat(self, calc_num: tik.Scalar):
        with self.tik_inst.if_scope(calc_num <= 64):
            self.mask.set_as(calc_num)
            self.repeat.set_as(1)
        with self.tik_inst.else_scope():
            self.mask.set_as(64)
            self.repeat.set_as(2)

    def _move_roi_data_to_ub(self, roi_num: tik.Scalar, roi_offset: tik.Scalar) -> None:
        """move roi data from GM to UB"""
        with self.tik_inst.new_stmt_scope():
            rois_data_tmp = self.tik_inst.Tensor("float32", (roi_num, self.op.rois_row_size),
                                                 name="rois_data_tmp", scope=tik.scope_ubuf)
            if tbe_platform.api_check_support("tik.data_move_pad"):
                self.tik_inst.data_move_pad(dst=rois_data_tmp, src=self.op.rois_gm[roi_offset * self.op.rois_row_size],
                                            nburst=1, burst=roi_num * self.op.rois_row_size * 4,
                                            dst_gap=0, src_gap=0)
            else:
                self.tik_inst.data_move(dst=rois_data_tmp, src=self.op.rois_gm[roi_offset * self.op.rois_row_size],
                                        sid=0, nburst=1, burst=ceil_div(roi_num * self.op.rois_row_size * 4, 32),
                                        src_stride=0, dst_stride=0)
            with self.tik_inst.new_stmt_scope(disable_sync=True):
                if self.support_fp32_vextract:
                    with self.tik_inst.if_scope(self.op.rois_row_size != 8):
                        with self.tik_inst.for_range(0, roi_num) as idx:
                            self.roi_in_ub[0, idx].set_as(rois_data_tmp[idx, 0])
                            self.roi_in_ub[1, idx].set_as(rois_data_tmp[idx, 1])
                            self.roi_in_ub[2, idx].set_as(rois_data_tmp[idx, 2])
                            self.roi_in_ub[3, idx].set_as(rois_data_tmp[idx, 3])
                            self.roi_in_ub[4, idx].set_as(rois_data_tmp[idx, 4])
                    with self.tik_inst.else_scope():
                        rep_times = self.tik_inst.Scalar("int64", "rep_times", init_value=ceil_div(roi_num, 16))
                        self.tik_inst.vextract(self.roi_in_ub[0, 0], rois_data_tmp, rep_times, 0)
                        self.tik_inst.vextract(self.roi_in_ub[1, 0], rois_data_tmp, rep_times, 1)
                        self.tik_inst.vextract(self.roi_in_ub[2, 0], rois_data_tmp, rep_times, 2)
                        self.tik_inst.vextract(self.roi_in_ub[3, 0], rois_data_tmp, rep_times, 3)
                        self.tik_inst.vextract(self.roi_in_ub[4, 0], rois_data_tmp, rep_times, 4)
                else:
                    with self.tik_inst.for_range(0, roi_num) as idx:
                        self.roi_in_ub[0, idx].set_as(rois_data_tmp[idx, 0])
                        self.roi_in_ub[1, idx].set_as(rois_data_tmp[idx, 1])
                        self.roi_in_ub[2, idx].set_as(rois_data_tmp[idx, 2])
                        self.roi_in_ub[3, idx].set_as(rois_data_tmp[idx, 3])
                        self.roi_in_ub[4, idx].set_as(rois_data_tmp[idx, 4])

    def _calc_fm_start_end_coordinate(self):
        """calculate feature map _calc_fm_coordinate: x0', y0', x1', y1'"""
        op = self.op
        # 4 instructions in one
        # x0' = x0 * scale
        self.tik_inst.vmuls(64, self.roi_in_ub[1, 0], self.roi_in_ub[1, 0], op.scale, 2 * 4, 1, 1, 8, 8)
        with self.tik_inst.if_scope(op.roi_end_mode > 0):
            with self.tik_inst.if_scope(op.roi_end_mode == 1):
                # x1' = (x1 + 1) * scale
                self.tik_inst.vadds(64, self.roi_in_ub[3, 0], self.roi_in_ub[3, 0], op.scale, 2 * 2, 1, 1, 8, 8)
            with self.tik_inst.else_scope():
                # x0' = x0 * scale - offset
                self.tik_inst.vadds(64, self.roi_in_ub[1, 0], self.roi_in_ub[1, 0], -0.5, 2 * 4, 1, 1, 8, 8)

    def _calc_roi_size(self, roi_num: tik.Scalar):
        """calculate roi width and height"""
        op = self.op
        # 2 instructions in one
        # roi_w = x1' - x0'
        self.tik_inst.vsub(64, self.roi_wh, self.roi_in_ub[3, 0], self.roi_in_ub[1, 0], 2 * 2, 1, 1, 1, 8, 8, 8)
        with self.tik_inst.if_scope(op.roi_end_mode < 2):
            # `roi_w = max(roi_w, 1)`
            self.tik_inst.vmax(64, self.roi_wh, self.roi_wh, self.const_one_fp, 2 * 2, 1, 1, 0, 8, 8, 0)

    def _calc_grid_size(self, roi_num: tik.Scalar) -> None:
        """calculate grid width and height"""
        op = self.op
        # `bin_w = roi_w / pool_w`
        self.tik_inst.vmuls(self.mask, self.grid_w, self.roi_wh[0, 0], op.pool_w_reciprocal, self.repeat, 1, 1, 8, 8)
        self.tik_inst.vmuls(self.mask, self.grid_h, self.roi_wh[1, 0], op.pool_h_reciprocal, self.repeat, 1, 1, 8, 8)
        # count sample counts if not positive
        with self.tik_inst.if_scope(op.samples <= 0):
            # `sample_w = ceil(bin_w)`
            self.tik_inst.vconv(self.mask, 'ceil', self.sample_w, self.grid_w, self.repeat, 1, 1, 8, 8)
            self.tik_inst.vconv(self.mask, 'ceil', self.sample_h, self.grid_h, self.repeat, 1, 1, 8, 8)
            self.tik_inst.vconv(self.mask, "", self.sample_w_fp, self.sample_w, self.repeat, 1, 1, 8, 8)
            self.tik_inst.vconv(self.mask, "", self.sample_h_fp, self.sample_h, self.repeat, 1, 1, 8, 8)
            # `grid_w = bin_w / samples_w`
            self.tik_inst.vdiv(self.mask, self.grid_w, self.grid_w, self.sample_w_fp, self.repeat, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vdiv(self.mask, self.grid_h, self.grid_h, self.sample_h_fp, self.repeat, 1, 1, 1, 8, 8, 8)
        with self.tik_inst.else_scope():
            # `grid_w = bin_w * samples_reciprocal`
            self.tik_inst.vmuls(self.mask, self.grid_w, self.grid_w, op.samples_reciprocal, self.repeat, 1, 1, 8, 8)
            self.tik_inst.vmuls(self.mask, self.grid_h, self.grid_h, op.samples_reciprocal, self.repeat, 1, 1, 8, 8)

    def _convert_fm_idx(self, roi_num: tik.Scalar) -> None:
        """convert indices in ROI from float to int"""
        self.tik_inst.vconv(self.mask, "floor", self.fm_idx, self.roi_in_ub[0, 0], self.repeat, 1, 1, 8, 8)

    def _calc_c1_offset_num(self, core_roi_idx: tik.Scalar):
        c1_start = self.tik_inst.Scalar(dtype="int64", name="c1_start")
        c1_num = self.tik_inst.Scalar(dtype="int64", name="c1_num")
        c1_end = self.tik_inst.Scalar(dtype="int64", name="c1_end")

        with self.tik_inst.if_scope(core_roi_idx == 0):
            c1_start.set_as(self.op.core_nc_offset % self.op.c1)
            c1_end.set_as(c1_start + self.op.core_nc)
            self.tik_inst.scalar_min(c1_end, c1_end, self.op.c1)
        with self.tik_inst.else_scope():
            c1_start.set_as(0)
            c1_end.set_as(self.op.core_nc + (self.op.core_nc_offset % self.op.c1) - (self.op.c1 * core_roi_idx))
            self.tik_inst.scalar_min(c1_end, c1_end, self.op.c1)

        c1_num.set_as(c1_end - c1_start)
        return c1_start, c1_num

    def _calc_pool_idx_of_grid(self, dst_pool_idx: tik.Tensor, grid_num: tik.Scalar,
                               grid_idx_offset: tik.Scalar, samples_fp: tik.Scalar):
        """pool_idx = (grid_idx_offset + grid_idx) // sample_count"""
        # grid_idx_offset + grid_idx where grid_idx is 0..127
        self.tik_inst.vadds(self.mask, self.tmp_fp[0, 0], self.const_0_127_fp, grid_idx_offset, self.repeat, 1, 1, 8, 8)
        self.tik_inst.vector_dup(self.mask, self.tmp_fp[1, 0], samples_fp, self.repeat, 1, 8)
        self.tik_inst.vdiv(self.mask, self.tmp_fp[0, 0],
                           self.tmp_fp[0, 0], self.tmp_fp[1, 0], self.repeat, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vconv(self.mask, "floor", dst_pool_idx, self.tmp_fp[0, 0], self.repeat, 1, 1, 8, 8)

    # 'pylint: disable=too-many-arguments
    def _calc_bilinear_interpolate_coordinate(self, dst_x_fp: tik.Tensor, dst_x_low: tik.Tensor, dst_x_high: tik.Tensor,
                                              dst_lx: tik.Tensor, dst_hx: tik.Tensor, samples_fp: tik.Scalar,
                                              grid_num: tik.Scalar, grid_idx_offset: tik.Scalar,
                                              grid_distance: tik.Scalar, start_coordinate: tik.Scalar,
                                              is_width: bool = True):
        if is_width:
            fm_border_fp = self.const_fm_w_fp
            fm_border_minus_1 = self.const_fm_w_minus_1
            fm_border_minus_1_fp = self.const_fm_w_minus_1_fp
        else:
            fm_border_fp = self.const_fm_h_fp
            fm_border_minus_1 = self.const_fm_h_minus_1
            fm_border_minus_1_fp = self.const_fm_h_minus_1_fp
        """grid_x = start_coordinate + grid_distance * (grid_idx_offset + grid_idx + 0.5) where grid_idx is 0..127"""
        # `grid_idx + grid_idx_offset * 1.0 + 0.5`
        self.tik_inst.vadds(self.mask, self.tmp_fp[0, 0], self.const_0_127_fp, grid_idx_offset * 1.0 + 0.5,
                            self.repeat, 1, 1, 8, 8)
        # `grid_distance * (grid_idx_offset + grid_idx + 0.5)`
        self.tik_inst.vmuls(self.mask, self.tmp_fp[0, 0], self.tmp_fp[0, 0], grid_distance,
                            self.repeat, 1, 1, 8, 8)
        # `start_coordinate + grid_distance * (grid_idx_offset + grid_idx + 0.5)`
        self.tik_inst.vadds(self.mask, self.tmp_fp[0, 0], self.tmp_fp[0, 0], start_coordinate, self.repeat, 1, 1, 8, 8)
        # `if x <= 0: x = 0`
        self.tik_inst.vmax(self.mask, dst_x_fp, self.tmp_fp[0, 0], self.const_zero_fp,
                           self.repeat, 1, 1, 0, 8, 8, 0)
        # `x_low = floor(x)`
        self.tik_inst.vconv(self.mask, "floor", dst_x_low, dst_x_fp, self.repeat, 1, 1, 8, 8)
        # `x_high = x_low + 1`
        self.tik_inst.vadd(self.mask, dst_x_high, dst_x_low, self.const_one_int, self.repeat, 1, 1, 0, 8, 8, 0)
        # `if x_high < x_low: x_high = x_low` int overflow detection
        self.tik_inst.vmax(self.mask, dst_x_high, dst_x_high, dst_x_low, self.repeat, 1, 1, 0, 8, 8, 0)
        # `if x_low >= width - 1: x_low = width - 1;  if x_high >= width - 1: x_high = width - 1`
        self.tik_inst.vmin(self.mask, dst_x_low, dst_x_low, fm_border_minus_1, self.repeat, 1, 1, 0, 8, 8, 0)
        self.tik_inst.vmin(self.mask, dst_x_high, dst_x_high, fm_border_minus_1, self.repeat, 1, 1, 0, 8, 8, 0)
        self.tik_inst.vmin(self.mask, dst_x_fp, dst_x_fp, fm_border_minus_1_fp, self.repeat, 1, 1, 0, 8, 8, 0)
        # `lx = x - x_low`
        self.tik_inst.vconv(self.mask, "", self.tmp_fp[1, 0], dst_x_low, self.repeat, 1, 1, 8, 8)
        self.tik_inst.vsub(self.mask, dst_lx, dst_x_fp, self.tmp_fp[1, 0], self.repeat, 1, 1, 1, 8, 8, 8)
        # `hx = 1 - lx`
        self.tik_inst.vsub(self.mask, dst_hx, self.const_one_fp, dst_lx, self.repeat, 1, 0, 1, 8, 0, 8)
        # `if x < -1.0 or x > width: lx = 0, hx = 0`
        tmp_cmp_mask = self.tik_inst.Tensor("uint16", (32,), name="tmp_cmp_mask", scope=tik.scope_ubuf)
        if tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION) in ("Ascend310", "Ascend910"):
            # mov_tensor_to_cmpmask supports 2*uint64 = 128bits each time
            with self.tik_inst.for_range(0, self.repeat) as _repeat_idx: # each time 64 numbers
                # vcmpv_lt compares 256bytes. (128fp16, 64fp32), dst must be 32 bytes align
                self.tik_inst.vcmpv_lt(tmp_cmp_mask, self.tmp_fp[0, _repeat_idx*64], self.const_minus_1_fp,
                                       1, 1, 0, 8, 0)
                self.tik_inst.vcmpv_gt(tmp_cmp_mask[16], self.tmp_fp[0, _repeat_idx*64], fm_border_fp,
                                       1, 1, 0, 8, 0)
                # vor only support uint16/int16
                self.tik_inst.vor(4, tmp_cmp_mask, tmp_cmp_mask[0], tmp_cmp_mask[16], 1, 1, 1, 1, 2, 2, 2)
                cmpmask = self.tik_inst.mov_tensor_to_cmpmask(tmp_cmp_mask)
                self.tik_inst.vsel(64, 0, dst_lx[_repeat_idx*64], cmpmask, self.const_zero_fp,
                                   dst_lx[_repeat_idx*64], 1, 1, 0, 1, 8, 0, 8)
                self.tik_inst.vsel(64, 0, dst_hx[_repeat_idx*64], cmpmask, self.const_zero_fp,
                                   dst_hx[_repeat_idx*64], 1, 1, 0, 1, 8, 0, 8)
        else:
            self.tik_inst.vcmpv_lt(tmp_cmp_mask, self.tmp_fp, self.const_minus_1_fp, self.repeat, 1, 0, 8, 0)
            self.tik_inst.vcmpv_gt(tmp_cmp_mask[16], self.tmp_fp, fm_border_fp, self.repeat, 1, 0, 8, 0)
            self.tik_inst.vor(8, tmp_cmp_mask, tmp_cmp_mask[0], tmp_cmp_mask[16], 1, 1, 1, 1, 2, 2, 2)
            self.tik_inst.vsel(self.mask, 2, dst_lx, tmp_cmp_mask, self.const_zero_fp, dst_lx,
                               self.repeat, 1, 0, 1, 8, 0, 8)
            self.tik_inst.vsel(self.mask, 2, dst_hx, tmp_cmp_mask, self.const_zero_fp, dst_hx,
                               self.repeat, 1, 0, 1, 8, 0, 8)
        self.tik_inst.vmuls(self.mask, dst_lx, dst_lx, 1.0 / samples_fp, self.repeat, 1, 1, 8, 8)
        self.tik_inst.vmuls(self.mask, dst_hx, dst_hx, 1.0 / samples_fp, self.repeat, 1, 1, 8, 8)

    def _calc_bilinear_interpolate_weight(self, grid_w_num: tik.Scalar, grid_h_idx: tik.Scalar):
        """w1=hy*hx, w2=hy*lx, w3=ly*hx, w4=ly*lx"""
        self._calc_instr_mask_repeat(grid_w_num)
        ly = self.tik_inst.Scalar("float32", "ly", init_value=self.ly[grid_h_idx])
        hy = self.tik_inst.Scalar("float32", "hy", init_value=self.hy[grid_h_idx])
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            self.tik_inst.vmuls(self.mask, self.w1, self.hx, hy, self.repeat, 1, 1, 8, 8)
            self.tik_inst.vmuls(self.mask, self.w2, self.lx, hy, self.repeat, 1, 1, 8, 8)
            self.tik_inst.vmuls(self.mask, self.w3, self.hx, ly, self.repeat, 1, 1, 8, 8)
            self.tik_inst.vmuls(self.mask, self.w4, self.lx, ly, self.repeat, 1, 1, 8, 8)

    # 'pylint: disable=too-many-arguments
    def _bilinear_interpolate_gradient(self, c1_num: tik.Scalar, grid_h_idx: tik.Scalar, grid_w_idx: tik.Scalar,
                                       y_diff_ub: tik.Scalar, y_diff_w_len: tik.Scalar,
                                       x_diff_ub: tik.Scalar, x_diff_w_len: tik.Scalar,
                                       x_diff_2_rows_stride: tik.Scalar):
        op = self.op
        with self.tik_inst.new_stmt_scope():
            x_low = self.tik_inst.Scalar("int32", "x_low", init_value=self.x_low[grid_w_idx])
            x_high = self.tik_inst.Scalar("int32", "x_high", init_value=self.x_high[grid_w_idx])
            y_low = self.tik_inst.Scalar("int32", "y_low", init_value=self.y_low[grid_h_idx])
            y_high = self.tik_inst.Scalar("int32", "y_high", init_value=self.y_high[grid_h_idx])

            w1 = self.tik_inst.Scalar("float32", "w1", init_value=self.w1[grid_w_idx])
            w2 = self.tik_inst.Scalar("float32", "w2", init_value=self.w2[grid_w_idx])
            w3 = self.tik_inst.Scalar("float32", "w3", init_value=self.w3[grid_w_idx])
            w4 = self.tik_inst.Scalar("float32", "w4", init_value=self.w4[grid_w_idx])

            tmp_result = self.tik_inst.Tensor("float32", [4, c1_num, op.c0],
                                              name="tmp_result", scope=tbe_platform.scope_ubuf)
            w2_offset = self.tik_inst.Scalar("int32", "w2_offset", init_value=x_high-x_low)
            w3_offset = self.tik_inst.Scalar("int32", "w3_offset", init_value=(y_high-y_low)*x_diff_2_rows_stride)
            w4_offset = self.tik_inst.Scalar("int32", "w4_offset", init_value=w3_offset+w2_offset)

            mask, repeat = op.c0, c1_num
            dst_rep_stride, src_rep_stride = op.c0_blocks, y_diff_w_len * op.c0_blocks
            with self.tik_inst.new_stmt_scope(disable_sync=True):
                with self.tik_inst.if_scope(src_rep_stride <= Constant.REPEAT_STRIDE_MAX):
                    self.tik_inst.vmuls(mask, tmp_result[0, 0, 0], y_diff_ub, w1,
                                        repeat, 1, 1, dst_rep_stride, src_rep_stride)
                    self.tik_inst.vmuls(mask, tmp_result[1, 0, 0], y_diff_ub, w2,
                                        repeat, 1, 1, dst_rep_stride, src_rep_stride)
                    self.tik_inst.vmuls(mask, tmp_result[2, 0, 0], y_diff_ub, w3,
                                        repeat, 1, 1, dst_rep_stride, src_rep_stride)
                    self.tik_inst.vmuls(mask, tmp_result[3, 0, 0], y_diff_ub, w4,
                                        repeat, 1, 1, dst_rep_stride, src_rep_stride)
                with self.tik_inst.else_scope():
                    with self.tik_inst.for_range(0, repeat) as _rep_idx:
                        src_offset = _rep_idx * y_diff_w_len * op.c0
                        self.tik_inst.vmuls(mask, tmp_result[0, _rep_idx, 0], y_diff_ub[src_offset:], w1,
                                            1, 1, 1, dst_rep_stride, 8)
                        self.tik_inst.vmuls(mask, tmp_result[1, _rep_idx, 0], y_diff_ub[src_offset:], w2,
                                            1, 1, 1, dst_rep_stride, 8)
                        self.tik_inst.vmuls(mask, tmp_result[2, _rep_idx, 0], y_diff_ub[src_offset:], w3,
                                            1, 1, 1, dst_rep_stride, 8)
                        self.tik_inst.vmuls(mask, tmp_result[3, _rep_idx, 0], y_diff_ub[src_offset:], w4,
                                            1, 1, 1, dst_rep_stride, 8)

            dst_rep_stride, src1_rep_stride = x_diff_w_len*op.c0_blocks, op.c0_blocks
            with self.tik_inst.if_scope(dst_rep_stride <= Constant.REPEAT_STRIDE_MAX):
                self.tik_inst.vadd(mask, x_diff_ub[0:],
                                   x_diff_ub[0:], tmp_result[0, 0, 0],
                                   repeat, 1, 1, 1, dst_rep_stride, dst_rep_stride, src1_rep_stride)
                self.tik_inst.vadd(mask, x_diff_ub[w2_offset * op.c0:],
                                   x_diff_ub[w2_offset * op.c0:], tmp_result[1, 0, 0],
                                   repeat, 1, 1, 1, dst_rep_stride, dst_rep_stride, src1_rep_stride)
                self.tik_inst.vadd(mask, x_diff_ub[w3_offset * op.c0:],
                                   x_diff_ub[w3_offset * op.c0:], tmp_result[2, 0, 0],
                                   repeat, 1, 1, 1, dst_rep_stride, dst_rep_stride, src1_rep_stride)
                self.tik_inst.vadd(mask, x_diff_ub[w4_offset * op.c0:],
                                   x_diff_ub[w4_offset * op.c0:], tmp_result[3, 0, 0],
                                   repeat, 1, 1, 1, dst_rep_stride, dst_rep_stride, src1_rep_stride)
            with self.tik_inst.else_scope():
                with self.tik_inst.for_range(0, repeat) as _rep_idx:
                    dst_offset = _rep_idx * x_diff_w_len * op.c0
                    self.tik_inst.vadd(mask, x_diff_ub[dst_offset:], x_diff_ub[dst_offset:],
                                       tmp_result[0, _rep_idx, 0], 1, 1, 1, 1, 8, 8, src1_rep_stride)
                    self.tik_inst.vadd(mask, x_diff_ub[dst_offset + w2_offset * op.c0:],
                                       x_diff_ub[dst_offset + w2_offset * op.c0:], tmp_result[1, _rep_idx, 0],
                                       1, 1, 1, 1, 8, 8, src1_rep_stride)
                    self.tik_inst.vadd(mask, x_diff_ub[dst_offset + w3_offset * op.c0:],
                                       x_diff_ub[dst_offset + w3_offset * op.c0:], tmp_result[2, _rep_idx, 0],
                                       1, 1, 1, 1, 8, 8, src1_rep_stride)
                    self.tik_inst.vadd(mask, x_diff_ub[dst_offset + w4_offset * op.c0:],
                                       x_diff_ub[dst_offset + w4_offset * op.c0:], tmp_result[3, _rep_idx, 0],
                                       1, 1, 1, 1, 8, 8, src1_rep_stride)

    def _clear_ub(self, ub_to_clear: tik.Tensor, clear_len: tik.Scalar):
        """clear ub to zero"""
        do_dtype = ub_to_clear.dtype
        byte_num_one = common_util.get_data_size(do_dtype)
        block_num = 32 // byte_num_one
        vector_num = block_num * 8

        one_loop_offset = vector_num * Constant.INSTR_REPEAT_MAX
        repeat = clear_len // one_loop_offset
        with self.tik_inst.if_scope(repeat > 0):
            with self.tik_inst.for_range(0, repeat) as index:
                tmp_offset = index * one_loop_offset
                self.tik_inst.vec_dup(vector_num, ub_to_clear[tmp_offset], 0, Constant.INSTR_REPEAT_MAX, 8)

        one_loop_repeat = (clear_len % one_loop_offset) // vector_num
        with self.tik_inst.if_scope(one_loop_repeat > 0):
            tmp_offset = repeat * one_loop_offset
            self.tik_inst.vec_dup(vector_num, ub_to_clear[tmp_offset], 0, one_loop_repeat, 8)

        last_num = clear_len % vector_num
        with self.tik_inst.if_scope(last_num > 0):
            tmp_offset = repeat * one_loop_offset + one_loop_repeat * vector_num
            self.tik_inst.vec_dup(last_num, ub_to_clear[tmp_offset], 0, 1, 8)

    def _calc_segment(self, total_seg: tik.Scalar, seg_index: tik.Scalar, seg_len: tik.Scalar):
        left_seg_len = self.tik_inst.Scalar(dtype="int64", name="left_seg_len")
        ret_seg_len = self.tik_inst.Scalar(dtype="int64", name="ret_seg_len")
        seg_gap = self.tik_inst.Scalar(dtype="int64", name="seg_gap", init_value=seg_len)
        left_seg_len.set_as(total_seg - seg_index * seg_len)
        self.tik_inst.scalar_min(ret_seg_len, left_seg_len, seg_gap)

        return ret_seg_len


class MoveOneRowYDiff(ProcessorBase):
    def __init__(self, op_obj: "RoiAlignGrad") -> None:
        super().__init__(op_obj)

    # 'pylint: disable=too-many-arguments
    def compute_gradient_batch_c1(self, roi_offset: tik.Scalar, roi_idx: tik.Scalar,
                                  c1_num: tik.Scalar, c1_offset: tik.Scalar,
                                  gird_h_num: tik.Scalar, grid_w_num: tik.Scalar):
        op = self.op
        with self.tik_inst.new_stmt_scope():
            y_diff_w_min = self.tik_inst.Scalar("int32", "y_diff_w_min", init_value=self.pool_w_idx[0])
            y_diff_w_max = self.tik_inst.Scalar("int32", "y_diff_w_max", init_value=self.pool_w_idx[grid_w_num - 1])
            y_diff_w_len = self.tik_inst.Scalar("int32", "y_diff_w_len", init_value=y_diff_w_max - y_diff_w_min + 1)
            y_diff_cache_h = self.tik_inst.Scalar("int32", "y_diff_cache_h", init_value=-1)
            y_diff_ub = self.tik_inst.Tensor("float32", [c1_num, y_diff_w_len, op.c0],
                                             name="y_diff_ub", scope=tik.scope_ubuf)

            y_low = self.tik_inst.Scalar("int32", "y_low")
            y_high = self.tik_inst.Scalar("int32", "y_high")
            y_diff_h = self.tik_inst.Scalar("int32", "y_diff_h")

            with self.tik_inst.for_range(0, gird_h_num) as _grid_h_idx:
                self._calc_bilinear_interpolate_weight(grid_w_num, _grid_h_idx)

                y_low.set_as(self.y_low[_grid_h_idx])
                y_high.set_as(self.y_high[_grid_h_idx])
                y_diff_h.set_as(self.pool_h_idx[_grid_h_idx])

                # data_move `y_diff_w_len` y_diff together in one row
                with self.tik_inst.if_scope(y_diff_cache_h != y_diff_h):
                    y_diff_gm_offset = ((roi_offset + roi_idx) * op.c1 + c1_offset) * op.pool_h * op.pool_w * op.c0 + \
                        (y_diff_h * op.pool_w + y_diff_w_min) * op.c0
                    c1_batch_move_ydiff_from_gm_one_row(self.tik_inst, op, y_diff_ub,
                                                        y_diff_gm_offset, c1_num, y_diff_w_len)
                    y_diff_cache_h.set_as(y_diff_h)

                self.compute_gradient(c1_num, c1_offset, grid_w_num, _grid_h_idx,
                                      y_diff_ub, y_diff_w_len, y_low, y_high)

    # 'pylint: disable=too-many-arguments
    def compute_gradient(self, c1_num: tik.Scalar, c1_offset: tik.Scalar,
                         grid_w_num: tik.Scalar, grid_h_idx: tik.Scalar,
                         y_diff_ub: tik.Tensor, y_diff_w_num: tik.Scalar,
                         y_low: tik.Scalar, y_high: tik.Scalar):
        def _calc_each_grid(grid_w_idx, _x_diff_ub):
            cur_y_diff_w = self.tik_inst.Scalar("int32", "cur_y_diff_w", init_value=self.pool_w_idx[grid_w_idx])
            self._bilinear_interpolate_gradient(c1_num, grid_h_idx, grid_w_idx,
                                                y_diff_ub[(cur_y_diff_w - y_diff_w_min) * op.c0:],
                                                y_diff_w_num, _x_diff_ub, 4, 2)
            # move x_diff_ub to GM
            x_low = self.tik_inst.Scalar("int32", "x_low", init_value=self.x_low[grid_w_idx])
            x_diff_gm_offset = (self.cur_fm_idx * op.c1 + c1_offset) * op.x_diff_h * op.x_diff_w * op.c0 +\
                (y_low * op.x_diff_w + x_low) * op.c0
            c1_batch_move_xdiff_to_gm_4_points(self.tik_inst, op, _x_diff_ub, x_diff_gm_offset, c1_num,
                                               x_low, y_low)

        op = self.op
        with self.tik_inst.new_stmt_scope():
            y_diff_w_min = self.tik_inst.Scalar("int32", "y_diff_w_min", init_value=self.pool_w_idx[0])
            x_diff_ub = self.tik_inst.Tensor("float32", [2, c1_num, 4, op.c0],
                                             name="x_diff_ub", scope=tbe_platform.scope_ubuf)
            with self.tik_inst.for_range(0, grid_w_num >> 1) as _grid_w_idx:
                self._clear_ub(x_diff_ub, 2 * c1_num * 4 * op.c0)
                _calc_each_grid(_grid_w_idx * 2, x_diff_ub[0:])
                _calc_each_grid(_grid_w_idx * 2 + 1, x_diff_ub[c1_num * 4 * op.c0:])
            with self.tik_inst.if_scope(grid_w_num % 2 == 1):
                self._clear_ub(x_diff_ub, 2 * c1_num * 4 * op.c0)
                _calc_each_grid(grid_w_num - 1, x_diff_ub[0:])


class MoveOneRowYDiffAndSumInUB(MoveOneRowYDiff):
    def __init__(self, op_obj: "RoiAlignGrad") -> None:
        super().__init__(op_obj)

    # 'pylint: disable=too-many-arguments
    def compute_gradient(self, c1_num: tik.Scalar, c1_offset: tik.Scalar,
                         grid_w_num: tik.Scalar, grid_h_idx: tik.Scalar,
                         y_diff_ub: tik.Tensor, y_diff_w_num: tik.Scalar,
                         y_low: tik.Scalar, y_high: tik.Scalar):
        def _calc_each_grid(grid_w_idx):
            x_low = self.tik_inst.Scalar("int32", "x_low", init_value=self.x_low[grid_w_idx])
            y_diff_w = self.tik_inst.Scalar("int32", "y_diff_w", init_value=self.pool_w_idx[grid_w_idx])
            x_diff_ub = x_diff_sum_ub[(x_low - x_low_min) * op.c0:]
            self._bilinear_interpolate_gradient(c1_num, grid_h_idx, grid_w_idx,
                                                y_diff_ub[(y_diff_w - y_diff_w_min) * op.c0:],
                                                y_diff_w_num, x_diff_ub, x_diff_w_len, op.x_diff_w)

        op = self.op
        with self.tik_inst.new_stmt_scope():
            y_diff_w_min = self.tik_inst.Scalar("int32", "y_diff_w_min", init_value=self.pool_w_idx[0])
            x_low_min = self.tik_inst.Scalar("int32", "x_low_min", init_value=self.x_low[0])
            x_high_max = self.tik_inst.Scalar("int32", "x_high_max", init_value=self.x_high[grid_w_num - 1])
            x_diff_w_len = self.tik_inst.Scalar("int32", "x_diff_w_len", init_value=x_high_max - x_low_min + 1)
            with self.tik_inst.if_scope(y_high != y_low):
                x_diff_w_len.set_as(x_high_max - x_low_min + 1 + op.x_diff_w)

            x_diff_sum_ub = self.tik_inst.Tensor("float32", [c1_num, x_diff_w_len, op.c0],
                                                 name="x_diff_sum_ub", scope=tik.scope_ubuf)
            self._clear_ub(x_diff_sum_ub, c1_num * x_diff_w_len * op.c0)

            with self.tik_inst.for_range(0, grid_w_num >> 1) as _grid_w_idx:
                _calc_each_grid(_grid_w_idx * 2)
                _calc_each_grid(_grid_w_idx * 2 + 1)
            with self.tik_inst.if_scope(grid_w_num % 2 == 1):
                _calc_each_grid(grid_w_num - 1)

            # move x_diff_sum_ub to GM
            x_diff_gm_offset = (self.cur_fm_idx * op.c1 + c1_offset) * op.x_diff_h * op.x_diff_w * op.c0 +\
                (y_low * op.x_diff_w + x_low_min) * op.c0
            c1_batch_move_xdiff_to_gm_one_row(self.tik_inst, op, x_diff_sum_ub, x_diff_gm_offset,
                                              c1_num, x_diff_w_len)


class Default(ProcessorBase):
    """ Default process class """
    def __init__(self, op_obj: "RoiAlignGrad") -> None:
        super().__init__(op_obj)

    # 'pylint: disable=too-many-arguments
    def compute_gradient_batch_c1(self, roi_offset: tik.Scalar, roi_idx: tik.Scalar,
                                  c1_num: tik.Scalar, c1_offset: tik.Scalar,
                                  gird_h_num: tik.Scalar, grid_w_num: tik.Scalar):
        y_low = self.tik_inst.Scalar("int32", "y_low")
        y_diff_h = self.tik_inst.Scalar("int32", "y_diff_h")

        with self.tik_inst.for_range(0, gird_h_num) as _grid_h_idx:
            self._calc_bilinear_interpolate_weight(grid_w_num, _grid_h_idx)

            y_low.set_as(self.y_low[_grid_h_idx])
            y_diff_h.set_as(self.pool_h_idx[_grid_h_idx])

            self.compute_gradient(roi_offset, roi_idx, c1_num, c1_offset, y_diff_h, grid_w_num, _grid_h_idx,
                                  y_low)

    # 'pylint: disable=too-many-arguments
    def compute_gradient(self, roi_offset: tik.Scalar, roi_idx: tik.Scalar,
                         c1_num: tik.Scalar, c1_offset: tik.Scalar,
                         y_diff_h: tik.Scalar, grid_w_num: tik.Scalar, grid_h_idx: tik.Scalar,
                         y_low: tik.Scalar):
        def _calc_each_grid(grid_w_idx, _x_diff_ub, _y_diff_ub):
            y_diff_w = self.tik_inst.Scalar("int32", "y_diff_w", init_value=self.pool_w_idx[grid_w_idx])
            y_diff_gm_offset = ((roi_offset + roi_idx) * op.c1 + c1_offset) * op.pool_h * op.pool_w * op.c0 +\
                (y_diff_h * op.pool_w + y_diff_w) * op.c0
            c1_batch_move_ydiff_from_gm_one_row(self.tik_inst, op, _y_diff_ub, y_diff_gm_offset, c1_num, 1)

            self._bilinear_interpolate_gradient(c1_num, grid_h_idx, grid_w_idx, _y_diff_ub, 1, _x_diff_ub, 4, 2)

            # move x_diff_ub to GM
            x_low = self.tik_inst.Scalar("int32", "x_low", init_value=self.x_low[grid_w_idx])
            x_diff_gm_offset = (self.cur_fm_idx * op.c1 + c1_offset) * op.x_diff_h * op.x_diff_w * op.c0 +\
                (y_low * op.x_diff_w + x_low) * op.c0
            c1_batch_move_xdiff_to_gm_4_points(self.tik_inst, op, _x_diff_ub, x_diff_gm_offset, c1_num,
                                               x_low, y_low)

        op = self.op
        with self.tik_inst.new_stmt_scope():
            x_diff_ub = self.tik_inst.Tensor("float32", [2, c1_num, 4, op.c0],
                                             name="x_diff_ub", scope=tbe_platform.scope_ubuf)
            y_diff_ub = self.tik_inst.Tensor("float32", [2, c1_num, 1, op.c0],
                                             name="y_diff_ub", scope=tik.scope_ubuf)
            with self.tik_inst.for_range(0, grid_w_num >> 1) as _grid_w_idx:
                self._clear_ub(x_diff_ub, 2 * c1_num * 4 * op.c0)
                _calc_each_grid(_grid_w_idx * 2, x_diff_ub[0:], y_diff_ub[0:])
                _calc_each_grid(_grid_w_idx * 2 + 1, x_diff_ub[c1_num * 4 * op.c0:], y_diff_ub[c1_num * 1 * op.c0:])
            with self.tik_inst.if_scope(grid_w_num % 2 == 1):
                self._clear_ub(x_diff_ub, 2 * c1_num * 4 * op.c0)
                _calc_each_grid(grid_w_num - 1, x_diff_ub[0:], y_diff_ub[0:])


# 'pylint: disable=too-many-arguments
def c1_batch_move_ydiff_from_gm_one_row(tik_inst: tik.Tik, op: "RoiAlignGrad", dst_ub: tik.Tensor,
                                        src_gm_offset: tik.Scalar, c1_num: tik.Scalar, w_num: tik.Scalar) -> None:
    """ y_diff GM -> L1/UB one row"""
    with tik_inst.new_stmt_scope(disable_sync=True):
        burst_num = c1_num
        burst_len = w_num * op.c0_blocks
        burst_src_gap = op.pool_h * op.pool_w * op.c0_blocks - burst_len
        with tik_inst.if_scope(burst_src_gap <= Constant.BURST_STRIDE_MAX):
            tik_inst.data_move(dst_ub[0:], op.y_diff_gm[src_gm_offset:], 0, burst_num, burst_len, burst_src_gap, 0)
        with tik_inst.else_scope():
            with tik_inst.for_range(0, burst_num) as _c1_idx:
                src_offset = src_gm_offset + _c1_idx * op.pool_h * op.pool_w * op.c0
                dst_offset = _c1_idx * w_num * op.c0
                tik_inst.data_move(dst_ub[dst_offset:], op.y_diff_gm[src_offset:], 0, 1, burst_len, 0, 0)


# 'pylint: disable=too-many-arguments
def c1_batch_move_xdiff_to_gm_one_row(tik_inst: tik.Tik, op: "RoiAlignGrad", src_ub: tik.Tensor,
                                      dst_gm_offset: tik.Scalar, c1_num: tik.Scalar, w_num: tik.Scalar) -> None:
    """ x_diff UB -> GM one row """
    with tik_inst.new_stmt_scope(disable_sync=True):
        burst_num = c1_num
        burst_len = w_num * op.c0_blocks
        burst_dst_gap = op.x_diff_h * op.x_diff_w * op.c0_blocks - burst_len
        with tik_inst.if_scope(burst_dst_gap <= Constant.BURST_STRIDE_MAX):
            tik_inst.data_move(op.x_diff_gm[dst_gm_offset:], src_ub, 0, burst_num, burst_len, 0, burst_dst_gap)
        with tik_inst.else_scope():
            with tik_inst.for_range(0, burst_num) as _c1_idx:
                dst_offset = dst_gm_offset + _c1_idx * op.x_diff_h * op.x_diff_w * op.c0
                src_offset = _c1_idx * w_num * op.c0
                tik_inst.data_move(op.x_diff_gm[dst_offset:], src_ub[src_offset:], 0, 1, burst_len, 0, 0)


# 'pylint: disable=too-many-arguments
def c1_batch_move_xdiff_to_gm_4_points(tik_inst: tik.Tik, op: "RoiAlignGrad", src_ub: tik.Tensor,
                                       dst_gm_offset: tik.Scalar, c1_num: tik.Scalar,
                                       x_low: tik.Scalar, y_low: tik.Scalar) -> None:
    """ x_diff UB -> GM 4 points only """
    def _move_one_row(dst_offset, src_offset):
        with tik_inst.if_scope(burst_dst_gap <= Constant.BURST_STRIDE_MAX):
            tik_inst.data_move(op.x_diff_gm[dst_offset:], src_ub[src_offset:],
                               0, burst_num, burst_len, burst_src_gap, burst_dst_gap)
        with tik_inst.else_scope():
            with tik_inst.for_range(0, burst_num) as _c1_idx:
                _dst_offset = dst_offset + _c1_idx * op.x_diff_h * op.x_diff_w * op.c0
                _src_offset = src_offset + _c1_idx * 4 * op.c0
                tik_inst.data_move(op.x_diff_gm[_dst_offset:], src_ub[_src_offset:], 0, 1, burst_len, 0, 0)

    with tik_inst.new_stmt_scope(disable_sync=True):
        burst_num = c1_num
        burst_len = tik_inst.Scalar("int32", "burst_len", init_value=2 * op.c0_blocks)
        with tik_inst.if_scope(x_low + 1 >= op.x_diff_w):
            burst_len.set_as(1 * op.c0_blocks)
        burst_dst_gap = op.x_diff_h * op.x_diff_w * op.c0_blocks - burst_len
        burst_src_gap = 4 * op.c0_blocks - burst_len
        _move_one_row(dst_gm_offset, 0)
        with tik_inst.if_scope(y_low + 1 < op.x_diff_h):
            _move_one_row(dst_gm_offset + 1 * op.x_diff_w * op.c0, 2 * op.c0)


# 'pylint: disable=too-many-instance-attributes,too-many-arguments
class RoiAlignGrad(OpBase):
    """RoiAlignGrad basic information"""

    # processor dictionary for different tiling keys
    _processors = {
        11100: lambda op: MoveOneRowYDiffAndSumInUB(op),
        11000: lambda op: MoveOneRowYDiff(op),
        10000: lambda op: Default(op)  # default
    }

    def __init__(self, y_diff, rois, rois_n, x_diff,
                 pooled_width: int, pooled_height: int,
                 spatial_scale: float, sample_num: int, roi_end_mode: int,
                 kernel_name: str):
        OpBase.__init__(self, kernel_name)
        # reset ub available size for temporary variables
        self.ub_size_bytes = self.ub_size_bytes - Constant.RESERVED_UB_SIZE

        # init gm address
        tiling_dict = {"dtype": "int64", "shape": (Constant.TILING_ARG_NUM,)}
        inputs = [y_diff, rois]
        if rois_n:
            inputs.append(rois_n)
        x_diff.update({"is_atomic_add": True})
        self.op_init_gm(inputs, [x_diff], tiling_info=tiling_dict)
        self.y_diff_gm, self.rois_gm = self.input_gm_list[0], self.input_gm_list[1]
        self.rois_n_gm = self.input_gm_list[2] if rois_n else None
        self.x_diff_gm = self.output_gm_list[0]

        # C0 size
        self.c0 = 16
        # data count per block
        self.block_num = 8
        # maximum data count can save in ub
        self.ub_max_num = self.ub_size_bytes // common_util.get_data_size(y_diff.get("dtype").lower())
        # applied blocks per C0
        self.c0_blocks = self.c0 // self.block_num

        # init tiling scalar
        self.tiling_key = self.tik_instance.Scalar("int64", "tiling_key")
        self.rois_n = self.tik_instance.Scalar("int64", "rois_n")
        self.rois_row_size = self.tik_instance.Scalar("int64", "rois_row_size")
        self.x_diff_n = self.tik_instance.Scalar("int64", "x_diff_n")
        self.c1 = self.tik_instance.Scalar("int64", "c1")
        self.x_diff_h = self.tik_instance.Scalar("int64", "x_diff_h")
        self.x_diff_w = self.tik_instance.Scalar("int64", "x_diff_w")
        self.c1_batch_max = self.tik_instance.Scalar("int64", name="c1_batch_max")
        self.pool_w = self.tik_instance.Scalar("int32", name="pool_w")
        self.pool_h = self.tik_instance.Scalar("int32", name="pool_h")
        self.samples = self.tik_instance.Scalar("int32", name="samples")
        self.roi_end_mode = self.tik_instance.Scalar("int32", name="roi_end_mode")
        self.scale = self.tik_instance.Scalar("float32", name="scale")
        self.pool_w_reciprocal = self.tik_instance.Scalar("float32", name="pool_w_reciprocal")
        self.pool_h_reciprocal = self.tik_instance.Scalar("float32", name="pool_h_reciprocal")
        self.samples_reciprocal = self.tik_instance.Scalar("float32", name="samples_reciprocal")

        # nc1 process len current core
        self.core_nc = self.tik_instance.Scalar("int64", "core_nc")
        # nc1 offset current core
        self.core_nc_offset = self.tik_instance.Scalar("int64", "core_nc_offset")
        # roi count current core
        self.core_rois = self.tik_instance.Scalar("int64", "core_rois")
        # roi offset current core
        self.core_roi_offset = self.tik_instance.Scalar("int64", "core_roi_offset")
        pass

    def roi_align_grad_compute(self):
        """
        roi_align_grad_operator
        """
        # register compute base on tiling_key
        register_func = partial(self.regist_compute, tiling_func=self._functions)
        for k in self._processors:
            register_func(k, key=k)

        self.tik_instance.set_atomic_add(1)
        self.op_run_compute()  # run all registered compute base tiling key
        self.tik_instance.set_atomic_add(0)

        # Build CCE
        tbe_context.get_context().add_compile_info("vars", {"core_num": self.core_nums,
                                                            "ub_size": self.ub_size_bytes})

        # set it as false. it can only be True in DSL.
        self.opt_config.update({"out_of_bound_sync_check": False})
        self.op_build_cce()

        return self.tik_instance

    def tiling_args(self):
        """
        tiling_args
        tiling key  tiling_key
        input info  tiling_batch, tiling_c1, input_height, input_width
        output info output_height, output_width
        cut info    tiling_bc1_, tiling_height_cut_num, tiling_width_cut_num
        """
        with self.tik_instance.new_stmt_scope():
            tiling_ub = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                                 name="tiling_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1,
                                        ceil_div(Constant.TILING_ARG_NUM,
                                                 32 // common_util.get_data_size(tiling_ub.dtype)),
                                        0, 0)
            self.tiling_key.set_as(tiling_ub[0])
            self.running_core_num.set_as(tiling_ub[1])
            self.rois_n.set_as(tiling_ub[2])
            self.rois_row_size.set_as(tiling_ub[3])
            self.x_diff_n.set_as(tiling_ub[4])
            self.c1.set_as(tiling_ub[5])
            self.x_diff_h.set_as(tiling_ub[6])
            self.x_diff_w.set_as(tiling_ub[7])
            self.c1_batch_max.set_as(tiling_ub[8])

            tiling_ub_int32 = tiling_ub.reinterpret_cast_to("int32")
            self.pool_w.set_as(tiling_ub_int32[18])
            self.pool_h.set_as(tiling_ub_int32[19])
            self.samples.set_as(tiling_ub_int32[20])
            self.roi_end_mode.set_as(tiling_ub_int32[21])
            tiling_ub_fp32 = tiling_ub.reinterpret_cast_to("float32")
            self.scale.set_as(tiling_ub_fp32[22])
            self.pool_w_reciprocal.set_as(tiling_ub_fp32[23])
            self.pool_h_reciprocal.set_as(tiling_ub_fp32[24])
            self.samples_reciprocal.set_as(tiling_ub_fp32[25])

    def core_scedule_args(self, core_idx):
        """prepare some variables as per core_id"""
        nc_per_core = (self.rois_n * self.c1) // self.running_core_num
        nc_per_core_tail = (self.rois_n * self.c1) % self.running_core_num
        with self.tik_instance.if_scope(core_idx == 0):
            self.core_nc.set_as(nc_per_core + nc_per_core_tail)
            self.core_nc_offset.set_as(0)
        with self.tik_instance.else_scope():
            self.core_nc.set_as(nc_per_core)
            self.core_nc_offset.set_as(core_idx * nc_per_core + nc_per_core_tail)

        self.core_roi_offset.set_as(self.core_nc_offset // self.c1)
        core_roi_end = (self.core_nc_offset + self.core_nc - 1) // self.c1
        self.core_rois.set_as(core_roi_end - self.core_roi_offset + 1)

    def _functions(self, key: int):
        """invoke each tiling functions

        Parameters
        ----------
        key : int
            tiling key
        """
        processor = self._processors.get(key)
        processor(self).run()


# 'pylint: disable=too-many-arguments
@register_operator("ROIAlignGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_LIST_INT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def roi_align_grad(y_diff, rois, rois_n,  # inputs
                   x_diff,  # outputs
                   xdiff_shape: List[int],
                   pooled_width: int, pooled_height: int,
                   spatial_scale: float, sample_num: int = 2, roi_end_mode: int = 1,  # attrs
                   kernel_name: str = "roi_align_grad"):
    """
    calculating roi_align_grad,
    the type of input_data is "float32"

    Parameters
    ----------
    y_diff: dict
        dict with keys(shape and dtype) of y_diff
    rois: dict
        dict with keys(shape and dtype) of rois
    rois_n: dict
        dict with keys(shape and dtype) of rois_n
    x_diff: dict
        dict with keys(shape and dtype) of x_diff
    xdiff_shape: list
        list xdiff_shape
    pooled_width: int
        pooled_width
    pooled_height: int
        pooled_height
    spatial_scale: float
        spatial_scale
    sample_num: int
        sample_num
    roi_end_mode: int
        roi_end_mode
    kernel_name: str
        kernel name

    Returns
    -------
    tik_instance: tik_instance
    """
    input_list = [y_diff, rois]
    for input_data in input_list:
        input_dtype = input_data.get("dtype").lower()
        check_list = ("float32",)
        para_check.check_dtype(input_dtype, check_list, param_name="y_diff")

    obj = RoiAlignGrad(y_diff, rois, rois_n, x_diff,
                       pooled_width, pooled_height, spatial_scale, sample_num, roi_end_mode,
                       kernel_name)

    return obj.roi_align_grad_compute()
