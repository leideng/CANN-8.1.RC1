#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2020 Huawei Technologies Co., Ltd
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
max_pool3d_with_argmax
"""
import math
import functools
from functools import partial
from abc import ABCMeta
from abc import abstractmethod
from enum import Enum
from enum import unique

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl import constant_util as const


@unique
class L1MoveStrategy(Enum):
    """
    L1 move strategy Enum: EMPTY, NO_NEED_TO_CUT
    """
    EMPTY = 0
    NO_NEED_TO_CUT = 1


@unique
class UbKernelStrategy(Enum):
    """
    UB kernel strategy Enum: EMPTY, WHOLE_KERNEL, KH_AND_KW, ONLY_KW
    """
    EMPTY = 0
    WHOLE_KERNEL = 1
    CUT_H = 2
    CUT_H_AND_W = 3


# 'pylint: disable=too-few-public-methods
class Core:
    """
    Calculate n, c1, d_step based on core_index
    """

    def __init__(self, core_ind, c1, d_step):
        self.ind_n = core_ind // (c1 * d_step)
        n_rest = core_ind % (c1 * d_step)
        self.ind_d = n_rest // c1
        self.ind_c1 = n_rest % c1


# 'pylint: disable=too-few-public-methods
class MaxPool3DWithArgmax(metaclass=ABCMeta):
    """
    MaxPool3DWithArgmax: compute definition of max_pool3d_with_argmax
    """

    def __init__(self, x, kernel_size, stride, kernel_name):
        self.inst = tik.Tik()

        self.kernel_name = kernel_name
        self.input_shape = x.get('shape')
        self.input_dtype = x.get('dtype')
        self.block_size = 32
        self.vector_size_bytes = 256
        self.each_elem_bytes = 2
        self.each_uint16_bytes = 2
        self.each_fp16_bytes = 2
        self.ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.ub_max_elem = self.ub_size_bytes // self.each_elem_bytes
        self.vector_size_elem = self.vector_size_bytes // self.each_elem_bytes
        self.each_block_elem = self.block_size // self.each_elem_bytes
        self.out_mask_type = 'uint16'
        self.mask4compute = 128
        self.upper_rep_limit = 255

        # notice: though we can aligned the shape of bitmask_ub, but each repeat of vcmpv has 128bits(16B) result.
        # we have to aligned the address to 32B.
        self.upper_cmp_rep = 254

        self.n, self.input_d, self.c1, self.input_h, self.input_w, self.c0 = x.get('shape')
        _, _, self.k_d, self.k_h, self.k_w = kernel_size
        _, _, self.stride_d, self.stride_h, self.stride_w = stride

        self.d_step = (self.input_d - self.k_d) // self.stride_d + 1
        self.h_step = (self.input_h - self.k_h) // self.stride_h + 1
        self.w_step = (self.input_w - self.k_w) // self.stride_w + 1
        self.k_elem = self.k_h * self.k_w * self.k_d
        self.output_shape = (self.n, self.d_step, self.c1, self.h_step, self.w_step, self.c0)

        # first, we have to aligned the data points to uint16, not yet aligned the uint16 to block_size(32B)
        # this is approximately equal to w_step * h_step, aligned to 16.(compliment by zero)
        self.aligned_bm_line = math.ceil(self.w_step * self.h_step / 16) * 16
        self.dirty_l1_pad = math.ceil((self.aligned_bm_line - self.h_step * self.w_step) / self.w_step) * self.stride_h

        self.l1_strategy = L1MoveStrategy.EMPTY
        self.ub_kernel_stg = UbKernelStrategy.EMPTY

        # init some gm Tensor
        self.input_gm_tensor = self.inst.Tensor(dtype=self.input_dtype,
                                                shape=self.input_shape,
                                                name='input_tensor',
                                                scope=tik.scope_gm)
        self.output_gm_tensor = self.inst.Tensor(dtype=self.input_dtype,
                                                 shape=self.output_shape,
                                                 name='output_tensor',
                                                 scope=tik.scope_gm)
        self.output_gm_tensor = self.output_gm_tensor.reshape((self.n, self.d_step, self.c1,
                                                               self.h_step * self.w_step * self.c0))

        # the output bitmask
        self.final_bitmask_gm_tensor = self.inst.Tensor(
            dtype=self.out_mask_type,
            # the last two dimension of the shape aim to align to 32B.
            shape=[self.n, self.d_step, self.c1 * self.k_elem, self.aligned_bm_line // 16 * 16],
            name='final_bitmask',
            scope=tik.scope_gm
        )

        # encapsulate the data_move function when the data is continue.
        self._move_ctn = partial(self.inst.data_move,
                                 sid=0,
                                 nburst=1,
                                 src_stride=0,
                                 dst_stride=0)

        # encapsulate the vnot function when the data is continue.
        self._vnot_ctn = partial(self.inst.vnot,
                                 src_blk_stride=1,
                                 dst_blk_stride=1,
                                 src_rep_stride=8,
                                 dst_rep_stride=8,
                                 repeat_times=1,
                                 mask=128)

        # encapsulate the vcmpv function when the data is continue.
        self._cmp_ctn = partial(self.inst.vcmpv_eq,
                                src0_blk_stride=1,
                                src1_blk_stride=1,
                                src0_rep_stride=8,
                                src1_rep_stride=8)

    # encapsulate the two-number compute function.
    @staticmethod
    # 'pylint: disable=too-many-arguments
    def _compute_two_ctn(method, dst, src0, src1, rep_times=1, mask=128):
        method(mask=mask,
               dst=dst,
               src0=src0,
               src1=src1,
               repeat_times=rep_times,
               dst_blk_stride=1,
               src0_blk_stride=1,
               src1_blk_stride=1,
               dst_rep_stride=8,
               src0_rep_stride=8,
               src1_rep_stride=8)

    def _vector_dup(self, src, src_start, shape, dup_reg):
        vector_fp16_size = 128
        max_vector_repeat_time = 255

        ele_num = functools.reduce(lambda x, y: x * y, shape)
        total_repeat_time = ele_num // vector_fp16_size
        remain_ele = ele_num % vector_fp16_size
        mask_value = vector_fp16_size
        repeat_max_time = total_repeat_time // max_vector_repeat_time
        remain_repeat_time = total_repeat_time % max_vector_repeat_time

        with self.inst.for_range(0, repeat_max_time) as loop:
            self.inst.vector_dup(mask_value, src[src_start + loop * max_vector_repeat_time * mask_value],
                                 dup_reg, max_vector_repeat_time, 1, 8)

        if remain_repeat_time > 0:
            self.inst.vector_dup(mask_value, src[src_start + repeat_max_time * max_vector_repeat_time * mask_value],
                                 dup_reg, remain_repeat_time, 1, 8)

        if remain_ele > 0:
            self.inst.vector_dup(remain_ele, src[src_start + repeat_max_time * max_vector_repeat_time * mask_value +
                                                 remain_repeat_time * mask_value], dup_reg, 1, 1, 8)

    def _img2col(self, aux_l1, big_matrix_ub, l1_begin_pos, rep_times):
        if self.ub_kernel_stg is UbKernelStrategy.WHOLE_KERNEL:
            with self.inst.for_range(0, self.k_d) as tmp_kd:
                # we adopt the scheme that repeat_mode = 1
                with self.inst.for_range(0, self.k_h) as tmp_kh:
                    with self.inst.for_range(0, self.k_w) as tmp_kw:
                        self.inst.load3dv1(
                            big_matrix_ub[tmp_kd * self.k_h * self.k_w + tmp_kh * self.k_w + tmp_kw, 0],
                            aux_l1,
                            (0, 0, 0, 0),
                            aux_l1.shape[2],
                            aux_l1.shape[3],
                            tmp_kd,
                            tmp_kw,
                            tmp_kh,
                            left_top_w=l1_begin_pos[1],
                            left_top_h=l1_begin_pos[0],
                            stride_w=self.stride_w,
                            stride_h=self.stride_h,
                            filter_w=self.k_w,
                            filter_h=self.k_h,
                            dilation_filter_w=1,
                            dilation_filter_h=1,
                            jump_offset=1,
                            repeat_mode=1,
                            repeat_time=rep_times
                        )

    # 'pylint: disable=too-many-arguments
    def _calc_maxline(self, max_line_ub, big_matrix_ub, line_blk,
                      line_rep_loop, line_rep_tail, super_line_loop, super_line_tail):
        self._move_ctn(dst=max_line_ub,
                       src=big_matrix_ub,
                       burst=line_blk)
        with self.inst.for_range(1, self.k_elem) as line_ind:
            if line_rep_loop != 0:
                if super_line_loop != 0:
                    with self.inst.for_range(0, super_line_loop) as super_loop_ind:
                        self._compute_two_ctn(
                            method=self.inst.vmax,
                            dst=max_line_ub[super_loop_ind * self.upper_rep_limit * self.mask4compute],
                            src0=max_line_ub[super_loop_ind * self.upper_rep_limit * self.mask4compute],
                            src1=big_matrix_ub[line_ind, super_loop_ind * self.upper_rep_limit * self.mask4compute],
                            rep_times=self.upper_rep_limit
                        )
                if super_line_tail != 0:
                    self._compute_two_ctn(
                        method=self.inst.vmax,
                        dst=max_line_ub[super_line_loop * self.upper_rep_limit * self.mask4compute],
                        src0=max_line_ub[super_line_loop * self.upper_rep_limit * self.mask4compute],
                        src1=big_matrix_ub[line_ind, super_line_loop * self.upper_rep_limit * self.mask4compute],
                        rep_times=super_line_tail
                    )
            if line_rep_tail != 0:
                self._compute_two_ctn(
                    method=self.inst.vmax,
                    dst=max_line_ub[line_rep_loop * self.mask4compute],
                    src0=max_line_ub[line_rep_loop * self.mask4compute],
                    src1=big_matrix_ub[line_ind, line_rep_loop * self.mask4compute],
                    rep_times= const.REPEAT_TIME_ONCE,
                    mask=line_rep_tail
                )

    # 'pylint: disable=too-many-arguments
    def _calc_bitmask(self, bitmask_ub, big_matrix_ub, max_line_ub, super_line_loop, super_line_tail):
        with self.inst.for_range(0, self.k_elem) as line_ind:
            if super_line_loop != 0:
                with self.inst.for_range(0, super_line_loop) as super_loop_ind:
                    self._cmp_ctn(
                        dst=bitmask_ub[line_ind, super_loop_ind * self.upper_cmp_rep * self.mask4compute // 16],
                        src0=big_matrix_ub[line_ind, super_loop_ind * self.upper_cmp_rep * self.mask4compute],
                        src1=max_line_ub[super_loop_ind * self.upper_cmp_rep * self.mask4compute],
                        repeat_times=self.upper_cmp_rep
                    )
            if super_line_tail != 0:
                self._cmp_ctn(
                    dst=bitmask_ub[line_ind, super_line_loop * self.upper_cmp_rep * self.mask4compute // 16],
                    src0=big_matrix_ub[line_ind, super_line_loop * self.upper_cmp_rep * self.mask4compute],
                    src1=max_line_ub[super_line_loop * self.upper_cmp_rep * self.mask4compute],
                    repeat_times=super_line_tail
                )

    # 'pylint: disable=too-many-arguments
    def _deduplicate_bitmask(self, mask_or_ub, mask_not_ub, bitmask_ub, data_blk, bm_loop, bm_tail):
        # first, init the mask_not and mask_or
        if bm_loop != 0:
            self._vnot_ctn(dst=mask_not_ub,
                           src=bitmask_ub,
                           repeat_times=bm_loop)
        if bm_tail != 0:
            self._vnot_ctn(dst=mask_not_ub[bm_loop * self.mask4compute],
                           src=bitmask_ub[0, bm_loop * self.mask4compute],
                           mask=bm_tail)
        self._move_ctn(dst=mask_or_ub,
                       src=bitmask_ub,
                       burst=data_blk)
        with self.inst.for_range(1, self.k_elem) as line_ind:
            if bm_loop != 0:
                self._compute_two_ctn(method=self.inst.vor,
                                      dst=mask_or_ub,
                                      src0=mask_or_ub,
                                      src1=bitmask_ub[line_ind, 0],
                                      rep_times=bm_loop)
                self._compute_two_ctn(method=self.inst.vand,
                                      dst=bitmask_ub[line_ind, 0],
                                      src0=bitmask_ub[line_ind, 0],
                                      src1=mask_not_ub,
                                      rep_times=bm_loop)
                self._vnot_ctn(dst=mask_not_ub,
                               src=mask_or_ub,
                               repeat_times=bm_loop)
            if bm_tail != 0:
                self._compute_two_ctn(method=self.inst.vor,
                                      dst=mask_or_ub[bm_loop * self.mask4compute],
                                      src0=mask_or_ub[bm_loop * self.mask4compute],
                                      src1=bitmask_ub[line_ind, bm_loop * self.mask4compute],
                                      mask=bm_tail)
                self._compute_two_ctn(method=self.inst.vand,
                                      dst=bitmask_ub[line_ind, bm_loop * self.mask4compute],
                                      src0=bitmask_ub[line_ind, bm_loop * self.mask4compute],
                                      src1=mask_not_ub[bm_loop * self.mask4compute],
                                      mask=bm_tail)
                self._vnot_ctn(dst=mask_not_ub[bm_loop * self.mask4compute],
                               src=mask_or_ub[bm_loop * self.mask4compute],
                               mask=bm_tail)

    @abstractmethod
    def run(self):
        """
        run function
        """
        pass


# 'pylint: disable=too-few-public-methods
class MaxPool3DWithArgmaxWholeKernel(MaxPool3DWithArgmax):
    """
    MaxPool3DWithArgmaxWholeKernel: inherited from MaxPool3DWithArgmax
    """

    def __init__(self, x, kernel_size, stride, kernel_name):
        super().__init__(x, kernel_size, stride, kernel_name)
        # init some value for cut-ub process
        # all the tensors that occupy the ub:(X represents the size of bitmask line)
        # -------------------------------------------------------------------------------------------------------
        # name                        |  size (regardless of ub-memory)   |  hypothetical variable (NOT accuracy)
        # -------------------------------------------------------------------------------------------------------
        # bigmat_ub_T                 |  k_elem * w_step * h_step * C0    |  X * 16 * k_elem
        # bitmask_ub_T                |  k_elem * aligned_bitmask_line    |  X * k_elem
        # maxline_ub_T                |  w_step * h_step * C0             |  X * 16
        # maskor_ub_T + masknot_ub_T  |  aligned_bitmask_line * 2         |  X * 2
        # -------------------------------------------------------------------------------------------------------
        # the formula of ub tiling is: 17 * k_elem * X + 18 * X = ub_size
        self.cut_ub_val = self.ub_max_elem // (self.k_elem * 17 + 18) // 16 * 16
        self.ub_line = self.cut_ub_val * 16

        self.cut_ub_loop = self.h_step * self.w_step // self.cut_ub_val
        self.cut_ub_tail_aligned = self.aligned_bm_line - self.cut_ub_loop * self.cut_ub_val
        self.cut_ub_tail_ori = self.h_step * self.w_step % self.cut_ub_val

        self.l1_strategy = L1MoveStrategy.NO_NEED_TO_CUT
        self.ub_kernel_stg = UbKernelStrategy.WHOLE_KERNEL

        # Notice: 1. line_rep_loop probably more than 255.
        #         2. ub_line has already aligned to 256, so there is no need to consider line_rep_tail.
        self.line_rep_loop = self.ub_line // self.vector_size_elem
        self.super_line_loop = self.line_rep_loop // self.upper_rep_limit
        self.super_line_tail = self.line_rep_loop % self.upper_rep_limit
        self.bm_line_loop = self.cut_ub_val // self.vector_size_elem
        self.bm_line_tail = self.cut_ub_val % self.vector_size_elem

        self.super_line_loop_cmp = self.line_rep_loop // self.upper_cmp_rep
        self.super_line_tail_cmp = self.line_rep_loop % self.upper_cmp_rep

        # compute some values for ub-cut-tail process
        # Notice: cut_ub_tail_aligned has already aligned to 16, so there is no need to consider ubtail_ub_tail
        self.ubtail_ub_loop = self.cut_ub_tail_aligned * self.c0 // self.vector_size_elem
        self.ubtail_bm_loop = self.cut_ub_tail_aligned // self.vector_size_elem
        self.ubtail_bm_tail = self.cut_ub_tail_aligned % self.vector_size_elem
        self.ubtail_super_loop = self.ubtail_ub_loop // self.upper_rep_limit
        self.ubtail_super_tail = self.ubtail_ub_loop % self.upper_rep_limit

        self.ubtail_super_loop_cmp = self.ubtail_ub_loop // self.upper_cmp_rep
        self.ubtail_super_tail_cmp = self.ubtail_ub_loop % self.upper_cmp_rep

        # init l1
        l1_shape = [1, self.k_d, self.input_h + self.dirty_l1_pad, self.input_w, self.c0]

        self.aux_l1_tensor = self.inst.Tensor(dtype=self.input_dtype,
                                              shape=l1_shape,
                                              name='aux_l1',
                                              scope=tik.scope_cbuf)

    # 'pylint: disable=too-many-locals
    def _cut_ub_process(self, ind_ctx, mode):
        ind_n = ind_ctx['ind_n']
        ind_c1 = ind_ctx['ind_c1']
        ind_d = ind_ctx['ind_d']

        # init a big matrix in ub, for register the img2col.
        bigmat_ub_tensor = self.inst.Tensor(dtype=self.input_dtype,
                                            shape=[self.k_elem, self.cut_ub_val * self.c0],
                                            name='big_matrix',
                                            scope=tik.scope_ubuf)
        # init a register to store the max line.
        maxline_ub_tensor = self.inst.Tensor(dtype=self.input_dtype,
                                             shape=[self.cut_ub_val * self.c0, ],
                                             name='max_line',
                                             scope=tik.scope_ubuf)
        # init a uint16 bitmask, each number represents 16 bits. (16 pairs of numbers)
        bitmask_ub_tensor = self.inst.Tensor(dtype=self.out_mask_type,
                                             shape=[self.k_elem, self.cut_ub_val],
                                             name='bitmask_ub',
                                             scope=tik.scope_ubuf)
        # init some tensors for deduplication during the inter process.
        maskor_ub_tensor = self.inst.Tensor(dtype=self.out_mask_type,
                                            shape=[self.cut_ub_val, ],
                                            name='mask_or',
                                            scope=tik.scope_ubuf)
        masknot_ub_tensor = self.inst.Tensor(dtype=self.out_mask_type,
                                             shape=[self.cut_ub_val, ],
                                             name='mask_not',
                                             scope=tik.scope_ubuf)

        # loop process
        if mode == 'loop':
            cut_ub_ind = ind_ctx['cut_ub_ind']
            # step1: begin to calculate img2col.
            l1_begin_h = (self.cut_ub_val * cut_ub_ind // self.w_step) * self.stride_h
            l1_begin_w = (self.cut_ub_val * cut_ub_ind % self.w_step) * self.stride_w
            l1_begin_pos = (l1_begin_h, l1_begin_w)
            super()._img2col(aux_l1=self.aux_l1_tensor,
                             big_matrix_ub=bigmat_ub_tensor,
                             l1_begin_pos=l1_begin_pos,
                             rep_times=self.cut_ub_val // 16)

            # step2: begin to calculate max line.
            super()._calc_maxline(max_line_ub=maxline_ub_tensor,
                                  big_matrix_ub=bigmat_ub_tensor,
                                  line_blk=self.cut_ub_val,
                                  line_rep_loop=self.line_rep_loop,
                                  line_rep_tail=0,
                                  super_line_loop=self.super_line_loop,
                                  super_line_tail=self.super_line_tail)

            # step3: move the max_line_ub to gm
            self._move_ctn(dst=self.output_gm_tensor[ind_n, ind_d, ind_c1, cut_ub_ind * self.ub_line],
                           src=maxline_ub_tensor,
                           burst=self.ub_line * self.each_fp16_bytes // self.block_size)

            # step4: compute the bitmask
            super()._calc_bitmask(big_matrix_ub=bigmat_ub_tensor,
                                  bitmask_ub=bitmask_ub_tensor,
                                  max_line_ub=maxline_ub_tensor,
                                  super_line_loop=self.super_line_loop_cmp,
                                  super_line_tail=self.super_line_tail_cmp)

            # step5: deduplicate the bitmask, each column must have at most one '1'.
            super()._deduplicate_bitmask(mask_or_ub=maskor_ub_tensor,
                                         mask_not_ub=masknot_ub_tensor,
                                         bitmask_ub=bitmask_ub_tensor,
                                         bm_loop=self.bm_line_loop,
                                         bm_tail=self.bm_line_tail,
                                         data_blk=self.cut_ub_val // 16)

            # step6: move bitmask to gm
            self.inst.data_move(
                dst=self.final_bitmask_gm_tensor[ind_n, ind_d, ind_c1 * self.k_elem,
                                                 cut_ub_ind * self.cut_ub_val // 16 * 16],
                src=bitmask_ub_tensor,
                sid=0,
                nburst=self.k_elem,
                burst=self.cut_ub_val * self.each_uint16_bytes // self.block_size,
                src_stride=0,
                dst_stride=(self.aligned_bm_line - self.cut_ub_val) // 16
            )
            return

        # tail process
        # step1: begin to calculate img2col
        l1_begin_h = (self.cut_ub_val * self.cut_ub_loop // self.w_step) * self.stride_h
        l1_begin_w = (self.cut_ub_val * self.cut_ub_loop % self.w_step) * self.stride_w
        l1_begin_pos = (l1_begin_h, l1_begin_w)
        super()._img2col(aux_l1=self.aux_l1_tensor,
                         big_matrix_ub=bigmat_ub_tensor,
                         l1_begin_pos=l1_begin_pos,
                         rep_times=self.cut_ub_tail_aligned // 16)

        # step2: begin to calculate max line.
        super()._calc_maxline(max_line_ub=maxline_ub_tensor,
                              big_matrix_ub=bigmat_ub_tensor,
                              line_blk=self.cut_ub_tail_aligned,
                              line_rep_loop=self.ubtail_ub_loop,
                              line_rep_tail=0,
                              super_line_loop=self.ubtail_super_loop,
                              super_line_tail=self.ubtail_super_tail)

        # step3: move the max_line_ub to gm
        self._move_ctn(dst=self.output_gm_tensor[ind_n, ind_d, ind_c1, self.cut_ub_loop * self.ub_line],
                       src=maxline_ub_tensor,
                       burst=self.cut_ub_tail_ori * self.c0 * self.each_fp16_bytes // self.block_size)

        # step4: compute the bitmask
        super()._calc_bitmask(big_matrix_ub=bigmat_ub_tensor,
                              bitmask_ub=bitmask_ub_tensor,
                              max_line_ub=maxline_ub_tensor,
                              super_line_loop=self.ubtail_super_loop_cmp,
                              super_line_tail=self.ubtail_super_tail_cmp)

        # step5: deduplicate the bitmask, each column must have at most one '1'.
        super()._deduplicate_bitmask(mask_or_ub=maskor_ub_tensor,
                                     mask_not_ub=masknot_ub_tensor,
                                     bitmask_ub=bitmask_ub_tensor,
                                     bm_loop=self.ubtail_bm_loop,
                                     bm_tail=self.ubtail_bm_tail,
                                     data_blk=self.cut_ub_tail_aligned // 16)

        # step6: move bitmask to gm
        self.inst.data_move(
            dst=self.final_bitmask_gm_tensor[ind_n, ind_d, ind_c1 * self.k_elem,
                                             self.cut_ub_loop * self.cut_ub_val // 16 * 16],
            src=bitmask_ub_tensor,
            sid=0,
            nburst=self.k_elem,
            burst=self.cut_ub_tail_aligned // 16,
            src_stride=(self.cut_ub_val - self.cut_ub_tail_aligned) // 16,
            dst_stride=(self.aligned_bm_line - self.cut_ub_tail_aligned) // 16
        )
        return

    def run(self):
        core_nums = self.n * self.c1 * self.d_step

        with self.inst.for_range(0, core_nums, block_num=core_nums) as core_ind:

            core = Core(core_ind, self.c1, self.d_step)

            # if we don't need to cut l1, now is a good time to move data that all we need in this d-loop.
            if self.l1_strategy is L1MoveStrategy.NO_NEED_TO_CUT:
                if (self.c1 - 1) * self.input_h * self.input_w <= 65535:
                    self.inst.data_move(
                        dst=self.aux_l1_tensor[0],
                        src=self.input_gm_tensor[core.ind_n, core.ind_d * self.stride_d, core.ind_c1, 0, 0, 0],
                        sid=0,
                        nburst=self.k_d,
                        burst=self.input_h * self.input_w,
                        src_stride=(self.c1 - 1) * self.input_h * self.input_w,
                        dst_stride=self.dirty_l1_pad * self.input_w
                    )
                else:
                    with self.inst.for_range(0, self.k_d) as d_ind:
                        self._move_ctn(
                            dst=self.aux_l1_tensor[0, d_ind, 0, 0, 0],
                            src=self.input_gm_tensor[core.ind_n, core.ind_d * self.stride_d + d_ind,
                                                     core.ind_c1, 0, 0, 0],
                            burst=self.input_h * self.input_w
                        )

            ub_context = {"ind_n": core.ind_n,
                          "ind_c1": core.ind_c1,
                          "ind_d": core.ind_d}
            # begin to calculate cut ub-process
            if self.cut_ub_loop != 0:
                with self.inst.for_range(0, self.cut_ub_loop) as cut_ub_ind:
                    ub_context["cut_ub_ind"] = cut_ub_ind
                    self._cut_ub_process(ub_context, mode='loop')
            if self.cut_ub_tail_aligned != 0:
                self._cut_ub_process(ub_context, mode='tail')

        output_gm_tensor_reshape = self.output_gm_tensor.reshape(self.output_shape)
        self.inst.BuildCCE(kernel_name=self.kernel_name, inputs=[self.input_gm_tensor],
                           outputs=[output_gm_tensor_reshape, self.final_bitmask_gm_tensor])
        return self.inst


# 'pylint: disable=too-few-public-methods
class MaxPool3DWithArgmaxMilan(MaxPool3DWithArgmax):
    """
    MaxPool3DWithArgmaxMilan: inherited from MaxPool3DWithArgmax
    """

    def __init__(self, x, kernel_size, stride, kernel_name):
        # init some value for cut-ub process
        super().__init__(x, kernel_size, stride, kernel_name)

        self.ub_kernel_stg = self._ub_tiling_stg()
        if self.ub_kernel_stg == UbKernelStrategy.WHOLE_KERNEL:
            self.h_step_once = self.h_step
            self.w_step_once = self.w_step

            self.h_step_loop = 1
            self.h_step_tail = 0

            self.cut_ub_val = self.h_step_once * self.w_step_once
            self.aligned_cut_ub_val = math.ceil(self.cut_ub_val / self.each_block_elem) * self.each_block_elem
            self.tail_cut_ub_val = 0

            self.is_memory_conflict = False

        if self.ub_kernel_stg == UbKernelStrategy.CUT_H:
            self.h_step_once = self._calc_h_step_once()
            self.w_step_once = self.w_step
            if 0 == self.h_step_once:
                raise RuntimeError("The once h_step can not be 0!")

            self.h_step_loop = self.h_step // self.h_step_once
            self.h_step_tail = self.h_step % self.h_step_once

            self.cut_ub_val = self.h_step_once * self.w_step_once
            self.aligned_cut_ub_val = math.ceil(self.cut_ub_val / self.each_block_elem) * self.each_block_elem
            self.tail_cut_ub_val = self.h_step_tail * self.w_step
            self.aligned_tail_cut_ub_val = math.ceil(self.tail_cut_ub_val / self.each_block_elem) * self.each_block_elem

            self.is_memory_conflict = self._cut_h_memory_conflict()

        if self.ub_kernel_stg == UbKernelStrategy.CUT_H_AND_W:
            self.h_step_once = 1
            self.w_step_once = self._calc_w_step_once(self.min_w_step_once)
            if 0 == self.w_step_once:
                raise RuntimeError("The once w_step can not be 0!")

            self.w_step_loop = self.w_step // self.w_step_once
            self.w_step_tail = self.w_step % self.w_step_once

            self.cut_ub_val = self.h_step_once * self.w_step_once
            self.aligned_cut_ub_val = math.ceil(self.cut_ub_val / self.each_block_elem) * self.each_block_elem
            self.tail_cut_ub_val = self.w_step_tail
            self.aligned_tail_cut_ub_val = math.ceil(self.tail_cut_ub_val / self.each_block_elem) * self.each_block_elem

            self.is_memory_conflict = self._cut_w_memory_conflict()

        # Notice: line_rep_loop probably more than 255
        self.ub_line = self.cut_ub_val * const.C0_SIZE
        self.line_rep_loop = self.ub_line // self.vector_size_elem
        self.line_rep_tail = self.ub_line % self.vector_size_elem
        self.super_line_loop = self.line_rep_loop // self.upper_rep_limit
        self.super_line_tail = self.line_rep_loop % self.upper_rep_limit

        self.line_rep_loop_cmp = math.ceil(self.ub_line / self.vector_size_elem)
        self.super_line_loop_cmp = self.line_rep_loop_cmp // self.upper_cmp_rep
        self.super_line_tail_cmp = self.line_rep_loop_cmp % self.upper_cmp_rep

        self.bm_line_loop = self.cut_ub_val // self.vector_size_elem
        self.bm_line_tail = self.cut_ub_val % self.vector_size_elem

        # compute some values for ub-cut-tail process
        self.tail_ub_line = self.tail_cut_ub_val * const.C0_SIZE
        self.tail_line_rep_loop = self.tail_ub_line // self.vector_size_elem
        self.tail_line_rep_tail = self.tail_ub_line % self.vector_size_elem
        self.tail_super_line_loop = self.tail_line_rep_loop // self.upper_rep_limit
        self.tail_super_line_tail = self.tail_line_rep_loop % self.upper_rep_limit

        self.tail_line_rep_loop_cmp = math.ceil(self.tail_ub_line / self.vector_size_elem)
        self.tail_super_line_loop_cmp = self.tail_line_rep_loop_cmp // self.upper_cmp_rep
        self.tail_super_line_tail_cmp = self.tail_line_rep_loop_cmp % self.upper_cmp_rep

        self.tail_bm_line_loop = self.tail_cut_ub_val // self.vector_size_elem
        self.tail_bm_line_tail = self.tail_cut_ub_val % self.vector_size_elem

        # compute some values for copy data to storage
        self.storage_beg = self.aligned_bm_line - self.each_block_elem

        # init some gm_tensor and ub tensor
        self._init_ub_tensor()

    def run(self):
        core_nums = self.n * self.c1 * self.d_step

        with self.inst.for_range(0, core_nums, block_num=core_nums) as core_ind:
            core = Core(core_ind, self.c1, self.d_step)

            if self.ub_kernel_stg == UbKernelStrategy.WHOLE_KERNEL:
                self._cut_ub_process_cut_h_loop(core, h_step_ind=0)

            if self.ub_kernel_stg == UbKernelStrategy.CUT_H:
                # begin to calculate cut ub-process
                if self.h_step_loop != 0:
                    with self.inst.for_range(0, self.h_step_loop, name="h_step_ind") as h_step_ind:
                        self._cut_ub_process_cut_h_loop(core, h_step_ind=h_step_ind)
                if self.h_step_tail != 0:
                    self._cut_ub_process_cut_h_tail(core)
                if self.is_memory_conflict:
                    # 3.copy storage_ub to gm
                    self.inst.data_move(
                        dst=self.final_bitmask_gm_tensor[core.ind_n, core.ind_d,
                                                         core.ind_c1 * self.k_elem, self.storage_beg],
                        src=self.storage_ub_tensor,
                        sid=0,
                        nburst=self.k_elem,
                        burst=1,
                        src_stride=0,
                        dst_stride=self.aligned_bm_line // self.each_block_elem - 1
                    )

            if self.ub_kernel_stg == UbKernelStrategy.CUT_H_AND_W:
                # begin to calculate cut ub-process
                with self.inst.for_range(0, self.h_step, name="h_step_ind") as h_step_ind:
                    if self.w_step_loop != 0:
                        with self.inst.for_range(0, self.w_step_loop, name="w_step_ind") as w_step_ind:
                            self._cut_ub_process_cut_w_loop(core, h_step_ind, w_step_ind)
                    if self.w_step_tail != 0:
                        self._cut_ub_process_cut_w_tail(core, h_step_ind)
                if self.is_memory_conflict:
                    # 3.copy storage_ub to gm
                    self.inst.data_move(
                        dst=self.final_bitmask_gm_tensor[core.ind_n, core.ind_d,
                                                         core.ind_c1 * self.k_elem, self.storage_beg],
                        src=self.storage_ub_tensor,
                        sid=0,
                        nburst=self.k_elem,
                        burst=1,
                        src_stride=0,
                        dst_stride=self.aligned_bm_line // self.each_block_elem - 1
                    )

        output_gm_tensor_reshape = self.output_gm_tensor.reshape(self.output_shape)
        self.inst.BuildCCE(kernel_name=self.kernel_name, inputs=[self.input_gm_tensor],
                           outputs=[output_gm_tensor_reshape, self.final_bitmask_gm_tensor])

        return self.inst

    def img2col_milan(self, bigmat_ub_tensor, aux_ub, h_step):
        aux_ub_shape = aux_ub.shape
        with self.inst.for_range(0, self.k_d) as d_ind:
            with self.inst.for_range(0, self.k_h * self.k_w) as hw_ind:
                h_ind = hw_ind // self.k_w
                w_ind = hw_ind % self.k_w
                k_ind = d_ind * self.k_h * self.k_w + hw_ind

                vadds_h_loop = h_step // const.REPEAT_STRIDE_EIGHT
                vadds_h_tail = h_step % const.REPEAT_STRIDE_EIGHT
                vadds_w_loop = self.w_step_once // self.upper_rep_limit
                vadds_w_tail = self.w_step_once % self.upper_rep_limit
                if vadds_h_loop != 0:
                    if vadds_w_loop != 0:
                        with self.inst.for_range(0, vadds_h_loop) as vadds_h_loop_ind:
                            with self.inst.for_range(0, vadds_w_loop) as vadds_w_loop_ind:
                                self.inst.vadds(
                                    mask=const.MASK128,
                                    dst=bigmat_ub_tensor[
                                        k_ind,
                                        (vadds_h_loop_ind * const.REPEAT_STRIDE_EIGHT * self.w_step_once +
                                         vadds_w_loop_ind * self.upper_rep_limit) * const.C0_SIZE
                                    ],
                                    src=aux_ub[
                                        0, d_ind,
                                        vadds_h_loop_ind * const.REPEAT_STRIDE_EIGHT * self.stride_h + h_ind,
                                        vadds_w_loop_ind * self.upper_rep_limit * self.stride_w + w_ind, 0
                                    ],
                                    scalar=0,
                                    repeat_times=self.upper_rep_limit,
                                    dst_blk_stride=self.w_step_once,
                                    src_blk_stride=self.stride_h * aux_ub_shape[3],
                                    dst_rep_stride=1,
                                    src_rep_stride=self.stride_w)
                    if vadds_w_tail > 0:
                        with self.inst.for_range(0, vadds_h_loop) as vadds_h_loop_ind:
                            self.inst.vadds(
                                mask=const.MASK128,
                                dst=bigmat_ub_tensor[
                                    k_ind,
                                    (vadds_h_loop_ind * const.REPEAT_STRIDE_EIGHT * self.w_step_once +
                                     vadds_w_loop * self.upper_rep_limit) * const.C0_SIZE
                                ],
                                src=aux_ub[
                                    0, d_ind,
                                    vadds_h_loop_ind * const.REPEAT_STRIDE_EIGHT * self.stride_h + h_ind,
                                    vadds_w_loop * self.upper_rep_limit * self.stride_w + w_ind, 0
                                ],
                                scalar=0,
                                repeat_times=vadds_w_tail,
                                dst_blk_stride=self.w_step_once,
                                src_blk_stride=self.stride_h * aux_ub_shape[3],
                                dst_rep_stride=1,
                                src_rep_stride=self.stride_w
                            )
                if vadds_h_tail != 0:
                    if vadds_w_loop != 0:
                        with self.inst.for_range(0, vadds_w_loop) as vadds_w_loop_ind:
                            self.inst.vadds(
                                mask=vadds_h_tail * self.each_block_elem,
                                dst=bigmat_ub_tensor[
                                    k_ind, (vadds_h_loop * const.REPEAT_STRIDE_EIGHT * self.w_step_once +
                                    vadds_w_loop_ind * self.upper_rep_limit) * const.C0_SIZE
                                ],
                                src=aux_ub[0, d_ind, vadds_h_loop * const.REPEAT_STRIDE_EIGHT * self.stride_h + h_ind,
                                           vadds_w_loop_ind * self.upper_rep_limit * self.stride_w + w_ind, 0],
                                scalar=0,
                                repeat_times=self.upper_rep_limit,
                                dst_blk_stride=self.w_step_once,
                                src_blk_stride=self.stride_h * aux_ub_shape[3],
                                dst_rep_stride=1,
                                src_rep_stride=self.stride_w
                            )
                    if vadds_w_tail > 0:
                        self.inst.vadds(
                            mask=vadds_h_tail * self.each_block_elem,
                            dst=bigmat_ub_tensor[
                                k_ind,
                                (vadds_h_loop * const.REPEAT_STRIDE_EIGHT * self.w_step_once +
                                 vadds_w_loop * self.upper_rep_limit) * const.C0_SIZE
                            ],
                            src=aux_ub[
                                0, d_ind, vadds_h_loop * const.REPEAT_STRIDE_EIGHT * self.stride_h + h_ind,
                                vadds_w_loop * self.upper_rep_limit * self.stride_w + w_ind, 0
                            ],
                            scalar=0,
                            repeat_times=vadds_w_tail,
                            dst_blk_stride=self.w_step_once,
                            src_blk_stride=self.stride_h * aux_ub_shape[3],
                            dst_rep_stride=1,
                            src_rep_stride=self.stride_w
                        )

    def _ub_tiling_stg(self):
        if self._calc_ub_elem(self.h_step, self.w_step) - self.k_elem * self.each_block_elem < self.ub_max_elem:
            return UbKernelStrategy.WHOLE_KERNEL
        if self._calc_ub_elem(1, self.w_step) < self.ub_max_elem:
            return UbKernelStrategy.CUT_H
        if self.w_step >= self.each_block_elem and self._calc_ub_elem(1, self.each_block_elem) < self.ub_max_elem:
            self.min_w_step_once = self.each_block_elem
            return UbKernelStrategy.CUT_H_AND_W
        if self._calc_ub_elem(1, 1) < self.ub_max_elem:
            self.min_w_step_once = 1
            return UbKernelStrategy.CUT_H_AND_W

        raise RuntimeError("Kenrnel size is too large, please check!")

    def _calc_ub_elem(self, h_step_once, w_step_once):
        # all the tensors that occupy the ub:(X represents the size of bitmask line)
        # -------------------------------------------------------------------------------------------------------
        # name                        |  size (regardless of ub-memory)   |  hypothetical variable (NOT accuracy)
        # -------------------------------------------------------------------------------------------------------
        # aux_ub_T                    |  k_d * input_h * input_w * C0     |
        # bigmat_ub_T                 |  k_elem * w_step * h_step * C0    |  X * 16 * k_elem
        # bitmask_ub_T                |  k_elem * aligned_bitmask_line    |  X  * k_elem
        # maxline_ub_T                |  w_step * h_step * C0             |  X * 16
        # maskor_ub_T + masknot_ub_T  |  aligned_bitmask_line * 2         |  X * 2
        # storage_ub_T                |  k_elem * 16                      |  k_elem * 16
        # -------------------------------------------------------------------------------------------------------

        aux_ub_size = self.k_d * ((h_step_once - 1) * self.stride_h + self.k_h) * \
                      ((w_step_once - 1) * self.stride_w + self.k_w) * const.C0_SIZE
        bigmat_ub_size = h_step_once * w_step_once * const.C0_SIZE * self.k_elem
        bitmask_ub_size = math.ceil(h_step_once * w_step_once / const.SIZE_SIXTEEN) * const.SIZE_SIXTEEN * self.k_elem
        maxline_ub_size = math.ceil(h_step_once * w_step_once / 8) * 8 * const.C0_SIZE
        maskornot_ub_size = h_step_once * w_step_once * 2
        storage_ub_size = self.k_elem * self.each_block_elem
        return aux_ub_size + bigmat_ub_size + bitmask_ub_size + maxline_ub_size + maskornot_ub_size + storage_ub_size

    def _calc_h_step_once(self):
        # compute how much h_step moved into ub once when UbKernelStrategy is CUT_H
        h_step_once = 1
        while True:
            if self._calc_ub_elem(h_step_once, self.w_step) > self.ub_max_elem or h_step_once > self.h_step:
                h_step_once -= 1
                break
            h_step_once += 1
        return h_step_once

    def _cut_h_memory_conflict(self):
        if self.h_step_tail == 0:
            return (self.h_step_loop - 1) * self.cut_ub_val + self.aligned_cut_ub_val > self.aligned_bm_line
        else:
            return self.h_step_loop * self.cut_ub_val + self.aligned_tail_cut_ub_val > self.aligned_bm_line

    def _calc_w_step_once(self, min_w_step_once):
        # compute how much w_step moved into ub once when UbKernelStrategy is CUT_H_AND_W
        w_step_once = min_w_step_once
        while True:
            if self._calc_ub_elem(1, w_step_once) > self.ub_max_elem or w_step_once > self.w_step:
                w_step_once -= min_w_step_once
                break
            w_step_once += min_w_step_once
        return w_step_once

    def _cut_w_memory_conflict(self):
        if self.w_step_tail == 0:
            return (self.h_step - 1) * self.w_step + (self.w_step_loop - 1) * self.cut_ub_val + \
                   self.aligned_cut_ub_val > self.aligned_bm_line
        else:
            return (self.h_step - 1) * self.w_step + self.w_step_loop * self.cut_ub_val + \
                   self.aligned_tail_cut_ub_val > self.aligned_bm_line

    def _init_ub_tensor(self):
        aux_ub_shape = [1, self.k_d, (self.h_step_once - 1) * self.stride_h + self.k_h,
                        (self.w_step_once - 1) * self.stride_w + self.k_w, const.C0_SIZE]
        # init a map in ub
        self.aux_ub_tensor = self.inst.Tensor(dtype=self.input_dtype,
                                              shape=aux_ub_shape,
                                              name="aux_ub",
                                              scope=tik.scope_ubuf)

        # init a big maxtrix in ub, for register the img2col.
        self.bigmat_ub_tensor = self.inst.Tensor(dtype=self.input_dtype,
                                                 shape=[self.k_elem, self.cut_ub_val * const.C0_SIZE, ],
                                                 name="big_matrix",
                                                 scope=tik.scope_ubuf)

        # init a register to store the max line.
        self.maxline_ub_tensor = self.inst.Tensor(dtype=self.input_dtype,
                                                  shape=[math.ceil(self.cut_ub_val / 8) * 8 * const.C0_SIZE, ],
                                                  name="max_line",
                                                  scope=tik.scope_ubuf)

        # init a uint16 bitmask, each number represents 16 bits. (16 pairs of numbers)
        self.bitmask_ub_tensor = self.inst.Tensor(dtype=self.out_mask_type,
                                                  shape=[self.k_elem, self.aligned_cut_ub_val],
                                                  name="bitmask_ub",
                                                  scope=tik.scope_ubuf)

        # init some tensors for deduplication during the inter process.
        self.maskor_ub_tensor = self.inst.Tensor(dtype=self.out_mask_type,
                                                 shape=[self.cut_ub_val, ],
                                                 name="mask_or",
                                                 scope=tik.scope_ubuf)

        self.masknot_ub_tensor = self.inst.Tensor(dtype=self.out_mask_type,
                                                  shape=[self.cut_ub_val, ],
                                                  name="mask_not",
                                                  scope=tik.scope_ubuf)

        if self.is_memory_conflict:
            self.storage_ub_tensor = self.inst.Tensor(dtype=self.out_mask_type,
                                                      shape=[self.k_elem, const.C0_SIZE],
                                                      name="storage",
                                                      scope=tik.scope_ubuf)

    def _cut_ub_process_cut_h_loop(self, core, h_step_ind):
        # loop process for CUT_H strategy
        # step1: move feature map into ub
        with self.inst.for_range(0, self.k_d) as d_ind:
            self.inst.data_move(
                dst=self.aux_ub_tensor[0, d_ind, 0, 0, 0],
                src=self.input_gm_tensor[core.ind_n, core.ind_d * self.stride_d + d_ind, core.ind_c1,
                                         h_step_ind * self.h_step_once * self.stride_h, 0, 0],
                sid=0,
                nburst=self.aux_ub_tensor.shape[2],
                burst=self.aux_ub_tensor.shape[3],
                src_stride=self.input_w - self.aux_ub_tensor.shape[3],
                dst_stride=0
            )

        # step2: begin to calculate img2col
        self.img2col_milan(bigmat_ub_tensor=self.bigmat_ub_tensor,
                           aux_ub=self.aux_ub_tensor,
                           h_step=self.h_step_once)

        # step3: begin to calculate max line.
        super()._calc_maxline(max_line_ub=self.maxline_ub_tensor,
                              big_matrix_ub=self.bigmat_ub_tensor,
                              line_blk=self.cut_ub_val,
                              line_rep_loop=self.line_rep_loop,
                              line_rep_tail=self.line_rep_tail,
                              super_line_loop=self.super_line_loop,
                              super_line_tail=self.super_line_tail)

        # step4: move the max_line_ub to gm
        cut_ub_val_beg = h_step_ind * self.cut_ub_val
        self._move_ctn(dst=self.output_gm_tensor[core.ind_n, core.ind_d, core.ind_c1, cut_ub_val_beg * const.C0_SIZE],
                       src=self.maxline_ub_tensor,
                       burst=self.ub_line // self.each_block_elem)

        # step5: compute the bit mask
        super()._calc_bitmask(big_matrix_ub=self.bigmat_ub_tensor,
                              bitmask_ub=self.bitmask_ub_tensor,
                              max_line_ub=self.maxline_ub_tensor,
                              super_line_loop=self.super_line_loop_cmp,
                              super_line_tail=self.super_line_tail_cmp)

        # step6: deduplicate the bitmask, each column must have at most one "1".
        super()._deduplicate_bitmask(mask_or_ub=self.maskor_ub_tensor,
                                     mask_not_ub=self.masknot_ub_tensor,
                                     bitmask_ub=self.bitmask_ub_tensor,
                                     bm_loop=self.bm_line_loop,
                                     bm_tail=self.bm_line_tail,
                                     data_blk=self.aligned_cut_ub_val // self.each_block_elem)

        # step7: move bitmask to gm
        if self.is_memory_conflict:
            # 1.move first ceil(cut_ub_val / 16) - 1 block if could
            with self.inst.if_scope(cut_ub_val_beg + self.aligned_cut_ub_val > self.aligned_bm_line):
                if self.cut_ub_val > self.each_block_elem:
                    burst_first = self.aligned_cut_ub_val // self.each_block_elem - 1
                    self.inst.data_move(
                        dst=self.final_bitmask_gm_tensor[core.ind_n, core.ind_d,
                                                         core.ind_c1 * self.k_elem, cut_ub_val_beg],
                        src=self.bitmask_ub_tensor,
                        sid=0,
                        nburst=self.k_elem,
                        burst=burst_first,
                        src_stride=1,
                        dst_stride=self.aligned_bm_line // self.each_block_elem - burst_first
                    )
            with self.inst.else_scope():
                self.inst.data_move(
                    dst=self.final_bitmask_gm_tensor[core.ind_n, core.ind_d,
                                                     core.ind_c1 * self.k_elem, cut_ub_val_beg],
                    src=self.bitmask_ub_tensor,
                    sid=0,
                    nburst=self.k_elem,
                    burst=self.aligned_cut_ub_val // self.each_block_elem,
                    src_stride=0,
                    dst_stride=(self.aligned_bm_line - self.aligned_cut_ub_val) // self.each_block_elem
                )

            # 2.copy data to storage_ub
            self._copy_to_storage(cut_ub_val_beg=cut_ub_val_beg,
                                  valid_len=self.cut_ub_val,
                                  storage_ub_tensor=self.storage_ub_tensor,
                                  bitmask_ub_tensor=self.bitmask_ub_tensor)
        else:
            self.inst.data_move(
                dst=self.final_bitmask_gm_tensor[core.ind_n, core.ind_d,
                                                 core.ind_c1 * self.k_elem, cut_ub_val_beg],
                src=self.bitmask_ub_tensor,
                sid=0,
                nburst=self.k_elem,
                burst=self.aligned_cut_ub_val // self.each_block_elem,
                src_stride=0,
                dst_stride=(self.aligned_bm_line - self.aligned_cut_ub_val) // self.each_block_elem
            )

    def _cut_ub_process_cut_h_tail(self, core):
        # tail process for CUT_H strategy
        aux_ub_tail_shape = [1, self.k_d, (self.h_step_tail - 1) * self.stride_h + self.k_h,
                             (self.w_step - 1) * self.stride_w + self.k_w, const.C0_SIZE]

        # step1: move feature map into ub
        with self.inst.for_range(0, self.k_d) as d_ind:
            self.inst.data_move(
                dst=self.aux_ub_tensor[0, d_ind, 0, 0, 0],
                src=self.input_gm_tensor[core.ind_n, core.ind_d * self.stride_d + d_ind,
                                         core.ind_c1,
                                         self.h_step_loop * self.h_step_once * self.stride_h, 0, 0],
                sid=0,
                nburst=aux_ub_tail_shape[2],
                burst=aux_ub_tail_shape[3],
                src_stride=self.input_w - aux_ub_tail_shape[3],
                dst_stride=0
            )

        # step2: begin to calculate img2col
        self.img2col_milan(bigmat_ub_tensor=self.bigmat_ub_tensor,
                           aux_ub=self.aux_ub_tensor,
                           h_step=self.h_step_tail)

        # step3: begin to calculate max line
        super()._calc_maxline(max_line_ub=self.maxline_ub_tensor,
                              big_matrix_ub=self.bigmat_ub_tensor,
                              line_blk=self.tail_cut_ub_val,
                              line_rep_loop=self.tail_line_rep_loop,
                              line_rep_tail=self.tail_line_rep_tail,
                              super_line_loop=self.tail_super_line_loop,
                              super_line_tail=self.tail_super_line_tail)

        # step4: move the max_line_ub to gm
        cut_ub_val_beg = self.h_step_loop * self.cut_ub_val
        self._move_ctn(dst=self.output_gm_tensor[core.ind_n, core.ind_d, core.ind_c1, cut_ub_val_beg * const.C0_SIZE],
                       src=self.maxline_ub_tensor,
                       burst=self.tail_ub_line // self.each_block_elem)

        # step5: compute the bit mask
        super()._calc_bitmask(big_matrix_ub=self.bigmat_ub_tensor,
                              bitmask_ub=self.bitmask_ub_tensor,
                              max_line_ub=self.maxline_ub_tensor,
                              super_line_loop=self.tail_super_line_loop_cmp,
                              super_line_tail=self.tail_super_line_tail_cmp)

        # step6: deduplicate the bitmask, each column must have at most "1"
        super()._deduplicate_bitmask(mask_or_ub=self.maskor_ub_tensor,
                                     mask_not_ub=self.masknot_ub_tensor,
                                     bitmask_ub=self.bitmask_ub_tensor,
                                     bm_loop=self.tail_bm_line_loop,
                                     bm_tail=self.tail_bm_line_tail,
                                     data_blk=self.aligned_tail_cut_ub_val // self.each_block_elem)

        # step7: move bitmask to gm
        if self.is_memory_conflict:
            # 1.move first ceil(tail_cut_ub_val / 16) - 1 block if could
            if self.tail_cut_ub_val > self.each_block_elem:
                burst_first = self.aligned_tail_cut_ub_val // self.each_block_elem - 1
                self.inst.data_move(
                    dst=self.final_bitmask_gm_tensor[core.ind_n, core.ind_d,
                                                     core.ind_c1 * self.k_elem, cut_ub_val_beg],
                    src=self.bitmask_ub_tensor,
                    sid=0,
                    nburst=self.k_elem,
                    burst=burst_first,
                    src_stride=self.aligned_cut_ub_val // self.each_block_elem - burst_first,
                    dst_stride=self.aligned_bm_line // self.each_block_elem - burst_first
                )

            # 2.copy data to storage_ub
            self._copy_to_storage(cut_ub_val_beg=cut_ub_val_beg,
                                  valid_len=self.tail_cut_ub_val,
                                  storage_ub_tensor=self.storage_ub_tensor,
                                  bitmask_ub_tensor=self.bitmask_ub_tensor)
        else:
            self.inst.data_move(
                dst=self.final_bitmask_gm_tensor[core.ind_n, core.ind_d, core.ind_c1 * self.k_elem, cut_ub_val_beg],
                src=self.bitmask_ub_tensor,
                sid=0,
                nburst=self.k_elem,
                burst=self.aligned_tail_cut_ub_val // self.each_block_elem,
                src_stride=(self.aligned_cut_ub_val - self.aligned_tail_cut_ub_val) // self.each_block_elem,
                dst_stride=(self.aligned_bm_line - self.aligned_tail_cut_ub_val) // self.each_block_elem
            )

    def _cut_ub_process_cut_w_loop(self, core, h_step_ind, w_step_ind):
        # loop process for CUT_H_AND_W strategy
        # step1: move feature map into ub
        with self.inst.for_range(0, self.k_d) as k_d_ind:
            self.inst.data_move(
                dst=self.aux_ub_tensor[0, k_d_ind, 0, 0, 0],
                src=self.input_gm_tensor[core.ind_n, core.ind_d * self.stride_d + k_d_ind, core.ind_c1,
                                         h_step_ind * self.h_step_once * self.stride_h,
                                         w_step_ind * self.w_step_once * self.stride_w, 0],
                sid=0,
                nburst=self.aux_ub_tensor.shape[2],
                burst=self.aux_ub_tensor.shape[3],
                src_stride=self.input_w - self.aux_ub_tensor.shape[3],
                dst_stride=0
            )

        # step2: begin to calculate img2col
        self.img2col_milan(bigmat_ub_tensor=self.bigmat_ub_tensor,
                           aux_ub=self.aux_ub_tensor,
                           h_step=self.h_step_once)

        # step3: begin to calculate max line.
        super()._calc_maxline(max_line_ub=self.maxline_ub_tensor,
                              big_matrix_ub=self.bigmat_ub_tensor,
                              line_blk=self.cut_ub_val,
                              line_rep_loop=self.line_rep_loop,
                              line_rep_tail=self.line_rep_tail,
                              super_line_loop=self.super_line_loop,
                              super_line_tail=self.super_line_tail)

        # step4: move the max_line_ub to gm
        cut_ub_val_beg = h_step_ind * self.w_step + w_step_ind * self.cut_ub_val
        self._move_ctn(dst=self.output_gm_tensor[core.ind_n, core.ind_d, core.ind_c1, cut_ub_val_beg * const.C0_SIZE],
                       src=self.maxline_ub_tensor,
                       burst=self.ub_line // self.each_block_elem)

        # step5: compute the bit mask
        super()._calc_bitmask(big_matrix_ub=self.bigmat_ub_tensor,
                              bitmask_ub=self.bitmask_ub_tensor,
                              max_line_ub=self.maxline_ub_tensor,
                              super_line_loop=self.super_line_loop_cmp,
                              super_line_tail=self.super_line_tail_cmp)

        # step6: deduplicate the bitmask, each column must have at most one "1".
        super()._deduplicate_bitmask(mask_or_ub=self.maskor_ub_tensor,
                                     mask_not_ub=self.masknot_ub_tensor,
                                     bitmask_ub=self.bitmask_ub_tensor,
                                     bm_loop=self.bm_line_loop,
                                     bm_tail=self.bm_line_tail,
                                     data_blk=self.aligned_cut_ub_val // self.each_block_elem)

        # step7: move bitmask to gm
        if self.is_memory_conflict:
            # 1. move first ceil(cut_ub_val / 16) - 1 block if could
            with self.inst.if_scope(cut_ub_val_beg + self.aligned_cut_ub_val > self.aligned_bm_line):
                if self.cut_ub_val > self.each_block_elem:
                    burst_first = self.aligned_cut_ub_val // self.each_block_elem - 1
                    self.inst.data_move(
                        dst=self.final_bitmask_gm_tensor[core.ind_n, core.ind_d,
                                                         core.ind_c1 * self.k_elem, cut_ub_val_beg],
                        src=self.bitmask_ub_tensor,
                        sid=0,
                        nburst=self.k_elem,
                        burst=burst_first,
                        src_stride=1,
                        dst_stride=self.aligned_bm_line // self.each_block_elem - burst_first
                    )
            with self.inst.else_scope():
                self.inst.data_move(
                    dst=self.final_bitmask_gm_tensor[core.ind_n, core.ind_d, core.ind_c1 * self.k_elem, cut_ub_val_beg],
                    src=self.bitmask_ub_tensor,
                    sid=0,
                    nburst=self.k_elem,
                    burst=self.aligned_cut_ub_val // self.each_block_elem,
                    src_stride=0,
                    dst_stride=(self.aligned_bm_line - self.aligned_cut_ub_val) // self.each_block_elem
                )

            # 2.copy data to storage_ub
            self._copy_to_storage(cut_ub_val_beg=cut_ub_val_beg,
                                  valid_len=self.cut_ub_val,
                                  storage_ub_tensor=self.storage_ub_tensor,
                                  bitmask_ub_tensor=self.bitmask_ub_tensor)
        else:
            self.inst.data_move(
                dst=self.final_bitmask_gm_tensor[core.ind_n, core.ind_d, core.ind_c1 * self.k_elem, cut_ub_val_beg],
                src=self.bitmask_ub_tensor,
                sid=0,
                nburst=self.k_elem,
                burst=self.aligned_cut_ub_val // self.each_block_elem,
                src_stride=0,
                dst_stride=(self.aligned_bm_line - self.aligned_cut_ub_val) // self.each_block_elem
            )

    def _cut_ub_process_cut_w_tail(self, core, h_step_ind):
        # tail process for CUT_H_AND_W strategy
        aux_ub_tail_shape = [1, self.k_d, (self.h_step_once - 1) * self.stride_h + self.k_h,
                             (self.w_step_tail - 1) * self.stride_w + self.k_w, const.C0_SIZE]

        # step1: move feature map into ub
        with self.inst.for_range(0, self.k_d) as d_ind:
            self.inst.data_move(
                dst=self.aux_ub_tensor[0, d_ind, 0, 0, 0],
                src=self.input_gm_tensor[core.ind_n, core.ind_d * self.stride_d + d_ind,
                                         core.ind_c1,
                                         h_step_ind * self.h_step_once * self.stride_h,
                                         self.w_step_loop * self.w_step_once * self.stride_w, 0],
                sid=0,
                nburst=aux_ub_tail_shape[2],
                burst=aux_ub_tail_shape[3],
                src_stride=self.input_w - aux_ub_tail_shape[3],
                dst_stride=self.aux_ub_tensor.shape[3] - aux_ub_tail_shape[3]
            )

        # step2: begin to calculate img2col
        self.img2col_milan(bigmat_ub_tensor=self.bigmat_ub_tensor,
                           aux_ub=self.aux_ub_tensor,
                           h_step=self.h_step_once)

        # step3: begin to calculate max line
        super()._calc_maxline(max_line_ub=self.maxline_ub_tensor,
                              big_matrix_ub=self.bigmat_ub_tensor,
                              line_blk=self.tail_cut_ub_val,
                              line_rep_loop=self.tail_line_rep_loop,
                              line_rep_tail=self.tail_line_rep_tail,
                              super_line_loop=self.tail_super_line_loop,
                              super_line_tail=self.tail_super_line_tail)

        # step4: move the max_line_ub to gm
        cut_ub_val_beg = h_step_ind * self.w_step + self.w_step_loop * self.cut_ub_val
        self._move_ctn(dst=self.output_gm_tensor[core.ind_n, core.ind_d, core.ind_c1, cut_ub_val_beg * const.C0_SIZE],
                       src=self.maxline_ub_tensor,
                       burst=self.tail_ub_line // self.each_block_elem)

        # step5: compute the bit mask
        super()._calc_bitmask(big_matrix_ub=self.bigmat_ub_tensor,
                              bitmask_ub=self.bitmask_ub_tensor,
                              max_line_ub=self.maxline_ub_tensor,
                              super_line_loop=self.tail_super_line_loop_cmp,
                              super_line_tail=self.tail_super_line_tail_cmp)

        # step6: deduplicate the bitmask, each column must have at most "1"
        super()._deduplicate_bitmask(mask_or_ub=self.maskor_ub_tensor,
                                     mask_not_ub=self.masknot_ub_tensor,
                                     bitmask_ub=self.bitmask_ub_tensor,
                                     bm_loop=self.tail_bm_line_loop,
                                     bm_tail=self.tail_bm_line_tail,
                                     data_blk=self.aligned_tail_cut_ub_val // self.each_block_elem)

        # step7: move bitmask to gm
        if self.is_memory_conflict:
            with self.inst.if_scope(cut_ub_val_beg + self.aligned_tail_cut_ub_val > self.aligned_bm_line):
                # 1.move first ceil(tail_cut_ub_val / 16) - 1 block if could
                if self.tail_cut_ub_val > self.each_block_elem:
                    burst_first = self.aligned_tail_cut_ub_val // self.each_block_elem - 1
                    self.inst.data_move(
                        dst=self.final_bitmask_gm_tensor[core.ind_n, core.ind_d,
                                                         core.ind_c1 * self.k_elem, cut_ub_val_beg],
                        src=self.bitmask_ub_tensor,
                        sid=0,
                        nburst=self.k_elem,
                        burst=burst_first,
                        src_stride=self.aligned_cut_ub_val // self.each_block_elem - burst_first,
                        dst_stride=self.aligned_bm_line // self.each_block_elem - burst_first
                    )
            with self.inst.else_scope():
                self.inst.data_move(
                    dst=self.final_bitmask_gm_tensor[core.ind_n, core.ind_d,
                                                     core.ind_c1 * self.k_elem, cut_ub_val_beg],
                    src=self.bitmask_ub_tensor,
                    sid=0,
                    nburst=self.k_elem,
                    burst=self.aligned_tail_cut_ub_val // self.each_block_elem,
                    src_stride=(self.aligned_cut_ub_val - self.aligned_tail_cut_ub_val) // self.each_block_elem,
                    dst_stride=(self.aligned_bm_line - self.aligned_tail_cut_ub_val) // self.each_block_elem
                )

            # 2.copy data to storage_ub
            self._copy_to_storage(cut_ub_val_beg=cut_ub_val_beg,
                                  valid_len=self.tail_cut_ub_val,
                                  storage_ub_tensor=self.storage_ub_tensor,
                                  bitmask_ub_tensor=self.bitmask_ub_tensor)

        else:
            self.inst.data_move(
                dst=self.final_bitmask_gm_tensor[core.ind_n, core.ind_d, core.ind_c1 * self.k_elem, cut_ub_val_beg],
                src=self.bitmask_ub_tensor,
                sid=0,
                nburst=self.k_elem,
                burst=self.aligned_tail_cut_ub_val // self.each_block_elem,
                src_stride=(self.aligned_cut_ub_val - self.aligned_tail_cut_ub_val) // self.each_block_elem,
                dst_stride=(self.aligned_bm_line - self.aligned_tail_cut_ub_val) // self.each_block_elem
            )

    def _copy_to_storage(self, cut_ub_val_beg, valid_len, storage_ub_tensor, bitmask_ub_tensor):
        cut_ub_val_end = cut_ub_val_beg + valid_len
        with self.inst.if_scope(cut_ub_val_end > self.storage_beg):
            with self.inst.if_scope(cut_ub_val_beg > self.storage_beg):
                with self.inst.for_range(0, self.k_elem) as k_elem_ind:
                    with self.inst.for_range(0, valid_len) as inner_ind:
                        storage_ub_tensor[k_elem_ind, cut_ub_val_beg - self.storage_beg + inner_ind] = \
                            bitmask_ub_tensor[k_elem_ind, inner_ind]
            with self.inst.else_scope():
                with self.inst.for_range(0, self.k_elem) as k_elem_ind:
                    with self.inst.for_range(0, cut_ub_val_end - self.storage_beg) as inner_ind:
                        storage_ub_tensor[k_elem_ind, inner_ind] = \
                            bitmask_ub_tensor[k_elem_ind, self.storage_beg - cut_ub_val_beg + inner_ind]


# 'pylint: disable=too-many-arguments,too-many-branches
def _check_param(x, ksize, strides, pads, dilation, ceil_mode, argmax_type):
    input_shape = x.get("shape")
    if x.get("dtype").lower() != "float16":
        raise RuntimeError("max_pool3d_with_argmax only support float16!")
    if len(input_shape) != 6:
        raise RuntimeError("Invalid shape params, max_pool3d_with_argmax input shape must be 6D format!")
    if input_shape[-1] != 16:
        raise RuntimeError("C0 must be 16!")
    if len(ksize) != 5:
        raise RuntimeError("Dimension of ksize must be 5, value of (1, 1, kernel_d, kernel_h, kernel_w)!")
    if ksize[0] != 1 or ksize[1] != 1:
        raise RuntimeError("First two dimensions of ksize should be one!")
    if ksize[3] > 255 or ksize[4] > 255:
        raise RuntimeError("Current version don't support kernel_h or kernel_w > 255!")
    if len(strides) != 5:
        raise RuntimeError("Dimension of stride must be 5, value of (1, 1, stride_d, stride_h, stride_w)!")
    if any(0 == ele for ele in ksize):
        raise RuntimeError("There can be no zero values in kszie!")
    if any(0 == ele for ele in strides):
        raise RuntimeError("There can be no zero values in strides!")
    if any(0 == ele for ele in input_shape):
        raise RuntimeError("There can be no zero values in input shape!")
    if strides[0] != 1 or strides[1] != 1:
        raise RuntimeError("First two dimensions of strides should be one!")
    if len(pads) != 3:
        raise RuntimeError("pads' shape should be (3, 2)!")
    for pad in pads:
        if len(pad) != 2:
            raise RuntimeError("pads' shape should be (3, 2)!")
    if len(dilation) != 5:
        raise RuntimeError("Dimension of dilation must be 5, value of (1, 1, dil_d, dil_h, dil_w)!")
    if dilation[0] != 1 or dilation[1] != 1:
        raise RuntimeError("First two dimensions of dilation must be one!")
    if ceil_mode:
        raise RuntimeError("Current version only support ceil_mode=False!")
    if argmax_type != "bitmask":
        raise RuntimeError("Current version only support bitmask argmax, please set argmax_type=bitmask!")

    if tbe_platform.api_check_support("tik.load3dv1"):
        # this version only support the generalization of branch MaxPool3dWithArgmaxWholeKernel,
        # that means we have to put all the kernel into ub, at each loop of img2col.
        # please refer to MaxPool3dWithArgmaxWholeKernel's init function for more details, then you will know
        # what these magic numbers mean.
        k_bound = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) // 32 - 18) // 17
        if ksize[2] * ksize[3] * ksize[4] > k_bound:
            raise RuntimeError("The kernel is so big, we don't support such generalization!")
        wo = (input_shape[4] - ksize[4]) // strides[4] + 1
        ho = (input_shape[3] - ksize[3]) // strides[3] + 1
        l1_h_pad = math.ceil((math.ceil(ho * wo / 16) * 16 - ho * wo) / wo) * strides[3]

        # this version only support the strategy that with out cutting l1.
        # so we put size kernel_d * input_h * input_w into l1 at one time.
        if ksize[2] * (input_shape[3] + l1_h_pad) * input_shape[4] * const.C0_SIZE > \
            tbe_platform.get_soc_spec(tbe_platform.L1_SIZE):
            raise RuntimeError("L1 is not enough, please try a smaller kernel_d, or a smaller input_h or input_w!")
        if wo == 1 and ho != 1 and tbe_platform.get_soc_spec('SHORT_SOC_VERSION') != 'Ascend310':
            raise RuntimeError("Current version don't support W_out = 1 in this environment!")


# 'pylint: disable=unused-argument,too-many-arguments
@register_operator("MaxPool3DWithArgmax")
def max_pool3d_with_argmax(x, y, argmax, kernel_size, strides, pads=((0, 0), (0, 0), (0, 0)),
                           dilation=(1, 1, 1, 1, 1), ceil_mode=False, data_format="NCDHW", argmax_type="bitmask",
                           kernel_name='max_pool3d_with_argmax'):
    """
    interface of max_pool3d_with_argmax.

    Parameters
    ----------
    x: the input tensor dict, include keys: shape and dtype, this version only support float16.
    y: the output tensor dict, include keys: shape and dtype, dtype is consistent with x.
    argmax: the output tensor, represent the argmax bitmask, format as NC1HWC0,
        actually, it represent (N, Do, C1*k_size, Ho*Wo//16, 16).
    kernel_size: list of kernel_size, value of (1, 1, kernel_d, kernel_h, kernel_w).
    strides: list of stride_size, value of (1, 1, stride_d, stride_h, stride_w).
    pads: list of padding, spread out the six-directions of padding.
    dilation: list of dilation, value of (1, 1, dilation_d, dilation_h, dilation_w).
    ceil_mode: reserved field, current version only support ceil_mode=False.
    data_format: the format of origin input, default value for torch is "NCDHW"
    argmax_type: reserved field, determine whether the argmax is img2col mode or torch-output mode.
        current version only support argmax_type="bitmask".
    kernel_name: max_pool3d_with_argmax

    Returns
    -------
    tik_instance
    """
    _check_param(x, kernel_size, strides, pads, dilation, ceil_mode, argmax_type)
    check_load3d_support = tbe_platform.api_check_support("tik.load3dv1")
    if check_load3d_support:
        obj = MaxPool3DWithArgmaxWholeKernel(x, kernel_size, strides, kernel_name)
    else:
        obj = MaxPool3DWithArgmaxMilan(x, kernel_size, strides, kernel_name)
    return obj.run()
