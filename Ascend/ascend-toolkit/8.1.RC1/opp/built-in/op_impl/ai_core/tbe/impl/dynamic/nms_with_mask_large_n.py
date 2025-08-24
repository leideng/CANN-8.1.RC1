#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
nms_with_mask for large n
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tik
from impl.util import util_common


class Constant:
    """
    The class for constant.
    """
    # vector unit can compute 256 bytes in one cycle
    BYTES_ONE_CYCLE_VECTOR = 256
    MAX_REPEAT = 255
    TILING_PARAMS_NUM = 8
    MASK_FP16 = 128
    MASK_FP32 = 64
    # one proposal contains 8 elements
    PROPOSAL_NUM = 8


class NMSLargeN:
    def __init__(self, input_n, input_dtype, iou_thr):
        self.tik_instance = tik.Tik()

        if input_n == -1:
            self.tiling_gm = self.tik_instance.Tensor(
                "int32", (Constant.TILING_PARAMS_NUM, ), name="tiling_gm", scope=tik.scope_gm
            )
            self.tiling_ub = self.tik_instance.Tensor(
                "int32", (Constant.TILING_PARAMS_NUM, ), name="tiling_ub", scope=tik.scope_ubuf
            )
            self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 1, 0, 0)

            self.origin_n = self.tik_instance.Scalar(dtype="int32", name="boxes_num_scalar")
            self.origin_n.set_as(self.tiling_ub[0])
        else:
            self.origin_n = input_n

        self.is_bf16 = (input_dtype == "bfloat16")
        self.dma_dtype = "float16" if input_dtype in ("float16", "bfloat16") else "float32"
        self.dma_dtype_bytes = 2 if self.dma_dtype == "float16" else 4
        self.vec_dtype = "float16" if input_dtype == "float16" else "float32"
        self.vec_dtype_bytes = 2 if self.vec_dtype == "float16" else 4
        self.max_mask_vec = Constant.MASK_FP16 if self.vec_dtype == "float16" else Constant.MASK_FP32

        self.max_n_align = self._calc_max_n_align()

        # create gm tensor
        self._init_gm()
        # create ub tensor
        self._init_ub()
        # create scalar
        self._init_scalar()

        if iou_thr is None:
            iou_thr = self.tik_instance.Scalar(dtype="float32", name="iou_thr")
            iou_thr.set_as(self.tiling_ub[1])
        self.iou_thr_factor_fp32.set_as(iou_thr / (1 + iou_thr))
        self.iou_thr_factor_for_vec.set_as(self.iou_thr_factor_fp32)

    @staticmethod
    def _cmp_tensor_with_tensor_ctn(instruction, dst, src0, src1, extent=None):
        """
        _cmp_tensor_with_tensor_ctn: general instruction, compute contiously, src0 and src1 are both tensors
        Parameters
        ----------
        dst: dst tensor in ub
        src0: src0 tensor in ub
        src1: src1 tensor in ub
        extent: total compute nums, if is None then compute all tensor nums
        Returns
        -------
        None
        """
        if extent is None:
            extent = src0.size
        vec_dtype_bytes = 2 if src0.dtype == "float16" else 4
        mask = Constant.MASK_FP16 if src0.dtype == "float16" else Constant.MASK_FP32
        repeat_times = extent * vec_dtype_bytes // Constant.BYTES_ONE_CYCLE_VECTOR
        instruction(mask=mask,
                    dst=dst,
                    src0=src0,
                    src1=src1,
                    repeat_times=repeat_times,
                    dst_blk_stride=1,
                    src0_blk_stride=1,
                    src1_blk_stride=1,
                    dst_rep_stride=8,
                    src0_rep_stride=8,
                    src1_rep_stride=8)

    @staticmethod
    def _cmp_tensor_with_scalar_ctn(instruction, dst, src, scalar, extent=None):
        """
        _cmp_tensor_with_scalar_ctn: general instruction, compute tensor with scalar contiously
        Parameters
        ----------
        dst: dst tensor in ub
        src: src tensor in ub
        scalar: scalar
        extent: total compute nums, if is None then compute all tensor nums
        Returns
        -------
        None
        """
        if extent is None:
            extent = src.size
        vec_dtype_bytes = 2 if src.dtype == "float16" else 4
        mask = Constant.MASK_FP16 if src.dtype == "float16" else Constant.MASK_FP32
        repeat_times = extent * vec_dtype_bytes // Constant.BYTES_ONE_CYCLE_VECTOR
        instruction(mask=mask,
                    dst=dst,
                    src=src,
                    scalar=scalar,
                    repeat_times=repeat_times,
                    dst_blk_stride=1,
                    src_blk_stride=1,
                    dst_rep_stride=8,
                    src_rep_stride=8)

    def main_func(self, kernel_name):
        loop_times = self.origin_n // self.max_n_align
        tail_n = self.origin_n % self.max_n_align

        with self.tik_instance.for_range(0, loop_times) as i:
            with self.tik_instance.for_range(i, loop_times) as j:
                with self.tik_instance.if_scope(i == j):
                    self._loop(i, j, diag=True)
                with self.tik_instance.else_scope():
                    self._loop(i, j, diag=False)
                if tail_n > 0:
                    self._loop(i, loop_times, diag=False, tail=True)
        if tail_n > 0:
            self._loop(loop_times, loop_times, diag=True, tail=True)

        self.tik_instance.BuildCCE(
            kernel_name=kernel_name,
            inputs=[self.box_scores_gm],
            outputs=[self.selected_boxes_gm, self.selected_idx_gm, self.selected_mask_gm]
        )
        return self.tik_instance

    def main_func_dynamic(self, kernel_name):
        # add compile info
        tbe_context.get_context().add_compile_info("vars", {"max_boxes_num": 2 ** 31 - 1})

        loop_times = self.origin_n // self.max_n_align
        tail_n = self.origin_n % self.max_n_align

        with self.tik_instance.for_range(0, loop_times) as i:
            with self.tik_instance.for_range(i, loop_times) as j:
                with self.tik_instance.if_scope(i == j):
                    self._loop(i, j, diag=True)
                with self.tik_instance.else_scope():
                    self._loop(i, j, diag=False)
                with self.tik_instance.if_scope(tail_n > 0):
                    self._loop(i, loop_times, diag=False, tail=True)
        with self.tik_instance.if_scope(tail_n > 0):
            self._loop(loop_times, loop_times, diag=True, tail=True)

        self.tik_instance.BuildCCE(
            kernel_name=kernel_name,
            inputs=[self.box_scores_gm],
            outputs=[self.selected_boxes_gm, self.selected_idx_gm, self.selected_mask_gm],
            flowtable=[self.tiling_gm]
        )

    def _init_gm(self):
        n_align = util_common.ceil(self.origin_n, self.max_mask_vec) * self.max_mask_vec
        self.box_scores_gm = self.tik_instance.Tensor(
            self.dma_dtype, (n_align, Constant.PROPOSAL_NUM), scope=tik.scope_gm, name="box_scores_gm"
        )
        self.selected_boxes_gm = self.tik_instance.Tensor(
            self.dma_dtype, (n_align, 5), scope=tik.scope_gm, name="selected_boxes_gm"
        )
        self.selected_idx_gm = self.tik_instance.Tensor(
            "int32", (n_align, ), scope=tik.scope_gm, name="selected_idx_gm"
        )
        self.selected_mask_gm = self.tik_instance.Tensor(
            "uint8", (n_align, ), scope=tik.scope_gm, name="selected_mask_gm"
        )

    def _init_ub(self):
        if self.is_bf16:
            self.box_scores_ub_tmp = self.tik_instance.Tensor(
                self.dma_dtype, (self.max_n_align, Constant.PROPOSAL_NUM),
                scope=tik.scope_ubuf, name="box_scores_ub_tmp"
            )
        self.box_scores_ub = self.tik_instance.Tensor(
            self.vec_dtype, (self.max_n_align, Constant.PROPOSAL_NUM), scope=tik.scope_ubuf, name="box_scores_ub"
        )
        self.selected_boxes_ub = self.tik_instance.Tensor(
            self.dma_dtype, (self.max_n_align, 5), scope=tik.scope_ubuf, name="selected_boxes_ub"
        )
        pattern_ub_dtype = "uint16" if self.vec_dtype == "float16" else "uint32"
        self.pattern_ub = self.tik_instance.Tensor(
            pattern_ub_dtype, shape=(8, ), scope=tik.scope_ubuf, name="pattern_ub"
        )

        self.x1_i_ub = self.tik_instance.Tensor(
            dtype=self.vec_dtype, shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="x1_i_ub"
        )
        self.y1_i_ub = self.tik_instance.Tensor(
            dtype=self.vec_dtype, shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="y1_i_ub"
        )
        self.x2_i_ub = self.tik_instance.Tensor(
            dtype=self.vec_dtype, shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="x2_i_ub"
        )
        self.y2_i_ub = self.tik_instance.Tensor(
            dtype=self.vec_dtype, shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="y2_i_ub"
        )

        self.x1_j_ub = self.tik_instance.Tensor(
            dtype=self.vec_dtype, shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="x1_j_ub"
        )
        self.y1_j_ub = self.tik_instance.Tensor(
            dtype=self.vec_dtype, shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="y1_j_ub"
        )
        self.x2_j_ub = self.tik_instance.Tensor(
            dtype=self.vec_dtype, shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="x2_j_ub"
        )
        self.y2_j_ub = self.tik_instance.Tensor(
            dtype=self.vec_dtype, shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="y2_j_ub"
        )

        self.tmp_x1_ub = self.tik_instance.Tensor(
            dtype=self.vec_dtype, shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="tmp_x1_ub"
        )
        self.tmp_y1_ub = self.tik_instance.Tensor(
            dtype=self.vec_dtype, shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="tmp_y1_ub"
        )
        self.tmp_x2_ub = self.tik_instance.Tensor(
            dtype=self.vec_dtype, shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="tmp_x2_ub"
        )
        self.tmp_y2_ub = self.tik_instance.Tensor(
            dtype=self.vec_dtype, shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="tmp_y2_ub"
        )

        self.tmp_x_ub = self.tik_instance.Tensor(
            dtype=self.vec_dtype, shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="tmp_x_ub"
        )
        self.tmp_y_ub = self.tik_instance.Tensor(
            dtype=self.vec_dtype, shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="tmp_y_ub"
        )
        self.areas_ub = self.tik_instance.Tensor(
            dtype=self.vec_dtype, shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="areas_ub"
        )  # record areas when i == j

        self.bitmask_ub = self.tik_instance.Tensor(
            dtype="uint16", shape=(self.max_n_align // 16, ), scope=tik.scope_ubuf, name="bitmask_ub"
        )
        self.selected_idx_ub = self.tik_instance.Tensor(
            dtype="int32", shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="selected_idx_ub"
        )
        self.selected_mask_i_ub = self.tik_instance.Tensor(
            dtype="uint8", shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="selected_mask_i_ub"
        )
        self.selected_mask_i_ub[0] = 1  # The first row must be chosen
        self.selected_mask_j_ub = self.tik_instance.Tensor(
            dtype="uint8", shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="selected_mask_j_ub"
        )
        self.selected_mask_ub_fp16 = self.tik_instance.Tensor(
            dtype="float16", shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="selected_mask_ub_fp16"
        )

    def _init_scalar(self):
        self.iou_thr_factor_fp32 = self.tik_instance.Scalar(dtype="float32", name="iou_thr_factor_fp32")
        self.iou_thr_factor_for_vec = self.tik_instance.Scalar(dtype=self.vec_dtype, name="iou_thr_factor_for_vec")
        self.x1_ik_scalar = self.tik_instance.Scalar(dtype=self.vec_dtype, name="x1_ik_scalar")
        self.y1_ik_scalar = self.tik_instance.Scalar(dtype=self.vec_dtype, name="y1_ik_scalar")
        self.x2_ik_scalar = self.tik_instance.Scalar(dtype=self.vec_dtype, name="x2_ik_scalar")
        self.y2_ik_scalar = self.tik_instance.Scalar(dtype=self.vec_dtype, name="y2_ik_scalar")
        self.areas_scalar = self.tik_instance.Scalar(dtype=self.vec_dtype, name="areas_scalar")
        self.selected_mask_scalar = self.tik_instance.Scalar(dtype="uint8", name="selected_mask_scalar")

    def _calc_max_n_align(self):
        """
        calc max numbers in ub, independent with input
        +--------------------+-----------------+---------+---------------------+
        | ub tensor          | dtype           | lenth   | size(bytes)         |
        +--------------------+-----------------+---------+---------------------+
        | box_scores         | float16/float32 | n * 8   | n * 8 * dtype_bytes |
        +--------------------+-----------------+---------+---------------------+
        | selected_boxes     | float16/float32 | n * 5   | n * 5 * dtype_bytes |
        +--------------------+-----------------+---------+---------------------+
        | pattern            | uint16/uint32   | 8       | 8 * dtype_bytes     |
        +--------------------+-----------------+---------+---------------------+
        | x1_i               | float16/float32 | n       | n * dtype_bytes     |
        +--------------------+-----------------+---------+---------------------+
        | y1_i               | float16/float32 | n       | n * dtype_bytes     |
        +--------------------+-----------------+---------+---------------------+
        | x2_i               | float16/float32 | n       | n * dtype_bytes     |
        +--------------------+-----------------+---------+---------------------+
        | y2_i               | float16/float32 | n       | n * dtype_bytes     |
        +--------------------+-----------------+---------+---------------------+
        | x1_j               | float16/float32 | n       | n * dtype_bytes     |
        +--------------------+-----------------+---------+---------------------+
        | y1_j               | float16/float32 | n       | n * dtype_bytes     |
        +--------------------+-----------------+---------+---------------------+
        | x2_j               | float16/float32 | n       | n * dtype_bytes     |
        +--------------------+-----------------+---------+---------------------+
        | y2_j               | float16/float32 | n       | n * dtype_bytes     |
        +--------------------+-----------------+---------+---------------------+
        | tmp_x1             | float16/float32 | n       | n * dtype_bytes     |
        +--------------------+-----------------+---------+---------------------+
        | tmp_y1             | float16/float32 | n       | n * dtype_bytes     |
        +--------------------+-----------------+---------+---------------------+
        | tmp_x2             | float16/float32 | n       | n * dtype_bytes     |
        +--------------------+-----------------+---------+---------------------+
        | tmp_y2             | float16/float32 | n       | n * dtype_bytes     |
        +--------------------+-----------------+---------+---------------------+
        | tmp_x              | float16/float32 | n       | n * dtype_bytes     |
        +--------------------+-----------------+---------+---------------------+
        | tmp_y              | float16/float32 | n       | n * dtype_bytes     |
        +--------------------+-----------------+---------+---------------------+
        | areas              | float16/float32 | n       | n * dtype_bytes     |
        +--------------------+-----------------+---------+---------------------+
        | bitmask            | uint16          | n // 16 | n // 8              |
        +--------------------+-----------------+---------+---------------------+
        | selected_idx       | int32           | n       | n * 4               |
        +--------------------+-----------------+---------+---------------------+
        | selected_mask_i    | uint8           | n       | n                   |
        +--------------------+-----------------+---------+---------------------+
        | selected_mask_j    | uint8           | n       | n                   |
        +--------------------+-----------------+---------+---------------------+
        | selected_mask_fp16 | float16         | n       | n * 2               |
        +--------------------+-----------------+---------+---------------------+
        outcome in 1971: 23 repeats for fp16, 24 repeats for fp32
        outcome in 1911: 30 repeats for fp16, 32 repeats for fp32
        """
        ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        num_per_repeat = 128  # 128 for fp16, in fp32 case still has fp16 instruction
        max_n_align = (ub_size_bytes - 1 * 1024 - 8 * self.vec_dtype_bytes) // (28 * self.vec_dtype_bytes +
            8.125) // num_per_repeat * num_per_repeat # 1kb for scalar
        return int(max_n_align)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def _vreduce_ctn(self, dst, src0, src1_pattern_value, chosen_nums, extent=None):
        """
        _vreduce_ctn: means do vreduce contiously
        Parameters
        ----------
        dst: dst tensor in ub
        src0: src0 tensor in ub
        src1_pattern_value: pattern value, uint16 for fp16, uint32 for fp32
        chosen_nums: chosen nums per propaosal
        extent: total compute nums, if is None then compute all tensor nums
        Returns
        -------
        None
        """
        pattern_ub = self.pattern_ub
        if self.is_bf16 and src0.dtype == "float16":
            pattern_ub = self.pattern_ub.reinterpret_cast_to("uint16")
        if extent is None:
            extent = src0.size
        self.tik_instance.vector_dup(mask=8, dst=pattern_ub, scalar=src1_pattern_value,
            repeat_times=1, dst_blk_stride=1, dst_rep_stride=8)
        mask = {
            "float16": Constant.MASK_FP16,
            "float32": Constant.MASK_FP32,
            "float": Constant.MASK_FP32}.get(src0.dtype)
        repeat_times = extent // mask
        loop_times = repeat_times // Constant.MAX_REPEAT
        repeat_tail = repeat_times % Constant.MAX_REPEAT
        # normal mode
        with self.tik_instance.for_range(0, loop_times) as idx:
            self.tik_instance.vreduce(mask=mask,
                                      dst=dst[idx * mask *
                                              Constant.MAX_REPEAT // Constant.PROPOSAL_NUM * chosen_nums],
                                      src0=src0[idx * mask * Constant.MAX_REPEAT],
                                      src1_pattern=pattern_ub,
                                      repeat_times=Constant.MAX_REPEAT,
                                      src0_blk_stride=1,
                                      src0_rep_stride=8,
                                      src1_rep_stride=0)
        with self.tik_instance.if_scope(repeat_tail > 0):
            self.tik_instance.vreduce(mask=mask,
                                      dst=dst[loop_times * mask *
                                              Constant.MAX_REPEAT // Constant.PROPOSAL_NUM * chosen_nums],
                                      src0=src0[loop_times * mask * Constant.MAX_REPEAT],
                                      src1_pattern=pattern_ub,
                                      repeat_times=repeat_tail,
                                      src0_blk_stride=1,
                                      src0_rep_stride=8,
                                      src1_rep_stride=0)

    def _loop(self, i, j, diag, tail=False):
        """
        _loop: calc process for one loop
        Parameters
        ----------
        i: tik scalar or int, loop index i
        j: tik scalar or int, loop index j
        diag: bool, True means i is equal to j, False means i is not equal to j
        tail: bool, True means tal process
        Returns
        -------
        None
        """
        n_tail = self.origin_n % self.max_n_align
        extent = self.max_n_align if not tail else util_common.ceil(n_tail, self.max_mask_vec) * self.max_mask_vec
        dma_extent = self.max_n_align if not tail else n_tail
        if diag:
            x1_ub = self.x1_i_ub
            y1_ub = self.y1_i_ub
            x2_ub = self.x2_i_ub
            y2_ub = self.y2_i_ub
            selected_mask_ub = self.selected_mask_i_ub
            areas_ub = self.areas_ub
            if not tail:
                inner_loop_times = self.max_n_align - 1
            else:
                inner_loop_times = n_tail - 1
        else:
            x1_ub = self.x1_j_ub
            y1_ub = self.y1_j_ub
            x2_ub = self.x2_j_ub
            y2_ub = self.y2_j_ub
            selected_mask_ub = self.selected_mask_j_ub
            areas_ub = self.tmp_x_ub
            inner_loop_times = self.max_n_align

        # gen box_scores_ub
        cpgm2ub_dst = self.box_scores_ub_tmp if self.is_bf16 else self.box_scores_ub
        burst_len = dma_extent * Constant.PROPOSAL_NUM * self.dma_dtype_bytes
        self.tik_instance.data_move_pad(dst=cpgm2ub_dst, src=self.box_scores_gm[j * self.max_n_align, 0],
                                        nburst=1, burst=burst_len, dst_gap=0, src_gap=0)

        if diag:
            # gen selected_boxes
            # 7967 is [1 1 1 1 1 0 0 0 1 1 1 1 1 0 0 0] for fp16 inputs
            # 522133279 is [1 1 1 1 1 0 0 0 1 1 1 1 1 0 0 0 1 1 1 1 1 0 0 0 1 1 1 1 1 0 0 0] for fp32 inputs
            pattern_value_boxes = 7967 if self.dma_dtype == "float16" else 522133279
            self._vreduce_ctn(dst=self.selected_boxes_ub,
                              src0=cpgm2ub_dst,
                              src1_pattern_value=pattern_value_boxes,
                              chosen_nums=5,
                              extent=extent * Constant.PROPOSAL_NUM)
            burst_len = dma_extent * 5 * self.dma_dtype_bytes
            self.tik_instance.data_move_pad(dst=self.selected_boxes_gm[j * self.max_n_align, 0], 
                                            src=self.selected_boxes_ub,
                                            nburst=1, burst=burst_len, dst_gap=0, src_gap=0)

            # gen selected_idx
            with self.tik_instance.for_range(0, extent) as i0:
                self.selected_idx_ub[i0].set_as(j * self.max_n_align + i0)
            burst_len = dma_extent * 4
            self.tik_instance.data_move_pad(dst=self.selected_idx_gm[j * self.max_n_align], 
                                            src=self.selected_idx_ub,
                                            nburst=1, burst=burst_len, dst_gap=0, src_gap=0)

        # gen x1_ub, y1_ub, x2_ub, y2_ub
        # 257 is [1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0] for fp16 input
        # 16843009 is [1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0] for fp32 input
        pattern_value_x1 = 257 if self.vec_dtype == "float16" else 16843009
        # 514 is [0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0] for fp16 input
        # 33686018 is [0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0] for fp32 input
        pattern_value_y1 = 514 if self.vec_dtype == "float16" else 33686018
        # 1028 is [0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0] for fp16 input
        # 67372036 is [0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0] for fp32 input
        pattern_value_x2 = 1028 if self.vec_dtype == "float16" else 67372036
        # 2056 is [0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0] for fp16 input
        # 134744072 is [0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0] for fp32 input
        pattern_value_y2 = 2056 if self.vec_dtype == "float16" else 134744072

        if self.is_bf16:
            box_scores_ub = self.box_scores_ub_tmp.reinterpret_cast_to("bfloat16")
            self.tik_instance.vec_conv(
                64, "", self.box_scores_ub, box_scores_ub, extent * Constant.PROPOSAL_NUM // 64, 8, 4)

        self._vreduce_ctn(dst=x1_ub,
                          src0=self.box_scores_ub,
                          src1_pattern_value=pattern_value_x1,
                          chosen_nums=1,
                          extent=extent * Constant.PROPOSAL_NUM)
        self._vreduce_ctn(dst=y1_ub,
                          src0=self.box_scores_ub,
                          src1_pattern_value=pattern_value_y1,
                          chosen_nums=1,
                          extent=extent * Constant.PROPOSAL_NUM)
        self._vreduce_ctn(dst=x2_ub,
                          src0=self.box_scores_ub,
                          src1_pattern_value=pattern_value_x2,
                          chosen_nums=1,
                          extent=extent * Constant.PROPOSAL_NUM)
        self._vreduce_ctn(dst=y2_ub,
                          src0=self.box_scores_ub,
                          src1_pattern_value=pattern_value_y2,
                          chosen_nums=1,
                          extent=extent * Constant.PROPOSAL_NUM)

        # gen selected_mask
        with self.tik_instance.if_scope(i == 0):
            repeat_times = util_common.ceil(extent * 2, Constant.BYTES_ONE_CYCLE_VECTOR)
            self.tik_instance.vector_dup(mask=Constant.MASK_FP16, dst=self.selected_mask_ub_fp16, scalar=1,
                repeat_times=repeat_times, dst_blk_stride=1, dst_rep_stride=8)
        with self.tik_instance.else_scope():
            self.tik_instance.data_move_pad(dst=selected_mask_ub, 
                                src=self.selected_mask_gm[j * self.max_n_align],
                                nburst=1, burst=(dma_extent * 1), dst_gap=0, src_gap=0)

            # vconv uint8 -> fp16
            self.tik_instance.vec_conv(mask=Constant.MASK_FP16,
                                       round_mode='none',
                                       dst=self.selected_mask_ub_fp16,
                                       src=selected_mask_ub,
                                       repeat_times=util_common.ceil(extent * 2, Constant.BYTES_ONE_CYCLE_VECTOR),
                                       dst_rep_stride=8,
                                       src_rep_stride=4)

        with self.tik_instance.for_range(0, inner_loop_times) as k:
            self.selected_mask_scalar.set_as(self.selected_mask_i_ub[k])
            with self.tik_instance.if_scope(self.selected_mask_scalar > 0):
                self.x1_ik_scalar.set_as(self.x1_i_ub[k])
                self.y1_ik_scalar.set_as(self.y1_i_ub[k])
                self.x2_ik_scalar.set_as(self.x2_i_ub[k])
                self.y2_ik_scalar.set_as(self.y2_i_ub[k])

                self._cmp_tensor_with_scalar_ctn(instruction=self.tik_instance.vmaxs,
                                                 dst=self.tmp_x1_ub,
                                                 src=x1_ub,
                                                 scalar=self.x1_ik_scalar,
                                                 extent=extent)
                self._cmp_tensor_with_scalar_ctn(instruction=self.tik_instance.vmaxs,
                                                 dst=self.tmp_y1_ub,
                                                 src=y1_ub,
                                                 scalar=self.y1_ik_scalar,
                                                 extent=extent)
                self._cmp_tensor_with_scalar_ctn(instruction=self.tik_instance.vmins,
                                                 dst=self.tmp_x2_ub,
                                                 src=x2_ub,
                                                 scalar=self.x2_ik_scalar,
                                                 extent=extent)
                self._cmp_tensor_with_scalar_ctn(instruction=self.tik_instance.vmins,
                                                 dst=self.tmp_y2_ub,
                                                 src=y2_ub,
                                                 scalar=self.y2_ik_scalar,
                                                 extent=extent)

                self._cmp_tensor_with_tensor_ctn(instruction=self.tik_instance.vsub,
                                                 dst=self.tmp_x_ub,
                                                 src0=x2_ub,
                                                 src1=x1_ub,
                                                 extent=extent)
                self._cmp_tensor_with_tensor_ctn(instruction=self.tik_instance.vsub,
                                                 dst=self.tmp_y_ub,
                                                 src0=y2_ub,
                                                 src1=y1_ub,
                                                 extent=extent)

                self._cmp_tensor_with_tensor_ctn(instruction=self.tik_instance.vsub,
                                                 dst=self.tmp_x1_ub,
                                                 src0=self.tmp_x2_ub,
                                                 src1=self.tmp_x1_ub,
                                                 extent=extent)
                self._cmp_tensor_with_tensor_ctn(instruction=self.tik_instance.vsub,
                                                 dst=self.tmp_y1_ub,
                                                 src0=self.tmp_y2_ub,
                                                 src1=self.tmp_y1_ub,
                                                 extent=extent)
                self._cmp_tensor_with_tensor_ctn(instruction=self.tik_instance.vmul,
                                                 dst=areas_ub,
                                                 src0=self.tmp_x_ub,
                                                 src1=self.tmp_y_ub,
                                                 extent=extent)
                self.areas_scalar.set_as(self.areas_ub[k])

                self._cmp_tensor_with_scalar_ctn(instruction=self.tik_instance.vmaxs,
                                                 dst=self.tmp_x1_ub,
                                                 src=self.tmp_x1_ub,
                                                 scalar=0,
                                                 extent=extent)
                self._cmp_tensor_with_scalar_ctn(instruction=self.tik_instance.vmaxs,
                                                 dst=self.tmp_y1_ub,
                                                 src=self.tmp_y1_ub,
                                                 scalar=0,
                                                 extent=extent)
                self._cmp_tensor_with_scalar_ctn(instruction=self.tik_instance.vadds,
                                                 dst=self.tmp_x_ub,
                                                 src=areas_ub,
                                                 scalar=self.areas_scalar,
                                                 extent=extent)

                self._cmp_tensor_with_tensor_ctn(instruction=self.tik_instance.vmul,
                                                 dst=self.tmp_x1_ub,
                                                 src0=self.tmp_x1_ub,
                                                 src1=self.tmp_y1_ub,
                                                 extent=extent)
                self._cmp_tensor_with_scalar_ctn(instruction=self.tik_instance.vmuls,
                                                 dst=self.tmp_x_ub,
                                                 src=self.tmp_x_ub,
                                                 scalar=self.iou_thr_factor_for_vec,
                                                 extent=extent)

                repeat_times = extent * self.vec_dtype_bytes // Constant.BYTES_ONE_CYCLE_VECTOR
                self.tik_instance.vcmpv_le(dst=self.bitmask_ub,
                                           src0=self.tmp_x1_ub,
                                           src1=self.tmp_x_ub,
                                           repeat_times=repeat_times,
                                           src0_blk_stride=1,
                                           src1_blk_stride=1,
                                           src0_rep_stride=8,
                                           src1_rep_stride=8)

                self.tik_instance.vsel(mask=Constant.MASK_FP16,
                                       mode=1,
                                       dst=self.selected_mask_ub_fp16,
                                       sel=self.bitmask_ub,
                                       src0=self.selected_mask_ub_fp16,
                                       src1=0,
                                       repeat_times=util_common.ceil(extent * 2, Constant.BYTES_ONE_CYCLE_VECTOR),
                                       dst_blk_stride=1,
                                       src0_blk_stride=1,
                                       src1_blk_stride=1,
                                       dst_rep_stride=8,
                                       src0_rep_stride=8,
                                       src1_rep_stride=8)
                if diag:
                    self.selected_mask_ub_fp16[k] = 1.
                # vconv fp16 -> uint8
                self.tik_instance.vec_conv(mask=Constant.MASK_FP16,
                                           round_mode='none',
                                           dst=selected_mask_ub,
                                           src=self.selected_mask_ub_fp16,
                                           repeat_times=util_common.ceil(extent * 2, Constant.BYTES_ONE_CYCLE_VECTOR),
                                           dst_rep_stride=4,
                                           src_rep_stride=8)

        burst_len = dma_extent * 1
        self.tik_instance.data_move_pad(dst=self.selected_mask_gm[j * self.max_n_align], src=selected_mask_ub,
                                        nburst=1, burst=burst_len, dst_gap=0, src_gap=0)
