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
sorted_nms
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tik
from impl.util import util_common
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector


class Constant:
    """
    The class for constant.
    """
    # vector unit can compute 256 bytes in one cycle
    BYTES_ONE_CYCLE_VECTOR = 256
    TILING_PARAMS_NUM = 8
    MASK_FP16 = 128
    MASK_FP32 = 64
    MAX_INT32 = 2 ** 31 - 1


class SortedNMS:
    def __init__(self, origin_n, input_dtype, offset):
        self.tik_instance = tik.Tik()
        self.aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

        self.origin_n = origin_n
        self.input_dtype = input_dtype
        self.input_dtype_bytes = 2 if self.input_dtype == "float16" else 4
        self.max_mask_vec = Constant.MASK_FP16 if self.input_dtype == "float16" else Constant.MASK_FP32

        self.offset = offset

        self.max_n_align = 1024

        self.pattern_value = {}
        # 1 is [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] for fp16 input
        # 16843009 is [1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0] for fp32 input
        self.pattern_value["x1"] = 1 if self.input_dtype == "float16" else 16843009
        # 2 is [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0] for fp16 input
        # 33686018 is [0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0] for fp32 input
        self.pattern_value["y1"] = 2 if self.input_dtype == "float16" else 33686018
        # 4 is [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0] for fp16 input
        # 67372036 is [0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0] for fp32 input
        self.pattern_value["x2"] = 4 if self.input_dtype == "float16" else 67372036
        # 8 is [0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0] for fp16 input
        # 134744072 is [0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0] for fp32 input
        self.pattern_value["y2"] = 8 if self.input_dtype == "float16" else 134744072

        # create scalar
        self._init_scalar()
        # create gm tensor
        self._init_gm()
        # create ub tensor
        self._init_ub()

        self.loop_parts = util_common.ceil(self.origin_n_scalar, self.max_n_align)  # scalar expr
        self.loop_times = self.loop_parts * (self.loop_parts + 1) // 2

    @staticmethod
    def _cmpt_tensor_with_tensor(instruction, dst, src0, src1, extent=None):
        """
        _cmpt_tensor_with_tensor: general instruction, compute contiously, src0 and src1 are both tensors
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
        input_dtype_bytes = 2 if src0.dtype == "float16" else 4
        mask = Constant.MASK_FP16 if src0.dtype == "float16" else Constant.MASK_FP32
        repeat_times = extent * input_dtype_bytes // Constant.BYTES_ONE_CYCLE_VECTOR
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
    def _cmpt_tensor_with_scalar(instruction, dst, src, scalar, extent=None):
        """
        _cmpt_tensor_with_scalar: general instruction, compute tensor with scalar contiously
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
        input_dtype_bytes = 2 if src.dtype == "float16" else 4
        mask = Constant.MASK_FP16 if src.dtype == "float16" else Constant.MASK_FP32
        repeat_times = extent * input_dtype_bytes // Constant.BYTES_ONE_CYCLE_VECTOR
        instruction(mask=mask,
                    dst=dst,
                    src=src,
                    scalar=scalar,
                    repeat_times=repeat_times,
                    dst_blk_stride=1,
                    src_blk_stride=1,
                    dst_rep_stride=8,
                    src_rep_stride=8)

    def basic_api(self, kernel_name):
        self._read_tensor()

        block_loop_times = self.loop_times // self.aicore_num_scalar
        block_loop_tail = self.loop_times % self.aicore_num_scalar

        with self.tik_instance.for_range(0, self.aicore_num_scalar, block_num=self.aicore_num_scalar) as block_idx:
            with self.tik_instance.for_range(0, block_loop_times) as inner_loop_idx:
                self._loop_process(inner_loop_idx, block_idx)

            with self.tik_instance.if_scope(block_idx < block_loop_tail):
                self._loop_process(block_loop_times, block_idx)

            self.tik_instance.block_barrier(self.sync_workspace)
            with self.tik_instance.if_scope(block_idx == 0):
                self._threshold_process()
                self._move_shape_out()

        tbe_context.get_context().add_compile_info("vars", {
            "compile_core_num": self.aicore_num,
            "max_n_align": self.max_n_align
        })
        self.tik_instance.BuildCCE(
            kernel_name=kernel_name,
            inputs=[self.boxes_gm, self.sorted_scores_gm, self.input_indices_gm,
                self.max_output_size_gm, self.iou_threshold_gm, self.score_threshold_gm],
            outputs=[self.selected_indices_gm, self.shape_out_gm],
            flowtable=[self.tiling_gm]
        )
        return self.tik_instance

    def _init_scalar(self):
        self.origin_n_scalar = self.tik_instance.Scalar(dtype="int32", name="origin_n_scalar")
        self.aicore_num_scalar = self.tik_instance.Scalar(dtype="int32", name="aicore_num_scalar")
        self.max_output_size_scalar = self.tik_instance.Scalar(dtype="int32", name="max_output_size_scalar")
        self.iou_thr_scalar = self.tik_instance.Scalar(dtype=self.input_dtype, name="iou_thr_scalar")
        self.iou_thr_scalar_fp32 = self.tik_instance.Scalar(dtype="float32", name="iou_thr_scalar_fp32")
        self.iou_thr_factor_fp32 = self.tik_instance.Scalar(dtype="float32", name="iou_thr_factor_fp32")
        self.score_thr_scalar = self.tik_instance.Scalar(dtype=self.input_dtype, name="score_thr_scalar")

        self.i_scalar = self.tik_instance.Scalar(dtype="int32", name="i_scalar")
        self.j_scalar = self.tik_instance.Scalar(dtype="int32", name="j_scalar")
        self.index_scalar = self.tik_instance.Scalar(dtype="int32", name="index_scalar")
        self.x1_ik_scalar = self.tik_instance.Scalar(dtype="float32", name="x1_ik_scalar")
        self.y1_ik_scalar = self.tik_instance.Scalar(dtype="float32", name="y1_ik_scalar")
        self.x2_ik_scalar = self.tik_instance.Scalar(dtype="float32", name="x2_ik_scalar")
        self.y2_ik_scalar = self.tik_instance.Scalar(dtype="float32", name="y2_ik_scalar")
        self.areas_scalar = self.tik_instance.Scalar(dtype="float32", name="areas_scalar")

        self.max_loop_scalar = self.tik_instance.Scalar(
            dtype="int32", name="max_loop_scalar", init_value=Constant.MAX_INT32
        )
        self.sync_scalar = self.tik_instance.Scalar(dtype="int32", name="sync_scalar")
        self.sync_workspace_scalar = self.tik_instance.Scalar(dtype="int32", name="sync_workspace_scalar")
        # _threshold_process
        self.origin_extent_scalar = self.tik_instance.Scalar(dtype="int32", name="origin_extent_scalar")
        self.selected_mask_fp16_scalar = self.tik_instance.Scalar(dtype="float16", name="selected_mask_fp16_scalar")
        self.selected_mask_fp32_scalar = self.tik_instance.Scalar(dtype="float32", name="selected_mask_fp32_scalar")
        self.output_extent_scalar = self.tik_instance.Scalar(dtype="uint32", name="output_extent_scalar")
        self.moved_out_scalar = self.tik_instance.Scalar(dtype="int32", name="moved_out_scalar", init_value=0)
        
    def _init_gm(self):
        self.tiling_gm = self.tik_instance.Tensor(
            "int32", (Constant.TILING_PARAMS_NUM, ), scope=tik.scope_gm, name="tiling_gm"
        )
        n_align = util_common.ceil(self.origin_n_scalar, self.max_n_align) * self.max_n_align
        self.boxes_gm = self.tik_instance.Tensor(
            self.input_dtype, (n_align, 4), scope=tik.scope_gm, name="boxes_gm"
        )
        self.sorted_scores_gm = self.tik_instance.Tensor(
            self.input_dtype, (n_align, ), scope=tik.scope_gm, name="sorted_scores_gm"
        )
        self.input_indices_gm = self.tik_instance.Tensor(
            "int32", (n_align, ), scope=tik.scope_gm, name="input_indices_gm"
        )
        self.max_output_size_gm = self.tik_instance.Tensor(
            "int32", (1, ), scope=tik.scope_gm, name="max_output_size_gm"
        )
        self.iou_threshold_gm = self.tik_instance.Tensor(
            self.input_dtype, (1, ), scope=tik.scope_gm, name="iou_threshold_gm"
        )
        self.score_threshold_gm = self.tik_instance.Tensor(
            self.input_dtype, (1, ), scope=tik.scope_gm, name="score_threshold_gm"
        )
        self.selected_indices_gm = self.tik_instance.Tensor(
            "int32", (n_align, ), scope=tik.scope_gm, name="selected_indices_gm"
        )
        self.shape_out_gm = self.tik_instance.Tensor(
            "int32", (9, ), scope=tik.scope_gm, name="shape_out_gm"
        )

        self.workspace_gm = self.tik_instance.Tensor(
            "uint16",
            (Constant.MAX_INT32, self.max_n_align, self.max_n_align // 16),
            scope=tik.scope_gm,
            name="workspace_gm",
            is_workspace=True
        )
        self.selected_mask_workspace = self.tik_instance.Tensor(
            "float16",
            (Constant.MAX_INT32, ),
            scope=tik.scope_gm,
            name="selected_mask_workspace",
            is_workspace=True
        )
        self.sync_workspace = self.tik_instance.Tensor(
            "int64",
            (Constant.MAX_INT32, 32 // 8),
            scope=tik.scope_gm,
            name="barrier_workspace",
            is_workspace=True,
            is_atomic_add=True
        )

    def _init_ub(self):
        self.tiling_ub = self.tik_instance.Tensor(
            "int32", (Constant.TILING_PARAMS_NUM, ), scope=tik.scope_ubuf, name="tiling_ub"
        )
        self.boxes_ub = self.tik_instance.Tensor(
            self.input_dtype, (self.max_n_align, 32 // self.input_dtype_bytes), scope=tik.scope_ubuf, name="boxes_ub"
        )
        self.sorted_scores_ub = self.tik_instance.Tensor(
            self.input_dtype, (self.max_n_align, ), scope=tik.scope_ubuf, name="sorted_scores_ub"
        )
        self.input_indices_ub = self.tik_instance.Tensor(
            "int32", (self.max_n_align, ), scope=tik.scope_ubuf, name="input_indices_ub"
        )
        pattern_ub_dtype = "uint16" if self.input_dtype == "float16" else "uint32"
        self.pattern_ub = self.tik_instance.Tensor(
            pattern_ub_dtype, shape=(8, ), scope=tik.scope_ubuf, name="pattern_ub"
        )

        self.x1_i_ub = self.tik_instance.Tensor(
            dtype="float32", shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="x1_i_ub"
        )
        self.y1_i_ub = self.tik_instance.Tensor(
            dtype="float32", shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="y1_i_ub"
        )
        self.x2_i_ub = self.tik_instance.Tensor(
            dtype="float32", shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="x2_i_ub"
        )
        self.y2_i_ub = self.tik_instance.Tensor(
            dtype="float32", shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="y2_i_ub"
        )

        self.x1_j_ub = self.tik_instance.Tensor(
            dtype="float32", shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="x1_j_ub"
        )
        self.y1_j_ub = self.tik_instance.Tensor(
            dtype="float32", shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="y1_j_ub"
        )
        self.x2_j_ub = self.tik_instance.Tensor(
            dtype="float32", shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="x2_j_ub"
        )
        self.y2_j_ub = self.tik_instance.Tensor(
            dtype="float32", shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="y2_j_ub"
        )

        self.tmp_x1_ub = self.tik_instance.Tensor(
            dtype="float32", shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="tmp_x1_ub"
        )
        self.tmp_y1_ub = self.tik_instance.Tensor(
            dtype="float32", shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="tmp_y1_ub"
        )
        self.tmp_x2_ub = self.tik_instance.Tensor(
            dtype="float32", shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="tmp_x2_ub"
        )
        self.tmp_y2_ub = self.tik_instance.Tensor(
            dtype="float32", shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="tmp_y2_ub"
        )

        self.tmp_x_ub = self.tik_instance.Tensor(
            dtype="float32", shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="tmp_x_ub"
        )
        self.tmp_y_ub = self.tik_instance.Tensor(
            dtype="float32", shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="tmp_y_ub"
        )
        self.areas_i_ub = self.tik_instance.Tensor(
            dtype="float32", shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="areas_i_ub"
        )
        self.areas_j_ub = self.tik_instance.Tensor(
            dtype="float32", shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="areas_j_ub"
        )

        self.bitmask_ub = self.tik_instance.Tensor(
            dtype="uint16", shape=(self.max_n_align // 16, ), scope=tik.scope_ubuf, name="bitmask_ub"
        )
        self.bitmask_ub_uint32 = self.tik_instance.Tensor(
            dtype="uint32", shape=(self.max_n_align // 32, ), scope=tik.scope_ubuf, name="bitmask_ub_uint32"
        )
        self.selected_mask_ub_fp16 = self.tik_instance.Tensor(
            dtype="float16", shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="selected_mask_ub_fp16"
        )
        self.former_selected_mask_ub = self.tik_instance.Tensor(
            dtype="float16", shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="former_selected_mask_ub"
        )
        self.temp_one_ub = self.tik_instance.Tensor(
            dtype="float16", shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="temp_one_ub"
        )
        self.selected_indices_ub = self.tik_instance.Tensor(
            dtype="int32", shape=(self.max_n_align, ), scope=tik.scope_ubuf, name="selected_indices_ub"
        )
        self.shape_out_ub = self.tik_instance.Tensor(
            "int32", (9, ), scope=tik.scope_ubuf, name="shape_out_ub"
        )

    def _read_tensor(self):
        self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 1, 0, 0)
        self.origin_n_scalar.set_as(self.tiling_ub[0])
        self.aicore_num_scalar.set_as(self.tiling_ub[1])
        self.max_output_size_scalar.set_as(self.max_output_size_gm[0])

        self.iou_thr_scalar.set_as(self.iou_threshold_gm[0])
        self.iou_thr_scalar_fp32.set_as(self.iou_thr_scalar)
        self.iou_thr_factor_fp32.set_as(self.iou_thr_scalar_fp32 / (1 + self.iou_thr_scalar_fp32))

        self.score_thr_scalar.set_as(self.score_threshold_gm[0])

    def _loop_process(self, inner_loop_idx, block_idx):
        loop_idx = inner_loop_idx * self.aicore_num_scalar + block_idx

        with self.tik_instance.for_range(0, self.loop_parts) as j:
            with self.tik_instance.if_scope((j + 1) * (j + 2) // 2 > loop_idx):
                self.j_scalar.set_as(j)
                self.tik_instance.tik_break()
        self.i_scalar.set_as(loop_idx - self.j_scalar * (self.j_scalar + 1) // 2)

        self._loop_new(self.i_scalar, self.j_scalar, loop_idx)

    def _loop_new(self, i, j, loop_idx, tail=False):
        """
        _loop_new: calc process for one loop
        Parameters
        ----------
        i: tik scalar or int, loop index i
        j: tik scalar or int, loop index j
        diag: bool, True means i is equal to j, False means i is not equal to j
        tail: bool, True means tail process
        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(i == self.loop_parts - 1):
            self._gen_areas(i, self.x1_i_ub, self.y1_i_ub, self.x2_i_ub, self.y2_i_ub, self.areas_i_ub, last=True)
        with self.tik_instance.else_scope():
            self._gen_areas(i, self.x1_i_ub, self.y1_i_ub, self.x2_i_ub, self.y2_i_ub, self.areas_i_ub, last=False)
        
        with self.tik_instance.if_scope(j == self.loop_parts - 1):
            self._gen_areas(j, self.x1_j_ub, self.y1_j_ub, self.x2_j_ub, self.y2_j_ub, self.areas_j_ub, last=True)
        with self.tik_instance.else_scope():
            self._gen_areas(j, self.x1_j_ub, self.y1_j_ub, self.x2_j_ub, self.y2_j_ub, self.areas_j_ub, last=False)
        
        extent = self.max_n_align
        with self.tik_instance.for_range(0, self.max_n_align) as k:
            self.x1_ik_scalar.set_as(self.x1_i_ub[k])
            self.y1_ik_scalar.set_as(self.y1_i_ub[k])
            self.x2_ik_scalar.set_as(self.x2_i_ub[k])
            self.y2_ik_scalar.set_as(self.y2_i_ub[k])

            self._cmpt_tensor_with_scalar(instruction=self.tik_instance.vmaxs,
                                            dst=self.tmp_x1_ub,
                                            src=self.x1_j_ub,
                                            scalar=self.x1_ik_scalar,
                                            extent=extent)
            self._cmpt_tensor_with_scalar(instruction=self.tik_instance.vmaxs,
                                            dst=self.tmp_y1_ub,
                                            src=self.y1_j_ub,
                                            scalar=self.y1_ik_scalar,
                                            extent=extent)
            self._cmpt_tensor_with_scalar(instruction=self.tik_instance.vmins,
                                            dst=self.tmp_x2_ub,
                                            src=self.x2_j_ub,
                                            scalar=self.x2_ik_scalar,
                                            extent=extent)
            self._cmpt_tensor_with_scalar(instruction=self.tik_instance.vmins,
                                            dst=self.tmp_y2_ub,
                                            src=self.y2_j_ub,
                                            scalar=self.y2_ik_scalar,
                                            extent=extent)

            self._cmpt_tensor_with_tensor(instruction=self.tik_instance.vsub,
                                            dst=self.tmp_x1_ub,
                                            src0=self.tmp_x2_ub,
                                            src1=self.tmp_x1_ub,
                                            extent=extent)
            self._cmpt_tensor_with_tensor(instruction=self.tik_instance.vsub,
                                            dst=self.tmp_y1_ub,
                                            src0=self.tmp_y2_ub,
                                            src1=self.tmp_y1_ub,
                                            extent=extent)

            if self.offset > 0:
                self._cmpt_tensor_with_scalar(instruction=self.tik_instance.vadds,
                                                dst=self.tmp_x1_ub,
                                                src=self.tmp_x1_ub,
                                                scalar=self.offset,
                                                extent=extent)
                self._cmpt_tensor_with_scalar(instruction=self.tik_instance.vadds,
                                                dst=self.tmp_y1_ub,
                                                src=self.tmp_y1_ub,
                                                scalar=self.offset,
                                                extent=extent)

            self._cmpt_tensor_with_scalar(instruction=self.tik_instance.vmaxs,
                                            dst=self.tmp_x1_ub,
                                            src=self.tmp_x1_ub,
                                            scalar=0,
                                            extent=extent)
            self._cmpt_tensor_with_scalar(instruction=self.tik_instance.vmaxs,
                                            dst=self.tmp_y1_ub,
                                            src=self.tmp_y1_ub,
                                            scalar=0,
                                            extent=extent)
            self.areas_scalar.set_as(self.areas_i_ub[k])
            self._cmpt_tensor_with_scalar(instruction=self.tik_instance.vadds,
                                            dst=self.tmp_x_ub,
                                            src=self.areas_j_ub,
                                            scalar=self.areas_scalar,
                                            extent=extent)

            self._cmpt_tensor_with_tensor(instruction=self.tik_instance.vmul,
                                            dst=self.tmp_x1_ub,
                                            src0=self.tmp_x1_ub,
                                            src1=self.tmp_y1_ub,
                                            extent=extent)
            self._cmpt_tensor_with_scalar(instruction=self.tik_instance.vmuls,
                                            dst=self.tmp_x_ub,
                                            src=self.tmp_x_ub,
                                            scalar=self.iou_thr_factor_fp32,
                                            extent=extent)
            with self.tik_instance.if_scope(i == j):
                self.tmp_x1_ub[k] = 0.

            self.tik_instance.vcmpv_le(dst=self.bitmask_ub,
                                        src0=self.tmp_x1_ub,
                                        src1=self.tmp_x_ub,
                                        repeat_times=extent // Constant.MASK_FP32,
                                        src0_blk_stride=1,
                                        src1_blk_stride=1,
                                        src0_rep_stride=8,
                                        src1_rep_stride=8)

            burst = extent // 8 // util_common.BLOCK_SIZE
            self.tik_instance.data_move(dst=self.workspace_gm[loop_idx, k, 0],
                                        src=self.bitmask_ub,
                                        sid=0,
                                        nburst=1,
                                        burst=burst,
                                        src_stride=0,
                                        dst_stride=0)

    # 'pylint: disable=too-many-arguments
    def _gen_areas(self, loop_idx, x1_ub, y1_ub, x2_ub, y2_ub, areas_ub, last):
        extent = self.origin_n_scalar - loop_idx * self.max_n_align if last is True else self.max_n_align
        burst_bytes = extent * 4
        self.tik_instance.data_move_pad(
            self.input_indices_ub,
            self.input_indices_gm[loop_idx * self.max_n_align],
            1, burst_bytes, dst_gap=0, src_gap=0
        )
        with self.tik_instance.for_range(0, extent) as n:
            self.index_scalar.set_as(self.input_indices_ub[n])
            burst_bytes = 4 * self.input_dtype_bytes
            self.tik_instance.data_move_pad(
                self.boxes_ub[n, 0], self.boxes_gm[self.index_scalar, 0], 1, burst_bytes, dst_gap=0, src_gap=0
            )

        if self.input_dtype == "float32":
            self._vreduce_ctn(dst=x1_ub,
                            src0=self.boxes_ub,
                            src1_pattern_value=self.pattern_value.get("x1"),
                            extent=self.max_n_align * 32 // self.input_dtype_bytes)
            self._vreduce_ctn(dst=y1_ub,
                            src0=self.boxes_ub,
                            src1_pattern_value=self.pattern_value.get("y1"),
                            extent=self.max_n_align * 32 // self.input_dtype_bytes)
            self._vreduce_ctn(dst=x2_ub,
                            src0=self.boxes_ub,
                            src1_pattern_value=self.pattern_value.get("x2"),
                            extent=self.max_n_align * 32 // self.input_dtype_bytes)
            self._vreduce_ctn(dst=y2_ub,
                            src0=self.boxes_ub,
                            src1_pattern_value=self.pattern_value.get("y2"),
                            extent=self.max_n_align * 32 // self.input_dtype_bytes)
        else:
            x1_ub_fp16 = x1_ub.reinterpret_cast_to("float16")[self.max_n_align]
            y1_ub_fp16 = y1_ub.reinterpret_cast_to("float16")[self.max_n_align]
            x2_ub_fp16 = x2_ub.reinterpret_cast_to("float16")[self.max_n_align]
            y2_ub_fp16 = y2_ub.reinterpret_cast_to("float16")[self.max_n_align]
            self._vreduce_ctn(dst=x1_ub_fp16,
                            src0=self.boxes_ub,
                            src1_pattern_value=self.pattern_value.get("x1"),
                            extent=self.max_n_align * 32 // self.input_dtype_bytes)
            self._vreduce_ctn(dst=y1_ub_fp16,
                            src0=self.boxes_ub,
                            src1_pattern_value=self.pattern_value.get("y1"),
                            extent=self.max_n_align * 32 // self.input_dtype_bytes)
            self._vreduce_ctn(dst=x2_ub_fp16,
                            src0=self.boxes_ub,
                            src1_pattern_value=self.pattern_value.get("x2"),
                            extent=self.max_n_align * 32 // self.input_dtype_bytes)
            self._vreduce_ctn(dst=y2_ub_fp16,
                            src0=self.boxes_ub,
                            src1_pattern_value=self.pattern_value.get("y2"),
                            extent=self.max_n_align * 32 // self.input_dtype_bytes)
            self.tik_instance.vconv(mask=Constant.MASK_FP32,
                                    round_mode='none',
                                    dst=x1_ub,
                                    src=x1_ub_fp16,
                                    repeat_times=self.max_n_align // Constant.MASK_FP32,
                                    dst_blk_stride=1,
                                    src_blk_stride=1,
                                    dst_rep_stride=8,
                                    src_rep_stride=4)
            self.tik_instance.vconv(mask=Constant.MASK_FP32,
                                    round_mode='none',
                                    dst=y1_ub,
                                    src=y1_ub_fp16,
                                    repeat_times=self.max_n_align // Constant.MASK_FP32,
                                    dst_blk_stride=1,
                                    src_blk_stride=1,
                                    dst_rep_stride=8,
                                    src_rep_stride=4)
            self.tik_instance.vconv(mask=Constant.MASK_FP32,
                                    round_mode='none',
                                    dst=x2_ub,
                                    src=x2_ub_fp16,
                                    repeat_times=self.max_n_align // Constant.MASK_FP32,
                                    dst_blk_stride=1,
                                    src_blk_stride=1,
                                    dst_rep_stride=8,
                                    src_rep_stride=4)
            self.tik_instance.vconv(mask=Constant.MASK_FP32,
                                    round_mode='none',
                                    dst=y2_ub,
                                    src=y2_ub_fp16,
                                    repeat_times=self.max_n_align // Constant.MASK_FP32,
                                    dst_blk_stride=1,
                                    src_blk_stride=1,
                                    dst_rep_stride=8,
                                    src_rep_stride=4)

        self._cmpt_tensor_with_tensor(instruction=self.tik_instance.vsub,
                                      dst=self.tmp_x_ub,
                                      src0=x2_ub,
                                      src1=x1_ub,
                                      extent=self.max_n_align)
        self._cmpt_tensor_with_tensor(instruction=self.tik_instance.vsub,
                                      dst=self.tmp_y_ub,
                                      src0=y2_ub,
                                      src1=y1_ub,
                                      extent=self.max_n_align)

        if self.offset > 0:
            self._cmpt_tensor_with_scalar(instruction=self.tik_instance.vadds,
                                          dst=self.tmp_x_ub,
                                          src=self.tmp_x_ub,
                                          scalar=self.offset,
                                          extent=self.max_n_align)
            self._cmpt_tensor_with_scalar(instruction=self.tik_instance.vadds,
                                          dst=self.tmp_y_ub,
                                          src=self.tmp_y_ub,
                                          scalar=self.offset,
                                          extent=self.max_n_align)

        self._cmpt_tensor_with_tensor(instruction=self.tik_instance.vmul,
                                      dst=areas_ub,
                                      src0=self.tmp_x_ub,
                                      src1=self.tmp_y_ub,
                                      extent=self.max_n_align)

    def _threshold_process(self):
        with self.tik_instance.for_range(0, self.loop_parts) as j:
            with self.tik_instance.if_scope(self.moved_out_scalar < self.max_output_size_scalar):
                # move sorted_scores to ub
                with self.tik_instance.if_scope(j == self.loop_parts - 1):
                    self.origin_extent_scalar.set_as(self.origin_n_scalar - \
                        (self.loop_parts - 1) * self.max_n_align)

                    self.tik_instance.data_move_pad(
                        self.sorted_scores_ub, self.sorted_scores_gm[j * self.max_n_align], 1,
                        burst=self.origin_extent_scalar * self.input_dtype_bytes, dst_gap=0, src_gap=0,
                    )
                with self.tik_instance.else_scope():
                    self.origin_extent_scalar.set_as(self.max_n_align)
                    burst_bytes = self.origin_extent_scalar * self.input_dtype_bytes
                    self.tik_instance.data_move_pad(
                        self.sorted_scores_ub,
                        self.sorted_scores_gm[j * self.max_n_align],
                        1, burst_bytes, dst_gap=0, src_gap=0
                    )

                self.tik_instance.vcmpvs_gt(
                    dst=self.bitmask_ub,
                    src=self.sorted_scores_ub,
                    scalar=self.score_thr_scalar,
                    repeat_times=self.max_n_align * self.input_dtype_bytes // Constant.BYTES_ONE_CYCLE_VECTOR,
                    src_blk_stride=1,
                    src_rep_stride=8
                )
                self.tik_instance.vec_dup(mask=Constant.MASK_FP16,
                                            dst=self.temp_one_ub,
                                            scalar=1,
                                            repeat_times=self.max_n_align * 2 // Constant.BYTES_ONE_CYCLE_VECTOR,
                                            dst_rep_stride=8)

                self.tik_instance.vsel(mask=Constant.MASK_FP16,
                                        mode=1,
                                        dst=self.selected_mask_ub_fp16,
                                        sel=self.bitmask_ub,
                                        src0=self.temp_one_ub,
                                        src1=0,
                                        repeat_times=self.max_n_align * 2 // Constant.BYTES_ONE_CYCLE_VECTOR,
                                        dst_blk_stride=1,
                                        src0_blk_stride=1,
                                        src1_blk_stride=1,
                                        dst_rep_stride=8,
                                        src0_rep_stride=8,
                                        src1_rep_stride=8)
                with self.tik_instance.if_scope(j == self.loop_parts - 1):
                    origin_extent_align = util_common.align(self.origin_extent_scalar, Constant.MASK_FP16)
                    self.tik_instance.vec_dup(mask=Constant.MASK_FP16,
                        dst=self.selected_mask_ub_fp16[origin_extent_align],
                        scalar=0,
                        repeat_times=(self.max_n_align - origin_extent_align) * 2 // Constant.BYTES_ONE_CYCLE_VECTOR,
                        dst_rep_stride=8)
                    with self.tik_instance.for_range(self.origin_extent_scalar, origin_extent_align) as i:
                        self.selected_mask_ub_fp16[i].set_as(0)

                with self.tik_instance.for_range(0, j + 1) as i:
                    with self.tik_instance.if_scope(i == j):
                        self._gen_selected_mask(i, j, self.selected_mask_ub_fp16)
                        self.tik_instance.data_move(dst=self.selected_mask_workspace[i * self.max_n_align],
                                                    src=self.selected_mask_ub_fp16,
                                                    sid=0,
                                                    nburst=1,
                                                    burst=self.max_n_align * 2 // util_common.BLOCK_SIZE,
                                                    src_stride=0,
                                                    dst_stride=0)
                    with self.tik_instance.else_scope():
                        self.tik_instance.data_move(dst=self.former_selected_mask_ub,
                                                    src=self.selected_mask_workspace[i * self.max_n_align],
                                                    sid=0,
                                                    nburst=1,
                                                    burst=self.max_n_align * 2 // util_common.BLOCK_SIZE,
                                                    src_stride=0,
                                                    dst_stride=0)
                        self._gen_selected_mask(i, j, self.former_selected_mask_ub)

                self.tik_instance.vcmpvs_gt(dst=self.bitmask_ub_uint32,
                                            src=self.selected_mask_ub_fp16,
                                            scalar=0,
                                            repeat_times=self.max_n_align * 2 // Constant.BYTES_ONE_CYCLE_VECTOR,
                                            src_blk_stride=1,
                                            src_rep_stride=8)

                self.tik_instance.data_move_pad(
                    self.input_indices_ub,
                    self.input_indices_gm[j * self.max_n_align],
                    1, self.origin_extent_scalar * 4, dst_gap=0, src_gap=0
                )
                self.tik_instance.vreduce(mask=None,
                                            dst=self.selected_indices_ub,
                                            src0=self.input_indices_ub,
                                            src1_pattern=self.bitmask_ub_uint32,
                                            repeat_times=self.max_n_align * 4 // Constant.BYTES_ONE_CYCLE_VECTOR,
                                            src0_blk_stride=1,
                                            src0_rep_stride=8,
                                            src1_rep_stride=1,
                                            rsvd_scalar=self.output_extent_scalar)

                with self.tik_instance.if_scope(self.output_extent_scalar > 0):
                    with self.tik_instance.if_scope(
                        self.output_extent_scalar > self.max_output_size_scalar - self.moved_out_scalar
                    ):
                        self.output_extent_scalar.set_as(self.max_output_size_scalar - self.moved_out_scalar)
                    self.tik_instance.data_move_pad(dst=self.selected_indices_gm[self.moved_out_scalar],
                                                    src=self.selected_indices_ub,
                                                    nburst=1,
                                                    burst=self.output_extent_scalar * 4,
                                                    dst_gap=0,
                                                    src_gap=0)
                    self.moved_out_scalar.set_as(self.moved_out_scalar + self.output_extent_scalar)

    def _gen_selected_mask(self, i, j, selected_mask_ub):
        with self.tik_instance.for_range(0, self.max_n_align) as k:
            self.selected_mask_fp16_scalar.set_as(selected_mask_ub[k])
            self.selected_mask_fp32_scalar.set_as(self.selected_mask_fp16_scalar)
            with self.tik_instance.if_scope(self.selected_mask_fp32_scalar > 0):
                self.tik_instance.data_move(dst=self.bitmask_ub,
                                            src=self.workspace_gm[j * (j + 1) // 2 + i, k, 0],
                                            sid=0,
                                            nburst=1,
                                            burst=self.max_n_align // 8 // util_common.BLOCK_SIZE,
                                            src_stride=0,
                                            dst_stride=0)
                self.tik_instance.vsel(mask=Constant.MASK_FP16,
                                       mode=1,
                                       dst=self.selected_mask_ub_fp16,
                                       sel=self.bitmask_ub,
                                       src0=self.selected_mask_ub_fp16,
                                       src1=0,
                                       repeat_times=self.max_n_align * 2 // Constant.BYTES_ONE_CYCLE_VECTOR,
                                       dst_blk_stride=1,
                                       src0_blk_stride=1,
                                       src1_blk_stride=1,
                                       dst_rep_stride=8,
                                       src0_rep_stride=8,
                                       src1_rep_stride=8)

    def _move_shape_out(self):
        self.shape_out_ub[0].set_as(1)
        self.shape_out_ub[1].set_as(self.moved_out_scalar)
        self.tik_instance.data_move_pad(dst=self.shape_out_gm,
                                        src=self.shape_out_ub,
                                        nburst=1,
                                        burst=2 * 4,
                                        dst_gap=0,
                                        src_gap=0)

    def _vreduce_ctn(self, dst, src0, src1_pattern_value, extent=None):
        """
        _vreduce_ctn: means do vreduce contiously
        Parameters
        ----------
        dst: dst tensor in ub
        src0: src0 tensor in ub
        src1_pattern_value: pattern value, uint16 for fp16, uint32 for fp32
        extent: total compute nums, if is None then compute all Tensor nums
        Returns
        -------
        None
        """
        if extent is None:
            extent = src0.size
        self.tik_instance.vector_dup(mask=8, dst=self.pattern_ub, scalar=src1_pattern_value,
            repeat_times=1, dst_blk_stride=1, dst_rep_stride=8)
        repeat_times = extent * self.input_dtype_bytes // Constant.BYTES_ONE_CYCLE_VECTOR
        ## normal mode
        self.tik_instance.vreduce(mask=self.max_mask_vec, dst=dst, src0=src0, src1_pattern=self.pattern_ub,
            repeat_times=repeat_times, src0_blk_stride=1, src0_rep_stride=8, src1_rep_stride=0)


# 'pylint: disable=unused-argument,too-many-locals,too-many-arguments,inconsistent-return-statements
@register_operator("SortedNMS")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def sorted_nms(boxes, sorted_scores, input_indices,
               max_output_size, iou_threshold, score_threshold,
               selected_indices, offset=0, kernel_name="sorted_nms"):
    """
    algorithm: sorted_nms
    returns the indices of selected boxes

    Parameters
    ----------
    boxes: dict
        2-D shape and dtype of input tensor, only support shape [N, 4]
        including x1, y1, x2, y2 as the coordinates of boxes

    sorted_scores: dict
        1-D shape and dtype of input tensor, only support shape [N]
        scores after descending sorted

    input_indices: dict
        1-D shape and dtype of input tensor, only support shape [N]
        the boxes indices of sorted scores

    max_output_size: dict
        max output sizes

    iou_threshold: dict
        iou threshold

    score_threshold: dict
        score_threshold

    selected_indices: dict
        the indices of output boxes

    offset: int
        0 or 1, default 0

    kernel_name: str
        cce kernel name, default value is "sorted_nms"

    Returns
    -------
    None
    """
    input_shape = boxes.get("shape")
    input_dtype = boxes.get("dtype").lower()

    origin_n = input_shape[0]

    if offset not in [0, 1]:
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "offset", [0, 1], offset)

    nms_v3_ti = SortedNMS(origin_n, input_dtype, offset)
    nms_v3_ti.basic_api(kernel_name)