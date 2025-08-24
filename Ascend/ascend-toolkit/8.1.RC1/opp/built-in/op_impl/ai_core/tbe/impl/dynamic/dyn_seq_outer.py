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
dyn_seq_outer
"""

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from tbe.common.platform.platform_info import get_soc_spec


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    BLOCK_ALIGN = 16
    MAX_INT32 = 2 ** 31 - 1
    BATCH_MAX = 256
    OFFSET_NUMS = 3


@register_operator("dyn_seq_outer")
class DynSeqOuter(object):
    def __init__(self, enc_chunk_data, dec_chunk_data, frame_size, beam_size, y, kernel_name):
        self.kernel_name = kernel_name
        self.dtype = enc_chunk_data.get("dtype").lower()
        self.stride = 1 if self.dtype == "float16" else 2
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.version = get_soc_spec("SOC_VERSION")
        self.core_type = get_soc_spec("AICORE_TYPE")
        self.core_num = tik.Dprofile().get_aicore_num()
        self.core_num_var = self.tik_instance.Scalar(dtype="int32", name="core_num_var", init_value=self.core_num)

        self.enc_chunk_data_gm = self.tik_instance.Tensor(self.dtype, [Constant.MAX_INT32], name="enc_chunk_data",
                                                          scope=tik.scope_gm)
        self.dec_chunk_data_gm = self.tik_instance.Tensor(self.dtype, [Constant.MAX_INT32], name="dec_chunk_data",
                                                          scope=tik.scope_gm)
        self.beam_size_gm = self.tik_instance.Tensor(self.dtype, [Constant.MAX_INT32], name="beam_size",
                                                     scope=tik.scope_gm)
        self.frame_size_gm = self.tik_instance.Tensor(self.dtype, [Constant.MAX_INT32], name="frame_size",
                                                      scope=tik.scope_gm)

        self.tiling_gm = self.tik_instance.Tensor("int32", [Constant.BLOCK_ALIGN], name="tiling_gm",
                                                  scope=tik.scope_gm)
        self.y_gm = self.tik_instance.Tensor(self.dtype, [Constant.MAX_INT32], name="y", scope=tik.scope_gm,
                                             is_atomic_add=True)

        self.offset_gm = self.tik_instance.Tensor("int32", [Constant.OFFSET_NUMS * Constant.BATCH_MAX],
                                                  name="offset_gm", scope=tik.scope_gm, is_workspace=True)

        self.batch_size = self.tik_instance.Scalar("int32")
        self.batch_size_align = self.tik_instance.Scalar("int32")

        self.feature_dim = self.tik_instance.Scalar("int32")
        self.feature_dim_align = self.tik_instance.Scalar("int32")

        self.get_tiling_args()

        self.batch_size_align.set_as(
            (self.batch_size + 2 * Constant.BLOCK_ALIGN - 1) // Constant.BLOCK_ALIGN * Constant.BLOCK_ALIGN)
        self.feature_dim_align.set_as(
            (self.feature_dim + Constant.BLOCK_ALIGN - 1) // Constant.BLOCK_ALIGN * Constant.BLOCK_ALIGN)

    def get_tiling_args(self):
        """get_tiling_args"""
        tiling_ub = self.tik_instance.Tensor("int32", [Constant.BLOCK_ALIGN],
                                             name='tiling_ub', scope=tik.scope_ubuf)
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, 1, 0, 0)
        self.batch_size.set_as(tiling_ub[0])
        self.feature_dim.set_as(tiling_ub[1])
        self.core_num_var.set_as(tiling_ub[2])

    def dyn_seq_outer_compute(self):
        """
        To do: Implement the operator by referring to the
               TBE Operator Development Guide.
        """
        enc_offset_ub = self.tik_instance.Tensor("int32", [self.batch_size_align], name="enc_offset_ub",
                                                 scope=tik.scope_ubuf)
        dec_offset_ub = self.tik_instance.Tensor("int32", [self.batch_size_align], name="dec_offset_ub",
                                                 scope=tik.scope_ubuf)
        beam_size_ub = self.tik_instance.Tensor("int32", [self.batch_size_align], name="beam_size_ub",
                                                scope=tik.scope_ubuf)
        frame_size_ub = self.tik_instance.Tensor("int32", [self.batch_size_align], name="frame_size_ub",
                                                 scope=tik.scope_ubuf)
        idx_offset_ub = self.tik_instance.Tensor("int32", [self.batch_size_align], name="idx_offset_ub",
                                                 scope=tik.scope_ubuf)

        self.tik_instance.data_move(beam_size_ub, self.beam_size_gm, 0, 1,
                                    2 * self.batch_size_align // Constant.BLOCK_ALIGN, 0, 0)
        self.tik_instance.data_move(frame_size_ub, self.frame_size_gm, 0, 1,
                                    2 * self.batch_size_align // Constant.BLOCK_ALIGN, 0, 0)

        enc_offset = self.tik_instance.Scalar("int32", init_value=0)
        dec_offset = self.tik_instance.Scalar("int32", init_value=0)
        idx_offset = self.tik_instance.Scalar("int32", init_value=0)

        enc_offset_ub[0].set_as(enc_offset)
        dec_offset_ub[0].set_as(dec_offset)
        idx_offset_ub[0].set_as(idx_offset)

        current_beam = self.tik_instance.Scalar("int32")
        current_frame = self.tik_instance.Scalar("int32")
        with self.tik_instance.for_range(0, self.batch_size) as idx:
            current_beam.set_as(beam_size_ub[idx])
            current_frame.set_as(frame_size_ub[idx])

            enc_offset.set_as(enc_offset + current_frame)

            dec_offset.set_as(dec_offset + current_beam)
            idx_offset.set_as(idx_offset + current_beam * current_frame)

            dec_offset_ub[idx + 1].set_as(dec_offset)
            enc_offset_ub[idx + 1].set_as(enc_offset)
            idx_offset_ub[idx + 1].set_as(idx_offset)

        self.tik_instance.data_move(self.offset_gm, enc_offset_ub, 0, 1,
                                    2 * self.batch_size_align // Constant.BLOCK_ALIGN, 0, 0)
        self.tik_instance.data_move(self.offset_gm[Constant.BATCH_MAX], dec_offset_ub, 0, 1,
                                    2 * self.batch_size_align // Constant.BLOCK_ALIGN, 0, 0)
        self.tik_instance.data_move(self.offset_gm[Constant.BATCH_MAX * 2], idx_offset_ub, 0, 1,
                                    2 * self.batch_size_align // Constant.BLOCK_ALIGN, 0, 0)

        batch_num_per_aicore = self.tik_instance.Scalar("int32", init_value=self.batch_size // self.core_num_var)
        batch_tail = self.tik_instance.Scalar("int32", init_value=self.batch_size % self.core_num_var)

        with self.tik_instance.for_range(0, self.core_num_var, block_num=self.core_num_var) as i:
            with self.tik_instance.for_range(0, batch_num_per_aicore) as j:
                self.dyn_seq_outer_compute_core(i + j * self.core_num_var)
            with self.tik_instance.if_scope(i < batch_tail):
                self.dyn_seq_outer_compute_core(batch_num_per_aicore * self.core_num_var + i)

        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        tbe_context.get_context().add_compile_info(
            "vars", {
                "core_num": self.core_num,
            })
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.enc_chunk_data_gm, self.dec_chunk_data_gm,
                                           self.frame_size_gm, self.beam_size_gm],
                                   outputs=[self.y_gm], flowtable=[self.tiling_gm], config=opt_config)

        return self.tik_instance

    def dyn_seq_outer_compute_core(self, task_idx):
        enc_offset_ub = self.tik_instance.Tensor("int32", [Constant.BLOCK_ALIGN], name="enc_offset_ub",
                                                 scope=tik.scope_ubuf)
        dec_offset_ub = self.tik_instance.Tensor("int32", [Constant.BLOCK_ALIGN], name="dec_offset_ub",
                                                 scope=tik.scope_ubuf)
        idx_offset_ub = self.tik_instance.Tensor("int32", [Constant.BLOCK_ALIGN], name="idx_offset_ub",
                                                 scope=tik.scope_ubuf)

        self.tik_instance.data_move(enc_offset_ub, self.offset_gm[task_idx], 0, 1, 1, 0, 0)
        self.tik_instance.data_move(dec_offset_ub, self.offset_gm[Constant.BATCH_MAX + task_idx], 0, 1, 1, 0, 0)
        self.tik_instance.data_move(idx_offset_ub, self.offset_gm[Constant.BATCH_MAX * 2 + task_idx], 0, 1, 1, 0, 0)

        enc_offset = self.tik_instance.Scalar("int32", init_value=enc_offset_ub[0])
        dec_offset = self.tik_instance.Scalar("int32", init_value=dec_offset_ub[0])
        idx_offset = self.tik_instance.Scalar("int32", init_value=idx_offset_ub[0])

        enc_ub = self.tik_instance.Tensor(self.dtype, [self.feature_dim_align], name="enc_ub", scope=tik.scope_ubuf)
        dec_ub = self.tik_instance.Tensor(self.dtype, [self.feature_dim_align], name="dec_ub", scope=tik.scope_ubuf)
        y_ub = self.tik_instance.Tensor(self.dtype, [self.feature_dim_align], name="y_ub", scope=tik.scope_ubuf)

        task_num = self.tik_instance.Scalar("int32", init_value=idx_offset_ub[1])
        task_num.set_as(task_num - idx_offset)

        frame_size_ub = self.tik_instance.Tensor("int32", [self.batch_size_align], name="frame_size_ub",
                                                 scope=tik.scope_ubuf)
        self.tik_instance.data_move(frame_size_ub, self.frame_size_gm, 0, 1, self.batch_size_align // 8, 0, 0)
        frame_size = self.tik_instance.Scalar("int32", init_value=frame_size_ub[task_idx])

        with self.tik_instance.for_range(0, task_num) as idx:
            self.tik_instance.data_move(dec_ub,
                                        self.dec_chunk_data_gm[(dec_offset + idx // frame_size) * self.feature_dim],
                                        0, 1, self.stride * self.feature_dim_align // Constant.BLOCK_ALIGN, 0, 0)

            self.tik_instance.data_move(enc_ub,
                                        self.enc_chunk_data_gm[(enc_offset + idx % frame_size) * self.feature_dim],
                                        0, 1, self.stride * self.feature_dim_align // Constant.BLOCK_ALIGN, 0, 0)

            self.tik_instance.vec_add(Constant.BLOCK_ALIGN, y_ub, enc_ub, dec_ub,
                                      self.stride * self.feature_dim_align // Constant.BLOCK_ALIGN, 1, 1, 1)

            with self.tik_instance.for_range(self.feature_dim, self.feature_dim_align) as unvalid_idx:
                y_ub[unvalid_idx].set_as(0)

            if self.dtype == "float16" and self.version == "Ascend310P" and self.core_type == "AiCore":
                self.tik_instance.set_atomic_add(2)
            elif self.dtype == "float32" and self.version != "Ascend310":
                self.tik_instance.set_atomic_add(1)

            self.tik_instance.data_move(self.y_gm[idx_offset * self.feature_dim], y_ub, 0, 1,
                                        self.stride * self.feature_dim_align // Constant.BLOCK_ALIGN, 0, 0)
            if self.version != "Ascend310":
                self.tik_instance.set_atomic_add(0)

            idx_offset.set_as(idx_offset + 1)


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def dyn_seq_outer(x1, x2, seq_len1, seq_len2, y, kernel_name="dyn_seq_outer"):
    """
    To do: Implement the operator by referring to the
           TBE Operator Development Guide.
    """
    op_obj = DynSeqOuter(x1, x2, seq_len1, seq_len2, y, kernel_name)
    return op_obj.dyn_seq_outer_compute()
