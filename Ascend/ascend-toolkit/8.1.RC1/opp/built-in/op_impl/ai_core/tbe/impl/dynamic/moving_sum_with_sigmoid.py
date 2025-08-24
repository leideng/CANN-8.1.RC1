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
moving_sum_with_sigmoid
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
    REDUCE_ALIGN = 64
    BATCH_MAX = 256
    OFFSET_NUMS = 3

# 'pylint: disable=too-many-instance-attributes
@register_operator("moving_sum_with_sigmoid")
class MovingSumWithSigmoid(object):
    """class for moving_sum_with_sigmoid"""

    # 'pylint: disable=too-many-arguments
    def __init__(self, alpha, energy, offset, y, window_size, kernel_name):
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.kernel_name = kernel_name

        self.dtype = alpha.get("dtype").lower()
        self.conv = True if self.dtype == "float16" else False
        self.version = get_soc_spec("SOC_VERSION")
        self.core_type = get_soc_spec("AICORE_TYPE")
        self.alpha_gm = self.tik_instance.Tensor(self.dtype, [Constant.MAX_INT32], name="alpha_gm", scope=tik.scope_gm)
        self.energy_gm = self.tik_instance.Tensor(self.dtype, [Constant.MAX_INT32], name="energy_gm",
                                                  scope=tik.scope_gm)

        self.offset_gm = self.tik_instance.Tensor("int32", [Constant.MAX_INT32], name="offset_gm", scope=tik.scope_gm)

        self.y_gm = self.tik_instance.Tensor(self.dtype, [Constant.MAX_INT32], name="y", scope=tik.scope_gm,
                                             is_atomic_add=True)

        self.tiling_gm = self.tik_instance.Tensor("int32", [Constant.BLOCK_ALIGN], name="tiling_gm",
                                                  scope=tik.scope_gm)

        self.window_size = window_size
        self.window_size_align = (self.window_size + Constant.BLOCK_ALIGN - 1) // \
                                 Constant.BLOCK_ALIGN * Constant.BLOCK_ALIGN

        self.used_aicore_num = tik.Dprofile().get_aicore_num()
        self.tmp_offset_gm = self.tik_instance.Tensor("int32", [Constant.OFFSET_NUMS * Constant.BATCH_MAX],
                                                      name="tmp_offset_gm",
                                                      scope=tik.scope_gm, is_workspace=True)

        self.batch_size = self.tik_instance.Scalar("int32")
        self.batch_size_align = self.tik_instance.Scalar("int32")
        self.core_num_var = self.tik_instance.Scalar("int32")
        self.col_offset = self.tik_instance.Scalar("int32")

    def get_tiling_args(self):
        """get_tiling_args"""
        tiling_ub = self.tik_instance.Tensor("int32", [Constant.BLOCK_ALIGN],
                                             name='tiling_ub', scope=tik.scope_ubuf)
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, 1, 0, 0)
        self.batch_size.set_as(tiling_ub[0])
        self.core_num_var.set_as(tiling_ub[1])

    def moving_sum_with_sigmoid_compute(self):
        self.get_tiling_args()

        self.batch_size_align.set_as(
            (self.batch_size + Constant.BLOCK_ALIGN - 1) // Constant.BLOCK_ALIGN * Constant.BLOCK_ALIGN)

        alpha_offset_ub = self.tik_instance.Tensor("int32", [self.batch_size_align], name="alpha_offset_ub",
                                                   scope=tik.scope_ubuf)
        energy_row_offset_ub = self.tik_instance.Tensor("int32", [self.batch_size_align], name="energy_row_offset_ub",
                                                        scope=tik.scope_ubuf)
        energy_col_offset_ub = self.tik_instance.Tensor("int32", [self.batch_size_align], name="energy_col_offset_ub",
                                                        scope=tik.scope_ubuf)

        beam_size_ub = self.tik_instance.Tensor("int32", [self.batch_size_align], name="beam_size_ub",
                                                scope=tik.scope_ubuf)
        frame_size_ub = self.tik_instance.Tensor("int32", [self.batch_size_align], name="frame_size_ub",
                                                 scope=tik.scope_ubuf)

        self.tik_instance.data_move(beam_size_ub, self.offset_gm[0], 0, 1,
                                    2 * self.batch_size_align // Constant.BLOCK_ALIGN, 0, 0)
        self.tik_instance.data_move(frame_size_ub, self.offset_gm[self.batch_size], 0, 1,
                                    2 * self.batch_size_align // Constant.BLOCK_ALIGN, 0, 0)

        self.col_offset.set_as(0)
        tmp = self.tik_instance.Scalar("int32")
        with self.tik_instance.for_range(0, self.batch_size) as idx:
            tmp.set_as(frame_size_ub[idx])
            self.col_offset.set_as(self.col_offset + tmp)

        alpha_offset = self.tik_instance.Scalar("int32", init_value=0)

        energy_row_offset = self.tik_instance.Scalar("int32", init_value=0)
        energy_col_offset = self.tik_instance.Scalar("int32", init_value=0)

        alpha_offset_ub[0].set_as(alpha_offset)
        energy_row_offset_ub[0].set_as(energy_row_offset)
        energy_col_offset_ub[0].set_as(energy_col_offset)

        current_beam = self.tik_instance.Scalar("int32")
        current_frame = self.tik_instance.Scalar("int32")
        with self.tik_instance.for_range(0, self.batch_size - 1) as idx:
            current_beam.set_as(beam_size_ub[idx])
            current_frame.set_as(frame_size_ub[idx])

            alpha_offset.set_as(alpha_offset + current_beam * current_frame)
            energy_row_offset.set_as(energy_row_offset + current_beam)
            energy_col_offset.set_as(energy_col_offset + current_frame)

            alpha_offset_ub[idx + 1].set_as(alpha_offset)
            energy_row_offset_ub[idx + 1].set_as(energy_row_offset)
            energy_col_offset_ub[idx + 1].set_as(energy_col_offset)

        self.tik_instance.data_move(self.tmp_offset_gm, alpha_offset_ub, 0, 1,
                                    2 * self.batch_size_align // Constant.BLOCK_ALIGN, 0, 0)
        self.tik_instance.data_move(self.tmp_offset_gm[Constant.BATCH_MAX], energy_row_offset_ub, 0, 1,
                                    2 * self.batch_size_align // Constant.BLOCK_ALIGN, 0, 0)
        self.tik_instance.data_move(self.tmp_offset_gm[Constant.BATCH_MAX * 2], energy_col_offset_ub, 0, 1,
                                    2 * self.batch_size_align // Constant.BLOCK_ALIGN, 0, 0)

        batch_num_per_aicore = self.tik_instance.Scalar("int32", init_value=self.batch_size // self.core_num_var)
        batch_tail = self.tik_instance.Scalar("int32", init_value=self.batch_size % self.core_num_var)

        with self.tik_instance.for_range(0, self.core_num_var, block_num=self.core_num_var) as i:
            with self.tik_instance.for_range(0, batch_num_per_aicore) as j:
                self.moving_sum_with_sigmoid_compute_core(i + j * self.core_num_var)
            with self.tik_instance.if_scope(i < batch_tail):
                self.moving_sum_with_sigmoid_compute_core(batch_num_per_aicore * self.core_num_var + i)

        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        tbe_context.get_context().add_compile_info(
            "vars", {
                "core_num": self.used_aicore_num,
            })
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.alpha_gm, self.energy_gm, self.offset_gm],
                                   outputs=[self.y_gm], flowtable=[self.tiling_gm], config=opt_config)

        return self.tik_instance

    def moving_sum_with_sigmoid_compute_core(self, task_idx):
        alpha_offset_ub = self.tik_instance.Tensor("int32", [Constant.BLOCK_ALIGN], name="alpha_offset_ub",
                                                   scope=tik.scope_ubuf)
        energy_row_offset_ub = self.tik_instance.Tensor("int32", [Constant.BLOCK_ALIGN], name="energy_row_offset_ub",
                                                        scope=tik.scope_ubuf)
        energy_col_offset_ub = self.tik_instance.Tensor("int32", [Constant.BLOCK_ALIGN], name="energy_col_offset_ub",
                                                        scope=tik.scope_ubuf)
        beam_size_ub = self.tik_instance.Tensor("int32", [Constant.BLOCK_ALIGN], name="beam_size_ub",
                                                scope=tik.scope_ubuf)
        frame_size_ub = self.tik_instance.Tensor("int32", [Constant.BLOCK_ALIGN], name="frame_size_ub",
                                                 scope=tik.scope_ubuf)

        self.tik_instance.data_move(alpha_offset_ub, self.tmp_offset_gm[task_idx], 0, 1, 1, 0, 0)
        self.tik_instance.data_move(energy_row_offset_ub, self.tmp_offset_gm[Constant.BATCH_MAX + task_idx],
                                    0, 1, 1, 0, 0)
        self.tik_instance.data_move(energy_col_offset_ub, self.tmp_offset_gm[Constant.BATCH_MAX * 2 + task_idx],
                                    0, 1, 1, 0, 0)
        self.tik_instance.data_move(beam_size_ub, self.offset_gm[task_idx], 0, 1, 1, 0, 0)
        self.tik_instance.data_move(frame_size_ub, self.offset_gm[task_idx + self.batch_size], 0, 1, 1, 0, 0)

        alpha_offset = self.tik_instance.Scalar("int32", init_value=alpha_offset_ub[0])
        energy_row_offset = self.tik_instance.Scalar("int32", init_value=energy_row_offset_ub[0])
        energy_col_offset = self.tik_instance.Scalar("int32", init_value=energy_col_offset_ub[0])
        beam_size = self.tik_instance.Scalar("int32", init_value=beam_size_ub[0])
        frame_size = self.tik_instance.Scalar("int32", init_value=frame_size_ub[0])
        frame_size_align = self.tik_instance.Scalar("int32")
        frame_size_align.set_as((frame_size + Constant.BLOCK_ALIGN - 1) // Constant.BLOCK_ALIGN * Constant.BLOCK_ALIGN)

        alpha_ub = self.tik_instance.Tensor("float32", [frame_size_align], name="alpha_ub",
                                            scope=tik.scope_ubuf)
        energy_ub = self.tik_instance.Tensor("float32", [frame_size_align], name="energy_ub",
                                             scope=tik.scope_ubuf)
        y_ub = self.tik_instance.Tensor("float32", [frame_size_align], name="y_ub", scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, beam_size) as beam_idx:
            if self.conv:
                alpha_ub_fp16 = self.tik_instance.Tensor("float16", [frame_size_align], name="alpha_ub_fp16",
                                                         scope=tik.scope_ubuf)
                energy_ub_fp16 = self.tik_instance.Tensor("float16", [frame_size_align], name="energy_ub_fp16",
                                                          scope=tik.scope_ubuf)
                self.tik_instance.data_move(alpha_ub_fp16, self.alpha_gm[alpha_offset + beam_idx * frame_size], 0, 1,
                                            frame_size_align // Constant.BLOCK_ALIGN, 0, 0)
                self.tik_instance.data_move(energy_ub_fp16, self.energy_gm[alpha_offset + beam_idx * frame_size],
                                            0, 1, frame_size_align // Constant.BLOCK_ALIGN, 0, 0)
                self.tik_instance.vec_conv(Constant.BLOCK_ALIGN, "none", alpha_ub, alpha_ub_fp16,
                                           frame_size_align // Constant.BLOCK_ALIGN, 2, 1)
                self.tik_instance.vec_conv(Constant.BLOCK_ALIGN, "none", energy_ub, energy_ub_fp16,
                                           frame_size_align // Constant.BLOCK_ALIGN, 2, 1)
            else:
                self.tik_instance.data_move(alpha_ub, self.alpha_gm[alpha_offset + beam_idx * frame_size], 0, 1,
                                            2 * frame_size_align // Constant.BLOCK_ALIGN, 0, 0)
                self.tik_instance.data_move(energy_ub, self.energy_gm[alpha_offset + beam_idx * frame_size], 0, 1,
                                            2 * frame_size_align // Constant.BLOCK_ALIGN, 0, 0)

            ones_ub = self.tik_instance.Tensor("float32", [frame_size_align], name="ones_ub", scope=tik.scope_ubuf)
            zero_ub = self.tik_instance.Tensor("float32", [frame_size_align], name="zero_ub", scope=tik.scope_ubuf)
            tmp_ub = self.tik_instance.Tensor("float32", [frame_size_align], name="tmp_ub", scope=tik.scope_ubuf)
            sigmoid_ub = self.tik_instance.Tensor("float32", [frame_size_align], name="sigmoid_ub",
                                                  scope=tik.scope_ubuf)
            sum_ub = self.tik_instance.Tensor("float32", [frame_size_align], name="sum_ub", scope=tik.scope_ubuf)
            work_tensor_ub = self.tik_instance.Tensor("float32", [frame_size_align], name="work_tensor_ub",
                                                      scope=tik.scope_ubuf)
            self.tik_instance.vector_dup(Constant.BLOCK_ALIGN, ones_ub, 1, frame_size_align // Constant.BLOCK_ALIGN, 1,
                                         2)
            self.tik_instance.vector_dup(Constant.BLOCK_ALIGN, zero_ub, 0, frame_size_align // Constant.BLOCK_ALIGN, 1,
                                         2)

            # func '1 / (1 + np.exp(-x))'
            if self.version == "Ascend310":
                exp_ub = self.tik_instance.Tensor("float16", [frame_size_align], name="exp_ub", scope=tik.scope_ubuf)
                work_ub = self.tik_instance.Tensor("float16", [frame_size_align], name="work_ub", scope=tik.scope_ubuf)
                tmp_ub_ = self.tik_instance.Tensor("float32", [frame_size_align], name="tmp_ub_", scope=tik.scope_ubuf)

                self.tik_instance.vec_sub(Constant.BLOCK_ALIGN, tmp_ub, zero_ub, energy_ub,
                                          frame_size_align // Constant.BLOCK_ALIGN, 2, 2, 2)

                self.tik_instance.vec_conv(Constant.BLOCK_ALIGN, "none", work_ub, tmp_ub,
                                           frame_size_align // Constant.BLOCK_ALIGN, 1, 2)
                self.tik_instance.vec_exp(Constant.BLOCK_ALIGN, exp_ub, work_ub,
                                          frame_size_align // Constant.BLOCK_ALIGN,
                                          1, 1)
                self.tik_instance.vec_conv(Constant.BLOCK_ALIGN, "none", tmp_ub, exp_ub,
                                           frame_size_align // Constant.BLOCK_ALIGN, 2, 1)

                self.tik_instance.vec_add(Constant.BLOCK_ALIGN, tmp_ub, tmp_ub, ones_ub,
                                          frame_size_align // Constant.BLOCK_ALIGN, 2, 2, 2)
                self.tik_instance.vec_rec_high_preci(Constant.BLOCK_ALIGN, sigmoid_ub, tmp_ub, work_tensor_ub,
                                                     2 * frame_size_align // Constant.BLOCK_ALIGN, 2, 2)
                block_len = self.tik_instance.Scalar("int32")
                with self.tik_instance.for_range(0, frame_size) as idx:
                    with self.tik_instance.if_scope(frame_size - idx > self.window_size):
                        block_len.set_as(self.window_size + idx)
                    with self.tik_instance.else_scope():
                        block_len.set_as(frame_size)

                    with self.tik_instance.if_scope(block_len > Constant.REDUCE_ALIGN):
                        with self.tik_instance.if_scope(block_len % Constant.REDUCE_ALIGN > 0):
                            self.tik_instance.vec_reduce_add(Constant.REDUCE_ALIGN, tmp_ub_, alpha_ub, work_tensor_ub,
                                                             block_len // Constant.REDUCE_ALIGN, 8)
                            tmp_val = self.tik_instance.Scalar("float32", init_value=tmp_ub_[0])
                            self.tik_instance.vec_reduce_add(block_len % Constant.REDUCE_ALIGN, tmp_ub_,
                                                             alpha_ub[block_len - (block_len % Constant.REDUCE_ALIGN)],
                                                             work_tensor_ub, 1, 0)
                            tmp_ub_[1].set_as(tmp_val)
                            self.tik_instance.vec_reduce_add(2, tmp_ub, tmp_ub_, work_tensor_ub, 1, 0)
                        with self.tik_instance.else_scope():
                            self.tik_instance.vec_reduce_add(Constant.REDUCE_ALIGN, tmp_ub, alpha_ub, work_tensor_ub,
                                                             block_len // Constant.REDUCE_ALIGN, 8)

                    with self.tik_instance.else_scope():
                        self.tik_instance.vec_reduce_add(block_len, tmp_ub, alpha_ub, work_tensor_ub, 1, 0)
                    sum_ub[idx].set_as(tmp_ub[0])
                    alpha_ub[idx].set_as(0)
            else:
                tmp_val = self.tik_instance.Scalar("float32")
                sum_val = self.tik_instance.Scalar("float32")
                exp_ub = self.tik_instance.Tensor("float32", [frame_size_align], name="exp_ub", scope=tik.scope_ubuf)
                self.tik_instance.vec_sub(Constant.BLOCK_ALIGN, tmp_ub, zero_ub, energy_ub,
                                          frame_size_align // Constant.BLOCK_ALIGN, 2, 2, 2)
                self.tik_instance.vec_exp(Constant.BLOCK_ALIGN, exp_ub, tmp_ub,
                                          frame_size_align // Constant.BLOCK_ALIGN, 2, 2)
                self.tik_instance.vec_add(Constant.BLOCK_ALIGN, tmp_ub, exp_ub, ones_ub,
                                          frame_size_align // Constant.BLOCK_ALIGN, 2, 2, 2)
                self.tik_instance.vec_rec_high_preci(Constant.BLOCK_ALIGN, sigmoid_ub, tmp_ub, work_tensor_ub,
                                                     2 * frame_size_align // Constant.BLOCK_ALIGN, 2, 2)

                with self.tik_instance.if_scope(frame_size > self.window_size):
                    with self.tik_instance.if_scope(self.window_size > Constant.REDUCE_ALIGN):
                        with self.tik_instance.if_scope(self.window_size % Constant.REDUCE_ALIGN > 0):
                            self.tik_instance.vec_reduce_add(Constant.REDUCE_ALIGN, sum_ub, alpha_ub, work_tensor_ub,
                                                             self.window_size // Constant.REDUCE_ALIGN, 8)
                            tmp_val.set_as(sum_ub[0])
                            self.tik_instance.vec_reduce_add(self.window_size % Constant.REDUCE_ALIGN, sum_ub,
                                                             alpha_ub[
                                                                 self.window_size - (
                                                                         self.window_size % Constant.REDUCE_ALIGN)],
                                                             work_tensor_ub, 1, 0)
                            sum_val.set_as(sum_ub[0])
                            sum_ub[0].set_as(sum_val + tmp_val)
                        with self.tik_instance.else_scope():
                            self.tik_instance.vec_reduce_add(Constant.REDUCE_ALIGN, sum_ub, alpha_ub, work_tensor_ub,
                                                             self.window_size // Constant.REDUCE_ALIGN, 8)
                    with self.tik_instance.else_scope():
                        self.tik_instance.vec_reduce_add(self.window_size, sum_ub, alpha_ub, work_tensor_ub, 1, 0)
                with self.tik_instance.else_scope():
                    with self.tik_instance.if_scope(frame_size > Constant.REDUCE_ALIGN):
                        with self.tik_instance.if_scope(frame_size % Constant.REDUCE_ALIGN > 0):
                            self.tik_instance.vec_reduce_add(Constant.REDUCE_ALIGN, sum_ub, alpha_ub, work_tensor_ub,
                                                             frame_size // Constant.REDUCE_ALIGN, 8)
                            tmp_val.set_as(sum_ub[0])
                            self.tik_instance.vec_reduce_add(frame_size % Constant.REDUCE_ALIGN, sum_ub,
                                                             alpha_ub[
                                                                 frame_size -
                                                                 frame_size % Constant.REDUCE_ALIGN],
                                                             work_tensor_ub, 1, 0)
                            sum_val.set_as(sum_ub[0])
                            sum_ub[0].set_as(sum_val + tmp_val)
                        with self.tik_instance.else_scope():
                            self.tik_instance.vec_reduce_add(Constant.REDUCE_ALIGN, sum_ub, alpha_ub, work_tensor_ub,
                                                             frame_size // Constant.REDUCE_ALIGN, 8)

                    with self.tik_instance.else_scope():
                        self.tik_instance.vec_reduce_add(frame_size, sum_ub, alpha_ub, work_tensor_ub, 1, 1)

                sum_val.set_as(sum_ub[0])
                with self.tik_instance.for_range(1, frame_size) as idx:
                    tmp_val.set_as(alpha_ub[idx - 1])
                    sum_val.set_as(sum_val - tmp_val)

                    with self.tik_instance.if_scope(frame_size - idx >= self.window_size):
                        tmp_val.set_as(alpha_ub[idx + self.window_size - 1])
                        sum_val.set_as(sum_val + tmp_val)

                    sum_ub[idx].set_as(sum_val)

            self.tik_instance.vec_mul(Constant.BLOCK_ALIGN, y_ub, sum_ub, sigmoid_ub,
                                      frame_size_align // Constant.BLOCK_ALIGN, 2, 2, 2)

            with self.tik_instance.for_range(frame_size, frame_size_align) as idx:
                y_ub[idx].set_as(0)

            if self.conv:
                y_ub_fp16 = self.tik_instance.Tensor("float16", [frame_size_align], name="y_ub_fp16",
                                                     scope=tik.scope_ubuf)
                self.tik_instance.vec_conv(Constant.BLOCK_ALIGN, "none", y_ub_fp16, y_ub,
                                           frame_size_align // Constant.BLOCK_ALIGN, 1, 2)

                if self.version == "Ascend310P" and self.core_type == "AiCore":
                    self.tik_instance.set_atomic_add(2)
                    self.tik_instance.data_move(
                        self.y_gm[(energy_row_offset + beam_idx) * self.col_offset + energy_col_offset], y_ub_fp16, 0,
                        1, frame_size_align // Constant.BLOCK_ALIGN, 0, 0)
                    self.tik_instance.set_atomic_add(0)
                else:
                    self.tik_instance.data_move(
                        self.y_gm[(energy_row_offset + beam_idx) * self.col_offset + energy_col_offset], y_ub_fp16, 0,
                        1, frame_size_align // Constant.BLOCK_ALIGN, 0, 0)
            else:
                if self.version != "Ascend310":
                    self.tik_instance.set_atomic_add(1)
                    self.tik_instance.data_move(
                        self.y_gm[(energy_row_offset + beam_idx) * self.col_offset + energy_col_offset], y_ub, 0, 1,
                        2 * frame_size_align // Constant.BLOCK_ALIGN, 0, 0)
                    self.tik_instance.set_atomic_add(0)
                else:
                    self.tik_instance.data_move(
                        self.y_gm[(energy_row_offset + beam_idx) * self.col_offset + energy_col_offset], y_ub, 0, 1,
                        2 * frame_size_align // Constant.BLOCK_ALIGN, 0, 0)


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_INT, para_check.KERNEL_NAME)
def moving_sum_with_sigmoid(alpha, energy, offset, y, ksize, kernel_name="moving_sum_with_sigmoid"):
    """
    To do: Implement the operator by referring to the
           TBE Operator Development Guide.
    offset = beam_size + frame_size
    """

    op_obj = MovingSumWithSigmoid(alpha, energy, offset, y, ksize, kernel_name)

    return op_obj.moving_sum_with_sigmoid_compute()
