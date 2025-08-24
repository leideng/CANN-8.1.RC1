#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
rnn_gen_mask
"""

from impl.util.platform_adapter import register_operator, tbe_context, tbe_platform, tik


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """

    TILING_ARG_NUM = 32
    MAX_INT32 = 2 ** 31 - 1
    UB_AVALIB_SIZE = 10240
    BYTES_PER_BLOCK = 32
    BYTES_PER_KB = 1024


# 'pylint: disable=too-many-instance-attributes
class RnnGenMaskV2:
    """class for rnn_gen_mask"""

    # 'pylint: disable=too-many-arguments
    def __init__(self, seq_length, x, seq_mask, hidden_size, kernel_name):
        """__init__"""
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.available_aicore_num = tik.Dprofile().get_aicore_num()

        self.kernel_name = kernel_name
        self.tiling_param_dtype = "int32"
        self.dtype_in = seq_length.get("dtype").lower()
        self.dtype_out = seq_mask.get("dtype").lower()
        self.dtype_x = x.get("dtype").lower()

        self.block = 8  # default float32
        if self.dtype_out == "float16":
            self.block = 16

        self.seq_length_gm = self.tik_instance.Tensor(
            self.dtype_in, [Constant.MAX_INT32], name="seq_length_gm", scope=tik.scope_gm
        )
        self.x_gm = self.tik_instance.Tensor(self.dtype_x, [Constant.MAX_INT32], name="x_gm", scope=tik.scope_gm)

        self.seq_mask_gm = self.tik_instance.Tensor(
            self.dtype_out, [Constant.MAX_INT32], name="seq_mask_gm", scope=tik.scope_gm
        )

        self.tiling_dtype = "int32"
        self.tiling_gm = self.tik_instance.Tensor(
            self.tiling_dtype, (Constant.TILING_ARG_NUM,), name="tiling_gm", scope=tik.scope_gm
        )
        self.hidden_size = self.tik_instance.Scalar(self.tiling_param_dtype, name="hidden_size")
        self.hidden_size_block = (self.hidden_size + self.block - 1) // self.block * self.block
        self.repeat = self.hidden_size_block // self.block
        # unaligned case
        self.temp_gm = self.tik_instance.Tensor(
            self.dtype_out, [Constant.MAX_INT32], name="temp_gm", scope=tik.scope_gm, is_workspace=True
        )
        self.offset = self.tik_instance.Scalar("int32", name="offset")

        self.core_used = self.tik_instance.Scalar(self.tiling_param_dtype, name="core_used")
        self.batch_size = self.tik_instance.Scalar(self.tiling_param_dtype, name="batch_size")
        self.rounds = self.tik_instance.Scalar(self.tiling_param_dtype, name="rounds")
        self.batch_num_per_aicore = self.tik_instance.Scalar(self.tiling_param_dtype, name="batch_num_per_aicore")
        self.batch_tail = self.tik_instance.Scalar(self.tiling_param_dtype, name="batch_tail")

    def get_tiling_args(self):
        """get_tiling_args"""
        tiling_ub = self.tik_instance.Tensor(
            self.tiling_dtype, (Constant.TILING_ARG_NUM,), name="tiling_ub", scope=tik.scope_ubuf
        )
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, Constant.TILING_ARG_NUM // 8, 0, 0)  # 8 for int32

        self.core_used.set_as(tiling_ub[0])
        self.batch_size.set_as(tiling_ub[1])
        self.rounds.set_as(tiling_ub[2])
        self.batch_num_per_aicore.set_as(tiling_ub[3])
        self.batch_tail.set_as(tiling_ub[4])
        self.hidden_size.set_as(tiling_ub[5])
        self.offset.set_as(
            (self.hidden_size * self.batch_size + self.block - 1) // self.block * self.block + self.block
        )

    def compute_core(self, task_id):
        """compute_core"""
        temp_ub = self.tik_instance.Tensor("int32", [Constant.UB_AVALIB_SIZE], name="temp_ub", scope=tik.scope_ubuf)
        batch_size_round_up = self.tik_instance.Scalar("int32", init_value=(self.batch_size + 7) // 8)  # 8 for int32
        self.tik_instance.data_move(temp_ub, self.seq_length_gm[0], 0, 1, batch_size_round_up, 0, 0)
        valid_len = self.tik_instance.Scalar("int32")
        mask_ub_1 = self.tik_instance.Tensor(
            self.dtype_out, [Constant.UB_AVALIB_SIZE], name="mask_ub", scope=tik.scope_ubuf
        )
        mask_ub_0 = self.tik_instance.Tensor(
            self.dtype_out, [Constant.UB_AVALIB_SIZE], name="mask_ub", scope=tik.scope_ubuf
        )
        self.tik_instance.vector_dup(self.block, mask_ub_1, 1, self.repeat, 1, 1)
        self.tik_instance.vector_dup(self.block, mask_ub_0, 0, self.repeat, 1, 1)

        with self.tik_instance.for_range(0, self.batch_size) as batch_idx:
            valid_len.set_as(temp_ub[batch_idx])
            with self.tik_instance.if_scope(valid_len > task_id):
                with self.tik_instance.if_scope(self.hidden_size != self.hidden_size_block):
                    self.tik_instance.data_move(
                        self.temp_gm[task_id * self.offset + batch_idx * self.hidden_size],
                        mask_ub_1,
                        0,
                        1,
                        self.repeat,
                        0,
                        0,
                    )
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(
                        self.seq_mask_gm[task_id * self.hidden_size * self.batch_size + batch_idx * self.hidden_size],
                        mask_ub_1,
                        0,
                        1,
                        self.repeat,
                        0,
                        0,
                    )
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(self.hidden_size != self.hidden_size_block):
                    self.tik_instance.data_move(
                        self.temp_gm[task_id * self.offset + batch_idx * self.hidden_size],
                        mask_ub_0,
                        0,
                        1,
                        self.repeat,
                        0,
                        0,
                    )
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(
                        self.seq_mask_gm[task_id * self.hidden_size * self.batch_size + batch_idx * self.hidden_size],
                        mask_ub_0,
                        0,
                        1,
                        self.repeat,
                        0,
                        0,
                    )

    def data_tune(self):
        """data_tune"""
        ub_size_bytes = (
            (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - Constant.BYTES_PER_KB)
            // Constant.BYTES_PER_BLOCK
            * Constant.BYTES_PER_BLOCK
        )
        ub_num = ub_size_bytes * self.block // Constant.BYTES_PER_BLOCK
        move_rounds = self.tik_instance.Scalar("int32", init_value=self.offset // ub_num)
        move_tail = self.tik_instance.Scalar("int32", init_value=self.offset % ub_num)

        with self.tik_instance.for_range(0, self.rounds) as i:
            with self.tik_instance.for_range(0, move_rounds) as j:
                temp_ub = self.tik_instance.Tensor(self.dtype_out, [ub_num], name="temp_ub", scope=tik.scope_ubuf)
                self.tik_instance.data_move(
                    temp_ub, self.temp_gm[i * self.offset + j * ub_num], 0, 1, ub_num // self.block, 0, 0
                )
                self.tik_instance.data_move(
                    self.seq_mask_gm[i * self.hidden_size * self.batch_size + j * ub_num],
                    temp_ub,
                    0,
                    1,
                    ub_num // self.block,
                    0,
                    0,
                )
            with self.tik_instance.if_scope(move_tail != 0):
                move_tail.set_as((move_tail + self.block - 1) // self.block * self.block)
                temp_ub = self.tik_instance.Tensor(self.dtype_out, [move_tail], name="temp_ub", scope=tik.scope_ubuf)
                self.tik_instance.data_move(
                    temp_ub, self.temp_gm[i * self.offset + move_rounds * ub_num], 0, 1, move_tail // self.block, 0, 0
                )
                self.tik_instance.data_move(
                    self.seq_mask_gm[i * self.hidden_size * self.batch_size + move_rounds * ub_num],
                    temp_ub,
                    0,
                    1,
                    move_tail // self.block,
                    0,
                    0,
                )

    def rnn_gen_mask_compute(self):
        """rnn_gen_mask_compute"""
        self.get_tiling_args()
        with self.tik_instance.for_range(0, self.core_used, block_num=self.core_used) as i:
            with self.tik_instance.if_scope(i < self.core_used):
                with self.tik_instance.for_range(0, self.batch_num_per_aicore) as k:
                    self.compute_core(i + k * self.core_used)
                with self.tik_instance.if_scope(i < self.batch_tail):
                    self.compute_core(i + self.batch_num_per_aicore * self.core_used)

        with self.tik_instance.if_scope(self.hidden_size != self.hidden_size_block):
            self.data_tune()

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.seq_length_gm, self.x_gm],
            outputs=[self.seq_mask_gm],
            flowtable=[self.tiling_gm],
        )
        tbe_context.get_context().add_compile_info("vars", {"available_aicore_num": self.available_aicore_num})
        return self.tik_instance


# 'pylint: disable=invalid-name
@register_operator("rnn_gen_mask_v2")
def rnn_gen_mask_v2(seq_length, x, seq_mask, hidden_size, kernel_name="rnn_gen_mask_v2"):
    """
    Function: rnn_gen_mask_v2.
    Modify : 2021-04-22

    Init base parameters
    Parameters
    ----------
    input(seq_length): dict
        data of input
    input(x): dict
        data of input
    input(b): dict
        data of input
    output(seq_mask): dict
        data of output
    kernel_name: str
        the name of the operator
    ----------
    """
    op_obj = RnnGenMaskV2(seq_length, x, seq_mask, hidden_size, kernel_name)
    return op_obj.rnn_gen_mask_compute()
