#!/usr/bin/python
# -*- coding: utf-8 -*-
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
ctc_loss_v2
"""

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_platform as tbe_platform_adapter


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    Constant
    """
    BLOCK = 8
    BLOCK_B64 = 4
    MIN = -3.4e38
    MAX = 3.4e38
    BLANK_IDX = 5
    SHAPE_MODE_IDX = 6
    REPEAT_OFFSET = 255
    MAX_INT64 = 2 ** 63 - 1
    LABEL_MAX = 1000
    TIME_STEP_IDX = 0
    BATCH_SIZE_IDX = 1
    SYMBOL_SET_IDX = 2
    LABEL_ALIGN_IDX = 3
    CORE_NUM_VAR_IDX = 4
    DTYPE_PER_BLOCK_DICT = {"int32": 8, "int64": 4}
    TYPE_LEN_DICT = {"float32": 4, "int32": 4, "int64": 8, "float": 4}
    CONSTANT_COE = 2
    LAST_DIM = -1
    FIRST_DIM = 0
    SECOND_DIM = 1
    FP32_BYTES = 4
    INT64_BYTES = 8
    RESERVED_UB_SIZE = 2 * 1024
    C_LOOP_BLOCK = 64
    C_LOOP_SUM_BLOCK = 8
    C_UB_NUM = 5
    BATCH_UB_NUM = 2
    TARGETS_UB_NUM = 24
    ONE_FLOAT = 1.0


# 'pylint: disable=invalid-name,too-many-locals,too-many-arguments,unused-argument
def check_supported(log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank=0,
                    reduction="mean", zero_infinity=False, kernel_name="ctc_loss_v2"):
    """
    check the op support situation.
    """
    shape_log_alpha = log_alpha.get("shape")
    if int(-1) in shape_log_alpha or int(-2) in shape_log_alpha:
        return "Unknown"
    label_len = (shape_log_alpha[Constant.LAST_DIM] - 1) // Constant.CONSTANT_COE
    if label_len > Constant.LABEL_MAX:
        return False, "The label's length is over 1K."

    shape_log_probs = log_probs.get("shape")
    if int(-1) in shape_log_probs or int(-2) in shape_log_probs:
        return "Unknown"
    targets_dtype = targets.get("dtype").lower()
    input_lengths_dtype = input_lengths.get("dtype").lower()
    targets_dsize = Constant.TYPE_LEN_DICT.get(targets_dtype)
    input_lengths_dsize = Constant.TYPE_LEN_DICT.get(input_lengths_dtype)
    symbol_set_len = shape_log_probs[Constant.LAST_DIM]
    time_len = shape_log_probs[Constant.FIRST_DIM]
    batch_size = shape_log_probs[Constant.SECOND_DIM]

    max_all_data_size = label_len * (Constant.TARGETS_UB_NUM * Constant.FP32_BYTES + targets_dsize) +\
        symbol_set_len * Constant.FP32_BYTES * (Constant.C_UB_NUM + Constant.ONE_FLOAT / Constant.C_LOOP_BLOCK +\
        Constant.ONE_FLOAT / Constant.C_LOOP_SUM_BLOCK) + time_len * input_lengths_dsize                 
    ub_size = tbe_platform_adapter.get_soc_spec(tbe_platform_adapter.UB_SIZE)
    available_ub_size = ub_size - Constant.RESERVED_UB_SIZE
    if (max_all_data_size >= available_ub_size):
        reason = "the datasize of inputs is too large, aicpu recommended."
        return False, reason
    if (batch_size * (targets_dsize + Constant.FP32_BYTES) >= available_ub_size):
        reason = "the batchsize of inputs is too large, aicpu recommended."
        return False, reason
    return True, ""


@register_operator("ctc_loss_v2")
class CTCLossV2():
    """CTCLossV2"""

    def __init__(self, targets, input_lengths, target_lengths, blank, zero_infinity, kernel_name):
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.kernel_name = kernel_name
        self.zero_infinity = zero_infinity
        self.targets_dtype = targets.get("dtype").lower()
        self.targets_per_block = Constant.DTYPE_PER_BLOCK_DICT.get(self.targets_dtype, 8)
        self.input_lengths_dtype = input_lengths.get("dtype").lower()
        self.input_lengths_per_block = Constant.DTYPE_PER_BLOCK_DICT.get(self.input_lengths_dtype, 8)
        self.ub_size = tbe_platform_adapter.get_soc_spec(tbe_platform_adapter.UB_SIZE)
        self.target_lengths_dtype = target_lengths.get("dtype").lower()
        self.target_lengths_per_block = Constant.DTYPE_PER_BLOCK_DICT.get(self.target_lengths_dtype, 8)
        self.targets_dsize = Constant.TYPE_LEN_DICT.get(self.targets_dtype)
        self.input_lengths_dsize = Constant.TYPE_LEN_DICT.get(self.input_lengths_dtype)
        self.target_lengths_dsize = Constant.TYPE_LEN_DICT.get(self.target_lengths_dtype)
        self.int64 = "int64"
        self.is_pad_valid = tbe_platform.api_check_support("tik.data_move_pad")

        self._t = self.tik_instance.Scalar(self.int64)  # time step
        self._n = self.tik_instance.Scalar(self.int64)  # batch size
        self._c = self.tik_instance.Scalar(self.int64)  # size of symbol set
        self.label_align = self.tik_instance.Scalar(self.int64)
        self._s = self.tik_instance.Scalar(self.target_lengths_dtype)  # length of label
        self.dim_num = self.tik_instance.Scalar(self.int64)  # targets.dim_num
        self.blank = self.tik_instance.Scalar(self.int64, init_value=0)
        self.used_aicore_num = tik.Dprofile().get_aicore_num()
        self.core_num_var = self.tik_instance.Scalar(self.int64, init_value=self.used_aicore_num)

        self.c_block = self.tik_instance.Scalar(self.int64)
        self.n_block = self.tik_instance.Scalar(self.int64)
        self.offset_block = self.tik_instance.Scalar(self.int64)

        self.output_size = self.tik_instance.Scalar(self.int64)
        self.output_size_up = self.tik_instance.Scalar(self.int64)
        self.alpha_size = self.tik_instance.Scalar(self.int64)
        self.alpha_size_up = self.tik_instance.Scalar(self.int64)

        self.targets_block = self.tik_instance.Scalar(self.int64)
        self.neg_inf = Constant.MIN
        self.inf = Constant.MAX

        self.tiling_gm = self.tik_instance.Tensor(self.int64, [Constant.BLOCK], name="tiling_gm",
                                                  scope=tik.scope_gm)

        self.log_probs = self.tik_instance.Tensor("float32", [Constant.MAX_INT64], name="log_probs", scope=tik.scope_gm)
        self.targets = self.tik_instance.Tensor(self.targets_dtype, [Constant.MAX_INT64], name="targets",
                                                scope=tik.scope_gm)

        self.offset_gm = self.tik_instance.Tensor(self.int64, [Constant.MAX_INT64], name="offset_gm",
                                                  scope=tik.scope_gm, is_workspace=True)

        self.input_lengths = self.tik_instance.Tensor(self.input_lengths_dtype, [Constant.MAX_INT64],
                                                      name="input_lengths", scope=tik.scope_gm)
        self.target_lengths = self.tik_instance.Tensor(self.target_lengths_dtype, [Constant.MAX_INT64],
                                                       name="target_lengths", scope=tik.scope_gm)

        self.log_alpha = self.tik_instance.Tensor("float32", [Constant.MAX_INT64], name="log_alpha",
                                                  scope=tik.scope_gm)

        self.neg_log_likelihood = self.tik_instance.Tensor("float32", [Constant.MAX_INT64], name="neg_log_likelihood",
                                                           scope=tik.scope_gm, is_atomic_add=True)

    def set_running_core_num(self, tiling_core_num):
        self.core_num_var.set_as(tiling_core_num)

    def get_tiling_args(self):
        """get_tiling_args"""
        tiling_ub = self.tik_instance.Tensor(self.int64, [Constant.BLOCK], name='tiling_ub', scope=tik.scope_ubuf)
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, Constant.BLOCK // Constant.BLOCK_B64, 0, 0)
        self._t.set_as(tiling_ub[Constant.TIME_STEP_IDX])
        self._n.set_as(tiling_ub[Constant.BATCH_SIZE_IDX])
        self._c.set_as(tiling_ub[Constant.SYMBOL_SET_IDX])
        self.label_align.set_as(tiling_ub[Constant.LABEL_ALIGN_IDX])
        self.set_running_core_num(tiling_ub[Constant.CORE_NUM_VAR_IDX])
        self.blank.set_as(tiling_ub[Constant.BLANK_IDX])
        self.dim_num.set_as(tiling_ub[Constant.SHAPE_MODE_IDX])
        self.c_block = (self._c + Constant.BLOCK - 1) // Constant.BLOCK * Constant.BLOCK
        self.n_block = (self._n + Constant.BLOCK - 1) // Constant.BLOCK * Constant.BLOCK
        self.offset_block = (self._n + 1 + Constant.BLOCK_B64 - 1) // Constant.BLOCK_B64 * Constant.BLOCK_B64

    def get_output_size_and_inf(self):
        self.output_size = Constant.CONSTANT_COE * self._s + 1
        self.output_size_up = (self.output_size + Constant.BLOCK - 1) // Constant.BLOCK * Constant.BLOCK
        self.alpha_size = self._t * self.output_size
        self.alpha_size_up = self._t * self.output_size_up
        if self.is_pad_valid:
            self.neg_inf = float("-inf")
            self.inf = float("inf")

    def ctc_loss_compute(self):
        """ctc_loss_compute"""
        self.get_tiling_args()

        batch_num_per_aicore = self.tik_instance.Scalar(self.int64, init_value=self._n // self.core_num_var)
        batch_tail = self.tik_instance.Scalar(self.int64, init_value=self._n % self.core_num_var)
        self._s.set_as(0)
        with self.tik_instance.new_scope():
            target_lengths_block = (self._n + self.target_lengths_per_block - 1) // self.target_lengths_per_block * \
                                   self.target_lengths_per_block
            target_lengths_ub = self.tik_instance.Tensor(self.target_lengths_dtype, [target_lengths_block],
                                                         name="target_lengths_ub", scope=tik.scope_ubuf)
            offset_ub = self.tik_instance.Tensor(self.int64, [self.offset_block], name="offset_ub",
                                                 scope=tik.scope_ubuf)
            offset = self.tik_instance.Scalar(self.int64, init_value=0)
            offset_tmp = self.tik_instance.Scalar(self.int64)
            offset_ub[0].set_as(offset)

            self.int_move_to_ub(target_lengths_ub, self.target_lengths, self._n * self.target_lengths_dsize,
                                target_lengths_block // self.target_lengths_per_block)

            with self.tik_instance.for_range(0, self._n) as task_idx:
                offset_tmp.set_as(target_lengths_ub[task_idx])
                with self.tik_instance.if_scope(offset_tmp > self._s):
                    self._s.set_as(offset_tmp)
                with self.tik_instance.if_scope(self.dim_num == 1):
                    offset.set_as(offset + offset_tmp)
                    offset_ub[task_idx + 1].set_as(offset)
            with self.tik_instance.if_scope(self.dim_num == 1):
                self.tik_instance.data_move(self.offset_gm, offset_ub, 0, 1, self.offset_block // Constant.BLOCK_B64,
                                            0, 0)

        self.targets_block = (self._s + self.targets_per_block - 1) // self.targets_per_block * self.targets_per_block

        self.get_output_size_and_inf()

        with self.tik_instance.for_range(0, self.core_num_var, block_num=self.core_num_var) as i:
            with self.tik_instance.for_range(0, batch_num_per_aicore) as j:
                self.ctc_loss_compute_core(i + j * self.core_num_var)
            with self.tik_instance.if_scope(i < batch_tail):
                self.ctc_loss_compute_core(batch_num_per_aicore * self.core_num_var + i)

        tbe_context.get_context().add_compile_info(
            "vars", {
                "core_num": self.used_aicore_num,
                "ub_size": self.ub_size,
                "targets_dsize": self.targets_dsize,
                "input_lengths_dsize": self.input_lengths_dsize
            })

        opt_config = {
            "enable_const_fold": True
        }

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.log_probs, self.targets, self.input_lengths, self.target_lengths],
                                   outputs=[self.neg_log_likelihood, self.log_alpha],
                                   flowtable=[self.tiling_gm],
                                   config=opt_config)

        return self.tik_instance

    # 'pylint: disable=too-many-statements
    def ctc_loss_compute_core(self, task_idx):
        """ctc_loss_compute_core"""
        targets_ub = self.tik_instance.Tensor(self.targets_dtype, [self.targets_block], name="targets_ub",
                                              scope=tik.scope_ubuf)
        input_lengths_ub = self.tik_instance.Tensor(self.input_lengths_dtype, [self.input_lengths_per_block],
                                                    name="input_lengths_ub", scope=tik.scope_ubuf)
        target_lengths_ub = self.tik_instance.Tensor(self.target_lengths_dtype, [self.target_lengths_per_block],
                                                     name="target_lengths_ub", scope=tik.scope_ubuf)
        neg_inf_ub = self.tik_instance.Tensor("float32", [Constant.BLOCK], name="neg_inf_ub", scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(Constant.BLOCK, neg_inf_ub, self.neg_inf, 1, 1, 1)
        self.int_move_to_ub(target_lengths_ub, self.target_lengths, self.target_lengths_dsize, 1, task_idx)
        self.int_move_to_ub(input_lengths_ub, self.input_lengths, self.input_lengths_dsize, 1, task_idx)

        t_i = self.tik_instance.Scalar(self.input_lengths_dtype, init_value=input_lengths_ub[0])
        s_i = self.tik_instance.Scalar(self.target_lengths_dtype, init_value=target_lengths_ub[0])

        offset_ub = self.tik_instance.Tensor(self.int64, [Constant.BLOCK_B64], name="offset_ub", scope=tik.scope_ubuf)
        offset = self.tik_instance.Scalar(self.int64)
        with self.tik_instance.if_scope(self.dim_num == 1):
            self.offset_move_to_ub_one_elem(offset_ub, task_idx)
            offset.set_as(offset_ub[0])
            with self.tik_instance.if_scope(tik.all(s_i > 0, offset >= 0)):
                self.int_move_to_ub(targets_ub, self.targets, s_i * self.targets_dsize,
                                    (s_i + self.targets_per_block - 1) // self.targets_per_block, offset)
        with self.tik_instance.else_scope():
            self.int_move_to_ub(targets_ub, self.targets, self._s * self.targets_dsize,
                                self.targets_block // self.targets_per_block, task_idx * self.label_align)
        # func: block invalid tasks
        with self.tik_instance.if_scope(t_i >= s_i):
            repeats, s_inc, e_inc = self.count_trace(s_i, targets_ub)

            start = self.tik_instance.Scalar(self.int64)
            start_loop = self.tik_instance.Scalar(self.int64)
            end = self.tik_instance.Scalar(self.int64)
            remain = self.tik_instance.Scalar(self.int64)
            current_target = self.tik_instance.Scalar(self.targets_dtype)
            next_target = self.tik_instance.Scalar(self.targets_dtype)
            tmp = self.tik_instance.Scalar(self.int64)

            # func: a_ub/b_ub/tmp_ub: used in exp/log/add/sub api
            compute_tmp_ub = self.tik_instance.Tensor("float32", [self.output_size_up], name="compute_tmp_ub",
                                                      scope=tik.scope_ubuf)
            lamda1_ub = self.tik_instance.Tensor("float32", [self.output_size_up], name="lamda1_ub",
                                                 scope=tik.scope_ubuf)
            lamda2_ub = self.tik_instance.Tensor("float32", [self.output_size_up], name="lamda2_ub",
                                                 scope=tik.scope_ubuf)
            lamda3_ub = self.tik_instance.Tensor("float32", [self.output_size_up], name="lamda3_ub",
                                                 scope=tik.scope_ubuf)
            current_log_probs_ub = self.tik_instance.Tensor("float32", [self.output_size_up],
                                                            name="current_log_probs_ub",
                                                            scope=tik.scope_ubuf)
            exp_add_ub = self.tik_instance.Tensor("float32", [self.output_size_up], name="exp_add_ub",
                                                  scope=tik.scope_ubuf)
            ln_add_ub = self.tik_instance.Tensor("float32", [self.output_size_up], name="ln_add_ub",
                                                 scope=tik.scope_ubuf)
            a_ub = self.tik_instance.Tensor("float32", [Constant.BLOCK], name="a_ub",
                                            scope=tik.scope_ubuf)
            b_ub = self.tik_instance.Tensor("float32", [Constant.BLOCK], name="b_ub",
                                            scope=tik.scope_ubuf)
            res_ub = self.tik_instance.Tensor("float32", [Constant.BLOCK], name="res_ub",
                                              scope=tik.scope_ubuf)
            self.tik_instance.vector_dup(Constant.BLOCK, res_ub, 0, 1, 1, 1)
            work_tensor_ub = self.tik_instance.Tensor("float32", [Constant.BLOCK], name="work_tensor_ub",
                                                      scope=tik.scope_ubuf)

            a_tmp = self.tik_instance.Scalar("float32")
            b_tmp = self.tik_instance.Scalar("float32")
            c_tmp = self.tik_instance.Scalar("float32")
            max_tmp = self.tik_instance.Scalar("float32")
            log_probs_ub = self.tik_instance.Tensor("float32", [self.c_block], name="log_probs_ub",
                                                    scope=tik.scope_ubuf)
            self.select_data_move(log_probs_ub[0], self.log_probs[self._c * task_idx], self._c * Constant.FP32_BYTES,
                                  self.c_block // Constant.BLOCK)

            output_dst = self.tik_instance.Scalar(self.int64, init_value=0)
            output_src = self.tik_instance.Scalar(self.int64, init_value=self.output_size_up)

            log_alpha_ub = self.tik_instance.Tensor("float32", [Constant.CONSTANT_COE, self.output_size_up],
                                                    name="log_alpha_ub", scope=tik.scope_ubuf)
            self.tik_instance.vector_dup(Constant.BLOCK, log_alpha_ub[output_dst], self.neg_inf,
                                         self.output_size_up // Constant.BLOCK, 1, 1)

            lamax_ub = self.tik_instance.Tensor("float32", [self.output_size], name="lamax_ub", scope=tik.scope_ubuf)

            log_alpha_ub[output_dst].set_as(log_probs_ub[self.blank])
            current_target.set_as(targets_ub[0])
            with self.tik_instance.if_scope(current_target >= 0):
                log_alpha_ub[output_dst + 1].set_as(log_probs_ub[current_target])
            with self.tik_instance.else_scope():
                log_alpha_ub[output_dst + 1].set_as(log_probs_ub[self.blank])

            with self.tik_instance.if_scope(repeats < t_i - s_i):
                start.set_as(0)
            with self.tik_instance.else_scope():
                start.set_as(1)
            end.set_as(2)

            with self.tik_instance.if_scope(t_i >= 1):
                with self.tik_instance.for_range(1, t_i) as t:
                    self.select_data_move(log_probs_ub[0], self.log_probs[self._c * task_idx + self._n * self._c * t],
                                        self._c * Constant.FP32_BYTES, self.c_block // Constant.BLOCK)
                    self.tik_instance.vector_dup(Constant.BLOCK, log_alpha_ub[output_src], self.neg_inf,
                                                self.output_size_up // Constant.BLOCK, 1, 1)

                    remain.set_as(s_i + repeats - t_i + t)
                    with self.tik_instance.if_scope(remain >= 0):
                        tmp.set_as(s_inc[remain])
                        start.set_as(start + tmp)
                    start_loop.set_as(start)

                    with self.tik_instance.if_scope(t <= s_i + repeats):
                        tmp.set_as(e_inc[t - 1])
                        end.set_as(end + tmp)

                    with self.tik_instance.if_scope(start_loop == 0):
                        a_tmp.set_as(log_alpha_ub[output_dst])
                        b_tmp.set_as(log_probs_ub[self.blank])

                        log_alpha_ub[output_src].set_as(a_tmp + b_tmp)
                        start_loop.set_as(1)

                    with self.tik_instance.for_range(start_loop, end) as s:
                        with self.tik_instance.if_scope(s % Constant.CONSTANT_COE == 0):
                            current_target.set_as(self.blank)
                        with self.tik_instance.else_scope():
                            current_target.set_as(targets_ub[s // Constant.CONSTANT_COE])

                        offset.set_as(s - start_loop)

                        with self.tik_instance.if_scope(current_target >= 0):
                            current_log_probs_ub[offset].set_as(log_probs_ub[current_target])
                        with self.tik_instance.else_scope():
                            current_log_probs_ub[offset].set_as(log_probs_ub[0])
                        lamda1_ub[offset].set_as(log_alpha_ub[output_dst + s])
                        lamda2_ub[offset].set_as(log_alpha_ub[output_dst + s - 1])

                        with self.tik_instance.if_scope(tik.all((s % Constant.CONSTANT_COE != 0), (s != 1))):
                            next_target.set_as(targets_ub[s // Constant.CONSTANT_COE - 1])
                            with self.tik_instance.if_scope(current_target != next_target):
                                lamda3_ub[offset].set_as(log_alpha_ub[output_dst + s - Constant.CONSTANT_COE])
                            with self.tik_instance.else_scope():
                                lamda3_ub[offset].set_as(self.neg_inf)
                        with self.tik_instance.else_scope():
                            lamda3_ub[offset].set_as(self.neg_inf)

                    # func: get `lamax`
                    self.tik_instance.h_max(lamax_ub, lamda1_ub, lamda2_ub)
                    self.tik_instance.h_max(lamax_ub, lamax_ub, lamda3_ub)

                    # func: get `la-lamax`
                    self.tik_instance.h_sub(lamda1_ub, lamda1_ub, lamax_ub)
                    self.tik_instance.h_sub(lamda2_ub, lamda2_ub, lamax_ub)
                    self.tik_instance.h_sub(lamda3_ub, lamda3_ub, lamax_ub)

                    # func: `exp(a_tmp- max_tmp)  exp(b_tmp- max_tmp)  exp(b_tmp- max_tmp)`
                    self.tik_instance.h_exp(lamda1_ub, lamda1_ub)
                    self.tik_instance.h_exp(lamda2_ub, lamda2_ub)
                    self.tik_instance.h_exp(lamda3_ub, lamda3_ub)

                    # func: `exp(a_tmp- max_tmp) + exp(b_tmp- max_tmp) + exp(b_tmp- max_tmp)`
                    self.tik_instance.h_add(exp_add_ub, lamda1_ub, lamda2_ub)
                    self.tik_instance.h_add(exp_add_ub, exp_add_ub, lamda3_ub)

                    # func: `log(exp(a_tmp- max_tmp) + exp(b_tmp- max_tmp) + exp(b_tmp- max_tmp))`
                    self.tik_instance.h_ln(ln_add_ub, exp_add_ub)

                    # func: `log(exp(a_tmp- max_tmp) + exp(b_tmp- max_tmp) + exp(b_tmp- max_tmp)) + max_tmp`
                    self.tik_instance.h_add(compute_tmp_ub, ln_add_ub, lamax_ub)

                    # func: `log(exp(a_tmp- max_tmp) + exp(b_tmp- max_tmp) + exp(b_tmp- max_tmp)) + max_tmp + log_probs`
                    self.tik_instance.h_add(compute_tmp_ub, compute_tmp_ub, current_log_probs_ub)

                    # func: update log_beta in current T
                    with self.tik_instance.for_range(start_loop, end) as s:
                        log_alpha_ub[output_src + s].set_as(compute_tmp_ub[s - start_loop])

                    self.select_data_move(self.log_alpha[task_idx * self.alpha_size_up + (t - 1) * self.output_size],
                                        log_alpha_ub[output_dst], self.output_size * Constant.FP32_BYTES,
                                        self.output_size_up // Constant.BLOCK)

                    output_src.set_as(output_dst)
                    output_dst.set_as(self.output_size_up - output_src)

            with self.tik_instance.if_scope(task_idx * self.alpha_size_up + (t_i - 1) * self.output_size >= 0):
                self.select_data_move(self.log_alpha[task_idx * self.alpha_size_up + (t_i - 1) * self.output_size],
                                      log_alpha_ub[output_dst], self.output_size * Constant.FP32_BYTES,
                                      self.output_size_up // Constant.BLOCK)

            with self.tik_instance.if_scope(output_dst + Constant.CONSTANT_COE * s_i >= 0):
                a_tmp.set_as(log_alpha_ub[output_dst + Constant.CONSTANT_COE * s_i])
            with self.tik_instance.else_scope():
                a_tmp.set_as(log_alpha_ub[0])
            
            with self.tik_instance.if_scope(s_i <= 0):
                b_tmp.set_as(neg_inf_ub[0])
            with self.tik_instance.else_scope():
                b_tmp.set_as(log_alpha_ub[output_dst + Constant.CONSTANT_COE * s_i - 1])
            c_tmp.set_as(0)

            with self.tik_instance.if_scope(a_tmp > b_tmp):
                max_tmp.set_as(a_tmp)
                a_ub[0].set_as(c_tmp)
                a_ub[1].set_as(b_tmp - a_tmp)
            with self.tik_instance.elif_scope(tik.all(self.is_pad_valid, a_tmp == float("-inf"),
                                                      b_tmp == float("-inf"))):
                max_tmp.set_as(0)
                a_ub[0].set_as(a_tmp)
                a_ub[1].set_as(b_tmp)
            with self.tik_instance.else_scope():
                max_tmp.set_as(b_tmp)
                a_ub[0].set_as(a_tmp - b_tmp)
                a_ub[1].set_as(c_tmp)

            self.tik_instance.vec_exp(Constant.CONSTANT_COE, b_ub, a_ub, 1, 1, 1)
            self.tik_instance.vec_reduce_add(Constant.CONSTANT_COE, a_ub, b_ub, work_tensor_ub, 1, 1)

            self.tik_instance.vln(1, b_ub, a_ub, 1, 1, 1, 1, 1)
            a_tmp.set_as(b_ub[0])
            b_tmp.set_as(-a_tmp - max_tmp)
            if_nan = tik.all(tik.negate(b_tmp <= Constant.MAX), tik.negate(b_tmp > Constant.MAX))
            with self.tik_instance.if_scope(tik.any(b_tmp < self.inf, tik.negate(self.zero_infinity), if_nan)):
                res_ub[0].set_as(b_tmp)
            with self.tik_instance.else_scope():
                res_ub[0].set_as(0)
            self.tik_instance.set_atomic_add(1)
            self.select_data_move(self.neg_log_likelihood[task_idx], res_ub, Constant.FP32_BYTES, 1)
            self.tik_instance.set_atomic_add(0)
        with self.tik_instance.elif_scope(not self.zero_infinity):
            # fill 3.4e+38 to represent inf
            res_ub = self.tik_instance.Tensor("float32", [Constant.BLOCK], name="res_ub",
                                              scope=tik.scope_ubuf)
            self.tik_instance.vector_dup(Constant.BLOCK, res_ub, 0, 1, 1, 1)
            res_ub[0].set_as(self.inf)
            self.tik_instance.set_atomic_add(1)
            self.select_data_move(self.neg_log_likelihood[task_idx], res_ub, Constant.FP32_BYTES, 1)
            self.tik_instance.set_atomic_add(0)

    def count_trace(self, s_i, targets_ub):
        """count_trace"""
        s_inc = self.tik_instance.Tensor(self.int64, [self.output_size], name="s_inc", scope=tik.scope_ubuf)
        e_inc = self.tik_instance.Tensor(self.int64, [self.output_size], name="e_inc", scope=tik.scope_ubuf)

        one_step = self.tik_instance.Scalar(self.int64, init_value=1)
        two_step = self.tik_instance.Scalar(self.int64, init_value=2)

        left = self.tik_instance.Scalar(self.targets_dtype)
        right = self.tik_instance.Scalar(self.targets_dtype)

        repeats = self.tik_instance.Scalar(self.int64, init_value=0)
        idx_counter = self.tik_instance.Scalar(self.int64, init_value=1)

        s_inc[0].set_as(one_step)
        with self.tik_instance.if_scope(s_i >= 1):
            with self.tik_instance.for_range(1, s_i) as idx:
                left.set_as(targets_ub[idx - 1])
                right.set_as(targets_ub[idx])
                with self.tik_instance.if_scope(left == right):
                    s_inc[idx_counter].set_as(one_step)
                    e_inc[idx_counter - 1].set_as(one_step)

                    s_inc[idx_counter + 1].set_as(one_step)
                    e_inc[idx_counter].set_as(one_step)

                    idx_counter.set_as(idx_counter + two_step)
                    repeats.set_as(repeats + 1)
                with self.tik_instance.else_scope():
                    s_inc[idx_counter].set_as(two_step)
                    e_inc[idx_counter - 1].set_as(two_step)

                    idx_counter.set_as(idx_counter + 1)

        e_inc[idx_counter - 1].set_as(one_step)

        return repeats, s_inc, e_inc

    def int_move_to_ub(self, dst, src, data_move_pad_burst, data_move_burst=1, offset=0):
        if self.is_pad_valid and (dst.dtype == self.int64):
            dst = dst.reinterpret_cast_to("int32")
            src = src.reinterpret_cast_to("int32")
            self.tik_instance.data_move_pad(dst, src[offset * 2], nburst=1,
                                            burst=data_move_pad_burst, dst_gap=0,
                                            src_gap=0, left_padding=0, right_padding=0, padding_value=None)
            dst = dst.reinterpret_cast_to(self.int64)
            src = src.reinterpret_cast_to(self.int64)
        elif self.is_pad_valid:
            self.tik_instance.data_move_pad(dst, src[offset], nburst=1,
                                            burst=data_move_pad_burst, dst_gap=0,
                                            src_gap=0, left_padding=0, right_padding=0, padding_value=None)
        else:
            self.tik_instance.data_move(dst, src[offset], 0, 1,
                                        data_move_burst, 0, 0)
    
    def offset_move_to_ub_one_elem(self, offset_ub, task_idx):
        if self.is_pad_valid:
            offset_ub = offset_ub.reinterpret_cast_to("int32")
            self.offset_gm = self.offset_gm.reinterpret_cast_to("int32")
            self.tik_instance.data_move_pad(offset_ub, self.offset_gm[task_idx * 2], nburst=1,
                                            burst=Constant.INT64_BYTES, dst_gap=0,
                                            src_gap=0, left_padding=0, right_padding=0, padding_value=None)
            offset_ub = offset_ub.reinterpret_cast_to(self.int64)
            self.offset_gm = self.offset_gm.reinterpret_cast_to(self.int64)
        else:
            self.tik_instance.data_move(offset_ub, self.offset_gm[task_idx], 0, 1, 1, 0, 0)


    def select_data_move(self, dst, src, data_move_pad_burst, data_move_burst):
        if self.is_pad_valid:
            self.tik_instance.data_move_pad(dst, src, nburst=1, burst=data_move_pad_burst, dst_gap=0,
                                            src_gap=0, left_padding=0, right_padding=0, padding_value=None)
        else:
            self.tik_instance.data_move(dst, src, 0, 1, data_move_burst, 0, 0)


# 'pylint: disable=invalid-name,too-many-locals,too-many-arguments,unused-argument
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def ctc_loss_v2(log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank=0,
                reduction="mean", zero_infinity=False, kernel_name="ctc_loss_v2"):
    """
    Function: The Connectionist Temporal Classification loss.

    Init base parameters
    Parameters
    ----------
    Inputs:
    Log_probs: Tensor of size (T,N,C), where T =input length, N =batch size,
               and C = number of classes (including blank).
    Targets: Tensor of size (N, S), where S= max target length.
    It represent the target sequences.
    Input_lengths: Tuple or tensor of size (N).
    It represent the lengths of the inputs.
    Target_lengths: Tuple or tensor of size (N). It represent lengths of the targets.

    Attributes:
    blank: Blank label. Default 0.
    reduction: Specifies the reduction to apply to the output. Default: 'mean'.
    zero_infinity: Whether to zero infinite losses and the associated gradients.

    Outputs:
    neg_log_likelihood: A loss value which is differentiable with respect to each input node.
    log_alpha: The probability of possible trace of input to target.
    ----------
    """
    op_obj = CTCLossV2(targets, input_lengths, target_lengths, blank, zero_infinity, kernel_name)

    return op_obj.ctc_loss_compute()
