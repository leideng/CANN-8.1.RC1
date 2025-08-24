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
ctc_loss_v2_grad
"""

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_platform as tbe_platform_adapter


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    Constant
    """
    BLOCK = 8
    MIN = -3.4e38
    MAX = 3.4e38
    BLANK_IDX = 5
    SHAPE_MODE_IDX = 6
    REPEAT_OFFSET = 255
    MAX_INT32 = 2 ** 31 - 1
    LABEL_MAX = 1000
    TIME_STEP_IDX = 0
    BATCH_SIZE_IDX = 1
    SYMBOL_SET_IDX = 2
    LABEL_ALIGN_IDX = 3
    CORE_NUM_VAR_IDX = 4
    C_ALIGN = 64
    DTYPE_BYTES_DICT = {"int32": 8, "int64": 4}
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
    INT64_TO_INT32_NUM = 2
    INT64 = "int64"
    INT32 = "int32"


# 'pylint: disable=invalid-name,too-many-locals,too-many-arguments,unused-argument
def check_supported(grad_out, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, grad,
                    blank=0, reduction="mean", zero_infinity=False, kernel_name="ctc_loss_v2_grad"):
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


@register_operator("ctc_loss_v2_grad")
class CTCLossV2Grad():
    """Function: Class CTCLossV2Grad."""

    def __init__(self, targets, input_lengths, target_lengths, blank, zero_infinity, kernel_name):
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.kernel_name = kernel_name
        self.zero_infinity = zero_infinity
        self.targets_dtype = targets.get("dtype").lower()
        self.targets_block = Constant.DTYPE_BYTES_DICT.get(self.targets_dtype, 8)
        self.input_lengths_dtype = input_lengths.get("dtype").lower()
        self.input_lengths_block = Constant.DTYPE_BYTES_DICT.get(self.targets_dtype, 8)
        self.target_lengths_dtype = target_lengths.get("dtype").lower()
        self.target_lengths_block = Constant.DTYPE_BYTES_DICT.get(self.targets_dtype, 8)
        self.ub_size = tbe_platform_adapter.get_soc_spec(tbe_platform_adapter.UB_SIZE)
        self.is_pad_valid = tbe_platform_adapter.api_check_support("tik.data_move_pad")
        self.targets_dsize = Constant.TYPE_LEN_DICT.get(self.targets_dtype)
        self.input_lengths_dsize = Constant.TYPE_LEN_DICT.get(self.input_lengths_dtype)

        self._t = self.tik_instance.Scalar("int32")  # time step
        self._n = self.tik_instance.Scalar("int32")  # batch size
        self._c = self.tik_instance.Scalar("int32")  # size of symbol set
        self.label_align = self.tik_instance.Scalar("int32")
        self._s = self.tik_instance.Scalar(self.target_lengths_dtype)  # length of label
        self.dim_num = self.tik_instance.Scalar("int32")  # targets.dim_num
        self.blank = self.tik_instance.Scalar("int32", init_value=0)
        self.used_aicore_num = tik.Dprofile().get_aicore_num()
        self.core_num_var = self.tik_instance.Scalar("int32", init_value=self.used_aicore_num)

        self.c_block = self.tik_instance.Scalar("int32")
        self.n_block = self.tik_instance.Scalar("int32")
        self.s_block = self.tik_instance.Scalar("int32")

        self.output_size = self.tik_instance.Scalar("int32")
        self.output_size_up = self.tik_instance.Scalar("int32")
        self.alpha_size = self.tik_instance.Scalar("int32")
        self.alpha_size_up = self.tik_instance.Scalar("int32")
        self.rounds = self.tik_instance.Scalar("int32")

        self.tiling_gm = self.tik_instance.Tensor("int32", [Constant.BLOCK], name="tiling_gm",
                                                  scope=tik.scope_gm)
        self.grad_out = self.tik_instance.Tensor("float32", [Constant.MAX_INT32], name="grad_out", scope=tik.scope_gm)
        self.log_probs = self.tik_instance.Tensor("float32", [Constant.MAX_INT32], name="log_probs",
                                                  scope=tik.scope_gm)
        self.targets = self.tik_instance.Tensor(self.targets_dtype, [Constant.MAX_INT32], name="targets",
                                                scope=tik.scope_gm)

        self.offset_gm = self.tik_instance.Tensor("int32", [Constant.MAX_INT32], name="offset_gm", scope=tik.scope_gm,
                                                  is_workspace=True)

        self.input_lengths = self.tik_instance.Tensor(self.input_lengths_dtype, [Constant.MAX_INT32],
                                                      name="input_lengths", scope=tik.scope_gm)
        self.target_lengths = self.tik_instance.Tensor(self.target_lengths_dtype, [Constant.MAX_INT32],
                                                       name="target_lengths", scope=tik.scope_gm)

        self.neg_log_likelihood = self.tik_instance.Tensor("float32", [Constant.MAX_INT32], name="neg_log_likelihood",
                                                           scope=tik.scope_gm)
        self.log_alpha = self.tik_instance.Tensor("float32", [Constant.MAX_INT32], name="log_alpha",
                                                  scope=tik.scope_gm)
        self.grad = self.tik_instance.Tensor("float32", [Constant.MAX_INT32], name="grad",
                                             scope=tik.scope_gm, is_atomic_add=True)

    def get_tiling_args(self):
        """get_tiling_args"""
        tiling_ub = self.tik_instance.Tensor("int32", [Constant.BLOCK],
                                             name='tiling_ub', scope=tik.scope_ubuf)
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, 1, 0, 0)
        self._t.set_as(tiling_ub[Constant.TIME_STEP_IDX])
        self._n.set_as(tiling_ub[Constant.BATCH_SIZE_IDX])
        self._c.set_as(tiling_ub[Constant.SYMBOL_SET_IDX])
        self.label_align.set_as(tiling_ub[Constant.LABEL_ALIGN_IDX])
        self.core_num_var.set_as(tiling_ub[Constant.CORE_NUM_VAR_IDX])
        self.blank.set_as(tiling_ub[Constant.BLANK_IDX])
        self.dim_num.set_as(tiling_ub[Constant.SHAPE_MODE_IDX])

    def int_move_to_ub(self, dst, src, data_move_pad_burst, data_move_burst=1, offset=0):
        """choosing the method to move data from gm to ub, including int64"""
        if self.is_pad_valid and (dst.dtype == Constant.INT64):
            dst = dst.reinterpret_cast_to(Constant.INT32)
            src = src.reinterpret_cast_to(Constant.INT32)
            self.tik_instance.data_move_pad(dst, src[offset * Constant.INT64_TO_INT32_NUM], nburst=1,
                                            burst=data_move_pad_burst, dst_gap=0,
                                            src_gap=0, left_padding=0, right_padding=0, padding_value=None)
            dst = dst.reinterpret_cast_to(Constant.INT64)
            src = src.reinterpret_cast_to(Constant.INT64)
        elif self.is_pad_valid:
            self.tik_instance.data_move_pad(dst, src[offset], nburst=1,
                                            burst=data_move_pad_burst, dst_gap=0,
                                            src_gap=0, left_padding=0, right_padding=0, padding_value=None)
        else:
            self.tik_instance.data_move(dst, src[offset], 0, 1,
                                        data_move_burst, 0, 0)

    def select_data_move(self, dst, src, data_move_pad_burst, data_move_burst):
        """choosing the method to move data, except int64"""
        if self.is_pad_valid:
            self.tik_instance.data_move_pad(dst, src, nburst=1, burst=data_move_pad_burst, dst_gap=0,
                                            src_gap=0, left_padding=0, right_padding=0, padding_value=None)
        else:
            self.tik_instance.data_move(dst, src, 0, 1, data_move_burst, 0, 0)

    def ctc_loss_grad_compute(self):
        """
        Function: ctc_loss_grad_compute.
        """
        self.get_tiling_args()

        self.c_block = (self._c + Constant.BLOCK - 1) // Constant.BLOCK * Constant.BLOCK
        self.n_block = (self._n + Constant.BLOCK - 1) // Constant.BLOCK * Constant.BLOCK

        batch_num_per_aicore = self.tik_instance.Scalar("int32", init_value=self._n // self.core_num_var)
        batch_tail = self.tik_instance.Scalar("int32", init_value=self._n % self.core_num_var)
        self._s.set_as(0)
        with self.tik_instance.new_scope():
            target_lengths_ub = self.tik_instance.Tensor(self.targets_dtype, [self.n_block], name="target_lengths_ub",
                                                         scope=tik.scope_ubuf)
            offset_ub = self.tik_instance.Tensor("int32", [self.n_block], name="offset_ub",
                                                 scope=tik.scope_ubuf)
            offset = self.tik_instance.Scalar(self.targets_dtype, init_value=0)
            offset_tmp = self.tik_instance.Scalar(self.targets_dtype)
            offset_ub[0].set_as(offset)

            self.int_move_to_ub(target_lengths_ub, self.target_lengths,
                                data_move_pad_burst=self._n * self.targets_dsize,
                                data_move_burst=self.n_block // self.targets_block,
                                offset=0)

            with self.tik_instance.for_range(0, self._n) as task_idx:
                offset_tmp.set_as(target_lengths_ub[task_idx])
                with self.tik_instance.if_scope(offset_tmp > self._s):
                    self._s.set_as(offset_tmp)
                with self.tik_instance.if_scope(self.dim_num == 1):
                    offset.set_as(offset + offset_tmp)
                    offset_ub[task_idx + 1].set_as(offset)
            with self.tik_instance.if_scope(self.dim_num == 1):
                self.select_data_move(self.offset_gm,
                                      offset_ub,
                                      data_move_pad_burst=self._n * Constant.FP32_BYTES,
                                      data_move_burst=self.n_block // Constant.BLOCK)

        self.s_block = (self._s + Constant.BLOCK - 1) // Constant.BLOCK * Constant.BLOCK

        self.output_size = Constant.CONSTANT_COE * self._s + 1
        self.output_size_up = (self.output_size + Constant.BLOCK - 1) // Constant.BLOCK * Constant.BLOCK
        self.alpha_size = self._t * self.output_size
        self.alpha_size_up = self._t * self.output_size_up
        self.rounds = self.c_block // (Constant.BLOCK * Constant.REPEAT_OFFSET)

        with self.tik_instance.for_range(0, self.core_num_var, block_num=self.core_num_var) as i:
            self.tik_instance.set_atomic_add(1)
            with self.tik_instance.for_range(0, batch_num_per_aicore) as j:
                self.ctc_loss_grad_compute_core(i + j * self.core_num_var)
            with self.tik_instance.if_scope(i < batch_tail):
                self.ctc_loss_grad_compute_core(batch_num_per_aicore * self.core_num_var + i)
            self.tik_instance.set_atomic_add(0)

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
                                   inputs=[self.grad_out, self.log_probs, self.targets, self.input_lengths,
                                           self.target_lengths, self.neg_log_likelihood, self.log_alpha],
                                   outputs=[self.grad], flowtable=[self.tiling_gm],
                                   config=opt_config)

        return self.tik_instance

    # 'pylint: disable=too-many-statements
    def ctc_loss_grad_compute_core(self, task_idx):
        """
        Function: ctc_loss_grad_compute_core.
        """
        grad_out_ub = self.tik_instance.Tensor("float32", [Constant.BLOCK], name="grad_out_ub", scope=tik.scope_ubuf)
        targets_ub = self.tik_instance.Tensor(self.targets_dtype, [self.s_block], name="targets_ub",
                                              scope=tik.scope_ubuf)
        input_lengths_ub = self.tik_instance.Tensor(self.input_lengths_dtype, [self.input_lengths_block],
                                                    name="input_lengths_ub", scope=tik.scope_ubuf)
        target_lengths_ub = self.tik_instance.Tensor(self.target_lengths_dtype, [self.target_lengths_block],
                                                     name="target_lengths_ub", scope=tik.scope_ubuf)
        grad_ub = self.tik_instance.Tensor("float32", [self.c_block], name="grad_ub", scope=tik.scope_ubuf)
        grad_abs_ub = self.tik_instance.Tensor("float32", [self.c_block], name="grad_abs_ub", scope=tik.scope_ubuf)
        neg_log_likelihood_ub = self.tik_instance.Tensor("float32", [Constant.BLOCK], name="neg_log_likelihood_ub",
                                                         scope=tik.scope_ubuf)
        self.int_move_to_ub(input_lengths_ub,
                            self.input_lengths,
                            data_move_pad_burst=self.input_lengths_dsize,
                            data_move_burst=1,
                            offset=task_idx)
        self.int_move_to_ub(target_lengths_ub,
                            self.target_lengths,
                            data_move_pad_burst=self.targets_dsize,
                            data_move_burst=1,
                            offset=task_idx)
        self.select_data_move(neg_log_likelihood_ub,
                              self.neg_log_likelihood[task_idx],
                              data_move_pad_burst=Constant.FP32_BYTES,
                              data_move_burst=1)

        # func: recored current T and S
        t_i = self.tik_instance.Scalar(self.input_lengths_dtype, init_value=input_lengths_ub[0])
        s_i = self.tik_instance.Scalar(self.target_lengths_dtype, init_value=target_lengths_ub[0])

        # func: initial grad_ub
        with self.tik_instance.for_range(0, self.rounds) as j:
            self.tik_instance.vector_dup(Constant.BLOCK, grad_ub[(Constant.BLOCK * Constant.REPEAT_OFFSET) * j],
                                         0, Constant.REPEAT_OFFSET, 1, 1)
        self.tik_instance.vector_dup(Constant.BLOCK, grad_ub[(Constant.BLOCK * Constant.REPEAT_OFFSET) * self.rounds],
                                     0, (self.c_block % (Constant.BLOCK * Constant.REPEAT_OFFSET)) // Constant.BLOCK,
                                     1, 1)
        # func: move in grad_out and targets_ub
        offset_ub = self.tik_instance.Tensor("int32", [Constant.BLOCK], name="offset_ub", scope=tik.scope_ubuf)
        offset = self.tik_instance.Scalar("int32")

        self.select_data_move(grad_out_ub,
                              self.grad_out[task_idx],
                              data_move_pad_burst=Constant.FP32_BYTES,
                              data_move_burst=1)
        with self.tik_instance.if_scope(self.dim_num == 1):
            self.int_move_to_ub(offset_ub,
                                self.offset_gm,
                                data_move_pad_burst=Constant.FP32_BYTES,
                                data_move_burst=1,
                                offset=task_idx)
            offset.set_as(offset_ub[0])
            with self.tik_instance.if_scope(tik.all(s_i > 0, offset > -1)):
                self.int_move_to_ub(targets_ub,
                                    self.targets,
                                    data_move_pad_burst=s_i * self.targets_dsize,
                                    data_move_burst=(s_i + self.targets_block - 1) // self.targets_block,
                                    offset=offset)
        with self.tik_instance.else_scope():
            self.int_move_to_ub(targets_ub,
                                self.targets,
                                data_move_pad_burst=self._s * self.targets_dsize,
                                data_move_burst=self.s_block // self.targets_block,
                                offset=task_idx * self.label_align)

        # func: block invalid tasks
        with self.tik_instance.if_scope(tik.all(t_i >= s_i, t_i > 0)):
            # func: get valid compute trace
            repeats, s_inc, e_inc = self.count_trace(s_i, targets_ub)

            start = self.tik_instance.Scalar("int32")
            end_loop = self.tik_instance.Scalar("int32")
            end = self.tik_instance.Scalar("int32")
            remain = self.tik_instance.Scalar("int32")
            current_target = self.tik_instance.Scalar(self.targets_dtype, init_value=self.blank)
            next_target = self.tik_instance.Scalar(self.targets_dtype)
            tmp = self.tik_instance.Scalar("int32")
            a_tmp = self.tik_instance.Scalar("float32")
            b_tmp = self.tik_instance.Scalar("float32")
            lcab = self.tik_instance.Scalar("float32")
            res = self.tik_instance.Scalar("float32")
            lp = self.tik_instance.Scalar("float32")
            nll = self.tik_instance.Scalar("float32", init_value=neg_log_likelihood_ub[0])
            grad_out = self.tik_instance.Scalar("float32", init_value=grad_out_ub[0])
            min_float = self.tik_instance.Scalar("float32", init_value=Constant.MIN)

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
            a_ub = self.tik_instance.Tensor("float32", [Constant.BLOCK], name="a_ub", scope=tik.scope_ubuf)
            b_ub = self.tik_instance.Tensor("float32", [Constant.BLOCK], name="b_ub", scope=tik.scope_ubuf)
            tmp_ub = self.tik_instance.Tensor("float32", [Constant.BLOCK], name="tmp_ub", scope=tik.scope_ubuf)

            work_tensor_ub = self.tik_instance.Tensor("float32", [Constant.BLOCK], name="work_tensor_ub",
                                                      scope=tik.scope_ubuf)
            log_probs_ub = self.tik_instance.Tensor("float32", [self.c_block], name="log_probs_ub",
                                                    scope=tik.scope_ubuf)

            # beta_grad_update
            repeat_times = self.tik_instance.Scalar("int32")
            c_tmp = self.tik_instance.Scalar("float32")
            zero_tmp = self.tik_instance.Scalar("float32", init_value=0)
            alpha_beta_tmp = self.tik_instance.Scalar("float32")
            max_tmp = self.tik_instance.Scalar("float32")

            copy_ub_a = self.tik_instance.Tensor("float32", [self.c_block], name="copy_ub_a", scope=tik.scope_ubuf)
            copy_ub_b = self.tik_instance.Tensor("float32", [self.c_block], name="copy_ub_b", scope=tik.scope_ubuf)

            # func: get log_prob in current T
            self.select_data_move(log_probs_ub[0],
                                  self.log_probs[self._c * task_idx + self._n * self._c * (t_i - 1)],
                                  data_move_pad_burst=self._c * Constant.FP32_BYTES,
                                  data_move_burst=self.c_block // Constant.BLOCK)

            output_dst = self.tik_instance.Scalar("int32", init_value=0)
            output_src = self.tik_instance.Scalar("int32", init_value=self.output_size_up)
            # func: calculate log_beta in current T
            log_beta_ub = self.tik_instance.Tensor("float32", [Constant.CONSTANT_COE, self.output_size_up],
                                                   name="log_beta_ub", scope=tik.scope_ubuf)
            current_target_ub = self.tik_instance.Tensor("int32", [self.output_size_up], name="current_target_ub",
                                                         scope=tik.scope_ubuf)
            self.tik_instance.vector_dup(Constant.BLOCK, log_beta_ub[output_dst], Constant.MIN,
                                         self.output_size_up // Constant.BLOCK, 1, 1)
            lamax_ub = self.tik_instance.Tensor("float32", [self.output_size_up], name="lamax_ub", scope=tik.scope_ubuf)
            with self.tik_instance.if_scope(s_i > -1):
                log_beta_ub[output_dst + Constant.CONSTANT_COE * s_i].set_as(log_probs_ub[self.blank])
            with self.tik_instance.if_scope(s_i > 0):
                current_target.set_as(targets_ub[s_i - 1])
            with self.tik_instance.if_scope(tik.all(s_i > 0, current_target > -1)):
                log_beta_ub[output_dst + Constant.CONSTANT_COE * s_i - 1].set_as(log_probs_ub[current_target])
            # func: get log_alpha in current T
            log_alpha_ub = self.tik_instance.Tensor("float32", [self.output_size_up], name="log_alpha_ub",
                                                    scope=tik.scope_ubuf)
            self.select_data_move(log_alpha_ub,
                                  self.log_alpha[task_idx * self.alpha_size_up + (t_i - 1) * self.output_size],
                                  data_move_pad_burst=self.output_size * Constant.FP32_BYTES,
                                  data_move_burst=self.output_size_up // Constant.BLOCK)

            with self.tik_instance.if_scope(s_i > 0):
                a_ub[0].set_as(log_alpha_ub[output_dst + Constant.CONSTANT_COE * s_i])
                a_ub[1].set_as(log_alpha_ub[output_dst + Constant.CONSTANT_COE * s_i - 1])
                b_ub[0].set_as(log_beta_ub[output_dst + Constant.CONSTANT_COE * s_i])
                b_ub[1].set_as(log_beta_ub[output_dst + Constant.CONSTANT_COE * s_i - 1])
            self.tik_instance.vec_add(Constant.CONSTANT_COE, tmp_ub, a_ub, b_ub, 1, 1, 1, 1)
            # func: update grad_ub in current T with log_alpha and log_beta
            grad_ub[self.blank].set_as(tmp_ub[0])
            with self.tik_instance.if_scope(current_target > -1):
                grad_ub[current_target].set_as(tmp_ub[1])

            with self.tik_instance.if_scope(s_i > 0):
                start.set_as(Constant.CONSTANT_COE * s_i - 1)
            with self.tik_instance.else_scope():
                start.set_as(0)

            with self.tik_instance.if_scope(repeats < t_i - s_i):
                with self.tik_instance.if_scope(s_i > -1):
                    end.set_as(Constant.CONSTANT_COE * s_i + 1)
                with self.tik_instance.else_scope():
                    end.set_as(0)
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(s_i > -1):
                    end.set_as(Constant.CONSTANT_COE * s_i + 1)
                with self.tik_instance.else_scope():
                    end.set_as(0)

            t = self.tik_instance.Scalar("int32", init_value=t_i - 1)

            # func: `grad = exp(log_probs)`
            self.tik_instance.h_exp(copy_ub_a, log_probs_ub)

            c_loop = self.tik_instance.Scalar("int32")
            c_loop.set_as((self._c + Constant.C_ALIGN - 1) // Constant.C_ALIGN)

            # apply for filter
            work_tensor_ub = self.tik_instance.Tensor("float32", [c_loop], tik.scope_ubuf, "work_tensor_ub")
            sum_fp32_ub = self.tik_instance.Tensor("float32", [c_loop, 8], name="sum_fp32_ub", scope=tik.scope_ubuf)
            sum_val = self.tik_instance.Scalar("float32")
            c_offset = self.tik_instance.Scalar("int32")

            self.tik_instance.h_abs(grad_abs_ub, grad_ub)
            # count the valid num per batch
            with self.tik_instance.for_range(0, c_loop) as loop_idx:
                self.tik_instance.vec_reduce_add(Constant.C_ALIGN, sum_fp32_ub[loop_idx, 0],
                                                 grad_abs_ub[loop_idx * Constant.C_ALIGN], work_tensor_ub, 1, 0)
            # pick up valid res from each batch
            with self.tik_instance.for_range(0, c_loop) as loop_idx:
                # Filter invalid scenarios
                sum_val.set_as(sum_fp32_ub[loop_idx, 0])
                with self.tik_instance.if_scope(sum_val != 0):
                    with self.tik_instance.for_range(0, Constant.C_ALIGN) as current_idx:
                        c_offset.set_as(current_idx + loop_idx * Constant.C_ALIGN)
                        res.set_as(grad_ub[c_offset])
                        # func: update certain grad
                        with self.tik_instance.if_scope(res != 0):
                            with self.tik_instance.if_scope(c_offset < self._c):
                                lp.set_as(log_probs_ub[c_offset])
                                a_ub[0].set_as(res + nll - lp)
                                self.tik_instance.vec_exp(1, b_ub, a_ub, 1, 1, 1)

                                a_tmp.set_as(copy_ub_a[c_offset])
                                b_tmp.set_as(b_ub[0])
                                copy_ub_a[c_offset].set_as(a_tmp - b_tmp)

            self.tik_instance.h_mul(copy_ub_b, copy_ub_a, grad_out)

            with self.tik_instance.for_range(self._c, self.c_block) as tail_idx:
                copy_ub_b[tail_idx].set_as(0)
            self.select_data_move(self.grad[t * self._n * self._c + task_idx * self._c],
                                  copy_ub_b[0],
                                  data_move_pad_burst=self._c * Constant.FP32_BYTES,
                                  data_move_burst=self.c_block // Constant.BLOCK)

            with self.tik_instance.for_range(1, t_i):
                t.set_as(t - 1)

                # func: initial grad_ub
                with self.tik_instance.for_range(0, self.rounds) as j:
                    self.tik_instance.vector_dup(Constant.BLOCK,
                                                 grad_ub[(Constant.BLOCK * Constant.REPEAT_OFFSET) * j],
                                                 0, Constant.REPEAT_OFFSET, 1, 1)
                self.tik_instance.vector_dup(
                    Constant.BLOCK, grad_ub[(Constant.BLOCK * Constant.REPEAT_OFFSET) * self.rounds], 0,
                    (self.c_block % (Constant.BLOCK * Constant.REPEAT_OFFSET)) // Constant.BLOCK, 1, 1)

                self.select_data_move(log_probs_ub[0],
                                      self.log_probs[self._c * task_idx + self._n * self._c * t],
                                      data_move_pad_burst=self._c * Constant.FP32_BYTES,
                                      data_move_burst=self.c_block // Constant.BLOCK)
                self.select_data_move(log_alpha_ub[0],
                                      self.log_alpha[task_idx * self.alpha_size_up + t * self.output_size],
                                      data_move_pad_burst=self.output_size * Constant.FP32_BYTES,
                                      data_move_burst=self.output_size_up // Constant.BLOCK)
                self.tik_instance.vector_dup(Constant.BLOCK, log_beta_ub[output_src], Constant.MIN,
                                             self.output_size_up // Constant.BLOCK, 1, 1)

                remain.set_as(s_i + repeats - t_i + t)
                with self.tik_instance.if_scope(remain >= -1):
                    tmp.set_as(s_inc[remain + 1])
                    start.set_as(start - tmp)
                with self.tik_instance.if_scope(t < s_i + repeats):
                    tmp.set_as(e_inc[t])
                    end.set_as(end - tmp)
                end_loop.set_as(end)

                with self.tik_instance.if_scope(end_loop == Constant.CONSTANT_COE * s_i + 1):
                    current_target.set_as(self.blank)
                    with self.tik_instance.if_scope(s_i > -1):
                        a_tmp.set_as(log_beta_ub[output_dst + Constant.CONSTANT_COE * s_i])
                    b_tmp.set_as(log_probs_ub[self.blank])
                    # func: calculate log_beta in current T
                    with self.tik_instance.if_scope(s_i > -1):
                        log_beta_ub[output_src + Constant.CONSTANT_COE * s_i].set_as(a_tmp + b_tmp)
                        a_tmp.set_as(log_beta_ub[output_src + Constant.CONSTANT_COE * s_i])
                        b_tmp.set_as(log_alpha_ub[Constant.CONSTANT_COE * s_i])
                    end_loop.set_as(end_loop - 1)
                    # func: update grad_ub in current T with log_alpha and log_beta
                    grad_ub[current_target].set_as(a_tmp + b_tmp)

                with self.tik_instance.for_range(start, end_loop) as s:
                    with self.tik_instance.if_scope(s % Constant.CONSTANT_COE == 0):
                        current_target.set_as(self.blank)
                    with self.tik_instance.else_scope():
                        current_target.set_as(targets_ub[s // Constant.CONSTANT_COE])
                    current_target_ub[s].set_as(current_target)

                    offset.set_as(s - start)
                    with self.tik_instance.if_scope(current_target > -1):
                        current_log_probs_ub[offset].set_as(log_probs_ub[current_target])
                    lamda1_ub[offset].set_as(log_beta_ub[output_dst + s])
                    lamda2_ub[offset].set_as(log_beta_ub[output_dst + s + 1])

                    with self.tik_instance.if_scope(
                            tik.all((s % Constant.CONSTANT_COE != 0), (s < Constant.CONSTANT_COE * s_i - 1))):
                        next_target.set_as(targets_ub[s // Constant.CONSTANT_COE + 1])
                        with self.tik_instance.if_scope(current_target != next_target):
                            lamda3_ub[offset].set_as(log_beta_ub[output_dst + s + Constant.CONSTANT_COE])
                        with self.tik_instance.else_scope():
                            lamda3_ub[offset].set_as(min_float)
                    with self.tik_instance.else_scope():
                        lamda3_ub[offset].set_as(min_float)

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
                with self.tik_instance.for_range(start, end_loop) as s:
                    current_target.set_as(current_target_ub[s])
                    offset.set_as(s - start)

                    a_tmp.set_as(compute_tmp_ub[offset])
                    log_beta_ub[output_src + s].set_as(a_tmp)
                    # func: get log_alpha in current T
                    b_tmp.set_as(log_alpha_ub[s])
                    alpha_beta_tmp.set_as(a_tmp + b_tmp)
                    with self.tik_instance.if_scope(current_target > -1):
                        lcab.set_as(grad_ub[current_target])

                    # func: update grad_ub in current T with log_alpha and log_beta
                    with self.tik_instance.if_scope(lcab != 0):
                        with self.tik_instance.if_scope(lcab > alpha_beta_tmp):
                            max_tmp.set_as(lcab)
                            a_ub[0].set_as(zero_tmp)
                            a_ub[1].set_as(alpha_beta_tmp - lcab)
                        with self.tik_instance.else_scope():
                            max_tmp.set_as(alpha_beta_tmp)
                            a_ub[0].set_as(lcab - alpha_beta_tmp)
                            a_ub[1].set_as(zero_tmp)

                        self.tik_instance.vec_exp(Constant.CONSTANT_COE, b_ub, a_ub, 1, 0, 0)
                        self.tik_instance.vec_reduce_add(Constant.CONSTANT_COE, a_ub, b_ub, work_tensor_ub, 1, 1)

                        self.tik_instance.vln(1, b_ub, a_ub, 1, 1, 1, 1, 1)
                        a_tmp.set_as(b_ub[0])
                        with self.tik_instance.if_scope(current_target > -1):
                            grad_ub[current_target].set_as(a_tmp + max_tmp)
                    with self.tik_instance.else_scope():
                        with self.tik_instance.if_scope(current_target > -1):
                            grad_ub[current_target].set_as(alpha_beta_tmp)

                # func: `grad = exp(log_probs)`
                self.tik_instance.h_exp(copy_ub_a, log_probs_ub)

                self.tik_instance.h_abs(grad_abs_ub, grad_ub)
                # count the valid num per batch
                with self.tik_instance.for_range(0, c_loop) as loop_idx:
                    self.tik_instance.vec_reduce_add(Constant.C_ALIGN, sum_fp32_ub[loop_idx, 0],
                                                     grad_abs_ub[loop_idx * Constant.C_ALIGN], work_tensor_ub, 1, 0)
                # pick up valid res from each batch
                with self.tik_instance.for_range(0, c_loop) as loop_idx:
                    # Filter invalid scenarios
                    sum_val.set_as(sum_fp32_ub[loop_idx, 0])
                    with self.tik_instance.if_scope(sum_val != 0):
                        with self.tik_instance.for_range(0, Constant.C_ALIGN) as current_idx:
                            c_offset.set_as(current_idx + loop_idx * Constant.C_ALIGN)
                            res.set_as(grad_ub[c_offset])
                            # func: update certain grad
                            with self.tik_instance.if_scope(res != 0):
                                with self.tik_instance.if_scope(c_offset < self._c):
                                    lp.set_as(log_probs_ub[c_offset])
                                    a_ub[0].set_as(res + nll - lp)
                                    self.tik_instance.vec_exp(1, b_ub, a_ub, 1, 1, 1)

                                    a_tmp.set_as(copy_ub_a[c_offset])
                                    b_tmp.set_as(b_ub[0])
                                    copy_ub_a[c_offset].set_as(a_tmp - b_tmp)
                self.tik_instance.h_mul(copy_ub_b, copy_ub_a, grad_out)

                with self.tik_instance.for_range(self._c, self.c_block) as tail_idx:
                    copy_ub_b[tail_idx].set_as(0)

                self.select_data_move(self.grad[t * self._n * self._c + task_idx * self._c],
                                      copy_ub_b[0],
                                      data_move_pad_burst=self._c * Constant.FP32_BYTES,
                                      data_move_burst=self.c_block // Constant.BLOCK)
                output_src.set_as(output_dst)
                output_dst.set_as(self.output_size_up - output_src)
        with self.tik_instance.elif_scope(not self.zero_infinity):
            # fill 3.4e+38 to represent nan
            res_ub = self.tik_instance.Tensor("float32", [self.c_block], name="res_ub", scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, self.rounds) as j:
                self.tik_instance.vector_dup(Constant.BLOCK,
                                             res_ub[(Constant.BLOCK * Constant.REPEAT_OFFSET) * j],
                                             Constant.MAX, Constant.REPEAT_OFFSET, 1, 1)
            self.tik_instance.vector_dup(
                Constant.BLOCK, res_ub[(Constant.BLOCK * Constant.REPEAT_OFFSET) * self.rounds], Constant.MAX,
                (self.c_block % (Constant.BLOCK * Constant.REPEAT_OFFSET)) // Constant.BLOCK, 1, 1)
            with self.tik_instance.for_range(self._c, self.c_block) as tail_idx:
                res_ub[tail_idx].set_as(0)
            with self.tik_instance.for_range(0, t_i) as t:
                self.select_data_move(self.grad[t * self._n * self._c + task_idx * self._c],
                                      res_ub,
                                      data_move_pad_burst=self._c * Constant.FP32_BYTES,
                                      data_move_burst=self.c_block // Constant.BLOCK)

    def count_trace(self, s_i, targets_ub):
        """
        Function: mark the valid trace.

        Init base parameters
        Parameters
        ----------
        Inputs:
        s_i: label length.
        targets_ub: label index.
        ----------
        """
        s_inc = self.tik_instance.Tensor("int32", [self.output_size], name="s_inc", scope=tik.scope_ubuf)
        e_inc = self.tik_instance.Tensor("int32", [self.output_size], name="e_inc", scope=tik.scope_ubuf)

        one_step = self.tik_instance.Scalar("int32", init_value=1)
        two_step = self.tik_instance.Scalar("int32", init_value=2)

        left = self.tik_instance.Scalar(self.targets_dtype)
        right = self.tik_instance.Scalar(self.targets_dtype)

        repeats = self.tik_instance.Scalar("int32", init_value=0)
        idx_counter = self.tik_instance.Scalar("int32", init_value=1)

        s_inc[0].set_as(one_step)
        with self.tik_instance.if_scope(s_i > 0):
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


# 'pylint: disable=invalid-name,too-many-locals,too-many-arguments,unused-argument
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def ctc_loss_v2_grad(grad_out, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, grad,
                     blank=0, reduction="mean", zero_infinity=False, kernel_name="ctc_loss_v2_grad"):
    """
    Function: The grad of Connectionist Temporal Classification loss.

    Init base parameters
    Parameters
    ----------
    Inputs:
    grad_out: Gradient renewal coefficient. Tensor of size (N), where N = batch size.
    Log_probs: Tensor of size (T,N,C), where T =input length, N =batch size,
               and C = number of classes (including blank).
    Targets: Tensor of size (N, S), where S= max target length.
    It represent the target sequences.
    Input_lengths: Tuple or tensor of size (N).
    It represent the lengths of the inputs.
    Target_lengths: Tuple or tensor of size (N). It represent lengths of the targets.
    log_alpha: The probability of possible trace of input to target.
    neg_log_likelihood: A loss value which is differentiable with respect to each input node.

    Attributes:
    blank : Blank label. Default 0.
    reduction: Specifies the reduction to apply to the output. Default: 'mean'.
    zero_infinity : Whether to zero infinite losses and the associated gradients.

    Outputs:
    grad: The grad of Connectionist Temporal Classification loss.
    ----------
    """
    op_obj = CTCLossV2Grad(targets, input_lengths, target_lengths, blank, zero_infinity, kernel_name)
    return op_obj.ctc_loss_grad_compute()
