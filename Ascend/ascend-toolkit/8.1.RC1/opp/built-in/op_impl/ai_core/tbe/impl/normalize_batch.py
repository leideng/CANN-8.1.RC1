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
normalize_batch
"""
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector

from impl.ascend import AContainer
from impl.ascend import TensorOperatorParam
from impl.ascend import VecCmd
from impl.ascend import VecExecutor


class NormalizeBatch(object):

    def __init__(self, n_num, c_num, d_num, normalize_type, epsilon, data_type, int_type, kernel_name, cont):
        self.n_num = n_num
        self.c_num = c_num
        self.d_num = d_num
        self.normalize_type = normalize_type
        self.epsilon = epsilon
        self.data_type = data_type
        self.int_type = int_type
        self.kernel_name = kernel_name

        self.cont = cont
        self.tik = self.cont.tik
        self.tik_inst = self.cont.tinst
        self.ai_core_use = self.cont.const_aicore_num
        self.ub_size = self.cont.const_ub_max_byte
        self.block_size = self.cont.const_block_byte

        self.data_size, self.data_block_data_num, self.data_repeat_data_num = self.get_type_const(
            self.data_type)
        self.int_size, self.int_block_data_num, self.int_repeat_data_num = self.get_type_const(
            self.int_type)

        self.d_num_align = self.get_align_num(
            self.d_num, self.data_repeat_data_num)
        input_shape = (self.n_num, self.c_num, self.d_num)
        seq_shape = (self.n_num, )
        self.input_x = self.tik_inst.Tensor(
            self.data_type, input_shape, self.tik.scope_gm, "input_x")
        self.seq_len = self.tik_inst.Tensor(
            self.int_type, seq_shape, self.tik.scope_gm, "seq_len")
        is_atomic_add = (self.d_num % self.data_block_data_num != 0)
        self.output_y = self.tik_inst.Tensor(self.data_type, input_shape, self.tik.scope_gm, "output_y",
                                             is_atomic_add=is_atomic_add)

    def ceil_div(self, dividend_, divisor_):
        result_ = (dividend_ + divisor_ - 1) // divisor_
        return result_

    def get_loop_info(self, all_data_num_, each_loop_num_):
        loop_times_ = self.ceil_div(all_data_num_, each_loop_num_)
        last_loop_num_ = all_data_num_ - each_loop_num_ * (loop_times_ - 1)
        return loop_times_, last_loop_num_

    def get_align_num(self, input_num_, align_num_, ceil=True):
        if ceil:
            result_ = self.ceil_div(input_num_, align_num_) * align_num_
        else:
            result_ = input_num_ // align_num_ * align_num_
        return result_

    def get_type_const(self, data_type):
        data_size = self.cont.const_dtype_byte.get(data_type)
        block_data_num = self.cont.get_vec_proc_num_per_cmd_blk(data_type)
        repeat_data_num = self.cont.get_vec_proc_num_per_cmd(data_type)
        return data_size, block_data_num, repeat_data_num

    def get_mask(self, data_num, format_num_floor, format_num_ceil):
        # mask_h, mask_l split num is 64
        mask_split = 64
        data_num_floor = self.get_align_num(data_num, format_num_floor, False)
        data_num_ceil = self.get_align_num(data_num, format_num_ceil, True)
        mask_h, mask_l = 0, 0
        index_start = data_num - data_num_floor
        index_end = data_num_ceil - data_num_floor
        for index_l in range(index_start, min(index_end, mask_split)):
            mask_l += 2 ** index_l
        for index_h in range(max(mask_split, index_start), index_end):
            mask_h += 2 ** (index_h - mask_split)
        return mask_h, mask_l, data_num_floor

    def mode_compute(self):
        if self.normalize_type == "per_feature":
            batch_num = self.n_num * self.c_num
        else:
            batch_num = self.n_num
        self._mode_compute(batch_num)
        inputs_all = [self.input_x, self.seq_len]
        outputs_all = [self.output_y]
        self.tik_inst.BuildCCE(
            inputs=inputs_all,
            outputs=outputs_all,
            kernel_name=self.kernel_name)

    def _mode_compute(self, batch_num):
        each_core_batch_num = self.ceil_div(batch_num, self.ai_core_use)
        self.ai_core_use, last_core_batch_num = self.get_loop_info(
            batch_num, each_core_batch_num)
        with self.tik_inst.for_range(0, self.ai_core_use, block_num=self.ai_core_use) as core_index_s:
            batch_index_core_s = each_core_batch_num * core_index_s
            with self.tik_inst.if_scope(core_index_s != self.ai_core_use - 1):
                self._mode_compute_each_core(
                    batch_index_core_s, each_core_batch_num)
            with self.tik_inst.else_scope():
                self._mode_compute_each_core(
                    batch_index_core_s, last_core_batch_num)

    def _mode_compute_each_core(self, batch_index_core_s, batch_num):
        if self.normalize_type == "per_feature":
            per_compute, thread_num = self._get_per_split_mode_thread_num()
            with self.tik_inst.for_range(0, batch_num, thread_num=min(batch_num, thread_num)) as batch_index_s:
                batch_index_loop_s = batch_index_core_s + batch_index_s
                per_compute(batch_index_loop_s)
        else:
            per_compute = self._get_all_split_mode()
            with self.tik_inst.for_range(0, batch_num) as batch_index_s:
                batch_index_loop_s = batch_index_core_s + batch_index_s
                per_compute(batch_index_loop_s)

    def _get_per_split_mode_thread_num(self):
        """
        per_feature: get tiling mode split n or split d, and thread num
        """
        each_batch_size = self._get_split_n_size()
        if each_batch_size > self.ub_size:
            per_compute = self._per_compute_d
            thread_num = 1
        elif 2 * each_batch_size > self.ub_size:
            per_compute = self._per_compute_n
            thread_num = 1
        else:
            per_compute = self._per_compute_n
            thread_num = 2
        return per_compute, thread_num

    def _get_all_split_mode(self):
        """
        all_features: get tiling mode, split n or split d
        """
        each_batch_size = self._get_split_n_size(self.c_num)
        if each_batch_size > self.ub_size:
            per_compute = self._all_compute_d
        else:
            per_compute = self._all_compute_n
        return per_compute

    def _get_split_n_size(self, each_loop_batch_num=1):
        """
        algorithm: Count ub size split n.
        Parameters
        ----------
        each_loop_batch_num:
            int. Each loop batch num
        Returns
        -------
        ub_size
        """
        # mean ub size
        mean_ub_size = self.block_size * 3
        # data size, work ub size
        repeat_num = self.d_num_align // self.data_repeat_data_num
        work_ub_size = self.data_size * repeat_num + self.block_size
        data_ub_size = each_loop_batch_num * self.data_size * self.d_num_align * 2
        # init seq_len ub size
        seq_size = self.block_size * 2
        # size sum
        each_batch_size = mean_ub_size + work_ub_size + data_ub_size + seq_size
        return each_batch_size

    def _per_compute_n(self, batch_index_s):
        """
        algorithm: per_feature, split n, count normalize.
        """
        n_index_s, c_index_s = self._per_init_index(batch_index_s)
        # init tensor
        mean_buf = self._init_mean()
        data_buf = self._init_data_tensor_n()
        # data move in
        self._data_move_in(data_buf, n_index_s, c_index_s, 0, 0, self.d_num)
        # get seq_len
        seq_len_i_s, seq_len_f_s, unbiased_f_s = self._init_seq_len_s(
            n_index_s)
        # count sum(x) and sum(x^2)
        self._per_count_sum_n(data_buf, mean_buf, seq_len_i_s)
        # count rec_std and ne_mean
        self._count_rec_std_ne_mean(mean_buf, seq_len_f_s, unbiased_f_s)
        # normalize
        ne_mean_s, rec_std_s = self._get_ne_mean_rec_std_scalar(mean_buf)
        self._stand_data(data_buf, ne_mean_s, rec_std_s)
        # data move out
        self._data_move_out(data_buf, n_index_s, c_index_s, 0, 0, self.d_num)

    def _init_data_tensor_n(self, batch_num=1):
        """
        algorithm: init data tensor and work tensor
        """
        data_num_align = self.d_num_align
        data_shape = (batch_num, data_num_align)
        proc_num = batch_num * data_num_align
        repeat_num = data_num_align // self.data_repeat_data_num
        work_tensor_num = self.get_align_num(
            repeat_num, self.data_block_data_num)
        input_data_ub = self.tik_inst.Tensor(self.data_type, data_shape,
                                             self.tik.scope_ubuf, "input_data_ub")
        input_data_square_ub = self.tik_inst.Tensor(self.data_type, data_shape,
                                                    self.tik.scope_ubuf, "input_data_square_ub")
        work_tensor_ub = self.tik_inst.Tensor(self.data_type, (work_tensor_num, ),
                                              self.tik.scope_ubuf, "work_tensor_ub")
        data_buf = {
            "input_data_ub": TensorOperatorParam(input_data_ub, proc_num, 0),
            "input_data_square_ub": TensorOperatorParam(input_data_square_ub, proc_num, 0),
            "work_tensor_ub": TensorOperatorParam(work_tensor_ub, work_tensor_num, 0)}
        return data_buf

    def _per_count_sum_n(self, data_buf, mean_buf, seq_len_i_s):
        input_data_ub = data_buf.get("input_data_ub").const_tensor
        input_data_square_ub = data_buf.get(
            "input_data_square_ub").const_tensor
        work_tensor_ub = data_buf.get("work_tensor_ub").const_tensor
        sum_ub = mean_buf.get("batch_mean_ub").const_tensor
        square_sum_ub = mean_buf.get("batch_mean_square_ub").const_tensor

        self._vector_mean_tensor(mean_buf)
        cmd_square_ub = [VecCmd(cmd_name="vmul", dst_name="input_data_square_ub",
                                src0_name="input_data_ub", src1_name="input_data_ub")]
        VecExecutor.exec_vec_cmd(data_buf, cmd_square_ub, "input_data_ub")
        self._vector_reduce_add(sum_ub, input_data_ub,
                                work_tensor_ub, seq_len_i_s, 0)
        self._vector_reduce_add(
            square_sum_ub, input_data_square_ub, work_tensor_ub, seq_len_i_s, 0)

    def _per_compute_d(self, batch_index_s):
        """
        algorithm: per_feature, split d, count normalize.
        """
        n_index_s, c_index_s = self._per_init_index(batch_index_s)
        # init mean tensor
        mean_buf = self._init_mean()
        seq_len_i_s, seq_len_f_s, unbiased_f_s = self._init_seq_len_s(
            n_index_s)
        # count sum(x) and sum(x^2)
        self._per_count_sum_d(mean_buf, n_index_s, c_index_s, seq_len_i_s)
        # count rec_std and ne_mean
        self._count_rec_std_ne_mean(mean_buf, seq_len_f_s, unbiased_f_s)
        # normalize and move out
        self._per_stand_data_d(mean_buf, n_index_s, c_index_s)

    def _per_count_sum_d(self, mean_buf, n_index_, c_index_, seq_len_i_s):
        each_loop_d_num = self._get_loop_num_count_sum_d()
        loop_times_s, last_loop_index_s, last_loop_d_num_s = self._init_loop_info(
            seq_len_i_s, each_loop_d_num)
        self._vector_mean_tensor(mean_buf)
        with self.tik_inst.new_stmt_scope():
            data_buf = self._init_data_tensor_d(each_loop_d_num)
            with self.tik_inst.if_scope(loop_times_s > 0):
                with self.tik_inst.for_range(0, loop_times_s) as loop_index_s:
                    d_index_s = loop_index_s * each_loop_d_num
                    self._per_count_sum_d_each_loop(data_buf, mean_buf, n_index_, c_index_,
                                                    d_index_s, each_loop_d_num)
            with self.tik_inst.if_scope(last_loop_d_num_s > 0):
                self._per_count_sum_d_each_loop(data_buf, mean_buf, n_index_, c_index_,
                                                last_loop_index_s, last_loop_d_num_s)

    def _get_loop_num_count_sum_d(self):
        mean_ub_size = self.data_size * self.data_block_data_num * 3
        ub_size_last = self.ub_size - mean_ub_size - self.block_size
        each_repeat_size = self.data_size * self.data_repeat_data_num + self.data_size
        repeat_num = ub_size_last // each_repeat_size
        each_loop_d_num = repeat_num * self.data_repeat_data_num
        return each_loop_d_num

    def _per_count_sum_d_each_loop(self, data_buf, mean_buf, n_index_, c_index_, d_index_, d_num_):
        self._data_move_in(data_buf, n_index_, c_index_, d_index_, 0, d_num_)
        input_data_ub = data_buf.get("input_data_ub").const_tensor
        work_tensor_ub = data_buf.get("work_tensor_ub").const_tensor
        sum_ub = mean_buf.get("batch_mean_ub").const_tensor
        square_sum_ub = mean_buf.get("batch_mean_square_ub").const_tensor
        temp_ub = mean_buf.get("batch_variance_ub").const_tensor
        cmd_square_vec = [VecCmd(cmd_name="vmul", dst_name="input_data_ub",
                                 src0_name="input_data_ub", src1_name="input_data_ub")]
        self._vector_reduce_add_d(
            temp_ub, input_data_ub, work_tensor_ub, d_num_, 0)
        self.tik_inst.vadd(1, sum_ub, sum_ub, temp_ub, 1, 1, 1, 1, 8, 8, 8)
        VecExecutor.exec_vec_cmd(data_buf, cmd_square_vec, "input_data_ub")
        self._vector_reduce_add_d(
            temp_ub, input_data_ub, work_tensor_ub, d_num_, 0)
        self.tik_inst.vadd(1, square_sum_ub, square_sum_ub,
                           temp_ub, 1, 1, 1, 1, 8, 8, 8)

    def _vector_reduce_add_d(self, result_ub, data_ub, work_ub, d_num_, n_index_ub_):
        if isinstance(d_num_, int):
            mask = self.data_repeat_data_num
            repeat_num = d_num_ // mask
            self.tik_inst.vec_reduce_add(
                mask, result_ub, data_ub[n_index_ub_, 0], work_ub, repeat_num, 8)
        else:
            self._vector_reduce_add(
                result_ub, data_ub, work_ub, d_num_, n_index_ub_)

    def _per_stand_data_d(self, mean_buf, n_index_, c_index_):
        each_loop_d_num = self._get_loop_num_stand_data_d()
        loop_times, last_loop_d_num = self.get_loop_info(
            self.d_num, each_loop_d_num)
        ne_mean_s, rec_std_s = self._get_ne_mean_rec_std_scalar(mean_buf)
        with self.tik_inst.for_range(0, loop_times) as loop_index_s:
            d_index_s = loop_index_s * each_loop_d_num
            data_buf = self._init_data_tensor_d(each_loop_d_num, False)
            with self.tik_inst.if_scope(loop_index_s != loop_times - 1):
                self._per_stand_data_d_each_loop(data_buf, ne_mean_s, rec_std_s,
                                                 n_index_, c_index_, d_index_s, each_loop_d_num)
            with self.tik_inst.else_scope():
                self._per_stand_data_d_each_loop(data_buf, ne_mean_s, rec_std_s,
                                                 n_index_, c_index_, d_index_s, last_loop_d_num)

    def _get_loop_num_stand_data_d(self):
        mean_ub_size = self.data_size * self.data_block_data_num * 3
        ub_size_last = self.ub_size - mean_ub_size
        each_repeat_size = self.data_size * self.data_repeat_data_num
        repeat_num = ub_size_last // each_repeat_size
        each_loop_d_num = repeat_num * self.data_repeat_data_num
        return each_loop_d_num

    def _per_stand_data_d_each_loop(self, data_buf, ne_mean_, rec_std_, n_index_, c_index_, d_index_, d_num):
        self._data_move_in(data_buf, n_index_, c_index_, d_index_, 0, d_num)
        self._stand_data(data_buf, ne_mean_, rec_std_)
        self._data_move_out(data_buf, n_index_, c_index_, d_index_, 0, d_num)

    def _init_data_tensor_d(self, data_num_align, init_work_tensor=True):
        batch_num = 1
        data_shape = (batch_num, data_num_align)
        proc_num = batch_num * data_num_align
        input_data_ub = self.tik_inst.Tensor(
            self.data_type, data_shape, self.tik.scope_ubuf, "input_data_ub")
        data_buf = {"input_data_ub": TensorOperatorParam(
            input_data_ub, proc_num, 0)}
        if init_work_tensor:
            repeat_num = data_num_align // self.data_repeat_data_num
            work_tensor_num = self.get_align_num(
                repeat_num, self.data_block_data_num)

            work_tensor_ub = self.tik_inst.Tensor(self.data_type, (work_tensor_num,),
                                                  self.tik.scope_ubuf, "work_tensor_ub")
            data_buf["work_tensor_ub"] = TensorOperatorParam(
                work_tensor_ub, work_tensor_num, 0)
        return data_buf

    def _per_init_index(self, batch_index_s):
        n_index_s = self.tik_inst.Scalar(batch_index_s.dtype)
        c_index_s = self.tik_inst.Scalar(batch_index_s.dtype)
        n_index_s.set_as(batch_index_s // self.c_num)
        c_index_s.set_as(batch_index_s % self.c_num)
        return n_index_s, c_index_s

    def _all_compute_n(self, n_index_s):
        """
        algorithm: all_features, split n, count normalize.
        """
        # init tensor
        mean_buf = self._init_mean()
        data_buf = self._init_data_tensor_n(self.c_num)
        self._all_data_move_in_n(data_buf, n_index_s)
        seq_len_i_s, seq_len_f_s, unbiased_f_s = self._init_seq_len_s(
            n_index_s, self.c_num)
        # count sum(x) and sum(x^2)
        self._all_count_sum_n(data_buf, mean_buf, seq_len_i_s)
        # count rec_std and ne_mean
        self._count_rec_std_ne_mean(mean_buf, seq_len_f_s, unbiased_f_s)
        ne_mean_s, rec_std_s = self._get_ne_mean_rec_std_scalar(mean_buf)
        # normalize data
        self._stand_data(data_buf, ne_mean_s, rec_std_s)
        # data move out
        self._all_data_move_out(data_buf, n_index_s)

    def _all_data_move_in_n(self, data_buf, n_index_):
        input_data_ub = data_buf.get("input_data_ub").const_tensor
        if self.d_num % self.data_block_data_num != 0:
            block_num = self.ceil_div(self.d_num, self.data_block_data_num)
            with self.tik_inst.for_range(0, self.c_num) as c_index_s:
                self.tik_inst.data_move(input_data_ub[c_index_s, 0], self.input_x[n_index_, c_index_s, 0],
                                        0, 1, block_num, 0, 0)
        else:
            block_num = self.d_num // self.data_block_data_num
            data_num_stride = self.d_num_align - self.d_num
            block_num_stride = data_num_stride // self.data_block_data_num
            self.tik_inst.data_move(input_data_ub, self.input_x[n_index_, 0, 0],
                                    0, self.c_num, block_num, 0, block_num_stride)

    def _all_count_sum_n(self, data_buf, mean_buf, seq_len_i_s):
        sum_ub = mean_buf.get("batch_mean_ub").const_tensor
        square_sum_ub = mean_buf.get("batch_mean_square_ub").const_tensor
        temp_ub = mean_buf.get("batch_variance_ub").const_tensor
        input_data_ub = data_buf.get("input_data_ub").const_tensor
        input_data_square_ub = data_buf.get(
            "input_data_square_ub").const_tensor
        work_tensor_ub = data_buf.get("work_tensor_ub").const_tensor

        self._vector_mean_tensor(mean_buf)
        cmd_square_ub = [VecCmd(cmd_name="vmul", dst_name="input_data_square_ub",
                                src0_name="input_data_ub", src1_name="input_data_ub")]
        VecExecutor.exec_vec_cmd(data_buf, cmd_square_ub, "input_data_ub")
        with self.tik_inst.for_range(0, self.c_num) as c_index_s:
            self._vector_reduce_add(
                temp_ub, input_data_ub, work_tensor_ub, seq_len_i_s, c_index_s)
            self.tik_inst.vadd(1, sum_ub, sum_ub, temp_ub, 1, 1, 1, 1, 8, 8, 8)
            self._vector_reduce_add(
                temp_ub, input_data_square_ub, work_tensor_ub, seq_len_i_s, c_index_s)
            self.tik_inst.vadd(1, square_sum_ub, square_sum_ub,
                               temp_ub, 1, 1, 1, 1, 8, 8, 8)

    def _all_data_move_out(self, data_buf, n_index_):
        input_data_ub = data_buf.get("input_data_ub").const_tensor
        if self.d_num % self.data_block_data_num != 0:
            block_num = self.ceil_div(self.d_num, self.data_block_data_num)
            mask_h, mask_l, mask_index = self.get_mask(
                self.d_num, self.data_repeat_data_num, self.data_block_data_num)
            with self.tik_inst.for_range(0, self.c_num) as c_index_s:
                self.tik_inst.vector_dup(
                    [mask_h, mask_l], input_data_ub[c_index_s, mask_index], 0.0, 1, 1, 8)
                add_mode = 1
                self.tik_inst.set_atomic_add(add_mode)
                self.tik_inst.data_move(self.output_y[n_index_, c_index_s, 0], input_data_ub[c_index_s, 0],
                                        0, 1, block_num, 0, 0)
                self.tik_inst.set_atomic_add(0)
        else:
            block_num = self.d_num // self.data_block_data_num
            data_num_stride = self.d_num_align - self.d_num
            block_num_stride = data_num_stride // self.data_block_data_num
            self.tik_inst.data_move(self.output_y[n_index_, 0, 0], input_data_ub,
                                    0, self.c_num, block_num, block_num_stride, 0)

    def _all_compute_d(self, n_index_s):
        """
        algorithm: all_features, split d, count normalize.
        """
        # init mean tensor
        mean_buf = self._init_mean()
        seq_len_i_s, seq_len_f_s, unbiased_f_s = self._init_seq_len_s(
            n_index_s, self.c_num)
        # count sum(x) and sum(x^2)
        self._all_count_sum_d(mean_buf, n_index_s, seq_len_i_s)
        # count rec_std and ne_mean
        self._count_rec_std_ne_mean(mean_buf, seq_len_f_s, unbiased_f_s)
        # normalize and move out
        self._all_stand_data_d(mean_buf, n_index_s)

    def _all_count_sum_d(self, mean_buf, n_index_, seq_len_i_s):
        """
        algorithm: start count sum.
        """
        each_loop_d_num = self._get_loop_num_count_sum_d()
        loop_times_s, last_loop_index_s, last_loop_d_num_s = self._init_loop_info(
            seq_len_i_s, each_loop_d_num)
        self._vector_mean_tensor(mean_buf)
        with self.tik_inst.new_stmt_scope():
            data_buf = self._init_data_tensor_d(each_loop_d_num)
            with self.tik_inst.for_range(0, self.c_num) as c_index_s:
                with self.tik_inst.if_scope(loop_times_s > 0):
                    with self.tik_inst.for_range(0, loop_times_s) as loop_index_s:
                        d_index_s = loop_index_s * each_loop_d_num
                        self._per_count_sum_d_each_loop(data_buf, mean_buf, n_index_, c_index_s,
                                                        d_index_s, each_loop_d_num)
                with self.tik_inst.if_scope(last_loop_d_num_s > 0):
                    self._per_count_sum_d_each_loop(data_buf, mean_buf, n_index_, c_index_s,
                                                    last_loop_index_s, last_loop_d_num_s)

    def _all_stand_data_d(self, mean_buf, n_index_):
        each_loop_d_num = self._get_loop_num_stand_data_d()
        loop_times, last_loop_d_num = self.get_loop_info(
            self.d_num, each_loop_d_num)
        ne_mean_s, rec_std_s = self._get_ne_mean_rec_std_scalar(mean_buf)
        with self.tik_inst.for_range(0, self.c_num) as c_index_s:
            data_buf = self._init_data_tensor_d(each_loop_d_num, False)
            with self.tik_inst.for_range(0, loop_times) as loop_index_s:
                d_index_s = loop_index_s * each_loop_d_num
                with self.tik_inst.if_scope(loop_index_s != loop_times - 1):
                    self._per_stand_data_d_each_loop(data_buf, ne_mean_s, rec_std_s,
                                                     n_index_, c_index_s, d_index_s, each_loop_d_num)
                with self.tik_inst.else_scope():
                    self._per_stand_data_d_each_loop(data_buf, ne_mean_s, rec_std_s,
                                                     n_index_, c_index_s, d_index_s, last_loop_d_num)

    def _init_seq_len_s(self, n_index_, c_num=1):
        seq_len_i_s = self.tik_inst.Scalar(self.int_type)
        seq_len_f_s = self.tik_inst.Scalar(self.data_type)
        unbiased_f_s = self.tik_inst.Scalar(self.data_type)
        with self.tik_inst.new_stmt_scope():
            seq_len_f_ub = self.tik_inst.Tensor(self.data_type, (self.data_block_data_num, ),
                                                self.tik.scope_ubuf, "seq_len_f_ub")
            with self.tik_inst.new_stmt_scope():
                seq_len_i_ub = self.tik_inst.Tensor(self.int_type, (self.int_block_data_num,),
                                                    self.tik.scope_ubuf, "seq_len_i_ub")
                self.tik_inst.data_move(
                    seq_len_i_ub, self.seq_len[n_index_], 0, 1, 1, 0, 0)
                seq_len_i_s.set_as(seq_len_i_ub[0])
                mask, repeat = 1, 1
                self.tik_inst.vconv(mask, "", seq_len_f_ub,
                                    seq_len_i_ub, repeat, 1, 1, 8, 8)
            unbiased_f_ub = self.tik_inst.Tensor(self.data_type, (self.data_block_data_num,),
                                                 self.tik.scope_ubuf, "unbiased_f_ub")
            if c_num > 1:
                self.tik_inst.vmuls(mask, seq_len_f_ub,
                                    seq_len_f_ub, c_num, repeat, 1, 1, 8, 8)
            seq_len_f_s.set_as(seq_len_f_ub[0])
            self.tik_inst.vadds(mask, unbiased_f_ub,
                                seq_len_f_ub, -1, repeat, 1, 1, 8, 8)
            self.tik_inst.vdiv(mask, seq_len_f_ub, seq_len_f_ub,
                               unbiased_f_ub, repeat, 1, 1, 1, 8, 8, 8)
            unbiased_f_s.set_as(seq_len_f_ub[0])
        return seq_len_i_s, seq_len_f_s, unbiased_f_s

    def _init_mean(self, batch_num=1):
        mean_shape = (batch_num * self.data_block_data_num, )
        proc_num = batch_num * self.data_block_data_num
        batch_mean_ub = self.tik_inst.Tensor(self.data_type, mean_shape,
                                             self.tik.scope_ubuf, "batch_mean_ub")
        batch_mean_square_ub = self.tik_inst.Tensor(self.data_type, mean_shape,
                                                    self.tik.scope_ubuf, "batch_mean_square_ub")
        batch_variance_ub = self.tik_inst.Tensor(self.data_type, mean_shape,
                                                 self.tik.scope_ubuf, "batch_variance_ub")
        mean_buf = {
            "batch_mean_ub": TensorOperatorParam(batch_mean_ub, proc_num, 0),
            "batch_mean_square_ub": TensorOperatorParam(batch_mean_square_ub, proc_num, 0),
            "batch_variance_ub": TensorOperatorParam(batch_variance_ub, proc_num, 0)}
        return mean_buf

    def _vector_mean_tensor(self, mean_buf):
        cmd_dup_mean_tensor = [
            VecCmd(cmd_name="vector_dup", dst_name="batch_mean_ub", scalar=0),
            VecCmd(cmd_name="vector_dup",
                   dst_name="batch_mean_square_ub", scalar=0),
            VecCmd(cmd_name="vector_dup", dst_name="batch_variance_ub", scalar=0), ]
        VecExecutor.exec_vec_cmd(
            mean_buf, cmd_dup_mean_tensor, "batch_mean_ub")

    def _data_move_in(self, data_buf, n_index_, c_index_, d_index_, n_index_ub_, data_num_):
        input_data_ub = data_buf.get("input_data_ub").const_tensor
        block_num_ = self.ceil_div(data_num_, self.data_block_data_num)
        self.tik_inst.data_move(input_data_ub[n_index_ub_, 0], self.input_x[n_index_, c_index_, d_index_],
                                0, 1, block_num_, 0, 0)

    def _vector_reduce_add(self, result_ub, data_ub, work_ub, seq_len_i_s, n_index_ub_):
        mask = self.data_repeat_data_num
        repeat_num_s, last_index_s, last_num_s = self._init_loop_info(
            seq_len_i_s, mask)
        with self.tik_inst.if_scope(repeat_num_s > 0):
            self.tik_inst.vec_reduce_add(
                mask, result_ub, data_ub[n_index_ub_, 0], work_ub, repeat_num_s, 8)
            with self.tik_inst.if_scope(last_num_s > 0):
                self.tik_inst.vcadd(
                    last_num_s, work_ub, data_ub[n_index_ub_, last_index_s], 1, 1, 1, 8)
                self.tik_inst.vadd(1, result_ub, result_ub,
                                   work_ub, 1, 1, 1, 1, 8, 8, 8)
        with self.tik_inst.else_scope():
            with self.tik_inst.if_scope(last_num_s > 0):
                self.tik_inst.vcadd(
                    last_num_s, result_ub, data_ub[n_index_ub_, last_index_s], 1, 1, 1, 8)

    def _count_rec_std_ne_mean(self, mean_buf, seq_len_f_, unbiased_f_):
        cmd_count_rec_std_ne_mean = [
            VecCmd(cmd_name="vector_dup", dst_name="batch_variance_ub",
                   scalar=seq_len_f_),  # seq_len
            VecCmd(cmd_name="vdiv", dst_name="batch_mean_ub",
                   src0_name="batch_mean_ub", src1_name="batch_variance_ub"),  # E(x)
            VecCmd(cmd_name="vdiv", dst_name="batch_mean_square_ub",
                   src0_name="batch_mean_square_ub", src1_name="batch_variance_ub"),  # E(x^2)
            VecCmd(cmd_name="vmul", dst_name="batch_variance_ub",
                   src0_name="batch_mean_ub", src1_name="batch_mean_ub"),  # E(x)^2
            VecCmd(cmd_name="vsub", dst_name="batch_variance_ub",
                   src0_name="batch_mean_square_ub", src1_name="batch_variance_ub"),  # E(x^2) - E(x)^2
            VecCmd(cmd_name="vmuls", dst_name="batch_variance_ub",
                   src0_name="batch_variance_ub", scalar=unbiased_f_),
            VecCmd(cmd_name="vabs", dst_name="batch_variance_ub",
                   src0_name="batch_variance_ub"),  # abs(var)
            VecCmd(cmd_name="vmuls", dst_name="batch_mean_ub",
                   src0_name="batch_mean_ub", scalar=-1),  # -E(x)
            VecCmd(cmd_name="vsqrt", dst_name="batch_variance_ub",
                   src0_name="batch_variance_ub"),  # std(x)
            VecCmd(cmd_name="vadds", dst_name="batch_variance_ub",
                   src0_name="batch_variance_ub", scalar=self.epsilon),  # std(x) + eps
            VecCmd(cmd_name="vector_dup",
                   dst_name="batch_mean_square_ub", scalar=1),  # 1
            VecCmd(cmd_name="vdiv", dst_name="batch_variance_ub",
                   src0_name="batch_mean_square_ub", src1_name="batch_variance_ub"), ]  # 1/std(x)
        VecExecutor.exec_vec_cmd(
            mean_buf, cmd_count_rec_std_ne_mean, "batch_mean_ub")

    def _stand_data(self, data_buf, ne_mean_, rec_std_):
        count_stand_cmd = [
            VecCmd(cmd_name="vadds", dst_name="input_data_ub",
                   src0_name="input_data_ub", scalar=ne_mean_),
            VecCmd(cmd_name="vmuls", dst_name="input_data_ub",
                   src0_name="input_data_ub", scalar=rec_std_)
        ]
        VecExecutor.exec_vec_cmd(data_buf, count_stand_cmd, "input_data_ub")

    def _get_ne_mean_rec_std_scalar(self, mean_buf):
        batch_ne_mean_ub = mean_buf.get("batch_mean_ub").const_tensor
        batch_rec_std_ub = mean_buf.get("batch_variance_ub").const_tensor
        ne_mean_s = self.tik_inst.Scalar(self.data_type)
        rec_std_s = self.tik_inst.Scalar(self.data_type)
        ne_mean_s.set_as(batch_ne_mean_ub[0])
        rec_std_s.set_as(batch_rec_std_ub[0])
        return ne_mean_s, rec_std_s

    def _data_move_out(self, data_buf, n_index_, c_index_, d_index_, n_index_ub_, data_num):
        input_data_ub = data_buf.get("input_data_ub").const_tensor
        if data_num % self.data_block_data_num != 0:
            block_num = self.ceil_div(data_num, self.data_block_data_num)
            mask_h, mask_l, mask_index = self.get_mask(
                data_num, self.data_repeat_data_num, self.data_block_data_num)
            self.tik_inst.vector_dup(
                [mask_h, mask_l], input_data_ub[n_index_ub_, mask_index], 0.0, 1, 1, 8)

            add_mode = 1
            self.tik_inst.set_atomic_add(add_mode)
            self.tik_inst.data_move(self.output_y[n_index_, c_index_, d_index_], input_data_ub[n_index_ub_, 0],
                                    0, 1, block_num, 0, 0)
            self.tik_inst.set_atomic_add(0)
        else:
            block_num = data_num // self.data_block_data_num
            self.tik_inst.data_move(self.output_y[n_index_, c_index_, d_index_], input_data_ub[n_index_ub_, 0],
                                    0, 1, block_num, 0, 0)

    def _init_loop_info(self, data_num_s, each_loop_num_):
        loop_times_s = self.tik_inst.Scalar(data_num_s.dtype)
        last_loop_index_s = self.tik_inst.Scalar(data_num_s.dtype)
        last_loop_d_num_s = self.tik_inst.Scalar(data_num_s.dtype)
        loop_times_s.set_as(data_num_s // each_loop_num_)
        last_loop_index_s.set_as(loop_times_s * each_loop_num_)
        last_loop_d_num_s.set_as(data_num_s - last_loop_index_s)
        return loop_times_s, last_loop_index_s, last_loop_d_num_s


def check_params(input_x, seq_len, output_y, normalize_type, epsilon, kernel_name):
    """
    check params
    """
    input_shape = input_x.get("shape")
    input_dtype = input_x.get("dtype").lower()
    if len(input_shape) != 3:
        error_manager_vector.raise_err_input_value_invalid(
            kernel_name, "len of input_x shape", "equal 3", str(len(input_shape)))
    if input_dtype != "float32":
        error_manager_vector.raise_err_input_value_invalid(
            kernel_name, "dtype of input_x", "float32", input_dtype)
    n_num, c_num, d_num = input_shape
    if d_num <= 1:
        error_manager_vector.raise_err_input_value_invalid(
            kernel_name, "d_num of input_x shape", "larger than 1", str(d_num))
    shape_seq_len = (n_num, )
    int_type = "int32"
    param_list = (seq_len, output_y)
    dtype_list = (int_type, input_dtype)
    shape_list = (shape_seq_len, input_shape)
    param_name_list = ("seq_len", "output_y")
    for param, dtype, shape, param_name in zip(param_list, dtype_list, shape_list, param_name_list):
        param_shape = param.get("shape")
        param_dtype = param.get("dtype").lower()
        if param_shape != shape:
            error_manager_vector.raise_err_input_value_invalid(
                kernel_name, "shape of {}".format(param_name), str(shape), str(param_shape))
        if param_dtype != dtype:
            error_manager_vector.raise_err_input_value_invalid(
                kernel_name, "dtype of {}".format(param_name), dtype, param_dtype)
    if normalize_type not in ("per_feature", "all_features"):
        error_manager_vector.raise_err_input_value_invalid(
            kernel_name, "normalize_type", "per_feature or all_features", normalize_type)
    if not isinstance(epsilon, float) or epsilon <= 0:
        error_manager_vector.raise_err_input_value_invalid(
            kernel_name, "epsilon", "must be float and greater than 0", "not float or less than 0")


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_STR, para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
# 'pylint: disable=unused-argument,too-many-arguments
def normalize_batch(input_x, seq_len, output_y, normalize_type, epsilon=0.00001, kernel_name="NormalizeBatch"):
    """
    algorithm: Count normalize of input_x based on seq_len
    Parameters
    ----------
    input_x:
        A Tensor. Support float32. shape (n, c, d)
    seq_len :
        A Tensor. Each batch normalize data num. Support Int32. Shape (n, ).
    output_y:
        A Tensor. Support float32. shape (n, c, d)
    normalize_type :
        Str. Support "per_feature" or "all_features".
    epsilon :
        Float. The epsilon value to use avoid division by zero.
    kernel_name : str
        cce kernel name, default value is NormalizeBatch
    Returns
    -------
    None
    """
    check_params(input_x, seq_len, output_y,
                 normalize_type, epsilon, kernel_name)
    AContainer.reset_instance()
    cont = AContainer.get_instance()
    n_num, c_num, d_num = input_x.get("shape")
    data_type = input_x.get("dtype")
    int_type = seq_len.get("dtype")
    obj = NormalizeBatch(n_num, c_num, d_num, normalize_type,
                         epsilon, data_type, int_type, kernel_name, cont)
    obj.mode_compute()
