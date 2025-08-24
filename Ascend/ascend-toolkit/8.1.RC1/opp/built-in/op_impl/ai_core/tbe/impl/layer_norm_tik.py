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
layer_norm_tik.py
"""
from impl.util.platform_adapter import tbe_platform
from impl.ascend import AContainer
from impl.ascend import TensorOperatorParam
from impl.ascend import VecCmd
from impl.ascend import VecExecutor
from impl.util.util_tik_comm_func import ceil_div


class LayerNormalizeBase:
    """
    layer normalize
    """

    # 'pylint: disable=too-many-arguments
    def __init__(self, batch_num, data_num, epsilon, kernel_name, cont, data_type, elementwise, output_mean_var):
        """
        init LayerNormalize attrs, init gm tensor
        Args:
            batch_num: int
            data_num: int
            epsilon: float; Minimum positive number greater than 0
            kernel_name: str
            cont: AContainer
            data_type: str; support float16, float32
            elementwise: if count x * gamma + beta
            output_mean_var: if output mean variance
        """
        self.batch_num = batch_num
        self.data_num = data_num
        self.epsilon = epsilon
        self.elementwise = elementwise
        self.output_mean_var = output_mean_var
        self.kernel_name = kernel_name

        self.cont = cont
        self.tik = self.cont.tik
        self.tik_inst = self.cont.tinst
        self.ai_core_use = self.cont.const_aicore_num
        self.repeat_time_max = self.cont.const_vector_proc_max_rpt
        self.ub_size = self.cont.const_ub_max_byte
        self.block_size = self.cont.const_block_byte

        self.gm_type = data_type
        self.gm_data_size, self.gm_block_data_num, self.gm_repeat_data_num = self._get_type_const(self.gm_type)
        self.ub_type = "float32"
        self.ub_data_size, self.ub_block_data_num, self.ub_repeat_data_num = self._get_type_const(self.ub_type)

        input_shape = (self.batch_num, self.data_num)
        self.input_x = self.tik_inst.Tensor(self.gm_type, input_shape, self.tik.scope_gm, "x")
        if self.elementwise:
            data_shape = (1, self.data_num)
            self.gamma = self.tik_inst.Tensor(self.gm_type, data_shape, self.tik.scope_gm, "gamma")
            self.beta = self.tik_inst.Tensor(self.gm_type, data_shape, self.tik.scope_gm, "beta")
        self.input_y = self.tik_inst.Tensor(self.gm_type, input_shape, self.tik.scope_gm, "y")
        if self.output_mean_var:
            batch_shape = (self.batch_num,)
            is_atomic_add = (self.ub_type == self.gm_type)
            self.mean = self.tik_inst.Tensor(self.gm_type, batch_shape, self.tik.scope_gm,
                                             "mean", is_atomic_add=is_atomic_add)
            self.variance = self.tik_inst.Tensor(self.gm_type, batch_shape, self.tik.scope_gm,
                                                 "variance", is_atomic_add=is_atomic_add)
            if self.gm_type != self.ub_type:
                batch_shape_align = (self.batch_num + self.gm_block_data_num,)
                self.mean_ub_type = self.tik_inst.Tensor(self.ub_type, batch_shape_align, self.tik.scope_gm,
                                                         "mean_ub_type", is_workspace=True, is_atomic_add=True)
                self.variance_ub_type = self.tik_inst.Tensor(self.ub_type, batch_shape_align, self.tik.scope_gm,
                                                             "variance_ub_type", is_workspace=True, is_atomic_add=True)

    @staticmethod
    def _get_loop_info(all_data_num, each_loop_num):
        loop_times = ceil_div(all_data_num, each_loop_num)
        last_loop_num = all_data_num - each_loop_num * (loop_times - 1)
        return loop_times, last_loop_num

    @staticmethod
    def _get_align_num(input_num, align_num, ceil=True):
        if ceil:
            result = ceil_div(input_num, align_num) * align_num
        else:
            result = input_num // align_num * align_num
        return result

    def _get_type_const(self, data_type):
        data_size = self.cont.const_dtype_byte.get(data_type)
        block_data_num = self.cont.get_vec_proc_num_per_cmd_blk(data_type)
        repeat_data_num = self.cont.get_vec_proc_num_per_cmd(data_type)
        return data_size, block_data_num, repeat_data_num

    def _get_mask(self, data_num, format_num_floor, format_num_ceil):
        """
        Args:
            data_num: last data num
            format_num_floor: data_num align floor
            format_num_ceil: data_num align ceil
        Returns: mask_h, mask_l, start_index
        """
        mask_split = 64
        data_num_floor = self._get_align_num(data_num, format_num_floor, False)
        data_num_ceil = self._get_align_num(data_num, format_num_ceil, True)
        mask_h, mask_l = 0, 0
        index_start = data_num - data_num_floor
        index_end = data_num_ceil - data_num_floor
        for index_l in range(index_start, min(index_end, mask_split)):
            mask_l += 2 ** index_l
        for index_h in range(max(mask_split, index_start), index_end):
            mask_h += 2 ** (index_h - mask_split)
        return mask_h, mask_l, data_num_floor

    @staticmethod
    def split_shape(input_tensor, dim_split):
        """
        split shape
        :return: number, batch_num, data_num, dim_split
        """
        shape_split = input_tensor.get("shape")
        if dim_split < 0:
            dim_split = dim_split + len(shape_split)

        batch_num = 1
        data_num = 1
        for i in range(dim_split):
            batch_num *= shape_split[i]
        for i in range(dim_split, len(shape_split)):
            data_num *= shape_split[i]
        return batch_num, data_num, dim_split

    @staticmethod
    def _get_each_data_size(gm_type, cont):
        """
        get each data size
        :return: number, each data size
        """
        ub_type = "float32"
        ub_data_size = float(cont.const_dtype_byte.get(ub_type))
        ub_repeat_data_num = cont.get_vec_proc_num_per_cmd(ub_type)
        gm_data_size = float(cont.const_dtype_byte.get(gm_type))
        # count tensor: input_data_ub, input_data_square_ub
        count_tensor_num = 2
        each_data_size = count_tensor_num * ub_data_size + ub_data_size / ub_repeat_data_num
        if ub_type != gm_type:
            each_data_size += gm_data_size
        return each_data_size

    @staticmethod
    def mode_split_n_max_num(ub_size, gm_type, cont):
        """
        split max number
        :return: number
        """
        block_size = cont.const_block_byte
        # expand_tensor: batch_mean_ub, batch_mean_square_ub, batch_variance_ub, work_tensor_ub
        expand_tensor_num = 4
        expand_size = expand_tensor_num * block_size
        ub_size_remain = ub_size - expand_size
        each_data_size = LayerNormalizeBase._get_each_data_size(gm_type, cont)
        data_num_max = int(ub_size_remain / each_data_size)

        ub_type = "float32"
        align_num = cont.get_vec_proc_num_per_cmd(ub_type)
        data_num_align = (data_num_max // align_num * align_num)
        data_num_max = cont.const_vector_proc_max_rpt * align_num
        return min(data_num_align, data_num_max)

    def _init_mean_tensor(self, batch_num):
        """
        init tensor
        """
        data_type = self.ub_type
        num_per_cmd = self.ub_repeat_data_num
        mean_shape = (batch_num,)

        batch_mean_ub = self.tik_inst.Tensor(data_type, mean_shape, self.tik.scope_ubuf, "batch_mean_ub")
        batch_mean_square_ub = self.tik_inst.Tensor(data_type, mean_shape, self.tik.scope_ubuf, "batch_mean_square_ub")
        batch_variance_ub = self.tik_inst.Tensor(data_type, mean_shape, self.tik.scope_ubuf, "batch_variance_ub")

        buf_mean_all = {
            "batch_mean_ub": TensorOperatorParam(batch_mean_ub, batch_num, 0, num_per_cmd=num_per_cmd),
            "batch_mean_square_ub": TensorOperatorParam(batch_mean_square_ub, batch_num, 0, num_per_cmd=num_per_cmd),
            "batch_variance_ub": TensorOperatorParam(batch_variance_ub, batch_num, 0, num_per_cmd=num_per_cmd)}
        cmd_dup_mean_tensor = [
            VecCmd(cmd_name="vector_dup", dst_name="batch_mean_ub", scalar=0),
            VecCmd(cmd_name="vector_dup", dst_name="batch_mean_square_ub", scalar=0),
            VecCmd(cmd_name="vector_dup", dst_name="batch_variance_ub", scalar=0), ]
        VecExecutor.exec_vec_cmd(buf_mean_all, cmd_dup_mean_tensor, "batch_mean_ub")
        return buf_mean_all

    def _mean_var_move_out(self, batch_mean_ub, batch_variance_ub, batch_index_start, batch_num):
        if self.gm_type != self.ub_type:
            mean_gm = self.mean_ub_type
            variance_gm = self.variance_ub_type
        else:
            mean_gm = self.mean
            variance_gm = self.variance
        if self.output_mean_var:
            self.tik_inst.set_atomic_add(1)
            block_num = ceil_div(batch_num, self.ub_block_data_num)
            self._data_move(mean_gm[batch_index_start], batch_mean_ub, 0, 1, block_num, 0, 0)
            self._data_move(variance_gm[batch_index_start], batch_variance_ub, 0, 1, block_num, 0, 0)
            self.tik_inst.set_atomic_add(0)

    def _mean_var_move_out_fp16(self, batch_index_start, batch_num):
        # init tensor num is 4
        each_loop_batch_num_max = self.ub_size // (self.ub_data_size * 2 + 2 * self.gm_data_size)
        each_loop_batch_num_max = self._get_align_num(each_loop_batch_num_max, self.gm_block_data_num, ceil=False)
        each_loop_batch_num = min(each_loop_batch_num_max, batch_num)
        loop_times, last_loop_batch_num = self._get_loop_info(batch_num, each_loop_batch_num)
        if loop_times == 1:
            self._mean_var_move_out_fp16_loop(batch_index_start, batch_num)
        else:
            with self.tik_inst.for_range(0, loop_times) as loop_index:
                batch_index = batch_index_start + each_loop_batch_num * loop_index
                with self.tik_inst.if_scope(loop_index != loop_times - 1):
                    self._mean_var_move_out_fp16_loop(batch_index, each_loop_batch_num)
                with self.tik_inst.else_scope():
                    self._mean_var_move_out_fp16_loop(batch_index, last_loop_batch_num)

    def _mean_var_move_out_fp16_loop(self, batch_index_start, batch_num):
        batch_num_align = self._get_align_num(batch_num, self.gm_block_data_num)
        block_num_move_in = batch_num_align // self.ub_block_data_num
        block_num_move_out = batch_num_align // self.gm_block_data_num
        ub_shape = (batch_num_align,)
        mean_ub_type_ub = self.tik_inst.Tensor(self.ub_type, ub_shape, self.tik.scope_ubuf, "mean_ub_type_ub")
        mean_gm_type_ub = self.tik_inst.Tensor(self.gm_type, ub_shape, self.tik.scope_ubuf, "mean_gm_type_ub")
        var_ub_type_ub = self.tik_inst.Tensor(self.ub_type, ub_shape, self.tik.scope_ubuf, "var_ub_type_ub")
        var_gm_type_ub = self.tik_inst.Tensor(self.gm_type, ub_shape, self.tik.scope_ubuf, "var_gm_type_ub")
        buf_mean_all = {
            "mean_ub_type_ub": TensorOperatorParam(mean_ub_type_ub, batch_num_align,
                                                   0, num_per_cmd=self.ub_repeat_data_num),
            "mean_gm_type_ub": TensorOperatorParam(mean_gm_type_ub, batch_num_align,
                                                   0, num_per_cmd=self.ub_repeat_data_num),
            "var_ub_type_ub": TensorOperatorParam(var_ub_type_ub, batch_num_align,
                                                  0, num_per_cmd=self.ub_repeat_data_num),
            "var_gm_type_ub": TensorOperatorParam(var_gm_type_ub, batch_num_align,
                                                  0, num_per_cmd=self.ub_repeat_data_num)}
        cmd_vconv = [VecCmd(cmd_name="vconv", dst_name="mean_gm_type_ub",
                            src0_name="mean_ub_type_ub", round_mode=""),
                     VecCmd(cmd_name="vconv", dst_name="var_gm_type_ub",
                            src0_name="var_ub_type_ub", round_mode="")
                     ]
        self._data_move(mean_ub_type_ub, self.mean_ub_type[batch_index_start],
                        0, 1, block_num_move_in, 0, 0)
        self._data_move(var_ub_type_ub, self.variance_ub_type[batch_index_start],
                        0, 1, block_num_move_in, 0, 0)
        VecExecutor.exec_vec_cmd(buf_mean_all, cmd_vconv, "mean_ub_type_ub")
        self._data_move(self.mean[batch_index_start], mean_gm_type_ub,
                        0, 1, block_num_move_out, 0, 0)
        self._data_move(self.variance[batch_index_start], var_gm_type_ub,
                        0, 1, block_num_move_out, 0, 0)

    def _count_rec_std_ne_mean(self, buf_mean_all):
        cmd_count_rec_std_ne_mean = [
            VecCmd(cmd_name="vmuls", dst_name="batch_mean_ub",
                   src0_name="batch_mean_ub", scalar=-1),  # -E(x)
            VecCmd(cmd_name="vadds", dst_name="batch_variance_ub",
                   src0_name="batch_variance_ub", scalar=self.epsilon),  # var(x) + eps
            VecCmd(cmd_name="vsqrt", dst_name="batch_variance_ub", src0_name="batch_variance_ub"),  # std(x)
            VecCmd(cmd_name="vector_dup", dst_name="batch_mean_square_ub", scalar=1),  # 1
            VecCmd(cmd_name="vdiv", dst_name="batch_variance_ub",
                   src0_name="batch_mean_square_ub", src1_name="batch_variance_ub"), ]  # 1/std(x)
        VecExecutor.exec_vec_cmd(buf_mean_all, cmd_count_rec_std_ne_mean, "batch_mean_ub")

    # 'pylint: disable=too-many-arguments
    def _data_move(self, dst, src, sid, nburst, burst, src_stride, dst_stride):
        """
        move data
        """
        if nburst > 0 and burst > 0:
            self.tik_inst.data_move(dst, src, sid, nburst, burst, src_stride, dst_stride)

    def _mode_compute(self, mode_compute_each_core):
        each_core_batch_num = ceil_div(self.batch_num, self.ai_core_use)
        self.ai_core_use, last_core_batch_num = self._get_loop_info(self.batch_num, each_core_batch_num)
        sync_workspace = None
        if self.gm_type != self.ub_type and self.ai_core_use > 1:
            sync_type = "int64"
            sync_block_data_num = self.cont.get_vec_proc_num_per_cmd_blk(sync_type)
            sync_data_num = self.ai_core_use * sync_block_data_num
            sync_workspace = self.tik_inst.Tensor("int64", (sync_data_num,), name="gm_barrier",
                                                  scope=self.tik.scope_gm, is_workspace=True, is_atomic_add=True)

        with self.tik_inst.for_range(0, self.ai_core_use, block_num=self.ai_core_use) as core_index:
            batch_index = each_core_batch_num * core_index
            with self.tik_inst.if_scope(core_index != self.ai_core_use - 1):
                mode_compute_each_core(batch_index, each_core_batch_num)
            with self.tik_inst.else_scope():
                mode_compute_each_core(batch_index, last_core_batch_num)
            if self.gm_type != self.ub_type:
                if self.ai_core_use > 1:
                    self.tik_inst.block_barrier(sync_workspace)
                with self.tik_inst.if_scope(core_index != self.ai_core_use - 1):
                    self._mean_var_move_out_fp16(batch_index, each_core_batch_num)
                with self.tik_inst.else_scope():
                    self._mean_var_move_out_fp16(batch_index, last_core_batch_num)
        if self.elementwise:
            inputs_all = [self.input_x, self.gamma, self.beta]
        else:
            inputs_all = [self.input_x]
        if self.output_mean_var:
            outputs_all = [self.input_y, self.mean, self.variance]
        else:
            outputs_all = [self.input_y]
        self.tik_inst.BuildCCE(
            inputs=inputs_all,
            outputs=outputs_all,
            kernel_name=self.kernel_name)


class LayerNormalizeSplitD(LayerNormalizeBase):
    """
    layer normalize:
        split mode: split d dim, each batch count
    """

    def mode_compute(self):
        """
        start mode compute, d larger
        """
        self._mode_compute(self._mode_compute_each_core)

    def _mode_compute_each_core(self, batch_index_start, batch_num):
        if batch_num == 1:
            thread_num = 1
        else:
            thread_num = 2
        with self.tik_inst.for_range(0, batch_num, thread_num=thread_num) as batch_index_temp:
            batch_index = batch_index_start + batch_index_temp
            self._mode_compute_each_loop(batch_index, thread_num)

    def _mode_compute_each_loop(self, batch_index, thread_num):
        ub_size = self.cont.const_ub_max_byte // thread_num
        buf_mean_all = self._init_mean_tensor(self.ub_block_data_num)
        batch_mean_ub = buf_mean_all.get("batch_mean_ub").const_tensor
        batch_mean_square_ub = buf_mean_all.get("batch_mean_square_ub").const_tensor
        batch_variance_ub = buf_mean_all.get("batch_variance_ub").const_tensor

        mean_ub_all = (batch_mean_ub, batch_mean_square_ub, batch_variance_ub)
        # count sum
        self._count_mean_each_batch(ub_size, batch_index, mean_ub_all)
        # count mean, var
        self._count_mean_var(buf_mean_all)
        # move out mean var
        self._mean_var_move_out(batch_mean_ub, batch_variance_ub, batch_index, 1)
        # count rec std, ne mean
        self._count_rec_std_ne_mean(buf_mean_all)
        self._count_stand_each_batch(ub_size, batch_index, mean_ub_all)

    def _count_mean_each_batch(self, ub_size, batch_index, mean_ub_all):
        each_loop_mean_num = self._get_mean_each_loop_num(ub_size)
        mean_loop_num, last_loop_mean_num = self._get_loop_info(self.data_num, each_loop_mean_num)
        with self.tik_inst.for_range(0, mean_loop_num) as loop_index:
            start_index = loop_index * each_loop_mean_num
            with self.tik_inst.if_scope(loop_index != mean_loop_num - 1):
                self._count_mean_each_batch_loop(batch_index, start_index, each_loop_mean_num, mean_ub_all)
            with self.tik_inst.else_scope():
                self._count_mean_each_batch_loop(batch_index, start_index, last_loop_mean_num, mean_ub_all)

    def _get_mean_each_loop_num(self, ub_size):
        gm_data_size = float(self.gm_data_size)
        ub_data_size = float(self.ub_data_size)
        tensor_num = 4
        ub_size_last = ub_size - tensor_num * self.block_size
        each_data_size = ub_data_size + ub_data_size / self.ub_repeat_data_num
        if self.gm_type != self.ub_type:
            each_data_size = each_data_size + gm_data_size
        data_num_max = int(ub_size_last / each_data_size)
        data_num_max_align = self._get_align_num(data_num_max, self.ub_repeat_data_num, False)
        return data_num_max_align

    # 'pylint: disable=too-many-locals
    def _count_mean_each_batch_loop(self, batch_index, start_index, data_num, mean_ub_all):
        mean_data_ub, mean_data_square_ub, mean_temp_ub = mean_ub_all
        data_num_align = self._get_align_num(data_num, self.ub_repeat_data_num)

        input_data_ub = self.tik_inst.Tensor(self.ub_type, (data_num_align,), self.tik.scope_ubuf, "input_data_ub")
        buf_sum_ub = {
            "input_data_ub": TensorOperatorParam(input_data_ub, data_num_align, 0, num_per_cmd=self.ub_repeat_data_num)
        }

        repeat_num = data_num_align // self.ub_repeat_data_num
        work_tensor_num = self._get_align_num(repeat_num, self.ub_block_data_num)
        work_tensor_ub = self.tik_inst.Tensor(self.ub_type, (work_tensor_num,), self.tik.scope_ubuf, "work_tensor_ub")

        gm_data_tensor_ub = None
        if self.gm_type != self.ub_type:
            gm_data_tensor_ub = self.tik_inst.Tensor(
                self.gm_type, (data_num_align,), self.tik.scope_ubuf, "gm_data_tensor_ub")

            buf_sum_ub["gm_data_tensor_ub"] = TensorOperatorParam(gm_data_tensor_ub, data_num_align, 0,
                                                                  num_per_cmd=self.gm_repeat_data_num)
            buf_sum_ub["gm_data_tensor_vconv_ub"] = TensorOperatorParam(gm_data_tensor_ub, data_num_align, 0,
                                                                        num_per_cmd=self.ub_repeat_data_num)

        self._data_move_in(input_data_ub, "input_data_ub", gm_data_tensor_ub, self.input_x,
                           buf_sum_ub, batch_index, start_index, data_num, 0)
        # count mean(x)
        self._count_mean_each_batch_loop_count(mean_temp_ub, input_data_ub, work_tensor_ub,
                                               repeat_num, mean_data_ub)
        # count mean(x^2)
        cmd_square = [
            VecCmd(cmd_name="vmul", dst_name="input_data_ub",
                   src0_name="input_data_ub", src1_name="input_data_ub")]
        VecExecutor.exec_vec_cmd(buf_sum_ub, cmd_square, "input_data_ub")
        self._count_mean_each_batch_loop_count(mean_temp_ub, input_data_ub, work_tensor_ub,
                                               repeat_num, mean_data_square_ub)

    # 'pylint: disable=too-many-arguments,too-many-locals
    def _data_move_in(self, ub_data_tensor_ub, ub_data_tensor_ub_name, gm_data_tensor_ub, tensor_gm,
                      buf_sum_ub, batch_index, start_index, data_num, mode):
        if self.gm_type != self.ub_type:
            tensor_ub, tensor_name = gm_data_tensor_ub, "gm_data_tensor_ub"
        else:
            tensor_ub, tensor_name = ub_data_tensor_ub, ub_data_tensor_ub_name
        cmd_dup_tensor = [VecCmd(cmd_name="vector_dup", dst_name=tensor_name, scalar=0)]
        VecExecutor.exec_vec_cmd(buf_sum_ub, cmd_dup_tensor, tensor_name)
        if mode == 0:
            block_num = ceil_div(data_num, self.gm_block_data_num)
            self._data_move(tensor_ub, tensor_gm[batch_index, start_index], 0, 1, block_num, 0, 0)
            if data_num % self.gm_block_data_num != 0:
                mask_h, mask_l, data_num_floor = self._get_mask(data_num,
                                                                self.gm_repeat_data_num,
                                                                self.gm_block_data_num)
                self.tik_inst.vector_dup([mask_h, mask_l], tensor_ub[data_num_floor], 0.0, 1, 1, 8)
        else:
            block_num = data_num // self.gm_block_data_num
            self._data_move(tensor_ub, tensor_gm[batch_index, start_index], 0, 1, block_num, 0, 0)
            if data_num % self.gm_block_data_num != 0:
                last_block_ub_index, last_block_gm_index = self._get_last_block_info(start_index, data_num)
                self._data_move(tensor_ub[last_block_ub_index],
                                tensor_gm[batch_index, last_block_gm_index],
                                0, 1, 1, 0, 0)

        if self.gm_type != self.ub_type:
            cmd_vconv = [VecCmd(cmd_name="vconv", dst_name=ub_data_tensor_ub_name,
                                src0_name="gm_data_tensor_vconv_ub", round_mode="")]
            VecExecutor.exec_vec_cmd(buf_sum_ub, cmd_vconv, "gm_data_tensor_vconv_ub")

    def _get_last_block_info(self, start_index, data_num):
        last_block_ub_index = self._get_align_num(data_num, self.gm_block_data_num, False)
        last_block_gm_index = start_index + data_num - self.gm_block_data_num
        return last_block_ub_index, last_block_gm_index

    # 'pylint: disable=too-many-arguments
    def _count_mean_each_batch_loop_count(self, mean_temp_ub, src_tensor_ub, work_tensor_ub, repeat_num, mean_data_ub):
        mask_sum = self.ub_repeat_data_num
        mask_mean = self.ub_block_data_num
        self.tik_inst.vec_reduce_add(mask_sum, mean_temp_ub, src_tensor_ub, work_tensor_ub, repeat_num, 8)
        self.tik_inst.vmuls(mask_mean, mean_temp_ub, mean_temp_ub, 1.0 / self.data_num, 1, 1, 1, 8, 8)
        self.tik_inst.vadd(mask_mean, mean_data_ub, mean_data_ub, mean_temp_ub, 1, 1, 1, 1, 8, 8, 8)

    @staticmethod
    def _count_mean_var(buf_mean_all):
        # count mean, mean_square, variance
        cmd_count_mean_var = [
            VecCmd(cmd_name="vmul", dst_name="batch_variance_ub",
                   src0_name="batch_mean_ub", src1_name="batch_mean_ub"),  # E(x)^2
            VecCmd(cmd_name="vsub", dst_name="batch_variance_ub",
                   src0_name="batch_mean_square_ub", src1_name="batch_variance_ub"),  # E(x)^2 - E(x)^2
            VecCmd(cmd_name="vabs", dst_name="batch_variance_ub", src0_name="batch_variance_ub")  # abs(var)
        ]
        VecExecutor.exec_vec_cmd(buf_mean_all, cmd_count_mean_var, "batch_mean_ub")

    def _count_stand_each_batch(self, ub_size, batch_index, mean_ub_all):
        ne_mean_ub, _, rec_std_ub = mean_ub_all
        ne_mean_scalar = self.tik_inst.Scalar(self.ub_type)
        rec_std_scalar = self.tik_inst.Scalar(self.ub_type)
        ne_mean_scalar.set_as(ne_mean_ub[0])
        rec_std_scalar.set_as(rec_std_ub[0])
        scalar_all = (ne_mean_scalar, rec_std_scalar)
        each_loop_stand_num = self._get_count_each_loop_num(ub_size)
        stand_loop_num, last_loop_stand_num, = self._get_loop_info(self.data_num, each_loop_stand_num)
        with self.tik_inst.for_range(0, stand_loop_num) as loop_index:
            start_index = loop_index * each_loop_stand_num
            with self.tik_inst.if_scope(loop_index != stand_loop_num - 1):
                self._count_stand_each_batch_loop(batch_index, start_index, each_loop_stand_num, scalar_all)
            with self.tik_inst.else_scope():
                self._count_stand_each_batch_loop(batch_index, start_index, last_loop_stand_num, scalar_all)

    def _get_count_each_loop_num(self, ub_size):
        gm_data_size = float(self.gm_data_size)
        ub_data_size = float(self.ub_data_size)
        tensor_num = 3
        ub_size_last = ub_size - tensor_num * self.cont.const_block_byte

        each_data_size = ub_data_size
        if self.elementwise:
            each_data_size += ub_data_size
        if self.gm_type != self.ub_type:
            each_data_size += gm_data_size
        data_num_max = int(ub_size_last / each_data_size)
        data_num_max_align = self._get_align_num(data_num_max, self.ub_repeat_data_num, False)
        return data_num_max_align

    def _count_stand_each_batch_loop(self, batch_index, start_index, data_num, scalar_all):
        ne_mean_scalar, rec_std_scalar = scalar_all

        data_num_align = self._get_align_num(data_num, self.ub_repeat_data_num)
        src_stand_ub = self.tik_inst.Tensor(self.ub_type, (data_num_align,), self.tik.scope_ubuf, "src_stand_ub")
        buf_stand_ub = {
            "src_stand_ub": TensorOperatorParam(src_stand_ub, data_num_align, 0, num_per_cmd=self.ub_repeat_data_num)}

        gamma_beta_tensor_ub = None
        gm_data_tensor_ub = None
        if self.elementwise:
            gamma_beta_tensor_ub = self.tik_inst.Tensor(self.ub_type, (data_num_align,),
                                                        self.tik.scope_ubuf, "gamma_beta_tensor_ub")
            buf_stand_ub["gamma_beta_tensor_ub"] = TensorOperatorParam(gamma_beta_tensor_ub, data_num_align, 0,
                                                                       num_per_cmd=self.ub_repeat_data_num)
        if self.gm_type != self.ub_type:
            gm_data_tensor_ub = self.tik_inst.Tensor(self.gm_type, (data_num_align,),
                                                     self.tik.scope_ubuf, "gm_data_tensor_ub")
            buf_stand_ub["gm_data_tensor_ub"] = TensorOperatorParam(gm_data_tensor_ub, data_num_align, 0,
                                                                    num_per_cmd=self.gm_repeat_data_num)
            buf_stand_ub["gm_data_tensor_vconv_ub"] = TensorOperatorParam(gm_data_tensor_ub, data_num_align, 0,
                                                                          num_per_cmd=self.ub_repeat_data_num)
        self._data_move_in(src_stand_ub, "src_stand_ub", gm_data_tensor_ub, self.input_x,
                           buf_stand_ub, batch_index, start_index, data_num, 1)
        cmd_stand = [
            VecCmd(cmd_name="vadds", dst_name="src_stand_ub", src0_name="src_stand_ub", scalar=ne_mean_scalar),
            VecCmd(cmd_name="vmuls", dst_name="src_stand_ub", src0_name="src_stand_ub", scalar=rec_std_scalar)]
        VecExecutor.exec_vec_cmd(buf_stand_ub, cmd_stand, "src_stand_ub")

        if self.elementwise:
            self._data_move_in(gamma_beta_tensor_ub, "gamma_beta_tensor_ub", gm_data_tensor_ub, self.gamma,
                               buf_stand_ub, 0, start_index, data_num, 1)
            cmd_mul_gamma = [
                VecCmd(cmd_name="vmul", dst_name="src_stand_ub",
                       src0_name="src_stand_ub", src1_name="gamma_beta_tensor_ub")]
            VecExecutor.exec_vec_cmd(buf_stand_ub, cmd_mul_gamma, "src_stand_ub")

            self._data_move_in(gamma_beta_tensor_ub, "gamma_beta_tensor_ub", gm_data_tensor_ub, self.beta,
                               buf_stand_ub, 0, start_index, data_num, 1)
            cmd_add_beta = [
                VecCmd(cmd_name="vadd", dst_name="src_stand_ub",
                       src0_name="src_stand_ub", src1_name="gamma_beta_tensor_ub")]
            VecExecutor.exec_vec_cmd(buf_stand_ub, cmd_add_beta, "src_stand_ub")
        self._mode_data_move_out(
            src_stand_ub, gm_data_tensor_ub, buf_stand_ub, batch_index,
            start_index, data_num)

    # 'pylint: disable=too-many-arguments
    def _mode_data_move_out(self, data_tensor_ub, gm_data_tensor_ub, buf_stand_ub,
                            batch_index, start_index, data_num):
        if self.gm_type != self.ub_type:
            cmd_ub_vconv_gm = [
                VecCmd(cmd_name="vconv", dst_name="gm_data_tensor_vconv_ub",
                       src0_name="src_stand_ub", round_mode="")]
            VecExecutor.exec_vec_cmd(buf_stand_ub, cmd_ub_vconv_gm, "src_stand_ub")
            tensor_ub = gm_data_tensor_ub
        else:
            tensor_ub = data_tensor_ub
        block_num = data_num // self.gm_block_data_num
        self._data_move(self.input_y[batch_index, start_index], tensor_ub, 0, 1, block_num, 0, 0)
        if data_num % self.gm_block_data_num != 0:
            last_block_ub_index, last_block_gm_index = self._get_last_block_info(start_index, data_num)
            self._data_move(self.input_y[batch_index, last_block_gm_index], tensor_ub[last_block_ub_index], 0, 1, 1, 0,
                            0)


class LayerNormalizeSplitN(LayerNormalizeBase):
    """
    layer normalize:
        split mode:split n dim, each loop count multi batch
    """

    def mode_compute(self):
        """
        compute
        """
        self._mode_compute(self._mode1_compute_each_core)

    def _mode1_compute_each_core(self, batch_index_start, batch_num):
        if batch_num == 1:
            thread_num = 1
        else:
            thread_num = 2
        each_loop_batch_num = self._get_each_loop_batch_num(batch_num, thread_num)
        loop_times, last_loop_batch_num = self._get_loop_info(batch_num, each_loop_batch_num)
        with self.tik_inst.for_range(0, loop_times, thread_num=thread_num) as loop_index:
            batch_index = batch_index_start + each_loop_batch_num * loop_index
            with self.tik_inst.if_scope(loop_index != loop_times - 1):
                self._mode1_compute_each_loop(batch_index, each_loop_batch_num)
            with self.tik_inst.else_scope():
                self._mode1_compute_each_loop(batch_index, last_loop_batch_num)

    def _get_each_loop_batch_num(self, batch_num, thread_num):
        ub_size = self.ub_size // thread_num
        each_batch_data_num_align = self._get_align_num(self.data_num, self.ub_repeat_data_num)

        ub_data_size = float(self.ub_data_size)
        expand_tensor_num = 4
        expand_size = expand_tensor_num * (self.block_size - ub_data_size)
        ub_size_remain = ub_size - expand_size
        each_data_size = self._get_each_data_size(self.gm_type, self.cont)
        each_batch_data_size = each_data_size * each_batch_data_num_align + ub_data_size * expand_tensor_num
        max_loop_batch_num = int(ub_size_remain / each_batch_data_size)
        each_thread_batch_num = ceil_div(batch_num, thread_num)
        each_loop_batch_num = min(max_loop_batch_num, each_thread_batch_num)
        return each_loop_batch_num

    # 'pylint: disable=too-many-locals
    def _mode1_compute_each_loop(self, batch_index_start, batch_num):
        # fp32 data info
        batch_num_align = self._get_align_num(batch_num, self.ub_block_data_num)
        data_num_align = self._get_align_num(self.data_num, self.ub_repeat_data_num)
        data_repeat_num = data_num_align // self.ub_repeat_data_num

        # init vec_reduce_add work tensor
        work_tensor_num = self._get_align_num(data_repeat_num, self.ub_block_data_num)
        work_tensor_ub = self.tik_inst.Tensor(self.ub_type, (work_tensor_num,), self.tik.scope_ubuf, "work_tensor_ub")

        # init mean var ub
        buf_mean_all = self._init_mean_tensor(batch_num_align)
        batch_mean_ub = buf_mean_all.get("batch_mean_ub").const_tensor
        batch_mean_square_ub = buf_mean_all.get("batch_mean_square_ub").const_tensor
        batch_variance_ub = buf_mean_all.get("batch_variance_ub").const_tensor

        # init data ub
        buf_data_all, gm_data_tensor_ub = self._init_data_tensor(batch_num, data_num_align)
        input_data_ub = buf_data_all.get("input_data_ub").const_tensor
        input_data_square_ub = buf_data_all.get("input_data_square_ub").const_tensor
        data_ub_all = (input_data_ub, input_data_square_ub, gm_data_tensor_ub)
        mean_ub_all = (batch_mean_ub, batch_mean_square_ub, batch_variance_ub)
        cmd_gm_vconv_ub = [
            VecCmd(cmd_name="vconv", dst_name="input_data_ub",
                   src0_name="gm_data_tensor_vconv_ub", round_mode=""),
            VecCmd(cmd_name="vconv", dst_name="input_data_square_ub",
                   src0_name="gm_data_tensor_vconv_ub", round_mode="")]
        # data move in
        self._data_move_in(data_ub_all, self.input_x, buf_data_all, cmd_gm_vconv_ub,
                           batch_index_start, batch_num, data_num_align, 0)
        # count sum
        self._mode1_count_sum(data_ub_all, mean_ub_all, work_tensor_ub, batch_num, data_num_align)
        # count mean, var
        self._count_mean_var(buf_mean_all)
        # move out mean var
        self._mean_var_move_out(batch_mean_ub, batch_variance_ub, batch_index_start, batch_num)
        # count rec std, ne mean
        self._count_rec_std_ne_mean(buf_mean_all)
        # adjust last 32b data
        self._mode1_adjust_input(data_ub_all, batch_index_start, batch_num)

        # count (data add ne_mean) mul rec_std
        self._stand_data(input_data_ub, batch_mean_ub, batch_variance_ub, batch_num, data_num_align)
        # data mul gamma add beta
        if self.elementwise:
            self._mode1_count_elementwise(data_ub_all, batch_num, data_num_align, buf_data_all, cmd_gm_vconv_ub)
        self._mode1_data_move_out(data_ub_all, buf_data_all, batch_index_start, batch_num, data_num_align)

    def _init_data_tensor(self, batch_num, data_num):
        # init fp32 x tensor, fp32 x^2 tensor
        input_data_ub = self.tik_inst.Tensor(self.ub_type, (batch_num, data_num),
                                             self.tik.scope_ubuf, "input_data_ub")
        input_data_square_ub = self.tik_inst.Tensor(self.ub_type, (batch_num, data_num),
                                                    self.tik.scope_ubuf, "input_data_square_ub")
        buf_data_all = {
            "input_data_ub": TensorOperatorParam(input_data_ub, batch_num * data_num, 0,
                                                 num_per_cmd=self.ub_repeat_data_num),
            "input_data_square_ub": TensorOperatorParam(input_data_square_ub, batch_num * data_num, 0,
                                                        num_per_cmd=self.ub_repeat_data_num)}
        # init fp16 tensor
        gm_data_tensor_ub = None
        if self.gm_type != self.ub_type:
            gm_data_tensor_ub = self.tik_inst.Tensor(self.gm_type, (batch_num, data_num),
                                                     self.tik.scope_ubuf, "gm_data_tensor_ub")
            buf_data_all["gm_data_tensor_ub"] = TensorOperatorParam(gm_data_tensor_ub, batch_num * data_num, 0,
                                                                    num_per_cmd=self.gm_repeat_data_num)
            buf_data_all["gm_data_tensor_vconv_ub"] = TensorOperatorParam(gm_data_tensor_ub, batch_num * data_num, 0,
                                                                          num_per_cmd=self.ub_repeat_data_num)
            cmd_dup_input_tensor = [VecCmd(cmd_name="vector_dup", dst_name="gm_data_tensor_ub", scalar=0)]
            VecExecutor.exec_vec_cmd(buf_data_all, cmd_dup_input_tensor, "gm_data_tensor_ub")
        else:
            cmd_dup_input_tensor = [VecCmd(cmd_name="vector_dup", dst_name="input_data_ub", scalar=0)]
            VecExecutor.exec_vec_cmd(buf_data_all, cmd_dup_input_tensor, "input_data_ub")
        return buf_data_all, gm_data_tensor_ub

    # 'pylint: disable=too-many-arguments
    def _data_move_in(self, data_ub_all, tensor_gm, buf_data_all, cmd_gm_vconv_ub,
                      batch_index_start, batch_num, data_num_align, mode):
        # get block format num
        if self.gm_type != self.ub_type:
            tensor_ub = data_ub_all[2]
        else:
            tensor_ub = data_ub_all[mode]
        # get each batch info
        each_batch_ub_block_num = data_num_align // self.gm_block_data_num
        each_batch_gm_block_num = self.data_num // self.gm_block_data_num
        # if data 32byte align
        if self.data_num == data_num_align:
            self._data_move(tensor_ub, tensor_gm[batch_index_start, 0], 0, 1, batch_num * each_batch_gm_block_num,
                            0, 0)
        elif self.data_num % self.gm_block_data_num == 0:
            self._data_move(tensor_ub, tensor_gm[batch_index_start, 0], 0, batch_num, each_batch_gm_block_num,
                            0, each_batch_ub_block_num - each_batch_gm_block_num)
        # if data not 32byte align
        else:
            self._data_move_in_not_align(tensor_ub, tensor_gm, batch_index_start, batch_num, mode)
        if self.gm_type != self.ub_type:
            VecExecutor.exec_vec_cmd(buf_data_all, [cmd_gm_vconv_ub[mode]], "gm_data_tensor_vconv_ub")

    # 'pylint: disable=too-many-arguments
    def _data_move_in_not_align(self, tensor_ub, tensor_gm, batch_index_start, batch_num, mode):
        if mode == 0:
            each_batch_block_num = ceil_div(self.data_num, self.gm_block_data_num)
            mask_h, mask_l, data_num_floor = self._get_mask(self.data_num, self.gm_repeat_data_num,
                                                            self.gm_block_data_num)
            with self.tik_inst.for_range(0, batch_num) as batch_index:
                self._data_move(tensor_ub[batch_index, 0],
                                tensor_gm[batch_index_start + batch_index, 0],
                                0, 1, each_batch_block_num, 0, 0)
                self.tik_inst.vector_dup([mask_h, mask_l], tensor_ub[batch_index, data_num_floor],
                                         0.0, 1, 1, 8)
        else:
            block_num_floor = self.data_num // self.gm_block_data_num
            last_block_ub_index, last_block_gm_index = self._get_last_block_info()
            with self.tik_inst.for_range(0, batch_num) as batch_index:
                self._data_move(tensor_ub[batch_index, 0],
                                tensor_gm[batch_index_start + batch_index, 0],
                                0, 1, block_num_floor, 0, 0)
                self._data_move(tensor_ub[0, last_block_ub_index],
                                tensor_gm[batch_index_start + batch_index, last_block_gm_index],
                                0, 1, 1, 0, 0)

    def _get_last_block_info(self):
        last_block_ub_index = self._get_align_num(self.data_num, self.gm_block_data_num, False)
        last_block_gm_index = self.data_num - self.gm_block_data_num
        return last_block_ub_index, last_block_gm_index

    # 'pylint: disable=too-many-arguments
    def _mode1_count_sum(self, data_ub_all, mean_ub_all, reduce_work_tensor, batch_num, data_num_align):
        input_data_ub, input_data_square_ub, _ = data_ub_all
        batch_mean_ub, batch_mean_square_ub, _ = mean_ub_all
        mask = self.ub_repeat_data_num
        each_batch_repeat_num = data_num_align // mask
        with self.tik_inst.for_range(0, batch_num) as batch_index:
            self.tik_inst.vmul(mask,
                               input_data_square_ub[batch_index, 0],
                               input_data_ub[batch_index, 0],
                               input_data_ub[batch_index, 0],
                               each_batch_repeat_num, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vec_reduce_add(mask,
                                         batch_mean_ub[batch_index],
                                         input_data_ub[batch_index, 0],
                                         reduce_work_tensor, each_batch_repeat_num, 8)
            self.tik_inst.vec_reduce_add(mask,
                                         batch_mean_square_ub[batch_index],
                                         input_data_square_ub[batch_index, 0],
                                         reduce_work_tensor, each_batch_repeat_num, 8)

    def _count_mean_var(self, buf_mean_all):
        # count mean, mean square, variance
        cmd_count_mean_var = [
            VecCmd(cmd_name="vmuls", dst_name="batch_mean_ub",
                   src0_name="batch_mean_ub", scalar=1.0 / self.data_num),  # E(x)
            VecCmd(cmd_name="vmuls", dst_name="batch_mean_square_ub",
                   src0_name="batch_mean_square_ub", scalar=1.0 / self.data_num),  # E(x^2)
            VecCmd(cmd_name="vmul", dst_name="batch_variance_ub",
                   src0_name="batch_mean_ub", src1_name="batch_mean_ub"),  # E(x)^2
            VecCmd(cmd_name="vsub", dst_name="batch_variance_ub",
                   src0_name="batch_mean_square_ub", src1_name="batch_variance_ub"),  # E(x^2) - E(x)^2
            VecCmd(cmd_name="vabs", dst_name="batch_variance_ub", src0_name="batch_variance_ub")  # abs(var)
        ]
        VecExecutor.exec_vec_cmd(buf_mean_all, cmd_count_mean_var, "batch_mean_ub")

    def _mode1_adjust_input(self, data_ub_all, batch_index_start, batch_num):
        if self.data_num % self.gm_block_data_num != 0:
            input_data_ub, _, gm_data_tensor_ub = data_ub_all
            last_block_ub_index, last_block_gm_index = self._get_last_block_info()
            if self.gm_type == self.ub_type:
                with self.tik_inst.for_range(0, batch_num) as batch_index:
                    self._data_move(input_data_ub[batch_index, last_block_ub_index],
                                    self.input_x[batch_index_start + batch_index, last_block_gm_index],
                                    0, 1, 1, 0, 0)
            else:
                with self.tik_inst.for_range(0, batch_num) as batch_index:
                    self._data_move(gm_data_tensor_ub,
                                    self.input_x[batch_index_start + batch_index, last_block_gm_index],
                                    0, 1, 1, 0, 0)
                    self.tik_inst.vconv(self.gm_block_data_num, "",
                                        input_data_ub[batch_index, last_block_ub_index],
                                        gm_data_tensor_ub, 1, 1, 1, 8, 8)

    # 'pylint: disable=too-many-arguments
    def _stand_data(self, input_data_ub, batch_ne_mean_ub, batch_rec_std_ub, batch_num, data_num_align):
        scalar_type = self.ub_type
        mask = self.ub_repeat_data_num
        repeat_num = data_num_align // mask
        ne_mean_scalar = self.tik_inst.Scalar(scalar_type)
        rec_std_scalar = self.tik_inst.Scalar(scalar_type)
        with self.tik_inst.for_range(0, batch_num) as batch_index:
            ne_mean_scalar.set_as(batch_ne_mean_ub[batch_index])
            rec_std_scalar.set_as(batch_rec_std_ub[batch_index])
            self.tik_inst.vadds(mask, input_data_ub[batch_index, 0], input_data_ub[batch_index, 0],
                                ne_mean_scalar, repeat_num, 1, 1, 8, 8)
            self.tik_inst.vmuls(mask, input_data_ub[batch_index, 0], input_data_ub[batch_index, 0],
                                rec_std_scalar, repeat_num, 1, 1, 8, 8)

    # 'pylint: disable=too-many-arguments
    def _mode1_count_elementwise(self, data_ub_all, batch_num, data_num_align, buf_data_all, cmd_gm_vconv_ub):
        input_data_ub, input_data_square_ub, _ = data_ub_all
        mask = self.ub_repeat_data_num
        each_batch_repeat_num = data_num_align // self.ub_repeat_data_num

        self._data_move_in(data_ub_all, self.gamma, buf_data_all, cmd_gm_vconv_ub, 0, 1, data_num_align, 1)
        with self.tik_inst.for_range(0, batch_num) as batch_index:
            self.tik_inst.vmul(mask, input_data_ub[batch_index, 0], input_data_ub[batch_index, 0],
                               input_data_square_ub, each_batch_repeat_num, 1, 1, 1, 8, 8, 8)

        self._data_move_in(data_ub_all, self.beta, buf_data_all, cmd_gm_vconv_ub, 0, 1, data_num_align, 1)
        with self.tik_inst.for_range(0, batch_num) as batch_index:
            self.tik_inst.vadd(mask, input_data_ub[batch_index, 0], input_data_ub[batch_index, 0],
                               input_data_square_ub, each_batch_repeat_num, 1, 1, 1, 8, 8, 8)

    # 'pylint: disable=too-many-arguments,too-many-locals
    def _mode1_data_move_out(self, data_ub_all, buf_data_all, batch_index_start, batch_num, data_num_align):
        input_data_ub, _, gm_data_tensor_ub = data_ub_all
        if self.gm_type == self.ub_type:
            tensor_ub = input_data_ub
        else:
            cmd_ub_vconv_gm = [VecCmd(cmd_name="vconv", dst_name="gm_data_tensor_vconv_ub",
                                      src0_name="input_data_ub", round_mode="")]
            VecExecutor.exec_vec_cmd(buf_data_all, cmd_ub_vconv_gm, "input_data_ub")
            tensor_ub = gm_data_tensor_ub

        each_batch_ub_block_num = data_num_align // self.gm_block_data_num
        each_batch_gm_block_num = self.data_num // self.gm_block_data_num
        if self.data_num == data_num_align:
            self._data_move(self.input_y[batch_index_start, 0], tensor_ub, 0, 1, batch_num * each_batch_gm_block_num,
                            0, 0)
        elif self.data_num % self.gm_block_data_num == 0:
            self._data_move(self.input_y[batch_index_start, 0], tensor_ub, 0, batch_num, each_batch_gm_block_num,
                            each_batch_ub_block_num - each_batch_gm_block_num, 0)
        else:
            last_block_ub_index, last_block_gm_index = self._get_last_block_info()
            with self.tik_inst.for_range(0, batch_num) as batch_index:
                self._data_move(self.input_y[batch_index_start + batch_index, 0], tensor_ub[batch_index, 0],
                                0, 1, each_batch_gm_block_num, 0, 0)
                self._data_move(self.input_y[batch_index_start + batch_index, last_block_gm_index],
                                tensor_ub[batch_index, last_block_ub_index], 0, 1, 1, 0, 0)


def if_support_dtype(params, support_dtype):
    """
    check dtype
    """
    for param in params:
        dtype = param.get("dtype").lower()
        if dtype not in support_dtype:
            return False
    return True


def if_support_format(params, support_format):
    """
    check format
    """
    for param in params:
        data_format = param.get("format")
        if data_format not in support_format:
            return False
    return True


def check_shape(params, support_shape):
    """
    check shape
    """
    for param in params:
        param_shape = param.get("shape")
        param_shape = tuple(param_shape)
        if param_shape != support_shape:
            return False
    return True


# 'pylint: disable=too-many-arguments,too-many-return-statements
def if_support_shape(input_x, input_gamma, input_beta, output_y,
                     output_mean, output_variance, begin_norm_axis):
    """
    check shape
    """
    input_shape = input_x.get("shape")
    input_shape = tuple(input_shape)
    if len(input_shape) <= 0:
        return False
    for dim_num in input_shape:
        if dim_num <= 0:
            return False
    if begin_norm_axis < 0:
        begin_norm_axis += len(input_shape)
    if begin_norm_axis < 0 or begin_norm_axis >= len(input_shape):
        return False
    gamma_shape = input_shape[begin_norm_axis:]
    mean_shape = list(input_shape)
    for dim_num in range(begin_norm_axis, len(input_shape)):
        mean_shape[dim_num] = 1
    mean_shape = tuple(mean_shape)
    if not check_shape((output_y,), input_shape):
        return False
    if not check_shape((input_gamma, input_beta), gamma_shape):
        return False
    if not check_shape((output_mean, output_variance), mean_shape):
        return False

    min_support_num = 128
    batch_num, data_num, begin_norm_axis = LayerNormalizeBase.split_shape(input_x, begin_norm_axis)
    if data_num < min_support_num:
        return False
    aicore_num = min(batch_num, tbe_platform.get_soc_spec(tbe_platform.CORE_NUM))
    if aicore_num == 1 and data_num % 16 == 0:
        return False
    return True


# 'pylint: disable=too-many-arguments,too-many-locals
def if_tik_support(input_x, input_gamma, input_beta,
                   output_y, output_mean, output_variance,
                   begin_norm_axis, begin_params_axis, epsilon):
    """
    check if tik support or not
    """
    if_support = True
    params = (input_x, input_gamma, input_beta, output_y, output_mean, output_variance)
    support_version = ("Ascend910",)
    support_dtype = (input_x.get("dtype").lower(),)
    support_format = ("NCHW", "ND")

    soc_version = tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION)
    aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    max_aicore_num = 32
    if soc_version not in support_version:
        if_support = False
    elif aicore_num > max_aicore_num or aicore_num <= 0:
        if_support = False
    elif not if_support_dtype(params, support_dtype):
        if_support = False
    elif not if_support_format(params, support_format):
        if_support = False
    elif begin_norm_axis != begin_params_axis:
        if_support = False
    elif epsilon <= 0:
        if_support = False
    elif not if_support_shape(input_x, input_gamma, input_beta, output_y,
                              output_mean, output_variance, begin_norm_axis):
        if_support = False
    return if_support


def select_layer_normalize(batch_num, data_num, gm_type, cont, ai_core_use):
    """
    select compute branch
    """
    obj = LayerNormalizeSplitD
    if batch_num <= ai_core_use:
        ub_size = cont.const_ub_max_byte
    else:
        ub_size = cont.const_ub_max_byte // 2
    mode_split_n_max_num = LayerNormalizeBase.mode_split_n_max_num(ub_size, gm_type, cont)
    if data_num <= mode_split_n_max_num:
        obj = LayerNormalizeSplitN

    return obj


# 'pylint: disable=unused-argument,too-many-arguments,too-many-locals
def layer_normalize(input_x, gamma, beta, input_y, mean, variance, begin_norm_axis, begin_params_axis, epsilon,
                    kernel_name="LayerNormalize"):
    """
    op layer_norm
    """
    AContainer.reset_instance()
    cont = AContainer.get_instance()
    batch_num, data_num, _ = LayerNormalizeBase.split_shape(input_x, begin_norm_axis)

    data_type = input_x.get("dtype").lower()
    ai_core_use = cont.const_aicore_num

    elementwise = True
    output_mean_var = True
    class_layer_normalize = select_layer_normalize(batch_num, data_num, data_type, cont, ai_core_use)
    obj_layer_normalize = class_layer_normalize(batch_num, data_num, epsilon, kernel_name,
                                                cont, data_type, elementwise, output_mean_var)
    obj_layer_normalize.mode_compute()
