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
from impl.util.platform_adapter import op_tiling
from impl.util.platform_adapter import shape_util
from impl.ascend import AContainer
from impl.ascend import TensorOperatorParam
from impl.ascend import VecCmd
from impl.ascend import VecExecutor
from impl.util.platform_adapter import operation


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    TILING_ARG_NUM = 8
    DTYPE_INT32 = "int32"
    MAX_MEM_SIZE = 36
    MAX_INT32_SIZE = 2**31 - 1
    SELECT_KEY_MODE_SPLIT_N = 0
    SELECT_KEY_MODE_SPLIT_D = 1
    CONST = "const"
    DYNAMIC = "dynamic"


# 'pylint: disable=too-many-instance-attributes
class LayerNormalizeBase:
    """
    layer normalize
    """

    # 'pylint: disable=too-many-arguments
    def __init__(self, tiling_ub_list, epsilon, kernel_name, cont, data_type, elementwise, output_mean_var,
                 input_output_tensor_dict, core_index, sync_workspace, n_max_data_num, input_mode,
                 atomic_clean_diff_shape):
        """
        init LayerNormalize attrs, init gm tensor
        Args:
            epsilon: float; Minimum positive number greater than 0
            kernel_name: str
            cont: AContainer
            data_type: str; support float16, float32
            elementwise: if count x * gamma + beta
            output_mean_var: if output mean variance
        """
        self.each_loop_mv_num = None
        self.epsilon = epsilon
        self.elementwise = elementwise
        self.output_mean_var = output_mean_var
        self.kernel_name = kernel_name
        self.atomic_clean_diff_shape = atomic_clean_diff_shape

        self.cont = cont
        self.tik = self.cont.tik
        self.tik_inst = self.cont.tinst
        self.core_index = core_index
        self.repeat_time_max = self.cont.const_vector_proc_max_rpt
        self.ub_size = self.cont.const_ub_max_byte
        self.max_ub_size = Constant.MAX_MEM_SIZE
        self.sync_workspace = sync_workspace
        self.block_size = self.cont.const_block_byte
        self.n_max_data_num = n_max_data_num
        self.input_mode = input_mode

        self.gm_type = data_type
        # __get tiling_args
        self.tiling_ub_list = tiling_ub_list
        self.tiling_args()
        self.gm_data_size, self.gm_block_data_num, self.gm_repeat_data_num = self._get_type_const(self.gm_type)

        self.ub_type = "float32"
        self.ub_data_size, self.ub_block_data_num, self.ub_repeat_data_num = self._get_type_const(self.ub_type)

        self.input_x = input_output_tensor_dict["input_x"]
        if self.elementwise:
            self.gamma = input_output_tensor_dict["gamma"]
            self.beta = input_output_tensor_dict["beta"]
        self.input_y = input_output_tensor_dict["input_y"]
        if self.output_mean_var:
            self.mean = input_output_tensor_dict["mean"]
            self.variance = input_output_tensor_dict["variance"]
            if self.gm_type != self.ub_type:
                self.y_ub_type = input_output_tensor_dict["y_ub_type"]
                self.mean_ub_type = input_output_tensor_dict["mean_ub_type"]
                self.variance_ub_type = input_output_tensor_dict["variance_ub_type"]

    def _update_gm_repeat_data_num(self, data_num):
        # scalar is max uint64, limit get mask func return mask_l and mask_h value exceed the max number
        if self.gm_repeat_data_num > 64:
            data_num_repeat_floor = data_num // self.gm_repeat_data_num * self.gm_repeat_data_num
            data_num_repeat_ceil = (data_num + self.gm_block_data_num -
                                    1) // self.gm_block_data_num * self.gm_block_data_num
            if isinstance(data_num, int):
                if data_num_repeat_ceil - data_num_repeat_floor > 64:
                    self.gm_repeat_data_num = 64
            else:
                with self.tik_inst.if_scope(data_num_repeat_ceil - data_num_repeat_floor > 64):
                    self.gm_repeat_data_num = 64

    def tiling_args(self):
        """
        tiling_args
        """
        self.tiling_gm = self.tiling_ub_list[0]
        self.tiling_mode = self.tiling_ub_list[1]
        self.batch_num = self.tiling_ub_list[2]
        self.data_num = self.tiling_ub_list[3]
        self.act_core_use = self.tiling_ub_list[4]
        self.each_core_batch_num = self.tiling_ub_list[5]
        self.last_core_batch_num = self.tiling_ub_list[6]
        self.mean_cof = self.tiling_ub_list[7]

    @staticmethod
    def _ceil_div(dividend, divisor):
        result = (dividend + divisor - 1) // divisor
        return result

    def _get_loop_info(self, all_data_num, each_loop_num):
        loop_times = self._ceil_div(all_data_num, each_loop_num)
        last_loop_num = all_data_num - each_loop_num * (loop_times - 1)
        return loop_times, last_loop_num

    def _get_align_num(self, input_num, align_num, ceil=True):
        if ceil:
            result = self._ceil_div(input_num, align_num) * align_num
        else:
            result = input_num // align_num * align_num
        return result

    def _get_type_const(self, data_type):
        data_size = self.cont.const_dtype_byte.get(data_type)
        block_data_num = self.cont.get_vec_proc_num_per_cmd_blk(data_type)
        repeat_data_num = self.cont.get_vec_proc_num_per_cmd(data_type)
        return data_size, block_data_num, repeat_data_num

    # 'pylint: disable=too-many-locals
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

        index_start = data_num - data_num_floor

        index_end = data_num_ceil - data_num_floor

        # if D split, data_num maybe int in dynamic mode
        if self.input_mode == Constant.DYNAMIC and not isinstance(data_num, int):
            mask_h = self.tik_inst.Scalar("uint64", "mask_h", 0)
            mask_l = self.tik_inst.Scalar("uint64", "mask_l", 0)
            with self.tik_inst.if_scope(index_start < min(index_end, mask_split)):
                mul_mask_l = self.tik_inst.Scalar("uint64", "mul_mask_l", 0)
                with self.tik_inst.for_range(index_start, min(index_end, mask_split)) as index_l:
                    mul_mask_l.set_as(1)

                    with self.tik_inst.for_range(0, index_l):
                        mul_mask_l.set_as(mul_mask_l * 2)

                    mask_l.set_as(mask_l + mul_mask_l)
            with self.tik_inst.if_scope(max(mask_split, index_start) < index_end):
                mul_mask_h = self.tik_inst.Scalar("uint64", "mul_mask_h", 0)
                with self.tik_inst.for_range(max(mask_split, index_start), index_end) as index_h:
                    mul_mask_h.set_as(1)

                    with self.tik_inst.for_range(0, (index_h - mask_split)):
                        mul_mask_h.set_as(mul_mask_h * 2)
                    mask_h.set_as(mask_h + mul_mask_h)
        else:
            mask_h, mask_l = 0, 0
            for index_l in range(index_start, min(index_end, mask_split)):
                mask_l += 2**index_l
            for index_h in range(max(mask_split, index_start), index_end):
                mask_h += 2**(index_h - mask_split)
        return mask_h, mask_l, data_num_floor

    @staticmethod
    def split_shape(input_tensor, dim_split):
        """
        split shape
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

    def _init_mean_tensor(self, max_batch_num, batch_num):
        """
        init tensor
        """
        data_type = self.ub_type
        num_per_cmd = self.ub_repeat_data_num
        mean_shape = (max_batch_num,)

        batch_mean_ub = self.tik_inst.Tensor(data_type, mean_shape, self.tik.scope_ubuf, "batch_mean_ub")
        batch_mean_square_ub = self.tik_inst.Tensor(data_type, mean_shape, self.tik.scope_ubuf, "batch_mean_square_ub")
        batch_variance_ub = self.tik_inst.Tensor(data_type, mean_shape, self.tik.scope_ubuf, "batch_variance_ub")

        buf_mean_all = {
            "batch_mean_ub": TensorOperatorParam(batch_mean_ub, batch_num, 0, num_per_cmd=num_per_cmd),
            "batch_mean_square_ub": TensorOperatorParam(batch_mean_square_ub, batch_num, 0, num_per_cmd=num_per_cmd),
            "batch_variance_ub": TensorOperatorParam(batch_variance_ub, batch_num, 0, num_per_cmd=num_per_cmd)
        }
        cmd_dup_mean_tensor = [
            VecCmd(cmd_name="vector_dup", dst_name="batch_mean_ub", scalar=0),
            VecCmd(cmd_name="vector_dup", dst_name="batch_mean_square_ub", scalar=0),
            VecCmd(cmd_name="vector_dup", dst_name="batch_variance_ub", scalar=0),
        ]
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
            block_num = self._ceil_div(batch_num, self.ub_block_data_num)
            self.data_move(mean_gm[batch_index_start], batch_mean_ub, 0, 1, block_num, 0, 0)
            self.data_move(variance_gm[batch_index_start], batch_variance_ub, 0, 1, block_num, 0, 0)
            self.tik_inst.set_atomic_add(0)

    def _mean_var_move_out_each_max_batch(self, batch_index_start, batch_num):
        self.each_loop_mv_num = int(self.n_max_data_num / 2)
        mv_loop_num, last_loop_mv_num = self._get_loop_info(batch_num, self.each_loop_mv_num)
        with self.tik_inst.for_range(0, mv_loop_num) as loop_index:
            batch_index = batch_index_start + loop_index * self.each_loop_mv_num
            with self.tik_inst.if_scope(loop_index != mv_loop_num - 1):
                self._mean_var_move_out_fp16(batch_index, self.each_loop_mv_num)
            with self.tik_inst.else_scope():
                self._mean_var_move_out_fp16(batch_index, last_loop_mv_num)

    def _mean_var_move_out_fp16(self, batch_index_start, batch_num):
        batch_num_align = self._get_align_num(batch_num, self.gm_block_data_num)
        if self.input_mode == Constant.DYNAMIC:
            batch_num_align_shape = self._get_align_num(self.each_loop_mv_num, self.gm_block_data_num)
        else:
            batch_num_align_shape = batch_num_align
        block_num_move_in = batch_num_align // self.ub_block_data_num
        block_num_move_out = batch_num_align // self.gm_block_data_num
        ub_shape = (batch_num_align_shape,)
        mean_ub_type_ub = self.tik_inst.Tensor(self.ub_type, ub_shape, self.tik.scope_ubuf, "mean_ub_type_ub")
        mean_gm_type_ub = self.tik_inst.Tensor(self.gm_type, ub_shape, self.tik.scope_ubuf, "mean_gm_type_ub")
        var_ub_type_ub = self.tik_inst.Tensor(self.ub_type, ub_shape, self.tik.scope_ubuf, "var_ub_type_ub")
        var_gm_type_ub = self.tik_inst.Tensor(self.gm_type, ub_shape, self.tik.scope_ubuf, "var_gm_type_ub")
        buf_mean_all = {
            "mean_ub_type_ub":
                TensorOperatorParam(mean_ub_type_ub, batch_num_align, 0, num_per_cmd=self.ub_repeat_data_num),
            "mean_gm_type_ub":
                TensorOperatorParam(mean_gm_type_ub, batch_num_align, 0, num_per_cmd=self.ub_repeat_data_num),
            "var_ub_type_ub":
                TensorOperatorParam(var_ub_type_ub, batch_num_align, 0, num_per_cmd=self.ub_repeat_data_num),
            "var_gm_type_ub":
                TensorOperatorParam(var_gm_type_ub, batch_num_align, 0, num_per_cmd=self.ub_repeat_data_num)
        }
        cmd_vconv = [
            VecCmd(cmd_name="vconv", dst_name="mean_gm_type_ub", src0_name="mean_ub_type_ub", round_mode=""),
            VecCmd(cmd_name="vconv", dst_name="var_gm_type_ub", src0_name="var_ub_type_ub", round_mode="")
        ]
        self.data_move(mean_ub_type_ub, self.mean_ub_type[batch_index_start], 0, 1, block_num_move_in, 0, 0)
        self.data_move(var_ub_type_ub, self.variance_ub_type[batch_index_start], 0, 1, block_num_move_in, 0, 0)
        VecExecutor.exec_vec_cmd(buf_mean_all, cmd_vconv, "mean_ub_type_ub")
        self.data_move(self.mean[batch_index_start], mean_gm_type_ub, 0, 1, block_num_move_out, 0, 0)
        self.data_move(self.variance[batch_index_start], var_gm_type_ub, 0, 1, block_num_move_out, 0, 0)

    def _count_rec_std_ne_mean(self, buf_mean_all):
        cmd_count_rec_std_ne_mean = [
            VecCmd(cmd_name="vmuls", dst_name="batch_mean_ub", src0_name="batch_mean_ub", scalar=-1),  # -E(x)
            VecCmd(cmd_name="vadds", dst_name="batch_variance_ub", src0_name="batch_variance_ub",
                   scalar=self.epsilon),  # var(x) + eps
            VecCmd(cmd_name="vsqrt", dst_name="batch_variance_ub", src0_name="batch_variance_ub"),  # std(x)
            VecCmd(cmd_name="vector_dup", dst_name="batch_mean_square_ub", scalar=1),  # 1
            VecCmd(cmd_name="vdiv",
                   dst_name="batch_variance_ub",
                   src0_name="batch_mean_square_ub",
                   src1_name="batch_variance_ub"),
        ]  # 1/std(x)
        VecExecutor.exec_vec_cmd(buf_mean_all, cmd_count_rec_std_ne_mean, "batch_mean_ub")

    # 'pylint: disable=too-many-arguments
    def data_move(self, dst, src, sid, nburst, burst, src_stride, dst_stride):
        """
        move data
        """
        with self.tik_inst.if_scope(nburst > 0 and burst > 0):
            self.tik_inst.data_move(dst, src, sid, nburst, burst, src_stride, dst_stride)

    def _mode_compute(self, mode_compute_each_core):
        batch_index = self.each_core_batch_num * self.core_index
        with self.tik_inst.if_scope(self.core_index < self.act_core_use - 1):
            mode_compute_each_core(batch_index, self.each_core_batch_num)
        with self.tik_inst.elif_scope(self.core_index == self.act_core_use - 1):
            mode_compute_each_core(batch_index, self.last_core_batch_num)

    def _mean_var_move_out_base(self):
        batch_index = self.each_core_batch_num * self.core_index
        if self.gm_type != self.ub_type:
            if self.input_mode == Constant.DYNAMIC:
                with self.tik_inst.if_scope(self.act_core_use > 1):
                    self.tik_inst.block_barrier(self.sync_workspace)
            elif self.input_mode != Constant.DYNAMIC and self.act_core_use > 1:
                self.tik_inst.block_barrier(self.sync_workspace)
            with self.tik_inst.if_scope(self.core_index < self.act_core_use - 1):
                self._mean_var_move_out_each_max_batch(batch_index, self.each_core_batch_num)
            with self.tik_inst.elif_scope(self.core_index == self.act_core_use - 1):
                self._mean_var_move_out_each_max_batch(batch_index, self.last_core_batch_num)

    def _y_workspace_move_out(self):
        batch_index = self.each_core_batch_num * self.core_index
        if self.gm_type != self.ub_type and self.input_mode == Constant.DYNAMIC and self.atomic_clean_diff_shape:
            with self.tik_inst.if_scope(self.tiling_mode == 0):
                with self.tik_inst.if_scope(self.data_num < self.gm_block_data_num):
                    with self.tik_inst.if_scope(self.core_index < self.act_core_use - 1):
                        self._y_workspace_move_out_each_max_batch(batch_index, self.each_core_batch_num)
                    with self.tik_inst.elif_scope(self.core_index == self.act_core_use - 1):
                        self._y_workspace_move_out_each_max_batch(batch_index, self.last_core_batch_num)
        elif ((self.gm_type != self.ub_type) and (self.input_mode != Constant.DYNAMIC)
              and (self.tiling_mode == 0) and self.atomic_clean_diff_shape):
            if self.data_num < self.gm_block_data_num:
                with self.tik_inst.if_scope(self.core_index < self.act_core_use - 1):
                    self._y_workspace_move_out_each_max_batch(batch_index, self.each_core_batch_num)
                with self.tik_inst.elif_scope(self.core_index == self.act_core_use - 1):
                    self._y_workspace_move_out_each_max_batch(batch_index, self.last_core_batch_num)

    def _y_workspace_move_out_each_max_batch(self, batch_index_start, batch_num):
        with self.tik_inst.for_range(0, batch_num) as loop_index:
            batch_index = batch_index_start + loop_index
            self._y_move_out_fp16(batch_index)

    def _y_move_out_fp16(self, batch_index_start):
        data_num_align = self.gm_block_data_num
        block_num_move_in = data_num_align // self.ub_block_data_num
        block_num_move_out = data_num_align // self.gm_block_data_num
        ub_shape = (data_num_align,)
        y_ub_type_ub = self.tik_inst.Tensor(self.ub_type, ub_shape, self.tik.scope_ubuf, "y_ub_type_ub")
        y_gm_type_ub = self.tik_inst.Tensor(self.gm_type, ub_shape, self.tik.scope_ubuf, "y_gm_type_ub")

        buf_y_all = {
            "y_ub_type_ub": TensorOperatorParam(y_ub_type_ub, data_num_align, 0, num_per_cmd=self.ub_repeat_data_num),
            "y_gm_type_ub": TensorOperatorParam(y_gm_type_ub, data_num_align, 0, num_per_cmd=self.ub_repeat_data_num)
        }
        cmd_vconv = [
            VecCmd(cmd_name="vconv", dst_name="y_gm_type_ub", src0_name="y_ub_type_ub", round_mode=""),
        ]
        self.data_move(y_ub_type_ub, self.y_ub_type[batch_index_start, 0], 0, 1, block_num_move_in, 0, 0)
        VecExecutor.exec_vec_cmd(buf_y_all, cmd_vconv, "y_ub_type_ub")
        self.data_move(self.input_y[batch_index_start, 0], y_gm_type_ub, 0, 1, block_num_move_out, 0, 0)

    def build_cce(self):
        """
        build_cce
        """
        if self.elementwise:
            inputs_all = [self.input_x, self.gamma, self.beta]
        else:
            inputs_all = [self.input_x]
        if self.output_mean_var:
            outputs_all = [self.input_y, self.mean, self.variance]
        else:
            outputs_all = [self.input_y]
        op_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        if self.input_mode == Constant.DYNAMIC:
            self.tik_inst.BuildCCE(inputs=inputs_all,
                                   outputs=outputs_all,
                                   flowtable=[self.tiling_gm],
                                   kernel_name=self.kernel_name,
                                   config=op_config)
        else:
            self.tik_inst.BuildCCE(inputs=inputs_all,
                                   outputs=outputs_all,
                                   kernel_name=self.kernel_name,
                                   config=op_config)
        return self.tik_inst


class LayerNormalizeSplitD(LayerNormalizeBase):
    """
    layer normalize:
        split mode: split d dim, each batch count
    """

    def __init__(self, *args, **kwargs):
        """
        init
        """
        super(LayerNormalizeSplitD, self).__init__(*args, **kwargs)
        self.each_loop_stand_num = None
        self.each_loop_mean_num = None

    def mode_compute(self):
        """
        start mode compute, d larger
        """
        self._mode_compute(self._mode_compute_each_core)

    def _mode_compute_each_core(self, batch_index_start, batch_num):
        thread_num = 1
        with self.tik_inst.for_range(0, batch_num, thread_num=thread_num) as batch_index_temp:
            batch_index = batch_index_start + batch_index_temp
            self._mode_compute_each_loop(batch_index, thread_num)

    def _mode_compute_each_loop(self, batch_index, thread_num):
        ub_size = self.cont.const_ub_max_byte // thread_num
        buf_mean_all = self._init_mean_tensor(self.ub_block_data_num, self.ub_block_data_num)
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
        self.each_loop_mean_num = self._get_mean_each_loop_num(ub_size)
        mean_loop_num, last_loop_mean_num = self._get_loop_info(self.data_num, self.each_loop_mean_num)
        self._update_gm_repeat_data_num(self.each_loop_mean_num)
        self._update_gm_repeat_data_num(last_loop_mean_num)
        with self.tik_inst.for_range(0, mean_loop_num) as loop_index:
            start_index = loop_index * self.each_loop_mean_num
            with self.tik_inst.if_scope(loop_index != mean_loop_num - 1):
                self._count_mean_each_batch_loop(batch_index, start_index, self.each_loop_mean_num, mean_ub_all)
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
        if self.input_mode == Constant.DYNAMIC:
            data_num_align_shape = self._get_align_num(self.each_loop_mean_num, self.ub_repeat_data_num)
        else:
            data_num_align_shape = data_num_align

        input_data_ub = self.tik_inst.Tensor(self.ub_type, (data_num_align_shape,), self.tik.scope_ubuf,
                                             "input_data_ub")
        buf_sum_ub = {
            "input_data_ub": TensorOperatorParam(input_data_ub, data_num_align, 0, num_per_cmd=self.ub_repeat_data_num)
        }

        repeat_num_shape = data_num_align_shape // self.ub_repeat_data_num
        repeat_num = data_num_align // self.ub_repeat_data_num
        work_tensor_num = self._get_align_num(repeat_num_shape, self.ub_block_data_num)
        work_tensor_ub = self.tik_inst.Tensor(self.ub_type, (work_tensor_num,), self.tik.scope_ubuf, "work_tensor_ub")

        gm_data_tensor_ub = None
        if self.gm_type != self.ub_type:
            gm_data_tensor_ub = self.tik_inst.Tensor(self.gm_type, (data_num_align_shape,), self.tik.scope_ubuf,
                                                     "gm_data_tensor_ub")

            buf_sum_ub["gm_data_tensor_ub"] = TensorOperatorParam(gm_data_tensor_ub,
                                                                  data_num_align,
                                                                  0,
                                                                  num_per_cmd=self.gm_repeat_data_num)
            buf_sum_ub["gm_data_tensor_vconv_ub"] = TensorOperatorParam(gm_data_tensor_ub,
                                                                        data_num_align,
                                                                        0,
                                                                        num_per_cmd=self.ub_repeat_data_num)

        self._data_move_in(input_data_ub, "input_data_ub", gm_data_tensor_ub, self.input_x, buf_sum_ub, batch_index,
                           start_index, data_num, 0)
        # count mean(x)
        self._count_mean_each_batch_loop_count(mean_temp_ub, input_data_ub, work_tensor_ub, repeat_num, mean_data_ub)
        # count mean(x^2)
        cmd_square = [
            VecCmd(cmd_name="vmul", dst_name="input_data_ub", src0_name="input_data_ub", src1_name="input_data_ub")
        ]
        VecExecutor.exec_vec_cmd(buf_sum_ub, cmd_square, "input_data_ub")
        self._count_mean_each_batch_loop_count(mean_temp_ub, input_data_ub, work_tensor_ub, repeat_num,
                                               mean_data_square_ub)

    # 'pylint: disable=too-many-arguments,too-many-locals
    def _data_move_in(self, ub_data_tensor_ub, ub_data_tensor_ub_name, gm_data_tensor_ub, tensor_gm, buf_sum_ub,
                      batch_index, start_index, data_num, mode):
        if self.gm_type != self.ub_type:
            tensor_ub, tensor_name = gm_data_tensor_ub, "gm_data_tensor_ub"
        else:
            tensor_ub, tensor_name = ub_data_tensor_ub, ub_data_tensor_ub_name
        cmd_dup_tensor = [VecCmd(cmd_name="vector_dup", dst_name=tensor_name, scalar=0)]
        VecExecutor.exec_vec_cmd(buf_sum_ub, cmd_dup_tensor, tensor_name)
        if mode == 0:
            block_num = self._ceil_div(data_num, self.gm_block_data_num)
            self.data_move(tensor_ub, tensor_gm[batch_index, start_index], 0, 1, block_num, 0, 0)
            if self.input_mode == Constant.DYNAMIC and not isinstance(data_num, int):
                with self.tik_inst.if_scope(data_num % self.gm_block_data_num != 0):
                    mask_h, mask_l, data_num_floor = self._get_mask(data_num, self.gm_repeat_data_num,
                                                                    self.gm_block_data_num)
                    self.tik_inst.vector_dup([mask_h, mask_l], tensor_ub[data_num_floor], 0.0, 1, 1, 8)

            else:
                if data_num % self.gm_block_data_num != 0:
                    mask_h, mask_l, data_num_floor = self._get_mask(data_num, self.gm_repeat_data_num,
                                                                    self.gm_block_data_num)
                    self.tik_inst.vector_dup([mask_h, mask_l], tensor_ub[data_num_floor], 0.0, 1, 1, 8)
        else:
            block_num = data_num // self.gm_block_data_num
            self.data_move(tensor_ub, tensor_gm[batch_index, start_index], 0, 1, block_num, 0, 0)
            if self.input_mode == Constant.DYNAMIC and not isinstance(data_num, int):
                with self.tik_inst.if_scope(data_num % self.gm_block_data_num != 0):
                    last_block_ub_index, last_block_gm_index = self._get_last_block_info(start_index, data_num)
                    self.data_move(tensor_ub[last_block_ub_index], tensor_gm[batch_index, last_block_gm_index], 0, 1, 1,
                                   0, 0)

            else:
                if data_num % self.gm_block_data_num != 0:
                    last_block_ub_index, last_block_gm_index = self._get_last_block_info(start_index, data_num)
                    self.data_move(tensor_ub[last_block_ub_index], tensor_gm[batch_index, last_block_gm_index], 0, 1, 1,
                                   0, 0)

        if self.gm_type != self.ub_type:
            cmd_vconv = [
                VecCmd(cmd_name="vconv",
                       dst_name=ub_data_tensor_ub_name,
                       src0_name="gm_data_tensor_vconv_ub",
                       round_mode="")
            ]
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
        self.tik_inst.vmuls(mask_mean, mean_temp_ub, mean_temp_ub, self.mean_cof, 1, 1, 1, 8, 8)
        self.tik_inst.vadd(mask_mean, mean_data_ub, mean_data_ub, mean_temp_ub, 1, 1, 1, 1, 8, 8, 8)

    @staticmethod
    def _count_mean_var(buf_mean_all):
        # count mean, mean_square, variance
        cmd_count_mean_var = [
            VecCmd(cmd_name="vmul", dst_name="batch_variance_ub", src0_name="batch_mean_ub",
                   src1_name="batch_mean_ub"),  # E(x)^2
            VecCmd(cmd_name="vsub",
                   dst_name="batch_variance_ub",
                   src0_name="batch_mean_square_ub",
                   src1_name="batch_variance_ub"),  # E(x)^2 - E(x)^2
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
        self.each_loop_stand_num = self._get_count_each_loop_num(ub_size)
        stand_loop_num, last_loop_stand_num, = self._get_loop_info(self.data_num, self.each_loop_stand_num)
        with self.tik_inst.for_range(0, stand_loop_num) as loop_index:
            start_index = loop_index * self.each_loop_stand_num
            with self.tik_inst.if_scope(loop_index != stand_loop_num - 1):
                self._count_stand_each_batch_loop(batch_index, start_index, self.each_loop_stand_num, scalar_all)
            with self.tik_inst.else_scope():
                self._count_stand_each_batch_loop(batch_index, start_index, last_loop_stand_num, scalar_all)

    def _get_count_each_loop_num(self, ub_size):
        gm_data_size = float(self.gm_data_size)
        ub_data_size = float(self.ub_data_size)
        tensor_num = 3
        ub_size_last = ub_size - tensor_num * self.block_size

        each_data_size = ub_data_size
        if self.elementwise:
            each_data_size += ub_data_size
        if self.gm_type != self.ub_type:
            each_data_size += gm_data_size
        data_num_max = int(ub_size_last / each_data_size)
        data_num_max_align = self._get_align_num(data_num_max, self.ub_repeat_data_num, False)
        return data_num_max_align

    # 'pylint: disable=too-many-locals
    def _count_stand_each_batch_loop(self, batch_index, start_index, data_num, scalar_all):
        ne_mean_scalar, rec_std_scalar = scalar_all

        data_num_align = self._get_align_num(data_num, self.ub_repeat_data_num)
        if self.input_mode == Constant.DYNAMIC:
            data_num_align_shape = self._get_align_num(self.each_loop_stand_num, self.ub_repeat_data_num)
        else:
            data_num_align_shape = data_num_align
        src_stand_ub = self.tik_inst.Tensor(self.ub_type, (data_num_align_shape,), self.tik.scope_ubuf, "src_stand_ub")
        buf_stand_ub = {
            "src_stand_ub": TensorOperatorParam(src_stand_ub, data_num_align, 0, num_per_cmd=self.ub_repeat_data_num)
        }

        gamma_beta_tensor_ub = None
        gm_data_tensor_ub = None
        if self.elementwise:
            gamma_beta_tensor_ub = self.tik_inst.Tensor(self.ub_type, (data_num_align_shape,), self.tik.scope_ubuf,
                                                        "gamma_beta_tensor_ub")
            buf_stand_ub["gamma_beta_tensor_ub"] = TensorOperatorParam(gamma_beta_tensor_ub,
                                                                       data_num_align,
                                                                       0,
                                                                       num_per_cmd=self.ub_repeat_data_num)
        if self.gm_type != self.ub_type:
            gm_data_tensor_ub = self.tik_inst.Tensor(self.gm_type, (data_num_align_shape,), self.tik.scope_ubuf,
                                                     "gm_data_tensor_ub")
            buf_stand_ub["gm_data_tensor_ub"] = TensorOperatorParam(gm_data_tensor_ub,
                                                                    data_num_align,
                                                                    0,
                                                                    num_per_cmd=self.gm_repeat_data_num)
            buf_stand_ub["gm_data_tensor_vconv_ub"] = TensorOperatorParam(gm_data_tensor_ub,
                                                                          data_num_align,
                                                                          0,
                                                                          num_per_cmd=self.ub_repeat_data_num)
        self._data_move_in(src_stand_ub, "src_stand_ub", gm_data_tensor_ub, self.input_x, buf_stand_ub, batch_index,
                           start_index, data_num, 1)
        cmd_stand = [
            VecCmd(cmd_name="vadds", dst_name="src_stand_ub", src0_name="src_stand_ub", scalar=ne_mean_scalar),
            VecCmd(cmd_name="vmuls", dst_name="src_stand_ub", src0_name="src_stand_ub", scalar=rec_std_scalar)
        ]
        VecExecutor.exec_vec_cmd(buf_stand_ub, cmd_stand, "src_stand_ub")

        if self.elementwise:
            self._data_move_in(gamma_beta_tensor_ub, "gamma_beta_tensor_ub", gm_data_tensor_ub, self.gamma,
                               buf_stand_ub, 0, start_index, data_num, 1)
            cmd_mul_gamma = [
                VecCmd(cmd_name="vmul",
                       dst_name="src_stand_ub",
                       src0_name="src_stand_ub",
                       src1_name="gamma_beta_tensor_ub")
            ]
            VecExecutor.exec_vec_cmd(buf_stand_ub, cmd_mul_gamma, "src_stand_ub")

            self._data_move_in(gamma_beta_tensor_ub, "gamma_beta_tensor_ub", gm_data_tensor_ub, self.beta, buf_stand_ub,
                               0, start_index, data_num, 1)
            cmd_add_beta = [
                VecCmd(cmd_name="vadd",
                       dst_name="src_stand_ub",
                       src0_name="src_stand_ub",
                       src1_name="gamma_beta_tensor_ub")
            ]
            VecExecutor.exec_vec_cmd(buf_stand_ub, cmd_add_beta, "src_stand_ub")
        self._mode_data_move_out(src_stand_ub, gm_data_tensor_ub, buf_stand_ub, batch_index, start_index, data_num)

    # 'pylint: disable=too-many-arguments
    def _mode_data_move_out(self, data_tensor_ub, gm_data_tensor_ub, buf_stand_ub, batch_index, start_index, data_num):
        if self.gm_type != self.ub_type:
            cmd_ub_vconv_gm = [
                VecCmd(cmd_name="vconv", dst_name="gm_data_tensor_vconv_ub", src0_name="src_stand_ub", round_mode="")
            ]
            VecExecutor.exec_vec_cmd(buf_stand_ub, cmd_ub_vconv_gm, "src_stand_ub")
            tensor_ub = gm_data_tensor_ub
        else:
            tensor_ub = data_tensor_ub
        block_num = data_num // self.gm_block_data_num
        self.data_move(self.input_y[batch_index, start_index], tensor_ub, 0, 1, block_num, 0, 0)
        if self.input_mode == Constant.DYNAMIC and not isinstance(data_num, int):
            with self.tik_inst.if_scope(data_num % self.gm_block_data_num != 0):
                last_block_ub_index, last_block_gm_index = self._get_last_block_info(start_index, data_num)
                self.data_move(self.input_y[batch_index, last_block_gm_index], tensor_ub[last_block_ub_index], 0, 1, 1,
                               0, 0)
        else:
            if data_num % self.gm_block_data_num != 0:
                last_block_ub_index, last_block_gm_index = self._get_last_block_info(start_index, data_num)
                self.data_move(self.input_y[batch_index, last_block_gm_index], tensor_ub[last_block_ub_index], 0, 1, 1,
                               0, 0)


class LayerNormalizeSplitN(LayerNormalizeBase):
    """
    layer normalize:
        split mode:split n dim, each loop count multi batch
    """

    def mode_compute(self):
        """
        compute
        """
        self._update_gm_repeat_data_num(self.data_num)
        self._mode_compute(self._mode1_compute_each_core)

    def _mode1_compute_each_core(self, batch_index_start, batch_num):
        thread_num = 1
        each_loop_batch_num, max_loop_batch_num = self._get_each_loop_batch_num(batch_num, thread_num)
        max_loop_batch_num += self.ub_block_data_num
        loop_times, last_loop_batch_num = self._get_loop_info(batch_num, each_loop_batch_num)
        with self.tik_inst.for_range(0, loop_times, thread_num=thread_num) as loop_index:
            batch_index = batch_index_start + each_loop_batch_num * loop_index
            with self.tik_inst.if_scope(loop_index != loop_times - 1):
                self._mode1_compute_each_loop(batch_index, each_loop_batch_num, max_loop_batch_num)
            with self.tik_inst.else_scope():
                self._mode1_compute_each_loop(batch_index, last_loop_batch_num, max_loop_batch_num)

    # 'pylint: disable=too-many-locals
    def _get_each_loop_batch_num(self, batch_num, thread_num):
        ub_size = self.ub_size // thread_num
        each_batch_data_num_align = self._get_align_num(self.data_num, self.ub_repeat_data_num)
        each_batch_data_num_align_1 = self._get_align_num(1, self.ub_repeat_data_num)

        ub_data_size = self.ub_data_size
        expand_tensor_num = 4
        expand_size = expand_tensor_num * (self.block_size - ub_data_size)
        ub_size_remain = ub_size - expand_size
        each_data_size = int(self._get_each_data_size(self.gm_type, self.cont) + 1)
        each_batch_data_size = each_data_size * each_batch_data_num_align + ub_data_size * expand_tensor_num
        each_batch_data_size_1 = each_data_size * each_batch_data_num_align_1 + ub_data_size * expand_tensor_num
        if self.input_mode == Constant.DYNAMIC:
            max_loop_batch_num = int(ub_size_remain) // each_batch_data_size
        else:
            max_loop_batch_num = int(ub_size_remain / each_batch_data_size)
        max_loop_batch_num_1 = int(ub_size_remain / each_batch_data_size_1)
        each_thread_batch_num = self._ceil_div(batch_num, thread_num)
        each_loop_batch_num = min(max_loop_batch_num, each_thread_batch_num)
        return each_loop_batch_num, max_loop_batch_num_1

    # 'pylint: disable=too-many-locals
    def _mode1_compute_each_loop(self, batch_index_start, batch_num, max_loop_batch_num):
        # fp32 data info
        batch_num_align = self._get_align_num(batch_num, self.ub_block_data_num)
        data_num_align = self._get_align_num(self.data_num, self.ub_repeat_data_num)
        data_repeat_num = data_num_align // self.ub_repeat_data_num

        if self.input_mode == Constant.DYNAMIC:
            max_batch_num_align = self._get_align_num(max_loop_batch_num, self.ub_block_data_num)
            max_data_num_align = self._get_align_num(self.n_max_data_num, self.ub_repeat_data_num)
            max_data_repeat_num = max_data_num_align // self.ub_repeat_data_num
            max_batch_num = self.n_max_data_num
        else:
            max_batch_num_align = batch_num_align
            max_data_repeat_num = data_repeat_num
            max_batch_num = batch_num

        # init vec_reduce_add work tensor
        work_tensor_num = self._get_align_num(max_data_repeat_num, self.ub_block_data_num)
        work_tensor_ub = self.tik_inst.Tensor(self.ub_type, (work_tensor_num,), self.tik.scope_ubuf, "work_tensor_ub")

        # init mean var ub
        buf_mean_all = self._init_mean_tensor(max_batch_num_align, batch_num_align)
        batch_mean_ub = buf_mean_all.get("batch_mean_ub").const_tensor
        batch_mean_square_ub = buf_mean_all.get("batch_mean_square_ub").const_tensor
        batch_variance_ub = buf_mean_all.get("batch_variance_ub").const_tensor

        # init data ub
        buf_data_all, gm_data_tensor_ub = self._init_data_tensor(max_batch_num, batch_num, data_num_align)
        input_data_ub = buf_data_all.get("input_data_ub").const_tensor
        input_data_square_ub = buf_data_all.get("input_data_square_ub").const_tensor
        data_ub_all = (input_data_ub, input_data_square_ub, gm_data_tensor_ub)
        mean_ub_all = (batch_mean_ub, batch_mean_square_ub, batch_variance_ub)
        cmd_gm_vconv_ub = [
            VecCmd(cmd_name="vconv", dst_name="input_data_ub", src0_name="gm_data_tensor_vconv_ub", round_mode=""),
            VecCmd(cmd_name="vconv",
                   dst_name="input_data_square_ub",
                   src0_name="gm_data_tensor_vconv_ub",
                   round_mode="")
        ]
        # data move in
        self._data_move_in(data_ub_all, self.input_x, buf_data_all, cmd_gm_vconv_ub, batch_index_start, batch_num,
                           data_num_align, 0)
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

    def _init_data_tensor(self, max_ub_num, batch_num, data_num):
        # init fp32 x tensor, fp32 x^2 tensor
        # init fp16 tensor
        gm_data_tensor_ub = None
        if self.input_mode == Constant.DYNAMIC:
            input_data_ub = self.tik_inst.Tensor(self.ub_type, (max_ub_num,), self.tik.scope_ubuf, "input_data_ub")
            input_data_ub = input_data_ub.reshape((max_ub_num // data_num, data_num))
            input_data_square_ub = self.tik_inst.Tensor(self.ub_type, (max_ub_num,), self.tik.scope_ubuf,
                                                        "input_data_square_ub")
            input_data_square_ub = input_data_square_ub.reshape((max_ub_num // data_num, data_num))
            if self.gm_type != self.ub_type:
                gm_data_tensor_ub = self.tik_inst.Tensor(self.gm_type, (max_ub_num,), self.tik.scope_ubuf,
                                                         "gm_data_tensor_ub")
                gm_data_tensor_ub = gm_data_tensor_ub.reshape((max_ub_num // data_num, data_num))
        else:
            input_data_ub = self.tik_inst.Tensor(self.ub_type, (batch_num, data_num), self.tik.scope_ubuf,
                                                 "input_data_ub")
            input_data_square_ub = self.tik_inst.Tensor(self.ub_type, (batch_num, data_num), self.tik.scope_ubuf,
                                                        "input_data_square_ub")
            if self.gm_type != self.ub_type:
                gm_data_tensor_ub = self.tik_inst.Tensor(self.gm_type, (batch_num, data_num), self.tik.scope_ubuf,
                                                         "gm_data_tensor_ub")
        buf_data_all = {
            "input_data_ub":
                TensorOperatorParam(input_data_ub, batch_num * data_num, 0, num_per_cmd=self.ub_repeat_data_num),
            "input_data_square_ub":
                TensorOperatorParam(input_data_square_ub, batch_num * data_num, 0, num_per_cmd=self.ub_repeat_data_num)
        }

        # init fp16 tensor
        if self.gm_type != self.ub_type:
            buf_data_all["gm_data_tensor_ub"] = TensorOperatorParam(gm_data_tensor_ub,
                                                                    batch_num * data_num,
                                                                    0,
                                                                    num_per_cmd=self.gm_repeat_data_num)
            buf_data_all["gm_data_tensor_vconv_ub"] = TensorOperatorParam(gm_data_tensor_ub,
                                                                          batch_num * data_num,
                                                                          0,
                                                                          num_per_cmd=self.ub_repeat_data_num)
            cmd_dup_input_tensor = [VecCmd(cmd_name="vector_dup", dst_name="gm_data_tensor_ub", scalar=0)]
            VecExecutor.exec_vec_cmd(buf_data_all, cmd_dup_input_tensor, "gm_data_tensor_ub")
        else:
            cmd_dup_input_tensor = [VecCmd(cmd_name="vector_dup", dst_name="input_data_ub", scalar=0)]
            VecExecutor.exec_vec_cmd(buf_data_all, cmd_dup_input_tensor, "input_data_ub")
        return buf_data_all, gm_data_tensor_ub

    # 'pylint: disable=too-many-arguments
    def _data_move_in(self, data_ub_all, tensor_gm, buf_data_all, cmd_gm_vconv_ub, batch_index_start, batch_num,
                      data_num_align, mode):
        # get block format num
        if self.gm_type != self.ub_type:
            tensor_ub = data_ub_all[2]
        else:
            tensor_ub = data_ub_all[mode]
        # get each batch info
        each_batch_ub_block_num = data_num_align // self.gm_block_data_num
        each_batch_gm_block_num = self.data_num // self.gm_block_data_num
        # if data 32byte align
        if self.input_mode == Constant.DYNAMIC:
            with self.tik_inst.if_scope(self.data_num == data_num_align):
                self.data_move(tensor_ub, tensor_gm[batch_index_start, 0], 0, 1, batch_num * each_batch_gm_block_num, 0,
                               0)
            with self.tik_inst.elif_scope(self.data_num % self.gm_block_data_num == 0):
                self.data_move(tensor_ub, tensor_gm[batch_index_start, 0], 0, batch_num, each_batch_gm_block_num, 0,
                               each_batch_ub_block_num - each_batch_gm_block_num)
            with self.tik_inst.else_scope():
                # if data not 32byte align
                self._data_move_in_not_align(tensor_ub, tensor_gm, batch_index_start, batch_num, mode)
        else:
            if self.data_num == data_num_align:
                self.data_move(tensor_ub, tensor_gm[batch_index_start, 0], 0, 1, batch_num * each_batch_gm_block_num, 0,
                               0)
            elif self.data_num % self.gm_block_data_num == 0:
                self.data_move(tensor_ub, tensor_gm[batch_index_start, 0], 0, batch_num, each_batch_gm_block_num, 0,
                               each_batch_ub_block_num - each_batch_gm_block_num)
            # if data not 32byte align
            else:
                self._data_move_in_not_align(tensor_ub, tensor_gm, batch_index_start, batch_num, mode)
        if self.gm_type != self.ub_type:
            VecExecutor.exec_vec_cmd(buf_data_all, [cmd_gm_vconv_ub[mode]], "gm_data_tensor_vconv_ub")

    # 'pylint: disable=too-many-arguments
    def _data_move_in_not_align(self, tensor_ub, tensor_gm, batch_index_start, batch_num, mode):
        if mode == 0:
            each_batch_block_num = self._ceil_div(self.data_num, self.gm_block_data_num)
            mask_h, mask_l, data_num_floor = self._get_mask(self.data_num, self.gm_repeat_data_num,
                                                            self.gm_block_data_num)
            with self.tik_inst.for_range(0, batch_num) as batch_index:
                self.data_move(tensor_ub[batch_index, 0], tensor_gm[batch_index_start + batch_index, 0], 0, 1,
                               each_batch_block_num, 0, 0)
                self.tik_inst.vector_dup([mask_h, mask_l], tensor_ub[batch_index, data_num_floor], 0.0, 1, 1, 8)
        else:
            block_num_floor = self.data_num // self.gm_block_data_num
            last_block_ub_index, last_block_gm_index = self._get_last_block_info()
            if self.input_mode != Constant.DYNAMIC and block_num_floor == 0:
                with self.tik_inst.for_range(0, batch_num) as batch_index:
                    self.data_move(tensor_ub[0, last_block_ub_index], tensor_gm[batch_index_start + batch_index,
                                                                                last_block_gm_index], 0, 1, 1, 0, 0)

            else:
                with self.tik_inst.for_range(0, batch_num) as batch_index:
                    with self.tik_inst.if_scope(block_num_floor != 0):
                        self.data_move(tensor_ub[batch_index, 0], tensor_gm[batch_index_start + batch_index, 0], 0, 1,
                                       block_num_floor, 0, 0)
                    self.data_move(tensor_ub[0, last_block_ub_index], tensor_gm[batch_index_start + batch_index,
                                                                                last_block_gm_index], 0, 1, 1, 0, 0)

    def _get_last_block_info(self, block_data_num=None):
        if block_data_num is None:
            block_data_num = self.gm_block_data_num
        last_block_ub_index = self._get_align_num(self.data_num, block_data_num, False)
        last_block_gm_index = self.data_num - block_data_num
        if self.input_mode == Constant.DYNAMIC:
            last_block_ub_index = self.tik_inst.Scalar("int32", "last_block_ub_index", last_block_ub_index)
            last_block_gm_index = self.tik_inst.Scalar("int32", "last_block_gm_index", last_block_gm_index)
            with self.tik_inst.if_scope(self.data_num < block_data_num):
                last_block_ub_index.set_as(0)
                last_block_gm_index.set_as(0)
        else:
            if last_block_gm_index < 0:
                last_block_ub_index, last_block_gm_index = 0, 0
        return last_block_ub_index, last_block_gm_index

    # 'pylint: disable=too-many-arguments
    def _mode1_count_sum(self, data_ub_all, mean_ub_all, reduce_work_tensor, batch_num, data_num_align):
        input_data_ub, input_data_square_ub, _ = data_ub_all
        batch_mean_ub, batch_mean_square_ub, _ = mean_ub_all
        mask = self.ub_repeat_data_num
        each_batch_repeat_num = data_num_align // mask
        with self.tik_inst.for_range(0, batch_num) as batch_index:
            self.tik_inst.vmul(mask, input_data_square_ub[batch_index, 0], input_data_ub[batch_index, 0],
                               input_data_ub[batch_index, 0], each_batch_repeat_num, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vec_reduce_add(mask, batch_mean_ub[batch_index], input_data_ub[batch_index, 0],
                                         reduce_work_tensor, each_batch_repeat_num, 8)
            self.tik_inst.vec_reduce_add(mask, batch_mean_square_ub[batch_index], input_data_square_ub[batch_index, 0],
                                         reduce_work_tensor, each_batch_repeat_num, 8)

    def _count_mean_var(self, buf_mean_all):
        # count mean, mean square, variance
        cmd_count_mean_var = [
            VecCmd(cmd_name="vmuls", dst_name="batch_mean_ub", src0_name="batch_mean_ub", scalar=self.mean_cof),  # E(x)
            VecCmd(cmd_name="vmuls",
                   dst_name="batch_mean_square_ub",
                   src0_name="batch_mean_square_ub",
                   scalar=self.mean_cof),  # E(x^2)
            VecCmd(cmd_name="vmul", dst_name="batch_variance_ub", src0_name="batch_mean_ub",
                   src1_name="batch_mean_ub"),  # E(x)^2
            VecCmd(cmd_name="vsub",
                   dst_name="batch_variance_ub",
                   src0_name="batch_mean_square_ub",
                   src1_name="batch_variance_ub"),  # E(x^2) - E(x)^2
            VecCmd(cmd_name="vabs", dst_name="batch_variance_ub", src0_name="batch_variance_ub")  # abs(var)
        ]
        VecExecutor.exec_vec_cmd(buf_mean_all, cmd_count_mean_var, "batch_mean_ub")

    def _mode1_adjust_input(self, data_ub_all, batch_index_start, batch_num):
        if self.input_mode == Constant.DYNAMIC:
            with self.tik_inst.if_scope(self.data_num % self.gm_block_data_num != 0):
                self._mode1_adjust_input_per(data_ub_all, batch_index_start, batch_num)
        elif self.data_num % self.gm_block_data_num != 0:
            self._mode1_adjust_input_per(data_ub_all, batch_index_start, batch_num)

    def _mode1_adjust_input_per(self, data_ub_all, batch_index_start, batch_num):
        input_data_ub, _, gm_data_tensor_ub = data_ub_all
        last_block_ub_index, last_block_gm_index = self._get_last_block_info()
        if self.gm_type == self.ub_type:
            with self.tik_inst.for_range(0, batch_num) as batch_index:
                self.data_move(input_data_ub[batch_index, last_block_ub_index],
                               self.input_x[batch_index_start + batch_index, last_block_gm_index], 0, 1, 1, 0, 0)
        else:
            with self.tik_inst.for_range(0, batch_num) as batch_index:
                self.data_move(gm_data_tensor_ub, self.input_x[batch_index_start + batch_index, last_block_gm_index], 0,
                               1, 1, 0, 0)
                self.tik_inst.vconv(self.gm_block_data_num, "", input_data_ub[batch_index, last_block_ub_index],
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
            self.tik_inst.vadds(mask, input_data_ub[batch_index, 0], input_data_ub[batch_index, 0], ne_mean_scalar,
                                repeat_num, 1, 1, 8, 8)
            self.tik_inst.vmuls(mask, input_data_ub[batch_index, 0], input_data_ub[batch_index, 0], rec_std_scalar,
                                repeat_num, 1, 1, 8, 8)

    # 'pylint: disable=too-many-arguments
    def _mode1_count_elementwise(self, data_ub_all, batch_num, data_num_align, buf_data_all, cmd_gm_vconv_ub):
        input_data_ub, input_data_square_ub, _ = data_ub_all
        mask = self.ub_repeat_data_num
        each_batch_repeat_num = data_num_align // self.ub_repeat_data_num

        self._data_move_in(data_ub_all, self.gamma, buf_data_all, cmd_gm_vconv_ub, 0, 1, data_num_align, 1)
        with self.tik_inst.for_range(0, batch_num) as batch_index:
            self.tik_inst.vmul(mask, input_data_ub[batch_index, 0], input_data_ub[batch_index, 0], input_data_square_ub,
                               each_batch_repeat_num, 1, 1, 1, 8, 8, 8)

        self._data_move_in(data_ub_all, self.beta, buf_data_all, cmd_gm_vconv_ub, 0, 1, data_num_align, 1)
        with self.tik_inst.for_range(0, batch_num) as batch_index:
            self.tik_inst.vadd(mask, input_data_ub[batch_index, 0], input_data_ub[batch_index, 0], input_data_square_ub,
                               each_batch_repeat_num, 1, 1, 1, 8, 8, 8)

    # 'pylint: disable=too-many-arguments,too-many-statements
    def _mode1_data_move_out(self, data_ub_all, buf_data_all, batch_index_start, batch_num, data_num_align):
        input_data_ub, _, gm_data_tensor_ub = data_ub_all
        if self.input_mode == Constant.DYNAMIC and self.atomic_clean_diff_shape:
            block_data_num = self.tik_inst.Scalar("int32", "block_data_num")
            move_mode = self.tik_inst.Scalar("int32", "move_mode")
            if self.ub_type == self.gm_type:
                tensor_ub = input_data_ub
                block_data_num.set_as(self.gm_block_data_num)
                input_y = self.input_y
                with self.tik_inst.if_scope(self.data_num < self.gm_block_data_num):
                    move_mode.set_as(1)
                    self.tik_inst.set_atomic_add(1)
                    self._mode1_data_move_out_exe(tensor_ub, batch_index_start, batch_num, data_num_align, input_y,
                                                  block_data_num, move_mode)
                    self.tik_inst.set_atomic_add(0)
                with self.tik_inst.else_scope():
                    move_mode.set_as(0)
                    self._mode1_data_move_out_exe(tensor_ub, batch_index_start, batch_num, data_num_align, input_y,
                                                  block_data_num, move_mode)
            else:
                with self.tik_inst.if_scope(self.data_num >= self.gm_block_data_num):
                    cmd_ub_vconv_gm = [
                        VecCmd(cmd_name="vconv",
                               dst_name="gm_data_tensor_vconv_ub",
                               src0_name="input_data_ub",
                               round_mode="")
                    ]
                    VecExecutor.exec_vec_cmd(buf_data_all, cmd_ub_vconv_gm, "input_data_ub")
                    tensor_ub = gm_data_tensor_ub
                    block_data_num.set_as(self.gm_block_data_num)
                    move_mode.set_as(0)
                    input_y = self.input_y
                    self._mode1_data_move_out_exe(tensor_ub, batch_index_start, batch_num, data_num_align, input_y,
                                                  block_data_num, move_mode)
                with self.tik_inst.else_scope():
                    tensor_ub = input_data_ub
                    block_data_num.set_as(self.ub_block_data_num)
                    move_mode.set_as(1)
                    input_y = self.y_ub_type
                    self.tik_inst.set_atomic_add(1)
                    self._mode1_data_move_out_exe(tensor_ub, batch_index_start, batch_num, data_num_align, input_y,
                                                  block_data_num, move_mode)
                    self.tik_inst.set_atomic_add(0)
        elif self.input_mode == Constant.DYNAMIC and not self.atomic_clean_diff_shape:
            block_data_num = self.tik_inst.Scalar("int32", "block_data_num")
            move_mode = self.tik_inst.Scalar("int32", "move_mode")
            if self.ub_type == self.gm_type:
                tensor_ub = input_data_ub
                block_data_num.set_as(self.gm_block_data_num)
                input_y = self.input_y
                move_mode.set_as(0)
                self._mode1_data_move_out_exe(tensor_ub, batch_index_start, batch_num, data_num_align, input_y,
                                              block_data_num, move_mode)
            else:
                cmd_ub_vconv_gm = [
                    VecCmd(cmd_name="vconv",
                           dst_name="gm_data_tensor_vconv_ub",
                           src0_name="input_data_ub",
                           round_mode="")
                ]
                VecExecutor.exec_vec_cmd(buf_data_all, cmd_ub_vconv_gm, "input_data_ub")
                tensor_ub = gm_data_tensor_ub
                block_data_num.set_as(self.gm_block_data_num)
                move_mode.set_as(0)
                input_y = self.input_y
                self._mode1_data_move_out_exe(tensor_ub, batch_index_start, batch_num, data_num_align, input_y,
                                              block_data_num, move_mode)
        else:
            if self.ub_type == self.gm_type:
                tensor_ub = input_data_ub
                block_data_num = self.gm_block_data_num
                input_y = self.input_y
                if self.data_num < self.gm_block_data_num and self.atomic_clean_diff_shape:
                    move_mode = 1
                    self.tik_inst.set_atomic_add(1)
                    self._mode1_data_move_out_exe(tensor_ub, batch_index_start, batch_num, data_num_align, input_y,
                                                  block_data_num, move_mode)
                    self.tik_inst.set_atomic_add(0)
                else:
                    move_mode = 0
                    self._mode1_data_move_out_exe(tensor_ub, batch_index_start, batch_num, data_num_align, input_y,
                                                  block_data_num, move_mode)
            elif (self.data_num >= self.gm_block_data_num and
                  self.ub_type != self.gm_type) or not self.atomic_clean_diff_shape:
                cmd_ub_vconv_gm = [
                    VecCmd(cmd_name="vconv",
                           dst_name="gm_data_tensor_vconv_ub",
                           src0_name="input_data_ub",
                           round_mode="")
                ]
                VecExecutor.exec_vec_cmd(buf_data_all, cmd_ub_vconv_gm, "input_data_ub")
                tensor_ub = gm_data_tensor_ub
                block_data_num = self.gm_block_data_num
                move_mode = 0
                input_y = self.input_y
                self._mode1_data_move_out_exe(tensor_ub, batch_index_start, batch_num, data_num_align, input_y,
                                              block_data_num, move_mode)
            else:
                tensor_ub = input_data_ub
                block_data_num = self.ub_block_data_num
                move_mode = 1
                input_y = self.y_ub_type
                self.tik_inst.set_atomic_add(1)
                self._mode1_data_move_out_exe(tensor_ub, batch_index_start, batch_num, data_num_align, input_y,
                                              block_data_num, move_mode)
                self.tik_inst.set_atomic_add(0)

    # 'pylint: disable=too-many-arguments
    def _mode1_data_move_out_exe(self, tensor_ub, batch_index_start, batch_num, data_num_align, input_y, block_data_num,
                                 move_mode):

        each_batch_ub_block_num = data_num_align // block_data_num
        each_batch_gm_block_num = self.data_num // block_data_num
        if self.input_mode == Constant.DYNAMIC:
            with self.tik_inst.if_scope(self.data_num == data_num_align):
                self.data_move(input_y[batch_index_start, 0], tensor_ub, 0, 1, batch_num * each_batch_gm_block_num, 0,
                               0)
            with self.tik_inst.elif_scope(self.data_num % self.gm_block_data_num == 0):
                self.data_move(input_y[batch_index_start, 0], tensor_ub, 0, batch_num, each_batch_gm_block_num,
                               each_batch_ub_block_num - each_batch_gm_block_num, 0)
            with self.tik_inst.else_scope():
                last_block_ub_index, last_block_gm_index = self._get_last_block_info(block_data_num)
                with self.tik_inst.if_scope(move_mode == 1):
                    last_block_gm_index.set_as(last_block_ub_index)
                with self.tik_inst.for_range(0, batch_num) as batch_index:
                    self.data_move(input_y[batch_index_start + batch_index, 0], tensor_ub[batch_index, 0], 0, 1,
                                   each_batch_gm_block_num, 0, 0)
                    self.data_move(input_y[batch_index_start + batch_index, last_block_gm_index],
                                   tensor_ub[batch_index, last_block_ub_index], 0, 1, 1, 0, 0)
        else:
            if self.data_num == data_num_align:
                self.data_move(input_y[batch_index_start, 0], tensor_ub, 0, 1, batch_num * each_batch_gm_block_num, 0,
                               0)
            elif self.data_num % self.gm_block_data_num == 0:
                self.data_move(input_y[batch_index_start, 0], tensor_ub, 0, batch_num, each_batch_gm_block_num,
                               each_batch_ub_block_num - each_batch_gm_block_num, 0)
            else:
                last_block_ub_index, last_block_gm_index = self._get_last_block_info(block_data_num)
                if move_mode == 1:
                    last_block_gm_index = last_block_ub_index
                with self.tik_inst.for_range(0, batch_num) as batch_index:
                    if each_batch_gm_block_num != 0:
                        self.data_move(input_y[batch_index_start + batch_index, 0], tensor_ub[batch_index, 0], 0, 1,
                                       each_batch_gm_block_num, 0, 0)
                    self.data_move(input_y[batch_index_start + batch_index, last_block_gm_index],
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


# 'pylint: disable=unused-argument,too-many-arguments
def if_support_shape(input_x, input_gamma, input_beta, output_y, output_mean, output_variance, begin_norm_axis):
    """
    check shape
    """
    input_shape = input_x.get("shape")
    input_shape = tuple(input_shape)
    if len(input_shape) <= 1 or len(input_shape) > 4:
        return False
    if begin_norm_axis < 0 or begin_norm_axis >= len(input_shape):
        return False
    return True


# 'pylint: disable=too-many-locals,too-many-arguments
def if_tik_support(input_x, input_gamma, input_beta, output_y, output_mean, output_variance, begin_norm_axis,
                   begin_params_axis, epsilon):
    """
    check if tik support or not
    """
    if_support = True
    params = (input_x, input_gamma, input_beta, output_y, output_mean, output_variance)
    support_version = ("Ascend910",)
    support_dtype = (input_x.get("dtype").lower(),)
    support_format = ("NCHW", "ND", "NHWC")
    if begin_norm_axis < 0:
        begin_norm_axis += len(input_x["shape"])
    if begin_params_axis < 0:
        begin_params_axis += len(input_x["shape"])
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
    elif not if_support_shape(input_x, input_gamma, input_beta, output_y, output_mean, output_variance,
                              begin_norm_axis):
        if_support = False
    return if_support


def tiling_init_dynamic(cont):
    """
    check if tik support or not
    """
    byte_size = cont.const_dtype_byte.get(Constant.DTYPE_INT32)
    block_size = cont.const_block_byte
    tiling_gm = cont.tinst.Tensor(Constant.DTYPE_INT32, (Constant.TILING_ARG_NUM,),
                                  name="tiling_gm",
                                  scope=cont.tik.scope_gm)
    tiling_ub = cont.tinst.Tensor(Constant.DTYPE_INT32, (Constant.TILING_ARG_NUM,),
                                  name="tiling_ub",
                                  scope=cont.tik.scope_ubuf)
    # move tiling params from gm to ub

    cont.tinst.data_move(tiling_ub, tiling_gm, 0, 1, Constant.TILING_ARG_NUM * byte_size // block_size, 0, 0)
    tiling_mode = cont.tinst.Scalar("int32", name="tiling_mode")
    tiling_mode.set_as(tiling_ub[0])
    batch_num = cont.tinst.Scalar("int32", name="batch_num")
    batch_num.set_as(tiling_ub[1])
    data_num = cont.tinst.Scalar("int32", name="data_num")
    data_num.set_as(tiling_ub[2])
    act_core_num = cont.tinst.Scalar("int32", name="act_core_num")
    act_core_num.set_as(tiling_ub[3])
    each_core_batch_num = cont.tinst.Scalar("int32", name="each_core_batch_num")
    each_core_batch_num.set_as(tiling_ub[4])
    last_core_batch_num = cont.tinst.Scalar("int32", name="last_core_batch_num")
    last_core_batch_num.set_as(tiling_ub[5])
    mean_cof = cont.tinst.Scalar("float32", name="mean_cof")
    mean_cof.set_as(tiling_ub[6])
    tiling_ub_list = [
        tiling_gm, tiling_mode, batch_num, data_num, act_core_num, each_core_batch_num, last_core_batch_num, mean_cof
    ]
    return tiling_ub_list


def _gen_tiling_case_const(input_x, gamma, beta, input_y):
    inputs = []
    inputs.insert(0, {"shape": shape_util.shape_to_list(input_x["shape"]), "dtype": input_x["dtype"]})
    inputs.insert(1, {"shape": shape_util.shape_to_list(gamma["shape"]), "dtype": gamma["dtype"]})
    inputs.insert(2, {"shape": shape_util.shape_to_list(beta["shape"]), "dtype": beta["dtype"]})
    outputs = [{"shape": shape_util.shape_to_list(input_y["shape"]), "dtype": input_y["dtype"]}]
    run_info = op_tiling.do_op_tiling(operation.get_context().get_op_type(), operation.get_compile_info(), inputs,
                                      outputs)
    tiling_format = {
        "tiling_mode": "int",
        "batch_num": "int",
        "data_num": "int",
        "act_core_num": "int",
        "each_core_batch_num": "int",
        "last_core_batch_num": "int",
        "mean_cof": "int"
    }
    tiling_data = op_tiling.decode(run_info["tiling_data"], tiling_format)
    tiling_data["mean_cof"] = 1.0 / tiling_data.get("data_num")
    tiling_ub_list = [None]
    for key in tiling_format:
        tiling_ub_list.append(tiling_data.get(key))
    return tiling_ub_list


def _check_input_mode(input_x):
    """
    if input_x shape has -1 or -2 ==> DYNAMIC mode
    else: CONST mode
    """
    x_shape = shape_util.shape_to_list(input_x["shape"])
    if -1 in x_shape or -2 in x_shape:
        return Constant.DYNAMIC
    return Constant.CONST


# 'pylint: disable=too-many-locals
def max_data_num(cont, gm_type, ub_type):
    """
    max_data_num
    """
    ub_size = cont.const_ub_max_byte
    expand_tensor_num = 4
    block_size = 32
    expand_size = expand_tensor_num * block_size
    ub_size_remain = ub_size - expand_size

    ub_data_size = cont.const_dtype_byte.get(ub_type)
    gm_data_size = cont.const_dtype_byte.get(gm_type)
    ub_repeat_data_num = cont.const_vector_proc_byte / ub_data_size

    count_tensor_num = 2
    each_data_size = count_tensor_num * ub_data_size + ub_data_size / ub_repeat_data_num
    if ub_type != gm_type:
        each_data_size += gm_data_size

    data_num_max = int(ub_size_remain / each_data_size)
    data_num_align = int((data_num_max + ub_repeat_data_num - 1) // ub_repeat_data_num * ub_repeat_data_num)
    const_vector_proc_max_rpt = cont.const_vector_proc_max_rpt
    data_num_max = const_vector_proc_max_rpt * ub_repeat_data_num
    n_max_data_num = max({data_num_align, data_num_max})
    return n_max_data_num


# 'pylint: disable=unused-argument,protected-access,too-many-arguments,too-many-locals,too-many-statements
def layer_normalize(input_x,
                    gamma,
                    beta,
                    input_y,
                    mean,
                    variance,
                    begin_norm_axis,
                    begin_params_axis,
                    epsilon,
                    kernel_name="LayerNormalize",
                    atomic_clean_diff_shape=True):
    """
    op layer_norm
    """
    tik_container = AContainer()
    tik_container.reset_instance()
    cont = tik_container.get_instance()

    data_type = input_x.get("dtype").lower()
    ub_type = 'float32'

    elementwise = True
    output_mean_var = True
    input_output_tensor_dict = {}
    # get input_mode
    input_mode = _check_input_mode(input_x)
    y_atomic_add = (ub_type == data_type and atomic_clean_diff_shape)
    if input_mode == Constant.DYNAMIC:
        tiling_ub_list = tiling_init_dynamic(cont)
        input_shape = (Constant.MAX_INT32_SIZE,)
        input_x = cont.tinst.Tensor(data_type, input_shape, cont.tik.scope_gm, "x")
        input_x = input_x.reshape((Constant.MAX_INT32_SIZE // tiling_ub_list[3], tiling_ub_list[3]))
        input_y = cont.tinst.Tensor(data_type, input_shape, cont.tik.scope_gm, "y", is_atomic_add=y_atomic_add)
        input_y = input_y.reshape((Constant.MAX_INT32_SIZE // tiling_ub_list[3], tiling_ub_list[3]))
        data_shape = (1, Constant.MAX_INT32_SIZE)
        batch_shape = (Constant.MAX_INT32_SIZE,)
        batch_shape_align = batch_shape
        ai_core_num = cont.const_aicore_num
    else:
        tiling_ub_list = _gen_tiling_case_const(input_x, gamma, beta, input_y)
        input_shape = (tiling_ub_list[2], tiling_ub_list[3])
        input_x = cont.tinst.Tensor(data_type, input_shape, cont.tik.scope_gm, "x")
        input_y = cont.tinst.Tensor(data_type, input_shape, cont.tik.scope_gm, "y", is_atomic_add=y_atomic_add)
        data_shape = (1, tiling_ub_list[3])
        batch_shape = (tiling_ub_list[2],)
        batch_shape_align = (tiling_ub_list[2] + cont.get_vec_proc_num_per_cmd_blk(data_type),)
        ai_core_num = tiling_ub_list[4]
    tiling_mode = tiling_ub_list[1]

    input_output_tensor_dict["input_x"] = input_x
    if elementwise:
        gamma = cont.tinst.Tensor(data_type, data_shape, cont.tik.scope_gm, "gamma")
        beta = cont.tinst.Tensor(data_type, data_shape, cont.tik.scope_gm, "beta")
        input_output_tensor_dict["gamma"] = gamma
        input_output_tensor_dict["beta"] = beta
    input_output_tensor_dict["input_y"] = input_y
    if output_mean_var:
        mv_atomic_add = (ub_type == data_type)
        mean = cont.tinst.Tensor(data_type, batch_shape, cont.tik.scope_gm, "mean", is_atomic_add=mv_atomic_add)
        variance = cont.tinst.Tensor(data_type, batch_shape, cont.tik.scope_gm, "variance", is_atomic_add=mv_atomic_add)
        input_output_tensor_dict["mean"] = mean
        input_output_tensor_dict["variance"] = variance
        if data_type != ub_type:
            y_ub_type = None
            if atomic_clean_diff_shape:
                y_ub_type = cont.tinst.Tensor(ub_type,
                                              input_shape,
                                              cont.tik.scope_gm,
                                              "y_ub_type",
                                              is_workspace=True,
                                              is_atomic_add=True)
            if input_mode == Constant.DYNAMIC and atomic_clean_diff_shape:
                y_ub_type = y_ub_type.reshape((Constant.MAX_INT32_SIZE // tiling_ub_list[3], tiling_ub_list[3]))
            mean_ub_type = cont.tinst.Tensor(ub_type,
                                             batch_shape_align,
                                             cont.tik.scope_gm,
                                             "mean_ub_type",
                                             is_workspace=True,
                                             is_atomic_add=True)
            variance_ub_type = cont.tinst.Tensor(ub_type,
                                                 batch_shape_align,
                                                 cont.tik.scope_gm,
                                                 "variance_ub_type",
                                                 is_workspace=True,
                                                 is_atomic_add=True)
            input_output_tensor_dict["mean_ub_type"] = mean_ub_type
            input_output_tensor_dict["variance_ub_type"] = variance_ub_type
            input_output_tensor_dict["y_ub_type"] = y_ub_type

    sync_workspace = None

    if data_type != ub_type:
        sync_type = "int64"
        sync_block_data_num = cont.get_vec_proc_num_per_cmd_blk(sync_type)
        sync_data_num = ai_core_num * sync_block_data_num
        sync_workspace = cont.tinst.Tensor(sync_type, (sync_data_num,),
                                           name="gm_barrier",
                                           scope=cont.tik.scope_gm,
                                           is_workspace=True,
                                           is_atomic_add=True)

    n_max_data_num = max_data_num(cont, data_type, ub_type)
    act_core_use = tiling_ub_list[4]
    with cont.tinst.for_range(0, ai_core_num, block_num=ai_core_num) as core_index:
        if input_mode == Constant.DYNAMIC:
            with cont.tinst.if_scope(tiling_mode == Constant.SELECT_KEY_MODE_SPLIT_N and core_index < act_core_use):
                obj_layer_normalize = LayerNormalizeSplitN(tiling_ub_list, epsilon, kernel_name, cont, data_type,
                                                           elementwise, output_mean_var, input_output_tensor_dict,
                                                           core_index, sync_workspace, n_max_data_num, input_mode,
                                                           atomic_clean_diff_shape)
                obj_layer_normalize.mode_compute()
            with cont.tinst.elif_scope(tiling_mode == Constant.SELECT_KEY_MODE_SPLIT_D and core_index < act_core_use):
                obj_layer_normalize = LayerNormalizeSplitD(tiling_ub_list, epsilon, kernel_name, cont, data_type,
                                                           elementwise, output_mean_var, input_output_tensor_dict,
                                                           core_index, sync_workspace, n_max_data_num, input_mode,
                                                           atomic_clean_diff_shape)
                obj_layer_normalize.mode_compute()
        else:
            if tiling_mode == Constant.SELECT_KEY_MODE_SPLIT_N:
                obj_layer_normalize = LayerNormalizeSplitN(tiling_ub_list, epsilon, kernel_name, cont, data_type,
                                                           elementwise, output_mean_var, input_output_tensor_dict,
                                                           core_index, sync_workspace, n_max_data_num, input_mode,
                                                           atomic_clean_diff_shape)
            else:
                obj_layer_normalize = LayerNormalizeSplitD(tiling_ub_list, epsilon, kernel_name, cont, data_type,
                                                           elementwise, output_mean_var, input_output_tensor_dict,
                                                           core_index, sync_workspace, n_max_data_num, input_mode,
                                                           atomic_clean_diff_shape)
            obj_layer_normalize.mode_compute()
        obj_layer_normalize._mean_var_move_out_base()
        obj_layer_normalize._y_workspace_move_out()
    tik_instance = obj_layer_normalize.build_cce()
    return tik_instance
