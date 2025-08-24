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
repeat_interleave
"""

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator


class Constant:
    """
    The class for constant
    """
    # max int32
    MAX_INT32 = 2 ** 31 - 1
    TILING_ARG_NUM = 20
    BLOCK_BYTE_SIZE = 32
    REPEATS_NUM = 512
    INT64_BYTE_SIZE = 8
    # soc version supports data_move_pad (910b/910_93) or not
    SOC_VERSION_DATA_MOVE_PAD = 1
    SOC_VERSION_OTHER = 0
    SUPPORT_VECTOR_DUP = 11
    NOT_SUPPORT_VECTOR_DUP = 10


class RepeatInterleave():
    """
    Function: use to finish RepeatInterleave main functions
    """

    def __init__(self, x, repeats, y, axis, kernel_name):
        self.tik_instance = tik.Tik()
        self.x_dtype = x.get("dtype").lower()
        self.y_dtype = y.get("dtype").lower()
        self.repeats_dtype = repeats.get("dtype").lower()

        # check dtype
        para_check.check_dtype(
            self.x_dtype,
            ("float16", "float32", "int8", "int32", "int64", "uint8", "int16", "uint16", "uint32", "uint64",
             "bool", "bfloat16"),
            param_name="x")
        para_check.check_dtype(
            self.repeats_dtype, ("int32", "int64"), param_name="repeats")
        para_check.check_dtype(
            self.y_dtype,
            ("float16", "float32", "int8", "int32", "int64", "uint8", "int16", "uint16", "uint32", "uint64",
             "bool", "bfloat16"),
            param_name="y")
        if self.x_dtype != self.y_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal("RepeatInterleave", "x", "y", self.x_dtype,
                                                                  self.y_dtype)
        if self.x_dtype == "bool":
            self.x_dtype = "int8"
        if self.x_dtype == "bfloat16":
            self.x_dtype = "float16"
        if self.y_dtype == "bool":
            self.y_dtype = "int8"
        if self.y_dtype == "bfloat16":
            self.y_dtype = "float16"

        self.kernel_name = kernel_name
        self.full_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        # reserved ub size 8KB
        reserved_ub_size = 8 * 1024
        self.ub_available = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - reserved_ub_size
        self.x_dtype_size = tbe_platform.get_bit_len(self.x_dtype) // 8
        self.repeats_dtype_size = tbe_platform.get_bit_len(self.repeats_dtype) // 8
        self.x_data_one_block = Constant.BLOCK_BYTE_SIZE // self.x_dtype_size
        self.repeats_data_one_block = Constant.BLOCK_BYTE_SIZE // self.repeats_dtype_size

        self.init_gm_tensor()
        self.init_tiling_scalar()
        self.repeats_num = self.tik_instance.Scalar("int64", name="repeats_num")
        self.tiling_ub = None
        self.x_ub = None
        self.repeats_ub = None
        self.tmp_ub = None
        self.previous_block_ub = None
        self.dup_ub = None
        self.repeats_scalar = None
        self.repeats_sum_ub = None
        self.repeats_ub_scalar = None
        self.repeats_sum_ub_scalar = None

        self.support_data_move_pad = tbe_platform.api_check_support("tik.data_move_pad", "int8") or \
                                     tbe_platform.api_check_support("tik.data_move_pad", self.x_dtype)
        self.support_vector_dup = tbe_platform.api_check_support("tik.vector_dup", self.x_dtype)
        if self.support_data_move_pad:
            self.soc_version = Constant.SOC_VERSION_DATA_MOVE_PAD
        else:
            self.soc_version = Constant.SOC_VERSION_OTHER
        if self.support_vector_dup:
            self.dtype_support_vector_dup = Constant.SUPPORT_VECTOR_DUP
        else:
            self.dtype_support_vector_dup = Constant.NOT_SUPPORT_VECTOR_DUP

    def init_tiling_scalar(self):
        self.repeats_i_dim_num = self.tik_instance.Scalar("int64", name="repeats_i_dim_num")
        self.repeats_o_dim_num = self.tik_instance.Scalar("int64", name="repeats_o_dim_num")
        self.is_last_axis = self.tik_instance.Scalar("int64", name="is_last_axis")
        self.batch_dim_num = self.tik_instance.Scalar("int64", name="batch_dim_num")
        self.data_dim_num = self.tik_instance.Scalar("int64", name="data_dim_num")
        self.data_num = self.tik_instance.Scalar("int64", name="data_num")
        self.batch_dim_num_each_core = self.tik_instance.Scalar("int64", name="batch_dim_num_each_core")
        self.batch_dim_num_last_core = self.tik_instance.Scalar("int64", name="batch_dim_num_last_core")
        self.core_num_each_batch = self.tik_instance.Scalar("int64", name="core_num_each_batch")
        self.element_num_each_core = self.tik_instance.Scalar("int64", name="element_num_each_core")
        self.element_num_last_core = self.tik_instance.Scalar("int64", name="element_num_last_core")
        self.core_num = self.tik_instance.Scalar("int64", name="core_num")
        self.element_num_each_loop = self.tik_instance.Scalar("int64", name="element_num_each_loop")
        self.init_repeats_ub = self.tik_instance.Scalar("int64", name="init_repeats_ub")
        self.front_core_num = self.tik_instance.Scalar("int64", name="front_core_num")
        self.tail_core_num = self.tik_instance.Scalar("int64", name="tail_core_num")
        self.front_core_move_data = self.tik_instance.Scalar("int64", name="front_core_move_data")
        self.tail_core_move_data = self.tik_instance.Scalar("int64", name="tail_core_move_data")

    def init_gm_tensor(self):
        """init_gm_tensor
        """
        self.tiling_gm = self.tik_instance.Tensor(
            "int64", (Constant.TILING_ARG_NUM,), name="tiling_gm", scope=tik.scope_gm)
        self.x_gm = self.tik_instance.Tensor(
            self.x_dtype, (Constant.MAX_INT32,), name="x", scope=tik.scope_gm)
        self.repeats_gm = self.tik_instance.Tensor(
            self.repeats_dtype, (Constant.MAX_INT32,), name="repeats", scope=tik.scope_gm)
        self.y_gm = self.tik_instance.Tensor(
            self.y_dtype, (Constant.MAX_INT32,), name="y", scope=tik.scope_gm)

    def repeat_interleave_compute(self):
        """RepeatInterleave compute
        """
        self._tiling_args()

        with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as core_index:
            self.repeats_ub = self.tik_instance.Tensor(self.repeats_dtype, (self.init_repeats_ub,),
                                                name="repeats_ub", scope=tik.scope_ubuf)
            self.tmp_ub = self.tik_instance.Tensor(self.x_dtype, (self.x_data_one_block,),
                                                name="tmp_ub", scope=tik.scope_ubuf)
            self.previous_block_ub = self.tik_instance.Tensor(self.x_dtype, (self.x_data_one_block,),
                                                name="previous_block_ub", scope=tik.scope_ubuf)
            self.dup_ub = self.tik_instance.Tensor(self.x_dtype, (512,), name="dup_ub", scope=tik.scope_ubuf)
            self.calculate_each_core(core_index)

        self._add_compile_info()
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.x_gm, self.repeats_gm],
                                   outputs=[self.y_gm],
                                   flowtable=[self.tiling_gm],
                                   config=opt_config)
        return self.tik_instance

    # 'pylint: disable=huawei-too-many-arguments
    def data_move_pad_bit64_2_int8(self, data_dtype, dst, src, nburst, burst, dst_gap, src_gap,
                                   right_padding=0, left_padding=0, padding_value=None):
        if self.support_data_move_pad:
            if tbe_platform.api_check_support("tik.data_move_pad", data_dtype):
                self.tik_instance.data_move_pad(dst, src, nburst, burst, dst_gap,
                                                src_gap, right_padding, left_padding, padding_value)
            else:
                dst_int8 = dst.reinterpret_cast_to("int8")
                src_int8 = src.reinterpret_cast_to("int8")
                self.tik_instance.data_move_pad(dst_int8, src_int8, nburst, burst, dst_gap,
                                                src_gap, right_padding, left_padding, padding_value)

    def compute_repeats_sum(self):
        self.data_move_pad_bit64_2_int8(self.repeats_dtype,
                                        self.repeats_ub, self.repeats_gm,
                                        1, self.repeats_dtype_size * self.repeats_i_dim_num,
                                        0, 0)
        self.repeats_scalar = self.tik_instance.Scalar("int64", name="repeats_scalar")
        self.repeats_scalar.set_as(self.repeats_ub[0])
        self.repeats_sum_ub = self.tik_instance.Tensor(
                                            self.repeats_dtype, (self.init_repeats_ub + self.repeats_data_one_block,),
                                            name="repeats_sum_ub", scope=tik.scope_ubuf)
        self.repeats_sum_ub[0].set_as(0)
        self.repeats_ub_scalar = self.tik_instance.Scalar(self.repeats_dtype, name="repeats_ub_scalar")
        self.repeats_sum_ub_scalar = self.tik_instance.Scalar(self.repeats_dtype, name="repeats_sum_ub_scalar")
        with self.tik_instance.for_range(1, self.repeats_i_dim_num + 1) as i:
            self.repeats_ub_scalar.set_as(self.repeats_ub[i - 1])
            self.repeats_sum_ub_scalar.set_as(self.repeats_sum_ub[i - 1])
            self.repeats_sum_ub[i].set_as(self.repeats_ub_scalar + self.repeats_sum_ub_scalar)

    def calculate_each_core(self, core_index):
        batch_dim_num_core = self.tik_instance.Scalar("int64", name="batch_dim_num_core")
        with self.tik_instance.if_scope(core_index < (self.core_num - 1)):
            batch_dim_num_core.set_as(self.batch_dim_num_each_core)
        with self.tik_instance.else_scope():
            batch_dim_num_core.set_as(self.batch_dim_num_last_core)

        batch_index_start_core = self.tik_instance.Scalar("int64", name="batch_index_start_core",
                                init_value=core_index // self.core_num_each_batch * self.batch_dim_num_each_core)
        loop_offset_start_core = self.tik_instance.Scalar("int64", name="loop_offset_start_core",
                                init_value=core_index % self.core_num_each_batch * self.element_num_each_core)

        element_num = self.tik_instance.Scalar("int64", name="element_num")
        with self.tik_instance.if_scope((core_index % self.core_num_each_batch) == (self.core_num_each_batch - 1)):
            element_num.set_as(self.element_num_last_core)
        with self.tik_instance.else_scope():
            element_num.set_as(self.element_num_each_core)

        loop_times = self.tik_instance.Scalar("int64", name="loop_times",
                            init_value=(element_num + self.element_num_each_loop - 1) // self.element_num_each_loop)
        element_num_last_loop = self.tik_instance.Scalar("int64", name="element_num_last_loop",
                            init_value=element_num - self.element_num_each_loop * (loop_times - 1))

        with self.tik_instance.if_scope(self.repeats_i_dim_num == 1):
            if self.support_data_move_pad:
                self.data_move_pad_bit64_2_int8(self.repeats_dtype,
                                                self.repeats_ub, self.repeats_gm,
                                                1, self.repeats_dtype_size,
                                                0, 0)
            else:
                self.tik_instance.data_move(self.repeats_ub, self.repeats_gm, 0, 1, 1, 0, 0)
            self.repeats_num.set_as(self.repeats_ub[0])

        # repeat为int, input data aligned
        with self.tik_instance.if_scope(
                    tik.all((self.repeats_i_dim_num == 1),
                            (self.core_num_each_batch == 1),
                            (element_num_last_loop % self.x_data_one_block == 0),
                            (loop_times == 1),
                            (self.repeats_num > 0),
                            ((self.repeats_num - 1) * self.data_dim_num // self.x_data_one_block <= 65535))):
            self.compute_repeat_int_align(core_index, batch_dim_num_core, element_num_last_loop, batch_index_start_core)

        # one-to-one mapping between input and repeats, input dtype support vector_dup, 910B/C
        with self.tik_instance.if_scope(
                    tik.all((self.data_num == self.repeats_i_dim_num),
                            (self.repeats_i_dim_num + 1 <= self.ub_available // self.repeats_dtype_size // 2),
                            (self.soc_version == 1),
                            (self.support_vector_dup),
                            (self.data_num != 0))):
            self.compute_repeats_tensor(core_index)

        with self.tik_instance.else_scope():
            self.x_ub = self.tik_instance.Tensor(self.x_dtype, (self.element_num_each_loop,),
                                                 name="x_ub", scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, batch_dim_num_core) as batch_dim_index:
                self.calculate_each_batch(batch_dim_index, batch_index_start_core, loop_offset_start_core,
                                        loop_times, element_num_last_loop)

    def repeats_value_less_512(self, repeats_value, max_vector_dup_repeat_times, output_offset):
        max_mask_tail_num = self.tik_instance.Scalar("int64", name="max_mask_tail_num",
                                                init_value=repeats_value % Constant.REPEATS_NUM)
        block_mask_repeat_times = self.tik_instance.Scalar(self.repeats_dtype, name="block_mask_repeat_times",
                                    init_value=max_mask_tail_num // self.x_data_one_block)
        output_offset.set_as(self.repeats_sum_ub_scalar + max_vector_dup_repeat_times * Constant.REPEATS_NUM)
        with self.tik_instance.if_scope(block_mask_repeat_times != 0):
            self.tik_instance.data_move(self.y_gm[output_offset], 
                                        self.dup_ub,
                                        0, 1, 
                                        block_mask_repeat_times, 0, 0)

        with self.tik_instance.if_scope(repeats_value % self.x_data_one_block != 0):
            output_offset.set_as(output_offset + block_mask_repeat_times * self.x_data_one_block)
            self.data_move_pad_bit64_2_int8(self.x_dtype,
                                            self.y_gm[output_offset], self.dup_ub,
                                            1, self.x_dtype_size * (repeats_value % self.x_data_one_block),
                                            0, 0)

    # 'pylint: disable=huawei-too-many-arguments
    def compute_repeats_tensor_tail_core(self, core_index, x_value, repeats_value, max_vector_dup_repeat_times,
                                         max_mask, max_mask_repeat_times, output_offset):
        with self.tik_instance.for_range(0, self.tail_core_move_data) as repeat_index:
            self.data_move_pad_bit64_2_int8(
                                    self.x_dtype,
                                    self.x_ub,
                                    self.x_gm[self.front_core_num * self.front_core_move_data + \
                                              (core_index - self.front_core_num) * self.tail_core_move_data + \
                                              repeat_index],
                                    1, self.x_dtype_size,
                                    0, 0)
            x_value.set_as(self.x_ub[0])

            self.data_move_pad_bit64_2_int8(
                                    self.repeats_dtype,
                                    self.repeats_ub,
                                    self.repeats_gm[self.front_core_num * self.front_core_move_data + \
                                                    (core_index - self.front_core_num) * self.tail_core_move_data + \
                                                    repeat_index],
                                    1, self.repeats_dtype_size,
                                    0, 0)
            repeats_value.set_as(self.repeats_ub[0])

            max_vector_dup_repeat_times.set_as(repeats_value // Constant.REPEATS_NUM)
            self.tik_instance.vector_dup(max_mask, self.dup_ub, x_value, max_mask_repeat_times, 1, 8)         
            self.repeats_sum_ub_scalar.set_as(
                self.repeats_sum_ub[self.front_core_num * self.front_core_move_data + \
                                    (core_index - self.front_core_num) * self.tail_core_move_data + repeat_index])
            # self.dup_ub -> y_gm
            with self.tik_instance.for_range(0, max_vector_dup_repeat_times) as max_vector_dup_repeat_index:
                output_offset.set_as(self.repeats_sum_ub_scalar + max_vector_dup_repeat_index * Constant.REPEATS_NUM)
                self.tik_instance.data_move(self.y_gm[output_offset], 
                                            self.dup_ub, 
                                            0, 1,
                                            Constant.REPEATS_NUM // self.x_data_one_block, 0, 0)
            self.repeats_value_less_512(repeats_value, max_vector_dup_repeat_times, output_offset)

    # 'pylint: disable=huawei-too-many-arguments
    def repeats_tensor_each_core(self, core_index, x_value, repeats_value, max_vector_dup_repeat_times, 
                                         max_mask, max_mask_repeat_times, output_offset):
        with self.tik_instance.if_scope(core_index < self.front_core_num):
            with self.tik_instance.for_range(0, self.front_core_move_data) as repeat_index:
                self.data_move_pad_bit64_2_int8(self.x_dtype,
                                                self.x_ub,
                                                self.x_gm[core_index * self.front_core_move_data + repeat_index],
                                                1, self.x_dtype_size,
                                                0, 0)
                x_value.set_as(self.x_ub[0])

                self.data_move_pad_bit64_2_int8(self.repeats_dtype,
                                                self.repeats_ub,
                                                self.repeats_gm[core_index * self.front_core_move_data + repeat_index],
                                                1, self.repeats_dtype_size,
                                                0, 0)
                repeats_value.set_as(self.repeats_ub[0])

                max_vector_dup_repeat_times.set_as(repeats_value // Constant.REPEATS_NUM)
                self.tik_instance.vector_dup(max_mask, self.dup_ub, x_value, max_mask_repeat_times, 1, 8)
                self.repeats_sum_ub_scalar.set_as(
                                        self.repeats_sum_ub[core_index * self.front_core_move_data + repeat_index])
                # self.dup_ub -> y_gm
                with self.tik_instance.for_range(0, max_vector_dup_repeat_times) as max_vector_dup_repeat_index:
                    output_offset.set_as(self.repeats_sum_ub_scalar + \
                                            max_vector_dup_repeat_index * Constant.REPEATS_NUM)
                    self.tik_instance.data_move(self.y_gm[output_offset], 
                                                self.dup_ub, 0, 1,
                                                Constant.REPEATS_NUM // self.x_data_one_block, 0, 0)
                self.repeats_value_less_512(repeats_value, max_vector_dup_repeat_times, output_offset)

        with self.tik_instance.else_scope():
            self.compute_repeats_tensor_tail_core(core_index, x_value, repeats_value, max_vector_dup_repeat_times,
                                        max_mask, max_mask_repeat_times, output_offset)

    def compute_repeats_tensor(self, core_index):
        if self.support_vector_dup:
            self.compute_repeats_sum()
            self.x_ub = self.tik_instance.Tensor(self.x_dtype, (self.x_data_one_block,), 
                                                 name="x_ub", scope=tik.scope_ubuf)
            x_value = self.tik_instance.Scalar(dtype=self.x_dtype, name="x_value", init_value=0)
            repeats_value = self.tik_instance.Scalar(dtype=self.repeats_dtype, name="repeats_value", init_value=0)
            # mask  fp32->64   fp16->128
            max_mask = 8 * Constant.BLOCK_BYTE_SIZE // self.x_dtype_size   

            max_mask_repeat_times = self.tik_instance.Scalar("int64", name="max_mask_repeat_times",
                                                            init_value=Constant.REPEATS_NUM // max_mask)
            max_vector_dup_repeat_times = self.tik_instance.Scalar("int64", name="max_vector_dup_repeat_times")
            output_offset = self.tik_instance.Scalar("int64", name="output_offset")
            self.repeats_tensor_each_core(core_index, x_value, repeats_value, max_vector_dup_repeat_times, 
                                                  max_mask, max_mask_repeat_times, output_offset)

    def compute_repeat_int_align(self, core_index, batch_dim_num_core, element_num_last_loop, batch_index_start_core):
        # repeats is Scalar; input data is aligned and less than ub size
        element_num_loop = self.tik_instance.Scalar("int64", name="element_num_loop")
        element_num_loop.set_as(element_num_last_loop)
        each_loop_batch_num = self.tik_instance.Scalar("int64", name="each_loop_batch_num",
                init_value=self.ub_available // Constant.BLOCK_BYTE_SIZE * self.x_data_one_block // element_num_loop)
        batch_loop_times = self.tik_instance.Scalar("int64", name="batch_loop_times",
                                                    init_value=batch_dim_num_core // each_loop_batch_num)
        each_loop_batch_num_tail = self.tik_instance.Scalar("int64", name="each_loop_batch_num_tail",
                                                init_value=batch_dim_num_core - batch_loop_times * each_loop_batch_num)

        self.x_ub = self.tik_instance.Tensor(self.x_dtype, (each_loop_batch_num * self.element_num_each_loop,),
                                             name="x_ub", scope=tik.scope_ubuf)
        with self.tik_instance.for_range(0, batch_loop_times) as batch_loop_times_index:
            batch_dim_index_gm_0 = self.tik_instance.Scalar("int64", name="batch_dim_index_gm_0",
                                init_value=batch_index_start_core + batch_loop_times_index * each_loop_batch_num)
            input_offset_gm_0 = self.tik_instance.Scalar("int64", name="input_offset_gm",
                                init_value=batch_dim_index_gm_0 * self.repeats_i_dim_num * self.data_dim_num)

            self.tik_instance.data_move(self.x_ub, self.x_gm[input_offset_gm_0], 0, 1,
                                        each_loop_batch_num * element_num_loop // self.x_data_one_block, 0, 0)
            with self.tik_instance.for_range(0, self.repeats_num) as repeats_num_index:
                self.tik_instance.data_move(self.y_gm[core_index * self.batch_dim_num_each_core *
                                                      self.data_dim_num * self.repeats_num + \
                                                      batch_loop_times_index * each_loop_batch_num *
                                                      self.data_dim_num * self.repeats_num + \
                                                      self.data_dim_num * repeats_num_index],
                                            self.x_ub,
                                            0, each_loop_batch_num,
                                            element_num_loop // self.x_data_one_block,
                                            0, (self.repeats_num - 1) * self.data_dim_num // self.x_data_one_block)

        with self.tik_instance.if_scope(each_loop_batch_num_tail != 0):
            input_offset_tail_gm = self.tik_instance.Scalar("int64", name="input_offset_tail_gm",
                                        init_value=(batch_index_start_core + batch_loop_times * each_loop_batch_num) *
                                                    self.repeats_i_dim_num * self.data_dim_num)
            self.tik_instance.data_move(self.x_ub,
                                        self.x_gm[input_offset_tail_gm],
                                        0, 1,
                                        each_loop_batch_num_tail * element_num_loop // self.x_data_one_block,
                                        0, 0)
            with self.tik_instance.for_range(0, self.repeats_num) as repeats_num_index:
                self.tik_instance.data_move(self.y_gm[core_index * self.batch_dim_num_each_core *
                                                      self.data_dim_num * self.repeats_num + \
                                                      batch_loop_times * each_loop_batch_num *
                                                      self.data_dim_num * self.repeats_num + \
                                                      self.data_dim_num * repeats_num_index],
                                            self.x_ub,
                                            0, each_loop_batch_num_tail,
                                            element_num_loop // self.x_data_one_block,
                                            0, (self.repeats_num - 1) * self.data_dim_num // self.x_data_one_block)

    def calculate_each_batch(self, batch_dim_index, batch_index_start_core, loop_offset_start_core,
                             loop_times, element_num_last_loop):
        batch_dim_index_gm = self.tik_instance.Scalar("int64", name="batch_dim_index_gm",
                                                      init_value=batch_index_start_core + batch_dim_index)
        input_repeat_offset_gm = self.tik_instance.Scalar("int64", name="input_repeat_offset_gm",
                    init_value=batch_dim_index_gm * self.repeats_o_dim_num * self.data_dim_num + loop_offset_start_core)

        with self.tik_instance.for_range(0, self.repeats_i_dim_num) as repeats_index:
            with self.tik_instance.if_scope(self.repeats_i_dim_num != 1):
                if self.support_data_move_pad:
                    self.data_move_pad_bit64_2_int8(self.repeats_dtype, self.repeats_ub, self.repeats_gm[repeats_index],
                                                    1, self.repeats_dtype_size, 0, 0)
                else:
                    self.tik_instance.data_move(self.repeats_ub, self.repeats_gm[repeats_index], 0, 1, 1, 0, 0)
                self.repeats_num.set_as(self.repeats_ub[0])

            with self.tik_instance.for_range(0, loop_times) as loop_index:
                self.calculate_each_loop(repeats_index, loop_index, loop_times, element_num_last_loop,
                                         batch_dim_index_gm, input_repeat_offset_gm, loop_offset_start_core)

            input_repeat_offset_gm.set_as(input_repeat_offset_gm + self.repeats_num * self.data_dim_num)

    # 'pylint: disable=huawei-too-many-arguments
    def calculate_each_loop(self, repeats_index, loop_index, loop_times, element_num_last_loop, batch_dim_index_gm,
                            input_repeat_offset_gm, loop_offset_start_core):
        element_num_loop = self.tik_instance.Scalar("int64", name="element_num_loop")
        with self.tik_instance.if_scope(loop_index < (loop_times - 1)):
            element_num_loop.set_as(self.element_num_each_loop)
        with self.tik_instance.else_scope():
            element_num_loop.set_as(element_num_last_loop)

        input_offset_gm = self.tik_instance.Scalar("int64", name="input_offset_gm",
                            init_value=batch_dim_index_gm * self.repeats_i_dim_num * self.data_dim_num + \
                                       repeats_index * self.data_dim_num + loop_offset_start_core + \
                                       loop_index * self.element_num_each_loop)
        output_offset_gm = self.tik_instance.Scalar("int64", name="output_offset_gm")
        if self.support_data_move_pad:
            self.data_move_pad_bit64_2_int8(self.x_dtype, self.x_ub, self.x_gm[input_offset_gm],
                                            1, self.x_dtype_size * element_num_loop, 0, 0)
        else:
            self.tik_instance.data_move(self.x_ub, self.x_gm[input_offset_gm], 0, 1,
                                        (element_num_loop + self.x_data_one_block - 1) // self.x_data_one_block, 0, 0)

        less_one_block_tail_data = element_num_loop % self.x_data_one_block

        with self.tik_instance.if_scope(self.data_dim_num < self.x_data_one_block):
            self.small_data_dim_num(batch_dim_index_gm, loop_index, element_num_loop,
                                    input_repeat_offset_gm, output_offset_gm)
        with self.tik_instance.elif_scope(less_one_block_tail_data == 0):
            self.large_data_dim_num_align(loop_index, element_num_loop, input_repeat_offset_gm, output_offset_gm)
        with self.tik_instance.else_scope():
            self.large_data_dim_num_not_align(repeats_index, loop_index, batch_dim_index_gm, element_num_loop,
                                              input_repeat_offset_gm, output_offset_gm,
                                              less_one_block_tail_data, loop_offset_start_core)

    def small_data_dim_num(self, batch_dim_index_gm, loop_index, element_num_loop,
                           input_repeat_offset_gm, output_offset_gm):
        if self.x_dtype in ["uint32", "int32", "float32", "uint16", "int16", "float16", "bfloat16"]:
            with self.tik_instance.if_scope(tik.all((self.repeats_i_dim_num == 1),
                                                    (self.is_last_axis == 1),
                                                    (self.repeats_num >= self.x_data_one_block))):
                self.small_data_dim_num_high_performance(batch_dim_index_gm, input_repeat_offset_gm, output_offset_gm)

            with self.tik_instance.else_scope():
                self.small_data_dim_num_cal(loop_index, element_num_loop, input_repeat_offset_gm, output_offset_gm)

        else:
            self.small_data_dim_num_cal(loop_index, element_num_loop, input_repeat_offset_gm, output_offset_gm)

    def small_data_dim_num_cal(self, loop_index, element_num_loop, input_repeat_offset_gm, output_offset_gm):
        with self.tik_instance.for_range(0, self.repeats_num) as repeats_num_index:
            output_offset_gm.set_as(input_repeat_offset_gm + loop_index * self.element_num_each_loop + \
                                    repeats_num_index * self.data_dim_num)
            if self.support_data_move_pad:
                self.data_move_pad_bit64_2_int8(self.x_dtype, self.y_gm[output_offset_gm], self.x_ub,
                                                1, self.x_dtype_size * element_num_loop, 0, 0)
            else:
                self.tik_instance.data_move(self.y_gm[output_offset_gm],
                                            self.x_ub,
                                            0, 1,
                                            (element_num_loop + self.x_data_one_block - 1) // self.x_data_one_block,
                                            0, 0)

    def small_data_dim_num_high_performance(self, batch_dim_index_gm, input_repeat_offset_gm, output_offset_gm):
        x_value = self.tik_instance.Scalar(dtype=self.x_dtype, name="x_value", init_value=0)
        x_value.set_as(self.x_ub[0])
        # fp32->64   fp16->128
        max_mask = 8 * Constant.BLOCK_BYTE_SIZE // self.x_dtype_size
        max_vector_dup_repeat_times = self.tik_instance.Scalar(dtype=self.repeats_dtype,
                                      name="max_vector_dup_repeat_times",
                                      init_value=(self.repeats_num) // Constant.REPEATS_NUM)
        max_mask_repeat_times = self.tik_instance.Scalar(dtype=self.repeats_dtype, name="max_mask_repeat_times",
                                                         init_value=Constant.REPEATS_NUM // max_mask)

        self.tik_instance.vector_dup(max_mask, self.dup_ub, x_value, max_mask_repeat_times, 1, 8)
        with self.tik_instance.for_range(0, max_vector_dup_repeat_times) as max_vector_dup_repeat_index:
            output_offset_gm.set_as(input_repeat_offset_gm + max_vector_dup_repeat_index * Constant.REPEATS_NUM)
            self.tik_instance.data_move(self.y_gm[output_offset_gm],
                                        self.dup_ub, 0, 1,
                                        Constant.REPEATS_NUM // self.x_data_one_block, 0, 0)

        max_mask_tail_num = self.tik_instance.Scalar(dtype=self.repeats_dtype, name="max_mask_tail_num",
                                                     init_value=self.repeats_num % Constant.REPEATS_NUM)
        block_mask_repeat_times = self.tik_instance.Scalar(dtype=self.repeats_dtype, name="block_mask_repeat_times",
                                                           init_value=max_mask_tail_num // self.x_data_one_block)

        output_offset_gm.set_as(input_repeat_offset_gm + max_vector_dup_repeat_times * Constant.REPEATS_NUM)
        with self.tik_instance.if_scope(block_mask_repeat_times != 0):
            self.tik_instance.data_move(self.y_gm[output_offset_gm], self.dup_ub, 0, 1, block_mask_repeat_times, 0, 0)
        with self.tik_instance.if_scope(self.repeats_num % self.x_data_one_block != 0):
            self.tik_instance.data_move(self.y_gm[batch_dim_index_gm * self.repeats_num + \
                                                self.repeats_num - self.x_data_one_block],
                                        self.dup_ub, 0, 1,
                                        1, 0, 0)

    def large_data_dim_num_align(self, loop_index, element_num_loop, input_repeat_offset_gm, output_offset_gm):
        with self.tik_instance.for_range(0, self.repeats_num) as repeats_num_index:
            output_offset_gm.set_as(input_repeat_offset_gm + loop_index * self.element_num_each_loop + \
                                    repeats_num_index * self.data_dim_num)
            self.tik_instance.data_move(self.y_gm[output_offset_gm], self.x_ub, 0, 1,
                                        (element_num_loop + self.x_data_one_block - 1) // self.x_data_one_block, 0, 0)

    # 'pylint: disable=huawei-too-many-arguments
    def large_data_dim_num_not_align(self, repeats_index, loop_index, batch_dim_index_gm, element_num_loop,
                                     input_repeat_offset_gm, output_offset_gm, less_one_block_tail_data,
                                     loop_offset_start_core):
        block_loop_times = self.tik_instance.Scalar("int64", name="input_offset_gm",
                                                    init_value=element_num_loop // self.x_data_one_block)
        with self.tik_instance.for_range(0, less_one_block_tail_data) as data_index:
            self.tmp_ub[self.x_data_one_block - (less_one_block_tail_data - data_index)].set_as(
                        self.x_ub[block_loop_times * self.x_data_one_block + data_index])

        with self.tik_instance.if_scope(block_loop_times >= 1):
            with self.tik_instance.for_range(0, self.x_data_one_block - less_one_block_tail_data) as data_index:
                self.tmp_ub[data_index].set_as(
                    self.x_ub[(block_loop_times - 1) * self.x_data_one_block + less_one_block_tail_data + data_index])
        with self.tik_instance.else_scope():
            if self.support_data_move_pad:
                self.data_move_pad_bit64_2_int8(
                                        self.x_dtype,
                                        self.previous_block_ub,
                                        self.x_gm[batch_dim_index_gm * self.repeats_i_dim_num * self.data_dim_num + \
                                                  repeats_index * self.data_dim_num + loop_offset_start_core + \
                                                  loop_index * self.element_num_each_loop - \
                                                  self.x_data_one_block + less_one_block_tail_data],
                                        1,
                                        self.x_dtype_size * (self.x_data_one_block - less_one_block_tail_data),
                                        0, 0)
            else:
                self.tik_instance.data_move(
                                        self.previous_block_ub,
                                        self.x_gm[batch_dim_index_gm * self.repeats_i_dim_num * self.data_dim_num + \
                                                  repeats_index * self.data_dim_num + loop_offset_start_core + \
                                                  loop_index * self.element_num_each_loop - \
                                                  self.x_data_one_block + less_one_block_tail_data],
                                        0, 1, 1, 0, 0)
            with self.tik_instance.for_range(0, self.x_data_one_block - less_one_block_tail_data) as data_index:
                self.tmp_ub[data_index].set_as(self.previous_block_ub[data_index])

        with self.tik_instance.for_range(0, self.repeats_num) as repeats_num_index:
            output_offset_gm.set_as(input_repeat_offset_gm + loop_index * self.element_num_each_loop + \
                                    repeats_num_index * self.data_dim_num)
            with self.tik_instance.if_scope(element_num_loop // self.x_data_one_block != 0):
                self.tik_instance.data_move(self.y_gm[output_offset_gm], self.x_ub, 0, 1,
                                            element_num_loop // self.x_data_one_block, 0, 0)
            self.tik_instance.data_move(
                self.y_gm[output_offset_gm + (block_loop_times - 1) * self.x_data_one_block + less_one_block_tail_data],
                self.tmp_ub, 0, 1, 1, 0, 0)

    def _add_compile_info(self):
        """Add compile info
        """
        tbe_context.get_context().add_compile_info("vars", {
            "core_num": self.full_core_num,
            "ub_size": self.ub_available,
            "x_data_one_block": self.x_data_one_block,
            "repeats_data_one_block": self.repeats_data_one_block,
            "soc_version": self.soc_version,
            "dtype_support_vector_dup": self.dtype_support_vector_dup
        })

    def _tiling_args(self):
        """Get runtime tiling parameters from tiling
        """
        self.tiling_ub = self.tik_instance.Tensor(
            "int64", (Constant.TILING_ARG_NUM,), name="tiling_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 5, 0, 0)

        # read tiling int64 scalar
        self.repeats_i_dim_num.set_as(self.tiling_ub[0])
        self.repeats_o_dim_num.set_as(self.tiling_ub[1])
        self.is_last_axis.set_as(self.tiling_ub[2])
        self.batch_dim_num.set_as(self.tiling_ub[3])
        self.data_dim_num.set_as(self.tiling_ub[4])
        self.data_num.set_as(self.tiling_ub[5])
        self.batch_dim_num_each_core.set_as(self.tiling_ub[6])
        self.batch_dim_num_last_core.set_as(self.tiling_ub[7])
        self.core_num_each_batch.set_as(self.tiling_ub[8])
        self.element_num_each_core.set_as(self.tiling_ub[9])
        self.element_num_last_core.set_as(self.tiling_ub[10])
        self.core_num.set_as(self.tiling_ub[11])
        self.element_num_each_loop.set_as(self.tiling_ub[12])
        self.init_repeats_ub.set_as(self.tiling_ub[13])
        self.front_core_num.set_as(self.tiling_ub[14])
        self.tail_core_num.set_as(self.tiling_ub[15])
        self.front_core_move_data.set_as(self.tiling_ub[16])
        self.tail_core_move_data.set_as(self.tiling_ub[17])


# 'pylint: disable=unused-argument,invalid-name
@register_operator("RepeatInterleave")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def repeat_interleave(x, repeats, y, axis=1000, kernel_name="repeat_interleave"):
    """return a copy of the tensor collapsed into one dimension.

    For example:
    x = [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], with shape (2, 3, 2);
    axis = 1;
    repeats = 2;
    Then, y = [[[1, 2], [1, 2], [3, 4], [3, 4], [5, 6], [5, 6]],
               [[7, 8], [7, 8], [9, 10], [9, 10], [11, 12], [11, 12]]], with shape (2, 6, 2).

    Parameters
    ----------
    x : dict
        shape and dtype of input.
    repeats : dict
        shape and dtype of input.
    y : dict
        shape and dtype of output.
    kernel_name : str
        kernel name, default value is "repeat_interleave"

    Returns
    -------
    None
    """
    obj = RepeatInterleave(x, repeats, y, axis, kernel_name)
    return obj.repeat_interleave_compute()
