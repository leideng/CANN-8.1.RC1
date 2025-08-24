#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
scatter_max_with_argmax
"""
import warnings
import functools
from functools import reduce as functools_reduce
import numpy as np

from tbe import tik
from tbe.common import platform as tbe_platform
from tbe.common.platform import platform_info
from tbe.common.platform import set_current_compile_soc_info
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator


class Constant:
    # tiling param num
    TILING_ARG_NUM = 16
    # max int32 value
    MAX_INT32_VALUE = 2**32 - 1
    # init zero_ub size
    INIT_ZERO_UB_SIZE = 1024
    MINUS_INF = -3.4e+38
    MAX_FILL_UB_SIZE = 8192
    BLOCK_NUM = 128
    DUPLICATE_NUM = 16
    MAX_MASK = 256


# 'pylint: disable=import-error
# 'pylint: disable=too-many-arguments,unused-argument,invalid-name,redefined-outer-name,too-many-return-values
class Scatter:
    """
       Function: use to store scatter base parameters
       Modify : 2023-1-13
    """
    def __init__(self, var, indices, updates, kernel_name, compute_type):
        self.tik_instance = tik.Tik()
        self.var_dtype = var.get("dtype").lower()
        self.indices_dtype = indices.get("dtype").lower()
        self.updates_dtype = updates.get("dtype").lower()
        self.kernel_name = kernel_name
        self.compute_type = compute_type

        self.max_num_one_repeat = 64

        self.init_gm_tensor()

        self.indice_step = self.tik_instance.Scalar("int32", name="indice_step")
        self.block_num = self.tik_instance.Scalar("int32", name="block_num")
        self.indices_loop_num = self.tik_instance.Scalar("int32", name="indices_loop_num")
        self.indices_last_num = self.tik_instance.Scalar("int32", name="indices_last_num")
        self.update_data_num = self.tik_instance.Scalar("int32", name="update_data_num")
        self.indices_data_num = self.tik_instance.Scalar("int32", name="indices_data_num")
        self.var_ele_num = self.tik_instance.Scalar("int32", name="var_ele_num")
        self.indices_ub_number = self.tik_instance.Scalar("int32", name="indices_ub_number")
        self.updates_ub_number = self.tik_instance.Scalar("int32", name="updates_ub_number")
        self.max_update_indice = self.tik_instance.Scalar("int32", name="max_update_indice")
        self.max_indice = self.tik_instance.Scalar("int32", name="max_indice")
        self.indices_each_burst_len = self.tik_instance.Scalar("int32", name="indices_each_burst_len")
        self.indices_last_burst_len = self.tik_instance.Scalar("int32", name="indices_last_burst_len")
        self.updates_burst_len = self.tik_instance.Scalar("int32", name="updates_burst_len")

        self.init_ub_tensor_para()

        self.core_nums = platform_info.get_soc_spec(tbe_platform.CORE_NUM)

        self.init_ub_tensor()

        self.sync_workspace = self.tik_instance.Tensor('int64', (Constant.MAX_INT32_VALUE,),
                                                       tik.scope_gm,
                                                       'sync_workspace',
                                                       is_workspace=True,
                                                       is_atomic_add=True)

    def init_gm_tensor(self):
        """
        initialize the gm tensors

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.tiling_gm = self.tik_instance.Tensor(
            "int32",
            (Constant.TILING_ARG_NUM,),
            name="tiling_gm",
            scope=tik.scope_gm)
        self.var_gm = self.tik_instance.Tensor(
            self.var_dtype,
            (Constant.MAX_INT32_VALUE,),
            name="var_gm",
            scope=tik.scope_gm)
        self.indices_gm = self.tik_instance.Tensor(
            self.indices_dtype,
            (Constant.MAX_INT32_VALUE,),
            name="indices_gm",
            scope=tik.scope_gm)
        self.updates_gm = self.tik_instance.Tensor(
            self.updates_dtype,
            (Constant.MAX_INT32_VALUE,),
            name="updates_gm",
            scope=tik.scope_gm)
        self.out_gm = self.tik_instance.Tensor(
            self.var_dtype,
            (Constant.MAX_INT32_VALUE,),
            name="out_gm",
            is_atomic_add=True,
            scope=tik.scope_gm)
        self.out_argmax_gm = self.tik_instance.Tensor(
            self.indices_dtype,
            (Constant.MAX_INT32_VALUE,),
            name="out_argmax_gm",
            scope=tik.scope_gm)

    def init_ub_tensor_para(self):
        """
        Compute the ub size of tensors

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.ub_size_bytes = (platform_info.get_soc_spec(tbe_platform.UB_SIZE) - 8192)
        self.var_dtype_bytes_size = platform_info.get_bit_len(self.var_dtype) // 8
        self.indices_dtype_bytes_size = platform_info.get_bit_len(self.indices_dtype) // 8
        self.var_data_each_block = 32 // self.var_dtype_bytes_size
        self.indices_data_each_block = 32 // self.indices_dtype_bytes_size

    # 'pylint: disable=too-many-lines
    def init_ub_tensor(self):
        """
        initialize the ub tensors

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.tiling_ub = self.tik_instance.Tensor(
            "int32", (Constant.TILING_ARG_NUM,),
            name="tiling_ub",
            scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 2, 0, 0)
        self.tiling_args()

        self.var_ub = self.tik_instance.Tensor(
            self.var_dtype, (self.updates_ub_number,),
            name="var_ub",
            scope=tik.scope_ubuf)
        self.updates_ub = self.tik_instance.Tensor(
            self.updates_dtype, (self.updates_ub_number,),
            name="updates_ub",
            scope=tik.scope_ubuf)
        self.indices_ub = self.tik_instance.Tensor(
            self.indices_dtype, (self.indices_ub_number,),
            name="indices_ub",
            scope=tik.scope_ubuf)
        self.out_indice_ub = self.tik_instance.Tensor("int32", 
                                                      (self.indice_step,),
                                                      name="out_indice_ub",
                                                      scope=tik.scope_ubuf)
        self.tik_instance.h_duplicate(self.out_indice_ub, 0)

        self.argmax_temp_ub = self.tik_instance.Tensor(
            self.indices_dtype, (self.updates_ub_number,),
            name="argmax_temp_ub",
            scope=tik.scope_ubuf)
        self.src_argmax_ub = self.tik_instance.Tensor(
            self.indices_dtype, (self.updates_ub_number,),
            name="argmax_src_ub",
            scope=tik.scope_ubuf)

        self.argmax_ub1 = self.tik_instance.Tensor(
            "float32", (self.updates_ub_number,),
            name="argmax_ub1",
            scope=tik.scope_ubuf)
        self.argmax_ub2 = self.tik_instance.Tensor(
            "float32", (self.updates_ub_number,),
            name="argmax_ub2",
            scope=tik.scope_ubuf)
        self.argmax_ub_dst = self.tik_instance.Tensor(
            "float32", (self.updates_ub_number,),
            name="argmax_ub_dst",
            scope=tik.scope_ubuf)
        
        self.all_zero_ub = self.tik_instance.Tensor(
            self.updates_dtype, (self.updates_ub_number,),
            name="all_zero_ub",
            scope=tik.scope_ubuf
        )
        self.all_minus_infinity_ub = self.tik_instance.Tensor(
            self.updates_dtype, (self.updates_ub_number,),
            name="all_minus_infinity_ub",
            scope=tik.scope_ubuf
        )
        self.tik_instance.h_duplicate(self.all_zero_ub, 0.0)
        self.tik_instance.h_duplicate(self.all_minus_infinity_ub, Constant.MINUS_INF)

        self.mask_tensor = self.tik_instance.Tensor("bool", (self.updates_ub_number,), name="mask_tensor",
                                                    scope=tik.scope_ubuf)
        self.mask_minus_inf_tensor = self.tik_instance.Tensor("bool", (self.updates_ub_number,),
                                                              name="mask_minus_inf_tensor", scope=tik.scope_ubuf)
        self.mask_sum_tensor = self.tik_instance.Tensor("float16", (self.updates_ub_number,),
                                                        name="mask_sum_tensor", scope=tik.scope_ubuf)
        
        self.var_read_index = self.tik_instance.Scalar("int32")
        self.var_read_index.set_as(0)

        self.updates_read_index = self.tik_instance.Scalar("int32")
        self.updates_read_index.set_as(0)

        self.indices_loop_index = self.tik_instance.Scalar("int32")
        self.indices_loop_index.set_as(0)

    def calc_core_args(self, core_index, output_shape):
        """

        calculate args for each core

        """
        core_offset = self.tik_instance.Scalar(dtype="int32", name="core_offset", init_value=0)
        offset_index = self.tik_instance.Scalar(dtype="int32", name="offset_index", init_value=0)
        core_offset.set_as((output_shape + self.block_num - 1) // self.block_num)
        core_offset.set_as((core_offset + self.indices_data_each_block - 1) // self.indices_data_each_block)
        core_offset.set_as(core_offset * self.indices_data_each_block)
        offset_index.set_as(core_index * core_offset)

        with self.tik_instance.if_scope((offset_index + core_offset) > output_shape):
            core_offset.set_as(0)
            with self.tik_instance.if_scope(offset_index < output_shape):
                core_offset.set_as(output_shape - offset_index)
                core_offset.set_as((core_offset + self.indices_data_each_block - 1) // self.indices_data_each_block)
                core_offset.set_as(core_offset * self.indices_data_each_block)
                offset_index.set_as(output_shape - core_offset)

        return offset_index, core_offset

    def fill_gm_output_tensor(self, core_index):
        with self.tik_instance.new_stmt_scope():
            fill_gm_ub_shape = Constant.MAX_FILL_UB_SIZE // self.indices_dtype_bytes_size
            output_shape = self.tik_instance.Scalar(dtype="int32", name="output_shape", init_value=0)
            output_shape.set_as(self.var_ele_num)
            fill_gm_ub_argmax = self.tik_instance.Tensor(self.indices_dtype, (fill_gm_ub_shape,),
                                                         name="fill_gm_ub_argmax", scope=tik.scope_ubuf)
            fill_gm_ub_output = self.tik_instance.Tensor(self.updates_dtype, (fill_gm_ub_shape,),
                                                         name="fill_gm_ub_output", scope=tik.scope_ubuf)
            self.tik_instance.vector_dup(Constant.MAX_MASK // self.indices_dtype_bytes_size,
                                         fill_gm_ub_argmax, self.max_update_indice, Constant.DUPLICATE_NUM, 1, 8)
            self.tik_instance.vector_dup(Constant.MAX_MASK // self.var_dtype_bytes_size,
                                         fill_gm_ub_output, Constant.MINUS_INF, Constant.DUPLICATE_NUM, 1, 8)
            
            offset_index, core_offset = self.calc_core_args(core_index, output_shape)
            core_offset_block_num = self.tik_instance.Scalar(dtype="int32", name="core_offset_block_num", init_value=0)
            data_move_loop_num = self.tik_instance.Scalar(dtype="int32", name="data_move_loop_num", init_value=0)
            data_move_loop_left = self.tik_instance.Scalar(dtype="int32", name="data_move_loop_left", init_value=0)

            core_offset_block_num.set_as((core_offset +
                                          self.indices_data_each_block - 1) // self.indices_data_each_block)
            data_move_loop_num.set_as(core_offset_block_num // Constant.BLOCK_NUM)
            data_move_loop_left.set_as(core_offset_block_num % Constant.BLOCK_NUM)

            with self.tik_instance.new_stmt_scope(disable_sync=True):
                with self.tik_instance.for_range(0, data_move_loop_num) as loop_index:
                    core_offset_loop = loop_index * Constant.BLOCK_NUM * self.indices_data_each_block + offset_index
                    self.tik_instance.data_move(self.out_argmax_gm[core_offset_loop], fill_gm_ub_argmax,
                                                0, 1, Constant.BLOCK_NUM, 0, 0)
                    self.tik_instance.data_move(self.out_gm[core_offset_loop], fill_gm_ub_output, 0, 1,
                                                Constant.BLOCK_NUM, 0, 0)

                with self.tik_instance.if_scope(data_move_loop_left > 0):
                    core_offset_loop = data_move_loop_num * Constant.BLOCK_NUM * self.indices_data_each_block +\
                                       offset_index
                    self.tik_instance.data_move(self.out_argmax_gm[core_offset_loop], fill_gm_ub_argmax, 0, 1,
                                                data_move_loop_left, 0, 0)
                    self.tik_instance.data_move(self.out_gm[core_offset_loop], fill_gm_ub_output, 0, 1,
                                                data_move_loop_left, 0, 0)

    def fill_argmax_output_tensor(self):
        with self.tik_instance.new_stmt_scope():
            output_shape = self.tik_instance.Scalar(dtype="int32", name="output_shape", init_value=0)
            output_shape.set_as(self.var_ele_num)

            fill_gm_ub_argmax = self.tik_instance.Tensor(self.indices_dtype, (self.indices_data_each_block,),
                                                         name="fill_gm_ub_argmax", scope=tik.scope_ubuf)
            fill_gm_ub_output = self.tik_instance.Tensor(self.updates_dtype, (self.var_data_each_block,),
                                                         name="fill_gm_ub_output", scope=tik.scope_ubuf)
            self.tik_instance.vector_dup(8, fill_gm_ub_argmax, self.max_update_indice, 1, 1, 8)
            self.tik_instance.vector_dup(8, fill_gm_ub_output, Constant.MINUS_INF, 1, 1, 8)

            data_move_loop_num = output_shape // self.indices_data_each_block
            with self.tik_instance.for_range(0, data_move_loop_num) as loop_index:
                addr_offset = loop_index * self.indices_data_each_block
                self.tik_instance.data_move(self.out_argmax_gm[addr_offset], fill_gm_ub_argmax, 0, 1, 1, 0, 0)
                self.tik_instance.data_move(self.out_gm[addr_offset], fill_gm_ub_output, 0, 1, 1, 0, 0)

            data_move_loop_num_last = output_shape % self.indices_data_each_block
            with self.tik_instance.if_scope(data_move_loop_num_last > 0):
                last_addr_offset = (data_move_loop_num - 2) * self.indices_data_each_block + data_move_loop_num_last
                self.tik_instance.data_move(self.out_argmax_gm[last_addr_offset], fill_gm_ub_argmax, 0, 1, 1, 0, 0)
                self.tik_instance.data_move(self.out_gm[last_addr_offset], fill_gm_ub_output, 0, 1, 1, 0, 0)

    def traversing_indices(self):
        """
        Traversing the index in the indices

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(self.indices_loop_num > 0):
            with self.tik_instance.for_range(0, self.indices_loop_num) as indices_loop_index:
                self.updates_the_var(indices_loop_index * self.indices_ub_number, \
                                     self.indices_ub_number, self.indices_each_burst_len)

        with self.tik_instance.if_scope(self.indices_last_num > 0):
            self.updates_the_var(self.indices_loop_num * self.indices_ub_number, \
                                 self.indices_last_num, self.indices_last_burst_len)
        
        # fill the zeros where indices don't exist
        out_indices_value = self.tik_instance.Scalar("int32", name="out_indices_value")
        out_offset = self.tik_instance.Scalar("int32", name="out_offset")
        with self.tik_instance.for_range(0, self.indice_step) as out_index:  
            out_indices_value.set_as(self.out_indice_ub[out_index])
            out_offset.set_as(self.indices_loop_index * self.indice_step + out_index)
            with self.tik_instance.if_scope(tik.all(out_indices_value == 0, out_offset < self.max_indice)):
                self.tik_instance.data_move(
                    self.out_gm[out_offset * self.updates_ub_number], 
                                self.all_zero_ub, 0, 1, self.updates_burst_len, 0, 0) 

    def updates_the_var(self, indices_in_index, indices_last_num_index, indices_burst_len):
        """
        Update the update fragment corresponding to the index

        Parameters
        ----------
        indices_in_index: int32
            Indices index on GM
        indices_last_num_index: int32
            the number of indexes in the indices on UB
        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(self.indices_data_num == 1):
            self.tik_instance.data_move(self.indices_ub, self.indices_gm, 0, 1, indices_burst_len, 0, 0)
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(self.indices_ub, self.indices_gm[indices_in_index], 0, 1,
                                        indices_burst_len, 0, 0)

        with self.tik_instance.for_range(0, indices_last_num_index) as indices_ub_index:
            self.get_var_read_index(indices_ub_index)
            with self.tik_instance.if_scope(self.block_num > 1):
                with self.tik_instance.if_scope(self.indices_loop_index * self.indice_step <= self.var_read_index):
                    with self.tik_instance.if_scope(
                            (self.indices_loop_index + 1) * self.indice_step > self.var_read_index):
                        self.get_updates_read_index(indices_in_index + indices_ub_index)
                        self.tik_instance.h_duplicate(self.src_argmax_ub,
                                                      indices_in_index + indices_ub_index)
                        self.calc_updates_small(self.updates_ub_number)
                        self.out_indice_ub[self.var_read_index % self.indice_step].set_as(1)               

            with self.tik_instance.else_scope():
                self.get_updates_read_index(indices_in_index + indices_ub_index)
                self.tik_instance.h_duplicate(self.src_argmax_ub, indices_in_index + indices_ub_index)
                self.calc_updates_small(self.updates_ub_number)
                self.out_indice_ub[self.var_read_index].set_as(1)  

    def get_var_read_index(self, indices_ub_index):
        """
        Calculate the index of the read var

        Parameters
        ----------
        indices_ub_index: int32
            the index of the currently traversed indices in UB

        Returns
        -------
        None
        """
        self.var_read_index.set_as(self.indices_ub[indices_ub_index])

    def get_updates_read_index(self, indices_ub_index):
        """
        Calculate the index of the read updates

        Parameters
        ----------
        indices_ub_index:int32
            the index of the currently traversed indices in UB

        Returns
        -------
        None
        """
        read_index = indices_ub_index * self.update_data_num
        self.updates_read_index.set_as(read_index)

    def calc_updates_small(self, element_num):
        """
        Transfer update to UB and calculate

        Parameters
        ----------
        element_num:
            the number of elements in the slice of updates

        Returns
        -------
        None
        """
        self.tik_instance.data_move(self.var_ub, self.out_gm[self.var_read_index * self.update_data_num], 0, 1,
                                    self.updates_burst_len, 0, 0)
        self.tik_instance.data_move(self.argmax_temp_ub, self.out_argmax_gm[self.var_read_index * self.update_data_num],
                                    0, 1, self.updates_burst_len, 0, 0)

        self.tik_instance.data_move(self.updates_ub, self.updates_gm[self.updates_read_index], \
                                    0, 1, self.updates_burst_len, 0, 0)

        self.calc_process()
        self.tik_instance.data_move(self.out_gm[self.var_read_index * self.update_data_num], self.var_ub,
                                    0, 1, self.updates_burst_len, 0, 0)
        self.tik_instance.data_move(self.out_argmax_gm[self.var_read_index * self.update_data_num], self.argmax_temp_ub,
                                    0, 1, self.updates_burst_len, 0, 0)

    def calc_process(self):
        """
        Execute the corresponding calculation instruction

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        compute_repeat_strid, dst_out_ub, dst_argmax_ub, src1_ub, src2_ub = self.convert_before_calc()
        if self.compute_type == "vmax":
            self.tik_instance.h_cmpv(self.mask_tensor, src2_ub, src1_ub, "GT")
            self.tik_instance.h_sel(dst_out_ub, src2_ub, src1_ub, self.mask_tensor)            
            self.tik_instance.h_cast(self.argmax_ub1, self.argmax_temp_ub, "none")
            self.tik_instance.h_cast(self.argmax_ub2, self.src_argmax_ub, "none")
            self.tik_instance.h_sel(self.argmax_ub_dst, self.argmax_ub2, self.argmax_ub1, self.mask_tensor)    
            self.tik_instance.h_cast(self.argmax_temp_ub, self.argmax_ub_dst, "round")
        else:
            raise RuntimeError("the operater [%s] is not supported" % self.compute_type)

    def convert_before_calc(self):
        """
        convert the src and dst ub
        """
        compute_repeat_strid = (self.max_num_one_repeat // self.var_data_each_block)
        src1_ub = self.var_ub
        src2_ub = self.updates_ub
        dst_argmax_ub = self.argmax_temp_ub
        dst_out_ub = self.var_ub

        return compute_repeat_strid, dst_out_ub, dst_argmax_ub, src1_ub, src2_ub

    def tiling_args(self):
        """
        get tiling data
        """
        self.indice_step.set_as(self.tiling_ub[0])
        self.block_num.set_as(self.tiling_ub[1])
        self.indices_loop_num.set_as(self.tiling_ub[2])
        self.indices_last_num.set_as(self.tiling_ub[3])
        self.update_data_num.set_as(self.tiling_ub[4])
        self.indices_data_num.set_as(self.tiling_ub[5])
        self.var_ele_num.set_as(self.tiling_ub[6])
        self.indices_ub_number.set_as(self.tiling_ub[7])
        self.updates_ub_number.set_as(self.tiling_ub[8])
        self.max_update_indice.set_as(self.tiling_ub[9])
        self.max_indice.set_as(self.tiling_ub[10])
        self.indices_each_burst_len.set_as(self.tiling_ub[11])
        self.indices_last_burst_len.set_as(self.tiling_ub[12])
        self.updates_burst_len.set_as(self.tiling_ub[13])

    def scatter_operator(self):
        """
        Scatter operation

        Parameters
        ----------
        None

        Returns:
        ----------
        tik_instance: tik instance
        """
        tbe_context.get_context().add_compile_info(
            "vars", {
                "ub_size": self.ub_size_bytes,
                "core_num": self.core_nums,
                "var_size": self.var_dtype_bytes_size,
                "indices_size": self.indices_dtype_bytes_size,
            })

        with self.tik_instance.if_scope(self.block_num > 1):
            with self.tik_instance.for_range(0, self.block_num, block_num=self.block_num) as indices_loop_index:
                self.indices_loop_index.set_as(indices_loop_index)

                self.fill_gm_output_tensor(indices_loop_index)
                self.tik_instance.block_barrier(self.sync_workspace)

                self.traversing_indices()
        with self.tik_instance.else_scope():
            self.indices_loop_index.set_as(0)
            self.fill_argmax_output_tensor()

            self.traversing_indices()

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=(self.var_gm, self.indices_gm, self.updates_gm),
            outputs=(self.out_gm, self.out_argmax_gm),
            flowtable=[self.tiling_gm],
            enable_l2=False)

        return self.tik_instance


# # 'pylint: disable=too-many-arguments,unused-argument,invalid-name
@register_operator("ScatterMaxWithArgmax")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def scatter_max_with_argmax(x,
                            indices,
                            updates,
                            y,
                            argmax,
                            kernel_name="scatter_max_with_argmax"):
    """
    Subtracts sparse updates to a variable reference.

    Parameters
    ----------
    x: dict
    data of input.
    source data type, support "float32"
    indices: dict
    A tensor of indices into var, support "int32"
    updates: dict
    data of updates
    source data type should ne same as var
    y: dict
    data of output.
    argmax: dict
    data of output.
    kernel_name: str
    kernel name, default value is "scatter_max_with_argmax"

    Returns:
    None
    """
    scatter_nd = Scatter(x, indices, updates, kernel_name, "vmax")

    return scatter_nd.scatter_operator()