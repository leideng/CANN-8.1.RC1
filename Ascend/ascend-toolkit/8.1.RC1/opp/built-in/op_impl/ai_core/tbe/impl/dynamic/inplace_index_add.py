#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
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
inplace_index_add.py
"""

from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import error_manager_vector
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info
from tbe.common.platform import get_bit_len


# 'pylint: disable=locally-disabled,unused-argument,invalid-name
def check_supported(var, indices, updates, alpha, var_out, axis, kernel_name="index_add"):
    """
    AICORE or AICPU selection
    """
    var_dtype = var.get("dtype")
    support_vmuls = tbe_platform.api_check_support("tik.vmuls", var_dtype)
    if (alpha is not None) and (var_dtype not in ("bfloat16",)) and (not support_vmuls):
        return False, "tik.vmuls does not support this dtype, switch to AICPU."
    return True, ""


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant
    """
    MAX_INT64_VALUE = 2 ** 64 - 1
    MAX_INT32_VALUE = 2 ** 31 - 1
    TILING_ARG_NUM = 16
    RESERVED_UB_SIZE = 8 * 1024
    EIGHT_BIT = 8
    NUM_EACH_BURST_INT64 = 4
    # mask size
    ONE_VECTOR_CALC_SIZE = 256
    # mask of type in calculation
    TYPE_BYTES_MAP = {"float16": 2, "float32": 4, "int8": 2, "uint8": 2, "int16": 2, "int32": 4, "bfloat16": 4}
    # bytes of one block
    ONE_BLOCK_SIZE = 32
    # mask of fp32
    MASK_32 = 64
    # max repeat times
    MAX_REPEAT_TIMES = 255
    # soc version supports atomic add for multiple data types or not
    SOC_VERSION_MULTI_ATOMIC_ADD = 1
    SOC_VERSION_OTHER = 0


# 'pylint: disable=unused-argument,undefined-variable,too-many-arguments
def get_op_support_info(var, indices, updates, alpha, var_out, axis, kernel_name="index_add"):
    """
    get_op_support_info
    """
    format_var = var.get("format").upper()
    format_indices = indices.get("format").upper()
    format_updates = updates.get("format").upper()
    if alpha is not None:
        format_alpha = alpha.get("format").upper()
    else:
        format_alpha = "ND"
    shape_indices_len = len(indices.get("shape"))
    if format_var == "ND" and format_indices == "ND" and format_updates == "ND" and format_alpha == "ND":
        axis_split_matrix = []
        for j in range(shape_indices_len):
            split_0 = [SplitInput([1, [j], [-1], [-1]]), SplitOutput([0, [j]])]
            axis_split_matrix.append(split_0)
        axis_reduce_list = None

    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


# 'pylint: disable=too-many-arguments,too-many-instance-attributes,too-many-public-methods
# 'pylint: disable=invalid-name,attribute-defined-outside-init,unused-argument
class InplaceIndexAdd():
    """
       Function: use to store scatter base parameters
    """

    def __init__(self, var, indices, updates, alpha, var_out, axis, kernel_name):
        """
        Init scatter base parameters
        :param var: dict
            data of input
            datatype supports float32,float16,int32,int8,uint8
        :param indices: dict
            data of indices
            datatype supports int32
        :param updates: dict
            data of updates
            datatype supports float32,float16,int32,int8,uint8
        :param alpha: dict
            data of alpha
            datatype supports float32,float16,int32,int8,uint8
        :param var_out: dicts
            data of input
        :param axis: int
            which axis to compute index add
        :param kernel_name:
        """
        if alpha is not None:
            self.is_alpha = True
        else:
            self.is_alpha = False
        self.tik_instance = tik.Tik()
        self.var_dtype = var.get("dtype").lower()
        self.indices_dtype = indices.get("dtype").lower()
        self.updates_dtype = updates.get("dtype").lower()
        if self.is_alpha:
            self.alpha_dtype = alpha.get("dtype").lower()
        self.var_out_dtype = var_out.get("dtype").lower()
        self.kernel_name = kernel_name

        # check input attr params
        self.check_param()

        # check if the soc version supports atomic add for multiple data types
        self.support_multi_atomic_add = tbe_platform.api_check_support("tik.data_move_pad")
        if self.support_multi_atomic_add:
            self.soc_version = Constant.SOC_VERSION_MULTI_ATOMIC_ADD
        else:
            self.soc_version = Constant.SOC_VERSION_OTHER

        # check if atomic add is supported
        atomic_add_support_dtype_list = ("float16", "float32", "int32", "int8", "int16", "bfloat16")
        if self.var_dtype in atomic_add_support_dtype_list:
            self.atomic_add = 1
            self.atomic_add_flag = True
        else:
            self.atomic_add = 0
            self.atomic_add_flag = False

        # conv dtype for int8/uint8/bfloat16
        self.vconv_dst_dtype = self.var_dtype
        self.conv_dtype()
        self.vconv_dtype_bytes_size = get_bit_len(self.vconv_dst_dtype) // Constant.EIGHT_BIT
        self.vconv_data_each_block = Constant.ONE_BLOCK_SIZE // self.vconv_dtype_bytes_size

        self.ai_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size_bytes = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - Constant.RESERVED_UB_SIZE)
        self.var_dtype_bytes_size = get_bit_len(self.var_dtype) // Constant.EIGHT_BIT
        self.indices_dtype_bytes_size = get_bit_len(self.indices_dtype) // Constant.EIGHT_BIT
        self.var_data_each_block = Constant.ONE_BLOCK_SIZE // self.var_dtype_bytes_size
        self.indices_data_each_block = Constant.ONE_BLOCK_SIZE // self.indices_dtype_bytes_size

        self.tiling_gm = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                                  name="tiling_gm", scope=tik.scope_gm)
        self.var_gm = self.tik_instance.Tensor(self.var_dtype, (Constant.MAX_INT64_VALUE,),
                                               name="var_gm", scope=tik.scope_gm)
        self.indices_gm = self.tik_instance.Tensor(self.indices_dtype, (Constant.MAX_INT64_VALUE,),
                                                   name="indices_gm", scope=tik.scope_gm)
        self.updates_gm = self.tik_instance.Tensor(self.updates_dtype, (Constant.MAX_INT64_VALUE,),
                                                   name="updates_gm", scope=tik.scope_gm)
        self.var_workspace = self.tik_instance.Tensor("float32", (Constant.MAX_INT32_VALUE,),
                                                      name="var_workspace", scope=tik.scope_gm, is_workspace=True)
        self.updates_workspace = self.tik_instance.Tensor("float32", (Constant.MAX_INT32_VALUE,),
                                                          name="updates_workspace", scope=tik.scope_gm,
                                                          is_workspace=True)
        if self.is_alpha:
            self.alpha_gm = self.tik_instance.Tensor(self.alpha_dtype, (Constant.MAX_INT64_VALUE,),
                                                     name="alpha_gm", scope=tik.scope_gm)
        self.var_out_gm = self.tik_instance.Tensor(self.var_out_dtype, (Constant.MAX_INT64_VALUE,),
                                                   name="var_out_gm", scope=tik.scope_gm)

        # decide the mask of computation
        self.max_num_one_repeat = Constant.ONE_VECTOR_CALC_SIZE // Constant.TYPE_BYTES_MAP.get(self.var_dtype)

        # init some variable
        self.init_variable()

    def check_param(self):
        """
        Check whether the input parameters are valid or not
        """
        indices_support_dtype_list = ("int32", "int64")
        var_support_dtype_list = ("float16", "float32", "int32", "int8", "uint8", "int16", "bfloat16")
        if self.var_out_dtype == "bool":
            self.var_out_dtype = "int8"
        para_check.check_dtype(self.indices_dtype, indices_support_dtype_list, param_name="indices")
        para_check.check_dtype(self.var_dtype, var_support_dtype_list, param_name="var")
        if self.var_dtype != self.updates_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal(self.kernel_name, "update", "var",
                                                                  self.updates_dtype, self.var_dtype)
        if self.is_alpha:
            if self.var_dtype != self.alpha_dtype:
                error_manager_vector.raise_err_inputs_dtype_not_equal(self.kernel_name, "alpha", "var",
                                                                      self.alpha_dtype, self.var_dtype)
        if self.var_dtype != self.var_out_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal(self.kernel_name, "var_out", "var",
                                                                  self.var_out_dtype, self.var_dtype)

    def conv_dtype(self):
        """
        Conv dtype
        """
        need_vconv_dtype_int8_uint8 = ("int8", "uint8")
        need_vconv_dtype_bfloat16 = ("bfloat16",)
        if self.var_dtype in need_vconv_dtype_int8_uint8:
            if (not self.support_multi_atomic_add):
                self.vconv_dst_dtype = "float16"
            elif (not self.atomic_add_flag) and (not self.is_alpha):
                self.vconv_dst_dtype = "float16"
        if self.var_dtype in need_vconv_dtype_bfloat16:
            self.vconv_dst_dtype = "float32"

    def init_variable(self):
        """
        Init Variable
        :return:
        """
        # state some ub tensor
        self.var_vconv_ub = None
        self.updates_vconv_ub = None

        self.var_ub = None
        self.updates_ub = None
        self.indices_ub = None
        self.tiling_ub = None

        self.var_read_index = None
        self.updates_read_index = None

        self.outer_loop_start_index_every_block = None
        self.outer_loops_ub_per_block = None
        self.outer_loop_start_index_of_var = None
        self.outer_loop_start_index_of_updates = None

        self.index_offset = None
        self.alpha_scalar = None

        self.core_num = None
        self.indices_num = None
        self.outer_loop = None
        self.outer_loops_per_block = None
        self.tail_num = None
        self.axis_and_after_data_num_of_updates = None
        self.axis_and_after_data_num_of_var = None
        self.update_data_num = None
        self.axis = None
        self.updates_ub_size = None
        self.indices_ub_size = None
        self.core_num_var = None
        self.var_shape_num = None
        self.updates_shape_num = None

    def tiling_args(self):
        """
        Get runtime params from tiling

        Parameters
        ----------
        tiling_ub: tensor, runtime params from inpalce_index_add tiling

        Returns
        -------
        None
        """
        self.core_num = self.tik_instance.Scalar(name="core_num")
        self.indices_num = self.tik_instance.Scalar(name="indices_num")
        self.outer_loop = self.tik_instance.Scalar(name="outer_loop")
        self.outer_loops_per_block = self.tik_instance.Scalar(name="outer_loops_per_block")
        self.tail_num = self.tik_instance.Scalar(name="tail_num")
        self.axis_and_after_data_num_of_updates = self.tik_instance.Scalar(name="axis_and_after_data_num_of_updates")
        self.axis_and_after_data_num_of_var = self.tik_instance.Scalar(name="axis_and_after_data_num_of_var")
        self.update_data_num = self.tik_instance.Scalar(name="update_data_num")
        self.axis = self.tik_instance.Scalar(name="axis")
        self.updates_ub_size = self.tik_instance.Scalar(name="updates_ub_size")
        self.indices_ub_size = self.tik_instance.Scalar(name="indices_ub_size")
        self.core_num_var = self.tik_instance.Scalar(name="core_num_var", init_value=self.ai_core_num)
        self.var_shape_num = self.tik_instance.Scalar(name="var_shape_num")
        self.updates_shape_num = self.tik_instance.Scalar(name="updates_shape_num")

        self.core_num.set_as(self.tiling_ub[0])
        self.indices_num.set_as(self.tiling_ub[1])
        self.outer_loop.set_as(self.tiling_ub[2])
        self.outer_loops_per_block.set_as(self.tiling_ub[3])
        self.tail_num.set_as(self.tiling_ub[4])
        self.axis_and_after_data_num_of_updates.set_as(self.tiling_ub[5])
        self.axis_and_after_data_num_of_var.set_as(self.tiling_ub[6])
        self.update_data_num.set_as(self.tiling_ub[7])
        self.axis.set_as(self.tiling_ub[8])
        self.updates_ub_size.set_as(self.tiling_ub[9])
        self.indices_ub_size.set_as(self.tiling_ub[10])
        self.core_num_var.set_as(self.tiling_ub[11])
        self.var_shape_num.set_as(self.tiling_ub[12])
        self.updates_shape_num.set_as(self.tiling_ub[13])

    def init_ub_tensor(self):
        """
        Init the ub tensor
        :return:
        """
        # init int8/uint8/bfloat16 ub tensors
        self.init_ub_tensor_need_cast()

        # init var/indices/updates ub
        self.init_ub_tensor_general()

        # init ub tensor of read index
        self.init_ub_tensor_of_read_index()

    def init_ub_tensor_need_cast(self):
        """
        Init ub tensor if var dtype is in (int8, uint8, bfloat16)
        :return:
        """
        need_vconv_dtype = ("int8", "uint8", "bfloat16")
        if self.var_dtype in need_vconv_dtype:
            self.var_vconv_ub = self.tik_instance.Tensor(
                self.vconv_dst_dtype, (self.updates_ub_size,),
                name="var_vconv_ub",
                scope=tik.scope_ubuf)
            self.updates_vconv_ub = self.tik_instance.Tensor(
                self.vconv_dst_dtype, (self.updates_ub_size,),
                name="updates_vconv_ub",
                scope=tik.scope_ubuf)

    def init_ub_tensor_general(self):
        """
        Init ub tensor for all var dtype
        :return:
        """
        self.var_ub = self.tik_instance.Tensor(
            self.var_dtype, (self.updates_ub_size,),
            name="var_ub",
            scope=tik.scope_ubuf)
        self.updates_ub = self.tik_instance.Tensor(
            self.updates_dtype, (self.updates_ub_size,),
            name="updates_ub",
            scope=tik.scope_ubuf)
        self.indices_ub = self.tik_instance.Tensor(
            self.indices_dtype, (self.indices_ub_size,),
            name="indices_ub",
            scope=tik.scope_ubuf)

    def init_ub_tensor_of_read_index(self):
        """
        Init ub tensor of loop/index
        :return:
        """
        self.var_read_index = self.tik_instance.Scalar()
        self.var_read_index.set_as(0)

        self.updates_read_index = self.tik_instance.Scalar()
        self.updates_read_index.set_as(0)

        self.outer_loop_start_index_every_block = self.tik_instance.Scalar()
        self.outer_loop_start_index_every_block.set_as(0)

        self.outer_loops_ub_per_block = self.tik_instance.Scalar()
        self.outer_loops_ub_per_block.set_as(self.outer_loop)

        self.outer_loop_start_index_of_var = self.tik_instance.Scalar()
        self.outer_loop_start_index_of_var.set_as(0)

        self.outer_loop_start_index_of_updates = self.tik_instance.Scalar()
        self.outer_loop_start_index_of_updates.set_as(0)

        self.index_offset = self.tik_instance.Scalar(name="index_offset")

    def get_index_val(self, indices_ub_index):
        """
        Get selected index value
        :param indices_ub_index: looping index in indices ub
        :return:
        """
        read_index_var = self.tik_instance.Scalar(name="read_index_var")
        read_index_var.set_as(self.indices_ub[indices_ub_index])
        self.var_read_index.set_as(read_index_var)

    def get_updates_read_index(self, indices_ub_index):
        """
        Get absolute updates index according to indices index
        :param indices_ub_index: looping index in indices ub
        :return:
        """
        read_index_updates = self.tik_instance.Scalar(name="read_index_updates")
        read_index_updates.set_as(self.outer_loop_start_index_of_updates +
                                  indices_ub_index * self.update_data_num)
        self.updates_read_index.set_as(read_index_updates)

    def get_var_read_index(self):
        """
        Get absolute updates index according to indices index
        :return:
        """
        self.var_read_index.set_as(self.var_read_index * self.update_data_num +
                                   self.outer_loop_start_index_of_var)

    def update_the_var(self, offset_indices, indice_num):
        """
        Update the update fragment corresponding to the index
        :param offset_indices: start offset of indices
        :param indice_num: indices number this time to update
        :return:
        """
        # calc the burst length of indices
        indices_burst_len = self.tik_instance.Scalar(name="indices_burst_len")
        indices_burst_len.set_as((indice_num - 1) // self.indices_data_each_block + 1)

        # move indices from gm to ub
        with self.tik_instance.if_scope(self.indices_num == 1):
            self.tik_instance.data_move(self.indices_ub, self.indices_gm,
                                        0, 1, indices_burst_len, 0, 0)
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(self.indices_ub,
                                        self.indices_gm[offset_indices],
                                        0, 1, indices_burst_len, 0, 0)

        # loop over indices
        with self.tik_instance.for_range(0, indice_num) as indices_ub_index:
            self.get_index_val(indices_ub_index)
            self.get_updates_read_index(indices_ub_index + offset_indices)
            self.get_var_read_index()
            self.calc_updates()

    def calc_updates(self):
        """
        Calculate updates fragment
        :return:
        """
        # calc the loop times of update data once to ub
        updates_loop = self.tik_instance.Scalar(name="updates_loop")
        updates_loop.set_as(self.update_data_num // self.updates_ub_size)

        with self.tik_instance.if_scope(updates_loop > 0):
            with self.tik_instance.for_range(0, updates_loop) as loop_index:
                self.calc_updates_small(loop_index * self.updates_ub_size,
                                        self.updates_ub_size)

        # deal with tail num
        last_num_updates = self.tik_instance.Scalar(name="last_num_updates")
        last_num_updates.set_as(self.update_data_num % self.updates_ub_size)

        with self.tik_instance.if_scope(last_num_updates > 0):
            self.calc_updates_small(updates_loop * self.updates_ub_size,
                                    last_num_updates)

    def calc_updates_small(self, read_index_offset, element_num):
        """
        Move corresponding updates/var to UB and calculate
        :param read_index_offset: offset of inner ub in loop
        :param element_num: element number once to ub
        :return:
        """
        # calc the burst length of updates/var
        updates_burst_len = self.tik_instance.Scalar(name="updates_burst_len")
        updates_burst_len.set_as((element_num - 1) // self.var_data_each_block + 1)

        # move var and updates from gm to ub
        self.tik_instance.data_move(
            self.var_ub, self.var_gm[self.var_read_index + read_index_offset],
            0, 1, updates_burst_len, 0, 0)
        self.tik_instance.data_move(
            self.updates_ub,
            self.updates_gm[self.updates_read_index + read_index_offset],
            0, 1, updates_burst_len, 0, 0)

        # calc repeat loop
        self.calc_repeat_loop(element_num)

        # compute the mask
        self.compute_mask(read_index_offset, element_num, updates_burst_len)

    def calc_repeat_loop(self, element_num):
        """
        Calc repeat loop
        :param element_num: element number once to ub
        :return:
        """
        # calc the num of full loops
        compute_loop = self.tik_instance.Scalar(name="compute_loop")
        compute_loop.set_as(element_num // self.max_num_one_repeat // Constant.MAX_REPEAT_TIMES)

        with self.tik_instance.if_scope(compute_loop > 0):
            with self.tik_instance.for_range(0, compute_loop) as index:
                self.index_offset.set_as(index * self.max_num_one_repeat * Constant.MAX_REPEAT_TIMES)
                self.calc_process(self.max_num_one_repeat, self.index_offset,
                                  self.index_offset, self.index_offset, Constant.MAX_REPEAT_TIMES)

        # deal with last loop
        last_loop = self.tik_instance.Scalar(name="last_loop")
        last_loop.set_as(element_num % (self.max_num_one_repeat *
                                        Constant.MAX_REPEAT_TIMES) // self.max_num_one_repeat)

        with self.tik_instance.if_scope(last_loop > 0):
            self.index_offset.set_as(compute_loop * self.max_num_one_repeat * Constant.MAX_REPEAT_TIMES)
            self.calc_process(self.max_num_one_repeat, self.index_offset,
                              self.index_offset, self.index_offset, last_loop)

    def compute_mask(self, read_index_offset, element_num, updates_burst_len):
        """
        Compute the var and update data according to every repeat and data move out
        :param read_index_offset: offset of inner ub in loop
        :param element_num: element number once to ub
        :param updates_burst_len: burst length of updates/var
        :return:
        """
        # calc the num of repeats
        compute_mask = self.tik_instance.Scalar(name="compute_mask")
        compute_mask.set_as(element_num % self.max_num_one_repeat)

        with self.tik_instance.if_scope(compute_mask > 0):
            self.index_offset.set_as(
                element_num // self.max_num_one_repeat *
                self.max_num_one_repeat)
            self.calc_process(compute_mask, self.index_offset, self.index_offset,
                              self.index_offset, 1)
        self.tik_instance.data_move(
            self.var_out_gm[self.var_read_index + read_index_offset],
            self.var_ub,
            0, 1, updates_burst_len, 0, 0)

    def calc_process(self, mask, dst_addr, src1_addr, src2_addr, repeat_times):
        """
        Execute the corresponding calculation instruction
        :param mask: calc mask
        :param dst_addr: dst addr
        :param src1_addr: src1 addr
        :param src2_addr: src2 addr
        :param repeat_times: repeat times
        :return:
        """
        need_vconv_dtype_int8_uint8 = ("int8", "uint8")
        (mask, dst_ub, src1_ub, src2_ub, compute_repeat_stride) = self.compute_paras(
            mask, dst_addr, src1_addr, src2_addr, repeat_times)

        if self.is_alpha:
            # scale by alpha
            self.tik_instance.vmuls(mask, src2_ub, src2_ub, self.alpha_scalar, repeat_times,
                                    1, 1, compute_repeat_stride, compute_repeat_stride)

        self.tik_instance.vadd(mask, dst_ub, src1_ub, src2_ub, repeat_times,
                               1, 1, 1, compute_repeat_stride,
                               compute_repeat_stride, compute_repeat_stride)

        if self.var_dtype in need_vconv_dtype_int8_uint8:
            self.tik_instance.vconv(mask, "", self.var_ub[src1_addr],
                                    self.var_vconv_ub[dst_addr],
                                    repeat_times, 1, 1, 4, 8)            

    def compute_paras(self, mask, dst_addr, src1_addr, src2_addr, repeat_times):
        """
        Compute the computation parameters
        :param mask: calc mask
        :param dst_addr: dst addr
        :param src1_addr: src1 addr
        :param src2_addr: src2 addr
        :param repeat_times: repeat times
        :return:
        """
        need_vconv_dtype = ("int8", "uint8")
        if self.var_dtype in need_vconv_dtype:
            self.tik_instance.vconv(mask, "", self.var_vconv_ub[dst_addr],
                                    self.var_ub[src1_addr],
                                    repeat_times, 1, 1, 8, 4)
            self.tik_instance.vconv(mask, "",
                                    self.updates_vconv_ub[dst_addr],
                                    self.updates_ub[src2_addr],
                                    repeat_times, 1, 1, 8, 4)
            compute_repeat_stride = 8
            src1_ub = self.var_vconv_ub[src1_addr]
            src2_ub = self.updates_vconv_ub[src2_addr]
            dst_ub = self.var_vconv_ub[dst_addr]

        else:
            compute_repeat_stride = (
                self.max_num_one_repeat // self.var_data_each_block)
            src1_ub = self.var_ub[src1_addr]
            src2_ub = self.updates_ub[src2_addr]
            dst_ub = self.var_ub[dst_addr]

        compute_info_dict = (mask, dst_ub, src1_ub, src2_ub, compute_repeat_stride)
        return compute_info_dict

    def traversing_indices(self):
        """
        Traverse the indices and update the var
        :return:
        """
        # calc the loop times of indices once to ub
        indices_loop_num = self.tik_instance.Scalar(name="indices_loop_num")
        indices_loop_num.set_as(self.indices_num // self.indices_ub_size)

        with self.tik_instance.if_scope(indices_loop_num > 0):
            with self.tik_instance.for_range(0, indices_loop_num) as indices_loop_index:
                self.update_the_var(indices_loop_index * self.indices_ub_size,
                                    self.indices_ub_size)

        # tail num of indices data
        indices_last_num = self.tik_instance.Scalar(name="indices_last_num")
        indices_last_num.set_as(self.indices_num % self.indices_ub_size)
        with self.tik_instance.if_scope(indices_last_num > 0):
            self.update_the_var(indices_loop_num * self.indices_ub_size,
                                indices_last_num)

    def traversing_indices_atomic_add(self):
        """
        Traverse the indices and update the var using atomic add
        :return:
        """
        # calc the loop times of indices once to ub
        indices_loop_num = self.tik_instance.Scalar(name="indices_loop_num")
        indices_loop_num.set_as(self.outer_loops_ub_per_block // self.indices_ub_size)

        with self.tik_instance.if_scope(indices_loop_num > 0):
            with self.tik_instance.for_range(0, indices_loop_num) as indices_loop_index:
                self.update_the_var_atomic_add(
                    self.outer_loop_start_index_every_block + indices_loop_index * self.indices_ub_size,
                    self.indices_ub_size)

        # tail num of indices data
        indices_last_num = self.tik_instance.Scalar(name="indices_last_num")
        indices_last_num.set_as(self.outer_loops_ub_per_block % self.indices_ub_size)

        with self.tik_instance.if_scope(indices_last_num > 0):
            self.update_the_var_atomic_add(
                self.outer_loop_start_index_every_block + indices_loop_num * self.indices_ub_size,
                indices_last_num)

    def get_outer_loop_index_of_var(self, outer_loop_num):
        """
        Get absolute loop start offset of var
        :param outer_loop_num: which outer loop it belongs to
        :return:
        """
        real_index_var = self.tik_instance.Scalar(name="real_index_var")
        real_index_var.set_as(outer_loop_num * self.axis_and_after_data_num_of_var)
        self.outer_loop_start_index_of_var.set_as(real_index_var)

    def get_outer_loop_index_of_updates(self, outer_loop_num):
        """
        Get absolute loop start offset of update
        :param outer_loop_num: which outer loop it belongs to
        :return:
        """
        real_index_updates = self.tik_instance.Scalar(name="real_index_updates")
        real_index_updates.set_as(outer_loop_num * self.axis_and_after_data_num_of_updates)
        self.outer_loop_start_index_of_updates.set_as(real_index_updates)

    def update_the_var_axis_0(self, offset_indices, indice_num):
        """
        Update the update fragment corresponding to the index when axis 0
        :param offset_indices: start offset of indices
        :param indice_num: indices number this time to update
        :return:
        """
        # calc the burst length of indices
        indices_burst_len = self.tik_instance.Scalar(name="indices_burst_len")
        indices_burst_len.set_as((indice_num - 1) // self.indices_data_each_block + 1)

        # move indices from gm to ub
        self.tik_instance.data_move(self.indices_ub, self.indices_gm[offset_indices],
                                    0, 1, indices_burst_len, 0, 0)

        read_index_var = self.tik_instance.Scalar(name="read_index_var")

        # loop over indices
        with self.tik_instance.for_range(0, indice_num) as indices_ub_index:
            read_index_var.set_as(self.indices_ub[indices_ub_index])
            with self.tik_instance.if_scope(
                    tik.all(read_index_var >= self.outer_loop_start_index_every_block,
                            read_index_var < self.outer_loop_start_index_every_block + self.outer_loops_ub_per_block)):
                self.var_read_index.set_as(read_index_var * self.update_data_num)
                self.updates_read_index.set_as((indices_ub_index + offset_indices) * self.update_data_num)
                self.calc_updates()

    def update_the_var_atomic_add(self, offset_indices, indice_num):
        """
        Update the update fragment corresponding to the index using atomic add
        :param offset_indices: start offset of indices
        :param indice_num: indices number this time to update
        :return:
        """
        # calc the burst length of indices
        indices_burst_len = self.tik_instance.Scalar(name="indices_burst_len")
        indices_burst_len.set_as((indice_num - 1) // self.indices_data_each_block + 1)

        # move indices from gm to ub
        self.data_move_pad_bit64_2_int8(self.indices_ub, self.indices_gm[offset_indices], 1,
                                        indice_num * self.indices_dtype_bytes_size, 0, 0)

        # loop over indices
        with self.tik_instance.for_range(0, indice_num) as indices_ub_index:
            self.get_index_val(indices_ub_index)
            self.get_updates_read_index(indices_ub_index + offset_indices)
            self.get_var_read_index()
            self.tik_instance.set_atomic_add(self.vconv_dst_dtype)
            self.atomic_add_updates()
            self.tik_instance.set_atomic_add(0)

    def update_the_var_axis_0_atomic_add(self, offset_indices, indice_num):
        """
        Update the update fragment corresponding to the index using atomic add when axis 0
        :param offset_indices: start offset of indices
        :param indice_num: indices number this time to update
        :return:
        """
        # calc the burst length of indices
        indices_burst_len = self.tik_instance.Scalar(name="indices_burst_len")
        indices_burst_len.set_as((indice_num - 1) // self.indices_data_each_block + 1)

        # move indices from gm to ub
        self.data_move_pad_bit64_2_int8(self.indices_ub, self.indices_gm[offset_indices], 1,
                                        indice_num * self.indices_dtype_bytes_size, 0, 0)

        read_index_var = self.tik_instance.Scalar(name="read_index_var")

        # loop over indices
        with self.tik_instance.for_range(0, indice_num) as indices_ub_index:
            read_index_var.set_as(self.indices_ub[indices_ub_index])
            self.var_read_index.set_as(read_index_var * self.update_data_num)
            self.updates_read_index.set_as((indices_ub_index + offset_indices) * self.update_data_num)
            self.tik_instance.set_atomic_add(self.vconv_dst_dtype)
            self.atomic_add_updates()
            self.tik_instance.set_atomic_add(0)

    def atomic_add_updates(self):
        """
        Calculate updates fragment using atomic add
        :return:
        """
        # calc the loop times of update data once to ub
        updates_loop = self.tik_instance.Scalar(name="updates_loop")
        updates_loop.set_as(self.update_data_num // self.updates_ub_size)
        # calc the burst length of update data once in data move
        updates_burst_len = self.tik_instance.Scalar(name="updates_burst_len")
        updates_burst_len.set_as(
            (self.updates_ub_size - 1) // self.vconv_data_each_block + 1)

        # full loops of update data
        with self.tik_instance.if_scope(updates_loop > 0):
            with self.tik_instance.for_range(0, updates_loop) as loop_index:
                if self.var_dtype == "bfloat16":
                    self.atomic_add_compute_updates(self.var_workspace, self.updates_workspace, self.updates_vconv_ub,
                                                    self.updates_ub_size, updates_burst_len, loop_index, 0)
                else:
                    self.atomic_add_compute_updates(self.var_out_gm, self.updates_gm, self.updates_ub,
                                                    self.updates_ub_size, updates_burst_len, loop_index, 0)

        # tail num of update data
        last_num_updates = self.tik_instance.Scalar(name="last_num_updates")
        last_num_updates.set_as(self.update_data_num % self.updates_ub_size)
        # update burst length
        updates_burst_len.set_as((last_num_updates - 1) // self.vconv_data_each_block + 1)
        # tail in data move
        tail_ele_num = self.tik_instance.Scalar(name="tail_ele_num")
        tail_ele_num.set_as(last_num_updates % self.vconv_data_each_block)

        # tail of update data
        with self.tik_instance.if_scope(last_num_updates > 0):
            if self.var_dtype == "bfloat16":
                self.atomic_add_compute_updates(self.var_workspace, self.updates_workspace, self.updates_vconv_ub,
                                                last_num_updates, updates_burst_len, updates_loop, tail_ele_num)
            else:
                self.atomic_add_compute_updates(self.var_out_gm, self.updates_gm, self.updates_ub,
                                                last_num_updates, updates_burst_len, updates_loop, tail_ele_num)

    def atomic_add_compute_updates(self, dst_gm, src_gm, calc_ub,
                                   element_num, updates_burst_len, loop_index, tail_ele_num=0):
        """
        Compute function using atomic add
        :param dst_gm: dst gm, var out gm or var workspace for bf16
        :param src_gm: src gm, updates gm or updates workspace for bf16
        :param calc_ub: updates ub
        :param element_num: num of update data once in data move
        :param updates_burst_len: burst length of updates/var
        :param loop_index: loop index of loop times in update data
        :param tail_ele_num: tail element num of update data
        :return:
        """
        # move update to ub
        self.tik_instance.data_move_pad(
            calc_ub,
            src_gm[self.updates_read_index + loop_index * self.updates_ub_size],
            1, element_num * self.vconv_dtype_bytes_size, 0, 0)

        # recover tail in alignment
        with self.tik_instance.if_scope(tail_ele_num != 0):
            # keep aligned elements after tail unchanged
            recover_ele_num = self.tik_instance.Scalar(name="recover_ele_num")
            recover_ele_num.set_as(self.vconv_data_each_block - tail_ele_num)
            with self.tik_instance.for_range(0, recover_ele_num) as recover_ele_index:
                calc_ub[element_num + recover_ele_index].set_as(0)

        if self.is_alpha:
            # scale by alpha
            self.data_muls(calc_ub, calc_ub, self.alpha_scalar, [0, 0], num=self.updates_ub_size)

        # move update to var_out_gm
        self.tik_instance.data_move_pad(
            dst_gm[self.var_read_index + loop_index * self.updates_ub_size],
            calc_ub,
            1, element_num * self.vconv_dtype_bytes_size, 0, 0)

    def data_conv_gm_workspace(self, src_type):
        """
        Conv between gm and workspace
        :param src_type: conv bf16->fp32 or fp32->bf16
        :return:
        """
        if src_type == "bfloat16":
            # conv gm bf16 to workspace fp32
            self.data_move_gm_workspace(self.var_workspace, self.var_gm, self.var_vconv_ub, self.var_ub,
                                        self.var_shape_num, "",
                                        self.vconv_data_each_block, self.var_data_each_block,
                                        dst_stride=8, src_stride=4)
            self.data_move_gm_workspace(self.updates_workspace, self.updates_gm, self.updates_vconv_ub, self.updates_ub,
                                        self.updates_shape_num, "",
                                        self.vconv_data_each_block, self.var_data_each_block,
                                        dst_stride=8, src_stride=4)
        if src_type == "float32":
            # conv workspace fp32 to gm bf16
            self.data_move_gm_workspace(self.var_out_gm, self.var_workspace, self.var_ub, self.var_vconv_ub,
                                        self.var_shape_num, "round",
                                        self.var_data_each_block, self.vconv_data_each_block,
                                        dst_stride=4, src_stride=8)

    def data_move_gm_workspace(self, dst_gm, src_gm, dst_ub, src_ub,
                               element_num, round_mode,
                               dst_data_each_block, src_data_each_block, dst_stride, src_stride):
        """
        Data move between gm and workspace
        :param dst_gm: dst gm, var out gm or var workspace for bf16
        :param src_gm: src gm, updates gm or updates workspace for bf16
        :param dst_ub: ub with same dtype as dst gm
        :param src_ub: ub with same dtype as src gm
        :param element_num: num of updates/var data once in data move
        :param round_mode: round mode in data conv
        :param dst_data_each_block: dst data each block
        :param src_data_each_block: src data each block
        :param dst_stride: dst stride in data conv
        :param src_stride: src stride in data conv
        :return:
        """
        # calc the loop times of update data once to ub
        updates_loop = self.tik_instance.Scalar(name="updates_loop")
        updates_loop.set_as(element_num // self.updates_ub_size)
        # calc the burst length of updates/var
        dst_updates_burst_len = self.tik_instance.Scalar(name="dst_updates_burst_len")
        dst_updates_burst_len.set_as((self.updates_ub_size - 1) // dst_data_each_block + 1)
        src_updates_burst_len = self.tik_instance.Scalar(name="src_updates_burst_len")
        src_updates_burst_len.set_as((self.updates_ub_size - 1) // src_data_each_block + 1)

        with self.tik_instance.if_scope(updates_loop > 0):
            with self.tik_instance.for_range(0, updates_loop) as loop_index:
                self.data_move_gm_loop(dst_gm, src_gm, dst_ub, src_ub,
                                       dst_updates_burst_len, src_updates_burst_len,
                                       loop_index, round_mode, dst_stride, src_stride)
        # deal with tail num
        last_num_updates = self.tik_instance.Scalar(name="last_num_updates")
        last_num_updates.set_as(element_num % self.updates_ub_size)
        # update burst length
        dst_updates_burst_len.set_as((last_num_updates - 1) // dst_data_each_block + 1)
        src_updates_burst_len.set_as((last_num_updates - 1) // src_data_each_block + 1)

        # tail of update data
        with self.tik_instance.if_scope(last_num_updates > 0):
            self.data_move_gm_loop(dst_gm, src_gm, dst_ub, src_ub,
                                   dst_updates_burst_len, src_updates_burst_len,
                                   updates_loop, round_mode, dst_stride, src_stride)

    def data_move_gm_loop(self, dst_gm, src_gm, dst_ub, src_ub,
                          dst_updates_burst_len, src_updates_burst_len,
                          loop_index, round_mode, dst_stride, src_stride):
        """
        Data move in single loop
        :param dst_gm: dst gm, var out gm or var workspace for bf16
        :param src_gm: src gm, updates gm or updates workspace for bf16
        :param dst_ub: ub with same dtype as dst gm
        :param src_ub: ub with same dtype as src gm
        :param dst_updates_burst_len: dst burst length of updates/var
        :param src_updates_burst_len: src burst length of updates/var
        :param loop_index: loop index of loop times in updates/var data
        :param round_mode: round mode in data conv
        :param dst_stride: dst stride in data conv
        :param src_stride: src stride in data conv
        :return:
        """
        self.tik_instance.data_move(
            src_ub, src_gm[loop_index * self.updates_ub_size],
            0, 1, src_updates_burst_len, 0, 0)
        # conv between bf16/fp32
        self.data_conv(dst_ub, src_ub, [0, 0], round_mode, num=self.updates_ub_size,
                       dst_stride=dst_stride, src_stride=src_stride)
        self.tik_instance.data_move(
            dst_gm[loop_index * self.updates_ub_size], dst_ub,
            0, 1, dst_updates_burst_len, 0, 0)

    def data_muls(self, dst, src, scalar, offsets, num=0, dst_stride=8, src_stride=8):
        """
        Tik api vmuls
        """
        dst_offset, src_offset = offsets

        loop = self.tik_instance.Scalar(name="loop")
        loop.set_as(
            num // (self.max_num_one_repeat * Constant.MAX_REPEAT_TIMES))

        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst_offset = dst_offset + index * self.max_num_one_repeat * Constant.MAX_REPEAT_TIMES
                tmp_src_offset = src_offset + index * self.max_num_one_repeat * Constant.MAX_REPEAT_TIMES
                self.tik_instance.vmuls(self.max_num_one_repeat, dst[tmp_dst_offset], src[tmp_src_offset], scalar,
                                        Constant.MAX_REPEAT_TIMES, 1, 1, dst_stride, src_stride)

            dst_offset += loop * self.max_num_one_repeat * Constant.MAX_REPEAT_TIMES
            src_offset += loop * self.max_num_one_repeat * Constant.MAX_REPEAT_TIMES

        repeat_time = self.tik_instance.Scalar(name="repeat_time")
        repeat_time.set_as(
            (num % (self.max_num_one_repeat * Constant.MAX_REPEAT_TIMES)) // self.max_num_one_repeat)

        with self.tik_instance.if_scope(repeat_time > 0):
            self.tik_instance.vmuls(self.max_num_one_repeat, dst[dst_offset], src[src_offset], scalar,
                                    repeat_time, 1, 1, dst_stride, src_stride)

            dst_offset += repeat_time * self.max_num_one_repeat
            src_offset += repeat_time * self.max_num_one_repeat

        last_num = self.tik_instance.Scalar(name="last_num")
        last_num.set_as(num % self.max_num_one_repeat)

        with self.tik_instance.if_scope(last_num > 0):
            self.tik_instance.vmuls(last_num, dst[dst_offset], src[src_offset], scalar,
                                    1, 1, 1, dst_stride, src_stride)

    def data_conv(self, dst, src, offsets, mode="", num=0, dst_stride=8, src_stride=8):
        """
        Tik api vec_conv
        """
        dst_offset, src_offset = offsets

        loop = self.tik_instance.Scalar(name="loop")
        loop.set_as(
            num // (Constant.MASK_32 * Constant.MAX_REPEAT_TIMES))

        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst_offset = dst_offset + index * Constant.MASK_32 * Constant.MAX_REPEAT_TIMES
                tmp_src_offset = src_offset + index * Constant.MASK_32 * Constant.MAX_REPEAT_TIMES
                self.tik_instance.vec_conv(Constant.MASK_32, mode, dst[tmp_dst_offset], src[tmp_src_offset],
                                           Constant.MAX_REPEAT_TIMES, dst_stride, src_stride)

            dst_offset += loop * Constant.MASK_32 * Constant.MAX_REPEAT_TIMES
            src_offset += loop * Constant.MASK_32 * Constant.MAX_REPEAT_TIMES

        repeat_time = self.tik_instance.Scalar(name="repeat_time")
        repeat_time.set_as(
            (num % (Constant.MASK_32 * Constant.MAX_REPEAT_TIMES)) // Constant.MASK_32)

        with self.tik_instance.if_scope(repeat_time > 0):
            self.tik_instance.vec_conv(Constant.MASK_32, mode, dst[dst_offset], src[src_offset], repeat_time,
                                       dst_stride, src_stride)
            dst_offset += repeat_time * Constant.MASK_32
            src_offset += repeat_time * Constant.MASK_32

        last_num = self.tik_instance.Scalar(name="last_num")
        last_num.set_as(num % Constant.MASK_32)

        with self.tik_instance.if_scope(last_num > 0):
            self.tik_instance.vec_conv(last_num, mode, dst[dst_offset], src[src_offset], 1, dst_stride, src_stride)

    def data_move_pad_bit64_2_int8(self, dst, src, nburst, burst, dst_stride, src_stride,
                                   right_padding=0, left_padding=0, padding_value=None):
        if tbe_platform.api_check_support("tik.data_move_pad", self.indices_dtype):
            self.tik_instance.data_move_pad(dst, src, nburst, burst, dst_stride,
                                            src_stride, right_padding, left_padding, padding_value)
        else:
            dst_int8 = dst.reinterpret_cast_to("int8")
            src_int8 = src.reinterpret_cast_to("int8")
            self.tik_instance.data_move_pad(dst_int8, src_int8, nburst, burst, dst_stride,
                                            src_stride, right_padding, left_padding, padding_value)

    def traversing_outer_loop_per_block(self):
        """
        Traverse outer loop per block
        :return:
        """
        with self.tik_instance.if_scope(self.axis != 0):
            with self.tik_instance.for_range(0, self.outer_loops_ub_per_block) as outer_i:
                self.get_outer_loop_index_of_var((self.outer_loop_start_index_every_block + outer_i))
                self.get_outer_loop_index_of_updates((self.outer_loop_start_index_every_block + outer_i))
                self.traversing_indices()
        with self.tik_instance.else_scope():
            # calc the loop times of indices once to ub
            indices_loop_num = self.tik_instance.Scalar(name="indices_loop_num")
            indices_loop_num.set_as(self.indices_num // self.indices_ub_size)

            with self.tik_instance.if_scope(indices_loop_num > 0):
                with self.tik_instance.for_range(0, indices_loop_num) as indices_loop_index:
                    self.update_the_var_axis_0(indices_loop_index * self.indices_ub_size, self.indices_ub_size)

            indices_last_num = self.tik_instance.Scalar(name="indices_last_num")
            indices_last_num.set_as(self.indices_num % self.indices_ub_size)
            with self.tik_instance.if_scope(indices_last_num > 0):
                self.update_the_var_axis_0(indices_loop_num * self.indices_ub_size, indices_last_num)

    def traversing_outer_loop_atomic_add(self):
        """
        Traverse outer loop per block by indices using atomic add
        :return:
        """
        with self.tik_instance.if_scope(self.axis != 0):
            with self.tik_instance.for_range(0, self.outer_loop) as var_index:
                self.get_outer_loop_index_of_var(var_index)
                self.get_outer_loop_index_of_updates(var_index)
                self.traversing_indices_atomic_add()
        with self.tik_instance.else_scope():
            # calc the loop times of indices once to ub
            indices_loop_num = self.tik_instance.Scalar(name="indices_loop_num")
            indices_loop_num.set_as(self.outer_loops_ub_per_block // self.indices_ub_size)

            with self.tik_instance.if_scope(indices_loop_num > 0):
                with self.tik_instance.for_range(0, indices_loop_num) as indices_loop_index:
                    self.update_the_var_axis_0_atomic_add(
                        self.outer_loop_start_index_every_block + indices_loop_index * self.indices_ub_size,
                        self.indices_ub_size)

            # tail num of indices data
            indices_last_num = self.tik_instance.Scalar(name="indices_last_num")
            indices_last_num.set_as(self.outer_loops_ub_per_block % self.indices_ub_size)

            with self.tik_instance.if_scope(indices_last_num > 0):
                self.update_the_var_axis_0_atomic_add(
                    self.outer_loop_start_index_every_block + indices_loop_num * self.indices_ub_size,
                    indices_last_num)

    def inplace_index_add_computer_tiling(self):
        """
        Main process of inplace_index_add
        """
        self.tiling_ub = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                                  name="tiling_ub", scope=tik.scope_ubuf)
        burst_len = (Constant.TILING_ARG_NUM - 1) // Constant.NUM_EACH_BURST_INT64 + 1
        self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, burst_len, 0, 0)
        self.tiling_args()
        self.alpha_scalar = self.tik_instance.Scalar(self.vconv_dst_dtype, name="alpha_scalar", init_value=1)
        if self.is_alpha:
            with self.tik_instance.new_stmt_scope():
                alpha_ub = self.tik_instance.Tensor(
                    self.alpha_dtype, (self.var_data_each_block,),
                    name="alpha_ub",
                    scope=tik.scope_ubuf)
                # move alpha from gm to ub
                self.tik_instance.data_move(alpha_ub, self.alpha_gm, 0, 1, 1, 0, 0)
                if self.var_dtype == "bfloat16":
                    alpha_conv_ub = self.tik_instance.Tensor(
                        self.vconv_dst_dtype, (self.var_data_each_block,),
                        name="alpha_conv_ub",
                        scope=tik.scope_ubuf)
                    self.tik_instance.vconv(1, "",
                                            alpha_conv_ub,
                                            alpha_ub,
                                            1, 1, 1, 8, 4)
                    self.alpha_scalar.set_as(alpha_conv_ub[0])
                else:
                    # scalar value of alpha
                    self.alpha_scalar.set_as(alpha_ub[0])

        if self.var_dtype == "bfloat16":
            self.init_ub_tensor()
            self.data_conv_gm_workspace("bfloat16")
            self.outer_loop_start_index_every_block.set_as(0)
            self.outer_loops_ub_per_block.set_as(self.outer_loops_per_block)
            self.traversing_outer_loop_atomic_add()
            self.data_conv_gm_workspace("float32")

        elif self.support_multi_atomic_add and self.atomic_add_flag:
            with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as core_index:
                self.init_ub_tensor()
                self.outer_loop_start_index_every_block.set_as(core_index * self.outer_loops_per_block)
                self.outer_loops_ub_per_block.set_as(self.outer_loops_per_block)
                with self.tik_instance.if_scope(core_index == self.core_num - 1):
                    self.outer_loops_ub_per_block.set_as(self.tail_num)
                self.traversing_outer_loop_atomic_add()
        else:
            self.init_ub_tensor()
            self.outer_loops_ub_per_block.set_as(self.outer_loops_per_block)
            self.traversing_outer_loop_per_block()

    def inplace_index_add_operator(self):
        """
        inplace_index_add operation
        """
        self.inplace_index_add_computer_tiling()
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}

        ub_size_bytes = self.ub_size_bytes

        tbe_context.get_context().add_compile_info(
            "vars", {
                "ub_size": ub_size_bytes,
                "core_num": self.ai_core_num,
                "var_size": self.var_dtype_bytes_size,
                "var_data_each_block": self.var_data_each_block,
                "indices_size": self.indices_dtype_bytes_size,
                "indices_data_each_block": self.indices_data_each_block,
                "soc_version": self.soc_version,
                "atomic_add": self.atomic_add
            })

        if self.is_alpha:
            sch_list = [self.var_gm, self.indices_gm, self.updates_gm, self.alpha_gm]
        else:
            sch_list = [self.var_gm, self.indices_gm, self.updates_gm]
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=sch_list,
                                   outputs=(self.var_out_gm),
                                   flowtable=[self.tiling_gm],
                                   config=opt_config)

        return self.tik_instance


# 'pylint: disable=unused-argument
@register_operator("InplaceIndexAdd")
# Change REQUIRED_ATTR_INT to OPTION_ATTR_INT so that to support binary.
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def inplace_index_add(var, indices, updates, alpha, var_out, axis, kernel_name="index_add"):
    """
    inplace_index_add interface
    :param var: input var data
    :param indices: input indices
    :param updates: update data
    :param alpha: alpha to scale the update data
    :param var_out: output
    :param axis: axis to update
    :param kernel_name:
    :return: inplace index add result will return
    """
    obj = InplaceIndexAdd(var, indices, updates, alpha, var_out, axis, kernel_name)
    return obj.inplace_index_add_operator()
