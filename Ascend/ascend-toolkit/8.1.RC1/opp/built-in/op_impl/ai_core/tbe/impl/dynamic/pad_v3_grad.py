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
pad_v3_Grad
"""
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator
from impl.dynamic.replication_pad_v3_grad import replication_pad_v3_grad
from tbe.common.platform import get_bit_len


CHECK_DIMS = 5


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # max int64
    MAX_INT64 = 2**64 - 1
    MAX_INT32 = 2**32 - 1
    # tiling param nums
    TILING_NUMS = 20
    # `1 byte = 8 bit`
    EIGHT_BIT = 8
    # reserved ub size
    RESERVED_UB = 1024 * 13
    # vnchw the minest block
    TRANS_MIN_BLKS = 16
    MODE0 = 0
    MODE1 = 1
    MAX_MASK_DIC = {"float32": 64, "float16": 128}
    MAX_MASK = 0
    # the block size
    BLOCK_SIZE = 32
    WORK_SIZE = 150000
    ALIGN_16 = 16
    FLOAT32_TRANS_BLK = 16 * 8
    FLOAT32_BYTES = 4
    DST_OFFSET = 8
    DST_CYCLES = 8
    MAX_UB_SIZE = 256
    ONE_KB = 1024


# 'pylint: disable=too-many-instance-attributes,too-many-statements,too-many-locals,too-many-lines
# 'pylint: disable=too-many-arguments,invalid-name
class PadV3GradInit:
    """
    Function: class that execute pad_v3_grad
    """

    def __init__(self, x, paddings, y, mode, padding_contiguous=True, kernel_name='pad_v3_grad'):
        """
        init the op
        :param
        x: the input tensor
        :param
        paddings: the list of paddings
        :param
        y: the output of op
        :param
        kernel_name: the kernel name of op
        :return
        None
        """
        self.tik_instance = tik.Tik()
        self.tiling_dtype = "int64"
        self.x_dtype = x.get("dtype")
        self.unknown_max_shape = (Constant.MAX_INT64,)
        self.tiling_shape = (Constant.TILING_NUMS,)
        self.inner_dtype = self.x_dtype
        Constant.MAX_MASK = Constant.MAX_MASK_DIC.get(self.inner_dtype)
        if self.inner_dtype == "float32":
            Constant.WORK_SIZE = 95000
        self.y_dtype = y.get('dtype')
        self.paddings_dtype = paddings.get('dtype')
        self.kernel_name = kernel_name
        self.padding_contiguous = padding_contiguous
        self.mode = mode
        self.input_gm_list = []
        self.output_gm_list = []
        self.input_bytes_size = 0
        self.tiling_gm = None
        self.output_gm = None
        self.input_gm = None
        self.block_less_16 = None
        self.work_space = None
        self.inner_bytes_size = get_bit_len(
            self.inner_dtype) // Constant.EIGHT_BIT

        self.max_repeat_time = 255
        self.block_num = Constant.BLOCK_SIZE // self.inner_bytes_size
        self.dump_mask_max_x = Constant.EIGHT_BIT * self.block_num

        self.ub_size_bytes = tbe_platform.get_soc_spec(
            tbe_platform.UB_SIZE) - Constant.RESERVED_UB
        self.core_nums = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.workspace_shape = (Constant.WORK_SIZE * self.core_nums, )
        self.ub_number = self.ub_size_bytes // self.inner_bytes_size

        # tiling scalar init
        self.tiling_key = self.tik_instance.Scalar(
            self.tiling_dtype, "tiling_key", init_value=0)
        self.tiling_input_dim_0 = self.tik_instance.Scalar(
            self.tiling_dtype, "tiling_input_dim_0", init_value=0)
        self.tiling_input_dim_1 = self.tik_instance.Scalar(
            self.tiling_dtype, "tiling_input_dim_1", init_value=0)
        self.tiling_input_dim_2 = self.tik_instance.Scalar(
            self.tiling_dtype, "tiling_input_dim_2", init_value=0)
        self.tiling_input_dim_3 = self.tik_instance.Scalar(
            self.tiling_dtype, "tiling_input_dim_3", init_value=0)
        self.tiling_output_dim_0 = self.tik_instance.Scalar(
            self.tiling_dtype, "tiling_output_dim_0", init_value=0)
        self.tiling_output_dim_1 = self.tik_instance.Scalar(
            self.tiling_dtype, "tiling_output_dim_1", init_value=0)
        self.tiling_output_dim_2 = self.tik_instance.Scalar(
            self.tiling_dtype, "tiling_output_dim_2", init_value=0)
        self.tiling_output_dim_3 = self.tik_instance.Scalar(
            self.tiling_dtype, "tiling_output_dim_3", init_value=0)
        self.block_over_16 = None
        self.tiling_output_shape = None
        self.tiling_input_shape = None
        self.per_core_num = self.tik_instance.Scalar(
            self.tiling_dtype, "per_core_num", init_value=0)
        self.padding_index_0 = self.tik_instance.Scalar(
            self.tiling_dtype, "padding_index_0", init_value=0)
        self.padding_index_1 = self.tik_instance.Scalar(
            self.tiling_dtype, "padding_index_1", init_value=0)
        self.padding_index_2 = self.tik_instance.Scalar(
            self.tiling_dtype, "padding_index_2", init_value=0)
        self.padding_index_3 = self.tik_instance.Scalar(
            self.tiling_dtype, "padding_index_3", init_value=0)

        # core scaler init
        self.core_uesd_num = self.tik_instance.Scalar(
            self.tiling_dtype, "core_uesd_num", init_value=0)
        self.core_num_var = self.tik_instance.Scalar(
            self.tiling_dtype, "core_num_var", init_value=self.core_nums)
        self.not_last_core_num = self.tik_instance.Scalar(
            self.tiling_dtype, "not_last_core_num", init_value=0)
        self.last_core_num = self.tik_instance.Scalar(
            self.tiling_dtype, "last_core_num", init_value=0)
        self.sum_eles = self.tik_instance.Scalar("int64")

    def tiling_args(self, tiling_ub):
        """
        when input shape is less 6, will. expand to 6
        tiling_input_dim_cut_axis: which dim will be cut
        """
        with self.tik_instance.new_stmt_scope():
            self.tiling_key.set_as(tiling_ub[0])
            self.tiling_input_dim_0.set_as(tiling_ub[1])
            self.tiling_input_dim_1.set_as(tiling_ub[2])
            self.tiling_input_dim_2.set_as(tiling_ub[3])
            self.tiling_input_dim_3.set_as(tiling_ub[4])
            self.tiling_output_dim_0.set_as(tiling_ub[5])
            self.tiling_output_dim_1.set_as(tiling_ub[6])
            self.tiling_output_dim_2.set_as(tiling_ub[7])
            self.tiling_output_dim_3.set_as(tiling_ub[8])
            self.padding_index_0.set_as(tiling_ub[10])
            self.padding_index_1.set_as(tiling_ub[11])
            self.padding_index_2.set_as(tiling_ub[12])
            self.padding_index_3.set_as(tiling_ub[13])
            self.not_last_core_num.set_as(tiling_ub[14])
            self.last_core_num.set_as(tiling_ub[15])
            self.core_num_var.set_as(tiling_ub[16])
        self.sum_eles.set_as(self.tiling_input_dim_0 * self.tiling_input_dim_1 *
                             self.tiling_input_dim_2 * self.tiling_input_dim_3)

    # 'pylint:disable=unused-argument
    def init_src_dst_gm(self, input_dict_list, output_dict_list, pad_input_idx=0, pad_outnput_idx=0):
        """
        init gm tensor set tiling, input, paddings output tensor(gm)
        :param
        input_dict_list: the dict of input_dict
        :param
        output_dict_list: output_dict_list
        :param
        pad_input_idx: pad_input_idx
        :param
        pad_outnput_idx: pad_outnput_idx
        :return:
        None
        """
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype,
                                                  self.tiling_shape,
                                                  name="tiling_gm",
                                                  scope=tik.scope_gm)
        self.work_space = self.tik_instance.Tensor(self.inner_dtype,
                                                   self.workspace_shape,
                                                   name="work_space",
                                                   scope=tik.scope_gm,
                                                   is_workspace=True)
        x_dtype = input_dict_list[0].get("dtype")
        paddings_dtype = input_dict_list[1].get("dtype")
        x_gm = self.tik_instance.Tensor(
            self.inner_dtype, self.unknown_max_shape, name="x", scope=tik.scope_gm)
        paddings_gm = self.tik_instance.Tensor(paddings_dtype,
                                               self.unknown_max_shape,
                                               name="paddings",
                                               scope=tik.scope_gm)
        self.input_gm_list.append(x_gm)
        self.input_gm_list.append(paddings_gm)

        y_gm = self.tik_instance.Tensor(
            self.inner_dtype, self.unknown_max_shape, name="y", scope=tik.scope_gm)
        self.input_bytes_size = get_bit_len(x_dtype) // Constant.EIGHT_BIT
        self.output_gm_list.append(y_gm)

        self.output_gm = self.output_gm_list[pad_outnput_idx]
        self.input_gm = self.input_gm_list[pad_input_idx]

    def pad_v3_grad_compute_tiling(self):
        """
        pad_v3_grad_compute_tiling
        """
        tiling_ub = self.tik_instance.Tensor(
            "int64", (Constant.TILING_NUMS,), name="tiling_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(
            tiling_ub, self.tiling_gm, 0, 1, Constant.TILING_NUMS // 4, 0, 0)
        self.core_uesd_num.set_as(tiling_ub[9])
        self.tiling_args(tiling_ub)
        with self.tik_instance.for_range(0, self.core_num_var, block_num=self.core_num_var) as core_index:
            with self.tik_instance.if_scope(core_index < self.core_uesd_num):
                self.do_pad(core_index)

    def do_tiling_key_mode_0(self, core_index):
        """
        do_tiling_key_mode_0 when tiling key = 0
        """
        ub_size_second_mode = 2
        per_ub_size = self.ub_size_bytes // self.inner_bytes_size // ub_size_second_mode
        align_input_dim_2 = self.tik_instance.Scalar(
            'int64', name='align_output_dim_2')
        align_input_dim_3 = self.tik_instance.Scalar(
            'int64', name='align_output_dim_3')
        time_2 = self.tik_instance.Scalar('int64', name='time_2')
        time_3 = self.tik_instance.Scalar('int64', name='time_3')
        input_ele_per_core = self.tik_instance.Scalar(
            'int64', name='input_ele_per_core')
        output_ele_per_core = self.tik_instance.Scalar(
            'int64', name='output_ele_per_core')
        ranges = self.tik_instance.Scalar('int64', name='ranges')
        flag = self.tik_instance.Scalar('int64', name='flag')
        units = self.tik_instance.Scalar('int64', name='units')
        first_ub_not_complete_offset = self.tik_instance.Scalar(
            'int64', name='first_ub_not_complete_offset')
        first_ub_need_first_move_lines = self.tik_instance.Scalar(
            'int64', name='first_ub_need_first_move_lines')
        gm_offset = self.tik_instance.Scalar('int64', name='gm_offset')
        first_ub_first_offset = self.tik_instance.Scalar(
            'int64', name='first_ub_first_offset')
        first_ub_need_last_move_lines = self.tik_instance.Scalar(
            'int64', name='first_ub_need_last_move_lines')
        align_input_dim_2.set_as(
            ((self.tiling_input_dim_2 - 1) // Constant.ALIGN_16 + 1) * Constant.ALIGN_16)
        align_input_dim_3.set_as(
            ((self.tiling_input_dim_3 - 1) // Constant.ALIGN_16 + 1) * Constant.ALIGN_16)
        time_2.set_as(align_input_dim_2 // Constant.TRANS_MIN_BLKS)
        time_3.set_as(align_input_dim_3 // Constant.TRANS_MIN_BLKS)
        units.set_as((Constant.TRANS_MIN_BLKS - 1) //
                     (self.tiling_output_dim_2 * self.tiling_output_dim_3) + 1)
        first_ub_need_first_move_lines.set_as(units * self.tiling_output_dim_2 -
                                              Constant.TRANS_MIN_BLKS // self.tiling_output_dim_3)
        first_ub_not_complete_offset.set_as(self.tiling_output_dim_3 - (
            (Constant.TRANS_MIN_BLKS -
             (units - 1) * self.tiling_output_dim_3 * self.tiling_output_dim_2) % self.tiling_output_dim_3))
        first_ub_need_last_move_lines.set_as(
            (Constant.TRANS_MIN_BLKS -
             (units - 1) * self.tiling_output_dim_3 * self.tiling_output_dim_2) // self.tiling_output_dim_3)
        first_ub_first_offset.set_as((first_ub_need_first_move_lines - 1) * align_input_dim_3 +
                                     first_ub_not_complete_offset)

        input_ele_per_core.set_as(
            self.not_last_core_num * self.tiling_input_dim_2 * self.tiling_input_dim_3)
        output_ele_per_core.set_as(
            self.not_last_core_num * self.tiling_output_dim_2 * self.tiling_output_dim_3)
        with self.tik_instance.if_scope(core_index == self.core_uesd_num - 1):
            ranges.set_as(self.last_core_num)
        with self.tik_instance.else_scope():
            ranges.set_as(self.not_last_core_num)
            flag.set_as((output_ele_per_core - Constant.TRANS_MIN_BLKS) //
                        (self.tiling_output_dim_3 * self.tiling_output_dim_2))
            gm_offset.set_as((core_index + 1) *
                             output_ele_per_core - Constant.TRANS_MIN_BLKS)

        with self.tik_instance.new_stmt_scope():
            ping_ub_1 = self.tik_instance.Tensor(self.inner_dtype, (per_ub_size,),
                                                 name='ping_ub_1',
                                                 scope=tik.scope_ubuf)
            pang_ub_1 = self.tik_instance.Tensor(self.inner_dtype, (per_ub_size,),
                                                 name='pang_ub_1',
                                                 scope=tik.scope_ubuf)
            self.block_less_16 = self.tik_instance.Tensor(self.inner_dtype, (Constant.TRANS_MIN_BLKS,),
                                                          name='block_less_16',
                                                          scope=tik.scope_ubuf)
            self.block_over_16 = self.tik_instance.Tensor(self.inner_dtype, (Constant.TRANS_MIN_BLKS,),
                                                          name='block_over_16',
                                                          scope=tik.scope_ubuf)
            loop = self.tik_instance.Scalar('int64', name='loop')
            remain_mask = self.tik_instance.Scalar('int64', name='remain_mask')
            with self.tik_instance.for_range(0, ranges) as index:
                loop.set_as(align_input_dim_3 // Constant.MAX_MASK)
                remain_mask.set_as(align_input_dim_3 % Constant.MAX_MASK)
                with self.tik_instance.for_range(0, self.tiling_input_dim_2) as i:
                    if tbe_platform.api_check_support("tik.data_move_pad"):
                        self.tik_instance.data_move_pad(ping_ub_1[i * align_input_dim_3],
                                                        self.input_gm[core_index * input_ele_per_core +
                                                            index * self.tiling_input_dim_2 * self.tiling_input_dim_3 +
                                                            i * self.tiling_input_dim_3], 
                                                        1, 
                                                        self.tiling_input_dim_3 * self.inner_bytes_size, 
                                                        0, 0, 0, 0, None)
                    else:
                        self.tik_instance.data_move(
                        ping_ub_1[i * align_input_dim_3],
                        self.input_gm[core_index * input_ele_per_core +
                                      index * self.tiling_input_dim_2 * self.tiling_input_dim_3 +
                                      i * self.tiling_input_dim_3], 0, 1, align_input_dim_3 // self.block_num, 0, 0)

                with self.tik_instance.for_range(0, self.padding_index_2) as i:
                    self.tik_instance.vec_add(
                        Constant.MAX_MASK,
                        ping_ub_1[(self.padding_index_2 +
                                   self.padding_index_2 - i) * align_input_dim_3],
                        ping_ub_1[(self.padding_index_2 +
                                   self.padding_index_2 - i) * align_input_dim_3],
                        ping_ub_1[i * align_input_dim_3], loop, 8, 8, 8)
                    with self.tik_instance.if_scope(remain_mask > 0):
                        self.tik_instance.vec_add(
                            remain_mask,
                            ping_ub_1[(self.padding_index_2 + self.padding_index_2 - i) * align_input_dim_3 +
                                      loop * Constant.MAX_MASK],
                            ping_ub_1[(self.padding_index_2 + self.padding_index_2 - i) * align_input_dim_3 +
                                      loop * Constant.MAX_MASK],
                            ping_ub_1[i * align_input_dim_3 + loop * Constant.MAX_MASK], 1, 8, 8, 8)

                with self.tik_instance.for_range(0, self.padding_index_3) as i:
                    self.tik_instance.vec_add(
                        Constant.MAX_MASK,
                        ping_ub_1[(self.tiling_input_dim_2 - 1 -
                                   self.padding_index_3 * 2 + i) * align_input_dim_3],
                        ping_ub_1[(self.tiling_input_dim_2 - 1 -
                                   self.padding_index_3 * 2 + i) * align_input_dim_3],
                        ping_ub_1[(self.tiling_input_dim_2 - 1 - i) * align_input_dim_3], loop, 8, 8, 8)
                    with self.tik_instance.if_scope(remain_mask > 0):
                        self.tik_instance.vec_add(
                            remain_mask,
                            ping_ub_1[(self.tiling_input_dim_2 - 1 - self.padding_index_3 * 2 + i) * align_input_dim_3 +
                                      loop * Constant.MAX_MASK],
                            ping_ub_1[(self.tiling_input_dim_2 - 1 - self.padding_index_3 * 2 + i) * align_input_dim_3 +
                                      loop * Constant.MAX_MASK],
                            ping_ub_1[(self.tiling_input_dim_2 - 1 - i) *
                                      align_input_dim_3 + loop * Constant.MAX_MASK],
                            1, 8, 8, 8)

                with self.tik_instance.for_range(0, time_2) as i:
                    with self.tik_instance.for_range(0, time_3) as j:
                        src_list = []
                        dst_list = []
                        if self.inner_dtype == "float16":
                            for k in range(Constant.TRANS_MIN_BLKS):
                                src_list.append(
                                    ping_ub_1[time_3 * Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * i +
                                              Constant.TRANS_MIN_BLKS * j + time_3 * Constant.TRANS_MIN_BLKS *
                                              (k + self.padding_index_2)])
                                dst_list.append(
                                    pang_ub_1[time_2 * Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * j +
                                              Constant.TRANS_MIN_BLKS * i + time_2 * Constant.TRANS_MIN_BLKS * k])
                            self.tik_instance.vnchwconv(
                                True, False, dst_list, src_list, 1, 0, 0)

                        elif self.inner_dtype == "float32":
                            src_stride = 1
                            dst_stride = time_2 * Constant.TRANS_MIN_BLKS * self.block_num // self.block_num
                            for k in range(Constant.TRANS_MIN_BLKS):
                                src_list.append(
                                    ping_ub_1[time_3 * Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * i +
                                              Constant.TRANS_MIN_BLKS * j + time_3 * Constant.TRANS_MIN_BLKS *
                                              (k + self.padding_index_2)])
                            for k in range(Constant.DST_CYCLES):
                                dst_list.append(
                                    pang_ub_1[time_2 * Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * j +
                                              Constant.TRANS_MIN_BLKS * i + time_2 * Constant.TRANS_MIN_BLKS * k])
                                dst_list.append(
                                    pang_ub_1[time_2 * Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * j +
                                              Constant.TRANS_MIN_BLKS * i + time_2 * Constant.TRANS_MIN_BLKS * k
                                              + Constant.DST_OFFSET])
                            self.tik_instance.vnchwconv(
                                True, False, dst_list, src_list, 2, dst_stride, src_stride)

                loop.set_as(align_input_dim_2 // Constant.MAX_MASK)
                remain_mask.set_as(align_input_dim_2 % Constant.MAX_MASK)
                with self.tik_instance.for_range(0, self.padding_index_0) as i:
                    self.tik_instance.vec_add(
                        Constant.MAX_MASK,
                        pang_ub_1[(self.padding_index_0 +
                                   self.padding_index_0 - i) * align_input_dim_2],
                        pang_ub_1[(self.padding_index_0 +
                                   self.padding_index_0 - i) * align_input_dim_2],
                        pang_ub_1[i * align_input_dim_2], loop, 8, 8, 8)
                    with self.tik_instance.if_scope(remain_mask > 0):
                        self.tik_instance.vec_add(
                            remain_mask,
                            pang_ub_1[(self.padding_index_0 + self.padding_index_0 - i) * align_input_dim_2 +
                                      Constant.MAX_MASK * loop],
                            pang_ub_1[(self.padding_index_0 + self.padding_index_0 - i) * align_input_dim_2 +
                                      Constant.MAX_MASK * loop],
                            pang_ub_1[i * align_input_dim_2 + Constant.MAX_MASK * loop], 1, 8, 8, 8)

                with self.tik_instance.for_range(0, self.padding_index_1) as i:
                    self.tik_instance.vec_add(
                        Constant.MAX_MASK,
                        pang_ub_1[(self.tiling_input_dim_3 - 1 -
                                   self.padding_index_1 * 2 + i) * align_input_dim_2],
                        pang_ub_1[(self.tiling_input_dim_3 - 1 -
                                   self.padding_index_1 * 2 + i) * align_input_dim_2],
                        pang_ub_1[(self.tiling_input_dim_3 - 1 - i) * align_input_dim_2], loop, 8, 8, 8)
                    with self.tik_instance.if_scope(remain_mask > 0):
                        self.tik_instance.vec_add(
                            remain_mask,
                            pang_ub_1[(self.tiling_input_dim_3 - 1 - self.padding_index_1 * 2 + i) * align_input_dim_2 +
                                      Constant.MAX_MASK * loop],
                            pang_ub_1[(self.tiling_input_dim_3 - 1 - self.padding_index_1 * 2 + i) * align_input_dim_2 +
                                      Constant.MAX_MASK * loop],
                            pang_ub_1[(self.tiling_input_dim_3 - 1 - i) *
                                      align_input_dim_2 + Constant.MAX_MASK * loop],
                            1, 8, 8, 8)

                with self.tik_instance.for_range(0, time_2) as i:
                    with self.tik_instance.for_range(0, time_3) as j:
                        src_list = []
                        dst_list = []
                        if self.inner_dtype == "float16":
                            for k in range(Constant.TRANS_MIN_BLKS):
                                src_list.append(
                                    pang_ub_1[time_2 * Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * j +
                                              Constant.TRANS_MIN_BLKS * i + time_2 * Constant.TRANS_MIN_BLKS *
                                              (k + self.padding_index_0)])
                                dst_list.append(
                                    ping_ub_1[time_3 * Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * i +
                                              Constant.TRANS_MIN_BLKS * j + time_3 * Constant.TRANS_MIN_BLKS * k])
                            self.tik_instance.vnchwconv(
                                True, False, dst_list, src_list, 1, 0, 0)
                        elif self.inner_dtype == "float32":
                            src_stride = 1
                            dst_stride = time_3 * Constant.TRANS_MIN_BLKS * self.block_num // self.block_num
                            for k in range(Constant.TRANS_MIN_BLKS):
                                src_list.append(
                                    pang_ub_1[time_2 * Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * j +
                                              Constant.TRANS_MIN_BLKS * i + time_2 * Constant.TRANS_MIN_BLKS *
                                              (k + self.padding_index_0)])
                            for k in range(Constant.DST_CYCLES):
                                dst_list.append(
                                    ping_ub_1[time_3 * Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * i +
                                              Constant.TRANS_MIN_BLKS * j + time_3 * Constant.TRANS_MIN_BLKS * k])
                                dst_list.append(ping_ub_1[time_3 * Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * i
                                                + Constant.TRANS_MIN_BLKS * j + time_3 * Constant.TRANS_MIN_BLKS * k
                                                + Constant.DST_OFFSET])
                            self.tik_instance.vnchwconv(
                                True, False, dst_list, src_list, 2, dst_stride, src_stride)

                align_output_dim_3 = self.tik_instance.Scalar(
                    'int64', name='align_output_dim_3')
                align_output_dim_3.set_as(
                    ((self.tiling_output_dim_3 - 1) // Constant.ALIGN_16 + 1) * Constant.ALIGN_16)
                with self.tik_instance.if_scope(core_index == self.core_uesd_num - 1):
                    with self.tik_instance.for_range(0, self.tiling_output_dim_2) as i:
                        if tbe_platform.api_check_support("tik.data_move_pad"):
                            self.tik_instance.data_move_pad(self.output_gm[core_index * output_ele_per_core +
                                            index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                            i * self.tiling_output_dim_3], ping_ub_1[i * align_input_dim_3], 
                                                            1, 
                                                            self.tiling_output_dim_3 * self.inner_bytes_size, 
                                                            0, 0, 0, 0, None)
                        else:
                            self.tik_instance.data_move(
                                self.output_gm[core_index * output_ele_per_core +
                                            index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                            i * self.tiling_output_dim_3], ping_ub_1[i * align_input_dim_3], 0, 1,
                                align_output_dim_3 // self.block_num, 0, 0)

                with self.tik_instance.if_scope(tik.all(self.core_uesd_num > 1, core_index < self.core_uesd_num - 1)):
                    with self.tik_instance.if_scope(self.tiling_output_dim_3 <= Constant.TRANS_MIN_BLKS):
                        with self.tik_instance.if_scope(index < flag):
                            with self.tik_instance.for_range(0, self.tiling_output_dim_2) as i:
                                self.tik_instance.data_move(
                                    self.output_gm[core_index * output_ele_per_core +
                                                   index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                                   i * self.tiling_output_dim_3], ping_ub_1[i * align_input_dim_3], 0,
                                    1, align_output_dim_3 // self.block_num, 0, 0)

                        with self.tik_instance.if_scope(index == flag):
                            with self.tik_instance.for_range(0, first_ub_need_first_move_lines) as i:
                                self.tik_instance.data_move(
                                    self.output_gm[core_index * output_ele_per_core +
                                                   index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                                   i * self.tiling_output_dim_3], ping_ub_1[i * align_input_dim_3], 0,
                                    1, Constant.TRANS_MIN_BLKS // self.block_num, 0, 0)

                            with self.tik_instance.for_range(first_ub_first_offset,
                                                             (first_ub_need_first_move_lines - 1) * align_input_dim_3 +
                                                             self.tiling_output_dim_3) as i:
                                self.block_less_16[i -
                                                   first_ub_first_offset].set_as(ping_ub_1[i])

                            with self.tik_instance.for_range(first_ub_need_first_move_lines,
                                                             self.tiling_output_dim_2) as i:
                                with self.tik_instance.for_range(0, self.tiling_output_dim_3) as j:
                                    self.block_less_16[(first_ub_need_first_move_lines - 1) * align_input_dim_3 +
                                                       self.tiling_output_dim_3 - first_ub_first_offset +
                                                       (i - first_ub_need_first_move_lines) * self.tiling_output_dim_3 +
                                                       j].set_as(ping_ub_1[i * align_input_dim_3 + j])
                            with self.tik_instance.if_scope(flag == ranges - 1):
                                self.tik_instance.data_move(self.output_gm[gm_offset], self.block_less_16, 0, 1,
                                                            Constant.TRANS_MIN_BLKS // self.block_num, 0,
                                                            0)

                        with self.tik_instance.if_scope(index > flag):
                            with self.tik_instance.for_range(0, self.tiling_output_dim_2) as i:
                                with self.tik_instance.for_range(0, self.tiling_output_dim_3) as j:
                                    self.block_less_16[(index - flag - 1) * self.tiling_output_dim_3 *
                                                       self.tiling_output_dim_2 +
                                                       (first_ub_need_first_move_lines - 1) * align_output_dim_3 +
                                                       self.tiling_output_dim_3 - first_ub_first_offset +
                                                       first_ub_need_last_move_lines * self.tiling_output_dim_3 +
                                                       i * self.tiling_output_dim_3 + j].set_as(
                                        ping_ub_1[i * align_input_dim_3 + j])
                            with self.tik_instance.if_scope(index == ranges - 1):
                                self.tik_instance.data_move(self.output_gm[gm_offset], self.block_less_16, 0, 1,
                                                            Constant.TRANS_MIN_BLKS // self.block_num, 0,
                                                            0)

                    with self.tik_instance.if_scope(self.tiling_output_dim_3 > Constant.TRANS_MIN_BLKS):
                        with self.tik_instance.if_scope(index == ranges - 1):
                            with self.tik_instance.for_range(0, self.tiling_output_dim_2) as i:
                                with self.tik_instance.if_scope(i == self.tiling_output_dim_2 - 1):
                                    self.tik_instance.data_move(
                                        self.output_gm[core_index * output_ele_per_core +
                                                       index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                                       i * self.tiling_output_dim_3], ping_ub_1[i * align_input_dim_3],
                                        0, 1, self.tiling_output_dim_3 // self.block_num, 0, 0)
                                    with self.tik_instance.for_range(
                                        (self.tiling_output_dim_3 // Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS -
                                         (Constant.TRANS_MIN_BLKS -
                                          self.tiling_output_dim_3 % Constant.TRANS_MIN_BLKS)),
                                            self.tiling_output_dim_3) as j:

                                        self.block_over_16[j - (self.tiling_output_dim_3 // Constant.TRANS_MIN_BLKS *
                                                                Constant.TRANS_MIN_BLKS -
                                                                (Constant.TRANS_MIN_BLKS - self.tiling_output_dim_3 %
                                                                 Constant.TRANS_MIN_BLKS))].set_as(
                                            ping_ub_1[i * align_input_dim_3 + j])
                                    self.tik_instance.data_move(
                                        self.output_gm[core_index * output_ele_per_core +
                                                       index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                                       i * self.tiling_output_dim_3 +
                                                       (self.tiling_output_dim_3 // Constant.TRANS_MIN_BLKS *
                                                        Constant.TRANS_MIN_BLKS -
                                                        (Constant.TRANS_MIN_BLKS -
                                                         self.tiling_output_dim_3 % Constant.TRANS_MIN_BLKS))],
                                        self.block_over_16, 0, 1, Constant.TRANS_MIN_BLKS // self.block_num, 0, 0)

                                with self.tik_instance.else_scope():
                                    self.tik_instance.data_move(
                                        self.output_gm[core_index * output_ele_per_core +
                                                       index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                                       i * self.tiling_output_dim_3], ping_ub_1[i * align_input_dim_3],
                                        0, 1, align_output_dim_3 // self.block_num, 0, 0)

                        with self.tik_instance.else_scope():
                            with self.tik_instance.for_range(0, self.tiling_output_dim_2) as i:
                                self.tik_instance.data_move(
                                    self.output_gm[core_index * output_ele_per_core +
                                                   index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                                   i * self.tiling_output_dim_3], ping_ub_1[i * align_input_dim_3], 0,
                                    1, align_output_dim_3 // self.block_num, 0, 0)

    def do_tiling_key_mode_1(self, core_index):
        """
        do_tiling_key_mode_1 when tiling key = 1
        """
        ub_size_second_mode = 4
        per_ub_size = self.ub_size_bytes // self.inner_bytes_size // ub_size_second_mode
        input_ele_per_core = self.tik_instance.Scalar(
            'int64', name='input_ele_per_core')
        output_ele_per_core = self.tik_instance.Scalar(
            'int64', name='output_ele_per_core')
        ranges = self.tik_instance.Scalar('int64', name='ranges')
        input_ele_per_core.set_as(
            self.not_last_core_num * self.tiling_input_dim_2 * self.tiling_input_dim_3)
        output_ele_per_core.set_as(
            self.not_last_core_num * self.tiling_output_dim_2 * self.tiling_output_dim_3)
        with self.tik_instance.if_scope(core_index == self.core_uesd_num - 1):
            ranges.set_as(self.last_core_num)
        with self.tik_instance.else_scope():
            ranges.set_as(self.not_last_core_num)

        with self.tik_instance.new_stmt_scope():
            ping_ub_1 = self.tik_instance.Tensor(self.inner_dtype, (per_ub_size,),
                                                 name='ping_ub_1',
                                                 scope=tik.scope_ubuf)
            ping_ub_2 = self.tik_instance.Tensor(self.inner_dtype, (per_ub_size,),
                                                 name='ping_ub_2',
                                                 scope=tik.scope_ubuf)
            pang_ub_1 = self.tik_instance.Tensor(self.inner_dtype, (per_ub_size,),
                                                 name='pang_ub_1',
                                                 scope=tik.scope_ubuf)
            pang_ub_2 = self.tik_instance.Tensor(self.inner_dtype, (per_ub_size,),
                                                 name='pang_ub_2',
                                                 scope=tik.scope_ubuf)
            block_ub = self.tik_instance.Tensor(self.inner_dtype, (Constant.TRANS_MIN_BLKS,),
                                                name='block_ub',
                                                scope=tik.scope_ubuf)

            align_input_dim_3 = self.tik_instance.Scalar(
                'int64', 'align_out_dim_3')
            align_input_dim_3.set_as(
                ((self.tiling_input_dim_3 - 1) // Constant.ALIGN_16 + 1) * Constant.ALIGN_16)
            time = self.tik_instance.Scalar('int64', name='time')
            time.set_as(align_input_dim_3 // Constant.TRANS_MIN_BLKS)

            loop = self.tik_instance.Scalar('int64', name='loop')
            remain_mask = self.tik_instance.Scalar('int64', name='remain_mask')
            loop.set_as(align_input_dim_3 // Constant.MAX_MASK)
            remain_mask.set_as(align_input_dim_3 % Constant.MAX_MASK)
            with self.tik_instance.for_range(0, ranges) as index:
                with self.tik_instance.for_range(0, Constant.TRANS_MIN_BLKS) as i:
                    self.tik_instance.data_move(
                        ping_ub_1[i * align_input_dim_3],
                        self.input_gm[core_index * input_ele_per_core +
                                      index * self.tiling_input_dim_2 * self.tiling_input_dim_3 +
                                      i * self.tiling_input_dim_3], 0, 1, align_input_dim_3 // self.block_num, 0, 0)

                with self.tik_instance.for_range(0, self.padding_index_2) as i:
                    self.tik_instance.vec_add(
                        Constant.MAX_MASK,
                        ping_ub_1[(self.padding_index_2 +
                                   self.padding_index_2 - i) * align_input_dim_3],
                        ping_ub_1[(self.padding_index_2 +
                                   self.padding_index_2 - i) * align_input_dim_3],
                        ping_ub_1[i * align_input_dim_3], loop, 8, 8, 8)
                    with self.tik_instance.if_scope(remain_mask > 0):
                        self.tik_instance.vec_add(
                            remain_mask,
                            ping_ub_1[(self.padding_index_2 + self.padding_index_2 - i) * align_input_dim_3 +
                                      loop * Constant.MAX_MASK],
                            ping_ub_1[(self.padding_index_2 + self.padding_index_2 - i) * align_input_dim_3 +
                                      loop * Constant.MAX_MASK],
                            ping_ub_1[i * align_input_dim_3 + loop * Constant.MAX_MASK], 1, 8, 8, 8)

                with self.tik_instance.for_range(0, Constant.TRANS_MIN_BLKS) as i:
                    self.tik_instance.data_move(
                        ping_ub_1[(Constant.TRANS_MIN_BLKS + i)
                                  * align_input_dim_3],
                        self.input_gm[core_index * input_ele_per_core +
                                      index * self.tiling_input_dim_2 * self.tiling_input_dim_3 +
                                      (self.tiling_input_dim_2 - Constant.TRANS_MIN_BLKS + i) *
                                      self.tiling_input_dim_3], 0, 1, align_input_dim_3 // self.block_num, 0, 0)

                with self.tik_instance.for_range(0, self.padding_index_3) as i:
                    self.tik_instance.vec_add(
                        Constant.MAX_MASK,
                        ping_ub_1[Constant.TRANS_MIN_BLKS * align_input_dim_3 +
                                  (Constant.TRANS_MIN_BLKS - 1 - self.padding_index_3 * 2 + i) * align_input_dim_3],
                        ping_ub_1[Constant.TRANS_MIN_BLKS * align_input_dim_3 +
                                  (Constant.TRANS_MIN_BLKS - 1 - self.padding_index_3 * 2 + i) * align_input_dim_3],
                        ping_ub_1[Constant.TRANS_MIN_BLKS * align_input_dim_3 +
                                  (Constant.TRANS_MIN_BLKS - 1 - i) * align_input_dim_3], loop, 8, 8, 8)
                    with self.tik_instance.if_scope(remain_mask > 0):
                        self.tik_instance.vec_add(
                            remain_mask,
                            ping_ub_1[Constant.TRANS_MIN_BLKS * align_input_dim_3 +
                                      (Constant.TRANS_MIN_BLKS - 1 - self.padding_index_3 * 2 + i) * align_input_dim_3 +
                                      loop * Constant.MAX_MASK],
                            ping_ub_1[Constant.TRANS_MIN_BLKS * align_input_dim_3 +
                                      (Constant.TRANS_MIN_BLKS - 1 - self.padding_index_3 * 2 + i) * align_input_dim_3 +
                                      loop * Constant.MAX_MASK],
                            ping_ub_1[Constant.TRANS_MIN_BLKS * align_input_dim_3 +
                                      (Constant.TRANS_MIN_BLKS - 1 - i) * align_input_dim_3 + loop * Constant.MAX_MASK],
                            1, 8, 8, 8)
                with self.tik_instance.for_range(self.padding_index_2, Constant.TRANS_MIN_BLKS) as i:
                    self.tik_instance.data_move(ping_ub_2[(i - self.padding_index_2) * Constant.TRANS_MIN_BLKS],
                                                ping_ub_1[i *
                                                          align_input_dim_3], 0, 1,
                                                Constant.TRANS_MIN_BLKS // self.block_num, 0, 0)
                with self.tik_instance.for_range(Constant.TRANS_MIN_BLKS,
                                                 self.tiling_input_dim_2 - Constant.TRANS_MIN_BLKS) as i:
                    self.tik_instance.data_move(
                        ping_ub_2[(i - self.padding_index_2)
                                  * Constant.TRANS_MIN_BLKS],
                        self.input_gm[core_index * input_ele_per_core +
                                      index * self.tiling_input_dim_2 * self.tiling_input_dim_3 +
                                      i * self.tiling_input_dim_3], 0, 1, Constant.TRANS_MIN_BLKS // self.block_num, 0,
                        0)

                with self.tik_instance.for_range(0, Constant.TRANS_MIN_BLKS - self.padding_index_3) as i:
                    self.tik_instance.data_move(
                        ping_ub_2[(i + self.tiling_output_dim_2 - Constant.TRANS_MIN_BLKS + self.padding_index_3) *
                                  Constant.TRANS_MIN_BLKS],

                        ping_ub_1[(Constant.TRANS_MIN_BLKS + i)
                                  * align_input_dim_3], 0, 1,
                        Constant.TRANS_MIN_BLKS // self.block_num, 0, 0)

                align_inter = self.tik_instance.Scalar(
                    'int64', name='align_inter')
                align_inter.set_as(
                    ((self.tiling_output_dim_2 - 1) // Constant.ALIGN_16 + 1) * Constant.ALIGN_16)
                time = self.tik_instance.Scalar('int64', name='time')
                time.set_as(align_inter // Constant.ALIGN_16)
                if self.inner_dtype == "float32":
                    float32_trans_block = Constant.FLOAT32_TRANS_BLK
                    src_stride = 1
                    dst_stride = time * float32_trans_block * \
                        Constant.FLOAT32_BYTES // Constant.BLOCK_SIZE
                with self.tik_instance.for_range(0, time) as i:
                    src_list = []
                    dst_list = []
                    for j in range(self.block_num):
                        if self.inner_dtype == "float32":
                            src_list.append(ping_ub_2[Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * i +
                                                      Constant.TRANS_MIN_BLKS * (2 * j)])
                            src_list.append(ping_ub_2[Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * i +
                                                      Constant.TRANS_MIN_BLKS * (2 * j + 1)])
                            dst_list.append(
                                pang_ub_1[time * Constant.TRANS_MIN_BLKS * j + Constant.TRANS_MIN_BLKS * i])
                            dst_list.append(pang_ub_1[time * Constant.TRANS_MIN_BLKS * j +
                                                      Constant.TRANS_MIN_BLKS * i + 8])
                        else:
                            src_list.append(ping_ub_2[Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * i +
                                                      Constant.TRANS_MIN_BLKS * j])
                            dst_list.append(
                                pang_ub_1[time * Constant.TRANS_MIN_BLKS * j + Constant.TRANS_MIN_BLKS * i])
                    if self.inner_dtype == "float32":
                        self.tik_instance.vnchwconv(
                            True, False, dst_list, src_list, 2, dst_stride, src_stride)
                    else:
                        self.tik_instance.vnchwconv(
                            True, False, dst_list, src_list, 1, 0, 0)

                loop_col = self.tik_instance.Scalar('int64', name='loop_col')
                remain_mask_col = self.tik_instance.Scalar(
                    'int64', name='remain_mask_col')
                loop_col.set_as(align_inter // Constant.MAX_MASK)
                remain_mask_col.set_as(align_inter % Constant.MAX_MASK)
                with self.tik_instance.for_range(0, self.padding_index_0) as i:
                    self.tik_instance.vec_add(
                        Constant.MAX_MASK, pang_ub_1[(
                            self.padding_index_0 + self.padding_index_0 - i) * align_inter],
                        pang_ub_1[(self.padding_index_0 +
                                   self.padding_index_0 - i) * align_inter],
                        pang_ub_1[i * align_inter], loop_col, 8, 8, 8)
                    with self.tik_instance.if_scope(remain_mask_col > 0):
                        self.tik_instance.vec_add(
                            remain_mask_col, pang_ub_1[(self.padding_index_0 + self.padding_index_0 - i) * align_inter +
                                                       loop_col * Constant.MAX_MASK],
                            pang_ub_1[(self.padding_index_0 + self.padding_index_0 - i) * align_inter +
                                      loop_col * Constant.MAX_MASK],
                            pang_ub_1[i * align_inter + loop_col * Constant.MAX_MASK], 1, 8, 8, 8)
                if self.inner_dtype == "float32":
                    float32_trans_block = Constant.FLOAT32_TRANS_BLK
                    src_stride = 1
                    dst_stride = float32_trans_block * Constant.FLOAT32_BYTES // Constant.BLOCK_SIZE
                with self.tik_instance.for_range(0, time) as i:
                    src_list = []
                    dst_list = []
                    for j in range(self.block_num):
                        if self.inner_dtype == "float32":
                            dst_list.append(ping_ub_2[Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * i +
                                                      Constant.TRANS_MIN_BLKS * j])
                            dst_list.append(ping_ub_2[Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * i +
                                                      Constant.TRANS_MIN_BLKS * j + 8])
                            src_list.append(pang_ub_1[time * Constant.TRANS_MIN_BLKS * (2 * j + self.padding_index_0) +
                                                      Constant.TRANS_MIN_BLKS * i])
                            src_list.append(pang_ub_1[time * Constant.TRANS_MIN_BLKS *
                                                      (2 * j + 1 + self.padding_index_0) + Constant.TRANS_MIN_BLKS * i])
                        else:
                            dst_list.append(ping_ub_2[Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * i +
                                                      Constant.TRANS_MIN_BLKS * j])
                            src_list.append(pang_ub_1[time * Constant.TRANS_MIN_BLKS * (j + self.padding_index_0) +
                                                      Constant.TRANS_MIN_BLKS * i])
                    if self.inner_dtype == "float32":
                        self.tik_instance.vnchwconv(
                            True, False, dst_list, src_list, 2, dst_stride, src_stride)
                    else:
                        self.tik_instance.vnchwconv(
                            True, False, dst_list, src_list, 1, 0, 0)

                self.tik_instance.data_move(self.work_space[core_index * Constant.WORK_SIZE], ping_ub_1, 0, 1,
                                            Constant.TRANS_MIN_BLKS * 2 * align_input_dim_3 // self.block_num, 0, 0)

                with self.tik_instance.for_range(self.padding_index_2, Constant.TRANS_MIN_BLKS) as i:
                    self.tik_instance.data_move(
                        block_ub, self.work_space[core_index * Constant.WORK_SIZE + i * align_input_dim_3 +
                                                  self.tiling_input_dim_3 - Constant.TRANS_MIN_BLKS], 0, 1,
                        Constant.TRANS_MIN_BLKS // self.block_num, 0, 0)
                    self.tik_instance.data_move(pang_ub_1[(i - self.padding_index_2) * Constant.TRANS_MIN_BLKS],
                                                block_ub, 0, 1, Constant.TRANS_MIN_BLKS // self.block_num, 0, 0)

                with self.tik_instance.for_range(Constant.TRANS_MIN_BLKS,
                                                 self.tiling_input_dim_2 - Constant.TRANS_MIN_BLKS) as i:
                    self.tik_instance.data_move(
                        pang_ub_1[(i - self.padding_index_2)
                                  * Constant.TRANS_MIN_BLKS],
                        self.input_gm[core_index * input_ele_per_core +
                                      index * self.tiling_input_dim_2 * self.tiling_input_dim_3 +
                                      (i + 1) * self.tiling_input_dim_3 - Constant.TRANS_MIN_BLKS], 0, 1,
                        Constant.TRANS_MIN_BLKS // self.block_num, 0, 0)

                with self.tik_instance.for_range(0, Constant.TRANS_MIN_BLKS - self.padding_index_3) as i:
                    self.tik_instance.data_move(
                        block_ub, self.work_space[core_index * Constant.WORK_SIZE +
                                                  (Constant.TRANS_MIN_BLKS + i) * align_input_dim_3 +
                                                  self.tiling_input_dim_3 - Constant.TRANS_MIN_BLKS], 0, 1,
                        Constant.TRANS_MIN_BLKS // self.block_num, 0, 0)
                    self.tik_instance.data_move(
                        pang_ub_1[(i + self.tiling_output_dim_2 - Constant.TRANS_MIN_BLKS + self.padding_index_3) *
                                  Constant.TRANS_MIN_BLKS], block_ub, 0, 1, Constant.TRANS_MIN_BLKS // self.block_num,
                        0, 0)

                if self.inner_dtype == "float32":
                    float32_trans_block = Constant.FLOAT32_TRANS_BLK
                    src_stride = 1
                    dst_stride = time * float32_trans_block * \
                        Constant.FLOAT32_BYTES // Constant.BLOCK_SIZE
                with self.tik_instance.for_range(0, time) as i:
                    src_list = []
                    dst_list = []
                    for j in range(self.block_num):
                        if self.inner_dtype == "float32":
                            src_list.append(pang_ub_1[Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * i +
                                                      Constant.TRANS_MIN_BLKS * (2 * j)])
                            src_list.append(pang_ub_1[Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * i +
                                                      Constant.TRANS_MIN_BLKS * (2 * j + 1)])
                            dst_list.append(
                                pang_ub_2[time * Constant.TRANS_MIN_BLKS * j + Constant.TRANS_MIN_BLKS * i])
                            dst_list.append(pang_ub_2[time * Constant.TRANS_MIN_BLKS * j +
                                                      Constant.TRANS_MIN_BLKS * i + 8])
                        else:
                            src_list.append(pang_ub_1[Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * i +
                                                      Constant.TRANS_MIN_BLKS * j])
                            dst_list.append(
                                pang_ub_2[time * Constant.TRANS_MIN_BLKS * j + Constant.TRANS_MIN_BLKS * i])
                    if self.inner_dtype == "float32":
                        self.tik_instance.vnchwconv(
                            True, False, dst_list, src_list, 2, dst_stride, src_stride)
                    else:
                        self.tik_instance.vnchwconv(
                            True, False, dst_list, src_list, 1, 0, 0)

                with self.tik_instance.for_range(0, self.padding_index_1) as i:
                    self.tik_instance.vec_add(
                        Constant.MAX_MASK,
                        pang_ub_2[(Constant.TRANS_MIN_BLKS - 1 -
                                   self.padding_index_1 * 2 + i) * align_inter],
                        pang_ub_2[(Constant.TRANS_MIN_BLKS - 1 -
                                   self.padding_index_1 * 2 + i) * align_inter],
                        pang_ub_2[(Constant.TRANS_MIN_BLKS - 1 - i) * align_inter], loop_col, 8, 8, 8)
                    with self.tik_instance.if_scope(remain_mask_col > 0):
                        self.tik_instance.vec_add(
                            remain_mask_col,
                            pang_ub_2[(Constant.TRANS_MIN_BLKS - 1 - self.padding_index_1 * 2 + i) * align_inter +
                                      loop_col * Constant.MAX_MASK],
                            pang_ub_2[(Constant.TRANS_MIN_BLKS - 1 - self.padding_index_1 * 2 + i) * align_inter +
                                      loop_col * Constant.MAX_MASK],
                            pang_ub_2[(Constant.TRANS_MIN_BLKS - 1 - i) *
                                      align_inter + loop_col * Constant.MAX_MASK],
                            1, 8, 8, 8)

                if self.inner_dtype == "float32":
                    float32_trans_block = Constant.FLOAT32_TRANS_BLK
                    src_stride = 1
                    dst_stride = float32_trans_block * Constant.FLOAT32_BYTES // Constant.BLOCK_SIZE

                with self.tik_instance.for_range(0, time) as i:
                    src_list = []
                    dst_list = []
                    for j in range(self.block_num):
                        if self.inner_dtype == "float32":
                            dst_list.append(pang_ub_1[Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * i +
                                                      Constant.TRANS_MIN_BLKS * j])
                            dst_list.append(pang_ub_1[Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * i +
                                                      Constant.TRANS_MIN_BLKS * j + 8])
                            src_list.append(pang_ub_2[time * Constant.TRANS_MIN_BLKS * (2 * j) +
                                                      Constant.TRANS_MIN_BLKS * i])
                            src_list.append(pang_ub_2[time * Constant.TRANS_MIN_BLKS * (2 * j + 1) +
                                                      Constant.TRANS_MIN_BLKS * i])
                        else:
                            src_list.append(
                                pang_ub_2[time * Constant.TRANS_MIN_BLKS * j + Constant.TRANS_MIN_BLKS * i])
                            dst_list.append(pang_ub_1[Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * i +
                                            Constant.TRANS_MIN_BLKS * j])
                    if self.inner_dtype == "float32":
                        self.tik_instance.vnchwconv(
                            True, False, dst_list, src_list, 2, dst_stride, src_stride)
                    else:
                        self.tik_instance.vnchwconv(
                            True, False, dst_list, src_list, 1, 0, 0)

                burst = self.tik_instance.Scalar('int64', name='burst')

                burst.set_as((self.tiling_output_dim_3 -
                              (Constant.TRANS_MIN_BLKS - self.padding_index_0)) // Constant.TRANS_MIN_BLKS *
                             Constant.TRANS_MIN_BLKS // self.block_num)

                special_offset = self.tik_instance.Scalar(
                    dtype='int64', name='special_offset')
                special_offset.set_as(
                    (Constant.TRANS_MIN_BLKS - self.padding_index_0) + burst * self.block_num -
                    (Constant.TRANS_MIN_BLKS -
                     ((self.padding_index_1 + self.tiling_output_dim_3) -
                      ((Constant.TRANS_MIN_BLKS - self.padding_index_0) + burst * self.block_num))))

                with self.tik_instance.for_range(0, Constant.TRANS_MIN_BLKS - self.padding_index_2) as i:
                    self.tik_instance.data_move(
                        self.output_gm[core_index * output_ele_per_core +
                                       index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                       i * self.tiling_output_dim_3], ping_ub_2[i * Constant.TRANS_MIN_BLKS], 0, 1,
                        Constant.TRANS_MIN_BLKS // self.block_num, 0, 0)

                    self.tik_instance.data_move(
                        ping_ub_1,
                        self.work_space[Constant.WORK_SIZE * core_index +
                                        (i + self.padding_index_2) * align_input_dim_3 + Constant.TRANS_MIN_BLKS], 0, 1,
                        burst, 0, 0)
                    self.tik_instance.data_move(
                        self.output_gm[core_index * output_ele_per_core +
                                       index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                       i * self.tiling_output_dim_3 + Constant.TRANS_MIN_BLKS - self.padding_index_0],
                        ping_ub_1, 0, 1, burst, 0, 0)
                    with self.tik_instance.if_scope(special_offset > ((Constant.TRANS_MIN_BLKS - self.padding_index_0)
                                                                      + burst * self.block_num)):
                        new_block_ub = self.tik_instance.Tensor(self.inner_dtype, (Constant.ALIGN_16,),
                                                                name='new_block_ub',
                                                                scope=tik.scope_ubuf)
                        self.tik_instance.data_move(
                            new_block_ub,
                            self.work_space[Constant.WORK_SIZE * core_index +
                                            (i + self.padding_index_2) * align_input_dim_3 + Constant.TRANS_MIN_BLKS +
                                            burst * self.block_num -
                                            (Constant.TRANS_MIN_BLKS -
                                             (special_offset - ((Constant.TRANS_MIN_BLKS - self.padding_index_0) +
                                                                burst * self.block_num)))], 0, 1,
                            Constant.TRANS_MIN_BLKS // self.block_num, 0, 0)
                        self.tik_instance.data_move(
                            self.output_gm[core_index * output_ele_per_core +
                                           index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                           i * self.tiling_output_dim_3 + Constant.TRANS_MIN_BLKS -
                                           self.padding_index_0 + burst * self.block_num -
                                           (Constant.TRANS_MIN_BLKS -
                                            (special_offset - ((Constant.TRANS_MIN_BLKS - self.padding_index_0) +
                                                               burst * self.block_num)))], new_block_ub, 0, 1,
                            Constant.TRANS_MIN_BLKS // self.block_num, 0, 0)
                    self.tik_instance.data_move(
                        self.output_gm[
                            core_index * output_ele_per_core +
                            index * self.tiling_output_dim_2 * self.tiling_output_dim_3 + i * self.tiling_output_dim_3 +
                            (Constant.TRANS_MIN_BLKS - self.padding_index_0) + burst * self.block_num -
                            (Constant.TRANS_MIN_BLKS -
                             ((self.padding_index_1 + self.tiling_output_dim_3) -
                              ((Constant.TRANS_MIN_BLKS - self.padding_index_0) + burst * self.block_num)))],
                        pang_ub_1[i * Constant.TRANS_MIN_BLKS], 0, 1, Constant.TRANS_MIN_BLKS // self.block_num, 0, 0)

                with self.tik_instance.for_range(
                        Constant.TRANS_MIN_BLKS - self.padding_index_2,
                        self.tiling_output_dim_2 - (Constant.TRANS_MIN_BLKS - self.padding_index_3)) as i:
                    self.tik_instance.data_move(
                        self.output_gm[core_index * output_ele_per_core +
                                       index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                       i * self.tiling_output_dim_3], ping_ub_2[i * Constant.TRANS_MIN_BLKS], 0, 1,
                        Constant.TRANS_MIN_BLKS // self.block_num, 0, 0)
                    self.tik_instance.data_move(
                        ping_ub_1,
                        self.input_gm[core_index * input_ele_per_core +
                                      index * self.tiling_input_dim_2 * self.tiling_input_dim_3 +
                                      (i + self.padding_index_2) * self.tiling_input_dim_3 + Constant.TRANS_MIN_BLKS],
                        0, 1, burst, 0, 0)
                    self.tik_instance.data_move(
                        self.output_gm[core_index * output_ele_per_core +
                                       index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                       i * self.tiling_output_dim_3 + Constant.TRANS_MIN_BLKS - self.padding_index_0],
                        ping_ub_1, 0, 1, burst, 0, 0)
                    with self.tik_instance.if_scope(special_offset > (
                            (Constant.TRANS_MIN_BLKS - self.padding_index_0) + burst * self.block_num)):
                        new_block_ub = self.tik_instance.Tensor(self.inner_dtype, (Constant.ALIGN_16,),
                                                                name='new_block_ub',
                                                                scope=tik.scope_ubuf)
                        self.tik_instance.data_move(
                            new_block_ub,
                            self.input_gm[core_index * input_ele_per_core +
                                          index * self.tiling_input_dim_2 * self.tiling_input_dim_3 +
                                          (i + self.padding_index_2) * self.tiling_input_dim_3 +
                                          Constant.TRANS_MIN_BLKS + burst * self.block_num -
                                          (Constant.TRANS_MIN_BLKS -
                                           (special_offset - ((Constant.TRANS_MIN_BLKS - self.padding_index_0) +
                                                              burst * self.block_num)))], 0, 1, 2, 0, 0)
                        self.tik_instance.data_move(
                            self.output_gm[core_index * output_ele_per_core +
                                           index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                           i * self.tiling_output_dim_3 + Constant.TRANS_MIN_BLKS -
                                           self.padding_index_0 + burst * self.block_num -
                                           (Constant.TRANS_MIN_BLKS -
                                            (special_offset - ((Constant.TRANS_MIN_BLKS - self.padding_index_0) +
                                                               burst * self.block_num)))], new_block_ub, 0, 1,
                            Constant.TRANS_MIN_BLKS // self.block_num, 0, 0)

                    self.tik_instance.data_move(
                        self.output_gm[
                            core_index * output_ele_per_core +
                            index * self.tiling_output_dim_2 * self.tiling_output_dim_3 + i * self.tiling_output_dim_3 +
                            (Constant.TRANS_MIN_BLKS - self.padding_index_0) + burst * self.block_num -
                            (Constant.TRANS_MIN_BLKS -
                             ((self.padding_index_1 + self.tiling_output_dim_3) -
                              ((Constant.TRANS_MIN_BLKS - self.padding_index_0) + burst * self.block_num)))],
                        pang_ub_1[i * Constant.TRANS_MIN_BLKS], 0, 1, Constant.TRANS_MIN_BLKS // self.block_num, 0, 0)

                with self.tik_instance.for_range(
                        self.tiling_output_dim_2 -
                    (Constant.TRANS_MIN_BLKS - self.padding_index_3),
                        self.tiling_output_dim_2) as i:
                    self.tik_instance.data_move(
                        self.output_gm[core_index * output_ele_per_core +
                                       index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                       i * self.tiling_output_dim_3], ping_ub_2[i * Constant.TRANS_MIN_BLKS], 0, 1,
                        Constant.TRANS_MIN_BLKS // self.block_num, 0, 0)
                    self.tik_instance.data_move(
                        ping_ub_1,
                        self.work_space[Constant.WORK_SIZE * core_index +
                                        (i + Constant.TRANS_MIN_BLKS -
                                         (self.tiling_output_dim_2 -
                                          (Constant.TRANS_MIN_BLKS - self.padding_index_3))) * align_input_dim_3 +
                                        Constant.TRANS_MIN_BLKS], 0, 1, burst, 0, 0)
                    self.tik_instance.data_move(
                        self.output_gm[core_index * output_ele_per_core +
                                       index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                       i * self.tiling_output_dim_3 + Constant.TRANS_MIN_BLKS - self.padding_index_0],
                        ping_ub_1, 0, 1, burst, 0, 0)
                    with self.tik_instance.if_scope(special_offset > (
                            (Constant.TRANS_MIN_BLKS - self.padding_index_0) + burst * self.block_num)):
                        new_block_ub = self.tik_instance.Tensor(self.inner_dtype, (Constant.TRANS_MIN_BLKS,),
                                                                name='new_block_ub',
                                                                scope=tik.scope_ubuf)
                        self.tik_instance.data_move(
                            new_block_ub,
                            self.work_space[Constant.WORK_SIZE * core_index +
                                            (i + Constant.TRANS_MIN_BLKS -
                                             (self.tiling_output_dim_2 -
                                              (Constant.TRANS_MIN_BLKS - self.padding_index_3))) * align_input_dim_3 +
                                            Constant.TRANS_MIN_BLKS + burst * self.block_num -
                                            (Constant.TRANS_MIN_BLKS -
                                             (special_offset - ((Constant.TRANS_MIN_BLKS - self.padding_index_0) +
                                                                burst * self.block_num)))], 0, 1,
                            Constant.TRANS_MIN_BLKS // self.block_num, 0, 0)
                        self.tik_instance.data_move(
                            self.output_gm[core_index * output_ele_per_core +
                                           index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                           i * self.tiling_output_dim_3 + Constant.TRANS_MIN_BLKS -
                                           self.padding_index_0 + burst * self.block_num -
                                           (Constant.TRANS_MIN_BLKS -
                                            (special_offset - ((Constant.TRANS_MIN_BLKS - self.padding_index_0) +
                                                               burst * self.block_num)))], new_block_ub, 0, 1,
                            Constant.TRANS_MIN_BLKS // self.block_num, 0, 0)
                    with self.tik_instance.if_scope(i == self.tiling_output_dim_2 - 1):
                        with self.tik_instance.if_scope(special_offset > (
                                (Constant.TRANS_MIN_BLKS - self.padding_index_0) + burst * self.block_num)):
                            new_block_ub = self.tik_instance.Tensor(self.inner_dtype, (Constant.TRANS_MIN_BLKS,),
                                                                    name='new_block_ub',
                                                                    scope=tik.scope_ubuf)
                            self.tik_instance.data_move(
                                new_block_ub,
                                self.work_space[Constant.WORK_SIZE * core_index +
                                                (i + Constant.TRANS_MIN_BLKS -
                                                 (self.tiling_output_dim_2 -
                                                  (Constant.TRANS_MIN_BLKS - self.padding_index_3))) * align_input_dim_3
                                                + Constant.TRANS_MIN_BLKS + burst * self.block_num -
                                                (Constant.TRANS_MIN_BLKS -
                                                 (special_offset - ((Constant.TRANS_MIN_BLKS - self.padding_index_0) +
                                                                    burst * self.block_num)))], 0, 1,
                                Constant.TRANS_MIN_BLKS // self.block_num, 0, 0)
                            with self.tik_instance.for_range(Constant.TRANS_MIN_BLKS - self.padding_index_0,
                                                             Constant.TRANS_MIN_BLKS) as j:
                                block_ub[j - (Constant.TRANS_MIN_BLKS -
                                              self.padding_index_0)].set_as(new_block_ub[j])
                        with self.tik_instance.else_scope():
                            with self.tik_instance.for_range(
                                    self.tiling_output_dim_3 - Constant.TRANS_MIN_BLKS -
                                (Constant.TRANS_MIN_BLKS - self.padding_index_0),
                                    self.tiling_output_dim_3 - Constant.TRANS_MIN_BLKS -
                                    (Constant.TRANS_MIN_BLKS - self.padding_index_0) + self.padding_index_1) as j:
                                block_ub[j - (self.tiling_output_dim_3 - Constant.TRANS_MIN_BLKS -
                                              (Constant.TRANS_MIN_BLKS - self.padding_index_0))].set_as(ping_ub_1[j])
                        with self.tik_instance.for_range(0, Constant.TRANS_MIN_BLKS - self.padding_index_1) as j:
                            block_ub[j + self.padding_index_1].set_as(
                                pang_ub_1[i * Constant.TRANS_MIN_BLKS + j])
                        self.tik_instance.data_move(
                            self.output_gm[core_index * output_ele_per_core +
                                           index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                           (i + 1) * self.tiling_output_dim_3 - Constant.TRANS_MIN_BLKS], block_ub, 0,
                            1, Constant.TRANS_MIN_BLKS // self.block_num, 0, 0)
                    with self.tik_instance.else_scope():
                        self.tik_instance.data_move(
                            self.output_gm[core_index * output_ele_per_core +
                                           index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                           i * self.tiling_output_dim_3 +
                                           (Constant.TRANS_MIN_BLKS - self.padding_index_0) +
                                           burst * self.block_num -
                                           (Constant.TRANS_MIN_BLKS -
                                            ((self.padding_index_1 + self.tiling_output_dim_3) -
                                             ((Constant.TRANS_MIN_BLKS - self.padding_index_0) +
                                              burst * self.block_num)))],
                            pang_ub_1[i * Constant.TRANS_MIN_BLKS], 0, 1, Constant.TRANS_MIN_BLKS // self.block_num, 0,
                            0)

    def do_pad(self, core_index):
        """
        do_pad with different tiling key
        """
        with self.tik_instance.if_scope(self.tiling_key == Constant.MODE0):
            with self.tik_instance.new_stmt_scope():
                self.do_tiling_key_mode_0(core_index)
        with self.tik_instance.if_scope(self.tiling_key == Constant.MODE1):
            with self.tik_instance.new_stmt_scope():
                self.do_tiling_key_mode_1(core_index)

    def pad_grad_compute(self, outer_compile_info=None):
        """
        pad_grad_compute
        """
        self.pad_v3_grad_compute_tiling()
        opt_config = {"out_of_bound_sync_check": True,
                      "enable_const_fold": True}

        # add compile info
        wr_compile_info = {}
        wr_compile_info["core_num"] = self.core_nums
        wr_compile_info["padding_contiguous"] = self.padding_contiguous
        if outer_compile_info is not None:
            for key in outer_compile_info.keys():
                wr_compile_info[key] = outer_compile_info[key]
        tbe_context.get_context().add_compile_info("vars", wr_compile_info)
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=self.input_gm_list,
                                   flowtable=[self.tiling_gm],
                                   outputs=self.output_gm_list,
                                   config=opt_config)

        return self.tik_instance


def check_supported(x, paddings, y, mode='reflect', padding_contiguous=True, kernel_name="pad_v3_grad"):
    x_shape = x.get("shape")
    x_dtype = x.get("dtype").lower()
    if mode not in ("reflect", "edge"):
        reason = "current mode is not supported"
        return False, reason
    if int(-1) in x_shape or int(-2) in x_shape:
        return "Unknown"
    if len(x_shape) >= CHECK_DIMS:
        reason = "input dims is larger than 4"
        return False, reason
    
    # the threshold of 910 and 910B is different
    max_h_w_910_fp32 = {"h": 960, "w": 1920}
    max_h_w_910b_fp32 = {"h": 800, "w": 1500}
    max_h_w_910_fp16 = {"h": 1920, "w": 3840}
    max_h_w_910b_fp16 = {"h": 1600, "w": 3000}
    h_dim = x_shape[-2]
    w_dim = x_shape[-1]
    reason = "shape is larger than the 910B threshold"
    # in 910B, ub_size is smaller than MAX_UB_SIZE
    ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) // Constant.ONE_KB
    if (x_dtype == "float32"):
        if ub_size < Constant.MAX_UB_SIZE:
            if h_dim > max_h_w_910b_fp32.get('h') or w_dim > max_h_w_910b_fp32.get('w'):
                return False, reason
        if h_dim > max_h_w_910_fp32.get('h') or w_dim > max_h_w_910_fp32.get('w'):
            reason = "shape is largeer than the 310/910 fp32 threshold"
            return False, reason
        
    if (x_dtype == "float16"):
        if ub_size < Constant.MAX_UB_SIZE:
            if h_dim > max_h_w_910b_fp16.get('h') or w_dim > max_h_w_910b_fp16.get('w'):
                return False, reason
        if h_dim > max_h_w_910_fp16.get('h') or w_dim > max_h_w_910_fp16.get('w'):
            reason = "shape is largeer than the 310/910 fp16 threshold"
            return False, reason
    return True


@register_operator("PadV3Grad")
def pad_v3_grad(x, paddings, y, mode='reflect', padding_contiguous=True, kernel_name="pad_v3_grad"):
    """ calculating pad_v3_grad tensor by paddings parameters

    Parameters
    ----------
    x : dict
        shape and dtype of input
    paddings: dict
        shape and dtype of output
        For each dimension D of input, paddings[D, 0] indicates how many
        values to add
        before the contents of tensor in that dimension, and paddings[D, 1]
        indicates
        how many values to add after the contents of tensor in that dimension.
    y : dict
        shape and dtype of output
    mode : str
        the mode of pad
    padding_contiguous : str
        judge the pad style
    kernel_name : str
        cce kernel name, default value is "pad_v3_grad"

    Returns
    -------
    None.
    """
    if mode == "reflect":
        paddings_dtype = paddings.get("dtype").lower()
        src_dtype = x.get("dtype").lower()
        supported_dtype = ("float16", "float32")
        para_check.check_dtype(src_dtype, supported_dtype, param_name="x")
        para_check.check_dtype(
            paddings_dtype, ("int32", "int64"), param_name="paddings")
        obj = PadV3GradInit(x, paddings, y, mode,
                            padding_contiguous, kernel_name)
        obj.init_src_dst_gm((x, paddings), (y,),
                            pad_input_idx=0, pad_outnput_idx=0)
        return obj.pad_grad_compute()
    if mode == "edge":
        return replication_pad_v3_grad(x, paddings, y, mode, padding_contiguous, kernel_name)
    return None
