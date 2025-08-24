#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

search_n
"""

# 'pylint: disable=import-error
from math import ceil
import functools
from te import tik
from te import platform as cce
from te.utils import para_check
from tbe.common.platform import set_current_compile_soc_info
from tbe.common.platform.platform_info import get_soc_spec

# 'pylint: disable=locally-disabled,too-many-arguments
# 'pylint: disable=too-many-branches, too-many-statements, too-many-locals, attribute-defined-outside-init
# 'pylint: disable=too-many-instance-attributes, no-self-use, too-many-instance-attributes, protected-access
# 'pylint: disable=too-few-public-methods


# 'pylint: disable=too-few-public-methods
class Constant(object):
    """
    This class for Constant.
    """
    FP16_MAX = 65504
    INT16_MAX = 32767
    INT16_MIN = -32768
    INT32_MAX = 2 ** 31 - 1
    HW_NUM_THRES = 9
    MOVE_STRIDE_MAX = 65535
    MOVE_NBURST_MAX = 4095
    N_STEP = 16
    BASE_TWO = 2
    # when fp32 data exceeds 2**24, the precision will be not enough
    FP32_LOSS_MAX = 2 ** 24
    SECOND_DIV_FACTOR = 100
    MAX_REPEAT_TIME = 255
    MASK_MAX_BYTE = 256
    BLOCK_BYTE_SIZE = 32

    def __init__(self):
        return


class SearchN(object):
    """
    SearchN: find an optimal N for shift-N
    """
    def __init__(
            self,
            data,
            scale_d,
            scale_w,
            output_n,
            kernel_name):

        self._check_inputs(data, scale_d, scale_w)
        self._get_params(data, scale_d, scale_w, kernel_name)

        # input&output global memory
        self.data = self.tik_instance.Tensor(self.input_data_dtype, self.data_shape, tik.scope_gm, 'data')
        self.scale_d = self.tik_instance.Tensor(self.scale_d_dtype, self.scale_d_shape, tik.scope_gm, 'scale_d')
        self.scale_w = self.tik_instance.Tensor(self.scale_w_dtype, self.scale_w_shape, tik.scope_gm, 'scale_w')
        self.output_n = self.tik_instance.Tensor('int8', self.scale_w_shape, tik.scope_gm, 'output_n')
        
        self.loss_max_rec_ub = None
        self.total_remain_loss = None
        self.int16_max_tensor = None
        self.int16_min_tensor = None
        self.dw_fp16_1_intg = None
        self.fp16_max_tensor = None
        self.scale_d_ub = None
        self.carry_num = None
        self.loss4 = None
        self.one_main_loss_ub = None
        self.remain_loss_ub = None
        self.scale_w_ub = None
        self.cur_valid_data_num = None
        self.d_intg_fp16_scalar = None
        self.loss3 = None
        self.tmp_loss_high = None
        self.dw_fp16_decm = None
        self.final_loss_1 = None
        self.loss2 = None
        self.loss1 = None
        self.scale_w_intg_fp16 = None
        self.scale_w_int32 = None
        self.total_main_loss = None
        self.final_loss_2 = None
        self.main_loss_ub = None
        self.cmp_result_scalar = None
        self.total_loss = None
        self.tmp_loss_ub = None
        self.sub_loss_ub = None
        self.cmp_result = None
        self.dw_int32 = None
        self.tmp_loss_low = None
        self.scale_d_rec = None
        self.scale_w_rec = None
        self.cur_loss_size = None
        self.dw_fp16_2_intg = None
        self.one_remain_loss_ub = None
        self.loss_max_ub = None
        self.final_loss_3 = None
        self.final_loss_4 = None

        # data split to multi core
        if self.data_format == 'ND' or self.ori_scale_w_len == 1:
            self.per_tensor_quant = True
            self.data_num_each_core = self.data_num // self.data_each_block // self.aicore_num * self.data_each_block
            self.data_num_last_core = self.data_num % self.data_each_block + \
                self.data_num // self.data_each_block % self.aicore_num * self.data_each_block
        else:
            self.per_tensor_quant = False
            self.data_hw_each_core = self.dim_h * self.dim_w // self.aicore_num
            self.data_hw_last_core = self.dim_h * self.dim_w % self.aicore_num

        # define loss
        if self.per_tensor_quant or (self.dim_h * self.dim_w <= Constant.HW_NUM_THRES):
            self.total_loss_shape = [Constant.N_STEP]
            self.total_loss_size = Constant.N_STEP
            self.per_tensor_loss = True
        else:
            self.total_loss_shape = [Constant.N_STEP, self.c_len]
            self.total_loss_size = Constant.N_STEP * self.c_len
            self.per_tensor_loss = False
        loss_workspace_shape = [self.aicore_num] + self.total_loss_shape
        # define loss workspace on gm
        if self.soc_version == 'SD3403':
            # true loss is final_loss1 * 2**32 + final_loss2 * 2**16 + final_loss3 * 2**16 + final_loss4
            self.final_loss1_workspace = self.tik_instance.Tensor(self.loss_precision, loss_workspace_shape,
                tik.scope_gm, 'final_loss1_workspace', is_workspace=True, is_atomic_add=True)
            self.final_loss2_workspace = self.tik_instance.Tensor(self.loss_precision, loss_workspace_shape,
                tik.scope_gm, 'final_loss2_workspace', is_workspace=True, is_atomic_add=True)
            self.final_loss3_workspace = self.tik_instance.Tensor(self.loss_precision, loss_workspace_shape,
                tik.scope_gm, 'final_loss3_workspace', is_workspace=True, is_atomic_add=True)
            self.final_loss4_workspace = self.tik_instance.Tensor(self.loss_precision, loss_workspace_shape,
                tik.scope_gm, 'final_loss4_workspace', is_workspace=True, is_atomic_add=True)
        else:
            # true loss is main_loss * loss_max + remain_loss
            self.main_loss_workspace = self.tik_instance.Tensor(self.loss_precision, loss_workspace_shape,
                tik.scope_gm, 'main_loss_workspace', is_workspace=True, is_atomic_add=True)
            self.remain_loss_workspace = self.tik_instance.Tensor(self.loss_precision, loss_workspace_shape,
                tik.scope_gm, 'remain_loss_workspace', is_workspace=True, is_atomic_add=True)
            self.loss_max = (Constant.FP32_LOSS_MAX - 1) // self.aicore_num

        if self.aicore_num > 1:
            self.barrier_workspace = self.tik_instance.Tensor(
                'int64', (self.aicore_num*Constant.BLOCK_BYTE_SIZE//(cce.cce_intrin.get_bit_len('int64')//8),),
                tik.scope_gm, 'barrier_workspace', is_workspace=True, is_atomic_add=True)

        # calculate max num that can be processed once
        self._calc_max_process_num()

    def search_n_compute(self):
        with self.tik_instance.for_range(0, self.aicore_num, block_num=self.aicore_num) as core_ind:
            self._compute_quant_output_and_loss(core_ind)
            if self.aicore_num > 1:
                self.tik_instance.block_barrier(self.barrier_workspace)
            # core 0 do reduce
            with self.tik_instance.if_scope(core_ind == 0):
                self._reduce_and_output()

        self.tik_instance.BuildCCE(self.kernel_name, [self.data, self.scale_d, self.scale_w], [self.output_n])
        return self.tik_instance

    def _check_inputs(self, data, scale_d, scale_w):
        self.soc_version = get_soc_spec("FULL_SOC_VERSION")
        status = set_current_compile_soc_info(self.soc_version)
        if status != "success":
            raise ValueError('Set soc_version failed, please check!')
        self.tik_instance = tik.Tik()
        if self.soc_version in ['Ascend610', 'BS9SX1AA', 'BS9SX1AB', 'BS9SX1AC']:
            self.aicore_num = 1
        else:
            self.aicore_num = get_soc_spec("CORE_NUM")

        if len(scale_w.get('shape')) != 1:
            raise ValueError('The dimension of "scale_w" must be 1!')
        if len(scale_d.get('shape')) != 1 or scale_d.get('shape')[0] != 1:
            raise ValueError('The shape of "scale_d" must be "[1]"!')

        self.ori_data_shape = data.get('ori_shape')
        self.ori_data_format = data.get('ori_format')
        if self.ori_data_format == 'NCHW' or self.ori_data_format == 'NCDHW':
            ori_c_index = 1
        elif self.ori_data_format == 'NHWC':
            ori_c_index = 3
        elif self.ori_data_format == 'NDHWC':
            ori_c_index = 4

        self.data_format = data.get('format')
        self.ori_scale_w_len = scale_w.get('shape')[0]
        if self.data_format == 'ND':
            if self.ori_scale_w_len != 1:
                raise ValueError('The num of "scale_w" must be 1 when format is ND!')
        else:
            if self.ori_scale_w_len != 1 and self.ori_scale_w_len != self.ori_data_shape[ori_c_index]:
                raise ValueError('The num of "scale_w" must be 1 or equal to the length of C-dimension '\
                    'when ori_format is NCHW/NHWC!')

    def _get_params(self, data, scale_d, scale_w, kernel_name):
        self.input_data_dtype = data.get('dtype')
        self.data_shape = data.get('shape')
        self.data_num = functools.reduce(lambda x, y: x * y, self.data_shape)
        self.scale_d_dtype = scale_d.get('dtype')
        self.scale_d_shape = scale_d.get('shape')
        self.scale_w_dtype = scale_w.get('dtype')
        self.scale_w_shape = scale_w.get('shape')
        self.kernel_name = kernel_name

        # uniform use calc_precision in processing
        if self.soc_version == 'SD3403':
            self.calc_precision = 'float16'
            self.loss_precision = 'int32'
        else:
            self.calc_precision = 'float32'
            self.loss_precision = 'float32'
        self.data_quant_type = 'int32'

        self.input_data_byte_size = cce.cce_intrin.get_bit_len(self.input_data_dtype) // 8
        self.input_scale_d_byte_size = cce.cce_intrin.get_bit_len(self.scale_d_dtype) // 8
        self.input_scale_w_byte_size = cce.cce_intrin.get_bit_len(self.scale_w_dtype) // 8

        self.data_byte_size = cce.cce_intrin.get_bit_len(self.calc_precision) // 8
        self.loss_byte_size = cce.cce_intrin.get_bit_len(self.loss_precision) // 8
        self.data_each_block = Constant.BLOCK_BYTE_SIZE // self.data_byte_size
        self.loss_each_block = Constant.BLOCK_BYTE_SIZE // self.loss_byte_size
        self.vector_mask_max = Constant.MASK_MAX_BYTE // self.data_byte_size
        self.blk_per_rep = Constant.MASK_MAX_BYTE // Constant.BLOCK_BYTE_SIZE
        self.fp16_vector_mask_max = Constant.MASK_MAX_BYTE // 2
        self.fp16_data_each_block = Constant.BLOCK_BYTE_SIZE // 2
        self.int32_vector_mask_max = Constant.MASK_MAX_BYTE // 4

        if self.data_format == 'NC1HWC0':
            self.dim_n = self.data_shape[0]
            self.dim_c1 = self.data_shape[1]
            self.dim_h = self.data_shape[2]
            self.dim_w = self.data_shape[3]
            self.dim_c0 = self.data_shape[4]
            self.c_len = self.dim_c1 * self.dim_c0
        elif self.data_format == 'NDC1HWC0':
            self.dim_n = self.data_shape[0] * self.data_shape[1]
            self.dim_c1 = self.data_shape[2]
            self.dim_h = self.data_shape[3]
            self.dim_w = self.data_shape[4]
            self.dim_c0 = self.data_shape[5]
            self.c_len = self.dim_c1 * self.dim_c0

    def _calc_max_process_num(self):
        self.unified_buffer_size = tik.Dprofile().get_unified_buffer_size()
        # max num that can be processed once when compute quant data and loss
        if self.soc_version == 'Ascend310':
            ub_size_factor = 9
            loss_factor = 2
            output_n_size_factor = 12
        elif self.soc_version == 'SD3403':
            ub_size_factor = 20
            loss_factor = 4
            output_n_size_factor = 15
        else:
            ub_size_factor = 4
            loss_factor = 2
            output_n_size_factor = 12
        if self.per_tensor_loss:
            loss_num_ub = Constant.N_STEP
        else:
            loss_num_ub = Constant.N_STEP * self.dim_c0
        self.ub_tensor_size = ((self.unified_buffer_size - loss_factor * loss_num_ub * self.loss_byte_size) // 2 //
            self.data_byte_size // ub_size_factor // self.vector_mask_max * self.vector_mask_max)
        if not self.per_tensor_quant:
            self.ub_tensor_hw = self.ub_tensor_size // (self.dim_c0 * self.dim_n)
            if self.ub_tensor_hw < 1:
                self.new_n = 1
                self.loop_n = self.dim_n
                self.ub_tensor_hw = self.ub_tensor_size // self.dim_c0
                self.ub_tensor_size = self.ub_tensor_hw * self.dim_c0
            else:
                self.new_n = self.dim_n
                self.loop_n = 1
                self.ub_tensor_size = self.ub_tensor_hw * self.dim_c0 * self.dim_n
            # align up to vector_mask_max, no need to process last_num
            # valid num is ub_tensor_hw * dim_c0 * new_n
            self.ub_tensor_size = ceil(self.ub_tensor_size / self.vector_mask_max) * self.vector_mask_max

        self.data_repeats = self.ub_tensor_size // self.vector_mask_max
        # max c num that can be processed once when reduce loss
        if self.per_tensor_loss:
            self.output_ub_tensor_size = ((self.unified_buffer_size - 4 * Constant.MASK_MAX_BYTE)
                // 1 // self.fp16_vector_mask_max * self.fp16_vector_mask_max)
            self.output_ub_tensor_size = min(self.output_ub_tensor_size,
                Constant.MAX_REPEAT_TIME*self.fp16_vector_mask_max)
        else:
            self.output_ub_tensor_size = (self.unified_buffer_size // output_n_size_factor
                // Constant.N_STEP // self.loss_byte_size // self.fp16_data_each_block * self.fp16_data_each_block)

    def _load_nd_data_to_ub(self, move_offset, move_num):
        data_ub = self.tik_instance.Tensor(self.calc_precision, (self.ub_tensor_size,), tik.scope_ubuf, 'data_ub')
        if self.input_data_dtype == 'float16' and self.calc_precision == 'float32':
            # conv fp16 to fp32
            with self.tik_instance.new_stmt_scope():
                data_ub_fp16 = self.tik_instance.Tensor(
                    'float16', (self.ub_tensor_size,), tik.scope_ubuf, 'data_ub_fp16')
                fp16_burst_len = (move_num * self.input_data_byte_size +
                    Constant.BLOCK_BYTE_SIZE - 1) // Constant.BLOCK_BYTE_SIZE
                self.tik_instance.data_move(data_ub_fp16, self.data[move_offset], 0, 1, fp16_burst_len, 0, 0)
                self._convert_by_loop(data_ub, data_ub_fp16, move_num)
        else:
            burst_len = (move_num + self.data_each_block - 1) // self.data_each_block
            self.tik_instance.data_move(data_ub, self.data[move_offset], 0, 1, burst_len, 0, 0)
        return data_ub

    def _load_5hd_data_by_split_hw(self, dst, move_offset, move_hw):
        burst_len = move_hw * self.dim_c0 * self.input_data_byte_size // Constant.BLOCK_BYTE_SIZE
        move_src_stride = (self.dim_h * self.dim_w * self.dim_c1 - move_hw) * self.dim_c0 * \
            self.input_data_byte_size // Constant.BLOCK_BYTE_SIZE
        move_n_burst = self.new_n
        if move_src_stride <= Constant.MOVE_STRIDE_MAX and move_n_burst <= Constant.MOVE_NBURST_MAX:
            self.tik_instance.data_move(dst, self.data[move_offset], 0, move_n_burst, burst_len, move_src_stride, 0)
        else:
            with self.tik_instance.for_range(0, move_n_burst) as move_index:
                self.tik_instance.data_move(dst[move_index*move_hw*self.dim_c0],
                    self.data[move_offset+move_index*self.dim_h*self.dim_w*self.dim_c1*self.dim_c0],
                    0, 1, burst_len, 0, 0)

    def _load_5hd_data_to_ub(self, move_offset, move_hw):
        data_ub = self.tik_instance.Tensor(self.calc_precision, (self.ub_tensor_size,),
            tik.scope_ubuf, 'data_ub')
        if self.input_data_dtype == 'float16' and self.calc_precision == 'float32':
            with self.tik_instance.new_stmt_scope():
                move_num = move_hw * self.new_n * self.dim_c0
                data_ub_fp16 = self.tik_instance.Tensor('float16', (move_num,), tik.scope_ubuf, 'data_ub_fp16')
                # move data to ub
                self._load_5hd_data_by_split_hw(data_ub_fp16, move_offset, move_hw)
                # conv fp16 to fp32
                self._convert_by_loop(data_ub, data_ub_fp16, move_num)
        else:
            self._load_5hd_data_by_split_hw(data_ub, move_offset, move_hw)
        return data_ub

    def _convert_by_loop(self, data_ub, data_ub_fp16, data_num):
        convert_loop = data_num // (self.vector_mask_max * Constant.MAX_REPEAT_TIME)
        convert_offset = 0
        if convert_loop > 0:
            with self.tik_instance.for_range(0, convert_loop) as convert_index:
                convert_offset = convert_index * self.vector_mask_max * Constant.MAX_REPEAT_TIME
                repeat_times = Constant.MAX_REPEAT_TIME
                self.tik_instance.vec_conv(self.vector_mask_max, '', data_ub[convert_offset],
                    data_ub_fp16[convert_offset], repeat_times, self.blk_per_rep, self.blk_per_rep//2)

        repeat_time = (data_num % (self.vector_mask_max * Constant.MAX_REPEAT_TIME) // self.vector_mask_max)
        if repeat_time > 0:
            convert_offset = convert_loop * self.vector_mask_max * Constant.MAX_REPEAT_TIME
            repeat_times = repeat_time
            self.tik_instance.vec_conv(self.vector_mask_max, '', data_ub[convert_offset],
                data_ub_fp16[convert_offset], repeat_times, self.blk_per_rep, self.blk_per_rep//2)

        last_num = data_num % self.vector_mask_max
        if last_num > 0:
            convert_offset = convert_loop * self.vector_mask_max * Constant.MAX_REPEAT_TIME + \
                repeat_time * self.vector_mask_max
            repeat_times = 1
            self.tik_instance.vec_conv(last_num, '', data_ub[convert_offset], data_ub_fp16[convert_offset],
                repeat_times, self.blk_per_rep, self.blk_per_rep//2)

    def _dup_by_repeat(self, dst, src_scalar, data_num, mask_num):
        loop_time = data_num // (mask_num * Constant.MAX_REPEAT_TIME)
        repeat_time = data_num % (mask_num * Constant.MAX_REPEAT_TIME) // mask_num
        last_num = data_num % mask_num
        if loop_time > 0:
            with self.tik_instance.for_range(0, loop_time) as loop_id:
                offset = loop_id * mask_num * Constant.MAX_REPEAT_TIME
                self.tik_instance.vec_dup(mask_num, dst[offset], src_scalar,
                    Constant.MAX_REPEAT_TIME, self.blk_per_rep)
        if repeat_time > 0:
            offset = loop_time * mask_num * Constant.MAX_REPEAT_TIME
            self.tik_instance.vec_dup(mask_num, dst[offset], src_scalar, repeat_time, self.blk_per_rep)
        if last_num > 0:
            offset = (loop_time * Constant.MAX_REPEAT_TIME + repeat_time) * mask_num
            self.tik_instance.vec_dup(last_num, dst[offset], src_scalar, 1, self.blk_per_rep)

    def _load_scale_d_to_ub(self):
        ori_scale_d_ub = self.tik_instance.Tensor(self.scale_d_dtype,
            (Constant.BLOCK_BYTE_SIZE//self.input_scale_d_byte_size,), tik.scope_ubuf, 'ori_scale_d_ub')
        self.tik_instance.data_move(ori_scale_d_ub, self.scale_d, 0, 1, 1, 0, 0)
        self.scale_d_ub = ori_scale_d_ub

        # convert if need
        if self.calc_precision != self.scale_d_dtype:
            scale_d_ub_conv = self.tik_instance.Tensor(self.calc_precision,
                (Constant.BLOCK_BYTE_SIZE//self.input_scale_d_byte_size,), tik.scope_ubuf, 'scale_d_ub_conv')
            if self.soc_version in ['SD3403']:
                ori_scale_d_scalar = self.tik_instance.Scalar(self.scale_d_dtype, 'ori_scale_d_scalar')
                ori_scale_d_scalar.set_as(ori_scale_d_ub[0])
                scale_d_scalar = self.tik_instance.Scalar(self.calc_precision, 'scale_d_scalar')
                self.tik_instance.scalar_conv('', scale_d_scalar, ori_scale_d_scalar)
                scale_d_ub_conv[0].set_as(scale_d_scalar)
            else:
                self.tik_instance.vec_conv(1, '', scale_d_ub_conv, ori_scale_d_ub, 1, 1, 1)
            self.scale_d_ub = scale_d_ub_conv

        if self.soc_version in ['Ascend310', 'SD3403']:
            # rec for scale_d
            self.scale_d_rec = self.tik_instance.Tensor(self.calc_precision, (self.data_each_block,),
                tik.scope_ubuf, 'scale_d_rec')
            self._calculate_rec_for_tensor(self.scale_d_rec, self.scale_d_ub, 1)

    def _load_5hd_scale_w_to_ub(self, offset):
        '''load c0-size scale w for one time'''
        ori_scale_w_ub = self.tik_instance.Tensor(self.scale_w_dtype,
            (self.dim_c0,), tik.scope_ubuf, 'ori_scale_w_ub')
        burst_len = self.dim_c0 * self.input_scale_w_byte_size // Constant.BLOCK_BYTE_SIZE
        self.tik_instance.data_move(ori_scale_w_ub, self.scale_w[offset], 0, 1, burst_len, 0, 0)
        self.scale_w_ub = ori_scale_w_ub

        if self.calc_precision != self.scale_w_dtype:
            scale_w_ub_conv = self.tik_instance.Tensor(self.calc_precision, (self.dim_c0,),
                tik.scope_ubuf, 'scale_w_ub_conv')
            if self.soc_version in ['SD3403']:
                with self.tik_instance.for_range(0, self.dim_c0) as conv_ind:
                    ori_scale_w_scalar = self.tik_instance.Scalar(self.scale_w_dtype, 'ori_scale_w_scalar')
                    ori_scale_w_scalar.set_as(self.scale_w_ub[conv_ind])
                    scale_w_scalar = self.tik_instance.Scalar(self.calc_precision, 'scale_w_scalar')
                    self.tik_instance.scalar_conv('', scale_w_scalar, ori_scale_w_scalar)
                    scale_w_ub_conv[conv_ind].set_as(scale_w_scalar)
            else:
                self.tik_instance.vec_conv(self.dim_c0, '', scale_w_ub_conv, self.scale_w_ub, 1,
                    self.blk_per_rep, self.blk_per_rep//2)
            self.scale_w_ub = scale_w_ub_conv

        # use the last valid scale_w to fill up
        scale_w_expand = self.tik_instance.Tensor(self.calc_precision,
            (self.dim_c0,), tik.scope_ubuf, 'scale_w_expand')
        burst_len = self.dim_c0 // self.data_each_block
        rest_num = self.ori_scale_w_len - offset
        with self.tik_instance.if_scope(self.dim_c0 > rest_num):
            scale_w_last_one = self.tik_instance.Scalar(self.calc_precision, 'scale_w_last_one')
            scale_w_last_one.set_as(self.scale_w_ub[self.ori_scale_w_len-offset-1])
            self.tik_instance.vec_dup(self.dim_c0, scale_w_expand, scale_w_last_one, 1, self.blk_per_rep)
            self.tik_instance.vec_dup(self.ori_scale_w_len-offset, scale_w_expand, 0, 1, self.blk_per_rep)
            self.tik_instance.vec_add(self.ori_scale_w_len-offset, scale_w_expand, scale_w_expand,
                self.scale_w_ub, 1, self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
            self.tik_instance.data_move(self.scale_w_ub, scale_w_expand, 0, 1, burst_len, 0, 0)

        if self.soc_version in ['Ascend310', 'SD3403']:
            # rec for scale_w
            self.scale_w_rec = self.tik_instance.Tensor(self.calc_precision, (self.dim_c0,),
                tik.scope_ubuf, 'scale_w_rec')
            self._calculate_rec_for_tensor(self.scale_w_rec, self.scale_w_ub, self.dim_c0)

    def _load_nd_scale_w_to_ub(self):
        # load from gm to ub
        ori_scale_w_ub = self.tik_instance.Tensor(self.scale_w_dtype,
            (Constant.BLOCK_BYTE_SIZE//self.input_scale_w_byte_size,), tik.scope_ubuf, 'ori_scale_w_ub')
        self.tik_instance.data_move(ori_scale_w_ub, self.scale_w, 0, 1, 1, 0, 0)
        self.scale_w_ub = ori_scale_w_ub

        if self.calc_precision != self.scale_w_dtype:
            scale_w_ub = self.tik_instance.Tensor(self.calc_precision,
                (Constant.BLOCK_BYTE_SIZE//self.input_scale_w_byte_size,), tik.scope_ubuf, 'scale_w_ub')
            if self.soc_version in ['SD3403']:
                ori_scale_w_scalar = self.tik_instance.Scalar(self.scale_w_dtype, 'ori_scale_w_scalar')
                ori_scale_w_scalar.set_as(ori_scale_w_ub[0])
                scale_w_scalar = self.tik_instance.Scalar(self.calc_precision, 'scale_w_scalar')
                self.tik_instance.scalar_conv('', scale_w_scalar, ori_scale_w_scalar)
                scale_w_ub[0].set_as(scale_w_scalar)
            else:
                self.tik_instance.vec_conv(1, '', scale_w_ub, ori_scale_w_ub, 1, 1, 1)
            self.scale_w_ub = scale_w_ub

        if self.soc_version in ['Ascend310', 'SD3403']:
            # rec for scale_w
            self.scale_w_rec = self.tik_instance.Tensor(self.calc_precision, (self.data_each_block,),
                tik.scope_ubuf, 'scale_w_rec')
            self._calculate_rec_for_tensor(self.scale_w_rec, self.scale_w_ub, 1)

    def _init_loss(self):
        if self.soc_version == 'SD3403':
            if self.per_tensor_loss:
                final_loss_shape = (Constant.N_STEP,)
                mask = Constant.N_STEP
                repeats = 1
            else:
                final_loss_shape = (Constant.N_STEP, self.dim_c0)
                mask = self.int32_vector_mask_max
                repeats = Constant.N_STEP * self.dim_c0 // self.int32_vector_mask_max
            self.final_loss_1 = self.tik_instance.Tensor(self.loss_precision, final_loss_shape,
                tik.scope_ubuf, 'final_loss_1')
            self.final_loss_2 = self.tik_instance.Tensor(self.loss_precision, final_loss_shape,
                tik.scope_ubuf, 'final_loss_2')
            self.final_loss_3 = self.tik_instance.Tensor(self.loss_precision, final_loss_shape,
                tik.scope_ubuf, 'final_loss_3')
            self.final_loss_4 = self.tik_instance.Tensor(self.loss_precision, final_loss_shape,
                tik.scope_ubuf, 'final_loss_4')
            self.tik_instance.vec_dup(mask, self.final_loss_1, 0, repeats, self.blk_per_rep)
            self.tik_instance.vec_dup(mask, self.final_loss_2, 0, repeats, self.blk_per_rep)
            self.tik_instance.vec_dup(mask, self.final_loss_3, 0, repeats, self.blk_per_rep)
            self.tik_instance.vec_dup(mask, self.final_loss_4, 0, repeats, self.blk_per_rep)
        else:
            if self.per_tensor_loss:
                # define total loss on ub
                self.main_loss_ub = self.tik_instance.Tensor(self.loss_precision, (Constant.N_STEP,),
                    tik.scope_ubuf, 'main_loss_ub')
                self.remain_loss_ub = self.tik_instance.Tensor(self.loss_precision, (Constant.N_STEP,),
                    tik.scope_ubuf, 'remain_loss_ub')
                self.tik_instance.vec_dup(Constant.N_STEP, self.main_loss_ub, 0, 1, self.blk_per_rep)
                self.tik_instance.vec_dup(Constant.N_STEP, self.remain_loss_ub, 0, 1, self.blk_per_rep)
                # define one loss for each candidate n
                self.one_remain_loss_ub = self.tik_instance.Tensor(self.loss_precision, (self.data_each_block,),
                    tik.scope_ubuf, 'one_remain_loss_ub')
                self.one_main_loss_ub = self.tik_instance.Tensor(self.loss_precision, (self.data_each_block,),
                    tik.scope_ubuf, 'one_main_loss_ub')
                self.tik_instance.vec_dup(1, self.one_remain_loss_ub, 0, 1, self.blk_per_rep)
                self.tik_instance.vec_dup(1, self.one_main_loss_ub, 0, 1, self.blk_per_rep)
            else:
                self.main_loss_ub = self.tik_instance.Tensor(self.loss_precision, (Constant.N_STEP, self.dim_c0),
                    tik.scope_ubuf, 'main_loss_ub')
                self.remain_loss_ub = self.tik_instance.Tensor(self.loss_precision, (Constant.N_STEP, self.dim_c0),
                    tik.scope_ubuf, 'remain_loss_ub')
                repeat_time = Constant.N_STEP * self.dim_c0 // self.vector_mask_max
                self.tik_instance.vec_dup(self.vector_mask_max, self.main_loss_ub, 0, repeat_time, self.blk_per_rep)
                self.tik_instance.vec_dup(self.vector_mask_max, self.remain_loss_ub, 0, repeat_time, self.blk_per_rep)

    def _reset_per_channel_loss_to_zeros(self):
        if self.soc_version == 'SD3403':
            self._dup_by_repeat(self.final_loss_1, 0, Constant.N_STEP*self.dim_c0, self.int32_vector_mask_max)
            self._dup_by_repeat(self.final_loss_2, 0, Constant.N_STEP*self.dim_c0, self.int32_vector_mask_max)
            self._dup_by_repeat(self.final_loss_3, 0, Constant.N_STEP*self.dim_c0, self.int32_vector_mask_max)
            self._dup_by_repeat(self.final_loss_4, 0, Constant.N_STEP*self.dim_c0, self.int32_vector_mask_max)
        else:
            self._dup_by_repeat(self.main_loss_ub, 0, Constant.N_STEP*self.dim_c0, self.vector_mask_max)
            self._dup_by_repeat(self.remain_loss_ub, 0, Constant.N_STEP*self.dim_c0, self.vector_mask_max)

    def _compute_quant_output_and_loss_nd(self, core_ind, input_data_current_core):
        loop_time = input_data_current_core // self.ub_tensor_size
        last_num = input_data_current_core % self.ub_tensor_size
        offset_per_loop = self.ub_tensor_size
        threads = 2 if loop_time > 1 else 1
        self._load_nd_scale_w_to_ub()
        if loop_time > 0:
            # cur_valid_data_num: valid data num of current loop
            self.cur_valid_data_num = self.ub_tensor_size
            with self.tik_instance.new_stmt_scope():
                move_offset = core_ind * self.data_num_each_core
                with self.tik_instance.for_range(0, loop_time, thread_num=threads) as loop_index:
                    data_quant_ub = self.tik_instance.Tensor(self.data_quant_type, (self.ub_tensor_size,),
                        tik.scope_ubuf, 'data_quant_ub')
                    move_offset += loop_index * offset_per_loop
                    data_ub = self._load_nd_data_to_ub(move_offset, self.ub_tensor_size)
                    self._compute_quant_output(data_ub, data_quant_ub)
                    self._compute_loss(data_ub, data_quant_ub)
        if last_num > 0:
            self.cur_valid_data_num = last_num
            with self.tik_instance.new_stmt_scope():
                data_quant_ub = self.tik_instance.Tensor(self.data_quant_type, (self.ub_tensor_size,),
                    tik.scope_ubuf, 'data_quant_ub')
                move_offset = core_ind * self.data_num_each_core + loop_time * offset_per_loop
                data_ub = self._load_nd_data_to_ub(move_offset, last_num)
                self._compute_quant_output(data_ub, data_quant_ub)
                self._compute_loss(data_ub, data_quant_ub)

    def _compute_quant_output_and_loss_5hd(self, core_ind, input_hw_current_core):
        loop_time = input_hw_current_core // self.ub_tensor_hw
        last_hw_num = input_hw_current_core % self.ub_tensor_hw
        offset_per_loop = self.ub_tensor_hw * self.dim_c0
        threads = 2 if loop_time > 1 else 1

        with self.tik_instance.for_range(0, self.dim_c1) as c1_index:
            offset = c1_index * self.dim_c0
            self._load_5hd_scale_w_to_ub(offset)
            with self.tik_instance.for_range(0, self.loop_n) as n_index:
                if loop_time > 0:
                    # cur_valid_data_num: valid data num of current loop
                    self.cur_valid_data_num = self.ub_tensor_hw * self.dim_c0 * self.new_n
                    with self.tik_instance.new_stmt_scope():
                        move_offset = core_ind * self.dim_c0 * self.data_hw_each_core
                        with self.tik_instance.for_range(0, loop_time, thread_num=threads) as loop_index:
                            data_quant_ub = self.tik_instance.Tensor(self.data_quant_type, (self.ub_tensor_size,),
                                tik.scope_ubuf, 'data_quant_ub')
                            move_offset += (loop_index * offset_per_loop + \
                                (n_index * self.dim_c1 + c1_index) * self.dim_h * self.dim_w * self.dim_c0)
                            data_ub = self._load_5hd_data_to_ub(move_offset, self.ub_tensor_hw)
                            self._compute_quant_output(data_ub, data_quant_ub)
                            self._compute_loss(data_ub, data_quant_ub)
                if last_hw_num > 0:
                    with self.tik_instance.new_stmt_scope():
                        move_offset = core_ind * self.dim_c0 * self.data_hw_each_core + (loop_time * offset_per_loop
                            + (n_index * self.dim_c1 + c1_index) * self.dim_h * self.dim_w * self.dim_c0)
                        self.cur_valid_data_num = last_hw_num * self.dim_c0 * self.new_n
                        data_quant_ub = self.tik_instance.Tensor(self.data_quant_type, (self.ub_tensor_size,),
                            tik.scope_ubuf, 'data_quant_ub')
                        data_ub = self._load_5hd_data_to_ub(move_offset, last_hw_num)
                        self._compute_quant_output(data_ub, data_quant_ub)
                        self._compute_loss(data_ub, data_quant_ub)
            # if per channel, for each c1, move c0-size loss out
            if not self.per_tensor_loss:
                self._move_per_channel_loss_out(core_ind, offset)
                self._reset_per_channel_loss_to_zeros()

    def _compute_quant_output_and_loss(self, core_ind):
        # load scale_d
        self._load_scale_d_to_ub()
        # define loss and initial by zero
        self._init_loss()

        if self.per_tensor_quant:
            with self.tik_instance.if_scope(core_ind == (self.aicore_num - 1)):
                input_data_current_core = self.data_num_each_core + self.data_num_last_core
                self._compute_quant_output_and_loss_nd(core_ind, input_data_current_core)
            with self.tik_instance.else_scope():
                input_data_current_core = self.data_num_each_core
                self._compute_quant_output_and_loss_nd(core_ind, input_data_current_core)
        else:
            with self.tik_instance.if_scope(core_ind == (self.aicore_num - 1)):
                input_hw_current_core = self.data_hw_each_core + self.data_hw_last_core
                self._compute_quant_output_and_loss_5hd(core_ind, input_hw_current_core)
            with self.tik_instance.else_scope():
                input_hw_current_core = self.data_hw_each_core
                self._compute_quant_output_and_loss_5hd(core_ind, input_hw_current_core)
        # if per tensor, move loss out after all computation is finished
        if self.per_tensor_loss:
            self._move_per_tensor_loss_out(core_ind)

    def _calc_work_tensor_size(self, repeat_times, mask, src_rep_stride, dtype):
        if dtype not in ['float16', 'float32']:
            raise ValueError('The dtype of vec_rec_high_preci must be float16 or float32!')
        dtype_byte_size = cce.cce_intrin.get_bit_len(dtype) // 8
        block_len = Constant.BLOCK_BYTE_SIZE // dtype_byte_size
        work_size_factor = 8 // dtype_byte_size
        src_extent_size = (repeat_times - 1) * src_rep_stride * block_len + mask
        wk_size_unit = ((src_extent_size + block_len - 1) // block_len) * block_len
        work_tensor_size = work_size_factor * wk_size_unit
        return work_tensor_size

    def _calculate_rec_for_tensor(self, dst, src, mask):
        '''mask is no more than vector_mask_max, only repeat once'''
        with self.tik_instance.new_stmt_scope():
            if self.soc_version == 'SD3403':
                # SD3403 do not support vec_rec, use 1.0 divide src
                ones = self.tik_instance.Tensor(self.calc_precision, (self.vector_mask_max,), tik.scope_ubuf, 'ones')
                self.tik_instance.vec_dup(mask, ones, 1, 1, self.blk_per_rep)
                self.tik_instance.vdiv(mask, dst, ones, src, 1, 1, 1, 1,
                    self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
            else:
                # for other soc, use vec_rec_high_preci
                work_tensor_size = self._calc_work_tensor_size(1, mask, self.blk_per_rep, self.calc_precision)
                work_tensor = self.tik_instance.Tensor(self.calc_precision, (work_tensor_size,),
                    tik.scope_ubuf, 'work_tensor')
                self.tik_instance.vec_rec_high_preci(mask, dst, src, work_tensor, 1,
                    self.blk_per_rep, self.blk_per_rep)

    def _data_multiply_scale_w_rec(self, data_ub):
        # multiply data and scale_w_rec
        if self.per_tensor_quant:
            scale_w_scalar = self.tik_instance.Scalar(self.calc_precision, 'scale_w_scalar')
            if self.soc_version == 'Ascend310':
                scale_w_scalar.set_as(self.scale_w_rec[0])
                self.tik_instance.vec_muls(self.vector_mask_max, data_ub, data_ub, scale_w_scalar,
                    self.data_repeats, self.blk_per_rep, self.blk_per_rep)
            else:
                scale_w_scalar.set_as(self.scale_w_ub[0])
                scale_w_dup = self.tik_instance.Tensor(self.calc_precision, (self.vector_mask_max,),
                    tik.scope_ubuf, 'scale_w_dup')
                self.tik_instance.vec_dup(self.vector_mask_max, scale_w_dup, scale_w_scalar, 1, self.blk_per_rep)
                self.tik_instance.vdiv(self.vector_mask_max, data_ub, data_ub, scale_w_dup,
                    self.data_repeats, 1, 1, 1, self.blk_per_rep, self.blk_per_rep, 0)
        else:
            # calc_precision is fp32, C0 is 16, but a block only has 8 fp32
            nhwc0_repeat_time = self.new_n * self.ub_tensor_hw * self.dim_c0 // 2 // self.vector_mask_max
            nhwc0_last_num = self.new_n * self.ub_tensor_hw * self.dim_c0 // 2 % self.vector_mask_max
            if self.soc_version == 'Ascend310':
                if nhwc0_repeat_time > 0:
                    self.tik_instance.vmul(self.vector_mask_max, data_ub, data_ub, self.scale_w_rec,
                        nhwc0_repeat_time, 2, 2, 0, self.blk_per_rep*2, self.blk_per_rep*2, 0)
                    self.tik_instance.vmul(self.vector_mask_max, data_ub[self.dim_c0//2], data_ub[self.dim_c0//2],
                        self.scale_w_rec[self.dim_c0//2], nhwc0_repeat_time, 2, 2, 0,
                        self.blk_per_rep*2, self.blk_per_rep*2, 0)
                if nhwc0_last_num > 0:
                    nhwc0_offset = nhwc0_repeat_time * self.vector_mask_max * 2
                    self.tik_instance.vmul(nhwc0_last_num, data_ub[nhwc0_offset], data_ub[nhwc0_offset],
                        self.scale_w_rec, 1, 2, 2, 0, self.blk_per_rep*2, self.blk_per_rep*2, 0)
                    self.tik_instance.vmul(nhwc0_last_num, data_ub[nhwc0_offset+self.dim_c0//2],
                        data_ub[nhwc0_offset+self.dim_c0//2], self.scale_w_rec[self.dim_c0//2],
                        1, 2, 2, 0, self.blk_per_rep*2, self.blk_per_rep*2, 0)
            else:
                if nhwc0_repeat_time > 0:
                    self.tik_instance.vdiv(self.vector_mask_max, data_ub, data_ub, self.scale_w_ub,
                        nhwc0_repeat_time, 2, 2, 0, self.blk_per_rep*2, self.blk_per_rep*2, 0)
                    self.tik_instance.vdiv(self.vector_mask_max, data_ub[self.dim_c0//2], data_ub[self.dim_c0//2],
                        self.scale_w_ub[self.dim_c0//2], nhwc0_repeat_time, 2, 2, 0,
                        self.blk_per_rep*2, self.blk_per_rep*2, 0)
                if nhwc0_last_num > 0:
                    nhwc0_offset = nhwc0_repeat_time * self.vector_mask_max * 2
                    self.tik_instance.vdiv(nhwc0_last_num, data_ub[nhwc0_offset], data_ub[nhwc0_offset],
                        self.scale_w_ub, 1, 2, 2, 0, self.blk_per_rep*2, self.blk_per_rep*2, 0)
                    self.tik_instance.vdiv(nhwc0_last_num, data_ub[nhwc0_offset+self.dim_c0//2],
                        data_ub[nhwc0_offset+self.dim_c0//2], self.scale_w_ub[self.dim_c0//2],
                        1, 2, 2, 0, self.blk_per_rep*2, self.blk_per_rep*2, 0)

    def _compute_quant_output(self, data_ub, data_quant_ub):
        if self.soc_version == 'SD3403':
            with self.tik_instance.new_stmt_scope():
                self._scale_d_mul_scale_w_by_split()
                self._data_mul_dw_by_split(data_ub, data_quant_ub)
        else:
            if self.soc_version == 'Ascend310':
                # multiply data and scale_d_rec
                scale_d_rec_scalar = self.tik_instance.Scalar(self.calc_precision, 'scale_d_rec_scalar')
                scale_d_rec_scalar.set_as(self.scale_d_rec[0])
                self.tik_instance.vec_muls(self.vector_mask_max, data_ub, data_ub, scale_d_rec_scalar,
                    self.data_repeats, self.blk_per_rep, self.blk_per_rep)
            else:
                # data divide scale_d
                scale_d_scalar = self.tik_instance.Scalar(self.calc_precision, 'scale_d_scalar')
                scale_d_scalar.set_as(self.scale_d_ub[0])
                scale_d_dup = self.tik_instance.Tensor(self.calc_precision, (self.vector_mask_max,),
                    tik.scope_ubuf, 'scale_d_dup')
                self.tik_instance.vec_dup(self.vector_mask_max, scale_d_dup, scale_d_scalar, 1, self.blk_per_rep)
                self.tik_instance.vdiv(self.vector_mask_max, data_ub, data_ub, scale_d_dup,
                    self.data_repeats, 1, 1, 1, self.blk_per_rep, self.blk_per_rep, 0)
            self._data_multiply_scale_w_rec(data_ub)
            self._round_fp32_precisely(self.vector_mask_max, data_quant_ub, data_ub, self.ub_tensor_size)

    def _round_fp32_precisely(self, mask, dst, src, data_num, mode='round'):
        if self.soc_version == 'Ascend310':
            self._round_fp32_by_split_to_fp16(mask, dst, src, data_num, mode)
        else:
            self._round_fp32_directly(mask, dst, src, data_num, mode)

    def _round_fp32_by_split_to_fp16(self, mask, dst, src, data_num, mode='round'):
        repeats = data_num // mask
        last_num = data_num % mask
        offset = mask * repeats
        data_remd = self.tik_instance.Tensor(self.calc_precision, (data_num,), tik.scope_ubuf, 'data_remd')
        with self.tik_instance.new_stmt_scope():
            data_quot1 = self.tik_instance.Tensor(self.calc_precision, (data_num,), tik.scope_ubuf, 'data_quot1')
            data_quot2 = self.tik_instance.Tensor(self.calc_precision, (data_num,), tik.scope_ubuf, 'data_quot2')
            self._split_fp32_to_fp16(src, data_num, data_quot1, data_quot2, data_remd)
        # src sub data_remd
        if repeats > 0:
            self.tik_instance.vec_sub(mask, src, src, data_remd, repeats,
                self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
        if last_num > 0:
            self.tik_instance.vec_sub(last_num, src[offset],
                src[offset], data_remd[offset], 1, self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
        self._round_fp32_directly(mask, dst, data_remd, data_num, mode)
        if repeats > 0:
            self.tik_instance.vec_add(mask, src, src, data_remd, repeats,
                self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
        if last_num > 0:
            self.tik_instance.vec_add(last_num, src[offset],
                src[offset], data_remd[offset], 1, self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)

    def _round_fp32_directly(self, mask, dst, src, data_num, mode='round'):
        repeats = data_num // mask
        last_num = data_num % mask
        offset = mask * repeats
        with self.tik_instance.new_stmt_scope():
            if self.soc_version == 'Ascend310':
                tmp_fp16 = self.tik_instance.Tensor('float16', (data_num,), tik.scope_ubuf, 'tmp_fp16')
                if repeats > 0:
                    # fp32 conv to fp16
                    self.tik_instance.vec_conv(mask, '', tmp_fp16, src, repeats, self.blk_per_rep//2, self.blk_per_rep)
                    # fp16 round to int32
                    self.tik_instance.vec_conv(mask, mode, dst, tmp_fp16,
                        repeats, self.blk_per_rep, self.blk_per_rep//2)
                    # int32 conv to fp16
                    self.tik_instance.vec_conv(mask, '', tmp_fp16, dst, repeats,
                        self.blk_per_rep//2, self.blk_per_rep, 1.0)
                    # fp16 conv to fp32
                    self.tik_instance.vec_conv(mask, '', src, tmp_fp16, repeats, self.blk_per_rep, self.blk_per_rep//2)
                if last_num > 0:
                    offset = repeats * mask
                    self.tik_instance.vec_conv(last_num, '', tmp_fp16[offset], src[offset],
                        1, self.blk_per_rep//2, self.blk_per_rep)
                    self.tik_instance.vec_conv(last_num, mode, dst[offset], tmp_fp16[offset],
                        1, self.blk_per_rep, self.blk_per_rep//2)
                    # int32 conv to fp16
                    self.tik_instance.vec_conv(last_num, '', tmp_fp16[offset], dst[offset],
                        1, self.blk_per_rep//2, self.blk_per_rep, 1.0)
                    # fp16 conv to fp32
                    self.tik_instance.vec_conv(last_num, '', src[offset], tmp_fp16[offset],
                        1, self.blk_per_rep, self.blk_per_rep//2)
            else:
                if repeats > 0:
                    # fp32 conv to int32(round)
                    self.tik_instance.vec_conv(mask, mode, dst, src, repeats, self.blk_per_rep, self.blk_per_rep)
                    # int32 conv to fp32
                    self.tik_instance.vec_conv(mask, '', src, dst, repeats, self.blk_per_rep, self.blk_per_rep)
                if last_num > 0:
                    self.tik_instance.vec_conv(last_num, mode, dst[offset], src[offset],
                        1, self.blk_per_rep, self.blk_per_rep)
                    self.tik_instance.vec_conv(last_num, '', src[offset], dst[offset],
                        1, self.blk_per_rep, self.blk_per_rep)

    def _inner_split_fp16_integer_decimal(self, mask, src_fp16, dst_int32, dst_intg_fp16, dst_decm, repeats):
        # src_fp16 round to dst_intg(int32)
        self.tik_instance.vec_conv(mask, 'round', dst_int32, src_fp16, repeats, self.blk_per_rep, self.blk_per_rep//2)
        with self.tik_instance.new_stmt_scope():
            # dst_intg(int32) conv to tmp_int16
            tmp_int16 = self.tik_instance.Tensor('int16', (mask*repeats,), tik.scope_ubuf, 'tmp_int16')
            self.tik_instance.vcbd(mask, tmp_int16, dst_int32, repeats, 1, 1, self.blk_per_rep//2, self.blk_per_rep)
            # tmp_int16 conv to fp16
            self.tik_instance.vec_conv(mask, '', dst_intg_fp16, tmp_int16, repeats,
                self.blk_per_rep//2, self.blk_per_rep//2)
        # for src_fp16 inside int16 boundary, dst_decm is the decimal part
        # else, dst_decm is the clipped integer and decimal
        self.tik_instance.vec_sub(mask, dst_decm, src_fp16, dst_intg_fp16, repeats,
            self.blk_per_rep//2, self.blk_per_rep//2, self.blk_per_rep//2)

    def _split_fp16_integer_decimal(self, mask, src_fp16, dst_decm, dst_int32, repeats):
        dst_intg_fp16 = self.tik_instance.Tensor('float16', (mask*repeats,), tik.scope_ubuf, 'dst_intg_fp16')
        with self.tik_instance.new_stmt_scope():
            self._inner_split_fp16_integer_decimal(mask, src_fp16, dst_int32, dst_intg_fp16, dst_decm, repeats)
            dst_intg_fp16_1 = self.tik_instance.Tensor('float16', (mask*repeats,), tik.scope_ubuf, 'dst_intg_fp16_1')
            self._inner_split_fp16_integer_decimal(mask, dst_decm, dst_int32, dst_intg_fp16_1, dst_decm, repeats)
            self.tik_instance.vec_add(mask, dst_intg_fp16, dst_intg_fp16, dst_intg_fp16_1, repeats,
                self.blk_per_rep//2, self.blk_per_rep//2, self.blk_per_rep//2)
        self.tik_instance.vec_conv(mask, 'round', dst_int32, dst_intg_fp16, repeats,
            self.blk_per_rep, self.blk_per_rep//2)
        return dst_intg_fp16

    def _scale_d_mul_scale_w_by_split(self):
        if self.per_tensor_quant:
            mask_w = 1
        else:
            mask_w = self.dim_c0
        # split scale_d_rec to integer and decimal
        scale_d_decm = self.tik_instance.Tensor(self.calc_precision, (self.data_each_block,),
            tik.scope_ubuf, 'scale_d_decm')
        scale_d_int32 = self.tik_instance.Tensor('int32', (self.data_each_block,), tik.scope_ubuf, 'scale_d_int32')
        scale_d_intg_fp16 = self._split_fp16_integer_decimal(1, self.scale_d_rec, scale_d_decm, scale_d_int32, 1)
        # split scale_w_rec to integer and decimal
        scale_w_decm = self.tik_instance.Tensor(self.calc_precision, (self.data_each_block,),
            tik.scope_ubuf, 'scale_w_decm')
        self.scale_w_int32 = self.tik_instance.Tensor('int32', (self.data_each_block,), tik.scope_ubuf, 'scale_w_int32')
        self.scale_w_intg_fp16 = self._split_fp16_integer_decimal(mask_w, self.scale_w_rec,
            scale_w_decm, self.scale_w_int32, 1)
        # int32 mul int32
        d_int32_scalar = self.tik_instance.Scalar('int32', 'd_int32_scalar')
        d_int32_scalar.set_as(scale_d_int32[0])
        self.dw_int32 = self.tik_instance.Tensor('int32', (self.data_each_block,), tik.scope_ubuf, 'dw_int32')
        self.tik_instance.vec_muls(mask_w, self.dw_int32, self.scale_w_int32, d_int32_scalar, 1, 0, 0)

        # scale_d_intg_fp16 mul scale_w_decm
        self.d_intg_fp16_scalar = self.tik_instance.Scalar('float16', 'd_intg_fp16_scalar')
        self.d_intg_fp16_scalar.set_as(scale_d_intg_fp16[0])
        dw_fp16_1 = self.tik_instance.Tensor(self.calc_precision, (self.data_each_block,),
            tik.scope_ubuf, 'dw_fp16_1')
        self.tik_instance.vec_muls(mask_w, dw_fp16_1, scale_w_decm, self.d_intg_fp16_scalar, 1, 0, 0)
        dw_fp16_1_decm =  self.tik_instance.Tensor(self.calc_precision, (self.data_each_block,),
            tik.scope_ubuf, 'dw_fp16_1_decm')
        dw_fp16_1_int32 = self.tik_instance.Tensor('int32', (self.data_each_block,), tik.scope_ubuf, 'dw_fp16_1_int32')
        self.dw_fp16_1_intg = self._split_fp16_integer_decimal(mask_w, dw_fp16_1, dw_fp16_1_decm, dw_fp16_1_int32, 1)

        # scale_d_decm mul scale_w_rec
        dw_fp16_2 = self.tik_instance.Tensor(self.calc_precision, (self.data_each_block,),
            tik.scope_ubuf, 'dw_fp16_2')
        scale_d_decm_scalar = self.tik_instance.Scalar('float16', 'scale_d_decm_scalar')
        scale_d_decm_scalar.set_as(scale_d_decm[0])
        self.tik_instance.vec_muls(mask_w, dw_fp16_2, self.scale_w_rec, scale_d_decm_scalar, 1, 0, 0)
        dw_fp16_2_decm =  self.tik_instance.Tensor(self.calc_precision, (self.data_each_block,),
            tik.scope_ubuf, 'dw_fp16_2_decm')
        dw_fp16_2_int32 = self.tik_instance.Tensor('int32', (self.data_each_block,), tik.scope_ubuf, 'dw_fp16_2_int32')
        self.dw_fp16_2_intg = self._split_fp16_integer_decimal(mask_w, dw_fp16_2, dw_fp16_2_decm, dw_fp16_2_int32, 1)

        # add dw_fp16_1_int32 and dw_fp16_2_int32 to dw_int32
        self.tik_instance.vec_add(mask_w, self.dw_int32, self.dw_int32, dw_fp16_1_int32, 1, 0, 0, 0)
        self.tik_instance.vec_add(mask_w, self.dw_int32, self.dw_int32, dw_fp16_2_int32, 1, 0, 0, 0)
        # add dw_fp16_1_decm and dw_fp16_2_decm
        self.tik_instance.vec_add(mask_w, dw_fp16_1_decm, dw_fp16_1_decm, dw_fp16_2_decm, 1, 0, 0, 0)
        self.dw_fp16_decm =  self.tik_instance.Tensor(self.calc_precision, (self.data_each_block,),
            tik.scope_ubuf, 'dw_fp16_decm')
        dw_fp16_3_int32 = self.tik_instance.Tensor('int32', (self.data_each_block,), tik.scope_ubuf, 'dw_fp16_3_int32')
        dw_fp16_3_intg = self._split_fp16_integer_decimal(mask_w, dw_fp16_1_decm,
            self.dw_fp16_decm, dw_fp16_3_int32, 1)
        self.tik_instance.vec_add(mask_w, self.dw_int32, self.dw_int32, dw_fp16_3_int32, 1, 0, 0, 0)
        self.tik_instance.vec_add(mask_w, self.dw_fp16_1_intg, self.dw_fp16_1_intg, dw_fp16_3_intg, 1, 0, 0, 0)
        # dw_int32, d_intg_fp16_scalar, scale_w_intg_fp16, scale_w_int32, dw_fp16_1_intg,
        # dw_fp16_2_intg, dw_fp16_decm will be reused

    def _mul_by_per_tensor_or_not(self, dtype, dst, src0, src1, data_num):
        '''
        multiply src0 and src1 to dst
        dtype is float16 or int32
        src0 is data tensor, src1 is scale tensor
        if per tensor, src1 has only one valid value
        else, src1 has dim_c0 valid value
        '''
        if dtype == 'float16':
            mask = self.vector_mask_max
        else:
            mask = self.int32_vector_mask_max
        if self.per_tensor_quant:
            src_scalar = self.tik_instance.Scalar(dtype, 'src_scalar')
            src_scalar.set_as(src1[0])
            repeats = data_num // mask
            self.tik_instance.vec_muls(mask, dst, src0, src_scalar, repeats, self.blk_per_rep, self.blk_per_rep)
        else:
            if dtype == 'float16':
                repeats = data_num // mask
                self.tik_instance.vmul(mask, dst, src0, src1, repeats, 1, 1, 0, self.blk_per_rep, self.blk_per_rep, 0)
            else:
                repeats = data_num // 2 // mask
                self.tik_instance.vmul(mask, dst, src0, src1,
                    repeats, 2, 2, 0, self.blk_per_rep*2, self.blk_per_rep*2, 0)
                self.tik_instance.vmul(mask, dst[self.dim_c0//2], src0[self.dim_c0//2], src1[self.dim_c0//2],
                    repeats, 2, 2, 0, self.blk_per_rep*2, self.blk_per_rep*2, 0)

    def _data_mul_dw_by_split(self, data_ub, data_quant_ub):
        '''used by SD3403'''
        repeats_int32 = self.ub_tensor_size // self.int32_vector_mask_max
        repeats_fp16 = self.ub_tensor_size // self.vector_mask_max
        # first split data
        data_decm = self.tik_instance.Tensor(self.calc_precision, (self.ub_tensor_size,), tik.scope_ubuf, 'data_decm')
        with self.tik_instance.new_stmt_scope():
            _ = self._split_fp16_integer_decimal(self.int32_vector_mask_max, data_ub,
                data_decm, data_quant_ub, repeats_int32)

        # data_quant_ub mul dw_int32
        self._mul_by_per_tensor_or_not('int32', data_quant_ub, data_quant_ub, self.dw_int32, self.ub_tensor_size)

        # data_ub mul dw_fp16_decm
        self._mul_by_per_tensor_or_not('float16', data_ub, data_ub, self.dw_fp16_decm, self.ub_tensor_size)
        data_fp16 = data_ub

        # split data_fp16
        tmp_int32 = self.tik_instance.Tensor('int32', (self.ub_tensor_size,), tik.scope_ubuf, 'tmp_int32')
        with self.tik_instance.new_stmt_scope():
            _ = self._split_fp16_integer_decimal(self.int32_vector_mask_max, data_fp16,
                data_fp16, tmp_int32, repeats_int32)
        self.tik_instance.vec_add(self.int32_vector_mask_max, data_quant_ub, data_quant_ub, tmp_int32,
            repeats_int32, self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)

        # data_decm mul (scale_d_intg_fp16*scale_w_intg_fp16+dw_fp16_1_intg+dw_fp16_2_intg)
        # step1. multiplication expansion
        # data_decm mul scale_d_intg_fp16 -> data_decm_1
        data_decm_1 = self.tik_instance.Tensor(self.calc_precision, (self.ub_tensor_size,),
            tik.scope_ubuf, 'data_decm_1')
        self.tik_instance.vec_muls(self.vector_mask_max, data_decm_1, data_decm, self.d_intg_fp16_scalar,
            self.data_repeats, self.blk_per_rep, self.blk_per_rep)

        # step2. data_decm_1 mul scale_w_intg_fp16 -> tmp_int32 plus tmp_decm
        tmp_decm = self.tik_instance.Tensor(self.calc_precision, (self.ub_tensor_size,), tik.scope_ubuf, 'tmp_decm')
        with self.tik_instance.new_stmt_scope():
            _ = self._split_fp16_integer_decimal(self.int32_vector_mask_max, data_decm_1,
                tmp_decm, tmp_int32, repeats_int32)
        # mul int32 and add up to data_quant_ub
        self._mul_by_per_tensor_or_not('int32', tmp_int32, tmp_int32, self.scale_w_int32, self.ub_tensor_size)
        self.tik_instance.vec_add(self.int32_vector_mask_max, data_quant_ub, data_quant_ub, tmp_int32,
            repeats_int32, self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
        # mul fp16 and add up tp data_fp16
        self._mul_by_per_tensor_or_not('float16', tmp_decm, tmp_decm, self.scale_w_intg_fp16, self.ub_tensor_size)
        self.tik_instance.vec_add(self.vector_mask_max, data_fp16, data_fp16, tmp_decm,
            repeats_fp16, self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)

        # step3. data_decm mul dw_fp16_1_intg -> data_decm_1
        self._mul_by_per_tensor_or_not('float16', data_decm_1, data_decm, self.dw_fp16_1_intg, self.ub_tensor_size)
        # split data_decm_1, and add up to data_quant_ub and data_fp16
        with self.tik_instance.new_stmt_scope():
            _ = self._split_fp16_integer_decimal(self.int32_vector_mask_max, data_decm_1,
                tmp_decm, tmp_int32, repeats_int32)
        self.tik_instance.vec_add(self.int32_vector_mask_max, data_quant_ub, data_quant_ub, tmp_int32,
            repeats_int32, self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
        self.tik_instance.vec_add(self.vector_mask_max, data_fp16, data_fp16, tmp_decm,
            repeats_fp16, self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)

        # step4. data_decm mul dw_fp16_2_intg -> data_decm
        self._mul_by_per_tensor_or_not('float16', data_decm, data_decm, self.dw_fp16_2_intg, self.ub_tensor_size)
        # split data_decm, and add up to data_quant_ub and data_fp16
        with self.tik_instance.new_stmt_scope():
            _ = self._split_fp16_integer_decimal(self.int32_vector_mask_max, data_decm,
                tmp_decm, tmp_int32, repeats_int32)
        self.tik_instance.vec_add(self.int32_vector_mask_max, data_quant_ub, data_quant_ub, tmp_int32,
            repeats_int32, self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
        self.tik_instance.vec_add(self.vector_mask_max, data_fp16, data_fp16, tmp_decm,
            repeats_fp16, self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
        # final result is stored in data_quant_ub and data_fp16

        # data_fp16 round to int32
        self.tik_instance.vec_conv(self.int32_vector_mask_max, 'round', tmp_int32, data_fp16,
            repeats_int32, self.blk_per_rep, self.blk_per_rep//2)
        # add up to data_quant_ub
        self.tik_instance.vec_add(self.int32_vector_mask_max, data_quant_ub, data_quant_ub, tmp_int32,
            repeats_int32, self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)

    def _compute_loss(self, data_quant_ub, data_quant_ub_int32):
        # define 2**1 2**2 ... 2**16 as shift_factor_tensor
        shift_factor_tensor = self.tik_instance.Tensor(self.loss_precision, (Constant.N_STEP,),
            tik.scope_ubuf, 'shift_factor_tensor')
        tmp_scalar = self.tik_instance.Scalar(self.loss_precision, 'tmp_scalar')
        for i in range(0, Constant.N_STEP):
            tmp_scalar.set_as(Constant.BASE_TWO**(i+1))
            shift_factor_tensor[i].set_as(tmp_scalar)
        if self.soc_version not in ['SD3403']:
            # data_shift_ub: used to store data after shift-N and also loss
            data_shift_ub = self.tik_instance.Tensor(self.calc_precision, (self.ub_tensor_size,),
                tik.scope_ubuf, 'data_shift_ub')
            # define shift_factor_tensor_rec
            shift_factor_tensor_rec = self.tik_instance.Tensor(self.calc_precision, (Constant.N_STEP,),
                tik.scope_ubuf, 'shift_factor_tensor_rec')
            for i in range(0, Constant.N_STEP):
                shift_factor_tensor_rec[i].set_as(1/(Constant.BASE_TWO**(i+1)))
            self.int16_max_tensor = self.tik_instance.Tensor(self.calc_precision, (self.vector_mask_max,),
                tik.scope_ubuf, 'int16_max_tensor')
            self.tik_instance.vec_dup(self.vector_mask_max, self.int16_max_tensor, Constant.INT16_MAX,
                1, self.blk_per_rep)
            self.int16_min_tensor = self.tik_instance.Tensor(self.calc_precision, (self.vector_mask_max,),
                tik.scope_ubuf, 'int16_min_tensor')
            self.tik_instance.vec_dup(self.vector_mask_max, self.int16_min_tensor, Constant.INT16_MIN,
                1, self.blk_per_rep)

            shift_factor_scalar = self.tik_instance.Scalar(self.calc_precision, 'shift_factor_scalar')
            shift_factor_rec_scalar = self.tik_instance.Scalar(self.calc_precision, 'shift_factor_rec_scalar')
            # iterate for each shift_n
            with self.tik_instance.for_range(0, Constant.N_STEP) as n_step:
                shift_factor_scalar.set_as(shift_factor_tensor[n_step])
                shift_factor_rec_scalar.set_as(shift_factor_tensor_rec[n_step])
                self._compute_loss_for_each_n_by_mul(data_shift_ub, data_quant_ub, data_quant_ub_int32,
                    shift_factor_scalar, shift_factor_rec_scalar)
                self._reduce_loss_for_each_n(data_shift_ub, n_step)
        else:
            # SD3403
            with self.tik_instance.for_range(0, Constant.N_STEP) as n_step:
                shift_factor_scalar = self.tik_instance.Scalar(self.loss_precision, 'shift_factor_scalar')
                shift_factor_scalar.set_as(shift_factor_tensor[n_step])
                # loss_sum: used to store loss
                loss_sum = self.tik_instance.Tensor(self.loss_precision, (self.ub_tensor_size,),
                    tik.scope_ubuf, 'loss_sum')
                self._compute_loss_for_each_n_by_shift(loss_sum, data_quant_ub_int32, shift_factor_scalar, n_step)
                self._reduce_loss_for_each_n(loss_sum, n_step)

    def _compute_loss_for_each_n_by_mul(self, data_shift_ub, data_quant_ub, data_quant_ub_int32,
        shift_factor_scalar, shift_factor_rec_scalar):
        '''used by Ascend310/310P/910, compute loss by mul/round/clip/sub'''
        # right shift-N
        self.tik_instance.vec_muls(self.vector_mask_max, data_shift_ub, data_quant_ub,
            shift_factor_rec_scalar, self.data_repeats, self.blk_per_rep, self.blk_per_rep)
        self._round_fp32_precisely(self.vector_mask_max, data_quant_ub_int32, data_shift_ub,
            self.ub_tensor_size, 'floor')
        # clip by int16
        self.tik_instance.vmin(self.vector_mask_max, data_shift_ub, data_shift_ub, self.int16_max_tensor,
            self.data_repeats, 1, 1, 1, self.blk_per_rep, self.blk_per_rep, 0)
        self.tik_instance.vmax(self.vector_mask_max, data_shift_ub, data_shift_ub, self.int16_min_tensor,
            self.data_repeats, 1, 1, 1, self.blk_per_rep, self.blk_per_rep, 0)
        # left shift-N
        self.tik_instance.vec_muls(self.vector_mask_max, data_shift_ub, data_shift_ub,
            shift_factor_scalar, self.data_repeats, self.blk_per_rep, self.blk_per_rep)
        # calc loss
        self.tik_instance.vec_sub(self.vector_mask_max, data_shift_ub, data_shift_ub,
            data_quant_ub, self.data_repeats, self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
        self.tik_instance.vec_abs(self.vector_mask_max, data_shift_ub, data_shift_ub,
            self.data_repeats, self.blk_per_rep, self.blk_per_rep)

    def _compute_loss_for_each_n_by_shift(self, loss_sum, data_quant_ub, shift_factor_scalar, n_step):
        # 1. loss_mid
        shift_sat_scalar = self.tik_instance.Scalar(self.loss_precision, 'shift_sat_scalar')
        shift_sat_scalar.set_as(shift_factor_scalar - 1)
        shift_sat_ub = self.tik_instance.Tensor(self.loss_precision, (self.int32_vector_mask_max,),
            tik.scope_ubuf, 'shift_sat_ub')
        self.tik_instance.vec_dup(self.int32_vector_mask_max, shift_sat_ub, shift_sat_scalar, 1, self.blk_per_rep)
        shift_sat_ub_uint16 = shift_sat_ub.reinterpret_cast_to("uint16")
        data_quant_ub_uint16 = data_quant_ub.reinterpret_cast_to("uint16")
        # store loss_mid in loss_sum
        loss_mid_uint16 = loss_sum.reinterpret_cast_to("uint16")
        # for loss process, data_num is ub_tensor_size
        # for uint16 process, data_num is 2*ub_tensor_size
        data_num = 2 * self.ub_tensor_size
        loss_mid_repeats = data_num // self.vector_mask_max
        self.tik_instance.vec_and(self.vector_mask_max, loss_mid_uint16, data_quant_ub_uint16, shift_sat_ub_uint16,
            loss_mid_repeats, self.blk_per_rep, self.blk_per_rep, 0)

        # if n is 16(n_step is 15), no data will be saturated to int16-boundary
        # therefore no need to calculate loss_right and loss_left
        with self.tik_instance.if_scope(n_step < 15):
            # 2. loss_right
            scalar_0 = self.tik_instance.Scalar(self.loss_precision, 'scalar_0')
            scalar_0.set_as(0)
            loss_mask_max = self.int32_vector_mask_max
            loss_repeats = self.ub_tensor_size // loss_mask_max

            right_sat_scalar = self.tik_instance.Scalar(self.loss_precision, 'right_sat_scalar')
            right_sat_scalar.set_as(shift_factor_scalar*Constant.INT16_MAX)
            right_sat_ub = self.tik_instance.Tensor(self.loss_precision, (loss_mask_max,),
                tik.scope_ubuf, 'right_sat_ub')
            self.tik_instance.vec_dup(loss_mask_max, right_sat_ub, right_sat_scalar, 1, self.blk_per_rep)

            loss_right = self.tik_instance.Tensor(self.loss_precision, (self.ub_tensor_size,),
                tik.scope_ubuf, 'loss_right')
            with self.tik_instance.new_stmt_scope():
                pos_data = self.tik_instance.Tensor(self.loss_precision, (self.ub_tensor_size,),
                    tik.scope_ubuf, 'pos_data')
                # get the positive data
                self.tik_instance.vmaxs(loss_mask_max, pos_data, data_quant_ub, scalar_0, loss_repeats,
                    1, 1, self.blk_per_rep, self.blk_per_rep)
                # construct sat_data as int16_max * (2**n) + loss_mid, store sat_data in loss_right
                self.tik_instance.vec_add(loss_mask_max, loss_right, loss_sum, right_sat_ub,
                    loss_repeats, self.blk_per_rep, self.blk_per_rep, 0)
                # pos_data - sat_data
                self.tik_instance.vec_sub(loss_mask_max, loss_right, pos_data, loss_right,
                    loss_repeats, self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
                # relu to get the loss_right
                self.tik_instance.vec_relu(loss_mask_max, loss_right, loss_right,
                    loss_repeats, self.blk_per_rep, self.blk_per_rep)

            # 3. loss_left
            left_sat_scalar = self.tik_instance.Scalar(self.loss_precision, 'left_sat_scalar')
            left_sat_scalar.set_as(shift_factor_scalar*Constant.INT16_MIN)
            left_sat_ub = self.tik_instance.Tensor(self.loss_precision, (loss_mask_max,),
                tik.scope_ubuf, 'left_sat_ub')
            self.tik_instance.vec_dup(loss_mask_max, left_sat_ub, left_sat_scalar, 1, self.blk_per_rep)

            loss_left = self.tik_instance.Tensor(self.loss_precision, (self.ub_tensor_size,),
                tik.scope_ubuf, 'loss_left')
            with self.tik_instance.new_stmt_scope():
                neg_data = self.tik_instance.Tensor(self.loss_precision, (self.ub_tensor_size,),
                    tik.scope_ubuf, 'neg_data')
                # get the negative data
                self.tik_instance.vmins(loss_mask_max, neg_data, data_quant_ub, scalar_0, loss_repeats,
                    1, 1, self.blk_per_rep, self.blk_per_rep)
                # construct sat_data as int16_min * (2**n) - loss_mid, store sat_data in loss_left
                self.tik_instance.vec_sub(loss_mask_max, loss_left, left_sat_ub, loss_sum,
                    loss_repeats, self.blk_per_rep, 0, self.blk_per_rep)
                # sat_data - neg_data
                self.tik_instance.vec_sub(loss_mask_max, loss_left, loss_left, neg_data,
                    loss_repeats, self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
                # relu to get the loss_left
                self.tik_instance.vec_relu(loss_mask_max, loss_left, loss_left,
                    loss_repeats, self.blk_per_rep, self.blk_per_rep)

            # 4.sum
            self.tik_instance.vec_add(loss_mask_max, loss_sum, loss_sum, loss_right, loss_repeats,
                self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
            self.tik_instance.vec_add(loss_mask_max, loss_sum, loss_sum, loss_left, loss_repeats,
                self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)

    def _reduce_loss_for_each_n(self, loss_ub, n_step):
        if self.soc_version == 'SD3403':
            # split int32 loss to high-16bit int32 and low-16bit int32
            loss_high = self.tik_instance.Tensor(self.loss_precision, (self.ub_tensor_size,),
                tik.scope_ubuf, 'loss_high')
            loss_low = self.tik_instance.Tensor(self.loss_precision, (self.ub_tensor_size,),
                tik.scope_ubuf, 'loss_low')
            loss_split_rep = self.ub_tensor_size // self.int32_vector_mask_max
            self._split_int32_by_reinterpret(self.int32_vector_mask_max, loss_high, loss_low, loss_ub, loss_split_rep)
            with self.tik_instance.new_stmt_scope():
                self.tmp_loss_high = self.tik_instance.Tensor(self.loss_precision, (self.data_each_block,),
                    tik.scope_ubuf, 'tmp_loss_high')
                self.tik_instance.vec_dup(self.data_each_block, self.tmp_loss_high, 0, 1, self.blk_per_rep)
                self.tmp_loss_low = self.tik_instance.Tensor(self.loss_precision, (self.data_each_block,),
                    tik.scope_ubuf, 'tmp_loss_low')
                self.tik_instance.vec_dup(self.data_each_block, self.tmp_loss_low, 0, 1, self.blk_per_rep)
                self._reduce_add_loss_high_and_low(loss_high, loss_low)
                if self.per_tensor_loss:
                    tmp_loss_scalar1 = self.tik_instance.Scalar(self.loss_precision, 'tmp_loss_scalar1')
                    tmp_loss_scalar2 = self.tik_instance.Scalar(self.loss_precision, 'tmp_loss_scalar2')
                    with self.tik_instance.for_range(1, self.data_each_block) as add_idx:
                        tmp_loss_scalar1.set_as(self.tmp_loss_high[0])
                        tmp_loss_scalar2.set_as(self.tmp_loss_high[add_idx])
                        self.tmp_loss_high[0].set_as(tmp_loss_scalar1+tmp_loss_scalar2)
                        tmp_loss_scalar1.set_as(self.tmp_loss_low[0])
                        tmp_loss_scalar2.set_as(self.tmp_loss_low[add_idx])
                        self.tmp_loss_low[0].set_as(tmp_loss_scalar1+tmp_loss_scalar2)
                    self._add_loss_high_low_to_total_loss(1, n_step)
                else:
                    self._add_loss_high_low_to_total_loss(self.dim_c0, n_step)
        else:
            # divide loss by ub_tensor_size, prevent overflow
            norm_factor = self.tik_instance.Scalar(self.calc_precision, 'norm_factor')
            norm_factor.set_as(1/self.ub_tensor_size)
            self.tik_instance.vec_muls(self.vector_mask_max, loss_ub, loss_ub, norm_factor,
                self.data_repeats, self.blk_per_rep, self.blk_per_rep)
            # reduce loss according to per-tensor or per-channel
            with self.tik_instance.new_stmt_scope():
                # reduce add ub_tensor_size data to tmp_loss_ub
                # vcmpv instruction requests tensor to be 256Bytes-aligned
                self.tmp_loss_ub = self.tik_instance.Tensor(self.calc_precision, (self.vector_mask_max,),
                    tik.scope_ubuf, 'tmp_loss_ub')
                # initialize by 0
                self.tik_instance.vec_dup(self.vector_mask_max, self.tmp_loss_ub, 0, 1, self.blk_per_rep)
                if self.per_tensor_loss:
                    self._reduce_add_to_tmp_loss_per_tensor(loss_ub)
                    # check whether overflow and add tmp_loss to main/remain loss
                    self._add_to_total_loss_per_tensor(n_step)
                else:
                    self._reduce_add_to_tmp_loss_per_channel(loss_ub)
                    self._add_to_total_loss_per_channel(n_step)

    def _reduce_add_to_tmp_loss_per_tensor(self, loss_ub):
        '''used by soc except SD3403'''
        tmp_loss_scalar = self.tik_instance.Scalar(self.calc_precision, 'tmp_loss_scalar')
        tmp_loss_scalar.set_as(0)
        loss_add_repeats = self.cur_valid_data_num // self.vector_mask_max
        loss_add_last_num = self.cur_valid_data_num % self.vector_mask_max
        work_tensor_size = max(loss_add_repeats, 1)
        work_tensor = self.tik_instance.Tensor(self.calc_precision, (work_tensor_size,),
            tik.scope_ubuf, 'work_tensor')
        if loss_add_repeats > 0:
            self.tik_instance.vec_reduce_add(self.vector_mask_max, self.tmp_loss_ub, loss_ub,
                work_tensor, loss_add_repeats, self.blk_per_rep)
            tmp_loss_scalar.set_as(self.tmp_loss_ub[0])
        if loss_add_last_num > 0:
            self.tik_instance.vec_reduce_add(loss_add_last_num, self.tmp_loss_ub,
                loss_ub[loss_add_repeats*self.vector_mask_max], work_tensor, 1, self.blk_per_rep)
            self.tik_instance.vec_adds(1, self.tmp_loss_ub, self.tmp_loss_ub, tmp_loss_scalar, 1,
                self.blk_per_rep, self.blk_per_rep)

    def _reduce_add_to_tmp_loss_per_channel(self, loss_ub):
        '''used by soc except SD3403'''
        if self.soc_version == 'Ascend310':
            loop_time = self.cur_valid_data_num // self.dim_c0
            with self.tik_instance.for_range(0, loop_time) as loop_ind:
                self.tik_instance.vadd(self.dim_c0, self.tmp_loss_ub, loss_ub[loop_ind*self.dim_c0],
                    self.tmp_loss_ub, 1, 1, 1, 1, 0, self.dim_c0//self.data_each_block, 0)
        else:
            loop_time = self.cur_valid_data_num // self.dim_c0 // Constant.MAX_REPEAT_TIME
            last_repeat_time = self.cur_valid_data_num // self.dim_c0 % Constant.MAX_REPEAT_TIME
            if loop_time > 0:
                with self.tik_instance.for_range(0, loop_time) as loop_ind:
                    self.tik_instance.vadd(self.dim_c0, self.tmp_loss_ub,
                        loss_ub[loop_ind*self.dim_c0*Constant.MAX_REPEAT_TIME], self.tmp_loss_ub,
                        Constant.MAX_REPEAT_TIME, 1, 1, 1, 0, self.dim_c0//self.data_each_block, 0)
            if last_repeat_time > 0:
                self.tik_instance.vadd(self.dim_c0, self.tmp_loss_ub,
                    loss_ub[loop_time*self.dim_c0*Constant.MAX_REPEAT_TIME], self.tmp_loss_ub,
                    last_repeat_time, 1, 1, 1, 0, self.dim_c0//self.data_each_block, 0)

    def _reduce_add_loss_high_and_low(self, loss_high, loss_low):
        '''used by both per tensor and per channel of SD3403, loss is int32 type'''
        # for per-channel, c0 is 16; for per-tensor first reduce add to 16 loss, then 16 -> 1
        reduce_add_mask = 16
        loop_time = self.cur_valid_data_num // reduce_add_mask // Constant.MAX_REPEAT_TIME
        last_repeat_time = self.cur_valid_data_num // reduce_add_mask % Constant.MAX_REPEAT_TIME
        last_num = self.cur_valid_data_num % reduce_add_mask
        if loop_time > 0:
            with self.tik_instance.for_range(0, loop_time) as loop_ind:
                self.tik_instance.vadd(reduce_add_mask, self.tmp_loss_high,
                    loss_high[loop_ind*reduce_add_mask*Constant.MAX_REPEAT_TIME], self.tmp_loss_high,
                    Constant.MAX_REPEAT_TIME, 1, 1, 1, 0, reduce_add_mask//self.loss_each_block, 0)
                self.tik_instance.vadd(reduce_add_mask, self.tmp_loss_low,
                    loss_low[loop_ind*reduce_add_mask*Constant.MAX_REPEAT_TIME], self.tmp_loss_low,
                    Constant.MAX_REPEAT_TIME, 1, 1, 1, 0, reduce_add_mask//self.loss_each_block, 0)
        if last_repeat_time > 0:
            offset = loop_time * reduce_add_mask * Constant.MAX_REPEAT_TIME
            self.tik_instance.vadd(reduce_add_mask, self.tmp_loss_high, loss_high[offset],
                self.tmp_loss_high, last_repeat_time, 1, 1, 1, 0, reduce_add_mask//self.loss_each_block, 0)
            self.tik_instance.vadd(reduce_add_mask, self.tmp_loss_low, loss_low[offset],
                self.tmp_loss_low, last_repeat_time, 1, 1, 1, 0, reduce_add_mask//self.loss_each_block, 0)
        if last_num > 0:
            offset = (loop_time * Constant.MAX_REPEAT_TIME + last_repeat_time) * reduce_add_mask
            self.tik_instance.vadd(last_num, self.tmp_loss_high, loss_high[offset], self.tmp_loss_high,
                1, 1, 1, 1, 0, reduce_add_mask//self.loss_each_block, 0)
            self.tik_instance.vadd(last_num, self.tmp_loss_low, loss_low[offset], self.tmp_loss_low,
                1, 1, 1, 1, 0, reduce_add_mask//self.loss_each_block, 0)

    def _get_one_out_of_four_fp16(self, dst, src, data_num, sel_num):
        '''
        used by SD3403
        get aaaa (or bbbb or cccc or dddd) from abcdabcdabcdabcd
        data_num: the size of src
        dst: has half size of src but aligned up to 128 Byte
        sel_num: an int16 number, 0x1111 or 0x2222 or 0x4444 or 0x8888
        '''
        repeats = data_num // self.vector_mask_max
        last_num = data_num % self.vector_mask_max
        offset = repeats * self.vector_mask_max
        with self.tik_instance.new_stmt_scope():
            sel_tensor = self.tik_instance.Tensor('uint16', (self.data_each_block,), tik.scope_ubuf, 'sel_tensor')
            self.tik_instance.vec_dup(self.data_each_block, sel_tensor, sel_num, 1, self.blk_per_rep)
            sel_fp16 = self.tik_instance.Tensor('float16', (data_num,), tik.scope_ubuf, 'sel_fp16')
            # zero tensor
            zeros = self.tik_instance.Tensor('float16', (self.vector_mask_max,), tik.scope_ubuf, 'zeros')
            self.tik_instance.vec_dup(self.vector_mask_max, zeros, 0, 1, self.blk_per_rep)
            if repeats > 0:
                # select all "a", set "bcd" as zero
                self.tik_instance.vec_sel(self.vector_mask_max, 0, sel_fp16, sel_tensor, src, zeros,
                    repeats, self.blk_per_rep, self.blk_per_rep, 0)
            if last_num > 0:
                self.tik_instance.vec_sel(last_num, 0, sel_fp16[offset], sel_tensor, src[offset], zeros,
                    1, self.blk_per_rep, self.blk_per_rep, 0)
            # "a000a000" -> "a0a0"
            self._vcpadd_by_loop(dst, sel_fp16, data_num)
            # "a0a0" -> "aa"
            self._vcpadd_by_loop(dst, dst, data_num//2)

    def _split_int32_by_reinterpret(self, mask_32bit, dst1, dst2, src, repeats_32bit):
        '''split int32 to dst1(high-16bit) and dst2(low-16bit)'''
        data_num = mask_32bit * repeats_32bit
        with self.tik_instance.new_stmt_scope():
            src_u8 = src.reinterpret_cast_to('uint8')
            src_fp16 = self.tik_instance.Tensor('float16', (data_num*4,), tik.scope_ubuf, 'src_fp16')
            # uint8 conv to fp16
            self._int8_conv_to_fp16(src_fp16, src_u8, data_num*4)
            # adjust abcdabcdabcdabcd to aaaa bbbb cccc dddd
            # merge c and d to dst1, a and b to dst2
            # first select all "d"(the highest 8 bit)
            sel_num = 0x8888
            # vec_cpadd requests dst to be 128Byte-aligned
            loss_fp16_size = ceil(data_num * 2 / 64) * 64
            loss_fp16 = self.tik_instance.Tensor('float16', (loss_fp16_size,), tik.scope_ubuf, 'loss_fp16')
            self._get_one_out_of_four_fp16(loss_fp16, src_fp16, data_num*4, sel_num)
            # conv to int32
            self.tik_instance.vec_conv(mask_32bit, 'round', dst1, loss_fp16, repeats_32bit,
                self.blk_per_rep, self.blk_per_rep//2)
            # select all "c"
            sel_num = 0x4444
            self._get_one_out_of_four_fp16(loss_fp16, src_fp16, data_num*4, sel_num)
            # conv to int32
            self.tik_instance.vec_conv(mask_32bit, 'round', dst2, loss_fp16, repeats_32bit,
                self.blk_per_rep, self.blk_per_rep//2)
            # "d" mul 2**8 and add with "c"
            mul_scalar = 2 ** 8
            self.tik_instance.vec_muls(mask_32bit, dst1, dst1, mul_scalar, repeats_32bit,
                self.blk_per_rep, self.blk_per_rep)
            self.tik_instance.vec_add(mask_32bit, dst1, dst1, dst2, repeats_32bit,
                self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
            # select all "b"
            sel_num = 0x2222
            self._get_one_out_of_four_fp16(loss_fp16, src_fp16, data_num*4, sel_num)
            # conv to int32
            self.tik_instance.vec_conv(mask_32bit, 'round', dst2, loss_fp16, repeats_32bit,
                self.blk_per_rep, self.blk_per_rep//2)
            # select all "a"
            sel_num = 0x1111
            self._get_one_out_of_four_fp16(loss_fp16, src_fp16, data_num*4, sel_num)
            tmp_int32 = self.tik_instance.Tensor('int32', (data_num,), tik.scope_ubuf, 'tmp_int32')
            # conv to int32
            self.tik_instance.vec_conv(mask_32bit, 'round', tmp_int32, loss_fp16, repeats_32bit,
                self.blk_per_rep, self.blk_per_rep//2)
            # "b" mul 2**8 and add with "a"
            self.tik_instance.vec_muls(mask_32bit, dst2, dst2, mul_scalar, repeats_32bit,
                self.blk_per_rep, self.blk_per_rep)
            self.tik_instance.vec_add(mask_32bit, dst2, dst2, tmp_int32, repeats_32bit,
                self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)

    def _add_loss_high_low_to_total_loss(self, loss_num, n_step):
        '''
        used by SD3403, add tmp_loss up to total loss
        loss_num: 1 for per_tensor, c0 for per_channel
        '''
        # first split tmp_loss_high and tmp_loss_low
        tmp_loss_1 = self.tik_instance.Tensor(self.loss_precision, (loss_num,), tik.scope_ubuf, 'tmp_loss_1')
        tmp_loss_2 = self.tik_instance.Tensor(self.loss_precision, (loss_num,), tik.scope_ubuf, 'tmp_loss_2')
        self._split_int32_by_reinterpret(loss_num, tmp_loss_1, tmp_loss_2, self.tmp_loss_high, 1)
        tmp_loss_3 = self.tik_instance.Tensor(self.loss_precision, (loss_num,), tik.scope_ubuf, 'tmp_loss_3')
        tmp_loss_4 = self.tik_instance.Tensor(self.loss_precision, (loss_num,), tik.scope_ubuf, 'tmp_loss_4')
        self._split_int32_by_reinterpret(loss_num, tmp_loss_3, tmp_loss_4, self.tmp_loss_low, 1)
        # add up to final_loss
        if self.per_tensor_loss:
            tmp_loss_scalar1 = self.tik_instance.Scalar(self.loss_precision, 'tmp_loss_scalar1')
            tmp_loss_scalar2 = self.tik_instance.Scalar(self.loss_precision, 'tmp_loss_scalar2')
            tmp_loss_scalar1.set_as(self.final_loss_1[n_step])
            tmp_loss_scalar2.set_as(tmp_loss_1[0])
            self.final_loss_1[n_step].set_as(tmp_loss_scalar1+tmp_loss_scalar2)
            tmp_loss_scalar1.set_as(self.final_loss_2[n_step])
            tmp_loss_scalar2.set_as(tmp_loss_2[0])
            self.final_loss_2[n_step].set_as(tmp_loss_scalar1+tmp_loss_scalar2)
            tmp_loss_scalar1.set_as(self.final_loss_3[n_step])
            tmp_loss_scalar2.set_as(tmp_loss_3[0])
            self.final_loss_3[n_step].set_as(tmp_loss_scalar1+tmp_loss_scalar2)
            tmp_loss_scalar1.set_as(self.final_loss_4[n_step])
            tmp_loss_scalar2.set_as(tmp_loss_4[0])
            self.final_loss_4[n_step].set_as(tmp_loss_scalar1+tmp_loss_scalar2)
        else:
            self.tik_instance.vec_add(loss_num, self.final_loss_1[n_step*self.dim_c0],
                self.final_loss_1[n_step*self.dim_c0], tmp_loss_1, 1, 0, 0, 0)
            self.tik_instance.vec_add(loss_num, self.final_loss_2[n_step*self.dim_c0],
                self.final_loss_2[n_step*self.dim_c0], tmp_loss_2, 1, 0, 0, 0)
            self.tik_instance.vec_add(loss_num, self.final_loss_3[n_step*self.dim_c0],
                self.final_loss_3[n_step*self.dim_c0], tmp_loss_3, 1, 0, 0, 0)
            self.tik_instance.vec_add(loss_num, self.final_loss_4[n_step*self.dim_c0],
                self.final_loss_4[n_step*self.dim_c0], tmp_loss_4, 1, 0, 0, 0)

    def _fp32_compare(self, dst, src0, src1, data_num, mode):
        '''compare src0 and src1, support mode "gt" "lt" "eq"'''
        if self.soc_version == 'Ascend310':
            diff = self.tik_instance.Tensor(self.calc_precision, (self.vector_mask_max,), tik.scope_ubuf, 'diff')
            self.tik_instance.vec_sub(data_num, diff, src0, src1, 1,
                self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
            diff_fp16 = self.tik_instance.Tensor('float16', (self.fp16_vector_mask_max,),
                tik.scope_ubuf, 'diff_fp16')
            zeros = self.tik_instance.Tensor('float16', (self.fp16_vector_mask_max,), tik.scope_ubuf, 'zeros')
            self.tik_instance.vec_dup(self.fp16_vector_mask_max, zeros, 0, 1, self.blk_per_rep)
            if mode == 'gt':
                self.tik_instance.vec_dup(self.fp16_vector_mask_max, diff_fp16, -1, 1, self.blk_per_rep)
                self.tik_instance.vec_conv(data_num, '', diff_fp16, diff, 1, self.blk_per_rep//2, self.blk_per_rep)
                self.tik_instance.vcmpv_gt(dst, diff_fp16, zeros, 1, 1, 1, self.blk_per_rep, self.blk_per_rep)
            elif mode == 'lt':
                self.tik_instance.vec_dup(self.fp16_vector_mask_max, diff_fp16, 1, 1, self.blk_per_rep)
                self.tik_instance.vec_conv(data_num, '', diff_fp16, diff, 1, self.blk_per_rep//2, self.blk_per_rep)
                self.tik_instance.vcmpv_lt(dst, diff_fp16, zeros, 1, 1, 1, self.blk_per_rep, self.blk_per_rep)
            elif mode == 'eq':
                self.tik_instance.vec_dup(self.fp16_vector_mask_max, diff_fp16, -1, 1, self.blk_per_rep)
                self.tik_instance.vec_conv(data_num, '', diff_fp16, diff, 1, self.blk_per_rep//2, self.blk_per_rep)
                self.tik_instance.vcmpv_eq(dst, diff_fp16, zeros, 1, 1, 1, self.blk_per_rep, self.blk_per_rep)
        else:
            if mode == 'gt':
                self.tik_instance.vcmpv_gt(dst, src0, src1, 1, 1, 1, self.blk_per_rep, self.blk_per_rep)
            elif mode == 'lt':
                self.tik_instance.vcmpv_lt(dst, src0, src1, 1, 1, 1, self.blk_per_rep, self.blk_per_rep)
            elif mode == 'eq':
                self.tik_instance.vcmpv_eq(dst, src0, src1, 1, 1, 1, self.blk_per_rep, self.blk_per_rep)

    def _add_to_total_loss_per_tensor(self, n_step):
        # define loss_max and loss_max_rec
        self.loss_max_ub = self.tik_instance.Tensor(self.calc_precision, (self.data_each_block,),
            tik.scope_ubuf, 'loss_max_ub')
        self.tik_instance.vec_dup(1, self.loss_max_ub, self.loss_max, 1, self.blk_per_rep)
        self.loss_max_rec_ub = self.tik_instance.Tensor(self.calc_precision, (self.data_each_block,),
            tik.scope_ubuf, 'loss_max_rec_ub')
        self.tik_instance.vec_dup(1, self.loss_max_rec_ub, 1/self.loss_max, 1, self.blk_per_rep)
        # define sub loss, vcmpv instruction requests tensor to be 256Bytes-aligned
        self.sub_loss_ub = self.tik_instance.Tensor(self.calc_precision, (self.vector_mask_max,),
            tik.scope_ubuf, 'sub_loss_ub')
        self.tik_instance.vec_dup(self.vector_mask_max, self.sub_loss_ub, 0, 1, self.blk_per_rep)
        # compare tmp_loss with sub loss
        self.one_main_loss_ub[0].set_as(self.main_loss_ub[n_step])
        self.one_remain_loss_ub[0].set_as(self.remain_loss_ub[n_step])
        self.tik_instance.vec_sub(1, self.sub_loss_ub, self.loss_max_ub, self.one_remain_loss_ub, 1,
            self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
        self.cmp_result = self.tik_instance.Tensor('uint32', (self.data_each_block,), tik.scope_ubuf, 'cmp_result')
        self.tik_instance.vec_dup(self.data_each_block, self.cmp_result, 0, 1, self.blk_per_rep)
        self._fp32_compare(self.cmp_result, self.sub_loss_ub, self.tmp_loss_ub, 1, 'gt')
        self.cmp_result_scalar = self.tik_instance.Scalar(dtype='uint32')
        self.cmp_result_scalar.set_as(self.cmp_result[0])
        # if smaller, directly add to remain loss; else carry in to main loss
        with self.tik_instance.if_scope(self.cmp_result_scalar):
            self.tik_instance.vec_add(1, self.one_remain_loss_ub, self.one_remain_loss_ub, self.tmp_loss_ub, 1,
                self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
        with self.tik_instance.else_scope():
            self.tik_instance.vec_sub(1, self.one_remain_loss_ub, self.tmp_loss_ub, self.sub_loss_ub, 1,
                self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
            self.carry_num = self.tik_instance.Tensor(self.calc_precision, (self.data_each_block,),
                tik.scope_ubuf, 'carry_num')
            self.tik_instance.vec_mul(1, self.carry_num, self.one_remain_loss_ub, self.loss_max_rec_ub,
                1, self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
            carry_num_int32 = self.tik_instance.Tensor('int32', (self.data_each_block,),
                tik.scope_ubuf, 'carry_num_int32')
            self._round_fp32_directly(1, carry_num_int32, self.carry_num, 1, 'floor')
            self.tik_instance.vec_mul(1, self.sub_loss_ub, self.carry_num, self.loss_max_ub, 1,
                self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
            self.tik_instance.vec_sub(1, self.one_remain_loss_ub, self.one_remain_loss_ub, self.sub_loss_ub, 1,
                self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
            self.tik_instance.vec_adds(1, self.carry_num, self.carry_num, 1, 1, self.blk_per_rep, self.blk_per_rep)
            self.tik_instance.vec_add(1, self.one_main_loss_ub, self.one_main_loss_ub, self.carry_num, 1,
                self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
            self.main_loss_ub[n_step].set_as(self.one_main_loss_ub[0])
        self.remain_loss_ub[n_step].set_as(self.one_remain_loss_ub[0])

    def _add_to_total_loss_per_channel(self, n_step):
        # define loss_max and loss_max_rec
        self.loss_max_ub = self.tik_instance.Tensor(self.calc_precision, (self.dim_c0,), tik.scope_ubuf, 'loss_max_ub')
        self.tik_instance.vec_dup(self.dim_c0, self.loss_max_ub, self.loss_max, 1, self.blk_per_rep)
        self.loss_max_rec_ub = self.tik_instance.Tensor(self.calc_precision, (self.dim_c0,),
            tik.scope_ubuf, 'loss_max_rec_ub')
        self.tik_instance.vec_dup(self.dim_c0, self.loss_max_rec_ub, 1/self.loss_max, 1, self.blk_per_rep)
        # define sub loss
        self.sub_loss_ub = self.tik_instance.Tensor(self.calc_precision, (self.vector_mask_max,),
            tik.scope_ubuf, 'sub_loss_ub')
        self.tik_instance.vec_dup(self.vector_mask_max, self.sub_loss_ub, 0, 1, self.blk_per_rep)
        # compare tmp_loss with sub loss
        self.tik_instance.vec_sub(self.dim_c0, self.sub_loss_ub, self.loss_max_ub,
            self.remain_loss_ub[n_step*self.dim_c0], 1, self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
        self.cmp_result = self.tik_instance.Tensor('uint32', (self.data_each_block,), tik.scope_ubuf, 'cmp_result')
        self.tik_instance.vec_dup(self.data_each_block, self.cmp_result, 0, 1, self.blk_per_rep)
        self._fp32_compare(self.cmp_result, self.sub_loss_ub, self.tmp_loss_ub, self.dim_c0, 'gt')
        cmp_mask = self.cmp_result.reinterpret_cast_to('uint64')
        # define vadd mask list as [0, cmp_mask]
        mask_h_scalar = self.tik_instance.Scalar(dtype='uint64')
        mask_h_scalar.set_as(0)
        mask_l_scalar = self.tik_instance.Scalar(dtype='uint64')
        mask_l_scalar.set_as(cmp_mask[0])
        with self.tik_instance.if_scope(mask_l_scalar != 0):
            mask_list = [mask_h_scalar, mask_l_scalar]
            # according to mask, if sub loss is greater than tmp loss, directly add to remain loss
            self.tik_instance.vec_add(mask_list, self.remain_loss_ub[n_step*self.dim_c0],
                self.remain_loss_ub[n_step*self.dim_c0], self.tmp_loss_ub, 1,
                self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
        # invert cmp_result to process the overflow part
        cmp_mask_invert = cmp_mask.reinterpret_cast_to('int16')
        self.tik_instance.vec_not(1, cmp_mask_invert, cmp_mask_invert, 1, self.blk_per_rep, self.blk_per_rep)
        cmp_mask_invert_u64 = cmp_mask_invert.reinterpret_cast_to('uint64')
        mask_l_scalar.set_as(cmp_mask_invert_u64[0])
        # for overflow part, carry in to main loss
        with self.tik_instance.if_scope(mask_l_scalar != 0):
            mask_list = [mask_h_scalar, mask_l_scalar]
            self.tik_instance.vec_sub(mask_list, self.remain_loss_ub[n_step*self.dim_c0], self.tmp_loss_ub,
                self.sub_loss_ub, 1, self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
            self.carry_num = self.tik_instance.Tensor(self.calc_precision, (self.dim_c0,), tik.scope_ubuf, 'carry_num')
            self.tik_instance.vec_mul(mask_list, self.carry_num, self.remain_loss_ub[n_step*self.dim_c0],
                self.loss_max_rec_ub, 1, self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
            carry_num_int32 = self.tik_instance.Tensor('int32', (self.dim_c0,), tik.scope_ubuf, 'carry_num_int32')
            self._round_fp32_directly(self.dim_c0, carry_num_int32, self.carry_num, self.dim_c0, 'floor')
            self.tik_instance.vec_mul(mask_list, self.sub_loss_ub, self.carry_num, self.loss_max_ub, 1,
                self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
            self.tik_instance.vec_sub(mask_list, self.remain_loss_ub[n_step*self.dim_c0],
                self.remain_loss_ub[n_step*self.dim_c0], self.sub_loss_ub, 1,
                self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
            self.tik_instance.vec_adds(mask_list, self.carry_num, self.carry_num, 1, 1,
                self.blk_per_rep, self.blk_per_rep)
            self.tik_instance.vec_add(mask_list,
                self.main_loss_ub[n_step*self.dim_c0], self.main_loss_ub[n_step*self.dim_c0],
                self.carry_num, 1, self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)

    def _move_per_tensor_loss_out(self, core_ind):
        burst_len = Constant.N_STEP // self.loss_each_block
        if self.soc_version == 'SD3403':
            self.tik_instance.tensor_mov(
                self.final_loss1_workspace[core_ind*Constant.N_STEP], self.final_loss_1, '', 1, burst_len, 0, 0)
            self.tik_instance.tensor_mov(
                self.final_loss2_workspace[core_ind*Constant.N_STEP], self.final_loss_2, '', 1, burst_len, 0, 0)
            self.tik_instance.tensor_mov(
                self.final_loss3_workspace[core_ind*Constant.N_STEP], self.final_loss_3, '', 1, burst_len, 0, 0)
            self.tik_instance.tensor_mov(
                self.final_loss4_workspace[core_ind*Constant.N_STEP], self.final_loss_4, '', 1, burst_len, 0, 0)
        else:
            self.tik_instance.tensor_mov(
                self.main_loss_workspace[core_ind*Constant.N_STEP], self.main_loss_ub, '', 1, burst_len, 0, 0)
            self.tik_instance.tensor_mov(
                self.remain_loss_workspace[core_ind*Constant.N_STEP], self.remain_loss_ub, '', 1, burst_len, 0, 0)

    def _move_per_channel_loss_out(self, core_ind, c_offset):
        dst_stride = (self.c_len - self.dim_c0) // self.loss_each_block
        if dst_stride <= Constant.MOVE_STRIDE_MAX:
            if self.soc_version == 'SD3403':
                self.tik_instance.data_move(self.final_loss1_workspace[core_ind*self.total_loss_size+c_offset],
                    self.final_loss_1, 0, Constant.N_STEP, self.dim_c0//self.loss_each_block, 0, dst_stride)
                self.tik_instance.data_move(self.final_loss2_workspace[core_ind*self.total_loss_size+c_offset],
                    self.final_loss_2, 0, Constant.N_STEP, self.dim_c0//self.loss_each_block, 0, dst_stride)
                self.tik_instance.data_move(self.final_loss3_workspace[core_ind*self.total_loss_size+c_offset],
                    self.final_loss_3, 0, Constant.N_STEP, self.dim_c0//self.loss_each_block, 0, dst_stride)
                self.tik_instance.data_move(self.final_loss4_workspace[core_ind*self.total_loss_size+c_offset],
                    self.final_loss_4, 0, Constant.N_STEP, self.dim_c0//self.loss_each_block, 0, dst_stride)
            else:
                self.tik_instance.data_move(self.main_loss_workspace[core_ind*self.total_loss_size+c_offset],
                    self.main_loss_ub, 0, Constant.N_STEP, self.dim_c0//self.loss_each_block, 0, dst_stride)
                self.tik_instance.data_move(self.remain_loss_workspace[core_ind*self.total_loss_size+c_offset],
                    self.remain_loss_ub, 0, Constant.N_STEP, self.dim_c0//self.loss_each_block, 0, dst_stride)
        else:
            with self.tik_instance.for_range(0, Constant.N_STEP) as move_index:
                offset = core_ind * self.total_loss_size + move_index * self.c_len + c_offset
                if self.soc_version == 'SD3403':
                    self.tik_instance.data_move(self.final_loss1_workspace[offset],
                        self.final_loss_1[move_index*self.dim_c0], 0, 1, self.dim_c0//self.loss_each_block, 0, 0)
                    self.tik_instance.data_move(self.final_loss2_workspace[offset],
                        self.final_loss_2[move_index*self.dim_c0], 0, 1, self.dim_c0//self.loss_each_block, 0, 0)
                    self.tik_instance.data_move(self.final_loss3_workspace[offset],
                        self.final_loss_3[move_index*self.dim_c0], 0, 1, self.dim_c0//self.loss_each_block, 0, 0)
                    self.tik_instance.data_move(self.final_loss4_workspace[offset],
                        self.final_loss_4[move_index*self.dim_c0], 0, 1, self.dim_c0//self.loss_each_block, 0, 0)
                else:
                    self.tik_instance.data_move(self.main_loss_workspace[offset],
                        self.main_loss_ub[move_index*self.dim_c0], 0, 1, self.dim_c0//self.loss_each_block, 0, 0)
                    self.tik_instance.data_move(self.remain_loss_workspace[offset],
                        self.remain_loss_ub[move_index*self.dim_c0], 0, 1, self.dim_c0//self.loss_each_block, 0, 0)

    def _gather_total_loss_per_tensor(self):
        # define total loss and init by 0
        if self.soc_version == 'SD3403':
            # total_loss: 16 * 4, stores final_loss1, final_loss2, final_loss3, final_loss4 continuously
            self.total_loss = self.tik_instance.Tensor(self.loss_precision, (Constant.N_STEP*3,),
                tik.scope_ubuf, 'total_loss')
            self.tik_instance.vec_dup(Constant.N_STEP*3, self.total_loss, 0, 1, self.blk_per_rep)
            burst_len = Constant.N_STEP // self.loss_each_block
            with self.tik_instance.new_stmt_scope():
                # define a tmp loss to receive loss from gm to ub
                loss = self.tik_instance.Tensor(self.loss_precision, (Constant.N_STEP,), tik.scope_ubuf, 'loss')
                # define loss1 to loss4 on ub
                self.loss1 = self.tik_instance.Tensor(self.loss_precision, (Constant.N_STEP,), tik.scope_ubuf, 'loss1')
                self.loss2 = self.tik_instance.Tensor(self.loss_precision, (Constant.N_STEP,), tik.scope_ubuf, 'loss2')
                self.loss3 = self.tik_instance.Tensor(self.loss_precision, (Constant.N_STEP,), tik.scope_ubuf, 'loss3')
                self.loss4 = self.tik_instance.Tensor(self.loss_precision, (Constant.N_STEP,), tik.scope_ubuf, 'loss4')
                self.tik_instance.vec_dup(Constant.N_STEP, self.loss1, 0, 1, self.blk_per_rep)
                self.tik_instance.vec_dup(Constant.N_STEP, self.loss2, 0, 1, self.blk_per_rep)
                self.tik_instance.vec_dup(Constant.N_STEP, self.loss3, 0, 1, self.blk_per_rep)
                self.tik_instance.vec_dup(Constant.N_STEP, self.loss4, 0, 1, self.blk_per_rep)
                for i in range(0, self.aicore_num):
                    self.tik_instance.data_move(loss, self.final_loss1_workspace[Constant.N_STEP*i],
                        0, 1, burst_len, 0, 0)
                    self.tik_instance.vec_add(Constant.N_STEP, self.loss1, loss, self.loss1,
                        1, self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
                    self.tik_instance.data_move(loss, self.final_loss2_workspace[Constant.N_STEP*i],
                        0, 1, burst_len, 0, 0)
                    self.tik_instance.vec_add(Constant.N_STEP, self.loss2, loss, self.loss2,
                        1, self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
                    self.tik_instance.data_move(loss, self.final_loss3_workspace[Constant.N_STEP*i],
                        0, 1, burst_len, 0, 0)
                    self.tik_instance.vec_add(Constant.N_STEP, self.loss3, loss, self.loss3,
                        1, self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
                    self.tik_instance.data_move(loss, self.final_loss4_workspace[Constant.N_STEP*i],
                        0, 1, burst_len, 0, 0)
                    self.tik_instance.vec_add(Constant.N_STEP, self.loss4, loss, self.loss4,
                        1, self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
                self._adjust_loss_for_rank_cmp(Constant.N_STEP, 1)
        else:
            self.total_main_loss = self.tik_instance.Tensor(self.calc_precision, (self.vector_mask_max,),
                tik.scope_ubuf, 'total_loss')
            self.tik_instance.vec_dup(self.vector_mask_max, self.total_main_loss, 0, 1, self.blk_per_rep)
            self.total_remain_loss = self.tik_instance.Tensor(self.calc_precision, (self.vector_mask_max,),
                tik.scope_ubuf, 'total_remain_loss')
            self.tik_instance.vec_dup(self.vector_mask_max, self.total_remain_loss, 0, 1, self.blk_per_rep)
            burst_len = self.total_loss_size // self.data_each_block
            with self.tik_instance.new_stmt_scope():
                # define a tmp loss to receive loss from gm to ub
                loss = self.tik_instance.Tensor(self.calc_precision, self.total_loss_shape, tik.scope_ubuf, 'loss')
                remain_loss = self.tik_instance.Tensor(self.calc_precision, self.total_loss_shape,
                    tik.scope_ubuf, 'remain_loss')
                for i in range(0, self.aicore_num):
                    self.tik_instance.data_move(loss, self.main_loss_workspace[self.total_loss_size * i],
                        0, 1, burst_len, 0, 0)
                    self.tik_instance.data_move(remain_loss, self.remain_loss_workspace[self.total_loss_size * i],
                        0, 1, burst_len, 0, 0)
                    # total_loss_size is 16, finish vadd by one repeat
                    self.tik_instance.vec_add(self.total_loss_size, self.total_main_loss, loss, self.total_main_loss,
                        1, self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
                    self.tik_instance.vec_add(self.total_loss_size, self.total_remain_loss, remain_loss,
                        self.total_remain_loss, 1, self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)

    def _add_by_repeat(self, dst, src, data_num, mask_num, offset=0):
        repeat_time = data_num // mask_num
        last_num = data_num % mask_num
        if repeat_time > 0:
            self.tik_instance.vec_add(mask_num, dst[offset], dst[offset], src, repeat_time,
                self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
        if last_num > 0:
            offset += repeat_time * mask_num
            self.tik_instance.vec_add(last_num, dst[offset], dst[offset], src, 1,
                self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)

    def _gather_total_loss_per_channel(self, c_offset, c_num):
        if self.soc_version == 'SD3403':
            all_loss_size = 3 * Constant.N_STEP * c_num
            # define total loss(16 * c_num * 3) and init by 0
            self.total_loss = self.tik_instance.Tensor(self.loss_precision, (3, Constant.N_STEP, c_num),
                tik.scope_ubuf, 'total_loss')
            self._dup_by_repeat(self.total_loss, 0, all_loss_size, self.int32_vector_mask_max)
            with self.tik_instance.new_stmt_scope():
                # define a tmp loss to receive loss from gm to ub
                loss = self.tik_instance.Tensor(self.loss_precision, (Constant.N_STEP, c_num), tik.scope_ubuf, 'loss')
                # define loss1 to loss4 on ub
                self.loss1 = self.tik_instance.Tensor(self.loss_precision, (Constant.N_STEP, c_num),
                    tik.scope_ubuf, 'loss1')
                self.loss2 = self.tik_instance.Tensor(self.loss_precision, (Constant.N_STEP, c_num),
                    tik.scope_ubuf, 'loss2')
                self.loss3 = self.tik_instance.Tensor(self.loss_precision, (Constant.N_STEP, c_num),
                    tik.scope_ubuf, 'loss3')
                self.loss4 = self.tik_instance.Tensor(self.loss_precision, (Constant.N_STEP, c_num),
                    tik.scope_ubuf, 'loss4')
                self._dup_by_repeat(self.loss1, 0, Constant.N_STEP*c_num, self.int32_vector_mask_max)
                self._dup_by_repeat(self.loss2, 0, Constant.N_STEP*c_num, self.int32_vector_mask_max)
                self._dup_by_repeat(self.loss3, 0, Constant.N_STEP*c_num, self.int32_vector_mask_max)
                self._dup_by_repeat(self.loss4, 0, Constant.N_STEP*c_num, self.int32_vector_mask_max)
                src_stride = (self.c_len - c_num) // self.loss_each_block
                burst_len = c_num // self.loss_each_block
                for i in range(0, self.aicore_num):
                    gm_offset = self.total_loss_size * i + c_offset
                    self.tik_instance.data_move(loss, self.final_loss1_workspace[gm_offset], 0,
                        Constant.N_STEP, burst_len, src_stride, 0)
                    self._add_by_repeat(self.loss1, loss, Constant.N_STEP*c_num, self.int32_vector_mask_max, 0)
                    self.tik_instance.data_move(loss, self.final_loss2_workspace[gm_offset], 0,
                        Constant.N_STEP, burst_len, src_stride, 0)
                    self._add_by_repeat(self.loss2, loss, Constant.N_STEP*c_num, self.int32_vector_mask_max, 0)
                    self.tik_instance.data_move(loss, self.final_loss3_workspace[gm_offset], 0,
                        Constant.N_STEP, burst_len, src_stride, 0)
                    self._add_by_repeat(self.loss3, loss, Constant.N_STEP*c_num, self.int32_vector_mask_max, 0)
                    self.tik_instance.data_move(loss, self.final_loss4_workspace[gm_offset], 0,
                        Constant.N_STEP, burst_len, src_stride, 0)
                    self._add_by_repeat(self.loss4, loss, Constant.N_STEP*c_num, self.int32_vector_mask_max, 0)
                repeats = Constant.N_STEP * c_num // self.int32_vector_mask_max
                self._adjust_loss_for_rank_cmp(self.int32_vector_mask_max, repeats)
        else:
            cur_loss_shape = (Constant.N_STEP, c_num)
            self.cur_loss_size = Constant.N_STEP * c_num
            # define total loss and init by 0
            self.total_main_loss = self.tik_instance.Tensor(self.calc_precision,
                cur_loss_shape, tik.scope_ubuf, 'total_main_loss')
            self._dup_by_repeat(self.total_main_loss, 0, self.cur_loss_size, self.vector_mask_max)
            self.total_remain_loss = self.tik_instance.Tensor(self.calc_precision, cur_loss_shape,
                tik.scope_ubuf, 'total_remain_loss')
            self._dup_by_repeat(self.total_remain_loss, 0, self.cur_loss_size, self.vector_mask_max)
            with self.tik_instance.new_stmt_scope():
                # define a tmp loss to receive loss from gm to ub
                loss = self.tik_instance.Tensor(self.calc_precision, cur_loss_shape, tik.scope_ubuf, 'loss')
                remain_loss = self.tik_instance.Tensor(self.calc_precision, cur_loss_shape,
                    tik.scope_ubuf, 'remain_loss')
                src_stride = (self.c_len - c_num) // self.data_each_block
                burst_len = c_num // self.data_each_block
                for i in range(0, self.aicore_num):
                    gm_offset = self.total_loss_size * i + c_offset
                    self.tik_instance.data_move(loss, self.main_loss_workspace[gm_offset],
                        0, Constant.N_STEP, burst_len, src_stride, 0)
                    self.tik_instance.data_move(remain_loss, self.remain_loss_workspace[gm_offset],
                        0, Constant.N_STEP, burst_len, src_stride, 0)
                    self._add_by_repeat(self.total_main_loss, loss, self.cur_loss_size, self.vector_mask_max)
                    self._add_by_repeat(self.total_remain_loss, remain_loss, self.cur_loss_size, self.vector_mask_max)

    def _reduce_and_output(self):
        if self.per_tensor_loss:
            # move loss from gm to ub
            self._gather_total_loss_per_tensor()
            self._reduce_and_output_per_tensor()
        else:
            # output_ub_tensor_size is 16-aligned
            loop_time = self.c_len // self.output_ub_tensor_size
            last_c_num = self.c_len % self.output_ub_tensor_size
            if loop_time > 0:
                with self.tik_instance.new_stmt_scope():
                    with self.tik_instance.for_range(0, loop_time) as loop_index:
                        c_offset = loop_index * self.output_ub_tensor_size
                        self._gather_total_loss_per_channel(c_offset, self.output_ub_tensor_size)
                        self._reduce_and_output_per_channel(c_offset, self.output_ub_tensor_size)
            if last_c_num > 0:
                c_offset = loop_time * self.output_ub_tensor_size
                self._gather_total_loss_per_channel(c_offset, last_c_num)
                self._reduce_and_output_per_channel(c_offset, last_c_num)

    def _reduce_and_output_per_tensor(self):
        if self.soc_version == 'SD3403':
            optimal_n_tensor_s8 = self._find_optimal_n_by_scalar_cmp()
        else:
            optimal_n_tensor_s8 = self._find_optimal_n_by_vcmp()
        if self.ori_scale_w_len == 1:
            # move one optimal n out to gm
            self.tik_instance.tensor_mov(self.output_n, optimal_n_tensor_s8, '', 1, 1, 0, 0)
        else:
            # duplicate optimal n to ori_scale_w_len and move out to gm
            self._duplicate_n_and_output(optimal_n_tensor_s8)

    def _adjust_loss_for_rank_cmp(self, mask, repeats):
        '''
        used by SD3403
        initially the loss is loss1 * 2**32 + loss2 * 2**16 + loss3 * 2**16 + loss4
        change it to loss_highest * 2**32 + loss_mid * 2**16 + loss_lowest
        and restrict loss_mid and loss_lowest to be smaller than 2**16
        continuously store loss_highest, loss_mid and loss_lowest in total_loss
        '''
        # split loss2 to loss_a * 2**16 + loss_b
        data_num = mask * repeats
        loss_a = self.tik_instance.Tensor(self.loss_precision, (data_num,), tik.scope_ubuf, 'loss_a')
        loss_b = self.loss2
        # split loss2 to loss_a * 2**16 + loss_b
        self._split_int32_by_reinterpret(mask, loss_a, loss_b, self.loss2, repeats)
        # add loss_a up to total_loss highest loss
        self.tik_instance.vec_add(mask, self.total_loss, loss_a, self.loss1, repeats,
            self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)

        loss_c = self.loss3
        # split loss3 to loss_a * 2**16 + loss_c
        self._split_int32_by_reinterpret(mask, loss_a, loss_c, self.loss3, repeats)
        # add loss_a up to total_loss highest loss
        self.tik_instance.vec_add(mask, self.total_loss, loss_a, self.total_loss, repeats,
            self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
        # add loss_b and loss_c to loss_d(mid loss)
        # loss1 is not useful, the space can be reused
        loss_d = self.loss1
        self.tik_instance.vec_add(mask, loss_d, loss_b, loss_c, repeats,
            self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)

        # split loss4 to loss_a * 2**16 + loss_b
        self._split_int32_by_reinterpret(mask, loss_a, loss_b, self.loss4, repeats)
        # add loss_a up to loss_d(mid loss)
        self.tik_instance.vec_add(mask, loss_d, loss_a, loss_d, repeats,
            self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
        # add loss_b up to total_loss lowest loss
        self.tik_instance.vec_add(mask, self.total_loss[data_num*2], loss_b, self.total_loss[data_num*2], repeats,
            self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)

        # split loss_d(mid loss) to loss_a * 2**16 + loss_b
        self._split_int32_by_reinterpret(mask, loss_a, loss_b, loss_d, repeats)
        # add loss_a up to total_loss highest loss
        self.tik_instance.vec_add(mask, self.total_loss, loss_a, self.total_loss, repeats,
            self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
        # add loss_b up to total_loss mid loss
        self.tik_instance.vec_add(mask, self.total_loss[data_num], loss_b, self.total_loss[data_num], repeats,
            self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)

    def _find_optimal_n_by_scalar_cmp(self):
        '''used by SD3403'''
        # only 16 loss * 4 groups int32 loss, compare scalar by pair
        min_loss_scalar = self.tik_instance.Scalar(self.loss_precision, 'min_loss_scalar')
        cur_loss_scalar = self.tik_instance.Scalar(self.loss_precision, 'cur_loss_scalar')
        # optimal_nums: record how many n has the smallest loss
        optimal_nums = self.tik_instance.Scalar(self.loss_precision, 'optimal_nums')
        # define the result, the first element is the optimal n
        optimal_n_tensor_s8 = self.tik_instance.Tensor('int8', (Constant.BLOCK_BYTE_SIZE,),
            tik.scope_ubuf, 'optimal_n_tensor_s8')
        zero_scalar = self.tik_instance.Scalar('int8', 'zero_scalar')
        zero_scalar.set_as(0)
        optimal_n_tensor_s8[0].set_as(zero_scalar)
        # 16 * 3, record whether each n needs to be compared, 1 means need, 0 means not
        optimal_n_record = self.tik_instance.Tensor(self.loss_precision, (Constant.N_STEP*3,),
            tik.scope_ubuf, 'optimal_n_record')
        # set the records in the first group to be all 1
        self.tik_instance.vec_dup(Constant.N_STEP, optimal_n_record, 1, 1, self.blk_per_rep)
        # set the records in the last 2 group to be all 0
        self.tik_instance.vec_dup(Constant.N_STEP*2, optimal_n_record[Constant.N_STEP], 0, 1, self.blk_per_rep)
        cur_record = self.tik_instance.Scalar(self.loss_precision, 'cur_record')
        one_scalar = self.tik_instance.Scalar(self.loss_precision, 'one_scalar')
        one_scalar.set_as(1)
        with self.tik_instance.for_range(0, 3) as group_ind:
            min_loss_scalar.set_as(Constant.INT32_MAX)
            optimal_nums.set_as(0)
            with self.tik_instance.for_range(0, Constant.N_STEP) as cmp_ind:
                cur_record.set_as(optimal_n_record[group_ind*Constant.N_STEP+cmp_ind])
                # cur_record is 1 means current loss needs to be compared
                with self.tik_instance.if_scope(cur_record == 1):
                    cur_loss_scalar.set_as(self.total_loss[group_ind*Constant.N_STEP+cmp_ind])
                    with self.tik_instance.if_scope(cur_loss_scalar < min_loss_scalar):
                        min_loss_scalar.set_as(cur_loss_scalar)
                        optimal_n_tensor_s8[0].set_as(cmp_ind+1)
                        optimal_nums.set_as(1)
                        # if is not the last group, record for the next group
                        with self.tik_instance.if_scope(group_ind != 2):
                            self.tik_instance.vec_dup(Constant.N_STEP, optimal_n_record[Constant.N_STEP*(group_ind+1)],
                                0, 1, self.blk_per_rep)
                            optimal_n_record[Constant.N_STEP*(group_ind+1)+cmp_ind].set_as(one_scalar)
                    with self.tik_instance.elif_scope(cur_loss_scalar == min_loss_scalar):
                        optimal_nums.set_as(optimal_nums+1)
                        with self.tik_instance.if_scope(group_ind != 2):
                            optimal_n_record[Constant.N_STEP*(group_ind+1)+cmp_ind].set_as(one_scalar)
        return optimal_n_tensor_s8

    def _find_optimal_n_by_vcmp(self):
        '''used by soc except SD3403'''
        # only 16 loss * 2 groups fp32 loss, compare by pair
        # define optimal loss and init as [MAX, -1, -1, ..., -1]
        optimal_main_loss = self.tik_instance.Tensor(self.calc_precision, (self.vector_mask_max,),
            tik.scope_ubuf, 'optimal_main_loss')
        self.tik_instance.vec_dup(self.vector_mask_max, optimal_main_loss, -1, 1, self.blk_per_rep)
        self.tik_instance.vec_dup(1, optimal_main_loss, Constant.FP32_LOSS_MAX, 1, self.blk_per_rep)
        # record whether each n has the optimal main loss or not, 1 means optimal, 0 means not
        optimal_n_record = self.tik_instance.Tensor('uint32', (Constant.N_STEP,),
            tik.scope_ubuf, 'optimal_n_record')
        # init optimal_n_record by 0
        self.tik_instance.vec_dup(Constant.N_STEP, optimal_n_record, 0, 1, self.blk_per_rep)
        cmp_result = self.tik_instance.Tensor('uint32', (self.data_each_block,), tik.scope_ubuf, 'cmp_result')
        cmp_result_scalar = self.tik_instance.Scalar('uint32')
        with self.tik_instance.for_range(0, Constant.N_STEP) as cmp_index:
            self.tik_instance.vec_dup(self.data_each_block, cmp_result, 0, 1, self.blk_per_rep)
            self.total_main_loss[0].set_as(self.total_main_loss[cmp_index])
            # if loss of current n is less than the optimal
            self._fp32_compare(cmp_result, self.total_main_loss, optimal_main_loss, 1, 'lt')
            cmp_result_scalar.set_as(cmp_result[0])
            with self.tik_instance.if_scope(cmp_result_scalar):
                optimal_main_loss[0].set_as(self.total_main_loss[0])
                # set the whole record to be 0, then set the current n to be the only optimal n
                self.tik_instance.vec_dup(Constant.N_STEP, optimal_n_record, 0, 1, self.blk_per_rep)
                optimal_n_record[cmp_index].set_as(1)
            # if loss of current n is equal to the optimal
            self._fp32_compare(cmp_result, self.total_main_loss, optimal_main_loss, 1, 'eq')
            cmp_result_scalar.set_as(cmp_result[0])
            with self.tik_instance.if_scope(cmp_result_scalar):
                optimal_main_loss[0].set_as(self.total_main_loss[0])
                # set the current n to be one of the optimal n
                optimal_n_record[cmp_index].set_as(1)
        # compare remain loss
        # define optimal remain loss and init as [MAX, -1, -1, ..., -1]
        optimal_remain_loss = self.tik_instance.Tensor(self.calc_precision, (self.vector_mask_max,),
            tik.scope_ubuf, 'optimal_remain_loss')
        self.tik_instance.vec_dup(self.vector_mask_max, optimal_remain_loss, -1, 1, self.blk_per_rep)
        self.tik_instance.vec_dup(1, optimal_remain_loss, Constant.FP32_LOSS_MAX, 1, self.blk_per_rep)
        optimal_n_scalar = self.tik_instance.Scalar('uint32')
        # define optimal_n_tensor_s8, the first element is the optimal n
        optimal_n_tensor_s8 = self.tik_instance.Tensor('int8', (Constant.BLOCK_BYTE_SIZE,),
            tik.scope_ubuf, 'optimal_n_tensor_s8')
        with self.tik_instance.for_range(0, Constant.N_STEP) as cmp_index:
            optimal_n_scalar.set_as(optimal_n_record[cmp_index])
            # if current n has the optimal main loss
            with self.tik_instance.if_scope(optimal_n_scalar):
                self.tik_instance.vec_dup(self.data_each_block, cmp_result, 0, 1, self.blk_per_rep)
                self.total_remain_loss[0].set_as(self.total_remain_loss[cmp_index])
                self._fp32_compare(cmp_result, self.total_remain_loss, optimal_remain_loss, 1, 'lt')
                cmp_result_scalar.set_as(cmp_result[0])
                with self.tik_instance.if_scope(cmp_result_scalar):
                    optimal_remain_loss[0].set_as(self.total_remain_loss[0])
                    optimal_n_tensor_s8[0].set_as(cmp_index+1)
        return optimal_n_tensor_s8

    def _duplicate_n_and_output(self, optimal_n_tensor_s8):
        # vec_dup do not support int8, first vconv to fp16
        optimal_n_tensor_fp16 = self.tik_instance.Tensor('float16', (self.data_each_block*2,),
            tik.scope_ubuf, 'optimal_n_tensor_fp16')
        self.tik_instance.vec_conv(1, '', optimal_n_tensor_fp16, optimal_n_tensor_s8, 1,
            self.blk_per_rep, self.blk_per_rep)
        optimal_n_value_fp16 = self.tik_instance.Scalar('float16', 'optimal_n_value_fp16')
        optimal_n_value_fp16.set_as(optimal_n_tensor_fp16[0])
        # duplicate
        move_time = self.ori_scale_w_len // self.output_ub_tensor_size
        last_move_num = self.ori_scale_w_len % self.output_ub_tensor_size
        optimal_n_tensor_dup = self.tik_instance.Tensor('float16', (self.fp16_vector_mask_max,),
            tik.scope_ubuf, 'optimal_n_tensor_dup')
        self.tik_instance.vec_dup(self.fp16_vector_mask_max, optimal_n_tensor_dup,
            optimal_n_value_fp16, 1, self.blk_per_rep)
        optimal_n_tensor_dup_s8 = self.tik_instance.Tensor('int8', (self.output_ub_tensor_size,),
            tik.scope_ubuf, 'optimal_n_tensor_dup_s8')
        if move_time > 0:
            # output_ub_tensor_size is 128-aligned
            repeats = self.output_ub_tensor_size // self.fp16_vector_mask_max
            # vconv to int8
            self.tik_instance.vec_conv(self.fp16_vector_mask_max, '', optimal_n_tensor_dup_s8,
                optimal_n_tensor_dup, repeats, self.blk_per_rep//2, 0)
            burst_len = self.output_ub_tensor_size // Constant.BLOCK_BYTE_SIZE
            with self.tik_instance.for_range(0, move_time) as move_index:
                self.tik_instance.tensor_mov(self.output_n[move_index*self.output_ub_tensor_size],
                    optimal_n_tensor_dup_s8, '', 1, burst_len, 0, 0)
        if last_move_num > 0:
            repeats = last_move_num // self.fp16_vector_mask_max
            last_num = last_move_num % self.fp16_vector_mask_max
            if repeats > 0:
                self.tik_instance.vec_conv(self.fp16_vector_mask_max, '', optimal_n_tensor_dup_s8,
                    optimal_n_tensor_dup, repeats, self.blk_per_rep//2, 0)
            if last_num > 0:
                self.tik_instance.vec_conv(last_num, '', optimal_n_tensor_dup_s8[repeats*self.fp16_vector_mask_max],
                    optimal_n_tensor_dup, 1, self.blk_per_rep//2, 0)
            burst_len = ceil(last_move_num / Constant.BLOCK_BYTE_SIZE)
            self.tik_instance.tensor_mov(self.output_n[move_time*self.output_ub_tensor_size],
                optimal_n_tensor_dup_s8, '', 1, burst_len, 0, 0)

    def _define_work_tensor_for_reduce(self, repeat_time, data_each_block, data_each_repeat):
        it1_output_count = 2 * repeat_time
        it2_align_start = ceil(it1_output_count / data_each_block) * data_each_block
        it2_output_count = ceil(it1_output_count / data_each_repeat) * 2
        it3_align_start = ceil(it2_output_count / data_each_block) * data_each_block
        it3_output_count = ceil(it2_output_count / data_each_repeat) * 2
        it4_align_start = ceil(it3_output_count / data_each_block) * data_each_block
        it4_output_count = ceil(it3_output_count / data_each_repeat) * 2
        final_work_tensor_need_size = it2_align_start + it3_align_start + it4_align_start + it4_output_count
        work_tensor_ub = self.tik_instance.Tensor(
            'float16', (final_work_tensor_need_size,), tik.scope_ubuf, 'work_tensor_ub')
        return work_tensor_ub

    def _preprocess_loss_to_three_fp16(self, loss, c_num, quotient1_trans, quotient2_trans, remainder2_trans):
        with self.tik_instance.new_stmt_scope():
            quotient1 = self.tik_instance.Tensor(self.calc_precision, (c_num*Constant.N_STEP,),
                tik.scope_ubuf, 'quotient1')
            quotient2 = self.tik_instance.Tensor(self.calc_precision, (c_num*Constant.N_STEP,),
                tik.scope_ubuf, 'quotient2')
            remainder2 = self.tik_instance.Tensor(self.calc_precision, (c_num*Constant.N_STEP,),
                tik.scope_ubuf, 'remainder2')
            # split total_main_loss
            self._split_fp32_to_fp16(loss, self.cur_loss_size, quotient1, quotient2, remainder2)
            quotient1_fp16 = self.tik_instance.Tensor('float16', (c_num*Constant.N_STEP,),
                tik.scope_ubuf, 'quotient1_fp16')
            quotient2_fp16 = self.tik_instance.Tensor('float16', (c_num*Constant.N_STEP,),
                tik.scope_ubuf, 'quotient2_fp16')
            remainder2_fp16 = self.tik_instance.Tensor('float16', (c_num*Constant.N_STEP,),
                tik.scope_ubuf, 'remainder2_fp16')
            self._conv_fp32_to_fp16(quotient1_fp16, quotient1, self.cur_loss_size, self.vector_mask_max)
            self._conv_fp32_to_fp16(quotient2_fp16, quotient2, self.cur_loss_size, self.vector_mask_max)
            self._conv_fp32_to_fp16(remainder2_fp16, remainder2, self.cur_loss_size, self.vector_mask_max)
            self._transpose_loss_fp16(quotient1_trans, quotient1_fp16, c_num)
            self._transpose_loss_fp16(quotient2_trans, quotient2_fp16, c_num)
            self._transpose_loss_fp16(remainder2_trans, remainder2_fp16, c_num)

    def _reduce_and_output_per_channel(self, c_offset, c_num):
        with self.tik_instance.new_stmt_scope():
            self.fp16_max_tensor = self.tik_instance.Tensor('float16', (Constant.N_STEP,),
                tik.scope_ubuf, 'fp16_max_tensor')
            self.tik_instance.vec_dup(Constant.N_STEP, self.fp16_max_tensor, Constant.FP16_MAX, 1, self.blk_per_rep)
            final_loss_fp16 = self.tik_instance.Tensor('float16', (c_num*Constant.N_STEP,),
                tik.scope_ubuf, 'final_loss_fp16')
            if self.soc_version == 'SD3403':
                self._find_min_loss_by_int32_split(c_num, final_loss_fp16)
            else:
                self._find_min_loss_by_fp32_split(c_num, final_loss_fp16)
            # vcmin for final_loss_fp16
            min_result = self._vcmin_n_steps_by_repeats(final_loss_fp16, c_num)
            min_index = min_result.reinterpret_cast_to('uint16')
            # vand 11...100...0, get the min_index at the high 16 bit
            and_num = (2 ** 16 - 1) * (2 ** 16)
            self._data_vand_num(min_index, min_index, and_num, c_num*2)
            # u16 can not conv to fp16
            # so first reinterpret to int8, then vconv to fp16 and vcpadd
            min_index_s8 = min_index.reinterpret_cast_to('int8')
            min_index_fp16 = self.tik_instance.Tensor('float16', (c_num*4,), tik.scope_ubuf, 'min_index_fp16')
            self._int8_conv_to_fp16(min_index_fp16, min_index_s8, c_num*4)
            # 0 0 index1 0 0 0 index2 0 -> 0 index1 0 index2
            min_index_fp16_tmp = self.tik_instance.Tensor('float16', (c_num*4,), tik.scope_ubuf, 'min_index_fp16_tmp')
            self._vcpadd_by_loop(min_index_fp16_tmp, min_index_fp16, c_num*4)
            # 0 index1 0 index2 -> index1 index2
            self._vcpadd_by_loop(min_index_fp16, min_index_fp16_tmp, c_num*2)
            repeat_time = c_num // self.fp16_vector_mask_max
            last_num = c_num % self.fp16_vector_mask_max
            if repeat_time > 0:
                # true n is index plus 1
                self.tik_instance.vec_adds(self.fp16_vector_mask_max, min_index_fp16, min_index_fp16, 1,
                    repeat_time, self.blk_per_rep, self.blk_per_rep)
                self.tik_instance.vec_conv(self.fp16_vector_mask_max, '', min_index_s8, min_index_fp16, repeat_time,
                    self.blk_per_rep//2, self.blk_per_rep)
            if last_num > 0:
                offset = repeat_time * self.fp16_vector_mask_max
                self.tik_instance.vec_adds(last_num, min_index_fp16[offset],
                    min_index_fp16[offset], 1, 1, self.blk_per_rep, self.blk_per_rep)
                self.tik_instance.vec_conv(last_num, '', min_index_s8[offset],
                    min_index_fp16[offset], 1, self.blk_per_rep//2, self.blk_per_rep)
            burst_len = ceil(c_num / Constant.BLOCK_BYTE_SIZE)
            self.tik_instance.tensor_mov(self.output_n[c_offset], min_index_s8, '', 1, burst_len, 0, 0)

    def _find_min_loss_by_fp32_split(self, c_num, final_loss_fp16):
        # split total_main_loss
        quotient1_trans = self.tik_instance.Tensor('float16', (c_num*Constant.N_STEP,),
            tik.scope_ubuf, 'quotient1_trans')
        quotient2_trans = self.tik_instance.Tensor('float16', (c_num*Constant.N_STEP,),
            tik.scope_ubuf, 'quotient2_trans')
        self._preprocess_loss_to_three_fp16(self.total_main_loss, c_num, quotient1_trans,
            quotient2_trans, final_loss_fp16)
        # reduce quotient1_trans, quotient2_trans, final_loss_fp16 in order
        min_result = self.tik_instance.Tensor('float16', (c_num,), tik.scope_ubuf, 'min_result')
        sel_mask = self.tik_instance.Tensor('uint64', (c_num*4,), tik.scope_ubuf, 'sel_mask')
        need_sel = self.tik_instance.Scalar('int32', 'need_sel')
        need_sel.set_as(0)
        # first reduce quotient1_trans
        self._vsel_and_vcgmin(quotient1_trans, sel_mask, c_num, min_result, need_sel)
        need_sel.set_as(1)
        # second reduce quotient2_trans
        self._vsel_and_vcgmin(quotient2_trans, sel_mask, c_num, min_result, need_sel)
        # third reduce final_loss_fp16
        self._vsel_and_vcgmin(final_loss_fp16, sel_mask, c_num, min_result, need_sel)
        # split total_remain_loss
        self._preprocess_loss_to_three_fp16(self.total_remain_loss, c_num, quotient1_trans,
            quotient2_trans, final_loss_fp16)
        # first reduce quotient1_trans
        self._vsel_and_vcgmin(quotient1_trans, sel_mask, c_num, min_result, need_sel)
        # second reduce quotient2_trans
        self._vsel_and_vcgmin(quotient2_trans, sel_mask, c_num, min_result, need_sel)
        # finally use sel_mask to select final_loss_fp16
        with self.tik_instance.for_range(0, c_num) as c_index:
            cur_mask = self.tik_instance.mov_tensor_to_cmpmask(sel_mask[4*c_index])
            self.tik_instance.vsel(Constant.N_STEP, 0, final_loss_fp16[c_index*Constant.N_STEP], cur_mask,
                final_loss_fp16[c_index*Constant.N_STEP], self.fp16_max_tensor, 1, 1, 1, 1,
                self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)

    def _find_min_loss_by_int32_split(self, c_num, final_loss_fp16):
        with self.tik_instance.new_stmt_scope():
            total_loss_u8 = self.total_loss.reinterpret_cast_to('uint8')
            # each c has a 128bit sel_mask(2*u64), for 32B-aligned, the space should be c_num * 4
            sel_mask = self.tik_instance.Tensor('uint64', (c_num*4,), tik.scope_ubuf, 'sel_mask')
            # for the first time, no value need to be masked
            need_sel = self.tik_instance.Scalar('int32', 'need_sel')
            need_sel.set_as(0)
            with self.tik_instance.for_range(0, 3) as group_idx:
                # split a int32 tensor to 4 fp16 tensor
                loss_num = Constant.N_STEP * c_num
                total_loss_fp16 = self.tik_instance.Tensor('float16', (loss_num*4,), tik.scope_ubuf, 'total_loss_fp16')
                # uint8 conv to fp16
                self._int8_conv_to_fp16(total_loss_fp16, total_loss_u8[loss_num*4*group_idx], loss_num*4)
                # extract dddd cccc bbbb aaaa from abcdabcdabcdabcd and compare in order
                sel_num_list = [0x8888, 0x4444, 0x2222, 0x1111]
                loss_split_size = ceil(loss_num * 2 / 64) * 64
                loss_split_fp16 = self.tik_instance.Tensor('float16', (loss_split_size,),
                    tik.scope_ubuf, 'loss_split_fp16')
                min_result = self.tik_instance.Tensor('float16', (c_num,), tik.scope_ubuf, 'min_result')
                for sel_num in sel_num_list:
                    with self.tik_instance.new_stmt_scope():
                        self._get_one_out_of_four_fp16(loss_split_fp16, total_loss_fp16, loss_num*4, sel_num)
                        self._transpose_loss_fp16(final_loss_fp16, loss_split_fp16, c_num)
                        self._vsel_and_vcgmin(final_loss_fp16, sel_mask, c_num, min_result, need_sel)
                    need_sel.set_as(1)

    def _vcpadd_by_loop(self, dst, src, data_num):
        loop_time = data_num // self.fp16_vector_mask_max // Constant.MAX_REPEAT_TIME
        repeat_time = data_num // self.fp16_vector_mask_max % Constant.MAX_REPEAT_TIME
        last_num = data_num % self.fp16_vector_mask_max
        if loop_time > 0:
            with self.tik_instance.for_range(0, loop_time) as loop_ind:
                offset = loop_ind * self.fp16_vector_mask_max
                self.tik_instance.vec_cpadd(self.fp16_vector_mask_max, dst[offset//2],
                    src[offset], Constant.MAX_REPEAT_TIME, 1, self.blk_per_rep)
        if repeat_time > 0:
            offset = loop_time * self.fp16_vector_mask_max * Constant.MAX_REPEAT_TIME
            self.tik_instance.vec_cpadd(self.fp16_vector_mask_max, dst[offset//2],
                src[offset], repeat_time, 1, self.blk_per_rep)
        if last_num > 0:
            offset = (loop_time * Constant.MAX_REPEAT_TIME + repeat_time) * self.fp16_vector_mask_max
            self.tik_instance.vec_cpadd(last_num, dst[offset//2], src[offset], 1, 1, self.blk_per_rep)

    def _int8_conv_to_fp16(self, dst, src, data_num):
        '''support int8 and uint8'''
        loop_time = data_num // self.fp16_vector_mask_max // Constant.MAX_REPEAT_TIME
        repeat_time = data_num // self.fp16_vector_mask_max % Constant.MAX_REPEAT_TIME
        last_num = data_num % self.fp16_vector_mask_max
        if loop_time > 0:
            with self.tik_instance.for_range(0, loop_time) as loop_ind:
                offset = loop_ind * self.fp16_vector_mask_max
                self.tik_instance.vec_conv(self.fp16_vector_mask_max, '', dst[offset],
                    src[offset], Constant.MAX_REPEAT_TIME, self.blk_per_rep, self.blk_per_rep//2)
        if repeat_time > 0:
            offset = loop_time * self.fp16_vector_mask_max * Constant.MAX_REPEAT_TIME
            self.tik_instance.vec_conv(self.fp16_vector_mask_max, '', dst[offset],
                src[offset], repeat_time, self.blk_per_rep, self.blk_per_rep//2)
        if last_num > 0:
            offset = (loop_time * Constant.MAX_REPEAT_TIME + repeat_time) * self.fp16_vector_mask_max
            self.tik_instance.vec_conv(last_num, '', dst[offset],
                src[offset], 1, self.blk_per_rep, self.blk_per_rep//2)

    def _transpose_loss_fp16(self, data_fp16_trans, data_fp16, c_num):
        src_list = [data_fp16[i*c_num] for i in range(Constant.N_STEP)]
        dst_list = [data_fp16_trans[i*Constant.N_STEP] for i in range(Constant.N_STEP)]
        repeats = c_num // Constant.N_STEP
        if repeats > 1:
            dst_rep_stride = Constant.N_STEP
            src_rep_stride = 1
        else:
            dst_rep_stride = 0
            src_rep_stride = 0
        self.tik_instance.vnchwconv(False, False, dst_list, src_list, repeats, dst_rep_stride, src_rep_stride)

    def _data_vand_num(self, dst, src, and_num, data_num, offset=0):
        '''a u16 tensor vand a int32 number'''
        and_tensor = self.tik_instance.Tensor('int32', (self.vector_mask_max,), tik.scope_ubuf, 'and_tensor')
        self.tik_instance.vec_dup(self.int32_vector_mask_max, and_tensor, and_num, 1, self.blk_per_rep)
        and_tensor_u16 = and_tensor.reinterpret_cast_to('uint16')
        repeats = data_num // self.fp16_vector_mask_max
        last_num = data_num % self.fp16_vector_mask_max
        if repeats > 0:
            self.tik_instance.vec_and(self.fp16_vector_mask_max, dst[offset], src[offset], and_tensor_u16, repeats,
                self.blk_per_rep, self.blk_per_rep, 0)
        if last_num > 0:
            offset += self.fp16_vector_mask_max * repeats
            self.tik_instance.vec_and(last_num, dst[offset], src[offset], and_tensor_u16, 1,
                self.blk_per_rep, self.blk_per_rep, 0)

    def _vsel_and_vcgmin(self, tensor, sel_mask, c_num, min_result, need_sel):
        '''
        use sel_mask to select loss tensor, then find the minimum and its index, return the new sel_mask
        sel_mask: an u64 tensor, for every 4 value there are 2 value that can be trans to a cmp_mask
        need_sel: 1 means some values in the tensor need to be masked by sel_mask; 0 means not
        '''
        with self.tik_instance.new_stmt_scope():
            with self.tik_instance.if_scope(need_sel):
                with self.tik_instance.for_range(0, c_num) as c_index:
                    cur_mask = self.tik_instance.mov_tensor_to_cmpmask(sel_mask[4*c_index])
                    self.tik_instance.vsel(Constant.N_STEP, 0, tensor[c_index*Constant.N_STEP], cur_mask,
                        tensor[c_index*Constant.N_STEP], self.fp16_max_tensor, 1, 1, 1, 1,
                        self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
            vcgmin_repeat_time = Constant.N_STEP * c_num // self.fp16_vector_mask_max
            self.tik_instance.vcgmin(self.fp16_vector_mask_max, min_result, tensor, vcgmin_repeat_time,
                1, 1, self.blk_per_rep)
            min_loss_scalar = self.tik_instance.Scalar('float16')
            min_loss_tensor = self.tik_instance.Tensor('float16', (Constant.N_STEP,),
                tik.scope_ubuf, 'min_loss_tensor')
            with self.tik_instance.for_range(0, c_num) as c_index:
                min_loss_scalar.set_as(min_result[c_index])
                self.tik_instance.vec_dup(Constant.N_STEP, min_loss_tensor, min_loss_scalar, 1, self.blk_per_rep)
                cmp_mask = self.tik_instance.vcmp_eq(Constant.N_STEP, min_loss_tensor,
                    tensor[c_index*Constant.N_STEP], self.blk_per_rep, self.blk_per_rep)
                # trans cmp_mask to tensor
                self.tik_instance.mov_cmpmask_to_tensor(sel_mask[4*c_index], cmp_mask)

    def _vcmin_n_steps_by_repeats(self, data, c_num):
        # one loop do 128 repeats, make sure dst_offset can be 32Byte-aligned
        tmp_max_repeats = 128
        vcmin_loop_time = c_num // tmp_max_repeats
        vcmin_last_repeat = c_num % tmp_max_repeats
        min_result = self.tik_instance.Tensor('float16', (c_num*2,), tik.scope_ubuf, 'min_result')
        if vcmin_loop_time > 0:
            with self.tik_instance.for_range(0, vcmin_loop_time) as loop_index:
                self.tik_instance.vcmin(Constant.N_STEP, min_result[loop_index*tmp_max_repeats*2],
                    data[loop_index*tmp_max_repeats*Constant.N_STEP],
                    tmp_max_repeats, 1, 1, 1)
        if vcmin_last_repeat > 0:
            dst_offset = vcmin_loop_time * tmp_max_repeats * 2
            src_offset = vcmin_loop_time * tmp_max_repeats * Constant.N_STEP
            self.tik_instance.vcmin(Constant.N_STEP, min_result[dst_offset], data[src_offset],
                vcmin_last_repeat, 1, 1, 1)
        return min_result

    def _fp32_mod(self, data, div_factor_value, data_num, quotient, remainder):
        '''fp32 data mod a div_factor, return fp32 quotient and remainder'''
        div_factor = self.tik_instance.Scalar('float32', 'div_factor')
        div_factor.set_as(div_factor_value)
        div_factor_rec = self.tik_instance.Scalar('float32', 'div_factor_rec')
        div_factor_rec.set_as(1/div_factor_value)
        repeats = data_num // self.vector_mask_max
        last_num = data_num % self.vector_mask_max
        if repeats > 0:
            self.tik_instance.vec_muls(self.vector_mask_max, quotient, data, div_factor_rec, repeats,
                self.blk_per_rep, self.blk_per_rep)
        if last_num > 0:
            self.tik_instance.vec_muls(last_num, quotient[self.vector_mask_max*repeats],
                data[self.vector_mask_max*repeats], div_factor_rec, 1, self.blk_per_rep, self.blk_per_rep)
        with self.tik_instance.new_stmt_scope():
            quotient_int32 = self.tik_instance.Tensor('int32', (data_num,), tik.scope_ubuf, 'quotient_int32')
            self._round_fp32_directly(self.vector_mask_max, quotient_int32, quotient, data_num, 'floor')
        if repeats > 0:
            self.tik_instance.vec_muls(self.vector_mask_max, remainder, quotient, div_factor, repeats,
                self.blk_per_rep, self.blk_per_rep)
            self.tik_instance.vec_sub(self.vector_mask_max, remainder, data, remainder, repeats,
                self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)
        if last_num > 0:
            self.tik_instance.vec_muls(last_num, remainder[repeats*self.vector_mask_max],
                quotient[repeats*self.vector_mask_max], div_factor, 1, self.blk_per_rep, self.blk_per_rep)
            self.tik_instance.vec_sub(last_num, remainder[repeats*self.vector_mask_max],
                data[repeats*self.vector_mask_max], remainder[repeats*self.vector_mask_max],
                1, self.blk_per_rep, self.blk_per_rep, self.blk_per_rep)

    def _conv_fp32_to_fp16(self, dst, src, data_num, mask, offset=0):
        repeats = data_num // mask
        last_num = data_num % mask
        if repeats > 0:
            self.tik_instance.vec_conv(mask, '', dst[offset], src[offset],
                repeats, self.blk_per_rep//2, self.blk_per_rep)
        if last_num > 0:
            self.tik_instance.vec_conv(last_num, '', dst[offset+repeats*mask],
                src[offset+repeats*mask], 1, self.blk_per_rep//2, self.blk_per_rep)

    def _split_fp32_to_fp16(self, data, data_num, quotient1, quotient2, remainder2):
        '''fp32 data split to (quotient1_fp32 * FP16_MAX + quotient2_fp32 * 100 + remainder2_fp32)'''
        remainder1 = self.tik_instance.Tensor('float32', (data_num,), tik.scope_ubuf, 'remainder1')
        self._fp32_mod(data, Constant.FP16_MAX, data_num, quotient1, remainder1)
        self._fp32_mod(remainder1, Constant.SECOND_DIV_FACTOR, data_num, quotient2, remainder2)


@para_check.check_op_params(
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.KERNEL_NAME)
def search_n(
        x,
        scale_d,
        scale_w,
        n,
        kernel_name='search_n'):
    """
    SearchN op
    """
    search_n_inst = SearchN(
        x,
        scale_d,
        scale_w,
        n,
        kernel_name)

    return search_n_inst.search_n_compute()
