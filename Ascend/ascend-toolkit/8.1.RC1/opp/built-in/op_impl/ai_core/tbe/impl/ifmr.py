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

ifmr
"""

# 'pylint: disable=import-error
from math import ceil
import functools
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.util_binary import get_bit_len
from impl.util.platform_adapter import para_check
from tbe.common.platform import set_current_compile_soc_info
from impl.util.platform_adapter import register_operator

# 'pylint: disable=locally-disabled,too-many-arguments
# 'pylint: disable=too-many-branches, too-many-statements, too-many-locals, attribute-defined-outside-init
# 'pylint: disable=too-many-instance-attributes, no-self-use, too-many-instance-attributes, protected-access
# 'pylint: disable=too-few-public-methods


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    This class for Constant.
    """
    SCALAR_MAX_FP16 = 65504
    SHAPE_SIZE_LIMIT = 2 ** 31
    MAX_BINS = 8192
    MAX_STEPS = 4096
    EPS = 1.192092896e-07
    REC_FLOAT16_PRECISION = 2 ** 24
    REC_FLOAT16_MIN_NORM_NUM = 2 ** 14
    DATA_TYPE_SIZE = {'float16': 2,
                      'int16': 2,
                      'float32': 4,
                      'int32': 4,
                      'int64': 8}

    def __init__(self):
        return


class Reconstruction():
    """IFMR: input feature map reconstruction"""
    def __init__(
            self,
            input_data,
            input_min,
            input_max,
            input_cumsum,
            output_scale,
            output_offset,
            min_percentile,
            max_percentile,
            search_range,
            search_step,
            with_offset,
            quant_bits,
            kernel_name):

        # The name with suffix "num" represent the true value of
        # variable. The name with suffix "size" represent the memory
        # space of variable. The name with suffix "repeat" represent the
        # number of repeat time when processing this variable.
        self.soc_version = tbe_platform.get_soc_spec("FULL_SOC_VERSION")
        status = set_current_compile_soc_info(self.soc_version)
        if status != "success":
            raise ValueError('Set soc_version failed, please check!')
        self.tik_instance = tik.Tik()
        if self.soc_version in ['Ascend610', 'BS9SX1AA', 'BS9SX1AB', 'BS9SX1AC']:
            self.aicore_num = 1
        else:
            self.aicore_num = tik.Dprofile().get_aicore_num()
        self.unified_buffer_size = tik.Dprofile().get_unified_buffer_size()
        
        self.sub_loss_ub = None
        self.total_loss = None
        self.loss_ub = None
        self.cmp_result_scalar = None
        self.cmp_result_scalar_fp32 = None
        self.carry_num = None
        self.input_fm_fp16_ub = None
        self.clip_max_ub = None
        self.clip_min_ub = None
        self.total_remain_loss = None
        self.input_fm_ub = None
        self.input_fm_quant_ub = None
        self.scale = None
        self.offset = None
        self.loss_max_ub = None
        self.one_loss_ub = None
        self.tmp_loss_ub = None
        self.remain_loss_ub = None
        self.work_ub = None
        self.one_remain_loss_ub = None

        self.data_dtype = input_data.get('dtype')
        data_shape = input_data.get('shape')
        self.data_num = functools.reduce(lambda x, y: x * y, data_shape)
        if self.data_num > Constant.SHAPE_SIZE_LIMIT:
            raise ValueError('Excessive amount of "input_data"(more than 2^31)!')

        if self.soc_version in ('SD3403'):
            self.calc_precision = 'float16'
        else:
            self.calc_precision = 'float32'

        if len(input_min.get('shape')) != 1 or input_min.get('shape')[0] != 1:
            raise ValueError('The shape of "input_min" must be "[1]"!')
        if len(input_max.get('shape')) != 1 or input_max.get('shape')[0] != 1:
            raise ValueError('The shape of "input_max" must be "[1]"!')

        cumsum_shape = input_cumsum.get('shape')
        if len(cumsum_shape) != 1:
            raise ValueError('The shape of "input_cumsum" must be "(x,)"!')
        self.cumsum_num = cumsum_shape[0]
        if cumsum_shape[0] > Constant.MAX_BINS:
            raise ValueError('Excessive amount of "input_cumsum"(more than 8192)!')
        self.cumsum_data_byte = Constant.DATA_TYPE_SIZE.get('int32')

        # input&output global memory
        self.input_data = self.tik_instance.Tensor(self.data_dtype, data_shape, tik.scope_gm, 'input_data')
        self.input_max = self.tik_instance.Tensor(self.data_dtype, (1,), tik.scope_gm, 'input_max')
        self.input_min = self.tik_instance.Tensor(self.data_dtype, (1,), tik.scope_gm, 'input_min')
        self.input_cumsum = self.tik_instance.Tensor('int32', cumsum_shape, tik.scope_gm, 'input_cumsum')
        self.output_scale = self.tik_instance.Tensor('float32', (1,), tik.scope_gm, 'output_scale')
        self.output_offset = self.tik_instance.Tensor('float32', (1,), tik.scope_gm, 'output_offset')

        # IFMR parameter initialize
        self.max_percentile = max_percentile
        self.min_percentile = 1 - min_percentile
        if search_range[1] <= search_range[0]:
            raise ValueError('The "search_range" must be monotonically increasing!')
        self.search_range = search_range
        if search_step <= 0:
            raise ValueError('The "search_step" must be positive!')
        self.search_step = search_step
        self.with_offset = with_offset
        if quant_bits != 8 and quant_bits != 16:
            raise ValueError('The "quant_bits" must be 8 or 16!')
        self.quant_bits = quant_bits
        self.quant_step_num = 2 ** self.quant_bits - 1
        self.clip_max = 2 ** (self.quant_bits - 1) - 1
        self.clip_min = -2 ** (self.quant_bits - 1)

        self.kernel_name = kernel_name

        # uniform use calc_precision in processing
        self.data_byte_size = Constant.DATA_TYPE_SIZE.get(self.calc_precision)
        self.block_byte_size = 32
        self.data_each_block = self.block_byte_size // self.data_byte_size
        self.vector_mask_max = 8 * self.data_each_block

        # search space
        self.steps_num = int((search_range[1] - search_range[0]) // search_step + 1)
        if self.steps_num > Constant.MAX_STEPS:
            raise ValueError('step size should be equal or less than 4096')

        self.steps_size = ceil(self.steps_num / self.vector_mask_max) * self.vector_mask_max
        self.step_repeat = self.steps_size // self.vector_mask_max

        if self.aicore_num > 1:
            self.barrier_workspace = self.tik_instance.Tensor(
                'int64', (self.aicore_num*self.block_byte_size//Constant.DATA_TYPE_SIZE.get('int64'),),
                tik.scope_gm, 'barrier_workspace', is_workspace=True, is_atomic_add=True)

        self.input_data_bytes_size = get_bit_len(self.data_dtype) // 8
        self.input_data_each_block = self.block_byte_size // self.input_data_bytes_size
        self.data_num_each_core = self.data_num // self.data_each_block // self.aicore_num * self.data_each_block
        self.data_num_last_core = self.data_num % self.data_each_block + \
            self.data_num // self.data_each_block % self.aicore_num * self.data_each_block

        if self.calc_precision == 'float32':
            ub_size_factor = 4
        else:
            ub_size_factor = 5
        self.ub_tensor_size = (
            (self.unified_buffer_size - 2 * self.steps_size * self.data_byte_size) //
            self.data_byte_size // ub_size_factor // self.data_each_block * self.data_each_block)

        self.loss_each_core = (
            self.steps_num + self.data_each_block - 1) // self.data_each_block * self.data_each_block
        self.loss_workspace = self.tik_instance.Tensor(
            self.calc_precision, (self.loss_each_core * self.aicore_num,), tik.scope_gm,
            'loss_workspace', is_workspace=True, is_atomic_add=True)
        if self.soc_version in ['SD3403']:
            self.loss_max = (Constant.SCALAR_MAX_FP16 - 1) // self.aicore_num
            self.remain_loss_workspace = self.tik_instance.Tensor(
                self.calc_precision, (self.loss_each_core * self.aicore_num,), tik.scope_gm,
                'remain_loss_workspace', is_workspace=True, is_atomic_add=True)

    def ifmr_compute(self):
        """
        IFMR compute function
        """
        with self.tik_instance.for_range(0, self.aicore_num, block_num=self.aicore_num) as index:
            self.scale = self.tik_instance.Tensor(self.calc_precision, (self.steps_size,), tik.scope_ubuf, 'scale')
            if self.with_offset:
                self.offset = self.tik_instance.Tensor(self.calc_precision, (self.steps_size,),
                    tik.scope_ubuf, 'offset')

            self._compute_scale_offset()
            self._compute_mse_loss(index)

            # define loss ub tensor here
            if self.aicore_num > 1:
                self.tik_instance.block_barrier(self.barrier_workspace)
            # core 0 do reduce
            with self.tik_instance.if_scope(index == 0):
                self._reduce_and_output()

        self.tik_instance.BuildCCE(
            self.kernel_name, [self.input_data, self.input_min, self.input_max, self.input_cumsum],
            [self.output_scale, self.output_offset])
        return self.tik_instance

    # used by soc versions except Ascend310 and SD3403
    def _int32_cumsum_to_fp32_cdf(self, cumsum_size, cumsum_repeat, move_burst):
        cumsum_int = self.tik_instance.Tensor('int32', (cumsum_size,), tik.scope_ubuf, 'cumsum_int')
        self.tik_instance.vec_dup(self.vector_mask_max, cumsum_int, 0, cumsum_repeat, 8)
        self.tik_instance.data_move(cumsum_int, self.input_cumsum, 0, 1, move_burst, 0, 0)

        cumsum_float = self.tik_instance.Tensor(self.calc_precision, (cumsum_size,), tik.scope_ubuf, 'cumsum_float')
        self.tik_instance.vec_conv(self.vector_mask_max, 'none', cumsum_float, cumsum_int, cumsum_repeat, 8, 8)
        cumsum_max = self.tik_instance.Tensor(self.calc_precision, (self.vector_mask_max,),
            tik.scope_ubuf, 'cumsum_max')
        cdf = self.tik_instance.Tensor(self.calc_precision, (cumsum_size,), tik.scope_ubuf, 'cdf')
        self.tik_instance.vec_dup(self.vector_mask_max, cumsum_max, self.data_num, 1, 8)
        self.tik_instance.vdiv(
            self.vector_mask_max, cdf, cumsum_float, cumsum_max, cumsum_repeat, 1, 1, 1, 8, 8, 0)
        return cdf

    # used by Ascend310, vconv int32 to fp16 with deqscale
    def _int32_cumsum_to_fp16_cdf_by_deqscale(self, cumsum_size, cumsum_repeat, move_burst):
        cumsum_int = self.tik_instance.Tensor('int32', (cumsum_size,), tik.scope_ubuf, 'cumsum_int')
        self.tik_instance.vec_dup(self.vector_mask_max, cumsum_int, 0, cumsum_repeat*2, 8)
        self.tik_instance.data_move(cumsum_int, self.input_cumsum, 0, 1, move_burst, 0, 0)

        vector_mask_max_fp16 = 128
        cdf = self.tik_instance.Tensor('float16', (cumsum_size,), tik.scope_ubuf, 'cdf')
        # fp16 minimum is 1/2^24, deqscale can not be smaller than 1/2^24
        if self.data_num >= Constant.REC_FLOAT16_PRECISION:
            scale_for_vconv = Constant.REC_FLOAT16_PRECISION
        # fp16 1/2^14 ~ 1/2^24 is denormed number, precision is not that subtle, deqscale set to 1/2^14
        elif self.data_num > Constant.REC_FLOAT16_MIN_NORM_NUM:
            scale_for_vconv = Constant.REC_FLOAT16_MIN_NORM_NUM
        else:
            scale_for_vconv = self.data_num
        deqscale_for_vconv = 1.0 / scale_for_vconv
        self.tik_instance.vec_conv(self.vector_mask_max, 'none', cdf, cumsum_int,
            cumsum_repeat*2, 4, 8, deqscale_for_vconv)
        # if data num larger than 2^14, multiply scale_for_vconv/data_num additionally
        if self.data_num > Constant.REC_FLOAT16_MIN_NORM_NUM:
            mul_factor = scale_for_vconv / self.data_num
            mul_factor_tensor = self.tik_instance.Tensor('float16', (vector_mask_max_fp16,),
                tik.scope_ubuf, 'mul_factor_tensor')
            self.tik_instance.vec_dup(vector_mask_max_fp16, mul_factor_tensor, mul_factor, 1, 8)
            self.tik_instance.vec_mul(vector_mask_max_fp16, cdf, cdf, mul_factor_tensor, cumsum_repeat, 8, 8, 0)
        return cdf

    # used by SD3403, split int32 to two int16, then conv to fp16
    def _int32_cumsum_to_fp16_cdf_by_int16(self, cumsum_size, cumsum_repeat, move_burst):
        cumsum_int16 = self.input_cumsum.reinterpret_cast_to("int16")
        cumsum_int16_size = cumsum_size * 2
        cumsum_int16_repeat = cumsum_repeat * 2

        cumsum_int16_ub = self.tik_instance.Tensor('int16', (cumsum_int16_size,), tik.scope_ubuf, 'cumsum_int16')
        self.tik_instance.data_move(cumsum_int16_ub, cumsum_int16, 0, 1, move_burst, 0, 0)
        cumsum_fp16_ub = self.tik_instance.Tensor('float16', (cumsum_int16_size,),
            tik.scope_ubuf, 'cumsum_fp16_ub')
        self.tik_instance.vec_conv(self.vector_mask_max, 'none', cumsum_fp16_ub,
            cumsum_int16_ub, cumsum_int16_repeat, 8, 8)

        adjust_num = 65536 / self.data_num
        adjust_num_tensor = self.tik_instance.Tensor('float16', (self.vector_mask_max,),
            tik.scope_ubuf, 'adjust_num_tensor')
        self.tik_instance.vec_dup(self.vector_mask_max, adjust_num_tensor, adjust_num, 1, 8)
        mask_cumsum_high = 0xAAAAAAAAAAAAAAAA
        self.tik_instance.vec_mul([mask_cumsum_high, mask_cumsum_high], cumsum_fp16_ub, cumsum_fp16_ub,
            adjust_num_tensor, cumsum_int16_repeat, 8, 8, 0)

        zeros = self.tik_instance.Tensor('float16', (self.vector_mask_max,), tik.scope_ubuf, 'zeros')
        self.tik_instance.vec_dup(self.vector_mask_max, zeros, 0, 1, 8)
        rec_data_num = 1.0 / self.data_num
        rec_data_num_tensor = self.tik_instance.Tensor('float16', (self.vector_mask_max,),
            tik.scope_ubuf, 'rec_data_num_tensor')
        self.tik_instance.vec_dup(self.vector_mask_max, rec_data_num_tensor, rec_data_num, 1, 8)

        mask_cumsum_low = 0x5555555555555555
        self.tik_instance.vec_mul([mask_cumsum_low, mask_cumsum_low], cumsum_fp16_ub, cumsum_fp16_ub,
            rec_data_num_tensor, cumsum_int16_repeat, 8, 8, 0)
        sel_tensor_size = ceil((cumsum_int16_size // 16) / self.vector_mask_max) * self.vector_mask_max
        sel_tensor = self.tik_instance.Tensor('uint16', (sel_tensor_size,), tik.scope_ubuf, 'sel_tensor')
        self.tik_instance.vec_cmpv_lt(sel_tensor, cumsum_fp16_ub, zeros, cumsum_int16_repeat, 8, 0)
        low_add_factor_tensor = self.tik_instance.Tensor('float16', (cumsum_int16_size,),
            tik.scope_ubuf, 'low_add_factor_tensor')
        self.tik_instance.vec_sel(self.vector_mask_max, 2, low_add_factor_tensor, sel_tensor,
            adjust_num_tensor, zeros, cumsum_int16_repeat, 8, 0, 0)
        self.tik_instance.vec_add([mask_cumsum_low, mask_cumsum_low], cumsum_fp16_ub,
            cumsum_fp16_ub, low_add_factor_tensor, cumsum_int16_repeat, 8, 8, 8)

        cdf = self.tik_instance.Tensor('float16', (cumsum_size,), tik.scope_ubuf, 'cdf')
        self.tik_instance.vec_dup(self.vector_mask_max, cdf, 0, cumsum_repeat, 8)
        repeats = self.cumsum_num * 2 // self.vector_mask_max
        mask = self.cumsum_num * 2 % self.vector_mask_max
        if repeats > 0:
            self.tik_instance.vec_cpadd(self.vector_mask_max, cdf, cumsum_fp16_ub, repeats, 1, 8)
        if mask > 0:
            self.tik_instance.vec_cpadd(mask, cdf[self.vector_mask_max*repeats//2],
                cumsum_fp16_ub[self.vector_mask_max*repeats], 1, 1, 8)
        return cdf

    def _calculate_max_list(self, max_init):
        max_list = self.tik_instance.Tensor(self.calc_precision, (self.steps_size,), tik.scope_ubuf, 'max_list')
        self.tik_instance.vec_dup(self.vector_mask_max, max_list, 0, self.step_repeat, 8)
        search_step = self.tik_instance.Tensor(self.calc_precision, (self.data_each_block,),
            tik.scope_ubuf, 'search_step')
        self.tik_instance.vec_dup(1, search_step, self.search_step, 1, 8)
        search_min = self.tik_instance.Tensor(self.calc_precision, (self.data_each_block,),
            tik.scope_ubuf, 'search_min')
        self.tik_instance.vec_dup(1, search_min, self.search_range[0], 1, 8)
        step_length = self.tik_instance.Tensor(self.calc_precision, (self.data_each_block,),
            tik.scope_ubuf, 'step_length')
        if self.soc_version in ['Ascend310']:
            step_length_int = self.tik_instance.Tensor('int32', (self.data_each_block,),
                tik.scope_ubuf, 'step_length_int')
            step_length_fp16 = self.tik_instance.Tensor('float16', (self.data_each_block,),
                tik.scope_ubuf, 'step_length_fp16')
        with self.tik_instance.for_range(0, self.steps_num) as i:
            if self.soc_version in ['Ascend310']:
                self.tik_instance.vec_dup(1, step_length_int, i, 1, 8)
                self.tik_instance.vec_conv(self.data_each_block, '', step_length_fp16, step_length_int, 1, 4, 8, 1.0)
                self.tik_instance.vec_conv(self.data_each_block, '', step_length, step_length_fp16, 1, 8, 4)
            else:
                self.tik_instance.vec_dup(1, step_length, i, 1, 8)
            self.tik_instance.vec_mul(1, step_length, step_length, search_step, 1, 8, 8, 8)
            self.tik_instance.vec_add(1, step_length, step_length, search_min, 1, 8, 8, 8)
            self.tik_instance.vec_mul(1, step_length, step_length, max_init, 1, 8, 8, 8)
            max_list[i].set_as(step_length[0])
        return max_list

    def _compute_scale_offset(self):
        """Calculate the candidate scale and offset."""
        if self.soc_version in ['SD3403', 'Ascend310']:
            cdf_precision = 'float16'
            cdf_vector_mask_max = 128
        else:
            cdf_precision = 'float32'
            cdf_vector_mask_max = 64

        cumsum_size = ceil(self.cumsum_num / cdf_vector_mask_max) * cdf_vector_mask_max
        cumsum_repeat = cumsum_size // cdf_vector_mask_max
        cumsum_redundance = cumsum_size - self.cumsum_num
        vector_mask_max_int32 = 64
        data_each_block_int32 = 8

        with self.tik_instance.new_stmt_scope():
            move_burst = ceil(self.cumsum_num * self.cumsum_data_byte / self.block_byte_size)
            if self.soc_version in ['SD3403']:
                cdf = self._int32_cumsum_to_fp16_cdf_by_int16(cumsum_size, cumsum_repeat, move_burst)
            elif self.soc_version in ['Ascend310']:
                cdf = self._int32_cumsum_to_fp16_cdf_by_deqscale(cumsum_size, cumsum_repeat, move_burst)
            else:
                cdf = self._int32_cumsum_to_fp32_cdf(cumsum_size, cumsum_repeat, move_burst)

            max_tensor = self.tik_instance.Tensor(cdf_precision, (cdf_vector_mask_max,),
                tik.scope_ubuf, 'max_tensor')
            min_tensor = self.tik_instance.Tensor(cdf_precision, (cdf_vector_mask_max,),
                tik.scope_ubuf, 'min_tensor')
            self.tik_instance.vec_dup(cdf_vector_mask_max, max_tensor, self.max_percentile, 1, 8)
            self.tik_instance.vec_dup(cdf_vector_mask_max, min_tensor, self.min_percentile, 1, 8)

            # 0&1 vector for comparing, just need length of "vector_mask_max"
            zeros = self.tik_instance.Tensor(cdf_precision, (cdf_vector_mask_max,), tik.scope_ubuf, 'zeros')
            ones = self.tik_instance.Tensor(cdf_precision, (cdf_vector_mask_max,), tik.scope_ubuf, 'ones')
            self.tik_instance.vec_dup(cdf_vector_mask_max, zeros, 0, 1, 8)
            self.tik_instance.vec_dup(cdf_vector_mask_max, ones, 1, 1, 8)

            # For convenience,  the length of the comparison results is
            # selected from only four values, respectively 2048, 4096,
            # 6144 and 8192
            compare_repeat = (
                self.cumsum_num + self.vector_mask_max * self.data_byte_size * 8 - 1) // (
                    self.vector_mask_max * self.data_byte_size * 8)
            compare_size = compare_repeat * vector_mask_max_int32
            result_max = self.tik_instance.Tensor(
                'uint32', (compare_size + data_each_block_int32,), tik.scope_ubuf, 'result_max')
            result_min = self.tik_instance.Tensor(
                'uint32', (compare_size + data_each_block_int32,), tik.scope_ubuf, 'result_min')
            self.tik_instance.vec_dup(vector_mask_max_int32, result_max, 0, compare_repeat, 8)
            self.tik_instance.vec_dup(vector_mask_max_int32, result_min, 0, compare_repeat, 8)

            self.tik_instance.vec_cmpv_gt(result_max, max_tensor, cdf, cumsum_repeat, 0, 8)
            self.tik_instance.vec_cmpv_gt(result_min, min_tensor, cdf, cumsum_repeat, 0, 8)

            stat_max = self.tik_instance.Tensor(cdf_precision, (cumsum_size,), tik.scope_ubuf, 'stat_max')
            stat_min = self.tik_instance.Tensor(cdf_precision, (cumsum_size,), tik.scope_ubuf, 'stat_min')
            self.tik_instance.vec_dup(cdf_vector_mask_max, stat_max, 0, cumsum_repeat, 8)
            self.tik_instance.vec_dup(cdf_vector_mask_max, stat_min, 0, cumsum_repeat, 8)

            select = self.tik_instance.Tensor('uint32', (data_each_block_int32,), tik.scope_ubuf, 'select')
            with self.tik_instance.for_range(0, cumsum_repeat) as i:
                if cdf_precision == 'float32':
                    select[0].set_as(result_max[i * 2])
                    select[1].set_as(result_max[i * 2 + 1])
                else:
                    select[0].set_as(result_max[i * 4])
                    select[1].set_as(result_max[i * 4 + 1])
                    select[2].set_as(result_max[i * 4 + 2])
                    select[3].set_as(result_max[i * 4 + 3])
                self.tik_instance.vec_sel(
                    cdf_vector_mask_max, 0, stat_max[i * cdf_vector_mask_max], select, ones, zeros, 1)
                if cdf_precision == 'float32':
                    select[0].set_as(result_min[i * 2])
                    select[1].set_as(result_min[i * 2 + 1])
                else:
                    select[0].set_as(result_min[i * 4])
                    select[1].set_as(result_min[i * 4 + 1])
                    select[2].set_as(result_min[i * 4 + 2])
                    select[3].set_as(result_min[i * 4 + 3])
                self.tik_instance.vec_sel(
                    cdf_vector_mask_max, 0, stat_min[i * cdf_vector_mask_max], select, ones, zeros, 1)

            max_index = self.tik_instance.Tensor(self.calc_precision, (self.data_each_block,),
                tik.scope_ubuf, 'max_index')
            min_index = self.tik_instance.Tensor(self.calc_precision, (self.data_each_block,),
                tik.scope_ubuf, 'min_index')
            if self.soc_version in ['Ascend310']:
                work_tensor = self.tik_instance.Tensor(self.calc_precision, (cumsum_repeat*2,),
                    tik.scope_ubuf, 'work_tensor')
                stat_max_tmp = self.tik_instance.Tensor(self.calc_precision, (cumsum_size,),
                    tik.scope_ubuf, 'stat_max_tmp')
                stat_min_tmp = self.tik_instance.Tensor(self.calc_precision, (cumsum_size,),
                    tik.scope_ubuf, 'stat_min_tmp')
                self.tik_instance.vec_conv(self.vector_mask_max, 'none', stat_max_tmp, stat_max, cumsum_repeat*2, 8, 4)
                self.tik_instance.vec_conv(self.vector_mask_max, 'none', stat_min_tmp, stat_min, cumsum_repeat*2, 8, 4)
                self.tik_instance.vec_reduce_add(self.vector_mask_max, max_index, stat_max_tmp,
                    work_tensor, cumsum_repeat*2, 8)
                self.tik_instance.vec_reduce_add(self.vector_mask_max, min_index, stat_min_tmp,
                    work_tensor, cumsum_repeat*2, 8)
            else:
                work_tensor = self.tik_instance.Tensor(self.calc_precision, (cumsum_repeat,),
                    tik.scope_ubuf, 'work_tensor')
                self.tik_instance.vec_reduce_add(self.vector_mask_max, max_index, stat_max,
                    work_tensor, cumsum_repeat, 8)
                self.tik_instance.vec_reduce_add(self.vector_mask_max, min_index, stat_min,
                    work_tensor, cumsum_repeat, 8)

            cumsum_redundance_tensor = self.tik_instance.Tensor(
                self.calc_precision, (self.data_each_block,), tik.scope_ubuf, 'cumsum_redundance_tensor')
            self.tik_instance.vec_dup(1, cumsum_redundance_tensor, cumsum_redundance, 1, 8)
            self.tik_instance.vec_sub(1, max_index, max_index, cumsum_redundance_tensor, 1, 8, 8, 8)
            self.tik_instance.vec_sub(1, min_index, min_index, cumsum_redundance_tensor, 1, 8, 8, 8)

            data_max = self.tik_instance.Tensor(self.calc_precision, (self.data_each_block,),
                tik.scope_ubuf, 'data_max')
            data_min = self.tik_instance.Tensor(self.calc_precision, (self.data_each_block,),
                tik.scope_ubuf, 'data_min')
            if self.data_dtype == self.calc_precision:
                self.tik_instance.data_move(data_max, self.input_max, 0, 1, 1, 0, 0)
                self.tik_instance.data_move(data_min, self.input_min, 0, 1, 1, 0, 0)
            else:
                input_data_max = self.tik_instance.Tensor(
                    self.data_dtype, (self.input_data_each_block,), tik.scope_ubuf, 'input_data_max')
                input_data_min = self.tik_instance.Tensor(
                    self.data_dtype, (self.input_data_each_block,), tik.scope_ubuf, 'input_data_min')
                self.tik_instance.data_move(input_data_max, self.input_max, 0, 1, 1, 0, 0)
                self.tik_instance.data_move(input_data_min, self.input_min, 0, 1, 1, 0, 0)
                self.tik_instance.vec_conv(1, 'none', data_max, input_data_max, 1, 8, 8)
                self.tik_instance.vec_conv(1, 'none', data_min, input_data_min, 1, 8, 8)

            max_init = self.tik_instance.Tensor(self.calc_precision, (self.data_each_block,),
                tik.scope_ubuf, 'max_init')
            min_init = self.tik_instance.Tensor(self.calc_precision, (self.data_each_block,),
                tik.scope_ubuf, 'min_init')
            cumsum_num_tensor = self.tik_instance.Tensor(
                self.calc_precision, (self.data_each_block,), tik.scope_ubuf, 'cumsum_num')
            data_range = self.tik_instance.Tensor(self.calc_precision, (self.data_each_block,),
                tik.scope_ubuf, 'data_range')
            if self.soc_version in ['Ascend310']:
                self.tik_instance.vec_dup(1, cumsum_num_tensor, 1.0/self.cumsum_num, 1, 8)
                self.tik_instance.vmul(1, max_init, max_index, cumsum_num_tensor, 1, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vmul(1, min_init, min_index, cumsum_num_tensor, 1, 1, 1, 1, 8, 8, 8)
            else:
                self.tik_instance.vec_dup(1, cumsum_num_tensor, self.cumsum_num, 1, 8)
                self.tik_instance.vdiv(1, max_init, max_index, cumsum_num_tensor, 1, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vdiv(1, min_init, min_index, cumsum_num_tensor, 1, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vec_sub(1, data_range, data_max, data_min, 1, 8, 8, 8)
            self.tik_instance.vec_mul(1, max_init, max_init, data_range, 1, 8, 8, 8)
            self.tik_instance.vec_mul(1, min_init, min_init, data_range, 1, 8, 8, 8)
            self.tik_instance.vec_add(1, max_init, max_init, data_min, 1, 8, 8, 8)
            self.tik_instance.vec_add(1, min_init, min_init, data_min, 1, 8, 8, 8)

            if self.with_offset:
                if self.soc_version in ['Ascend310']:
                    zeros = self.tik_instance.Tensor(self.calc_precision, (self.vector_mask_max,),
                        tik.scope_ubuf, 'zeros')
                    self.tik_instance.vec_dup(self.vector_mask_max, zeros, 0, 1, 8)
                self.tik_instance.vec_max(1, max_init, max_init, zeros, 1, 8, 8, 8)
                self.tik_instance.vec_min(1, min_init, min_init, zeros, 1, 8, 8, 8)
                min_value = self.tik_instance.Scalar(self.calc_precision, 'min_value', min_init[0])
                min_list = self.tik_instance.Tensor(self.calc_precision, (self.steps_size,), tik.scope_ubuf, 'min_list')
                self.tik_instance.vec_dup(self.vector_mask_max, min_list, min_value, self.step_repeat, 8)
            else:
                self.tik_instance.vec_abs(1, max_init, max_init, 1, 8, 8)
                self.tik_instance.vec_abs(1, min_init, min_init, 1, 8, 8)
                self.tik_instance.vec_max(1, max_init, max_init, min_init, 1, 8, 8, 8)

            max_list = self._calculate_max_list(max_init)

            if self.with_offset:
                self.tik_instance.vec_sub(
                    self.vector_mask_max, self.scale, max_list, min_list, self.step_repeat, 8, 8, 8)
                quant_step = self.tik_instance.Tensor(self.calc_precision, (self.vector_mask_max,),
                    tik.scope_ubuf, 'quant_step')
                if self.soc_version in ['Ascend310']:
                    self.tik_instance.vec_dup(self.vector_mask_max, quant_step, 1/self.quant_step_num, 1, 8)
                    self.tik_instance.vec_mul(self.vector_mask_max, self.scale, self.scale, quant_step,
                        self.step_repeat, 8, 8, 0)
                    src_extent_size = (self.step_repeat - 1) * 8 * self.data_each_block + self.vector_mask_max
                    wk_size = ((src_extent_size + self.data_each_block - 1)
                        // self.data_each_block) * self.data_each_block * 2
                    rec_work_tensor = self.tik_instance.Tensor(self.calc_precision, (wk_size,),
                        tik.scope_ubuf, 'rec_work_tensor')
                    rec_scale = self.tik_instance.Tensor(self.calc_precision, (self.steps_size,),
                        tik.scope_ubuf, 'rec_scale')
                    self.tik_instance.vec_rec_high_preci(self.vector_mask_max, rec_scale, self.scale,
                        rec_work_tensor, self.step_repeat, 8, 8)
                    self.tik_instance.vec_mul(self.vector_mask_max, self.offset, min_list, rec_scale,
                        self.step_repeat, 8, 8, 8)
                    self._round_fp32_through_fp16(self.vector_mask_max, self.offset, 0, self.step_repeat)
                else:
                    self.tik_instance.vec_dup(self.vector_mask_max, quant_step, self.quant_step_num, 1, 8)
                    self.tik_instance.vdiv(
                        self.vector_mask_max, self.scale, self.scale, quant_step, self.step_repeat, 1, 1, 1, 8, 8, 0)
                    self.tik_instance.vdiv(
                        self.vector_mask_max, self.offset, min_list, self.scale, self.step_repeat, 1, 1, 1, 8, 8, 8)
                    if self.soc_version in ['SD3403']:
                        data_num = self.vector_mask_max * self.step_repeat
                        self._round_fp16_through_int16(data_num, self.offset, 0)
                    else:
                        self._round_fp32(self.vector_mask_max, self.offset, 0, self.step_repeat)

                half_quant_step_const = 2 ** (self.quant_bits - 1)
                half_quant_step = self.tik_instance.Tensor(
                    self.calc_precision, (self.vector_mask_max,), tik.scope_ubuf, 'half_quant_step')
                self.tik_instance.vec_dup(self.vector_mask_max, half_quant_step, half_quant_step_const, 1, 8)
                self.tik_instance.vec_add(
                    self.vector_mask_max, self.offset, self.offset, half_quant_step, self.step_repeat, 8, 8, 0)
                negative = self.tik_instance.Tensor(self.calc_precision, (self.vector_mask_max,),
                    tik.scope_ubuf, 'negative')
                self.tik_instance.vec_dup(self.vector_mask_max, negative, -1, 1, 8)
                self.tik_instance.vec_mul(
                    self.vector_mask_max, self.offset, self.offset, negative, self.step_repeat, 8, 8, 0)
            else:
                quant_step_const = 2 ** (self.quant_bits - 1) - 1
                if self.soc_version in ['Ascend310']:
                    quant_step_scalar = self.tik_instance.Scalar(self.calc_precision, 'quant_step')
                    quant_step_scalar.set_as(1/quant_step_const)
                    self.tik_instance.vec_muls(self.vector_mask_max, self.scale, max_list, quant_step_scalar,
                        self.step_repeat, 8, 8)
                else:
                    quant_step = self.tik_instance.Tensor(self.calc_precision, (self.vector_mask_max,),
                        tik.scope_ubuf, 'quant_step')
                    self.tik_instance.vec_dup(self.vector_mask_max, quant_step, quant_step_const, 1, 8)
                    self.tik_instance.vdiv(
                        self.vector_mask_max, self.scale, max_list, quant_step, self.step_repeat, 1, 1, 1, 8, 8, 0)

    def _compute_mse_loss(self, index):
        with self.tik_instance.new_stmt_scope():
            loss_repeat = ceil(self.steps_size / self.vector_mask_max)
            self.loss_ub = self.tik_instance.Tensor(self.calc_precision, (self.steps_size,), tik.scope_ubuf, 'loss_ub')
            self.tik_instance.vec_dup(self.vector_mask_max, self.loss_ub, 0, loss_repeat, 8)
            self.input_fm_ub = self.tik_instance.Tensor(self.calc_precision, (self.ub_tensor_size, ),
                tik.scope_ubuf, 'fm_ub')
            self.input_fm_quant_ub = self.tik_instance.Tensor(
                self.calc_precision, (self.ub_tensor_size, ), tik.scope_ubuf, 'fm_quant_ub')
            self.one_loss_ub = self.tik_instance.Tensor(
                self.calc_precision, (self.data_each_block,), tik.scope_ubuf, 'one_loss_ub')
            if self.soc_version in ['SD3403']:
                self.remain_loss_ub = self.tik_instance.Tensor(self.calc_precision, (self.steps_size,),
                    tik.scope_ubuf, 'remain_loss_ub')
                self.one_remain_loss_ub = self.tik_instance.Tensor(
                    self.calc_precision, (self.data_each_block,), tik.scope_ubuf, 'one_remain_loss_ub')
                self.loss_max_ub = self.tik_instance.Tensor(
                    self.calc_precision, (self.data_each_block,), tik.scope_ubuf, 'loss_max_ub')
                self.tik_instance.vec_dup(self.data_each_block, self.loss_max_ub, self.loss_max, 1, 8)
                self.sub_loss_ub = self.tik_instance.Tensor(
                    self.calc_precision, (self.data_each_block,), tik.scope_ubuf, 'sub_loss_ub')
            with self.tik_instance.for_range(0, self.steps_num) as search_index:
                self.tik_instance.vec_dup(self.data_each_block, self.one_loss_ub, 0, 1, 8)
                if self.soc_version in ['SD3403']:
                    self.tik_instance.vec_dup(self.data_each_block, self.one_remain_loss_ub, 0, 1, 8)
                self._compute_mse(index, search_index)
            self._move_loss_out(index)

    def _compute_mse(self, index, search_index):
        with self.tik_instance.if_scope(index == (self.aicore_num - 1)):
            input_data_each_core = self.data_num_last_core + self.data_num_each_core
            move_offset = index * self.data_num_each_core
            loop_time = input_data_each_core // self.ub_tensor_size
            if loop_time > 0:
                with self.tik_instance.for_range(0, loop_time) as loop_index:
                    move_offset += loop_index * self.ub_tensor_size
                    # do vector mse
                    self._mse_compute_each_loop(move_offset, self.ub_tensor_size, search_index)
            move_offset = index * self.data_num_each_core + loop_time * self.ub_tensor_size
            last_num = input_data_each_core % self.ub_tensor_size
            if last_num > 0:
                # do last num mse
                self._mse_compute_each_loop(move_offset, last_num, search_index)

        with self.tik_instance.else_scope():
            move_offset = index * self.data_num_each_core
            loop_time = self.data_num_each_core // self.ub_tensor_size
            if loop_time > 0:
                with self.tik_instance.for_range(0, loop_time) as loop_index:
                    move_offset += loop_index * self.ub_tensor_size
                    # do vector mse
                    self._mse_compute_each_loop(move_offset, self.ub_tensor_size, search_index)
            move_offset = index * self.data_num_each_core + loop_time * self.ub_tensor_size
            last_num = self.data_num_each_core % self.ub_tensor_size
            if last_num > 0:
                # do last num mse
                self._mse_compute_each_loop(move_offset, last_num, search_index)

    def _mse_compute_each_loop(self, move_offset, move_num, search_index):
        # cal each loop move burst
        if self.data_dtype == 'float16' and self.calc_precision == 'float32':
            # conv fp16 to fp32
            with self.tik_instance.new_stmt_scope():
                self.input_fm_fp16_ub = self.tik_instance.Tensor(
                    'float16', (self.ub_tensor_size,), tik.scope_ubuf, 'input_fm_fp16_ub')
                self._convert_each_loop(move_offset, move_num)
        else:
            burst_len = ceil(move_num / self.data_each_block)
            self.tik_instance.data_move(self.input_fm_ub, self.input_data[move_offset], 0, 1, burst_len, 0, 0)
        mse_loop = move_num // (self.vector_mask_max * 255)
        mse_offset = 0
        if mse_loop > 0:
            with self.tik_instance.for_range(0, mse_loop) as mse_index:
                mse_offset = mse_index * self.vector_mask_max * 255
                # do mse
                repeat_times = 255
                self._ifmr_mse(self.vector_mask_max, repeat_times, mse_offset, search_index)

        repeat_time = (move_num % (self.vector_mask_max * 255) // self.vector_mask_max)
        if repeat_time > 0:
            # do mse
            mse_offset = mse_loop * self.vector_mask_max * 255
            repeat_times = repeat_time
            self._ifmr_mse(self.vector_mask_max, repeat_times, mse_offset, search_index)
        last_num = move_num % self.vector_mask_max

        if last_num > 0:
            # do mse
            mse_offset = mse_loop * self.vector_mask_max * 255 + repeat_time * self.vector_mask_max
            repeat_times = 1
            self._ifmr_mse(last_num, repeat_times, mse_offset, search_index)

    def _convert_each_loop(self, move_offset, move_num):
        fp16_burst_len = ceil(move_num / self.input_data_each_block)
        self.tik_instance.data_move(self.input_fm_fp16_ub, self.input_data[move_offset], 0, 1, fp16_burst_len, 0, 0)
        convert_loop = move_num // (self.vector_mask_max * 255)
        convert_offset = 0
        if convert_loop > 0:
            with self.tik_instance.for_range(0, convert_loop) as convert_index:
                convert_offset = convert_index * self.vector_mask_max * 255
                repeat_times = 255
                self.tik_instance.vec_conv(
                    self.vector_mask_max, '', self.input_fm_ub[convert_offset], self.input_fm_fp16_ub[convert_offset],
                    repeat_times, 8, 4)

        repeat_time = (move_num % (self.vector_mask_max * 255) // self.vector_mask_max)
        if repeat_time > 0:
            convert_offset = convert_loop * self.vector_mask_max * 255
            repeat_times = repeat_time
            self.tik_instance.vec_conv(
                self.vector_mask_max, '', self.input_fm_ub[convert_offset], self.input_fm_fp16_ub[convert_offset],
                repeat_times, 8, 4)
        last_num = move_num % self.vector_mask_max

        if last_num > 0:
            convert_offset = convert_loop * self.vector_mask_max * 255 + repeat_time * self.vector_mask_max
            repeat_times = 1
            self.tik_instance.vec_conv(
                last_num, '', self.input_fm_ub[convert_offset], self.input_fm_fp16_ub[convert_offset], repeat_times, 8,
                4)

    # fp32->int32(round), int32->fp32, used by soc versions except Ascend310 and SD3403
    def _round_fp32(self, mask, src, offset, repeat):
        tmp_ub_int32 = self.tik_instance.Tensor('int32', (mask*repeat,), tik.scope_ubuf, 'tmp_ub_int32')
        self.tik_instance.vec_conv(mask, 'round', tmp_ub_int32, src[offset], repeat, 8, 8)
        self.tik_instance.vec_conv(mask, '', src[offset], tmp_ub_int32, repeat, 8, 8)

    # fp32->fp16, fp16->int32(round), int32->fp16, fp16->fp32, used by Ascend310
    def _round_fp32_through_fp16(self, mask, src, offset, repeat):
        tmp_ub_int32 = self.tik_instance.Tensor('int32', (mask*repeat,), tik.scope_ubuf, 'tmp_ub_int32')
        tmp_ub_fp16 = self.tik_instance.Tensor('float16', (mask*repeat,), tik.scope_ubuf, 'tmp_ub_fp16')
        self.tik_instance.vec_conv(mask, '', tmp_ub_fp16, src[offset], repeat, 4, 8)
        self.tik_instance.vec_conv(mask, 'round', tmp_ub_int32, tmp_ub_fp16, repeat, 8, 4)
        self.tik_instance.vec_conv(mask, '', tmp_ub_fp16, tmp_ub_int32, repeat, 4, 8, 1.0)
        self.tik_instance.vec_conv(mask, '', src[offset], tmp_ub_fp16, repeat, 8, 4)

    # fp16->int32(round), int32->int16, int16->fp16, used by SD3403
    def _round_fp16_through_int16(self, data_num, src, offset):
        vector_mask_max_int32 = 64
        repeat_time_int32 = data_num // vector_mask_max_int32
        loop_repeat_time = repeat_time_int32 // 255
        remain_repeat_time = repeat_time_int32 % 255
        last_num_int32 = data_num % vector_mask_max_int32

        if loop_repeat_time > 0:
            tmp_ub_int16 = self.tik_instance.Tensor('int16', (vector_mask_max_int32 * 255,),
                tik.scope_ubuf, 'tmp_ub_int16')
            tmp_ub_int32 = self.tik_instance.Tensor('int32', (vector_mask_max_int32 * 255,),
                tik.scope_ubuf, 'tmp_ub_int32')
        elif remain_repeat_time > 0:
            tmp_ub_int16 = self.tik_instance.Tensor('int16', (vector_mask_max_int32 * remain_repeat_time,),
                tik.scope_ubuf, 'tmp_ub_int16')
            tmp_ub_int32 = self.tik_instance.Tensor('int32', (vector_mask_max_int32 * remain_repeat_time,),
                tik.scope_ubuf, 'tmp_ub_int32')
        elif last_num_int32 > 0:
            tmp_ub_int16 = self.tik_instance.Tensor('int16', (vector_mask_max_int32,), tik.scope_ubuf, 'tmp_ub_int16')
            tmp_ub_int32 = self.tik_instance.Tensor('int32', (vector_mask_max_int32,), tik.scope_ubuf, 'tmp_ub_int32')

        if loop_repeat_time > 0:
            with self.tik_instance.for_range(0, loop_repeat_time) as loop_id:
                # fp16 to int32 (round)
                self.tik_instance.vec_conv(vector_mask_max_int32, 'round', tmp_ub_int32,
                    src[offset+loop_id*255*vector_mask_max_int32], 255, 8, 4)
                # int32 to int16
                self.tik_instance.vcbd(vector_mask_max_int32, tmp_ub_int16, tmp_ub_int32, 255, 1, 1, 4, 8)
                # int16 to fp16 (none)
                self.tik_instance.vec_conv(vector_mask_max_int32, '',
                    src[offset+loop_id*255*vector_mask_max_int32], tmp_ub_int16, 255, 4, 4)
        if remain_repeat_time > 0:
            self.tik_instance.vec_conv(vector_mask_max_int32, 'round', tmp_ub_int32,
                src[offset+loop_repeat_time*255*vector_mask_max_int32], remain_repeat_time, 8, 4)
            self.tik_instance.vcbd(vector_mask_max_int32, tmp_ub_int16, tmp_ub_int32, remain_repeat_time, 1, 1, 4, 8)
            self.tik_instance.vec_conv(vector_mask_max_int32, '',
                src[offset+loop_repeat_time*255*vector_mask_max_int32], tmp_ub_int16, remain_repeat_time, 4, 4)
        if last_num_int32 > 0:
            self.tik_instance.vec_conv(last_num_int32, 'round', tmp_ub_int32,
                src[offset+repeat_time_int32*vector_mask_max_int32], 1, 8, 4)
            self.tik_instance.vcbd(last_num_int32, tmp_ub_int16, tmp_ub_int32, 1, 1, 1, 4, 8)
            self.tik_instance.vec_conv(last_num_int32, '',
                src[offset+repeat_time_int32*vector_mask_max_int32], tmp_ub_int16, 1, 4, 4)

    def _ifmr_mse(self, mask_num, repeat_time, mse_offset, search_index):
        with self.tik_instance.new_stmt_scope():
            scale_scalar = self.tik_instance.Scalar(self.calc_precision)
            scale_scalar.set_as(self.scale[search_index])
            if self.soc_version in ['SD3403']:
                new_scale = self.tik_instance.Tensor(self.calc_precision, (self.vector_mask_max,),
                    tik.scope_ubuf, 'new_scale')
                self.tik_instance.vec_dup(self.vector_mask_max, new_scale, scale_scalar, 1, 8)
                with self.tik_instance.new_stmt_scope():
                    self.tik_instance.vdiv(
                        mask_num, self.input_fm_quant_ub[mse_offset], self.input_fm_ub[mse_offset], new_scale,
                        repeat_time, 1, 1, 1, 8, 8, 0)
                    data_num = mask_num * repeat_time
                    self._round_fp16_through_int16(data_num, self.input_fm_quant_ub, mse_offset)
            elif self.soc_version in ['Ascend310']:
                new_scale = self.tik_instance.Tensor(self.calc_precision, (self.vector_mask_max,),
                    tik.scope_ubuf, 'new_scale')
                self.tik_instance.vec_dup(self.vector_mask_max, new_scale, scale_scalar, 1, 8)
                rec_scale_tensor = self.tik_instance.Tensor(self.calc_precision, (self.vector_mask_max,),
                    tik.scope_ubuf, 'rec_scale_tensor')
                wk_size = ((self.vector_mask_max + self.data_each_block - 1)
                    // self.data_each_block) * self.data_each_block * 2
                rec_work_tensor = self.tik_instance.Tensor(self.calc_precision, (wk_size,),
                    tik.scope_ubuf, 'rec_work_tensor')
                self.tik_instance.vec_rec_high_preci(self.vector_mask_max, rec_scale_tensor, new_scale,
                    rec_work_tensor, 1, 8, 8)
                with self.tik_instance.new_stmt_scope():
                    self.tik_instance.vec_mul(mask_num, self.input_fm_quant_ub[mse_offset],
                        self.input_fm_ub[mse_offset], rec_scale_tensor, repeat_time, 8, 8, 0)
                    self._round_fp32_through_fp16(mask_num, self.input_fm_quant_ub, mse_offset, repeat_time)
            else:
                new_scale_scalar = self.tik_instance.Scalar(self.calc_precision)
                new_scale_scalar.set_as(1.0 / scale_scalar)
                with self.tik_instance.new_stmt_scope():
                    self.tik_instance.vec_muls(
                        mask_num, self.input_fm_quant_ub[mse_offset], self.input_fm_ub[mse_offset], new_scale_scalar,
                        repeat_time, 8, 8)
                    self._round_fp32(mask_num, self.input_fm_quant_ub, mse_offset, repeat_time)

            if self.with_offset:
                offset_scalar = self.tik_instance.Scalar(self.calc_precision)
                offset_scalar.set_as(self.offset[search_index])
                self.tik_instance.vec_adds(
                    mask_num, self.input_fm_quant_ub[mse_offset], self.input_fm_quant_ub[mse_offset], offset_scalar,
                    repeat_time, 8, 8)
            # clip fm_quant max min
            with self.tik_instance.new_stmt_scope():
                self.clip_max_ub = self.tik_instance.Tensor(self.calc_precision, (mask_num,),
                    tik.scope_ubuf, 'clip_max_ub')
                self.tik_instance.vec_dup(mask_num, self.clip_max_ub, self.clip_max, 1, 8)
                self.tik_instance.vec_min(
                    mask_num, self.input_fm_quant_ub[mse_offset], self.input_fm_quant_ub[mse_offset], self.clip_max_ub,
                    repeat_time, 8, 8, 0)
            with self.tik_instance.new_stmt_scope():
                self.clip_min_ub = self.tik_instance.Tensor(self.calc_precision, (mask_num,),
                    tik.scope_ubuf, 'clip_min_ub')
                self.tik_instance.vec_dup(mask_num, self.clip_min_ub, self.clip_min, 1, 8)
                self.tik_instance.vec_max(
                    mask_num, self.input_fm_quant_ub[mse_offset], self.input_fm_quant_ub[mse_offset], self.clip_min_ub,
                    repeat_time, 8, 8, 0)
            if self.with_offset:
                new_offset_scalar = self.tik_instance.Scalar(self.calc_precision)
                if self.soc_version in ['SD3403', 'Ascend310']:
                    new_offset_scalar.set_as(offset_scalar)
                    new_offset = self.tik_instance.Tensor(self.calc_precision, (self.vector_mask_max,),
                        tik.scope_ubuf, 'new_offset')
                    self.tik_instance.vec_dup(self.vector_mask_max, new_offset, new_offset_scalar, 1, 8)
                    new_offset_tensor = self.tik_instance.Tensor(self.calc_precision, (self.vector_mask_max,),
                        tik.scope_ubuf, 'new_offset_tensor')
                    self.tik_instance.vec_muls(self.vector_mask_max, new_offset_tensor, new_offset, -1.0, 1, 8, 8)
                    self.tik_instance.vec_add(mask_num, self.input_fm_quant_ub[mse_offset],
                        self.input_fm_quant_ub[mse_offset], new_offset_tensor, repeat_time, 8, 8, 0)
                else:
                    new_offset_scalar.set_as(-1.0 * offset_scalar)
                    self.tik_instance.vec_adds(
                        mask_num, self.input_fm_quant_ub[mse_offset], self.input_fm_quant_ub[mse_offset],
                        new_offset_scalar, repeat_time, 8, 8)
            self.tik_instance.vec_muls(
                mask_num, self.input_fm_quant_ub[mse_offset], self.input_fm_quant_ub[mse_offset], scale_scalar,
                repeat_time, 8, 8)
            self.tik_instance.vec_sub(
                mask_num, self.input_fm_quant_ub[mse_offset], self.input_fm_quant_ub[mse_offset],
                self.input_fm_ub[mse_offset], repeat_time, 8, 8, 8)
            self.tik_instance.vec_mul(
                mask_num, self.input_fm_quant_ub[mse_offset], self.input_fm_quant_ub[mse_offset],
                self.input_fm_quant_ub[mse_offset], repeat_time, 8, 8, 8)

        # reduce each loss and move to loss_ub
        with self.tik_instance.new_stmt_scope():
            self.work_ub = self.tik_instance.Tensor(self.calc_precision, (repeat_time,), tik.scope_ubuf, 'work_ub')
            self.tmp_loss_ub = self.tik_instance.Tensor(
                self.calc_precision, (self.data_each_block,), tik.scope_ubuf, 'tmp_loss_ub')
            self.tik_instance.vec_dup(self.data_each_block, self.tmp_loss_ub, 0, 1, 8)
            self.tik_instance.vec_reduce_add(
                mask_num, self.tmp_loss_ub, self.input_fm_quant_ub[mse_offset], self.work_ub, repeat_time, 8)
            if self.soc_version in ['SD3403']:
                self.tik_instance.vec_sub(1, self.sub_loss_ub,
                    self.loss_max_ub, self.one_remain_loss_ub, 1, 8, 8, 8)
                cmp_mask = self.tik_instance.vcmp_gt(1, self.sub_loss_ub, self.tmp_loss_ub, 1, 1)
                ones = self.tik_instance.Tensor(self.calc_precision, (self.data_each_block,), tik.scope_ubuf, 'ones')
                self.tik_instance.vec_dup(self.data_each_block, ones, 1, 1, 8)
                zeros = self.tik_instance.Tensor(self.calc_precision, (self.data_each_block,), tik.scope_ubuf, 'zeros')
                self.tik_instance.vec_dup(self.data_each_block, zeros, 0, 1, 8)
                self.tik_instance.vsel(1, 0, ones, cmp_mask, ones, zeros, 1, 1, 1, 1)
                self.cmp_result_scalar = self.tik_instance.Scalar(dtype=self.calc_precision)
                self.cmp_result_scalar.set_as(ones[0])
                self.cmp_result_scalar_fp32 = self.tik_instance.Scalar(dtype='float32')
                self.tik_instance.scalar_conv('', self.cmp_result_scalar_fp32, self.cmp_result_scalar)

                with self.tik_instance.if_scope(self.cmp_result_scalar_fp32):
                    self.tik_instance.vec_add(1, self.one_remain_loss_ub,
                        self.one_remain_loss_ub, self.tmp_loss_ub, 1, 8, 8, 8)
                with self.tik_instance.else_scope():
                    self.tik_instance.vec_sub(1, self.one_remain_loss_ub,
                        self.tmp_loss_ub, self.sub_loss_ub, 1, 8, 8, 8)
                    self.carry_num = self.tik_instance.Tensor(
                        self.calc_precision, (self.data_each_block,), tik.scope_ubuf, 'carry_num')
                    self.tik_instance.vdiv(1, self.carry_num, self.one_remain_loss_ub, self.loss_max_ub,
                        1, 1, 1, 1, 8, 8, 8)
                    carry_num_int32 = self.tik_instance.Tensor('int32',
                        (self.data_each_block//2,), tik.scope_ubuf, 'carry_num_int32')
                    carry_num_int16 = self.tik_instance.Tensor('int16',
                        (self.data_each_block,), tik.scope_ubuf, 'carry_num_int16')
                    self.tik_instance.vec_conv(1, 'floor', carry_num_int32, self.carry_num, 1, 8, 4)
                    self.tik_instance.vcbd(1, carry_num_int16, carry_num_int32, 1, 1, 1, 4, 8)
                    self.tik_instance.vec_conv(1, '', self.carry_num, carry_num_int16, 1, 8, 8)
                    self.tik_instance.vec_mul(1, self.sub_loss_ub, self.carry_num, self.loss_max_ub, 1, 8, 8, 8)
                    self.tik_instance.vec_sub(1, self.one_remain_loss_ub, self.one_remain_loss_ub,
                        self.sub_loss_ub, 1, 8, 8, 8)
                    self.tik_instance.vec_adds(1, self.carry_num, self.carry_num, 1, 1, 8, 8)
                    self.tik_instance.vec_add(1, self.one_loss_ub, self.one_loss_ub, self.carry_num, 1, 8, 8, 8)
                self.remain_loss_ub[search_index].set_as(self.one_remain_loss_ub[0])
            else:
                self.tik_instance.vec_add(1, self.one_loss_ub, self.one_loss_ub, self.tmp_loss_ub, 1, 8, 8, 8)
            self.loss_ub[search_index].set_as(self.one_loss_ub[0])

    def _move_loss_out(self, index):
        burst_len = self.loss_each_core // self.data_each_block
        self.tik_instance.tensor_mov(
            self.loss_workspace[index * self.loss_each_core], self.loss_ub, '', 1, burst_len, 0, 0)
        if self.soc_version in ['SD3403']:
            self.tik_instance.tensor_mov(
                self.remain_loss_workspace[index * self.loss_each_core], self.remain_loss_ub, '', 1, burst_len, 0, 0)

    def _gather_total_loss(self, repeats, mask):
        if self.soc_version in ['SD3403']:
            self.total_loss = self.tik_instance.Tensor(self.calc_precision, (self.steps_size,),
                tik.scope_ubuf, 'total_loss')
            self.tik_instance.vec_dup(self.vector_mask_max, self.total_loss,
                Constant.SCALAR_MAX_FP16, self.step_repeat, 8)
            self.total_remain_loss = self.tik_instance.Tensor(self.calc_precision, (self.steps_size,),
                tik.scope_ubuf, 'total_remain_loss')
            self.tik_instance.vec_dup(self.vector_mask_max, self.total_remain_loss,
                Constant.SCALAR_MAX_FP16, self.step_repeat, 8)
        else:
            self.total_loss = self.tik_instance.Tensor(self.calc_precision, (self.loss_each_core,),
                tik.scope_ubuf, 'total_loss')

        # zero out every item
        if repeats > 0:
            self.tik_instance.vec_dup(self.vector_mask_max, self.total_loss, 0, repeats, 8)
        if mask > 0:
            self.tik_instance.vec_dup(mask, self.total_loss[self.vector_mask_max * repeats], 0, 1, 8)
        # loss of one core
        loss = self.tik_instance.Tensor(self.calc_precision, (self.loss_each_core,), tik.scope_ubuf, 'loss')
        burst_len = self.loss_each_core // self.data_each_block
        for i in range(0, self.aicore_num):
            self.tik_instance.data_move(loss, self.loss_workspace[self.loss_each_core * i], 0, 1, burst_len, 0, 0)
            if repeats > 0:
                self.tik_instance.vec_add(self.vector_mask_max, self.total_loss, loss, self.total_loss,
                    repeats, 8, 8, 8)
            if mask > 0:
                self.tik_instance.vec_add(
                    mask, self.total_loss[self.vector_mask_max * repeats], loss[self.vector_mask_max * repeats],
                    self.total_loss[self.vector_mask_max * repeats], 1, 8, 8, 8)
        if self.soc_version in ['SD3403']:
            if repeats > 0:
                self.tik_instance.vec_dup(self.vector_mask_max, self.total_remain_loss, 0, repeats, 8)
            if mask > 0:
                self.tik_instance.vec_dup(mask, self.total_remain_loss[self.vector_mask_max * repeats], 0, 1, 8)
            remain_loss = self.tik_instance.Tensor(self.calc_precision, (self.loss_each_core,),
                tik.scope_ubuf, 'remain_loss')
            for i in range(0, self.aicore_num):
                self.tik_instance.data_move(remain_loss, self.remain_loss_workspace[self.loss_each_core * i],
                    0, 1, burst_len, 0, 0)
                if repeats > 0:
                    self.tik_instance.vec_add(self.vector_mask_max, self.total_remain_loss,
                        remain_loss, self.total_remain_loss, repeats, 8, 8, 8)
                if mask > 0:
                    self.tik_instance.vec_add(mask, self.total_remain_loss[self.vector_mask_max * repeats],
                        remain_loss[self.vector_mask_max * repeats],
                        self.total_remain_loss[self.vector_mask_max * repeats], 1, 8, 8, 8)

    def _reduce_and_output(self):
        repeats = self.steps_num // self.vector_mask_max
        mask = self.steps_num % self.vector_mask_max

        self._gather_total_loss(repeats, mask)

        # do 8-block align
        if self.soc_version in ['SD3403']:
            fp16_each_block = self.data_each_block
            fp16_8_block = self.vector_mask_max
            fp16_repeat_time = self.step_repeat
        else:
            fp16_each_block = self.block_byte_size // 2
            fp16_8_block = fp16_each_block * 8
            fp16_loss_each_core = (self.loss_each_core + fp16_8_block - 1) // fp16_8_block * fp16_8_block
            total_loss_fp16 = self.tik_instance.Tensor('float16', (fp16_loss_each_core,), tik.scope_ubuf, 'loss_sum')
            fp16_repeat_time = fp16_loss_each_core // fp16_8_block
            if fp16_repeat_time > 0:
                self.tik_instance.vec_dup(fp16_8_block, total_loss_fp16, Constant.SCALAR_MAX_FP16, fp16_repeat_time, 8)
            # conv fp32 to fp16
            data_num_scalar = self.tik_instance.Scalar(self.calc_precision)
            data_num = self.tik_instance.Tensor(self.calc_precision, (self.data_each_block,),
                tik.scope_ubuf, 'data_num')
            loss = self.tik_instance.Tensor(self.calc_precision, (self.loss_each_core,), tik.scope_ubuf, 'loss')
            if self.soc_version in ['Ascend310']:
                data_num_scalar.set_as(1/self.data_num)
                self.tik_instance.vec_dup(self.data_each_block, data_num, data_num_scalar, 1, 8)
                if repeats > 0:
                    self.tik_instance.vmul(self.vector_mask_max, loss, self.total_loss, data_num,
                        repeats, 1, 1, 0, 8, 8, 0)
                    self.tik_instance.vec_conv(self.vector_mask_max, 'none', total_loss_fp16, loss, repeats, 4, 8)
                if mask > 0:
                    self.tik_instance.vmul(
                        mask, loss[self.vector_mask_max * repeats], self.total_loss[self.vector_mask_max * repeats],
                        data_num, 1, 1, 1, 0, 8, 8, 0)
                    self.tik_instance.vec_conv(
                        mask, 'none', total_loss_fp16[self.vector_mask_max * repeats],
                        loss[self.vector_mask_max * repeats], 1, 4, 8)
            else:
                data_num_scalar.set_as(self.data_num)
                self.tik_instance.vec_dup(self.data_each_block, data_num, data_num_scalar, 1, 8)
                if repeats > 0:
                    self.tik_instance.vdiv(self.vector_mask_max, loss, self.total_loss, data_num,
                        repeats, 1, 1, 0, 8, 8, 0)
                    self.tik_instance.vec_conv(self.vector_mask_max, 'none', total_loss_fp16, loss, repeats, 4, 8)
                if mask > 0:
                    self.tik_instance.vdiv(
                        mask, loss[self.vector_mask_max * repeats], self.total_loss[self.vector_mask_max * repeats],
                        data_num, 1, 1, 1, 0, 8, 8, 0)
                    self.tik_instance.vec_conv(
                        mask, 'none', total_loss_fp16[self.vector_mask_max * repeats],
                        loss[self.vector_mask_max * repeats], 1, 4, 8)

        it1_output_count = 2 * fp16_repeat_time
        it2_align_start = ceil(it1_output_count / fp16_each_block) * fp16_each_block
        it2_output_count = ceil(it1_output_count / fp16_8_block) * 2
        it3_align_start = ceil(it2_output_count / fp16_each_block) * fp16_each_block
        it3_output_count = ceil(it2_output_count / fp16_8_block) * 2
        it4_align_start = ceil(it3_output_count / fp16_each_block) * fp16_each_block
        it4_output_count = ceil(it3_output_count / fp16_8_block) * 2
        final_work_tensor_need_size = it2_align_start + it3_align_start + it4_align_start + it4_output_count
        work_tensor_ub = self.tik_instance.Tensor(
            'float16', (final_work_tensor_need_size,), tik.scope_ubuf, 'work_tensor_ub')

        result = self.tik_instance.Tensor('float16', (32,), tik.scope_ubuf, 'result')
        index_scalar = self.tik_instance.Scalar('uint16')
        if self.soc_version in ['SD3403']:
            self.tik_instance.vec_reduce_min(
                128, result, self.total_loss, work_tensor_ub, fp16_repeat_time, 8, cal_index=True)
            min_loss_scalar = self.tik_instance.Scalar('float16')
            min_loss_scalar.set_as(result[0])
            min_loss_tensor = self.tik_instance.Tensor(self.calc_precision, (self.steps_size,),
                tik.scope_ubuf, 'min_loss_tensor')
            self.tik_instance.vec_dup(self.vector_mask_max, min_loss_tensor, min_loss_scalar, self.step_repeat, 8)
            sel_tensor_size = ceil((self.steps_size // 16) / self.vector_mask_max) * self.vector_mask_max
            sel_tensor = self.tik_instance.Tensor('uint16', (sel_tensor_size,), tik.scope_ubuf, 'sel_tensor')
            self.tik_instance.vec_cmpv_eq(sel_tensor, min_loss_tensor, self.total_loss, self.step_repeat, 8, 8)
            fp16_max_tensor = self.tik_instance.Tensor('float16', (self.vector_mask_max,),
                tik.scope_ubuf, 'fp16_max_tensor')
            self.tik_instance.vec_dup(self.vector_mask_max, fp16_max_tensor, Constant.SCALAR_MAX_FP16, 1, 8)
            self.tik_instance.vec_sel(self.vector_mask_max, 2, self.total_remain_loss, sel_tensor,
                self.total_remain_loss, fp16_max_tensor, self.step_repeat, 8, 8, 0)
            self.tik_instance.vec_reduce_min(
                128, result, self.total_remain_loss, work_tensor_ub, fp16_repeat_time, 8, cal_index=True)
            index_scalar.set_as(result[1])
        else:
            self.tik_instance.vec_reduce_min(
                128, result, total_loss_fp16, work_tensor_ub, fp16_repeat_time, 8, cal_index=True)
            index_scalar.set_as(result[1])
            minimal_scalar = self.tik_instance.Scalar('float32')
            minimal_scalar.set_as(self.total_loss[index_scalar])
            minimal = self.tik_instance.Tensor('float32', (self.data_each_block,), tik.scope_ubuf, 'minial_loss')
            self.tik_instance.vec_dup(self.data_each_block, minimal, minimal_scalar, 1, 8)
            if repeats > 0:
                self.tik_instance.vsub(self.vector_mask_max, loss, self.total_loss, minimal, repeats, 1, 1, 0, 8, 8, 0)
                self.tik_instance.vec_conv(self.vector_mask_max, 'none', total_loss_fp16, loss, repeats, 4, 8)
            if mask > 0:
                self.tik_instance.vsub(
                    mask, loss[self.vector_mask_max * repeats], self.total_loss[self.vector_mask_max * repeats],
                    minimal, 1, 1, 1, 0, 8, 8, 0)
                self.tik_instance.vec_conv(
                    mask, 'none', total_loss_fp16[self.vector_mask_max * repeats],
                    loss[self.vector_mask_max * repeats], 1, 4, 8)

            self.tik_instance.vec_reduce_min(
                128, result, total_loss_fp16, work_tensor_ub, fp16_repeat_time, 8, cal_index=True)
            index_scalar.set_as(result[1])

        optimal_scale = self.tik_instance.Scalar(self.calc_precision)
        optimal_scale.set_as(self.scale[index_scalar])

        if self.soc_version in ['Ascend310']:
            scale_tensor_tmp = self.tik_instance.Tensor(self.calc_precision, (self.data_each_block,),
                tik.scope_ubuf, 'scale_tensor_tmp')
            self.tik_instance.vec_dup(1, scale_tensor_tmp, optimal_scale, 1, 8)
            scale_tensor = self.tik_instance.Tensor('float16', (self.data_each_block,),
                tik.scope_ubuf, 'scale_tensor')
            self.tik_instance.vec_conv(self.data_each_block, 'none', scale_tensor, scale_tensor_tmp, 1, 8, 4)
            # set scale to one if too small
            eps_scalar = self.tik_instance.Scalar('float16')
            eps_scalar.set_as(Constant.EPS)
            eps_tensor = self.tik_instance.Tensor('float16', (self.data_each_block,), tik.scope_ubuf, 'eps_scale')
        else:
            scale_tensor = self.tik_instance.Tensor(self.calc_precision, (self.data_each_block,),
                tik.scope_ubuf, 'scale_tensor')
            self.tik_instance.vec_dup(1, scale_tensor, optimal_scale, 1, 8)
            # set scale to one if too small
            eps_scalar = self.tik_instance.Scalar(self.calc_precision)
            eps_scalar.set_as(Constant.EPS)
            eps_tensor = self.tik_instance.Tensor(self.calc_precision, (self.data_each_block,),
                tik.scope_ubuf, 'eps_scale')
        self.tik_instance.vec_dup(1, eps_tensor, eps_scalar, 1, 8)
        cmpmask = self.tik_instance.vcmp_le(self.data_each_block, scale_tensor, eps_tensor, 1, 1)
        eps_scalar.set_as(1)
        self.tik_instance.vec_dup(1, eps_tensor, eps_scalar, 1, 8)
        self.tik_instance.vsel(self.data_each_block, 0, scale_tensor, cmpmask, eps_tensor, scale_tensor,
            1, 1, 1, 1, 8, 8, 8)

        optimal_offset = self.tik_instance.Scalar(self.calc_precision)
        if self.with_offset:
            optimal_offset.set_as(self.offset[index_scalar])
        else:
            optimal_offset.set_as(0.0)
        if self.soc_version in ['Ascend310']:
            offset_tensor_tmp = self.tik_instance.Tensor(self.calc_precision, (8,), tik.scope_ubuf, 'offset_tensor_tmp')
            offset_tensor = self.tik_instance.Tensor('float16', (8,), tik.scope_ubuf, 'optimal_offset')
            self.tik_instance.vec_dup(1, offset_tensor_tmp, optimal_offset, 1, 8)
            self.tik_instance.vec_conv(8, 'none', offset_tensor, offset_tensor_tmp, 1, 8, 4)
        else:
            offset_tensor = self.tik_instance.Tensor(self.calc_precision, (8,), tik.scope_ubuf, 'optimal_offset')
            self.tik_instance.vec_dup(1, offset_tensor, optimal_offset, 1, 8)
        if self.with_offset:
            eps_scalar.set_as(self.clip_min)
            self.tik_instance.vec_dup(1, eps_tensor, eps_scalar, 1, 8)
            self.tik_instance.vsel(8, 0, offset_tensor, cmpmask, eps_tensor, offset_tensor, 1, 1, 1, 1, 8, 8, 8)

        if self.soc_version in ['SD3403', 'Ascend310']:
            scale_tensor_tmp = self.tik_instance.Tensor('float32', (8,), tik.scope_ubuf, 'scale_tensor_tmp')
            self.tik_instance.vec_conv(8, 'none', scale_tensor_tmp, scale_tensor, 1, 8, 4)
            offset_tensor_tmp = self.tik_instance.Tensor('float32', (8,), tik.scope_ubuf, 'offset_tensor_tmp')
            self.tik_instance.vec_conv(8, 'none', offset_tensor_tmp, offset_tensor, 1, 8, 4)
            self.tik_instance.tensor_mov(self.output_scale, scale_tensor_tmp, '', 1, 1, 0, 0)
            self.tik_instance.tensor_mov(self.output_offset, offset_tensor_tmp, '', 1, 1, 0, 0)
        else:
            self.tik_instance.tensor_mov(self.output_scale, scale_tensor, '', 1, 1, 0, 0)
            self.tik_instance.tensor_mov(self.output_offset, offset_tensor, '', 1, 1, 0, 0)


@para_check.check_op_params(
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_ATTR_FLOAT,
    para_check.REQUIRED_ATTR_FLOAT,
    para_check.REQUIRED_ATTR_LIST_FLOAT,
    para_check.REQUIRED_ATTR_FLOAT,
    para_check.REQUIRED_ATTR_BOOL,
    para_check.OPTION_ATTR_INT,
    para_check.KERNEL_NAME)
def ifmr(
        data,
        data_min,
        data_max,
        cumsum,
        scale,
        offset,
        min_percentile,
        max_percentile,
        search_range,
        search_step,
        with_offset,
        quant_bits=8,
        kernel_name='ifmr'):
    """
    IFMR op
    """
    ifmr_inst = Reconstruction(
        data,
        data_min,
        data_max,
        cumsum,
        scale,
        offset,
        min_percentile,
        max_percentile,
        search_range,
        search_step,
        with_offset,
        quant_bits,
        kernel_name)

    return ifmr_inst.ifmr_compute()
