#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2020 Huawei Technologies Co., Ltd
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
dropout_v2
"""
import functools
import math
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform as cce


# 'pylint: disable=too-few-public-methods,not-use-list-comprehension
class Constant:
    """
    The class for constant
    """
    PRIME_NUM = 509.0 # Prime number for LCG calculation
    MAX = 1023.0 # Maximum period of LCG
    BIAS = math.sqrt(2)


def _ceil_div(value, factor):
    return (value + factor - 1) // factor


# 'pylint: disable=invalid-name,useless-object-inheritance,too-few-public-methods
# 'pylint: disable=too-many-instance-attributes,too-many-arguments,unused-argument,self-assigning-variable
# 'pylint: disable=attribute-defined-outside-init
def _check_param_dtype(x, seed, y, mask, new_seed):
    if x.get("dtype") != y.get("dtype"):
        raise RuntimeError('dtype of input and output should be same')
    if x.get("dtype") != "float32" and x.get("dtype") != "float16":
        raise RuntimeError('dtype of x should be float32 or float16')
    if seed.get("dtype") != "float32" or mask.get("dtype") != "float32" or new_seed.get("dtype") != "float32":
        raise RuntimeError('dtype of seed, mask and new_seed should be float32')


def _check_param_shape(x, seed, y, mask, new_seed):
    if x.get("shape") != y.get("shape") or x.get("shape") != mask.get("shape"):
        raise RuntimeError('shape of x, y and mask should be same')
    if seed.get("shape") != new_seed.get("shape"):
        raise RuntimeError('shape of seed and new_seed should be same')


class DtypeConfig(object):
    """
    ub data config
    """
    fp16 = {'bytes_size': 2, 'vector_mask_max': 128}
    fp32 = {'bytes_size': 4, 'vector_mask_max': 64}


class DropoutV2(object):
    """
    Define dropout calculation process
    """
    # 'pylint: disable=too-many-arguments
    def __init__(self, x, seed, y, mask, new_seed, p, kernel_name="dropout_v2"):
        """
        Use the lcg algorithm to generate random numbers and implement dropout calculations
        :param x: input data
        :param seed:  Pre-generated seed
        :param y: dropout output result
        :param mask: the generated mask data
        :param new_seed: the updated seed data
        :param p: data drop probability
        :param kernel_name: default=dropout_v2
        """
        self.x_shape = x.get("shape")
        self.x_dtype = x.get("dtype")
        self.seed_shape = seed.get("shape")
        self.seed_dtype = seed.get("dtype")
        self.y_shape = y.get("shape")
        self.y_dtype = y.get("dtype")

        self.kernel_name = kernel_name
        self.prob = p
        _check_param_dtype(x, seed, y, mask, new_seed)
        _check_param_shape(x, seed, y, mask, new_seed)

        self.tik_instance = tik.Tik(tik.Dprofile("v100", "cloud"))
        self.aicore_use = tik.Dprofile().get_aicore_num()

        block_bite_size = tik.Dprofile().get_unified_buffer_size() - 16 * 1024

        self.bytes_dtype_x = cce.get_bit_len(self.x_dtype) // 8
        self.block_num_x = 32 // self.bytes_dtype_x

        self.bytes_dtype_seed = cce.get_bit_len(self.seed_dtype) // 8
        self.block_num_seed = 32 // self.bytes_dtype_seed

        self.vector_mask_max_fp16 = DtypeConfig.fp16['vector_mask_max']
        self.vector_mask_max_fp32 = DtypeConfig.fp32['vector_mask_max']

        process_data_len = functools.reduce(lambda x, y: x * y, self.x_shape)  # input data len
        self.core_use_select(process_data_len)

        self.x_ub_size = block_bite_size // 5 // self.bytes_dtype_seed

        self.x_gm = self.tik_instance.Tensor(self.x_dtype, self.x_shape, tik.scope_gm, "x_gm")
        self.seed_gm = self.tik_instance.Tensor(self.seed_dtype, self.seed_shape, tik.scope_gm, "seed_gm")
        self.y_gm = self.tik_instance.Tensor(self.y_dtype, self.y_shape, tik.scope_gm, "y_gm")
        self.mask_gm = self.tik_instance.Tensor(self.seed_dtype, self.y_shape, tik.scope_gm, "mask_gm")
        self.new_seed_gm = self.tik_instance.Tensor(self.seed_dtype, self.seed_shape, tik.scope_gm, "new_seed_gm")
        self.seed_ub = None
        self.seed_mask_ub = None
        self.seed_drop_ub = None
        self.seed_tmp_int = None
        self.seed_drop_ub_fp16 = None
        self.x_ub = None
        self.seed_tmp_uint = None
        self.tmp_one_ub = None

    def core_use_select(self, data_len):
        """
        Calculate the number of cores used
        :param data_len: Length of data to be processed
        :return:
        """
        self.data_num_each_core = (data_len + self.aicore_use - 1) // self.aicore_use
        self.data_num_each_core = (self.data_num_each_core + self.block_num_x - 1) // (self.block_num_x) * \
                                  (self.block_num_x)
        self.aicore_use = (data_len + self.data_num_each_core - 1) // self.data_num_each_core
        self.data_num_last_core = data_len - self.data_num_each_core * (self.aicore_use - 1)

    def dropout_compute(self):
        """
        Sub-core calculation
        :return:
        """
        all_aicore = tik.Dprofile().get_aicore_num()
        with self.tik_instance.for_range(0, all_aicore, block_num=all_aicore) as index:
            move_offset = index * self.data_num_each_core
            seed_offset = index * self.x_ub_size
            self.seed_ub = self.tik_instance.Tensor(self.seed_dtype, (self.x_ub_size,), name="seed_ub",
                                                    scope=tik.scope_ubuf)
            self.tik_instance.data_move(self.seed_ub, self.seed_gm[seed_offset], 0, 1,
                                        _ceil_div(self.x_ub_size, self.block_num_seed), 0, 0)
            with self.tik_instance.if_scope(index < self.aicore_use - 1):
                self._dropout_compute_each_core(move_offset, self.data_num_each_core)
            with self.tik_instance.else_scope():
                self._dropout_compute_each_core(move_offset, self.data_num_last_core, True)

            seed_offset_ = index * self.x_ub_size
            seed_out_offset = 8

            self.tik_instance.data_move(self.new_seed_gm[seed_offset_], self.seed_ub[seed_out_offset], 0, 1,
                                        _ceil_div((self.x_ub_size - 8), self.block_num_seed), 0, 0)
            self.tik_instance.data_move(self.new_seed_gm[((index + 1) * self.x_ub_size) - seed_out_offset],
                                        self.seed_ub, 0, 1, _ceil_div(8, self.block_num_seed), 0, 0)

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[self.x_gm, self.seed_gm],
                                   outputs=[self.y_gm, self.mask_gm, self.new_seed_gm])
        return self.tik_instance

    def _dropout_compute_each_core(self, data_offset, process_len, is_last_core=False):
        """
        :param data_offset: x_gm offset
        :param process_len: the length of the data calculated this time. It is aligned to 8 when data in fp32 or aligned
                            to 16 when data in fp16, except for the last core = True
        :param is_last_core: when it is true, should process tail data
        :return:
        """
        ub_avg_avail = self.x_ub_size
        process_times = process_len // ub_avg_avail
        process_remain = process_len % ub_avg_avail
        # always align
        if process_times > 0:
            with self.tik_instance.for_range(0, process_times) as ps_times:
                self._compute_each_times(data_offset + ps_times * ub_avg_avail, ub_avg_avail, is_last_core)

        if process_remain > 0:
            self._compute_each_times(data_offset + process_times * ub_avg_avail, process_remain, is_last_core)

    def _compute_each_times(self, data_offset, process_len, is_last_core=False):
        """
        :param data_offset: relative address offset
        :param process_len: the length of the data calculated this time. It is aligned to 8 when data in fp32 or aligned
                            to 16 when data in fp16, except for the last core = True
        :param is_last_core: when the value is true, the tail block data needs to be processed
        :return:
        """
        process_len_align = process_len - process_len % (self.block_num_x)
        if process_len_align > 0:
            self.seed_mask_ub = self.tik_instance.Tensor(self.seed_dtype, (self.x_ub_size,), name="seed_mask_ub",
                                                         scope=tik.scope_ubuf)
            self.seed_drop_ub = self.tik_instance.Tensor(self.seed_dtype, (self.x_ub_size,), name="seed_drop_ub",
                                                         scope=tik.scope_ubuf)
            self.seed_tmp_int = self.tik_instance.Tensor("int32", (self.x_ub_size,), name="seed_tmp_int",
                                                         scope=tik.scope_ubuf)
            self._compute_each_thread(data_offset, process_len_align)

        if process_len % (self.block_num_x) > 0 and is_last_core:
            tail_data = _ceil_div(process_len % (self.block_num_x),
                                  self.block_num_x) * self.block_num_x
            if self.x_dtype == "float16":
                self.seed_drop_ub_fp16 = self.tik_instance.Tensor(self.x_dtype, (64,), name="seed_drop_ub_fp16",
                                                                  scope=tik.scope_ubuf)

            self.seed_mask_ub = self.tik_instance.Tensor(self.seed_dtype, (64,), name="seed_mask_ub",
                                                         scope=tik.scope_ubuf)
            self.seed_drop_ub = self.tik_instance.Tensor(self.seed_dtype, (64,), name="seed_drop_ub",
                                                         scope=tik.scope_ubuf)
            self.seed_tmp_int = self.tik_instance.Tensor("int32", (64,), name="seed_tmp_int",
                                                         scope=tik.scope_ubuf)
            tail_offset = data_offset + process_len - tail_data
            with self.tik_instance.if_scope(tail_offset < 0):
                tail_offset = 0
            self._compute_each_thread(data_offset + tail_offset, tail_data)

    def _compute_each_thread(self, data_offset, process_len):
        """
        :param data_offset: offset of data in input x and output y
        :param process_len: process data len of each time
        :return:
        """
        repeat_time = 255
        repeat_time = process_len % (repeat_time * 64) // 64
        left_num = process_len % self.vector_mask_max_fp32
        if repeat_time > 0:
            offset = 0
            self._update_mask(self.vector_mask_max_fp32, repeat_time, offset)
            self._gen_mask(self.vector_mask_max_fp32, repeat_time, offset)

        if left_num > 0:
            offset = process_len // self.vector_mask_max_fp32 * self.vector_mask_max_fp32
            self._update_mask(left_num, 1, offset)
            self._gen_mask(left_num, 1, offset)

        with self.tik_instance.new_stmt_scope():
            if self.x_dtype == "float16":
                self.seed_drop_ub_fp16 = self.tik_instance.Tensor(self.x_dtype, (self.x_ub_size,),
                                                                  name="seed_drop_ub_fp16", scope=tik.scope_ubuf)
            self.x_ub = self.tik_instance.Tensor(self.x_dtype, (self.x_ub_size,), name="x_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(self.x_ub, self.x_gm[data_offset], 0, 1,
                                        _ceil_div(process_len, self.block_num_x), 0, 0)
            if repeat_time > 0:
                offset = 0
                self._gen_dropout_result(self.vector_mask_max_fp32, repeat_time, offset)
            if left_num > 0:
                offset = process_len // self.vector_mask_max_fp32 * self.vector_mask_max_fp32
                self._gen_dropout_result(left_num, 1, offset)

            if self.x_dtype == "float16":
                self.tik_instance.data_move(self.y_gm[data_offset], self.seed_drop_ub_fp16, 0, 1,
                                            _ceil_div(process_len, self.block_num_x), 0, 0)
            else:
                self.tik_instance.data_move(self.y_gm[data_offset], self.seed_drop_ub, 0, 1,
                                            _ceil_div(process_len, self.block_num_x), 0, 0)
        self.tik_instance.data_move(self.mask_gm[data_offset], self.seed_mask_ub, 0, 1,
                                    _ceil_div(process_len, self.block_num_seed), 0, 0)

    def _update_mask(self, mask, repeat_time, offset):
        """
        Update seed data
        :param mask: instruction calculation length
        :param repeat_time: instruction repeat time value
        :param offset: relative address offset
        :return:
        """
        repeat_time = int(repeat_time)
        a = self.tik_instance.Scalar(dtype="float32", init_value=Constant.PRIME_NUM)
        bias = self.tik_instance.Scalar(dtype="float32", init_value=Constant.BIAS)
        m = self.tik_instance.Scalar(dtype="float32", init_value=Constant.PRIME_NUM / Constant.MAX)
        self.tik_instance.vec_muls(mask, self.seed_drop_ub[offset], self.seed_ub[offset], m, repeat_time, 8, 8)
        self.tik_instance.vec_muls(mask, self.seed_ub[offset], self.seed_ub[offset], a, repeat_time, 8, 8)
        self.tik_instance.vec_adds(mask, self.seed_ub[offset], self.seed_ub[offset], bias, repeat_time, 8, 8)
        self.tik_instance.vec_conv(mask, 'floor', self.seed_tmp_int[offset], self.seed_drop_ub[offset], repeat_time,
                                   8, 8)
        self.tik_instance.vec_conv(mask, "", self.seed_drop_ub[offset], self.seed_tmp_int[offset], repeat_time, 8,
                                   8)
        self.tik_instance.vec_muls(mask, self.seed_drop_ub[offset], self.seed_drop_ub[offset], Constant.MAX,
                                   repeat_time, 8, 8)
        self.tik_instance.vec_sub(mask, self.seed_ub[offset], self.seed_ub[offset], self.seed_drop_ub[offset],
                                  repeat_time, 8, 8, 8)
        self.tik_instance.vec_abs(mask, self.seed_ub[offset], self.seed_ub[offset], repeat_time, 8, 8)

    def _gen_mask(self, mask, repeat_time, offset):
        """
        Calculate the current mask data
        :param mask: instruction calculation length
        :param repeat_time: instruction repeat time value
        :param offset: relative address offset
        :return:
        """
        threshold = Constant.MAX * self.prob
        self.seed_tmp_uint = self.seed_tmp_int.reinterpret_cast_to("uint64")
        self.tik_instance.vec_dup(mask, self.seed_drop_ub[offset], threshold, repeat_time, 8)
        self.tik_instance.vec_dup(mask, self.seed_mask_ub[offset], 1, repeat_time, 8)
        with self.tik_instance.new_stmt_scope():
            self.tmp_one_ub = self.tik_instance.Tensor(self.seed_dtype, (self.x_ub_size,), name="tmp_one_ub",
                                                       scope=tik.scope_ubuf)
            self.tik_instance.vec_dup(mask, self.tmp_one_ub[offset], 0, repeat_time, 8)
            with self.tik_instance.for_range(0, repeat_time) as vsel_time:
                self.tik_instance.vec_cmpv_le(self.seed_tmp_uint, self.seed_drop_ub[offset + mask * vsel_time],
                                              self.seed_ub[offset + mask * vsel_time], 1, 8, 8)
                self.tik_instance.vec_sel(mask, 0, self.seed_mask_ub[offset + mask * vsel_time], self.seed_tmp_uint,
                                          self.seed_mask_ub[offset + mask * vsel_time],
                                          self.tmp_one_ub[offset + mask * vsel_time], 1, 8, 8, 8)

    def _gen_dropout_result(self, mask, repeat_time, offset):
        """
        Use 0,1 tensor from seed_mask_ub to compute result
        :param mask: instruction calculation length
        :param repeat_time: instruction repeat time value
        :param offset: relative address offset
        :return:
        """
        if self.x_dtype == "float16":
            self.tik_instance.vec_conv(mask, "", self.seed_drop_ub_fp16[offset], self.seed_mask_ub[offset],
                                       repeat_time, 4, 8)
            self.tik_instance.vec_mul(mask, self.seed_drop_ub_fp16[offset], self.seed_drop_ub_fp16[offset], self.x_ub,
                                      repeat_time, 4, 4, 4)
            self.tik_instance.vec_muls(mask, self.seed_drop_ub_fp16[offset], self.seed_drop_ub_fp16[offset],
                                       1 / (1 - self.prob), repeat_time, 4, 4)
        else:
            self.tik_instance.vec_mul(mask, self.seed_drop_ub[offset], self.seed_mask_ub[offset], self.x_ub,
                                      repeat_time, 8, 8, 8)
            self.tik_instance.vec_muls(mask, self.seed_drop_ub[offset], self.seed_drop_ub[offset],
                                       1 / (1 - self.prob), repeat_time, 8, 8)


# 'pylint: disable=too-many-arguments
@register_operator_compute("dropout_v2", op_mode="static", support_fusion=True)
def dropout_v2(x, seed, y, mask, new_seed, p, kernel_name='dropout_v2'):
    """
    call dropout
    :param x: input data
    :param seed:  Pre-generated seed
    :param y: dropout output result
    :param mask: the generated mask data
    :param new_seed: the updated seed data
    :param p: data drop probability
    :param kernel_name: default=dropout_v2
    :return:
    """
    dropout = DropoutV2(x, seed, y, mask, new_seed, p, kernel_name)
    dropout.dropout_compute()
