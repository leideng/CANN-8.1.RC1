#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
embedding_dense_grad
"""

import math
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform as cce
from impl.util.platform_adapter import OpImplMode
from impl.util.platform_adapter import register_operator
from impl.util.util_common import check_op_impl_mode


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    RESERVE_SIZE = 16 * 1024
    BLOCK = 8
    HIGH_PERFORMANCE_NUM = 32
    MASK = 128
    THRESHOLD_NUM = 10240
    MID_DTYPE = 'float32'
    FLOAT32_BYTES = 4
    BLOCK_BYTES = 32


class EmbeddingDenseGrad:
    """
    Function: store EmbeddingDenseGrad parameters  and compute EmbeddingDenseGrad
    """

    # 'pylint: disable=unused-argument, too-many-statements, disable=too-many-arguments
    def __init__(
            self,
            grad,
            indices,
            y,
            num_weights,
            padding_idx,
            scale_grad_by_freq,
            kernel_name,
            impl_mode):
        """
        init the ShuffleChannel parameters

        Parameters
        ----------
            input_dict: input_dict is a dict, the keys as follow:
                grad: dict,shape and datatype,datatype supports float32
                indices: dict,shape and datatype,datatype supports int32
                y:dict,shape and datatype,datatype supports float32
                num_weights:the number of words in dict
                padding_idx:judge grad_weight of which word is zero
                scale_grad_by_freq: judge whether or not  scale_grad
                kernel_name: cce kernel name, default value is "embedding_dense_grad"
        Returns
        -------
        None
        """
        self.impl_mode = impl_mode
        self.grad_shape = grad["shape"]
        self.dtype_grad = grad.get("dtype")
        self.indices_shape = indices["shape"]
        self.dtype_indices = indices.get("dtype")
        self.embedding_dim = grad["shape"][-1]
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.padding_idx = padding_idx
        self.num_weights = num_weights
        self.kernel_name = kernel_name
        self.scale_grad_by_freq = scale_grad_by_freq
        self.block_ub = None
        self.align_ub = None
        self.add_y_ub = None
        self.task_num = None
        self.task_num_last_core = None

        '''Data reading and writing on UB must be 32B aligned. This parameter is used
        to calculate tensor division and data handling instruction parameters
        '''
        block_bite_size = 32
        # Get the size of UB space in Bytes
        self.ub_size_bytes = cce.get_soc_spec(cce.UB_SIZE)
        self.aicore_num = cce.get_soc_spec(cce.CORE_NUM)

        '''Calculate how many grad elements can be stored in a block according to
        the input data type
        '''
        self.dtype_bytes_size_grad = cce.get_bit_len(self.dtype_grad) // Constant.BLOCK
        self.grad_each_block = block_bite_size // self.dtype_bytes_size_grad
        '''Calculate how many counts elements can be stored in a block according
        to the input data type
        '''
        self.dtype_bytes_size_counts = cce.get_bit_len(self.dtype_indices) // Constant.BLOCK
        self.counts_each_block = block_bite_size // self.dtype_bytes_size_counts

        '''Calculate how many indicators elements can be stored in a block
        according to the input data type
        '''
        self.dtype_bytes_size_indices = cce.get_bit_len(self.dtype_indices) // Constant.BLOCK
        self.indices_each_block = block_bite_size // self.dtype_bytes_size_indices
        self.high_precision = False
        '''The vector instruction calculates a maximum of 8 blocks per repeat.
        This parameter is the maximum value of the mask when grad performs vector calculation
        '''
        self.vector_mask_max_counts = Constant.BLOCK * self.counts_each_block
        self.new_numel_indices = None
        self.grad_ub = None
        self.indices_ub = None
        self.grad = None
        self.grad_fp32 = None
        self.grad_ub_fp32 = None
        self.grad_fp16_or_bf16 = None
        self.fp32_workspace = None
        self.core_used_cast = None
        self.begin = None
        self.k = None
        self.scale_float = None
        self.scale_float_tmp = None
        self.indices = None
        self.counts_ub = None
        self.grad_weight = None
        self.numel_indices = None
        self.numel_grad = None
        self.add_tensor = None
        self.index = None
        self.vector_mask_max_grad = Constant.BLOCK * self.grad_each_block
        self.ranges = None
        self.ele_not_last_core = None
        self.ele_last_core = None
        self.used_core = None
        self.index_not_last_core = None
        self.ub_grad_size = None
        self.end = None
        self.scale_int = None
        self.ub_indices_size = None
        self.counts_size = None
        self.cast_task_num = None
        self.task_num_not_last_core = None
        self.task_num_last_core = None
        self.ub_cast_size = None
        self.sync_workspace = None
        # the dividing threshold of the calculation mode with the embedding_dim
        self.count = 1000000
        if self.impl_mode == OpImplMode.HIGH_PRECISION and not self.scale_grad_by_freq and \
                (self.dtype_grad == "float16" or self.dtype_grad == "bfloat16"):
            self.high_precision = True
            self.dtype_bytes_size_grad += Constant.FLOAT32_BYTES
            self.vector_mask_max_grad = Constant.BLOCK * (Constant.BLOCK_BYTES // Constant.FLOAT32_BYTES)

    @staticmethod
    def get_dtype_size(dtype):
        """
        get byte size of dtype
        """
        dtype_dict = {"int64": 8, "float32": 4, "uint8": 1, "int32": 4,
                      "float16": 2, "bfloat16": 2}
        return dtype_dict.get(dtype)

    def embedding_dense_grad_compute(self):
        """
        compute embedding_dense_grad

        Parameters
        ----------
        None
        Returns
        -------
        tik_instance: tik_instance
        """
        self.element_of_grad_and_indices()
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.grad, self.indices], outputs=[self.grad_weight])
        return self.tik_instance

    def element_of_grad_and_indices(self):
        """
        Count the number of elements of indicators and grad

        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        self.numel_grad = 1
        self.numel_indices = 1
        for y in self.grad_shape:
            self.numel_grad *= y
        for x in self.indices_shape:
            self.numel_indices *= x
        self.new_numel_indices = math.ceil(
            self.numel_indices / self.dtype_bytes_size_indices) * self.dtype_bytes_size_indices
        if self.numel_indices // self.aicore_num * self.embedding_dim < self.grad_each_block:
            self.aicore_num = 1
        self.gm_for_data_and_fill_grad_weight()

    def cal_core_num(self, dtype_bytes_size_per_grad):
        """
        cal_core_num
        """
        if self.embedding_dim > self.grad_each_block and self.embedding_dim % self.grad_each_block != 0:
            scale = ((self.embedding_dim - 1) // self.grad_each_block + 1) * self.grad_each_block
        elif self.numel_indices >= Constant.THRESHOLD_NUM and self.embedding_dim <= self.grad_each_block and \
                            self.num_weights <= Constant.HIGH_PERFORMANCE_NUM:
            scale = self.grad_each_block
        else:
            scale = self.embedding_dim
        if self.scale_grad_by_freq:
            self.index_not_last_core = (self.num_weights - 1) // self.aicore_num + 1
            self.used_core = (self.num_weights - 1) // self.index_not_last_core + 1
            self.counts_size = self.num_weights // self.aicore_num + \
                               self.num_weights % self.aicore_num
            self.ub_indices_size = (self.ub_size_bytes - self.counts_size *
                                    self.dtype_bytes_size_counts - Constant.RESERVE_SIZE) \
                                   // (self.embedding_dim * dtype_bytes_size_per_grad +
                                       self.dtype_bytes_size_indices) \
                                   // self.indices_each_block * self.indices_each_block
        else:
            self.ele_not_last_core = (self.numel_indices - 1) // self.aicore_num + 1
            self.used_core = (self.numel_indices - 1) // self.ele_not_last_core + 1
            self.ele_last_core = self.numel_indices - (self.used_core - 1) * self.ele_not_last_core
            self.ub_indices_size = (self.ub_size_bytes - Constant.RESERVE_SIZE) \
                                       // (scale * dtype_bytes_size_per_grad +
                                           self.dtype_bytes_size_indices) \
                                       // self.indices_each_block * self.indices_each_block
        self.ub_grad_size = self.ub_indices_size * scale
        if self.ub_indices_size == 0:
            self.ub_grad_size = self.embedding_dim
            remain_num = self.ub_size_bytes - Constant.RESERVE_SIZE - \
                         self.ub_grad_size * dtype_bytes_size_per_grad
            self.ub_indices_size = remain_num // self.dtype_bytes_size_indices \
                                   // self.indices_each_block * self.indices_each_block
            self.count = ((self.ub_size_bytes - Constant.RESERVE_SIZE) \
                                // (self.indices_each_block * dtype_bytes_size_per_grad) \
                                - self.dtype_bytes_size_indices // dtype_bytes_size_per_grad) \
                         // self.grad_each_block * self.grad_each_block

    def dup_value(self, dst, num, dup_value=0, offset=0):
        """
        dup value to ub
        """
        dtype_byte_size = EmbeddingDenseGrad.get_dtype_size(dst.dtype)
        if dst.dtype == "bfloat16":
            dst = dst.reinterpret_cast_to("float16")
            mask = 256 // dtype_byte_size
            stride = 8
            loop = num // (mask * 255)
            if loop > 0:
                with self.tik_instance.for_range(0, loop) as index:
                    tmp_offset = offset + index * mask * 255
                    self.tik_instance.vec_dup(mask, dst[tmp_offset], dup_value, 255, stride)
                offset += loop * mask * 255

            repeat_time = (num % (mask * 255)) // mask
            if repeat_time > 0:
                self.tik_instance.vec_dup(mask, dst[offset], dup_value, repeat_time, stride)
                offset += repeat_time * mask
            last_num = num % mask
            if last_num > 0:
                self.tik_instance.vec_dup(last_num, dst[offset], dup_value, 1, stride)
            dst = dst.reinterpret_cast_to("bfloat16")
        else:
            mask = 256 // dtype_byte_size
            stride = 8
            loop = num // (mask * 255)
            if loop > 0:
                with self.tik_instance.for_range(0, loop) as index:
                    tmp_offset = offset + index * mask * 255
                    self.tik_instance.vec_dup(mask, dst[tmp_offset], dup_value, 255, stride)
                offset += loop * mask * 255

            repeat_time = (num % (mask * 255)) // mask
            if repeat_time > 0:
                self.tik_instance.vec_dup(mask, dst[offset], dup_value, repeat_time, stride)
                offset += repeat_time * mask
            last_num = num % mask
            if last_num > 0:
                self.tik_instance.vec_dup(last_num, dst[offset], dup_value, 1, stride)

    def gm_for_data_and_fill_grad_weight(self):
        """
        Allocate space for grad, indices and grad_weight on gm
        use 0 to fill grad_weight
        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        # Allocate space for grad, indices and grad_weight on gm
        self.indices = self.tik_instance.Tensor(self.dtype_indices, (self.new_numel_indices,), name="indices",
                                                scope=tik.scope_gm)
        if self.high_precision:
            self.fp32_workspace = self.tik_instance.Tensor('float32', (self.num_weights, self.embedding_dim),
                                                           scope=tik.scope_gm,
                                                           name='fp32_workspace',
                                                           is_workspace=True,
                                                           is_atomic_add=True)
            self.sync_workspace = self.tik_instance.Tensor("int64", (4 * self.aicore_num,), tik.scope_gm,
                                                           name='sync_workspace',
                                                           is_workspace=True,
                                                           is_atomic_add=True)
        self.grad_weight = self.tik_instance.Tensor(self.dtype_grad, (self.num_weights, self.embedding_dim),
                                                    name="grad_weight", scope=tik.scope_gm,
                                                    is_atomic_add=(not self.high_precision))
        self.grad = self.tik_instance.Tensor(self.dtype_grad, self.grad_shape, name="grad", scope=tik.scope_gm)
        # Create a new space to initialize grad_weight
        self.cal_core_num(self.dtype_bytes_size_grad)
        if self.high_precision:
            self.cal_cast_task_num()
        with self.tik_instance.for_range(0, self.aicore_num, block_num=self.aicore_num) as core_index:
            with self.tik_instance.if_scope(core_index < self.used_core):
                if self.scale_grad_by_freq:
                    self.begin = core_index * self.index_not_last_core
                    with self.tik_instance.if_scope(core_index == self.aicore_num - 1):
                        self.end = self.num_weights
                    with self.tik_instance.else_scope():
                        self.end = (core_index + 1) * self.index_not_last_core
                with self.tik_instance.new_stmt_scope():
                    self.ub_for_data(core_index)
            if self.high_precision:
                self.tik_instance.block_barrier(self.sync_workspace)
                with self.tik_instance.if_scope(core_index < self.core_used_cast):
                    self.fp32_cast_to(self.dtype_grad, core_index)

    def fp32_cast_to(self, dst_dtype, core_index):
        self.alloc_ub()
        self.task_num = self.task_num_not_last_core
        repeat_time = self.task_num // self.ub_cast_size
        with self.tik_instance.if_scope(core_index != (self.core_used_cast - 1)):
            for repeat_index in range(repeat_time):
                self.tik_instance.data_move(self.grad_fp32,
                                            self.fp32_workspace[core_index * self.task_num_not_last_core \
                                            + repeat_index * self.ub_cast_size],
                                            0, 1, self.ub_cast_size // 8, 0, 0)
                self.conv_to_bf16_or_fp16(self.grad_fp32, self.grad_fp16_or_bf16, self.ub_cast_size)
                self.tik_instance.data_move(self.grad_weight[core_index * self.task_num_not_last_core \
                                            + repeat_index * self.ub_cast_size],
                                            self.grad_fp16_or_bf16,
                                            0, 1, self.ub_cast_size // 16, 0, 0)
        with self.tik_instance.else_scope():
            self.task_num = self.task_num_last_core
            repeat_time = (self.task_num - 1) // self.ub_cast_size + 1
            task_num_every_repeat = self.ub_cast_size
            for repeat_index in range(repeat_time):
                if repeat_index == (repeat_time - 1):
                    task_num_every_repeat = self.cast_task_num - \
                        (core_index * self.task_num_not_last_core + repeat_index * self.ub_cast_size)
                self.tik_instance.data_move(self.grad_fp32,
                                            self.fp32_workspace[core_index * self.task_num_not_last_core \
                                                                + repeat_index * self.ub_cast_size],
                                            0, 1,
                                            (task_num_every_repeat - 1) \
                                            // (Constant.BLOCK_BYTES // Constant.FLOAT32_BYTES) + 1,
                                            0, 0)
                self.conv_to_bf16_or_fp16(self.grad_fp32, self.grad_fp16_or_bf16, task_num_every_repeat)
                self.tik_instance.data_move(self.grad_weight[core_index * self.task_num_not_last_core \
                                            + repeat_index * self.ub_cast_size],
                                            self.grad_fp16_or_bf16,
                                            0, 1, (task_num_every_repeat - 1) // 16 + 1, 0, 0)

    def data_move_to_gm(self, core_index, repeat_index, task_num_every_repeat):
        self.tik_instance.data_move(self.grad_weight[core_index * self.task_num_not_last_core \
                                                     + repeat_index * self.ub_cast_size],
                                    self.grad_fp16_or_bf16,
                                    0, 1,
                                    (task_num_every_repeat - 1) \
                                    // (Constant.BLOCK_BYTES // Constant.FLOAT32_BYTES) + 1,
                                    0, 0)

    def cal_cast_task_num(self):
        self.cast_task_num = self.num_weights * self.embedding_dim
        self.ub_cast_size = (self.ub_size_bytes - Constant.RESERVE_SIZE) \
                                // self.dtype_bytes_size_grad \
                                // self.vector_mask_max_grad * self.vector_mask_max_grad
        self.task_num_not_last_core = ((((self.cast_task_num - 1) // self.aicore_num + 1) - 1) \
                                        // self.ub_cast_size + 1) * self.ub_cast_size
        self.core_used_cast = (self.cast_task_num - 1) // self.task_num_not_last_core + 1
        self.task_num_last_core = self.cast_task_num - self.task_num_not_last_core * (self.core_used_cast - 1)

    def alloc_ub(self):
        # Allocate space for grad_fp32, grad_fp16
        self.grad_fp32 = self.tik_instance.Tensor("float32", (self.ub_cast_size,), name="grad_fp32",
                                                   scope=tik.scope_ubuf)
        self.grad_fp16_or_bf16 = self.tik_instance.Tensor(self.dtype_grad, (self.ub_cast_size,),
                                                          name="grad_fp16_or_bf16", scope=tik.scope_ubuf)

    def conv_fp32(self, src, dst, num):
        """
        conv_fp32
        """
        offset = 0
        mask = 64
        loop = num // (mask * 255)
        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as index:
                tmp_offset = offset + index * mask * 255
                self.tik_instance.vec_conv(mask,
                                            'none',
                                            dst[tmp_offset], 
                                            src[tmp_offset], 
                                            255, 
                                            8, 
                                            4)
            offset += loop * mask * 255
        
        repeat_time = (num % (mask * 255)) // mask
        with self.tik_instance.if_scope(repeat_time > 0):
            self.tik_instance.vec_conv(mask,
                                        'none',
                                        dst[offset], 
                                        src[offset], 
                                        repeat_time, 
                                        8, 
                                        4)
            offset += repeat_time * mask
        last_num = num % mask
        with self.tik_instance.if_scope(last_num > 0):
            self.tik_instance.vec_conv(last_num,
                                        'none',
                                        dst[offset], 
                                        src[offset], 
                                        1, 
                                        0, 
                                        0)

    def conv_to_bf16_or_fp16(self, src, dst, num, offset=0):
        """
        conv_to_bf16_or_fp16
        """
        mask = 64
        loop = num // (mask * 255)
        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as index:
                tmp_offset = offset + index * mask * 255
                self.tik_instance.vec_conv(mask,
                                            'round',
                                            dst[tmp_offset], 
                                            src[tmp_offset], 
                                            255, 
                                            4, 
                                            8)
            offset += loop * mask * 255
        repeat_time = (num % (mask * 255)) // mask
        with self.tik_instance.if_scope(repeat_time > 0):
            self.tik_instance.vec_conv(mask,
                                        'round',
                                        dst[offset], 
                                        src[offset], 
                                        repeat_time, 
                                        4, 
                                        8)
            offset += repeat_time * mask
        last_num = num % mask
        with self.tik_instance.if_scope(last_num > 0):
            self.tik_instance.vec_conv(last_num,
                                        'round',
                                        dst[offset], 
                                        src[offset], 
                                        1, 
                                        0, 
                                        0)

    def ub_for_data(self, core_index):
        """
        Allocate space for grad, indices and counts on ub
        use 0 to fill counts

        Parameters
        ----------
        core_index: the index of aicore
        Returns
        -------
        None
        """
        # Allocate space for grad, indices and counts on ub
        self.indices_ub = self.tik_instance.Tensor(self.dtype_indices, (self.ub_indices_size,), name="indices_ub",
                                                   scope=tik.scope_ubuf)
        if self.scale_grad_by_freq:
            self.counts_ub = self.tik_instance.Tensor(self.dtype_indices, (self.counts_size,), name="counts_ub",
                                                      scope=tik.scope_ubuf)
            self.dup_value(self.counts_ub, self.counts_size)
        self.grad_ub = self.tik_instance.Tensor(self.dtype_grad, (self.ub_grad_size,),
                                                name="grad_ub", scope=tik.scope_ubuf)
        if self.high_precision:
            self.grad_ub_fp32 = self.tik_instance.Tensor("float32", (self.ub_grad_size,),
                                                name="grad_ub", scope=tik.scope_ubuf)
        self.base_count_words_compute(core_index)

    def base_count_words_compute(self, core_index):
        """
        when sf is True,use base function to count words

        Parameters
        ----------
        core_index: the index of aicore
        Returns
        -------
        None
        """
        if self.scale_grad_by_freq:
            # Define k, the scalar used to index the elements of indicators
            self.k = self.tik_instance.Scalar(dtype=self.dtype_indices)
            # Move indexes blocks from gm to ub
            with self.tik_instance.for_range(0, self.numel_indices // self.ub_indices_size) as i1:
                self.tik_instance.data_move(self.indices_ub, self.indices[i1 * self.ub_indices_size], 0, 1,
                                            self.ub_indices_size // self.indices_each_block, 0, 0)
                # Use self.counts_ub to count word frequency
                self.count_words_compute(self.ub_indices_size)
        self.remaining_count_words_compute(core_index)

    def add_same_word_grad_not_need_scale(self, total):
        """
        when sf is False,use this function to compute grad_weight
        by add and scale=1
        when sf is True,use this function to compute grad_weight
        by add and scale=1/counts[k]

        Parameters
        ----------
        total:int32,the total size need to compute grad_weight

        Returns
        -------
        None
        """
        if self.embedding_dim > self.grad_each_block and self.embedding_dim % self.grad_each_block != 0:
            self.add_grad_no_scale_not_align(total)
        else:
            if self.numel_indices >= Constant.THRESHOLD_NUM and self.embedding_dim <= self.grad_each_block and \
                            self.num_weights <= Constant.HIGH_PERFORMANCE_NUM:
                with self.tik_instance.for_range(0, total) as self.index:
                    self.k.set_as(self.indices_ub[self.index])
                    with self.tik_instance.if_scope(self.k != self.padding_idx):
                        with self.tik_instance.if_scope(tik.all(self.k >= 0, self.k < self.num_weights)):
                            self.tik_instance.vec_add(self.grad_each_block, self.add_y_ub[self.k *
                              self.grad_each_block], self.add_y_ub[self.k * self.grad_each_block],
                              self.grad_ub[self.index * 8], 1, 8, 8, 8)
            else:
                with self.tik_instance.for_range(0, total) as self.index:
                    self.k.set_as(self.indices_ub[self.index])
                    with self.tik_instance.if_scope(self.k != self.padding_idx):
                        with self.tik_instance.if_scope(tik.all(self.k >= 0, self.k < self.num_weights)):
                            if self.high_precision:
                                self.tik_instance.set_atomic_add("float32")
                                self.tik_instance.data_move(self.fp32_workspace[self.k * self.embedding_dim],
                                                            self.grad_ub_fp32[self.index * self.embedding_dim], 0,
                                                            1, self.embedding_dim // 8, 0, 0)
                                self.tik_instance.set_atomic_add(0)
                            else:
                                self.tik_instance.set_atomic_add(self.dtype_grad)
                                if self.embedding_dim < self.grad_each_block:
                                    with self.tik_instance.for_range(0, self.embedding_dim) as i:
                                        self.block_ub[i].set_as(self.grad_ub[self.index * self.embedding_dim + i])
                                    self.tik_instance.data_move(self.grad_weight[self.k * self.embedding_dim],
                                                                self.block_ub, 0,
                                                                1, 1, 0, 0)
                                else:
                                    self.tik_instance.data_move(self.grad_weight[self.k * self.embedding_dim],
                                                                self.grad_ub[self.index * self.embedding_dim], 0,
                                                                1, self.embedding_dim // self.grad_each_block, 0, 0)
                                self.tik_instance.set_atomic_add(0)

    def base_compute_grad_weight_not_need_scale(self, core_index):
        """
        when sf is False,just use base function to compute the grad_weight
        when sf is True,use base function to compute the grad_weight by scale

        Parameters
        ----------
        core_index: the index of aicore
        Returns
        -------
        None
        """
        # Define k, the scalar used to index the elements of indicators
        self.k = self.tik_instance.Scalar(dtype=self.dtype_indices)
        if self.dtype_grad == "bfloat16":
            scalar_float0 = self.tik_instance.Scalar(dtype="float16", init_value=0)
        else:
            scalar_float0 = self.tik_instance.Scalar(dtype=self.dtype_grad, init_value=0)
        self.ranges = self.tik_instance.Scalar('int32', name='ranges')
        with self.tik_instance.if_scope(core_index == self.used_core - 1):
            self.ranges.set_as(self.ele_last_core)
        with self.tik_instance.else_scope():
            self.ranges.set_as(self.ele_not_last_core)
        # Move indexes and grad blocks from gm to ub
        if self.embedding_dim > self.count:
            self.base_compute_no_scale_huge_embedding_dim(core_index)
        elif self.embedding_dim > self.grad_each_block and self.embedding_dim % self.grad_each_block != 0:
            self.base_compute_no_scale_not_align(core_index)
        else:
            if self.numel_indices >= Constant.THRESHOLD_NUM and self.embedding_dim <= self.grad_each_block and \
                            self.num_weights <= Constant.HIGH_PERFORMANCE_NUM:
                self.add_y_ub = self.tik_instance.Tensor(self.dtype_grad, (self.num_weights, 8), name="add_y_ub",
                                                         scope=tik.scope_ubuf)
                self.dup_value(self.add_y_ub, self.num_weights * self.grad_each_block)
                with self.tik_instance.if_scope(self.ranges // self.ub_indices_size > 0):
                    with self.tik_instance.for_range(0, self.ranges // self.ub_indices_size) as i1:
                        self.tik_instance.data_move(self.indices_ub, self.indices[core_index * self.ele_not_last_core +
                                                                                  i1 * self.ub_indices_size], 0, 1,
                                                    self.ub_indices_size // self.indices_each_block, 0, 0)
                        with self.tik_instance.for_range(0, self.ub_indices_size) as i2:
                            self.tik_instance.data_move(self.grad_ub[i2 * self.grad_each_block],
                                                        self.grad[(core_index * self.ele_not_last_core + i1 *
                                                                   self.ub_indices_size + i2) * self.embedding_dim],
                                                        0, 1, 1, 0, 0)
                        self.add_same_word_grad_not_need_scale(self.ub_indices_size)

            else:
                if self.embedding_dim < self.grad_each_block:
                    self.block_ub = self.tik_instance.Tensor(self.dtype_grad, (self.grad_each_block,), name="block_ub",
                                                            scope=tik.scope_ubuf)
                    self.tik_instance.vec_dup(self.grad_each_block, self.block_ub, scalar_float0, 1, 8)
                with self.tik_instance.if_scope(self.ranges // self.ub_indices_size > 0):
                    with self.tik_instance.for_range(0, self.ranges // self.ub_indices_size) as i1:
                        self.tik_instance.data_move(self.indices_ub, self.indices[core_index * self.ele_not_last_core +
                                                                                  i1 * self.ub_indices_size], 0, 1,
                                                    self.ub_indices_size // self.indices_each_block, 0, 0)
                        self.tik_instance.data_move(self.grad_ub,
                                                    self.grad[(core_index * self.ele_not_last_core + i1 *
                                                               self.ub_indices_size) * self.embedding_dim],
                                                    0, 1, 
                                                    self.ub_indices_size * self.embedding_dim // self.grad_each_block,
                                                    0, 0)
                        if self.high_precision:
                            self.conv_fp32(self.grad_ub, self.grad_ub_fp32, self.ub_indices_size * self.embedding_dim)
                        self.add_same_word_grad_not_need_scale(self.ub_indices_size)
            self.remaining_compute_grad_weight_not_need_scale(core_index)

    def remaining_count_words_compute(self, core_index):
        """
        when sf is True,use remaining function to count words

        Parameters
        ----------
        core_index: the index of aicore
        Returns
        -------
        None
        """
        if self.scale_grad_by_freq:
            if self.numel_indices % self.ub_indices_size != 0:
                offset_indices_move = self.numel_indices // self.ub_indices_size * self.ub_indices_size
                burst_len_indices = math.ceil(self.numel_indices % self.ub_indices_size / self.indices_each_block)
                self.tik_instance.data_move(self.indices_ub, self.indices[offset_indices_move], 0, 1, burst_len_indices,
                                            0, 0)
                # Use self.counts_ub to count word frequency
                self.count_words_compute(self.numel_indices % self.ub_indices_size)
            self.base_compute_grad_weight_need_scale()
        else:
            self.base_compute_grad_weight_not_need_scale(core_index)

    def base_compute_grad_weight_need_scale(self):
        """
        when sf is False,just use base function to compute the grad_weight
        when sf is True,use base function to compute the grad_weight by scale

        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        self.add_tensor = self.tik_instance.Tensor(
            self.dtype_grad, (1, self.embedding_dim), name="add_tensor", scope=tik.scope_ubuf)
        self.scale_int = self.tik_instance.Scalar(dtype=self.dtype_indices)
        self.scale_float = self.tik_instance.Scalar(
            init_value=1.0, dtype=self.dtype_grad)
        # Define k, the scalar used to index the elements of indicators
        self.k = self.tik_instance.Scalar(dtype=self.dtype_indices)
        # Move indexes and grad blocks from gm to ub
        with self.tik_instance.for_range(0, self.numel_indices // self.ub_indices_size) as i1:
            self.tik_instance.data_move(self.indices_ub, self.indices[i1 * self.ub_indices_size], 0, 1,
                                        self.ub_indices_size // self.indices_each_block, 0, 0)
            self.tik_instance.data_move(self.grad_ub, self.grad[i1 * self.ub_indices_size * self.embedding_dim],
                                        0, 1, self.ub_indices_size * self.embedding_dim // self.grad_each_block, 0, 0)
            self.add_same_word_grad_need_scale(self.ub_indices_size)
        self.remaining_compute_grad_weight_need_scale()

    def cac_result(self):
        """
        caculate the rersult of grad_weight

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.scale_int.set_as(self.counts_ub[self.k - self.begin])
        self.scale_float_tmp = self.tik_instance.Scalar(dtype=Constant.MID_DTYPE)
        self.tik_instance.scalar_conv('', self.scale_float_tmp, self.scale_int)
        numerator = 1.0
        self.scale_float_tmp.set_as(numerator / self.scale_float_tmp)
        if self.dtype_grad != Constant.MID_DTYPE:
            self.tik_instance.scalar_conv('', self.scale_float, self.scale_float_tmp)
        else:
            self.scale_float.set_as(self.scale_float_tmp)
        if self.embedding_dim // self.vector_mask_max_grad > 0:
            self.tik_instance.vec_axpy(self.vector_mask_max_grad, self.add_tensor,
                                       self.grad_ub[self.index * self.embedding_dim], self.scale_float,
                                       self.embedding_dim // self.vector_mask_max_grad, 8, 8)
        if self.embedding_dim % self.vector_mask_max_grad > 0:
            self.tik_instance.vec_axpy(self.embedding_dim % self.vector_mask_max_grad,
                                       self.add_tensor[self.embedding_dim // self.vector_mask_max_grad *
                                                       self.vector_mask_max_grad],
                                       self.grad_ub[self.index * self.embedding_dim + self.embedding_dim //
                                                    self.vector_mask_max_grad * self.vector_mask_max_grad],
                                       self.scale_float, 1, 8, 8)
        self.tik_instance.data_move(self.grad_weight[self.k * self.embedding_dim], self.add_tensor, 0, 1,
                                    self.embedding_dim // self.grad_each_block, 0, 0)

    def add_same_word_grad_need_scale(self, total):
        """
        when sf is False,use this function to compute grad_weight
        by add and scale=1
        when sf is True,use this function to compute grad_weight
        by add and scale=1/counts[k]

        Parameters
        ----------
        total:int32,the total size need to compute grad_weight

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, total) as self.index:
            self.k.set_as(self.indices_ub[self.index])
            with self.tik_instance.if_scope(self.k != self.padding_idx):
                with self.tik_instance.if_scope(tik.all(self.k < self.end, self.k >= self.begin)):
                    self.tik_instance.data_move(self.add_tensor, self.grad_weight[self.k * self.embedding_dim], 0, 1,
                                                self.embedding_dim // self.grad_each_block, 0, 0)
                    self.cac_result()

    def remaining_compute_grad_weight_need_scale(self):
        """
        when sf is False,just use base function to compute the grad_weight
        when sf is True,use base function to compute the grad_weight by scale

        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        if self.numel_indices % self.ub_indices_size != 0:
            offset_indices_move = self.numel_indices // self.ub_indices_size * self.ub_indices_size
            burst_len_indices = math.ceil(
                self.numel_indices %
                self.ub_indices_size /
                self.indices_each_block)
            offset_grad_move = self.numel_indices // self.ub_indices_size * \
                               self.ub_indices_size * self.embedding_dim
            burst_len_grad = self.numel_indices % self.ub_indices_size * \
                             self.embedding_dim // self.grad_each_block

            self.tik_instance.data_move(self.indices_ub, self.indices[offset_indices_move], 0, 1, burst_len_indices, 0,
                                        0)
            self.tik_instance.data_move(self.grad_ub, self.grad[offset_grad_move], 0, 1, burst_len_grad, 0, 0)
            self.add_same_word_grad_need_scale(self.numel_indices % self.ub_indices_size)

    def count_words_compute(self, total):
        """
        when sf is True,use this function to count word frequency

        Parameters
        ----------
        total:int32,the total size need to count word frequency

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, total) as index:
            self.k.set_as(self.indices_ub[index])
            with self.tik_instance.if_scope(tik.all(self.k < self.end, self.k >= self.begin)):
                with self.tik_instance.if_scope(self.k != self.padding_idx):
                    tmp = self.tik_instance.Scalar(dtype=self.dtype_indices)
                    tmp.set_as(self.counts_ub[self.k - self.begin])
                    self.counts_ub[self.k - self.begin].set_as(tmp + 1)

    def add_grad_no_scale_not_align(self, total):
        """
        add_grad_no_scale_not_align
        """
        align_size = ((self.embedding_dim - 1) // self.grad_each_block + 1) * self.grad_each_block
        with self.tik_instance.for_range(0, total) as index:
            self.k.set_as(self.indices_ub[index])
            with self.tik_instance.if_scope(tik.all(self.k >= 0, self.k < self.num_weights,
                                                    self.k != self.padding_idx)):
                with self.tik_instance.for_range(0, self.grad_each_block - \
                                                self.embedding_dim % self.grad_each_block) as i:
                    self.grad_ub[index * align_size + self.embedding_dim + i].set_as(0)
                self.tik_instance.set_atomic_add(self.dtype_grad)
                self.tik_instance.data_move(self.grad_weight[self.k * self.embedding_dim],
                                            self.grad_ub[index * align_size], 0,
                                            1, align_size // self.grad_each_block, 0, 0)
                self.tik_instance.set_atomic_add(0)

    def remaining_compute_grad_weight_not_need_scale(self, core_index):
        """
        when sf is False,just use base function to compute the grad_weight
        when sf is True,use base function to compute the grad_weight by scale

        Parameters
        ----------
        core_index: the index of aicore
        Returns
        -------
        None
        """
        if self.numel_indices >= Constant.THRESHOLD_NUM and self.embedding_dim <= self.grad_each_block and \
                            self.num_weights <= Constant.HIGH_PERFORMANCE_NUM:
            with self.tik_instance.if_scope(self.ranges % self.ub_indices_size != 0):
                offset_indices_move = self.tik_instance.Scalar('int32', name='offset_indices_move')
                offset_indices_move.set_as(core_index * self.ele_not_last_core + self.ranges // self.ub_indices_size \
                                                                            * self.ub_indices_size)
                burst_len_indices = self.tik_instance.Scalar('int32', name='burst_len_indices')
                offset_grad_move = self.tik_instance.Scalar('int32', name='offset_grad_move')
                burst_len_indices.set_as((
                    self.ranges %
                    self.ub_indices_size - 1) //
                    self.indices_each_block + 1)
                offset_grad_move.set_as((core_index * self.ele_not_last_core + self.ranges // self.ub_indices_size *
                                    self.ub_indices_size) * self.embedding_dim)

                self.tik_instance.data_move(self.indices_ub, self.indices[offset_indices_move], 0, 1,
                                            burst_len_indices, 0, 0)
                with self.tik_instance.for_range(0, self.ranges % self.ub_indices_size) as i:
                    self.tik_instance.data_move(self.grad_ub[i * self.grad_each_block], self.grad[offset_grad_move + i *
                                                                                            self.embedding_dim],
                                                0, 1, 1, 0, 0)
                self.add_same_word_grad_not_need_scale(self.ranges % self.ub_indices_size)
                self.transpose_and_move_data_out_with_six()
        else:
            with self.tik_instance.if_scope(self.ranges % self.ub_indices_size != 0):
                offset_indices_move = self.tik_instance.Scalar('int32', name='offset_indices_move')
                offset_indices_move.set_as(core_index * self.ele_not_last_core + self.ranges // self.ub_indices_size \
                                            * self.ub_indices_size)
                burst_len_indices = self.tik_instance.Scalar('int32', name='burst_len_indices')
                burst_len_grad = self.tik_instance.Scalar('int32', name='burst_len_grad')
                offset_grad_move = self.tik_instance.Scalar('int32', name='offset_grad_move')
                burst_len_indices.set_as((
                    self.ranges %
                    self.ub_indices_size - 1) //
                    self.indices_each_block + 1)
                offset_grad_move.set_as((core_index * self.ele_not_last_core + self.ranges // self.ub_indices_size *
                                    self.ub_indices_size) * self.embedding_dim)
                burst_len_grad.set_as(((self.ranges % self.ub_indices_size) * self.embedding_dim - 1)
                                      // self.grad_each_block + 1)

                self.tik_instance.data_move(self.indices_ub, self.indices[offset_indices_move], 0, 1,
                                            burst_len_indices, 0, 0)
                self.tik_instance.data_move(self.grad_ub, self.grad[offset_grad_move], 0, 1, burst_len_grad, 0, 0)
                if self.high_precision:
                    self.conv_fp32(self.grad_ub, self.grad_ub_fp32,
                                   (self.ranges % self.ub_indices_size) * self.embedding_dim)
                self.add_same_word_grad_not_need_scale(self.ranges % self.ub_indices_size)

    def base_compute_no_scale_huge_embedding_dim(self, core_index):
        """
        function to base compute the grad_weight when no scale and with huge embedding dim
        """
        with self.tik_instance.if_scope(self.ranges // self.ub_indices_size > 0):
            with self.tik_instance.for_range(0, self.ranges // self.ub_indices_size) as i:
                self.tik_instance.data_move(self.indices_ub, self.indices[core_index * self.ele_not_last_core +
                                                                          i * self.ub_indices_size], 0, 1,
                                            self.ub_indices_size // self.indices_each_block, 0, 0)
                self.add_same_grad_no_scale_huge_embedding_dim(core_index, self.ub_indices_size, i)
        self.remain_compute_no_scale_huge_embedding_dim(core_index)

    def remain_compute_no_scale_huge_embedding_dim(self, core_index):
        """
        function to compute the remain of input to grad_weight when no scale and with huge embedding dim
        """
        with self.tik_instance.if_scope(self.ranges % self.ub_indices_size != 0):
            offset_indices_move = self.tik_instance.Scalar('int32', name='offset_indices_move')
            offset_indices_move.set_as(core_index * self.ele_not_last_core + self.ranges // self.ub_indices_size \
                                       * self.ub_indices_size)
            burst_len_indices = self.tik_instance.Scalar('int32', name='burst_len_indices')
            offset_grad_move = self.tik_instance.Scalar('int32', name='offset_grad_move')
            burst_len_indices.set_as((self.ranges % self.ub_indices_size - 1) // self.indices_each_block + 1)
            offset_grad_move.set_as((core_index * self.ele_not_last_core + self.ranges // self.ub_indices_size *
                                     self.ub_indices_size) * self.embedding_dim)

            self.tik_instance.data_move(self.indices_ub, self.indices[offset_indices_move], 0, 1, burst_len_indices, 0,
                                        0)
            self.add_same_grad_no_scale_huge_embedding_dim(core_index, self.ranges % self.ub_indices_size,
                                                           self.ranges // self.ub_indices_size)

    def add_same_grad_no_scale_huge_embedding_dim(self, core_index, total, ind):
        """
        use this function to compute grad_weight
        by add with no scale and a huge embedding dim
        """
        with self.tik_instance.for_range(0, total) as self.index:
            self.tik_instance.data_move(self.grad_ub,
                                        self.grad[(core_index * self.ele_not_last_core + ind *
                                                   self.ub_indices_size + self.index) * self.embedding_dim],
                                        0, 1, self.embedding_dim // self.grad_each_block, 0, 0)
            if self.high_precision:
                self.conv_fp32(self.grad_ub, self.grad_ub_fp32, self.embedding_dim)
            self.k.set_as(self.indices_ub[self.index])
            with self.tik_instance.if_scope(self.k != self.padding_idx):
                with self.tik_instance.if_scope(tik.all(self.k >= 0, self.k < self.num_weights)):
                    if self.high_precision:
                        self.tik_instance.set_atomic_add("float32")
                        self.tik_instance.data_move(self.fp32_workspace[self.k * self.embedding_dim],
                                                    self.grad_ub_fp32, 0,
                                                    1, self.embedding_dim // 8, 0, 0)
                        self.tik_instance.set_atomic_add(0)
                    else:
                        self.tik_instance.set_atomic_add(self.dtype_grad)
                        self.tik_instance.data_move(self.grad_weight[self.k * self.embedding_dim],
                                                    self.grad_ub, 0,
                                                    1, self.embedding_dim // self.grad_each_block, 0, 0)
                        self.tik_instance.set_atomic_add(0)
        if self.embedding_dim % self.grad_each_block != 0:
            floor_align_size = self.embedding_dim // self.grad_each_block * self.grad_each_block
            tail_grad_ub = self.tik_instance.Tensor(self.dtype_grad, 
                                                    (self.grad_each_block,), 
                                                    name='grad_ub', scope=tik.scope_ubuf)
            
            with self.tik_instance.for_range(0, total) as self.index:
                self.tik_instance.data_move(tail_grad_ub,
                                            self.grad[(core_index * self.ele_not_last_core + ind *
                                                    self.ub_indices_size + self.index) \
                                                    * self.embedding_dim + floor_align_size],
                                            0, 1, 1, 0, 0)
                self.k.set_as(self.indices_ub[self.index])
                with self.tik_instance.for_range(0, self.grad_each_block - \
                                                 (self.embedding_dim % self.grad_each_block)) as i:
                    tail_grad_ub[self.embedding_dim % self.grad_each_block + i].set_as(0)
                with self.tik_instance.if_scope(tik.all(self.k >= 0, self.k < self.num_weights,
                                                        self.k != self.padding_idx)):
                    self.tik_instance.set_atomic_add(self.dtype_grad)
                    self.tik_instance.data_move(self.grad_weight[self.k * self.embedding_dim \
                                                                + floor_align_size],
                                                tail_grad_ub, 0,
                                                1, 1, 0, 0)
                    self.tik_instance.set_atomic_add(0)

    def base_compute_no_scale_not_align(self, core_index):
        """
        base_compute_no_scale_not_align
        """
        align_length = (self.embedding_dim - 1) // self.grad_each_block + 1
        with self.tik_instance.if_scope(self.ranges // self.ub_indices_size > 0):
            with self.tik_instance.for_range(0, self.ranges // self.ub_indices_size) as i:
                self.tik_instance.data_move(self.indices_ub, self.indices[core_index * self.ele_not_last_core +
                                                                          i * self.ub_indices_size], 0, 1,
                                            self.ub_indices_size // self.indices_each_block, 0, 0)
                with self.tik_instance.for_range(0, self.ub_indices_size) as j:
                    self.tik_instance.data_move(self.grad_ub[j * align_length * self.grad_each_block],
                                                self.grad[(core_index * self.ele_not_last_core + i *
                                                           self.ub_indices_size + j) * self.embedding_dim],
                                                0, 1, align_length, 0, 0)
                self.add_same_word_grad_not_need_scale(self.ub_indices_size)
        self.remain_compute_no_scale_not_align(core_index, align_length)

    def remain_compute_no_scale_not_align(self, core_index, align_size):
        """
        remain_compute_no_scale_not_align
        """
        with self.tik_instance.if_scope(self.ranges % self.ub_indices_size != 0):
            offset_indices_move = self.tik_instance.Scalar('int32', name='offset_indices_move')
            offset_indices_move.set_as(core_index * self.ele_not_last_core + self.ranges // self.ub_indices_size \
                                                                        * self.ub_indices_size)
            burst_len_indices = self.tik_instance.Scalar('int32', name='burst_len_indices')
            offset_grad_move = self.tik_instance.Scalar('int32', name='offset_grad_move')
            burst_len_indices.set_as((
                self.ranges %
                self.ub_indices_size - 1) //
                self.indices_each_block + 1)
            offset_grad_move.set_as((core_index * self.ele_not_last_core + self.ranges // self.ub_indices_size *
                                self.ub_indices_size) * self.embedding_dim)

            self.tik_instance.data_move(self.indices_ub, self.indices[offset_indices_move], 0, 1, burst_len_indices, 0,
                                        0)
            with self.tik_instance.for_range(0, self.ranges % self.ub_indices_size) as i:
                self.tik_instance.data_move(self.grad_ub[i * align_size * self.grad_each_block],
                                            self.grad[(core_index * self.ele_not_last_core + self.ranges //
                                                       self.ub_indices_size * self.ub_indices_size + i)
                                                      * self.embedding_dim],
                                            0, 1, align_size, 0, 0)
            self.add_same_word_grad_not_need_scale(self.ranges % self.ub_indices_size)

    def transpose_and_move_data_out_with_six(self):
        """
        when the embedding_dim is less than 8, the indices_num is more than 10240
        and nums_weights is less than 32, use this transpose and pad func
        """
        tik_instance = self.tik_instance
        block = 16
        scale_32_to_16 = 2
        align_num_weights = ((self.num_weights - 1) // block + 1) * block
        time = align_num_weights // block
        conv_ub = tik_instance.Tensor("float16", (block, align_num_weights), name="conv_ub",
                                                         scope=tik.scope_ubuf)
        add_y_ub_fp16 = self.add_y_ub.reinterpret_cast_to("float16")
        with self.tik_instance.for_range(0, time) as i:
            src_list = []
            dst_list = []
            for j in range(block):
                src_list.append(add_y_ub_fp16[block * block * i + block * j])
                dst_list.append(conv_ub[time * block * j + block * i])
            self.tik_instance.vnchwconv(True, False, dst_list, src_list, 1, 0, 0)
        if self.embedding_dim * scale_32_to_16 * align_num_weights < 512:
            tik_instance.vec_dup(Constant.MASK, conv_ub[self.embedding_dim * scale_32_to_16 * align_num_weights],
                                 0, 1, 8)
        with self.tik_instance.for_range(0, time) as i:
            src_list = []
            dst_list = []
            for j in range(block):
                dst_list.append(add_y_ub_fp16[block * block * i + block * j])
                src_list.append(conv_ub[time * block * j + block * i])
            self.tik_instance.vnchwconv(True, False, dst_list, src_list, 1, 0, 0)
        add_y_ub_fp32 = add_y_ub_fp16.reinterpret_cast_to("float32")
        self.tik_instance.set_atomic_add(self.dtype_grad)
        with self.tik_instance.for_range(0, self.num_weights) as ind:
            self.tik_instance.data_move(self.grad_weight[ind * self.embedding_dim],
                                        add_y_ub_fp32[ind * self.grad_each_block], 0, 1, 1, 0, 0)
        self.tik_instance.set_atomic_add(0)


# 'pylint: disable=too-many-arguments
@register_operator('EmbeddingDenseGrad')
def embedding_dense_grad(
        grad,
        indices,
        y,
        num_weights,
        padding_idx,
        scale_grad_by_freq,
        kernel_name="embedding_dense_grad",
        impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """
    the main function of embedding_dense_grad

    Parameters
    ----------
    grad: dict,shape and datatype,
    datatype supports float32
    indices: dict,shape and datatype,
    datatype supports int32
    y:dict,shape and datatype,
    datatype supports float32
    num_weights:the number of words in dict
    padding_idx:judge grad_weight of which word is zero
    scale_grad_by_freq: judge whether or not  scale_grad
    kernel_name: cce kernel name, default value is "embedding_dense_grad"
    Returns
    -------
    tik_instance: tik_instance
    """
    check_op_impl_mode(impl_mode, [OpImplMode.HIGH_PERFORMANCE, OpImplMode.HIGH_PRECISION], kernel_name)
    embedding_dense_grad_instance = EmbeddingDenseGrad(
        grad, indices, y, num_weights, padding_idx, scale_grad_by_freq, kernel_name, impl_mode)
    tik_instance = embedding_dense_grad_instance.embedding_dense_grad_compute()
    return tik_instance
