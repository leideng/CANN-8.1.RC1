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
embedding_dense_grad
"""

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform as cce
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    RESERVE_SIZE = 16 * 1024
    BLOCK = 8
    MAX_INT32 = 2 ** 31 - 1
    SCALAR_TENSOR_SIZE = 7
    TILING_ARG_NUM = 64
    TILING_MODE_1 = 1
    GRAD_TENSOR_PART = 512
    THRESHOLD_NUM = 10240
    MASK = 128
    HIGH_PERFORMANCE_NUM = 32
    RESERVE_SIZE = 16 * 1024
    FLOAT32_BYTES = 4
    TOTAL_PART = 513
    ORI_DEPENDENCY_INDICES_DTYPE = 'int32'
    TILING_DTYPE = 'int32'
    RELIES_ON_FLOAT32 = 'float32'
    BLOCK_BYTES_SIZE = 32
    COUNT_INIT = 400000
    FOUR_BYTES = 4
    ALIGN_4_CORE_TASK = 128
    DIM_SIX = 6


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
            kernel_name):
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
        self.dtype_grad = grad.get("dtype")
        self.dtype_indices = indices.get("dtype")
        self.embedding_dim = None
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.padding_idx = None
        self.num_weights = None
        self.kernel_name = kernel_name
        self.scale_grad_by_freq = scale_grad_by_freq
        self.block_ub = None
        self.align_ub = None
        self.new_numel_indices = None
        self.grad_ub = None
        self.grad_movement_times = None
        self.grad_last_movement = None
        self.indices_ub = None
        self.grad = None
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
        self.add_tensor_temp = None
        self.index = None
        self.add_tensor_size = None
        self.counts_size = None
        self.tiling_core_num = None
        self.end = None
        self.scale_int = None
        self.add_y_ub = None
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
        '''The vector instruction calculates a maximum of 8 blocks per repeat.
        This parameter is the maximum value of the mask when grad performs vector calculation
        '''
        self.vector_mask_max_counts = Constant.BLOCK * self.counts_each_block
        self.vector_mask_max_grad = Constant.BLOCK * self.grad_each_block
        self.ele_not_last_core = self.tik_instance.Scalar(Constant.ORI_DEPENDENCY_INDICES_DTYPE,
                                                          name='ele_not_last_core')
        self.ele_last_core = self.tik_instance.Scalar(Constant.ORI_DEPENDENCY_INDICES_DTYPE,
                                                      name='ele_last_core')
        self.move_length = self.tik_instance.Scalar("int32", name='move_length')
        self.ranges = self.tik_instance.Scalar("int32", name='ranges')
        self.count = self.tik_instance.Scalar("int32", name='count', init_value=Constant.COUNT_INIT)
        self.index_not_last_core = None
        self.ub_grad_size = self.tik_instance.Scalar(Constant.ORI_DEPENDENCY_INDICES_DTYPE, name='ub_grad_size')
        self.ub_indices_size = self.tik_instance.Scalar(Constant.ORI_DEPENDENCY_INDICES_DTYPE, name='ub_indices_size')
        self.tiling_dtype = 'int32'
        self.dtype_bytes_size_tiling = cce.get_bit_len(self.tiling_dtype) // Constant.BLOCK
        self.tiling_each_block = block_bite_size // self.dtype_bytes_size_tiling
        self.core_used = self.tik_instance.Scalar(Constant.ORI_DEPENDENCY_INDICES_DTYPE, name='core_used')
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype, (Constant.TILING_ARG_NUM,),
                                                  name='tiling_gm', scope=tik.scope_gm)

    def get_tiling_args(self):
        """
        get tiling args from tling_ub

        Parameters
        ----------
        tiling_ub: s tensor with tiling_args in ub

        Returns
        -------
        None
        """
        tiling_ub = self.tik_instance.Tensor(self.tiling_dtype, (Constant.TILING_ARG_NUM,),
                                             name='tiling_ub', scope=tik.scope_ubuf)
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, (Constant.SCALAR_TENSOR_SIZE - 1) //
                                    self.tiling_each_block + 1, 0, 0)
        self.padding_idx = self.tik_instance.Scalar(Constant.TILING_DTYPE, name='padding_idx')
        self.num_weights = self.tik_instance.Scalar(Constant.TILING_DTYPE, name='num_weights')
        self.embedding_dim = self.tik_instance.Scalar(Constant.TILING_DTYPE, name='embedding_dim')
        self.numel_indices = self.tik_instance.Scalar(Constant.TILING_DTYPE, name='numel_indices')
        self.tiling_core_num = self.tik_instance.Scalar(Constant.TILING_DTYPE, name='tiling_core_num',
                                                        init_value=self.aicore_num)

        self.numel_indices.set_as(tiling_ub[0])
        self.embedding_dim.set_as(tiling_ub[1])
        self.num_weights.set_as(tiling_ub[4])
        self.padding_idx.set_as(tiling_ub[5])
        self.tiling_core_num.set_as(tiling_ub[6])

    def cal_ub_size(self):
        """
        cal_ub_size
        """
        embedding_dim = self.tik_instance.Scalar("int32", name="embedding_dim", init_value=self.embedding_dim)
        need_align = 0
        with self.tik_instance.if_scope(self.embedding_dim % self.grad_each_block != 0):
            embedding_dim.set_as(((self.embedding_dim - 1) // self.grad_each_block + 1) * self.grad_each_block)
            need_align = 1
        if self.scale_grad_by_freq:
            self.index_not_last_core = (self.num_weights - 1) // self.tiling_core_num + 1
            core_used = (self.num_weights - 1) // self.index_not_last_core + 1
            self.core_used.set_as(core_used)
            self.counts_size = self.num_weights // core_used + self.num_weights % core_used
            self.ub_indices_size.set_as((self.ub_size_bytes - self.counts_size * \
                                self.dtype_bytes_size_counts - Constant.RESERVE_SIZE) \
                                // (self.embedding_dim * self.dtype_bytes_size_grad + \
                                    self.dtype_bytes_size_indices) \
                                // self.indices_each_block * self.indices_each_block)
            self.ub_grad_size.set_as(self.ub_indices_size * self.embedding_dim)
        else:
            self.grad_movement_times = self.tik_instance.Scalar(Constant.ORI_DEPENDENCY_INDICES_DTYPE,
                                                                   name="grad_movement_times", init_value = 0)
            self.grad_last_movement = self.tik_instance.Scalar(Constant.ORI_DEPENDENCY_INDICES_DTYPE,
                                                                   name="grad_last_movement", init_value = 0)
            self.ele_not_last_core.set_as(((self.numel_indices + self.tiling_core_num - 1) // self.tiling_core_num 
                                           + Constant.ALIGN_4_CORE_TASK - 1) // Constant.ALIGN_4_CORE_TASK 
                                          * Constant.ALIGN_4_CORE_TASK)
            self.core_used.set_as((self.numel_indices - 1) // self.ele_not_last_core + 1)
            with self.tik_instance.if_scope(self.numel_indices // self.core_used \
                                            * self.embedding_dim < self.grad_each_block):
                self.core_used.set_as(1)
            self.ele_last_core.set_as(self.numel_indices - (self.core_used - 1) * self.ele_not_last_core)
            self.ub_indices_size.set_as((self.ub_size_bytes - Constant.RESERVE_SIZE - \
                                         need_align * embedding_dim * self.dtype_bytes_size_grad) \
                                    // (embedding_dim * self.dtype_bytes_size_grad + \
                                        self.dtype_bytes_size_indices) \
                                    // self.indices_each_block * self.indices_each_block)
            self.ub_grad_size.set_as(self.ub_indices_size * embedding_dim)
            # The situation where 1 to indices_each_block Grad rows can be accommodated in one UB.
            with self.tik_instance.if_scope(tik.all(self.ub_indices_size == 0,
                                            self.ub_size_bytes - Constant.RESERVE_SIZE - \
                                            Constant.BLOCK_BYTES_SIZE >= (1 + need_align) * embedding_dim * \
                                            self.dtype_bytes_size_grad)):
                self.ub_grad_size.set_as(embedding_dim)
                self.ub_indices_size.set_as(1)
                self.count.set_as((self.ub_size_bytes - Constant.RESERVE_SIZE) 
                                  // (self.indices_each_block * self.dtype_bytes_size_grad) - 
                                  self.dtype_bytes_size_indices // self.dtype_bytes_size_grad)
            # The situation is one Grad row can or cannot be accommodated in one UB.
            with self.tik_instance.elif_scope(tik.all(self.ub_indices_size == 0,
                                            self.ub_size_bytes - Constant.RESERVE_SIZE - \
                                            Constant.BLOCK_BYTES_SIZE < (1 + need_align) * embedding_dim *
                                            self.dtype_bytes_size_grad)):
                self.ub_indices_size.set_as(self.indices_each_block)
                self.ub_grad_size.set_as((((self.ub_size_bytes - Constant.RESERVE_SIZE - 
                                        Constant.BLOCK_BYTES_SIZE) // self.dtype_bytes_size_grad)\
                                        // self.grad_each_block * self.grad_each_block))
                #if grad_movement_times > 0 ,embedding_dim overflow a UB
                self.grad_movement_times.set_as(embedding_dim // self.ub_grad_size)
                #when grad_movement_times > 0 ,embedding_dim overflow a UB, last datamove need move grad numels
                self.grad_last_movement.set_as(embedding_dim % self.ub_grad_size)
                self.count.set_as((self.ub_size_bytes - Constant.RESERVE_SIZE) \
                                  // (self.indices_each_block * self.dtype_bytes_size_grad) \
                                  - self.dtype_bytes_size_indices // self.dtype_bytes_size_grad)

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
        self.indices = self.tik_instance.Tensor(self.dtype_indices, (Constant.MAX_INT32,), name="indices",
                                                scope=tik.scope_gm)

        self.grad_weight = self.tik_instance.Tensor(self.dtype_grad, (Constant.MAX_INT32,),
                                                    name="grad_weight", scope=tik.scope_gm, is_atomic_add=True)
        self.grad = self.tik_instance.Tensor(self.dtype_grad, (Constant.MAX_INT32,), name="grad", scope=tik.scope_gm)

    def embedding_dense_grad_compute_tiling(self):
        """
        Compute the embedding_dense_grad op

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.get_tiling_args()
        self.gm_for_data_and_fill_grad_weight()
        self.cal_ub_size()
        with self.tik_instance.for_range(0, self.tiling_core_num, block_num=self.tiling_core_num) as core_index:
            with self.tik_instance.if_scope(core_index < self.core_used):
                if self.scale_grad_by_freq:
                    self.begin = core_index * self.index_not_last_core
                    with self.tik_instance.if_scope(core_index == self.core_used - 1):
                        self.end = self.num_weights
                    with self.tik_instance.else_scope():
                        self.end = (core_index + 1) * self.index_not_last_core
                self.ub_for_data(core_index)

        tbe_context.get_context().add_compile_info('vars', {'core_num': self.aicore_num,
                                                   'scale_grad_by_freq': self.scale_grad_by_freq})
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[self.grad, self.indices],
                                   outputs=[self.grad_weight], flowtable=[self.tiling_gm])
        return self.tik_instance

    @staticmethod
    def get_dtype_size(dtype):
        """
        get byte size of dtype
        """
        dtype_dict = {"float32": 4, "uint8": 1, "int32": 4, "int64": 8, "float16": 2, "bfloat16": 2}
        return dtype_dict.get(dtype)

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
            with self.tik_instance.if_scope(loop > 0):
                with self.tik_instance.for_range(0, loop) as index:
                    tmp_offset = offset + index * mask * 255
                    self.tik_instance.vec_dup(mask, dst[tmp_offset], dup_value, 255, stride)
                offset += loop * mask * 255

            repeat_time = (num % (mask * 255)) // mask
            with self.tik_instance.if_scope(repeat_time > 0):
                self.tik_instance.vec_dup(mask, dst[offset], dup_value, repeat_time, stride)
                offset += repeat_time * mask
            last_num = num % mask
            with self.tik_instance.if_scope(last_num > 0):
                self.tik_instance.vec_dup(last_num, dst[offset], dup_value, 1, 0)
            dst = dst.reinterpret_cast_to("bfloat16")
        else:
            mask = 256 // dtype_byte_size
            stride = 8
            loop = num // (mask * 255)
            with self.tik_instance.if_scope(loop > 0):
                with self.tik_instance.for_range(0, loop) as index:
                    tmp_offset = offset + index * mask * 255
                    self.tik_instance.vec_dup(mask, dst[tmp_offset], dup_value, 255, stride)
                offset += loop * mask * 255

            repeat_time = (num % (mask * 255)) // mask
            with self.tik_instance.if_scope(repeat_time > 0):
                self.tik_instance.vec_dup(mask, dst[offset], dup_value, repeat_time, stride)
                offset += repeat_time * mask
            last_num = num % mask
            with self.tik_instance.if_scope(last_num > 0):
                self.tik_instance.vec_dup(last_num, dst[offset], dup_value, 1, 0)

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
        self.move_length.set_as(self.ub_indices_size // self.indices_each_block)
        with self.tik_instance.if_scope(self.ub_indices_size // self.indices_each_block == 0):
            self.move_length.set_as(1)
        self.indices_ub = self.tik_instance.Tensor(self.dtype_indices, (self.ub_indices_size,), name="indices_ub",
                                                   scope=tik.scope_ubuf)
        if self.scale_grad_by_freq:
            self.counts_ub = self.tik_instance.Tensor(Constant.ORI_DEPENDENCY_INDICES_DTYPE, (self.counts_size,), 
                                                      name="counts_ub",
                                                      scope=tik.scope_ubuf)
            self.dup_value(self.counts_ub, self.counts_size)
        self.grad_ub = self.tik_instance.Tensor(self.dtype_grad, (self.ub_grad_size,),
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
            self.k = self.tik_instance.Scalar(dtype=Constant.ORI_DEPENDENCY_INDICES_DTYPE)
            # Move indexes blocks from gm to ub
            with self.tik_instance.for_range(0, self.numel_indices // self.ub_indices_size) as i1:
                self.tik_instance.data_move(self.indices_ub, self.indices[i1 * self.ub_indices_size], 0, 1,
                                            self.move_length, 0, 0)
                # Use self.counts_ub to count word frequency
                self.count_words_compute(self.ub_indices_size)
        self.remaining_count_words_compute(core_index)

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
            with self.tik_instance.if_scope(self.numel_indices % self.ub_indices_size != 0):
                offset_indices_move = self.tik_instance.Scalar(Constant.ORI_DEPENDENCY_INDICES_DTYPE, 
                                                               name='offset_indices_move')
                offset_indices_move.set_as(self.numel_indices // self.ub_indices_size * self.ub_indices_size)
                burst_len_indices = self.tik_instance.Scalar(Constant.ORI_DEPENDENCY_INDICES_DTYPE, 
                                                             name='burst_len_indices')
                burst_len_indices.set_as((self.numel_indices % self.ub_indices_size - 1) 
                                         // self.indices_each_block + 1)
                self.tik_instance.data_move(self.indices_ub, self.indices[offset_indices_move],
                                            0, 1, burst_len_indices,
                                            0, 0)
                # Use self.counts_ub to count word frequency
                self.count_words_compute(self.numel_indices % self.ub_indices_size)
            self.base_compute_grad_weight_need_scale()
        else:
            self.base_compute_grad_weight_not_need_scale(core_index)

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
                    tmp = self.tik_instance.Scalar(dtype=Constant.ORI_DEPENDENCY_INDICES_DTYPE)
                    tmp.set_as(self.counts_ub[self.k - self.begin])
                    self.counts_ub[self.k - self.begin].set_as(tmp + 1)

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
        self.add_tensor_size = self.tik_instance.Scalar(dtype=Constant.ORI_DEPENDENCY_INDICES_DTYPE,
                                                        name="add_tensor_size")
        self.add_tensor_size.set_as(((self.embedding_dim - 1) // self.grad_each_block + 1) * self.grad_each_block)
        if self.dtype_grad != "bfloat16":
            self.add_tensor = self.tik_instance.Tensor(
                self.dtype_grad, (1, self.add_tensor_size), name="add_tensor", scope=tik.scope_ubuf)
        else:
            self.add_tensor = self.tik_instance.Tensor(
                "float32", (1, self.add_tensor_size), name="add_tensor", scope=tik.scope_ubuf)
            self.add_tensor_temp = self.tik_instance.Tensor(
                "bfloat16", (1, self.add_tensor_size), name="add_tensor_temp", scope=tik.scope_ubuf)
            
        self.scale_int = self.tik_instance.Scalar(dtype=Constant.ORI_DEPENDENCY_INDICES_DTYPE)
        self.scale_float_tmp = self.tik_instance.Scalar(init_value=1.0, dtype=Constant.RELIES_ON_FLOAT32,
                                                        name='scale_float_tmp')
        # Define k, the scalar used to index the elements of indicators
        self.k = self.tik_instance.Scalar(dtype=Constant.ORI_DEPENDENCY_INDICES_DTYPE)
        # Move indexes and grad blocks from gm to ub
        with self.tik_instance.for_range(0, self.numel_indices // self.ub_indices_size) as i1:
            self.tik_instance.data_move(self.indices_ub, self.indices[i1 * self.ub_indices_size], 0, 1,
                                        self.move_length, 0, 0)
            self.tik_instance.data_move(self.grad_ub, self.grad[i1 * self.ub_indices_size * self.embedding_dim],
                                        0, 1, self.ub_indices_size * self.embedding_dim // self.grad_each_block, 0, 0)
            self.add_same_word_grad_need_scale(self.ub_indices_size)
        self.remaining_compute_grad_weight_need_scale()

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
        with self.tik_instance.if_scope(self.numel_indices % self.ub_indices_size != 0):
            offset_indices_move = self.tik_instance.Scalar("int32", name='offset_indices_move')
            burst_len_indices = self.tik_instance.Scalar("int32", name='burst_len_indices')
            offset_grad_move = self.tik_instance.Scalar("int32", name='offset_grad_move')
            burst_len_grad = self.tik_instance.Scalar("int32", name='burst_len_grad')
            offset_indices_move.set_as(self.numel_indices //
                                       self.ub_indices_size *
                                       self.ub_indices_size)
            burst_len_indices.set_as((self.numel_indices %
                                     self.ub_indices_size - 1) //
                                     self.indices_each_block + 1)
            offset_grad_move.set_as(self.numel_indices //
                                    self.ub_indices_size *
                                    self.ub_indices_size *
                                    self.embedding_dim)
            burst_len_grad.set_as((self.numel_indices %
                                  self.ub_indices_size *
                                  self.embedding_dim - 1) //
                                  self.grad_each_block + 1)

            self.tik_instance.data_move(self.indices_ub, self.indices[offset_indices_move],
                                        0, 1, burst_len_indices, 0, 0)
            self.tik_instance.data_move(self.grad_ub, self.grad[offset_grad_move], 0, 1, burst_len_grad, 0, 0)
            self.add_same_word_grad_need_scale(self.numel_indices % self.ub_indices_size)

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
                    self.scale_int.set_as(self.counts_ub[self.k - self.begin])
                    self.tik_instance.scalar_conv('', self.scale_float_tmp, self.scale_int)
                    numerator = 1.0
                    self.scale_float_tmp.set_as(numerator / self.scale_float_tmp)
                    self.scale_float = self.tik_instance.Scalar(self.dtype_grad, name='scale_float')
                    if self.scale_float.dtype != "float32":
                        self.tik_instance.scalar_conv('', self.scale_float, self.scale_float_tmp)
                    else:
                        self.scale_float.set_as(self.scale_float_tmp)
                    self.dup_value(self.add_tensor, self.add_tensor_size)
                    with self.tik_instance.if_scope(self.embedding_dim < self.grad_each_block):
                        self.cac_em_dim_less_than_eight()
                    with self.tik_instance.else_scope():
                        self.cac_em_dim_more_than_eight()

    def cac_em_dim_less_than_eight(self):
        """
        caculate the rersult of grad_weight

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.cac_em_dim_one_by_one(self.embedding_dim)
        self.tik_instance.set_atomic_add(1)
        self.tik_instance.data_move(self.grad_weight[self.k * self.embedding_dim], self.add_tensor,
                                    0, 1, 1, 0, 0)
        self.tik_instance.set_atomic_add(0)

    def cac_em_dim_more_than_eight(self):
        """
        caculate the rersult of grad_weight

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        cal_range = self.tik_instance.Scalar(Constant.ORI_DEPENDENCY_INDICES_DTYPE, name="cal_range", init_value=0)
        with self.tik_instance.if_scope((self.index * self.embedding_dim) % self.grad_each_block != 0):
            cal_range.set_as(((self.index * self.embedding_dim) //
                            self.grad_each_block + 1) * self.grad_each_block -
                            self.index * self.embedding_dim)
            self.cac_em_dim_one_by_one(cal_range)
            self.tik_instance.set_atomic_add(1)
            self.tik_instance.data_move(self.grad_weight[self.k * self.embedding_dim], self.add_tensor,
                                        0, 1, self.add_tensor_size // self.grad_each_block, 0, 0)
            self.tik_instance.set_atomic_add(0)
            self.dup_value(self.add_tensor, self.add_tensor_size)
        with self.tik_instance.if_scope((self.embedding_dim - cal_range) // self.vector_mask_max_grad > 0):
            self.tik_instance.vec_muls(self.vector_mask_max_grad, self.add_tensor,
                                    self.grad_ub[self.index * self.embedding_dim + cal_range],
                                    self.scale_float,
                                    (self.embedding_dim - cal_range) // self.vector_mask_max_grad, 8, 8)
        with self.tik_instance.if_scope((self.embedding_dim - cal_range) % self.vector_mask_max_grad > 0):
            self.tik_instance.vec_muls((self.embedding_dim - cal_range) % self.vector_mask_max_grad,
                                    self.add_tensor[(self.embedding_dim - cal_range) //
                                                    self.vector_mask_max_grad *
                                                    self.vector_mask_max_grad],
                                    self.grad_ub[self.index * self.embedding_dim +
                                                    cal_range + (self.embedding_dim - cal_range) //
                                                    self.vector_mask_max_grad * self.vector_mask_max_grad],
                                    self.scale_float, 1, 8, 8)
        self.tik_instance.set_atomic_add(1)
        self.tik_instance.data_move(self.grad_weight[self.k * self.embedding_dim + cal_range], self.add_tensor, 
                                    0, 1, self.add_tensor_size // self.grad_each_block, 0, 0)
        self.tik_instance.set_atomic_add(0)

    def cac_em_dim_one_by_one(self, cal_range):
        """
        caculate the rersult one_by_one

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.dtype_bytes_size_grad != Constant.FOUR_BYTES:
            grad_idx_data = self.tik_instance.Scalar(Constant.RELIES_ON_FLOAT32, name='grad_idx_data')
            grad_mul_data = self.tik_instance.Scalar(Constant.RELIES_ON_FLOAT32, name='grad_mul_data')
            tmp = self.tik_instance.Scalar(self.dtype_grad, name='tmp')
            with self.tik_instance.for_range(0, cal_range) as j:
                tmp.set_as(self.grad_ub[self.index * self.embedding_dim + j])
                self.tik_instance.scalar_conv('', grad_idx_data, tmp)
                grad_mul_data.set_as(grad_idx_data * self.scale_float_tmp)
                self.tik_instance.scalar_conv('', tmp, grad_mul_data)
                self.add_tensor[j].set_as(tmp)
        else:
            grad_idx_data = self.tik_instance.Scalar(self.dtype_grad, name='grad_idx_data')
            grad_mul_data = self.tik_instance.Scalar(self.dtype_grad, name='grad_mul_data')
            with self.tik_instance.for_range(0, cal_range) as j:
                grad_idx_data.set_as(self.grad_ub[self.index * self.embedding_dim + j])
                grad_mul_data.set_as(grad_idx_data * self.scale_float)
                self.add_tensor[j].set_as(grad_mul_data)

    def select_data_move_method(self, dst, src, data_move_pad_length, data_move_length):
        if cce.api_check_support("tik.data_move_pad") and src.dtype != "int64" and dst.dtype != "int64":
            self.tik_instance.data_move_pad(dst, src, nburst=1,
                                            burst=data_move_pad_length,
                                            dst_gap=0, src_gap=0, right_padding=0, left_padding=0,
                                            padding_value=None)
        else:
            self.tik_instance.data_move(dst, src, 0, 1, data_move_length, 0, 0)

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
        self.k = self.tik_instance.Scalar(dtype=Constant.ORI_DEPENDENCY_INDICES_DTYPE)
        with self.tik_instance.if_scope(core_index == self.core_used - 1):
            self.ranges.set_as(self.ele_last_core)
        with self.tik_instance.else_scope():
            self.ranges.set_as(self.ele_not_last_core)
        self.block_ub = self.tik_instance.Tensor(self.dtype_grad, (self.grad_each_block,), name="block_ub",
                                                scope=tik.scope_ubuf)
        self.dup_value(self.block_ub, self.grad_each_block)
        # Move indexes and grad blocks from gm to ub
        with self.tik_instance.if_scope(tik.all(self.embedding_dim > self.grad_each_block,
                                                self.embedding_dim <= self.count,
                                                self.embedding_dim % self.grad_each_block != 0)):
            self.base_compute_no_scale_not_align(core_index)
        with self.tik_instance.elif_scope(self.embedding_dim > self.count):
            self.base_compute_no_scale_huge_embedding_dim(core_index)
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(tik.all(self.numel_indices >= Constant.THRESHOLD_NUM,
                                                    self.embedding_dim == Constant.DIM_SIX,
                                                    self.num_weights <= Constant.HIGH_PERFORMANCE_NUM)):
                with self.tik_instance.new_stmt_scope():
                    self.add_y_ub = self.tik_instance.Tensor(self.dtype_grad, (self.num_weights, 8), name="add_y_ub",
                                                            scope=tik.scope_ubuf)
                    self.dup_value(self.add_y_ub, self.num_weights * self.grad_each_block)
                    with self.tik_instance.if_scope(self.ranges // self.ub_indices_size > 0):
                        with self.tik_instance.for_range(0, self.ranges // self.ub_indices_size) as i1:
                            self.tik_instance.data_move(self.indices_ub,
                                                        self.indices[core_index * self.ele_not_last_core +
                                                                     i1 * self.ub_indices_size], 0, 1,
                                                        self.move_length, 0, 0)
                            with self.tik_instance.for_range(0, self.ub_indices_size) as i2:
                                self.tik_instance.data_move(self.grad_ub[i2 * self.grad_each_block],
                                                            self.grad[(core_index * self.ele_not_last_core + i1 *
                                                                    self.ub_indices_size + i2) * self.embedding_dim],
                                                            0, 1, 1, 0, 0)
                            self.add_same_word_grad_not_need_scale_with_dim_six(self.ub_indices_size)
                    self.remaining_compute_not_need_scale_with_dim_six(core_index)
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(self.ranges // self.ub_indices_size > 0):
                    with self.tik_instance.for_range(0, self.ranges // self.ub_indices_size) as i1:
                        self.select_data_move_method(self.indices_ub,
                                                    self.indices[core_index * self.ele_not_last_core +
                                                                i1 * self.ub_indices_size],
                                                    self.ub_indices_size * self.dtype_bytes_size_indices,
                                                    self.move_length)
                        self.select_data_move_method(self.grad_ub,
                                                    self.grad[(core_index * self.ele_not_last_core + i1 *
                                                              self.ub_indices_size) * self.embedding_dim],
                                                    self.ub_indices_size * self.embedding_dim *
                                                    self.dtype_bytes_size_grad,
                                                    self.ub_indices_size *
                                                    self.embedding_dim // self.grad_each_block)
                        self.add_same_word_grad_not_need_scale(self.ub_indices_size)
                self.remaining_compute_grad_weight_not_need_scale(core_index)

    def base_compute_no_scale_huge_embedding_dim(self, core_index):
        """
        base_compute_no_scale_huge_embedding_dim
        """
        with self.tik_instance.if_scope(self.ranges // self.ub_indices_size > 0):
            with self.tik_instance.for_range(0, self.ranges // self.ub_indices_size) as i:
                self.select_data_move_method(self.indices_ub,
                                            self.indices[core_index * self.ele_not_last_core +
                                                        i * self.ub_indices_size],
                                            self.ub_indices_size * self.dtype_bytes_size_indices,
                                            self.move_length)
                self.add_same_grad_no_scale_huge_embedding_dim(core_index, self.ub_indices_size, i)
        self.remain_compute_no_scale_huge_embedding_dim(core_index)

    def remain_compute_no_scale_huge_embedding_dim(self, core_index):
        """
        remain_compute_no_scale_huge_embedding_dim
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
            self.select_data_move_method(self.indices_ub,
                                         self.indices[offset_indices_move],
                                         self.ranges % self.ub_indices_size * self.dtype_bytes_size_indices,
                                         burst_len_indices)
            self.add_same_grad_no_scale_huge_embedding_dim(core_index, self.ranges % self.ub_indices_size,
                                                           self.ranges // self.ub_indices_size)

    def base_compute_no_scale_not_align(self, core_index):
        """
        base_compute_no_scale_not_align
        """
        align_length = (self.embedding_dim - 1) // self.grad_each_block + 1
        with self.tik_instance.if_scope(self.ranges // self.ub_indices_size > 0):
            with self.tik_instance.for_range(0, self.ranges // self.ub_indices_size) as i:
                self.select_data_move_method(self.indices_ub,
                                            self.indices[core_index * self.ele_not_last_core +
                                                        i * self.ub_indices_size],
                                            self.ub_indices_size * self.dtype_bytes_size_indices,
                                            self.move_length)
                with self.tik_instance.for_range(0, self.ub_indices_size) as j:
                    self.select_data_move_method(self.grad_ub[j * align_length * self.grad_each_block],
                                                self.grad[(core_index * self.ele_not_last_core + i *
                                                self.ub_indices_size + j) * self.embedding_dim],
                                                self.embedding_dim * self.dtype_bytes_size_grad,
                                                align_length)
                self.add_same_word_grad_not_need_scale(self.ub_indices_size)
        self.remain_compute_no_scale_not_align(core_index, align_length)

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
        with self.tik_instance.if_scope(self.ranges % self.ub_indices_size != 0):
            offset_indices_move = self.tik_instance.Scalar('int32', name='offset_indices_move')
            offset_indices_move.set_as(core_index * self.ele_not_last_core + self.ranges // self.ub_indices_size \
                                                                        * self.ub_indices_size)
            burst_len_indices = self.tik_instance.Scalar('int32', name='burst_len_indices')
            burst_len_grad = self.tik_instance.Scalar('int32', name='burst_len_grad')
            offset_grad_move = self.tik_instance.Scalar('int32', name='offset_grad_move')
            burst_len_indices.set_as((self.ranges % self.ub_indices_size - 1) // self.indices_each_block + 1)
            offset_grad_move.set_as((core_index * self.ele_not_last_core + self.ranges // self.ub_indices_size *
                                self.ub_indices_size) * self.embedding_dim)
            burst_len_grad.set_as(((self.ranges % self.ub_indices_size) * self.embedding_dim - 1)
                                  // self.grad_each_block + 1)
            self.tik_instance.data_move(self.indices_ub, self.indices[offset_indices_move],
                                        0, 1, burst_len_indices, 0, 0)
            self.tik_instance.data_move(self.grad_ub, self.grad[offset_grad_move], 0, 1, burst_len_grad, 0, 0)
            self.add_same_word_grad_not_need_scale(self.ranges % self.ub_indices_size)

    def remaining_compute_not_need_scale_with_dim_six(self, core_index):
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
        with self.tik_instance.if_scope(self.ranges % self.ub_indices_size != 0):
            offset_indices_move = self.tik_instance.Scalar(Constant.ORI_DEPENDENCY_INDICES_DTYPE,
                                                           name='offset_indices_move')
            offset_indices_move.set_as(core_index * self.ele_not_last_core + self.ranges // self.ub_indices_size \
                                                                        * self.ub_indices_size)
            burst_len_indices = self.tik_instance.Scalar(Constant.ORI_DEPENDENCY_INDICES_DTYPE,
                                                         name='burst_len_indices')
            offset_grad_move = self.tik_instance.Scalar(Constant.ORI_DEPENDENCY_INDICES_DTYPE,
                                                        name='offset_grad_move')
            burst_len_indices.set_as((self.ranges % self.ub_indices_size - 1) // self.indices_each_block + 1)
            offset_grad_move.set_as((core_index * self.ele_not_last_core + self.ranges // self.ub_indices_size *
                                self.ub_indices_size) * self.embedding_dim)
            self.select_data_move_method(self.indices_ub,
                                        self.indices[offset_indices_move],
                                        self.ranges % self.ub_indices_size * self.dtype_bytes_size_indices,
                                        burst_len_indices)
            with self.tik_instance.for_range(0, self.ranges % self.ub_indices_size) as i:
                self.select_data_move_method(self.grad_ub[i * self.grad_each_block],
                                            self.grad[offset_grad_move + i * self.embedding_dim],
                                            self.embedding_dim * self.dtype_bytes_size_grad, 1)
            self.add_same_word_grad_not_need_scale_with_dim_six(self.ranges % self.ub_indices_size)
            self.transpose_and_move_data_out_with_six()

    def add_same_grad_no_scale_huge_embedding_dim(self, core_index, total, ind):
        """
        add_same_grad_no_scale_huge_embedding_dim
        """
        with self.tik_instance.for_range(0, total) as self.index:
            self.k.set_as(self.indices_ub[self.index])
            # Determine whether to process in the current core based on k
            with self.tik_instance.if_scope(tik.all(self.k >= 0, self.k < self.num_weights, 
                                                    self.k != self.padding_idx)):
                # Determine whether a row of grad exceeds one UB based on grad_movement_times.
                with self.tik_instance.if_scope(self.grad_movement_times > 0):
                    # Processing a complete block.
                    with self.tik_instance.for_range(0, self.grad_movement_times) as times:
                        self.select_data_move_method(self.grad_ub,
                                                    self.grad[(core_index * self.ele_not_last_core + ind *
                                                               self.ub_indices_size + self.index) * 
                                                               self.embedding_dim + times * self.ub_grad_size],
                                                    self.ub_grad_size * self.dtype_bytes_size_grad,
                                                    self.ub_grad_size // self.grad_each_block)
                        self.tik_instance.set_atomic_add(1)
                        self.select_data_move_method(self.grad_weight[self.k * self.embedding_dim + times * 
                                                                      self.ub_grad_size],
                                                    self.grad_ub,
                                                    self.ub_grad_size * self.dtype_bytes_size_grad,
                                                    self.ub_grad_size // self.grad_each_block)
                        self.tik_instance.set_atomic_add(0)
                    # Processing the tail block aligned with the block
                    with self.tik_instance.if_scope(self.embedding_dim % self.grad_each_block == 0):
                        self.select_data_move_method(self.grad_ub,
                                                    self.grad[(core_index * self.ele_not_last_core + ind *
                                                        self.ub_indices_size + self.index) * 
                                                        self.embedding_dim + self.grad_movement_times * 
                                                        self.ub_grad_size],
                                                    self.grad_last_movement * self.dtype_bytes_size_grad,
                                                    self.grad_last_movement // self.grad_each_block)
                        self.tik_instance.set_atomic_add(1)
                        self.select_data_move_method(self.grad_weight[self.k * self.embedding_dim + 
                                                                    self.grad_movement_times * self.ub_grad_size],
                                                    self.grad_ub,
                                                    self.grad_last_movement * self.dtype_bytes_size_grad,
                                                    self.grad_last_movement // self.grad_each_block)
                        self.tik_instance.set_atomic_add(0)
                    # Processing the unaligned tail block with the block  
                    with self.tik_instance.elif_scope(self.embedding_dim % self.grad_each_block != 0):
                        if cce.api_check_support("tik.data_move_pad"):
                            self.tik_instance.data_move_pad(self.grad_ub,
                                                            self.grad[(core_index * self.ele_not_last_core + ind *
                                                            self.ub_indices_size + self.index) *
                                                            self.embedding_dim +
                                                            self.grad_movement_times * self.ub_grad_size],
                                                            nburst=1,
                                                            burst=self.embedding_dim % self.ub_grad_size *
                                                            self.dtype_bytes_size_grad,
                                                            dst_gap=0,
                                                            src_gap=0,
                                                            right_padding=0,
                                                            left_padding=0,
                                                            padding_value=None)
                        else:
                            self.tik_instance.data_move(self.grad_ub,
                                                    self.grad[(core_index * self.ele_not_last_core + ind *
                                                            self.ub_indices_size + self.index) * 
                                                            self.embedding_dim + 
                                                            self.grad_movement_times * self.ub_grad_size],
                                                    0, 1, self.grad_last_movement // self.grad_each_block + 1, 0, 0)
                        with self.tik_instance.for_range(0, self.grad_each_block - 
                                                        (self.embedding_dim % self.grad_each_block)) as i:
                            self.grad_ub[self.embedding_dim % self.ub_grad_size + i].set_as(0)
                        self.tik_instance.set_atomic_add(1)
                        self.tik_instance.data_move(self.grad_weight[self.k * self.embedding_dim + 
                                                                    self.grad_movement_times * 
                                                                    self.ub_grad_size], self.grad_ub, 
                                                    0, 1, self.grad_last_movement // self.grad_each_block, 
                                                    0, 0)
                        self.tik_instance.set_atomic_add(0)
                # The line of grad is within the upper bound (UB).
                with self.tik_instance.elif_scope(self.grad_movement_times == 0):
                    # Processing the tail block aligned with the block
                    with self.tik_instance.if_scope(self.embedding_dim % self.grad_each_block == 0):
                        self.tik_instance.data_move(self.grad_ub,
                                                self.grad[(core_index * self.ele_not_last_core + ind *
                                                        self.ub_indices_size + self.index) * self.embedding_dim],
                                                0, 1, self.embedding_dim // self.grad_each_block, 0, 0)
                        self.tik_instance.set_atomic_add(1)
                        self.tik_instance.data_move(self.grad_weight[self.k * self.embedding_dim],
                                                    self.grad_ub, 0,
                                                    1, self.embedding_dim // self.grad_each_block, 0, 0)
                        self.tik_instance.set_atomic_add(0)
                    # Processing the unaligned tail block with the block    
                    with self.tik_instance.elif_scope(self.embedding_dim % self.grad_each_block != 0):
                        self.tik_instance.data_move(self.grad_ub,
                                                    self.grad[(core_index * self.ele_not_last_core + ind * \
                                                               self.ub_indices_size + self.index) * self.embedding_dim],
                                                    0, 1, self.embedding_dim // self.grad_each_block + 1, 0, 0)
                        # Padding with zeros
                        ub_offset = self.embedding_dim // self.grad_each_block * self.grad_each_block
                        gm_offset = (core_index * self.ele_not_last_core + ind * self.ub_indices_size + self.index) * \
                                    self.embedding_dim + self.embedding_dim // \
                                    self.grad_each_block * self.grad_each_block
                        r_padding_num = self.grad_each_block - (self.embedding_dim % self.grad_each_block)
                        dtype_byte_size = EmbeddingDenseGrad.get_dtype_size(self.dtype_grad)
                        tail_num = self.embedding_dim % self.grad_each_block * dtype_byte_size
                        if cce.api_check_support("tik.data_move_pad"):
                            self.tik_instance.data_move_pad(self.grad_ub[ub_offset],
                                                            self.grad[gm_offset],
                                                            nburst=1, 
                                                            burst=tail_num, 
                                                            dst_gap=0,
                                                            src_gap=0,
                                                            right_padding=r_padding_num,
                                                            left_padding=0,
                                                            padding_value=0)
                        else:
                            with self.tik_instance.for_range(0, self.grad_each_block - 
                                                            (self.embedding_dim % self.grad_each_block)) as i:
                                self.grad_ub[self.embedding_dim + i].set_as(0)
                                
                        self.tik_instance.set_atomic_add(1)
                        self.tik_instance.data_move(self.grad_weight[self.k * self.embedding_dim],
                                                    self.grad_ub, 0,
                                                    1, self.embedding_dim // self.grad_each_block + 1, 0, 0)
                        self.tik_instance.set_atomic_add(0)

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
            self.tik_instance.data_move(self.indices_ub, self.indices[offset_indices_move],
                                        0, 1, burst_len_indices, 0, 0)
            with self.tik_instance.for_range(0, self.ranges % self.ub_indices_size) as i:
                self.select_data_move_method(self.grad_ub[i * align_size * self.grad_each_block],
                                            self.grad[(core_index * self.ele_not_last_core + self.ranges //
                                            self.ub_indices_size * self.ub_indices_size + i)
                                            * self.embedding_dim],
                                            self.embedding_dim * self.dtype_bytes_size_grad,
                                            align_size)
            self.add_same_word_grad_not_need_scale(self.ranges % self.ub_indices_size)

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
        with self.tik_instance.for_range(0, total) as self.index:
            self.k.set_as(self.indices_ub[self.index])
            with self.tik_instance.if_scope(self.k != self.padding_idx):
                with self.tik_instance.if_scope(tik.all(self.k >= 0, self.k < self.num_weights)):
                    with self.tik_instance.if_scope(tik.all(self.embedding_dim >= self.grad_each_block,
                                        self.embedding_dim % self.grad_each_block == 0)):
                        self.tik_instance.set_atomic_add(1)
                        self.tik_instance.data_move(self.grad_weight[self.k * self.embedding_dim],
                                                    self.grad_ub[self.index * self.embedding_dim], 0,
                                                    1, self.embedding_dim // self.grad_each_block, 0, 0)
                        self.tik_instance.set_atomic_add(0)
                    with self.tik_instance.elif_scope(tik.all(self.embedding_dim > self.grad_each_block,
                                                    self.embedding_dim % self.grad_each_block != 0)):
                        align_size = ((self.embedding_dim - 1) // self.grad_each_block + 1) * self.grad_each_block
                        align_ub = self.tik_instance.Tensor(self.dtype_grad, (align_size,),
                                                            name='align_ub', scope=tik.scope_ubuf)
                        self.dup_value(align_ub, align_size)
                        self.tik_instance.data_move(align_ub, self.grad_ub[self.index * align_size], 0, 1,
                                                    self.embedding_dim // self.grad_each_block, 0, 0)
                        with self.tik_instance.for_range(0, self.embedding_dim % self.grad_each_block) as i:
                            align_ub[self.embedding_dim // self.grad_each_block *
                                    self.grad_each_block + i].set_as(self.grad_ub[self.index * align_size +
                                                        self.embedding_dim // self.grad_each_block *
                                                        self.grad_each_block + i])
                        self.tik_instance.set_atomic_add(1)
                        self.select_data_move_method(self.grad_weight[self.k * self.embedding_dim],
                                                     align_ub, self.embedding_dim * self.dtype_bytes_size_grad,
                                                     align_size // self.grad_each_block)
                        self.tik_instance.set_atomic_add(0)
                    with self.tik_instance.elif_scope(self.embedding_dim < self.grad_each_block):
                        with self.tik_instance.for_range(0, self.embedding_dim) as i:
                            self.block_ub[i].set_as(self.grad_ub[self.index * self.embedding_dim + i])
                        self.tik_instance.set_atomic_add(1)
                        self.select_data_move_method(self.grad_weight[self.k * self.embedding_dim],
                                                     self.block_ub,
                                                     self.embedding_dim * self.dtype_bytes_size_grad, 1)
                        self.tik_instance.set_atomic_add(0)

    def add_same_word_grad_not_need_scale_with_dim_six(self, total):
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
                with self.tik_instance.if_scope(tik.all(self.k >= 0, self.k < self.num_weights)):
                    self.tik_instance.vec_add(self.grad_each_block, self.add_y_ub[self.k *
                            self.grad_each_block], self.add_y_ub[self.k * self.grad_each_block],
                            self.grad_ub[self.index * 8], 1, 8, 8, 8)

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
        with self.tik_instance.if_scope(self.embedding_dim * scale_32_to_16 * align_num_weights < 512):
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
        self.tik_instance.set_atomic_add(1)
        with self.tik_instance.for_range(0, self.num_weights) as ind:
            self.select_data_move_method(self.grad_weight[ind * self.embedding_dim],
                                        add_y_ub_fp32[ind * self.grad_each_block],
                                        self.embedding_dim * self.dtype_bytes_size_grad, 1)
        self.tik_instance.set_atomic_add(0)


# 'pylint: disable=too-many-arguments
@register_operator("EmbeddingDenseGrad")
def embedding_dense_grad(
        grad,
        indices,
        y,
        num_weights,
        padding_idx,
        scale_grad_by_freq,
        kernel_name="embedding_dense_grad"):
    """
    the main function of op embedding_dense_grad

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
    embedding_dense_grad_instance = EmbeddingDenseGrad(
        grad, indices, y, num_weights, padding_idx, scale_grad_by_freq, kernel_name)
    tik_instance = embedding_dense_grad_instance.embedding_dense_grad_compute_tiling()
    return tik_instance
