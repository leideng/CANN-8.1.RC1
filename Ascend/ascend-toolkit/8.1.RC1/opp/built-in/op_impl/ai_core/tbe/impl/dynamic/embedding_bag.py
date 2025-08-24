#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2022 Huawei Technologies Co., Ltd
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
embedding_bag
"""
from impl.util import util_tik_comm_func
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl import common_util

# 'pylint: disable=too-many-lines,too-many-instance-attributes,too-many-statements
# 'pylint: disable=too-many-arguments,unused-argument,too-many-locals,too-many-branches


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    TASK = 257
    # process 128 weight at one time
    WEIGHT_NUM_MIN = 128
    MAX_BAG_SIZE = 8
    MAX_INT32 = 2**31 - 1
    TILING_NUM = 8
    NUM_EMBEDDINGS_IDX = 0
    EMBEDDING_DIM_IDX = 1
    BATCH = 2
    IS_SINGLE_INDICES_IDX = 3
    INDICES_ZERO_DIM = 4
    INDICES_ONE_DIM = 5
    OFFSETS_SHAPE = 6
    INDEX_ATTR_PADDING_IDX = 7
    MIN_FP32 = -3.4e38
    MIN_FP16 = -65500.0
    BLOCK_SIZE = 32
    REPEAT_STRIDE_EIGHT = 8
    MAX_REPEAT_NUM = 255
    SINGLE_INDICES = 1
    TEN_THOUSAND = 10000
    MAX_MOVE_NUM = 512

    # Elements num per block of 8bit, 16bit, 32bit and 64bit dtype
    B8_NUM_PER_BLOCK = 32
    B16_NUM_PER_BLOCK = 16
    B32_NUM_PER_BLOCK = 8
    B64_NUM_PER_BLOCK = 4
    # Mum of processed 8bit, 16bit, 32bit and 64bit dtype elements each iteration of vector calculation
    B8_VEC_MASK = 256
    B16_VEC_MASK = 128
    B32_VEC_MASK = 64
    B64_VEC_MASK = 32
    T = 32
    MASK = 64
    INT32 = "int32"
    INT8 = "int8"


class DataMover:
    def moving(self, src, move_num, data_offset, i):
        pass

    def move(self, move_num):
        pass


class LittleDataMover(DataMover):
    def __init__(self, tik_instance, weight_dtype, ub_size, dst, data_offset):
        self.tik_instance = tik_instance
        self.output_ub = self.tik_instance.Tensor(weight_dtype, ub_size, name="output_ub", scope=tik.scope_ubuf)
        self.dst = dst
        self.data_offset = data_offset

    def moving(self, src, move_num, data_offset, i):
        with self.tik_instance.for_range(0, move_num) as j:
            self.output_ub[i, j].set_as(src[j])

    def move(self, data_move_pad_length, move_num):
        if tbe_platform.api_check_support("tik.data_move_pad"):
            self.tik_instance.data_move_pad(self.dst[self.data_offset], self.output_ub, nburst=1,
                                            burst=data_move_pad_length,
                                            dst_gap=0, src_gap=0, right_padding=0, left_padding=0,
                                            padding_value=None)
        else:
            util_tik_comm_func.ub2gm(self.tik_instance, self.dst[self.data_offset], self.output_ub, move_num)


class HugeDataMover(DataMover):
    def __init__(self, tik_instance, weight_dtype, dst, offset_counts, num_per_block):
        self.tik_instance = tik_instance
        self.weight_dtype = weight_dtype
        self.dst = dst
        self.offset_counts = offset_counts
        self.num_per_block = num_per_block

    def moving(self, src, move_num, data_offset, i):
        tail_block = self.tik_instance.Tensor(self.weight_dtype, (self.num_per_block,),
                                                name="tail_block", scope=tik.scope_ubuf)
        with self.tik_instance.if_scope(i != (self.offset_counts - 1 - 1)):
            self.tik_instance.data_move(self.dst[(data_offset + i) * move_num], src, 0, 1,
                                        (move_num + self.num_per_block - 1) // self.num_per_block, 0, 0)
        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, self.num_per_block) as j:
                tail_block[j].set_as(src[move_num - self.num_per_block + j])
    
            self.tik_instance.data_move(self.dst[(data_offset + i) * move_num], src, 0, 1,
                                                        move_num // self.num_per_block, 0, 0)
            self.tik_instance.data_move(self.dst[(data_offset + i + 1) * move_num - self.num_per_block],
                                                                                        tail_block, 0, 1, 1, 0, 0)


class EmbeddingBag:
    def __init__(self, weight, indices, offsets, per_sample_weights, mode, scale_grid_by_freq, sparse,
                                                                        include_last_offset, padding_idx):
        """
        Init EmbeddingBag base parameters

        Returns
        -------
        None
        """
        # define general var
        self.tik_instance = tik.Tik()
        self.aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)        
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.input_gm_list = []
        self.output_gm_list = []

        self.num_embeddings = self.tik_instance.Scalar("int32", name="num_embeddings")
        self.embedding_dim = self.tik_instance.Scalar("int32", name="embedding_dim")
        self.batch = self.tik_instance.Scalar("int32", name="batch")
        self.is_single_indices = self.tik_instance.Scalar("int32", name="is_single_indices")
        self.indices_zero_dim = self.tik_instance.Scalar("int32", name="indices_zero_dim")
        self.indices_one_dim = self.tik_instance.Scalar("int32", name="indices_one_dim")
        self.offsets_shape = self.tik_instance.Scalar("int32", name="offsets_shape")
        self.per_sample_weights_shape = self.tik_instance.Scalar("int32", name="per_sample_weights_shape")
        self.padding_idx = self.tik_instance.Scalar("int32", name="padding_idx")
        
        self.tiling_gm = self.tik_instance.Tensor("int32", [Constant.TILING_NUM], name="tiling_gm", scope=tik.scope_gm)
        self.get_tiling_args()

        self.weight_dtype = weight.get("dtype")
        self.output_dtype = weight.get("dtype")
        self.indices_dtype = indices.get("dtype")
        self.offsets_dtype = indices.get("dtype")
        if self.weight_dtype in ["bfloat16", "float16"]:
            self.weight_dtype_size = 2
        else:
            self.weight_dtype_size = common_util.get_data_size(self.weight_dtype)
        self.indices_dtype_size = common_util.get_data_size(self.indices_dtype)
        self.weight_num = self.tik_instance.Scalar("int32", name="weight_num",
                                                   init_value=self.num_embeddings * self.embedding_dim)
        self.output_num = self.tik_instance.Scalar("int32", name="output_num",
                                                   init_value=self.batch * self.embedding_dim)
        self.max_indices_num = self.tik_instance.Scalar(self.offsets_dtype, name="output_num",
                                                        init_value=self.batch * self.embedding_dim)
        self.indices_num = self.tik_instance.Scalar(self.indices_dtype, name="indices_num")
        self.indices_num.set_as(self.indices_zero_dim * self.indices_one_dim)
        self.task_num = self.tik_instance.Scalar("int32", name="task_num")
        if per_sample_weights is None:
            self.has_per_sample_weights = False
        else:
            self.has_per_sample_weights = True
            self.per_sample_weights_dtype = per_sample_weights.get("dtype")

        if tbe_platform.api_check_support("tik.data_move_pad"):
            self.conv_mode = "round"
        else:
            self.conv_mode = ""

        # attr
        self.mode = mode
        self.sparse = sparse
        self.scale_grid_by_freq = scale_grid_by_freq
        self.include_last_offset = include_last_offset
        # define var
        self.max_bag_size = self.tik_instance.Scalar("int32", init_value=self.get_max_bag_size())

        self.weight_ub_temp = None
        self.indices_bag_l1 = None
        self.output_ub_temp = None
        self.indices_bag_gm = None
        self.per_sample_index_gm = None
        self.bag_divided_ub = None
        self.tail_size = 0

        self.zero_scalar = self.tik_instance.Scalar(dtype="int32", init_value=0)
        self.neg_one_scalar = self.tik_instance.Scalar(dtype="int32", init_value=-1)
        self.zero_fp32 = self.tik_instance.Scalar(dtype="float32", init_value=0)

        self.data_mover = DataMover()

    def get_tiling_args(self):
        """get_tiling_args"""
        tiling_ub = self.tik_instance.Tensor("int32", [Constant.TILING_NUM], name="tiling_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, 1, 0, 0)

        self.num_embeddings.set_as(tiling_ub[Constant.NUM_EMBEDDINGS_IDX])
        self.embedding_dim.set_as(tiling_ub[Constant.EMBEDDING_DIM_IDX])
        self.batch.set_as(tiling_ub[Constant.BATCH])

        self.is_single_indices.set_as(tiling_ub[Constant.IS_SINGLE_INDICES_IDX])
        self.indices_zero_dim.set_as(tiling_ub[Constant.INDICES_ZERO_DIM])
        self.indices_one_dim.set_as(tiling_ub[Constant.INDICES_ONE_DIM])
        self.offsets_shape.set_as(tiling_ub[Constant.OFFSETS_SHAPE])
        self.padding_idx.set_as(tiling_ub[Constant.INDEX_ATTR_PADDING_IDX])
        self.per_sample_weights_shape.set_as(tiling_ub[Constant.INDICES_ZERO_DIM])

    def select_data_move_method(self, dst, src, data_move_pad_length, data_move_length):
        if tbe_platform.api_check_support("tik.data_move_pad"):
            reinterpret_src = src.reinterpret_cast_to(Constant.INT8)
            reinterpret_dst = dst.reinterpret_cast_to(Constant.INT8)
            self.tik_instance.data_move_pad(reinterpret_dst, reinterpret_src, nburst=1,
                                            burst=data_move_pad_length,
                                            dst_gap=0, src_gap=0, right_padding=0, left_padding=0,
                                            padding_value=None)
        else:
            self.tik_instance.data_move(dst, src, 0, 1, data_move_length, 0, 0)

    def get_max_bag_size(self):
        """get max bag size"""
        res = self.tik_instance.Scalar("int32")
        with self.tik_instance.if_scope(self.is_single_indices == Constant.SINGLE_INDICES):
            res.set_as(Constant.MAX_BAG_SIZE)
        with self.tik_instance.else_scope():
            res.set_as(self.mem_aligned("int32", self.indices_one_dim))
        return res

    def init_tik_mem(self):
        """init tik gm mem"""
        # init gm input
        weight_gm = self.tik_instance.Tensor(self.weight_dtype, [Constant.MAX_INT32],
                                             name="weight_gm", scope=tik.scope_gm)
        indices_gm = self.tik_instance.Tensor(self.indices_dtype, [Constant.MAX_INT32],
                                              name="indices_gm", scope=tik.scope_gm)
        offsets_gm = self.tik_instance.Tensor(self.offsets_dtype, [Constant.MAX_INT32],
                                              name="offsets_gm", scope=tik.scope_gm)
        self.input_gm_list = [weight_gm, indices_gm, offsets_gm]
        if self.has_per_sample_weights:
            per_sample_weights_gm = self.tik_instance.Tensor(self.per_sample_weights_dtype, [Constant.MAX_INT32],
                                                             name="per_sample_weights_gm", scope=tik.scope_gm)
            self.input_gm_list.append(per_sample_weights_gm)

        # init gm output
        embedding_bag_output_gm = self.tik_instance.Tensor(self.output_dtype, [Constant.MAX_INT32],
                                                           name="embedding_bag_output_gm", 
                                                           scope=tik.scope_gm, is_atomic_add=True)
        offset2bag_gm = self.tik_instance.Tensor(self.offsets_dtype, [Constant.MAX_INT32],
                                                 name="offset2bag_gm", scope=tik.scope_gm, is_atomic_add=True)
        bag_size_gm = self.tik_instance.Tensor(self.offsets_dtype, [Constant.MAX_INT32],
                                               name="bag_size_gm", scope=tik.scope_gm, is_atomic_add=True)
        max_indices_gm = self.tik_instance.Tensor(self.indices_dtype, [Constant.MAX_INT32],
                                                  name="max_indices_gm", scope=tik.scope_gm, is_atomic_add=True)
        self.output_gm_list = [embedding_bag_output_gm, offset2bag_gm, bag_size_gm, max_indices_gm]

        # init temp tik gm
        self.indices_bag_gm = self.tik_instance.Tensor(self.indices_dtype, [Constant.MAX_INT32],
                                                       name="indices_bag_gm", scope=tik.scope_gm, is_workspace=True)
        if self.has_per_sample_weights:
            self.per_sample_index_gm = self.tik_instance.Tensor("int32", [Constant.MAX_INT32],
                                                                name="per_sample_index_gm", scope=tik.scope_gm,
                                                                is_workspace=True)

    def init_ub_mem(self):
        """init tik ub mem"""
        self.weight_ub_temp = self.tik_instance.Tensor(self.weight_dtype, (128,),
                                                       name="weight_ub_temp", scope=tik.scope_ubuf)
        with self.tik_instance.if_scope(self.weight_num <= Constant.WEIGHT_NUM_MIN):
            util_tik_comm_func.gm2ub(self.tik_instance, self.weight_ub_temp, self.input_gm_list[0], self.weight_num)
        with self.tik_instance.else_scope():
            util_tik_comm_func.gm2ub(self.tik_instance, self.weight_ub_temp,
                                     self.input_gm_list[0][self.weight_num - Constant.WEIGHT_NUM_MIN],
                                     Constant.WEIGHT_NUM_MIN)
        self.output_ub_temp = self.tik_instance.Tensor(self.weight_dtype, (128,),
                                                       name="output_ub_temp", scope=tik.scope_ubuf)
        util_tik_comm_func.tik_func_vector(self.tik_instance, self.output_ub_temp, -1, 128)

        # init tail ub for tail data move to gm
        self.tail_size = min_data_block(self.weight_dtype)

    def get_tik_instance(self):
        """get tik instance"""
        return self.tik_instance

    def build_tik_instance(self, kernel_name_value):
        """build_tik_instance"""
        tbe_context.get_context().add_compile_info(
            "vars",
            {
                "core_num": self.aicore_num,
                "has_per_sample_weights": self.has_per_sample_weights,
            },
        )

        opt_config = {"enable_const_fold": True}

        self.tik_instance.BuildCCE(kernel_name=kernel_name_value, inputs=self.input_gm_list,
                                   outputs=self.output_gm_list, flowtable=[self.tiling_gm],
                                   output_files_path=None, enable_l2=False, config=opt_config)

        return self.tik_instance

    def mem_aligned(self, dtype, in_num):
        """aligned mem for ub"""
        out_num = self.tik_instance.Scalar("int32")
        if dtype in ["int32", "float32"]:
            out_num.set_as(self.ceil_div(in_num, 8) * 8)
        elif dtype in ["float16", "bfloat16"]:
            out_num.set_as(self.ceil_div(in_num, 16) * 16)
        elif dtype in ["int64"]:
            out_num.set_as(self.ceil_div(in_num, 4) * 4)
        else:
            RuntimeError("dtype is not support !!")
        return out_num

    def burst_len_compute(self, dtype, in_num):
        """burst_len compute"""
        out_num = self.tik_instance.Scalar("int32")
        if dtype in ["int32", "float32"]:
            out_num.set_as(self.ceil_div(in_num, 8))

        elif dtype in ["float16", "bfloat16"]:
            out_num.set_as(self.ceil_div(in_num, 16))
        elif dtype in ["int64"]:
            out_num.set_as(self.ceil_div(in_num, 4))
        else:
            RuntimeError("dtype is not support !!")
        return out_num

    def tail_lens_cal(self, dtype, in_num):
        """tail_lens cal"""
        out_num = self.tik_instance.Scalar("int32")
        if dtype in ["int32", "float32"]:
            out_num.set_as(in_num % 8)
        elif dtype in ["float16", "bfloat16"]:
            out_num.set_as(in_num % 16)
        elif dtype in ["int64"]:
            out_num.set_as(in_num % 4)
        else:
            RuntimeError("dtype is not support !!")
        return out_num

    def ceil_div(self, value, factor):
        """Compute the smallest integer value that is greater than
        or equal to value/factor
        """
        result = self.tik_instance.Scalar("int32", init_value=(value + (factor - 1)) // factor)
        return result

    def tik_func_vector(self, _ub, value, dup_len):
        """
        tik_func_vector
        """
        repeat = self.tik_instance.Scalar("int32")
        repeat_times = self.tik_instance.Scalar("int32")
        repeat_tail = self.tik_instance.Scalar("int32")
        offset = self.tik_instance.Scalar("int64")

        do_dtype = _ub.dtype
        if do_dtype in ["bfloat16", ]:
            byte_num_one = 2
        else:
            byte_num_one = util_tik_comm_func.common_util.get_data_size(do_dtype)
        block_num = Constant.BLOCK_SIZE // byte_num_one
        vector_num = block_num * Constant.REPEAT_STRIDE_EIGHT

        repeat.set_as(dup_len // vector_num)
        repeat_times.set_as(repeat // Constant.MAX_REPEAT_NUM)
        repeat_tail.set_as(dup_len % vector_num)
        offset.set_as(util_tik_comm_func.ub_offset(_ub))

        with self.tik_instance.for_range(0, repeat_times):
            self.tik_instance.vector_dup(vector_num, _ub[offset], value, Constant.MAX_REPEAT_NUM, 1, 8)
            repeat.set_as(repeat - Constant.MAX_REPEAT_NUM)
            offset.set_as(offset + vector_num * Constant.MAX_REPEAT_NUM)
        with self.tik_instance.if_scope(repeat > 0):
            self.tik_instance.vector_dup(vector_num, _ub[offset], value, repeat, 1, 8)
            offset.set_as(offset + vector_num * repeat)
        with self.tik_instance.if_scope(repeat_tail > 0):
            self.tik_instance.vector_dup(repeat_tail, _ub[offset], value, 1, 1, 8)

    def tik_func_vmuls(self, dst_ub, src_ub, value, do_len, dst_blk=1, src_blk=1, dst_rep=8, src_rep=8):
        """
        _tik_func_single
        """
        repeat = self.tik_instance.Scalar("int32")
        repeat_times = self.tik_instance.Scalar("int32")
        repeat_tail = self.tik_instance.Scalar("int32")

        vmuls_type = dst_ub.dtype
        if vmuls_type in ["bfloat16", ]:
            byte_num_one = 2
        else:
            byte_num_one = util_tik_comm_func.common_util.get_data_size(vmuls_type)
        block_num = Constant.BLOCK_SIZE // byte_num_one
        vector_num = block_num * Constant.REPEAT_STRIDE_EIGHT

        repeat.set_as(do_len // vector_num)
        repeat_times.set_as(repeat // Constant.MAX_REPEAT_NUM)
        repeat_tail.set_as(do_len % vector_num)

        dst_offset = self.tik_instance.Scalar("int32", init_value=util_tik_comm_func.ub_offset(dst_ub))
        src_offset = self.tik_instance.Scalar("int32", init_value=util_tik_comm_func.ub_offset(src_ub))

        with self.tik_instance.for_range(0, repeat_times):
            self.tik_instance.vmuls(vector_num, dst_ub[dst_offset], src_ub[src_offset], value,
                                    Constant.MAX_REPEAT_NUM, dst_blk, src_blk, dst_rep, src_rep)
            repeat.set_as(repeat - Constant.MAX_REPEAT_NUM)
            dst_offset.set_as(dst_offset + block_num * Constant.MAX_REPEAT_NUM * dst_rep)
            src_offset.set_as(src_offset + block_num * Constant.MAX_REPEAT_NUM * src_rep)
        with self.tik_instance.if_scope(repeat > 0):
            self.tik_instance.vmuls(vector_num, dst_ub[dst_offset], src_ub[src_offset], value,
                                    repeat, dst_blk, src_blk, dst_rep, src_rep)
            dst_offset.set_as(dst_offset + block_num * repeat * dst_rep)
            src_offset.set_as(src_offset + block_num * repeat * src_rep)
        with self.tik_instance.if_scope(repeat_tail > 0):
            self.tik_instance.vmuls(repeat_tail, dst_ub[dst_offset], src_ub[src_offset], value,
                                    1, dst_blk, src_blk, dst_rep, src_rep)

    def ub2ub(self, tik_instance, dst, src, count, tail_overlap=True):
        """
        Move data from ub to ub
        """
        _, _, block_ele = get_mask_rep_stride(src)
        if tail_overlap:
            burst = self.ceil_div(count, block_ele)
            tik_instance.data_move(dst, src, 0, 1, burst, 0, 0)
        else:
            burst = count // block_ele
            with tik_instance.if_scope(burst != 0):
                tik_instance.data_move(dst, src, 0, 1, burst, 0, 0)
            new_index = block_ele * burst
            with tik_instance.for_range(new_index, count) as index:
                dst[index] = src[index]

    def tik_vec_rec_compute(self, divided_a_temp_ub):
        """tik_vec_rec_compute"""
        src_tensor_size = self.tik_instance.Scalar("int32", init_value=self.bag_divided_ub.shape[0])

        dst_rep_stride = 8
        src_rep_stride = 8
        if self.weight_dtype == "float32":
            block_len = self.ceil_div(src_tensor_size, 8)
            repeat_times = self.ceil_div(src_tensor_size, 8)
            mask_len = 8
        else:
            block_len = self.ceil_div(src_tensor_size, 16)
            repeat_times = self.ceil_div(src_tensor_size, 16)
            mask_len = 16
        wk_size = self.work_tensor_size_compute(self.weight_dtype, block_len, repeat_times, src_rep_stride, mask_len)
        work_tensor_ub = self.tik_instance.Tensor("float32", (wk_size,), name="work_tensor_ub", scope=tik.scope_ubuf)
        self.tik_instance.vec_rec_high_preci(8, divided_a_temp_ub, self.bag_divided_ub, work_tensor_ub, 1, 
                                                                            dst_rep_stride, src_rep_stride)
        self.ub2ub(self.tik_instance, self.bag_divided_ub, divided_a_temp_ub, self.bag_divided_ub.shape[0])

    def work_tensor_size_compute(self, dtype, block_len, repeat_times, src_rep_stride, mask_len):
        """return size of work tensor"""
        src_extent_size = self.tik_instance.Scalar("int32")
        wk_size_unit = self.tik_instance.Scalar("int32")
        work_size = self.tik_instance.Scalar("int32")

        src_extent_size.set_as((repeat_times - 1) * src_rep_stride * block_len + mask_len)
        wk_size_unit.set_as(((src_extent_size + block_len - 1) // block_len) * block_len)
        if dtype == "float16" or dtype == "bfloat16":
            work_size.set_as(4 * wk_size_unit)
        else:
            work_size.set_as(2 * wk_size_unit)
        return work_size

    def embeddingbag_compute_reduction(self, task_idx):

        if self.mode == "max" or self.mode == "sum" or self.mode == "mean":
            self.embedding_bag_compute_max_sum_mean(task_idx)

        with self.tik_instance.if_scope(
                                    tik.all(self.is_single_indices == Constant.SINGLE_INDICES, 
                                            tbe_platform.api_check_support("tik.vec_sel", "float32"))):
            self.compute_offset2bag_core(task_idx)
            self.compute_bag_size_core(task_idx)
            self.compute_max_indices(task_idx)

        # 'pylint: disable=too-many-locals,huawei-too-many-arguments
    def vec_conv(self, out_dst, src, copy_num, round_mode="none", dst_rep=8, src_rep=8):
        "vec_conv"
        repeat = self.tik_instance.Scalar("int32")
        repeat_times = self.tik_instance.Scalar("int32")
        repeat_tail = self.tik_instance.Scalar("int32")

        do_dtype = out_dst.dtype
        
        if do_dtype in ["bfloat16", ]:
            byte_num_one = 2
        else:
            byte_num_one = util_tik_comm_func.common_util.get_data_size(do_dtype)
        block_num = Constant.BLOCK_SIZE // byte_num_one
        vector_num = block_num * Constant.REPEAT_STRIDE_EIGHT
        repeat.set_as(copy_num // vector_num)
        repeat_times.set_as(repeat // Constant.MAX_REPEAT_NUM)
        repeat_tail.set_as(copy_num % vector_num)
        offset = self.tik_instance.Scalar("int32", init_value=0)
        
        with self.tik_instance.for_range(0, repeat_times):
            self.tik_instance.vec_conv(copy_num, round_mode, out_dst[offset], src[offset], 255, dst_rep, src_rep)
            repeat.set_as(repeat - Constant.MAX_REPEAT_NUM)
            offset.set_as(offset + Constant.MAX_REPEAT_NUM * block_num * dst_rep)
        with self.tik_instance.if_scope(repeat > 0):
            self.tik_instance.vec_conv(copy_num, round_mode, out_dst[offset], src[offset], repeat, dst_rep, src_rep)
            offset.set_as(offset + repeat * block_num * dst_rep)
        with self.tik_instance.if_scope(repeat_tail > 0):
            self.tik_instance.vec_conv(copy_num, round_mode, out_dst[offset], src[offset], 1, dst_rep, src_rep)


    def compute_max_indices_ascend(self, task_idx):
        with self.tik_instance.new_stmt_scope():
            # init weight ub for compute     
            data_offset = task_idx * (Constant.TASK - 1)
            offset_counts = self.tik_instance.Scalar("int32", init_value=Constant.TASK)
            with self.tik_instance.if_scope(self.batch < (Constant.TASK * task_idx + Constant.TASK)):
                offset_counts.set_as(self.batch - data_offset + 1)
            offset_num = self.mem_aligned(self.indices_dtype, offset_counts)
            offset_ub = self.tik_instance.Tensor(self.offsets_dtype, (offset_num,),
                                                 name="offset_ub", scope=tik.scope_ubuf)
            dtype_size = common_util.get_data_size(self.offsets_dtype)
            block_element = Constant.BLOCK_SIZE // dtype_size
            burst = self.ceil_div(offset_num, block_element)

            with self.tik_instance.if_scope(self.batch < (Constant.TASK * task_idx + Constant.TASK)):
                self.select_data_move_method(offset_ub, self.input_gm_list[2][data_offset],
                                         (offset_counts - 1) * dtype_size, burst)
                offset_ub[offset_counts - 1].set_as(self.indices_zero_dim)
            with self.tik_instance.else_scope():
                self.select_data_move_method(offset_ub, self.input_gm_list[2][data_offset],
                                         offset_counts * dtype_size, burst)
            weight_num = self.mem_aligned(self.weight_dtype, self.embedding_dim)
            num_per_block = min_data_block("int32")
            count_scalar = self.tik_instance.Scalar("int32", init_value=0)
            index_scalar = self.tik_instance.Scalar(dtype="int32")
            tail_block = self.tik_instance.Tensor("int32", (Constant.B32_NUM_PER_BLOCK,),
                                                  name="tail_block", scope=tik.scope_ubuf)
            with self.tik_instance.if_scope(self.embedding_dim < Constant.T):
                result_scalar = self.tik_instance.Scalar("float32")
                tmp_scalar = self.tik_instance.Scalar("float32")
                result_scalar_tmp = self.tik_instance.Scalar(self.weight_dtype)
                tmp_scalar_tmp = self.tik_instance.Scalar(self.weight_dtype)  
                res_indices_ub = self.tik_instance.Tensor(self.indices_dtype, (weight_num,),
                                                          name="res_indices_ub", scope=tik.scope_ubuf)
                output_ub = self.tik_instance.Tensor("int32", [offset_counts - 1, self.embedding_dim],
                                                     name="output_ub", scope=tik.scope_ubuf)
                with self.tik_instance.for_range(0, offset_counts - 1) as i:
                    count_scalar.set_as(0)  
                    result_ub = self.tik_instance.Tensor(self.weight_dtype, (weight_num,),
                                                         name="result_ub", scope=tik.scope_ubuf)
                    temp_sum_ub = self.tik_instance.Tensor(self.weight_dtype, (weight_num,),
                                                           name="temp_sum_ub", scope=tik.scope_ubuf)
                    self.tik_func_vector(result_ub, 0, weight_num)
                    self.tik_func_vector(temp_sum_ub, 0, weight_num)                    
                    
                    bag_len = self.tik_instance.Scalar("int32")
                    bag_tail_num = self.tik_instance.Scalar("int32")
                    bag_head_num = self.tik_instance.Scalar("int32")
                    find_first_idx = self.tik_instance.Scalar("int32", init_value=0)
                    bag_tail_num.set_as(offset_ub[i + 1])
                    bag_head_num.set_as(offset_ub[i])
                    bag_len.set_as(bag_tail_num - bag_head_num)
                    indices_num = self.mem_aligned(self.indices_dtype, bag_len)
                    indices_ub = self.tik_instance.Tensor(self.indices_dtype, (indices_num,),
                                                          name="indices_ub", scope=tik.scope_ubuf)
                    with self.tik_instance.if_scope(indices_num > 0):
                        dtype_size = common_util.get_data_size(self.indices_dtype)
                        block_element = Constant.BLOCK_SIZE // dtype_size
                        burst = self.ceil_div(indices_num, block_element)
                        self.select_data_move_method(indices_ub, self.input_gm_list[1][bag_head_num],
                                                     bag_len * dtype_size, burst)

                        with self.tik_instance.for_range(0, bag_len) as j:
                            with self.tik_instance.if_scope(find_first_idx == 0):
                                index_scalar.set_as(indices_ub[j])
                                with self.tik_instance.if_scope(tik.all(index_scalar != -1,
                                                                        index_scalar != self.padding_idx)):
                                    count_scalar.set_as(count_scalar + 1)
                                    self.gather_from_weight(result_ub, self.input_gm_list[0], index_scalar)
                                    self.tik_func_vector(res_indices_ub, index_scalar, weight_num)
                                    find_first_idx.set_as(1)
                            with self.tik_instance.else_scope():
                                index_scalar.set_as(indices_ub[j])
                                with self.tik_instance.if_scope(
                                                            tik.all(index_scalar != -1,
                                                                    index_scalar != self.padding_idx)):
                                    count_scalar.set_as(count_scalar + 1)
                                    self.gather_from_weight(temp_sum_ub, self.input_gm_list[0], index_scalar)
                                with self.tik_instance.else_scope():
                                    self.tik_instance.tik_continue()

                                with self.tik_instance.for_range(0, self.embedding_dim) as k:
                                    if self.weight_dtype == "float32":
                                        result_scalar.set_as(result_ub[k])
                                        tmp_scalar.set_as(temp_sum_ub[k])
                                    if self.weight_dtype in ["float16", ]:
                                        result_scalar_tmp.set_as(result_ub[k])
                                        tmp_scalar_tmp.set_as(temp_sum_ub[k])
                                        self.tik_instance.scalar_conv("", result_scalar, result_scalar_tmp)
                                        self.tik_instance.scalar_conv("", tmp_scalar, tmp_scalar_tmp)
                                    if self.weight_dtype in ["bfloat16", ]:
                                        result_scalar_tmp.set_as(result_ub[k])
                                        tmp_scalar_tmp.set_as(temp_sum_ub[k])
                                        tmp_bf16_ub = self.tik_instance.Tensor(self.weight_dtype, (2,),
                                                                               name="tmp_bf16_ub",
                                                                               scope=tik.scope_ubuf)
                                        tmp_bf16tofp32_ub = self.tik_instance.Tensor("float32", (2,), 
                                                                                    name="tmp_bf16tofp32_ub",
                                                                                    scope=tik.scope_ubuf)
                                        tmp_bf16_ub[0].set_as(result_scalar_tmp)
                                        tmp_bf16_ub[1].set_as(tmp_scalar_tmp)
                                        self.tik_instance.vec_conv(2, "", tmp_bf16tofp32_ub, tmp_bf16_ub, 1, 8, 4)
                                        result_scalar.set_as(tmp_bf16tofp32_ub[0])
                                        tmp_scalar.set_as(tmp_bf16tofp32_ub[1])
                                    with self.tik_instance.if_scope(result_scalar < tmp_scalar):
                                        result_ub[k].set_as(temp_sum_ub[k])
                                        res_indices_ub[k].set_as(index_scalar)

                        with self.tik_instance.if_scope(count_scalar == 0):
                            self.tik_func_vector(res_indices_ub, 0, weight_num)
                        
                        with self.tik_instance.for_range(0, self.embedding_dim) as j:
                            output_ub[i, j].set_as(res_indices_ub[j])

                    block_element = Constant.BLOCK_SIZE // self.indices_dtype_size
                    burst = self.ceil_div((offset_counts - 1) * self.embedding_dim, block_element)
                    self.select_data_move_method(self.output_gm_list[3][data_offset * self.embedding_dim], 
                                                 output_ub,
                                                 (offset_counts - 1) * self.embedding_dim * self.indices_dtype_size,
                                                 burst)
            with self.tik_instance.else_scope():
                result_scalar = self.tik_instance.Scalar("float32")
                tmp_scalar = self.tik_instance.Scalar("float32")
                result_scalar_tmp = self.tik_instance.Scalar(self.weight_dtype)
                tmp_scalar_tmp = self.tik_instance.Scalar(self.weight_dtype)                
                res_indices_ub = self.tik_instance.Tensor(self.indices_dtype, (weight_num,),
                                                          name="res_indices_ub", scope=tik.scope_ubuf)

                with self.tik_instance.for_range(0, offset_counts - 1) as i:
                    count_scalar.set_as(0)
                    self.tik_func_vector(res_indices_ub, 0, weight_num)

                    result_ub = self.tik_instance.Tensor(self.weight_dtype, (weight_num,),
                                                         name="result_ub", scope=tik.scope_ubuf)
                    temp_sum_ub = self.tik_instance.Tensor(self.weight_dtype, (weight_num,),
                                                           name="temp_sum_ub", scope=tik.scope_ubuf)
                    self.tik_func_vector(result_ub, 0, weight_num)
                    self.tik_func_vector(temp_sum_ub, 0, weight_num)                    
                    
                    bag_len = self.tik_instance.Scalar("int32")
                    bag_tail_num = self.tik_instance.Scalar("int32")
                    bag_head_num = self.tik_instance.Scalar("int32")
                    find_first_idx = self.tik_instance.Scalar("int32", init_value=0)
                    bag_tail_num.set_as(offset_ub[i + 1])
                    bag_head_num.set_as(offset_ub[i])
                    bag_len.set_as(bag_tail_num - bag_head_num)
                    indices_num = self.mem_aligned(self.indices_dtype, bag_len)
                    indices_ub = self.tik_instance.Tensor(self.indices_dtype, (indices_num,),
                                                          name="indices_ub", scope=tik.scope_ubuf)
                    with self.tik_instance.if_scope(indices_num > 0):
                        dtype_size = common_util.get_data_size(self.indices_dtype)
                        block_element = Constant.BLOCK_SIZE // dtype_size
                        burst = self.ceil_div(indices_num, block_element)
                        self.select_data_move_method(indices_ub, self.input_gm_list[1][bag_head_num],
                                                     bag_len * dtype_size, burst)
                        with self.tik_instance.for_range(0, bag_len) as j:
                            with self.tik_instance.if_scope(find_first_idx == 0):
                                index_scalar.set_as(indices_ub[j])
                                with self.tik_instance.if_scope(tik.all(index_scalar != -1,
                                                                        index_scalar != self.padding_idx)):
                                    count_scalar.set_as(count_scalar + 1)
                                    self.gather_from_weight(result_ub, self.input_gm_list[0], index_scalar)
                                    self.tik_func_vector(res_indices_ub, index_scalar, weight_num)
                                    find_first_idx.set_as(1)
                            with self.tik_instance.else_scope():
                                index_scalar.set_as(indices_ub[j])
                                with self.tik_instance.if_scope(
                                                            tik.all(index_scalar != - 1,
                                                                    index_scalar != self.padding_idx)):
                                    count_scalar.set_as(count_scalar + 1)
                                    self.gather_from_weight(temp_sum_ub, self.input_gm_list[0], index_scalar)
                                with self.tik_instance.else_scope():
                                    self.tik_instance.tik_continue()
                                
                                with self.tik_instance.for_range(0, self.embedding_dim) as k:
                                    if self.weight_dtype in ["float32", ]:
                                        result_scalar.set_as(result_ub[k])
                                        tmp_scalar.set_as(temp_sum_ub[k])
                                    if self.weight_dtype in ["float16", ]:
                                        result_scalar_tmp.set_as(result_ub[k])
                                        tmp_scalar_tmp.set_as(temp_sum_ub[k])
                                        self.tik_instance.scalar_conv("", result_scalar, result_scalar_tmp)
                                        self.tik_instance.scalar_conv("", tmp_scalar, tmp_scalar_tmp)
                                    if self.weight_dtype in ["bfloat16", ]:
                                        result_scalar_tmp.set_as(result_ub[k])
                                        tmp_scalar_tmp.set_as(temp_sum_ub[k])
                                        tmp_bf16_ub = self.tik_instance.Tensor(self.weight_dtype, (2,),
                                                                               name="tmp_bf16_ub",
                                                                               scope=tik.scope_ubuf)
                                        tmp_bf16tofp32_ub = self.tik_instance.Tensor("float32", (2,),
                                                                                    name="tmp_bf16tofp32_ub",
                                                                                    scope=tik.scope_ubuf)
                                        tmp_bf16_ub[0].set_as(result_scalar_tmp)
                                        tmp_bf16_ub[1].set_as(tmp_scalar_tmp)
                                        self.tik_instance.vec_conv(2, "", tmp_bf16tofp32_ub, tmp_bf16_ub, 1, 8, 4)
                                        result_scalar.set_as(tmp_bf16tofp32_ub[0])
                                        tmp_scalar.set_as(tmp_bf16tofp32_ub[1])
                                    with self.tik_instance.if_scope(result_scalar < tmp_scalar):
                                        result_ub[k].set_as(temp_sum_ub[k])
                                        res_indices_ub[k].set_as(index_scalar)

                        with self.tik_instance.if_scope(count_scalar == 0):
                            self.tik_func_vector(res_indices_ub, 0, weight_num)                        
                        with self.tik_instance.if_scope(i != (offset_counts - 1 - 1)):
                            self.tik_instance.data_move(self.output_gm_list[3][(data_offset + i) * self.embedding_dim],
                                                                res_indices_ub, 0, 1, weight_num // num_per_block, 0, 0)
                        with self.tik_instance.else_scope():
                            tail_block = self.tik_instance.Tensor("int32", (num_per_block,),
                                                                name="tail_block", scope=tik.scope_ubuf)
                            with self.tik_instance.for_range(0, Constant.B32_NUM_PER_BLOCK) as j:
                                tail_block[j].set_as(
                                    res_indices_ub[self.embedding_dim - Constant.B32_NUM_PER_BLOCK + j])
                    
                            self.tik_instance.data_move(self.output_gm_list[3][(data_offset + i) * self.embedding_dim], 
                                                        res_indices_ub, 0, 1, self.embedding_dim // num_per_block, 0, 0)
                            self.tik_instance.data_move(
                        self.output_gm_list[3][(data_offset + i + 1) * self.embedding_dim - Constant.B32_NUM_PER_BLOCK],
                                                                                            tail_block, 0, 1, 1, 0, 0)

    def tik_func_vcmpv_eq(self, mask_ub, result_ub, temp_ub, res_fp32, src1_fp32, copy_num, dst_rep=8):
        """
        tik_func_vcomple
        """
        repeat = self.tik_instance.Scalar("int32")
        repeat_tail = self.tik_instance.Scalar("int32")

        do_dtype = mask_ub.dtype
        if do_dtype in ["bfloat16", ]:
            byte_num_one = 2
        else:
            byte_num_one = util_tik_comm_func.common_util.get_data_size(do_dtype)
        block_num = Constant.BLOCK_SIZE // byte_num_one
        vector_num = block_num * Constant.REPEAT_STRIDE_EIGHT
        repeat.set_as(copy_num // vector_num)
        repeat_tail.set_as(copy_num % vector_num)
        offset = self.tik_instance.Scalar("int32", init_value=0)

        with self.tik_instance.if_scope(repeat > 0):
            with self.tik_instance.for_range(0, repeat) as i:
                offset.set_as(i * block_num * dst_rep)
                self.tik_instance.vcmpv_eq(mask_ub, result_ub[offset], temp_ub[offset], 1, 1, 1, 8, 8)
                self.tik_instance.vec_sel(vector_num, 0, res_fp32[offset], mask_ub, src1_fp32[offset],
                                                                            res_fp32[offset], 1, 8, 8, 8)
        
        with self.tik_instance.if_scope(repeat_tail > 0):
            offset.set_as(offset + block_num * dst_rep)
            with self.tik_instance.if_scope(self.embedding_dim < vector_num):
                offset.set_as(0)
            self.tik_instance.vcmpv_eq(mask_ub, result_ub[offset], temp_ub[offset], 1, 1, 1, 8, 8)
            self.tik_instance.vec_sel(repeat_tail, 0, res_fp32[offset], mask_ub, src1_fp32[offset],
                                                                        res_fp32[offset], 1, 8, 8, 8)

    
    def tik_func_vcomple(self, function, out_dst, src0, src1, copy_num, dst_blk=1, src0_blk=1,
                                                    src1_blk=1, dst_rep=8, src0_rep=8, src1_rep=8):
        """
        tik_func_vcomple
        """
        repeat = self.tik_instance.Scalar("int32")
        repeat_times = self.tik_instance.Scalar("int32")
        repeat_tail = self.tik_instance.Scalar("int32")

        do_dtype = out_dst.dtype
        if do_dtype in ["bfloat16", ]:
            byte_num_one = 2
        else:
            byte_num_one = util_tik_comm_func.common_util.get_data_size(do_dtype)
        block_num = Constant.BLOCK_SIZE // byte_num_one # 8
        vector_num = block_num * Constant.REPEAT_STRIDE_EIGHT
        repeat.set_as(copy_num // vector_num)
        repeat_times.set_as(repeat // Constant.MAX_REPEAT_NUM)
        repeat_tail.set_as(copy_num % vector_num)
        tik_fun = None
        ori_offset_dst = self.tik_instance.Scalar("int32", init_value=util_tik_comm_func.ub_offset(out_dst))
        ori_offset_src0 = self.tik_instance.Scalar("int32", init_value=util_tik_comm_func.ub_offset(src0))
        ori_offset_src1 = self.tik_instance.Scalar("int32", init_value=util_tik_comm_func.ub_offset(src1))
        if function == "vmax":
            tik_fun = self.tik_instance.vmax
        elif function == "vadd":
            tik_fun = self.tik_instance.vadd
        elif function == "vsub":
            tik_fun = self.tik_instance.vsub

        with self.tik_instance.for_range(0, repeat_times):
            tik_fun(vector_num, out_dst[ori_offset_dst], src0[ori_offset_src0], src1[ori_offset_src1], 255,
                                                    dst_blk, src0_blk, src1_blk, dst_rep, src0_rep, src1_rep)

            repeat.set_as(repeat - Constant.MAX_REPEAT_NUM)
            ori_offset_dst.set_as(ori_offset_dst + Constant.MAX_REPEAT_NUM * block_num * dst_rep)
            ori_offset_src0.set_as(ori_offset_src0 + Constant.MAX_REPEAT_NUM * block_num * src0_rep)
            ori_offset_src1.set_as(ori_offset_src1 + Constant.MAX_REPEAT_NUM * block_num * src1_rep)

        with self.tik_instance.if_scope(repeat > 0):
            tik_fun(vector_num, out_dst[ori_offset_dst], src0[ori_offset_src0], src1[ori_offset_src1], repeat,
                                                        dst_blk, src0_blk, src1_blk, dst_rep, src0_rep, src1_rep)

            ori_offset_dst.set_as(ori_offset_dst + repeat * block_num * dst_rep)
            ori_offset_src0.set_as(ori_offset_src0 + repeat * block_num * src0_rep)
            ori_offset_src1.set_as(ori_offset_src1 + repeat * block_num * src1_rep)

        with self.tik_instance.if_scope(repeat_tail > 0):
            tik_fun(repeat_tail, out_dst[ori_offset_dst], src0[ori_offset_src0], src1[ori_offset_src1], 1,
                                                    dst_blk, src0_blk, src1_blk, dst_rep, src0_rep, src1_rep)

    def embedding_bag_compute_sum(self, i, bag_len, bag_head_num, weight_num, indices_ub, result_ub, temp_ub, 
                                                                                            output_ub_temp=None):
        loop = self.embedding_dim // Constant.MASK
        tail = self.embedding_dim % Constant.MASK
        tmp_ub = self.tik_instance.Tensor(self.weight_dtype, (Constant.MASK,), name="tmp_ub", scope=tik.scope_ubuf)
        res_fp32_ub = self.tik_instance.Tensor("float32", (Constant.MASK,), name="res_fp32_ub", scope=tik.scope_ubuf)
        tmp_fp32_ub = self.tik_instance.Tensor("float32", (Constant.MASK,), name="tmp_fp32_ub", scope=tik.scope_ubuf)
        indices_num = self.mem_aligned(self.indices_dtype, bag_len)

        index_scalar = self.tik_instance.Scalar("int32")
        count_scalar = self.tik_instance.Scalar("int32", init_value=0)

        num_per_block = min_data_block(self.weight_dtype)
        offset = self.tik_instance.Scalar("int32", init_value=0)
        scale_value_scalar = self.tik_instance.Scalar(dtype=self.weight_dtype, init_value=1.0)
        scale_value_scalar_fp32 = self.tik_instance.Scalar(dtype="float32", init_value=1.0)
        
        if self.has_per_sample_weights:
            per_ub = self.tik_instance.Tensor(self.weight_dtype, (indices_num,), name="per_ub", scope=tik.scope_ubuf)
            if self.weight_dtype in ["bfloat16", "float16"]:
                dtype_size = 2
            else:
                dtype_size = common_util.get_data_size(self.weight_dtype)
            block_element = Constant.BLOCK_SIZE // dtype_size
            burst = self.ceil_div(indices_num, block_element)
            self.select_data_move_method(per_ub, self.input_gm_list[3][bag_head_num],
                                         bag_len * dtype_size, burst)

        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as mindex:
                self.tik_func_vector(res_fp32_ub, 0, Constant.MASK)

                with self.tik_instance.for_range(0, bag_len) as j:
                    index_scalar.set_as(indices_ub[j])
                    with self.tik_instance.if_scope(tik.all(index_scalar != -1, index_scalar != self.padding_idx)):
                        if self.has_per_sample_weights:
                            scale_value_scalar.set_as(per_ub[j])
                        count_scalar.set_as(count_scalar + 1)
                        self.tik_instance.data_move(tmp_ub,
                                    self.input_gm_list[0][index_scalar * self.embedding_dim + mindex * Constant.MASK],
                                                                            0, 1, Constant.MASK // num_per_block, 0, 0)
                    with self.tik_instance.else_scope():
                        self.tik_instance.tik_continue()
                    if self.weight_dtype in ["bfloat16", "float16"]:
                        self.tik_instance.vec_conv(Constant.MASK, "", tmp_fp32_ub, tmp_ub, 1, 8, 4)
                    if self.has_per_sample_weights:
                        if self.weight_dtype in ["float16", ]:
                            self.tik_instance.scalar_conv("", scale_value_scalar_fp32, scale_value_scalar)
                            self.tik_func_vmuls(tmp_fp32_ub, tmp_fp32_ub, scale_value_scalar_fp32, Constant.MASK)
                        if self.weight_dtype in ["bfloat16", ]:
                            tmp_bf16_ub = self.tik_instance.Tensor(self.weight_dtype, (2,), name="tmp_bf16_ub",
                                                                   scope=tik.scope_ubuf)
                            tmp_bf16tofp32_ub = self.tik_instance.Tensor("float32", (2,), name="tmp_bf16tofp32_ub",
                                                                         scope=tik.scope_ubuf)
                            tmp_bf16_ub[0].set_as(scale_value_scalar)
                            self.tik_instance.vec_conv(2, "", tmp_bf16tofp32_ub, tmp_bf16_ub, 1, 8, 4)
                            scale_value_scalar_fp32.set_as(tmp_bf16tofp32_ub[0])
                            self.tik_func_vmuls(tmp_fp32_ub, tmp_fp32_ub, scale_value_scalar_fp32, Constant.MASK)
                        if self.weight_dtype == "float32":
                            self.tik_func_vmuls(tmp_ub, tmp_ub, scale_value_scalar, Constant.MASK)
                    if self.weight_dtype in ["float16", "bfloat16"]:
                        self.tik_func_vcomple("vadd", res_fp32_ub, res_fp32_ub, tmp_fp32_ub, Constant.MASK)
                    if self.weight_dtype == "float32":
                        self.tik_func_vcomple("vadd", res_fp32_ub, res_fp32_ub, tmp_ub, Constant.MASK)

                with self.tik_instance.if_scope(count_scalar == 0):
                    self.tik_func_vector(res_fp32_ub, 0, Constant.MASK)
                if self.weight_dtype in ["float16", "bfloat16"]:
                    self.tik_instance.vec_conv(Constant.MASK, self.conv_mode, result_ub[mindex * Constant.MASK],
                                                                                res_fp32_ub, 1, 4, 8)
                if self.weight_dtype == "float32":
                    self.tik_instance.data_move(result_ub[mindex * Constant.MASK], res_fp32_ub, 0, 1,
                                                                Constant.MASK // num_per_block, 0, 0)
            offset.set_as(offset + loop * Constant.MASK)
        with self.tik_instance.if_scope(tail > 0):

            self.tik_func_vector(res_fp32_ub, 0, tail)
            with self.tik_instance.for_range(0, bag_len) as j:
                index_scalar.set_as(indices_ub[j])
                with self.tik_instance.if_scope(tik.all(index_scalar != -1, index_scalar != self.padding_idx)):
                    count_scalar.set_as(count_scalar + 1)
                    self.select_data_move_method(tmp_ub,
                                                self.input_gm_list[0][index_scalar * self.embedding_dim + offset],
                                                tail * self.weight_dtype_size,
                                                (tail + num_per_block - 1) // num_per_block)

                    if self.has_per_sample_weights:
                        scale_value_scalar.set_as(per_ub[j])
                with self.tik_instance.else_scope():
                    self.tik_instance.tik_continue()

                if self.weight_dtype in ["float16", "bfloat16"]:
                    self.tik_instance.vec_conv(tail, "", tmp_fp32_ub, tmp_ub, 1, 8, 4)
                if self.has_per_sample_weights:
                    if self.weight_dtype in ["float16", ]:
                        self.tik_instance.scalar_conv("", scale_value_scalar_fp32, scale_value_scalar)
                        self.tik_func_vmuls(tmp_fp32_ub, tmp_fp32_ub, scale_value_scalar_fp32, tail)
                    if self.weight_dtype in ["bfloat16"]:
                        tmp_bf16_ub = self.tik_instance.Tensor(self.weight_dtype, (2,), name="tmp_bf16_ub",
                                                               scope=tik.scope_ubuf)
                        tmp_bf16tofp32_ub = self.tik_instance.Tensor("float32", (2,), name="tmp_bf16tofp32_ub",
                                                                     scope=tik.scope_ubuf)
                        tmp_bf16_ub[0].set_as(scale_value_scalar)
                        self.tik_instance.vec_conv(2, "", tmp_bf16tofp32_ub, tmp_bf16_ub, 1, 8, 4)
                        scale_value_scalar_fp32.set_as(tmp_bf16tofp32_ub[0])
                        self.tik_func_vmuls(tmp_fp32_ub, tmp_fp32_ub, scale_value_scalar_fp32, tail)
                    if self.weight_dtype == "float32":
                        self.tik_func_vmuls(tmp_ub, tmp_ub, scale_value_scalar, tail)
                if self.weight_dtype in ["float16", "bfloat16"]:
                    self.tik_func_vcomple("vadd", res_fp32_ub, res_fp32_ub, tmp_fp32_ub, tail)
                if self.weight_dtype == "float32":
                    self.tik_func_vcomple("vadd", res_fp32_ub, res_fp32_ub, tmp_ub, tail)

            with self.tik_instance.if_scope(count_scalar == 0):
                self.tik_func_vector(res_fp32_ub, 0, tail)
            if self.weight_dtype in ["float16", "bfloat16"]:
                self.tik_instance.vec_conv(tail, self.conv_mode, result_ub[offset], res_fp32_ub, 1, 4, 8)
            if self.weight_dtype == "float32":
                self.tik_instance.data_move(result_ub[offset], res_fp32_ub, 0, 1,
                                            (tail + num_per_block - 1) // num_per_block, 0, 0)

    def embedding_bag_compute_max(self, i, bag_len, weight_num, indices_ub, result_ub, temp_ub, output_ub_temp=None):
        with self.tik_instance.if_scope(self.weight_dtype == "bfloat16"):
            self.embedding_bag_compute_max_bf16(i, bag_len, weight_num, indices_ub, result_ub, temp_ub)
        with self.tik_instance.else_scope():
            index_scalar = self.tik_instance.Scalar("int32")
            count_scalar = self.tik_instance.Scalar("int32", init_value=0)

            self.tik_func_vector(result_ub, Constant.MIN_FP32, weight_num)
            with self.tik_instance.if_scope(self.weight_dtype == "float16"):
                self.tik_func_vector(result_ub, Constant.MIN_FP16, weight_num)          
            self.tik_func_vector(temp_ub, 0, weight_num)
            with self.tik_instance.for_range(0, bag_len) as j:
                index_scalar.set_as(indices_ub[j])
                with self.tik_instance.if_scope(tik.all(index_scalar != -1, index_scalar != self.padding_idx)):
                    self.gather_from_weight(temp_ub, self.input_gm_list[0], index_scalar)
                    count_scalar.set_as(count_scalar + 1)
                with self.tik_instance.else_scope():
                    self.tik_instance.tik_continue()
                self.tik_func_vcomple("vmax", result_ub, result_ub, temp_ub, weight_num)

            with self.tik_instance.if_scope(count_scalar == 0):
                self.tik_func_vector(result_ub, 0, weight_num)

    def embedding_bag_compute_max_bf16(self, i, bag_len, weight_num, indices_ub, result_ub, temp_ub,
                                       output_ub_temp=None):
        loop = self.embedding_dim // Constant.MASK
        tail = self.embedding_dim % Constant.MASK
        index_scalar = self.tik_instance.Scalar(Constant.INT32)
        count_scalar = self.tik_instance.Scalar(Constant.INT32, init_value=0)

        
        offset = self.tik_instance.Scalar(Constant.INT32, init_value=0)

        with self.tik_instance.if_scope(loop > 0):
            self.embedding_bag_compute_max_bf16_loop(i, bag_len, weight_num, indices_ub, result_ub, temp_ub,
                                    offset, index_scalar, count_scalar)
        with self.tik_instance.if_scope(tail > 0):
            self.embedding_bag_compute_max_bf16_tail(i, bag_len, weight_num, indices_ub, result_ub, temp_ub,
                                    offset, index_scalar, count_scalar)


    def embedding_bag_compute_max_bf16_loop(self, i, bag_len, weight_num, indices_ub, result_ub, temp_ub,
                                            offset, index_scalar, count_scalar, output_ub_temp=None):
        loop = self.embedding_dim // Constant.MASK
        tmp_ub = self.tik_instance.Tensor(self.weight_dtype, (Constant.MASK,), name="tmp_ub", scope=tik.scope_ubuf)
        res_fp32_ub = self.tik_instance.Tensor("float32", (Constant.MASK,), name="res_fp32_ub", scope=tik.scope_ubuf)
        tmp_fp32_ub = self.tik_instance.Tensor("float32", (Constant.MASK,), name="tmp_fp32_ub", scope=tik.scope_ubuf)

        num_per_block = min_data_block(self.weight_dtype)

        with self.tik_instance.for_range(0, loop) as mindex:
            self.tik_func_vector(res_fp32_ub, Constant.MIN_FP32, Constant.MASK)

            with self.tik_instance.for_range(0, bag_len) as j:
                index_scalar.set_as(indices_ub[j])
                with self.tik_instance.if_scope(tik.all(index_scalar != -1, index_scalar != self.padding_idx)):
                    count_scalar.set_as(count_scalar + 1)
                    self.tik_instance.data_move(tmp_ub,
                                self.input_gm_list[0][index_scalar * self.embedding_dim + mindex * Constant.MASK],
                                                                        0, 1, Constant.MASK // num_per_block, 0, 0)
                with self.tik_instance.else_scope():
                    self.tik_instance.tik_continue()
                self.tik_instance.vec_conv(Constant.MASK, "", tmp_fp32_ub, tmp_ub, 1, 8, 4)
                self.tik_func_vcomple("vmax", res_fp32_ub, res_fp32_ub, tmp_fp32_ub, Constant.MASK)

            with self.tik_instance.if_scope(count_scalar == 0):
                self.tik_func_vector(res_fp32_ub, 0, Constant.MASK)
            
            self.tik_instance.vec_conv(Constant.MASK, self.conv_mode, result_ub[mindex * Constant.MASK],
                                                                            res_fp32_ub, 1, 4, 8)
        offset.set_as(offset + loop * Constant.MASK)
        
    def embedding_bag_compute_max_bf16_tail(self, i, bag_len, weight_num, indices_ub, result_ub, temp_ub,
                                            offset, index_scalar, count_scalar, output_ub_temp=None):
        tail = self.embedding_dim % Constant.MASK
        num_per_block = min_data_block(self.weight_dtype)
        tmp_ub = self.tik_instance.Tensor(self.weight_dtype, (Constant.MASK,), name="tmp_ub", scope=tik.scope_ubuf)
        res_fp32_ub = self.tik_instance.Tensor("float32", (Constant.MASK,), name="res_fp32_ub", scope=tik.scope_ubuf)
        tmp_fp32_ub = self.tik_instance.Tensor("float32", (Constant.MASK,), name="tmp_fp32_ub", scope=tik.scope_ubuf)
        self.tik_func_vector(res_fp32_ub, Constant.MIN_FP32, tail)
        with self.tik_instance.for_range(0, bag_len) as j:
            index_scalar.set_as(indices_ub[j])
            with self.tik_instance.if_scope(tik.all(index_scalar != -1, index_scalar != self.padding_idx)):
                count_scalar.set_as(count_scalar + 1)
                self.tik_instance.data_move(tmp_ub, 
                                            self.input_gm_list[0][index_scalar * self.embedding_dim + offset],
                                            0, 1, (tail + num_per_block - 1) // num_per_block, 0, 0)
            with self.tik_instance.else_scope():
                self.tik_instance.tik_continue()


            self.tik_instance.vec_conv(tail, "", tmp_fp32_ub, tmp_ub, 1, 8, 4)
            self.tik_func_vcomple("vmax", res_fp32_ub, res_fp32_ub, tmp_fp32_ub, tail)

        with self.tik_instance.if_scope(count_scalar == 0):
            self.tik_func_vector(res_fp32_ub, 0, tail)
        self.tik_instance.vec_conv(tail, self.conv_mode, result_ub[offset], res_fp32_ub, 1, 4, 8)

    def embedding_bag_compute_mean(self, i, task_idx, offset_counts, bag_len, weight_num, indices_ub, result_ub,
                                                                        temp_ub, offset_ub, output_ub_temp=None):
        loop = self.embedding_dim // Constant.MASK
        tail = self.embedding_dim % Constant.MASK
        tmp_ub = self.tik_instance.Tensor(self.weight_dtype, (Constant.MASK,), name="tmp_ub", scope=tik.scope_ubuf)
        res_fp32_ub = self.tik_instance.Tensor("float32", (Constant.MASK,), name="res_fp32_ub", scope=tik.scope_ubuf)
        tmp_fp32_ub = self.tik_instance.Tensor("float32", (Constant.MASK,), name="tmp_fp32_ub", scope=tik.scope_ubuf)
        num_per_block = min_data_block(self.weight_dtype)
        offset = self.tik_instance.Scalar("int32", init_value=0)
        index_scalar = self.tik_instance.Scalar("int32")
        count_scalar = self.tik_instance.Scalar("float32", init_value=0)
        divided_scalar = self.tik_instance.Scalar(dtype="float32")
        with self.tik_instance.for_range(0, bag_len) as j:
            index_scalar.set_as(indices_ub[j])
            with self.tik_instance.if_scope(tik.all(index_scalar != -1, index_scalar != self.padding_idx)):
                count_scalar.set_as(count_scalar + 1)
            with self.tik_instance.else_scope():
                self.tik_instance.tik_continue()

        with self.tik_instance.if_scope(count_scalar != 0):
            with self.tik_instance.if_scope(self.batch < (Constant.TASK * task_idx + Constant.TASK)):
                with self.tik_instance.if_scope(i == (offset_counts - 1 - 1)):
                    if self.include_last_offset:
                        tmp_scalar = self.tik_instance.Scalar("int32")
                        tmp_scalar.set_as(offset_ub[i])
                        count_scalar.set_as(self.indices_zero_dim - tmp_scalar)
            divided_scalar.set_as(1 / count_scalar)

            with self.tik_instance.if_scope(loop > 0):
                with self.tik_instance.for_range(0, loop) as mindex:
                    self.tik_func_vector(res_fp32_ub, 0, Constant.MASK)
                    with self.tik_instance.for_range(0, bag_len) as j:
                        index_scalar.set_as(indices_ub[j])
                        with self.tik_instance.if_scope(tik.all(index_scalar != -1, index_scalar != self.padding_idx)):
                            self.tik_instance.data_move(tmp_ub,
                                    self.input_gm_list[0][index_scalar * self.embedding_dim + mindex * Constant.MASK],
                                                                             0, 1, Constant.MASK // num_per_block, 0, 0)
                        with self.tik_instance.else_scope():
                            self.tik_instance.tik_continue()
                        if self.weight_dtype in ["float16", "bfloat16"]:
                            self.tik_instance.vec_conv(Constant.MASK, "", tmp_fp32_ub, tmp_ub, 1, 8, 4)
                            self.tik_func_vcomple("vadd", res_fp32_ub, res_fp32_ub, tmp_fp32_ub, Constant.MASK)
                        if self.weight_dtype == "float32":
                            self.tik_func_vcomple("vadd", res_fp32_ub, res_fp32_ub, tmp_ub, Constant.MASK)
                    self.tik_func_vmuls(res_fp32_ub, res_fp32_ub, divided_scalar, Constant.MASK)
                    if self.weight_dtype in ["float16", "bfloat16"]:
                        self.tik_instance.vec_conv(64, self.conv_mode,
                                                   result_ub[mindex * Constant.MASK], res_fp32_ub, 1, 4, 8)
                    if self.weight_dtype == "float32":
                        self.tik_instance.data_move(result_ub[mindex * Constant.MASK],
                                                            res_fp32_ub, 0, 1, Constant.MASK // num_per_block, 0, 0)
                offset.set_as(offset + loop * Constant.MASK)
            with self.tik_instance.if_scope(tail > 0):
                self.tik_func_vector(res_fp32_ub, 0, tail)
                with self.tik_instance.for_range(0, bag_len) as j:
                    index_scalar.set_as(indices_ub[j])
                    with self.tik_instance.if_scope(tik.all(index_scalar != -1, index_scalar != self.padding_idx)):
                        if self.weight_dtype in ["bfloat16", "float16"]:
                            dtype_size = 2
                        else:
                            dtype_size = common_util.get_data_size(self.weight_dtype)

                        self.select_data_move_method(tmp_ub,
                                                     self.input_gm_list[0][index_scalar * self.embedding_dim + offset],
                                                     tail * dtype_size, (tail + num_per_block - 1) // num_per_block)

                    with self.tik_instance.else_scope():
                        self.tik_instance.tik_continue()
                    if self.weight_dtype in ["float16", "bfloat16"]:
                        self.tik_instance.vec_conv(tail, "", tmp_fp32_ub, tmp_ub, 1, 8, 4)
                        self.tik_func_vcomple("vadd", res_fp32_ub, res_fp32_ub, tmp_fp32_ub, tail)
                    if self.weight_dtype == "float32":
                        self.tik_func_vcomple("vadd", res_fp32_ub, res_fp32_ub, tmp_ub, tail)
                self.tik_func_vmuls(res_fp32_ub, res_fp32_ub, divided_scalar, tail)
                if self.weight_dtype in ["float16", "bfloat16"]:
                    self.tik_instance.vec_conv(tail, self.conv_mode, result_ub[offset], res_fp32_ub, 1, 4, 8)
                if self.weight_dtype == "float32":
                    self.tik_instance.data_move(result_ub[offset], res_fp32_ub, 0, 1,
                                                (tail + num_per_block - 1) // num_per_block, 0, 0)
        with self.tik_instance.else_scope():
            self.tik_func_vector(result_ub, 0, weight_num)

    def embedding_bag_compute_max_sum_mean(self, task_idx):
        """
        embedding_bag_compute_max_sum_mean
        """
        with self.tik_instance.new_stmt_scope():
            data_offset = task_idx * (Constant.TASK - 1)
            offset_counts = self.tik_instance.Scalar("int32", init_value=Constant.TASK)
            with self.tik_instance.if_scope(self.batch < (Constant.TASK * task_idx + Constant.TASK)):
                offset_counts.set_as(self.batch - data_offset + 1)

            offset_num = self.mem_aligned(self.indices_dtype, offset_counts)
            offset_ub = self.tik_instance.Tensor(self.offsets_dtype, (offset_num,),
                                                 name="offset_ub", scope=tik.scope_ubuf)
            
            dtype_size = common_util.get_data_size(self.offsets_dtype)
            block_element = Constant.BLOCK_SIZE // dtype_size
            burst = self.ceil_div(offset_num, block_element)
            with self.tik_instance.if_scope(self.batch < (Constant.TASK * task_idx + Constant.TASK)):
                self.select_data_move_method(offset_ub, self.input_gm_list[2][data_offset],
                                         (offset_counts - 1) * dtype_size, burst)
                offset_ub[offset_counts - 1].set_as(self.indices_zero_dim)
            with self.tik_instance.else_scope():
                self.select_data_move_method(offset_ub, self.input_gm_list[2][data_offset],
                                         offset_counts * dtype_size, burst)    

            bag_len = self.tik_instance.Scalar("int32", init_value=0)
            bag_tail_num = self.tik_instance.Scalar("int32", init_value=0)
            bag_head_num = self.tik_instance.Scalar("int32", init_value=0)
            index_scalar = self.tik_instance.Scalar("int32", init_value=0)
            num_per_block = min_data_block(self.weight_dtype)
            weight_num = self.mem_aligned(self.weight_dtype, self.embedding_dim)
            result_ub = self.tik_instance.Tensor(self.weight_dtype, (weight_num,),
                                                 name="result_ub", scope=tik.scope_ubuf)
          
            with self.tik_instance.if_scope(self.embedding_dim >= Constant.T):
                self.data_mover = HugeDataMover(self.tik_instance, self.weight_dtype, self.output_gm_list[0],
                                                                                    offset_counts, num_per_block)
                with self.tik_instance.for_range(0, offset_counts - 1) as i:
                    count_scalar = self.tik_instance.Scalar("int32", init_value=0)
                    bag_tail_num.set_as(offset_ub[i + 1])
                    bag_head_num.set_as(offset_ub[i])
                    bag_len.set_as(bag_tail_num - bag_head_num)
                    indices_num = self.mem_aligned(self.indices_dtype, bag_len)
                    indices_ub = self.tik_instance.Tensor(self.indices_dtype, (indices_num,),
                                                        name="indices_ub", scope=tik.scope_ubuf)
                    temp_ub = self.tik_instance.Tensor(self.weight_dtype, (weight_num,), name="temp_ub",
                                                                                    scope=tik.scope_ubuf)
                    self.tik_func_vector(result_ub, 0, weight_num)
                    self.tik_func_vector(temp_ub, 0, weight_num)

                    with self.tik_instance.if_scope(indices_num > 0):
                        dtype_size = common_util.get_data_size(self.indices_dtype)
                        block_element = Constant.BLOCK_SIZE // dtype_size
                        burst = self.ceil_div(indices_num, block_element)
                        self.select_data_move_method(indices_ub, self.input_gm_list[1][bag_head_num],
                                                     bag_len * dtype_size, burst)
                        if self.mode == "sum":
                            self.embedding_bag_compute_sum(i, bag_len, bag_head_num, weight_num, indices_ub,
                                                                                            result_ub, temp_ub)
                        if self.mode == "max":
                            self.embedding_bag_compute_max(i, bag_len, weight_num, indices_ub, result_ub, temp_ub)
                        if self.mode == "mean":
                            self.embedding_bag_compute_mean(i, task_idx, offset_counts, bag_len, weight_num, indices_ub,
                                                                                        result_ub, temp_ub, offset_ub)
                    self.data_mover.moving(result_ub, self.embedding_dim, data_offset, i)

            with self.tik_instance.else_scope():
                self.data_mover = LittleDataMover(self.tik_instance, self.weight_dtype,
                                                  [(offset_counts - 1), self.embedding_dim], 
                                                  self.output_gm_list[0], data_offset * self.embedding_dim)
                with self.tik_instance.for_range(0, offset_counts - 1) as i:
                    count_scalar = self.tik_instance.Scalar("int32", init_value=0)
                    bag_tail_num.set_as(offset_ub[i + 1])
                    bag_head_num.set_as(offset_ub[i])
                    bag_len.set_as(bag_tail_num - bag_head_num)

                    indices_num = self.mem_aligned(self.indices_dtype, bag_len)
                    indices_ub = self.tik_instance.Tensor(self.indices_dtype, (indices_num,),
                                                          name="indices_ub", scope=tik.scope_ubuf)
                    temp_ub = self.tik_instance.Tensor(self.weight_dtype, (weight_num,), name="temp_ub",
                                                                                    scope=tik.scope_ubuf)
                    self.tik_func_vector(result_ub, 0, weight_num)
                    self.tik_func_vector(temp_ub, 0, weight_num)

                    with self.tik_instance.if_scope(indices_num > 0):
                        dtype_size = common_util.get_data_size(self.indices_dtype)
                        block_element = Constant.BLOCK_SIZE // dtype_size
                        burst = self.ceil_div(indices_num, block_element)
                        self.select_data_move_method(indices_ub, self.input_gm_list[1][bag_head_num],
                                                     bag_len * dtype_size, burst)
                        if self.mode == "sum":
                            self.embedding_bag_compute_sum(i, bag_len, bag_head_num, weight_num, indices_ub,
                                                                                            result_ub, temp_ub)
                        if self.mode == "max":
                            self.embedding_bag_compute_max(i, bag_len, weight_num, indices_ub, result_ub, temp_ub)
                        if self.mode == "mean":
                            self.embedding_bag_compute_mean(i, task_idx, offset_counts, bag_len, weight_num, indices_ub,
                                                                                        result_ub, temp_ub, offset_ub)

                    self.data_mover.moving(result_ub, self.embedding_dim, data_offset, i)
                self.data_mover.move((offset_counts - 1) * self.embedding_dim * self.weight_dtype_size,
                                     (offset_counts - 1) * self.embedding_dim)


    def compute_offset2bag_core(self, task_idx):
        """
        compute offset2bag
        """
        with self.tik_instance.new_stmt_scope():
            # # init indice ub
            data_offset = task_idx * (Constant.TASK - 1)
            offset_counts = self.tik_instance.Scalar("int32", init_value=Constant.TASK)
            with self.tik_instance.if_scope(self.batch < (Constant.TASK * task_idx + Constant.TASK)):
                offset_counts.set_as(self.batch - data_offset + 1)

            offset_num = self.mem_aligned(self.indices_dtype, offset_counts)
            offset_ub = self.tik_instance.Tensor(self.offsets_dtype, (offset_num,),
                                                 name="offset_ub", scope=tik.scope_ubuf)
            dtype_size = common_util.get_data_size(self.offsets_dtype)
            block_element = Constant.BLOCK_SIZE // dtype_size
            burst = self.ceil_div(offset_num, block_element)
            with self.tik_instance.if_scope(self.batch < (Constant.TASK * task_idx + Constant.TASK)):
                self.select_data_move_method(offset_ub, self.input_gm_list[2][data_offset],
                                         (offset_counts - 1) * dtype_size, burst)
                offset_ub[offset_counts - 1].set_as(self.indices_zero_dim)
            with self.tik_instance.else_scope():
                self.select_data_move_method(offset_ub, self.input_gm_list[2][data_offset],
                                         offset_counts * dtype_size, burst)   
            with self.tik_instance.for_range(offset_counts, offset_num) as i:
                offset_ub[i].set_as(self.neg_one_scalar)

            indices_len = self.tik_instance.Scalar("int32")
            tail_num = self.tik_instance.Scalar("int32")
            head_num = self.tik_instance.Scalar("int32")
            tail_num.set_as(offset_ub[offset_counts - 1])
            head_num.set_as(offset_ub[0])
            indices_len.set_as(tail_num - head_num)

            # init indice ub
            indices_num = self.mem_aligned(self.indices_dtype, Constant.MAX_MOVE_NUM)
            offset2bag_ub = self.tik_instance.Tensor(self.indices_dtype, (indices_num,),
                                                     name="offset2bag_ub", scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, self.indices_num) as i:
                offset2bag_ub[i].set_as(-1)
            start_scalar = self.tik_instance.Scalar(dtype=self.offsets_dtype, init_value=-1)
            end_scalar = self.tik_instance.Scalar(dtype=self.offsets_dtype, init_value=-1)
            number_scalar = self.tik_instance.Scalar(dtype=self.offsets_dtype, init_value=-1)

            with self.tik_instance.for_range(0, offset_counts - 1) as i:
                start_scalar.set_as(offset_ub[i])
                end_scalar.set_as(offset_ub[i + 1])
                count = self.ceil_div((end_scalar - start_scalar), Constant.MAX_MOVE_NUM)
                with self.tik_instance.for_range(0, count) as j:
                    self.tik_instance.vector_dup(64, offset2bag_ub, i + data_offset, 8, 1, 8)
                    move_num = Constant.MAX_MOVE_NUM
                    with self.tik_instance.if_scope(j == count - 1):
                        move_num = end_scalar - start_scalar - j * Constant.MAX_MOVE_NUM
                    self.select_data_move_method(self.output_gm_list[1][start_scalar + j * Constant.MAX_MOVE_NUM],
                                                 offset2bag_ub, move_num * dtype_size, Constant.MAX_MOVE_NUM)

    def compute_bag_size_core(self, task_idx):
        """
        compute bag_size
        """
        with self.tik_instance.new_stmt_scope():
            if self.mode != "sum":
                # # init indice ub
                data_offset = task_idx * (Constant.TASK - 1)
                offset_counts = self.tik_instance.Scalar("int32", init_value=Constant.TASK)
                bag_size_ub_scalar = self.tik_instance.Scalar("int32", init_value=Constant.TASK - 1)
                with self.tik_instance.if_scope(self.batch < (Constant.TASK * task_idx + Constant.TASK)):
                    if not self.include_last_offset:
                        offset_counts.set_as(self.offsets_shape - data_offset + 1)
                        bag_size_ub_scalar.set_as(offset_counts - 1)
                    else:
                        offset_counts.set_as(self.offsets_shape - data_offset)
                        bag_size_ub_scalar.set_as(offset_counts - 1)

                offset_num = self.mem_aligned(self.indices_dtype, offset_counts)
                bag_size_ub_num = self.mem_aligned(self.indices_dtype, bag_size_ub_scalar)
                offset_ub = self.tik_instance.Tensor(self.offsets_dtype, (offset_num,),
                                                     name="offset_ub", scope=tik.scope_ubuf)
                dtype_size = common_util.get_data_size(self.offsets_dtype)
                block_element = Constant.BLOCK_SIZE // dtype_size
                burst = self.ceil_div(offset_num, block_element)

                with self.tik_instance.if_scope(self.batch < (Constant.TASK * task_idx + Constant.TASK)):
                    self.select_data_move_method(offset_ub, self.input_gm_list[2][data_offset],
                                            (offset_counts - 1) * dtype_size, burst)
                    offset_ub[offset_counts - 1].set_as(self.indices_zero_dim)
                with self.tik_instance.else_scope():
                    self.select_data_move_method(offset_ub, self.input_gm_list[2][data_offset],
                                         offset_counts * dtype_size, burst) 

                bag_size_ub = self.tik_instance.Tensor(self.indices_dtype, (bag_size_ub_num,),
                                                       name="bag_size_ub", scope=tik.scope_ubuf)
                with self.tik_instance.for_range(0, bag_size_ub_num) as i:
                    bag_size_ub[i].set_as(-1)
                pre_scalar = self.tik_instance.Scalar(dtype="int32", init_value=0)
                beh_scalar = self.tik_instance.Scalar(dtype="int32", init_value=0)
                bag_len = self.tik_instance.Scalar(dtype="int32", init_value=0)
                real_bag_len = self.tik_instance.Scalar(dtype="int32", init_value=0)
                indices_num = self.tik_instance.Scalar(dtype="int32", init_value=0)
                number_scalar = self.tik_instance.Scalar(dtype="int32", init_value=0)
                dtype_size = self.tik_instance.Scalar(dtype="int32")
                block_element = self.tik_instance.Scalar(dtype="int32", init_value=0)
                burst = self.tik_instance.Scalar(dtype="int32", init_value=0)

                with self.tik_instance.for_range(0, offset_counts - 1) as i:
                    pre_scalar.set_as(offset_ub[i])
                    beh_scalar.set_as(offset_ub[i + 1])
                    bag_len.set_as(beh_scalar - pre_scalar)
                    real_bag_len.set_as(bag_len)
                    indices_num = self.mem_aligned(self.indices_dtype, bag_len)
                    indices_ub = self.tik_instance.Tensor(self.indices_dtype, (indices_num,),
                                                          name="offset_ub", scope=tik.scope_ubuf)
                    with self.tik_instance.if_scope(indices_num > 0):
                        dtype_size = common_util.get_data_size(self.indices_dtype)
                        block_element = Constant.BLOCK_SIZE // dtype_size
                        burst = self.ceil_div(indices_num, block_element)
                        self.select_data_move_method(indices_ub, self.input_gm_list[1][pre_scalar],
                                                    bag_len * dtype_size, burst)
                        
                        with self.tik_instance.for_range(0, bag_len) as j:
                            number_scalar.set_as(indices_ub[j])
                            with self.tik_instance.if_scope(number_scalar == self.padding_idx):
                                real_bag_len.set_as(real_bag_len - 1)
                    bag_size_ub[i].set_as(real_bag_len)


                with self.tik_instance.if_scope(self.batch < (Constant.TASK * task_idx + Constant.TASK)):
                    length_scalar = self.tik_instance.Scalar(dtype=self.indices_dtype)
                    if not self.include_last_offset:
                        length_scalar.set_as(offset_ub[offset_counts - 1])
                        bag_size_ub[offset_counts - 1].set_as(self.indices_zero_dim - length_scalar)
                    else:
                        length_scalar.set_as(offset_ub[offset_counts - 1 - 1])
                        bag_size_ub[offset_counts - 1 - 1].set_as(self.indices_zero_dim - length_scalar)
                block_element = Constant.BLOCK_SIZE // self.indices_dtype_size
                burst = self.ceil_div(bag_size_ub_num, block_element)
                self.select_data_move_method(self.output_gm_list[2][data_offset],
                                             bag_size_ub,
                                             bag_size_ub_scalar * dtype_size, burst)
                if self.mode == "mean":
                    self.select_data_move_method(self.output_gm_list[3][data_offset],
                                                bag_size_ub,
                                                bag_size_ub_scalar * dtype_size, burst)

    def compute_max_indices(self, task_idx):
        if self.mode == "max":
            self.compute_max_indices_ascend(task_idx)

    def gather_from_weight(self, slice_ub, weight_gm, index):
        """gather_from_weight"""
        with self.tik_instance.if_scope(self.weight_num <= Constant.WEIGHT_NUM_MIN):
            with self.tik_instance.for_range(0, self.embedding_dim) as i:
                slice_ub[i].set_as(self.weight_ub_temp[index * self.embedding_dim + i])
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(index < self.num_embeddings - 1):
                with self.tik_instance.if_scope((self.weight_num - index * self.embedding_dim) > 32):
                    util_tik_comm_func.gm2ub(self.tik_instance, slice_ub, weight_gm[index * self.embedding_dim], 
                                                                                                self.embedding_dim)

                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, self.embedding_dim) as i:
                        weight_offset = i + 128 - (self.weight_num - index * self.embedding_dim)
                        slice_ub[i].set_as(self.weight_ub_temp[weight_offset])
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(self.embedding_dim > 32):
                    burst = self.burst_len_compute(self.weight_dtype, self.embedding_dim)
                    self.tik_instance.data_move(slice_ub, weight_gm[index * self.embedding_dim], 0, 1, burst, 0, 0)
                    tail_lens = self.tail_lens_cal(self.weight_dtype, self.embedding_dim)
                    with self.tik_instance.for_range(0, tail_lens) as i:
                        slice_ub[self.embedding_dim - i - 1].set_as(self.weight_ub_temp[127 - i])
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, self.embedding_dim) as i:
                        slice_ub[self.embedding_dim - i - 1].set_as(self.weight_ub_temp[127 - i])

    def embedding_bag_compute(self):
        """embedding_bag_compute"""
        task_num_per_aicore = self.tik_instance.Scalar("int32")
        tail = self.tik_instance.Scalar("int32")
        # infer task allocation
        self.task_num.set_as((self.batch + Constant.TASK - 1 - 1) // (Constant.TASK - 1))
        task_num_per_aicore.set_as(self.task_num // self.aicore_num)
        tail.set_as(self.task_num % self.aicore_num)
        
        with self.tik_instance.for_range(0, self.aicore_num, block_num=self.aicore_num) as i:
            with self.tik_instance.for_range(0, task_num_per_aicore) as j:
                self.embeddingbag_compute_reduction(i + j * self.aicore_num)
            with self.tik_instance.if_scope(i < tail):
                self.embeddingbag_compute_reduction(task_num_per_aicore * self.aicore_num + i)


def get_mask_rep_stride(src):
    """
    Calculate mask and repeat stride for vector commands
    """
    if src.dtype in ["float16", "int16", "uint16", "bfloat16"]:
        mask = Constant.B16_VEC_MASK
        block_ele = Constant.B16_NUM_PER_BLOCK
    elif src.dtype in ["float32", "int32", "uint32"]:
        mask = Constant.B32_VEC_MASK
        block_ele = Constant.B32_NUM_PER_BLOCK
    elif src.dtype in ["int8"]:
        mask = Constant.B8_VEC_MASK
        block_ele = Constant.B8_NUM_PER_BLOCK
    elif src.dtype in ["int64", "uint64"]:
        mask = Constant.B64_VEC_MASK
        block_ele = Constant.B64_NUM_PER_BLOCK
    else:
        raise RuntimeError("Incorrect dtype of src tensor.")
    rep_stride = mask // block_ele
    return mask, rep_stride, block_ele


def min_data_block(dtype):
    """min_data_block"""
    out_num = 0
    if dtype in ["int32", "float32"]:
        out_num = 8
    elif dtype in ["float16", "bfloat16"]:
        out_num = 16
    elif dtype in ["int64"]:
        out_num = 4
    else:
        RuntimeError("dtype is not support !!")
    return out_num


@register_operator("EmbeddingBag")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.OPTION_INPUT, 
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, 
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_STR, 
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL, 
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def embedding_bag(weight, indices, offsets, per_sample_weights, y, offset2bag, bag_size, 
                  max_indices, mode="mean", scale_grid_by_freq=False, sparse=False, 
                  include_last_offset=False, padding_idx=-1, kernel_name="embedding_bag"):
    """
    Computes sums or means of 'bag' of embeddings, without instantiating the intermediate embeddings.

    Parameters:
    ----------
    weight : dict.
        shape, dtype of weight
        An input tensor with shape [num_embeddings, embedding_dim].
        the learnable weights of the module of shape.
    indices : dict.
        shape, dtype of indices
        If input is 1D of shape (N):
            it will be treated as a concatenation of multiple bags (sequences).
            offsets is required to be a 1D tensor containing the starting index
            positions of each bag in input. Therefore, for offsets of shape (B),
            input will be viewed as having B bags. Empty bags (i.e., having 0-length)
            will have returned vectors filled by zeros.
        If input is 2D of shape (B, N):
            it will be treated as B bags (sequences) each of fixed length N, and this
            will return B values aggregated in a way depending on the mode.
            offsets is ignored and required to be None in this case.
    offsets : dict.
        shape, dtype of offsets
        An input tensor with shape [offset_num]
    per_sample_weight : dict.
        shape, dtype of scores
        per_sample_weight to indicate all weights should be taken to be 1.
        If specified, per_sample_weights must have exactly the same shape as input and
        is treated as having the same offsets, if those are not None. Only supported for mode='sum'.
    mode : str.
        A optional attribute of type str, which use "sum", "mean" or "max".
        Specifies the way to reduce the bag.
    scale_grid_by_freq : bool.
        A optional attribute of type bool,
         If "True", "grad_weight" will be scale by word_frequency.
         If "False", "grad_weight" will not be scale by word_frequency.
    sparse : bool.
        A optional attribute of type bool,
         if True, gradient w.r.t.attr weight matrix will be a sparse tensor
    include_last_offset : bool.
        A optional attribute of type bool,
        if True, attr offsets  has one additional element, where the last element
        is equivalent to the size of indices. This matches the CSR format
    kernel_name : str.
        cce kernel name, default value is "embedding_bag"
    Returns
    -------
    tik_instance
    """


    em_bag = EmbeddingBag(weight, indices, offsets, per_sample_weights, mode, scale_grid_by_freq, sparse, 
                          include_last_offset, padding_idx)

    # init gm mem
    em_bag.init_tik_mem()
    # init ub
    em_bag.init_ub_mem()
    # embedding compute
    em_bag.embedding_bag_compute()
    return em_bag.build_tik_instance(kernel_name)