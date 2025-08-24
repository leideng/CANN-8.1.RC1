# Copyright 2021 Huawei Technologies Co., Ltd
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

from functools import reduce as functools_reduce
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util import util_tik_comm_func

# 'pylint: disable=too-many-lines,too-many-instance-attributes,too-many-statements
# 'pylint: disable=too-many-arguments,unused-argument,too-many-locals,too-many-branches


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # process 128 weight at one time
    WEIGHT_NUM_MIN = 128
    MAX_BAG_SIZE = 16


class EmbeddingBag:
    """
    Function: use to store EmbeddingBag base parameters
    Modify : 2021-03-01
    """

    def __init__(self,
                 weight,
                 indices,
                 offsets,
                 per_sample_weights,
                 mode,
                 scale_grid_by_freq,
                 sparse,
                 include_last_offset):
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

        self.weight_shape = list(weight.get("shape"))
        self.weight_dtype = weight.get("dtype")

        self.indices_shape = list(indices.get("shape"))
        self.indices_dtype = indices.get("dtype")

        if offsets is None:
            self.has_offset = False
        else:
            self.has_offset = True
            self.offsets_shape = list(offsets.get("shape"))
            self.offsets_dtype = offsets.get("dtype")

        if per_sample_weights is None:
            self.has_per_sample_weights = False
        else:
            self.has_per_sample_weights = True
            self.per_sample_weights_shape = list(per_sample_weights.get("shape"))
            self.per_sample_weights_dtype = per_sample_weights.get("dtype")
        # attr
        self.mode = mode
        self.sparse = sparse
        self.scale_grid_by_freq = scale_grid_by_freq
        self.include_last_offset = include_last_offset

        self.is_single_indices = True
        if len(self.indices_shape) > 1:
            self.is_single_indices = False

        # get output shape
        self.output_shape = self.get_output_shape(self.weight_shape, self.offsets_shape)
        self.output_dtype = weight.get("dtype")

        # define var
        self.max_bag_size = self.get_max_bag_size()
        self.embedding_dim = self.weight_shape[1]

        self.weight_ub_temp = None
        self.indices_bag_l1 = None
        self.output_ub_temp = None
        self.output_ub_tail = None
        self.indices_bag_gm = None
        self.per_sample_index_gm = None
        self.bag_divided_ub = None
        self.bag_count_ub = None
        self.tail_size = 0

        self.zero_scalar = self.tik_instance.Scalar(dtype="int32", init_value=0)
        self.neg_one_scalar = self.tik_instance.Scalar(dtype="int32", init_value=-1)

    def get_max_bag_size(self):
        """get max bag size
        """
        if self.is_single_indices:
            res = Constant.MAX_BAG_SIZE
        else:
            res = mem_aligned("int32", self.indices_shape[1])
        return res

    def get_output_shape(self, weight_shape, offsets_shape):
        """get output shape
        """
        embedding_dim = weight_shape[1]
        if self.is_single_indices:
            if self.include_last_offset:
                batch = offsets_shape[0] - 1
            else:
                batch = offsets_shape[0]
        else:
            batch = self.indices_shape[0]
        output_shape = [batch, embedding_dim]
        return output_shape

    def init_tik_mem(self):
        """init tik gm mem
        """
        # init gm input
        weight_gm = self.tik_instance.Tensor(self.weight_dtype, self.weight_shape,
                                             name="weight_gm", scope=tbe_platform.scope_gm)
        indices_gm = self.tik_instance.Tensor(self.indices_dtype, self.indices_shape, name="indices_gm",
                                              scope=tbe_platform.scope_gm)

        offsets_gm = None
        per_sample_weights_gm = None

        if self.has_offset:
            offsets_gm = self.tik_instance.Tensor(self.offsets_dtype, self.offsets_shape, name="offsets_gm",
                                                  scope=tbe_platform.scope_gm)
        if self.has_per_sample_weights:
            per_sample_weights_gm = self.tik_instance.Tensor(self.per_sample_weights_dtype,
                                                             self.per_sample_weights_shape,
                                                             name="per_sample_weights_gm",
                                                             scope=tbe_platform.scope_gm)
        if self.has_offset and self.has_per_sample_weights:
            self.input_gm_list = [weight_gm, indices_gm, offsets_gm, per_sample_weights_gm]
        elif self.has_offset:
            self.input_gm_list = [weight_gm, indices_gm, offsets_gm]
        elif self.has_per_sample_weights:
            self.input_gm_list = [weight_gm, indices_gm, per_sample_weights_gm]
        else:
            self.input_gm_list = [weight_gm, indices_gm]

        # init gm output
        embedding_bag_output_gm = self.tik_instance.Tensor(self.output_dtype, self.output_shape,
                                                           name="embedding_bag_output_gm",
                                                           scope=tbe_platform.scope_gm)
        self.output_gm_list = [embedding_bag_output_gm]

        # init temp tbe_platform gm
        self.indices_bag_gm = self.tik_instance.Tensor("int32", (self.output_shape[0], self.max_bag_size),
                                                       name="indices_bag_gm",
                                                       scope=tbe_platform.scope_gm, is_workspace=True)
        if self.has_per_sample_weights:
            self.per_sample_index_gm = self.tik_instance.Tensor("int32", (self.output_shape[0], self.max_bag_size),
                                                                name="per_sample_index_gm",
                                                                scope=tbe_platform.scope_gm, is_workspace=True)

    def init_ub_mem(self):
        """init tik ub mem
        """
        weight_total_size = total_num(self.weight_shape)
        self.weight_ub_temp = self.tik_instance.Tensor(self.weight_dtype, (128,),
                                                       name="weight_ub_temp", scope=tbe_platform.scope_ubuf)
        if weight_total_size <= Constant.WEIGHT_NUM_MIN:
            util_tik_comm_func.gm2ub(self.tik_instance, self.weight_ub_temp, self.input_gm_list[0], weight_total_size)
        else:
            util_tik_comm_func.gm2ub(self.tik_instance, self.weight_ub_temp,
                                     self.input_gm_list[0][weight_total_size - Constant.WEIGHT_NUM_MIN],
                                     Constant.WEIGHT_NUM_MIN)
        self.output_ub_temp = self.tik_instance.Tensor(self.weight_dtype, (128,),
                                                       name="output_ub_temp", scope=tbe_platform.scope_ubuf)
        util_tik_comm_func.tik_func_vector(self.tik_instance, self.output_ub_temp, -1, 128)
        # init tail ub for tail data move to gm
        self.tail_size = min_data_block(self.weight_dtype)
        self.output_ub_tail = self.tik_instance.Tensor(self.weight_dtype, (self.tail_size,),
                                                       name="output_ub_tail", scope=tbe_platform.scope_ubuf)
        util_tik_comm_func.tik_func_vector(self.tik_instance, self.output_ub_tail, -1, self.tail_size)

    def get_tik_instance(self):
        """get tik instance
        """
        return self.tik_instance

    def build_tik_instance(self, kernel_name_value):
        """build_tik_instance
        """
        self.tik_instance.BuildCCE(kernel_name=kernel_name_value,
                                   inputs=self.input_gm_list,
                                   outputs=self.output_gm_list,
                                   output_files_path=None,
                                   enable_l2=False)
        return self.tik_instance

    def offset_1d_count_compute(self, indices_bag_ub):
        """offset_1d_count_compute
        """
        with self.tik_instance.new_stmt_scope():
            # init temp ub for divide value
            divided_a_temp_ub = self.tik_instance.Tensor(self.weight_dtype, (self.bag_divided_ub.shape[0],),
                                                         name="divided_a_temp_ub", scope=tbe_platform.scope_ubuf)
            divided_b_temp_ub = self.tik_instance.Tensor(self.weight_dtype, (self.bag_divided_ub.shape[0],),
                                                         name="divided_b_temp_ub", scope=tbe_platform.scope_ubuf)

            count_a_temp_ub = self.tik_instance.Tensor("int32", (self.bag_count_ub.shape[0],),
                                                       name="count_a_temp_ub", scope=tbe_platform.scope_ubuf)
            count_b_temp_ub = self.tik_instance.Tensor("int32", (self.bag_count_ub.shape[0],),
                                                       name="count_b_temp_ub", scope=tbe_platform.scope_ubuf)
            util_tik_comm_func.tik_func_vector(self.tik_instance, self.bag_divided_ub, 1.0,
                                               self.bag_divided_ub.shape[0])
            util_tik_comm_func.tik_func_vector(self.tik_instance, self.bag_count_ub, 1,
                                               self.bag_count_ub.shape[0])

            # move data to diveded ub
            util_tik_comm_func.ub2ub(self.tik_instance, divided_b_temp_ub, self.bag_divided_ub,
                                     self.bag_divided_ub.shape[0])
            # move data to count ub
            util_tik_comm_func.ub2ub(self.tik_instance, count_b_temp_ub, self.bag_count_ub,
                                     self.bag_count_ub.shape[0])
            # define scalar to compute
            valid_scalar = self.tik_instance.Scalar(dtype="int32")
            with self.tik_instance.for_range(0, self.output_shape[0]) as i:
                util_tik_comm_func.tik_func_vector(self.tik_instance, divided_a_temp_ub, 1.0,
                                                   self.bag_divided_ub.shape[0])
                util_tik_comm_func.tik_func_vector(self.tik_instance, count_a_temp_ub, 1,
                                                   self.bag_count_ub.shape[0])
                with self.tik_instance.for_range(1, self.max_bag_size) as j:
                    valid_scalar.set_as(indices_bag_ub[i * self.max_bag_size + j])
                    with self.tik_instance.if_scope(valid_scalar != -1):
                        util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vadd", count_a_temp_ub,
                                                            count_a_temp_ub,
                                                            count_b_temp_ub, self.bag_count_ub.shape[0])
                        util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vadd", divided_a_temp_ub,
                                                            divided_a_temp_ub,
                                                            divided_b_temp_ub, self.bag_divided_ub.shape[0])
                self.bag_count_ub[i].set_as(count_a_temp_ub[0])
                self.bag_divided_ub[i].set_as(divided_a_temp_ub[0])
            self.tik_vec_rec_compute(divided_a_temp_ub)

    def tik_vec_rec_compute(self, divided_a_temp_ub):
        """tik_vec_rec_compute
        """
        src_tensor_size = self.bag_divided_ub.shape[0]

        dst_rep_stride = 8
        src_rep_stride = 8
        if self.weight_dtype == "float32":
            block_len = ceil_div(src_tensor_size, 8)
            repeat_times = ceil_div(src_tensor_size, 8)
            mask_len = 8
        else:
            block_len = ceil_div(src_tensor_size, 16)
            repeat_times = ceil_div(src_tensor_size, 16)
            mask_len = 16
        wk_size = work_tensor_size_compute(self.weight_dtype, block_len, repeat_times, src_rep_stride, mask_len)
        work_tensor_ub = self.tik_instance.Tensor("float32", (wk_size,), name="work_tensor_ub",
                                                  scope=tbe_platform.scope_ubuf)
        self.tik_instance.vec_rec_high_preci(8, divided_a_temp_ub, self.bag_divided_ub, work_tensor_ub,
                                             1, dst_rep_stride, src_rep_stride)
        util_tik_comm_func.ub2ub(self.tik_instance, self.bag_divided_ub, divided_a_temp_ub,
                                 self.bag_divided_ub.shape[0])

    def offset_to_bag_1d(self):
        """ make offset to bag 1d
        """
        with self.tik_instance.new_stmt_scope():
            # init indice ub
            indices_num = mem_aligned("int32", self.indices_shape[0])
            indices_ub = self.tik_instance.Tensor("int32", (indices_num,),
                                                  name="indices_ub", scope=tbe_platform.scope_ubuf)
            util_tik_comm_func.gm2ub(self.tik_instance, indices_ub, self.input_gm_list[1], indices_num)
            with self.tik_instance.for_range(self.indices_shape[0], indices_num) as i:
                indices_ub[i].set_as(self.neg_one_scalar)
            # init offset ub
            offset_num = mem_aligned("int32", self.offsets_shape[0])
            offset_ub = self.tik_instance.Tensor("int32", (offset_num,),
                                                 name="offset_ub", scope=tbe_platform.scope_ubuf)
            util_tik_comm_func.gm2ub(self.tik_instance, offset_ub, self.input_gm_list[2], offset_num)
            with self.tik_instance.for_range(self.offsets_shape[0], offset_num) as i:
                offset_ub[i].set_as(self.neg_one_scalar)
            indices_bag_ub = self.tik_instance.Tensor("int32", (self.offsets_shape[0], self.max_bag_size),
                                                      name="indices_bag_ub", scope=tbe_platform.scope_ubuf)
            util_tik_comm_func.tik_func_vector(self.tik_instance, indices_bag_ub, -1,
                                               self.offsets_shape[0] * self.max_bag_size)
            with self.tik_instance.for_range(0, self.offsets_shape[0]) as i:
                start_scalar = self.tik_instance.Scalar(dtype="int32", init_value=-1)
                end_scalar = self.tik_instance.Scalar(dtype="int32", init_value=-1)
                start_scalar.set_as(offset_ub[i])
                end_scalar.set_as(offset_ub[i + 1])
                # use end_scalar if unsafe
                with self.tik_instance.if_scope(end_scalar < 0):
                    end_scalar.set_as(indices_num)
                    with self.tik_instance.for_range(start_scalar, end_scalar) as j:
                        indices_bag_ub[i, j - start_scalar].set_as(indices_ub[j])
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(start_scalar, end_scalar) as j:
                        indices_bag_ub[i, j - start_scalar].set_as(indices_ub[j])
            self.offset_1d_count_compute(indices_bag_ub)
            util_tik_comm_func.ub2gm(self.tik_instance, self.indices_bag_gm, indices_bag_ub,
                                     self.offsets_shape[0] * self.max_bag_size)

    def offset_to_bag_2d(self):
        """ make offset_to_bag_2d
        """
        indices_round_size = mem_aligned("int32", self.indices_shape[1])
        indices_size = total_num(self.indices_shape)
        util_tik_comm_func.tik_func_vector(self.tik_instance, self.bag_count_ub, self.indices_shape[1],
                                           self.bag_count_ub.shape[0])
        util_tik_comm_func.tik_func_vector(self.tik_instance, self.bag_divided_ub, 1.0 / self.indices_shape[1],
                                           self.bag_divided_ub.shape[0])
        with self.tik_instance.new_stmt_scope():
            temp_min_ub = self.tik_instance.Tensor("int32", (8,), name="temp_min_ub", scope=tbe_platform.scope_ubuf)
            temp_bag_2d_ub = self.tik_instance.Tensor("int32", (self.max_bag_size,),
                                                      name="temp_bag_2d_ub", scope=tbe_platform.scope_ubuf)
            util_tik_comm_func.tik_func_vector(self.tik_instance, temp_bag_2d_ub, -1,
                                               self.max_bag_size)
            with self.tik_instance.if_scope(indices_size > 8):
                util_tik_comm_func.gm2ub(self.tik_instance, temp_min_ub,
                                         self.input_gm_list[1][indices_size - 8], 8)
                with self.tik_instance.for_range(0, self.indices_shape[0]) as i:
                    with self.tik_instance.if_scope(i * self.indices_shape[1] + indices_round_size < indices_size):
                        util_tik_comm_func.gm2ub(self.tik_instance, temp_bag_2d_ub,
                                                 self.input_gm_list[1][i * self.indices_shape[1]],
                                                 self.indices_shape[1])
                        with self.tik_instance.for_range(self.indices_shape[1], self.max_bag_size) as j:
                            temp_bag_2d_ub[j].set_as(self.neg_one_scalar)
                    with self.tik_instance.else_scope():
                        with self.tik_instance.if_scope(self.indices_shape[1] > 8):
                            # index is last slice
                            burst = burst_len_compute("int32", self.indices_shape[1])
                            if burst > 0:
                                self.tik_instance.data_move(temp_bag_2d_ub,
                                                            self.input_gm_list[1][i * self.indices_shape[1]],
                                                            0, 1, burst, 0, 0)
                            tail_size = tail_lens_cal("int32", self.indices_shape[1])
                            with self.tik_instance.for_range(0, tail_size) as k:
                                temp_bag_2d_ub[burst * 8 + k].set_as(temp_min_ub[8 - tail_size + k])
                        with self.tik_instance.else_scope():
                            with self.tik_instance.for_range(0, self.indices_shape[1]) as j:
                                tmp_index = j + 8 - indices_size + i * self.indices_shape[1]
                                temp_bag_2d_ub[j].set_as(temp_min_ub[tmp_index])
                    util_tik_comm_func.ub2gm(self.tik_instance, self.indices_bag_gm[i * self.max_bag_size],
                                             temp_bag_2d_ub, self.max_bag_size)
            with self.tik_instance.else_scope():
                util_tik_comm_func.gm2ub(self.tik_instance, temp_min_ub,
                                         self.input_gm_list[1], indices_size)
                with self.tik_instance.for_range(0, self.indices_shape[0]) as i:
                    with self.tik_instance.for_range(0, self.indices_shape[1]) as j:
                        temp_bag_2d_ub[j].set_as(temp_min_ub[i * self.indices_shape[1] + j])
                    util_tik_comm_func.ub2gm(self.tik_instance, self.indices_bag_gm[i * self.max_bag_size],
                                             temp_bag_2d_ub, self.max_bag_size)

    def offset_to_bag(self):
        """make offset to bag
        """
        bag_divided_num = mem_aligned(self.weight_dtype, self.output_shape[0])
        bag_count_num = mem_aligned("int32", self.output_shape[0])

        self.bag_divided_ub = self.tik_instance.Tensor(self.weight_dtype, (bag_divided_num,),
                                                       name="bag_divided_ub", scope=tbe_platform.scope_ubuf)
        self.bag_count_ub = self.tik_instance.Tensor("int32", (bag_count_num,),
                                                     name="bag_count_ub", scope=tbe_platform.scope_ubuf)
        # get offset to bag
        if self.is_single_indices:
            self.offset_to_bag_1d()
        else:
            self.offset_to_bag_2d()

    def embedding_bag_sum_compute(self):
        """embedding sum compute
        """
        output_total_size = total_num(self.output_shape)

        with self.tik_instance.new_stmt_scope():
            # init weight ub for compute
            weight_num = mem_aligned(self.weight_dtype, self.embedding_dim)
            result_sum_ub = self.tik_instance.Tensor(self.weight_dtype, (weight_num,),
                                                     name="result_sum_ub", scope=tbe_platform.scope_ubuf)
            temp_sum_ub = self.tik_instance.Tensor(self.weight_dtype, (weight_num,),
                                                   name="temp_sum_ub", scope=tbe_platform.scope_ubuf)
            util_tik_comm_func.tik_func_vector(self.tik_instance, result_sum_ub, 0, total_num(result_sum_ub.shape))
            util_tik_comm_func.tik_func_vector(self.tik_instance, temp_sum_ub, 0, total_num(temp_sum_ub.shape))

            # init bag ub to compute indices
            bag_temp_ub = self.tik_instance.Tensor("int32", (self.max_bag_size,),
                                                   name="bag_temp_ub", scope=tbe_platform.scope_ubuf)
            bag_temp_scalar = self.tik_instance.Scalar(dtype="int32")
            index_scalar = self.tik_instance.Scalar(dtype="int32")
            valid_count_scalar = self.tik_instance.Scalar(dtype="int32")
            with self.tik_instance.for_range(0, self.output_shape[0]) as i:
                index_scalar.set_as(i)
                # move indices from bag gm to bag ub
                util_tik_comm_func.gm2ub(self.tik_instance, bag_temp_ub, self.indices_bag_gm[i * self.max_bag_size],
                                         self.max_bag_size)
                bag_temp_scalar.set_as(bag_temp_ub[0])

                self.gather_from_weight(result_sum_ub, self.input_gm_list[0], bag_temp_scalar)
                valid_count_scalar.set_as(self.bag_count_ub[i])
                with self.tik_instance.for_range(1, valid_count_scalar) as j:
                    bag_temp_scalar.set_as(bag_temp_ub[j])
                    with self.tik_instance.if_scope(bag_temp_scalar != -1):
                        self.gather_from_weight(temp_sum_ub, self.input_gm_list[0], bag_temp_scalar)
                        util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vadd", result_sum_ub, result_sum_ub,
                                                            temp_sum_ub, weight_num)
                with self.tik_instance.if_scope(output_total_size < Constant.WEIGHT_NUM_MIN):
                    with self.tik_instance.for_range(0, self.embedding_dim) as i:
                        self.output_ub_temp[index_scalar * self.embedding_dim + i].set_as(result_sum_ub[i])
                with self.tik_instance.else_scope():
                    self.embedding_bag_out_by_index(self.output_gm_list[0], result_sum_ub, index_scalar)
            with self.tik_instance.if_scope(output_total_size < Constant.WEIGHT_NUM_MIN):
                self.embedding_bag_one_time_out()
            with self.tik_instance.else_scope():
                # move tail data
                tail_index = total_num(self.output_shape) - self.tail_size
                util_tik_comm_func.ub2gm(self.tik_instance, self.output_gm_list[0][tail_index], self.output_ub_tail,
                                         self.tail_size)

    def embedding_bag_mean_compute(self):
        """embedding_bag_mean_compute
        """
        output_total_size = total_num(self.output_shape)

        with self.tik_instance.new_stmt_scope():
            # init weight ub for compute
            weight_num = mem_aligned(self.weight_dtype, self.embedding_dim)
            result_sum_ub = self.tik_instance.Tensor(self.weight_dtype, (weight_num,),
                                                     name="result_sum_ub", scope=tbe_platform.scope_ubuf)
            temp_sum_ub = self.tik_instance.Tensor(self.weight_dtype, (weight_num,),
                                                   name="temp_sum_ub", scope=tbe_platform.scope_ubuf)
            util_tik_comm_func.tik_func_vector(self.tik_instance, result_sum_ub, 0, total_num(result_sum_ub.shape))
            util_tik_comm_func.tik_func_vector(self.tik_instance, temp_sum_ub, 0, total_num(temp_sum_ub.shape))

            # init bag ub to compute indices
            bag_temp_ub = self.tik_instance.Tensor("int32", (self.max_bag_size,),
                                                   name="bag_temp_ub", scope=tbe_platform.scope_ubuf)
            bag_temp_scalar = self.tik_instance.Scalar(dtype="int32")
            index_scalar = self.tik_instance.Scalar(dtype="int32")
            valid_count_scalar = self.tik_instance.Scalar(dtype="int32")
            divided_scalar = self.tik_instance.Scalar(dtype=self.weight_dtype)

            with self.tik_instance.for_range(0, self.output_shape[0]) as i:
                index_scalar.set_as(i)
                # move indices from bag gm to bag ub
                util_tik_comm_func.gm2ub(self.tik_instance, bag_temp_ub, self.indices_bag_gm[i * self.max_bag_size],
                                         self.max_bag_size)
                bag_temp_scalar.set_as(bag_temp_ub[0])

                self.gather_from_weight(result_sum_ub, self.input_gm_list[0], bag_temp_scalar)
                valid_count_scalar.set_as(self.bag_count_ub[i])
                with self.tik_instance.for_range(1, valid_count_scalar) as j:
                    bag_temp_scalar.set_as(bag_temp_ub[j])
                    with self.tik_instance.if_scope(bag_temp_scalar != -1):
                        self.gather_from_weight(temp_sum_ub, self.input_gm_list[0], bag_temp_scalar)
                        util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vadd", result_sum_ub, result_sum_ub,
                                                            temp_sum_ub, weight_num)
                divided_scalar.set_as(self.bag_divided_ub[i])
                util_tik_comm_func.tik_func_vmuls(self.tik_instance, result_sum_ub, result_sum_ub, divided_scalar,
                                                  weight_num)
                with self.tik_instance.if_scope(output_total_size < Constant.WEIGHT_NUM_MIN):
                    with self.tik_instance.for_range(0, self.embedding_dim) as i:
                        self.output_ub_temp[index_scalar * self.embedding_dim + i].set_as(result_sum_ub[i])
                with self.tik_instance.else_scope():
                    self.embedding_bag_out_by_index(self.output_gm_list[0], result_sum_ub, index_scalar)
            with self.tik_instance.if_scope(output_total_size < Constant.WEIGHT_NUM_MIN):
                self.embedding_bag_one_time_out()
            with self.tik_instance.else_scope():
                # move tail data
                tail_index = total_num(self.output_shape) - self.tail_size
                util_tik_comm_func.ub2gm(self.tik_instance, self.output_gm_list[0][tail_index], self.output_ub_tail,
                                         self.tail_size)

    def embedding_bag_max_compute(self):
        """embedding_bag_max_compute
        """
        output_total_size = total_num(self.output_shape)

        with self.tik_instance.new_stmt_scope():
            # init weight ub for compute
            weight_num = mem_aligned(self.weight_dtype, self.embedding_dim)
            result_sum_ub = self.tik_instance.Tensor(self.weight_dtype, (weight_num,),
                                                     name="result_sum_ub", scope=tbe_platform.scope_ubuf)
            temp_sum_ub = self.tik_instance.Tensor(self.weight_dtype, (weight_num,),
                                                   name="temp_sum_ub", scope=tbe_platform.scope_ubuf)
            util_tik_comm_func.tik_func_vector(self.tik_instance, result_sum_ub, 0, total_num(result_sum_ub.shape))
            util_tik_comm_func.tik_func_vector(self.tik_instance, temp_sum_ub, 0, total_num(temp_sum_ub.shape))

            # init bag ub to compute indices
            bag_temp_ub = self.tik_instance.Tensor("int32", (self.max_bag_size,),
                                                   name="bag_temp_ub", scope=tbe_platform.scope_ubuf)
            bag_temp_scalar = self.tik_instance.Scalar(dtype="int32")
            index_scalar = self.tik_instance.Scalar(dtype="int32")
            valid_count_scalar = self.tik_instance.Scalar(dtype="int32")
            with self.tik_instance.for_range(0, self.output_shape[0]) as i:
                index_scalar.set_as(i)
                # move indices from bag gm to bag ub
                util_tik_comm_func.gm2ub(self.tik_instance, bag_temp_ub, self.indices_bag_gm[i * self.max_bag_size],
                                         self.max_bag_size)
                bag_temp_scalar.set_as(bag_temp_ub[0])

                self.gather_from_weight(result_sum_ub, self.input_gm_list[0], bag_temp_scalar)
                valid_count_scalar.set_as(self.bag_count_ub[i])
                with self.tik_instance.for_range(1, valid_count_scalar) as j:
                    bag_temp_scalar.set_as(bag_temp_ub[j])
                    with self.tik_instance.if_scope(bag_temp_scalar != -1):
                        self.gather_from_weight(temp_sum_ub, self.input_gm_list[0], bag_temp_scalar)
                        util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vmax", result_sum_ub, result_sum_ub,
                                                            temp_sum_ub, weight_num)
                with self.tik_instance.if_scope(output_total_size < Constant.WEIGHT_NUM_MIN):
                    with self.tik_instance.for_range(0, self.embedding_dim) as i:
                        self.output_ub_temp[index_scalar * self.embedding_dim + i].set_as(result_sum_ub[i])
                with self.tik_instance.else_scope():
                    self.embedding_bag_out_by_index(self.output_gm_list[0], result_sum_ub, index_scalar)
            with self.tik_instance.if_scope(output_total_size < Constant.WEIGHT_NUM_MIN):
                self.embedding_bag_one_time_out()
            with self.tik_instance.else_scope():
                # move tail data
                tail_index = total_num(self.output_shape) - self.tail_size
                util_tik_comm_func.ub2gm(self.tik_instance, self.output_gm_list[0][tail_index], self.output_ub_tail,
                                         self.tail_size)

    def embedding_bag_out_by_index(self, output_gm, temp_result_ub, index):
        """embedding_bag_out_by_index
        """
        total_output_size = total_num(self.output_shape)
        with self.tik_instance.if_scope(index * self.embedding_dim + temp_result_ub.shape[0] < total_output_size):
            with self.tik_instance.if_scope(total_output_size - ((index + 1) * self.embedding_dim < self.tail_size)):
                tmp_lens = total_output_size - (index + 1) * self.embedding_dim
                x1 = self.tail_size - tmp_lens
                start_index = self.embedding_dim - x1
                with self.tik_instance.for_range(start_index, self.embedding_dim) as i:
                    tail_index = i - start_index
                    self.output_ub_tail[tail_index].set_as(temp_result_ub[i])
            util_tik_comm_func.ub2gm(self.tik_instance, output_gm[index * self.embedding_dim], temp_result_ub,
                                     temp_result_ub.shape[0])
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(self.embedding_dim > self.tail_size):
                burst = burst_len_compute(self.weight_dtype, self.embedding_dim)
                if burst > 0:
                    self.tik_instance.data_move(output_gm[index * self.embedding_dim], temp_result_ub,
                                                0, 1, burst, 0, 0)
                    with self.tik_instance.for_range(self.embedding_dim - self.tail_size, self.embedding_dim) as i:
                        self.output_ub_tail[i - self.embedding_dim + self.tail_size] = temp_result_ub[i]
            with self.tik_instance.else_scope():
                tmp_lens = total_output_size - index * self.embedding_dim
                start_index = self.tail_size - tmp_lens
                with self.tik_instance.for_range(0, self.embedding_dim) as i:
                    tail_index = start_index + i
                    self.output_ub_tail[tail_index].set_as(temp_result_ub[i])

    def embedding_bag_one_time_out(self):
        """embedding_bag_one_time_out
        """
        output_total_size = total_num(self.output_shape)
        output_num = mem_aligned(self.weight_dtype, output_total_size)
        util_tik_comm_func.ub2gm(self.tik_instance, self.output_gm_list[0], self.output_ub_temp,
                                 output_num)

    def gather_from_weight(self, slice_ub, weight_gm, index):
        """gather_from_weight
        """
        weight_total_size = total_num(self.weight_shape)
        if weight_total_size <= Constant.WEIGHT_NUM_MIN:
            with self.tik_instance.for_range(0, self.embedding_dim) as i:
                slice_ub[i].set_as(self.weight_ub_temp[index * self.embedding_dim + i])
        else:
            with self.tik_instance.if_scope(index < self.weight_shape[0] - 1):
                with self.tik_instance.if_scope((weight_total_size - index * self.embedding_dim) > 32):
                    util_tik_comm_func.gm2ub(self.tik_instance, slice_ub, weight_gm[index * self.embedding_dim],
                                             self.embedding_dim)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, self.embedding_dim) as i:
                        weight_offset = i + 128 - (weight_total_size - index * self.embedding_dim)
                        slice_ub[i].set_as(self.weight_ub_temp[weight_offset])
            with self.tik_instance.else_scope():
                if self.embedding_dim > 32:
                    burst = burst_len_compute(self.weight_dtype, self.embedding_dim)
                    self.tik_instance.data_move(slice_ub, weight_gm[index * self.embedding_dim], 0, 1, burst, 0, 0)
                    tail_lens = tail_lens_cal(self.weight_dtype, self.embedding_dim)
                    with self.tik_instance.for_range(0, tail_lens) as i:
                        slice_ub[self.embedding_dim - i - 1].set_as(self.weight_ub_temp[127 - i])
                else:
                    with self.tik_instance.for_range(0, self.embedding_dim) as i:
                        slice_ub[self.embedding_dim - i - 1].set_as(self.weight_ub_temp[127 - i])

    def embedding_bag_compute(self):
        """embedding_bag_compute
        """
        self.offset_to_bag()
        # select mode
        if self.mode == "sum":
            self.embedding_bag_sum_compute()
        elif self.mode == "mean":
            self.embedding_bag_mean_compute()
        else:
            self.embedding_bag_max_compute()


def work_tensor_size_compute(dtype, block_len, repeat_times, src_rep_stride, mask_len):
    """return size of work tensor
    """
    src_extent_size = (repeat_times - 1) * src_rep_stride * block_len + mask_len
    wk_size_unit = ((src_extent_size + block_len - 1) // block_len) * block_len
    if dtype == "float16":
        work_size = 4 * wk_size_unit
    else:
        work_size = 2 * wk_size_unit
    return work_size


def mem_aligned(dtype, in_num):
    """aligned mem for ub
    """
    out_num = 0
    if dtype in ["int32", "float32"]:
        out_num = ceil_div(in_num, 8) * 8
    elif dtype in ["float16"]:
        out_num = ceil_div(in_num, 16) * 16
    else:
        RuntimeError("dtype is not support !!")
    return out_num


def burst_len_compute(dtype, in_num):
    """burst_len compute
    """
    out_num = 0
    if dtype in ["int32", "float32"]:
        out_num = ceil_div(in_num, 8) - 1
    elif dtype in ["float16"]:
        out_num = ceil_div(in_num, 16) - 1
    else:
        RuntimeError("dtype is not support !!")
    return out_num


def tail_lens_cal(dtype, in_num):
    """tail_lens cal
    """
    out_num = 0
    if dtype in ["int32", "float32"]:
        out_num = in_num % 8
    elif dtype in ["float16"]:
        out_num = in_num % 16
    else:
        RuntimeError("dtype is not support !!")
    return out_num


def min_data_block(dtype):
    """min_data_block
    """
    out_num = 0
    if dtype in ["int32", "float32"]:
        out_num = 8
    elif dtype in ["float16"]:
        out_num = 16
    else:
        RuntimeError("dtype is not support !!")
    return out_num


def total_num(shape):
    """total_num"""
    shape_total_num = functools_reduce(lambda a, b: a * b, shape)
    return shape_total_num


def ceil_div(value, factor):
    """Compute the smallest integer value that is greater than
    or equal to value/factor
    """
    result = (value + (factor - 1)) // factor
    return result


def check_indices_param(indices_dic):
    """check indices param"""
    indices_dtype = indices_dic.get("dtype").lower()
    indices_shape = indices_dic.get("shape")
    para_check.check_shape(indices_shape)
    para_check.check_dtype(indices_dtype, ["int32"])


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def embedding_bag(weight, indices,
                  offsets, per_sample_weights,
                  y, mode='mean',
                  scale_grid_by_freq=False, sparse=False,
                  include_last_offset=False, kernel_name="embedding_bag"):
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
    # check para
    check_indices_param(indices)
    em_bag = EmbeddingBag(weight, indices, offsets,
                          per_sample_weights, mode,
                          scale_grid_by_freq,
                          sparse, include_last_offset)
    # init gm mem
    em_bag.init_tik_mem()
    # init ub
    em_bag.init_ub_mem()
    # embedding compute
    em_bag.embedding_bag_compute()
    return em_bag.build_tik_instance(kernel_name)
