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
sub_sample_labels
"""

from functools import reduce as functools_reduce
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util import util_tik_comm_func

# 'pylint: disable=too-many-lines,too-many-instance-attributes,too-many-statements
# 'pylint: disable=too-many-arguments,unused-argument,too-many-locals,too-many-branches

# process each batch
BATCH_SIZE = 10000


class SubSampleLabels(object):
    """
    Function: use to store SubSampleLabels base parameters
    Modify : 2021-03-20
    """

    def __init__(self,
                 labels,
                 shuffle_matrix,
                 batch_size_per_images,
                 positive_fraction):
        """
        Init SubSampleLabels base parameters

        Returns
        -------
        None
        """
        # define general var
        self.tik_instance = tik.Tik()
        self.aicore_num = \
            tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size = \
            tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.input_gm_list = []
        self.output_gm_list = []

        # get labels shape and dtype
        self.labels_shape = list(labels.get("shape"))
        self.labels_dtype = labels.get("dtype")

        self.shuffle_matrix_shape = list(shuffle_matrix.get("shape"))
        self.shuffle_matrix_dtype = shuffle_matrix.get("dtype")

        # attr
        self.batch_size_per_images = batch_size_per_images
        self.positive_fraction = positive_fraction

        # init gm
        self.pos_mask_gm = None
        self.neg_mask_gm = None

        self.labels_num = mem_aligned("int32", self.labels_shape[0])
        self.loop_time = self.labels_shape[0] // BATCH_SIZE
        self.tail_num = self.labels_shape[0] % BATCH_SIZE

        # init scalar
        self.two_scalar = self.tik_instance.Scalar(dtype="int32", init_value=2)
        self.one_scalar = self.tik_instance.Scalar(dtype="int32", init_value=1)
        self.zero_scalar = self.tik_instance.Scalar(dtype="int32", init_value=0)
        self.neg_one_scalar = self.tik_instance.Scalar(dtype="int32", init_value=-1)
        # init pos and neg num
        self.neg_num_scalar = self.tik_instance.Scalar(dtype="int32", init_value=-1)

        self.pos_num_scalar = self.get_pos_num()

    def get_pos_num(self):
        """get_pos_num
        """
        pos_num = int(self.batch_size_per_images * self.positive_fraction)
        pos_num_scalar = self.tik_instance.Scalar(dtype="int32", init_value=pos_num)
        return pos_num_scalar

    def init_tik_mem(self):
        """init tik gm mem
        """
        # init gm input
        labels_gm = self.tik_instance.Tensor(self.labels_dtype, self.labels_shape, name="labels_gm", scope=tik.scope_gm)
        shuffle_matrix_gm = self.tik_instance.Tensor(self.shuffle_matrix_dtype, self.shuffle_matrix_shape,
                                                     name="shuffle_matrix_gm", scope=tik.scope_gm)

        self.input_gm_list = [labels_gm, shuffle_matrix_gm]

        # init gm output
        labels_output_gm = self.tik_instance.Tensor(self.labels_dtype, self.labels_shape,
                                                    name="labels_output_gm",
                                                    scope=tik.scope_gm)

        self.output_gm_list = [labels_output_gm]

        # init mid gm

        self.pos_mask_gm = self.tik_instance.Tensor("int32", (self.labels_num,),
                                                    name="pos_mask_gm",
                                                    scope=tik.scope_gm, is_workspace=True)

        self.neg_mask_gm = self.tik_instance.Tensor("int32", (self.labels_num,),
                                                    name="neg_mask_gm",
                                                    scope=tik.scope_gm, is_workspace=True)

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

    def label_mask_compute_per_batch(self, loop_index):
        """compute positive and negtive mask for labels
        """
        with self.tik_instance.new_stmt_scope():
            pos_mask_ub = self.tik_instance.Tensor("int32", (BATCH_SIZE,),
                                                   name="pos_mask_ub", scope=tik.scope_ubuf)
            neg_mask_ub = self.tik_instance.Tensor("int32", (BATCH_SIZE,),
                                                   name="neg_mask_ub", scope=tik.scope_ubuf)
            labels_ub = self.tik_instance.Tensor("int32", (BATCH_SIZE,),
                                                 name="labels_ub", scope=tik.scope_ubuf)

            util_tik_comm_func.tik_func_vector(self.tik_instance, pos_mask_ub, 0, BATCH_SIZE)
            util_tik_comm_func.tik_func_vector(self.tik_instance, neg_mask_ub, 0, BATCH_SIZE)

            util_tik_comm_func.gm2ub(self.tik_instance, labels_ub,
                                     self.input_gm_list[0][loop_index * BATCH_SIZE],
                                     BATCH_SIZE)
            with self.tik_instance.for_range(0, BATCH_SIZE) as i:
                with self.tik_instance.if_scope(labels_ub[i] == self.one_scalar):
                    pos_mask_ub[i].set_as(self.two_scalar)
                with self.tik_instance.if_scope(labels_ub[i] == self.zero_scalar):
                    neg_mask_ub[i].set_as(self.one_scalar)
            util_tik_comm_func.ub2gm(self.tik_instance, self.pos_mask_gm[loop_index * BATCH_SIZE], pos_mask_ub,
                                     BATCH_SIZE)
            util_tik_comm_func.ub2gm(self.tik_instance, self.neg_mask_gm[loop_index * BATCH_SIZE], neg_mask_ub,
                                     BATCH_SIZE)

    def label_mask_tail_compute(self, offset, tail_size):
        """label_mask_tail_compute
        """
        with self.tik_instance.new_stmt_scope():
            tail_ub_size = mem_aligned("int32", tail_size)
            pos_mask_tail_ub = self.tik_instance.Tensor("int32", (tail_ub_size,),
                                                        name="pos_mask_tail_ub", scope=tik.scope_ubuf)
            neg_mask_tail_ub = self.tik_instance.Tensor("int32", (tail_ub_size,),
                                                        name="neg_mask_tail_ub", scope=tik.scope_ubuf)

            labels_tail_ub = self.tik_instance.Tensor("int32", (tail_ub_size,),
                                                      name="labels_tail_ub", scope=tik.scope_ubuf)
            util_tik_comm_func.tik_func_vector(self.tik_instance, pos_mask_tail_ub, 0, tail_ub_size)
            util_tik_comm_func.tik_func_vector(self.tik_instance, neg_mask_tail_ub, 0, tail_ub_size)
            # init ub mem

            tail_ub = self.tik_instance.Tensor("int32", (8,),
                                               name="tail_ub", scope=tik.scope_ubuf)
            offset_tail_count = self.labels_shape[0] - 8
            util_tik_comm_func.gm2ub(self.tik_instance, tail_ub,
                                     self.input_gm_list[0][offset_tail_count], 8)
            tail_copy_num = (tail_size // 8) * 8
            # may be out of range
            util_tik_comm_func.gm2ub(self.tik_instance, labels_tail_ub, self.input_gm_list[0][offset],
                                     tail_copy_num)
            with self.tik_instance.for_range(0, 8) as i:
                labels_tail_ub[tail_size - i - 1].set_as(tail_ub[7 - i])
            with self.tik_instance.for_range(0, tail_size) as i:
                with self.tik_instance.if_scope(labels_tail_ub[i] == self.one_scalar):
                    pos_mask_tail_ub[i].set_as(self.two_scalar)
                with self.tik_instance.if_scope(labels_tail_ub[i] == self.zero_scalar):
                    neg_mask_tail_ub[i].set_as(self.one_scalar)
            util_tik_comm_func.ub2gm(self.tik_instance, self.pos_mask_gm[offset], pos_mask_tail_ub, tail_size)
            util_tik_comm_func.ub2gm(self.tik_instance, self.neg_mask_gm[offset], neg_mask_tail_ub, tail_size)

    def label_mask_compute(self):
        """label_mask_compute
        """
        with self.tik_instance.new_stmt_scope():
            loop_time = self.loop_time
            tail_size = self.tail_num
            tail_offset = loop_time * BATCH_SIZE
            with self.tik_instance.for_range(0, loop_time) as i:
                self.label_mask_compute_per_batch(i)
            with self.tik_instance.if_scope(tail_size > 0):
                self.label_mask_tail_compute(tail_offset, tail_size)

    def update_pos_mask_per_batch(self, pos_mask_ub_temp,
                                  count_ub, one_ub, loop_index):
        """update_pos_mask_per_batch
        """
        with self.tik_instance.new_stmt_scope():
            shuffle_batch_ub = self.tik_instance.Tensor("int32", (BATCH_SIZE,),
                                                        name="shuffle_batch_ub", scope=tik.scope_ubuf)
            util_tik_comm_func.gm2ub(self.tik_instance, shuffle_batch_ub,
                                     self.input_gm_list[1][loop_index * BATCH_SIZE],
                                     BATCH_SIZE)
            pos_index_scalar = self.tik_instance.Scalar(dtype="int32", init_value=0)
            with self.tik_instance.for_range(0, BATCH_SIZE) as i:
                pos_index_scalar.set_as(shuffle_batch_ub[i])
                with self.tik_instance.if_scope(pos_mask_ub_temp[pos_index_scalar] == self.two_scalar):
                    with self.tik_instance.if_scope(count_ub[0] < self.pos_num_scalar):
                        util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vadd", count_ub, count_ub,
                                                            one_ub, 16)
                    with self.tik_instance.else_scope():
                        pos_mask_ub_temp[pos_index_scalar].set_as(self.zero_scalar)

    def update_pos_mask_tail(self, pos_mask_ub_temp, shuffle_tail_ub,
                             count_ub, one_ub, offset, tail_size):
        """update_pos_mask_tail
        """
        with self.tik_instance.new_stmt_scope():
            tail_pos_ub_size = mem_aligned("int32", tail_size)
            shuffle_head_ub = self.tik_instance.Tensor("int32", (tail_pos_ub_size,),
                                                       name="shuffle_head_ub",
                                                       scope=tik.scope_ubuf)
            move_num = (tail_size // 8) * 8
            util_tik_comm_func.gm2ub(self.tik_instance, shuffle_head_ub,
                                     self.input_gm_list[1][offset], move_num)
            with self.tik_instance.for_range(0, 8) as i:
                shuffle_head_ub[tail_size - 8 + i] = shuffle_tail_ub[i]
            pos_index_tail_scalar = self.tik_instance.Scalar(dtype="int32", init_value=0)

            with self.tik_instance.for_range(0, tail_size) as i:
                pos_index_tail_scalar.set_as(shuffle_head_ub[i])
                with self.tik_instance.if_scope(pos_mask_ub_temp[pos_index_tail_scalar] == self.two_scalar):
                    with self.tik_instance.if_scope(count_ub[0] < self.pos_num_scalar):
                        util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vadd", count_ub, count_ub,
                                                            one_ub, 16)
                    with self.tik_instance.else_scope():
                        pos_mask_ub_temp[pos_index_tail_scalar].set_as(self.zero_scalar)

    def update_pos_mask(self):
        """update_pos_mask
        """
        with self.tik_instance.new_stmt_scope():
            loop_time = self.loop_time
            tail_size = self.tail_num
            tail_offset = loop_time * BATCH_SIZE
            pos_mask_ub_temp = self.tik_instance.Tensor("int32", (self.labels_num,),
                                                        name="pos_mask_ub_temp", scope=tik.scope_ubuf)
            util_tik_comm_func.gm2ub(self.tik_instance, pos_mask_ub_temp,
                                     self.pos_mask_gm, self.labels_num)

            shuffle_tail_ub = self.tik_instance.Tensor("int32", (8,),
                                                       name="shuffle_tail_ub", scope=tik.scope_ubuf)
            shuffle_tail_count = self.labels_shape[0] - 8
            util_tik_comm_func.gm2ub(self.tik_instance, shuffle_tail_ub,
                                     self.input_gm_list[1][shuffle_tail_count], 8)
            count_ub = self.tik_instance.Tensor("int32", (16,),
                                                name="count_ub", scope=tik.scope_ubuf)
            one_ub = self.tik_instance.Tensor("int32", (16,),
                                              name="one_ub", scope=tik.scope_ubuf)
            util_tik_comm_func.tik_func_vector(self.tik_instance, count_ub, 0, 16)
            util_tik_comm_func.tik_func_vector(self.tik_instance, one_ub, 1, 16)
            with self.tik_instance.for_range(0, loop_time) as i:
                self.update_pos_mask_per_batch(pos_mask_ub_temp, count_ub, one_ub, i)
            with self.tik_instance.if_scope(tail_size > 0):
                self.update_pos_mask_tail(pos_mask_ub_temp, shuffle_tail_ub,
                                          count_ub, one_ub, tail_offset, tail_size)
            with self.tik_instance.if_scope(count_ub[0] < self.pos_num_scalar):
                self.pos_num_scalar.set_as(count_ub[0])
            util_tik_comm_func.ub2gm(self.tik_instance, self.pos_mask_gm, pos_mask_ub_temp, self.labels_shape[0])

    def update_neg_mask_per_batch(self, neg_mask_ub_temp, count_ub_neg,
                                  one_ub_neg, loop_index):
        """update_neg_mask_per_batch
        """
        with self.tik_instance.new_stmt_scope():
            shuffle_batch_ub_neg = self.tik_instance.Tensor("int32", (BATCH_SIZE,),
                                                            name="shuffle_batch_ub_neg", scope=tik.scope_ubuf)
            util_tik_comm_func.gm2ub(self.tik_instance, shuffle_batch_ub_neg,
                                     self.input_gm_list[1][loop_index * BATCH_SIZE],
                                     BATCH_SIZE)
            neg_index_scalar = self.tik_instance.Scalar(dtype="int32", init_value=0)
            with self.tik_instance.for_range(0, BATCH_SIZE) as i:
                neg_index_scalar.set_as(shuffle_batch_ub_neg[i])
                with self.tik_instance.if_scope(neg_mask_ub_temp[neg_index_scalar] == self.one_scalar):
                    with self.tik_instance.if_scope(count_ub_neg[0] < self.neg_num_scalar):
                        util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vadd", count_ub_neg, count_ub_neg,
                                                            one_ub_neg, 16)
                    with self.tik_instance.else_scope():
                        neg_mask_ub_temp[neg_index_scalar].set_as(self.zero_scalar)

    def update_neg_mask_tail(self, neg_mask_ub_temp, shuffle_tail_neg_ub,
                             count_ub_neg, one_ub_neg, offset, tail_size):
        """update_neg_mask_tail
        """
        with self.tik_instance.new_stmt_scope():
            tail_neg_ub_size = mem_aligned("int32", tail_size)
            shuffle_head_ub_neg = self.tik_instance.Tensor("int32", (tail_neg_ub_size,),
                                                           name="shuffle_head_ub_neg",
                                                           scope=tik.scope_ubuf)
            move_num = (tail_size // 8) * 8
            util_tik_comm_func.gm2ub(self.tik_instance, shuffle_head_ub_neg,
                                     self.input_gm_list[1][offset], move_num)
            with self.tik_instance.for_range(0, 8) as i:
                shuffle_head_ub_neg[tail_size - 8 + i] = shuffle_tail_neg_ub[i]
            neg_index_tail_scalar = self.tik_instance.Scalar(dtype="int32", init_value=0)

            with self.tik_instance.for_range(0, tail_size) as i:
                neg_index_tail_scalar.set_as(shuffle_head_ub_neg[i])
                with self.tik_instance.if_scope(neg_mask_ub_temp[neg_index_tail_scalar] == self.one_scalar):
                    with self.tik_instance.if_scope(count_ub_neg[0] < self.neg_num_scalar):
                        util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vadd", count_ub_neg, count_ub_neg,
                                                            one_ub_neg, 16)
                    with self.tik_instance.else_scope():
                        neg_mask_ub_temp[neg_index_tail_scalar].set_as(self.zero_scalar)

    def update_neg_mask(self):
        """update_pos_mask
        """
        with self.tik_instance.new_stmt_scope():
            self.neg_num_scalar = self.batch_size_per_images - self.pos_num_scalar
            loop_time = self.loop_time
            tail_size = self.tail_num
            tail_offset = loop_time * BATCH_SIZE
            neg_mask_ub_temp = self.tik_instance.Tensor("int32", (self.labels_num,),
                                                        name="neg_mask_ub_temp", scope=tik.scope_ubuf)
            util_tik_comm_func.gm2ub(self.tik_instance, neg_mask_ub_temp, self.neg_mask_gm, self.labels_num)

            shuffle_tail_neg_ub = self.tik_instance.Tensor("int32", (8,),
                                                           name="shuffle_tail_neg_ub", scope=tik.scope_ubuf)
            shuffle_tail_neg_count = self.labels_shape[0] - 8
            util_tik_comm_func.gm2ub(self.tik_instance, shuffle_tail_neg_ub,
                                     self.input_gm_list[1][shuffle_tail_neg_count], 8)
            count_ub_neg = self.tik_instance.Tensor("int32", (16,),
                                                    name="count_ub_neg", scope=tik.scope_ubuf)
            one_ub_neg = self.tik_instance.Tensor("int32", (16,),
                                                  name="one_ub_neg", scope=tik.scope_ubuf)
            util_tik_comm_func.tik_func_vector(self.tik_instance, count_ub_neg, 0, 16)
            util_tik_comm_func.tik_func_vector(self.tik_instance, one_ub_neg, 1, 16)
            with self.tik_instance.for_range(0, loop_time) as i:
                self.update_neg_mask_per_batch(neg_mask_ub_temp, count_ub_neg, one_ub_neg, i)
            with self.tik_instance.if_scope(tail_size > 0):
                self.update_neg_mask_tail(neg_mask_ub_temp, shuffle_tail_neg_ub,
                                          count_ub_neg, one_ub_neg, tail_offset, tail_size)

            util_tik_comm_func.ub2gm(self.tik_instance, self.neg_mask_gm, neg_mask_ub_temp, self.labels_shape[0])

    def labels_output_per_batch(self, loop_index):
        """labels_output_per_batch
        """
        with self.tik_instance.new_stmt_scope():
            labels_ub_out = self.tik_instance.Tensor("int32", (BATCH_SIZE,),
                                                     name="labels_ub_out", scope=tik.scope_ubuf)
            pos_mask_ub_out = self.tik_instance.Tensor("int32", (BATCH_SIZE,),
                                                       name="pos_mask_ub_out", scope=tik.scope_ubuf)

            neg_mask_ub_out = self.tik_instance.Tensor("int32", (BATCH_SIZE,),
                                                       name="neg_mask_ub_out", scope=tik.scope_ubuf)

            util_tik_comm_func.gm2ub(self.tik_instance, pos_mask_ub_out,
                                     self.pos_mask_gm[loop_index * BATCH_SIZE],
                                     BATCH_SIZE)
            util_tik_comm_func.gm2ub(self.tik_instance, neg_mask_ub_out,
                                     self.neg_mask_gm[loop_index * BATCH_SIZE],
                                     BATCH_SIZE)

            util_tik_comm_func.tik_func_vector(self.tik_instance, labels_ub_out, -1, BATCH_SIZE)
            util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vadd", labels_ub_out, labels_ub_out,
                                                pos_mask_ub_out, BATCH_SIZE)

            util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vadd", labels_ub_out, labels_ub_out,
                                                neg_mask_ub_out, BATCH_SIZE)

            util_tik_comm_func.ub2gm(self.tik_instance, self.output_gm_list[0][loop_index * BATCH_SIZE], labels_ub_out,
                                     BATCH_SIZE)

    def labels_output_tail(self, offset, tail_size):
        """labels_output_tail
        """
        with self.tik_instance.new_stmt_scope():
            tail_ub_size = mem_aligned("int32", tail_size)
            labels_ub_out_tail = self.tik_instance.Tensor("int32", (tail_ub_size,),
                                                          name="labels_ub_out_tail", scope=tik.scope_ubuf)
            pos_mask_ub_out_tail = self.tik_instance.Tensor("int32", (tail_ub_size,),
                                                            name="pos_mask_ub_out_tail", scope=tik.scope_ubuf)

            neg_mask_ub_out_tail = self.tik_instance.Tensor("int32", (tail_ub_size,),
                                                            name="neg_mask_ub_out_tail", scope=tik.scope_ubuf)

            util_tik_comm_func.gm2ub(self.tik_instance, pos_mask_ub_out_tail,
                                     self.pos_mask_gm[offset],
                                     tail_ub_size)
            util_tik_comm_func.gm2ub(self.tik_instance, neg_mask_ub_out_tail,
                                     self.neg_mask_gm[offset],
                                     tail_ub_size)

            util_tik_comm_func.tik_func_vector(self.tik_instance, labels_ub_out_tail, -1, tail_ub_size)

            util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vadd", labels_ub_out_tail, labels_ub_out_tail,
                                                pos_mask_ub_out_tail, tail_ub_size)

            util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vadd", labels_ub_out_tail, labels_ub_out_tail,
                                                neg_mask_ub_out_tail, tail_ub_size)

            tail_copy_num = (tail_size // 8) * 8
            util_tik_comm_func.ub2gm(self.tik_instance, self.output_gm_list[0][offset],
                                     labels_ub_out_tail, tail_copy_num)

            # copy tail data to gm
            tail_ub_out = self.tik_instance.Tensor("int32", (8,),
                                                   name="tail_ub_out", scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, 8) as i:
                tail_ub_out[i].set_as(labels_ub_out_tail[tail_size - 8 + i])

            util_tik_comm_func.ub2gm(self.tik_instance, self.output_gm_list[0][self.labels_shape[0] - 8],
                                     tail_ub_out, 8)

    def labels_output_compute(self):
        """label_output_compute
        """
        with self.tik_instance.new_stmt_scope():
            loop_time = self.loop_time
            tail_size = self.tail_num
            tail_offset = loop_time * BATCH_SIZE
            with self.tik_instance.for_range(0, loop_time) as i:
                self.labels_output_per_batch(i)
            with self.tik_instance.if_scope(tail_size > 0):
                self.labels_output_tail(tail_offset, tail_size)

    def sub_sample_compute(self):
        """sub_sample_compute
        """
        self.label_mask_compute()
        self.update_pos_mask()
        self.update_neg_mask()
        self.labels_output_compute()


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


def total_num(shape):
    """return total_num"""
    shape_total_num = functools_reduce(lambda a, b: a * b, shape)
    return shape_total_num


def ceil_div(value, factor):
    """Compute the smallest integer value that is greater than
    or equal to value/factor
    """
    result = (value + (factor - 1)) // factor
    return result


@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_INT,
                            para_check.REQUIRED_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def sub_sample_labels(labels,
                      shuffle_matrix,
                      y,
                      batch_size_per_images,
                      positive_fraction,
                      kernel_name="sub_sample_labels"):
    """
    Randomly sample a subset of positive and negative examples,and overwrite
    the label vector to the ignore value (-1) for all elements that are not
    included in the sample
    Parameters:
    ----------
    labels : dict.
        shape of labels,(N, ) label vector with values:
        * -1: ignore
        * bg_label: background("negative") class
        * otherwise: one or more foreground ("positive") classes

    shuffle_matrix : dict.
        random matrix with shape (N, )

    batch_size_per_images : int.
        A require attribute of type int
    positive_fraction : float.
        A require attribute of type int
    kernel_name : str.
        cce kernel name, default value is "sub_sample_labels"
    Returns
    -------
    tik_instance
    """
    sub_sample = SubSampleLabels(labels,
                                 shuffle_matrix,
                                 batch_size_per_images,
                                 positive_fraction)
    # init gm mem
    sub_sample.init_tik_mem()
    # sub_sample compute
    sub_sample.sub_sample_compute()
    return sub_sample.build_tik_instance(kernel_name)
