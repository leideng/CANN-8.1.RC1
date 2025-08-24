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
mat_mul_softmax_dropout_matmul
"""

from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.vit_flash_attention_score import vit_flash_attention_score, check_supported_vit


# 'pylint: disable=too-many-arguments,too-many-locals
# 'pylint: disable=too-few-public-methods
# 'pylint: disable=too-many-statements, too-many-arguments, too-many-lines
# 'pylint: disable=too-many-public-methods
class Constant:
    """
    constant of swin_attention_score
    """
    DUP_INIT_VALUE = -60000


class MatMulSoftmax:
    """
    MatMulSoftmax class
    """

    # 'pylint: disable=unused-argument
    def __init__(self, x1, x2, mul_x, add_x1, add_x2, drop_mask, x3,
                 softmax_output, y, input_keep_prob, axis,
                 first_transpose_a, first_transpose_b,
                 second_transpose_a, second_transpose_b, kernel_name):
        self.tik_instance = tik.Tik()
        self.cur_op_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.matmul_dtype = "float16"
        self.vector_dtype = "float16"
        self.x1_shape = x1["shape"]
        self.x2_shape = x2["shape"]
        self.x3_shape = x3["shape"]
        self.x_shape_ori = x1["ori_shape"]
        self.vit_struc = True
        self.gpt_struc = False
        if add_x1 is not None:
            self.vit_struc = False
            self.ele_shape1 = add_x1["shape"]
            if self.ele_shape1[0] == self.ele_shape1[1] == 1:
                self.gpt_struc = True
        self.swin_struc = False
        if add_x2 is not None:
            self.swin_struc = True
            self.ele_shape2 = add_x2["shape"]
            self.drop_shape = drop_mask["shape"]
        self.y_shape = y["shape"]

        self.batch_per_core = (self.x1_shape[0] * self.x1_shape[1] + self.cur_op_core_num - 1) // self.cur_op_core_num
        self.batch_small_per_core = self.batch_per_core - 1
        self.batch_large_core_num = self.x1_shape[0] * self.x1_shape[1] - \
                                    self.batch_small_per_core * self.cur_op_core_num
        self.batch_outer_num = self.tik_instance.Scalar("int32", name="batch_outer_num")

        self.first_m_dim = self.x1_shape[3]
        self.first_k_dim = self.x1_shape[2]
        self.first_n_dim = self.x2_shape[3]
        self.second_m_dim = self.x1_shape[3]
        self.second_k_dim = self.x2_shape[3]
        self.second_n_dim = self.x3_shape[2]

        self.block_stride = 1
        self.double_factor = 2
        self.repeat_stride = 8
        self.trans_cube_target = 8
        self.block_num = 16
        self.repeat_once_size = 128

        self.input_keep_prob = input_keep_prob

        self.mul_x_shape = [self.block_num]
        self.kernel_name = kernel_name
        self.init_gm()

    def init_gm(self):
        self.x1_gm = self.tik_instance.Tensor(self.matmul_dtype, self.x1_shape, name="x1_gm",
                                              scope=tbe_platform.scope_gm)
        self.x2_gm = self.tik_instance.Tensor(self.matmul_dtype, self.x2_shape, name="x2_gm",
                                              scope=tbe_platform.scope_gm)
        self.mul_gm = self.tik_instance.Tensor(self.matmul_dtype, self.mul_x_shape,
                                               name="mul_gm", scope=tbe_platform.scope_gm)
        if not self.vit_struc:
            self.add1_gm = self.tik_instance.Tensor(self.matmul_dtype, self.ele_shape1,
                                                    name="add1_gm", scope=tbe_platform.scope_gm)
        if self.swin_struc:
            self.add2_gm = self.tik_instance.Tensor(self.matmul_dtype, self.ele_shape2,
                                                    name="add2_gm", scope=tbe_platform.scope_gm)
            self.drop_mask_gm = self.tik_instance.Tensor("uint8", self.drop_shape,
                                                         name="drop_mask_gm", scope=tbe_platform.scope_gm)
        self.x3_gm = self.tik_instance.Tensor(self.matmul_dtype, self.x3_shape, name="x3_gm",
                                              scope=tbe_platform.scope_gm)

        self.y_gm = self.tik_instance.Tensor(self.matmul_dtype, self.y_shape, name="y_gm", scope=tbe_platform.scope_gm)

    def mid_data_to_ub(self, tensor_c, tensor_c_ub, om_size, cur_m_idx, cur_m_size, m, single_m_size,
                       ub_mask, block_idx, cur_b_idx, cur_om_idx):
        tensor_a_src_offset = cur_m_idx * cur_m_size * self.block_num * self.block_num + \
                              m * single_m_size * self.block_num * self.block_num
        tensor_a_dst_offset = 0
        tensor_a_repeat_times = self.first_n_dim
        tesnor_a_data_size = single_m_size * self.block_num
        tensor_a_src_stride = (om_size - single_m_size) * self.block_num
        tensor_a_dst_stride = 0
        self.tik_instance.data_move(tensor_c_ub[tensor_a_dst_offset], tensor_c[tensor_a_src_offset], sid=0,
                                    nburst=tensor_a_repeat_times, burst=tesnor_a_data_size,
                                    src_stride=tensor_a_src_stride, dst_stride=tensor_a_dst_stride)

    def mat_mul_second_compute_front(self, second_bmm_compute_buffers, second_bmm_compute_db_buffers,
                                     second_bmm_compute_idxs, second_bmm_compute_each_layer_size,
                                     input_keep_prob):
        tensor_a, tensor_b, tensor_c, tensor_a_l0a, tensor_b_l0b, tensor_c_l0c, tensor_c_ub, ub_mask, ub_cast, \
        reduce_ub, trans_cube_tensor_b_l1, trans_cube_tensor_b_l0b, second_bmm_tensor_b = second_bmm_compute_buffers
        tensor_a_l1a_s_ub, tensor_b_s, tensor_a_l0a_ub, tensor_c_l0c_ub, tensor_c_ub2, ub_mask2, ub_cast2, \
        reduce_ub2, trans_cube_tensor_b_l1_db, trans_cube_tensor_b_l0b_db = second_bmm_compute_db_buffers
        block_idx, cur_b_idx, cur_om_idx, cur_m_idx = second_bmm_compute_idxs
        om_size, cur_m_size, cur_k_size, cur_n_size = second_bmm_compute_each_layer_size

        single_m_size, single_k_size, single_n_size = cur_m_size, cur_k_size, cur_n_size
        m_range = cur_m_size // single_m_size
        n_range = cur_n_size // single_n_size

        with self.tik_instance.for_range(0, m_range) as m:
            with self.tik_instance.for_range(0, n_range) as n:
                self.mid_data_to_ub(tensor_c, tensor_c_ub, om_size, cur_m_idx, cur_m_size, m, single_m_size, ub_mask,
                                    block_idx, cur_b_idx, cur_om_idx)
                softmax_last_part_buffers = [tensor_c_ub, ub_cast, reduce_ub, ub_mask,
                                             tensor_a, trans_cube_tensor_b_l1,
                                             tensor_a_l0a, trans_cube_tensor_b_l0b, tensor_c_l0c]
                softmax_last_part_size = [om_size, cur_m_size, cur_k_size, single_m_size, single_k_size]
                softmax_last_part_idx = [input_keep_prob, None, block_idx, cur_b_idx, cur_om_idx, cur_m_idx, m]
                tensor_c_ub_back = self.softmax_compute_last_part(softmax_last_part_buffers, softmax_last_part_size,
                                                                  softmax_last_part_idx)
                self.tik_instance.data_move(tensor_a[0], tensor_c_ub_back[0],
                                            sid=0, nburst=1, burst=single_k_size * single_m_size * self.block_num,
                                            src_stride=0, dst_stride=0)

    def mat_mul_second_compute_mid(self, second_bmm_compute_buffers, second_bmm_compute_db_buffers,
                                   second_bmm_compute_idxs, second_bmm_compute_each_layer_size, input_keep_prob):
        tensor_a, tensor_b, tensor_c, tensor_a_l0a, tensor_b_l0b, tensor_c_l0c, tensor_c_ub, ub_mask, ub_cast, \
        reduce_ub, trans_cube_tensor_b_l1, trans_cube_tensor_b_l0b, second_bmm_tensor_b = second_bmm_compute_buffers
        tensor_a_l1a_s_ub, tensor_b_s, tensor_a_l0a_ub, tensor_c_l0c_ub, tensor_c_ub2, ub_mask2, ub_cast2, \
        reduce_ub2, trans_cube_tensor_b_l1_db, trans_cube_tensor_b_l0b_db = second_bmm_compute_db_buffers
        block_idx, cur_b_idx, cur_om_idx, cur_m_idx = second_bmm_compute_idxs
        om_size, cur_m_size, cur_k_size, cur_n_size = second_bmm_compute_each_layer_size

        single_m_size, single_k_size, single_n_size = cur_m_size, cur_k_size, cur_n_size
        m_range = cur_m_size // single_m_size
        n_range = cur_n_size // single_n_size

        with self.tik_instance.for_range(0, m_range) as m:
            with self.tik_instance.for_range(0, n_range) as n:
                self.mid_data_to_ub(tensor_c, tensor_c_ub2, om_size, cur_m_idx + 1, cur_m_size, m, single_m_size,
                                    ub_mask2, block_idx, cur_b_idx, cur_om_idx)
                l1a_offset = 0
                l1b_offset = 0
                l1a_repeat_times = single_m_size * cur_k_size
                l1b_repeat_times = single_n_size * cur_k_size

                self.tik_instance.load2dv2(tensor_a_l0a, tensor_a[l1a_offset], 0, l1a_repeat_times, 0, 1, 0, False)

                if self.first_m_dim == self.trans_cube_target:
                    self.tik_instance.load2dv2(tensor_b_l0b, second_bmm_tensor_b[0],
                                               0, self.first_n_dim * self.second_n_dim, 0, 1, 0, True)
                self.tik_instance.mmad(tensor_c_l0c, tensor_a_l0a, tensor_b_l0b,
                                       single_m_size * self.block_num,
                                       single_k_size * self.block_num,
                                       single_n_size * self.block_num,
                                       0)

                softmax_last_part_buffers = [tensor_c_ub2, ub_cast2, reduce_ub2, ub_mask2, tensor_a_l1a_s_ub,
                                             trans_cube_tensor_b_l1_db, tensor_a_l0a_ub, trans_cube_tensor_b_l0b_db,
                                             tensor_c_l0c_ub]
                softmax_last_part_size = [om_size, cur_m_size, cur_k_size, single_m_size, single_k_size]
                softmax_last_part_idx = [input_keep_prob, None, block_idx, cur_b_idx, cur_om_idx, cur_m_idx + 1, m]
                tensor_c_ub_back_2 = self.softmax_compute_last_part(softmax_last_part_buffers, softmax_last_part_size,
                                                                    softmax_last_part_idx)
                self.tik_instance.data_move(tensor_a_l1a_s_ub[0], tensor_c_ub_back_2[0], sid=0,
                                            nburst=1, burst=single_k_size * single_m_size * self.block_num,
                                            src_stride=0, dst_stride=0)

                # do the last time
                cc_to_ub_dst_stride = 0
                cc_to_ub_src_stride = 0
                ub_mask16 = ub_mask.reinterpret_cast_to("float16")
                self.tik_instance.tensor_mov(ub_mask16, tensor_c_l0c, 'm', 1, single_m_size * single_n_size,
                                             cc_to_ub_dst_stride, cc_to_ub_src_stride)
                single_data_size = single_m_size * self.block_num
                repeat_times = single_n_size
                output_dst_stride = (self.second_m_dim - single_m_size) * self.block_num
                output_dst_offset = self.batch_outer_num * self.second_n_dim * \
                                    self.second_m_dim * self.block_num * self.block_num + \
                                    cur_om_idx * om_size * self.block_num * self.block_num + \
                                    cur_m_idx * cur_m_size * self.block_num * self.block_num + \
                                    m * single_m_size * self.block_num * self.block_num

                self.tik_instance.data_move(self.y_gm[output_dst_offset], ub_mask16[0],
                                            sid=0, nburst=repeat_times, burst=single_data_size,
                                            src_stride=0, dst_stride=output_dst_stride)

    def mat_mul_second_compute_last(self, second_bmm_compute_buffers, second_bmm_compute_db_buffers,
                                    second_bmm_compute_idxs, second_bmm_compute_each_layer_size, input_keep_prob):
        tensor_a, tensor_b, tensor_c, tensor_a_l0a, tensor_b_l0b, tensor_c_l0c, tensor_c_ub, ub_mask, ub_cast, \
        reduce_ub, trans_cube_tensor_b_l1, trans_cube_tensor_b_l0b, second_bmm_tensor_b = second_bmm_compute_buffers
        tensor_a_l1a_s_ub, tensor_b_s, tensor_a_l0a_ub, tensor_c_l0c_ub, tensor_c_ub2, ub_mask2, ub_cast2, \
        reduce_ub2, trans_cube_tensor_b_l1_db, trans_cube_tensor_b_l0b_db = second_bmm_compute_db_buffers
        block_idx, cur_b_idx, cur_om_idx, cur_m_idx = second_bmm_compute_idxs
        om_size, cur_m_size, cur_k_size, cur_n_size = second_bmm_compute_each_layer_size

        single_m_size, single_k_size, single_n_size = cur_m_size, cur_k_size, cur_n_size
        m_range = cur_m_size // single_m_size
        n_range = cur_n_size // single_n_size

        with self.tik_instance.for_range(0, m_range) as m:
            with self.tik_instance.for_range(0, n_range) as n:
                l1a_offset = 0
                l1b_offset = 0
                l1a_repeat_times = single_m_size * cur_k_size
                l1b_repeat_times = single_n_size * cur_k_size
                self.tik_instance.load2dv2(tensor_a_l0a_ub, tensor_a_l1a_s_ub[l1a_offset],
                                           0, l1a_repeat_times, 0, 1, 0, False)
                if self.first_m_dim == self.trans_cube_target:
                    self.tik_instance.load2dv2(tensor_b_l0b, second_bmm_tensor_b[0],
                                               0, self.first_n_dim * self.second_n_dim, 0, 1, 0, True)
                self.tik_instance.mmad(tensor_c_l0c_ub, tensor_a_l0a_ub, tensor_b_l0b,
                                       single_m_size * self.block_num,
                                       single_k_size * self.block_num,
                                       single_n_size * self.block_num,
                                       0)

                cc_to_ub_dst_stride = 0
                cc_to_ub_src_stride = 0
                self.tik_instance.tensor_mov(tensor_c_ub2, tensor_c_l0c_ub, 'm', 1, single_m_size * single_n_size,
                                             cc_to_ub_dst_stride, cc_to_ub_src_stride)

                if self.x_shape_ori[2] % 16 != 0:
                    dup_tail = self.x_shape_ori[2] % 16
                    dup_repeat = 1
                    max_dup_tail = 8
                    if dup_tail >= max_dup_tail:
                        dup_num = (self.block_num - dup_tail) * self.block_num
                        for n_idx in range(single_n_size):
                            dup_offset = (n_idx * single_m_size + single_m_size - 1) * self.block_num * \
                                         self.block_num + dup_tail * self.block_num
                            self.tik_instance.vector_dup(dup_num, tensor_c_ub2[dup_offset],
                                                         self.tik_instance.Scalar(init_value=0, dtype="float16"),
                                                         dup_repeat, self.block_stride, self.repeat_stride)
                    else:
                        for n_idx in range(single_n_size):
                            dup_offset = (n_idx * single_m_size + single_m_size - 1) * self.block_num * \
                                         self.block_num + dup_tail * self.block_num
                            self.tik_instance.vector_dup(self.repeat_once_size, tensor_c_ub2[dup_offset],
                                                         self.tik_instance.Scalar(init_value=0, dtype="float16"),
                                                         1, self.block_stride, self.repeat_stride)
                            dup_offset = (n_idx * single_m_size + single_m_size - 1) * self.block_num * \
                                         self.block_num + dup_tail * self.block_num + self.repeat_once_size
                            dup_num = (self.block_num - dup_tail) * self.block_num - self.repeat_once_size
                            self.tik_instance.vector_dup(dup_num, tensor_c_ub2[dup_offset],
                                                         self.tik_instance.Scalar(init_value=0, dtype="float16"),
                                                         dup_repeat, self.block_stride, self.repeat_stride)
                single_data_size = single_m_size * self.block_num
                repeat_times = single_n_size
                output_dst_stride = (self.second_m_dim - single_m_size) * self.block_num
                output_dst_offset = self.batch_outer_num * self.second_n_dim * \
                                    self.second_m_dim * self.block_num * self.block_num + \
                                    cur_om_idx * om_size * self.block_num * self.block_num + \
                                    cur_m_idx * cur_m_size * self.block_num * self.block_num + \
                                    m * single_m_size * self.block_num * self.block_num

                self.tik_instance.data_move(self.y_gm[output_dst_offset], tensor_c_ub2[0],
                                            sid=0, nburst=repeat_times, burst=single_data_size,
                                            src_stride=0, dst_stride=output_dst_stride)

    def mat_mul_compute(self, first_bmm_compute_buffers, first_bmm_compute_idx, first_bmm_compute_each_layer_size):
        """
        first bmm compute.
        """
        tensor_a, tensor_b, tensor_y, tensor_a_l0a, tensor_b_l0b, tensor_c_l0c, tensor_c_ub, \
        mul_value, elewise_data_ub2 = first_bmm_compute_buffers
        block_idx, cur_b_idx, cur_om_idx, cur_m_idx = first_bmm_compute_idx
        om_size, cur_m_size, cur_k_size, cur_n_size = first_bmm_compute_each_layer_size
        single_m_size, single_k_size, single_n_size = cur_m_size, cur_k_size, cur_n_size
        m_range = cur_m_size // single_m_size
        n_range = cur_n_size // single_n_size

        l0a_shape = [single_k_size, single_m_size, self.block_num, self.block_num]
        l0b_shape = [single_k_size, single_n_size, self.block_num, self.block_num]
        l0c_shape = [single_n_size, single_m_size, self.block_num, self.block_num]

        with self.tik_instance.for_range(0, m_range) as m:
            with self.tik_instance.for_range(0, n_range) as n:
                for ck in range(cur_k_size):
                    tensor_a_src_offset = self.batch_outer_num * self.first_m_dim * \
                                          self.first_k_dim * self.block_num * self.block_num + \
                                          cur_om_idx * om_size * self.block_num * self.block_num + \
                                          cur_m_idx * cur_m_size * self.block_num * self.block_num + \
                                          m * single_m_size * self.block_num * self.block_num + \
                                          ck * self.first_m_dim * self.block_num * self.block_num
                    tensor_a_dst_offset = ck * self.block_num * self.block_num
                    tesnor_a_data_size = self.block_num
                    tensor_a_repeat_times = single_m_size
                    tensor_a_src_stride = 0
                    tensor_a_dst_stride = (cur_k_size - 1) * self.block_num
                    self.tik_instance.data_move(tensor_a[tensor_a_dst_offset], self.x1_gm[tensor_a_src_offset], sid=0,
                                                nburst=tensor_a_repeat_times, burst=tesnor_a_data_size,
                                                src_stride=tensor_a_src_stride, dst_stride=tensor_a_dst_stride)

                l1a_offset = 0
                l1b_offset = 0
                l1a_repeat_times = single_m_size * cur_k_size
                l1b_repeat_times = single_n_size * cur_k_size
                self.tik_instance.load2dv2(tensor_a_l0a, tensor_a[l1a_offset], 0, l1a_repeat_times, 0, 1, 0, False)

                self.tik_instance.mmad(tensor_c_l0c, tensor_a_l0a, tensor_b_l0b,
                                       single_m_size * self.block_num,
                                       single_k_size * self.block_num,
                                       single_n_size * self.block_num,
                                       0)

                cc_to_ub_dst_stride = 0
                cc_to_ub_src_stride = 0

                self.tik_instance.tensor_mov(tensor_c_ub, tensor_c_l0c, 'm', 1, single_m_size * single_n_size,
                                             cc_to_ub_dst_stride, cc_to_ub_src_stride)

                ele_compute_repeat_times = single_m_size * single_n_size * 16 * 16 // self.repeat_once_size
                tail = max(ele_compute_repeat_times // 255, 0) * (ele_compute_repeat_times % 255)
                ele_compute_repeat_times = min(255, ele_compute_repeat_times)
                self.tik_instance.vmuls(self.repeat_once_size, tensor_c_ub, tensor_c_ub,
                                        mul_value, ele_compute_repeat_times,
                                        self.block_stride, self.block_stride,
                                        self.repeat_stride, self.repeat_stride)
                if tail != 0:
                    self.tik_instance.vmuls(self.repeat_once_size, tensor_c_ub[255 * self.repeat_once_size],
                                            tensor_c_ub[255 * self.repeat_once_size], mul_value, tail,
                                            self.block_stride, self.block_stride,
                                            self.repeat_stride, self.repeat_stride)

                if not self.vit_struc:
                    first_mov1 = self.batch_outer_num
                    if self.x1_shape[0] != self.ele_shape1[0]:
                        first_mov1 = self.batch_outer_num % self.x1_shape[1]
                    if self.gpt_struc:
                        first_mov1 = 0
                    ele_move_offset1 = first_mov1 * self.first_m_dim * self.first_n_dim * self.block_num * \
                                       self.block_num + cur_om_idx * om_size * self.block_num * self.block_num + \
                                       cur_m_idx * cur_m_size * self.block_num * self.block_num + \
                                       m * single_m_size * self.block_num * self.block_num + \
                                       n * self.first_m_dim * self.block_num * self.block_num
                    ele_move_repeat_times = single_n_size
                    ele_move_data_size = single_m_size * self.block_num
                    ele_move_src_stride = (self.first_m_dim - single_m_size) * self.block_num
                    ele_move_dst_stride = 0
                    self.tik_instance.data_move(elewise_data_ub2, self.add1_gm[ele_move_offset1],
                                                sid=0, nburst=ele_move_repeat_times, burst=ele_move_data_size,
                                                src_stride=ele_move_src_stride, dst_stride=ele_move_dst_stride)

                    self.tik_instance.vadd(self.repeat_once_size, tensor_c_ub, tensor_c_ub,
                                           elewise_data_ub2, ele_compute_repeat_times,
                                           self.block_stride, self.block_stride, self.block_stride,
                                           self.repeat_stride, self.repeat_stride, self.repeat_stride)

                    if tail != 0:
                        self.tik_instance.vadd(self.repeat_once_size, tensor_c_ub[255 * self.repeat_once_size],
                                               tensor_c_ub[255 * self.repeat_once_size],
                                               elewise_data_ub2[255 * self.repeat_once_size], tail,
                                               self.block_stride, self.block_stride, self.block_stride,
                                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

                if self.swin_struc:
                    first_mov2 = (self.batch_outer_num % (self.ele_shape2[1] * self.x1_shape[1])) // self.x1_shape[1]
                    ele_move_offset2 = first_mov2 * self.first_m_dim * self.first_n_dim * self.block_num * \
                                       self.block_num + cur_om_idx * om_size * self.block_num * self.block_num + \
                                       cur_m_idx * cur_m_size * self.block_num * self.block_num + \
                                       m * single_m_size * self.block_num * self.block_num + \
                                       n * self.first_m_dim * self.block_num * self.block_num
                    self.tik_instance.data_move(elewise_data_ub2, self.add2_gm[ele_move_offset2],
                                                sid=0, nburst=ele_move_repeat_times, burst=ele_move_data_size,
                                                src_stride=ele_move_src_stride, dst_stride=ele_move_dst_stride)

                    self.tik_instance.vadd(self.repeat_once_size, tensor_c_ub, tensor_c_ub,
                                           elewise_data_ub2, ele_compute_repeat_times,
                                           self.block_stride, self.block_stride, self.block_stride,
                                           self.repeat_stride, self.repeat_stride, self.repeat_stride)

                    if tail != 0:
                        self.tik_instance.vadd(self.repeat_once_size, tensor_c_ub[255 * self.repeat_once_size],
                                               tensor_c_ub[255 * self.repeat_once_size],
                                               elewise_data_ub2[255 * self.repeat_once_size], tail,
                                               self.block_stride, self.block_stride, self.block_stride,
                                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

                self.softmax_compute(tensor_c_ub, single_m_size, single_n_size)

                # copy_ub_to_l1
                mid_offset = cur_m_idx * cur_m_size * self.block_num * self.block_num + \
                             m * single_m_size * self.block_num * self.block_num
                mid_data_lengh = single_m_size * self.block_num
                mid_src_stride = 0
                mid_dst_stride = (om_size - single_m_size) * self.block_num
                self.tik_instance.data_move(tensor_y[mid_offset], tensor_c_ub, sid=0,
                                            nburst=single_n_size, burst=mid_data_lengh,
                                            src_stride=mid_src_stride, dst_stride=mid_dst_stride)

    def vadds_transfor_cube(self, compute_buffers, compute_sizes):
        tensor_a_ub, tensor_b_ub, tensor_a_l1, tensor_b_l1, \
        tensor_a_l0a, tensor_b_l0b, tensor_c_l0c, output_ub = compute_buffers
        single_m_size, single_k_size, single_n_size = compute_sizes
        vec_repeat_times = single_k_size * self.block_num * self.block_num // self.repeat_once_size
        self.tik_instance.vector_dup(self.repeat_once_size, tensor_b_ub[0], 1, vec_repeat_times, 1, 8)
        tensor_a_burst = self.block_num
        tensor_a_nburst = single_k_size
        tensor_a_src_stride = (single_m_size - 1) * self.block_num
        tensor_a_dst_stride = 0
        for i in range(single_m_size):
            src_offset = i * self.block_num * self.block_num
            dst_offset = i * single_k_size * self.block_num * self.block_num
            self.tik_instance.data_move(tensor_a_l1[dst_offset], tensor_a_ub[src_offset],
                                        sid=0, nburst=tensor_a_nburst, burst=tensor_a_burst,
                                        src_stride=tensor_a_src_stride, dst_stride=tensor_a_dst_stride)
        if single_m_size != 1:
            self.tik_instance.data_move(tensor_a_l1[single_k_size * self.block_num * self.block_num],
                                        tensor_a_ub[self.block_num * self.block_num],
                                        sid=0, nburst=tensor_a_nburst, burst=tensor_a_burst,
                                        src_stride=tensor_a_src_stride, dst_stride=tensor_a_dst_stride)
        tensor_b_burst = single_n_size * single_k_size * self.block_num
        tensor_b_nburst = 1
        tensor_b_src_stride = 0
        tensor_b_dst_stride = 0
        self.tik_instance.data_move(tensor_b_l1, tensor_b_ub[0], sid=0,
                                    nburst=tensor_b_nburst, burst=tensor_b_burst,
                                    src_stride=tensor_b_src_stride, dst_stride=tensor_b_dst_stride)
        self.tik_instance.load2dv2(tensor_a_l0a, tensor_a_l1[0], 0, single_m_size * single_k_size, 0, 1, 0, False)
        self.tik_instance.load2dv2(tensor_b_l0b, tensor_b_l1[0], 0, single_n_size * single_k_size, 0, 1, 0, False)
        self.tik_instance.mmad(tensor_c_l0c, tensor_a_l0a, tensor_b_l0b,
                               single_m_size * self.block_num,
                               single_k_size * self.block_num,
                               single_n_size * self.block_num,
                               0)
        self.tik_instance.tensor_mov(tensor_b_ub, tensor_c_l0c, 'm', 1, single_m_size * single_n_size, 0, 0)

    def softmax_compute_last_part(self, softmax_last_part_buffers, softmax_last_part_size, softmax_last_part_idx):
        tensor_c_ub, ub_cast, reduce_ub, ub_mask, \
        tensor_a, tensor_b, tensor_a_l0a, tensor_b_l0b, tensor_c_l0c = softmax_last_part_buffers
        om_size, cur_m_size, cur_n_size, cur_m, cur_n = softmax_last_part_size
        input_keep_prob, ub_mask_fp16, block_idx, cur_b_idx, cur_om_idx, cur_m_idx, m = softmax_last_part_idx
        fp32_repeat_once_nums = 64
        max_repeat_times = 255
        repeat_times = cur_m * cur_n * self.block_num * self.block_num // fp32_repeat_once_nums
        insn_tail = max(repeat_times // 255, 0) * (repeat_times % 255)
        repeat_times = min(repeat_times, max_repeat_times)

        self.tik_instance.vconv(fp32_repeat_once_nums, "", ub_cast[0], tensor_c_ub[0], repeat_times, 1, 1, 8, 4)
        self.tik_instance.vexp(fp32_repeat_once_nums, ub_cast[0], ub_cast[0], repeat_times, 1, 1, 8, 8)
        self.tik_instance.vconv(fp32_repeat_once_nums, "", tensor_c_ub[0], ub_cast[0], repeat_times, 1, 1, 4, 8)
        if insn_tail > 0:
            self.tik_instance.vconv(fp32_repeat_once_nums, "", ub_cast[repeat_times * fp32_repeat_once_nums],
                                    tensor_c_ub[repeat_times * fp32_repeat_once_nums], insn_tail, 1, 1, 8, 4)
            self.tik_instance.vexp(fp32_repeat_once_nums, ub_cast[repeat_times * fp32_repeat_once_nums],
                                   ub_cast[repeat_times * fp32_repeat_once_nums], insn_tail, 1, 1, 8, 8)
            self.tik_instance.vconv(fp32_repeat_once_nums, "", tensor_c_ub[repeat_times * fp32_repeat_once_nums],
                                    ub_cast[repeat_times * fp32_repeat_once_nums], insn_tail, 1, 1, 4, 8)
        vmax_range = cur_n
        src_tensor = ub_cast
        ub_broadcast_fp16 = ub_cast.reinterpret_cast_to("float16")
        if self.first_m_dim != self.trans_cube_target:
            while (vmax_range > 1):
                if vmax_range % 2 == 0:
                    repeat_time = cur_m * vmax_range * self.block_num * self.block_num // fp32_repeat_once_nums // 2
                    src_offset = cur_m * vmax_range * self.block_num * self.block_num // 2
                    self.tik_instance.vadd(fp32_repeat_once_nums, src_tensor[0], src_tensor[0], src_tensor[src_offset],
                                           repeat_time, self.block_stride, self.block_stride, self.block_stride,
                                           self.repeat_stride, self.repeat_stride, self.repeat_stride)
                    vmax_range = vmax_range // 2
                else:
                    repeat_time = cur_m * self.block_num * self.block_num // fp32_repeat_once_nums
                    src_offset = (vmax_range - 1) * cur_m * self.block_num * self.block_num
                    self.tik_instance.vadd(fp32_repeat_once_nums, src_tensor[0], src_tensor[0], src_tensor[src_offset],
                                           repeat_time, self.block_stride, self.block_stride, self.block_stride,
                                           self.repeat_stride, self.repeat_stride, self.repeat_stride)
                    vmax_range = vmax_range - 1

            repeat_time = cur_m * self.block_num
            self.tik_instance.vcadd(self.block_num, reduce_ub[0], src_tensor[0],
                                    repeat_time, self.block_stride, self.block_stride, 2)
            vrec_mask = repeat_time
            self.tik_instance.vrec(vrec_mask, reduce_ub[0], reduce_ub[0], 1, 1, 1, 0, 0)
            ub_reduceadd_fp16 = reduce_ub.reinterpret_cast_to("float16")
            self.tik_instance.vconv(vrec_mask, "", ub_reduceadd_fp16[0], reduce_ub[0], 1, 1, 1, 0, 0)
            # broadcast
            ub_broadcast = ub_cast.reinterpret_cast_to("uint16")
            self.tik_instance.vector_dup(self.repeat_once_size,
                                         ub_broadcast[cur_m * cur_n * self.block_num * self.block_num],
                                         self.tik_instance.Scalar(init_value=0, dtype="uint16"), 1, 1, 8)

            ub_reduceadd_int16 = reduce_ub.reinterpret_cast_to("uint16")

            for cur_fz in range(cur_m):
                dst_offset = cur_fz * self.block_num * self.block_num
                src_offset = cur_fz * self.block_num
                self.tik_instance.vor(self.block_num, ub_broadcast[dst_offset], ub_reduceadd_int16[src_offset],
                                      ub_broadcast[cur_m * cur_n * self.block_num * self.block_num],
                                      self.block_num, 1, 1, 0, 1, 0, 0)
                self.tik_instance.vtranspose(ub_broadcast[dst_offset], ub_broadcast[dst_offset])
        else:
            trans_cube_buffers = [tensor_c_ub, ub_broadcast_fp16, tensor_a, tensor_b,
                                  tensor_a_l0a, tensor_b_l0b, tensor_c_l0c, reduce_ub]
            trans_cube_size = [cur_m, cur_n, 1]
            self.vadds_transfor_cube(trans_cube_buffers, trans_cube_size)
            rec_repeat_times = cur_m * self.block_num * self.block_num // self.repeat_once_size
            self.tik_instance.vrec(self.repeat_once_size, ub_broadcast_fp16[0],
                                   ub_broadcast_fp16[0], rec_repeat_times, 1, 1, 8, 8)

        sub_range = cur_m * self.block_num * self.block_num // self.repeat_once_size
        with self.tik_instance.for_range(0, sub_range) as idx:
            self.tik_instance.vmul(self.repeat_once_size, tensor_c_ub[idx * self.repeat_once_size],
                                   tensor_c_ub[idx * self.repeat_once_size],
                                   ub_broadcast_fp16[idx * self.repeat_once_size],
                                   cur_n, 1, 1, 1, cur_m * self.block_num, cur_m * self.block_num, 0)
        trans_nz_zz_dst_repeat_stride = self.block_num
        trans_nz_zz_src_repeat_stride = cur_m * self.block_num
        for i in range(cur_m):
            self.tik_instance.vadds(self.repeat_once_size,
                                    ub_broadcast_fp16[i * cur_n * self.block_num * self.block_num],
                                    tensor_c_ub[i * self.block_num * self.block_num], 0, cur_n, 1, 1,
                                    trans_nz_zz_dst_repeat_stride, trans_nz_zz_src_repeat_stride, 0)
            self.tik_instance.vadds(self.repeat_once_size,
                                    ub_broadcast_fp16[i * cur_n * self.block_num * self.block_num +
                                                      self.repeat_once_size],
                                    tensor_c_ub[i * self.block_num * self.block_num + self.repeat_once_size],
                                    0, cur_n, 1, 1, trans_nz_zz_dst_repeat_stride, trans_nz_zz_src_repeat_stride, 0)

        return ub_broadcast_fp16

    def softmax_compute(self, tensor_input, cur_m, cur_n):
        softmax_ub = self.tik_instance.Tensor(self.matmul_dtype, [cur_n, cur_m, self.block_num, self.block_num],
                                              name="softmax_ub", scope=tbe_platform.scope_ubuf)
        reduce_ub = self.tik_instance.Tensor(self.matmul_dtype, [cur_m * self.block_num],
                                             name="reduce_ub", scope=tbe_platform.scope_ubuf)
        ub_broadcast = self.tik_instance.Tensor("uint16", [self.first_n_dim * self.block_num],
                                                name="ub_broadcast", scope=tbe_platform.scope_ubuf)
        if self.x_shape_ori[2] % 16 != 0:
            dup_tail = self.x_shape_ori[2] % 16
            dup_value = -1
            for j in range(0, dup_tail):
                # compute the mask for vector_dup
                dup_value -= 2 ** (0 + j) + 2 ** (16 + j) + 2 ** (32 + j) + 2 ** (48 + j)
            dup_mask_h = self.tik_instance.Scalar(init_value=dup_value, dtype="int64")
            dup_mask_l = self.tik_instance.Scalar(init_value=dup_value, dtype="int64")
            dup_offset = cur_m * (cur_n - 1) * self.block_num * self.block_num
            dup_repeat = cur_m * self.block_num * self.block_num // self.repeat_once_size
            self.tik_instance.vector_dup([dup_mask_h, dup_mask_l], tensor_input[dup_offset],
                                         self.tik_instance.Scalar(init_value=Constant.DUP_INIT_VALUE, dtype="float16"),
                                         dup_repeat, self.block_stride, self.repeat_stride)

        # dichotomy compare to get line max value
        vmax_range = cur_n  # col num
        burst = cur_n * cur_m * self.block_num
        self.tik_instance.data_move(softmax_ub, tensor_input, 0, 1, burst, 0, 0)
        while vmax_range > 1:
            if vmax_range % 2 == 0:
                vmax_range = vmax_range // 2
                src_offset = vmax_range * cur_m * self.block_num * self.block_num
                repeat_time = src_offset // self.repeat_once_size
                self.tik_instance.vmax(self.repeat_once_size, softmax_ub[0], softmax_ub[0], softmax_ub[src_offset],
                                       repeat_time, self.block_stride, self.block_stride, self.block_stride,
                                       self.repeat_stride, self.repeat_stride, self.repeat_stride)
            else:  # keep the first part, dichotomy reduce compare the last pair part
                compare_range = vmax_range // 2  # compute col num for this loop
                vmax_range = compare_range + 1  # remain col num for next loop
                start = cur_m * self.block_num * self.block_num
                src_offset = compare_range * cur_m * self.block_num * self.block_num
                repeat_time = src_offset // self.repeat_once_size
                self.tik_instance.vmax(self.repeat_once_size, softmax_ub[start], softmax_ub[start],
                                       softmax_ub[start + src_offset], repeat_time,
                                       self.block_stride, self.block_stride, self.block_stride,
                                       self.repeat_stride, self.repeat_stride, self.repeat_stride)

        repeat_time = cur_m * self.block_num * self.block_num // self.repeat_once_size
        self.tik_instance.vcgmax(self.repeat_once_size, reduce_ub[0], softmax_ub[0], repeat_time,
                                 self.block_stride, self.block_stride, self.repeat_stride)

        ub_dup = softmax_ub.reinterpret_cast_to("uint16")
        self.tik_instance.vector_dup(self.repeat_once_size, ub_dup[cur_m * self.block_num * self.block_num],
                                     self.tik_instance.Scalar(init_value=0, dtype="uint16"),
                                     self.block_stride, self.block_stride, self.repeat_stride)
        ub_reducemax_int16 = reduce_ub.reinterpret_cast_to("uint16")
        for cur_fz in range(cur_m):
            src_offset = cur_fz * self.block_num
            dst_offset = cur_fz * self.block_num * self.block_num
            self.tik_instance.vor(self.block_num, ub_dup[dst_offset], ub_reducemax_int16[src_offset],
                                  ub_dup[cur_m * self.block_num * self.block_num],
                                  self.block_num, 1, 1, 0, 1, 0, 0)
            self.tik_instance.vtranspose(ub_dup[dst_offset], ub_dup[dst_offset])

        sub_range = cur_m * self.block_num * self.block_num // self.repeat_once_size
        sub_dst_stride = cur_m * self.block_num
        sub_src_stride = cur_m * self.block_num
        with self.tik_instance.for_range(0, sub_range) as idx:
            self.tik_instance.vsub(self.repeat_once_size, tensor_input[idx * self.repeat_once_size],
                                   tensor_input[idx * self.repeat_once_size], softmax_ub[idx * self.repeat_once_size],
                                   cur_n, 1, 1, 1, sub_dst_stride, sub_src_stride, 0)

        return tensor_input

    def tiling_batch_m_axis(self, batch, m_size):
        outer_m_range_value = 1
        batch_range_value = self.x1_shape[0] * self.x1_shape[1] // self.cur_op_core_num
        if m_size % 4 != 0:
            inner_m_range_value = self.x1_shape[3]
        elif m_size > 16:
            inner_m_range_value = self.x1_shape[3] // 2
        elif m_size >= 12:
            inner_m_range_value = 4
        else:
            inner_m_range_value = 2
        return batch_range_value, outer_m_range_value, inner_m_range_value

    def first_bmm_move_tensor_b_from_gm_to_l1(self, first_bmm_tensor_b):
        first_bmm_tensor_b_offset = self.batch_outer_num * \
                                    self.first_n_dim * self.first_k_dim * self.block_num * self.block_num
        first_bmm_tensor_b_burst = self.first_n_dim * self.block_num
        first_bmm_tensor_b_repeat_times = self.first_k_dim
        first_bmm_tensor_b_src_stride = 0
        first_bmm_tensor_b_dst_stride = 0
        self.tik_instance.data_move(first_bmm_tensor_b, self.x2_gm[first_bmm_tensor_b_offset], sid=0,
                                    nburst=first_bmm_tensor_b_repeat_times, burst=first_bmm_tensor_b_burst,
                                    src_stride=first_bmm_tensor_b_src_stride,
                                    dst_stride=first_bmm_tensor_b_dst_stride)
        return first_bmm_tensor_b

    def second_bmm_move_tensor_b_from_gm_to_l1(self, second_bmm_tensor_b):
        for dma_idx in range(self.second_n_dim):
            second_bmm_tensor_b_src_offset = self.batch_outer_num * self.second_n_dim * self.second_k_dim * \
                                             self.block_num * self.block_num + \
                                             dma_idx * self.second_k_dim * self.block_num * self.block_num
            second_bmm_tensor_b_dst_offset = dma_idx * self.block_num * self.block_num
            second_bmm_tensor_b_burst = self.block_num
            second_bmm_tensor_b_repeat_times = self.second_k_dim
            second_bmm_tensor_b_src_stride = 0
            second_bmm_tensor_b_dst_stride = (self.second_n_dim - 1) * self.block_num
            self.tik_instance.data_move(second_bmm_tensor_b[second_bmm_tensor_b_dst_offset],
                                        self.x3_gm[second_bmm_tensor_b_src_offset], sid=0,
                                        nburst=second_bmm_tensor_b_repeat_times, burst=second_bmm_tensor_b_burst,
                                        src_stride=second_bmm_tensor_b_src_stride,
                                        dst_stride=second_bmm_tensor_b_dst_stride)
        return second_bmm_tensor_b

    def apply_buffer_for_tensor_c_l1(self, outer_m_range_value):
        first_bmm_tensor_c = self.tik_instance.Tensor(self.vector_dtype,
                                                      [self.first_n_dim, self.first_m_dim // outer_m_range_value,
                                                       self.block_num, self.block_num],
                                                      name="first_bmm_tensor_c", scope=tbe_platform.scope_cbuf)

        return first_bmm_tensor_c

    def apply_buffer_for_tensor_b_l0_and_move_data_in(self, first_bmm_tensor_b):
        first_bmm_tensor_b_l0b = self.tik_instance.Tensor(self.matmul_dtype,
                                                          [self.first_k_dim, self.first_n_dim,
                                                           self.block_num, self.block_num],
                                                          name="first_bmm_tensor_b_l0b",
                                                          scope=tbe_platform.scope_cb)
        self.tik_instance.load2dv2(first_bmm_tensor_b_l0b, first_bmm_tensor_b[0],
                                   0, self.first_k_dim * self.first_n_dim, 0, 1, 0, False)
        return first_bmm_tensor_b_l0b

    def first_bmm_compute_for_outer_m_once(self, preload_buffers, range_idxs, outer_m_range_once_m_size,
                                           inner_m_range_value, mul_value):
        """
        compute first bmm once for outer m range.
        """
        first_bmm_tensor_b, first_bmm_tensor_c, first_bmm_tensor_b_l0b = preload_buffers
        block_idx, cur_b_idx, cur_om_idx = range_idxs
        with self.tik_instance.for_range(0, inner_m_range_value // self.double_factor) as inner_m_idx:
            inner_m_size, inner_k_size, inner_n_size = outer_m_range_once_m_size // inner_m_range_value, \
                                                       self.first_k_dim, self.first_n_dim
            l1a_shape = [inner_k_size, inner_m_size, self.block_num, self.block_num]
            first_bmm_tensor_a = self.tik_instance.Tensor(self.matmul_dtype, l1a_shape, name="first_bmm_tensor_a",
                                                          scope=tbe_platform.scope_cbuf)
            first_bmm_tensor_a_l0a = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * inner_k_size *
                                                                                  self.block_num * self.block_num],
                                                              name="first_bmm_tensor_a_l0a",
                                                              scope=tbe_platform.scope_ca)
            first_bmm_tensor_c_l0c = self.tik_instance.Tensor("float32", [inner_m_size * inner_n_size *
                                                                          self.block_num * self.block_num],
                                                              name="first_bmm_tensor_c_l0c",
                                                              scope=tbe_platform.scope_cc)
            first_bmm_tensor_c_ub = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * inner_n_size *
                                                                                 self.block_num * self.block_num],
                                                             name="first_bmm_tensor_c_ub",
                                                             scope=tbe_platform.scope_ubuf)
            elewise_add_data_ub = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * self.first_n_dim *
                                                                               self.block_num * self.block_num],
                                                           name="elewise_add_data_ub", scope=tbe_platform.scope_ubuf)
            first_bmm_compute_buffers = [first_bmm_tensor_a, first_bmm_tensor_b, first_bmm_tensor_c,
                                         first_bmm_tensor_a_l0a, first_bmm_tensor_b_l0b, first_bmm_tensor_c_l0c,
                                         first_bmm_tensor_c_ub, mul_value, elewise_add_data_ub]
            first_bmm_compute_idx = [block_idx, cur_b_idx, cur_om_idx, 2 * inner_m_idx]
            first_bmm_compute_each_layer_size = [outer_m_range_once_m_size, inner_m_size, inner_k_size, inner_n_size]
            self.mat_mul_compute(first_bmm_compute_buffers, first_bmm_compute_idx, first_bmm_compute_each_layer_size)

            first_bmm_tensor_a_db = self.tik_instance.Tensor(self.matmul_dtype, l1a_shape,
                                                             name="first_bmm_tensor_a_db",
                                                             scope=tbe_platform.scope_cbuf)

            first_bmm_tensor_a_l0a_db = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * inner_k_size *
                                                                                     self.block_num * self.block_num],
                                                                 name="first_bmm_tensor_a_l0a_db",
                                                                 scope=tbe_platform.scope_ca, start_addr=32768)
            first_bmm_tensor_c_l0c_db = self.tik_instance.Tensor("float32", [inner_m_size * inner_n_size *
                                                                             self.block_num * self.block_num],
                                                                 name="first_bmm_tensor_c_l0c_db",
                                                                 scope=tbe_platform.scope_cc, start_addr=32768)
            first_bmm_tensor_c_ub_db = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * inner_n_size *
                                                                                    self.block_num * self.block_num],
                                                                name="first_bmm_tensor_c_ub_db",
                                                                scope=tbe_platform.scope_ubuf)
            elewise_add_data_ub_db = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * inner_n_size *
                                                                                  self.block_num * self.block_num],
                                                              name="elewise_add_data_ub_db",
                                                              scope=tbe_platform.scope_ubuf)
            first_bmm_compute_db_buffers = [first_bmm_tensor_a_db, first_bmm_tensor_b, first_bmm_tensor_c,
                                            first_bmm_tensor_a_l0a_db, first_bmm_tensor_b_l0b,
                                            first_bmm_tensor_c_l0c_db, first_bmm_tensor_c_ub_db,
                                            mul_value, elewise_add_data_ub_db]
            first_bmm_compute_idx = [block_idx, cur_b_idx, cur_om_idx, 2 * inner_m_idx + 1]
            self.mat_mul_compute(first_bmm_compute_db_buffers, first_bmm_compute_idx, first_bmm_compute_each_layer_size)

        with self.tik_instance.if_scope(inner_m_range_value % self.double_factor != 0):
            inner_m_size, inner_k_size, inner_n_size = outer_m_range_once_m_size // inner_m_range_value, \
                                                       self.first_k_dim, self.first_n_dim
            l1a_shape = [inner_k_size, inner_m_size, self.block_num, self.block_num]
            first_bmm_tensor_a = self.tik_instance.Tensor(self.matmul_dtype, l1a_shape, name="first_bmm_tensor_a",
                                                          scope=tbe_platform.scope_cbuf)
            first_bmm_tensor_a_l0a = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * inner_k_size *
                                                                                  self.block_num * self.block_num],
                                                              name="first_bmm_tensor_a_l0a",
                                                              scope=tbe_platform.scope_ca)
            first_bmm_tensor_c_l0c = self.tik_instance.Tensor("float32", [inner_m_size * inner_n_size *
                                                                          self.block_num * self.block_num],
                                                              name="first_bmm_tensor_c_l0c",
                                                              scope=tbe_platform.scope_cc)
            first_bmm_tensor_c_ub = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * inner_n_size *
                                                                                 self.block_num * self.block_num],
                                                             name="first_bmm_tensor_c_ub",
                                                             scope=tbe_platform.scope_ubuf)
            elewise_add_data_ub = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * self.first_n_dim *
                                                                               self.block_num * self.block_num],
                                                           name="elewise_add_data_ub", scope=tbe_platform.scope_ubuf)
            first_bmm_compute_buffers = [first_bmm_tensor_a, first_bmm_tensor_b, first_bmm_tensor_c,
                                         first_bmm_tensor_a_l0a, first_bmm_tensor_b_l0b, first_bmm_tensor_c_l0c,
                                         first_bmm_tensor_c_ub, mul_value, elewise_add_data_ub]
            first_bmm_compute_idx = [block_idx, cur_b_idx, cur_om_idx, inner_m_range_value - 1]
            first_bmm_compute_each_layer_size = [outer_m_range_once_m_size, inner_m_size, inner_k_size, inner_n_size]
            self.mat_mul_compute(first_bmm_compute_buffers, first_bmm_compute_idx, first_bmm_compute_each_layer_size)

    def second_bmm_compute_for_outer_m_once(self, preload_buffers, range_idxs, outer_m_range_once_m_size,
                                            inner_m_range_value):
        second_bmm_tensor_b, second_bmm_tensor_b_l0b, second_bmm_tensor_c = preload_buffers
        block_idx, cur_b_idx, cur_om_idx = range_idxs
        if self.first_m_dim != self.trans_cube_target:
            self.tik_instance.load2dv2(second_bmm_tensor_b_l0b, second_bmm_tensor_b[0],
                                       0, self.first_n_dim * self.second_n_dim, 0, 1, 0, True)
        inner_m_size, inner_k_size, inner_n_size = outer_m_range_once_m_size // inner_m_range_value, \
                                                   self.second_k_dim, self.second_n_dim
        l1a_shape = [inner_k_size, inner_m_size, self.block_num, self.block_num]
        ub_start_addr = 0
        second_bmm_tensor_a = self.tik_instance.Tensor(self.matmul_dtype, l1a_shape, name="second_bmm_tensor_a",
                                                       scope=tbe_platform.scope_cbuf)
        second_bmm_tensor_a_l0a = self.tik_instance.Tensor(self.matmul_dtype, l1a_shape,
                                                           name="second_bmm_tensor_a_l0a", scope=tbe_platform.scope_ca)

        second_bmm_tensor_c_l0c = self.tik_instance.Tensor("float32", [inner_m_size * inner_n_size *
                                                                       self.block_num * self.block_num],
                                                           name="second_bmm_tensor_c_l0c", scope=tbe_platform.scope_cc)

        second_bmm_tensor_c_ub = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * inner_k_size *
                                                                              self.block_num * self.block_num],
                                                          name="second_bmm_tensor_c_ub", scope=tbe_platform.scope_ubuf,
                                                          start_addr=0)
        ub_start_addr = ub_start_addr + 32768
        second_bmm_mask_ub = self.tik_instance.Tensor("uint8", [inner_m_size * inner_k_size *
                                                                self.block_num * self.block_num],
                                                      name="second_bmm_mask_ub", scope=tbe_platform.scope_ubuf,
                                                      start_addr=ub_start_addr)

        ub_start_addr = ub_start_addr + inner_m_size * inner_k_size * self.block_num * self.block_num
        second_bmm_mask_cast_ub = self.tik_instance.Tensor("float32", [inner_m_size * inner_k_size *
                                                                       self.block_num * self.block_num],
                                                           name="second_bmm_mask_cast_ub",
                                                           scope=tbe_platform.scope_ubuf,
                                                           start_addr=ub_start_addr)

        ub_start_addr = ub_start_addr + inner_m_size * inner_k_size * self.block_num * self.block_num * 4
        second_bmm_softmax_reduce_ub = self.tik_instance.Tensor("float32", [inner_m_size * self.block_num],
                                                                name="second_bmm_softmax_reduce_ub",
                                                                scope=tbe_platform.scope_ubuf, start_addr=ub_start_addr)
        trans_cube_tensor_b_l0b = second_bmm_tensor_b_l0b[0]
        trans_cube_tensor_b_l1 = None
        if self.first_m_dim == self.trans_cube_target:
            trans_cube_tensor_b_l1 = self.tik_instance.Tensor(self.matmul_dtype,
                                                              [inner_k_size * self.block_num + self.block_num],
                                                              name="trans_cube_tensor_b_l1",
                                                              scope=tbe_platform.scope_cbuf)
        # db part tensor
        second_bmm_tensor_a_db = self.tik_instance.Tensor(self.matmul_dtype, l1a_shape, name="second_bmm_tensor_a_db",
                                                          scope=tbe_platform.scope_cbuf, start_addr=524288)
        second_bmm_tensor_a_l0a_db = self.tik_instance.Tensor(self.matmul_dtype, l1a_shape,
                                                              name="second_bmm_tensor_a_l0a_db",
                                                              scope=tbe_platform.scope_ca, start_addr=32768)

        second_bmm_tensor_c_l0c_db = self.tik_instance.Tensor("float32", [inner_m_size * inner_n_size *
                                                                          self.block_num * self.block_num],
                                                              name="second_bmm_tensor_c_l0c_db",
                                                              scope=tbe_platform.scope_cc, start_addr=32768)

        ub_start_addr = 131072
        second_bmm_tensor_c_ub_db = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * inner_k_size *
                                                                                 self.block_num * self.block_num],
                                                             name="second_bmm_tensor_c_ub_db",
                                                             scope=tbe_platform.scope_ubuf, start_addr=ub_start_addr)

        ub_start_addr = ub_start_addr + inner_m_size * inner_k_size * self.block_num * self.block_num * 2
        second_bmm_mask_ub_db = self.tik_instance.Tensor("uint8", [inner_m_size * inner_k_size *
                                                                   self.block_num * self.block_num],
                                                         name="second_bmm_mask_ub_db",
                                                         scope=tbe_platform.scope_ubuf, start_addr=ub_start_addr)

        ub_start_addr = ub_start_addr + inner_m_size * inner_k_size * self.block_num * self.block_num
        second_bmm_mask_cast_ub_db = self.tik_instance.Tensor("float32", [inner_m_size * inner_k_size *
                                                                          self.block_num * self.block_num],
                                                              name="second_bmm_mask_cast_ub_db",
                                                              scope=tbe_platform.scope_ubuf,
                                                              start_addr=ub_start_addr)

        ub_start_addr = ub_start_addr + inner_m_size * inner_k_size * self.block_num * self.block_num * 4
        second_bmm_softmax_reduce_ub_db = self.tik_instance.Tensor("float32", [inner_m_size * self.block_num],
                                                                   name="second_bmm_softmax_reduce_ub_db",
                                                                   scope=tbe_platform.scope_ubuf,
                                                                   start_addr=ub_start_addr)
        trans_cube_tensor_b_l0b_db = second_bmm_tensor_b_l0b[self.first_k_dim * self.first_n_dim * self.block_num * \
                                                             self.block_num // 2]
        trans_cube_tensor_b_l1_db = None
        if self.first_m_dim == self.trans_cube_target:
            trans_cube_tensor_b_l1_db = self.tik_instance.Tensor(self.matmul_dtype,
                                                                 [inner_k_size * self.block_num + self.block_num],
                                                                 name="trans_cube_tensor_b_l1_db",
                                                                 scope=tbe_platform.scope_cbuf)
        second_bmm_compute_buffers = [second_bmm_tensor_a, second_bmm_tensor_b_l0b, second_bmm_tensor_c,
                                      second_bmm_tensor_a_l0a, second_bmm_tensor_b_l0b, second_bmm_tensor_c_l0c,
                                      second_bmm_tensor_c_ub, second_bmm_mask_ub, second_bmm_mask_cast_ub,
                                      second_bmm_softmax_reduce_ub, trans_cube_tensor_b_l1,
                                      trans_cube_tensor_b_l0b, second_bmm_tensor_b]

        second_bmm_compute_db_buffers = [second_bmm_tensor_a_db, second_bmm_tensor_b_l0b,
                                         second_bmm_tensor_a_l0a_db, second_bmm_tensor_c_l0c_db,
                                         second_bmm_tensor_c_ub_db, second_bmm_mask_ub_db, second_bmm_mask_cast_ub_db,
                                         second_bmm_softmax_reduce_ub_db, trans_cube_tensor_b_l1_db,
                                         trans_cube_tensor_b_l0b_db]
        second_bmm_compute_idxs = [block_idx, cur_b_idx, cur_om_idx, 0]
        second_bmm_compute_each_layer_size = [outer_m_range_once_m_size, inner_m_size, inner_k_size, inner_n_size]
        self.mat_mul_second_compute_front(second_bmm_compute_buffers, second_bmm_compute_db_buffers,
                                          second_bmm_compute_idxs, second_bmm_compute_each_layer_size,
                                          self.input_keep_prob)
        unroll_range = inner_m_range_value - 1
        for cur_m_idx in range(unroll_range):
            second_bmm_compute_idxs = [block_idx, cur_b_idx, cur_om_idx, cur_m_idx]
            if cur_m_idx % 2 == 0:
                second_bmm_compute_buffers = [second_bmm_tensor_a, second_bmm_tensor_b, second_bmm_tensor_c,
                                              second_bmm_tensor_a_l0a, second_bmm_tensor_b_l0b, second_bmm_tensor_c_l0c,
                                              second_bmm_tensor_c_ub, second_bmm_mask_ub, second_bmm_mask_cast_ub,
                                              second_bmm_softmax_reduce_ub, trans_cube_tensor_b_l1,
                                              trans_cube_tensor_b_l0b, second_bmm_tensor_b]

                second_bmm_compute_db_buffers = [second_bmm_tensor_a_db, second_bmm_tensor_b,
                                                 second_bmm_tensor_a_l0a_db, second_bmm_tensor_c_l0c_db,
                                                 second_bmm_tensor_c_ub_db, second_bmm_mask_ub_db,
                                                 second_bmm_mask_cast_ub_db, second_bmm_softmax_reduce_ub_db,
                                                 trans_cube_tensor_b_l1_db, trans_cube_tensor_b_l0b_db]
            else:
                second_bmm_compute_buffers = [second_bmm_tensor_a_db, second_bmm_tensor_b, second_bmm_tensor_c,
                                              second_bmm_tensor_a_l0a_db, second_bmm_tensor_b_l0b,
                                              second_bmm_tensor_c_l0c_db, second_bmm_tensor_c_ub_db,
                                              second_bmm_mask_ub_db, second_bmm_mask_cast_ub_db,
                                              second_bmm_softmax_reduce_ub_db,
                                              trans_cube_tensor_b_l1_db, trans_cube_tensor_b_l0b_db,
                                              second_bmm_tensor_b]

                second_bmm_compute_db_buffers = [second_bmm_tensor_a, second_bmm_tensor_b,
                                                 second_bmm_tensor_a_l0a, second_bmm_tensor_c_l0c,
                                                 second_bmm_tensor_c_ub, second_bmm_mask_ub,
                                                 second_bmm_mask_cast_ub, second_bmm_softmax_reduce_ub,
                                                 trans_cube_tensor_b_l1, trans_cube_tensor_b_l0b]
            self.mat_mul_second_compute_mid(second_bmm_compute_buffers, second_bmm_compute_db_buffers,
                                            second_bmm_compute_idxs, second_bmm_compute_each_layer_size,
                                            self.input_keep_prob)
        if inner_m_range_value % 2 == 0:
            second_bmm_compute_buffers = [second_bmm_tensor_a, second_bmm_tensor_b, second_bmm_tensor_c,
                                          second_bmm_tensor_a_l0a, second_bmm_tensor_b_l0b, second_bmm_tensor_c_l0c,
                                          second_bmm_tensor_c_ub, second_bmm_mask_ub, second_bmm_mask_cast_ub,
                                          second_bmm_softmax_reduce_ub, trans_cube_tensor_b_l1,
                                          trans_cube_tensor_b_l0b, second_bmm_tensor_b]
            second_bmm_compute_db_buffers = [second_bmm_tensor_a_db, second_bmm_tensor_b,
                                             second_bmm_tensor_a_l0a_db, second_bmm_tensor_c_l0c_db,
                                             second_bmm_tensor_c_ub_db, second_bmm_mask_ub_db,
                                             second_bmm_mask_cast_ub_db, second_bmm_softmax_reduce_ub_db,
                                             trans_cube_tensor_b_l1_db, trans_cube_tensor_b_l0b_db]
        else:
            second_bmm_compute_buffers = [second_bmm_tensor_a_db, second_bmm_tensor_b, second_bmm_tensor_c,
                                          second_bmm_tensor_a_l0a_db, second_bmm_tensor_b_l0b,
                                          second_bmm_tensor_c_l0c_db, second_bmm_tensor_c_ub_db,
                                          second_bmm_mask_ub_db, second_bmm_mask_cast_ub_db,
                                          second_bmm_softmax_reduce_ub_db,
                                          trans_cube_tensor_b_l1_db, trans_cube_tensor_b_l0b_db,
                                          second_bmm_tensor_b]
            second_bmm_compute_db_buffers = [second_bmm_tensor_a, second_bmm_tensor_b,
                                             second_bmm_tensor_a_l0a, second_bmm_tensor_c_l0c,
                                             second_bmm_tensor_c_ub, second_bmm_mask_ub,
                                             second_bmm_mask_cast_ub, second_bmm_softmax_reduce_ub,
                                             trans_cube_tensor_b_l1,
                                             trans_cube_tensor_b_l0b]
        second_bmm_compute_idxs = [block_idx, cur_b_idx, cur_om_idx, unroll_range]
        self.mat_mul_second_compute_last(second_bmm_compute_buffers, second_bmm_compute_db_buffers,
                                         second_bmm_compute_idxs, second_bmm_compute_each_layer_size,
                                         self.input_keep_prob)

    def compute_process(self):
        with self.tik_instance.for_range(0, self.cur_op_core_num, block_num=self.cur_op_core_num) as block_idx:
            batch_range_value, outer_m_range_value, inner_m_range_value = self.tiling_batch_m_axis(self.batch_per_core,
                                                                                                   self.first_m_dim)
            outer_m_range_once_m_size = self.first_m_dim // outer_m_range_value
            mul_value = self.tik_instance.Scalar("float16", "mul_value", init_value=-1)
            mul_x_ub = self.tik_instance.Tensor(self.matmul_dtype, [self.block_num],
                                                name="mul_x_ub", scope=tbe_platform.scope_ubuf)
            self.tik_instance.data_move(mul_x_ub, self.mul_gm[0], sid=0, nburst=1, burst=1, src_stride=0, dst_stride=0)
            mul_value.set_as(mul_x_ub[0])

            first_bmm_tensor_b = self.tik_instance.Tensor(self.matmul_dtype, [self.first_k_dim, self.first_n_dim,
                                                                              self.block_num, self.block_num],
                                                          name="first_bmm_tensor_b", scope=tbe_platform.scope_cbuf)
            second_bmm_tensor_b = self.tik_instance.Tensor(self.matmul_dtype, [self.second_n_dim, self.second_k_dim,
                                                                               self.block_num, self.block_num],
                                                           name="second_bmm_tensor_b", scope=tbe_platform.scope_cbuf)
            core_offset = self.tik_instance.Scalar("int32", name="core_offset")
            batch_per_core_truth = self.tik_instance.Scalar("int32", name="batch_per_core_truth")
            with self.tik_instance.if_scope(block_idx < self.batch_large_core_num):
                batch_per_core_truth.set_as(self.batch_per_core)
                core_offset.set_as(block_idx * self.batch_per_core)
            with self.tik_instance.else_scope():
                batch_per_core_truth.set_as(self.batch_small_per_core)
                core_offset.set_as(self.batch_large_core_num * self.batch_per_core + \
                                   (block_idx - self.batch_large_core_num) * self.batch_small_per_core)
            with self.tik_instance.for_range(0, batch_per_core_truth) as cur_b_idx:
                self.batch_outer_num = core_offset + cur_b_idx
                self.first_bmm_move_tensor_b_from_gm_to_l1(first_bmm_tensor_b)
                self.second_bmm_move_tensor_b_from_gm_to_l1(second_bmm_tensor_b)
                first_bmm_tensor_c = self.apply_buffer_for_tensor_c_l1(outer_m_range_value)
                first_bmm_tensor_b_l0b = self.apply_buffer_for_tensor_b_l0_and_move_data_in(first_bmm_tensor_b)
                with self.tik_instance.for_range(0, outer_m_range_value) as cur_om_idx:
                    range_idxs = [block_idx, cur_b_idx, cur_om_idx]
                    first_preload_buffers = [first_bmm_tensor_b, first_bmm_tensor_c, first_bmm_tensor_b_l0b]
                    self.first_bmm_compute_for_outer_m_once(first_preload_buffers, range_idxs,
                                                            outer_m_range_once_m_size, inner_m_range_value, mul_value)

                with self.tik_instance.for_range(0, outer_m_range_value) as cur_om_idx:
                    range_idxs = [block_idx, cur_b_idx, cur_om_idx]
                    second_bmm_tensor_b_l0b = first_bmm_tensor_b_l0b
                    second_bmm_tensor_c = first_bmm_tensor_c
                    second_bmm_preload_buffers = [second_bmm_tensor_b, second_bmm_tensor_b_l0b, second_bmm_tensor_c]
                    self.second_bmm_compute_for_outer_m_once(second_bmm_preload_buffers, range_idxs,
                                                             outer_m_range_once_m_size, inner_m_range_value)
        if self.swin_struc:
            input_gm_list = [self.x1_gm, self.x2_gm, self.x3_gm, self.add1_gm, self.add2_gm,
                             self.mul_gm, self.drop_mask_gm]
        elif self.vit_struc:
            input_gm_list = [self.x1_gm, self.x2_gm, self.x3_gm, self.mul_gm]
        else:
            input_gm_list = [self.x1_gm, self.x2_gm, self.x3_gm, self.add1_gm, self.mul_gm]
        output_gm_list = [self.y_gm]
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=input_gm_list,
                                   outputs=output_gm_list, config={})


# 'pylint: disable=redefined-builtin
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            (para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_LIST_INT),
                            para_check.KERNEL_NAME)
def swin_attention_score(query, key, value, padding_mask1, padding_mask2, scale, drop_mask,
                         attention_score_output, softmax_output, keep_prob,
                         query_transpose=False, key_transpose=False,
                         bmm_score_transpose_a=False, bmm_score_transpose_b=False,
                         softmax_axes=-1, kernel_name="swin_attention_score"):
    
    if check_supported_vit(query, key, value, padding_mask1, padding_mask2, scale, drop_mask):
        vit_flash_attention_score(query, key, value, padding_mask1, padding_mask2, scale, drop_mask,
                                  attention_score_output, softmax_output, keep_prob,
                                  query_transpose, key_transpose, bmm_score_transpose_a,
                                  bmm_score_transpose_b, softmax_axes, kernel_name)
    else:
        op_init = MatMulSoftmax(query, key, scale, padding_mask1, padding_mask2, drop_mask, value,
                                softmax_output, attention_score_output, keep_prob, softmax_axes,
                                query_transpose, key_transpose, bmm_score_transpose_a, bmm_score_transpose_b,
                                kernel_name)
        op_init.compute_process()
