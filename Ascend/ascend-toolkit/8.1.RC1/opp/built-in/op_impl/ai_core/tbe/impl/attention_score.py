#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

mat_mul_softmax_dropout_matmul
"""

from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=too-many-arguments,too-many-locals
# 'pylint: disable=too-few-public-methods
# 'pylint: disable=too-many-statements, too-many-arguments, too-many-lines
class Constant:
    """
    The class for constant
    """
    MININUM_NUM_FLOAT = -(3.4028235 ** 38)
    DTYPE_BYTES = {"float32": 4, "float16": 2}
    TRAINGING = 0
    TUILI = 1


# 'pylint: disable=too-many-public-methods
class MatMulSoftmax:
    """
    MatMulSoftmax class
    """

    # 'pylint: disable=unused-argument
    def __init__(self, x1, x2, mul_x, add_x, drop_mask, x3,
                 softmax_output, y,
                 input_keep_prob, axis,
                 first_transpose_a, first_transpose_b,
                 second_transpose_a, second_transpose_b,
                 kernel_name):
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.tik = tik
        self.cur_op_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.matmul_dtype = "float16"
        self.vector_dtype = "float16"
        self.x1_shape = x1["shape"]
        self.x2_shape = x2["shape"]
        self.x3_shape = x3["shape"]
        self.model_type = Constant.TRAINGING
        self.drop_v3 = False
        if drop_mask["dtype"] == "float16":
            self.model_type = Constant.TUILI
        if self.model_type == Constant.TRAINGING:
            self.softmax_output_shape = softmax_output["shape"]
        self.block_num = 16
        self.drop_shape = drop_mask["shape"]
        if self.drop_shape[0] == self.x1_shape[0] or \
        self.drop_shape[0] == self.x1_shape[0] * self.x1_shape[1] * self.x1_shape[3] * \
        self.x1_shape[3] * self.block_num * self.block_num:
            self.drop_v3 = True
        self.ele_shape = add_x["shape"]
        if self.model_type == Constant.TRAINGING:
            self.softmax_output_shape = softmax_output["shape"]
        self.y_shape = y["shape"]

        self.first_m_dim = self.x1_shape[3]
        self.first_k_dim = self.x1_shape[2]
        self.first_n_dim = self.x2_shape[3]

        self.second_m_dim = self.x1_shape[3]
        self.second_k_dim = self.x2_shape[3]
        self.second_n_dim = self.x3_shape[2]

        self.repeat_once_size = 128
        self.block_stride = 1
        self.repeat_stride = 8
        self.trans_cube_target = 8

        self.m_num = self.first_m_dim
        self.input_keep_prob = input_keep_prob
        self.batch_outer_num = 0

        self.mul_x_shape = [self.block_num]
        self.double_factor = 2
        self.kernel_name = kernel_name
        self.init_gm()

    @staticmethod
    def get_eatch_core_batch(b, n, core):
        all_ranges = b * n
        batch_range_value = (all_ranges + core - 1) // core
        used_core_num = (all_ranges + batch_range_value - 1) // batch_range_value
        batch_range_value_last = all_ranges - batch_range_value * (used_core_num - 1)
        return batch_range_value, batch_range_value_last, used_core_num

    def init_gm(self):
        self.x1_gm = self.tik_instance.Tensor(self.matmul_dtype, self.x1_shape,
                                              name="x1_gm", scope=self.tik.scope_gm)
        self.x2_gm = self.tik_instance.Tensor(self.matmul_dtype, self.x2_shape,
                                              name="x2_gm", scope=self.tik.scope_gm)
        self.mul_gm = self.tik_instance.Tensor(self.matmul_dtype, self.mul_x_shape,
                                               name="mul_gm", scope=self.tik.scope_gm)
        self.add_gm = self.tik_instance.Tensor(self.matmul_dtype, self.ele_shape,
                                               name="add_gm", scope=self.tik.scope_gm)
        self.drop_mask_gm = self.tik_instance.Tensor("uint8", self.drop_shape,
                                                     name="drop_mask_gm", scope=self.tik.scope_gm)
        self.x3_gm = self.tik_instance.Tensor(self.matmul_dtype, self.x3_shape,
                                              name="x3_gm", scope=self.tik.scope_gm)
        if self.model_type == Constant.TRAINGING:
            self.softmax_output_gm = self.tik_instance.Tensor(self.matmul_dtype, self.softmax_output_shape,
                                                              name="softmax_output_gm", scope=self.tik.scope_gm)

        self.y_gm = self.tik_instance.Tensor(self.matmul_dtype, self.y_shape,
                                             name="y_gm", scope=self.tik.scope_gm)

    def load_2d(self, src, dst, instr_params):
        '''
        load_2d instr is different in different platforms
        '''
        start_index, repeat, repeat_stride, sid, is_transpose = instr_params
        if tbe_platform.api_check_support("tik.load2dv2"):
            self.tik_instance.load2dv2(src, dst, start_index, repeat, 0, repeat_stride, sid, is_transpose)
        elif tbe_platform.api_check_support("tik.load2dv1"):
            self.tik_instance.load2dv1(src, dst, start_index, repeat, repeat_stride, sid, is_transpose)
        else:
            error_manager_cube.raise_err_specific_user("attention_score", "load2d instr unsupported.")

    def mid_data_to_ub(self, tensor_c, tensor_c_ub, om_size, cur_m_idx, cur_m_size, m, single_m_size,
                       ub_mask, block_idx, cur_b_idx, cur_om_idx):
        tensor_a_src_offset = cur_m_idx * cur_m_size * self.block_num * self.block_num + \
                              m * single_m_size * self.block_num * self.block_num
        tensor_a_dst_offset = 0
        tensor_a_repeat_times = self.first_n_dim
        tesnor_a_data_size = single_m_size * self.block_num
        tensor_a_src_stride = (om_size - single_m_size) * self.block_num
        tensor_a_dst_stride = 0
        self.tik_instance.data_move(tensor_c_ub[tensor_a_dst_offset], tensor_c[tensor_a_src_offset],
                                    sid=0, nburst=tensor_a_repeat_times, burst=tesnor_a_data_size,
                                    src_stride=tensor_a_src_stride, dst_stride=tensor_a_dst_stride)

        mask_offset = block_idx * self.batch_outer_num * self.first_m_dim * \
                      self.first_n_dim * self.block_num * self.block_num + \
                      cur_b_idx * self.first_m_dim * self.first_n_dim * self.block_num * self.block_num + \
                      cur_om_idx * om_size * self.block_num * self.block_num + \
                      cur_m_idx * cur_m_size * self.block_num * self.block_num
        mask_repeat_times = self.first_n_dim
        mask_length = single_m_size * self.block_num // 2
        mask_src_stride = (self.first_m_dim - single_m_size) * self.block_num // 2
        mask_dst_stride = 0
        if self.model_type == Constant.TRAINGING:
            if self.drop_v3 is not True:
                mask_offset = block_idx * self.batch_outer_num * self.first_m_dim * self.first_n_dim * \
                              self.block_num * self.block_num // 8 + cur_b_idx * self.first_m_dim * \
                              self.first_n_dim * self.block_num * self.block_num // 8 + \
                              cur_m_idx * single_m_size * 32
                mask_repeat_times = self.first_n_dim
                mask_length = single_m_size * self.block_num * self.block_num // 32 // 8
                mask_src_stride = (self.first_m_dim - single_m_size)
                mask_dst_stride = 0
            self.tik_instance.data_move(ub_mask[0], self.drop_mask_gm[mask_offset],
                                        0, mask_repeat_times, mask_length,
                                        mask_src_stride, mask_dst_stride)

    def get_second_single_time_mkn(self, m, k, n):
        """
        compute M K N value for once time.
        """
        return m, k, n

    def mat_mul_second_compute_front(self, second_bmm_compute_buffers, second_bmm_compute_db_buffers,
                                     second_bmm_compute_idxs, second_bmm_compute_each_layer_size,
                                     input_keep_prob):
        tensor_a, tensor_b, tensor_c, tensor_a_l0a, tensor_b_l0b, \
        tensor_c_l0c, tensor_c_ub, ub_mask, ub_cast, reduce_ub, \
        trans_cube_tensor_b_l1, trans_cube_tensor_b_l0b, second_bmm_tensor_b = second_bmm_compute_buffers
        tensor_a_l1a_s_ub, tensor_b_s, tensor_a_l0a_ub, tensor_c_l0c_ub, \
        tensor_c_ub2, ub_mask2, ub_cast2, reduce_ub2, \
        trans_cube_tensor_b_l1_db, trans_cube_tensor_b_l0b_db = second_bmm_compute_db_buffers
        block_idx, cur_b_idx, cur_om_idx, cur_m_idx = second_bmm_compute_idxs
        om_size, cur_m_size, cur_k_size, cur_n_size = second_bmm_compute_each_layer_size

        single_m_size, single_k_size, single_n_size = self.get_second_single_time_mkn(cur_m_size,
                                                                                      cur_k_size, cur_n_size)
        m_range = cur_m_size // single_m_size
        n_range = cur_n_size // single_n_size

        with self.tik_instance.for_range(0, m_range) as m:
            with self.tik_instance.for_range(0, n_range) as n:
                self.mid_data_to_ub(tensor_c, tensor_c_ub, om_size, cur_m_idx, cur_m_size, m, single_m_size,
                                    ub_mask, block_idx, cur_b_idx, cur_om_idx)
                softmax_last_part_buffers = [tensor_c_ub, ub_cast, reduce_ub, ub_mask,
                                             tensor_a, trans_cube_tensor_b_l1,
                                             tensor_a_l0a, trans_cube_tensor_b_l0b, tensor_c_l0c]
                softmax_last_part_size = [om_size, cur_m_size, cur_k_size, single_m_size, single_k_size]
                softmax_last_part_idx = [input_keep_prob, None, block_idx, cur_b_idx, cur_om_idx, cur_m_idx, m]
                tensor_c_ub_back = self.softmax_compute_last_part(softmax_last_part_buffers, softmax_last_part_size,
                                                                  softmax_last_part_idx)
                self.tik_instance.data_move(tensor_a[0], tensor_c_ub_back[0],
                                            sid=0, nburst = 1, burst = single_k_size * single_m_size * self.block_num,
                                            src_stride=0, dst_stride=0)

    def mat_mul_second_compute_mid(self, second_bmm_compute_buffers, second_bmm_compute_db_buffers,
                                   second_bmm_compute_idxs, second_bmm_compute_each_layer_size,
                                   input_keep_prob):
        tensor_a, tensor_b, tensor_c, tensor_a_l0a, tensor_b_l0b, \
        tensor_c_l0c, tensor_c_ub, ub_mask, ub_cast, reduce_ub, \
        trans_cube_tensor_b_l1, trans_cube_tensor_b_l0b, second_bmm_tensor_b = second_bmm_compute_buffers
        tensor_a_l1a_s_ub, tensor_b_s, tensor_a_l0a_ub, tensor_c_l0c_ub, \
        tensor_c_ub2, ub_mask2, ub_cast2, reduce_ub2, \
        trans_cube_tensor_b_l1_db, trans_cube_tensor_b_l0b_db = second_bmm_compute_db_buffers
        block_idx, cur_b_idx, cur_om_idx, cur_m_idx = second_bmm_compute_idxs
        om_size, cur_m_size, cur_k_size, cur_n_size = second_bmm_compute_each_layer_size

        single_m_size, single_k_size, single_n_size = self.get_second_single_time_mkn(cur_m_size,
                                                                                      cur_k_size, cur_n_size)
        m_range = cur_m_size // single_m_size
        n_range = cur_n_size // single_n_size

        with self.tik_instance.for_range(0, m_range) as m:
            with self.tik_instance.for_range(0, n_range) as n:
                # do the next time
                self.mid_data_to_ub(tensor_c, tensor_c_ub2, om_size, cur_m_idx + 1, cur_m_size, m, single_m_size,
                                    ub_mask2, block_idx, cur_b_idx, cur_om_idx)
                l1a_offset = 0
                l1b_offset = 0
                l1a_repeat_times = single_m_size * cur_k_size
                l1b_repeat_times = single_n_size * cur_k_size
                # do the last time
                self.load_2d(tensor_a_l0a, tensor_a[l1a_offset], [0, l1a_repeat_times, 1, 0, False])
                if self.first_m_dim == self.trans_cube_target:
                    self.load_2d(tensor_b_l0b, second_bmm_tensor_b[0], \
                                [0, self.first_n_dim * self.second_n_dim, 1, 0, True])
                self.tik_instance.mmad(tensor_c_l0c, tensor_a_l0a, tensor_b_l0b,
                                       single_m_size * self.block_num,
                                       single_k_size * self.block_num,
                                       single_n_size * self.block_num,
                                       0)
                # do the next time
                softmax_last_part_buffers = [tensor_c_ub2, ub_cast2, reduce_ub2, ub_mask2,
                                             tensor_a_l1a_s_ub, trans_cube_tensor_b_l1_db,
                                             tensor_a_l0a_ub, trans_cube_tensor_b_l0b_db, tensor_c_l0c_ub]
                softmax_last_part_size = [om_size, cur_m_size, cur_k_size, single_m_size, single_k_size]
                softmax_last_part_idx = [input_keep_prob, None, block_idx, cur_b_idx, cur_om_idx, cur_m_idx + 1, m]
                tensor_c_ub_back_2 = self.softmax_compute_last_part(softmax_last_part_buffers, softmax_last_part_size,
                                                                    softmax_last_part_idx)
                self.tik_instance.data_move(tensor_a_l1a_s_ub[0], tensor_c_ub_back_2[0],
                                            sid=0, nburst=1, burst=single_k_size * single_m_size * self.block_num,
                                            src_stride=0, dst_stride=0)

                # do the last time
                cc_to_ub_dst_stride = 0
                cc_to_ub_src_stride = 0
                ub_mask16 = ub_mask.reinterpret_cast_to("float16")
                self.tik_instance.tensor_mov(ub_mask16, tensor_c_l0c,
                                             'm', 1, single_m_size * single_n_size,
                                             cc_to_ub_dst_stride, cc_to_ub_src_stride)
                outer_blk = (block_idx * self.batch_outer_num + cur_b_idx) // self.x1_shape[1]
                inner_blk = (block_idx * self.batch_outer_num + cur_b_idx) % self.x1_shape[1]
                single_data_size = single_m_size * self.block_num
                repeat_times = single_n_size
                output_dst_stride = (self.x1_shape[0] * self.second_m_dim - single_m_size) * self.block_num
                # inner depend which row
                # outer depends which 2[32, 16, 16]
                output_dst_offset = inner_blk * single_n_size * self.x1_shape[0] * \
                                    self.second_m_dim * self.block_num * self.block_num + \
                                    outer_blk * self.second_m_dim * self.block_num * self.block_num + \
                                    cur_om_idx * om_size * self.block_num * self.block_num + \
                                    cur_m_idx * cur_m_size * self.block_num * self.block_num + \
                                    m * single_m_size * self.block_num * self.block_num

                self.tik_instance.data_move(self.y_gm[output_dst_offset], ub_mask16[0],
                                            sid=0, nburst=repeat_times, burst=single_data_size,
                                            src_stride=0, dst_stride=output_dst_stride)

    def mat_mul_second_compute_last(self, second_bmm_compute_buffers, second_bmm_compute_db_buffers,
                                   second_bmm_compute_idxs, second_bmm_compute_each_layer_size,
                                   input_keep_prob):
        tensor_a, tensor_b, tensor_c, tensor_a_l0a, tensor_b_l0b, \
        tensor_c_l0c, tensor_c_ub, ub_mask, ub_cast, reduce_ub, \
        trans_cube_tensor_b_l1, trans_cube_tensor_b_l0b, second_bmm_tensor_b = second_bmm_compute_buffers
        tensor_a_l1a_s_ub, tensor_b_s, tensor_a_l0a_ub, tensor_c_l0c_ub, \
        tensor_c_ub2, ub_mask2, ub_cast2, reduce_ub2, \
        trans_cube_tensor_b_l1_db, trans_cube_tensor_b_l0b_db = second_bmm_compute_db_buffers
        block_idx, cur_b_idx, cur_om_idx, cur_m_idx = second_bmm_compute_idxs
        om_size, cur_m_size, cur_k_size, cur_n_size = second_bmm_compute_each_layer_size

        single_m_size, single_k_size, single_n_size = self.get_second_single_time_mkn(cur_m_size,
                                                                                      cur_k_size, cur_n_size)
        m_range = cur_m_size // single_m_size
        n_range = cur_n_size // single_n_size

        with self.tik_instance.for_range(0, m_range) as m:
            with self.tik_instance.for_range(0, n_range) as n:
                l1a_offset = 0
                l1b_offset = 0
                l1a_repeat_times = single_m_size * cur_k_size
                l1b_repeat_times = single_n_size * cur_k_size
                # do the last time
                self.load_2d(tensor_a_l0a_ub, tensor_a_l1a_s_ub[l1a_offset], [0, l1a_repeat_times, 1, 0, False])
                if self.first_m_dim == self.trans_cube_target:
                    self.load_2d(tensor_b_l0b, second_bmm_tensor_b[0], \
                                [0, self.first_n_dim * self.second_n_dim, 1, 0, True])
                self.tik_instance.mmad(tensor_c_l0c_ub, tensor_a_l0a_ub, tensor_b_l0b,
                                       single_m_size * self.block_num,
                                       single_k_size * self.block_num,
                                       single_n_size * self.block_num,
                                       0)

                # do the last time
                cc_to_ub_dst_stride = 0
                cc_to_ub_src_stride = 0

                self.tik_instance.tensor_mov(tensor_c_ub2, tensor_c_l0c_ub,
                                             'm', 1, single_m_size * single_n_size,
                                             cc_to_ub_dst_stride, cc_to_ub_src_stride)
                outer_blk = (block_idx * self.batch_outer_num + cur_b_idx) // self.x1_shape[1]
                inner_blk = (block_idx * self.batch_outer_num + cur_b_idx) % self.x1_shape[1]
                single_data_size = single_m_size * self.block_num
                repeat_times = single_n_size
                output_dst_stride = (self.x1_shape[0] * self.second_m_dim - single_m_size) * self.block_num
                # inner depend which row
                # outer depends which 2[32, 16, 16]
                output_dst_offset = inner_blk * single_n_size * self.x1_shape[0] * \
                                    self.second_m_dim * self.block_num * self.block_num + \
                                    outer_blk * self.second_m_dim * self.block_num * self.block_num + \
                                    cur_om_idx * om_size * self.block_num * self.block_num + \
                                    cur_m_idx * cur_m_size * self.block_num * self.block_num + \
                                    m * single_m_size * self.block_num * self.block_num

                self.tik_instance.data_move(self.y_gm[output_dst_offset], tensor_c_ub2[0],
                                            sid=0, nburst=repeat_times, burst=single_data_size,
                                            src_stride=0, dst_stride=output_dst_stride)

    def get_first_single_time_mkn(self, m, k, n):
        return m, k, n

    def mat_mul_compute(self, first_bmm_compute_buffers, first_bmm_compute_idx, first_bmm_compute_each_layer_size):
        """
        first bmm compute.
        """
        tensor_a, tensor_b, tensor_y, tensor_a_l0a, \
        tensor_b_l0b, tensor_c_l0c, tensor_c_ub, mul_value, elewise_data_ub2 = first_bmm_compute_buffers
        block_idx, cur_b_idx, cur_om_idx, cur_m_idx = first_bmm_compute_idx
        om_size, cur_m_size, cur_k_size, cur_n_size = first_bmm_compute_each_layer_size
        single_m_size, single_k_size, single_n_size = self.get_first_single_time_mkn(cur_m_size,
                                                                                     cur_k_size, cur_n_size)
        m_range = cur_m_size // single_m_size
        n_range = cur_n_size // single_n_size

        l0a_shape = [single_k_size, single_m_size, self.block_num, self.block_num]
        l0b_shape = [single_k_size, single_n_size, self.block_num, self.block_num]
        l0c_shape = [single_n_size, single_m_size, self.block_num, self.block_num]

        with self.tik_instance.for_range(0, m_range) as m:
            with self.tik_instance.for_range(0, n_range) as n:
                for ck in range(cur_k_size):
                    tensor_a_src_offset = block_idx * self.batch_outer_num * self.first_m_dim * self.first_k_dim * \
                                          self.block_num * self.block_num + \
                                          cur_b_idx * self.first_m_dim * self.first_k_dim * \
                                          self.block_num * self.block_num + \
                                          cur_om_idx * om_size * self.block_num * self.block_num + \
                                          cur_m_idx * cur_m_size * self.block_num * self.block_num + \
                                          m * single_m_size * self.block_num * self.block_num + \
                                          ck * self.first_m_dim * \
                                          self.block_num * self.block_num
                    tensor_a_dst_offset = ck * self.block_num * self.block_num
                    tesnor_a_data_size = self.block_num
                    tensor_a_repeat_times = single_m_size
                    tensor_a_src_stride = 0
                    tensor_a_dst_stride = (cur_k_size - 1) * self.block_num
                    self.tik_instance.data_move(tensor_a[tensor_a_dst_offset], self.x1_gm[tensor_a_src_offset],
                                                sid=0, nburst=tensor_a_repeat_times, burst=tesnor_a_data_size,
                                                src_stride=tensor_a_src_stride, dst_stride=tensor_a_dst_stride)

                first_mov = (block_idx * self.batch_outer_num + cur_b_idx) // self.x1_shape[1]
                ele_move_offset = first_mov * self.first_m_dim * self.first_n_dim * self.block_num * self.block_num + \
                                  cur_om_idx * om_size * self.block_num * self.block_num + \
                                  cur_m_idx * cur_m_size * self.block_num * self.block_num + \
                                  m * single_m_size * self.block_num * self.block_num + \
                                  n * self.first_m_dim * self.block_num * self.block_num
                ele_move_repeat_times = single_n_size
                ele_move_data_size = single_m_size * self.block_num
                ele_move_src_stride = (self.first_m_dim - single_m_size) * self.block_num
                ele_move_dst_stride = 0
                self.tik_instance.data_move(elewise_data_ub2, self.add_gm[ele_move_offset],
                                            sid=0, nburst=ele_move_repeat_times, burst=ele_move_data_size,
                                            src_stride=ele_move_src_stride, dst_stride=ele_move_dst_stride)

                l1a_offset = 0
                l1b_offset = 0
                l1a_repeat_times = single_m_size * cur_k_size
                l1b_repeat_times = single_n_size * cur_k_size
                self.load_2d(tensor_a_l0a, tensor_a[l1a_offset], [0, l1a_repeat_times, 1, 0, False])

                self.tik_instance.mmad(tensor_c_l0c, tensor_a_l0a, tensor_b_l0b,
                                       single_m_size * self.block_num,
                                       single_k_size * self.block_num,
                                       single_n_size * self.block_num,
                                       0)

                cc_to_ub_dst_stride = 0
                cc_to_ub_src_stride = 0

                self.tik_instance.tensor_mov(tensor_c_ub, tensor_c_l0c,
                                             'm', 1, single_m_size * single_n_size,
                                             cc_to_ub_dst_stride, cc_to_ub_src_stride)

                ele_compute_repeat_times = single_m_size * single_n_size * 16 * 16 // self.repeat_once_size
                tail = max(ele_compute_repeat_times // 255, 0) * (ele_compute_repeat_times % 255)
                ele_compute_repeat_times = min(255, ele_compute_repeat_times)
                self.tik_instance.vmuls(self.repeat_once_size, tensor_c_ub, tensor_c_ub,
                                        mul_value,
                                        ele_compute_repeat_times,
                                        self.block_stride, self.block_stride,
                                        self.repeat_stride, self.repeat_stride)
                if tail != 0:
                    self.tik_instance.vmuls(self.repeat_once_size, tensor_c_ub[255 * self.repeat_once_size],
                                            tensor_c_ub[255 * self.repeat_once_size],
                                            mul_value,
                                            tail,
                                            self.block_stride, self.block_stride,
                                            self.repeat_stride, self.repeat_stride)

                self.tik_instance.vadd(self.repeat_once_size, tensor_c_ub, tensor_c_ub,
                                       elewise_data_ub2,
                                       ele_compute_repeat_times,
                                       self.block_stride, self.block_stride, self.block_stride,
                                       self.repeat_stride, self.repeat_stride, self.repeat_stride)

                if tail != 0:
                    self.tik_instance.vadd(self.repeat_once_size, tensor_c_ub[255 * self.repeat_once_size],
                                           tensor_c_ub[255 * self.repeat_once_size],
                                           elewise_data_ub2[255 * self.repeat_once_size],
                                           tail,
                                           self.block_stride, self.block_stride, self.block_stride,
                                           self.repeat_stride, self.repeat_stride, self.repeat_stride)
                self.softmax_compute(tensor_c_ub, single_m_size, single_n_size)

                # copy_ub_to_l1
                mid_offset = cur_m_idx * cur_m_size * self.block_num * self.block_num + \
                             m * single_m_size * self.block_num * self.block_num
                mid_data_lengh = single_m_size * self.block_num
                mid_src_stride = 0
                mid_dst_stride = (om_size - single_m_size) * self.block_num
                self.tik_instance.data_move(tensor_y[mid_offset], tensor_c_ub,
                                            sid=0, nburst=single_n_size, burst=mid_data_lengh,
                                            src_stride=mid_src_stride, dst_stride=mid_dst_stride)

    def vadds_transfor_cube(self, compute_buffers, compute_sizes):
        tensor_a_ub, tensor_b_ub, \
        tensor_a_l1, tensor_b_l1, \
        tensor_a_l0a, tensor_b_l0b, tensor_c_l0c, output_ub = compute_buffers
        single_m_size, single_k_size, single_n_size = compute_sizes
        vec_repeat_times = single_k_size * self.block_num * self.block_num // self.repeat_once_size
        self.tik_instance.vector_dup(self.repeat_once_size,
                                     tensor_b_ub[0],
                                     1, vec_repeat_times, 1, 8)
        tensor_a_burst = self.block_num
        tensor_a_nburst = single_k_size
        tensor_a_src_stride = (single_m_size - 1) * self.block_num
        tensor_a_dst_stride = 0
        for i in range(single_m_size):
            src_offset = i * self.block_num * self.block_num
            dst_offset = i * single_k_size * self.block_num * self.block_num
            self.tik_instance.data_move(tensor_a_l1[dst_offset],
                                        tensor_a_ub[src_offset],
                                        sid=0, nburst=tensor_a_nburst,
                                        burst=tensor_a_burst,
                                        src_stride=tensor_a_src_stride,
                                        dst_stride=tensor_a_dst_stride)
        if single_m_size != 1:
            self.tik_instance.data_move(tensor_a_l1[single_k_size * self.block_num * self.block_num],
                                        tensor_a_ub[self.block_num * self.block_num],
                                        sid=0, nburst=tensor_a_nburst,
                                        burst=tensor_a_burst,
                                        src_stride=tensor_a_src_stride,
                                        dst_stride=tensor_a_dst_stride)
        tensor_b_burst = single_n_size * single_k_size * self.block_num
        tensor_b_nburst = 1
        tensor_b_src_stride = 0
        tensor_b_dst_stride = 0
        self.tik_instance.data_move(tensor_b_l1, tensor_b_ub[0],
                                    sid=0, nburst=tensor_b_nburst,
                                    burst=tensor_b_burst,
                                    src_stride=tensor_b_src_stride,
                                    dst_stride=tensor_b_dst_stride)
        self.load_2d(tensor_a_l0a, tensor_a_l1[0], [0, single_m_size * single_k_size, 1, 0, False])
        self.load_2d(tensor_b_l0b, tensor_b_l1[0], [0, single_n_size * single_k_size, 1, 0, False])
        self.tik_instance.mmad(tensor_c_l0c, tensor_a_l0a, tensor_b_l0b,
                               single_m_size * self.block_num,
                               single_k_size * self.block_num,
                               single_n_size * self.block_num,
                               0)
        self.tik_instance.tensor_mov(tensor_b_ub, tensor_c_l0c,
                                     'm', 1, single_m_size * single_n_size,
                                     0, 0)

    def softmax_compute_last_part(self, softmax_last_part_buffers, softmax_last_part_size, softmax_last_part_idx):
        """
        """
        tensor_c_ub, ub_cast, reduce_ub, ub_mask, \
        tensor_a, tensor_b, tensor_a_l0a, tensor_b_l0b, tensor_c_l0c = softmax_last_part_buffers
        om_size, cur_m_size, cur_n_size, cur_m, cur_n = softmax_last_part_size
        input_keep_prob, ub_mask_fp16, block_idx, cur_b_idx, cur_om_idx, cur_m_idx, m = softmax_last_part_idx
        fp32_repeat_once_nums = 64
        max_repeat_times = 255
        repeat_times = cur_m * cur_n * self.block_num * self.block_num // fp32_repeat_once_nums
        insn_tail = max(repeat_times // 255, 0) * (repeat_times % 255)
        repeat_times = min(repeat_times, max_repeat_times)

        self.tik_instance.vconv(fp32_repeat_once_nums, "",
                                ub_cast[0], tensor_c_ub[0], repeat_times, 1, 1, 8, 4)
        self.tik_instance.vexp(fp32_repeat_once_nums,
                               ub_cast[0], ub_cast[0], repeat_times, 1, 1, 8, 8)
        self.tik_instance.vconv(fp32_repeat_once_nums, "",
                                tensor_c_ub[0], ub_cast[0], repeat_times, 1, 1, 4, 8)
        if insn_tail > 0:
            self.tik_instance.vconv(fp32_repeat_once_nums, "",
                                    ub_cast[repeat_times * fp32_repeat_once_nums],
                                    tensor_c_ub[repeat_times * fp32_repeat_once_nums],
                                    insn_tail, 1, 1, 8, 4)
            self.tik_instance.vexp(fp32_repeat_once_nums,
                                   ub_cast[repeat_times * fp32_repeat_once_nums],
                                   ub_cast[repeat_times * fp32_repeat_once_nums],
                                   insn_tail, 1, 1, 8, 8)
            self.tik_instance.vconv(fp32_repeat_once_nums, "",
                                    tensor_c_ub[repeat_times * fp32_repeat_once_nums],
                                    ub_cast[repeat_times * fp32_repeat_once_nums],
                                    insn_tail, 1, 1, 4, 8)
        vmax_range = cur_n
        src_tensor = ub_cast
        ub_broadcast_fp16 = ub_cast.reinterpret_cast_to("float16")
        if self.first_m_dim != self.trans_cube_target:
            while (vmax_range > 1):
                if vmax_range % 2 == 0:
                    repeat_time = cur_m * vmax_range * self.block_num * self.block_num // fp32_repeat_once_nums // 2
                    src_offset = cur_m * vmax_range * self.block_num * self.block_num // 2
                    self.tik_instance.vadd(fp32_repeat_once_nums, src_tensor[0], src_tensor[0], src_tensor[src_offset],
                                           repeat_time,
                                           self.block_stride, self.block_stride, self.block_stride,
                                           self.repeat_stride, self.repeat_stride, self.repeat_stride)
                    vmax_range = vmax_range // 2
                else:
                    repeat_time = cur_m * self.block_num * self.block_num // fp32_repeat_once_nums
                    src_offset = (vmax_range - 1) * cur_m * self.block_num * self.block_num
                    self.tik_instance.vadd(fp32_repeat_once_nums, src_tensor[0], src_tensor[0], src_tensor[src_offset],
                                           repeat_time,
                                           self.block_stride, self.block_stride, self.block_stride,
                                           self.repeat_stride, self.repeat_stride, self.repeat_stride)
                    vmax_range = vmax_range - 1

            repeat_time = cur_m * self.block_num
            self.tik_instance.vcadd(self.block_num, reduce_ub[0], src_tensor[0], repeat_time,
                                    self.block_stride, self.block_stride, 2)
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
                self.tik_instance.vor(self.block_num, ub_broadcast[dst_offset],
                                      ub_reduceadd_int16[src_offset],
                                      ub_broadcast[cur_m * cur_n * self.block_num * self.block_num],
                                      self.block_num,
                                      1, 1, 0, 1, 0, 0)
                self.tik_instance.vtranspose(ub_broadcast[dst_offset], ub_broadcast[dst_offset])
        else:
            trans_cube_buffers = [tensor_c_ub, ub_broadcast_fp16, tensor_a, tensor_b,
                                  tensor_a_l0a, tensor_b_l0b, tensor_c_l0c, reduce_ub]
            trans_cube_size = [cur_m, cur_n, 1]
            self.vadds_transfor_cube(trans_cube_buffers, trans_cube_size)
            rec_repeat_times = cur_m * self.block_num * self.block_num // self.repeat_once_size
            self.tik_instance.vrec(self.repeat_once_size, ub_broadcast_fp16[0], ub_broadcast_fp16[0],
                                   rec_repeat_times, 1, 1, 8, 8)

        sub_range = cur_m * self.block_num * self.block_num // self.repeat_once_size
        with self.tik_instance.for_range(0, sub_range) as idx:
            self.tik_instance.vmul(self.repeat_once_size, tensor_c_ub[idx * self.repeat_once_size],
                                   tensor_c_ub[idx * self.repeat_once_size],
                                   ub_broadcast_fp16[idx * self.repeat_once_size],
                                   cur_n, 1, 1, 1, cur_m * self.block_num, cur_m * self.block_num, 0)
        if self.model_type == Constant.TRAINGING:
            ub_broadcast_fp16_mid_offset = cur_n * cur_m * self.block_num * self.block_num
            vconv_repeat_times = cur_n * cur_m * self.block_num * self.block_num // self.repeat_once_size
            if self.drop_v3 is not True:
                self.tik_instance.vector_dup(self.repeat_once_size, ub_broadcast_fp16[ub_broadcast_fp16_mid_offset],
                                             self.tik_instance.Scalar(init_value=0, dtype="float16"),
                                             vconv_repeat_times, 1, 8)
                ub_mask_int64 = ub_mask.reinterpret_cast_to("uint64")
                droptimes = vconv_repeat_times
                repeat_times_dropout_vadds = 1
                dst_repeat_stride_dropout_vadds = 8
                src_repeat_stride_dropout_vadds = 8
                scalar_heigher = self.tik_instance.Scalar(dtype = "uint64")
                scalar_lower = self.tik_instance.Scalar(dtype = "uint64")
                with self.tik_instance.for_range(0, droptimes) as dropidx:
                    scalar_heigher.set_as(ub_mask_int64[2*dropidx + 1])
                    scalar_lower.set_as(ub_mask_int64[2*dropidx])
                    with self.tik_instance.if_scope (tik.any(scalar_heigher != 0, scalar_lower != 0)) :
                        self.tik_instance.vadds([scalar_heigher, scalar_lower],
                                                ub_broadcast_fp16[ub_broadcast_fp16_mid_offset +
                                                                  self.repeat_once_size * dropidx],
                                                ub_broadcast_fp16[ub_broadcast_fp16_mid_offset +
                                                                  self.repeat_once_size * dropidx],
                                                self.tik_instance.Scalar(init_value=1, dtype="float16"),
                                                repeat_times_dropout_vadds,
                                                1, 1,
                                                dst_repeat_stride_dropout_vadds, src_repeat_stride_dropout_vadds,
                                                0)
            else:
                self.tik_instance.vconv(self.repeat_once_size, "",
                                    ub_broadcast_fp16[ub_broadcast_fp16_mid_offset],
                                    ub_mask[0],
                                    vconv_repeat_times, 1, 1, 8, 4)
            self.tik_instance.vmuls(self.repeat_once_size,
                                    ub_broadcast_fp16[ub_broadcast_fp16_mid_offset],
                                    ub_broadcast_fp16[ub_broadcast_fp16_mid_offset],
                                    self.tik_instance.Scalar(init_value = 1 / input_keep_prob, dtype="float16"),
                                    vconv_repeat_times, 1, 1, 8, 8)
            self.tik_instance.vmul(self.repeat_once_size, ub_broadcast_fp16[ub_broadcast_fp16_mid_offset],
                                   ub_broadcast_fp16[ub_broadcast_fp16_mid_offset], tensor_c_ub[0],
                                   vconv_repeat_times, 1, 1, 1, 8, 8, 8)
            trans_nz_zz_dst_repeat_stride = self.block_num
            trans_nz_zz_src_repeat_stride = cur_m * self.block_num
            for i in range(cur_m):
                self.tik_instance.vadds(self.repeat_once_size,
                                        ub_broadcast_fp16[i * cur_n * self.block_num * self.block_num],
                                        ub_broadcast_fp16[ub_broadcast_fp16_mid_offset +
                                                          i * self.block_num * self.block_num],
                                        0,
                                        cur_n,
                                        1, 1,
                                        trans_nz_zz_dst_repeat_stride, trans_nz_zz_src_repeat_stride,
                                        0)
                self.tik_instance.vadds(self.repeat_once_size,
                                        ub_broadcast_fp16[i * cur_n * self.block_num * self.block_num +
                                                          self.repeat_once_size],
                                        ub_broadcast_fp16[ub_broadcast_fp16_mid_offset +
                                                          i * self.block_num * self.block_num +
                                                          self.repeat_once_size],
                                        0,
                                        cur_n,
                                        1, 1,
                                        trans_nz_zz_dst_repeat_stride, trans_nz_zz_src_repeat_stride,
                                        0)
        else:
            trans_nz_zz_dst_repeat_stride = self.block_num
            trans_nz_zz_src_repeat_stride = cur_m * self.block_num
            for i in range(cur_m):
                self.tik_instance.vadds(self.repeat_once_size,
                                        ub_broadcast_fp16[i * cur_n * self.block_num * self.block_num],
                                        tensor_c_ub[i * self.block_num * self.block_num],
                                        0,
                                        cur_n,
                                        1, 1,
                                        trans_nz_zz_dst_repeat_stride, trans_nz_zz_src_repeat_stride,
                                        0)
                self.tik_instance.vadds(self.repeat_once_size,
                                        ub_broadcast_fp16[i * cur_n * self.block_num * self.block_num +
                                                          self.repeat_once_size],
                                        tensor_c_ub[i * self.block_num * self.block_num +
                                                    self.repeat_once_size],
                                        0,
                                        cur_n,
                                        1, 1,
                                        trans_nz_zz_dst_repeat_stride, trans_nz_zz_src_repeat_stride,
                                        0)

        # dma copy_ub_to_gm
        mid_gm_offset = block_idx * self.batch_outer_num * self.first_m_dim * \
                        self.first_n_dim * self.block_num * self.block_num + \
                        cur_b_idx * self.first_n_dim * self.first_m_dim * self.block_num * self.block_num + \
                        cur_om_idx * om_size * self.block_num * self.block_num + \
                        cur_m_idx * cur_m_size * self.block_num * self.block_num + \
                        m * cur_m * self.block_num * self.block_num

        mid_data_lengh = cur_m * self.block_num
        mid_src_stride = 0
        mid_dst_stride = (self.first_m_dim - cur_m) * self.block_num
        if self.model_type == Constant.TRAINGING:
            self.tik_instance.data_move(self.softmax_output_gm[mid_gm_offset], tensor_c_ub,
                                        sid=0, nburst=cur_n, burst=mid_data_lengh,
                                        src_stride=mid_src_stride, dst_stride=mid_dst_stride)

        return ub_broadcast_fp16

    def softmax_compute(self, tensor_input, cur_m, cur_n):
        # do softmax compute ori 32 32 16 16] cur 32 2 16 16
        softmax_ub = self.tik_instance.Tensor(self.matmul_dtype, [cur_n, cur_m, self.block_num, self.block_num],
                                              name="softmax_ub", scope=self.tik.scope_ubuf)
        reduce_ub = self.tik_instance.Tensor(self.matmul_dtype, [cur_m * self.block_num],
                                             name="reduce_ub", scope=self.tik.scope_ubuf)
        ub_broadcast = self.tik_instance.Tensor("uint16", (self.first_n_dim * self.block_num,),
                                                name="ub_broadcast", scope=self.tik.scope_ubuf)
        vmax_range = cur_n
        src_tensor = tensor_input

        while (vmax_range > 1):
            if vmax_range != cur_n:
                src_tensor = softmax_ub
            repeat_time = vmax_range // 2 * cur_m * self.block_num * self.block_num // self.repeat_once_size
            src_offset = vmax_range // 2 * cur_m * self.block_num * self.block_num
            self.tik_instance.vmax(self.repeat_once_size, softmax_ub[0], src_tensor[0], src_tensor[src_offset],
                                   repeat_time, self.block_stride, self.block_stride, self.block_stride,
                                   self.repeat_stride, self.repeat_stride, self.repeat_stride)
            vmax_range = vmax_range // 2
        
        if cur_n % 2 != 0:
            repeat_time = cur_m * self.block_num * self.block_num // self.repeat_once_size
            src_offset = (cur_n - 1) * cur_m * self.block_num * self.block_num
            self.tik_instance.vmax(self.repeat_once_size, softmax_ub[0], softmax_ub[0], tensor_input[src_offset],
                                   repeat_time, self.block_stride, self.block_stride, self.block_stride,
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
            dst_offset = cur_fz * self.block_num * self.block_num
            src_offset = cur_fz * self.block_num
            self.tik_instance.vor(self.block_num, ub_dup[dst_offset],
                                  ub_reducemax_int16[src_offset], ub_dup[cur_m * self.block_num * self.block_num],
                                  self.block_num,
                                  1, 1, 0, 1, 0, 0)
            self.tik_instance.vtranspose(ub_dup[dst_offset], ub_dup[dst_offset])

        sub_range = cur_m * self.block_num * self.block_num // self.repeat_once_size
        sub_dst_stride = cur_m * self.block_num
        sub_src_stride = cur_m * self.block_num
        with self.tik_instance.for_range(0, sub_range) as idx:
            self.tik_instance.vsub(self.repeat_once_size, tensor_input[idx * self.repeat_once_size],
                                   tensor_input[idx * self.repeat_once_size],
                                   softmax_ub[idx * self.repeat_once_size],
                                   cur_n, 1, 1, 1, sub_dst_stride, sub_src_stride, 0)

        # part_1_end?
        return tensor_input

    def get_first_compute_one_core_mkn(self, m, m_range, k, n):
        return m // m_range, k, n

    def tiling_m_axis(self, m_size):
        outer_m_range_value = 1
        if m_size % 4 != 0:
            inner_m_range_value = self.x1_shape[3]
        elif m_size > 16:
            inner_m_range_value = self.x1_shape[3] // 2
        elif m_size >= 12:
            inner_m_range_value = 4
        else:
            inner_m_range_value = 2

        return outer_m_range_value, inner_m_range_value

    def get_second_compute_one_core_mkn(self, m, m_range, k, n):
        return m // m_range, k, n

    def first_bmm_move_tensor_b_from_gm_to_l1(self, first_bmm_tensor_b, block_idx, batch_range_value, cur_b_idx):
        first_bmm_tensor_b_offset = block_idx * batch_range_value * self.first_n_dim * self.first_k_dim * \
                                    self.block_num * self.block_num + \
                                    cur_b_idx * self.first_n_dim * self.first_k_dim * self.block_num * self.block_num
        first_bmm_tensor_b_burst = self.first_n_dim * self.block_num
        first_bmm_tensor_b_repeat_times = self.first_k_dim
        first_bmm_tensor_b_src_stride = 0
        first_bmm_tensor_b_dst_stride = 0
        self.tik_instance.data_move(first_bmm_tensor_b, self.x2_gm[first_bmm_tensor_b_offset],
                                    sid=0, nburst=first_bmm_tensor_b_repeat_times,
                                    burst=first_bmm_tensor_b_burst,
                                    src_stride=first_bmm_tensor_b_src_stride,
                                    dst_stride=first_bmm_tensor_b_dst_stride)
        return first_bmm_tensor_b

    def second_bmm_move_tensor_b_from_gm_to_l1(self, second_bmm_tensor_b, block_idx, cur_b_idx):
        for dma_idx in range(self.second_n_dim):
            second_bmm_tensor_b_src_offset = block_idx * self.batch_outer_num * self.second_n_dim * \
                                             self.second_k_dim * self.block_num * self.block_num + \
                                             cur_b_idx * self.second_n_dim * self.second_k_dim * \
                                             self.block_num * self.block_num + \
                                             dma_idx * self.second_k_dim * self.block_num * self.block_num
            second_bmm_tensor_b_dst_offset = dma_idx * self.block_num * self.block_num
            second_bmm_tensor_b_burst = self.block_num
            second_bmm_tensor_b_repeat_times = self.second_k_dim
            second_bmm_tensor_b_src_stride = 0
            second_bmm_tensor_b_dst_stride = (self.second_n_dim - 1) * self.block_num
            self.tik_instance.data_move(second_bmm_tensor_b[second_bmm_tensor_b_dst_offset],
                                        self.x3_gm[second_bmm_tensor_b_src_offset],
                                        sid=0, nburst=second_bmm_tensor_b_repeat_times,
                                        burst=second_bmm_tensor_b_burst,
                                        src_stride=second_bmm_tensor_b_src_stride,
                                        dst_stride=second_bmm_tensor_b_dst_stride)
        return second_bmm_tensor_b

    def apply_buffer_for_tensor_c_l1(self, outer_m_range_value):
        """

        """
        first_bmm_tensor_c = self.tik_instance.Tensor(self.vector_dtype,
                                                      [self.first_n_dim, self.first_m_dim // outer_m_range_value,
                                                       self.block_num, self.block_num],
                                                      name="first_bmm_tensor_c", scope=self.tik.scope_cbuf)

        return first_bmm_tensor_c

    def apply_buffer_for_tensor_b_l0_and_move_data_in(self, first_bmm_tensor_b):
        """

        """
        first_bmm_tensor_b_l0b = self.tik_instance.Tensor(self.matmul_dtype,
                                                          [self.first_k_dim, self.first_n_dim,
                                                           self.block_num, self.block_num],
                                                          name="first_bmm_tensor_b_l0b",
                                                          scope=self.tik.scope_cb)
        self.load_2d(first_bmm_tensor_b_l0b, first_bmm_tensor_b[0], \
                    [0, self.first_k_dim * self.first_n_dim, 1, 0, False])
        return first_bmm_tensor_b_l0b

    def first_bmm_compute_for_outer_m_once(self, preload_buffers, range_idxs, outer_m_range_once_m_size,
                                           inner_m_range_value, mul_value):
        """
        compute first bmm once for outer m range.
        """
        first_bmm_tensor_b, first_bmm_tensor_c, first_bmm_tensor_b_l0b = preload_buffers
        block_idx, cur_b_idx, cur_om_idx = range_idxs
        with self.tik_instance.for_range(0, inner_m_range_value // self.double_factor) as inner_m_idx:
            inner_m_size, inner_k_size, inner_n_size = self.get_first_compute_one_core_mkn(outer_m_range_once_m_size,
                                                                                            inner_m_range_value,
                                                                                            self.first_k_dim,
                                                                                            self.first_n_dim)
            l1a_shape = [inner_k_size, inner_m_size, self.block_num, self.block_num]
            l1b_shape = [inner_k_size, inner_n_size, self.block_num, self.block_num]
            first_bmm_tensor_a = self.tik_instance.Tensor(self.matmul_dtype, l1a_shape,
                                                          name="first_bmm_tensor_a",
                                                          scope=self.tik.scope_cbuf)
            first_bmm_tensor_a_l0a = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * inner_k_size *
                                                                                  self.block_num * self.block_num],
                                                              name="first_bmm_tensor_a_l0a", scope=self.tik.scope_ca)

            first_bmm_tensor_c_l0c = self.tik_instance.Tensor("float32", [inner_m_size * inner_n_size *
                                                                          self.block_num * self.block_num],
                                                              name="first_bmm_tensor_c_l0c", scope=self.tik.scope_cc)

            first_bmm_tensor_c_ub = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * inner_n_size *
                                                                                 self.block_num * self.block_num],
                                                             name="first_bmm_tensor_c_ub",
                                                             scope=self.tik.scope_ubuf)

            elewise_add_data_ub = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * self.first_n_dim *
                                                                               self.block_num * self.block_num],
                                                             name="elewise_add_data_ub",
                                                             scope=self.tik.scope_ubuf)
            first_bmm_compute_buffers = [first_bmm_tensor_a, first_bmm_tensor_b, first_bmm_tensor_c,
                                         first_bmm_tensor_a_l0a, first_bmm_tensor_b_l0b, first_bmm_tensor_c_l0c,
                                         first_bmm_tensor_c_ub, mul_value, elewise_add_data_ub]
            first_bmm_compute_idx = [block_idx, cur_b_idx, cur_om_idx, 2 * inner_m_idx]
            first_bmm_compute_each_layer_size = [outer_m_range_once_m_size, inner_m_size, inner_k_size, inner_n_size]
            self.mat_mul_compute(first_bmm_compute_buffers, first_bmm_compute_idx, first_bmm_compute_each_layer_size)

            first_bmm_tensor_a_db = self.tik_instance.Tensor(self.matmul_dtype, l1a_shape,
                                                             name="first_bmm_tensor_a_db",
                                                             scope=self.tik.scope_cbuf)

            first_bmm_tensor_a_l0a_db = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * inner_k_size *
                                                                                     self.block_num * self.block_num],
                                                                 name="first_bmm_tensor_a_l0a_db",
                                                                 scope=self.tik.scope_ca, start_addr=32768)

            first_bmm_tensor_c_l0c_db = self.tik_instance.Tensor("float32", [inner_m_size * inner_n_size *
                                                                          self.block_num * self.block_num],
                                                                 name="first_bmm_tensor_c_l0c_db",
                                                                 scope=self.tik.scope_cc, start_addr=32768)

            first_bmm_tensor_c_ub_db = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * inner_n_size *
                                                                                    self.block_num * self.block_num],
                                                                name="first_bmm_tensor_c_ub_db",
                                                                scope=self.tik.scope_ubuf)

            elewise_add_data_ub_db = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * inner_n_size *
                                                                               self.block_num * self.block_num],
                                                              name="elewise_add_data_ub_db",
                                                              scope=self.tik.scope_ubuf)
            first_bmm_compute_db_buffers = [first_bmm_tensor_a_db, first_bmm_tensor_b, first_bmm_tensor_c,
                                            first_bmm_tensor_a_l0a_db, first_bmm_tensor_b_l0b,
                                            first_bmm_tensor_c_l0c_db,
                                            first_bmm_tensor_c_ub_db, mul_value, elewise_add_data_ub_db]
            first_bmm_compute_idx = [block_idx, cur_b_idx, cur_om_idx, 2 * inner_m_idx + 1]
            self.mat_mul_compute(first_bmm_compute_db_buffers, first_bmm_compute_idx, first_bmm_compute_each_layer_size)
        
        with self.tik_instance.if_scope(inner_m_range_value % self.double_factor != 0):
            inner_m_size, inner_k_size, inner_n_size = outer_m_range_once_m_size // inner_m_range_value, \
                                                       self.first_k_dim, self.first_n_dim
            l1a_shape = [inner_k_size, inner_m_size, self.block_num, self.block_num]
            first_bmm_tensor_a = self.tik_instance.Tensor(self.matmul_dtype, l1a_shape, name="first_bmm_tensor_a",
                                                          scope=self.tik.scope_cbuf)
            first_bmm_tensor_a_l0a = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * inner_k_size *
                                                                                  self.block_num * self.block_num],
                                                              name="first_bmm_tensor_a_l0a", scope=self.tik.scope_ca)
            first_bmm_tensor_c_l0c = self.tik_instance.Tensor("float32", [inner_m_size * inner_n_size *
                                                                          self.block_num * self.block_num],
                                                              name="first_bmm_tensor_c_l0c", scope=self.tik.scope_cc)
            first_bmm_tensor_c_ub = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * inner_n_size *
                                                                                 self.block_num * self.block_num],
                                                             name="first_bmm_tensor_c_ub", scope=self.tik.scope_ubuf)
            elewise_add_data_ub = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * self.first_n_dim *
                                                                               self.block_num * self.block_num],
                                                           name="elewise_add_data_ub", scope=self.tik.scope_ubuf)
            first_bmm_compute_buffers = [first_bmm_tensor_a, first_bmm_tensor_b, first_bmm_tensor_c,
                                         first_bmm_tensor_a_l0a, first_bmm_tensor_b_l0b, first_bmm_tensor_c_l0c,
                                         first_bmm_tensor_c_ub, mul_value, elewise_add_data_ub]
            first_bmm_compute_idx = [block_idx, cur_b_idx, cur_om_idx, inner_m_range_value - 1]
            first_bmm_compute_each_layer_size = [outer_m_range_once_m_size, inner_m_size, inner_k_size, inner_n_size]
            self.mat_mul_compute(first_bmm_compute_buffers, first_bmm_compute_idx, first_bmm_compute_each_layer_size)

    def second_bmm_compute_for_outer_m_once(self, preload_buffers, range_idxs, outer_m_range_once_m_size,
                                            inner_m_range_value):
        """
        second bmm compute for outer m once.
        """
        second_bmm_tensor_b, second_bmm_tensor_b_l0b, second_bmm_tensor_c = preload_buffers
        block_idx, cur_b_idx, cur_om_idx = range_idxs
        if self.first_m_dim != self.trans_cube_target:
            self.load_2d(second_bmm_tensor_b_l0b, second_bmm_tensor_b[0], \
                        [0, self.first_n_dim * self.second_n_dim, 1, 0, True])
        inner_m_size, inner_k_size, inner_n_size = self.get_second_compute_one_core_mkn(outer_m_range_once_m_size,
                                                                                        inner_m_range_value,
                                                                                        self.second_k_dim,
                                                                                        self.second_n_dim)
        l1a_shape = [inner_k_size, inner_m_size, self.block_num, self.block_num]
        l1b_shape = [inner_k_size, inner_n_size, self.block_num, self.block_num]
        ub_start_addr = 0
        second_bmm_tensor_a = self.tik_instance.Tensor(self.matmul_dtype, l1a_shape,
                                                       name="second_bmm_tensor_a",
                                                       scope=self.tik.scope_cbuf)
        second_bmm_tensor_a_l0a = self.tik_instance.Tensor(self.matmul_dtype, l1a_shape,
                                                           name="second_bmm_tensor_a_l0a",
                                                           scope=self.tik.scope_ca)

        second_bmm_tensor_c_l0c = self.tik_instance.Tensor("float32", [inner_m_size * inner_n_size *
                                                                       self.block_num * self.block_num],
                                                           name="second_bmm_tensor_c_l0c",
                                                           scope=self.tik.scope_cc)

        second_bmm_tensor_c_ub = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * inner_k_size *
                                                                              self.block_num * self.block_num],
                                                          name="second_bmm_tensor_c_ub",
                                                          scope=self.tik.scope_ubuf, start_addr=0)
        ub_start_addr = ub_start_addr + 32768
        second_bmm_mask_ub = self.tik_instance.Tensor("uint8", [inner_m_size * inner_k_size *
                                                                self.block_num * self.block_num],
                                                      name="second_bmm_mask_ub",
                                                      scope=self.tik.scope_ubuf, start_addr=ub_start_addr)

        ub_start_addr = ub_start_addr + inner_m_size * inner_k_size * self.block_num * self.block_num
        second_bmm_mask_cast_ub = self.tik_instance.Tensor("float32", [inner_m_size * inner_k_size *
                                                                       self.block_num * self.block_num],
                                                           name="second_bmm_mask_cast_ub",
                                                           scope=self.tik.scope_ubuf, start_addr=ub_start_addr)

        ub_start_addr = ub_start_addr + inner_m_size * inner_k_size * self.block_num * self.block_num * 4
        second_bmm_softmax_reduce_ub = self.tik_instance.Tensor("float32", [inner_m_size * self.block_num],
                                                                name="second_bmm_softmax_reduce_ub",
                                                                scope=self.tik.scope_ubuf, start_addr=ub_start_addr)
        trans_cube_tensor_b_l0b = second_bmm_tensor_b_l0b[0]
        trans_cube_tensor_b_l1 = None
        if self.first_m_dim == self.trans_cube_target:
            trans_cube_tensor_b_l1 = self.tik_instance.Tensor(self.matmul_dtype,
                                                              [inner_k_size * self.block_num + \
                                                               self.block_num],
                                                              name="trans_cube_tensor_b_l1",
                                                              scope=self.tik.scope_cbuf)
        # db part tensor
        second_bmm_tensor_a_db = self.tik_instance.Tensor(self.matmul_dtype, l1a_shape,
                                                          name="second_bmm_tensor_a_db",
                                                          scope=self.tik.scope_cbuf,
                                                          start_addr=524288)
        second_bmm_tensor_a_l0a_db = self.tik_instance.Tensor(self.matmul_dtype, l1a_shape,
                                                              name="second_bmm_tensor_a_l0a_db",
                                                              scope=self.tik.scope_ca, start_addr=32768)

        second_bmm_tensor_c_l0c_db = self.tik_instance.Tensor("float32", [inner_m_size * inner_n_size *
                                                                          self.block_num * self.block_num],
                                                              name="second_bmm_tensor_c_l0c_db",
                                                              scope=self.tik.scope_cc, start_addr=32768)
        ub_start_addr = 131072

        second_bmm_tensor_c_ub_db = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * inner_k_size *
                                                                              self.block_num * self.block_num],
                                                          name="second_bmm_tensor_c_ub_db",
                                                          scope=self.tik.scope_ubuf, start_addr=ub_start_addr)
        ub_start_addr = ub_start_addr + inner_m_size * inner_k_size * self.block_num * self.block_num * 2
        second_bmm_mask_ub_db = self.tik_instance.Tensor("uint8", [inner_m_size * inner_k_size *
                                                                   self.block_num * self.block_num],
                                                         name="second_bmm_mask_ub_db",
                                                         scope=self.tik.scope_ubuf, start_addr=ub_start_addr)

        ub_start_addr = ub_start_addr + inner_m_size * inner_k_size * self.block_num * self.block_num
        second_bmm_mask_cast_ub_db = self.tik_instance.Tensor("float32", [inner_m_size * inner_k_size *
                                                                       self.block_num * self.block_num],
                                                              name="second_bmm_mask_cast_ub_db",
                                                              scope=self.tik.scope_ubuf, start_addr=ub_start_addr)

        ub_start_addr = ub_start_addr + inner_m_size * inner_k_size * self.block_num * self.block_num * 4
        second_bmm_softmax_reduce_ub_db = self.tik_instance.Tensor("float32", [inner_m_size * self.block_num],
                                                                   name="second_bmm_softmax_reduce_ub_db",
                                                                   scope=self.tik.scope_ubuf, start_addr=ub_start_addr)
        trans_cube_tensor_b_l0b_db = second_bmm_tensor_b_l0b[self.first_k_dim * self.first_n_dim * self.block_num * \
                                                             self.block_num // 2]
        trans_cube_tensor_b_l1_db = None
        if self.first_m_dim == self.trans_cube_target:
            trans_cube_tensor_b_l1_db = self.tik_instance.Tensor(self.matmul_dtype,
                                                                 [inner_k_size * self.block_num + \
                                                                  self.block_num],
                                                                 name="trans_cube_tensor_b_l1_db",
                                                                 scope=self.tik.scope_cbuf)
        second_bmm_compute_buffers = [second_bmm_tensor_a, second_bmm_tensor_b_l0b, second_bmm_tensor_c,
                                      second_bmm_tensor_a_l0a, second_bmm_tensor_b_l0b, second_bmm_tensor_c_l0c,
                                      second_bmm_tensor_c_ub, second_bmm_mask_ub, second_bmm_mask_cast_ub,
                                      second_bmm_softmax_reduce_ub,
                                      trans_cube_tensor_b_l1, trans_cube_tensor_b_l0b, second_bmm_tensor_b]

        second_bmm_compute_db_buffers = [second_bmm_tensor_a_db, second_bmm_tensor_b_l0b,
                                         second_bmm_tensor_a_l0a_db, second_bmm_tensor_c_l0c_db,
                                         second_bmm_tensor_c_ub_db, second_bmm_mask_ub_db, second_bmm_mask_cast_ub_db,
                                         second_bmm_softmax_reduce_ub_db,
                                         trans_cube_tensor_b_l1_db, trans_cube_tensor_b_l0b_db]
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
                                              second_bmm_softmax_reduce_ub,
                                              trans_cube_tensor_b_l1, trans_cube_tensor_b_l0b, second_bmm_tensor_b]

                second_bmm_compute_db_buffers = [second_bmm_tensor_a_db, second_bmm_tensor_b,
                                                 second_bmm_tensor_a_l0a_db, second_bmm_tensor_c_l0c_db,
                                                 second_bmm_tensor_c_ub_db, second_bmm_mask_ub_db,
                                                 second_bmm_mask_cast_ub_db,
                                                 second_bmm_softmax_reduce_ub_db,
                                                 trans_cube_tensor_b_l1_db, trans_cube_tensor_b_l0b_db]
            else:
                second_bmm_compute_buffers = [second_bmm_tensor_a_db, second_bmm_tensor_b, second_bmm_tensor_c,
                                              second_bmm_tensor_a_l0a_db, second_bmm_tensor_b_l0b,
                                              second_bmm_tensor_c_l0c_db,
                                              second_bmm_tensor_c_ub_db, second_bmm_mask_ub_db,
                                              second_bmm_mask_cast_ub_db,
                                              second_bmm_softmax_reduce_ub_db,
                                              trans_cube_tensor_b_l1_db, trans_cube_tensor_b_l0b_db,
                                              second_bmm_tensor_b]

                second_bmm_compute_db_buffers = [second_bmm_tensor_a, second_bmm_tensor_b,
                                                 second_bmm_tensor_a_l0a, second_bmm_tensor_c_l0c,
                                                 second_bmm_tensor_c_ub, second_bmm_mask_ub,
                                                 second_bmm_mask_cast_ub,
                                                 second_bmm_softmax_reduce_ub,
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
                                             trans_cube_tensor_b_l1, trans_cube_tensor_b_l0b]
        second_bmm_compute_idxs = [block_idx, cur_b_idx, cur_om_idx, unroll_range]
        self.mat_mul_second_compute_last(second_bmm_compute_buffers, second_bmm_compute_db_buffers,
                                         second_bmm_compute_idxs, second_bmm_compute_each_layer_size,
                                         self.input_keep_prob)

    def compute_one_core(self, block_idx, batch_range_value, batch_range_value_last, used_core_num):
        """
        """
        outer_m_range_value, inner_m_range_value = self.tiling_m_axis(self.first_m_dim)
        self.batch_outer_num = batch_range_value
        outer_m_range_once_m_size = self.first_m_dim // outer_m_range_value
        mul_value = self.tik_instance.Scalar("float16", "mul_value", init_value=-1)
        mul_x_ub = self.tik_instance.Tensor(self.matmul_dtype,
                                            [self.block_num],
                                            name="mul_x_ub", scope=self.tik.scope_ubuf)
        self.tik_instance.data_move(mul_x_ub, self.mul_gm[0],
                                    sid=0, nburst=1, burst=1,
                                    src_stride=0, dst_stride=0)
        mul_value.set_as(mul_x_ub[0])

        first_bmm_tensor_b = self.tik_instance.Tensor(self.matmul_dtype, [self.first_k_dim, self.first_n_dim,
                                                                          self.block_num, self.block_num],
                                                      name="first_bmm_tensor_b", scope=self.tik.scope_cbuf)
        second_bmm_tensor_b = self.tik_instance.Tensor(self.matmul_dtype, [self.second_n_dim, self.second_k_dim,
                                                                           self.block_num, self.block_num],
                                                       name="second_bmm_tensor_b", scope=self.tik.scope_cbuf)
        self.first_bmm_move_tensor_b_from_gm_to_l1(first_bmm_tensor_b, block_idx, batch_range_value, 0)
        self.second_bmm_move_tensor_b_from_gm_to_l1(second_bmm_tensor_b, block_idx, 0)

        batch_ranges = self.tik_instance.Scalar("int32", "batch_ranges", init_value=0)
        with self.tik_instance.if_scope(block_idx < used_core_num - 1):
            batch_ranges.set_as(batch_range_value)
        with self.tik_instance.else_scope():
            batch_ranges.set_as(batch_range_value_last)
        with self.tik_instance.for_range(0, batch_ranges) as cur_b_idx:
            first_bmm_tensor_c = self.apply_buffer_for_tensor_c_l1(outer_m_range_value)
            first_bmm_tensor_b_l0b = self.apply_buffer_for_tensor_b_l0_and_move_data_in(first_bmm_tensor_b)
            with self.tik_instance.for_range(0, outer_m_range_value) as cur_om_idx:
                first_preload_buffers = [first_bmm_tensor_b, first_bmm_tensor_c, first_bmm_tensor_b_l0b]
                range_idxs = [block_idx, cur_b_idx, cur_om_idx]
                self.first_bmm_compute_for_outer_m_once(first_preload_buffers, range_idxs,
                                                        outer_m_range_once_m_size, inner_m_range_value, mul_value)

            with self.tik_instance.for_range(0, outer_m_range_value) as cur_om_idx:
                second_bmm_tensor_b_l0b = first_bmm_tensor_b_l0b
                second_bmm_tensor_c = first_bmm_tensor_c
                range_idxs = [block_idx, cur_b_idx, cur_om_idx]
                second_bmm_preload_buffers = [second_bmm_tensor_b, second_bmm_tensor_b_l0b, second_bmm_tensor_c]
                self.second_bmm_compute_for_outer_m_once(second_bmm_preload_buffers, range_idxs,
                                                         outer_m_range_once_m_size, inner_m_range_value)
            self.first_bmm_move_tensor_b_from_gm_to_l1(first_bmm_tensor_b, block_idx, batch_range_value, cur_b_idx + 1)
            self.second_bmm_move_tensor_b_from_gm_to_l1(second_bmm_tensor_b, block_idx, cur_b_idx + 1)

    def compute_process(self):
        batch_range_value, batch_range_value_last, used_core_num = \
            self.get_eatch_core_batch(self.x1_shape[0], self.x1_shape[1], self.cur_op_core_num)
        with self.tik_instance.for_range(0, used_core_num, block_num=used_core_num) as block_idx:
            self.compute_one_core(block_idx, batch_range_value, batch_range_value_last, used_core_num)

        input_gm_list = [self.x1_gm, self.x2_gm, self.x3_gm, self.add_gm, self.mul_gm, self.drop_mask_gm]
        output_gm_list = [self.y_gm]
        if self.model_type == Constant.TRAINGING:
            output_gm_list = [self.y_gm, self.softmax_output_gm]
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=input_gm_list,
                                   outputs=output_gm_list, config={})


@register_operator("AttentionScore")
# 'pylint: disable=redefined-builtin
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            (para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_LIST_INT),
                            para_check.KERNEL_NAME)
def attention_score(query, key, value, padding_mask, scale, drop_mask,
                    attention_score_output, softmax_output,
                    keep_prob,
                    query_transpose=False, key_transpose=False,
                    bmm_score_transpose_a=False, bmm_score_transpose_b=False,
                    softmax_axes=-1,
                    kernel_name="attention_score"):
    """
    algorithm: mat_mul_softmax_dropout_matmul
    calculating distence mtr.

    Parameters
    ----------
    coord : dict. shape and dtype of input, means the all neighbour coords, only support float32
    type : dict. shape and dtype of input, means the all neighbour types, only support int32
    natoms: dict. shape and dtype of input, contains the nloc and nall value, only support int32
    box: dict. shape and dtype of input
    mesh: dict. shape and dtype of input, the input data contains the neighbour coords, only support int32.
    davg: dict. shape and dtype of input, only support float32.
    dstd: dict. shape and dtype of input, only support float32.
    descrpt: dict. shape and dtype of output, only support float32.
    descrpt_deriv: dict. shape and dtype of output, only support float32.
    rij: dict. shape and dtype of output, only support float32.
    nlist: dict. shape and dtype of output, only support int32.
    kernel_name : str cce kernel name, default value is real_div

    Returns
    -------
    None
    """
    op_init = MatMulSoftmax(query, key, scale, padding_mask, drop_mask, value, softmax_output, attention_score_output,
                            keep_prob, softmax_axes,
                            query_transpose, key_transpose, bmm_score_transpose_a, bmm_score_transpose_b, kernel_name)
    op_init.compute_process()
