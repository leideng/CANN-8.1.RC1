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
class AttentionScoreGrad:
    """
    AttentionScoreGrad class
    """
    # 'pylint: disable=unused-argument
    def __init__(self, attention_score, dx, query, key, value, scale, drop_mask,
                 value_dw, query_dx, key_dw, keep_prob, query_transpose,
                 key_transpose, value_transpose, dx_transpose, softmax_axes,
                 kernel_name):
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.tik = tik
        self.cur_op_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.matmul_dtype = "float16"
        self.vector_dtype = "float16"
        self.mask_dtype = "uint8"
        self.front_bmm_x1_shape = attention_score["shape"]
        self.x1_shape = dx["shape"]
        self.x2_shape = value["shape"]
        self.x3_shape = query["shape"]
        self.x4_shape = key["shape"]
        self.drop_shape = drop_mask["shape"]

        self.value_dw_shape = value_dw["shape"]
        self.query_dx_shape = query_dx["shape"]
        self.key_dw_shape = key_dw["shape"]
        self.block_num = 16
        self.drop_v3 = False
        if self.drop_shape[0] == self.x1_shape[0] or \
        self.drop_shape[0] == self.x1_shape[0] * self.x1_shape[1] * self.x1_shape[3] * \
        self.x1_shape[3] * self.block_num * self.block_num:
            self.drop_v3 = True

        self.front_bmm_m_dim = self.front_bmm_x1_shape[2]
        self.front_bmm_k_dim = self.front_bmm_x1_shape[3]

        self.first_m_dim = self.x1_shape[3]
        self.first_k_dim = self.x1_shape[2]
        self.first_n_dim = self.x2_shape[3]

        self.second_m_dim = self.x1_shape[3]
        self.second_k_dim = self.x2_shape[3]
        self.second_n_dim = self.x3_shape[2]

        self.repeat_once_size = 128
        self.block_stride = 1
        self.repeat_stride = 8

        self.batch_outer_num = 0
        self.m_num = self.first_m_dim
        self.input_keep_prob = keep_prob

        self.mul_x_shape = [self.block_num]
        self.double_factor = 2
        self.kernel_name = kernel_name
        self.init_gm()

    @staticmethod
    def get_first_single_time_mkn(m, k, n):
        return m, k, n

    @staticmethod
    def get_second_single_time_mkn(m, k, n):
        """
        compute M K N value for once time.
        """
        return m, k, n

    @staticmethod
    def get_first_compute_one_core_mkn(m, m_range, k, n):
        return m // m_range, k, n

    @staticmethod
    def get_second_compute_one_core_mkn(m, m_range, k, n):
        return m // m_range, k, n

    @staticmethod
    def get_eatch_core_batch(b, n, core):
        all_ranges = b * n
        batch_range_value = (all_ranges + core - 1) // core
        used_core_num = (all_ranges + batch_range_value - 1) // batch_range_value
        batch_range_value_last = all_ranges - batch_range_value * (used_core_num - 1)
        return batch_range_value, batch_range_value_last, used_core_num

    def init_gm(self):
        self.attention_score_gm = self.tik_instance.Tensor(self.matmul_dtype, self.front_bmm_x1_shape,
                                                           name="attention_score_gm", scope=self.tik.scope_gm)
        self.x1_gm = self.tik_instance.Tensor(self.matmul_dtype, self.x1_shape,
                                              name="x1_gm", scope=self.tik.scope_gm)
        self.x2_gm = self.tik_instance.Tensor(self.matmul_dtype, self.x2_shape,
                                              name="x2_gm", scope=self.tik.scope_gm)
        self.mul_gm = self.tik_instance.Tensor(self.matmul_dtype, self.mul_x_shape,
                                               name="mul_gm", scope=self.tik.scope_gm)
        self.drop_mask_gm = self.tik_instance.Tensor(self.mask_dtype, self.drop_shape,
                                                     name="drop_mask_gm", scope=self.tik.scope_gm)
        self.x3_gm = self.tik_instance.Tensor(self.matmul_dtype, self.x3_shape,
                                              name="x3_gm", scope=self.tik.scope_gm)
        self.x4_gm = self.tik_instance.Tensor(self.matmul_dtype, self.x4_shape,
                                              name="x4_gm", scope=self.tik.scope_gm)


        self.value_dw_gm = self.tik_instance.Tensor(self.matmul_dtype, self.value_dw_shape,
                                                    name="value_dw_gm", scope=self.tik.scope_gm)
        self.query_dx_gm = self.tik_instance.Tensor(self.matmul_dtype, self.query_dx_shape,
                                                    name="query_dx_gm", scope=self.tik.scope_gm)
        self.key_dw_gm = self.tik_instance.Tensor(self.matmul_dtype, self.key_dw_shape,
                                                  name="key_dw_gm", scope=self.tik.scope_gm)

    def do_mask_to_ub_for_attention_score(self, range_idx, m_range_size, ub_mask):
        block_idx, cur_b_idx, cur_om_idx, cur_m_idx = range_idx
        om_size, single_m_size = m_range_size
        if self.drop_v3:
            mask_gm_offset = block_idx * self.batch_outer_num * self.front_bmm_m_dim * \
                             self.front_bmm_k_dim * self.block_num * self.block_num + \
                             cur_b_idx * self.front_bmm_m_dim * self.front_bmm_k_dim * \
                             self.block_num * self.block_num + \
                             cur_om_idx * om_size * self.front_bmm_k_dim * self.block_num * self.block_num + \
                             cur_m_idx * single_m_size * self.front_bmm_k_dim * self.block_num * self.block_num
            mask_length = single_m_size * self.front_bmm_k_dim * self.block_num // 2
        else:
            mask_gm_offset = block_idx * self.batch_outer_num * self.front_bmm_m_dim * self.front_bmm_k_dim * \
                             self.block_num * self.block_num // 8 + cur_b_idx * self.front_bmm_m_dim * \
                             self.front_bmm_k_dim * self.block_num * self.block_num // 8 + \
                             cur_om_idx * om_size * self.front_bmm_k_dim * self.block_num * self.block_num // 8 + \
                             cur_m_idx * single_m_size * self.front_bmm_k_dim * self.block_num * self.block_num // 8
            mask_length = single_m_size * self.front_bmm_k_dim * self.block_num // 2 // 8

        self.tik_instance.data_move(ub_mask[0], self.drop_mask_gm[mask_gm_offset],
                                    0, 1, mask_length, 0, 0)

    def do_mask_to_ub_for_drop_out(self, range_idx, m_range_size, ub_mask):
        block_idx, cur_b_idx, cur_om_idx, cur_m_idx = range_idx
        om_size, single_m_size, single_n_size = m_range_size
        if self.drop_v3:
            ele_move_offset = block_idx * self.batch_outer_num * self.first_m_dim * self.first_n_dim * \
                          self.block_num * self.block_num + \
                          cur_b_idx * self.first_m_dim * self.first_n_dim * self.block_num * self.block_num + \
                          cur_om_idx * om_size * self.block_num * self.block_num + \
                          cur_m_idx * single_m_size * self.block_num * self.block_num
            ele_move_repeat_times = single_n_size
            ele_move_data_size = single_m_size * self.block_num // 2
            ele_move_src_stride = (self.first_m_dim - single_m_size) * self.block_num // 2
        else:
            ele_move_offset = block_idx * self.batch_outer_num * self.first_m_dim * self.first_n_dim * \
                              self.block_num * self.block_num // 8 + cur_b_idx * self.first_m_dim * \
                              self.first_n_dim * self.block_num * self.block_num // 8 + \
                              cur_om_idx * om_size * 2 * self.block_num + \
                              cur_m_idx * single_m_size * 2 * self.block_num
            ele_move_repeat_times = self.first_n_dim
            ele_move_data_size = single_m_size * self.block_num * self.block_num // 32 // 8
            ele_move_src_stride = (self.first_m_dim - single_m_size)
        ele_move_dst_stride = 0
        self.tik_instance.data_move(ub_mask, self.drop_mask_gm[ele_move_offset],
                                    sid=0, nburst=ele_move_repeat_times, burst=ele_move_data_size,
                                    src_stride=ele_move_src_stride, dst_stride=ele_move_dst_stride)

    def drop_out_do_mask(self, ub_mask, ub_mask_fp16, tensor_a_ub, single_m_size, single_k_size):
        vconv_repeat_times = single_k_size * single_m_size * self.block_num * self.block_num // self.repeat_once_size
        if self.drop_v3:
            self.tik_instance.vconv(self.repeat_once_size, "",
                                ub_mask_fp16[0],
                                ub_mask[0],
                                vconv_repeat_times, 1, 1, 8, 4)
        else:
            self.tik_instance.vector_dup(self.repeat_once_size,
                                ub_mask_fp16[0],
                                self.tik_instance.Scalar(init_value=0, dtype="float16"), vconv_repeat_times, 1, 8)
            ub_mask_int64 = ub_mask.reinterpret_cast_to("uint64")
            droptimes = vconv_repeat_times
            repeat_times_dropout_vadds = 1
            scalar_heigher = self.tik_instance.Scalar(dtype="uint64")
            scalar_lower = self.tik_instance.Scalar(dtype="uint64")
            with self.tik_instance.for_range(0, droptimes) as dropidx:
                scalar_heigher.set_as(ub_mask_int64[2 * dropidx + 1])
                scalar_lower.set_as(ub_mask_int64[2 * dropidx])
                with self.tik_instance.if_scope(tik.any(scalar_heigher != 0, scalar_lower != 0)):
                    self.tik_instance.vadds([scalar_heigher, scalar_lower],
                                            ub_mask_fp16[self.repeat_once_size * dropidx],
                                            ub_mask_fp16[self.repeat_once_size * dropidx],
                                            self.tik_instance.Scalar(init_value=1, dtype="float16"),
                                            repeat_times_dropout_vadds,
                                            1, 1, 8, 8,
                                            0)
        self.tik_instance.vmuls(self.repeat_once_size,
                                ub_mask_fp16[0],
                                ub_mask_fp16[0],
                                self.tik_instance.Scalar(init_value = 1 / self.input_keep_prob,
                                                         dtype=self.matmul_dtype),
                                vconv_repeat_times, 1, 1, 8, 8)
        self.tik_instance.vmul(self.repeat_once_size, tensor_a_ub[0],
                                ub_mask_fp16[0],
                                tensor_a_ub[0],
                                vconv_repeat_times, 1, 1, 1, 8, 8, 8)

    def second_bmm_mat_mul_second_compute_mid_process(self, second_bmm_compute_buffers, second_bmm_compute_db_buffers,
                                                      second_bmm_compute_idxs, second_bmm_compute_each_layer_size,
                                                      target_output):
        tensor_a, tensor_b, tensor_c, tensor_a_l0a, tensor_b_l0b, \
        tensor_c_l0c, tensor_c_ub, ub_mask, ub_cast, reduce_ub = second_bmm_compute_buffers
        tensor_a_l1a_s_ub, tensor_b_s, tensor_a_l0a_ub, tensor_c_l0c_ub, \
        tensor_c_ub2, ub_mask2, ub_cast2, reduce_ub2 = second_bmm_compute_db_buffers
        block_idx, cur_b_idx, cur_om_idx, cur_m_idx = second_bmm_compute_idxs
        om_size, cur_m_size, cur_k_size, cur_n_size = second_bmm_compute_each_layer_size

        single_m_size, single_k_size, single_n_size = AttentionScoreGrad.get_second_single_time_mkn(cur_m_size,
                                                                                                    cur_k_size,
                                                                                                    cur_n_size)
        m_range = cur_m_size // single_m_size
        n_range = cur_n_size // single_n_size

        with self.tik_instance.for_range(0, m_range) as m:
            with self.tik_instance.for_range(0, n_range) as n:
                # do the next time
                l1a_offset = cur_om_idx * om_size * single_k_size * self.block_num * self.block_num + \
                             cur_m_idx * cur_m_size * single_k_size * self.block_num * self.block_num
                l1b_offset = 0
                l1a_repeat_times = single_m_size * cur_k_size
                l1b_repeat_times = single_n_size * cur_k_size
                # do the last time
                if target_output.name == "query_dx_gm":
                    self.tik_instance.load2dv1(tensor_a_l0a, tensor_c[l1a_offset],
                                               0, l1a_repeat_times, 1, 0, False)
                else:
                    base_l1a_offset = cur_om_idx * om_size * self.block_num * self.block_num + \
                                      cur_m_idx * cur_m_size * self.block_num * self.block_num
                    for l0a in range(single_m_size):
                        dst_offset = l0a * single_k_size * self.block_num * self.block_num
                        l1a_offset = base_l1a_offset + l0a * self.block_num * self.block_num
                        l1a_repeat_times = single_k_size
                        self.tik_instance.load2dv1(tensor_a_l0a[dst_offset], tensor_c[l1a_offset],
                                                   0, l1a_repeat_times, self.second_m_dim, 0, True)

                self.tik_instance.mmad(tensor_c_l0c, tensor_a_l0a, tensor_b_l0b,
                                       single_m_size * self.block_num,
                                       single_k_size * self.block_num,
                                       single_n_size * self.block_num,
                                       0)
                # do the next time
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
                output_dst_offset = inner_blk * single_n_size * self.x1_shape[0] * \
                                    self.second_m_dim * self.block_num * self.block_num + \
                                    outer_blk * self.second_m_dim * self.block_num * self.block_num + \
                                    cur_om_idx * om_size * self.block_num * self.block_num + \
                                    cur_m_idx * cur_m_size * self.block_num * self.block_num + \
                                    m * single_m_size * self.block_num * self.block_num

                self.tik_instance.data_move(target_output[output_dst_offset], ub_mask16[0],
                                            sid=0, nburst=repeat_times, burst=single_data_size,
                                            src_stride=0, dst_stride=output_dst_stride)

    def second_bmm_mat_mul_second_compute_last_process(self, second_bmm_compute_buffers, second_bmm_compute_db_buffers,
                                                       second_bmm_compute_idxs, second_bmm_compute_each_layer_size,
                                                       target_output):
        tensor_a, tensor_b, tensor_c, tensor_a_l0a, tensor_b_l0b, \
        tensor_c_l0c, tensor_c_ub, ub_mask, ub_cast, reduce_ub = second_bmm_compute_buffers
        tensor_a_l1a_s_ub, tensor_b_s, tensor_a_l0a_ub, tensor_c_l0c_ub, \
        tensor_c_ub2, ub_mask2, ub_cast2, reduce_ub2 = second_bmm_compute_db_buffers
        block_idx, cur_b_idx, cur_om_idx, cur_m_idx = second_bmm_compute_idxs
        om_size, cur_m_size, cur_k_size, cur_n_size = second_bmm_compute_each_layer_size

        single_m_size, single_k_size, single_n_size = AttentionScoreGrad.get_second_single_time_mkn(cur_m_size,
                                                                                                    cur_k_size,
                                                                                                    cur_n_size)
        m_range = cur_m_size // single_m_size
        n_range = cur_n_size // single_n_size

        with self.tik_instance.for_range(0, m_range) as m:
            with self.tik_instance.for_range(0, n_range) as n:
                l1a_offset = cur_om_idx * om_size * single_k_size * self.block_num * self.block_num + \
                             cur_m_idx * cur_m_size * single_k_size * self.block_num * self.block_num
                l1a_repeat_times = single_m_size * cur_k_size
                # do the last time
                if target_output.name == "query_dx_gm":
                    self.tik_instance.load2dv1(tensor_a_l0a_ub, tensor_c[l1a_offset],
                                               0, l1a_repeat_times, 1, 0, False)
                else:
                    base_l1a_offset = cur_om_idx * om_size * self.block_num * self.block_num + \
                                      cur_m_idx * cur_m_size * self.block_num * self.block_num
                    for l0a in range(single_m_size):
                        dst_offset = l0a * single_k_size * self.block_num * self.block_num
                        l1a_offset = base_l1a_offset + l0a * self.block_num * self.block_num
                        l1a_repeat_times = single_k_size
                        self.tik_instance.load2dv1(tensor_a_l0a_ub[dst_offset], tensor_c[l1a_offset],
                                                   0, l1a_repeat_times, self.second_m_dim, 0, True)
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
                output_dst_offset = inner_blk * single_n_size * self.x1_shape[0] * \
                                    self.second_m_dim * self.block_num * self.block_num + \
                                    outer_blk * self.second_m_dim * self.block_num * self.block_num + \
                                    cur_om_idx * om_size * self.block_num * self.block_num + \
                                    cur_m_idx * cur_m_size * self.block_num * self.block_num + \
                                    m * single_m_size * self.block_num * self.block_num

                self.tik_instance.data_move(target_output[output_dst_offset], tensor_c_ub2[0],
                                            sid=0, nburst=repeat_times, burst=single_data_size,
                                            src_stride=0, dst_stride=output_dst_stride)

    def softmax_grad_compute(self, tensor_input, data_x1, mul_value, cur_m, cur_n, ub_mask, softmax_ub, reduce_ub):
        self.drop_out_do_mask(ub_mask, softmax_ub, tensor_input, cur_m, cur_n)

        ele_compute_repeat_times = cur_m * cur_n * 16 * 16 // self.repeat_once_size
        tail = max(ele_compute_repeat_times // 255, 0) * (ele_compute_repeat_times % 255)
        ele_compute_repeat_times = min(255, ele_compute_repeat_times)
        self.tik_instance.vmul(self.repeat_once_size, softmax_ub, tensor_input,
                               data_x1,
                               ele_compute_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        if tail != 0:
            self.tik_instance.vmul(self.repeat_once_size, softmax_ub[255 * self.repeat_once_size],
                                   tensor_input[255 * self.repeat_once_size],
                                   data_x1[255 * self.repeat_once_size],
                                   ele_compute_repeat_times,
                                   self.block_stride, self.block_stride, self.block_stride,
                                   self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmuls(self.repeat_once_size, data_x1, data_x1,
                                mul_value,
                                ele_compute_repeat_times,
                                self.block_stride, self.block_stride,
                                self.repeat_stride, self.repeat_stride)
        if tail != 0:
            self.tik_instance.vmuls(self.repeat_once_size, data_x1[255 * self.repeat_once_size],
                                    data_x1[255 * self.repeat_once_size],
                                    mul_value,
                                    ele_compute_repeat_times,
                                    self.block_stride, self.block_stride,
                                    self.repeat_stride, self.repeat_stride)

        vmax_range = cur_n
        src_tensor = softmax_ub
        while (vmax_range > 1):
            if vmax_range % 2 == 0:
                repeat_time = cur_m * vmax_range * self.block_num * self.block_num // self.repeat_once_size // 2
                src_offset = cur_m * vmax_range * self.block_num * self.block_num // 2
                self.tik_instance.vadd(self.repeat_once_size, softmax_ub[0], src_tensor[0], src_tensor[src_offset],
                                       repeat_time,
                                       self.block_stride, self.block_stride, self.block_stride,
                                       self.repeat_stride, self.repeat_stride, self.repeat_stride)
                vmax_range = vmax_range // 2
            else:
                repeat_time = cur_m * self.block_num * self.block_num // self.repeat_once_size
                src_offset = (vmax_range - 1) * cur_m * self.block_num * self.block_num
                self.tik_instance.vadd(self.repeat_once_size, softmax_ub[0], src_tensor[0], src_tensor[src_offset],
                                       repeat_time,
                                       self.block_stride, self.block_stride, self.block_stride,
                                       self.repeat_stride, self.repeat_stride, self.repeat_stride)
                vmax_range = vmax_range - 1

        repeat_time = cur_m * self.block_num * self.block_num // self.repeat_once_size

        self.tik_instance.vcgadd(self.repeat_once_size, reduce_ub[0], softmax_ub[0], repeat_time,
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

        self.tik_instance.vmul(self.repeat_once_size, tensor_input, tensor_input,
                               data_x1,
                               ele_compute_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        if tail != 0:
            self.tik_instance.vmul(self.repeat_once_size, tensor_input[255 * self.repeat_once_size],
                                   tensor_input[255 * self.repeat_once_size],
                                   data_x1[255 * self.repeat_once_size],
                                   ele_compute_repeat_times,
                                   self.block_stride, self.block_stride, self.block_stride,
                                   self.repeat_stride, self.repeat_stride, self.repeat_stride)

        return tensor_input

    def first_bmm_mat_mul_compute_insn_process(self, first_bmm_compute_buffers, first_bmm_compute_idx,
                                               first_bmm_compute_each_layer_size):
        """
        first bmm compute.
        """
        tensor_a, tensor_b, tensor_y, tensor_a_l0a, \
        tensor_b_l0b, tensor_c_l0c, tensor_c_ub, mul_value, \
        elewise_data_ub2, ub_mask, ub_mask_fp16, reduce_ub = first_bmm_compute_buffers
        block_idx, cur_b_idx, cur_om_idx, cur_m_idx = first_bmm_compute_idx
        om_size, cur_m_size, cur_k_size, cur_n_size = first_bmm_compute_each_layer_size
        single_m_size, single_k_size, single_n_size = AttentionScoreGrad.get_first_single_time_mkn(cur_m_size,
                                                                                                   cur_k_size,
                                                                                                   cur_n_size)
        m_range = cur_m_size // single_m_size
        n_range = cur_n_size // single_n_size

        with self.tik_instance.for_range(0, m_range) as m:
            with self.tik_instance.for_range(0, n_range) as n:
                ele_move_offset = block_idx * self.batch_outer_num * self.first_m_dim * self.first_n_dim * \
                                  self.block_num * self.block_num + \
                                  cur_b_idx * self.first_m_dim * self.first_n_dim * self.block_num * self.block_num + \
                                  cur_om_idx * om_size * self.block_num * self.block_num + \
                                  cur_m_idx * cur_m_size * self.block_num * self.block_num + \
                                  m * single_m_size * self.block_num * self.block_num + \
                                  n * self.first_m_dim * self.block_num * self.block_num
                ele_move_repeat_times = single_n_size
                ele_move_data_size = single_m_size * self.block_num
                ele_move_src_stride = (self.first_m_dim - single_m_size) * self.block_num
                ele_move_dst_stride = 0
                self.do_mask_to_ub_for_drop_out([block_idx, cur_b_idx, cur_om_idx, cur_m_idx],
                                                [om_size, cur_m_size, single_n_size], ub_mask)
                self.tik_instance.data_move(elewise_data_ub2, self.attention_score_gm[ele_move_offset],
                                            sid=0, nburst=ele_move_repeat_times, burst=ele_move_data_size,
                                            src_stride=ele_move_src_stride, dst_stride=ele_move_dst_stride)

                l1a_offset = cur_om_idx * om_size * single_k_size * self.block_num * self.block_num + \
                             cur_m_idx * cur_m_size * single_k_size * self.block_num * self.block_num + \
                             m * single_m_size * single_k_size * self.block_num * self.block_num
                l1a_repeat_times = single_m_size * cur_k_size
                self.tik_instance.load2dv1(tensor_a_l0a, tensor_a[l1a_offset],
                                           0, l1a_repeat_times, 1, 0, False)

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

                self.softmax_grad_compute(tensor_c_ub, elewise_data_ub2, mul_value, single_m_size,
                                          single_n_size, ub_mask, ub_mask_fp16, reduce_ub)

                # copy_ub_to_l1
                mid_data_lengh = self.block_num
                mid_src_stride = (single_m_size - 1) * self.block_num
                mid_dst_stride = 0
                output_outer_m_offset = cur_om_idx * om_size * single_n_size * self.block_num * self.block_num + \
                                        cur_m_idx * cur_m_size * single_n_size * self.block_num * self.block_num
                for _trans_ids in range(single_m_size):
                    mid_offset = output_outer_m_offset + _trans_ids * single_n_size * self.block_num * self.block_num
                    mid_src_offset = _trans_ids * self.block_num * self.block_num
                    self.tik_instance.data_move(tensor_y[mid_offset],
                                                tensor_c_ub[mid_src_offset],
                                                sid=0, nburst=single_n_size, burst=mid_data_lengh,
                                                src_stride=mid_src_stride, dst_stride=mid_dst_stride)

    def third_bmm_move_tensor_b_from_gm_to_l1(self, block_idx, cur_b_idx):
        second_bmm_tensor_b = self.tik_instance.Tensor(self.matmul_dtype, [self.second_n_dim, self.second_k_dim,
                                                                           self.block_num, self.block_num],
                                                       name="second_bmm_tensor_b", scope=self.tik.scope_cbuf)
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
                                        self.x4_gm[second_bmm_tensor_b_src_offset],
                                        sid=0, nburst=second_bmm_tensor_b_repeat_times,
                                        burst=second_bmm_tensor_b_burst,
                                        src_stride=second_bmm_tensor_b_src_stride,
                                        dst_stride=second_bmm_tensor_b_dst_stride)

        return second_bmm_tensor_b

    def second_bmm_compute_for_range_partition(self, second_bmm_apply_compute_buffers,
                                               second_bmm_apply_compute_db_buffers,
                                               second_bmm_ori_compute_idxs, second_bmm_compute_each_layer_size,
                                               target_output, unroll_range):
        for cur_m_idx in range(unroll_range):
            second_bmm_compute_idxs = list(second_bmm_ori_compute_idxs) + [cur_m_idx]
            if cur_m_idx % 2 == 0:
                second_bmm_compute_buffers = second_bmm_apply_compute_buffers[0]

                second_bmm_compute_db_buffers = second_bmm_apply_compute_db_buffers[0]
            else:
                second_bmm_compute_buffers = second_bmm_apply_compute_buffers[1]

                second_bmm_compute_db_buffers = second_bmm_apply_compute_db_buffers[1]
            self.second_bmm_mat_mul_second_compute_mid_process(second_bmm_compute_buffers,
                                                               second_bmm_compute_db_buffers,
                                                               second_bmm_compute_idxs,
                                                               second_bmm_compute_each_layer_size,
                                                               target_output)

        second_bmm_compute_idxs = list(second_bmm_ori_compute_idxs) + [unroll_range]
        self.second_bmm_mat_mul_second_compute_last_process(second_bmm_apply_compute_buffers[0],
                                                            second_bmm_apply_compute_db_buffers[0],
                                                            second_bmm_compute_idxs, second_bmm_compute_each_layer_size,
                                                            target_output)

    def second_bmm_compute_for_outer_m_once(self, preload_buffers, range_idxs, outer_m_range_once_m_size,
                                            inner_m_range_value, target_output):
        """
        second bmm compute for outer m once.
        """
        second_bmm_tensor_b, second_bmm_tensor_b_l0b, second_bmm_tensor_c = preload_buffers
        block_idx, cur_b_idx, cur_om_idx = range_idxs
        self.tik_instance.load2dv1(second_bmm_tensor_b_l0b, second_bmm_tensor_b[0],
                                   0, self.first_n_dim * self.second_n_dim, 1, 0, True)
        inner_m_size, inner_k_size, \
        inner_n_size = AttentionScoreGrad.get_second_compute_one_core_mkn(outer_m_range_once_m_size,
                                                                          inner_m_range_value,
                                                                          self.second_k_dim,
                                                                          self.second_n_dim)
        l1a_shape = [inner_k_size, inner_m_size, self.block_num, self.block_num]
        ub_start_addr = 0
        second_bmm_tensor_a = None
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
        second_bmm_tensor_a_db = None
        second_bmm_tensor_a_l0a_db = self.tik_instance.Tensor(self.matmul_dtype, l1a_shape,
                                                              name="second_bmm_tensor_a_l0a_db",
                                                              scope=self.tik.scope_ca,
                                                              start_addr=32768)

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
        second_bmm_compute_each_layer_size = [outer_m_range_once_m_size, inner_m_size, inner_k_size, inner_n_size]
        unroll_range = inner_m_range_value - 1
        second_bmm_compute_buffers = [second_bmm_tensor_a, second_bmm_tensor_b, second_bmm_tensor_c,
                                      second_bmm_tensor_a_l0a, second_bmm_tensor_b_l0b, second_bmm_tensor_c_l0c,
                                      second_bmm_tensor_c_ub, second_bmm_mask_ub, second_bmm_mask_cast_ub,
                                      second_bmm_softmax_reduce_ub]
        second_bmm_compute_db_buffers = [second_bmm_tensor_a_db, second_bmm_tensor_b,
                                         second_bmm_tensor_a_l0a_db, second_bmm_tensor_c_l0c_db,
                                         second_bmm_tensor_c_ub_db, second_bmm_mask_ub_db,
                                         second_bmm_mask_cast_ub_db,
                                         second_bmm_softmax_reduce_ub_db]
        second_bmm_compute_next_buffers = [second_bmm_tensor_a_db, second_bmm_tensor_b, second_bmm_tensor_c,
                                           second_bmm_tensor_a_l0a_db, second_bmm_tensor_b_l0b,
                                           second_bmm_tensor_c_l0c_db,
                                           second_bmm_tensor_c_ub_db, second_bmm_mask_ub_db,
                                           second_bmm_mask_cast_ub_db,
                                           second_bmm_softmax_reduce_ub_db]
        second_bmm_compute_db_next_buffers = [second_bmm_tensor_a, second_bmm_tensor_b,
                                              second_bmm_tensor_a_l0a, second_bmm_tensor_c_l0c,
                                              second_bmm_tensor_c_ub, second_bmm_mask_ub,
                                              second_bmm_mask_cast_ub,
                                              second_bmm_softmax_reduce_ub]
        second_bmm_compute_idxs = (block_idx, cur_b_idx, cur_om_idx)
        self.second_bmm_compute_for_range_partition([second_bmm_compute_buffers, second_bmm_compute_next_buffers],
                                                    [second_bmm_compute_db_buffers, second_bmm_compute_db_next_buffers],
                                                    second_bmm_compute_idxs, second_bmm_compute_each_layer_size,
                                                    target_output, unroll_range)

    def first_bmm_compute_for_outer_m_once(self, preload_buffers, range_idxs, outer_m_range_once_m_size,
                                           inner_m_range_value, mul_value):
        """
        compute first bmm once for outer m range.
        """
        first_bmm_tensor_b, first_bmm_tensor_c, first_bmm_tensor_b_l0b, first_bmm_tensor_a = preload_buffers
        block_idx, cur_b_idx, cur_om_idx = range_idxs
        with self.tik_instance.for_range(0, inner_m_range_value // self.double_factor) as inner_m_idx:
            inner_m_size, inner_k_size, \
            inner_n_size = AttentionScoreGrad.get_first_compute_one_core_mkn(outer_m_range_once_m_size,
                                                                             inner_m_range_value,
                                                                             self.first_k_dim,
                                                                             self.first_n_dim)

            first_bmm_tensor_a_l0a = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * inner_k_size *
                                                                                  self.block_num * self.block_num],
                                                              name="first_bmm_tensor_a_l0a", scope=self.tik.scope_ca)

            first_bmm_tensor_c_l0c = self.tik_instance.Tensor("float32", [inner_m_size * inner_n_size *
                                                                          self.block_num * self.block_num],
                                                              name="first_bmm_tensor_c_l0c", scope=self.tik.scope_cc)
            ub_addr = 32
            first_bmm_tensor_c_ub = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * inner_n_size *
                                                                                 self.block_num * self.block_num],
                                                             name="first_bmm_tensor_c_ub",
                                                             scope=self.tik.scope_ubuf, start_addr=ub_addr)
            ub_addr = ub_addr + inner_m_size * inner_n_size * self.block_num * self.block_num * 2
            elewise_add_data_ub = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * self.first_n_dim *
                                                                               self.block_num * self.block_num],
                                                             name="elewise_add_data_ub",
                                                             scope=self.tik.scope_ubuf, start_addr=ub_addr)
            ub_addr = ub_addr + inner_m_size * self.first_n_dim * self.block_num * self.block_num * 2
            first_ub_mask = self.tik_instance.Tensor(self.mask_dtype, [inner_m_size * self.first_n_dim *
                                                                         self.block_num * self.block_num],
                                                     name="first_ub_mask",
                                                     scope=self.tik.scope_ubuf, start_addr=ub_addr)
            ub_addr = ub_addr + inner_m_size * self.first_n_dim * self.block_num * self.block_num
            first_ub_mask_fp16 = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * self.first_n_dim *
                                                                         self.block_num * self.block_num],
                                                          name="first_ub_mask_fp16",
                                                          scope=self.tik.scope_ubuf, start_addr=ub_addr)
            ub_addr = ub_addr + inner_m_size * self.first_n_dim * self.block_num * self.block_num * 2
            reduce_ub = self.tik_instance.Tensor(self.matmul_dtype, [2 * self.block_num],
                                                 name="reduce_ub",
                                                 scope=self.tik.scope_ubuf, start_addr=ub_addr)
            first_bmm_compute_buffers = [first_bmm_tensor_a, first_bmm_tensor_b, first_bmm_tensor_c,
                                         first_bmm_tensor_a_l0a, first_bmm_tensor_b_l0b, first_bmm_tensor_c_l0c,
                                         first_bmm_tensor_c_ub, mul_value, elewise_add_data_ub,
                                         first_ub_mask, first_ub_mask_fp16, reduce_ub]

            first_bmm_compute_idx = [block_idx, cur_b_idx, cur_om_idx, 2 * inner_m_idx]
            first_bmm_compute_each_layer_size = [outer_m_range_once_m_size, inner_m_size, inner_k_size, inner_n_size]
            self.first_bmm_mat_mul_compute_insn_process(first_bmm_compute_buffers,
                                                        first_bmm_compute_idx, first_bmm_compute_each_layer_size)

            first_bmm_tensor_a_db = first_bmm_tensor_a

            first_bmm_tensor_a_l0a_db = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * inner_k_size *
                                                                                     self.block_num * self.block_num],
                                                                 name="first_bmm_tensor_a_l0a_db",
                                                                 scope=self.tik.scope_ca, start_addr=32768)

            first_bmm_tensor_c_l0c_db = self.tik_instance.Tensor("float32", [inner_m_size * inner_n_size *
                                                                          self.block_num * self.block_num],
                                                                 name="first_bmm_tensor_c_l0c_db",
                                                                 scope=self.tik.scope_cc, start_addr=32768)
            ub_addr = 131072
            first_bmm_tensor_c_ub_db = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * inner_n_size *
                                                                                    self.block_num * self.block_num],
                                                                name="first_bmm_tensor_c_ub_db",
                                                                scope=self.tik.scope_ubuf, start_addr=ub_addr)
            ub_addr = ub_addr + inner_m_size * inner_n_size * self.block_num * self.block_num * 2
            elewise_add_data_ub_db = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * inner_n_size *
                                                                               self.block_num * self.block_num],
                                                              name="elewise_add_data_ub_db",
                                                              scope=self.tik.scope_ubuf, start_addr=ub_addr)
            ub_addr = ub_addr + inner_m_size * inner_n_size * self.block_num * self.block_num * 2
            first_ub_mask_db = self.tik_instance.Tensor(self.mask_dtype, [inner_m_size * self.first_n_dim *
                                                                          self.block_num * self.block_num],
                                                        name="first_ub_mask_db",
                                                        scope=self.tik.scope_ubuf, start_addr=ub_addr)
            ub_addr = ub_addr + inner_m_size * inner_n_size * self.block_num * self.block_num
            first_ub_mask_fp16_db = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * self.first_n_dim *
                                                                              self.block_num * self.block_num],
                                                             name="first_ub_mask_fp16_db",
                                                             scope=self.tik.scope_ubuf, start_addr=ub_addr)
            ub_addr = ub_addr + inner_m_size * self.first_n_dim * self.block_num * self.block_num * 2
            reduce_ub_db = self.tik_instance.Tensor(self.matmul_dtype, [2 * self.block_num],
                                                    name="reduce_ub_db",
                                                    scope=self.tik.scope_ubuf, start_addr=ub_addr)
            first_bmm_compute_db_buffers = [first_bmm_tensor_a_db, first_bmm_tensor_b, first_bmm_tensor_c,
                                            first_bmm_tensor_a_l0a_db, first_bmm_tensor_b_l0b,
                                            first_bmm_tensor_c_l0c_db,
                                            first_bmm_tensor_c_ub_db, mul_value, elewise_add_data_ub_db,
                                            first_ub_mask_db, first_ub_mask_fp16_db, reduce_ub_db]
            first_bmm_compute_idx = [block_idx, cur_b_idx, cur_om_idx, 2 * inner_m_idx + 1]
            self.first_bmm_mat_mul_compute_insn_process(first_bmm_compute_db_buffers,
                                                        first_bmm_compute_idx, first_bmm_compute_each_layer_size)

    def apply_buffer_for_tensor_b_l0_and_move_data_in(self, first_bmm_tensor_b):
        """

        """
        first_bmm_tensor_b_l0b = self.tik_instance.Tensor(self.matmul_dtype,
                                                          [self.first_k_dim, self.first_n_dim,
                                                           self.block_num, self.block_num],
                                                          name="first_bmm_tensor_b_l0b",
                                                          scope=self.tik.scope_cb)

        self.tik_instance.load2dv1(first_bmm_tensor_b_l0b, first_bmm_tensor_b[0],
                                   0, self.first_k_dim * self.first_n_dim,
                                   1, 0, False)

        return first_bmm_tensor_b_l0b

    def apply_buffer_for_tensor_c_l1(self):
        """

        """
        first_bmm_tensor_c = self.tik_instance.Tensor(self.matmul_dtype,
                                                      [self.first_n_dim, self.first_m_dim,
                                                       self.block_num, self.block_num],
                                                      name="first_bmm_tensor_c", scope=self.tik.scope_cbuf)

        return first_bmm_tensor_c

    def second_bmm_move_tensor_b_from_gm_to_l1(self, block_idx, cur_b_idx):
        """

        """
        second_bmm_tensor_b = self.tik_instance.Tensor(self.matmul_dtype, [self.second_n_dim, self.second_k_dim,
                                                                          self.block_num, self.block_num],
                                                      name="second_bmm_tensor_b", scope=self.tik.scope_cbuf)
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

    def first_bmm_move_tensor_b_from_gm_to_l1(self, block_idx, batch_range_value, cur_b_idx):
        """

        """
        first_bmm_tensor_b = self.tik_instance.Tensor(self.matmul_dtype, [self.first_k_dim, self.first_n_dim,
                                                                          self.block_num, self.block_num],
                                                      name="first_bmm_tensor_b", scope=self.tik.scope_cbuf)
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

    def front_bmm_compute_for_outer_m_once_insn_process(self, front_bmm_compute_buffers, front_bmm_compute_idx,
                                                        front_bmm_compute_each_layer_size):
        """
        compute the front bmm res.
        """
        tensor_a, tensor_b, tensor_a_l0a, tensor_b_l0b, \
        tensor_c_l0c, tensor_c_ub, ub_mask, ub_mask_fp16, tensor_a_ub = front_bmm_compute_buffers
        single_m_size, single_k_size, single_n_size = front_bmm_compute_each_layer_size
        blk_idx, batch_outer_idx, m_idx, db_idx = front_bmm_compute_idx
        attention_score_offset = blk_idx * self.batch_outer_num * self.front_bmm_m_dim * \
                                 self.front_bmm_k_dim * self.block_num * self.block_num + \
                                 batch_outer_idx * self.front_bmm_m_dim * self.front_bmm_k_dim * \
                                 self.block_num * self.block_num + \
                                 m_idx * 2 * single_m_size * self.front_bmm_k_dim * self.block_num * self.block_num + \
                                 db_idx * single_m_size * self.front_bmm_k_dim * self.block_num * self.block_num

        self.tik_instance.data_move(tensor_a_ub[0], self.attention_score_gm[attention_score_offset],
                                    sid=0, nburst=1, burst=single_k_size * single_m_size * self.block_num,
                                    src_stride=0, dst_stride=0)
        self.drop_out_do_mask(ub_mask, ub_mask_fp16, tensor_a_ub, single_m_size, single_k_size)
        self.tik_instance.data_move(tensor_a[0], tensor_a_ub[0],
                                    sid=0, nburst=1, burst=single_k_size * single_m_size * self.block_num,
                                    src_stride=0, dst_stride=0)
        l1a_repeat_times = single_k_size * single_m_size
        # load tensor_a to l0a
        self.tik_instance.load2dv1(tensor_a_l0a, tensor_a[0],
                                   0, l1a_repeat_times, 1, 0, True)
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

        outer_blk = (blk_idx * self.batch_outer_num + batch_outer_idx) // self.x1_shape[1]
        inner_blk = (blk_idx * self.batch_outer_num + batch_outer_idx) % self.x1_shape[1]
        single_data_size = single_m_size * self.block_num
        repeat_times = single_n_size
        output_dst_stride = (self.x1_shape[0] * self.front_bmm_m_dim - single_m_size) * self.block_num

        output_dst_offset = inner_blk * single_n_size * self.x1_shape[0] * \
                            self.front_bmm_m_dim * self.block_num * self.block_num + \
                            outer_blk * self.front_bmm_m_dim * self.block_num * self.block_num + \
                            m_idx * 2 * single_m_size * self.block_num * self.block_num + \
                            db_idx * single_m_size * self.block_num * self.block_num

        self.tik_instance.data_move(self.value_dw_gm[output_dst_offset], tensor_c_ub[0],
                                    sid=0, nburst=repeat_times, burst=single_data_size,
                                    src_stride=0, dst_stride=output_dst_stride)

    def front_bmm_compute_for_outer_m_once(self, front_bmm_preload_buffers, front_bmm_m_range_once_m_size,
                                           range_idxs):
        """
        compute front bmm for once m repeat.
        """
        single_m_size = front_bmm_m_range_once_m_size
        single_k_size = self.front_bmm_x1_shape[3]
        single_n_size = self.x1_shape[2]
        front_bmm_tensor_b_l1, front_bmm_tensor_b_l0b = front_bmm_preload_buffers
        block_idx, cur_b_idx, cur_om_idx = range_idxs
        ub_addr = 32
        ub_mask = self.tik_instance.Tensor(self.mask_dtype, [single_m_size * single_k_size *
                                                             self.block_num * self.block_num],
                                           name="ub_mask",
                                           scope=self.tik.scope_ubuf, start_addr=ub_addr)
        ub_addr = ub_addr + single_m_size * single_k_size * self.block_num * self.block_num
        ub_mask_fp16 = self.tik_instance.Tensor(self.matmul_dtype, [single_m_size * single_k_size *
                                                                    self.block_num * self.block_num],
                                                name="ub_mask_fp16",
                                                scope=self.tik.scope_ubuf, start_addr=ub_addr)
        self.do_mask_to_ub_for_attention_score([block_idx, cur_b_idx, 0, 2 * cur_om_idx], [0, single_m_size], ub_mask)
        ub_addr = ub_addr + single_m_size * single_k_size * self.block_num * self.block_num * 2
        front_bmm_tensor_a_ub = self.tik_instance.Tensor(self.matmul_dtype, [single_m_size * single_k_size *
                                                                             self.block_num * self.block_num],
                                                         name="front_bmm_tensor_a_ub",
                                                         scope=self.tik.scope_ubuf, start_addr=ub_addr)

        front_bmm_tensor_a_l1 = self.tik_instance.Tensor(self.matmul_dtype, [single_m_size * single_k_size *
                                                                             self.block_num * self.block_num],
                                                         name="front_bmm_tensor_a_l1",
                                                         scope=self.tik.scope_cbuf)
        front_bmm_tensor_a_l0a = self.tik_instance.Tensor(self.matmul_dtype, [single_m_size * single_k_size *
                                                                              self.block_num * self.block_num],
                                                          name="front_bmm_tensor_a_l0a", scope=self.tik.scope_ca)

        front_bmm_tensor_c_l0c = self.tik_instance.Tensor("float32", [single_m_size * single_n_size *
                                                                      self.block_num * self.block_num],
                                                          name="front_bmm_tensor_c_l0c", scope=self.tik.scope_cc)
        ub_addr = ub_addr + single_m_size * single_k_size * self.block_num * self.block_num * 2
        front_bmm_tensor_c_ub = self.tik_instance.Tensor(self.matmul_dtype, [single_m_size * single_n_size *
                                                                             self.block_num * self.block_num],
                                                         name="front_bmm_tensor_c_ub",
                                                         scope=self.tik.scope_ubuf, start_addr=ub_addr)
        front_bmm_compute_buffers = [front_bmm_tensor_a_l1, front_bmm_tensor_b_l1,
                                     front_bmm_tensor_a_l0a, front_bmm_tensor_b_l0b, front_bmm_tensor_c_l0c,
                                     front_bmm_tensor_c_ub, ub_mask, ub_mask_fp16, front_bmm_tensor_a_ub]

        front_bmm_compute_idx = [block_idx, cur_b_idx, cur_om_idx, 0]
        front_bmm_compute_each_layer_size = [single_m_size, single_k_size, single_n_size]
        self.front_bmm_compute_for_outer_m_once_insn_process(front_bmm_compute_buffers, front_bmm_compute_idx,
                                                             front_bmm_compute_each_layer_size)

        ub_addr = 131072
        ub_mask_db = self.tik_instance.Tensor(self.mask_dtype, [single_m_size * single_k_size *
                                                             self.block_num * self.block_num],
                                              name="ub_mask_db",
                                              scope=self.tik.scope_ubuf, start_addr=ub_addr)
        ub_addr = ub_addr + single_m_size * single_k_size * self.block_num * self.block_num
        ub_mask_fp16_db = self.tik_instance.Tensor(self.matmul_dtype, [single_m_size * single_k_size *
                                                                    self.block_num * self.block_num],
                                                   name="ub_mask_fp16_db",
                                                   scope=self.tik.scope_ubuf, start_addr=ub_addr)
        self.do_mask_to_ub_for_attention_score([block_idx, cur_b_idx, 0, 2 * cur_om_idx + 1],
                                               [0, single_m_size], ub_mask_db)
        ub_addr = ub_addr + single_m_size * single_k_size * self.block_num * self.block_num * 2
        front_bmm_tensor_a_ub_db = self.tik_instance.Tensor(self.matmul_dtype, [single_m_size * single_k_size *
                                                                             self.block_num * self.block_num],
                                                            name="front_bmm_tensor_a_ub_db",
                                                            scope=self.tik.scope_ubuf, start_addr=ub_addr)
        front_bmm_tensor_a_l1_db = self.tik_instance.Tensor(self.matmul_dtype, [single_m_size * single_k_size *
                                                                                self.block_num * self.block_num],
                                                            name="front_bmm_tensor_a_l1_db",
                                                            scope=self.tik.scope_cbuf)

        front_bmm_tensor_a_l0a_db = self.tik_instance.Tensor(self.matmul_dtype, [single_m_size * single_k_size *
                                                                                 self.block_num * self.block_num],
                                                             name="front_bmm_tensor_a_l0a_db",
                                                             scope=self.tik.scope_ca, start_addr=32768)

        front_bmm_tensor_c_l0c_db = self.tik_instance.Tensor("float32", [single_m_size * single_n_size *
                                                                         self.block_num * self.block_num],
                                                             name="front_bmm_tensor_c_l0c_db",
                                                             scope=self.tik.scope_cc, start_addr=32768)
        ub_addr = ub_addr + single_m_size * single_k_size * self.block_num * self.block_num * 2
        front_bmm_tensor_c_ub_db = self.tik_instance.Tensor(self.matmul_dtype, [single_m_size * single_n_size *
                                                                                self.block_num * self.block_num],
                                                            name="front_bmm_tensor_c_ub_db",
                                                            scope=self.tik.scope_ubuf, start_addr=ub_addr)
        front_bmm_compute_db_buffers = [front_bmm_tensor_a_l1_db, front_bmm_tensor_b_l1,
                                        front_bmm_tensor_a_l0a_db, front_bmm_tensor_b_l0b, front_bmm_tensor_c_l0c_db,
                                        front_bmm_tensor_c_ub_db, ub_mask_db, ub_mask_fp16_db,
                                        front_bmm_tensor_a_ub_db]
        front_bmm_compute_idx = [block_idx, cur_b_idx, cur_om_idx, 1]
        self.front_bmm_compute_for_outer_m_once_insn_process(front_bmm_compute_db_buffers, front_bmm_compute_idx,
                                                             front_bmm_compute_each_layer_size)

    def apply_buffer_for_front_bmm_l0b_and_move_data_in(self, front_bmm_tensor_b_l1):
        front_bmm_tensor_b_l0b = self.tik_instance.Tensor(self.matmul_dtype, [self.x1_shape[2] *
                                                                              self.x1_shape[3] *
                                                                              self.block_num * self.block_num],
                                                          name="front_bmm_tensor_b_l0b",
                                                          scope=self.tik.scope_cb)
        self.tik_instance.load2dv1(front_bmm_tensor_b_l0b, front_bmm_tensor_b_l1[0],
                                   0, self.x1_shape[2] * self.x1_shape[3], 1, 0, True)

        return front_bmm_tensor_b_l0b

    def apply_buffer_for_front_bmm_tensor_b_and_move_data_in(self, block_idx, cur_b_idx):
        front_bmm_tensor_b_l1 = self.tik_instance.Tensor(self.matmul_dtype, [self.x1_shape[2] * self.x1_shape[3] *
                                                                             self.block_num * self.block_num],
                                                         name="front_bmm_tensor_b_l1",
                                                         scope=self.tik.scope_cbuf)
        for dma_idx in range(self.x1_shape[2]):
            second_bmm_tensor_b_src_offset = block_idx * self.batch_outer_num * self.x1_shape[2] * self.x1_shape[3] * \
                                             self.block_num * self.block_num + \
                                             cur_b_idx * self.x1_shape[2] * self.x1_shape[3] * \
                                             self.block_num * self.block_num + \
                                             dma_idx * self.x1_shape[3] * self.block_num * self.block_num
            second_bmm_tensor_b_dst_offset = dma_idx * self.block_num * self.block_num
            second_bmm_tensor_b_burst = self.block_num
            second_bmm_tensor_b_repeat_times = self.x1_shape[3]
            second_bmm_tensor_b_src_stride = 0
            second_bmm_tensor_b_dst_stride = (self.x1_shape[2] - 1) * self.block_num
            self.tik_instance.data_move(front_bmm_tensor_b_l1[second_bmm_tensor_b_dst_offset],
                                        self.x1_gm[second_bmm_tensor_b_src_offset],
                                        sid=0, nburst=second_bmm_tensor_b_repeat_times,
                                        burst=second_bmm_tensor_b_burst,
                                        src_stride=second_bmm_tensor_b_src_stride,
                                        dst_stride=second_bmm_tensor_b_dst_stride)

        return front_bmm_tensor_b_l1

    def set_mul_scalar_value(self, mul_sclar):
        mul_x_ub = self.tik_instance.Tensor(self.matmul_dtype,
                                            [self.block_num],
                                            name="mul_x_ub", scope=self.tik.scope_ubuf)
        self.tik_instance.data_move(mul_x_ub, self.mul_gm[0],
                                    sid=0, nburst=1, burst=1,
                                    src_stride=0, dst_stride=0)
        mul_sclar.set_as(mul_x_ub[0])

    def tiling_batch_m_axis(self, m_size):
        if m_size % 8 == 0:
            outer_m_range_value = 2
            inner_m_range_value = self.x1_shape[3] // 4
        elif m_size % 4 == 0:
            outer_m_range_value = 1
            inner_m_range_value = self.x1_shape[3] // 2
        else:
            outer_m_range_value = 1
            inner_m_range_value = self.x1_shape[3]

        return outer_m_range_value, inner_m_range_value

    def compute_one_core(self, block_idx, batch_range_value, batch_range_value_last, used_core_num):
        """
        """
        outer_m_range_value, inner_m_range_value = self.tiling_batch_m_axis(self.first_m_dim)
        self.batch_outer_num = batch_range_value
        outer_m_range_once_m_size = self.first_m_dim // outer_m_range_value
        mul_value = self.tik_instance.Scalar("float16", "mul_value", init_value=-1)
        self.set_mul_scalar_value(mul_value)

        batch_ranges = self.tik_instance.Scalar("int32", "batch_ranges", init_value=0)
        with self.tik_instance.if_scope(block_idx < used_core_num - 1):
            batch_ranges.set_as(batch_range_value)
        with self.tik_instance.else_scope():
            batch_ranges.set_as(batch_range_value_last)

        with self.tik_instance.for_range(0, batch_ranges) as cur_b_idx:
            # tensor_b_l1 for front_bmm
            front_bmm_tensor_b_l1 = self.apply_buffer_for_front_bmm_tensor_b_and_move_data_in(block_idx, cur_b_idx)
            front_bmm_m_range_value = self.front_bmm_x1_shape[2] // 4
            front_bmm_m_range_once_m_size = 2
            with self.tik_instance.new_stmt_scope(disable_sync=False):
                front_bmm_tensor_b_l0b = self.apply_buffer_for_front_bmm_l0b_and_move_data_in(front_bmm_tensor_b_l1)
                with self.tik_instance.for_range(0, front_bmm_m_range_value)as cur_om_idx:
                    front_bmm_preload_buffers = [front_bmm_tensor_b_l1, front_bmm_tensor_b_l0b]
                    range_idxs = [block_idx, cur_b_idx, cur_om_idx]
                    self.front_bmm_compute_for_outer_m_once(front_bmm_preload_buffers, front_bmm_m_range_once_m_size,
                                                            range_idxs)

            first_bmm_tensor_b = self.first_bmm_move_tensor_b_from_gm_to_l1(block_idx, batch_range_value, cur_b_idx)
            second_bmm_tensor_b = self.second_bmm_move_tensor_b_from_gm_to_l1(block_idx, cur_b_idx)

            first_bmm_tensor_c = self.apply_buffer_for_tensor_c_l1()
            with self.tik_instance.for_range(0, outer_m_range_value) as cur_om_idx:
                first_bmm_tensor_b_l0b = self.apply_buffer_for_tensor_b_l0_and_move_data_in(first_bmm_tensor_b)
                first_preload_buffers = [first_bmm_tensor_b, first_bmm_tensor_c,
                                         first_bmm_tensor_b_l0b, front_bmm_tensor_b_l1]
                range_idxs = [block_idx, cur_b_idx, cur_om_idx]
                with self.tik_instance.new_stmt_scope(disable_sync=False):
                    self.first_bmm_compute_for_outer_m_once(first_preload_buffers, range_idxs,
                                                            outer_m_range_once_m_size, inner_m_range_value, mul_value)
                second_bmm_tensor_b_l0b = first_bmm_tensor_b_l0b
                second_bmm_tensor_c = first_bmm_tensor_c
                second_bmm_preload_buffers = [second_bmm_tensor_b, second_bmm_tensor_b_l0b, second_bmm_tensor_c]
                target_output = self.query_dx_gm
                self.second_bmm_compute_for_outer_m_once(second_bmm_preload_buffers, range_idxs,
                                                         outer_m_range_once_m_size, inner_m_range_value, target_output)

            third_bmm_tensor_b = self.third_bmm_move_tensor_b_from_gm_to_l1(block_idx, cur_b_idx)
            with self.tik_instance.for_range(0, outer_m_range_value) as cur_om_idx:
                third_bmm_tensor_b_l0b = self.tik_instance.Tensor(self.matmul_dtype,
                                                                  [self.first_k_dim, self.first_n_dim,
                                                                   self.block_num, self.block_num],
                                                                   name="third_bmm_tensor_b_l0b",
                                                                   scope=self.tik.scope_cb)
                third_bmm_tensor_c = first_bmm_tensor_c
                third_bmm_preload_buffers = [third_bmm_tensor_b, third_bmm_tensor_b_l0b, third_bmm_tensor_c]
                range_idxs = [block_idx, cur_b_idx, cur_om_idx]
                target_output = self.key_dw_gm
                self.second_bmm_compute_for_outer_m_once(third_bmm_preload_buffers, range_idxs,
                                                         outer_m_range_once_m_size, inner_m_range_value, target_output)

    def compute_process(self):
        batch_range_value, batch_range_value_last, used_core_num = \
            self.get_eatch_core_batch(self.x1_shape[0], self.x1_shape[1], self.cur_op_core_num)
        with self.tik_instance.for_range(0, used_core_num, block_num=used_core_num) as block_idx:
            self.compute_one_core(block_idx, batch_range_value, batch_range_value_last, used_core_num)

        input_gm_list = [self.attention_score_gm, self.x1_gm, self.x2_gm, self.x3_gm, self.x4_gm,
                         self.mul_gm, self.drop_mask_gm]
        output_gm_list = [self.value_dw_gm, self.query_dx_gm, self.key_dw_gm]

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=input_gm_list,
                                   outputs=output_gm_list, config={})


@register_operator("AttentionScoreGrad")
# 'pylint: disable=redefined-builtin
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            (para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_LIST_INT),
                            para_check.KERNEL_NAME)
def attention_score_grad(attention_score, dx, query, key, value, scale,
                         drop_mask,
                         value_dw, query_dx, key_dw,
                         keep_prob,
                         query_transpose=False, key_transpose=False,
                         value_transpose=False, dx_transpose=False,
                         softmax_axes=-1,
                         kernel_name="attention_score_grad"):
    """
    algorithm: mat_mul_softmax_dropout_matmul
    calculating distence mtr.

    Parameters
    ----------
    attention_score : dict. shape and dtype of input, means the all neighbour coords, only support float16
    dx : dict. shape and dtype of input, means the all neighbour types, only support float16
    query: dict. shape and dtype of input, contains the nloc and nall value, only support float16
    key: dict. shape and dtype of input
    value: dict. shape and dtype of input, the input data contains the neighbour coords, only support float16.
    scale: dict. shape and dtype of input, only support float16.
    drop_mask: dict. shape and dtype of input, only support uint8.
    value_dw: dict. shape and dtype of output, only support float16.
    query_dx: dict. shape and dtype of output, only support float16.
    key_dw: dict. shape and dtype of output, only support float16.
    keep_prob: float. only support float value.
    query_transpose: bool.
    key_transpose: bool.
    value_transpose: bool.
    dx_transpose: bool.
    softmax_axes: int or listint.
    kernel_name : str cce kernel name, default value is attention_score_grad

    Returns
    -------
    None
    """
    op_init = AttentionScoreGrad(attention_score, dx, query, key, value, scale, drop_mask,
                                 value_dw, query_dx, key_dw, keep_prob, query_transpose,
                                 key_transpose, value_transpose, dx_transpose, softmax_axes,
                                 kernel_name)
    op_init.compute_process()
