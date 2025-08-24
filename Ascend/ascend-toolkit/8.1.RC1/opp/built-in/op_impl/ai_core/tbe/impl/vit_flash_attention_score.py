"""
Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

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
from functools import reduce
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik


# 'pylint: disable=too-many-arguments,too-many-locals
# 'pylint: disable=too-few-public-methods
# 'pylint: disable=too-many-statements, too-many-lines
# 'pylint: disable=too-many-public-methods
class Constant:
    """
    constant of vit_attention_score
    """
    H_DIM_THRESHOLD = 512
    DICHOTOMY = 2
    DUP_INIT_MIN_VALUE = -60000
    DUP_INIT_ZERO_VALUE = 0
    FP16_BYTES = 2
    FP32_BYTES = 4
    BLOCK_BYTES = 32
    FP16_MAX_MASK = 128
    FP32_MAX_MASK = 64
    BLOCK_LINE_HEIGHT = 1
    CUBE_BLOCK = 16
    CUBE_BLOCK_SIZE = 256  # 16 * 16
    CUBE_BLOCK_BLOCK_NUMS = 16  # 16 * 16 * 2 // 32
    CUBE_FP32_BLOCK_BLOCK_NUMS = 32  # 16 * 16 * 4 // 32
    CUBE_MATRIX_MAX_SIZE = 32768  # 32KB
    CUBE_MATRIX_NUM_MAX_SIZE = 16384  # 32KB // 2Bytes
    CUBE_MATRIX_BLOCK_NUM = 64  # 32KB // 2Bytes // 256
    TILING_MAX_M = 8  # spliting m axis max number is 8 (128)
    MAX_REPEAT_TIMES = 255
    SINGLE_BUFFER = 1
    DOUBLE_BUFFER = 2
    BUFFER_SWITCH_BLOCK_NUM = 7


# 'pylint: disable=too-many-arguments, unused-argument
def check_supported_vit(query, key, value, add_x1, add_x2, scale, drop_mask):
    """checked input data whether is ViT model, 
    and whether ND input shape H axis is larger than 512
    """
    ori_shape = query['ori_shape']
    ori_h_axis = ori_shape[2]

    vit_struc = True
    gpt_struc = False
    swin_struc = False
    if add_x1 is not None:
        vit_struc = False
        ele_shape1 = add_x1["shape"]
        if ele_shape1[0] == ele_shape1[1] == 1:
            gpt_struc = True
    if add_x2 is not None:
        swin_struc = True

    return vit_struc and ori_h_axis > Constant.H_DIM_THRESHOLD


class VitFlashAttentionScore:
    """
    VitFlashAttentionScore class
    y = SoftMax(QK^T / sqrt(d_k))V
    """

    # 'pylint: disable=too-many-arguments, unused-argument
    def __init__(self, query, key, scale, add_x1, add_x2, drop_mask, value,
                 softmax_output, y, input_keep_prob, softmax_axes,
                 query_transpose, key_transpose,
                 bmm_score_transpose_a, bmm_score_transpose_b, kernel_name):
        self.tik_instance = tik.Tik()
        self.cur_op_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.matmul_dtype = "float16"
        self.vector_dtype = "float16"
        self.tmp_vector_dtype = "float32"
        self.query_shape = query["shape"]
        self.key_shape = key["shape"]
        self.value_shape = value["shape"]
        self.input_shape_ori = query["ori_shape"]
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

        self.input_keep_prob = input_keep_prob
        self.double_factor = 2
        self.block_num = 16
        self.buffer_num = Constant.SINGLE_BUFFER

        self.mul_shape = [self.block_num]
        self.mul_scalar = self.tik_instance.Scalar(self.vector_dtype, "mul_scalar", init_value=1)
        self.mul_scalar_fp32 = self.tik_instance.Scalar(self.tmp_vector_dtype, "mul_scalar_fp32")
        self.kernel_name = kernel_name

        self.parse_matrix_shape()
        self.init_gm()
        self.init_mul_scalar()
        self.batch_tiling()
        self.matrix_tiling_single_batch()

    def init_gm(self):
        """init GM variables
        """
        self.query_gm = self.tik_instance.Tensor(self.matmul_dtype, self.query_shape, name="query_gm",
                                                 scope=tbe_platform.scope_gm)
        self.key_gm = self.tik_instance.Tensor(self.matmul_dtype, self.key_shape, name="key_gm",
                                               scope=tbe_platform.scope_gm)
        self.value_gm = self.tik_instance.Tensor(self.matmul_dtype, self.value_shape, name="value_gm",
                                                 scope=tbe_platform.scope_gm)
        self.mul_gm = self.tik_instance.Tensor(self.matmul_dtype, self.mul_shape,
                                               name="mul_gm", scope=tbe_platform.scope_gm)
        if not self.vit_struc:
            self.add1_gm = self.tik_instance.Tensor(self.matmul_dtype, self.ele_shape1,
                                                    name="add1_gm", scope=tbe_platform.scope_gm)
        if self.swin_struc:
            self.add2_gm = self.tik_instance.Tensor(self.matmul_dtype, self.ele_shape2,
                                                    name="add2_gm", scope=tbe_platform.scope_gm)
            self.drop_mask_gm = self.tik_instance.Tensor("uint8", self.drop_shape,
                                                         name="drop_mask_gm", scope=tbe_platform.scope_gm)

        self.y_gm = self.tik_instance.Tensor(self.matmul_dtype, self.y_shape, name="y_gm",
                                             scope=tbe_platform.scope_gm)

    def init_mul_scalar(self):
        """get scale value in ViT
        """
        with self.tik_instance.new_stmt_scope():
            mul_ub = self.tik_instance.Tensor(self.vector_dtype, self.mul_shape, name='mul_ub',
                                              scope=tbe_platform.scope_ubuf)
            self.tik_instance.data_move(mul_ub, self.mul_gm, 0, 1, 1, 0, 0)
            self.mul_scalar.set_as(mul_ub[0])
            self.tik_instance.scalar_conv("", self.mul_scalar_fp32, self.mul_scalar)

    def parse_matrix_shape(self):
        """get matrix shape for matmul
        matmul shape info: (m, k) x (k, n) --> (m, n)
        input format: NCHW --> FRACTAL_NZ(N, C, W1, H1, H0, W0)
        """
        self.first_m_dim = self.query_shape[3]   # H1
        self.first_k_dim = self.query_shape[2]   # W1
        self.first_n_dim = self.key_shape[3]     # H1 --> (H1, H1)
        self.second_m_dim = self.query_shape[3]  # H1
        self.second_k_dim = self.key_shape[3]    # H1
        self.second_n_dim = self.value_shape[2]  # W1 --> (H1, W1)
        self.batch_block_size = self.first_k_dim * self.first_m_dim * 16 * 16  # W1 * H1 * 16 * 16

    def batch_tiling(self):
        """calculate batches distributed per core
        """
        self.total_batch = self.query_shape[0] * self.query_shape[1]  # total_batch = N * C
        self.batch_per_core = (self.total_batch + self.cur_op_core_num - 1) // self.cur_op_core_num
        # when batches cannot be evenly distributed to each core, the last few cores will process one less batch
        self.batch_small_per_core = self.batch_per_core - 1
        # cores in the front whose need to process self.batch_per_core batches
        self.batch_large_core_num = self.total_batch - self.batch_small_per_core * self.cur_op_core_num
        self.large_core_end_batch = self.batch_large_core_num * self.batch_per_core

    def matrix_tiling_single_batch(self):
        """Matrix tiling strategy in a single batch
        (m, k) x (k, n) --> (m, n), split the m axis when ** k <= 1024 ** in ND format

        1. 16 <= m <= 128 in ND format, or 1 <= m <= 8 in FRACTAL_NZ format;
        2. m * k <= 16384 (32KB) in float16 data type and ND format, or m * k <= 64 in FRACTAL_NZ format.

        first_m_dim = split_m_count * split_m + last_split_m

        when ND origin input shape H axis is not aligned by 16, 
        tail is the unaligned data length, pad_tail is transdata padded data length
        """
        self.tail = self.input_shape_ori[2] % self.block_num
        self.tail_pad = 0 if self.tail == 0 else self.block_num - self.tail

        if self.first_m_dim < Constant.TILING_MAX_M \
           and self.first_m_dim * self.first_k_dim <= Constant.CUBE_MATRIX_BLOCK_NUM:
            self.split_m = self.first_m_dim if self.tail == 0 else self.first_m_dim - 1
            self.last_split_m = 0 if self.tail == 0 else 1
            self.split_m_count = 1
        else:
            m_len = Constant.CUBE_MATRIX_BLOCK_NUM // self.first_k_dim
            if m_len >= Constant.TILING_MAX_M:
                self.split_m = Constant.TILING_MAX_M
            else:
                self.split_m = m_len
            self.split_m_count = self.first_m_dim // self.split_m
            self.last_split_m = self.first_m_dim % self.split_m
        self.splited_batch_offset_size = self.split_m * Constant.CUBE_BLOCK_SIZE

        if self.first_k_dim <= Constant.BUFFER_SWITCH_BLOCK_NUM:
            self.buffer_num = Constant.DOUBLE_BUFFER

    def flash_attention(self, cur_batch_idx):
        """compute flash attention in single batch: y = SoftMax(QK^T / sqrt(d_k))V
        1. S = mmad(query, key.T)
        2. P = Softmax(S)
        3. y = mmad(P, value)

        Parameters:
        -----------
        cur_batch_idx: Scalar
            the index of current batch
        """
        # current HW batch start offset
        cur_batch_start_offset = self.tik_instance.Scalar("int32", name='cur_batch_start_offset')
        cur_batch_start_offset.set_as(cur_batch_idx * self.batch_block_size)
        outer_loop_offset = self.tik_instance.Scalar("int32", name='outer_loop_offset')
        inner_loop_offset = self.tik_instance.Scalar("int32", name='inner_loop_offset')

        # 1. out aligned loop
        with self.tik_instance.new_stmt_scope():
            first_mm_l_input_block_shape, first_mm_r_input_block_shape, first_mm_result_block_shape,\
            first_mm_last_r_input_block_shape, first_mm_last_result_block_shape,\
            second_mm_l_input_block_shape, second_mm_r_input_block_shape,\
            second_mm_last_l_input_block_shape, second_mm_last_r_input_block_shape, second_mm_result_block_shape,\
            output_shape, line_vector_shape = self.prepare_shapes()

            shapes = (line_vector_shape, output_shape)
            line_sum_ub, _line_sum_ub, line_max_ub, _line_max_ub, output_ub = self.prepare_global_variables(shapes)

            # 1.1 split Query to split_m and repeat split_m_count times
            with self.tik_instance.for_range(0, self.split_m_count, thread_num=self.buffer_num) as outer_loop_idx:
                # current loop query_gm aligned start offset
                outer_loop_offset.set_as(cur_batch_start_offset + outer_loop_idx * self.splited_batch_offset_size)

                self.init_vector_by_value(line_sum_ub, Constant.DUP_INIT_ZERO_VALUE)
                self.init_vector_by_value(line_max_ub, Constant.DUP_INIT_MIN_VALUE)
                self.init_vector_by_value(output_ub, Constant.DUP_INIT_ZERO_VALUE)
                outer_matrix_shape = (first_mm_l_input_block_shape, first_mm_r_input_block_shape,
                                      first_mm_result_block_shape, second_mm_l_input_block_shape,
                                      second_mm_r_input_block_shape, second_mm_result_block_shape,
                                      first_mm_last_r_input_block_shape, first_mm_last_result_block_shape,
                                      second_mm_last_l_input_block_shape, second_mm_last_r_input_block_shape)
                matrixes = output_ub
                vectors = (line_max_ub, _line_max_ub, line_sum_ub, _line_sum_ub)
                self.flash_attention_in_outer_loop(outer_matrix_shape, line_vector_shape, cur_batch_start_offset,
                                                   matrixes, vectors, outer_loop_offset, inner_loop_offset)
                # 1.2 move aligned output to y_gm
                self.move_output_2_gm(output_ub, line_sum_ub, outer_loop_offset)

        # 1.3. unaligned part, deal with Query's last_split_m
        if self.last_split_m > 0:
            # query_gm last unaligned part start offset
            outer_loop_offset.set_as(cur_batch_start_offset + self.split_m_count * self.splited_batch_offset_size)

            first_mm_l_input_block_shape, first_mm_r_input_block_shape, first_mm_result_block_shape,\
            first_mm_last_r_input_block_shape, first_mm_last_result_block_shape,\
            second_mm_l_input_block_shape, second_mm_r_input_block_shape,\
            second_mm_last_l_input_block_shape, second_mm_last_r_input_block_shape, second_mm_result_block_shape,\
            output_shape, line_vector_shape = self.prepare_shapes(is_last_block=True)

            shapes = (line_vector_shape, output_shape)
            line_sum_ub, _line_sum_ub, line_max_ub, _line_max_ub, output_ub = self.prepare_global_variables(shapes)
            self.init_vector_by_value(line_sum_ub, Constant.DUP_INIT_ZERO_VALUE)
            self.init_vector_by_value(line_max_ub, Constant.DUP_INIT_MIN_VALUE)
            self.init_vector_by_value(output_ub, Constant.DUP_INIT_ZERO_VALUE)

            # prepare outer loop parameters
            outer_matrix_shape = (first_mm_l_input_block_shape, first_mm_r_input_block_shape,
                                  first_mm_result_block_shape, second_mm_l_input_block_shape,
                                  second_mm_r_input_block_shape, second_mm_result_block_shape,
                                  first_mm_last_r_input_block_shape, first_mm_last_result_block_shape,
                                  second_mm_last_l_input_block_shape, second_mm_last_r_input_block_shape)
            matrixes = output_ub
            vectors = (line_max_ub, _line_max_ub, line_sum_ub, _line_sum_ub)
            self.flash_attention_in_outer_loop(outer_matrix_shape, line_vector_shape, cur_batch_start_offset,
                                               matrixes, vectors, outer_loop_offset, inner_loop_offset)
            # 1.4 move unaligned output to y_gm
            self.move_output_2_gm(output_ub, line_sum_ub, outer_loop_offset)

    # 'pylint: disable=too-many-arguments,too-many-locals
    def flash_attention_in_outer_loop(self, matrix_shape, vector_shape, cur_batch_start_offset, global_matrix,
                                      global_vector, outer_loop_offset, inner_loop_offset):
        """single time flash_attention calculation of outer loop
        split query matrix
        the vector_shape == matrix_shape[0] in ND format, or vector_shape == matrix_shape[1] * 16 in Nz format

        Parameters:
        -----------
        matrix_shape: tuple or list
            the shape of query, key, value and output matrixes
            includes:
                (first_mm_l_input_block_shape, first_mm_r_input_block_shape,
                 first_mm_result_block_shape, second_mm_l_input_block_shape,
                 second_mm_r_input_block_shape, second_mm_result_block_shape,
                 first_mm_last_r_input_block_shape, first_mm_last_result_block_shape,
                 second_mm_last_l_input_block_shape, second_mm_last_r_input_block_shape)
        vector_shape: tuple or list
            the shape of line_sum and line_max vectors
        cur_batch_start_offset: Scalar
            current batch data's start offset in GM
        global_matrix: Tensors(matrixes)
            Tensors(matrix) defind in the upper flash_attention method
            include: (output_ub)
        global_vector: tuple or list of Tensor(vectors)
            Tensors(vectors) defind in the upper flash_attention method
            include:
                (line_max_ub, _line_max_ub, line_sum_ub, _line_sum_ub)
        outer_loop_offset: Scalar
            outer loop data start offset
        inner_loop_offset: Scalar
            the placehold of inner loop data start offset, calculate it in this method
        """
        first_mm_l_input_block_shape, first_mm_r_input_block_shape, \
        first_mm_result_block_shape, second_mm_l_input_block_shape, \
        second_mm_r_input_block_shape, second_mm_result_block_shape, \
        first_mm_last_r_input_block_shape, first_mm_last_result_block_shape, \
        second_mm_last_l_input_block_shape, second_mm_last_r_input_block_shape = matrix_shape
        output_ub = global_matrix
        line_max_ub, _line_max_ub, line_sum_ub, _line_sum_ub = global_vector

        first_mm_l_l1 = self.tik_instance.Tensor(self.matmul_dtype, first_mm_l_input_block_shape,
                                                 name='first_mm_l_l1', scope=tbe_platform.scope_cbuf)
        self.move_gm_2_l1(first_mm_l_l1, self.query_gm, outer_loop_offset)

        # 2. inner loop:
        # 2.1 split Key and Value to split_m and repeat split_m_count times
        with self.tik_instance.for_range(0, self.split_m_count, thread_num=self.buffer_num) as inner_loop_idx:
            inner_loop_offset.set_as(cur_batch_start_offset + inner_loop_idx * self.splited_batch_offset_size)
            # prepare inner loop parameters
            inner_matrix_shape = (first_mm_l_input_block_shape, first_mm_r_input_block_shape,
                                  first_mm_result_block_shape, second_mm_l_input_block_shape,
                                  second_mm_r_input_block_shape, second_mm_result_block_shape)
            matrixes = (first_mm_l_l1, output_ub)
            vectors = (line_max_ub, _line_max_ub, line_sum_ub, _line_sum_ub)

            self.flash_attention_in_inner_loop(inner_matrix_shape, vector_shape,
                                               matrixes, vectors, inner_loop_offset)

        # 2.2 deal with key and value last_split_m
        if self.last_split_m > 0:
            inner_loop_offset.set_as(cur_batch_start_offset + self.split_m_count * self.splited_batch_offset_size)
            # prepare inner loop parameters
            inner_matrix_shape = (first_mm_l_input_block_shape, first_mm_last_r_input_block_shape,
                                  first_mm_last_result_block_shape, second_mm_last_l_input_block_shape,
                                  second_mm_last_r_input_block_shape, second_mm_result_block_shape)
            matrixes = (first_mm_l_l1, output_ub)
            vectors = (line_max_ub, _line_max_ub, line_sum_ub, _line_sum_ub)

            self.flash_attention_in_inner_loop(inner_matrix_shape, vector_shape,
                                               matrixes, vectors, inner_loop_offset,
                                               is_tail=True)

    # 'pylint: disable=too-many-arguments,too-many-locals
    def flash_attention_in_inner_loop(self, matrix_shape, vector_shape, outer_matrixes,
                                      outer_vectors, inner_loop_offset, is_tail=False):
        """single time flash_attention calculation of inner loop
        split key and query matrix,
        the vector_shape == matrix_shape[0] in ND format, or vector_shape == matrix_shape[1] * 16 in Nz format
        compute (1)S = Q * K.T; (2)P = softmax(S); (3)O = P * V

        Parameters:
        -----------
        matrix_shape: tuple or list
            the shape of query, key, value and output matrixes,
            include: 
                (first_mm_l_input_block_shape, first_mm_r_input_block_shape,
                 first_mm_result_block_shape, second_mm_l_input_block_shape,
                 second_mm_r_input_block_shape, second_mm_result_block_shape)
        vector_shape: tuple or list
            thw shape of line_sum and line_max vectors
        outer_matrixes: list or tuple of Tensores(matrixes)
            Tensors(matrix) defind in the outer loop
            include:
                (first_mm_l_l1, output_ub)
        outer_vectors: list or tuple of Tensors(vectors)
            Tensors(vectors) defind in the out loop
            include:
                (line_max_ub, _line_max_ub, line_sum_ub, _line_sum_ub)
        inner_loop_offset: Scalar
            inner loop data start offset
        is_tail: bool
            if is_tail==True, do mask_softmax for the first matmul result
        """
        first_mm_l_input_block_shape, first_mm_r_input_block_shape, first_mm_result_block_shape, \
        second_mm_l_input_block_shape, second_mm_r_input_block_shape, second_mm_result_block_shape = matrix_shape
        # get outer input matrixes
        first_mm_l_l1, output_ub = outer_matrixes
        # get outer input vectors
        line_max_ub, _line_max_ub, line_sum_ub, _line_sum_ub = outer_vectors
        first_mm_r_l1 = self.tik_instance.Tensor(self.matmul_dtype, first_mm_r_input_block_shape,
                                                 name='first_mm_r_l1', scope=tbe_platform.scope_cbuf)
        first_mm_result_l0c = self.tik_instance.Tensor(self.tmp_vector_dtype, first_mm_result_block_shape,
                                                       name='first_mm_result_l0c', scope=tbe_platform.scope_cc)
        second_mm_l_l1 = self.tik_instance.Tensor(self.matmul_dtype, second_mm_l_input_block_shape,
                                                  name='second_mm_l_l1', scope=tbe_platform.scope_cbuf)
        second_mm_r_l1 = self.tik_instance.Tensor(self.matmul_dtype, second_mm_r_input_block_shape,
                                                  name='second_mm_r_l1', scope=tbe_platform.scope_cbuf)
        second_mm_result_l0c = self.tik_instance.Tensor(self.tmp_vector_dtype, second_mm_result_block_shape,
                                                        name='second_mm_result_l0c', scope=tbe_platform.scope_cc)
        # use new_stmt_scope to reduce the occupancy of UB space
        with self.tik_instance.new_stmt_scope():
            first_mm_l_l0a = self.tik_instance.Tensor(self.matmul_dtype, first_mm_l_input_block_shape,
                                                      name='first_mm_l_l0a', scope=tbe_platform.scope_ca)
        
            first_mm_r_l0b = self.tik_instance.Tensor(self.matmul_dtype, first_mm_r_input_block_shape,
                                                      name='first_mm_r_l0b', scope=tbe_platform.scope_cb)
            self.move_gm_2_l1(first_mm_r_l1, self.key_gm, inner_loop_offset, is_transpose=True)
            self.load_l1_2_l0(first_mm_l_l0a, first_mm_l_l1)
            self.load_l1_2_l0(first_mm_r_l0b, first_mm_r_l1)
            # first matmul: S = mmad(query, key.T)
            self.matmul(first_mm_result_l0c, first_mm_l_l0a, first_mm_r_l0b, is_r_transpose=True)
            # 3. compute SoftMax(S) iteratively
            # 3.1 S = tor * S
            first_mm_result_ub = self.tik_instance.Tensor(self.tmp_vector_dtype, first_mm_result_block_shape,
                                                          name='first_mm_result_ub', scope=tbe_platform.scope_ubuf)
            self.move_l0c_2_ub(first_mm_result_ub, first_mm_result_l0c)
            self.multiply_scalar(first_mm_result_ub, first_mm_result_ub, self.mul_scalar_fp32)
            # if is_tail==True, do mask_softmax for the first matmul result
            if is_tail and self.tail_pad > 0:
                self.mask_softmax_input(first_mm_result_ub)
            # 3.2 get S reduce max (m) by row
            self.get_line_max(_line_max_ub, first_mm_result_ub)
            # 3.3 get max value (m_bar) between line_max and _line_max by element
            self.update_line_max(_line_max_ub, line_max_ub, _line_max_ub)
            # 3.4 _S = S - _line_max (S - m_bar)
            self.broadcast_sub(first_mm_result_ub, first_mm_result_ub, _line_max_ub)
            # 3.5 _S = exp(_S)
            self.exp_fp32(first_mm_result_ub, first_mm_result_ub)
            # 3.6 P = cast_fp32_2_fp16(_S)
            second_mm_l_ub = self.tik_instance.Tensor(self.matmul_dtype, second_mm_l_input_block_shape,
                                                      name='second_mm_l_ub', scope=tbe_platform.scope_ubuf)
            self.cast_fp32_2_fp16(second_mm_l_ub, first_mm_result_ub)
            # 3.7 l_b =rowsum(_S): _line_sum = rowsum(_S)
            self.get_line_sum(_line_sum_ub, first_mm_result_ub)
            self.move_ub_2_l1(second_mm_l_l1, second_mm_l_ub)
        second_mm_l_l0a = self.tik_instance.Tensor(self.matmul_dtype, second_mm_l_input_block_shape,
                                                   name='second_mm_l_l0a', scope=tbe_platform.scope_ca)
        second_mm_r_l0b = self.tik_instance.Tensor(self.matmul_dtype, second_mm_r_input_block_shape,
                                                   name='second_mm_r_l0b', scope=tbe_platform.scope_cb)
        self.move_gm_2_l1(second_mm_r_l1, self.value_gm, inner_loop_offset)
        self.load_l1_2_l0(second_mm_l_l0a, second_mm_l_l1)
        self.load_l1_2_l0(second_mm_r_l0b, second_mm_r_l1, is_transpose=True)
        # 3.8 second matmul: O = mmad(P, V) FP32
        self.matmul(second_mm_result_l0c, second_mm_l_l0a, second_mm_r_l0b)
        second_mm_result_ub = self.tik_instance.Tensor(self.tmp_vector_dtype, second_mm_result_block_shape,
                                                       name='second_mm_result_ub', scope=tbe_platform.scope_ubuf)
        self.move_l0c_2_ub(second_mm_result_ub, second_mm_result_l0c)
        # 3.9 __line_max_ub = line_max - _line_max (m_dot = m - m_bar)
        __line_max_ub = self.tik_instance.Tensor(self.tmp_vector_dtype, vector_shape,
                                                 name='__line_max_ub', scope=tbe_platform.scope_ubuf)  # FP32
        ele_num = __line_max_ub.shape[0]
        self.element_wise_compute("sub", __line_max_ub, line_max_ub, _line_max_ub, ele_num, 0, 0, 0)
        # 3.10 m_dot = exp(m_dot)
        self.exp_fp32(__line_max_ub, __line_max_ub)
        # 3.11 l = l_b + m_dot * l
        self.update_line_sum(line_sum_ub, _line_sum_ub, __line_max_ub)
        # 3.12 O = O_b + m_dot * O
        self.update_output(output_ub, second_mm_result_ub, __line_max_ub)
        # 3.13 m = m_bar
        self.tik_instance.data_move(line_max_ub, _line_max_ub, 0, 1,
                                    vector_shape[0] * Constant.FP32_BYTES // Constant.BLOCK_BYTES, 0, 0)

    def prepare_global_variables(self, shapes):
        """prepare global matrixes and vectors

        Parameters:
        -----------
        shapes: tuple or list of shape
            global matrixes and vectors' shape, include: (line_vector_shape, output_shape)

        Return:
        -------
        tuple of global variables
        """
        line_vector_shape, output_shape = shapes

        # tensors for softmax
        line_sum_ub = self.tik_instance.Tensor(self.tmp_vector_dtype, line_vector_shape,
                                               name='line_sum', scope=tbe_platform.scope_ubuf)  # l, FP32
        _line_sum_ub = self.tik_instance.Tensor(self.tmp_vector_dtype, line_vector_shape,
                                                name='_line_sum_ub', scope=tbe_platform.scope_ubuf)  # FP32
        line_max_ub = self.tik_instance.Tensor(self.tmp_vector_dtype, line_vector_shape,
                                               name='line_max', scope=tbe_platform.scope_ubuf)  # m, FP32
        _line_max_ub = self.tik_instance.Tensor(self.tmp_vector_dtype, line_vector_shape,
                                                name='_line_max_ub', scope=tbe_platform.scope_ubuf)  # FP32
        # tensor for output
        output_ub = self.tik_instance.Tensor(self.tmp_vector_dtype, output_shape,
                                             name='output_ub', scope=tbe_platform.scope_ubuf)  # O, FP32

        variables = (line_sum_ub, _line_sum_ub, line_max_ub, _line_max_ub, output_ub)
        return variables

    def prepare_shapes(self, is_last_block=False):
        """prepare shape for matrixes and vectors
        
        Parameters:
        -----------
        is_last_block: bool
            if True, prepare shape for last_split_m part, else for split_m part in loop
        
        Return:
        -------
        a tuple of shapes
        """
        if not is_last_block:
            # query shape
            first_mm_l_input_block_shape = (self.first_k_dim, self.split_m, self.block_num, self.block_num)
            # key shape
            first_mm_r_input_block_shape = (self.first_k_dim, self.split_m, self.block_num, self.block_num)
            # first matmul result shape
            first_mm_result_block_shape = (self.split_m, self.split_m, self.block_num, self.block_num)
            # first matmul last part result shape
            first_mm_last_result_block_shape = (self.last_split_m, self.split_m, self.block_num, self.block_num)
            # row length
            line_vector_shape = (self.split_m * self.block_num, 1)
        else:
            first_mm_l_input_block_shape = (self.first_k_dim, self.last_split_m, self.block_num, self.block_num)
            first_mm_r_input_block_shape = (self.first_k_dim, self.split_m, self.block_num, self.block_num)
            first_mm_result_block_shape = (self.split_m, self.last_split_m, self.block_num, self.block_num)
            first_mm_last_result_block_shape = (self.last_split_m, self.last_split_m, self.block_num, self.block_num)
            line_vector_shape = (self.last_split_m * self.block_num, 1)
        # equal to first matmul result shape
        second_mm_l_input_block_shape = first_mm_result_block_shape
        # value shape, equal to key shape
        second_mm_r_input_block_shape = first_mm_r_input_block_shape
        # last part key shape in inner last_split_m
        first_mm_last_r_input_block_shape = (self.first_k_dim, self.last_split_m, self.block_num, self.block_num)
        # second matmul last part left matrix shape
        second_mm_last_l_input_block_shape = first_mm_last_result_block_shape
        # last part value shape, equal to last part key shape
        second_mm_last_r_input_block_shape = first_mm_last_r_input_block_shape
        # second matmul result shape, equal to query shape, even last part
        second_mm_result_block_shape = first_mm_l_input_block_shape
        output_shape = second_mm_result_block_shape

        shapes = (first_mm_l_input_block_shape, first_mm_r_input_block_shape, first_mm_result_block_shape,
                  first_mm_last_r_input_block_shape, first_mm_last_result_block_shape,
                  second_mm_l_input_block_shape, second_mm_r_input_block_shape,
                  second_mm_last_l_input_block_shape, second_mm_last_r_input_block_shape, second_mm_result_block_shape,
                  output_shape, line_vector_shape)
        return shapes

    def move_gm_2_l1(self, dst, src, src_start_offset, is_transpose=False):
        """move data from GM to L1 in matmul
        if GM data shape is large, split m axis and move to L1

        Parameters:
        -----------
        dst: Tensor
            data_move destination in L1
        src: Tensor
            data_move source in GM
        src_start_offset: Scalar
            src Tensor start offset of batch in current data_move
        is_transpose: bool
            whether to transpose the data
        """
        dst_shape = dst.shape  # (col, row, 16, 16)
        col, row = dst_shape[0], dst_shape[1]
        if is_transpose:
            nburst = col
            burst = row * Constant.CUBE_BLOCK_BLOCK_NUMS
            src_stride = (self.first_m_dim - row) * Constant.CUBE_BLOCK_BLOCK_NUMS
            self.tik_instance.data_move(dst, src[src_start_offset], 0, nburst, burst, src_stride, 0)
        else:
            nburst = col
            burst = Constant.CUBE_BLOCK_BLOCK_NUMS
            src_stride = (self.first_m_dim - 1) * Constant.CUBE_BLOCK_BLOCK_NUMS
            src_offset = self.tik_instance.Scalar('int32', name='src_offset')
            for i in range(row):
                src_offset.set_as(i * Constant.CUBE_BLOCK_SIZE + src_start_offset)
                dst_offset = i * col * Constant.CUBE_BLOCK_SIZE
                self.tik_instance.data_move(dst[dst_offset], src[src_offset], 0, nburst, burst, src_stride, 0)

    def move_ub_2_l1(self, dst, src, is_transpose=False):
        """move data from UB to L1 in matmul
        dst.shape == src.shape

        Parameters:
        -----------
        dst: Tensor
            data_move destination in L1
        src: Tensor
            data_move source in UB
        is_transpose: bool
            whether to transpose the data
        """
        shape = dst.shape  # (col, row, 16, 16)
        col, row = shape[0], shape[1]
        if is_transpose:
            burst = col * row * Constant.CUBE_BLOCK_BLOCK_NUMS
            self.tik_instance.data_move(dst, src, 0, 1, burst, 0, 0)
        else:
            nburst = col
            burst = Constant.CUBE_BLOCK_BLOCK_NUMS
            src_stride = (row - 1) * Constant.CUBE_BLOCK_BLOCK_NUMS
            for i in range(row):
                src_offset = i * Constant.CUBE_BLOCK_SIZE
                dst_offset = i * col * Constant.CUBE_BLOCK_SIZE
                self.tik_instance.data_move(dst[dst_offset], src[src_offset], 0, nburst, burst, src_stride, 0)

    def load_l1_2_l0(self, dst, src, is_transpose=False):
        """load data from L1 to L0A/L0B in matmul

        Parameters:
        ----------
        dst: Tensor[offset]
            load2dv2 destination
        src: Tensor[offset]
            load2dv2 source
        is_transpose: bool
            whether to transpose the matrix in fractal block
        """
        shape = dst.shape
        col, row = shape[0], shape[1]
        repeat_times = col * row
        self.tik_instance.load2dv2(dst, src, 0, repeat_times, 0, 1, 0, is_transpose)

    def move_l0c_2_ub(self, dst, src):
        """move mmad result in L0C to UB

        Parameters:
        -----------
        dst: Tensor
            data_move destination in UB
        src: Tensor
            data_move source in L0C which is mmad result
        """
        shape = src.shape
        burst = shape[0] * shape[1]
        self.tik_instance.data_move(dst, src, 0, 1, burst, 0, 0)

    def move_data_2_gm(self, dst, src, dst_start_offset):
        """move output to gm

        Parameters:
        -----------
        dst: Tensor
            data_move destination in GM
        src: Tensor
            data_move source in UB which is result of FlashAttentionScore
        dst_start_offset: Scalar
            dst Tensor start offset in GM
        """
        shape = src.shape
        col, row = shape[0], shape[1]
        nburst = col
        burst = row * Constant.CUBE_BLOCK_BLOCK_NUMS
        src_stride = 0
        dst_stride = (self.first_m_dim - row) * Constant.CUBE_BLOCK_BLOCK_NUMS
        self.tik_instance.data_move(dst[dst_start_offset], src, 0, nburst, burst, src_stride, dst_stride)

    def move_output_2_gm(self, output, line_sum, outer_offset):
        """update and move output to y_gm
        O = O / l, cast O from FP32 to FP16

        Parameters:
        -----------
        output: Tensor
        line_sum: Tensor
        outer_offset: Scalar
        """
        output_shape = output.shape
        # calculate O = O / l
        self.broadcast_div(output, output, line_sum)
        output_fp16_ub = self.tik_instance.Tensor(self.matmul_dtype, output_shape, name='output_fp16_ub',
                                                  scope=tbe_platform.scope_ubuf)
        self.cast_fp32_2_fp16(output_fp16_ub, output)
        self.move_data_2_gm(self.y_gm, output_fp16_ub, outer_offset)

    def matmul(self, dst, l0a, l0b, is_r_transpose=False):
        """matrix multiply: S = mmad(query, key.T) or O = mmad(P, value)

        Parameters:
        -----------
        dst: Tensor[offset]
            mmad result
        l0a: Tensor[offset]
            mmad left matrix
        l0b: Tensor[offset]
            mmad right matrix
        is_r_transpose: bool
            whether right matrix transposed
        """
        l_shape, r_shape = l0a.shape, l0b.shape  # shape likes [col, row, 16, 16]
        matrix_m = l_shape[1] * Constant.CUBE_BLOCK
        matrix_k = l_shape[0] * Constant.CUBE_BLOCK
        if is_r_transpose:
            matrix_n = r_shape[1] * Constant.CUBE_BLOCK
        else:
            matrix_n = r_shape[0] * Constant.CUBE_BLOCK
        self.tik_instance.mmad(dst, l0a, l0b, matrix_m, matrix_k, matrix_n, 0)

    def multiply_scalar(self, dst, src, scalar):
        """multiply tensor and scalar
        like. S = tor * S, where S is first matmul result, tor is self.mul_scalar

        Parameters:
        -----------
        dst: Tensor
            vec_muls compute result in UB
        src: Tensor
            first matmul result in UB
        scalar: Scalar
            saclar need to multiply
        """
        shape = src.shape
        dtype = src.dtype
        elements = reduce(lambda x, y: x * y, shape)
        mask = Constant.FP32_MAX_MASK if dtype == "float32" else Constant.FP16_MAX_MASK
        repeat_times = elements // mask
        last_elements = elements % mask
        offset = 0

        while repeat_times > Constant.MAX_REPEAT_TIMES:
            self.tik_instance.vec_muls(mask, dst[offset], src[offset], scalar, Constant.MAX_REPEAT_TIMES, 8, 8)
            repeat_times -= Constant.MAX_REPEAT_TIMES
            offset += Constant.MAX_REPEAT_TIMES * mask
        if repeat_times > 0:
            self.tik_instance.vec_muls(mask, dst[offset], src[offset], scalar, repeat_times, 8, 8)
            offset += repeat_times * mask
        if last_elements > 0:
            self.tik_instance.vec_muls(last_elements, dst[offset], src[offset], scalar, 1, 8, 8)

    def mask_softmax_input(self, src):
        """vec_dup a negative large number in first matmul result where number is calculated from pad data,
        keep actual matmul shape, set pad matmul result as -60000, after softmax the pad result will be 0.

        src: Tensor
            first matmul result, which is the input of softmax
        """
        shape = src.shape
        col, row = shape[0], shape[1]
        # calculate the bit mask for vec_dup instruction
        mask = 0
        for i in range(self.tail_pad):
            mask += 2 ** (self.block_num - 1 - i)

        # self.tail_pad is less than 16, so only need to vec_dup for last col in Nz format
        offset = (col - 1) * row * Constant.CUBE_BLOCK_SIZE
        dup_scalar = self.tik_instance.Scalar(src.dtype, name="dup_scalar", init_value=Constant.DUP_INIT_MIN_VALUE)
        repeat_times = row * self.block_num
        dst_rep_stride = self.block_num * Constant.FP32_BYTES // Constant.BLOCK_BYTES
        # vec_dup uses bit mask
        self.tik_instance.vec_dup([0, mask], src[offset], dup_scalar, repeat_times, dst_rep_stride)

    def cast_fp32_2_fp16(self, dst, src):
        """cast data type form fp32 to fp16

        Parameters:
        -----------
        dst: Tensor
            FP16 dtype tensor
        src: Tensor
            FP32 dtype tensor which need to cast
        """
        shape = src.shape
        elements = reduce(lambda x, y: x * y, shape)
        repeat_times = elements // Constant.FP32_MAX_MASK
        last_elements = elements % Constant.FP32_MAX_MASK
        offset = 0

        dst_rep_stride = 4  # mask * 2Bytes // 32Bytes/block = 4block
        src_rep_stride = 8  # mask * 4Bytes // 32Bytes/block = 8block

        while repeat_times > Constant.MAX_REPEAT_TIMES:
            self.tik_instance.vconv(Constant.FP32_MAX_MASK, "", dst[offset], src[offset],
                                    Constant.MAX_REPEAT_TIMES, 1, 1, dst_rep_stride, src_rep_stride)
            repeat_times -= Constant.MAX_REPEAT_TIMES
            offset += Constant.MAX_REPEAT_TIMES * Constant.FP32_MAX_MASK
        if repeat_times > 0:
            self.tik_instance.vconv(Constant.FP32_MAX_MASK, "", dst[offset], src[offset],
                                    repeat_times, 1, 1, dst_rep_stride, src_rep_stride)
            offset += repeat_times * Constant.FP32_MAX_MASK
        if last_elements > 0:
            self.tik_instance.vconv(last_elements, "", dst[offset], src[offset],
                                    1, 1, 1, dst_rep_stride, src_rep_stride)

    def update_line_max(self, dst, src0, src1):
        """compare and get max value between src0 and src1 by elements

        parameter:
        ----------
        dst: Tensor FP32
            max vlaue by elements
        src0: Tensor FP32
        src1: Tensor FP32
        """
        shape = dst.shape
        elements = reduce(lambda x, y: x * y, shape)
        repeat_times = elements // Constant.FP32_MAX_MASK
        last_elements = elements % Constant.FP32_MAX_MASK
        offset = 0

        if repeat_times > 0:
            self.tik_instance.vmax(Constant.FP32_MAX_MASK, dst[offset], src0[offset], src1[offset],
                                   repeat_times, 1, 1, 1, 8, 8, 8)
            offset += repeat_times * Constant.FP32_MAX_MASK
        if last_elements > 0:
            self.tik_instance.vmax(last_elements, dst[offset], src0[offset], src1[offset],
                                   1, 1, 1, 1, 8, 8, 8)

    def get_line_sum(self, dst, src):
        """l_b = rowsum(S_b_tor), get S_b_tor reduce sum by row

        Parameters:
        -----------
        dst: Tensor FP32
            row sum value
        src: Tensor FP32
            matrix need to compute row sum value,
            before calculating line_sum, we had move src(S_b_tor) data to P(for second matmul),
            so src data can be changed, do this inplace
        """
        shape = src.shape
        col, row = shape[0], shape[1]

        while col > 1:
            if col % Constant.DICHOTOMY == 0:  # dichotomy reduce sum
                col = col // Constant.DICHOTOMY
                start = 0
                offset = col * row * self.block_num * self.block_num
                self.element_wise_compute("sum", src, src, src, offset, start, start, start + offset)
            else:  # keep the first part, dichotomy reduce sum the last part
                sum_col = col // Constant.DICHOTOMY  # compute col num for this loop
                col = sum_col + 1  # remain col num for next loop
                start = row * self.block_num * self.block_num
                offset = sum_col * row * self.block_num * self.block_num
                self.element_wise_compute("sum", src, src, src, offset, start, start, start + offset)

        # now col = 1, shape is [row, col, 16, 16] FP32, use vcadd to compute elements sum in each repeat
        mask = self.block_num
        repeat_times = row * self.block_num
        dst_rep_stride = 1  # the unit of dst_rep_stride in vcadd is the dtype's Bytes of dst
        src_blk_stride = 1
        src_rep_stride = 2  # mask * 4Bytes // 32Bytes/block = 2block
        self.tik_instance.vcadd(mask, dst[0], src[0], repeat_times, dst_rep_stride, src_blk_stride, src_rep_stride)
    
    def get_line_max(self, dst, src):
        """m_b = rowmax(S_b), get S reduce max value by row

        Parameters:
        -----------
        dst: Tensor FP32
            row max value
        src: Tensor FP32
            matrix need to get row max value,
            src original data needs to remain unchanged
        """
        shape = src.shape
        col, row = shape[0], shape[1]
        reduce_shape = (1, row, self.block_num, self.block_num)

        with self.tik_instance.new_stmt_scope():
            reduce_max_ub = self.tik_instance.Tensor(self.tmp_vector_dtype, reduce_shape,
                                                     name='reduce_max_ub', scope=tbe_platform.scope_ubuf)

            if col > 1:
                offset = row * Constant.CUBE_BLOCK_SIZE
                repeat_times = offset // Constant.FP32_MAX_MASK
                # compare first two col and save max value in reduce_max_ub
                self.tik_instance.vmax(Constant.FP32_MAX_MASK, reduce_max_ub, src[0], src[offset],
                                       repeat_times, 1, 1, 1, 8, 8, 8)
                # compare reduce_max_ub and each left cols
                for cur_col in range(2, col):
                    self.tik_instance.vmax(Constant.FP32_MAX_MASK, reduce_max_ub, reduce_max_ub,
                                           src[cur_col * offset], repeat_times, 1, 1, 1, 8, 8, 8)
            else:  # col == 1
                burst = row * Constant.CUBE_FP32_BLOCK_BLOCK_NUMS
                self.tik_instance.data_move(reduce_max_ub, src, 0, 1, burst, 0, 0)

            # finally, compare data in the reduce_max_ub, and get max value per line
            mask = offset = self.block_num // 2
            repeat_times = row * self.block_num
            dst_rep_stride = 2  # self.block_num * Constant.FP32_BYTES // Constant.BLOCK_BYTES = 16 * 4 // 32
            src_rep_stride = 2
            self.tik_instance.vmax(mask, reduce_max_ub, reduce_max_ub, reduce_max_ub[offset], repeat_times,
                                   1, 1, 1, dst_rep_stride, src_rep_stride, src_rep_stride)
            # compare each line in one block
            repeat_times = row * 2  # row * 16 * (16 // 2) // 64
            dst_rep_stride = 1
            src_blk_stride = 2  # jump 8 elements
            src_rep_stride = 16  # mask * src_blk_stride * 4Bytes // 32Bytes/block
            self.tik_instance.vcgmax(Constant.FP32_MAX_MASK, dst, reduce_max_ub, repeat_times,
                                     dst_rep_stride, src_blk_stride, src_rep_stride)

    def vector_broadcast(self, dst, src):
        """broadcat a vector (row * 16,) to a matrix (row * 16, 16)

        Parameter:
        ----------
        dst: Tensor
            matrix broadcasted
        src: Tensor
            vector need to broadcast
        """
        dtype = dst.dtype
        vector_len = src.shape[0]
        row = vector_len // self.block_num

        if row > 1:  # when row > 1, use vnchwconv
            broadcast_l0c = self.tik_instance.Tensor(dtype, dst.shape, name='broadcast_l0c',
                                                     scope=tbe_platform.scope_cc)
            self.tik_instance.broadcast_ub_to_l0c(broadcast_l0c, src, 1, row, 1, 1)
            if dtype == "float16":
                self.tik_instance.data_move(dst, broadcast_l0c, 1, 1, row, 0, 0)
                self.tik_instance.vec_trans(dst, dst, row, 1, 1)
            elif dtype == "float32":
                tmp_broadcast_ub = self.tik_instance.Tensor(dtype, dst.shape, name='tmp_broadcast_ub',
                                                            scope=tbe_platform.scope_ubuf)
                self.tik_instance.data_move(tmp_broadcast_ub, broadcast_l0c, 1, 1, row, 0, 0)
                src_list = [tmp_broadcast_ub[i * 16] for i in range(16)]
                dst_list = [dst[i * 8] for i in range(16)]
                self.tik_instance.vnchwconv(False, False, dst_list, src_list, row, 32, 32)

                src_list = [tmp_broadcast_ub[i * 16 + 8] for i in range(16)]
                dst_list = [dst[i * 8 + 128] for i in range(16)]
                self.tik_instance.vnchwconv(False, False, dst_list, src_list, row, 32, 32)
        else:  # when row == 1, use vec_dup
            broadcast_scalar = self.tik_instance.Scalar(dtype, "broadcast_scalar")
            for i in range(vector_len):
                offset = i * self.block_num
                broadcast_scalar.set_as(src[i])
                self.tik_instance.vec_dup(self.block_num, dst[offset], broadcast_scalar, 1, 8)

    def broadcast_sub(self, dst, matrix, subtrahend):
        """dst = matrix - broadcast(subtrahend)
        matrix[i, :] - need_to_broadcast[i] by elements
        matrix.shape[0] == need_to_broadcast.shape[0]

        Parameters:
        -----------
        dst: Tensor FP32
        matrix: Tensor FP32
        subtrahend: Tensor FP32
        """
        matrix_shape = matrix.shape
        col, row = matrix_shape[0], matrix_shape[1]
        broadcast_shape = (1, row, self.block_num, self.block_num)
        with self.tik_instance.new_stmt_scope():
            broadcast_ub = self.tik_instance.Tensor(self.tmp_vector_dtype, broadcast_shape, name='broadcast_ub',
                                                    scope=tbe_platform.scope_ubuf)
            self.vector_broadcast(broadcast_ub, subtrahend)

            # matrix - broadcast
            repeat_times = row * 4  # 1 * row * 16 * 16 // 64
            for i in range(col):
                offset = i * row * Constant.CUBE_BLOCK_SIZE
                self.tik_instance.vsub(Constant.FP32_MAX_MASK, dst[offset], matrix[offset], broadcast_ub[0],
                                       repeat_times, 1, 1, 1, 8, 8, 8)

    def broadcast_div(self, dst, matrix, divisor):
        """dst = matrix / broadcast(divisor)
        matrix[i, :] / need_to_broadcast[i] by elements
        matrix.shape[0] == need_to_broadcast.shape[0]

        Parameters:
        -----------
        dst: Tensor
        matrix: Tensor
        divisor: Tensor
        """
        matrix_shape = matrix.shape
        col, row = matrix_shape[0], matrix_shape[1]
        broadcast_shape = (1, row, self.block_num, self.block_num)
        with self.tik_instance.new_stmt_scope():
            broadcast_ub = self.tik_instance.Tensor(dst.dtype, broadcast_shape, name='broadcast_ub',
                                                    scope=tbe_platform.scope_ubuf)
            self.vector_broadcast(broadcast_ub, divisor)
            # calculate matrix / broadcast(divisor)
            for i in range(col):
                offset = i * row * Constant.CUBE_BLOCK_SIZE
                ele_num = row * Constant.CUBE_BLOCK_SIZE
                self.element_wise_compute("div", dst, matrix, broadcast_ub, ele_num, offset, offset, 0)

    def exp_fp32(self, dst, src):
        """dst = exp(src)

        Parameters:
        ----------
        dst: Tensor FP32
        src: Tensor FP32
        """
        shape = src.shape
        elements = reduce(lambda x, y: x * y, shape)
        repeat_times = elements // Constant.FP32_MAX_MASK
        last_elements = elements % Constant.FP32_MAX_MASK
        offset = 0

        while repeat_times > Constant.MAX_REPEAT_TIMES:
            self.tik_instance.vexp(Constant.FP32_MAX_MASK, dst[offset], src[offset],
                                   Constant.MAX_REPEAT_TIMES, 1, 1, 8, 8)
            repeat_times -= Constant.MAX_REPEAT_TIMES
            offset += Constant.MAX_REPEAT_TIMES * Constant.FP32_MAX_MASK
        if repeat_times > 0:
            self.tik_instance.vexp(Constant.FP32_MAX_MASK, dst[offset], src[offset], repeat_times, 1, 1, 8, 8)
            offset += repeat_times * Constant.FP32_MAX_MASK
        if last_elements > 0:
            self.tik_instance.vexp(last_elements, dst[offset], src[offset], 1, 1, 1, 8, 8)

    def init_vector_by_value(self, dst, init_value):
        """use vec_dup to init a vector by the init_value

        Parameters:
        ----------
        dst: Tensor FP32
        init_value: Scalar
            the number to duplicate
        """
        shape = dst.shape
        dtype = dst.dtype
        elements = reduce(lambda x, y: x * y, shape)
        mask = Constant.FP32_MAX_MASK if dtype == "float32" else Constant.FP16_MAX_MASK
        repeat_times = elements // mask
        last_elements = elements % mask
        offset = 0

        while repeat_times > Constant.MAX_REPEAT_TIMES:
            self.tik_instance.vec_dup(mask, dst[offset], init_value, Constant.MAX_REPEAT_TIMES, 8)
            repeat_times -= Constant.MAX_REPEAT_TIMES
            offset += Constant.MAX_REPEAT_TIMES * mask
        if repeat_times > 0:
            self.tik_instance.vec_dup(mask, dst[offset], init_value, repeat_times, 8)
            offset += repeat_times * mask
        if last_elements > 0:
            self.tik_instance.vec_dup(last_elements, dst[offset], init_value, 1, 8)

    # 'pylint: disable=too-many-arguments,too-many-locals
    def element_wise_compute(self, method, dst, src0, src1, ele_num, dst_offset=0, src0_offset=0, src1_offset=0):
        """dst = method(src0, src1) by elements
        method: String
            compute method, every method maps to a double-operator tik_instance instructions
            like. sum --> vadd, mul --> vmul, sub --> vsub
        dst: Tensor
        src0: Tenosr
        src1: Tensor
        ele_num: int
            number of elements
        dst_offset: int
        src0_offset: int
        src1_offset: int
        """
        method_map = {
            "sum": self.tik_instance.vadd,
            "add": self.tik_instance.vadd,
            "mul": self.tik_instance.vmul,
            "sub": self.tik_instance.vsub,
            "div": self.tik_instance.vdiv
        }
        tik_method = method_map.get(method)
        dtype = dst.dtype
        mask = Constant.FP32_MAX_MASK if dtype == "float32" else Constant.FP16_MAX_MASK
        repeat_times = ele_num // mask
        last_elements = ele_num % mask
        offset = 0

        while repeat_times > Constant.MAX_REPEAT_TIMES:
            tik_method(mask, dst[dst_offset + offset], src0[src0_offset + offset],
                       src1[src1_offset + offset], Constant.MAX_REPEAT_TIMES, 1, 1, 1, 8, 8, 8)
            repeat_times -= Constant.MAX_REPEAT_TIMES
            offset += Constant.MAX_REPEAT_TIMES * mask
        if repeat_times > 0:
            tik_method(mask, dst[dst_offset + offset], src0[src0_offset + offset],
                       src1[src1_offset + offset], repeat_times, 1, 1, 1, 8, 8, 8)
            offset += repeat_times * mask
        if last_elements > 0:
            tik_method(last_elements, dst[dst_offset + offset], src0[src0_offset + offset],
                       src1[src1_offset + offset], 1, 1, 1, 1, 8, 8, 8)

    def update_line_sum(self, line_sum, batch_line_sum, line_max_parameter):
        """update line sum value: l = l_b + m_dot * l

        Paramters:
        ----------
        line_sum: Tensor FP32
            the tensor need to update
        batch_line_sum: Tensor FP32
            rowsum of softmax matrix in inner loop
        line_max_parameter: Tensor FP32
            the parameter about line_max for updating line_sum
        """
        shape = line_sum.shape  # 128,
        ele_num = reduce(lambda x, y: x * y, shape)

        self.element_wise_compute("mul", line_sum, line_max_parameter, line_sum, ele_num, 0, 0, 0)
        self.element_wise_compute("sum", line_sum, batch_line_sum, line_sum, ele_num, 0, 0, 0)

    def update_output(self, output, batch_output, line_max_parameter):
        """update output matrix: O = O_b + m_dot * O

        Paramters:
        ----------
        output: Tensor FP32
            the matrix need to update
        batch_output: Tensor FP32
            second matmul result
        line_max_parameter: Tensor FP32
            the parameter about line_max for updating output
        """
        matrix_shape = output.shape
        col, row = matrix_shape[0], matrix_shape[1]
        broadcast_shape = (1, row, self.block_num, self.block_num)
        with self.tik_instance.new_stmt_scope():
            # broadcast line_max_parameter
            broadcast_ub = self.tik_instance.Tensor(self.tmp_vector_dtype, broadcast_shape, name='broadcast_ub',
                                                    scope=tbe_platform.scope_ubuf)
            self.vector_broadcast(broadcast_ub, line_max_parameter)

            # multiply
            repeat_times = row * 4  # 1 * row * 16 * 16 // 64
            for i in range(col):
                offset = i * row * Constant.CUBE_BLOCK_SIZE
                self.tik_instance.vmul(Constant.FP32_MAX_MASK, output[offset], broadcast_ub[0], output[offset],
                                       repeat_times, 1, 1, 1, 8, 8, 8)
            # add
            ele_num = reduce(lambda x, y: x * y, matrix_shape)
            self.element_wise_compute("sum", output, batch_output, output, ele_num, 0, 0, 0)

    def start_compute(self):
        """
        entry point of SwinFlashAttentionScore tik instance
        distribute batches data to each core and compute in each batch
        """
        with self.tik_instance.for_range(0, self.cur_op_core_num, block_num=self.cur_op_core_num) as core_idx:
            core_batch_offset = self.tik_instance.Scalar("int32", name="core_batch_offset")
            batch_per_core_truth = self.tik_instance.Scalar("int32", name="batch_per_core_truth")
            cur_batch_idx = self.tik_instance.Scalar("int32", name="cur_batch_idx")

            # cores in the front, each core processes self.batch_per_core batches
            with self.tik_instance.if_scope(core_idx < self.batch_large_core_num):
                batch_per_core_truth.set_as(self.batch_per_core)
                core_batch_offset.set_as(core_idx * self.batch_per_core)
            # cores in the back, each core processes self.batch_small_per_core batches
            with self.tik_instance.else_scope():
                batch_per_core_truth.set_as(self.batch_small_per_core)
                core_batch_offset.set_as(self.large_core_end_batch + (
                    core_idx - self.batch_large_core_num) * self.batch_small_per_core)

            # compute in each batch
            with self.tik_instance.for_range(0, batch_per_core_truth) as batch_idx:
                cur_batch_idx.set_as(core_batch_offset + batch_idx)
                # compute flash_attention_score
                self.flash_attention(cur_batch_idx)

        if self.swin_struc:
            input_gm_list = [self.query_gm, self.key_gm, self.value_gm, self.add1_gm, self.add2_gm,
                             self.mul_gm, self.drop_mask_gm]
        elif self.vit_struc:
            input_gm_list = [self.query_gm, self.key_gm, self.value_gm, self.mul_gm]
        else:
            input_gm_list = [self.query_gm, self.key_gm, self.value_gm, self.add1_gm, self.mul_gm]
        output_gm_list = [self.y_gm]
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=input_gm_list,
                                   outputs=output_gm_list, config={})


# 'pylint: disable=redefined-builtin, too-many-arguments
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            (para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_LIST_INT),
                            para_check.KERNEL_NAME)
def vit_flash_attention_score(query, key, value, padding_mask1, padding_mask2, scale, drop_mask,
                              attention_score_output, softmax_output, keep_prob,
                              query_transpose=False, key_transpose=False,
                              bmm_score_transpose_a=False, bmm_score_transpose_b=False,
                              softmax_axes=-1, kernel_name="swin_attention_score"):
    """
    y = SoftMax(Q K^T / sqrt(d_k)) V

    Parameters
    ----------
    query: dict
        shape and dtype of input query, only support FRACTAL_NZ float16
    key: dict
        shape and dtype of input key, only support FRACTAL_NZ float16
    value: dict
        shape and dtype of input value, only support FRACTAL_NZ float16
    padding_mask1: dict (optional)
        shape and dtype of input padding_mask1, only support FRACTAL_NZ float16. Used in GPT model
    padding_mask2: dict (optional)
        shape and dtype of input padding_mask2, only support FRACTAL_NZ float16. Used in Swin model
    scale: dict
        shape and dtype of input scale whisch is the value of sqrt(d_k), only support ND float16
    drop_mask: dict
        shape and dtype of input drop_mask, support uint8 and float16
    attention_score_output: dict
        shape and dtype of output attention_score, only support FRACTAL_NZ float16
    softmax_output: dict (optional)
        shape and dtype of output softmax_output, only support FRACTAL_NZ float16

    keep_prob: float
        the probability of retaining input elements, default is 1.0
    query_transpose: bool
    key_transpose: bool
    bmm_score_transpose_a: bool
    bmm_score_transpose_b: bool
    softmax_axes: list_int
        the axis to softmax, default is [-1]
    kernel_name: str
        cce kernel name, default is "swin_attention_score"

    Returns
    -------
    None
    """
    op_init = VitFlashAttentionScore(query, key, scale, padding_mask1, padding_mask2, drop_mask, value,
                                     softmax_output, attention_score_output, keep_prob, softmax_axes,
                                     query_transpose, key_transpose, bmm_score_transpose_a, bmm_score_transpose_b,
                                     kernel_name)
    op_init.start_compute()
