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
pad_v3_5hd.py
"""
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator
from tbe.common.platform import get_bit_len


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant
    """
    # max int64
    MAX_INT64 = 2**64 - 1
    # tiling param nums
    TILING_NUMS = 32
    # `1 byte = 8 bit`
    EIGHT_BIT = 8
    # 32bits = 4Bytes
    FOUR_BYTES = 4
    # reserved ub size
    RESERVED_UB = 8 * 1024
    MODE0 = 0
    MODE1 = 1
    MODE2 = 2
    MODE3 = 3
    MODE4 = 4
    # the block size
    BLOCK_SIZE = 32
    BLOCK = 16
    MAX_REPEAT_TIME = 4095


# 'pylint: disable=too-many-instance-attributes,too-many-statements,too-many-locals,too-many-lines
# 'pylint: disable=too-many-arguments,invalid-name
class PadV35HDInit:
    """
    Function: class that execute pad_v3_5hd
    """

    def __init__(self, x, paddings, constant_values, y, mode, padding_contiguous=True, kernel_name='pad_v3'):
        """
        init the op
        :param
        x: the input tensor
        :param
        paddings: the list of paddings
        :param
        y: the output of op
        :param
        kernel_name: the kernel name of op
        :return
        None
        """
        self.tik_instance = tik.Tik()
        self.unknown_max_shape = (Constant.MAX_INT64,)
        self.tiling_dtype = "int64"
        self.tiling_shape = (Constant.TILING_NUMS,)
        self.four_bytes_dtype = {"float32", "int32", "uint32"}
        self.two_bytes_dtype = {"bfloat16", "float16", "int16", "uint16"}
        self.x_dtype = x.get("dtype")
        if self.x_dtype in self.two_bytes_dtype:
            self.size = 2  # 4Bytes can put 2 float16
        elif self.x_dtype in self.four_bytes_dtype:
            self.size = 1  # 4Bytes can put 1 float32/int32
        self.inner_dtype = self.x_dtype
        self.paddings_dtype = paddings.get('dtype')
        self.constant_values = constant_values
        if self.constant_values:
            self.constant_values_dtype = constant_values.get('dtype')
        self.y_dtype = y.get('dtype')
        self.mode = mode
        self.padding_contiguous = padding_contiguous
        self.kernel_name = kernel_name
        self.input_gm = None
        self.output_gm = None
        self.tiling_gm = None
        self.input_gm_list = []
        self.output_gm_list = []
        self.input_bytes_size = 0

        self.inner_bytes_size = get_bit_len(self.inner_dtype) // Constant.EIGHT_BIT
        self.block_num = Constant.BLOCK_SIZE // self.inner_bytes_size
        self.dump_mask_max_x = Constant.EIGHT_BIT * self.block_num
        self.max_repeat_time = 255
        self.dump_max_x = self.dump_mask_max_x * self.max_repeat_time

        self.ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - Constant.RESERVED_UB
        self.ub_number = self.ub_size_bytes // self.inner_bytes_size
        self.ub_number_4_bytes = self.ub_size_bytes // Constant.FOUR_BYTES
        self.core_nums = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.pad_scalar = self.tik_instance.Scalar(dtype=self.x_dtype, name='pad_scalar')
        if self.constant_values:
            self.constant_values_gm = self.tik_instance.Tensor(self.x_dtype, 
                                                               (self.block_num,),
                                                               name='constant_values_gm',
                                                               scope=tik.scope_gm)

        self.tiling_key = self.tik_instance.Scalar(self.tiling_dtype, "tiling_key", init_value=0)
        self.tiling_input_dim_0 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_input_dim_0", init_value=0)
        self.tiling_input_dim_1 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_input_dim_1", init_value=0)
        self.tiling_input_dim_2 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_input_dim_2", init_value=0)
        self.tiling_input_dim_3 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_input_dim_3", init_value=0)
        self.tiling_input_dim_4 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_input_dim_4", init_value=0)
        self.tiling_output_dim_0 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_output_dim_0", init_value=0)
        self.tiling_output_dim_1 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_output_dim_1", init_value=0)
        self.tiling_output_dim_2 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_output_dim_2", init_value=0)
        self.tiling_output_dim_3 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_output_dim_3", init_value=0)
        self.tiling_output_dim_4 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_output_dim_4", init_value=0)
        self.tiling_input_shape = None
        self.tiling_output_shape = None
        self.padding_index_0 = self.tik_instance.Scalar(self.tiling_dtype, "padding_index_0", init_value=0)
        self.padding_index_1 = self.tik_instance.Scalar(self.tiling_dtype, "padding_index_1", init_value=0)
        self.padding_index_2 = self.tik_instance.Scalar(self.tiling_dtype, "padding_index_2", init_value=0)
        self.padding_index_3 = self.tik_instance.Scalar(self.tiling_dtype, "padding_index_3", init_value=0)

        self.core_uesd_num = self.tik_instance.Scalar(self.tiling_dtype, "core_uesd_num", init_value=0)
        self.not_last_core_num = self.tik_instance.Scalar(self.tiling_dtype, "not_last_core_num", init_value=0)
        self.last_core_num = self.tik_instance.Scalar(self.tiling_dtype, "last_core_num", init_value=0)
        self.last_three_dims_output = None
        self.input_ele_per_core = None
        self.output_ele_per_core = None
        self.last_three_dims_input = None
        self.ranges = None

    def get_pad_scalar(self):
        """
        get_pad_scalar
        """
        constant_values_ub = self.tik_instance.Tensor(self.constant_values_dtype, 
                                                      (self.block_num,),
                                                      name='constant_values_ub',
                                                      scope=tik.scope_ubuf)
        self.tik_instance.data_move(constant_values_ub, self.constant_values_gm, 0, 1, 1, 0, 0)
        self.pad_scalar.set_as(constant_values_ub[0])

    def get_tiling_args(self):
        """
        when input shape is less 6, will. expand to 6
        tiling_input_dim_cut_axis: which dim will be cut
        """
        with self.tik_instance.new_stmt_scope():
            tiling_ub = self.tik_instance.Tensor(self.tiling_dtype,
                                                 (Constant.TILING_NUMS,),
                                                 name="tiling_ub",
                                                 scope=tik.scope_ubuf)
            self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, Constant.TILING_NUMS // 4, 0, 0)
            self.tiling_key.set_as(tiling_ub[0])
            self.tiling_input_dim_0.set_as(tiling_ub[1])
            self.tiling_input_dim_1.set_as(tiling_ub[2])
            self.tiling_input_dim_2.set_as(tiling_ub[3])
            self.tiling_input_dim_3.set_as(tiling_ub[4])
            self.tiling_input_dim_4.set_as(tiling_ub[5])
            self.tiling_output_dim_0.set_as(tiling_ub[6])
            self.tiling_output_dim_1.set_as(tiling_ub[7])
            self.tiling_output_dim_2.set_as(tiling_ub[8])
            self.tiling_output_dim_3.set_as(tiling_ub[9])
            self.tiling_output_dim_4.set_as(tiling_ub[10])
            self.core_uesd_num.set_as(tiling_ub[11])
            self.padding_index_0.set_as(tiling_ub[12])
            self.padding_index_1.set_as(tiling_ub[13])
            self.padding_index_2.set_as(tiling_ub[14])
            self.padding_index_3.set_as(tiling_ub[15])
            self.not_last_core_num.set_as(tiling_ub[16])
            self.last_core_num.set_as(tiling_ub[17])

    def init_src_dst_gm(self, pad_input_idx=0, pad_outnput_idx=0):
        """
        init gm tensor set tiling, input, paddings output tensor(gm)
        :param
        input_dict_list: the dict of input_dict
        :param
        output_dict_list: output_dict_list
        :param
        pad_input_idx: pad_input_idx
        :param
        pad_outnput_idx: pad_outnput_idx
        :return:
        None
        """
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype,
                                                  self.tiling_shape,
                                                  name="tiling",
                                                  scope=tik.scope_gm)
        x_gm = self.tik_instance.Tensor(self.x_dtype, self.unknown_max_shape, name="x", scope=tik.scope_gm)
        paddings_gm = self.tik_instance.Tensor(self.paddings_dtype,
                                               self.unknown_max_shape,
                                               name="paddings",
                                               scope=tik.scope_gm)
        self.input_gm_list.append(x_gm)
        self.input_gm_list.append(paddings_gm)
        if self.constant_values is not None:
            self.input_gm_list.append(self.constant_values_gm)

        y_gm = self.tik_instance.Tensor(self.x_dtype, self.unknown_max_shape, name="y", scope=tik.scope_gm)
        self.output_gm_list.append(y_gm)

        self.input_gm = self.input_gm_list[pad_input_idx]
        self.output_gm = self.output_gm_list[pad_outnput_idx]

    def get_cal_scalar(self, core_index):
        """
        get_cal_scalar
        """
        self.last_three_dims_output = self.tik_instance.Scalar(self.tiling_dtype, "last_three_dims_output")
        self.input_ele_per_core = self.tik_instance.Scalar(self.tiling_dtype, name='input_ele_per_core')
        self.output_ele_per_core = self.tik_instance.Scalar(self.tiling_dtype, name='output_ele_per_core')
        self.last_three_dims_input = self.tik_instance.Scalar(self.tiling_dtype, name='last_three_dims_input')
        self.ranges = self.tik_instance.Scalar(self.tiling_dtype, name='ranges')

        self.last_three_dims_output.set_as(self.tiling_output_dim_2 * self.tiling_output_dim_3 *
                                           self.tiling_output_dim_4)
        self.last_three_dims_input.set_as(self.tiling_input_dim_2 * self.tiling_input_dim_3 * self.tiling_input_dim_4)
        self.input_ele_per_core.set_as(self.not_last_core_num * self.last_three_dims_input)
        self.output_ele_per_core.set_as(self.not_last_core_num * self.last_three_dims_output)
        with self.tik_instance.if_scope(core_index == self.core_uesd_num - 1):
            self.ranges.set_as(self.last_core_num)
        with self.tik_instance.else_scope():
            self.ranges.set_as(self.not_last_core_num)

    def pad_v3_compute_tiling(self):
        """
        reflection_pad_v3_compute_tiling
        """
        with self.tik_instance.for_range(0, self.core_nums, block_num=self.core_nums) as core_index:
            self.get_tiling_args()
            self.get_cal_scalar(core_index)
            with self.tik_instance.if_scope(self.tiling_key == 3):
                self.core_uesd_num.set_as(self.core_nums)
            with self.tik_instance.if_scope(core_index < self.core_uesd_num):
                if not self.constant_values:
                    self.pad_scalar.set_as(0)
                else:
                    self.get_pad_scalar()
                self.do_pad(core_index)

    def vec_dup_ub(self, only_one_ub, last_three_dims_output):
        """
        vec_dup_only_one_ub
        """
        with self.tik_instance.if_scope(last_three_dims_output // self.dump_max_x > 0):
            with self.tik_instance.for_range(0, last_three_dims_output // self.dump_max_x) as i:
                self.tik_instance.vec_dup(self.dump_mask_max_x, only_one_ub[i * self.dump_max_x], self.pad_scalar,
                                          self.max_repeat_time, 8)
        with self.tik_instance.if_scope((last_three_dims_output % self.dump_max_x) > self.dump_mask_max_x):
            self.tik_instance.vec_dup(self.dump_mask_max_x,
                                      only_one_ub[last_three_dims_output // self.dump_max_x * self.dump_max_x],
                                      self.pad_scalar, last_three_dims_output % self.dump_max_x // self.dump_mask_max_x,
                                      8)
        with self.tik_instance.if_scope((last_three_dims_output % self.dump_max_x) % self.dump_mask_max_x > 0):
            self.tik_instance.vec_dup(
                (last_three_dims_output % self.dump_max_x) % self.dump_mask_max_x,
                only_one_ub[last_three_dims_output // self.dump_max_x * self.dump_max_x +
                            (last_three_dims_output % self.dump_max_x) // self.dump_mask_max_x * self.dump_mask_max_x],
                self.pad_scalar, 1, 8)

    # When the size of ub is large enough to hold all the output elements at one time, we use this branch
    def do_tiling_key_mode_0(self, core_index):
        """
        do_tiling_key_mode_0 when tiling key = 0
        """
        only_one_ub = self.tik_instance.Tensor(self.x_dtype, 
                                               (self.ub_number,),
                                               name='only_one_ub',
                                               scope=tik.scope_ubuf)
        self.vec_dup_ub(only_one_ub, self.last_three_dims_output)

        with self.tik_instance.for_range(0, self.ranges) as index:
            self.tik_instance.data_move(
                only_one_ub[self.padding_index_2 * self.tiling_output_dim_3 * self.tiling_output_dim_4 +
                            self.padding_index_0 * self.tiling_output_dim_4],
                self.input_gm[core_index * self.input_ele_per_core + index * self.last_three_dims_input], 0,
                self.tiling_input_dim_2, self.tiling_input_dim_4 * self.tiling_input_dim_3 // self.block_num, 0,
                (self.padding_index_0 + self.padding_index_1) * self.tiling_output_dim_4 // self.block_num)
            self.tik_instance.data_move(
                self.output_gm[core_index * self.output_ele_per_core + index * self.last_three_dims_output],
                only_one_ub, 0, 1,
                self.tiling_output_dim_4 * self.tiling_output_dim_3 * self.tiling_output_dim_2 // self.block_num, 0, 0)

    def data_move_with_move_times_mode_1(self, core_index, index, move_times_index, useful_real_lines, remain_lines,
                                         real_lines, just_one_ub):
        """
        data_move_with_move_times_mode_1 when tiling key = 1
        """
        with self.tik_instance.if_scope(move_times_index == 0):
            self.tik_instance.data_move(
                self.output_gm[core_index * self.output_ele_per_core + index * self.last_three_dims_output],
                just_one_ub, 0, 1, (real_lines - self.padding_index_3) * self.tiling_output_dim_3 *
                self.tiling_output_dim_4 // self.block_num, 0, 0)

        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(remain_lines > 0):
                self.tik_instance.data_move(
                    self.output_gm[core_index * self.output_ele_per_core + index * self.last_three_dims_output +
                                   (move_times_index * useful_real_lines + self.padding_index_0) *
                                   self.tiling_output_dim_3 * self.tiling_output_dim_4],
                    just_one_ub[self.padding_index_2 * self.tiling_output_dim_3 * self.tiling_output_dim_4], 0, 1,
                    useful_real_lines * self.tiling_output_dim_3 * self.tiling_output_dim_4 // self.block_num, 0, 0)
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(
                    self.output_gm[core_index * self.output_ele_per_core + index * self.last_three_dims_output +
                                   (move_times_index * useful_real_lines + self.padding_index_0) *
                                   self.tiling_output_dim_3 * self.tiling_output_dim_4],
                    just_one_ub[self.padding_index_2 * self.tiling_output_dim_3 * self.tiling_output_dim_4], 0, 1,
                    (useful_real_lines + self.padding_index_3) * self.tiling_output_dim_3 * self.tiling_output_dim_4 //
                    self.block_num, 0, 0)

    # When each dimension of paddings is less than 4, and ub can only hold a few rows of elements at a time,
    # we use this branch
    def do_tiling_key_mode_1(self, core_index):
        """
        do_tiling_key_mode_1 when tiling key = 1
        """
        just_one_ub = self.tik_instance.Tensor(self.x_dtype, 
                                               (self.ub_number,),
                                               name='just_one_ub',
                                               scope=tik.scope_ubuf)
        real_lines = self.tik_instance.Scalar(self.tiling_dtype, name='real_lines')
        real_lines.set_as(self.ub_number // (self.tiling_output_dim_3 * self.tiling_output_dim_4))
        self.vec_dup_ub(just_one_ub, real_lines * (self.tiling_output_dim_3 * self.tiling_output_dim_4))
        useful_real_lines = self.tik_instance.Scalar(self.tiling_dtype, name='useful_real_lines')
        useful_real_lines.set_as(real_lines - self.padding_index_2 - self.padding_index_3)
        move_times = self.tik_instance.Scalar(self.tiling_dtype, name='move_times')
        move_times.set_as(self.tiling_input_dim_2 // useful_real_lines)
        remain_lines = self.tik_instance.Scalar(self.tiling_dtype, name='remain_lines')
        remain_lines.set_as(self.tiling_input_dim_2 % useful_real_lines)

        with self.tik_instance.for_range(0, self.ranges) as index:
            with self.tik_instance.for_range(0, move_times) as i:
                self.tik_instance.data_move(
                    just_one_ub[self.padding_index_2 * self.tiling_output_dim_3 * self.tiling_output_dim_4 +
                                self.padding_index_0 * self.tiling_output_dim_4],
                    self.input_gm[core_index * self.input_ele_per_core + index * self.last_three_dims_input +
                                  i * useful_real_lines * self.tiling_input_dim_3 * self.tiling_input_dim_4], 0,
                    useful_real_lines, self.tiling_input_dim_4 * self.tiling_input_dim_3 // self.block_num, 0,
                    (self.padding_index_0 + self.padding_index_1) * self.tiling_output_dim_4 // self.block_num)
                self.data_move_with_move_times_mode_1(core_index, index, i, useful_real_lines, remain_lines, real_lines,
                                                      just_one_ub)

            with self.tik_instance.if_scope(remain_lines > 0):
                self.tik_instance.data_move(
                    just_one_ub[(real_lines - self.padding_index_3 - remain_lines) * self.tiling_output_dim_3 *
                                self.tiling_output_dim_4 + self.padding_index_0 * self.tiling_output_dim_4],
                    self.input_gm[core_index * self.input_ele_per_core + index * self.last_three_dims_input +
                                  move_times * useful_real_lines * self.tiling_input_dim_3 * self.tiling_input_dim_4],
                    0, remain_lines, self.tiling_input_dim_4 * self.tiling_input_dim_3 // self.block_num, 0,
                    (self.padding_index_0 + self.padding_index_1) * self.tiling_output_dim_4 // self.block_num)

                self.tik_instance.data_move(
                    self.output_gm[core_index * self.output_ele_per_core + index * self.last_three_dims_output +
                                   (move_times * useful_real_lines + self.padding_index_0) * self.tiling_output_dim_3 *
                                   self.tiling_output_dim_4],
                    just_one_ub[(real_lines - self.padding_index_3 - remain_lines) * self.tiling_output_dim_3 *
                                self.tiling_output_dim_4], 0, 1, (remain_lines + self.padding_index_3) *
                    self.tiling_output_dim_3 * self.tiling_output_dim_4 // self.block_num, 0, 0)

    def move_paddings_data_top_and_bottom_mode_2(self, core_index, index, last_two_dims, ub_number, one_ub, padding_dim,
                                                 offset):
        """
        move_paddings_data_top_and_bottom_mode_2
        """
        with self.tik_instance.for_range(0, padding_dim) as i:
            with self.tik_instance.if_scope(last_two_dims // ub_number > 0):
                with self.tik_instance.for_range(0, last_two_dims // ub_number) as j:
                    self.tik_instance.data_move(
                        self.output_gm[core_index * self.output_ele_per_core + index * self.last_three_dims_output +
                                       (i + offset) * last_two_dims + j * ub_number], one_ub, 0, 1,
                        ub_number // self.block_num, 0, 0)
            with self.tik_instance.if_scope(last_two_dims % ub_number > 0):
                self.tik_instance.data_move(
                    self.output_gm[core_index * self.output_ele_per_core + index * self.last_three_dims_output +
                                   (i + offset) * last_two_dims + last_two_dims // ub_number * ub_number], one_ub, 0, 1,
                    (last_two_dims % ub_number) // self.block_num, 0, 0)

    def move_paddings_data_left_and_right_mode_2(self, core_index, index, last_two_dims, one_ub, padding_dim, gm_offset,
                                                 dst_offset):
        """
        move_paddings_data_left_and_right_mode_2
        """
        with self.tik_instance.if_scope(padding_dim > 0):
            self.tik_instance.data_move(
                self.output_gm[core_index * self.output_ele_per_core + index * self.last_three_dims_output +
                               self.padding_index_2 * last_two_dims + gm_offset], one_ub, 0, self.tiling_input_dim_2,
                padding_dim * self.tiling_output_dim_4 // self.block_num, 0,
                (self.tiling_input_dim_3 + dst_offset) * self.tiling_output_dim_4 // self.block_num)

    # When each dimension of paddings is less than 4, and ub cannot accommodate a row of output,
    # and the H dimension is less than 3000, we use this branch
    def do_tiling_key_mode_2(self, core_index):
        """
        do_tiling_key_mode_2 when tiling key = 2
        """
        one_ub = self.tik_instance.Tensor(self.x_dtype, (self.ub_number,), name='one_ub', scope=tik.scope_ubuf)
        ub_number = self.tik_instance.Scalar(dtype=self.tiling_dtype, name='ub_number')
        ub_number.set_as(self.ub_number - Constant.BLOCK)
        last_two_dims = self.tik_instance.Scalar(dtype=self.tiling_dtype, name='last_two_dims')
        last_two_dims.set_as(self.tiling_output_dim_3 * self.tiling_output_dim_4)

        with self.tik_instance.for_range(0, self.ranges) as index:
            self.vec_dup_ub(one_ub, ub_number)
            self.move_paddings_data_top_and_bottom_mode_2(core_index, index, last_two_dims, ub_number, one_ub,
                                                          self.padding_index_2, 0)
            self.move_paddings_data_top_and_bottom_mode_2(core_index, index, last_two_dims, ub_number, one_ub,
                                                          self.padding_index_3,
                                                          (self.padding_index_2 + self.tiling_input_dim_2))

            self.move_paddings_data_left_and_right_mode_2(core_index, index, last_two_dims, one_ub,
                                                          self.padding_index_0, 0, self.padding_index_1)
            self.move_paddings_data_left_and_right_mode_2(
                core_index, index, last_two_dims, one_ub, self.padding_index_1,
                (self.tiling_input_dim_3 + self.padding_index_0) * self.tiling_output_dim_4, self.padding_index_0)

            with self.tik_instance.for_range(0, self.tiling_input_dim_2) as i:
                with self.tik_instance.if_scope(self.tiling_input_dim_3 * self.tiling_input_dim_4 // ub_number > 0):
                    with self.tik_instance.for_range(0, self.tiling_input_dim_3 * self.tiling_input_dim_4 //
                                                     ub_number) as j:
                        self.tik_instance.data_move(
                            one_ub,
                            self.input_gm[core_index * self.input_ele_per_core + index * self.last_three_dims_input +
                                          i * self.tiling_input_dim_3 * self.tiling_input_dim_4 + j * ub_number], 0, 1,
                            ub_number // self.block_num, 0, 0)
                        self.tik_instance.data_move(
                            self.output_gm[core_index * self.output_ele_per_core + index * self.last_three_dims_output +
                                           (self.padding_index_2 + i) * last_two_dims +
                                           self.padding_index_0 * self.tiling_output_dim_4 + j * ub_number], one_ub, 0,
                            1, ub_number // self.block_num, 0, 0)

                with self.tik_instance.if_scope(self.tiling_input_dim_3 * self.tiling_input_dim_4 % ub_number > 0):
                    self.tik_instance.data_move(
                        one_ub,
                        self.input_gm[core_index * self.input_ele_per_core + index * self.last_three_dims_input +
                                      i * self.tiling_input_dim_3 * self.tiling_input_dim_4 +
                                      self.tiling_input_dim_3 * self.tiling_input_dim_4 // ub_number * ub_number], 0, 1,
                        (self.tiling_input_dim_3 * self.tiling_input_dim_4 % ub_number) // self.block_num, 0, 0)
                    self.tik_instance.data_move(
                        self.output_gm[core_index * self.output_ele_per_core + index * self.last_three_dims_output +
                                       (self.padding_index_2 + i) * last_two_dims +
                                       self.padding_index_0 * self.tiling_output_dim_4 +
                                       self.tiling_input_dim_3 * self.tiling_input_dim_4 // ub_number * ub_number],
                        one_ub, 0, 1, (self.tiling_input_dim_3 * self.tiling_input_dim_4 % ub_number) // self.block_num,
                        0, 0)

    def data_move_in_out_for_mode_3(self, offset_gm, block, process_num, align_burst):
        with self.tik_instance.new_stmt_scope():
            move_ub = self.tik_instance.Tensor(dtype=self.inner_dtype,
                                                shape=(self.ub_number,),
                                                name='move_ub',
                                                scope=tik.scope_ubuf)
            with self.tik_instance.if_scope(process_num // self.ub_number > 0):
                with self.tik_instance.for_range(0, process_num // self.ub_number) as i:
                    self.tik_instance.data_move(move_ub, self.input_gm[offset_gm + i * self.ub_number], 0, 1,
                                                self.ub_number // block, 0, 0)
                    self.tik_instance.data_move(self.output_gm[offset_gm + i * self.ub_number], move_ub, 0, 1,
                                                self.ub_number // block, 0, 0)
            with self.tik_instance.if_scope(process_num % self.ub_number > 0):
                self.tik_instance.data_move(
                    move_ub,
                    self.input_gm[offset_gm + process_num // self.ub_number * self.ub_number], 0,
                    1, align_burst, 0, 0)
                self.tik_instance.data_move(
                    self.output_gm[offset_gm + process_num // self.ub_number * self.ub_number],
                    move_ub, 0, 1, align_burst, 0, 0)

    # When paddings are all 0, we use this branch
    def do_tiling_key_mode_3(self, core_index):
        """
        do_tiling_key_mode_3 when tiling key = 3
        """
        total_output_tensor = self.tik_instance.Scalar(dtype=self.tiling_dtype,
                                                       name='total_output_tensor',
                                                       init_value=1)
        total_output_tensor_each_core = self.tik_instance.Scalar(dtype=self.tiling_dtype,
                                                                 name='total_output_tensor_each_core')
        total_output_tensor_tail_core = self.tik_instance.Scalar(dtype=self.tiling_dtype,
                                                                 name='total_output_tensor_tail_core')
        offset_gm = self.tik_instance.Scalar(dtype=self.tiling_dtype, name='offset_gm')
        
        align_burst = self.tik_instance.Scalar(dtype=self.tiling_dtype, name='align_burst')
        self.tiling_output_shape = [
            self.tiling_output_dim_0, self.tiling_output_dim_1, self.tiling_output_dim_2, self.tiling_output_dim_3,
            self.tiling_output_dim_4
        ]
        for ele in self.tiling_output_shape:
            total_output_tensor.set_as(total_output_tensor * ele)
        block = self.tik_instance.Scalar(dtype=self.tiling_dtype, name='block')
        block.set_as(Constant.BLOCK_SIZE // self.inner_bytes_size)
        core_nums = self.tik_instance.Scalar(dtype=self.tiling_dtype, name='core_nums')
        core_nums.set_as(self.core_nums)
        total_output_tensor_each_core.set_as((total_output_tensor - 1) // core_nums + 1)
        total_output_tensor_each_core.set_as(((total_output_tensor_each_core - 1) // block + 1) * block)
        core_nums.set_as((total_output_tensor - 1) // total_output_tensor_each_core + 1)
        total_output_tensor_tail_core.set_as(total_output_tensor - total_output_tensor_each_core * (core_nums - 1))
        total_output_tensor_tail_core.set_as(((total_output_tensor_tail_core - 1) // block + 1) * block)
        offset_gm.set_as(core_index * total_output_tensor_each_core)
        with self.tik_instance.if_scope(core_index < core_nums - 1):
            align_burst.set_as(((total_output_tensor_each_core % self.ub_number) - 1) // block + 1)
            self.data_move_in_out_for_mode_3(offset_gm, block, total_output_tensor_each_core, align_burst)
        with self.tik_instance.elif_scope(core_index == core_nums - 1):
            align_burst.set_as(((total_output_tensor_tail_core % self.ub_number) - 1) // block + 1)
            self.data_move_in_out_for_mode_3(offset_gm, block, total_output_tensor_tail_core, align_burst)

    def move_input_gm_data_to_output_gm_by_ub_small_shape(self, core_index, real_lines, move_times, remain_lines,
                                                          just_one_ub):
        """
        move_input_gm_data_to_output_gm_by_ub_small_shape
        """
        with self.tik_instance.for_range(0, self.ranges) as index:
            with self.tik_instance.for_range(0, move_times) as i:
                self.tik_instance.data_move(
                    just_one_ub,
                    self.input_gm[core_index * self.input_ele_per_core + index * self.last_three_dims_input +
                                  i * real_lines * self.tiling_input_dim_3 * self.tiling_input_dim_4], 0, 1,
                    real_lines * self.tiling_input_dim_4 * self.tiling_input_dim_3 // self.block_num, 0, 0)
                with self.tik_instance.if_scope(real_lines <= Constant.MAX_REPEAT_TIME):
                    self.tik_instance.data_move(
                        self.output_gm[core_index * self.output_ele_per_core + index * self.last_three_dims_output +
                                    (self.padding_index_2 + i * real_lines) * self.tiling_output_dim_3 *
                                    self.tiling_output_dim_4 + self.padding_index_0 * self.tiling_output_dim_4],
                        just_one_ub, 
                        0, real_lines, self.tiling_input_dim_3 * self.tiling_input_dim_4 // self.block_num,
                        0, (self.padding_index_0 + self.padding_index_1) * self.tiling_output_dim_4 // self.block_num)
                with self.tik_instance.elif_scope(real_lines > Constant.MAX_REPEAT_TIME):
                    self.tik_instance.data_move(
                        self.output_gm[core_index * self.output_ele_per_core + index * self.last_three_dims_output +
                                    (self.padding_index_2 + i * real_lines) * self.tiling_output_dim_3 *
                                    self.tiling_output_dim_4 + self.padding_index_0 * self.tiling_output_dim_4],
                        just_one_ub,
                        0, Constant.MAX_REPEAT_TIME, 
                        self.tiling_input_dim_3 * self.tiling_input_dim_4 // self.block_num,
                        0, (self.padding_index_0 + self.padding_index_1) * self.tiling_output_dim_4 // self.block_num)
                    self.tik_instance.data_move(
                        self.output_gm[core_index * self.output_ele_per_core + index * self.last_three_dims_output +
                                    (self.padding_index_2 + i * (real_lines) + 
                                    Constant.MAX_REPEAT_TIME) * self.tiling_output_dim_3 *
                                    self.tiling_output_dim_4 + self.padding_index_0 * self.tiling_output_dim_4],
                        just_one_ub[Constant.MAX_REPEAT_TIME * self.tiling_input_dim_4 * self.tiling_input_dim_3],
                        0, real_lines - Constant.MAX_REPEAT_TIME, 
                        self.tiling_input_dim_3 * self.tiling_input_dim_4 // self.block_num,
                        0, (self.padding_index_0 + self.padding_index_1) * self.tiling_output_dim_4 // self.block_num)

            with self.tik_instance.if_scope(remain_lines > 0):
                self.tik_instance.data_move(
                    just_one_ub,
                    self.input_gm[core_index * self.input_ele_per_core + index * self.last_three_dims_input +
                                  move_times * real_lines * self.tiling_input_dim_3 * self.tiling_input_dim_4], 0, 1,
                    remain_lines * self.tiling_input_dim_4 * self.tiling_input_dim_3 // self.block_num, 0, 0)

                with self.tik_instance.if_scope(remain_lines > Constant.MAX_REPEAT_TIME):
                    self.tik_instance.data_move(
                        self.output_gm[core_index * self.output_ele_per_core + index * self.last_three_dims_output +
                                    (self.padding_index_2 + move_times * real_lines) * self.tiling_output_dim_3 *
                                    self.tiling_output_dim_4 + self.padding_index_0 * self.tiling_output_dim_4],
                        just_one_ub,
                        0, 
                        Constant.MAX_REPEAT_TIME, 
                        self.tiling_input_dim_3 * self.tiling_input_dim_4 // self.block_num,
                        0, 
                        (self.padding_index_0 + self.padding_index_1) * self.tiling_output_dim_4 // self.block_num)
                        
                    self.tik_instance.data_move(
                        self.output_gm[core_index * self.output_ele_per_core + index * self.last_three_dims_output +
                                    (self.padding_index_2 + move_times * real_lines + 
                                    Constant.MAX_REPEAT_TIME) * self.tiling_output_dim_3 *
                                    self.tiling_output_dim_4 + self.padding_index_0 * self.tiling_output_dim_4],
                        just_one_ub[Constant.MAX_REPEAT_TIME * self.tiling_input_dim_4 * self.tiling_input_dim_3],
                        0, 
                        remain_lines - Constant.MAX_REPEAT_TIME, 
                        self.tiling_input_dim_3 * self.tiling_input_dim_4 // self.block_num,
                        0, 
                        (self.padding_index_0 + self.padding_index_1) * self.tiling_output_dim_4 // self.block_num)  
                
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(
                        self.output_gm[core_index * self.output_ele_per_core + index * self.last_three_dims_output +
                                    (self.padding_index_2 + move_times * real_lines) * self.tiling_output_dim_3 *
                                    self.tiling_output_dim_4 + self.padding_index_0 * self.tiling_output_dim_4],
                        just_one_ub, 
                        0, 
                        remain_lines, 
                        self.tiling_input_dim_3 * self.tiling_input_dim_4 // self.block_num,
                        0, 
                        (self.padding_index_0 + self.padding_index_1) * self.tiling_output_dim_4 // self.block_num)

    def move_input_gm_data_to_output_gm_by_ub_big_shape(self, core_index, one_line_time, ub_number,
                                                        one_line_remain_data, just_one_ub):
        """
        move_input_gm_data_to_output_gm_by_ub_big_shape
        """
        with self.tik_instance.for_range(0, self.ranges) as index:
            with self.tik_instance.for_range(0, self.tiling_input_dim_2) as i:
                with self.tik_instance.for_range(0, one_line_time) as j:
                    self.tik_instance.data_move(
                        just_one_ub,
                        self.input_gm[core_index * self.input_ele_per_core + index * self.last_three_dims_input +
                                      i * self.tiling_input_dim_3 * self.tiling_input_dim_4 + j * ub_number], 0, 1,
                        ub_number // self.block_num, 0, 0)
                    self.tik_instance.data_move(
                        self.output_gm[core_index * self.output_ele_per_core + index * self.last_three_dims_output +
                                       (self.padding_index_2 + i) * self.tiling_output_dim_3 * self.tiling_output_dim_4
                                       + self.padding_index_0 * self.tiling_output_dim_4 + j * ub_number], just_one_ub,
                        0, 1, ub_number // self.block_num, 0, 0)

                with self.tik_instance.if_scope(one_line_remain_data > 0):
                    self.tik_instance.data_move(
                        just_one_ub,
                        self.input_gm[core_index * self.input_ele_per_core + index * self.last_three_dims_input +
                                      i * self.tiling_input_dim_3 * self.tiling_input_dim_4 +
                                      one_line_time * ub_number], 0, 1, one_line_remain_data // self.block_num, 0, 0)
                    self.tik_instance.data_move(
                        self.output_gm[core_index * self.output_ele_per_core + index * self.last_three_dims_output +
                                       (self.padding_index_2 + i) * self.tiling_output_dim_3 * self.tiling_output_dim_4
                                       + self.padding_index_0 * self.tiling_output_dim_4 + one_line_time * ub_number],
                        just_one_ub, 0, 1, one_line_remain_data // self.block_num, 0, 0)

    # For other scenarios not included before, we use this branch
    def do_tiling_key_mode_4(self, core_index):
        """
        do_tiling_key_mode_1 when tiling key = 4
        """
        just_one_ub = self.tik_instance.Tensor(self.x_dtype, 
                                               (self.ub_number,),
                                               name='just_one_ub',
                                               scope=tik.scope_ubuf)
        ub_number = self.tik_instance.Scalar(dtype=self.tiling_dtype, name='ub_number')
        ub_number.set_as(self.ub_number - Constant.BLOCK)
        self.vec_dup_ub(just_one_ub, ub_number)
        real_lines = self.tik_instance.Scalar(self.tiling_dtype, name='real_lines')
        real_lines.set_as(ub_number // (self.tiling_input_dim_3 * self.tiling_input_dim_4))

        move_times = self.tik_instance.Scalar(self.tiling_dtype, name='move_times')
        remain_lines = self.tik_instance.Scalar(self.tiling_dtype, name='remain_lines')
        one_line_time = self.tik_instance.Scalar(self.tiling_dtype, name='one_line_time')
        one_line_remain_data = self.tik_instance.Scalar(self.tiling_dtype, name='one_line_remain_data')
        with self.tik_instance.if_scope(real_lines > 0):
            move_times.set_as(self.tiling_input_dim_2 // real_lines)
            remain_lines.set_as(self.tiling_input_dim_2 % real_lines)
        with self.tik_instance.else_scope():
            one_line_time.set_as((self.tiling_input_dim_3 * self.tiling_input_dim_4) // ub_number)
            one_line_remain_data.set_as((self.tiling_input_dim_3 * self.tiling_input_dim_4) % ub_number)
        all_ele_per_ub = self.tik_instance.Scalar(dtype=self.tiling_dtype, name='all_ele_per_ub')
        all_ele_per_ub.set_as(self.ranges * self.last_three_dims_output)
        dump_time = self.tik_instance.Scalar(dtype=self.tiling_dtype, name='dump_time')
        dump_time.set_as(all_ele_per_ub // ub_number)
        remain_dump_data = self.tik_instance.Scalar(dtype=self.tiling_dtype, name='remain_dump_data')
        remain_dump_data.set_as(all_ele_per_ub % ub_number)

        # dump the y_gm with paddings by data_move data from ub to y_gm
        with self.tik_instance.for_range(0, dump_time) as index:
            self.tik_instance.data_move(self.output_gm[core_index * self.output_ele_per_core + index * ub_number],
                                        just_one_ub, 0, 1, ub_number // self.block_num, 0, 0)
        with self.tik_instance.if_scope(remain_dump_data > 0):
            self.tik_instance.data_move(self.output_gm[core_index * self.output_ele_per_core + dump_time * ub_number],
                                        just_one_ub, 0, 1, remain_dump_data // self.block_num, 0, 0)
        # move x_gm data to y_gm by ub
        with self.tik_instance.if_scope(real_lines > 0):
            self.move_input_gm_data_to_output_gm_by_ub_small_shape(core_index, real_lines, move_times, remain_lines,
                                                                   just_one_ub)
        with self.tik_instance.else_scope():
            self.move_input_gm_data_to_output_gm_by_ub_big_shape(core_index, one_line_time, ub_number,
                                                                 one_line_remain_data, just_one_ub)

    def do_pad(self, core_index):
        """
        do_pad with different tiling key
        """
        with self.tik_instance.if_scope(self.tiling_key == Constant.MODE0):
            with self.tik_instance.new_stmt_scope():
                self.do_tiling_key_mode_0(core_index)
        with self.tik_instance.if_scope(self.tiling_key == Constant.MODE1):
            with self.tik_instance.new_stmt_scope():
                self.do_tiling_key_mode_1(core_index)
        with self.tik_instance.if_scope(self.tiling_key == Constant.MODE2):
            with self.tik_instance.new_stmt_scope():
                self.do_tiling_key_mode_2(core_index)
        with self.tik_instance.if_scope(self.tiling_key == Constant.MODE3):
            with self.tik_instance.new_stmt_scope():
                self.do_tiling_key_mode_3(core_index)
        with self.tik_instance.if_scope(self.tiling_key == Constant.MODE4):
            with self.tik_instance.new_stmt_scope():
                self.do_tiling_key_mode_4(core_index)

    def pad_compute(self, outer_compile_info=None):
        """
        pad_compute
        """
        self.pad_v3_compute_tiling()
        opt_config = {"out_of_bound_sync_check": True}

        # add compile info
        wr_compile_info = {
            "core_num": self.core_nums,
            "padding_contiguous": self.padding_contiguous,
            "size": self.size,
            "mode": self.mode,
            "ub_size": self.ub_number_4_bytes
        }
        if outer_compile_info is not None:
            for key in outer_compile_info.keys():
                wr_compile_info[key] = outer_compile_info[key]
        tbe_context.get_context().add_compile_info("vars", wr_compile_info)
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=self.input_gm_list,
                                   flowtable=[self.tiling_gm],
                                   outputs=self.output_gm_list,
                                   config=opt_config)
        return self.tik_instance


@register_operator("PadV3")
def pad_v3_5hd(x, paddings, constant_values, y, mode, padding_contiguous=True, kernel_name="pad_v3"):
    """ calculating pad_v3_5hd tensor by paddings parameters

    Parameters
    ----------
    x : dict
        shape and dtype of input
    paddings: dict
        shape and dtype of output
        For each dimension D of input, paddings[D, 0] indicates how many
        values to add
        before the contents of tensor in that dimension, and paddings[D, 1]
        indicates
        how many values to add after the contents of tensor in that dimension.
    constant_values: dict
        the value to fill the tensor
    y: dict
        shape and dtype of output
    mode: str
        the mode of calculate
    padding_contiguous: bool
        judge whether the memory is contiguous
    kernel_name : str
        cce kernel name, default value is "reflection_pad_v3"

    Returns
    -------
    None.
    """
    src_dtype = x.get("dtype").lower()
    paddings_dtype = paddings.get("dtype").lower()
    supported_dtype = ("float16", "float32", "int32", "bfloat16")
    para_check.check_dtype(src_dtype, supported_dtype, param_name="x")
    para_check.check_dtype(paddings_dtype, ("int32", "int64"), param_name="paddings")
    pad_v3_5hd_instance = PadV35HDInit(x, paddings, constant_values, y, mode, padding_contiguous, kernel_name)
    pad_v3_5hd_instance.init_src_dst_gm(pad_input_idx=0, pad_outnput_idx=0)
    return pad_v3_5hd_instance.pad_compute()
