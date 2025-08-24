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
reflection_pad_v3.py
"""
from impl import constant_util as constant
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator
from tbe.common.platform import get_bit_len


# 'pylint: disable=too-few-public-methods
class Constant(object):
    """
    The class for constant
    """
    # tiling param nums
    TILING_NUMS = 20
    # `1 byte = 8 bit`
    EIGHT_BIT = 8
    # vnchw the minest block
    TRANS_MIN_BLKS = 16
    # max int64
    MAX_INT64 = 2**64 - 1
    # compute only zero axis, cut last dim
    MODE0 = 0
    # compute only one axis
    MODE1 = 1
    # network case
    MODE2 = 2
    # compute only one axis
    MODE3 = 3
    # tiling mode 1 divide by outer:[0,1,2] inner:[3,4]
    AXIS_DIMS_3 = 3
    # tiling mode 0 divide by outer:[0,1,2,3] inner:[4]
    AXIS_DIMS_4 = 4
    # reserved ub size
    RESERVED_UB_SIZE = 10240
    INIT_NUM = 65


# 'pylint: disable=too-many-instance-attributes,too-many-statements,too-many-locals,too-many-lines
# 'pylint: disable=too-many-arguments,invalid-name,too-many-public-methods
class ReflectionPadInit(object):
    """
    Function: class that execute reflection_pad
    """

    def __init__(self, mode, constant_values, padding_contiguous, kernel_name):
        self.tik_instance = tik.Tik()
        self.core_nums = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.input_gm_list = []
        self.output_gm_list = []
        self.tiling_gm = None
        self.unknown_max_shape = (Constant.MAX_INT64,)
        self.kernel_name = None
        self.opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True, "dynamic_tik": True}
        self.tiling_key = None
        self.tiling_dtype = "int64"
        self.tiling_shape = (Constant.TILING_NUMS,)
        self.kernel_name = kernel_name
        self.mode = mode
        # op para init
        self.inner_dtype = "float16"
        self.input_gm = None
        self.output_gm = None
        self.true_input_bytes_size = 0
        self.inner_bytes_size = get_bit_len(self.inner_dtype) // Constant.EIGHT_BIT
        self.block_num = constant.BLOCK_SIZE // self.inner_bytes_size
        self.output_dim4_max_cnt = (self.ub_size_bytes - Constant.RESERVED_UB_SIZE) // self.inner_bytes_size // 32
        self.output_dim4_max_cnt_block_align = (self.output_dim4_max_cnt + self.block_num - 1) // self.block_num
        self.core_num_var = self.tik_instance.Scalar(name="core_num_var", init_value=self.core_nums)
        self.mode = mode
        self.padding_contiguous = padding_contiguous
        self.constant_values = constant_values
        lis = [None] * Constant.INIT_NUM
        self.dtype_rate, self.input_dim_0, self.input_dim_1, self.input_dim_2, self.input_dim_3, self.input_dim_4,\
            self.pading_00, self.pading_01, self.pading_10, self.pading_11, self.pading_20, self.pading_21,\
            self.pading_30, self.pading_31, self.pading_40, self.pading_41, self.output_dim_0, self.output_dim_1,\
            self.output_dim_2, self.output_dim_3, self.output_dim_4, self.core_used_num, self.num_per_core,\
            self.num_tail_core, self.tiling_input_shape, self.tiling_output_shape, self.pading_value,\
            self.outer_move_out_core_offset, self.outer_move_out_index, self.in_and_out_indexes, self.dividends,\
            self.conv1_stride_0, self.conv1_stride_1, self.conv2_stride_0, self.conv2_stride_1, self.inner_outsize,\
            self.inner_insize, self.input_dim_4_align_blocks, self.output_dim_4_align_blocks, self.inner_loop_times,\
            self.inner_process_nums_per_loop, self.inner_process_nums_tail_loop, self.ub_first, self.ub_second,\
            self.ub_mode2, self.ub_addr_rollback, self.one_loop_process_lines, self.move_out_start_line,\
            self.padding40_align_blocks, self.padding41_align_blocks, self.input_dim_4_align_blocks,\
            self.tail_loop_true_nums, self.outer_move_in_index, self.outer_process_num, self.segment_total,\
            self.process_blks, self.segment_move_offset, self.num_tail_segment, self.inner_process_num,\
            self.address_rollback, self.address_rollback_sum, self.start_blocks, self.now_process_lines,\
            self.move_in_start_blocks, self.start_line = lis

    def set_running_core_num(self, tiling_core_num):
        self.core_num_var.set_as(tiling_core_num)

    def tiling_args(self):
        """
        when input shape is less 5, will. expand to 5
        tiling info:
            tiling_key:
            input_dim_0
            input_dim_1
            input_dim_2
            input_dim_3
            input_dim_4
            pading_00
            pading_01
            pading_10
            pading_11
            pading_20
            pading_21
            pading_30
            pading_31
            pading_40
            pading_41
        """
        # tiling scaler init
        self.tiling_key = self.tik_instance.Scalar(self.tiling_dtype, "tiling_key", init_value=0)
        self.input_dim_0 = self.tik_instance.Scalar(self.tiling_dtype, "input_dim_0", init_value=0)
        self.input_dim_1 = self.tik_instance.Scalar(self.tiling_dtype, "input_dim_1", init_value=0)
        self.input_dim_2 = self.tik_instance.Scalar(self.tiling_dtype, "input_dim_2", init_value=0)
        self.input_dim_3 = self.tik_instance.Scalar(self.tiling_dtype, "input_dim_3", init_value=0)
        self.input_dim_4 = self.tik_instance.Scalar(self.tiling_dtype, "input_dim_4", init_value=0)
        self.pading_00 = self.tik_instance.Scalar(self.tiling_dtype, "pading_00", init_value=0)
        self.pading_01 = self.tik_instance.Scalar(self.tiling_dtype, "pading_01", init_value=0)
        self.pading_10 = self.tik_instance.Scalar(self.tiling_dtype, "pading_10", init_value=0)
        self.pading_11 = self.tik_instance.Scalar(self.tiling_dtype, "pading_11", init_value=0)
        self.pading_20 = self.tik_instance.Scalar(self.tiling_dtype, "pading_20", init_value=0)
        self.pading_21 = self.tik_instance.Scalar(self.tiling_dtype, "pading_21", init_value=0)
        self.pading_30 = self.tik_instance.Scalar(self.tiling_dtype, "pading_30", init_value=0)
        self.pading_31 = self.tik_instance.Scalar(self.tiling_dtype, "pading_31", init_value=0)
        self.pading_40 = self.tik_instance.Scalar(self.tiling_dtype, "pading_40", init_value=0)
        self.pading_41 = self.tik_instance.Scalar(self.tiling_dtype, "pading_41", init_value=0)
        self.output_dim_0 = self.tik_instance.Scalar(self.tiling_dtype, "output_dim_0", init_value=0)
        self.output_dim_1 = self.tik_instance.Scalar(self.tiling_dtype, "output_dim_1", init_value=0)
        self.output_dim_2 = self.tik_instance.Scalar(self.tiling_dtype, "output_dim_2", init_value=0)
        self.output_dim_3 = self.tik_instance.Scalar(self.tiling_dtype, "output_dim_3", init_value=0)
        self.output_dim_4 = self.tik_instance.Scalar(self.tiling_dtype, "output_dim_4", init_value=0)
        self.core_used_num = self.tik_instance.Scalar(self.tiling_dtype, "core_used_num", init_value=0)
        self.num_per_core = self.tik_instance.Scalar(self.tiling_dtype, "num_per_core", init_value=0)
        self.num_tail_core = self.tik_instance.Scalar(self.tiling_dtype, "num_tail_core", init_value=0)

        self.tiling_input_shape = [
            self.input_dim_0, self.input_dim_1, self.input_dim_2, self.input_dim_3, self.input_dim_4
        ]
        self.tiling_output_shape = [
            self.output_dim_0, self.output_dim_1, self.output_dim_2, self.output_dim_3, self.output_dim_4
        ]
        self.pading_value = [[self.pading_00, self.pading_01], [self.pading_10, self.pading_11],
                             [self.pading_20, self.pading_21], [self.pading_30, self.pading_31],
                             [self.pading_40, self.pading_41]]
        # core scalar init
        self.outer_move_out_core_offset = self.tik_instance.Scalar(self.tiling_dtype,
                                                                   "outer_move_out_core_offset",
                                                                   init_value=0)
        self.dividends = self.tik_instance.ScalarArray(self.tiling_dtype, 20, "dividends", init_value=1)
        tiling_ub = self.tik_instance.Tensor("int64", (Constant.TILING_NUMS,), name="tiling_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, Constant.TILING_NUMS // 4, 0, 0)
        self.tiling_key.set_as(tiling_ub[0])
        self.input_dim_0.set_as(tiling_ub[1])
        self.input_dim_1.set_as(tiling_ub[2])
        self.input_dim_2.set_as(tiling_ub[3])
        self.input_dim_3.set_as(tiling_ub[4])
        self.input_dim_4.set_as(tiling_ub[5])
        self.pading_00.set_as(tiling_ub[6])
        self.pading_01.set_as(tiling_ub[7])
        self.pading_10.set_as(tiling_ub[8])
        self.pading_11.set_as(tiling_ub[9])
        self.pading_20.set_as(tiling_ub[10])
        self.pading_21.set_as(tiling_ub[11])
        self.pading_30.set_as(tiling_ub[12])
        self.pading_31.set_as(tiling_ub[13])
        self.pading_40.set_as(tiling_ub[14])
        self.pading_41.set_as(tiling_ub[15])
        self.core_used_num.set_as(tiling_ub[16])
        self.num_per_core.set_as(tiling_ub[17])
        self.num_tail_core.set_as(tiling_ub[18])
        self.set_running_core_num(tiling_ub[19])

        # calcu output_dim
        for i, _ in enumerate(self.tiling_input_shape):
            input_dims = self.tiling_input_shape[i]
            pad_left = self.pading_value[i][0]
            pad_right = self.pading_value[i][1]
            output_dims = self.tiling_output_shape[i]
            output_dims.set_as(input_dims + pad_left + pad_right)

    def init_src_dst_gm(self, input_dict, padding_dict, output_dict, constant_dict):
        """
        init gm tensor set tiling, input, paddings output tensor(gm)
        """
        self.true_input_bytes_size = get_bit_len(input_dict.get("dtype")) // Constant.EIGHT_BIT
        self.dtype_rate = self.true_input_bytes_size // self.inner_bytes_size
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype,
                                                  self.tiling_shape,
                                                  name="tiling_gm",
                                                  scope=tik.scope_gm)
        gm_in = self.tik_instance.Tensor(input_dict.get("dtype"), self.unknown_max_shape, name="x", scope=tik.scope_gm)
        gm_paddings = self.tik_instance.Tensor(padding_dict.get("dtype"),
                                               self.unknown_max_shape,
                                               name="paddings",
                                               scope=tik.scope_gm)
        gm_out = self.tik_instance.Tensor(output_dict.get("dtype"),
                                          self.unknown_max_shape,
                                          name="y",
                                          scope=tik.scope_gm)
        self.input_gm_list.append(gm_in)
        self.input_gm_list.append(gm_paddings)
        if self.constant_values:
            constant_values_gm = self.tik_instance.Tensor(constant_dict.get("dtype"),
                                          self.unknown_max_shape,
                                          name="constant_values",
                                          scope=tik.scope_gm)
            self.input_gm_list.append(constant_values_gm)
        self.output_gm_list.append(gm_out)
        self.input_gm = self.input_gm_list[0].reinterpret_cast_to(self.inner_dtype)
        self.output_gm = self.output_gm_list[0].reinterpret_cast_to(self.inner_dtype)

    def get_outer_move_in_index(self, axis_nums):
        """
        get outer move in's index
        """
        self.in_and_out_indexes[0].set_as(self.outer_move_out_index / self.dividends[0])
        self.dividends[17].set_as(self.outer_move_out_index % self.dividends[0])
        self.in_and_out_indexes[1].set_as(self.dividends[17] / self.dividends[1])
        self.dividends[18].set_as(self.dividends[17] % self.dividends[1])
        self.in_and_out_indexes[2].set_as(self.dividends[18] / self.dividends[2])
        if axis_nums == Constant.AXIS_DIMS_4:
            self.in_and_out_indexes[3].set_as(self.dividends[18] % self.dividends[2])
        with self.tik_instance.if_scope(self.input_dim_0 == self.output_dim_0):
            self.in_and_out_indexes[4].set_as(self.in_and_out_indexes[0])
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(self.in_and_out_indexes[0] < self.pading_00):
                self.in_and_out_indexes[4].set_as(self.dividends[9] - self.in_and_out_indexes[0])
            with self.tik_instance.elif_scope(self.in_and_out_indexes[0] >= self.dividends[5]):
                self.in_and_out_indexes[4].set_as(self.dividends[13] - self.in_and_out_indexes[0])
            with self.tik_instance.else_scope():
                self.in_and_out_indexes[4].set_as(self.in_and_out_indexes[0] - self.pading_00)
        with self.tik_instance.if_scope(self.input_dim_1 == self.output_dim_1):
            self.in_and_out_indexes[5].set_as(self.in_and_out_indexes[1])
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(self.in_and_out_indexes[1] < self.pading_10):
                self.in_and_out_indexes[5].set_as(self.dividends[10] - self.in_and_out_indexes[1])
            with self.tik_instance.elif_scope(self.in_and_out_indexes[1] >= self.dividends[6]):
                self.in_and_out_indexes[5].set_as(self.dividends[14] - self.in_and_out_indexes[1])
            with self.tik_instance.else_scope():
                self.in_and_out_indexes[5].set_as(self.in_and_out_indexes[1] - self.pading_10)
        with self.tik_instance.if_scope(self.input_dim_2 == self.output_dim_2):
            self.in_and_out_indexes[6].set_as(self.in_and_out_indexes[2])
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(self.in_and_out_indexes[2] < self.pading_20):
                self.in_and_out_indexes[6].set_as(self.dividends[11] - self.in_and_out_indexes[2])
            with self.tik_instance.elif_scope(self.in_and_out_indexes[2] >= self.dividends[7]):
                self.in_and_out_indexes[6].set_as(self.dividends[15] - self.in_and_out_indexes[2])
            with self.tik_instance.else_scope():
                self.in_and_out_indexes[6].set_as(self.in_and_out_indexes[2] - self.pading_20)
        if axis_nums == Constant.AXIS_DIMS_4:
            with self.tik_instance.if_scope(self.input_dim_3 == self.output_dim_3):
                self.in_and_out_indexes[7].set_as(self.in_and_out_indexes[3])
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(self.in_and_out_indexes[3] < self.pading_30):
                    self.in_and_out_indexes[7].set_as(self.dividends[12] - self.in_and_out_indexes[3])
                with self.tik_instance.elif_scope(self.in_and_out_indexes[3] >= self.dividends[8]):
                    self.in_and_out_indexes[7].set_as(self.dividends[16] - self.in_and_out_indexes[3])
                with self.tik_instance.else_scope():
                    self.in_and_out_indexes[7].set_as(self.in_and_out_indexes[3] - self.pading_30)
        if axis_nums == Constant.AXIS_DIMS_3:
            self.outer_move_in_index.set_as(self.in_and_out_indexes[4] * self.dividends[3] +
                                            self.in_and_out_indexes[5] * self.input_dim_2 + self.in_and_out_indexes[6])
        elif axis_nums == Constant.AXIS_DIMS_4:
            self.outer_move_in_index.set_as(self.in_and_out_indexes[4] * self.dividends[3] +
                                            self.in_and_out_indexes[5] * self.dividends[4] +
                                            self.in_and_out_indexes[6] * self.input_dim_3 + self.in_and_out_indexes[7])
        else:
            pass

    def set_dividends_scalars(self, axis_nums):
        """
        gen index compute args
        """
        self.dividends[5].set_as(self.pading_00 + self.tiling_input_shape[0])
        self.dividends[6].set_as(self.pading_10 + self.tiling_input_shape[1])
        self.dividends[7].set_as(self.pading_20 + self.tiling_input_shape[2])
        self.dividends[9].set_as(self.pading_00)
        self.dividends[10].set_as(self.pading_10)
        self.dividends[11].set_as(self.pading_20)
        self.dividends[13].set_as(2 * self.tiling_input_shape[0] + self.pading_00 - 2)
        self.dividends[14].set_as(2 * self.tiling_input_shape[1] + self.pading_10 - 2)
        self.dividends[15].set_as(2 * self.tiling_input_shape[2] + self.pading_20 - 2)
        if axis_nums == Constant.AXIS_DIMS_3:
            self.dividends[0].set_as(self.tiling_output_shape[1] * self.tiling_output_shape[2])
            self.dividends[1].set_as(self.tiling_output_shape[2])
            self.dividends[2].set_as(1)
            # 3 -> self.input_dim_1 * self.input_dim_2
            self.dividends[3].set_as(self.input_dim_1 * self.input_dim_2)
        elif axis_nums == Constant.AXIS_DIMS_4:
            self.dividends[0].set_as(self.tiling_output_shape[1] * self.tiling_output_shape[2] *
                                     self.tiling_output_shape[3])
            self.dividends[1].set_as(self.tiling_output_shape[2] * self.tiling_output_shape[3])
            self.dividends[2].set_as(self.tiling_output_shape[3])
            # 3,4 -> self.input_dim_1 * self.input_dim_2 * self.input_dim_3
            self.dividends[3].set_as(self.input_dim_1 * self.input_dim_2 * self.input_dim_3)
            self.dividends[4].set_as(self.input_dim_2 * self.input_dim_3)
            self.dividends[8].set_as(self.pading_30 + self.tiling_input_shape[3])
            self.dividends[12].set_as(self.pading_30)
            self.dividends[16].set_as(2 * self.tiling_input_shape[3] + self.pading_30 - 2)
        else:
            pass

    def do_tiling_key_mode_1(self, core_id):
        """
        divide by outer:[0,1,2] inner:[3,4]
        """
        self.set_dividends_scalars(Constant.AXIS_DIMS_3)
        self.outer_move_out_core_offset.set_as(self.num_per_core * core_id)
        self.address_rollback = self.tik_instance.ScalarArray(self.tiling_dtype, 4, "address_rollback", init_value=0)
        self.address_rollback_sum = self.tik_instance.Scalar(self.tiling_dtype, "address_rollback_sum", init_value=0)
        with self.tik_instance.if_scope(self.pading_31 == 0):
            self.address_rollback[3].set_as(10)
        with self.tik_instance.else_scope():
            self.address_rollback[3].set_as(20)
        self.outer_process_num = self.tik_instance.Scalar(self.tiling_dtype, "outer_process_num", init_value=0)
        with self.tik_instance.if_scope(core_id < self.core_used_num - 1):
            self.outer_process_num.set_as(self.num_per_core)
            self.address_rollback[0].set_as(1)
        with self.tik_instance.else_scope():
            self.outer_process_num.set_as(self.num_tail_core)
            self.address_rollback[0].set_as(0)
        self.outer_process_per_core()

    def do_tiling_key_mode_0(self, core_id):
        """
        divide by outer:[0,1,2,3] inner:[4]
        """
        self.set_dividends_scalars(Constant.AXIS_DIMS_4)
        self.outer_move_out_core_offset.set_as(self.num_per_core * core_id)
        self.address_rollback = self.tik_instance.ScalarArray(self.tiling_dtype, 3, "address_rollback", init_value=0)
        self.address_rollback_sum = self.tik_instance.Scalar(self.tiling_dtype, "address_rollback_sum", init_value=0)
        self.outer_process_num = self.tik_instance.Scalar(self.tiling_dtype, "outer_process_num", init_value=0)
        with self.tik_instance.if_scope(core_id < self.core_used_num - 1):
            self.outer_process_num.set_as(self.num_per_core)
            self.address_rollback[0].set_as(1)
        with self.tik_instance.else_scope():
            self.outer_process_num.set_as(self.num_tail_core)
            self.address_rollback[0].set_as(0)
        self.outer_process_per_core_mode_0()

    def do_tiling_key_mode_2(self, core_id):
        """
        padding_40 == 0 & padding_40 == 0 & dim4 32B align
        divide by outer:[0,1,2,3] inner:[4]
        """
        self.set_dividends_scalars(Constant.AXIS_DIMS_4)
        self.outer_move_out_core_offset.set_as(self.num_per_core * core_id)
        self.outer_process_num = self.tik_instance.Scalar(self.tiling_dtype, "outer_process_num", init_value=0)
        with self.tik_instance.if_scope(core_id < self.core_used_num - 1):
            self.outer_process_num.set_as(self.num_per_core)
        with self.tik_instance.else_scope():
            self.outer_process_num.set_as(self.num_tail_core)
        self.outer_process_per_core_mode_2()

    def outer_process_per_core_mode_2(self):
        """
        mode_2 outer prcocess
        """
        self.inner_process_prepare(Constant.MODE2)
        self.process_blks = self.tik_instance.Scalar(self.tiling_dtype, "process_blks", init_value=0)
        self.segment_move_offset = self.tik_instance.Scalar(self.tiling_dtype, "segment_move_offset", init_value=0)
        self.process_blks.set_as(self.output_dim_4 // Constant.TRANS_MIN_BLKS)
        segment_max_num = self.output_dim4_max_cnt_block_align * 32 // self.process_blks
        self.segment_total.set_as((self.outer_process_num + segment_max_num - 1) // segment_max_num)
        self.num_tail_segment.set_as(self.outer_process_num - (self.segment_total - 1) * segment_max_num)
        self.ub_mode2 = self.tik_instance.Tensor(self.inner_dtype, [self.output_dim4_max_cnt_block_align * 16 * 16 * 2],
                                                 name="ub_mode2",
                                                 scope=tik.scope_ubuf)
        with self.tik_instance.for_range(0, self.segment_total) as i:
            with self.tik_instance.if_scope(i < self.segment_total - 1):
                self.inner_process_num.set_as(segment_max_num)
            with self.tik_instance.else_scope():
                self.inner_process_num.set_as(self.num_tail_segment)
            self.segment_move_offset.set_as(self.outer_move_out_core_offset + i * segment_max_num)
            with self.tik_instance.for_range(0, self.inner_process_num) as j:
                self.outer_move_out_index.set_as(self.segment_move_offset + j)
                self.get_outer_move_in_index(Constant.AXIS_DIMS_4)
                self.tik_instance.data_move(self.ub_mode2[self.output_dim_4 * j],
                                            self.input_gm[self.outer_move_in_index * self.output_dim_4], 0, 1,
                                            self.process_blks, 0, 0)
            self.tik_instance.data_move(self.output_gm[self.segment_move_offset * self.output_dim_4], self.ub_mode2, 0,
                                        1, self.process_blks * self.inner_process_num, 0, 0)

    def outer_process_per_core_mode_0(self):
        """
        mode_0 outer prcocess
        """
        self.inner_process_prepare(Constant.MODE0)
        with self.tik_instance.for_range(0, self.outer_process_num) as i:
            self.outer_move_out_index.set_as(self.outer_move_out_core_offset + i)
            self.get_outer_move_in_index(Constant.AXIS_DIMS_4)
            with self.tik_instance.if_scope(i < self.outer_process_num - 1):
                self.address_rollback[1].set_as(0)
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(self.pading_41 != Constant.TRANS_MIN_BLKS):
                    self.address_rollback[1].set_as(1)
                with self.tik_instance.else_scope():
                    self.address_rollback[1].set_as(0)
            self.last_dim_inner_process()

    def outer_process_per_core(self):
        """
        mode_1 outer prcocess
        """
        self.inner_process_prepare(Constant.MODE1)
        with self.tik_instance.for_range(0, self.outer_process_num) as i:
            self.outer_move_out_index.set_as(self.outer_move_out_core_offset + i)
            self.get_outer_move_in_index(Constant.AXIS_DIMS_3)
            with self.tik_instance.if_scope(i < self.outer_process_num - 1):
                self.address_rollback[1].set_as(0)
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(self.output_dim_4 % Constant.TRANS_MIN_BLKS != 0):
                    self.address_rollback[1].set_as(1)
                with self.tik_instance.else_scope():
                    self.address_rollback[1].set_as(0)
            self.inner_process()

    def last_dim_inner_process(self):
        """
        mode_0 inner prcocess
        """
        # do padding40 segment
        with self.tik_instance.if_scope(self.pading_40 != 0):
            one_loop_process_blocks = self.output_dim4_max_cnt_block_align
            self.inner_loop_times.set_as(
                (self.padding40_align_blocks + one_loop_process_blocks - 1) / one_loop_process_blocks)
            self.inner_process_nums_tail_loop.set_as(self.padding40_align_blocks -
                                                     (self.inner_loop_times - 1) * one_loop_process_blocks)
            self.tail_loop_true_nums.set_as(self.pading_40 - (self.inner_loop_times - 1) * one_loop_process_blocks *
                                            Constant.TRANS_MIN_BLKS)
            self.move_in_start_blocks.set_as((self.inner_loop_times - 1) * one_loop_process_blocks)
            self.get_vnchw_stride(self.inner_process_nums_tail_loop, True)
            self.tik_instance.data_move(
                self.ub_first, self.input_gm[self.outer_move_in_index * self.input_dim_4 +
                                             self.move_in_start_blocks * Constant.TRANS_MIN_BLKS +
                                             self.dtype_rate], 0, 1, self.inner_process_nums_tail_loop, 0, 0)
            dst_list_0 = [self.ub_second[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
            src_list_0 = [self.ub_first] * Constant.TRANS_MIN_BLKS
            self.tik_instance.vnchwconv(False, False, dst_list_0, src_list_0, self.inner_process_nums_tail_loop,
                                        self.conv1_stride_0, self.conv1_stride_1)
            with self.tik_instance.for_range(0, self.tail_loop_true_nums / self.dtype_rate) as pading_40_index:
                self.tik_instance.data_move(
                    self.ub_first[pading_40_index * self.dtype_rate * Constant.TRANS_MIN_BLKS],
                    self.ub_second[(self.tail_loop_true_nums / self.dtype_rate - pading_40_index - 1) *
                                   self.dtype_rate * Constant.TRANS_MIN_BLKS], 0, 1, self.dtype_rate, 0, 0)
            dst_list_1 = [self.ub_second[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
            src_list_1 = [self.ub_first[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
            self.tik_instance.vnchwconv(False, False, dst_list_1, src_list_1, self.inner_process_nums_tail_loop,
                                        self.conv2_stride_0, self.conv2_stride_1)
            self.tik_instance.data_move(self.output_gm[self.outer_move_out_index * self.output_dim_4], self.ub_second,
                                        0, 1, self.inner_process_nums_tail_loop, 0, 0)
            self.get_vnchw_stride(one_loop_process_blocks, True)
            with self.tik_instance.for_range(0, self.inner_loop_times - 1) as inner_loop_index:
                self.move_in_start_blocks.set_as(
                    (self.inner_loop_times - 2 - inner_loop_index) * one_loop_process_blocks)
                self.tik_instance.data_move(
                    self.ub_first, self.input_gm[self.outer_move_in_index * self.input_dim_4 +
                                                 self.move_in_start_blocks * Constant.TRANS_MIN_BLKS +
                                                 self.dtype_rate], 0, 1,
                    one_loop_process_blocks, 0, 0)
                dst_list_0 = [self.ub_second[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
                src_list_0 = [self.ub_first] * Constant.TRANS_MIN_BLKS
                self.tik_instance.vnchwconv(False, False, dst_list_0, src_list_0, one_loop_process_blocks,
                                            self.conv1_stride_0, self.conv1_stride_1)
                with self.tik_instance.for_range(0, one_loop_process_blocks * Constant.TRANS_MIN_BLKS /
                                                 self.dtype_rate) as pading_40_index:
                    self.tik_instance.data_move(
                        self.ub_first[pading_40_index * self.dtype_rate * Constant.TRANS_MIN_BLKS],
                        self.ub_second[(one_loop_process_blocks * Constant.TRANS_MIN_BLKS / self.dtype_rate -
                                        pading_40_index - 1) * self.dtype_rate * Constant.TRANS_MIN_BLKS], 0, 1,
                        self.dtype_rate, 0, 0)
                dst_list_1 = [self.ub_second[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
                src_list_1 = [self.ub_first[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
                self.tik_instance.vnchwconv(False, False, dst_list_1, src_list_1, one_loop_process_blocks,
                                            self.conv2_stride_0, self.conv2_stride_1)
                self.tik_instance.data_move(
                    self.output_gm[self.outer_move_out_index * self.output_dim_4 + self.tail_loop_true_nums +
                                   inner_loop_index * one_loop_process_blocks * Constant.TRANS_MIN_BLKS],
                    self.ub_second, 0, 1, one_loop_process_blocks, 0, 0)
        # do input_dim_4 segment
        one_loop_process_blocks = self.output_dim4_max_cnt_block_align * 16
        self.inner_loop_times.set_as(
            (self.input_dim_4_align_blocks + one_loop_process_blocks - 1) / one_loop_process_blocks)
        self.inner_process_nums_tail_loop.set_as(self.input_dim_4_align_blocks -
                                                 (self.inner_loop_times - 1) * one_loop_process_blocks)
        with self.tik_instance.if_scope(self.inner_loop_times > 1):
            with self.tik_instance.for_range(0, self.inner_loop_times - 1) as inner_loop_index:
                self.move_in_start_blocks.set_as(inner_loop_index * one_loop_process_blocks)
                self.tik_instance.data_move(
                    self.ub_first, self.input_gm[self.outer_move_in_index * self.input_dim_4 +
                                                 self.move_in_start_blocks * Constant.TRANS_MIN_BLKS], 0, 1,
                    one_loop_process_blocks, 0, 0)
                self.tik_instance.data_move(
                    self.output_gm[self.outer_move_out_index * self.output_dim_4 + self.pading_40 +
                                   self.move_in_start_blocks * Constant.TRANS_MIN_BLKS], self.ub_first, 0, 1,
                    one_loop_process_blocks, 0, 0)
            self.tik_instance.data_move(
                self.ub_first, self.input_gm[self.outer_move_in_index * self.input_dim_4 + self.input_dim_4 -
                                             self.inner_process_nums_tail_loop * Constant.TRANS_MIN_BLKS], 0, 1,
                self.inner_process_nums_tail_loop, 0, 0)
            self.tik_instance.data_move(
                self.output_gm[self.outer_move_out_index * self.output_dim_4 + self.pading_40 + self.input_dim_4 -
                               self.inner_process_nums_tail_loop * Constant.TRANS_MIN_BLKS], self.ub_first, 0, 1,
                self.inner_process_nums_tail_loop, 0, 0)
        with self.tik_instance.if_scope(self.inner_loop_times == 1):
            self.tik_instance.data_move(self.ub_first, self.input_gm[self.outer_move_in_index * self.input_dim_4], 0, 1,
                                        self.inner_process_nums_tail_loop - 1, 0, 0)
            self.tik_instance.data_move(self.output_gm[self.outer_move_out_index * self.output_dim_4 + self.pading_40],
                                        self.ub_first, 0, 1, self.inner_process_nums_tail_loop - 1, 0, 0)
            self.tik_instance.data_move(
                self.ub_first,
                self.input_gm[self.outer_move_in_index * self.input_dim_4 + self.input_dim_4 - Constant.TRANS_MIN_BLKS],
                0, 1, 1, 0, 0)
            self.tik_instance.data_move(
                self.output_gm[self.outer_move_out_index * self.output_dim_4 + self.pading_40 + self.input_dim_4 -
                               Constant.TRANS_MIN_BLKS], self.ub_first, 0, 1, 1, 0, 0)
        # do pading_41 segment
        with self.tik_instance.if_scope(self.pading_41 != 0):
            one_loop_process_blocks = self.output_dim4_max_cnt_block_align
            self.address_rollback_sum.set_as(self.address_rollback[0] + self.address_rollback[1])
            with self.tik_instance.if_scope((self.pading_41 < 16) & (self.address_rollback_sum == 2)):
                self.tik_instance.data_move(
                    self.ub_addr_rollback, self.input_gm[self.outer_move_in_index * self.input_dim_4 +
                                                         self.input_dim_4 + self.pading_41 - Constant.TRANS_MIN_BLKS],
                    0, 1, 1, 0, 0)
                self.tik_instance.data_move(
                    self.ub_first, self.input_gm[self.outer_move_in_index * self.input_dim_4 + self.input_dim_4 -
                                                 Constant.TRANS_MIN_BLKS - self.dtype_rate],
                    0, 1, 1, 0, 0)
                with self.tik_instance.for_range(0, self.pading_41 / self.dtype_rate) as index_out:
                    with self.tik_instance.for_range(0, self.dtype_rate) as index_in:
                        self.ub_addr_rollback[index_out * self.dtype_rate + Constant.TRANS_MIN_BLKS - self.pading_41 +
                                              index_in].set_as(
                                                  self.ub_first[Constant.TRANS_MIN_BLKS -
                                                                (index_out + 1) * self.dtype_rate + index_in])
                self.tik_instance.data_move(
                    self.output_gm[(self.outer_move_out_index + 1) * self.output_dim_4 - Constant.TRANS_MIN_BLKS],
                    self.ub_addr_rollback, 0, 1, 1, 0, 0)
            with self.tik_instance.else_scope():
                self.inner_loop_times.set_as(
                    (self.padding41_align_blocks + one_loop_process_blocks - 1) / one_loop_process_blocks)
                self.inner_process_nums_tail_loop.set_as(self.padding41_align_blocks -
                                                         (self.inner_loop_times - 1) * one_loop_process_blocks)
                self.tail_loop_true_nums.set_as(self.pading_41 - (self.inner_loop_times - 1) * one_loop_process_blocks *
                                                Constant.TRANS_MIN_BLKS)
                self.tik_instance.data_move(
                    self.ub_first, self.input_gm[(self.outer_move_in_index + 1) * self.input_dim_4 -
                                                 self.tail_loop_true_nums - self.dtype_rate],
                    0, 1, self.inner_process_nums_tail_loop, 0, 0)
                dst_list_0 = [self.ub_second[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
                src_list_0 = [self.ub_first] * Constant.TRANS_MIN_BLKS
                self.get_vnchw_stride(self.inner_process_nums_tail_loop, True)
                self.tik_instance.vnchwconv(False, False, dst_list_0, src_list_0, self.inner_process_nums_tail_loop,
                                            self.conv1_stride_0, self.conv1_stride_1)
                with self.tik_instance.for_range(0, self.tail_loop_true_nums / self.dtype_rate) as pading_41_index:
                    self.tik_instance.data_move(
                        self.ub_first[pading_41_index * self.dtype_rate * Constant.TRANS_MIN_BLKS],
                        self.ub_second[(self.tail_loop_true_nums / self.dtype_rate - pading_41_index - 1) *
                                       self.dtype_rate * Constant.TRANS_MIN_BLKS], 0, 1, self.dtype_rate, 0, 0)
                dst_list_1 = [self.ub_second[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
                src_list_1 = [self.ub_first[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
                self.tik_instance.vnchwconv(False, False, dst_list_1, src_list_1, self.inner_process_nums_tail_loop,
                                            self.conv2_stride_0, self.conv2_stride_1)
                with self.tik_instance.if_scope((self.address_rollback_sum == 2) & (self.inner_loop_times == 1)):
                    for i in range(16):
                        self.ub_addr_rollback[15 - i].set_as(self.ub_second[self.tail_loop_true_nums - i - 1])
                    self.tik_instance.data_move(
                        self.output_gm[self.outer_move_out_index * self.output_dim_4 + self.pading_40 +
                                       self.input_dim_4], self.ub_second, 0, 1, self.inner_process_nums_tail_loop - 1,
                        0, 0)
                    self.tik_instance.data_move(
                        self.output_gm[self.outer_move_out_index * self.output_dim_4 + self.pading_40 +
                                       self.input_dim_4 + self.tail_loop_true_nums - 16], self.ub_addr_rollback, 0, 1,
                        1, 0, 0)
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(
                        self.output_gm[self.outer_move_out_index * self.output_dim_4 + self.pading_40 +
                                       self.input_dim_4], self.ub_second, 0, 1, self.inner_process_nums_tail_loop, 0, 0)

                self.get_vnchw_stride(one_loop_process_blocks, True)
                with self.tik_instance.for_range(0, self.inner_loop_times - 1) as inner_loop_index:
                    self.move_in_start_blocks.set_as((inner_loop_index + 1) * one_loop_process_blocks)
                    self.tik_instance.data_move(
                        self.ub_first,
                        self.input_gm[(self.outer_move_in_index + 1) * self.input_dim_4 - self.tail_loop_true_nums -
                                      self.dtype_rate - self.move_in_start_blocks * Constant.TRANS_MIN_BLKS], 0, 1,
                                      one_loop_process_blocks, 0, 0)
                    dst_list_0 = [self.ub_second[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
                    src_list_0 = [self.ub_first] * Constant.TRANS_MIN_BLKS
                    self.tik_instance.vnchwconv(False, False, dst_list_0, src_list_0, one_loop_process_blocks,
                                                self.conv1_stride_0, self.conv1_stride_1)
                    with self.tik_instance.for_range(
                            0, one_loop_process_blocks * Constant.TRANS_MIN_BLKS / self.dtype_rate) as pading_41_index:
                        self.tik_instance.data_move(
                            self.ub_first[pading_41_index * self.dtype_rate * Constant.TRANS_MIN_BLKS],
                            self.ub_second[(one_loop_process_blocks * Constant.TRANS_MIN_BLKS / self.dtype_rate -
                                            pading_41_index - 1) * self.dtype_rate * Constant.TRANS_MIN_BLKS], 0, 1,
                            self.dtype_rate, 0, 0)
                    dst_list_1 = [self.ub_second[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
                    src_list_1 = [self.ub_first[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
                    self.tik_instance.vnchwconv(False, False, dst_list_1, src_list_1, one_loop_process_blocks,
                                                self.conv2_stride_0, self.conv2_stride_1)
                    self.tik_instance.data_move(
                        self.output_gm[self.outer_move_out_index * self.output_dim_4 + self.input_dim_4 +
                                       self.pading_40 + self.tail_loop_true_nums +
                                       inner_loop_index * one_loop_process_blocks * Constant.TRANS_MIN_BLKS],
                        self.ub_second, 0, 1, one_loop_process_blocks, 0, 0)

    def get_vnchw_stride(self, line_num, is_last_dim):
        """
        if vnchwconv repeat=1 stride should be 0
        """
        if is_last_dim:
            with self.tik_instance.if_scope(line_num <= 1):
                self.conv1_stride_0.set_as(0)
                self.conv1_stride_1.set_as(0)
            with self.tik_instance.else_scope():
                self.conv1_stride_0.set_as(16)
                self.conv1_stride_1.set_as(1)
            with self.tik_instance.if_scope(line_num <= 1):
                self.conv2_stride_0.set_as(0)
                self.conv2_stride_1.set_as(0)
            with self.tik_instance.else_scope():
                self.conv2_stride_0.set_as(1)
                self.conv2_stride_1.set_as(16)
        else:
            with self.tik_instance.if_scope(line_num * self.input_dim_4 <= 16):
                self.conv1_stride_0.set_as(0)
                self.conv1_stride_1.set_as(0)
            with self.tik_instance.else_scope():
                self.conv1_stride_0.set_as(16)
                self.conv1_stride_1.set_as(1)
            with self.tik_instance.if_scope(line_num * self.output_dim_4 <= 16):
                self.conv2_stride_0.set_as(0)
                self.conv2_stride_1.set_as(0)
            with self.tik_instance.else_scope():
                self.conv2_stride_0.set_as(1)
                self.conv2_stride_1.set_as(16)

    def inner_process_prepare(self, mode):
        """
        prepare for scalar and UB
        """
        self.inner_process_num = self.tik_instance.Scalar(self.tiling_dtype, "num_tail_segment", init_value=0)
        self.in_and_out_indexes = self.tik_instance.ScalarArray(self.tiling_dtype,
                                                                8,
                                                                "in_and_out_indexes",
                                                                init_value=0)
        self.outer_move_in_index = self.tik_instance.Scalar(self.tiling_dtype, "outer_move_in_index", init_value=0)
        self.outer_move_out_index = self.tik_instance.Scalar(self.tiling_dtype, "outer_move_out_index", init_value=0)
        if mode == Constant.MODE2:
            self.segment_total = self.tik_instance.Scalar(self.tiling_dtype, "segment_total", init_value=0)
            self.num_tail_segment = self.tik_instance.Scalar(self.tiling_dtype, "num_tail_segment", init_value=0)
            self.segment_total.set_as((self.outer_process_num + Constant.TRANS_MIN_BLKS - 1) // Constant.TRANS_MIN_BLKS)
            self.num_tail_segment.set_as(self.outer_process_num - (self.segment_total - 1) * Constant.TRANS_MIN_BLKS)
        if mode in (Constant.MODE0, Constant.MODE1):
            self.conv1_stride_0 = self.tik_instance.Scalar(self.tiling_dtype, "conv1_stride_0", init_value=0)
            self.conv1_stride_1 = self.tik_instance.Scalar(self.tiling_dtype, "conv1_stride_1", init_value=0)
            self.conv2_stride_0 = self.tik_instance.Scalar(self.tiling_dtype, "conv2_stride_0", init_value=0)
            self.conv2_stride_1 = self.tik_instance.Scalar(self.tiling_dtype, "conv2_stride_1", init_value=0)
            self.inner_outsize = self.tik_instance.Scalar(self.tiling_dtype, "inner_outsize", init_value=0)
            self.inner_insize = self.tik_instance.Scalar(self.tiling_dtype, "inner_insize", init_value=0)
            self.padding40_align_blocks = self.tik_instance.Scalar(self.tiling_dtype,
                                                                   "padding40_align_blocks",
                                                                   init_value=0)
            self.padding41_align_blocks = self.tik_instance.Scalar(self.tiling_dtype,
                                                                   "padding41_align_blocks",
                                                                   init_value=0)
            self.input_dim_4_align_blocks = self.tik_instance.Scalar(self.tiling_dtype,
                                                                     "input_dim_4_align_blocks",
                                                                     init_value=0)
            self.output_dim_4_align_blocks = self.tik_instance.Scalar(self.tiling_dtype,
                                                                      "output_dim_4_align_blocks",
                                                                      init_value=0)
            self.inner_loop_times = self.tik_instance.Scalar(self.tiling_dtype, "inner_loop_times", init_value=0)
            self.inner_process_nums_tail_loop = self.tik_instance.Scalar(self.tiling_dtype,
                                                                         "inner_process_nums_tail_loop",
                                                                         init_value=0)
            self.ub_first = self.tik_instance.Tensor(self.inner_dtype, [self.output_dim4_max_cnt_block_align * 16 * 16],
                                                     name="ub_first",
                                                     scope=tik.scope_ubuf)
            self.ub_second = self.tik_instance.Tensor(self.inner_dtype,
                                                      [self.output_dim4_max_cnt_block_align * 16 * 16],
                                                      name="ub_second",
                                                      scope=tik.scope_ubuf)
            self.ub_addr_rollback = self.tik_instance.Tensor(self.inner_dtype, [16],
                                                             name="ub_addr_rollback",
                                                             scope=tik.scope_ubuf)
            self.one_loop_process_lines = self.tik_instance.Scalar(self.tiling_dtype,
                                                                   "one_loop_process_lines",
                                                                   init_value=0)
            self.tail_loop_true_nums = self.tik_instance.Scalar(self.tiling_dtype, "tail_loop_true_nums", init_value=0)
            self.padding40_align_blocks.set_as((self.pading_40 + 15) / 16)
            self.padding41_align_blocks.set_as((self.pading_41 + 15) / 16)
            self.input_dim_4_align_blocks.set_as((self.input_dim_4 + 15) / 16)
            self.output_dim_4_align_blocks.set_as((self.output_dim_4 + 15) / 16)
            self.start_blocks = self.tik_instance.Scalar(self.tiling_dtype, "start_index", init_value=0)
            self.move_in_start_blocks = self.tik_instance.Scalar(self.tiling_dtype,
                                                                 "move_in_start_blocks",
                                                                 init_value=0)
            self.now_process_lines = self.tik_instance.Scalar(self.tiling_dtype, "now_process_lines", init_value=0)
            self.start_line = self.tik_instance.Scalar(self.tiling_dtype, "start_line", init_value=0)
            self.process_blks = self.tik_instance.Scalar(self.tiling_dtype, "process_blks", init_value=0)
            self.move_out_start_line = self.tik_instance.Scalar(self.tiling_dtype, "move_out_start_line", init_value=0)
            self.inner_outsize.set_as(self.output_dim_3 * self.output_dim_4)
            self.inner_insize.set_as(self.input_dim_3 * self.input_dim_4)
            self.one_loop_process_lines.set_as(self.output_dim4_max_cnt_block_align / self.output_dim_4_align_blocks)

    def inner_process(self):
        """
        mode_1 inner prcocess
        """
        # in this tiling :input_dim_4_align_blocks <= max_blocks_per_line
        # do process padding30 segment
        with self.tik_instance.if_scope(self.pading_30 != 0):
            self.inner_loop_times.set_as(
                (self.pading_30 + self.one_loop_process_lines - 1) / self.one_loop_process_lines)
            with self.tik_instance.for_range(0, self.inner_loop_times) as inner_loop_index:
                with self.tik_instance.if_scope(inner_loop_index < self.inner_loop_times - 1):
                    self.start_line.set_as(self.pading_30 + 1 -
                                           (inner_loop_index + 1) * self.one_loop_process_lines)
                    self.move_out_start_line.set_as(inner_loop_index * self.one_loop_process_lines)
                    self.now_process_lines.set_as(self.one_loop_process_lines)
                with self.tik_instance.else_scope():
                    self.inner_process_nums_tail_loop.set_as(self.pading_30 -
                                                             (self.inner_loop_times - 1) * self.one_loop_process_lines)
                    self.start_line.set_as(1)
                    self.move_out_start_line.set_as((self.inner_loop_times - 1) * self.one_loop_process_lines)
                    self.now_process_lines.set_as(self.inner_process_nums_tail_loop)
                self.inner_pading30_process()

        # do process input_dim3 segment
        self.inner_loop_times.set_as((self.input_dim_3 + self.one_loop_process_lines - 1) / self.one_loop_process_lines)
        self.inner_process_nums_tail_loop.set_as(self.input_dim_3 -
                                                 (self.inner_loop_times - 1) * self.one_loop_process_lines)
        with self.tik_instance.for_range(0, self.inner_loop_times) as inner_loop_index:
            with self.tik_instance.if_scope(inner_loop_index < self.inner_loop_times - 1):
                self.move_out_start_line.set_as(inner_loop_index * self.one_loop_process_lines)
                self.now_process_lines.set_as(self.one_loop_process_lines)
                self.address_rollback[2].set_as(0)
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(self.address_rollback[0] * self.address_rollback[1] != 0):
                    with self.tik_instance.if_scope(self.inner_process_nums_tail_loop * self.output_dim_4 < 16):
                        self.inner_process_nums_tail_loop.set_as(16 + self.output_dim_4 - 1 / self.output_dim_4)
                        self.move_out_start_line.set_as(self.input_dim_3 - self.inner_process_nums_tail_loop)
                        self.now_process_lines.set_as(self.inner_process_nums_tail_loop)
                        self.address_rollback[2].set_as(1)
                    with self.tik_instance.elif_scope(self.inner_process_nums_tail_loop * self.output_dim_4 > 16):
                        self.inner_process_nums_tail_loop.set_as(self.input_dim_3 - (self.inner_loop_times - 1) *
                                                                 self.one_loop_process_lines)
                        self.move_out_start_line.set_as((self.inner_loop_times - 1) * self.one_loop_process_lines)
                        self.now_process_lines.set_as(self.inner_process_nums_tail_loop)
                        self.address_rollback[2].set_as(1)
                    with self.tik_instance.else_scope():
                        self.inner_process_nums_tail_loop.set_as(self.input_dim_3 - (self.inner_loop_times - 1) *
                                                                 self.one_loop_process_lines)
                        self.move_out_start_line.set_as((self.inner_loop_times - 1) * self.one_loop_process_lines)
                        self.now_process_lines.set_as(self.inner_process_nums_tail_loop)
                        self.address_rollback[2].set_as(0)
                with self.tik_instance.else_scope():
                    self.inner_process_nums_tail_loop.set_as(self.input_dim_3 -
                                                             (self.inner_loop_times - 1) * self.one_loop_process_lines)
                    self.move_out_start_line.set_as((self.inner_loop_times - 1) * self.one_loop_process_lines)
                    self.now_process_lines.set_as(self.inner_process_nums_tail_loop)
                    self.address_rollback[2].set_as(0)

            self.inner_input_dim3_process()

        # do process padding31 segment
        with self.tik_instance.if_scope(self.pading_31 != 0):
            self.inner_loop_times.set_as(
                (self.pading_31 + self.one_loop_process_lines - 1) / self.one_loop_process_lines)
            self.inner_process_nums_tail_loop.set_as(self.pading_31 -
                                                     (self.inner_loop_times - 1) * self.one_loop_process_lines)
            with self.tik_instance.for_range(0, self.inner_loop_times) as inner_loop_index:
                with self.tik_instance.if_scope(inner_loop_index < self.inner_loop_times - 1):
                    self.start_line.set_as(self.input_dim_3 - 1 -
                                           (inner_loop_index + 1) * self.one_loop_process_lines)
                    self.move_out_start_line.set_as(inner_loop_index * self.one_loop_process_lines)
                    self.now_process_lines.set_as(self.one_loop_process_lines)
                    self.address_rollback[2].set_as(0)
                with self.tik_instance.else_scope():
                    with self.tik_instance.if_scope(self.address_rollback[0] * self.address_rollback[1] != 0):
                        with self.tik_instance.if_scope(self.inner_process_nums_tail_loop * self.output_dim_4 < 16):
                            self.inner_process_nums_tail_loop.set_as(16 + self.output_dim_4 - 1 / self.output_dim_4)
                            self.start_line.set_as(self.input_dim_3 - self.pading_31 - 1)
                            self.move_out_start_line.set_as(self.pading_31 - self.inner_process_nums_tail_loop)
                            self.now_process_lines.set_as(self.inner_process_nums_tail_loop)
                            self.address_rollback[2].set_as(1)
                        with self.tik_instance.elif_scope(self.inner_process_nums_tail_loop * self.output_dim_4 > 16):
                            self.start_line.set_as(self.input_dim_3 - self.pading_31 - 1)
                            self.move_out_start_line.set_as((self.inner_loop_times - 1) * self.one_loop_process_lines)
                            self.inner_process_nums_tail_loop.set_as(self.pading_31 - (self.inner_loop_times - 1) *
                                                                     self.one_loop_process_lines)
                            self.now_process_lines.set_as(self.inner_process_nums_tail_loop)
                            self.address_rollback[2].set_as(1)
                        with self.tik_instance.else_scope():
                            self.start_line.set_as(self.input_dim_3 - self.pading_31 - 1)
                            self.move_out_start_line.set_as((self.inner_loop_times - 1) * self.one_loop_process_lines)
                            self.inner_process_nums_tail_loop.set_as(self.pading_31 - (self.inner_loop_times - 1) *
                                                                     self.one_loop_process_lines)
                            self.now_process_lines.set_as(self.inner_process_nums_tail_loop)
                            self.address_rollback[2].set_as(0)
                    with self.tik_instance.else_scope():
                        self.start_line.set_as(self.input_dim_3 - self.pading_31 - 1)
                        self.move_out_start_line.set_as((self.inner_loop_times - 1) * self.one_loop_process_lines)
                        self.inner_process_nums_tail_loop.set_as(self.pading_31 - (self.inner_loop_times - 1) *
                                                                 self.one_loop_process_lines)
                        self.now_process_lines.set_as(self.inner_process_nums_tail_loop)
                        self.address_rollback[2].set_as(0)
                self.inner_pading31_process()

    def inner_pading30_process(self):
        """
        pading dim3 left padding by data_move
        """
        self.process_blks.set_as((self.now_process_lines * self.input_dim_4 + 15) / 16)
        self.get_vnchw_stride(self.now_process_lines, False)
        self.tik_instance.data_move(
            self.ub_first,
            self.input_gm[self.outer_move_in_index * self.inner_insize + self.start_line * self.input_dim_4], 0, 1,
            self.process_blks, 0, 0)
        dst_list_0 = [self.ub_second[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
        src_list_0 = [self.ub_first] * Constant.TRANS_MIN_BLKS
        self.tik_instance.vnchwconv(False, False, dst_list_0, src_list_0, self.process_blks, self.conv1_stride_0,
                                    self.conv1_stride_1)
        with self.tik_instance.for_range(0, self.now_process_lines) as line_index:
            self.tik_instance.data_move(
                self.ub_first[line_index * self.input_dim_4 * Constant.TRANS_MIN_BLKS],
                self.ub_second[(self.now_process_lines - line_index - 1) * self.input_dim_4 * Constant.TRANS_MIN_BLKS],
                0, 1, self.input_dim_4, 0, 0)
        self.do_data_move_for_pading_dim_4(self.ub_first, self.ub_second, self.now_process_lines)
        self.process_blks.set_as((self.now_process_lines * self.output_dim_4 + 15) / 16)
        dst_list_1 = [self.ub_first[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
        src_list_1 = [self.ub_second[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
        self.tik_instance.vnchwconv(False, False, dst_list_1, src_list_1, self.process_blks, self.conv2_stride_0,
                                    self.conv2_stride_1)
        self.tik_instance.data_move(
            self.output_gm[self.outer_move_out_index * self.inner_outsize +
                           self.move_out_start_line * self.output_dim_4], self.ub_first, 0, 1, self.process_blks, 0, 0)

    def inner_input_dim3_process(self):
        """
        pading dim3 padding by data_move
        """

        self.get_vnchw_stride(self.now_process_lines, False)
        self.process_blks.set_as((self.now_process_lines * self.input_dim_4 + 15) / 16)
        self.tik_instance.data_move(
            self.ub_first,
            self.input_gm[self.outer_move_in_index * self.inner_insize + self.move_out_start_line * self.input_dim_4],
            0, 1, self.process_blks, 0, 0)
        dst_list_0 = [self.ub_second[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
        src_list_0 = [self.ub_first] * Constant.TRANS_MIN_BLKS
        self.tik_instance.vnchwconv(False, False, dst_list_0, src_list_0, self.process_blks, self.conv1_stride_0,
                                    self.conv1_stride_1)
        self.do_data_move_for_pading_dim_4(self.ub_second, self.ub_first, self.now_process_lines)
        self.process_blks.set_as((self.now_process_lines * self.output_dim_4 + 15) / 16)
        dst_list_1 = [self.ub_second[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
        src_list_1 = [self.ub_first[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
        self.tik_instance.vnchwconv(False, False, dst_list_1, src_list_1, self.process_blks, self.conv2_stride_0,
                                    self.conv2_stride_1)
        self.address_rollback_sum.set_as(self.address_rollback[0] + self.address_rollback[1] +
                                         self.address_rollback[2] + self.address_rollback[3])
        with self.tik_instance.if_scope(self.address_rollback_sum == 13):
            self.tik_instance.data_move(
                self.output_gm[self.outer_move_out_index * self.inner_outsize +
                               (self.pading_30 + self.move_out_start_line) * self.output_dim_4], self.ub_second, 0, 1,
                self.process_blks - 1, 0, 0)
            for i in range(16):
                self.ub_addr_rollback[15 - i].set_as(self.ub_second[self.now_process_lines * self.output_dim_4 - i - 1])
            self.tik_instance.data_move(self.output_gm[(self.outer_move_out_index + 1) * self.inner_outsize - 16],
                                        self.ub_addr_rollback, 0, 1, 1, 0, 0)
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(
                self.output_gm[self.outer_move_out_index * self.inner_outsize +
                               (self.pading_30 + self.move_out_start_line) * self.output_dim_4], self.ub_second, 0, 1,
                self.process_blks, 0, 0)

    def inner_pading31_process(self):
        """
        pading dim3 right padding by data_move
        """
        self.get_vnchw_stride(self.now_process_lines, False)
        self.process_blks.set_as((self.now_process_lines * self.input_dim_4 + 15) / 16)
        self.tik_instance.data_move(
            self.ub_first,
            self.input_gm[self.outer_move_in_index * self.inner_insize + self.start_line * self.input_dim_4], 0, 1,
            self.process_blks, 0, 0)
        dst_list_0 = [self.ub_second[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
        src_list_0 = [self.ub_first] * Constant.TRANS_MIN_BLKS
        self.tik_instance.vnchwconv(False, False, dst_list_0, src_list_0, self.process_blks, self.conv1_stride_0,
                                    self.conv1_stride_1)
        with self.tik_instance.for_range(0, self.now_process_lines) as line_index:
            self.tik_instance.data_move(
                self.ub_first[line_index * self.input_dim_4 * Constant.TRANS_MIN_BLKS],
                self.ub_second[(self.now_process_lines - line_index - 1) * self.input_dim_4 * Constant.TRANS_MIN_BLKS],
                0, 1, self.input_dim_4, 0, 0)
        self.do_data_move_for_pading_dim_4(self.ub_first, self.ub_second, self.now_process_lines)
        self.process_blks.set_as((self.now_process_lines * self.output_dim_4 + 15) / 16)
        dst_list_1 = [self.ub_first[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
        src_list_1 = [self.ub_second[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
        self.tik_instance.vnchwconv(False, False, dst_list_1, src_list_1, self.process_blks, self.conv2_stride_0,
                                    self.conv2_stride_1)
        self.address_rollback_sum.set_as(self.address_rollback[0] + self.address_rollback[1] +
                                         self.address_rollback[2] + self.address_rollback[3])
        with self.tik_instance.if_scope(self.address_rollback_sum == 23):
            self.tik_instance.data_move(
                self.output_gm[self.outer_move_out_index * self.inner_outsize +
                               (self.pading_30 + self.input_dim_3 + self.move_out_start_line) * self.output_dim_4],
                self.ub_first, 0, 1, self.process_blks - 1, 0, 0)
            for i in range(16):
                self.ub_addr_rollback[15 - i].set_as(self.ub_first[self.now_process_lines * self.output_dim_4 - i - 1])
            self.tik_instance.data_move(self.output_gm[(self.outer_move_out_index + 1) * self.inner_outsize - 16],
                                        self.ub_addr_rollback, 0, 1, 1, 0, 0)
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(
                self.output_gm[self.outer_move_out_index * self.inner_outsize +
                               (self.pading_30 + self.input_dim_3 + self.move_out_start_line) * self.output_dim_4],
                self.ub_first, 0, 1, self.process_blks, 0, 0)

    def do_data_move_for_pading_dim_4(self, in_ub, out_ub, process_lines):
        """
        pading last dim by data_move
        """
        self.tik_instance.data_move(out_ub[self.pading_40 * Constant.TRANS_MIN_BLKS], in_ub, 0, process_lines,
                                    self.input_dim_4, 0, self.pading_40 + self.pading_41)
        with self.tik_instance.for_range(0, self.pading_40 / self.dtype_rate) as pading_40_index:
            self.tik_instance.data_move(
                out_ub[pading_40_index * self.dtype_rate * Constant.TRANS_MIN_BLKS],
                in_ub[(self.pading_40 - pading_40_index * self.dtype_rate) *
                      Constant.TRANS_MIN_BLKS], 0, process_lines, self.dtype_rate, self.input_dim_4 - self.dtype_rate,
                self.output_dim_4 - self.dtype_rate)
        with self.tik_instance.for_range(0, self.pading_41 / self.dtype_rate) as pading_41_index:
            self.tik_instance.data_move(
                out_ub[(self.pading_40 + self.input_dim_4 + pading_41_index * self.dtype_rate) *
                       Constant.TRANS_MIN_BLKS],
                in_ub[(self.input_dim_4 - (pading_41_index + 2) * self.dtype_rate) *
                      Constant.TRANS_MIN_BLKS], 0, process_lines, self.dtype_rate, self.input_dim_4 - self.dtype_rate,
                self.output_dim_4 - self.dtype_rate)
            
    def do_tiling_key_mode_3(self, core_id):
        """
        divide by outer:[0,1,2] inner:[3,4]
        """
        self.set_dividends_scalars(Constant.AXIS_DIMS_3)
        self.outer_move_out_core_offset.set_as(self.num_per_core * core_id)
        self.address_rollback = self.tik_instance.ScalarArray(self.tiling_dtype, 4, "address_rollback", init_value=0)
        self.address_rollback_sum = self.tik_instance.Scalar(self.tiling_dtype, "address_rollback_sum", init_value=0)
        with self.tik_instance.if_scope(self.pading_31 == 0):
            self.address_rollback[3].set_as(10)
        with self.tik_instance.else_scope():
            self.address_rollback[3].set_as(20)
        self.outer_process_num = self.tik_instance.Scalar(self.tiling_dtype, "outer_process_num", init_value=0)
        with self.tik_instance.if_scope(core_id < self.core_used_num - 1):
            self.outer_process_num.set_as(self.num_per_core)
            self.address_rollback[0].set_as(0)
        with self.tik_instance.else_scope():
            self.outer_process_num.set_as(self.num_tail_core)
            self.address_rollback[0].set_as(0)
        self.outer_process_per_core_mode_3()

    def outer_process_per_core_mode_3(self):
        """
        mode_3 outer prcocess
        """
        self.inner_process_prepare(Constant.MODE1)
        with self.tik_instance.for_range(0, self.outer_process_num) as i:
            self.outer_move_out_index.set_as(self.outer_move_out_core_offset + i)
            self.get_outer_move_in_index(Constant.AXIS_DIMS_3)
            with self.tik_instance.if_scope(i < self.outer_process_num - 1):
                self.address_rollback[1].set_as(0)
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(self.output_dim_4 % Constant.TRANS_MIN_BLKS != 0):
                    self.address_rollback[1].set_as(1)
                with self.tik_instance.else_scope():
                    self.address_rollback[1].set_as(0)
            self.inner_process_mode_3()

    def inner_process_mode_3(self):
        """
        mode_3 inner prcocess
        """
        # in this tiling :input_dim_4_align_blocks <= max_blocks_per_line
        # do process padding30 segment
        with self.tik_instance.if_scope(self.pading_30 != 0):
            self.inner_loop_times.set_as(
                (self.pading_30 + self.one_loop_process_lines - 1) / self.one_loop_process_lines)
            with self.tik_instance.for_range(0, self.inner_loop_times) as inner_loop_index:
                with self.tik_instance.if_scope(inner_loop_index < self.inner_loop_times - 1):
                    self.start_line.set_as(self.pading_30 + 1 -
                                           (inner_loop_index + 1) * self.one_loop_process_lines)
                    self.move_out_start_line.set_as(inner_loop_index * self.one_loop_process_lines)
                    self.now_process_lines.set_as(self.one_loop_process_lines)
                with self.tik_instance.else_scope():
                    self.inner_process_nums_tail_loop.set_as(self.pading_30 -
                                                             (self.inner_loop_times - 1) * self.one_loop_process_lines)
                    self.start_line.set_as(1)
                    self.move_out_start_line.set_as((self.inner_loop_times - 1) * self.one_loop_process_lines)
                    self.now_process_lines.set_as(self.inner_process_nums_tail_loop)
                self.inner_pading30_process()

        # do process input_dim3 segment
        self.inner_loop_times.set_as((self.input_dim_3 + self.one_loop_process_lines - 1) / self.one_loop_process_lines)
        self.inner_process_nums_tail_loop.set_as(self.input_dim_3 -
                                                 (self.inner_loop_times - 1) * self.one_loop_process_lines)
        with self.tik_instance.for_range(0, self.inner_loop_times) as inner_loop_index:
            with self.tik_instance.if_scope(inner_loop_index < self.inner_loop_times - 1):
                self.move_out_start_line.set_as(inner_loop_index * self.one_loop_process_lines)
                self.now_process_lines.set_as(self.one_loop_process_lines)
                self.address_rollback[2].set_as(0)
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(self.address_rollback[0] * self.address_rollback[1] != 0):
                    with self.tik_instance.if_scope(self.inner_process_nums_tail_loop * self.output_dim_4 < 16):
                        self.inner_process_nums_tail_loop.set_as(16 + self.output_dim_4 - 1 / self.output_dim_4)
                        self.move_out_start_line.set_as(self.input_dim_3 - self.inner_process_nums_tail_loop)
                        self.now_process_lines.set_as(self.inner_process_nums_tail_loop)
                        self.address_rollback[2].set_as(1)
                    with self.tik_instance.elif_scope(self.inner_process_nums_tail_loop * self.output_dim_4 > 16):
                        self.inner_process_nums_tail_loop.set_as(self.input_dim_3 - (self.inner_loop_times - 1) *
                                                                 self.one_loop_process_lines)
                        self.move_out_start_line.set_as((self.inner_loop_times - 1) * self.one_loop_process_lines)
                        self.now_process_lines.set_as(self.inner_process_nums_tail_loop)
                        self.address_rollback[2].set_as(1)
                    with self.tik_instance.else_scope():
                        self.inner_process_nums_tail_loop.set_as(self.input_dim_3 - (self.inner_loop_times - 1) *
                                                                 self.one_loop_process_lines)
                        self.move_out_start_line.set_as((self.inner_loop_times - 1) * self.one_loop_process_lines)
                        self.now_process_lines.set_as(self.inner_process_nums_tail_loop)
                        self.address_rollback[2].set_as(0)
                with self.tik_instance.else_scope():
                    self.inner_process_nums_tail_loop.set_as(self.input_dim_3 -
                                                             (self.inner_loop_times - 1) * self.one_loop_process_lines)
                    self.move_out_start_line.set_as((self.inner_loop_times - 1) * self.one_loop_process_lines)
                    self.now_process_lines.set_as(self.inner_process_nums_tail_loop)
                    self.address_rollback[2].set_as(1)

            self.inner_input_dim3_process_mode_3()

        # do process padding31 segment
        with self.tik_instance.if_scope(self.pading_31 != 0):
            self.inner_loop_times.set_as(
                (self.pading_31 + self.one_loop_process_lines - 1) / self.one_loop_process_lines)
            self.inner_process_nums_tail_loop.set_as(self.pading_31 -
                                                     (self.inner_loop_times - 1) * self.one_loop_process_lines)
            with self.tik_instance.for_range(0, self.inner_loop_times) as inner_loop_index:
                with self.tik_instance.if_scope(inner_loop_index < self.inner_loop_times - 1):
                    self.start_line.set_as(self.input_dim_3 - 1 -
                                           (inner_loop_index + 1) * self.one_loop_process_lines)
                    self.move_out_start_line.set_as(inner_loop_index * self.one_loop_process_lines)
                    self.now_process_lines.set_as(self.one_loop_process_lines)
                    self.address_rollback[2].set_as(0)
                with self.tik_instance.else_scope():
                    with self.tik_instance.if_scope(self.address_rollback[0] * self.address_rollback[1] != 0):
                        with self.tik_instance.if_scope(self.inner_process_nums_tail_loop * self.output_dim_4 < 16):
                            self.inner_process_nums_tail_loop.set_as(16 + self.output_dim_4 - 1 / self.output_dim_4)
                            self.start_line.set_as(self.input_dim_3 - self.pading_31 - 1)
                            self.move_out_start_line.set_as(self.pading_31 - self.inner_process_nums_tail_loop)
                            self.now_process_lines.set_as(self.inner_process_nums_tail_loop)
                            self.address_rollback[2].set_as(1)
                        with self.tik_instance.elif_scope(self.inner_process_nums_tail_loop * self.output_dim_4 > 16):
                            self.start_line.set_as(self.input_dim_3 - self.pading_31 - 1)
                            self.move_out_start_line.set_as((self.inner_loop_times - 1) * self.one_loop_process_lines)
                            self.inner_process_nums_tail_loop.set_as(self.pading_31 - (self.inner_loop_times - 1) *
                                                                     self.one_loop_process_lines)
                            self.now_process_lines.set_as(self.inner_process_nums_tail_loop)
                            self.address_rollback[2].set_as(1)
                        with self.tik_instance.else_scope():
                            self.start_line.set_as(self.input_dim_3 - self.pading_31 - 1)
                            self.move_out_start_line.set_as((self.inner_loop_times - 1) * self.one_loop_process_lines)
                            self.inner_process_nums_tail_loop.set_as(self.pading_31 - (self.inner_loop_times - 1) *
                                                                     self.one_loop_process_lines)
                            self.now_process_lines.set_as(self.inner_process_nums_tail_loop)
                            self.address_rollback[2].set_as(0)
                    with self.tik_instance.else_scope():
                        self.start_line.set_as(self.input_dim_3 - self.pading_31 - 1)
                        self.move_out_start_line.set_as((self.inner_loop_times - 1) * self.one_loop_process_lines)
                        self.inner_process_nums_tail_loop.set_as(self.pading_31 - (self.inner_loop_times - 1) *
                                                                 self.one_loop_process_lines)
                        self.now_process_lines.set_as(self.inner_process_nums_tail_loop)
                        self.address_rollback[2].set_as(0)
                self.inner_pading31_process_mode_3()

    def inner_input_dim3_process_mode_3(self):
        """
        pading dim3 padding by data_move
        """

        self.get_vnchw_stride(self.now_process_lines, False)
        self.process_blks.set_as((self.now_process_lines * self.input_dim_4 + 15) / 16)
        self.tik_instance.data_move(
            self.ub_first,
            self.input_gm[self.outer_move_in_index * self.inner_insize + self.move_out_start_line * self.input_dim_4],
            0, 1, self.process_blks, 0, 0)
        dst_list_0 = [self.ub_second[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
        src_list_0 = [self.ub_first] * Constant.TRANS_MIN_BLKS
        self.tik_instance.vnchwconv(False, False, dst_list_0, src_list_0, self.process_blks, self.conv1_stride_0,
                                    self.conv1_stride_1)
        self.do_data_move_for_pading_dim_4(self.ub_second, self.ub_first, self.now_process_lines)
        self.process_blks.set_as((self.now_process_lines * self.output_dim_4 + 15) / 16)
        dst_list_1 = [self.ub_second[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
        src_list_1 = [self.ub_first[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
        self.tik_instance.vnchwconv(False, False, dst_list_1, src_list_1, self.process_blks, self.conv2_stride_0,
                                    self.conv2_stride_1)
        self.address_rollback_sum.set_as(self.address_rollback[0] + self.address_rollback[1] +
                                         self.address_rollback[2] + self.address_rollback[3])
        with self.tik_instance.if_scope(self.address_rollback_sum == 13):
            self.tik_instance.data_move(
                self.output_gm[self.outer_move_out_index * self.inner_outsize +
                               (self.pading_30 + self.move_out_start_line) * self.output_dim_4], self.ub_second, 0, 1,
                self.process_blks - 1, 0, 0)
            for i in range(16):
                self.ub_addr_rollback[15 - i].set_as(self.ub_second[self.now_process_lines * self.output_dim_4 - i - 1])
            self.tik_instance.data_move(self.output_gm[(self.outer_move_out_index + 1) * self.inner_outsize - 16],
                                        self.ub_addr_rollback, 0, 1, 1, 0, 0)
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(
                self.output_gm[self.outer_move_out_index * self.inner_outsize +
                               (self.pading_30 + self.move_out_start_line) * self.output_dim_4], self.ub_second, 0, 1,
                self.process_blks, 0, 0)
        with self.tik_instance.if_scope(self.address_rollback[2] == 1):
            for i in range(16):
                self.ub_addr_rollback[15 - i].set_as(self.ub_second[self.now_process_lines * self.output_dim_4 - i - 1])

    def inner_pading31_process_mode_3(self):
        """
        pading dim3 right padding by data_move
        """
        self.get_vnchw_stride(self.now_process_lines, False)
        self.process_blks.set_as((self.now_process_lines * self.input_dim_4 + 15) / 16)
        self.tik_instance.data_move(
            self.ub_first,
            self.input_gm[self.outer_move_in_index * self.inner_insize + self.start_line * self.input_dim_4], 0, 1,
            self.process_blks, 0, 0)
        dst_list_0 = [self.ub_second[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
        src_list_0 = [self.ub_first] * Constant.TRANS_MIN_BLKS
        self.tik_instance.vnchwconv(False, False, dst_list_0, src_list_0, self.process_blks, self.conv1_stride_0,
                                    self.conv1_stride_1)
        with self.tik_instance.for_range(0, self.now_process_lines) as line_index:
            self.tik_instance.data_move(
                self.ub_first[line_index * self.input_dim_4 * Constant.TRANS_MIN_BLKS],
                self.ub_second[(self.now_process_lines - line_index - 1) * self.input_dim_4 * Constant.TRANS_MIN_BLKS],
                0, 1, self.input_dim_4, 0, 0)
        self.do_data_move_for_pading_dim_4(self.ub_first, self.ub_second, self.now_process_lines)
        self.process_blks.set_as((self.now_process_lines * self.output_dim_4 + 15) / 16)
        dst_list_1 = [self.ub_first[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
        src_list_1 = [self.ub_second[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
        self.tik_instance.vnchwconv(False, False, dst_list_1, src_list_1, self.process_blks, self.conv2_stride_0,
                                    self.conv2_stride_1)
        self.address_rollback_sum.set_as(self.address_rollback[0] + self.address_rollback[1] +
                                         self.address_rollback[2] + self.address_rollback[3])
        with self.tik_instance.if_scope(self.address_rollback_sum == 23):
            self.tik_instance.data_move(
                self.output_gm[self.outer_move_out_index * self.inner_outsize +
                               (self.pading_30 + self.input_dim_3 + self.move_out_start_line) * self.output_dim_4],
                self.ub_first, 0, 1, self.process_blks - 1, 0, 0)
            for i in range(16):
                self.ub_addr_rollback[15 - i].set_as(self.ub_first[self.now_process_lines * self.output_dim_4 - i - 1])
            self.tik_instance.data_move(self.output_gm[(self.outer_move_out_index + 1) * self.inner_outsize - 16],
                                        self.ub_addr_rollback, 0, 1, 1, 0, 0)
        with self.tik_instance.else_scope():
            offset = 16 - self.pading_31 * self.output_dim_4
            with self.tik_instance.for_range(0, self.pading_31 * self.output_dim_4) as i:
                self.ub_first[15 - i].set_as(self.ub_first[15 - i - offset])
            with self.tik_instance.for_range(0, offset) as i:
                self.ub_first[i].set_as(self.ub_addr_rollback[16 - offset + i])
            self.tik_instance.data_move(
                self.output_gm[self.outer_move_out_index * self.inner_outsize +
                               (self.pading_30 + self.input_dim_3 + self.move_out_start_line) * self.output_dim_4 -
                                offset],
                self.ub_first, 0, 1, self.process_blks, 0, 0)

    def reflection_pad_compute_tiling(self):
        """
        reflection_pad_compute_tiling
        """
        self.tiling_args()
        with self.tik_instance.for_range(0, self.core_num_var, block_num=self.core_num_var) as core_index:
            with self.tik_instance.if_scope(core_index < self.core_used_num):
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

    def reflection_pad_compute(self):
        """
        reflection_pad_compute
        do_pad with different tiling key
        Constant.MODE0: the last dim of output > 128*core_num, and cut by last dim
                and do pad with data move
        Constant.MODE1: the last dim of output => 960, and cut by outer dim(0-3)
                and do pad with data move
        """
        self.reflection_pad_compute_tiling()
        is_support_data_move_pad = tbe_platform.api_check_support("tik.data_move_pad")
        wr_compile_info = {
            "output_dim4_max_cnt": self.output_dim4_max_cnt,
            "core_num": self.core_nums,
            "dtype_rate": self.dtype_rate,
            "mode": self.mode,
            "padding_contiguous": self.padding_contiguous,
            "is_support_data_move_pad": is_support_data_move_pad
        }
        tbe_context.get_context().add_compile_info("vars", wr_compile_info)
        flowtable_list = [self.tiling_gm]
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=self.input_gm_list,
                                   outputs=self.output_gm_list,
                                   flowtable=flowtable_list,
                                   config=self.opt_config)

        return self.tik_instance


# 'pylint: disable=huawei-too-many-arguments
@register_operator("PadV3")
def reflection_pad_v3(x, paddings, constant_values, y, mode, 
                       padding_contiguous=True, kernel_name="reflection_pad_v3"):
    """ calculating pad tensor by paddings parameters

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
    mode:str
        the cal mode of op
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
    supported_dtype = ("float16", "int16", "uint16", "float32", "int32", "int64", "uint32")
    para_check.check_dtype(src_dtype, supported_dtype, param_name="x")
    para_check.check_dtype(paddings_dtype, ("int32", "int64"), param_name="paddings")
    obj = ReflectionPadInit(mode, constant_values, padding_contiguous, kernel_name)
    obj.init_src_dst_gm(x, paddings, y, constant_values)

    return obj.reflection_pad_compute()
