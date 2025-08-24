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
replication_pad_v3.py
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
    MAX_UINT64 = 2**64 - 1
    # compute only zero axis, cut last dim
    MODE0 = 0
    # compute only one axis
    MODE1 = 1
    # network case
    MODE2 = 2
    # divide first 4 axis
    MODE4 = 4
    # tiling mode 1 divide by outer:[0,1,2] inner:[3,4]
    AXIS_DIMS_3 = 3
    # tiling mode 0 divide by outer:[0,1,2,3] inner:[4]
    AXIS_DIMS_4 = 4
    # reserved ub size
    RESERVED_UB_SIZE = 10240
    # VEC once process length `16 * 16`
    VEC_LENGTH = 256
    # fp32 mask num
    FP32_MASK = 64
    # fp16 mask num
    FP16_MASK = 128
    # 8 blocks
    COMMON_STRIDE = 8
    # two copies
    TWO_COPIES = 2
    # dtype rates two
    DTYPE_RATE_TWO = 2
    # dtype rates one
    DTYPE_RATE_ONE = 1
    INIT_NUM = 77


# 'pylint: disable=too-many-instance-attributes,too-many-statements,too-many-locals,too-many-lines
# 'pylint: disable=too-many-arguments,invalid-name,too-many-public-methods
class ReplicationPadInit(object):
    """
    Function: class that execute replication_pad
    """

    def __init__(self, mode, constant_values, padding_contiguous, kernel_name):
        self.tik_instance = tik.Tik()
        self.core_nums = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.input_gm_list = []
        self.output_gm_list = []
        self.tiling_gm = None
        self.unknown_max_shape = (Constant.MAX_UINT64,)
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
            self.padding_00, self.padding_01, self.padding_10, self.padding_11, self.padding_20, self.padding_21,\
            self.padding_30, self.padding_31, self.padding_40, self.padding_41, self.output_dim_0, self.output_dim_1,\
            self.output_dim_2, self.output_dim_3, self.output_dim_4, self.core_used_num, self.num_per_core,\
            self.num_tail_core, self.tiling_input_shape, self.tiling_output_shape, self.padding_value,\
            self.outer_move_out_core_offset, self.outer_move_out_index, self.in_and_out_indexes, self.dividends,\
            self.conv1_stride_0, self.conv1_stride_1, self.conv2_stride_0, self.conv2_stride_1, self.inner_outsize,\
            self.inner_insize, self.input_dim_4_align_blocks, self.output_dim_4_align_blocks, self.inner_loop_times,\
            self.inner_process_nums_per_loop, self.inner_process_nums_tail_loop, self.ub_first, self.ub_second,\
            self.ub_mode2, self.ub_addr_rollback, self.one_loop_process_lines, self.move_out_start_line,\
            self.padding40_align_blocks, self.padding41_align_blocks, self.input_dim_4_align_blocks,\
            self.tail_loop_true_nums, self.outer_move_in_index, self.outer_process_num, self.segment_total,\
            self.process_blks, self.segment_move_offset, self.num_tail_segment, self.inner_process_num,\
            self.address_rollback, self.address_rollback_sum, self.start_blocks, self.now_process_lines,\
            self.move_in_start_blocks, self.start_line, self.inner_col_num_non_last_row, self.inner_col_num_last_row,\
            self.inner_row_num, self.loop_num, self.dim2_rest_outer_num, self.dim2_outer_move_out_offset,\
            self.process_blk_in, self.input_ceil_algin, self.process_blk_out, self.output_ceil_algin,\
            self.dim2_src_offset, self.dim2_outer_loop_num = lis

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
            padding_00
            padding_01
            padding_10
            padding_11
            padding_20
            padding_21
            padding_30
            padding_31
            padding_40
            padding_41
        """
        # tiling scaler init
        self.tiling_key = self.tik_instance.Scalar(self.tiling_dtype, "tiling_key", init_value=0)
        self.input_dim_0 = self.tik_instance.Scalar(self.tiling_dtype, "input_dim_0", init_value=0)
        self.input_dim_1 = self.tik_instance.Scalar(self.tiling_dtype, "input_dim_1", init_value=0)
        self.input_dim_2 = self.tik_instance.Scalar(self.tiling_dtype, "input_dim_2", init_value=0)
        self.input_dim_3 = self.tik_instance.Scalar(self.tiling_dtype, "input_dim_3", init_value=0)
        self.input_dim_4 = self.tik_instance.Scalar(self.tiling_dtype, "input_dim_4", init_value=0)
        self.padding_00 = self.tik_instance.Scalar(self.tiling_dtype, "padding_00", init_value=0)
        self.padding_01 = self.tik_instance.Scalar(self.tiling_dtype, "padding_01", init_value=0)
        self.padding_10 = self.tik_instance.Scalar(self.tiling_dtype, "padding_10", init_value=0)
        self.padding_11 = self.tik_instance.Scalar(self.tiling_dtype, "padding_11", init_value=0)
        self.padding_20 = self.tik_instance.Scalar(self.tiling_dtype, "padding_20", init_value=0)
        self.padding_21 = self.tik_instance.Scalar(self.tiling_dtype, "padding_21", init_value=0)
        self.padding_30 = self.tik_instance.Scalar(self.tiling_dtype, "padding_30", init_value=0)
        self.padding_31 = self.tik_instance.Scalar(self.tiling_dtype, "padding_31", init_value=0)
        self.padding_40 = self.tik_instance.Scalar(self.tiling_dtype, "padding_40", init_value=0)
        self.padding_41 = self.tik_instance.Scalar(self.tiling_dtype, "padding_41", init_value=0)
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
        self.padding_value = [[self.padding_00, self.padding_01], [self.padding_10, self.padding_11],
                             [self.padding_20, self.padding_21], [self.padding_30, self.padding_31],
                             [self.padding_40, self.padding_41]]
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
        self.padding_00.set_as(tiling_ub[6])
        self.padding_01.set_as(tiling_ub[7])
        self.padding_10.set_as(tiling_ub[8])
        self.padding_11.set_as(tiling_ub[9])
        self.padding_20.set_as(tiling_ub[10])
        self.padding_21.set_as(tiling_ub[11])
        self.padding_30.set_as(tiling_ub[12])
        self.padding_31.set_as(tiling_ub[13])
        self.padding_40.set_as(tiling_ub[14])
        self.padding_41.set_as(tiling_ub[15])
        self.core_used_num.set_as(tiling_ub[16])
        self.num_per_core.set_as(tiling_ub[17])
        self.num_tail_core.set_as(tiling_ub[18])
        self.set_running_core_num(tiling_ub[19])

        # calcu output_dim
        for i, _ in enumerate(self.tiling_input_shape):
            input_dims = self.tiling_input_shape[i]
            pad_left = self.padding_value[i][0]
            pad_right = self.padding_value[i][1]
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
            with self.tik_instance.if_scope(self.in_and_out_indexes[0] < self.padding_00):
                self.in_and_out_indexes[4].set_as(self.dividends[9] - self.in_and_out_indexes[0])
            with self.tik_instance.elif_scope(self.in_and_out_indexes[0] >= self.dividends[5]):
                self.in_and_out_indexes[4].set_as(self.dividends[13] - self.in_and_out_indexes[0])
            with self.tik_instance.else_scope():
                self.in_and_out_indexes[4].set_as(self.in_and_out_indexes[0] - self.padding_00)
        with self.tik_instance.if_scope(self.input_dim_1 == self.output_dim_1):
            self.in_and_out_indexes[5].set_as(self.in_and_out_indexes[1])
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(self.in_and_out_indexes[1] < self.padding_10):
                self.in_and_out_indexes[5].set_as(self.dividends[10] - self.in_and_out_indexes[1])
            with self.tik_instance.elif_scope(self.in_and_out_indexes[1] >= self.dividends[6]):
                self.in_and_out_indexes[5].set_as(self.dividends[14] - self.in_and_out_indexes[1])
            with self.tik_instance.else_scope():
                self.in_and_out_indexes[5].set_as(self.in_and_out_indexes[1] - self.padding_10)
        with self.tik_instance.if_scope(self.input_dim_2 == self.output_dim_2):
            self.in_and_out_indexes[6].set_as(self.in_and_out_indexes[2])
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(self.in_and_out_indexes[2] < self.padding_20):
                self.in_and_out_indexes[6].set_as(0)
            with self.tik_instance.elif_scope(self.in_and_out_indexes[2] >= self.dividends[7]):
                self.in_and_out_indexes[6].set_as(self.tiling_input_shape[2] - 1)
            with self.tik_instance.else_scope():
                self.in_and_out_indexes[6].set_as(self.in_and_out_indexes[2] - self.padding_20)
        if axis_nums == Constant.AXIS_DIMS_4:
            with self.tik_instance.if_scope(self.input_dim_3 == self.output_dim_3):
                self.in_and_out_indexes[7].set_as(self.in_and_out_indexes[3])
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(self.in_and_out_indexes[3] < self.padding_30):
                    self.in_and_out_indexes[7].set_as(0)
                with self.tik_instance.elif_scope(self.in_and_out_indexes[3] >= self.dividends[8]):
                    self.in_and_out_indexes[7].set_as(self.tiling_input_shape[3] - 1)
                with self.tik_instance.else_scope():
                    self.in_and_out_indexes[7].set_as(self.in_and_out_indexes[3] - self.padding_30)
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
        self.dividends[5].set_as(self.padding_00 + self.tiling_input_shape[0])
        self.dividends[6].set_as(self.padding_10 + self.tiling_input_shape[1])
        self.dividends[7].set_as(self.padding_20 + self.tiling_input_shape[2])
        self.dividends[9].set_as(self.padding_00)
        self.dividends[10].set_as(self.padding_10)
        self.dividends[11].set_as(self.padding_20)
        self.dividends[13].set_as(2 * self.tiling_input_shape[0] + self.padding_00 - 2)
        self.dividends[14].set_as(2 * self.tiling_input_shape[1] + self.padding_10 - 2)
        self.dividends[15].set_as(2 * self.tiling_input_shape[2] + self.padding_20 - 2)
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
            self.dividends[8].set_as(self.padding_30 + self.tiling_input_shape[3])
            self.dividends[12].set_as(self.padding_30)
            self.dividends[16].set_as(2 * self.tiling_input_shape[3] + self.padding_30 - 2)
        else:
            pass

    def search_src_offset_and_loop_num(self):
        src_offset = self.tik_instance.Scalar(self.tiling_dtype, "src_offset")
        temp_scalar = self.tik_instance.Scalar(self.tiling_dtype, "temp_scalar")
        temp_scalar.set_as(self.dim2_outer_move_out_offset % (self.output_dim_2))
        src_dim2 = self.tik_instance.Scalar(self.tiling_dtype, "src_dim2")
        outer_loop_num = self.tik_instance.Scalar(self.tiling_dtype, "outer_loop_num")
        with self.tik_instance.if_scope(self.input_dim_2 == 1):
            src_dim2.set_as(0)
            outer_loop_num.set_as(self.output_dim_2 - temp_scalar)
        with self.tik_instance.elif_scope(temp_scalar <= self.padding_20):
            src_dim2.set_as(0)
            outer_loop_num.set_as(self.padding_20 - temp_scalar + 1)
        with self.tik_instance.elif_scope(temp_scalar < (self.padding_20 + self.input_dim_2 - 1)):
            src_dim2.set_as(temp_scalar - self.padding_20)
            outer_loop_num.set_as(1)
        with self.tik_instance.elif_scope(temp_scalar >= (self.padding_20 + self.input_dim_2 - 1)):
            src_dim2.set_as(self.input_dim_2 - 1)
            outer_loop_num.set_as(self.output_dim_2 - temp_scalar)
        src_offset.set_as(self.dim2_outer_move_out_offset // self.output_dim_2 * self.input_dim_2 + src_dim2)
        rest_outer_num = self.tik_instance.Scalar(self.tiling_dtype, "rest_outer_num")
        rest_outer_num.set_as(self.dim2_rest_outer_num - outer_loop_num)
        with self.tik_instance.if_scope(rest_outer_num < 0):
            outer_loop_num.set_as(self.dim2_rest_outer_num)
            rest_outer_num.set_as(0)
        return src_offset, outer_loop_num, rest_outer_num
    
        
    def data_copy_in_and_process_ub(self, src_offset_dim_3):
        self.get_vnchw_stride(self.inner_col_num_non_last_row, False)
        with self.tik_instance.new_stmt_scope(True):
            burst = self.inner_col_num_non_last_row * self.input_dim_4 * self.inner_bytes_size
            with self.tik_instance.for_range(0, self.inner_row_num - 1) as inner_row_index:
                self.tik_instance.data_move_pad(self.ub_first[inner_row_index * self.input_ceil_algin],
                                                self.input_gm[
                                                    (src_offset_dim_3 + \
                                                    inner_row_index * self.inner_col_num_non_last_row) * \
                                                    self.input_dim_4],
                                                nburst=1, burst=burst, dst_gap=0, src_gap=0,
                                                right_padding=0, left_padding=0, padding_value=None)
            burst_last = self.inner_col_num_last_row * self.input_dim_4 * self.inner_bytes_size
            self.tik_instance.data_move_pad(self.ub_first[(self.inner_row_num - 1) * self.input_ceil_algin],
                                            self.input_gm[
                                                (src_offset_dim_3 + \
                                                (self.inner_row_num - 1) * self.inner_col_num_non_last_row) * \
                                                self.input_dim_4],
                                            nburst=1, burst=burst_last, dst_gap=0, src_gap=0,
                                            right_padding=0, left_padding=0, padding_value=None)
        dst_list_0 = [self.ub_second[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
        src_list_0 = [self.ub_first[self.input_ceil_algin * i] for i in range(Constant.TRANS_MIN_BLKS)]
        self.tik_instance.vnchwconv(False, False, dst_list_0, src_list_0, 
                                    self.process_blk_in, self.conv1_stride_0, self.conv1_stride_1)
        self.do_data_move_for_padding_dim_4(self.ub_second, self.ub_first, self.inner_col_num_non_last_row)
        dst_list_1 = [self.ub_second[self.output_ceil_algin * i] for i in range(Constant.TRANS_MIN_BLKS)]
        src_list_1 = [self.ub_first[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
        self.tik_instance.vnchwconv(False, False, dst_list_1, src_list_1, 
                                    self.process_blk_out, self.conv2_stride_0, self.conv2_stride_1)
        
    def padding_dim_30(self, dst_offset_dim3):
        with self.tik_instance.new_stmt_scope(True):
            burst = self.output_dim_4 * self.inner_bytes_size
            with self.tik_instance.for_range(0, self.padding_30) as inner_row_index:
                self.tik_instance.data_move_pad(self.output_gm[
                                                    (dst_offset_dim3 + inner_row_index) * self.output_dim_4],
                                                self.ub_second,
                                                nburst=1, burst=burst, dst_gap=0, src_gap=0,
                                                right_padding=0, left_padding=0, padding_value=None)
            
    def padding_dim_31(self, dst_offset_dim3):
        with self.tik_instance.new_stmt_scope(True):
            burst = self.output_dim_4 * self.inner_bytes_size
            with self.tik_instance.for_range(0, self.padding_31) as inner_row_index:
                self.tik_instance.data_move_pad(self.output_gm[
                                                    (dst_offset_dim3 + inner_row_index) * self.output_dim_4],
                                                self.ub_second[self.output_ceil_algin * (self.inner_row_num - 1)],
                                                nburst=1, burst=burst, dst_gap=0, src_gap=0,
                                                right_padding=0, left_padding=0, padding_value=None)
            
    def fill_dim_3(self, dst_offset_dim3):
        with self.tik_instance.new_stmt_scope(True):
            burst = self.inner_col_num_non_last_row * self.output_dim_4 * self.inner_bytes_size
            with self.tik_instance.for_range(0, self.inner_row_num - 1) as inner_row_index:
                self.tik_instance.data_move_pad(self.output_gm[
                                                    (dst_offset_dim3 + inner_row_index * \
                                                    self.inner_col_num_non_last_row) * self.output_dim_4],
                                                self.ub_second[inner_row_index * self.output_ceil_algin],
                                                nburst=1, burst=burst, dst_gap=0, src_gap=0,
                                                right_padding=0, left_padding=0, padding_value=None)
            burst_last = self.inner_col_num_last_row * self.output_dim_4 * self.inner_bytes_size
            self.tik_instance.data_move_pad(self.output_gm[
                                                (dst_offset_dim3 + \
                                                (self.inner_row_num - 1) * self.inner_col_num_non_last_row) * \
                                                self.output_dim_4],
                                            self.ub_second[(self.inner_row_num - 1) * self.output_ceil_algin],
                                            nburst=1, burst=burst_last, dst_gap=0, src_gap=0,
                                            right_padding=0, left_padding=0, padding_value=None)
        
    def process_multi_copy_in(self, src_offset_dim2, dst_offset_dim2):
        with self.tik_instance.for_range(0, self.inner_loop_times) as in_loop_index:
            src_offset_dim3 = self.tik_instance.Scalar(self.tiling_dtype, "src_offset_dim3")
            src_offset_dim3.set_as(src_offset_dim2 * self.input_dim_3 + \
                                   (self.one_loop_process_lines * Constant.TRANS_MIN_BLKS) * in_loop_index)
            with self.tik_instance.if_scope(in_loop_index < self.inner_loop_times - 1):
                self.inner_col_num_non_last_row.set_as(self.one_loop_process_lines)
                self.inner_row_num.set_as(Constant.TRANS_MIN_BLKS)
                self.inner_col_num_last_row.set_as(self.one_loop_process_lines)
            with self.tik_instance.else_scope():
                self.inner_col_num_non_last_row.set_as((self.inner_process_nums_tail_loop + \
                                            Constant.TRANS_MIN_BLKS - 1) // Constant.TRANS_MIN_BLKS)
                self.inner_row_num.set_as((self.inner_process_nums_tail_loop + \
                                            self.inner_col_num_non_last_row - 1) // self.inner_col_num_non_last_row)
                self.inner_col_num_last_row.set_as(self.inner_process_nums_tail_loop - \
                                            self.inner_col_num_non_last_row * (self.inner_row_num - 1))
            self.process_blk_in.set_as((self.inner_col_num_non_last_row * self.input_dim_4 + \
                                        self.block_num - 1) // self.block_num)
            self.input_ceil_algin.set_as(self.process_blk_in * self.block_num)
            self.process_blk_out.set_as((self.inner_col_num_non_last_row * self.output_dim_4 + \
                                         self.block_num - 1) // self.block_num)
            self.output_ceil_algin.set_as(self.process_blk_out * self.block_num)
            self.data_copy_in_and_process_ub(src_offset_dim3)
            with self.tik_instance.for_range(0, self.dim2_outer_loop_num) as out_loop_index:
                dst_offset_dim3 = self.tik_instance.Scalar(self.tiling_dtype, "dst_offset_dim3")
                dst_offset_dim3.set_as((dst_offset_dim2 + out_loop_index) * self.output_dim_3)
                self.fill_dim_3(dst_offset_dim3 + self.padding_30 + in_loop_index * \
                                (self.one_loop_process_lines * Constant.TRANS_MIN_BLKS))
                with self.tik_instance.if_scope(tik.all(self.padding_30 != 0, in_loop_index == 0)):
                    self.padding_dim_30(dst_offset_dim3)
            
            with self.tik_instance.if_scope(tik.all(self.padding_31 != 0, in_loop_index == self.inner_loop_times - 1)):
                dst_list = [self.ub_second[self.output_ceil_algin * i] for i in range(Constant.TRANS_MIN_BLKS)]
                src_list = [self.ub_first[Constant.TRANS_MIN_BLKS * \
                            ((self.inner_col_num_last_row - 1) * self.output_dim_4 + i)]
                            for i in range(Constant.TRANS_MIN_BLKS)]
                self.tik_instance.vnchwconv(False, False, dst_list, src_list,
                                            (self.output_dim_4 + self.block_num - 1) // self.block_num, 1, 16)
                with self.tik_instance.for_range(0, self.dim2_outer_loop_num) as out_loop_index:
                    dst_offset_dim3 = self.tik_instance.Scalar(self.tiling_dtype, "dst_offset_dim3")
                    dst_offset_dim3.set_as((dst_offset_dim2 + out_loop_index) * self.output_dim_3)
                    self.padding_dim_31(dst_offset_dim3 + self.padding_30 + self.input_dim_3)
        
    def process_once_copy_in(self, src_offset_dim2, dst_offset_dim2):
        src_offset_dim3 = self.tik_instance.Scalar(self.tiling_dtype, "src_offset_dim3")
        src_offset_dim3.set_as(src_offset_dim2 * self.input_dim_3)
        self.inner_col_num_non_last_row.set_as((self.inner_process_nums_tail_loop + \
                                           Constant.TRANS_MIN_BLKS - 1) // Constant.TRANS_MIN_BLKS)
        self.inner_row_num.set_as((self.inner_process_nums_tail_loop + \
                                           self.inner_col_num_non_last_row - 1) // self.inner_col_num_non_last_row)
        self.inner_col_num_last_row.set_as(self.inner_process_nums_tail_loop - \
                                           self.inner_col_num_non_last_row * (self.inner_row_num - 1))
        self.process_blk_in.set_as((self.inner_col_num_non_last_row * self.input_dim_4 + \
                                    self.block_num - 1) // self.block_num)
        self.input_ceil_algin.set_as(self.process_blk_in * self.block_num)
        self.process_blk_out.set_as((self.inner_col_num_non_last_row * self.output_dim_4 + \
                                     self.block_num - 1) // self.block_num)
        self.output_ceil_algin.set_as(self.process_blk_out * self.block_num)
        self.data_copy_in_and_process_ub(src_offset_dim3)
        with self.tik_instance.for_range(0, self.dim2_outer_loop_num) as out_loop_index:
            dst_offset_dim3 = self.tik_instance.Scalar(self.tiling_dtype, "dst_offset_dim3")
            dst_offset_dim3.set_as((dst_offset_dim2 + out_loop_index) * self.output_dim_3)
            with self.tik_instance.if_scope(self.padding_30 != 0):
                self.padding_dim_30(dst_offset_dim3)
            self.fill_dim_3(dst_offset_dim3 + self.padding_30)
        with self.tik_instance.if_scope(self.padding_31 != 0):
            dst_list = [self.ub_second[self.output_ceil_algin * i] for i in range(Constant.TRANS_MIN_BLKS)]
            src_list = [self.ub_first[Constant.TRANS_MIN_BLKS * \
                        ((self.inner_col_num_last_row - 1) * self.output_dim_4 + i)] 
                        for i in range(Constant.TRANS_MIN_BLKS)]
            self.tik_instance.vnchwconv(False, False, dst_list, src_list,
                                        (self.output_dim_4 + self.block_num - 1) // self.block_num, 1, 16)
            with self.tik_instance.for_range(0, self.dim2_outer_loop_num) as out_loop_index:
                dst_offset_dim3 = self.tik_instance.Scalar(self.tiling_dtype, "dst_offset_dim3")
                dst_offset_dim3.set_as((dst_offset_dim2 + out_loop_index) * self.output_dim_3)
                self.padding_dim_31(dst_offset_dim3 + self.padding_30 + self.input_dim_3)
                
    def cal_dim2_loop_num(self):
        start_idx = self.tik_instance.Scalar(self.tiling_dtype, "start_idx")
        start_idx.set_as(self.outer_move_out_core_offset % self.output_dim_2)
        end_idx = self.tik_instance.Scalar(self.tiling_dtype, "end_idx")
        end_idx.set_as((self.outer_move_out_core_offset + self.outer_process_num - 1) % self.output_dim_2)
        start_rest_loop = self.tik_instance.Scalar(self.tiling_dtype, "start_rest_loop")
        end_rest_loop = self.tik_instance.Scalar(self.tiling_dtype, "end_rest_loop")
        with self.tik_instance.if_scope(start_idx <= self.padding_20):
            start_rest_loop.set_as(0)
        with self.tik_instance.elif_scope(start_idx < (self.padding_20 + self.input_dim_2 - 1)):
            start_rest_loop.set_as(start_idx - self.padding_20)
        with self.tik_instance.elif_scope(start_idx >= (self.padding_20 + self.input_dim_2 - 1)):
            start_rest_loop.set_as(self.input_dim_2 - 1)
            
        with self.tik_instance.if_scope(end_idx <= self.padding_20):
            end_rest_loop.set_as(self.input_dim_2 - 1)
        with self.tik_instance.elif_scope(end_idx <= (self.padding_20 + self.input_dim_2 - 1)):
            end_rest_loop.set_as(self.input_dim_2 + self.padding_20 - end_idx - 1)
        with self.tik_instance.elif_scope(end_idx > (self.padding_20 + self.input_dim_2 - 1)):
            end_rest_loop.set_as(0)
        self.loop_num.set_as(((self.outer_process_num + start_idx + \
                               (self.output_dim_2 - end_idx - 1)) // self.output_dim_2) * \
                                self.input_dim_2 - start_rest_loop - end_rest_loop)

    def do_tiling_key_mode_1(self, core_id):
        """
        divide by outer:[0,1,2] inner:[3,4]
        """
        if tbe_platform.api_check_support("tik.data_move_pad"):
            self.inner_process_prepare(Constant.MODE1)
            self.one_loop_process_lines = self.tik_instance.Scalar(
                self.tiling_dtype, "one_loop_process_lines",
                init_value=self.output_dim4_max_cnt_block_align // self.output_dim_4_align_blocks)
            self.inner_loop_times = self.tik_instance.Scalar(
                self.tiling_dtype, "inner_loop_times",
                init_value=(self.input_dim_3 + self.one_loop_process_lines * Constant.TRANS_MIN_BLKS - 1) \
                    // (self.one_loop_process_lines * Constant.TRANS_MIN_BLKS))
            self.inner_process_nums_tail_loop = self.tik_instance.Scalar(
                self.tiling_dtype, "inner_process_nums_tail_loop",
                init_value=self.input_dim_3 - \
                    (self.one_loop_process_lines * Constant.TRANS_MIN_BLKS) * (self.inner_loop_times - 1))
            self.outer_move_out_core_offset.set_as(self.num_per_core * core_id)
            self.outer_process_num = self.tik_instance.Scalar(self.tiling_dtype, "outer_process_num")
            with self.tik_instance.if_scope(core_id < self.core_used_num - 1):
                self.outer_process_num.set_as(self.num_per_core)
            with self.tik_instance.else_scope():
                self.outer_process_num.set_as(self.num_tail_core)
            self.dim2_rest_outer_num = self.tik_instance.Scalar(self.tiling_dtype, "dim2_rest_outer_num")
            self.dim2_rest_outer_num.set_as(self.outer_process_num)
            self.dim2_outer_move_out_offset = self.tik_instance.Scalar(self.tiling_dtype, "dim2_outer_move_out_offset")
            self.dim2_outer_move_out_offset.set_as(self.outer_move_out_core_offset)
            self.loop_num = self.tik_instance.Scalar(self.tiling_dtype, "loop_num")
            self.cal_dim2_loop_num()
            with self.tik_instance.for_range(0, self.loop_num):
                self.dim2_src_offset, self.dim2_outer_loop_num, self.dim2_rest_outer_num = \
                    self.search_src_offset_and_loop_num()
                with self.tik_instance.if_scope(self.inner_loop_times == 1):
                    self.process_once_copy_in(self.dim2_src_offset, self.dim2_outer_move_out_offset)
                with self.tik_instance.else_scope():
                    self.process_multi_copy_in(self.dim2_src_offset, self.dim2_outer_move_out_offset)
                self.dim2_outer_move_out_offset.set_as(self.dim2_outer_move_out_offset + self.dim2_outer_loop_num)
        else:
            self.set_dividends_scalars(Constant.AXIS_DIMS_3)
            self.outer_move_out_core_offset.set_as(self.num_per_core * core_id)
            self.address_rollback = self.tik_instance.ScalarArray(self.tiling_dtype, 4, "address_rollback",
                                                                  init_value=0)
            self.address_rollback_sum = self.tik_instance.Scalar(self.tiling_dtype, "address_rollback_sum",
                                                                 init_value=0)
            with self.tik_instance.if_scope(self.padding_31 == 0):
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
        
    def inner_input_dim4_process(self):
        self.get_vnchw_stride(self.inner_col_num_non_last_row, False)
        process_blk_in = (self.inner_col_num_non_last_row * self.input_dim_4 + self.block_num - 1) // self.block_num
        input_ceil_algin = process_blk_in * self.block_num
        process_blk_out = (self.inner_col_num_non_last_row * self.output_dim_4 + self.block_num - 1) // self.block_num
        output_ceil_algin = process_blk_out * self.block_num
        if tbe_platform.api_check_support("tik.data_move_pad"):
            with self.tik_instance.new_stmt_scope(True):
                burst = self.inner_col_num_non_last_row * self.input_dim_4 * self.inner_bytes_size
                with self.tik_instance.for_range(0, self.inner_row_num - 1) as inner_row_index:
                    self.tik_instance.data_move_pad(self.ub_first[inner_row_index * input_ceil_algin],
                                                    self.input_gm[
                                                        (self.outer_move_in_index + self.move_out_start_line + \
                                                        inner_row_index * self.inner_col_num_non_last_row) * \
                                                        self.input_dim_4],
                                                    nburst=1, burst=burst, dst_gap=0, src_gap=0,
                                                    right_padding=0, left_padding=0, padding_value=None)
                burst_last = self.inner_col_num_last_row * self.input_dim_4 * self.inner_bytes_size
                self.tik_instance.data_move_pad(self.ub_first[(self.inner_row_num - 1) * input_ceil_algin],
                                                self.input_gm[
                                                    (self.outer_move_in_index + self.move_out_start_line + \
                                                    (self.inner_row_num - 1) * self.inner_col_num_non_last_row) * \
                                                    self.input_dim_4],
                                                nburst=1, burst=burst_last, dst_gap=0, src_gap=0,
                                                right_padding=0, left_padding=0, padding_value=None)
        dst_list_0 = [self.ub_second[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
        src_list_0 = [self.ub_first[input_ceil_algin * i] for i in range(Constant.TRANS_MIN_BLKS)]
        self.tik_instance.vnchwconv(False, False, dst_list_0, src_list_0, 
                                    process_blk_in, self.conv1_stride_0, self.conv1_stride_1)
        self.do_data_move_for_padding_dim_4(self.ub_second, self.ub_first, self.inner_col_num_non_last_row)
        dst_list_1 = [self.ub_second[output_ceil_algin * i] for i in range(Constant.TRANS_MIN_BLKS)]
        src_list_1 = [self.ub_first[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
        self.tik_instance.vnchwconv(False, False, dst_list_1, src_list_1, 
                                    process_blk_out, self.conv2_stride_0, self.conv2_stride_1)
        if tbe_platform.api_check_support("tik.data_move_pad"):
            with self.tik_instance.new_stmt_scope(True):
                burst = self.inner_col_num_non_last_row * self.output_dim_4 * self.inner_bytes_size
                with self.tik_instance.for_range(0, self.inner_row_num - 1) as inner_row_index:
                    self.tik_instance.data_move_pad(self.output_gm[
                                                        (self.outer_move_in_index + self.move_out_start_line + \
                                                        inner_row_index * self.inner_col_num_non_last_row) * \
                                                        self.output_dim_4],
                                                    self.ub_second[output_ceil_algin * inner_row_index],
                                                    nburst=1, burst=burst, dst_gap=0, src_gap=0,
                                                    right_padding=0, left_padding=0, padding_value=None)
                burst_last = self.inner_col_num_last_row * self.output_dim_4 * self.inner_bytes_size
                self.tik_instance.data_move_pad(self.output_gm[
                                                    (self.outer_move_in_index + self.move_out_start_line + \
                                                    (self.inner_row_num - 1) * self.inner_col_num_non_last_row) * \
                                                    self.output_dim_4],
                                                self.ub_second[(self.inner_row_num - 1) * output_ceil_algin],
                                                nburst=1, burst=burst_last, dst_gap=0, src_gap=0,
                                                right_padding=0, left_padding=0, padding_value=None)

    def last_dim_inner_process_for_mode_4(self):
        self.inner_loop_times.set_as(self.outer_process_num // self.one_loop_process_lines // Constant.TRANS_MIN_BLKS)
        self.inner_process_nums_tail_loop.set_as(self.outer_process_num - self.inner_loop_times * \
                                                 self.one_loop_process_lines * Constant.TRANS_MIN_BLKS)
        with self.tik_instance.if_scope(self.inner_process_nums_tail_loop > 0):
            self.move_out_start_line.set_as(self.inner_loop_times * \
                                            self.one_loop_process_lines * Constant.TRANS_MIN_BLKS)
            self.inner_col_num_non_last_row.set_as((self.inner_process_nums_tail_loop + \
                                                    Constant.TRANS_MIN_BLKS - 1) // Constant.TRANS_MIN_BLKS)
            self.inner_row_num.set_as((self.inner_process_nums_tail_loop + \
                                       self.inner_col_num_non_last_row - 1) // self.inner_col_num_non_last_row)
            self.inner_col_num_last_row.set_as(self.inner_process_nums_tail_loop - \
                                               self.inner_col_num_non_last_row * (self.inner_row_num - 1))
            self.inner_input_dim4_process()
        with self.tik_instance.if_scope(self.inner_loop_times > 0):
            with self.tik_instance.for_range(0, self.inner_loop_times) as inner_loop_index:
                self.inner_row_num.set_as(Constant.TRANS_MIN_BLKS)
                self.inner_col_num_last_row.set_as(self.one_loop_process_lines)
                self.inner_col_num_non_last_row.set_as(self.one_loop_process_lines)
                self.move_out_start_line.set_as(inner_loop_index * \
                                                self.one_loop_process_lines * Constant.TRANS_MIN_BLKS)
                self.inner_input_dim4_process()

    def do_tiling_key_mode_4(self, core_id):
        """
        padding_00 == 0 & padding_01 == 0 && padding_10 == 0 & padding_11 == 0 &&
        padding_20 == 0 & padding_21 == 0 && padding_30 == 0 & padding_31 == 0
        divide by outer:[0,1,2,3] inner:[4]
        """
        self.inner_process_prepare(Constant.MODE4)
        self.outer_process_num = self.tik_instance.Scalar(self.tiling_dtype, "outer_process_num", init_value=0)
        with self.tik_instance.if_scope(core_id < self.core_used_num - 1):
            self.outer_process_num.set_as(self.num_per_core)
        with self.tik_instance.else_scope():
            self.outer_process_num.set_as(self.num_tail_core)
        self.outer_move_out_core_offset.set_as(self.num_per_core * core_id)
        self.outer_move_in_index.set_as(self.outer_move_out_core_offset)
        self.last_dim_inner_process_for_mode_4()

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
                with self.tik_instance.if_scope(self.padding_41 != Constant.TRANS_MIN_BLKS):
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

    def support_move_pad_padding_front(self):
        # 1.搬入pad值
        ub_first = self.ub_first
        self.tik_instance.data_move_pad(ub_first,
                                        self.input_gm[self.outer_move_in_index * self.input_dim_4],
                                        nburst=1, burst=self.dtype_rate * self.inner_bytes_size,
                                        dst_gap=0, src_gap=0, right_padding=0, left_padding=0, padding_value=None)
        # 2.计算要循环搬出的次数
        max_once_process_num = self.output_dim4_max_cnt_block_align * Constant.VEC_LENGTH // Constant.TWO_COPIES
        loop_num = (self.padding_40 + max_once_process_num - 1) // max_once_process_num
        repeat_time = self.tik_instance.Scalar(self.tiling_dtype, "repeat_time")
        tail_process_num = self.tik_instance.Scalar(self.tiling_dtype, "tail_process_num")
        with self.tik_instance.if_scope(max_once_process_num > self.padding_40):
            tail_process_num.set_as(self.padding_40)
            repeat_time.set_as((self.padding_40 + Constant.FP16_MASK - 1) // Constant.FP16_MASK)
        with self.tik_instance.else_scope():
            tail_process_num.set_as(self.padding_40 % max_once_process_num)
            repeat_time.set_as((max_once_process_num + Constant.FP16_MASK - 1) // Constant.FP16_MASK)
        
        # 3.如果是4字节数据特殊处理
        if self.dtype_rate == Constant.DTYPE_RATE_TWO:
            mask = Constant.FP32_MASK
            dst_rep_stride = Constant.COMMON_STRIDE
            ub_first = ub_first.reinterpret_cast_to("uint32")
            pad_scalar_left = self.tik_instance.Scalar("uint32", "pad_scalar_left")
            pad_scalar_left.set_as(ub_first[0])
            self.tik_instance.vec_dup(mask, ub_first, pad_scalar_left, repeat_time, dst_rep_stride)
            ub_first = ub_first.reinterpret_cast_to(self.inner_dtype)
        elif self.dtype_rate == Constant.DTYPE_RATE_ONE:
            mask = Constant.FP16_MASK
            dst_rep_stride = Constant.COMMON_STRIDE
            pad_scalar_left = self.tik_instance.Scalar(self.inner_dtype, "pad_scalar_left")
            pad_scalar_left.set_as(ub_first[0])
            self.tik_instance.vec_dup(mask, ub_first, pad_scalar_left, repeat_time, dst_rep_stride)
        # 4.搬出到GM
        with self.tik_instance.for_range(0, loop_num - 1) as inner_loop_index:
            self.tik_instance.data_move_pad(self.output_gm[self.outer_move_out_index * self.output_dim_4 + 
                                                        inner_loop_index * max_once_process_num],
                                            ub_first,
                                            nburst=1, burst=max_once_process_num * self.inner_bytes_size, 
                                            dst_gap=0, src_gap=0,
                                            right_padding=0, left_padding=0, padding_value=None)
        self.tik_instance.data_move_pad(self.output_gm[self.outer_move_out_index * self.output_dim_4 +
                                                    (loop_num - 1) * max_once_process_num],
                                        ub_first,
                                        nburst=1, burst=tail_process_num * self.inner_bytes_size,
                                        dst_gap=0, src_gap=0, right_padding=0, left_padding=0, padding_value=None)

    def not_support_move_pad_padding_front(self):
        one_loop_process_blocks = self.output_dim4_max_cnt_block_align
        self.inner_loop_times.set_as(
            (self.padding40_align_blocks + one_loop_process_blocks - 1) / one_loop_process_blocks)
        self.inner_process_nums_tail_loop.set_as(self.padding40_align_blocks -
                                                (self.inner_loop_times - 1) * one_loop_process_blocks)
        self.tail_loop_true_nums.set_as(self.padding_40 - (self.inner_loop_times - 1) * one_loop_process_blocks *
                                        Constant.TRANS_MIN_BLKS)
        self.move_in_start_blocks.set_as((self.inner_loop_times - 1) * one_loop_process_blocks)
        self.get_vnchw_stride(self.inner_process_nums_tail_loop, True)
        self.tik_instance.data_move(
            self.ub_first, self.input_gm[self.outer_move_in_index * self.input_dim_4], 0, 1,
                                        self.inner_process_nums_tail_loop, 0, 0)
        dst_list_0 = [self.ub_second[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
        src_list_0 = [self.ub_first] * Constant.TRANS_MIN_BLKS
        self.tik_instance.vnchwconv(False, False, dst_list_0, src_list_0, self.inner_process_nums_tail_loop,
                                    self.conv1_stride_0, self.conv1_stride_1)
        with self.tik_instance.for_range(0, self.tail_loop_true_nums / self.dtype_rate) as padding_40_index:
            self.tik_instance.data_move(
                self.ub_first[padding_40_index * self.dtype_rate * Constant.TRANS_MIN_BLKS],
                self.ub_second[0], 0, 1, self.dtype_rate, 0, 0)
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
                self.ub_first, self.input_gm[self.outer_move_in_index * self.input_dim_4], 0, 1,
                one_loop_process_blocks, 0, 0)
            dst_list_0 = [self.ub_second[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
            src_list_0 = [self.ub_first] * Constant.TRANS_MIN_BLKS
            self.tik_instance.vnchwconv(False, False, dst_list_0, src_list_0, one_loop_process_blocks,
                                        self.conv1_stride_0, self.conv1_stride_1)
            with self.tik_instance.for_range(0, one_loop_process_blocks * Constant.TRANS_MIN_BLKS /
                                            self.dtype_rate) as padding_40_index:
                self.tik_instance.data_move(
                    self.ub_first[padding_40_index * self.dtype_rate * Constant.TRANS_MIN_BLKS],
                    self.ub_second[0], 0, 1, self.dtype_rate, 0, 0)
            dst_list_1 = [self.ub_second[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
            src_list_1 = [self.ub_first[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
            self.tik_instance.vnchwconv(False, False, dst_list_1, src_list_1, one_loop_process_blocks,
                                        self.conv2_stride_0, self.conv2_stride_1)
            self.tik_instance.data_move(
                self.output_gm[self.outer_move_out_index * self.output_dim_4 + self.tail_loop_true_nums +
                            inner_loop_index * one_loop_process_blocks * Constant.TRANS_MIN_BLKS],
                self.ub_second, 0, 1, one_loop_process_blocks, 0, 0)

    def process_padding_front(self):
        if tbe_platform.api_check_support("tik.data_move_pad"):
            self.support_move_pad_padding_front()
        else:
            self.not_support_move_pad_padding_front()

    def support_move_pad_middle(self):
        # 1.计算要循环搬出的次数
        max_once_process_num = self.output_dim4_max_cnt_block_align * Constant.VEC_LENGTH
        loop_num = (self.input_dim_4 + max_once_process_num - 1) // max_once_process_num
        tail_process_num = self.tik_instance.Scalar(self.tiling_dtype, "tail_process_num")
        with self.tik_instance.if_scope(max_once_process_num > self.input_dim_4):
            tail_process_num.set_as(self.input_dim_4)
        with self.tik_instance.else_scope():
            tail_process_num.set_as(self.input_dim_4 % max_once_process_num)
    
        # 2.搬搬入搬出到GM
        with self.tik_instance.for_range(0, loop_num - 1) as inner_loop_index:
            self.tik_instance.data_move_pad(self.ub_second,
                                        self.input_gm[self.outer_move_in_index * self.input_dim_4 +
                                                    inner_loop_index * max_once_process_num],
                                        nburst=1, burst=max_once_process_num * self.inner_bytes_size,
                                        dst_gap=0, src_gap=0, right_padding=0, left_padding=0, padding_value=None)
            self.tik_instance.data_move_pad(self.output_gm[self.outer_move_out_index * self.output_dim_4 + 
                                                self.padding_40 + inner_loop_index * max_once_process_num],
                                            self.ub_second,
                                            nburst=1, burst=max_once_process_num * self.inner_bytes_size, 
                                            dst_gap=0, src_gap=0,
                                            right_padding=0, left_padding=0, padding_value=None)
        self.tik_instance.data_move_pad(self.ub_second,
                                        self.input_gm[(self.outer_move_in_index + 1) * self.input_dim_4
                                                    - tail_process_num],
                                        nburst=1, burst=tail_process_num * self.inner_bytes_size,
                                        dst_gap=0, src_gap=0, right_padding=0, left_padding=0, padding_value=None)
        self.tik_instance.data_move_pad(self.output_gm[(self.outer_move_out_index + 1) * self.output_dim_4
                                                    - tail_process_num - self.padding_41],
                                        self.ub_second,
                                        nburst=1, burst=tail_process_num * self.inner_bytes_size,
                                        dst_gap=0, src_gap=0, right_padding=0, left_padding=0, padding_value=None)
    
    def not_support_move_pad_middle(self):
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
                    self.output_gm[self.outer_move_out_index * self.output_dim_4 + self.padding_40 +
                                self.move_in_start_blocks * Constant.TRANS_MIN_BLKS], self.ub_first, 0, 1,
                    one_loop_process_blocks, 0, 0)
            self.tik_instance.data_move(
                self.ub_first, self.input_gm[self.outer_move_in_index * self.input_dim_4 + self.input_dim_4 -
                                            self.inner_process_nums_tail_loop * Constant.TRANS_MIN_BLKS], 0, 1,
                self.inner_process_nums_tail_loop, 0, 0)
            self.tik_instance.data_move(
                self.output_gm[self.outer_move_out_index * self.output_dim_4 + self.padding_40 + self.input_dim_4 -
                            self.inner_process_nums_tail_loop * Constant.TRANS_MIN_BLKS], self.ub_first, 0, 1,
                self.inner_process_nums_tail_loop, 0, 0)
        with self.tik_instance.if_scope(self.inner_loop_times == 1):
            self.tik_instance.data_move(self.ub_first, self.input_gm[self.outer_move_in_index * self.input_dim_4],
                                        0, 1, self.inner_process_nums_tail_loop - 1, 0, 0)
            self.tik_instance.data_move(self.output_gm[self.outer_move_out_index * self.output_dim_4 +
                                        self.padding_40],
                                        self.ub_first, 0, 1, self.inner_process_nums_tail_loop - 1, 0, 0)
            self.tik_instance.data_move(self.ub_first,
                                        self.input_gm[self.outer_move_in_index * self.input_dim_4 +
                                                        self.input_dim_4 - Constant.TRANS_MIN_BLKS],
                                        0, 1, 1, 0, 0)
            self.tik_instance.data_move(self.output_gm[self.outer_move_out_index * self.output_dim_4 +
                                            self.padding_40 + self.input_dim_4 - Constant.TRANS_MIN_BLKS],
                                        self.ub_first, 0, 1, 1, 0, 0)

    def process_middle(self):
        if tbe_platform.api_check_support("tik.data_move_pad"):
            self.support_move_pad_middle()
        else:
            self.not_support_move_pad_middle()

    def support_move_pad_padding_back(self):
        # 1.搬入pad值
        ub_tri = self.ub_first[self.output_dim4_max_cnt_block_align * Constant.VEC_LENGTH // Constant.TWO_COPIES]
        self.tik_instance.data_move_pad(ub_tri,
                                        self.input_gm[(self.outer_move_in_index + 1) * self.input_dim_4 - 
                                                    self.dtype_rate],
                                        nburst=1, burst=self.dtype_rate * self.inner_bytes_size,
                                        dst_gap=0, src_gap=0, right_padding=0, left_padding=0, padding_value=None)
        # 2.计算要循环搬出的次数
        max_once_process_num = self.output_dim4_max_cnt_block_align * Constant.VEC_LENGTH // Constant.TWO_COPIES
        loop_num = (self.padding_41 + max_once_process_num - 1) // max_once_process_num
        repeat_time = self.tik_instance.Scalar(self.tiling_dtype, "repeat_time")
        tail_process_num = self.tik_instance.Scalar(self.tiling_dtype, "tail_process_num")
        with self.tik_instance.if_scope(max_once_process_num > self.padding_41):
            tail_process_num.set_as(self.padding_41)
            repeat_time.set_as((self.padding_41 + Constant.FP16_MASK - 1) // Constant.FP16_MASK)
        with self.tik_instance.else_scope():
            tail_process_num.set_as(self.padding_41 % max_once_process_num)
            repeat_time.set_as((max_once_process_num + Constant.FP16_MASK - 1) // Constant.FP16_MASK)
        # 3.如果是4字节数据特殊处理
        if self.dtype_rate == Constant.DTYPE_RATE_TWO:
            mask = Constant.FP32_MASK
            dst_rep_stride = Constant.COMMON_STRIDE
            # 无法直接使用 ub_tri.reinterpret_cast_to("uint32")
            self.ub_first = self.ub_first.reinterpret_cast_to("uint32")
            ub_tri = self.ub_first[self.output_dim4_max_cnt_block_align * Constant.VEC_LENGTH // 4]
            pad_scalar_right = self.tik_instance.Scalar("uint32", "pad_scalar_right")
            pad_scalar_right.set_as(ub_tri[0])
            self.tik_instance.vec_dup(mask, ub_tri, pad_scalar_right, repeat_time, dst_rep_stride)
            self.ub_first = self.ub_first.reinterpret_cast_to(self.inner_dtype)
            ub_tri = ub_tri.reinterpret_cast_to(self.inner_dtype)
        elif self.dtype_rate == Constant.DTYPE_RATE_ONE:
            mask = Constant.FP16_MASK
            dst_rep_stride = Constant.COMMON_STRIDE
            pad_scalar_right = self.tik_instance.Scalar(self.inner_dtype, "pad_scalar_right")
            pad_scalar_right.set_as(ub_tri[0])
            self.tik_instance.vec_dup(mask, ub_tri, pad_scalar_right, repeat_time, dst_rep_stride)
        # 4.搬出到GM
        with self.tik_instance.for_range(0, loop_num - 1) as inner_loop_index:
            self.tik_instance.data_move_pad(self.output_gm[(self.outer_move_out_index + 1) * self.output_dim_4 - 
                                                        self.padding_41 + inner_loop_index * max_once_process_num],
                                            ub_tri,
                                            nburst=1, burst=max_once_process_num * self.inner_bytes_size, 
                                            dst_gap=0, src_gap=0,
                                            right_padding=0, left_padding=0, padding_value=None)
        self.tik_instance.data_move_pad(self.output_gm[(self.outer_move_out_index + 1) * self.output_dim_4
                                                    - tail_process_num],
                                        ub_tri,
                                        nburst=1, burst=tail_process_num * self.inner_bytes_size, dst_gap=0, src_gap=0,
                                        right_padding=0, left_padding=0, padding_value=None)

    def not_support_move_pad_padding_back(self):
        one_loop_process_blocks = self.output_dim4_max_cnt_block_align
        self.address_rollback_sum.set_as(self.address_rollback[0] + self.address_rollback[1])
        with self.tik_instance.if_scope((self.padding_41 < 16) & (self.address_rollback_sum == 2)):
            self.tik_instance.data_move(
                self.ub_addr_rollback,
                self.input_gm[self.outer_move_in_index * self.input_dim_4 +
                                self.input_dim_4 + self.padding_41 - Constant.TRANS_MIN_BLKS], 0, 1, 1, 0, 0)
            self.tik_instance.data_move(
                self.ub_first,
                self.input_gm[self.outer_move_in_index * self.input_dim_4 +
                                self.input_dim_4 - Constant.TRANS_MIN_BLKS], 0, 1, 1, 0, 0)
            with self.tik_instance.for_range(0, self.padding_41 / self.dtype_rate) as index_out:
                with self.tik_instance.for_range(0, self.dtype_rate) as index_in:
                    self.ub_addr_rollback[index_out * self.dtype_rate + Constant.TRANS_MIN_BLKS -
                                            self.padding_41 + index_in].set_as(
                                            self.ub_first[Constant.TRANS_MIN_BLKS - self.dtype_rate + index_in])
            self.tik_instance.data_move(
                self.output_gm[(self.outer_move_out_index + 1) * self.output_dim_4 - Constant.TRANS_MIN_BLKS],
                self.ub_addr_rollback, 0, 1, 1, 0, 0)
        with self.tik_instance.else_scope():
            self.inner_loop_times.set_as(
                (self.padding41_align_blocks + one_loop_process_blocks - 1) / one_loop_process_blocks)
            self.inner_process_nums_tail_loop.set_as(self.padding41_align_blocks -
                                                    (self.inner_loop_times - 1) * one_loop_process_blocks)
            self.tail_loop_true_nums.set_as(self.padding_41 - (self.inner_loop_times - 1) *
                                            one_loop_process_blocks * Constant.TRANS_MIN_BLKS)
            self.tik_instance.data_move(
                self.ub_first, self.input_gm[(self.outer_move_in_index + 1) * self.input_dim_4 -
                                            self.tail_loop_true_nums],
                0, 1, self.inner_process_nums_tail_loop, 0, 0)
            dst_list_0 = [self.ub_second[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
            src_list_0 = [self.ub_first] * Constant.TRANS_MIN_BLKS
            self.get_vnchw_stride(self.inner_process_nums_tail_loop, True)
            self.tik_instance.vnchwconv(False, False, dst_list_0, src_list_0, self.inner_process_nums_tail_loop,
                                        self.conv1_stride_0, self.conv1_stride_1)
            with self.tik_instance.for_range(0, self.tail_loop_true_nums / self.dtype_rate) as padding_41_index:
                self.tik_instance.data_move(
                    self.ub_first[padding_41_index * self.dtype_rate * Constant.TRANS_MIN_BLKS],
                    self.ub_second[(self.tail_loop_true_nums / self.dtype_rate - 1) *
                                self.dtype_rate * Constant.TRANS_MIN_BLKS], 0, 1, self.dtype_rate, 0, 0)
            dst_list_1 = [self.ub_second[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
            src_list_1 = [self.ub_first[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
            self.tik_instance.vnchwconv(False, False, dst_list_1, src_list_1, self.inner_process_nums_tail_loop,
                                        self.conv2_stride_0, self.conv2_stride_1)
            with self.tik_instance.if_scope((self.address_rollback_sum == 2) & (self.inner_loop_times == 1)):
                for i in range(16):
                    self.ub_addr_rollback[15 - i].set_as(self.ub_second[self.tail_loop_true_nums - i - 1])
                self.tik_instance.data_move(
                    self.output_gm[self.outer_move_out_index * self.output_dim_4 + self.padding_40 +
                                self.input_dim_4], self.ub_second, 0, 1, self.inner_process_nums_tail_loop - 1,
                    0, 0)
                self.tik_instance.data_move(
                    self.output_gm[self.outer_move_out_index * self.output_dim_4 + self.padding_40 +
                                self.input_dim_4 + self.tail_loop_true_nums - 16], self.ub_addr_rollback, 0, 1,
                    1, 0, 0)
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(
                    self.output_gm[self.outer_move_out_index * self.output_dim_4 + self.padding_40 +
                                self.input_dim_4], self.ub_second, 0, 1, self.inner_process_nums_tail_loop, 0, 0)

            self.get_vnchw_stride(one_loop_process_blocks, True)
            with self.tik_instance.for_range(0, self.inner_loop_times - 1) as inner_loop_index:
                self.move_in_start_blocks.set_as((inner_loop_index + 1) * one_loop_process_blocks)
                self.tik_instance.data_move(
                    self.ub_first,
                    self.input_gm[(self.outer_move_in_index + 1) * self.input_dim_4 - self.dtype_rate], 0, 1,
                                one_loop_process_blocks, 0, 0)
                dst_list_0 = [self.ub_second[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
                src_list_0 = [self.ub_first] * Constant.TRANS_MIN_BLKS
                self.tik_instance.vnchwconv(False, False, dst_list_0, src_list_0, one_loop_process_blocks,
                                            self.conv1_stride_0, self.conv1_stride_1)
                with self.tik_instance.for_range(
                        0, one_loop_process_blocks * Constant.TRANS_MIN_BLKS / self.dtype_rate) as padding_41_index:
                    self.tik_instance.data_move(
                        self.ub_first[padding_41_index * self.dtype_rate * Constant.TRANS_MIN_BLKS],
                        self.ub_second[0], 0, 1, self.dtype_rate, 0, 0)
                dst_list_1 = [self.ub_second[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
                src_list_1 = [self.ub_first[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
                self.tik_instance.vnchwconv(False, False, dst_list_1, src_list_1, one_loop_process_blocks,
                                            self.conv2_stride_0, self.conv2_stride_1)
                self.tik_instance.data_move(
                    self.output_gm[self.outer_move_out_index * self.output_dim_4 + self.input_dim_4 +
                                self.padding_40 + self.tail_loop_true_nums +
                                inner_loop_index * one_loop_process_blocks * Constant.TRANS_MIN_BLKS],
                    self.ub_second, 0, 1, one_loop_process_blocks, 0, 0)

    def process_padding_back(self):
        if tbe_platform.api_check_support("tik.data_move_pad"):
            self.support_move_pad_padding_back()
        else:
            self.not_support_move_pad_padding_back()

    def last_dim_inner_process(self):
        """
        mode_0 inner prcocess
        """
        with self.tik_instance.if_scope(self.padding_40 != 0):
            self.process_padding_front()
        self.process_middle()
        with self.tik_instance.if_scope(self.padding_41 != 0):
            self.process_padding_back()

    def get_vnchw_stride(self, line_num, is_last_dim):
        """
        if vnchwconv repeat=1 stride should be 0
        """
        if is_last_dim:
            with self.tik_instance.if_scope(line_num <= 1):
                self.conv1_stride_0.set_as(0)
                self.conv1_stride_1.set_as(0)
                self.conv2_stride_0.set_as(0)
                self.conv2_stride_1.set_as(0)
            with self.tik_instance.else_scope():
                self.conv1_stride_0.set_as(16)
                self.conv1_stride_1.set_as(1)
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
        elif mode == Constant.MODE4:
            self.inner_loop_times = self.tik_instance.Scalar(self.tiling_dtype, "inner_loop_times")
            self.inner_col_num_non_last_row = self.tik_instance.Scalar(self.tiling_dtype, "inner_col_num_non_last_row")
            self.inner_col_num_last_row = self.tik_instance.Scalar(self.tiling_dtype, "inner_col_num_last_row")
            self.inner_row_num = self.tik_instance.Scalar(self.tiling_dtype, "inner_row_num")
            self.inner_process_nums_tail_loop = self.tik_instance.Scalar(self.tiling_dtype, 
                                                                         "inner_process_nums_tail_loop")
            self.conv1_stride_0 = self.tik_instance.Scalar(self.tiling_dtype, "conv1_stride_0", init_value=0)
            self.conv1_stride_1 = self.tik_instance.Scalar(self.tiling_dtype, "conv1_stride_1", init_value=0)
            self.conv2_stride_0 = self.tik_instance.Scalar(self.tiling_dtype, "conv2_stride_0", init_value=0)
            self.conv2_stride_1 = self.tik_instance.Scalar(self.tiling_dtype, "conv2_stride_1", init_value=0)
            self.output_dim_4_align_blocks = self.tik_instance.Scalar(self.tiling_dtype,
                                                                      "output_dim_4_align_blocks",
                                                                      init_value=(self.output_dim_4 + 15) // 16)
            self.ub_first = self.tik_instance.Tensor(self.inner_dtype, [self.output_dim4_max_cnt_block_align * 16 * 16],
                                                     name="ub_first",
                                                     scope=tik.scope_ubuf)
            self.ub_second = self.tik_instance.Tensor(self.inner_dtype,
                                                      [self.output_dim4_max_cnt_block_align * 16 * 16],
                                                      name="ub_second",
                                                      scope=tik.scope_ubuf)
            self.one_loop_process_lines = self.tik_instance.Scalar(
                self.tiling_dtype, "one_loop_process_lines",
                init_value=self.output_dim4_max_cnt_block_align // self.output_dim_4_align_blocks)
            self.now_process_lines = self.tik_instance.Scalar(self.tiling_dtype, "now_process_lines", init_value=0)
            self.move_out_start_line = self.tik_instance.Scalar(self.tiling_dtype, "move_out_start_line", init_value=0)
        elif mode in (Constant.MODE0, Constant.MODE1):
            self.process_blk_in = self.tik_instance.Scalar(self.tiling_dtype, "process_blk_in")
            self.input_ceil_algin = self.tik_instance.Scalar(self.tiling_dtype, "input_ceil_algin")
            self.process_blk_out = self.tik_instance.Scalar(self.tiling_dtype, "process_blk_out")
            self.output_ceil_algin = self.tik_instance.Scalar(self.tiling_dtype, "output_ceil_algin")
            self.inner_col_num_non_last_row = self.tik_instance.Scalar(self.tiling_dtype, "inner_col_num_non_last_row")
            self.inner_col_num_last_row = self.tik_instance.Scalar(self.tiling_dtype, "inner_col_num_last_row")
            self.inner_row_num = self.tik_instance.Scalar(self.tiling_dtype, "inner_row_num")
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
            self.ub_first = self.tik_instance.Tensor(self.inner_dtype, 
                                                     [self.output_dim4_max_cnt_block_align * Constant.VEC_LENGTH],
                                                     name="ub_first",
                                                     scope=tik.scope_ubuf)
            self.ub_second = self.tik_instance.Tensor(self.inner_dtype,
                                                      [self.output_dim4_max_cnt_block_align * Constant.VEC_LENGTH],
                                                      name="ub_second",
                                                      scope=tik.scope_ubuf)
            self.ub_addr_rollback = self.tik_instance.Tensor(self.inner_dtype, [16],
                                                             name="ub_addr_rollback",
                                                             scope=tik.scope_ubuf)
            self.one_loop_process_lines = self.tik_instance.Scalar(self.tiling_dtype,
                                                                   "one_loop_process_lines",
                                                                   init_value=0)
            self.tail_loop_true_nums = self.tik_instance.Scalar(self.tiling_dtype, "tail_loop_true_nums", init_value=0)
            self.padding40_align_blocks.set_as((self.padding_40 + 15) / 16)
            self.padding41_align_blocks.set_as((self.padding_41 + 15) / 16)
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
        # do process padding_30 segment
        with self.tik_instance.if_scope(self.padding_30 != 0):
            self.inner_loop_times.set_as(
                (self.padding_30 + self.one_loop_process_lines - 1) / self.one_loop_process_lines)
            with self.tik_instance.for_range(0, self.inner_loop_times) as inner_loop_index:
                with self.tik_instance.if_scope(inner_loop_index < self.inner_loop_times - 1):
                    self.start_line.set_as(0)
                    self.move_out_start_line.set_as(inner_loop_index * self.one_loop_process_lines)
                    self.now_process_lines.set_as(self.one_loop_process_lines)
                with self.tik_instance.else_scope():
                    self.inner_process_nums_tail_loop.set_as(self.padding_30 -
                                                             (self.inner_loop_times - 1) * self.one_loop_process_lines)
                    self.start_line.set_as(0)
                    self.move_out_start_line.set_as((self.inner_loop_times - 1) * self.one_loop_process_lines)
                    self.now_process_lines.set_as(self.inner_process_nums_tail_loop)
                self.inner_padding30_process()

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
                        self.inner_process_nums_tail_loop.set_as((16 + self.output_dim_4 - 1) / 16)
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
        with self.tik_instance.if_scope(self.padding_31 != 0):
            self.inner_loop_times.set_as(
                (self.padding_31 + self.one_loop_process_lines - 1) / self.one_loop_process_lines)
            self.inner_process_nums_tail_loop.set_as(self.padding_31 -
                                                     (self.inner_loop_times - 1) * self.one_loop_process_lines)
            with self.tik_instance.for_range(0, self.inner_loop_times) as inner_loop_index:
                with self.tik_instance.if_scope(inner_loop_index < self.inner_loop_times - 1):
                    self.start_line.set_as(self.input_dim_3 - 1)
                    self.move_out_start_line.set_as(inner_loop_index * self.one_loop_process_lines)
                    self.now_process_lines.set_as(self.one_loop_process_lines)
                    self.address_rollback[2].set_as(0)
                with self.tik_instance.else_scope():
                    with self.tik_instance.if_scope(self.address_rollback[0] * self.address_rollback[1] != 0):
                        with self.tik_instance.if_scope(self.inner_process_nums_tail_loop * self.output_dim_4 < 16):
                            self.inner_process_nums_tail_loop.set_as(16 + self.output_dim_4 - 1 / self.output_dim_4)
                            self.start_line.set_as(self.input_dim_3 - 1)
                            self.move_out_start_line.set_as(self.padding_31 - self.inner_process_nums_tail_loop)
                            self.now_process_lines.set_as(self.inner_process_nums_tail_loop)
                            self.address_rollback[2].set_as(1)
                        with self.tik_instance.elif_scope(self.inner_process_nums_tail_loop * self.output_dim_4 > 16):
                            self.start_line.set_as(self.input_dim_3 - 1)
                            self.move_out_start_line.set_as((self.inner_loop_times - 1) * self.one_loop_process_lines)
                            self.inner_process_nums_tail_loop.set_as(self.padding_31 - (self.inner_loop_times - 1) *
                                                                     self.one_loop_process_lines)
                            self.now_process_lines.set_as(self.inner_process_nums_tail_loop)
                            self.address_rollback[2].set_as(1)
                        with self.tik_instance.else_scope():
                            self.start_line.set_as(self.input_dim_3 - 1)
                            self.move_out_start_line.set_as((self.inner_loop_times - 1) * self.one_loop_process_lines)
                            self.inner_process_nums_tail_loop.set_as(self.padding_31 - (self.inner_loop_times - 1) *
                                                                     self.one_loop_process_lines)
                            self.now_process_lines.set_as(self.inner_process_nums_tail_loop)
                            self.address_rollback[2].set_as(0)
                    with self.tik_instance.else_scope():
                        self.start_line.set_as(self.input_dim_3 - 1)
                        self.move_out_start_line.set_as((self.inner_loop_times - 1) * self.one_loop_process_lines)
                        self.inner_process_nums_tail_loop.set_as(self.padding_31 - (self.inner_loop_times - 1) *
                                                                 self.one_loop_process_lines)
                        self.now_process_lines.set_as(self.inner_process_nums_tail_loop)
                        self.address_rollback[2].set_as(0)
                self.inner_padding31_process()

    def inner_padding30_process(self):
        """
        padding dim3 left padding by data_move
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
                self.ub_second[0 * self.input_dim_4 * Constant.TRANS_MIN_BLKS],
                0, 1, self.input_dim_4, 0, 0)
        self.do_data_move_for_padding_dim_4(self.ub_first, self.ub_second, self.now_process_lines)
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
        padding dim3 padding by data_move
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
        self.do_data_move_for_padding_dim_4(self.ub_second, self.ub_first, self.now_process_lines)
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
                               (self.padding_30 + self.move_out_start_line) * self.output_dim_4], self.ub_second, 0, 1,
                self.process_blks - 1, 0, 0)
            for i in range(16):
                self.ub_addr_rollback[15 - i].set_as(self.ub_second[self.now_process_lines * self.output_dim_4 - i - 1])
            self.tik_instance.data_move(self.output_gm[(self.outer_move_out_index + 1) * self.inner_outsize - 16],
                                        self.ub_addr_rollback, 0, 1, 1, 0, 0)
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(
                self.output_gm[self.outer_move_out_index * self.inner_outsize +
                               (self.padding_30 + self.move_out_start_line) * self.output_dim_4], self.ub_second, 0, 1,
                self.process_blks, 0, 0)

    def inner_padding31_process(self):
        """
        padding dim3 right padding by data_move
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
                self.ub_second[0 * self.input_dim_4 * Constant.TRANS_MIN_BLKS],
                0, 1, self.input_dim_4, 0, 0)
        self.do_data_move_for_padding_dim_4(self.ub_first, self.ub_second, self.now_process_lines)
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
                               (self.padding_30 + self.input_dim_3 + self.move_out_start_line) * self.output_dim_4],
                self.ub_first, 0, 1, self.process_blks - 1, 0, 0)
            for i in range(16):
                self.ub_addr_rollback[15 - i].set_as(self.ub_first[self.now_process_lines * self.output_dim_4 - i - 1])
            self.tik_instance.data_move(self.output_gm[(self.outer_move_out_index + 1) * self.inner_outsize - 16],
                                        self.ub_addr_rollback, 0, 1, 1, 0, 0)
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(
                self.output_gm[self.outer_move_out_index * self.inner_outsize +
                               (self.padding_30 + self.input_dim_3 + self.move_out_start_line) * self.output_dim_4],
                self.ub_first, 0, 1, self.process_blks, 0, 0)

    def do_data_move_for_padding_dim_4(self, in_ub, out_ub, process_lines):
        """
        padding last dim by data_move
        """
        self.tik_instance.data_move(out_ub[self.padding_40 * Constant.TRANS_MIN_BLKS], in_ub, 0, process_lines,
                                    self.input_dim_4, 0, self.padding_40 + self.padding_41)
        with self.tik_instance.for_range(0, self.padding_40 / self.dtype_rate) as padding_40_index:
            self.tik_instance.data_move(
                out_ub[padding_40_index * self.dtype_rate * Constant.TRANS_MIN_BLKS],
                in_ub, 0, process_lines, self.dtype_rate,
                self.input_dim_4 - self.dtype_rate, self.output_dim_4 - self.dtype_rate)
        with self.tik_instance.for_range(0, self.padding_41 / self.dtype_rate) as padding_41_index:
            self.tik_instance.data_move(
                out_ub[(self.padding_40 + self.input_dim_4 + padding_41_index * self.dtype_rate) *
                       Constant.TRANS_MIN_BLKS],
                in_ub[(self.input_dim_4 - self.dtype_rate) *
                      Constant.TRANS_MIN_BLKS], 0, process_lines, self.dtype_rate,
                self.input_dim_4 - self.dtype_rate, self.output_dim_4 - self.dtype_rate)

    def replication_pad_compute_tiling(self):
        """
        replication_pad_compute_tiling
        """
        self.tiling_args()
        with self.tik_instance.for_range(0, self.core_num_var, block_num=self.core_num_var) as core_index:
            with self.tik_instance.if_scope(core_index < self.core_used_num):
                with self.tik_instance.if_scope(self.tiling_key == Constant.MODE0):
                    with self.tik_instance.new_stmt_scope():
                        self.do_tiling_key_mode_0(core_index)
                with self.tik_instance.elif_scope(self.tiling_key == Constant.MODE1):
                    with self.tik_instance.new_stmt_scope():
                        self.do_tiling_key_mode_1(core_index)
                with self.tik_instance.elif_scope(self.tiling_key == Constant.MODE2):
                    with self.tik_instance.new_stmt_scope():
                        self.do_tiling_key_mode_2(core_index)
                with self.tik_instance.elif_scope(self.tiling_key == Constant.MODE4):
                    with self.tik_instance.new_stmt_scope():
                        self.do_tiling_key_mode_4(core_index)

    def replication_pad_compute(self):
        """
        replication_pad_compute
        do_pad with different tiling key
        Constant.MODE0: the last dim of output > 128*core_num, and cut by last dim
                and do pad with data move
        Constant.MODE1: the last dim of output => 960, and cut by outer dim(0-3)
                and do pad with data move
        """
        self.replication_pad_compute_tiling()
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
def replication_pad_v3(x, paddings, constant_values, y, mode, 
                       padding_contiguous=True, kernel_name="replication_pad_v3"):
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
        cce kernel name, default value is "replication_pad_v3"

    Returns
    -------
    None.
    """
    src_dtype = x.get("dtype").lower()
    paddings_dtype = paddings.get("dtype").lower()
    supported_dtype = ("float16", "int16", "uint16", "float32", "int32", "int64", "uint32")
    para_check.check_dtype(src_dtype, supported_dtype, param_name="x")
    para_check.check_dtype(paddings_dtype, ("int32", "int64"), param_name="paddings")
    obj = ReplicationPadInit(mode, constant_values, padding_contiguous, kernel_name)
    obj.init_src_dst_gm(x, paddings, y, constant_values)

    return obj.replication_pad_compute()
