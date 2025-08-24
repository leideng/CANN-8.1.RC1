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
reverse_v2.py
"""
import functools
from tbe.common.platform import get_bit_len
from impl import constant_util as constant
from impl.util import util_tik_comm_func
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.util_common import ceil_div_scalar as ceil_div
from impl.util.util_common import div_align_scalar as div_align
from impl.batch_multi_class_nms_topk import sort_within_ub
from tbe.common.platform.platform_info import api_check_support


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant.
    """
    MAX_INT64 = 2 ** 63 - 1
    EIGHT_BIT = 8
    RESERVED_UB = 1024 * 8
    VNHW_MIN_NUM = 256
    TILING_NUM = 35
    TOPK_THRESHOLD = 16
    MAX_ELE_LAST_LARGE_SIZE = 512


# 'pylint: disable=too-many-lines,too-many-instance-attributes,too-many-statements,too-many-arguments,too-many-locals
class ReverseExt2:
    """
    Function: use to store reverse_ext2 schedule parameters
    Modify : 2021-04-09
    """
    def __init__(self, shape_x, dtype_x, axis, kernel_name):
        """
        init scatter_nd base parameters

        Parameters
        ----------
        shape_x: tuple or list
            the shape of input tensor
        dtype_x: string
            the dtype of input tensor
        axis: dict
            the axis list for reverse
        kernel_name: str
            kernel name, default value is "reverse_ext2"

        Returns
        -------
        None
        """
        self.tik_instance = tik.Tik()
        self.aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - Constant.RESERVED_UB
        self.shape_x = list(shape_x)
        self.dtype_x = dtype_x
        self.inner_dtype = "float16"
        self.inner_bytes_size = get_bit_len(self.inner_dtype) // Constant.EIGHT_BIT
        self.input_bytes_size = get_bit_len(dtype_x) // Constant.EIGHT_BIT
        self.ub_element_number = self.ub_size_bytes // self.inner_bytes_size
        self.avaliable_ub = self.ub_element_number // 2
        self.block_num = constant.BLOCK_SIZE // self.inner_bytes_size
        self.vector_num = self.block_num * 8
        self.process_num = 0
        self.data_move_pad_support = api_check_support("tik.data_move_pad", self.inner_dtype)
        self.tiling_num_align = div_align(Constant.TILING_NUM, 4)

        self.axis = axis
        self.input_data_build = self.tik_instance.Tensor(self.dtype_x, (Constant.MAX_INT64 // 4,), name="input_data",
                                                   scope=tik.scope_gm)
        self.output_data_build = self.tik_instance.Tensor(self.dtype_x, (Constant.MAX_INT64 // 4,), name="output_data",
                                                    scope=tik.scope_gm)
        self.input_data = self.input_data_build.reinterpret_cast_to(self.inner_dtype)
        self.input_axis = self.tik_instance.Tensor(self.axis.get("dtype"), (Constant.MAX_INT64,),
                                                   name="input_axis",
                                                   scope=tik.scope_gm)
        self.output_data = self.output_data_build.reinterpret_cast_to(self.inner_dtype)
        self.tiling_gm = self.tik_instance.Tensor("int64", (self.tiling_num_align,), 
                                                  name="tiling_gm", scope=tik.scope_gm)

        # assist data for topk (1023, 1022, 1021 ......  2, 1, 0)
        assist_data = list(range(2048))
        self.assist_num_gm = self.tik_instance.Tensor(self.inner_dtype, (2048,), name="assist_num_gm",
                                                      scope=tik.scope_gm, init_value=assist_data)

        self.kernel_name = kernel_name
        self.inner_shape_0 = self.tik_instance.Scalar(dtype="int64", name="tiling_inner_shape_0")
        self.inner_shape_1 = self.tik_instance.Scalar(dtype="int64", name="tiling_inner_shape_1")
        self.inner_shape_2 = self.tik_instance.Scalar(dtype="int64", name="tiling_inner_shape_2")
        self.inner_shape_3 = self.tik_instance.Scalar(dtype="int64", name="tiling_inner_shape_3")
        self.inner_shape_4 = self.tik_instance.Scalar(dtype="int64", name="tiling_inner_shape_4")
        self.inner_shape_5 = self.tik_instance.Scalar(dtype="int64", name="tiling_inner_shape_5")
        self.inner_shape_6 = self.tik_instance.Scalar(dtype="int64", name="tiling_inner_shape_6")
        self.inner_dividends_0 = self.tik_instance.Scalar(dtype="int64", name="inner_dividends_0")
        self.inner_dividends_1 = self.tik_instance.Scalar(dtype="int64", name="inner_dividends_1")
        self.inner_dividends_2 = self.tik_instance.Scalar(dtype="int64", name="inner_dividends_2")
        self.inner_dividends_3 = self.tik_instance.Scalar(dtype="int64", name="inner_dividends_3")
        self.inner_dividends_4 = self.tik_instance.Scalar(dtype="int64", name="inner_dividends_4")
        self.inner_dividends_5 = self.tik_instance.Scalar(dtype="int64", name="inner_dividends_5")
        self.inner_dividends_6 = self.tik_instance.Scalar(dtype="int64", name="inner_dividends_6")
        self.inner_axis_0 = self.tik_instance.Scalar(dtype="int64", name="tiling_inner_axis_0")
        self.inner_axis_1 = self.tik_instance.Scalar(dtype="int64", name="tiling_inner_axis_1")
        self.inner_axis_2 = self.tik_instance.Scalar(dtype="int64", name="tiling_inner_axis_2")
        self.inner_axis_3 = self.tik_instance.Scalar(dtype="int64", name="tiling_inner_axis_3")
        self.inner_axis_4 = self.tik_instance.Scalar(dtype="int64", name="tiling_inner_axis_4")
        self.inner_axis_5 = self.tik_instance.Scalar(dtype="int64", name="tiling_inner_axis_5")
        self.inner_axis_6 = self.tik_instance.Scalar(dtype="int64", name="tiling_inner_axis_6")

        self.is_split_axi_reverse = self.tik_instance.Scalar("int64", name="tiling_is_split_axi_reverse")
        self.split_part_num = self.tik_instance.Scalar("int64", name="tiling_split_part_num")
        self.split_dim = self.tik_instance.Scalar("int64", name="tiling_split_dim")
        self.tiling_key = self.tik_instance.Scalar("int64", name="tiling_tiling_key")
        self.inner_real_dims = self.tik_instance.Scalar("int64", name="inner_real_dims")
        self.outer_real_dims = self.tik_instance.Scalar("int64", name="outer_real_dims")
        self.outer_shape = self.tik_instance.ScalarArray(dtype="int64", length=7, name="outer_shape",
                                                         init_value=1)

        self.outer_dividends = self.tik_instance.ScalarArray(dtype="int64", length=7, name="outer_dividends",
                                                             init_value=1)
        self.outer_axis = self.tik_instance.ScalarArray(dtype="int64", length=7, name="outer_axis",
                                                        init_value=1)
        self.inner_shape = [
            self.inner_shape_0, self.inner_shape_1, self.inner_shape_2, self.inner_shape_3, self.inner_shape_4,
            self.inner_shape_5, self.inner_shape_6
        ]
        self.inner_dividends = None
        self.inner_total_num_list = None
        self.inner_loop = None
        self.inner_axis = None
        self.move_out_index = self.tik_instance.Scalar(dtype="int64", name="move_out_index")
        self.current_index = self.tik_instance.Scalar(dtype="int64", name="current_index")

        self.core_outer_num = None
        self.core_outer_start = None
        self.input_data_ub = None
        self.output_data_ub = None
        # max_vnchw_block_num must be 256 align
        self.max_vnchw_block_num = (self.avaliable_ub // 16 // 256) * 256
        self.axis_tmp_scalar = self.tik_instance.Scalar("int64", name="axis_tmp_scalar")
        self.outer_total_num = self.tik_instance.Scalar(dtype="int64", name="outer_total_num")
        self.core_num_scalar = self.tik_instance.Scalar(dtype="int64", name="core_num_scalar",
                                                        init_value=self.aicore_num)

        # get vconcat support info
        if tbe_platform.api_check_support("tik.vconcat") and tbe_platform.api_check_support("tik.vrpsort16"):
            self.is_vconcat_int = 1
        else:
            self.is_vconcat_int = 0

    def execute_tilling(self):
        """
        execute_tilling, copy tiling and read
        """
        with self.tik_instance.new_stmt_scope():
            tiling_ub = self.tik_instance.Tensor("int64", (self.tiling_num_align,), name="tiling_ub",
                                                 scope=tik.scope_ubuf)
            self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, ceil_div(self.tiling_num_align, 4), 0, 0)
            self.set_tiling_param(tiling_ub)
        self.inner_total_num_list = []
        for i, _ in enumerate(self.inner_shape):
            inner_shape_offset = self.tik_instance.Scalar(dtype="int64", name="inner_shape_offset_" + str(i))
            inner_shape_offset.set_as(functools.reduce(lambda x, y: x * y, self.inner_shape[0:i + 1]))
            self.inner_total_num_list.append(inner_shape_offset)

    def core_scudule_tilling(self, _core_idx):
        """
        core_scudule_tilling
        """
        self.core_outer_num = self.tik_instance.Scalar("int64", name="core_outer_num")
        self.core_outer_start = self.tik_instance.Scalar("int64", name="core_outer_start")
        outer_num = self.outer_total_num
        self.core_outer_num.set_as((outer_num + self.core_num_scalar - 1) // self.core_num_scalar)
        self.core_outer_start.set_as(_core_idx * self.core_outer_num)
        with self.tik_instance.if_scope(outer_num % self.core_num_scalar != 0):
            with self.tik_instance.if_scope(_core_idx >= (outer_num % self.core_num_scalar)):
                self.core_outer_num.set_as(self.core_outer_num - 1)
                self.core_outer_start.set_as(_core_idx * self.core_outer_num + outer_num % self.core_num_scalar)

    def set_tiling_param(self, tiling_ub):
        """
        set_tiling_param
        """
        self.tiling_key.set_as(tiling_ub[0])
        self.inner_shape_0.set_as(tiling_ub[1])
        self.inner_shape_1.set_as(tiling_ub[2])
        self.inner_shape_2.set_as(tiling_ub[3])
        self.inner_shape_3.set_as(tiling_ub[4])
        self.inner_shape_4.set_as(tiling_ub[5])
        self.inner_shape_5.set_as(tiling_ub[6])
        self.inner_shape_6.set_as(tiling_ub[7])

        self.inner_axis_0.set_as(tiling_ub[8])
        self.inner_axis_1.set_as(tiling_ub[9])
        self.inner_axis_2.set_as(tiling_ub[10])
        self.inner_axis_3.set_as(tiling_ub[11])
        self.inner_axis_4.set_as(tiling_ub[12])
        self.inner_axis_5.set_as(tiling_ub[13])
        self.inner_axis_6.set_as(tiling_ub[14])

        self.outer_shape[0].set_as(tiling_ub[15])
        self.outer_shape[1].set_as(tiling_ub[16])
        self.outer_shape[2].set_as(tiling_ub[17])
        self.outer_shape[3].set_as(tiling_ub[18])
        self.outer_shape[4].set_as(tiling_ub[19])
        self.outer_shape[5].set_as(tiling_ub[20])
        self.outer_shape[6].set_as(tiling_ub[21])

        self.outer_axis[0].set_as(tiling_ub[22])
        self.outer_axis[1].set_as(tiling_ub[23])
        self.outer_axis[2].set_as(tiling_ub[24])
        self.outer_axis[3].set_as(tiling_ub[25])
        self.outer_axis[4].set_as(tiling_ub[26])
        self.outer_axis[5].set_as(tiling_ub[27])
        self.outer_axis[6].set_as(tiling_ub[28])

        self.is_split_axi_reverse.set_as(tiling_ub[29])
        self.split_part_num.set_as(tiling_ub[30])
        self.split_dim.set_as(tiling_ub[31])
        self.inner_real_dims.set_as(tiling_ub[32])
        self.tik_instance.scalar_max(self.inner_real_dims, self.inner_real_dims, 2)
        self.outer_real_dims.set_as(tiling_ub[33])
        self.core_num_scalar.set_as(tiling_ub[34])

        # update outer_shape/outer_axis with real dims
        with self.tik_instance.for_range(0, self.outer_real_dims) as real_idx:
            self.outer_shape[real_idx].set_as(self.outer_shape[7 - self.outer_real_dims + real_idx])
            self.outer_axis[real_idx].set_as(self.outer_axis[7 - self.outer_real_dims + real_idx])

        # update outer_dividends for case than last dim do not reverse
        self.axis_tmp_scalar.set_as(1)
        with self.tik_instance.for_range(0, self.outer_real_dims) as real_idx:
            dividends_idx = self.outer_real_dims - 1 - real_idx
            self.outer_dividends[dividends_idx].set_as(self.axis_tmp_scalar)
            self.axis_tmp_scalar.set_as(self.axis_tmp_scalar * self.outer_shape[dividends_idx])
        # update outer_dividends for case than last dim do not reverse end
        self.outer_total_num.set_as(self.axis_tmp_scalar)

    def axis_compute_for_new(self, index, current_index, move_out_index, loop_shape, axies, dividends):
        """
        axis_compute_for_new
        """
        current_index.set_as(self.outer_axis[0])
        move_out_index.set_as(0)
        # first idx
        idx_tmp_not_axis = (index // dividends[0] * dividends[0]) * (1 - axies[0])
        for axis_id in range(6):
            idx_tmp_not_axis = idx_tmp_not_axis \
                               + ((index % dividends[axis_id]) // dividends[axis_id + 1] * dividends[axis_id + 1]) \
                               * (1 - axies[axis_id + 1])

        idx_tmp_axis = ((loop_shape[0] - 1 - index // dividends[0]) * dividends[0]) * axies[0]
        for axis_id in range(6):
            idx_tmp_axis = idx_tmp_axis \
                           + ((loop_shape[axis_id + 1] - 1 - (index % dividends[axis_id]) // dividends[axis_id + 1])
                              * dividends[axis_id + 1]) * axies[axis_id + 1]
        move_out_index.set_as(move_out_index + idx_tmp_not_axis + idx_tmp_axis)

    def axis_compute(self, index, current_index, move_out_index, loop_shape, axies, dividends):
        """
        axis_compute_for_new
        """
        current_index.set_as(index)
        move_out_index.set_as(0)
        for axis_id in range(7):
            with self.tik_instance.if_scope(axies[axis_id] == 0):
                move_out_index.set_as(move_out_index + current_index // dividends[axis_id] * dividends[axis_id])
            with self.tik_instance.if_scope(axies[axis_id] != 0):
                move_out_index.set_as(move_out_index +
                                      (loop_shape[axis_id] - 1 - current_index // dividends[axis_id]) *
                                      dividends[axis_id])
            current_index.set_as(current_index - current_index // dividends[axis_id] * dividends[axis_id])

    def axis_compute_with_scalar_array(self, index, current_index, move_out_index,
                                       loop_shape, axies, dividends, real_dims):
        """
        axis_compute_with_scalar_array
        """
        current_index.set_as(index)
        move_out_index.set_as(0)
        with self.tik_instance.for_range(0, real_dims) as axis_id:
            self.axis_tmp_scalar.set_as(current_index // dividends[axis_id] * dividends[axis_id])
            with self.tik_instance.if_scope(axies[axis_id] == 0):
                move_out_index.set_as(move_out_index + self.axis_tmp_scalar)
            with self.tik_instance.if_scope(axies[axis_id] != 0):
                move_out_index.set_as(
                    move_out_index
                    + (loop_shape[axis_id] - 1) * dividends[axis_id]
                    - self.axis_tmp_scalar)
            current_index.set_as(current_index - self.axis_tmp_scalar)

    def copy_data(self, dst, src, nburst, burst_block, burst_byte):
        with self.tik_instance.if_scope(nburst > 0):
            if self.data_move_pad_support:
                self.tik_instance.data_move_pad(dst, src, nburst, burst_byte, 0, 0)
            else:
                self.tik_instance.data_move(dst, src, 0, nburst, burst_block, 0, 0)

    def tiling_rules(self):
        """
        tiling_rules
        tiling_0: do not reverse with last dim, and last dim < 16
        tiling_1: do not reverse with last dim, and last dim is 16 align and < 512
        tiling_2: do not reverse with last dim, and last dim is not 16 align and < 512
        tiling_3: do not reverse with last dim, and last dim > 512
        tiling_4: do reverse with last dim, and last dim < 128
        tiling_5: do reverse with last dim, and last dim > 128 and < 512
        tiling_6: do reverse with last dim, and last dim > 512
        tiling_11: do reverse with topk, when last dim is less
        """
        # apply for scalar array
        self.inner_loop = self.tik_instance.ScalarArray(dtype="int64", length=7, name="inner_loop",
                                                        init_value=self.inner_shape)
        inner_axis = [self.inner_axis_0, self.inner_axis_1, self.inner_axis_2, self.inner_axis_3, self.inner_axis_4,
                      self.inner_axis_5, self.inner_axis_6]
        self.inner_axis = self.tik_instance.ScalarArray(dtype="int64", length=7, name="inner_axis",
                                                        init_value=inner_axis)
        self.inner_dividends = self.tik_instance.ScalarArray(dtype="int64", length=7, name="inner_dividends",
                                                             init_value=1)
        # update inner_loop with real dims
        with self.tik_instance.for_range(0, self.inner_real_dims) as real_idx:
            self.inner_loop[real_idx].set_as(self.inner_loop[7 - self.inner_real_dims + real_idx])
            self.inner_axis[real_idx].set_as(self.inner_axis[7 - self.inner_real_dims + real_idx])

        # add new performance case: use proposal topk to reverse
        self.axis_tmp_scalar.set_as(1)
        with self.tik_instance.for_range(0, self.inner_real_dims) as real_idx:
            dividends_idx = self.inner_real_dims - 1 - real_idx
            self.inner_dividends[dividends_idx].set_as(self.axis_tmp_scalar)
            self.axis_tmp_scalar.set_as(self.axis_tmp_scalar * self.inner_loop[dividends_idx])

        # if inner is less 512 change to tiling 4 or 0
        with self.tik_instance.if_scope(self.outer_real_dims != 0):
            with self.tik_instance.if_scope(tik.all(self.tiling_key == 11, self.axis_tmp_scalar < 512)):
                self.tiling_key.set_as(0)
                with self.tik_instance.if_scope(self.inner_axis_6 == 1):
                    self.tiling_key.set_as(4)
        if self.is_vconcat_int == 1:
            with self.tik_instance.if_scope(self.tiling_key == 11):
                with self.tik_instance.new_stmt_scope():
                    self.tiling_topk_compute()

        # update inner_dividends for case than last dim do not reverse
        self.axis_tmp_scalar.set_as(1)
        with self.tik_instance.for_range(0, self.inner_real_dims - 1) as real_idx:
            inner_dividends_idx = self.inner_real_dims - 2 - real_idx
            self.inner_dividends[inner_dividends_idx].set_as(self.axis_tmp_scalar)
            self.axis_tmp_scalar.set_as(self.axis_tmp_scalar * self.inner_loop[inner_dividends_idx])
        # update inner_dividends for case than last dim do not reverse end

        max_split_part_num = self.tik_instance.Scalar(dtype="int64", name="max_split_part_num")
        with self.tik_instance.if_scope(self.split_dim == self.inner_total_num_list[-1]):
            self.split_dim.set_as(1)
            self.split_part_num.set_as(1)
        with self.tik_instance.if_scope(self.tiling_key == 0):
            with self.tik_instance.new_stmt_scope():
                mid_split_num = self.inner_total_num_list[-2] // self.split_dim
                max_split_part_num.set_as(self.max_vnchw_block_num // mid_split_num)
                self.tik_instance.scalar_min(self.split_part_num, self.split_dim, max_split_part_num)
                with self.tik_instance.for_range(self.core_outer_start,
                                                 self.core_outer_num + self.core_outer_start) as index:
                    current_index = self.tik_instance.Scalar(dtype="int64", name="current_index")
                    move_out_index = self.tik_instance.Scalar(dtype="int64", name="move_out_index")
                    self.axis_compute_with_scalar_array(
                        index, current_index, move_out_index, self.outer_shape, self.outer_axis,
                        self.outer_dividends, self.outer_real_dims)
                    self.reverse_non_last_axis_small(index, move_out_index)
        with self.tik_instance.if_scope(self.tiling_key == 1):
            with self.tik_instance.new_stmt_scope():
                with self.tik_instance.for_range(self.core_outer_start,
                                                 self.core_outer_num + self.core_outer_start) as index:
                    current_index = self.tik_instance.Scalar(dtype="int64", name="current_index")
                    move_out_index = self.tik_instance.Scalar(dtype="int64", name="move_out_index")
                    self.axis_compute_with_scalar_array(
                        index, current_index, move_out_index, self.outer_shape, self.outer_axis,
                        self.outer_dividends, self.outer_real_dims)
                    self.reverse_non_last_axis_16_aligned(index, move_out_index)
        with self.tik_instance.if_scope(self.tiling_key == 2):
            with self.tik_instance.new_stmt_scope():
                with self.tik_instance.for_range(self.core_outer_start,
                                                 self.core_outer_num + self.core_outer_start) as index:
                    current_index = self.tik_instance.Scalar(dtype="int64", name="current_index")
                    move_out_index = self.tik_instance.Scalar(dtype="int64", name="move_out_index")
                    self.axis_compute_with_scalar_array(
                        index, current_index, move_out_index, self.outer_shape, self.outer_axis,
                        self.outer_dividends, self.outer_real_dims)
                    self.reverse_non_last_axis(index, move_out_index)
        with self.tik_instance.if_scope(self.tiling_key == 3):
            with self.tik_instance.new_stmt_scope():
                self.tiling_reverse_non_last_axis_large_compute()

        # update inner_dividends for case than last dim do reverse
        self.axis_tmp_scalar.set_as(1)
        with self.tik_instance.for_range(0, self.inner_real_dims - 2) as real_idx:
            inner_dividends_idx = self.inner_real_dims - 3 - real_idx
            self.inner_dividends[inner_dividends_idx].set_as(self.axis_tmp_scalar)
            self.axis_tmp_scalar.set_as(self.axis_tmp_scalar * self.inner_loop[inner_dividends_idx])
        # update inner_dividends for case than last dim do reverse end

        with self.tik_instance.if_scope(self.tiling_key == 4):
            with self.tik_instance.new_stmt_scope():
                mid_split_num = self.inner_total_num_list[-2] // self.split_dim
                last_dim_block = ceil_div(self.inner_shape[-1], self.block_num)
                block_num_align = div_align(self.max_vnchw_block_num // last_dim_block,
                                            Constant.VNHW_MIN_NUM, div_mode="floor")
                max_split_part_num.set_as(block_num_align // mid_split_num)
                self.tik_instance.scalar_min(self.split_part_num, self.split_dim, max_split_part_num)
                with self.tik_instance.for_range(self.core_outer_start,
                                                 self.core_outer_num + self.core_outer_start) as index:
                    current_index = self.tik_instance.Scalar(dtype="int64", name="current_index")
                    move_out_index = self.tik_instance.Scalar(dtype="int64", name="move_out_index")
                    self.axis_compute_with_scalar_array(
                        index, current_index, move_out_index, self.outer_shape, self.outer_axis,
                        self.outer_dividends, self.outer_real_dims)
                    self.reverse_small_shape(index, move_out_index)
        with self.tik_instance.if_scope(self.tiling_key == 5):
            with self.tik_instance.new_stmt_scope():
                mid_split_num = self.inner_total_num_list[-2] // self.split_dim
                last_dim_block = div_align(self.inner_shape[-1], Constant.VNHW_MIN_NUM) // self.block_num
                max_split_part_num.set_as(
                    div_align(self.max_vnchw_block_num // last_dim_block, 16, "floor") // mid_split_num)
                self.tik_instance.scalar_min(self.split_part_num, self.split_dim, max_split_part_num)
                with self.tik_instance.for_range(self.core_outer_start,
                                                 self.core_outer_num + self.core_outer_start) as index:
                    current_index = self.tik_instance.Scalar(dtype="int64", name="current_index")
                    move_out_index = self.tik_instance.Scalar(dtype="int64", name="move_out_index")
                    self.axis_compute_with_scalar_array(
                        index, current_index, move_out_index, self.outer_shape, self.outer_axis,
                        self.outer_dividends, self.outer_real_dims)
                    self.large_reverse(index, move_out_index)
        with self.tik_instance.if_scope(self.tiling_key == 6):
            with self.tik_instance.new_stmt_scope():
                with self.tik_instance.for_range(self.core_outer_start,
                                                 self.core_outer_num + self.core_outer_start) as index:
                    current_index = self.tik_instance.Scalar(dtype="int64", name="current_index")
                    move_out_index = self.tik_instance.Scalar(dtype="int64", name="move_out_index")
                    self.axis_compute_with_scalar_array(
                        index, current_index, move_out_index, self.outer_shape, self.outer_axis,
                        self.outer_dividends, self.outer_real_dims)
                    self.reverse_last_axis_large(index, move_out_index)

    def op_run(self):
        """
        op_run: core scedule and cut core
        """
        self.execute_tilling()
        with self.tik_instance.for_range(0, self.core_num_scalar, block_num=self.core_num_scalar) as core_index:
            self.core_scudule_tilling(core_index)
            self.tiling_rules()

        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}

        tbe_context.get_context().add_compile_info("global_variable_link", True)

        # add compile info
        # dtype_rate mean input_dtype byte // inner_dtype(fp16)
        # input_dtype is fp16/int16 dtype_rate == 1
        # input_dtype is fp32/int32 dtype_rate == 2
        dtype_rate = self.input_bytes_size // self.inner_bytes_size
        tbe_context.get_context().add_compile_info("vars", {
            "core_num": self.aicore_num,
            "max_elements": self.max_vnchw_block_num * 16,
            "max_elements_last_large_size": Constant.MAX_ELE_LAST_LARGE_SIZE,
            "dtype_rate": dtype_rate,
            "topk_threshold": Constant.TOPK_THRESHOLD,
            "is_vconcat": self.is_vconcat_int
        })

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.input_data_build, self.input_axis),
                                   outputs=[self.output_data_build],
                                   flowtable=[self.tiling_gm],
                                   config=opt_config)

        return self.tik_instance

    def reverse_non_last_axis(self, outer_move_in=0, outer_move_out=0):
        """
        reverse_non_last_axis for tiling 2
        """
        self.input_data_ub = self.tik_instance.Tensor(self.inner_dtype, (self.avaliable_ub,),
                                                      name="input_data_ub",
                                                      scope=tik.scope_ubuf)
        self.output_data_ub = self.tik_instance.Tensor(self.inner_dtype, (self.avaliable_ub,),
                                                       name="output_data_ub",
                                                       scope=tik.scope_ubuf)
        phase = self.tik_instance.Scalar("int64", name="phase")
        with self.tik_instance.if_scope(self.is_split_axi_reverse != 0):
            phase.set_as(self.inner_total_num_list[-2] // self.split_dim * (self.split_dim - self.split_part_num))
        with self.tik_instance.else_scope():
            phase.set_as(0)
        total_num = self.inner_total_num_list[-1]

        move_in_bias = self.tik_instance.Scalar("int32", name="bias")
        move_in_bias.set_as(
            self.inner_total_num_list[-1] // self.split_dim * self.split_part_num)

        loop_times = self.tik_instance.Scalar("int32", name="loop_times")
        loop_times.set_as((self.split_dim + self.split_part_num - 1) // self.split_part_num)

        block_num = self.tik_instance.Scalar("int32", name="block_num")
        block_num.set_as((self.inner_shape[-1] + 15) // 16)
        split_part_num = self.tik_instance.Scalar("int32", name="tmp_split_num")
        split_part_num.set_as(self.split_part_num)
        move_num = self.inner_total_num_list[-2] // self.split_dim * split_part_num
        mid_loop = self.inner_total_num_list[-2] // self.split_dim * split_part_num
        index_lenth = self.inner_total_num_list[-1]

        # for copy_in_info
        stride_copy = self.tik_instance.Scalar("int32", name="stride_copy")
        single_copy = self.tik_instance.Scalar("int32", name="stride_copy")
        stride_copy.set_as((self.inner_total_num_list[-2] // self.split_dim * split_part_num) // 16)
        single_copy.set_as((self.inner_total_num_list[-2] // self.split_dim * split_part_num) % 16)
        with self.tik_instance.for_range(0, loop_times) as split_id:
            bias = self.tik_instance.Scalar("int32", name="bias")
            with self.tik_instance.if_scope(split_id == loop_times - 1):
                split_part_num.set_as(self.split_dim - self.split_part_num * split_id)
                with self.tik_instance.if_scope(self.is_split_axi_reverse != 0):
                    phase.set_as(
                        self.inner_total_num_list[-2] // self.split_dim *
                        (self.split_dim - split_part_num))
                with self.tik_instance.else_scope():
                    phase.set_as(0)

                with self.tik_instance.if_scope(self.is_split_axi_reverse):
                    bias.set_as(0)
                with self.tik_instance.else_scope():
                    bias.set_as(total_num - mid_loop * self.inner_shape[-1])
                stride_copy.set_as((self.inner_total_num_list[-2] // self.split_dim * split_part_num) // 16)
                single_copy.set_as((self.inner_total_num_list[-2] // self.split_dim * split_part_num) % 16)
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(self.is_split_axi_reverse):
                    bias.set_as(total_num - (split_id + 1) * mid_loop * self.inner_shape[-1])
                with self.tik_instance.else_scope():
                    bias.set_as(split_id * mid_loop * self.inner_shape[-1])

            with self.tik_instance.if_scope(stride_copy > 0):
                with self.tik_instance.for_range(0, 16) as index:
                    if self.data_move_pad_support:
                        self.tik_instance.data_move_pad(self.input_data_ub[index * block_num * 16],
                                                        self.input_data[split_id * move_in_bias +
                                                                        outer_move_in * index_lenth +
                                                                        index * self.inner_shape[-1]],
                                                        stride_copy, self.inner_shape[-1] * self.inner_bytes_size,
                                                        block_num * 15, 
                                                        (self.inner_shape[-1] - block_num) * 32 +
                                                        (block_num * 16 - self.inner_shape[-1]) * self.inner_bytes_size)
                    else:
                        self.tik_instance.data_move(self.input_data_ub[index * block_num * 16],
                                                    self.input_data[split_id * move_in_bias +
                                                                    outer_move_in * index_lenth +
                                                                    index * self.inner_shape[-1]],
                                                    0, stride_copy, block_num,
                                                    self.inner_shape[-1] - block_num, block_num * 15)
            with self.tik_instance.for_range(0, single_copy) as index:
                self.copy_data(self.input_data_ub[index * block_num * 16 + stride_copy * 16 * block_num * 16],
                               self.input_data[split_id * move_in_bias + outer_move_in * index_lenth +
                                               index * self.inner_shape[-1] + stride_copy * 16 * self.inner_shape[-1]],
                               1, 
                               block_num, 
                               self.inner_shape[-1] * self.inner_bytes_size)

            current_index = self.tik_instance.Scalar(dtype="int32", name="current_index")
            move_out_index = self.tik_instance.Scalar(dtype="int32", name="move_out_index")
            with self.tik_instance.for_range(0, move_num) as index:
                self.axis_compute_with_scalar_array(
                    index, current_index, move_out_index,
                    self.inner_loop, self.inner_axis, self.inner_dividends,
                    self.inner_real_dims - 1)

                move_out_index = move_out_index - phase
                self.tik_instance.data_move(self.output_data_ub[move_out_index * block_num * 16],
                                            self.input_data_ub[index * block_num * 16], 0, 1, block_num, 0, 0)

            with self.tik_instance.if_scope(move_num % 16 != 0):
                with self.tik_instance.if_scope(move_num // 16 > 0):
                    with self.tik_instance.for_range(0, 16) as index:
                        self.tik_instance.data_move(
                            self.output_data[bias + outer_move_out * index_lenth + index * self.inner_shape[-1]],
                            self.output_data_ub[index * block_num * 16],
                            0, move_num // 16, block_num, block_num * 15, self.inner_shape[-1] - block_num)
                    self.tik_instance.data_move(self.output_data[bias + outer_move_out * index_lenth],
                                                self.output_data_ub, 0, move_num // 16, 1,
                                                block_num * 15 + block_num - 1,
                                                self.inner_shape[-1] - 1)

                with self.tik_instance.for_range(0, move_num % 16 - 1) as index:
                    self.tik_instance.data_move(
                        self.output_data[bias + outer_move_out * index_lenth +
                                         move_num // 16 * 16 * self.inner_shape[-1] + index * self.inner_shape[-1]],
                        self.output_data_ub[(move_num // 16 * 16 + index) * block_num * 16], 0, 1, block_num, 0, 0)
                self.tik_instance.data_move(
                    self.output_data[bias + outer_move_out * index_lenth +
                                     (move_num // 16 * 16 + move_num % 16 - 1) * self.inner_shape[-1]],
                    self.output_data_ub[(move_num // 16 * 16 + move_num % 16 - 1) * block_num * 16],
                    0, 1, block_num - 1, 0, 0)

            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(move_num // 16 > 0):
                    with self.tik_instance.for_range(0, 15) as index:
                        self.tik_instance.data_move(
                            self.output_data[bias + outer_move_out * index_lenth + index * self.inner_shape[-1]],
                            self.output_data_ub[index * block_num * 16],
                            0, move_num // 16, block_num, block_num * 15, self.inner_shape[-1] - block_num)
                    with self.tik_instance.if_scope(move_num // 16 - 1 > 0):
                        self.tik_instance.data_move(
                            self.output_data[bias + outer_move_out * index_lenth + 15 * self.inner_shape[-1]],
                            self.output_data_ub[15 * block_num * 16], 0, move_num // 16 - 1,
                            block_num, block_num * 15, self.inner_shape[-1] - block_num)
                    self.tik_instance.data_move(self.output_data[bias + outer_move_out * index_lenth],
                                                self.output_data_ub, 0, move_num // 16, 1,
                                                block_num * 15 + block_num - 1,
                                                self.inner_shape[-1] - 1)

                self.tik_instance.data_move(
                    self.output_data[bias + outer_move_out * index_lenth + self.inner_shape[-1] * (move_num - 1)],
                    self.output_data_ub[(move_num - 1) * block_num * 16], 0, 1, block_num - 1, 0, 0)

            with self.tik_instance.for_range(0, 16) as last_loop:
                self.input_data_ub[last_loop].set_as(self.output_data_ub[(move_num - 1) * block_num * 16 +
                                                                         self.inner_shape[-1] - 16 + last_loop])
            self.tik_instance.data_move(
                self.output_data[bias + outer_move_out * index_lenth + self.inner_shape[-1] * move_num - 16],
                self.input_data_ub, 0, 1, 1, 0, 0)

    def reverse_non_last_axis_16_aligned(self, outer_move_in=0, outer_move_out=0):
        """
        reverse_non_last_axis_16_aligned for tiling 1
        """
        self.input_data_ub = self.tik_instance.Tensor(self.inner_dtype, (self.avaliable_ub,),
                                                      name="input_data_ub",
                                                      scope=tik.scope_ubuf)
        self.output_data_ub = self.tik_instance.Tensor(self.inner_dtype, (self.avaliable_ub,),
                                                       name="output_data_ub",
                                                       scope=tik.scope_ubuf)

        phase = self.tik_instance.Scalar("int64", name="phase")
        with self.tik_instance.if_scope(self.is_split_axi_reverse != 0):
            phase.set_as(
                self.inner_total_num_list[-2] // self.split_dim *
                (self.split_dim - self.split_part_num))
        with self.tik_instance.else_scope():
            phase.set_as(0)
        total_num = self.inner_total_num_list[-1]

        move_in_bias = self.tik_instance.Scalar("int32", name="bias")
        move_in_bias.set_as(
            self.inner_total_num_list[-1] // self.split_dim * self.split_part_num)

        loop_times = self.tik_instance.Scalar("int32", name="loop_times")
        with self.tik_instance.if_scope(self.split_dim % self.split_part_num == 0):
            loop_times.set_as(self.split_dim // self.split_part_num)
        with self.tik_instance.else_scope():
            loop_times.set_as(self.split_dim // self.split_part_num + 1)

        block_num = self.tik_instance.Scalar("int32", name="block_num")
        with self.tik_instance.if_scope(self.inner_shape[-1] % 16 == 0):
            block_num.set_as(self.inner_shape[-1] // 16)
        with self.tik_instance.else_scope():
            block_num.set_as((self.inner_shape[-1] // 16 + 1))
        split_part_num = self.tik_instance.Scalar("int32", name="tmp_split_num")
        split_part_num.set_as(self.split_part_num)

        with self.tik_instance.for_range(0, loop_times) as split_id:
            with self.tik_instance.if_scope(split_id == loop_times - 1):
                split_part_num.set_as(self.split_dim - self.split_part_num * split_id)
                with self.tik_instance.if_scope(self.is_split_axi_reverse != 0):
                    phase.set_as(
                        self.inner_total_num_list[-2] // self.split_dim *
                        (self.split_dim - split_part_num))
                with self.tik_instance.else_scope():
                    phase.set_as(0)

            move_num = self.inner_total_num_list[-2] // self.split_dim * split_part_num
            mid_loop = self.inner_total_num_list[-2] // self.split_dim * split_part_num

            index_lenth = self.inner_total_num_list[-1]

            current_index = self.tik_instance.Scalar(dtype="int32", name="current_index")
            move_out_index = self.tik_instance.Scalar(dtype="int32", name="move_out_index")

            bias = self.tik_instance.Scalar("int32", name="bias")
            with self.tik_instance.if_scope(split_id == loop_times - 1):
                with self.tik_instance.if_scope(self.is_split_axi_reverse):
                    bias.set_as(0)
                with self.tik_instance.else_scope():
                    bias.set_as(total_num - mid_loop * self.inner_shape[-1])
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(self.is_split_axi_reverse):
                    bias.set_as(total_num - (split_id + 1) * mid_loop * self.inner_shape[-1])
                with self.tik_instance.else_scope():
                    bias.set_as(split_id * mid_loop * self.inner_shape[-1])

            self.tik_instance.data_move(self.input_data_ub,
                                        self.input_data[split_id * move_in_bias + outer_move_in * index_lenth], 0, 1,
                                        move_num * block_num, 0, 0)

            with self.tik_instance.for_range(0, move_num) as index:
                self.axis_compute_with_scalar_array(
                    index, current_index, move_out_index,
                    self.inner_loop, self.inner_axis, self.inner_dividends,
                    self.inner_real_dims - 1)
                move_out_index = move_out_index - phase
                self.tik_instance.data_move(self.output_data_ub[move_out_index * block_num * 16],
                                            self.input_data_ub[index * block_num * 16], 0, 1, block_num, 0, 0)

            self.tik_instance.data_move(self.output_data[bias + outer_move_out * index_lenth], self.output_data_ub, 0,
                                        1, move_num * block_num, 0, 0)

    def reverse_non_last_axis_small(self, outer_move_in=0, outer_move_out=0):
        """
        reverse_non_last_axis_small for tiling 0
        """
        input_data_ub = self.tik_instance.Tensor(self.inner_dtype, (self.avaliable_ub,),
                                                 name="input_data_ub",
                                                 scope=tik.scope_ubuf)
        output_data_ub = self.tik_instance.Tensor(self.inner_dtype, (self.avaliable_ub,),
                                                  name="output_data_ub",
                                                  scope=tik.scope_ubuf)

        phase = self.tik_instance.Scalar("int64", name="phase")
        with self.tik_instance.if_scope(self.is_split_axi_reverse != 0):
            phase.set_as(
                self.inner_total_num_list[-2] // self.split_dim * (self.split_dim - self.split_part_num))
        with self.tik_instance.else_scope():
            phase.set_as(0)

        total_num = self.inner_total_num_list[-1]

        move_in_bias = self.tik_instance.Scalar("int32", name="move_in_bias")
        move_in_bias.set_as(self.inner_total_num_list[-1] // self.split_dim * self.split_part_num)

        mat_num = self.tik_instance.Scalar("int32", name="mat_num")

        loop_times = self.tik_instance.Scalar("int32", name="loop_times")
        loop_times.set_as(ceil_div(self.split_dim, self.split_part_num))

        block_num = 1
        split_part_num = self.tik_instance.Scalar("int32", name="tmp_split_num")
        split_part_num.set_as(self.split_part_num)

        # for copy_in_info
        stride_copy = self.tik_instance.Scalar("int32", name="stride_copy")
        single_copy = self.tik_instance.Scalar("int32", name="single_copy")
        stride_copy.set_as((self.inner_total_num_list[-2] // self.split_dim * split_part_num) // 16)
        single_copy.set_as((self.inner_total_num_list[-2] // self.split_dim * split_part_num) % 16)
        with self.tik_instance.for_range(0, loop_times) as split_id:
            with self.tik_instance.if_scope(split_id == loop_times - 1):
                split_part_num.set_as(self.split_dim - self.split_part_num * split_id)
                with self.tik_instance.if_scope(self.is_split_axi_reverse != 0):
                    phase.set_as(
                        self.inner_total_num_list[-2] // self.split_dim *
                        (self.split_dim - split_part_num))
                with self.tik_instance.else_scope():
                    phase.set_as(0)
                stride_copy.set_as((self.inner_total_num_list[-2] // self.split_dim * split_part_num) // 16)
                single_copy.set_as((self.inner_total_num_list[-2] // self.split_dim * split_part_num) % 16)

            move_num = self.inner_total_num_list[-2] // self.split_dim * split_part_num
            mid_loop = self.inner_total_num_list[-2] // self.split_dim * split_part_num

            index_lenth = self.inner_total_num_list[-1]

            current_index = self.tik_instance.Scalar(dtype="int32", name="current_index")
            move_out_index = self.tik_instance.Scalar(dtype="int32", name="move_out_index")

            bias = self.tik_instance.Scalar("int64", name="bias")
            with self.tik_instance.if_scope(split_id == loop_times - 1):
                with self.tik_instance.if_scope(self.is_split_axi_reverse):
                    bias.set_as(0)
                with self.tik_instance.else_scope():
                    bias.set_as(total_num - mid_loop * self.inner_shape[-1])
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(self.is_split_axi_reverse):
                    bias.set_as(total_num - (split_id + 1) * mid_loop * self.inner_shape[-1])
                with self.tik_instance.else_scope():
                    bias.set_as(split_id * mid_loop * self.inner_shape[-1])

            with self.tik_instance.if_scope(stride_copy > 0):
                with self.tik_instance.for_range(0, 16) as index:
                    if self.data_move_pad_support:
                        self.tik_instance.data_move_pad(input_data_ub[index * block_num * 16],
                                                        self.input_data[split_id * move_in_bias +
                                                                        outer_move_in * index_lenth +
                                                                        index * self.inner_shape[-1]],
                                                        stride_copy, self.inner_shape[-1] * self.inner_bytes_size,
                                                        block_num * 15, 
                                                        (self.inner_shape[-1] - block_num) * 32 +
                                                        (block_num * 16 - self.inner_shape[-1]) * self.inner_bytes_size)
                    else:
                        self.tik_instance.data_move(input_data_ub[index * block_num * 16],
                                                    self.input_data[split_id * move_in_bias +
                                                                    outer_move_in * index_lenth +
                                                                    index * self.inner_shape[-1]],
                                                    0, stride_copy, block_num,
                                                    self.inner_shape[-1] - block_num, block_num * 15)
            with self.tik_instance.for_range(0, single_copy) as index:
                self.copy_data(input_data_ub[index * block_num * 16 + stride_copy * 16 * 16], 
                               self.input_data[split_id * move_in_bias + outer_move_in * index_lenth +
                                               index * self.inner_shape[-1] + stride_copy * 16 * self.inner_shape[-1]], 
                               1, 
                               block_num, 
                               self.inner_shape[-1] * self.inner_bytes_size)

            with self.tik_instance.for_range(0, move_num) as index:
                self.axis_compute_with_scalar_array(
                    index, current_index, move_out_index,
                    self.inner_loop, self.inner_axis, self.inner_dividends,
                    self.inner_real_dims - 1)
                move_out_index = move_out_index - phase
                self.tik_instance.data_move(output_data_ub[move_out_index * block_num * 16],
                                            input_data_ub[index * block_num * 16], 0, 1, block_num, 0, 0)

            mat_num.set_as((mid_loop + 255) // 256)

            src_list = [output_data_ub[i * 16 * 16 * block_num * mat_num] for i in range(16)]
            dst_list = [input_data_ub[i * 16] for i in range(16)]
            self.tik_instance.vnchwconv(True, False, dst_list, src_list, 16 * block_num * mat_num, 16, 1)

            with self.tik_instance.if_scope(mat_num > 0):
                self.tik_instance.data_move(output_data_ub, input_data_ub, 0, 16 * mat_num, self.inner_shape[-1],
                                            16 - self.inner_shape[-1], 0)

            src_list = [output_data_ub[i * 16] for i in range(16)]
            dst_list = [input_data_ub[i * 16 * self.inner_shape[-1] * mat_num] for i in range(16)]
            self.tik_instance.vnchwconv(True, False, dst_list, src_list, self.inner_shape[-1] * mat_num, 1, 16)

            with self.tik_instance.if_scope(self.inner_total_num_list[-1] > 16):
                move_out_num = mid_loop * self.inner_shape[-1]
                with self.tik_instance.if_scope(move_out_num % 16 == 0):
                    self.tik_instance.data_move(self.output_data[outer_move_out * total_num + bias],
                                                input_data_ub, 0, 1,
                                                move_out_num // 16, 0, 0)
                with self.tik_instance.else_scope():
                    if self.data_move_pad_support:
                        self.tik_instance.data_move_pad(self.output_data[outer_move_out * total_num + bias],
                                                        input_data_ub, 1, 2 * move_out_num, 0, 0)
                    else:
                        self.tik_instance.data_move(self.output_data[outer_move_out * total_num + bias],
                                                    input_data_ub, 0, 1,
                                                    move_out_num // 16, 0, 0)
                        with self.tik_instance.for_range(0, 16) as refill_id:
                            input_data_ub[refill_id].set_as(input_data_ub[move_out_num - 16 + refill_id])

                        self.tik_instance.data_move(self.output_data[outer_move_out * total_num
                                                                     + bias + move_out_num - 16],
                                                    input_data_ub, 0, 1, 1, 0, 0)
            with self.tik_instance.else_scope():
                if self.data_move_pad_support:
                    self.tik_instance.data_move_pad(self.output_data[outer_move_out * total_num + bias], input_data_ub,
                                                    1, 2 * total_num, 0, 0)
                else:
                    self.tik_instance.data_move(self.output_data[outer_move_out * total_num + bias], input_data_ub,
                                                0, 1, 1, 0, 0)

    def tiling_reverse_non_last_axis_large_compute(self):
        """
        tiling_reverse_non_last_axis_large_compute
        base last dim size, run diff staged
        """
        staged_list = (4, 2, 1)
        for i, current_staged in enumerate(staged_list):
            if i == 0:
                pre_staged = self.inner_shape[-1] + 1
            else:
                pre_staged = staged_list[i - 1] * 3200 * 2
            min_staged = 3200 * current_staged * 2
            if i == len(staged_list) - 1:
                min_staged = 0
            # outer loop
            with self.tik_instance.if_scope(tik.all(self.inner_shape[-1] >= min_staged,
                                                    self.inner_shape[-1] < pre_staged)):
                with self.tik_instance.for_range(self.core_outer_start,
                                                 self.core_outer_num + self.core_outer_start) as index:
                    current_index = self.tik_instance.Scalar(dtype="int64", name="current_index")
                    move_out_index = self.tik_instance.Scalar(dtype="int64", name="move_out_index")
                    self.axis_compute_with_scalar_array(
                        index, current_index, move_out_index, self.outer_shape, self.outer_axis,
                        self.outer_dividends, self.outer_real_dims)
                    # inner compute
                    self.reverse_non_last_axis_large(index, move_out_index, 3200 * current_staged)

    def reverse_non_last_axis_large(self, outer_move_in=0, outer_move_out=0, process_num=3200):
        """
        reverse_non_last_axis_large for tiling 3
        """
        self.process_num = process_num
        offset_after_first_copy = self.tik_instance.Scalar("int32",
                                                           name="offset_after_first_copy",
                                                           init_value=self.block_num)
        with self.tik_instance.if_scope(self.inner_shape[-1] % self.block_num != 0):
            offset_after_first_copy.set_as(self.inner_shape[-1] % self.block_num)

        data_ub_one_block = self.tik_instance.Tensor(self.inner_dtype, (self.block_num,),
                                                     name="data_ub_one_block",
                                                     scope=tik.scope_ubuf)
        self.tik_instance.data_move(data_ub_one_block, self.input_data[outer_move_in * self.inner_shape[-1]], 0, 1, 1,
                                    0, 0)
        self.tik_instance.data_move(self.output_data[outer_move_out * self.inner_shape[-1]], data_ub_one_block, 0, 1,
                                    1, 0, 0)
        new_copy_num = ((self.inner_shape[-1] - 1) // self.block_num) * self.block_num
        copy_loop = new_copy_num // self.process_num
        copy_tail = new_copy_num % self.process_num

        src_origin_offset = outer_move_in * self.inner_shape[-1] + offset_after_first_copy
        dst_origin_offset = outer_move_out * self.inner_shape[-1] + offset_after_first_copy
        with self.tik_instance.for_range(0, copy_loop, thread_num=2) as index:
            data_ub = self.tik_instance.Tensor("int16", (self.process_num,), name="data_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(data_ub, self.input_data[src_origin_offset + index * self.process_num], 0, 1,
                                        self.process_num // self.block_num, 0, 0)
            self.tik_instance.data_move(self.output_data[dst_origin_offset + index * self.process_num], data_ub, 0, 1,
                                        self.process_num // self.block_num, 0, 0)
        with self.tik_instance.if_scope(copy_tail != 0):
            data_ub_tail = self.tik_instance.Tensor("int16", (self.process_num,),
                                                    name="data_ub_tail",
                                                    scope=tik.scope_ubuf)
            self.tik_instance.data_move(data_ub_tail,
                                        self.input_data[src_origin_offset + copy_loop * self.process_num], 0, 1,
                                        copy_tail // self.block_num, 0, 0)
            self.tik_instance.data_move(self.output_data[dst_origin_offset + copy_loop * self.process_num],
                                        data_ub_tail, 0, 1, copy_tail // self.block_num, 0, 0)

    def reverse_last_axis_large(self, outer_move_in=0, outer_move_out=0):
        """
        reverse_last_axis_large for tiling 6
        """
        self.process_num = 10240
        offset_after_first_copy = self.tik_instance.Scalar("int32",
                                                           name="offset_after_first_copy",
                                                           init_value=self.block_num)
        with self.tik_instance.if_scope(self.inner_shape[-1] % (self.block_num * 16) != 0):
            offset_after_first_copy.set_as(self.inner_shape[-1] % (self.block_num * 16))
        with self.tik_instance.else_scope():
            offset_after_first_copy.set_as(256)

        data_ub_16_block_a = self.tik_instance.Tensor(self.inner_dtype, (self.block_num * 16,),
                                                      name="data_ub_16_block_a",
                                                      scope=tik.scope_ubuf)

        data_ub_16_block_b = self.tik_instance.Tensor(self.inner_dtype, (self.block_num * 16,),
                                                      name="data_ub_16_block_b",
                                                      scope=tik.scope_ubuf)
        self.tik_instance.data_move(data_ub_16_block_a,
                                    self.input_data[(outer_move_in + 1) * self.inner_shape[-1] - 256], 0, 1, 16, 0, 0)

        src_list = [data_ub_16_block_a[i * 16] for i in range(16)]
        dst_list = [data_ub_16_block_b[i * 16] for i in range(15, -1, -1)]
        self.tik_instance.vnchwconv(True, False, dst_list, src_list, 1, 0, 0)

        src_list = [data_ub_16_block_b[i * 16] for i in range(16)]
        dst_list = [data_ub_16_block_a[i * 16] for i in range(15, -1, -1)]
        self.tik_instance.vnchwconv(True, False, dst_list, src_list, 1, 0, 0)

        self.tik_instance.data_move(self.output_data[outer_move_out * self.inner_shape[-1]], data_ub_16_block_a, 0, 1,
                                    16, 0, 0)

        new_copy_num = ((self.inner_shape[-1] - 1) // (self.block_num * 16)) * self.block_num * 16

        copy_loop = new_copy_num // self.process_num
        copy_tail = self.tik_instance.Scalar("int32", name="copy_tail")
        with self.tik_instance.if_scope(new_copy_num < self.process_num):
            copy_tail.set_as(new_copy_num)
        with self.tik_instance.else_scope():
            copy_tail.set_as(new_copy_num % self.process_num)

        with self.tik_instance.for_range(0, copy_loop, thread_num=2) as index:
            data_ub_a = self.tik_instance.Tensor(self.inner_dtype, (self.process_num,),
                                                 name="data_ub_a", scope=tik.scope_ubuf)
            data_ub_b = self.tik_instance.Tensor(self.inner_dtype, (self.process_num,),
                                                 name="data_ub_b", scope=tik.scope_ubuf)
            self.tik_instance.data_move(
                data_ub_a, self.input_data[(outer_move_in + 1) * self.inner_shape[-1] -
                                           (index + 1) * self.process_num - offset_after_first_copy], 0, 1,
                self.process_num // self.block_num, 0, 0)

            src_list = [data_ub_a[i * 16] for i in range(16)]
            dst_list = [data_ub_b[i * 16] for i in range(15, -1, -1)]
            self.tik_instance.vnchwconv(True, False, dst_list, src_list, self.process_num // 256, 16, 16)

            src_list = [data_ub_b[i * 16] for i in range(16)]
            dst_list = [data_ub_a[i * 16] for i in range(15, -1, -1)]
            self.tik_instance.vnchwconv(True, False, dst_list, src_list, self.process_num // 256, 16, 16)

            with self.tik_instance.for_range(0, self.process_num // 256) as reorder_index:
                self.tik_instance.data_move(data_ub_b[(self.process_num // 256 - reorder_index - 1) * 256],
                                            data_ub_a[reorder_index * 256], 0, 1, 16, 0, 0)

            self.tik_instance.data_move(
                self.output_data[outer_move_out * self.inner_shape[-1] + offset_after_first_copy +
                                 index * self.process_num], data_ub_b, 0, 1, self.process_num // self.block_num, 0, 0)

        with self.tik_instance.if_scope(copy_tail != 0):
            copy_tail_vnchw_stride = self.tik_instance.Scalar("int32", name="copy_tail_vnchw_stride", init_value=16)
            with self.tik_instance.if_scope(copy_tail // Constant.VNHW_MIN_NUM == 1):
                copy_tail_vnchw_stride.set_as(0)
            data_ub_tail_a = self.tik_instance.Tensor(self.inner_dtype, (self.process_num,),
                                                      name="data_ub_tail_a",
                                                      scope=tik.scope_ubuf)

            data_ub_tail_b = self.tik_instance.Tensor(self.inner_dtype, (self.process_num,),
                                                      name="data_ub_tail_b",
                                                      scope=tik.scope_ubuf)

            self.tik_instance.data_move(data_ub_tail_a, self.input_data[outer_move_in * self.inner_shape[-1]], 0, 1,
                                        copy_tail // self.block_num, 0, 0)

            src_list = [data_ub_tail_a[i * 16] for i in range(16)]
            dst_list = [data_ub_tail_b[i * 16] for i in range(15, -1, -1)]
            self.tik_instance.vnchwconv(True, False, dst_list, src_list, copy_tail // Constant.VNHW_MIN_NUM,
                                        copy_tail_vnchw_stride, copy_tail_vnchw_stride)

            src_list = [data_ub_tail_b[i * 16] for i in range(16)]
            dst_list = [data_ub_tail_a[i * 16] for i in range(15, -1, -1)]
            self.tik_instance.vnchwconv(True, False, dst_list, src_list, copy_tail // Constant.VNHW_MIN_NUM,
                                        copy_tail_vnchw_stride, copy_tail_vnchw_stride)

            with self.tik_instance.for_range(0, copy_tail // Constant.VNHW_MIN_NUM) as reorder_index:
                self.tik_instance.data_move(
                    data_ub_tail_b[(copy_tail // Constant.VNHW_MIN_NUM - 1 - reorder_index) * 256],
                    data_ub_tail_a[reorder_index * Constant.VNHW_MIN_NUM], 0, 1, 16, 0, 0)

            self.tik_instance.data_move(self.output_data[(outer_move_out + 1) * self.inner_shape[-1] - copy_tail],
                                        data_ub_tail_b, 0, 1, copy_tail // self.block_num, 0, 0)

    def large_reverse(self, outer_move_in=0, outer_move_out=0):
        """
        large_reverse for tiling 5
        """
        self.input_data_ub = self.tik_instance.Tensor(self.inner_dtype, (self.avaliable_ub,),
                                                      name="input_data_ub",
                                                      scope=tik.scope_ubuf)
        self.output_data_ub = self.tik_instance.Tensor(self.inner_dtype, (self.avaliable_ub,),
                                                       name="output_data_ub",
                                                       scope=tik.scope_ubuf)

        phase = self.tik_instance.Scalar("int64", name="phase")

        phase.set_as(
            self.inner_total_num_list[-3] // self.split_dim *
            (self.split_dim - self.split_part_num))
        total_num = self.inner_total_num_list[-1]

        move_in_bias = self.tik_instance.Scalar("int32", name="bias")
        move_in_bias.set_as(
            self.inner_total_num_list[-1] // self.split_dim * self.split_part_num)

        loop_times = self.tik_instance.Scalar("int32", name="loop_times")
        loop_times.set_as(ceil_div(self.split_dim, self.split_part_num))

        block_num = self.tik_instance.Scalar("int32", name="block_num")
        with self.tik_instance.if_scope(self.inner_shape[-1] % 256 == 0):
            block_num.set_as(self.inner_shape[-1] // 256 * 16)
        with self.tik_instance.else_scope():
            block_num.set_as((self.inner_shape[-1] // 256 + 1) * 16)

        split_part_num = self.tik_instance.Scalar("int32", name="tmp_split_num")
        split_part_num.set_as(self.split_part_num)
        # for copy_in_info
        stride_copy = self.tik_instance.Scalar("int32", name="stride_copy")
        single_copy = self.tik_instance.Scalar("int32", name="single_copy")
        stride_copy.set_as((self.inner_total_num_list[-2] // self.split_dim * split_part_num) // 16)
        single_copy.set_as((self.inner_total_num_list[-2] // self.split_dim * split_part_num) % 16)

        move_num = self.tik_instance.Scalar("int32", name="move_num")
        inner_shape_2 = self.tik_instance.Scalar("int32", name="inner_shape_2")
        inner_shape_2.set_as(self.inner_shape[-2])
        move_num.set_as(self.inner_total_num_list[-3] // self.split_dim * split_part_num)
        with self.tik_instance.if_scope(move_num == 0):
            move_num.set_as(1)
            inner_shape_2.set_as(split_part_num)

        # first vnchw repeat stride
        first_vnchw_repeat_stride = self.tik_instance.Scalar("int32", name="first_vnchw_repeat_stride", init_value=16)
        first_vnchw_repeat = self.tik_instance.Scalar("int32", name="first_vnchw_repeat")
        first_vnchw_repeat.set_as(self.inner_total_num_list[-2] // self.split_dim * split_part_num * block_num // 16)
        with self.tik_instance.if_scope(first_vnchw_repeat == 1):
            first_vnchw_repeat_stride.set_as(0)

        with self.tik_instance.for_range(0, loop_times) as split_id:
            with self.tik_instance.if_scope(split_id == loop_times - 1):
                split_part_num.set_as(self.split_dim - self.split_part_num * split_id)
                phase.set_as(
                    self.inner_total_num_list[-3] // self.split_dim *
                    (self.split_dim - split_part_num))
                stride_copy.set_as((self.inner_total_num_list[-2] // self.split_dim * split_part_num) // 16)
                single_copy.set_as((self.inner_total_num_list[-2] // self.split_dim * split_part_num) % 16)
                first_vnchw_repeat.set_as(
                    self.inner_total_num_list[-2] // self.split_dim * split_part_num * block_num // 16)
                with self.tik_instance.if_scope(first_vnchw_repeat == 1):
                    first_vnchw_repeat_stride.set_as(0)
                move_num.set_as(self.inner_total_num_list[-3] // self.split_dim * split_part_num)
                with self.tik_instance.if_scope(move_num == 0):
                    move_num.set_as(1)
                    inner_shape_2.set_as(split_part_num)

            mid_loop = self.inner_total_num_list[-2] // self.split_dim * split_part_num

            index_lenth = self.inner_total_num_list[-1]

            reorder_pieces = block_num // 16

            current_index = self.tik_instance.Scalar(dtype="int32", name="current_index")
            move_out_index = self.tik_instance.Scalar(dtype="int32", name="move_out_index")
            copy_block_num = (self.inner_shape[-1] + 15) / 16
            with self.tik_instance.if_scope(stride_copy > 0):
                with self.tik_instance.for_range(0, 16) as index:
                    if self.data_move_pad_support:
                        self.tik_instance.data_move_pad(self.input_data_ub[index * block_num * 16],
                                                        self.input_data[split_id * move_in_bias +
                                                                        outer_move_in * index_lenth +
                                                                        index * self.inner_shape[-1]],
                                                        stride_copy, self.inner_shape[-1] * self.inner_bytes_size,
                                                        block_num * 16 - copy_block_num, 
                                                        (self.inner_shape[-1] - copy_block_num) * 32 +
                                                        (copy_block_num * 16 - self.inner_shape[-1]) *
                                                        self.inner_bytes_size)
                    else:
                        self.tik_instance.data_move(self.input_data_ub[index * block_num * 16],
                                                    self.input_data[split_id * move_in_bias +
                                                                    outer_move_in * index_lenth +
                                                                    index * self.inner_shape[-1]],
                                                    0, stride_copy, copy_block_num,
                                                    self.inner_shape[-1] - copy_block_num,
                                                    block_num * 16 - copy_block_num)
            with self.tik_instance.for_range(0, single_copy) as index:
                self.copy_data(self.input_data_ub[index * block_num * 16 + stride_copy * 16 * block_num * 16],
                               self.input_data[split_id * move_in_bias + outer_move_in * index_lenth +
                                               index * self.inner_shape[-1] + stride_copy * 16 * self.inner_shape[-1]],
                               1, 
                               copy_block_num, 
                               self.inner_shape[-1] * self.inner_bytes_size)

            with self.tik_instance.for_range(0, move_num) as index:
                self.axis_compute_with_scalar_array(
                    index, current_index, move_out_index,
                    self.inner_loop, self.inner_axis, self.inner_dividends,
                    self.inner_real_dims - 2)

                move_out_index = move_out_index - phase
                with self.tik_instance.for_range(0, reorder_pieces) as block_index:
                    with self.tik_instance.if_scope(inner_shape_2 > 0):
                        self.tik_instance.data_move(
                            self.output_data_ub[move_out_index * inner_shape_2 * block_num * 16 +
                                                block_index * 256],
                            self.input_data_ub[index * inner_shape_2 * block_num * 16 +
                                            (reorder_pieces - 1 - block_index) * 256],
                            0, inner_shape_2, 16,
                            (reorder_pieces - 1) * 16, (reorder_pieces - 1) * 16)

            src_list = [self.output_data_ub[i * 16] for i in range(16)]
            dst_list = [self.input_data_ub[i * 16] for i in range(15, -1, -1)]

            self.tik_instance.vnchwconv(False, False, dst_list, src_list,
                                        first_vnchw_repeat, first_vnchw_repeat_stride, first_vnchw_repeat_stride)

            src_list = [self.input_data_ub[i * 16] for i in range(16)]
            dst_list = [self.output_data_ub[i * 16] for i in range(15, -1, -1)]

            self.tik_instance.vnchwconv(False, False, dst_list, src_list,
                                        first_vnchw_repeat, first_vnchw_repeat_stride, first_vnchw_repeat_stride)

            bias = self.tik_instance.Scalar("int32", name="bias")
            with self.tik_instance.if_scope(split_id == loop_times - 1):
                with self.tik_instance.if_scope(self.is_split_axi_reverse):
                    bias.set_as(0)
                with self.tik_instance.else_scope():
                    bias.set_as(total_num - mid_loop * self.inner_shape[-1])
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(self.is_split_axi_reverse):
                    bias.set_as(total_num - (split_id + 1) * mid_loop * self.inner_shape[-1])
                with self.tik_instance.else_scope():
                    bias.set_as(split_id * mid_loop * self.inner_shape[-1])

            with self.tik_instance.if_scope(self.inner_shape[-1] % 256 == 0):
                self.tik_instance.data_move(self.output_data[bias + outer_move_out * index_lenth],
                                            self.output_data_ub,
                                            0, 1, mid_loop * self.inner_shape[-1] // 16, 0, 0)

            with self.tik_instance.else_scope():
                width = self.tik_instance.Scalar(dtype="int32", name="width")
                width.set_as((mid_loop + 15) // 16 * block_num)

                src_list = [self.output_data_ub[i * width * 16] for i in range(16)]
                dst_list = [self.input_data_ub[i * 16] for i in range(16)]
                self.tik_instance.vnchwconv(False, False, dst_list, src_list, width, 16, 1)

                redundant_part = block_num * 16 - self.inner_shape[-1]
                with self.tik_instance.if_scope(width // block_num > 0):
                    self.tik_instance.data_move(self.output_data_ub, self.input_data_ub[redundant_part * 16], 0,
                                                width // block_num, block_num * 16 - redundant_part, redundant_part, 0)

                adjusted_width = ((width // block_num) * self.inner_shape[-1] // 256 + 1) * 16
                src_list = [self.output_data_ub[i * 16] for i in range(16)]
                dst_list = [self.input_data_ub[i * adjusted_width * 16] for i in range(16)]

                self.tik_instance.vnchwconv(False, False, dst_list, src_list, adjusted_width, 1, 16)

                cols = self.tik_instance.Scalar("int32", name="cols")
                with self.tik_instance.if_scope(mid_loop % 16 == 0):
                    cols.set_as(mid_loop // 16)
                with self.tik_instance.else_scope():
                    cols.set_as(mid_loop // 16 + 1)

                loop = self.tik_instance.Scalar("int32", name="loop")
                with self.tik_instance.if_scope(mid_loop % cols == 0):
                    loop.set_as(mid_loop // cols)
                with self.tik_instance.else_scope():
                    loop.set_as(mid_loop // cols + 1)

                last_row_valid_data = mid_loop - (loop - 1) * cols

                with self.tik_instance.for_range(0, loop - 1) as index:
                    is_overlap = ((index * cols * self.inner_shape[-1] +
                                  adjusted_width * self.block_num) <= mid_loop * self.inner_shape[-1])
                    with self.tik_instance.if_scope(is_overlap):
                        self.tik_instance.data_move(
                            self.output_data[bias + index * cols * self.inner_shape[-1] +
                                             outer_move_out * index_lenth],
                            self.input_data_ub[index * adjusted_width * 16], 0, 1, adjusted_width, 0, 0)
                    with self.tik_instance.else_scope():
                        tail_block = (cols * self.inner_shape[-1] + self.block_num - 1) // self.block_num
                        self.tik_instance.data_move(
                            self.output_data[bias + index * cols * self.inner_shape[-1] +
                                             outer_move_out * index_lenth],
                            self.input_data_ub[index * adjusted_width * 16], 0, 1, tail_block, 0, 0)
                self.tik_instance.data_move(
                    self.output_data[bias + (loop - 1) * cols * self.inner_shape[-1] + outer_move_out * index_lenth],
                    self.input_data_ub[(loop - 1) * adjusted_width * 16], 0, 1,
                    last_row_valid_data * self.inner_shape[-1] // 16, 0, 0)
                move_out_num = adjusted_width * 16 * (loop - 1) + last_row_valid_data * self.inner_shape[-1]
                with self.tik_instance.for_range(0, 16) as refill_id:
                    self.input_data_ub[refill_id].set_as(self.input_data_ub[move_out_num - 16 + refill_id])
                self.tik_instance.data_move(
                    self.output_data[outer_move_out * index_lenth + bias + mid_loop * self.inner_shape[-1] - 16],
                    self.input_data_ub, 0, 1, 1, 0, 0)

    def reverse_small_shape(self, outer_move_in=0, outer_move_out=0):
        """
        reverse_small_shape for tiling 4
        """
        self.input_data_ub = self.tik_instance.Tensor(self.inner_dtype, (self.avaliable_ub,),
                                                      name="input_data_ub",
                                                      scope=tik.scope_ubuf)
        self.output_data_ub = self.tik_instance.Tensor(self.inner_dtype, (self.avaliable_ub,),
                                                       name="output_data_ub",
                                                       scope=tik.scope_ubuf)
        phase = self.tik_instance.Scalar("int64", name="phase")
        with self.tik_instance.if_scope(self.is_split_axi_reverse != 0):
            phase.set_as(
                self.inner_total_num_list[-3] // self.split_dim *
                (self.split_dim - self.split_part_num))
        with self.tik_instance.else_scope():
            phase.set_as(0)
        total_num = self.inner_total_num_list[-1]

        move_in_bias = self.tik_instance.Scalar("int32", name="bias")
        move_in_bias.set_as(
            self.inner_total_num_list[-1] // self.split_dim * self.split_part_num)

        loop_times = self.tik_instance.Scalar("int32", name="loop_times")
        with self.tik_instance.if_scope(self.split_dim % self.split_part_num == 0):
            loop_times.set_as(self.split_dim // self.split_part_num)
        with self.tik_instance.else_scope():
            loop_times.set_as(self.split_dim // self.split_part_num + 1)

        mat_num = self.tik_instance.Scalar("int64", name="mat_num")

        block_num = self.tik_instance.Scalar("int32", name="block_num")
        block_num.set_as((self.inner_shape[-1] + 15) // 16)

        split_part_num = self.tik_instance.Scalar("int32", name="tmp_split_num")
        split_part_num.set_as(self.split_part_num)

        # for copy_in_info
        stride_copy = self.tik_instance.Scalar("int32", name="stride_copy")
        single_copy = self.tik_instance.Scalar("int32", name="single_copy")
        stride_copy.set_as((self.inner_total_num_list[-2] // self.split_dim * split_part_num) // 16)
        single_copy.set_as((self.inner_total_num_list[-2] // self.split_dim * split_part_num) % 16)

        move_num = self.tik_instance.Scalar("int32", name="move_num")
        inner_shape_2 = self.tik_instance.Scalar("int32", name="inner_shape_2")
        inner_shape_2.set_as(self.inner_shape[-2])
        move_num.set_as(self.inner_total_num_list[-3] // self.split_dim * split_part_num)
        with self.tik_instance.if_scope(move_num == 0):
            move_num.set_as(1)
            inner_shape_2.set_as(split_part_num)

        with self.tik_instance.for_range(0, loop_times) as split_id:
            mid_loop = self.inner_total_num_list[-2] // self.split_dim * split_part_num

            index_lenth = self.inner_total_num_list[-1]

            current_index = self.tik_instance.Scalar(dtype="int32", name="current_index")
            move_out_index = self.tik_instance.Scalar(dtype="int32", name="move_out_index")
            bias = self.tik_instance.Scalar("int32", name="bias")
            bias.set_as(split_id * mid_loop * self.inner_shape[-1])
            with self.tik_instance.if_scope(split_id == loop_times - 1):
                split_part_num.set_as(self.split_dim - self.split_part_num * split_id)
                with self.tik_instance.if_scope(self.is_split_axi_reverse != 0):
                    phase.set_as(
                        self.inner_total_num_list[-3] // self.split_dim *
                        (self.split_dim - split_part_num))
                with self.tik_instance.else_scope():
                    phase.set_as(0)
                with self.tik_instance.if_scope(self.is_split_axi_reverse):
                    bias.set_as(0)
                stride_copy.set_as((self.inner_total_num_list[-2] // self.split_dim * split_part_num) // 16)
                single_copy.set_as((self.inner_total_num_list[-2] // self.split_dim * split_part_num) % 16)
                with self.tik_instance.if_scope(move_num == 0):
                    move_num.set_as(1)
                    inner_shape_2.set_as(split_part_num)
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(self.is_split_axi_reverse):
                    bias.set_as(total_num - (split_id + 1) * mid_loop * self.inner_shape[-1])

            with self.tik_instance.if_scope(stride_copy > 0):
                with self.tik_instance.for_range(0, 16) as index:
                    if self.data_move_pad_support:
                        self.tik_instance.data_move_pad(self.input_data_ub[index * block_num * 16],
                                                        self.input_data[split_id * move_in_bias +
                                                                        outer_move_in * index_lenth +
                                                                        index * self.inner_shape[-1]],
                                                        stride_copy, self.inner_shape[-1] * self.inner_bytes_size,
                                                        block_num * 15, 
                                                        (self.inner_shape[-1] - block_num) * 32 +
                                                        (block_num * 16 - self.inner_shape[-1]) * self.inner_bytes_size)
                    else:
                        self.tik_instance.data_move(self.input_data_ub[index * block_num * 16],
                                                    self.input_data[split_id * move_in_bias +
                                                                    outer_move_in * index_lenth +
                                                                    index * self.inner_shape[-1]],
                                                    0, stride_copy, block_num,
                                                    self.inner_shape[-1] - block_num, block_num * 15)
            with self.tik_instance.for_range(0, single_copy) as index:
                self.copy_data(self.input_data_ub[index * block_num * 16 + stride_copy * 16 * block_num * 16],
                              self.input_data[split_id * move_in_bias + outer_move_in * index_lenth +
                                              index * self.inner_shape[-1] + stride_copy * 16 * self.inner_shape[-1]],
                              1, 
                              block_num, 
                              self.inner_shape[-1] * self.inner_bytes_size)

            with self.tik_instance.for_range(0, move_num) as index:
                self.axis_compute_with_scalar_array(
                    index, current_index, move_out_index,
                    self.inner_loop, self.inner_axis, self.inner_dividends,
                    self.inner_real_dims - 2)

                move_out_index = move_out_index - phase
                with self.tik_instance.if_scope(inner_shape_2 > 0):
                    self.tik_instance.data_move(
                        self.output_data_ub[move_out_index * inner_shape_2 * block_num * 16],
                        self.input_data_ub[index * inner_shape_2 * block_num * 16], 0,
                        inner_shape_2 * block_num, 1, 0, 0)

            mat_num.set_as((mid_loop + 255) // 256)

            src_list = [self.output_data_ub[i * 16 * 16 * block_num * mat_num] for i in range(16)]
            dst_list = [self.input_data_ub[i * 16] for i in range(15, -1, -1)]

            last_block_num = self.tik_instance.Scalar("int32", name="block_num")
            last_block_num.set_as(self.inner_shape[-1] - self.inner_shape[-1] // 16 * 16)
            with self.tik_instance.if_scope(self.inner_shape[-1] % 16 == 0):
                last_block_num.set_as(16)

            self.tik_instance.vnchwconv(True, False, dst_list, src_list, 16 * block_num * mat_num, 16, 1)
            with self.tik_instance.if_scope(mat_num > 0):
                with self.tik_instance.for_range(0, block_num - 1) as data_move_index:
                    self.tik_instance.data_move(
                        self.output_data_ub[(self.inner_shape[-1] - data_move_index * 16 - 16) * 16],
                        self.input_data_ub[data_move_index * 16 * 16],
                        0, 16 * mat_num, 16, 16 * (block_num - 1), last_block_num + (block_num - 2) * 16)
                self.tik_instance.data_move(self.output_data_ub,
                                            self.input_data_ub[block_num * 16 * 16 - last_block_num * 16],
                                            0, 16 * mat_num,
                                            last_block_num, 16 - last_block_num + (block_num - 1) * 16,
                                            16 * (block_num - 1))

            src_list = [self.output_data_ub[i * 16] for i in range(16)]
            dst_list = [self.input_data_ub[i * 16 * self.inner_shape[-1] * mat_num] for i in range(16)]
            self.tik_instance.vnchwconv(True, False, dst_list, src_list, self.inner_shape[-1] * mat_num, 1, 16)

            move_out_num = mid_loop * self.inner_shape[-1]
            with self.tik_instance.if_scope(move_out_num >= 16):
                self.tik_instance.data_move(self.output_data[outer_move_out * total_num + bias],
                                            self.input_data_ub, 0, 1,
                                            move_out_num // 16, 0, 0)
                with self.tik_instance.if_scope(move_out_num % 16 != 0):
                    with self.tik_instance.for_range(0, 16) as refill_id:
                        self.input_data_ub[refill_id].set_as(self.input_data_ub[move_out_num - 16 + refill_id])

                    self.tik_instance.data_move(self.output_data[outer_move_out * total_num
                                                                 + bias + move_out_num - 16],
                                                self.input_data_ub, 0, 1, 1, 0, 0)
            with self.tik_instance.else_scope():
                self.copy_data(self.output_data[outer_move_out * total_num + bias],
                               self.input_data_ub,
                               1, 
                               (move_out_num + 15) // 16, 
                               move_out_num * self.inner_bytes_size)

        return self.tik_instance

    def gen_assist_data(self, assist_ub, max_topk_size=1024):
        """
        gen_assist_data
        """
        with self.tik_instance.new_stmt_scope():
            inner_loop_tmp = self.tik_instance.ScalarArray(dtype="int64", length=7, name="inner_loop_tmp")
            with self.tik_instance.for_range(0, self.inner_real_dims) as real_idx:
                inner_loop_tmp[real_idx].set_as(self.inner_loop[real_idx])
            inner_loop_tmp[0].set_as(max_topk_size // self.inner_dividends[0])

            assist_ub_copy = self.tik_instance.Tensor(assist_ub.dtype, (max_topk_size,),
                                                      name="assist_ub_copy", scope=tik.scope_ubuf)
            self.tik_instance.data_move(assist_ub_copy, self.assist_num_gm, 0, 1, ceil_div(max_topk_size, 16), 0, 0)

            # self.inner_dividends to ub int32
            inner_dividends_ub_int = self.tik_instance.Tensor("int32", (self.block_num * self.block_num,),
                                                              name="inner_dividends_ub_int", scope=tik.scope_ubuf)
            inner_dividends_ub_fp16 = self.tik_instance.Tensor(assist_ub.dtype, (self.block_num * self.block_num,),
                                                               name="inner_dividends_ub_int", scope=tik.scope_ubuf)
            inner_shape_ub_int = self.tik_instance.Tensor("int32", (self.block_num * self.block_num,),
                                                          name="inner_shape_ub_int", scope=tik.scope_ubuf)

            # init inner_dividends_ub_int to value 0
            util_tik_comm_func.tik_func_vector(self.tik_instance, inner_dividends_ub_int, 0,
                                               self.block_num * self.block_num)

            with self.tik_instance.for_range(0, self.inner_real_dims) as real_idx:
                gen_assist_scalar_32 = self.tik_instance.Scalar("int32", name="gen_assist_scalar_32")
                gen_assist_scalar_32.set_as(inner_loop_tmp[real_idx] - 1)
                util_tik_comm_func.tik_func_vector(self.tik_instance, inner_shape_ub_int[real_idx * self.block_num],
                                                   gen_assist_scalar_32, self.block_num)
                gen_assist_scalar_32.set_as(self.inner_dividends[real_idx])
                util_tik_comm_func.tik_func_vector(self.tik_instance,
                                                   inner_dividends_ub_int[real_idx * self.block_num],
                                                   gen_assist_scalar_32, self.block_num)

            util_tik_comm_func.tik_func_vconv(self.tik_instance, inner_dividends_ub_fp16, inner_dividends_ub_int,
                                              self.block_num * self.block_num)
            assist_ub_tmp1 = self.tik_instance.Tensor(assist_ub.dtype, (max_topk_size,),
                                                      name="assist_ub_tmp1", scope=tik.scope_ubuf)
            assist_ub_int32 = self.tik_instance.Tensor("int32", (max_topk_size,),
                                                       name="assist_ub_int32", scope=tik.scope_ubuf)
            assist_ub_int32_tmp1 = self.tik_instance.Tensor("int32", (max_topk_size,),
                                                            name="assist_ub_int32_tmp1", scope=tik.scope_ubuf)
            assist_ub_int32_tmp2 = self.tik_instance.Tensor("int32", (self.block_num,),
                                                            name="assist_ub_int32_tmp2", scope=tik.scope_ubuf)

            util_tik_comm_func.tik_func_vector(self.tik_instance, assist_ub_int32, 0, max_topk_size)
            with self.tik_instance.for_range(0, self.inner_real_dims) as real_idx:
                util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vdiv",
                                                    assist_ub_tmp1,
                                                    assist_ub_copy,
                                                    inner_dividends_ub_fp16[real_idx * self.block_num],
                                                    max_topk_size,
                                                    src1_blk=0, src1_rep=0)
                util_tik_comm_func.tik_func_vconv(self.tik_instance, assist_ub_int32_tmp1, assist_ub_tmp1,
                                                  max_topk_size, mode="floor")
                util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vmul",
                                                    assist_ub_int32_tmp1,
                                                    assist_ub_int32_tmp1,
                                                    inner_dividends_ub_int[real_idx * self.block_num],
                                                    max_topk_size,
                                                    src1_blk=0, src1_rep=0)
                util_tik_comm_func.tik_func_vconv(self.tik_instance, assist_ub_tmp1, assist_ub_int32_tmp1,
                                                  max_topk_size)
                with self.tik_instance.if_scope(self.inner_axis[real_idx] == 1):
                    util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vmul",
                                                        assist_ub_int32_tmp2,
                                                        inner_shape_ub_int[real_idx * self.block_num],
                                                        inner_dividends_ub_int[real_idx * self.block_num],
                                                        self.block_num)
                    util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vsub",
                                                        assist_ub_int32_tmp1,
                                                        assist_ub_int32_tmp2,
                                                        assist_ub_int32_tmp1,
                                                        max_topk_size, src0_blk=0, src0_rep=0)
                util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vsub",
                                                    assist_ub_copy,
                                                    assist_ub_copy,
                                                    assist_ub_tmp1,
                                                    max_topk_size)
                util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vadd",
                                                    assist_ub_int32,
                                                    assist_ub_int32,
                                                    assist_ub_int32_tmp1,
                                                    max_topk_size)
            util_tik_comm_func.tik_func_vector(self.tik_instance, assist_ub_int32_tmp2, max_topk_size, self.block_num)
            util_tik_comm_func.tik_func_vcomple(self.tik_instance, "vsub",
                                                assist_ub_int32,
                                                assist_ub_int32_tmp2,
                                                assist_ub_int32,
                                                max_topk_size, src0_blk=0, src0_rep=0)
            util_tik_comm_func.tik_func_vconv(self.tik_instance, assist_ub, assist_ub_int32,
                                              max_topk_size)

    def reverse_with_topk(self, max_topk_size=1024, align_num=16):
        """
        reverse_with_topk
        use topi to do reverse
        ex:
            input = [[1,2,3,4,5,6],[11,12,13,14,15,16]] axis = -1
            output = [[6,5,4,3,2,1],[16,15,14,13,12,11]]

            process as follows:
               gen assist score ub  [7,8,9,10,11,12,1,2,3,4,5,6]
               topk input with score ub will get the output
        """
        assist_ub = self.tik_instance.Tensor(self.input_data.dtype, (max_topk_size,),
                                             name="assist_ub", scope=tik.scope_ubuf)
        assist_ub_tail = self.tik_instance.Tensor(self.input_data.dtype, (max_topk_size,),
                                                  name="assist_ub_tail", scope=tik.scope_ubuf)
        util_tik_comm_func.tik_func_vector(self.tik_instance, assist_ub, -1.0, max_topk_size)
        util_tik_comm_func.tik_func_vector(self.tik_instance, assist_ub_tail, -1.0, max_topk_size)
        # split num in inner loop
        inner_first_dim_cut_num = self.tik_instance.Scalar("int64", name="inner_first_dim_cut_num")
        inner_first_dim_cut_num.set_as(max_topk_size // self.inner_dividends[0])
        inner_first_dim_cut_num.set_as(inner_first_dim_cut_num // align_num * align_num)
        self.tik_instance.scalar_min(inner_first_dim_cut_num, inner_first_dim_cut_num, self.inner_loop[0])
        # gen assist data
        with self.tik_instance.new_stmt_scope():
            assist_ub_tmp = self.tik_instance.Tensor(self.input_data.dtype, (max_topk_size,),
                                                     name="assist_ub_tmp", scope=tik.scope_ubuf)
            self.gen_assist_data(assist_ub_tmp, max_topk_size=max_topk_size)
            one_topk_num = inner_first_dim_cut_num * self.inner_dividends[0]
            with self.tik_instance.if_scope(one_topk_num >= self.vector_num):
                self.tik_instance.vmuls(self.vector_num, assist_ub, assist_ub_tmp, 1.0,
                                        one_topk_num // self.vector_num,
                                        1, 1, 8, 8)
            with self.tik_instance.if_scope(one_topk_num % self.vector_num != 0):
                ub_offset = (one_topk_num // self.vector_num) * self.vector_num
                self.tik_instance.vmuls(one_topk_num % self.vector_num,
                                        assist_ub[ub_offset], assist_ub_tmp[ub_offset], 1.0,
                                        1,
                                        1, 1, 8, 8)

        inner_first_dim_cut_last_num = self.tik_instance.Scalar("int64", name="inner_first_dim_cut_last_num")
        inner_first_dim_cut_last_num.set_as(
            ceil_div(self.inner_loop[0] % inner_first_dim_cut_num, align_num) * align_num)
        inner_first_dim_cut_last_offset = self.tik_instance.Scalar("int64", name="inner_first_dim_cut_last_offset")
        inner_first_dim_cut_last_offset.set_as(self.inner_loop[0] - inner_first_dim_cut_last_num)
        # gen tail assist data
        one_topk_num = inner_first_dim_cut_last_num * self.inner_dividends[0]
        with self.tik_instance.if_scope(one_topk_num >= self.vector_num):
            self.tik_instance.vmuls(self.vector_num, assist_ub_tail, assist_ub, 1.0,
                                    one_topk_num // self.vector_num,
                                    1, 1, 8, 8)
        with self.tik_instance.if_scope(one_topk_num % self.vector_num != 0):
            ub_offset = (one_topk_num // self.vector_num) * self.vector_num
            self.tik_instance.vmuls(one_topk_num % self.vector_num,
                                    assist_ub_tail[ub_offset], assist_ub[ub_offset], 1.0,
                                    1,
                                    1, 1, 8, 8)

        inner_first_dim_cut_full_size = self.inner_loop[0] // inner_first_dim_cut_num

        inner_first_dim_cut_full_floor_loop = inner_first_dim_cut_full_size // 4
        inner_first_dim_cut_full_ceil_loop = ceil_div(inner_first_dim_cut_full_size, 4)
        inner_first_dim_cut_full_ceil_tail_size = inner_first_dim_cut_full_size % 4

        def _run_inner_reverse(in_offset, out_offset, reverse_num, do_inner_first_num, ub_list):
            input_ub, topk_ub_1, output_ub, assist_score_ub = ub_list
            # step 1 copy 4 line in ub
            with self.tik_instance.if_scope(self.inner_axis[0] == 0):
                with self.tik_instance.new_stmt_scope(disable_sync=True):
                    with self.tik_instance.for_range(0, do_inner_first_num) as _copy_idx:
                        burst_len = (reverse_num + self.block_num - 1) // self.block_num
                        src_offset = in_offset + _copy_idx * reverse_num
                        self.tik_instance.data_move(input_ub[_copy_idx * max_topk_size],
                                                    self.input_data[src_offset],
                                                    0, 1, burst_len, 0, 0)
            with self.tik_instance.if_scope(self.inner_axis[0] == 1):
                with self.tik_instance.new_stmt_scope(disable_sync=True):
                    with self.tik_instance.for_range(0, do_inner_first_num) as _copy_idx:
                        burst_len = (reverse_num + self.block_num - 1) // self.block_num
                        src_offset = in_offset + _copy_idx * reverse_num
                        input_ub_offset = do_inner_first_num - 1 - _copy_idx
                        self.tik_instance.data_move(input_ub[input_ub_offset * max_topk_size],
                                                    self.input_data[src_offset],
                                                    0, 1, burst_len, 0, 0)
            # step 2 trans to proposal format
            for i in range(4):
                self.tik_instance.vconcat(topk_ub_1, input_ub[max_topk_size * i:], max_topk_size // 16, i)
            self.tik_instance.vconcat(topk_ub_1, assist_score_ub, max_topk_size // 16, 4)
            # step 3 sort proposal ub to a ordered queue
            sort_within_ub(self.tik_instance, topk_ub_1, ceil_div(max_topk_size, 16) * 16)
            # step 4 extract the proposal ub to 4 ub
            for i in range(4):
                self.tik_instance.vextract(output_ub[max_topk_size * i:], topk_ub_1, max_topk_size // 16, i)
            # step 5 copy output to gm
            with self.tik_instance.if_scope(reverse_num % self.block_num == 0):
                with self.tik_instance.new_stmt_scope(disable_sync=True):
                    with self.tik_instance.for_range(0, do_inner_first_num) as _copy_idx:
                        burst_len = reverse_num // self.block_num
                        gm_out_offset = out_offset + _copy_idx * reverse_num
                        self.tik_instance.data_move(self.output_data[gm_out_offset],
                                                    output_ub[_copy_idx * max_topk_size],
                                                    0, 1, burst_len, 0, 0)
            with self.tik_instance.if_scope(reverse_num % self.block_num != 0):
                with self.tik_instance.for_range(0, do_inner_first_num) as _copy_idx:
                    burst_len = reverse_num // self.block_num
                    gm_out_offset = out_offset + _copy_idx * reverse_num
                    self.tik_instance.data_move(self.output_data[gm_out_offset],
                                                output_ub[_copy_idx * max_topk_size],
                                                0, 1, burst_len, 0, 0)
                    tail_block_num = self.tik_instance.Tensor(self.input_data.dtype, (self.block_num,),
                                                              name="tail_block_num", scope=tik.scope_ubuf)
                    with self.tik_instance.for_range(0, self.block_num) as block_idx:
                        tail_block_num[block_idx].set_as(
                            output_ub[_copy_idx * max_topk_size + reverse_num - self.block_num + block_idx])
                    self.tik_instance.data_move(self.output_data[gm_out_offset + reverse_num - self.block_num],
                                                tail_block_num,
                                                0, 1, 1, 0, 0)

        # outer loop
        with self.tik_instance.for_range(self.core_outer_start,
                                         self.core_outer_num + self.core_outer_start) as outer_idx:
            outer_current_index = self.tik_instance.Scalar(dtype="int64", name="current_index")
            outer_move_out_index = self.tik_instance.Scalar(dtype="int64", name="outer_move_out_index")
            self.axis_compute_with_scalar_array(
                outer_idx, outer_current_index, outer_move_out_index, self.outer_shape, self.outer_axis,
                self.outer_dividends, self.outer_real_dims)
            # inner first loop
            with self.tik_instance.for_range(0, inner_first_dim_cut_full_ceil_loop) as inner_first_dim_idx:
                inner_first_in_idx = self.tik_instance.Scalar(dtype="int64", name="inner_first_in_idx")
                inner_first_num = self.tik_instance.Scalar(dtype="int64", name="inner_first_num", init_value=4)
                inner_first_in_idx.set_as(inner_first_dim_idx * inner_first_dim_cut_num * 4)
                inner_first_out_index = self.tik_instance.Scalar(dtype="int64", name="inner_first_out_index")
                inner_first_out_index.set_as(inner_first_in_idx)
                with self.tik_instance.if_scope(inner_first_dim_cut_full_floor_loop == inner_first_dim_idx):
                    inner_first_num.set_as(inner_first_dim_cut_full_ceil_tail_size)
                with self.tik_instance.if_scope(self.inner_axis[0] == 1):
                    inner_first_out_index.set_as(
                        self.inner_loop[0] - (inner_first_in_idx + inner_first_num * inner_first_dim_cut_num))
                    self.tik_instance.scalar_max(inner_first_out_index, inner_first_out_index, 0)
                input_offset = \
                    outer_idx * self.inner_total_num_list[-1] + inner_first_in_idx * self.inner_dividends[0]
                output_offset = \
                    outer_move_out_index * self.inner_total_num_list[-1] \
                    + inner_first_out_index * self.inner_dividends[0]
                process_num = inner_first_dim_cut_num * self.inner_dividends[0]
                input_ub_ping = self.tik_instance.Tensor(self.input_data.dtype, (max_topk_size * 4,),
                                                         name="input_ub_ping", scope=tik.scope_ubuf)
                topk_ub_1_ping = self.tik_instance.Tensor(self.input_data.dtype, (max_topk_size * 8,),
                                                          name="topk_ub_1_ping", scope=tik.scope_ubuf)
                output_ub_ping = self.tik_instance.Tensor(self.input_data.dtype, (max_topk_size * 4,),
                                                          name="output_ub_ping", scope=tik.scope_ubuf)
                ping_ub_list = [input_ub_ping, topk_ub_1_ping, output_ub_ping, assist_ub]
                _run_inner_reverse(input_offset, output_offset, process_num, inner_first_num, ping_ub_list)

            with self.tik_instance.if_scope(inner_first_dim_cut_last_num != 0):
                inner_first_in_idx = inner_first_dim_cut_last_offset
                inner_first_out_index = self.tik_instance.Scalar(dtype="int64", name="inner_first_out_index")
                inner_first_out_index.set_as(inner_first_in_idx)
                with self.tik_instance.if_scope(self.inner_axis[0] == 1):
                    inner_first_out_index.set_as(0)
                input_offset = \
                    outer_idx * self.inner_total_num_list[-1] + inner_first_in_idx * self.inner_dividends[0]
                output_offset = \
                    outer_move_out_index * self.inner_total_num_list[-1] \
                    + inner_first_out_index * self.inner_dividends[0]
                process_num = inner_first_dim_cut_last_num * self.inner_dividends[0]
                input_ub_ping = self.tik_instance.Tensor(self.input_data.dtype, (max_topk_size * 4,),
                                                         name="input_ub_ping", scope=tik.scope_ubuf)
                topk_ub_1_ping = self.tik_instance.Tensor(self.input_data.dtype, (max_topk_size * 8,),
                                                          name="topk_ub_1_ping", scope=tik.scope_ubuf)
                output_ub_ping = self.tik_instance.Tensor(self.input_data.dtype, (max_topk_size * 4,),
                                                          name="output_ub_ping", scope=tik.scope_ubuf)
                ping_ub_list = [input_ub_ping, topk_ub_1_ping, output_ub_ping, assist_ub_tail]
                _run_inner_reverse(input_offset, output_offset, process_num, 1, ping_ub_list)

    def tiling_topk_compute(self):
        """
        tiling_topk_compute
        """
        align_num = self.tik_instance.Scalar(dtype="int32", name="align_num")
        with self.tik_instance.for_range(0, self.block_num) as _idx:
            align_idx = self.block_num - _idx
            with self.tik_instance.if_scope((self.inner_dividends[0] * align_idx) % self.block_num == 0):
                align_num.set_as(align_idx)

        with self.tik_instance.if_scope(self.inner_dividends[0] * align_num > 1024):
            with self.tik_instance.if_scope(align_num > (2048 // self.inner_dividends[0])):
                align_num.set_as(2048 // self.inner_dividends[0])
            with self.tik_instance.if_scope(align_num > self.inner_loop[0]):
                align_num.set_as(self.inner_loop[0])
            with self.tik_instance.new_stmt_scope():
                self.reverse_with_topk(max_topk_size=2048, align_num=align_num)
        with self.tik_instance.if_scope(self.inner_dividends[0] * align_num <= 1024):
            with self.tik_instance.if_scope(align_num > (1024 // self.inner_dividends[0])):
                align_num.set_as(1024 // self.inner_dividends[0])
            with self.tik_instance.if_scope(align_num > self.inner_loop[0]):
                align_num.set_as(self.inner_loop[0])
            with self.tik_instance.new_stmt_scope():
                self.reverse_with_topk(max_topk_size=1024, align_num=align_num)


# 'pylint: disable=unused-argument,invalid-name
@register_operator("ReverseV2")
def reverse_v2(x, axis, y, kernel_name="reverse_v2"):
    """ calculating reverse_v2 tensor by axis parameters

    Parameters
    ----------
    x : dict
        shape and dtype of input
    axis: dict
        shape and dtype of axis
    y: dict
        shape and dtype of output

    kernel_name : str
        cce kernel name, default value is "pad"

    Returns
    -------
    None.
    """
    x_dtype = x.get("dtype")
    x_shape = x.get("shape")
    tik_instance = ReverseExt2(x_shape, x_dtype, axis, kernel_name)
    tik_instance = tik_instance.op_run()

    return tik_instance
