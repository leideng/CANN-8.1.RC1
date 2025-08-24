# Copyright 2020 Huawei Technologies Co., Ltd
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
tabulate_fusion
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from tbe.common.platform import get_bit_len


# 'pylint: disable=invalid-name,too-many-statements,too-many-locals,too-many-arguments
# 'pylint: disable=too-many-instance-attributes,unused-argument,too-few-public-methods
def ceiling_value(value, factor):
    """
    if not divide exactly then plus 1

    Parameters
    ----------
    value: input value
    factor: factor

    Returns
    -------
    the ceiling value
    """
    return (value + factor - 1) // factor


def aligned_value(value, factor):
    """
    Alignment value based on factor.

    Parameters
    ----------
    value: input value
    factor: alignment base

    Returns
    -------
    aligned value
    """
    return (value + factor - 1) // factor * factor


class TabulateFusion():
    """
    TabulateFusion class
    """

    # int32's max value
    MAX_SHAPE_SIZE = 2 ** 31 - 1
    # tiling param num
    TILING_ARG_NUM = 12
    # reserved ub size
    RESERVED_UB_SIZE = 24 * 1024
    # 8 bit
    EIGHT_BIT = 8
    # bytes of one block
    BLOCK_BYTES = 32
    # 256 bytes
    VECTOR_BYTES = 256
    # there are six numbers in the table_info, the first five numbers are used.
    TABLE_INFO_NUM = 6

    NUM_0 = 0
    NUM_1 = 1
    NUM_2 = 2
    NUM_3 = 3
    NUM_4 = 4
    NUM_5 = 5
    NUM_6 = 6
    NUM_7 = 7
    NUM_8 = 8
    NUM_9 = 9
    NUM_10 = 10
    NUM_64 = 64
    NUM_128 = 128

    def __init__(self, table, table_info, em_x, em, descriptor, last_layer_size):
        """
        Init TabulateFusion parameters
        """
        self.last_layer_size = last_layer_size
        self.tik_inst = tik.Tik()
        self.dtype = table.get("dtype").lower()
        self.dtype_int32 = "int32"
        self.dsize = get_bit_len(self.dtype) // self.EIGHT_BIT
        self.table_idx_dsize = get_bit_len(self.dtype_int32) // self.EIGHT_BIT
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - self.RESERVED_UB_SIZE
        self.block_elems = self.BLOCK_BYTES // self.dsize
        self.vector_elems = self.VECTOR_BYTES // self.dsize
        self.last_size_align = aligned_value(self.last_layer_size, self.vector_elems)

        self.ub_elems = self.ub_size // self.dsize
        self.one_portion_ub_elems = ((self.ub_elems // self.NUM_4) // self.vector_elems) * self.vector_elems
        self.mask_elems_max = aligned_value(self.one_portion_ub_elems // self.vector_elems, self.vector_elems)

        self.mask_dtype = "uint64"
        self.dtype_int16 = "int16"
        self.tiling_dtype = "int64"
        self.tiling_align = aligned_value(self.TILING_ARG_NUM, self.NUM_4)
        self.tiling_gm = self.tik_inst.Tensor(self.tiling_dtype, (self.tiling_align,), name="tiling_gm",
                                              scope=tik.scope_gm)
        self.table_info_size_align = aligned_value(self.TABLE_INFO_NUM, self.block_elems)
        self.table_gm, self.table_info_gm, self.em_x_gm, self.em_gm, self.descriptor_gm = self._init_gm_tensor()

        self.lower = None
        self.upper = None
        self.max = None
        self.lower_ub = None
        self.upper_ub = None
        self.max_ub = None
        self.stride0 = None
        self.stride1 = None

        # tiling params
        self.need_core_num = None
        # nloc offset of aicore or vectorcore engine
        self.nloc_engine_offset = None
        self.nnei = None
        self.nloc_one_core = None
        self.nloc_last_core = None
        # process num of nloc per loop
        self.nloc_per_loop = None
        # nloc loops for post core
        self.pre_core_loops = None
        self.pre_core_nloc_tail = None
        # nloc loops for last core
        self.last_core_loops = None
        self.last_core_nloc_tail = None
        self.core_num_var = None

    def tabulate_fusion_compute(self):
        """
        main process of tabulate_fusion
        """
        # get tiling data
        self._get_tiling_args()
        with self.tik_inst.for_range(0, self.core_num_var, block_num=self.core_num_var) as core_i:
            self._get_table_info_data()

            with self.tik_inst.if_scope(core_i < self.need_core_num - 1):
                with self.tik_inst.new_stmt_scope():
                    self._pre_core_compute(core_i)

            with self.tik_inst.if_scope(core_i == self.need_core_num - 1):
                with self.tik_inst.new_stmt_scope():
                    self._last_core_compute(core_i)

    def _init_gm_tensor(self):
        """
        init gm tensor
        """
        table_gm = self.tik_inst.Tensor(self.dtype, (self.MAX_SHAPE_SIZE,), name="table_gm", scope=tik.scope_gm)
        table_info_gm = self.tik_inst.Tensor(self.dtype, (self.table_info_size_align,), name="table_info_gm",
                                             scope=tik.scope_gm)
        em_x_gm = self.tik_inst.Tensor(self.dtype, (self.MAX_SHAPE_SIZE,), name="em_x_gm", scope=tik.scope_gm)
        em_gm = self.tik_inst.Tensor(self.dtype, (self.MAX_SHAPE_SIZE,), name="em_gm", scope=tik.scope_gm)
        descriptor_gm = self.tik_inst.Tensor(self.dtype, (self.MAX_SHAPE_SIZE,), name="descriptor_gm",
                                             scope=tik.scope_gm)

        return [table_gm, table_info_gm, em_x_gm, em_gm, descriptor_gm]

    def _get_table_info_data(self):
        """
        get lower, upper, max, stride0, stride1 from table_info
        """
        table_info_ub = self.tik_inst.Tensor(self.dtype, (self.table_info_size_align,), name="table_info_ub",
                                             scope=tik.scope_ubuf)
        self.tik_inst.data_move(table_info_ub, self.table_info_gm, 0, 1,
                                self.table_info_size_align // self.block_elems, 0, 0)

        self.lower = self.tik_inst.Scalar(dtype=self.dtype, name="lower")
        self.upper = self.tik_inst.Scalar(dtype=self.dtype, name="upper")
        self.max = self.tik_inst.Scalar(dtype=self.dtype, name="max")
        self.stride0 = self.tik_inst.Scalar(dtype=self.dtype, name="stride0")
        self.stride1 = self.tik_inst.Scalar(dtype=self.dtype, name="stride1")

        self.lower.set_as(table_info_ub[self.NUM_0])
        self.upper.set_as(table_info_ub[self.NUM_1])
        self.max.set_as(table_info_ub[self.NUM_2])
        self.stride0.set_as(table_info_ub[self.NUM_3])
        self.stride1.set_as(table_info_ub[self.NUM_4])

        self.lower_ub = self.tik_inst.Tensor(self.dtype, (self.vector_elems,), name="lower_ub", scope=tik.scope_ubuf)
        self.upper_ub = self.tik_inst.Tensor(self.dtype, (self.vector_elems,), name="upper_ub", scope=tik.scope_ubuf)
        self.max_ub = self.tik_inst.Tensor(self.dtype, (self.vector_elems,), name="max_ub", scope=tik.scope_ubuf)
        self.tik_inst.vector_dup(self.NUM_64, self.lower_ub, self.lower, 1, 1, 8)
        self.tik_inst.vector_dup(self.NUM_64, self.upper_ub, self.upper, 1, 1, 8)
        self.tik_inst.vector_dup(self.NUM_64, self.max_ub, self.max, 1, 1, 8)

    def _get_tiling_args(self):
        """
        get runtime params from tiling data
        """
        self.need_core_num = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="need_core_num")
        self.nloc_engine_offset = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="nloc_engine_offset")
        self.nnei = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="nnei")
        self.nloc_one_core = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="nloc_one_core")
        self.nloc_last_core = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="nloc_last_core")
        self.nloc_per_loop = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="nloc_per_loop")
        self.pre_core_loops = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="pre_core_loops")
        self.pre_core_nloc_tail = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="pre_core_nloc_tail")
        self.last_core_loops = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="last_core_loops")
        self.last_core_nloc_tail = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="last_core_nloc_tail")
        self.core_num_var = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="core_num_var")

        with self.tik_inst.new_stmt_scope():
            tiling_ub = self.tik_inst.Tensor(self.tiling_dtype, (self.tiling_align,), name="tiling_ub",
                                             scope=tik.scope_ubuf)
            self.tik_inst.data_move(tiling_ub, self.tiling_gm, 0, 1, self.tiling_align // self.NUM_4, 0, 0)

            self.need_core_num.set_as(tiling_ub[self.NUM_0])
            self.nloc_engine_offset.set_as(tiling_ub[self.NUM_1])
            self.nnei.set_as(tiling_ub[self.NUM_2])
            self.nloc_one_core.set_as(tiling_ub[self.NUM_3])
            self.nloc_last_core.set_as(tiling_ub[self.NUM_4])
            self.nloc_per_loop.set_as(tiling_ub[self.NUM_5])
            self.pre_core_loops.set_as(tiling_ub[self.NUM_6])
            self.pre_core_nloc_tail.set_as(tiling_ub[self.NUM_7])
            self.last_core_loops.set_as(tiling_ub[self.NUM_8])
            self.last_core_nloc_tail.set_as(tiling_ub[self.NUM_9])
            self.core_num_var.set_as(tiling_ub[self.NUM_10])

    def _cal_locate_xx_masks(self, mask_tensors, em_x_ub, repeats):
        """
        calculate mask_tensor2, mask_tensor3, mask_tensor4.
        mask_tensors is (mask_tensor2, mask_tensor3, mask_tensor4).
        """
        with self.tik_inst.new_stmt_scope():
            mask_tensor2, mask_tensor3, mask_tensor4 = mask_tensors
            mask_tensor1_not = self.tik_inst.Tensor(self.mask_dtype, (self.mask_elems_max,),
                                                    name="mask_tensor1_not", scope=tik.scope_ubuf)
            mask_tensor2_not = self.tik_inst.Tensor(self.dtype_int16, (self.mask_elems_max * self.NUM_4,),
                                                    name="mask_tensor2_not", scope=tik.scope_ubuf)

            repeats_int16 = ceiling_value(repeats * self.NUM_4, self.NUM_128)
            # `xx >= lower`
            self.tik_inst.vcmpv_ge(mask_tensor1_not, em_x_ub, self.lower_ub, repeats, 1, 1, 8, 0)
            mask_tensor1_not_int16 = mask_tensor1_not.reinterpret_cast_to(self.dtype_int16)

            # `xx < upper`
            self.tik_inst.vcmpv_lt(mask_tensor2, em_x_ub, self.upper_ub, repeats, 1, 1, 8, 0)
            mask_tensor2_int16 = mask_tensor2.reinterpret_cast_to(self.dtype_int16)
            # `vnot: xx >= upper`
            self.tik_inst.vnot(self.NUM_128, mask_tensor2_not, mask_tensor2_int16, repeats_int16, 1, 1, 8, 8)
            # `2. lower <= xx < upper`
            self.tik_inst.vand(self.NUM_128, mask_tensor2_int16, mask_tensor2_int16, mask_tensor1_not_int16,
                               repeats_int16, 1, 1, 1, 8, 8, 8)

            # `xx < max`
            self.tik_inst.vcmpv_lt(mask_tensor3, em_x_ub, self.max_ub, repeats, 1, 1, 8, 0)
            mask_tensor3_int16 = mask_tensor3.reinterpret_cast_to(self.dtype_int16)
            # `4. vnot: xx >= max`
            mask_tensor4_int16 = mask_tensor4.reinterpret_cast_to(self.dtype_int16)
            self.tik_inst.vnot(self.NUM_128, mask_tensor4_int16, mask_tensor3_int16, repeats_int16, 1, 1, 8, 8)
            # `3. upper <= xx < max`
            self.tik_inst.vand(self.NUM_128, mask_tensor3_int16, mask_tensor3_int16, mask_tensor2_not,
                               repeats_int16, 1, 1, 1, 8, 8, 8)

    def _locate_xx_branch_2(self, ub_tensors, repeats, mask_left, mask_right, mask_tensor2):
        """
        Calculate table_idx and xx_new of branch 2 of locate_xx.
        ub_tensors is (em_x_ub, xx_new_ub, table_idx_ub, inner_fp32_ub).

        int table_idx = (int)((xx - lower) / stride0);
        xx -= (table_idx * stride0 + lower);
        """
        em_x_ub, xx_new_ub, table_idx_ub, inner_fp32_ub = ub_tensors
        stride0_r_s = 1 / self.stride0

        with self.tik_inst.for_range(0, repeats) as i:
            mask_right.set_as(mask_tensor2[i])
            with self.tik_inst.if_scope(mask_right != 0):
                mask = [mask_left, mask_right]
                # calculate table_idx
                self.tik_inst.vadds(mask, inner_fp32_ub, em_x_ub[self.NUM_64 * i], -1.0 * self.lower, 1, 1, 1, 8, 8)
                self.tik_inst.vmuls(mask, inner_fp32_ub, inner_fp32_ub, stride0_r_s, 1, 1, 1, 8, 8)
                self.tik_inst.vconv(mask, "floor", table_idx_ub[self.NUM_64 * i], inner_fp32_ub, 1, 1, 1, 8, 8)

                # calculate xx_new
                self.tik_inst.vconv(mask, "none", inner_fp32_ub, table_idx_ub[self.NUM_64 * i], 1, 1, 1, 8, 8)
                self.tik_inst.vmuls(mask, inner_fp32_ub, inner_fp32_ub, self.stride0, 1, 1, 1, 8, 8)
                self.tik_inst.vadds(mask, inner_fp32_ub, inner_fp32_ub, self.lower, 1, 1, 1, 8, 8)
                self.tik_inst.vsub(mask, xx_new_ub[self.NUM_64 * i], em_x_ub[self.NUM_64 * i], inner_fp32_ub,
                                   1, 1, 1, 1, 8, 8, 8)

    def _locate_xx_branch_3(self, ub_tensors, repeats, mask_left, mask_right, mask_tensor3, first_stride):
        """
        Calculate table_idx and xx_new of branch 3 of locate_xx.
        ub_tensors is (em_x_ub, xx_new_ub, table_idx_ub, inner_fp32_ub).

        int first_stride = int((upper - lower) / stride0);
        int table_idx = first_stride + int((xx - upper) / stride1);
        xx -= ((table_idx - first_stride) * stride1 + upper);
        """
        with self.tik_inst.new_stmt_scope():
            em_x_ub, xx_new_ub, table_idx_ub, inner_fp32_ub = ub_tensors
            inner_int32_ub = self.tik_inst.Tensor(self.dtype_int32, (self.vector_elems,), name="inner_int32_ub",
                                                  scope=tik.scope_ubuf)
            first_stride_ub = self.tik_inst.Tensor(self.dtype_int32, (self.vector_elems,), name="first_stride_ub",
                                                   scope=tik.scope_ubuf)
            self.tik_inst.vector_dup(self.NUM_64, first_stride_ub, first_stride, 1, 1, 8)
            stride1_r_s = 1 / self.stride1

            with self.tik_inst.for_range(0, repeats) as i:
                mask_right.set_as(mask_tensor3[i])
                with self.tik_inst.if_scope(mask_right != 0):
                    mask = [mask_left, mask_right]
                    # calculate table_idx
                    self.tik_inst.vadds(mask, inner_fp32_ub, em_x_ub[self.NUM_64 * i], -1.0 * self.upper,
                                        1, 1, 1, 8, 8)
                    self.tik_inst.vmuls(mask, inner_fp32_ub, inner_fp32_ub, stride1_r_s, 1, 1, 1, 8, 8)
                    # `int((xx - upper) / stride1)`
                    self.tik_inst.vconv(mask, "floor", inner_int32_ub, inner_fp32_ub, 1, 1, 1, 8, 8)
                    self.tik_inst.vadd(mask, table_idx_ub[self.NUM_64 * i], inner_int32_ub, first_stride_ub,
                                       1, 1, 1, 1, 8, 8, 8)

                    # calculate xx_new
                    self.tik_inst.vconv(mask, "none", inner_fp32_ub, inner_int32_ub, 1, 1, 1, 8, 8)
                    self.tik_inst.vmuls(mask, inner_fp32_ub, inner_fp32_ub, self.stride1, 1, 1, 1, 8, 8)
                    self.tik_inst.vadds(mask, inner_fp32_ub, inner_fp32_ub, self.upper, 1, 1, 1, 8, 8)
                    self.tik_inst.vsub(mask, xx_new_ub[self.NUM_64 * i], em_x_ub[self.NUM_64 * i], inner_fp32_ub,
                                       1, 1, 1, 1, 8, 8, 8)

    def _locate_xx_branch_4(self, table_idx_ub, repeats, mask_left, mask_right, mask_tensor4, first_stride):
        """
        Calculate table_idx and xx_new of branch 4 of locate_xx.

        int first_stride = int((upper - lower) / stride0);
        int table_idx = first_stride + (int)((max - upper) / stride1) - 1;
        xx = 0;
        """
        with self.tik_inst.new_stmt_scope():
            second_stride = self.tik_inst.Scalar(dtype=self.dtype_int32, name="second_stride")
            max_sub_upper = self.max - self.upper
            second_stride.set_as(max_sub_upper / self.stride1)
            # calculate table_idx
            table_idx_value = first_stride + second_stride - 1
            table_idx_value_ub = self.tik_inst.Tensor(self.dtype_int32, (self.vector_elems,),
                                                      name="table_idx_value_ub", scope=tik.scope_ubuf)
            self.tik_inst.vector_dup(self.NUM_64, table_idx_value_ub, table_idx_value, 1, 1, 8)

            with self.tik_inst.for_range(0, repeats) as i:
                mask_right.set_as(mask_tensor4[i])
                with self.tik_inst.if_scope(mask_right != 0):
                    self.tik_inst.vadd([mask_left, mask_right], table_idx_ub[self.NUM_64 * i],
                                       table_idx_ub[self.NUM_64 * i], table_idx_value_ub, 1, 1, 1, 1, 8, 8, 8)

    def _cal_table_idx_and_xx(self, mask_tensors, ub_tensors, repeats):
        """
        calculate table_idx and xx.
        mask_tensors is (mask_tensor2, mask_tensor3, mask_tensor4).
        ub_tensors is (em_x_ub, xx_new_ub, table_idx_ub).
        """
        with self.tik_inst.new_stmt_scope():
            em_x_ub, xx_new_ub, table_idx_ub = ub_tensors
            mask_tensor2, mask_tensor3, mask_tensor4 = mask_tensors

            mask_left = self.tik_inst.Scalar(dtype=self.mask_dtype, name="mask_left", init_value=0)
            mask_right = self.tik_inst.Scalar(dtype=self.mask_dtype, name="mask_right")
            inner_fp32_ub = self.tik_inst.Tensor(self.dtype, (self.vector_elems,), name="inner_fp32_ub",
                                                 scope=tik.scope_ubuf)
            tensors = (em_x_ub, xx_new_ub, table_idx_ub, inner_fp32_ub)

            # branch 2 of locate_xx
            self._locate_xx_branch_2(tensors, repeats, mask_left, mask_right, mask_tensor2)

            first_stride = self.tik_inst.Scalar(dtype=self.dtype_int32, name="first_stride")
            upper_sub_lower = self.upper - self.lower
            first_stride.set_as(upper_sub_lower / self.stride0)
            self._locate_xx_branch_3(tensors, repeats, mask_left, mask_right, mask_tensor3, first_stride)
            self._locate_xx_branch_4(table_idx_ub, repeats, mask_left, mask_right, mask_tensor4, first_stride)

    def _locate_xx(self, ub_tensors, nloc_offset, elems_align):
        """
        locate_xx, calculate table_idx and xx.
        ub_tensors is (em_x_ub, xx_new_ub, table_idx_ub).
        """
        em_x_ub, xx_new_ub, table_idx_ub = ub_tensors
        self.tik_inst.data_move(em_x_ub, self.em_x_gm[nloc_offset * self.nnei], 0, 1,
                                elems_align // self.block_elems, 0, 0)

        repeats = elems_align // self.vector_elems
        self.tik_inst.vector_dup(self.NUM_64, xx_new_ub, 0, repeats, 1, 8)
        self.tik_inst.vector_dup(self.NUM_64, table_idx_ub, 0, repeats, 1, 8)

        with self.tik_inst.new_stmt_scope():
            mask_tensor2 = self.tik_inst.Tensor(self.mask_dtype, (self.mask_elems_max,),
                                                name="mask_tensor2", scope=tik.scope_ubuf)
            mask_tensor3 = self.tik_inst.Tensor(self.mask_dtype, (self.mask_elems_max,),
                                                name="mask_tensor3", scope=tik.scope_ubuf)
            mask_tensor4 = self.tik_inst.Tensor(self.mask_dtype, (self.mask_elems_max,),
                                                name="mask_tensor4", scope=tik.scope_ubuf)
            mask_tensors = (mask_tensor2, mask_tensor3, mask_tensor4)

            self._cal_locate_xx_masks(mask_tensors, em_x_ub, repeats)
            self._cal_table_idx_and_xx(mask_tensors, ub_tensors, repeats)

    def _cal_var(self, var_ub, xx_new_ub, table_idx_ub, offset):
        """
        get a0,a1,a2,a3,a4,a5,a6, then calculate var.
        """
        with self.tik_inst.new_stmt_scope():
            # one row data of table is last_size_align*6
            a_i_ub = self.tik_inst.Tensor(self.dtype, (self.last_size_align * self.NUM_6,),
                                          name="a_i_ub", scope=tik.scope_ubuf)
            xx_new = self.tik_inst.Scalar(dtype=self.dtype, name="xx_new")
            xx_new.set_as(xx_new_ub[offset])
            table_idx = self.tik_inst.Scalar(dtype=self.dtype_int32, name="table_idx")
            table_idx.set_as(table_idx_ub[offset])

            self.tik_inst.data_move(a_i_ub, self.table_gm[table_idx * (self.last_size_align * self.NUM_6)], 0, 1,
                                    ceiling_value(self.last_size_align * self.NUM_6, self.block_elems), 0, 0)

            # `var = a0 + (a1 + (a2 + (a3 + (a4 + a5 * xx) * xx) * xx) * xx) * xx`
            repeats = self.last_size_align // self.vector_elems
            xx_new_dup = self.tik_inst.Tensor(self.dtype, (self.vector_elems,),
                                              name="xx_new_dup", scope=tik.scope_ubuf)
            self.tik_inst.vector_dup(self.NUM_64, xx_new_dup, xx_new, 1, 1, 8)
            self.tik_inst.vmla(self.NUM_64, a_i_ub[self.last_size_align * self.NUM_4],
                               a_i_ub[self.last_size_align * self.NUM_5], xx_new_dup, repeats, 1, 1, 1, 8, 8, 0)
            self.tik_inst.vmla(self.NUM_64, a_i_ub[self.last_size_align * self.NUM_3],
                               a_i_ub[self.last_size_align * self.NUM_4], xx_new_dup, repeats, 1, 1, 1, 8, 8, 0)
            self.tik_inst.vmla(self.NUM_64, a_i_ub[self.last_size_align * self.NUM_2],
                               a_i_ub[self.last_size_align * self.NUM_3], xx_new_dup, repeats, 1, 1, 1, 8, 8, 0)
            self.tik_inst.vmla(self.NUM_64, a_i_ub[self.last_size_align * self.NUM_1],
                               a_i_ub[self.last_size_align * self.NUM_2], xx_new_dup, repeats, 1, 1, 1, 8, 8, 0)
            self.tik_inst.vmuls(self.NUM_64, a_i_ub[self.last_size_align * self.NUM_1],
                                a_i_ub[self.last_size_align * self.NUM_1], xx_new, repeats, 1, 1, 8, 8)
            self.tik_inst.vadd(self.NUM_64, var_ub, a_i_ub[self.last_size_align * self.NUM_1], a_i_ub[0],
                               repeats, 1, 1, 1, 8, 8, 8)

    def _cal_output(self, nloc_i_res, res_size, var_ub, ll_values, ago, xx, out_i_offset, nnei_j, for_end_s):
        """
        calculate result of nloc_i, last_layer_size is 32 bytes align.
        res_size is aligned_value(self.last_layer_size * self.NUM_4, self.vector_elems) + self.vector_elems.
        ll_values is (ll_0, ll_1, ll_2, ll_3).
        """
        with self.tik_inst.new_stmt_scope():
            # `var * ll[0], var * ll[1], var * ll[2], var * ll[3]`
            ll_0, ll_1, ll_2, ll_3 = ll_values
            out_i = self.tik_inst.Tensor(self.dtype, (res_size,), name="out_i_align", scope=tik.scope_ubuf)
            ll_repeats = ceiling_value(self.last_layer_size, self.vector_elems)
            self.tik_inst.vmuls(self.NUM_64, out_i[0], var_ub, ll_0, ll_repeats, 1, 1, 8, 8)
            self.tik_inst.vmuls(self.NUM_64, out_i[self.last_layer_size], var_ub, ll_1, ll_repeats, 1, 1, 8, 8)
            self.tik_inst.vmuls(self.NUM_64, out_i[self.last_layer_size * self.NUM_2], var_ub, ll_2,
                                ll_repeats, 1, 1, 8, 8)
            self.tik_inst.vmuls(self.NUM_64, out_i[self.last_layer_size * self.NUM_3], var_ub, ll_3,
                                ll_repeats, 1, 1, 8, 8)

            out_repeats = ceiling_value(self.last_layer_size * self.NUM_4, self.vector_elems)
            with self.tik_inst.if_scope(ago != xx):
                self.tik_inst.vadd(self.NUM_64, nloc_i_res, nloc_i_res, out_i, out_repeats, 1, 1, 1, 8, 8, 8)
            with self.tik_inst.else_scope():
                # `out_i * (nnei - nnei_j)`
                self.tik_inst.vmuls(self.NUM_64, out_i, out_i, self.nnei - nnei_j, out_repeats, 1, 1, 8, 8)
                self.tik_inst.vadd(self.NUM_64, nloc_i_res, nloc_i_res, out_i, out_repeats, 1, 1, 1, 8, 8, 8)
                # move result to gm
                self.tik_inst.data_move(self.descriptor_gm[out_i_offset], nloc_i_res, 0, 1,
                                        self.last_layer_size * self.NUM_4 // self.block_elems, 0, 0)

                # `ago==xx, break`
                for_end_s.set_as(0)

    def _move_result_not_align(self, nloc_i_res, out_i_offset):
        """
        move result to gm, last_layer_size is not align.
        """
        tail = self.last_layer_size % self.block_elems
        last_layer_size_pad = aligned_value(self.last_layer_size, self.block_elems)
        # unroll
        for row in range(3):
            for idx in range(tail):
                nloc_i_res[row * self.last_size_align + self.last_layer_size + idx].set_as(
                    nloc_i_res[(row + 1) * self.last_size_align + idx])

        # last 1 block
        block_ub = self.tik_inst.Tensor(self.dtype, (self.block_elems,), name="block_ub", scope=tik.scope_ubuf)
        last_block_start = self.last_size_align * self.NUM_3 + self.last_layer_size - self.block_elems
        for i in range(0, 8):
            block_ub[i].set_as(nloc_i_res[last_block_start + i])

        with self.tik_inst.new_stmt_scope(disable_sync=True):
            self.tik_inst.data_move(self.descriptor_gm[out_i_offset], nloc_i_res, 0, 1,
                                    last_layer_size_pad // self.block_elems, 0, 0)
            self.tik_inst.data_move(self.descriptor_gm[out_i_offset + self.last_layer_size],
                                    nloc_i_res[self.last_size_align], 0, 1,
                                    last_layer_size_pad // self.block_elems, 0, 0)
            self.tik_inst.data_move(self.descriptor_gm[out_i_offset + self.last_layer_size * 2],
                                    nloc_i_res[self.last_size_align * 2], 0, 1,
                                    last_layer_size_pad // self.block_elems, 0, 0)
        self.tik_inst.data_move(self.descriptor_gm[out_i_offset + self.last_layer_size * 3],
                                nloc_i_res[self.last_size_align * 3], 0, 1,
                                self.last_layer_size // self.block_elems, 0, 0)
        self.tik_inst.data_move(
            self.descriptor_gm[out_i_offset + self.last_layer_size * self.NUM_4 - self.block_elems],
            block_ub, 0, 1, 1, 0, 0)

    def _move_result_with_v4dtrans(self, out_i, nloc_i_res, out_i_offset):
        """
        move result to gm, last_layer_size is not align, by using v4dtrans
        """
        self.tik_inst.v4dtrans(False, out_i, nloc_i_res, self.NUM_8, self.last_size_align)
        self.tik_inst.v4dtrans(True, nloc_i_res, out_i, self.NUM_8, self.last_layer_size)

        # move result to gm
        if self.last_layer_size % self.NUM_2 == 0:
            self.tik_inst.data_move(self.descriptor_gm[out_i_offset], nloc_i_res, 0, 1,
                                    self.last_layer_size * self.NUM_4 // self.block_elems, 0, 0)
        else:
            # store last 1 block
            block_ub = self.tik_inst.Tensor(self.dtype, (self.block_elems,), name="block_ub",
                                            scope=tik.scope_ubuf)
            last_size_start = self.last_size_align * self.NUM_3 + self.last_layer_size - self.block_elems
            # unroll
            for i in range(0, 8):
                block_ub[i].set_as(nloc_i_res[last_size_start + i])

            self.tik_inst.data_move(self.descriptor_gm[out_i_offset], nloc_i_res, 0, 1,
                                    self.last_layer_size * self.NUM_4 // self.block_elems - 1, 0, 0)
            self.tik_inst.data_move(
                self.descriptor_gm[out_i_offset + self.last_layer_size * self.NUM_4 - self.block_elems],
                block_ub, 0, 1, 1, 0, 0)

    def _cal_output_not_align(self, nloc_i_res, res_size, var_ub, ll_values, ago, xx, out_i_offset, nnei_j, for_end_s):
        """
        calculate result of nloc_i, last_layer_size is not 32 bytes align.
        res_size is self.last_size_align * self.NUM_4.
        ll_values is (ll_0, ll_1, ll_2, ll_3).
        """
        ll_0, ll_1, ll_2, ll_3 = ll_values
        out_i = self.tik_inst.Tensor(self.dtype, (res_size,), name="out_i", scope=tik.scope_ubuf)
        ll_repeats = ceiling_value(self.last_size_align, self.vector_elems)
        self.tik_inst.vmuls(self.NUM_64, out_i[0], var_ub, ll_0, ll_repeats, 1, 1, 8, 8)
        self.tik_inst.vmuls(self.NUM_64, out_i[self.last_size_align], var_ub, ll_1, ll_repeats, 1, 1, 8, 8)
        self.tik_inst.vmuls(self.NUM_64, out_i[self.last_size_align * self.NUM_2], var_ub, ll_2,
                            ll_repeats, 1, 1, 8, 8)
        self.tik_inst.vmuls(self.NUM_64, out_i[self.last_size_align * self.NUM_3], var_ub, ll_3,
                            ll_repeats, 1, 1, 8, 8)

        out_repeats = ceiling_value(self.last_size_align * self.NUM_4, self.vector_elems)
        with self.tik_inst.if_scope(ago != xx):
            self.tik_inst.vadd(self.NUM_64, nloc_i_res, nloc_i_res, out_i, out_repeats, 1, 1, 1, 8, 8, 8)
        with self.tik_inst.else_scope():
            # `out_i * (nnei - nnei_j)`
            self.tik_inst.vmuls(self.NUM_64, out_i, out_i, self.nnei - nnei_j, out_repeats, 1, 1, 8, 8)
            self.tik_inst.vadd(self.NUM_64, nloc_i_res, nloc_i_res, out_i, out_repeats, 1, 1, 1, 8, 8, 8)

            if tbe_platform.api_check_support("tik.v4dtrans", "float32"):
                self._move_result_with_v4dtrans(out_i, nloc_i_res, out_i_offset)
            else:
                self._move_result_not_align(nloc_i_res, out_i_offset)

            # `ago==xx, break`
            for_end_s.set_as(0)

    def _one_loop_compute(self, core_id, loop_i, nloc_num, elems_align, nloc_offset):
        """
        one loop compute
        """
        em_x_ub = self.tik_inst.Tensor(self.dtype, (self.one_portion_ub_elems,), name="em_x_ub", scope=tik.scope_ubuf)
        xx_new_ub = self.tik_inst.Tensor(self.dtype, (self.one_portion_ub_elems,), name="xx_new_ub",
                                         scope=tik.scope_ubuf)
        table_idx_ub = self.tik_inst.Tensor(self.dtype_int32, (self.one_portion_ub_elems,), name="table_idx_ub",
                                            scope=tik.scope_ubuf)
        ub_tensors = (em_x_ub, xx_new_ub, table_idx_ub)

        # calculate table_idx and xx_new
        self._locate_xx(ub_tensors, nloc_offset, elems_align)

        with self.tik_inst.for_range(0, nloc_num) as nloc_i:
            xx_base_offset = nloc_i * self.nnei
            ago = self.tik_inst.Scalar(dtype=self.dtype, name="ago")
            ago.set_as(em_x_ub[xx_base_offset + self.nnei - 1])

            # last_layer_size is 32 bytes align
            if self.last_layer_size % self.block_elems == 0:
                res_size = aligned_value(self.NUM_4 * self.last_layer_size, self.vector_elems) + self.vector_elems
                dup_size = res_size
            else:
                res_size = self.NUM_8 * self.last_size_align
                # in the first 4 rows of data, the first last_layer_size elements of each row are valid data
                dup_size = self.NUM_4 * self.last_size_align
            nloc_i_res = self.tik_inst.Tensor(self.dtype, (res_size,), name="nloc_i_res", scope=tik.scope_ubuf)
            self.tik_inst.vector_dup(self.NUM_64, nloc_i_res, 0, dup_size // self.vector_elems, 1, 8)

            # em is (nloc, nnei, 4), move nnei*4 every nloc to em_ub
            em_ub = self.tik_inst.Tensor(self.dtype, (self.one_portion_ub_elems,),
                                         name="em_ub", scope=tik.scope_ubuf)
            nloc_i_offset = core_id * self.nloc_one_core + loop_i * self.nloc_per_loop + nloc_i
            em_offset = (nloc_offset + nloc_i) * (self.nnei * self.NUM_4)
            self.tik_inst.data_move(em_ub, self.em_gm[em_offset], 0, 1,
                                    ceiling_value(self.nnei * self.NUM_4, self.block_elems), 0, 0)

            out_i_offset = nloc_i_offset * (self.NUM_4 * self.last_layer_size)
            for_end_s = self.tik_inst.Scalar(dtype=self.dtype_int32, name="for_end_s")
            for_end_s.set_as(self.nnei)
            with self.tik_inst.for_range(0, for_end_s) as nnei_j:
                xx = self.tik_inst.Scalar(dtype=self.dtype, name="xx")
                xx.set_as(em_x_ub[xx_base_offset + nnei_j])

                # calculate var
                var_ub = self.tik_inst.Tensor(self.dtype, (self.last_size_align,), name="var_ub", scope=tik.scope_ubuf)
                self._cal_var(var_ub, xx_new_ub, table_idx_ub, xx_base_offset + nnei_j)

                ll_0 = self.tik_inst.Scalar(dtype=self.dtype, name="ll_0")
                ll_1 = self.tik_inst.Scalar(dtype=self.dtype, name="ll_1")
                ll_2 = self.tik_inst.Scalar(dtype=self.dtype, name="ll_2")
                ll_3 = self.tik_inst.Scalar(dtype=self.dtype, name="ll_3")
                ll_0_index = nnei_j * self.NUM_4
                ll_0.set_as(em_ub[ll_0_index])
                ll_1.set_as(em_ub[ll_0_index + self.NUM_1])
                ll_2.set_as(em_ub[ll_0_index + self.NUM_2])
                ll_3.set_as(em_ub[ll_0_index + self.NUM_3])
                # calculate out
                if self.last_layer_size % self.block_elems == 0:
                    self._cal_output(nloc_i_res, res_size, var_ub, (ll_0, ll_1, ll_2, ll_3),
                                     ago, xx, out_i_offset, nnei_j, for_end_s)
                else:
                    self._cal_output_not_align(nloc_i_res, res_size, var_ub, (ll_0, ll_1, ll_2, ll_3),
                                               ago, xx, out_i_offset, nnei_j, for_end_s)

    def _pre_core_compute(self, core_id):
        """
        compute for one pre core
        """
        elems_align = aligned_value(self.nloc_per_loop * self.nnei, self.vector_elems)
        with self.tik_inst.for_range(0, self.pre_core_loops) as loop_i:
            with self.tik_inst.new_stmt_scope():
                # process self.nloc_per_loop per loop
                nloc_offset = self.nloc_engine_offset + (core_id * self.nloc_one_core + loop_i * self.nloc_per_loop)
                self._one_loop_compute(core_id, loop_i, self.nloc_per_loop, elems_align, nloc_offset)

        with self.tik_inst.if_scope(self.pre_core_nloc_tail > 0):
            with self.tik_inst.new_stmt_scope():
                # process self.pre_core_nloc_tail
                elems_tail_align = aligned_value(self.pre_core_nloc_tail * self.nnei, self.vector_elems)
                nloc_offset = self.nloc_engine_offset + (core_id * self.nloc_one_core +
                                                         self.pre_core_loops * self.nloc_per_loop)
                self._one_loop_compute(core_id, self.pre_core_loops, self.pre_core_nloc_tail, elems_tail_align,
                                       nloc_offset)

    def _last_core_compute(self, core_id):
        """
        compute for last core
        """
        elems_align = aligned_value(self.nloc_per_loop * self.nnei, self.vector_elems)
        with self.tik_inst.for_range(0, self.last_core_loops) as loop_i:
            with self.tik_inst.new_stmt_scope():
                # process self.nloc_per_loop per loop for last core
                nloc_offset = self.nloc_engine_offset + (core_id * self.nloc_one_core + loop_i * self.nloc_per_loop)
                self._one_loop_compute(core_id, loop_i, self.nloc_per_loop, elems_align, nloc_offset)

        with self.tik_inst.if_scope(self.last_core_nloc_tail > 0):
            with self.tik_inst.new_stmt_scope():
                # process self.last_core_nloc_tail for last core
                elems_tail_align = aligned_value(self.last_core_nloc_tail * self.nnei, self.vector_elems)
                nloc_offset = self.nloc_engine_offset + (core_id * self.nloc_one_core +
                                                         self.last_core_loops * self.nloc_per_loop)
                self._one_loop_compute(core_id, self.last_core_loops, self.last_core_nloc_tail, elems_tail_align,
                                       nloc_offset)


def _check_input_params(table, table_info, em_x, em, descriptor, last_layer_size, split_count, split_index,
                        kernel_name):
    """
    check input parameters
    """
    if last_layer_size < 0:
        rule = "last_layer_size should be greater than 0"
        error_manager_vector.raise_err_specific_reson(kernel_name, rule)
    if any((split_count < 1, split_count > TabulateFusion.NUM_2, split_index < 0, split_count <= split_index)):
        rule = "Failed to check split info"
        error_manager_vector.raise_err_specific_reson(kernel_name, rule)

    table_dtype = table.get("dtype").lower()
    table_info_dtype = table_info.get("dtype").lower()
    em_x_dtype = em_x.get("dtype").lower()
    em_dtype = em.get("dtype").lower()
    descriptor_dtype = descriptor.get("dtype").lower()

    check_list = ("float32",)
    para_check.check_dtype(table_dtype, check_list, param_name="table")
    para_check.check_dtype(table_info_dtype, check_list, param_name="table_info")
    para_check.check_dtype(em_x_dtype, check_list, param_name="em_x")
    para_check.check_dtype(em_dtype, check_list, param_name="em")
    para_check.check_dtype(descriptor_dtype, check_list, param_name="descriptor")


@register_operator("TabulateFusion")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_INT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def tabulate_fusion(table, table_info, em_x, em, descriptor, last_layer_size, split_count=1, split_index=0,
                    kernel_name="tabulate_fusion"):
    """
    TabulateFusion op

    Parameters
    ----------
    table: dict
        the dict of input tensor, 2D, the dim 2 is last_layer_size*6.
    table_info: dict
        the dict of input size_splits tensor, 1D.
    em_x: dict
        the dict of input split_dim tensor, 2D, [nloc * nnei, 1].
    em: dict
        the dict of input split_dim tensor, 3D, [nloc, nnei, 4].
    descriptor: dict
        the list of output tensor, 3D, [nloc, 4, last_layer_size].
    last_layer_size: int
        an integer indicating the last layer size of output.
    split_count: int
        an optional attr, default value is 1.
        1: aicore, 2: aicore + vectorcore.
    split_index: int
        an optional attr, default value is 0.
        0: aicore, 1: vectorcore.
    kernel_name: str
        cce kernel name, default value is "tabulate_fusion".

    Returns
    -------
    tik_instance
    """
    _check_input_params(table, table_info, em_x, em, descriptor, last_layer_size, split_count, split_index,
                        kernel_name)

    obj = TabulateFusion(table, table_info, em_x, em, descriptor, last_layer_size)
    obj.tabulate_fusion_compute()

    # add compile info
    tbe_context.get_context().add_compile_info("vars", {
        "core_num": obj.core_num,
        "last_layer_size": obj.last_layer_size,
        "one_portion_ub_elems": obj.one_portion_ub_elems,
        "split_count": split_count,
        "split_index": split_index
    })

    tik_inst = obj.tik_inst
    tik_inst.BuildCCE(kernel_name=kernel_name,
                      inputs=(obj.table_gm, obj.table_info_gm, obj.em_x_gm, obj.em_gm),
                      outputs=(obj.descriptor_gm,),
                      flowtable=(obj.tiling_gm,))

    return tik_inst
