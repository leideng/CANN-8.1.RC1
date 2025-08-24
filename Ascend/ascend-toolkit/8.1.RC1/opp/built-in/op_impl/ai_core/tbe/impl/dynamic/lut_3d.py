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
lut_3d
"""
from impl import constant_util as constant
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context


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


class LUT3D:
    """
    LUT3D class
    """

    # int32's max value
    MAX_SHAPE_SIZE = constant.SHAPE_SIZE_LIMIT
    # tiling param num
    TILING_ARG_NUM = 16
    # reserved ub size
    RESERVED_UB_SIZE = 24 * 1024
    # 8 bit
    EIGHT_BIT = 8
    # bytes of one block
    BLOCK_BYTES = constant.BLOCK_SIZE
    # 256 bytes
    VECTOR_BYTES = constant.VECTOR_BYTE_SIZE
    ONE_PIXEL_ELEMS = 3
    ONE_PIXEL_UB_WORKSPACE_BYTES = 222

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
    NUM_11 = 11
    NUM_12 = 12
    VECTOR_32_MASK = 64
    VECTOR_16_MASK = 128

    def __init__(self, img, lut_table, lut_img):
        """
        Init LUT3D parameters
        """
        self.tik_inst = tik.Tik()
        self.dtype = img.get("dtype").lower()
        self.shape_img = img.get("shape")
        self.shape_lut_table = lut_table.get("shape")
        self.lut_table_dtype = "float32"
        self.lut_table_dsize = tbe_platform.get_bit_len(self.lut_table_dtype) // self.EIGHT_BIT
        self.lut_table_ub = None

        self.dsize = tbe_platform.get_bit_len(self.dtype) // self.EIGHT_BIT
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - self.RESERVED_UB_SIZE
        self.block_elems = self.BLOCK_BYTES // self.dsize
        self.vector_elems = self.VECTOR_BYTES // self.dsize

        self.tiling_dtype = "int64"
        self.tiling_align = aligned_value(self.TILING_ARG_NUM, self.NUM_4)
        self.tiling_gm = self.tik_inst.Tensor(self.tiling_dtype, (self.tiling_align,), name="tiling_gm",
                                              scope=tik.scope_gm)
        self.img_gm, self.lut_table_gm, self.lut_img_gm = self._init_gm_tensor()

        # tiling params
        self.need_core_num = None
        self.input_img_elems = None
        self.input_lut_table_elems = None
        self.input_img_pixels = None
        self.lut_table_ub_elems = None
        self.lut_table_n = None
        self.one_portion_ub_pixels = None
        self.pre_core_pixels = None
        self.last_core_pixels = None
        self.pre_core_loops = None
        self.last_core_loops = None
        self.pre_core_pixels_tail = None
        self.last_core_pixels_tail = None

    def lut_3d_compute(self):
        """
        main process of lut_3d
        """
        with self.tik_inst.for_range(0, self.core_num, block_num=self.core_num) as core_i:
            # get tiling data
            self._get_tiling_args()
            self._set_lut_table()
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
        img_gm = self.tik_inst.Tensor(self.dtype, (self.MAX_SHAPE_SIZE,), name="img_gm", scope=tik.scope_gm)
        lut_table_gm = self.tik_inst.Tensor(self.dtype, (self.MAX_SHAPE_SIZE,), name="lut_table_gm",
                                            scope=tik.scope_gm)
        lut_img_gm = self.tik_inst.Tensor("float32", (self.MAX_SHAPE_SIZE,), name="lut_img_gm",
                                          scope=tik.scope_gm)

        return [img_gm, lut_table_gm, lut_img_gm]

    def _get_tiling_args(self):
        """
        get runtime params from tiling data
        """
        self.need_core_num = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="need_core_num")
        self.input_img_elems = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="input_img_elems")
        self.input_lut_table_elems = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="input_lut_table_elems")
        self.input_img_pixels = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="input_img_pixels")
        self.lut_table_ub_elems = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="lut_table_ub_elems")
        self.lut_table_n = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="lut_table_n")
        self.one_portion_ub_pixels = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="one_portion_ub_pixels")
        self.pre_core_pixels = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="pre_core_pixels")
        self.last_core_pixels = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="last_core_pixels")
        self.pre_core_loops = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="pre_core_loops")
        self.last_core_loops = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="last_core_loops")
        self.pre_core_pixels_tail = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="pre_core_pixels_tail")
        self.last_core_pixels_tail = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="last_core_pixels_tail")

        with self.tik_inst.new_stmt_scope():
            tiling_ub = self.tik_inst.Tensor(self.tiling_dtype, (self.tiling_align,), name="tiling_ub",
                                             scope=tik.scope_ubuf)
            self.tik_inst.data_move(tiling_ub, self.tiling_gm, 0, 1, self.tiling_align // self.NUM_4, 0, 0)

            self.need_core_num.set_as(tiling_ub[self.NUM_0])
            self.input_img_elems.set_as(tiling_ub[self.NUM_1])
            self.input_lut_table_elems.set_as(tiling_ub[self.NUM_2])
            self.input_img_pixels.set_as(tiling_ub[self.NUM_3])
            self.lut_table_ub_elems.set_as(tiling_ub[self.NUM_4])
            self.lut_table_n.set_as(tiling_ub[self.NUM_5])
            self.one_portion_ub_pixels.set_as(tiling_ub[self.NUM_6])
            self.pre_core_pixels.set_as(tiling_ub[self.NUM_7])
            self.last_core_pixels.set_as(tiling_ub[self.NUM_8])
            self.pre_core_loops.set_as(tiling_ub[self.NUM_9])
            self.last_core_loops.set_as(tiling_ub[self.NUM_10])
            self.pre_core_pixels_tail.set_as(tiling_ub[self.NUM_11])
            self.last_core_pixels_tail.set_as(tiling_ub[self.NUM_12])

    def _set_lut_table(self):
        self.lut_table_ub = self.tik_inst.Tensor(self.lut_table_dtype, (self.lut_table_ub_elems,), name="lut_table_ub",
                                                 scope=tik.scope_ubuf)
        if self.dtype == "uint8":
            lut_table_ub_uint8 = self.tik_inst.Tensor("uint8", (self.lut_table_ub_elems,), name="lut_table_ub_uint8",
                                                      scope=tik.scope_ubuf)
            lut_table_ub_fp16 = self.tik_inst.Tensor("float16", (self.lut_table_ub_elems,), name="lut_table_ub_fp16",
                                                     scope=tik.scope_ubuf)
            self.tik_inst.data_move(lut_table_ub_uint8, self.lut_table_gm, 0, 1,
                                    ceiling_value(self.lut_table_ub_elems, self.block_elems), 0, 0)
            self.tik_inst.vconv(self.VECTOR_16_MASK, "none", lut_table_ub_fp16, lut_table_ub_uint8,
                                self.lut_table_ub_elems // self.VECTOR_16_MASK, 1, 1, 8, 4)
            with self.tik_inst.if_scope((self.lut_table_ub_elems % self.VECTOR_16_MASK) != 0):
                offset = self.lut_table_ub_elems // self.VECTOR_16_MASK * self.VECTOR_16_MASK
                self.tik_inst.vconv(self.lut_table_ub_elems % self.VECTOR_16_MASK, "none", lut_table_ub_fp16[offset],
                                    lut_table_ub_uint8[offset], 1, 1, 1, 8, 4)
            self.tik_inst.vconv(self.VECTOR_32_MASK, "none", self.lut_table_ub, lut_table_ub_fp16,
                                self.lut_table_ub_elems // self.VECTOR_32_MASK, 1, 1, 8, 4)
            with self.tik_inst.if_scope((self.lut_table_ub_elems % self.VECTOR_32_MASK) != 0):
                self.tik_inst.vconv(self.lut_table_ub_elems % self.VECTOR_32_MASK, "none", self.lut_table_ub[offset],
                                    lut_table_ub_fp16[offset], 1, 1, 1, 8, 4)
        elif self.dtype == "float32":
            self.tik_inst.data_move(self.lut_table_ub, self.lut_table_gm, 0, 1,
                                    ceiling_value(self.lut_table_ub_elems, self.block_elems), 0, 0)

    def _pre_core_compute(self, core_id):
        """
        compute for one pre core
        """
        with self.tik_inst.for_range(0, self.pre_core_loops) as loop_i:
            with self.tik_inst.if_scope(loop_i == 0):
                with self.tik_inst.new_stmt_scope():
                    move_offset = core_id * self.pre_core_pixels * self.ONE_PIXEL_ELEMS
                    self._one_loop_compute(self.one_portion_ub_pixels, move_offset)

            with self.tik_inst.if_scope(loop_i > 0):
                with self.tik_inst.new_stmt_scope():
                    move_offset = (core_id * self.pre_core_pixels + (
                            loop_i - 1) * self.one_portion_ub_pixels + self.pre_core_pixels_tail) * self.ONE_PIXEL_ELEMS
                    self._one_loop_compute(self.one_portion_ub_pixels, move_offset)

    def _last_core_compute(self, core_id):
        """
        compute for one last core
        """
        with self.tik_inst.for_range(0, self.pre_core_loops) as loop_i:
            with self.tik_inst.if_scope(loop_i == 0):
                with self.tik_inst.new_stmt_scope():
                    move_offset = core_id * self.pre_core_pixels * self.ONE_PIXEL_ELEMS
                    self._one_loop_compute(self.one_portion_ub_pixels, move_offset)

            with self.tik_inst.if_scope(loop_i > 0):
                with self.tik_inst.new_stmt_scope():
                    move_offset = (core_id * self.pre_core_pixels + (loop_i - 1)
                                   * self.one_portion_ub_pixels + self.last_core_pixels_tail) * self.ONE_PIXEL_ELEMS
                    self._one_loop_compute(self.one_portion_ub_pixels, move_offset)

    def _one_loop_compute(self, work_pixels, move_offset):
        compute_space = LUTUBComputeSpace(self, work_pixels)
        img_ub_fp32 = self._set_img_ub(move_offset, work_pixels)

        self._trans2chw(compute_space, img_ub_fp32, work_pixels)

        self.tik_inst.vconv(self.VECTOR_32_MASK, "none", compute_space.tensor_img_fl_trans_fp32,
                            compute_space.tensor_img_fl_trans,
                            ceiling_value(work_pixels * self.ONE_PIXEL_ELEMS, self.VECTOR_32_MASK), 1, 1, 8, 8)
        self.tik_inst.vconv(self.VECTOR_32_MASK, "none", compute_space.tensor_img_cl_trans_fp32,
                            compute_space.tensor_img_cl_trans,
                            ceiling_value(work_pixels * self.ONE_PIXEL_ELEMS, self.VECTOR_32_MASK), 1, 1, 8, 8)

        self._cal_channel_index(compute_space, work_pixels)
        self._cal_channel_weights(compute_space, work_pixels)

        lut_output = compute_space.tensor_img_trans
        # for in (r,g,b)
        with self.tik_inst.for_range(0, self.ONE_PIXEL_ELEMS) as c_offset:
            lut_tensors = self._get_lut_tensor(compute_space, c_offset, work_pixels)
            self._cal_lut_output(compute_space, lut_tensors, lut_output, c_offset, work_pixels)

        # Trans lut_output format chw to hwc img_ub_fp32
        support_v4dtrans = tbe_platform.api_check_support("tik.v4dtrans", "float32")
        if support_v4dtrans:
            chw2hwc = True
            self.tik_inst.v4dtrans(chw2hwc, img_ub_fp32, lut_output, work_pixels, 3)
        else:
            with self.tik_inst.for_range(0, work_pixels) as i:
                img_ub_fp32[i * self.ONE_PIXEL_ELEMS].set_as(lut_output[i])
                img_ub_fp32[i * self.ONE_PIXEL_ELEMS + 1].set_as(lut_output[work_pixels + i])
                img_ub_fp32[i * self.ONE_PIXEL_ELEMS + 2].set_as(lut_output[2 * work_pixels + i])
        with self.tik_inst.if_scope(work_pixels > self.input_img_pixels):
            move_burst = ceiling_value(self.input_img_pixels * self.ONE_PIXEL_ELEMS * self.lut_table_dsize,
                                       self.block_elems)
            self.tik_inst.data_move(self.lut_img_gm[move_offset], img_ub_fp32, 0, 1, move_burst, 0, 0)
        with self.tik_inst.else_scope():
            move_burst = ceiling_value(work_pixels * self.ONE_PIXEL_ELEMS * self.lut_table_dsize,
                                       self.block_elems)
            self.tik_inst.data_move(self.lut_img_gm[move_offset], img_ub_fp32, 0, 1, move_burst, 0, 0)

    def _cal_lut_output(self, compute_space, lut_tensors, lut_output, c_offset, work_pixels):
        # `lut_tensor1 = lut_tensor1 * fract_r_1 + lut_tensor4 * fract_r`
        self.tik_inst.vmul(self.VECTOR_32_MASK, lut_tensors[self.NUM_0], lut_tensors[self.NUM_0],
                           compute_space.fract_r_1,
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        self.tik_inst.vmul(self.VECTOR_32_MASK, lut_tensors[self.NUM_3], lut_tensors[self.NUM_3], compute_space.fract_r,
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(self.VECTOR_32_MASK, lut_tensors[self.NUM_0], lut_tensors[self.NUM_0],
                           lut_tensors[self.NUM_3],
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        # `lut_tensor3 = lut_tensor3 * fract_r_1 + lut_tensor5 * fract_r`
        self.tik_inst.vmul(self.VECTOR_32_MASK, lut_tensors[self.NUM_2], lut_tensors[self.NUM_2],
                           compute_space.fract_r_1,
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        self.tik_inst.vmul(self.VECTOR_32_MASK, lut_tensors[self.NUM_4], lut_tensors[self.NUM_4], compute_space.fract_r,
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(self.VECTOR_32_MASK, lut_tensors[self.NUM_2], lut_tensors[self.NUM_2],
                           lut_tensors[self.NUM_4],
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        # `lut_tensor2 =lut_tensor2 * fract_r_1 + lut_tensor6 * fract_r`
        self.tik_inst.vmul(self.VECTOR_32_MASK, lut_tensors[self.NUM_1], lut_tensors[self.NUM_1],
                           compute_space.fract_r_1,
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        self.tik_inst.vmul(self.VECTOR_32_MASK, lut_tensors[self.NUM_5], lut_tensors[self.NUM_5], compute_space.fract_r,
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(self.VECTOR_32_MASK, lut_tensors[self.NUM_1], lut_tensors[self.NUM_1],
                           lut_tensors[self.NUM_5],
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        # `lut_tensor7 = lut_tensor7 * fract_r_1 + lut_tensor8 * fract_r`
        self.tik_inst.vmul(self.VECTOR_32_MASK, lut_tensors[self.NUM_6], lut_tensors[self.NUM_6],
                           compute_space.fract_r_1,
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        self.tik_inst.vmul(self.VECTOR_32_MASK, lut_tensors[self.NUM_7], lut_tensors[self.NUM_7], compute_space.fract_r,
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(self.VECTOR_32_MASK, lut_tensors[self.NUM_6], lut_tensors[self.NUM_6],
                           lut_tensors[self.NUM_7],
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        # `lut_tensor1 = lut_tensor1 * fract_g_1 + lut_tensor3 * fract_g`
        self.tik_inst.vmul(self.VECTOR_32_MASK, lut_tensors[self.NUM_0], lut_tensors[self.NUM_0],
                           compute_space.fract_g_1,
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        self.tik_inst.vmul(self.VECTOR_32_MASK, lut_tensors[self.NUM_2], lut_tensors[self.NUM_2], compute_space.fract_g,
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(self.VECTOR_32_MASK, lut_tensors[self.NUM_0], lut_tensors[self.NUM_0],
                           lut_tensors[self.NUM_2],
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        # `lut_tensor2 = lut_tensor2 * fract_g_1 + lut_tensor7 * fract_g`
        self.tik_inst.vmul(self.VECTOR_32_MASK, lut_tensors[self.NUM_1], lut_tensors[self.NUM_1],
                           compute_space.fract_g_1,
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        self.tik_inst.vmul(self.VECTOR_32_MASK, lut_tensors[self.NUM_6], lut_tensors[self.NUM_6], compute_space.fract_g,
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(self.VECTOR_32_MASK, lut_tensors[self.NUM_1], lut_tensors[self.NUM_1],
                           lut_tensors[self.NUM_6],
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        # `lut_output = lut_tensor1 * fract_r_1 + lut_tensor2 * fract_r`
        self.tik_inst.vmul(self.VECTOR_32_MASK, lut_tensors[self.NUM_0], lut_tensors[self.NUM_0],
                           compute_space.fract_b_1,
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        self.tik_inst.vmul(self.VECTOR_32_MASK, lut_tensors[self.NUM_1], lut_tensors[self.NUM_1], compute_space.fract_b,
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(self.VECTOR_32_MASK, lut_output[work_pixels * c_offset], lut_tensors[self.NUM_0],
                           lut_tensors[self.NUM_1],
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)

    def _get_lut_tensor(self, compute_space, c_offset, work_pixels):
        """
        Calculate lut_tensor.
        lut_indices[i] = b_index + g_index + r_index
        lut_tensors[i] = lut_table[lut_indices[i]]
        """
        lut_tensors = []
        lut_indices = []
        for i in range(self.NUM_8):
            lut_tensor = self.tik_inst.Tensor("float32", (work_pixels,), tik.scope_ubuf, "lut_tensors_" + str(i))
            lut_index = self.tik_inst.Tensor("float32", (work_pixels,), tik.scope_ubuf, "lut_indices_" + str(i))
            lut_tensors.append(lut_tensor)
            lut_indices.append(lut_index)
        self.tik_inst.vadd(self.VECTOR_32_MASK, lut_indices[self.NUM_0], compute_space.b_floor_index,
                           compute_space.g_floor_index,
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(self.VECTOR_32_MASK, lut_indices[self.NUM_0], lut_indices[self.NUM_0],
                           compute_space.tensor_img_fl_trans_fp32[work_pixels * 2],
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(self.VECTOR_32_MASK, lut_indices[self.NUM_1], compute_space.b_ceil_index,
                           compute_space.g_floor_index,
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(self.VECTOR_32_MASK, lut_indices[self.NUM_1], lut_indices[self.NUM_1],
                           compute_space.tensor_img_fl_trans_fp32[work_pixels * 2],
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(self.VECTOR_32_MASK, lut_indices[self.NUM_2], compute_space.b_floor_index,
                           compute_space.g_ceil_index,
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(self.VECTOR_32_MASK, lut_indices[self.NUM_2], lut_indices[self.NUM_2],
                           compute_space.tensor_img_fl_trans_fp32[work_pixels * 2],
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(self.VECTOR_32_MASK, lut_indices[self.NUM_3], compute_space.b_floor_index,
                           compute_space.g_floor_index,
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(self.VECTOR_32_MASK, lut_indices[self.NUM_3], lut_indices[self.NUM_3],
                           compute_space.tensor_img_cl_trans_fp32[work_pixels * 2],
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(self.VECTOR_32_MASK, lut_indices[self.NUM_4], compute_space.b_floor_index,
                           compute_space.g_ceil_index,
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(self.VECTOR_32_MASK, lut_indices[self.NUM_4], lut_indices[self.NUM_4],
                           compute_space.tensor_img_cl_trans_fp32[work_pixels * 2],
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(self.VECTOR_32_MASK, lut_indices[self.NUM_5], compute_space.b_ceil_index,
                           compute_space.g_floor_index,
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(self.VECTOR_32_MASK, lut_indices[self.NUM_5], lut_indices[self.NUM_5],
                           compute_space.tensor_img_cl_trans_fp32[work_pixels * 2],
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(self.VECTOR_32_MASK, lut_indices[self.NUM_6], compute_space.b_ceil_index,
                           compute_space.g_ceil_index,
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(self.VECTOR_32_MASK, lut_indices[self.NUM_6], lut_indices[self.NUM_6],
                           compute_space.tensor_img_fl_trans_fp32[work_pixels * 2],
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(self.VECTOR_32_MASK, lut_indices[self.NUM_7], compute_space.b_ceil_index,
                           compute_space.g_ceil_index,
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(self.VECTOR_32_MASK, lut_indices[self.NUM_7], lut_indices[self.NUM_7],
                           compute_space.tensor_img_cl_trans_fp32[work_pixels * 2],
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)

        lut_index = self.tik_inst.Tensor("int32", (work_pixels,), tik.scope_ubuf, "lut_index")
        index_scalar = self.tik_inst.Scalar(dtype="int32")

        for i in range(self.NUM_8):
            support_vgather = tbe_platform.api_check_support("tik.vgather")
            if support_vgather:
                self.tik_inst.vmuls(self.VECTOR_32_MASK, lut_indices[i], lut_indices[i],
                                    self.lut_table_dsize * self.ONE_PIXEL_ELEMS,
                                    ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 8, 8)
                self.tik_inst.vadds(self.VECTOR_32_MASK, lut_indices[i], lut_indices[i],
                                    self.lut_table_dsize * c_offset,
                                    ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 8, 8)
                self.tik_inst.vconv(self.VECTOR_32_MASK, "round", lut_index, lut_indices[i],
                                    ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 8, 8)
                self.tik_inst.vgather(self.VECTOR_32_MASK, lut_tensors[i], self.lut_table_ub, lut_index,
                                      ceiling_value(work_pixels, self.VECTOR_32_MASK), 8)
            else:
                self.tik_inst.vmuls(self.VECTOR_32_MASK, lut_indices[i], lut_indices[i], self.ONE_PIXEL_ELEMS,
                                    ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 8, 8)
                self.tik_inst.vconv(self.VECTOR_32_MASK, "round", lut_index, lut_indices[i],
                                    ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 8, 8)
                with self.tik_inst.for_range(0, work_pixels) as index:
                    index_scalar.set_as(lut_index[index])
                    lut_tensors[i][index].set_as(self.lut_table_ub[index_scalar + c_offset])

        return lut_tensors

    def _cal_channel_weights(self, compute_space, work_pixels):
        """
        Calculate fract_b fract_b_1 fract_g fract_g_1 fract_r fract_r_1.
        """
        self.tik_inst.vsub(self.VECTOR_32_MASK, compute_space.fract_b, compute_space.tensor_img_trans,
                           compute_space.tensor_img_fl_trans_fp32,
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        self.tik_inst.vsub(self.VECTOR_32_MASK, compute_space.fract_g, compute_space.tensor_img_trans[work_pixels],
                           compute_space.tensor_img_fl_trans_fp32[work_pixels],
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        self.tik_inst.vsub(self.VECTOR_32_MASK, compute_space.fract_r, compute_space.tensor_img_trans[work_pixels * 2],
                           compute_space.tensor_img_fl_trans_fp32[work_pixels * 2],
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadds(self.VECTOR_32_MASK, compute_space.fract_b_1, compute_space.fract_b, -1,
                            ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 8, 8)
        self.tik_inst.vadds(self.VECTOR_32_MASK, compute_space.fract_g_1, compute_space.fract_g, -1,
                            ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 8, 8)
        self.tik_inst.vadds(self.VECTOR_32_MASK, compute_space.fract_r_1, compute_space.fract_r, -1,
                            ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 8, 8)
        self.tik_inst.vabs(self.VECTOR_32_MASK, compute_space.fract_b_1, compute_space.fract_b_1,
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 8, 8)
        self.tik_inst.vabs(self.VECTOR_32_MASK, compute_space.fract_g_1, compute_space.fract_g_1,
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 8, 8)
        self.tik_inst.vabs(self.VECTOR_32_MASK, compute_space.fract_r_1, compute_space.fract_r_1,
                           ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 8, 8)

    def _cal_channel_index(self, compute_space, work_pixels):
        """
        Calculate b_floor_index g_floor_index b_ceil_index g_ceil_index.
        """
        lut_table_n = self.tik_inst.Scalar(dtype="int32")
        lut_table_n_fp32 = self.tik_inst.Scalar(dtype="float32")
        lut_table_n.set_as(self.lut_table_n)
        self.tik_inst.scalar_conv("none", lut_table_n_fp32, lut_table_n)
        self.tik_inst.vmuls(self.VECTOR_32_MASK, compute_space.b_floor_index, compute_space.tensor_img_fl_trans_fp32,
                            lut_table_n_fp32 * lut_table_n_fp32,
                            ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 8, 8)
        self.tik_inst.vmuls(self.VECTOR_32_MASK, compute_space.g_floor_index,
                            compute_space.tensor_img_fl_trans_fp32[work_pixels],
                            lut_table_n_fp32, ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 8, 8)
        self.tik_inst.vmuls(self.VECTOR_32_MASK, compute_space.b_ceil_index, compute_space.tensor_img_cl_trans_fp32,
                            lut_table_n_fp32 * lut_table_n_fp32,
                            ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 8, 8)
        self.tik_inst.vmuls(self.VECTOR_32_MASK, compute_space.g_ceil_index,
                            compute_space.tensor_img_cl_trans_fp32[work_pixels],
                            lut_table_n_fp32, ceiling_value(work_pixels, self.VECTOR_32_MASK), 1, 1, 8, 8)

    def _trans2chw(self, compute_space, img_ub_fp32, work_pixels):
        """
        Calculate img_fp32 = img_fp32 * (lut_table_n - 1) / 255.
        Calculate img_fp32 floor and ceil tensor_img_fl tensor_img_cl
        Trans img_fp32 format hwc to chw tensor_img_trans
        Trans tensor_img_fl format hwc to chw tensor_img_fl_trans
        Trans tensor_img_cl format hwc to chw tensor_img_cl_trans
        """
        table_value_it32 = self.tik_inst.Scalar(dtype="int32")
        table_value = self.tik_inst.Scalar(dtype="float32")
        table_value_it32.set_as(self.lut_table_n - 1)
        self.tik_inst.scalar_conv("none", table_value, table_value_it32)

        support_v4dtrans = tbe_platform.api_check_support("tik.v4dtrans", "float32")
        if support_v4dtrans:
            hwc2chw = False
            self.tik_inst.v4dtrans(hwc2chw, compute_space.tensor_img_trans, img_ub_fp32, work_pixels, 3)
        else:
            with self.tik_inst.for_range(0, work_pixels) as i:
                compute_space.tensor_img_trans[i].set_as(img_ub_fp32[i * self.ONE_PIXEL_ELEMS])
                compute_space.tensor_img_trans[work_pixels + i].set_as(img_ub_fp32[i * self.ONE_PIXEL_ELEMS + 1])
                compute_space.tensor_img_trans[2 * work_pixels + i].set_as(img_ub_fp32[i * self.ONE_PIXEL_ELEMS + 2])

        self.tik_inst.vmuls(self.VECTOR_32_MASK, compute_space.tensor_img_trans, compute_space.tensor_img_trans,
                            table_value / 255.,
                            ceiling_value(work_pixels * self.ONE_PIXEL_ELEMS, self.VECTOR_32_MASK), 1, 1, 8, 8)
        self.tik_inst.vconv(self.VECTOR_32_MASK, "floor", compute_space.tensor_img_fl_trans,
                            compute_space.tensor_img_trans,
                            ceiling_value(work_pixels * self.ONE_PIXEL_ELEMS, self.VECTOR_32_MASK), 1, 1, 8, 8)
        self.tik_inst.vconv(self.VECTOR_32_MASK, "ceil", compute_space.tensor_img_cl_trans,
                            compute_space.tensor_img_trans,
                            ceiling_value(work_pixels * self.ONE_PIXEL_ELEMS, self.VECTOR_32_MASK), 1, 1, 8, 8)
        self.tik_inst.vconv(self.VECTOR_32_MASK, "none", compute_space.tensor_img_fl_trans_fp32,
                            compute_space.tensor_img_fl_trans,
                            ceiling_value(work_pixels * self.ONE_PIXEL_ELEMS, self.VECTOR_32_MASK), 1, 1, 8, 8)
        self.tik_inst.vconv(self.VECTOR_32_MASK, "none", compute_space.tensor_img_cl_trans_fp32,
                            compute_space.tensor_img_cl_trans,
                            ceiling_value(work_pixels * self.ONE_PIXEL_ELEMS, self.VECTOR_32_MASK), 1, 1, 8, 8)

    def _set_img_ub(self, move_offset, work_pixels):
        img_ub_fp32 = self.tik_inst.Tensor("float32", (work_pixels * self.ONE_PIXEL_ELEMS,), name="img_ub_fp32",
                                           scope=tik.scope_ubuf)
        if self.dtype == "uint8":
            img_ub = self.tik_inst.Tensor(self.dtype, (work_pixels * self.ONE_PIXEL_ELEMS,), name="img_ub",
                                          scope=tik.scope_ubuf)
            img_ub_fp16 = self.tik_inst.Tensor("float16", (work_pixels * self.ONE_PIXEL_ELEMS,), name="img_ub_fp16",
                                               scope=tik.scope_ubuf)
            self.tik_inst.data_move(img_ub, self.img_gm[move_offset], 0, 1,
                                    ceiling_value(work_pixels * self.ONE_PIXEL_ELEMS, self.block_elems), 0, 0)
            self.tik_inst.vconv(self.VECTOR_16_MASK, "none", img_ub_fp16, img_ub,
                                ceiling_value(work_pixels * self.ONE_PIXEL_ELEMS, self.VECTOR_16_MASK), 1, 1, 8, 4)
            self.tik_inst.vconv(self.VECTOR_32_MASK, "none", img_ub_fp32, img_ub_fp16,
                                ceiling_value(work_pixels * self.ONE_PIXEL_ELEMS, self.VECTOR_32_MASK), 1, 1, 8, 4)
        elif self.dtype == "float32":
            self.tik_inst.data_move(img_ub_fp32, self.img_gm[move_offset], 0, 1,
                                    ceiling_value(work_pixels * self.ONE_PIXEL_ELEMS, self.block_elems), 0, 0)
        return img_ub_fp32


class LUTUBComputeSpace:
    tensor_img_fl = None
    tensor_img_cl = None
    tensor_img_trans = None
    tensor_img_fl_trans = None
    tensor_img_cl_trans = None
    tensor_img_fl_trans_fp32 = None
    tensor_img_cl_trans_fp32 = None
    fract_b = None
    fract_b_1 = None
    fract_g = None
    fract_g_1 = None
    fract_r = None
    fract_r_1 = None
    b_floor_index = None
    b_ceil_index = None
    g_floor_index = None
    g_ceil_index = None

    def __init__(self, lut_obj, work_pixels):
        self.tensor_img_fl = lut_obj.tik_inst.Tensor("int32", (work_pixels * lut_obj.ONE_PIXEL_ELEMS,),
                                                     name="tensor_img_fl",
                                                     scope=tik.scope_ubuf)
        self.tensor_img_cl = lut_obj.tik_inst.Tensor("int32", (work_pixels * lut_obj.ONE_PIXEL_ELEMS,),
                                                     name="tensor_img_cl",
                                                     scope=tik.scope_ubuf)
        self.tensor_img_trans = lut_obj.tik_inst.Tensor("float32", (work_pixels * lut_obj.ONE_PIXEL_ELEMS,),
                                                        name="tensor_img_trans",
                                                        scope=tik.scope_ubuf)
        self.tensor_img_fl_trans = lut_obj.tik_inst.Tensor("int32", (work_pixels * lut_obj.ONE_PIXEL_ELEMS,),
                                                           name="tensor_img_fl_trans",
                                                           scope=tik.scope_ubuf)
        self.tensor_img_cl_trans = lut_obj.tik_inst.Tensor("int32", (work_pixels * lut_obj.ONE_PIXEL_ELEMS,),
                                                           name="tensor_img_cl_trans",
                                                           scope=tik.scope_ubuf)
        self.tensor_img_fl_trans_fp32 = lut_obj.tik_inst.Tensor("float32", (work_pixels * lut_obj.ONE_PIXEL_ELEMS,),
                                                                name="tensor_img_fl_trans_fp32",
                                                                scope=tik.scope_ubuf)
        self.tensor_img_cl_trans_fp32 = lut_obj.tik_inst.Tensor("float32", (work_pixels * lut_obj.ONE_PIXEL_ELEMS,),
                                                                name="tensor_img_cl_trans_fp32",
                                                                scope=tik.scope_ubuf)

        self.fract_b = lut_obj.tik_inst.Tensor("float32", (work_pixels,), tik.scope_ubuf, "fract_b")
        self.fract_b_1 = lut_obj.tik_inst.Tensor("float32", (work_pixels,), tik.scope_ubuf, "fract_b_1")
        self.fract_g = lut_obj.tik_inst.Tensor("float32", (work_pixels,), tik.scope_ubuf, "fract_g")
        self.fract_g_1 = lut_obj.tik_inst.Tensor("float32", (work_pixels,), tik.scope_ubuf, "fract_g_1")
        self.fract_r = lut_obj.tik_inst.Tensor("float32", (work_pixels,), tik.scope_ubuf, "fract_r")
        self.fract_r_1 = lut_obj.tik_inst.Tensor("float32", (work_pixels,), tik.scope_ubuf, "fract_r_1")
        self.b_floor_index = lut_obj.tik_inst.Tensor("float32", (work_pixels,), tik.scope_ubuf, "b_floor_index")
        self.b_ceil_index = lut_obj.tik_inst.Tensor("float32", (work_pixels,), tik.scope_ubuf, "b_ceil_index")
        self.g_floor_index = lut_obj.tik_inst.Tensor("float32", (work_pixels,), tik.scope_ubuf, "g_floor_index")
        self.g_ceil_index = lut_obj.tik_inst.Tensor("float32", (work_pixels,), tik.scope_ubuf, "g_ceil_index")


def _check_input_params(img, lut_table, lut_img, kernel_name):
    """
    check input parameters
    """
    img_dtype = img.get("dtype").lower()
    lut_table_dtype = lut_table.get("dtype").lower()
    lut_img_dtype = lut_img.get("dtype").lower()

    input_check_list = ("float32", "uint8")
    output_check_list = ("float32")
    para_check.check_dtype(img_dtype, input_check_list, param_name="img")
    para_check.check_dtype(lut_table_dtype, input_check_list, param_name="lut_table")
    para_check.check_dtype(lut_img_dtype, output_check_list, param_name="lut_img")

    img_shape = img.get("shape")
    para_check.check_shape(img_shape, min_rank=3, max_rank=4, param_name="img")

    lut_table_shape = lut_table.get("shape")
    para_check.check_shape(lut_table_shape, min_rank=4, max_rank=4, param_name="lut_table")


@register_operator("LUT3D")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def lut_3d(img, lut_table, lut_img, kernel_name="lut_3d"):
    """
    LUT3D op

    Parameters
    ----------
    img: dict
        the dict of input tensor, 3D or 4D, shape of [h,w,c] or [n,h,w,c].
    lut_table: dict
        the dict of input tensor, 4D, shape of [lut_table_n, lut_table_n, lut_table_n, 3].
    lut_img: dict
        the dict of output tensor, 3D, [h,w,c].
    kernel_name: str
        cce kernel name, default value is "lut_3d".

    Returns
    -------
    tik_instance
    """
    _check_input_params(img, lut_table, lut_img, kernel_name)

    obj = LUT3D(img, lut_table, lut_img)
    obj.lut_3d_compute()

    # add compile info
    tbe_context.get_context().add_compile_info("vars", {
        "core_num": obj.core_num,
        "ub_size": obj.ub_size,
        "block_elems": obj.block_elems,
        "vector_elems": obj.vector_elems,
    })

    tik_inst = obj.tik_inst
    tik_inst.BuildCCE(kernel_name=kernel_name,
                      inputs=(obj.img_gm, obj.lut_table_gm),
                      outputs=(obj.lut_img_gm,),
                      flowtable=(obj.tiling_gm,))

    return tik_inst
