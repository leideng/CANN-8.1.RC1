# Copyright 2023 Huawei Technologies Co., Ltd
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
paste_sub_img
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util import util_common
from tbe.common.platform import get_bit_len


# 'pylint: disable=invalid-name,too-many-statements,too-many-locals,too-many-arguments,huawei-too-many-arguments
# 'pylint: disable=too-many-instance-attributes,unused-argument,too-few-public-methods
class PasteSubImg():
    """
    PasteSubImg class
    """
    MAX_INT32 = 2 ** 31 - 1
    BLOCK_I64 = 4

    TILING_ARG_NUM = 8
    # ub row slice: 128kb
    UB_SLICE = 128 * 1024
    UB_SLICE_INT8 = 4 * 1024
    BLOCK_BYTES = 32
    BLOCK_I32 = 8

    def __init__(self, patch_img, patch_coord, core_area_coord, combine_img, scale):
        """
        init
        """
        self.tik_inst = tik.Tik()
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.dtype = patch_img.get("dtype").lower()
        self.patch_img_shape = patch_img.get("shape")
        self.combine_img_shape = combine_img.get("shape")

        self.tiling_dtype = "int64"
        self.dtype_i32 = "int32"
        self.dtype_f16 = "float16"
        self.dtype_u8 = "uint8"
        self.tiling_align = util_common.align(self.TILING_ARG_NUM, self.BLOCK_I64)

        self.dtype_byte_size = get_bit_len(self.dtype) // 8
        self.slice_count = self.UB_SLICE_INT8 if self.dtype == self.dtype_u8 else self.UB_SLICE // self.dtype_byte_size
        self.block_count = self.BLOCK_BYTES // self.dtype_byte_size

        self.patch_img_gm = None
        self.patch_coord_gm = None
        self.core_area_coord_gm = None
        self.combine_img_gm = None
        self.combine_img_out_gm = None
        self.tiling_gm = None

        self._init_gm_tensor()

        # tiling params
        self.h = None
        self.w = None
        self.c = None
        self.h_out = None
        self.w_out = None
        self.scale = scale

        self.cx1 = None
        self.cy1 = None
        self.cx2 = None
        self.cy2 = None

        self.px1 = None
        self.py1 = None

        self.offset_h = None
        self.offset_w = None
        self.move_count = None
        self.offset_h_out = None
        self.offset_w_out = None
        self.need_core_num = None
        self.low_core_num = None
        self.rows_num_low = None

        # ub params
        self.patch_img_ub = None
        self.patch_img_ub_f16 = None
        self.combine_img_ub = None
        self.combine_img_ub_f16 = None
        self.offset_in = None
        self.offset_out = None
        self.burst = None
        self.w_slice_loop = None
        self.w_slice_tail = None
        self.w_block_loop = None
        self.w_block_tail = None
        self.loop_row_num = None

    def paste_sub_img_compute(self):
        """
        paste_sub_img compute
        """
        with self.tik_inst.for_range(0, self.core_num, block_num=self.core_num) as core_id:
            self._get_input_args()
            self._get_core_coord()
            self._get_patch_coord()
            self._cal_tiling_args(self.scale)

            with self.tik_inst.if_scope(core_id < self.need_core_num):
                with self.tik_inst.if_scope(core_id < self.low_core_num):
                    with self.tik_inst.new_stmt_scope():
                        start_idx = self.rows_num_low * core_id
                        self._one_core_compute(start_idx, self.rows_num_low)
                with self.tik_inst.else_scope():
                    with self.tik_inst.new_stmt_scope():
                        rows_num = self.rows_num_low - 1
                        start_idx = self.rows_num_low * self.low_core_num + rows_num * (core_id - self.low_core_num)
                        self._one_core_compute(start_idx, rows_num)

    def get_inputs_outputs_gm(self):
        """
        inputs and outputs gm tensor
        """
        inputs_gm = (self.patch_img_gm, self.patch_coord_gm, self.core_area_coord_gm, self.combine_img_gm)
        outputs_gm = self.combine_img_out_gm
        return inputs_gm, outputs_gm

    def _init_gm_tensor(self):
        """
        init gm tensor
        """
        self.patch_img_gm = self.tik_inst.Tensor(self.dtype, (self.MAX_INT32,), name="patch_img_gm", scope=tik.scope_gm)
        self.patch_coord_gm = self.tik_inst.Tensor(self.dtype_i32, (4,), name="patch_coord_gm", scope=tik.scope_gm)
        self.core_area_coord_gm = self.tik_inst.Tensor(self.dtype_i32, (4,), name="core_area_coord_gm",
                                                       scope=tik.scope_gm)
        self.combine_img_gm = self.tik_inst.Tensor(self.dtype, (self.MAX_INT32,), name="combine_img_gm",
                                                   scope=tik.scope_gm, is_atomic_add=True)
        self.combine_img_out_gm = self.tik_inst.Tensor(self.dtype, (self.MAX_INT32,), name="combine_img_out_gm",
                                                       scope=tik.scope_gm)
        self.tiling_gm = self.tik_inst.Tensor(self.tiling_dtype, (self.tiling_align,), name="tiling_gm",
                                              scope=tik.scope_gm)

    def _init_ub_tensor(self, c):
        """
        init ub tensor
        """
        self.patch_img_ub = self.tik_inst.Tensor(self.dtype_u8, (c,), name="patch_img_ub", scope=tik.scope_ubuf)
        self.patch_img_ub_f16 = self.tik_inst.Tensor(self.dtype_f16, (c,), name="patch_img_ub_f16",
                                                     scope=tik.scope_ubuf)
        self.combine_img_ub_f16 = self.tik_inst.Tensor(self.dtype_f16, (c,), name="combine_img_ub_f16",
                                                       scope=tik.scope_ubuf)
        self.combine_img_ub = self.tik_inst.Tensor(self.dtype_u8, (c,), name="combine_img_ub", scope=tik.scope_ubuf)

    def _init_scalar(self):
        """
        init scalar
        """
        self.offset_in = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="offset_in")
        self.offset_out = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="offset_out")
        self.burst = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="burst")

        self.w_slice_loop = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="w_slice_loop")
        self.w_slice_tail = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="w_slice_tail")
        self.w_block_loop = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="w_block_loop")
        self.w_block_tail = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="w_block_tail")
        self.loop_row_num = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="loop_row_num")

    def _get_input_args(self):
        """
        get runtime params from tiling data
        """
        self.h = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="h")
        self.w = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="w")
        self.c = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="c")
        self.h_out = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="h_out")
        self.w_out = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="w_out")
        if self.patch_img_shape[0] > 0:
            self.h.set_as(self.patch_img_shape[0])
            self.w.set_as(self.patch_img_shape[1])
            self.c.set_as(self.patch_img_shape[2])
            self.h_out.set_as(self.combine_img_shape[0])
            self.w_out.set_as(self.combine_img_shape[1])
        else:
            with self.tik_inst.new_stmt_scope():
                tiling_ub = self.tik_inst.Tensor(self.tiling_dtype, (self.tiling_align,), name="tiling_ub",
                                                 scope=tik.scope_ubuf)
                self.tik_inst.data_move(tiling_ub, self.tiling_gm, 0, 1, self.tiling_align // self.BLOCK_I64, 0, 0)
                self.h.set_as(tiling_ub[0])
                self.w.set_as(tiling_ub[1])
                self.c.set_as(tiling_ub[2])
                self.h_out.set_as(tiling_ub[3])
                self.w_out.set_as(tiling_ub[4])

    def _get_patch_coord(self):
        """
        get patch img coordinate
        """
        self.px1 = self.tik_inst.Scalar(dtype=self.dtype_i32, name="px1")
        self.py1 = self.tik_inst.Scalar(dtype=self.dtype_i32, name="py1")
        with self.tik_inst.new_stmt_scope():
            patch_coord_ub = self.tik_inst.Tensor(self.dtype_i32, (self.BLOCK_I32,), name="patch_coord_ub",
                                                  scope=tik.scope_ubuf)
            self.tik_inst.data_move(patch_coord_ub, self.patch_coord_gm, 0, 1, 1, 0, 0)
            self.px1.set_as(patch_coord_ub[0])
            self.py1.set_as(patch_coord_ub[1])

    def _get_core_coord(self):
        """
        get core area coordinate
        """
        self.cx1 = self.tik_inst.Scalar(dtype=self.dtype_i32, name="cx1")
        self.cy1 = self.tik_inst.Scalar(dtype=self.dtype_i32, name="cy1")
        self.cx2 = self.tik_inst.Scalar(dtype=self.dtype_i32, name="cx2")
        self.cy2 = self.tik_inst.Scalar(dtype=self.dtype_i32, name="cy2")
        with self.tik_inst.new_stmt_scope():
            core_area_coord_ub = self.tik_inst.Tensor(self.dtype_i32, (self.BLOCK_I32,), name="core_area_coord_ub",
                                                      scope=tik.scope_ubuf)
            self.tik_inst.data_move(core_area_coord_ub, self.core_area_coord_gm, 0, 1, 1, 0, 0)
            self.cx1.set_as(core_area_coord_ub[0])
            self.cy1.set_as(core_area_coord_ub[1])
            self.cx2.set_as(core_area_coord_ub[2])
            self.cy2.set_as(core_area_coord_ub[3])

    def _cal_tiling_args(self, scale):
        self.offset_h = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="offset_h")
        self.offset_w = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="offset_w")
        self.move_count = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="move_count")
        self.offset_h_out = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="offset_h_out")
        self.offset_w_out = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="offset_w_out")
        self.need_core_num = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="need_core_num")
        self.low_core_num = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="low_core_num")
        self.rows_num_low = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="rows_num_low")

        self.offset_h.set_as(self.w * self.c * self.cy1 * scale)
        self.offset_w.set_as(self.c * self.cx1 * scale)
        self.move_count.set_as(self.c * scale * (self.cx2 - self.cx1))
        self.offset_h_out.set_as(self.w_out * self.c * (self.cy1 + self.py1) * scale)
        self.offset_w_out.set_as(self.c * (self.cx1 + self.px1) * scale)

        core_h = (self.cy2 - self.cy1) * scale
        with self.tik_inst.if_scope(core_h < self.core_num):
            self.need_core_num.set_as(core_h)
            self.low_core_num.set_as(core_h)
            self.rows_num_low.set_as(1)
        with self.tik_inst.else_scope():
            self.need_core_num.set_as(self.core_num)
            tail = core_h % self.need_core_num
            with self.tik_inst.if_scope(tail > 0):
                self.low_core_num.set_as(tail)
            with self.tik_inst.else_scope():
                self.low_core_num.set_as(self.need_core_num)
            self.rows_num_low.set_as((core_h + self.need_core_num - 1) // self.need_core_num)
        # The amount of data is less than 32B, using single-core processing
        with self.tik_inst.if_scope(tik.all(self.move_count < self.block_count, self.dtype == self.dtype_u8)):
            self.need_core_num.set_as(1)
            self.low_core_num.set_as(1)
            self.rows_num_low.set_as(core_h)

    def _one_core_compute(self, rows_start_idx, rows_num):
        """
        compute for one core
        """
        self._init_scalar()
        # if the shape and data is aligned, try processing multi-rows together for performance
        with self.tik_inst.if_scope(self._is_align()):
            # calculate slice loop by row num
            self.loop_row_num.set_as(self.slice_count // self.move_count)
            with self.tik_inst.if_scope(self.loop_row_num > 0):
                if self.dtype == self.dtype_u8:
                    self._multi_row_process_for_int(rows_start_idx, rows_num)
                else:
                    self._multi_row_process(rows_start_idx, rows_num)
            with self.tik_inst.else_scope():
                if self.dtype == self.dtype_u8:
                    self._one_row_process_for_int(rows_start_idx, rows_num)
                else:
                    self._one_row_process(rows_start_idx, rows_num)
        with self.tik_inst.else_scope():
            if self.dtype == self.dtype_u8:
                with self.tik_inst.if_scope(self.move_count < self.block_count):
                    self._one_row_process_for_small_size_int8(rows_start_idx, rows_num)
                with self.tik_inst.else_scope():
                    self._one_row_process_for_int(rows_start_idx, rows_num)
            else:
                self._one_row_process(rows_start_idx, rows_num)

    def _one_row_process(self, rows_start_idx, rows_num):
        """
        process one row.
        """
        self.patch_img_ub = self.tik_inst.Tensor(self.dtype, (self.slice_count,),
                                                 name="patch_img_ub",
                                                 scope=tik.scope_ubuf)
        self.w_slice_loop.set_as(self.move_count * self.dtype_byte_size // self.UB_SLICE)
        self.w_slice_tail.set_as((self.move_count * self.dtype_byte_size) % self.UB_SLICE)
        self.w_block_loop.set_as(self.w_slice_tail // self.BLOCK_BYTES)
        self.w_block_tail.set_as(self.w_slice_tail % self.BLOCK_BYTES)

        with self.tik_inst.for_range(0, rows_num) as idx:
            base_offset_in = self.offset_h + (rows_start_idx + idx) * self.w * self.c + self.offset_w
            base_offset_out = self.offset_h_out + (rows_start_idx + idx) * self.w_out * self.c + self.offset_w_out
            with self.tik_inst.for_range(0, self.w_slice_loop) as loop_i:
                offset_loop = self.slice_count * loop_i
                self.offset_in.set_as(base_offset_in + offset_loop)
                self.offset_out.set_as(base_offset_out + offset_loop)
                self.burst.set_as(self.UB_SLICE // self.BLOCK_BYTES)
                self.tik_inst.data_move(self.patch_img_ub, self.patch_img_gm[self.offset_in], 0, 1, self.burst, 0, 0)
                self.tik_inst.set_atomic_add(self.dtype)
                self.tik_inst.data_move(self.combine_img_out_gm[self.offset_out], self.patch_img_ub, 0, 1,
                                        self.burst, 0, 0)
                self.tik_inst.set_atomic_add(0)
            # tail
            with self.tik_inst.if_scope(self.w_slice_tail > 0):
                offset_loop = self.slice_count * self.w_slice_loop
                self.offset_in.set_as(base_offset_in + offset_loop)
                self.offset_out.set_as(base_offset_out + offset_loop)
                self.burst.set_as((self.w_slice_tail + 31) // self.BLOCK_BYTES)
                self.tik_inst.data_move(self.patch_img_ub, self.patch_img_gm[self.offset_in], 0, 1, self.burst, 0, 0)
                # 32 byte align
                with self.tik_inst.if_scope(self.w_block_tail > 0):
                    with self.tik_inst.for_range(self.w_block_tail // self.dtype_byte_size, self.block_count) as i:
                        self.patch_img_ub[self.w_block_loop * self.block_count + i].set_as(0)
                self.tik_inst.set_atomic_add(self.dtype)
                self.tik_inst.data_move(self.combine_img_out_gm[self.offset_out], self.patch_img_ub, 0, 1,
                                        self.burst, 0, 0)
                self.tik_inst.set_atomic_add(0)

    def _one_row_process_for_int(self, rows_start_idx, rows_num):
        self._init_ub_tensor(self.slice_count)
        self.w_slice_loop.set_as(self.move_count // self.slice_count)
        self.w_slice_tail.set_as(self.move_count % self.slice_count)
        self.w_block_loop.set_as(self.w_slice_tail // self.block_count)
        self.w_block_tail.set_as(self.w_slice_tail % self.block_count)
        with self.tik_inst.for_range(0, rows_num) as idx:
            base_offset_in = self.offset_h + (rows_start_idx + idx) * self.w * self.c + self.offset_w
            base_offset_out = self.offset_h_out + (rows_start_idx + idx) * self.w_out * self.c + self.offset_w_out
            with self.tik_inst.for_range(0, self.w_slice_loop) as loop_i:
                offset_slice_loop = self.slice_count * loop_i
                self.offset_in.set_as(base_offset_in + offset_slice_loop)
                self.offset_out.set_as(base_offset_out + offset_slice_loop)
                self.burst.set_as(self.UB_SLICE_INT8 // self.BLOCK_BYTES)
                self.tik_inst.data_move(self.patch_img_ub, self.patch_img_gm[self.offset_in], 0, 1,
                                        self.burst, 0, 0)
                self.tik_inst.data_move(self.combine_img_ub, self.combine_img_gm[self.offset_out], 0, 1,
                                        self.burst, 0, 0)
                self.tik_inst.vconv(self.block_count, "", self.patch_img_ub_f16, self.patch_img_ub,
                                    self.slice_count // self.block_count, 1, 1, 2, 1)
                self.tik_inst.vconv(self.block_count, "", self.combine_img_ub_f16, self.combine_img_ub,
                                    self.slice_count // self.block_count, 1, 1, 2, 1)
                self.tik_inst.vadd(self.block_count, self.combine_img_ub_f16, self.combine_img_ub_f16,
                                   self.patch_img_ub_f16, self.slice_count // self.block_count, 1, 1, 1, 2, 2, 2)
                self.tik_inst.vconv(self.block_count, "", self.combine_img_ub, self.combine_img_ub_f16,
                                    self.slice_count // self.block_count, 1, 1, 1, 2)
                self.tik_inst.data_move(self.combine_img_out_gm[self.offset_out], self.combine_img_ub, 0, 1,
                                        self.burst, 0, 0)
            with self.tik_inst.if_scope(self.w_slice_tail > 0):
                offset_slice_loop = self.slice_count * self.w_slice_loop
                self.offset_in.set_as(base_offset_in + offset_slice_loop)
                self.offset_out.set_as(base_offset_out + offset_slice_loop)
                self.burst.set_as((self.w_slice_tail + 31) // self.BLOCK_BYTES)
                offset_block_loop = self.block_count * self.w_block_loop
                offset_revert = offset_block_loop - (self.block_count - self.w_block_tail)
                # 32B align
                with self.tik_inst.if_scope(self.w_block_tail > 0):
                    with self.tik_inst.if_scope(self.burst > 1):
                        self.tik_inst.data_move(self.patch_img_ub, self.patch_img_gm[self.offset_in], 0, 1,
                                                self.burst - 1, 0, 0)
                        self.tik_inst.data_move(self.combine_img_ub, self.combine_img_gm[self.offset_out], 0, 1,
                                                self.burst - 1, 0, 0)
                    self.tik_inst.data_move(self.patch_img_ub[offset_block_loop],
                                            self.patch_img_gm[self.offset_in + offset_revert], 0, 1, 1, 0, 0)
                    self.tik_inst.data_move(self.combine_img_ub[offset_block_loop],
                                            self.combine_img_gm[self.offset_out + offset_revert], 0, 1, 1, 0, 0)
                    with self.tik_inst.if_scope(offset_block_loop == 0):
                        with self.tik_inst.for_range(0, self.block_count - self.w_block_tail) as i:
                            self.patch_img_ub[i].set_as(0)
                with self.tik_inst.else_scope():
                    self.tik_inst.data_move(self.patch_img_ub, self.patch_img_gm[self.offset_in], 0, 1,
                                            self.burst, 0, 0)
                    self.tik_inst.data_move(self.combine_img_ub, self.combine_img_gm[self.offset_out], 0, 1,
                                            self.burst, 0, 0)
                self.tik_inst.vconv(self.block_count, "", self.patch_img_ub_f16, self.patch_img_ub,
                                    self.burst, 1, 1, 2, 1)
                self.tik_inst.vconv(self.block_count, "", self.combine_img_ub_f16, self.combine_img_ub,
                                    self.burst, 1, 1, 2, 1)
                self.tik_inst.vadd(self.block_count, self.combine_img_ub_f16, self.combine_img_ub_f16,
                                   self.patch_img_ub_f16, self.burst, 1, 1, 1, 2, 2, 2)
                self.tik_inst.vconv(self.block_count, "", self.combine_img_ub, self.combine_img_ub_f16,
                                    self.burst, 1, 1, 1, 2)
                with self.tik_inst.if_scope(self.w_block_tail > 0):
                    with self.tik_inst.if_scope(self.burst > 1):
                        self.tik_inst.data_move(self.combine_img_out_gm[self.offset_out], self.combine_img_ub, 0, 1,
                                                self.burst - 1, 0, 0)
                    self.tik_inst.data_move(self.combine_img_out_gm[self.offset_out + offset_revert],
                                            self.combine_img_ub[offset_block_loop], 0, 1, 1, 0, 0)
                with self.tik_inst.else_scope():
                    self.tik_inst.data_move(self.combine_img_out_gm[self.offset_out], self.combine_img_ub, 0, 1,
                                            self.burst, 0, 0)

    def _one_row_process_for_small_size_int8(self, rows_start_idx, rows_num):
        """
        process small amount of uint8 data by row
        """
        self._init_ub_tensor(self.block_count)
        with self.tik_inst.for_range(0, rows_num) as idx:
            base_offset_in = self.offset_h + (rows_start_idx + idx) * self.w * self.c + self.offset_w
            base_offset_out = self.offset_h_out + (rows_start_idx + idx) * self.w_out * self.c + self.offset_w_out
            self.tik_inst.data_move(self.patch_img_ub, self.patch_img_gm[base_offset_in], 0, 1, 1, 0, 0)
            # 32B align
            with self.tik_inst.for_range(self.move_count, self.block_count) as i:
                self.patch_img_ub[i].set_as(0)
            self.tik_inst.data_move(self.combine_img_ub, self.combine_img_gm[base_offset_out], 0, 1, 1, 0, 0)
            self.tik_inst.vconv(self.block_count, "", self.patch_img_ub_f16, self.patch_img_ub, 1, 1, 1, 2, 1)
            self.tik_inst.vconv(self.block_count, "", self.combine_img_ub_f16, self.combine_img_ub, 1, 1, 1, 2, 1)
            self.tik_inst.vadd(self.block_count, self.combine_img_ub_f16, self.combine_img_ub_f16,
                               self.patch_img_ub_f16, 1, 1, 1, 1, 2, 2, 2)
            self.tik_inst.vconv(self.block_count, "", self.combine_img_ub, self.combine_img_ub_f16, 1, 1, 1, 1, 2)
            self.tik_inst.data_move(self.combine_img_out_gm[base_offset_out], self.combine_img_ub, 0, 1, 1, 0, 0)

    def _multi_row_process(self, row_idx, rows_num):
        """
        process multi rows
        """
        self.w_slice_loop.set_as(rows_num // self.loop_row_num)
        self.w_slice_tail.set_as(rows_num % self.loop_row_num)
        self.w_block_tail.set_as(0)
        self.burst.set_as(self.move_count // self.block_count)
        base_offset_in = self.offset_h + row_idx * self.w * self.c + self.offset_w
        base_offset_out = self.offset_h_out + row_idx * self.w_out * self.c + self.offset_w_out
        src_stride = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="src_stride")
        src_stride.set_as((self.w * self.c - self.move_count) // self.block_count)
        dst_stride = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="dst_stride")
        dst_stride.set_as((self.w_out * self.c - self.move_count) // self.block_count)
        self.patch_img_ub = self.tik_inst.Tensor(self.dtype, (self.slice_count,),
                                                 name="patch_img_ub",
                                                 scope=tik.scope_ubuf)
        with self.tik_inst.for_range(0, self.w_slice_loop) as loop_idx:
            offset_loop_in = self.loop_row_num * loop_idx * self.w * self.c
            offset_loop_out = self.loop_row_num * loop_idx * self.w_out * self.c
            self.offset_in.set_as(base_offset_in + offset_loop_in)
            self.offset_out.set_as(base_offset_out + offset_loop_out)
            self.tik_inst.data_move(self.patch_img_ub, self.patch_img_gm[self.offset_in], 0,
                                    self.loop_row_num, self.burst, src_stride, 0)
            self.tik_inst.set_atomic_add(self.dtype)
            self.tik_inst.data_move(self.combine_img_out_gm[self.offset_out], self.patch_img_ub, 0,
                                    self.loop_row_num, self.burst, 0, dst_stride)
            self.tik_inst.set_atomic_add(0)
        with self.tik_inst.if_scope(self.w_slice_tail > 0):
            offset_loop_in = self.loop_row_num * self.w_slice_loop * self.w * self.c
            offset_loop_out = self.loop_row_num * self.w_slice_loop * self.w_out * self.c
            self.offset_in.set_as(base_offset_in + offset_loop_in)
            self.offset_out.set_as(base_offset_out + offset_loop_out)
            self.tik_inst.data_move(self.patch_img_ub, self.patch_img_gm[self.offset_in], 0,
                                    self.w_slice_tail, self.burst, src_stride, 0)
            self.tik_inst.set_atomic_add(self.dtype)
            self.tik_inst.data_move(self.combine_img_out_gm[self.offset_out], self.patch_img_ub, 0,
                                    self.w_slice_tail, self.burst, 0, dst_stride)
            self.tik_inst.set_atomic_add(0)

    def _multi_row_process_for_int(self, row_idx, rows_num):
        """
        process multi rows for uint8
        """
        self._init_ub_tensor(self.slice_count)
        self.w_slice_loop.set_as(rows_num // self.loop_row_num)
        self.w_slice_tail.set_as(rows_num % self.loop_row_num)
        self.burst.set_as(self.move_count // self.block_count)
        base_offset_in = self.offset_h + row_idx * self.w * self.c + self.offset_w
        base_offset_out = self.offset_h_out + row_idx * self.w_out * self.c + self.offset_w_out
        src_stride = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="src_stride")
        src_stride.set_as((self.w * self.c - self.move_count) // self.block_count)
        dst_stride = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="dst_stride")
        dst_stride.set_as((self.w_out * self.c - self.move_count) // self.block_count)
        with self.tik_inst.for_range(0, self.w_slice_loop) as loop_idx:
            offset_loop_in = self.loop_row_num * loop_idx * self.w * self.c
            offset_loop_out = self.loop_row_num * loop_idx * self.w_out * self.c
            self.offset_in.set_as(base_offset_in + offset_loop_in)
            self.offset_out.set_as(base_offset_out + offset_loop_out)
            self.tik_inst.data_move(self.patch_img_ub, self.patch_img_gm[self.offset_in], 0, self.loop_row_num,
                                    self.burst, src_stride, 0)
            self.tik_inst.data_move(self.combine_img_ub, self.combine_img_gm[self.offset_out], 0, self.loop_row_num,
                                    self.burst, dst_stride, 0)
            self.tik_inst.vconv(self.block_count, "", self.patch_img_ub_f16, self.patch_img_ub,
                                self.burst * self.loop_row_num, 1, 1, 2, 1)
            self.tik_inst.vconv(self.block_count, "", self.combine_img_ub_f16, self.combine_img_ub,
                                self.burst * self.loop_row_num, 1, 1, 2, 1)
            self.tik_inst.vadd(self.block_count, self.combine_img_ub_f16, self.combine_img_ub_f16,
                               self.patch_img_ub_f16, self.burst * self.loop_row_num, 1, 1, 1, 2, 2, 2)
            self.tik_inst.vconv(self.block_count, "", self.combine_img_ub, self.combine_img_ub_f16,
                                self.burst * self.loop_row_num, 1, 1, 1, 2)
            self.tik_inst.data_move(self.combine_img_out_gm[self.offset_out], self.combine_img_ub, 0, self.loop_row_num,
                                    self.burst, 0, dst_stride)
        # tail
        with self.tik_inst.if_scope(self.w_slice_tail > 0):
            offset_loop_in = self.loop_row_num * self.w_slice_loop * self.w * self.c
            offset_loop_out = self.loop_row_num * self.w_slice_loop * self.w_out * self.c
            self.offset_in.set_as(base_offset_in + offset_loop_in)
            self.offset_out.set_as(base_offset_out + offset_loop_out)
            self.tik_inst.data_move(self.patch_img_ub, self.patch_img_gm[self.offset_in], 0, self.w_slice_tail,
                                    self.burst, src_stride, 0)
            self.tik_inst.data_move(self.combine_img_ub, self.combine_img_gm[self.offset_out], 0, self.w_slice_tail,
                                    self.burst, dst_stride, 0)
            self.tik_inst.vconv(self.block_count, "", self.patch_img_ub_f16, self.patch_img_ub,
                                self.burst * self.w_slice_tail, 1, 1, 2, 1)
            self.tik_inst.vconv(self.block_count, "", self.combine_img_ub_f16, self.combine_img_ub,
                                self.burst * self.w_slice_tail, 1, 1, 2, 1)
            self.tik_inst.vadd(self.block_count, self.combine_img_ub_f16, self.combine_img_ub_f16,
                               self.patch_img_ub_f16, self.burst * self.w_slice_tail, 1, 1, 1, 2, 2, 2)
            self.tik_inst.vconv(self.block_count, "", self.combine_img_ub, self.combine_img_ub_f16,
                                self.burst * self.w_slice_tail, 1, 1, 1, 2)
            self.tik_inst.data_move(self.combine_img_out_gm[self.offset_out], self.combine_img_ub, 0, self.w_slice_tail,
                                    self.burst, 0, dst_stride)

    def _is_align(self):
        return tik.all((self.w * self.c) % self.block_count == 0,
                       (self.w_out * self.c) % self.block_count == 0,
                       self.move_count % self.block_count == 0,
                       self.offset_w % self.block_count == 0,
                       self.offset_w_out % self.block_count == 0)


def _check_input_params(input_list, output_list):
    """
    check input parameters.
    input_list is (patch_img, patch_coord, core_area_coord, combine_img, scale)
    output_list is (combine_img_out)
    """
    (patch_img, patch_coord, core_area_coord, combine_img, scale) = input_list
    combine_img_out = output_list

    patch_img_dtype = patch_img.get("dtype").lower()
    patch_coord_dtype = patch_coord.get("dtype").lower()
    core_area_coord_dtype = core_area_coord.get("dtype").lower()
    combine_img_dtype = combine_img.get("dtype").lower()
    combine_img_out_dtype = combine_img_out.get("dtype").lower()

    # type check
    para_check.check_dtype(patch_img_dtype, ("uint8", "float16", "float32"), param_name="patch_img")
    para_check.check_dtype(patch_coord_dtype, ("int32",), param_name="patch_coord")
    para_check.check_dtype(core_area_coord_dtype, ("int32",), param_name="core_area_coord")
    para_check.check_dtype(combine_img_dtype, ("uint8", "float16", "float32"), param_name="combine_img")
    para_check.check_dtype(combine_img_out_dtype, ("uint8", "float16", "float32"), param_name="combine_img_out")

    # shape check
    patch_img_shape = patch_img.get("shape")
    patch_coord_shape = patch_coord.get("shape")
    core_area_coord_shape = core_area_coord.get("shape")
    combine_img_shape = combine_img.get("shape")
    para_check.check_shape(patch_img_shape, min_rank=3, max_rank=3, param_name="patch_img")
    para_check.check_shape(patch_coord_shape, min_rank=1, max_rank=1, param_name="patch_coord")
    para_check.check_shape(core_area_coord_shape, min_rank=1, max_rank=1, param_name="core_area_coord")
    para_check.check_shape(combine_img_shape, min_rank=3, max_rank=3, param_name="combine_img")


@register_operator("PasteSubImg")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def paste_sub_img(patch_img, patch_coord, core_area_coord, combine_img, combine_img_out, scale,
                  kernel_name="paste_sub_img"):
    """
    PasteSubImg op

    Parameters
    ----------
    patch_img: dict
        the dict of input patch_img, shape is [h, w, c]
    patch_coord: dict
        the dict of input patch_coord, shape is [4], value is [left, top, right, bottom]
    core_area_coord: dict
        the dict of input core_area_coord, shape is [4], value is [left, top, right, bottom]
    combine_img: dict
        the dict of input combine_img, shape is [H, W, C]
    combine_img_out: dict
        the dict of output combine_img, shape is [H, W, C]
    scale: float
        the scale of coordinate
    kernel_name: str
        cce kernel name, default value is "paste_sub_img"

    Returns
    -------
    tik instance
    """
    input_list = (patch_img, patch_coord, core_area_coord, combine_img, scale)
    output_list = combine_img_out
    _check_input_params(input_list, output_list)

    obj = PasteSubImg(patch_img, patch_coord, core_area_coord, combine_img, scale)
    obj.paste_sub_img_compute()

    # add compile info
    tbe_context.get_context().add_compile_info("vars", {
        "core_num": obj.core_num
    })

    tik_inst = obj.tik_inst
    inputs_gm, outputs_gm = obj.get_inputs_outputs_gm()
    opt_config = {"enable_const_fold": True}
    tik_inst.BuildCCE(kernel_name=kernel_name,
                      inputs=inputs_gm,
                      outputs=outputs_gm,
                      flowtable=(obj.tiling_gm,),
                      config=opt_config)

    return tik_inst
