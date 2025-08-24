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
img_raw_decode_post_handle
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util import util_common


# 'pylint: disable=invalid-name,too-many-statements,too-many-locals,too-many-arguments
# 'pylint: disable=too-many-instance-attributes,unused-argument,too-few-public-methods
# 'pylint: disable=old-style-class
class ImgYuv2Raw():
    """
    ImgYuv2Raw class
    """
    MAX_INT32 = 2 ** 31 - 1
    MASK_U16 = 128
    MASK_I32 = 64
    BLOCK_I64 = 4
    BLOCK_I32 = 8
    BLOCK_U16 = 16

    N_BITS = 10
    BLC = 56.0
    BLC_1 = BLC - 1
    SCALE = float(2**N_BITS - 1 - BLC)
    CHANNELS = 4
    CHANNELS_QUAD = 2
    PATTERN_1 = 1
    IMG_SIZE_DIMS = 2
    GAMMA_DIMS = 4
    ROW_SLICE = 3072
    TILING_ARG_NUM = 12
    QUAD_SCALE = 2
    SHIFT = 64

    def __init__(self, img_channel_0, img_size, gamma, raw_img):
        """
        init
        """
        self.tik_inst = tik.Tik()
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

        self.channel_dtype = img_channel_0.get("dtype").lower()
        self.img_size_dtype = img_size.get("dtype").lower()
        self.gamma_dtype = gamma.get("dtype").lower()
        self.raw_img_dtype = raw_img.get("dtype").lower()
        self.dtype_i32 = "int32"
        self.tiling_dtype = "int64"
        self.tiling_align = util_common.align(self.TILING_ARG_NUM, self.BLOCK_I64)

        self.channel_0_gm = None
        self.channel_1_gm = None
        self.channel_2_gm = None
        self.channel_3_gm = None
        self.img_size_gm = None
        self.gamma_gm = None
        self.tiling_gm = None
        self.raw_img_gm = None
        self.gamma_0 = None
        self.gamma_1 = None
        self.gamma_2 = None
        self.gamma_3 = None
        self.gamma_list = None
        self.channel_gm_list = None
        self._init_gm_tensor()
        self.gamma_ub = None
        self.one_ub = None
        self.scale_ub = None
        self.shift_ub = None

        # tiling params
        self.h_gm = None
        self.w_gm = None
        self.h_out = None
        self.w_out = None
        self.h_in = None
        self.w_in = None
        self.need_core_num = None
        self.low_core_num = None
        self.rows_num_low = None

        self.support_data_move_pad = tbe_platform.api_check_support("tik.data_move_pad")

    def yuv_2_raw_compute(self):
        """
        compute of yuv to raw
        """
        with self.tik_inst.for_range(0, self.core_num, block_num=self.core_num) as core_id:
            # get tiling data
            self._get_tiling_args()

            with self.tik_inst.if_scope(core_id < self.need_core_num):
                self._get_gamma_data()
                self._init_ub_tensor()

                with self.tik_inst.if_scope(core_id < self.low_core_num):
                    with self.tik_inst.new_stmt_scope():
                        start_idx = self.rows_num_low * core_id
                        self._one_core_compute(start_idx, self.rows_num_low)

                with self.tik_inst.else_scope():
                    with self.tik_inst.new_stmt_scope():
                        rows_num = self.rows_num_low - 1
                        start_idx = self.rows_num_low * self.low_core_num + rows_num * (core_id - self.low_core_num)
                        self._one_core_compute(start_idx, rows_num)

    def yuv_2_raw_compute_quad(self):
        """
        quad mode compute of yuv to raw
        """
        with self.tik_inst.for_range(0, self.core_num, block_num=self.core_num) as core_id:
            # get tiling data
            self._get_tiling_args()

            with self.tik_inst.if_scope(core_id < self.need_core_num):
                self._get_gamma_data()
                self._init_ub_tensor()

                with self.tik_inst.if_scope(core_id < self.low_core_num):
                    with self.tik_inst.new_stmt_scope():
                        start_idx = self.rows_num_low * core_id * self.QUAD_SCALE
                        self._one_core_compute_quad(start_idx, self.rows_num_low)

                with self.tik_inst.else_scope():
                    with self.tik_inst.new_stmt_scope():
                        rows_num = self.rows_num_low - 1
                        start_idx = self.rows_num_low * self.low_core_num + rows_num * (core_id - self.low_core_num)
                        start_idx = start_idx * self.QUAD_SCALE
                        self._one_core_compute_quad(start_idx, rows_num)

    def get_inputs_outputs_gm(self):
        inputs_gm = (self.channel_0_gm, self.channel_1_gm, self.channel_2_gm, self.channel_3_gm,
                     self.img_size_gm, self.gamma_gm)
        outputs_gm = (self.raw_img_gm,)

        return inputs_gm, outputs_gm

    def _init_gm_tensor(self):
        """
        init gm tensor
        """
        self.channel_0_gm = self.tik_inst.Tensor(self.channel_dtype, (self.MAX_INT32,), name="img_channel_0_gm",
                                                 scope=tik.scope_gm)
        self.channel_1_gm = self.tik_inst.Tensor(self.channel_dtype, (self.MAX_INT32,), name="img_channel_1_gm",
                                                 scope=tik.scope_gm)
        self.channel_2_gm = self.tik_inst.Tensor(self.channel_dtype, (self.MAX_INT32,), name="img_channel_2_gm",
                                                 scope=tik.scope_gm)
        self.channel_3_gm = self.tik_inst.Tensor(self.channel_dtype, (self.MAX_INT32,), name="img_channel_3_gm",
                                                 scope=tik.scope_gm)
        self.channel_gm_list = [self.channel_0_gm, self.channel_1_gm, self.channel_2_gm, self.channel_3_gm]
        self.img_size_gm = self.tik_inst.Tensor(self.img_size_dtype, (self.IMG_SIZE_DIMS,), name="img_size_gm",
                                                scope=tik.scope_gm)
        self.gamma_gm = self.tik_inst.Tensor(self.gamma_dtype, (self.GAMMA_DIMS,), name="gamma_gm",
                                             scope=tik.scope_gm)
        self.tiling_gm = self.tik_inst.Tensor(self.tiling_dtype, (self.tiling_align,), name="tiling_gm",
                                              scope=tik.scope_gm)

        self.raw_img_gm = self.tik_inst.Tensor(self.raw_img_dtype, (self.MAX_INT32,), name="raw_img_gm",
                                               scope=tik.scope_gm)

    def _init_ub_tensor(self):
        """
        init ub tensor
        """
        self.one_ub = self.tik_inst.Tensor(self.gamma_dtype, (self.MASK_I32,), name="one_ub", scope=tik.scope_ubuf)
        self.tik_inst.vector_dup(self.MASK_I32, self.one_ub, 1, 1, 1, 8)

        self.scale_ub = self.tik_inst.Tensor(self.gamma_dtype, (self.MASK_I32,), name="scale_ub",
                                             scope=tik.scope_ubuf)
        self.tik_inst.vector_dup(self.MASK_I32, self.scale_ub, self.SCALE, 1, 1, 8)

        self.shift_ub = self.tik_inst.Tensor(self.gamma_dtype, (self.MASK_I32,), name="shift_ub",
                                             scope=tik.scope_ubuf)
        self.tik_inst.vector_dup(self.MASK_I32, self.shift_ub, self.SHIFT, 1, 1, 8)

    def _get_tiling_args(self):
        """
        get runtime params from tiling data
        """
        self.h_gm = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="h_gm")
        self.w_gm = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="w_gm")
        self.h_out = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="h_out")
        self.w_out = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="w_out")
        self.h_in = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="h_in")
        self.w_in = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="w_in")
        self.need_core_num = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="need_core_num")
        self.low_core_num = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="low_core_num")
        self.rows_num_low = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="rows_num_low")

        with self.tik_inst.new_stmt_scope():
            tiling_ub = self.tik_inst.Tensor(self.tiling_dtype, (self.tiling_align,), name="tiling_ub",
                                             scope=tik.scope_ubuf)
            self.tik_inst.data_move(tiling_ub, self.tiling_gm, 0, 1, self.tiling_align // self.BLOCK_I64, 0, 0)

            self.h_gm.set_as(tiling_ub[0])
            self.w_gm.set_as(tiling_ub[1])
            self.h_out.set_as(tiling_ub[2])
            self.w_out.set_as(tiling_ub[3])
            self.h_in.set_as(tiling_ub[4])
            self.w_in.set_as(tiling_ub[5])
            self.need_core_num.set_as(tiling_ub[6])
            self.low_core_num.set_as(tiling_ub[7])
            self.rows_num_low.set_as(tiling_ub[8])

    def _get_gamma_data(self):
        """
        get gamma params
        """
        self.gamma_ub = self.tik_inst.Tensor(self.gamma_dtype, (self.BLOCK_I32,), name="gamma_ub",
                                             scope=tik.scope_ubuf)
        self.tik_inst.data_move(self.gamma_ub, self.gamma_gm, 0, 1, 1, 0, 0)

        self.gamma_0 = self.tik_inst.Scalar(dtype=self.gamma_dtype, name="gamma_0")
        self.gamma_1 = self.tik_inst.Scalar(dtype=self.gamma_dtype, name="gamma_1")
        self.gamma_2 = self.tik_inst.Scalar(dtype=self.gamma_dtype, name="gamma_2")
        self.gamma_3 = self.tik_inst.Scalar(dtype=self.gamma_dtype, name="gamma_3")
        self.gamma_0.set_as(self.gamma_ub[0])
        self.gamma_1.set_as(self.gamma_ub[1])
        self.gamma_2.set_as(self.gamma_ub[2])
        self.gamma_3.set_as(self.gamma_ub[3])
        self.gamma_list = [self.gamma_0, self.gamma_1, self.gamma_2, self.gamma_3]

    def _cast_and_shift(self, channel_f32_ub, repeats, rows_idx, loop_i):
        """
        cast uint16 to float32, and shift
        """
        with self.tik_inst.new_stmt_scope():
            channel_trans_ub = self.tik_inst.Tensor(self.channel_dtype, (self.ROW_SLICE * self.CHANNELS + 256,),
                                                    name="channel_trans_ub", scope=tik.scope_ubuf)
            self.tik_inst.vector_dup(self.MASK_U16, channel_trans_ub[self.ROW_SLICE * self.CHANNELS], 0,
                                     256 // self.MASK_U16, 1, 8)

            with self.tik_inst.new_stmt_scope():
                channel_u16_ub = self.tik_inst.Tensor(self.channel_dtype, (self.ROW_SLICE * self.CHANNELS,),
                                                      name="channel_u16_ub", scope=tik.scope_ubuf)
                for i in range(self.CHANNELS):
                    self.tik_inst.data_move(channel_u16_ub[self.ROW_SLICE * i],
                                            self.channel_gm_list[i][rows_idx * self.w_gm + self.ROW_SLICE * loop_i],
                                            0, 1, self.ROW_SLICE // self.BLOCK_U16, 0, 0)

                # first vnchwconv
                repeat = self.ROW_SLICE // 256
                for channel in range(self.CHANNELS):
                    src_list = [channel_u16_ub[self.ROW_SLICE * channel + 16 * i] for i in range(16)]
                    dst_list = [channel_trans_ub[self.ROW_SLICE * channel + 16 * repeat * i] for i in range(16)]
                    self.tik_inst.vnchwconv(False, False, dst_list, src_list, repeat,
                                            16 // self.BLOCK_U16, 256 // self.BLOCK_U16)

            channel_ub = self.tik_inst.Tensor(self.channel_dtype, (self.ROW_SLICE * self.CHANNELS * 2,),
                                              name="channel_ub", scope=tik.scope_ubuf)
            # second vnchwconv
            repeat2 = self.ROW_SLICE // 2 // 128
            dst_rep_stride = 256 * 2 // self.BLOCK_U16
            src_rep_stride = 16 // self.BLOCK_U16

            for channel in range(self.CHANNELS):
                index_list_1 = []
                for i in range(0, 8):
                    index_list_1.extend([self.ROW_SLICE * channel + 16 * repeat2 * i, self.ROW_SLICE * self.CHANNELS])
                src_list_1 = [channel_trans_ub[i] for i in index_list_1]
                dst_list_1 = [channel_ub[self.ROW_SLICE * 2 * channel + 16 * 2 * i] for i in range(16)]

                index_list_2 = []
                for i in range(8, 16):
                    index_list_2.extend([self.ROW_SLICE * channel + 16 * repeat2 * i, self.ROW_SLICE * self.CHANNELS])
                src_list_2 = [channel_trans_ub[i] for i in index_list_2]
                dst_list_2 = [channel_ub[self.ROW_SLICE * 2 * channel + 16 * 2 * i + 16] for i in range(16)]

                self.tik_inst.vnchwconv(False, False, dst_list_1, src_list_1, repeat2, dst_rep_stride, src_rep_stride)
                self.tik_inst.vnchwconv(False, False, dst_list_2, src_list_2, repeat2, dst_rep_stride, src_rep_stride)

            channel_i32_ub = channel_ub.reinterpret_cast_to(self.dtype_i32)
            self.tik_inst.vconv(self.MASK_I32, "", channel_f32_ub, channel_i32_ub, repeats, 1, 1, 8, 8)

        self.tik_inst.vdiv(self.MASK_I32, channel_f32_ub, channel_f32_ub, self.shift_ub, repeats, 1, 1, 1, 8, 8, 0)

    def _cast_and_shift_quad(self, channel_f32_ub, repeats, rows_idx, loop_i, begin):
        """
        quad mode cast uint16 to float32, and shift
        """
        with self.tik_inst.new_stmt_scope():
            channel_trans_ub = self.tik_inst.Tensor(self.channel_dtype, (self.ROW_SLICE * self.CHANNELS_QUAD + 256,),
                                                    name="channel_trans_ub", scope=tik.scope_ubuf)
            self.tik_inst.vector_dup(self.MASK_U16, channel_trans_ub[self.ROW_SLICE * self.CHANNELS_QUAD], 0,
                                     256 // self.MASK_U16, 1, 8)

            with self.tik_inst.new_stmt_scope():
                channel_u16_ub = self.tik_inst.Tensor(self.channel_dtype, (self.ROW_SLICE * self.CHANNELS_QUAD,),
                                                      name="channel_u16_ub", scope=tik.scope_ubuf)
                for i in range(begin, begin + self.CHANNELS_QUAD):
                    self.tik_inst.data_move(channel_u16_ub[self.ROW_SLICE * (i - begin)],
                                            self.channel_gm_list[i][rows_idx * self.w_gm + self.ROW_SLICE * loop_i],
                                            0, 1, self.ROW_SLICE // self.BLOCK_U16, 0, 0)

                # first vnchwconv
                repeat = self.ROW_SLICE // 256
                for channel in range(self.CHANNELS_QUAD):
                    src_list = [channel_u16_ub[self.ROW_SLICE * channel + 16 * i] for i in range(16)]
                    dst_list = [channel_trans_ub[self.ROW_SLICE * channel + 16 * repeat * i] for i in range(16)]
                    self.tik_inst.vnchwconv(False, False, dst_list, src_list, repeat,
                                            16 // self.BLOCK_U16, 256 // self.BLOCK_U16)

            channel_ub = self.tik_inst.Tensor(self.channel_dtype, (self.ROW_SLICE * self.CHANNELS_QUAD * 2,),
                                              name="channel_ub", scope=tik.scope_ubuf)
            # second vnchwconv
            repeat2 = self.ROW_SLICE // 2 // 128
            dst_rep_stride = 256 * 2 // self.BLOCK_U16
            src_rep_stride = 16 // self.BLOCK_U16

            for channel in range(self.CHANNELS_QUAD):
                index_list_1 = []
                for i in range(0, 8):
                    index_list_1.extend([self.ROW_SLICE * channel + 16 * repeat2 * i,
                                         self.ROW_SLICE * self.CHANNELS_QUAD])
                src_list_1 = [channel_trans_ub[i] for i in index_list_1]
                dst_list_1 = [channel_ub[self.ROW_SLICE * 2 * channel + 16 * 2 * i] for i in range(16)]

                index_list_2 = []
                for i in range(8, 16):
                    index_list_2.extend([self.ROW_SLICE * channel + 16 * repeat2 * i,
                                         self.ROW_SLICE * self.CHANNELS_QUAD])
                src_list_2 = [channel_trans_ub[i] for i in index_list_2]
                dst_list_2 = [channel_ub[self.ROW_SLICE * 2 * channel + 16 * 2 * i + 16] for i in range(16)]

                self.tik_inst.vnchwconv(False, False, dst_list_1, src_list_1, repeat2, dst_rep_stride, src_rep_stride)
                self.tik_inst.vnchwconv(False, False, dst_list_2, src_list_2, repeat2, dst_rep_stride, src_rep_stride)

            channel_i32_ub = channel_ub.reinterpret_cast_to(self.dtype_i32)
            self.tik_inst.vconv(self.MASK_I32, "", channel_f32_ub, channel_i32_ub, repeats, 1, 1, 8, 8)

        self.tik_inst.vdiv(self.MASK_I32, channel_f32_ub, channel_f32_ub, self.shift_ub, repeats, 1, 1, 1, 8, 8, 0)

    def _calculate(self, channel_f32_ub, repeats, row_size, data_align):
        """
        calculate
        """
        with self.tik_inst.new_stmt_scope():
            mask_ub = self.tik_inst.Tensor(self.gamma_dtype, (row_size * 4,), name="mask_ub", scope=tik.scope_ubuf)
            self.tik_inst.vmaxs(self.MASK_I32, mask_ub, channel_f32_ub, self.BLC_1, repeats, 1, 1, 8, 8)
            self.tik_inst.vmins(self.MASK_I32, mask_ub, mask_ub, self.BLC, repeats, 1, 1, 8, 8)
            self.tik_inst.vadds(self.MASK_I32, mask_ub, mask_ub, -1 * self.BLC_1, repeats, 1, 1, 8, 8)

            d_gamma_ub = self.tik_inst.Tensor(self.gamma_dtype, (row_size * 4,), name="d_gamma_ub",
                                              scope=tik.scope_ubuf)
            self.tik_inst.vadds(self.MASK_I32, d_gamma_ub, channel_f32_ub, -1 * self.BLC, repeats, 1, 1, 8, 8)
            self.tik_inst.vdiv(self.MASK_I32, d_gamma_ub, d_gamma_ub, self.scale_ub, repeats, 1, 1, 1, 8, 8, 0)
            self.tik_inst.vmul(self.MASK_I32, d_gamma_ub, d_gamma_ub, mask_ub, repeats, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vln(self.MASK_I32, d_gamma_ub, d_gamma_ub, repeats, 1, 1, 8, 8)
            for i in range(self.CHANNELS):
                self.tik_inst.vmuls(self.MASK_I32, d_gamma_ub[data_align * i], d_gamma_ub[data_align * i],
                                    self.gamma_list[i], data_align // self.MASK_I32, 1, 1, 8, 8)
            self.tik_inst.vexp(self.MASK_I32, d_gamma_ub, d_gamma_ub, repeats, 1, 1, 8, 8)
            self.tik_inst.vmuls(self.MASK_I32, d_gamma_ub, d_gamma_ub, self.SCALE, repeats, 1, 1, 8, 8)
            self.tik_inst.vadds(self.MASK_I32, d_gamma_ub, d_gamma_ub, self.BLC, repeats, 1, 1, 8, 8)

            self.tik_inst.vmul(self.MASK_I32, d_gamma_ub, d_gamma_ub, mask_ub, repeats, 1, 1, 1, 8, 8, 8)

            self.tik_inst.vsub(self.MASK_I32, mask_ub, self.one_ub, mask_ub, repeats, 1, 1, 1, 8, 0, 8)
            self.tik_inst.vmul(self.MASK_I32, channel_f32_ub, channel_f32_ub, mask_ub, repeats, 1, 1, 1, 8, 8, 8)

            self.tik_inst.vadd(self.MASK_I32, channel_f32_ub, channel_f32_ub, d_gamma_ub, repeats, 1, 1, 1, 8, 8, 8)

    def _calculate_quad(self, channel_f32_ub, repeats, row_size, data_align, begin):
        """
        quad mode calculate
        """
        with self.tik_inst.new_stmt_scope():
            mask_ub = self.tik_inst.Tensor(self.gamma_dtype, (row_size * 2,), name="mask_ub", scope=tik.scope_ubuf)
            self.tik_inst.vmaxs(self.MASK_I32, mask_ub, channel_f32_ub, self.BLC_1, repeats, 1, 1, 8, 8)
            self.tik_inst.vmins(self.MASK_I32, mask_ub, mask_ub, self.BLC, repeats, 1, 1, 8, 8)
            self.tik_inst.vadds(self.MASK_I32, mask_ub, mask_ub, -1 * self.BLC_1, repeats, 1, 1, 8, 8)

            d_gamma_ub = self.tik_inst.Tensor(self.gamma_dtype, (row_size * 2,), name="d_gamma_ub",
                                              scope=tik.scope_ubuf)
            self.tik_inst.vadds(self.MASK_I32, d_gamma_ub, channel_f32_ub, -1 * self.BLC, repeats, 1, 1, 8, 8)
            self.tik_inst.vdiv(self.MASK_I32, d_gamma_ub, d_gamma_ub, self.scale_ub, repeats, 1, 1, 1, 8, 8, 0)
            self.tik_inst.vmul(self.MASK_I32, d_gamma_ub, d_gamma_ub, mask_ub, repeats, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vln(self.MASK_I32, d_gamma_ub, d_gamma_ub, repeats, 1, 1, 8, 8)
            for i in range(begin, begin + self.CHANNELS_QUAD):
                self.tik_inst.vmuls(self.MASK_I32,
                                    d_gamma_ub[data_align * (i - begin)], d_gamma_ub[data_align * (i - begin)],
                                    self.gamma_list[i], data_align // self.MASK_I32, 1, 1, 8, 8)
            self.tik_inst.vexp(self.MASK_I32, d_gamma_ub, d_gamma_ub, repeats, 1, 1, 8, 8)
            self.tik_inst.vmuls(self.MASK_I32, d_gamma_ub, d_gamma_ub, self.SCALE, repeats, 1, 1, 8, 8)
            self.tik_inst.vadds(self.MASK_I32, d_gamma_ub, d_gamma_ub, self.BLC, repeats, 1, 1, 8, 8)

            self.tik_inst.vmul(self.MASK_I32, d_gamma_ub, d_gamma_ub, mask_ub, repeats, 1, 1, 1, 8, 8, 8)

            self.tik_inst.vsub(self.MASK_I32, mask_ub, self.one_ub, mask_ub, repeats, 1, 1, 1, 8, 0, 8)
            self.tik_inst.vmul(self.MASK_I32, channel_f32_ub, channel_f32_ub, mask_ub, repeats, 1, 1, 1, 8, 8, 8)

            self.tik_inst.vadd(self.MASK_I32, channel_f32_ub, channel_f32_ub, d_gamma_ub, repeats, 1, 1, 1, 8, 8, 8)

    def _combine_to_raw_impl(self, ub_tensors, elems, idx):
        (res_ub, channel_u16_ub, trans_ub) = ub_tensors

        # first vnchwconv
        repeat = elems // 256
        src_rep_stride = 256 // self.BLOCK_U16
        dst_rep_stride = 256 * 2 // self.BLOCK_U16

        src_list_1 = [channel_u16_ub[elems * 2 * idx + 16 * i] for i in range(16)]
        dst_list_1 = [trans_ub[32 * i] for i in range(16)]

        src_list_2 = [channel_u16_ub[elems * 2 * idx + elems + 16 * i] for i in range(16)]
        dst_list_2 = [trans_ub[32 * i + 16] for i in range(16)]

        with self.tik_inst.if_scope(repeat == 1):
            self.tik_inst.vnchwconv(False, False, dst_list_1, src_list_1, 1, 0, 0)
            self.tik_inst.vnchwconv(False, False, dst_list_2, src_list_2, 1, 0, 0)
        with self.tik_inst.else_scope():
            self.tik_inst.vnchwconv(False, False, dst_list_1, src_list_1, repeat, dst_rep_stride, src_rep_stride)
            self.tik_inst.vnchwconv(False, False, dst_list_2, src_list_2, repeat, dst_rep_stride, src_rep_stride)

        # second vnchwconv
        repeat2 = elems // 256
        src_rep_stride = 256 * 2 // self.BLOCK_U16
        dst_rep_stride = 256 * 2 // self.BLOCK_U16

        src_list_1 = [trans_ub[16 * i] for i in range(16)]
        dst_list_1 = [res_ub[32 * i] for i in range(16)]

        src_list_2 = [trans_ub[256 + 16 * i] for i in range(16)]
        dst_list_2 = [res_ub[16 + 32 * i] for i in range(16)]

        with self.tik_inst.if_scope(repeat2 == 1):
            self.tik_inst.vnchwconv(False, False, dst_list_1, src_list_1, 1, 0, 0)
            self.tik_inst.vnchwconv(False, False, dst_list_2, src_list_2, 1, 0, 0)
        with self.tik_inst.else_scope():
            self.tik_inst.vnchwconv(False, False, dst_list_1, src_list_1, repeat2, dst_rep_stride, src_rep_stride)
            self.tik_inst.vnchwconv(False, False, dst_list_2, src_list_2, repeat2, dst_rep_stride, src_rep_stride)

    def _combine_to_raw_impl_quad(self, ub_tensors, elems, idx):
        (res_ub, channel_u16_ub, trans_ub) = ub_tensors

        # first vnchwconv
        repeat = elems // 256
        src_rep_stride = 256 // self.BLOCK_U16
        dst_rep_stride = 256 * 2 // self.BLOCK_U16

        src_list_1 = [channel_u16_ub[elems * 2 * idx + 16 * i] for i in range(16)]
        index_list_1 = []
        for i in range(0, 8):
            index_list_1.extend([0 + 64 * i, 16 + 64 * i])
        dst_list_1 = [trans_ub[i] for i in index_list_1]

        src_list_2 = [channel_u16_ub[elems * 2 * idx + elems + 16 * i] for i in range(16)]
        index_list_2 = []
        for i in range(0, 8):
            index_list_2.extend([32 + 64 * i, 48 + 64 * i])
        dst_list_2 = [trans_ub[i] for i in index_list_2]

        with self.tik_inst.if_scope(repeat == 1):
            self.tik_inst.vnchwconv(False, False, dst_list_1, src_list_1, 1, 0, 0)
            self.tik_inst.vnchwconv(False, False, dst_list_2, src_list_2, 1, 0, 0)
        with self.tik_inst.else_scope():
            self.tik_inst.vnchwconv(False, False, dst_list_1, src_list_1, repeat, dst_rep_stride, src_rep_stride)
            self.tik_inst.vnchwconv(False, False, dst_list_2, src_list_2, repeat, dst_rep_stride, src_rep_stride)

        # second vnchwconv
        repeat2 = elems // 256
        src_rep_stride = 256 * 2 // self.BLOCK_U16
        dst_rep_stride = 256 * 2 // self.BLOCK_U16

        src_list_1 = [trans_ub[16 * i] for i in range(16)]
        dst_list_1 = [res_ub[32 * i] for i in range(16)]

        src_list_2 = [trans_ub[256 + 16 * i] for i in range(16)]
        dst_list_2 = [res_ub[16 + 32 * i] for i in range(16)]

        with self.tik_inst.if_scope(repeat2 == 1):
            self.tik_inst.vnchwconv(False, False, dst_list_1, src_list_1, 1, 0, 0)
            self.tik_inst.vnchwconv(False, False, dst_list_2, src_list_2, 1, 0, 0)
        with self.tik_inst.else_scope():
            self.tik_inst.vnchwconv(False, False, dst_list_1, src_list_1, repeat2, dst_rep_stride, src_rep_stride)
            self.tik_inst.vnchwconv(False, False, dst_list_2, src_list_2, repeat2, dst_rep_stride, src_rep_stride)

    def _combine_to_raw(self, channel_f32_ub, repeats, rows_idx, loop_i):
        """
        combine 4 channel to raw
        """
        with self.tik_inst.new_stmt_scope():
            channel_u16_ub = self.tik_inst.Tensor(self.channel_dtype, (self.ROW_SLICE * 4,), name="channel_u16_ub",
                                                  scope=tik.scope_ubuf)
            with self.tik_inst.new_stmt_scope():
                y_i32_ub = self.tik_inst.Tensor(self.dtype_i32, (self.ROW_SLICE * 4,), name="y_i32_ub",
                                                scope=tik.scope_ubuf)
                self.tik_inst.vconv(self.MASK_I32, "round", y_i32_ub, channel_f32_ub, repeats, 1, 1, 8, 8)
                y_u16_ub = y_i32_ub.reinterpret_cast_to(self.channel_dtype)

                self.tik_inst.vreduce(self.MASK_U16, channel_u16_ub, y_u16_ub,
                                      self.PATTERN_1, self.ROW_SLICE * 2 * 4 // self.MASK_U16, 1, 8, 0)

            res_ub = self.tik_inst.Tensor(self.channel_dtype, (self.ROW_SLICE * 2,), name="res_ub",
                                          scope=tik.scope_ubuf)
            trans_ub = self.tik_inst.Tensor(self.channel_dtype, (self.ROW_SLICE * 2,), name="trans_ub",
                                            scope=tik.scope_ubuf)
            with self.tik_inst.for_range(0, 2) as idx:
                self._combine_to_raw_impl((res_ub, channel_u16_ub, trans_ub), self.ROW_SLICE, idx)

                # move result to gm
                self.tik_inst.data_move(
                    self.raw_img_gm[(rows_idx * 2 + idx) * self.w_out + self.ROW_SLICE * 2 * loop_i],
                    res_ub, 0, 1, self.ROW_SLICE * 2 // self.BLOCK_U16, 0, 0)

    def _combine_to_raw_quad(self, channel_f32_ub, repeats, out_rows_idx, loop_i):
        """
        quad mode combine 4 channel to raw
        """
        with self.tik_inst.new_stmt_scope():
            channel_u16_ub = self.tik_inst.Tensor(self.channel_dtype, (self.ROW_SLICE * 2,), name="channel_u16_ub",
                                                  scope=tik.scope_ubuf)
            with self.tik_inst.new_stmt_scope():
                y_i32_ub = self.tik_inst.Tensor(self.dtype_i32, (self.ROW_SLICE * 2,), name="y_i32_ub",
                                                scope=tik.scope_ubuf)
                self.tik_inst.vconv(self.MASK_I32, "round", y_i32_ub, channel_f32_ub, repeats, 1, 1, 8, 8)
                y_u16_ub = y_i32_ub.reinterpret_cast_to(self.channel_dtype)

                self.tik_inst.vreduce(self.MASK_U16, channel_u16_ub, y_u16_ub,
                                      self.PATTERN_1, self.ROW_SLICE * 2 * 2 // self.MASK_U16, 1, 8, 0)

            res_ub = self.tik_inst.Tensor(self.channel_dtype, (self.ROW_SLICE * 2,), name="res_ub",
                                          scope=tik.scope_ubuf)
            trans_ub = self.tik_inst.Tensor(self.channel_dtype, (self.ROW_SLICE * 2,), name="trans_ub",
                                            scope=tik.scope_ubuf)

            self._combine_to_raw_impl_quad((res_ub, channel_u16_ub, trans_ub), self.ROW_SLICE, 0)

            # move result to gm
            self.tik_inst.data_move(
                self.raw_img_gm[(out_rows_idx + 0) * self.w_out + self.ROW_SLICE * 2 * loop_i],
                res_ub, 0, 1, self.ROW_SLICE * 2 // self.BLOCK_U16, 0, 0)

    def _cast_and_shift_tail(self, channel_f32_ub, rows_idx, row_size, params):
        """
        cast uint16 to float32, and shift.
        params is (loop, tail, tail_align, repeats)
        """
        (loop, tail, tail_align, repeats) = params

        with self.tik_inst.new_stmt_scope():
            channel_trans_ub = self.tik_inst.Tensor(self.channel_dtype, (row_size * self.CHANNELS + 256,),
                                                    name="channel_trans_ub", scope=tik.scope_ubuf)
            self.tik_inst.vector_dup(self.MASK_U16, channel_trans_ub[row_size * self.CHANNELS], 0,
                                     256 // self.MASK_U16, 1, 8)

            with self.tik_inst.new_stmt_scope():
                channel_u16_ub = self.tik_inst.Tensor(self.channel_dtype, (row_size * self.CHANNELS,),
                                                      name="channel_u16_ub", scope=tik.scope_ubuf)
                for i in range(self.CHANNELS):
                    if self.support_data_move_pad:
                        self.tik_inst.data_move_pad(channel_u16_ub[row_size * i],
                                            self.channel_gm_list[i][rows_idx * self.w_gm + self.ROW_SLICE * loop],
                                            1, tail * 2, 0, 0)
                    else:
                        self.tik_inst.data_move(channel_u16_ub[row_size * i],
                                            self.channel_gm_list[i][rows_idx * self.w_gm + self.ROW_SLICE * loop],
                                            0, 1, util_common.ceil(tail, self.BLOCK_U16), 0, 0)

                # first vnchwconv
                repeat = tail_align // 256
                with self.tik_inst.if_scope(repeat == 1):
                    for channel in range(self.CHANNELS):
                        src_list = [channel_u16_ub[row_size * channel + 16 * i] for i in range(16)]
                        dst_list = [channel_trans_ub[row_size * channel + 16 * i] for i in range(16)]
                        self.tik_inst.vnchwconv(False, False, dst_list, src_list, 1, 0, 0)
                with self.tik_inst.else_scope():
                    for channel in range(self.CHANNELS):
                        src_list = [channel_u16_ub[row_size * channel + 16 * i] for i in range(16)]
                        dst_list = [channel_trans_ub[row_size * channel + 16 * repeat * i] for i in range(16)]
                        self.tik_inst.vnchwconv(False, False, dst_list, src_list, repeat,
                                                16 // self.BLOCK_U16, 256 // self.BLOCK_U16)

            channel_ub = self.tik_inst.Tensor(self.channel_dtype, (row_size * self.CHANNELS * 2,),
                                              name="channel_ub", scope=tik.scope_ubuf)
            # second vnchwconv
            repeat2 = tail_align // 2 // 128
            dst_rep_stride = 256 * 2 // self.BLOCK_U16
            src_rep_stride = 16 // self.BLOCK_U16
            with self.tik_inst.if_scope(repeat2 == 1):
                for channel in range(self.CHANNELS):
                    index_list_1 = []
                    for i in range(0, 8):
                        index_list_1.extend([row_size * channel + 16 * i, row_size * self.CHANNELS])
                    src_list_1 = [channel_trans_ub[i] for i in index_list_1]
                    dst_list_1 = [channel_ub[tail_align * 2 * channel + 16 * 2 * i] for i in range(16)]

                    index_list_2 = []
                    for i in range(8, 16):
                        index_list_2.extend([row_size * channel + 16 * i, row_size * self.CHANNELS])
                    src_list_2 = [channel_trans_ub[i] for i in index_list_2]
                    dst_list_2 = [channel_ub[tail_align * 2 * channel + 16 * 2 * i + 16] for i in range(16)]

                    self.tik_inst.vnchwconv(False, False, dst_list_1, src_list_1, 1, 0, 0)
                    self.tik_inst.vnchwconv(False, False, dst_list_2, src_list_2, 1, 0, 0)
            with self.tik_inst.else_scope():
                for channel in range(self.CHANNELS):
                    index_list_1 = []
                    for i in range(0, 8):
                        index_list_1.extend([row_size * channel + 16 * repeat2 * i, row_size * self.CHANNELS])
                    src_list_1 = [channel_trans_ub[i] for i in index_list_1]
                    dst_list_1 = [channel_ub[tail_align * 2 * channel + 16 * 2 * i] for i in range(16)]

                    index_list_2 = []
                    for i in range(8, 16):
                        index_list_2.extend([row_size * channel + 16 * repeat2 * i, row_size * self.CHANNELS])
                    src_list_2 = [channel_trans_ub[i] for i in index_list_2]
                    dst_list_2 = [channel_ub[tail_align * 2 * channel + 16 * 2 * i + 16] for i in range(16)]

                    self.tik_inst.vnchwconv(False, False, dst_list_1, src_list_1, repeat2,
                                            dst_rep_stride, src_rep_stride)
                    self.tik_inst.vnchwconv(False, False, dst_list_2, src_list_2, repeat2,
                                            dst_rep_stride, src_rep_stride)

            channel_i32_ub = channel_ub.reinterpret_cast_to(self.dtype_i32)
            self.tik_inst.vconv(self.MASK_I32, "", channel_f32_ub, channel_i32_ub, repeats, 1, 1, 8, 8)

        self.tik_inst.vdiv(self.MASK_I32, channel_f32_ub, channel_f32_ub, self.shift_ub, repeats, 1, 1, 1, 8, 8, 0)

    def _cast_and_shift_tail_quad(self, channel_f32_ub, rows_idx, row_size, params):
        """
        quad mode cast uint16 to float32, and shift.
        params is (loop, tail, tail_align, repeats, begin)
        """
        (loop, tail, tail_align, repeats, begin) = params

        with self.tik_inst.new_stmt_scope():
            channel_trans_ub = self.tik_inst.Tensor(self.channel_dtype, (row_size * self.CHANNELS_QUAD + 256,),
                                                    name="channel_trans_ub", scope=tik.scope_ubuf)
            self.tik_inst.vector_dup(self.MASK_U16, channel_trans_ub[row_size * self.CHANNELS_QUAD], 0,
                                     256 // self.MASK_U16, 1, 8)

            with self.tik_inst.new_stmt_scope():
                channel_u16_ub = self.tik_inst.Tensor(self.channel_dtype, (row_size * self.CHANNELS_QUAD,),
                                                      name="channel_u16_ub", scope=tik.scope_ubuf)
                for i in range(begin, begin + self.CHANNELS_QUAD):
                    if self.support_data_move_pad:
                        self.tik_inst.data_move_pad(channel_u16_ub[row_size * (i - begin)],
                                            self.channel_gm_list[i][rows_idx * self.w_gm + self.ROW_SLICE * loop],
                                            1, tail * 2, 0, 0)
                    else:
                        self.tik_inst.data_move(channel_u16_ub[row_size * (i - begin)],
                                            self.channel_gm_list[i][rows_idx * self.w_gm + self.ROW_SLICE * loop],
                                            0, 1, util_common.ceil(tail, self.BLOCK_U16), 0, 0)

                # first vnchwconv
                repeat = tail_align // 256
                with self.tik_inst.if_scope(repeat == 1):
                    for channel in range(self.CHANNELS_QUAD):
                        src_list = [channel_u16_ub[row_size * channel + 16 * i] for i in range(16)]
                        dst_list = [channel_trans_ub[row_size * channel + 16 * i] for i in range(16)]
                        self.tik_inst.vnchwconv(False, False, dst_list, src_list, 1, 0, 0)
                with self.tik_inst.else_scope():
                    for channel in range(self.CHANNELS_QUAD):
                        src_list = [channel_u16_ub[row_size * channel + 16 * i] for i in range(16)]
                        dst_list = [channel_trans_ub[row_size * channel + 16 * repeat * i] for i in range(16)]
                        self.tik_inst.vnchwconv(False, False, dst_list, src_list, repeat,
                                                16 // self.BLOCK_U16, 256 // self.BLOCK_U16)

            channel_ub = self.tik_inst.Tensor(self.channel_dtype, (row_size * self.CHANNELS_QUAD * 2,),
                                              name="channel_ub", scope=tik.scope_ubuf)
            # second vnchwconv
            repeat2 = tail_align // 2 // 128
            dst_rep_stride = 256 * 2 // self.BLOCK_U16
            src_rep_stride = 16 // self.BLOCK_U16
            with self.tik_inst.if_scope(repeat2 == 1):
                for channel in range(self.CHANNELS_QUAD):
                    index_list_1 = []
                    for i in range(0, 8):
                        index_list_1.extend([row_size * channel + 16 * i, row_size * self.CHANNELS_QUAD])
                    src_list_1 = [channel_trans_ub[i] for i in index_list_1]
                    dst_list_1 = [channel_ub[tail_align * 2 * channel + 16 * 2 * i] for i in range(16)]

                    index_list_2 = []
                    for i in range(8, 16):
                        index_list_2.extend([row_size * channel + 16 * i, row_size * self.CHANNELS_QUAD])
                    src_list_2 = [channel_trans_ub[i] for i in index_list_2]
                    dst_list_2 = [channel_ub[tail_align * 2 * channel + 16 * 2 * i + 16] for i in range(16)]

                    self.tik_inst.vnchwconv(False, False, dst_list_1, src_list_1, 1, 0, 0)
                    self.tik_inst.vnchwconv(False, False, dst_list_2, src_list_2, 1, 0, 0)
            with self.tik_inst.else_scope():
                for channel in range(self.CHANNELS_QUAD):
                    index_list_1 = []
                    for i in range(0, 8):
                        index_list_1.extend([row_size * channel + 16 * repeat2 * i, row_size * self.CHANNELS_QUAD])
                    src_list_1 = [channel_trans_ub[i] for i in index_list_1]
                    dst_list_1 = [channel_ub[tail_align * 2 * channel + 16 * 2 * i] for i in range(16)]

                    index_list_2 = []
                    for i in range(8, 16):
                        index_list_2.extend([row_size * channel + 16 * repeat2 * i, row_size * self.CHANNELS_QUAD])
                    src_list_2 = [channel_trans_ub[i] for i in index_list_2]
                    dst_list_2 = [channel_ub[tail_align * 2 * channel + 16 * 2 * i + 16] for i in range(16)]

                    self.tik_inst.vnchwconv(False, False, dst_list_1, src_list_1, repeat2,
                                            dst_rep_stride, src_rep_stride)
                    self.tik_inst.vnchwconv(False, False, dst_list_2, src_list_2, repeat2,
                                            dst_rep_stride, src_rep_stride)

            channel_i32_ub = channel_ub.reinterpret_cast_to(self.dtype_i32)
            self.tik_inst.vconv(self.MASK_I32, "", channel_f32_ub, channel_i32_ub, repeats, 1, 1, 8, 8)

        self.tik_inst.vdiv(self.MASK_I32, channel_f32_ub, channel_f32_ub, self.shift_ub, repeats, 1, 1, 1, 8, 8, 0)

    def _combine_to_raw_tail(self, channel_f32_ub, row_size, rows_idx, params):
        """
        combine 4 channel to raw for tail.
        params is (loop, tail, tail_align, repeats)
        """
        (loop, tail, tail_align, repeats) = params

        with self.tik_inst.new_stmt_scope():
            channel_u16_ub = self.tik_inst.Tensor(self.channel_dtype, (row_size * 4,), name="channel_u16_ub",
                                                  scope=tik.scope_ubuf)
            with self.tik_inst.new_stmt_scope():
                y_i32_ub = self.tik_inst.Tensor(self.dtype_i32, (row_size * 4,), name="y_i32_ub", scope=tik.scope_ubuf)
                self.tik_inst.vconv(self.MASK_I32, "round", y_i32_ub, channel_f32_ub, repeats, 1, 1, 8, 8)
                y_u16_ub = y_i32_ub.reinterpret_cast_to(self.channel_dtype)

                for i in range(self.CHANNELS):
                    self.tik_inst.vreduce(self.MASK_U16, channel_u16_ub[tail_align * i], y_u16_ub[tail_align * 2 * i],
                                          self.PATTERN_1, tail_align * 2 // self.MASK_U16, 1, 8, 0)

            res_ub = self.tik_inst.Tensor(self.channel_dtype, (row_size * 2,), name="res_ub", scope=tik.scope_ubuf)
            trans_ub = self.tik_inst.Tensor(self.channel_dtype, (row_size * 2,), name="trans_ub", scope=tik.scope_ubuf)
            with self.tik_inst.for_range(0, 2) as idx:
                # [2, tail_align] -> [tail_align, 2]
                self._combine_to_raw_impl((res_ub, channel_u16_ub, trans_ub), tail_align, idx)

                # move result to gm
                tail_out = tail * 2
                out_offset = (rows_idx * 2 + idx) * self.w_out + self.ROW_SLICE * 2 * loop
                self._move_raw_to_gm(res_ub, tail_out, out_offset)

    def _combine_to_raw_tail_quad(self, channel_f32_ub, row_size, out_rows_idx, params):
        """
        quad mode combine 4 channel to raw for tail.
        params is (loop, tail, tail_align, repeats)
        """
        (loop, tail, tail_align, repeats) = params

        with self.tik_inst.new_stmt_scope():
            channel_u16_ub = self.tik_inst.Tensor(self.channel_dtype, (row_size * 2,), name="channel_u16_ub",
                                                  scope=tik.scope_ubuf)
            with self.tik_inst.new_stmt_scope():
                y_i32_ub = self.tik_inst.Tensor(self.dtype_i32, (row_size * 2,), name="y_i32_ub", scope=tik.scope_ubuf)
                self.tik_inst.vconv(self.MASK_I32, "round", y_i32_ub, channel_f32_ub, repeats, 1, 1, 8, 8)
                y_u16_ub = y_i32_ub.reinterpret_cast_to(self.channel_dtype)

                for i in range(self.CHANNELS_QUAD):
                    self.tik_inst.vreduce(self.MASK_U16, channel_u16_ub[tail_align * i], y_u16_ub[tail_align * 2 * i],
                                          self.PATTERN_1, tail_align * 2 // self.MASK_U16, 1, 8, 0)

            res_ub = self.tik_inst.Tensor(self.channel_dtype, (row_size * 2,), name="res_ub", scope=tik.scope_ubuf)
            trans_ub = self.tik_inst.Tensor(self.channel_dtype, (row_size * 2,), name="trans_ub", scope=tik.scope_ubuf)

            # [2, tail_align] -> [tail_align, 2]
            self._combine_to_raw_impl_quad((res_ub, channel_u16_ub, trans_ub), tail_align, 0)

            # move result to gm
            tail_out = tail * 2
            out_offset = (out_rows_idx + 0) * self.w_out + self.ROW_SLICE * 2 * loop
            self._move_raw_to_gm(res_ub, tail_out, out_offset)

    def _move_raw_to_gm(self, trans_ub, tail_out, out_offset):
        """
        move result raw to out
        """
        with self.tik_inst.if_scope(tail_out % self.BLOCK_U16 == 0):
            self.tik_inst.data_move(self.raw_img_gm[out_offset], trans_ub, 0, 1, tail_out // self.BLOCK_U16, 0, 0)
        with self.tik_inst.else_scope():
            block_ub = self.tik_inst.Tensor(self.channel_dtype, (self.BLOCK_U16,), name="block_ub",
                                            scope=tik.scope_ubuf)
            last_block_offset = tail_out - self.BLOCK_U16
            with self.tik_inst.new_stmt_scope(disable_sync=True):
                for i in range(self.BLOCK_U16):
                    block_ub[i].set_as(trans_ub[last_block_offset + i])
            self.tik_inst.data_move(self.raw_img_gm[out_offset], trans_ub, 0, 1, tail_out // self.BLOCK_U16, 0, 0)
            self.tik_inst.data_move(self.raw_img_gm[out_offset + last_block_offset], block_ub, 0, 1, 1, 0, 0)

    def _one_row_process(self, rows_idx, loop, tail):
        """
        process one row
        """
        with self.tik_inst.for_range(0, loop) as loop_i:
            with self.tik_inst.new_stmt_scope():
                repeats = self.ROW_SLICE * self.CHANNELS // self.MASK_I32
                channel_f32_ub = self.tik_inst.Tensor(self.gamma_dtype, (self.ROW_SLICE * self.CHANNELS,),
                                                      name="channel_f32_ub", scope=tik.scope_ubuf)
                self._cast_and_shift(channel_f32_ub, repeats, rows_idx, loop_i)
                self._calculate(channel_f32_ub, repeats, self.ROW_SLICE, self.ROW_SLICE)
                self._combine_to_raw(channel_f32_ub, repeats, rows_idx, loop_i)

        with self.tik_inst.if_scope(tail > 0):
            with self.tik_inst.new_stmt_scope():
                row_size = self.ROW_SLICE + 256
                tail_align = util_common.align(tail, 256)
                repeats = tail_align * self.CHANNELS // self.MASK_I32
                channel_f32_ub = self.tik_inst.Tensor(self.gamma_dtype, (row_size * self.CHANNELS,),
                                                      name="channel_f32_ub", scope=tik.scope_ubuf)
                self._cast_and_shift_tail(channel_f32_ub, rows_idx, row_size, (loop, tail, tail_align, repeats))
                self._calculate(channel_f32_ub, repeats, row_size, tail_align)
                self._combine_to_raw_tail(channel_f32_ub, row_size, rows_idx, (loop, tail, tail_align, repeats))

    def _one_row_process_quad(self, rows_idx, out_rows_idx, loop, tail, begin):
        """
        quad mode process one row
        """
        with self.tik_inst.for_range(0, loop) as loop_i:
            with self.tik_inst.new_stmt_scope():
                repeats = self.ROW_SLICE * self.CHANNELS_QUAD // self.MASK_I32
                channel_f32_ub = self.tik_inst.Tensor(self.gamma_dtype, (self.ROW_SLICE * self.CHANNELS_QUAD,),
                                                      name="channel_f32_ub", scope=tik.scope_ubuf)
                self._cast_and_shift_quad(channel_f32_ub, repeats, rows_idx, loop_i, begin)
                self._calculate_quad(channel_f32_ub, repeats, self.ROW_SLICE, self.ROW_SLICE, begin)
                self._combine_to_raw_quad(channel_f32_ub, repeats, out_rows_idx, loop_i)

        with self.tik_inst.if_scope(tail > 0):
            with self.tik_inst.new_stmt_scope():
                row_size = self.ROW_SLICE + 256
                tail_align = util_common.align(tail, 256)
                repeats = tail_align * self.CHANNELS_QUAD // self.MASK_I32
                channel_f32_ub = self.tik_inst.Tensor(self.gamma_dtype, (row_size * self.CHANNELS_QUAD,),
                                                      name="channel_f32_ub", scope=tik.scope_ubuf)
                self._cast_and_shift_tail_quad(channel_f32_ub, rows_idx, row_size,
                                               (loop, tail, tail_align, repeats, begin))
                self._calculate_quad(channel_f32_ub, repeats, row_size, tail_align, begin)
                self._combine_to_raw_tail_quad(channel_f32_ub, row_size, out_rows_idx,
                                               (loop, tail, tail_align, repeats))

    def _one_core_compute(self, rows_start_idx, rows_num):
        """
        compute for one core
        """
        loop = self.tik_inst.Scalar(dtype=self.dtype_i32, name="loop")
        tail = self.tik_inst.Scalar(dtype=self.dtype_i32, name="tail")
        loop.set_as(self.w_in // self.ROW_SLICE)
        tail.set_as(self.w_in % self.ROW_SLICE)
        with self.tik_inst.if_scope(tik.all(tail > 0, tail < (self.BLOCK_U16 // 2), loop > 0)):
            loop.set_as(loop - 1)
            tail.set_as(self.ROW_SLICE + tail)

        with self.tik_inst.for_range(0, rows_num) as idx:
            self._one_row_process(rows_start_idx + idx, loop, tail)

    def _one_core_compute_quad(self, rows_start_idx, rows_num):
        """
        quad mode compute for one core
        """
        loop = self.tik_inst.Scalar(dtype=self.dtype_i32, name="loop")
        tail = self.tik_inst.Scalar(dtype=self.dtype_i32, name="tail")
        loop.set_as(self.w_in // self.ROW_SLICE)
        tail.set_as(self.w_in % self.ROW_SLICE)
        with self.tik_inst.if_scope(tik.all(tail > 0, tail < (self.BLOCK_U16 // 2), loop > 0)):
            loop.set_as(loop - 1)
            tail.set_as(self.ROW_SLICE + tail)

        with self.tik_inst.for_range(0, rows_num) as idx:
            out_rows_idx = rows_start_idx * 2 + idx * 4
            new_rows_start_idx = rows_start_idx + idx * 2
            self._one_row_process_quad(new_rows_start_idx, out_rows_idx, loop, tail, 0)
            self._one_row_process_quad(new_rows_start_idx + 1, out_rows_idx + 1, loop, tail, 0)

            self._one_row_process_quad(new_rows_start_idx, out_rows_idx + 2, loop, tail, 2)
            self._one_row_process_quad(new_rows_start_idx + 1, out_rows_idx + 3, loop, tail, 2)


def _check_input_params(args_list):
    """
    check input parameters.
    args_list is (img_channel_0, img_channel_1, img_channel_2, img_channel_3, img_size, gamma, raw_img)
    """
    (img_channel_0, img_channel_1, img_channel_2, img_channel_3, img_size, gamma, raw_img) = args_list

    channel_0_dtype = img_channel_0.get("dtype").lower()
    channel_1_dtype = img_channel_1.get("dtype").lower()
    channel_2_dtype = img_channel_2.get("dtype").lower()
    channel_3_dtype = img_channel_3.get("dtype").lower()
    img_size_dtype = img_size.get("dtype").lower()
    gamma_dtype = gamma.get("dtype").lower()
    raw_img_dtype = raw_img.get("dtype").lower()
    para_check.check_dtype(channel_0_dtype, ("uint16",), param_name="img_channel_0")
    para_check.check_dtype(channel_1_dtype, ("uint16",), param_name="img_channel_1")
    para_check.check_dtype(channel_2_dtype, ("uint16",), param_name="img_channel_2")
    para_check.check_dtype(channel_3_dtype, ("uint16",), param_name="img_channel_3")
    para_check.check_dtype(img_size_dtype, ("int32",), param_name="img_size")
    para_check.check_dtype(gamma_dtype, ("float32",), param_name="gamma")
    para_check.check_dtype(raw_img_dtype, ("uint16",), param_name="raw_img")

    img_size_shape = img_size.get("shape")
    para_check.check_shape(img_size_shape, min_rank=1, max_rank=1, param_name="img_size")
    if img_size_shape[0] != 2:
        error_manager_vector.raise_err_check_params_rules("ImgRawDecodePostHandle", "img_size shape should be (2, )")

    gamma_shape = gamma.get("shape")
    para_check.check_shape(gamma_shape, min_rank=1, max_rank=1, param_name="gamma")
    if gamma_shape[0] != 4:
        error_manager_vector.raise_err_check_params_rules("ImgRawDecodePostHandle", "gamma shape should be (4, )")


def _check_attr(bayer_pattern):
    """
    check attr bayer_pattern.
    """
    if bayer_pattern not in ('binning', 'quad'):
        error_manager_vector.raise_err_check_params_rules("ImgRawDecodePostHandle",
                                                          "bayer_pattern in ['binning', 'quad']",
                                                          "bayer_pattern", str(bayer_pattern))


@register_operator("ImgRawDecodePostHandle")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def img_raw_decode_post_handle(img_channel_0, img_channel_1, img_channel_2, img_channel_3, img_size, gamma,
                               raw_img, bayer_pattern="binning", kernel_name="img_raw_decode_post_handle"):
    """
    ImgRawDecodePostHandle op

    Parameters
    ----------
    img_channel_0: dict
        the dict of input img_channel_0, shape is [h, w]
    img_channel_1: dict
        the dict of input img_channel_1, shape is [h, w]
    img_channel_2: dict
        the dict of input img_channel_2, shape is [h, w]
    img_channel_3: dict
        the dict of input img_channel_3, shape is [h, w]
    img_size: dict
        the dict of input img_size, shape is [2], value is h_out and w_out, indicates the output height and width
    gamma: dict
        the dict of input gamma, shape is [4], value corresponds to the input gamma values of the four channels
    raw_img: dict
        the dict of output raw_img, shape is [h_out, w_out]
    bayer_pattern: str
        choce calculate mode, the value must be one of ["binning", "quad"], default value is "binning"
    kernel_name: str
        cce kernel name, default value is "img_raw_decode_post_handle"

    Returns
    -------
    tik instance
    """
    args_list = (img_channel_0, img_channel_1, img_channel_2, img_channel_3,
                 img_size, gamma, raw_img)
    _check_input_params(args_list)
    _check_attr(bayer_pattern)

    obj = ImgYuv2Raw(img_channel_0, img_size, gamma, raw_img)
    bayer_pattern_int = 0
    if bayer_pattern in ('quad',):
        bayer_pattern_int = 1
        obj.yuv_2_raw_compute_quad()
    else:
        obj.yuv_2_raw_compute()

    # add compile info
    tbe_context.get_context().add_compile_info("vars", {
        "core_num": obj.core_num,
        "bayer_pattern_int": bayer_pattern_int
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
