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
img_raw_decode_post_handle_v2
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
class ImgYuv2RawV2():
    """
    ImgYuv2RawV2 class
    """
    MAX_INT32 = 2 ** 31 - 1
    MASK_U16 = 128
    MASK_I32 = 64
    BLOCK_I64 = 4
    BLOCK_I32 = 8
    BLOCK_U16 = 16

    N_BITS = 10
    BLC = 56.0
    BLC_REC = 1 / BLC
    BLC_1 = BLC - 1
    SCALE = 959.0
    BL_FIX_FLOAT = 1.0 / 16
    BAYER_FACTOR = 2 ** N_BITS - 1

    # noise profile for jade_main mode
    K_A = 0.000282310008
    K_B = 0.00109075999
    B_A = 2.29999998e-07
    B_B = 5.20600006e-05
    B_C = 0.0189176109

    ROW_SLICE = 3072
    CHANNELS = 4
    DIMS_4 = 4
    DIMS_8 = 8
    TILING_ARG_NUM = 16

    def __init__(self, img_channel_0, raw_img):
        """
        init
        """
        self.tik_inst = tik.Tik()
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

        self.channel_dtype = img_channel_0.get("dtype").lower()
        self.raw_img_dtype = raw_img.get("dtype").lower()
        self.dtype_fp32 = "float32"
        self.dtype_i32 = "int32"
        self.tiling_dtype = "int64"
        self.tiling_align = util_common.align(self.TILING_ARG_NUM, self.BLOCK_I64)

        self.channel_0_gm = None
        self.channel_1_gm = None
        self.channel_2_gm = None
        self.channel_3_gm = None
        self.gamma_gm = None
        self.bayer_coordinate_gm = None
        self.bayer_params_gm = None
        self.bayer_ptn_gm = None
        self.tiling_gm = None
        self.raw_img_gm = None
        self.gamma_list = None
        self._init_gm_tensor()

        self.one_ub = None
        self.scale_ub = None
        self.bayer_factor_ub = None
        self.rgb_gain_arr = None
        self.iso = None
        self.ev_gain = None
        self.iso_long = None
        self.evSL = None
        self.exposure_gain = None
        self.max_clip = None
        self.gain_list = None
        self.a_ub_list = None
        self.bias_list = None

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
        self.has_last_row = None
        self.combine_mode = None
        self.input_lt_x = None
        self.input_lt_y = None

    def yuv_2_raw_v2_compute(self):
        """
        compute of yuv to raw
        """
        with self.tik_inst.for_range(0, self.core_num, block_num=self.core_num) as core_id:
            # get tiling data
            self._get_tiling_args()

            with self.tik_inst.if_scope(core_id < self.need_core_num):
                self._init_ub_tensor()
                self._get_gamma_and_bayer_params()

                with self.tik_inst.if_scope(core_id < self.low_core_num):
                    with self.tik_inst.new_stmt_scope():
                        start_idx = self.rows_num_low * core_id
                        self._one_core_compute(core_id, start_idx, self.rows_num_low)

                with self.tik_inst.else_scope():
                    with self.tik_inst.new_stmt_scope():
                        rows_num = self.rows_num_low - 1
                        start_idx = self.rows_num_low * self.low_core_num + rows_num * (core_id - self.low_core_num)
                        self._one_core_compute(core_id, start_idx, rows_num)

    def get_inputs_outputs_gm(self):
        inputs_gm = (self.channel_0_gm, self.channel_1_gm, self.channel_2_gm, self.channel_3_gm, self.gamma_gm,
                     self.bayer_coordinate_gm, self.bayer_params_gm, self.bayer_ptn_gm)
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
        self.gamma_gm = self.tik_inst.Tensor(self.dtype_fp32, (self.DIMS_4,), name="gamma_gm", scope=tik.scope_gm)
        self.bayer_coordinate_gm = self.tik_inst.Tensor(self.dtype_i32, (self.DIMS_4,), name="bayer_coordinate_gm",
                                                        scope=tik.scope_gm)
        self.bayer_params_gm = self.tik_inst.Tensor(self.dtype_fp32, (self.DIMS_8,), name="bayer_params_gm",
                                                    scope=tik.scope_gm)
        self.bayer_ptn_gm = self.tik_inst.Tensor(self.dtype_i32, (self.DIMS_4,), name="bayer_ptn_gm",
                                                 scope=tik.scope_gm)
        self.tiling_gm = self.tik_inst.Tensor(self.tiling_dtype, (self.tiling_align,), name="tiling_gm",
                                              scope=tik.scope_gm)

        self.raw_img_gm = self.tik_inst.Tensor(self.raw_img_dtype, (self.MAX_INT32,), name="raw_img_gm",
                                               scope=tik.scope_gm)

    def _init_ub_tensor(self):
        """
        init ub tensor
        """
        self.one_ub = self.tik_inst.Tensor(self.dtype_fp32, (self.MASK_I32,), name="one_ub", scope=tik.scope_ubuf)
        self.scale_ub = self.tik_inst.Tensor(self.dtype_fp32, (self.MASK_I32,), name="scale_ub",
                                             scope=tik.scope_ubuf)
        self.bayer_factor_ub = self.tik_inst.Tensor(self.dtype_fp32, (self.MASK_I32,), name="bayer_factor_ub",
                                                    scope=tik.scope_ubuf)
        self.tik_inst.vector_dup(self.MASK_I32, self.one_ub, 1, 1, 1, 8)
        self.tik_inst.vector_dup(self.MASK_I32, self.scale_ub, self.SCALE, 1, 1, 8)
        self.tik_inst.vector_dup(self.MASK_I32, self.bayer_factor_ub, self.BAYER_FACTOR, 1, 1, 8)

        self.a_ub_list = []
        for i in range(self.CHANNELS):
            a_ub = self.tik_inst.Tensor(self.dtype_fp32, (self.MASK_I32,), name="a_ub", scope=tik.scope_ubuf)
            self.a_ub_list.append(a_ub)

        self.rgb_gain_arr = self.tik_inst.ScalarArray(dtype=self.dtype_fp32, length=3, name="rgb_gain_arr",
                                                      init_value=0)
        self.iso = self.tik_inst.Scalar(dtype=self.dtype_fp32, name="iso")
        self.ev_gain = self.tik_inst.Scalar(dtype=self.dtype_fp32, name="ev_gain")
        self.iso_long = self.tik_inst.Scalar(dtype=self.dtype_fp32, name="iso_long")
        self.evSL = self.tik_inst.Scalar(dtype=self.dtype_fp32, name="evSL")
        self.exposure_gain = self.tik_inst.Scalar(dtype=self.dtype_fp32, name="exposure_gain")

        with self.tik_inst.new_stmt_scope():
            # bayer params
            bayer_params_ub = self.tik_inst.Tensor(self.dtype_fp32, (self.BLOCK_I32,), name="bayer_params_ub",
                                                   scope=tik.scope_ubuf)
            self.tik_inst.data_move(bayer_params_ub, self.bayer_params_gm, 0, 1, 1, 0, 0)

            self.rgb_gain_arr[0].set_as(bayer_params_ub[0])
            self.rgb_gain_arr[1].set_as(bayer_params_ub[1])
            self.rgb_gain_arr[2].set_as(bayer_params_ub[2])
            self.iso.set_as(bayer_params_ub[3])
            self.ev_gain.set_as(bayer_params_ub[4])
            self.iso_long.set_as(bayer_params_ub[5])
            self.evSL.set_as(bayer_params_ub[6])
            self.exposure_gain.set_as(bayer_params_ub[7])

        self.max_clip = self.BL_FIX_FLOAT + (1.0 - self.BL_FIX_FLOAT) * self.ev_gain

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
        self.has_last_row = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="has_last_row")
        self.combine_mode = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="combine_mode")
        self.input_lt_x = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="input_lt_x")
        self.input_lt_y = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="input_lt_y")

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
            self.has_last_row.set_as(tiling_ub[9])
            self.combine_mode.set_as(tiling_ub[10])
            self.input_lt_x.set_as(tiling_ub[11])
            self.input_lt_y.set_as(tiling_ub[12])

    def _set_a_ub_list(self, gain_list):
        for i in range(self.CHANNELS):
            a_base = self.K_A * self.iso + self.K_B
            a_new = a_base * gain_list[i]
            a = a_new / 255.0

            a_exposure_gain = a * self.exposure_gain
            self.tik_inst.vector_dup(self.MASK_I32, self.a_ub_list[i], a_exposure_gain, 1, 1, 8)

    def _set_bias_list(self, gain_list):
        self.bias_list = []

        for i in range(self.CHANNELS):
            a_base = self.K_A * self.iso_long + self.K_B
            b_base = self.B_A * (self.iso_long * self.iso_long) + self.B_B * self.iso_long + self.B_C

            gain = gain_list[i]
            a_new = a_base * gain
            b_new = b_base * gain * gain

            b = b_new / 255.0 / 255.0
            a = a_new / 255.0

            b_scalar = self.tik_inst.Scalar(dtype=self.dtype_fp32, name="b_scalar", init_value=b)
            self.tik_inst.scalar_max(b_scalar, b_scalar, 0)
            b_a = b_scalar / (a * a)

            bias = (3.0 / 8 + b_a) * gain / self.evSL
            self.bias_list.append(bias)

    def _get_gamma_and_bayer_params(self):
        """
        get gamma and bayer params
        """
        gamma_arr = self.tik_inst.ScalarArray(dtype=self.dtype_fp32, length=self.CHANNELS,
                                              name="gamma_arr", init_value=0)
        bayer_ptn_arr = self.tik_inst.ScalarArray(dtype=self.dtype_i32, length=self.CHANNELS,
                                                  name="bayer_ptn_arr", init_value=0)

        with self.tik_inst.new_stmt_scope():
            gamma_ub = self.tik_inst.Tensor(self.dtype_fp32, (self.BLOCK_I32,), name="gamma_ub",
                                            scope=tik.scope_ubuf)
            self.tik_inst.data_move(gamma_ub, self.gamma_gm, 0, 1, 1, 0, 0)

            bayer_ptn_ub = self.tik_inst.Tensor(self.dtype_i32, (self.BLOCK_I32,), name="bayer_ptn_ub",
                                                scope=tik.scope_ubuf)
            self.tik_inst.data_move(bayer_ptn_ub, self.bayer_ptn_gm, 0, 1, 1, 0, 0)

            with self.tik_inst.if_scope(self.combine_mode == 1):
                gamma_arr[0].set_as(gamma_ub[0])
                gamma_arr[1].set_as(gamma_ub[1])
                gamma_arr[2].set_as(gamma_ub[2])
                gamma_arr[3].set_as(gamma_ub[3])
                bayer_ptn_arr[0].set_as(bayer_ptn_ub[0])
                bayer_ptn_arr[1].set_as(bayer_ptn_ub[1])
                bayer_ptn_arr[2].set_as(bayer_ptn_ub[2])
                bayer_ptn_arr[3].set_as(bayer_ptn_ub[3])
            with self.tik_inst.if_scope(self.combine_mode == 2):
                gamma_arr[0].set_as(gamma_ub[1])
                gamma_arr[1].set_as(gamma_ub[0])
                gamma_arr[2].set_as(gamma_ub[3])
                gamma_arr[3].set_as(gamma_ub[2])
                bayer_ptn_arr[0].set_as(bayer_ptn_ub[1])
                bayer_ptn_arr[1].set_as(bayer_ptn_ub[0])
                bayer_ptn_arr[2].set_as(bayer_ptn_ub[3])
                bayer_ptn_arr[3].set_as(bayer_ptn_ub[2])
            with self.tik_inst.if_scope(self.combine_mode == 3):
                gamma_arr[0].set_as(gamma_ub[2])
                gamma_arr[1].set_as(gamma_ub[3])
                gamma_arr[2].set_as(gamma_ub[0])
                gamma_arr[3].set_as(gamma_ub[1])
                bayer_ptn_arr[0].set_as(bayer_ptn_ub[2])
                bayer_ptn_arr[1].set_as(bayer_ptn_ub[3])
                bayer_ptn_arr[2].set_as(bayer_ptn_ub[0])
                bayer_ptn_arr[3].set_as(bayer_ptn_ub[1])
            with self.tik_inst.if_scope(self.combine_mode == 4):
                gamma_arr[0].set_as(gamma_ub[3])
                gamma_arr[1].set_as(gamma_ub[2])
                gamma_arr[2].set_as(gamma_ub[1])
                gamma_arr[3].set_as(gamma_ub[0])
                bayer_ptn_arr[0].set_as(bayer_ptn_ub[3])
                bayer_ptn_arr[1].set_as(bayer_ptn_ub[2])
                bayer_ptn_arr[2].set_as(bayer_ptn_ub[1])
                bayer_ptn_arr[3].set_as(bayer_ptn_ub[0])

        self.gamma_list = [gamma_arr[0], gamma_arr[1], gamma_arr[2], gamma_arr[3]]
        self.gain_list = [self.rgb_gain_arr[bayer_ptn_arr[0]], self.rgb_gain_arr[bayer_ptn_arr[1]],
                          self.rgb_gain_arr[bayer_ptn_arr[2]], self.rgb_gain_arr[bayer_ptn_arr[3]]]

        self._set_a_ub_list(self.gain_list)
        self._set_bias_list(self.gain_list)

    def _move_channel_to_ub(self, channel_u16_ub, rows_idx, params):
        (loop_i, row_size, tail) = params
        burst_len = util_common.ceil(tail, self.BLOCK_U16)

        with self.tik_inst.if_scope(self.combine_mode == 1):
            index1 = (rows_idx + self.input_lt_y) * self.w_gm + self.input_lt_x + self.ROW_SLICE * loop_i
            self.tik_inst.data_move(channel_u16_ub[0], self.channel_0_gm[index1], 0, 1, burst_len, 0, 0)
            self.tik_inst.data_move(channel_u16_ub[row_size], self.channel_1_gm[index1], 0, 1, burst_len, 0, 0)
            self.tik_inst.data_move(channel_u16_ub[row_size * 2], self.channel_2_gm[index1], 0, 1, burst_len, 0, 0)
            self.tik_inst.data_move(channel_u16_ub[row_size * 3], self.channel_3_gm[index1], 0, 1, burst_len, 0, 0)

        with self.tik_inst.if_scope(self.combine_mode == 2):
            index2_0 = (rows_idx + self.input_lt_y) * self.w_gm + self.input_lt_x + self.ROW_SLICE * loop_i
            index2_1 = (rows_idx + self.input_lt_y) * self.w_gm + (self.input_lt_x + 1) + self.ROW_SLICE * loop_i
            self.tik_inst.data_move(channel_u16_ub[0], self.channel_1_gm[index2_0], 0, 1, burst_len, 0, 0)
            self.tik_inst.data_move(channel_u16_ub[row_size], self.channel_0_gm[index2_1], 0, 1, burst_len, 0, 0)
            self.tik_inst.data_move(channel_u16_ub[row_size * 2], self.channel_3_gm[index2_0], 0, 1, burst_len, 0, 0)
            self.tik_inst.data_move(channel_u16_ub[row_size * 3], self.channel_2_gm[index2_1], 0, 1, burst_len, 0, 0)

        with self.tik_inst.if_scope(self.combine_mode == 3):
            index3_0 = (rows_idx + self.input_lt_y) * self.w_gm + self.input_lt_x + self.ROW_SLICE * loop_i
            index3_1 = (rows_idx + self.input_lt_y + 1) * self.w_gm + self.input_lt_x + self.ROW_SLICE * loop_i
            self.tik_inst.data_move(channel_u16_ub[0], self.channel_2_gm[index3_0], 0, 1, burst_len, 0, 0)
            self.tik_inst.data_move(channel_u16_ub[row_size], self.channel_3_gm[index3_0], 0, 1, burst_len, 0, 0)
            self.tik_inst.data_move(channel_u16_ub[row_size * 2], self.channel_0_gm[index3_1], 0, 1, burst_len, 0, 0)
            self.tik_inst.data_move(channel_u16_ub[row_size * 3], self.channel_1_gm[index3_1], 0, 1, burst_len, 0, 0)

        with self.tik_inst.if_scope(self.combine_mode == 4):
            index4_0 = (rows_idx + self.input_lt_y) * self.w_gm + self.input_lt_x + self.ROW_SLICE * loop_i
            index4_1 = (rows_idx + self.input_lt_y) * self.w_gm + (self.input_lt_x + 1) + self.ROW_SLICE * loop_i
            index4_2 = (rows_idx + self.input_lt_y + 1) * self.w_gm + self.input_lt_x + self.ROW_SLICE * loop_i
            index4_3 = (rows_idx + self.input_lt_y + 1) * self.w_gm + (self.input_lt_x + 1) + self.ROW_SLICE * loop_i
            self.tik_inst.data_move(channel_u16_ub[0], self.channel_3_gm[index4_0], 0, 1, burst_len, 0, 0)
            self.tik_inst.data_move(channel_u16_ub[row_size], self.channel_2_gm[index4_1], 0, 1, burst_len, 0, 0)
            self.tik_inst.data_move(channel_u16_ub[row_size * 2], self.channel_1_gm[index4_2], 0, 1, burst_len, 0, 0)
            self.tik_inst.data_move(channel_u16_ub[row_size * 3], self.channel_0_gm[index4_3], 0, 1, burst_len, 0, 0)

    def _move_channel_to_ub_last_row(self, channel_u16_ub, rows_idx, params):
        (loop_i, row_size, tail) = params
        burst_len = util_common.ceil(tail, self.BLOCK_U16)

        with self.tik_inst.if_scope(self.combine_mode == 1):
            index1 = (rows_idx + self.input_lt_y) * self.w_gm + self.input_lt_x + self.ROW_SLICE * loop_i
            self.tik_inst.data_move(channel_u16_ub[0], self.channel_0_gm[index1], 0, 1, burst_len, 0, 0)
            self.tik_inst.data_move(channel_u16_ub[row_size], self.channel_1_gm[index1], 0, 1, burst_len, 0, 0)

        with self.tik_inst.if_scope(self.combine_mode == 2):
            index2_0 = (rows_idx + self.input_lt_y) * self.w_gm + self.input_lt_x + self.ROW_SLICE * loop_i
            index2_1 = (rows_idx + self.input_lt_y) * self.w_gm + (self.input_lt_x + 1) + self.ROW_SLICE * loop_i
            self.tik_inst.data_move(channel_u16_ub[0], self.channel_1_gm[index2_0], 0, 1, burst_len, 0, 0)
            self.tik_inst.data_move(channel_u16_ub[row_size], self.channel_0_gm[index2_1], 0, 1, burst_len, 0, 0)

        with self.tik_inst.if_scope(self.combine_mode == 3):
            index3 = (rows_idx + self.input_lt_y) * self.w_gm + self.input_lt_x + self.ROW_SLICE * loop_i
            self.tik_inst.data_move(channel_u16_ub[0], self.channel_2_gm[index3], 0, 1, burst_len, 0, 0)
            self.tik_inst.data_move(channel_u16_ub[row_size], self.channel_3_gm[index3], 0, 1, burst_len, 0, 0)

        with self.tik_inst.if_scope(self.combine_mode == 4):
            index4_0 = (rows_idx + self.input_lt_y) * self.w_gm + self.input_lt_x + self.ROW_SLICE * loop_i
            index4_1 = (rows_idx + self.input_lt_y) * self.w_gm + (self.input_lt_x + 1) + self.ROW_SLICE * loop_i
            self.tik_inst.data_move(channel_u16_ub[0], self.channel_3_gm[index4_0], 0, 1, burst_len, 0, 0)
            self.tik_inst.data_move(channel_u16_ub[row_size], self.channel_2_gm[index4_1], 0, 1, burst_len, 0, 0)

    def _cast_and_shift(self, channel_f32_ub, repeats, rows_idx, params):
        """
        cast uint16 to float32, and shift.
        params is (channels, row_size, loop_i, tail, tail_align)
        """
        (channels, row_size, loop_i, tail, tail_align) = params

        with self.tik_inst.new_stmt_scope():
            channel_trans_ub = self.tik_inst.Tensor(self.channel_dtype, (row_size * channels + 256,),
                                                    name="channel_trans_ub", scope=tik.scope_ubuf)
            self.tik_inst.vector_dup(self.MASK_U16, channel_trans_ub[row_size * channels], 0,
                                     256 // self.MASK_U16, 1, 8)

            with self.tik_inst.new_stmt_scope():
                channel_u16_ub = self.tik_inst.Tensor(self.channel_dtype, (row_size * channels,),
                                                      name="channel_u16_ub", scope=tik.scope_ubuf)
                if channels == self.CHANNELS:
                    self._move_channel_to_ub(channel_u16_ub, rows_idx, (loop_i, row_size, tail))
                else:
                    self._move_channel_to_ub_last_row(channel_u16_ub, rows_idx, (loop_i, row_size, tail))

                # first vnchwconv
                repeat = tail_align // 256
                with self.tik_inst.if_scope(repeat == 1):
                    for channel in range(channels):
                        src_list = [channel_u16_ub[row_size * channel + 16 * i] for i in range(16)]
                        dst_list = [channel_trans_ub[row_size * channel + 16 * i] for i in range(16)]
                        self.tik_inst.vnchwconv(False, False, dst_list, src_list, 1, 0, 0)
                with self.tik_inst.else_scope():
                    for channel in range(channels):
                        src_list = [channel_u16_ub[row_size * channel + 16 * i] for i in range(16)]
                        dst_list = [channel_trans_ub[row_size * channel + 16 * repeat * i] for i in range(16)]
                        self.tik_inst.vnchwconv(False, False, dst_list, src_list, repeat,
                                                16 // self.BLOCK_U16, 256 // self.BLOCK_U16)

            channel_ub = self.tik_inst.Tensor(self.channel_dtype, (row_size * channels * 2,),
                                              name="channel_ub", scope=tik.scope_ubuf)
            # second vnchwconv
            repeat2 = tail_align // 2 // 128
            dst_rep_stride = 256 * 2 // self.BLOCK_U16
            src_rep_stride = 16 // self.BLOCK_U16
            with self.tik_inst.if_scope(repeat2 == 1):
                for channel in range(channels):
                    index_list_1 = []
                    for i in range(0, 8):
                        index_list_1.extend([row_size * channel + 16 * i, row_size * channels])
                    src_list_1 = [channel_trans_ub[i] for i in index_list_1]
                    dst_list_1 = [channel_ub[tail_align * 2 * channel + 16 * 2 * i] for i in range(16)]

                    index_list_2 = []
                    for i in range(8, 16):
                        index_list_2.extend([row_size * channel + 16 * i, row_size * channels])
                    src_list_2 = [channel_trans_ub[i] for i in index_list_2]
                    dst_list_2 = [channel_ub[tail_align * 2 * channel + 16 * 2 * i + 16] for i in range(16)]

                    self.tik_inst.vnchwconv(False, False, dst_list_1, src_list_1, 1, 0, 0)
                    self.tik_inst.vnchwconv(False, False, dst_list_2, src_list_2, 1, 0, 0)
            with self.tik_inst.else_scope():
                for channel in range(channels):
                    index_list_1 = []
                    for i in range(0, 8):
                        index_list_1.extend([row_size * channel + 16 * repeat2 * i, row_size * channels])
                    src_list_1 = [channel_trans_ub[i] for i in index_list_1]
                    dst_list_1 = [channel_ub[tail_align * 2 * channel + 16 * 2 * i] for i in range(16)]

                    index_list_2 = []
                    for i in range(8, 16):
                        index_list_2.extend([row_size * channel + 16 * repeat2 * i, row_size * channels])
                    src_list_2 = [channel_trans_ub[i] for i in index_list_2]
                    dst_list_2 = [channel_ub[tail_align * 2 * channel + 16 * 2 * i + 16] for i in range(16)]

                    self.tik_inst.vnchwconv(False, False, dst_list_1, src_list_1, repeat2,
                                            dst_rep_stride, src_rep_stride)
                    self.tik_inst.vnchwconv(False, False, dst_list_2, src_list_2, repeat2,
                                            dst_rep_stride, src_rep_stride)

            channel_i32_ub = channel_ub.reinterpret_cast_to(self.dtype_i32)
            self.tik_inst.vconv(self.MASK_I32, "", channel_f32_ub, channel_i32_ub, repeats, 1, 1, 8, 8)

        self.tik_inst.vmuls(self.MASK_I32, channel_f32_ub, channel_f32_ub, self.BLC_REC, repeats, 1, 1, 8, 8)

    def _calculate_gamma(self, channel_f32_ub, repeats, channels, params):
        """
        calculate gamma.
        params is (row_size, data_align)
        """
        (row_size, data_align) = params

        with self.tik_inst.new_stmt_scope():
            mask_ub = self.tik_inst.Tensor(self.dtype_fp32, (row_size * channels,), name="mask_ub",
                                           scope=tik.scope_ubuf)
            self.tik_inst.vmaxs(self.MASK_I32, mask_ub, channel_f32_ub, self.BLC_1, repeats, 1, 1, 8, 8)
            self.tik_inst.vmins(self.MASK_I32, mask_ub, mask_ub, self.BLC, repeats, 1, 1, 8, 8)
            self.tik_inst.vadds(self.MASK_I32, mask_ub, mask_ub, -1 * self.BLC_1, repeats, 1, 1, 8, 8)

            d_gamma_ub = self.tik_inst.Tensor(self.dtype_fp32, (row_size * channels,), name="d_gamma_ub",
                                              scope=tik.scope_ubuf)
            self.tik_inst.vadds(self.MASK_I32, d_gamma_ub, channel_f32_ub, -1 * self.BLC, repeats, 1, 1, 8, 8)
            self.tik_inst.vdiv(self.MASK_I32, d_gamma_ub, d_gamma_ub, self.scale_ub, repeats, 1, 1, 1, 8, 8, 0)
            self.tik_inst.vmul(self.MASK_I32, d_gamma_ub, d_gamma_ub, mask_ub, repeats, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vln(self.MASK_I32, d_gamma_ub, d_gamma_ub, repeats, 1, 1, 8, 8)
            for i in range(channels):
                self.tik_inst.vmuls(self.MASK_I32, d_gamma_ub[data_align * i], d_gamma_ub[data_align * i],
                                    self.gamma_list[i], data_align // self.MASK_I32, 1, 1, 8, 8)
            self.tik_inst.vexp(self.MASK_I32, d_gamma_ub, d_gamma_ub, repeats, 1, 1, 8, 8)
            self.tik_inst.vmuls(self.MASK_I32, d_gamma_ub, d_gamma_ub, self.SCALE, repeats, 1, 1, 8, 8)
            self.tik_inst.vadds(self.MASK_I32, d_gamma_ub, d_gamma_ub, self.BLC, repeats, 1, 1, 8, 8)

            self.tik_inst.vmul(self.MASK_I32, d_gamma_ub, d_gamma_ub, mask_ub, repeats, 1, 1, 1, 8, 8, 8)

            self.tik_inst.vsub(self.MASK_I32, mask_ub, self.one_ub, mask_ub, repeats, 1, 1, 1, 8, 0, 8)
            self.tik_inst.vmul(self.MASK_I32, channel_f32_ub, channel_f32_ub, mask_ub, repeats, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vadd(self.MASK_I32, channel_f32_ub, channel_f32_ub, d_gamma_ub, repeats, 1, 1, 1, 8, 8, 8)

    def _calculate_bayer(self, channel_f32_ub, repeats, channels, data_align):
        """
        calculate bayer
        """
        with self.tik_inst.new_stmt_scope():
            self.tik_inst.vdiv(self.MASK_I32, channel_f32_ub, channel_f32_ub, self.bayer_factor_ub,
                               repeats, 1, 1, 1, 8, 8, 0)
            self.tik_inst.vadds(self.MASK_I32, channel_f32_ub, channel_f32_ub, -1 * self.BL_FIX_FLOAT,
                                repeats, 1, 1, 8, 8)
            for i in range(channels):
                self.tik_inst.vmuls(self.MASK_I32, channel_f32_ub[data_align * i], channel_f32_ub[data_align * i],
                                    self.gain_list[i], data_align // self.MASK_I32, 1, 1, 8, 8)
            self.tik_inst.vadds(self.MASK_I32, channel_f32_ub, channel_f32_ub, self.BL_FIX_FLOAT, repeats, 1, 1, 8, 8)
            self.tik_inst.vmins(self.MASK_I32, channel_f32_ub, channel_f32_ub, self.max_clip, repeats, 1, 1, 8, 8)

            self.tik_inst.vadds(self.MASK_I32, channel_f32_ub, channel_f32_ub, -1 * self.BL_FIX_FLOAT,
                                repeats, 1, 1, 8, 8)
            for i in range(channels):
                self.tik_inst.vdiv(self.MASK_I32, channel_f32_ub[data_align * i], channel_f32_ub[data_align * i],
                                   self.a_ub_list[i], data_align // self.MASK_I32, 1, 1, 1, 8, 8, 0)

            # photon * gain + bias
            for i in range(channels):
                self.tik_inst.vmuls(self.MASK_I32, channel_f32_ub[data_align * i], channel_f32_ub[data_align * i],
                                    self.gain_list[i], data_align // self.MASK_I32, 1, 1, 8, 8)
                self.tik_inst.vadds(self.MASK_I32, channel_f32_ub[data_align * i], channel_f32_ub[data_align * i],
                                    self.bias_list[i], data_align // self.MASK_I32, 1, 1, 8, 8)

            self.tik_inst.vmaxs(self.MASK_I32, channel_f32_ub, channel_f32_ub, 0, repeats, 1, 1, 8, 8)
            self.tik_inst.vsqrt(self.MASK_I32, channel_f32_ub, channel_f32_ub, repeats, 1, 1, 8, 8)
            self.tik_inst.vmuls(self.MASK_I32, channel_f32_ub, channel_f32_ub, 2, repeats, 1, 1, 8, 8)

    def _combine_to_raw_impl(self, ub_tensors, elems, idx):
        (res_ub, channel_f32_ub, trans_ub) = ub_tensors

        # first vnchwconv
        repeat = elems // 128
        src_rep_stride = 128 // self.BLOCK_I32
        dst_rep_stride = 256 // self.BLOCK_I32

        index_list = []
        for i in range(8):
            index_list.extend([32 * i, 32 * i + 8])

        src_list_1 = [channel_f32_ub[elems * 2 * idx + 8 * i] for i in range(16)]
        dst_list_1 = [trans_ub[i] for i in index_list]

        src_list_2 = [channel_f32_ub[elems * 2 * idx + elems + 8 * i] for i in range(16)]
        dst_list_2 = [trans_ub[i + 16] for i in index_list]

        with self.tik_inst.if_scope(repeat == 1):
            self.tik_inst.vnchwconv(False, False, dst_list_1, src_list_1, 1, 0, 0)
            self.tik_inst.vnchwconv(False, False, dst_list_2, src_list_2, 1, 0, 0)
        with self.tik_inst.else_scope():
            self.tik_inst.vnchwconv(False, False, dst_list_1, src_list_1, repeat, dst_rep_stride, src_rep_stride)
            self.tik_inst.vnchwconv(False, False, dst_list_2, src_list_2, repeat, dst_rep_stride, src_rep_stride)

        # second vnchwconv
        repeat2 = elems // 128
        src_rep_stride = 256 // self.BLOCK_I32
        dst_rep_stride = 256 // self.BLOCK_I32

        src_list_1 = [trans_ub[16 * i] for i in range(16)]
        dst_list_1 = [res_ub[8 * i] for i in range(16)]

        src_list_2 = [trans_ub[8 + 16 * i] for i in range(16)]
        dst_list_2 = [res_ub[128 + 8 * i] for i in range(16)]

        with self.tik_inst.if_scope(repeat2 == 1):
            self.tik_inst.vnchwconv(False, False, dst_list_1, src_list_1, 1, 0, 0)
            self.tik_inst.vnchwconv(False, False, dst_list_2, src_list_2, 1, 0, 0)
        with self.tik_inst.else_scope():
            self.tik_inst.vnchwconv(False, False, dst_list_1, src_list_1, repeat2, dst_rep_stride, src_rep_stride)
            self.tik_inst.vnchwconv(False, False, dst_list_2, src_list_2, repeat2, dst_rep_stride, src_rep_stride)

    def _combine_to_raw(self, channel_f32_ub, rows_idx, out_channels, loop_i):
        """
        combine 4 or 2 channel to raw
        """
        with self.tik_inst.new_stmt_scope():
            trans_ub = self.tik_inst.Tensor(self.dtype_fp32, (self.ROW_SLICE * 2,), name="trans_ub",
                                            scope=tik.scope_ubuf)
            res_ub = self.tik_inst.Tensor(self.dtype_fp32, (self.ROW_SLICE * 2,), name="res_ub",
                                          scope=tik.scope_ubuf)
            with self.tik_inst.for_range(0, out_channels) as idx:
                self._combine_to_raw_impl((res_ub, channel_f32_ub, trans_ub), self.ROW_SLICE, idx)

                # move result to gm
                self.tik_inst.data_move(
                    self.raw_img_gm[(rows_idx * 2 + idx) * self.w_out + self.ROW_SLICE * 2 * loop_i],
                    res_ub, 0, 1, self.ROW_SLICE * 2 // self.BLOCK_I32, 0, 0)

    def _combine_to_raw_tail(self, channel_f32_ub, rows_idx, out_channels, params):
        """
        combine 4 or 2 channel to raw.
        params is (row_size, loop_i, tail_align, res_tail)
        """
        (row_size, loop_i, tail_align, res_tail) = params

        with self.tik_inst.new_stmt_scope():
            trans_ub = self.tik_inst.Tensor(self.dtype_fp32, (row_size * 2,), name="trans_ub",
                                            scope=tik.scope_ubuf)
            res_ub = self.tik_inst.Tensor(self.dtype_fp32, (row_size * 2,), name="res_ub",
                                          scope=tik.scope_ubuf)
            with self.tik_inst.for_range(0, out_channels) as idx:
                self._combine_to_raw_impl((res_ub, channel_f32_ub, trans_ub), tail_align, idx)

                # move result to gm
                out_offset = (rows_idx * 2 + idx) * self.w_out + self.ROW_SLICE * 2 * loop_i
                self._move_res_tail_to_gm(res_ub, res_tail, out_offset)

    def _move_res_tail_to_gm(self, res_ub, res_tail, out_offset):
        """
        move tail result to out
        """
        with self.tik_inst.if_scope(res_tail % self.BLOCK_I32 == 0):
            self.tik_inst.data_move(self.raw_img_gm[out_offset], res_ub, 0, 1, res_tail // self.BLOCK_I32, 0, 0)

        with self.tik_inst.else_scope():
            block_ub = self.tik_inst.Tensor(self.dtype_fp32, (self.BLOCK_I32,), name="block_ub",
                                            scope=tik.scope_ubuf)
            last_block_offset = res_tail - self.BLOCK_I32
            with self.tik_inst.new_stmt_scope(disable_sync=True):
                for i in range(self.BLOCK_I32):
                    block_ub[i].set_as(res_ub[last_block_offset + i])
            self.tik_inst.data_move(self.raw_img_gm[out_offset], res_ub, 0, 1, res_tail // self.BLOCK_I32, 0, 0)
            self.tik_inst.data_move(self.raw_img_gm[out_offset + last_block_offset], block_ub, 0, 1, 1, 0, 0)

    def _one_row_process(self, rows_idx, channels, loop_params):
        """
        process one row.
        loop_params is (loop, tail, res_tail)
        """
        (loop, tail, res_tail) = loop_params

        with self.tik_inst.for_range(0, loop) as loop_i:
            with self.tik_inst.new_stmt_scope():
                repeats = self.ROW_SLICE * channels // self.MASK_I32
                channel_f32_ub = self.tik_inst.Tensor(self.dtype_fp32, (self.ROW_SLICE * channels,),
                                                      name="channel_f32_ub", scope=tik.scope_ubuf)
                self._cast_and_shift(channel_f32_ub, repeats, rows_idx,
                                     (channels, self.ROW_SLICE, loop_i, self.ROW_SLICE, self.ROW_SLICE))
                self._calculate_gamma(channel_f32_ub, repeats, channels, (self.ROW_SLICE, self.ROW_SLICE))
                self._calculate_bayer(channel_f32_ub, repeats, channels, self.ROW_SLICE)
                self._combine_to_raw(channel_f32_ub, rows_idx, channels // 2, loop_i)

        with self.tik_inst.if_scope(tail > 0):
            with self.tik_inst.new_stmt_scope():
                row_size = self.ROW_SLICE + 256
                tail_align = util_common.align(tail, 256)
                repeats = tail_align * channels // self.MASK_I32
                channel_f32_ub = self.tik_inst.Tensor(self.dtype_fp32, (row_size * channels,),
                                                      name="channel_f32_ub", scope=tik.scope_ubuf)
                self._cast_and_shift(channel_f32_ub, repeats, rows_idx, (channels, row_size, loop, tail, tail_align))
                self._calculate_gamma(channel_f32_ub, repeats, channels, (row_size, tail_align))
                self._calculate_bayer(channel_f32_ub, repeats, channels, tail_align)
                self._combine_to_raw_tail(channel_f32_ub, rows_idx, channels // 2,
                                          (row_size, loop, tail_align, res_tail))

    def _one_core_compute(self, core_id, rows_start_idx, rows_num):
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

        res_tail = self.tik_inst.Scalar(dtype=self.dtype_i32, name="res_tail")
        res_tail.set_as(tail * 2)
        with self.tik_inst.if_scope(self.w_out % 2 == 1):
            res_tail.set_as(tail * 2 - 1)

        with self.tik_inst.for_range(0, rows_num) as idx:
            self._one_row_process(rows_start_idx + idx, self.CHANNELS, (loop, tail, res_tail))

        with self.tik_inst.if_scope(tik.all(core_id == 0, self.has_last_row == 1)):
            self._one_row_process(self.h_in, 2, (loop, tail, res_tail))


def _check_input_params(args_list):
    """
    check input parameters.
    args_list is (img_channel_0, img_channel_1, img_channel_2, img_channel_3, gamma,
                  bayer_coordinate, bayer_params, bayer_ptn, raw_img)
    """
    (img_channel_0, img_channel_1, img_channel_2, img_channel_3, gamma,
     bayer_coordinate, bayer_params, bayer_ptn, raw_img) = args_list

    channel_0_dtype = img_channel_0.get("dtype").lower()
    channel_1_dtype = img_channel_1.get("dtype").lower()
    channel_2_dtype = img_channel_2.get("dtype").lower()
    channel_3_dtype = img_channel_3.get("dtype").lower()
    gamma_dtype = gamma.get("dtype").lower()
    bayer_coordinate_dtype = bayer_coordinate.get("dtype").lower()
    bayer_params_dtype = bayer_params.get("dtype").lower()
    bayer_ptn_dtype = bayer_ptn.get("dtype").lower()
    raw_img_dtype = raw_img.get("dtype").lower()
    para_check.check_dtype(channel_0_dtype, ("uint16",), param_name="img_channel_0")
    para_check.check_dtype(channel_1_dtype, ("uint16",), param_name="img_channel_1")
    para_check.check_dtype(channel_2_dtype, ("uint16",), param_name="img_channel_2")
    para_check.check_dtype(channel_3_dtype, ("uint16",), param_name="img_channel_3")
    para_check.check_dtype(gamma_dtype, ("float32",), param_name="gamma")
    para_check.check_dtype(bayer_coordinate_dtype, ("int32",), param_name="bayer_coordinate")
    para_check.check_dtype(bayer_params_dtype, ("float32",), param_name="bayer_params")
    para_check.check_dtype(bayer_ptn_dtype, ("int32",), param_name="bayer_ptn")
    para_check.check_dtype(raw_img_dtype, ("float32",), param_name="raw_img")

    gamma_shape = gamma.get("shape")
    para_check.check_shape(gamma_shape, min_rank=1, max_rank=1, param_name="gamma")
    if gamma_shape[0] != 4:
        error_manager_vector.raise_err_check_params_rules("ImgRawDecodePostHandleV2", "gamma shape should be (4, )")

    bayer_coordinate_shape = bayer_coordinate.get("shape")
    para_check.check_shape(bayer_coordinate_shape, min_rank=1, max_rank=1, param_name="bayer_coordinate")
    if bayer_coordinate_shape[0] != 4:
        error_manager_vector.raise_err_check_params_rules("ImgRawDecodePostHandleV2",
                                                          "bayer_coordinate shape should be (4, )")

    bayer_params_shape = bayer_params.get("shape")
    para_check.check_shape(bayer_params_shape, min_rank=1, max_rank=1, param_name="bayer_params")
    if bayer_params_shape[0] != 8:
        error_manager_vector.raise_err_check_params_rules("ImgRawDecodePostHandleV2",
                                                          "bayer_params shape should be (8, )")

    bayer_ptn_shape = bayer_ptn.get("shape")
    para_check.check_shape(bayer_ptn_shape, min_rank=1, max_rank=1, param_name="bayer_ptn")
    if bayer_ptn_shape[0] != 4:
        error_manager_vector.raise_err_check_params_rules("ImgRawDecodePostHandleV2",
                                                          "bayer_ptn shape should be (4, )")


@register_operator("ImgRawDecodePostHandleV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def img_raw_decode_post_handle_v2(img_channel_0, img_channel_1, img_channel_2, img_channel_3, gamma,
                                  bayer_coordinate, bayer_params, bayer_ptn, raw_img,
                                  kernel_name="img_raw_decode_post_handle_v2"):
    """
    ImgRawDecodePostHandleV2 op

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
    gamma: dict
        the dict of input gamma, shape is [4], value corresponds to the input gamma values of the four channels
    bayer_coordinate: dict
        the dict of input bayer_coordinate, shape is [4], value is [lt_x, lt_y, rb_x, rb_y]
    bayer_params: dict
        the dict of input bayer_params, shape is [8],
        value is [r_gain, g_gain, b_gain, iso, ev_gain, iso_long, evSL, exposure_gain]
    bayer_ptn: dict
        the dict of input bayer_ptn, shape is [4], value is the index of rgb_gain
    raw_img: dict
        the dict of output raw_img, shape is [h_out, w_out]
    kernel_name: str
        cce kernel name, default value is "img_raw_decode_post_handle_v2"

    Returns
    -------
    tik instance
    """
    args_list = (img_channel_0, img_channel_1, img_channel_2, img_channel_3, gamma,
                 bayer_coordinate, bayer_params, bayer_ptn, raw_img)
    _check_input_params(args_list)

    obj = ImgYuv2RawV2(img_channel_0, raw_img)
    obj.yuv_2_raw_v2_compute()

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
