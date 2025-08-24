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
rgb_2_yuv422
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


class RGB2YUV422:
    """
    RGB2YUV422 class
    """
    DTYPE_SIZE = {
        'uint8': 1,
        'float16': 2,
        'float32': 4
    }
    # int32's max value
    MAX_SHAPE_SIZE = constant.SHAPE_SIZE_LIMIT
    # tiling param num
    TILING_ARG_NUM = 12
    # reserved ub size
    RESERVED_UB_SIZE = 24 * 1024
    # 8 bit
    EIGHT_BIT = 8
    # bytes of one block
    BLOCK_BYTES = constant.BLOCK_SIZE
    # 256 bytes
    VECTOR_BYTES = constant.VECTOR_BYTE_SIZE
    MAX_REPEAT_TIMES = constant.MAX_REPEAT_TIMES
    BLOCK_PER_REPEAT = 8
    U8_DTYPE = "uint8"
    FP16_DTYPE = "float16"
    NUM_U8_PER_BLOCK = BLOCK_BYTES // DTYPE_SIZE.get(U8_DTYPE)
    NUM_FP16_PER_BLOCK = BLOCK_BYTES // DTYPE_SIZE.get(FP16_DTYPE)
    U8_REPEAT_SIZE = BLOCK_PER_REPEAT * NUM_U8_PER_BLOCK
    FP16_REPEAT_SIZE = BLOCK_PER_REPEAT * NUM_FP16_PER_BLOCK

    ONE_PIXEL_ELEMS = 3
    ONE_PIXEL_L1_ELEMS = 4
    ONE_PIXEL_OUT_ELEMS = 2
    ONE_PIXEL_UB_WORKSPACE_BYTES = 14
    YUV_422_PATTERN_SCALAR = 2 ** 0 + 2 ** 1 + 2 ** 4 + 2 ** 6 + 2 ** 8 + 2 ** 9 + 2 ** 12 + 2 ** 14

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
    VECTOR_FP16_MASK = 128

    FORMAT_CONVERT = 0
    CSC_MATRIX = [[77, 150, 29],
                  [-43, -85, 128],
                  [128, -107, -21]]
    CSC_OUT_BIAS = [0, 128, 128]
    CSC_IN_BIAS = [0, 0, 0]

    def __init__(self, rgb, yuv):
        """
        Init RGB2YUV422 parameters
        """
        self.tik_inst = tik.Tik()
        self.dtype = rgb.get("dtype").lower()
        self.shape_img = rgb.get("shape")

        self.dsize = tbe_platform.get_bit_len(self.dtype) // self.EIGHT_BIT
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - self.RESERVED_UB_SIZE
        self.block_elems = self.BLOCK_BYTES // self.dsize
        self.vector_elems = self.VECTOR_BYTES // self.dsize

        self.tiling_dtype = "int64"
        self.tiling_align = aligned_value(self.TILING_ARG_NUM, self.NUM_4)
        self.tiling_gm = self.tik_inst.Tensor(self.tiling_dtype, (self.tiling_align,), name="tiling_gm",
                                              scope=tik.scope_gm)
        self.rgb_gm, self.yuv_gm = self._init_gm_tensor()

        # tiling params
        self.need_core_num = None
        self.input_rgb_elems = None
        self.input_rgb_pixels = None
        self.one_portion_ub_pixels = None
        self.pre_core_pixels = None
        self.last_core_pixels = None
        self.pre_core_l1_pixels = None
        self.last_core_l1_pixels = None
        self.pre_core_l1_loops = None
        self.last_core_l1_loops = None
        self.pre_core_l1_pixels_tail = None
        self.last_core_l1_pixels_tail = None

    def rgb_2_yuv422_compute(self):
        """
        main process of rgb_2_yuv422
        """
        with self.tik_inst.for_range(0, self.core_num, block_num=self.core_num) as core_i:
            # get tiling data
            self._get_tiling_args()
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
        rgb_gm = self.tik_inst.Tensor(self.dtype, (self.MAX_SHAPE_SIZE,), name="rgb_gm", scope=tik.scope_gm)
        yuv_gm = self.tik_inst.Tensor(self.dtype, (self.MAX_SHAPE_SIZE,), name="yuv_gm",
                                      scope=tik.scope_gm)

        return [rgb_gm, yuv_gm]

    def _get_tiling_args(self):
        """
        get runtime params from tiling data
        """
        self.need_core_num = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="need_core_num")
        self.input_rgb_elems = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="input_rgb_elems")
        self.input_rgb_pixels = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="input_rgb_pixels")
        self.one_portion_ub_pixels = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="one_portion_ub_pixels")
        self.pre_core_pixels = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="pre_core_pixels")
        self.last_core_pixels = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="last_core_pixels")
        self.pre_core_l1_pixels = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="pre_core_l1_pixels")
        self.last_core_l1_pixels = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="last_core_l1_pixels")
        self.pre_core_l1_loops = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="pre_core_l1_loops")
        self.last_core_l1_loops = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="last_core_l1_loops")
        self.pre_core_l1_pixels_tail = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="pre_core_l1_pixels_tail")
        self.last_core_l1_pixels_tail = self.tik_inst.Scalar(dtype=self.tiling_dtype, name="last_core_l1_pixels_tail")

        with self.tik_inst.new_stmt_scope():
            tiling_ub = self.tik_inst.Tensor(self.tiling_dtype, (self.tiling_align,), name="tiling_ub",
                                             scope=tik.scope_ubuf)
            self.tik_inst.data_move(tiling_ub, self.tiling_gm, 0, 1, self.tiling_align // self.NUM_4, 0, 0)

            self.need_core_num.set_as(tiling_ub[self.NUM_0])
            self.input_rgb_elems.set_as(tiling_ub[self.NUM_1])
            self.input_rgb_pixels.set_as(tiling_ub[self.NUM_2])
            self.one_portion_ub_pixels.set_as(tiling_ub[self.NUM_3])
            self.pre_core_pixels.set_as(tiling_ub[self.NUM_4])
            self.last_core_pixels.set_as(tiling_ub[self.NUM_5])
            self.pre_core_l1_pixels.set_as(tiling_ub[self.NUM_6])
            self.last_core_l1_pixels.set_as(tiling_ub[self.NUM_7])
            self.pre_core_l1_loops.set_as(tiling_ub[self.NUM_8])
            self.last_core_l1_loops.set_as(tiling_ub[self.NUM_9])
            self.pre_core_l1_pixels_tail.set_as(tiling_ub[self.NUM_10])
            self.last_core_l1_pixels_tail.set_as(tiling_ub[self.NUM_11])

    def _pre_core_compute(self, core_id):
        """
        compute for pre core
        """
        with self.tik_inst.for_range(0, self.pre_core_l1_loops) as loop_i:
            with self.tik_inst.if_scope(loop_i < self.pre_core_l1_loops - 1):
                with self.tik_inst.new_stmt_scope():
                    work_pixels = self.pre_core_l1_pixels
                    input_offset = (core_id * self.pre_core_pixels + loop_i * self.pre_core_l1_pixels) \
                                   * self.ONE_PIXEL_ELEMS
                    output_offset = (core_id * self.pre_core_pixels + loop_i * self.pre_core_l1_pixels) \
                                    * self.ONE_PIXEL_OUT_ELEMS
                    self._load_image(work_pixels, input_offset, output_offset)

            with self.tik_inst.if_scope(loop_i == self.pre_core_l1_loops - 1):
                with self.tik_inst.new_stmt_scope():
                    work_pixels = aligned_value(self.pre_core_l1_pixels_tail, self.NUM_FP16_PER_BLOCK)
                    input_offset = ((core_id + 1) * self.pre_core_pixels - work_pixels) \
                                   * self.ONE_PIXEL_ELEMS
                    output_offset = ((core_id + 1) * self.pre_core_pixels - work_pixels) \
                                    * self.ONE_PIXEL_OUT_ELEMS
                    self._load_image(work_pixels, input_offset, output_offset)

    def _last_core_compute(self, core_id):
        """
        compute for last core
        """
        with self.tik_inst.for_range(0, self.last_core_l1_loops) as loop_i:
            with self.tik_inst.if_scope(loop_i < self.last_core_l1_loops - 1):
                with self.tik_inst.new_stmt_scope():
                    work_pixels = self.last_core_l1_pixels
                    input_offset = (core_id * self.last_core_pixels + loop_i * self.last_core_l1_pixels) \
                                   * self.ONE_PIXEL_ELEMS
                    output_offset = (core_id * self.last_core_pixels + loop_i * self.last_core_l1_pixels) \
                                    * self.ONE_PIXEL_OUT_ELEMS
                    self._load_image(work_pixels, input_offset, output_offset)

            with self.tik_inst.if_scope(loop_i == self.last_core_l1_loops - 1):
                with self.tik_inst.new_stmt_scope():
                    work_pixels = aligned_value(self.last_core_l1_pixels_tail, self.NUM_FP16_PER_BLOCK)
                    input_offset = ((core_id + 1) * self.last_core_pixels - work_pixels) \
                                   * self.ONE_PIXEL_ELEMS
                    output_offset = ((core_id + 1) * self.last_core_pixels - work_pixels) \
                                    * self.ONE_PIXEL_OUT_ELEMS
                    self._load_image(work_pixels, input_offset, output_offset)

    def _load_image(self, work_pixels, input_offset, output_offset):
        src_info = {
            'src_horizontal_size': 32,
            'src_vertical_size': work_pixels // 32
        }
        input_format = 4
        function_switch = 2 ** 3 + 2 ** 6 + 2 ** 9
        csc_info = {
            'format_convert': self.FORMAT_CONVERT,
            'csc_matrix': self.CSC_MATRIX,
            'csc_out_bias': self.CSC_OUT_BIAS,
            'csc_in_bias': self.CSC_IN_BIAS
        }
        dtc_info = {
            'dtc_mean_type': 1,
            'dtc_mean': [0.0, 0.0, 0.0, 0.0],
            'dtc_min': [0.0, 0.0, 0.0, 0.0],
            'dtc_var': [1.0, 1.0, 1.0, 1.0],
            'raw_to_f16_n': 0
        }
        channel_pad_info = {
            'channel_pad_mode': 2,
            'channel_pad_value': 0.0
        }
        # 未使能参数
        crop_info = {
            'dst_horizontal_size': 32,
            'dst_vertical_size': work_pixels // 32,
            'crop_horizontal_start': 0,
            'crop_vertical_start': 0,
            'single_line_enable': 0
        }
        pre_clip_info = {
            'pre_top_clip_number': 0,
            'pre_botton_clip_number': 0
        }
        post_clip_info = {
            'post_left_clip_number': 0,
            'post_right_clip_number': 0,
            'post_top_clip_number': 0,
            'post_botton_clip_number': 0
        }
        swap_list = [0, 0, 0]
        flip_mode = 0
        stretch_info = {
            'dst_stride_pixel': 16,
        }
        raw_info = {
            'raw_image_channel': 0,
            'raw_start_channel': 0,
        }
        scf_info = {
            'scf_horizontal_size': 16,
            'scf_vertical_size': work_pixels // 16,
            'scaling_mode': 0,
            'scf_horizontal_start': 0,
            'scf_vertical_start': 0
        }
        area_pad_info = {
            'area_pad_mode': 0,
            'top_pad_rows': 1,
            'botton_pad_rows': 1,
            'left_pad_cols': 1,
            'right_pad_cols': 1,
            'channel0_pad_value': 0,
            'channel1_pad_value': 0,
            'channel2_pad_value': 0,
            'channel3_pad_value': 0
        }
        sid = 0
        l1_work_pixels = aligned_value(work_pixels, self.vector_elems)
        yuv_l1 = self.tik_inst.Tensor("float16", (l1_work_pixels * self.ONE_PIXEL_L1_ELEMS,), name="yuv_l1",
                                      scope=tik.scope_cbuf)
        self.tik_inst.load_image(yuv_l1, self.rgb_gm[input_offset], None, input_format,
                                 function_switch, src_info, crop_info, pre_clip_info, swap_list,
                                 csc_info, scf_info, post_clip_info, dtc_info, flip_mode,
                                 channel_pad_info, area_pad_info, stretch_info, raw_info, sid)
        one_portion_ub_pixels = self.tik_inst.Scalar(dtype="int32")
        with self.tik_inst.if_scope(work_pixels > self.one_portion_ub_pixels):
            one_portion_ub_pixels.set_as(self.one_portion_ub_pixels)
        with self.tik_inst.else_scope():
            one_portion_ub_pixels.set_as(work_pixels)
        one_portion_ub_loops = ceiling_value(work_pixels, self.one_portion_ub_pixels)

        with self.tik_inst.for_range(0, one_portion_ub_loops) as loop_i:
            with self.tik_inst.if_scope(loop_i < one_portion_ub_loops - 1):
                with self.tik_inst.new_stmt_scope():
                    ub_pixels = one_portion_ub_pixels
                    ub_work_pixels = ub_pixels
                    l1_offset = self.one_portion_ub_pixels * loop_i * self.ONE_PIXEL_L1_ELEMS
                    gm_offset = output_offset + self.one_portion_ub_pixels * loop_i * self.ONE_PIXEL_OUT_ELEMS
                    self._one_loop_compute(yuv_l1, ub_pixels, ub_work_pixels, l1_offset, gm_offset)

            with self.tik_inst.if_scope(loop_i == one_portion_ub_loops - 1):
                with self.tik_inst.new_stmt_scope():
                    ub_pixels = work_pixels - self.one_portion_ub_pixels * (one_portion_ub_loops - 1)
                    ub_work_pixels = aligned_value(ub_pixels, self.vector_elems)
                    l1_offset = self.one_portion_ub_pixels * loop_i * self.ONE_PIXEL_L1_ELEMS
                    gm_offset = output_offset + self.one_portion_ub_pixels * loop_i * self.ONE_PIXEL_OUT_ELEMS
                    self._one_loop_compute(yuv_l1, ub_pixels, ub_work_pixels, l1_offset, gm_offset)

    def _one_loop_compute(self, yuv_l1, ub_pixel, ub_work_pixels, l1_offset, gm_offset):
        yuv_ub = self.tik_inst.Tensor("float16", (ub_work_pixels * self.ONE_PIXEL_L1_ELEMS,), tik.scope_ubuf, "yuv_ub")
        yuv_422_ub_fp16 = self.tik_inst.Tensor("float16", (ub_work_pixels * self.ONE_PIXEL_OUT_ELEMS,), tik.scope_ubuf,
                                               "yuv_422_ub_fp16")
        yuv_422_ub_u8 = self.tik_inst.Tensor("uint8", (ub_work_pixels * self.ONE_PIXEL_OUT_ELEMS,), tik.scope_ubuf,
                                             "yuv_422_ub_u8")
        yuv_422_pattern = self.tik_inst.Tensor("uint16", (8,), tik.scope_ubuf, "yuv_422_pattern")

        self.tik_inst.vector_dup(8, yuv_422_pattern, self.YUV_422_PATTERN_SCALAR, 1, 0, 0)
        self.tik_inst.data_move(yuv_ub, yuv_l1[l1_offset], 0, 1,
                                ub_work_pixels * self.ONE_PIXEL_L1_ELEMS // self.NUM_FP16_PER_BLOCK, 0, 0)

        vreduce_times = ub_work_pixels * self.ONE_PIXEL_L1_ELEMS // self.VECTOR_FP16_MASK
        vreduce_loops = vreduce_times // self.MAX_REPEAT_TIMES
        with self.tik_inst.for_range(0, vreduce_loops) as i:
            yuv_offset = i * self.FP16_REPEAT_SIZE * self.MAX_REPEAT_TIMES
            yuv_422_offset = yuv_offset // 2
            self.tik_inst.vreduce(self.VECTOR_FP16_MASK, yuv_422_ub_fp16[yuv_422_offset:], yuv_ub[yuv_offset:],
                                  yuv_422_pattern, self.MAX_REPEAT_TIMES, 1, 8, 0, mask_mode="normal")
        yuv_offset = vreduce_loops * self.FP16_REPEAT_SIZE * self.MAX_REPEAT_TIMES
        yuv_422_offset = yuv_offset // 2
        vreduce_times = vreduce_times - vreduce_loops * self.MAX_REPEAT_TIMES
        self.tik_inst.vreduce(self.VECTOR_FP16_MASK, yuv_422_ub_fp16[yuv_422_offset:], yuv_ub[yuv_offset:],
                              yuv_422_pattern, vreduce_times, 1, 8, 0, mask_mode="normal")

        vconv_times = ub_work_pixels * self.ONE_PIXEL_OUT_ELEMS // self.VECTOR_FP16_MASK
        vconv_loops = vconv_times // self.MAX_REPEAT_TIMES
        with self.tik_inst.for_range(0, vconv_loops) as i:
            offset = i * self.FP16_REPEAT_SIZE * self.MAX_REPEAT_TIMES
            self.tik_inst.vec_conv(self.VECTOR_FP16_MASK, "none", yuv_422_ub_u8[offset:], yuv_422_ub_fp16[offset:],
                                   self.MAX_REPEAT_TIMES, 4, 8)
        vconv_times = vconv_times - vconv_loops * self.MAX_REPEAT_TIMES
        offset = vconv_loops * self.FP16_REPEAT_SIZE * self.MAX_REPEAT_TIMES
        self.tik_inst.vec_conv(self.VECTOR_FP16_MASK, "none", yuv_422_ub_u8[offset:], yuv_422_ub_fp16[offset:],
                               vconv_times, 4, 8)
        self.tik_inst.data_move(self.yuv_gm[gm_offset], yuv_422_ub_u8, 0, 1,
                                ub_pixel * self.ONE_PIXEL_OUT_ELEMS // self.NUM_U8_PER_BLOCK, 0, 0)


def _check_input_params(rgb, yuv, kernel_name):
    """
    check input parameters
    """
    rgb_dtype = rgb.get("dtype").lower()
    yuv_dtype = yuv.get("dtype").lower()

    input_check_list = ("uint8")
    para_check.check_dtype(rgb_dtype, input_check_list, param_name="rgb")
    para_check.check_dtype(yuv_dtype, input_check_list, param_name="yuv")

    rgb_shape = rgb.get("shape")
    para_check.check_shape(rgb_shape, min_rank=3, max_rank=3, param_name="rgb")

    yuv_shape = yuv.get("shape")
    para_check.check_shape(yuv_shape, min_rank=3, max_rank=3, param_name="yuv")


@register_operator("RGB2YUV422")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def rgb_2_yuv422(rgb, yuv, kernel_name="rgb_2_yuv422"):
    """
    RGB2YUV422 op

    Parameters
    ----------
    rgb: dict
        the dict of input tensor, 3D, shape of [h,w,c].
    yuv: dict
        the dict of input tensor, 3D, shape of [h,w,2].
    kernel_name: str
        cce kernel name, default value is "rgb_2_yuv422".

    Returns
    -------
    tik_instance
    """
    _check_input_params(rgb, yuv, kernel_name)

    obj = RGB2YUV422(rgb, yuv)
    obj.rgb_2_yuv422_compute()

    # add compile info
    tbe_context.get_context().add_compile_info("vars", {
        "core_num": obj.core_num,
        "ub_size": obj.ub_size,
        "block_elems": obj.block_elems,
        "vector_elems": obj.vector_elems,
    })

    tik_inst = obj.tik_inst
    tik_inst.BuildCCE(kernel_name=kernel_name,
                      inputs=(obj.rgb_gm,),
                      outputs=(obj.yuv_gm,),
                      flowtable=(obj.tiling_gm,))

    return tik_inst
