"""
Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

extract_glimpse_v2
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl import constant_util


class Constant:
    """
    The class for constant.
    """
    MAX_INT64 = 2**63 - 1
    EIGHT_BIT = 8
    RESERVED_UB = 1024 * 8
    VNHW_MIN_NUM = 256
    TILING_NUM = 16
    TOPK_THRESHOLD = 16
    MAX_ELE_LAST_LARGE_SIZE = 512
    MAX_ONE_TIME_IMG = 10
    ONE_REPEAT_CALC_IMG_NUM = 64
    OFF_SETS_NUM = 2
    MAX_DATA_MOVE_BLOCK = 512
    SEVEN = 7
    ONE_BLOCK_FLOAT = 8
    SIXTYTHREE = 63
    # This is the position of the image width in the coordinate array
    FACTOR_WEIGHT = 0xaaaaaaaaaaaaaaaa
    # This is the position of the image height in the coordinate array
    FACTOR_HEIGHT = 0x5555555555555555


class ExtractInfo:
    """
    The class for extractinfo
    """
    slice_offset_x = 0
    slice_offset_y = 0
    slice_extent_x = 0
    slice_extent_y = 0
    base_offset_x = 0
    base_offset_y = 0


# 'pylint: disable=unused-argument,unused-variable,too-many-arguments,too-many-locals
def check_supported(input,
                    size,
                    offsets,
                    glimpse,
                    centered,
                    normalized,
                    uniform_noise,
                    noise,
                    kernel_name="extract_glimpse_v2"):
    """
    check_supported
    """
    if uniform_noise:
        return False, "The value of the uniform_noise only support False"

    if noise not in ("zero",):
        return False, "The value of the noise variable only support [zero]"
    return True, ""


class ExtractGlimpseV2:
    """
    Class for Dynamic shape for operator ExtractGlimpseV2
    """

    def __init__(self,
                 input,
                 size,
                 offsets,
                 glimpse,
                 centered=True,
                 normalized=True,
                 uniform_noise=False,
                 noise="zero",
                 kernel_name="extractglimpsev2"):
        self.kernel_name = kernel_name
        self.centered = centered
        self.normalized = normalized
        self.uniform_noise = uniform_noise
        self.noise = noise
        self.tik_instance = tik.Tik()

        self.size = self.tik_instance.Tensor("int32", (Constant.MAX_INT64,), name="size", scope=tik.scope_gm)
        self.aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.unified_buffer_size = tbe_platform.get_soc_spec("UB_SIZE")

        self.tiling_gm = self.tik_instance.Tensor("int32", (Constant.TILING_NUM,), name="tiling_gm",
                                                  scope=tik.scope_gm)
        self.tiling_ub = self.tik_instance.Tensor("int32", (Constant.TILING_NUM,),
                                                  name="tiling_ub",
                                                  scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 2, 0, 0)

        self.images_num = self.tik_instance.Scalar("int32", name="images_num")
        self.images_num.set_as(self.tiling_ub[0])
        self.aicore_num_loop = self.tik_instance.Scalar("int32", name="ai_core_num_loop", init_value=self.aicore_num)
        self.aicore_num_loop.set_as(self.tiling_ub[1])
        self.residue_image = self.tik_instance.Scalar("int32", name="residue_image")
        self.residue_image.set_as(self.tiling_ub[2])
        self.each_aicore_num = self.tik_instance.Scalar("int32", name="each_aicore_num")
        self.each_aicore_num.set_as(self.tiling_ub[3])

        self.origin_image_height = self.tik_instance.Scalar("int32", name="origin_image_height")
        self.origin_image_height.set_as(self.tiling_ub[4])
        self.origin_image_weight = self.tik_instance.Scalar("int32", name="origin_image_weight")
        self.origin_image_weight.set_as(self.tiling_ub[5])
        self.output_image_channel = self.tik_instance.Scalar("int32", name="output_image_channel")
        self.output_image_channel.set_as(self.tiling_ub[6])
        self.output_image_height = self.tik_instance.Scalar("int32", name="output_image_height")
        self.output_image_height.set_as(self.tiling_ub[7])
        self.output_image_weight = self.tik_instance.Scalar("int32", name="output_image_weight")
        self.output_image_weight.set_as(self.tiling_ub[8])

        self.virtual_each_aicore_num = self.tik_instance.Scalar("int32",
                                                                name="virtual_each_aicore_num",
                                                                init_value=self.each_aicore_num)
        self.stride_height = self.tik_instance.Scalar("int32", name="stride_height", init_value=0)
        self.stride_height.set_as(self.tiling_ub[9])
        self.stride_weight = self.tik_instance.Scalar("int32", name="stride_weight", init_value=0)
        self.stride_weight.set_as(self.tiling_ub[10])
        self.output_stride_height = self.tik_instance.Scalar("int32", name="output_stride_height", init_value=0)
        self.output_stride_height.set_as(self.tiling_ub[11])
        self.output_stride_weight = self.tik_instance.Scalar("int32", name="output_stride_weight", init_value=0)
        self.output_stride_weight.set_as(self.tiling_ub[12])
        self.new_core_num = self.tik_instance.Scalar("int32", name="new_core_num", init_value=0)
        self.new_core_num.set_as(self.tiling_ub[13])
        self.input = self.tik_instance.Tensor("float32", (Constant.MAX_INT64,), name="input", scope=tik.scope_gm)
        self.glimpse = self.tik_instance.Tensor("float32", (Constant.MAX_INT64,),
                                                name="glimpse",
                                                scope=tik.scope_gm,
                                                is_atomic_add=True)
        self.offsets = self.tik_instance.Tensor("float32", (Constant.MAX_INT64,), name="offsets",
                                                scope=tik.scope_gm)

    def compute_once_ub_offset(self, each_core_img_offset, repeat_num_each, img_num, imgs_front_of_current, is_last):
        """
        compute each ub offsets
        """
        cur_imgs_front_of_current = self.tik_instance.Scalar("int32", name="cur_imgs_front_of_current")
        cur_imgs_front_of_current.set_as(imgs_front_of_current + each_core_img_offset)
        offset_start = self.tik_instance.Scalar("int32", name="offset_start")
        offset_start.set_as(cur_imgs_front_of_current)
        y_local_start = self.tik_instance.Scalar("int32", name="y_local_start")
        y_local_start.set_as(cur_imgs_front_of_current * Constant.OFF_SETS_NUM)
        calc_mem_size = self.tik_instance.Scalar("int32", name="calc_mem_size")
        calc_mem_size.set_as(
            (img_num + Constant.SIXTYTHREE) // Constant.ONE_REPEAT_CALC_IMG_NUM * Constant.ONE_REPEAT_CALC_IMG_NUM * 2)
        # The following is the calculation process of type transform
        origin_image_weight = self.tik_instance.Scalar("float32",
                                                       name="origin_image_weight",
                                                       init_value=self.origin_image_weight)
        origin_image_height = self.tik_instance.Scalar("float32",
                                                       name="origin_image_height",
                                                       init_value=self.origin_image_height)

        tf_set_factor = self.tik_instance.Tensor("float32", (calc_mem_size,),
                                                 name="tf_set_factor",
                                                 scope=tik.scope_ubuf)
        x = self.tik_instance.Tensor("float32", (calc_mem_size,), name="x", scope=tik.scope_ubuf)
        repeat_num_each = repeat_num_each * 2
        self.tik_instance.vec_dup(Constant.ONE_REPEAT_CALC_IMG_NUM, x, 0, repeat_num_each, 8)

        with self.tik_instance.if_scope(not is_last):
            self.tik_instance.data_move(x, self.offsets[y_local_start], 0, repeat_num_each, 8, 0, 0)
        with self.tik_instance.else_scope():
            block_offset = (img_num * 2 + Constant.SEVEN) // Constant.ONE_BLOCK_FLOAT
            self.tik_instance.data_move(x, self.offsets[y_local_start], 0, 1, block_offset, 0, 0)

        with self.tik_instance.if_scope(self.normalized):
            self.tik_instance.vec_dup([0, Constant.FACTOR_WEIGHT], tf_set_factor, origin_image_weight, repeat_num_each,
                                      8)
            self.tik_instance.vec_dup([0, Constant.FACTOR_HEIGHT], tf_set_factor, origin_image_height, repeat_num_each,
                                      8)
            self.tik_instance.vec_mul(Constant.ONE_REPEAT_CALC_IMG_NUM, x, x, tf_set_factor, repeat_num_each, 8, 8, 8)

            with self.tik_instance.if_scope(self.centered):
                calc_half = self.tik_instance.Tensor("float32", (calc_mem_size,),
                                                     name="calc_half",
                                                     scope=tik.scope_ubuf)
                # The following is the calculation process of x /= 2.0f y /= 2.0f
                self.tik_instance.vec_dup(Constant.ONE_REPEAT_CALC_IMG_NUM, tf_set_factor, 0.5, repeat_num_each, 8)
                self.tik_instance.vec_mul(Constant.ONE_REPEAT_CALC_IMG_NUM, x, x, tf_set_factor, repeat_num_each, 8, 8,
                                          8)

                # The following is the calculation process of x = width_ / 2.0f y = height_ / 2.0f
                output_image_weight = self.tik_instance.Scalar("float32",
                                                               name="output_image_weight",
                                                               init_value=self.output_image_weight)
                self.tik_instance.vec_dup([0, Constant.FACTOR_WEIGHT], calc_half, output_image_weight, repeat_num_each,
                                          8)
                output_image_height = self.tik_instance.Scalar("float32",
                                                               name="output_image_height",
                                                               init_value=self.output_image_height)
                self.tik_instance.vec_dup([0, Constant.FACTOR_HEIGHT], calc_half, output_image_height, repeat_num_each,
                                          8)
                self.tik_instance.vec_mul(Constant.ONE_REPEAT_CALC_IMG_NUM, calc_half, calc_half, tf_set_factor,
                                          repeat_num_each, 8, 8, 8)
                self.tik_instance.vec_muls(Constant.ONE_REPEAT_CALC_IMG_NUM, calc_half, calc_half, -1.0,
                                           repeat_num_each, 8, 8)
                self.tik_instance.vec_add(Constant.ONE_REPEAT_CALC_IMG_NUM, x, x, calc_half, repeat_num_each, 8, 8, 0)

                # The following is the calculation process of x += input_width / 2.0f y += input_height / 2.0f
                self.tik_instance.vec_dup([0, Constant.FACTOR_WEIGHT], calc_half, origin_image_weight, repeat_num_each,
                                          8)
                self.tik_instance.vec_dup([0, Constant.FACTOR_HEIGHT], calc_half, origin_image_height, repeat_num_each,
                                          8)
                self.tik_instance.vec_mul(Constant.ONE_REPEAT_CALC_IMG_NUM, calc_half, calc_half, tf_set_factor,
                                          repeat_num_each, 8, 8, 8)
                self.tik_instance.vec_add(Constant.ONE_REPEAT_CALC_IMG_NUM, x, x, calc_half, repeat_num_each, 8, 8, 0)

        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(self.centered):
                calc_half = self.tik_instance.Tensor("float32", (calc_mem_size,),
                                                     name="calc_half",
                                                     scope=tik.scope_ubuf)
                self.tik_instance.vec_dup(Constant.ONE_REPEAT_CALC_IMG_NUM, tf_set_factor, 0.5, repeat_num_each, 8)
                # The following is the calculation process of x += input_width / 2.0;y += input_height / 2.0f;
                self.tik_instance.vec_dup([0, Constant.FACTOR_WEIGHT], calc_half, origin_image_weight, repeat_num_each,
                                          8)
                self.tik_instance.vec_dup([0, Constant.FACTOR_HEIGHT], calc_half, origin_image_height, repeat_num_each,
                                          8)
                self.tik_instance.vec_mul(Constant.ONE_REPEAT_CALC_IMG_NUM, calc_half, calc_half, tf_set_factor,
                                          repeat_num_each, 8, 8, 8)
                self.tik_instance.vec_add(Constant.ONE_REPEAT_CALC_IMG_NUM, x, x, calc_half, repeat_num_each, 8, 8, 8)

        slice_offset = self.tik_instance.Tensor("int32", (calc_mem_size,), name="slice_offset", scope=tik.scope_ubuf)
        dst_ub = self.tik_instance.Tensor("uint32", (2 * repeat_num_each,), name="dst_ub", scope=tik.scope_ubuf)
        temp = self.tik_instance.Tensor("float32", (calc_mem_size,), name="temp", scope=tik.scope_ubuf)
        temp_base = self.tik_instance.Tensor("float32", (calc_mem_size,), name="temp_base", scope=tik.scope_ubuf)
        zero = self.tik_instance.Tensor("float32", (calc_mem_size,), name="zero", scope=tik.scope_ubuf)
        self.tik_instance.vec_dup(Constant.ONE_REPEAT_CALC_IMG_NUM, zero, 0, repeat_num_each, 8)

        # The following is the calculation process of compute slice_offset
        self.tik_instance.vec_max(Constant.ONE_REPEAT_CALC_IMG_NUM, temp, x, zero, repeat_num_each, 8, 8, 8)
        self.tik_instance.h_cast(slice_offset, temp, "floor")

        # The following is the calculation process of compute slice_extent
        glimpse_width_height_lt_zero = self.tik_instance.Tensor("float32", (calc_mem_size,),
                                                                name="glimpse_width_height_lt_zero",
                                                                scope=tik.scope_ubuf)
        slice_extent = self.tik_instance.Tensor("int32", (calc_mem_size,), name="slice_extent", scope=tik.scope_ubuf)
        output_width_height = self.tik_instance.Tensor("float32", (calc_mem_size,),
                                                       name="output_width_height",
                                                       scope=tik.scope_ubuf)
        temp_output_width = self.tik_instance.Scalar("float32",
                                                     name="temp_output_width",
                                                     init_value=self.output_image_weight)
        temp_output_height = self.tik_instance.Scalar("float32",
                                                      name="temp_output_height",
                                                      init_value=self.output_image_height)
        self.tik_instance.vec_dup([0, Constant.FACTOR_WEIGHT], output_width_height, temp_output_width,
                                  repeat_num_each, 8)
        self.tik_instance.vec_dup([0, Constant.FACTOR_HEIGHT], output_width_height, temp_output_height,
                                  repeat_num_each, 8)
        input_width_height = self.tik_instance.Tensor("float32", (calc_mem_size,),
                                                      name="input_width_height",
                                                      scope=tik.scope_ubuf)
        self.tik_instance.vec_dup([0, Constant.FACTOR_WEIGHT], input_width_height, origin_image_weight,
                                  repeat_num_each, 8)
        self.tik_instance.vec_dup([0, Constant.FACTOR_HEIGHT], input_width_height, origin_image_height,
                                  repeat_num_each, 8)
        glimpse_width_height_gt = self.tik_instance.Tensor("float32", (calc_mem_size,),
                                                           name="glimpse_width_height_gt",
                                                           scope=tik.scope_ubuf)
        slice_offset_f32 = self.tik_instance.Tensor("float32", (calc_mem_size,),
                                                    name="slice_offset_f32",
                                                    scope=tik.scope_ubuf)
        self.tik_instance.h_cast(slice_offset_f32, slice_offset, "")
        self.tik_instance.vec_sub(Constant.ONE_REPEAT_CALC_IMG_NUM, glimpse_width_height_gt, input_width_height,
                                  slice_offset_f32, repeat_num_each, 8, 8, 8)

        # The following is the calculation process of compute base_offset
        base_offset = self.tik_instance.Tensor("int32", (calc_mem_size,), name="base_offset", scope=tik.scope_ubuf)
        self.tik_instance.vec_dup(Constant.ONE_REPEAT_CALC_IMG_NUM, base_offset, 0, repeat_num_each, 8)
        glimpse_base_offset_lt_zero = self.tik_instance.Tensor("float32", (calc_mem_size,),
                                                               name="glimpse_base_offset_lt_zero",
                                                               scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, repeat_num_each) as index:
            # The following is the calculation process of compute slice_extent
            offset = index * Constant.ONE_REPEAT_CALC_IMG_NUM
            self.tik_instance.vec_cmpv_ge(dst_ub, glimpse_width_height_gt[offset],
                                          output_width_height[offset], 1, 8, 8)
            self.tik_instance.vec_max(Constant.ONE_REPEAT_CALC_IMG_NUM, glimpse_width_height_gt[offset],
                                      glimpse_width_height_gt[offset], zero[offset], 1, 8, 8, 8)
            self.tik_instance.vec_sel(Constant.ONE_REPEAT_CALC_IMG_NUM, 0, temp[offset], dst_ub,
                                      output_width_height[offset], glimpse_width_height_gt[offset], 1, 8, 8, 8)
            self.tik_instance.vec_cmpv_lt(dst_ub, x[offset], zero[offset], 1, 8, 8)
            self.tik_instance.vec_add(Constant.ONE_REPEAT_CALC_IMG_NUM, glimpse_width_height_lt_zero[offset],
                                      output_width_height[offset], x[offset], 1, 8, 8, 8)
            self.tik_instance.vec_max(Constant.ONE_REPEAT_CALC_IMG_NUM, glimpse_width_height_lt_zero[offset],
                                      glimpse_width_height_lt_zero[offset], zero[offset], 1, 8, 8, 8)
            self.tik_instance.vec_sel(Constant.ONE_REPEAT_CALC_IMG_NUM, 0, temp[offset], dst_ub,
                                      glimpse_width_height_lt_zero[offset], temp[offset], 1, 8, 8, 8)
            # The following is the calculation process of compute base_offset
            self.tik_instance.vec_sub(Constant.ONE_REPEAT_CALC_IMG_NUM, glimpse_base_offset_lt_zero[offset],
                                      output_width_height[offset], glimpse_width_height_lt_zero[offset], 1, 8, 8, 8)
            self.tik_instance.vec_sel(Constant.ONE_REPEAT_CALC_IMG_NUM, 0, temp_base[offset], dst_ub,
                                      glimpse_base_offset_lt_zero[offset], zero[offset], 1, 8, 8, 8)
        self.tik_instance.vec_min(Constant.ONE_REPEAT_CALC_IMG_NUM, temp, temp, input_width_height, repeat_num_each, 8,
                                  8, 8)
        self.tik_instance.h_cast(slice_extent, temp, "ceil")
        self.tik_instance.h_cast(base_offset, temp_base, "floor")

        with self.tik_instance.for_range(0, img_num) as extract_img:
            slice_extent_y = self.tik_instance.Scalar("int32",
                                                      name="slice_extent_y1",
                                                      init_value=slice_extent[extract_img * 2])
            with self.tik_instance.if_scope(slice_extent_y != 0):
                ExtractInfo.slice_offset_x = self.tik_instance.Scalar("int32",
                                                                      name="slice_offset_x",
                                                                      init_value=slice_offset[extract_img * 2 + 1])
                ExtractInfo.slice_offset_y = self.tik_instance.Scalar("int32",
                                                                      name="slice_offset_y",
                                                                      init_value=slice_offset[extract_img * 2])
                ExtractInfo.slice_extent_x = self.tik_instance.Scalar("int32",
                                                                      name="slice_extent_x",
                                                                      init_value=slice_extent[extract_img * 2 + 1])
                ExtractInfo.slice_extent_y = self.tik_instance.Scalar("int32",
                                                                      name="slice_extent_y",
                                                                      init_value=slice_extent_y)
                ExtractInfo.base_offset_x = self.tik_instance.Scalar("int32",
                                                                     name="base_offset_x",
                                                                     init_value=base_offset[extract_img * 2 + 1])
                ExtractInfo.base_offset_y = self.tik_instance.Scalar("int32",
                                                                     name="base_offset_y",
                                                                     init_value=base_offset[extract_img * 2])
                self.extract_image(extract_img, cur_imgs_front_of_current)

    def each_core_compute(self):
        """
        The tik implementation of operator ExtractGlimpseV2
        """
        with self.tik_instance.for_range(0, self.new_core_num, block_num=self.new_core_num) as i:
            with self.tik_instance.if_scope(i < self.aicore_num_loop):
                with self.tik_instance.if_scope(i < self.residue_image):
                    self.virtual_each_aicore_num.set_as(self.each_aicore_num + 1)

                imgs_front_of_current = self.tik_instance.Scalar("int32", name="imgs_front_of_current", init_value=0)
                with self.tik_instance.if_scope(self.virtual_each_aicore_num > self.each_aicore_num):
                    imgs_front_of_current.set_as(self.virtual_each_aicore_num * i)
                with self.tik_instance.elif_scope(self.virtual_each_aicore_num == self.each_aicore_num):
                    imgs_front_of_current.set_as(self.residue_image + i * self.each_aicore_num)

                each_core_img_offset = self.tik_instance.Scalar("int32", name="each_core_img_offset", init_value=0)
                img_num = self.tik_instance.Scalar("int32", name="img_num", init_value=0)
                adds_repeat = self.tik_instance.Scalar("int32", name="adds_repeat", init_value=0)
                max_ub_calc_img_num = self.virtual_each_aicore_num // Constant.MAX_ONE_TIME_IMG \
                                      // Constant.ONE_REPEAT_CALC_IMG_NUM
                self.divide_image_to_compute(imgs_front_of_current, each_core_img_offset, img_num, adds_repeat,
                                             max_ub_calc_img_num)

    def extract_image(self, img_i, imgs_front_of_current):
        """
        extract image
        """
        with self.tik_instance.if_scope(tik.all(ExtractInfo.slice_extent_x != 0, ExtractInfo.slice_extent_y != 0)):
            with self.tik_instance.if_scope(
                    ExtractInfo.slice_extent_x * self.output_image_channel < Constant.ONE_BLOCK_FLOAT):
                self.extract_wc_not_full_block(img_i)

            with self.tik_instance.else_scope():
                self.extractglimpse_full_block(img_i, imgs_front_of_current)

    def extractglimpse_full_block(self, img_i, imgs_front_of_current):
        """
        extract image when extract image's w*c over one block size
        """
        input_offset = self.tik_instance.Scalar("int32", name="input_offset", init_value=0)
        output_offset = self.tik_instance.Scalar("int32", name="output_offset", init_value=0)
        img_move_wc_num = ExtractInfo.slice_extent_x * self.output_image_channel
        img_move_wc_block = img_move_wc_num // Constant.ONE_BLOCK_FLOAT
        img_move_tail_num = Constant.ONE_BLOCK_FLOAT - img_move_wc_num % Constant.ONE_BLOCK_FLOAT
        input_data_ub = self.tik_instance.Tensor("float32", (Constant.MAX_DATA_MOVE_BLOCK * Constant.ONE_BLOCK_FLOAT,),
                                                 name="input_data_ub",
                                                 scope=tik.scope_ubuf)
        move_num = img_move_wc_block // Constant.MAX_DATA_MOVE_BLOCK
        residue_block = img_move_wc_block % Constant.MAX_DATA_MOVE_BLOCK

        with self.tik_instance.for_range(0, ExtractInfo.slice_extent_y) as index_y:
            with self.tik_instance.for_range(0, move_num) as move:
                input_offset.set_as((imgs_front_of_current + img_i) * self.stride_height +
                                    (ExtractInfo.slice_offset_y + index_y) * self.stride_weight +
                                    ExtractInfo.slice_offset_x * self.output_image_channel +
                                    move * Constant.MAX_DATA_MOVE_BLOCK * Constant.ONE_BLOCK_FLOAT)
                output_offset.set_as((imgs_front_of_current + img_i) * self.output_stride_height +
                                     (ExtractInfo.base_offset_y + index_y) * self.output_stride_weight +
                                     ExtractInfo.base_offset_x * self.output_image_channel +
                                     move * Constant.MAX_DATA_MOVE_BLOCK * Constant.ONE_BLOCK_FLOAT)
                self.tik_instance.data_move(input_data_ub, self.input[input_offset], 0, 1,
                                            Constant.MAX_DATA_MOVE_BLOCK, 0, 0)
                self.tik_instance.data_move(self.glimpse[output_offset], input_data_ub, 0, 1,
                                            Constant.MAX_DATA_MOVE_BLOCK, 0, 0)
            with self.tik_instance.if_scope(residue_block > 0):
                input_offset.set_as((imgs_front_of_current + img_i) * self.stride_height +
                                    (ExtractInfo.slice_offset_y + index_y) * self.stride_weight +
                                    ExtractInfo.slice_offset_x * self.output_image_channel +
                                    move_num * Constant.MAX_DATA_MOVE_BLOCK * Constant.ONE_BLOCK_FLOAT)
                output_offset.set_as((imgs_front_of_current + img_i) * self.output_stride_height +
                                     (ExtractInfo.base_offset_y + index_y) * self.output_stride_weight +
                                     ExtractInfo.base_offset_x * self.output_image_channel +
                                     move_num * Constant.MAX_DATA_MOVE_BLOCK * Constant.ONE_BLOCK_FLOAT)
                self.tik_instance.data_move(input_data_ub, self.input[input_offset], 0, 1, residue_block, 0, 0)
                self.tik_instance.data_move(self.glimpse[output_offset], input_data_ub, 0, 1, residue_block, 0, 0)
            with self.tik_instance.if_scope(img_move_tail_num < Constant.ONE_BLOCK_FLOAT):
                input_offset.set_as((imgs_front_of_current + img_i) * self.stride_height +
                                    (ExtractInfo.slice_offset_y + index_y) * self.stride_weight +
                                    ExtractInfo.slice_offset_x * self.output_image_channel +
                                    img_move_wc_block * Constant.ONE_BLOCK_FLOAT - img_move_tail_num)
                output_offset.set_as((imgs_front_of_current + img_i) * self.output_stride_height +
                                     (ExtractInfo.base_offset_y + index_y) * self.output_stride_weight +
                                     ExtractInfo.base_offset_x * self.output_image_channel +
                                     img_move_wc_block * Constant.ONE_BLOCK_FLOAT - img_move_tail_num)
                self.tik_instance.data_move(input_data_ub, self.input[input_offset], 0, 1, 1, 0, 0)
                self.tik_instance.data_move(self.glimpse[output_offset], input_data_ub, 0, 1, 1, 0, 0)

    def extract_wc_not_full_block(self, img_i):
        """
        extract image when extract image's w*c not full one block size
        """
        input_start_location = self.tik_instance.Scalar("int32", name="input_start_location", init_value=0)
        output_start_location = self.tik_instance.Scalar("int32", name="output_start_location", init_value=0)
        input_data_ub = self.tik_instance.Tensor("float32", (Constant.MAX_DATA_MOVE_BLOCK * Constant.ONE_BLOCK_FLOAT,),
                                                 name="input_data_ub",
                                                 scope=tik.scope_ubuf)
        extent_wc = self.tik_instance.Scalar("int32",
                                             name="extent_wc",
                                             init_value=self.output_image_channel * ExtractInfo.slice_extent_x)
        c_return_num = self.tik_instance.Scalar("int32", name="c_return_num", init_value=0)
        c_return_num.set_as(Constant.ONE_BLOCK_FLOAT - extent_wc)

        with self.tik_instance.for_range(0, ExtractInfo.slice_extent_y) as index_y:
            # not full a block,can't return address
            input_start_location.set_as(img_i * self.stride_height +
                                        (ExtractInfo.slice_offset_y + index_y) * self.stride_weight +
                                        ExtractInfo.slice_offset_x * self.output_image_channel)
            output_start_location.set_as(img_i * self.output_stride_height +
                                         (ExtractInfo.base_offset_y + index_y) * self.output_stride_weight +
                                         ExtractInfo.base_offset_x * self.output_image_channel)

            self.tik_instance.data_move(input_data_ub, self.input[input_start_location], 0, 1, 1, 0, 0)
            # set over factor as 0
            with self.tik_instance.if_scope(c_return_num != 0):
                with self.tik_instance.for_range(0, c_return_num) as over:
                    input_data_ub[extent_wc + over] = 0
            self.tik_instance.data_move(self.glimpse[output_start_location], input_data_ub, 0, 1, 1, 0, 0)

    def divide_image_to_compute(self, imgs_front_of_current, each_core_img_offset, img_num, adds_repeat,
                                max_ub_calc_img_num):
        """
        divide images to compute
        """
        # frequency of the largest of every core deal
        adds_repeat.set_as(Constant.MAX_ONE_TIME_IMG)
        with self.tik_instance.for_range(0, max_ub_calc_img_num) as index:
            each_core_img_offset.set_as(index * Constant.MAX_ONE_TIME_IMG * Constant.ONE_REPEAT_CALC_IMG_NUM)
            img_num.set_as(Constant.MAX_ONE_TIME_IMG * Constant.ONE_REPEAT_CALC_IMG_NUM)
            self.compute_once_ub_offset(each_core_img_offset, adds_repeat, img_num, imgs_front_of_current, False)

        # frequency of residue images
        adds_repeat.set_as(self.virtual_each_aicore_num %
                           (Constant.MAX_ONE_TIME_IMG * Constant.ONE_REPEAT_CALC_IMG_NUM) //
                           Constant.ONE_REPEAT_CALC_IMG_NUM)
        with self.tik_instance.if_scope(adds_repeat > 0):
            each_core_img_offset.set_as(max_ub_calc_img_num * Constant.MAX_ONE_TIME_IMG *
                                        Constant.ONE_REPEAT_CALC_IMG_NUM)
            img_num.set_as(adds_repeat * Constant.ONE_REPEAT_CALC_IMG_NUM)
            self.compute_once_ub_offset(each_core_img_offset, adds_repeat, img_num, imgs_front_of_current, False)

        # residue img not full 64 piece
        last_img_num = self.virtual_each_aicore_num % Constant.ONE_REPEAT_CALC_IMG_NUM
        with self.tik_instance.if_scope(last_img_num > 0):
            each_core_img_offset.set_as(max_ub_calc_img_num * Constant.MAX_ONE_TIME_IMG *
                                        Constant.ONE_REPEAT_CALC_IMG_NUM +
                                        adds_repeat * Constant.ONE_REPEAT_CALC_IMG_NUM)
            img_num.set_as(last_img_num)
            adds_repeat.set_as(1)
            self.compute_once_ub_offset(each_core_img_offset, adds_repeat, img_num, imgs_front_of_current, True)


@register_operator("ExtractGlimpseV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def extract_glimpse_v2(input,
                       size,
                       offsets,
                       glimpse,
                       centered=True,
                       normalized=True,
                       uniform_noise=False,
                       noise="zero",
                       kernel_name="extract_glimpse_v2"):
    """
    Generate arg_min operator use arg_min

    Parameters
    ----------
    input: dict
        data of input, support "float32".
    size: dict
        size of glimpse, support "int32".
    offsets: dict
        location of glimpse image, support "float32".
    glimpse: dict
        shape of glimpse, support "float32".
    centered: bool
        indicates if the offset coordinates are centered relative to the image.
    normalized: bool
        indicate if the offset coordinate are normalized.
    uniform_noise: bool
        indicates if the noise should be generated uniform or Gaussian distribution.
    noise: str
        indicate if the noise should uniform, gaussian, or zero.
    kernel_name: str
        kernel name, default value is "extract_glimpse_v2"

    Returns
    -------
    tik_instance
    """
    input_dtype = input.get("dtype")
    size_dtype = size.get("dtype")
    offsets_dtype = offsets.get("dtype")
    output_dtype = glimpse.get("dtype")
    obj = ExtractGlimpseV2(input_dtype,
                           size_dtype,
                           offsets_dtype,
                           output_dtype,
                           centered,
                           normalized,
                           kernel_name=kernel_name)
    obj.each_core_compute()

    opt_config = {"out_of_bound_sync_check": True}
    tbe_context.get_context().add_compile_info("vars",
                                               {"ub_size": obj.unified_buffer_size, "core_num": obj.aicore_num})
    obj.tik_instance.BuildCCE(kernel_name=obj.kernel_name,
                              inputs=(obj.input, obj.size, obj.offsets),
                              outputs=obj.glimpse,
                              flowtable=obj.tiling_gm,
                              config=opt_config)

    return obj.tik_instance
