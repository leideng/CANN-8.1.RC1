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

image_projective_transform
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.util_common import ceil_div_scalar as ceil_div
from impl import constant_util
from tbe.common.platform import get_bit_len


# 'pylint: disable=unused-argument,too-many-boolean-expressions,too-many-arguments
def check_supported(images,
                    transforms,
                    output_shape,
                    transformed_image,
                    interpolation,
                    fill_mode="CONSTANT",
                    kernel_name="image_projective_transform"):
    """
    Verify the last dim of images
    """
    img_shape = images.get("shape")
    img_dtype = images.get("dtype").lower()

    if (img_shape[3] < 8 and img_dtype == "float32") or (
            img_shape[3] < 8 and img_dtype == "int32") or (
            img_shape[3] < 16 and img_dtype == "float16") or (
            img_shape[3] < 32 and img_dtype == "uint8"):

        reason = "aicore do not support the last dim is smaller than 1 block space"
        return False, reason
    else:
        return True, ""


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # tiling param num
    TILING_ARG_NUM = 32
    # 8 bit
    EIGHT_BIT = 8
    # 7 bit
    SEVEN_BIT = 7
    # ub size
    MAX_DATA_UB = 1024
    # ub size for block
    BLC_MAX_DATA_UB = 512
    # 64 num for fp32
    FP32_NUM = 64
    # 64 num for fp32
    CEIL_NUM = 63
    # float one
    FLOAT_ONE = 1
    # repeat size
    REP_SIZE = 256
    # block byte size
    BLC_BYTE_SIZE = 32


# 'pylint: disable=too-many-lines,too-many-public-methods,too-many-instance-attributes,too-many-arguments
# 'pylint: disable=too-many-locals, too-many-statements
# 'pylint: disable=attribute-defined-outside-init
class ImageProjectiveTransform:
    """
    Class for Dynamic shape operator ImageProjectiveTransform
    """

    def __init__(self, images_dtype, transform_dtype, interpolation, fill_mode, kernel_name):
        # check interpolation
        if interpolation not in ("NEAREST", "BILINEAR"):
            error_manager_vector.raise_err_input_value_invalid(kernel_name, "interpolation", "NEAREST,BILINEAR",
                                                               interpolation)

        # check fill mode
        if fill_mode not in ("REFLECT", "WRAP", "CONSTANT", "NEAREST"):
            error_manager_vector.raise_err_input_value_invalid(kernel_name, "fill_mode",
                                                               "REFLECT,WRAP,CONSTANT,NEAREST",
                                                               fill_mode)
        self.tik_instance = tik.Tik()
        self.tmp_dtype = "float16"
        self.dtype = images_dtype
        if images_dtype == "float32":
            self.flag = 0
        elif images_dtype == "float16":
            self.flag = 1
        elif images_dtype == "int32":
            self.flag = 2
        elif images_dtype == "uint8":
            self.flag = 3
        self.trans_dtype = transform_dtype
        self.aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.dtype_size = get_bit_len(self.dtype) // Constant.EIGHT_BIT
        self.kernel_name = kernel_name
        self.exist_fill_value_n = False
        self.tiling_gm = None
        self.input_gm = None
        self.transform_gm = None
        self.output_gm = None
        self.output_shape_gm = None
        self.fill_value_gm = None
        # NEAREST:0 BILINEAR:1
        self.interpolation = 0 if interpolation == "NEAREST" else 1
        # CONSTANT:0 REFLECT:1 WRAP:2 NEAREST:3
        if fill_mode == "CONSTANT":
            self.fill_mode = 0
        elif fill_mode == "REFLECT":
            self.fill_mode = 1
        elif fill_mode == "WRAP":
            self.fill_mode = 2
        elif fill_mode == "NEAREST":
            self.fill_mode = 3
        self.input_x_float = self.tik_instance.Scalar("float32", name="input_x_float")
        self.input_x_int = self.tik_instance.Scalar("int32", name="input_x_int")
        self.input_y_float = self.tik_instance.Scalar("float32", name="input_y_float")
        self.input_y_int = self.tik_instance.Scalar("int32", name="input_y_int")
        self.img_num = self.tik_instance.Scalar("int32", name="img_num")
        self.trans_a0 = self.tik_instance.Scalar("float32", name="trans_a0")
        self.trans_a1 = self.tik_instance.Scalar("float32", name="trans_a1")
        self.trans_a2 = self.tik_instance.Scalar("float32", name="trans_a2")
        self.trans_b0 = self.tik_instance.Scalar("float32", name="trans_b0")
        self.trans_b1 = self.tik_instance.Scalar("float32", name="trans_b1")
        self.trans_b2 = self.tik_instance.Scalar("float32", name="trans_b2")
        self.trans_c0 = self.tik_instance.Scalar("float32", name="trans_c0")
        self.trans_c1 = self.tik_instance.Scalar("float32", name="trans_c1")
        self.flag_xyfloor = self.tik_instance.Scalar("int32", name="flag_xyfloor")
        self.ub_c_repeat = self.tik_instance.Scalar("int32", name="ub_c_repeat")
        self.ub_c_left = self.tik_instance.Scalar("int32", name="ub_c_left")
        self.ub_c_left_num = self.tik_instance.Scalar("int32", name="ub_c_left_num")
        self.c_repeat = self.tik_instance.Scalar("int32", name="c_repeat")
        self.mask_size = self.tik_instance.Scalar("int32", name="mask_size")
        self.block_size = self.tik_instance.Scalar("int32", name="block_size")
        self.cal_repeat = self.tik_instance.Scalar("int32", name="cal_repeat")
        self.fill_val = self.tik_instance.Scalar(self.dtype, name="fill_val", init_value=0)
        self.h_start = self.tik_instance.Scalar("int32", name="h_start")
        self.h_end = self.tik_instance.Scalar("int32", name="h_end")
        self.offset_i = self.tik_instance.Scalar("int32", name="offset_i")
        self.offset_ini = self.tik_instance.Scalar("int32", name="offset_ini")
        self.offset_img = self.tik_instance.Scalar("int32", name="offset_img")
        self.offset_ni = self.tik_instance.Scalar("int32", name="offset_ni")
        self.ub_h = self.tik_instance.Scalar("int32", name="ub_h")
        self.burst_long_map = self.tik_instance.Scalar("int32", name="burst_long_map")

    def img_compute(self):
        """
        excute V2 operator
        """
        self._init_gm_tensor()
        inputs = [self.input_gm, self.transform_gm, self.output_shape_gm]
        if self.exist_fill_value_n is True:
            self.fill_value_gm = self.tik_instance.Tensor(self.dtype,
                                                          (8,),
                                                          name="fill_value_gm",
                                                          scope=tik.scope_gm)
            inputs.append(self.fill_value_gm)

        self._run()

        opt_config = {"enable_const_fold": True}
        tbe_context.get_context().add_compile_info(
            "vars", {
                "ub_ele": Constant.MAX_DATA_UB,
                "core_num": self.aicore_num
            })

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=inputs,
                                   outputs=[self.output_gm],
                                   flowtable=[self.tiling_gm],
                                   config=opt_config)

    def image_projective_compute(self):
        """
        excute operator
        """
        self._init_gm_tensor()
        self._run()
        opt_config = {"enable_const_fold": True}

        tbe_context.get_context().add_compile_info(
            "vars", {
                "ub_ele": Constant.MAX_DATA_UB,
                "core_num": self.aicore_num
            })

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_gm, self.transform_gm, self.output_shape_gm],
                                   outputs=[self.output_gm],
                                   flowtable=[self.tiling_gm],
                                   config=opt_config)

    def _run(self):
        """
        execute_tilling, copy tiling and read
        """
        self._set_tiling_param()
        with self.tik_instance.for_range(0, self.new_core_num, block_num=self.new_core_num) as core_idx:
            self.block_size.set_as(Constant.BLC_BYTE_SIZE // self.dtype_size)
            with self.tik_instance.new_stmt_scope():
                with self.tik_instance.if_scope(self.exist_fill_value_n is True):
                    self.ub_fill = self.tik_instance.Tensor(self.dtype,
                                                            (8,),
                                                            name="ub_fill",
                                                            scope=tik.scope_ubuf)
                    self.tik_instance.data_move(self.ub_fill, self.fill_value_gm, 0, 1, 1, 0, 0)
                    self.fill_val.set_as(self.ub_fill[0])

            with self.tik_instance.if_scope(core_idx < self.act_core_num):
                # images_b more than core num and transform_b is equal to 1
                with self.tik_instance.if_scope(self.tiling_mode == 0):
                    with self.tik_instance.new_stmt_scope():
                        with self.tik_instance.if_scope(core_idx < self.imgnum_cal_left):
                            self.img_num.set_as(self.imgnum_cal_ceil_repeat)
                        with self.tik_instance.else_scope():
                            self.img_num.set_as(self.imgnum_cal_repeat)
                        self._move_one_transform()
                        self._gt_ub_copy(core_idx)

                # images_b more than core num and transform_b is equal to images_b
                with self.tik_instance.if_scope(self.tiling_mode == 1):
                    with self.tik_instance.new_stmt_scope():
                        with self.tik_instance.if_scope(core_idx < self.imgnum_cal_left):
                            self.img_num.set_as(self.imgnum_cal_ceil_repeat)
                        with self.tik_instance.else_scope():
                            self.img_num.set_as(self.imgnum_cal_repeat)
                        self._gt_ub_copy_transforms(core_idx)

                # images_b more than core num and transform_b is equal to one
                with self.tik_instance.if_scope(self.tiling_mode == 2):
                    with self.tik_instance.new_stmt_scope():
                        self._move_one_transform()
                        self._one_block_for_one_transform()

                # images_b more than core num and transform_b is equal to images_b
                with self.tik_instance.if_scope(self.tiling_mode == 3):
                    with self.tik_instance.new_stmt_scope():
                        self._one_block_for_transforms()

    def _init_gm_tensor(self):
        """
        Init gm tensor
        """
        self.tiling_gm = self.tik_instance.Tensor("int32", (Constant.TILING_ARG_NUM,),
                                                  name="tiling_gm",
                                                  scope=tik.scope_gm)
        self.input_gm = self.tik_instance.Tensor(self.dtype, (constant_util.SHAPE_SIZE_LIMIT,),
                                                 name="input_gm",
                                                 scope=tik.scope_gm)
        self.transform_gm = self.tik_instance.Tensor(self.trans_dtype, (constant_util.SHAPE_SIZE_LIMIT,),
                                                     name="transform_gm",
                                                     scope=tik.scope_gm)
        self.output_gm = self.tik_instance.Tensor(self.dtype, (constant_util.SHAPE_SIZE_LIMIT,),
                                                  name="output_gm",
                                                  scope=tik.scope_gm)
        self.output_shape_gm = self.tik_instance.Tensor("int32", (2,),
                                                        name="output_shape_gm",
                                                        scope=tik.scope_gm)

    def _init_xy_ub_tensor(self, ub_size):
        """
        Init x y coords tensor
        """
        self.ub_input_x = self.tik_instance.Tensor("float32", (ub_size,), name="ub_input_x", scope=tik.scope_ubuf)
        self.ub_input_y = self.tik_instance.Tensor("float32", (ub_size,), name="ub_input_y", scope=tik.scope_ubuf)

    def _init_ub_tensor(self, ub_size, cal_ubsize):
        """
        Init ub tensor
        """
        self.ub_images = self.tik_instance.Tensor(self.dtype, (cal_ubsize,),
                                                  name="ub_images", scope=tik.scope_ubuf)
        self.ub_bili_image = self.tik_instance.Tensor("float32", (cal_ubsize,),
                                                      name="ub_images",
                                                      scope=tik.scope_ubuf)
        self.ub_output = self.tik_instance.Tensor(self.dtype, (ub_size,), name="ub_output", scope=tik.scope_ubuf)
        self.ub_aligned = self.tik_instance.Tensor(self.dtype, (32,), name="ub_aligned", scope=tik.scope_ubuf)
        self.mv_ub_value_out_cast = self.tik_instance.Tensor(self.dtype, (cal_ubsize,),
                                                             name="mv_ub_value_out_cast",
                                                             scope=tik.scope_ubuf)
        self.mv_ub_value_origin_cast = self.tik_instance.Tensor(self.tmp_dtype, (cal_ubsize,),
                                                                name="mv_ub_value_origin_cast",
                                                                scope=tik.scope_ubuf)
        self.ub_value_gm_cast = self.tik_instance.Tensor(self.tmp_dtype, (cal_ubsize,), name="ub_value_gm_cast",
                                                         scope=tik.scope_ubuf)
        self.ub_value_origin_h = self.tik_instance.Tensor(self.dtype, (cal_ubsize,), name="ub_value_origin_h",
                                                          scope=tik.scope_ubuf)
        self.ub_value_out_cast = self.tik_instance.Tensor(self.dtype, (cal_ubsize,), name="ub_value_out_cast",
                                                          scope=tik.scope_ubuf)
        self.ub_value_origin_cast = self.tik_instance.Tensor(self.tmp_dtype, (cal_ubsize,), name="ub_value_origin_cast",
                                                             scope=tik.scope_ubuf)

    def _init_block_ub_tensor(self, ub_size, cal_ubsize):
        """
        Init ub tensor
        """
        self.ub_images = self.tik_instance.Tensor(self.dtype, (cal_ubsize,), name="ub_images", scope=tik.scope_ubuf)
        self.ub_bili_image = self.tik_instance.Tensor("float32", (cal_ubsize,), name="ub_images", scope=tik.scope_ubuf)
        self.ub_output = self.tik_instance.Tensor(self.dtype, (ub_size,), name="ub_output", scope=tik.scope_ubuf)
        self.ub_aligned = self.tik_instance.Tensor(self.dtype, (32,), name="ub_aligned", scope=tik.scope_ubuf)
        self.ub_value_gm_cast = self.tik_instance.Tensor(self.tmp_dtype, (cal_ubsize,), name="ub_value_gm_cast",
                                                         scope=tik.scope_ubuf)
        self.ub_value_origin_h = self.tik_instance.Tensor(self.dtype, (cal_ubsize,), name="ub_value_origin_h",
                                                          scope=tik.scope_ubuf)
        self.ub_value_out_cast = self.tik_instance.Tensor(self.dtype, (cal_ubsize,), name="ub_value_out_cast",
                                                          scope=tik.scope_ubuf)
        self.ub_value_origin_cast = self.tik_instance.Tensor(self.tmp_dtype, (cal_ubsize,), name="ub_value_origin_cast",
                                                             scope=tik.scope_ubuf)

    def _move_tiling_to_ub(self):
        """
        set tiling numger to ub
        """
        with self.tik_instance.new_stmt_scope():
            tiling_ub = self.tik_instance.Tensor("int32", (Constant.TILING_ARG_NUM,),
                                                 name="tiling_ub",
                                                 scope=tik.scope_ubuf)
            self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, ceil_div(Constant.TILING_ARG_NUM, 8), 0, 0)

            self.tiling_mode.set_as(tiling_ub[0])
            self.act_core_num.set_as(tiling_ub[1])
            self.input_b.set_as(tiling_ub[2])
            self.input_h.set_as(tiling_ub[3])
            self.input_w.set_as(tiling_ub[4])
            self.input_c.set_as(tiling_ub[5])
            self.input_size.set_as(tiling_ub[6])
            self.output_h.set_as(tiling_ub[7])
            self.output_w.set_as(tiling_ub[8])
            self.ub_height.set_as(tiling_ub[9])
            self.ub_repeat_time.set_as(tiling_ub[10])
            self.ub_repeat_left.set_as(tiling_ub[11])
            self.ub_floor_repeat.set_as(tiling_ub[12])
            self.imgnum_cal_repeat.set_as(tiling_ub[13])
            self.imgnum_cal_ceil_repeat.set_as(tiling_ub[14])
            self.imgnum_cal_left.set_as(tiling_ub[15])
            self.burst_long_bili.set_as(tiling_ub[16])
            self.calc_ubsize.set_as(tiling_ub[17])
            self.bc_repeat.set_as(tiling_ub[18])
            self.c_mask.set_as(tiling_ub[19])
            self.new_core_num.set_as(tiling_ub[20])

    def _set_tiling_param(self):
        """
        _set_tiling_param
        """
        self.tiling_mode = self.tik_instance.Scalar("int32", name="tiling_mode")
        self.act_core_num = self.tik_instance.Scalar("int32", name="act_core_num")
        self.input_b = self.tik_instance.Scalar("int32", name="input_b")
        self.input_h = self.tik_instance.Scalar("int32", name="input_h")
        self.input_w = self.tik_instance.Scalar("int32", name="input_w")
        self.input_c = self.tik_instance.Scalar("int32", name="input_c")
        self.output_h = self.tik_instance.Scalar("int32", name="output_h")
        self.output_w = self.tik_instance.Scalar("int32", name="output_w")
        self.ub_height = self.tik_instance.Scalar("int32", name="ub_height")
        self.ub_repeat_time = self.tik_instance.Scalar("int32", name="ub_repeat_time")
        self.ub_repeat_left = self.tik_instance.Scalar("int32", name="ub_repeat_left")
        self.ub_floor_repeat = self.tik_instance.Scalar("int32", name="ub_floor_repeat")
        self.input_size = self.tik_instance.Scalar("int32", name="input_size")
        self.imgnum_cal_repeat = self.tik_instance.Scalar("int32", name="imgnum_cal_repeat")
        self.imgnum_cal_ceil_repeat = self.tik_instance.Scalar("int32", name="imgnum_cal_ceil_repeat")
        self.imgnum_cal_left = self.tik_instance.Scalar("int32", name="imgnum_cal_left")
        self.burst_long_bili = self.tik_instance.Scalar("int32", name="burst_long_bili")
        self.calc_ubsize = self.tik_instance.Scalar("int32", name="calc_ubsize")
        self.bc_repeat = self.tik_instance.Scalar("int32", name="bc_repeat")
        self.c_mask = self.tik_instance.Scalar("int32", name="c_mask")
        self.new_core_num = self.tik_instance.Scalar("int32", name="new_core_num")

        self._move_tiling_to_ub()

    def _map_coord(self, ub_size, m_flag):
        """
        do fill mode select
        """
        if m_flag == 0:
            self.burst_long_map.set_as((self.output_w * self.ub_h + Constant.SEVEN_BIT) // Constant.EIGHT_BIT)
        else:
            self.burst_long_map.set_as(self.input_b)

        if self.fill_mode == 1:
            self._map_coord_reflect(self.ub_input_x, self.input_w, ub_size)
            self._map_coord_reflect(self.ub_input_y, self.input_h, ub_size)
        elif self.fill_mode == 2:
            self._map_coord_wrap(self.ub_input_y, self.input_h, ub_size)
            self._map_coord_wrap(self.ub_input_x, self.input_w, ub_size)
        elif self.fill_mode == 3:
            self._map_coord_nearest(self.ub_input_x, self.input_w, ub_size)
            self._map_coord_nearest(self.ub_input_y, self.input_h, ub_size)

    def _cal_coords(self, ub_size, rep):
        """
        create two tensor by x and y loop
        calculate transform data with output x and y coords to get input x and y coords
        """
        with self.tik_instance.new_stmt_scope():
            # init some temporary tensor for cal coords
            ub_output_x = self.tik_instance.Tensor("float32", (ub_size,), name="ub_output_x", scope=tik.scope_ubuf)
            ub_output_y = self.tik_instance.Tensor("float32", (ub_size,), name="ub_output_y", scope=tik.scope_ubuf)
            projection_ub = self.tik_instance.Tensor("float32", (ub_size,), name="projection_ub", scope=tik.scope_ubuf)
            ub_tmp = self.tik_instance.Tensor("float32", (ub_size,), name="ub_tmp", scope=tik.scope_ubuf)
            ub_tmp_y = self.tik_instance.Tensor("float32", (ub_size,), name="ub_tmp_y", scope=tik.scope_ubuf)
            rep_idx = self.tik_instance.Scalar("int32", name="rep_idx")
            offset_o = self.tik_instance.Scalar("int32", name="offset_o")

            # ub could store h*w ele
            with self.tik_instance.if_scope(rep == -1):
                rep_idx.set_as(0)
            # h*w more than ub
            with self.tik_instance.else_scope():
                rep_idx.set_as(rep)

            # a tensor with output y coords from zero to height
            with self.tik_instance.for_range(self.h_start, self.h_end) as loop_h_idx:
                # a tensor with output x coords from zero to width
                with self.tik_instance.for_range(0, self.output_w) as loop_w_idx:
                    offset_o.set_as((loop_h_idx - rep_idx * self.ub_height) * self.output_w + loop_w_idx)
                    ub_output_y[offset_o].set_as(loop_h_idx * 1.0)
                    ub_output_x[offset_o].set_as(loop_w_idx * 1.0)

            with self.tik_instance.if_scope(self.ub_floor_repeat == 0):
                self.cal_repeat.set_as((self.output_h * self.output_w + Constant.CEIL_NUM) // Constant.FP32_NUM)
            with self.tik_instance.else_scope():
                self.cal_repeat.set_as((self.ub_height * self.output_w + Constant.CEIL_NUM) // Constant.FP32_NUM)

            # cal projection `c0 * output_x + c1 * output_y + 1.f`
            self.tik_instance.vec_muls(Constant.FP32_NUM, self.ub_input_x, ub_output_x, self.trans_c0, self.cal_repeat,
                                       8, 8)
            self.tik_instance.vec_muls(Constant.FP32_NUM, self.ub_input_y, ub_output_y, self.trans_c1, self.cal_repeat,
                                       8, 8)
            self.tik_instance.vec_add(Constant.FP32_NUM, self.ub_input_x, self.ub_input_x, self.ub_input_y,
                                      self.cal_repeat, 8, 8, 8)
            self.tik_instance.vec_adds(Constant.FP32_NUM, projection_ub, self.ub_input_x, Constant.FLOAT_ONE,
                                       self.cal_repeat, 8, 8)

            # cal input_x `(a0 * output_x + a1 * output_y + a2) / projection`
            self.tik_instance.vec_muls(Constant.FP32_NUM, ub_tmp, ub_output_x, self.trans_a0, self.cal_repeat, 8, 8)
            self.tik_instance.vec_muls(Constant.FP32_NUM, self.ub_input_y, ub_output_y, self.trans_a1, self.cal_repeat,
                                       8, 8)
            self.tik_instance.vec_add(Constant.FP32_NUM, self.ub_input_x, ub_tmp, self.ub_input_y, self.cal_repeat, 8,
                                      8, 8)
            self.tik_instance.vec_adds(Constant.FP32_NUM, ub_tmp, self.ub_input_x, self.trans_a2, self.cal_repeat, 8, 8)
            self.tik_instance.vdiv(Constant.FP32_NUM, self.ub_input_x, ub_tmp, projection_ub, self.cal_repeat, 1, 1, 1,
                                   8, 8, 8)

            # cal input_y `(b0 * output_x + b1 * output_y + b3) / projection`
            self.tik_instance.vec_muls(Constant.FP32_NUM, self.ub_input_y, ub_output_y, self.trans_b1, self.cal_repeat,
                                       8, 8)
            self.tik_instance.vec_muls(Constant.FP32_NUM, ub_tmp_y, ub_output_x, self.trans_b0, self.cal_repeat, 8, 8)
            self.tik_instance.vec_add(Constant.FP32_NUM, self.ub_input_y, ub_tmp_y, self.ub_input_y, self.cal_repeat, 8,
                                      8, 8)
            self.tik_instance.vec_adds(Constant.FP32_NUM, ub_tmp_y, self.ub_input_y, self.trans_b2, self.cal_repeat, 8,
                                       8)
            self.tik_instance.vdiv(Constant.FP32_NUM, self.ub_input_y, ub_tmp_y, projection_ub, self.cal_repeat, 1, 1,
                                   1, 8, 8, 8)

    def _cal_coords_one_block(self):
        """
        create two tensor by x and y loop
        calculate transform data with output x and y coords to get input x and y coords
        """
        with self.tik_instance.new_stmt_scope():
            # init some temporary tensor for cal coords
            ub_output_x_b = self.tik_instance.Tensor("float32", (64,), name="ub_output_x_b", scope=tik.scope_ubuf)
            ub_output_y_b = self.tik_instance.Tensor("float32", (64,), name="ub_output_y_b", scope=tik.scope_ubuf)
            projection_ub_b = self.tik_instance.Tensor("float32", (64,), name="projection_ub_b", scope=tik.scope_ubuf)
            ub_input_y_tmp_b = self.tik_instance.Tensor("float32", (64,), name="ub_input_y_tmp_b", scope=tik.scope_ubuf)
            ub_input_x_tmp_b = self.tik_instance.Tensor("float32", (64,), name="ub_input_x_tmp_b", scope=tik.scope_ubuf)
            ub_tmp_b = self.tik_instance.Tensor("float32", (64,), name="ub_tmp_b", scope=tik.scope_ubuf)
            ub_tmp_y_b = self.tik_instance.Tensor("float32", (64,), name="ub_tmp_y_b", scope=tik.scope_ubuf)

            # a tensor with output y coords from zero to height
            with self.tik_instance.for_range(0, self.output_h) as loop_h_idx:
                # a tensor with output x coords from zero to width
                with self.tik_instance.for_range(0, self.output_w) as loop_w_idx:
                    offset_o = loop_h_idx * self.output_w + loop_w_idx
                    ub_output_y_b[offset_o].set_as(loop_h_idx * 1.0)
                    ub_output_x_b[offset_o].set_as(loop_w_idx * 1.0)

            self.cal_repeat.set_as((self.input_b * Constant.EIGHT_BIT + Constant.CEIL_NUM) // Constant.FP32_NUM)

            with self.tik_instance.for_range(0, self.input_b) as img_idx:
                self._move_transforms(img_idx)
                offset_s = img_idx * Constant.EIGHT_BIT
                # cal projection `c0 * output_x + c1 * output_y + 1.f`
                self.tik_instance.vec_muls(Constant.EIGHT_BIT, self.ub_input_x[offset_s], ub_output_x_b, self.trans_c0,
                                           1, 8, 8)
                self.tik_instance.vec_muls(Constant.EIGHT_BIT, self.ub_input_y[offset_s], ub_output_y_b, self.trans_c1,
                                           1, 8, 8)
                self.tik_instance.vec_add(Constant.EIGHT_BIT, ub_input_x_tmp_b, self.ub_input_x[offset_s],
                                          self.ub_input_y[offset_s], 1, 8, 8, 8)
                self.tik_instance.vec_adds(Constant.EIGHT_BIT, projection_ub_b, ub_input_x_tmp_b, Constant.FLOAT_ONE, 1,
                                           8, 8)

                # cal input_x `(a0 * output_x + a1 * output_y + a2) / projection`
                self.tik_instance.vec_muls(Constant.EIGHT_BIT, ub_tmp_b, ub_output_x_b, self.trans_a0, 1, 8, 8)
                self.tik_instance.vec_muls(Constant.EIGHT_BIT, self.ub_input_y[offset_s], ub_output_y_b, self.trans_a1,
                                           1, 8, 8)
                self.tik_instance.vec_add(Constant.EIGHT_BIT, self.ub_input_x[offset_s], ub_tmp_b,
                                          self.ub_input_y[offset_s], 1, 8, 8, 8)
                self.tik_instance.vec_adds(Constant.EIGHT_BIT, ub_tmp_b, self.ub_input_x[offset_s], self.trans_a2,
                                           1, 8, 8)
                self.tik_instance.vdiv(Constant.EIGHT_BIT, self.ub_input_x[offset_s], ub_tmp_b, projection_ub_b,
                                       1, 1, 1, 1, 8, 8, 8)

                # cal input_y `(b0 * output_x + b1 * output_y + b3) / projection`
                self.tik_instance.vec_muls(Constant.EIGHT_BIT, self.ub_input_y[offset_s], ub_output_y_b, self.trans_b1,
                                           1, 8, 8)
                self.tik_instance.vec_muls(Constant.EIGHT_BIT, ub_tmp_y_b, ub_output_x_b, self.trans_b0, 1, 8, 8)
                self.tik_instance.vec_add(Constant.EIGHT_BIT, ub_input_y_tmp_b,
                                          self.ub_input_y[offset_s], ub_tmp_y_b, 1, 8, 8, 8)
                self.tik_instance.vec_adds(Constant.EIGHT_BIT, ub_tmp_y_b, ub_input_y_tmp_b, self.trans_b2, 1, 8, 8)
                self.tik_instance.vdiv(Constant.EIGHT_BIT, self.ub_input_y[offset_s], ub_tmp_y_b, projection_ub_b,
                                       1, 1, 1, 1, 8, 8, 8)

    def _map_coord_reflect(self, ub_input, len_hw, ub_size):
        """
        cal the map coord for reflect fill mode
        """
        with self.tik_instance.new_stmt_scope():
            r_ub_sz2 = self.tik_instance.Tensor("float32", (ub_size,), name="r_ub_sz2", scope=tik.scope_ubuf)
            r_ub_one = self.tik_instance.Tensor("float32", (ub_size,), name="r_ub_one", scope=tik.scope_ubuf)
            r_ub_neg_input = self.tik_instance.Tensor("float32", (ub_size,), name="r_ub_neg_input",
                                                      scope=tik.scope_ubuf)
            r_ub_input_div_sz2 = self.tik_instance.Tensor("float32", (ub_size,), name="r_ub_input_div_sz2",
                                                          scope=tik.scope_ubuf)
            r_ub_div_mul_sz2 = self.tik_instance.Tensor("float32", (ub_size,), name="r_ub_div_mul_sz2",
                                                        scope=tik.scope_ubuf)
            r_ub_input_add_one = self.tik_instance.Tensor("float32", (ub_size,), name="r_ub_input_add_one",
                                                          scope=tik.scope_ubuf)
            r_ub_div_one = self.tik_instance.Tensor("float32", (ub_size,), name="r_ub_div_one", scope=tik.scope_ubuf)
            r_ub_h_cast = self.tik_instance.Tensor("float32", (ub_size,), name="r_ub_h_cast", scope=tik.scope_ubuf)
            r_ub_dup_one = self.tik_instance.Tensor("float32", (ub_size,), name="r_ub_dup_one", scope=tik.scope_ubuf)
            r_ub_cmp = self.tik_instance.Tensor("float32", (64,), name="r_ub_cmp", scope=tik.scope_ubuf)
            r_ub_int = self.tik_instance.Tensor("int32", (ub_size,), name="r_ub_int", scope=tik.scope_ubuf)
            r_ub_tmp = self.tik_instance.Tensor("float32", (ub_size,), name="r_ub_tmp", scope=tik.scope_ubuf)
            r_ub_sel = self.tik_instance.Tensor("uint16", (8,), name="r_ub_sel", scope=tik.scope_ubuf)
            r_ub_compare = self.tik_instance.Tensor("float32", (64,), name="r_ub_compare", scope=tik.scope_ubuf)
            r_len_sub_one = self.tik_instance.Scalar("float32", name="r_len_sub_one")
            r_neg_len = self.tik_instance.Scalar("float32", name="r_neg_len")
            r_len_float = self.tik_instance.Scalar("float32", name="r_len_float")
            sz2_r = self.tik_instance.Scalar("float32", name="sz2_r")
            r_len_sub_one.set_as(len_hw - 1)
            r_neg_len.set_as(len_hw * -1)
            r_len_float.set_as(len_hw)
            sz2_r.set_as(len_hw * 2)

            self.tik_instance.data_move(r_ub_tmp, ub_input, 0, 1, self.burst_long_map, 0, 0)
            # select case len <= 1
            with self.tik_instance.if_scope(len_hw <= 1):
                self.tik_instance.vec_dup(Constant.FP32_NUM, ub_input, 0, self.cal_repeat, 8)

            with self.tik_instance.else_scope():
                # calculate `sz2 * (int)(-coord / sz2) + coord`
                self.tik_instance.vec_dup(Constant.FP32_NUM, r_ub_sz2, sz2_r, self.cal_repeat, 8)
                self.tik_instance.vec_muls(Constant.FP32_NUM, r_ub_neg_input, r_ub_tmp, -1, self.cal_repeat, 8, 8)
                self.tik_instance.vdiv(Constant.FP32_NUM, r_ub_input_div_sz2, r_ub_neg_input, r_ub_sz2, self.cal_repeat,
                                       1, 1, 1, 8, 8, 8)
                self.tik_instance.h_cast(r_ub_int, r_ub_input_div_sz2, "to-zero")
                self.tik_instance.h_cast(r_ub_div_mul_sz2, r_ub_int, "none")
                self.tik_instance.vec_mul(Constant.FP32_NUM, r_ub_input_div_sz2, r_ub_div_mul_sz2, r_ub_sz2,
                                          self.cal_repeat, 8, 8, 8)
                self.tik_instance.vec_add(Constant.FP32_NUM, r_ub_input_div_sz2, r_ub_input_div_sz2, r_ub_tmp,
                                          self.cal_repeat, 8, 8, 8)

                # case2 ub_neg_input in_coord + sz2(with)
                self.tik_instance.vec_add(Constant.FP32_NUM, r_ub_neg_input, r_ub_input_div_sz2, r_ub_sz2,
                                          self.cal_repeat, 8, 8, 8)

                # case3 r_ub_div_mul_sz2 -in_coord - 1(with)
                self.tik_instance.vec_muls(Constant.FP32_NUM, r_ub_sz2, r_ub_input_div_sz2, -1, self.cal_repeat, 8, 8)
                self.tik_instance.vec_dup(Constant.FP32_NUM, r_ub_one, 1, self.cal_repeat, 8)
                self.tik_instance.vec_sub(Constant.FP32_NUM, r_ub_div_mul_sz2, r_ub_sz2, r_ub_one, self.cal_repeat, 8,
                                          8, 8)

                # case 2 r_ub_sz2 -in_coord - 1(no)
                self.tik_instance.vec_muls(Constant.FP32_NUM, r_ub_sz2, r_ub_tmp, -1, self.cal_repeat, 8, 8)
                self.tik_instance.vec_sub(Constant.FP32_NUM, r_ub_sz2, r_ub_sz2, r_ub_one, self.cal_repeat, 8, 8, 8)

                # case 1 r_ub_input_add_one in_coord + sz2(no)
                self.tik_instance.vec_dup(Constant.FP32_NUM, r_ub_one, sz2_r, self.cal_repeat, 8)
                self.tik_instance.vec_add(Constant.FP32_NUM, r_ub_input_add_one, r_ub_one, r_ub_tmp, self.cal_repeat, 8,
                                          8, 8)

                # calculate `coord - sz2 * (int)(coord / sz2)`
                self.tik_instance.vdiv(Constant.FP32_NUM, r_ub_div_one, r_ub_tmp, r_ub_one, self.cal_repeat, 1, 1, 1, 8,
                                       8, 8)
                self.tik_instance.h_cast(r_ub_int, r_ub_div_one, "to-zero")
                self.tik_instance.h_cast(r_ub_h_cast, r_ub_int, "none")
                self.tik_instance.vec_mul(Constant.FP32_NUM, r_ub_h_cast, r_ub_h_cast, r_ub_one, self.cal_repeat, 8, 8,
                                          8)
                self.tik_instance.vec_sub(Constant.FP32_NUM, r_ub_div_one, r_ub_tmp, r_ub_h_cast, self.cal_repeat, 8, 8,
                                          8)

                # case 6 r_ub_h_cast (sz2 - coord - 1)
                self.tik_instance.vec_sub(Constant.FP32_NUM, r_ub_h_cast, r_ub_one, r_ub_div_one, self.cal_repeat, 8, 8,
                                          8)
                self.tik_instance.vec_dup(Constant.FP32_NUM, r_ub_dup_one, 1, self.cal_repeat, 8)
                self.tik_instance.vec_sub(Constant.FP32_NUM, r_ub_h_cast, r_ub_h_cast, r_ub_dup_one, self.cal_repeat, 8,
                                          8, 8)

                with self.tik_instance.for_range(0, self.cal_repeat) as re_idx:
                    # select in_coord < sz2? r_ub_input_add_one
                    self.tik_instance.vec_dup(Constant.FP32_NUM, r_ub_compare, r_neg_len, 1, 8)
                    self.tik_instance.vec_cmpv_lt(r_ub_sel, r_ub_input_div_sz2[re_idx * Constant.FP32_NUM],
                                                  r_ub_compare, 1, 8, 8)
                    self.tik_instance.vec_sel(Constant.FP32_NUM, 0, r_ub_cmp, r_ub_sel,
                                              r_ub_neg_input[re_idx * Constant.FP32_NUM],
                                              r_ub_div_mul_sz2[re_idx * Constant.FP32_NUM], 1, 8, 8, 8)

                    self.tik_instance.vec_cmpv_lt(r_ub_sel, r_ub_tmp[re_idx * Constant.FP32_NUM], r_ub_compare, 1, 8, 8)
                    self.tik_instance.vec_sel(Constant.FP32_NUM, 0, r_ub_neg_input[re_idx * Constant.FP32_NUM],
                                              r_ub_sel, r_ub_input_add_one[re_idx * Constant.FP32_NUM],
                                              r_ub_sz2[re_idx * Constant.FP32_NUM], 1, 8, 8, 8)

                    self.tik_instance.vec_cmpv_lt(r_ub_sel, r_ub_tmp[re_idx * Constant.FP32_NUM],
                                                  r_ub_one[re_idx * Constant.FP32_NUM], 1, 8, 8)
                    self.tik_instance.vec_sel(Constant.FP32_NUM, 0, r_ub_input_add_one[re_idx * Constant.FP32_NUM],
                                              r_ub_sel, r_ub_cmp, r_ub_neg_input[re_idx * Constant.FP32_NUM], 1, 8, 8,
                                              8)

                    # select in_coord >= len ? r_ub_div_mul_sz2
                    self.tik_instance.vec_dup(Constant.FP32_NUM, r_ub_compare, r_len_float, 1, 8)
                    self.tik_instance.vec_cmpv_ge(r_ub_sel, r_ub_div_one[re_idx * Constant.FP32_NUM], r_ub_compare, 1,
                                                  8, 8)
                    self.tik_instance.vec_sel(Constant.FP32_NUM, 0, r_ub_div_mul_sz2[re_idx * Constant.FP32_NUM],
                                              r_ub_sel, r_ub_h_cast[re_idx * Constant.FP32_NUM],
                                              r_ub_div_one[re_idx * Constant.FP32_NUM], 1, 8, 8, 8)

                    # select r_ub_cmp
                    self.tik_instance.vec_dup(Constant.FP32_NUM, r_ub_compare, r_len_sub_one, 1, 8)
                    self.tik_instance.vec_cmpv_gt(r_ub_sel, r_ub_tmp[re_idx * Constant.FP32_NUM], r_ub_compare, 1, 8, 8)
                    self.tik_instance.vec_sel(Constant.FP32_NUM, 0, r_ub_cmp, r_ub_sel,
                                              r_ub_div_mul_sz2[re_idx * Constant.FP32_NUM],
                                              r_ub_tmp[re_idx * Constant.FP32_NUM], 1, 8, 8, 8)

                    # select ub_neg_input
                    self.tik_instance.vec_dup(Constant.FP32_NUM, r_ub_compare, 0, 1, 8)
                    self.tik_instance.vec_cmpv_lt(r_ub_sel, r_ub_tmp[re_idx * Constant.FP32_NUM], r_ub_compare, 1, 8, 8)
                    self.tik_instance.vec_sel(Constant.FP32_NUM, 0, r_ub_neg_input[re_idx * Constant.FP32_NUM],
                                              r_ub_sel, r_ub_input_add_one[re_idx * Constant.FP32_NUM], r_ub_cmp, 1, 8,
                                              8, 8)

                # compare 0 with ub_neg_input to select max result r_ub_sz2
                self.tik_instance.vec_dup(Constant.FP32_NUM, r_ub_one, 0, self.cal_repeat, 8)
                self.tik_instance.vmax(Constant.FP32_NUM, r_ub_sz2, r_ub_one, r_ub_neg_input, self.cal_repeat, 1, 1, 1,
                                       8, 8, 8)

                # compare len-1 with ub_neg_input to select min result r_ub_input_div_sz2
                self.tik_instance.vec_dup(Constant.FP32_NUM, r_ub_one, r_len_sub_one, self.cal_repeat, 8)
                self.tik_instance.vmin(Constant.FP32_NUM, r_ub_input_div_sz2, r_ub_one, r_ub_sz2, self.cal_repeat, 1, 1,
                                       1, 8, 8, 8)

                # move this part result to ub_input
                self.tik_instance.data_move(ub_input, r_ub_input_div_sz2, 0, 1, self.burst_long_map, 0, 0)

    def _map_coord_wrap(self, ub_input, len_hw, ub_size):
        """
        cal the map coord for wrap fill mode
        """
        with self.tik_instance.new_stmt_scope():
            w_ub_sz2 = self.tik_instance.Tensor("float32", (ub_size,), name="w_ub_sz2", scope=tik.scope_ubuf)
            w_ub_one = self.tik_instance.Tensor("float32", (ub_size,), name="w_ub_one", scope=tik.scope_ubuf)
            w_ub_neg_input = self.tik_instance.Tensor("float32", (ub_size,), name="w_ub_neg_input",
                                                      scope=tik.scope_ubuf)
            w_ub_input_div_sz2 = self.tik_instance.Tensor("float32", (ub_size,), name="w_ub_input_div_sz2",
                                                          scope=tik.scope_ubuf)
            w_ub_div_mul_sz2 = self.tik_instance.Tensor("float32", (ub_size,), name="w_ub_div_mul_sz2",
                                                        scope=tik.scope_ubuf)
            w_ub_int = self.tik_instance.Tensor("int32", (ub_size,), name="w_ub_int", scope=tik.scope_ubuf)
            w_ub_tmp = self.tik_instance.Tensor("float32", (ub_size,), name="w_ub_tmp", scope=tik.scope_ubuf)
            w_ub_sel = self.tik_instance.Tensor("uint16", (8,), name="w_ub_sel", scope=tik.scope_ubuf)
            w_ub_compare = self.tik_instance.Tensor("float32", (64,), name="w_ub_compare", scope=tik.scope_ubuf)
            w_len_sw_ub_one = self.tik_instance.Scalar("float32", name="w_len_sw_ub_one")
            w_len_float = self.tik_instance.Scalar("float32", name="w_len_float")
            sz2_w = self.tik_instance.Scalar("float32", name="sz2_w")
            w_len_sw_ub_one.set_as(len_hw - 1)
            w_len_float.set_as(len_hw)
            sz2_w.set_as(len_hw - 1)

            self.tik_instance.data_move(w_ub_tmp, ub_input, 0, 1, self.burst_long_map, 0, 0)
            # select case len <= 1
            with self.tik_instance.if_scope(w_len_float <= 1):
                self.tik_instance.vec_dup(Constant.FP32_NUM, ub_input, 0, self.cal_repeat, 8)

            with self.tik_instance.else_scope():
                # calculate `in_coord + len  * (int)(-coord / sz2) + 1`
                # w_ub_neg_input
                self.tik_instance.vec_dup(Constant.FP32_NUM, w_ub_one, sz2_w, self.cal_repeat, 8)
                self.tik_instance.vec_dup(Constant.FP32_NUM, w_ub_div_mul_sz2, w_len_float, self.cal_repeat, 8)
                self.tik_instance.vec_muls(Constant.FP32_NUM, w_ub_sz2, w_ub_tmp, -1, self.cal_repeat, 8, 8)
                self.tik_instance.vdiv(Constant.FP32_NUM, w_ub_neg_input, w_ub_sz2, w_ub_one, self.cal_repeat, 1, 1, 1,
                                       8, 8, 8)
                self.tik_instance.h_cast(w_ub_int, w_ub_neg_input, "to-zero")
                self.tik_instance.h_cast(w_ub_sz2, w_ub_int, "none")
                self.tik_instance.vec_adds(Constant.FP32_NUM, w_ub_input_div_sz2, w_ub_sz2, 1, self.cal_repeat, 8, 8)
                self.tik_instance.vec_mul(Constant.FP32_NUM, w_ub_input_div_sz2, w_ub_input_div_sz2, w_ub_div_mul_sz2,
                                          self.cal_repeat, 8, 8, 8)
                self.tik_instance.vec_add(Constant.FP32_NUM, w_ub_neg_input, w_ub_tmp, w_ub_input_div_sz2,
                                          self.cal_repeat, 8, 8, 8)

                # calculate `in_coord - len  * (int)(-coord / sz2)`
                self.tik_instance.vdiv(Constant.FP32_NUM, w_ub_sz2, w_ub_tmp, w_ub_one, self.cal_repeat, 1, 1, 1, 8, 8,
                                       8)
                self.tik_instance.h_cast(w_ub_int, w_ub_sz2, "to-zero")
                self.tik_instance.h_cast(w_ub_sz2, w_ub_int, "none")
                self.tik_instance.vec_mul(Constant.FP32_NUM, w_ub_sz2, w_ub_sz2, w_ub_div_mul_sz2, self.cal_repeat, 8,
                                          8, 8)
                self.tik_instance.vec_sub(Constant.FP32_NUM, w_ub_one, w_ub_tmp, w_ub_sz2, self.cal_repeat, 8, 8, 8)

                with self.tik_instance.for_range(0, self.cal_repeat) as re_idx:
                    # select w_ub_sz2
                    self.tik_instance.vec_dup(Constant.FP32_NUM, w_ub_compare, w_len_sw_ub_one, 1, 8)
                    self.tik_instance.vec_cmpv_gt(w_ub_sel, w_ub_tmp[re_idx * Constant.FP32_NUM], w_ub_compare, 1, 8, 8)
                    self.tik_instance.vec_sel(Constant.FP32_NUM, 0, w_ub_sz2[re_idx * Constant.FP32_NUM], w_ub_sel,
                                              w_ub_one[re_idx * Constant.FP32_NUM],
                                              w_ub_tmp[re_idx * Constant.FP32_NUM], 1, 8, 8, 8)

                    # select w_ub_one
                    self.tik_instance.vec_dup(Constant.FP32_NUM, w_ub_input_div_sz2[re_idx * Constant.FP32_NUM], 0, 1,
                                              8)
                    self.tik_instance.vec_cmpv_lt(w_ub_sel, w_ub_tmp[re_idx * Constant.FP32_NUM],
                                                  w_ub_input_div_sz2[re_idx * Constant.FP32_NUM], 1, 8, 8)
                    self.tik_instance.vec_sel(Constant.FP32_NUM, 0, w_ub_one[re_idx * Constant.FP32_NUM], w_ub_sel,
                                              w_ub_neg_input[re_idx * Constant.FP32_NUM],
                                              w_ub_sz2[re_idx * Constant.FP32_NUM], 1, 8, 8, 8)

                # compare 0 with w_ub_neg_input to select max result w_ub_sz2
                self.tik_instance.vec_dup(Constant.FP32_NUM, w_ub_neg_input, 0, self.cal_repeat, 8)
                self.tik_instance.vmax(Constant.FP32_NUM, w_ub_sz2, w_ub_neg_input, w_ub_one, self.cal_repeat, 1, 1, 1,
                                       8, 8, 8)

                # compare len-1 with w_ub_neg_input to select min result w_ub_input_div_sz2
                self.tik_instance.vec_dup(Constant.FP32_NUM, w_ub_one, w_len_sw_ub_one, self.cal_repeat, 8)
                self.tik_instance.vmin(Constant.FP32_NUM, w_ub_input_div_sz2, w_ub_one, w_ub_sz2, self.cal_repeat, 1, 1,
                                       1, 8, 8, 8)

                # move this part result to ub_input
                self.tik_instance.data_move(ub_input, w_ub_input_div_sz2, 0, 1, self.burst_long_map, 0, 0)

    def _map_coord_nearest(self, ub_input, len_hw, ub_size):
        """
        cal the map coord for nearest fill mode
        """
        with self.tik_instance.new_stmt_scope():
            n_ub_sz2 = self.tik_instance.Tensor("float32", (ub_size,), name="n_ub_sz2", scope=tik.scope_ubuf)
            n_ub_one = self.tik_instance.Tensor("float32", (ub_size,), name="n_ub_one", scope=tik.scope_ubuf)
            n_ub_neg_input = self.tik_instance.Tensor("float32", (ub_size,), name="n_ub_neg_input",
                                                      scope=tik.scope_ubuf)
            n_ub_tmp = self.tik_instance.Tensor("float32", (ub_size,), name="n_ub_tmp", scope=tik.scope_ubuf)
            n_ub_compare = self.tik_instance.Tensor("float32", (ub_size,), name="n_ub_compare", scope=tik.scope_ubuf)
            n_len_sn_ub_one = self.tik_instance.Scalar("float32", name="n_len_sn_ub_one")
            n_len_sn_ub_one.set_as(len_hw - 1)

            self.tik_instance.data_move(n_ub_tmp, ub_input, 0, 1, self.burst_long_map, 0, 0)
            # compare 0 with n_ub_neg_input to select max result n_ub_sz2
            self.tik_instance.vec_dup(Constant.FP32_NUM, n_ub_neg_input, 0, self.cal_repeat, 8)
            self.tik_instance.vmax(Constant.FP32_NUM, n_ub_sz2, n_ub_neg_input, n_ub_tmp, self.cal_repeat, 1, 1, 1, 8,
                                   8, 8)

            # compare len-1 with n_ub_neg_input to select min result n_ub_one
            self.tik_instance.vec_dup(Constant.FP32_NUM, n_ub_compare, n_len_sn_ub_one, self.cal_repeat, 8)
            self.tik_instance.vmin(Constant.FP32_NUM, n_ub_one, n_ub_compare, n_ub_sz2, self.cal_repeat, 1, 1, 1, 8, 8,
                                   8)

            # move this part result to ub_input
            self.tik_instance.data_move(ub_input, n_ub_one, 0, 1, self.burst_long_map, 0, 0)

    def _read_with_fill_value(self, input_x, input_y, input_width, input_height):
        """
        do the selection for coord
        """
        with self.tik_instance.if_scope(tik.all(input_x >= 0, input_x < input_width)):
            with self.tik_instance.if_scope(tik.all(input_y >= 0, input_y < input_height)):
                self.flag_xyfloor.set_as(1)
            with self.tik_instance.else_scope():
                self.flag_xyfloor.set_as(0)
        with self.tik_instance.else_scope():
            self.flag_xyfloor.set_as(0)

    def _set_output_value(self, offset):
        """
        set coord xfloor yfloor value
        """
        if self.flag == 0:
            with self.tik_instance.for_range(0, self.input_c) as ub_c_idx:
                of = offset + ub_c_idx
                self.ub_output[of].set_as(self.ub_bili_image[ub_c_idx])
        elif self.flag == 1:
            self.tik_instance.h_cast(self.ub_value_out_cast, self.ub_bili_image, "none")
            with self.tik_instance.for_range(0, self.input_c) as ub_c_idx:
                of = offset + ub_c_idx
                self.ub_output[of].set_as(self.ub_value_out_cast[ub_c_idx])
        elif self.flag == 2:
            self.tik_instance.h_cast(self.ub_value_out_cast, self.ub_bili_image, "to-zero")
            with self.tik_instance.for_range(0, self.input_c) as ub_c_idx:
                of = offset + ub_c_idx
                self.ub_output[of].set_as(self.ub_value_out_cast[ub_c_idx])
        elif self.flag == 3:
            self.tik_instance.h_cast(self.ub_value_origin_cast, self.ub_bili_image, "none")
            self.tik_instance.h_cast(self.ub_value_out_cast, self.ub_value_origin_cast, "to-zero")
            with self.tik_instance.for_range(0, self.input_c) as ub_c_idx:
                of = offset + ub_c_idx
                self.ub_output[of].set_as(self.ub_value_out_cast[ub_c_idx])

    def _move_output_value(self, img_idx, loop_h_idx, loop_w_idx, rep_idx):
        """
        set coord xfloor yfloor value
        """
        if self.flag == 1:
            self.tik_instance.h_cast(self.mv_ub_value_out_cast, self.ub_bili_image, "none")
        elif self.flag == 2:
            self.tik_instance.h_cast(self.mv_ub_value_out_cast, self.ub_bili_image, "to-zero")
        elif self.flag == 3:
            self.tik_instance.h_cast(self.mv_ub_value_origin_cast, self.ub_bili_image, "none")
            self.tik_instance.h_cast(self.mv_ub_value_out_cast, self.mv_ub_value_origin_cast, "to-zero")
        elif self.flag == 0:
            self.tik_instance.data_move(self.mv_ub_value_out_cast, self.ub_bili_image, 0, 1, self.burst_long_bili, 0, 0)

        self.tik_instance.data_move(self.ub_output, self.mv_ub_value_out_cast, 0, 1, self.burst_long_bili, 0, 0)

        self._move_ub_c_to_gm(loop_h_idx, loop_w_idx, rep_idx, img_idx)

    def _set_xy_value_gm(self, val_xy, core_idx, int_y, int_x):
        """
        set coord xfloor yfloor value
        """
        self.burst_long_bili.set_as((self.input_c + self.block_size - 1) // self.block_size)

        with self.tik_instance.if_scope(self.flag_xyfloor == 0):
            if self.flag == 3:
                with self.tik_instance.for_range(0, self.input_c) as c_idx:
                    self.ub_value_origin_h[c_idx].set_as(self.fill_val)
            else:
                self.tik_instance.vec_dup(self.c_mask, self.ub_value_origin_h, self.fill_val, self.bc_repeat, 8)
        with self.tik_instance.else_scope():
            self.offset_img.set_as(
                core_idx * self.input_size + int_y * self.input_w * self.input_c + int_x * self.input_c)
            self.tik_instance.data_move(self.ub_value_origin_h, self.input_gm[self.offset_img], 0, 1,
                                        self.burst_long_bili, 0, 0)

        if self.flag == 0:
            self.tik_instance.data_move(val_xy, self.ub_value_origin_h, 0, 1, self.burst_long_bili, 0, 0)
        elif self.flag == 3:
            self.tik_instance.h_cast(self.ub_value_gm_cast, self.ub_value_origin_h, "none")
            self.tik_instance.h_cast(val_xy, self.ub_value_gm_cast, "none")
        else:
            self.tik_instance.h_cast(val_xy, self.ub_value_origin_h, "none")

    def _bilinear_interpolation(self, core_idx, cal_ubsize):
        """
        cal coord value for bilinear interpolation
        """
        with self.tik_instance.new_stmt_scope():
            # coord_y and coord_x from float to int by floor
            value_xyfloor = self.tik_instance.Tensor("float32", (cal_ubsize,), name="value_xyfloor",
                                                     scope=tik.scope_ubuf)
            value_y_xceil = self.tik_instance.Tensor("float32", (cal_ubsize,), name="value_y_xceil",
                                                     scope=tik.scope_ubuf)
            value_x_yceil = self.tik_instance.Tensor("float32", (cal_ubsize,), name="value_x_yceil",
                                                     scope=tik.scope_ubuf)
            value_xyceil = self.tik_instance.Tensor("float32", (cal_ubsize,), name="value_xyceil", scope=tik.scope_ubuf)
            val_ceil_sub_xy = self.tik_instance.Tensor("float32", (cal_ubsize,), name="val_ceil_sub_xy",
                                                       scope=tik.scope_ubuf)
            val_xy_sub_floor = self.tik_instance.Tensor("float32", (cal_ubsize,), name="val_xy_sub_floor",
                                                        scope=tik.scope_ubuf)
            ub_float_x = self.tik_instance.Tensor("float32", (cal_ubsize,), name="ub_float_x", scope=tik.scope_ubuf)
            ub_ceil_x = self.tik_instance.Tensor("float32", (cal_ubsize,), name="ub_ceil_x", scope=tik.scope_ubuf)
            ub_bili_tmp = self.tik_instance.Tensor("float32", (cal_ubsize,), name="ub_bili_tmp", scope=tik.scope_ubuf)

            int_yfloor = self.tik_instance.Scalar("int32", name="int_yfloor")
            int_xfloor = self.tik_instance.Scalar("int32", name="int_xfloor")
            floor_y = self.tik_instance.Scalar("float32", name="floor_y")
            floor_x = self.tik_instance.Scalar("float32", name="floor_x")
            int_yceil = self.tik_instance.Scalar("int32", name="int_yceil")
            int_xceil = self.tik_instance.Scalar("int32", name="int_xceil")
            float_yceil = self.tik_instance.Scalar("float32", name="float_yceil")
            float_xceil = self.tik_instance.Scalar("float32", name="float_xceil")

            self.tik_instance.scalar_conv('floor', int_yfloor, self.input_y_float)
            self.tik_instance.scalar_conv('floor', int_xfloor, self.input_x_float)

            # int coord_y and int coord_x conv to float
            self.tik_instance.scalar_conv('none', floor_y, int_yfloor)
            self.tik_instance.scalar_conv('none', floor_x, int_xfloor)

            # create int y_ceil and int x_ceil
            int_yceil.set_as(int_yfloor + 1)
            int_xceil.set_as(int_xfloor + 1)
            self.tik_instance.scalar_conv('none', float_yceil, int_yceil)
            self.tik_instance.scalar_conv('none', float_xceil, int_xceil)

            self._read_with_fill_value(int_xfloor, int_yfloor, self.input_w, self.input_h)
            self._set_xy_value_gm(value_xyfloor, core_idx, int_yfloor, int_xfloor)

            self._read_with_fill_value(int_xceil, int_yfloor, self.input_w, self.input_h)
            self._set_xy_value_gm(value_y_xceil, core_idx, int_yfloor, int_xceil)

            self._read_with_fill_value(int_xfloor, int_yceil, self.input_w, self.input_h)
            self._set_xy_value_gm(value_x_yceil, core_idx, int_yceil, int_xfloor)

            self._read_with_fill_value(int_xceil, int_yceil, self.input_w, self.input_h)
            self._set_xy_value_gm(value_xyceil, core_idx, int_yceil, int_xceil)

            self.tik_instance.vec_dup(self.c_mask, ub_ceil_x, float_xceil, self.bc_repeat, 8)
            self.tik_instance.vec_dup(self.c_mask, ub_float_x, self.input_x_float, self.bc_repeat, 8)
            self.tik_instance.vec_sub(self.c_mask, val_ceil_sub_xy, ub_ceil_x, ub_float_x, self.bc_repeat, 8, 8, 8)

            self.tik_instance.vec_dup(self.c_mask, ub_ceil_x, floor_x, self.bc_repeat, 8)
            self.tik_instance.vec_sub(self.c_mask, val_xy_sub_floor, ub_float_x, ub_ceil_x, self.bc_repeat, 8, 8, 8)

            self.tik_instance.vec_mul(self.c_mask, value_xyfloor, value_xyfloor, val_ceil_sub_xy, self.bc_repeat, 8, 8,
                                      8)
            self.tik_instance.vec_mul(self.c_mask, value_y_xceil, value_y_xceil, val_xy_sub_floor, self.bc_repeat, 8, 8,
                                      8)
            self.tik_instance.vec_add(self.c_mask, value_xyfloor, value_xyfloor, value_y_xceil, self.bc_repeat, 8, 8, 8)

            self.tik_instance.vec_mul(self.c_mask, value_x_yceil, value_x_yceil, val_ceil_sub_xy, self.bc_repeat, 8, 8,
                                      8)
            self.tik_instance.vec_mul(self.c_mask, value_xyceil, value_xyceil, val_xy_sub_floor, self.bc_repeat, 8, 8,
                                      8)
            self.tik_instance.vec_add(self.c_mask, value_x_yceil, value_x_yceil, value_xyceil, self.bc_repeat, 8, 8, 8)

            self.tik_instance.vec_dup(self.c_mask, ub_ceil_x, float_yceil, self.bc_repeat, 8)
            self.tik_instance.vec_dup(self.c_mask, ub_float_x, self.input_y_float, self.bc_repeat, 8)
            self.tik_instance.vec_sub(self.c_mask, val_ceil_sub_xy, ub_ceil_x, ub_float_x, self.bc_repeat, 8, 8, 8)

            self.tik_instance.vec_dup(self.c_mask, ub_ceil_x, floor_y, self.bc_repeat, 8)
            self.tik_instance.vec_sub(self.c_mask, val_xy_sub_floor, ub_float_x, ub_ceil_x, self.bc_repeat, 8, 8, 8)

            self.tik_instance.vec_mul(self.c_mask, val_ceil_sub_xy, val_ceil_sub_xy, value_xyfloor, self.bc_repeat, 8,
                                      8, 8)
            self.tik_instance.vec_mul(self.c_mask, val_xy_sub_floor, val_xy_sub_floor, value_x_yceil, self.bc_repeat, 8,
                                      8, 8)
            self.tik_instance.vec_add(self.c_mask, ub_bili_tmp, val_xy_sub_floor, val_ceil_sub_xy, self.bc_repeat, 8, 8,
                                      8)

            self.tik_instance.data_move(self.ub_bili_image, ub_bili_tmp, 0, 1, self.burst_long_bili, 0, 0)

    def _nearest_interpolation(self, img_idx, offset_out):
        """
        cal coord value for nearest interpolation
        """
        self.tik_instance.scalar_conv('round', self.input_x_int, self.input_x_float)
        self.tik_instance.scalar_conv('round', self.input_y_int, self.input_y_float)

        with self.tik_instance.if_scope(tik.all(self.input_x_int >= 0, self.input_x_int < self.input_w)):
            with self.tik_instance.if_scope(tik.all(self.input_y_int >= 0,
                                                    self.input_y_int < self.input_h)):
                self.offset_img.set_as(
                    img_idx * self.input_size + self.input_y_int * self.input_w * self.input_c +
                    self.input_x_int * self.input_c)
                self.tik_instance.data_move(self.ub_images, self.input_gm[self.offset_img], 0, 1, 1, 0, 0)
                with self.tik_instance.for_range(0, self.input_c) as loop_c_idx:
                    self.ub_output[offset_out + loop_c_idx].set_as(self.ub_images[loop_c_idx])
            with self.tik_instance.else_scope():
                with self.tik_instance.for_range(0, self.input_c) as loop_c_idx:
                    self.ub_output[offset_out + loop_c_idx].set_as(self.fill_val)
        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, self.input_c) as loop_c_idx:
                self.ub_output[offset_out + loop_c_idx].set_as(self.fill_val)

    def _c_huge_nearest_interpolation(self, img_idx, loop_h_idx, loop_w_idx, rep_idx):
        """
        cal coord value for nearest interpolation
        """
        if self.flag == 3:
            with self.tik_instance.for_range(0, self.input_c) as lop_c_idx:
                self.ub_output[lop_c_idx].set_as(self.fill_val)
        else:
            self.tik_instance.vec_dup(self.mask_size, self.ub_output, self.fill_val, self.c_repeat, 8)

        self.tik_instance.scalar_conv('round', self.input_x_int, self.input_x_float)
        self.tik_instance.scalar_conv('round', self.input_y_int, self.input_y_float)

        with self.tik_instance.if_scope(tik.all(self.input_x_int >= 0, self.input_x_int < self.input_w)):
            with self.tik_instance.if_scope(tik.all(self.input_y_int >= 0,
                                                    self.input_y_int < self.input_h)):
                self.offset_img.set_as(img_idx * self.input_size + self.input_y_int * self.input_w * self.input_c +
                                       self.input_x_int * self.input_c)
                self.tik_instance.data_move(self.ub_output, self.input_gm[self.offset_img], 0, 1, self.ub_c_repeat, 0,
                                            0)
        self._move_ub_c_to_gm(loop_h_idx, loop_w_idx, rep_idx, img_idx)

    def _copy_only_process_ubs(self, img_idx, ub_height, o_flag, io_flag):
        """
        Only execute case that h*w is more than ub
        """
        with self.tik_instance.for_range(0, ub_height) as loop_h_idx:
            with self.tik_instance.for_range(0, self.output_w) as loop_w_idx:
                # do fill mode select
                offset_io = io_flag * img_idx * Constant.EIGHT_BIT + loop_h_idx * self.output_w + loop_w_idx
                self.input_x_float.set_as(self.ub_input_x[offset_io])
                self.input_y_float.set_as(self.ub_input_y[offset_io])
                offset_oout = o_flag * img_idx * self.output_h * self.output_w * self.input_c + \
                              loop_h_idx * self.output_w * self.input_c + loop_w_idx * self.input_c
                # do interpolation select
                if self.interpolation == 0:
                    self._nearest_interpolation(img_idx, offset_oout)

                else:
                    self._bilinear_interpolation(img_idx, self.calc_ubsize)
                    self._set_output_value(offset_oout)

    def _copy_only_process_c_huge(self, img_idx, rep_idx, ub_height):
        """
        Only execute case that h*w is more than ub
        """
        with self.tik_instance.for_range(0, ub_height) as loop_h_idx:
            with self.tik_instance.for_range(0, self.output_w) as loop_w_idx:
                # do fill mode select
                offset_oo = loop_h_idx * self.output_w + loop_w_idx
                self.input_x_float.set_as(self.ub_input_x[offset_oo])
                self.input_y_float.set_as(self.ub_input_y[offset_oo])

                # do interpolation select
                if self.interpolation == 0:
                    self._c_huge_nearest_interpolation(img_idx, loop_h_idx, loop_w_idx, rep_idx)

                else:
                    self._bilinear_interpolation(img_idx, self.calc_ubsize)
                    self._move_output_value(img_idx, loop_h_idx, loop_w_idx, rep_idx)

    def _move_ub_to_gm(self, rep_idx, img_idx):
        """
        move one w*c to output_gm
        """
        rec_idx = self.tik_instance.Scalar("int32", name="rec_idx")
        self.offset_i.set_as(img_idx * self.output_h * self.output_w * self.input_c + (rep_idx *
                             self.ub_height) * self.output_w * self.input_c)
        self.offset_ini.set_as(self.offset_i + self.offset_ni - self.ub_c_left_num)

        self.tik_instance.data_move(self.output_gm[self.offset_i], self.ub_output, 0, 1, self.ub_c_repeat, 0, 0)

        with self.tik_instance.if_scope(self.ub_c_left > 0):
            with self.tik_instance.for_range(0, self.ub_c_left_num) as dup_idx:
                rec_idx.set_as(self.ub_c_left_num - dup_idx)
                self.ub_aligned[dup_idx].set_as(self.ub_output[self.offset_ni - rec_idx])

            with self.tik_instance.for_range(0, self.ub_c_left) as data_idx:
                self.ub_aligned[self.ub_c_left_num + data_idx].set_as(
                    self.ub_output[self.offset_ni + data_idx])

        self.tik_instance.data_move(self.output_gm[self.offset_ini], self.ub_aligned, 0, 1, 1, 0, 0)

    def _move_ub_c_to_gm(self, loop_h_idx, loop_w_idx, rep_idx, img_idx):
        """
        move one w*c to output_gm
        """
        self.offset_i.set_as(img_idx * self.output_h * self.output_w * self.input_c + (loop_h_idx +
                             rep_idx * self.ub_height) * self.output_w * self.input_c + loop_w_idx * self.input_c)
        with self.tik_instance.if_scope(self.ub_c_left_num > 0):
            self.tik_instance.data_move(self.output_gm[self.offset_i], self.ub_output, 0, 1,
                                        self.ub_c_repeat - 1, 0, 0)

            with self.tik_instance.for_range(0, self.ub_c_left) as dup_idx:
                rec_idx = self.ub_c_left - dup_idx
                self.ub_aligned[dup_idx].set_as(self.ub_output[self.offset_ni - rec_idx])

            with self.tik_instance.for_range(0, self.ub_c_left_num) as data_idx:
                self.ub_aligned[self.ub_c_left + data_idx].set_as(
                    self.ub_output[self.offset_ni + data_idx])
            self.offset_ini.set_as(self.offset_i + self.offset_ni - self.ub_c_left)
            self.tik_instance.data_move(self.output_gm[self.offset_ini], self.ub_aligned, 0, 1, 1, 0, 0)
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(self.output_gm[self.offset_i], self.ub_output, 0, 1, self.ub_c_repeat, 0, 0)

    def _move_ub_image_to_gm(self):
        """
        move image form ub to output gm
        """
        self.ub_c_repeat.set_as(self.input_c * self.output_w * self.output_h * self.input_b // self.block_size)
        self.ub_c_left_num.set_as(self.input_c * self.output_w * self.output_h * self.input_b % self.block_size)
        self.ub_c_left.set_as(self.block_size - self.ub_c_left_num)
        self.offset_ni.set_as(self.ub_c_repeat * self.block_size)
        self.offset_ini.set_as(self.offset_ni - self.ub_c_left)

        with self.tik_instance.if_scope(self.ub_c_repeat > 0):
            self.tik_instance.data_move(self.output_gm, self.ub_output, 0, 1, self.ub_c_repeat, 0, 0)

        with self.tik_instance.if_scope(self.ub_c_left_num > 0):
            with self.tik_instance.if_scope(self.ub_c_repeat == 0):
                self.tik_instance.data_move(self.output_gm, self.ub_output, 0, 1, 1, 0, 0)
            with self.tik_instance.else_scope():
                with self.tik_instance.for_range(0, self.ub_c_left) as dup_idx:
                    rec_idx = self.ub_c_left - dup_idx
                    self.ub_aligned[dup_idx].set_as(self.ub_output[self.offset_ni - rec_idx])
                with self.tik_instance.for_range(0, self.ub_c_left_num) as data_idx:
                    self.ub_aligned[self.ub_c_left + data_idx].set_as(self.ub_output[self.offset_ni + data_idx])
                self.tik_instance.data_move(self.output_gm[self.offset_ini], self.ub_aligned, 0, 1, 1, 0, 0)

    def _cal_param_huge(self):
        """
        calculate parameters that "moving img" need
        """
        self.ub_c_left_num.set_as(self.input_c % self.block_size)
        self.ub_c_repeat.set_as((self.input_c + self.block_size - 1) // self.block_size)
        self.ub_c_left.set_as(self.block_size - self.ub_c_left_num)
        self.mask_size.set_as(Constant.REP_SIZE // self.block_size)
        self.c_repeat.set_as((self.input_c + self.mask_size - 1) // self.block_size)
        self.offset_ni.set_as((self.ub_c_repeat - 1) * self.block_size)

    def _cal_param(self, ub_h):
        """
        calculate parameters that "moving img" need
        """
        self.ub_c_repeat.set_as(self.input_c * self.output_w * ub_h // self.block_size)
        self.ub_c_left.set_as(self.input_c * self.output_w * ub_h % self.block_size)
        self.ub_c_left_num.set_as(self.block_size - self.ub_c_left)
        self.offset_ni.set_as(self.ub_c_repeat * self.block_size)

    def _gt_ub_copy(self, core_idx):
        """
        To store images h*w need several times.
        transforms_b is one
        """
        with self.tik_instance.for_range(0, self.img_num) as img_num_idx:
            img_idx = core_idx + img_num_idx * 32

            with self.tik_instance.for_range(0, self.ub_repeat_time) as rep_idx:
                self.h_start.set_as(rep_idx * self.ub_height)

                with self.tik_instance.if_scope(rep_idx == self.ub_floor_repeat):
                    self.ub_h.set_as(self.ub_repeat_left)
                with self.tik_instance.else_scope():
                    self.ub_h.set_as(self.ub_height)

                self.h_end.set_as(self.ub_h + rep_idx * self.ub_height)
                self._init_xy_ub_tensor(Constant.MAX_DATA_UB)
                self._cal_coords(Constant.MAX_DATA_UB, rep_idx)
                self._map_coord(Constant.MAX_DATA_UB, 0)
                self._init_ub_tensor(Constant.MAX_DATA_UB * 8, self.calc_ubsize)

                with self.tik_instance.if_scope(self.input_c >= self.block_size):
                    self._cal_param_huge()
                    self._copy_only_process_c_huge(img_idx, rep_idx, self.ub_h)
                with self.tik_instance.else_scope():
                    self._cal_param(self.ub_h)
                    self._copy_only_process_ubs(img_idx, self.ub_h, 0, 0)
                    self._move_ub_to_gm(rep_idx, img_idx)

    def _gt_ub_copy_transforms(self, core_idx):
        """
        To store images h*w need several times.
        transforms_b is one
        """
        with self.tik_instance.for_range(0, self.img_num) as img_num_idx:
            img_idx = core_idx + img_num_idx * 32
            self._move_transforms(img_idx)

            with self.tik_instance.for_range(0, self.ub_repeat_time) as rep_idx:
                self.h_start.set_as(rep_idx * self.ub_height)

                with self.tik_instance.if_scope(rep_idx == self.ub_floor_repeat):
                    self.ub_h.set_as(self.ub_repeat_left)
                with self.tik_instance.else_scope():
                    self.ub_h.set_as(self.ub_height)

                self.h_end.set_as(self.ub_h + rep_idx * self.ub_height)
                self._init_xy_ub_tensor(Constant.MAX_DATA_UB)
                self._cal_coords(Constant.MAX_DATA_UB, rep_idx)
                self._map_coord(Constant.MAX_DATA_UB, 0)
                self._init_ub_tensor(Constant.MAX_DATA_UB * 8, self.calc_ubsize)

                with self.tik_instance.if_scope(self.input_c >= self.block_size):
                    self._cal_param_huge()
                    self._copy_only_process_c_huge(img_idx, rep_idx, self.ub_h)
                with self.tik_instance.else_scope():
                    self._cal_param(self.ub_h)
                    self._copy_only_process_ubs(img_idx, self.ub_h, 0, 0)
                    self._move_ub_to_gm(rep_idx, img_idx)

    def _one_block_for_one_transform(self):
        """
        one core to deal with case "h*w*c is less than 1 block"
        """
        self._init_xy_ub_tensor(Constant.BLC_MAX_DATA_UB)
        self.h_start.set_as(0)
        self.h_end.set_as(self.output_h)
        self.ub_h.set_as(self.output_h)
        self._cal_coords(Constant.BLC_MAX_DATA_UB, -1)
        self._map_coord(Constant.BLC_MAX_DATA_UB, 0)
        self._init_block_ub_tensor(Constant.BLC_MAX_DATA_UB * 8, self.calc_ubsize)

        with self.tik_instance.for_range(0, self.input_b) as img_idx:
            self._copy_only_process_ubs(img_idx, self.output_h, 1, 0)

        self._move_ub_image_to_gm()

    def _one_block_for_transforms(self):
        """
        one core to deal with case "h*w*c is less than 1 block"
        images_b is equal to transform_b
        """
        self._init_xy_ub_tensor(Constant.BLC_MAX_DATA_UB)
        self.ub_h.set_as(self.output_h)
        self._cal_coords_one_block()
        self._map_coord(Constant.BLC_MAX_DATA_UB, 1)
        self._init_block_ub_tensor(Constant.BLC_MAX_DATA_UB * 8, self.calc_ubsize)

        with self.tik_instance.for_range(0, self.input_b) as img_idx:
            self._copy_only_process_ubs(img_idx, self.output_h, 1, 1)

        self._move_ub_image_to_gm()

    def _move_transforms(self, core_idx):
        """
        transform_b is equal to images_b
        """
        with self.tik_instance.new_stmt_scope():
            # move transform params from gm to ub
            ub_transforms = self.tik_instance.Tensor(self.trans_dtype, (8,),
                                                     name="ub_transforms",
                                                     scope=tik.scope_ubuf)
            offset_trans = core_idx * 8
            self.tik_instance.data_move(ub_transforms, self.transform_gm[offset_trans], 0, 1, 1, 0, 0)

            # scalar for store transforms data
            self.trans_a0.set_as(ub_transforms[0])
            self.trans_a1.set_as(ub_transforms[1])
            self.trans_a2.set_as(ub_transforms[2])
            self.trans_b0.set_as(ub_transforms[3])
            self.trans_b1.set_as(ub_transforms[4])
            self.trans_b2.set_as(ub_transforms[5])
            self.trans_c0.set_as(ub_transforms[6])
            self.trans_c1.set_as(ub_transforms[7])

    def _move_one_transform(self):
        """
        transform_b is equal to one
        """
        with self.tik_instance.new_stmt_scope():
            # move transform params from gm to ub
            ub_one_transform = self.tik_instance.Tensor(self.trans_dtype, (8,),
                                                        name="ub_one_transform",
                                                        scope=tik.scope_ubuf)
            self.tik_instance.data_move(ub_one_transform, self.transform_gm, 0, 1, 1, 0, 0)

            # scalar for store transforms data
            self.trans_a0.set_as(ub_one_transform[0])
            self.trans_a1.set_as(ub_one_transform[1])
            self.trans_a2.set_as(ub_one_transform[2])
            self.trans_b0.set_as(ub_one_transform[3])
            self.trans_b1.set_as(ub_one_transform[4])
            self.trans_b2.set_as(ub_one_transform[5])
            self.trans_c0.set_as(ub_one_transform[6])
            self.trans_c1.set_as(ub_one_transform[7])


# 'pylint: disable=unused-argument
@register_operator("ImageProjectiveTransform")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_STR,
                            para_check.KERNEL_NAME)
def image_projective_transform(images,
                               transforms,
                               output_shape,
                               transformed_image,
                               interpolation,
                               fill_mode="CONSTANT",
                               kernel_name="image_projective_transform"):
    """
    Generate arg_min operator use arg_min

    Parameters
    ----------
    images: dict
        data of input, support "float16", "float32", "uint8", "int32".
    transforms: dict
        3 x 3 projective transformation matrix, support "float32".
    output_shape: dict
        shape of output, support "int32".
    interpolation: str
        interpolation method, support "NEAREST" or "BILINEAR".
    fill_mode: str
        An optional string, Default is "CONSTANT", support "REFLECT", "WRAP", "NEAREST" or "CONSTANT".
    y: dict
        index of output.
    kernel_name: str
        kernel name, default value is "image_projective_transform"

    Returns
    -------
    tik_instance
    """
    images_dtype = images.get("dtype").lower()
    transforms_dtype = transforms.get("dtype").lower()
    output_shape_dtype = output_shape.get("dtype").lower()

    # check input shape, format and dtype
    para_check.check_dtype(images_dtype, ("float16", "float32", "uint8", "int32"), param_name="images")
    para_check.check_dtype(transforms_dtype, ("float32",), param_name="transforms")
    para_check.check_dtype(output_shape_dtype, ("int32",), param_name="output_shape")

    obj = ImageProjectiveTransform(images_dtype,
                                   transforms_dtype,
                                   interpolation,
                                   fill_mode,
                                   kernel_name)

    return obj.image_projective_compute()
