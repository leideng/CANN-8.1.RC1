"""
Copyright (c) Huawei Technologies Co., Ltd. 2021-2023. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

resize_nearest_neighbor_v2
"""
from abc import ABCMeta, abstractmethod
from functools import partial
from impl import common_util
from impl.util import util_tik_comm_func
from impl.util.util_tik_comm_func import OpBase
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.dynamic.resize_nearest_bilinear_2d import resize_2d
from tbe.tik.tik_lib.tik_soc_manager import TikSocManager


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # ting param num
    TILING_ARG_NUM = 12
    # reserved ub size
    RESERVED_UB_SIZE = 8 * 1024
    # 8 numbers per block for float32 / int32
    FP32_PER_BLOCK = 8
    # nBurst max value of `data_move`
    BURST_NUM_MAX = 4095
    # burstLen max value of `data_move`
    BURST_LEN_MAX = 65535
    # stride max value of `data_move`
    BURST_STRIDE_MAX = 65535
    # whether exceed BURST_STRIDE_MAX
    EXCEED_BURST_STRIDE = 1
    # max h/w value in compute dtype of float16
    MAX_H_W = 2048


# 'pylint: disable=too-many-instance-attributes,too-many-arguments,unused-argument
# 'pylint: disable=too-many-locals,too-many-statements,unused-argument,invalid-name
class ResizeNearestNeighbor(OpBase):
    """ResizeNearestNeighbor basic information

    Tiling key expression: 1abcde
        a: whether H is integer zoom-in.
        b: whether W is integer zoom-in. 3: srcW = dstW
        c/d: not used currently
        e: particular flag for SMALL images AlignHW
    Note: Scenarios that align_corners or half_pixel_centers is True will be considered as 100000
    """

    # processor dictionary for different tiling keys
    _processors = {
        111000: lambda op: AlignHW(op, False),
        113000: lambda op: AlignHW(op, True),
        111001: lambda op: AlignHWSmall(op, False),
        113001: lambda op: AlignHWSmall(op, True),
        101000: lambda op: AlignOnlyW(op, False),
        103000: lambda op: AlignOnlyW(op, True),
        100000: lambda op: Default(op)
    }

    def __init__(self, images, size, y, align_corners, half_pixel_centers, kernel_name):
        OpBase.__init__(self)
        self.images_dtype = images.get("dtype").lower()
        self.images_dtype = self.images_dtype if self.images_dtype != "bfloat16" else "float16"
        self.size_dtype = size.get("dtype").lower()
        self.align_corners = align_corners
        self.half_pixel_centers = half_pixel_centers

        # check dtype
        para_check.check_dtype(self.size_dtype, ("int64", "int32"), param_name="size")
        para_check.check_dtype(self.images_dtype, ("float32", "float16"), param_name="images")

        self.kernel_name = kernel_name
        self.ub_size_bytes = self.ub_size_bytes - Constant.RESERVED_UB_SIZE

        self.shape_c0 = 16
        self.block_num = 16 if self.images_dtype in ("float16",) else 8
        self.ub_max_num = self.ub_size_bytes // 32 // 2 * self.block_num
        self.c0_blocks = self.shape_c0 // self.block_num

        self.l1_size_bytes = 0 \
            if not tbe_platform.intrinsic_check_support("Intrinsic_data_move_l12ub") \
            else tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)
        self.l1_exists = True if self.l1_size_bytes > 0 else False
        self.l1_max_num = self.l1_size_bytes // 2 // (2 if self.images_dtype in ("float16",) else 4)

        # init gm addr
        tiling_dict = {"dtype": "int64", "shape": (Constant.TILING_ARG_NUM,)}
        self.op_init_gm([images, size], [y], tiling_info=tiling_dict, is_fused_1d=True)
        self.images_gm, self.size_gm = self.input_gm_list
        self.out_gm = self.output_gm_list[0]

        self.compute_dtype = "float32" if tbe_platform.api_check_support("tik.vadds", "float32") else "float16"
        self.coordinate_dtype = "int32"
        if self.compute_dtype == "float16" and tbe_platform.api_check_support("tik.vconv", "s162f16"):
            self.coordinate_dtype = "int16"
        self.tiling_dtype = self.coordinate_dtype if self.compute_dtype == "float16" else "int64"
        self.compute_block_num = 32 // common_util.get_data_size(self.compute_dtype)
        self.block_strides = 64 // self.compute_block_num

        self.is_support_vdiv = tbe_platform.api_check_support("tik.vdiv", self.compute_dtype)
        self.is_support_vcopy = tbe_platform.api_check_support("tik.vcopy")

        # will be set as `(IN_X / OUT_X)` or `((IN_X - 1)/(OUT_X - 1))` when needed
        self.h_rec_scale_fpx = self.tik_instance.Scalar(self.compute_dtype, name="h_rec_scale_fpx")
        self.w_rec_scale_fpx = self.tik_instance.Scalar(self.compute_dtype, name="w_rec_scale_fpx")
        # will be set as `(OUT_X // IN_X)` w/o considering align_corners
        self.h_scale = self.tik_instance.Scalar("int64", name="h_scale")
        self.w_scale = self.tik_instance.Scalar("int64", name="w_scale")

        # init tiling data
        self.tiling_key = self.tik_instance.Scalar("int64", name="tiling_key")
        self.image_nc1 = self.tik_instance.Scalar("int64", name="image_nc1")
        self.input_height = self.tik_instance.Scalar("int64", name="input_height")
        self.input_width = self.tik_instance.Scalar("int64", name="input_width")
        self.output_height = self.tik_instance.Scalar("int64", name="output_height")
        self.output_width = self.tik_instance.Scalar("int64", name="output_width")
        self.tiling_nc1_cut_num = self.tik_instance.Scalar("int64", name="tiling_nc1_cut_num")
        self.tiling_height_cut_num = self.tik_instance.Scalar("int64", name="tiling_height_cut_num")
        self.tiling_width_cut_num = self.tik_instance.Scalar("int64", name="tiling_width_cut_num")

        # init scalars for each core
        # nc1 start addr offset per core
        self.core_nc_start = self.tik_instance.Scalar("int64", name="core_nc_start")
        # h start addr offset per core
        self.core_height_start = self.tik_instance.Scalar("int64", name="core_height_start")
        # w start addr offset per core
        self.core_width_start = self.tik_instance.Scalar("int64", name="core_width_start")
        # nc1 process len per core
        self.nc_per_core = self.tik_instance.Scalar("int64", name="nc_per_core")
        # h process len per core
        self.h_per_core = self.tik_instance.Scalar("int64", name="h_per_core")
        # w process len per core
        self.w_per_core = self.tik_instance.Scalar("int64", name="w_per_core")
        # cut-ed height, input or output height
        self.cut_height_num = self.tik_instance.Scalar("int64", name="cut_height_num")
        # cut-ed width, input or output width
        self.cut_width_num = self.tik_instance.Scalar("int64", name="cut_width_num")

    def resize_nearest_neighbor_v2_operator(self):
        """
        resize_nearest_neighbor_v2_operator
        """
        # register compute base on tiling_key
        register_func = partial(self.regist_compute, tiling_func=self._functions)
        for k in self._processors:
            register_func(k, key=k)

        # run all registered compute base tiling key
        self.op_run_compute()

        # Build CCE
        max_w_len = (self.ub_max_num // self.shape_c0) if self.l1_exists else ((self.ub_max_num // self.shape_c0) - 1)
        tbe_context.get_context().add_compile_info("vars", {"ub_size": self.ub_size_bytes,
                                                            "core_num": self.core_nums,
                                                            "max_w_len": max_w_len,  # max_w_len actually is w_scale
                                                            "align_corners": int(self.align_corners),
                                                            "half_pixel_centers": int(self.half_pixel_centers)})
        # Set it as false. it can only be True in DSL
        self.opt_config.update({"out_of_bound_sync_check": False})
        self.op_build_cce()

        return self.tik_instance

    def tiling_args(self):
        """
        tiling_args
        tiling key  tiling_key
        input info  tiling_batch, tiling_c1, input_height, input_width
        output info output_height, output_width
        cut info    tiling_bc1_, tiling_height_cut_num, tiling_width_cut_num
        """
        with self.tik_instance.new_stmt_scope():
            tiling_ub = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                                 name="tiling_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, (Constant.TILING_ARG_NUM + 3) // 4, 0, 0)
            self.tiling_key.set_as(tiling_ub[0])
            tmp_batch = self.tik_instance.Scalar("int64", "tmp_batch", tiling_ub[1])
            tmp_c1 = self.tik_instance.Scalar("int64", "tmp_c1", tiling_ub[2])
            self.image_nc1.set_as(tmp_batch * tmp_c1)
            self.input_height.set_as(tiling_ub[3])
            self.input_width.set_as(tiling_ub[4])
            self.output_height.set_as(tiling_ub[5])
            self.output_width.set_as(tiling_ub[6])
            self.tiling_nc1_cut_num.set_as(tiling_ub[7])
            self.tiling_height_cut_num.set_as(tiling_ub[8])
            self.tiling_width_cut_num.set_as(tiling_ub[9])
            self.running_core_num.set_as(tiling_ub[10])

    def core_scedule_args(self, core_idx):
        """
        get runtime tiling parameters from tiling data with core_id

        need_input1: image info -->
                   tiling_batch*tiling_c1 input_height input_width output_height output_width
        need_input2: cut core info ---> tiling_nc1_cut_num tiling_height_cut_num tiling_width_cut_num
        output: the process info for each core -->
                   self.core_nc_start/self.nc_per_core
                   self.core_height_start/self.h_per_core
                   self.core_width_start/self.w_per_core

        proc:
            nc_per_core = (batch*c1 + bc1_cut_num - 1) // bc1_cut_num
            core_nc_start = (core_id // (height_cut_num * width_cut_num)) * nc_per_core
            h_per_core = (height + height_cut_num - 1) // height_cut_num
            core_height_start = ((core_id % (height_cut_num * width_cut_num)) // width_cut_num) * h_per_core
            w_per_core = (width + width_cut_num - 1) // width_cut_num
            core_width_start = ((core_id % (height_cut_num * width_cut_num)) // width_cut_num) * w_per_core

            for example:
                input info:
                    16, 2, 32, 32, 16 resize to 16, 2, 64, 64, 16     h from 32->64 w from 32->64
                cut info: tiling_nc1_cut_num, tiling_height_cut_num, tiling_width_cut_num
                    4, 4, 2

                nc_per_core = ceil(32, 4) = 8
                core_nc_start = (core_idx // (4*2)) * nc_per_core
                   ---> 0 <= core_idx < 8  core_nc_start = 0
                   ---> 8 <= core_idx < 16  core_nc_start = 8
                   ---> 16 <= core_idx < 24  core_nc_start = 16
                   ---> 24 <= core_idx < 32  core_nc_start = 24
        """
        self.cut_height_num.set_as(self.output_height)
        self.cut_width_num.set_as(self.output_width)
        with self.tik_instance.if_scope(tik.any(self.tiling_key == 111000, self.tiling_key == 113000,
                                                self.tiling_key == 111001, self.tiling_key == 113001)):
            # when tiling_key is 111000 / 113000 / 111001 / 113001, will cut by input
            self.cut_height_num.set_as(self.input_height)
            self.cut_width_num.set_as(self.input_width)
        with self.tik_instance.if_scope(self.tiling_key == 101000):
            self.cut_width_num.set_as(self.input_width)

        # fix the core cut num
        # fix for height_cut_num
        self.tiling_height_cut_num.set_as(
            (self.cut_height_num + self.tiling_height_cut_num - 1) // self.tiling_height_cut_num)
        self.tiling_height_cut_num.set_as(
            (self.cut_height_num + self.tiling_height_cut_num - 1) // self.tiling_height_cut_num)
        # fix for width_cut_num
        self.tiling_width_cut_num.set_as(
            (self.cut_width_num + self.tiling_width_cut_num - 1) // self.tiling_width_cut_num)
        self.tiling_width_cut_num.set_as(
            (self.cut_width_num + self.tiling_width_cut_num - 1) // self.tiling_width_cut_num)
        # fix for nc_cut_num
        self.tiling_nc1_cut_num.set_as(
            (self.image_nc1 + self.tiling_nc1_cut_num - 1) // self.tiling_nc1_cut_num)
        self.tiling_nc1_cut_num.set_as(
            (self.image_nc1 + self.tiling_nc1_cut_num - 1) // self.tiling_nc1_cut_num)

        nc_per_core = (self.image_nc1 + self.tiling_nc1_cut_num - 1) // self.tiling_nc1_cut_num
        h_per_core = (self.cut_height_num + self.tiling_height_cut_num - 1) // self.tiling_height_cut_num
        w_per_core = (self.cut_width_num + self.tiling_width_cut_num - 1) // self.tiling_width_cut_num
        self.core_nc_start.set_as(
            (core_idx // (self.tiling_height_cut_num * self.tiling_width_cut_num)) * nc_per_core)
        self.core_height_start.set_as(
            ((core_idx % (self.tiling_height_cut_num * self.tiling_width_cut_num))
             // self.tiling_width_cut_num) * h_per_core)
        self.core_width_start.set_as(
            ((core_idx % (self.tiling_height_cut_num * self.tiling_width_cut_num))
             % self.tiling_width_cut_num) * w_per_core)
        self.nc_per_core.set_as(nc_per_core)
        self.h_per_core.set_as(h_per_core)
        self.w_per_core.set_as(w_per_core)

        with self.tik_instance.if_scope(
                self.core_nc_start + self.nc_per_core >= self.image_nc1):
            self.nc_per_core.set_as(self.image_nc1 - self.core_nc_start)
        with self.tik_instance.if_scope(
                self.core_height_start + self.h_per_core >= self.cut_height_num):
            self.h_per_core.set_as(self.cut_height_num - self.core_height_start)
        with self.tik_instance.if_scope(
                self.core_width_start + self.w_per_core >= self.cut_width_num):
            self.w_per_core.set_as(self.cut_width_num - self.core_width_start)

        core_used = self.tiling_width_cut_num * self.tiling_height_cut_num * self.tiling_nc1_cut_num
        with self.tik_instance.if_scope(core_idx >= core_used):
            self.nc_per_core.set_as(0)
            self.h_per_core.set_as(0)
            self.w_per_core.set_as(0)
        with self.tik_instance.else_scope():
            self._calculate_scale()

    def _calculate_scale(self):
        """ calculate scale by input h/w and output h/w """
        def _do_calculate(input_value, output_value, scale_fpx):
            with self.tik_instance.new_stmt_scope():
                if self.is_support_vcopy:
                    # for A920 precision fall down issue. Fix later
                    tmp_int32_tensor = self.tik_instance.Tensor("int32", (Constant.FP32_PER_BLOCK,),
                                                                name="tmp_int32_tensor", scope=tik.scope_ubuf)
                    tmp_fp32_tensor = self.tik_instance.Tensor("float32", (Constant.FP32_PER_BLOCK,),
                                                            name="tmp_fp32_tensor", scope=tik.scope_ubuf)

                    tmp_int32_tensor[0].set_as(input_value)
                    tmp_int32_tensor[1].set_as(output_value)
                    util_tik_comm_func.tik_func_vconv(self.tik_instance, tmp_fp32_tensor, tmp_int32_tensor, 2)

                    input_fp32 = self.tik_instance.Scalar("float32", "tmp_input_fp32", tmp_fp32_tensor[0])
                    output_fp32 = self.tik_instance.Scalar("float32", "tmp_output_fp32", tmp_fp32_tensor[1])
                    with self.tik_instance.if_scope(tik.all(self.align_corners, output_value > 1)):
                        input_fp32.set_as(input_fp32 - 1.0)
                        output_fp32.set_as(output_fp32 - 1.0)
                    scale_fpx.set_as(input_fp32 / output_fp32)
                else:
                    tmp_intx_tensor = self.tik_instance.Tensor(self.coordinate_dtype, (self.compute_block_num * 2,),
                                                               name="tmp_intx_tensor", scope=tik.scope_ubuf)
                    tmp_fpx_tensor = self.tik_instance.Tensor(self.compute_dtype, (self.compute_block_num * 3,),
                                                              name="tmp_fpx_tensor", scope=tik.scope_ubuf)

                    tmp_intx_tensor[0].set_as(input_value)
                    tmp_intx_tensor[self.compute_block_num].set_as(output_value)
                    util_tik_comm_func.tik_func_vconv(self.tik_instance, tmp_fpx_tensor[0:], tmp_intx_tensor[0:], 1)
                    util_tik_comm_func.tik_func_vconv(self.tik_instance, tmp_fpx_tensor[self.compute_block_num:],
                                                      tmp_intx_tensor[self.compute_block_num:], 1)

                    with self.tik_instance.if_scope(tik.all(self.align_corners, output_value > 1)):
                        self.tik_instance.vadds(1, tmp_fpx_tensor[0:], tmp_fpx_tensor[0:], -1.0, 2, 1, 1, 1, 1)

                    if self.is_support_vdiv:
                        self.tik_instance.vdiv(1, tmp_fpx_tensor[0:], tmp_fpx_tensor[0:],
                                               tmp_fpx_tensor[self.compute_block_num:], 1, 1, 1, 1, 1, 1, 1)
                    else:
                        # for A310
                        util_tik_comm_func.tik_func_vrec(self.tik_instance,
                                                         tmp_fpx_tensor[self.compute_block_num * 2:],
                                                         tmp_fpx_tensor[self.compute_block_num:], 1, 1, 1, 1, 1, 1)
                        self.tik_instance.vmul(1, tmp_fpx_tensor[0:], tmp_fpx_tensor[0:],
                                               tmp_fpx_tensor[self.compute_block_num * 2:], 1, 1, 1, 1, 1, 1, 1)
                    scale_fpx.set_as(tmp_fpx_tensor[0])

        self.h_scale.set_as(self.output_height // self.input_height)
        self.w_scale.set_as(self.output_width // self.input_width)

        with self.tik_instance.if_scope(tik.any(self.tiling_key == 101000, self.tiling_key == 103000)):
            _do_calculate(self.input_height, self.output_height, self.h_rec_scale_fpx)
        with self.tik_instance.elif_scope(self.tiling_key == 100000):
            _do_calculate(self.input_height, self.output_height, self.h_rec_scale_fpx)
            _do_calculate(self.input_width, self.output_width, self.w_rec_scale_fpx)

    def _functions(self, key: int):
        """invoke each tiling functions

        Parameters
        ----------
        key : int
            tiling key
        """
        with self.tik_instance.if_scope(self.w_per_core == 0):
            self.tik_instance.tik_return()

        processor = self._processors.get(key)
        processor(self).run()


class ProcessorBase(metaclass=ABCMeta):
    def __init__(self, resize_op_obj: ResizeNearestNeighbor, is_w_equal: bool = False) -> None:
        self.op = resize_op_obj
        self.tik_inst = resize_op_obj.tik_instance
        self.is_w_equal = is_w_equal

        # memory in UB/L1 used to save data moved from GM
        self.image_in_buf_ping, self.image_in_buf_pong = None, None
        # memory in UB used to save temporary re-ordered data
        self.reorder_buf_ping, self.reorder_buf_pong = None, None

        # scalars for loop
        self.w_in_per_loop = self.tik_inst.Scalar("int64", name="w_in_per_loop")
        self.w_out_per_loop = self.tik_inst.Scalar("int64", name="w_out_per_loop")
        self.w_loop_num = self.tik_inst.Scalar("int64", name="w_loop_num")
        self.w_loop_tail = self.tik_inst.Scalar("int64", name="w_loop_tail")

        # GM in / out start per core
        self.gm_in_start = self.tik_inst.Scalar("int64", "gm_in_start")
        self.gm_out_start = self.tik_inst.Scalar("int64", "gm_out_start")

    @staticmethod
    def calculate_src_idx(tik_instance: tik.Tik, scale, in_idx_ub, out_idx_fp_ub,
                          idx_num: int, align_corners: bool, half_pixel_centers: bool) -> None:
        """
        if not self.align_corners and self.half_pixel_centers:
            # vconv_f322s32f((idx + 0.5) * scale)
        if not self.align_corners and not self.half_pixel_centers:
            # vconv_f322s32f(idx * scale)
        if self.align_corners and not self.half_pixel_centers:
            # vconv_f322s32r(idx * scale)
        if self.align_corners and self.half_pixel_centers:
            # vconv_f322s32r((idx + 0.5) * scale)
        """
        with tik_instance.new_stmt_scope():
            calc_out_in_idx_tmp_ub = tik_instance.Tensor(out_idx_fp_ub.dtype, out_idx_fp_ub.shape,
                                                         name="calc_out_in_idx_tmp_ub", scope=tik.scope_ubuf)
            vector_repeat_num = (idx_num + 63) // 64
            block_stride = 64 // (32 // common_util.get_data_size(out_idx_fp_ub.dtype))
            if half_pixel_centers:
                # `calc: (idx + 0.5) * scale`
                tik_instance.vadds(64, calc_out_in_idx_tmp_ub, out_idx_fp_ub, 0.5,
                                   vector_repeat_num, 1, 1, block_stride, block_stride)
                tik_instance.vmuls(64, calc_out_in_idx_tmp_ub, calc_out_in_idx_tmp_ub, scale,
                                   vector_repeat_num, 1, 1, block_stride, block_stride)
            else:
                # `calc: idx * scale`
                tik_instance.vmuls(64, calc_out_in_idx_tmp_ub, out_idx_fp_ub, scale,
                                   vector_repeat_num, 1, 1, block_stride, block_stride)
            if align_corners:
                # fix bug
                conv_mode = "away-zero" if tbe_platform.api_check_support("tik.data_move_pad") is True else "round"
                if TikSocManager.is_v210_soc():
                    conv_mode = "away-zero"
                # will use vconv_f322s32r to cast to int32
                util_tik_comm_func.tik_func_vconv(tik_instance, in_idx_ub, calc_out_in_idx_tmp_ub,
                                                  vector_repeat_num * 64, mode=conv_mode)
            else:
                # will use vconv_f322s32f to cast to int32
                util_tik_comm_func.tik_func_vconv(tik_instance, in_idx_ub, calc_out_in_idx_tmp_ub,
                                                  vector_repeat_num * 64, mode="floor")

    @staticmethod
    def nc_batch_move_from_gm(tik_instance: tik.Tik, op: ResizeNearestNeighbor,
                              gm_in_offset, in_image_buf, nc_len, w_in_len, image_in_exceed_stride) -> None:
        """ GM -> L1/UB """
        with tik_instance.new_stmt_scope(disable_sync=True):
            burst_num = nc_len
            burst_len = w_in_len * op.c0_blocks
            with tik_instance.if_scope(image_in_exceed_stride != Constant.EXCEED_BURST_STRIDE):
                burst_src_gap = op.input_height * op.input_width * op.c0_blocks - burst_len
                tik_instance.data_move(in_image_buf[0:], op.images_gm[gm_in_offset:],
                                       0, burst_num, burst_len, burst_src_gap, 0)
            with tik_instance.else_scope():
                with tik_instance.for_range(0, burst_num) as _nc_idx:
                    src_offset = gm_in_offset + _nc_idx * op.input_height * op.input_width * op.shape_c0
                    dst_offset = _nc_idx * w_in_len * op.shape_c0
                    tik_instance.data_move(in_image_buf[dst_offset:], op.images_gm[src_offset:],
                                           0, 1, burst_len, 0, 0)

    @staticmethod
    def nc_batch_move_to_gm(tik_instance: tik.Tik, op: ResizeNearestNeighbor,
                            gm_out_offset, reorder_buf, nc_len, w_out_len, image_out_exceed_stride) -> None:
        """ UG -> GM """
        with tik_instance.new_stmt_scope(disable_sync=True):
            burst_num = nc_len
            burst_len = w_out_len * op.c0_blocks
            with tik_instance.if_scope(image_out_exceed_stride != Constant.EXCEED_BURST_STRIDE):
                burst_dst_gap = op.output_height * op.output_width * op.c0_blocks - burst_len
                tik_instance.data_move(op.out_gm[gm_out_offset:], reorder_buf[0:],
                                       0, burst_num, burst_len, 0, burst_dst_gap)
            with tik_instance.else_scope():
                with tik_instance.for_range(0, burst_num) as _nc_idx:
                    src_offset = _nc_idx * w_out_len * op.shape_c0
                    dst_offset = gm_out_offset + _nc_idx * op.output_height * op.output_width * op.shape_c0
                    tik_instance.data_move(op.out_gm[dst_offset:], reorder_buf[src_offset:],
                                           0, 1, burst_len, 0, 0)

    @staticmethod
    def reorder_in_ub_for_w_align(tik_instance: tik.Tik, op: ResizeNearestNeighbor,
                                  in_image_buf, reorder_buf, burst_num) -> None:
        """ L1/UB -> UB """
        burst_len = op.c0_blocks
        burst_dst_gap = op.w_scale * op.c0_blocks - burst_len
        if op.l1_exists:
            # L1 -> UB
            tik_instance.data_move(reorder_buf[0:], in_image_buf[0:],
                                   0, burst_num, burst_len, 0, burst_dst_gap)
            # UB -> UB
            with tik_instance.new_stmt_scope(disable_sync=True):
                with tik_instance.for_range(1, op.w_scale) as w_cp_idx:
                    dst_offset = w_cp_idx * op.shape_c0
                    tik_instance.data_move(reorder_buf[dst_offset:], reorder_buf[0:],
                                           0, burst_num, burst_len, burst_dst_gap, burst_dst_gap)
        else:
            # UB -> UB
            with tik_instance.new_stmt_scope(disable_sync=True):
                with tik_instance.for_range(0, op.w_scale) as w_cp_idx:
                    dst_offset = w_cp_idx * op.shape_c0
                    tik_instance.data_move(reorder_buf[dst_offset:], in_image_buf[0:],
                                           0, burst_num, burst_len, 0, burst_dst_gap)

    @abstractmethod
    def adjust_w_per_loop(self) -> None:
        """adjust width loop length as per L1/UB size & data_move ISA limitations"""
        pass

    @abstractmethod
    def init_loop_parameters(self) -> None:
        """init H/NC loop parameters once width loop length is done"""
        pass

    @abstractmethod
    def malloc_buf(self) -> None:
        """malloc buffer for moving in image and reordering"""
        pass

    @abstractmethod
    def pre_image_process(self) -> None:
        """do some prepare before image processing"""
        pass

    @abstractmethod
    def image_process(self) -> None:
        """process/resize images: move in -> reorder in UB -> move out"""
        pass

    def run(self) -> None:
        """main procedure for resizeNN"""
        self.adjust_w_per_loop()
        self.init_loop_parameters()
        self.malloc_buf()
        self.pre_image_process()
        self.image_process()

    def adjust_w_per_loop_for_w_align(self) -> None:
        """adjust w_in_per_loop based on l1/ub buffer size & limitations of data_move ISA"""
        op = self.op
        # if use ub only, number in ub each loop is `(w_scale+1) * nc_per_loop * w_in_per_loop * C0`
        w_scale_factor = op.w_scale + 1
        if op.l1_exists or self.is_w_equal:
            # number in ub each loop is `w_scale * nc_per_loop * w_in_per_loop * C0`
            w_scale_factor = op.w_scale

        """Limits:
        COMMON LIMITS:
          1. [ MUST ] w_in_per_loop <= BURST_LEN_MAX * self.block_num // c0              <-- will be ok if 3 is ok
          2. [SHOULD] w_in_per_loop * w_scale <= BURST_LEN_MAX * self.block_num // c0    <-- will be ok if 3 is ok
          3. [SHOULD] w_in_per_loop * w_scale_factor <= ub_max // c0   <-- will limit by compile parameter when tiling
        if not is_w_equal:
          1. [ MUST ] nc_per_loop * w_in_per_loop <= BURST_NUM_MAX
          2. [SHOULD] w_scale <= BURST_STRIDE_MAX * self.block_num // c0     <-- will be ok if COMMON-3 is ok
        """
        self.w_in_per_loop.set_as(op.w_per_core)
        max_num_in_ub_per_loop = w_scale_factor * self.w_in_per_loop * op.shape_c0
        with self.tik_inst.if_scope(max_num_in_ub_per_loop > op.ub_max_num):
            self.w_in_per_loop.set_as(op.ub_max_num // op.shape_c0 // w_scale_factor)
        if not self.is_w_equal:
            with self.tik_inst.if_scope(self.w_in_per_loop > Constant.BURST_NUM_MAX):
                self.w_in_per_loop.set_as(Constant.BURST_NUM_MAX)

    def malloc_ping_pong_buf(self, tik_instance: tik.Tik, dtype: str,
                             input_num_per_loop, reorder_num_per_loop,
                             is_w_equal: bool, l1_exists: bool) -> None:
        """malloc buffer for data move when image-reorder

        Warning
        ----------
        DO NOT place any new_stmt_scope / if_scope / else_scope of TIK around this function.
        Otherwise, the Tensor malloc here will be retrieved by TIK.

        Parameters
        ----------
        tik_instance: tik instance
        dtype: image dtype
        input_num_per_loop : Scalar or Expr with Scalar
            max count of number moved from GM to L1/UB
        reorder_num_per_loop : Scalar or Expr with Scalar
            max count of reordered which is going to be moved from L1/UB to GM
        is_w_equal: True or False
            whether w_in equals to w_out
        l1_exists: True or False
            whether L1 buffer exists
        """
        def _malloc(name: str, count, buf_scope: str):
            ping = tik_instance.Tensor(dtype, (count,), name='%s_ping' % name, scope=buf_scope)
            pong = tik_instance.Tensor(dtype, (count,), name='%s_pong' % name, scope=buf_scope)
            return ping, pong

        if is_w_equal:
            self.image_in_buf_ping, self.image_in_buf_pong = _malloc(
                "image_in_buf", input_num_per_loop, tik.scope_ubuf)
            self.reorder_buf_ping, self.reorder_buf_pong = None, None
        else:
            self.image_in_buf_ping, self.image_in_buf_pong = _malloc(
                "image_in_buf", input_num_per_loop, tik.scope_cbuf if l1_exists else tik.scope_ubuf)
            self.reorder_buf_ping, self.reorder_buf_pong = _malloc(
                "reorder_buf", reorder_num_per_loop, tik.scope_ubuf)


class AlignHW(ProcessorBase):
    """Both height & width are align, also both align_corners & half_pixel_centers are False

    Attributes:
    ------
        is_w_equal: whether w_in equals w_out
    """

    def __init__(self, resize_op_obj: ResizeNearestNeighbor, is_w_equal: bool = False) -> None:
        super().__init__(resize_op_obj, is_w_equal)

        # scalars for loop
        self.h_per_loop = self.tik_inst.Scalar("int64", name="h_per_loop")
        self.h_loop_num = self.tik_inst.Scalar("int64", name="h_loop_num")
        self.h_loop_tail = self.tik_inst.Scalar("int64", name="h_loop_tail")

    def adjust_w_per_loop(self) -> None:
        """Limits:
        COMMON LIMITS:
          1. [ MUST ] w_in_per_loop <= BURST_LEN_MAX * self.block_num // c0              <-- will be ok if 3 is ok
          2. [SHOULD] w_in_per_loop * w_scale <= BURST_LEN_MAX * self.block_num // c0    <-- will be ok if 3 is ok
          3. [SHOULD] w_in_per_loop * w_scale_factor <= ub_max // c0   <-- will limit by compile parameter when tiling
          4. [ MUST ] W-in <= BURST_LEN_MAX * self.block_num // c0           <-- will limit when tiling
          5. [ MUST ] h_scale * W-out <= BURST_STRIDE_MAX * self.block_num // c0   <-- will limit when tiling
        if not is_w_equal:
          1. [ MUST ] h_in_per_loop * w_in_per_loop <= BURST_NUM_MAX
          2. [SHOULD] w_scale <= BURST_STRIDE_MAX * self.block_num // c0     <-- will be ok if COMMON-3 is ok
        """
        self.adjust_w_per_loop_for_w_align()

    def init_loop_parameters(self) -> None:
        op = self.op

        self.w_out_per_loop.set_as(self.w_in_per_loop * op.w_scale)

        self.w_loop_num.set_as(op.w_per_core // self.w_in_per_loop)
        self.w_loop_tail.set_as(op.w_per_core % self.w_in_per_loop)

        if op.l1_exists or self.is_w_equal:
            self.h_per_loop.set_as(op.ub_max_num // op.shape_c0 // self.w_out_per_loop)
        else:
            self.h_per_loop.set_as(op.ub_max_num // op.shape_c0 // (self.w_in_per_loop + self.w_out_per_loop))

        """considering for data_move ISA:
        1. h_per_loop <= BURST_NUM_MAX
        2. if not is_w_equal: h_per_loop * w_in_per_loop <= BURST_NUM_MAX
        """
        if self.is_w_equal:
            with self.tik_inst.if_scope(self.h_per_loop > Constant.BURST_NUM_MAX):
                self.h_per_loop.set_as(Constant.BURST_NUM_MAX)
        else:
            w_max = self.tik_inst.Scalar("int64", "w_max", init_value=self.w_in_per_loop)
            with self.tik_inst.if_scope(self.w_loop_num == 0):
                w_max.set_as(self.w_loop_tail)
            with self.tik_inst.if_scope(self.h_per_loop * w_max > Constant.BURST_NUM_MAX):
                self.h_per_loop.set_as(Constant.BURST_NUM_MAX // w_max)

        self.h_loop_num.set_as(op.h_per_core // self.h_per_loop)
        self.h_loop_tail.set_as(op.h_per_core % self.h_per_loop)

    def malloc_buf(self) -> None:
        _h_max = self.tik_inst.Scalar("int64", "h_max", init_value=self.h_per_loop)
        with self.tik_inst.if_scope(self.h_loop_num == 0):
            _h_max.set_as(self.h_loop_tail)

        op = self.op
        in_num_per_loop = _h_max * self.w_in_per_loop * op.shape_c0
        reorder_num_per_loop = in_num_per_loop * op.w_scale
        self.malloc_ping_pong_buf(self.tik_inst, op.images_dtype,
                                  in_num_per_loop, reorder_num_per_loop, self.is_w_equal, op.l1_exists)

    def pre_image_process(self) -> None: pass

    def image_process(self) -> None:
        with self.tik_inst.if_scope(self.h_loop_num > 1):
            self._h_ping_pong()
        with self.tik_inst.else_scope():
            self._nc_ping_pong()

    def _h_ping_pong(self) -> None:
        op = self.op
        nc_start = self.tik_inst.Scalar("int64", "nc_start")

        def _invoke_ping_pong(w_in_len):
            self.gm_in_start.set_as((nc_start * op.input_height + op.core_height_start)
                                    * op.input_width + w_in_start)
            self.gm_out_start.set_as((nc_start * op.input_height + op.core_height_start) * op.h_scale
                                     * op.output_width + w_in_start * op.w_scale)

            def _do_reorder(h_loop_idx, h_in_len, in_image_buf, reorder_buf):
                gm_in_offset = op.shape_c0 * (self.gm_in_start + h_loop_idx
                                              * self.h_per_loop * op.input_width)
                gm_out_offset = op.shape_c0 * (self.gm_out_start + h_loop_idx
                                               * self.h_per_loop * op.h_scale * op.output_width)
                self._image_reorder(gm_in_offset, gm_out_offset, h_in_len, w_in_len, in_image_buf, reorder_buf)

            with self.tik_inst.for_range(0, self.h_loop_num >> 1) as _h_loop_idx:
                _do_reorder(_h_loop_idx * 2, self.h_per_loop, self.image_in_buf_ping, self.reorder_buf_ping)
                _do_reorder(_h_loop_idx * 2 + 1, self.h_per_loop, self.image_in_buf_pong, self.reorder_buf_pong)
            with self.tik_inst.if_scope(self.h_loop_num % 2 == 1):
                _do_reorder(self.h_loop_num - 1, self.h_per_loop, self.image_in_buf_ping, self.reorder_buf_ping)
            with self.tik_inst.if_scope(self.h_loop_tail > 0):
                _do_reorder(self.h_loop_num, self.h_loop_tail, self.image_in_buf_ping, self.reorder_buf_ping)

        with self.tik_inst.for_range(0, op.nc_per_core) as _nc_idx:
            nc_start.set_as(op.core_nc_start + _nc_idx)
            with self.tik_inst.for_range(0, self.w_loop_num) as _w_loop_idx:
                w_in_start = op.core_width_start + _w_loop_idx * self.w_in_per_loop
                _invoke_ping_pong(self.w_in_per_loop)
            with self.tik_inst.if_scope(self.w_loop_tail > 0):
                w_in_start = op.core_width_start + self.w_loop_num * self.w_in_per_loop
                _invoke_ping_pong(self.w_loop_tail)

    def _nc_ping_pong(self) -> None:
        op = self.op

        def _w_loop_of_nc_ping_pong(h_in_len):
            def _invoke_ping_pong(w_in_len):
                self.gm_in_start.set_as((op.core_nc_start * op.input_height + h_in_start)
                                        * op.input_width + w_in_start)
                self.gm_out_start.set_as((op.core_nc_start * op.input_height + h_in_start) * op.h_scale
                                         * op.output_width + w_in_start * op.w_scale)

                def _do_reorder(nc_idx, in_image_buf, reorder_buf):
                    gm_in_offset = (self.gm_in_start + nc_idx
                                    * op.input_height * op.input_width) * op.shape_c0
                    gm_out_offset = (self.gm_out_start + nc_idx
                                     * op.output_height * op.output_width) * op.shape_c0
                    self._image_reorder(gm_in_offset, gm_out_offset, h_in_len, w_in_len, in_image_buf, reorder_buf)

                with self.tik_inst.for_range(0, op.nc_per_core >> 1) as _nc_idx:
                    _do_reorder(_nc_idx * 2, self.image_in_buf_ping, self.reorder_buf_ping)
                    _do_reorder(_nc_idx * 2 + 1, self.image_in_buf_pong, self.reorder_buf_pong)
                with self.tik_inst.if_scope(op.nc_per_core % 2 == 1):
                    _do_reorder(op.nc_per_core - 1, self.image_in_buf_ping, self.reorder_buf_ping)

            with self.tik_inst.for_range(0, self.w_loop_num) as _w_loop_idx:
                w_in_start = op.core_width_start + _w_loop_idx * self.w_in_per_loop
                _invoke_ping_pong(self.w_in_per_loop)
            with self.tik_inst.if_scope(self.w_loop_tail > 0):
                w_in_start = op.core_width_start + self.w_loop_num * self.w_in_per_loop
                _invoke_ping_pong(self.w_loop_tail)

        with self.tik_inst.if_scope(self.h_loop_num == 1):
            h_in_start = op.core_height_start
            _w_loop_of_nc_ping_pong(self.h_per_loop)
        with self.tik_inst.if_scope(self.h_loop_tail > 0):
            h_in_start = op.core_height_start + self.h_loop_num * self.h_per_loop
            _w_loop_of_nc_ping_pong(self.h_loop_tail)

    def _image_reorder(self, gm_in_offset, gm_out_offset, h_in_len, w_in_len, in_image_buf, reorder_buf) -> None:
        self._move_from_gm(gm_in_offset, in_image_buf, h_in_len, w_in_len)
        if self.is_w_equal:
            self._move_to_gm(gm_out_offset, in_image_buf, h_in_len, w_in_len)
        else:
            self.reorder_in_ub_for_w_align(self.tik_inst, self.op, in_image_buf, reorder_buf, h_in_len * w_in_len)
            self._move_to_gm(gm_out_offset, reorder_buf, h_in_len, w_in_len)

    def _move_from_gm(self, gm_in_offset, in_image_buf, h_in_len, w_in_len) -> None:
        """ GM -> L1/UB """
        op = self.op
        burst_num = h_in_len
        burst_len = w_in_len * op.c0_blocks
        burst_src_gap = op.input_width * op.c0_blocks - burst_len
        self.tik_inst.data_move(in_image_buf[0:], op.images_gm[gm_in_offset:],
                                0, burst_num, burst_len, burst_src_gap, 0)

    def _move_to_gm(self, gm_out_offset, reorder_buf, h_in_len, w_in_len) -> None:
        """ UB -> GM """
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            op = self.op
            burst_num = h_in_len
            burst_len = w_in_len * op.w_scale * op.c0_blocks
            burst_dst_gap = op.h_scale * op.output_width * op.c0_blocks - burst_len
            with self.tik_inst.for_range(0, op.h_scale) as _h_cp_idx:
                dst_offset = gm_out_offset + _h_cp_idx * op.output_width * op.shape_c0
                self.tik_inst.data_move(op.out_gm[dst_offset:], reorder_buf[0:],
                                        0, burst_num, burst_len, 0, burst_dst_gap)


class AlignHWSmall(ProcessorBase):
    """Scenario for particular AlignHW with large NC1 but small H/W.
    H/W will not be cut between COREs, but NC1 only.
    Also, H/W will not be cut within each CORE.

    Attributes:
    ------
        is_w_equal: whether w_in equals w_out
    """
    def __init__(self, resize_op_obj: ResizeNearestNeighbor, is_w_equal: bool = False) -> None:
        super().__init__(resize_op_obj, is_w_equal)

        # scalars for loop
        self.nc_per_loop = self.tik_inst.Scalar("int64", name="nc_per_loop")
        self.nc_loop_num = self.tik_inst.Scalar("int64", name="nc_loop_num")
        self.nc_loop_tail = self.tik_inst.Scalar("int64", name="nc_loop_tail")

    def adjust_w_per_loop(self) -> None: pass

    def init_loop_parameters(self) -> None:
        """Limits:

        COMMON LIMITS:
          1. [ MUST ] nc_per_loop <= BURST_NUM_MAX
          2. [ MUST ] w_per_core <= BURST_LEN_MAX * self.block_num // c0     <-- will be ok if COMMON-4 is ok
          3. [ MUST ] H-out * W-out <= BURST_STRIDE_MAX * self.block_num // c0  <-- will limit when tiling
          4. [SHOULD] w_per_core * w_scale <= BURST_LEN_MAX * self.block_num // c0  <-- will limit when tiling (128)
          5. [SHOULD] nc_per_loop * w_per_core * w_scale <= ub_max // c0

        if not is_w_equal:
          1. [ MUST ] nc_per_loop * w_per_core <= BURST_NUM_MAX
          2. [SHOULD] w_scale <= BURST_STRIDE_MAX * self.block_num // c0     <-- will be ok if COMMON-4 is ok
        """
        op = self.op

        if op.l1_exists or self.is_w_equal:
            self.nc_per_loop.set_as(op.ub_max_num // op.shape_c0 // (op.w_per_core * op.w_scale))
        else:
            self.nc_per_loop.set_as(op.ub_max_num // op.shape_c0 // (op.w_per_core * (1 + op.w_scale)))

        """considering for data_move ISA"""
        if self.is_w_equal:
            with self.tik_inst.if_scope(self.nc_per_loop > Constant.BURST_NUM_MAX):
                self.nc_per_loop.set_as(Constant.BURST_NUM_MAX)
        else:
            with self.tik_inst.if_scope(self.nc_per_loop * op.w_per_core > Constant.BURST_NUM_MAX):
                self.nc_per_loop.set_as(Constant.BURST_NUM_MAX // op.w_per_core)

        self.nc_loop_num.set_as(op.nc_per_core // self.nc_per_loop)
        self.nc_loop_tail.set_as(op.nc_per_core % self.nc_per_loop)

    def malloc_buf(self) -> None:
        op = self.op
        _nc_max = self.tik_inst.Scalar("int64", "nc_max", init_value=self.nc_per_loop)
        with self.tik_inst.if_scope(self.nc_loop_num == 0):
            _nc_max.set_as(self.nc_loop_tail)

        in_num_per_loop = _nc_max * op.w_per_core * op.shape_c0
        reorder_num_per_loop = in_num_per_loop * op.w_scale
        self.malloc_ping_pong_buf(self.tik_inst, op.images_dtype,
                                  in_num_per_loop, reorder_num_per_loop, self.is_w_equal, op.l1_exists)

    def pre_image_process(self) -> None: pass

    def image_process(self) -> None:
        self._h_ping_pong()

    def _h_ping_pong(self) -> None:
        op = self.op

        def _invoke_ping_pong(nc_len):
            self.gm_in_start.set_as((nc_start * op.input_height + op.core_height_start)
                                    * op.input_width + op.core_width_start)
            self.gm_out_start.set_as((nc_start * op.input_height + op.core_height_start) * op.h_scale
                                     * op.output_width + op.core_width_start * op.w_scale)

            def _do_reorder(h_idx, in_image_buf, reorder_buf):
                gm_in_offset = op.shape_c0 * (self.gm_in_start + h_idx * op.input_width)
                gm_out_offset = op.shape_c0 * (self.gm_out_start + h_idx * op.h_scale * op.output_width)
                self._image_reorder(gm_in_offset, gm_out_offset, nc_len, in_image_buf, reorder_buf)

            with self.tik_inst.for_range(0, op.h_per_core >> 1) as _h_idx:
                _do_reorder(_h_idx * 2, self.image_in_buf_ping, self.reorder_buf_ping)
                _do_reorder(_h_idx * 2 + 1, self.image_in_buf_pong, self.reorder_buf_pong)
            with self.tik_inst.if_scope(op.h_per_core % 2 == 1):
                _do_reorder(op.h_per_core - 1, self.image_in_buf_ping, self.reorder_buf_ping)

        with self.tik_inst.for_range(0, self.nc_loop_num) as _nc_loop_idx:
            nc_start = op.core_nc_start + _nc_loop_idx * self.nc_per_loop
            _invoke_ping_pong(self.nc_per_loop)
        with self.tik_inst.if_scope(self.nc_loop_tail > 0):
            nc_start = op.core_nc_start + self.nc_loop_num * self.nc_per_loop
            _invoke_ping_pong(self.nc_loop_tail)

    def _image_reorder(self, gm_in_offset, gm_out_offset, nc_len, in_image_buf, reorder_buf) -> None:
        self._move_from_gm(gm_in_offset, in_image_buf, nc_len)
        if self.is_w_equal:
            self._move_to_gm(gm_out_offset, in_image_buf, nc_len)
        else:
            self.reorder_in_ub_for_w_align(self.tik_inst, self.op, in_image_buf, reorder_buf,
                                           nc_len * self.op.w_per_core)
            self._move_to_gm(gm_out_offset, reorder_buf, nc_len)

    def _move_from_gm(self, gm_in_offset, in_image_buf, nc_len) -> None:
        """ GM -> L1/UB """
        op = self.op
        burst_num = nc_len
        burst_len = op.w_per_core * op.c0_blocks
        burst_src_gap = op.input_height * op.input_width * op.c0_blocks - burst_len
        self.tik_inst.data_move(in_image_buf[0:], op.images_gm[gm_in_offset:],
                                0, burst_num, burst_len, burst_src_gap, 0)

    def _move_to_gm(self, gm_out_offset, reorder_buf, nc_len) -> None:
        """ UB -> GM """
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            op = self.op
            burst_num = nc_len
            burst_len = op.w_per_core * op.w_scale * op.c0_blocks
            burst_dst_gap = op.output_height * op.output_width * op.c0_blocks - burst_len
            with self.tik_inst.for_range(0, op.h_scale) as _h_cp_idx:
                dst_offset = gm_out_offset + _h_cp_idx * op.output_width * op.shape_c0
                self.tik_inst.data_move(op.out_gm[dst_offset:], reorder_buf[0:],
                                        0, burst_num, burst_len, 0, burst_dst_gap)


class AlignOnlyW(ProcessorBase):
    """Only width align, both align_corners & half_pixel_centers are False

    Attributes:
    ------
        is_w_equal: whether w_in equals w_out
    """

    def __init__(self, resize_op_obj: ResizeNearestNeighbor, is_w_equal: bool = False) -> None:
        super().__init__(resize_op_obj, is_w_equal)

        # scalars for loop
        self.nc_per_loop = self.tik_inst.Scalar("int64", name="nc_per_loop")
        self.nc_loop_num = self.tik_inst.Scalar("int64", name="nc_loop_num")
        self.nc_loop_tail = self.tik_inst.Scalar("int64", name="nc_loop_tail")

        # batch count for index calculation
        self.height_idx_per_loop = 64

        # parameters to decide how to data_move from / to GM.
        self.image_in_exceed_stride = self.tik_inst.Scalar("int8", name="image_in_exceed_stride", init_value=0)
        self.image_out_exceed_stride = self.tik_inst.Scalar("int8", name="image_out_exceed_stride", init_value=0)

        # assistant scalar
        self.h_out_idx = self.tik_inst.Scalar("int32", "h_out_idx")
        self.h_in_idx = self.tik_inst.Scalar("int32", "h_in_idx")

        # prepare to batch calculate height-in index
        self.h_in_idx_ub = self.tik_inst.Tensor("int32", (self.height_idx_per_loop,),
                                                name="h_in_idx_ub", scope=tik.scope_ubuf)
        self.h_out_idx_ub_fpx = self.tik_inst.Tensor(self.op.compute_dtype, (self.height_idx_per_loop,),
                                                     name="h_out_idx_ub_fpx", scope=tik.scope_ubuf)
        # fill in 0,1,2,3,4,...,63 into h_out_idx_ub_fp32
        # NOTE: DO NOT move function below after malloc_buf because of some unknown issues...
        fill_index_in_ub(self.tik_inst, self.h_out_idx_ub_fpx, self.height_idx_per_loop)

    def adjust_w_per_loop(self) -> None:
        """Limits:
        COMMON LIMITS:
          1. [ MUST ] w_in_per_loop <= BURST_LEN_MAX * self.block_num // c0              <-- will be ok if 3 is ok
          2. [SHOULD] w_in_per_loop * w_scale <= BURST_LEN_MAX * self.block_num // c0    <-- will be ok if 3 is ok
          3. [SHOULD] w_in_per_loop * w_scale_factor <= ub_max // c0   <-- will limit by compile parameter when tiling
        if not is_w_equal:
          1. [ MUST ] nc_per_loop * w_in_per_loop <= BURST_NUM_MAX
          2. [SHOULD] w_scale <= BURST_STRIDE_MAX * self.block_num // c0     <-- will be ok if COMMON-3 is ok
        """
        self.adjust_w_per_loop_for_w_align()

    def init_loop_parameters(self) -> None:
        op = self.op

        self.w_out_per_loop.set_as(self.w_in_per_loop * op.w_scale)

        self.w_loop_num.set_as(op.w_per_core // self.w_in_per_loop)
        self.w_loop_tail.set_as(op.w_per_core % self.w_in_per_loop)

        if op.l1_exists or self.is_w_equal:
            self.nc_per_loop.set_as(op.ub_max_num // op.shape_c0 // self.w_out_per_loop)
        else:
            self.nc_per_loop.set_as(op.ub_max_num // op.shape_c0 // (self.w_in_per_loop + self.w_out_per_loop))

        """considering for data_move ISA:
        1. nc_per_loop <= BURST_NUM_MAX
        2. if not is_w_equal: nc_per_loop * w_in_per_loop <= BURST_NUM_MAX
        """
        if self.is_w_equal:
            with self.tik_inst.if_scope(self.nc_per_loop > Constant.BURST_NUM_MAX):
                self.nc_per_loop.set_as(Constant.BURST_NUM_MAX)
        else:
            w_max = self.tik_inst.Scalar("int64", "w_max", init_value=self.w_in_per_loop)
            with self.tik_inst.if_scope(self.w_loop_num == 0):
                w_max.set_as(self.w_loop_tail)
            with self.tik_inst.if_scope(self.nc_per_loop * w_max > Constant.BURST_NUM_MAX):
                self.nc_per_loop.set_as(Constant.BURST_NUM_MAX // w_max)

        self.nc_loop_num.set_as(op.nc_per_core // self.nc_per_loop)
        self.nc_loop_tail.set_as(op.nc_per_core % self.nc_per_loop)

    def malloc_buf(self) -> None:
        _nc_max = self.tik_inst.Scalar("int64", "nc_max", init_value=self.nc_per_loop)
        with self.tik_inst.if_scope(self.nc_loop_num == 0):
            _nc_max.set_as(self.nc_loop_tail)

        op = self.op
        in_num_per_loop = _nc_max * self.w_in_per_loop * op.shape_c0
        reorder_num_per_loop = in_num_per_loop * op.w_scale
        self.malloc_ping_pong_buf(self.tik_inst, op.images_dtype,
                                  in_num_per_loop, reorder_num_per_loop, self.is_w_equal, op.l1_exists)

    def pre_image_process(self) -> None:
        op = self.op
        with self.tik_inst.if_scope(op.input_height * op.input_width * op.c0_blocks > Constant.BURST_STRIDE_MAX):
            self.image_in_exceed_stride.set_as(Constant.EXCEED_BURST_STRIDE)
        with self.tik_inst.if_scope(op.output_height * op.output_width * op.c0_blocks > Constant.BURST_STRIDE_MAX):
            self.image_out_exceed_stride.set_as(Constant.EXCEED_BURST_STRIDE)

    def image_process(self) -> None:
        with self.tik_inst.new_stmt_scope():
            op = self.op

            scalar_idx_fpx = self.tik_inst.Scalar(op.compute_dtype, name="scalar_idx_fpx")
            scalar_vconv_int_to_fp(self.tik_inst, op.core_height_start, scalar_idx_fpx)
            # do vadds 0,1,2,3,4,...,63 + core_height_start
            self.tik_inst.vadds(64, self.h_out_idx_ub_fpx, self.h_out_idx_ub_fpx, scalar_idx_fpx,
                                (self.height_idx_per_loop + 63) // 64, 1, 1, op.block_strides, op.block_strides)
            # convert height_idx_per_loop for the loop below
            scalar_vconv_int_to_fp(self.tik_inst, self.height_idx_per_loop, scalar_idx_fpx)

            with self.tik_inst.for_range(0, op.h_per_core) as _h_idx:
                # batch calculate height-in index once per 64
                with self.tik_inst.if_scope((_h_idx % self.height_idx_per_loop) == 0):
                    self.calculate_src_idx(self.tik_inst, op.h_rec_scale_fpx, self.h_in_idx_ub,
                                           self.h_out_idx_ub_fpx, self.height_idx_per_loop,
                                           op.align_corners, op.half_pixel_centers)
                    self.tik_inst.vadds(64, self.h_out_idx_ub_fpx, self.h_out_idx_ub_fpx, scalar_idx_fpx,
                                        (self.height_idx_per_loop + 63) // 64, 1, 1, op.block_strides, op.block_strides)

                self.h_out_idx.set_as(op.core_height_start + _h_idx)
                self.h_in_idx.set_as(self.h_in_idx_ub[_h_idx % self.height_idx_per_loop])
                self._nc_ping_pong()

    def _nc_ping_pong(self) -> None:
        op = self.op

        def _invoke_ping_pong(w_in_len):
            with self.tik_inst.if_scope(self.h_in_idx > op.input_height - 1):
                self.h_in_idx.set_as(op.input_height - 1)
            self.gm_in_start.set_as((op.core_nc_start * op.input_height + self.h_in_idx)
                                    * op.input_width + w_in_start)
            self.gm_out_start.set_as((op.core_nc_start * op.output_height + self.h_out_idx)
                                     * op.output_width + w_in_start * op.w_scale)

            def _do_reorder(nc_loop_idx, nc_len, in_image_buf, reorder_buf):
                gm_in_offset = (self.gm_in_start + nc_loop_idx * self.nc_per_loop
                                * op.input_height * op.input_width) * op.shape_c0
                gm_out_offset = (self.gm_out_start + nc_loop_idx * self.nc_per_loop
                                 * op.output_height * op.output_width) * op.shape_c0
                self._image_reorder(gm_in_offset, gm_out_offset, nc_len, w_in_len, in_image_buf, reorder_buf)

            with self.tik_inst.for_range(0, self.nc_loop_num >> 1) as _nc_loop_idx:
                _do_reorder(_nc_loop_idx * 2, self.nc_per_loop, self.image_in_buf_ping, self.reorder_buf_ping)
                _do_reorder(_nc_loop_idx * 2 + 1, self.nc_per_loop, self.image_in_buf_pong, self.reorder_buf_pong)
            with self.tik_inst.if_scope(self.nc_loop_num % 2 == 1):
                _do_reorder(self.nc_loop_num - 1, self.nc_per_loop, self.image_in_buf_ping, self.reorder_buf_ping)
            with self.tik_inst.if_scope(self.nc_loop_tail > 0):
                _do_reorder(self.nc_loop_num, self.nc_loop_tail, self.image_in_buf_ping, self.reorder_buf_ping)

        with self.tik_inst.for_range(0, self.w_loop_num) as _w_loop_idx:
            w_in_start = op.core_width_start + _w_loop_idx * self.w_in_per_loop
            _invoke_ping_pong(self.w_in_per_loop)
        with self.tik_inst.if_scope(self.w_loop_tail > 0):
            w_in_start = op.core_width_start + self.w_loop_num * self.w_in_per_loop
            _invoke_ping_pong(self.w_loop_tail)

    def _image_reorder(self, gm_in_offset, gm_out_offset, nc_len, w_in_len, in_image_buf, reorder_buf) -> None:
        self.nc_batch_move_from_gm(self.tik_inst, self.op, gm_in_offset, in_image_buf,
                                   nc_len, w_in_len, self.image_in_exceed_stride)
        if self.is_w_equal:
            self.nc_batch_move_to_gm(self.tik_inst, self.op, gm_out_offset, in_image_buf,
                                     nc_len, w_in_len * self.op.w_scale, self.image_out_exceed_stride)
        else:
            self.reorder_in_ub_for_w_align(self.tik_inst, self.op, in_image_buf, reorder_buf, nc_len * w_in_len)
            self.nc_batch_move_to_gm(self.tik_inst, self.op, gm_out_offset, reorder_buf,
                                     nc_len, w_in_len * self.op.w_scale, self.image_out_exceed_stride)


class Default(ProcessorBase):
    """ Default process class """
    def __init__(self, resize_op_obj: ResizeNearestNeighbor) -> None:
        super().__init__(resize_op_obj)

        # scalars for loop
        self.nc_per_loop = self.tik_inst.Scalar("int64", name="nc_per_loop")
        self.nc_loop_num = self.tik_inst.Scalar("int64", name="nc_loop_num")
        self.nc_loop_tail = self.tik_inst.Scalar("int64", name="nc_loop_tail")

        # batch count for index calculation
        self.width_idx_per_loop = 128

        # parameters to decide how to data_move from / to GM.
        self.image_in_exceed_stride = self.tik_inst.Scalar("int8", name="image_in_exceed_stride", init_value=0)
        self.image_out_exceed_stride = self.tik_inst.Scalar("int8", name="image_out_exceed_stride", init_value=0)

        # assistant scalars
        self.w_in_start_idx = self.tik_inst.Scalar("int32", name="w_in_start_idx")
        self.w_in_end_idx = self.tik_inst.Scalar("int32", name="w_in_end_idx")
        self.w_in_len = self.tik_inst.Scalar("int32", name="w_in_len")
        self.w_out_start = self.tik_inst.Scalar("int32", name="w_out_start")
        self.nc_start = self.tik_inst.Scalar("int32", name="nc_start")

        # width index array for batch calculation
        self.w_in_idx_array = self.tik_inst.Tensor("int32", (self.width_idx_per_loop,),
                                                   name="w_in_idx_array", scope=tik.scope_ubuf)
        self.w_out_idx_ub_fpx = self.tik_inst.Tensor(self.op.compute_dtype, (self.width_idx_per_loop,),
                                                     name="w_out_idx_ub_fpx", scope=tik.scope_ubuf)
        self.scalar_idx_fpx = self.tik_inst.Scalar(self.op.compute_dtype, name="scalar_idx_fpx")

        # fill 0-127 into fp32 index array
        # NOTE: DO NOT move function below after malloc_buf because of some unknown issues...
        fill_index_in_ub(self.tik_inst, self.w_out_idx_ub_fpx, self.width_idx_per_loop)

    def adjust_w_per_loop(self) -> None:
        op = self.op

        self.w_out_per_loop.set_as(self.width_idx_per_loop)
        with self.tik_inst.if_scope(op.w_per_core < self.w_out_per_loop):
            self.w_out_per_loop.set_as(op.w_per_core)

        w_out_per_loop_max = self.tik_inst.Scalar("int64", "w_out_per_loop_max", init_value=self.w_out_per_loop)
        with self.tik_inst.if_scope(op.output_width >= op.input_width):
            # zoom in
            if not op.l1_exists:
                """Limit:
                w_out_per_loop + w_in_per_loop <= ub_max // c0
                # w_out_per_loop + w_out_per_loop/w_scale <= ub_max // c0
                # w_out_per_loop <= ub_max // c0 / (1 + 1 / w_scale)
                # w_out_per_loop <= ub_max // c0 / ((input_width + output_width) / output_width)
                                 < ub_max // c0 / ((input_width + output_width - 2) / (output_width - 1))
                So: 128 (default max width per loop) < 0.5 * (ub_max // c0) < w_out_per_loop_max < (ub_max // c0)
                """
                w_out_per_loop_max.set_as(((op.ub_max_num // op.shape_c0) * op.output_width)
                                          // (op.input_width + op.output_width))
        with self.tik_inst.else_scope():
            # zoom out
            if op.l1_exists:
                """Limits:
                1. w_in_per_loop  <= l1_max_num // c0 - 1 , minus 1 is to protect nc_per_loop calculation
                2. w_in_per_loop  <= BURST_STRIDE_MAX * self.block_num // c0
                3. w_out_per_loop <= ub_max_num // c0
                """
                w_in_limit_min = min(op.l1_max_num // op.shape_c0 - 1,
                                     Constant.BURST_STRIDE_MAX * op.block_num // op.shape_c0)
                if not op.align_corners:
                    # likely
                    w_out_per_loop_max.set_as((w_in_limit_min * op.output_width) // op.input_width)
                else:
                    with self.tik_inst.if_scope(op.output_width > 1):
                        # likely
                        w_out_per_loop_max.set_as((w_in_limit_min * (op.output_width - 1)) // (op.input_width - 1))
                    with self.tik_inst.else_scope():  # op.output_width == 1
                        w_out_per_loop_max.set_as(w_in_limit_min // op.input_width)
                with self.tik_inst.if_scope(w_out_per_loop_max > op.ub_max_num // op.shape_c0):
                    w_out_per_loop_max.set_as(op.ub_max_num // op.shape_c0)
            else:  # not op.l1_exists
                """Limit:
                w_out_per_loop + w_in_per_loop <= ub_max // c0
                # w_out_per_loop + w_out_per_loop/w_scale <= ub_max // c0
                # w_out_per_loop <= ub_max // c0 / (1 + 1 / w_scale)
                # w_out_per_loop <= ub_max // c0 / ((input_width + output_width - 2) / (output_width - 1)
                                 < ub_max // c0 / ((input_width + output_width) / output_width)
                So:  0 < w_out_per_loop_max < 0.5 * (ub_max // c0) < w_in_per_loop_max < ub_max // c0
                Also: w_in_per_loop_max < ub_max // c0 < BURST_STRIDE_MAX * self.block_num // c0
                """
                w_in_limit = op.ub_max_num // op.shape_c0
                if not op.align_corners:
                    # likely
                    w_out_per_loop_max.set_as(w_in_limit * op.output_width // (op.input_width + op.output_width))
                else:
                    with self.tik_inst.if_scope(op.output_width > 1):
                        w_out_per_loop_max.set_as(w_in_limit * (op.output_width - 1)
                                                  // (op.input_width + op.output_width - 2))
                    with self.tik_inst.else_scope():  # op.output_width == 1
                        w_out_per_loop_max.set_as(w_in_limit // (op.input_width + 1))

        with self.tik_inst.if_scope(self.w_out_per_loop > w_out_per_loop_max):
            self.w_out_per_loop.set_as(w_out_per_loop_max)
        with self.tik_inst.if_scope(self.w_out_per_loop < 1):
            self.w_out_per_loop.set_as(1)

    def init_loop_parameters(self) -> None:
        op = self.op

        # attribute `half_pixel_centers` has already been considered
        if not op.align_corners:
            self.w_in_per_loop.set_as(1 +
                                      util_tik_comm_func.ceil_div((self.w_out_per_loop - 1) * op.input_width,
                                                                  op.output_width))
        else:
            with self.tik_inst.if_scope(op.output_width > 1):
                self.w_in_per_loop.set_as(1 +
                                          util_tik_comm_func.ceil_div((self.w_out_per_loop - 1) * (op.input_width - 1),
                                                                      (op.output_width - 1)))
            with self.tik_inst.else_scope():
                self.w_in_per_loop.set_as(1)

        self.w_loop_num.set_as(op.w_per_core // self.w_out_per_loop)
        self.w_loop_tail.set_as(op.w_per_core % self.w_out_per_loop)

        if op.l1_exists:
            self.nc_per_loop.set_as(op.ub_max_num // op.shape_c0 // self.w_out_per_loop)
            with self.tik_inst.if_scope(self.nc_per_loop * self.w_in_per_loop > op.l1_max_num // op.shape_c0):
                self.nc_per_loop.set_as(op.l1_max_num // op.shape_c0 // self.w_in_per_loop)
        else:
            self.nc_per_loop.set_as(op.ub_max_num // op.shape_c0 // (self.w_in_per_loop + self.w_out_per_loop))

        # considering for data_move ISA
        with self.tik_inst.if_scope(self.nc_per_loop > Constant.BURST_NUM_MAX):
            self.nc_per_loop.set_as(Constant.BURST_NUM_MAX)

        self.nc_loop_num.set_as(op.nc_per_core // self.nc_per_loop)
        self.nc_loop_tail.set_as(op.nc_per_core % self.nc_per_loop)

    def malloc_buf(self) -> None:
        op = self.op
        _nc_max = self.tik_inst.Scalar("int64", "nc_max", init_value=self.nc_per_loop)
        with self.tik_inst.if_scope(self.nc_loop_num == 0):
            _nc_max.set_as(self.nc_loop_tail)

        in_num_per_loop = _nc_max * self.w_in_per_loop * op.shape_c0
        reorder_num_per_loop = _nc_max * self.w_out_per_loop * op.shape_c0
        self.malloc_ping_pong_buf(self.tik_inst, op.images_dtype,
                                  in_num_per_loop, reorder_num_per_loop, False, op.l1_exists)

    def pre_image_process(self) -> None:
        op = self.op
        with self.tik_inst.if_scope(op.input_height * op.input_width * op.c0_blocks > Constant.BURST_STRIDE_MAX):
            self.image_in_exceed_stride.set_as(Constant.EXCEED_BURST_STRIDE)
        with self.tik_inst.if_scope(op.output_height * op.output_width * op.c0_blocks > Constant.BURST_STRIDE_MAX):
            self.image_out_exceed_stride.set_as(Constant.EXCEED_BURST_STRIDE)

    def image_process(self) -> None:
        op = self.op

        # vconv start idx from int32 scalar to fp32 scalar
        scalar_vconv_int_to_fp(self.tik_inst, op.core_width_start, self.scalar_idx_fpx)
        # do vadds 0,1,2,3,4 + fp32_scalar
        self.tik_inst.vadds(64, self.w_out_idx_ub_fpx, self.w_out_idx_ub_fpx, self.scalar_idx_fpx,
                            (self.w_out_per_loop + 63) // 64, 1, 1, op.block_strides, op.block_strides)
        scalar_vconv_int_to_fp(self.tik_inst, self.w_out_per_loop, self.scalar_idx_fpx)

        with self.tik_inst.for_range(0, self.w_loop_num) as _w_loop_idx:
            self.w_out_start.set_as(op.core_width_start + _w_loop_idx * self.w_out_per_loop)
            self._do_one_w_loop(self.w_out_per_loop)
        with self.tik_inst.if_scope(self.w_loop_tail > 0):
            self.w_out_start.set_as(op.core_width_start + self.w_loop_num * self.w_out_per_loop)
            self._do_one_w_loop(self.w_loop_tail)

    def _do_one_w_loop(self, w_out_len) -> None:
        op = self.op
        # batch calculate src width index
        self.calculate_src_idx(self.tik_inst, op.w_rec_scale_fpx,
                               self.w_in_idx_array, self.w_out_idx_ub_fpx, self.width_idx_per_loop,
                               op.align_corners, op.half_pixel_centers)
        self.tik_inst.vadds(64, self.w_out_idx_ub_fpx, self.w_out_idx_ub_fpx, self.scalar_idx_fpx,
                            (self.width_idx_per_loop + 63) // 64, 1, 1, op.block_strides, op.block_strides)
        self.w_in_start_idx.set_as(self.w_in_idx_array[0])
        self.w_in_end_idx.set_as(self.w_in_idx_array[w_out_len-1])
        with self.tik_inst.if_scope(self.w_in_start_idx > op.input_width - 1):
            self.w_in_start_idx.set_as(op.input_width - 1)
        with self.tik_inst.if_scope(self.w_in_end_idx > op.input_width - 1):
            self.w_in_end_idx.set_as(op.input_width - 1)
        self.w_in_len.set_as(self.w_in_end_idx - self.w_in_start_idx + 1)

        with self.tik_inst.for_range(0, self.nc_loop_num) as _nc_loop_idx:
            self.nc_start.set_as(op.core_nc_start + _nc_loop_idx * self.nc_per_loop)
            self._do_one_nc_loop(self.nc_per_loop, self.w_in_len, w_out_len)
        with self.tik_inst.if_scope(self.nc_loop_tail > 0):
            self.nc_start.set_as(op.core_nc_start + self.nc_loop_num * self.nc_per_loop)
            self._do_one_nc_loop(self.nc_loop_tail, self.w_in_len, w_out_len)

    def _do_one_nc_loop(self, nc_len, w_in_len, w_out_len) -> None:
        op = self.op

        self.gm_in_start.set_as(self.nc_start * op.input_height * op.input_width + self.w_in_start_idx)
        self.gm_out_start.set_as(self.nc_start * op.output_height * op.output_width + self.w_out_start)

        with self.tik_inst.for_range(0, op.h_per_core >> 1) as _h_idx:
            self._do_one_height(op.core_height_start + _h_idx * 2, nc_len, w_in_len, w_out_len,
                                self.image_in_buf_ping, self.reorder_buf_ping)
            self._do_one_height(op.core_height_start + _h_idx * 2 + 1, nc_len, w_in_len, w_out_len,
                                self.image_in_buf_pong, self.reorder_buf_pong)
        with self.tik_inst.if_scope(op.h_per_core % 2 == 1):
            self._do_one_height(op.core_height_start + op.h_per_core - 1, nc_len, w_in_len, w_out_len,
                                self.image_in_buf_ping, self.reorder_buf_ping)

    def _do_one_height(self, h_out_idx, nc_len, w_in_len, w_out_len, in_image_buf, reorder_buf) -> None:
        with self.tik_inst.new_stmt_scope():
            op = self.op
            h_in_idx = self.tik_inst.Scalar(op.coordinate_dtype, name="h_in_idx")
            h_in_idx_array = self.tik_inst.Tensor(op.coordinate_dtype, (64,),
                                                  name="h_in_idx_array", scope=tik.scope_ubuf)
            h_in_idx_array_int = self.tik_inst.Tensor("int32", (64,), name="h_in_idx_array_int", scope=tik.scope_ubuf)
            h_in_idx_array_fpx = self.tik_inst.Tensor(op.compute_dtype, (64,), name="h_in_idx_array_fpx",
                                                      scope=tik.scope_ubuf)

            util_tik_comm_func.tik_func_vector(self.tik_inst, h_in_idx_array, h_out_idx, 64)
            util_tik_comm_func.tik_func_vconv(self.tik_inst, h_in_idx_array_fpx, h_in_idx_array, 64)
            self.calculate_src_idx(self.tik_inst, op.h_rec_scale_fpx, h_in_idx_array_int, h_in_idx_array_fpx, 1,
                                   op.align_corners, op.half_pixel_centers)

            h_in_idx.set_as(h_in_idx_array_int[0])
            with self.tik_inst.if_scope(h_in_idx > op.input_height - 1):
                h_in_idx.set_as(op.input_height - 1)

            gm_in_offset = op.shape_c0 * (self.gm_in_start + h_in_idx * op.input_width)
            gm_out_offset = op.shape_c0 * (self.gm_out_start + h_out_idx * op.output_width)
            self._image_reorder(gm_in_offset, gm_out_offset, nc_len, w_in_len, w_out_len, in_image_buf, reorder_buf)

    def _image_reorder(self, gm_in_offset, gm_out_offset, nc_len, w_in_len, w_out_len,
                       in_image_buf, reorder_buf) -> None:
        self.nc_batch_move_from_gm(self.tik_inst, self.op, gm_in_offset, in_image_buf,
                                   nc_len, w_in_len, self.image_in_exceed_stride)
        self._reorder_in_ub(in_image_buf, reorder_buf, nc_len, w_in_len, w_out_len)
        self.nc_batch_move_to_gm(self.tik_inst, self.op, gm_out_offset, reorder_buf,
                                 nc_len, w_out_len, self.image_out_exceed_stride)

    def _reorder_in_ub(self, in_image_buf, reorder_buf, nc_len, w_in_len, w_out_len) -> None:
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, w_out_len) as _w_idx:
                op = self.op
                w_in_idx = self.tik_inst.Scalar("int32", name="w_in_idx", init_value=self.w_in_idx_array[_w_idx])
                burst_num = nc_len
                burst_len = op.c0_blocks
                burst_src_gap = w_in_len * op.c0_blocks - burst_len
                burst_dst_gap = w_out_len * op.c0_blocks - burst_len
                src_offset = (w_in_idx - self.w_in_start_idx) * op.shape_c0
                dst_offset = _w_idx * op.shape_c0
                self.tik_inst.data_move(reorder_buf[dst_offset:], in_image_buf[src_offset:],
                                        0, burst_num, burst_len, burst_src_gap, burst_dst_gap)



# 'pylint: disable=unused-argument
def check_supported(images, size, y, align_corners=False, half_pixel_centers=False,
                    kernel_name="resize_nearest_neighbor_v2"):
    """
    check whether ai_core is supported
    """
    image_shape = images.get("shape")
    h_in = image_shape[2]
    w_in = image_shape[3]
    is_support_vadds = tbe_platform.api_check_support("tik.vadds", "float32")
    if not is_support_vadds:
        size_value = size.get("const_value")
        if not size_value:
            return False, "size is not const."
        h_out = size_value[0]
        w_out = size_value[1]
        axis_list = [h_in, h_out, w_in, w_out]
        for axis in axis_list:
            if axis < 1 or axis > Constant.MAX_H_W:
                return False, "H or W is too large to compute with float16."

    return True, ""


def fill_index_in_ub(tik_instance: tik.Tik, idx_ub, idx_num, vector_num=64) -> None:
    """
    fill 0,1,2  .... (idx_num -1) in idx_ub
    when the idx_num is less than 16, fill it one by one
    when the type is not int32, will fill in int32 ub and cast to idx_ub dtype
    when the type is int32, will fill in int32 one by one
    """
    # when the idx_num is less than 16, fill it one by one
    with tik_instance.new_stmt_scope():
        vector_num_ub = tik_instance.Tensor(idx_ub.dtype, (vector_num,),
                                            name="vector_num_ub", scope=tik.scope_ubuf)
        block_num = 32 // common_util.get_data_size(idx_ub.dtype)
        loop_num = vector_num // block_num
        for _idx in range(block_num):
            idx_ub[_idx].set_as(_idx)
        tik_instance.vector_dup(vector_num, vector_num_ub, block_num, 1, 1, block_num)
        with tik_instance.for_range(1, loop_num) as add_idx:
            add_offset = add_idx * block_num
            tik_instance.vadd(block_num, idx_ub[add_offset:], vector_num_ub,
                              idx_ub[add_offset - block_num:],
                              1, 1, 1, 1, block_num, 0, block_num)

        idx_vector_num = (idx_num + vector_num - 1) // vector_num
        with tik_instance.if_scope(idx_vector_num > 1):
            tik_instance.vector_dup(vector_num, vector_num_ub, vector_num, 1, 1, block_num)
            with tik_instance.for_range(1, idx_vector_num) as add_idx:
                add_offset = add_idx * vector_num
                tik_instance.vadd(vector_num, idx_ub[add_offset:], vector_num_ub, idx_ub[add_offset - vector_num:],
                                  1, 1, 1, 1, block_num, 0, block_num)


def scalar_vconv_int_to_fp(tik_instance: tik.Tik, int_value, float_value) -> None:
    """
    vconv one scalar from int32 to fp32 using vector
    """
    with tik_instance.new_stmt_scope():
        compute_block_num = 32 // common_util.get_data_size(float_value.dtype)
        int_value_dtype = "int16" if float_value.dtype == "float16" else "int32"
        idx_int_tmp = tik_instance.Tensor(int_value_dtype, (compute_block_num,),
                                          name="idx_int_tmp", scope=tik.scope_ubuf)
        idx_fp_tmp = tik_instance.Tensor(float_value.dtype, (compute_block_num,),
                                         name="idx_fp_tmp", scope=tik.scope_ubuf)
        if float_value.dtype == "float16":
            int_value_tmp = tik_instance.Scalar(int_value_dtype, name="int_value_tmp", init_value=int_value)
            util_tik_comm_func.tik_func_vector(tik_instance, idx_int_tmp, int_value_tmp, 1)
        else:
            idx_int_tmp[0].set_as(int_value)
        util_tik_comm_func.tik_func_vconv(tik_instance, idx_fp_tmp, idx_int_tmp, 1)
        float_value.set_as(idx_fp_tmp[0])


@register_operator("ResizeNearestNeighborV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def resize_nearest_neighbor_v2(images, size, y, align_corners=False, half_pixel_centers=False,
                               kernel_name="resize_nearest_neighbor_v2"):
    """Resize `images` to `size` using nearest neighbor interpolation.

    Parameters
    ----------
    images: dict
        the dict of input, include shape of input_tensor which layout
        only support 5HD and dtype supports 'float16', 'float32'
    size: dict
        the dict of input, positive height and width of output tensor
        only support 1D and dtype supports 'int32'
    y: dict
        the dict of output, include shape of input_tensor which layout
        only support 5HD and dtype supports 'float16', 'float32'
    align_corners: bool
        whether align_corners
    half_pixel_centers: bool
        whether half_pixel_centers
        align_corners & half_pixel_centers can not be both TRUE in zoom-out scenario,
        otherwise, the input index will be out of range
    kernel_name: str
        cce kernel name, default value is `resize_nearest_neighbor`

    Returns
    -------
    tik_instance

    Note
    -------
    Please refer to `test_resize_nearest_neighbor_v2_dynamic_impl.py` for python implemention.
    """
    if images.get("format") == "NCHW":
        return resize_2d(images, size, y, align_corners, half_pixel_centers, kernel_name)

    obj = ResizeNearestNeighbor(images, size, y, align_corners, half_pixel_centers, kernel_name)
    return obj.resize_nearest_neighbor_v2_operator()
