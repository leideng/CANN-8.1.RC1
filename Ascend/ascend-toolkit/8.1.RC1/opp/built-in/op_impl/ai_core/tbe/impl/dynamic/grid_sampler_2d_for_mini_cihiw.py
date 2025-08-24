#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

grid_sampler_2d_for_mini_cihiw
"""
# 'pylint: disable=too-many-lines
from impl import constant_util as constant
from impl.util import util_common
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.util_tik_comm_func import ceil_align
from impl.util.util_tik_comm_func import ceil_div
from impl.util.util_tik_comm_func import floor_align


# 'pylint: disable=old-style-class,no-init,too-few-public-methods
class Gs2Constant:
    TILING_ARG_NUM = 16

    X_UB_SIZE_4_MINI_C_IH_IW = 81920  # 80KB

    X_UB_SIZE_4_GENERAL = 16384  # 16KB
    GRID_UB_SIZE_4_GENERAL = 2048   # 2KB

    OUT_VAL_NUM = 4096  # 4KB

    INTERPOLATION_MODE_BILINEAR = 'bilinear'
    PADDING_MODE_ZEROS = 'zeros'
    PADDING_MODE_BORDER = 'border'

    TILING_MODE_2 = 2  # tling mode for split core to some group, compute n on one group
    TILING_MODE_4 = 4  # tling mode for split core by H*W


# 'pylint: disable=old-style-class,too-few-public-methods,too-many-instance-attributes
class Params:
    # 'pylint: disable=unused-argument,too-many-arguments,huawei-too-many-arguments
    def __init__(self, x, grid, y, interpolation_mode, padding_mode, align_corners):
        x_shape = x.get("shape")
        self.in_n = x_shape[0]
        self.in_c = x_shape[1]
        self.in_h = x_shape[2]
        self.in_w = x_shape[3]

        self.channel_last = 0

        x_format = x.get("format").upper()
        if x_format == "NHWC":
            self.channel_last = 1
            self.in_h = x_shape[1]
            self.in_w = x_shape[2]
            self.in_c = x_shape[3]

        grid_shape = grid.get("shape")
        self.out_h = grid_shape[1]
        self.out_w = grid_shape[2]

        self.interpolation_mode = interpolation_mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners


# 'pylint: disable=old-style-class,too-few-public-methods
class Gm:
    def __init__(self, tik_inst, d_type, need_clean_y):
        self.tiling_gm = tik_inst.Tensor(constant.DATA_TYPE_INT64, (Gs2Constant.TILING_ARG_NUM,),
                                         name="tiling_gm", scope=tik.scope_gm)
        self.x_gm = tik_inst.Tensor(d_type, (constant.SHAPE_SIZE_LIMIT,), name="x_gm", scope=tik.scope_gm)
        self.grid_gm = tik_inst.Tensor(d_type, (constant.SHAPE_SIZE_LIMIT,), name="grid_gm", scope=tik.scope_gm)
        self.y_gm = None
        if need_clean_y:
            self.y_gm = tik_inst.Tensor(d_type, (constant.SHAPE_SIZE_LIMIT,), name="y_gm", scope=tik.scope_gm,
                                        is_atomic_add=True)
        else:
            self.y_gm = tik_inst.Tensor(d_type, (constant.SHAPE_SIZE_LIMIT,), name="y_gm", scope=tik.scope_gm)


# 'pylint: disable=old-style-class,too-few-public-methods,too-many-instance-attributes
class Ub4MiniCIhIw:
    def __init__(self, tik_inst, d_type, d_size):
        grid_ub_num = Gs2Constant.GRID_UB_SIZE_4_GENERAL // d_size
        clip_mask_ub_num = grid_ub_num // constant.SIZE_SIXTEEN

        self.x_ub = tik_inst.Tensor(d_type, (Gs2Constant.X_UB_SIZE_4_GENERAL // d_size,),
                                    name="x_ub", scope=tik.scope_ubuf)              # 16KB
        self.grid_ub = tik_inst.Tensor(d_type, (grid_ub_num * 2,),
                                       name="grid_ub", scope=tik.scope_ubuf)        # 4KB
        self.y_ub = tik_inst.Tensor(d_type, (grid_ub_num,),
                                    name="y_ub", scope=tik.scope_ubuf)              # 2KB

        self.x_cache_ub = tik_inst.Tensor(d_type, (Gs2Constant.X_UB_SIZE_4_MINI_C_IH_IW // d_size,),
                                          name="x_cache_ub", scope=tik.scope_ubuf)  # 80KB
        self.ixy_fp = tik_inst.Tensor(d_type, (grid_ub_num * 2,),
                                      name="ixy_fp", scope=tik.scope_ubuf)          # 2KB

        self.ix_nw_sw_int = tik_inst.Tensor(constant.DATA_TYPE_INT32, (grid_ub_num,),
                                            name="ix_nw_sw_int", scope=tik.scope_ubuf)        # 2KB
        self.iy_nw_sw_int = tik_inst.Tensor(constant.DATA_TYPE_INT32, (grid_ub_num,),
                                            name="iy_nw_sw_int", scope=tik.scope_ubuf)        # 2KB
        self.ix_nw_sw_fp = tik_inst.Tensor(constant.DATA_TYPE_FP32, (grid_ub_num,),
                                           name="ix_nw_sw_fp", scope=tik.scope_ubuf)          # 2KB
        self.iy_nw_sw_fp = tik_inst.Tensor(constant.DATA_TYPE_FP32, (grid_ub_num,),
                                           name="iy_nw_sw_fp", scope=tik.scope_ubuf)          # 2KB

        self.ix_ne_se_int = tik_inst.Tensor(constant.DATA_TYPE_INT32, (grid_ub_num,),
                                            name="ix_ne_se_int", scope=tik.scope_ubuf)        # 2KB
        self.iy_ne_se_int = tik_inst.Tensor(constant.DATA_TYPE_INT32, (grid_ub_num,),
                                            name="iy_ne_se_int", scope=tik.scope_ubuf)        # 2KB
        self.ix_ne_se_fp = tik_inst.Tensor(constant.DATA_TYPE_FP32, (grid_ub_num,),
                                           name="ix_ne_se_fp", scope=tik.scope_ubuf)          # 2KB
        self.iy_ne_se_fp = tik_inst.Tensor(constant.DATA_TYPE_FP32, (grid_ub_num,),
                                           name="iy_ne_se_fp", scope=tik.scope_ubuf)          # 2KB

        self.clip_mask1 = tik_inst.Tensor(constant.DATA_TYPE_UINT16, (clip_mask_ub_num,),
                                          name="clip_mask1", scope=tik.scope_ubuf)    # 128B
        self.clip_mask2 = tik_inst.Tensor(constant.DATA_TYPE_UINT16, (clip_mask_ub_num,),
                                          name="clip_mask2", scope=tik.scope_ubuf)    # 128B

        self.xx = tik_inst.Tensor(constant.DATA_TYPE_INT32, (grid_ub_num,),
                                  name="xx", scope=tik.scope_ubuf)          # 2KB
        self.yy = tik_inst.Tensor(constant.DATA_TYPE_INT32, (grid_ub_num,),
                                  name="yy", scope=tik.scope_ubuf)          # 2KB
        self.gather_offset_factor = tik_inst.Tensor(constant.DATA_TYPE_INT32, (8,),
                                                    name="gather_offset_factor", scope=tik.scope_ubuf)  # 32B

        self.weight = tik_inst.Tensor(constant.DATA_TYPE_FP32, (grid_ub_num,),
                                      name="weight", scope=tik.scope_ubuf)          # 2KB
        self.nw = tik_inst.Tensor(constant.DATA_TYPE_FP32, (grid_ub_num,),
                                  name="nw", scope=tik.scope_ubuf)          # 2KB
        self.ne = tik_inst.Tensor(constant.DATA_TYPE_FP32, (grid_ub_num,),
                                  name="ne", scope=tik.scope_ubuf)          # 2KB
        self.sw = tik_inst.Tensor(constant.DATA_TYPE_FP32, (grid_ub_num,),
                                  name="sw", scope=tik.scope_ubuf)          # 2KB
        self.se = tik_inst.Tensor(constant.DATA_TYPE_FP32, (grid_ub_num,),
                                  name="se", scope=tik.scope_ubuf)          # 2KB
        self.out_val = tik_inst.Tensor(d_type, (Gs2Constant.X_UB_SIZE_4_GENERAL // d_size,),
                                       name="out_val", scope=tik.scope_ubuf)        # 16KB


# 'pylint: disable=old-style-class,too-many-instance-attributes,too-few-public-methods
class Shape:
    def __init__(self, tik_inst, tiling_ub, params):
        self.in_n = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="in_n")
        if params.in_n > 0:
            self.in_n.set_as(params.in_n)
        else:
            self.in_n.set_as(tiling_ub[1])

        self.in_c = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="in_c")
        if params.in_c > 0:
            self.in_c.set_as(params.in_c)
        else:
            self.in_c.set_as(tiling_ub[2])

        self.in_h = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="in_h")
        if params.in_h > 0:
            self.in_h.set_as(params.in_h)
        else:
            self.in_h.set_as(tiling_ub[3])

        self.in_w = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="in_w")
        if params.in_w > 0:
            self.in_w.set_as(params.in_w)
        else:
            self.in_w.set_as(tiling_ub[4])

        self.out_h = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="out_h")
        if params.out_h > 0:
            self.out_h.set_as(params.out_h)
        else:
            self.out_h.set_as(tiling_ub[5])

        self.out_w = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="out_w")
        if params.out_w > 0:
            self.out_w.set_as(params.out_w)
        else:
            self.out_w.set_as(tiling_ub[6])

        self.in_hw = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="in_hw")
        self.in_chw = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="in_chw")
        self.grid_hw = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="grid_hw")
        self.out_chw = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="out_chw")
        self.in_hw.set_as(self.in_h * self.in_w)
        self.in_chw.set_as(self.in_c * self.in_h * self.in_w)
        self.grid_hw.set_as(self.out_h * self.out_w)
        self.out_chw.set_as(self.in_c * self.out_h * self.out_w)

        # only support `IH < 2^31` and `IW < 2^31`
        self.in_c_int32 = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT32, name="in_c_int32")
        self.in_h_int32 = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT32, name="in_h_int32")
        self.in_w_int32 = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT32, name="in_w_int32")
        self.in_c_int32.set_as(self.in_c)
        self.in_h_int32.set_as(self.in_h)
        self.in_w_int32.set_as(self.in_w)

        self.in_c_fp32 = tik_inst.Scalar(dtype=constant.DATA_TYPE_FP32, name="in_c_fp32")
        self.in_h_fp32 = tik_inst.Scalar(dtype=constant.DATA_TYPE_FP32, name="in_h_fp32")
        self.in_w_fp32 = tik_inst.Scalar(dtype=constant.DATA_TYPE_FP32, name="in_w_fp32")
        tik_inst.scalar_conv('', self.in_c_fp32, self.in_c_int32)
        tik_inst.scalar_conv('', self.in_h_fp32, self.in_h_int32)
        tik_inst.scalar_conv('', self.in_w_fp32, self.in_w_int32)


# 'pylint: disable=old-style-class,too-many-instance-attributes,too-few-public-methods
class Args:
    def __init__(self, tik_inst, tiling_ub, params):
        self.core_num_var = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="core_num_var")
        self.interpolation_mode = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="interpolation_mode")
        self.padding_mode = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="padding_mode")
        self.align_corners = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="align_corners")
        self.channel_last = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="channel_last")
        self.need_core_num = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="need_core_num")
        self.pre_core_num = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="pre_core_num")
        self.pre_num_per_core = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="pre_num_per_core")
        self.post_num_per_core = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="post_num_per_core")

        self.core_num_var.set_as(tiling_ub[0])
        self.interpolation_mode.set_as(tiling_ub[7])  # 0: 'bilinear'; 1: 'nearest' ; 2: 'bicubic'
        self.padding_mode.set_as(tiling_ub[8])        # 0: 'zeros'; 1: 'border'; 2: 'reflection'
        self.align_corners.set_as(tiling_ub[9])       # 0: false; 1: true
        self.channel_last.set_as(tiling_ub[10])       # 0: nchw; 1: nhwc
        self.need_core_num.set_as(tiling_ub[11])
        self.pre_core_num.set_as(tiling_ub[12])
        self.pre_num_per_core.set_as(tiling_ub[13])
        self.post_num_per_core.set_as(tiling_ub[14])

        self.interp_mode_const = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64,
                                                 name="interp_mode_const", init_value=0)
        if params.interpolation_mode == 'nearest':
            self.interp_mode_const.set_as(1)
        elif params.interpolation_mode == 'bicubic':
            self.interp_mode_const.set_as(2)

        self.tiling_mode = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="tiling_mode", init_value=0)


# 'pylint: disable=old-style-class,too-few-public-methods,unused-variable
class Reg:
    def __init__(self, tik_inst):
        self.grid_hw_offset = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="grid_hw_offset")
        self.grid_cur_num = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="grid_cur_num")
        self.grid_cur_num_aln = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="grid_cur_num_aln")
        self.grid_rep_times = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="grid_rep_times")

        self.x_loc = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT32, name="x_loc")
        self.x_weight = tik_inst.Scalar(dtype=constant.DATA_TYPE_FP32, name="x_weight")

        self.y_offset = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="y_offset")
        self.mte_burst = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="mte_burst")

        self.tail_offset = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="tail_offset")


# 'pylint: disable=old-style-class,too-many-instance-attributes,too-few-public-methods
class GridSampler2D4MiniCihiw:
    """
    GridSampler2D4MiniCihiw op implement
    """

    # 'pylint: disable=unused-argument,too-many-arguments,huawei-too-many-arguments
    def __init__(self, x, grid, y, interpolation_mode, padding_mode, align_corners, kernel_name, tik_inst):
        self.params = Params(x, grid, y, interpolation_mode, padding_mode, align_corners)
        self.interpolation_mode = interpolation_mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.kernel_name = kernel_name

        self.tik_inst = tik_inst
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.support_data_move_pad = tbe_platform.api_check_support("tik.data_move_pad")

        self.d_type = x.get("dtype").lower()
        self.d_size = tbe_platform.get_bit_len(self.d_type) // constant.DATA_SIZE_EIGHT
        self.d_num_1block = constant.MAX_BLOCK_NUMBER // self.d_size

        self.vmask = constant.VECTOR_BYTE_SIZE // self.d_size
        self.grid_ub_num = Gs2Constant.GRID_UB_SIZE_4_GENERAL // self.d_size
        self.clip_mask_ub_num = self.grid_ub_num // constant.SIZE_SIXTEEN

        self.gm = Gm(self.tik_inst, self.d_type, True)
        self.shape = None
        self.args = None
        self.reg = None
        self.ub = None

    # 'pylint: disable=too-many-return-statements
    def check_params(self):
        if not tbe_platform.api_check_support("tik.vcopy"):
            return False, "no support for vbcb(int32)"

        if self.params.channel_last != 1:
            return False, "only support NHWC"
        if self.d_type != "float32":
            return False, "only support float32"
        if self.interpolation_mode != Gs2Constant.INTERPOLATION_MODE_BILINEAR:
            return False, "interpolation_mode only support bilinear"
        if self.padding_mode not in ('zeros', 'border'):
            return False, "padding_mode only support zeros or border"

        mini_cihiw_cases = (
            (8, 32, 24, 24, 900, 4),
            (8, 32, 24, 24, 196416, 4),
        )
        if (self.params.in_n, self.params.in_c, self.params.in_h, self.params.in_w,
            self.params.out_h, self.params.out_w) in mini_cihiw_cases:
            return True, ""

        return False, "no support this case"

    def compute(self):
        """
        compute
        """
        tiling_ub = self.tik_inst.Tensor(constant.DATA_TYPE_INT64, (constant.SIZE_SIXTEEN,),
                                         name="tiling_ub", scope=tik.scope_ubuf)
        self.tik_inst.data_move(tiling_ub, self.gm.tiling_gm, 0, 1, 4, 0, 0)         # no need DataMovePad
        self.shape = Shape(self.tik_inst, tiling_ub, self.params)
        self.args = Args(self.tik_inst, tiling_ub, self.params)
        self.reg = Reg(self.tik_inst)

        with self.tik_inst.for_range(0, self.args.core_num_var, block_num=self.args.core_num_var) as core_id:
            with self.tik_inst.if_scope(core_id < self.args.need_core_num):
                with self.tik_inst.new_stmt_scope():
                    self.ub = Ub4MiniCIhIw(self.tik_inst, self.d_type, self.d_size)
                    self.tik_inst.set_atomic_add(self.d_type)
                    self._compute_one_core(core_id)
                    self.tik_inst.set_atomic_add(0)

        tbe_context.get_context().add_compile_info("vars", {"core_num": self.core_num,
                                                            "is_support_minicihiw": 1})
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                               inputs=[self.gm.x_gm, self.gm.grid_gm],
                               outputs=[self.gm.y_gm],
                               flowtable=(self.gm.tiling_gm,),
                               config=opt_config)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def _compute_weight(self, w1, w2, x1, x2, y1, y2):
        """
        compute weight: `weight = ( x1 - x2 ) * ( y1 - y2 )`
        """
        self.tik_inst.vsub(self.vmask, w1, x1, x2, self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vsub(self.vmask, w2, y1, y2, self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vmul(self.vmask, w1, w1, w2, self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def _gather_val(self, ii_loc):
        self.tik_inst.vbcb(self.ub.yy, self.ub.xx[ii_loc * self.vmask], 8, 1, 8)
        self.tik_inst.vreduce(512, self.ub.yy, self.ub.yy, 1, 1, 1, 8, 0, 0, None, 'counter')
        self.tik_inst.vmuls(self.vmask, self.ub.yy, self.ub.yy, self.shape.in_c_int32, 4, 1, 1, 8, 8)
        self.tik_inst.vadd(self.vmask, self.ub.yy, self.ub.yy, self.ub.gather_offset_factor,
                           4, 1, 1, 0, 8, 8, 0)
        self.tik_inst.vmuls(self.vmask, self.ub.yy, self.ub.yy, self.d_size, 4, 1, 1, 8, 8)
        self.tik_inst.vgatherb(self.ub.x_ub, self.ub.x_cache_ub, self.ub.yy, 32, 1, 8)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def _simple_vand(self, dst, src0, src1, d_mask, d_num):
        with self.tik_inst.if_scope(d_num >= d_mask):
            self.tik_inst.vand(d_mask, dst, src0, src1, d_num // d_mask, 1, 1, 1, 8, 8, 8)
        with self.tik_inst.if_scope(d_num % d_mask > 0):
            tail_start = self.clip_mask_ub_num // d_mask * d_mask
            self.tik_inst.vand(self.clip_mask_ub_num % d_mask,
                               self.ub.clip_mask1[tail_start],
                               self.ub.clip_mask1[tail_start],
                               self.ub.clip_mask2[tail_start], 1, 1, 1, 1, 8, 8, 8)

    def _simple_data_move(self, dst, src, num, d_type):
        if self.support_data_move_pad:
            self.tik_inst.data_move_pad(dst, src, 1, num * self.d_size, 0, 0)
        else:
            burst_len = (num * self.d_size + constant.BLOCK_SIZE - 1) // constant.BLOCK_SIZE
            self.tik_inst.data_move(dst, src, 0, 1, burst_len, 0, 0)

    def _simple_trans_fp32_64_8x(self, dst, src, channel_align):
        if tbe_platform.api_check_support("tik.v4dtrans"):
            self.tik_inst.v4dtrans(False, dst, src, self.vmask, channel_align)
        elif self.vmask == 64:  # only support channel_align eq 32
            with self.tik_inst.for_range(0, 4) as ii:
                src_list = [src[i * channel_align + ii * 16 * channel_align] for i in range(16)]
                dst_list = []
                for i in range(8):
                    dst_list = dst_list + [dst[64 * i + ii * 16], dst[64 * i + 8 + ii * 16]]
                self.tik_inst.vnchwconv(False, False, dst_list, src_list, channel_align // 8, 64, 1)

    def _simple_trans_fp32_x_64(self, dst, src, channel):
        if tbe_platform.api_check_support("tik.v4dtrans"):
            self.tik_inst.v4dtrans(True, dst, src, self.vmask, channel)
        elif self.vmask == 64:  # only support channel eq 32
            with self.tik_inst.for_range(0, 4) as ii:
                src_list = [src[i * 64 + ii * 16] for i in range(8)] +\
                            [src[i * 64 + 8 + ii * 16] for i in range(8)]
                dst_list = []
                for i in range(8):
                    dst_list = dst_list + [dst[channel * (i + ii * 16)],
                                            dst[channel * (i + 8 + ii * 16)]]
                self.tik_inst.vnchwconv(False, False, dst_list, src_list, channel // 8, 1, 64)

    def _move_to_gm(self, n, ii_loc, channel_align):
        if tbe_platform.api_check_support("tik.v4dtrans"):
            self.reg.y_offset.set_as((n * self.shape.grid_hw + self.reg.grid_hw_offset +
                                      ii_loc * self.vmask) * self.shape.in_c)
            self.reg.mte_burst.set_as(self.shape.in_c * self.vmask * self.d_size // 32)
            self.tik_inst.data_move(self.gm.y_gm[self.reg.y_offset], self.ub.out_val,
                                    0, 1, self.reg.mte_burst, 0, 0)             # c eq 32, no need DataMovePad
        else:
            with self.tik_inst.if_scope(tik.any(self.shape.in_c == 1, self.shape.in_c == 3,
                                                self.shape.in_c == channel_align)):
                self.reg.y_offset.set_as((n * self.shape.grid_hw + self.reg.grid_hw_offset +
                                          ii_loc * self.vmask) * self.shape.in_c)
                self.reg.mte_burst.set_as(self.shape.in_c * self.vmask * self.d_size // 32)
                self.tik_inst.data_move(self.gm.y_gm[self.reg.y_offset], self.ub.out_val,
                                        0, 1, self.reg.mte_burst, 0, 0)         # c eq 32, no need DataMovePad
            with self.tik_inst.else_scope():
                with self.tik_inst.for_range(0, self.vmask) as iii:
                    self.reg.y_offset.set_as((n * self.shape.grid_hw + self.reg.grid_hw_offset +
                                              ii_loc * self.vmask + iii) * self.shape.in_c)
                    self.reg.mte_burst.set_as(channel_align * self.d_size // 32)
                    self.tik_inst.data_move(self.gm.y_gm[self.reg.y_offset], self.ub.out_val[iii * channel_align],
                                            0, 1, self.reg.mte_burst, 0, 0)     # c eq 32, no need DataMovePad

    def _clip_coordinates(self, ix_fp, iy_fp, ix_int, iy_int, out_coor, out_mask=None):
        with self.tik_inst.if_scope(self.padding_mode == Gs2Constant.PADDING_MODE_ZEROS):
            self.tik_inst.vcmpvs_ge(self.ub.clip_mask1, ix_fp, 0, self.reg.grid_rep_times, 1, 8)

            self.tik_inst.vcmpvs_ge(self.ub.clip_mask2, iy_fp, 0, self.reg.grid_rep_times, 1, 8)
            self._simple_vand(self.ub.clip_mask1, self.ub.clip_mask1, self.ub.clip_mask2,
                              constant.MASK128, self.clip_mask_ub_num)

            self.tik_inst.vcmpvs_lt(self.ub.clip_mask2, ix_fp, self.shape.in_w_fp32,
                                    self.reg.grid_rep_times, 1, 8)
            self._simple_vand(self.ub.clip_mask1, self.ub.clip_mask1, self.ub.clip_mask2,
                              constant.MASK128, self.clip_mask_ub_num)

            self.tik_inst.vcmpvs_lt(self.ub.clip_mask2, iy_fp, self.shape.in_h_fp32,
                                    self.reg.grid_rep_times, 1, 8)
            self._simple_vand(self.ub.clip_mask1, self.ub.clip_mask1, self.ub.clip_mask2,
                              constant.MASK128, self.clip_mask_ub_num)

        # bound value for padding_mode is 'zeros' or 'border'
        tmp_coor = self.ub.yy
        self.tik_inst.vmins(self.vmask, out_coor, ix_int,
                            self.shape.in_w_int32 - 1, self.reg.grid_rep_times, 1, 1, 8, 8)
        self.tik_inst.vmins(self.vmask, tmp_coor, iy_int,
                            self.shape.in_h - 1, self.reg.grid_rep_times, 1, 1, 8, 8)
        self.tik_inst.vmaxs(self.vmask, out_coor, out_coor,
                            0, self.reg.grid_rep_times, 1, 1, 8, 8)
        self.tik_inst.vmaxs(self.vmask, tmp_coor, tmp_coor,
                            0, self.reg.grid_rep_times, 1, 1, 8, 8)

        self.tik_inst.vmuls(self.vmask, tmp_coor, tmp_coor,
                            self.shape.in_w_int32, self.reg.grid_rep_times, 1, 1, 8, 8)
        self.tik_inst.vadd(self.vmask, out_coor, out_coor, tmp_coor,
                           self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)

    def _process_point_bilinear_nhwc_gt1_le64(self, n, weight):
        """case: NHWC and 1 < C <= 64"""
        channel_align = ceil_align(self.shape.in_c, self.d_num_1block)
        with self.tik_inst.for_range(0, self.reg.grid_cur_num // self.vmask) as ii_loc:
            self._gather_val(ii_loc)

            # [128or64, channel_align] --> [channel_align, 128or64]
            self._simple_trans_fp32_64_8x(self.ub.out_val, self.ub.x_ub, channel_align)

            # x_value x weight, [channel, 128or64] x [128or64,], repeat channel times
            self.tik_inst.vmul(self.vmask, self.ub.x_ub, self.ub.out_val, weight[ii_loc * self.vmask],
                               self.shape.in_c, 1, 1, 1, 8, 8, 0)

            # [channel_align, 128or64] --> [128or64, channel]
            self._simple_trans_fp32_x_64(self.ub.out_val, self.ub.x_ub, self.shape.in_c_int32)

            # Move out [128or64, channel]
            self._move_to_gm(n, ii_loc, channel_align)

        self.reg.tail_offset.set_as(floor_align(self.reg.grid_cur_num, self.vmask))
        with self.tik_inst.if_scope(self.reg.grid_cur_num > self.reg.tail_offset):
            self.tik_inst.vec_dup(self.vmask, self.ub.out_val, 0.0, Gs2Constant.OUT_VAL_NUM // self.vmask, 8)
            with self.tik_inst.for_range(self.reg.tail_offset, self.reg.grid_cur_num) as iii_loc:
                self.reg.x_loc.set_as(self.ub.xx[iii_loc])
                self.reg.x_loc.set_as(n * self.shape.in_chw + self.reg.x_loc * self.shape.in_c)
                self.reg.mte_burst.set_as((self.shape.in_c * self.d_size + 31) // 32)
                self.tik_inst.data_move(self.ub.x_ub, self.gm.x_gm[self.reg.x_loc], 0, 1,
                                        self.reg.mte_burst, 0, 0)         # c eq 32, no need DataMovePad

                self.reg.x_weight.set_as(weight[iii_loc])
                self.tik_inst.vmuls(self.shape.in_c, self.ub.x_ub, self.ub.x_ub, self.reg.x_weight,
                                    1, 1, 1, 8, 8)
                with self.tik_inst.for_range(0, self.shape.in_c) as j:
                    self.ub.out_val[(iii_loc - self.reg.tail_offset) * self.shape.in_c +
                                    j].set_as(self.ub.x_ub[j])

            self.reg.y_offset.set_as((n * self.shape.grid_hw + self.reg.grid_hw_offset +
                                      self.reg.tail_offset) * self.shape.in_c)
            self.reg.mte_burst.set_as(((self.reg.grid_cur_num -
                                        self.reg.tail_offset) * self.shape.in_c * self.d_size + 31) // 32)
            self.tik_inst.data_move(self.gm.y_gm[self.reg.y_offset], self.ub.out_val, 0, 1,
                                    self.reg.mte_burst, 0, 0)         # c eq 32, no need DataMovePad

    def _process_point_bilinear(self, n, weight):
        with self.tik_inst.if_scope(self.padding_mode == Gs2Constant.PADDING_MODE_ZEROS):
            self.tik_inst.vsel(self.vmask, 1, weight, self.ub.clip_mask1, weight, 0.0,
                               self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)

        # only support `C==32`
        self._process_point_bilinear_nhwc_gt1_le64(n, weight)

    def _compute_coordinates_bilinear(self):
        if self.padding_mode == Gs2Constant.PADDING_MODE_BORDER:
            self.tik_inst.vmaxs(self.vmask, self.ub.grid_ub, self.ub.grid_ub,
                                0, self.reg.grid_rep_times, 1, 1, 8, 8)
            self.tik_inst.vmaxs(self.vmask, self.ub.grid_ub[self.grid_ub_num], self.ub.grid_ub[self.grid_ub_num],
                                0, self.reg.grid_rep_times, 1, 1, 8, 8)
            self.tik_inst.vmins(self.vmask, self.ub.grid_ub, self.ub.grid_ub,
                                self.shape.in_w_fp32 - 1.0, self.reg.grid_rep_times, 1, 1, 8, 8)
            self.tik_inst.vmins(self.vmask, self.ub.grid_ub[self.grid_ub_num], self.ub.grid_ub[self.grid_ub_num],
                                self.shape.in_h_fp32 - 1.0, self.reg.grid_rep_times, 1, 1, 8, 8)

    # 'pylint: disable=too-many-locals
    def _compute_val_bilinear(self, n):
        self._compute_coordinates_bilinear()

        self.tik_inst.vconv(self.vmask, 'floor', self.ub.ix_nw_sw_int, self.ub.grid_ub,
                            self.reg.grid_rep_times, 1, 1, 8, 8)               # ix floor co-ordinates
        self.tik_inst.vconv(self.vmask, 'none', self.ub.ix_nw_sw_fp, self.ub.ix_nw_sw_int,
                            self.reg.grid_rep_times, 1, 1, 8, 8)               # ix floor co-ordinates
        self.tik_inst.vadds(self.vmask, self.ub.ix_ne_se_int, self.ub.ix_nw_sw_int, 1,
                            self.reg.grid_rep_times, 1, 1, 8, 8)               # ix ceil co-ordinates
        self.tik_inst.vadds(self.vmask, self.ub.ix_ne_se_fp, self.ub.ix_nw_sw_fp, 1.0,
                            self.reg.grid_rep_times, 1, 1, 8, 8)               # ix ceil co-ordinates

        self.tik_inst.vconv(self.vmask, 'floor', self.ub.iy_nw_sw_int, self.ub.grid_ub[self.grid_ub_num],
                            self.reg.grid_rep_times, 1, 1, 8, 8)               # iy floor co-ordinates
        self.tik_inst.vconv(self.vmask, 'none', self.ub.iy_nw_sw_fp, self.ub.iy_nw_sw_int,
                            self.reg.grid_rep_times, 1, 1, 8, 8)               # iy floor co-ordinates
        self.tik_inst.vadds(self.vmask, self.ub.iy_ne_se_int, self.ub.iy_nw_sw_int, 1,
                            self.reg.grid_rep_times, 1, 1, 8, 8)               # iy ceil co-ordinates
        self.tik_inst.vadds(self.vmask, self.ub.iy_ne_se_fp, self.ub.iy_nw_sw_fp, 1.0,
                            self.reg.grid_rep_times, 1, 1, 8, 8)               # iy ceil co-ordinates

        ix, iy = self.ub.grid_ub, self.ub.grid_ub[self.grid_ub_num]

        ix_nw_fp, iy_nw_fp = self.ub.ix_nw_sw_fp, self.ub.iy_nw_sw_fp
        ix_nw_int, iy_nw_int = self.ub.ix_nw_sw_int, self.ub.iy_nw_sw_int

        ix_ne_fp, iy_ne_fp = self.ub.ix_ne_se_fp, self.ub.iy_nw_sw_fp
        ix_ne_int, iy_ne_int = self.ub.ix_ne_se_int, self.ub.iy_nw_sw_int

        ix_sw_fp, iy_sw_fp = self.ub.ix_nw_sw_fp, self.ub.iy_ne_se_fp
        ix_sw_int, iy_sw_int = self.ub.ix_nw_sw_int, self.ub.iy_ne_se_int

        ix_se_fp, iy_se_fp = self.ub.ix_ne_se_fp, self.ub.iy_ne_se_fp
        ix_se_int, iy_se_int = self.ub.ix_ne_se_int, self.ub.iy_ne_se_int

        self._compute_weight(self.ub.nw, self.ub.weight, ix_se_fp, ix, iy_se_fp, iy)  # nw: ceil ceil weight
        self._compute_weight(self.ub.ne, self.ub.weight, ix, ix_sw_fp, iy_sw_fp, iy)  # ne: floor ceil weight
        self._compute_weight(self.ub.sw, self.ub.weight, ix_ne_fp, ix, iy, iy_ne_fp)  # sw: ceil floor weight
        self._compute_weight(self.ub.se, self.ub.weight, ix, ix_nw_fp, iy, iy_nw_fp)  # se: floor floor weight

        self._clip_coordinates(ix_nw_fp, iy_nw_fp, ix_nw_int, iy_nw_int, self.ub.xx)
        self._process_point_bilinear(n, self.ub.nw)

        self._clip_coordinates(ix_ne_fp, iy_ne_fp, ix_ne_int, iy_ne_int, self.ub.xx)
        self._process_point_bilinear(n, self.ub.ne)

        self._clip_coordinates(ix_sw_fp, iy_sw_fp, ix_sw_int, iy_sw_int, self.ub.xx)
        self._process_point_bilinear(n, self.ub.sw)

        self._clip_coordinates(ix_se_fp, iy_se_fp, ix_se_int, iy_se_int, self.ub.xx)
        self._process_point_bilinear(n, self.ub.se)

    def _compute_val(self, n):
        if self.interpolation_mode == Gs2Constant.INTERPOLATION_MODE_BILINEAR:
            self._compute_val_bilinear(n)
        else:
            pass

    def _unnormalize(self):
        self.tik_inst.vreduce(self.grid_ub_num * 2, self.ub.grid_ub, self.ub.ixy_fp,
                              1, 1, 1, 8, 0, 0, None, 'counter')
        self.tik_inst.vreduce(self.grid_ub_num * 2, self.ub.grid_ub[self.grid_ub_num], self.ub.ixy_fp,
                              2, 1, 1, 8, 0, 0, None, 'counter')

        with self.tik_inst.if_scope(self.args.align_corners == 1):
            # unnormalize coord from [-1, 1] to [0, size - 1]
            self.tik_inst.vmuls(self.vmask, self.ub.grid_ub, self.ub.grid_ub,
                                0.5 * (self.shape.in_w - 1.0), self.reg.grid_rep_times,
                                1, 1, 8, 8)                                    # ix
            self.tik_inst.vmuls(self.vmask, self.ub.grid_ub[self.grid_ub_num], self.ub.grid_ub[self.grid_ub_num],
                                0.5 * (self.shape.in_h - 1.0), self.reg.grid_rep_times,
                                1, 1, 8, 8)                                    # iy
        with self.tik_inst.else_scope():
            # unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
            self.tik_inst.vmuls(self.vmask, self.ub.grid_ub, self.ub.grid_ub,
                                0.5 * self.shape.in_w, self.reg.grid_rep_times,
                                1, 1, 8, 8)                                    # ix
            self.tik_inst.vmuls(self.vmask, self.ub.grid_ub[self.grid_ub_num], self.ub.grid_ub[self.grid_ub_num],
                                0.5 * self.shape.in_h, self.reg.grid_rep_times,
                                1, 1, 8, 8)                                    # iy
            self.tik_inst.vadds(self.vmask, self.ub.grid_ub, self.ub.grid_ub,
                                -0.5, self.reg.grid_rep_times, 1, 1, 8, 8)
            self.tik_inst.vadds(self.vmask, self.ub.grid_ub[self.grid_ub_num], self.ub.grid_ub[self.grid_ub_num],
                                -0.5, self.reg.grid_rep_times, 1, 1, 8, 8)

    def _compute_hw(self, n):
        self._simple_data_move(self.ub.grid_ub,
                               self.gm.grid_gm[n * self.shape.grid_hw * 2 + self.reg.grid_hw_offset * 2],
                               self.reg.grid_cur_num * 2,
                               self.d_type)

        with self.tik_inst.if_scope(self.reg.grid_cur_num_aln * 2 // self.vmask > 0):
            self.tik_inst.vadds(self.vmask, self.ub.ixy_fp, self.ub.grid_ub, 1.0,
                                self.reg.grid_cur_num_aln * 2 // self.vmask, 1, 1, 8, 8)
        with self.tik_inst.if_scope(self.reg.grid_cur_num_aln * 2 % self.vmask > 0):
            self.tik_inst.vadds(self.reg.grid_cur_num_aln * 2 % self.vmask,
                                self.ub.ixy_fp[self.reg.grid_cur_num_aln * 2 // self.vmask * self.vmask],
                                self.ub.grid_ub[self.reg.grid_cur_num_aln * 2 // self.vmask * self.vmask],
                                1.0, 1, 1, 1, 8, 8)
        self._unnormalize()
        self._compute_val(n)

    def _compute_one_loop(self, i_n, i_hw, hw_offset_cur_loop, hw_loop_times, hw_num_cur_core):
        self.reg.grid_hw_offset.set_as(hw_offset_cur_loop)
        self.reg.grid_cur_num.set_as(self.grid_ub_num)
        with self.tik_inst.if_scope(tik.all(hw_num_cur_core % self.grid_ub_num > 0, i_hw == hw_loop_times - 1)):
            self.reg.grid_cur_num.set_as(hw_num_cur_core % self.grid_ub_num)
        hw_offset_cur_loop.set_as(hw_offset_cur_loop + self.reg.grid_cur_num)

        self.reg.grid_cur_num_aln.set_as(util_common.div_align_scalar(self.reg.grid_cur_num, self.d_num_1block))
        self.reg.grid_rep_times.set_as(util_common.ceil_div_scalar(self.reg.grid_cur_num_aln, self.vmask))

        self._compute_hw(i_n)

    def _compute_n(self, core_id, i_n, core_num_cur_group):
        hw_loop_times = self.tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="hw_loop_times")
        hw_num_cur_core = self.tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="hw_num_cur_core")
        hw_offset_cur_loop = self.tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64,
                                                  name="hw_offset_cur_loop", init_value=0)

        with self.tik_inst.if_scope(self.args.tiling_mode == Gs2Constant.TILING_MODE_2):
            hw_num_cur_core.set_as(ceil_div(self.shape.grid_hw, core_num_cur_group))
            hw_offset_cur_loop.set_as(core_id % core_num_cur_group * hw_num_cur_core)
            with self.tik_inst.if_scope(tik.all(self.shape.grid_hw % core_num_cur_group > 0,
                                        core_id % core_num_cur_group >= self.shape.grid_hw % core_num_cur_group)):
                hw_num_cur_core.set_as(self.shape.grid_hw // core_num_cur_group)
                hw_offset_cur_loop.set_as(core_id % core_num_cur_group * hw_num_cur_core +
                                          self.shape.grid_hw % core_num_cur_group)
            hw_loop_times.set_as(ceil_div(hw_num_cur_core, self.grid_ub_num))
        with self.tik_inst.else_scope():
            hw_num_cur_core.set_as(ceil_div(self.shape.grid_hw, self.args.need_core_num))
            hw_offset_cur_loop.set_as(core_id * hw_num_cur_core)
            with self.tik_inst.if_scope(tik.all(self.shape.grid_hw % self.args.need_core_num > 0,
                                                core_id >= self.shape.grid_hw % self.args.need_core_num)):
                hw_num_cur_core.set_as(self.shape.grid_hw // self.args.need_core_num)
                hw_offset_cur_loop.set_as(core_id * hw_num_cur_core + self.shape.grid_hw % self.args.need_core_num)
            hw_loop_times.set_as(ceil_div(hw_num_cur_core, self.grid_ub_num))

        # initial offset factor
        self.ub.gather_offset_factor[0].set_as(0)
        self.ub.gather_offset_factor[1].set_as(8)
        self.ub.gather_offset_factor[2].set_as(16)
        self.ub.gather_offset_factor[3].set_as(24)
        self.ub.gather_offset_factor[4].set_as(0)
        self.ub.gather_offset_factor[5].set_as(8)
        self.ub.gather_offset_factor[6].set_as(16)
        self.ub.gather_offset_factor[7].set_as(24)

        # cache x in ub
        self.tik_inst.data_move(self.ub.x_cache_ub, self.gm.x_gm[i_n * self.shape.in_chw], 0, 1,
                                (self.shape.in_chw * self.d_size + 31) // 32, 0, 0)  # c eq 32, no need DataMovePad

        # Assert `hw_loop_times > 0`
        with self.tik_inst.for_range(0, hw_loop_times) as i_hw:
            self._compute_one_loop(i_n, i_hw, hw_offset_cur_loop, hw_loop_times, hw_num_cur_core)

    def _compute_one_core(self, core_id):
        n_start = self.tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="n_start")
        n_num = self.tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="n_num")
        core_num_cur_group = self.tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="core_num_cur_group")

        # Assert: `need_core_num > 0`
        with self.tik_inst.if_scope(tik.all(self.args.need_core_num % self.shape.in_n == 0,
                                              self.shape.grid_hw >= self.d_num_1block * self.args.need_core_num)):
            # Case 1: `1 == N                              and HW >= xxx`
            # Case 2: `1 <  N <= core_num and N | core_num and HW >= xxx`
            # Solution: split core to some group, compute n on one group
            #           grid_hw are compute on one core or more than one core
            self.args.tiling_mode.set_as(Gs2Constant.TILING_MODE_2)
            n_num.set_as(1)
            core_num_cur_group.set_as(self.args.need_core_num // self.shape.in_n)
            n_start.set_as(core_id // core_num_cur_group)
        with self.tik_inst.else_scope():
            # Case: `grid_hw >= 1024 * core_num`
            # Solution: split core by grid_hw
            #           N will be traversed when compute grid_hw on one core
            self.args.tiling_mode.set_as(Gs2Constant.TILING_MODE_4)
            n_num.set_as(self.shape.in_n)
            n_start.set_as(0)

        # Assert `n_num > 0`
        with self.tik_inst.for_range(n_start, n_start + n_num) as i_n:
            self._compute_n(core_id, i_n, core_num_cur_group)
