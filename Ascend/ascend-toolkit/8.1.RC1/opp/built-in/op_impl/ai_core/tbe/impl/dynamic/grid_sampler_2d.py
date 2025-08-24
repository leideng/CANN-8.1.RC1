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

grid_sampler_2d
"""
# 'pylint: disable=too-many-lines
from impl import constant_util as constant
from impl.dynamic.grid_sampler_2d_for_mini_cihiw import GridSampler2D4MiniCihiw
from impl.util import util_common
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.util.util_tik_comm_func import ceil_align
from impl.util.util_tik_comm_func import ceil_div
from impl.util.util_tik_comm_func import floor_align


# 'pylint: disable=unused-argument,too-many-arguments,huawei-too-many-arguments,too-many-return-statements
# 'pylint: disable=too-many-locals
def op_select_format(x, grid, y, interpolation_mode="bilinear", padding_mode="zeros", align_corners=False,
                     kernel_name="grid_sampler_2d"):
    x_format = shape_util.shape_to_list(x.get("ori_format"))
    x_shape = shape_util.shape_to_list(x.get("ori_shape"))
    input_c = x_shape[1]
    if x_format == "NHWC":
        input_c = x_shape[3]
    if input_c == 1:
        dtype_list = ("float32", "float32")
        x_format_list = ("NCHW", "NHWC")
        grid_format_list = ("ND", "ND")
        input0 = gen_param(classify="input0",
                           name="x",
                           datatype=",".join(dtype_list),
                           format=",".join(x_format_list),
                           unknownshape_format=",".join(x_format_list))
        input1 = gen_param(classify="input1",
                           name="grid",
                           datatype=",".join(dtype_list),
                           format=",".join(grid_format_list),
                           unknownshape_format=",".join(grid_format_list))
        output0 = gen_param(classify="output0",
                            name="y",
                            datatype=",".join(dtype_list),
                            format=",".join(x_format_list),
                            unknownshape_format=",".join(x_format_list))
    else:
        dtype_list = ("float32",)
        x_format_list = ("NHWC",)
        grid_format_list = ("ND",)
        input0 = gen_param(classify="input0",
                           name="x",
                           datatype=",".join(dtype_list),
                           format=",".join(x_format_list),
                           unknownshape_format=",".join(x_format_list))
        input1 = gen_param(classify="input1",
                           name="grid",
                           datatype=",".join(dtype_list),
                           format=",".join(grid_format_list),
                           unknownshape_format=",".join(grid_format_list))
        output0 = gen_param(classify="output0",
                            name="y",
                            datatype=",".join(dtype_list),
                            format=",".join(x_format_list),
                            unknownshape_format=",".join(x_format_list))

    param_list = [input0, input1, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


# 'pylint: disable=unused-argument,too-many-arguments,too-many-return-statements,huawei-too-many-arguments
def check_supported(x, grid, y, interpolation_mode="bilinear", padding_mode="zeros", align_corners=False,
                    kernel_name="grid_sampler_2d"):
    """
    Parameters
    ----------
    x : dict. shape and dtype of input data x
    grid : dict. shape and dtype of input data grid
    y : dict. shape and dtype of input data y
    interpolation_mode : value of attr interpolation_mode
    padding_mode : value of attr padding_mode
    align_corners : value of attr align_corners
    kernel_name : str. cce kernel name, default value is "grid_sampler_2d"

    Returns
    -------
    True or False
    """
    ret1, msg1 = _validate_support_check(x, grid, y, interpolation_mode, padding_mode, align_corners)
    if not ret1:
        return ret1, msg1

    ret2, msg2 = _check_support_static_mini_ih_iw(x, grid, y, interpolation_mode, padding_mode, align_corners)
    if ret2:
        return ret2, msg2

    return _check_support_general(x, grid, y, interpolation_mode, padding_mode, align_corners)


# 'pylint: disable=unused-argument,too-many-arguments,huawei-too-many-arguments
def _check_support_static_mini_ih_iw(x, grid, y, interpolation_mode, padding_mode, align_corners):
    """
    IH * IW can be move to ub

    Parameters
    ----------
    x : dict. shape and dtype of input data x
    grid : dict. shape and dtype of input data grid
    y : dict. shape and dtype of input data y
    interpolation_mode : value of attr interpolation_mode
    padding_mode : value of attr padding_mode
    align_corners : value of attr align_corners

    Returns
    -------
    True or False
    """
    if not tbe_platform.api_check_support("tik.vgather"):
        return False, "[case mini IH*IW]no support for gather in UB"

    x_shape = x.get("shape")
    if len(x_shape) != 4:
        return False, "[static case mini IH*IW]x shape length should be 4"
    input_c, input_h, input_w = x_shape[1], x_shape[2], x_shape[3]
    x_format = x.get("format").upper()
    if x_format == "NHWC":
        input_h, input_w, input_c = x_shape[1], x_shape[2], x_shape[3]
    if any((input_h == -1, input_w == -1)):
        return False, "[static case mini IH*IW]no support for unkown input H or input W"
    d_type = x.get("dtype").lower()
    d_size = tbe_platform.get_bit_len(d_type) // constant.DATA_SIZE_EIGHT
    if any((input_c != 1, input_c * input_h * input_w * 4 > 64 * 1024)):
        return False, "[static case mini IH*IW]only support C == 1, and C * IH * IW * data_size less than 64KB"

    if interpolation_mode != Gs2Constant.INTERPOLATION_MODE_BILINEAR:
        return False, "[static case mini IH*IW]interpolation_mode only support bilinear"

    if padding_mode not in ('zeros', 'border', 'reflection'):
        return False, "[static case mini IH*IW]padding_mode only support zeros and border and reflection "

    return True, ""


# 'pylint: disable=unused-argument,too-many-arguments,huawei-too-many-arguments
def _check_support_general(x, grid, y, interpolation_mode, padding_mode, align_corners):
    """
    x format is NHWC

    Parameters
    ----------
    x : dict. shape and dtype of input data x
    grid : dict. shape and dtype of input data grid
    y : dict. shape and dtype of input data y
    interpolation_mode : value of attr interpolation_mode
    padding_mode : value of attr padding_mode
    align_corners : value of attr align_corners

    Returns
    -------
    True or False
    """
    if not tbe_platform.api_check_support("tik.vgather"):
        return False, "No support for this platform"

    if interpolation_mode not in (Gs2Constant.INTERPOLATION_MODE_BILINEAR, Gs2Constant.INTERPOLATION_MODE_BICUBIC):
        return False, "interpolation_mode only support bilinear or bicubic"

    return True, ""


# 'pylint: disable=unused-argument,too-many-arguments,huawei-too-many-arguments
def _validate_support_check(x, grid, y, interpolation_mode, padding_mode, align_corners):
    x_format = x.get("format").upper()
    if x_format not in ("NCHW", "NHWC"):
        return False, "only support NCHW or NHWC for x"

    x_dtype = x.get("dtype").lower()
    grid_dtype = grid.get("dtype").lower()
    y_dtype = y.get("dtype").lower()
    if any((x_dtype not in ("float32"), grid_dtype != x_dtype, y_dtype != x_dtype)):
        return False, "only support float32"

    if padding_mode not in ('zeros', 'border', 'reflection'):
        return False, "padding_mode only support zeros、border or reflection"

    return True, ""


# 'pylint: disable=old-style-class,no-init,too-few-public-methods
class Gs2Constant:
    UNROLL_NUM = 16
    TILING_ARG_NUM = 16
    B32_MASK = 64

    X_UB_SIZE_4_MINI_IH_IW = 65536    # 64KB
    X_UB_SIZE_4_MINI_IH_IW_FP16 = 32768    # 32KB
    GRID_UB_SIZE_4_MINI_IH_IW = 4096  # 8KB <- MUST keep the same value with tiling

    X_UB_SIZE_4_GENERAL = 16384  # 16KB
    GRID_UB_SIZE_4_GENERAL = 2048   # 2KB
    OUT_VAL_NUM = 4096  # 4KB
    X_UB_OFFSET = 512

    INTERPOLATION_MODE_BILINEAR = 'bilinear'
    INTERPOLATION_MODE_NEAREST = 'nearest'
    INTERPOLATION_MODE_BICUBIC = 'bicubic'
    PADDING_MODE_ZEROS = 'zeros'
    PADDING_MODE_BORDER = 'border'
    PADDING_MODE_REFLECTION = 'reflection'

    TILING_MODE_1 = 1
    TILING_MODE_2 = 2
    TILING_MODE_3 = 3
    TILING_MODE_4 = 4


# 'pylint: disable=old-style-class,too-few-public-methods,too-many-instance-attributes
class Params:
    # 'pylint: disable=unused-argument,too-many-arguments,huawei-too-many-arguments
    def __init__(self, x, grid, y, interpolation_mode, padding_mode, align_corners):
        x_shape = x.get("shape")
        self.in_n = x_shape[0]
        self.in_c = x_shape[1]
        self.in_h = x_shape[2]
        self.in_w = x_shape[3]

        x_format = x.get("format").upper()
        if x_format == "NHWC":
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
class Ub4MiniIhIw:
    def __init__(self, tik_inst, d_type):
        d_size = tbe_platform.get_bit_len(d_type) // constant.DATA_SIZE_EIGHT
        x_ub_num = Gs2Constant.X_UB_SIZE_4_MINI_IH_IW // 4
        grid_ub_num = Gs2Constant.GRID_UB_SIZE_4_MINI_IH_IW // 4
        clip_mask_ub_num = grid_ub_num // constant.SIZE_SIXTEEN
        d_num_1block = constant.MAX_BLOCK_NUMBER // d_size

        self.x_ub = tik_inst.Tensor(d_type, (x_ub_num,),
                                    name="x_ub", scope=tik.scope_ubuf)              # 64KB
        self.grid_ub = tik_inst.Tensor(d_type, (grid_ub_num * 2,),
                                       name="grid_ub", scope=tik.scope_ubuf)        # 16KB
        self.y_ub = tik_inst.Tensor(d_type, (grid_ub_num,),
                                    name="y_ub", scope=tik.scope_ubuf)             # 8KB

        self.int_tmp = tik_inst.Tensor(constant.DATA_TYPE_INT32, (grid_ub_num,),
                                       name="int_tmp", scope=tik.scope_ubuf)    # 8KB
        if d_type == "float16":
            self.xx_fp_tmp = tik_inst.Tensor("float32", (grid_ub_num,),
                                             name="xx_fp_tmp", scope=tik.scope_ubuf)    # 8KB
            self.fxy_fp_tmp = tik_inst.Tensor("float32", (grid_ub_num * 3,),
                                              name="fxy_fp_tmp", scope=tik.scope_ubuf)    # 24KB
        elif d_type == "float32":
            self.ixy_fp_tmp = tik_inst.Tensor(d_type, (grid_ub_num * 3,),
                                              name="ixy_fp_tmp", scope=tik.scope_ubuf)    # 24KB

        self.ixy_int = tik_inst.Tensor(constant.DATA_TYPE_INT32, (grid_ub_num * 2,),
                                       name="ixy_int", scope=tik.scope_ubuf)        # 16KB
        self.ixy_fp = tik_inst.Tensor(d_type, (grid_ub_num * 2,),
                                      name="ixy_fp", scope=tik.scope_ubuf)          # 16KB

        self.clip_mask1 = tik_inst.Tensor(constant.DATA_TYPE_UINT16, (clip_mask_ub_num,),
                                          name="clip_mask1", scope=tik.scope_ubuf)  # 1KB
        self.clip_mask2 = tik_inst.Tensor(constant.DATA_TYPE_UINT16, (clip_mask_ub_num,),
                                          name="clip_mask2", scope=tik.scope_ubuf)  # 1KB

        self.weight = tik_inst.Tensor(d_type, (grid_ub_num * 2,),
                                      name="weight", scope=tik.scope_ubuf)          # 16KB
        self.out_val = tik_inst.Tensor(d_type, (grid_ub_num,),
                                       name="out_val", scope=tik.scope_ubuf)        # 8KB
        self.tail_block = tik_inst.Tensor(d_type, (d_num_1block * 8,),
                                          name="tail_block", scope=tik.scope_ubuf)  # 256B
        self.clip_mask_eq_mini = tik_inst.Tensor(constant.DATA_TYPE_UINT64, (4,),
                                            name="clip_mask_eq", scope=tik.scope_ubuf)   # 32B
        self.clip_mask_ne_mini = tik_inst.Tensor(constant.DATA_TYPE_UINT64, (4,),
                                            name="clip_mask_ne", scope=tik.scope_ubuf)   # 32B


# 'pylint: disable=old-style-class,too-few-public-methods,too-many-instance-attributes,too-many-statements
class Ub4General:
    def __init__(self, tik_inst, d_type, interpolation_mode):
        d_size = tbe_platform.get_bit_len(d_type) // constant.DATA_SIZE_EIGHT
        x_ub_num = Gs2Constant.X_UB_SIZE_4_GENERAL // 4
        grid_ub_num = Gs2Constant.GRID_UB_SIZE_4_GENERAL // 4
        clip_mask_ub_num = grid_ub_num // constant.SIZE_SIXTEEN
        d_num_1block = constant.MAX_BLOCK_NUMBER // d_size
        self.x_ub_offset = (Gs2Constant.X_UB_OFFSET * 2 if d_type == "float16" else Gs2Constant.X_UB_OFFSET)

        self.x_ub = tik_inst.Tensor(d_type, (x_ub_num,),
                                    name="x_ub", scope=tik.scope_ubuf)              # 16KB
        self.grid_ub = tik_inst.Tensor(d_type, (grid_ub_num * 2,),
                                       name="grid_ub", scope=tik.scope_ubuf)        # 4KB
        self.y_ub = tik_inst.Tensor(d_type, (grid_ub_num,),
                                    name="y_ub", scope=tik.scope_ubuf)              # 2KB
        self.ixy_fp = tik_inst.Tensor(d_type, (grid_ub_num * 2,),
                                      name="ixy_fp", scope=tik.scope_ubuf)          # 2KB
        self.ix_nw_sw_int = tik_inst.Tensor(constant.DATA_TYPE_INT32, (grid_ub_num,),
                                            name="ix_nw_sw_int", scope=tik.scope_ubuf)        # 2KB
        self.iy_nw_sw_int = tik_inst.Tensor(constant.DATA_TYPE_INT32, (grid_ub_num,),
                                            name="iy_nw_sw_int", scope=tik.scope_ubuf)        # 2KB
        self.ix_nw_sw_fp = tik_inst.Tensor(d_type, (grid_ub_num,),
                                           name="ix_nw_sw_fp", scope=tik.scope_ubuf)          # 2KB
        self.iy_nw_sw_fp = tik_inst.Tensor(d_type, (grid_ub_num,),
                                           name="iy_nw_sw_fp", scope=tik.scope_ubuf)          # 2KB

        self.ix_ne_se_int = tik_inst.Tensor(constant.DATA_TYPE_INT32, (grid_ub_num,),
                                            name="ix_ne_se_int", scope=tik.scope_ubuf)        # 2KB
        self.iy_ne_se_int = tik_inst.Tensor(constant.DATA_TYPE_INT32, (grid_ub_num,),
                                            name="iy_ne_se_int", scope=tik.scope_ubuf)        # 2KB
        self.ix_ne_se_fp = tik_inst.Tensor(d_type, (grid_ub_num,),
                                           name="ix_ne_se_fp", scope=tik.scope_ubuf)          # 2KB
        self.iy_ne_se_fp = tik_inst.Tensor(d_type, (grid_ub_num,),
                                           name="iy_ne_se_fp", scope=tik.scope_ubuf)          # 2KB
        if interpolation_mode == Gs2Constant.INTERPOLATION_MODE_BICUBIC:
            self.cubic_tx = tik_inst.Tensor(d_type, (grid_ub_num,),
                                            name="cubic_tx", scope=tik.scope_ubuf)            # 2KB
            self.cubic_ty = tik_inst.Tensor(d_type, (grid_ub_num,),
                                            name="cubic_ty", scope=tik.scope_ubuf)            # 2KB
            self.coeff_tx0 = tik_inst.Tensor(d_type, (grid_ub_num,),
                                             name="coeff_tx0", scope=tik.scope_ubuf)          # 2KB
            self.coeff_tx1 = tik_inst.Tensor(d_type, (grid_ub_num,),
                                             name="coeff_tx1", scope=tik.scope_ubuf)          # 2KB
            self.coeff_tx2 = tik_inst.Tensor(d_type, (grid_ub_num,),
                                             name="coeff_tx2", scope=tik.scope_ubuf)          # 2KB
            self.coeff_tx3 = tik_inst.Tensor(d_type, (grid_ub_num,),
                                             name="coeff_tx3", scope=tik.scope_ubuf)          # 2KB
            self.coeff_ty0 = tik_inst.Tensor(d_type, (grid_ub_num,),
                                             name="coeff_ty0", scope=tik.scope_ubuf)          # 2KB
            self.coeff_ty1 = tik_inst.Tensor(d_type, (grid_ub_num,),
                                             name="coeff_ty1", scope=tik.scope_ubuf)          # 2KB
            self.coeff_ty2 = tik_inst.Tensor(d_type, (grid_ub_num,),
                                             name="coeff_ty2", scope=tik.scope_ubuf)          # 2KB
            self.coeff_ty3 = tik_inst.Tensor(d_type, (grid_ub_num,),
                                             name="coeff_ty3", scope=tik.scope_ubuf)          # 2KB
            self.coor_x00 = tik_inst.Tensor(constant.DATA_TYPE_INT32, (grid_ub_num,),
                                            name="coor_x00", scope=tik.scope_ubuf)            # 2KB
            self.coor_x01 = tik_inst.Tensor(constant.DATA_TYPE_INT32, (grid_ub_num,),
                                            name="coor_x01", scope=tik.scope_ubuf)            # 2KB
            self.coor_x02 = tik_inst.Tensor(constant.DATA_TYPE_INT32, (grid_ub_num,),
                                            name="coor_x02", scope=tik.scope_ubuf)            # 2KB
            self.coor_x03 = tik_inst.Tensor(constant.DATA_TYPE_INT32, (grid_ub_num,),
                                            name="coor_x03", scope=tik.scope_ubuf)            # 2KB
            self.clip_mask00 = tik_inst.Tensor(d_type, (grid_ub_num,),
                                               name="clip_mask00", scope=tik.scope_ubuf)      # 2KB
            self.clip_mask01 = tik_inst.Tensor(d_type, (grid_ub_num,),
                                               name="clip_mask01", scope=tik.scope_ubuf)      # 2KB
            self.clip_mask02 = tik_inst.Tensor(d_type, (grid_ub_num,),
                                               name="clip_mask02", scope=tik.scope_ubuf)      # 2KB
            self.clip_mask03 = tik_inst.Tensor(d_type, (grid_ub_num,),
                                               name="clip_mask03", scope=tik.scope_ubuf)      # 2KB
            self.val_x0 = tik_inst.Tensor(d_type, (x_ub_num,),
                                          name="val_x0", scope=tik.scope_ubuf)        # 16KB
            self.val_x1 = tik_inst.Tensor(d_type, (x_ub_num,),
                                          name="val_x1", scope=tik.scope_ubuf)        # 16KB
            self.val_x2 = tik_inst.Tensor(d_type, (x_ub_num,),
                                          name="val_x2", scope=tik.scope_ubuf)        # 16KB
            self.val_x3 = tik_inst.Tensor(d_type, (x_ub_num,),
                                          name="val_x3", scope=tik.scope_ubuf)        # 16KB
            self.val_x4 = tik_inst.Tensor(d_type, (x_ub_num,),
                                          name="val_x4", scope=tik.scope_ubuf)        # 16KB

        self.clip_mask1 = tik_inst.Tensor(constant.DATA_TYPE_UINT16, (clip_mask_ub_num,),
                                          name="clip_mask1", scope=tik.scope_ubuf)    # 128B
        self.clip_mask2 = tik_inst.Tensor(constant.DATA_TYPE_UINT16, (clip_mask_ub_num,),
                                          name="clip_mask2", scope=tik.scope_ubuf)    # 128B
        self.clip_mask_eq = tik_inst.Tensor(constant.DATA_TYPE_UINT64, (4,),
                                            name="clip_mask_eq", scope=tik.scope_ubuf)      # 32B
        self.clip_mask_ne = tik_inst.Tensor(constant.DATA_TYPE_UINT64, (4,),
                                            name="clip_mask_ne", scope=tik.scope_ubuf)      # 32B

        self.xx = tik_inst.Tensor(constant.DATA_TYPE_INT32, (grid_ub_num,),
                                  name="xx", scope=tik.scope_ubuf)          # 2KB
        self.yy = tik_inst.Tensor(constant.DATA_TYPE_INT32, (grid_ub_num,),
                                  name="yy", scope=tik.scope_ubuf)          # 2KB

        self.weight = tik_inst.Tensor(d_type, (grid_ub_num,),
                                      name="weight", scope=tik.scope_ubuf)  # 2KB
        self.nw = tik_inst.Tensor(d_type, (grid_ub_num,),
                                  name="nw", scope=tik.scope_ubuf)          # 2KB
        self.ne = tik_inst.Tensor(d_type, (grid_ub_num,),
                                  name="ne", scope=tik.scope_ubuf)          # 2KB
        self.sw = tik_inst.Tensor(d_type, (grid_ub_num,),
                                  name="sw", scope=tik.scope_ubuf)          # 2KB
        self.se = tik_inst.Tensor(d_type, (grid_ub_num,),
                                  name="se", scope=tik.scope_ubuf)          # 2KB
        self.out_val = tik_inst.Tensor(d_type, (x_ub_num,),
                                       name="out_val", scope=tik.scope_ubuf)        # 16KB

        if d_type == "float16":
            self.weight_fp32 = tik_inst.Tensor("float32", (grid_ub_num,),
                                      name="weight_fp32", scope=tik.scope_ubuf)  # 2KB
            self.nw_fp32 = tik_inst.Tensor("float32", (grid_ub_num,),
                                    name="nw_fp32", scope=tik.scope_ubuf)          # 2KB
            self.ne_fp32 = tik_inst.Tensor("float32", (grid_ub_num,),
                                    name="ne_fp32", scope=tik.scope_ubuf)          # 2KB
            self.sw_fp32 = tik_inst.Tensor("float32", (grid_ub_num,),
                                    name="sw_fp32", scope=tik.scope_ubuf)          # 2KB
            self.se_fp32 = tik_inst.Tensor("float32", (grid_ub_num,),
                                    name="se_fp32", scope=tik.scope_ubuf)          # 2KB
            self.ix_nw_sw_fp32 = tik_inst.Tensor("float32", (grid_ub_num,),
                                           name="ix_nw_sw_fp32", scope=tik.scope_ubuf) 
            self.iy_nw_sw_fp32 = tik_inst.Tensor("float32", (grid_ub_num,),
                                            name="iy_nw_sw_fp32", scope=tik.scope_ubuf)
            self.ix_ne_se_fp32 = tik_inst.Tensor("float32", (grid_ub_num,),
                                           name="ix_nw_sw_fp32", scope=tik.scope_ubuf) 
            self.iy_ne_se_fp32 = tik_inst.Tensor("float32", (grid_ub_num,),
                                            name="iy_nw_sw_fp32", scope=tik.scope_ubuf)
            self.x_ub_temp = tik_inst.Tensor("float32", (x_ub_num,),
                                             name="x_ub_temp", scope=tik.scope_ubuf)   
            self.y_ub_temp = tik_inst.Tensor("float32", (x_ub_num,),
                                             name="y_ub_temp", scope=tik.scope_ubuf) 
        self.tail_block = tik_inst.Tensor(d_type, (d_num_1block * d_num_1block,),
                                        name="tail_block", scope=tik.scope_ubuf)  # 256B


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
        self.in_c_fp16 = tik_inst.Scalar(dtype=constant.DATA_TYPE_FP16, name="in_c_fp16")
        self.in_h_fp16 = tik_inst.Scalar(dtype=constant.DATA_TYPE_FP16, name="in_h_fp16")
        self.in_w_fp16 = tik_inst.Scalar(dtype=constant.DATA_TYPE_FP16, name="in_w_fp16")
        tik_inst.scalar_conv('', self.in_c_fp32, self.in_c_int32)
        tik_inst.scalar_conv('', self.in_h_fp32, self.in_h_int32)
        tik_inst.scalar_conv('', self.in_w_fp32, self.in_w_int32)
        tik_inst.scalar_conv('', self.in_c_fp16, self.in_c_fp32)
        tik_inst.scalar_conv('', self.in_h_fp16, self.in_h_fp32)
        tik_inst.scalar_conv('', self.in_w_fp16, self.in_w_fp32)


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
    def __init__(self, tik_inst, d_type):
        self.in_hw_offset = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="in_hw_offset")
        self.grid_hw_offset = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="grid_hw_offset")
        self.grid_cur_num = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="grid_cur_num")
        self.grid_cur_num_aln = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="grid_cur_num_aln")
        self.grid_rep_times = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="grid_rep_times")
        self.grid_rep_times_int32 = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="grid_rep_times_int32")

        self.x_loc = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT32, name="x_loc")
        self.x_weight = tik_inst.Scalar(dtype=d_type, name="x_weight")
        self.x_offset = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="x_offset")

        self.y_offset = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="y_offset")
        self.mte_burst = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="mte_burst")

        self.tail_offset = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="tail_offset")
        self.tail_num = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="tail_num")
        self.tail_aln = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="tail_aln")
        self.tail_tail = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="tail_tail")

        unroll_num = Gs2Constant.UNROLL_NUM
        self.xlocs = [tik_inst.Scalar(constant.DATA_TYPE_INT64, "xlocs_" + str(i)) for i in range(unroll_num)]
        self.coeff_vals = [tik_inst.Scalar(d_type, "coeff_vals_" + str(i)) for i in range(4)]
        self.mask_vals = [tik_inst.Scalar(d_type, "mask_vals_" + str(i)) for i in range(4)]

        self.is_support_vgather = tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64,
                                                  name="is_support_vgather", init_value=0)
        if tbe_platform.api_check_support("tik.vgather"):
            self.is_support_vgather.set_as(1)


# 'pylint: disable=old-style-class,too-many-instance-attributes,too-few-public-methods
class GridSampler2DMiniIhIw:
    """
    GridSampler2DMiniIhIw op implement
    """

    # 'pylint: disable=unused-argument,too-many-arguments,huawei-too-many-arguments
    def __init__(self, x, grid, y, interpolation_mode, padding_mode, align_corners, kernel_name, tik_inst):
        self.params = Params(x, grid, y, interpolation_mode, padding_mode, align_corners)
        self.kernel_name = kernel_name
        self.tik_inst = tik_inst

        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.support_data_move_pad = tbe_platform.api_check_support("tik.data_move_pad")

        self.d_type = x.get("dtype").lower()
        self.d_size = tbe_platform.get_bit_len(self.d_type) // constant.DATA_SIZE_EIGHT
        self.d_num_1block = constant.MAX_BLOCK_NUMBER // self.d_size

        self.vmask = constant.VECTOR_BYTE_SIZE // self.d_size
        self.grid_ub_num = Gs2Constant.GRID_UB_SIZE_4_MINI_IH_IW // 4
        self.h_w_unit_num = 2048
        self.clip_mask_ub_num = self.grid_ub_num // constant.SIZE_SIXTEEN

        self.gm = Gm(self.tik_inst, self.d_type, False)
        self.shape = None
        self.args = None
        self.reg = None
        self.ub = None

    def compute(self):
        """
        compute
        """
        tiling_ub = self.tik_inst.Tensor(constant.DATA_TYPE_INT64, (constant.SIZE_SIXTEEN,),
                                         name="tiling_ub", scope=tik.scope_ubuf)
        self.tik_inst.data_move(tiling_ub, self.gm.tiling_gm, 0, 1, 4, 0, 0)         # no need DataMovePad
        self.shape = Shape(self.tik_inst, tiling_ub, self.params)
        self.args = Args(self.tik_inst, tiling_ub, self.params)
        self.reg = Reg(self.tik_inst, self.d_type)

        with self.tik_inst.for_range(0, self.args.core_num_var, block_num=self.args.core_num_var) as core_id:
            with self.tik_inst.if_scope(core_id < self.args.need_core_num):
                self.ub = Ub4MiniIhIw(self.tik_inst, self.d_type)
                self.compute_one_core(core_id)

        tbe_context.get_context().add_compile_info("vars", {"core_num": self.core_num,
                                                            "is_support_vgather": 1})
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                               inputs=[self.gm.x_gm, self.gm.grid_gm],
                               outputs=[self.gm.y_gm],
                               flowtable=(self.gm.tiling_gm,),
                               config=opt_config)

    def compute_one_core(self, core_id):
        # Must keep same policy with tiling
        with self.tik_inst.if_scope(tik.all(self.shape.in_n == 1,
                                            self.shape.grid_hw >= self.h_w_unit_num * self.args.need_core_num)):
            with self.tik_inst.if_scope(core_id == 0):
                self.reg.grid_hw_offset.set_as(0)
                self._compute_hw_on_cores(self.args.pre_num_per_core)
            with self.tik_inst.else_scope():
                self.reg.grid_hw_offset.set_as(self.args.pre_num_per_core + (core_id - 1) * self.args.post_num_per_core)
                self._compute_hw_on_cores(self.args.post_num_per_core)
        with self.tik_inst.else_scope():
            n_start = self.tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="n_start")
            n_num = self.tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="n_num")
            with self.tik_inst.if_scope(core_id < self.args.pre_core_num):
                n_start.set_as(self.args.pre_num_per_core * core_id)
                n_num.set_as(self.args.pre_num_per_core)
            with self.tik_inst.else_scope():
                n_start.set_as(self.args.post_num_per_core * core_id + self.args.pre_core_num)
                n_num.set_as(self.args.post_num_per_core)

            with self.tik_inst.if_scope(tik.all(self.shape.grid_hw < self.grid_ub_num, self.args.need_core_num > 1,
                                                core_id == self.args.need_core_num - 1)):
                self.tik_inst.vec_dup(self.d_num_1block, self.ub.tail_block, 0, 1, 8)
                self.tik_inst.data_move(self.gm.y_gm[(n_start + n_num) * self.shape.grid_hw - self.d_num_1block],
                                        self.ub.tail_block, 0, 1, 1, 0, 0)  # Address fallback, no need DataMovePad

            with self.tik_inst.for_range(n_start, n_start + n_num) as n:
                self.reg.in_hw_offset.set_as(n * self.shape.in_chw)
                with self.tik_inst.if_scope(self.shape.grid_hw >= self.grid_ub_num):
                    with self.tik_inst.for_range(0, self.shape.grid_hw // self.grid_ub_num) as i_hw:
                        self._compute_loop_align(n, i_hw)
                    with self.tik_inst.if_scope(self.shape.grid_hw % self.grid_ub_num > 0):
                        self._compute_loop_tail(n)
                with self.tik_inst.elif_scope(self.shape.grid_hw >= self.d_num_1block):
                    self._compute_32b_ge_hw_lt_xk(n_start, n_num, n)
                with self.tik_inst.else_scope():
                    self._compute_hw_lt_32b(core_id, n_start, n_num, n)

    def _simple_data_move(self, dst, src, num, d_type):
        if self.support_data_move_pad:
            self.tik_inst.data_move_pad(dst, src, 1, num * self.d_size, 0, 0)
        else:
            burst_len = (num * self.d_size + constant.BLOCK_SIZE - 1) // constant.BLOCK_SIZE
            self.tik_inst.data_move(dst, src, 0, 1, burst_len, 0, 0)

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

    def _reflect_coordinates_mini(self, coord_fp, twice_low, twice_high):
        if self.d_type == "float16":
            coord_fp_x = self.ub.xx_fp_tmp
            vmask = Gs2Constant.B32_MASK
            grid_rep_times_int = self.reg.grid_rep_times_int32
            self.tik_inst.vconv(vmask, 'none', coord_fp_x, coord_fp, grid_rep_times_int, 1, 1, 8, 4)

            temp_int = self.ub.int_tmp
            with self.tik_inst.if_scope(twice_low == twice_high):
                self.tik_inst.vec_dup(vmask, coord_fp_x, 0.0, grid_rep_times_int, 8)
            
            with self.tik_inst.else_scope():
                min_s = twice_low / 2.0
                span_s = (twice_high - twice_low) / 2.0

                min_scalar = self.tik_inst.Scalar(dtype="float32", name="min_scalar", init_value=min_s)
                span_scalar = self.tik_inst.Scalar(dtype="float32", name="span_scalar", init_value=span_s)
                span_b_scalar = self.tik_inst.Scalar(dtype="float32", name="span_b_scalar", init_value=1 / span_s)

                dupl_fp = self.ub.fxy_fp_tmp
                self.tik_inst.vec_dup(vmask, dupl_fp, min_scalar, grid_rep_times_int, 8)
                self.tik_inst.vsub(vmask, coord_fp_x, coord_fp_x, dupl_fp, grid_rep_times_int, 1, 1, 1, 8, 8, 8)
                self.tik_inst.vabs(vmask, coord_fp_x, coord_fp_x, grid_rep_times_int, 1, 1, 8, 8)
        
                extra_fp = self.ub.fxy_fp_tmp[self.grid_ub_num]
                self.tik_inst.vmuls(vmask, extra_fp, coord_fp_x, span_b_scalar, grid_rep_times_int, 1, 1, 8, 8)
                self.tik_inst.vconv(vmask, 'floor', temp_int, extra_fp, grid_rep_times_int, 1, 1, 8, 8)
                self.tik_inst.vconv(vmask, 'none', extra_fp, temp_int, grid_rep_times_int, 1, 1, 8, 8)
                
                self.tik_inst.vmuls(vmask, extra_fp, extra_fp, span_scalar, grid_rep_times_int, 1, 1, 8, 8)
                self.tik_inst.vsub(vmask, extra_fp, coord_fp_x, extra_fp, grid_rep_times_int, 1, 1, 1, 8, 8, 8)
                
                self.tik_inst.vmuls(vmask, coord_fp_x, coord_fp_x, span_b_scalar, grid_rep_times_int, 1, 1, 8, 8)
                self.tik_inst.vconv(vmask, 'floor', temp_int, coord_fp_x, grid_rep_times_int, 1, 1, 8, 8)
                self.tik_inst.vconv(vmask, 'none', coord_fp_x, temp_int, grid_rep_times_int, 1, 1, 8, 8)
              
                fmod_fp = self.ub.fxy_fp_tmp[self.grid_ub_num * 2]
                self.tik_inst.vmuls(vmask, fmod_fp, coord_fp_x, 1 / 2.0, grid_rep_times_int, 1, 1, 8, 8)
                self.tik_inst.vconv(vmask, 'floor', temp_int, fmod_fp, grid_rep_times_int, 1, 1, 8, 8)
                self.tik_inst.vconv(vmask, 'none', fmod_fp, temp_int, grid_rep_times_int, 1, 1, 8, 8)
  
                self.tik_inst.vmuls(vmask, fmod_fp, fmod_fp, 2, grid_rep_times_int, 1, 1, 8, 8)
                self.tik_inst.vsub(vmask, fmod_fp, coord_fp_x, fmod_fp, grid_rep_times_int, 1, 1, 1, 8, 8, 8)
                
                self.tik_inst.vcmpvs_eq(self.ub.clip_mask_eq_mini, fmod_fp, 0.0, grid_rep_times_int, 1, 8)
                ub_mask_eq = self.ub.clip_mask_eq_mini.reinterpret_cast_to('uint64')
                high_mask_eq = self.tik_inst.Scalar(dtype='uint64', name='high_mask_eq', init_value=0)
                low_mask_eq = self.tik_inst.Scalar(dtype='uint64', name='low_mask_eq', init_value=0)
                with self.tik_inst.for_range(0, grid_rep_times_int) as i_loc:
                    low_mask_eq.set_as(ub_mask_eq[i_loc])
                    self.tik_inst.vadds([high_mask_eq, low_mask_eq], coord_fp_x[i_loc * vmask],
                                        extra_fp[i_loc * vmask], min_scalar, 1, 1, 1, 8, 8)

                self.tik_inst.vcmpvs_ne(self.ub.clip_mask_ne_mini, fmod_fp, 0.0, grid_rep_times_int, 1, 8)
                ub_mask_ne = self.ub.clip_mask_ne_mini.reinterpret_cast_to('uint64')
                high_mask_ne = self.tik_inst.Scalar(dtype='uint64', name='high_mask_ne', init_value=0)
                low_mask_ne = self.tik_inst.Scalar(dtype='uint64', name='low_mask_ne', init_value=0)
                init_scalar = self.tik_inst.Scalar(dtype='float32', name='init_scalar', init_value=-1)
                with self.tik_inst.for_range(0, grid_rep_times_int) as j_loc:
                    low_mask_ne.set_as(ub_mask_ne[j_loc])
                    self.tik_inst.vmuls([high_mask_ne, low_mask_ne], extra_fp[j_loc * vmask],
                                        extra_fp[j_loc * vmask], init_scalar, 1, 1, 1, 8, 8)
                    self.tik_inst.vadds([high_mask_ne, low_mask_ne], extra_fp[j_loc * vmask],
                                        extra_fp[j_loc * vmask], span_scalar, 1, 1, 1, 8, 8)
                    self.tik_inst.vadds([high_mask_ne, low_mask_ne], coord_fp_x[j_loc * vmask],
                                        extra_fp[j_loc * vmask], min_scalar, 1, 1, 1, 8, 8)
            self.tik_inst.vconv(vmask, 'none', coord_fp, coord_fp_x, grid_rep_times_int, 1, 1, 4, 8)

        elif self.d_type == "float32":   
            temp_int = self.ub.int_tmp
            with self.tik_inst.if_scope(twice_low == twice_high):
                self.tik_inst.vec_dup(self.vmask, coord_fp, 0.0, self.reg.grid_rep_times, 8)
            
            with self.tik_inst.else_scope():
                min_s = twice_low / 2.0
                span_s = (twice_high - twice_low) / 2.0

                min_scalar = self.tik_inst.Scalar(dtype="float32", name="min_scalar", init_value=min_s)
                span_scalar = self.tik_inst.Scalar(dtype="float32", name="span_scalar", init_value=span_s)
                span_b_scalar = self.tik_inst.Scalar(dtype="float32", name="span_b_scalar", init_value=1 / span_s)

                dupl_fp = self.ub.ixy_fp_tmp
                self.tik_inst.vec_dup(self.vmask, dupl_fp, min_scalar, self.reg.grid_rep_times, 8)
                self.tik_inst.vsub(self.vmask, coord_fp, coord_fp, dupl_fp, self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)
                self.tik_inst.vabs(self.vmask, coord_fp, coord_fp, self.reg.grid_rep_times, 1, 1, 8, 8)
        
                extra_fp = self.ub.ixy_fp_tmp[self.grid_ub_num]
                self.tik_inst.vmuls(self.vmask, extra_fp, coord_fp, span_b_scalar,
                                    self.reg.grid_rep_times, 1, 1, 8, 8)
                self.tik_inst.vconv(self.vmask, 'floor', temp_int, extra_fp,
                                    self.reg.grid_rep_times, 1, 1, 8, 8)
                self.tik_inst.vconv(self.vmask, 'none', extra_fp, temp_int,
                                    self.reg.grid_rep_times, 1, 1, 8, 8)
                
                self.tik_inst.vmuls(self.vmask, extra_fp, extra_fp, span_scalar,
                                    self.reg.grid_rep_times, 1, 1, 8, 8)
                self.tik_inst.vsub(self.vmask, extra_fp, coord_fp, extra_fp,
                                   self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)
                
                self.tik_inst.vmuls(self.vmask, coord_fp, coord_fp, span_b_scalar,
                                    self.reg.grid_rep_times, 1, 1, 8, 8)
                self.tik_inst.vconv(self.vmask, 'floor', temp_int, coord_fp,
                                    self.reg.grid_rep_times, 1, 1, 8, 8)

                self.tik_inst.vconv(self.vmask, 'none', coord_fp, temp_int,
                                    self.reg.grid_rep_times, 1, 1, 8, 8)
                        
                fmod_fp = self.ub.ixy_fp_tmp[self.grid_ub_num * 2]
                self.tik_inst.vmuls(self.vmask, fmod_fp, coord_fp, 1 / 2.0,
                                    self.reg.grid_rep_times, 1, 1, 8, 8)
                self.tik_inst.vconv(self.vmask, 'floor', temp_int, fmod_fp,
                                    self.reg.grid_rep_times, 1, 1, 8, 8)
                self.tik_inst.vconv(self.vmask, 'none', fmod_fp, temp_int,
                                    self.reg.grid_rep_times, 1, 1, 8, 8)           
                self.tik_inst.vmuls(self.vmask, fmod_fp, fmod_fp, 2, self.reg.grid_rep_times, 1, 1, 8, 8)
                self.tik_inst.vsub(self.vmask, fmod_fp, coord_fp, fmod_fp,
                                   self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)
                
                self.tik_inst.vcmpvs_eq(self.ub.clip_mask_eq_mini, fmod_fp, 0.0, self.reg.grid_rep_times, 1, 8)
                ub_mask_eq = self.ub.clip_mask_eq_mini.reinterpret_cast_to('uint64')
                self._fmod_eq_branch(ub_mask_eq, extra_fp, coord_fp, min_scalar)

                self.tik_inst.vcmpvs_ne(self.ub.clip_mask_ne_mini, fmod_fp, 0.0, self.reg.grid_rep_times, 1, 8)
                ub_mask_ne = self.ub.clip_mask_ne_mini.reinterpret_cast_to('uint64')
                self._fmod_ne_branch(ub_mask_ne, extra_fp, coord_fp, span_scalar, min_scalar)

    def _fmod_eq_branch(self, ub_mask_eq, extra_fp, coord_fp, min_scalar):
        high_mask_eq = self.tik_inst.Scalar(dtype='uint64', name='high_mask_eq', init_value=0)
        low_mask_eq = self.tik_inst.Scalar(dtype='uint64', name='low_mask_eq', init_value=0)
        with self.tik_inst.for_range(0, self.reg.grid_rep_times) as i_loc:
            low_mask_eq.set_as(ub_mask_eq[i_loc])
            with self.tik_inst.if_scope(low_mask_eq != 0):
                self.tik_inst.vadds([high_mask_eq, low_mask_eq], coord_fp[i_loc * self.vmask],
                                extra_fp[i_loc * self.vmask], min_scalar, 1, 1, 1, 8, 8)

    def _fmod_ne_branch(self, ub_mask_ne, extra_fp, coord_fp, span_scalar, min_scalar):
        high_mask_ne = self.tik_inst.Scalar(dtype='uint64', name='high_mask_ne', init_value=0)
        low_mask_ne = self.tik_inst.Scalar(dtype='uint64', name='low_mask_ne', init_value=0)
        init_scalar = self.tik_inst.Scalar(dtype='float32', name='init_scalar', init_value=-1)
        with self.tik_inst.for_range(0, self.reg.grid_rep_times) as j_loc:
            low_mask_ne.set_as(ub_mask_ne[j_loc])
            with self.tik_inst.if_scope(low_mask_ne != 0):
                self.tik_inst.vmuls([high_mask_ne, low_mask_ne], extra_fp[j_loc * self.vmask],
                                    extra_fp[j_loc * self.vmask], init_scalar, 1, 1, 1, 8, 8)
                self.tik_inst.vadds([high_mask_ne, low_mask_ne], extra_fp[j_loc * self.vmask],
                                    extra_fp[j_loc * self.vmask], span_scalar, 1, 1, 1, 8, 8)
                self.tik_inst.vadds([high_mask_ne, low_mask_ne], coord_fp[j_loc * self.vmask],
                                    extra_fp[j_loc * self.vmask], min_scalar, 1, 1, 1, 8, 8)

    def _gather_val(self):
        with self.tik_inst.if_scope(self.args.padding_mode == 2):
            with self.tik_inst.if_scope(self.args.align_corners == 1):
                self._reflect_coordinates_mini(self.ub.ixy_fp, 0, 2 * (self.shape.in_w_fp32 - 1))
                self._reflect_coordinates_mini(self.ub.ixy_fp[self.grid_ub_num], 0, 2 * (self.shape.in_h_fp32 - 1))
            with self.tik_inst.else_scope():
                self._reflect_coordinates_mini(self.ub.ixy_fp, -1, 2 * self.shape.in_w_fp32 - 1)
                self._reflect_coordinates_mini(self.ub.ixy_fp[self.grid_ub_num], -1, 2 * self.shape.in_h_fp32 - 1)
        
            with self.tik_inst.if_scope(self.d_type == "float16"):
                self.tik_inst.vconv(Gs2Constant.B32_MASK, 'floor', self.ub.ixy_int, self.ub.ixy_fp,
                                    self.reg.grid_rep_times_int32, 1, 1, 8, 4)
                self.tik_inst.vconv(Gs2Constant.B32_MASK, 'floor', self.ub.ixy_int[self.grid_ub_num],
                                    self.ub.ixy_fp[self.grid_ub_num], self.reg.grid_rep_times_int32, 1, 1, 8, 4)
            with self.tik_inst.else_scope():
                self.tik_inst.vconv(self.vmask, 'floor', self.ub.ixy_int, self.ub.ixy_fp, self.reg.grid_rep_times,
                                    1, 1, 8, 8)
                self.tik_inst.vconv(self.vmask, 'floor', self.ub.ixy_int[self.grid_ub_num],
                                    self.ub.ixy_fp[self.grid_ub_num], self.reg.grid_rep_times, 1, 1, 8, 8)
        with self.tik_inst.if_scope(self.args.padding_mode == 0):
            self.tik_inst.vcmpvs_ge(self.ub.clip_mask1, self.ub.ixy_fp, 0, self.reg.grid_rep_times, 1, 8)

            self.tik_inst.vcmpvs_ge(self.ub.clip_mask2, self.ub.ixy_fp[self.grid_ub_num], 0,
                                    self.reg.grid_rep_times, 1, 8)
            self._simple_vand(self.ub.clip_mask1, self.ub.clip_mask1, self.ub.clip_mask2,
                              constant.MASK128, self.clip_mask_ub_num)

            with self.tik_inst.if_scope(self.d_type == "float16"):
                self.tik_inst.vcmpvs_lt(self.ub.clip_mask2, self.ub.ixy_fp, self.shape.in_w_fp16,
                                        self.reg.grid_rep_times, 1, 8)
            with self.tik_inst.else_scope():
                self.tik_inst.vcmpvs_lt(self.ub.clip_mask2, self.ub.ixy_fp, self.shape.in_w_fp32,
                                        self.reg.grid_rep_times, 1, 8)
            self._simple_vand(self.ub.clip_mask1, self.ub.clip_mask1, self.ub.clip_mask2,
                              constant.MASK128, self.clip_mask_ub_num)

            with self.tik_inst.if_scope(self.d_type == "float16"):
                self.tik_inst.vcmpvs_lt(self.ub.clip_mask2, self.ub.ixy_fp[self.grid_ub_num], self.shape.in_h_fp16,
                                    self.reg.grid_rep_times, 1, 8)
            with self.tik_inst.else_scope():
                self.tik_inst.vcmpvs_lt(self.ub.clip_mask2, self.ub.ixy_fp[self.grid_ub_num], self.shape.in_h_fp32,
                                    self.reg.grid_rep_times, 1, 8)
            self._simple_vand(self.ub.clip_mask1, self.ub.clip_mask1, self.ub.clip_mask2,
                              constant.MASK128, self.clip_mask_ub_num)

        # bound value for padding_mode is 'zeros' or 'border'
        self.tik_inst.vmins(Gs2Constant.B32_MASK, self.ub.ixy_int, self.ub.ixy_int,
                            self.shape.in_w_int32 - 1, self.reg.grid_rep_times_int32, 1, 1, 8, 8)
        self.tik_inst.vmins(Gs2Constant.B32_MASK, self.ub.ixy_int[self.grid_ub_num], self.ub.ixy_int[self.grid_ub_num],
                            self.shape.in_h_int32 - 1, self.reg.grid_rep_times_int32, 1, 1, 8, 8)
        self.tik_inst.vmaxs(Gs2Constant.B32_MASK, self.ub.ixy_int, self.ub.ixy_int,
                            0, self.reg.grid_rep_times_int32, 1, 1, 8, 8)
        self.tik_inst.vmaxs(Gs2Constant.B32_MASK, self.ub.ixy_int[self.grid_ub_num], self.ub.ixy_int[self.grid_ub_num],
                            0, self.reg.grid_rep_times_int32, 1, 1, 8, 8)

        self.tik_inst.vmuls(Gs2Constant.B32_MASK, self.ub.ixy_int[self.grid_ub_num], self.ub.ixy_int[self.grid_ub_num],
                            self.shape.in_w_int32, self.reg.grid_rep_times_int32, 1, 1, 8, 8)
        self.tik_inst.vadd(Gs2Constant.B32_MASK, self.ub.ixy_int, self.ub.ixy_int, self.ub.ixy_int[self.grid_ub_num],
                           self.reg.grid_rep_times_int32, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vmuls(Gs2Constant.B32_MASK, self.ub.ixy_int, self.ub.ixy_int, self.d_size,
                            self.reg.grid_rep_times_int32, 1, 1, 8, 8)
        self.tik_inst.vgather(self.reg.grid_cur_num, self.ub.out_val, self.ub.x_ub, self.ub.ixy_int,
                              1, 8, 0, 0, 'counter')

        with self.tik_inst.if_scope(self.args.padding_mode == 0):
            self.tik_inst.vsel(self.vmask, 1, self.ub.out_val, self.ub.clip_mask1, self.ub.out_val, 0.0,
                               self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)

    def _compute_floor_floor(self):
        with self.tik_inst.if_scope(self.d_type == "float16"):
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'floor', self.ub.ixy_int, self.ub.grid_ub,
                            self.reg.grid_rep_times_int32, 1, 1, 8, 4)               # ixy_nw: floor floor co-ordinates
        with self.tik_inst.else_scope():
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'floor', self.ub.ixy_int, self.ub.grid_ub,
                            self.reg.grid_rep_times_int32, 1, 1, 8, 8)               # ixy_nw: floor floor co-ordinates

        with self.tik_inst.if_scope(self.d_type == "float16"):
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'floor', self.ub.ixy_int[self.grid_ub_num],
                        self.ub.grid_ub[self.grid_ub_num], self.reg.grid_rep_times_int32, 1, 1, 8, 4)
        with self.tik_inst.else_scope():
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'floor', self.ub.ixy_int[self.grid_ub_num],
                        self.ub.grid_ub[self.grid_ub_num], self.reg.grid_rep_times_int32, 1, 1, 8, 8)
        
        with self.tik_inst.if_scope(self.d_type == "float16"):
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.ixy_fp, self.ub.ixy_int,
                                self.reg.grid_rep_times_int32, 1, 1, 4, 8, 1.0)      # ixy_nw: floor floor co-ordinates
        with self.tik_inst.else_scope():
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.ixy_fp, self.ub.ixy_int,
                                self.reg.grid_rep_times_int32, 1, 1, 8, 8)           # ixy_nw: floor floor co-ordinates

        with self.tik_inst.if_scope(self.d_type == "float16"):
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.ixy_fp[self.grid_ub_num],
                    self.ub.ixy_int[self.grid_ub_num], self.reg.grid_rep_times_int32, 1, 1, 4, 8, 1.0)
        with self.tik_inst.else_scope():
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.ixy_fp[self.grid_ub_num],
                    self.ub.ixy_int[self.grid_ub_num], self.reg.grid_rep_times_int32, 1, 1, 8, 8)

        self.tik_inst.vsub(self.vmask, self.ub.weight, self.ub.ixy_fp, self.ub.grid_ub,
                           self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vsub(self.vmask, self.ub.weight[self.grid_ub_num],
                           self.ub.ixy_fp[self.grid_ub_num], self.ub.grid_ub[self.grid_ub_num],
                           self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadds(self.vmask, self.ub.weight, self.ub.weight, 1.0,
                            self.reg.grid_rep_times, 1, 1, 8, 8)
        self.tik_inst.vadds(self.vmask, self.ub.weight[self.grid_ub_num], self.ub.weight[self.grid_ub_num], 1.0,
                            self.reg.grid_rep_times, 1, 1, 8, 8)
        self.tik_inst.vmul(self.vmask, self.ub.weight, self.ub.weight, self.ub.weight[self.grid_ub_num],
                           self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)              # se: floor floor weight

        self._gather_val()                                                     # nw_val: floor floor val
        self.tik_inst.vmul(self.vmask, self.ub.y_ub, self.ub.out_val, self.ub.weight,
                           self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)              # nw_val * se

    def _compute_ceil_floor(self):
        with self.tik_inst.if_scope(self.d_type == "float16"):
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'floor', self.ub.ixy_int, self.ub.grid_ub,
                            self.reg.grid_rep_times_int32, 1, 1, 8, 4)               # ixy_nw: floor floor co-ordinates
        with self.tik_inst.else_scope():
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'floor', self.ub.ixy_int, self.ub.grid_ub,
                            self.reg.grid_rep_times_int32, 1, 1, 8, 8)               # ixy_nw: floor floor co-ordinates

        with self.tik_inst.if_scope(self.d_type == "float16"):
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'floor', self.ub.ixy_int[self.grid_ub_num],
                        self.ub.grid_ub[self.grid_ub_num], self.reg.grid_rep_times_int32, 1, 1, 8, 4)
        with self.tik_inst.else_scope():
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'floor', self.ub.ixy_int[self.grid_ub_num],
                        self.ub.grid_ub[self.grid_ub_num], self.reg.grid_rep_times_int32, 1, 1, 8, 8)   
        
        with self.tik_inst.if_scope(self.d_type == "float16"):
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.ixy_fp, self.ub.ixy_int,
                            self.reg.grid_rep_times_int32, 1, 1, 4, 8, 1.0)          # ixy_nw: floor floor co-ordinates
        with self.tik_inst.else_scope():
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.ixy_fp, self.ub.ixy_int,
                            self.reg.grid_rep_times_int32, 1, 1, 8, 8)               # ixy_nw: floor floor co-ordinates

        with self.tik_inst.if_scope(self.d_type == "float16"):
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.ixy_fp[self.grid_ub_num],
                    self.ub.ixy_int[self.grid_ub_num], self.reg.grid_rep_times_int32, 1, 1, 4, 8, 1.0)
        with self.tik_inst.else_scope():
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.ixy_fp[self.grid_ub_num],
                self.ub.ixy_int[self.grid_ub_num], self.reg.grid_rep_times_int32, 1, 1, 8, 8)      
        
        self.tik_inst.vadds(Gs2Constant.B32_MASK, self.ub.ixy_int, self.ub.ixy_int, 1,
                            self.reg.grid_rep_times_int32, 1, 1, 8, 8)      # ixy_ne: ceil floor co-ordinates
        self.tik_inst.vadds(self.vmask, self.ub.ixy_fp, self.ub.ixy_fp, 1.0,
                            self.reg.grid_rep_times, 1, 1, 8, 8)            # ixy_ne: ceil floor co-ordinates

        self.tik_inst.vsub(self.vmask, self.ub.weight, self.ub.grid_ub, self.ub.ixy_fp,
                           self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vsub(self.vmask, self.ub.weight[self.grid_ub_num],
                           self.ub.ixy_fp[self.grid_ub_num], self.ub.grid_ub[self.grid_ub_num],
                           self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadds(self.vmask, self.ub.weight, self.ub.weight, 1.0,
                            self.reg.grid_rep_times, 1, 1, 8, 8)
        self.tik_inst.vadds(self.vmask, self.ub.weight[self.grid_ub_num], self.ub.weight[self.grid_ub_num], 1.0,
                            self.reg.grid_rep_times, 1, 1, 8, 8)
        self.tik_inst.vmul(self.vmask, self.ub.weight, self.ub.weight, self.ub.weight[self.grid_ub_num],
                           self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)              # sw: ceil floor weight

        self._gather_val()                                                     # ne_val: ceil floor val
        self.tik_inst.vmul(self.vmask, self.ub.out_val, self.ub.out_val, self.ub.weight,
                           self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)              # ne_val * sw
        self.tik_inst.vadd(self.vmask, self.ub.y_ub, self.ub.y_ub, self.ub.out_val,
                           self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)

    def _compute_floor_ceil(self):
        with self.tik_inst.if_scope(self.d_type == "float16"):
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'floor', self.ub.ixy_int, self.ub.grid_ub,
                            self.reg.grid_rep_times_int32, 1, 1, 8, 4)     # ixy_nw: floor floor co-ordinates
        with self.tik_inst.else_scope():
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'floor', self.ub.ixy_int, self.ub.grid_ub,
                            self.reg.grid_rep_times_int32, 1, 1, 8, 8)     # ixy_nw: floor floor co-ordinates

        with self.tik_inst.if_scope(self.d_type == "float16"):
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'floor', self.ub.ixy_int[self.grid_ub_num],
                    self.ub.grid_ub[self.grid_ub_num], self.reg.grid_rep_times_int32, 1, 1, 8, 4)
        with self.tik_inst.else_scope():
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'floor', self.ub.ixy_int[self.grid_ub_num],
                    self.ub.grid_ub[self.grid_ub_num], self.reg.grid_rep_times_int32, 1, 1, 8, 8)       
        

        with self.tik_inst.if_scope(self.d_type == "float16"):
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.ixy_fp, self.ub.ixy_int,
                                self.reg.grid_rep_times_int32, 1, 1, 4, 8, 1.0)      # ixy_nw: floor floor co-ordinates
        with self.tik_inst.else_scope():
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.ixy_fp, self.ub.ixy_int,
                                self.reg.grid_rep_times_int32, 1, 1, 8, 8)           # ixy_nw: floor floor co-ordinates 

        with self.tik_inst.if_scope(self.d_type == "float16"):
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.ixy_fp[self.grid_ub_num],
                self.ub.ixy_int[self.grid_ub_num], self.reg.grid_rep_times_int32, 1, 1, 4, 8, 1.0)
        with self.tik_inst.else_scope():
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.ixy_fp[self.grid_ub_num],
                self.ub.ixy_int[self.grid_ub_num], self.reg.grid_rep_times_int32, 1, 1, 8, 8)        
        
        self.tik_inst.vadds(Gs2Constant.B32_MASK, self.ub.ixy_int[self.grid_ub_num], self.ub.ixy_int[self.grid_ub_num],
                            1, self.reg.grid_rep_times_int32, 1, 1, 8, 8)            # ixy_sw: floor ceil co-ordinates
        self.tik_inst.vadds(self.vmask, self.ub.ixy_fp[self.grid_ub_num], self.ub.ixy_fp[self.grid_ub_num],
                            1.0, self.reg.grid_rep_times, 1, 1, 8, 8)              # ixy_sw: floor ceil co-ordinates

        self.tik_inst.vsub(self.vmask, self.ub.weight, self.ub.ixy_fp, self.ub.grid_ub,
                           self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vsub(self.vmask, self.ub.weight[self.grid_ub_num],
                           self.ub.grid_ub[self.grid_ub_num], self.ub.ixy_fp[self.grid_ub_num],
                           self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadds(self.vmask, self.ub.weight, self.ub.weight, 1.0,
                            self.reg.grid_rep_times, 1, 1, 8, 8)
        self.tik_inst.vadds(self.vmask, self.ub.weight[self.grid_ub_num], self.ub.weight[self.grid_ub_num], 1.0,
                            self.reg.grid_rep_times, 1, 1, 8, 8)
        self.tik_inst.vmul(self.vmask, self.ub.weight, self.ub.weight, self.ub.weight[self.grid_ub_num],
                           self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)              # ne: floor ceil weight

        self._gather_val()                                                     # sw_val: floor ceil val
        self.tik_inst.vmul(self.vmask, self.ub.out_val, self.ub.out_val, self.ub.weight,
                           self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)              # sw_val * ne
        self.tik_inst.vadd(self.vmask, self.ub.y_ub, self.ub.y_ub, self.ub.out_val,
                           self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)

    def _compute_ceil_ceil(self):
        with self.tik_inst.if_scope(self.d_type == "float16"):
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'floor', self.ub.ixy_int, self.ub.grid_ub,
                            self.reg.grid_rep_times_int32, 1, 1, 8, 4)               # ixy_nw: floor floor co-ordinates
        with self.tik_inst.else_scope():
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'floor', self.ub.ixy_int, self.ub.grid_ub,
                            self.reg.grid_rep_times_int32, 1, 1, 8, 8)               # ixy_nw: floor floor co-ordinates

        with self.tik_inst.if_scope(self.d_type == "float16"):
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'floor', self.ub.ixy_int[self.grid_ub_num],
                 self.ub.grid_ub[self.grid_ub_num], self.reg.grid_rep_times_int32, 1, 1, 8, 4)
        with self.tik_inst.else_scope():
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'floor', self.ub.ixy_int[self.grid_ub_num],
                 self.ub.grid_ub[self.grid_ub_num], self.reg.grid_rep_times_int32, 1, 1, 8, 8)
        
        with self.tik_inst.if_scope(self.d_type == "float16"):
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.ixy_fp, self.ub.ixy_int,
                                self.reg.grid_rep_times_int32, 1, 1, 4, 8, 1.0)      # ixy_nw: floor floor co-ordinates
        with self.tik_inst.else_scope():
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.ixy_fp, self.ub.ixy_int,
                            self.reg.grid_rep_times_int32, 1, 1, 8, 8)               # ixy_nw: floor floor co-ordinates

        with self.tik_inst.if_scope(self.d_type == "float16"):
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.ixy_fp[self.grid_ub_num],
                 self.ub.ixy_int[self.grid_ub_num], self.reg.grid_rep_times_int32, 1, 1, 4, 8, 1.0)
        with self.tik_inst.else_scope():
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.ixy_fp[self.grid_ub_num],
                 self.ub.ixy_int[self.grid_ub_num], self.reg.grid_rep_times_int32, 1, 1, 8, 8) 
        
        self.tik_inst.vadds(Gs2Constant.B32_MASK, self.ub.ixy_int, self.ub.ixy_int, 1,
                            self.reg.grid_rep_times_int32, 1, 1, 8, 8)               # ixy_se: ceil ceil co-ordinates
        self.tik_inst.vadds(Gs2Constant.B32_MASK, self.ub.ixy_int[self.grid_ub_num],
                            self.ub.ixy_int[self.grid_ub_num], 1, self.reg.grid_rep_times_int32, 1, 1, 8, 8)
        self.tik_inst.vadds(self.vmask, self.ub.ixy_fp, self.ub.ixy_fp, 1.0,
                            self.reg.grid_rep_times, 1, 1, 8, 8)               # ixy_se: ceil ceil co-ordinates
        self.tik_inst.vadds(self.vmask, self.ub.ixy_fp[self.grid_ub_num], self.ub.ixy_fp[self.grid_ub_num], 1.0,
                            self.reg.grid_rep_times, 1, 1, 8, 8)               # ixy_se: ceil ceil co-ordinates

        self.tik_inst.vsub(self.vmask, self.ub.weight, self.ub.grid_ub, self.ub.ixy_fp,
                           self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vsub(self.vmask, self.ub.weight[self.grid_ub_num],
                           self.ub.grid_ub[self.grid_ub_num], self.ub.ixy_fp[self.grid_ub_num],
                           self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadds(self.vmask, self.ub.weight, self.ub.weight, 1.0,
                            self.reg.grid_rep_times, 1, 1, 8, 8)
        self.tik_inst.vadds(self.vmask, self.ub.weight[self.grid_ub_num], self.ub.weight[self.grid_ub_num], 1.0,
                            self.reg.grid_rep_times, 1, 1, 8, 8)
        self.tik_inst.vmul(self.vmask, self.ub.weight, self.ub.weight, self.ub.weight[self.grid_ub_num],
                           self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)              # nw: ceil ceil weight

        self._gather_val()                                                     # se_val: ceil ceil val
        self.tik_inst.vmul(self.vmask, self.ub.out_val, self.ub.out_val, self.ub.weight,
                           self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)              # se_val * nw
        self.tik_inst.vadd(self.vmask, self.ub.y_ub, self.ub.y_ub, self.ub.out_val,
                           self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)

    def _compute_coordinates_0(self):
        if self.params.padding_mode == Gs2Constant.PADDING_MODE_BORDER:
            self.tik_inst.vmaxs(self.vmask, self.ub.grid_ub, self.ub.grid_ub,
                                0, self.reg.grid_rep_times, 1, 1, 8, 8)
            self.tik_inst.vmaxs(self.vmask, self.ub.grid_ub[self.grid_ub_num], self.ub.grid_ub[self.grid_ub_num],
                                0, self.reg.grid_rep_times, 1, 1, 8, 8)
            with self.tik_inst.if_scope(self.d_type == "float16"):
                self.tik_inst.vmins(self.vmask, self.ub.grid_ub, self.ub.grid_ub,
                                    self.shape.in_w_fp32 - 1.0, self.reg.grid_rep_times, 1, 1, 8, 8)
                self.tik_inst.vmins(self.vmask, self.ub.grid_ub[self.grid_ub_num], self.ub.grid_ub[self.grid_ub_num],
                                    self.shape.in_h_fp32 - 1.0, self.reg.grid_rep_times, 1, 1, 8, 8)
            with self.tik_inst.else_scope():
                self.tik_inst.vmins(self.vmask, self.ub.grid_ub, self.ub.grid_ub,
                                    self.shape.in_w_fp32 - 1.0, self.reg.grid_rep_times, 1, 1, 8, 8)
                self.tik_inst.vmins(self.vmask, self.ub.grid_ub[self.grid_ub_num], self.ub.grid_ub[self.grid_ub_num],
                                    self.shape.in_h_fp32 - 1.0, self.reg.grid_rep_times, 1, 1, 8, 8)

    def _compute_grid_ub(self):
        with self.tik_inst.if_scope(self.d_type == "float16"):
            x_ub_fp32 = self.ub.x_ub.reinterpret_cast_to("float32")
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', x_ub_fp32,
                                self.ub.grid_ub, (self.grid_ub_num * 2 + 63) // 64, 1, 1, 8, 4)
            self.tik_inst.vadds(64, x_ub_fp32, x_ub_fp32,
                                1.0, self.grid_ub_num * 2 // 64, 1, 1, 8, 8)
        with self.tik_inst.else_scope():
            self.tik_inst.vadds(self.vmask, self.ub.ixy_fp, self.ub.grid_ub, 1.0,
                                self.grid_ub_num * 2 // self.vmask, 1, 1, 8, 8)
        with self.tik_inst.if_scope(self.d_type == "float16"):
            self.tik_inst.vreduce(self.grid_ub_num * 2, x_ub_fp32[4096], x_ub_fp32,
                                1, 1, 1, 8, 0, 0, None, 'counter')
            self.tik_inst.vreduce(self.grid_ub_num * 2, x_ub_fp32[4096 + self.grid_ub_num], x_ub_fp32,
                                2, 1, 1, 8, 0, 0, None, 'counter')
        with self.tik_inst.else_scope():
            self.tik_inst.vreduce(self.grid_ub_num * 2, self.ub.grid_ub, self.ub.ixy_fp,
                                1, 1, 1, 8, 0, 0, None, 'counter')
            self.tik_inst.vreduce(self.grid_ub_num * 2, self.ub.grid_ub[self.grid_ub_num], self.ub.ixy_fp,
                                2, 1, 1, 8, 0, 0, None, 'counter')

        with self.tik_inst.if_scope(self.d_type == "float16"):
            with self.tik_inst.if_scope(self.args.align_corners == 1):
                # unnormalize coord from [-1, 1] to [0, size - 1]
                self.tik_inst.vmuls(64, x_ub_fp32, x_ub_fp32[4096],
                                    0.5 * (self.shape.in_w_fp32 - 1.0), self.reg.grid_rep_times_int32,
                                    1, 1, 8, 8)                                    # ix
                self.tik_inst.vmuls(64, x_ub_fp32[self.grid_ub_num], x_ub_fp32[4096 + self.grid_ub_num],
                                    0.5 * (self.shape.in_h_fp32 - 1.0), self.reg.grid_rep_times_int32,
                                    1, 1, 8, 8)                                    # iy
                self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.grid_ub, x_ub_fp32,
                                    self.grid_ub_num * 2 // 64, 1, 1, 4, 8)
            with self.tik_inst.else_scope():
                # unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
                self.tik_inst.vmuls(64, x_ub_fp32, x_ub_fp32[4096],
                                    0.5 * self.shape.in_w_fp32, self.reg.grid_rep_times_int32,
                                    1, 1, 8, 8)                                    # ix
                self.tik_inst.vmuls(64, x_ub_fp32[self.grid_ub_num], x_ub_fp32[4096 + self.grid_ub_num],
                                    0.5 * self.shape.in_h_fp32, self.reg.grid_rep_times_int32,
                                    1, 1, 8, 8)                                    # iy
                self.tik_inst.vadds(64, x_ub_fp32, x_ub_fp32,
                                    -0.5, self.reg.grid_rep_times_int32, 1, 1, 8, 8)
                self.tik_inst.vadds(64, x_ub_fp32[self.grid_ub_num], x_ub_fp32[self.grid_ub_num],
                                    -0.5, self.reg.grid_rep_times_int32, 1, 1, 8, 8)
                self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.grid_ub, x_ub_fp32,
                                    self.grid_ub_num * 2 // 64, 1, 1, 4, 8)
        with self.tik_inst.else_scope():
            with self.tik_inst.if_scope(self.args.align_corners == 1):
                # unnormalize coord from [-1, 1] to [0, size - 1]
                self.tik_inst.vmuls(self.vmask, self.ub.grid_ub, self.ub.grid_ub,
                                    0.5 * (self.shape.in_w_fp32 - 1.0), self.reg.grid_rep_times,
                                    1, 1, 8, 8)                                    # ix
                self.tik_inst.vmuls(self.vmask, self.ub.grid_ub[self.grid_ub_num], self.ub.grid_ub[self.grid_ub_num],
                                    0.5 * (self.shape.in_h_fp32 - 1.0), self.reg.grid_rep_times,
                                    1, 1, 8, 8)                                    # iy
            with self.tik_inst.else_scope():
                # unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
                self.tik_inst.vmuls(self.vmask, self.ub.grid_ub, self.ub.grid_ub,
                                    0.5 * self.shape.in_w_fp32, self.reg.grid_rep_times,
                                    1, 1, 8, 8)                                    # ix
                self.tik_inst.vmuls(self.vmask, self.ub.grid_ub[self.grid_ub_num], self.ub.grid_ub[self.grid_ub_num],
                                    0.5 * self.shape.in_h_fp32, self.reg.grid_rep_times,
                                    1, 1, 8, 8)                                    # iy
                self.tik_inst.vadds(self.vmask, self.ub.grid_ub, self.ub.grid_ub,
                                    -0.5, self.reg.grid_rep_times, 1, 1, 8, 8)
                self.tik_inst.vadds(self.vmask, self.ub.grid_ub[self.grid_ub_num], self.ub.grid_ub[self.grid_ub_num],
                                    -0.5, self.reg.grid_rep_times, 1, 1, 8, 8)

        self._compute_coordinates_0()

        # only support C == 1
        self._simple_data_move(self.ub.x_ub, self.gm.x_gm[self.reg.in_hw_offset], self.shape.in_hw, self.d_type)
        self._compute_floor_floor()
        self._compute_ceil_floor()
        self._compute_floor_ceil()
        self._compute_ceil_ceil()

    def _compute_hw_on_cores(self, hw_num):
        """HW is greater than or equal Gs2Constant.GRID_UB_SIZE_4_MINI_IH_IW on each core"""

        def _simple_compute():
            self.reg.grid_cur_num.set_as(self.grid_ub_num)
            self.reg.grid_rep_times.set_as(self.grid_ub_num // self.vmask)
            self.reg.grid_rep_times_int32.set_as(self.grid_ub_num // Gs2Constant.B32_MASK)

            self.tik_inst.data_move(self.ub.grid_ub, self.gm.grid_gm[self.reg.grid_hw_offset * 2], 0, 1,
                                    self.grid_ub_num * 2 * self.d_size // 32, 0, 0)  # no need DataMovePad
            self._compute_grid_ub()
            self.tik_inst.data_move(self.gm.y_gm[self.reg.grid_hw_offset], self.ub.y_ub, 0, 1,
                                    self.grid_ub_num * self.d_size // 32, 0, 0)      # no need DataMovePad

        hw_start = self.tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="hw_start")
        hw_start.set_as(self.reg.grid_hw_offset)
        with self.tik_inst.for_range(0, hw_num // self.grid_ub_num) as i_hw:
            self.reg.grid_hw_offset.set_as(hw_start + i_hw * self.grid_ub_num)
            _simple_compute()
        with self.tik_inst.if_scope(hw_num % self.grid_ub_num > 0):
            self.reg.grid_hw_offset.set_as(hw_start + hw_num - self.grid_ub_num)
            _simple_compute()

    def _compute_loop_align(self, n, i_hw):
        self.reg.grid_hw_offset.set_as(n * self.shape.grid_hw + i_hw * self.grid_ub_num)
        self.reg.grid_rep_times.set_as(self.grid_ub_num // self.vmask)
        self.reg.grid_rep_times_int32.set_as(self.grid_ub_num // Gs2Constant.B32_MASK)
        self.reg.grid_cur_num.set_as(self.grid_ub_num)
        self.tik_inst.data_move(self.ub.grid_ub, self.gm.grid_gm[self.reg.grid_hw_offset * 2], 0, 1,
                                self.grid_ub_num * 2 * self.d_size // 32, 0, 0)     # no need DataMovePad
        self._compute_grid_ub()
        self.tik_inst.data_move(self.gm.y_gm[self.reg.grid_hw_offset], self.ub.y_ub, 0, 1,
                                self.grid_ub_num * self.d_size // 32, 0, 0)         # no need DataMovePad

    def _clean_ub_for_tail_compute(self):
        self.tik_inst.vec_dup(self.vmask, self.ub.grid_ub, 0, self.grid_ub_num // self.vmask, 8)
        self.tik_inst.vec_dup(self.vmask, self.ub.ixy_fp, 0, self.grid_ub_num // self.vmask, 8)
        self.tik_inst.vec_dup(self.vmask, self.ub.weight, 0, self.grid_ub_num // self.vmask, 8)
        self.tik_inst.vec_dup(self.vmask, self.ub.out_val, 0, self.reg.grid_rep_times, 8)
        self.tik_inst.vec_dup(self.vmask, self.ub.y_ub, 0, self.reg.grid_rep_times, 8)

    def _compute_loop_tail(self, n):
        self.reg.grid_cur_num.set_as(util_common.div_align_scalar(self.shape.grid_hw % self.grid_ub_num,
                                                                  self.d_num_1block))
        self.reg.grid_hw_offset.set_as((n + 1) * self.shape.grid_hw - self.reg.grid_cur_num)
        self.reg.grid_rep_times.set_as(util_common.ceil_div_scalar(self.reg.grid_cur_num, self.vmask))
        self.reg.grid_rep_times_int32.set_as(util_common.ceil_div_scalar(self.reg.grid_cur_num, Gs2Constant.B32_MASK))

        self._clean_ub_for_tail_compute()
        self.tik_inst.data_move(self.ub.grid_ub, self.gm.grid_gm[self.reg.grid_hw_offset * 2], 0, 1,
                                self.reg.grid_cur_num * 2 * self.d_size // 32, 0, 0)       # no need DataMovePad
        self._compute_grid_ub()
        self.tik_inst.data_move(self.gm.y_gm[self.reg.grid_hw_offset], self.ub.y_ub, 0, 1,
                                (self.reg.grid_cur_num * self.d_size) // 32, 0, 0)         # no need DataMovePad

    def _compute_32b_ge_hw_lt_xk(self, n_start, n_num, n):
        self.reg.grid_cur_num.set_as(self.shape.grid_hw)
        self.reg.grid_hw_offset.set_as(n * self.shape.grid_hw)
        self.reg.grid_rep_times.set_as(util_common.ceil_div_scalar(self.shape.grid_hw, self.vmask))
        self.reg.grid_rep_times_int32.set_as(util_common.ceil_div_scalar(self.shape.grid_hw, Gs2Constant.B32_MASK))

        self._clean_ub_for_tail_compute()
        self._simple_data_move(self.ub.grid_ub, self.gm.grid_gm[self.reg.grid_hw_offset * 2],
                               self.reg.grid_cur_num * 2, self.d_type)
        self._compute_grid_ub()

        with self.tik_inst.if_scope(n < n_start + n_num - 1):
            self._simple_data_move(self.gm.y_gm[self.reg.grid_hw_offset], self.ub.y_ub,
                                   self.reg.grid_cur_num, self.d_type)
        with self.tik_inst.else_scope():
            self.tik_inst.data_move(self.gm.y_gm[self.reg.grid_hw_offset], self.ub.y_ub, 0, 1,
                                    (self.reg.grid_cur_num * self.d_size) // 32, 0, 0)        # no need DataMovePad
            with self.tik_inst.if_scope(self.reg.grid_cur_num % self.d_num_1block != 0):
                for i in range(self.d_num_1block):
                    self.ub.tail_block[i].set_as(self.ub.y_ub[self.reg.grid_cur_num - self.d_num_1block + i])
                self.tik_inst.data_move(self.gm.y_gm[(n + 1) * self.shape.grid_hw - self.d_num_1block],
                                        self.ub.tail_block, 0, 1, 1, 0, 0)  # Address fallback, no need DataMovePad

    def _compute_hw_lt_32b(self, core_id, n_start, n_num, n):
        self.reg.grid_cur_num.set_as(self.shape.grid_hw)
        self.reg.grid_hw_offset.set_as(n * self.shape.grid_hw)
        self.reg.grid_rep_times.set_as(util_common.ceil_div_scalar(self.shape.grid_hw, self.vmask))
        self.reg.grid_rep_times_int32.set_as(util_common.ceil_div_scalar(self.shape.grid_hw, Gs2Constant.B32_MASK))

        self._clean_ub_for_tail_compute()
        self._simple_data_move(self.ub.grid_ub, self.gm.grid_gm[self.reg.grid_hw_offset * 2],
                               self.reg.grid_cur_num * 2, self.d_type)
        self._compute_grid_ub()

        with self.tik_inst.if_scope(tik.any(self.args.need_core_num == 1, core_id == self.args.need_core_num - 1)):
            self._simple_data_move(self.gm.y_gm[self.reg.grid_hw_offset], self.ub.y_ub,
                                   self.reg.grid_cur_num, self.d_type)
        with self.tik_inst.else_scope():
            self.tik_inst.vec_dup(self.d_num_1block, self.ub.tail_block, 0, 1, 8)
            self.tik_inst.vadds(self.shape.grid_hw, self.ub.tail_block, self.ub.y_ub, 0.0, 1, 1, 1, 8, 8)
            with self.tik_inst.if_scope(n < n_start + n_num - self.d_num_1block // self.shape.grid_hw):
                self._simple_data_move(self.gm.y_gm[self.reg.grid_hw_offset], self.ub.tail_block,
                                       self.reg.grid_cur_num, self.d_type)
            with self.tik_inst.else_scope():
                self.tik_inst.set_atomic_add(self.d_type)
                self._simple_data_move(self.gm.y_gm[self.reg.grid_hw_offset], self.ub.tail_block,
                                       self.reg.grid_cur_num, self.d_type)
                self.tik_inst.set_atomic_add(0)


# 'pylint: disable=too-few-public-methods
class GridSampler2D:
    """
    GridSampler2D op implement
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
        self.grid_ub_num = Gs2Constant.GRID_UB_SIZE_4_GENERAL // 4
        self.clip_mask_ub_num = self.grid_ub_num // constant.SIZE_SIXTEEN

        self.cubic_a = -0.75

        self.gm = Gm(self.tik_inst, self.d_type, True)
        self.shape = None
        self.args = None
        self.reg = None
        self.ub = None

        self.mini_ih_iw = GridSampler2DMiniIhIw(x, grid, y, interpolation_mode, padding_mode, align_corners,
                                                kernel_name, tik_inst)
        self.mini_ih_iw.gm = self.gm

    def compute(self):
        """
        compute
        """
        tiling_ub = self.tik_inst.Tensor(constant.DATA_TYPE_INT64, (constant.SIZE_SIXTEEN,),
                                         name="tiling_ub", scope=tik.scope_ubuf)
        self.tik_inst.data_move(tiling_ub, self.gm.tiling_gm, 0, 1, 4, 0, 0)            # no need DataMovePad
        self.shape = Shape(self.tik_inst, tiling_ub, self.params)
        self.args = Args(self.tik_inst, tiling_ub, self.params)
        self.reg = Reg(self.tik_inst, self.d_type)

        with self.tik_inst.for_range(0, self.args.core_num_var, block_num=self.args.core_num_var) as core_id:
            with self.tik_inst.if_scope(core_id < self.args.need_core_num):
                with self.tik_inst.if_scope(tik.all(self.reg.is_support_vgather == 1,
                        self.args.interp_mode_const == 0,
                        self.shape.in_c == 1,
                        self.shape.in_hw < Gs2Constant.X_UB_SIZE_4_MINI_IH_IW // 4)):
                    with self.tik_inst.new_stmt_scope():
                        self.mini_ih_iw.shape = self.shape
                        self.mini_ih_iw.args = self.args
                        self.mini_ih_iw.reg = self.reg
                        self.mini_ih_iw.ub = Ub4MiniIhIw(self.tik_inst, self.d_type)
                        self.mini_ih_iw.compute_one_core(core_id)
                with self.tik_inst.else_scope():
                    with self.tik_inst.new_stmt_scope():
                        self.ub = Ub4General(self.tik_inst, self.d_type, self.interpolation_mode)
                        self.tik_inst.set_atomic_add(self.d_type)
                        self._compute_one_core(core_id)
                        self.tik_inst.set_atomic_add(0)

        is_support_vgather = 0
        if tbe_platform.api_check_support("tik.vgather"):
            is_support_vgather = 1
        tbe_context.get_context().add_compile_info("vars", {"core_num": self.core_num,
                                                            "is_support_vgather": is_support_vgather})
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                               inputs=[self.gm.x_gm, self.gm.grid_gm],
                               outputs=[self.gm.y_gm],
                               flowtable=(self.gm.tiling_gm,),
                               config=opt_config)

    def _simple_data_move(self, dst, src, num, d_type):
        if self.support_data_move_pad:
            self.tik_inst.data_move_pad(dst, src, 1, num * self.d_size, 0, 0)
        else:
            burst_len = (num * self.d_size + constant.BLOCK_SIZE - 1) // constant.BLOCK_SIZE
            self.tik_inst.data_move(dst, src, 0, 1, burst_len, 0, 0)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def _compute_weight(self, w1, w2, x1, x2, y1, y2):
        self.tik_inst.vsub(Gs2Constant.B32_MASK, w1, x1, x2, self.reg.grid_rep_times_int32, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vsub(Gs2Constant.B32_MASK, w2, y1, y2, self.reg.grid_rep_times_int32, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vmul(Gs2Constant.B32_MASK, w1, w1, w2, self.reg.grid_rep_times_int32, 1, 1, 1, 8, 8, 8)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def _simple_move_in(self, n, ii_loc, channel_align):
        """
        clear PIPE_MTE2
        """
        def _inner_mte2(bnum):
            with self.tik_inst.for_range(0, self.vmask) as iii_loc:
                self.reg.x_loc.set_as(self.ub.xx[ii_loc * self.vmask + iii_loc])
                self.reg.x_loc.set_as(n * self.shape.in_chw + self.reg.x_loc * self.shape.in_c)
                self.tik_inst.data_move(self.ub.x_ub[iii_loc * self.d_num_1block * bnum],
                                        self.gm.x_gm[self.reg.x_loc], 0, 1, bnum, 0, 0)

        def _inner_mte2_pad(_num_align, _c_len):
            with self.tik_inst.for_range(0, self.vmask) as iii_loc:
                self.reg.x_loc.set_as(self.ub.xx[ii_loc * self.vmask + iii_loc])
                self.reg.x_loc.set_as(n * self.shape.in_chw + self.reg.x_loc * self.shape.in_c)
                self.tik_inst.data_move_pad(self.ub.x_ub[iii_loc * _num_align],
                                            self.gm.x_gm[self.reg.x_loc], 1, _c_len, 0, 0)

        if self.support_data_move_pad:
            for c_iter in range(1, self.vmask + 1):
                with self.tik_inst.if_scope(self.shape.in_c == c_iter):
                    num_align = ceil_align(c_iter, self.d_num_1block)
                    c_len = c_iter * self.d_size
                    _inner_mte2_pad(num_align, c_len)
        else:
            with self.tik_inst.if_scope(self.d_type == "float16"):
                with self.tik_inst.if_scope(channel_align == 16):
                    _inner_mte2(1)
                with self.tik_inst.elif_scope(channel_align == 32):
                    _inner_mte2(2)
                with self.tik_inst.elif_scope(channel_align == 48):
                    _inner_mte2(3)
                with self.tik_inst.elif_scope(channel_align == 64):
                    _inner_mte2(4)
                with self.tik_inst.elif_scope(channel_align == 80):
                    _inner_mte2(5)
                with self.tik_inst.elif_scope(channel_align == 96):
                    _inner_mte2(6)
                with self.tik_inst.elif_scope(channel_align == 112):
                    _inner_mte2(7)
                with self.tik_inst.elif_scope(channel_align == 128):
                    _inner_mte2(8)
            with self.tik_inst.else_scope():
                with self.tik_inst.if_scope(channel_align == 8):
                    _inner_mte2(1)
                with self.tik_inst.elif_scope(channel_align == 16):
                    _inner_mte2(2)
                with self.tik_inst.elif_scope(channel_align == 24):
                    _inner_mte2(3)
                with self.tik_inst.elif_scope(channel_align == 32):
                    _inner_mte2(4)
                with self.tik_inst.elif_scope(channel_align == 40):
                    _inner_mte2(5)
                with self.tik_inst.elif_scope(channel_align == 48):
                    _inner_mte2(6)
                with self.tik_inst.elif_scope(channel_align == 56):
                    _inner_mte2(7)
                with self.tik_inst.elif_scope(channel_align == 64):
                    _inner_mte2(8)

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

    def _simple_trans_fp32_64_8x(self, dst, src, channel_align):
        if tbe_platform.api_check_support("tik.v4dtrans"):
            self.tik_inst.v4dtrans(False, dst, src, self.vmask, channel_align)
        elif self.vmask == 64:  # only support channel_align le 64
            with self.tik_inst.if_scope(channel_align == 8):
                src_list = [src[i * 8] for i in range(16)]
                dst_list = []
                for ii in range(8):
                    dst_list = dst_list + [dst[64 * ii], dst[64 * ii + 8]]
                self.tik_inst.vnchwconv(False, False, dst_list, src_list, 4, 2, 16)
            with self.tik_inst.else_scope():
                with self.tik_inst.for_range(0, 4) as ii:
                    src_list = [src[i * channel_align + ii * 16 * channel_align] for i in range(16)]
                    dst_list = []
                    for i in range(8):
                        dst_list = dst_list + [dst[64 * i + ii * 16], dst[64 * i + 8 + ii * 16]]
                    self.tik_inst.vnchwconv(False, False, dst_list, src_list, channel_align // 8, 64, 1)

    def _simple_trans_fp16_128_16x(self, dst, src, channel_align):
        if tbe_platform.api_check_support("tik.v4dtrans"):
            self.tik_inst.v4dtrans(False, dst, src, self.vmask, channel_align)
        elif self.vmask == 128:  # only support channel_align le 128
            with self.tik_inst.if_scope(channel_align == 16):
                src_list = [src[i * 16] for i in range(16)]
                dst_list = [dst[i * 128] for i in range(16)]
                self.tik_inst.vnchwconv(False, False, dst_list, src_list, 8, 1, 16)
            with self.tik_inst.else_scope():
                with self.tik_inst.for_range(0, 8) as ii:
                    src_list = [src[i * channel_align + ii * 16 * channel_align] for i in range(16)]
                    dst_list = [dst[i * 128 + ii * 16] for i in range(16)]
                    self.tik_inst.vnchwconv(False, False, dst_list, src_list, channel_align // 16, 128, 1)

    def _simple_trans_fp32_x_64(self, dst, src, channel):
        with self.tik_inst.if_scope(self.shape.in_c == 1):
            self.tik_inst.vadds(self.vmask, dst, src, 0.0, 1, 1, 1, 8, 8)
        with self.tik_inst.else_scope():
            if tbe_platform.api_check_support("tik.v4dtrans"):
                self.tik_inst.v4dtrans(True, dst, src, self.vmask, channel)
            elif self.vmask == 64:  # only support channel_align le 64
                channel_align = ceil_align(channel, self.d_num_1block)
                with self.tik_inst.if_scope(channel < channel_align):
                    self.tik_inst.vec_dup(self.vmask, src[self.vmask * channel], 0, channel_align - channel, 8)
                    self.tik_inst.vec_dup(self.vmask, dst, 0, channel_align, 8)

                with self.tik_inst.if_scope(channel == 3):
                    ub1 = self.tik_inst.Tensor(self.d_type, (256, ), tik.scope_ubuf, "ub1")
                    ub2 = self.tik_inst.Tensor(self.d_type, (256, ), tik.scope_ubuf, "ub2")

                    src_list = [src[i * 8] for i in range(16)]
                    dst_list = [ub1[i * 8] for i in range(16)]
                    self.tik_inst.vnchwconv(False, False, dst_list, src_list, 2, 16, 16)

                    src_list = [ub1[0], ub1[8], ub1[128], ub1[16], ub1[24], ub1[144], ub1[32], ub1[40],
                                ub1[160], ub1[48], ub1[56], ub1[176], ub1[64], ub1[72], ub1[192], ub1[80]]
                    dst_list = [ub2[i * 8] for i in range(16)]
                    self.tik_inst.vnchwconv(False, False, dst_list, src_list, 1, 0, 0)

                    src_list = [ub1[88], ub1[208], ub1[96], ub1[104], ub1[224], ub1[112], ub1[120], ub1[240]]
                    src_list += [ub1[248], ] * 8
                    dst_list = [ub2[128 + i * 8] for i in range(16)]
                    self.tik_inst.vnchwconv(False, False, dst_list, src_list, 1, 0, 0)

                    self.tik_inst.vadds(16, dst, ub2, 0.0, 8, 1, 1, 3, 2)
                    self.tik_inst.vadds(8, dst[16], ub2[128], 0.0, 8, 1, 1, 3, 2)

                with self.tik_inst.elif_scope(channel_align == 8):
                    src_list = [src[i * 64] for i in range(8)] + [src[i * 64 + 8] for i in range(8)]
                    dst_list = []
                    for ii in range(8):
                        dst_list = dst_list + [dst[8 * ii], dst[8 * ii + 64]]
                    self.tik_inst.vnchwconv(False, False, dst_list, src_list, 4, 16, 2)

                with self.tik_inst.else_scope():
                    with self.tik_inst.for_range(0, 4) as ii:
                        src_list = [src[i * 64 + ii * 16] for i in range(8)] +\
                                   [src[i * 64 + 8 + ii * 16] for i in range(8)]
                        dst_list = []
                        for i in range(8):
                            dst_list = dst_list + [dst[channel_align * (i + ii * 16)],
                                                   dst[channel_align * (i + 8 + ii * 16)]]
                        self.tik_inst.vnchwconv(False, False, dst_list, src_list, channel_align // 8, 1, 64)

    def _simple_trans_fp16_x_128(self, dst, src, channel):
        with self.tik_inst.if_scope(self.shape.in_c == 1):
            self.tik_inst.vadds(self.vmask, dst, src, 0.0, 1, 1, 1, 8, 8)
        with self.tik_inst.else_scope():
            if tbe_platform.api_check_support("tik.v4dtrans"):
                self.tik_inst.v4dtrans(True, dst, src, self.vmask, channel)
            elif self.vmask == 128:  # only support channel_align le 64
                channel_align = ceil_align(channel, self.d_num_1block)
                with self.tik_inst.if_scope(channel < channel_align):
                    self.tik_inst.vec_dup(self.vmask, src[self.vmask * channel], 0, channel_align - channel, 8)
                    self.tik_inst.vec_dup(self.vmask, dst, 0, channel_align, 8)

                with self.tik_inst.if_scope(channel == 3):
                    ub1 = self.tik_inst.Tensor(self.d_type, (512, ), tik.scope_ubuf, "ub1")
                    ub2 = self.tik_inst.Tensor(self.d_type, (512, ), tik.scope_ubuf, "ub2")

                    src_list = [src[i * 16] for i in range(16)]
                    dst_list = [ub1[i * 16] for i in range(16)]
                    self.tik_inst.vnchwconv(False, False, dst_list, src_list, 2, 16, 16)

                    src_list = [ub1[0], ub1[16], ub1[256], ub1[32], ub1[48], ub1[288], ub1[64], ub1[80],
                                ub1[320], ub1[96], ub1[112], ub1[352], ub1[128], ub1[144], ub1[384], ub1[160]]
                    dst_list = [ub2[i * 16] for i in range(16)]
                    self.tik_inst.vnchwconv(False, False, dst_list, src_list, 1, 0, 0)

                    src_list = [ub1[176], ub1[416], ub1[192], ub1[208], ub1[448], ub1[224], ub1[240], ub1[480]]
                    src_list += [ub1[496], ] * 8
                    dst_list = [ub2[256 + i * 16] for i in range(16)]
                    self.tik_inst.vnchwconv(False, False, dst_list, src_list, 1, 0, 0)

                    self.tik_inst.vadds(64, dst, ub2, 0.0, 4, 1, 1, 6, 4)
                    self.tik_inst.vadds(32, dst[64], ub2[256], 0.0, 4, 1, 1, 6, 4)

                with self.tik_inst.elif_scope(channel_align == 16):
                    src_list = [src[i * 128] for i in range(16)]
                    dst_list = []
                    for ii in range(16):
                        dst_list = dst_list + [dst[16 * ii]]
                    self.tik_inst.vnchwconv(False, False, dst_list, src_list, 8, 16, 1)

                with self.tik_inst.else_scope():
                    with self.tik_inst.for_range(0, 8) as ii:
                        src_list = [src[i * 128 + ii * 16] for i in range(16)]
                        dst_list = []
                        for i in range(16):
                            dst_list = dst_list + [dst[channel_align * (i + ii * 16)]]
                        self.tik_inst.vnchwconv(False, False, dst_list, src_list, channel_align // 16, 1, 128)

    def _move_to_gm(self, n, ii_loc, channel_align):
        if tbe_platform.api_check_support("tik.v4dtrans"):
            self.reg.y_offset.set_as((n * self.shape.grid_hw + self.reg.grid_hw_offset +
                                      ii_loc * self.vmask) * self.shape.in_c)
            self.reg.mte_burst.set_as(self.shape.in_c * self.vmask * self.d_size // 32)
            self.tik_inst.data_move(self.gm.y_gm[self.reg.y_offset], self.ub.out_val,
                                    0, 1, self.reg.mte_burst, 0, 0)                     # no need DataMovePad
        else:
            with self.tik_inst.if_scope(tik.any(self.shape.in_c == 1, self.shape.in_c == 3,
                                                self.shape.in_c == channel_align)):
                self.reg.y_offset.set_as((n * self.shape.grid_hw + self.reg.grid_hw_offset +
                                          ii_loc * self.vmask) * self.shape.in_c)
                self.reg.mte_burst.set_as(self.shape.in_c * self.vmask * self.d_size // 32)
                self.tik_inst.data_move(self.gm.y_gm[self.reg.y_offset], self.ub.out_val,
                                        0, 1, self.reg.mte_burst, 0, 0)                  # no need DataMovePad
            with self.tik_inst.else_scope():
                with self.tik_inst.for_range(0, self.vmask) as iii:
                    self.reg.y_offset.set_as((n * self.shape.grid_hw + self.reg.grid_hw_offset +
                                              ii_loc * self.vmask + iii) * self.shape.in_c)
                    self._simple_data_move(self.gm.y_gm[self.reg.y_offset], self.ub.out_val[iii * channel_align],
                                           self.shape.in_c, self.d_type)

    def _reflect_coordinates_general(self, coord_fp, twice_low, twice_high):
        if self.d_type == "float16":
            coord_fp_x = self.ub.x_ub_temp
            vmask = Gs2Constant.B32_MASK
            grid_rep_times_int = self.reg.grid_rep_times_int32
            self.tik_inst.vconv(vmask, 'none', coord_fp_x, coord_fp, grid_rep_times_int, 1, 1, 8, 4)

            temp_int = self.ub.yy
            with self.tik_inst.if_scope(twice_low == twice_high):
                self.tik_inst.vec_dup(vmask, coord_fp_x, 0.0, grid_rep_times_int, 8)
            
            with self.tik_inst.else_scope():
                min_s = twice_low / 2.0
                span_s = (twice_high - twice_low) / 2.0

                min_scalar = self.tik_inst.Scalar(dtype="float32", name="min_scalar", init_value=min_s)
                span_scalar = self.tik_inst.Scalar(dtype="float32", name="span_scalar", init_value=span_s)
                span_b_scalar = self.tik_inst.Scalar(dtype="float32", name="span_b_scalar", init_value=1 / span_s)

                dupl_fp = self.ub.y_ub_temp
                self.tik_inst.vec_dup(vmask, dupl_fp, min_scalar, grid_rep_times_int, 8)
                self.tik_inst.vsub(vmask, coord_fp_x, coord_fp_x, dupl_fp, grid_rep_times_int, 1, 1, 1, 8, 8, 8)
                self.tik_inst.vabs(vmask, coord_fp_x, coord_fp_x, grid_rep_times_int, 1, 1, 8, 8)

                extra_fp = self.ub.y_ub_temp[self.ub.x_ub_offset]
                self.tik_inst.vmuls(vmask, extra_fp, coord_fp_x, span_b_scalar,
                                    grid_rep_times_int, 1, 1, 8, 8)

                self.tik_inst.vconv(vmask, 'floor', temp_int, extra_fp,
                                    grid_rep_times_int, 1, 1, 8, 8)
                self.tik_inst.vconv(vmask, 'none', extra_fp, temp_int,
                                    grid_rep_times_int, 1, 1, 8, 8)
                self.tik_inst.vmuls(vmask, extra_fp, extra_fp, span_scalar,
                                    grid_rep_times_int, 1, 1, 8, 8)
                self.tik_inst.vsub(vmask, extra_fp, coord_fp_x, extra_fp,
                                   grid_rep_times_int, 1, 1, 1, 8, 8, 8)
                
                self.tik_inst.vmuls(vmask, coord_fp_x, coord_fp_x, span_b_scalar,
                                    grid_rep_times_int, 1, 1, 8, 8)
                self.tik_inst.vconv(vmask, 'floor', temp_int, coord_fp_x,
                                    grid_rep_times_int, 1, 1, 8, 8)
                self.tik_inst.vconv(vmask, 'none', coord_fp_x, temp_int,
                                    grid_rep_times_int, 1, 1, 8, 8)
                
                fmod_fp = self.ub.y_ub_temp[self.ub.x_ub_offset * 2]
                self.tik_inst.vmuls(vmask, fmod_fp, coord_fp_x, 1 / 2.0,
                                    grid_rep_times_int, 1, 1, 8, 8)
                self.tik_inst.vconv(vmask, 'floor', temp_int, fmod_fp,
                                    grid_rep_times_int, 1, 1, 8, 8)
                self.tik_inst.vconv(vmask, 'none', fmod_fp, temp_int,
                                    grid_rep_times_int, 1, 1, 8, 8)
                self.tik_inst.vmuls(vmask, fmod_fp, fmod_fp, 2, grid_rep_times_int, 1, 1, 8, 8)
                self.tik_inst.vsub(vmask, fmod_fp, coord_fp_x, fmod_fp,
                                   grid_rep_times_int, 1, 1, 1, 8, 8, 8)
                
                self.tik_inst.vcmpvs_eq(self.ub.clip_mask_eq, fmod_fp, 0.0, grid_rep_times_int, 1, 8)
                ub_mask_eq = self.ub.clip_mask_eq.reinterpret_cast_to('uint64')
                high_mask_eq = self.tik_inst.Scalar(dtype='uint64', name='high_mask_eq', init_value=0)
                low_mask_eq = self.tik_inst.Scalar(dtype='uint64', name='low_mask_eq', init_value=0)
                with self.tik_inst.for_range(0, grid_rep_times_int) as i_loc:
                    low_mask_eq.set_as(ub_mask_eq[i_loc])
                    self.tik_inst.vadds([high_mask_eq, low_mask_eq], coord_fp_x[i_loc * vmask],
                                        extra_fp[i_loc * vmask], min_scalar, 1, 1, 1, 8, 8)

                self.tik_inst.vcmpvs_ne(self.ub.clip_mask_ne, fmod_fp, 0.0, grid_rep_times_int, 1, 8)
                ub_mask_ne = self.ub.clip_mask_ne.reinterpret_cast_to('uint64')
                high_mask_ne = self.tik_inst.Scalar(dtype='uint64', name='high_mask_ne', init_value=0)
                low_mask_ne = self.tik_inst.Scalar(dtype='uint64', name='low_mask_ne', init_value=0)
                init_scalar = self.tik_inst.Scalar(dtype='float32', name='init_scalar', init_value=-1)
                with self.tik_inst.for_range(0, grid_rep_times_int) as j_loc:
                    low_mask_ne.set_as(ub_mask_ne[j_loc])
                    self.tik_inst.vmuls([high_mask_ne, low_mask_ne], extra_fp[j_loc * vmask],
                                        extra_fp[j_loc * vmask], init_scalar, 1, 1, 1, 8, 8)
                    self.tik_inst.vadds([high_mask_ne, low_mask_ne], extra_fp[j_loc * vmask],
                                        extra_fp[j_loc * vmask], span_scalar, 1, 1, 1, 8, 8)
                    self.tik_inst.vadds([high_mask_ne, low_mask_ne], coord_fp_x[j_loc * vmask],
                                        extra_fp[j_loc * vmask], min_scalar, 1, 1, 1, 8, 8)
            self.tik_inst.vconv(vmask, 'none', coord_fp, coord_fp_x, grid_rep_times_int, 1, 1, 4, 8)

        elif self.d_type == "float32":
            temp_int = self.ub.yy
            with self.tik_inst.if_scope(twice_low == twice_high):
                self.tik_inst.vec_dup(self.vmask, coord_fp, 0.0, self.reg.grid_rep_times, 8)
            
            with self.tik_inst.else_scope():
                min_s = twice_low / 2.0
                span_s = (twice_high - twice_low) / 2.0

                min_scalar = self.tik_inst.Scalar(dtype="float32", name="min_scalar", init_value=min_s)
                span_scalar = self.tik_inst.Scalar(dtype="float32", name="span_scalar", init_value=span_s)
                span_b_scalar = self.tik_inst.Scalar(dtype="float32", name="span_b_scalar", init_value=1 / span_s)

                dupl_fp = self.ub.x_ub
                self.tik_inst.vec_dup(self.vmask, dupl_fp, min_scalar, self.reg.grid_rep_times, 8)
                self.tik_inst.vsub(self.vmask, coord_fp, coord_fp, dupl_fp, self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)
                self.tik_inst.vabs(self.vmask, coord_fp, coord_fp, self.reg.grid_rep_times, 1, 1, 8, 8)

                extra_fp = self.ub.x_ub[self.ub.x_ub_offset]
                self.tik_inst.vmuls(self.vmask, extra_fp, coord_fp, span_b_scalar,
                                    self.reg.grid_rep_times, 1, 1, 8, 8)
                self.tik_inst.vconv(self.vmask, 'floor', temp_int, extra_fp,
                                    self.reg.grid_rep_times, 1, 1, 8, 8)
                self.tik_inst.vconv(self.vmask, 'none', extra_fp, temp_int,
                                    self.reg.grid_rep_times, 1, 1, 8, 8)
                self.tik_inst.vmuls(self.vmask, extra_fp, extra_fp, span_scalar,
                                    self.reg.grid_rep_times, 1, 1, 8, 8)
                self.tik_inst.vsub(self.vmask, extra_fp, coord_fp, extra_fp,
                                   self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)
                
                self.tik_inst.vmuls(self.vmask, coord_fp, coord_fp, span_b_scalar,
                                    self.reg.grid_rep_times, 1, 1, 8, 8)
                self.tik_inst.vconv(self.vmask, 'floor', temp_int, coord_fp,
                                    self.reg.grid_rep_times, 1, 1, 8, 8)
                self.tik_inst.vconv(self.vmask, 'none', coord_fp, temp_int,
                                    self.reg.grid_rep_times, 1, 1, 8, 8)
                
                fmod_fp = self.ub.x_ub[self.ub.x_ub_offset * 2]
                self.tik_inst.vmuls(self.vmask, fmod_fp, coord_fp, 1 / 2.0,
                                    self.reg.grid_rep_times, 1, 1, 8, 8)
                self.tik_inst.vconv(self.vmask, 'floor', temp_int, fmod_fp,
                                    self.reg.grid_rep_times, 1, 1, 8, 8)
                self.tik_inst.vconv(self.vmask, 'none', fmod_fp, temp_int,
                                    self.reg.grid_rep_times, 1, 1, 8, 8)
                self.tik_inst.vmuls(self.vmask, fmod_fp, fmod_fp, 2, self.reg.grid_rep_times, 1, 1, 8, 8)
                self.tik_inst.vsub(self.vmask, fmod_fp, coord_fp, fmod_fp,
                                   self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)
                
                self.tik_inst.vcmpvs_eq(self.ub.clip_mask_eq, fmod_fp, 0.0, self.reg.grid_rep_times, 1, 8)
                ub_mask_eq = self.ub.clip_mask_eq.reinterpret_cast_to('uint64')
                self._fmod_eq_branch(ub_mask_eq, extra_fp, coord_fp, min_scalar)

                self.tik_inst.vcmpvs_ne(self.ub.clip_mask_ne, fmod_fp, 0.0, self.reg.grid_rep_times, 1, 8)
                ub_mask_ne = self.ub.clip_mask_ne.reinterpret_cast_to('uint64')
                self._fmod_ne_branch(ub_mask_ne, extra_fp, coord_fp, span_scalar, min_scalar)

    def _fmod_eq_branch(self, ub_mask_eq, extra_fp, coord_fp, min_scalar):
        high_mask_eq = self.tik_inst.Scalar(dtype='uint64', name='high_mask_eq', init_value=0)
        low_mask_eq = self.tik_inst.Scalar(dtype='uint64', name='low_mask_eq', init_value=0)
        with self.tik_inst.for_range(0, self.reg.grid_rep_times) as i_loc:
            low_mask_eq.set_as(ub_mask_eq[i_loc])
            with self.tik_inst.if_scope(low_mask_eq != 0):
                self.tik_inst.vadds([high_mask_eq, low_mask_eq], coord_fp[i_loc * self.vmask],
                                extra_fp[i_loc * self.vmask], min_scalar, 1, 1, 1, 8, 8)

    def _fmod_ne_branch(self, ub_mask_ne, extra_fp, coord_fp, span_scalar, min_scalar):
        high_mask_ne = self.tik_inst.Scalar(dtype='uint64', name='high_mask_ne', init_value=0)
        low_mask_ne = self.tik_inst.Scalar(dtype='uint64', name='low_mask_ne', init_value=0)
        init_scalar = self.tik_inst.Scalar(dtype='float32', name='init_scalar', init_value=-1)
        with self.tik_inst.for_range(0, self.reg.grid_rep_times) as j_loc:
            low_mask_ne.set_as(ub_mask_ne[j_loc])
            with self.tik_inst.if_scope(low_mask_ne != 0):
                self.tik_inst.vmuls([high_mask_ne, low_mask_ne], extra_fp[j_loc * self.vmask],
                                    extra_fp[j_loc * self.vmask], init_scalar, 1, 1, 1, 8, 8)
                self.tik_inst.vadds([high_mask_ne, low_mask_ne], extra_fp[j_loc * self.vmask],
                                    extra_fp[j_loc * self.vmask], span_scalar, 1, 1, 1, 8, 8)
                self.tik_inst.vadds([high_mask_ne, low_mask_ne], coord_fp[j_loc * self.vmask],
                                    extra_fp[j_loc * self.vmask], min_scalar, 1, 1, 1, 8, 8)

    def _clip_coordinates(self, ix_fp, iy_fp, ix_int, iy_int, out_coor, out_mask=None):
        ix_fp_tmp = self.ub.out_val
        iy_fp_tmp = self.ub.out_val[self.ub.x_ub_offset]
        ix_int_tmp = self.ub.xx
        iy_int_tmp = self.ub.yy

        self.tik_inst.vadds(self.vmask, ix_fp_tmp, ix_fp, 0, self.reg.grid_rep_times, 1, 1, 8, 8)
        self.tik_inst.vadds(self.vmask, iy_fp_tmp, iy_fp, 0, self.reg.grid_rep_times, 1, 1, 8, 8)
        with self.tik_inst.if_scope(self.d_type == "float16"):
            self.tik_inst.vadds(Gs2Constant.B32_MASK, ix_int_tmp, ix_int, 0, self.reg.grid_rep_times_int32, 1, 1, 8, 8)
            self.tik_inst.vadds(Gs2Constant.B32_MASK, iy_int_tmp, iy_int, 0, self.reg.grid_rep_times_int32, 1, 1, 8, 8)
        with self.tik_inst.else_scope():
            self.tik_inst.vadds(self.vmask, ix_int_tmp, ix_int, 0, self.reg.grid_rep_times, 1, 1, 8, 8)
            self.tik_inst.vadds(self.vmask, iy_int_tmp, iy_int, 0, self.reg.grid_rep_times, 1, 1, 8, 8)

        with self.tik_inst.if_scope(self.padding_mode == Gs2Constant.PADDING_MODE_REFLECTION):
            with self.tik_inst.if_scope(self.args.align_corners == 1):
                self._reflect_coordinates_general(ix_fp_tmp, 0, 2 * (self.shape.in_w_fp32 - 1))
                self._reflect_coordinates_general(iy_fp_tmp, 0, 2 * (self.shape.in_h_fp32 - 1))   
            with self.tik_inst.else_scope():
                self._reflect_coordinates_general(ix_fp_tmp, -1, 2 * self.shape.in_w_fp32 - 1)
                self._reflect_coordinates_general(iy_fp_tmp, -1, 2 * self.shape.in_h_fp32 - 1)
        
            with self.tik_inst.if_scope(self.d_type == "float16"):
                self.tik_inst.vconv(Gs2Constant.B32_MASK, 'floor', ix_int_tmp, ix_fp_tmp,
                                    self.reg.grid_rep_times_int32, 1, 1, 8, 4)
                self.tik_inst.vconv(Gs2Constant.B32_MASK, 'floor', iy_int_tmp, iy_fp_tmp,
                                    self.reg.grid_rep_times_int32, 1, 1, 8, 4)
            with self.tik_inst.else_scope():
                self.tik_inst.vconv(self.vmask, 'floor', ix_int_tmp, ix_fp_tmp, self.reg.grid_rep_times, 1, 1, 8, 8)
                self.tik_inst.vconv(self.vmask, 'floor', iy_int_tmp, iy_fp_tmp, self.reg.grid_rep_times, 1, 1, 8, 8)
             
        with self.tik_inst.if_scope(self.padding_mode == Gs2Constant.PADDING_MODE_ZEROS):
            self.tik_inst.vcmpvs_ge(self.ub.clip_mask1, ix_fp, 0, self.reg.grid_rep_times, 1, 8)

            self.tik_inst.vcmpvs_ge(self.ub.clip_mask2, iy_fp, 0, self.reg.grid_rep_times, 1, 8)
            self._simple_vand(self.ub.clip_mask1, self.ub.clip_mask1, self.ub.clip_mask2,
                              constant.MASK128, self.clip_mask_ub_num)

            with self.tik_inst.if_scope(self.d_type == "float16"):
                self.tik_inst.vcmpvs_lt(self.ub.clip_mask2, ix_fp, self.shape.in_w_fp16,
                                    self.reg.grid_rep_times, 1, 8)
            with self.tik_inst.else_scope():
                self.tik_inst.vcmpvs_lt(self.ub.clip_mask2, ix_fp, self.shape.in_w_fp32,
                                    self.reg.grid_rep_times, 1, 8)
            self._simple_vand(self.ub.clip_mask1, self.ub.clip_mask1, self.ub.clip_mask2,
                              constant.MASK128, self.clip_mask_ub_num)

            with self.tik_inst.if_scope(self.d_type == "float16"):
                self.tik_inst.vcmpvs_lt(self.ub.clip_mask2, iy_fp, self.shape.in_h_fp16,
                                    self.reg.grid_rep_times, 1, 8)
            with self.tik_inst.else_scope():
                self.tik_inst.vcmpvs_lt(self.ub.clip_mask2, iy_fp, self.shape.in_h_fp32,
                                    self.reg.grid_rep_times, 1, 8)
            self._simple_vand(self.ub.clip_mask1, self.ub.clip_mask1, self.ub.clip_mask2,
                              constant.MASK128, self.clip_mask_ub_num)

            if self.interpolation_mode == Gs2Constant.INTERPOLATION_MODE_BICUBIC:
                self.tik_inst.vec_dup(self.vmask, out_mask, 1.0, self.reg.grid_rep_times, 8)
                self.tik_inst.vsel(self.vmask, 1, out_mask, self.ub.clip_mask1, out_mask, 0.0,
                                   self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)

        # bound value for padding_mode is 'zeros' or 'border' or 'reflection'
        tmp_coor = self.ub.yy
        self.tik_inst.vmins(Gs2Constant.B32_MASK, out_coor, ix_int_tmp,
                            self.shape.in_w_int32 - 1, self.reg.grid_rep_times_int32, 1, 1, 8, 8)
        self.tik_inst.vmins(Gs2Constant.B32_MASK, tmp_coor, iy_int_tmp,
                            self.shape.in_h - 1, self.reg.grid_rep_times_int32, 1, 1, 8, 8)
        self.tik_inst.vmaxs(Gs2Constant.B32_MASK, out_coor, out_coor,
                            0, self.reg.grid_rep_times_int32, 1, 1, 8, 8)
        self.tik_inst.vmaxs(Gs2Constant.B32_MASK, tmp_coor, tmp_coor,
                            0, self.reg.grid_rep_times_int32, 1, 1, 8, 8)

        self.tik_inst.vmuls(Gs2Constant.B32_MASK, tmp_coor, tmp_coor,
                            self.shape.in_w_int32, self.reg.grid_rep_times_int32, 1, 1, 8, 8)
        self.tik_inst.vadd(Gs2Constant.B32_MASK, out_coor, out_coor, tmp_coor,
                           self.reg.grid_rep_times_int32, 1, 1, 1, 8, 8, 8)

    def _process_point_bilinear_nhwc_gt1_le64(self, n, weight):
        """case: NHWC and 1 < C <= 64"""
        channel_align = ceil_align(self.shape.in_c, self.d_num_1block)
        with self.tik_inst.for_range(0, self.reg.grid_cur_num // self.vmask) as ii_loc:
            self._simple_move_in(n, ii_loc, channel_align)

            # [128or64, channel_align] --> [channel_align, 128or64]
            with self.tik_inst.if_scope(self.d_type == "float16"):
                self._simple_trans_fp16_128_16x(self.ub.out_val, self.ub.x_ub, channel_align)
            with self.tik_inst.else_scope():
                self._simple_trans_fp32_64_8x(self.ub.out_val, self.ub.x_ub, channel_align)

            # x_value x weight, [channel, 128or64] x [128or64,], repeat channel times
            self.tik_inst.vmul(self.vmask, self.ub.x_ub, self.ub.out_val, weight[ii_loc * self.vmask],
                               self.shape.in_c, 1, 1, 1, 8, 8, 0)

            # [channel_align, 128or64] --> [128or64, channel]
            with self.tik_inst.if_scope(self.d_type == "float16"):
                self._simple_trans_fp16_x_128(self.ub.out_val, self.ub.x_ub, self.shape.in_c_int32)
            with self.tik_inst.else_scope():
                self._simple_trans_fp32_x_64(self.ub.out_val, self.ub.x_ub, self.shape.in_c_int32)           

            # Move out [128or64, channel]
            self._move_to_gm(n, ii_loc, channel_align)

        self.reg.tail_offset.set_as(floor_align(self.reg.grid_cur_num, self.vmask))
        with self.tik_inst.if_scope(self.reg.grid_cur_num > self.reg.tail_offset):
            self.tik_inst.vec_dup(self.vmask, self.ub.out_val, 0.0, Gs2Constant.OUT_VAL_NUM // self.vmask, 8)
            with self.tik_inst.for_range(self.reg.tail_offset, self.reg.grid_cur_num) as iii_loc:
                self.reg.x_loc.set_as(self.ub.xx[iii_loc])
                self.reg.x_loc.set_as(n * self.shape.in_chw + self.reg.x_loc * self.shape.in_c)
                self._simple_data_move(self.ub.x_ub, self.gm.x_gm[self.reg.x_loc], self.shape.in_c, self.d_type)

                self.reg.x_weight.set_as(weight[iii_loc])
                self.tik_inst.vmuls(self.shape.in_c, self.ub.x_ub, self.ub.x_ub, self.reg.x_weight,
                                    1, 1, 1, 8, 8)
                with self.tik_inst.for_range(0, self.shape.in_c) as j:
                    self.ub.out_val[(iii_loc - self.reg.tail_offset) * self.shape.in_c +
                                    j].set_as(self.ub.x_ub[j])

            self.reg.y_offset.set_as((n * self.shape.grid_hw + self.reg.grid_hw_offset +
                                      self.reg.tail_offset) * self.shape.in_c)
            self._simple_data_move(self.gm.y_gm[self.reg.y_offset], self.ub.out_val,
                                   (self.reg.grid_cur_num - self.reg.tail_offset) * self.shape.in_c, self.d_type)

    def _process_point_bilinear_nhwc_cgt64(self, n, weight):
        """case: NHWC and C > 64"""
        with self.tik_inst.for_range(0, self.reg.grid_cur_num) as ii_loc:
            self.reg.x_loc.set_as(self.ub.xx[ii_loc])
            self.reg.x_loc.set_as(n * self.shape.in_chw + self.reg.x_loc * self.shape.in_c)
            self.reg.x_weight.set_as(weight[ii_loc])
            with self.tik_inst.for_range(0, self.shape.in_c // self.grid_ub_num) as iii:
                self.tik_inst.data_move(self.ub.x_ub, self.gm.x_gm[self.reg.x_loc + self.grid_ub_num * iii], 0, 1,
                                        self.grid_ub_num * self.d_size // 32, 0, 0)        # no need DataMovePad
                self.tik_inst.vmuls(self.vmask, self.ub.out_val, self.ub.x_ub, self.reg.x_weight,
                                    self.grid_ub_num // self.vmask, 1, 1, 8, 8)

                self.reg.y_offset.set_as((n * self.shape.grid_hw + self.reg.grid_hw_offset +
                                          ii_loc) * self.shape.in_c + self.grid_ub_num * iii)
                self.tik_inst.data_move(self.gm.y_gm[self.reg.y_offset], self.ub.out_val, 0, 1,
                                        self.grid_ub_num * self.d_size // 32, 0, 0)        # no need DataMovePad

            self.reg.tail_offset.set_as(floor_align(self.shape.in_c, self.grid_ub_num))
            with self.tik_inst.if_scope(self.shape.in_c > self.reg.tail_offset):
                self.reg.tail_num.set_as(self.shape.in_c - self.reg.tail_offset)
                self.reg.tail_aln.set_as(floor_align(self.reg.tail_num, self.vmask))
                self.reg.tail_tail.set_as(self.reg.tail_num - self.reg.tail_aln)

                self.reg.x_offset.set_as(self.reg.x_loc + floor_align(self.shape.in_c, self.grid_ub_num))
                self._simple_data_move(self.ub.x_ub, self.gm.x_gm[self.reg.x_offset],
                                       self.reg.tail_num, self.d_type)

                self.tik_inst.vec_dup(self.vmask, self.ub.out_val, 0.0, ceil_div(self.reg.tail_num, self.vmask), 8)
                with self.tik_inst.if_scope(self.reg.tail_num > self.vmask):
                    self.tik_inst.vmuls(self.vmask, self.ub.out_val, self.ub.x_ub, self.reg.x_weight,
                                        self.reg.tail_num // self.vmask, 1, 1, 8, 8)
                with self.tik_inst.if_scope(self.reg.tail_tail > 0):
                    self.tik_inst.vmuls(self.reg.tail_tail, self.ub.out_val[self.reg.tail_aln],
                                        self.ub.x_ub[self.reg.tail_aln], self.reg.x_weight, 1, 1, 1, 8, 8)

                self.reg.y_offset.set_as((n * self.shape.grid_hw + self.reg.grid_hw_offset +
                                          ii_loc) * self.shape.in_c + self.reg.tail_offset)
                self._simple_data_move(self.gm.y_gm[self.reg.y_offset], self.ub.out_val,
                                       self.reg.tail_num, self.d_type)

    def _process_point_bilinear_nchw(self, n, weight):
        """Support case NCHW or NHW1. However, these scenarios currently do not occur."""
        with self.tik_inst.for_range(0, self.shape.in_c) as i_c:
            self.tik_inst.vec_dup(self.vmask, self.ub.out_val, 0.0, self.reg.grid_rep_times, 8)
            with self.tik_inst.for_range(0, self.reg.grid_cur_num / Gs2Constant.UNROLL_NUM) as ii:
                for j in range(Gs2Constant.UNROLL_NUM):
                    self.reg.x_loc.set_as(self.ub.xx[ii * Gs2Constant.UNROLL_NUM + j])
                    self.reg.xlocs[j].set_as(n * self.shape.in_chw + i_c * self.shape.in_hw + self.reg.x_loc)
                for j in range(Gs2Constant.UNROLL_NUM):
                    self._simple_data_move(self.ub.x_ub[j * self.d_num_1block], self.gm.x_gm[self.reg.xlocs[j]],
                                           1, self.d_type)
                for j in range(Gs2Constant.UNROLL_NUM):
                    self.ub.out_val[ii * Gs2Constant.UNROLL_NUM + j].set_as(self.ub.x_ub[j * self.d_num_1block])
            with self.tik_inst.for_range(floor_align(self.reg.grid_cur_num, Gs2Constant.UNROLL_NUM),
                                         self.reg.grid_cur_num) as ii:
                self.reg.x_loc.set_as(self.ub.xx[ii])
                self.reg.x_loc.set_as(n * self.shape.in_chw + i_c * self.shape.in_hw + self.reg.x_loc)
                self._simple_data_move(self.ub.x_ub, self.gm.x_gm[self.reg.x_loc], 1, self.d_type)
                self.ub.out_val[ii].set_as(self.ub.x_ub)

            self.tik_inst.vmul(self.vmask, self.ub.out_val, self.ub.out_val, weight,
                               self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)

            self.reg.y_offset.set_as(n * self.shape.out_chw + i_c * self.shape.grid_hw + self.reg.grid_hw_offset)
            self._simple_data_move(self.gm.y_gm[self.reg.y_offset], self.ub.out_val,
                                   self.reg.grid_cur_num, self.d_type)

    def _process_point_bilinear(self, n, weight):
        with self.tik_inst.if_scope(self.padding_mode == Gs2Constant.PADDING_MODE_ZEROS):
            self.tik_inst.vsel(self.vmask, 1, weight, self.ub.clip_mask1, weight, 0.0,
                               self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)

        with self.tik_inst.if_scope(tik.any(self.shape.in_c == 1,
                                            tik.all(self.args.channel_last == 1, self.shape.in_c <= 64))):
            self._process_point_bilinear_nhwc_gt1_le64(n, weight)
        with self.tik_inst.elif_scope(tik.all(self.args.channel_last == 1, self.shape.in_c > 64)):
            self._process_point_bilinear_nhwc_cgt64(n, weight)
        with self.tik_inst.elif_scope(self.args.channel_last == 0):
            self._process_point_bilinear_nchw(n, weight)

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
    def _compute_val_bilinear_fp16(self, n):
        self._compute_coordinates_bilinear()

        self.tik_inst.vconv(Gs2Constant.B32_MASK, 'floor', self.ub.ix_nw_sw_int, self.ub.grid_ub,
                        self.reg.grid_rep_times_int32, 1, 1, 8, 4)               # ix floor co-ordinates      

        self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.ix_nw_sw_fp32, self.ub.ix_nw_sw_int,
                        self.reg.grid_rep_times_int32, 1, 1, 8, 8)          # ix floor co-ordinates
        
        self.tik_inst.vadds(Gs2Constant.B32_MASK, self.ub.ix_ne_se_int, self.ub.ix_nw_sw_int, 1,
                            self.reg.grid_rep_times_int32, 1, 1, 8, 8)               # ix ceil co-ordinates
        self.tik_inst.vadds(Gs2Constant.B32_MASK, self.ub.ix_ne_se_fp32, self.ub.ix_nw_sw_fp32, 1.0,
                            self.reg.grid_rep_times_int32, 1, 1, 8, 8)               # ix ceil co-ordinates

        self.tik_inst.vconv(Gs2Constant.B32_MASK, 'floor', self.ub.iy_nw_sw_int, self.ub.grid_ub[self.grid_ub_num],
                            self.reg.grid_rep_times_int32, 1, 1, 8, 4)               # iy floor co-ordinates
        
        self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.iy_nw_sw_fp32, self.ub.iy_nw_sw_int,
                            self.reg.grid_rep_times_int32, 1, 1, 8, 8)               # iy floor co-ordinates
        
        self.tik_inst.vadds(Gs2Constant.B32_MASK, self.ub.iy_ne_se_int, self.ub.iy_nw_sw_int, 1,
                            self.reg.grid_rep_times_int32, 1, 1, 8, 8)               # iy ceil co-ordinates
        self.tik_inst.vadds(Gs2Constant.B32_MASK, self.ub.iy_ne_se_fp32, self.ub.iy_nw_sw_fp32, 1.0,
                            self.reg.grid_rep_times_int32, 1, 1, 8, 8)               # iy ceil co-ordinates

        ix, iy = self.ub.grid_ub, self.ub.grid_ub[self.grid_ub_num]

        out_val_fp32 = self.ub.x_ub.reinterpret_cast_to('float32')
        ix_fp32 = out_val_fp32
        iy_fp32 = out_val_fp32[self.grid_ub_num]
        self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', out_val_fp32, ix,
                            self.reg.grid_rep_times_int32, 1, 1, 8, 4)
        self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', out_val_fp32[self.grid_ub_num], iy,
                            self.reg.grid_rep_times_int32, 1, 1, 8, 4)
        
        ix_nw_fp32, iy_nw_fp32 = self.ub.ix_nw_sw_fp32, self.ub.iy_nw_sw_fp32
        ix_nw_int, iy_nw_int = self.ub.ix_nw_sw_int, self.ub.iy_nw_sw_int

        ix_ne_fp32, iy_ne_fp32 = self.ub.ix_ne_se_fp32, self.ub.iy_nw_sw_fp32
        ix_ne_int, iy_ne_int = self.ub.ix_ne_se_int, self.ub.iy_nw_sw_int

        ix_sw_fp32, iy_sw_fp32 = self.ub.ix_nw_sw_fp32, self.ub.iy_ne_se_fp32
        ix_sw_int, iy_sw_int = self.ub.ix_nw_sw_int, self.ub.iy_ne_se_int

        ix_se_fp32, iy_se_fp32 = self.ub.ix_ne_se_fp32, self.ub.iy_ne_se_fp32
        ix_se_int, iy_se_int = self.ub.ix_ne_se_int, self.ub.iy_ne_se_int

        self._compute_weight(self.ub.nw_fp32, self.ub.weight_fp32, ix_se_fp32, ix_fp32, iy_se_fp32, iy_fp32)
        self._compute_weight(self.ub.ne_fp32, self.ub.weight_fp32, ix_fp32, ix_sw_fp32, iy_sw_fp32, iy_fp32)
        self._compute_weight(self.ub.sw_fp32, self.ub.weight_fp32, ix_ne_fp32, ix_fp32, iy_fp32, iy_ne_fp32)
        self._compute_weight(self.ub.se_fp32, self.ub.weight_fp32, ix_fp32, ix_nw_fp32, iy_fp32, iy_nw_fp32)

        ix_nw_fp, iy_nw_fp = self.ub.ix_nw_sw_fp, self.ub.iy_nw_sw_fp

        ix_ne_fp, iy_ne_fp = self.ub.ix_ne_se_fp, self.ub.iy_nw_sw_fp

        ix_sw_fp, iy_sw_fp = self.ub.ix_nw_sw_fp, self.ub.iy_ne_se_fp

        ix_se_fp, iy_se_fp = self.ub.ix_ne_se_fp, self.ub.iy_ne_se_fp

        self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', ix_nw_fp, ix_nw_fp32,
                            self.reg.grid_rep_times_int32, 1, 1, 4, 8)
        self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', iy_nw_fp, iy_nw_fp32,
                            self.reg.grid_rep_times_int32, 1, 1, 4, 8)
        self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', ix_ne_fp, ix_ne_fp32,
                            self.reg.grid_rep_times_int32, 1, 1, 4, 8)
        self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', iy_ne_fp, iy_ne_fp32,
                            self.reg.grid_rep_times_int32, 1, 1, 4, 8)
        self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', ix_sw_fp, ix_sw_fp32,
                            self.reg.grid_rep_times_int32, 1, 1, 4, 8)
        self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', iy_sw_fp, iy_sw_fp32,
                            self.reg.grid_rep_times_int32, 1, 1, 4, 8)
        self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', ix_se_fp, ix_se_fp32,
                            self.reg.grid_rep_times_int32, 1, 1, 4, 8)
        self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', iy_se_fp, iy_se_fp32,
                            self.reg.grid_rep_times_int32, 1, 1, 4, 8)

        self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.nw, self.ub.nw_fp32,
                            self.reg.grid_rep_times_int32, 1, 1, 4, 8)
        self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.ne, self.ub.ne_fp32,
                            self.reg.grid_rep_times_int32, 1, 1, 4, 8)
        self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.sw, self.ub.sw_fp32,
                            self.reg.grid_rep_times_int32, 1, 1, 4, 8)
        self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.se, self.ub.se_fp32,
                            self.reg.grid_rep_times_int32, 1, 1, 4, 8)
        self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.weight, self.ub.weight_fp32,
                            self.reg.grid_rep_times_int32, 1, 1, 4, 8)

        self._clip_coordinates(ix_nw_fp, iy_nw_fp, ix_nw_int, iy_nw_int, self.ub.xx)
        self._process_point_bilinear(n, self.ub.nw)

        self._clip_coordinates(ix_ne_fp, iy_ne_fp, ix_ne_int, iy_ne_int, self.ub.xx)
        self._process_point_bilinear(n, self.ub.ne)

        self._clip_coordinates(ix_sw_fp, iy_sw_fp, ix_sw_int, iy_sw_int, self.ub.xx)
        self._process_point_bilinear(n, self.ub.sw)

        self._clip_coordinates(ix_se_fp, iy_se_fp, ix_se_int, iy_se_int, self.ub.xx)
        self._process_point_bilinear(n, self.ub.se)

    def _compute_val_bilinear(self, n):
        self._compute_coordinates_bilinear()

        with self.tik_inst.if_scope(self.d_type == "float16"):
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'floor', self.ub.ix_nw_sw_int, self.ub.grid_ub,
                            self.reg.grid_rep_times_int32, 1, 1, 8, 4)               # ix floor co-ordinates
        with self.tik_inst.else_scope():
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'floor', self.ub.ix_nw_sw_int, self.ub.grid_ub,
                            self.reg.grid_rep_times_int32, 1, 1, 8, 8)               # ix floor co-ordinates       

        with self.tik_inst.if_scope(self.d_type == "float16"):
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.ix_nw_sw_fp, self.ub.ix_nw_sw_int,
                            self.reg.grid_rep_times_int32, 1, 1, 4, 8, 1.0)          # ix floor co-ordinates
        with self.tik_inst.else_scope():
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.ix_nw_sw_fp, self.ub.ix_nw_sw_int,
                            self.reg.grid_rep_times_int32, 1, 1, 8, 8)               # ix floor co-ordinates
        
        self.tik_inst.vadds(Gs2Constant.B32_MASK, self.ub.ix_ne_se_int, self.ub.ix_nw_sw_int, 1,
                            self.reg.grid_rep_times_int32, 1, 1, 8, 8)               # ix ceil co-ordinates
        self.tik_inst.vadds(self.vmask, self.ub.ix_ne_se_fp, self.ub.ix_nw_sw_fp, 1.0,
                            self.reg.grid_rep_times, 1, 1, 8, 8)               # ix ceil co-ordinates

        with self.tik_inst.if_scope(self.d_type == "float16"):
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'floor', self.ub.iy_nw_sw_int, self.ub.grid_ub[self.grid_ub_num],
                            self.reg.grid_rep_times_int32, 1, 1, 8, 4)               # iy floor co-ordinates
        with self.tik_inst.else_scope():
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'floor', self.ub.iy_nw_sw_int, self.ub.grid_ub[self.grid_ub_num],
                            self.reg.grid_rep_times_int32, 1, 1, 8, 8)               # iy floor co-ordinates
        
        with self.tik_inst.if_scope(self.d_type == "float16"):
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.iy_nw_sw_fp, self.ub.iy_nw_sw_int,
                            self.reg.grid_rep_times_int32, 1, 1, 4, 8, 1.0)               # iy floor co-ordinates
        with self.tik_inst.else_scope():
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.iy_nw_sw_fp, self.ub.iy_nw_sw_int,
                            self.reg.grid_rep_times_int32, 1, 1, 8, 8)               # iy floor co-ordinates
        
        self.tik_inst.vadds(Gs2Constant.B32_MASK, self.ub.iy_ne_se_int, self.ub.iy_nw_sw_int, 1,
                            self.reg.grid_rep_times_int32, 1, 1, 8, 8)               # iy ceil co-ordinates
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

    def _cubic_convolution1(self, x, out):
        self.tik_inst.vmuls(self.vmask, out, x, self.cubic_a + 2, self.reg.grid_rep_times, 1, 1, 8, 8)
        self.tik_inst.vadds(self.vmask, out, out, -3 - self.cubic_a, self.reg.grid_rep_times, 1, 1, 8, 8)
        self.tik_inst.vmul(self.vmask, out, out, x, self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vmul(self.vmask, out, out, x, self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadds(self.vmask, out, out, 1, self.reg.grid_rep_times, 1, 1, 8, 8)

    def _cubic_convolution2(self, x, out):
        self.tik_inst.vmuls(self.vmask, out, x, self.cubic_a, self.reg.grid_rep_times, 1, 1, 8, 8)
        self.tik_inst.vadds(self.vmask, out, out, self.cubic_a * (-5), self.reg.grid_rep_times, 1, 1, 8, 8)
        self.tik_inst.vmul(self.vmask, out, out, x, self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadds(self.vmask, out, out, self.cubic_a * 8, self.reg.grid_rep_times, 1, 1, 8, 8)
        self.tik_inst.vmul(self.vmask, out, out, x, self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadds(self.vmask, out, out, self.cubic_a * (-4), self.reg.grid_rep_times, 1, 1, 8, 8)

    def _get_cubic_upsampling_coefficients(self, coeffs, t):
        x12 = self.ub.weight

        self.tik_inst.vadds(self.vmask, x12, t, 1, self.reg.grid_rep_times, 1, 1, 8, 8)
        self._cubic_convolution2(x12, coeffs[0])

        self._cubic_convolution1(t, coeffs[1])

        self.tik_inst.vmuls(self.vmask, x12, t, -1, self.reg.grid_rep_times, 1, 1, 8, 8)
        self.tik_inst.vadds(self.vmask, x12, x12, 1, self.reg.grid_rep_times, 1, 1, 8, 8)
        self._cubic_convolution1(x12, coeffs[2])

        self.tik_inst.vadds(self.vmask, x12, x12, 1, self.reg.grid_rep_times, 1, 1, 8, 8)
        self._cubic_convolution2(x12, coeffs[3])

    def _cubic_interp1d(self, ii, x0, x1, x2, x3, coeff_t, out):
        ub_tmp = self.ub.val_x4
        self.tik_inst.vmul(self.vmask, out, x0, coeff_t[0][self.vmask * ii],
                           self.shape.in_c, 1, 1, 1, 8, 8, 0)

        self.tik_inst.vmul(self.vmask, ub_tmp, x1, coeff_t[1][self.vmask * ii],
                           self.shape.in_c, 1, 1, 1, 8, 8, 0)
        self.tik_inst.vadd(self.vmask, out, out, ub_tmp,
                           self.shape.in_c, 1, 1, 1, 8, 8, 8)

        self.tik_inst.vmul(self.vmask, ub_tmp, x2, coeff_t[2][self.vmask * ii],
                           self.shape.in_c, 1, 1, 1, 8, 8, 0)
        self.tik_inst.vadd(self.vmask, out, out, ub_tmp,
                           self.shape.in_c, 1, 1, 1, 8, 8, 8)

        self.tik_inst.vmul(self.vmask, ub_tmp, x3, coeff_t[3][self.vmask * ii],
                           self.shape.in_c, 1, 1, 1, 8, 8, 0)
        self.tik_inst.vadd(self.vmask, out, out, ub_tmp,
                           self.shape.in_c, 1, 1, 1, 8, 8, 8)

    def _get_value_bounded(self, n, ii, coors, masks, channel_align, x_val):
        with self.tik_inst.for_range(0, self.vmask) as iii:
            self.reg.x_offset.set_as(coors[ii * self.vmask + iii])
            self.reg.x_offset.set_as(n * self.shape.in_chw + self.reg.x_offset * self.shape.in_c)
            self._simple_data_move(self.ub.x_ub[iii * channel_align], self.gm.x_gm[self.reg.x_offset],
                                   self.shape.in_c, self.d_type)

        # [128or64, channel_align] --> [channel_align, 128or64]
        with self.tik_inst.if_scope(self.d_type == "float16"):
            self._simple_trans_fp16_128_16x(x_val, self.ub.x_ub, channel_align)
        with self.tik_inst.else_scope():
            self._simple_trans_fp32_64_8x(x_val, self.ub.x_ub, channel_align)
        
        with self.tik_inst.if_scope(self.padding_mode == Gs2Constant.PADDING_MODE_ZEROS):
            self.tik_inst.vmul(self.vmask, x_val, x_val, masks[ii * self.vmask],
                               self.shape.in_c, 1, 1, 1, 8, 8, 0)

    def _process_point_bicubic_nhwc_gt1_le64(self, n, coors, masks, coeff_tx, coeff_ty_i):
        channel_align = ceil_align(self.shape.in_c, self.d_num_1block)
        with self.tik_inst.for_range(0, self.reg.grid_cur_num // self.vmask) as ii:
            self._get_value_bounded(n, ii, coors[0], masks[0], channel_align, self.ub.val_x0)
            self._get_value_bounded(n, ii, coors[1], masks[1], channel_align, self.ub.val_x1)
            self._get_value_bounded(n, ii, coors[2], masks[2], channel_align, self.ub.val_x2)
            self._get_value_bounded(n, ii, coors[3], masks[3], channel_align, self.ub.val_x3)

            self._cubic_interp1d(ii, self.ub.val_x0, self.ub.val_x1, self.ub.val_x2, self.ub.val_x3,
                                 coeff_tx, self.ub.out_val)  # cubic_interp1d for x

            self.tik_inst.vmul(self.vmask, self.ub.x_ub, coeff_ty_i[ii * self.vmask], self.ub.out_val,
                               self.shape.in_c, 1, 1, 1, 8, 0, 8)  # cubic_interp1d for y

            # [channel_align, 128or64] --> [128or64, channel]
            with self.tik_inst.if_scope(self.d_type == "float16"):
                self._simple_trans_fp16_x_128(self.ub.out_val, self.ub.x_ub, self.shape.in_c_int32)
            with self.tik_inst.else_scope():
                self._simple_trans_fp32_x_64(self.ub.out_val, self.ub.x_ub, self.shape.in_c_int32) 

            # Move out [128or64, channel]
            self._move_to_gm(n, ii, channel_align)

        self.reg.tail_offset.set_as(floor_align(self.reg.grid_cur_num, self.vmask))
        with self.tik_inst.if_scope(self.reg.grid_cur_num > self.reg.tail_offset):
            self.reg.tail_num.set_as(self.reg.grid_cur_num - self.reg.tail_offset)
            self.reg.x_loc.set_as(self.reg.grid_cur_num / self.vmask)
            self._get_value_bounded(n, self.reg.x_loc, coors[0], masks[0], channel_align, self.ub.val_x0)
            self._get_value_bounded(n, self.reg.x_loc, coors[1], masks[1], channel_align, self.ub.val_x1)
            self._get_value_bounded(n, self.reg.x_loc, coors[2], masks[2], channel_align, self.ub.val_x2)
            self._get_value_bounded(n, self.reg.x_loc, coors[3], masks[3], channel_align, self.ub.val_x3)

            self._cubic_interp1d(self.reg.x_loc, self.ub.val_x0,
                                 self.ub.val_x1, self.ub.val_x2, self.ub.val_x3,
                                 coeff_tx, self.ub.out_val)                     # cubic_interp1d for x

            self.tik_inst.vmul(self.vmask, self.ub.x_ub,
                               coeff_ty_i[self.reg.x_loc * self.vmask], self.ub.out_val,
                               self.shape.in_c, 1, 1, 1, 8, 0, 8)         # cubic_interp1d for y

            self.tik_inst.vec_dup(self.vmask, self.ub.out_val, 0.0, Gs2Constant.OUT_VAL_NUM // self.vmask, 8)
            with self.tik_inst.for_range(0, self.reg.tail_num) as ii:
                with self.tik_inst.for_range(0, self.shape.in_c) as jj:
                    self.ub.out_val[ii * self.shape.in_c + jj].set_as(self.ub.x_ub[ii + jj * self.vmask])

            self.reg.y_offset.set_as((n * self.shape.grid_hw + self.reg.grid_hw_offset +
                                      self.reg.tail_offset) * self.shape.in_c)
            self._simple_data_move(self.gm.y_gm[self.reg.y_offset], self.ub.out_val,
                                   self.reg.tail_num * self.shape.in_c, self.d_type)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def _process_point_bicubic_nhwc_cgt64(self, n, coors, masks, coeff_tx, coeff_ty_i):
        """case NHWC and C > 64"""
        def _inner_vals(_ii):
            self.reg.xlocs[0].set_as(coors[0][_ii])
            self.reg.xlocs[1].set_as(coors[1][_ii])
            self.reg.xlocs[2].set_as(coors[2][_ii])
            self.reg.xlocs[3].set_as(coors[3][_ii])
            self.reg.xlocs[0].set_as(n * self.shape.in_chw + self.reg.xlocs[0] * self.shape.in_c)
            self.reg.xlocs[1].set_as(n * self.shape.in_chw + self.reg.xlocs[1] * self.shape.in_c)
            self.reg.xlocs[2].set_as(n * self.shape.in_chw + self.reg.xlocs[2] * self.shape.in_c)
            self.reg.xlocs[3].set_as(n * self.shape.in_chw + self.reg.xlocs[3] * self.shape.in_c)

            self.reg.mask_vals[0].set_as(masks[0][_ii])
            self.reg.mask_vals[1].set_as(masks[1][_ii])
            self.reg.mask_vals[2].set_as(masks[2][_ii])
            self.reg.mask_vals[3].set_as(masks[3][_ii])

            self.reg.coeff_vals[0].set_as(coeff_tx[0][_ii])
            self.reg.coeff_vals[1].set_as(coeff_tx[1][_ii])
            self.reg.coeff_vals[2].set_as(coeff_tx[2][_ii])
            self.reg.coeff_vals[3].set_as(coeff_tx[3][_ii])

            self.reg.x_weight.set_as(coeff_ty_i[_ii])

        def _inner_x_gm2ub_align(_offset_i, _burst):
            self.tik_inst.data_move(self.ub.val_x0, self.gm.x_gm[self.reg.xlocs[0] + _offset_i],
                                    0, 1, _burst, 0, 0)                                     # no need DataMovePad
            self.tik_inst.data_move(self.ub.val_x1, self.gm.x_gm[self.reg.xlocs[1] + _offset_i],
                                    0, 1, _burst, 0, 0)                                     # no need DataMovePad
            self.tik_inst.data_move(self.ub.val_x2, self.gm.x_gm[self.reg.xlocs[2] + _offset_i],
                                    0, 1, _burst, 0, 0)                                     # no need DataMovePad
            self.tik_inst.data_move(self.ub.val_x3, self.gm.x_gm[self.reg.xlocs[3] + _offset_i],
                                    0, 1, _burst, 0, 0)                                     # no need DataMovePad

        def _inner_x_gm2ub_unalign(_offset_i, _num):
            self._simple_data_move(self.ub.val_x0, self.gm.x_gm[self.reg.xlocs[0] + _offset_i],
                                   _num, self.d_type)
            self._simple_data_move(self.ub.val_x1, self.gm.x_gm[self.reg.xlocs[1] + _offset_i],
                                   _num, self.d_type)
            self._simple_data_move(self.ub.val_x2, self.gm.x_gm[self.reg.xlocs[2] + _offset_i],
                                   _num, self.d_type)
            self._simple_data_move(self.ub.val_x3, self.gm.x_gm[self.reg.xlocs[3] + _offset_i],
                                   _num, self.d_type)

        def _inner_mask(_num):
            self.tik_inst.vmuls(self.vmask, self.ub.val_x0, self.ub.val_x0, self.reg.mask_vals[0],
                                ceil_div(_num, self.vmask), 1, 1, 8, 8)
            self.tik_inst.vmuls(self.vmask, self.ub.val_x1, self.ub.val_x1, self.reg.mask_vals[1],
                                ceil_div(_num, self.vmask), 1, 1, 8, 8)
            self.tik_inst.vmuls(self.vmask, self.ub.val_x2, self.ub.val_x2, self.reg.mask_vals[2],
                                ceil_div(_num, self.vmask), 1, 1, 8, 8)
            self.tik_inst.vmuls(self.vmask, self.ub.val_x3, self.ub.val_x3, self.reg.mask_vals[3],
                                ceil_div(_num, self.vmask), 1, 1, 8, 8)

        def _inner_cubic_interp1d_x(_num):
            self.tik_inst.vmuls(self.vmask, self.ub.out_val, self.ub.val_x0, self.reg.coeff_vals[0],
                                ceil_div(_num, self.vmask), 1, 1, 8, 8)

            self.tik_inst.vmuls(self.vmask, self.ub.val_x4, self.ub.val_x1, self.reg.coeff_vals[1],
                                ceil_div(_num, self.vmask), 1, 1, 8, 8)
            self.tik_inst.vadd(self.vmask, self.ub.out_val, self.ub.out_val, self.ub.val_x4,
                                ceil_div(_num, self.vmask), 1, 1, 1, 8, 8, 8)

            self.tik_inst.vmuls(self.vmask, self.ub.val_x4, self.ub.val_x2, self.reg.coeff_vals[2],
                                ceil_div(_num, self.vmask), 1, 1, 8, 8)
            self.tik_inst.vadd(self.vmask, self.ub.out_val, self.ub.out_val, self.ub.val_x4,
                                ceil_div(_num, self.vmask), 1, 1, 1, 8, 8, 8)

            self.tik_inst.vmuls(self.vmask, self.ub.val_x4, self.ub.val_x3, self.reg.coeff_vals[3],
                                ceil_div(_num, self.vmask), 1, 1, 8, 8)
            self.tik_inst.vadd(self.vmask, self.ub.out_val, self.ub.out_val, self.ub.val_x4,
                                ceil_div(_num, self.vmask), 1, 1, 1, 8, 8, 8)

        with self.tik_inst.for_range(0, self.reg.grid_cur_num) as ii:
            _inner_vals(ii)

            with self.tik_inst.for_range(0, self.shape.in_c // self.grid_ub_num) as iii:
                _inner_x_gm2ub_align(self.grid_ub_num * iii, self.grid_ub_num * self.d_size // 32)

                with self.tik_inst.if_scope(self.padding_mode == Gs2Constant.PADDING_MODE_ZEROS):
                    _inner_mask(self.grid_ub_num)

                # cubic_interp1d for x
                _inner_cubic_interp1d_x(self.grid_ub_num)

                # cubic_interp1d for y
                self.tik_inst.vmuls(self.vmask, self.ub.out_val, self.ub.out_val, self.reg.x_weight,
                                    self.grid_ub_num // self.vmask, 1, 1, 8, 8)

                self.reg.y_offset.set_as((n * self.shape.grid_hw + self.reg.grid_hw_offset +
                                          ii) * self.shape.in_c + self.grid_ub_num * iii)
                self.tik_inst.data_move(self.gm.y_gm[self.reg.y_offset], self.ub.out_val, 0, 1,
                                        self.grid_ub_num * self.d_size // 32, 0, 0)     # no need DataMovePad

            self.reg.tail_offset.set_as(floor_align(self.shape.in_c, self.grid_ub_num))
            with self.tik_inst.if_scope(self.shape.in_c > self.reg.tail_offset):
                self.reg.tail_num.set_as(self.shape.in_c - self.reg.tail_offset)
                self.reg.tail_aln.set_as(floor_align(self.reg.tail_num, self.vmask))
                self.reg.tail_tail.set_as(self.reg.tail_num - self.reg.tail_aln)

                _inner_x_gm2ub_unalign(self.reg.tail_offset, self.reg.tail_num)

                with self.tik_inst.if_scope(self.padding_mode == Gs2Constant.PADDING_MODE_ZEROS):
                    _inner_mask(self.reg.tail_num)

                # cubic_interp1d for x
                _inner_cubic_interp1d_x(self.reg.tail_num)

                # cubic_interp1d for y
                self.tik_inst.vec_dup(self.vmask, self.ub.x_ub, 0.0, ceil_div(self.reg.tail_num, self.vmask), 8)
                self.tik_inst.vmuls(self.reg.tail_num, self.ub.x_ub, self.ub.out_val, self.reg.x_weight,
                                    1, 1, 1, 8, 8, 0, "counter")

                # ub to gm
                self.reg.y_offset.set_as((n * self.shape.grid_hw + self.reg.grid_hw_offset +
                                          ii) * self.shape.in_c + self.reg.tail_offset)
                self._simple_data_move(self.gm.y_gm[self.reg.y_offset], self.ub.x_ub, self.reg.tail_num, self.d_type)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def _process_point_bicubic_nchw(self, n, coors, masks, coeff_tx, coeff_ty_i):
        """Support case NCHW or NHW1. However, these scenarios currently do not occur."""
        def _inner_gm2ub_unroll(_val_x, _coors, _ic, _ii):
            for j in range(Gs2Constant.UNROLL_NUM):
                self.reg.x_loc.set_as(_coors[_ii * Gs2Constant.UNROLL_NUM + j])
                self.reg.xlocs[j].set_as(n * self.shape.in_chw + _ic * self.shape.in_hw + self.reg.x_loc)
            for j in range(Gs2Constant.UNROLL_NUM):
                self._simple_data_move(self.ub.x_ub[j * self.d_num_1block],
                                       self.gm.x_gm[self.reg.xlocs[j]], 1, self.d_type)
            for j in range(Gs2Constant.UNROLL_NUM):
                _val_x[_ii * Gs2Constant.UNROLL_NUM + j].set_as(self.ub.x_ub[j * self.d_num_1block])

        def _inner_gm2ub_tail(_val_x, _coors, _ic, _ii):
            self.reg.x_loc.set_as(_coors[_ii])
            self.reg.x_loc.set_as(n * self.shape.in_chw + _ic * self.shape.in_hw + self.reg.x_loc)
            self._simple_data_move(self.ub.x_ub, self.gm.x_gm[self.reg.x_loc], 1, self.d_type)
            _val_x[_ii].set_as(self.ub.x_ub)

        def _inner_mask(_val_x, _masks):
            self.tik_inst.vmul(self.vmask, _val_x, _val_x, _masks,
                               self.grid_ub_num // self.vmask, 1, 1, 1, 8, 8, 8)

        def _inner_cubic_interp1d_x():
            self.tik_inst.vmul(self.vmask, self.ub.out_val, self.ub.val_x0, coeff_tx[0],
                               self.grid_ub_num // self.vmask, 1, 1, 1, 8, 8, 8)

            self.tik_inst.vmul(self.vmask, self.ub.val_x4, self.ub.val_x1, coeff_tx[1],
                               self.grid_ub_num // self.vmask, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vadd(self.vmask, self.ub.out_val, self.ub.out_val, self.ub.val_x4,
                               self.grid_ub_num // self.vmask, 1, 1, 1, 8, 8, 8)

            self.tik_inst.vmul(self.vmask, self.ub.val_x4, self.ub.val_x2, coeff_tx[2],
                               self.grid_ub_num // self.vmask, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vadd(self.vmask, self.ub.out_val, self.ub.out_val, self.ub.val_x4,
                               self.grid_ub_num // self.vmask, 1, 1, 1, 8, 8, 8)

            self.tik_inst.vmul(self.vmask, self.ub.val_x4, self.ub.val_x3, coeff_tx[3],
                               self.grid_ub_num // self.vmask, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vadd(self.vmask, self.ub.out_val, self.ub.out_val, self.ub.val_x4,
                               self.grid_ub_num // self.vmask, 1, 1, 1, 8, 8, 8)

        def _inner_cubic_interp1d_y():
            self.tik_inst.vec_dup(self.vmask, self.ub.x_ub, 0.0, ceil_div(self.reg.grid_cur_num, self.vmask), 8)
            with self.tik_inst.if_scope(self.reg.grid_cur_num > self.vmask):
                self.tik_inst.vmul(self.vmask, self.ub.x_ub, self.ub.out_val, coeff_ty_i,
                                   ceil_div(self.reg.grid_cur_num, self.vmask), 1, 1, 1, 8, 8, 8)

            self.reg.tail_offset.set_as(floor_align(self.reg.grid_cur_num, self.vmask))
            with self.tik_inst.if_scope(self.reg.grid_cur_num > self.reg.tail_offset):
                self.reg.tail_num.set_as(self.reg.grid_cur_num - self.reg.tail_offset)
                self.tik_inst.vmul(self.reg.tail_num, self.ub.x_ub[self.reg.tail_offset],
                                   self.ub.out_val[self.reg.tail_offset], coeff_ty_i[self.reg.tail_offset],
                                   1, 1, 1, 1, 8, 8, 8)

        with self.tik_inst.for_range(0, self.shape.in_c) as i_c:
            self.tik_inst.vec_dup(self.vmask, self.ub.out_val, 0.0, self.reg.grid_rep_times, 8)
            with self.tik_inst.for_range(0, self.reg.grid_cur_num / Gs2Constant.UNROLL_NUM) as ii:
                _inner_gm2ub_unroll(self.ub.val_x0, coors[0], i_c, ii)
                _inner_gm2ub_unroll(self.ub.val_x1, coors[1], i_c, ii)
                _inner_gm2ub_unroll(self.ub.val_x2, coors[2], i_c, ii)
                _inner_gm2ub_unroll(self.ub.val_x3, coors[3], i_c, ii)

            with self.tik_inst.for_range(floor_align(self.reg.grid_cur_num, Gs2Constant.UNROLL_NUM),
                                         self.reg.grid_cur_num) as ii:
                _inner_gm2ub_tail(self.ub.val_x0, coors[0], i_c, ii)
                _inner_gm2ub_tail(self.ub.val_x1, coors[1], i_c, ii)
                _inner_gm2ub_tail(self.ub.val_x2, coors[2], i_c, ii)
                _inner_gm2ub_tail(self.ub.val_x3, coors[3], i_c, ii)

            with self.tik_inst.if_scope(self.padding_mode == Gs2Constant.PADDING_MODE_ZEROS):
                _inner_mask(self.ub.val_x0, masks[0])
                _inner_mask(self.ub.val_x1, masks[1])
                _inner_mask(self.ub.val_x2, masks[2])
                _inner_mask(self.ub.val_x3, masks[3])

            _inner_cubic_interp1d_x()
            _inner_cubic_interp1d_y()

            # move to gm
            self.reg.y_offset.set_as(n * self.shape.out_chw + i_c * self.shape.grid_hw + self.reg.grid_hw_offset)
            self._simple_data_move(self.gm.y_gm[self.reg.y_offset], self.ub.x_ub, self.reg.grid_cur_num, self.d_type)

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def _process_point_bicubic(self, n, coors, masks, coeff_tx, coeff_ty_i):
        with self.tik_inst.if_scope(tik.any(self.shape.in_c == 1,
                                            tik.all(self.args.channel_last == 1, self.shape.in_c <= 64))):
            self._process_point_bicubic_nhwc_gt1_le64(n, coors, masks, coeff_tx, coeff_ty_i)
        with self.tik_inst.elif_scope(tik.all(self.args.channel_last == 1, self.shape.in_c > 64)):
            self._process_point_bicubic_nhwc_cgt64(n, coors, masks, coeff_tx, coeff_ty_i)
        with self.tik_inst.elif_scope(self.args.channel_last == 0):
            self._process_point_bicubic_nchw(n, coors, masks, coeff_tx, coeff_ty_i)

    # 'pylint: disable=too-many-statements
    def _compute_val_bicubic(self, n):
        with self.tik_inst.if_scope(self.d_type == "float16"):
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'floor', self.ub.ix_nw_sw_int, self.ub.grid_ub,
                            self.reg.grid_rep_times_int32, 1, 1, 8, 4)               # ix floor co-ordinates
        with self.tik_inst.else_scope():
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'floor', self.ub.ix_nw_sw_int, self.ub.grid_ub,
                            self.reg.grid_rep_times_int32, 1, 1, 8, 8)               # ix floor co-ordinates
        
        with self.tik_inst.if_scope(self.d_type == "float16"):
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.ix_nw_sw_fp, self.ub.ix_nw_sw_int,
                            self.reg.grid_rep_times_int32, 1, 1, 4, 8, 1.0)          # ix floor co-ordinates
        with self.tik_inst.else_scope():
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.ix_nw_sw_fp, self.ub.ix_nw_sw_int,
                            self.reg.grid_rep_times_int32, 1, 1, 8, 8)               # ix floor co-ordinates
        
        with self.tik_inst.if_scope(self.d_type == "float16"):
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'floor', self.ub.iy_nw_sw_int, self.ub.grid_ub[self.grid_ub_num],
                            self.reg.grid_rep_times_int32, 1, 1, 8, 4)               # iy floor co-ordinates
        with self.tik_inst.else_scope():
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'floor', self.ub.iy_nw_sw_int, self.ub.grid_ub[self.grid_ub_num],
                            self.reg.grid_rep_times_int32, 1, 1, 8, 8)               # iy floor co-ordinates      

        with self.tik_inst.if_scope(self.d_type == "float16"):
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.iy_nw_sw_fp, self.ub.iy_nw_sw_int,
                            self.reg.grid_rep_times_int32, 1, 1, 4, 8, 1.0)               # iy floor co-ordinates
        with self.tik_inst.else_scope():
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.iy_nw_sw_fp, self.ub.iy_nw_sw_int,
                            self.reg.grid_rep_times_int32, 1, 1, 8, 8)               # iy floor co-ordinates

        ix_int, ix_fp = self.ub.ix_nw_sw_int, self.ub.ix_nw_sw_fp
        iy_int, iy_fp = self.ub.iy_nw_sw_int, self.ub.iy_nw_sw_fp
        x_i, x_f = self.ub.ix_ne_se_int, self.ub.ix_ne_se_fp
        y_i, y_f = self.ub.iy_ne_se_int, self.ub.iy_ne_se_fp

        self.tik_inst.vsub(self.vmask, self.ub.cubic_tx, self.ub.grid_ub, ix_fp,
                           self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vsub(self.vmask, self.ub.cubic_ty, self.ub.grid_ub[self.grid_ub_num], iy_fp,
                           self.reg.grid_rep_times, 1, 1, 1, 8, 8, 8)

        coeff_tx = [self.ub.coeff_tx0, self.ub.coeff_tx1, self.ub.coeff_tx2, self.ub.coeff_tx3]
        self._get_cubic_upsampling_coefficients(coeff_tx, self.ub.cubic_tx)

        coeff_ty = [self.ub.coeff_ty0, self.ub.coeff_ty1, self.ub.coeff_ty2, self.ub.coeff_ty3]
        self._get_cubic_upsampling_coefficients(coeff_ty, self.ub.cubic_ty)

        coors = [self.ub.coor_x00, self.ub.coor_x01, self.ub.coor_x02, self.ub.coor_x03]
        masks = [self.ub.clip_mask00, self.ub.clip_mask01, self.ub.clip_mask02, self.ub.clip_mask03]

        for ci in range(4):
            self.tik_inst.vadds(Gs2Constant.B32_MASK, y_i, iy_int, ci - 1, self.reg.grid_rep_times_int32, 1, 1, 8, 8)
            self.tik_inst.vadds(self.vmask, y_f, iy_fp, ci - 1, self.reg.grid_rep_times, 1, 1, 8, 8)

            self.tik_inst.vadds(Gs2Constant.B32_MASK, x_i, ix_int, -1, self.reg.grid_rep_times_int32, 1, 1, 8, 8)
            self.tik_inst.vadds(self.vmask, x_f, ix_fp, -1, self.reg.grid_rep_times, 1, 1, 8, 8)
            self._clip_coordinates(x_f, y_f, x_i, y_i, coors[0], masks[0])

            self._clip_coordinates(ix_fp, y_f, ix_int, y_i, coors[1], masks[1])

            self.tik_inst.vadds(Gs2Constant.B32_MASK, x_i, ix_int, 1, self.reg.grid_rep_times_int32, 1, 1, 8, 8)
            self.tik_inst.vadds(self.vmask, x_f, ix_fp, 1, self.reg.grid_rep_times, 1, 1, 8, 8)
            self._clip_coordinates(x_f, y_f, x_i, y_i, coors[2], masks[2])

            self.tik_inst.vadds(Gs2Constant.B32_MASK, x_i, ix_int, 2, self.reg.grid_rep_times_int32, 1, 1, 8, 8)
            self.tik_inst.vadds(self.vmask, x_f, ix_fp, 2, self.reg.grid_rep_times, 1, 1, 8, 8)
            self._clip_coordinates(x_f, y_f, x_i, y_i, coors[3], masks[3])

            self._process_point_bicubic(n, coors, masks, coeff_tx, coeff_ty[ci])

    def _compute_val(self, n):
        if self.interpolation_mode == Gs2Constant.INTERPOLATION_MODE_BILINEAR and self.d_type == "float16":
            self._compute_val_bilinear_fp16(n)
        elif self.interpolation_mode == Gs2Constant.INTERPOLATION_MODE_BILINEAR and self.d_type == "float32":
            self._compute_val_bilinear(n)
        elif self.interpolation_mode == Gs2Constant.INTERPOLATION_MODE_BICUBIC:
            self._compute_val_bicubic(n)
        else:
            pass

    def _unnormalize(self):
        with self.tik_inst.if_scope(self.d_type == "float16"):
            x_ub_fp32 = self.ub.x_ub.reinterpret_cast_to("float32")
            self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', x_ub_fp32, self.ub.grid_ub,
                           (self.reg.grid_cur_num_aln * 2 + 63) // 64, 1, 1, 8, 4)
            with self.tik_inst.if_scope(self.reg.grid_cur_num_aln * 2 // 64 > 0):
                self.tik_inst.vadds(64, x_ub_fp32, x_ub_fp32, 1.0,
                                    self.reg.grid_cur_num_aln * 2 // 64, 1, 1, 8, 8)
            with self.tik_inst.if_scope(self.reg.grid_cur_num_aln * 2 % 64 > 0):
                self.tik_inst.vadds(self.reg.grid_cur_num_aln * 2 % 64,
                                    x_ub_fp32[self.reg.grid_cur_num_aln * 2 // 64 * 64],
                                    x_ub_fp32[self.reg.grid_cur_num_aln * 2 // 64 * 64],
                                    1.0, 1, 1, 1, 8, 8)
        with self.tik_inst.else_scope():
            with self.tik_inst.if_scope(self.reg.grid_cur_num_aln * 2 // self.vmask > 0):
                self.tik_inst.vadds(self.vmask, self.ub.ixy_fp, self.ub.grid_ub, 1.0,
                                    self.reg.grid_cur_num_aln * 2 // self.vmask, 1, 1, 8, 8)
            with self.tik_inst.if_scope(self.reg.grid_cur_num_aln * 2 % self.vmask > 0):
                self.tik_inst.vadds(self.reg.grid_cur_num_aln * 2 % self.vmask,
                                    self.ub.ixy_fp[self.reg.grid_cur_num_aln * 2 // self.vmask * self.vmask],
                                    self.ub.grid_ub[self.reg.grid_cur_num_aln * 2 // self.vmask * self.vmask],
                                    1.0, 1, 1, 1, 8, 8)
        with self.tik_inst.if_scope(self.d_type == "float16"):
            self.tik_inst.vreduce(self.grid_ub_num * 2, x_ub_fp32[1024], x_ub_fp32,
                                1, 1, 1, 8, 0, 0, None, 'counter')
            self.tik_inst.vreduce(self.grid_ub_num * 2, x_ub_fp32[1024 + self.grid_ub_num], x_ub_fp32,
                                2, 1, 1, 8, 0, 0, None, 'counter')
        with self.tik_inst.else_scope():
            self.tik_inst.vreduce(self.grid_ub_num * 2, self.ub.grid_ub, self.ub.ixy_fp,
                                1, 1, 1, 8, 0, 0, None, 'counter')
            self.tik_inst.vreduce(self.grid_ub_num * 2, self.ub.grid_ub[self.grid_ub_num], self.ub.ixy_fp,
                                2, 1, 1, 8, 0, 0, None, 'counter')

        with self.tik_inst.if_scope(self.d_type == "float16"):
            with self.tik_inst.if_scope(self.args.align_corners == 1):
            # unnormalize coord from [-1, 1] to [0, size - 1]
                self.tik_inst.vmuls(64, x_ub_fp32, x_ub_fp32[1024],
                                    0.5 * (self.shape.in_w - 1.0), self.reg.grid_rep_times_int32,
                                    1, 1, 8, 8)                                    # ix
                self.tik_inst.vmuls(64, x_ub_fp32[self.grid_ub_num], x_ub_fp32[1024 + self.grid_ub_num],
                                    0.5 * (self.shape.in_h - 1.0), self.reg.grid_rep_times_int32,
                                    1, 1, 8, 8)                                    # iy
                self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.grid_ub, x_ub_fp32,
                                    self.reg.grid_rep_times_int32, 1, 1, 4, 8)
                self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.grid_ub[self.grid_ub_num],
                                    x_ub_fp32[self.grid_ub_num], self.reg.grid_rep_times_int32, 1, 1, 4, 8)
            with self.tik_inst.else_scope():
                # unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
                self.tik_inst.vmuls(64, x_ub_fp32, x_ub_fp32[1024],
                                    0.5 * self.shape.in_w, self.reg.grid_rep_times_int32,
                                    1, 1, 8, 8)                                    # ix
                self.tik_inst.vmuls(64, x_ub_fp32[self.grid_ub_num], x_ub_fp32[1024 + self.grid_ub_num],
                                    0.5 * self.shape.in_h, self.reg.grid_rep_times_int32,
                                    1, 1, 8, 8)                                    # iy
                self.tik_inst.vadds(64, x_ub_fp32, x_ub_fp32,
                                    -0.5, self.reg.grid_rep_times_int32, 1, 1, 8, 8)
                self.tik_inst.vadds(64, x_ub_fp32[self.grid_ub_num], x_ub_fp32[self.grid_ub_num],
                                    -0.5, self.reg.grid_rep_times_int32, 1, 1, 8, 8)
                self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.grid_ub, x_ub_fp32,
                                    self.reg.grid_rep_times_int32, 1, 1, 4, 8)
                self.tik_inst.vconv(Gs2Constant.B32_MASK, 'none', self.ub.grid_ub[self.grid_ub_num],
                                    x_ub_fp32[self.grid_ub_num], self.reg.grid_rep_times_int32, 1, 1, 4, 8)
        with self.tik_inst.else_scope():
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
        self.reg.grid_rep_times_int32.set_as(util_common.ceil_div_scalar(self.reg.grid_cur_num_aln,
                                                                         Gs2Constant.B32_MASK))

        self._compute_hw(i_n)

    def _compute_n(self, core_id, i_n, core_num_cur_group):
        hw_loop_times = self.tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="hw_loop_times")
        hw_num_cur_core = self.tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="hw_num_cur_core")
        hw_offset_cur_loop = self.tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64,
                                                  name="hw_offset_cur_loop", init_value=0)

        with self.tik_inst.if_scope(self.args.tiling_mode == Gs2Constant.TILING_MODE_1):
            hw_num_cur_core.set_as(self.shape.grid_hw)
            hw_loop_times.set_as(ceil_div(self.shape.grid_hw, self.grid_ub_num))
        with self.tik_inst.elif_scope(self.args.tiling_mode == Gs2Constant.TILING_MODE_2):
            hw_num_cur_core.set_as(ceil_div(self.shape.grid_hw, core_num_cur_group))
            hw_offset_cur_loop.set_as(core_id % core_num_cur_group * hw_num_cur_core)
            with self.tik_inst.if_scope(tik.all(self.shape.grid_hw % core_num_cur_group > 0,
                                        core_id % core_num_cur_group >= self.shape.grid_hw % core_num_cur_group)):
                hw_num_cur_core.set_as(self.shape.grid_hw // core_num_cur_group)
                hw_offset_cur_loop.set_as(core_id % core_num_cur_group * hw_num_cur_core +
                                          self.shape.grid_hw % core_num_cur_group)
            hw_loop_times.set_as(ceil_div(hw_num_cur_core, self.grid_ub_num))
        with self.tik_inst.elif_scope(self.args.tiling_mode == Gs2Constant.TILING_MODE_3):
            hw_num_cur_core.set_as(self.shape.grid_hw)
            hw_loop_times.set_as(ceil_div(self.shape.grid_hw, self.grid_ub_num))
        with self.tik_inst.else_scope():
            hw_num_cur_core.set_as(ceil_div(self.shape.grid_hw, self.args.need_core_num))
            hw_offset_cur_loop.set_as(core_id * hw_num_cur_core)
            with self.tik_inst.if_scope(tik.all(self.shape.grid_hw % self.args.need_core_num > 0,
                                                core_id >= self.shape.grid_hw % self.args.need_core_num)):
                hw_num_cur_core.set_as(self.shape.grid_hw // self.args.need_core_num)
                hw_offset_cur_loop.set_as(core_id * hw_num_cur_core + self.shape.grid_hw % self.args.need_core_num)
            hw_loop_times.set_as(ceil_div(hw_num_cur_core, self.grid_ub_num))

        # Assert `hw_loop_times > 0`
        with self.tik_inst.for_range(0, hw_loop_times) as i_hw:
            self._compute_one_loop(i_n, i_hw, hw_offset_cur_loop, hw_loop_times, hw_num_cur_core)

    def _compute_one_core(self, core_id):
        n_start = self.tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="n_start")
        n_num = self.tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="n_num")
        core_num_cur_group = self.tik_inst.Scalar(dtype=constant.DATA_TYPE_INT64, name="core_num_cur_group")

        # Assert: `need_core_num > 0`
        with self.tik_inst.if_scope(self.shape.in_n % self.args.need_core_num == 0):
            # Case 1: `1 < N <  core_num      and core_num % N != 0`
            # Case 2: `    N >= core_num >= 1 and core_num | N`
            # Solution: split core by N
            #           grid_hw are compute on one core
            self.args.tiling_mode.set_as(Gs2Constant.TILING_MODE_1)
            n_num.set_as(self.shape.in_n // self.args.need_core_num)
            n_start.set_as(n_num * core_id)
        with self.tik_inst.elif_scope(tik.all(self.args.need_core_num % self.shape.in_n == 0,
                                              self.shape.grid_hw >= self.d_num_1block * self.args.need_core_num)):
            # Case 1: `1 == N                              and HW >= xxx`
            # Case 2: `1 <  N <= core_num and N | core_num and HW >= xxx`
            # Solution: split core to some group, compute n on one group
            #           grid_hw are compute on one core or more than one core
            self.args.tiling_mode.set_as(Gs2Constant.TILING_MODE_2)
            n_num.set_as(1)
            core_num_cur_group.set_as(self.args.need_core_num // self.shape.in_n)
            n_start.set_as(core_id // core_num_cur_group)
        with self.tik_inst.elif_scope(tik.any(self.shape.grid_hw < 1024 * self.args.need_core_num,
                                              self.shape.in_n >= 2 * self.args.need_core_num)):
            # Case 1: `grid_hw < 1024 * core_num`
            # Case 2: `n >= 2 * core_num`
            # Assert: `N > core_num and N % core_num > 0`
            # Solution: split core by N
            #           grid_hw are compute on one core
            self.args.tiling_mode.set_as(Gs2Constant.TILING_MODE_3)
            with self.tik_inst.if_scope(core_id < self.shape.in_n % self.args.need_core_num):
                n_num.set_as(ceil_div(self.shape.in_n, self.args.need_core_num))
                n_start.set_as(n_num * core_id)
            with self.tik_inst.else_scope():
                n_num.set_as(self.shape.in_n // self.args.need_core_num)
                n_start.set_as(self.shape.in_n % self.args.need_core_num + n_num * core_id)
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


# 'pylint: disable=too-many-arguments,huawei-too-many-arguments
@register_operator("GridSampler2D")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def grid_sampler_2d(x, grid, y, interpolation_mode="bilinear", padding_mode="zeros", align_corners=False,
                    kernel_name="grid_sampler_2d"):
    """
    Compute GridSampler2D

    Parameters
    ----------
    x : dict. shape and dtype of input data x
    grid : dict. shape and dtype of input data grid
    y : dict. shape and dtype of input data y
    interpolation_mode : value of attr interpolation_mode
    padding_mode : value of attr padding_mode
    align_corners : value of attr align_corners
    kernel_name : str. cce kernel name, default value is "grid_sampler_2d"

    Returns
    -------
    None
    """
    ret1, msg1 = _validate_support_check(x, grid, y, interpolation_mode, padding_mode, align_corners)
    if not ret1:
        error_manager_vector.raise_err_specific_reson(kernel_name, msg1)

    tik_inst = tik.Tik(tik.Dprofile)
    ret2, _ = _check_support_static_mini_ih_iw(x, grid, y, interpolation_mode, padding_mode, align_corners)
    if ret2:
        obj2 = GridSampler2DMiniIhIw(x, grid, y, interpolation_mode, padding_mode, align_corners,
                                     kernel_name, tik_inst)
        obj2.compute()
        return

    obj3 = GridSampler2D4MiniCihiw(x, grid, y, interpolation_mode, padding_mode, align_corners,
                                   kernel_name, tik_inst)
    ret3, _ = obj3.check_params()
    if ret3:
        obj3.compute()
        return

    ret4, msg4 = _check_support_general(x, grid, y, interpolation_mode, padding_mode, align_corners)
    if ret4:
        obj4 = GridSampler2D(x, grid, y, interpolation_mode, padding_mode, align_corners,
                             kernel_name, tik_inst)
        obj4.compute()
    else:
        error_manager_vector.raise_err_specific_reson(kernel_name, msg4)
