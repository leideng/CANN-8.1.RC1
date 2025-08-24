#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2019 Huawei Technologies Co., Ltd
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
roi_align_vbi
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tik

NEG_ONE = -1
HALF = 0.5
ONE = 1
TWO = 2
THREE = 3
FOUR = 4
FIVE = 5
SIX = 6
SEVEN = 7
EIGHT = 8
SIXTEEN = 16
NINETY_SIX = 96

NoneType = type(None)
POOL_H = SEVEN
POOL_W = SEVEN
SAMPLING_RATIO = 1
NUM_SAMPLING_W = POOL_W * SAMPLING_RATIO
VBI_NUM_BLOCKS_ONEROW = (POOL_H * POOL_W + SEVEN) // EIGHT
VBI_NUM_ELEMENTS_ONEROW = VBI_NUM_BLOCKS_ONEROW * EIGHT
NUM_ELEMENTS_ONEROW = VBI_NUM_ELEMENTS_ONEROW * 2
NUM_ELMENTS_ONEBIN = SAMPLING_RATIO * SAMPLING_RATIO * FOUR
VBI_TOTAL_ELEMENTS = VBI_NUM_ELEMENTS_ONEROW * NUM_ELMENTS_ONEBIN

STRIDE_H = POOL_H * SAMPLING_RATIO - ONE
STRIDE_W = POOL_W * SAMPLING_RATIO - ONE
C0SIZE = SIXTEEN
BYTES = 2

BLOCK_DIM = EIGHT

ROINUM_LIMIT = 128
FM_BUFFER_SIZE_LIMIT = 150 * 1024
MAX_NUM_GRIDW = 128

ROI_PARA_UNIT = 64
ROI_BATCH_UNIT = 64
L1_ADDR_GRID_PARA = 0
NUM_ADDR_ONE_VBI = 4 * 8
FP_SIXTEEN_ONE_TIME = 128
FP_32_ONE_TIME = 64


# 'pylint: disable=super-with-arguments
# 'pylint: disable=too-many-locals,too-many-arguments,too-many-statements,too-many-lines,unused-argument
def _roi_align_perf_scale(tik_instance, rois, feature_map_h, feature_map_w):
    """
    calculate the pos of roi box and  wide and height of grid
    :param tik_instance:
    :param rois: the coordinates of the roi box
    :return: the pos of roi box and  wide and height of grid
    """
    zero = tik_instance.Scalar(dtype="float32", init_value=0.0)
    repeat = ROINUM_LIMIT // SIXTEEN
    tmp_buf = tik_instance.Tensor("float16", [FOUR, ROINUM_LIMIT], \
                                  name="tmp_buf",
                                  scope=tbe_platform.scope_ubuf)
    # x1, y1, x2, y2
    tik_instance.vreduce(FP_SIXTEEN_ONE_TIME, tmp_buf[0, 0], rois, THREE,
                         FOUR, ONE, EIGHT, 0, 0, None, "normal")
    tik_instance.vreduce(FP_SIXTEEN_ONE_TIME, tmp_buf[ONE, 0], rois, FOUR,
                         FOUR, ONE, EIGHT, 0, 0, None, "normal")
    tik_instance.vreduce(FP_SIXTEEN_ONE_TIME, tmp_buf[TWO, 0], rois, FIVE,
                         FOUR, ONE, EIGHT, 0, 0, None, "normal")
    tik_instance.vreduce(FP_SIXTEEN_ONE_TIME, tmp_buf[THREE, 0], rois, SIX,
                         FOUR, ONE, EIGHT, 0, 0, None, "normal")
    rois_fp32 = tik_instance.Tensor("float32",
                                    [FOUR, repeat * SIXTEEN],
                                    name="rois_fp32",
                                    scope=tbe_platform.scope_ubuf)
    rois_fp32_orig = tik_instance.Tensor("float32", [TWO, repeat * SIXTEEN],
                                         name="rois_fp32_orig",
                                         scope=tbe_platform.scope_ubuf)
    tik_instance.vconv(FP_32_ONE_TIME, "", rois_fp32, tmp_buf, TWO * FOUR,
                       ONE, ONE, EIGHT, FOUR)
    tik_instance.vadds(FP_32_ONE_TIME, rois_fp32_orig[0, 0], rois_fp32[TWO, 0],
                       zero, TWO, ONE, ONE, EIGHT, EIGHT)
    tik_instance.vadds(FP_32_ONE_TIME, rois_fp32_orig[ONE, 0],
                       rois_fp32[THREE, 0], zero, TWO, ONE, ONE, EIGHT, EIGHT)
    # height and width of each RoI
    tik_instance.vsub(FP_32_ONE_TIME, rois_fp32[THREE, 0],
                      rois_fp32[THREE, 0], rois_fp32[ONE, 0], TWO, ONE,
                      ONE, ONE, EIGHT, EIGHT, EIGHT)
    tik_instance.vsub(FP_32_ONE_TIME, rois_fp32[TWO, 0], rois_fp32[TWO, 0],
                      rois_fp32[0, 0], TWO, ONE, ONE, ONE, EIGHT, EIGHT, EIGHT)

    grid_hw_fp32 = tik_instance.Tensor("float32", [TWO, FP_SIXTEEN_ONE_TIME],
                                       name="rois_fp32",
                                       scope=tbe_platform.scope_ubuf)
    grid_h_fp32 = grid_hw_fp32[0, 0]
    grid_w_fp32 = grid_hw_fp32[ONE, 0]


    scale_grid_hw = \
        tik_instance.Scalar(dtype="float32",
                            init_value=ONE / (POOL_H * SAMPLING_RATIO))
    tik_instance.vmuls(FP_32_ONE_TIME, grid_h_fp32, rois_fp32[THREE, 0],
                       scale_grid_hw, TWO, ONE, ONE, EIGHT, EIGHT)
    tik_instance.vmuls(FP_32_ONE_TIME, grid_w_fp32, rois_fp32[TWO, 0],
                       scale_grid_hw, TWO, ONE, ONE, EIGHT, EIGHT)
    return rois_fp32, grid_hw_fp32, rois_fp32_orig


def _roi_align_perf_gengrid_fp32(tik_instance, curr_roi, rois_fp32,
                                 grid_hw_fp32,
                                 feature_shape, index_array_fp32):
    """
    :param tik_instance:
    :param curr_roi: the number of roi box
    :param rois_fp32: the position of roi box
    :param grid_hw_fp32: the wide and  height  of grid in roi box
    :param feature_shape: the shape of the input featuremap
    :param index_array_fp32: a index, corresponding to center of grid
    :return: the position of 4 pixels around the center of gird ;
            and distance to them:
            x_low_int y_low_int x_high_int y_high_int
    """
    # lx ly hx hy
    point_weights_fp32 = tik_instance.Tensor(
        "float32", [FOUR, ROI_PARA_UNIT], name="point_weights_fp32",
        scope=tbe_platform.scope_ubuf)
    point_positions_int32 = tik_instance.Tensor(
        "int32", [FOUR, ROI_PARA_UNIT], name="point_positions_int32",
        scope=tbe_platform.scope_ubuf)
    point_positions_fp32 = tik_instance.Tensor(
        "float32", [FOUR, ROI_PARA_UNIT], name="pointPositionFp32",
        scope=tbe_platform.scope_ubuf)
    delta_w = tik_instance.Scalar(dtype="float32",
                                  init_value=grid_hw_fp32[
                                      ONE, curr_roi])
    delta_h = tik_instance.Scalar(dtype="float32",
                                  init_value=grid_hw_fp32[
                                      0, curr_roi])

    w_start = tik_instance.Scalar(dtype="float32",
                                  init_value=rois_fp32[0, curr_roi])

    h_start = tik_instance.Scalar(dtype="float32",
                                  init_value=rois_fp32[ONE, curr_roi])

    height = feature_shape[TWO]
    width = feature_shape[THREE]

    point_positions = tik_instance.Tensor(
        "float32", [TWO, ROI_PARA_UNIT], name="point_positions",
        scope=tbe_platform.scope_ubuf)
    x_pos_fp32 = point_positions[0, 0]
    y_pos_fp32 = point_positions[ONE, 0]

    tik_instance.vmuls(FP_32_ONE_TIME, x_pos_fp32, index_array_fp32[0, ],
                       delta_w, ONE, ONE, ONE, EIGHT, EIGHT)
    tik_instance.vmuls(FP_32_ONE_TIME, y_pos_fp32, index_array_fp32[0, ],
                       delta_h, ONE, ONE, ONE, EIGHT, EIGHT)
    tik_instance.vadds(FP_32_ONE_TIME, x_pos_fp32, x_pos_fp32, w_start,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    tik_instance.vadds(FP_32_ONE_TIME, y_pos_fp32, y_pos_fp32, h_start,
                       ONE, ONE, ONE, EIGHT, EIGHT)

    # need to substract 0.5 in TensorFlow
    neg_point_five = tik_instance.Scalar(dtype="float32",
                                         init_value=NEG_ONE * HALF)
    tik_instance.vadds(FP_32_ONE_TIME, x_pos_fp32, x_pos_fp32, neg_point_five,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    tik_instance.vadds(FP_32_ONE_TIME, y_pos_fp32, y_pos_fp32, neg_point_five,
                       ONE, ONE, ONE, EIGHT, EIGHT)

    x_low_int = point_positions_int32[0, 0]
    y_low_int = point_positions_int32[ONE, 0]
    x_high_int = point_positions_int32[TWO, 0]
    y_high_int = point_positions_int32[THREE, 0]
    tik_instance.vconv(FP_32_ONE_TIME, "floor", x_low_int, x_pos_fp32,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    tik_instance.vconv(FP_32_ONE_TIME, "floor", y_low_int, y_pos_fp32,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    x_low_fp32 = point_positions_fp32[0, 0]
    y_low_fp32 = point_positions_fp32[ONE, 0]
    x_high_fp32 = point_positions_fp32[TWO, 0]
    y_high_fp32 = point_positions_fp32[THREE, 0]

    tik_instance.vconv(FP_32_ONE_TIME, "", x_low_fp32, x_low_int, ONE,
                       ONE, ONE, EIGHT, EIGHT)
    tik_instance.vconv(FP_32_ONE_TIME, "", y_low_fp32, y_low_int, ONE,
                       ONE, ONE, EIGHT, EIGHT)
    point_five = tik_instance.Scalar(dtype="float32", init_value=HALF)
    one = tik_instance.Scalar(dtype="float32", init_value=ONE)
    neg_one = tik_instance.Scalar(dtype="float32", init_value=NEG_ONE)
    zero = tik_instance.Scalar(dtype="float32", init_value=0.0)
    tik_instance.vadds(FP_32_ONE_TIME, x_high_fp32, x_low_fp32, point_five,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    tik_instance.vadds(FP_32_ONE_TIME, y_high_fp32, y_low_fp32, point_five,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    tik_instance.vconv(FP_32_ONE_TIME, "ceil", x_high_int, x_high_fp32,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    tik_instance.vconv(FP_32_ONE_TIME, "ceil", y_high_int, y_high_fp32,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    tik_instance.vadds(FP_32_ONE_TIME, x_high_fp32, x_low_fp32, one, ONE, ONE,
                       ONE, EIGHT, EIGHT)
    tik_instance.vadds(FP_32_ONE_TIME, y_high_fp32, y_low_fp32, one, ONE, ONE,
                       ONE, EIGHT, EIGHT)
    # get xhigh/yhigh for all bins, in x_high_int, y_high_int
    # lx, ly, hx, hy are the weights for interpolation
    lx_fp32 = point_weights_fp32[0, 0]
    ly_fp32 = point_weights_fp32[ONE, 0]
    hx_fp32 = point_weights_fp32[TWO, 0]
    hy_fp32 = point_weights_fp32[THREE, 0]
    tik_instance.vsub(FP_32_ONE_TIME, lx_fp32, x_pos_fp32, x_low_fp32,
                      ONE, ONE, ONE, ONE, EIGHT, EIGHT, EIGHT)
    tik_instance.vsub(FP_32_ONE_TIME, ly_fp32, y_pos_fp32, y_low_fp32,
                      ONE, ONE, ONE, ONE, EIGHT, EIGHT, EIGHT)
    tik_instance.vsub(FP_32_ONE_TIME, hx_fp32, x_high_fp32, x_pos_fp32,
                      ONE, ONE, ONE, ONE, EIGHT, EIGHT, EIGHT)
    tik_instance.vsub(FP_32_ONE_TIME, hy_fp32, y_high_fp32, y_pos_fp32,
                      ONE, ONE, ONE, ONE, EIGHT, EIGHT, EIGHT)
    pos_cmp = tik_instance.Tensor("float32", [FOUR, FP_32_ONE_TIME],
                                  name="pos_cmp",
                                  scope=tbe_platform.scope_ubuf)
    lx_pos_cmp = pos_cmp[0, 0]
    ly_pos_cmp = pos_cmp[ONE, 0]
    hx_pos_cmp = pos_cmp[TWO, 0]
    hy_pos_cmp = pos_cmp[THREE, 0]
    neg_pos_cmp = tik_instance.Tensor("float32", [FOUR, FP_32_ONE_TIME],
                                      name="neg_pos_cmp",
                                      scope=tbe_platform.scope_ubuf)
    lx_neg_cmp = neg_pos_cmp[0, 0]
    ly_neg_cmp = neg_pos_cmp[ONE, 0]
    hx_neg_cmp = neg_pos_cmp[TWO, 0]
    hy_neg_cmp = neg_pos_cmp[THREE, 0]
    end_pos_cmp = tik_instance.Tensor("float32", [FOUR, FP_32_ONE_TIME],
                                      name="end_pos_cmp",
                                      scope=tbe_platform.scope_ubuf)
    x_low_pos_cmp = end_pos_cmp[0, 0]
    y_low_pos_cmp = end_pos_cmp[ONE, 0]
    x_high_pos_cmp = end_pos_cmp[TWO, 0]
    h_high_pos_cmp = end_pos_cmp[THREE, 0]
    neg_end_pos_cmp = tik_instance.Tensor("float32", [FOUR, FP_32_ONE_TIME],
                                          name="neg_end_pos_cmp",
                                          scope=tbe_platform.scope_ubuf)
    x_low_neg_cmp = neg_end_pos_cmp[0, 0]
    y_low_neg_cmp = neg_end_pos_cmp[ONE, 0]
    x_high_neg_cmp = neg_end_pos_cmp[TWO, 0]
    y_high_neg_cmp = neg_end_pos_cmp[THREE, 0]
    # temporary data for comparision
    tik_instance.vadds(FP_32_ONE_TIME, lx_pos_cmp, lx_fp32, one, ONE,
                       ONE, ONE, EIGHT, EIGHT)
    tik_instance.vadds(FP_32_ONE_TIME, ly_pos_cmp, ly_fp32, one, ONE,
                       ONE, ONE, EIGHT, EIGHT)
    tik_instance.vadds(FP_32_ONE_TIME, hx_pos_cmp, hx_fp32, one, ONE,
                       ONE, ONE, EIGHT, EIGHT)
    tik_instance.vadds(FP_32_ONE_TIME, hy_pos_cmp, hy_fp32, one, ONE,
                       ONE, ONE, EIGHT, EIGHT)
    tik_instance.vadds(FP_32_ONE_TIME, lx_neg_cmp, lx_fp32, neg_one,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    tik_instance.vadds(FP_32_ONE_TIME, ly_neg_cmp, ly_fp32, neg_one,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    tik_instance.vadds(FP_32_ONE_TIME, hx_neg_cmp, hx_fp32, neg_one,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    tik_instance.vadds(FP_32_ONE_TIME, hy_neg_cmp, hy_fp32, neg_one,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    tik_instance.vadds(FP_32_ONE_TIME, x_low_pos_cmp, x_low_fp32, one,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    tik_instance.vadds(FP_32_ONE_TIME, y_low_pos_cmp, y_low_fp32, one,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    tik_instance.vadds(FP_32_ONE_TIME, x_high_pos_cmp, x_high_fp32, one,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    tik_instance.vadds(FP_32_ONE_TIME, h_high_pos_cmp, y_high_fp32, one,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    tik_instance.vadds(FP_32_ONE_TIME, x_low_neg_cmp, x_low_fp32, neg_one,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    tik_instance.vadds(FP_32_ONE_TIME, y_low_neg_cmp, y_low_fp32, neg_one,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    tik_instance.vadds(FP_32_ONE_TIME, x_high_neg_cmp, x_high_fp32, neg_one,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    tik_instance.vadds(FP_32_ONE_TIME, y_high_neg_cmp, y_high_fp32, neg_one,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    # compare lx ly with 0 and 1

    cmp_const_fp32 = tik_instance.Tensor("float32", [FP_32_ONE_TIME, ],
                                         name="cmp_const",
                                         scope=tbe_platform.scope_ubuf)
    cmp_const = cmp_const_fp32[0, ]
    tik_instance.vector_dup(FP_32_ONE_TIME, cmp_const, one, ONE, ONE, EIGHT)
    cmpmask = tik_instance.vcmp_gt(FP_32_ONE_TIME, lx_fp32,
                                   cmp_const[0, ], ONE, ONE)
    tik_instance.vsel(FP_32_ONE_TIME, 0, lx_fp32, cmpmask, lx_neg_cmp,
                      lx_fp32, ONE, ONE, ONE, ONE, EIGHT, EIGHT, EIGHT)
    tik_instance.vsel(FP_32_ONE_TIME, 0, hx_fp32, cmpmask, hx_pos_cmp,
                      hx_fp32, ONE, ONE, ONE, ONE, EIGHT, EIGHT, EIGHT)
    tik_instance.vsel(FP_32_ONE_TIME, 0, x_low_fp32, cmpmask, x_low_pos_cmp,
                      x_low_fp32, ONE, ONE, ONE, ONE, EIGHT, EIGHT, EIGHT)
    tik_instance.vsel(FP_32_ONE_TIME, 0, x_high_fp32, cmpmask, x_high_pos_cmp,
                      x_high_fp32,
                      ONE, ONE, ONE, ONE, EIGHT, EIGHT, EIGHT)

    tik_instance.vector_dup(FP_32_ONE_TIME, cmp_const, zero, ONE, ONE, EIGHT)
    cmpmask = tik_instance.vcmp_lt(FP_32_ONE_TIME, lx_fp32,
                                   cmp_const[0, ], ONE, ONE)
    tik_instance.vsel(FP_32_ONE_TIME, 0, lx_fp32, cmpmask, lx_pos_cmp,
                      lx_fp32, ONE, ONE, ONE, ONE, EIGHT, EIGHT, EIGHT)
    tik_instance.vsel(FP_32_ONE_TIME, 0, hx_fp32, cmpmask, hx_neg_cmp,
                      hx_fp32, ONE, ONE, ONE, ONE, EIGHT, EIGHT, EIGHT)
    tik_instance.vsel(FP_32_ONE_TIME, 0, x_low_fp32, cmpmask, x_low_neg_cmp,
                      x_low_fp32, ONE, ONE, ONE, ONE, EIGHT, EIGHT, EIGHT)
    tik_instance.vsel(FP_32_ONE_TIME, 0, x_high_fp32, cmpmask, x_high_neg_cmp,
                      x_high_fp32,
                      ONE, ONE, ONE, ONE, EIGHT, EIGHT, EIGHT)

    tik_instance.vector_dup(FP_32_ONE_TIME, cmp_const, one, ONE, ONE, EIGHT)
    cmpmask = tik_instance.vcmp_gt(FP_32_ONE_TIME, ly_fp32,
                                   cmp_const, ONE, ONE)
    tik_instance.vsel(FP_32_ONE_TIME, 0, ly_fp32, cmpmask, ly_neg_cmp,
                      ly_fp32, ONE, ONE, ONE, ONE, EIGHT, EIGHT, EIGHT)
    tik_instance.vsel(FP_32_ONE_TIME, 0, hy_fp32, cmpmask, hy_pos_cmp,
                      hy_fp32, ONE, ONE, ONE, ONE, EIGHT, EIGHT, EIGHT)
    tik_instance.vsel(FP_32_ONE_TIME, 0, y_low_fp32, cmpmask, y_low_pos_cmp,
                      y_low_fp32, ONE, ONE, ONE, ONE, EIGHT, EIGHT, EIGHT)
    tik_instance.vsel(FP_32_ONE_TIME, 0, y_high_fp32, cmpmask, h_high_pos_cmp,
                      y_high_fp32,
                      ONE, ONE, ONE, ONE, EIGHT, EIGHT, EIGHT)

    tik_instance.vector_dup(FP_32_ONE_TIME, cmp_const, zero, ONE, ONE, EIGHT)
    cmpmask = tik_instance.vcmp_lt(FP_32_ONE_TIME, ly_fp32,
                                   cmp_const, ONE, ONE)
    tik_instance.vsel(FP_32_ONE_TIME, 0, ly_fp32, cmpmask, ly_pos_cmp,
                      ly_fp32, ONE, ONE, ONE, ONE, EIGHT, EIGHT, EIGHT)
    tik_instance.vsel(FP_32_ONE_TIME, 0, hy_fp32, cmpmask, hy_neg_cmp,
                      hy_fp32, ONE, ONE, ONE, ONE, EIGHT, EIGHT, EIGHT)
    tik_instance.vsel(FP_32_ONE_TIME, 0, y_low_fp32, cmpmask, y_low_neg_cmp,
                      y_low_fp32, ONE, ONE, ONE, ONE, EIGHT, EIGHT, EIGHT)
    tik_instance.vsel(FP_32_ONE_TIME, 0, y_high_fp32, cmpmask, y_high_neg_cmp,
                      y_high_fp32,
                      ONE, ONE, ONE, ONE, EIGHT, EIGHT, EIGHT)
    # update x_low_int, y_low_int, x_high_int, y_high_int
    cmp_buf_fp32 = tik_instance.Tensor("float32", [FP_32_ONE_TIME, ],
                                       name="cmp_buf",
                                       scope=tbe_platform.scope_ubuf)
    cmp_buf = cmp_buf_fp32[0, ]
    tik_instance.vadds(FP_32_ONE_TIME, cmp_buf, x_low_fp32, point_five,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    tik_instance.vconv(FP_32_ONE_TIME, "floor", x_low_int, cmp_buf,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    tik_instance.vconv(FP_32_ONE_TIME, "ceil", x_high_int, cmp_buf,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    tik_instance.vadds(FP_32_ONE_TIME, cmp_buf, y_low_fp32, point_five,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    tik_instance.vconv(FP_32_ONE_TIME, "floor", y_low_int, cmp_buf,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    tik_instance.vconv(FP_32_ONE_TIME, "ceil", y_high_int, cmp_buf,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    # below are the conditions for TensorFlow:

    cmp_xy_fp32 = tik_instance.Tensor("float32", [FP_32_ONE_TIME, ],
                                      name="cmp_xy",
                                      scope=tbe_platform.scope_ubuf)
    cmp_xy = cmp_xy_fp32[0, ]
    tik_instance.vector_dup(FP_32_ONE_TIME, cmp_buf, width, ONE, ONE, EIGHT)
    tik_instance.vadds(FP_32_ONE_TIME, cmp_buf, cmp_buf, neg_one,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    tik_instance.vadds(FP_32_ONE_TIME, cmp_xy, x_low_fp32, point_five,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    cmpmask = tik_instance.vcmp_gt(FP_32_ONE_TIME, cmp_xy, cmp_buf, ONE, ONE)
    tik_instance.vector_dup(FP_32_ONE_TIME, cmp_const, zero, ONE, ONE, EIGHT)
    tik_instance.vsel(FP_32_ONE_TIME, 0, hx_fp32, cmpmask, cmp_const,
                      hx_fp32, ONE, ONE, ONE, ONE, EIGHT, EIGHT, EIGHT)
    tik_instance.vsel(FP_32_ONE_TIME, 0, lx_fp32, cmpmask, cmp_const,
                      lx_fp32, ONE, ONE, ONE, ONE, EIGHT, EIGHT, EIGHT)

    tik_instance.vector_dup(FP_32_ONE_TIME, cmp_buf, height, ONE, ONE, EIGHT)
    tik_instance.vadds(FP_32_ONE_TIME, cmp_buf, cmp_buf, neg_one,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    tik_instance.vadds(FP_32_ONE_TIME, cmp_xy, y_low_fp32, point_five,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    cmpmask = tik_instance.vcmp_gt(FP_32_ONE_TIME, cmp_xy, cmp_buf, ONE, ONE)
    tik_instance.vector_dup(FP_32_ONE_TIME, cmp_const, zero, ONE, ONE, EIGHT)
    tik_instance.vsel(FP_32_ONE_TIME, 0, hy_fp32, cmpmask, cmp_const,
                      hy_fp32, ONE, ONE, ONE, ONE, EIGHT, EIGHT, EIGHT)
    tik_instance.vsel(FP_32_ONE_TIME, 0, ly_fp32, cmpmask, cmp_const,
                      ly_fp32, ONE, ONE, ONE, ONE, EIGHT, EIGHT, EIGHT)

    tik_instance.vector_dup(FP_32_ONE_TIME, cmp_buf, zero, ONE, ONE, EIGHT)
    tik_instance.vector_dup(FP_32_ONE_TIME, cmp_const, zero, ONE, ONE, EIGHT)
    tik_instance.vadds(FP_32_ONE_TIME, cmp_xy, x_low_fp32, point_five,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    cmpmask = tik_instance.vcmp_lt(FP_32_ONE_TIME, cmp_xy, cmp_buf, ONE, ONE)
    tik_instance.vsel(FP_32_ONE_TIME, 0, hx_fp32, cmpmask, cmp_const,
                      hx_fp32, ONE, ONE, ONE, ONE, EIGHT, EIGHT, EIGHT)
    tik_instance.vsel(FP_32_ONE_TIME, 0, lx_fp32, cmpmask, cmp_const,
                      lx_fp32, ONE, ONE, ONE, ONE, EIGHT, EIGHT, EIGHT)

    tik_instance.vector_dup(FP_32_ONE_TIME, cmp_buf, zero, ONE, ONE, EIGHT)
    tik_instance.vector_dup(FP_32_ONE_TIME, cmp_const, zero, ONE, ONE, EIGHT)
    tik_instance.vadds(FP_32_ONE_TIME, cmp_xy, y_low_fp32, point_five,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    cmpmask = tik_instance.vcmp_lt(FP_32_ONE_TIME, cmp_xy, cmp_buf, ONE, ONE)
    tik_instance.vsel(FP_32_ONE_TIME, 0, hy_fp32, cmpmask, cmp_const,
                      hy_fp32, ONE, ONE, ONE, ONE, EIGHT, EIGHT, EIGHT)
    tik_instance.vsel(FP_32_ONE_TIME, 0, ly_fp32, cmpmask, cmp_const,
                      ly_fp32, ONE, ONE, ONE, ONE, EIGHT, EIGHT, EIGHT)
    return point_positions_int32, point_weights_fp32


def _get_delta_addresses(tik_instance, point_positions_int32, width):
    """
    :param tik_instance:
    :param point_positions_int32:  the position of 4 pixels around;\
                                    delta_x_low deltaY_Low
                                    delta_x_high deltaY_High
    :param width: the wide of the roibox
    :return: a tmp variable  point_distance_int32

    get feature map delta address
    Get delta address for xlow/xhigh/ylow/yhigh of every grid
    (for preparing VBI addresses).
    deltaX = (w-wstart)
    deltaAddr_X = (w-wstart)*C0 (input format: CONEHWC0)
    deltaAddr_Y = (h-hstart)*RoiWidth*C0 (input format: C1HWC0)
    """
    index_height = 0
    x_start_value = tik_instance.Scalar(dtype="int32",
                                        init_value=point_positions_int32[0, 0])
    neg_xstart = tik_instance.Scalar(dtype="int32",
                                     init_value=NEG_ONE * x_start_value)
    y_start_value = tik_instance.Scalar(
        dtype="int32",
        init_value=point_positions_int32[ONE, index_height * SAMPLING_RATIO])
    neg_ystart = tik_instance.Scalar(dtype="int32",
                                     init_value=NEG_ONE * y_start_value)
    c_0 = C0SIZE * BYTES
    h_offset = tik_instance.Scalar(dtype="int32",
                                   init_value=c_0 * width)
    point_distance_int32 = tik_instance.Tensor(
        "int32", [FOUR, FP_32_ONE_TIME], name="point_distance_int32",
        scope=tbe_platform.scope_ubuf)

    tik_instance.vadds(FP_32_ONE_TIME, point_distance_int32[0, 0],
                       point_positions_int32[0, 0],
                       neg_xstart, ONE, ONE, ONE, EIGHT, EIGHT)
    tik_instance.vadds(FP_32_ONE_TIME, point_distance_int32[ONE, 0],
                       point_positions_int32[ONE,
                                             index_height * SAMPLING_RATIO],
                       neg_ystart,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    tik_instance.vadds(FP_32_ONE_TIME, point_distance_int32[TWO, 0],
                       point_positions_int32[TWO, 0],
                       neg_xstart, ONE, ONE, ONE, EIGHT, EIGHT)
    tik_instance.vadds(FP_32_ONE_TIME, point_distance_int32[THREE, 0],
                       point_positions_int32[THREE,
                                             index_height * SAMPLING_RATIO],
                       neg_ystart,
                       ONE, ONE, ONE, EIGHT, EIGHT)
    tik_instance.vmuls(FP_32_ONE_TIME, point_distance_int32[0, 0],
                       point_distance_int32[0, 0],
                       c_0, TWO, ONE, ONE, SIXTEEN, SIXTEEN)
    tik_instance.vmuls(FP_32_ONE_TIME, point_distance_int32[ONE, 0],
                       point_distance_int32[ONE, 0],
                       h_offset, TWO, ONE, ONE, SIXTEEN, SIXTEEN)
    return point_distance_int32


def _get_vbi_addr_1x1grid(tik_instance, point_distance_int32, flag):
    """
    :param tik_instance:
    :param point_distance_int32: the tmp variable to rearranged
            address of 4 pixels
    :return:Rearranged address(Xn)
    """
    point_ph_addr = tik_instance.Tensor(
        "int32", [FOUR, NUM_ELEMENTS_ONEROW],
        name="point_ph_addr", scope=tbe_platform.scope_ubuf)
    point_ph_addr_res = tik_instance.Tensor(
        "int32", [FOUR, NUM_ELEMENTS_ONEROW],
        name="point_ph_addr_res",
        scope=tbe_platform.scope_ubuf)
    point_ph_addr_float = tik_instance.Tensor(
        "float32", [FOUR, NUM_ELEMENTS_ONEROW],
        name="point_ph_addr_float",
        scope=tbe_platform.scope_ubuf)
    point_ph_addr_float_res = tik_instance.Tensor(
        "float32", [FOUR, NUM_ELEMENTS_ONEROW],
        name="point_ph_addr_float_res",
        scope=tbe_platform.scope_ubuf)
    if flag == 'one_row':
        tik_instance.vector_dup(VBI_NUM_ELEMENTS_ONEROW,
                                point_ph_addr[0, 0], 0, EIGHT, ONE, SEVEN)
    else:
        tik_instance.vector_dup(VBI_NUM_ELEMENTS_ONEROW, point_ph_addr[0, 0],
                                NEG_ONE, EIGHT, ONE, SEVEN)
    tik_instance.vector_dup(VBI_NUM_ELEMENTS_ONEROW,
                            point_ph_addr_res[0, 0], 0, EIGHT, ONE, SEVEN)
    tik_instance.vector_dup(VBI_NUM_ELEMENTS_ONEROW,
                            point_ph_addr_float[0, 0], 0, EIGHT, ONE, SEVEN)
    tik_instance.vector_dup(VBI_NUM_ELEMENTS_ONEROW,
                            point_ph_addr_float_res[0, 0], 0,
                            EIGHT, ONE, SEVEN)
    delta_x_low = point_distance_int32[0, 0]
    delta_x_high = point_distance_int32[TWO, 0]
    num_sampling_block = NUM_SAMPLING_W + ONE
    for pool_h in range(0, POOL_H):
        delta_y_ph0_g0 = tik_instance.Scalar(
            dtype="int32",
            init_value=point_distance_int32[ONE, pool_h * SAMPLING_RATIO])
        delta_y_ph0_g1 = tik_instance.Scalar(
            dtype="int32",
            init_value=point_distance_int32[THREE, pool_h * SAMPLING_RATIO])
        fm_start_addr = 0
        h_start_addr = tik_instance.Scalar(
            dtype="int32", init_value=fm_start_addr + delta_y_ph0_g0)

        tik_instance.vadds(NUM_SAMPLING_W, point_ph_addr[
            0, pool_h * num_sampling_block],
                           delta_x_low, h_start_addr, ONE,
                           ONE, ONE, EIGHT, EIGHT)
        tik_instance.vadds(NUM_SAMPLING_W, point_ph_addr[
            ONE, pool_h * num_sampling_block],
                           delta_x_high, h_start_addr, ONE,
                           ONE, ONE, EIGHT, EIGHT)

        h_start_addr.set_as(fm_start_addr + delta_y_ph0_g1)

        tik_instance.vadds(NUM_SAMPLING_W, point_ph_addr[
            TWO, pool_h * num_sampling_block],
                           delta_x_low, h_start_addr, ONE,
                           ONE, ONE, EIGHT, EIGHT)
        tik_instance.vadds(NUM_SAMPLING_W, point_ph_addr[
            THREE, pool_h * num_sampling_block],
                           delta_x_high, h_start_addr, ONE,
                           ONE, ONE, EIGHT, EIGHT)

    tik_instance.vconv(VBI_NUM_ELEMENTS_ONEROW, '', point_ph_addr_float,
                       point_ph_addr, EIGHT,
                       ONE, ONE, SEVEN, SEVEN)

    mask_reduce = tik_instance.Tensor("uint32", [EIGHT, ],
                                      name="mask_reduce",
                                      scope=tbe_platform.scope_ubuf)

    cmp_vct_1 = tik_instance.Tensor("float32", [ONE, VBI_NUM_ELEMENTS_ONEROW],
                                    name="cmp_vct_1",
                                    scope=tbe_platform.scope_ubuf)

    tik_instance.vector_dup(VBI_NUM_ELEMENTS_ONEROW, cmp_vct_1,
                            NEG_ONE, ONE, ONE, SEVEN)

    cmpmask = tik_instance.vcmp_gt(VBI_NUM_ELEMENTS_ONEROW,
                                   point_ph_addr_float, cmp_vct_1,
                                   ONE, ONE)

    tik_instance.mov_cmpmask_to_tensor(
        mask_reduce.reinterpret_cast_to("uint64"),
        cmpmask)
    with tik_instance.for_range(0, FOUR) as i:
        tik_instance.vreduce(VBI_NUM_ELEMENTS_ONEROW,
                             point_ph_addr_float_res[i, 0],
                             point_ph_addr_float[i, 0],
                             mask_reduce, ONE, ONE, SEVEN, 0, 0, None,
                             "normal")
        tik_instance.vconv(VBI_NUM_ELEMENTS_ONEROW, 'to-zero',
                           point_ph_addr_res[i, 0],
                           point_ph_addr_float_res[i, 0], ONE, ONE, ONE, SEVEN,
                           SEVEN)

    vbi_addr = tik_instance.Tensor("int32",
                                   [VBI_NUM_ELEMENTS_ONEROW * FOUR, ],
                                   name="vbi_addr",
                                   scope=tbe_platform.scope_ubuf)
    tik_instance.vector_dup(VBI_NUM_ELEMENTS_ONEROW, vbi_addr[0, ],
                            0, FOUR, ONE, SEVEN)
    horizontal_repeat = NUM_ELMENTS_ONEBIN
    num_burst = VBI_NUM_BLOCKS_ONEROW

    dst_stride = horizontal_repeat - ONE
    for i in range(0, FOUR):
        tik_instance.data_move(vbi_addr[i * EIGHT],
                               point_ph_addr_res[i, 0], 0,
                               num_burst, ONE, 0, dst_stride)
    return vbi_addr


def _get_vbi_weights_1x1grid(tik_instance, point_weights_fp32, flag='None'):
    """
    :param tik_instance:
    :param point_weights_fp32: the distance to 4 pixels around lx ly hx hy
    :return:vbi_weights ,the rearranged weights(xm), hx*hy  lx*hy  hx*ly lx*ly
    """
    vbi_tmp_weights_res = \
        tik_instance.Tensor("float32", [FOUR, NUM_ELEMENTS_ONEROW],
                            name="vbi_tmp_weights_res",
                            scope=tbe_platform.scope_ubuf)

    vbi_tmp_weights = tik_instance.Tensor("float32",
                                          [FOUR, NUM_ELEMENTS_ONEROW],
                                          name="vbi_tmp_weights",
                                          scope=tbe_platform.scope_ubuf)

    tik_instance.vector_dup(VBI_NUM_ELEMENTS_ONEROW,
                            vbi_tmp_weights_res, 0, EIGHT, ONE, SEVEN)
    if flag == 'one_row':
        tik_instance.vector_dup(VBI_NUM_ELEMENTS_ONEROW,
                                vbi_tmp_weights, 0, EIGHT, ONE, SEVEN)
    else:
        tik_instance.vector_dup(VBI_NUM_ELEMENTS_ONEROW, vbi_tmp_weights,
                                NEG_ONE, EIGHT, ONE, SEVEN)

    lx_fp32 = point_weights_fp32[0, 0]
    hx_fp32 = point_weights_fp32[TWO, 0]
    for pool_h in range(0, POOL_H):
        ly0 = tik_instance.Scalar(dtype="float32",
                                  init_value=point_weights_fp32[
                                      ONE, pool_h * SAMPLING_RATIO])
        hy0 = tik_instance.Scalar(dtype="float32",
                                  init_value=point_weights_fp32[
                                      THREE, pool_h * SAMPLING_RATIO])

        num_sampling_w_block = NUM_SAMPLING_W + ONE

        tik_instance.vmuls(NUM_SAMPLING_W, vbi_tmp_weights[
            0, num_sampling_w_block * pool_h],
                           hx_fp32, hy0, ONE, ONE, ONE, EIGHT, EIGHT)
        tik_instance.vmuls(NUM_SAMPLING_W, vbi_tmp_weights[
            ONE, num_sampling_w_block * pool_h],
                           lx_fp32, hy0, ONE, ONE, ONE, EIGHT, EIGHT)
        tik_instance.vmuls(NUM_SAMPLING_W, vbi_tmp_weights[
            TWO, num_sampling_w_block * pool_h],
                           hx_fp32, ly0, ONE, ONE, ONE, EIGHT, EIGHT)
        tik_instance.vmuls(NUM_SAMPLING_W, vbi_tmp_weights[
            THREE, num_sampling_w_block * pool_h],
                           lx_fp32, ly0, ONE, ONE, ONE, EIGHT, EIGHT)

    mask_reduce_1 = tik_instance.Tensor("uint32", [EIGHT, ],
                                        name="mask_reduce1",
                                        scope=tbe_platform.scope_ubuf)

    cmp_vct = tik_instance.Tensor("float32", [ONE, VBI_NUM_ELEMENTS_ONEROW],
                                  name="cmp_vct",
                                  scope=tbe_platform.scope_ubuf)
    tik_instance.vector_dup(VBI_NUM_ELEMENTS_ONEROW,
                            cmp_vct, 0, ONE, ONE, SEVEN)

    cmpmask = tik_instance.vcmp_ge(VBI_NUM_ELEMENTS_ONEROW,
                                   vbi_tmp_weights, cmp_vct, ONE, ONE)

    tik_instance.mov_cmpmask_to_tensor(
        mask_reduce_1.reinterpret_cast_to("uint64"), cmpmask)
    with tik_instance.for_range(0, FOUR) as i:
        tik_instance.vreduce(VBI_NUM_ELEMENTS_ONEROW,
                             vbi_tmp_weights_res[i, 0],
                             vbi_tmp_weights[i, 0], mask_reduce_1, ONE,
                             ONE, SEVEN, 0, 0, None, "normal")

    vbi_weights = tik_instance.Tensor("float16",
                                      [VBI_NUM_ELEMENTS_ONEROW * FOUR, ],
                                      name="vbi_weights",
                                      scope=tbe_platform.scope_ubuf)

    tik_instance.vector_dup(VBI_NUM_ELEMENTS_ONEROW * TWO, vbi_weights[0, ],
                            0, TWO, ONE, SEVEN)
    dst_ub = tik_instance.Tensor("float32", [VBI_NUM_ELEMENTS_ONEROW * FOUR, ],
                                name="dst_ub", scope=tbe_platform.scope_ubuf)
    tik_instance.data_move(dst_ub, vbi_tmp_weights_res[0, 0], 0,
                            SEVEN, ONE, 0, THREE)
    tik_instance.data_move(dst_ub[EIGHT], vbi_tmp_weights_res[ONE, 0],
                            0, SEVEN, ONE, 0, THREE)
    tik_instance.data_move(dst_ub[TWO * EIGHT], vbi_tmp_weights_res[TWO, 0],
                            0, SEVEN, ONE, 0, THREE)
    tik_instance.data_move(dst_ub[THREE * EIGHT], vbi_tmp_weights_res[THREE, 0],
                            0, SEVEN, ONE, 0, THREE)
    tik_instance.vec_conv(TWO * SIXTEEN, "", vbi_weights, dst_ub, SEVEN, TWO, FOUR)
    return vbi_weights


def _prepare_vbi_xn(tik_instance, feature_shape,
                    point_positions_int32, flag='None'):
    """
    :param tik_instance:
    :param feature_shape:the shape of the featuremap
    :param point_positions_int32: the position of 4 pixels around the
    center of gird
    :return:Rearranged address(Xn)
    """
    # point_positions_int32 : x_low_int y_low_int x_high_int y_high_int
    fm_width = feature_shape[THREE]
    wstart = tik_instance.Scalar(dtype="int32",
                                 init_value=point_positions_int32[
                                     0, 0])
    wend = tik_instance.Scalar(dtype="int32",
                               init_value=point_positions_int32[
                                   TWO, STRIDE_W])
    with tik_instance.if_scope(wend >= fm_width):
        wend.set_as(fm_width)
    with tik_instance.else_scope():
        wend.set_as(wend + ONE)
    with tik_instance.if_scope(wend <= 0):
        wend.set_as(ONE)
    width = tik_instance.Scalar(dtype="int32",
                                init_value=wend - wstart)
    point_distance_int32 = _get_delta_addresses(tik_instance,
                                                point_positions_int32,
                                                width)
    vbi_addr = _get_vbi_addr_1x1grid(tik_instance,
                                     point_distance_int32, flag)
    return vbi_addr


def _prepare_vbi_xm(tik_instance, point_weights_fp32, flag='None'):
    """
    :param tik_instance:
    :param point_weights_fp32: the distance to 4 pixels around
    :return: vbi_weights ,the rearranged weights(xm),
            hx*hy  lx*hy  hx*ly lx*ly
    """
    vbi_weights = _get_vbi_weights_1x1grid(tik_instance,
                                           point_weights_fp32, flag)
    return vbi_weights


def _do_vbi_full_featuremap_mode(tik_instance, cur_roi_num,
                                 feature_shape, featuremap_gm,
                                 output_gm,
                                 point_positions_int32,
                                 point_weights_fp32,
                                 wstart, hstart, width, height):
    """
    :param tik_instance:
    :param block_id: the block used
    :param cur_roi_num: the num of current roi box
    :param feature_shape: shape of featuremap
    :param featuremap_gm: the gm of featuremap
    :param output_gm: the gm for output
    :param point_positions_int32: positions of 4 pixels around the gird center
                                x_low_int  y_low_int  x_high_int  y_high_int
    :param point_weights_fp32: distance to 4 pixels
    :param cut_type_flag: the type of processing mode
    :return:None
    """
    feature_map_c1 = feature_shape[ONE]
    feature_map_w = feature_shape[THREE]
    featuremap_ub = tik_instance.Tensor("float16", [
        FM_BUFFER_SIZE_LIMIT // BYTES, ],
                                        name="featuremap_ub",
                                        scope=tbe_platform.scope_ubuf)
    vbi_weights = _prepare_vbi_xm(tik_instance, point_weights_fp32)
    vbi_addr = _prepare_vbi_xn(tik_instance, feature_shape,
                               point_positions_int32)
    result_ub = tik_instance.Tensor("float16",
                                    [ONE, ONE, POOL_H + ONE, POOL_W,
                                     C0SIZE],
                                    name="result_ub",
                                    scope=tbe_platform.scope_ubuf)
    for i in range(0, feature_map_c1):
        repeat = (POOL_H * POOL_W + SEVEN) // EIGHT
        tik_instance.data_move(featuremap_ub[0, ],
                               featuremap_gm[
                                   0, i, hstart, wstart, 0],
                               0, height, width,
                               feature_map_w - width, 0)
        tik_instance.vbi(FP_SIXTEEN_ONE_TIME, result_ub[0:],
                         featuremap_ub[0:], vbi_weights[0:],
                         vbi_addr[0:], ONE, repeat, NUM_ELMENTS_ONEBIN,
                         ONE, EIGHT * NUM_ADDR_ONE_VBI // BYTES)
        tik_instance.data_move(output_gm[cur_roi_num, i, 0, 0, 0],
                               result_ub[0, 0, 0, 0, 0],
                               0, ONE, POOL_H * POOL_W, 0, 0)


def _process_one_roi_vbi(tik_instance,
                         cur_roi_num, feature_shape,
                         featuremap_gm, output_gm,
                         point_positions_int32,
                         point_weights_fp32):
    """
    :param tik_instance:
    :param block_id:   the block used
    :param cur_roi_num: the num of current roi box
    :param feature_shape: shape of featuremap
    :param featuremap_gm: the gm  of featuremap
    :param output_gm: the gm  for output
    :param point_positions_int32: positions of 4 pixels
            around the center of gird /
            x_low_int  y_low_int  x_high_int  y_high_int
    :param point_weights_fp32: distance to 4 pixels
    :return: None
    """
    feature_map_h = feature_shape[TWO]
    feature_map_w = feature_shape[THREE]

    tik_instance.vrelu(FP_32_ONE_TIME, point_positions_int32[0, 0],
                       point_positions_int32[0, 0], ONE, ONE,
                       ONE, EIGHT, EIGHT)
    tik_instance.vrelu(FP_32_ONE_TIME, point_positions_int32[ONE, 0],
                       point_positions_int32[ONE, 0], ONE, ONE,
                       ONE, EIGHT, EIGHT)

    hstart = tik_instance.Scalar(dtype="int32",
                                 init_value=point_positions_int32[
                                     ONE, 0])
    hend = tik_instance.Scalar(dtype="int32",
                               init_value=point_positions_int32[
                                   THREE, STRIDE_H])
    wstart = tik_instance.Scalar(dtype="int32",
                                 init_value=point_positions_int32[
                                     0, 0])
    wend = tik_instance.Scalar(dtype="int32",
                               init_value=point_positions_int32[
                                   TWO, STRIDE_W])
    with tik_instance.if_scope(wend >= feature_map_w):
        wend.set_as(feature_map_w)
    with tik_instance.else_scope():
        wend.set_as(wend + ONE)
    with tik_instance.if_scope(hend >= feature_map_h):
        hend.set_as(feature_map_h)
    with tik_instance.else_scope():
        hend.set_as(hend + ONE)
    with tik_instance.if_scope(wend <= 0):
        wend.set_as(ONE)
    with tik_instance.if_scope(hend <= 0):
        hend.set_as(ONE)
    width = tik_instance.Scalar(dtype="int32",
                                init_value=wend - wstart)
    height = tik_instance.Scalar(dtype="int32",
                                 init_value=hend - hstart)
    with tik_instance.if_scope(width >= ONE):
        with tik_instance.if_scope(height >= ONE):
            _do_vbi_full_featuremap_mode(tik_instance,
                                         cur_roi_num,
                                         feature_shape,
                                         featuremap_gm,
                                         output_gm,
                                         point_positions_int32,
                                         point_weights_fp32,
                                         wstart, hstart,
                                         width, height)

        with tik_instance.else_scope():
            result_ub = tik_instance.Tensor("float16",
                                            [ONE, feature_shape[ONE],
                                             POOL_H + ONE, POOL_W, C0SIZE],
                                            name="result_ub",
                                            scope=tbe_platform.scope_ubuf)
            tik_instance.vector_dup(FP_SIXTEEN_ONE_TIME, result_ub, 0,
                                    POOL_W * feature_shape[ONE], ONE, EIGHT)
            tik_instance.data_move(output_gm[cur_roi_num, 0, 0, 0, 0],
                                   result_ub[0, 0, 0, 0, 0],
                                   0, feature_shape[ONE],
                                   POOL_H * POOL_W, 0, 0)
    with tik_instance.else_scope():
        result_ub = tik_instance.Tensor("float16",
                                        [ONE, feature_shape[ONE],
                                         POOL_H + ONE, POOL_W, C0SIZE],
                                        name="result_ub",
                                        scope=tbe_platform.scope_ubuf)
        tik_instance.vector_dup(FP_SIXTEEN_ONE_TIME, result_ub, 0,
                                POOL_W * feature_shape[ONE], ONE, EIGHT)
        tik_instance.data_move(output_gm[cur_roi_num, 0, 0, 0, 0],
                               result_ub[0, 0, 0, 0, 0],
                               0, feature_shape[ONE], POOL_H * POOL_W, 0, 0)


def _process_one_roi_vbi_c1_cut(tik_instance, block_id,
                                cur_roi_num, feature_shape,
                                featuremap_gm, output_gm,
                                point_positions_int32,
                                point_weights_fp32):
    """
    :param tik_instance:
    :param block_id:   the block used
    :param cur_roi_num: the num of current roi box
    :param feature_shape: shape of featuremap
    :param featuremap_gm: the gm  of featuremap
    :param output_gm: the gm  for output
    :param point_positions_int32: positions of 4 pixels
            around the center of gird /
            x_low_int  y_low_int  x_high_int  y_high_int
    :param point_weights_fp32: distance to 4 pixels
    :return: None
    """
    feature_map_h = feature_shape[TWO]
    feature_map_w = feature_shape[THREE]
    tik_instance.vrelu(FP_32_ONE_TIME, point_positions_int32[0, 0],
                       point_positions_int32[0, 0], ONE, ONE,
                       ONE, EIGHT, EIGHT)
    tik_instance.vrelu(FP_32_ONE_TIME, point_positions_int32[ONE, 0],
                       point_positions_int32[ONE, 0], ONE, ONE,
                       ONE, EIGHT, EIGHT)
    hstart = tik_instance.Scalar(dtype="int32",
                                 init_value=point_positions_int32[
                                     ONE, 0])
    hend = tik_instance.Scalar(dtype="int32",
                               init_value=point_positions_int32[
                                   THREE, STRIDE_H])
    wstart = tik_instance.Scalar(dtype="int32",
                                 init_value=point_positions_int32[
                                     0, 0])
    wend = tik_instance.Scalar(dtype="int32",
                               init_value=point_positions_int32[
                                   TWO, STRIDE_W])
    with tik_instance.if_scope(wend >= feature_map_w):
        wend.set_as(feature_map_w)
    with tik_instance.else_scope():
        wend.set_as(wend + ONE)
    with tik_instance.if_scope(hend >= feature_map_h):
        hend.set_as(feature_map_h)
    with tik_instance.else_scope():
        hend.set_as(hend + ONE)
    with tik_instance.if_scope(wend <= 0):
        wend.set_as(ONE)
    with tik_instance.if_scope(hend <= 0):
        hend.set_as(ONE)
    width = tik_instance.Scalar(dtype="int32",
                                init_value=wend - wstart)
    height = tik_instance.Scalar(dtype="int32",
                                 init_value=hend - hstart)
    with tik_instance.if_scope(width >= ONE):
        with tik_instance.if_scope(height >= ONE):
            roi_feature_map_size = tik_instance.Scalar(
                dtype="int32",
                init_value=width * height * C0SIZE * BYTES)
            with tik_instance.if_scope(
                    FM_BUFFER_SIZE_LIMIT >= roi_feature_map_size):
                _do_vbi_full_feature_mode_c1_cut(
                    tik_instance, block_id, cur_roi_num,
                    feature_shape,
                    featuremap_gm,
                    output_gm,
                    point_positions_int32,
                    point_weights_fp32,
                    wstart, hstart, width, height)
            with tik_instance.else_scope():
                _do_vbi_one_row_mode_c1_cut(
                    tik_instance, block_id, cur_roi_num,
                    feature_shape, featuremap_gm,
                    output_gm, point_positions_int32,
                    point_weights_fp32,
                    wstart, width)
        with tik_instance.else_scope():
            result_ub = tik_instance.Tensor("float16",
                                            [ONE, feature_shape[ONE],
                                             POOL_H + ONE, POOL_W, C0SIZE],
                                            name="result_ub",
                                            scope=tbe_platform.scope_ubuf)
            tik_instance.vector_dup(FP_SIXTEEN_ONE_TIME, result_ub, 0,
                                    POOL_W * feature_shape[ONE], ONE, EIGHT)
            tik_instance.data_move(output_gm[cur_roi_num, 0, 0, 0, 0],
                                   result_ub[0, 0, 0, 0, 0],
                                   0, feature_shape[ONE],
                                   POOL_H * POOL_W, 0, 0)
    with tik_instance.else_scope():
        result_ub = tik_instance.Tensor("float16",
                                        [ONE, feature_shape[ONE],
                                         POOL_H + ONE, POOL_W, C0SIZE],
                                        name="result_ub",
                                        scope=tbe_platform.scope_ubuf)
        tik_instance.vector_dup(FP_SIXTEEN_ONE_TIME, result_ub, 0,
                                POOL_W * feature_shape[ONE], ONE, EIGHT)
        tik_instance.data_move(output_gm[cur_roi_num, 0, 0, 0, 0],
                               result_ub[0, 0, 0, 0, 0],
                               0, feature_shape[ONE], POOL_H * POOL_W, 0, 0)


def _do_vbi_full_feature_mode_c1_cut(tik_instance, block_id, cur_roi_num,
                                     feature_shape, featuremap_gm,
                                     output_gm,
                                     point_positions_int32,
                                     point_weights_fp32,
                                     wstart, hstart, width, height):
    """
    :param tik_instance:
    :param block_id: the block used
    :param cur_roi_num: the num of current roi box
    :param feature_shape: shape of featuremap
    :param featuremap_gm: the gm of featuremap
    :param output_gm: the gm for output
    :param point_positions_int32: positions of 4 pixels around the gird center
                                x_low_int  y_low_int  x_high_int  y_high_int
    :param point_weights_fp32: distance to 4 pixels
    :param cut_type_flag: the type of processing mode
    :return:None
    """
    feature_map_c1 = (feature_shape[ONE] + BLOCK_DIM - ONE) // BLOCK_DIM
    feature_map_w = feature_shape[THREE]
    featuremap_ub = tik_instance.Tensor("float16", [FM_BUFFER_SIZE_LIMIT // BYTES, ],
                                        name="featuremap_ub",
                                        scope=tbe_platform.scope_ubuf)
    vbi_weights = _prepare_vbi_xm(tik_instance, point_weights_fp32)
    vbi_addr = _prepare_vbi_xn(tik_instance, feature_shape,
                               point_positions_int32)
    result_ub = tik_instance.Tensor("float16",
                                    [ONE, ONE, POOL_H + ONE, POOL_W,
                                     C0SIZE], \
                                    name="result_ub",
                                    scope=tbe_platform.scope_ubuf)
    for i in range(0, feature_map_c1):
        repeat = (POOL_H * POOL_W + SEVEN) // EIGHT
        tik_instance.data_move(featuremap_ub[0, ], \
                               featuremap_gm[
                                   0,
                                   feature_map_c1 * block_id + i,
                                   hstart, wstart, 0], \
                               0, height, width, \
                               feature_map_w - width, 0)
        tik_instance.vbi(FP_SIXTEEN_ONE_TIME, result_ub[0:],
                         featuremap_ub[0:], vbi_weights[0:], \
                         vbi_addr[0:], ONE, repeat, NUM_ELMENTS_ONEBIN,
                         ONE, EIGHT * NUM_ADDR_ONE_VBI // BYTES)
        tik_instance.data_move(output_gm[cur_roi_num,
                                         feature_map_c1 * block_id + i,
                                         0, 0, 0],
                               result_ub[0, 0, 0, 0, 0],
                               0, ONE, POOL_H * POOL_W, 0, 0)


def _do_vbi_one_row_mode_c1_cut(tik_instance, block_id, cur_roi_num,
                                feature_shape,
                                featuremap_gm, output_gm,
                                point_positions_int32,
                                point_weights_fp32,
                                wstart, width):
    """
    :param tik_instance:
    :param block_id: the block used
    :param cur_roi_num: the num of current roi box
    :param feature_shape: shape of featuremap
    :param featuremap_gm: the gm of featuremap
    :param output_gm: the gm  of output
    :param point_positions_int32: positions of 4 pixels around the gird center
                                 x_low_int y_low_int x_high_int y_high_int
    :param point_weights_fp32: distance to 4 pixels
    :return: None
    """
    feature_map_c1 = (feature_shape[ONE] + BLOCK_DIM - ONE) // BLOCK_DIM
    feature_map_w = feature_shape[THREE]
    featuremap_ub = tik_instance.Tensor("float16", [FM_BUFFER_SIZE_LIMIT // BYTES, ],
                                        name="featuremap_ub",
                                        scope=tbe_platform.scope_ubuf)
    vbi_weights = _prepare_vbi_xm(tik_instance, point_weights_fp32, 'one_row')
    vbi_addr = _prepare_vbi_xn(tik_instance, feature_shape, \
                               point_positions_int32, 'one_row')
    for index_height in range(0, POOL_H):
        hstart = tik_instance.Scalar(dtype="int32",
                                     init_value=point_positions_int32[
                                         ONE, index_height])
        addr_one_time = tik_instance.Tensor('int32', [NUM_ADDR_ONE_VBI, ],
                                            name='addr_one_time',
                                            scope=tbe_platform.scope_ubuf)
        tik_instance.vadds(NUM_ADDR_ONE_VBI, addr_one_time, vbi_addr, 0,
                           ONE, ONE, ONE, EIGHT, EIGHT)
        result_ub = tik_instance.Tensor("float16",
                                        [ONE, ONE, POOL_H, POOL_W + ONE,
                                         C0SIZE], \
                                        name="result_ub",
                                        scope=tbe_platform.scope_ubuf)
        for i in range(0, feature_map_c1):
            repeat = (POOL_W + SEVEN) // EIGHT
            tik_instance.data_move(featuremap_ub[0, ], \
                                   featuremap_gm[
                                       0,
                                       feature_map_c1 * \
                                       block_id + i, \
                                       hstart, wstart, 0], 0, \
                                   TWO, width,
                                   feature_map_w - width, 0)
            tik_instance.vbi(FP_SIXTEEN_ONE_TIME, result_ub[index_height * (POOL_W + ONE) * C0SIZE:],
                             featuremap_ub[0:], \
                             vbi_weights[NUM_ADDR_ONE_VBI * index_height:], \
                             addr_one_time, ONE, repeat, \
                             NUM_ELMENTS_ONEBIN, ONE, \
                             EIGHT * NUM_ADDR_ONE_VBI // BYTES)

            tik_instance.data_move(
                output_gm[cur_roi_num,
                          feature_map_c1 * block_id + i, index_height, 0,
                          0],
                result_ub[0, 0, index_height, 0, 0], 0, ONE, POOL_W, 0, 0)


def roi_align_v200_compute(tik_instance, block_id, featuremap_gm, rois_gm,
                           output_gm, feature_map_dict, rois_dict):
    """
    :param tik_instance:
    :param block_id:  the block used
    :param featuremap_gm: the gm  of featuremap
    :param rois_gm: the gm for roi_boxes
    :param output_gm: the gm for output
    :param feature_map_dict: placeholder of featuremap
    :param rois_dict: the placeholder for roi_boxes
    :return: None
    """
    feature_shape = feature_map_dict.get("shape")
    feature_map_c1 = feature_shape[ONE]
    feature_map_h = feature_shape[TWO]
    feature_map_w = feature_shape[THREE]
    rois_shape = rois_dict.get("shape")
    rois_num = rois_shape[0]
    rois = tik_instance.Tensor("float16", [ROINUM_LIMIT, FOUR],
                               name="rois_ub",
                               scope=tbe_platform.scope_ubuf)
    tik_instance.vector_dup(FP_SIXTEEN_ONE_TIME, rois, 0, FOUR, ONE, EIGHT)
    tik_instance.data_move(rois, rois_gm, 0, ONE, rois_num // FOUR, 0, 0)
    tik_instance.vmaxs(FP_SIXTEEN_ONE_TIME, rois, rois, 0,
                       FOUR, ONE, ONE, EIGHT, EIGHT)
    rois_fp32, grid_hw_fp32, rois_fp32_orig = \
        _roi_align_perf_scale(tik_instance, rois, feature_map_h, feature_map_w)

    index_array_fp32 = tik_instance.Tensor("float32",
                                           [ROINUM_LIMIT, ],
                                           name="index_array_fp32",
                                           scope=tbe_platform.scope_ubuf)
    index = tik_instance.Scalar(dtype="float32", init_value=0.0)
    for i in range(0, ROINUM_LIMIT):
        index.set_as(i + HALF)
        index_array_fp32[i].set_as(index)

    if feature_map_c1 * feature_map_h * feature_map_w * C0SIZE * BYTES > \
            FM_BUFFER_SIZE_LIMIT:
        _roi_align_c1_cut(tik_instance, block_id, rois_fp32, grid_hw_fp32, \
                          featuremap_gm, output_gm, feature_shape, rois_num,
                          index_array_fp32, rois_fp32_orig)
    else:
        _roi_align_roi_num_cut(tik_instance, block_id, rois_fp32, \
                               grid_hw_fp32, featuremap_gm, \
                               output_gm, feature_shape, rois_num, \
                               index_array_fp32, rois_fp32_orig)


def _roi_align_roi_num_cut(tik_instance, block_id, rois_fp32,
                           grid_hw_fp32, featuremap_gm,
                           output_gm, feature_shape,
                           rois_num, index_array_fp32, rois_fp32_orig):
    """
    :param tik_instance:
    :param block_id: the block used
    :param rois_fp32: the pos of the roi_box
    :param grid_hw_fp32: the wide and hight of grid in roi box
    :param featuremap_gm: the gm  of featuremap
    :param output_gm: the gm for output
    :param feature_shape: shape of featuremap
    :param rois_num: the number of roi_box
    :param index_array_fp32: a array corresponding to center of every  grid
    :param rois_fp32_orig: the coordition of the roi_box
    :return: None
    """
    roi_percore = (rois_num + BLOCK_DIM - ONE) // BLOCK_DIM
    with tik_instance.for_range(0, roi_percore) as roi_percore_index:
        curr_roi = block_id + roi_percore_index * EIGHT

        zero = tik_instance.Scalar(dtype="float32",
                                   init_value=0)
        campare_w = tik_instance.Scalar(dtype="float32",
                                        init_value=rois_fp32_orig[0,
                                                                  curr_roi])
        campare_h = tik_instance.Scalar(dtype="float32",
                                        init_value=rois_fp32_orig[ONE,
                                                                  curr_roi])

        with tik_instance.if_scope(campare_w > zero):
            with tik_instance.if_scope(campare_h > zero):
                point_positions_int32, point_weights_fp32 = \
                    _roi_align_perf_gengrid_fp32(tik_instance,
                                                 curr_roi,
                                                 rois_fp32,
                                                 grid_hw_fp32,
                                                 feature_shape,
                                                 index_array_fp32)
                _process_one_roi_vbi(tik_instance, curr_roi,
                                     feature_shape, featuremap_gm,
                                     output_gm, point_positions_int32,
                                     point_weights_fp32)
            with tik_instance.else_scope():
                result_ub = tik_instance.Tensor("float16",
                                                [ONE, feature_shape[ONE],
                                                 POOL_H + ONE, POOL_W, C0SIZE],
                                                name="result_ub",
                                                scope=tbe_platform.scope_ubuf)
                tik_instance.vector_dup(FP_SIXTEEN_ONE_TIME, result_ub, 0,
                                        POOL_W * feature_shape[ONE],
                                        ONE, EIGHT)
                tik_instance.data_move(output_gm[curr_roi, 0, 0, 0, 0],
                                       result_ub[0, 0, 0, 0, 0],
                                       0, feature_shape[ONE],
                                       POOL_H * POOL_W, 0, 0)
        with tik_instance.else_scope():
            result_ub = tik_instance.Tensor("float16",
                                            [ONE, feature_shape[ONE],
                                             POOL_H + ONE, POOL_W, C0SIZE],
                                            name="result_ub",
                                            scope=tbe_platform.scope_ubuf)
            tik_instance.vector_dup(FP_SIXTEEN_ONE_TIME, result_ub, 0,
                                    POOL_W * feature_shape[ONE], ONE, EIGHT)
            tik_instance.data_move(output_gm[curr_roi, 0, 0, 0, 0],
                                   result_ub[0, 0, 0, 0, 0],
                                   0, feature_shape[ONE],
                                   POOL_H * POOL_W, 0, 0)


def _roi_align_c1_cut(tik_instance, block_id, rois_fp32, grid_hw_fp32,
                      featuremap_gm, output_gm,
                      feature_shape, rois_num,
                      index_array_fp32, rois_fp32_orig):
    """
    :param tik_instance:
    :param block_id: the block used
    :param rois_fp32: the pos of the roi_box
    :param grid_hw_fp32: the wide and hight of grid in roi box
    :param featuremap_gm: the gm  of featuremap
    :param output_gm: the gm for output
    :param feature_shape: shape of featuremap
    :param rois_num: the number of roi_box
    :param index_array_fp32: a array corresponding to center of every  grid
    :return: None
        deal with big featuremap shape, eltwise mabe ub is not enough for
    featuremap,
        so we use feature map c1-cut mode
    """
    with tik_instance.for_range(0, rois_num) as curr_roi:
        zero = tik_instance.Scalar(dtype="float32",
                                   init_value=0)
        campare_w = tik_instance.Scalar(dtype="float32",
                                        init_value=rois_fp32_orig[0,
                                                                  curr_roi])
        campare_h = tik_instance.Scalar(dtype="float32",
                                        init_value=rois_fp32_orig[ONE,
                                                                  curr_roi])
        with tik_instance.if_scope(campare_w > zero):
            with tik_instance.if_scope(campare_h > zero):
                point_positions_int32, point_weights_fp32 = \
                    _roi_align_perf_gengrid_fp32(tik_instance,
                                                 curr_roi, rois_fp32,
                                                 grid_hw_fp32,
                                                 feature_shape,
                                                 index_array_fp32)
                _process_one_roi_vbi_c1_cut(tik_instance, block_id, curr_roi,
                                            feature_shape, featuremap_gm,
                                            output_gm, point_positions_int32,
                                            point_weights_fp32)
            with tik_instance.else_scope():
                result_ub = tik_instance.Tensor("float16",
                                                [ONE, feature_shape[ONE],
                                                 POOL_H + ONE, POOL_W, C0SIZE],
                                                name="result_ub",
                                                scope=tbe_platform.scope_ubuf)
                tik_instance.vector_dup(FP_SIXTEEN_ONE_TIME, result_ub, 0,
                                        POOL_W * feature_shape[ONE],
                                        ONE, EIGHT)
                tik_instance.data_move(output_gm[curr_roi, 0, 0, 0, 0],
                                       result_ub[0, 0, 0, 0, 0],
                                       0, feature_shape[ONE],
                                       POOL_H * POOL_W, 0, 0)
        with tik_instance.else_scope():
            result_ub = tik_instance.Tensor("float16",
                                            [ONE, feature_shape[ONE],
                                             POOL_H + ONE, POOL_W, C0SIZE],
                                            name="result_ub",
                                            scope=tbe_platform.scope_ubuf)
            tik_instance.vector_dup(FP_SIXTEEN_ONE_TIME, result_ub, 0,
                                    POOL_W * feature_shape[ONE], ONE, EIGHT)
            tik_instance.data_move(output_gm[curr_roi, 0, 0, 0, 0],
                                   result_ub[0, 0, 0, 0, 0],
                                   0, feature_shape[ONE],
                                   POOL_H * POOL_W, 0, 0)


def _check_roi_align_vbi_params(feature_map, rois):
    """ 4*n
    :param feature_map:  placeholder of  feature_map
    :param rois: placeholder of  rois
    :return: None
    """
    shape_featuremap = feature_map.get('shape')
    shape_rois = rois.get('shape')
    dtype_featuremap = feature_map.get('dtype').lower()
    dtype_rois = rois.get('dtype').lower()
    if dtype_featuremap != 'float16':
        raise RuntimeError("dtype of feature_map should be float16")
    if dtype_rois != 'float16':
        raise RuntimeError("dtype of rois should be float16")
    if len(shape_featuremap) != FIVE:
        raise RuntimeError("dimension of featuremap should be FIVE")
    if shape_featuremap[0] != ONE:
        raise RuntimeError("first dimension of featuremap should be ONE")
    if len(shape_rois) != TWO:
        raise RuntimeError("dimension of rois should be TWO")
    if shape_rois[ONE] != FOUR:
        raise RuntimeError("second dimension of rois should be FOUR")
    if shape_rois[0] < ONE:
        raise RuntimeError("the num of rois should be no less than ONE")
    if shape_rois[0] % FOUR:
        raise RuntimeError("the num of rois should be divisible by FOUR")
    if shape_rois[0] > NINETY_SIX:
        raise RuntimeError("the num of rois should be no more than NINETY_SIX")


@para_check.check_input_type(dict, dict, str)
def roi_align_vbi(featuremap, rois_box,
                  kernel_name="roi_align"):
    """
    roi_align API used only for 2d-h1 net in v200 aic (vbi support)
    network type: tensorflow
    dtype: float16
    pool_h: 7
    pool_w: 7
    sample_ratio: 1
    rois num range: 1-96
    block_dim: 8
    :param feature_map_dict: placeholder of  feature_map
    :param rois_dict:  placeholder of  rois
    :param kernel_name: name of kernel
    :return: the roi_align_vbi result
    """
    _check_roi_align_vbi_params(featuremap, rois_box)
    para_check.check_kernel_name(kernel_name)

    tik_instance = tik.Tik(tik.Dprofile(), True)
    rois_shape = rois_box.get("shape")
    dtype = featuremap.get("dtype")
    feature_shape = featuremap.get("shape")
    feature_map = tik_instance.Tensor(
        dtype, feature_shape, name="feature_map", scope=tbe_platform.scope_gm)
    rois = tik_instance.Tensor(
        dtype, rois_shape, name="rois", scope=tbe_platform.scope_gm)
    fm_c1 = feature_shape[ONE]
    ret = tik_instance.Tensor(
        dtype, [rois_shape[0], fm_c1, POOL_H, POOL_W, C0SIZE],
        name="ret",
        scope=tbe_platform.scope_gm)

    with tik_instance.for_range(0, BLOCK_DIM,
                                block_num=BLOCK_DIM) as block_id:
        roi_align_v200_compute(tik_instance, block_id,
                               feature_map,
                               rois, ret, featuremap,
                               rois_box)
    tik_instance.BuildCCE(
        kernel_name=kernel_name, inputs=[feature_map, rois],
        outputs=[ret])
    return tik_instance
