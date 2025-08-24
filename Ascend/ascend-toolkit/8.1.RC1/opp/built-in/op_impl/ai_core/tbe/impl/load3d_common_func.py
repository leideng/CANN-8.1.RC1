#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
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
load3d_common_func
"""
from impl.util.platform_adapter import tik


# 'pylint: disable=too-many-arguments,too-many-locals,too-many-statements
def img2col(tik_instance, ori_input_l1, ori_input_col_ub, l1_start_offset, ub_start_offset, index_kh, index_kw,
            left_top_h, left_top_w, l1_h, l1_w, kernel_h, kernel_w, stride_h, stride_w, repeat_times, repeat_mode, pad):
    """
    load3dv1
    """
    C0 = 16
    SCALAR_TYPE = "int64"
    # load3d process num per repeat
    LOAD3D_NUM_PER_REPEAT = 256
    # load3d max loop count for repeat_mode = 0
    MAX_LOOP_COUNT = 16

    pad_left, pad_right, pad_top, pad_bottom = pad
    left_top_h_scalar = tik_instance.Scalar(dtype=SCALAR_TYPE, name="left_top_h", init_value=left_top_h)
    left_top_w_scalar = tik_instance.Scalar(dtype=SCALAR_TYPE, name="left_top_w", init_value=left_top_w)

    padding_l1_h = tik_instance.Scalar(dtype=SCALAR_TYPE, name="padding_l1_h")
    padding_l1_w = tik_instance.Scalar(dtype=SCALAR_TYPE, name="padding_l1_w")
    padding_l1_h.set_as(l1_h + pad_top + pad_bottom)
    padding_l1_w.set_as(l1_w + pad_left + pad_right)

    ho = tik_instance.Scalar(dtype=SCALAR_TYPE, name="ho")
    wo = tik_instance.Scalar(dtype=SCALAR_TYPE, name="wo")
    top_wo = tik_instance.Scalar(dtype=SCALAR_TYPE, name="top_wo")
    ho.set_as((padding_l1_h - pad_top - left_top_h_scalar + stride_h - 1) // stride_h - 1)
    wo.set_as((padding_l1_w - kernel_w) // stride_w + 1)
    top_wo.set_as((padding_l1_w - pad_left - left_top_w_scalar - kernel_w) // stride_w + 1)

    index_h = tik_instance.Scalar(dtype=SCALAR_TYPE, name="index_h")
    index_w = tik_instance.Scalar(dtype=SCALAR_TYPE, name="index_w")
    offset_l1 = tik_instance.Scalar(dtype=SCALAR_TYPE, name="offset_l1")
    offset_ub = tik_instance.Scalar(dtype=SCALAR_TYPE, name="offset_ub")
    n_burst = tik_instance.Scalar(dtype=SCALAR_TYPE, name="n_burst")
    index_wo_min = tik_instance.Scalar(dtype=SCALAR_TYPE, name="index_wo_min")
    index_wo_max = tik_instance.Scalar(dtype=SCALAR_TYPE, name="index_wo_max")
    index_ho = tik_instance.Scalar(dtype=SCALAR_TYPE, name="index_ho")

    def load3d_l1_to_ub(idx_ho, idx_kh, idx_kw, max_wo, first_wi, first_wo, actual_pad_left):
        index_ho.set_as(idx_ho)
        index_h.set_as(stride_h * (index_ho + 1) + idx_kh + left_top_h_scalar + pad_top)
        with tik_instance.if_scope(tik.all(index_h >= pad_top, index_h < l1_h + pad_top)):
            # `for (0, wo) as index_wo:`
            # `index_w = index_kw + left_top_w_scalar + pad_left + stride_w * index_wo`
            # `index_w in range [pad_left, l1_w + pad_left)`
            index_wo_min.set_as((pad_left - idx_kw - first_wi - actual_pad_left + stride_w - 1) // stride_w)
            index_wo_max.set_as((l1_w + pad_left - idx_kw - first_wi - actual_pad_left - 1) // stride_w)
            with tik_instance.if_scope(index_wo_min < 0):
                index_wo_min.set_as(0)
            with tik_instance.if_scope(index_wo_max >= max_wo):
                index_wo_max.set_as(max_wo - 1)

            n_burst.set_as(index_wo_max - index_wo_min + 1)
            with tik_instance.if_scope(index_ho == -1):
                index_ho.set_as(0)
            # load num cannot exceed repeat_times * 256
            with tik_instance.if_scope(
                    (first_wo + index_ho * wo + index_wo_max + 1) * C0 > repeat_times * LOAD3D_NUM_PER_REPEAT):
                n_burst.set_as(repeat_times * LOAD3D_NUM_PER_REPEAT // C0 - index_wo_min - first_wo - index_ho * wo)
            # `if index_wo_max < 0, n_burst = 0`
            with tik_instance.if_scope((l1_w + pad_left - idx_kw - first_wi - actual_pad_left) < 1):
                n_burst.set_as(0)

            with tik_instance.if_scope(n_burst > 0):
                index_w.set_as(idx_kw + first_wi + actual_pad_left + stride_w * index_wo_min)
                offset_l1.set_as(((index_h - pad_top) * l1_w + (index_w - pad_left)) * C0)
                if repeat_mode == 1:
                    offset_ub.set_as((first_wo + index_ho * wo + index_wo_min) * C0)
                else:
                    offset_ub.set_as((idx_kh * kernel_w + idx_kw) * LOAD3D_NUM_PER_REPEAT +
                                     (first_wo + index_ho * wo + index_wo_min) * C0)
                tik_instance.data_move(ori_input_col_ub[ub_start_offset + offset_ub],
                                       ori_input_l1[l1_start_offset + offset_l1],
                                       0, n_burst, 1, stride_w - 1, 0)

    if repeat_mode == 1:
        # process first ho
        load3d_l1_to_ub(-1, index_kh, index_kw, top_wo, left_top_w_scalar, 0, pad_left)
        # process remain ho
        with tik_instance.for_range(0, ho) as idx_ho:
            load3d_l1_to_ub(idx_ho, index_kh, index_kw, wo, 0, top_wo, 0)
    else:
        with tik_instance.if_scope(top_wo >= MAX_LOOP_COUNT):
            with tik_instance.for_range(index_kh, kernel_h) as idx_kh:
                with tik_instance.for_range(index_kw, kernel_w) as idx_kw:
                    load3d_l1_to_ub(-1, idx_kh, idx_kw, MAX_LOOP_COUNT, left_top_w_scalar, 0, pad_left)
        with tik_instance.else_scope():
            remain_wo = tik_instance.Scalar(dtype=SCALAR_TYPE, name="remain_wo", init_value=0)
            with tik_instance.if_scope((top_wo + ho * wo) > MAX_LOOP_COUNT):
                ho.set_as((MAX_LOOP_COUNT - top_wo) // wo)
                remain_wo.set_as((MAX_LOOP_COUNT - top_wo) % wo)

            with tik_instance.for_range(index_kh, kernel_h) as idx_kh:
                with tik_instance.for_range(index_kw, kernel_w) as idx_kw:
                    # process first ho
                    load3d_l1_to_ub(-1, idx_kh, idx_kw, top_wo, left_top_w_scalar, 0, pad_left)
                    # process remain ho
                    with tik_instance.for_range(0, ho) as idx_ho:
                        load3d_l1_to_ub(idx_ho, idx_kh, idx_kw, wo, 0, top_wo, 0)
                    # process remain wo
                    load3d_l1_to_ub(ho, idx_kh, idx_kw, remain_wo, 0, top_wo, 0)
