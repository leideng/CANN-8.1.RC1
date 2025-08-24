#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2019-2022. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0
trans_data_normalization
"""
import copy
from impl.util.util_common import is_unknown_rank_input
from impl.util.util_common import is_dynamic_input
from impl.util.util_common import BIT_RATIO_DICT, BLOCK_SIZE, B32_SIZE

NUMBER_16 = 16
NEG_ONE = -1


def reverse_infos(_infos):
    result = []
    for i, j, k in _infos:
        info = [j, i, k]
        result.append(info)
    return result


def get_possible_c0(bit_size):
    _c0_set = {BLOCK_SIZE // bit_size}
    if bit_size == B32_SIZE:
        _c0_set.add(NUMBER_16)
    return _c0_set


def nd2nz_normalize(src, dst):
    # binary is -2.
    if is_unknown_rank_input([src, dst]):
        result = []
        for c0 in get_possible_c0(BIT_RATIO_DICT.get(src.get("dtype"))):
            src["shape"] = [-1, -1, -1]
            dst["shape"] = [-1, -1, -1, 16, c0]
            result.append([copy.deepcopy(src), copy.deepcopy(dst), c0])
        return result

    # dynamic (user define C0)
    if is_dynamic_input([src, dst]):
        src["shape"] = [-1, -1, -1]
        dst["shape"] = [-1, -1, -1, 16, dst["shape"][-1]]
        return [[src, dst, dst["shape"][-1]], ]

    # const
    return [[src, dst, dst["shape"][-1]], ]


def nchw2nc1hwc0_normalize(src, dst):
    # binary is -2.
    if is_unknown_rank_input([src, dst]):
        result = []
        for c0 in get_possible_c0(BIT_RATIO_DICT.get(src.get("dtype"))):
            src["shape"] = [-1, -1, -1, -1]
            dst["shape"] = [-1, -1, -1, -1, c0]
            result.append([copy.deepcopy(src), copy.deepcopy(dst), c0])
        return result

    # dynamic (user define C0)
    if is_dynamic_input([src, dst]):
        src["shape"] = [-1, -1, -1, -1]
        dst["shape"] = [-1, -1, -1, -1, dst["shape"][-1]]
        return [[src, dst, dst["shape"][-1]], ]

    # const
    return [[src, dst, dst["shape"][-1]], ]


def nhwc2nc1hwc0_normalize(src, dst):
    return nchw2nc1hwc0_normalize(src, dst)


def hwcn2fz_normalize(src, dst):
    # binary
    if is_unknown_rank_input([src, dst]):
        result = []
        for c0 in get_possible_c0(BIT_RATIO_DICT.get(src.get("dtype"))):
            src["shape"] = [-1, -1, -1, -1]  # HWCN
            dst["shape"] = [-1, -1, -1, -1, 16, c0]  # C1HWN1N0C0
            result.append([copy.deepcopy(src), copy.deepcopy(dst), c0])
        return result

    # dynamic (user define C0)
    if is_dynamic_input([src, dst]):
        src["shape"] = [-1, -1, -1, -1]
        dst["shape"] = [-1, -1, -1, -1, 16, dst["shape"][-1]]
        return [[src, dst, dst["shape"][-1]], ]

    # const
    h, w, c, n = src.get("shape")
    c1hw, n1, n0, c0 = dst.get("shape")
    new_dst_shape = [c1hw // h // w, h, w, n1, n0, c0]
    dst["shape"] = new_dst_shape
    return [[src, dst, dst["shape"][-1]], ]


def fz2hwcn_normalize(src, dst):
    # binary
    if is_unknown_rank_input([src, dst]):
        result = []
        for c0 in get_possible_c0(BIT_RATIO_DICT.get(src.get("dtype"))):
            dst["shape"] = [-1, -1, -1, -1]  # HWCN
            src["shape"] = [-1, -1, -1, -1, 16, c0]  # C1HWN1N0C0
            result.append([copy.deepcopy(src), copy.deepcopy(dst), c0])
        return result

    # dynamic (user define C0)
    if is_dynamic_input([src, dst]):
        dst["shape"] = [-1, -1, -1, -1]
        src["shape"] = [-1, -1, -1, -1, 16, src["shape"][-1]]
        return [[src, dst, src["shape"][-1]], ]

    # const
    h, w, c, n = dst.get("shape")
    c1hw, n1, n0, c0 = src.get("shape")
    new_src_shape = [c1hw // h // w, h, w, n1, n0, c0]
    src["shape"] = new_src_shape
    return [[src, dst, src["shape"][-1]], ]


def nchw2fz_normalize(src, dst):
    # binary
    if is_unknown_rank_input([src, dst]):
        result = []
        for c0 in get_possible_c0(BIT_RATIO_DICT.get(src.get("dtype"))):
            src["shape"] = [-1, -1, -1, -1]  # NCHW
            dst["shape"] = [-1, -1, -1, -1, 16, c0]  # C1HWN1N0C0
            result.append([copy.deepcopy(src), copy.deepcopy(dst), c0])
        return result

    # dynamic (user define C0)
    if is_dynamic_input([src, dst]):
        src["shape"] = [-1, -1, -1, -1]
        dst["shape"] = [-1, -1, -1, -1, 16, dst["shape"][-1]]
        return [[src, dst, dst["shape"][-1]], ]

    # const
    n, c, h, w = src.get("shape")
    c1hw, n1, n0, c0 = dst.get("shape")
    new_dst_shape = [c1hw // h // w, h, w, n1, n0, c0]
    dst["shape"] = new_dst_shape
    return [[src, dst, dst["shape"][-1]], ]


def fz2nchw_normalize(src, dst):
    # binary
    if is_unknown_rank_input([src, dst]):
        result = []
        for c0 in get_possible_c0(BIT_RATIO_DICT.get(src.get("dtype"))):
            dst["shape"] = [-1, -1, -1, -1]  # NCHW
            src["shape"] = [-1, -1, -1, -1, 16, c0]  # C1HWN1N0C0
            result.append([copy.deepcopy(src), copy.deepcopy(dst), c0])
        return result

    # dynamic (user define C0)
    if is_dynamic_input([src, dst]):
        dst["shape"] = [-1, -1, -1, -1]
        src["shape"] = [-1, -1, -1, -1, 16, src["shape"][-1]]
        return [[src, dst, src["shape"][-1]], ]

    # const
    n, c, h, w = dst.get("shape")
    c1hw, n1, n0, c0 = src.get("shape")
    new_src_shape = [c1hw // h // w, h, w, n1, n0, c0]
    src["shape"] = new_src_shape
    return [[src, dst, src["shape"][-1]], ]


def ndhwc_2_ndc1hwc0_normalize(src, dst):
    # binary is -2.
    if is_unknown_rank_input([src, dst]):
        result = []
        for c0 in get_possible_c0(BIT_RATIO_DICT.get(src.get("dtype"))):
            src["shape"] = [NEG_ONE, NEG_ONE, NEG_ONE, NEG_ONE, NEG_ONE]
            dst["shape"] = [NEG_ONE, NEG_ONE, NEG_ONE, NEG_ONE, NEG_ONE, c0]
            result.append([copy.deepcopy(src), copy.deepcopy(dst), c0])
        return result

    # dynamic scene user define C0
    if is_dynamic_input([src, dst]):
        src["shape"] = [NEG_ONE, NEG_ONE, NEG_ONE, NEG_ONE, NEG_ONE]
        dst["shape"] = [NEG_ONE, NEG_ONE, NEG_ONE, NEG_ONE, NEG_ONE, dst["shape"][NEG_ONE]]
        return [[src, dst, dst["shape"][NEG_ONE]], ]

    # const
    return [[src, dst, dst["shape"][NEG_ONE]], ]


def trans_data_normalization(src, dst, src_format, dst_format):
    src = copy.deepcopy(src)
    dst = copy.deepcopy(dst)
    if (src_format, dst_format) == ("ND", "FRACTAL_NZ"):
        return nd2nz_normalize(src, dst)
    elif (src_format, dst_format) == ("FRACTAL_NZ", "ND"):
        return reverse_infos(nd2nz_normalize(dst, src))
    elif (src_format, dst_format) == ("NHWC", "NC1HWC0"):
        return nhwc2nc1hwc0_normalize(src, dst)
    elif (src_format, dst_format) == ("NC1HWC0", "NHWC"):
        return reverse_infos(nhwc2nc1hwc0_normalize(dst, src))
    elif (src_format, dst_format) == ("NCHW", "NC1HWC0"):
        return nchw2nc1hwc0_normalize(src, dst)
    elif (src_format, dst_format) == ("NC1HWC0", "NCHW"):
        return reverse_infos(nchw2nc1hwc0_normalize(dst, src))
    elif (src_format, dst_format) == ("HWCN", "FRACTAL_Z"):
        return hwcn2fz_normalize(src, dst)
    elif (src_format, dst_format) == ("FRACTAL_Z", "HWCN"):
        return reverse_infos(hwcn2fz_normalize(dst, src))
    elif (src_format, dst_format) == ("NCHW", "FRACTAL_Z"):
        return nchw2fz_normalize(src, dst)
    elif (src_format, dst_format) == ("FRACTAL_Z", "NCHW"):
        return reverse_infos(nchw2fz_normalize(dst, src))
    elif (src_format, dst_format) == ("NDHWC", "NDC1HWC0"):
        return ndhwc_2_ndc1hwc0_normalize(src, dst)
    elif (src_format, dst_format) == ("NDC1HWC0", "NDHWC"):
        return reverse_infos(ndhwc_2_ndc1hwc0_normalize(dst, src))
    else:
        raise RuntimeError("Not support format")
