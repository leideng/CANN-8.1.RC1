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

trans_data_format_rule
"""
from impl.util.platform_adapter import error_manager_vector

# LENGTH
LENGTH_6 = 6
LENGTH_5 = 5
LENGTH_4 = 4
LENGTH_2 = 2


def check_nd_and_nz(src_shape, dst_shape, is_forward=True):
    if is_forward and len(src_shape) + LENGTH_2 != len(dst_shape):
        message = "src is %s and dst is %s in ND_2_NZ that is illegal, " \
                  "len(src) should equal to len(dst) - 2." % (src_shape, dst_shape)
        error_manager_vector.raise_err_specific_reson("trans_data", message)

    if not is_forward and len(src_shape) - LENGTH_2 != len(dst_shape):
        message = "src is %s and dst is %s in NZ_2_ND that is illegal, " \
                  "len(src) should equal to len(dst) + 2." % (src_shape, dst_shape)
        error_manager_vector.raise_err_specific_reson("trans_data", message)


def check_nchw_and_nc1hwc0(src_shape, dst_shape, is_forward=True):
    if is_forward and (len(src_shape) != LENGTH_4 or len(dst_shape) != LENGTH_5):
        message = "src is %s and dst is %s in nchw_2_nc1hwc0 that is illegal, " \
                  "len(src) should be 4 and len(dst) should be 5." % (src_shape, dst_shape)
        error_manager_vector.raise_err_specific_reson("trans_data", message)

    if not is_forward and (len(src_shape) != LENGTH_5 or len(dst_shape) != LENGTH_4):
        message = "src is %s and dst is %s in nc1hwc0_2_nchw that is illegal, " \
                  "len(src) should be 5 and len(dst) should be 4." % (src_shape, dst_shape)
        error_manager_vector.raise_err_specific_reson("trans_data", message)


def check_nhwc_and_nc1hwc0(src_shape, dst_shape, is_forward=True):
    if is_forward and (len(src_shape) != LENGTH_4 or len(dst_shape) != LENGTH_5):
        message = "src is %s and dst is %s in nhwc_2_nc1hwc0 that is illegal, " \
                  "len(src) should be 4 and len(dst) should be 5." % (src_shape, dst_shape)
        error_manager_vector.raise_err_specific_reson("trans_data", message)

    if not is_forward and (len(src_shape) != LENGTH_5 or len(dst_shape) != LENGTH_4):
        message = "src is %s and dst is %s in nc1hwc0_2_nhwc that is illegal, " \
                  "len(src) should be 5 and len(dst) should be 4." % (src_shape, dst_shape)
        error_manager_vector.raise_err_specific_reson("trans_data", message)


def check_hwcn_and_fractal_z(src_shape, dst_shape, is_forward=True):
    if is_forward and (len(src_shape) != LENGTH_4 or len(dst_shape) != LENGTH_6):
        message = "src is %s and dst is %s in hwcn_2_fractal_z that is illegal, " \
                  "len(src) should be 4 and len(dst) should be 6." % (src_shape, dst_shape)
        error_manager_vector.raise_err_specific_reson("trans_data", message)

    if not is_forward and (len(src_shape) != LENGTH_6 or len(dst_shape) != LENGTH_4):
        message = "src is %s and dst is %s in fractal_z_2_hwcn that is illegal, " \
                  "len(src) should be 6 and len(dst) should be 4." % (src_shape, dst_shape)
        error_manager_vector.raise_err_specific_reson("trans_data", message)


def check_nchw_and_fractal_z(src_shape, dst_shape, is_forward=True):
    if is_forward and (len(src_shape) != LENGTH_4 or len(dst_shape) != LENGTH_6):
        message = "src is %s and dst is %s in nchw_2_fractal_z that is illegal, " \
                  "len(src) should be 4 and len(dst) should be 6." % (src_shape, dst_shape)
        error_manager_vector.raise_err_specific_reson("trans_data", message)

    if not is_forward and (len(src_shape) != LENGTH_6 or len(dst_shape) != LENGTH_4):
        message = "src is %s and dst is %s in fractal_z_2_nchw that is illegal, " \
                  "len(src) should be 6 and len(dst) should be 4." % (src_shape, dst_shape)
        error_manager_vector.raise_err_specific_reson("trans_data", message)


def check_ndhwc_and_ndc1hwc0(src_shape, dst_shape, is_forward=True):
    if is_forward and (len(src_shape) != LENGTH_5 or len(dst_shape) != LENGTH_6):
        message = "src is %s and dst is %s in ndhwc_2_ndc1hwc0 that is illegal, " \
                  "len(src) should be 5 and len(dst) should be 6." % (src_shape, dst_shape)
        error_manager_vector.raise_err_specific_reson("trans_data", message)

    if not is_forward and (len(src_shape) != LENGTH_6 or len(dst_shape) != LENGTH_5):
        message = "src is %s and dst is %s in ndc1hwc0_2_ndhwc that is illegal, " \
                  "len(src) should be 6 and len(dst) should be 5." % (src_shape, dst_shape)
        error_manager_vector.raise_err_specific_reson("trans_data", message)


def fractal_nz_2_nd(src_shape, dst_shape):
    # rule from nz_2_nd
    check_nd_and_nz(src_shape, dst_shape, is_forward=False)
    dst = list(range(len(dst_shape)))
    begin, left, right = [], [1, 2], [0, 3]
    extent = len(src_shape) - LENGTH_4
    if extent > 0:
        left = list(x + extent for x in left)
        right = list(x + extent for x in right)
        begin = list(range(extent))
    src = tuple(begin + [tuple(left), ] + [tuple(right), ])
    axes_map = dict(zip(src, dst))
    return axes_map


def nd_2_fractal_nz(src_shape, dst_shape):
    # rule from nd_2_nz
    check_nd_and_nz(src_shape, dst_shape, is_forward=True)
    src = list(range(len(src_shape)))
    begin, left, right = [], [1, 2], [0, 3]
    extent = len(dst_shape) - LENGTH_4
    if extent > 0:
        left = list(x + extent for x in left)
        right = list(x + extent for x in right)
        begin = list(range(extent))
    dst = tuple(begin + [tuple(left), ] + [tuple(right), ])
    axes_map = dict(zip(src, dst))
    return axes_map


def nchw_2_nc1hwc0(src_shape, dst_shape):
    check_nchw_and_nc1hwc0(src_shape, dst_shape, is_forward=True)
    return {0: 0, 1: (1, 4), 2: 2, 3: 3, }


def nc1hwc0_2_nchw(src_shape, dst_shape):
    check_nchw_and_nc1hwc0(src_shape, dst_shape, is_forward=False)
    return {0: 0, (1, 4): 1, 2: 2, 3: 3, }


def nhwc_2_nc1hwc0(src_shape, dst_shape):
    check_nhwc_and_nc1hwc0(src_shape, dst_shape, is_forward=True)
    return {0: 0, 1: 2, 2: 3, 3: (1, 4), }


def nc1hwc0_2_nhwc(src_shape, dst_shape):
    check_nhwc_and_nc1hwc0(src_shape, dst_shape, is_forward=False)
    return {0: 0, 2: 1, 3: 2, (1, 4): 3, }


def hwcn_2_fractal_z(src_shape, dst_shape):
    check_hwcn_and_fractal_z(src_shape, dst_shape, is_forward=True)
    return {0: 1, 1: 2, 2: (0, 5), 3: (3, 4)}


def fractal_z_2_hwcn(src_shape, dst_shape):
    check_hwcn_and_fractal_z(src_shape, dst_shape, is_forward=False)
    return {(0, 5): 2, 1: 0, 2: 1, (3, 4): 3}


def nchw_2_fractal_z(src_shape, dst_shape):
    check_nchw_and_fractal_z(src_shape, dst_shape, is_forward=True)
    return {0: (3, 4), 1: (0, 5), 2: 1, 3: 2}


def fractal_z_2_nchw(src_shape, dst_shape):
    check_nchw_and_fractal_z(src_shape, dst_shape, is_forward=False)
    return {(0, 5): 1, 1: 2, 2: 3, (3, 4): 0}


def ndhwc_2_ndc1hwc0(src_shape, dst_shape):
    check_ndhwc_and_ndc1hwc0(src_shape, dst_shape, is_forward=True)
    return {0: 0, 1: 1, 2 : 3, 3: 4, 4: (2, 5), }


def ndc1hwc0_2_ndhwc(src_shape, dst_shape):
    check_ndhwc_and_ndc1hwc0(src_shape, dst_shape, is_forward=False)
    return {0: 0, 1: 1, 3 : 2, 4: 3, (2, 5) : 4, }


def trans_data_format_rule(src, dst, src_format, dst_format):
    _hash = {("ND", "FRACTAL_NZ"): nd_2_fractal_nz, ("FRACTAL_NZ", "ND"): fractal_nz_2_nd,
             ("NCHW", "NC1HWC0"): nchw_2_nc1hwc0, ("NC1HWC0", "NCHW"): nc1hwc0_2_nchw,
             ("NHWC", "NC1HWC0"): nhwc_2_nc1hwc0, ("NC1HWC0", "NHWC"): nc1hwc0_2_nhwc,
             ("HWCN", "FRACTAL_Z"): hwcn_2_fractal_z, ("FRACTAL_Z", "HWCN"): fractal_z_2_hwcn,
             ("NCHW", "FRACTAL_Z"): nchw_2_fractal_z, ("FRACTAL_Z", "NCHW"): fractal_z_2_nchw,
             ("NDHWC", "NDC1HWC0"): ndhwc_2_ndc1hwc0, ("NDC1HWC0", "NDHWC"): ndc1hwc0_2_ndhwc, }

    # rule for trans_data
    src_shape = src.get("shape", None)
    dst_shape = dst.get("shape", None)
    func = _hash.get((src_format, dst_format), None)
    if not func:
        error_manager_vector.raise_err_specific_reson("trans_data", "format is not support.")
    return func(src_shape, dst_shape)

