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
layer_norm_unify
"""
from copy import deepcopy
from impl.util.platform_adapter import tbe_platform
from impl.util import util_common
from tbe.dsl.static_schedule.util import gcd
from tbe.dsl.static_schedule.util import get_block_factor_conservative
from tbe.dsl.static_schedule.util import get_block_factor_radical
from tbe.dsl.static_schedule.util import get_ub_factor


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant.
    """
    UB_SIZE = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    CORE_NUM = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    # Maximum number of surviving nodes for special cases, such as size of reduce axis is 768 or 1024
    TOTAL_WIDTH_0 = {"float32": 7.9, "float16": 7.9}
    # Maximum number of surviving nodes for common cases
    TOTAL_WIDTH_1 = {"float32": 8.1, "float16": 10}
    # impl_mode is high_precision, Maximum number of surviving nodes for common case
    TOTAL_WIDTH_2 = 11
    # used by TOTAL_WIDTH_0
    SPECIAL_REDUCE_AXES = [768, 1024]
    # UB buffer alignment number
    UB_ALIGN_FACTOR = 128
    # minimum alignment number
    MIN_ALIGN_FACTOR = {"float32": 8, "float16": 16}
    DEFAULT_INDEX = -1
    INIT_SIZE = 1
    TILING_RADICAL = 0
    TILING_CONSERVATIVE = 1


# 'pylint: disable = unused-argument
# 'pylint: disable=too-many-arguments,too-many-locals
def set_range(input_x, input_gamma, input_beta):
    """
    Set range information
    """
    range_x = []
    for dim in input_x.get("shape"):
        range_x.append((dim, dim))
    input_x["range"] = range_x
    range_gamma = []
    for dim in input_gamma.get("shape"):
        range_gamma.append((dim, dim))
    input_gamma["range"] = range_gamma
    range_beta = []
    for dim in input_beta.get("shape"):
        range_beta.append((dim, dim))
    input_beta["range"] = range_beta

    return input_x, input_gamma, input_beta


def _is_support_device():
    cur_cce_product = tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION)
    support_device = (tbe_platform.ASCEND_910, tbe_platform.ASCEND_310P,)
    if cur_cce_product in support_device:
        return True
    return False


def _is_in_white_list(input_x, input_gamma, input_beta):
    # Rules that besides the func:_is_unsupported_or_single_core
    # reserved for future
    shape_x = list(input_x.get("shape"))
    shape_gamma = list(input_gamma.get("shape"))
    shape_beta = list(input_beta.get("shape"))
    white_list_x = [[50, 32, 768]]
    white_list_gamma = [[768]]
    white_list_beta = [[768]]

    if shape_x in white_list_x and shape_gamma in white_list_gamma and shape_beta in white_list_beta:
        return True
    return False


# 'pylint: disable = unused-argument
def _is_unsupported_or_single_core(input_x, input_gamma, input_beta, begin_norm_axis, impl_mode):
    # Scenes that reduce_multi_schedule unsupported or unable to enable multi-core
    shape_x = list(input_x.get("shape"))
    dtype = input_x.get("dtype").lower()
    ori_shape_x = list(input_x.get("ori_shape"))
    input_format = input_x.get("format").upper()
    index_list = [index for index, _ in enumerate(ori_shape_x)]
    reduce_axis_list = index_list[begin_norm_axis:]
    if input_format == "FRACTAL_NZ":
        reduce_axis_list = to_frac_z_axis(ori_shape_x, reduce_axis_list)

    # all dims reduce
    if not begin_norm_axis:
        return True

    size_reduce_axis = Constant.INIT_SIZE
    for dim in reduce_axis_list:
        size_reduce_axis *= shape_x[dim]
    is_last_axis_align = size_reduce_axis % Constant.MIN_ALIGN_FACTOR.get(dtype)
    # Scenes with misaligned reduce axes
    if is_last_axis_align:
        return True

    if impl_mode == "high_precision":
        total_width = Constant.TOTAL_WIDTH_2
    else:
        total_width = Constant.TOTAL_WIDTH_0.get(dtype) if size_reduce_axis in Constant.SPECIAL_REDUCE_AXES \
            else Constant.TOTAL_WIDTH_1.get(dtype)
    # Bytes of fp16
    bytes_size = 2
    total_size = Constant.UB_SIZE // bytes_size
    max_ub_count = int(total_size / total_width)
    max_ub_count = int(max_ub_count // Constant.UB_ALIGN_FACTOR) * Constant.UB_ALIGN_FACTOR
    # get reduce dims size
    limit_ub_count = 1
    middle_output_shape = deepcopy(shape_x)
    for i in reduce_axis_list:
        limit_ub_count *= shape_x[i]
        middle_output_shape[i] = 1

    res_size = max_ub_count // limit_ub_count
    # Ensure that the result after reduce exceeds 1 block
    if res_size < Constant.MIN_ALIGN_FACTOR.get(dtype):
        return True

    return __penultimate_two_dimensional_case(shape_x, reduce_axis_list, dtype, limit_ub_count, max_ub_count,
                                              middle_output_shape)


def __penultimate_two_dimensional_case(shape_x, reduce_axis_list, dtype, limit_ub_count, max_ub_count,
                                       middle_output_shape):
    # Penultimate two-dimensional
    # last dim
    last_dim = -1
    penultimate_two_dims = -2
    if len(shape_x) > 1:
        pre_reduce = 1
        for dim in shape_x[:last_dim]:
            pre_reduce *= dim
        if shape_x[penultimate_two_dims] != 1 and shape_x[penultimate_two_dims] % Constant.MIN_ALIGN_FACTOR.get(dtype) \
                and pre_reduce > Constant.MIN_ALIGN_FACTOR.get(dtype):
            return True
    return __calcu_align_in_reduce_last_single_core(shape_x, reduce_axis_list, dtype, limit_ub_count, max_ub_count,
                                                    middle_output_shape)


def __calcu_align_in_reduce_last_single_core(shape_x, reduce_axis_list, dtype, limit_ub_count, max_ub_count,
                                             middle_output_shape):
    # ub calculate align, util touch non reduce axis
    core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    _last_axis_index = len(shape_x) - 1
    special_reduce_axis_size = Constant.INIT_SIZE
    reduce_align_flag = True
    for idx in reversed(range(len(shape_x))):
        if idx not in reduce_axis_list:
            break
        special_reduce_axis_size *= shape_x[idx]
        if special_reduce_axis_size % Constant.MIN_ALIGN_FACTOR.get(dtype) == 0:
            reduce_align_flag = False

    align_count = Constant.MIN_ALIGN_FACTOR.get(dtype)
    if reduce_align_flag:
        align_axis_ori_size = shape_x[_last_axis_index]
        ori_coef = gcd(align_axis_ori_size, Constant.MIN_ALIGN_FACTOR.get(dtype))
        new_coef = gcd(special_reduce_axis_size, Constant.MIN_ALIGN_FACTOR.get(dtype))
        align_count = int(Constant.MIN_ALIGN_FACTOR.get(dtype) / (new_coef / ori_coef))
        align_axis_new_size = util_common.align(align_axis_ori_size, align_count)
        shape_x[_last_axis_index] //= align_axis_new_size
        limit_ub_count = limit_ub_count / align_axis_ori_size * align_axis_new_size

    # scenes with size of reduce axis exceed the UB size
    if limit_ub_count > max_ub_count:
        return True

    return __calcu_align_in_gm_single_core(shape_x, reduce_axis_list, dtype, limit_ub_count, max_ub_count,
                                           middle_output_shape, _last_axis_index, core_num, align_count)


def __calcu_align_in_gm_single_core(shape_x, reduce_axis_list, dtype, limit_ub_count, max_ub_count,
                                    middle_output_shape, _last_axis_index, core_num, align_count):
    # gm align forcus on middle output data, it must more than one block
    # or multi core pipe will be a problem
    res_align_dim = Constant.DEFAULT_INDEX
    res_align_factor = Constant.DEFAULT_INDEX
    middle_output_tensor = (shape_x, middle_output_shape, middle_output_shape)
    for middle_output in middle_output_tensor:
        cur_align_status = False
        cur_factor = Constant.MIN_ALIGN_FACTOR.get(dtype)
        for i in reversed(range(len(middle_output))):
            # find align dim and factor in middle_output_tensor
            if cur_factor <= middle_output[i]:
                cur_align_status = True
                if (res_align_dim == Constant.DEFAULT_INDEX or res_align_dim > i or
                        (res_align_dim == i and cur_factor > res_align_factor)) and i not in reduce_axis_list:
                    res_align_dim = i
                    res_align_factor = cur_factor
                break
            cur_factor = util_common.ceil(cur_factor, middle_output[i])
        # not fit gm align, unsupport
        if not cur_align_status:
            return True
    # all align default by current barrier
    if res_align_dim == Constant.DEFAULT_INDEX or res_align_factor == Constant.DEFAULT_INDEX or \
            res_align_dim == _last_axis_index:
        return True

    # make align axis as barrier
    for i in range(res_align_dim + 1, len(shape_x)):
        if i not in reduce_axis_list:
            limit_ub_count *= shape_x[i]
            reduce_axis_list.append(i)

    return __calcu_other_tiling_single_core(shape_x, reduce_axis_list, limit_ub_count, max_ub_count, core_num,
                                            _last_axis_index, align_count, res_align_dim, res_align_factor)


def __calcu_other_tiling_single_core(shape_x, reduce_axis_list, limit_ub_count, max_ub_count, core_num,
                                     _last_axis_index, align_count, res_align_dim, res_align_factor):
    res_size = int(max_ub_count // limit_ub_count)
    # not enough space to align gm
    if res_size < res_align_factor:
        return True

    # consider multi core
    free_size_after_block_tiling = Constant.INIT_SIZE
    for i in range(res_align_dim + 1):
        if i not in reduce_axis_list:
            free_size_after_block_tiling *= shape_x[i]
    free_size_after_block_tiling = util_common.ceil(free_size_after_block_tiling, core_num)
    # use barrier to make sure align
    cur_align_size = res_align_factor
    upper_size = min(res_size, shape_x[res_align_dim])
    # find perfect cut
    for align_size in range(res_align_factor, upper_size + 1):
        if free_size_after_block_tiling % align_size == 0:
            cur_align_size = align_size
            break
    misalign_size = shape_x[res_align_dim] % cur_align_size
    if misalign_size and misalign_size < res_align_factor:
        return True
    default_shape_x = shape_x.copy()
    shape_x[res_align_dim] //= res_align_factor
    limit_ub_count *= cur_align_size

    # consider db condition
    double_buffer_size = limit_ub_count * 2
    if double_buffer_size <= max_ub_count:
        limit_ub_count = double_buffer_size
    res_size = max_ub_count // limit_ub_count

    block_tiling_para = __calc_tiling_strategy(shape_x, reduce_axis_list, res_size, res_align_dim, core_num,
                                               _last_axis_index, align_count, cur_align_size)
    if not __check_tiling_res(shape_x, block_tiling_para):
        while core_num > 1:
            core_num -= 1
            block_tiling_para = __calc_tiling_strategy(shape_x, reduce_axis_list, res_size, res_align_dim, core_num,
                                                       _last_axis_index, align_count, cur_align_size)

    return _do_tiling_strategy(default_shape_x, reduce_axis_list, block_tiling_para)


def is_special_cases(input_x, input_gamma, input_beta, begin_norm_axis, impl_mode):
    """
    Judge whether it is a special case
    """
    is_support_device = _is_support_device()
    is_unsupported_or_single_core = _is_unsupported_or_single_core(
        input_x, input_gamma, input_beta, begin_norm_axis, impl_mode)
    is_in_white_list = _is_in_white_list(input_x, input_gamma, input_beta)

    return (is_unsupported_or_single_core or is_in_white_list) and is_support_device


def __calc_tiling_strategy(shape_x, reduce_axis_list, rest_size, res_align_dim, core_num,
                           _last_axis_index, align_count, cur_align_size):
    tiling_shape = deepcopy(shape_x)
    tiling_barrier = reduce_axis_list
    block_tiling_axes, block_factor = get_block_factor_radical(tiling_shape, tiling_barrier, core_num)
    # set barrier base on block tiling result
    tiling_shape[block_tiling_axes[0]] = block_factor
    tiling_barrier = tiling_barrier + list(range(block_tiling_axes[0]))
    tiling_barrier = tiling_barrier + block_tiling_axes[1:]
    # ub tiling
    ub_axis_idx, ub_factor = get_ub_factor(tiling_shape, tiling_barrier, rest_size)
    # check result
    if ub_axis_idx not in block_tiling_axes or len(block_tiling_axes) == 1:
        tiling_strategy = Constant.TILING_RADICAL
    else:
        tiling_shape = deepcopy(shape_x)
        tiling_barrier = reduce_axis_list
        # use fill, at last, make it successful
        block_tiling_axes, block_factor = get_block_factor_conservative(tiling_shape, tiling_barrier, core_num)

        tiling_barrier += list(range(block_tiling_axes[-1]))
        tiling_shape[block_tiling_axes[-1]] = block_factor[-1]
        # ub_tiling
        ub_axis_idx, ub_factor = get_ub_factor(tiling_shape, tiling_barrier, rest_size)
        tiling_strategy = Constant.TILING_CONSERVATIVE

    # modify cut factor as barrier last axis to ub align
    if _last_axis_index in block_tiling_axes:
        if tiling_strategy == Constant.TILING_RADICAL:
            block_factor *= align_count
        if tiling_strategy == Constant.TILING_CONSERVATIVE:
            block_factor[-1] = block_factor[-1] * align_count
    if ub_axis_idx == _last_axis_index:
        ub_factor *= align_count

    # modify cut factor as barrier some axes to gm align
    if res_align_dim in block_tiling_axes:
        if tiling_strategy == Constant.TILING_RADICAL:
            block_factor *= cur_align_size
        if tiling_strategy == Constant.TILING_CONSERVATIVE:
            block_factor[-1] = block_factor[-1] * cur_align_size
    if ub_axis_idx == res_align_dim:
        ub_factor *= cur_align_size
    ub_factor = __check_ub_factor(ub_axis_idx, block_tiling_axes, block_factor, ub_factor)
    block_tiling_para = {"axes": block_tiling_axes, "factor": block_factor,
                         "tiling_strategy": tiling_strategy, "ub_axes": ub_axis_idx, "ub_factor": ub_factor}

    return block_tiling_para


def _do_tiling_strategy(shape_x, reduce_axis_list, block_tiling_para):
    # align tiling
    res_axes = []
    leave_in_ub_axes = []
    front_ub_axis_idx = Constant.DEFAULT_INDEX
    for idx in reduce_axis_list:
        leave_in_ub_axes.append(idx)
        if idx < front_ub_axis_idx or front_ub_axis_idx == Constant.DEFAULT_INDEX:
            front_ub_axis_idx = idx
    for _, val in enumerate(shape_x):
        res_axes.append(val)
    # get params
    block_tiling_axes = block_tiling_para.get("axes")
    block_tiling_factor = block_tiling_para.get("factor")
    tiling_strategy = block_tiling_para.get("tiling_strategy")
    # make init
    if tiling_strategy == Constant.TILING_RADICAL:
        # block tiling
        block_target_axis = res_axes[block_tiling_axes[0]]
        fuse_axes_idx = block_tiling_axes[1:]
        for d_i in fuse_axes_idx:
            block_target_axis *= res_axes[d_i]
        block_target_axis = util_common.ceil(block_target_axis, block_tiling_factor)
    else:
        # block tiling
        # 'pylint: disable=unsubscriptable-object
        if len(block_tiling_axes) == 1:
            block_target_axis = util_common.ceil(res_axes[block_tiling_axes[0]], block_tiling_factor[0])
        else:
            suf_outer = util_common.ceil(res_axes[block_tiling_axes[-1]], block_tiling_factor[-1])
            block_target_axis = block_tiling_factor[0]
            block_fuse_axes_idx = block_tiling_axes[1:-1]
            for d_i in block_fuse_axes_idx:
                block_target_axis *= res_axes[d_i]
            block_target_axis *= suf_outer

    multi_core_fused_axis = block_target_axis
    if multi_core_fused_axis > 1:
        return False
    else:
        return True


def to_frac_z_axis(ori_shape, ori_axis):
    """
    judge the format is fractal NZ

    Parameters
    ----------
    ori_shape: list or tuple
        original shape of input
    ori_axis: list or tuple
        original axis of original shape to operate

    Returns
    -------
    output: list
        axis of the fractal Nz shape
    """
    frac_z_axis = list(ori_axis)
    ori_shape_len = len(ori_shape)
    axis_count = len(frac_z_axis)
    axis_negative_1 = ori_shape_len - 1
    axis_negative_2 = ori_shape_len - 2
    for i in range(axis_count):
        axis_index = (frac_z_axis[i] + ori_shape_len) % ori_shape_len
        if axis_index == axis_negative_1:
            if frac_z_axis[i] <= ori_shape_len - 2:
                frac_z_axis[i] = axis_index - 1
                frac_z_axis.append(axis_index + 2)
            else:
                frac_z_axis[i] = axis_index - 1
                frac_z_axis.append(axis_index + 1)
        elif axis_index == axis_negative_2:
            frac_z_axis[i] = axis_index + 1
            frac_z_axis.append(axis_index + 2)
        else:
            frac_z_axis[i] = axis_index
    return frac_z_axis


def __check_ub_factor(ub_axis_idx, block_tiling_axes, block_factor, ub_factor):
    """
    check ub_factor is need modify
    """
    is_need_modify_factor = ub_axis_idx in block_tiling_axes and \
        len(block_tiling_axes) == 1
    _block_factor = 1
    if is_need_modify_factor:
        if isinstance(block_factor, int):
            _block_factor = block_factor
        if isinstance(block_factor, list) and \
                len(block_factor) == 1:
            _block_factor = block_factor[0]

    if _block_factor > 1:
        while _block_factor % ub_factor == 1:
            ub_factor -= 1

    return ub_factor


def __check_tiling_res(tiling_shape, tiling_para):
    block_tiling_axes = tiling_para["axes"]
    block_factor = tiling_para["factor"]
    ub_axis_idx = tiling_para["ub_axes"]
    ub_factor = tiling_para["ub_factor"]

    is_need_modify_factor = ub_axis_idx in block_tiling_axes and \
        len(block_tiling_axes) == 1

    if is_need_modify_factor:
        block_tiling_axis = block_tiling_axes[0]
        if isinstance(block_factor, int):
            _block_factor = block_factor
        if isinstance(block_factor, list) and \
                len(block_factor) == 1:
            _block_factor = block_factor[0]
        if _block_factor > 1:
            block_tile = tiling_shape[block_tiling_axis] % _block_factor
            if _block_factor % ub_factor == 1 or \
                    block_tile % ub_factor == 1:
                return False

    return True
