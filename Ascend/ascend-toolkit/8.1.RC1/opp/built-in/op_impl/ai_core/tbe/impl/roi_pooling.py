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
roi_pooling
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl import roi_pooling_base
from impl import roi_pooling_128c0
from impl import roi_pooling_1c0_fm_l1
from impl import roi_pooling_onec0
from impl import roi_pooling_l1
from impl import roi_pooling_four_c0
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=C0103
# 'pylint: disable=C0301
# 'pylint: disable=C0111
# 'pylint: disable=C0103
# 'pylint: disable=unused-argument,no-member,no-else-return
# 'pylint: disable=too-many-instance-attributes
# 'pylint: disable=too-many-locals,too-many-lines
# 'pylint: disable=too-many-arguments,attribute-defined-outside-init

NoneType = type(None)


# 8 C0
def _get_roi_ub_cost(pooled_h, pooled_w, proposal_num_per_tiling):
    """
    get roi ub cost of 8 C0
    """
    roi_start_h_cost = pooled_h * proposal_num_per_tiling * 4
    roi_start_w_cost = pooled_w * proposal_num_per_tiling * 4
    roi_bin_h_cost = pooled_h * proposal_num_per_tiling * 4
    roi_bin_w_cost = pooled_w * proposal_num_per_tiling * 4
    roi_start_w_from0_cost = pooled_w * proposal_num_per_tiling * 4
    proposals_ub_int32_cost = 5 * proposal_num_per_tiling * 4
    roi_height_cost = proposal_num_per_tiling * 4
    roi_width_cost = proposal_num_per_tiling * 4
    const_value_cost = 64 * 4
    const_zero_cost = 64 * 4
    calced_rois_scalar = 4
    range_end_scalar = 4
    proposal_ub_validnum = 4

    return roi_start_h_cost + roi_start_w_cost + roi_bin_h_cost + \
           roi_bin_w_cost + roi_start_w_from0_cost + \
           proposals_ub_int32_cost + roi_height_cost + roi_width_cost + \
           const_value_cost + const_zero_cost + calced_rois_scalar + \
           range_end_scalar + proposal_ub_validnum


def _get_pool_ub_cost_128c0(block_num, fm_h, fm_w, fm_c0, dtype, pooled_h,
                            pooled_w):
    """
    get pooling ub cost of 8 C0
    """
    proposal_fm_data_cost = block_num * fm_h * fm_w * fm_c0 * roi_pooling_base.TYPELEN_DICT[dtype]
    proposals_ub_batchid_scalar = 4

    pooled_h_res_cost = block_num*1*fm_w * fm_c0 * roi_pooling_base.TYPELEN_DICT[dtype]
    pooled_res_cost = block_num * pooled_h * pooled_w * fm_c0 * roi_pooling_base.TYPELEN_DICT[dtype]
    scalar_roi_start_w_cost = 4
    scalar_roi_width_cost = 4
    scalar_roi_start_h_cost = 4
    scalar_roi_bin_h_cost = 4
    scalar_roi_start_w_from0_cost = 4
    scalar_roi_bin_w_cost = 4

    return proposal_fm_data_cost + proposals_ub_batchid_scalar + \
           (pooled_h_res_cost + pooled_res_cost + scalar_roi_start_w_cost + \
            scalar_roi_width_cost + scalar_roi_start_h_cost +
            scalar_roi_bin_h_cost + scalar_roi_start_w_from0_cost +
            scalar_roi_bin_w_cost)*2


# 4 C0
def _get_4c0_ub_roi_cost(pooled_h, pooled_w):
    """
    _get_4c0_ub_roi_cost
    """
    proposal_num_l1_ub_tiling = 128
    roi_start_h_cost = pooled_h * proposal_num_l1_ub_tiling * 4
    roi_start_w_cost = pooled_w * proposal_num_l1_ub_tiling * 4
    roi_bin_h_cost = pooled_h * proposal_num_l1_ub_tiling * 4
    roi_bin_w_cost = pooled_w * proposal_num_l1_ub_tiling * 4
    roi_start_w_from0_cost = pooled_w * proposal_num_l1_ub_tiling * 4
    proposals_ub_int32_cost = 5 * proposal_num_l1_ub_tiling * 4
    roi_width_cost = proposal_num_l1_ub_tiling*4

    return roi_start_h_cost + roi_start_w_cost + roi_bin_h_cost + \
           roi_bin_w_cost + roi_start_w_from0_cost + \
           proposals_ub_int32_cost + roi_width_cost


def _get_pool_ub_cost_4c0(fm_h, fm_w, fm_c0, dtype, pooled_h, pooled_w):
    """
    get pooling ub cost of 4 C0
    """
    four_c0 = 4
    align_eight = 8
    proposal_fm_data_cost = four_c0 * fm_h * fm_w * fm_c0 * roi_pooling_base.TYPELEN_DICT[dtype]

    pooled_h_res_cost = four_c0 * 1 * (fm_w + align_eight - 1) // align_eight \
                        * fm_c0 * roi_pooling_base.TYPELEN_DICT[dtype]
    pooled_res_cost = four_c0 * pooled_h * pooled_w * fm_c0 * roi_pooling_base.TYPELEN_DICT[dtype]

    return proposal_fm_data_cost + (pooled_h_res_cost + pooled_res_cost) * 2 \
           + _get_4c0_ub_roi_cost(pooled_h, pooled_w)


def _get_bin_one_ub(pooled_h, pooled_w, feature_batch, dtype):
    proposal_num_l1_ub_tiling = 128
    output_offset = feature_batch * 64 * 4
    roi_actual_num_ub = 8 * 4
    roi_height = proposal_num_l1_ub_tiling * 4
    roi_width = proposal_num_l1_ub_tiling * 4
    const_value = 4 * 64
    const_zero = 4 * 64
    bin_h_fp16 = (pooled_h + 1) * proposal_num_l1_ub_tiling * roi_pooling_base.TYPELEN_DICT[dtype]
    bin_w_fp16 = (pooled_w + 1) * proposal_num_l1_ub_tiling * roi_pooling_base.TYPELEN_DICT[dtype]
    proposals_ub = 5 * proposal_num_l1_ub_tiling * roi_pooling_base.TYPELEN_DICT[dtype]
    res = output_offset + roi_actual_num_ub + roi_height + roi_width + const_value + \
          const_zero + bin_w_fp16 + bin_h_fp16 + proposals_ub
    return res


# 1 C0
def _get_roi_onec0_ub_cost(fm_h, fm_w, fm_c0, dtype, pooled_h, pooled_w, feature_batch):
    fm_w_align = roi_pooling_base.align(fm_w, 8)
    proposal_fm_data_cost = (fm_h + 2) * fm_w_align * fm_c0 * roi_pooling_base.TYPELEN_DICT[dtype]
    pooled_h_align = roi_pooling_base.align(pooled_h, 8)
    pooled_h_res_cost = pooled_h_align * fm_w_align * fm_c0 * roi_pooling_base.TYPELEN_DICT[dtype]
    pooled_res_cost = pooled_h_align * pooled_w * fm_c0 * roi_pooling_base.TYPELEN_DICT[dtype]

    return proposal_fm_data_cost + (pooled_h_res_cost+pooled_res_cost) * 2 + \
           _get_4c0_ub_roi_cost(pooled_h, pooled_w) + \
           _get_bin_one_ub(pooled_h, pooled_w, feature_batch, dtype)


# 1 C0 and rois in L1
def _get_roi_onec0_posl1_ub_rois_cost(pooled_h, pooled_w):
    """
    _get_roi_onec0_posl1_ub_cost
    """
    proposal_num_l1_ub_tiling = 8
    roi_start_h_cost = pooled_h * proposal_num_l1_ub_tiling * 4
    roi_start_w_cost = pooled_w * proposal_num_l1_ub_tiling * 4
    roi_bin_h_cost = pooled_h * proposal_num_l1_ub_tiling * 4
    roi_bin_w_cost = pooled_w * proposal_num_l1_ub_tiling * 4
    roi_start_w_from0_cost = pooled_w * proposal_num_l1_ub_tiling * 4
    proposals_ub_int32_cost = 5 * proposal_num_l1_ub_tiling * 4
    roi_width_cost = proposal_num_l1_ub_tiling * 4

    return roi_start_h_cost + roi_start_w_cost + roi_bin_h_cost + \
           roi_bin_w_cost + roi_start_w_from0_cost + \
           proposals_ub_int32_cost + roi_width_cost


def _get_roi_onec0_posl1_ub_fm_cost(fm_h, fm_w, fm_c0, dtype, pooled_h, pooled_w):
    """
    _get_roi_onec0_posl1_ub_fm_cost
    """

    fm_w_align = roi_pooling_base.align(fm_w, 8)
    proposal_fm_data_cost = (fm_h + 2) * fm_w_align * fm_c0 * roi_pooling_base.TYPELEN_DICT[dtype]
    pooled_h_align = roi_pooling_base.align(pooled_h, 8)
    pooled_h_res_cost = pooled_h_align * fm_w_align * fm_c0 * roi_pooling_base.TYPELEN_DICT[dtype]
    pooled_res_cost = pooled_h_align * pooled_w * fm_c0 * roi_pooling_base.TYPELEN_DICT[dtype]

    return proposal_fm_data_cost + (pooled_h_res_cost+pooled_res_cost) * 2


def _get_roi_onec0_posl1_ub_cost(fm_h, fm_w, fm_c0, dtype, pooled_h, pooled_w):
    """
    get roi ub cost of one c0 posl1
    """
    return _get_roi_onec0_posl1_ub_rois_cost(pooled_h, pooled_w) + \
           _get_roi_onec0_posl1_ub_fm_cost(fm_h, fm_w, fm_c0, dtype, pooled_h, pooled_w)


def _get_roi_onec0_posl1_l1_cost(pooled_h, pooled_w, propsal_num_pertiling):
    """
    _get_roi_onec0_posl1_l1_cost
    """
    roi_start_h_cost = pooled_h * propsal_num_pertiling * 4
    roi_start_w_cost = pooled_w * propsal_num_pertiling * 4
    roi_bin_h_cost = pooled_h * propsal_num_pertiling * 4
    roi_bin_w_cost = pooled_w * propsal_num_pertiling * 4
    roi_start_w_from0_cost = pooled_w * propsal_num_pertiling * 4
    proposals_ub_int32_cost = 5 * propsal_num_pertiling * 4
    roi_width_cost = propsal_num_pertiling * 4

    return roi_start_h_cost + roi_start_w_cost + roi_bin_h_cost + \
           roi_bin_w_cost + roi_start_w_from0_cost + \
           proposals_ub_int32_cost + roi_width_cost


# 1 C0 and fm in L1
def _get_l1_cost_1c0_fm_l1(fm_h, fm_w, fm_c0, dtype):
    """
    _get_L1_cost_1C0_FML1
    """
    return fm_h * fm_w * fm_c0 * roi_pooling_base.TYPELEN_DICT[dtype]


def _get_pool_ub_cost_1c0_fm_l1(fm_h, fm_w_align, fm_c0, dtype, pooled_h,
                                pooled_w, res_pad):
    """
    _get_pool_Ub_cost_1C0_FML1
    """
    proposal_fm_data_cost = (fm_h // pooled_h + 2) * fm_w_align * fm_c0 * roi_pooling_base.TYPELEN_DICT[dtype]
    proposals_ub_batchid_scalar = 4

    pooled_h_res_cost = (pooled_h + res_pad) * fm_w_align * fm_c0
    pooled_res_cost = (pooled_h + res_pad) * pooled_w * fm_c0 * roi_pooling_base.TYPELEN_DICT[dtype]

    proposals_ub_batchid = 4
    scalar_propoal_width_cost = 4

    scalar_roi_start_h_cost = 4
    scalar_roi_start_w_cost = 4

    scalar_roi_width_cost = 4
    scalar_roi_bin_h_cost = 4

    scalar_roi_start_w_from0_cost = 4
    scalar_roi_bin_w_cost = 4

    return proposals_ub_batchid + (scalar_propoal_width_cost + \
        proposal_fm_data_cost + proposals_ub_batchid_scalar + \
        pooled_h_res_cost + pooled_res_cost + scalar_roi_start_w_cost + \
        scalar_roi_width_cost + scalar_roi_start_h_cost + \
        scalar_roi_bin_h_cost + scalar_roi_start_w_from0_cost + \
        scalar_roi_bin_w_cost) * 2


def _get_subroiclass(x_dict, pooled_h, pooled_w):
    """
    _get_subroiclass
    """
    feature_batch = x_dict.get("shape")[0]
    fm_h = x_dict.get("shape")[2]
    fm_w = x_dict.get("shape")[3]
    fm_c0 = x_dict.get("shape")[4]
    dtype = x_dict.get("dtype").lower()

    proposal_num_per_tiling = 128
    ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    l1_size = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)

    ub_cost_8c0 = _get_pool_ub_cost_128c0(8, fm_h, fm_w, fm_c0, dtype,
                                          pooled_h, pooled_w) + \
                  _get_roi_ub_cost(pooled_h, pooled_w, proposal_num_per_tiling)

    ub_cost_4c0 = _get_pool_ub_cost_4c0(fm_h, fm_w, fm_c0, dtype, pooled_h,
                                        pooled_w)

    ub_cost_onec0 = _get_roi_onec0_ub_cost(fm_h, fm_w, fm_c0, dtype, pooled_h,
                                           pooled_w, feature_batch)

    roi_onec0_posl1_ub = _get_roi_onec0_posl1_ub_cost(fm_h, fm_w, fm_c0,
                                                      dtype, pooled_h, pooled_w)
    get_roi_onec0_posl1_l1 = _get_roi_onec0_posl1_l1_cost(pooled_h, pooled_w, proposal_num_per_tiling)

    res_pad = 0 if((pooled_h % 8) == 0) else (roi_pooling_base.align(pooled_h, 8) - pooled_h)
    fm_w_align = roi_pooling_base.align(fm_w, 8)
    ub_cost_1c0_fm_l1 = _get_pool_ub_cost_1c0_fm_l1(fm_h, fm_w_align, fm_c0, dtype, pooled_h, pooled_w, res_pad) + \
                        _get_roi_ub_cost(pooled_h, pooled_w, proposal_num_per_tiling)

    if ub_size > ub_cost_8c0 and pooled_h <= 8:
        # 8c0
        return roi_pooling_128c0.RoiClass128C0()
    elif ub_size >= ub_cost_4c0 and pooled_h <= 6:
        # 4c0, pooled_h must be smaller than 6
        return roi_pooling_four_c0.RoiClass4C0()
    elif ub_size >= ub_cost_onec0:
        # 1c0
        return roi_pooling_onec0.RoiOneC0Class(0)
    elif (ub_size >= roi_onec0_posl1_ub) and (l1_size >= get_roi_onec0_posl1_l1):
        # 1c0PosL1
        return roi_pooling_onec0.RoiOneC0Class(1)

    elif l1_size > _get_l1_cost_1c0_fm_l1(fm_h, fm_w, fm_c0, dtype) and \
            ub_size > ub_cost_1c0_fm_l1:
        # 1c0FML1
        return roi_pooling_1c0_fm_l1.RoiClassOneC0FML1()
    else:
        # L1
        return roi_pooling_l1.RoiClassL1()


def _safe_check(dicts, kernel_name):
    """
    check if the inputs are legal

    Parameters
    ----------
    dicts: (x_dict, rois_dict, actual_dict, y_dict)
    kernel_name: kernel name

    Returns
    -------
    None
    """
    x_shape = dicts[0].get("shape")
    x_dtype = dicts[0].get("dtype").lower()
    rois_shape = dicts[1].get("shape")
    rois_dtype = dicts[1].get("dtype").lower()

    y_dtype = dicts[3].get("dtype").lower()
    y_shape = dicts[3].get("shape")

    tik_name_check = tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION)
    if tik_name_check in ("Ascend310", "Ascend910", "Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        para_check.check_dtype(x_dtype, ["float16"], param_name="input_x")
        para_check.check_dtype(rois_dtype, ["float16"], param_name="input_rois")
    else:
        para_check.check_dtype(x_dtype, ["float16", "float32"], param_name="input_x")
        para_check.check_dtype(rois_dtype, ["float16", "float32"], param_name="input_rois")

    if x_dtype != rois_dtype or x_dtype != y_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal("roi_pooling", "x", "rois and y",
                                                              x_dtype, rois_dtype + " and " + y_dtype)

    para_check.check_shape(x_shape, min_rank=5, max_rank=5, param_name="input_x")
    para_check.check_shape(rois_shape, min_rank=2, max_rank=3, param_name="input_rois")
    para_check.check_shape(y_shape, min_rank=5, max_rank=5, param_name="output_y")
    if len(rois_shape) == 2: # [num_rois, 5]
        roi_max_num = ((rois_shape[0] + 15) // 16) * 16
    else:
        roi_max_num = rois_shape[2]
    if roi_max_num > 6000 or roi_max_num % 16 != 0:
        error_manager_vector.raise_err_input_value_invalid("roi_pooling", "rois_shape[2]", "less than \
                                                           6000 and can be divided by 16", str(rois_shape[2]))


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_INT,
                            para_check.REQUIRED_ATTR_FLOAT, para_check.REQUIRED_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def roi_pooling(x_dict, rois_dict, actual_dict, y_dict, pooled_h, pooled_w,
                spatial_scale_h, spatial_scale_w, kernel_name="roi_pooling"):
    """
    roi pooling interface

    Parameters
    ----------
    x_dict: feature map size and data type
    rois_dict: rois_dictsize and data type
    actual_dict: actual num of rois size and data type
    y_dict: output size and data type
    pooled_h: pooled_h size
    pooled_w: pooled_w size
    spatial_scale_h: spatial scale h
    spatial_scale_w: spatial scale w
    kernel_name: kernel name of roi pooling op

    Returns
    -------
    None
    """
    _safe_check((x_dict, rois_dict, actual_dict, y_dict), kernel_name)

    roi_pooling_instance = _get_subroiclass(x_dict, pooled_h, pooled_w)
    roi_pooling_instance.init_param((pooled_h, pooled_w),
                                    (x_dict, rois_dict, actual_dict, y_dict),
                                    (spatial_scale_h, spatial_scale_w),
                                    kernel_name)
    roi_pooling_instance.roi_pooling_main()
