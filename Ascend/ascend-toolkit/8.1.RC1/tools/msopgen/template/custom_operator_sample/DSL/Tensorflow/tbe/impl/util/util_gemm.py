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
util_gemm
"""
from .platform_adapter import error_manager_cube


RANGE_MAX = 2147483647 # 2**31-1
BATCH_GEAR = [0, 1, 3, 7, 15, 31, RANGE_MAX]
SHAPE_GEAR_MATMUL_ND = [0, 16*3, 16*7, 16*15, 16*31, 16*63, 16*127, 16*191, 16*255,
                        16*511, 16*767, 16*1023, RANGE_MAX] # for fp16
SHAPE_GEAR_MATMUL_NZ = [0, 3, 7, 15, 31, 63, 127, 191, 255, 511, 767, 1023, RANGE_MAX]
DYNAMIC_DIM_VAL = -1


def _get_shape_gear(dim, shape_gear):
    pos = 1
    while(pos < len(shape_gear) and shape_gear[pos] < dim):
        pos += 1
    return (shape_gear[pos - 1] + 1, shape_gear[pos])

def cal_gemm_shape_range(shape, ori_format):
    shape_range = []
    shape_len = len(shape)
    if ori_format == "ND":
        # shape like (batch1, ..., batchn, m, k)
        # process batch dim
        for i in range(0, shape_len - 2):
            if shape[i] > RANGE_MAX:
                error_manager_cube.raise_err_one_para(
                    "E62306", "",
                    "Invalid generalize range, shape of tensor exceed 2147483647")
            shape_range.append(_get_shape_gear(shape[i], BATCH_GEAR))

        # process m/k/n dim and bias
        for i in range(-min(shape_len, 2), 0):
            if shape[i] > RANGE_MAX:
                error_manager_cube.raise_err_one_para(
                    "E62306", "",
                    "Invalid generalize range, shape of tensor exceed 2147483647")
            shape_range.append(_get_shape_gear(shape[i], SHAPE_GEAR_MATMUL_ND))
    elif ori_format in ("FRACTAL_NZ", "FRACTAL_Z"):
        # shape like (batch1, ..., batchn, m1, k1, C0, C0)
        # process batch dim
        for i in range(0, shape_len - 4):
            if shape[i] > RANGE_MAX:
                error_manager_cube.raise_err_one_para(
                    "E62306", "",
                    "Invalid generalize range, shape of tensor exceed 2147483647")
            shape_range.append(_get_shape_gear(shape[i], BATCH_GEAR))

        # process m1/k1/n1 dim
        for i in range(-4, -2):
            if shape[i] > RANGE_MAX:
                error_manager_cube.raise_err_one_para(
                    "E62306", "",
                    "Invalid generalize range, shape of tensor exceed 2147483647")
            shape_range.append(_get_shape_gear(shape[i], SHAPE_GEAR_MATMUL_NZ))

        # process n0/m0/k0 dim
        const_dim1 = shape[-1]
        const_dim2 = shape[-2]
        shape_range.extend([(const_dim2, const_dim2), (const_dim1, const_dim1)])
    else:
        error_manager_cube.raise_err_one_para(
                    "E62306", "",
                    "Invalid Matmul/BacthMatmul format, only support FRACTAL_NZ or FRACTAL_Z")

    return tuple(shape_range)

def _generate_unknown_shape_gemm(shape):
    return [DYNAMIC_DIM_VAL for i in shape]

def generalize_input_keep_rank_gemm(input_dict) :
    if input_dict["ori_format"] in ("NHWC", "ND"):
        input_dict["ori_shape"] = _generate_unknown_shape_gemm(input_dict["ori_shape"])
    elif input_dict["ori_format"] in ("FRACTAL_NZ", "FRACTAL_Z"):
        x_old_1, x_old_2 = input_dict["ori_shape"][-1], input_dict["ori_shape"][-2]
        input_dict["ori_shape"] = _generate_unknown_shape_gemm(input_dict["ori_shape"])
        input_dict["ori_shape"][-1] = x_old_1
        input_dict["ori_shape"][-2] = x_old_2