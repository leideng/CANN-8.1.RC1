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
resize_bilinear_v2.py
"""

import json

from impl.util import util_common
from impl.util import util_tik_auto_tune
from impl.util.platform_adapter import tbe_platform as tbe_platform_adapter


# 'pylint: disable=unused-argument,too-many-arguments,too-many-locals
def tune_space_resize_bilinear_v2(images, size, y, align_corners,
                                  half_pixel_centers, kernel_name):
    """
    get tune_param
    Parameters
    ----------
    images: dict
        the dict of input, include shape of input_tensor which layout
        only support 5HD and dtype supports 'float16', 'float32'
    size: dict
        the dict of input, the height and width of output tensor
        only support 5HD and dtype supports 'float16', 'float32'
    y: dict
        the dict of output, include shape of input_tensor which layout
        only support 5HD and dtype supports 'float16', 'float32'
    align_corners: bool
        whether align_corners
    half_pixel_centers: bool
        whether half_pixel_centers
    kernel_name: str
        cce kernel name, default value is `resize_bilinear_v2`

    Returns
    -------
    tune_param: param lists of auto tune
    """
    aicore_num = tbe_platform_adapter.get_soc_spec(tbe_platform_adapter.CORE_NUM)
    if util_common.is_unknown([images, y, size]):
        return '{}'
    param = util_tik_auto_tune.PARAM
    version = util_tik_auto_tune.VERSION
    tune_param = util_tik_auto_tune.TUNE_PARAM
    data_type = util_tik_auto_tune.DATA_TYPE
    dtype = util_tik_auto_tune.TYPE
    sub_param = util_tik_auto_tune.SUB_PARAM
    value = util_tik_auto_tune.VALUE
    par_list = util_tik_auto_tune.PARAM_LIST
    tune_timeout = util_tik_auto_tune.TUNE_TIMEOUT
    param_list = [{param: "cut_batch_c1_num", data_type: "int64", dtype: "range", value: [1, aicore_num]},
                  {param: "cut_height_num", data_type: "int64", dtype: "range", value: [1, aicore_num]},
                  {param: "cut_width_num", data_type: "int64", dtype: "range", value: [1, aicore_num]}]
    tune_param = {version: "1.0.0",
                  tune_timeout: 600,
                  tune_param: [{param: "tiling_key",
                                data_type: "int64",
                                dtype: "list",
                                sub_param: [{value: 999999},
                                            {value: 100110, par_list: param_list},
                                            {value: 100000, par_list: param_list}]}]}
    return json.dumps(tune_param)


# 'pylint: disable=unused-argument
def tune_param_check_supported_resize_bilinear_v2(images, size, y, align_corners, half_pixel_centers, kernel_name,
                                                  tune_param):
    """
    check tune_param
    Parameters
    ----------
    images: dict
        the dict of input, include shape of input_tensor which layout
        only support 5HD and dtype supports 'float16', 'float32'
    size: dict
        the dict of input, the height and width of output tensor
        only support 5HD and dtype supports 'float16', 'float32'
    y: dict
        the dict of output, include shape of input_tensor which layout
        only support 5HD and dtype supports 'float16', 'float32'
    align_corners: bool
        whether align_corners
    half_pixel_centers: bool
        whether half_pixel_centers
    kernel_name: str
        cce kernel name, default value is `resize_bilinear_v2`
    tune_param: param list of auto tune

    Returns
    -------
    check result of tune_param
    """
    if tune_param is None:
        return False
    aicore_num = tbe_platform_adapter.get_soc_spec(tbe_platform_adapter.CORE_NUM)
    tiling_mode_list = [999999, 100110, 100000]
    tune_param_dict = json.loads(tune_param)
    tiling_mode = tune_param_dict["tune_param"]["tiling_key"]
    if tiling_mode not in tiling_mode_list:
        return False
    if tiling_mode != 999999:
        cut_batch_c1_num = tune_param_dict["tune_param"]["cut_batch_c1_num"]
        cut_height_num = tune_param_dict["tune_param"]["cut_height_num"]
        cut_width_num = tune_param_dict["tune_param"]["cut_width_num"]
        if cut_batch_c1_num * cut_height_num * cut_width_num > aicore_num or \
           cut_batch_c1_num * cut_height_num * cut_width_num < aicore_num // 2:
            return False
    return True
