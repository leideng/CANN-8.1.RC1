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
util_conv3d
"""
from .platform_adapter import error_manager_util

BIAS_LENGTH = 1

def transform_shape_with_exception(src_format, to_format, ori_shape,
                                   format_white_list, attr_name):

    res = transform_shape_with_format(src_format, to_format, ori_shape,
                                      format_white_list)
    if res is None:
        dict_args = {
            'errCode': 'E60008',
            'param_name': attr_name,
            'expected_format_list': ",".join(format_white_list),
            'format': src_format if src_format not in format_white_list else to_format
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))
    return res

def transform_shape_with_format(src_format, to_format, ori_shape, format_white_list):
    # input format is not expected
    if ((src_format not in format_white_list) or
        (to_format not in format_white_list)):
        return None
    # need not to transform
    if src_format == to_format:
        return list(ori_shape)
    res_shape = [1 for _ in range(len(to_format))]
    for i in range(len(to_format)):
        for j in range(len(src_format)):
            if to_format[i] == src_format[j]:
                res_shape[i] = ori_shape[j]
                break
    return res_shape

def check_bias(bias, res_dtype):
    """
    algorithm: Check the input params of bias

    Parameters
    ----------

    bias: A dict with keys(shape and dtype) or None
        input bias tensor

    res_dtype: The dtype of output

    Returns
    -------
    None
    """
    bias_shape = bias.get("ori_shape")
    if len(bias_shape) != BIAS_LENGTH:
        dict_args = {
            'errCode': 'E60006',
            'param_name': 'bias',
            'expected_length': '1',
            'length': '{}'.format(len(bias_shape))
        }
        raise RuntimeError(dict_args,
                            error_manager_util.get_error_message(dict_args))
    bias_dtype = bias.get("dtype").lower()
    if bias_dtype != res_dtype:
        dict_args = {
            'errCode': 'E65002',
            'param_1': 'bias_dtype',
            'param_2': 'res_dtype'
        }
        raise RuntimeError(dict_args,
                            error_manager_util.get_error_message(dict_args))
