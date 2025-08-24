# Copyright 2020 Huawei Technologies Co., Ltd
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
unsorted_segment_sum
"""
# pylint: disable=too-many-lines
from ..util.platform_adapter import para_check
from ..util.platform_adapter import error_manager_vector
from . import unsorted_segment
from ..util.platform_adapter import register_operator


@register_operator("UnsortedSegmentMax")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def unsorted_segment_max(x_dict,
                         segment_ids_dict,
                         num_segments_dict,
                         y_dict,
                         kernel_name="UnsortedSegmentMax"):
    """
    unsorted_segment_sum entry interface

    Parameters
    ----------
    x_dict: input params shape, dtype and range
    segment_ids_dict: segment_ids shape, dtype and range
    num_segments_dict: num_segments shape, dtype and range
    y_dict: output shape, dtype and range
    kernel_name: kernel name of UnsortedSegmentMax op

    Returns
    -------
    compile info
    """
    x_dtype = x_dict.get("dtype").lower()
    x_dtype_check_list = ("float32", "float16", "int32")
    para_check.check_dtype(x_dtype, x_dtype_check_list, param_name="x_dict")

    segment_ids_dtype = segment_ids_dict.get("dtype").lower()
    segment_ids_dtype_check_list = ("int32")
    para_check.check_dtype(segment_ids_dtype,
                           segment_ids_dtype_check_list,
                           param_name="segment_ids_dict")

    num_segments_dtype = num_segments_dict.get("dtype").lower()
    num_segments_dtype_check_list = ("int32")
    para_check.check_dtype(num_segments_dtype,
                           num_segments_dtype_check_list,
                           param_name="num_segments_dict")

    y_dtype = y_dict.get("dtype").lower()
    para_check.check_dtype(y_dtype, x_dtype_check_list, param_name="y_dict")
    if x_dtype != y_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal(
            kernel_name, "x", "y", x_dtype, y_dtype)

    unsorted_segment.unsorted_segment(
        x_dict, segment_ids_dict, num_segments_dict, y_dict, kernel_name, instruction='segment_max')
