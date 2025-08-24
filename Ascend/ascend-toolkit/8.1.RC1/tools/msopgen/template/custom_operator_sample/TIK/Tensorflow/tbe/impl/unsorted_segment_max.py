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
unsorted_segment_max
"""
# pylint: disable=too-many-lines
# pylint: disable=unused-import
# pylint: disable=PY001
from .unsorted_segment_min import op_select_format
from .unsorted_segment_min import check_supported

# pylint: disable=unused-argument
def unsorted_segment_max(x_dict, segment_ids_dict, num_segments_dict, y_dict,
                         kernel_name="UnsortedSegmentMax"):
    """
    unsorted_segment_max entry interface

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
    pass
