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
reduce_min
"""
from impl.util.platform_adapter import para_check
from impl import reduce_min_d


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            (para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_INT),
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def reduce_min(input_min, output_min, axis, keep_dims=None, kernel_name="reduce_min"):
    """
    Reduce a tensor on a certain axis based on min

    Parameters:
    ----------
    input_min: dict
        dict of input, which contains shape and dtype
    output_min: dict
        dict of output, which contains shape and dtype
    axis: int or None
        The dimensions to reduce. If None (the default), reduces all dimensions.
        Must be in the range (-rank(input_tensor), rank(input_tensor))
    keep_dims: True or False
        if true, retains reduced dimensions with length 1,
        default value is None
    kernel_name: str
        cce kernel name, default value is "reduce_min"

    Returns
    -------
    None
    """
    return reduce_min_d.reduce_min_d(input_min, output_min, axis, keep_dims, kernel_name)
