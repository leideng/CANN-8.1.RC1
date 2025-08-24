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
concat_offset.py
"""
from impl.util.platform_adapter import para_check
from impl.util import util_common


# 'pylint: disable=unused-argument,invalid-name,unnecessary-pass
def check_supported(concat_dim, x, y, kernel_name="concat_offset"):
    """
    if the inputs is dynamic case, and the all num of x is <= 95, will support the aicore Op
    """
    if not util_common.is_dynamic_input(x) and not util_common.is_dynamic_input([concat_dim]):
        reason = "dynamic shape is not supported"
        return False, reason

    # when input is more than 95, changed to aicpu
    # because the system only support 192 gm num for one Op
    if len(x) > 95:
        reason = "aicore only support cases that all num of x is <= 95"
        return False, reason

    return True, ""


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.DYNAMIC_INPUT,
                            para_check.DYNAMIC_OUTPUT, para_check.KERNEL_NAME)
def concat_offset(concat_dim, x, y, kernel_name="concat_offset"):
    """
    Compute the concat offset of the input tensor along `concat_dim`.

    Parameters
    ----------
    concat_dim: dict
                a number of int32, The dimension along which to concatenate,
                must be in the range [-rank(shape), rank(shape))
    x: list of dict, dict include shape and dtype, dtype must be in ('int32')
    y: list of dict, dict include shape and dtype, dtype must be in ('int32')
    kernel_name: kernel name

    Returns
    -------
    None
    """
    pass
