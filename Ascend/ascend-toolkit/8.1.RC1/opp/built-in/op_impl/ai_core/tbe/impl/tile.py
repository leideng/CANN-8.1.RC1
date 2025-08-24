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
tile_d
"""
from te.utils import para_check


# 'pylint: disable=unnecessary-pass
# 'pylint: disable=unused-argument
def check_supported(input_x, input_m, output_x, kernel_name="tile"):
    """
    verify the types of cast supported by tbe
    """
    input_x_shape = list(input_x.get("ori_shape"))
    # currently , ai_core tile operator only support 5 or less dimension input
    if len(input_x_shape) > 5:
        reason = "ai_core tile operator only support 5 or less dimension input"
        return False, reason
    return True, ""


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def tile(input_x, multiples, output_x, kernel_name="tile"):
    """algorithm: tile.
    The tile in tensorflow can multiple the shape of the given tensor.
    For example, tiling [a b c d] by [2] produces [a b c d a b c d].
    The tile op in TBE is different from tf.tile, tile of TBE use broadcast
    api, and only support that at least an axis in shape is 1.The '1' axis
    is to be multipled.
    For example, if shape = [51, 1] and multiples = [1, 77], after computation,
    the output shape will be [51, 77].
    Abnormal condition:
    1. The length of shape must be equal to or less than the shape of multiples.
    2. The type of kernel_name is not string.
    3. The shape is neither list nor tuple.
    4. The dtype is not float32, float16, or int32.
    5. All of the axises of the multiples is 1.

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    multiples : list or tuple.
        Number of the axis replicates.
    output_x: dict
        dict of output.
    kernel_name : str.
        kernel name, default value is "tile_d".

    Returns
    -------
    None
    """
    pass
