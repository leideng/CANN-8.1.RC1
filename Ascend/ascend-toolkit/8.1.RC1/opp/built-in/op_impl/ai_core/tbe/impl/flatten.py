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
flatten
"""

from impl.util.platform_adapter import para_check
from impl import copy_only


# 'pylint: disable = invalid-name,unused-argument
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def flatten(x, y, axis=1, kernel_name="flatten"):
    """return a copy of the tensor collapsed into one dimension.

    Parameters
    ----------
    x : dict
        shape and dtype of input.
    y : dict
        shape and dtype of output.
    kernel_name : str
        kernel name, default value is "flatten"

    Returns
    -------
    None
    """
    shape = x.get("shape")
    dtype = x.get("dtype")
    dtype_lower = dtype.lower()
    check_list = ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32")

    para_check.check_shape(shape, param_name="x")
    para_check.check_dtype(dtype_lower, check_list, param_name="x")

    size = 1
    for i, _ in enumerate(shape):
        size = size * shape[i]

    shape_new = (size,)
    x.update({"shape": shape_new})
    copy_only.copy_only(x, x, kernel_name)
