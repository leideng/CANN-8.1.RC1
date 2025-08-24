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
dynamic arg_min
"""
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.util_common import is_unknown_rank_input
from impl.util.util_common import is_dynamic_input
from .arg_common import ArgCommon


class ArgMin(ArgCommon):
    def __init__(self, dtype_x, dtype_y, is_dynamic, kernel_name):
        super().__init__(True, dtype_x, dtype_y, is_dynamic, kernel_name)


@register_operator("ArgMin")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def arg_min(x, dimension, y, kernel_name="arg_min"):
    """
    Generate arg_min operator use arg_min

    Parameters
    ----------
    x: dict
        data of input, support "float16", "float32", "bfloat16", "int64".
    dimension: dict
        dimension input.
    y: dict
        index of output.
    kernel_name: str
        kernel name, default value is "arg_min"

    Returns
    -------
    tik_instance
    """
    shape_x = x.get("shape")
    dtype_x = x.get("dtype")
    dtype_y = y.get("dtype")
    para_check.check_shape(shape_x, param_name="x")
    check_list = ("float16", "float32", "bfloat16", "int64", "int32")
    para_check.check_dtype(dtype_x.lower(), check_list, param_name="x")
    check_list_y = ("int64", "int32")
    para_check.check_dtype(dtype_y.lower(), check_list_y, param_name="y")
    is_dynamic = is_unknown_rank_input(x) or is_dynamic_input(x)
    arg_min_op = ArgMin(dtype_x, dtype_y, is_dynamic, kernel_name)

    tik_instance = arg_min_op.get_tik_instance()
    return tik_instance
