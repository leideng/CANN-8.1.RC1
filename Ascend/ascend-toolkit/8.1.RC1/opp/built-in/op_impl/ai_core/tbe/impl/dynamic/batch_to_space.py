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
dynamic batch_to_space
"""
from impl.dynamic.batch_to_space_nd import BatchToSpaceND
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info


# 'pylint: disable = unused-argument,too-many-locals,invalid-name
def get_op_support_info(x, crops, y, block_size, kernel_name="batch_to_space"):
    """get op support info
    """
    format_x = x.get("format").upper()
    if format_x == "NC1HWC0":
        axis_split_matrix = [[SplitInput([0, [1], [-1], [-1]]), SplitOutput([0, [1]])]]
        axis_reduce_list = None

    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


# 'pylint: disable=invalid-name,unused-argument
@register_operator("BatchToSpace")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def batch_to_space(x, crops, y, block_size, kernel_name="batch_to_space"):
    """BatchToSpace for tensor.

    Parameters
    ----------
    x: dict
        the dict of input tensor.
    crops: dict
        the dict of crops tensor.
    y: dict
        the dict of output tensor.
    block_size: int
        the size of block.
    kernel_name: str
        cce kernel name, default value is "batch_to_space".

    Returns
    -------
    None.
    """
    # get input shape, format and dtype
    input_shape = x.get("shape")
    input_dtype = x.get("dtype").lower()
    input_format = x.get("format")

    # check input shape, format and dtype
    para_check.check_shape(input_shape, param_name="x")
    para_check.check_dtype(input_dtype, ("float16", "float32", "bfloat16"), param_name="x")
    if input_format not in ("NC1HWC0",):
        error_manager_vector.raise_err_input_format_invalid(kernel_name, "x", "NC1HWC0", input_format)

    # run tick
    obj = BatchToSpaceND(input_dtype, block_size, kernel_name)
    obj.batch_to_space_nd_compute_tiling()
    opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}

    tbe_context.get_context().add_compile_info("vars", {
        "ub_ele": obj.ub_ele,
        "core_num": obj.core_num,
    })
    obj.tik_instance.BuildCCE(kernel_name=obj.kernel_name,
                              inputs=[obj.input_gm, obj.crops_gm],
                              outputs=[obj.output_gm],
                              flowtable=[obj.tiling_gm],
                              config=opt_config)

    return obj.tik_instance
