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
dynamic space_to_depth
"""
from .transpose import Transpose
from ..util.platform_adapter import tik
from ..util.platform_adapter import tbe_platform
from ..util.platform_adapter import para_check
from ..util.platform_adapter import register_operator
from ..util.platform_adapter import tbe_context

CORE_NUM = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
UB_SIZE = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
BLOCK_SIZE = 32
MAX_INT64_VALUE = 2**64 - 1
TILING_MAX_SIZE_GM = 2048  # 16KB


# pylint: disable=invalid-name,unused-argument,too-many-locals,too-many-arguments,redefined-builtin,protected-access
@register_operator("SpaceToDepth")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_INT, para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def space_to_depth(x, filter, y, block_size, data_format="NHWC", kernel_name="space_to_depth"):
    """
    the main function of space_to_depth

    Parameters
    ----------
    x: dict
        dict with keys(shape, dtype) of input
    y: dict
        dict with keys(shape, dtype) of output
    block_size: int
        the size of the spatial block
    data_format: str
        data format, default value is "NHWC"
    kernel_name: str
        kernel name, default value is "space_to_depth"

    Returns
    -------
    tik_instance: tik_instance
    """
    input_shape = x.get("shape")
    input_dtype = x.get("dtype").lower()
    para_check.check_shape(input_shape, param_name="x")
    check_list = ("int8", "int16", "int32", "uint8", "uint16", "uint32", "uint64", "int64", "float16", "float32")
    para_check.check_dtype(input_dtype, check_list, param_name="x")

    # run tick
    tik_inst = tik.Tik()
    data_in = tik_inst.Tensor(input_dtype, (MAX_INT64_VALUE,), tik.scope_gm, "data_in")
    data_out = tik_inst.Tensor(input_dtype, (MAX_INT64_VALUE,), tik.scope_gm, "data_out")
    data_workspace = tik_inst.Tensor(input_dtype, (1024,), tik.scope_gm, "data_workspace", is_workspace=True)
    data_tiling = tik_inst.Tensor("int64", (TILING_MAX_SIZE_GM,), tik.scope_gm, "data_tiling")
    tensor_list = [data_in, None, data_out, data_workspace, data_tiling]
    obj = Transpose(tik_inst, input_dtype, tensor_list, kernel_name)
    obj.compute_tiling()

    tbe_context.get_context().add_compile_info("vars", {
        "ub_size": UB_SIZE // BLOCK_SIZE,
        "core_num": CORE_NUM,
        "dtype": input_dtype,
        "block_size": block_size,
    })
    # this "global_variable_link" flag suggest ccec.py do link without "-r" option
    # which will result in global variable in cce file with wrong address
    tbe_context.get_context().add_compile_info("global_variable_link", True)

    obj.tik_inst.BuildCCE(kernel_name=obj.kernel_name,
                          inputs=[obj.data_in],
                          outputs=[obj.data_out],
                          flowtable=[obj.data_tiling])

    return obj.tik_inst
