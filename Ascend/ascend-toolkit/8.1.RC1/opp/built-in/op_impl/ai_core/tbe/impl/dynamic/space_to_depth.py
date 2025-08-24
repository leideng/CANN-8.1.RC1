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
import math
from impl.dynamic.transpose import Transpose
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_common import is_unknown_rank_input


def check_supported(x, filter, y, block_size, data_format="NHWC", kernel_name="space_to_depth"):
    """
    check is support dynamic or cube
    """
    if filter is not None:
        return False, "filter is not none, not support"
    return True


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant
    """
    BLOCK_SIZE = 32
    MAX_INT64_VALUE = 2**64 - 1
    TILING_MAX_SIZE_GM = 2048  # 16KB


def space_to_depth_tik(x, y, block_size, data_format, kernel_name):
    '''
    space_to_depth interface for tik
    '''
    # run tick
    input_dtype = x.get("dtype").lower()
    if input_dtype == "bfloat16":
        input_dtype = "float16"

    tik_inst = tik.Tik()
    data_in = tik_inst.Tensor(input_dtype, (Constant.MAX_INT64_VALUE,), tik.scope_gm, "data_in")
    data_out = tik_inst.Tensor(input_dtype, (Constant.MAX_INT64_VALUE,), tik.scope_gm, "data_out")
    data_workspace = tik_inst.Tensor(input_dtype, (1024,), tik.scope_gm, "data_workspace", is_workspace=True)
    data_tiling = tik_inst.Tensor("int64", (Constant.TILING_MAX_SIZE_GM,), tik.scope_gm, "data_tiling")
    tensor_list = [data_in, None, data_out, data_workspace, data_tiling]
    obj = Transpose(tik_inst, input_dtype, tensor_list, kernel_name)
    obj.compute_tiling()

    tbe_context.get_context().add_compile_info("is_tik", True)
    tbe_context.get_context().add_compile_info("vars", {
        "ub_size": tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) // Constant.BLOCK_SIZE,
        "core_num": tbe_platform.get_soc_spec(tbe_platform.CORE_NUM),
        "dtype": input_dtype,
    })
    # this "global_variable_link" flag suggest ccec.py do link without "-r" option
    # which will result in global variable in cce file with wrong address
    tbe_context.get_context().add_compile_info("global_variable_link", True)

    obj.tik_inst.BuildCCE(kernel_name=obj.kernel_name,
                          inputs=[obj.data_in],
                          outputs=[obj.data_out],
                          flowtable=[obj.data_tiling])

    return obj.tik_inst


def input_parameters_compute(x, block_size):
    '''
    compute input shape and shape range
    '''
    x_shape = x.get("shape")
    x_range = x.get("range")
    x["shape"] = list(x_shape)
    x["shape"][0] = -1 if x_shape[0] == -1 or x_shape[1] == -1 else int(x_shape[0] * x_shape[1] / block_size)
    x["shape"][1] = block_size
    x["shape"][2] = -1 if x_shape[2] == -1 else int(x_shape[2] / block_size)
    x["shape"][3] = -1 if x_shape[3] == -1 else int(x_shape[3] * block_size)
    x["range"] = list(x_range)
    x["range"][0] = (math.ceil(x_range[0][0]*x_range[1][0]/block_size),
                     int(x_range[0][1]*x_range[1][1]/block_size)) if x_range[0][1] and x_range[1][1] \
                     else (math.ceil(x_range[0][0]*x_range[1][0]/block_size), None)
    x["range"][1] = (block_size, block_size)
    x["range"][2] = (math.ceil(x_range[2][0]/block_size), int(x_range[2][1]/block_size)) \
                     if x_range[2][1] else (math.ceil(x_range[2][0]/block_size), None)
    x["range"][3] = (x_range[3][0]*block_size, x_range[3][1]*block_size) if x_range[3][1] \
                     else (x_range[3][0]*block_size, None)
    x["range"] = tuple(x["range"])
    perm = [0, 2, 1, 3]
    return x, perm


@register_operator_compute("SpaceToDepth", op_mode="dynamic", support_fusion=True)
def space_to_depth_compute(x, perm, y, kernel_name):
    """
    algorithm: space_to_depth
    Rearranges blocks of spatial data, into depth.

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
    res : placeholder and res
    """
    res = tbe.transpose(x, perm)
    return res


def space_to_depth_dsl(x, y, block_size, data_format, kernel_name):
    """
    space_to_depth interface for dsl
    """
    x_dtype = x.get("dtype").lower()
    x, perm = input_parameters_compute(x, block_size)
    extra_params = {"axes": perm}
    ins = classify([x], "transpose", extra_params)
    schedules, tensors = [], []
    for (input_x, perm) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([input_x], "transpose")
            data_x = tvm.placeholder(x_shape, name="data_x", dtype=x_dtype)
            res = space_to_depth_compute(data_x, perm, y, kernel_name)
            tensors.append([data_x, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)


# 'pylint: disable=invalid-name,unused-argument,too-many-locals,too-many-arguments,redefined-builtin,protected-access
@register_operator("SpaceToDepth")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
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
    check_list = ("int8", "int16", "int32", "uint8", "uint16", "uint32", "uint64", "int64",
                  "float16", "float32", "bfloat16")
    para_check.check_dtype(input_dtype, check_list, param_name="x")

    if tbe_platform.api_check_support("tbe.dsl.vexp", "float32") and not is_unknown_rank_input(x):
        space_to_depth_dsl(x, y, block_size, data_format, kernel_name)
    else:
        space_to_depth_tik(x, y, block_size, data_format, kernel_name)
