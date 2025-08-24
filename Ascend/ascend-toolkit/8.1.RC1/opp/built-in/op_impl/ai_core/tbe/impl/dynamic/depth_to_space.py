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
dynamic depth_to_space
"""
import math
import copy
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
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info
from impl.util.util_common import is_unknown_rank_input
from impl.util.util_common import is_dynamic_input


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    MAX_INT64_VALUE = 2**64 - 1


# 'pylint: disable=invalid-name,unused-argument,too-many-arguments
def get_op_support_info(x, y, block_size, data_format='NHWC', kernel_name="depth_to_space"):
    """
    get_op_support_info
    """
    format_x = x.get("format").upper()
    if format_x == "NHWC":
        axis_split_matrix = [[SplitInput([0, [0], [-1], [-1]]), SplitOutput([0, [0]])],
                             [SplitInput([0, [1], [-1], [-1]]), SplitOutput([0, [1]])],
                             [SplitInput([0, [2], [-1], [-1]]), SplitOutput([0, [2]])]]
        axis_reduce_list = None

    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


def depth_to_space_tik(x, y, block_size, mode, data_format, kernel_name):
    """
    depth_to_space interface for tik
    """
    # run tick
    core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    block_size_1 = 32
    tiling_max_size_gm = 2048  # 16KB
    input_dtype = x.get("dtype").lower()
    if input_dtype == "bfloat16":
        input_dtype = "float16"

    tik_inst = tik.Tik()
    data_in = tik_inst.Tensor(input_dtype, (Constant.MAX_INT64_VALUE,), tik.scope_gm, "data_in")
    data_out = tik_inst.Tensor(input_dtype, (Constant.MAX_INT64_VALUE,), tik.scope_gm, "data_out")
    data_workspace = tik_inst.Tensor(input_dtype, (1024,), tik.scope_gm, "data_workspace", is_workspace=True)
    data_tiling = tik_inst.Tensor("int64", (tiling_max_size_gm,), tik.scope_gm, "data_tiling")
    tensor_list = [data_in, None, data_out, data_workspace, data_tiling]
    obj = Transpose(tik_inst, input_dtype, tensor_list, kernel_name)
    obj.compute_tiling()

    tbe_context.get_context().add_compile_info("vars", {
        "ub_size": ub_size // block_size_1,
        "core_num": core_num,
        "dtype": input_dtype,
        "mode": mode,
    })
    tbe_context.get_context().add_compile_info("is_tik", True)
    # this "global_variable_link" flag suggest ccec.py do link without "-r" option
    # which will result in global variable in cce file with wrong address
    tbe_context.get_context().add_compile_info("global_variable_link", True)
    opt_config = {"enable_const_fold": True}
    obj.tik_inst.BuildCCE(kernel_name=obj.kernel_name,
                          inputs=[obj.data_in],
                          outputs=[obj.data_out],
                          flowtable=[obj.data_tiling],
                          config=opt_config)

    return obj.tik_inst


def input_parameters_compute(x, blockSize, mode):
    '''
    compute input shape and shape range
    '''
    inShape = x.get("shape")
    x_format = x.get("format")
    x_range = x.get("range")
    if x_format == "NHWC" and mode == "DCR":
        x["shape"] = list(inShape)
        x["shape"][0] = inShape[0] * inShape[1] if inShape[0] != -1 and inShape[1] != -1 else -1
        x["shape"][1] = inShape[2]
        x["shape"][2] = blockSize
        x["shape"][3] = int(inShape[3] / blockSize) if inShape[3] != -1 else -1
        x["range"] = list(x_range)
        x["range"][0] = (x_range[0][0]*x_range[1][0], x_range[0][1]*x_range[1][1]) if x_range[0][1] \
                        and x_range[1][1] else (x_range[0][0]*x_range[1][0], None)
        x["range"][1] = x_range[2]
        x["range"][2] = (blockSize, blockSize)
        x["range"][3] = (math.ceil(x_range[3][0]/blockSize), int(x_range[3][1]/blockSize)) \
                        if x["range"][3][1] else (math.ceil(x_range[3][0]/blockSize), None)
        perm = [0, 2, 1, 3]
    elif x_format == "NHWC" and mode == "CRD":
        x["shape"] = (inShape[0], inShape[1], inShape[2], int(inShape[3]/blockSize/blockSize),
                      blockSize, blockSize) if inShape[3] != -1 else (inShape[0], inShape[1],
                      inShape[2], inShape[3], blockSize, blockSize)
        x["range"] = (x_range[0], x_range[1], x_range[2], (math.ceil(x_range[3][0]/blockSize/blockSize),
                      int(x_range[3][1]/blockSize/blockSize)), (blockSize, blockSize),
                      (blockSize, blockSize)) if x_range[3][1] else (x_range[0], x_range[1], x_range[2],
                      (math.ceil(x_range[3][0]/blockSize/blockSize), None), (blockSize, blockSize),
                      (blockSize, blockSize))
        perm = [0, 1, 4, 2, 5, 3]
    elif x_format == "NCHW" and mode == "DCR":
        x["shape"] = (inShape[0], blockSize, blockSize, int(inShape[1]/blockSize/blockSize),
                      inShape[2], inShape[3]) if inShape[1] != -1 else (inShape[0], blockSize, blockSize,
                      inShape[1], inShape[2], inShape[3])
        x["range"] = (x_range[0], (blockSize, blockSize), (blockSize, blockSize),
                      (math.ceil(x_range[1][0]/blockSize/blockSize), int(x_range[1][1]/blockSize/blockSize)),
                      x_range[2], x_range[3]) if x_range[1][1] else (x_range[0], (blockSize, blockSize),
                      (blockSize, blockSize), (math.ceil(x_range[1][0]/blockSize/blockSize), None),
                      x_range[2], x_range[3])
        perm = [0, 3, 4, 1, 5, 2]
    elif x_format == "NCHW" and mode == "CRD":
        x["shape"] = (inShape[0], int(inShape[1]/blockSize/blockSize), blockSize, blockSize,
                      inShape[2], inShape[3]) if inShape[1] != -1 else (inShape[0], inShape[1],
                      blockSize, blockSize, inShape[2], inShape[3])
        x["range"] = (x_range[0], (math.ceil(x_range[1][0]/blockSize/blockSize),
                      int(x_range[1][1]/blockSize/blockSize)), (blockSize, blockSize), (blockSize, blockSize),
                      x_range[2], x_range[3]) if x_range[1][1] else (x_range[0],
                      (math.ceil(x_range[1][0]/blockSize/blockSize), None), (blockSize, blockSize),
                      (blockSize, blockSize), x_range[2], x_range[3])
        perm = [0, 1, 4, 2, 5, 3]
    x["format"] = "ND"
    return x, perm


@register_operator_compute("DepthToSpace", op_mode="dynamic", support_fusion=True)
def depth_to_space_compute(x, perm, y, kernel_name):
    """
    algorithm: depth_to_space
    Rearranges data from depth into blocks of spatial data.

    Parameters
    ----------
    x: dict
        dict with keys(shape, dtype) of input
    y: dict
        dict with keys(shape, dtype) of output
    block_size: int
        the size of the spatial block
    mode: str
        mode default value is DCR, for onnx
    data_format: str
        data format, default value is "NHWC"
    kernel_name: str
        kernel name, default value is "depth_to_space"

    Returns
    -------
    res : placeholder and res
    """
    res = tbe.transpose(x, perm)
    return res


def depth_to_space_dsl(x, y, block_size, mode, data_format, kernel_name):
    """
    depth_to_space interface for dsl
    """
    x = replace_invalid_range(x)
    y = replace_invalid_range(y)
    x_dtype = x.get("dtype").lower()
    x, perm = input_parameters_compute(x, block_size, mode)
    extra_params = {"axes": perm}
    ins = classify([x], "transpose", extra_params)
    schedules, tensors = [], []
    for (input_x, perm) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([input_x], "transpose")
            data_x = tvm.placeholder(x_shape, name="data_x", dtype=x_dtype)
            res = depth_to_space_compute(data_x, perm, y, kernel_name)
            tensors.append([data_x, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
    tbe_context.get_context().add_compile_info("vars", {"mode": mode})


def replace_invalid_range(_input):
    # This op not support 0 in shape.
    if not is_unknown_rank_input(_input) and not is_dynamic_input(_input):
        return _input

    new_input = copy.deepcopy(_input)
    range_list = new_input.get("range", None)
    if range_list:
        new_range = [[1, value[1]] if value[0] == 0 else list(value) for value in range_list]
        new_input.update({'range': new_range})
    return new_input


# 'pylint: disable=invalid-name,unused-argument,too-many-locals,protected-access
@register_operator("DepthToSpace")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def depth_to_space(x, y, block_size, mode='DCR', data_format='NHWC', kernel_name="depth_to_space"):
    """
    the main function of depth_to_space

    Parameters
    ----------
    x: dict
        dict with keys(shape, dtype) of input
    y: dict
        dict with keys(shape, dtype) of output
    block_size: int
        the size of the spatial block
    mode: str
        mode default value is DCR, for onnx
    data_format: str
        data format, default value is "NHWC"
    kernel_name: str
        kernel name, default value is "depth_to_space"

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
    # check mode
    if mode not in ('DCR', 'CRD'):
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "mode", "DCR, CRD", mode)

    if tbe_platform.api_check_support("tbe.dsl.vexp", "float32") and not is_unknown_rank_input(x):
        depth_to_space_dsl(x, y, block_size, mode, data_format, kernel_name)
    else:
        depth_to_space_tik(x, y, block_size, mode, data_format, kernel_name)
