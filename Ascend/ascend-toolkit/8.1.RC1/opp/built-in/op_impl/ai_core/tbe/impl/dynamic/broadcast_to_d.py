# Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
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
broadcast_to_d
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_common import gen_range


# 'pylint: disable=locally-disabled,too-many-arguments
# 'pylint: disable=unused-argument,invalid-name
@register_operator_compute("BroadcastToD", op_mode="dynamic", support_fusion=False)
def broadcast_to_d_compute(x, y, shape, kernel_name='broadcast_to_d'):
    """
    Process broadcast_to operator.

    Parameters:
    ----------
    x : the input tensor.

    y : the dict of output.

    shape : the desired output shape.

    kernel_name : cce kernel name, default value is "broadcast_to_d".

    Returns:
    -------
    output_tensor : tensor after broadcast_to.
    """
    src_dtype = x.dtype.lower()
    if src_dtype == "int8" or src_dtype == "uint8":
        x = tbe.cast_to(x, "float16")
    res = tbe.broadcast(x, shape)
    if src_dtype == "int8" or src_dtype == "uint8":
        res = tbe.cast_to(res, src_dtype)

    return res


@register_operator("BroadcastToD")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.KERNEL_NAME)
def broadcast_to_d(x, y, shape, kernel_name="broadcast_to_d"):
    """
    Broadcast an array for a compatible shape.

    Parameters:
    ----------
    x : the dict of input. support data type: float32, float16, int8, uint8, int32, bfloat16.

    y : the dict of output.

    shape : shape of output tensor.

    kernel_name : cce kernel name, default value is "broadcast_to_d".

    Returns:
    -------
    None
    """
    inp_dtype = x.get('dtype').lower()
    inp_dtype = "int8" if inp_dtype == "bool" else inp_dtype
    x_shape = list(x.get('shape'))
    check_list = ('float32', 'bfloat16', 'float16', 'int8', 'uint8', 'int32')
    para_check.check_dtype(inp_dtype, check_list, param_name="x")

    if len(shape) == 0:
        shape = (1, )

    bro_shape = [1] * (len(shape) - len(x_shape)) + x_shape
    bro_shape_range = gen_range(bro_shape)

    x["shape"] = bro_shape
    x["range"] = bro_shape_range
    shape_align = [bro_shape[index] if dim_value == -1 else dim_value for index, dim_value in enumerate(shape)]

    extra_params = {"disable_optimization": True}
    ins = classify([x], OpPatternMode.ELEWISE_WITH_BROADCAST, extra_params)
    schedules, tensors = [], []
    for (_x, ) in ins:
        with tbe.compute():
            shape_x = shape_util.variable_shape([_x])[0]
            data_x = tvm.placeholder(shape_x, name="data_x", dtype=inp_dtype)
            res = broadcast_to_d_compute(data_x, y, shape_align, kernel_name)
            tensors.append([data_x, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}

    tbe.build(schedules, config)
