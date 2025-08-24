# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
dynamic in_training_reduce_v2
"""
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tuple_sum
from impl.util.util_common import is_unknown_rank_input
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json


# 'pylint: disable=locally-disabled,unused-argument,invalid-name,redefined-builtin
def op_select_format(x, sum, square_sum, kernel_name="in_training_reduce_v2"):
    """
    select format dynamically
    """
    input_format = "NC1HWC0, NC1HWC0"
    ori_format = x.get("ori_format")
    if ori_format in ("NDHWC", "NCDHW"):
        input_format = "NDC1HWC0, NDC1HWC0"

    input0 = gen_param(classify="input0",
                       name="x",
                       datatype="float16,float",
                       format=input_format,
                       unknownshape_format=input_format)
    output0 = gen_param(classify="output0",
                        name="sum",
                        datatype="float,float",
                        format=input_format,
                        unknownshape_format=input_format)
    output1 = gen_param(classify="output1",
                        name="square_sum",
                        datatype="float,float",
                        format=input_format,
                        unknownshape_format=input_format)

    param_list = [input0, output0, output1]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


@register_operator_compute("INTrainingReduceV2", op_mode="dynamic", support_fusion=True)
def in_training_reduce_v2_compute(x, sum, square_sum, kernel_name="in_training_reduce_v2", reduce_axis=None):
    """
    DSL description of the instancenorm operator's mathematical calculation process

    Parameters
    ----------
    x: TVM tensor
        the placeholder of input x
    sum: dict
        shape and dtype of input sum
    square_sum: dict
        shape and dtype of input square_sum
    kernel_name: str
        cce kernel name, default value is "in_training_reduce_v2"
    reduce_axis: list
        reduce axis of input shape

    Returns
    -------
    res_tuple: tuple
        (sum_x, square_sum_x)
    """
    dtype = x.dtype.lower()
    if dtype == "float16":
        x = tbe.cast_to(x, "float32")

    data_format = sum.get("format").upper()
    if not reduce_axis and data_format in ("NC1HWC0",):
        axis = [2, 3]
    elif not reduce_axis and data_format in ("NDC1HWC0",):
        axis = [1, 3, 4]
    else:
        axis = reduce_axis
    square_x = tbe.vmul(x, x)
    sum_x, square_sum_x = tuple_sum([x, square_x], axis, True)
    res = [sum_x, square_sum_x]

    return res


@register_operator("INTrainingReduceV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def in_training_reduce_v2(x, sum, square_sum, kernel_name="in_training_reduce_v2"):
    """
    instancenorm operator interface implementation

    Parameters
    ----------
    x: dict
        shape and dtype of input x, only support float16, float32
    sum: dict
        shape and dtype of input sum, only support float32
    square_sum: dict
        shape and dtype of input square_sum, only support float32
    kernel_name: str
        cce kernel name, default value is "in_training_reduce_v2"

    Returns
    -------
    None
    """
    dtype_x = x.get("dtype")
    para_check.check_dtype(dtype_x.lower(), ("float16", "float32"), param_name="x")

    data_format = x.get("format")
    if data_format in ("NC1HWC0",):
        list_axis = [2, 3]
    else:
        list_axis = [1, 3, 4]

    if is_unknown_rank_input(x):
        if data_format == "NC1HWC0":
            x["shape"] = [-1, -1, -1, -1, -1]
            x["range"] = [(1, None), (1, None), (1, None), (1, None), (1, None)]
        else:
            x["shape"] = [-1, -1, -1, -1, -1, -1]
            x["range"] = [(1, None), (1, None), (1, None), (1, None), (1, None), (1, None)]

    ins = classify([x, list_axis], OpPatternMode.TUPLE_REDUCE)
    schedules, tensors = [], []
    for (_x, _reduce_axis) in ins:
        with tbe.compute():
            shape_x = shape_util.variable_shape([_x])
            input_x = tvm.placeholder(shape_x[0], name="input_x", dtype=dtype_x)
            res = in_training_reduce_v2_compute(input_x, sum, square_sum, kernel_name, _reduce_axis)

            tensor_list = [input_x] + list(res)
            tensors.append(tensor_list)

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
