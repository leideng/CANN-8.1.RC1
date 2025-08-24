# Copyright (c) Huawei Technologies Co., Ltd. 2019-2022. All rights reserved.
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
dynamic lin_space_d
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=too-many-locals
@register_operator_compute("LinSpaceD", op_mode="dynamic", support_fusion=True)
def lin_space_d_compute(input_assist, input_start, input_stop, input_num,
                        output_op, kernel_name="lin_space_d"):
    """
    algorithm: linspace
    Description of calculating process with TE api,
    the computational formula is as follows.
    `step = (stop - start)/(num - 1)`
    `res = assist * step + start`

    Parameters
    ----------
    input_assist: TVM tensor
        contains assist data
    input_start: TVM tensor
        contains start data
    input_stop: TVM tensor
        contains stop data
    input_num: TVM tensor
        contains num data
    output_op: dict
        dict of output, which contains shape and dtype
    kernel_name: str
        cce kernel name, default value is "lin_space"

    Returns
    -------
    res: TVM tensor
        the result of linspace compute
    """
    shape_assist = shape_util.shape_to_list(input_assist.shape)
    shape_start = shape_util.shape_to_list(input_start.shape)
    shape_stop = shape_util.shape_to_list(input_stop.shape)
    shape_num = shape_util.shape_to_list(input_num.shape)

    _, _, _, _, shape_max = shape_util.unify_broadcast_shapes([shape_assist, shape_start, shape_stop, shape_num])
    data_assist = tbe.broadcast(input_assist, shape_max)
    data_start = tbe.broadcast(input_start, shape_max)
    data_stop = tbe.broadcast(input_stop, shape_max)
    data_num = tbe.broadcast(input_num, shape_max)

    num_float = tbe.cast_to(data_num, "float32")

    num_divided = tbe.vadds(num_float, -1.0)
    step_divider = tbe.vsub(data_stop, data_start)
    step = tbe.vdiv(step_divider, num_divided)
    res_temp = tbe.vmul(data_assist, step)
    res = tbe.vadd(res_temp, data_start)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def lin_space_d(input_assist, input_start, input_stop,
                input_num, output_op, kernel_name="lin_space_d"):
    """
    algorithm: linspace
    Generates values in an interval.
    A sequence of 'num' evenly-spaced values are generated beginning at 'start'.
    If 'num' > 1, the values in the sequence increase by 'stop-start / num-1',
    so that the last one is exactly 'stop'.
    For example:
    linspace(10.0, 12.0, 3, name="linspace") => [ 10.0  11.0  12.0]

    Parameters
    ----------
    input_assist: dict
        dict of input, which contains shape and dtype
    input_start: dict
        dict of start, which contains shape and dtype
    input_stop: dict
        dict of stop, which contains shape and dtype
    input_num: dict
        dict of num, which contains shape and dtype
    output_op: dict
        dict of output, which contains shape and dtype
    kernel_name: str
        kernel name, default value is "lin_space"

    Returns
    -------
    None
    """
    shape_assist = input_assist.get("shape")
    shape_start = input_start.get("shape")
    shape_stop = input_stop.get("shape")
    shape_num = input_num.get("shape")
    dtype_input = input_start.get("dtype")
    dtype_input_stop = input_stop.get("dtype")
    dtype_input_assist = input_assist.get("dtype")
    dtype_num = input_num.get("dtype")
    para_check.check_shape(shape_assist, param_name="input_assist")

    para_check.check_dtype(dtype_input_assist.lower(), ("float32",), param_name="input_assist")
    para_check.check_dtype(dtype_input.lower(), ("float32",), param_name="input_start")
    para_check.check_dtype(dtype_input_stop.lower(), ("float32",), param_name="input_stop")
    para_check.check_dtype(dtype_num.lower(), ("int32",), param_name="input_num")

    # check shape of assist, only support 1dim
    if len(shape_assist) != 1:
        error_detail = "lin_space only support rank=1 while length of assist shape is %d"\
                       % (len(shape_assist))
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "input_assist", error_detail)

    ins = classify([input_assist, input_start, input_stop, input_num], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for _input_assist, _input_start, _input_stop, _input_num in ins:
        with tbe.compute():
            shape_assist, shape_start, shape_stop, shape_num = \
                shape_util.variable_shape([_input_assist, _input_start, _input_stop, _input_num])
            assist_input = tvm.placeholder(shape_assist, name="assist_input", dtype=dtype_input.lower())
            start_input = tvm.placeholder(shape_start, name="start_input", dtype=dtype_input.lower())
            stop_input = tvm.placeholder(shape_stop, name="stop_input", dtype=dtype_input.lower())
            num_input = tvm.placeholder(shape_num, name="num_input", dtype=dtype_num.lower())
            res = lin_space_d_compute(assist_input, start_input, stop_input, num_input, output_op, kernel_name)

            tensors.append([assist_input, start_input, stop_input, num_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)

