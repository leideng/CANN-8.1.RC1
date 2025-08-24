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
dynamic range_d
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
# 'pylint: disable=invalid-name
@register_operator_compute("RangeD", op_mode="dynamic", support_fusion=True)
def range_d_compute(x, y, start, limit, delta, kernel_name="range_d"):
    """
    algorithm: range_d
    Description of calculating process with TE api, the computational formula
    is as follows.
    res = input_assist * delta + start

    Parameters
    ---------
    x: TVM tensor
        contains assist data
    start: scalar int float
        contains the data of start
    limit: scalar int float
        contains the data of limit
    delta: scalar int float
        contains the data of delta
    y: dict
        dict of output, which contains shape and dtype
    kernel_name: str
        cce kernel name, default value is "range_d"

    Returns
        ------
    res: TVM tensor
        the result of range_d compute
    """
    if isinstance(start, int) and isinstance(delta, int) and isinstance(limit, int):
        mid_res = tbe.vmuls(x, tvm.const(delta, dtype="int32"))
        res = tbe.vadds(mid_res, tvm.const(start, dtype="int32"))
        return res

    mid_res = tbe.vmuls(x, tvm.const(delta, dtype="float32"))
    res = tbe.vadds(mid_res, tvm.const(start, dtype="float32"))

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_FLOAT,
                            para_check.REQUIRED_ATTR_FLOAT, para_check.REQUIRED_ATTR_FLOAT, para_check.KERNEL_NAME)
def range_d(x, y, start, limit, delta, kernel_name="range_d"):
    """
    algorithm: range_d
    Generates values in an interval
    A sequence of delta evenly-spaced values are generated beginning at start
    so that the last one is exactly limit
    For example:
    range_d(1, 10.0, 2) => [ 1.0,3.0,5.0,7.0,9.0]
    range_d(1, 5)=> [1,2,3,4]
    range_d(5)=> [0,1,2,3,4]

    Parameters
    ----------
    x: dict
        dict of input, which contains shape and dtype
    y: dict
        dict of output, which contains shape and dtype
    start: scalar
        scalar of start, which contains int or float
    limit: scalar
        scalar of limit, which contains int or float
    delta: scalar
        scalar of delta, which contains int or float
    kernel_name: str
        kernel name, default value is "range_d"

    Returns
    -------
    None
    """
    shape_assist = x.get("shape")
    dtype_assist = x.get("dtype").lower()

    para_check.check_shape(shape_assist, param_name="x")
    para_check.check_dtype(dtype_assist.lower(), ("int32", "float32"), param_name="x")

    if dtype_assist == "int32":
        start = int(start)
        limit = int(limit)
        delta = int(delta)

    if limit == start:
        rule_desc = "start can not equal to limit"
        param_value = "%d,%d" % (limit, start)
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule_desc, "limit,start", param_value)
    if delta == 0:
        rule_desc = "the input of delta can not equal to zero"
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule_desc, "delta", delta)
    if (start > limit) and (delta > 0):
        rule_desc = "requires limit should more than start when delta is more than zero"
        param_value = "%d,%d" % (limit, start)
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule_desc, "limit,start", param_value)
    if (start < limit) and (delta < 0):
        rule_desc = "requires start should more than limit when delta is less than zero"
        param_value = "%d,%d" % (start, limit)
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule_desc, "start,limit", param_value)

    # check shape of assist,only support 1dim
    if len(shape_assist) != 1:
        error_detail = "range_d only support rank=1 while length of x shape is %d" % (len(shape_assist))
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "x", error_detail)

    ins = classify([x], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_x,) in ins:
        with tbe.compute():
            shape_x = shape_util.variable_shape([_x])
            x_input = tvm.placeholder(shape_x[0], name="x_input", dtype=dtype_assist)
            res = range_d_compute(x_input,  y, start, limit, delta, kernel_name)

            tensors.append([x_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)

