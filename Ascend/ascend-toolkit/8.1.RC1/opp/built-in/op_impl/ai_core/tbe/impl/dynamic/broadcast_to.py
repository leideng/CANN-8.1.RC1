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
dynamic broadcast_to
"""
from impl.util.util_common import is_unknown_rank_input
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_register
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.util_common import gen_range


def generalize_keep_rank(in_tensor):
    """
    generalize_keep_rank
    """
    in_tensor["shape"] = [-2]
    in_tensor["range"] = []
    in_tensor["ori_shape"] = [-2]
    # note: must set const_value = None, can not use `del in_tensor['const_value']`
    if "const_value" in in_tensor:
        in_tensor['const_value'] = None

    return in_tensor


def generalize_all_shape(in_tensor):
    """
    generalize_all_shape
    """
    dynamic_rank_shape = [-2]
    in_tensor["shape"] = dynamic_rank_shape
    in_tensor["ori_shape"] = dynamic_rank_shape
    if in_tensor.get("format") not in ("NC1HWC0",):
        in_tensor["format"] = "ND"
        in_tensor["ori_format"] = "ND"
    return in_tensor


@tbe_register.register_param_generalization("BroadcastTo")
def broadcast_to_generalization(x, shape, y, kernel_name="broadcast_to", generalize_config=None):
    """
    broadcast_to_generalization
    """
    if generalize_config is None:
        return None

    if generalize_config.get("mode") == "keep_rank":
        x = generalize_keep_rank(x)
        shape = generalize_keep_rank(shape)
        y = generalize_keep_rank(y)
    elif generalize_config.get("mode") == "all_shape":
        x = generalize_all_shape(x)
        shape = generalize_all_shape(shape)
        y = generalize_all_shape(y)

    return [[x, shape, y]]


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
@register_operator_compute("BroadcastTo", op_mode="dynamic", support_fusion=False)
def broadcast_to_compute(x, shape, y, kernel_name="broadcast_to"):
    """
    TVM calculation process, used for fusion operation.

    Parameters
    ----------
    x: list of placeholders.
        Input data.
    shape : list or tuple.
        Number of the axis replicates.
    y: dict.
        dict of output.

    kernel_name : str.
        Cce kernel name, default value is "broadcast_to_d".

    Returns
    -------
    res
    """
    shape_x, shape, shape_max = \
        shape_util.broadcast_shapes(x.shape, shape,
                                    param_name_input1="input_x",
                                    param_name_input2="input_y")
    src_dtype = x.dtype.lower()
    if src_dtype in ("int8", "uint8", "bool"):
        x = tbe.cast_to(x, "float16")
    res = tbe.broadcast(x, shape_max)
    if src_dtype in ("int8", "uint8", "bool"):
        res = tbe.cast_to(res, src_dtype)

    return res


def get_shape_adapt(input_x_shape, input_shape_shape, input_x_range, shape, kernel_name):
    """
    get shape adapt is shape and range
    """
    dims_value = input_shape_shape[0]

    if dims_value < -1:
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "shape", "shape[0] should be greater than -1")

    if dims_value == -1:
        dims_value = len(input_x_shape)
    dims_value = 1 if dims_value == 0 else dims_value

    if len(input_x_shape) > dims_value:
        error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "x", "shape", \
            "the dimensions of x should not be greater than shape[0]")

    shape_shape_adapt = []
    shape_range_adapt = []

    const_value = shape.get('const_value')
    if const_value:
        const_value = list(const_value)
        shape_shape_adapt = _pre_tiling(input_x_shape, const_value)
        shape_range_adapt = gen_range(shape_shape_adapt)

    else:
        for shape_i, range_i in zip(input_x_shape, input_x_range):
            if shape_i == 1 or (shape_i == -1 and range_i[0] <= 1):
                shape_shape_adapt.append(-1)
                shape_range_adapt.append((1, None))
            else:
                shape_shape_adapt.append(shape_i)
                shape_range_adapt.append(range_i)

        shape_shape_adapt = [-1] + shape_shape_adapt
        shape_range_adapt = [(1, None)] + shape_range_adapt

    return [shape_shape_adapt, shape_range_adapt]


def _accum_mul(vec):
    """
    accumlate vec by mul
    """
    res_mul = 1
    for i in vec:
        res_mul *= i

    return res_mul


def _pre_tiling(x, shape):
    """
    get broadcast shape just like tiling
    x [2, 1, 5]
    shape [6, 5, 2, 5, 5]
    return [30, 2, 5, 5]
    """
    len_diff = len(shape) - len(x)
    x = [1, ] * len_diff + x
    shape = [x[index] if dim_value == -1 else dim_value for index, dim_value in enumerate(shape)]

    return [_accum_mul(shape[:len_diff])] + shape[len_diff:]


# 'pylint: disable=too-many-locals,too-many-statements
@register_operator("BroadcastTo")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def broadcast_to(x, shape, y, kernel_name="broadcast_to"):
    """algorithm: broadcast_to.
    The broadcast_to in tensorflow can multiple the shape of the given tensor.
    For example, tiling [a b c d] by [2] produces [a b c d a b c d].
    The broadcast_to op in TBE is compatible with the tensorflow operator BroadcastTo
    Abnormal condition:
    1. The length of shape must be equal to or less than the shape of multiples.

    Parameters
    ----------
    x : dict
        shape and dtype of input
    shape : dict
        shape and dtype of multiples
    y: dict
        dict of output.
    kernel_name : str.
        kernel name, default value is "broadcast_to".

    Returns
    -------
    None
    """

    input_x_dtype = x.get("dtype").lower()
    input_shape_dtype = shape.get("dtype").lower()

    input_x_shape = list(x.get("shape"))
    input_shape_shape = list(shape.get("shape"))

    check_list = ('bfloat16', 'float16', 'float32', 'int8', 'uint8', 'int32', 'uint32', 'int64', 'bool')
    para_check.check_dtype(input_x_dtype, check_list, param_name="x")
    check_list = ('int32', 'int64')
    para_check.check_dtype(input_shape_dtype, check_list, param_name="shape")

    if len(input_shape_shape) > 1:
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "shape", "shape should be 1D")

    if is_unknown_rank_input([x, shape]):
        x, shape = [x, x] if is_unknown_rank_input(x) else [shape, shape]
    else:
        input_x_range = list(x.get("range"))
        shape_shape_adapt, shape_range_adapt = \
            get_shape_adapt(input_x_shape, input_shape_shape, input_x_range, shape, kernel_name)

        shape["shape"] = shape_shape_adapt
        shape["range"] = shape_range_adapt

    ins = classify([x, shape], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_x, _shape) in ins:
        with tbe.compute():
            shape_x, shape_shape = shape_util.variable_shape([_x, _shape])
            shape_input = tvm.placeholder(shape_shape, name="shape_input", dtype=input_shape_dtype)
            x_input = tvm.placeholder(shape_x, name="x_input", dtype=input_x_dtype)
            res = broadcast_to_compute(x_input, shape_shape, y, kernel_name=kernel_name)
            tensors.append([x_input, shape_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
