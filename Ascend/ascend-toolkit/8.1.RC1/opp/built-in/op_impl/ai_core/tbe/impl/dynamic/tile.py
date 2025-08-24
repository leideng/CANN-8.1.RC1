# Copyright 2020 Huawei Technologies Co., Ltd
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
dynamic tile
"""
from impl.util.util_common import is_unknown_rank_input
from impl.util.util_common import gen_range
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
@register_operator_compute("Tile", op_mode="dynamic", support_fusion=False)
def tile_compute(data, multiples, output_x, kernel_name="tile"):
    """
    TVM calculation process, used for fusion operation.

    Parameters
    ----------
    data: list of placeholders.
        Input data.
    multiples : list or tuple.
        Number of the axis replicates.
    output_x: dict.
        dict of output.

    kernel_name : str.
        Cce kernel name, default value is "tile_d".

    Returns
    -------
    res
    """
    shape_data, multiples, shape_max = \
        shape_util.broadcast_shapes(data.shape, multiples,
                                    param_name_input1="input_x",
                                    param_name_input2="input_y")

    src_dtype = data.dtype
    if src_dtype == "int8" or src_dtype == "uint8":
        data = tbe.cast_to(data, "float16")
    res = tbe.broadcast(data, shape_max)
    if src_dtype == "int8" or src_dtype == "uint8":
        res = tbe.cast_to(res, src_dtype)
    return res


def _do_adapt_shape(input_x_shape, input_x_range, input_m_const_value):
    """
    do adapt shape
    """
    shape_adapt = []
    multiples_adapt = []
    range_adapt = []
    multiples_range_adapt = []

    if input_m_const_value:
        input_m_const_value = list(input_m_const_value)
        input_m_const_value_range = gen_range(input_m_const_value)
        for shape_i, range_i, shape_m, range_m in \
            zip(input_x_shape, input_x_range,
                input_m_const_value, input_m_const_value_range):

            if shape_i == 1:
                shape_adapt.append(1)
                range_adapt.append((1, 1))
                multiples_adapt.append(shape_m)
                multiples_range_adapt.append(range_m)

            else:
                shape_adapt.append(1)
                range_adapt.append((1, 1))
                shape_adapt.append(shape_i)
                range_adapt.append(range_i)
                multiples_adapt.append(shape_m)
                multiples_range_adapt.append(range_m)
                if shape_i == -1:
                    multiples_adapt.append(-1)
                    multiples_range_adapt.append((1, None))

                else:
                    multiples_adapt.append(shape_i)
                    multiples_range_adapt.append(range_i)

    else:
        for shape_i, range_i in zip(input_x_shape, input_x_range):
            if shape_i == 1:
                shape_adapt.append(1)
                range_adapt.append((1, 1))
                multiples_adapt.append(-1)
                multiples_range_adapt.append((1, None))

            else:
                shape_adapt.append(1)
                range_adapt.append((1, 1))
                shape_adapt.append(shape_i)
                range_adapt.append(range_i)
                multiples_adapt.append(-1)
                multiples_range_adapt.append((1, None))
                if shape_i == -1:
                    multiples_adapt.append(-1)
                    multiples_range_adapt.append((1, None))

                else:
                    multiples_adapt.append(shape_i)
                    multiples_range_adapt.append(range_i)

    return [shape_adapt, range_adapt, multiples_adapt, multiples_range_adapt]


def adapt_shape_compute(input_x_shape, input_x_range, input_m_shape, input_m_const_value=None):
    """
    adapt_shape_compute
    """
    dims_value = input_m_shape[0]
    if dims_value < -1:
        error_manager_vector.raise_err_input_value_invalid("tile", "multiples", "shape value should be more than -1",
                                                           str(dims_value))

    dims_value = len(input_x_shape) if dims_value == -1 else dims_value
    if len(input_x_shape) == 1 and input_x_shape[0] == 1 and dims_value == 0:
        dims_value = 1

    if len(input_x_shape) > dims_value and input_m_const_value:
        input_m_const_value = [1] * (len(input_x_shape) - dims_value) + list(input_m_const_value)

    if len(input_x_shape) < dims_value:
        len_diff = dims_value - len(input_x_shape)
        input_x_shape = [1] * len_diff + input_x_shape
        input_x_range = [(1, 1)] * len_diff + input_x_range

    return _do_adapt_shape(input_x_shape, input_x_range, input_m_const_value)


# 'pylint: disable=too-many-locals,too-many-statements
@register_operator("Tile")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def tile(input_x, input_m, output_x, kernel_name="tile"):
    """algorithm: tile.
    The tile in tensorflow can multiple the shape of the given tensor.
    For example, tiling [a b c d] by [2] produces [a b c d a b c d].
    The tile op in TBE is compatible with the tensorflow operator Tile
    Abnormal condition:
    1. The length of shape must be equal to or less than the shape of multiples.

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    input_m : dict
        shape and dtype of multiples
    output_x: dict
        dict of output.
    kernel_name : str.
        kernel name, default value is "tile".

    Returns
    -------
    None
    """

    input_x_dtype = input_x.get("dtype").lower()
    check_list = ("float16", "float32", "int8", "uint8", "int32", "bool", "int64", "bfloat16",
                  "int16", "uint16", "uint32", "uint64", "complex64")
    para_check.check_dtype(input_x_dtype, check_list, param_name="input_x")
    if input_x_dtype == "bool":
        input_x_dtype = "int8"

    # multiples : A Tensor. Must be one of the following types: int32, int64
    input_m_dtype = input_m.get("dtype").lower()
    check_list = ("int32", "int64")
    para_check.check_dtype(input_m_dtype, check_list, param_name="input_multiples")

    # multiples : A Tensor. Must be 1-D
    input_m_shape = list(input_m.get("shape"))
    input_x_shape = list(input_x.get("shape"))
    compile_shape = input_x_shape.copy()
    if len(input_m_shape) > 1:
        error_manager_vector.raise_err_input_value_invalid("tile", "multiples", "should be 1-D",
                                                           str(len(input_m_shape)))

    is_unknown = False
    if is_unknown_rank_input([input_x, input_m]):
        input_x, input_m = [input_x, input_x] if is_unknown_rank_input(input_x) else [input_m, input_m]
        is_unknown = True
    else:
        input_m_const_value = input_m.get("const_value")
        input_x_range = list(input_x.get("range"))
        input_x["shape"], input_x["range"], input_m["shape"], input_m["range"] = \
            adapt_shape_compute(input_x_shape, input_x_range, input_m_shape, input_m_const_value)

    ins = classify([input_m, input_x], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_input_m, _input_x) in ins:
        with tbe.compute():
            shape_mul, shape_x = shape_util.variable_shape([_input_m, _input_x])
            data = tvm.placeholder(shape_x, name="input_x", dtype=input_x_dtype)
            input_mul = tvm.placeholder(shape_mul, name="multiples", dtype=input_m_dtype)

            res = tile_compute(data, shape_mul, output_x, kernel_name)
            tensors.append([data, input_mul, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}

    tbe.build(schedules, config)

    tbe_context.get_context().add_compile_info("compile_shape", compile_shape)
