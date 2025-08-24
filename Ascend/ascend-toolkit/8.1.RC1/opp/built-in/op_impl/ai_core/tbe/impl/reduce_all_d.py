# Copyright 2019 Huawei Technologies Co., Ltd
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
reduce_all_d
"""
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import error_manager_vector

# shape limit for aicore equals 2**31
SHAPE_SIZE_LIMIT = 2147483648


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=too-many-locals
@register_operator_compute("reduce_all_d", op_mode="static", support_fusion=True)
def reduce_all_d_compute(input_data, output_data, axes, keepdims, kernel_name="reduce_all_d"):
    """ TVM calculation process, used for fusion operation

    Parameters
    ----------
    input_data: TVM tensor
        the placeholder of input data
    output_data: dict
        shape and dtype of output, should be same shape and type as input
    axes: int, list ,tuple or None.
        the first axes to reduce, may be negative to index from the end
        (e.g., -1 for the last axes).
        axes may be int or list(e.g. [1,2])
    keepdims : bool or None .
        if true, retains reduced dimensions with length 1,
        default value is None
    kernel_name: str
        cce kernel name, default value is "all_cce"

    Returns
    -------
    result: TVM tensor.
    """
    shape = shape_util.shape_to_list(input_data.shape)
    shape_len = len(shape)
    if not axes:
        axes = range(shape_len)
    if hasattr(axes, 'index'):
        axes = list(axes)

    dtype = input_data.dtype
    data_fp16 = tbe.cast_to(input_data, "float16")
    data_abs = tbe.vabs(data_fp16)
    result_tmp = tbe.reduce_min(data_abs, axes, keepdims=False)
    result = tbe.cast_to(result_tmp, dtype, True)

    return result


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            (para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_LIST_INT),
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def reduce_all_d(input_data, output_data, axes, keep_dims=None, kernel_name="reduce_all_d"):
    """
    Reduce a tensor on a certain axes based on min

    Parameters:
    ----------
    input_data: dict
        shape and dtype of input_data, only support int8
    output_data: dict
        source data type, only support int8
    axes : int, list ,tuple or None.
        the first axes to reduce, may be negative to index from the end
        (e.g., -1 for the last axes).
        axes may be int or list(e.g. [1,2])
    keep_dims : bool or None .
        if true, retains reduced dimensions with length 1,
        default value is None
    kernel_name : str
        cce kernel name, default value is "cce_all"

    Returns
    -------
    None
    """
    input_shape = input_data.get("shape")
    input_dtype = input_data.get("dtype").lower()
    if input_dtype == "bool":
        input_dtype = "int8"
    para_check.check_shape(input_shape, param_name="input_data")
    para_check.check_dtype(input_dtype, ("int8"), param_name="input_data")

    shape_len = len(input_shape)
    if not axes:
        axes = range(shape_len)

    if hasattr(axes, 'index'):
        axes = list(axes)
    axes = shape_util.axis_check(shape_len, axes)

    if not isinstance(axes, int):
        for i in axes:
            if i >= len(input_shape):
                rule_desc = "axes should be less than dimension"
                param_value = "%d,%d" % (i, len(input_shape))
                error_manager_vector.raise_err_check_params_rules(kernel_name, rule_desc, \
                                                          "i,input_shape", param_value)
    else:
        if axes >= len(input_shape):
            rule_desc = "axes should be less than dimension"
            param_value = "%d,%d" % (axes, len(input_shape))
            error_manager_vector.raise_err_check_params_rules(kernel_name, rule_desc, \
                                                              "axes,input_shape", param_value)

    # 5HD Special param for 5hd schedule
    is_5hdc = para_check.check_and_init_5hdc_reduce_support(input_data, axes)
    if not is_5hdc:
        input_shape, axes = shape_util.shape_refine(list(input_shape), axes)
        input_shape, axes = shape_util.simplify_axis_shape(input_shape, axes)

    data_input = tvm.placeholder(input_shape, name="data_input_" + kernel_name, dtype=input_dtype)
    result = reduce_all_d_compute(data_input, output_data, axes, keep_dims, kernel_name)
    if is_5hdc:
        result.ori_shape = input_data["ori_shape"]
        result.ori_format = input_data["ori_format"]

    with tvm.target.cce():
        sch = auto_schedule(result)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": [data_input, result]}
    build(sch, config)
