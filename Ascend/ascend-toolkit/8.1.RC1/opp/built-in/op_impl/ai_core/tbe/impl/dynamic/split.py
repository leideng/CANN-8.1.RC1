# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
split
"""
from __future__ import absolute_import
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import error_manager_vector
from .split_v import SplitV
from .split_v import split_v_compute


def check_input_params(x, split_dim, y, num_split):
    """
    check input parameters
    """
    # split has 2 input tensors, so 62 is the maximum of output tensors
    if num_split > 62 or num_split < 1:
        error_manager_vector.raise_err_input_value_invalid("split", "num_split",
                                                           "62 is the maximum of num_split", num_split)

    x_dtype = x.get("dtype").lower()
    split_dim_dtype = split_dim.get("dtype").lower()
    output_dtype = y[0].get("dtype").lower()

    check_list = ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
                  "bool", "float16", "float32", "bfloat16")
    para_check.check_dtype(x_dtype, check_list, param_name="x")
    para_check.check_dtype(split_dim_dtype, ("int32",), param_name="split_dim")

    if x_dtype != output_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal("split", "x_dtype", "y_dtype",
                                                              x_dtype, output_dtype)


def split_tik(split_dim, x, y, num_split, kernel_name="split"):
    """
    split interface for tik
    """
    size_splits = {}
    obj = SplitV(x, size_splits, split_dim, y, num_split, kernel_name)
    obj.split_v_compute_tiling()

    # add compile info
    tbe_context.get_context().add_compile_info("vars", {"core_num": obj.core_num,
                                    "ub_elems": obj.ub_elems,
                                    "num_split": obj.num_split
                                    })
    # It is used to distinguish between Tik implementation and DSL implementation in the tilling phase
    tbe_context.get_context().add_compile_info("is_tik", True)

    tik_inst = obj.tik_instance
    tik_inst.BuildCCE(kernel_name=obj.kernel_name,
                      inputs=(obj.split_dim_gm, obj.x_gm),
                      outputs=obj.outputs_gm,
                      flowtable=(obj.tiling_gm,), enable_l2=True)
    return tik_inst


def split_dsl(split_dim, x, y, num_split, kernel_name="split_dsl"):
    '''
    split_v interface for dsl
    '''
    dtype_x = x.get("dtype")
    input0 = tvm.placeholder((1,), dtype=split_dim.get("dtype"), name="input0")
    extra_params = {"avg_split": True, "num_split":num_split}
    tbe_context.get_context().add_compile_info("split_axis_idx", 0)
    tbe_context.get_context().add_compile_info("input_idx", 1)

    schedules, tensors = [], []
    ins = classify([x, split_dim], "split", extra_params)

    for input_x_, axis_, size_splits_ in ins:
        with tbe.compute():
            shape_x, size_splits = shape_util.variable_shape([input_x_, size_splits_], "split")
            input_tensors = tvm.placeholder(shape_x, dtype=dtype_x, name="data_x")

            res = split_v_compute(input_tensors, size_splits, axis_, y, num_split, kernel_name)

            tensors.append([input0, input_tensors, *res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name":kernel_name, "tensor_list":tensors}
    tbe.build(schedules, config)


@register_operator("Split")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.DYNAMIC_OUTPUT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def split(split_dim, x, y, num_split, kernel_name="split"):
    """
    Split a tensor into num_split tensors along one dimension.

    Parameters
    ----------
    split_dim: dict
        the dict of input split_dim tensor.
        An int, specifies the dimension along which to split.
    x: dict
        the dict of input tensor.
    y: list or tuple
        the list of output tensor.
    num_split: int
        an integer indicating the number of outputs.
    kernel_name: str
        cce kernel name, default value is "split".

    Returns
    -------
    compile info
    """
    if num_split is None:
        num_split = len(y)

    check_input_params(x, split_dim, y, num_split)

    if tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        split_dsl(split_dim, x, y, num_split, kernel_name)
    else:
        split_tik(split_dim, x, y, num_split, kernel_name)
