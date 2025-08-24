# Copyright 2019-2022 Huawei Technologies Co., Ltd
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
gather
"""

from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import OpImplMode
from impl.util.platform_adapter import OpPatternMode
from impl.util.util_common import check_op_impl_mode
from impl.dynamic.gather_v2 import GatherV2


# 'pylint: disable=locally-disabled,invalid-name,unused-argument,too-many-arguments
def gather_tik(x,
               indices,
               y,
               validate_indices=True,
               batch_dims=0,
               kernel_name="Gather",
               impl_mode="high_precision"):
    """
    gather interface for tik
    """
    tbe_context.get_context().add_compile_info("is_gather_v2", False)
    axis_dict = {"dtype": "int32"}
    obj = GatherV2(x, indices, axis_dict, y, batch_dims, False, kernel_name, impl_mode)
    return obj.gather_compute()


def gather_compute(x,
                   indices,
                   y,
                   validate_indices=True,
                   batch_dims=0,
                   negative_index_support=False,
                   kernel_name="gather",
                   impl_mode=None):
    """
    gather compute

    Parameters
    ----------
    x: input params shape, dtype and range
    indices: input indices shape, dtype and range
    y: output shape, dtype and range
    validate_indices: Whether to verify the values of indices, not currently enabled
    batch_dims: the number of batch dimensions
    negative_index_support: Whether to support negative index, An optional bool. Defaults to false.
    kernel_name: kernel name of gather op

    Returns
    -------
    res: TVM tensor
        the result of gather
    """
    support_out_of_bound_index = True if impl_mode == "support_out_of_bound_index" else False

    res = tbe.gather(x, indices, batch_dims + 1, batch_dims, negative_index_support, support_out_of_bound_index)

    return res


def gather_dsl(x,
               indices,
               y,
               validate_indices=True,
               batch_dims=0,
               negative_index_support=False,
               kernel_name="gather",
               impl_mode=None):
    """
    gather interface for dsl
    """
    check_x_list = ("float16", "bfloat16", "float32", "int8", "uint8", "int32", "uint32", "int16", "uint16", "int64",
                    "uint64", "bool")
    check_list_ids = ("int32", "int64")
    x_dtype = x.get("dtype").lower()
    ids_dtype = indices.get("dtype").lower()
    para_check.check_dtype(x_dtype, check_x_list, param_name="x")
    para_check.check_dtype(ids_dtype, check_list_ids, param_name="indices")

    # In the gather scenario, when batch_dims is not 0, set axis and batch_dims to the same value.
    batch_dims = "unknown" if batch_dims is None else batch_dims
    tbe_context.get_context().add_compile_info("attr_name", "batch_dims")
    tbe_context.get_context().add_compile_info("batch_dims_attr_idx", 1)
    tbe_context.get_context().add_compile_info("impl_mode", impl_mode)
    ins = classify([x, indices, None, batch_dims], OpPatternMode.GATHER, {"impl_mode": impl_mode})
    schedules, tensors = [], []
    for shape_x, shape_indices, axis_input, batch_dims_input in ins:
        with tbe.compute():
            x_var, indices_var, axis_dim, batch_dims = \
                shape_util.variable_shape([shape_x, shape_indices, axis_input, batch_dims_input], "gather")
            x_tensor = tvm.placeholder(x_var, name="x", dtype=x_dtype)
            indices_tensor = tvm.placeholder(indices_var, name="indices", dtype=ids_dtype)

            res = gather_compute(x_tensor, indices_tensor, y, False, batch_dims, negative_index_support, kernel_name,
                                 impl_mode)
            tensors.append([x_tensor, indices_tensor, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)


@register_operator("Gather")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME, para_check.OPTION_ATTR_STR)
def gather(x,
           indices,
           y,
           validate_indices=True,
           batch_dims=0,
           negative_index_support=False,
           kernel_name="gather",
           impl_mode="high_precision"):
    """
    gather interface

    Parameters
    ----------
    x: input params shape, dtype and range
    indices: input indices shape, dtype and range
    y: output shape, dtype and range
    validate_indices: Whether to verify the values of indices, not currently enabled
    batch_dims: the number of batch dimensions
    kernel_name: kernel name of gather op
    impl_mode: str. The flag for cache data at index 0. No need to add into ops_info file. Tempoarily support
               high_performance, high_precision and support_out_of_bound_index.

    Returns
    -------
    None
    """
    check_op_impl_mode(impl_mode,
                       [OpImplMode.HIGH_PERFORMANCE, OpImplMode.HIGH_PRECISION, OpImplMode.SUPPORT_OUT_OF_BOUND_INDEX],
                       kernel_name)
    if x.get("dtype").lower() == "bfloat16":
        gather_dsl(x, indices, y, validate_indices, batch_dims, negative_index_support, kernel_name, impl_mode)
    elif tbe_platform.api_check_support("tbe.dsl.vexp", "float32") and impl_mode != "high_performance":
        gather_dsl(x, indices, y, validate_indices, batch_dims, negative_index_support, kernel_name, impl_mode)
    else:
        gather_tik(x, indices, y, validate_indices, batch_dims, kernel_name, impl_mode)
