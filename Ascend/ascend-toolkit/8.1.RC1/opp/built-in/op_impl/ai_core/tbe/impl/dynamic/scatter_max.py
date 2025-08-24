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
scatter_max
"""
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from . import scatter_common


def scatter_max_tik(var, indices, updates, var_out, kernel_name="ScatterMax"):
    obj = scatter_common.ScatterCommon(var, indices, updates, var_out, False, kernel_name, "vmax")
    return obj.scatter_common_operator()


def scatter_max_dsl(var, indices, updates, var_out, kernel_name="ScatterMax"):
    """
    scatter_max interface for dsl
    """
    check_list_var = ("float16", "float32", "int32")
    check_list_indices = ("int32", "int64")
    check_list_updates = ("float16", "float32", "int32")
    dtype_var = var.get("dtype").lower()
    dtype_indices = indices.get("dtype").lower()
    dtype_updates = updates.get("dtype").lower()
    para_check.check_dtype(dtype_var, check_list_var, param_name="var")
    para_check.check_dtype(dtype_indices, check_list_indices, param_name="indices")
    para_check.check_dtype(dtype_updates, check_list_updates, param_name="updates")

    op_type = "scatter"
    reduction = "max"
    ins = classify([var, indices, updates], op_type)
    schedules, tensors = [], []
    for var_input, indices_input, updates_input in ins:
        with tbe.compute():
            var_shape, indices_shape, updates_shape = \
                shape_util.variable_shape([var_input, indices_input, updates_input], op_type)
            var_tensor = tvm.placeholder(var_shape, name="var", dtype=dtype_var)
            indices_tensor = tvm.placeholder(indices_shape, name="indices", dtype=dtype_indices)
            updates_tensor = tvm.placeholder(updates_shape, name="updates", dtype=dtype_updates)
            res = tbe.scatter(var_tensor, indices_tensor, updates_tensor, reduction, support_out_of_bound_index=False)
            tensors.append([var_tensor, indices_tensor, updates_tensor, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)


# 'pylint: disable=unused-argument,too-many-arguments
@register_operator("ScatterMax")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def scatter_max(var, indices, updates, var_out, use_locking=False, kernel_name="scatter_max"):
    """
    scatter_max interface

    Parameters
    ----------
    var_dict: input var shape, dtype and range
    indices_dict: input indices shape, dtype and range
    updates_dict: input updates shape, dtype and range
    var_out_dict: output shape, dtype and range
    kernel_name: kernel name of scatter_max op

    Returns
    -------
    compile info
    """
    if not tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        scatter_max_tik(var, indices, updates, var_out, kernel_name)
    else:
        scatter_max_dsl(var, indices, updates, var_out, kernel_name)
