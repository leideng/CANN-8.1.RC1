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
inplace_update
"""
from impl.dynamic.scatter_update import ScatterUpdate
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context


# 'pylint: disable=locally-disabled,invalid-name,unused-argument,too-many-branches
# 'pylint: disable=superfluous-parens
def check_supported(x, indices, v, kernel_name):
    """
    func to check input params
    """
    dtype_x = x.get("dtype")
    dtype_v = v.get("dtype")
    shape_indices = indices.get("shape")


    check_list = ["float16", "float32"]
    input_dtype_x = dtype_x.lower()
    input_dtype_v = dtype_v.lower()
    if (input_dtype_x not in check_list):
        reason = "input_x's dtype %s is not support, it must be in %s" % (input_dtype_x, str(check_list))
        return False, reason

    if (input_dtype_v not in check_list):
        reason = "input_v's dtype %s is not support, it must be in %s" % (input_dtype_v, str(check_list))
        return False, reason

    if len(shape_indices) != 1:
        reason = "The shape length of indices only support 1"
        return False, reason
    return True, ""


@register_operator("InplaceUpdate")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def inplace_update(x, indices, v, y, kernel_name="inplace_update"):
    """
    Updates specified rows with values in v

    Parameters
    ----------
    x : dict
        shape and dtype of input tensor x, only support float16, float32, int32
    indices: dict
        Indices into the left-most dimension of x
    v : dict
        shape and dtype of input tensor v,
         should be same shape and type as input
    y : dict
        shape and dtype of output tensor should be same shape and type as input
    kernel_name : str
        kernel name, default value is "inplace_update"

    Returns
    -------
    tik_instance
    """
    obj = ScatterUpdate(x, indices, v, y, False, kernel_name, opname="inplace_update")
    obj.scatter_update_compute_tiling()
    opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
    tbe_context.get_context().add_compile_info(
        "vars", {
            "ub_size": obj.ub_size_bytes,
            "core_num": obj.ai_core_num,
            "var_size": obj.var_dtype_bytes_size,
            "indices_size": obj.indices_dtype_bytes_size
        })
    tbe_context.get_context().add_compile_info("is_tik", True)
    obj.tik_instance.BuildCCE(kernel_name=obj.kernel_name,
                              inputs=(obj.var_gm, obj.indices_gm, obj.updates_gm),
                              outputs=(obj.out_gm),
                              flowtable=[obj.tiling_gm],
                              config=opt_config)
    return obj.tik_instance
