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
apply_gradient_descent
"""
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util import util_apply_op_schedule


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
@register_operator_compute("apply_gradient_descent", op_mode="static", support_fusion=True)
def apply_gradient_descent_compute(var,
                                   alpha,
                                   delta,
                                   out,
                                   kernel_name="apply_gradient_descent"):
    """
    compute out_var = var - alpha * delta

    Parameters:
    ----------
    var: the placeholder of var.

    alpha : the placeholder of alpha.

    delta : the placeholder of delta.

    out : the dict of output.

    kernel_name :  cce kernel name, default value is "apply_gradient_descent".

    Returns
    -------
    out
    """

    # step 1: calculate delta * alpha
    var_change = tvm.compute(delta.shape,
                             lambda *indices: delta(*indices) * alpha[0],
                             tag='elewise_single_VS_mul')
    # step 2: calculate var - delta * alpha
    reuse_var = tbe.vsub(var, var_change)

    def _compute(*index):
        return reuse_var(*index), reuse_var(*index)

    return tvm.compute(var.shape, _compute, name="outputs")


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def apply_gradient_descent(var,
                           alpha,
                           delta,
                           out,
                           kernel_name="apply_gradient_descent"):
    """
    Update var by subtracting alpha * delta from it.

    var_{t} = var_{t-1} - alpha * delta

    Parameters:
    ----------
    var: dict of input_var, include shape and dtype,
        dtype support float16, float32.

    alpha : dict of input_alpha, include shape and dtype,
        dtype support float16, float32.
        Must have the same type as 'var', Must have the shape(1,).

    delta : dict of input_delta, include shape and dtype,
        dtype support float16, float32.
        Must have the same shape and dtype as input_var.

    out : dict of output, include shape and dtype.

    kernel_name : cce kernel name, default value is "apply_gradient_descent".

    Returns
    -------
    None
    """

    input_dict = (var, alpha, delta)

    args = util_apply_op_schedule.ApplyOpConfig.TensorArgs(input_dict, apply_gradient_descent_compute, out, 1.5)
    name = util_apply_op_schedule.ApplyOpConfig.TensorName(all=("var", "alpha", "delta"),
                                                           scalar=("alpha", ),
                                                           reuse=("var", ))

    util_apply_op_schedule.common_apply_op_process(util_apply_op_schedule.ApplyOpConfig(args, name), kernel_name)
