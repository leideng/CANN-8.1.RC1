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
Activation Universal Linear Quant Clamp Min Grad
"""


import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.constant_util import SHAPE_SIZE_LIMIT


# 'pylint: disable=unused-argument
@register_operator_compute('act_ulq_clamp_min_grad', op_mode="static", support_fusion=True)
def act_ulq_clamp_min_grad_compute(
    y_grad, clamp_min_mask, x_clamped_loss, output, kernel_name='act_ulq_clamp_min_grad'):
    """
    Function: Calculate the gradient of minimum clamped value.

    Parameters:
    ----------
    y_grad: the placeholder of gradient

    clamp_min_mask : the placeholder of clamp_min_mask

    x_clamped_loss : the placeholder of x_clamped_loss

    output: the dict of output

    kernel_name: cce kernel name, default value is "act_ulq_clamp_min_grad"

    Returns : A Tensor with float32 and (1,).
    -------
    """
    shape = y_grad.shape
    axis = list(range(len(shape)))
    signal = tbe.vsel(clamp_min_mask, tvm.const(0, 'float16'), tvm.const(1, 'float16'))
    signal = tbe.cast_to(signal, 'float32')
    x_clamped_loss = tbe.cast_to(x_clamped_loss, 'float32')
    x_min_grad = tbe.vsub(signal, x_clamped_loss)
    y_grad = tbe.cast_to(y_grad, 'float32')
    clamp_min_grad = tbe.vmul(y_grad, x_min_grad)
    clamp_min_grad = tbe.sum(clamp_min_grad, axis)

    return clamp_min_grad


@para_check.check_input_type(dict, dict, dict, dict, str)
def act_ulq_clamp_min_grad(input_x, input_y, input_z, output, kernel_name='act_ulq_clamp_min_grad'):
    """
    ----------
    Parameters:
    ----------
    input_x : the placeholder of y_grad

    input_y : the placeholder of clamp_min_mask

    input_z : the placeholder of x_clamped_loss

    output : the dict of clamp_min_grad

    Returns : None
    ----------
    """
    shape = input_x.get('shape')
    dtype = input_x.get('dtype')

    para_check.check_shape_size(shape, SHAPE_SIZE_LIMIT)
    if shape != input_y.get('shape') or shape != input_z.get('shape'):
        raise ValueError('All input must have the same shape!')
    if input_y.get('dtype') != 'bool' and input_y.get('dtype') != 'int8':
        raise ValueError('The type of "clamp_max_mask" must be "bool or int8"!')

    y_grad = tvm.placeholder(shape, dtype, 'y_grad')
    clamp_min_mask = tvm.placeholder(shape, 'bool', 'clamp_min_mask')
    x_clamped_loss = tvm.placeholder(shape, dtype, 'x_clamped_loss')

    res = act_ulq_clamp_min_grad_compute(y_grad, clamp_min_mask, x_clamped_loss, output, kernel_name)

    with tvm.target.cce():
        auto_sch = tbe.auto_schedule(res)

    config = {'name': kernel_name,
              'print_ir': False,
              'tensor_list': (y_grad, clamp_min_mask, x_clamped_loss, res),
              'bool_storage_as_1bit': False}

    tbe.cce_build_code(auto_sch, config)
