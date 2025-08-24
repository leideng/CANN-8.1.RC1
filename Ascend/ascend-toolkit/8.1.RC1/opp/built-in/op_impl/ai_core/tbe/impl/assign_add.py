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
assignadd
"""
import functools

from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_build
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=too-many-locals,invalid-name,unused-argument,unused-variable
def _assign_add_int64_schedule(res, tensor_ref, tensor_val, res_add):
    """
    assignadd int64 schedule

    Parameters
    ----------
    res: result of compute
    tensor_val: tensor val

    Returns
    -------
    output sch
    """
    def _ceil(m, n):
        return (m + n - 1) // n

    def _tiling(shape, dtype):
        ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        dtype_bytes_size = tbe_platform.get_bit_len(dtype) // 8
        # only use 1/2 ub
        total_ele = ub_size_bytes // dtype_bytes_size // 2
        # 2 input ub and 1 output ub
        total_ele = total_ele // 3
        core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        # 1 block is 32B
        block_ele = 32 // dtype_bytes_size

        fused_axis_factor = shape[0]
        if fused_axis_factor >= core_num:
            fused_axis_factor = _ceil(fused_axis_factor, core_num)
            fused_axis_factor = _ceil(fused_axis_factor, block_ele) * block_ele
        total_ele = ((total_ele + block_ele) // block_ele) * block_ele
        fused_factor = min(fused_axis_factor, total_ele)
        return fused_axis_factor, fused_factor

    # set ub
    tensor_input = tensor_ref
    tensor_add = tensor_val
    x_shape = [i.value for i in tensor_input.shape]
    core_factor, ub_factor = _tiling(x_shape, tensor_input.dtype)
    sch = tvm.create_schedule(res.op)
    tensor_input_in_ub = sch.cache_read(tensor_input, tbe_platform.scope_ubuf, [res_add])
    tensor_add_in_ub = sch.cache_read(tensor_add, tbe_platform.scope_ubuf, [res_add])
    sch[res_add].set_scope(tbe_platform.scope_ubuf)

    # set axis info
    axis_core_out, axis_core_in = sch[res].split(res.op.axis[0], core_factor)
    axis_ub_out, axis_ub_in = sch[res].split(axis_core_in, ub_factor)
    sch[tensor_input_in_ub].compute_at(sch[res], axis_ub_out)
    sch[tensor_add_in_ub].compute_at(sch[res], axis_ub_out)
    sch[res_add].compute_at(sch[res], axis_ub_out)

    # set ping pong
    sch[tensor_input_in_ub].preload()
    sch[tensor_add_in_ub].preload()
    sch[tensor_input_in_ub].double_buffer()
    sch[tensor_add_in_ub].double_buffer()
    sch[res_add].double_buffer()

    # set multi cores
    block = tvm.thread_axis("blockIdx.x")
    sch[res].bind(axis_core_out, block)

    # set emit_insn
    sch[tensor_input_in_ub].emit_insn(tensor_input_in_ub.op.axis[0], tbe_platform.DMA_COPY)
    sch[tensor_add_in_ub].emit_insn(tensor_add_in_ub.op.axis[0], tbe_platform.DMA_COPY)
    sch[res_add].emit_insn(res_add.op.axis[0], 'reg_add')
    sch[res].emit_insn(axis_ub_in, tbe_platform.DMA_COPY)
    return sch


# 'pylint: disable=locally-disabled,too-many-locals,unnecessary-lambda
# 'pylint: disable=locally-disabled,too-many-statements
@register_operator_compute("assign_add", op_mode="static", support_fusion=True)
def _compute_assign_add(tensor_x, tensor_y, output, kernel_name='assign_add'):
    """
    assignadd compute function for int8, uint8, int32, float16, float32

    Parameters
    ----------
    tensor_x : list or tuple
        shape of ref.
    tensor_y : list or tuple
        shape of val.
    dtype : str
        the data type.

    Returns
    -------
    res: tvm.tensor
        tensor of result
    """
    res = tbe.vadd(tensor_x, tensor_y)
    return res


def _check_param(shape_ref, shape_val, dtype_ref, dtype_value, kernel_name):
    """
    check param

    Parameters
    ----------
    shape_ref : shape
        shape of ref.
    shape_val : shape
        shape of val.
    dtype : str
        the data type.
    kernel_name : kernel name

    Returns
    -------
    None
    """
    para_check.check_shape(shape_ref, param_name="ref")
    para_check.check_shape(shape_val, param_name="value")
    check_list = ("float16", "float32", "int8", "uint8", "int32", "int64")
    para_check.check_dtype(dtype_ref, check_list, param_name="ref")
    para_check.check_dtype(dtype_value, check_list, param_name="value")
    if shape_ref != shape_val:
        error_detail = "Shape of ref and value should be same."
        error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "ref", "value", error_detail)
    if dtype_ref != dtype_value:
        error_detail = "Dtype of ref and value should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "ref", "value", error_detail)


# 'pylint: disable=locally-disabled,too-many-arguments, unused-argument
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def assign_add(ref, value, output, kernel_name="assign_add"):
    """
    algorithm: assign_add
    update ref by adding value to it
    calculating data's add, a = a + b

    Parameters
    ----------
    ref: dict
        dict of input_ref, include shape and dtype,
    value: dict
        dict of input_value, include shape and dtype,
        Must have the same shape and dtype as input_ref
    output: dict
        dict of output
    kernel_name : str
        cce kernel name, default value is assign_add

    Returns
    -------
    None
    """
    shape_ref = shape_util.scalar2tensor_one(ref.get("shape"))
    shape_val = shape_util.scalar2tensor_one(value.get("shape"))
    dtype_ref = ref.get("dtype").lower()
    dtype_value = value.get("dtype").lower()

    # check if the parameter is valid
    _check_param(shape_ref, shape_val, dtype_ref, dtype_value, kernel_name)

    fused_shape = [functools.reduce(lambda x, y: x * y, shape_ref[:])]

    tensor_ref = tvm.placeholder(fused_shape, dtype=dtype_ref, name="tensor_ref")
    tensor_val = tvm.placeholder(fused_shape, dtype=dtype_value, name="tensor_val")
    if dtype_ref == "int64":
        # process the data of int64
        res_add = tvm.compute(fused_shape, lambda *i: tensor_ref(*i) + tensor_val(*i), name='res_add')
        res = tvm.compute(fused_shape, lambda *i: res_add(*i), name='res')
        sch = _assign_add_int64_schedule(res, tensor_ref, tensor_val, res_add)
        with tbe_build.build_config():
            tvm.build(sch, [tensor_ref, tensor_val, res], "cce", name=kernel_name)
    else:
        res = _compute_assign_add(tensor_ref, tensor_val, output, kernel_name="assign_add")
        with tvm.target.cce():
            sch = auto_schedule(res)
        config = {"name": kernel_name, "tensor_list": [tensor_ref, tensor_val, res]}
        build(sch, config)
