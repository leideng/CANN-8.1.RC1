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
truncate_mod
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from te.utils import para_check
from te.utils import shape_util
from tbe import tvm


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals
@tbe_platform.fusion_manager.fusion_manager.register("truncate_mod")
def truncate_mod_compute(input_x, input_y, output_z,
                         kernel_name="truncate_mod"):
    """
    truncate_mod compute
    calculating data's truncatemod, res = x - truncate(x/y)*y

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input_x
    input_y: TVM tensor
        the placeholder of input_y
    output_z: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        kernel name, default value is "truncate_mod"

    Returns
    -------
    res: TVM tensor
        the result of truncate_mod(input_x,input_y)
    """
    input_data_x = shape_util.shape_to_list(input_x.shape)
    input_data_y = shape_util.shape_to_list(input_y.shape)
    shape_list = shape_util.broadcast_shapes(input_data_x, input_data_y,
                                             param_name_input1="input_x",
                                             param_name_input2="input_y")
    dtype = input_x.dtype
    tran_x = tbe.cast_to(input_x, "float32")
    tran_y = tbe.cast_to(input_y, "float32")
    data_x_broad = tbe.broadcast(tran_x, shape_list[2])
    data_y_broad = tbe.broadcast(tran_y, shape_list[2])

    vdiv_data = tbe.vdiv(data_x_broad, data_y_broad)
    truncate_data = tbe.cast_to(vdiv_data, "int32")
    cast_data = tbe.cast_to(truncate_data, "float32")
    mul_data = tbe.vmul(cast_data, data_y_broad)
    sub_data = tbe.vsub(data_x_broad, mul_data)
    res = tbe.cast_to(sub_data, dtype)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def truncate_mod(input_x, input_y, output_z, kernel_name="truncate_mod"):
    """
    algorithm: truncatemod
    calculating data's truncate, res = x - truncate(x/y)*y

    Parameters
    ----------
    input_x: dict
        dict with keys(shape and dtype) of input_x
    input_y: dict
        dict with keys(shape and dtype) of input_y
    output_div: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        kernel name, default value is "truncatemod"

    Returns
    -------
    None
    """
    shape_x = input_x.get("shape")
    dtype_x = input_x.get("dtype").lower()
    shape_y = input_y.get("shape")
    dtype_y = input_y.get("dtype").lower()

    para_check.check_shape(shape_x, param_name="input_x")
    para_check.check_shape(shape_y, param_name="input_y")

    shape_list = shape_util.broadcast_shapes(shape_x, shape_y,
                                             param_name_input1="input_x",
                                             param_name_input2="input_y")
    check_list = ("float16", "float32", "int8", "uint8", "int32")
    para_check.check_dtype(dtype_x, check_list, param_name="input_x")
    para_check.check_dtype(dtype_y, check_list, param_name="input_y")

    shape_x, shape_y = shape_util.refine_shapes_for_broadcast(shape_list[0],
                                                              shape_list[1])
    data_x = tvm.placeholder(shape_x, dtype=dtype_x, name="data_x")
    data_y = tvm.placeholder(shape_y, dtype=dtype_y, name="data_y")
    res = truncate_mod_compute(data_x, data_y, output_z, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_x, data_y, res]}
    tbe.cce_build_code(sch, config)
