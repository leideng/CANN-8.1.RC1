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
fused_mul_add
"""

from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.util_compute import batchmatmul_elem_nd2nz
from impl.util.util_compute import batchmatmul_elem_reshape
from impl.util.util_compute import check_batchmatmul_fuse
from impl.util.platform_adapter import error_manager_vector
from impl.dynamic.fused_mul_add import get_op_support_info as static_get_op_support_info
from impl.dynamic.fused_mul_add import op_select_format as static_op_select_format
from impl.dynamic.fused_mul_add import _infer_shape_one
from impl.dynamic.fused_mul_add import _infer_shape_two


# 'pylint: disable=locally-disabled,unused-variable,unused-argument
# 'pylint: disable=locally-disabled,too-many-locals,too-many-statements
# 'pylint: disable=locally-disabled,too-many-branches,unused-variable
def get_op_support_info(input0, input1, input2, output,
                        kernel_name="fused_mul_add"):
    """
    get_op_support_info
    """
    return static_get_op_support_info(input0, input1, input2, output,
                                      kernel_name)


def op_select_format(input0, input1, input2, output,
                     kernel_name="fused_mul_add"):
    """
    _division_sixteen : judge whether the last two dimensions are divided by 16
    scalar2tensor_one : convert scalar to tensor
    """
    return static_op_select_format(input0, input1, input2, output,
                                      kernel_name)


def check_format(format_input0, format_input1, format_input2):
    """
    check the format_list
    """
    list_format = [format_input0, format_input1, format_input2]
    nd_format = {"ND", "NHWC", "NCHW", "HWCN"}
    standard_format = []
    for item in list_format:
        if item in nd_format:
            standard_format.append("ND")
        else:
            standard_format.append(item)

    list_pattern = [["FRACTAL_NZ", "ND", "ND"],
                    ["ND", "FRACTAL_NZ", "ND"],
                    ["ND", "ND", "FRACTAL_NZ"],
                    ["FRACTAL_NZ", "ND", "FRACTAL_NZ"],
                    ["ND", "FRACTAL_NZ", "FRACTAL_NZ"],
                    ]
    if standard_format in list_pattern:
        format_pattern = list_pattern.index(standard_format) + 1
    else:
        format_pattern = 0

    return format_pattern


def check_ori_shape(input0, input1, input2):
    """
    check the ND shapes whether they can be broadcasted
    """
    shape_0 = list(shape_util.scalar2tensor_one(input0.get("ori_shape")))
    shape_1 = list(shape_util.scalar2tensor_one(input1.get("ori_shape")))
    shape_2 = list(shape_util.scalar2tensor_one(input2.get("ori_shape")))
    shape_input0, shape_input1, shape_max_mul = \
        shape_util.broadcast_shapes(shape_0, shape_1, param_name_input1="input0",
                                    param_name_input2="input1")
    shape_input2, shape_max_mul, shape_max_add0 = \
        shape_util.broadcast_shapes(shape_0, shape_2, param_name_input1="input0",
                                    param_name_input2="input2")



def shape_broadcast(data_1, data_2):
    """
    broadcast the two input

    Parameters
    ----------
    data_1: TVM tensor
        the placeholder of first input data
    data_2: TVM tensor
        the placeholder of second input data
    output_z: dict
        shape and dtype of output, should be broadcast shape and type as input

    Returns
    -------
    res : output of the data's divide
    """
    shape_x = shape_util.shape_to_list(data_1.shape)
    shape_y = shape_util.shape_to_list(data_2.shape)
    if shape_x != shape_y:
        shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x, shape_y,
                                                                  param_name_input1="data_1",
                                                                  param_name_input2="data_2")
        data_1 = tbe.broadcast(data_1, shape_max)
        data_2 = tbe.broadcast(data_2, shape_max)

    return data_1, data_2


@register_operator_compute("fused_mul_add", op_mode="static", support_fusion=True)
def fusion_mul_add_compute(data_input0, data_input1, data_input2,
                           output, kernel_name="fused_mul_add"):
    """
    mul+add calculation function for ub fusion

    Parameters
    ----------
    data_input0: TVM tensor
         the input tensor of mul
    data_input1: TVM tensor
         the input tensor of mul
    data_input2: TVM tensor
         the input tensor of add
    output: TVM tensor
         the output tensor of add
    kernel_name : str
        kernel name, default value is "fused_mul_add"

    Returns
    -------
    output tensor
    """
    shape_0 = shape_util.shape_to_list(data_input0.shape)
    shape_1 = shape_util.shape_to_list(data_input1.shape)
    batch_matmul_flag_lhs = check_batchmatmul_fuse(data_input0)
    batch_matmul_flag_rhs = check_batchmatmul_fuse(data_input1)

    if batch_matmul_flag_rhs:
        data_input0, data_input1 = data_input1, data_input0
    if "para_name" in data_input0.op.attrs:
        para_name = data_input0.op.attrs["para_name"].value
        para_name += "_muladd"
    else:
        para_name = "muladd"
    batch_shape = shape_util.shape_to_list(data_input0.op.attrs["batch_shape"])
    para_dict_1 = {"format_elem": data_input1.op.attrs["format"],
                   "batch_shape": batch_shape}
    para_dict_2 = {"format_elem": data_input2.op.attrs["format"],
                   "batch_shape": batch_shape}

    if batch_matmul_flag_lhs or batch_matmul_flag_rhs:
        data_input1, shape_max = batchmatmul_elem_nd2nz(data_input0, data_input1, para_dict_1, para_name + "1")
        data_input2, _ = batchmatmul_elem_nd2nz(data_input0, data_input2, para_dict_2, para_name + "2")
        data_input1 = tbe.broadcast(data_input1, shape_max)
        data_input2 = tbe.broadcast(data_input2, shape_max)
        data_input1 = batchmatmul_elem_reshape(data_input0, data_input1, batch_shape, para_name + "1")
        data_input2 = batchmatmul_elem_reshape(data_input0, data_input2, batch_shape, para_name + "2")
        mul_result = tbe.vmul(data_input0, data_input1)
        res = tbe.vadd(mul_result, data_input2)
        res.op.attrs["batch_shape"] = batch_shape
        res.op.attrs["para_name"] = para_name
    else:
        res = fused_mul_add_compute(data_input0, data_input1, data_input2, output, kernel_name)

    return res


def fused_mul_add_compute(data_input0, data_input1, data_input2,
                          output, kernel_name="fused_mul_add"):
    """
    mul+add calculation function

    Parameters
    ----------
    data_input0: TVM tensor
         the input tensor of mul
    data_input1: TVM tensor
         the input tensor of mul
    data_input2: TVM tensor
         the input tensor of add
    output: TVM tensor
         the output tensor of add
    kernel_name : str
        kernel name, default value is "fuesd_mul_add"

    Returns
    -------
    output tensor
    """

    # mul
    data_input0, data_input1 = shape_broadcast(data_input0, data_input1)
    mul_result = tbe.vmul(data_input0, data_input1)

    # add
    mul_result, data_input2 = shape_broadcast(mul_result, data_input2)
    res = tbe.vadd(mul_result, data_input2)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def fused_mul_add(input0, input1, input2,
                  output, kernel_name="fused_mul_add"):
    """
    function: fused for mul+add

    Parameters
    ----------
    input0: dict
         the dict of input of mul, support float16,float32,int32
    input1: dict
         the dict of input of mul, support float16,float32,int32
    input2: dict
         the dict of input of add, support float16,float32,int32
    output: dict
         the dict of output of add, support float16,float32,int32
    kernel_name: str
        cce kernel name, default value is fused_mul_add

    Returns
    -------
    None
    """
    shape_input0 = list(shape_util.scalar2tensor_one(input0.get("shape")))
    shape_input1 = list(shape_util.scalar2tensor_one(input1.get("shape")))
    shape_input2 = list(shape_util.scalar2tensor_one(input2.get("shape")))

    dtype_input0 = input0.get("dtype").lower()
    dtype_input1 = input1.get("dtype").lower()
    dtype_input2 = input2.get("dtype").lower()

    format_input0 = input0.get("format").upper()
    format_input1 = input1.get("format").upper()
    format_input2 = input2.get("format").upper()

    check_ori_shape(input0, input1, input2)
    format_pattern = check_format(format_input0, format_input1, format_input2)
    if format_pattern in [1, 2, 3]:
        shape_input0, shape_input1, shape_input2 = \
            _infer_shape_one(shape_input0, shape_input1,
                             shape_input2, format_pattern)
    elif format_pattern == 4:
        shape_input0, shape_input1, shape_input2 = \
            _infer_shape_two(shape_input0, shape_input1,
                             shape_input2, format_pattern)
    elif format_pattern == 5:
        shape_input1, shape_input0, shape_input2 = \
            _infer_shape_two(shape_input1, shape_input0,
                             shape_input2, format_pattern)
    else:
        shape_input0, shape_input1, shape_max_mul = \
            shape_util.broadcast_shapes(shape_input0, shape_input1, param_name_input1="input0",
                                        param_name_input2="input1")
        shape_input2, shape_max_mul, shape_max_add0 = \
            shape_util.broadcast_shapes(shape_input2, shape_max_mul, param_name_input1="input2",
                                        param_name_input2="shape_max_mul")

    data_input0 = tvm.placeholder(shape_input0,
                                  name="data_input0",
                                  dtype=dtype_input0)
    data_input1 = tvm.placeholder(shape_input1,
                                  name="data_input1",
                                  dtype=dtype_input1)
    data_input2 = tvm.placeholder(shape_input2,
                                  name="data_input2",
                                  dtype=dtype_input2)

    res = fused_mul_add_compute(data_input0, data_input1, data_input2,
                                output, kernel_name)

    with tvm.target.cce():
        sch = auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": (data_input0, data_input1, data_input2, res)}

    build(sch, config)
