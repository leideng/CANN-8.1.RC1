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
prelu
"""
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tvm
from impl.util import util_select_op_base
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=locally-disabled,too-many-branches,too-many-statements,invalid-name,unused-argument,too-many-locals
def op_select_format(x, weight, y, kernel_name="prelu"):
    """ calculating data

    Parameters
    ----------
    x : dict
        shape and dtype of x
    weight : dict
        shape and dtype of weight, should be same type as x
    y : dict
        shape and dtype of y, should be same shape and type as x
    kernel_name : str
        kernel name, default value is "prelu"
    Returns
    -------
    None.
    """

    x_shape = x.get("ori_shape")
    weight_shape = weight.get("ori_shape")
    weight_ori_format = weight.get("ori_format")

    product_version = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if product_version in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        dtype_base = ["float16"]
        format_x = ["NCHW", "ND"]
        format_weight = ["ND", "ND"]
        dtype_base_out = ["float16", "float16"]
    else:
        dtype_base = ["float16", "float"]
        format_x = ["NCHW", "NCHW", "ND", "ND"]
        format_weight = ["ND", "ND", "ND", "ND"]
        dtype_base_out = ["float16", "float", "float16", "float"]

    dtype_base_out, format_weight, format_x = get_fz_support_scene(dtype_base, dtype_base_out, format_weight, format_x,
                                                                   weight_shape, x_shape)
    dtype_base_out, format_weight, format_x = get_nc1hwc0_support_scene(dtype_base, dtype_base_out, format_weight,
                                                                        format_x, weight_ori_format, weight_shape)
    dtype_base_out, format_weight, format_x = get_ndc1hwc0_support_scene(dtype_base, dtype_base_out, format_weight,
                                                                         format_x, weight_shape)

    dtype_str = ','.join(dtype_base_out)
    format_x_str = ','.join(format_x)
    format_weight_str = ','.join(format_weight)

    input0 = util_select_op_base.gen_param(classify="input0",
                                           name="x",
                                           datatype=dtype_str,
                                           format=format_x_str,
                                           unknownshape_format=format_x_str)
    input1 = util_select_op_base.gen_param(classify="input1",
                                           name="weight",
                                           datatype=dtype_str,
                                           format=format_weight_str,
                                           unknownshape_format=format_weight_str)
    output0 = util_select_op_base.gen_param(classify="output0",
                                            name="y",
                                            datatype=dtype_str,
                                            format=format_x_str,
                                            unknownshape_format=format_x_str)
    param_list = [input0, input1, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def get_fz_support_scene(dtype_base, dtype_base_out, format_weight, format_x, weight_shape, x_shape):
    if len(x_shape) >= 2 and len(weight_shape) >= 2 and \
            x_shape[-1] == weight_shape[-1] and x_shape[-2] == weight_shape[-2]:
        dtype_base_out = dtype_base_out + dtype_base
        format_x = format_x + ["FRACTAL_NZ"] * len(dtype_base)
        format_weight = format_weight + ["FRACTAL_NZ"] * len(dtype_base)
    if len(x_shape) == 2 and len(weight_shape) == 1:
        if tbe_platform.api_check_support("tik.vcopy"):
            format_weight = format_weight + ["NC1HWC0"] * len(dtype_base)
        else:
            format_weight = format_weight + ["ND"] * len(dtype_base)
        dtype_base_out = dtype_base_out + dtype_base
        format_x = format_x + ["FRACTAL_NZ"] * len(dtype_base)
    return dtype_base_out, format_weight, format_x


def get_ndc1hwc0_support_scene(dtype_base, dtype_base_out, format_weight, format_x, weight_shape):
    if len(weight_shape) == sum(weight_shape):
        dtype_base_out = dtype_base_out + dtype_base
        format_x = format_x + ["NDC1HWC0"] * len(dtype_base)
        format_weight = format_weight + ["ND"] * len(dtype_base)
    elif not (len(weight_shape) == 4 and weight_shape[-1] == 1 and len(weight_shape) != sum(weight_shape)):
        dtype_base_out = dtype_base_out + dtype_base
        format_x = format_x + ["NDC1HWC0"] * len(dtype_base)
        format_weight = format_weight + ["NDC1HWC0"] * len(dtype_base)
    return dtype_base_out, format_weight, format_x


def get_nc1hwc0_support_scene(dtype_base, dtype_base_out, format_weight, format_x, weight_ori_format, weight_shape):
    if len(weight_shape) == sum(weight_shape):
        dtype_base_out = dtype_base_out + dtype_base
        format_x = format_x + ["NC1HWC0"] * len(dtype_base)
        format_weight = format_weight + ["ND"] * len(dtype_base)
    elif not (len(weight_shape) == 3 and weight_shape[-1] == 1 and len(weight_shape) != sum(weight_shape)
              and weight_ori_format != "NCHW"):
        dtype_base_out = dtype_base_out + dtype_base
        format_x = format_x + ["NC1HWC0"] * len(dtype_base)
        format_weight = format_weight + ["NC1HWC0"] * len(dtype_base)
    return dtype_base_out, format_weight, format_x


# 'pylint: disable=unused-variable,too-many-branches,too-many-statements
def reshape(tensor_in, new_shape):
    """
    :params:
    :input: tensor to be reshaped
    :new_shape: shape after input tensor reshaped
    :return: reshape tensor
    """
    def _nd2nz_compute(tensor, indices):
        axis_0, axis_1, axis_2, axis_3 = indices
        return tensor(0, 0, 0, axis_0 * 16 + axis_3)

    return tvm.compute(new_shape, lambda *indices: _nd2nz_compute(tensor_in, indices), name='reshape')


# 'pylint: disable=unused-argument
@register_operator_compute("prelu", op_mode="static", support_fusion=True)
def prelu_compute(input_x, weight_input, output_y, kernel_name="prelu"):
    """
    calculating data

    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    output_y : dict
        dict of output_y, include keys(shape and dtype)
    weight_input : TVM tensor
        the placeholder of weight_input
    kernel_name : str
        kernel name, default value is "prelu"

    Returns
    -------
    output tensor
    """
    shape_x = shape_util.shape_to_list(input_x.shape)
    shape_weight = shape_util.shape_to_list(weight_input.shape)
    if input_x.dtype == "float16":
        scalar_zero = tvm.const(0, dtype="float16")
    else:
        scalar_zero = tvm.const(0, dtype="float32")
    if "format" in input_x.op.attrs and "format" in weight_input.op.attrs:
        format_x = input_x.op.attrs["format"]
        format_weight = weight_input.op.attrs["format"]
        if format_x == "FRACTAL_NZ" and format_weight != "FRACTAL_NZ" and shape_weight[-1] != 1:
            target_shape = [1] * len(shape_x)
            if sum(shape_weight) != 1:
                target_shape[0] = shape_x[0]
                target_shape[-1] = shape_x[-1]
            weight_input = reshape(weight_input, target_shape)
            shape_weight = target_shape
    if list(shape_x) != list(shape_weight):
        shape_list = shape_util.broadcast_shapes(shape_x,
                                                 shape_weight,
                                                 param_name_input1="input_x",
                                                 param_name_input2="weight_input")
        input_x = tbe.broadcast(input_x, shape_list[2], input_x.dtype)
        weight_input = tbe.broadcast(weight_input, shape_list[2], input_x.dtype)
    val_max = tbe.vmaxs(input_x, scalar_zero)
    val_min = tbe.vmins(input_x, scalar_zero)
    val_prod = tbe.vmul(val_min, weight_input)
    res = tbe.vadd(val_max, val_prod)
    res.op.attrs["weight_input"] = weight_input

    return res


# 'pylint: disable=too-many-locals,invalid-name
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def prelu(input_x, input_A, output_y, kernel_name="prelu"):
    """
    calculating data

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    input_A : dict
        shape and dtype of input_A, should be same type as input_x
    output_y : dict
        shape and dtype of output, should be same shape and type as input_x
    kernel_name : str
        kernel name, default value is "prelu"
    Returns
    -------
    None
    """
    shape = input_x.get("shape")
    dtype = input_x.get("dtype")
    input_format = input_x.get("format")
    input_dtype = dtype.lower()
    para_check.check_shape(shape, param_name="x")

    check_list = ("float16", "float32")

    para_check.check_dtype(input_dtype, check_list, param_name="x")
    weight_shape = input_A.get("shape")
    weight_dtype = input_A.get("dtype").lower()
    para_check.check_shape(weight_shape, param_name="weight")
    para_check.check_dtype(weight_dtype, check_list, param_name="weight")

    if weight_dtype != input_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal('prelu', 'input_dtype', 'weight_dtype', str(weight_dtype),
                                                              str(input_dtype))

    weight_dim = len(weight_shape)
    feature_dim = len(shape)

    if input_format == "NC1HWC0":
        if weight_dim != 5:
            if weight_dim == 1 and weight_shape[0] == 1:
                weight_shape_new = [1] * 5
            else:
                shape_list = shape_util.broadcast_shapes(shape,
                                                         weight_shape,
                                                         param_name_input1="input_x",
                                                         param_name_input2="input_A")
                shape, weight_shape_new = shape_util.refine_shapes_for_broadcast(shape_list[0], shape_list[1])
        else:
            weight_shape_new = list(weight_shape)
    elif input_format == "NDC1HWC0":
        if weight_dim != 6:
            if weight_dim == 1 and weight_shape[0] == 1:
                weight_shape_new = [1] * 6
            else:
                shape_list = shape_util.broadcast_shapes(shape,
                                                         weight_shape,
                                                         param_name_input1="input_x",
                                                         param_name_input2="input_A")
                shape, weight_shape_new = shape_util.refine_shapes_for_broadcast(shape_list[0], shape_list[1])
        else:
            weight_shape_new = list(weight_shape)
    elif input_format == "FRACTAL_NZ":
        if weight_dim == 1 and weight_shape[0] == 1:
            weight_shape_new = [1] * feature_dim
        elif weight_dim == 1:
            weight_shape_new = [1] * feature_dim
            weight_shape_new[0] = shape[0]
            weight_shape_new[-1] = shape[-1]
        else:
            shape_list = shape_util.broadcast_shapes(shape,
                                                     weight_shape,
                                                     param_name_input1="input_x",
                                                     param_name_input2="input_A")
            shape, weight_shape_new = shape_util.refine_shapes_for_broadcast(shape_list[0], shape_list[1])
    elif input_format == "NHWC" and feature_dim == 4:
        if (weight_dim == 1 and weight_shape[0] != shape[-1] and weight_shape[0] != 1) or (weight_dim not in (1, 3)):
            shape_list = shape_util.broadcast_shapes(shape,
                                                     weight_shape,
                                                     param_name_input1="input_x",
                                                     param_name_input2="input_A")
            shape, weight_shape_new = shape_util.refine_shapes_for_broadcast(shape_list[0], shape_list[1])
        elif weight_dim == 1:
            weight_shape_new = [1] * feature_dim
            weight_shape_new[3] = weight_shape[0]
        else:
            weight_shape_new = list(weight_shape)
            weight_shape_new.insert(0, 1)
    elif feature_dim == 1:
        if weight_shape[0] != 1 or weight_dim != 1:
            shape_list = shape_util.broadcast_shapes(shape,
                                                     weight_shape,
                                                     param_name_input1="input_x",
                                                     param_name_input2="input_A")
            shape, weight_shape_new = shape_util.refine_shapes_for_broadcast(shape_list[0], shape_list[1])
        else:
            weight_shape_new = [1]
    # `input_x:DIM = 2,3,4,5,6,7...`
    else:
        if (weight_shape[0] != shape[1] and weight_shape[0] != 1) or (weight_dim not in (1, feature_dim - 1)):
            shape_list = shape_util.broadcast_shapes(shape,
                                                     weight_shape,
                                                     param_name_input1="input_x",
                                                     param_name_input2="input_A")
            shape, weight_shape_new = shape_util.refine_shapes_for_broadcast(shape_list[0], shape_list[1])
        elif weight_dim == 1:
            weight_shape_new = [1] * feature_dim
            weight_shape_new[1] = weight_shape[0]
        else:
            weight_shape_new = list(weight_shape)
            weight_shape_new.insert(0, 1)
    if len(weight_shape_new) == sum(weight_shape_new):
        weight_shape_new = [1]
        total_calc_num = 1
        for i, _ in enumerate(shape):
            total_calc_num = total_calc_num * shape[i]
        shape_new = [total_calc_num]
        data_input = tvm.placeholder(shape_new, name="data_input", dtype=input_dtype)
        weight_input = tvm.placeholder(weight_shape_new, name="weight_input", dtype=input_dtype)
    else:
        data_input = tvm.placeholder(shape, name="data_input", dtype=input_dtype)
        weight_input = tvm.placeholder(weight_shape_new, name="weight_input", dtype=input_dtype)

    res = prelu_compute(data_input, weight_input, output_y, kernel_name)

    with tvm.target.cce():
        sch = auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_input, weight_input, res]}

    build(sch, config)
