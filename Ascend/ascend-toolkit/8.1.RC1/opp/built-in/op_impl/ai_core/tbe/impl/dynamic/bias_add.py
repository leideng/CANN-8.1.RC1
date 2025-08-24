# Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
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
bias_add
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import error_manager_vector
from impl.util import util_select_op_base
from impl.util.util_common import is_unknown_rank_input
from impl.util.util_compute import only_static_support
from impl.util.util_soc_common import is_v220


# 'pylint: disable=too-many-statements,too-many-branches,invalid-name,too-many-locals,unused-argument
def op_select_format(x, bias, y, data_format="NHWC", kernel_name="bias_add"):
    """
    1.when the length of x's ori_shape is less than or equal
    to 4 and the first element of the shape of bias is a multiple
    of 16. The Op BiasAdd can support NC1HWC0, NCHW and NHWC.

        for example:
        x : Tensor of (shape=(16, 16, 16, 16), "NHWC")
        bias : Tensor of (shape=(16, 16, 16, 16), "NHWC")
        The Op BiasAdd can process with NC1HWC0
        x : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")
        bias : Tensor of (shape=(2), "ND")

    2.when the length of x's ori_shape is greater then 4 and
    the first element of the shape of bias is a multiple of 16.
    The Op BiasAdd can support NDHWC, NCDHW, NDC1HWC0.

        x : Tensor of (shape=(16, 1, 16, 16, 16), "NDHWC")
        bias : Tensor of (shape=(2), "ND")
    """
    shape_bias = bias.get("ori_shape")
    ori_shape_x = x.get("ori_shape")
    c0 = 16
    vmuls_support = tbe_platform.api_check_support("tbe.dsl.vmuls", "float32")
    bfp16_support = tbe_platform.intrinsic_check_support("Intrinsic_vconv", "bf162f32")
    if len(ori_shape_x) <= 4:
        if shape_bias[0] % c0 == 0 and len(ori_shape_x) == 4:
            # NC1HWC0+ND ND+ND
            if not vmuls_support and not bfp16_support:
                dtype_list = "float16, int32, float16"
                format_list = "NC1HWC0, ND, ND"
                bias_format_list = "ND, ND, ND"
            elif vmuls_support and bfp16_support:
                dtype_list = "bfloat16, float16, float, int32, bfloat16, float16, float"
                format_list = "NC1HWC0, NC1HWC0, NC1HWC0, ND, ND, ND, ND"
                bias_format_list = "ND, ND, ND, ND, ND, ND, ND"
            else:
                dtype_list = "float16, float, int32, float16, float"
                format_list = "NC1HWC0, NC1HWC0, ND, ND, ND"
                bias_format_list = "ND, ND, ND, ND, ND"
        elif shape_bias[0] % c0 != 0 and len(ori_shape_x) == 4:
            # NC1HWC0+NC1HWC0 ND+ND
            if not vmuls_support and not bfp16_support:
                dtype_list = "float16, int32, float16"
                format_list = "NC1HWC0, ND, ND"
                bias_format_list = "NC1HWC0, ND, ND"
            elif vmuls_support and bfp16_support:
                dtype_list = "bfloat16, float16, float, int32, bfloat16, float16, float"
                format_list = "NC1HWC0, NC1HWC0, NC1HWC0, ND, ND, ND, ND"
                bias_format_list = "NC1HWC0, NC1HWC0, NC1HWC0, ND, ND, ND, ND"
            else:
                dtype_list = "float16, float, int32, float16, float"
                format_list = "NC1HWC0, NC1HWC0, ND, ND, ND"
                bias_format_list = "NC1HWC0, NC1HWC0, ND, ND, ND"
        else:
            # ND+ND
            if not vmuls_support and not bfp16_support:
                dtype_list = "int32, float16"
                format_list = "ND, ND"
                bias_format_list = "ND, ND"
            elif vmuls_support and bfp16_support:
                dtype_list = "int32, float16, float, bfloat16"
                format_list = "ND, ND, ND, ND"
                bias_format_list = "ND, ND, ND, ND"
            else:
                dtype_list = "int32, float16, float"
                format_list = "ND, ND, ND"
                bias_format_list = "ND, ND, ND"

        if is_v220():
            dtype_list = dtype_list + ", int64"
            format_list = format_list + ", ND"
            bias_format_list = bias_format_list + ", ND"
    else:
        if shape_bias[0] % c0 == 0:
            # NDHWC+ND NCDHW+ND NDC1HWC0+ND
            if not vmuls_support and not bfp16_support:
                dtype_list = "int32, float16, int32, float16, int32, float16"
                format_list = "NDHWC, NDHWC, NCDHW, NCDHW, NDC1HWC0, NDC1HWC0"
                bias_format_list = "ND, ND, ND, ND, ND, ND"
            elif vmuls_support and bfp16_support:
                dtype_list = "int32, float16, bfloat16, float, int32, float16, bfloat16, float, int32, \
                                float16, bfloat16, float"
                format_list = "NDHWC, NDHWC, NDHWC, NDHWC, NCDHW, NCDHW, NCDHW, NCDHW, NDC1HWC0, NDC1HWC0, \
                                NDC1HWC0, NDC1HWC0"
                bias_format_list = "ND, ND, ND, ND, ND, ND, ND, ND, ND, ND, ND, ND"
            else:
                dtype_list = "int32, float16, float, int32, float16, float, int32, float16, float"
                format_list = "NDHWC, NDHWC, NDHWC, NCDHW, NCDHW, NCDHW, NDC1HWC0, NDC1HWC0, NDC1HWC0"
                bias_format_list = "ND, ND, ND, ND, ND, ND, ND, ND, ND"

            if is_v220():
                dtype_list = dtype_list + ", int64, int64, int64"
                format_list = format_list + ", NDHWC, NCDHW, NDC1HWC0"
                bias_format_list = bias_format_list + ", ND, ND, ND"
        else:
            # NDHWC+ND NCDHW+ND
            if not vmuls_support and not bfp16_support:
                dtype_list = "int32, float16, int32, float16"
                format_list = "NDHWC, NDHWC, NCDHW, NCDHW"
                bias_format_list = "ND, ND, ND, ND"
            elif vmuls_support and bfp16_support:
                dtype_list = "int32, float16, bfloat16, float, int32, float16, bfloat16, float"
                format_list = "NDHWC, NDHWC, NDHWC, NDHWC, NCDHW, NCDHW, NCDHW, NCDHW"
                bias_format_list = "ND, ND, ND, ND, ND, ND, ND, ND"
            else:
                dtype_list = "int32, float16, float, int32, float16, float"
                format_list = "NDHWC, NDHWC, NDHWC, NCDHW, NCDHW, NCDHW"
                bias_format_list = "ND, ND, ND, ND, ND, ND"

            if is_v220():
                dtype_list = dtype_list + ", int64, int64"
                format_list = format_list + ", NDHWC, NCDHW"
                bias_format_list = bias_format_list + ", ND, ND"

    input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                           datatype=dtype_list,
                                           format=format_list,
                                           unknownshape_format=format_list)
    input1 = util_select_op_base.gen_param(classify="input1", name="bias",
                                           datatype=dtype_list,
                                           format=bias_format_list,
                                           unknownshape_format=bias_format_list)
    output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                            datatype=dtype_list,
                                            format=format_list,
                                            unknownshape_format=format_list)
    param_list = [input0, input1, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def check_equal(a, b):
    """
    check whether a equal to b or not

    Parameters
    ----------
    a : int
    b : int
    Returns
    -------
    res : true or false
    """
    if a != -1 and b != -1 and a != b:
        return False
    return True


@register_operator_compute("BiasAdd", op_mode="dynamic", support_fusion=only_static_support, support_bfp16=True)
def bias_add_compute(x, bias, y, data_format, kernel_name="bias_add"):
    """
    calculating data's bias add

    Parameters
    ----------
    x : tvm tensor
              x data x
    bias : tvm tensor
              x data y
    y : tvm tensor
              y data
    data_format: A string.
                'N...C' and 'NC...' are supported.
    kernel_name : string
                  cce kernel name, default value is "bias_add"

    Returns
    -------
    res : y of the data's bias add
    """
    _, _, shape_max = shape_util.broadcast_shapes(x.shape,
                                                  bias.shape,
                                                  param_name_input1="x",
                                                  param_name_input2="bias")

    data_x = tbe.broadcast(x, shape_max)
    data_bias = tbe.broadcast(bias, shape_max)
    res = tbe.vadd(data_x, data_bias)

    return res


@register_operator("BiasAdd")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def bias_add(x, bias, y, data_format="NHWC", kernel_name="bias_add"):
    """
    algorithm: bias_and
    Reduce a tensor on a certain axis based on min

    Parameters
    ----------
    x : dict
              the shape and dtype of the tensor x
    bias : dict
              the shape and dtype of the tensor y
    y :  dict
              the shape and dtype of the tensor z
    data_format: A string.
                'N...C' and 'NC...' are supported.
    kernel_name : string
                  cce kernel name, default value is "bias_add"
    Returns
    -------
    None
    """
    shape_x = x.get("shape")
    shape_bias = bias.get("shape")
    range_bias = bias.get("range")
    range_x = x.get("range")

    dtype_x = x.get("dtype").lower()
    dtype_bias = bias.get("dtype").lower()
    dtype_y = y.get("dtype").lower()

    check_tuple = ("bfloat16", "float16", "float32", "int32", "int64")
    para_check.check_dtype(dtype_x, check_tuple, param_name="x")
    para_check.check_dtype(dtype_bias, check_tuple, param_name="bias")
    para_check.check_dtype(dtype_y, check_tuple, param_name="y")

    if dtype_x != dtype_bias:
        error_manager_vector.raise_err_inputs_dtype_not_equal("BiasAdd", "x", "bias",
                                                              str(dtype_x), str(dtype_bias))
    is_unknown_rank = is_unknown_rank_input([x, bias])
    if is_unknown_rank:
        x, bias = [x, x] if is_unknown_rank_input(x) else [bias, bias]
        shape_bias = [-2]
    else:
        data_format = data_format.upper()
        data_format_tuple = ("NCHW", "NHWC", "NDHWC", "NCDHW")
        para_check.check_format(data_format, data_format_tuple, param_name='input_format')
        if x.get("format") is not None and x.get("format").upper() == "NC1HWC0":
            ori_format_x = x.get("ori_format").upper()
            ori_shape_x = x.get("ori_shape")
            if len(shape_x) != 5:
                error_manager_vector.raise_err_specific_reson("BiasAdd",
                                                              "bias_add only support shape 5D, \
                                                              when input format is NC1HWC0")

            if ori_format_x != data_format:
                error_manager_vector.raise_err_two_input_format_invalid("BiasAdd", "ori_format", "data_format",
                                                                        "the input ori_format and data_format \
                                                                        must be the same")
            if bias.get("format") is not None and bias.get("format").upper() == "NC1HWC0":
                ori_shape_bias = bias.get("ori_shape")
                if ori_format_x == "NCHW" and not check_equal(ori_shape_x[1], ori_shape_bias[0]):
                    error_manager_vector.raise_err_specific_reson("BiasAdd", "data_format is NCHW, shape_bias must "
                                                                  "be equal to the second axis of shape_x")
                if ori_format_x == "NHWC" and not check_equal(ori_shape_x[-1], ori_shape_bias[0]):
                    error_manager_vector.raise_err_specific_reson("BiasAdd", "data_format is NHWC, shape_bias must \
                                                                  be equal to the last axis of shape_x")
            else:
                if ori_format_x == "NCHW" and not check_equal(ori_shape_x[1], shape_bias[0]):
                    error_manager_vector.raise_err_specific_reson("BiasAdd", "data_format is NCHW, shape_bias must "
                                                                  "be equal to the second axis of shape_x")
                if ori_format_x == "NHWC" and not check_equal(ori_shape_x[-1], shape_bias[0]):
                    error_manager_vector.raise_err_specific_reson("BiasAdd", "data_format is NHWC, shape_bias must \
                                                                  be equal to the last axis of shape_x")
            shape_bias = (1, shape_x[1], 1, 1, shape_x[4])
            range_bias = ((1, 1), range_x[1], (1, 1), (1, 1), range_x[4])

        elif x.get("format") is not None and x.get("format").upper() == "NDHWC":
            if len(shape_x) != 5:
                error_manager_vector.raise_err_specific_reson("BiasAdd", "bias_add only support shape 5D, \
                                                              when input format is NDHWC")

            if not check_equal(shape_x[4], shape_bias[0]):
                error_manager_vector.raise_err_specific_reson("BiasAdd", "data_format is NDHWC, shape_bias must \
                                                              be equal to the fifth axis of shape_x")
            shape_bias = (1, ) * (len(shape_x) - 1) + (shape_x[-1], )
            range_bias = ((1, 1), ) * (len(shape_x) - 1) + (range_x[-1], )

        elif x.get("format") is not None and x.get("format").upper() == "NCDHW":
            if len(shape_x) != 5:
                error_manager_vector.raise_err_specific_reson("BiasAdd", "bias_add only support shape 5D, \
                                                              when input format is NCDHW")
            if not check_equal(shape_x[1], shape_bias[0]):
                error_manager_vector.raise_err_specific_reson("BiasAdd", "data_format is NCDHW, shape_bias must \
                                                              be equal to the second axis of shape_x")
            shape_bias = (1, shape_x[1]) + (1, ) * (len(shape_x) - 2)
            range_bias = ((1, 1), range_x[1]) + ((1, 1), ) * (len(shape_x) - 2)

        elif x.get("format") is not None and x.get("format").upper() == "NDC1HWC0":
            if len(shape_x) != 6:
                error_manager_vector.raise_err_specific_reson("BiasAdd", "bias_add only support shape 6D \
                                                              when input format is NDC1HWC0")
            ori_shape_x = x.get("ori_shape")
            if x.get("ori_format").upper() == "NDHWC":
                if not check_equal(ori_shape_x[4], shape_bias[0]):
                    error_manager_vector.raise_err_specific_reson("BiasAdd", "data_format is NDHWC, shape_bias must "
                                                                  "be equal to the fifth axis of shape_x")
            elif x.get("ori_format").upper() == "NCDHW":
                if not check_equal(ori_shape_x[1], shape_bias[0]):
                    error_manager_vector.raise_err_specific_reson("BiasAdd", "data_format is NCDHW, shape_bias must \
                                                                  be equal to the second axis of shape_x")
            shape_bias = (1, 1, shape_x[2], 1, 1, shape_x[5])
            range_bias = ((1, 1), (1, 1), range_x[2], (1, 1), (1, 1), range_x[5])

        else:
            if data_format == "NCHW":
                if len(shape_x) < 2 or len(shape_x) > 4:
                    error_manager_vector.raise_err_specific_reson("BiasAdd", "bias_add only support shape \
                                                                  2D to 4D when input format is NCHW")
                if not check_equal(shape_x[1], shape_bias[0]):
                    error_manager_vector.raise_err_specific_reson("BiasAdd", "data_format is NCHW, shape_bias must"
                                                                  " be equal to the second axis of shape_x"
                                                                  ", but {} and {}".format(shape_bias[0], shape_x[1]))
                shape_bias = (1, shape_x[1],)
                range_bias = ((1, 1), range_x[1],)
                for i in range(2, len(shape_x)):
                    shape_bias = shape_bias + (1,)
                    range_bias = range_bias + ((1, 1),)
            else:
                if len(shape_x) < 2:
                    error_manager_vector.raise_err_specific_reson("BiasAdd", "only support shape larger than 1D")
                if not check_equal(shape_x[-1], shape_bias[0]):
                    error_manager_vector.raise_err_specific_reson("BiasAdd", "data_format is NHWC, shape_bias must be "
                                                                  "equal to the last axis of shape_x")
                shape_bias = ()
                range_bias = (())
                for i in range(0, len(shape_x)):
                    if i != len(shape_x) - 1:
                        shape_bias = shape_bias + (1,)
                        range_bias = range_bias + ((1, 1),)
                    else:
                        shape_bias = shape_bias + (shape_x[-1],)
                        range_bias = range_bias + (range_x[-1],)

        bias["shape"] = shape_bias
        bias["ori_shape"] = shape_bias
        bias["range"] = range_bias

        para_check.check_elewise_shape_range([x, bias], support_broadcast=True)

    ins = classify([x, bias], OpPatternMode.ELEWISE_WITH_BROADCAST)

    schedules, tensors = [], []
    for (_x, _bias) in ins:
        with tbe.compute():
            x_shape, bias_shape = shape_util.variable_shape([_x, _bias])
            tensor_x = tvm.placeholder(x_shape, dtype_x, "tensor_x")
            tensor_bias = tvm.placeholder(bias_shape, dtype_bias, "tensor_bias")

            res = bias_add_compute(tensor_x, tensor_bias, y, data_format, kernel_name)
            tensors.append((tensor_x, tensor_bias, res))
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe_context.get_context().add_compile_info("is_unknown_rank", is_unknown_rank)
    tbe_context.get_context().add_compile_info("boardcast_bias_shape", shape_bias)
    tbe.build(schedules, config)
