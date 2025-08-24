# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
dynamic bn_training_reduce
"""
from impl.util import util_common
from impl.util import util_select_op_base
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tuple_sum
from impl.util.util_common import is_unknown_rank_input
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import ReduceInput
from impl.util.util_select_op_base import ReduceOutput
from impl.util.util_select_op_base import get_op_cal_info
from impl.dynamic.bn_training_update import check_special_soc


# 'pylint: disable=unused-argument,invalid-name
# 'pylint: disable=too-many-locals, too-many-statements,redefined-builtin
def op_select_format(x, sum, square_sum, kernel_name="bn_training_reduce"):
    """
    1. when input(x)'s ori_shape is [1, ? ,1, ?] and the format is NCHW,
    the Op BNTrainingReduce can support NCHW.
    > for example:
    > x : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > the Op BNTrainingReduce can process with NC1HWC0:
    > x : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    2. In other scenes, the Op BNTrainingReduce can support NC1HWC0 and NDC1HWC0
    > for example:
    > x : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    """
    origin_format = x.get("ori_format").upper()
    origin_shape = x.get("ori_shape")
    if origin_format == "NCHW" and len(origin_shape) == 4 and origin_shape[0] == 1 and origin_shape[2] == 1:
        format_unknow = "NCHW,NCHW,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NCHW,NC1HWC0,NC1HWC0"
        input0 = util_select_op_base.gen_param(classify="input0",
                                               name="x",
                                               datatype="float16,float,float16,float,float16,float,\
                                                         bfloat16,bfloat16,bfloat16",
                                               format="NCHW,NCHW,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NCHW,NC1HWC0,NC1HWC0",
                                               unknownshape_format=format_unknow)
        output0 = util_select_op_base.gen_param(classify="output0",
                                                name="sum",
                                                datatype="float,float,float,float,float,float,float,float,float",
                                                format="NCHW,NCHW,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NCHW,NC1HWC0,NC1HWC0",
                                                unknownshape_format=format_unknow)
        output1 = util_select_op_base.gen_param(classify="output1",
                                                name="square_sum",
                                                datatype="float,float,float,float,float,float,float,float,float",
                                                format="NCHW,NCHW,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NCHW,NC1HWC0,NC1HWC0",
                                                unknownshape_format=format_unknow)
    else:
        format_unknow = "NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,NHWC,NHWC,NCHW,NCHW,NCDHW,NCDHW,\
                         NC1HWC0,NDC1HWC0,NHWC,NCHW,NCDHW"
        input0 = util_select_op_base.gen_param(classify="input0",
                                               name="x",
                                               datatype="float16,float,float16,float,float16,float,float16,float,\
                                                         float16,float,bfloat16,bfloat16,bfloat16,bfloat16,bfloat16",
                                               format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,\
                                                       NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NDC1HWC0,NC1HWC0,NC1HWC0,\
                                                       NC1HWC0",
                                               unknownshape_format=format_unknow)
        output0 = util_select_op_base.gen_param(classify="output0",
                                                name="sum",
                                                datatype="float,float,float,float,float,float,float,float,float,float,\
                                                          float,float,float,float,float",
                                               format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,\
                                                       NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NDC1HWC0,NC1HWC0,NC1HWC0,\
                                                       NC1HWC0",
                                                unknownshape_format=format_unknow)
        output1 = util_select_op_base.gen_param(classify="output1",
                                                name="square_sum",
                                                datatype="float,float,float,float,float,float,float,float,float,float,\
                                                          float,float,float,float,float",
                                               format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,\
                                                       NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NDC1HWC0,NC1HWC0,NC1HWC0,\
                                                       NC1HWC0",
                                                unknownshape_format=format_unknow)

    param_list = [input0, output0, output1]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def check_supported(x, sum, square_sum, kernel_name="bn_training_reduce"):
    """
    check supported
    """
    if util_common.is_unknown([x]):
        return True, ""

    # static shape only support special soc
    if check_special_soc():
        return True, ""

    return False, ""


def get_op_support_info(x, sum, square_sum, kernel_name="bn_training_reduce"):
    """
    get_op_support_info
    """
    format_x = x.get("format").upper()
    if format_x == "NC1HWC0":
        axis_split_matrix = [[SplitInput([0, [1], [-1], [-1]]), SplitOutput([0, [1]], [1, [1]])]]
        axis_reduce_list = [[ReduceInput([0, [0]]), ReduceOutput([0, 0, True], [1, 0, True])]]
    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


def _check_format(data_format, origin_foramt):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    data_format: str
        data format of data
    origin_foramt: str
        origin format of data

    Returns
    -------
    None
    """
    if data_format.upper() not in ("NC1HWC0", "NCHW", "NDC1HWC0", "NHWC", "NCDHW"):
        error_manager_vector.raise_err_specific_reson("bn_training_reduce",
                                                      "The data format only supports NC1HWC0,"
                                                      "NDC1HWC0, NCHW and NCDHW.")
    if data_format.upper() == "NCHW":
        if origin_foramt not in ("NCHW",):
            error_manager_vector.raise_err_specific_reson("bn_training_reduce",
                                                          "The origin format only supports NCHW when format is NCHW")


@register_operator_compute("BNTrainingReduce", op_mode="dynamic", support_fusion=True)
def bn_training_reduce_compute(x, sum, square_sum, kernel_name="bn_training_reduce", reduce_axis=None):
    """
    algorithm: part of batch_norm
    The first step of batch_norm
    which to calculate the sum and square sum of x.
    The major component of this operator is reduce operation.

    Parameters
    ----------
    x: TVM tensor
        contains x data
    sum: dict
        dict of sum, A `Tensor`. Sum of x.
    square_sum: dict
        dict of square_sum, A `Tensor`. Square sum of x.
    kernel_name: str
        kernel name, default value is "bn_training_reduce"
    reduce_axis: list
        reduce axis of input shape

    Returns
    -------
    res: TVM tensor list
        the result of bn_training_reduce compute
    """
    dtype = x.dtype.lower()
    if dtype in ("float16", "bfloat16"):
        x = tbe.cast_to(x, "float32")

    data_format = sum.get("format").upper()
    if not reduce_axis and data_format in ("NC1HWC0", "NCHW"):
        axis = [0, 2, 3]
    elif not reduce_axis and data_format in ("NDC1HWC0",):
        axis = [0, 1, 3, 4]
    elif not reduce_axis and data_format in ("NHWC",):
        axis = [0, 1, 2]
    elif not reduce_axis and data_format in ("NCDHW",):
        axis = [0, 2, 3, 4]
    else:
        axis = reduce_axis
    square_x = tbe.vmul(x, x)
    sum_x, square_sum_x = tuple_sum([x, square_x], axis, True)
    res = [sum_x, square_sum_x]

    return res


@register_operator("BNTrainingReduce")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def bn_training_reduce(x, sum, square_sum, kernel_name="bn_training_reduce"):
    """
    algorithm: part of batch_norm
    The first step of batch_norm
    which to calculate the sum and square sum of x.
    The major component of this operator is reduce operation.

    Parameters
    ----------
    x: dict
        dict of input, A 5HD Tensor for input data.
    sum: dict
        dict of sum, A `Tensor`. Sum of x.
    square_sum: dict
        dict of square_sum, A `Tensor`. Square sum of x.
    kernel_name: str
        kernel name, default value is "bn_training_reduce"

    Returns
    -------
    None
    """
    dtype_x = x.get("dtype")
    para_check.check_dtype(dtype_x.lower(), ("float16", "float32", "bfloat16"), param_name="x")

    data_format = x.get("format")
    origin_format = x.get("ori_format")
    _check_format(data_format, origin_format)

    if data_format in ("NC1HWC0", "NCHW"):
        list_axis = [0, 2, 3]
    elif data_format == "NHWC":
        list_axis = [0, 1, 2]
    elif data_format == "NCDHW":
        list_axis = [0, 2, 3, 4]
    else:
        list_axis = [0, 1, 3, 4]

    if is_unknown_rank_input(x):
        if data_format == "NC1HWC0":
            x["shape"] = [-1, -1, -1, -1, 16]
            x["range"] = [(1, None), (1, None), (1, None), (1, None), (16, 16)]
        elif data_format in ["NCHW", "NHWC"]:
            x["shape"] = [-1, -1, -1, -1]
            x["range"] = [(1, None), (1, None), (1, None), (1, None)]
        elif data_format in ["NCDHW", ]:
            x["shape"] = [-1, -1, -1, -1, -1]
            x["range"] = [(1, None), (1, None), (1, None), (1, None), (1, None)]
        else:
            x["shape"] = [-1, -1, -1, -1, -1, -1]
            x["range"] = [(1, None), (1, None), (1, None), (1, None), (1, None), (1, None)]

    ins = classify([x, list_axis], OpPatternMode.TUPLE_REDUCE)
    schedules, tensors = [], []
    for (_x, _reduce_axis) in ins:
        with tbe.compute():
            shape_x = shape_util.variable_shape([_x], op_mode=OpPatternMode.TUPLE_REDUCE)
            input_x = tvm.placeholder(shape_x[0], name="input_x", dtype=dtype_x)
            res = bn_training_reduce_compute(input_x, sum, square_sum, kernel_name, _reduce_axis)

            tensor_list = [input_x] + list(res)
            tensors.append(tensor_list)

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
            schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
