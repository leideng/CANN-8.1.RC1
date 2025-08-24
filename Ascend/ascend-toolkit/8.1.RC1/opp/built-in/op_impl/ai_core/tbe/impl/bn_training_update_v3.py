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
bn_training_update_v3
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from tbe import tvm
from te.utils import para_check
from te.utils import shape_util
from te.utils.error_manager import error_manager_vector
from impl.util import util_select_op_base


# 'pylint: disable=locally-disabled,too-many-locals,unused-argument,invalid-name
# 'pylint: disable=locally-disabled,too-many-arguments,redefined-builtin
def _check_shape(shape_x, shape_sum, shape_square_sum,
                 shape_scale, shape_offset, format):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_x: list or tuple
        x's data shape
    shape_sum: list or tuple
        sum's data shape
    shape_square_sum: list or tuple
        square_sum's data shape
    shape_scale: list or tuple
        scale's data shape
    shape_offset: list or tuple
        offset's data shape

    Returns
    -------
    None
    """
    para_check.check_shape_rule(shape_x)
    para_check.check_tensor_shape_size(shape_x)

    para_check.check_shape_rule(shape_sum)
    para_check.check_tensor_shape_size(shape_sum)

    para_check.check_shape_rule(shape_square_sum)
    para_check.check_tensor_shape_size(shape_square_sum)

    para_check.check_shape_rule(shape_scale)
    para_check.check_tensor_shape_size(shape_scale)

    para_check.check_shape_rule(shape_offset)
    para_check.check_tensor_shape_size(shape_offset)

    if len(shape_x) not in (5, 6) or len(shape_sum) not in (5, 6) \
            or len(shape_square_sum) not in (5, 6) or len(shape_scale) not in (5, 6) \
            or len(shape_offset) not in (5, 6):
        error_reson = "The data format is 5HD, but some input's shape length is not 5 or 6"
        error_manager_vector.raise_err_specific_reson("bn_training_update_v3", error_reson)
    dim_c1 = 0
    dim_c0 = 0
    c1 = 0
    c0 = 0
    if format == "NC1HWC0":
        dim_c1 = shape_x[1]
        dim_c0 = shape_x[4]
        c1 = 1
        c0 = 4
    else:
        dim_c1 = shape_x[2]
        dim_c0 = shape_x[5]
        c1 = 2
        c0 = 5

    if shape_sum[c1] != dim_c1 or shape_sum[c0] != dim_c0:
        error_manager_vector.raise_err_specific_reson("bn_training_update_v3",
                                                      "Dimension C of x and sum must be equal")
    if shape_square_sum[c1] != dim_c1 or shape_square_sum[c0] != dim_c0:
        error_manager_vector.raise_err_specific_reson("bn_training_update_v3",
                                                      "Dimension C of x and square_sum must be equal")
    if shape_scale[c1] != dim_c1 or shape_scale[c0] != dim_c0:
        error_manager_vector.raise_err_specific_reson("bn_training_update_v3",
                                                      "Dimension C of x and scale must be equal")
    if shape_offset[c1] != dim_c1 or shape_offset[c0] != dim_c0:
        error_manager_vector.raise_err_specific_reson("bn_training_update_v3",
                                                      "Dimension C of x and offset must be equal")


# 'pylint: disable=unused-argument,invalid-name
# 'pylint: disable=too-many-arguments,too-many-locals,redefined-builtin
def check_supported(x,
                    sum,
                    square_sum,
                    scale,
                    offset,
                    y,
                    batch_mean,
                    batch_variance,
                    reserve_1,
                    reserve_2,
                    epsilon,
                    before_split_ori_shape=None,
                    before_split_ori_format=None,
                    kernel_name="bn_training_update_v3"):
    return True, ""


# 'pylint: disable=unused-argument,invalid-name
# 'pylint: disable=too-many-arguments,too-many-locals,redefined-builtin
def op_select_format(x,
                     sum,
                     square_sum,
                     scale,
                     offset,
                     y,
                     batch_mean,
                     batch_variance,
                     reserve_1,
                     reserve_2,
                     epsilon,
                     before_split_ori_shape=None,
                     before_split_ori_format=None,
                     kernel_name="bn_training_update_v3"):
    input0 = util_select_op_base.gen_param(classify="input0",
                                           name="x",
                                           datatype="float16,float,float16,float",
                                           format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
    input1 = util_select_op_base.gen_param(classify="input1",
                                           name="sum",
                                           datatype="float,float,float,float",
                                           format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
    input2 = util_select_op_base.gen_param(classify="input2",
                                           name="square_sum",
                                           datatype="float,float,float,float",
                                           format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
    input3 = util_select_op_base.gen_param(classify="input3",
                                           name="scale",
                                           datatype="float,float,float,float",
                                           format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
    input4 = util_select_op_base.gen_param(classify="input4",
                                           name="offset",
                                           datatype="float,float,float,float",
                                           format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
    output0 = util_select_op_base.gen_param(classify="output0",
                                            name="y",
                                            datatype="float16,float,float16,float",
                                            format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
    output1 = util_select_op_base.gen_param(classify="output1",
                                            name="batch_mean",
                                            datatype="float,float,float,float",
                                            format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
    output2 = util_select_op_base.gen_param(classify="output2",
                                            name="batch_variance",
                                            datatype="float,float,float,float",
                                            format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
    output3 = util_select_op_base.gen_param(classify="output3",
                                            name="reserve_1",
                                            datatype="float,float,float,float",
                                            format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
    output4 = util_select_op_base.gen_param(classify="output4",
                                            name="reserve_2",
                                            datatype="float,float,float,float",
                                            format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
    param_list = [input0, input1, input2, input3, input4,
                  output0, output1, output2, output3, output4]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


# 'pylint: disable=locally-disabled,too-many-arguments
def _check_dtype(dtype_x, dtype_sum, dtype_square_sum,
                 dtype_scale, dtype_offset):
    """
    Function to check if the dtype is in line with norms.

    Parameters
    ----------
    dtype_x: str
        x's data type
    dtype_sum: str
        sum's data type
    dtype_square_sum: str
        square_sum's data type
    dtype_scale: str
        scale's data type
    dtype_offset: str
        offset's data type

    Returns
    -------
    None
    """
    para_check.check_dtype_rule(dtype_x.lower(), ("float16", "float32"))
    para_check.check_dtype_rule(dtype_sum.lower(), ("float32",))
    para_check.check_dtype_rule(dtype_square_sum.lower(), ("float32",))
    para_check.check_dtype_rule(dtype_scale.lower(), ("float32",))
    para_check.check_dtype_rule(dtype_offset.lower(), ("float32",))


@tbe_platform.fusion_manager.fusion_manager.register("bn_training_update_v3")
def bn_training_update_v3_compute(x, sum, square_sum, scale, offset,
                                  y, batch_mean, batch_variance,
                                  reserve_1, reserve_2, epsilon,
                                  before_split_ori_shape=None,
                                  before_split_ori_format=None,
                                  kernel_name="bn_training_update_v3"):
    """
    algorithm: fused_batch_norm_v2
    Batch normalization.

    Parameters
    ----------
    x: TVM tensor
        contains x data
    sum: TVM tensor
        contains sum data
    square_sum: TVM tensor
        contains square_sum data
    scale: TVM tensor
        contains scale data
    offset: TVM tensor
        contains offset data
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    batch_mean: dict
        dict of batch_mean, A `Tensor`.
        One of the result which is called save mean.
    batch_variance: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_mean`.
    reserve_1: dict
        dict of batch_mean, A `Tensor`.
        Has the same type as `batch_mean`.
    reserve_2: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_variance`.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_training_update_v3"

    Returns
    -------
    res: TVM tensor list
        the result of bn_training_update_v3 compute
    """
    shape_x = shape_util.shape_to_list(x.shape)

    num = shape_x[0] * shape_x[2] * shape_x[3]
    num_rec = 1.0/num

    # compute the saved mean of x
    save_mean_reduce = tbe.vmuls(sum, num_rec)

    # compute the saved variance of x
    variance_div = tbe.vmuls(square_sum, num_rec)
    variance_square = tbe.vmul(save_mean_reduce, save_mean_reduce)
    save_variance_reduce = tbe.vsub(variance_div, variance_square)

    # compute the oefficient of y
    multiplier_add = tbe.vadds(save_variance_reduce, epsilon)
    multiplier_sqrt = tbe.vsqrt(multiplier_add)
    multiplier_div = tbe.vdiv(scale, multiplier_sqrt)
    multiplier = tbe.broadcast(multiplier_div, shape_x)

    addend_mul = tbe.vmul(multiplier_div, save_mean_reduce)
    addend_sub = tbe.vsub(offset, addend_mul)
    addend = tbe.broadcast(addend_sub, shape_x)

    # compute the batch normalization of x
    is_cast = False
    if x.dtype == "float16":
        is_cast = True
        x = tbe.cast_to(x, "float32")

    res_y = tbe.vadd(tbe.vmul(multiplier, x), addend)
    if is_cast:
        res_y = tbe.cast_to(res_y, "float16")

    # compute batch_mean and batch_var
    res_batch_mean = tbe.vmuls(sum, num_rec)
    if num == 1:
        batch_var_scaler = 0.0
    else:
        batch_var_scaler = float(num) / (num - 1)
    res_batch_var = tbe.vmuls(save_variance_reduce, batch_var_scaler)

    res = [res_y, res_batch_mean, res_batch_var,
           save_mean_reduce, save_variance_reduce]

    return res


# 'pylint: disable=locally-disabled,too-many-arguments,too-many-locals
@para_check.check_input_type(dict, dict, dict, dict, dict, dict,
                             dict, dict, dict, dict, float, tuple, tuple, str)
def bn_training_update_v3(x, sum, square_sum, scale, offset,
                          y, batch_mean, batch_variance,
                          reserve_1, reserve_2, epsilon,
                          before_split_ori_shape=None,
                          before_split_ori_format=None,
                          kernel_name="bn_training_update_v3"):
    """
    algorithm: fused_batch_norm_v2
    Batch normalization.

    Parameters
    ----------
    x: dict
        dict of input, A 5HD Tensor for input data.
    sum: dict
        dict of sum, A 5HD Tensor for sum.
        The output of batch_normalization_forward_training_reduce.
    square_sum: dict
        dict of square_sum, A 5HD Tensor for square_sum.
        The output of batch_normalization_forward_training_reduce.
    scale: dict
        dict of scale, A 5HD Tensor for mean.
    offset: dict
        dict of offset, A 5HD Tensor for variance.
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    batch_mean: dict
        dict of batch_mean, A `Tensor`.
        One of the result which is called save mean.
    batch_variance: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_mean`.
    reserve_1: dict
        dict of batch_mean, A `Tensor`.
        Has the same type as `batch_mean`.
    reserve_2: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_variance`.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_training_update_v3"

    Returns
    -------
    None
    """

    shape_x = x.get("shape")
    shape_sum = sum.get("shape")
    shape_square_sum = square_sum.get("shape")
    shape_scale = scale.get("shape")
    shape_offset = offset.get("shape")

    dtype_x = x.get("dtype")
    dtype_sum = sum.get("dtype")
    dtype_square_sum = square_sum.get("dtype")
    dtype_scale = scale.get("dtype")
    dtype_offset = offset.get("dtype")
    format = x.get("format")

    _check_shape(shape_x, shape_sum, shape_square_sum,
                 shape_scale, shape_offset, format)

    _check_dtype(dtype_x, dtype_sum, dtype_square_sum,
                 dtype_scale, dtype_offset)
    if format == "NDC1HWC0":
        shape_x = [shape_x[0] * shape_x[1], shape_x[2], shape_x[3], shape_x[4], shape_x[5]]
        shape_square_sum = [shape_square_sum[0] * shape_square_sum[1], shape_square_sum[2],
                            shape_square_sum[3], shape_square_sum[4], shape_square_sum[5]]
        shape_sum = [shape_sum[0] * shape_sum[1], shape_sum[2],
                            shape_sum[3], shape_sum[4], shape_sum[5]]

    x_input = tvm.placeholder(shape_x, name="x_input", dtype=dtype_x.lower())
    sum_input = tvm.placeholder(shape_sum, name="sum_input",
                                dtype=dtype_sum.lower())
    square_sum_input = tvm.placeholder(shape_square_sum,
                                       name="square_sum_input",
                                       dtype=dtype_square_sum.lower())
    scale_input = tvm.placeholder(shape_sum, name="scale_input",
                                  dtype=dtype_scale.lower())
    offset_input = tvm.placeholder(shape_sum, name="offset_input",
                                   dtype=dtype_offset.lower())

    res = bn_training_update_v3_compute(x_input, sum_input, square_sum_input,
                                        scale_input, offset_input, y,
                                        batch_mean, batch_variance,
                                        reserve_1, reserve_2, epsilon,
                                        before_split_ori_shape,
                                        before_split_ori_format,
                                        kernel_name=kernel_name)
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    tensor_list = [x_input, sum_input, square_sum_input,
                   scale_input, offset_input, ] + list(res)

    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    tbe.cce_build_code(sch, config)
