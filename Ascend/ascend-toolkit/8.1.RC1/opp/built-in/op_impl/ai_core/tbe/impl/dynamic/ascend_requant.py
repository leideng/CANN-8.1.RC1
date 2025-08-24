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
ascend_requant
"""
from functools import reduce as function_reduce
from collections import namedtuple
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl import ascend_quant_util as util


# 'pylint: disable=invalid-name,unused-argument,unnecessary-lambda,too-many-arguments,too-many-locals
# 'pylint: disable=too-many-statements,too-many-branches
@register_operator_compute("AscendRequant", op_mode="dynamic", support_fusion=True)
def ascend_requant_compute(x, req_scale, y, relu_flag=False, kernel_name="ascend_requant"):
    """
    int32 -> int8

    Parameters:
    ----------
    x: the placeholder of input
    req_scale: the placeholder of req_scale
    y: the dict of output
    relu_flag: the relu mode, default value is False
    kernel_name: cce kernel name, default value is "ascend_requant"

    Returns:
    -------
    res : the result of ascend_requant
    """
    x_shape = x.shape
    x_shape_list = shape_util.shape_to_list(x_shape)

    conv_flag = 0
    if len(x.op.input_tensors) > 0 and ('mad1' in x.op.input_tensors[0].name or \
            'convolution_c_col_bias' in x.op.input_tensors[0].name):
        conv_flag = 1

    if conv_flag:
        x_input_shape_list = shape_util.shape_to_list(x.op.input_tensors[0].shape)
        align_shape = [x_input_shape_list[1],
                       x.shape[1],
                       x_input_shape_list[3],
                       x_input_shape_list[4]]
        c1_index = 1

    else:
        align_shape = x_shape_list.copy()
        c1_index = 1

    # the tensor is a constant or vector based on the original shape
    ori_shape_req = req_scale.op.attrs["ori_shape"]
    ori_shape_req_list = shape_util.shape_to_list(ori_shape_req)
    req_dim = function_reduce(lambda x, y: x * y, ori_shape_req_list[:])
    tensor_flag = False
    if req_dim > 1:
        tensor_flag = True

    if util.is_nz_format(x):
        c1_index = len(x_shape) - 4

    if x.op.tag == "depthwise_conv2d":
        tensor_dict = {}
        tensor_dict["mad_ubuf"] = x.op.input_tensors[0]
        if x.op.attrs['bias_flag'].value == 1:
            tensor_dict["mad_after_bias"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
            tensor_dict["mad"] = tensor_dict["mad_after_bias"].op.input_tensors[1]
        else:
            tensor_dict["mad"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
        tensor_dict["im2col_fractal"] = tensor_dict["mad"].op.input_tensors[0]
        tensor_dict["im2col_row_major"] = tensor_dict["im2col_fractal"].op.input_tensors[0]

        tensor_dict["temp"] = tensor_dict["im2col_row_major"].op.input_tensors[0]
        if tensor_dict["temp"].op.input_tensors:
            tensor_dict["fmap"] = tensor_dict["temp"].op.input_tensors[0]
        else:
            tensor_dict["fmap"] = tensor_dict["im2col_row_major"].op.input_tensors[0]
        x_ori_shape = tensor_dict["fmap"].op.attrs["ori_shape"]
        x_ori_format = tensor_dict["fmap"].op.attrs["ori_format"]
        x_ori_shape_list = shape_util.shape_to_list(x_ori_shape)
        if x_ori_format == "NCHW":
            valid_c1 = (x_ori_shape_list[1] + 15) // 16
        else:
            valid_c1 = (x_ori_shape_list[3] + 15) // 16

        align_shape[4] = 16
        align_shape[3] = (x_shape_list[3] + 15) // 16 * 16
        align_shape[2] = 1
        if tensor_flag:
            align_shape[1] = (x_shape_list[1] * x_shape_list[2] * 16 + 31) // 32 * 32 // 16
        else:
            align_shape[1] = x_shape_list[1] * x_shape_list[2]
        align_shape[0] = x_shape_list[0]

        if tensor_flag:
            res_ub = tvm.compute(align_shape,
                                 lambda batch, c1, ho, wo, c0:
                                 tvm.select(c1 < valid_c1,
                                            tvm.vdeq_cast(x(batch, c1 // 2, c1 % 2, wo, c0), req_scale(0, c1, 0, 0, c0),
                                                          "int8", do_relu=relu_flag),
                                            tvm.const(0, dtype="int8")), name="s32_to_s8", tag="requant_vector")

        else:
            res_ub = tvm.compute(align_shape,
                                 lambda batch, c1, ho, wo, c0:
                                 tvm.select(c1 < valid_c1,
                                            tvm.deq_cast(x(batch, c1 // 2, c1 % 2, wo, c0), req_scale(0, 0, 0, 0, 0),
                                                         "int8"),
                                            tvm.const(0, dtype="int8")), name="s32_to_s8", tag="requant_scale")
    else:
        align_shape[c1_index] = (align_shape[c1_index] + 1) // 2 * 2
        if util.is_nz_format(x) and len(align_shape) == 4 and align_shape[1] == 1 and align_shape[2] == 1:
            pass
        else:
            align_shape[-2] = (align_shape[-2] + 15) // 16 * 16
        compute_input = namedtuple('NormalCompute', "x req_scale align_shape c1_index tensor_flag relu_flag")
        res_ub = _s32_to_s8_normal_compute(
            conv_flag, compute_input(x, req_scale, align_shape, c1_index, tensor_flag, relu_flag)
            )

    if util.is_nz_format(x):
        res = _format_transfer_nz(align_shape, res_ub, c1_index)
        return res

    res_ub_reform = _format_transfer(align_shape, res_ub, c1_index)
    res_shape = shape_util.shape_to_list(res_ub_reform.shape)

    res_shape[-2] = x.shape[-2]
    if conv_flag:
        if x.op.attrs["remove_padded_column_in_next_op"].value == 1:
            res_shape[-2] = res_shape[-2]//2
            res_ub_reform = tvm.compute(res_shape,
                                        lambda batch, cout1, howo, cout0: res_ub_reform(batch, cout1, howo * 2, cout0),
                                        name='requant_remove_padded_column',
                                        tag='requant_remove_padded_column')
        res = tvm.compute(res_shape,
                          lambda batch, cout1, howo, cout0: res_ub_reform(batch, cout1, howo, cout0),
                          name="requant_remove_pad", tag="requant_remove_pad")
    else:
        res = tvm.compute(res_shape, lambda *indice: res_ub_reform(*indice),
                          name="requant_remove_pad", tag="requant_remove_pad")

    return res


def _s32_to_s8_normal_compute(conv_flag, compute_input):
    """
    generate s32_to_s8 compute
    """
    if compute_input.tensor_flag:
        res_ub = tvm.compute(compute_input.align_shape,
                             _deq_cast_compute(conv_flag, compute_input),
                             name="s32_to_s8", tag="requant_vector")
    else:
        res_ub = tvm.compute(compute_input.align_shape,
                             _deq_cast_compute(conv_flag, compute_input),
                             name="s32_to_s8", tag="requant_scale")
    return res_ub


def _deq_cast_compute(conv_flag, compute_input):
    """
    generate lambda func
    """
    if conv_flag:
        return _conv_deq_cast_compute(compute_input)
    return _normal_deq_cast_compute(compute_input)


def _conv_deq_cast_compute(compute_input):
    x, req_scale, align_shape, c1_index, tensor_flag, relu_flag = compute_input
    group = x.op.input_tensors[0].shape[0].value
    x_shape_list = shape_util.shape_to_list(x.op.input_tensors[0].shape)
    cout1_opt = x_shape_list[2]
    # shape of req_scale is [1, true_cout1, 1, 1, c0], req_compute shouldn't exceed true_cout1
    true_cout1 = x.op.attrs["conv_shape"][1] # output shape of conv, conv_shape = [batch, true_cout1, ho*wo, c0]

    def lambda_func(batch, cout1, howo, cout0):
        new_indice = [0] * 5
        if tensor_flag:
            new_indice[4] = cout0
            new_indice[1] = cout1

        if tensor_flag:
            res = tvm.select(
                    cout1 < true_cout1,
                    tvm.vdeq_cast(
                        x.op.input_tensors[0](0 if group == 1 else cout1 // cout1_opt,
                                              batch,
                                              cout1 if group == 1 else cout1 % cout1_opt, howo, cout0),
                        req_scale(*new_indice), "int8", do_relu=relu_flag),
                    tvm.const(0, dtype="int8"))

        else:
            res = tvm.select(cout1 < true_cout1,
                              tvm.deq_cast(
                                  x.op.input_tensors[0](0 if group == 1 else cout1 // cout1_opt,
                                                        batch,
                                                        cout1 if group == 1 else cout1 % cout1_opt, howo, cout0),
                                  req_scale(*new_indice), "int8"),
                              tvm.const(0, dtype="int8"))
        return res

    return lambda_func


def _normal_deq_cast_compute(compute_input):
    x, req_scale, align_shape, c1_index, tensor_flag, relu_flag = compute_input
    n_dim = len(align_shape)
    c0_index = n_dim - 1
    x_shape_list = shape_util.shape_to_list(x.shape)

    def lambda_func(*indice):
        new_indice = [0] * 5
        if tensor_flag:
            new_indice[4] = indice[c0_index]
            new_indice[1] = indice[c1_index]

        if tensor_flag:
            res = tvm.select(indice[c1_index] < x_shape_list[c1_index],
                             tvm.vdeq_cast(x(*indice), req_scale(*new_indice), "int8", do_relu=relu_flag),
                             tvm.const(0, dtype="int8"))
        else:
            res = tvm.select(indice[c1_index] < x_shape_list[c1_index],
                             tvm.deq_cast(x(*indice), req_scale(*new_indice), "int8"),
                             tvm.const(0, dtype="int8"))
        return res

    return lambda_func


def _format_compute(tensor, trans_shape, c1_index):
    """
    generate lambda func
    """
    n_dim = len(trans_shape)
    c0_index = n_dim - 1

    def lambda_func(*indice):
        new_indice = [0] * n_dim
        for i in range(n_dim):
            if i == c0_index:
                new_indice[i] = (indice[c1_index] * 32 + indice[c0_index]) % 16
            elif i == c1_index:
                new_indice[i] = (indice[c1_index] * 32 + indice[c0_index]) // 16
            else:
                new_indice[i] = indice[i]
        return tensor(*new_indice)

    return lambda_func


def _format_transfer_nz(shape, x, c1_index):
    """
    C0 from 16 to 32 for FRACTAL_NZ
    """
    trans_shape = shape[:]
    trans_shape[c1_index] = trans_shape[c1_index] // 2
    trans_shape[-1] = trans_shape[-1] * 2
    res = tvm.compute(trans_shape,
                      _format_compute(x, trans_shape, c1_index),
                      name="data_transfer", tag="requant_data_transfer")
    res = tvm.compute(trans_shape, lambda *i: res[i], name="res", tag="requant_NZ")
    return res


def _format_transfer(shape, x, c1_index):
    """
    C0 from 16 to 32 for NC1HWC0
    """
    trans_shape = shape[:]
    trans_shape[c1_index] = trans_shape[c1_index] // 2
    trans_shape[-1] = trans_shape[-1] * 2
    res = tvm.compute(trans_shape,
                      _format_compute(x, trans_shape, c1_index),
                      name="data_transfer", tag="data_transfer")
    return res


def _check_params(x, req_scale, y, relu_flag, kernel_name):
    """
    check the parameters including dtype, kernel_name, attr
    """
    format_x = x.get("format")

    format_req = req_scale.get("format")

    dtype_x = x.get("dtype")
    dtype_req = req_scale.get("dtype")

    check_list = [("int32",), ("uint64",)]
    format_list = ["NC1HWC0", "FRACTAL_NZ"]
    para_check.check_dtype(dtype_x, check_list[0], param_name="x")
    para_check.check_dtype(dtype_req, check_list[1], param_name="req_scale")
    para_check.check_format(format_x, format_list, param_name="x")

    para_check.check_format(format_req, ("NC1HWC0",), param_name="req_scale")


@register_operator("AscendRequant", pattern="requant")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def ascend_requant(x, req_scale, y, relu_flag=False, kernel_name="ascend_requant"):
    """
    int32 -> int8

    Parameters:
    ----------
    x: the placeholder of input
    req_scale: the placeholder of req_scale
    y: the dict of output
    relu_flag: the relu mode, default value is False
    kernel_name: cce kernel name, default value is "ascend_requant"

    Returns:
    -------
    None
    """

    _check_params(x, req_scale, y, relu_flag, kernel_name)

    dtype_x = x.get("dtype")
    shape_req = req_scale.get("shape")
    dtype_req = req_scale.get("dtype")
    ori_shape_req = req_scale.get("ori_shape")
    attr = {"ori_shape": ori_shape_req}

    schedules, tensors = [], []
    with tbe.compute():
        x_n = operation.var("x_n")
        x_c1 = operation.var("x_c1")
        x_hw = operation.var("x_hw")

        input_x_shape = []
        input_x_shape.append(x_n)
        input_x_shape.append(x_c1)
        input_x_shape.append(x_hw)
        input_x_shape.append(16)

        input_x = tvm.placeholder(input_x_shape, dtype_x, "x")
        input_req = tvm.placeholder(shape_req, name="req_scale", dtype=dtype_req, attrs=attr)
        res = ascend_requant_compute(input_x, input_req, y, relu_flag, kernel_name)

        tensors.append([input_x, input_req, res])

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
