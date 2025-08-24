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
ascend_dequant
"""
import functools
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import error_manager_vector
from impl import ascend_quant_util as util


# 'pylint: disable=locally-disabled,too-many-arguments, unused-argument
# 'pylint: disable=invalid-name,too-many-locals,unnecessary-lambda
# 'pylint: disable=len-as-condition,too-many-branches,too-many-statements
def _check_params(x, deq_scale, sqrt_mode, kernel_name):
    """
    check the parameters including dtype, kernel_name, attr
    """

    x_format = x.get("format")
    deq_format = deq_scale.get("format")

    x_dtype = x.get("dtype").lower()
    deq_dtype = deq_scale.get("dtype").lower()
    x_format_list = ["NC1HWC0", "FRACTAL_NZ"]
    para_check.check_format(x_format, x_format_list, param_name="x")
    para_check.check_format(deq_format, ("NC1HWC0",), param_name="deq_scale")

    para_check.check_dtype(x_dtype, ("int32",), param_name="x")

    deq_dtype_check = "float16"
    if _is_support_v200_instruction():
        deq_dtype_check = "uint64"

    para_check.check_dtype(deq_dtype, (deq_dtype_check,), param_name="deq_scale")

    if deq_dtype == "uint64" and sqrt_mode:
        rule = "when deq_scale dtype is uint64, sqrt_mode only support False"
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule, "sqrt_mode", sqrt_mode)


def _is_support_v200_instruction():
    if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ("Ascend310P", "Ascend610", "BS9SX1A", "Hi3796CV300CS",
                                                          "SD3403"):
        return True
    return False


def _matmul_vdeq_cast_compute(x, deq_scale, c1_index, flags):
    """
    generate lambda func
    """
    n_dim = len(x.shape)
    c0_index = n_dim - 1
    tensor_flag, relu_flag, _, _ = flags
    def lambda_func(*indice):
        new_indice = [0] * 5
        if tensor_flag:
            new_indice[4] = indice[c0_index]
            new_indice[1] = indice[c1_index]
        if _is_support_v200_instruction():
            if tensor_flag:
                func = tvm.vdeq_cast(x(*indice), deq_scale(*new_indice), dtype="float16", do_relu=relu_flag)
            else:
                func = tvm.deq_cast(x(*indice), deq_scale(*new_indice), dtype="float16")
        else:
            func = x(*indice).astype("float16") * deq_scale(*new_indice)
        return func

    return lambda_func


def _matmul_compute(x, deq_scale, shape_matmul_origin, c1_index, flags):
    """
    dequant for matmul
    """
    x_shape = x.shape
    tensor_flag, relu_flag, sqrt_mode, _ = flags
    if _is_support_v200_instruction():
        if tensor_flag:
            res_f16 = tvm.compute(x_shape,
                                  _matmul_vdeq_cast_compute(x, deq_scale, c1_index, flags),
                                  name="dequant", tag="dequant_vector")
        else:
            res_f16 = tvm.compute(x_shape,
                                  _matmul_vdeq_cast_compute(x, deq_scale, c1_index, flags),
                                  name="dequant", tag="dequant_scale")
    else:
        if tensor_flag:
            res_f16 = tvm.compute(x_shape,
                                  _matmul_vdeq_cast_compute(x, deq_scale, c1_index, flags),
                                  name="dequant", tag="dequant_vector")
        else:
            res_f16 = tvm.compute(x_shape,
                                  _matmul_vdeq_cast_compute(x, deq_scale, c1_index, flags),
                                  name="dequant", tag="dequant")
        if sqrt_mode:
            if tensor_flag:
                res_f16 = tvm.compute(x_shape,
                                      _matmul_vdeq_cast_compute(res_f16, deq_scale, c1_index, flags),
                                      name="dequant_sqrt", tag="dequant_vector_sqrt")
            else:
                res_f16 = tvm.compute(x_shape,
                                      _matmul_vdeq_cast_compute(res_f16, deq_scale, c1_index, flags),
                                      name="dequant_sqrt", tag="dequant_sqrt")

        if relu_flag:
            res_f16 = tvm.compute(x_shape, lambda *indices: tvm.relu(res_f16[indices]),
                                  name="dequant_relu", tag="dequant_relu")
    if not util.is_nz_format(x):
        # convert fractal_z to ND
        res_out = tvm.compute(shape_matmul_origin,
                              lambda batch, cout: res_f16[cout // 16, batch // 16, batch % 16, cout % 16],
                              name="dequant_ND", tag="dequant_ND", attrs={"format": "NC1HWC0"})
    else:
        # nz format
        res_out = tvm.compute(x_shape, lambda *indice: res_f16[indice], name="dequant_NZ", tag="dequant_NZ",
                              attrs={"format": "FRACTAL_NZ"})
    return res_out


def _vector_dequant_v100(x, x_shape, align_shape, deq_scale, flags):
    """
    dequant for vector in v100
    """
    _, relu_flag, sqrt_mode, conv_flag = flags
    if conv_flag:
        invalid_data_rm_flag = int(x.op.attrs["invalid_data_rm_flag"])
        group = x.op.input_tensors[0].shape[0].value
        cout1_opt = x.op.input_tensors[0].shape[2].value
        res_shape_nchw_after_removepad = x.op.attrs["conv_shape"]

        if relu_flag:
            res_f16 = tvm.compute(
                align_shape,
                lambda batch, cout1, howo, cout0:
                tvm.relu(x.op.input_tensors[0](
                    0 if group == 1 else cout1 // cout1_opt, batch,
                    cout1 if group == 1 else cout1 % cout1_opt, howo, cout0).astype("float16") *
                         deq_scale(0, cout1, 0, 0, cout0)),
                name="dequant1", tag="dequant1_vector", attrs={"relu_flag": 1})
        else:
            res_f16 = tvm.compute(
                align_shape,
                lambda batch, cout1, howo, cout0:
                x.op.input_tensors[0](
                    0 if group == 1 else cout1 // cout1_opt, batch,
                    cout1 if group == 1 else cout1 % cout1_opt, howo, cout0).astype("float16") *
                deq_scale(0, cout1, 0, 0, cout0),
                name="dequant1", tag="dequant1_vector", attrs={"relu_flag": 0})

        if x.op.attrs["remove_padded_column_in_next_op"].value == 1:
            remove_padded_column_shape = align_shape
            remove_padded_column_shape[-2] = res_shape_nchw_after_removepad[-2].value // 2 # rm padded column and pad
            res_shape_nchw_after_removepad = x.op.attrs["true_conv_shape"]
            x_shape = res_shape_nchw_after_removepad
            res_f16 = tvm.compute(remove_padded_column_shape,
                                  lambda batch, cout1, howo, cout0: res_f16(batch, cout1, howo * 2, cout0),
                                  name='dequant_remove_padded_column',
                                  tag='dequant_remove_padded_column')
        if invalid_data_rm_flag:
            res = tvm.compute(res_f16.shape,
                              lambda batch, cout1, howo, cout0: res_f16(batch, cout1, howo, cout0),
                              name='invalid_dequant_rmpad',
                              tag="invalid_dequant_rmpad")
        else:
            res = tvm.compute(res_shape_nchw_after_removepad,
                              lambda batch, cout1, howo, cout0: res_f16(batch, cout1, howo, cout0),
                              name='dequant_remove_pad',
                              tag="dequant_remove_pad")
    else:
        if relu_flag:
            res_f16 = tvm.compute(align_shape,
                                  lambda batch, c1, hw, c0: tvm.relu(
                                      x(batch, c1, hw, c0).astype("float16") * deq_scale(0, c1, 0, 0, c0)),
                                  name="dequant1", tag="dequant1_vector", attrs={"relu_flag": 1})
        else:
            res_f16 = tvm.compute(align_shape,
                                  lambda batch, c1, hw, c0: x(batch, c1, hw, c0).astype("float16") * deq_scale(
                                      0, c1, 0, 0, c0),
                                  name="dequant1", tag="dequant1_vector", attrs={"relu_flag": 0})

        res = tvm.compute(x_shape,
                          lambda *indice: res_f16(*indice), name="dequant_remove_pad", tag="dequant_remove_pad")

    if sqrt_mode:
        res = tvm.compute(x_shape,
                          lambda batch, c1, hw, c0: (res(batch, c1, hw, c0) * deq_scale(0, c1, 0, 0, c0)),
                          name="dequant2", tag="dequant2_vector")
    return res


def _scalar_dequant_v100(x, x_shape, align_shape, deq_scale, flags):
    """
    dequant for scale in v100
    """
    _, relu_flag, sqrt_mode, conv_flag = flags
    if conv_flag:
        invalid_data_rm_flag = int(x.op.attrs["invalid_data_rm_flag"])
        group = x.op.input_tensors[0].shape[0].value
        cout1_opt = x.op.input_tensors[0].shape[2].value
        res_shape_nchw_after_removepad = x.op.attrs["conv_shape"]
        res_f16 = tvm.compute(
            align_shape,
            lambda batch, cout1, howo, cout0:
            x.op.input_tensors[0](
                0 if group == 1 else cout1 // cout1_opt, batch,
                cout1 if group == 1 else cout1 % cout1_opt, howo, cout0).astype("float16") *
            deq_scale(0, 0, 0, 0, 0),
            name="dequant1", tag="dequant1_scale")

        if x.op.attrs["remove_padded_column_in_next_op"].value == 1:
            remove_padded_column_shape = align_shape
            remove_padded_column_shape[-2] = res_shape_nchw_after_removepad[-2].value // 2 # remove padded column
            res_shape_nchw_after_removepad = x.op.attrs["true_conv_shape"]
            res_f16 = tvm.compute(remove_padded_column_shape,
                                  lambda batch, cout1, howo, cout0: res_f16(batch, cout1, howo * 2, cout0),
                                  name='dequant_remove_padded_column',
                                  tag='dequant_remove_padded_column')

        if invalid_data_rm_flag:
            res = tvm.compute(res_f16.shape,
                              lambda batch, cout1, howo, cout0: res_f16(batch, cout1, howo, cout0),
                              name='invalid_dequant_rmpad',
                              tag="invalid_dequant_rmpad")
        else:
            res = tvm.compute(res_shape_nchw_after_removepad,
                              lambda batch, cout1, howo, cout0: res_f16(batch, cout1, howo, cout0),
                              name='dequant_remove_pad',
                              tag="dequant_remove_pad")

        x_shape = res_shape_nchw_after_removepad

    else:
        res_f16 = tvm.compute(align_shape,
                              lambda batch, c1, hw, c0: (x(batch, c1, hw, c0).astype("float16") * deq_scale(
                                  0, 0, 0, 0, 0)),
                              name="dequant1", tag="dequant1_scale")
        res = tvm.compute(x_shape,
                          lambda *indice: res_f16(*indice), name="dequant_remove_pad", tag="dequant_remove_pad")

    if relu_flag:
        res = tvm.compute(x_shape, lambda *indices: tvm.relu(res(*indices)), name="dequant_relu", tag="dequant_relu")
    if sqrt_mode:
        res = tvm.compute(x_shape,
                          lambda batch, c1, hw, c0: (res(batch, c1, hw, c0) * deq_scale(0, 0, 0, 0, 0)),
                          name="dequant2", tag="dequant2_scale")

    return res


def _vector_dequant_v200(x, x_shape, align_shape, deq_scale, flags):
    """
    dequant for vector in v200
    """
    _, relu_flag, _, conv_flag = flags
    if conv_flag:
        invalid_data_rm_flag = int(x.op.attrs["invalid_data_rm_flag"])
        group = x.op.input_tensors[0].shape[0].value
        cout1_opt = x.op.input_tensors[0].shape[2].value
        res_shape_nchw_after_removepad = x.op.attrs["conv_shape"]

        res_f16 = tvm.compute(
            align_shape,
            lambda batch, cout1, howo, cout0:
            tvm.vdeq_cast(
                x.op.input_tensors[0](
                    0 if group == 1 else cout1 // cout1_opt, batch,
                    cout1 if group == 1 else cout1 % cout1_opt, howo, cout0),
                deq_scale(0, cout1, 0, 0, cout0), dtype="float16", do_relu=relu_flag),
            name="dequant", tag="dequant_vector")

        if x.op.attrs["remove_padded_column_in_next_op"].value == 1:
            remove_padded_column_shape = align_shape
            remove_padded_column_shape[-2] = res_shape_nchw_after_removepad[-2].value // 2 # rm padded column and pad
            res_shape_nchw_after_removepad = x.op.attrs["true_conv_shape"]
            res_f16 = tvm.compute(remove_padded_column_shape,
                                  lambda batch, cout1, howo, cout0: res_f16(batch, cout1, howo * 2, cout0),
                                  name='dequant_remove_padded_column',
                                  tag='dequant_remove_padded_column')

        if invalid_data_rm_flag:
            res = tvm.compute(res_f16.shape,
                              lambda batch, cout1, howo, cout0: res_f16(batch, cout1, howo, cout0),
                              name='invalid_dequant_rmpad',
                              tag="invalid_dequant_rmpad")
        else:
            res = tvm.compute(res_shape_nchw_after_removepad,
                              lambda batch, cout1, howo, cout0: res_f16(batch, cout1, howo, cout0),
                              name='dequant_remove_pad',
                              tag="dequant_remove_pad")
    else:
        res_f16 = tvm.compute(
            align_shape,
            lambda batch, c1, hw, c0: tvm.vdeq_cast(x(batch, c1, hw, c0), deq_scale(
                0, c1, 0, 0, c0), dtype="float16", do_relu=relu_flag),
            name="dequant", tag="dequant_vector")
        res = tvm.compute(x_shape,
                          lambda *indice: res_f16(*indice), name="dequant_remove_pad", tag="dequant_remove_pad")

    return res


def _scalar_depthwise_fused_v100(x, x_shape, align_shape, deq_scale, flags):
    """
    dequant for vector in v100
    """
    _, relu_flag, sqrt_mode, _ = flags
    if relu_flag:
        res_f16 = tvm.compute(align_shape,
                              lambda batch, c1, ho, wo, c0: tvm.relu(
                                  x(batch, c1 // 2, c1 % 2, wo, c0).astype("float16") * deq_scale(0, 0, 0, 0, 0)),
                              name="dequant1", tag="dequant1_vector", attrs={"relu_flag": 1})

    else:
        res_f16 = tvm.compute(align_shape,
                              lambda batch, c1, ho, wo, c0: x(
                                  batch, c1 // 2, c1 % 2, wo, c0).astype("float16") * deq_scale(0, 0, 0, 0, 0),
                              name="dequant1", tag="dequant1_vector", attrs={"relu_flag": 0})

    align_shape[3] = x_shape[3].value

    if not sqrt_mode:
        res = tvm.compute(align_shape,
                          lambda *indice: res_f16(*indice), name="dequant_remove_pad",
                          tag="dequant_remove_pad", attrs={"sqrt_flag": 0})
    else:
        res_sqrt = tvm.compute(align_shape,
                               lambda batch, c1, ho, wo, c0: (res_f16(batch, c1, ho, wo, c0) * deq_scale(
                                   0, 0, 0, 0, 0)),
                               name="dequant2", tag="dequant2_vector")

        res = tvm.compute(align_shape, lambda *indice: res_sqrt(*indice), name="dequant2_remove_pad",
                          tag="dequant2_remove_pad", attrs={"sqrt_flag": 1})
    return res


def _vector_depthwise_fused_v100(x, x_shape, align_shape, deq_scale, flags):
    """
    dequant for vector in v100
    """
    _, relu_flag, sqrt_mode, _ = flags
    if relu_flag:
        res_f16 = tvm.compute(align_shape,
                              lambda batch, c1, ho, wo, c0: tvm.relu(
                                  x(batch, c1 // 2, c1 % 2, wo, c0).astype("float16") * deq_scale(0, c1, 0, 0, c0)),
                              name="dequant1", tag="dequant1_vector", attrs={"relu_flag": 1})
    else:
        res_f16 = tvm.compute(align_shape,
                              lambda batch, c1, ho, wo, c0: x(
                                  batch, c1 // 2, c1 % 2, wo, c0).astype("float16") * deq_scale(0, c1, ho, 0, c0),
                              name="dequant1", tag="dequant1_vector", attrs={"relu_flag": 0})

    align_shape[3] = x_shape[3].value

    if not sqrt_mode:
        res = tvm.compute(align_shape,
                          lambda *indice: res_f16(*indice), name="dequant_remove_pad",
                          tag="dequant_remove_pad", attrs={"sqrt_flag": 0})
    else:
        res_sqrt = tvm.compute(align_shape,
                               lambda batch, c1, ho, wo, c0: (res_f16(batch, c1, ho, wo, c0) * deq_scale(
                                   0, c1, ho, 0, c0)),
                               name="dequant2", tag="dequant2_vector")

        res = tvm.compute(align_shape,
                          lambda *indice: res_sqrt(*indice), name="dequant2_remove_pad",
                          tag="dequant2_remove_pad", attrs={"sqrt_flag": 1})
    return res


def _vector_depthwise_fused_v200(x, x_shape, align_shape, deq_scale, relu_flag):
    """
    depthwise dequant for vector in v200
    """
    res_f16 = tvm.compute(align_shape,
                          lambda batch, c1, ho, wo, c0: tvm.vdeq_cast(
                              x(batch, c1 // 2, c1 % 2, wo, c0), deq_scale(0, c1, 0, 0, c0), dtype="float16",
                              do_relu=relu_flag),
                          name="dequant1", tag="dequant1_vector", attrs={"relu_flag": relu_flag})

    align_shape[3] = x_shape[3].value

    res = tvm.compute(align_shape, lambda *indice: res_f16(*indice), name="dequant_remove_pad",
                      tag="dequant_remove_pad", attrs={"sqrt_flag": 0})

    return res


def _scalar_depthwise_fused_v200(x, x_shape, align_shape, deq_scale):
    """
    depthwise dequant for vector in v200
    """
    res_f16 = tvm.compute(align_shape,
                          lambda batch, c1, ho, wo, c0: tvm.deq_cast(
                              x(batch, c1 // 2, c1 % 2, wo, c0), deq_scale(0, 0, 0, 0, 0), dtype="float16"),
                          name="dequant1", tag="dequant1_scale")

    align_shape[3] = x_shape[3].value

    res = tvm.compute(align_shape, lambda *indice: res_f16(*indice), name="dequant_remove_pad",
                      tag="dequant_remove_pad", attrs={"sqrt_flag": 0})

    return res


def _scalar_dequant_v200(x, x_shape, align_shape, deq_scale, conv_flag):
    """
    dequant for scale in v200
    """
    if conv_flag:
        invalid_data_rm_flag = int(x.op.attrs["invalid_data_rm_flag"])
        group = x.op.input_tensors[0].shape[0].value
        cout1_opt = x.op.input_tensors[0].shape[2].value
        res_shape_nchw_after_removepad = x.op.attrs["conv_shape"]

        res_f16 = tvm.compute(
            align_shape,
            lambda batch, cout1, howo, cout0:
            tvm.deq_cast(x.op.input_tensors[0](0 if group == 1 else cout1 // cout1_opt, batch,
                                               cout1 if group == 1 else cout1 % cout1_opt, howo, cout0),
                         deq_scale(0, 0, 0, 0, 0), dtype="float16"),
            name="dequant", tag="dequant_scale")

        if x.op.attrs["remove_padded_column_in_next_op"].value == 1:
            remove_padded_column_shape = align_shape
            remove_padded_column_shape[-2] = res_shape_nchw_after_removepad[-2].value // 2 # remove padded column
            res_shape_nchw_after_removepad = x.op.attrs["true_conv_shape"]
            res_f16 = tvm.compute(remove_padded_column_shape,
                                  lambda batch, cout1, howo, cout0: res_f16(batch, cout1, howo * 2, cout0),
                                  name='dequant_remove_padded_column',
                                  tag='dequant_remove_padded_column')

        if invalid_data_rm_flag:
            res = tvm.compute(res_f16.shape,
                              lambda batch, cout1, howo, cout0: res_f16(batch, cout1, howo, cout0),
                              name='invalid_dequant_rmpad',
                              tag="invalid_dequant_rmpad")
        else:
            res = tvm.compute(res_shape_nchw_after_removepad,
                              lambda batch, cout1, howo, cout0: res_f16(batch, cout1, howo, cout0),
                              name='dequant_remove_pad',
                              tag="dequant_remove_pad")
    else:
        res_f16 = tvm.compute(align_shape,
                              lambda batch, c1, hw, c0: tvm.deq_cast(
                                  x(batch, c1, hw, c0), deq_scale(0, 0, 0, 0, 0), dtype="float16"),
                              name="dequant", tag="dequant_scale")
        res = tvm.compute(x_shape,
                          lambda *indice: res_f16(*indice), name="dequant_remove_pad", tag="dequant_remove_pad")
    return res


# 'pylint: disable=huawei-too-many-arguments
@register_operator_compute("AscendDequant", op_mode="dynamic", support_fusion=True)
def ascend_dequant_compute(x, deq_scale, y, sqrt_mode=False, relu_flag=False, dtype=1, kernel_name="ascend_dequant"):
    """
    int32 -> fp16

    Parameters:
    ----------
    x: the placeholder of input
    deq_scale: the placeholder of deq_scale
    y: the dict of output
    sqrt_mode: the sqrt mode, when true the result to do sqrt
    relu_flag: the relu mode, when true the result to do relu
    kernel_name: cce kernel name, default value is "ascend_dequant"

    Returns:
    -------
    res : the result of ascend_dequant
    """
    conv_flag = 0
    if len(x.op.input_tensors) and ('mad1' in x.op.input_tensors[0].name or \
            'convolution_c_col_bias' in x.op.input_tensors[0].name):
        conv_flag = 1

    x_shape = x.shape
    deq_shape = deq_scale.shape
    x_shape_list = shape_util.shape_to_list(x_shape)
    deq_shape_list = shape_util.shape_to_list(deq_shape)
    ori_shape_deq = deq_scale.op.attrs["ori_shape"]
    ori_shape_deq_list = shape_util.shape_to_list(ori_shape_deq)
    deq_dim = functools.reduce(lambda x, y: x * y, ori_shape_deq_list[:])
    tensor_flag = False
    if isinstance(deq_scale, (tvm.Tensor)) or deq_dim > 1:
        tensor_flag = True
    flags = [tensor_flag, relu_flag, sqrt_mode, conv_flag]
    if conv_flag:
        x_input_shape = shape_util.shape_to_list(x.op.input_tensors[0].shape)
        align_shape = [x_input_shape[1],
                       x.shape[1],
                       x_input_shape[3],
                       x_input_shape[4]]
    else:
        align_shape = x_shape_list.copy()

    if x.op.tag != "depthwise_conv2d":
        align_shape[2] = (align_shape[2] + 15) // 16 * 16

    if x.op.tag == "matmul" or x.op.tag == "matmul_gemv" or x.op.tag == "matmul_gevm":
        shape_matmul_origin = x.op.attrs["shape"]
        c1_index = len(x_shape) - 4
        res = _matmul_compute(x, deq_scale, shape_matmul_origin, c1_index, flags)
        return res
    if x.op.tag == "depthwise_conv2d":
        align_shape[4] = 16
        align_shape[3] = (x_shape_list[3] + 15) // 16 * 16
        align_shape[2] = 1
        if deq_shape_list[1] == 1:
            tensor_dict = util.get_depthwise_conv2d_tensor_info(x, is_dequant=True)
            x_ori_shape = tensor_dict.get("fmap").op.attrs["ori_shape"]
            x_ori_format = tensor_dict.get("fmap").op.attrs["ori_format"]
            x_ori_shape_list = shape_util.shape_to_list(x_ori_shape)
            if x_ori_format == "NCHW":
                align_shape[1] = (x_ori_shape_list[1] + 15) // 16
            else:
                align_shape[1] = (x_ori_shape_list[3] + 15) // 16
        else:
            align_shape[1] = (deq_shape_list[1] * deq_shape_list[4]) // 16
        align_shape[0] = x_shape_list[0]

        if tensor_flag:
            if _is_support_v200_instruction():
                res = _vector_depthwise_fused_v200(x, x_shape, align_shape, deq_scale, relu_flag)
            else:
                res = _vector_depthwise_fused_v100(x, x_shape, align_shape, deq_scale, flags)
        else:
            if _is_support_v200_instruction():
                res = _scalar_depthwise_fused_v200(x, x_shape, align_shape, deq_scale)
            else:
                res = _scalar_depthwise_fused_v100(x, x_shape, align_shape, deq_scale, flags)

        return res

    if tensor_flag:
        if _is_support_v200_instruction():
            res = _vector_dequant_v200(x, x_shape, align_shape, deq_scale, flags)
        else:
            res = _vector_dequant_v100(x, x_shape, align_shape, deq_scale, flags)
    else:
        if _is_support_v200_instruction():
            res = _scalar_dequant_v200(x, x_shape, align_shape, deq_scale, conv_flag)
        else:
            res = _scalar_dequant_v100(x, x_shape, align_shape, deq_scale, flags)

    return res


# 'pylint: disable=huawei-too-many-arguments
@register_operator("AscendDequant", pattern="dequant")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def ascend_dequant(x, deq_scale, y, sqrt_mode=False, relu_mode=False, dtype=1, kernel_name="ascend_dequant"):
    """
    int32 -> fp16

    Parameters:
    ----------
    x: the dict of input
    deq_scale: the dict of dequant num
    offset: the dict of offset num
    y: the dict of output
    sqrt_mode: the sqrt mode when true the result to do sqrt
    relu_flag: the relu mode when true the result to do relu
    kernel_name: cce kernel name, default value is "ascend_dequant"

    Returns:
    -------
    None
    """

    _check_params(x, deq_scale, sqrt_mode, kernel_name)

    dtype_x = x.get("dtype")
    shape_deq = deq_scale.get("shape")
    dtype_deq = deq_scale.get("dtype")
    ori_shape_deq = deq_scale.get("ori_shape")
    attr = {"ori_shape": ori_shape_deq}

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
        input_deq = tvm.placeholder(shape_deq, name="deq_scale", dtype=dtype_deq, attrs=attr)
        res = ascend_dequant_compute(input_x, input_deq, y, sqrt_mode, relu_mode, kernel_name)

        tensors.append([input_x, input_deq, res])

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
