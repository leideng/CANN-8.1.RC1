#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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
axpy_v2
"""

import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import error_manager_vector
from tbe.common.utils.para_check import check_op_params
from tbe.common.utils.para_check import check_dtype
from tbe.common.utils.para_check import REQUIRED_INPUT
from tbe.common.utils.para_check import OPTION_OUTPUT
from tbe.common.utils.para_check import KERNEL_NAME
from tbe.common.utils.shape_util import refine_shapes_for_broadcast
from impl.util.platform_adapter import tbe_platform
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.util import util_common
from impl.constant_util import SIZE_SIXTEEN
from impl.constant_util import SHAPE_SIZE_LIMIT


def generate_param(dtypes, formats):
    """
    generate param
    """
    dtype_x1, dtype_x2, dtype_alpha, dtype_output = dtypes
    format_x1, format_x2, format_alpha, format_output = formats

    input0 = gen_param(classify="input0", name="x1",
                       datatype=",".join(dtype_x1),
                       format=",".join(format_x1))
    input1 = gen_param(classify="input1", name="x2",
                       datatype=",".join(dtype_x2),
                       format=",".join(format_x2))
    input2 = gen_param(classify="input2", name="alpha",
                       datatype=",".join(dtype_alpha),
                       format=",".join(format_alpha))
    output0 = gen_param(classify="output0", name="y",
                        datatype=",".join(dtype_output),
                        format=",".join(format_output))
    return [input0, input1, input2, output0]


def get_format_same(dtype_list, format_list, dtype_total, alpha_dtypes, alpha_formats):
    """
    get format
    """
    for dtype in dtype_list:
        dtype_total = dtype_total + [dtype] * len(format_list) * len(alpha_dtypes)

    alpha_dtypes_new = []
    for dtype in alpha_dtypes:
        alpha_dtypes_new = alpha_dtypes_new + [dtype] * (len(dtype_total) // len(alpha_dtypes))
    alpha_formats = alpha_formats * (len(dtype_total) // len(alpha_dtypes))
    format_list = format_list * len(dtype_list) * len(alpha_dtypes)

    dtypes = [dtype_total, dtype_total, alpha_dtypes_new, dtype_total]
    formats = [format_list, format_list, alpha_formats, format_list]
    return dtypes, formats


# 'pylint: disable=too-many-arguments
def get_format_mix(dtype_list, format_list, dtype_total, alpha_dtypes, alpha_formats, len_format_list, format_nz,
                   format_nd):
    """
    get format
    """
    for dtype in dtype_list:
        dtype_total = dtype_total + [dtype] * len(format_list)
    format_list = format_list * len_format_list
    for dtype in dtype_list:
        dtype_total = dtype_total + [dtype] * 1

    alpha_formats_new = alpha_formats * len(dtype_total)
    alpha_dtypes_new = alpha_dtypes * len(dtype_total)
    dtype_total = dtype_total * len(alpha_dtypes)
    format_list0 = format_list + format_nz * len_format_list
    format_list1 = format_list + format_nd * len_format_list
    format_list0 = format_list0 * len(alpha_dtypes)
    format_list1 = format_list1 * len(alpha_dtypes)

    dtypes = [dtype_total, dtype_total, alpha_dtypes_new, dtype_total]
    formats0 = [format_list0, format_list1, alpha_formats_new, format_list0]
    formats1 = [format_list1, format_list0, alpha_formats_new, format_list1]
    formats2 = [format_list1, format_list0, alpha_formats_new, format_list0]
    return dtypes, formats0, formats1, formats2


def _can_broad(x, y):
    if x[2]:
        x[0] *= 16
        y[0] *= 16
    if x[3]:
        x[1] *= 16
        y[1] *= 16
    return (x[0] == y[0] and (x[1] == 16 or y[1] == 16 or x[1] == y[1])) or (
            x[1] == y[1] and (x[0] == 16 or y[0] == 16)) or x[0] == y[1] == 16 or x[0] == x[1] == 16 or x[1] == y[
               0] == 16 or y[0] == y[1] == 16


# op select format
# 'pylint: disable=unused-argument,too-many-locals,too-many-boolean-expressions,too-many-nested-blocks
# 'pylint: disable=too-many-branches,too-many-statements
def op_select_format(input_x, input_y, alpha, output_z, kernel_name="axpy_v2"):
    """
    select format dynamically
    """

    def _can_division_sixteen(shape):
        if len(shape) < 2:
            if shape[-1] == 0:
                expected_value = "equal to 0"
                real_value = "not equal to 0"
                error_manager_vector.raise_err_input_value_invalid("axpy", "value of shape", expected_value, real_value)
            return False
        if shape[-1] == 0 or shape[-2] == 0:
            raise RuntimeError("value of shape is illegal")

        if shape[-1] % SIZE_SIXTEEN == 0 and shape[-2] % SIZE_SIXTEEN == 0:
            return True

        return False

    shape_x = input_x.get("ori_shape")
    shape_y = input_y.get("ori_shape")

    shape_x = shape_util.scalar2tensor_one(shape_x)
    shape_y = shape_util.scalar2tensor_one(shape_y)

    format_4d_list = ["NCHW", "NHWC", "HWCN"]
    format_5d_list = ["NDHWC", "DHWCN", "NCDHW"]
    cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        dtype_list = ["float16", "int32"]
    elif cce_product == "Ascend910B":
        dtype_list = ["bfloat16", "float16", "float32", "int32"]
        alpha_dtypes = ["bfloat16", "float16", "float32", "int32"]
        alpha_formats = ["ND", "ND", "ND", "ND"]
    else:
        dtype_list = ["float16", "float32", "int32"]
        alpha_dtypes = ["float16", "float32", "int32"]
        alpha_formats = ["ND", "ND", "ND"]

    format_x = input_x.get("ori_format")
    format_y = input_y.get("ori_format")

    dtype_total = []
    format_nd = ["ND"]
    format_list = ["ND"]
    format_nz = ["FRACTAL_NZ"]
    len_format_list = len(dtype_list)
    list_input = [input_x, input_y]

    # if shape is same, then all formats are supported.
    if list(shape_x) == list(shape_y):
        format_list = ["ND", "FRACTAL_NZ", "NC1HWC0", "FRACTAL_Z", "C1HWNCoC0"]
        dtypes, formats = get_format_same(dtype_list, format_list, dtype_total, alpha_dtypes, alpha_formats)
        input0, input1, input2, output0 = generate_param(dtypes, formats)

        param_list = [input0, input1, input2, output0]
        param_dynamic_in_json = get_dynamic_param_in_json(param_list)
        return param_dynamic_in_json

    x_flag = {"5d": len(shape_x) == 5 and format_x in format_5d_list,
              "4d": len(shape_x) == 4 and format_x in format_4d_list,
              "Scalar": len(shape_x) == 1 and shape_x[0] == 1}
    y_flag = {"5d": len(shape_y) == 5 and format_y in format_5d_list,
              "4d": len(shape_y) == 4 and format_y in format_4d_list,
              "Scalar": len(shape_y) == 1 and shape_y[0] == 1}
    common_flag = {"half_16_div_flg": (_can_division_sixteen(shape_x) and not _can_division_sixteen(shape_y)) or (
            not _can_division_sixteen(shape_x) and _can_division_sixteen(shape_y))}
    if x_flag.get("5d") or x_flag.get("4d"):
        x_cdim = shape_x[format_x.index("C")]
        x_ndim = shape_x[format_x.index("N")]
    if y_flag.get("5d") or y_flag.get("4d"):
        y_cdim = shape_y[format_y.index("C")]
        y_ndim = shape_y[format_y.index("N")]

    format_flag = {"NDC1HWC0": x_flag.get("5d") and y_flag.get("5d") and x_cdim == y_cdim,
                   "FRACTAL_Z_3D": x_flag.get("5d") and y_flag.get("5d") and x_cdim == y_cdim and x_ndim == y_ndim,
                   "FRACTAL_NZ": len(shape_x) >= 2 and len(shape_y) >= 2 and shape_x[-2:] == shape_y[-2:],
                   "NC1HWC0": x_flag.get("4d") and
                              y_flag.get("4d") and
                              ((format_y == format_x and
                                ((x_cdim % 16 == 0 and y_cdim % 16 == 0) or x_cdim == y_cdim) and _can_broad(
                               [shape_x[format_x.index(format_x[0])], shape_x[format_x.index(format_x[1])],
                                format_x[0] != "C", format_x[1] != "C"],
                               [shape_y[format_y.index(format_y[0])], shape_y[format_y.index(format_y[1])],
                                format_y[0] != "C", format_y[1] != "C"])) or (
                                list(shape_x) == list(shape_y) and -1 not in shape_x) or (
                                common_flag.get("half_16_div_flg") and
                                (x_cdim == y_cdim or x_cdim == 16 or y_cdim == 16))),
                   "FRACTAL_Z": x_flag.get("4d") and y_flag.get("4d") and format_x == format_y and (
                           (all(i % 16 == 0 for i in [x_cdim, y_cdim, x_ndim, y_ndim])
                            and util_common.is_support_fractal_z_inputs(list_input)
                            and ((list(shape_x) == list(shape_y) and format_x.upper() in ("NCHW", "NHWC")) or
                                 (format_x.upper() == "HWCN" and shape_x[0]*shape_x[1] == shape_y[0]*shape_y[1])))
                           or (list(shape_x) == list(shape_y) and
                               util_common.is_support_fractal_z_inputs(list_input))),
                   "ND": True
                   }

    format_flag["NC1HWC0"] = format_flag.get("NC1HWC0") or (
            x_flag.get("4d") and y_flag.get("Scalar") and x_cdim % 16 == 0) or (
            x_flag.get("Scalar") and y_flag.get("4d") and y_cdim % 16 == 0) or (
            len(shape_x) == 1 and len(shape_y) == 1 and shape_x[0] % 16 == 0 and shape_y[0] % 16 == 0)
    format_flag["FRACTAL_Z"] = format_flag.get("FRACTAL_Z") or \
                               (util_common.is_support_fractal_z_inputs(list_input) and
                               (x_flag.get("4d") and y_flag.get("Scalar") and x_cdim % 16 == 0 and x_ndim % 16 == 0) or
                               (x_flag.get("Scalar") and y_flag.get("4d") and y_cdim % 16 == 0 and y_ndim % 16 == 0))


    format_list = [i for i in format_flag if format_flag.get(i)]
    # ND+ND NZ+NZ 5HD+5HD FZ+FZ
    if len(shape_x) >= 2 and len(shape_y) >= 2 and shape_x[-2:] == shape_y[-2:]:


        dtypes, formats = get_format_same(dtype_list, format_list, dtype_total, alpha_dtypes, alpha_formats)
        input0, input1, input2, output0 = generate_param(dtypes, formats)

    # NZ+ND,ND+ND,5HD+5HD,FZ+FZ,ND+NZ
    elif len(shape_x) >= 2 and len(shape_y) >= 2 and \
            ((_can_division_sixteen(shape_x) and
              not _can_division_sixteen(shape_y)) or
             (not _can_division_sixteen(shape_x) and
              _can_division_sixteen(shape_y))):

        dtypes, formats0, formats1, _ = get_format_mix(dtype_list, format_list, dtype_total, alpha_dtypes,
                                                       alpha_formats, len_format_list, format_nz, format_nd)

        if _can_division_sixteen(shape_x) and not _can_division_sixteen(shape_y):
            input0, input1, input2, output0 = generate_param(dtypes, formats0)
        else:
            input0, input1, input2, output0 = generate_param(dtypes, formats1)

    # 5HD+scalar,ND+ND,FZ+scalar
    elif len(shape_x) >= 2 and len(shape_y) == 1 and shape_y[0] == 1:
        dtypes, formats0, _, _ = get_format_mix(dtype_list, format_list, dtype_total, alpha_dtypes,
                                                alpha_formats, len_format_list, format_nz, format_nd)
        input0, input1, input2, output0 = generate_param(dtypes, formats0)

    # ND+ND,scalar+5HD,scalar+FZ
    elif len(shape_y) >= 2 and len(shape_x) == 1 and shape_x[0] == 1:
        dtypes, _, _, formats2 = get_format_mix(dtype_list, format_list, dtype_total, alpha_dtypes,
                                                alpha_formats, len_format_list, format_nz, format_nd)
        input0, input1, input2, output0 = generate_param(dtypes, formats2)
    # ND+ND,5HD+5HD
    else:
        dtypes, formats = get_format_same(dtype_list, format_list, dtype_total, alpha_dtypes, alpha_formats)
        input0, input1, input2, output0 = generate_param(dtypes, formats)

    param_list = [input0, input1, input2, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def _add_check_format(x, y):
    shape1 = x.get("shape")
    shape2 = y.get("shape")
    list_format = [x.get("format"), y.get("format")]
    shape1 = shape_util.scalar2tensor_one(shape1)
    shape2 = shape_util.scalar2tensor_one(shape2)

    format_list = ("ND", "NCHW", "NHWC")
    if list_format[0] == "FRACTAL_NZ" and list_format[1] in format_list \
            and (len(shape2) != 1 or (len(shape2) == 1 and shape2[0] != 1)):
        format_pattern = 1
    elif list_format[0] in format_list and list_format[1] == "FRACTAL_NZ" \
            and (len(shape1) != 1 or (len(shape1) == 1 and shape1[0] != 1)):
        format_pattern = 2
    else:
        format_pattern = 0

    return format_pattern


def _infer_shape(format_pattern, x, y):
    shape_x = x.get("shape")
    shape_y = y.get("shape")
    ori_shape_x = x.get("ori_shape")
    ori_shape_y = y.get("ori_shape")
    shape_x = shape_util.scalar2tensor_one(shape_x)
    shape_y = shape_util.scalar2tensor_one(shape_y)

    if format_pattern == 1:
        ori_shape_x, shape_y, _ = shape_util.produce_shapes(ori_shape_x, shape_y)

        if shape_y[-2] == 1 and shape_y[-1] == ori_shape_x[-1]:
            shape_y.append(1)
            shape_y.append(1)
            shape_y[-3] = 1
            shape_y[-1] = shape_x[-1]
            shape_y[-4] = shape_x[-4]

        elif shape_y[-2] == ori_shape_x[-2] and shape_y[-1] == 1:
            shape_y.append(1)
            shape_y.append(1)
            shape_y[-4] = 1
            shape_y[-2] = shape_x[-2]
            shape_y[-3] = shape_x[-3]

        elif shape_y[-2] == shape_y[-1] == 1:
            shape_y.append(1)
            shape_y.append(1)

    elif format_pattern == 2:
        shape_x, ori_shape_y, _ = shape_util.produce_shapes(shape_x, ori_shape_y)

        if shape_x[-2] == 1 and shape_x[-1] == ori_shape_y[-1]:
            shape_x.append(1)
            shape_x.append(1)
            shape_x[-3] = 1
            shape_x[-1] = shape_y[-1]
            shape_x[-4] = shape_y[-4]

        elif shape_x[-2] == ori_shape_y[-2] and shape_x[-1] == 1:
            shape_x.append(1)
            shape_x.append(1)
            shape_x[-4] = 1
            shape_x[-2] = shape_y[-2]
            shape_x[-3] = shape_y[-3]

        elif shape_x[-2] == shape_x[-1] == 1:
            shape_x.append(1)
            shape_x.append(1)

    return shape_x, shape_y


@register_operator_compute("axpy_v2", op_mode="static", support_fusion=True)
def axpy_v2_compute(x1, x2, alpha, y, kernel_name="axpy_v2"):
    """
    calculating data

    Parameters
    ----------
    x1 : TVM tensor
    the placeholder of input_x
    x2 : TVM tensor
    the placeholder of x2
    y : dict
    dict of y, include keys(shape and dtype)
    alpha : TVM tensor
    scalar of mul-factor
    kernel_name : str
    kernel name, default value is "axpy_v2"

    Returns
    -------
    output tensor
    """
    # broadcast
    shape_x1 = shape_util.shape_to_list(x1.shape)
    shape_x2 = shape_util.shape_to_list(x2.shape)
    dtype_alpha = alpha.dtype.lower()
    dtype = x1.dtype.lower()
    precision_dtype = "float32"

    # cast dtype
    if dtype in ("float16", "float32"):
        if dtype_alpha != dtype:
            alpha = tbe.cast_to(alpha, dtype)

    if dtype == "int32":
        x1 = tbe.cast_to(x1, precision_dtype)
        x2 = tbe.cast_to(x2, precision_dtype)
        if dtype_alpha != precision_dtype:
            alpha = tbe.cast_to(alpha, precision_dtype)

    if shape_x1 != shape_x2:
        # if shape not equal, then apply broadcast.
        _, _, shape_max = shape_util.produce_shapes(shape_x1, shape_x2)
        x1 = tbe.broadcast(x1, shape_max)
        x2 = tbe.broadcast(x2, shape_max)
        alpha = tbe.broadcast(alpha, shape_max)
    else:
        alpha = tbe.broadcast(alpha, shape_x1)

    res = tbe.vmla(x2, alpha, x1)
    if dtype == "int32":
        res = tbe.cast_to(res, dtype)
    return res


# 'pylint: disable=unused-argument,too-many-locals
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, OPTION_OUTPUT, KERNEL_NAME)
def axpy_v2(x1, x2, alpha, y, kernel_name="axpy_v2"):
    """
    calculating data of axpy

    Parameters
    ----------
    x1 : dict
    shape and dtype of input_x
    x2 : dict
    shape and dtype of input_y
    alpha : dict
    shape and dtype of alpha
    scalar apply to input_y:input_y*alpha
    y : dict
    shape and dtype of output, should be same shape and type as input

    kernel_name : str
    kernel name, default value is "axpy"

    Returns
    -------
    None
    """
    # check kernel name
    para_check.check_kernel_name(kernel_name)

    # infer shape according to the format pattern
    format_pattern = _add_check_format(x1, x2)

    shape_x1, shape_x2 = _infer_shape(format_pattern, x1, x2)

    dtype_x1 = x1.get("dtype").lower()
    dtype_x2 = x2.get("dtype").lower()
    alpha_dtype = alpha.get("dtype").lower()
    alpha_shape = alpha.get("shape")

    # check shape
    shape_x1 = shape_util.scalar2tensor_one(shape_x1)
    shape_x2 = shape_util.scalar2tensor_one(shape_x2)
    alpha_shape = shape_util.scalar2tensor_one(alpha_shape)
    para_check.check_shape(shape_x1)
    para_check.check_shape(shape_x2)
    para_check.check_shape(alpha_shape)

    # check dtype
    dtype_list0 = ("float16", "float32", "int32")

    check_dtype(dtype_x1, dtype_list0)
    check_dtype(dtype_x2, dtype_list0)
    check_dtype(alpha_dtype, dtype_list0)
    shape_util.compare_tensor_dict_key(x1, x2, "dtype")

    # check alpha is 0D or 1D tensor
    if len(alpha_shape) != 0 and not para_check.is_scalar(alpha_shape):
        raise RuntimeError("alpha should be 0D or 1D tensor")

    # produce shapes
    shape_x1, shape_x2, shape_max = shape_util.produce_shapes(shape_x1, shape_x2)
    if shape_x1[-1] == 1 and shape_x2[-1] == 1 and shape_max[-1] == 1:
        shape_x1 = shape_x1 if len(shape_x1) == 1 else shape_x1[:-1]
        shape_x2 = shape_x2 if len(shape_x2) == 1 else shape_x2[:-1]
        shape_max = shape_max if len(shape_max) == 1 else shape_max[:-1]

    para_check.check_shape_size(shape_max, SHAPE_SIZE_LIMIT)

    shape_x1, shape_x2 = refine_shapes_for_broadcast(shape_x1, shape_x2)
    alpha_shape = tuple([1] * (len(shape_x1) - len(alpha_shape))) + tuple(alpha_shape)

    data_input_x1 = tvm.placeholder(shape_x1, name="data_input_x1", dtype=dtype_x1)
    data_input_x2 = tvm.placeholder(shape_x2, name="data_input_x2", dtype=dtype_x2)
    alpha_input = tvm.placeholder(alpha_shape, name="alpha_input", dtype=alpha_dtype)

    res = axpy_v2_compute(data_input_x1, data_input_x2, alpha_input, y, kernel_name)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data_input_x1, data_input_x2, alpha_input, res]}

    tbe.cce_build_code(schedule, config)
