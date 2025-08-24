# Copyright 2020 Huawei Technologies Co., Ltd
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
axpy
"""
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util import util_common
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.util.platform_adapter import error_manager_vector
from impl.constant_util import SIZE_SIXTEEN


def _is_last_two_axis_16_multiple(shape):
    """
    whether last two axis can divide 16.

    Parameters
    ----------
    shape: list or tuple

    Returns:
    -------
    None
    """
    if shape[-1] == 0 or shape[-2] == 0:
        error_detail = "value of shape is illegal, shape[-1]:%s, shape[-2]:%s" % (shape[-1], shape[-2])
        error_manager_vector.raise_err_input_shape_invalid("axpy", "shape[-1], shape[-2]", error_detail)

    if shape[-1] % SIZE_SIXTEEN == 0 and shape[-2] % SIZE_SIXTEEN == 0:
        return True

    return False


def _is_support_nd_nz_nz(shape_1, shape_2):
    """
    _is_support_nd_nz_nz

    check the two shape like this:
    shapex  len >= 2
    shapey  len =1 and the size is 16 align and = shapex[-1]

    ex:
    shapex = [10, 10, 256, 256]
    shapey = [256]
    """
    return len(shape_1) >= 2 and len(shape_2) == 1 and shape_2[0] % SIZE_SIXTEEN == 0 and \
               shape_2[-1] == shape_1[-1]


# 'pylint: disable=unused-argument,too-many-nested-blocks,too-many-arguments
# 'pylint: disable=invalid-name,too-many-locals,too-many-branches
# 'pylint: disable=too-many-statements,too-many-boolean-expressions
def op_select_format(input_x, input_y, output_z, alpha, kernel_name="axpy"):
    """
    select format dynamically, supporting dynamic shape format selecting

    Parameters
    ----------
    input_x: dict
    dict of input_x, include keys(shape and dtype).
    input_y: dict
    dict of input_y, include keys(shape and dtype).
    output_z: dict
    dict of output_z, include keys(shape and dtype).
    alpha: float
    alpha value
    kernel_name: str
    kernel name, default value is axpy

    Returns:
    -------
    param_dynamic_in_json: dict
    dict of param_dynamic.
    """
    shape_x = input_x.get("ori_shape")
    shape_y = input_y.get("ori_shape")

    shape_x = shape_util.scalar2tensor_one(shape_x)
    shape_y = shape_util.scalar2tensor_one(shape_y)

    format_4d_list = ["NCHW", "NHWC", "HWCN"]

    cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    dtype_list = ["float16", "int32"]
    if cce_product not in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        dtype_list.append("float32")
    if tbe_platform.intrinsic_check_support("Intrinsic_vconv", "f322bf16f"):
        dtype_list.append("bfloat16")

    format_x = input_x.get("ori_format")
    format_y = input_y.get("ori_format")

    format_support_flag = {("ND", "ND", "ND"): 1,
                           ("FRACTAL_NZ", "FRACTAL_NZ", "FRACTAL_NZ"): 0,
                           ("NC1HWC0", "NC1HWC0", "NC1HWC0"): 0,
                           ("FRACTAL_Z", "FRACTAL_Z", "FRACTAL_Z"): 0,
                           ("C1HWNCoC0", "C1HWNCoC0", "C1HWNCoC0"): 0,
                           ("NC1HWC0", "ND", "NC1HWC0"): 0,
                           ("FRACTAL_Z", "ND", "FRACTAL_Z"): 0,
                           ("ND", "NC1HWC0", "NC1HWC0"): 0,
                           ("ND", "FRACTAL_NZ", "FRACTAL_NZ"): 0,
                           ("FRACTAL_NZ", "ND", "FRACTAL_NZ"): 0,
                           ("ND", "FRACTAL_Z", "FRACTAL_Z"): 0}

    list_input = [input_x, input_y]

    x_is_which_format = {"is_4d": len(shape_x) == 4 and format_x in format_4d_list,
                         "is_scalar": len(shape_x) == 1 and shape_x[0] == 1}
    y_is_which_format = {"is_4d": len(shape_y) == 4 and format_y in format_4d_list,
                         "is_scalar": len(shape_y) == 1 and shape_y[0] == 1}

    x_info = {"dim_n": 1, "dim_c": 1, "dim_h": 1, "dim_w": 1}
    y_info = {"dim_n": 1, "dim_c": 1, "dim_h": 1, "dim_w": 1}

    if (x_is_which_format.get("is_4d") and y_is_which_format.get("is_4d")):
        x_info["dim_c"] = shape_x[format_x.index("C")]
        x_info["dim_n"] = shape_x[format_x.index("N")]
        x_info["dim_h"] = shape_x[format_x.index("H")]
        x_info["dim_w"] = shape_x[format_x.index("W")]
        y_info["dim_c"] = shape_y[format_y.index("C")]
        y_info["dim_n"] = shape_y[format_y.index("N")]
        y_info["dim_h"] = shape_y[format_y.index("H")]
        y_info["dim_w"] = shape_y[format_y.index("W")]
    if y_is_which_format.get("is_scalar") and x_is_which_format.get("is_4d"):
        x_info["dim_c"] = shape_x[format_x.index("C")]
        x_info["dim_n"] = shape_x[format_x.index("N")]
    if x_is_which_format.get("is_scalar") and y_is_which_format.get("is_4d"):
        y_info["dim_c"] = shape_y[format_y.index("C")]
        y_info["dim_n"] = shape_y[format_y.index("N")]

    # if shape is same, then all formats are supported.
    if list(shape_x) == list(shape_y):
        format_support_flag[("ND", "ND", "ND")] = 1
        format_support_flag[("FRACTAL_NZ", "FRACTAL_NZ", "FRACTAL_NZ")] = 1
        format_support_flag[("NC1HWC0", "NC1HWC0", "NC1HWC0")] = 1
        format_support_flag[("FRACTAL_Z", "FRACTAL_Z", "FRACTAL_Z")] = 1
        format_support_flag[("C1HWNCoC0", "C1HWNCoC0", "C1HWNCoC0")] = 1

    # ND+ND NZ+NZ 5HD+5HD FZ+FZ
    if len(shape_x) >= 2 and len(shape_y) >= 2 and shape_x[-2:] == shape_y[-2:]:
        format_support_flag[("FRACTAL_NZ", "FRACTAL_NZ", "FRACTAL_NZ")] = 1
        if x_is_which_format.get("is_4d") and y_is_which_format.get("is_4d"):
            if x_info.get("dim_c") % 16 == 0 and y_info.get("dim_c") % 16 == 0:
                if format_x == format_y == "NCHW":
                    if (shape_x[1] == shape_y[1] or shape_x[1] == 16 or shape_y[1] == 16) \
                            or (shape_x[0] == shape_y[0] or shape_x[0] == 1 or shape_y[0] == 1):
                        format_support_flag[("NC1HWC0", "NC1HWC0", "NC1HWC0")] = 1
                if format_x == format_y in ("HWCN", "NHWC"):
                    if shape_x[0] == shape_y[0] and (shape_x[1] == 1 or shape_y[1] == 1):
                        format_support_flag[("NC1HWC0", "NC1HWC0", "NC1HWC0")] = 1
                    if shape_x[1] == shape_y[1] and (shape_x[0] == 1 or shape_y[0] == 1):
                        format_support_flag[("NC1HWC0", "NC1HWC0", "NC1HWC0")] = 1
                    if shape_x[0] == shape_y[0] and shape_x[1] == shape_y[1]:
                        format_support_flag[("NC1HWC0", "NC1HWC0", "NC1HWC0")] = 1
                    if (shape_x[1] == shape_x[0] == 1) or (shape_y[0] == shape_y[1] == 1):
                        format_support_flag[("NC1HWC0", "NC1HWC0", "NC1HWC0")] = 1
                    if (shape_x[0] == shape_y[1] == 1) or (shape_x[1] == shape_y[0] == 1):
                        format_support_flag[("NC1HWC0", "NC1HWC0", "NC1HWC0")] = 1
            if x_info.get("dim_c") % 16 == 0 and y_info.get("dim_c") % 16 == 0\
                    and y_info.get("dim_n") % 16 == 0 and x_info.get("dim_n") % 16 == 0 and \
                    util_common.is_support_fractal_z_inputs(list_input):
                if format_x == format_y == "HWCN" and shape_x[0] * shape_x[1] == shape_y[0] * shape_y[1]:
                    format_support_flag[("FRACTAL_Z", "FRACTAL_Z", "FRACTAL_Z")] = 1

    # NZ+ND,ND+ND,5HD+5HD,FZ+FZ,ND+NZ
    elif len(shape_x) >= 2 and len(shape_y) >= 2\
            and ((_is_last_two_axis_16_multiple(shape_x) and not _is_last_two_axis_16_multiple(shape_y))
                 or (not _is_last_two_axis_16_multiple(shape_x) and _is_last_two_axis_16_multiple(shape_y))):
        if (_is_last_two_axis_16_multiple(shape_x) and not _is_last_two_axis_16_multiple(shape_y)) or\
                (not _is_last_two_axis_16_multiple(shape_x) and _is_last_two_axis_16_multiple(shape_y)):
            if x_is_which_format.get("is_4d") and y_is_which_format.get("is_4d"):
                if x_info.get("dim_c") % 16 == 0 and y_info.get("dim_c") % 16 == 0:
                    if x_info.get("dim_c") == y_info.get("dim_c") or x_info.get("dim_c") == 16 or \
                       y_info.get("dim_c") == 16:
                        format_support_flag[("NC1HWC0", "NC1HWC0", "NC1HWC0")] = 1
                if x_info.get("dim_c") % 16 == 0 and x_info.get("dim_n") % 16 == 0\
                        and y_info.get("dim_c") % 16 == 0 and y_info.get("dim_n") % 16 == 0\
                        and util_common.is_support_fractal_z_inputs(list_input):
                    if format_x == format_y == "NCHW"\
                            and x_info.get("dim_h") * x_info.get("dim_w") == y_info.get("dim_h") * y_info.get("dim_w")\
                            and x_info.get("dim_c") == y_info.get("dim_c"):
                        if x_info.get("dim_n") == y_info.get("dim_n") or x_info.get("dim_n") == 16 or \
                           y_info.get("dim_n") == 16:
                            format_support_flag[("FRACTAL_Z", "FRACTAL_Z", "FRACTAL_Z")] = 1
                    if format_x == format_y == "NHWC"\
                            and x_info.get("dim_h") * x_info.get("dim_w") == y_info.get("dim_h") * y_info.get("dim_w"):
                        if x_info.get("dim_n") == y_info.get("dim_n") and x_info.get("dim_c") == y_info.get("dim_c"):
                            format_support_flag[("FRACTAL_Z", "FRACTAL_Z", "FRACTAL_Z")] = 1
        if _is_last_two_axis_16_multiple(shape_x) and not _is_last_two_axis_16_multiple(shape_y):
            format_support_flag[("FRACTAL_NZ", "ND", "FRACTAL_NZ")] = 1
        if not _is_last_two_axis_16_multiple(shape_x) and _is_last_two_axis_16_multiple(shape_y):
            format_support_flag[("ND", "FRACTAL_NZ", "ND")] = 1

    # 5HD+scalar,ND+ND,FZ+scalar
    elif len(shape_x) >= 2 and y_is_which_format.get("is_scalar"):
        if x_is_which_format.get("is_4d"):
            if x_info.get("dim_c") % 16 == 0:
                format_support_flag[("NC1HWC0", "ND", "NC1HWC0")] = 1
            if x_info.get("dim_c") % 16 == 0 and x_info.get("dim_n") % 16 == 0\
                    and util_common.is_support_fractal_z_inputs(list_input):
                format_support_flag[("FRACTAL_Z", "ND", "FRACTAL_Z")] = 1

    # ND+ND,scalar+5HD,scalar+FZ
    elif len(shape_y) >= 2 and x_is_which_format.get("is_scalar"):
        if y_is_which_format.get("is_4d"):
            if y_info.get("dim_c") % 16 == 0:
                format_support_flag[("ND", "NC1HWC0", "NC1HWC0")] = 1
            if y_info.get("dim_c") % 16 == 0 and y_info.get("dim_n") % 16 == 0 \
                    and util_common.is_support_fractal_z_inputs(list_input):
                format_support_flag[("ND", "FRACTAL_Z", "FRACTAL_Z")] = 1

    # ND+ND,5HD+5HD
    else:
        # 'pylint: disable=too-many-arguments
        def _is_support_5d_5d_5d(x_dim1, x_dim2, x_dim3, y_dim1, y_dim2, y_dim3):
            if x_dim1 == y_dim1:
                if x_dim2 == y_dim2 and (x_dim3 == 1 or y_dim3 == 1):
                    format_support_flag[("NC1HWC0", "NC1HWC0", "NC1HWC0")] = 1
                if x_dim3 == y_dim3 and (x_dim2 == 1 or y_dim2 == 1):
                    format_support_flag[("NC1HWC0", "NC1HWC0", "NC1HWC0")] = 1
                if x_dim2 == y_dim2 and x_dim3 == y_dim3:
                    format_support_flag[("NC1HWC0", "NC1HWC0", "NC1HWC0")] = 1
                if (x_dim3 == x_dim2 == 1) or (y_dim3 == y_dim2 == 1):
                    format_support_flag[("NC1HWC0", "NC1HWC0", "NC1HWC0")] = 1
                if (x_dim2 == 1 and y_dim3 == 1) or (x_dim3 == 1 and y_dim2 == 1):
                    format_support_flag[("NC1HWC0", "NC1HWC0", "NC1HWC0")] = 1

        if len(shape_x) == len(shape_y) == 1 and shape_x[0] % 16 == 0 and shape_y[0] % 16 == 0:
            format_support_flag[("NC1HWC0", "NC1HWC0", "NC1HWC0")] = 1
        if x_is_which_format.get("is_4d") and y_is_which_format.get("is_4d") and format_x == format_y:
            if x_info.get("dim_c") % 16 == 0 and y_info.get("dim_c") % 16 == 0:
                if (x_info.get("dim_c") == 16 or y_info.get("dim_c") == 16) or \
                   x_info.get("dim_c") == y_info.get("dim_c"):
                    _is_support_5d_5d_5d(x_info.get("dim_n"), x_info.get("dim_h"), x_info.get("dim_w"),
                                         y_info.get("dim_n"), y_info.get("dim_h"), y_info.get("dim_w"))
                    _is_support_5d_5d_5d(x_info.get("dim_h"), x_info.get("dim_n"), x_info.get("dim_w"),
                                         y_info.get("dim_h"), y_info.get("dim_n"), y_info.get("dim_w"))
                    _is_support_5d_5d_5d(x_info.get("dim_w"), x_info.get("dim_n"), x_info.get("dim_h"),
                                         y_info.get("dim_w"), y_info.get("dim_n"), y_info.get("dim_h"))

        # add case nz + nd or nd + nz (ex: [16, 16, 256] + [256])
        if _is_support_nd_nz_nz(shape_x, shape_y) or _is_support_nd_nz_nz(shape_y, shape_x):
            if len(shape_x) >= 2:
                format_support_flag[("FRACTAL_NZ", "ND", "FRACTAL_NZ")] = 1
            else:
                format_support_flag[("ND", "FRACTAL_NZ", "FRACTAL_NZ")] = 1

    # gen format and dtype
    format_list_input0 = []
    format_list_input1 = []
    format_list_output = []

    for _, format_tuple in enumerate(format_support_flag):
        if format_support_flag.get(format_tuple):
            format_list_input0.append(format_tuple[0])
            format_list_input1.append(format_tuple[1])
            format_list_output.append(format_tuple[2])

    dtype_total = []
    for dtype in dtype_list:
        dtype_total = dtype_total + [dtype] * len(format_list_output)
    len_dtype_list = len(dtype_list)
    format_list_input0 = format_list_input0 * len_dtype_list
    format_list_input1 = format_list_input1 * len_dtype_list
    format_list_output = format_list_output * len_dtype_list

    input0 = gen_param(classify="input0", name="x1", datatype=",".join(dtype_total),
                       unknownshape_format=",".join(format_list_input0), format=",".join(format_list_input0))
    input1 = gen_param(classify="input1", name="x2", datatype=",".join(dtype_total),
                       unknownshape_format=",".join(format_list_input1), format=",".join(format_list_input1))
    output0 = gen_param(classify="output0", name="y", datatype=",".join(dtype_total),
                        unknownshape_format=",".join(format_list_output), format=",".join(format_list_output))

    param_list = [input0, input1, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def _add_check_format(x, y):
    """
    check format of add

    Parameters
    ----------
    x: dict
    y: dict

    Returns
    -------
    format_pattern: int
    """
    shape1 = x.get("shape")
    shape2 = y.get("shape")
    list_format = [x.get("format"), y.get("format")]
    shape1 = shape_util.scalar2tensor_one(shape1)
    shape2 = shape_util.scalar2tensor_one(shape2)
    list_shape = [shape1, shape2]

    format_list = ("ND", "NCHW", "NHWC")
    if (list_format[0] == "FRACTAL_NZ" and len(list_shape[1]) == 1 \
            and list_shape[1][0] % SIZE_SIXTEEN == 0) \
            or (list_format[1] == "FRACTAL_NZ" and len(list_shape[0]) == 1 \
            and list_shape[0][0] % SIZE_SIXTEEN == 0):
        format_pattern = 3
    elif list_format[0] == "FRACTAL_NZ" and list_format[1] in format_list \
            and (len(shape2) != 1 or (len(shape2) == 1 and shape2[0] != 1)):
        format_pattern = 1
    elif list_format[0] in format_list and list_format[1] == "FRACTAL_NZ" \
            and (len(shape1) != 1 or (len(shape1) == 1 and shape1[0] != 1)):
        format_pattern = 2
    else:
        format_pattern = 0

    return format_pattern


def _infer_shape(format_pattern, x, y):
    """
    infer shape for x and y

    Parameters
    ----------
    format_pattern: format type
    x: dict
    y: dict

    Returns
    -------
    shape_x: shape of x
    shape_y: shape of y
    """
    shape_x = x.get("shape")
    shape_y = y.get("shape")
    ori_shape_x = x.get("ori_shape")
    ori_shape_y = y.get("ori_shape")
    shape_x = shape_util.scalar2tensor_one(shape_x)
    shape_y = shape_util.scalar2tensor_one(shape_y)

    if format_pattern == 1:
        ori_shape_x, shape_y, _ = shape_util.broadcast_shapes(ori_shape_x, shape_y,
                                                              param_name_input1='x',
                                                              param_name_input2='y')
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
        shape_x, ori_shape_y, _ = shape_util.broadcast_shapes(shape_x, ori_shape_y,
                                                              param_name_input1='x',
                                                              param_name_input2='y')
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
    elif format_pattern == 3:
        def _get_new_shape(_nz_shape, _nd_shape):
            _nd_new_shape = [1 for _ in _nz_shape]
            _nd_new_shape[-1] = _nz_shape[-1]
            _nd_new_shape[-4] = _nz_shape[-4]

            return _nz_shape, _nd_new_shape

        if len(shape_y) == 1:
            shape_x, shape_y = _get_new_shape(shape_x, shape_y)
        else:
            shape_y, shape_x = _get_new_shape(shape_y, shape_x)

    return shape_x, shape_y


@register_operator_compute("axpy", op_mode="static", support_fusion=True)
def axpy_compute(x1, x2, y, alpha, kernel_name="axpy"):
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
    alpha : float
    scalar of mul-factor
    kernel_name : str
    kernel name, default value is "axpy"

    Returns
    -------
    output tensor
    """
    # broadcast
    shape_x = shape_util.shape_to_list(x1.shape)
    shape_y = shape_util.shape_to_list(x2.shape)
    dtype = x1.dtype.lower()

    # neg_1_axis_flag
    neg_1_axis_flag = 0
    if shape_x != shape_y:
        # if shape not equal, then apply broadcast.
        shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x, shape_y,
                                                                  param_name_input1='x1',
                                                                  param_name_input2='x2')

        for i in range(len(shape_x) - 1):
            if shape_x[i] != shape_y[i]:
                neg_1_axis_flag = 1
                break
        para_check.check_shape(shape_max, param_name="x1")
        x1 = tbe.broadcast(x1, shape_max)
        x2 = tbe.broadcast(x2, shape_max)

    # start the main logic
    if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") == "Ascend910":
        if dtype in ("float16", "float32"):
            # fp16 or fp32
            if neg_1_axis_flag:
                res_muls = tbe.vmuls(x2, alpha)
                res = tbe.vadd(x1, res_muls)
            else:
                res = tbe.vaxpy(x2, x1, tvm.const(alpha, dtype=dtype))
        else:
            # int32
            if alpha != 1:
                # add+muls use fp32
                to_type = "float32"
                input_x_cast = tbe.cast_to(x1, to_type)
                input_y_cast = tbe.cast_to(x2, to_type)

                if neg_1_axis_flag:
                    res_muls = tbe.vmuls(x2, alpha)
                    res_tmp = tbe.vadd(x1, res_muls)
                else:
                    res_tmp = tbe.vaxpy(input_y_cast, input_x_cast,
                                                tvm.const(alpha, dtype=to_type))

                res = tbe.cast_to(res_tmp, dtype)

            else:
                # if alpha == 1
                res = tbe.vadd(x2, x1)
    else:
        if dtype in ("float16", "float32"):
            # fp16 or fp32
            res_muls = tbe.vmuls(x2, alpha)
            res = tbe.vadd(x1, res_muls)
        else:
            # int32
            if alpha != 1:
                # add+muls use fp32
                to_type = "float32"
                input_x1_cast = tbe.cast_to(x1, to_type)
                input_x2_cast = tbe.cast_to(x2, to_type)

                res_muls = tbe.vmuls(input_x2_cast, alpha)
                res_tmp = tbe.vadd(input_x1_cast, res_muls)

                res = tbe.cast_to(res_tmp, dtype)
            else:
                # if alpha == 1
                res = tbe.vadd(x2, x1)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def axpy(x1, x2, y, alpha, kernel_name="axpy"):
    """
    calculating data

    Parameters
    ----------
    x1 : dict
    shape and dtype of input_x
    x2 : dict
    shape and dtype of input_y
    y : dict
    shape and dtype of output, should be same shape and type as input
    alpha : float
    scalar apply to input_y:input_y*alpha
    kernel_name : str
    kernel name, default value is "axpy"

    Returns
    -------
    None
    """
    format_pattern = _add_check_format(x1, x2)
    shape_x1, shape_x2 = _infer_shape(format_pattern, x1, x2)

    # check shape
    shape_x1 = shape_util.scalar2tensor_one(shape_x1)
    para_check.check_shape(shape_x1, param_name="shape_x1")

    shape_x2 = shape_util.scalar2tensor_one(shape_x2)
    para_check.check_shape(shape_x2, param_name="shape_x2")

    # check dtype
    dtype_list = ("float16", "float32", "int32")

    dtype_x1 = x1.get("dtype").lower()
    para_check.check_dtype(dtype_x1, dtype_list)
    dtype_x2 = x2.get("dtype").lower()
    para_check.check_dtype(dtype_x2, dtype_list)

    # produce shapes
    shape_x1, shape_x2, shape_max = shape_util.broadcast_shapes(shape_x1, shape_x2,
                                                                param_name_input1='x1',
                                                                param_name_input2='x2')
    if shape_x1[-1] == 1 and shape_x2[-1] == 1 and shape_max[-1] == 1:
        shape_x1 = shape_x1 if len(shape_x1) == 1 else shape_x1[:-1]
        shape_x2 = shape_x2 if len(shape_x2) == 1 else shape_x2[:-1]
        shape_max = shape_max if len(shape_max) == 1 else shape_max[:-1]
    para_check.check_shape(shape_max, param_name="shape_max")

    shape_x1, shape_x2 = shape_util.refine_shapes_for_broadcast(shape_x1, shape_x2)

    data_input_x1 = tvm.placeholder(shape_x1, name="data_input_x1", dtype=dtype_x1)
    data_input_x2 = tvm.placeholder(shape_x2, name="data_input_x2", dtype=dtype_x2)

    res = axpy_compute(data_input_x1, data_input_x2, y, alpha, kernel_name)

    with tvm.target.cce():
        schedule = auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data_input_x1, data_input_x2, res]}

    build(schedule, config)
