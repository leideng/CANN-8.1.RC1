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
mul
"""
import functools

import tbe.dsl as tbe
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util import util_common
from impl.util import util_select_op_base

# Determine whether the 16 bit alignment
SIZE_SIXTEEN = 16


# pylint: disable=too-many-statements,too-many-branches,too-many-nested-blocks,too-many-boolean-expressions
# pylint: disable=too-many-locals
def _can_division_sixteen(shape):
    """
    _can_division_sixteen
    """
    if len(shape) < 2:
        if shape[-1] == 0:
            expected_value = "equal to 0"
            real_value = "not equal to 0"
            error_manager_vector.raise_err_input_value_invalid("mul", "value of shape", expected_value, real_value)
        return False

    if shape[-1] == 0 or shape[-2] == 0:
        expected_value = "equal to 0"
        real_value = "not equal to 0"
        error_manager_vector.raise_err_input_value_invalid("mul", "value of shape", expected_value, real_value)
    if shape[-1] % SIZE_SIXTEEN == 0 and shape[-2] % SIZE_SIXTEEN == 0:
        return True

    return False


# pylint: disable=too-many-locals
def _broadcast_zn_rule(shape0, shape1, format0, format1):
    """
    _broadcast_zn_rule
    """
    if format1 != format0:
        format_rule = "format should be same"
        error_manager_vector.raise_err_check_params_rules("mul", format_rule, "x", format0)

    if len(shape0) != len(shape1) != 4:
        expected_value = "equal to 4"
        real_value = "not equal to 4"
        error_manager_vector.raise_err_input_value_invalid("mul", "length of shapes", expected_value, real_value)

    x_cdim = shape0[format0.index("C")]
    x_wdim = shape0[format0.index("W")]
    x_hdim = shape0[format0.index("H")]
    x_ndim = shape0[format0.index("N")]
    y_cdim = shape1[format1.index("C")]
    y_wdim = shape1[format1.index("W")]
    y_hdim = shape1[format1.index("H")]
    y_ndim = shape1[format1.index("N")]

    x_c0 = 16
    x_n0 = 16
    x_c1 = x_cdim // 16
    x_n1 = x_ndim // 16
    shape0_zn = [x_hdim * x_wdim * x_c1, x_n1, x_n0, x_c0]

    y_c0 = 16
    y_n0 = 16
    y_c1 = y_cdim // 16
    y_n1 = y_ndim // 16
    shape1_zn = [y_hdim * y_wdim * y_c1, y_n1, y_n0, y_c0]

    if len(shape0_zn) < len(shape1_zn):
        shape0_zn, shape1_zn = shape1_zn, shape0_zn

    output_shape_len = len(shape0_zn)
    dec = output_shape_len - len(shape1_zn)
    for _, i in enumerate(range(dec)):
        shape1_zn = [1] + shape1_zn

    for _, i in enumerate(range(output_shape_len)):
        if (shape0_zn[i] != shape1_zn[i]) and (shape0_zn[i] != 1) and (shape1_zn[i] != 1):
            return False

    return True


# pylint: disable=unused-argument,too-many-locals,invalid-name,too-many-branches,too-many-statements
# pylint: disable=too-many-boolean-expressions,too-many-nested-blocks
def op_sub_select_format(x, y, output, kernel_name="mul"):
    """
    Dynamic matching format

    Parameters
    ----------
    x : dict
        shape and dtype of input_x
    y : dict
        shape and dtype of input_y
    output : dict
        shape and dtype of output, should be same shape and type as input

    kernel_name : str
        kernel name, default value is "pt_muls"

    Returns
    -------
    None
    """
    shape_x1 = x.get("ori_shape")
    shape_x2 = y.get("ori_shape")

    shape_x1 = shape_util.scalar2tensor_one(shape_x1)
    shape_x2 = shape_util.scalar2tensor_one(shape_x2)

    enum_x2 = functools.reduce(lambda x, y: x * y, shape_x2)

    dtype_list = ["float16", "float", "int32", "int16", "uint8", "int8"]
    vmul_support_s16 = tbe_platform.api_check_support("tbe.dsl.vmul", "int16")
    vmul_support_fp32 = tbe_platform.api_check_support("tbe.dsl.vmul", "float32")
    if not vmul_support_s16:
        dtype_list.remove("int16")
    if not vmul_support_fp32:
        dtype_list.remove("float")
        # If the platform does not support float32 data type,
        # neither of uint8 and int8 is supported at the same time
        dtype_list.remove("uint8")
        dtype_list.remove("int8")

    if len(shape_x2) == 1 and enum_x2 == 1:
        format_list = ("ND", "NCHW", "NHWC", "FRACTAL_NZ", "NC1HWC0", "FRACTAL_Z", "C1HWNCoC0")
        dtype_list_total = functools.reduce(lambda x, y: x + y, [[ele] * len(format_list) for ele in dtype_list])
        format_list_for_non_one = format_list * len(dtype_list)
        format_list_for_one = [y.get("format")] * len(format_list) * len(dtype_list)
        unknownshape_format_list = ["ND"] * len(format_list) * len(dtype_list)
    else:
        return None

    if -1 in shape_x1 or -1 in shape_x2:
        input0 = util_select_op_base.gen_param(classify="input0", name="x1",
                                               datatype=",".join(dtype_list_total),
                                               format=",".join(format_list_for_non_one),
                                               unknownshape_format=",".join(unknownshape_format_list))
        input1 = util_select_op_base.gen_param(classify="input1", name="x2",
                                               datatype=",".join(dtype_list_total),
                                               format=",".join(format_list_for_one),
                                               unknownshape_format=",".join(unknownshape_format_list))
        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                datatype=",".join(dtype_list_total),
                                                format=",".join(format_list_for_non_one),
                                                unknownshape_format=",".join(unknownshape_format_list))
    else:
        input0 = util_select_op_base.gen_param(classify="input0", name="x1",
                                               datatype=",".join(dtype_list_total),
                                               format=",".join(format_list_for_non_one))
        input1 = util_select_op_base.gen_param(classify="input1", name="x2",
                                               datatype=",".join(dtype_list_total),
                                               format=",".join(format_list_for_one))
        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                datatype=",".join(dtype_list_total),
                                                format=",".join(format_list_for_non_one))

    param_list = [input0, input1, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


# pylint: disable=unused-argument,too-many-locals,invalid-name,too-many-branches,too-many-statements
# pylint: disable=too-many-boolean-expressions,too-many-nested-blocks
def op_select_format(x, y, output, kernel_name="mul"):
    """
    select format dynamically\n

    1.when the lengths of x's shape and y's shape are the same and equal to 5, the formats of x and
    y are the same and are one of [NDHWC,DHWCN,NCDHW], and x's shape == y's shape: support ND, NDC1HWC0,
    FRACTAL_Z_3D format.\n

        example:\n
        original:\n
        x's Tensor(shape=(2, 3, 4, 5, 6), "NDHWC")\n
        y's Tensor(shape=(2, 3, 4, 5, 6), "NDHWC")\n
        support conversion to NDC1HWC0 operation:\n
        x's Tensor(shape=(2, 3, 1, 4, 5, 16), "NDC1HWC0")\n
        y's Tensor(shape=(2, 3, 1, 4, 5, 16), "NDC1HWC0")\n

    2.when the lengths of x's shape and y's shape are the same and equal to 5, the formats of x and
    y are the same and are one of [NDHWC,DHWCN,NCDHW], x's shape != y's shape, and x's dim of c == y's
    dim of c: support ND, NDC1HWC0 format.\n

        example:\n
        original:\n
        x's Tensor(shape=(2, 3, 4, 5, 6), "NDHWC")\n
        y's Tensor(shape=(1, 2, 3, 4, 6), "NDHWC")\n
        support conversion to NDC1HWC0 operation:\n
        x's Tensor(shape=(2, 3, 1, 4, 5, 16), "NDC1HWC0")\n
        y's Tensor(shape=(1, 2, 1, 3, 4, 16), "NDC1HWC0")\n

    3.when the lengths of x's shape and y's shape are the same and equal to 5,the formats of x and
    y are the same and are one of [NDHWC,DHWCN,NCDHW], x's shape != y's shape, x's dim of c == y's dim
    of c, and x's dim of n == y's dim of n: support ND, NDC1HWC0, FRACTAL_Z_3D format.\n

        example:\n
        original:\n
        x's Tensor(shape=(2, 3, 4, 5, 6), "NDHWC")\n
        y's Tensor(shape=(2, 2, 3, 4, 6), "NDHWC")\n
        support conversion to NDC1HWC0 operation:\n
        x's Tensor(shape=(2, 3, 1, 4, 5, 16), "NDC1HWC0")\n
        y's Tensor(shape=(2, 2, 1, 3, 4, 16), "NDC1HWC0")\n

    4.when the lengths of x's shape >= 2, the lengths of y's shape >= 2, and x's shape[-2:] == y's shape[-2:]:
    support ND, FRACTAL_NZ format.\n

        example:\n
        original:\n
        x's Tensor(shape=(2, 3, 4), "ND")\n
        y's Tensor(shape=(1, 3, 4), "ND")\n
        support conversion to FRACTAL_NZ operation:\n
        x's Tensor(shape=(2, 1, 1, 16, 16), "FRACTAL_NZ")\n
        y's Tensor(shape=(2, 1, 1, 16, 16), "FRACTAL_NZ")\n

    5.when the lengths of x's shape and y's shape are the same and equal to 4, x's dim of c % 16 == 0
    and y's dim of c % 16 == 0 or x's dim of c == y's dim of c, x's format == y's format and are NCHW,
    x's dim of c == y's dim of c or x's dim of c / 16 == 1 or y's dim of c / 16 == 1, and x's dim of
    n == y's dim of n or x's dim of n == 1 or y's dim of n == 1: support ND, NC1HWC0 format.\n

        example:\n
        original:\n
        x's Tensor(shape=(2, 16, 4, 5), "NCHW")\n
        y's Tensor(shape=(2, 16, 4, 16), "NCHW")\n
        support conversion to NC1HWC0 operation:\n
        x's Tensor(shape=(2, 1, 4, 5, 16), "NC1HWC0")\n
        y's Tensor(shape=(1, 1, 4, 16, 16), "NC1HWC0")\n

    6.when the lengths of x's shape and y's shape are the same and equal to 4, x's dim of c % 16 == 0
    and y's dim of c % 16 == 0 or x's dim of c == y's dim of c, x's format == y's format and are HWCN,
    x's dim of h == y's dim of h, and x's dim of w == 1 or y's dim of w == 1: support ND, NC1HWC0 format.\n

        example:\n
        original:\n
        x's Tensor(shape=(2, 4, 16, 5), "HWCN")\n
        y's Tensor(shape=(2, 1, 16, 4), "HWCN")\n
        support conversion to NC1HWC0 operation:\n
        x's Tensor(shape=(5, 1, 2, 4, 16), "NC1HWC0")\n
        y's Tensor(shape=(4, 1, 2, 1, 16), "NC1HWC0")\n

    7.when the lengths of x's shape and y's shape are the same and equal to 4, x's dim of c % 16 == 0
    and y's dim of c % 16 == 0 or x's dim of c == y's dim of c, x's format == y's format and are HWCN,
    x's dim of w == y's dim of w, and x's dim of h == 1 or y's dim of h == 1: support ND, NC1HWC0 format.\n

        example:\n
        original:\n
        x's Tensor(shape=(1, 4, 16, 5), "HWCN")\n
        y's Tensor(shape=(2, 4, 16, 4), "HWCN")\n
        support conversion to NC1HWC0 operation:\n
        x's Tensor(shape=(5, 1, 1, 4, 16), "NC1HWC0")\n
        y's Tensor(shape=(4, 1, 2, 4, 16), "NC1HWC0")\n

    8.when the lengths of x's shape and y's shape are the same and equal to 4, x's dim of c % 16 == 0
    and y's dim of c % 16 == 0 or x's dim of c == y's dim of c, x's format == y's format and are HWCN,
    x's dim of w == y's dim of w, and x's dim of h == y's dim of h: support ND, NC1HWC0 format.\n

        example:\n
        original:\n
        x's Tensor(shape=(2, 4, 16, 5), "HWCN")\n
        y's Tensor(shape=(2, 4, 16, 4), "HWCN")\n
        support conversion to NC1HWC0 operation:\n
        x's Tensor(shape=(5, 1, 2, 4, 16), "NC1HWC0")\n
        y's Tensor(shape=(4, 1, 2, 4, 16), "NC1HWC0")\n

    9.when the lengths of x's shape and y's shape are the same and equal to 4, x's dim of c % 16 == 0
    and y's dim of c % 16 == 0 or x's dim of c == y's dim of c, x's format == y's format and are HWCN,
    and x's dim of w == x's dim of h == 1 or y's dim of h == y's dim of w == 1: support ND, NC1HWC0 format.\n

        example:\n
        original:\n
        x's Tensor(shape=(1, 1, 16, 5), "HWCN")\n
        y's Tensor(shape=(2, 4, 16, 4), "HWCN")\n
        support conversion to NC1HWC0 operation:\n
        x's Tensor(shape=(5, 1, 1, 1, 16), "NC1HWC0")\n
        y's Tensor(shape=(4, 1, 2, 4, 16), "NC1HWC0")\n

    10.when the lengths of x's shape and y's shape are the same and equal to 4, x's dim of c % 16 == 0
    and y's dim of c % 16 == 0 or x's dim of c == y's dim of c, x's format == y's format and are HWCN,
    and x's dim of h == y's dim of w == 1 or x's dim of w == y's dim of h == 1: support ND, NC1HWC0 format.\n

        example:\n
        original:\n
        x's Tensor(shape=(1, 2, 16, 5), "HWCN")\n
        y's Tensor(shape=(2, 1, 16, 4), "HWCN")\n
        support conversion to NC1HWC0 operation:\n
        x's Tensor(shape=(5, 1, 1, 2, 16), "NC1HWC0")\n
        y's Tensor(shape=(4, 1, 2, 1, 16), "NC1HWC0")\n

    11.when the lengths of x's shape and y's shape are the same and equal to 4, x's dim of c % 16 == 0
    and y's dim of c % 16 == 0 or x's dim of c == y's dim of c, x's format == y's format and are NHWC,
    x's dim of h == y's dim of h, and x's dim of n == 1 or y's dim of n == 1: support ND, NC1HWC0 format.\n

        example:\n
        original:\n
        x's Tensor(shape=(1, 2, 3, 16), "NHWC")\n
        y's Tensor(shape=(2, 2, 4, 16), "NHWC")\n
        support conversion to NC1HWC0 operation:\n
        x's Tensor(shape=(1, 1, 2, 3, 16), "NC1HWC0")\n
        y's Tensor(shape=(2, 1, 2, 4, 16), "NC1HWC0")\n

    12.when the lengths of x's shape and y's shape are the same and equal to 4, x's dim of c % 16 == 0
    and y's dim of c % 16 == 0 or x's dim of c == y's dim of c, x's format == y's format and are NHWC,
    x's dim of n == y's dim of n, and x's dim of h == 1 or y's dim of h == 1: support ND, NC1HWC0 format.\n

        example:\n
        original:\n
        x's Tensor(shape=(2, 1, 3, 16), "NHWC")\n
        y's Tensor(shape=(2, 2, 4, 16), "NHWC")\n
        support conversion to NC1HWC0 operation:\n
        x's Tensor(shape=(2, 1, 1, 3, 16), "NC1HWC0")\n
        y's Tensor(shape=(2, 1, 2, 4, 16), "NC1HWC0")\n

    13.when the lengths of x's shape and y's shape are the same and equal to 4, x's dim of c % 16 == 0
    and y's dim of c % 16 == 0 or x's dim of c == y's dim of c, x's format == y's format and are NHWC,
    x's dim of n == y's dim of n, and x's dim of h == y's dim of h: support ND, NC1HWC0 format.\n

        example:\n
        original:\n
        x's Tensor(shape=(2, 3, 3, 16), "NHWC")\n
        y's Tensor(shape=(2, 3, 4, 16), "NHWC")\n
        support conversion to NC1HWC0 operation:\n
        x's Tensor(shape=(2, 1, 3, 3, 16), "NC1HWC0")\n
        y's Tensor(shape=(2, 1, 3, 4, 16), "NC1HWC0")\n

    14.when the lengths of x's shape and y's shape are the same and equal to 4, x's dim of c % 16 == 0
    and y's dim of c % 16 == 0 or x's dim of c == y's dim of c, x's format == y's format and are NHWC,
    and x's dim of n == x's dim of h == 1 or y's dim of n == y's dim of h == 1: support ND, NC1HWC0 format.\n

        example:\n
        original:\n
        x's Tensor(shape=(1, 1, 3, 16), "NHWC")\n
        y's Tensor(shape=(2, 3, 4, 16), "NHWC")\n
        support conversion to NC1HWC0 operation:\n
        x's Tensor(shape=(1, 1, 1, 3, 16), "NC1HWC0")\n
        y's Tensor(shape=(2, 1, 3, 4, 16), "NC1HWC0")\n

    15.when the lengths of x's shape and y's shape are the same and equal to 4, x's dim of c % 16 == 0
    and y's dim of c % 16 == 0 or x's dim of c == y's dim of c, x's format == y's format and are NHWC,
    and x's dim of n == y's dim of h == 1 or x's dim of h == y's dim of n == 1: support ND, NC1HWC0 format.\n

        example:\n
        original:\n
        x's Tensor(shape=(1, 3, 3, 16), "NHWC")\n
        y's Tensor(shape=(2, 1, 4, 16), "NHWC")\n
        support conversion to NC1HWC0 operation:\n
        x's Tensor(shape=(1, 1, 3, 3, 16), "NC1HWC0")\n
        y's Tensor(shape=(2, 1, 1, 4, 16), "NC1HWC0")\n

    16.when the lengths of x's shape and y's shape are the same and equal to 4, the formats of x and y are
    the same and are one of [NDHWC,DHWCN,NCDHW], x's dim of c % 16 == 0 and y's dim of c % 16 == 0, x's
    dim of n % 16 == 0 and y's dim of n % 16 == 0, and when x, y are converted to FRACTAL_Z format
    (each dim of x, y dim_i(x[dim_i] == y[dim_i] or x[dim_i] == 1 or y[dim_i] == 1)): support ND,
    FRACTAL_Z format.\n

        example:\n
        original:\n
        x's Tensor(shape=(16, 16, 3, 4), "NCHW")\n
        y's Tensor(shape=(32, 16, 2, 6), "NCHW")\n
        support conversion to FRACTAL_Z operation:\n
        x's Tensor(shape=(12, 1, 16, 16), "FRACTAL_Z")\n
        y's Tensor(shape=(12, 2, 16, 16), "FRACTAL_Z")\n

    17.when the lengths of x's shape and y's shape are the same and equal to 4, the formats of x and y
    are one of [NCHW,NHWC,HWCN], x's shape == y's shape, and any axis value in x != -1: support
    ND, NC1HWC0, FRACTAL_NZ format.\n

        example:\n
        original:\n
        x's Tensor(shape=(1, 2, 3, 4), "NCHW")\n
        y's Tensor(shape=(1, 2, 3, 4), "NHWC")\n
        support conversion to NC1HWC0 operation:\n
        x's Tensor(shape=(1, 1, 3, 4, 16), "NC1HWC0")\n
        y's Tensor(shape=(1, 1, 2, 3, 16), "NC1HWC0")\n

    18.when the lengths of y's shape == 1, first dim of y == 1, the lengths of x's shape == 4, and the
    format of x is one of [NCHW,NHWC,HWCN]: support ND, C1HWNCoC0, NC1HWC0 format.\n

        example:\n
        original:\n
        x's Tensor(shape=(1, 2, 3, 4), "NCHW")\n
        y's Tensor(shape=(1), "ND")\n
        support conversion to NC1HWC0 operation:\n
        x's Tensor(shape=(1, 1, 3, 4, 16), "NC1HWC0")\n
        y's Tensor(shape=(1, 1, 1, 1, 16), "NC1HWC0")\n

    19.when the lengths of y's shape == 1, first dim of y == 1, the lengths of x's shape == 4, the format of x
    is one of [NCHW,NHWC,HWCN], x's dim of c % 16 == 0, and x's dim of n % 16 == 0:
    support ND, C1HWNCoC0, NC1HWC0, FRACTAL_Z format.\n

        example:\n
        original:\n
        x's Tensor(shape=(16, 16, 3, 4), "NCHW")\n
        y's Tensor(shape=(1), "ND")\n
        support conversion to NC1HWC0 operation:\n
        x's Tensor(shape=(16, 1, 3, 4, 16), "NC1HWC0")\n
        y's Tensor(shape=(1, 1, 1, 1, 16), "NC1HWC0")\n

    20.when the lengths of x's shape == 1, first dim of x == 1, the lengths of y's shape == 4, the format of y
    is one of [NCHW,NHWC,HWCN]: support ND, C1HWNCoC0, NC1HWC0 format.\n

        example:\n
        original:\n
        x's Tensor(shape=(1), "ND")\n
        y's Tensor(shape=(1, 2, 3, 4), "NCHW")\n
        support conversion to NC1HWC0 operation:\n
        x's Tensor(shape=(1, 1, 1, 1, 16), "NC1HWC0")\n
        y's Tensor(shape=(1, 1, 3, 4, 16), "NC1HWC0")\n

    21.when the lengths of x's shape == 1, first dim of x == 1, the lengths of y's shape == 4, the format of y
    is one of [NCHW,NHWC,HWCN], y's dim of c % 16 == 0, and y's dim of n % 16 == 0:
    support ND, C1HWNCoC0, NC1HWC0, FRACTAL_Z format.\n

        example:\n
        original:\n
        x's Tensor(shape=(1), "ND")\n
        y's Tensor(shape=(16, 16, 3, 4), "NCHW")\n
        support conversion to NC1HWC0 operation:\n
        x's Tensor(shape=(1, 1, 1, 1, 16), "NC1HWC0")\n
        y's Tensor(shape=(16, 1, 3, 4, 16), "NC1HWC0")\n

    22.when the lengths of x's shape and y's shape are the same and equal to 1, first dim of x % 16 == 0,
    and first dim of y % 16 == 0: support ND, NC1HWC0 format.\n

        example:\n
        original:\n
        x's Tensor(shape=(16), "ND")\n
        y's Tensor(shape=(16), "ND")\n
        support conversion to NC1HWC0 operation:\n
        x's Tensor(shape=(16, 1, 1, 1, 16), "NC1HWC0")\n
        y's Tensor(shape=(16, 1, 1, 1, 16), "NC1HWC0")\n

    23.when first dim of x != 1, the lengths of x's shape == 1, the lengths of y's shape == 4, x's format == y's
    format, and the format of y is one of ("NHWC",): support ND, NC1HWC0 format.\n

        example:\n
        original:\n
        x's Tensor(shape=(2), "NHWC")\n
        y's Tensor(shape=(2, 3, 4, 5), "NHWC")\n
        support conversion to NC1HWC0 operation:\n
        x's Tensor(shape=(2, 1, 1, 1, 16), "NC1HWC0")\n
        y's Tensor(shape=(2, 1, 3, 3, 16), "NC1HWC0")\n

    24.when first dim of x != 1, the lengths of x's shape == 1, the lengths of y's shape == 4, x's format == y's
    format, the format of y is one of ("NCHW", "HWCN"), and y's dim of c == first dim of x or y's dim of c == 1 or
    first dim of x / 16 == 1: support ND, C1HWC0 format.\n

        example:\n
        original:\n
        x's Tensor(shape=(2), "NCHW")\n
        y's Tensor(shape=(2, 2, 4, 5), "NCHW")\n
        support conversion to NC1HWC0 operation:\n
        x's Tensor(shape=(2, 1, 1, 1, 16), "NC1HWC0")\n
        y's Tensor(shape=(2, 1, 4, 5, 16), "NC1HWC0")\n

    25.when first dim of y != 1, the lengths of x's shape == 4, the lengths of y's shape == 1, x's format == y's
    format, and the format of x is one of ("NHWC",): support ND, NC1HWC0 format.\n

        example:\n
        original:\n
        x's Tensor(shape=(2, 3, 4, 5), "NHWC")\n
        y's Tensor(shape=(2), "NHWC")\n
        support conversion to NC1HWC0 operation:\n
        x's Tensor(shape=(2, 1, 3, 4, 16), "NC1HWC0")\n
        y's Tensor(shape=(2, 1, 1, 1, 16), "NC1HWC0")\n

    26.when first dim of y != 1, the lengths of x's shape == 4, the lengths of y's shape == 1, x's format == y's
    format, the format of x is one of ("NCHW", "HWCN"), and x's dim of c == first dim of y or x's dim of c == 1 or
    y's dim of c / 16 == 1 or first dim of y / 16 == 1: support ND, NC1HWC0 format.\n

        example:\n
        original:\n
        x's Tensor(shape=(2, 2, 4, 5), "NCHW")\n
        y's Tensor(shape=(2), "NCHW")\n
        support conversion to NC1HWC0 operation:\n
        x's Tensor(shape=(2, 1, 4, 5, 16), "NC1HWC0")\n
        y's Tensor(shape=(2, 1, 1, 1, 16), "NC1HWC0")\n
    """
    cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if cce_product in ("Ascend910"):
        param_dynamic_in_json = op_sub_select_format(x, y, output, kernel_name)
        if param_dynamic_in_json is not None:
            return param_dynamic_in_json

    shape_x = x.get("ori_shape")
    shape_y = y.get("ori_shape")

    shape_x = shape_util.scalar2tensor_one(shape_x)
    shape_y = shape_util.scalar2tensor_one(shape_y)

    size_x = functools.reduce(lambda x, y: x * y, shape_x)
    size_y = functools.reduce(lambda x, y: x * y, shape_y)
    temp_format = None

    format_4d_list = ["NCHW", "NHWC", "HWCN"]
    format_5d_list = ["NDHWC", "DHWCN", "NCDHW"]
    dtype_list = ["float16", "float", "int32", "int16", "uint8", "int8"]
    vmul_support_s16 = tbe_platform.api_check_support("tbe.dsl.vmul", "int16")
    vmul_support_fp32 = tbe_platform.api_check_support("tbe.dsl.vmul", "float32")
    if not vmul_support_s16:
        dtype_list.remove("int16")
    if not vmul_support_fp32:
        dtype_list.remove("float")
        # If the platform does not support float32 data type,
        # neither of uint8 and int8 is supported at the same time
        dtype_list.remove("uint8")
        dtype_list.remove("int8")

    format_x = x.get("ori_format")
    format_y = y.get("ori_format")

    dtype_total = []
    format_nd = ["ND"]
    format_list = ["ND"]
    format_nz = ["FRACTAL_NZ"]
    len_format_list = len(dtype_list)
    list_input = [x, y]
    if (len(shape_x) == 4 and len(shape_y) == 4 and format_x in format_4d_list and format_y in format_4d_list) or \
            (len(shape_x) == 5 and len(shape_y) == 5 and format_x == format_y and format_x in format_5d_list):
        x_cdim = shape_x[format_x.index("C")]
        x_wdim = shape_x[format_x.index("W")]
        x_hdim = shape_x[format_x.index("H")]
        x_ndim = shape_x[format_x.index("N")]
        y_cdim = shape_y[format_y.index("C")]
        y_wdim = shape_y[format_y.index("W")]
        y_hdim = shape_y[format_y.index("H")]
        y_ndim = shape_y[format_y.index("N")]

    if (len(shape_y) == 1 and len(shape_x) == 4) and format_x in format_4d_list:
        x_cdim = shape_x[format_x.index("C")]
        x_ndim = shape_x[format_x.index("N")]
    if (len(shape_x) == 1 and len(shape_y) == 4) and format_y in format_4d_list:
        y_cdim = shape_y[format_y.index("C")]
        y_ndim = shape_y[format_y.index("N")]

    # NDC1HWC0 FRACTAL_Z_3D
    if len(shape_x) == 5 and len(shape_y) == 5 and format_x == format_y and format_x in format_5d_list:
        if list(shape_x) == list(shape_y):
            format_list.append("NDC1HWC0")
        elif x_cdim == y_cdim:
            format_list.append("NDC1HWC0")
    if len(shape_x) == 5 and len(shape_y) == 5 and format_x == format_y and format_x in format_5d_list:
        if list(shape_x) == list(shape_y):
            format_list.append("FRACTAL_Z_3D")
        elif x_cdim == y_cdim and x_ndim == y_ndim:
            format_list.append("FRACTAL_Z_3D")

    if (len(shape_x) == 4 and size_y == 1) or (len(shape_y) == 4 and size_x == 1):
        temp_format = format_x if len(shape_x) == 4 and size_y == 1 else format_y
    if ((len(shape_x) == 4 and size_y == 1) or (len(shape_y) == 4 and size_x == 1)) and temp_format in format_4d_list:
        temp_shape = shape_x if len(shape_x) == 4 and size_y == 1 else shape_y
        dim_c = temp_shape[temp_format.index("C")]
        dim_n = temp_shape[temp_format.index("N")]
        second_last_dim = temp_shape[-2]
        last_dim = temp_shape[-1]
        format_list.append("NC1HWC0")
        if last_dim % 16 == 0 and second_last_dim % 16 == 0:
            format_list.append("FRACTAL_NZ")
        if dim_c % 16 == 0 and dim_n % 16 == 0:
            format_list.append("FRACTAL_Z")
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype] * len(format_list)
        format_list = format_list * len_format_list
        if len(shape_x) == 4 and size_y == 1:
            x1_format_list = format_list
            x2_format_list = ["ND"] * len(format_list)
        else:
            x1_format_list = ["ND"] * len(format_list)
            x2_format_list = format_list
        y_format_list = format_list

        input0 = util_select_op_base.gen_param(classify="input0",
                                               name="x1",
                                               datatype=",".join(dtype_total),
                                               format=",".join(x1_format_list))
        input1 = util_select_op_base.gen_param(classify="input1",
                                               name="x2",
                                               datatype=",".join(dtype_total),
                                               format=",".join(x2_format_list))
        output0 = util_select_op_base.gen_param(classify="output0",
                                                name="y",
                                                datatype=",".join(dtype_total),
                                                format=",".join(y_format_list))

    # ND+ND NZ+NZ 5HD+5HD FZ+FZ
    elif len(shape_x) >= 2 and len(shape_y) >= 2:
        if shape_x[-2:] == shape_y[-2:]:
            format_list.append("FRACTAL_NZ")
        if len(shape_x) == 4 and len(shape_y) == 4 and format_x in format_4d_list and format_y in format_4d_list:
            if (x_cdim % 16 == 0 and y_cdim % 16 == 0) or x_cdim == y_cdim:
                if format_x == format_y == "NCHW" and (x_cdim == y_cdim or x_cdim // 16 == 1 or y_cdim // 16
                                                       == 1) and (x_ndim == y_ndim or x_ndim == 1 or y_ndim == 1):
                    format_list.append("NC1HWC0")
                if format_x == format_y == "HWCN":
                    if x_hdim == y_hdim and (x_wdim == 1 or y_wdim == 1):
                        format_list.append("NC1HWC0")
                    if x_wdim == y_wdim and (x_hdim == 1 or y_hdim == 1):
                        format_list.append("NC1HWC0")
                    if x_wdim == y_wdim and x_hdim == y_hdim:
                        format_list.append("NC1HWC0")
                    if (x_wdim == x_hdim == 1) or (y_hdim == y_wdim == 1):
                        format_list.append("NC1HWC0")
                    if (x_hdim == y_wdim == 1) or (x_wdim == y_hdim == 1):
                        format_list.append("NC1HWC0")
                if format_x == format_y == "NHWC":
                    if x_hdim == y_hdim and (x_ndim == 1 or y_ndim == 1):
                        format_list.append("NC1HWC0")
                    if x_ndim == y_ndim and (x_hdim == 1 or y_hdim == 1):
                        format_list.append("NC1HWC0")
                    if x_ndim == y_ndim and x_hdim == y_hdim:
                        format_list.append("NC1HWC0")
                    if (x_ndim == x_hdim == 1) or (y_ndim == y_hdim == 1):
                        format_list.append("NC1HWC0")
                    if (x_ndim == 1 and y_hdim == 1) or (x_hdim == 1 and y_ndim == 1):
                        format_list.append("NC1HWC0")
            if x_cdim % 16 == 0 and y_cdim % 16 == 0 and y_ndim % 16 == 0 and x_ndim % 16 == 0 and \
                    util_common.is_support_fractal_z_inputs(list_input):
                if format_x == format_y and _broadcast_zn_rule(shape_x, shape_y, format_x, format_y):
                    format_list.append("FRACTAL_Z")
            if x_cdim == 1 and y_cdim == 1 and format_x == format_y and format_x.upper() in ("NCHW", "HWCN"):
                format_list.append("FRACTAL_Z")
            if list(shape_x) == list(shape_y) and -1 not in shape_x:
                format_list.append("NC1HWC0")
            if format_x == format_y and list(shape_x) == list(shape_y) and \
                    util_common.is_same_group(list_input):
                format_list.append("FRACTAL_Z")

        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype] * len(format_list)
        format_list = format_list * len_format_list
        unknownshape_format_list = ["ND"] * len(format_list)
        if -1 in shape_x or -1 in shape_y:
            input0 = util_select_op_base.gen_param(classify="input0",
                                                   name="x1",
                                                   datatype=",".join(dtype_total),
                                                   format=",".join(format_list),
                                                   unknownshape_format=",".join(unknownshape_format_list))
            input1 = util_select_op_base.gen_param(classify="input1",
                                                   name="x2",
                                                   datatype=",".join(dtype_total),
                                                   format=",".join(format_list),
                                                   unknownshape_format=",".join(unknownshape_format_list))
            output0 = util_select_op_base.gen_param(classify="output0",
                                                    name="y",
                                                    datatype=",".join(dtype_total),
                                                    format=",".join(format_list),
                                                    unknownshape_format=",".join(unknownshape_format_list))
        else:
            input0 = util_select_op_base.gen_param(classify="input0",
                                                   name="x1",
                                                   datatype=",".join(dtype_total),
                                                   format=",".join(format_list))
            input1 = util_select_op_base.gen_param(classify="input1",
                                                   name="x2",
                                                   datatype=",".join(dtype_total),
                                                   format=",".join(format_list))
            output0 = util_select_op_base.gen_param(classify="output0",
                                                    name="y",
                                                    datatype=",".join(dtype_total),
                                                    format=",".join(format_list))

    # NZ+ND,ND+ND,5HD+5HD,FZ+FZ,ND+NZ
    elif len(shape_x) >= 2 and len(shape_y) >= 2 and (
            (_can_division_sixteen(shape_x) and not _can_division_sixteen(shape_y)) or
            (not _can_division_sixteen(shape_x) and _can_division_sixteen(shape_y))):
        if len(shape_x) == 4 and len(shape_y) == 4 and format_x in format_4d_list and format_y in format_4d_list:
            if x_cdim % 16 == 0 and y_cdim % 16 == 0:
                if x_cdim == y_cdim or x_cdim // 16 == 1 or y_cdim // 16 == 1:
                    format_list.append("NC1HWC0")
            if x_cdim % 16 == 0 and x_ndim % 16 == 0 and y_cdim % 16 == 0 and y_ndim % 16 == 0 and \
                    util_common.is_support_fractal_z_inputs(list_input):
                if format_x == format_y and _broadcast_zn_rule(shape_x, shape_y, format_x, format_y):
                    format_list.append("FRACTAL_Z")

        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype] * len(format_list)
        format_list = format_list * len_format_list
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype] * 1
        format_list0 = format_list + format_nz * len_format_list
        format_list1 = format_list + format_nd * len_format_list
        unknownshape_format_list = ["ND"] * len(dtype_total)
        if _can_division_sixteen(shape_x) and not _can_division_sixteen(shape_y):
            if -1 in shape_x or -1 in shape_y:
                input0 = util_select_op_base.gen_param(classify="input0",
                                                       name="x1",
                                                       datatype=",".join(dtype_total),
                                                       format=",".join(format_list0),
                                                       unknownshape_format=",".join(unknownshape_format_list))
                input1 = util_select_op_base.gen_param(classify="input1",
                                                       name="x2",
                                                       datatype=",".join(dtype_total),
                                                       format=",".join(format_list1),
                                                       unknownshape_format=",".join(unknownshape_format_list))
                output0 = util_select_op_base.gen_param(classify="output0",
                                                        name="y",
                                                        datatype=",".join(dtype_total),
                                                        format=",".join(format_list0),
                                                        unknownshape_format=",".join(unknownshape_format_list))
            else:
                input0 = util_select_op_base.gen_param(classify="input0",
                                                       name="x1",
                                                       datatype=",".join(dtype_total),
                                                       format=",".join(format_list0))
                input1 = util_select_op_base.gen_param(classify="input1",
                                                       name="x2",
                                                       datatype=",".join(dtype_total),
                                                       format=",".join(format_list1))
                output0 = util_select_op_base.gen_param(classify="output0",
                                                        name="y",
                                                        datatype=",".join(dtype_total),
                                                        format=",".join(format_list0))
        else:
            if -1 in shape_x or -1 in shape_y:
                input0 = util_select_op_base.gen_param(classify="input0",
                                                       name="x1",
                                                       datatype=",".join(dtype_total),
                                                       format=",".join(format_list1),
                                                       unknownshape_format=",".join(unknownshape_format_list))
                input1 = util_select_op_base.gen_param(classify="input1",
                                                       name="x2",
                                                       datatype=",".join(dtype_total),
                                                       format=",".join(format_list0),
                                                       unknownshape_format=",".join(unknownshape_format_list))
                output0 = util_select_op_base.gen_param(classify="output0",
                                                        name="y",
                                                        datatype=",".join(dtype_total),
                                                        format=",".join(format_list1),
                                                        unknownshape_format=",".join(unknownshape_format_list))
            else:
                input0 = util_select_op_base.gen_param(classify="input0",
                                                       name="x1",
                                                       datatype=",".join(dtype_total),
                                                       format=",".join(format_list1))
                input1 = util_select_op_base.gen_param(classify="input1",
                                                       name="x2",
                                                       datatype=",".join(dtype_total),
                                                       format=",".join(format_list0))
                output0 = util_select_op_base.gen_param(classify="output0",
                                                        name="y",
                                                        datatype=",".join(dtype_total),
                                                        format=",".join(format_list1))

    # 5HD+scalar,ND+ND,FZ+scalar,6D+scalar,NZ+ND
    elif (len(shape_x) >= 2 and len(shape_y) == 1 and shape_y[0] == 1) or (
          len(shape_x) == 1 and len(shape_y) == 1 and shape_y[0] == 1 and shape_x[0] % 16 == 0):
        if len(shape_x) == 4 and len(shape_y) == 1 and format_x in format_4d_list:
            format_list.append("C1HWNCoC0")
            format_list.append("NC1HWC0")
            format_list.append("FRACTAL_Z")
        if len(shape_x) >= 2:
            if shape_x[-1] % 16 == 0 and shape_x[-2] % 16 == 0:
                format_list.append("FRACTAL_NZ")
        if len(shape_x) == 2 and len(shape_y) == 1:
            x_1dim = shape_x[-1]
            x_2dim = shape_x[-2]
            if x_1dim % 16 == 0 and x_2dim % 16 == 0:
                format_list.append("FRACTAL_NZ")
        if len(shape_x) == 1 and len(shape_y) == 1 and shape_y[0] == 1 and shape_x[0] % 16 == 0:
            format_list.append("NC1HWC0")
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype] * len(format_list)
        format_list = format_list * len_format_list
        format_list0 = format_list
        format_list1 = format_nd * len(format_list)
        unknownshape_format_list = ["ND"] * len(dtype_total)
        if -1 in shape_x or -1 in shape_y:
            input0 = util_select_op_base.gen_param(classify="input0",
                                                   name="x1",
                                                   datatype=",".join(dtype_total),
                                                   format=",".join(format_list0),
                                                   unknownshape_format=",".join(unknownshape_format_list))
            input1 = util_select_op_base.gen_param(classify="input1",
                                                   name="x2",
                                                   datatype=",".join(dtype_total),
                                                   format=",".join(format_list1),
                                                   unknownshape_format=",".join(unknownshape_format_list))
            output0 = util_select_op_base.gen_param(classify="output0",
                                                    name="y",
                                                    datatype=",".join(dtype_total),
                                                    format=",".join(format_list0),
                                                    unknownshape_format=",".join(unknownshape_format_list))
        else:
            input0 = util_select_op_base.gen_param(classify="input0",
                                                   name="x1",
                                                   datatype=",".join(dtype_total),
                                                   format=",".join(format_list0))
            input1 = util_select_op_base.gen_param(classify="input1",
                                                   name="x2",
                                                   datatype=",".join(dtype_total),
                                                   format=",".join(format_list1))
            output0 = util_select_op_base.gen_param(classify="output0",
                                                    name="y",
                                                    datatype=",".join(dtype_total),
                                                    format=",".join(format_list0))

    # ND+ND,scalar+5HD,scalar+FZ,scalar+6D,ND+NZ
    elif (len(shape_y) >= 2 and len(shape_x) == 1 and shape_x[0] == 1) or (
          len(shape_y) == 1 and len(shape_x) == 1 and shape_x[0] == 1 and shape_y[0] % 16 == 0):
        if len(shape_x) == 1 and len(shape_y) == 4 and format_y in format_4d_list:
            format_list.append("C1HWNCoC0")
            format_list.append("NC1HWC0")
            format_list.append("FRACTAL_Z")
        if len(shape_y) >= 2:
            if shape_y[-1] % 16 == 0 and shape_y[-2] % 16 == 0:
                format_list.append("FRACTAL_NZ")
        if len(shape_y) == 2 and len(shape_x) == 1:
            y_1dim = shape_y[-1]
            y_2dim = shape_y[-2]
            if y_1dim % 16 == 0 and y_2dim % 16 == 0:
                format_list.append("FRACTAL_NZ")
        if len(shape_y) == 1 and len(shape_x) == 1 and shape_x[0] == 1 and shape_y[0] % 16 == 0:
            format_list.append("NC1HWC0")
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype] * len(format_list)
        format_list = format_list * len_format_list
        format_list0 = format_list
        format_list1 = format_nd * len(format_list)
        unknownshape_format_list = ["ND"] * len(dtype_total)
        if -1 in shape_x or -1 in shape_y:
            input0 = util_select_op_base.gen_param(classify="input0",
                                                   name="x1",
                                                   datatype=",".join(dtype_total),
                                                   format=",".join(format_list1),
                                                   unknownshape_format=",".join(unknownshape_format_list))
            input1 = util_select_op_base.gen_param(classify="input1",
                                                   name="x2",
                                                   datatype=",".join(dtype_total),
                                                   format=",".join(format_list0),
                                                   unknownshape_format=",".join(unknownshape_format_list))
            output0 = util_select_op_base.gen_param(classify="output0",
                                                    name="y",
                                                    datatype=",".join(dtype_total),
                                                    format=",".join(format_list0),
                                                    unknownshape_format=",".join(unknownshape_format_list))
        else:
            input0 = util_select_op_base.gen_param(classify="input0",
                                                   name="x1",
                                                   datatype=",".join(dtype_total),
                                                   format=",".join(format_list1))
            input1 = util_select_op_base.gen_param(classify="input1",
                                                   name="x2",
                                                   datatype=",".join(dtype_total),
                                                   format=",".join(format_list0))
            output0 = util_select_op_base.gen_param(classify="output0",
                                                    name="y",
                                                    datatype=",".join(dtype_total),
                                                    format=",".join(format_list0))
    # ND+ND,5HD+5HD
    else:
        if len(shape_x) == 1 and len(shape_y) == 1 and shape_x[0] % 16 == 0 and shape_y[0] % 16 == 0:
            format_list.append("NC1HWC0")
        if len(shape_x) == 1 and len(shape_y) == 4:
            if format_x == format_y and format_y in ("NHWC",):
                format_list.append("NC1HWC0")
            if format_x == format_y and format_y in ("NCHW", "HWCN"):
                if y_cdim == shape_x[0] or y_cdim == 1 or shape_x[0] == 1 or shape_x[0] // 16 == 1:
                    format_list.append("NC1HWC0")
        if len(shape_y) == 1 and len(shape_x) == 4:
            if format_x == format_y and format_x in ("NHWC",):
                format_list.append("NC1HWC0")
            if format_x == format_y and format_x in ("NCHW", "HWCN"):
                if x_cdim == shape_y[0] or x_cdim == 1 or shape_y[0] == 1 or x_cdim // 16 == 1 or shape_y[0] // 16 == 1:
                    format_list.append("NC1HWC0")
        if len(shape_x) == 4 and len(shape_y) == 4 and format_x in format_4d_list and format_y in format_4d_list:
            if format_x == format_y == "NCHW" or format_x == format_y == "HWCN" or format_x == format_y == "NHWC":
                if x_cdim % 16 == 0 and y_cdim % 16 == 0:
                    if (x_cdim // 16 == 1 or y_cdim // 16 == 1) or (x_cdim == y_cdim):
                        if x_ndim == y_ndim:
                            if x_hdim == y_hdim and (x_wdim == 1 or y_wdim == 1):
                                format_list.append("NC1HWC0")
                            if x_wdim == y_wdim and (x_hdim == 1 or y_hdim == 1):
                                format_list.append("NC1HWC0")
                            if x_hdim == y_hdim and x_wdim == y_wdim:
                                format_list.append("NC1HWC0")
                            if (x_wdim == x_hdim == 1) or (y_wdim == y_hdim == 1):
                                format_list.append("NC1HWC0")
                            if (x_hdim == 1 and y_wdim == 1) or (x_wdim == 1 and y_hdim == 1):
                                format_list.append("NC1HWC0")
                        if x_hdim == y_hdim:
                            if x_ndim == y_ndim and (x_wdim == 1 or y_wdim == 1):
                                format_list.append("NC1HWC0")
                            if x_wdim == y_wdim and (x_ndim == 1 or y_ndim == 1):
                                format_list.append("NC1HWC0")
                            if x_ndim == y_ndim and x_wdim == y_wdim:
                                format_list.append("NC1HWC0")
                            if (x_ndim == x_wdim == 1) or (y_ndim == y_wdim == 1):
                                format_list.append("NC1HWC0")
                            if (x_ndim == 1 and y_wdim == 1) or (x_wdim == 1 and y_ndim == 1):
                                format_list.append("NC1HWC0")
                        if x_wdim == y_wdim:
                            if x_ndim == y_ndim and (x_hdim == 1 or y_hdim == 1):
                                format_list.append("NC1HWC0")
                            if x_hdim == y_hdim and (x_ndim == 1 or y_ndim == 1):
                                format_list.append("NC1HWC0")
                            if x_ndim == y_ndim and x_hdim == y_hdim:
                                format_list.append("NC1HWC0")
                            if (x_ndim == x_hdim == 1) or (y_ndim == y_hdim == 1):
                                format_list.append("NC1HWC0")
                            if (x_ndim == 1 and y_hdim == 1) or (x_hdim == 1 and y_ndim == 1):
                                format_list.append("NC1HWC0")
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype] * len(format_list)
        len_format_list = len(dtype_list)
        format_list = format_list * len_format_list
        unknownshape_format_list = ["ND"] * len(dtype_total)
        if -1 in shape_x or -1 in shape_y:
            input0 = util_select_op_base.gen_param(classify="input0",
                                                   name="x1",
                                                   datatype=",".join(dtype_total),
                                                   format=",".join(format_list),
                                                   unknownshape_format=",".join(unknownshape_format_list))
            input1 = util_select_op_base.gen_param(classify="input1",
                                                   name="x2",
                                                   datatype=",".join(dtype_total),
                                                   format=",".join(format_list),
                                                   unknownshape_format=",".join(unknownshape_format_list))
            output0 = util_select_op_base.gen_param(classify="output0",
                                                    name="y",
                                                    datatype=",".join(dtype_total),
                                                    format=",".join(format_list),
                                                    unknownshape_format=",".join(unknownshape_format_list))
        else:
            input0 = util_select_op_base.gen_param(classify="input0",
                                                   name="x1",
                                                   datatype=",".join(dtype_total),
                                                   format=",".join(format_list))
            input1 = util_select_op_base.gen_param(classify="input1",
                                                   name="x2",
                                                   datatype=",".join(dtype_total),
                                                   format=",".join(format_list))
            output0 = util_select_op_base.gen_param(classify="output0",
                                                    name="y",
                                                    datatype=",".join(dtype_total),
                                                    format=",".join(format_list))

    param_list = [input0, input1, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def _mul_check_format(x, y):
    """
    _mul_check_format
    """
    format_pattern = 0
    shape1 = x.get("shape")
    shape2 = y.get("shape")
    list_format = [x.get("format"), y.get("format")]
    shape1 = shape_util.scalar2tensor_one(shape1)
    shape2 = shape_util.scalar2tensor_one(shape2)
    check_list = [["FRACTAL_NZ", "ND"], ["ND", "FRACTAL_NZ"], ["FRACTAL_NZ", "NHWC"], ["NHWC", "FRACTAL_NZ"],
                  ["FRACTAL_NZ", "NCHW"], ["NCHW", "FRACTAL_NZ"]]
    if list_format == check_list[0] and (len(shape2) != 1 or (len(shape2) == 1 and shape2[0] != 1)):
        format_pattern = 1
    elif list_format == check_list[1] and (len(shape1) != 1 or (len(shape1) == 1 and shape1[0] != 1)):
        format_pattern = 2
    elif list_format == check_list[2] and (len(shape2) != 1 or (len(shape2) == 1 and shape2[0] != 1)):
        format_pattern = 1
    elif list_format == check_list[3] and (len(shape1) != 1 or (len(shape1) == 1 and shape1[0] != 1)):
        format_pattern = 2
    elif list_format == check_list[4] and (len(shape2) != 1 or (len(shape2) == 1 and shape2[0] != 1)):
        format_pattern = 1
    elif list_format == check_list[5] and (len(shape1) != 1 or (len(shape1) == 1 and shape1[0] != 1)):
        format_pattern = 2

    return format_pattern


# pylint: disable=unused-variable,invalid-name
def _infer_shape(format_pattern, x, y):
    """
    _infer_shape
    """
    shape_x = x.get("shape")
    shape_y = y.get("shape")
    ori_shape_x = x.get("ori_shape")
    ori_shape_y = y.get("ori_shape")
    shape_x = shape_util.scalar2tensor_one(shape_x)
    shape_y = shape_util.scalar2tensor_one(shape_y)
    if format_pattern == 1:
        ori_shape_x, shape_y, shape_max = shape_util.broadcast_shapes(ori_shape_x,
                                                                      shape_y,
                                                                      param_name_input1="x",
                                                                      param_name_input2="y")
        if shape_y[-2] == ori_shape_x[-2] and shape_y[-1] == ori_shape_x[-1]:
            error_manager_vector.raise_err_check_params_rules("mul", "the input shape of y is illegal", 'y', shape_y)

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
        shape_x, ori_shape_y, shape_max = shape_util.broadcast_shapes(shape_x,
                                                                      ori_shape_y,
                                                                      param_name_input1="x",
                                                                      param_name_input2="y")
        if shape_x[-2] == ori_shape_y[-2] and shape_x[-1] == ori_shape_y[-1]:
            error_manager_vector.raise_err_check_params_rules("mul", "the input shape of x is illegal", 'x', shape_x)

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


@register_operator_compute("Mul", op_mode="static", support_fusion=True)
def mul_compute(input_x, input_y, output_data, is_scene_1D=False, kernel_name="mul"):
    """
    calculating element-wise mul

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of first input data
    input_y: TVM tensor
        the placeholder of second input data
    output_data: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name: str
        cce kernel name, default value is "mul"

    Returns
    -------
    output of the element-wise mul
    """
    shape_x = shape_util.shape_to_list(input_x.shape)
    shape_y = shape_util.shape_to_list(input_y.shape)
    x_dtype = input_x.dtype.lower()

    if is_scene_1D:
        if x_dtype in ("uint8", "int8"):
            input_x = tbe.cast_to(input_x, "float32")
            input_y = tbe.cast_to(input_y, "float32")
        input_y = tbe.broadcast(input_y, shape_x)
    else:
        shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x,
                                                                  shape_y,
                                                                  param_name_input1="x",
                                                                  param_name_input2="y")
        if shape_x != shape_y and len(shape_x) == 2 and len(shape_y) == 2:
            res = _mul_compute_ex(input_x, input_y, shape_x, shape_y, shape_max)
            if res is not None:
                return res

        if x_dtype in ("uint8", "int8"):
            input_x = tbe.cast_to(input_x, "float32")
            input_y = tbe.cast_to(input_y, "float32")

        input_x = tbe.broadcast(input_x, shape_max)
        input_y = tbe.broadcast(input_y, shape_max)
    res = tbe.vmul(input_x, input_y)

    if x_dtype in ("uint8", "int8"):
        res = util_common.uint8_int8_overflow_proc(res, x_dtype)

    return res


# pylint: disable=too-many-arguments,unused-argument,bad-continuation
def _mul_compute_ex(input_x, input_y, shape_x, shape_y, shape_max):
    """
    _mul_compute_ex
    """
    if shape_x == shape_max:
        small_input = input_y
        large_input = input_x
    elif shape_y == shape_max:
        small_input = input_x
        large_input = input_y
    else:
        return None

    small_index = []
    # Minimum shape size
    small_shape = 1
    for _, i in enumerate(range(len(small_input.shape))):
        if int(small_input.shape[i]) < int(shape_max[i]):
            small_index.append(i)
            small_shape *= shape_max[i]
        elif int(small_input.shape[i]) == int(shape_max[i]):
            pass
        else:
            return None

    if small_shape < 10000:
        return None

    if int(small_input.shape[-1]) != 1:
        return None

    def get_tensor_slice(inp, small_index, is_large, *shapes):
        def get_index(inp_tensor, index):
            return inp_tensor[index]

        if is_large:
            for axis in shapes:
                inp = get_index(inp, axis)
        else:
            for ind, _ in enumerate(shapes):
                if ind in small_index:
                    inp = get_index(inp, 0)
                else:
                    inp = get_index(inp, shapes[ind])

        return inp

    with tvm.tag_scope("elewise_binary_mul"):
        res = tvm.compute(shape_max,
                          lambda *indices: get_tensor_slice(large_input, small_index, True, *indices) *
                                           get_tensor_slice(small_input, small_index, False, *indices),
                          name="manual_mul_without_broadcast_" + str(tbe.te_compute.elewise_compute.NAME_INDEX[0]))
    tbe.te_compute.elewise_compute.NAME_INDEX[0] += 1

    return res


# pylint: disable=unused-argument, too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def mul(x, y, output, kernel_name="mul"):
    """
    do element-wise mul operation between two input tensors

    Parameters:
    ----------
    x : dict.
        shape, dtype of input x
    y : dict.
        shape, dtype of input y
    output : dict.
        shape, dtype of ouput
    kernel_name : str.
        cce kernel name, default value is "mul"

    Returns
    -------
    None
    """
    # format_pattern = 1  Nz and vector
    # format_pattern = 2  vector and Nz
    # format_pattern = 0  Nz scalar  Nz Nz  ND ND
    format_pattern = _mul_check_format(x, y)
    shape_x, shape_y = _infer_shape(format_pattern, x, y)

    shape_x = shape_util.scalar2tensor_one(shape_x)
    dtype_x = x.get("dtype").lower()
    shape_y = shape_util.scalar2tensor_one(shape_y)
    dtype_y = y.get("dtype").lower()

    para_check.check_shape(shape_x, param_name="x")
    para_check.check_shape(shape_y, param_name="y")

    if dtype_x != dtype_y:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, 'x', 'y', dtype_x, dtype_y)
    check_list = ("int32", "float16", "float32", "int16", "uint8", "int8")
    para_check.check_dtype(dtype_x, check_list, param_name="x")

    vmul_support = tbe_platform.api_check_support("tbe.dsl.vmul", "float32")
    if not vmul_support:
        new_check_list = list(check_list)
        new_check_list.remove("float32")
        para_check.check_dtype(dtype_x, new_check_list, param_name="x")

    cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if para_check.is_scalar(shape_y) and cce_product in ("Ascend910"):
        is_scene_1D = True
        shape_y = tuple([1] * (len(shape_x) - len(shape_y))) + tuple(shape_y)
    else:
        is_scene_1D = False
        shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x,
                                                                  shape_y,
                                                                  param_name_input1="x",
                                                                  param_name_input2="y")
        shape_x, shape_y = shape_util.refine_shapes_for_broadcast(shape_x, shape_y)

    input_x = tvm.placeholder(shape_x, dtype=dtype_x, name="x")
    input_y = tvm.placeholder(shape_y, dtype=dtype_x, name="y")

    res = mul_compute(input_x, input_y, output, is_scene_1D, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": (input_x, input_y, res)}
    tbe.build(sch, config)
