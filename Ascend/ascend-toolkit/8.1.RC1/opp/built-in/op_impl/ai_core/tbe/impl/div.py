#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
div
"""
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import error_manager_vector
from impl.dynamic.div import op_select_format as static_op_select_format
from impl.dynamic.div import check_format
from impl.dynamic.div import infer_shape
from impl.dynamic.div import calc_input_tensor
from impl.dynamic.div import div_compute_for_batchmatmul


# 'pylint: disable=unused-argument,too-many-locals,invalid-name,too-many-branches,too-many-statements
# 'pylint: disable=too-many-boolean-expressions,too-many-nested-blocks
def op_select_format(x, y, output, kernel_name="div"):
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

    27.when x's format is NZ and length of x > 2, not 16 align, y is a scalar, y's format is ND:
    support NZ, ND format.\n


        example:\n
        original:\n
        x's Tensor(shape=(20, 28, 15, 16), "NZ")\n
        y's Tensor(shape=(1,), "ND")\n
        support conversion to NZ operation:\n
        x's Tensor(shape=(20, 28, 15, 16), "NZ")\n
        y's Tensor(shape=(1,), "ND")\n
    """
    return static_op_select_format(x, y, output, kernel_name)


# 'pylint: disable=locally-disabled,too-many-locals,unused-argument
@register_operator_compute("div", op_mode="static", support_fusion=True)
def div_compute(input_x, input_y, output_div, kernel_name="div"):
    """
    div compute
    calculating data's div, res =x / y

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input_x
    input_y: TVM tensor
        the placeholder of input_y
    output_div: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        kernel name, default value is "div"

    Returns
    -------
    res: TVM tensor
        the result of div compute
    """
    batchmatmul_flag, input_x, input_y = calc_input_tensor(input_x, input_y)
    if batchmatmul_flag:
        return div_compute_for_batchmatmul(input_x, input_y)

    input_data1 = shape_util.shape_to_list(input_x.shape)
    input_data2 = shape_util.shape_to_list(input_y.shape)
    shape_list = shape_util.broadcast_shapes(input_data1,
                                             input_data2,
                                             param_name_input1="input_x",
                                             param_name_input2="input_y")
    dtype = input_x.dtype
    int_list = ("int8", "uint8", "int32")
    int_flag = dtype in int_list
    if tbe_platform.api_check_support("tbe.dsl.vdiv", "float32"):
        input_x = tbe.cast_to(input_x, "float32")
        input_y = tbe.cast_to(input_y, "float32")
    data_x_broad = tbe.broadcast(input_x, shape_list[2])
    data_y_broad = tbe.broadcast(input_y, shape_list[2])
    res = tbe.vdiv(data_x_broad, data_y_broad)

    if int_flag:
        res = tbe.floor(res)

    res = tbe.cast_to(res, dtype)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def div(input_x, input_y, output_div, kernel_name="div"):
    """
    algorithm: div
    calculating data's div, res =x / y

    Parameters
    ----------
    input_x: dict
        dict with keys(shape and dtype) of input_x
    input_y: dict
        dict with keys(shape and dtype) of input_y
    output_div: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        kernel name, default value is "div"

    Returns
    -------
    None
    """
    # `format_pattern = 1  Nz and vector`
    # `format_pattern = 2  vector and Nz`
    # `format_pattern = 0  Nz scalar  Nz Nz  ND ND
    format_pattern = check_format(input_x, input_y)
    shape_x, shape_y = infer_shape(format_pattern, input_x, input_y)
    shape_x = shape_util.scalar2tensor_one(shape_x)
    shape_y = shape_util.scalar2tensor_one(shape_y)
    para_check.check_shape(shape_x, param_name="input_x")
    para_check.check_shape(shape_y, param_name="x2")

    dtype_x = input_x.get("dtype").lower()
    dtype_y = input_y.get("dtype").lower()

    if dtype_x != dtype_y:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, 'x', 'y', dtype_x, dtype_y)
    check_list = ("int32", "float16", "float32", "int16", "uint8", "int8")
    para_check.check_dtype(dtype_x, check_list, param_name="x")

    vmul_support = tbe_platform.api_check_support("tbe.dsl.vmul", "float32")
    if not vmul_support:
        new_check_list = list(check_list)
        new_check_list.remove("float32")
        para_check.check_dtype(dtype_x, new_check_list, param_name="x")

    shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x,
                                                              shape_y,
                                                              param_name_input1="x1",
                                                              param_name_input2="x2")
    if shape_x[-1] == 1 and shape_y[-1] == 1 and shape_max[-1] == 1:
        shape_x = shape_x if len(shape_x) == 1 else shape_x[:-1]
        shape_y = shape_y if len(shape_y) == 1 else shape_y[:-1]
        shape_max = shape_max if len(shape_max) == 1 else shape_max[:-1]

    shape_x, shape_y = shape_util.refine_shapes_for_broadcast(shape_x, shape_y)
    data_x = tvm.placeholder(shape_x, name="data_x", dtype=dtype_x)
    data_y = tvm.placeholder(shape_y, name="data_y", dtype=dtype_y)

    res = div_compute(data_x, data_y, output_div, kernel_name)
    with tvm.target.cce():
        sch = auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_x, data_y, res]}
    build(sch, config)
