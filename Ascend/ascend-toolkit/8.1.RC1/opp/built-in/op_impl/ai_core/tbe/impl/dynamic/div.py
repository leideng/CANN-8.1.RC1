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
dynamic div
"""
import functools
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import in_record
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_compute import batchmatmul_elem_nd2nz
from impl.util.util_compute import batchmatmul_elem_reshape
from impl.util.util_compute import check_batchmatmul_fuse
from impl.util import util_common
from impl.util import util_select_op_base
from impl import constant_util as constant
from impl.util.util_soc_common import is_v200


# 'pylint: disable=too-few-public-methods,not-use-list-comprehension
class Constant:
    """
    The class for constant
    """
    # Determine whether the 16 bit alignment
    SIZE_SIXTEEN = 16


# 'pylint: disable=unused-argument,too-many-locals,invalid-name,too-many-branches,too-many-statements
# 'pylint: disable=too-many-boolean-expressions,too-many-nested-blocks,too-many-arguments,too-many-boolean-expressions
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
    shape_x = x.get("ori_shape")
    shape_y = y.get("ori_shape")

    shape_x = shape_util.scalar2tensor_one(shape_x)
    shape_y = shape_util.scalar2tensor_one(shape_y)
    reduce_y = functools.reduce(lambda x, y: x * y, shape_y)

    format_4d_list = ["NCHW", "NHWC", "HWCN"]
    format_5d_list = ["NDHWC", "DHWCN", "NCDHW"]
    dtype_list = ["float16", "float", "int32", "int16", "uint8", "int8"]
    vdiv_support_s16 = tbe_platform.api_check_support("te.lang.cce.vdiv", "int16")
    vdiv_support_fp32 = tbe_platform.api_check_support("te.lang.cce.vdiv", "float32")
    if tbe_platform.intrinsic_check_support("Intrinsic_vconv", "f322bf16f"):
        dtype_list.append("bfloat16")
    if not vdiv_support_s16:
        dtype_list.remove("int16")
    if not vdiv_support_fp32:
        dtype_list.remove("float")
        # If the platform does not support float32 data type,
        # neither of uint8 and int8 is supported at the same time
        dtype_list.remove("uint8")
        dtype_list.remove("int8")
    if tbe_platform.api_check_support("tbe.dsl.vdiv", "complex32"):
        dtype_list.append("complex32")
    if tbe_platform.api_check_support("tbe.dsl.vdiv", "complex64"):
        dtype_list.append("complex64")

    format_x = x.get("ori_format")
    format_y = y.get("ori_format")

    dtype_total = []
    format_nd = ["ND"]
    len_format_list = len(dtype_list)
    list_input = [x, y]

    x_flag = {
        "5d": len(shape_x) == 5 and format_x in format_5d_list,
        "4d": len(shape_x) == 4 and format_x in format_4d_list,
        "Scalar": len(shape_x) == 1 and shape_x[0] == 1
    }
    y_flag = {
        "5d": len(shape_y) == 5 and format_y in format_5d_list,
        "4d": len(shape_y) == 4 and format_y in format_4d_list,
        "Scalar": len(shape_y) == 1 and shape_y[0] == 1
    }
    if x_flag.get("5d") or x_flag.get("4d"):
        x_cdim = shape_x[format_x.index("C")]
        x_ndim = shape_x[format_x.index("N")]
    if y_flag.get("5d") or y_flag.get("4d"):
        y_cdim = shape_y[format_y.index("C")]
        y_ndim = shape_y[format_y.index("N")]

    format_flag = {
        "NDC1HWC0":
            x_flag.get("5d") and y_flag.get("5d") and x_cdim == y_cdim,
        "FRACTAL_Z_3D":
            x_flag.get("5d") and y_flag.get("5d") and x_cdim == y_cdim and x_ndim == y_ndim,
        "FRACTAL_NZ":
            len(shape_x) >= 2 and len(shape_y) >= 2 and shape_x[-2:] == shape_y[-2:],
        "NC1HWC0":
            x_flag.get("4d") and y_flag.get("4d") and
            ((format_y == format_x and ((x_cdim % 16 == 0 and y_cdim % 16 == 0) or x_cdim == y_cdim) and _can_broad([
                shape_x[format_x.index(format_x[0])], shape_x[format_x.index(format_x[1])], format_x[0] != "C",
                format_x[1] != "C"
            ], [
                shape_y[format_y.index(format_y[0])], shape_y[format_y.index(format_y[1])], format_y[0] != "C",
                format_y[1] != "C"
            ])) or (list(shape_x) == list(shape_y) and -1 not in shape_x)),
        "FRACTAL_Z":
            x_flag.get("4d") and y_flag.get("4d") and format_x == format_y and
            ((x_cdim % 16 == 0 and y_cdim % 16 == 0 and y_ndim % 16 == 0 and x_ndim % 16 == 0 and
              util_common.is_support_fractal_z_inputs(list_input) and
              _broadcast_zn_rule(shape_x, shape_y, format_x, format_y)) or
             (x_cdim == 1 and y_cdim == 1 and format_x.upper() in ("NCHW", "HWCN")) or
             (list(shape_x) == list(shape_y) and util_common.is_same_group(list_input)) or
             (x_cdim % 16 == 0 and x_ndim % 16 == 0 and y_cdim % 16 == 0 and y_ndim % 16 == 0 and
              util_common.is_support_fractal_z_inputs(list_input) and
              _broadcast_zn_rule(shape_x, shape_y, format_x, format_y))),
        "ND":
            True
    }

    format_flag["C1HWNCoC0"] = (x_flag.get("4d") and y_flag.get("Scalar")) or \
        (x_flag.get("Scalar") and y_flag.get("4d"))
    format_flag["NC1HWC0"] = format_flag.get("NC1HWC0") or (x_flag.get("4d") and \
        y_flag.get("Scalar")) or (x_flag.get("Scalar") and y_flag.get("4d"))
    format_flag["FRACTAL_Z"] = format_flag.get("NC1HWC0") or (x_flag.get("4d") and \
        y_flag.get("Scalar")) or (x_flag.get("Scalar") and y_flag.get("4d"))
    format_flag["FRACTAL_NZ"] = format_flag.get("FRACTAL_NZ") or (len(shape_x) >= 2 and \
        y_flag.get("Scalar")) or (len(shape_y) >= 2 and x_flag.get("Scalar") and \
            shape_y[-1] % 16 == 0 and shape_y[-2] % 16 == 0)

    format_flag["NC1HWC0"] = format_flag.get("NC1HWC0") or \
        (len(shape_x) == len(shape_y) == 1 and shape_x[0] % 16 == shape_y[0] % 16 == 0) or \
        (len(shape_x) == 1 and y_flag.get("4d") and format_x == format_y and
         ((format_y in ("NHWC",) and reduce_y != 1) or
          (format_y in ("NCHW", "HWCN") and
           (y_cdim == shape_x[0] or y_cdim == 1 or shape_x[0] == 1 or shape_x[0] // 16 == 1)))) or \
        (len(shape_y) == 1 and x_flag.get("4d") and format_x == format_y and
         (format_x in ("NHWC",) or
          (format_x in ("NCHW", "HWCN") and
           (x_cdim == shape_y[0] or x_cdim == 1 or shape_y[0] == 1 or shape_y[0] // 16 == 1)))) or \
        (x_flag.get("4d") and y_flag.get("4d") and x_cdim % 16 == 0 and y_cdim % 16 == 0 and ())

    # NDC1HWC0 FRACTAL_Z_3D
    format_list = [i for i in format_flag if format_flag.get(i)]

    # ND+ND NZ+NZ 5HD+5HD FZ+FZ
    if len(shape_x) >= 2 and len(shape_y) >= 2:

        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype] * len(format_list)
        format_list = format_list * len_format_list
        unknownshape_format_list = ["ND"] * len(format_list)
        param_list = _gen_para(dtype_total, format_list, format_list, format_list, unknownshape_format_list, shape_x,
                               shape_y)

    # 5HD+scalar,ND+ND,FZ+scalar,6D+scalar,NZ+ND
    elif len(shape_x) >= 2 and len(shape_y) == 1 and shape_y[0] == 1:

        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype] * len(format_list)
        format_list = format_list * len_format_list
        format_list0 = format_list
        format_list1 = format_nd * len(format_list)
        unknownshape_format_list = ["ND"] * len(dtype_total)
        param_list = _gen_para(dtype_total, format_list0, format_list1, format_list0, unknownshape_format_list, shape_x,
                               shape_y)

    # ND+ND,scalar+5HD,scalar+FZ,scalar+6D,ND+NZ
    elif len(shape_y) >= 2 and len(shape_x) == 1 and shape_x[0] == 1:

        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype] * len(format_list)
        format_list = format_list * len_format_list
        format_list0 = format_list
        format_list1 = format_nd * len(format_list)
        unknownshape_format_list = ["ND"] * len(dtype_total)
        param_list = _gen_para(dtype_total, format_list1, format_list0, format_list0, unknownshape_format_list, shape_x,
                               shape_y)
    # ND+ND,5HD+5HD
    else:

        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype] * len(format_list)
        len_format_list = len(dtype_list)
        format_list = format_list * len_format_list
        unknownshape_format_list = ["ND"] * len(dtype_total)
        param_list = _gen_para(dtype_total, format_list, format_list, format_list, unknownshape_format_list, shape_x,
                               shape_y)

    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def _gen_para(dtype_total, format_list0, format_list1, format_list2, unknownshape_format_list, shape_x, shape_y):
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
                                                format=",".join(format_list2),
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
                                                format=",".join(format_list2))
    return [input0, input1, output0]


def _broadcast_zn_rule(shape0, shape1, format0, format1):
    """
    _broadcast_zn_rule
    """
    if format1 != format0:
        format_rule = "format should be same"
        error_manager_vector.raise_err_check_params_rules("div", format_rule, "x", format0)

    if len(shape0) != len(shape1) != len(format0):
        expected_value = "equal to 4"
        real_value = "not equal to 4"
        error_manager_vector.raise_err_input_value_invalid("div", "length of shapes", expected_value, real_value)

    x_cdim = shape0[format0.index("C")]
    x_wdim = shape0[format0.index("W")]
    x_hdim = shape0[format0.index("H")]
    x_ndim = shape0[format0.index("N")]
    y_cdim = shape1[format1.index("C")]
    y_wdim = shape1[format1.index("W")]
    y_hdim = shape1[format1.index("H")]
    y_ndim = shape1[format1.index("N")]

    x_c0 = constant.C0_SIZE
    x_n0 = constant.C0_SIZE
    x_c1 = x_cdim // 16
    x_n1 = x_ndim // 16
    shape0_zn = [x_hdim * x_wdim * x_c1, x_n1, x_n0, x_c0]

    y_c0 = constant.C0_SIZE
    y_n0 = constant.C0_SIZE
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


def _can_division_sixteen(shape):
    """
    _can_division_sixteen
    """
    if len(shape) < 2:
        if shape[-1] == 0:
            expected_value = "equal to 0"
            real_value = "not equal to 0"
            error_manager_vector.raise_err_input_value_invalid("div", "value of shape", expected_value, real_value)
        return False

    if shape[-1] == 0 or shape[-2] == 0:
        expected_value = "equal to 0"
        real_value = "not equal to 0"
        error_manager_vector.raise_err_input_value_invalid("div", "value of shape", expected_value, real_value)
    if shape[-1] % Constant.SIZE_SIXTEEN == 0 and shape[-2] % Constant.SIZE_SIXTEEN == 0:
        return True

    return False


def _can_broad(x, y):
    if x[2]:
        x[0] *= 16
        y[0] *= 16
    if x[3]:
        x[1] *= 16
        y[1] *= 16
    return (x[0] == y[0] and (x[1] == 16 or y[1] == 16 or x[1] == y[1])) or (
        x[1] == y[1] and
        (x[0] == 16 or
         y[0] == 16)) or x[0] == y[1] == 16 or x[0] == x[1] == 16 or x[1] == y[0] == 16 or y[0] == y[1] == 16


def check_format(x, y):
    """
    funtion to check format

    Parameters
    ----------
    x: dict
        dict of x, include keys(shape and dtype).
    y: dict
        dict of x, include keys(shape and dtype).

    Returns:
    -------
    format_pattern: int
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


def infer_shape(format_pattern, x, y):
    """
    funtion to infer shape

    Parameters
    ----------
    format_pattern: int
    x: dict
        dict of x, include keys(shape and dtype).
    y: dict
        dict of x, include keys(shape and dtype).

    Returns:
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
        ori_shape_x, shape_y, _ = shape_util.broadcast_shapes(ori_shape_x,
                                                              shape_y,
                                                              param_name_input1="input_x",
                                                              param_name_input2="input_y")
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
        shape_x, ori_shape_y, _ = shape_util.broadcast_shapes(shape_x,
                                                              ori_shape_y,
                                                              param_name_input1="input_x",
                                                              param_name_input2="input_y")
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


def calc_input_tensor(input_x, input_y):
    """compute with batchmatmul"""
    batchmatmul_flag = False
    batch_matmul_flag_lhs = check_batchmatmul_fuse(input_x)
    batch_matmul_flag_rhs = check_batchmatmul_fuse(input_y)
    if batch_matmul_flag_lhs or batch_matmul_flag_rhs:
        batchmatmul_flag = True
        if batch_matmul_flag_rhs:
            input_x, input_y = input_y, input_x
    return batchmatmul_flag, input_x, input_y


def div_compute_for_batchmatmul(input_x, input_y):
    """
    div compute
    calculating data's div, res = x / y

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input_x
    input_y: TVM tensor
        the placeholder of input_y
    Returns
    -------
    res: TVM tensor
        the result of div compute
    """
    if "para_name" in input_x.op.attrs:
        para_name = input_x.op.attrs["para_name"].value
        para_name += "_div"
    else:
        para_name = "div"
    dtype = input_x.dtype
    batch_shape = shape_util.shape_to_list(input_x.op.attrs["batch_shape"])
    para_dict = {"format_elem": input_y.op.attrs["format"], "batch_shape": batch_shape}
    if tbe_platform.api_check_support("tbe.dsl.vdiv", "float32"):
        input_x = tbe.cast_to(input_x, "float32")
        input_y = tbe.cast_to(input_y, "float32")
    input_y, shape_max = batchmatmul_elem_nd2nz(input_x, input_y, para_dict, para_name)
    input_y = tbe.broadcast(input_y, shape_max)
    input_y = batchmatmul_elem_reshape(input_x, input_y, batch_shape, para_name)

    res = tbe.vdiv(input_x, input_y)
    res = tbe.cast_to(res, dtype)
    res.op.attrs["batch_shape"] = batch_shape
    res.op.attrs["para_name"] = para_name
    return res


@register_operator_compute("Div", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def div_compute(input_x, input_y, output_z, kernel_name="div"):
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
    complex_list = ("complex32", "complex64")
    if input_x.dtype in complex_list:
        x_shape, y_shape, z_shape = shape_util.broadcast_shapes(input_x.shape,
                                                                input_y.shape,
                                                                param_name_input1="input_x",
                                                                param_name_input2="input_y")
        input_x = tbe.broadcast(input_x, z_shape)
        input_y = tbe.broadcast(input_y, z_shape)
        res = tbe.vdiv(input_x, input_y)
        return res

    if not in_record():
        batchmatmul_flag, input_x, input_y = calc_input_tensor(input_x, input_y)
        if batchmatmul_flag:
            return div_compute_for_batchmatmul(input_x, input_y)

    x_shape, y_shape, z_shape = shape_util.broadcast_shapes(input_x.shape,
                                                            input_y.shape,
                                                            param_name_input1="input_x",
                                                            param_name_input2="input_y")
    dtype_x = input_x.dtype
    int_list = ("int8", "uint8", "int32")
    if tbe_platform.api_check_support("te.lang.cce.vdiv", "float32"):
        input_x = tbe.cast_to(input_x, "float32")
        input_y = tbe.cast_to(input_y, "float32")
    input_x = tbe.broadcast(input_x, z_shape)
    input_y = tbe.broadcast(input_y, z_shape)
    res = tbe.vdiv(input_x, input_y)

    if dtype_x in int_list:
        if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") == "Ascend310":
            res = tbe.cast_to(res, "float16")
        res = tbe.floor(res)
        if dtype_x in ("int8", "uint8") and is_v200():
            return util_common.int_cast_to_b8(res, dtype_x)

    res = tbe.cast_to(res, dtype_x)

    return res


# 'pylint: disable=redefined-argument-from-local
@register_operator("Div")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def div(input_x, input_y, output_z, kernel_name="div"):
    """
    algorithm: div
    calculating data's div, res =x / yq


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
    # check dtype
    if not util_common.is_unknown([input_x, input_y]):
        format_pattern = check_format(input_x, input_y)
        shape_x, shape_y = infer_shape(format_pattern, input_x, input_y)
        range_x = util_common.gen_range(shape_x)
        range_y = util_common.gen_range(shape_y)
        input_x["shape"] = shape_x
        input_x["range"] = range_x
        input_y["shape"] = shape_y
        input_y["range"] = range_y

    x_dtype = input_x.get("dtype").lower()
    y_dtype = input_y.get("dtype").lower()
    check_list = ("bfloat16", "float16", "float32", "int8", "uint8", "int32", "complex32", "complex64")
    para_check.check_dtype(x_dtype, check_list, param_name="input_x")
    para_check.check_dtype(y_dtype, check_list, param_name="input_y")

    if x_dtype != y_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal("div", "input_x", "input_y", str(x_dtype), str(y_dtype))

    ins = classify([input_x, input_y], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (input_x, input_y) in ins:
        with tbe.compute():
            x_shape, y_shape = shape_util.variable_shape([input_x, input_y])
            tensor_x = tvm.placeholder(x_shape, x_dtype, "tensor_x")
            tensor_y = tvm.placeholder(y_shape, y_dtype, "tensor_y")
            res = div_compute(tensor_x, tensor_y, output_z, kernel_name)

            tensors.append([tensor_x, tensor_y, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
