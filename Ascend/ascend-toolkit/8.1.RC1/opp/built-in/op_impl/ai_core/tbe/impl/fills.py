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
fill

Op_description :
This operation creates a tensor of shape `dims` and fills it with `value`.

# fill(
#   x,
#   y,
#   value,
#   kernel_name='fill'
# )

Supportive_dtype_format :
['int32', 'float32', 'float16']
all format
"""
from functools import reduce as reduceIns

import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util


# 'pylint: disable=invalid-name,unused-argument
@register_operator_compute("fills", op_mode="static", support_fusion=True)
def fills_compute(x, value, dtype, kernel_name="fills"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of input
    value : a number of float or int
    dtype : string
        the type of input
    kernel_name : str
        kernel name, default value is "fills"

    Returns
    -------
    res: TVM tensor
        the calculation results
    """
    if dtype == "int8":
        res = tbe.broadcast(tvm.const(value, dtype="float16"), x.shape)
        res = tbe.cast_to(res, dtype)
    else:
        res = tbe.broadcast(tvm.const(value, dtype=dtype), x.shape)
    with tvm.tag_scope("elewise_binary_phony"):
        res = tvm.compute(res.shape,
                          lambda *indices: res[indices] + x[indices],
                          name="elewise_binary_phony_output")

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def fills(x, y, value, kernel_name="fills"):
    """
    do  fill operation

    Parameters:
    ----------
    x : the dict of output
    y :  the dict of output
    value:  scalar  value,
    kernel_name : cce kernel name, default value is "fill"

    Returns
    -------
    None
    """
    # get the shape and dtype
    shape = x.get("shape")
    dtype = x.get("dtype").lower()

    # check whether dtypes are right
    check_list = ("int8", "int32", "float16", "float32")
    para_check.check_dtype(dtype, check_list)

    # fuse shapes
    shape = shape_util.shape_refine(shape)
    fuseshape = [1]
    fuseshape[0] = reduceIns(lambda x, y: x * y, shape)
    data_x = tvm.placeholder(fuseshape, name="data_x", dtype=dtype)

    res = fills_compute(data_x, value, dtype)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {
        "name": kernel_name,
        "tensor_list": (data_x, res),
        "print_ir": False
    }
    tbe.cce_build_code(sch, config)
