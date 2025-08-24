# Copyright 2018 Huawei Technologies Co., Ltd
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
leaky_relu
"""
from functools import reduce as reduceIns

from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import get_current_build_config
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm


# 'pylint: disable=invalid-name,too-many-locals
def get_fusion_params(x_tensor, y):
    """
    Get L1 fusion_params
    Parameters
    ----------
    x_tensor : tensor of input data
    y : dict of output data
    Returns
    -------
    fusion_params
    """
    # 0: L1 depth fusion, 1: L1 width fusion, -1: no L1 fusion
    l1_fusion_type = -1
    if not get_current_build_config("enable_op_prebuild"):
        l1_fusion_type = x_tensor.op.attrs["L1_fusion_type"].value if "L1_fusion_type" in x_tensor.op.attrs else -1
        if l1_fusion_type == 1:
            error_manager_vector.raise_err_specific_reson("leaky_relu", "leaky_relu does not support l1 width fusion")

    out_l1_flag = False
    if y is not None:
        out_l1_flag = y.get("addr_type", 0) == 1

    fusion_params = {"l1_fusion_type": l1_fusion_type,
                     "out_l1_flag": out_l1_flag}
    return fusion_params


# 'pylint: disable=locally-disabled,unused-argument,invalid-name
@register_operator_compute("leaky_relu", op_mode="static", support_fusion=True)
def leaky_relu_compute(x, y, negative_slope=0, kernel_name="leaky_relu"):
    """
    compute for caffe_relu_layer_cce
    """
    fusion_params = get_fusion_params(x, y)
    res = tbe.vlrelu(x, negative_slope)
    if x.op.attrs:
        if 'format' in x.op.attrs:
            res.op.attrs['format'] = x.op.attrs['format']
        if 'ori_shape' in x.op.attrs:
            res.op.attrs['ori_shape'] = x.op.attrs['ori_shape']
        if 'ori_format' in x.op.attrs:
            res.op.attrs['ori_format'] = x.op.attrs['ori_format']
    res.op.attrs["negative_slope"] = negative_slope
    res.op.attrs["ele_fusion_params"] = fusion_params
    res.op.attrs["L1_fusion_type"] = fusion_params["l1_fusion_type"]

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def leaky_relu(x, y, negative_slope=0, kernel_name="leaky_relu"):
    """leaky_relu op for input tensor

       f(x)= x(x>=0) or negative_slope*x(x<0) equal to
       f(x)=negative_slope*x

    Parameters
    ----------
    x : TVM tensor
        input tensor has shape and dtype attributes
    y : dict
        dict with keys(shape and dtype) of output

    negative_slope : float or int
        allow non-zero slope for negative inputs to speed up optimization

    kernel_name : str
        cce kernel name, default value is "leaky_relu"

    Returns
    ------
    None
    """

    # check input tensor shape
    shape = x.get("shape")
    dtype = x.get("dtype")
    para_check.check_shape(shape, param_name="x")

    # check input tensor data_type
    check_list = ["float16", "float32", "int32", "int8"]
    para_check.check_dtype(dtype.lower(), check_list, param_name="x")
    fuseshape = [1]
    fuseshape[0] = reduceIns(lambda x, y: x*y, shape)
    inp_dtype = dtype.lower()

    l1_fusion_type = -1
    if not get_current_build_config("enable_op_prebuild"):
        l1_fusion_type = x.get("L1_fusion_type", -1)
        if l1_fusion_type == 1:
            error_manager_vector.raise_err_specific_reson("leaky_relu", "leaky_relu does not support l1 width fusion")
    is_l1_depth_fusion = l1_fusion_type == 0
    addr_type = x.get("addr_type", 0)
    attr_x = {"addr_type": addr_type,
              "L1_fusion_type": l1_fusion_type}

    input_data_x = tvm.placeholder(fuseshape, name="input_data_x", dtype=inp_dtype, attrs=attr_x)

    with tvm.target.cce():

        res = leaky_relu_compute(input_data_x, y, negative_slope, kernel_name)
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [input_data_x, res],
              "l1_fusion_option": is_l1_depth_fusion}
    tbe.cce_build_code(sch, config)
