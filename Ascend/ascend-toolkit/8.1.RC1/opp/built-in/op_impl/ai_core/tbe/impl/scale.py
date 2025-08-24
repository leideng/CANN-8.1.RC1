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
scale
"""
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import shape_util
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import PlatformApi
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.util.util_select_op_base import get_op_cal_info

NONETYPE = type(None)


# 'pylint: disable=too-many-arguments,unused-argument,invalid-name,redefined-outer-name
def check_param_range(param_name, min_value, max_value, real_value, op_name='ssd_detection_output'):
    """
    check_param_range
    """
    error_manager_vector.raise_err_input_param_range_invalid("scale", param_name, str(min_value),
                                                             str(max_value), str(real_value))


# 'pylint: disable=too-many-arguments,unused-argument,invalid-name,redefined-outer-name
# 'pylint: disable=too-many-boolean-expressions,too-many-locals,unused-variable
def op_select_format(x, scale, bias, y, axis=1, num_axes=1, scale_from_blob=True,
                     kernel_name="scale"):
    """
    1. when length of input x's ori_shape is 4. The Op
    Scale can support NC1HWC0.
    > for example:
    > x : Tensor of (shape=(16, 16, 16, 16), "NCHW")
    > scale : Tensor of (shape=(16, 16, 16, 16), "NCHW")
    > the Op Scale can process with NC1HWC0:
    > x : Tensor of (shape=(16, 1, 16, 16, 16) ,"NC1HWC0")
    > scale : Tensor of (shape=(16, 1, 16, 16, 16) ,"NC1HWC0")

    2. In other scenes, the Op Select can support ND.
    > for example:
    > x : Tensor of (shape=(2), "ND")
    > scale : Tensor of (shape=(2), "ND")
    """
    shape_scale_ori = scale.get("ori_shape")
    shape_scale = scale.get("ori_shape")
    shape_x_ori = x.get("ori_shape")

    length_scale = len(shape_scale)
    length_x_ori = len(shape_x_ori)

    if (length_scale == 1 and shape_scale[0] == 1) or length_scale == 0:
        format_scale = "ND,ND,ND,ND"
        format_bias = "ND,ND,ND,ND"
        format_scale_hisi = "ND,ND"
        format_bias_hisi = "ND,ND"
    else:
        format_scale = "NC1HWC0,NC1HWC0,ND,ND"
        format_bias = "NC1HWC0,NC1HWC0,ND,ND"
        format_scale_hisi = "NC1HWC0,ND"
        format_bias_hisi = "NC1HWC0,ND"

    product_version = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if length_x_ori == 4:
        # NC1HWC0+ND
        if product_version in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16,float16",
                               format="NC1HWC0,ND")
            input1 = gen_param(classify="input1", name="scale",
                               datatype="float16,float16",
                               format=format_scale_hisi)
            input2 = gen_param(classify="input2", name="bias",
                               datatype="float16,float16",
                               format=format_bias_hisi)
            output0 = gen_param(classify="output0", name="y",
                                datatype="float16,float16",
                                format="NC1HWC0,ND")
        else:
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16,float,float16,float",
                               format="NC1HWC0,NC1HWC0,ND,ND")
            input1 = gen_param(classify="input1", name="scale",
                               datatype="float16,float,float16,float",
                               format=format_scale)
            input2 = gen_param(classify="input2", name="bias",
                               datatype="float16,float,float16,float",
                               format=format_bias)
            output0 = gen_param(classify="output0", name="y",
                                datatype="float16,float,float16,float",
                                format="NC1HWC0,NC1HWC0,ND,ND")
    else:
        # ND+ND
        if product_version in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16",
                               format="ND")
            input1 = gen_param(classify="input1", name="scale",
                               datatype="float16",
                               format="ND")
            input2 = gen_param(classify="input2", name="bias",
                               datatype="float16",
                               format="ND")
            output0 = gen_param(classify="output0", name="y",
                                datatype="float16",
                                format="ND")
        else:
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16,float",
                               format="ND,ND")
            input1 = gen_param(classify="input1", name="scale",
                               datatype="float16,float",
                               format="ND,ND")
            input2 = gen_param(classify="input2", name="bias",
                               datatype="float16,float",
                               format="ND,ND")
            output0 = gen_param(classify="output0", name="y",
                                datatype="float16,float",
                                format="ND,ND")

    param_list = [input0, input1, input2, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def get_op_support_info(x, scale, bias, y, axis=1, num_axes=1, scale_from_blob=True, kernel_name="scale"):
    """
    get split info
    """
    ori_shape = x.get("ori_shape")
    if axis < 0:
        axis = axis + len(ori_shape)
    dim_x = len(x.get("shape"))
    format_x = x.get("format").upper()
    not_cut_dim = []
    if format_x == "NC1HWC0":
        not_cut_dim = [1, 4]

    if format_x in ("ND", "NC1HWC0"):
        axis_split_list = []
        for i in range(dim_x):
            if i < axis and i not in not_cut_dim:
                split = [SplitInput([0, [i], [-1], [-1]]),
                         SplitOutput([0, [i]])]
                axis_split_list.append(split)
    else:
        axis_split_list = None
    axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_list, axis_reduce_list)
    return op_cal_info_in_json


def param_scale_check(shape_x, shape_scale):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_x : list or tuple.
    shape of x.
    shape_scale : list or tuple.
    shape of scale.

    Returns
    -------
    None
    """

    length_x = len(shape_x)
    length_scale = len(shape_scale)

    if not(length_scale == 1 and shape_scale[0] == 1):
        if length_x != length_scale:
            error_manager_vector.raise_err_specific_reson("scale", "the dims of input tensor x and tensor \
                                                          scale should be equal, but actually are \
                                                          [{}] and [{}]"
                                                          .format(str(length_x), str(length_scale)))

        for i in range(length_scale):
            if shape_scale[i] != shape_x[i] and shape_scale[i] != 1:
                error_manager_vector.raise_err_specific_reson("scale", "the inputs[{}][{}] could \
                                                                       not be broadcast \
                                                              together with shapes[{}][{}]."
                                                              .format("x", "scale",
                                                              str(shape_x), str(shape_scale)))


def get_param_scale_shape(shape_x, shape_scale):
    """
    Function to calculate the shape of scale.

    Parameters
    ----------
    shape_x : list or tuple.
    shape of x.
    shape_scale : list or tuple.
    shape of scale.

    Returns
    -------
    None
    """

    length_x = len(shape_x)
    length_scale = len(shape_scale)

    if length_scale == 1 and shape_scale[0] == 1:
        shape = [1] * length_x
    else:
        shape = list(shape_scale)

    return shape


def _check_dtype(input_dtype, name):
    """
    Function to check dtype of input data.

    Parameters
    ----------

    input_dtype: str
        dtype of input data
    Returns
    -------
    None
    """

    product_version = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if product_version in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        if input_dtype == "float32":
            rule_desc = "float32 is not support in HISI"
            error_manager_vector.raise_err_check_params_rules("scale", rule_desc, "input_dtype",
                                                            input_dtype)
        para_check.check_dtype(input_dtype, ["float16"], param_name=name)
    else:
        para_check.check_dtype(input_dtype, ["float16", "float32"], param_name=name)


# 'pylint: disable=too-many-branches
def _check_scale_shape_axis(shape_x, shape_scale, axis, num_axes, scale_from_blob):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_x: list or tuple
        x's data shape
    shape_scale: list or tuple
        scale's data shape
    axis : int
        A int num indicates shape of scale when scale is from bottom.
    num_axes:
        A int num indicates shape of scale when scale is from blob.
    scale_from_blob:
        A bool value indicates scale is from blob or bottom.

    Returns
    -------
    None
    """

    length_x = len(shape_x)
    length_scale = len(shape_scale)
    error_info = {}

    if (axis >= length_x) or (axis < (-length_x)):
        error_manager_vector.raise_err_input_param_range_invalid("scale", "axis",
                                                                 str(-length_x), str(length_x - 1), str(axis))

    if num_axes < -1:
        error_manager_vector.raise_err_input_value_invalid("scale", "num_axes",
                                                           "non-negative or -1", str(num_axes))

    if axis < 0:
        axis_ = length_x + axis
    else:
        axis_ = axis

    # from blob
    if scale_from_blob:
        if num_axes == -1:
            scale_num = length_x - axis_
            if length_scale != scale_num:
                error_manager_vector.raise_err_specific_reson("scale",
                                                              "length_scale[{}] and \
                                                               scale_num[{}] must be equal "
                                                              .format(length_scale, scale_num))
            for i in range(scale_num):
                if shape_x[axis_ + i] != shape_scale[i]:
                    error_manager_vector.raise_err_inputs_shape_not_equal("scale", "shape_x", "shape_scale",
                                                                          str(shape_x[axis_ + i]),
                                                                          str(shape_scale[i]), str(shape_scale[i]))
        if num_axes == 0:
            if length_scale != 1 or shape_scale[0] != 1:
                error_manager_vector.raise_err_specific_reson("scale", "scale must be a scalar!")
        if num_axes > 0:
            num_axis = axis_ + num_axes
            if num_axis > length_x:
                error_manager_vector.raise_err_specific_reson("scale", "scale shape \
                                                                       extends x shape when applied")
            if length_scale != num_axes:
                error_manager_vector.raise_err_specific_reson("scale",
                                                              "length_scale[{}] and num_axes[{}] must be equal"
                                                              .format(length_scale, num_axes))
            for i in range(num_axes):
                if shape_x[axis_ + i] != shape_scale[i]:
                    error_manager_vector.raise_err_inputs_shape_not_equal("scale", "shape_x", "shape_scale",
                                                                          str(shape_x[axis_ + i]),
                                                                          str(shape_scale[i]), str(shape_scale[i]))

    # from bottom
    if not scale_from_blob:
        if not(length_scale == 1 and shape_scale[0] == 1):
            scale_num = axis_ + length_scale
            if scale_num > length_x:
                error_manager_vector.raise_err_specific_reson("scale", "scale shape extends \
                                                                       x shape when applied")
            for i in range(length_scale):
                if shape_x[axis_ + i] != shape_scale[i]:
                    error_manager_vector.raise_err_specific_reson("scale",
                                                                  "Dimensions shape_x[{}] and \
                                                                  shape_scale[{}] must be equal"
                                                                  .format(shape_x[axis_ + i], shape_scale[i]))


def get_scale_shape(shape_x, shape_scale, axis_, num_axes, scale_from_blob):
    """
    Function to calculate shape of scale.

    Parameters
    ----------
    shape_x: list or tuple
    x's data shape
    shape_scale: list or tuple
    scale's data shape
    axis_ : int
    A int num indicates shape of scale when scale is from bottom.
    num_axes:
    A int num indicates shape of scale when scale is from blob.
    scale_from_blob:
    A bool value indicates scale is from blob or bottom.

    Returns
    -------
    shape: list or tuple
    the shape of scale
    """

    length_x = len(shape_x)
    length_scale = len(shape_scale)
    if scale_from_blob:
        if num_axes == -1:
            shape_left = [1] * axis_
            shape = shape_left + list(shape_scale)
        elif num_axes == 0:
            shape = [1] * length_x
        else:
            left_length = length_x - num_axes - axis_
            shape_left = [1] * axis_
            shape_right = [1] * left_length
            shape = shape_left + list(shape_scale) + shape_right
    else:
        if length_scale == 1 and shape_scale[0] == 1:
            shape = [1] * length_x
        else:
            left_length = length_x - length_scale - axis_
            shape_left = [1] * axis_
            shape_right = [1] * left_length
            shape = shape_left + list(shape_scale) + shape_right

    return shape


# 'pylint: disable=invalid-name,redefined-outer-name
def _fused_scale_compute(x, scale):
    """
    algorithm: Scale
    y = scale*x

    Parameters
    ----------
    x : TVM tensor
        contains x data
    scale : TVM tensor
        contains scale data

    Returns
    -------
    res: TVM tensor list
        the result of scale compute
    """

    dtype_x = x.dtype
    dtype_scale = scale.dtype

    is_cast = False
    product_version = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")

    if product_version not in ("Ascend310", "Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        if dtype_x == "float16":
            is_cast = True
            x = tbe.cast_to(x, 'float32')
        if dtype_scale == "float16":
            scale = tbe.cast_to(scale, 'float32')

    shape_x = shape_util.shape_to_list(x.shape)
    scale_broad = tbe.broadcast(scale, shape_x)

    res = tbe.vmul(x, scale_broad)

    if is_cast:
        res = tbe.cast_to(res, dtype_x)

    return res


# 'pylint: disable=invalid-name,redefined-outer-name
def _fused_scale_bias_compute(x, scale, bias):
    """
    algorithm: Scale
    y = scale*x + bias

    Parameters
    ----------
    x : TVM tensor
        contains x data
    scale : TVM tensor
        contains scale data
    bias : TVM tensor
        contains bias data
    Returns
    -------
    res: TVM tensor list
        the result of scale compute
    """

    dtype_x = x.dtype
    dtype_scale = scale.dtype
    dtype_bias = bias.dtype

    is_cast = False
    product_version = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")

    if product_version not in ("Ascend310", "Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        if dtype_x == "float16":
            is_cast = True
            x = tbe.cast_to(x, 'float32')
        if dtype_scale == "float16":
            scale = tbe.cast_to(scale, 'float32')
        if dtype_bias == "float16":
            bias = tbe.cast_to(bias, 'float32')

    shape_x = shape_util.shape_to_list(x.shape)

    scale_broad = tbe.broadcast(scale, shape_x)
    bias_broad = tbe.broadcast(bias, shape_x)

    res_tmp = tbe.vmul(x, scale_broad)
    res = tbe.vadd(res_tmp, bias_broad)

    if is_cast:
        res = tbe.cast_to(res, dtype_x)

    return res


# 'pylint: disable=too-many-arguments,unused-argument,invalid-name
@register_operator_compute("scale", op_mode="static", support_fusion=True)
def scale_compute(x, scale, bias, y, axis, num_axes, scale_from_blob,
                  kernel_name="scale"):
    """
    algorithm: Scale
    y = scale*x + bias

    Parameters
    ----------
    x : TVM tensor
    contains x data
    scale : TVM tensor
    contains scale data
    bias : TVM tensor
    contains bias data
    y : dict
    dict of output,
    A Tensor for output, should be same shape and type as x.
    axis : int
    A int num indicates shape of scale when scale is from bottom.
    num_axes: int
    A int num indicates shape of scale when scale is from blob.
    scale_from_blob:
    A bool value indicates scale is from blob or bottom.
    kernel_name : str
    kernel name, default value is "scale"

    Returns
    -------
    res: TVM tensor list
    the result of scale compute
    """
    tmp_y = {}
    tmp_y["addr_type"] = 0
    tmp_y["valid_shape"] = []
    tmp_y["slice_offset"] = []
    fuse_y = tmp_y if y is None else y

    res = None
    if bias is not None:
        res = _fused_scale_bias_compute(x, scale, bias)
    else:
        res = _fused_scale_compute(x, scale)

    return res


# 'pylint: disable=too-many-locals,no-member,invalid-name,too-many-statements,line-too-long
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def scale(x, scale, bias, y, axis=1, num_axes=1, scale_from_blob=True,
          kernel_name="scale"):
    """
    algorithm: Scale
    y = scale*x + bias

    Parameters
    ----------
    x : dict
    dict of input, A Tensor for input data.
    scale : dict
    dict of scale,
    A Tensor for scaling factor, to scale the input data.
    bias : dict
    dict of bias,
    A Tensor for bias, to shift to the input data.
    y : dict
    dict of output,
    A Tensor for y, should be same shape and type as x.
    axis : int
    A int num indicates shape of scale when scale is from bottom.
    num_axes: int
    A int num indicates shape of scale when scale is from blob.
    scale_from_blob:
    A bool value indicates scale is from blob or bottom.
    kernel_name : str
    kernel name, default value is "scale"

    Returns
    -------
    None
    """

    shape_x = x.get("shape")
    dtype_x = x.get("dtype")
    para_check.check_shape(shape_x, param_name="input_x")
    _check_dtype(dtype_x.lower(), "input_x")

    shape_scale = scale.get("shape")
    dtype_scale = scale.get("dtype")
    para_check.check_shape(shape_scale, param_name="input_scale")
    _check_dtype(dtype_scale.lower(), "input_scale")

    shape_bias = ()
    if bias is not None and bool(bias):
        shape_bias = bias.get("shape")
        dtype_bias = bias.get("dtype")
        para_check.check_shape(shape_bias, param_name="input_bias")
        _check_dtype(dtype_bias.lower(), "input_bias")

    shape_x_ori = x.get("ori_shape")
    length_x_ori = len(shape_x_ori)

    shape_scale_new = []
    shape_bias_new = []

    if length_x_ori == 4:
        param_scale_check(shape_x, shape_scale)
        shape_scale_new = get_param_scale_shape(shape_x, shape_scale)
        if len(shape_bias) > 0:
            shape_bias_new = shape_scale_new
    else:
        _check_scale_shape_axis(shape_x, shape_scale, axis, num_axes, scale_from_blob)

        length_x = len(shape_x)
        if axis < 0:
            axis_ = length_x + axis
        else:
            axis_ = axis

        shape_scale_new = get_scale_shape(shape_x, shape_scale, axis_, num_axes, scale_from_blob)
        if len(shape_bias) > 0:
            shape_bias_new = shape_scale_new

    input_list = [x, scale, bias]
    input_shape_list = [shape_x, shape_scale_new, shape_bias_new]
    name_list = ["x", "scale", "bias"]
    input_tensor_list = []
    is_l1_depth_fusion = False
    for input_, input_shape, name_ in zip(input_list, input_shape_list, name_list):
        if len(input_shape) > 0:
            l1_fusion_type = -1
            if PlatformApi.fusion_manager.get_build_cfg() != "disable":
                l1_fusion_type = input_.get("L1_fusion_type", -1)
                if l1_fusion_type == 1:
                    error_manager_vector.raise_err_specific_reson("scale",
                                    "Scale does not support l1 width fusion")
            is_l1_depth_fusion = (l1_fusion_type == 0) or is_l1_depth_fusion
            dtype = input_.get("dtype")
            addr_type = input_.get("addr_type", 0)
            valid_shape = input_.get("valid_shape", [])
            slice_offset = input_.get("slice_offset", [])
            attr_x = {"addr_type": addr_type,
                      "valid_shape": valid_shape,
                      "slice_offset": slice_offset,
                      "L1_fusion_type": l1_fusion_type}
            input_tensor = tvm.placeholder(input_shape, name=name_,
                                           dtype=dtype, attrs=attr_x)
            input_tensor_list.append(input_tensor)

    if len(shape_bias) == 0:
        input_tensor_list.append(None)

    x_input, scale_input, bias_input = input_tensor_list
    res = scale_compute(x_input, scale_input, bias_input, y,
                        axis, num_axes, scale_from_blob, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    tensor_list = (x_input, scale_input, res)
    if len(shape_bias) > 0:
        tensor_list = (x_input, scale_input, bias_input, res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list,
              "l1_fusion_option": is_l1_depth_fusion}
    tbe.cce_build_code(sch, config)
