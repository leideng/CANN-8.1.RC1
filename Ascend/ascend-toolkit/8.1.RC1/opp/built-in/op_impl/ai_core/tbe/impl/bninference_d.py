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
bninference_d
"""
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_fusion_manager
from impl.util.platform_adapter import build
from impl.util.platform_adapter import auto_schedule
from impl.util.platform_adapter import tbe_register
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=too-many-locals
# 'pylint: disable=locally-disabled,too-few-public-methods,no-init
def _format_check(arg_input):
    """
    Function to check if the data_format is in line with norms.

    Parameters
    ----------
    input: dict
        dict of input
    data_format: str
        format of input data

    Returns

    -------
    None
    """
    format_data = arg_input.get("format")
    excepted_format_list = ["ND", "NC1HWC0", "NCHW", "NHWC", "NC1HWC0_C04"]
    para_check.check_format(format_data, excepted_format_list, param_name="arg_input")


def _check_dims_equal(shape_x, shape, data_format):
    """
    Function to check the dimension C to be equal.

    Parameters
    ----------
    shape_x: list or tuple
        x's data shape
    shape: list or tuple
        data shape of test input
    data_format: str
        format of input data

    Returns
    -------
    None
    """

    if data_format in ("ND", "NCHW", "NHWC"):
        if len(shape_x) == 1:
            index_c = 0
        elif data_format != "NHWC":
            index_c = 1
        else:
            index_c = 3
        if shape_x[index_c] != shape[0]:
            shape_rule = "The dimension value of mean or variance must be equal to C value of input_x"
            error_manager_vector.raise_err_check_params_rules("bninference_d", shape_rule, "x",
                                                              shape_x[index_c])


def _check_shape_dims(shape, data_format):
    """
    Function to check input tensors must be 5D ones.

    Parameters
    ----------
    shape: list or tuple
        data shape of test input
    data_format: str
        format of input data
    is_x: bool
        data to check is input_x or not

    Returns
    -------
    None
    """
    if data_format in ("NC1HWC0", "NDC1HWC0"):
        if len(shape) not in (5, 6):
            error_detail = "bninference only support 5D or 6D Tensor"
            error_manager_vector.raise_err_input_shape_invalid("bninference", "input_x",
                                                               error_detail)


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

    if not (length_scale == 1 and shape_scale[0] == 1):
        if length_x != length_scale:
            error_detail = "The dims of input tensor x and tensor scale should be equal"
            error_manager_vector.raise_err_two_input_shape_invalid("bninference_d", "input_x", "scale",
                                                                   error_detail)

        for i in range(length_scale):
            if shape_scale[i] != shape_x[i] and shape_scale[i] != 1:
                error_detail = "The inputs x and scale could not be broadcast together with mismatched shapes"
                error_manager_vector.raise_err_two_input_shape_invalid("bninference_d", "input_x", "scale",
                                                                       error_detail)


# 'pylint: disable=locally-disabled,too-many-arguments
def _shape_check(shape_x, shape_mean, shape_variance, scale, x_format):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_x: list or tuple
        x's data shape
    shape_scale: list or tuple
        shape_scale's data shape
    shape_offset: list or tuple
        shape_offset's data shape
    shape_mean: list or tuple
        shape_mean's data shape
    shape_variance: list or tuple
        shape_variance's data shape
    is_training: bool
        A bool value to indicate the operation is for training or inference.

    Returns
    -------
    None
    """

    para_check.check_shape(shape_x, param_name="x")
    if x_format in ["NHWC", "NCHW", "ND"]:
        para_check.check_shape(shape_mean, max_rank=1, param_name="mean")
        para_check.check_shape(shape_variance, max_rank=1, param_name="variance")
    _check_shape_dims(shape_x, x_format)

    _check_dims_equal(shape_x, shape_mean, x_format)
    _check_dims_equal(shape_x, shape_variance, x_format)

    if scale is not None:
        shape_scale = scale.get("shape")
        param_scale_check(shape_x, shape_scale)


# 'pylint: disable=invalid-name,redefined-outer-name
# 'pylint: disable=too-many-locals
def _fused_scale_bias_compute(x, mean, variance, scale, bias):
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
    x_shape = shape_util.shape_to_list(x.shape)
    x_dtype = x.dtype
    scale_dtype = scale.dtype
    bias_dtype = bias.dtype
    mean_broadcast = tbe.broadcast(mean, x_shape)
    var_broadcast = tbe.broadcast(variance, x_shape)
    mean_add = tbe.vadd(x, mean_broadcast)
    res_y = tbe.vmul(var_broadcast, mean_add)

    is_cast = False
    product_version = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if product_version not in ("Ascend310", "Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        if x_dtype == "float16":
            is_cast = True
            res_y = tbe.cast_to(res_y, "float32")
        if scale_dtype == "float16":
            scale = tbe.cast_to(scale, "float32")
        if bias_dtype == "float16":
            bias = tbe.cast_to(bias, "float32")

    broad_scale = tbe.broadcast(scale, x_shape)
    broad_bias = tbe.broadcast(bias, x_shape)

    res_tmp = tbe.vmul(res_y, broad_scale)
    res = tbe.vadd(res_tmp, broad_bias)

    if is_cast:
        res = tbe.cast_to(res, x_dtype)
    return res


# 'pylint: disable=invalid-name,redefined-outer-name
def _fused_scale_compute(x, mean, variance, scale):
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
    x_shape = shape_util.shape_to_list(x.shape)
    x_dtype = x.dtype
    scale_dtype = scale.dtype

    broadcast_mean = tbe.broadcast(mean, x_shape)
    broadcast_var = tbe.broadcast(variance, x_shape)
    add_mean = tbe.vadd(x, broadcast_mean)
    res_tmp = tbe.vmul(broadcast_var, add_mean)

    is_cast = False
    product_version = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")

    if product_version not in ("Ascend310", "Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        if x_dtype == "float16":
            is_cast = True
            res_tmp = tbe.cast_to(res_tmp, 'float32')
        if scale_dtype == "float16":
            scale = tbe.cast_to(scale, 'float32')

    broad_scale = tbe.broadcast(scale, x_shape)

    res = tbe.vmul(res_tmp, broad_scale)

    if is_cast:
        res = tbe.cast_to(res, x_dtype)
    return res


# 'pylint: disable=invalid-name,redefined-outer-name
def _fused_compute(x, mean, variance):
    """
    Parameters
    ----------
    x: dict
        contains x data.
    mean: dict
        contains mean data.Must be 1D if input "x" Specifies the mean used for inference.
    variance: dict
        contains variance data.Must be 1D if input "x" Specifies the variance used for inference.

    Returns
    -------
    res: dict
        the result of compute
    """
    x_shape = shape_util.shape_to_list(x.shape)
    broadcast_mean = tbe.broadcast(mean, x_shape)
    broadcast_var = tbe.broadcast(variance, x_shape)
    add_mean = tbe.vadd(x, broadcast_mean)
    res = tbe.vmul(broadcast_var, add_mean)
    return res


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals,invalid-name,protected-access
@register_operator_compute("bninference_d", op_mode="static", support_fusion=True)
def bninference_d_compute(x, mean, variance, scale, bias, y,
                          momentum, epsilon, use_global_stats, mode):
    """
    Parameters
    ----------
    x: dict
        contains x data. A 4D or 5D Tensor of type float16 or float32.
    mean: dict
        contains mean data.Must be 1D if input "x" Specifies the mean used for inference.
    variance: dict
        contains variance data.Must be 1D if input "x" Specifies the variance used for inference.
    scale: dict
        no use in caffe batchnorm inference
    bias: dict
        no use in caffe batchnorm inference
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    momentum: float
        a float number of the variance and mean's scale factor
    epsilon: float
        a small float number added to the variance of x to avoid dividing by zero. Defaults to "0.00001".
    use_global_stats: bool
        means the caffe inference model, only can be True.
    mode: int
        an optional attr, no use

    Returns
    -------
    res: TVM tensor list
        the result of compute
    """

    fuse_y = y
    if y is None:
        fuse_y = {"addr_type": 0}

    # if l1 fusion x format must 5hd
    l1_fusion_type = x.op.attrs["L1_fusion_type"].value if "L1_fusion_type" in x.op.attrs else -1
    if l1_fusion_type != -1 and y.get("format").upper() != 'NC1HWC0':
        shape_rule = "when L1_FUSION is enabled for the bninference operator, the input format must be 5HD"
        error_manager_vector.raise_err_check_params_rules("bninference_d", shape_rule, "x",
                                                          y.get("format").upper())

    params_fusion = get_fusion_params(x, mean, variance, scale, bias, fuse_y)

    if scale is not None and bias is not None:
        res = _fused_scale_bias_compute(x, mean, variance, scale, bias)
    elif scale is not None and bias is None:
        res = _fused_scale_compute(x, mean, variance, scale)
    else:
        res = _fused_compute(x, mean, variance)
    res.op.attrs["ele_fusion_params"] = params_fusion
    res.op.attrs["L1_fusion_type"] = params_fusion["l1_fusion_type"]

    build_cfg = {
        'read_write_bank_conflict': True
    }
    tbe_register.set_fusion_buildcfg("bninference_d", build_cfg)

    return res


def _dtype_scale_offset_check(x, mean, variance, scale, offect):
    dtype_x = x.get("dtype")
    dtype_mean = mean.get("dtype")
    dtype_variance = variance.get("dtype")
    product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")

    if product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        checklist = ["float16"]
    else:
        checklist = ["float32", "float16"]
    para_check.check_dtype(dtype_mean.lower(), checklist, param_name="mean")
    para_check.check_dtype(dtype_x.lower(), checklist, param_name="x")
    para_check.check_dtype(dtype_variance.lower(), checklist, param_name="variance")

    if scale is not None:
        dtype_scale = scale.get("dtype")
        para_check.check_dtype(dtype_scale.lower(), checklist, param_name="scale")
    if offect is not None and bool(offect):
        dtype_offect = offect.get("dtype")
        para_check.check_dtype(dtype_offect.lower(), checklist, param_name="offect")


def _dtype_check(x, mean, variance):
    dtype_x = x.get("dtype")
    dtype_mean = mean.get("dtype")
    dtype_variance = variance.get("dtype")
    product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")

    if product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        checklist = ["float16"]
    else:
        checklist = ["float32", "float16"]
    para_check.check_dtype(dtype_mean.lower(), checklist, param_name="mean")
    para_check.check_dtype(dtype_x.lower(), checklist, param_name="x")
    para_check.check_dtype(dtype_variance.lower(), checklist, param_name="variance")


def para_shape_scale_offset_check(x, mean, variance, scale, offect, format_x):
    """
    :param x:input tensor
    :param mean:mean tensor
    :param variance: var tensor
    :param format_x: format tensor
    :return: None
    """
    mean_shape = mean.get("shape")
    variance_shape = variance.get("shape")
    shape_x = x.get("shape")
    para_check.check_shape(mean_shape, param_name="mean")
    para_check.check_shape(variance_shape, param_name="variance")

    if scale is not None:
        shape_scale = scale.get("shape")
        para_check.check_shape(shape_scale, param_name="scale")
    if offect is not None and bool(offect):
        shape_offect = offect.get("shape")
        para_check.check_shape(shape_offect, param_name="offect")

    _shape_check(shape_x, mean_shape, variance_shape, scale, format_x)


def para_shape_check(x, mean, variance, scale, format_x):
    """
    :param x:input tensor
    :param mean:mean tensor
    :param variance: var tensor
    :param format_x: format tensor
    :return: None
    """
    shape_mean = mean.get("shape")
    shape_variance = variance.get("shape")
    shape_x = x.get("shape")
    para_check.check_shape(shape_mean, param_name="mean")
    para_check.check_shape(shape_variance, param_name="variance")
    _shape_check(shape_x, shape_mean, shape_variance, scale, format_x)


# 'pylint: disable=redefined-argument-from-local
def get_fusion_params(x, mean, variance, scale, bias, y):
    """
    Get L1 fusion_params
    Parameters
    ----------
    x : tensor of input data
    y : dict of output data
    x_tensor_num: input tensor num
    Returns
    -------
    fusion_params
    """
    # 0: L1 depth fusion, 1: L1 width fusion, -1: no L1 fusion
    is_l1_depth_fusion = False

    input_tensor = [x, mean, variance, scale, bias]
    for x in input_tensor:
        if x is not None:
            l1_fusion_type = -1
            if tbe_fusion_manager.get_build_cfg() != "disable":
                l1_fusion_type = x.op.attrs["L1_fusion_type"].value if "L1_fusion_type" in x.op.attrs else -1
                if l1_fusion_type == 1:
                    error_manager_vector.raise_err_specific_reson("bninference",
                        "bninference does not support l1 width fusion")
            is_l1_depth_fusion = (l1_fusion_type == 0) or is_l1_depth_fusion

    l1_fusion_type = 0 if is_l1_depth_fusion else -1

    out_l1_flag = False
    if y is not None:
        out_l1_flag = y.get("addr_type", 0) == 1

    fusion_params = {"l1_fusion_type": l1_fusion_type,
                     "out_l1_flag": out_l1_flag}
    return fusion_params


def para_scale_bias_check(x, mean, variance, scale, offect, use_global_stats, kernel_name):
    """
    :param x:input tensor
    :param mean: mean tensor
    :param variance: var tensor
    :param use_global_stats: inference type
    :param kernel_name: kernel_name
    :return: none
    """
    format_x = x.get("format")
    _format_check(x)
    _dtype_scale_offset_check(x, mean, variance, scale, offect)
    if not use_global_stats:
        error_manager_vector.raise_err_input_value_invalid("bninference", "use_global_stats",
                                                           'True', str(use_global_stats))
    para_shape_scale_offset_check(x, mean, variance, scale, offect, format_x)


def _para_check(x, mean, variance, scale, use_global_stats, kernel_name):
    """
    :param x:input tensor
    :param mean: mean tensor
    :param variance: var tensor
    :param use_global_stats: inference type
    :param kernel_name: kernel_name
    :return: none
    """
    format_x = x.get("format")
    _format_check(x)
    _dtype_check(x, mean, variance)
    if not use_global_stats:
        error_manager_vector.raise_err_input_value_invalid("bninference", "use_global_stats",
                                                           'True', str(use_global_stats))
    para_shape_check(x, mean, variance, scale, format_x)


def get_param_scale_shape(shape_x, shape_scale, format_x):
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
    new shape
    """

    length_x = len(shape_x)
    length_scale = len(shape_scale)

    if length_scale == 1 and shape_scale[0] == 1:
        shape = [1] * length_x
    else:
        shape = list(shape_scale)
    if format_x == "NDC1HWC0":
        shape = [shape_scale[0] * shape_scale[1], shape_scale[2], shape_scale[3], shape_scale[4], shape_scale[5]]

    return shape


# 'pylint: disable=too-many-branches,too-many-statements
def gen_tensor(x, mean, variance, scale, offect):
    """
    :param x:x tensor
    :param mean: mean tensor
    :param variance:var tensor
    :return:
    x_input:x
    mean_input:mean
    var_input:var
    scale:scale,not use
    b:not use
    """
    shape_x = x.get("shape")
    format_x = x.get("format")
    dtype_x = x.get("dtype")
    if format_x in ("ND", "NCHW"):
        if len(shape_x) == 1:
            index_c = 0
        else:
            index_c = 1
    elif format_x == "NHWC":
        if len(shape_x) == 1:
            index_c = 0
        else:
            index_c = 3
    else:
        c1 = shape_x[1]
        c0 = shape_x[4]
    shape_mean = mean.get("shape")
    shape_variance = variance.get("shape")

    if format_x in ("ND", "NCHW", "NHWC"):
        shape_mean = [1] * len(shape_x[:index_c]) + list(shape_mean) \
                     + [1] * len(shape_x[index_c + 1:])
        shape_variance = [1] * len(shape_x[:index_c]) + list(shape_variance) \
                         + [1] * len(shape_x[index_c + 1:])
    else:
        shape_mean = [1, c1, 1, 1, c0]
        shape_variance = [1, c1, 1, 1, c0]

    shape_scale = {}
    shape_offect = {}
    if scale is not None:
        shape_scale = scale.get("shape")
    if offect is not None and bool(offect):
        shape_offect = offect.get("shape")

    is_l1_depth_fusion = False

    attr_x, l1_fusion_type = get_l1_paras(x)
    is_l1_depth_fusion = (l1_fusion_type == 0) or is_l1_depth_fusion
    x_input = tvm.placeholder(shape_x, name="x", dtype=dtype_x.lower(), attrs=attr_x)
    attr_mean, l1_fusion_type = get_l1_paras(mean)
    is_l1_depth_fusion = (l1_fusion_type == 0) or is_l1_depth_fusion
    mean_input = tvm.placeholder(shape_mean, name="mean",
                                 dtype=dtype_x.lower(), attrs=attr_mean)
    attr_variance, l1_fusion_type = get_l1_paras(variance)
    is_l1_depth_fusion = (l1_fusion_type == 0) or is_l1_depth_fusion
    variance_input = tvm.placeholder(shape_variance, name="variance",
                                     dtype=dtype_x.lower(), attrs=attr_variance)

    scale_input = None
    offset_input = None
    if len(shape_scale) > 0:
        dtype_scale = scale.get("dtype")
        shape_scale_new = get_param_scale_shape(shape_x, shape_scale, format_x)
        attr_scale, l1_fusion_type = get_l1_paras(scale)
        is_l1_depth_fusion = (l1_fusion_type == 0) or is_l1_depth_fusion
        scale_input = tvm.placeholder(shape_scale_new, name="scale_input",
                                      dtype=dtype_scale.lower(), attrs=attr_scale)
        if len(shape_offect) > 0:
            dtype_offect = offect.get("dtype")
            shape_offect_new = shape_scale_new
            attr_offect, l1_fusion_type = get_l1_paras(offect)
            is_l1_depth_fusion = (l1_fusion_type == 0) or is_l1_depth_fusion
            offset_input = tvm.placeholder(shape_offect_new, name="offset_input",
                                           dtype=dtype_offect.lower(), attrs=attr_offect)

    input_list = [x_input, mean_input, variance_input, scale_input, offset_input]
    return input_list, is_l1_depth_fusion


def get_l1_paras(x):
    """
    get_l1_paras
    """
    l1_fusion_type = -1
    if tbe_fusion_manager.get_build_cfg() != "disable":
        l1_fusion_type = x.get('L1_fusion_type', -1)
        if l1_fusion_type == 1:
            error_manager_vector.raise_err_specific_reson("bninference",
                                        "bninference does not support l1 width fusion")
    addr_type = x.get("addr_type", 0)
    attr_x = {"addr_type": addr_type,
              "L1_fusion_type": l1_fusion_type}
    return attr_x, l1_fusion_type


# 'pylint: disable=locally-disabled,no-member
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_FLOAT, para_check.REQUIRED_ATTR_FLOAT,
                            para_check.REQUIRED_ATTR_BOOL, para_check.REQUIRED_ATTR_INT,
                            para_check.KERNEL_NAME)
def bninference_d(x, mean, variance, scale, offect, y, momentum, epsilon,
                  use_global_stats, mode, kernel_name="bninference"):
    """

    Parameters
    ----------
    x: dict
        contains x data. A 4D or 5D Tensor of type float16 or float32.
    mean: dict
        contains mean data.Must be 1D if input "x" Specifies the mean used for inference.
    variance: dict
        contains variance data.Must be 1D if input "x" Specifies the variance used for inference.
    scale: dict
        no use in caffe batchnorm inference
    bias: dict
        no use in caffe batchnorm inference
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    momentum: float
        a float number of the variance and mean's scale factor
    epsilon: float
        a small float number added to the variance of x to avoid dividing by zero. Defaults to "0.00001".
    use_global_stats: bool
        means the caffe inference model, only can be True.
    mode: int
        an optional attr, no use
    kernel_name: str
        kernel name

    Returns
    -------
    None
    """
    if offect is not None or scale is not None:
        para_scale_bias_check(x, mean, variance, scale, offect, use_global_stats, kernel_name)
    else:
        _para_check(x, mean, variance, scale, use_global_stats, kernel_name)
    list_input, l1_depth_fusion = gen_tensor(x, mean, variance, scale, offect)
    x_input, mean_input, variance_input, scale_input, offset_input = list_input
    res = bninference_d_compute(x_input, mean_input,
                                variance_input, scale_input, offset_input,
                                y, momentum, epsilon,
                                use_global_stats, mode)
    with tvm.target.cce():
        sch = auto_schedule(res)
    if offect is None and scale is None:
        list_tensor = [x_input, mean_input, variance_input, res]
    elif offect is None and scale is not None:
        list_tensor = [x_input, mean_input, variance_input, scale_input, res]
    else:
        list_tensor = [x_input, mean_input, variance_input, scale_input, offset_input, res]
    config = {"name": kernel_name,
              "tensor_list": list_tensor,
              "l1_fusion_option": l1_depth_fusion}
    build(sch, config)
