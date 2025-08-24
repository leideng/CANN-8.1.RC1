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
conv2d
"""
from __future__ import absolute_import
import math
import json
from tbe import tvm
from tbe.dsl import auto_schedule
from tbe.dsl import build
from tbe.dsl.compute.conv_compute import conv
from tbe.common.register import register_op_compute
from tbe.common.platform import CUBE_MKN
from tbe.common.platform.platform_info import get_soc_spec
from tbe.common.utils import para_check
from tbe.common.utils import shape_util
from tbe.common.utils.errormgr import error_manager_cube as err_man
from .util import util_select_op_base
from .util import util_conv2d

DYNAMIC_VALUE = -1

@para_check.check_input_type(dict, dict, (dict, para_check.NONE_TYPE), (dict, para_check.NONE_TYPE), dict,
                             (tuple, list), (tuple, list), (tuple, list), int, str, int, str)
def check_supported(inputs, weights, bias, offset_w, outputs, strides,
                    pads, dilations, groups=1, data_format='NHWC',
                    offset_x=0, kernel_name="conv2d"):
    """
    1.The following are the supported data types and data formats:

    | Tensor    | x       | filter  | bias    | y       |
    | :-------: | :-----: | :-----: | :-----: | :-----: |
    | Data Type | float16 | float16 | float16 | float16 |
    |           | float32 | float32 | float32 | float32 |
    |           | int8    | int8    | int32   | int32   |
    | Format    | NCHW    | NCHW    | ND      | NCHW    |
    |           | NHWC    | HWCN    |         | NHWC    |

    Note: for float32 type, the actual calculation on the chip is based on float16.

    2.The following value range restrictions must be met:

    | Name             | Field    | Scope       |
    | :--------------: | :------: | :---------: |
    | Input Image Size | H        | [1, 100000] |
    |                  | W        | [1, 4096]   |
    | Filter Size      | H        | [1, 255]    |
    |                  | W        | [1, 255]    |
    | Stride           | H        | [1, 63]     |
    |                  | W        | [1, 63]     |
    | Padding          | Top      | [0, 255]    |
    |                  | Bottom   | [0, 255]    |
    |                  | Left     | [0, 255]    |
    |                  | Right    | [0, 255]    |
    | Dilation         | H        | [1, 255]    |
    |                  | W        | [1, 255]    |
    | Offset_x         | -        | [-128, 127] |

    Note: the W dimension of the input image supports cases exceeding 4096, but it may cause
    compilation errors.
    """
    try:
        check_list = [inputs, weights, strides, pads, dilations, outputs, data_format]
        return_list = util_conv2d.calc_para_from_dict(*check_list)
        offset_w_dtype = "int32"
        valid = isinstance(offset_w, dict) and isinstance(offset_w.get("dtype"), str)
        if valid:
            offset_w_dtype = offset_w.get("dtype")
        check_list = [*return_list[:6], inputs["dtype"], weights["dtype"], outputs["dtype"],
                      offset_w_dtype, (bias is None), kernel_name, *return_list[6:9], groups]
        return_list = util_conv2d.conv_layer_cce_para_check(*check_list)
    except RuntimeError as e:
        reason = e.args[1]
        return False, reason
    return True, ""


def op_select_format(inputs, weights, bias, offset_w, outputs, strides,
                     pads, dilations, groups=1, data_format='NHWC',
                     offset_x=0, kernel_name="conv2d"):
    r"""
    1.When input x type is float or float16, op select supports the following specification:

    | Tensor    | x        | filter     | bias    | y       |
    | :-------: | :------: | :--------: | :-----: | :-----: |
    | Data Type | float16  | float16    | float16 | float16 |
    | Format    | NC1HWC0  | FRACTAL_Z  |  ND     | NC1HWC0 |

    Note: C0 = 32 / sizeof(data type), C1 = ceil(in_channels / C0), for float16 type, C0 = 16

    2.When input x type is int8, op select supports the following specification:

    | Tensor    | x        | filter     | bias    | y       |
    | :-------: | :------: | :--------: | :-----: | :-----: |
    | Data Type | int8     | int8       | int32   | int32   |
    | Format    | NC1HWC0  | FRACTAL_Z  |  ND     | NC1HWC0 |

    Note: for int8 type, C0 = 16, for int32 type, C0 = 8

    3.When in_channels <=4, filter_height > 1, filter_width > 1, op select supports the additional
    FRACTAL_Z_C04 format of input filter:

    | Tensor    | x            | filter         | bias    | y       |
    | :-------: | :----------: | :------------: | :-----: | :-----: |
    | Format    | NC1HWC0      | FRACTAL_Z_C04  |  ND     | NC1HWC0 |

    Note: the data type rules ares the same as above

    for V200 chip, if it's the first layer, and the minimum data size load to l1 buffer dose not
    exceed l1 buffer size, op select supports the additional NC1HWC0_C04 format of input x:

    | Tensor    | x            | filter         | bias    | y       |
    | :-------: | :----------: | :------------: | :-----: | :-----: |
    | Format    | NC1HWC0_C04  | FRACTAL_Z_C04  |  ND     | NC1HWC0 |

    Note:
    - convolution input data calculation with in_channels reduce for every out_height line:

            =----------------------------=
            |\^                           \
            | \in_channels                 \
            |  \ v      <- in_width ->      \
            |   =----------------------------=
            |   |  ...                       |
            |   |                      ^     |
            |   | |#| |#| ...          |    -|--
            |   |  |                         |^
            |   | stride           in_height || image data needed for one line out on height
            |   |  |                         |v
            =   | |#| |#| ...          |    -|--
             \  |  ...                 v     |
              \ |                            |
               \|                            |
                =----------------------------=
            "|#|" is the kernel(after dilate) swipe on the input image

    - minimum size calculation:

        `limit_out_height = lcm(out_width, 16) // out_width ("lcm": least common multiple)`
        `dilated_filter_h = dilation_h * (filter_height - 1) + 1`
        `limit_in_height = (limit_out_height - 1) * stride_h + dilated_filter_h - pad_left - pad_right`
        `minimum_size = limit_in_height * in_width * 8 ("8": comes from 2btype*C0=4)`
    """
    def _select_format(params):
        inputs = params[0]
        weights = params[1]
        c0_optim_flg = False
        shape_x = inputs.get("ori_shape")
        shape_x = shape_util.scalar2tensor_one(shape_x)
        format_fm = inputs.get("ori_format")
        if format_fm == "NCHW" or shape_x == (-2,):
            shape_fm = shape_x
        elif format_fm == "NHWC":
            shape_fm = [shape_x[0], shape_x[3], shape_x[1], shape_x[2]]
        else:
            err_man.raise_err_input_format_invalid("conv2d", "inputs", \
                ["NCHW", "NHWC"], format_fm)

        shape_w = weights.get("ori_shape")
        if (not isinstance(shape_w, (tuple, list))) or len(shape_w) != 4:
            err_man.raise_err_should_be_4d("conv2d", "weights")
        format_w = weights.get("ori_format")
        if format_w == "NCHW":
            shape_filter = shape_w
        elif format_w == "NHWC":
            shape_filter = [shape_w[0], shape_w[3], shape_w[1], shape_w[2]]
        elif format_w == "HWCN":
            shape_filter = [shape_w[3], shape_w[2], shape_w[0], shape_w[1]]
        else:
            err_man.raise_err_input_format_invalid("conv2d", "weights", \
                ["NCHW", "NHWC", "HWCN"], format_w)
        if shape_fm != (-2,) and shape_fm[1] <= 4:
            c0_optim_flg = True
        # dynamic conv2d doesn't support C04
        if DYNAMIC_VALUE in shape_fm:
            c0_optim_flg = False
        if (shape_filter[2] == 1) and (shape_filter[3] == 1):
            c0_optim_flg = False
        # format NC1HWC0_C04 can only be used at first conv layer
        # for those soc using NC1HWC0_C04, ensure is_first_layer == 1
        if not inputs.get("is_first_layer") and get_soc_spec("SHORT_SOC_VERSION") \
                in ("Ascend310P", "BS9SX1A", "Ascend610", "Hi3796CV300CS", "SD3403"):
            c0_optim_flg = False
        if c0_optim_flg:
            use_v200_c04_flag = False
            if get_soc_spec("SHORT_SOC_VERSION") in \
                    ("Ascend310P", "BS9SX1A", "Ascend610", "Hi3796CV300CS", "SD3403"):
                use_v200_c04_flag = util_conv2d.use_v200_c04_check(shape_fm, shape_filter, params)
            if use_v200_c04_flag:
                input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                       datatype="float16,float16,int8,int8",
                                                       format="NC1HWC0_C04,NC1HWC0,"
                                                              "NC1HWC0_C04,NC1HWC0")
            else:
                input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                       datatype="float16,float16,int8,int8",
                                                       format="NC1HWC0,NC1HWC0,"
                                                              "NC1HWC0,NC1HWC0")
            input1 = util_select_op_base.gen_param(classify="input1", name="filter",
                                                   datatype="float16,float16,int8,int8",
                                                   format="FRACTAL_Z_C04,FRACTAL_Z,"
                                                          "FRACTAL_Z_C04,FRACTAL_Z")
            input2 = util_select_op_base.gen_param(classify="input2", name="bias",
                                                   datatype="float16,float16,int32,int32",
                                                   format="ND,ND,ND,ND")
            input3 = util_select_op_base.gen_param(classify="input3", name="offset_w",
                                                   datatype="int8,int8,int8,int8",
                                                   format="ND,ND,ND,ND")
            output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                    datatype="float16,float16,int32,int32",
                                                    format="NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0")
        else:
            # only dynamic_hw or dynamic_batch is supported by dynamic conv2d
            if (shape_fm == (-2,)) or (DYNAMIC_VALUE in shape_fm):
                input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                       datatype="float16,int8",
                                                       format="NC1HWC0,NC1HWC0",
                                                       unknownshape_format="NC1HWC0,NC1HWC0")
                input1 = util_select_op_base.gen_param(classify="input1", name="filter",
                                                       datatype="float16,int8",
                                                       format="FRACTAL_Z,FRACTAL_Z",
                                                       unknownshape_format="FRACTAL_Z,FRACTAL_Z")
                input2 = util_select_op_base.gen_param(classify="input2", name="bias",
                                                       datatype="float16,int32",
                                                       format="ND,ND")
                input3 = util_select_op_base.gen_param(classify="input3", name="offset_w",
                                                       datatype="int8,int8",
                                                       format="ND,ND")
                output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                        datatype="float16,int32",
                                                        format="NC1HWC0,NC1HWC0",
                                                        unknownshape_format="NC1HWC0,NC1HWC0")
            else:
                input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                       datatype="float16,int8,int4",
                                                       format="NC1HWC0,NC1HWC0,NC1HWC0")
                input1 = util_select_op_base.gen_param(classify="input1", name="filter",
                                                       datatype="float16,int8,int4",
                                                       format="FRACTAL_Z,FRACTAL_Z,FRACTAL_Z")
                input2 = util_select_op_base.gen_param(classify="input2", name="bias",
                                                       datatype="float16,int32,int32",
                                                       format="ND,ND,ND")
                input3 = util_select_op_base.gen_param(classify="input3", name="offset_w",
                                                       datatype="int8,int8,int8",
                                                       format="ND,ND,ND")
                output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                        datatype="float16,int32,int32",
                                                        format="NC1HWC0,NC1HWC0,NC1HWC0")
        return [input0, input1, input2, input3, output0]

    params = [inputs, weights, bias, offset_w, outputs, strides,
              pads, dilations, groups, data_format, offset_x,
              kernel_name]
    param_list = _select_format(params)
    return util_select_op_base.get_dynamic_param_in_json(param_list)


@register_op_compute("conv2d", op_mode="static", support_fusion=True)
def conv2d_compute(inputs, weights, bias, offset_w, outputs, strides, pads,
                   dilations, groups=1, data_format='NCHW', offset_x=0,
                   kernel_name="conv2d", options=None):
    """
    conv2d compute

    Notice
    ------
    only used by framework combine with IR

    Parameters
    ----------
    inputs: tvm placeholder
        input 5hd feature map tensor
    weights: tvm placeholder
        input frac_z weight tensor
    outputs: tvm placeholder
        output tensor, dtype must be assigned
    bias: tvm placeholder or None
        input 1d bias tensor
    offset_w: tvm placeholder or None
        offset_w bias tensor
    strides: tuple/list of 4 integers
        stride on H/W, format sensitive
    pads: tuple/list of 4 integers
        [pad_top, pad_bottom, pad_left, pad_right]
    dilations: tuple/list of 4 integers
        dilation on H/W, format sensitive
    groups: int
        param for group covolution
    data_format: string
        input data format
    offset_x: int
        offset for fmap

    Returns
    -------
    tvm compute res
    """
    para_dict, optim_dict = util_conv2d.calc_para_from_tensor(
        inputs, weights, bias, offset_w, strides, \
        pads, dilations, offset_x, groups, kernel_name, data_format, options)

    res = conv(inputs, weights, para_dict, optim_dict)

    return res


@para_check.check_input_type(dict, dict, (dict, para_check.NONE_TYPE), (dict, para_check.NONE_TYPE), dict,
                             (tuple, list), (tuple, list), (tuple, list), int, str, int, str)
def conv2d(inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
           groups=1, data_format='NCHW', offset_x=0, kernel_name="conv2d"):
    """
    algorithm: conv2d

    Notice
    ------
    only used by framework combine with IR

    Parameters
    ----------
    inputs: dict with keys(ori_shape, ori_format and dtype)
        input 4d feature map tensor
    weights: dict with keys(ori_shape, ori_format and dtype)
        input 4d weight tensor
    bias: dict or None
        input bias tensor
    offset_w: dict or None
        input offset_w tensor
    outputs: dict with keys(dtype)
        output tensor
    strides: tuple/list of 4 integers
        stride on H/W, format sensitive
    pads: tuple/list of 4 integers
        [pad_top, pad_bottom, pad_left, pad_right]
    dilations: tuple/list of 4 integers
        dilation on H/W, format sensitive
    groups: int
        param for group covolution
    data_format: string
        input data format, determine the effective dimension(H/W)
        value of strides and dilations
    offset_x: int
        offset of fmap
    kernel_name: str
        kernel name, default value is "conv2d"

    Returns
    -------
    None

    Examples
    --------
    float16_with_bias = [
        {'ori_shape': (4, 64, 64, 16), 'ori_format': 'NHWC', 'dtype': 'float16'},
        {'ori_shape': (1, 1, 16, 1), 'ori_format': 'HWCN', 'dtype': 'float16'},
        {},
        None,
        {'dtype': 'float16'},
        (1, 2, 2, 1),
        (0, 0, 0, 0),
        (1, 3, 3, 1),
        1,
        'NHWC']
    int8_without_bias = [
        {'ori_shape': (4, 32, 64, 64), 'ori_format': 'NCHW', 'dtype': 'int8'},
        {'ori_shape': (1, 32, 1, 1), 'ori_format': 'NCHW', 'dtype': 'int8'},
        None,
        None,
        {'dtype': 'int32'},
        (1, 1, 2, 2),
        (0, 0, 0, 0),
        (1, 1, 3, 3),
        1,
        'NCHW']
    conv2d(*float16_with_bias)
    conv2d(*int8_without_bias)
    """
    in_dtype = inputs.get("dtype")
    w_dtype = weights.get("dtype")
    res_dtype = outputs.get("dtype")

    shape_fm, shape_filter, padh, padw, strideh, stridew, \
    dlt_h, dlt_w, optim_dict, fusion_para = util_conv2d.calc_para_from_dict(
        inputs, weights, strides, pads, dilations, outputs, data_format)

    use_bias = True
    if bias is None:
        use_bias = False
    use_offset_w = True
    if offset_w is None:
        use_offset_w = False

    _conv_layer_cce(shape_fm, shape_filter, in_dtype, w_dtype, res_dtype,
                    padh, padw, strideh, stridew, dlt_h, dlt_w,
                    offset_x, groups=groups, offset_w=use_offset_w,
                    bias=use_bias, optim_dict=optim_dict,
                    fusion_para=fusion_para,
                    kernel_name=kernel_name, need_build=True,
                    need_print=False)


@para_check.check_input_type((list, tuple), (list, tuple), str, str, str,
                             (list, int), (list, int), int, int,
                             (int, para_check.NONE_TYPE), (int, para_check.NONE_TYPE),
                             int, int, str,
                             bool, bool,
                             dict, (dict, para_check.NONE_TYPE), str,
                             bool, bool)
def _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                    padh, padw, strideh, stridew, dilateh=1, dilatew=1,
                    offset_x=0, groups=1, offset_w_dtype='int32',
                    offset_w=False, bias=False,
                    optim_dict=None, fusion_para=None, kernel_name="cce_conv",
                    need_build=False, need_print=False):
    """

    Parameters
    ----------
    shape_in: shape of feature map

    shape_w: shape of weight

    in_dtype: the feature map data type

    w_dtype: the weight data type

    res_dtype: the result data type

    padh: H direction padding

    padw: W direction padding

    strideh: H direction stride

    stridew: W direction stride

    dilateh: H direction spacing between kernel

    dilatew: W direction spacing between kernel

    offset_x: the offset for fmap

    offset_w_dtype: weight offset data type, default 'int32'

    offset_w: the tag for offset_w or not

    bias: the tag for bias or not

    fusion_para: the config for L2 Fusion
                input_memory_type: feature map from L2/GM, 0 for GM, 2 for L2
                output_memory_type: calculation results are outputs to L2/GM
                valid_shape: valid shape in L1 buffer, NC1HWC0
                slice_offset: the offset of each dimension
                              between valid shape and shape in

    kernel_name: cce kernel name, default value is "cce_conv"

    need_build: if need to build CCEC kernel, default value is False

    need_print: if need to print the ir, default value is False

    Returns
    -------
    wrapped_tensor

    """
    # for pylint, otherwise "Dangerous default value [] as argument"
    if optim_dict is None:
        optim_dict = {"c0_optim_flg": False, "use_v200_c04_flg": False, "v220_c04_mode": "disabled"}

    if fusion_para is None:
        fusion_para = {"fmap_l1_addr_flag": 0, "fmap_l1_valid_size": -1, "slice_offset": (0, 0, 0, 0, 0)}

    in_dtype = in_dtype.lower()
    w_dtype = w_dtype.lower()
    res_dtype = res_dtype.lower()
    offset_w_dtype = offset_w_dtype.lower()

    shape_in = list(shape_in)
    shape_w = list(shape_w)
    # fix the weight's channel=cin_ori
    shape_w[1] = shape_in[1]
    weight_ori_shape_nchw = shape_w.copy()
    cin_ori = shape_in[1] // groups
    cout_ori = shape_w[0] // groups
    shape_in, shape_w = util_conv2d.conv_layer_cce_para_check(shape_in, shape_w, padh, padw,
                                                              strideh, stridew, in_dtype, w_dtype,
                                                              res_dtype, offset_w_dtype, bias,
                                                              kernel_name, dilateh, dilatew,
                                                              optim_dict, groups)

    c0_val = CUBE_MKN[in_dtype]['mac'][1]

    enlarge = min(
        util_conv2d.lcm(util_conv2d.lcm(cin_ori, c0_val)//cin_ori, util_conv2d.lcm(cout_ori, 16)//cout_ori), groups)
    c1_opt = math.ceil(cin_ori*enlarge/c0_val)
    cout1_opt = math.ceil(cout_ori*enlarge/16)
    group_opt = math.ceil(groups / enlarge)
    c1in_ori_align = math.ceil(cin_ori*groups/c0_val)

    _, _, filter_h, filter_w = shape_w

    fmap_shape_nc1hwc0, filter_shape_frac_z = util_conv2d.conv_layer_cce_shape_calc(
        shape_in, shape_w, in_dtype, w_dtype, optim_dict, cout1_opt, c1_opt, group_opt, c1in_ori_align)
    tensor_list = []
    with tvm.target.cce():
        data = tvm.placeholder(fmap_shape_nc1hwc0, name='Fmap', dtype=in_dtype)
        tensor_list.append(data)
        weight = tvm.placeholder(filter_shape_frac_z, name='Filter', dtype=w_dtype)
        tensor_list.append(weight)
        bias_tensor = None
        offset_w_tensor = None

        if bias:
            bias_tensor = tvm.placeholder((cout_ori * groups,), name='bias_tensor', dtype=res_dtype)
            tensor_list.append(bias_tensor)
        conv_res = conv(data, weight,
                        para_dict={"bias_tensor": bias_tensor,
                                   "offset_w_tensor": offset_w_tensor,
                                   "pad_h": padh, "pad_w": padw,
                                   "stride_h": strideh, "stride_w": stridew,
                                   "dilate_h": dilateh, "dilate_w": dilatew,
                                   "filter_h": filter_h, "filter_w": filter_w,
                                   "offset_x": offset_x, "groups": groups,
                                   "res_dtype": res_dtype,
                                   "fusion_para": fusion_para,
                                   "kernel_name": kernel_name,
                                   "group": groups,
                                   "enlarge": enlarge,
                                   "c1_opt": c1_opt,
                                   "cout1_opt": cout1_opt,
                                   "group_opt": group_opt,
                                   "a_shape": fmap_shape_nc1hwc0,
                                   "weight_fracz_shape": filter_shape_frac_z,
                                   "weight_ori_shape_nchw": weight_ori_shape_nchw,},
                        optim_dict=optim_dict,
                        dsl_flag=False)
        tensor_list.append(conv_res)
        sch = auto_schedule(conv_res)

    config = {
        "print_ir": need_print,
        "need_build": need_build,
        "name": kernel_name,
        "tensor_list": tensor_list,
        "dummy_placeholder": True
    }

    build(sch, config)

def get_op_support_info(inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
                        groups=1, data_format='NCHW', offset_x=0, kernel_name="conv2d"):
    """
    algorithm: get_op_support_info

    Notice
    ------
    get the conv2d split

    Parameters
    ----------
    inputs: dict with keys(shape and dtype)
        input 4d feature map tensor
    weights: dict with keys(shape and dtype)
        input 4d weight tensor
    outputs: dict with keys(shape and dtype)
        output tensor, dtype must be assigned
    bias: dict with keys(shape and dtype) or None
        input bias tensor
    offset_w: keys(shape and dtype) or None
        input offset_w tensor
    strides: tuple/list of 4 integers
        stride on H/W, format sensitive
    pads: tuple/list of 4 integers
        [pad_top, pad_bottom, pad_left, pad_right]
    dilations: tuple/list of 4 integers
        dilation on H/W, format sensitive
    groups: int
        param for group covolution
    data_format: string
        input data format
    offset_x: int
        offset of fmap
    kernel_name: str
        kernel name, default value is "conv2d"

    Returns
    -------
    None
    """
    bias_idx = 2
    slice_info = util_conv2d.get_op_support_info_static_common(bias, bias_idx)

    # >>> start: process for dynamic shape
    shape_x = inputs.get("ori_shape")
    shape_x = shape_util.scalar2tensor_one(shape_x)
    # shape is [-2], all axes do not support split
    if list(shape_x) == [-2]:
        slice_info["_op_slice_info"]["splitMaps"].clear()
    else:
        # H/W shape is -1, remove corresponding split info
        format_fm = inputs.get("ori_format")
        overlap_axis = {"H": [2], "W": [3]}
        temp_info = slice_info['_op_slice_info']["splitMaps"]
        for name, index in overlap_axis.items():
            if shape_x[format_fm.find(name)] == -1:
                last_maps = filter(lambda splits : splits["inputList"][0]["axis"] != index, temp_info)
                temp_info = list(last_maps)
        slice_info["_op_slice_info"]["splitMaps"] = temp_info
    # <<< end: process for dynamic shape

    return json.dumps(slice_info)
