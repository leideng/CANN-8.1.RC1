# Copyright 2021 Huawei Technologies Co., Ltd
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
extract_image_patches
"""
# 'pylint: disable=too-many-lines
from tbe.dsl.base import operation
from tbe.dsl.base.operation import get_context
from tbe.dsl.base.operation import get_compile_info
from tbe.common.utils import op_tiling
from tbe.common.utils.errormgr import error_manager_vector
from tbe.common.utils.errormgr import get_error_message
from impl.util import util_common
from impl.im2col_common_func import im2col_compute
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    BLOCK_SIZE = 16
    BLOCK_SIZE_INT8 = 32
    DOUBLE_BUFFER = 2
    FP16_SIZE = 2
    INT8_SIZE = 1
    MAX_INT32_VALUE = 2**31 - 1
    # shape size to allow workspace use mulit-core and split in howo axis
    DMA_SPLIT_THRESHOLD = 3072
    SIZE_L1 = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)


# 'pylint: disable=too-many-arguments
def tf_get_windowed_output_size_verbose_dynamic(input_size, filter_size, dilation_rate, stride, padding_type,
                                                kernel_name):
    """
    get output and padding size using tensorflow padding rule

    Parameters
    ----------
    input_size : int, feature map size

    filter_size : int, filter size

    dilation_rate: int, dilation rate

    stride: int, stride size

    padding_type: string, support "SAME", "VALID" or "EXPLICIT"

    Returns
    -------
    output_size: int, output feature map size

    padding_before: int, feature map padding before size

    padding_after: int, feature map padding after size
    """
    if isinstance(filter_size, int) and filter_size <= 0:
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The filter_size must be > 0, but filter_size is [%s]" % filter_size
        raise RuntimeError(dict_args, get_error_message(dict_args))

    if isinstance(stride, int) and stride <= 0:
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The stride must be > 0, but stride is [%s]" % stride
        raise RuntimeError(dict_args, get_error_message(dict_args))

    if isinstance(dilation_rate, int) and dilation_rate < 1:
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "dilation_rate must be >= 1, " \
                                      "but dilation_rate is [%s]" % dilation_rate
        raise RuntimeError(dict_args, get_error_message(dict_args))

    effective_filter_size = (filter_size - 1) * dilation_rate + 1
    if padding_type == "VALID":
        output_size = (input_size - effective_filter_size + stride) // stride
        padding_before = 0
        padding_after = 0
    elif padding_type == "SAME":
        output_size = (input_size + stride - 1) // stride
        padding_needed = tvm.max(0, (output_size - 1) * stride + effective_filter_size - input_size)
        padding_before = padding_needed // 2
        padding_after = padding_needed - padding_before
    else:
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "Unsupported padding type [%s], " \
                                      "padding_type must be VALID or SAME" % padding_type
        raise RuntimeError(dict_args, get_error_message(dict_args))

    if (isinstance(output_size, tvm.tir.IntImm) or isinstance(output_size, int)) and int(output_size) <= 0:
        error_manager_vector.raise_err_specific_reson(kernel_name,
                                                      "output_size[%s] must but be large than 0!" % output_size)

    return output_size, padding_before, padding_after


# 'pylint: disable=too-many-arguments
def param_check(ksizes, strides, dilates, kernel_name):
    _, kernel_h, kernel_w, _ = ksizes
    _, stride_h, stride_w, _ = strides
    _, dilate_h, dilate_w, _ = dilates

    if kernel_h >= 256 or kernel_w >= 256 or kernel_h <= 0 or kernel_w <= 0:
        error_manager_vector.raise_err_specific_reson(kernel_name, "kernel_h and kernel_w can not >= 256 or <= 0!")
    if stride_h >= 64 or stride_w >= 64 or stride_h <= 0 or stride_w <= 0:
        error_manager_vector.raise_err_specific_reson(kernel_name, "stride_h and stride_w can not >= 64 or <= 0!!")
    if dilate_h >= 256 or dilate_w >= 256 or dilate_h <= 0 or dilate_w <= 0:
        error_manager_vector.raise_err_specific_reson(kernel_name, "dilate_h and dilate_w can not >= 256 or <= 0!!")


# 'pylint: disable=unused-argument,too-many-locals,too-many-arguments
def extract_image_patches_compute(fmap,
                                  origin_c_in,
                                  ksizes,
                                  strides,
                                  dilates,
                                  padding,
                                  mode,
                                  cin_range,
                                  kernel_name="extract_image_patches"):
    """
    ops compute

    Parameters
    ----------
    fmap : TVM tensor
        the placeholder of fmap
    origin_c_in : real c size of input
    ksizes: input attr
    strides: input attr
    dilates: input attr
    padding: input attr
    mode: input attr origin_cin align or not
    cin_range: cin dim range
    kernel_name : str kernel name

    Returns
    -------
    output_res
    workspace_res
    workspace_shape
    """
    # fmap's format is NC1HWC0
    fmap_shape = fmap.shape
    fmap_h = fmap_shape[2]
    fmap_w = fmap_shape[3]

    _, kernel_h, kernel_w, _ = ksizes
    _, stride_h, stride_w, _ = strides
    _, dilate_h, dilate_w, _ = dilates

    is_var = False

    if is_var:
        out_h = tbe.var("out_h")
        out_w = tbe.var("out_w")
        padding_h_before = tbe.var("padding_h_before")
        padding_w_before = tbe.var("padding_w_before")
        padding_h_after = tbe.var("padding_h_after")
        padding_w_after = tbe.var("padding_w_after")

    if not is_var or (isinstance(fmap_h, tvm.tir.IntImm) and int(fmap_h) >= 0):
        out_h, padding_h_before, padding_h_after = tf_get_windowed_output_size_verbose_dynamic(
            fmap_h, kernel_h, dilate_h, stride_h, padding, kernel_name)
    if not is_var or (isinstance(fmap_w, tvm.tir.IntImm) and int(fmap_w) >= 0):
        out_w, padding_w_before, padding_w_after = tf_get_windowed_output_size_verbose_dynamic(
            fmap_w, kernel_w, dilate_w, stride_w, padding, kernel_name)

    pads = (padding_h_before, padding_h_after, padding_w_before, padding_w_after)
    ksize = (kernel_h, kernel_w)
    stride = (stride_h, stride_w)
    dilate = (dilate_h, dilate_w)

    operation.get_context().get_current_compute().add("is_origin_cin_align", mode)
    operation.get_context().get_current_compute().add("ori_cin_range", cin_range)
    output_res, workspace_res, workspace_shape = im2col_compute(fmap,
                                                                origin_c_in,
                                                                ksize,
                                                                stride,
                                                                dilate,
                                                                pads,
                                                                out_h,
                                                                out_w,
                                                                is_dynamic=True,
                                                                is_origin_cin_align=mode,
                                                                cin_range=cin_range)

    return output_res, workspace_res, workspace_shape


# 'pylint: disable=too-many-arguments,unused-argument,invalid-name,too-many-locals,too-many-branches
@register_operator("ExtractImagePatches", pattern="ExtractImagePatches")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_LIST_INT, para_check.OPTION_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_STR, para_check.KERNEL_NAME)
def extract_image_patches(images, y, ksizes, strides, dilates, padding, kernel_name="extract_image_patches"):
    """
    calculating data

    Parameters
    ----------
    images : dict
        shape and dtype of input, support float16/bfloat16/float/int8/uint8
    y : dict
        shape and dtype of output, should be same shape and type as input
    ksizes: input attr
    strides: input attr
    dilates: input attr
    padding: input attr
    kernel_name : str
        kernel name, default value is "extract_image_patches"

    Returns
    -------
    None
    """
    if images.get("format") == "NCHW":
        from impl.dynamic.extract_image_patches_nchw import ExtractImagePatchesNCHW
        ExtractImagePatchesNCHW(images, ksizes, strides, dilates, padding, [0, 0, 0, 0]).build(kernel_name)
        return

    shape_input_4d = images.get("ori_shape")
    shape_input_5d = images.get("shape")

    dtype_input = images.get("dtype").lower()
    if dtype_input not in ("int8", "uint8", "float16", "float", "float32", "bfloat16"):
        error_manager_vector.raise_err_specific_reson(
            kernel_name, "dtype can only be uint8, int8, float16, bfloat16 or float32!")
    if dtype_input == "bfloat16":
        dtype_input = "int16"
    if dtype_input in ('int8', 'uint8'):
        align_block_size = Constant.BLOCK_SIZE_INT8
    else:
        align_block_size = Constant.BLOCK_SIZE

    if -2 in shape_input_5d:
        shape_input_5d = [-1, -1, -1, -1, align_block_size]
        images["shape"] = shape_input_5d
        images["range"] = util_common.gen_range(shape_input_5d)
    if -2 in shape_input_4d:
        shape_input_4d = [-1, -1, -1, -1]
        images["ori_shape"] = shape_input_4d
        images["ori_range"] = util_common.gen_range(shape_input_4d)

    shape_range = images.get("range")
    if images.get("ori_range"):
        ori_shape_range = [list(sub_range) for sub_range in images.get("ori_range")]

    is_const_shape = -1 not in shape_input_4d
    is_binary = ksizes is None or strides is None or dilates is None
    operation.get_context().add("is_const_shape", is_const_shape)
    operation.get_context().add("is_binary", is_binary)
    if not is_binary:
        data_format = images.get('ori_format')
        format_list = ('NHWC', 'NCHW')
        if data_format not in format_list:
            error_manager_vector.raise_err_input_format_invalid(kernel_name, "x", format_list, data_format)
        if len(ksizes) != 4 or len(strides) != 4 or len(dilates) != 4:
            error_manager_vector.raise_err_check_params_rules(kernel_name, 'input params invalide',
                                                            ['ksizes', 'strides', 'dilates', 'shape_input_4d'],
                                                            [ksizes, strides, dilates, shape_input_4d])

        if len(shape_input_4d) != 4 or len(shape_range) != 5:
            error_manager_vector.raise_err_check_params_rules(kernel_name, 'input shape or range invalide',
                                                            ['shape_range', 'shape_input_4d'],
                                                            [shape_range, shape_input_4d])

        # NCHW -> NHWC
        if data_format == 'NCHW':
            shape_input_4d = [shape_input_4d[0], shape_input_4d[2], shape_input_4d[3], shape_input_4d[1]]
            ksizes = [ksizes[0], ksizes[2], ksizes[3], ksizes[1]]
            strides = [strides[0], strides[2], strides[3], strides[1]]
            dilates = [dilates[0], dilates[2], dilates[3], dilates[1]]
            if images.get("ori_range"):
                ori_shape_range = [list(ori_shape_range[0]), list(ori_shape_range[2]),
                                list(ori_shape_range[3]), list(ori_shape_range[1])]

    else:
        ksizes = [1, tbe.var("kernel_h", (1, 255)), tbe.var("kernel_w", (1, 255)), 1] # 1<=ksize<=255
        strides = [1, tbe.var("stride_h"), tbe.var("stride_w"), 1]
        dilates = [1, tbe.var("dilate_h"), tbe.var("dilate_w"), 1]

    if not images.get("ori_range"):
        ori_shape_range = [list(shape_range[0]), list(shape_range[2]), list(shape_range[3]), list(shape_range[1])]
        if shape_input_4d[-1] < 0:
            ori_shape_range[-1][0] = (ori_shape_range[-1][0] - 1) * align_block_size
            if ori_shape_range[-1][-1] is not None:
                ori_shape_range[-1][-1] = (ori_shape_range[-1][-1] * align_block_size)

    tiling_key = None
    if is_const_shape:
        if -1 in shape_input_5d:
            shape_input_5d = [shape_input_4d[0],
                              (shape_input_4d[3] + align_block_size - 1) // align_block_size,
                              shape_input_4d[1],
                              shape_input_4d[2],
                              align_block_size]
        input_list = [{
            "shape": shape_input_5d,
            "ori_shape": shape_input_4d,
            "format": "NC1HWC0",
            "ori_format": "NHWC",
            "dtype": dtype_input
        }]
        output_list = []
        attr_list = [{"name": "ksizes", "dtype": "list_int", "value": ksizes},
                    {"name": "strides", "dtype": "list_int", "value": strides},
                    {"name": "rates", "dtype": "list_int", "value": dilates},
                    {"name": "padding", "dtype": "str", "value": padding}]
        compile_info = {
            "envWithoutCbuf": True,
            "socVersion": tbe_platform.get_soc_spec("SHORT_SOC_VERSION"),
            "coreNum": tbe_platform.get_soc_spec("CORE_NUM"),
            "SIZE_L1": tbe_platform.get_soc_spec("L1_SIZE"),
            "SIZE_UB": tbe_platform.get_soc_spec("UB_SIZE"),
            "dtypeInput": dtype_input,
            "paddingType": padding,
            "isDB": True,
            "isVar": False,
            "isConst": True,
            "isBinary": False
        }
        run_info = op_tiling.do_op_tiling(get_context().get_op_type(),
                                          compile_info, input_list, output_list, None, None, attr_list)
        operation.get_context().add("run_info", run_info)
        tiling_key = run_info.get("tiling_key")

    schedules, tensors = [], []
    env_without_cbuf = (not tbe_platform.intrinsic_check_support("Intrinsic_data_move_l12ub"))

    only_compile_without_cbuf = False
    if dtype_input == "int16" or env_without_cbuf:
        only_compile_without_cbuf = True
    if is_const_shape and tiling_key >= 10000:
        only_compile_without_cbuf = True

    if only_compile_without_cbuf or not is_const_shape:
        from impl.dynamic.extract_image_patches_without_cbuf import ExtractImagePatchesWithoutCbuf
        eipwc_obj = ExtractImagePatchesWithoutCbuf(shape_input_4d, dtype_input,
                                                   ksizes, strides, dilates, padding, kernel_name)
        tensor_list, sch = eipwc_obj.do_without_cbuf(env_without_cbuf)
        tensors.append(tensor_list)
        schedules.append(sch)

        if only_compile_without_cbuf:
            eipwc_obj.add_compile_info()
            tbe.build(schedules, {"name": kernel_name, "tensor_list": tensors})
            return

    if not is_binary:
        param_check(ksizes, strides, dilates, kernel_name)

    var_map = _var_shape(shape_input_4d, shape_input_5d, align_block_size, is_const_shape)

    fmap_c0 = align_block_size
    shape_input = [var_map["fmap_n"], var_map["c1"], var_map["fmap_h"], var_map["fmap_w"], fmap_c0]
    soc_version = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    operation.get_context().add("paddingType", padding)
    operation.get_context().add("soc_version", soc_version)
    operation.get_context().add("var_map", var_map)
    operation.get_context().add("C0", align_block_size)
    operation.get_context().add("ori_shape_range", ori_shape_range)
    is_var = False
    operation.get_context().add("ISVAR", is_var)

    is_origin_cin_const = (shape_input_4d[3] != -1)
    # if not is_origin_cin_const
    if var_map["origin_c_in"] % align_block_size == 0:
        origin_cin_align_info = [True, var_map["origin_c_in"], [var_map["origin_c_in"], var_map["origin_c_in"]]]
    elif is_origin_cin_const:
        origin_cin_align_info = [False, var_map["origin_c_in"], [var_map["origin_c_in"], var_map["origin_c_in"]]]

    ins = [[True, var_map["origin_c_in"], ori_shape_range[-1]]]
    if not is_origin_cin_const and dtype_input not in ('int8', 'uint8') and ori_shape_range[-1][0] <= 1:
        ins.append([False, 1, (1, 1)])
    if not is_origin_cin_const and (ori_shape_range[-1][0] > fmap_c0 or
                                    (ori_shape_range[-1][1] is not None and ori_shape_range[-1][1] < fmap_c0)):
        ins.append([False, var_map["origin_c_in"], ori_shape_range[-1]])
    elif not is_origin_cin_const and (ori_shape_range[-1][0] < fmap_c0 and
                                      (ori_shape_range[-1][1] is None or ori_shape_range[-1][1] > fmap_c0)):
        ins.append([False, var_map["origin_c_in"], [ori_shape_range[-1][0], fmap_c0 - 1]])
        ins.append([False, var_map["origin_c_in"], [fmap_c0 + 1, ori_shape_range[-1][1]]])

    ins = [origin_cin_align_info] if is_origin_cin_const else ins

    for mode, origin_c_in, cin_range in ins:
        with tbe.compute():
            data_input = tvm.placeholder(shape_input, name="data_input_gm", dtype=dtype_input)
            output_res, _, _ = extract_image_patches_compute(data_input, origin_c_in, ksizes, strides,
                                                             dilates, padding, mode, cin_range, kernel_name)
            tensor_list = [data_input, output_res]
            tensors.append(tensor_list)
        with tvm.target.cce():
            sch = tbe.auto_schedule(output_res)
            schedules.append(sch)

    build_args = {"constant_realize_extent_in_infer_bound": False}

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors, "build_args": build_args}
    tbe.build(schedules, config)
    return


def _var_shape(shape_input_4d, shape_input_5d, align_block_size, is_const_shape):
    if is_const_shape:
        var_map = {
            "fmap_n": shape_input_4d[0],
            "fmap_h": shape_input_4d[1],
            "fmap_w": shape_input_4d[2],
            "origin_c_in": shape_input_4d[3],
            "c1": (shape_input_4d[3] + align_block_size - 1) // align_block_size
        }
        return var_map

    var_map = {
        "fmap_n": tbe.var("fmap_n"),
        "fmap_h": tbe.var("fmap_h"),
        "fmap_w": tbe.var("fmap_w"),
        "origin_c_in": tbe.var("origin_c_in"),
        "c1": tbe.var("c1")
    }
    key_string_map = {
        "fmap_n": [0, 0],
        "fmap_h": [1, 2],
        "fmap_w": [2, 3],
        "origin_c_in": [3, 1],
    }

    for key, value in key_string_map.items():
        if key != "origin_c_in":
            if shape_input_5d[value[1]] >= 0:
                var_map[key] = shape_input_5d[value[1]]
            elif shape_input_4d[value[0]] >= 0:
                var_map[key] = shape_input_4d[value[0]]
        else:
            if shape_input_4d[value[0]] >= 0:
                var_map["origin_c_in"] = shape_input_4d[value[0]]
                var_map["c1"] = (shape_input_4d[3] + align_block_size - 1) // align_block_size
            elif shape_input_5d[value[1]] >= 0:
                var_map["c1"] = shape_input_5d[value[1]]
    return var_map
