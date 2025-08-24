"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

cast_cce
"""
# 'pylint: disable=too-many-locals,invalid-name
from impl.util import util_common
from impl.util import util_soc_common
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from tbe.dsl.api import vand


def is_on_ascend910b_310p():
    return tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ("Ascend910B", "Ascend910_93", "Ascend310P")


# 'pylint: disable=inconsistent-return-statements,too-many-function-args
def _int8_uint8_process(data, dst_type):
    """
    deal with src dtype=int8 and uint8 case
    """
    check_list_value = ("uint8", "int32", "float16", "float32", "int64", "bfloat16", "bool", "int8",
                        "uint32", "int16", "uint16")
    para_check.check_dtype(dst_type, check_list_value, param_name="from_int8_uint8_to_dsttype")

    if dst_type in ("uint32", "int16", "uint16", "int8"):
        return tbe.cast_to(data, dst_type)

    if dst_type == "bool":
        data_fp16 = tbe.cast_to(data, "float16")
        data_abs = tbe.vabs(data_fp16)
        # both vmins and vmin have reached memory bound, performance is the same
        ones_broad = tbe.broadcast(tvm.const(1.0, "float16"), data.shape)
        res_min = tbe.vmin(data_abs, ones_broad)
        res = tbe.cast_to(res_min, "int8", True)
        return res

    if dst_type == "float16":
        return tbe.cast_to(data, "float16")

    if dst_type == "float32":
        data_fp16 = tbe.cast_to(data, "float16")
        return tbe.cast_to(data_fp16, "float32")

    if dst_type == "int32":
        data_fp16 = tbe.cast_to(data, "float16")
        return tbe.cast_to(data_fp16, "int32")

    if dst_type == "uint8":
        if util_soc_common.is_v220():
            return tbe.cast_to(data, "uint8")
        data_fp16 = tbe.cast_to(data, "float16")
        abs_fp16 = tbe.vabs(data_fp16)
        return tbe.cast_to(abs_fp16, "uint8")

    if dst_type == "int64":
        if data.dtype == "int8" and util_soc_common.is_v220():
            return tbe.cast_to(data, "int64")
        data_int32 = _int8_uint8_process(data, "int32")
        return _int32_process(data_int32, "int64")

    if dst_type == "bfloat16":
        data_fp16 = tbe.cast_to(data, "float16")
        data_fp32 = tbe.cast_to(data_fp16, "float32")
        return tbe.round(data_fp32, "bfloat16")


def _int32_process(data, dst_type):
    """
    deal with src dtype=int32 case
    """
    check_list_value = ("bool", "float16", "float32", "int64", "bfloat16", "uint32",
                        "int16", "uint16", "int8", "uint8")
    para_check.check_dtype(dst_type, check_list_value, param_name="from_int32_to_dsttype")
    if dst_type == "bool":
        const_one = tvm.const(1.0, "float16")
        const_broad = tbe.broadcast(const_one, data.shape)

        data = tbe.cast_to(data, "float16", True)
        x_abs = tbe.vabs(data)
        x_min = tbe.vmin(x_abs, const_broad)
        y_abs = tbe.vabs(x_min)
        return tbe.cast_to(y_abs, "int8", True)

    if dst_type == "int8":
        data_fp16 = tbe.cast_to(data, "float16")
        tensor_0 = tbe.vmuls(data_fp16, 0)
        tensor_256 = tbe.vadds(tensor_0, 256)
        result = tbe.vadds(data_fp16, 128)
        result = tbe.vmod(result, tensor_256)
        result = tbe.vadds(result, -128)
        result = tbe.cast_to(result, "float16")
        return tbe.cast_to(result, "int8", True)

    if dst_type == "uint8":
        if is_on_ascend910b_310p():
            return tbe.cast_to(data, "uint8")
        data_fp16 = tbe.cast_to(data, "float16")
        tensor_0 = tbe.vmuls(data_fp16, 0)
        tensor_256 = tbe.vadds(tensor_0, 256)
        result = tbe.vmod(data_fp16, tensor_256)
        result = tbe.cast_to(result, "float16")
        return tbe.cast_to(result, "uint8", True)

    if dst_type in ("int64", "uint32", "int16", "uint16"):
        return tbe.cast_to(data, dst_type)

    if dst_type == "float32":
        return tbe.cast_to(data, "float32")

    if dst_type == "float16":
        return tbe.cast_to(data, "float16")

    if dst_type == "bfloat16":
        data_fp32 = tbe.cast_to(data, "float32")
        return tbe.round(data_fp32, "bfloat16")


def _uint32_process(data, dst_type):
    """
    deal with src dtype=uint32 case
    """
    check_list_value = ("int64", "int32", "int16", "uint16", "int8", "uint8")
    para_check.check_dtype(dst_type, check_list_value, param_name="from_uint32_to_dsttype")

    if dst_type in ("int64", "int32", "int16", "uint16", "int8", "uint8"):
        return tbe.cast_to(data, dst_type)


def _float32_process(data, dst_type):
    """
    deal with src dtype=float32 case
    """
    check_list_value = ("int32", "float16", "int64", "bfloat16", "bool", "uint8", "complex64", "int16", "int8")
    para_check.check_dtype(dst_type, check_list_value, param_name="from_fp32_to_dsttype")
    if dst_type == "int32":
        return tbe.cast_to(data, "int32")
   
    if dst_type == "int16":
        data_int32 = tbe.cast_to(data, "int32")
        return _int32_process(data_int32, "int16")

    if dst_type == "int8":
        data_int32 = tbe.cast_to(data, "int32")
        data_int16 = _int32_process(data_int32, "int16")
        return _int16_process(data_int16, "int8")

    if dst_type == "float16":
        return tbe.cast_to(data, "float16")

    if dst_type == "int64":
        # Only some platforms support the trunc interface
        if tbe_platform.api_check_support("tik.vcopy"):
            return tbe.trunc(data, "int64")
        else:
            data_int32 = _float32_process(data, "int32")
            return _int32_process(data_int32, dst_type)

    if dst_type == "bfloat16":
        return tbe.round(data, "bfloat16")

    if dst_type == "bool":
        if util_soc_common.after_v200():
            res_cmp = tbe.vcmp(data, 0.0, "eq")
            one = tvm.const(1.0, dtype="float16")
            ones = tbe.broadcast(one, data.shape)
            zero = tvm.const(0.0, dtype="float16")
            zeros = tbe.broadcast(zero, data.shape)
            res_sel = tbe.vsel(res_cmp, zeros, ones)
            res = tbe.vcmp(res_sel, 0.0, "ne")
        else:
            positive_data = tbe.vabs(data)
            ceil_data = tbe.ceil(positive_data)
            data_int32 = _float32_process(ceil_data, "int32")
            res = _int32_process(data_int32, dst_type)
        return res

    if dst_type == "uint8":
        if tbe_platform.api_check_support("tik.vcopy"):
            data_int64 = tbe.trunc(data, "int64")
            return _int64_process(data_int64, "uint8")
        else:
            data_int32 = tbe.cast_to(data, "int32")
            data_fp32 = tbe.cast_to(data_int32, "float32")
            tensor_0 = tbe.vmuls(data_fp32, 0)
            tensor_256 = tbe.vadds(tensor_0, 256)
            result = tbe.vmod(data_fp32, tensor_256)
            return tbe.cast_to(result, "uint8", True)

    if dst_type == "complex64":
        return tbe.cast_to(data, "complex64")


def _float16_process(data, dst_type):
    """
    deal with src dtype=float16 case
    """
    check_list_value = ("uint8", "int32", "float32", "bfloat16", "int8", "bool", "int16", "complex32")
    if util_soc_common.is_v220():
        check_list_value = ("uint8", "int32", "float32", "bfloat16", "int8", "bool", "int16", "complex32", "int64")
    para_check.check_dtype(dst_type, check_list_value, param_name="from_fp16_to_dsttype")
    if dst_type in ["float32", "int32"]:
        return tbe.cast_to(data, dst_type)

    if dst_type == "bool":
        res_cmp = tbe.vcmp(data, 0.0, "eq")
        ones = tvm.const(1.0, dtype="float16")
        zeros = tvm.const(0.0, dtype="float16")
        res = tbe.vsel(res_cmp, zeros, ones)
        return tbe.vcmp(res, 0.0, "ne")

    if dst_type == "int8":
        data_int32 = tbe.cast_to(data, "int32")
        const_ff = tvm.const(255, "int32")
        shape_data = shape_util.shape_to_list(data.shape)
        const_broad = tbe.broadcast(const_ff, shape_data)
        data_and = vand(data_int32, const_broad)
        data_fp16 = tbe.cast_to(data_and, "float16")
        res = util_common.uint8_int8_overflow_proc(data_fp16, "int8")
        return res

    if dst_type == "uint8":
        if not tbe_platform.api_check_support("tbe.dsl.cast_to", "s322f16") and \
                tbe_platform.api_check_support("tbe.dsl.vmod", "float16"):
            return tbe.cast_to(data, "uint8", True)

        cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
        if util_soc_common.after_v200() or cce_product == "Ascend310P":
            data_int32 = tbe.cast_to(data, "int32")
            data_fp32 = tbe.cast_to(data_int32, "float32")
            return util_common.uint8_int8_overflow_proc(data_fp32, "uint8")

        data_int32 = tbe.cast_to(data, "int32")
        data_fp16 = tbe.cast_to(data_int32, "float16")
        tensor_0 = tbe.vmuls(data_fp16, 0)
        tensor_256 = tbe.vadds(tensor_0, 256)
        result = tbe.vmod(data_fp16, tensor_256)
        result = tbe.cast_to(result, "float16")
        return tbe.cast_to(result, "uint8", True)

    if dst_type == "bfloat16":
        data = tbe.cast_to(data, "float32")
        return tbe.round(data, "bfloat16")

    if dst_type == "int16":
        data = tbe.trunc(data, "int16")
        return data

    if dst_type == "complex32":
        return tbe.cast_to(data, "complex32")
    
    if dst_type == "int64":
        return tbe.cast_to(data, "int64")


def _bfloat16_process(data, dst_type):
    """
    deal with src dtype=bfloat16 case
    """
    check_list_value = ("int32", "float32", "float16", "int8", "bool", "uint8", "int16")
    para_check.check_dtype(dst_type, check_list_value, param_name="from_bf16_to_dsttype")
    # cast_to use trunc too for dst type int32 and int16
    if dst_type == "int32":
        return tbe.trunc(data, "int32")

    if dst_type == "float32":
        return tbe.cast_to(data, "float32")

    if dst_type == "float16":
        data_fp32 = tbe.cast_to(data, "float32")
        return tbe.cast_to(data_fp32, "float16")

    if dst_type == "int8":
        data_fp32 = tbe.cast_to(data, "float32")
        data_int32 = tbe.cast_to(data_fp32, "int32")
        const_ff = tvm.const(255, "int32")
        shape_data = shape_util.shape_to_list(data.shape)
        const_broad = tbe.broadcast(const_ff, shape_data)
        data_and = vand(data_int32, const_broad)
        data_fp16 = tbe.cast_to(data_and, "float16")
        res = util_common.uint8_int8_overflow_proc(data_fp16, "int8")
        return res

    if dst_type == "bool":
        data_fp32 = tbe.cast_to(data, "float32")
        res_cmp = tbe.vcmp(data_fp32, 0.0, "eq")
        one = tvm.const(1.0, dtype="float16")
        ones = tbe.broadcast(one, data_fp32.shape)
        zero = tvm.const(0.0, dtype="float16")
        zeros = tbe.broadcast(zero, data_fp32.shape)
        res_sel = tbe.vsel(res_cmp, zeros, ones)
        res = tbe.vcmp(res_sel, 0.0, "ne")
        return res

    if dst_type == "uint8":
        data_fp32 = tbe.cast_to(data, "float32")
        return _float32_process(data_fp32, "uint8")


def _int64_process(data, dst_type):
    """
    deal with src dtype=int64 case
    """
    check_list_value = ("int32", "float32", "float16", "uint8", "bool", "bfloat16", "uint32",
                        "int16", "uint16", "int8")
    para_check.check_dtype(dst_type, check_list_value, param_name="from_int64_to_dsttype")
    if dst_type == "float32":
        # Only some platforms support the round interface
        if tbe_platform.api_check_support("tik.vcopy"):
            return tbe.round(data, "float32")
        else:
            data_int32 = _int64_process(data, "int32")
            return _int32_process(data_int32, "float32")

    if dst_type == "bfloat16":
        data_fp32 = tbe.round(data, "float32")
        return tbe.round(data_fp32, "bfloat16")

    if dst_type in ("float16", "int32", "uint32", "int16", "uint16", "int8"):
        return tbe.cast_to(data, dst_type)

    if dst_type == "uint8":
        if is_on_ascend910b_310p():
            return tbe.cast_to(data, "uint8")
        data = _int64_process(data, "int32")
        data_fp16 = _int32_process(data, "float16")
        return _float16_process(data_fp16, "uint8")

    if dst_type == "bool":
        # Some platforms need to be converted to float32 without causing accuracy issues
        if tbe_platform.api_check_support("tik.vcopy"):
            const_one = tvm.const(1.0, "float32")
            const_broad = tbe.broadcast(const_one, data.shape)
            data = tbe.round(data, "float32")
        else:
            const_one = tvm.const(1.0, "float16")
            const_broad = tbe.broadcast(const_one, data.shape)
            data = tbe.cast_to(data, "float16", True)
        x_abs = tbe.vabs(data)
        x_min = tbe.vmin(x_abs, const_broad)
        y_abs = tbe.vabs(x_min)
        return tbe.cast_to(y_abs, "int8", True)


def _uint1_process(data, dst_type):
    """
    deal with src dtype=uint1 case
    """
    check_list_value = ("float16")
    para_check.check_dtype(dst_type, check_list_value, param_name="from_uint1_to_dsttype")
    if dst_type == "float16":
        ones = tbe.broadcast(tvm.const(1, dtype="float16"), data.shape, output_dtype="float16")
        zeros = tbe.broadcast(tvm.const(0, dtype="float16"), data.shape, output_dtype="float16")
        sel_0 = tbe.vsel(data, ones, zeros)
        return sel_0


def _int16_process(data, dst_type):
    """
    deal with src dtype=int16 case
    """
    check_list_value = ("float16", "float32", "int64", "int32", "uint32", "uint16", "int8", "uint8")
    para_check.check_dtype(dst_type, check_list_value, param_name="from_int16_to_dsttype")
    if dst_type == "float16":
        return tbe.round(data, dst_type)

    if dst_type == "float32":
        cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
        if cce_product == "Ascend310P":
            data_int32 = tbe.cast_to(data, "int32")
            return tbe.cast_to(data_int32, "float32")
        return tbe.cast_to(data, "float32")

    if dst_type in ("int64", "int32", "uint32", "uint16", "int8", "uint8"):
        return tbe.cast_to(data, dst_type)


def _uint16_process(data, dst_type):
    """
    deal with src dtype=uint16 case
    """
    check_list_value = ("int64", "int32", "uint32", "int16", "int8", "uint8")
    para_check.check_dtype(dst_type, check_list_value, param_name="from_uint16_to_dsttype")
    if dst_type in ("int64", "int32", "uint32", "int16", "int8", "uint8"):
        return tbe.cast_to(data, dst_type)


def _cast_dsttype_conversion(dst_type):
    """
    git cast_dsttype
    """
    if dst_type == 0:
        dst_type = "float32"
    if dst_type == 1:
        dst_type = "float16"
    if dst_type == 2:
        dst_type = "int8"
    if dst_type == 3:
        dst_type = "int32"
    if dst_type == 4:
        dst_type = "uint8"
    if dst_type == 6:
        dst_type = "int16"
    if dst_type == 7:
        dst_type = "uint16"
    if dst_type == 8:
        dst_type = "uint32"
    if dst_type == 9:
        dst_type = "int64"
    if dst_type == 10:
        dst_type = "uint64"
    if dst_type == 12:
        dst_type = "bool"
    if dst_type == 16:
        dst_type = "complex64"
    if dst_type == 27:
        dst_type = "bfloat16"
    if dst_type == 33:
        dst_type = "complex32"

    return dst_type


# 'pylint: disable=unused-argument
def check_supported(input_x, output_y, dst_type, kernel_name="cast"):
    """
    verify the types of cast supported by tbe
    """
    src_type = input_x.get("dtype").lower()
    out_type = output_y.get("dtype").lower()
    check_type = src_type not in ("int64", "uint1") and out_type != "int64"
    check_f322u8 = src_type != "float32" or out_type != "uint8"
    support_s322s64 = not tbe_platform.intrinsic_check_support("Intrinsic_vconv", "s322s64")
    support_s642s32 = not tbe_platform.intrinsic_check_support("Intrinsic_vconv", "s642s32")
    support_is_unknown = not util_common.is_unknown([input_x, output_y])
    soc_block_size = tbe_platform.get_block_size()
    is_soc_version_035 = False
    check_type_035 = False
    if soc_block_size == 16:
        is_soc_version_035 = True
        check_type_035 = src_type in ("int16", "float16") and out_type in ("float16", "int16")
    is_035_type = check_type_035 and is_soc_version_035
    # If the command in all is a static shape and is not a conversion of the type in the check_type
    # and currently does not support int32<->int64, the return fails, and the static operator is taken.
    if all((check_type, support_is_unknown, support_s322s64, support_s642s32)) and check_f322u8 and not is_035_type:
        return False, "Dynamic shapes are not currently supported."

    if src_type == "bool":
        src_type = "int8"

    dst_type = _cast_dsttype_conversion(dst_type)

    check_list = []
    if src_type == "float16":
        check_list = ["float32", "int32", "uint8", "bfloat16", "int8", "bool", "int16", "complex32", "int64"]
    elif src_type == "float32":
        check_list = ["float16", "int32", "int64", "bfloat16", "bool", "uint8", "complex64"]
    elif src_type == "int8":
        check_list = ["float32", "float16", "int32", "uint8", "int64", "bfloat16", "bool", "int8"]
    elif src_type == "uint8":
        check_list = ["float32", "float16", "int32", "int64", "bfloat16"]
    elif src_type == "int32":
        check_list = ["bool", "uint8", "int8", "float32", "float16", "int64", "bfloat16"]
    elif src_type == "int64":
        check_list = ["float32", "int32", "float16", "uint8", "bool", "bfloat16"]
    elif src_type == "bfloat16":
        check_list = ["float32", "int32", "float16", "int8", "uint8", "bool"]
    elif src_type == "uint1":
        check_list = ["float16"]
    elif src_type == "int16":
        check_list = ["float16"]

    if dst_type in check_list:
        return True, ""

    reason = "when input dtyps is %s, output dtype[%s] is not in checklist:%s" % (src_type, dst_type, str(check_list))
    return False, reason


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
@register_operator_compute("Cast", op_mode="dynamic", support_fusion=True)
def cast_compute(data, output_y, dst_type, kernel_name="cast"):
    """
    core func of tensor casting. cast a tensor form src data type to dst data
    type. restrictions of input algorithms are as follow
    only types' groups blow are support tensor process:
        float16->float32
        float16->int32
        float32->float16
        float32->int32
        int8->float32
        uint8->float32
        int8->float16
        uint8->float16
        int8->int32
        uint8->int32
        int32->uint8 // number out of [0,255] can get unexpected result
        int32->int8 // number out of [-128,127] can get unexpected result
        int32->float32 // For tans with fp16, only guarantees
                        number in [-1023,1023] get correct result
        int32->float16 // only guarantees
                        number in [-1023,1023] get correct result

        int64->float32
        float32->int64
        int64->int32
        int32->int64
        float32->bfloat16
        bfloat16->float32
        bfloat16->int32
        bfloat16->float16
        float16->bfloat16
        float16->int64
    Parameters
    ----------
    placeholders: list.
        the input tensor
    src_type: str
        the input data type.
    dst_type: str
        the output data type.

    Returns
    -------
        the compute result tensor with type dst_type
    """
    dst_type = _cast_dsttype_conversion(dst_type)
    src_data_type = data.dtype
    para_check.check_dtype(src_data_type,
                           ("float16", "float32", "int8", "uint8", "int32", "int64", "bfloat16", "uint1", "int16",
                            "uint32", "uint16"), param_name="input_x")

    if src_data_type in ("int8", "uint8"):
        return _int8_uint8_process(data, dst_type)

    if src_data_type == "float32":
        return _float32_process(data, dst_type)

    if src_data_type == "float16":
        return _float16_process(data, dst_type)

    if src_data_type == "int32":
        return _int32_process(data, dst_type)

    if src_data_type == "bfloat16":
        return _bfloat16_process(data, dst_type)

    if src_data_type == "int64":
        return _int64_process(data, dst_type)

    if src_data_type == "uint1":
        return _uint1_process(data, dst_type)

    if src_data_type == "int16":
        return _int16_process(data, dst_type)

    if src_data_type == "uint32":
        return _uint32_process(data, dst_type)

    if src_data_type == "uint16":
        return _uint16_process(data, dst_type)


@register_operator("Cast")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def cast(input_x, output_y, dst_type, kernel_name="cast"):
    """
    cast a tensor/scaler with input shape form src data type to dst data
    type. restrictions of input algorithms are as follow
    only types' groups blow are support tensor process:
        float16->float32
        float16->int32
        float32->float16
        float32->int32
        int8->float32
        uint8->float32
        int8->float16
        uint8->float16
        int8->int32
        uint8->int32
        int32->uint8 // number out of [0,255] can get unexpected result
        int32->int8 // number out of [-128,127] can get unexpected result
        int32->float32 // For tans with fp16, only guarantees
                        number in [-1023,1023] get correct result
        int32->float16 // only guarantees
                        number in [-1023,1023] get correct result

        int64->float32
        float32->int64
        float32->int16
        float32->int8
        int64->int32
        int32->int64
        int64->float16
        int16->float32
        float32->bfloat16
        bfloat16->float32
        bfloat16->int32
        bfloat16->float16
        float16->bfloat16

        int64->float16
        float16->int64
    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape as input,
        and the dtype is the dst dtype need to cast
    kernel_name : str
        cce kernel name, default value is cast

    Returns
    -------
    None
    """

    src_type = input_x.get("dtype").lower()

    if src_type == "bool":
        src_type = "int8"

    schedules, tensors = [], []
    ins = classify([input_x], OpPatternMode.ELEWISE)
    for (_input_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_input_x])[0]

            if dst_type is None:
                dst_type = output_y.get("dtype").lower()
            else:
                dst_type = _cast_dsttype_conversion(dst_type)

            data = tvm.placeholder(x_shape, name="data", dtype=src_type)
            res = cast_compute(data, output_y, dst_type, kernel_name)
            tensors.append([data, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
