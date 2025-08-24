#!/usr/bin/python
# -*- coding: utf-8 -*-
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
aipp_comm
"""
# 'pylint: disable=too-many-lines,invalid-name,too-many-statements,too-many-arguments,no-else-return
# 'pylint: disable=too-many-locals,too-many-branches,ungrouped-imports,too-many-boolean-expressions
# 'pylint: disable=too-many-return-values
import numpy
from impl.util.platform_adapter import get_const
from impl.util.platform_adapter import error_manager_vector

import te.platform as tbe_platform
from tbe import tvm


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class Const:
    """
    The class for constant.
    """
    # get available ub size
    DEFAULT_MATRIX_R0C0_YUV2RGB = 298
    DEFAULT_MATRIX_R0C1_YUV2RGB = 516
    DEFAULT_MATRIX_R0C2_YUV2RGB = 0
    DEFAULT_MATRIX_R1C0_YUV2RGB = 298
    DEFAULT_MATRIX_R1C1_YUV2RGB = -100
    DEFAULT_MATRIX_R1C2_YUV2RGB = -208
    DEFAULT_MATRIX_R2C0_YUV2RGB = 298
    DEFAULT_MATRIX_R2C1_YUV2RGB = 0
    DEFAULT_MATRIX_R2C2_YUV2RGB = 409

    DEFAULT_OUTPUT_BIAS_0 = 16
    DEFAULT_OUTPUT_BIAS_1 = 128
    DEFAULT_OUTPUT_BIAS_2 = 128
    DEFAULT_INPUT_BIAS_0 = 16
    DEFAULT_INPUT_BIAS_1 = 128
    DEFAULT_INPUT_BIAS_2 = 128
    DEFAULT_VAR_RECI_CHN = 1.0

    DYNC_PARAM_HEAD_STRUCT_SIZE = 64
    DYNC_PARAM_BATCH_STRUCT_SIZE = 96
    DYNC_PARAM_BATCH_SIZE_MAX = 32
    DYNC_PARAM_SIZE = DYNC_PARAM_HEAD_STRUCT_SIZE + \
                    DYNC_PARAM_BATCH_STRUCT_SIZE

    HEAD_OFFSET_INPUT_FORMAT = 0  # uint8
    HEAD_OFFSET_CSC_SWITCH = 1  # int8
    HEAD_OFFSET_RBUV_SWAP_SWITCH = 2  # int8
    HEAD_OFFSET_AX_SWAP_SWITCH = 3  # int8
    HEAD_OFFSET_BATCHNUM = 4  # int8
    HEAD_OFFSET_SRCIMAGE_W = 8  # int32
    HEAD_OFFSET_SRCIMAGE_H = 12  # int32
    HEAD_OFFSET_CSC_MATRIX_R0C0 = 16  # int16
    HEAD_OFFSET_CSC_MATRIX_R0C1 = 18  # int16
    HEAD_OFFSET_CSC_MATRIX_R0C2 = 20  # int16
    HEAD_OFFSET_CSC_MATRIX_R1C0 = 22  # int16
    HEAD_OFFSET_CSC_MATRIX_R1C1 = 24  # int16
    HEAD_OFFSET_CSC_MATRIX_R1C2 = 26  # int16
    HEAD_OFFSET_CSC_MATRIX_R2C0 = 28  # int16
    HEAD_OFFSET_CSC_MATRIX_R2C1 = 30  # int16
    HEAD_OFFSET_CSC_MATRIX_R2C2 = 32  # int16
    HEAD_OFFSET_CSC_OUTPUT_BIAS_R0 = 40  # uint8
    HEAD_OFFSET_CSC_OUTPUT_BIAS_R1 = 41  # uint8
    HEAD_OFFSET_CSC_OUTPUT_BIAS_R2 = 42  # uint8
    HEAD_OFFSET_CSC_INPUT_BIAS_R0 = 43  # uint8
    HEAD_OFFSET_CSC_INPUT_BIAS_R1 = 44  # uint8
    HEAD_OFFSET_CSC_INPUT_BIAS_R2 = 45  # uint8

    BATCH_OFFSET_CROP_SWITCH = 0 # int8
    BATCH_OFFSET_SCF_SWITCH = 1 # int8
    BATCH_OFFSET_PAD_SWITCH = 2 # int8
    BATCH_OFFSET_ROTATE_SWITCH = 3 # int8
    BATCH_OFFSET_CROP_STARTPOS_W = 8 # int32
    BATCH_OFFSET_CROP_STARTPOS_H = 12 # int32
    BATCH_OFFSET_CROP_W = 16 # int32
    BATCH_OFFSET_CROP_H = 20 # int32
    BATCH_OFFSET_SCF_INPUT_W = 24 # int32
    BATCH_OFFSET_SCF_INPUT_H = 28 # int32
    BATCH_OFFSET_SCF_OUTPUT_W = 32 # int32
    BATCH_OFFSET_SCF_OUTPUT_H = 36 # int32
    BATCH_OFFSET_PAD_TOP = 40 # int32
    BATCH_OFFSET_PAD_BOTTOM = 44 # int32
    BATCH_OFFSET_PAD_LEFT = 48 # int32
    BATCH_OFFSET_PAD_RIGHT = 52 # int32
    BATCH_OFFSET_DTC_MEAN_C0 = 56 # int16
    BATCH_OFFSET_DTC_MEAN_C1 = 58 # int16
    BATCH_OFFSET_DTC_MEAN_C2 = 60 # int16
    BATCH_OFFSET_DTC_MEAN_C3 = 62 # int16
    BATCH_OFFSET_DTC_MIN_C0 = 64 # uint16
    BATCH_OFFSET_DTC_MIN_C1 = 66 # uint16
    BATCH_OFFSET_DTC_MIN_C2 = 68 # uint16
    BATCH_OFFSET_DTC_MIN_C3 = 70 # uint16
    BATCH_OFFSET_DTC_VAR_C0 = 72 # uint16
    BATCH_OFFSET_DTC_VAR_C1 = 74 # uint16
    BATCH_OFFSET_DTC_VAR_C2 = 76 # uint16
    BATCH_OFFSET_DTC_VAR_C3 = 78 # uint16

    AIPP_OP_ERROR_CODE = 'E81012'

    STC_AIPP_SUPPORT_SOC_VERSION_SET = ("Ascend310", "Ascend910", "Ascend610", "Ascend310P",
                                        "BS9SX1A", "Hi3796CV300ES", "Hi3796CV300CS", "SD3403",
                                        "Ascend910B", "Ascend910_93", "Ascend310B",
                                        "AS31XM1", "Ascend610Lite", "BS9SX2A", "MC61AM21A")

    DYN_AIPP_SUPPORT_SOC_VERSION_SET = ("Ascend310", "Ascend910", "Ascend610", "Ascend310P", "BS9SX1A",
                                    "Hi3796CV300ES", "Hi3796CV300CS", "SD3403")

    C04_AIPP_SUPPORT_SOC_VERSION_SET = ("Ascend610", "Ascend310P", "BS9SX1A", "Hi3796CV300CS",
                                        "SD3403", "Ascend910B", "Ascend910_93", "Ascend310B",
                                        "AS31XM1", "Ascend610Lite", "BS9SX2A", "MC61AM21A")

    SUPPORT_IMAGE_FORMAT_MAP = {
        "Ascend310": ("YUV420SP_U8", "XRGB8888_U8", "NC1HWC0DI_FP16",
                    "NC1HWC0DI_S8", "RGB888_U8", "YUV400_U8"),
        "Ascend310B": ("YUV420SP_U8", "XRGB8888_U8", "RGB888_U8", "YUV400_U8",
                       "RAW8", "RAW10", "RAW12", "RAW14", "RAW16"),
        "AS31XM1": ("YUV420SP_U8", "XRGB8888_U8", "RGB888_U8", "YUV400_U8",
                       "RAW8", "RAW10", "RAW12", "RAW14", "RAW16"),
        "Ascend610Lite": ("YUV420SP_U8", "XRGB8888_U8", "RGB888_U8", "YUV400_U8",
                          "ARGB8888_U8", "YUV422SP_U8"),
        "BS9SX2A": ("YUV420SP_U8", "XRGB8888_U8", "RGB888_U8", "YUV400_U8",
                    "ARGB8888_U8", "YUV422SP_U8"),
        "MC61AM21A": ("YUV420SP_U8", "XRGB8888_U8", "RGB888_U8", "YUV400_U8",
                      "ARGB8888_U8", "YUV422SP_U8"),
        "Ascend910": ("YUV420SP_U8", "XRGB8888_U8", "RGB888_U8", "YUV400_U8"),
        "Ascend910B": ("YUV420SP_U8", "XRGB8888_U8", "RGB888_U8", "YUV400_U8"),
        "Ascend610": ("YUV420SP_U8", "XRGB8888_U8", "NC1HWC0DI_FP16", "NC1HWC0DI_S8",
                    "RGB888_U8", "YUV400_U8"),
        "Ascend310P": ("YUV420SP_U8", "XRGB8888_U8", "NC1HWC0DI_FP16", "NC1HWC0DI_S8",
                    "RGB888_U8", "YUV400_U8"),
        "BS9SX1A": ("YUV420SP_U8", "XRGB8888_U8", "NC1HWC0DI_FP16", "NC1HWC0DI_S8",
                    "RGB888_U8", "YUV400_U8", "RGB16", "RGB20", "RGB24", "RGB8_IR",
                    "RGB16_IR", "RGB24_IR"),
        "Hi3796CV300ES-Hi3796CV300CS-SD3403": (
                "YUV420SP_U8", "RGB888_U8", "XRGB8888_U8", "ARGB8888_U8", "YUYV_U8",
                "YUV422SP_U8", "AYUV444_U8", "YUV400_U8", "RAW10", "RAW12", "RAW16",
                "uint16")
    }

    V300_SOC_VERSION_LIST = ("Ascend310B", "AS31XM1", "Ascend610Lite", "BS9SX2A", "MC61AM21A")

    ODD_CROP_SUPPORT_SOC_VERSION_LIST = ("Ascend910B", "Ascend910_93", "Ascend310B",
                                         "AS31XM1", "Ascend610Lite", "BS9SX2A", "MC61AM21A")


def get_fp16(value):
    """
    :param value:
    :return:
    """
    if value != 0:
        data = numpy.float16(value).tobytes()
        result = int((data[0] & 0xff) | (data[1] & 0xff) << 8)
        return result

    return 0


def get_bin_value_from_fp32(value):
    """
    Description:
        Convert input value of fp32 to binary form, and get the int value of it.
    :param value: float form of given num
    :return: int value of the binary form
    """
    if value != 0:
        data = numpy.float32(value).tobytes()
        result = int((data[0] & 0xff) | (data[1] & 0xff) << 8 | (data[2] & 0xff) << 16 | (data[3] & 0xff) << 24)
        return result

    return 0


def convert_fp16_to_fp32(value):
    """
    Description:
        Convert input binary value of fp16 dtype to binary value of fp32 dtype.
        fp16: 1 bit (sign) + 5 bit (exponent) + 10 bit (fraction)
        fp32: 1 bit (sign) + 8 bit (exponent) + 23 bit (fraction)
    :param value: binary value of fp16 dtype, type: tvm.tir.Load
    :return: binary value of fp32 dtype, type: tvm.tir.Load
    """
    sign_mask = 0x8000
    exponent_mask = 0x7c00
    fraction_mask = 0x03ff
    sign = value & sign_mask
    # exponent_fp32  - 127 = exponent_fp16 - 15, exponent_diff = 127 - 15 = 112
    exponent_diff = tvm.const(112, dtype="uint64")
    exponent = ((value & exponent_mask) >> 10) + exponent_diff
    fraction = value & fraction_mask

    return tvm.const(0, dtype="uint64") | (sign << 16) | (exponent << 23) | (fraction << 13)


def get_l1_image_buf_max(actual_col_size, dtype, is_dynamic, output_format="NC1HWC0"):
    """
    :param actual_col_size:
    :param dtype:
    :param is_dynamic:
    :return:
    """
    if dtype == "float16":
        size = 2
        c0 = 16
    else:
        size = 1
        c0 = 32

    if output_format == "NC1HWC0_C04":
        c0 = 4

    l1_size = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)
    ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    if is_dynamic:
        ub_size -= (Const.DYNC_PARAM_SIZE + 1024 - 1) // 1024 * 1024

    buffer_upper_limit = l1_size
    if l1_size >= ub_size:
        buffer_upper_limit = ub_size

    if output_format == "NC1HWC0_C04":
        buffer_upper_limit = (buffer_upper_limit - 64) // size // c0
    else:
        buffer_upper_limit = buffer_upper_limit // size // c0

    if is_dynamic:
        return buffer_upper_limit

    if actual_col_size >= buffer_upper_limit:
        return buffer_upper_limit

    return actual_col_size


def _set_spr2_spr4_lhisi_dync_by_yuv(ib, dtype, spr, p_ub_buf, tmp):
    """
    :param ib:
    :param dtype:
    :param spr:
    :param p_ub_buf:
    :param tmp:
    :return:
    """

    ib.emit(tvm.call_extern("int8",  # actual data type
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_SWITCH)))
    #enable csc
    with ib.if_scope(tmp[0] == 1):
        # spr2
        ib.emit(tvm.call_extern("int16",  # actual data type
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R2C0)))
        spr[2] = ((tmp[0] * tvm.const(4, dtype="uint64")) & 0xffff)

        ib.emit(tvm.call_extern("int16",  # actual data type
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R2C1)))
        spr[2] = spr[2] | ((tmp[0] * tvm.const(4, dtype="uint64")) & 0xffff) << 16

        ib.emit(tvm.call_extern("int16",  # actual data type
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R2C2)))
        spr[2] = spr[2] | ((tmp[0] * tvm.const(4, dtype="uint64")) & 0xffff) << 32

        ib.emit(tvm.call_extern("int16",  # actual data type
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R1C0)))
        spr[2] = spr[2] | ((tmp[0] * tvm.const(4, dtype="uint64")) & 0xffff) << 48
        ib.emit(tvm.call_extern(dtype, "set_aipp_spr_2", spr[2]))

        # spr3
        ib.emit(tvm.call_extern("int16",  # actual data type
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R1C1)))
        spr[3] = ((tmp[0] * tvm.const(4, dtype="uint64")) & 0xffff)

        ib.emit(tvm.call_extern("int16",  # actual data type
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R1C2)))
        spr[3] = spr[3] | ((tmp[0] * tvm.const(4, dtype="uint64")) & 0xffff) << 16

        ib.emit(tvm.call_extern("int16",  # actual data type
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R0C0)))
        spr[3] = spr[3] | ((tmp[0] * tvm.const(4, dtype="uint64")) & 0xffff) << 32

        ib.emit(tvm.call_extern("int16",  # actual data type
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R0C1)))
        spr[3] = spr[3] | ((tmp[0] * tvm.const(4, dtype="uint64")) & 0xffff) << 48
        ib.emit(tvm.call_extern(dtype, "set_aipp_spr_3", spr[3]))

        # spr4
        ib.emit(tvm.call_extern("int16",  # actual data type
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R0C2)))
        spr[4] = ((tmp[0] * tvm.const(4, dtype="uint64")) & 0xffff)

        ib.emit(tvm.call_extern("uint8",  # actual data type
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_INPUT_BIAS_R2)))
        spr[4] = spr[4] | (tmp[0] & 0xff) << 40

        ib.emit(tvm.call_extern("uint8",  # actual data type
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_INPUT_BIAS_R1)))
        spr[4] = spr[4] | (tmp[0] & 0xff) << 48

        ib.emit(tvm.call_extern("uint8",  # actual data type
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_INPUT_BIAS_R0)))
        spr[4] = spr[4] | (tmp[0] & 0xff) << 56
        ib.emit(tvm.call_extern(dtype, "set_aipp_spr_4", spr[4]))
    with ib.else_scope():
        # spr_2->bits.csc_matrix_r0_c2 = (uint16_t)(1<<10)
        spr2 = 0
        spr2 = spr2 | (1 << 10 & 0xffff) << 32
        ib.emit(tvm.call_extern(dtype, "set_aipp_spr_2",
                                tvm.const(spr2, dtype="uint64")))

        spr3 = 0
        # spr_3->bits.csc_matrix_r1_c1
        spr3 = spr3 | (1 << 10 & 0xffff)
        # spr_3->bits.csc_matrix_r2_c0
        spr3 = spr3 | (1 << 10 & 0xffff) << 32
        ib.emit(tvm.call_extern(dtype, "set_aipp_spr_3",
                                tvm.const(spr3, dtype="uint64")))

        spr4 = 0
        ib.emit(tvm.call_extern(dtype, "set_aipp_spr_4", tvm.const(spr4, dtype="uint64")))


def _set_spr2_spr4_lhisi_dync_by_rgb(ib, dtype, spr, p_ub_buf, tmp):
    """
    :param ib:
    :param dtype:
    :param spr:
    :param p_ub_buf:
    :param tmp:
    :return:
    """

    ib.emit(tvm.call_extern("int8",  # actual data type
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_SWITCH)))
    #enable csc
    with ib.if_scope(tmp[0] == 1):
        # spr2
        ib.emit(tvm.call_extern("int16",  # actual data type
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R2C2)))
        spr[2] = ((tmp[0] * tvm.const(4, dtype="uint64")) & 0xffff)

        ib.emit(tvm.call_extern("int16",  # actual data type
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R2C1)))
        spr[2] = spr[2] | ((tmp[0] * tvm.const(4, dtype="uint64")) & 0xffff) << 16

        ib.emit(tvm.call_extern("int16",  # actual data type
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R2C0)))
        spr[2] = spr[2] | ((tmp[0] * tvm.const(4, dtype="uint64")) & 0xffff) << 32

        ib.emit(tvm.call_extern("int16",  # actual data type
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R1C2)))
        spr[2] = spr[2] | ((tmp[0] * tvm.const(4, dtype="uint64")) & 0xffff) << 48
        ib.emit(tvm.call_extern(dtype, "set_aipp_spr_2", spr[2]))

        # spr3
        ib.emit(tvm.call_extern("int16",  # actual data type
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R1C1)))
        spr[3] = ((tmp[0] * tvm.const(4, dtype="uint64")) & 0xffff)

        ib.emit(tvm.call_extern("int16",  # actual data type
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R1C0)))
        spr[3] = spr[3] | ((tmp[0] * tvm.const(4, dtype="uint64")) & 0xffff) << 16

        ib.emit(tvm.call_extern("int16",  # actual data type
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R0C2)))
        spr[3] = spr[3] | ((tmp[0] * tvm.const(4, dtype="uint64")) & 0xffff) << 32

        ib.emit(tvm.call_extern("int16",  # actual data type
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R0C1)))
        spr[3] = spr[3] | ((tmp[0] * tvm.const(4, dtype="uint64")) & 0xffff) << 48
        ib.emit(tvm.call_extern(dtype, "set_aipp_spr_3", spr[3]))

        # spr4
        ib.emit(tvm.call_extern("int16",  # actual data type
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R0C0)))
        spr[4] = ((tmp[0] * tvm.const(4, dtype="uint64")) & 0xffff)

        ib.emit(tvm.call_extern("uint8",  # actual data type
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_OUTPUT_BIAS_R0)))
        spr[4] = spr[4] | (tmp[0] & 0xff) << 16

        ib.emit(tvm.call_extern("uint8",  # actual data type
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_OUTPUT_BIAS_R1)))
        spr[4] = spr[4] | (tmp[0] & 0xff) << 24

        ib.emit(tvm.call_extern("uint8",  # actual data type
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_OUTPUT_BIAS_R2)))
        spr[4] = spr[4] | (tmp[0] & 0xff) << 32
        ib.emit(tvm.call_extern(dtype, "set_aipp_spr_4", spr[4]))


def set_spr_dync(ib, param_buf, dtype, cur_cce_product):
    """
    :param ib:
    :param param_buf:
    :param dtype:
    :param cur_cce_product:
    :return:
    """

    p_ub = ib.allocate("uint8", (Const.DYNC_PARAM_SIZE,), "p_ub", scope=tbe_platform.scope_ubuf)
    p_ub_buf = tvm.decl_buffer((Const.DYNC_PARAM_SIZE,), "uint8", "p_ub_buf",
                               scope=tbe_platform.scope_ubuf, data=p_ub)

    ib.emit(tvm.call_extern("uint8", 'copy_gm_to_ubuf',
                            p_ub_buf.access_ptr("w", ptr_type=dtype, offset=0),
                            param_buf.access_ptr("rw", ptr_type=dtype, offset=0),
                            0, 1, Const.DYNC_PARAM_SIZE//32, 0, 0))

    spr = ib.allocate("uint64", [17], name="spr", scope=tbe_platform.scope_reg)
    tmp = ib.allocate("uint64", [1], name="tmp", scope=tbe_platform.scope_reg)

    input_format_tmp = ib.allocate("uint64", [1], name="input_format_tmp", scope=tbe_platform.scope_reg)
    input_format_tmp[0] = tvm.const(0, dtype="uint64")
    ib.emit(tvm.call_extern("uint8",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_INPUT_FORMAT)))

    input_format_tmp[0] = tmp[0]
    if cur_cce_product in ["Hi3796CV300ES", "Hi3796CV300CS", "SD3403"]:
        # the value of input_format:YUV420SP_U8
        with ib.if_scope(input_format_tmp[0] == 1):
            _set_spr2_spr4_lhisi_dync_by_yuv(ib, dtype, spr, p_ub_buf, tmp)
        # the value of input_format:YUYV_U8
        with ib.if_scope(input_format_tmp[0] == 7):
            _set_spr2_spr4_lhisi_dync_by_yuv(ib, dtype, spr, p_ub_buf, tmp)
        # YUV422SP_U8
        with ib.if_scope(input_format_tmp[0] == 8):
            _set_spr2_spr4_lhisi_dync_by_yuv(ib, dtype, spr, p_ub_buf, tmp)
        # AYUV444_U8
        with ib.if_scope(input_format_tmp[0] == 9):
            _set_spr2_spr4_lhisi_dync_by_yuv(ib, dtype, spr, p_ub_buf, tmp)
        # XRGB8888_U8
        with ib.if_scope(input_format_tmp[0] == 2):
            _set_spr2_spr4_lhisi_dync_by_rgb(ib, dtype, spr, p_ub_buf, tmp)
        # RGB888_U8
        with ib.if_scope(input_format_tmp[0] == 5):
            _set_spr2_spr4_lhisi_dync_by_rgb(ib, dtype, spr, p_ub_buf, tmp)
        # ARGB8888
        with ib.if_scope(input_format_tmp[0] == 6):
            _set_spr2_spr4_lhisi_dync_by_rgb(ib, dtype, spr, p_ub_buf, tmp)
    else:
        # spr2
        ib.emit(tvm.call_extern("int16",
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R0C0)))
        spr[2] = (tmp[0] & 0xffff)

        ib.emit(tvm.call_extern("int16",
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R0C1)))
        spr[2] = spr[2] | (tmp[0] & 0xffff) << 16

        ib.emit(tvm.call_extern("int16",
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R0C2)))
        spr[2] = spr[2] | (tmp[0] & 0xffff) << 32

        ib.emit(tvm.call_extern("int16",
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R1C0)))
        spr[2] = spr[2] | (tmp[0] & 0xffff) << 48
        ib.emit(tvm.call_extern(dtype, "set_aipp_spr_2", spr[2]))

        # spr3
        ib.emit(tvm.call_extern("int16",
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R1C1)))
        spr[3] = (tmp[0] & 0xffff)

        ib.emit(tvm.call_extern("int16",
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R1C2)))
        spr[3] = spr[3] | (tmp[0] & 0xffff) << 16

        ib.emit(tvm.call_extern("int16",
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R2C0)))
        spr[3] = spr[3] | (tmp[0] & 0xffff) << 32

        ib.emit(tvm.call_extern("int16",
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R2C1)))
        spr[3] = spr[3] | (tmp[0] & 0xffff) << 48
        ib.emit(tvm.call_extern(dtype, "set_aipp_spr_3", spr[3]))

        # spr4
        ib.emit(tvm.call_extern("int16",
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R2C2)))
        spr[4] = (tmp[0] & 0xffff)

        ib.emit(tvm.call_extern("uint8",
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_OUTPUT_BIAS_R0)))
        spr[4] = spr[4] | (tmp[0] & 0xff) << 16

        ib.emit(tvm.call_extern("uint8",
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_OUTPUT_BIAS_R1)))
        spr[4] = spr[4] | (tmp[0] & 0xff) << 24

        ib.emit(tvm.call_extern("uint8",
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_OUTPUT_BIAS_R2)))
        spr[4] = spr[4] | (tmp[0] & 0xff) << 32

        ib.emit(tvm.call_extern("uint8",
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_INPUT_BIAS_R0)))
        spr[4] = spr[4] | (tmp[0] & 0xff) << 40

        ib.emit(tvm.call_extern("uint8",
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_INPUT_BIAS_R1)))
        spr[4] = spr[4] | (tmp[0] & 0xff) << 48

        ib.emit(tvm.call_extern("uint8",
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_INPUT_BIAS_R2)))
        spr[4] = spr[4] | (tmp[0] & 0xff) << 56
        ib.emit(tvm.call_extern(dtype, "set_aipp_spr_4", spr[4]))

    # spr8
    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_8", spr[8]))

    # spr9
    ib.emit(tvm.call_extern("int8",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_RBUV_SWAP_SWITCH)))
    # YUV400
    with ib.if_scope(input_format_tmp[0] != 10):
        spr[9] = (tmp[0] & 0x1) << 16
        spr[9] = spr[9] | (tmp[0] & 0x1) << 17

    # XRGB8888_U8
    with ib.if_scope(input_format_tmp[0] == 2):
        ib.emit(tvm.call_extern("int8",
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_AX_SWAP_SWITCH)))
        spr[9] = spr[9] | (tmp[0] & 0x1) << 18
    # ARGB8888_U8
    with ib.if_scope(input_format_tmp[0] == 6):
        ib.emit(tvm.call_extern("int8",
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_AX_SWAP_SWITCH)))
        spr[9] = spr[9] | (tmp[0] & 0x1) << 18
    # AYUV444_U8
    with ib.if_scope(input_format_tmp[0] == 9):
        ib.emit(tvm.call_extern("int8",
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_AX_SWAP_SWITCH)))
        spr[9] = spr[9] | (tmp[0] & 0x1) << 18

    ib.emit(tvm.call_extern("uint8",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            p_ub_buf.access_ptr('r', offset=Const.HEAD_OFFSET_INPUT_FORMAT)))
    spr[9] = spr[9] | ((tmp[0] - tvm.const(1, dtype="uint64")) & 0xf) << 19
    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_9", spr[9]))

    return p_ub_buf, spr, tmp


def set_spr_dync_from_gm(ib, param_buf, dtype):
    """
    :param ib:
    :param param_buf:
    :param dtype:
    :param cur_cce_product:
    :return:
    """
    spr = ib.allocate("uint64", [22], name="spr", scope=tbe_platform.scope_reg)
    tmp = ib.allocate("uint64", [1], name="tmp", scope=tbe_platform.scope_reg)

    input_format_tmp = ib.allocate("uint64", [1], name="input_format_tmp", scope=tbe_platform.scope_reg)
    input_format_tmp[0] = tvm.const(0, dtype="uint64")
    ib.emit(tvm.call_extern("uint8",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=Const.HEAD_OFFSET_INPUT_FORMAT)))

    input_format_tmp[0] = tmp[0]

    # spr2
    ib.emit(tvm.call_extern("int16",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R0C0)))
    spr[2] = (tmp[0] & 0xffff)

    ib.emit(tvm.call_extern("int16",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R0C1)))
    spr[2] = spr[2] | (tmp[0] & 0xffff) << 16

    ib.emit(tvm.call_extern("int16",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R0C2)))
    spr[2] = spr[2] | (tmp[0] & 0xffff) << 32

    ib.emit(tvm.call_extern("int16",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R1C0)))
    spr[2] = spr[2] | (tmp[0] & 0xffff) << 48
    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_2", spr[2]))

    # spr3
    ib.emit(tvm.call_extern("int16",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R1C1)))
    spr[3] = (tmp[0] & 0xffff)

    ib.emit(tvm.call_extern("int16",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R1C2)))
    spr[3] = spr[3] | (tmp[0] & 0xffff) << 16

    ib.emit(tvm.call_extern("int16",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R2C0)))
    spr[3] = spr[3] | (tmp[0] & 0xffff) << 32

    ib.emit(tvm.call_extern("int16",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R2C1)))
    spr[3] = spr[3] | (tmp[0] & 0xffff) << 48
    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_3", spr[3]))

    # spr4
    ib.emit(tvm.call_extern("int16",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_MATRIX_R2C2)))
    spr[4] = (tmp[0] & 0xffff)

    ib.emit(tvm.call_extern("uint8",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_OUTPUT_BIAS_R0)))
    spr[4] = spr[4] | (tmp[0] & 0xff) << 16

    ib.emit(tvm.call_extern("uint8",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_OUTPUT_BIAS_R1)))
    spr[4] = spr[4] | (tmp[0] & 0xff) << 24

    ib.emit(tvm.call_extern("uint8",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_OUTPUT_BIAS_R2)))
    spr[4] = spr[4] | (tmp[0] & 0xff) << 32

    ib.emit(tvm.call_extern("uint8",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_INPUT_BIAS_R0)))
    spr[4] = spr[4] | (tmp[0] & 0xff) << 40

    ib.emit(tvm.call_extern("uint8",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_INPUT_BIAS_R1)))
    spr[4] = spr[4] | (tmp[0] & 0xff) << 48

    ib.emit(tvm.call_extern("uint8",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=Const.HEAD_OFFSET_CSC_INPUT_BIAS_R2)))
    spr[4] = spr[4] | (tmp[0] & 0xff) << 56
    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_4", spr[4]))

    # spr8
    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_8", spr[8]))

    # spr9
    ib.emit(tvm.call_extern("int8",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=Const.HEAD_OFFSET_RBUV_SWAP_SWITCH)))
    # YUV400
    with ib.if_scope(input_format_tmp[0] != 10):
        spr[9] = (tmp[0] & 0x1) << 16
        spr[9] = spr[9] | (tmp[0] & 0x1) << 17

    # XRGB8888_U8
    with ib.if_scope(input_format_tmp[0] == 2):
        ib.emit(tvm.call_extern("int8",
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                param_buf.access_ptr('r', offset=Const.HEAD_OFFSET_AX_SWAP_SWITCH)))
        spr[9] = spr[9] | (tmp[0] & 0x1) << 18
    # ARGB8888_U8
    with ib.if_scope(input_format_tmp[0] == 6):
        ib.emit(tvm.call_extern("int8",
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                param_buf.access_ptr('r', offset=Const.HEAD_OFFSET_AX_SWAP_SWITCH)))
        spr[9] = spr[9] | (tmp[0] & 0x1) << 18
    # AYUV444_U8
    with ib.if_scope(input_format_tmp[0] == 9):
        ib.emit(tvm.call_extern("int8",
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                param_buf.access_ptr('r', offset=Const.HEAD_OFFSET_AX_SWAP_SWITCH)))
        spr[9] = spr[9] | (tmp[0] & 0x1) << 18

    ib.emit(tvm.call_extern("uint8",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=Const.HEAD_OFFSET_INPUT_FORMAT)))
    spr[9] = spr[9] | ((tmp[0] - tvm.const(1, dtype="uint64")) & 0xf) << 19
    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_9", spr[9]))

    return spr, tmp


def get_dync_padding_size(ib, param_buf, tmp, padding_info,
    offset=Const.DYNC_PARAM_HEAD_STRUCT_SIZE):
    """
    :param aipp_config:
    :return:
    """

    ib.emit(tvm.call_extern("int8",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=offset +
                                                 Const.BATCH_OFFSET_PAD_SWITCH)))
    with ib.if_scope(tmp[0] > 0):
        ib.emit(tvm.call_extern("int32",
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                param_buf.access_ptr('r', offset=offset +
                                                    Const.BATCH_OFFSET_PAD_TOP)))
        padding_info[0] = tmp[0]

        ib.emit(tvm.call_extern("int32",
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                param_buf.access_ptr('r', offset=offset +
                                                     Const.BATCH_OFFSET_PAD_BOTTOM)))
        padding_info[1] = tmp[0]

        ib.emit(tvm.call_extern("int32",
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                param_buf.access_ptr('r', offset=offset +
                                                     Const.BATCH_OFFSET_PAD_LEFT)))
        padding_info[2] = tmp[0]

        ib.emit(tvm.call_extern("int32",
                                "reg_mov",
                                tvm.call_extern("uint64", "reg", tmp[0]),
                                param_buf.access_ptr('r', offset=offset +
                                                     Const.BATCH_OFFSET_PAD_RIGHT)))
        padding_info[3] = tmp[0]


def get_dync_src_image_size(ib, param_buf, tmp, src_image_size):
    """
    :param ib:
    :param p_ub_buf:
    :param tmp:
    :param src_image_size:
    :return:
    """

    ib.emit(tvm.call_extern("int32", "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=Const.HEAD_OFFSET_SRCIMAGE_H)))
    with ib.if_scope(tmp[0] > 0):
        src_image_size[0] = tmp[0]

    ib.emit(tvm.call_extern("int32", "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=Const.HEAD_OFFSET_SRCIMAGE_W)))
    with ib.if_scope(tmp[0] > 0):
        src_image_size[1] = tmp[0]


def get_dync_crop_info(ib, param_buf, tmp, load_image_info,
    offset=Const.DYNC_PARAM_HEAD_STRUCT_SIZE):
    """
    :param ib:
    :param p_ub_buf:
    :param tmp:
    :param load_image_info:
    :return:
    """

    ib.emit(tvm.call_extern("int32", "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=offset + \
                                                            Const.BATCH_OFFSET_CROP_STARTPOS_W)))
    # load_start_pos_w
    load_image_info[1] = tmp[0]

    ib.emit(tvm.call_extern("int32", "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=offset + \
                                                            Const.BATCH_OFFSET_CROP_STARTPOS_H)))
    # load_start_pos_h
    load_image_info[0] = tmp[0]

    ib.emit(tvm.call_extern("int32", "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=offset + \
                                                            Const.BATCH_OFFSET_CROP_W)))
    # load_image_w
    load_image_info[3] = tmp[0]

    ib.emit(tvm.call_extern("int32", "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=offset + \
                                                            Const.BATCH_OFFSET_CROP_H)))
    # load_image_h
    load_image_info[2] = tmp[0]


def get_dync_actual_col_size(out_h, out_w, padding_info, actual_col_size_reg, support_vertical_padding):
    """
    :param out_h: height of aipp output image.
    :param out_w: width of aipp output image.
    :param padding_info: buffer for padding info, include top_padding_size, bottom_padding_size,
                         left_padding_size, right_padding_size
    :param actual_col_size_reg: buffer for actual size for aipp output image
    :param support_vertical_padding: flag for support vertical padding, True if support, otherwise False.
    :return: None.
    """
    if support_vertical_padding:
        actual_col_size_reg[0] = tvm.const(out_h, dtype="uint64") * tvm.const(out_w, dtype="uint64")
    else:
        out_h_tmp = out_h - padding_info[0] - padding_info[1]
        actual_col_size_reg[0] = get_const(out_h_tmp) * tvm.const(out_w, dtype="uint64")


def set_spr_dync_in_batch(ib, dtype, param_buf, spr, tmp,
    is_hisi_yuv400=False, offset=Const.DYNC_PARAM_HEAD_STRUCT_SIZE):
    """
    set_spr_dync_in_batch
    """
    chn_0_position = 0
    chn_2_position = 32
    if is_hisi_yuv400:
        chn_0_position = 32
        chn_2_position = 0

    # spr5
    ib.emit(tvm.call_extern("int16",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=offset + \
                                                            Const.BATCH_OFFSET_DTC_MEAN_C0)))
    spr[5] = (tmp[0] & 0xffff) << chn_0_position

    ib.emit(tvm.call_extern("int16",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=offset + \
                                                            Const.BATCH_OFFSET_DTC_MEAN_C1)))
    spr[5] = spr[5] | (tmp[0] & 0xffff) << 16

    ib.emit(tvm.call_extern("int16",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=offset + \
                                                            Const.BATCH_OFFSET_DTC_MEAN_C2)))
    spr[5] = spr[5] | (tmp[0] & 0xffff) << chn_2_position

    ib.emit(tvm.call_extern("int16",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=offset + \
                                                            Const.BATCH_OFFSET_DTC_MEAN_C3)))
    spr[5] = spr[5] | (tmp[0] & 0xffff) << 48
    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_5", spr[5]))

    # spr6
    ib.emit(tvm.call_extern("uint16",  # actual data type
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=offset + \
                                                            Const.BATCH_OFFSET_DTC_MIN_C0)))
    spr[6] = (tmp[0] & 0xffff) << chn_0_position

    ib.emit(tvm.call_extern("uint16",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=offset + \
                                                            Const.BATCH_OFFSET_DTC_MIN_C1)))
    spr[6] = spr[6] | (tmp[0] & 0xffff) << 16

    ib.emit(tvm.call_extern("uint16",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=offset + \
                                                            Const.BATCH_OFFSET_DTC_MIN_C2)))
    spr[6] = spr[6] | (tmp[0] & 0xffff) << chn_2_position

    ib.emit(tvm.call_extern("uint16",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=offset + \
                                                            Const.BATCH_OFFSET_DTC_MIN_C3)))
    spr[6] = spr[6] | (tmp[0] & 0xffff) << 48
    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_6", spr[6]))

    # spr7
    ib.emit(tvm.call_extern("uint16",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=offset + \
                                                            Const.BATCH_OFFSET_DTC_VAR_C0)))
    spr[7] = (tmp[0] & 0xffff) << chn_0_position

    ib.emit(tvm.call_extern("uint16",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=offset + \
                                                            Const.BATCH_OFFSET_DTC_VAR_C1)))
    spr[7] = spr[7] | (tmp[0] & 0xffff) << 16

    ib.emit(tvm.call_extern("uint16",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=offset + \
                                                            Const.BATCH_OFFSET_DTC_VAR_C2)))
    spr[7] = spr[7] | (tmp[0] & 0xffff) << chn_2_position

    ib.emit(tvm.call_extern("uint16",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=offset + \
                                                            Const.BATCH_OFFSET_DTC_VAR_C3)))
    spr[7] = spr[7] | (tmp[0] & 0xffff) << 48
    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_7", spr[7]))


def set_spr_dync_in_batch_v300(ib, dtype, param_buf, spr, tmp,
                               offset=Const.DYNC_PARAM_HEAD_STRUCT_SIZE):
    """
    set_spr_dync_in_batch_new
    """
    chn_offset = 32

    # spr18
    ib.emit(tvm.call_extern("uint16",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=offset + \
                                                            Const.BATCH_OFFSET_DTC_VAR_C0)))
    with ib.if_scope(tmp[0] != 0):
        spr[18] = convert_fp16_to_fp32(tmp[0])

    ib.emit(tvm.call_extern("uint16",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=offset + \
                                                            Const.BATCH_OFFSET_DTC_VAR_C1)))
    with ib.if_scope(tmp[0] != 0):
        spr[18] = spr[18] | convert_fp16_to_fp32(tmp[0]) << chn_offset
    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_18", spr[18]))

    # spr19
    ib.emit(tvm.call_extern("uint16",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=offset + \
                                                            Const.BATCH_OFFSET_DTC_VAR_C2)))
    with ib.if_scope(tmp[0] != 0):
        spr[19] = convert_fp16_to_fp32(tmp[0])

    ib.emit(tvm.call_extern("uint16",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=offset + \
                                                            Const.BATCH_OFFSET_DTC_VAR_C3)))
    with ib.if_scope(tmp[0] != 0):
        spr[19] = spr[19] | convert_fp16_to_fp32(tmp[0]) << chn_offset
    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_19", spr[19]))

    # spr20
    ib.emit(tvm.call_extern("int16",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=offset + \
                                                            Const.BATCH_OFFSET_DTC_MEAN_C0)))
    with ib.if_scope(tmp[0] != 0):
        spr[20] = convert_fp16_to_fp32(tmp[0])

    ib.emit(tvm.call_extern("int16",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=offset + \
                                                            Const.BATCH_OFFSET_DTC_MEAN_C1)))
    with ib.if_scope(tmp[0] != 0):
        spr[20] = spr[20] | convert_fp16_to_fp32(tmp[0]) << chn_offset
    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_20", spr[20]))

    ib.emit(tvm.call_extern("int16",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=offset + \
                                                            Const.BATCH_OFFSET_DTC_MEAN_C2)))
    with ib.if_scope(tmp[0] != 0):
        spr[21] =  convert_fp16_to_fp32(tmp[0])

    ib.emit(tvm.call_extern("int16",
                            "reg_mov",
                            tvm.call_extern("uint64", "reg", tmp[0]),
                            param_buf.access_ptr('r', offset=offset + \
                                                            Const.BATCH_OFFSET_DTC_MEAN_C3)))
    with ib.if_scope(tmp[0] != 0):
        spr[21] = spr[21] | convert_fp16_to_fp32(tmp[0]) << chn_offset
    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_21", spr[21]))


def get_spr9(aipp_config, dtype, output_format="NC1HWC0"):
    """
    :param aipp_config:
    :param dtype:
    :param output_format:
    :return:
    """
    spr9 = 0
    if 'cpadding_value' in aipp_config:
        if dtype == "float16":
            cpadding_value = get_fp16(float(aipp_config.get('cpadding_value', 0)))
        else:
            cpadding_value = aipp_config.get('cpadding_value', 0)
        spr9 = spr9 | (cpadding_value & 0xff)
    if 'rbuv_swap_switch' in aipp_config:
        if aipp_config.get('input_format') not in ("YUV400_U8", "RAW10", "RAW12", "RAW16"):
            spr9 = spr9 | (aipp_config.get('rbuv_swap_switch') & 0x1) << 16
            spr9 = spr9 | (aipp_config.get('rbuv_swap_switch') & 0x1) << 17
    if 'ax_swap_switch' in aipp_config:
        if aipp_config.get('input_format') in ("XRGB8888_U8", "ARGB8888_U8", "AYUV444_U8"):
            spr9 = spr9 | (aipp_config.get('ax_swap_switch') & 0x1) << 18
    if 'input_format' in aipp_config:
        if aipp_config.get('input_format') == "YUV420SP_U8":
            spr9 = spr9 | 0 << 19
        elif aipp_config.get('input_format') == "XRGB8888_U8":
            spr9 = spr9 | 1 << 19
        elif aipp_config.get('input_format') == "NC1HWC0DI_FP16":
            spr9 = spr9 | 2 << 19
        elif aipp_config.get('input_format') == "NC1HWC0DI_S8":
            spr9 = spr9 | 3 << 19
        elif aipp_config.get('input_format') == "RGB888_U8":
            spr9 = spr9 | 4 << 19
        elif aipp_config.get('input_format') == "ARGB8888_U8":
            spr9 = spr9 | 5 << 19
        elif aipp_config.get('input_format') == "YUYV_U8":
            spr9 = spr9 | 6 << 19
        elif aipp_config.get('input_format') == "YUV422SP_U8":
            spr9 = spr9 | 7 << 19
        elif aipp_config.get('input_format') == "AYUV444_U8":
            spr9 = spr9 | 8 << 19
        elif aipp_config.get('input_format') == "YUV400_U8":
            spr9 = spr9 | 9 << 19
        elif aipp_config.get('input_format') == "RAW10":
            spr9 = spr9 | 10 << 19
        elif aipp_config.get('input_format') == "RAW12":
            spr9 = spr9 | 11 << 19
        elif aipp_config.get('input_format') == "RAW16" or \
                aipp_config.get('input_format') == "uint16":
            spr9 = spr9 | 12 << 19
        elif aipp_config.get('input_format') == "RGB16":
            spr9 = spr9 | 18 << 19
        elif aipp_config.get('input_format') == "RGB20":
            spr9 = spr9 | 19 << 19
        elif aipp_config.get('input_format') == "RGB24":
            spr9 = spr9 | 20 << 19
        elif aipp_config.get('input_format') == "RGB8_IR":
            spr9 = spr9 | 21 << 19
        elif aipp_config.get('input_format') == "RGB16_IR":
            spr9 = spr9 | 22 << 19
        elif aipp_config.get('input_format') == "RGB24_IR":
            spr9 = spr9 | 23 << 19
    if 'single_line_mode' in aipp_config:
        spr9 = spr9 | (aipp_config.get('single_line_mode') & 0x1) << 24

    n = 8
    if 'raw_rgbir_to_f16_n' in aipp_config:
        n = aipp_config.get('raw_rgbir_to_f16_n')
    if aipp_config.get('input_format') in ("RAW10", "RAW12", "RAW16", "uint16") \
            and dtype == "float16":
        # [33:30]: raw_to_f16_n
        # the n = 8
        spr9 = spr9 | (n << 30)
        spr9 = spr9 | (1 << 35)

    if output_format == "NC1HWC0_C04":
        spr9 = spr9 | (1 << 40)

    if aipp_config.get('input_format') in ("RGB16", "RGB20", "RGB24",
                                           "RGB16_IR", "RGB24_IR"):
        spr9 = spr9 | (n << 30)

    if aipp_config.get('input_format') in ("RGB20", "RGB24", "RGB24_IR"):
        mean_chn_2 = aipp_config.get('mean_chn_2', 0)
        mean_chn_3 = aipp_config.get('mean_chn_3', 0)
        spr9 = spr9 | (((mean_chn_2 >> 16) & 0xff) << 48)
        spr9 = spr9 | (((mean_chn_3 >> 16) & 0xff) << 56)

    return spr9


def set_spr2_spr9(ib, aipp_config, dtype, cur_cce_product, output_format="NC1HWC0"):
    """
    :param ib:
    :param aipp_config:
    :param dtype:
    :param cur_cce_product:
    :return:
    """
    if cur_cce_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        spr2 = 0
        if aipp_config.get('input_format') in \
                ("YUV420SP_U8", "YUYV_U8", "YUV422SP_U8", "AYUV444_U8", "YUV400_U8"):
            if ('csc_switch' in aipp_config) and \
                    (aipp_config.get('csc_switch') == 1):
                if 'matrix_r2c0' in aipp_config:
                    spr2 = spr2 | (aipp_config.get('matrix_r2c0')*4) & 0xffff
                if 'matrix_r2c1' in aipp_config:
                    spr2 = spr2 | \
                           ((aipp_config.get('matrix_r2c1')*4) & 0xffff) << 16
                if 'matrix_r2c2' in aipp_config:
                    spr2 = spr2 | \
                           ((aipp_config.get('matrix_r2c2')*4) & 0xffff) << 32
                if 'matrix_r1c0' in aipp_config:
                    spr2 = spr2 | \
                           ((aipp_config.get('matrix_r1c0')*4) & 0xffff) << 48
                ib.emit(tvm.call_extern(dtype, "set_aipp_spr_2",
                                        tvm.const(spr2, dtype="uint64")))

                spr3 = 0
                if 'matrix_r1c1' in aipp_config:
                    spr3 = spr3 | ((aipp_config.get('matrix_r1c1')*4) & 0xffff)
                if 'matrix_r1c2' in aipp_config:
                    spr3 = spr3 | \
                           ((aipp_config.get('matrix_r1c2')*4) & 0xffff) << 16
                if 'matrix_r0c0' in aipp_config:
                    spr3 = spr3 | \
                           ((aipp_config.get('matrix_r0c0')*4) & 0xffff) << 32
                if 'matrix_r0c1' in aipp_config:
                    spr3 = spr3 | \
                           ((aipp_config.get('matrix_r0c1')*4) & 0xffff) << 48
                ib.emit(tvm.call_extern(dtype, "set_aipp_spr_3",
                                        tvm.const(spr3, dtype="uint64")))

                spr4 = 0
                if 'matrix_r0c2' in aipp_config:
                    spr4 = spr4 | ((aipp_config.get('matrix_r0c2')*4) & 0xffff)
                if 'input_bias_2' in aipp_config:
                    spr4 = spr4 | (aipp_config.get('input_bias_2') & 0xff) << 40
                if 'input_bias_1' in aipp_config:
                    spr4 = spr4 | (aipp_config.get('input_bias_1') & 0xff) << 48
                if 'input_bias_0' in aipp_config:
                    spr4 = spr4 | (aipp_config.get('input_bias_0') & 0xff) << 56
            else:
                # spr_2->bits.csc_matrix_r0_c2 = (uint16_t)(1<<10)
                spr2 = spr2 | (1 << 10 & 0xffff) << 32
                ib.emit(tvm.call_extern(dtype, "set_aipp_spr_2",
                                        tvm.const(spr2, dtype="uint64")))

                spr3 = 0
                # spr_3->bits.csc_matrix_r1_c1
                spr3 = spr3 | (1 << 10 & 0xffff)
                # spr_3->bits.csc_matrix_r2_c0
                spr3 = spr3 | (1 << 10 & 0xffff) << 32
                ib.emit(tvm.call_extern(dtype, "set_aipp_spr_3",
                                        tvm.const(spr3, dtype="uint64")))

                spr4 = 0
        else:
            if 'matrix_r2c2' in aipp_config:
                spr2 = spr2 | (aipp_config.get('matrix_r2c2')*4) & 0xffff
            if 'matrix_r2c1' in aipp_config:
                spr2 = spr2 | ((aipp_config.get('matrix_r2c1')*4) & 0xffff) << 16
            if 'matrix_r2c0' in aipp_config:
                spr2 = spr2 | ((aipp_config.get('matrix_r2c0')*4) & 0xffff) << 32
            if 'matrix_r1c2' in aipp_config:
                spr2 = spr2 | ((aipp_config.get('matrix_r1c2')*4) & 0xffff) << 48
            ib.emit(tvm.call_extern(dtype, "set_aipp_spr_2",
                                    tvm.const(spr2, dtype="uint64")))

            spr3 = 0
            if 'matrix_r1c1' in aipp_config:
                spr3 = spr3 | ((aipp_config.get('matrix_r1c1')*4) & 0xffff)
            if 'matrix_r1c0' in aipp_config:
                spr3 = spr3 | ((aipp_config.get('matrix_r1c0')*4) & 0xffff) << 16
            if 'matrix_r0c2' in aipp_config:
                spr3 = spr3 | ((aipp_config.get('matrix_r0c2')*4) & 0xffff) << 32
            if 'matrix_r0c1' in aipp_config:
                spr3 = spr3 | ((aipp_config.get('matrix_r0c1')*4) & 0xffff) << 48
            ib.emit(tvm.call_extern(dtype, "set_aipp_spr_3",
                                    tvm.const(spr3, dtype="uint64")))

            spr4 = 0
            if 'matrix_r0c0' in aipp_config:
                spr4 = spr4 | ((aipp_config.get('matrix_r0c0')*4) & 0xffff)
            if 'output_bias_0' in aipp_config:
                spr4 = spr4 | (aipp_config.get('output_bias_0') & 0xff) << 16
            if 'output_bias_1' in aipp_config:
                spr4 = spr4 | (aipp_config.get('output_bias_1') & 0xff) << 24
            if 'output_bias_2' in aipp_config:
                spr4 = spr4 | (aipp_config.get('output_bias_2') & 0xff) << 32
        ib.emit(tvm.call_extern(dtype, "set_aipp_spr_4",
                                tvm.const(spr4, dtype="uint64")))
    else:
        spr2 = 0
        if 'matrix_r0c0' in aipp_config:
            spr2 = spr2 | aipp_config.get('matrix_r0c0') & 0xffff
        if 'matrix_r0c1' in aipp_config:
            spr2 = spr2 | (aipp_config.get('matrix_r0c1') & 0xffff) << 16
        if 'matrix_r0c2' in aipp_config:
            spr2 = spr2 | (aipp_config.get('matrix_r0c2') & 0xffff) << 32
        if 'matrix_r1c0' in aipp_config:
            spr2 = spr2 | (aipp_config.get('matrix_r1c0') & 0xffff) << 48
        ib.emit(tvm.call_extern(dtype, "set_aipp_spr_2",
                                tvm.const(spr2, dtype="uint64")))

        spr3 = 0
        if 'matrix_r1c1' in aipp_config:
            spr3 = spr3 | (aipp_config.get('matrix_r1c1') & 0xffff)
        if 'matrix_r1c2' in aipp_config:
            spr3 = spr3 | (aipp_config.get('matrix_r1c2') & 0xffff) << 16
        if 'matrix_r2c0' in aipp_config:
            spr3 = spr3 | (aipp_config.get('matrix_r2c0') & 0xffff) << 32
        if 'matrix_r2c1' in aipp_config:
            spr3 = spr3 | (aipp_config.get('matrix_r2c1') & 0xffff) << 48
        ib.emit(tvm.call_extern(dtype, "set_aipp_spr_3",
                                tvm.const(spr3, dtype="uint64")))

        spr4 = 0
        if 'matrix_r2c2' in aipp_config:
            spr4 = spr4 | (aipp_config.get('matrix_r2c2') & 0xffff)
        if 'output_bias_0' in aipp_config:
            spr4 = spr4 | (aipp_config.get('output_bias_0') & 0xff) << 16
        if 'output_bias_1' in aipp_config:
            spr4 = spr4 | (aipp_config.get('output_bias_1') & 0xff) << 24
        if 'output_bias_2' in aipp_config:
            spr4 = spr4 | (aipp_config.get('output_bias_2') & 0xff) << 32
        if aipp_config.get('input_format') in \
                ("YUV420SP_U8", "YUYV_U8", "YUV422SP_U8", "AYUV444_U8", "YUV400_U8"):
            if 'input_bias_0' in aipp_config:
                spr4 = spr4 | (aipp_config.get('input_bias_0') & 0xff) << 40
            if 'input_bias_1' in aipp_config:
                spr4 = spr4 | (aipp_config.get('input_bias_1') & 0xff) << 48
            if 'input_bias_2' in aipp_config:
                spr4 = spr4 | (aipp_config.get('input_bias_2') & 0xff) << 56
        else:
            if 'output_bias_0' in aipp_config:
                spr4 = spr4 | (aipp_config.get('output_bias_0') & 0xff) << 16
            if 'output_bias_1' in aipp_config:
                spr4 = spr4 | (aipp_config.get('output_bias_1') & 0xff) << 24
            if 'output_bias_2' in aipp_config:
                spr4 = spr4 | (aipp_config.get('output_bias_2') & 0xff) << 32
        ib.emit(tvm.call_extern(dtype, "set_aipp_spr_4",
                                tvm.const(spr4, dtype="uint64")))

    spr8 = 0
    if 'padding' in aipp_config and aipp_config.get('padding') == 1:
        if dtype == "float16":
            padding_value = get_fp16(float(aipp_config.get('padding_value', 0)))
        else:
            padding_value = aipp_config.get('padding_value', 0)
        if aipp_config.get('input_format') in ('YUV400_U8',):
            if cur_cce_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
                spr8 = spr8 | (padding_value & 0xffff) << 32
            else:
                spr8 = spr8 | (padding_value & 0xffff)
        if aipp_config.get('input_format') in ("YUV420SP_U8", "YUYV_U8", "YUV422SP_U8",
                                               "RGB888_U8", "XRGB8888_U8", "RGB16"):
            spr8 = spr8 | (padding_value & 0xffff)
            spr8 = spr8 | (padding_value & 0xffff) << 16
            spr8 = spr8 | (padding_value & 0xffff) << 32
        if aipp_config.get('input_format') in ("AYUV444_U8", "ARGB8888_U8"):
            spr8 = spr8 | (padding_value & 0xffff)
            spr8 = spr8 | (padding_value & 0xffff) << 16
            spr8 = spr8 | (padding_value & 0xffff) << 32
            spr8 = spr8 | (padding_value & 0xffff) << 48
    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_8",
                            tvm.const(spr8, dtype="uint64")))

    spr9 = get_spr9(aipp_config, dtype, output_format)
    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_9",
                            tvm.const(spr9, dtype="uint64")))

    if cur_cce_product in Const.V300_SOC_VERSION_LIST:
        return
    chn_0_position = 0
    chn_2_position = 32
    if cur_cce_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403") and \
            aipp_config.get('input_format') in ("YUV400_U8",):
        chn_0_position = 32
        chn_2_position = 0
    spr5 = 0
    if aipp_config.get('input_format') in ("RAW10", "RAW12", "RAW16", "uint16") \
            and dtype == "float16":
        spr5 = 0
        if 'mean_chn_0' in aipp_config:
            mean_chn_0 = get_fp16(aipp_config.get('mean_chn_0'))
            spr5 = spr5 | (mean_chn_0 & 0xffff) << chn_0_position
        if 'mean_chn_1' in aipp_config:
            mean_chn_1 = get_fp16(aipp_config.get('mean_chn_1'))
            spr5 = spr5 | (mean_chn_1 & 0xffff) << 16
        if 'mean_chn_2' in aipp_config:
            mean_chn_2 = aipp_config.get('mean_chn_2')
            spr5 = spr5 | (mean_chn_2 & 0xffff) << chn_2_position
        if 'mean_chn_3' in aipp_config:
            mean_chn_3 = aipp_config.get('mean_chn_3')
            spr5 = spr5 | (mean_chn_3 & 0xffff) << 48
    else:
        if 'mean_chn_0' in aipp_config:
            spr5 = spr5 | (aipp_config.get('mean_chn_0') & 0xffff) << chn_0_position
        if 'mean_chn_1' in aipp_config:
            spr5 = spr5 | (aipp_config.get('mean_chn_1') & 0xffff) << 16
        if 'mean_chn_2' in aipp_config:
            spr5 = spr5 | (aipp_config.get('mean_chn_2') & 0xffff) << chn_2_position
        if 'mean_chn_3' in aipp_config:
            spr5 = spr5 | (aipp_config.get('mean_chn_3') & 0xffff) << 48
    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_5",
                            tvm.const(spr5, dtype="uint64")))

    spr6 = 0
    if 'min_chn_0' in aipp_config:
        min_chn_0 = get_fp16(float(aipp_config.get('min_chn_0')))
        spr6 = spr6 | (min_chn_0 & 0xffff) << chn_0_position
    if 'min_chn_1' in aipp_config:
        min_chn_1 = get_fp16(float(aipp_config.get('min_chn_1')))
        spr6 = spr6 | (min_chn_1 & 0xffff) << 16
    if 'min_chn_2' in aipp_config:
        min_chn_2 = get_fp16(float(aipp_config.get('min_chn_2')))
        spr6 = spr6 | (min_chn_2 & 0xffff) << chn_2_position
    if 'min_chn_3' in aipp_config:
        min_chn_3 = get_fp16(float(aipp_config.get('min_chn_3')))
        spr6 = spr6 | (min_chn_3 & 0xffff) << 48
    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_6",
                            tvm.const(spr6, dtype="uint64")))

    spr7 = 0
    if 'var_reci_chn_0' in aipp_config:
        var_reci_chn_0 = get_fp16(float(aipp_config.get('var_reci_chn_0')))
        spr7 = spr7 | (var_reci_chn_0 & 0xffff) << chn_0_position
    if 'var_reci_chn_1' in aipp_config:
        var_reci_chn_1 = get_fp16(float(aipp_config.get('var_reci_chn_1')))
        spr7 = spr7 | ((var_reci_chn_1) & 0xffff) << 16
    if 'var_reci_chn_2' in aipp_config:
        var_reci_chn_2 = get_fp16(float(aipp_config.get('var_reci_chn_2')))
        spr7 = spr7 | ((var_reci_chn_2) & 0xffff) << chn_2_position
    if 'var_reci_chn_3' in aipp_config:
        var_reci_chn_3 = get_fp16(float(aipp_config.get('var_reci_chn_3')))
        spr7 = spr7 | (var_reci_chn_3 & 0xffff) << 48
    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_7",
                            tvm.const(spr7, dtype="uint64")))


def get_spr18_spr21(aipp_config, dtype, aipp_map, output_format):
    """
    get spr18-spr21, only v300 soc
    """
    chn_offset = 32
    data_mask = 0xffffffff
    spr18 = 0
    spr19 = 0
    if 'var_reci_chn_0' in aipp_config:
        var_reci_chn_0 = get_bin_value_from_fp32(float(aipp_config.get('var_reci_chn_0')))
        spr18 = spr18 | (var_reci_chn_0 & data_mask)
    if 'var_reci_chn_1' in aipp_config:
        var_reci_chn_1 = get_bin_value_from_fp32(float(aipp_config.get('var_reci_chn_1')))
        spr18 = spr18 | ((var_reci_chn_1) & data_mask) << chn_offset
    if 'var_reci_chn_2' in aipp_config:
        var_reci_chn_2 = get_bin_value_from_fp32(float(aipp_config.get('var_reci_chn_2')))
        spr19 = spr19 | ((var_reci_chn_2) & data_mask)
    if 'var_reci_chn_3' in aipp_config:
        var_reci_chn_3 = get_bin_value_from_fp32(float(aipp_config.get('var_reci_chn_3')))
        spr19 = spr19 | (var_reci_chn_3 & data_mask) << chn_offset

    spr20 = 0
    spr21 = 0
    mean_chn_0 = get_bin_value_from_fp32(float(aipp_config.get('mean_chn_0', 0) + aipp_config.get('min_chn_0', 0)))
    spr20 = spr20 | (mean_chn_0 & data_mask)
    mean_chn_1 = get_bin_value_from_fp32(float(aipp_config.get('mean_chn_1', 0) + aipp_config.get('min_chn_1', 0)))
    spr20 = spr20 | (mean_chn_1 & data_mask) << chn_offset
    mean_chn_2 = get_bin_value_from_fp32(float(aipp_config.get('mean_chn_2', 0) + aipp_config.get('min_chn_2', 0)))
    spr21 = spr21 | (mean_chn_2 & data_mask)
    mean_chn_3 = get_bin_value_from_fp32(float(aipp_config.get('mean_chn_3', 0) + aipp_config.get('min_chn_3', 0)))
    spr21 = spr21 | (mean_chn_3 & data_mask) << chn_offset

    aipp_map["spr_18"] = spr18
    aipp_map["spr_19"] = spr19
    aipp_map["spr_20"] = spr20
    aipp_map["spr_21"] = spr21


def set_padding_size(aipp_config, aipp_map):
    """
    :param aipp_config: aipp config
    :param aipp_map: target aipp map (as a dict)
    read value from aipp config and set padding size entries in aipp_map
    """
    if aipp_config.get("padding") != 1:
        return
    aipp_map["padding_left"] = aipp_config.get("left_padding_size", 0)
    aipp_map["padding_right"] = aipp_config.get("right_padding_size", 0)
    aipp_map["padding_top"] = aipp_config.get("top_padding_size", 0)
    aipp_map["padding_bottom"] = aipp_config.get("bottom_padding_size", 0)


def set_spr18_spr21(ib, aipp_config, dtype, output_format):
    """
    :param ib: ir builder
    :param aipp_config: config dict of aipp
    :param dtype: output dtype of aipp
    :param output_format: output format of aipp
    :return: None
    """
    aipp_map = {}
    get_spr18_spr21(aipp_config, dtype, aipp_map, output_format)
    spr18 = aipp_map.get("spr_18")
    spr19 = aipp_map.get("spr_19")
    spr20 = aipp_map.get("spr_20")
    spr21 = aipp_map.get("spr_21")
    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_18",
                            tvm.const(spr18, dtype="uint64")))
    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_19",
                            tvm.const(spr19, dtype="uint64")))
    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_20",
                            tvm.const(spr20, dtype="uint64")))
    ib.emit(tvm.call_extern(dtype, "set_aipp_spr_21",
                            tvm.const(spr21, dtype="uint64")))


def get_spr2_spr9(aipp_config, dtype, cur_cce_product, output_format,
                  aipp_map):
    """
    :param aipp_config:
    :param dtype:
    :param cur_cce_product:
    :return:
    """

    if cur_cce_product in ["Hi3796CV300ES", "Hi3796CV300CS", "SD3403"]:
        spr2 = 0
        if aipp_config.get('input_format') in \
                ["YUV420SP_U8", "YUYV_U8", "YUV422SP_U8",
                 "AYUV444_U8", "YUV400_U8"]:
            if ('csc_switch' in aipp_config) and \
                    (aipp_config.get('csc_switch') == 1):
                if 'matrix_r2c0' in aipp_config:
                    spr2 = spr2 | (aipp_config.get('matrix_r2c0')*4) & 0xffff
                if 'matrix_r2c1' in aipp_config:
                    spr2 = spr2 | \
                           ((aipp_config.get('matrix_r2c1')*4) & 0xffff) << 16
                if 'matrix_r2c2' in aipp_config:
                    spr2 = spr2 | \
                           ((aipp_config.get('matrix_r2c2')*4) & 0xffff) << 32
                if 'matrix_r1c0' in aipp_config:
                    spr2 = spr2 | \
                           ((aipp_config.get('matrix_r1c0')*4) & 0xffff) << 48

                spr3 = 0
                if 'matrix_r1c1' in aipp_config:
                    spr3 = spr3 | ((aipp_config.get('matrix_r1c1')*4) & 0xffff)
                if 'matrix_r1c2' in aipp_config:
                    spr3 = spr3 | \
                           ((aipp_config.get('matrix_r1c2')*4) & 0xffff) << 16
                if 'matrix_r0c0' in aipp_config:
                    spr3 = spr3 | \
                           ((aipp_config.get('matrix_r0c0')*4) & 0xffff) << 32
                if 'matrix_r0c1' in aipp_config:
                    spr3 = spr3 | \
                           ((aipp_config.get('matrix_r0c1')*4) & 0xffff) << 48

                spr4 = 0
                if 'matrix_r0c2' in aipp_config:
                    spr4 = spr4 | \
                           ((aipp_config.get('matrix_r0c2')*4) & 0xffff)
                if 'input_bias_2' in aipp_config:
                    spr4 = spr4 | \
                           (aipp_config.get('input_bias_2') & 0xff) << 40
                if 'input_bias_1' in aipp_config:
                    spr4 = spr4 | \
                           (aipp_config.get('input_bias_1') & 0xff) << 48
                if 'input_bias_0' in aipp_config:
                    spr4 = spr4 | \
                           (aipp_config.get('input_bias_0') & 0xff) << 56
            else:
                # spr_2->bits.csc_matrix_r0_c2 = (uint16_t)(1<<10)
                spr2 = spr2 | (1 << 10 & 0xffff) << 32

                spr3 = 0
                # spr_3->bits.csc_matrix_r1_c1
                spr3 = spr3 | (1 << 10 & 0xffff)
                # spr_3->bits.csc_matrix_r2_c0
                spr3 = spr3 | (1 << 10 & 0xffff) << 32

                spr4 = 0
        else:
            if 'matrix_r2c2' in aipp_config:
                spr2 = spr2 | (aipp_config.get('matrix_r2c2')*4) & 0xffff
            if 'matrix_r2c1' in aipp_config:
                spr2 = spr2 | \
                       ((aipp_config.get('matrix_r2c1')*4) & 0xffff) << 16
            if 'matrix_r2c0' in aipp_config:
                spr2 = spr2 | \
                       ((aipp_config.get('matrix_r2c0')*4) & 0xffff) << 32
            if 'matrix_r1c2' in aipp_config:
                spr2 = spr2 | \
                       ((aipp_config.get('matrix_r1c2')*4) & 0xffff) << 48

            spr3 = 0
            if 'matrix_r1c1' in aipp_config:
                spr3 = spr3 | \
                       ((aipp_config.get('matrix_r1c1')*4) & 0xffff)
            if 'matrix_r1c0' in aipp_config:
                spr3 = spr3 | \
                       ((aipp_config.get('matrix_r1c0')*4) & 0xffff) << 16
            if 'matrix_r0c2' in aipp_config:
                spr3 = spr3 | \
                       ((aipp_config.get('matrix_r0c2')*4) & 0xffff) << 32
            if 'matrix_r0c1' in aipp_config:
                spr3 = spr3 | \
                       ((aipp_config.get('matrix_r0c1')*4) & 0xffff) << 48

            spr4 = 0
            if 'matrix_r0c0' in aipp_config:
                spr4 = spr4 | ((aipp_config.get('matrix_r0c0')*4) & 0xffff)
            if 'output_bias_0' in aipp_config:
                spr4 = spr4 | (aipp_config.get('output_bias_0') & 0xff) << 16
            if 'output_bias_1' in aipp_config:
                spr4 = spr4 | (aipp_config.get('output_bias_1') & 0xff) << 24
            if 'output_bias_2' in aipp_config:
                spr4 = spr4 | (aipp_config.get('output_bias_2') & 0xff) << 32
    else:
        spr2 = 0
        if 'matrix_r0c0' in aipp_config:
            spr2 = spr2 | aipp_config.get('matrix_r0c0') & 0xffff
        if 'matrix_r0c1' in aipp_config:
            spr2 = spr2 | (aipp_config.get('matrix_r0c1') & 0xffff) << 16
        if 'matrix_r0c2' in aipp_config:
            spr2 = spr2 | (aipp_config.get('matrix_r0c2') & 0xffff) << 32
        if 'matrix_r1c0' in aipp_config:
            spr2 = spr2 | (aipp_config.get('matrix_r1c0') & 0xffff) << 48

        spr3 = 0
        if 'matrix_r1c1' in aipp_config:
            spr3 = spr3 | (aipp_config.get('matrix_r1c1') & 0xffff)
        if 'matrix_r1c2' in aipp_config:
            spr3 = spr3 | (aipp_config.get('matrix_r1c2') & 0xffff) << 16
        if 'matrix_r2c0' in aipp_config:
            spr3 = spr3 | (aipp_config.get('matrix_r2c0') & 0xffff) << 32
        if 'matrix_r2c1' in aipp_config:
            spr3 = spr3 | (aipp_config.get('matrix_r2c1') & 0xffff) << 48

        spr4 = 0
        if 'matrix_r2c2' in aipp_config:
            spr4 = spr4 | (aipp_config.get('matrix_r2c2') & 0xffff)
        if 'output_bias_0' in aipp_config:
            spr4 = spr4 | (aipp_config.get('output_bias_0') & 0xff) << 16
        if 'output_bias_1' in aipp_config:
            spr4 = spr4 | (aipp_config.get('output_bias_1') & 0xff) << 24
        if 'output_bias_2' in aipp_config:
            spr4 = spr4 | (aipp_config.get('output_bias_2') & 0xff) << 32
        if aipp_config.get('input_format') in \
                ["YUV420SP_U8", "YUYV_U8", "YUV422SP_U8",
                 "AYUV444_U8", "YUV400_U8"]:
            if 'input_bias_0' in aipp_config:
                spr4 = spr4 | (aipp_config.get('input_bias_0') & 0xff) << 40
            if 'input_bias_1' in aipp_config:
                spr4 = spr4 | (aipp_config.get('input_bias_1') & 0xff) << 48
            if 'input_bias_2' in aipp_config:
                spr4 = spr4 | (aipp_config.get('input_bias_2') & 0xff) << 56
        else:
            if 'output_bias_0' in aipp_config:
                spr4 = spr4 | (aipp_config.get('output_bias_0') & 0xff) << 16
            if 'output_bias_1' in aipp_config:
                spr4 = spr4 | (aipp_config.get('output_bias_1') & 0xff) << 24
            if 'output_bias_2' in aipp_config:
                spr4 = spr4 | (aipp_config.get('output_bias_2') & 0xff) << 32
    if cur_cce_product not in Const.V300_SOC_VERSION_LIST:
        chn_0_position = 0
        chn_2_position = 32
        if cur_cce_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403") and \
                aipp_config.get('input_format') in ("YUV400_U8",):
            chn_0_position = 32
            chn_2_position = 0
        spr5 = 0
        if aipp_config.get('input_format') in ["RAW10", "RAW12",
                                               "RAW16", "uint16"] and \
                dtype == "float16":
            spr5 = 0
            if 'mean_chn_0' in aipp_config:
                mean_chn_0 = get_fp16(aipp_config.get('mean_chn_0'))
                spr5 = spr5 | (mean_chn_0 & 0xffff) << chn_0_position
            if 'mean_chn_1' in aipp_config:
                mean_chn_1 = get_fp16(aipp_config.get('mean_chn_1'))
                spr5 = spr5 | (mean_chn_1 & 0xffff) << 16
            if 'mean_chn_2' in aipp_config:
                mean_chn_2 = aipp_config.get('mean_chn_2')
                spr5 = spr5 | (mean_chn_2 & 0xffff) << chn_2_position
            if 'mean_chn_3' in aipp_config:
                mean_chn_3 = aipp_config.get('mean_chn_3')
                spr5 = spr5 | (mean_chn_3 & 0xffff) << 48
        else:
            if 'mean_chn_0' in aipp_config:
                spr5 = spr5 | (aipp_config.get('mean_chn_0') & 0xffff) << chn_0_position
            if 'mean_chn_1' in aipp_config:
                spr5 = spr5 | (aipp_config.get('mean_chn_1') & 0xffff) << 16
            if 'mean_chn_2' in aipp_config:
                spr5 = spr5 | (aipp_config.get('mean_chn_2') & 0xffff) << chn_2_position
            if 'mean_chn_3' in aipp_config:
                spr5 = spr5 | (aipp_config.get('mean_chn_3') & 0xffff) << 48

        spr6 = 0
        if 'min_chn_0' in aipp_config:
            min_chn_0 = get_fp16(float(aipp_config.get('min_chn_0')))
            spr6 = spr6 | (min_chn_0 & 0xffff) << chn_0_position
        if 'min_chn_1' in aipp_config:
            min_chn_1 = get_fp16(float(aipp_config.get('min_chn_1')))
            spr6 = spr6 | (min_chn_1 & 0xffff) << 16
        if 'min_chn_2' in aipp_config:
            min_chn_2 = get_fp16(float(aipp_config.get('min_chn_2')))
            spr6 = spr6 | (min_chn_2 & 0xffff) << chn_2_position
        if 'min_chn_3' in aipp_config:
            min_chn_3 = get_fp16(float(aipp_config.get('min_chn_3')))
            spr6 = spr6 | (min_chn_3 & 0xffff) << 48

        spr7 = 0
        if 'var_reci_chn_0' in aipp_config:
            var_reci_chn_0 = get_fp16(float(aipp_config.get('var_reci_chn_0')))
            spr7 = spr7 | (var_reci_chn_0 & 0xffff) << chn_0_position
        if 'var_reci_chn_1' in aipp_config:
            var_reci_chn_1 = get_fp16(float(aipp_config.get('var_reci_chn_1')))
            spr7 = spr7 | ((var_reci_chn_1) & 0xffff) << 16
        if 'var_reci_chn_2' in aipp_config:
            var_reci_chn_2 = get_fp16(float(aipp_config.get('var_reci_chn_2')))
            spr7 = spr7 | ((var_reci_chn_2) & 0xffff) << chn_2_position
        if 'var_reci_chn_3' in aipp_config:
            var_reci_chn_3 = get_fp16(float(aipp_config.get('var_reci_chn_3')))
            spr7 = spr7 | (var_reci_chn_3 & 0xffff) << 48

        aipp_map['spr_5'] = tvm.const(spr5, dtype="uint64")
        aipp_map['spr_6'] = tvm.const(spr6, dtype="uint64")
        aipp_map['spr_7'] = tvm.const(spr7, dtype="uint64")

    spr8 = 0
    if aipp_config.get('padding') == 1:
        if dtype == "float16":
            padding_value = get_fp16(float(aipp_config.get('padding_value', 0)))
        else:
            padding_value = aipp_config.get('padding_value', 0)
            if not isinstance(padding_value, int):
                error_desc = "dtype != float16, but padding_value is not an instance of int"
                error_manager_vector.raise_err_specific_reson("aipp", error_desc)
        spr8 = spr8 | (padding_value & 0xffff)
        spr8 = spr8 | (padding_value & 0xffff) << 16
        spr8 = spr8 | (padding_value & 0xffff) << 32
        spr8 = spr8 | (padding_value & 0xffff) << 48
    spr9 = get_spr9(aipp_config, dtype, output_format)
    aipp_map['spr_2'] = tvm.const(spr2, dtype="uint64")
    aipp_map['spr_3'] = tvm.const(spr3, dtype="uint64")
    aipp_map['spr_4'] = tvm.const(spr4, dtype="uint64")
    aipp_map['spr_8'] = tvm.const(spr8, dtype="uint64")
    aipp_map['spr_9'] = tvm.const(spr9, dtype="uint64")


def get_tiling_w(w, l1_image_buf_max, w_loop):
    """
    :param w:
    :param l1_image_buf_max:
    :param w_loop:
    :return:
    """

    if w <= l1_image_buf_max:
        return w, w_loop
    else:
        w_loop = w_loop * 2
        w = w // 2
        return get_tiling_w(w, l1_image_buf_max, w_loop)


def get_lenburst_and_nburst(len_burst, n_burst):
    """
    :param len_burst:
    :param n_burst:
    :return:
    """

    if len_burst <= 65535:
        return len_burst, n_burst
    else:
        n_burst = n_burst*2
        len_burst = len_burst // n_burst
        return get_lenburst_and_nburst(len_burst, n_burst)


def set_aipp_default_params(aipp_config):
    """
    :param aipp_config:
    :return:
    """

    if 'csc_switch' in aipp_config and aipp_config.get('csc_switch') == 1:
        if 'matrix_r0c0' not in aipp_config:
            aipp_config['matrix_r0c0'] = Const.DEFAULT_MATRIX_R0C0_YUV2RGB
        if 'matrix_r0c1' not in aipp_config:
            aipp_config['matrix_r0c1'] = Const.DEFAULT_MATRIX_R0C1_YUV2RGB
        if  'matrix_r0c2' not in aipp_config:
            aipp_config['matrix_r0c2'] = Const.DEFAULT_MATRIX_R0C2_YUV2RGB
        if 'matrix_r1c0' not in aipp_config:
            aipp_config['matrix_r1c0'] = Const.DEFAULT_MATRIX_R1C0_YUV2RGB
        if 'matrix_r1c1' not in aipp_config:
            aipp_config['matrix_r1c1'] = Const.DEFAULT_MATRIX_R1C1_YUV2RGB
        if 'matrix_r1c2' not in aipp_config:
            aipp_config['matrix_r1c2'] = Const.DEFAULT_MATRIX_R1C2_YUV2RGB
        if 'matrix_r2c0' not in aipp_config:
            aipp_config['matrix_r2c0'] = Const.DEFAULT_MATRIX_R2C0_YUV2RGB
        if 'matrix_r2c1' not in aipp_config:
            aipp_config['matrix_r2c1'] = Const.DEFAULT_MATRIX_R2C1_YUV2RGB
        if 'matrix_r2c2' not in aipp_config:
            aipp_config['matrix_r2c2'] = Const.DEFAULT_MATRIX_R2C2_YUV2RGB

        if 'input_bias_0' not in aipp_config:
            aipp_config['input_bias_0'] = Const.DEFAULT_INPUT_BIAS_0
        if 'input_bias_1' not in aipp_config:
            aipp_config['input_bias_1'] = Const.DEFAULT_INPUT_BIAS_1
        if 'input_bias_2' not in aipp_config:
            aipp_config['input_bias_2'] = Const.DEFAULT_INPUT_BIAS_2

        if 'output_bias_0' not in aipp_config:
            aipp_config['output_bias_0'] = Const.DEFAULT_OUTPUT_BIAS_0
        if 'output_bias_1' not in aipp_config:
            aipp_config['output_bias_1'] = Const.DEFAULT_OUTPUT_BIAS_1
        if 'output_bias_2' not in aipp_config:
            aipp_config['output_bias_2'] = Const.DEFAULT_OUTPUT_BIAS_2

    if 'var_reci_chn_0' not in aipp_config:
        aipp_config['var_reci_chn_0'] = Const.DEFAULT_VAR_RECI_CHN
    if 'var_reci_chn_1' not in aipp_config:
        aipp_config['var_reci_chn_1'] = Const.DEFAULT_VAR_RECI_CHN
    if 'var_reci_chn_2' not in aipp_config:
        aipp_config['var_reci_chn_2'] = Const.DEFAULT_VAR_RECI_CHN
    if 'var_reci_chn_3' not in aipp_config:
        aipp_config['var_reci_chn_3'] = Const.DEFAULT_VAR_RECI_CHN


def raise_runtime_error(cause_desc):
    """
    raise runtime error
    """
    error_info = {'errCode': Const.AIPP_OP_ERROR_CODE, 'cause_desc': cause_desc}

    raise RuntimeError(error_info,
                       "Compile op[aipp] failed, cause: %s." % cause_desc)


def check_param_range(param_name, param_value, min_value, max_value):
    """
    check param range
    """
    if param_value < min_value or param_value > max_value:
        cause_desc = "%s[%s] should be within[%d, %d]" % \
                     (param_name, param_value, min_value, max_value)
        raise_runtime_error(cause_desc)


def check_mean(aipp_config, mean_name, mean_value):
    """
    check mean
    """
    if aipp_config.get('input_format') in ["RGB16", "RGB16_IR"]:
        check_param_range(mean_name, mean_value, 0, 65535)
    elif aipp_config.get('input_format') in ["RGB20"]:
        check_param_range(mean_name, mean_value, 0, 1048575)
    elif aipp_config.get('input_format') in ["RGB24", "RGB24_IR"]:
        check_param_range(mean_name, mean_value, 0, 16777215)
    else:
        check_param_range(mean_name, mean_value, 0, 255)


def check_aipp_dtype(aipp_config, input_dtype, output_dtype):
    """
    check aipp dtype
    """
    if aipp_config.get("input_format") == "NC1HWC0DI_S8":
        if input_dtype != "int8":
            cause_desc = "when input_format is NC1HWC0DI_S8, the input dtype must be int8, " \
                         "but actually is %s" % input_dtype
            raise_runtime_error(cause_desc)

        if output_dtype != "int8":
            cause_desc = "when input_format is NC1HWC0DI_S8, the output dtype must be int8, " \
                         "but actually is %s" % output_dtype
            raise_runtime_error(cause_desc)
    elif aipp_config.get("input_format") == "NC1HWC0DI_FP16":
        if input_dtype != "float16":
            cause_desc = "when input_format is NC1HWC0DI_FP16, the input dtype must be float16, " \
                     "but actually is %s" % input_dtype
            raise_runtime_error(cause_desc)

        if output_dtype != "float16":
            cause_desc = "when input_format is NC1HWC0DI_FP16, the output dtype must be float16, " \
                         "but actually is %s" % output_dtype
            raise_runtime_error(cause_desc)
    elif aipp_config.get("input_format") in ["RAW10", "RAW12",
                                             "RAW16", "uint16"]:
        if input_dtype != "uint16":
            cause_desc = "when input_format is %s, the input dtype must be uint16, " \
                         "but actually is %s" % (aipp_config.get("input_format"), input_dtype)
            raise_runtime_error(cause_desc)
    elif aipp_config.get("input_format") in ["RGB16", "RGB16_IR"]:
        if input_dtype != "uint16":
            cause_desc = "when input_format is %s, the input dtype must be uint16, " \
                         "but actually is %s" % (aipp_config.get("input_format"), input_dtype)
            raise_runtime_error(cause_desc)
        if output_dtype != "float16":
            cause_desc = "when input_format is %s, the output dtype must be float16, " \
                         "but actually is %s" % (aipp_config.get("input_format"), output_dtype)
            raise_runtime_error(cause_desc)
    elif aipp_config.get("input_format") in ["RGB20", "RGB24", "RGB24_IR"]:
        if input_dtype != "uint32":
            cause_desc = "when input_format is %s, the input dtype must be uint32, " \
                         "but actually is %s" % (aipp_config.get("input_format"), input_dtype)
            raise_runtime_error(cause_desc)
        if output_dtype != "float16":
            cause_desc = "when input_format is %s, the output dtype must be float16, " \
                         "but actually is %s" % (aipp_config.get("input_format"), output_dtype)
            raise_runtime_error(cause_desc)
    else:
        if input_dtype != "uint8":
            cause_desc = "when input_format is %s, the input dtype must be uint8, " \
                         "but actually is %s" % (aipp_config.get("input_format"), input_dtype)
            raise_runtime_error(cause_desc)


def check_aipp_static_config(input_data, input_format, output_data, aipp_config, cur_cce_product):
    """
    :param input_data:
    :param input_format:
    :param aipp_config:
    :param cur_cce_product:
    :return:
    """

    if input_format == "NHWC":
        h = input_data[1]
        w = input_data[2]
        c = input_data[3]
    elif input_format == "NCHW":
        c = input_data[1]
        h = input_data[2]
        w = input_data[3]
    else:
        c1 = input_data[1]
        h = input_data[2]
        w = input_data[3]
        c0 = input_data[4]

    output_shape = output_data.get('shape')
    output_dtype = output_data.get('dtype').lower()
    output_ori_format = output_data.get('ori_format')
    output_ori_shape = output_data.get('ori_shape')
    _, c1, h, w, c0 = output_shape

    if 'input_format' not in aipp_config:
        cause_desc = "the input_format must be setted"
        raise_runtime_error(cause_desc)

    if input_format == "NC1HWC0_C04":
        if aipp_config.get("input_format") not in ["NC1HWC0DI_FP16", "NC1HWC0DI_S8"]:
            cause_desc = "when aipp op's input format is NC1HWC0_C04, input image's format[%s] " \
                         "must be NC1HWC0DI_FP16 or NC1HWC0DI_S8" % aipp_config.get("input_format")
            raise_runtime_error(cause_desc)

    if aipp_config.get("input_format") == "YUV400_U8" and \
            ('csc_switch' in aipp_config and aipp_config.get('csc_switch') == 1):
        cause_desc = "when input format is YUV400_U8, " \
                     "it doesn't make sense to convert to RGB, csc_switch[%d]" % \
                     aipp_config.get('csc_switch')
        raise_runtime_error(cause_desc)

    if aipp_config.get("input_format") in \
            ("YUV420SP_U8", "RGB888_U8", "YUYV_U8", "YUV420SP_U8", "RGB16", "RGB20", "RGB24"):
        if c != 3:
            cause_desc = "input c[%d] must be 3 when input_format is %s" % \
                         (c, aipp_config.get("input_format"))
            raise_runtime_error(cause_desc)

    if aipp_config.get("input_format") in \
            ("XRGB8888_U8", "ARGB8888_U8", "AYUV444_U8", "RGB8_IR", "RGB16_IR", "RGB24_IR"):
        if c != 4:
            cause_desc = "input c[%d] must be 4 when input_format is %s" % \
                         (c, aipp_config.get("input_format"))
            raise_runtime_error(cause_desc)

    if aipp_config.get("input_format") in \
            ("YUV400_U8", "RAW10", "RAW12", "RAW16", "uint16"):
        if c != 1:
            cause_desc = "input c[%d] must be 1 when input_format is %s" % \
                         (c, aipp_config.get("input_format"))
            raise_runtime_error(cause_desc)

    if aipp_config.get("input_format") in ["NC1HWC0DI_FP16", "NC1HWC0DI_S8"]:
        c0 = input_data[4]
        if c0 != 4:
            cause_desc = "input c0[%d] must be 4 when input_format is %s" % \
                         (c0, aipp_config.get("input_format"))
            raise_runtime_error(cause_desc)

    if aipp_config.get("input_format") not in ("NC1HWC0DI_S8", "NC1HWC0DI_FP16"):
        if c1 != 1:
            if output_ori_format == "NCHW":
                output_ori_c = output_ori_shape[1]
            elif output_ori_format == "NHWC":
                output_ori_c = output_ori_shape[3]
            else:
                cause_desc = "network input format[%s] is not supported" % output_ori_format
                raise_runtime_error(cause_desc)

            cause_desc = "network input c[%d] should be less than or equal to %d" % \
                         (output_ori_c, c0)
            raise_runtime_error(cause_desc)

    if aipp_config.get("input_format") in ["RGB16"] and \
        (('csc_switch' in aipp_config and aipp_config.get('csc_switch') == 1) or \
        ('resize' in aipp_config and aipp_config.get('resize') == 1)):
        if 'csc_switch' not in aipp_config:
            aipp_config['csc_switch'] = 0
        if 'resize' not in aipp_config:
            aipp_config['resize'] = 0

        cause_desc = "%s not support csc_switch[%d] and resize[%d]" % \
                     (aipp_config.get("input_format"),
                      aipp_config.get('csc_switch'),
                      aipp_config.get('resize'))
        raise_runtime_error(cause_desc)

    if aipp_config.get("input_format") in ["RGB8_IR"]:
        if ('resize' in aipp_config and aipp_config.get('resize') == 1) or \
           ('padding' in aipp_config and aipp_config.get('padding') == 1):
            if 'resize' not in aipp_config:
                aipp_config['resize'] = 0
            if 'padding' not in aipp_config:
                aipp_config['padding'] = 0

            cause_desc = "%s not support resize[%d] and padding[%d]" % \
                         (aipp_config.get("input_format"),
                          aipp_config.get('resize'),
                          aipp_config.get('padding'))
            raise_runtime_error(cause_desc)

    if ('csc_switch' in aipp_config and aipp_config.get('csc_switch') == 1) or \
       ('resize' in aipp_config and aipp_config.get('resize') == 1) or \
       ('padding' in aipp_config and aipp_config.get('padding') == 1):
        if aipp_config.get("input_format") in \
                ["RAW10", "RAW12", "RAW16", "uint16",
                 "NC1HWC0DI_S8", "NC1HWC0DI_FP16",
                 "RGB20", "RGB24", "RGB16_IR", "RGB24_IR"]:
            if 'csc_switch' not in aipp_config:
                aipp_config['csc_switch'] = 0
            if 'resize' not in aipp_config:
                aipp_config['resize'] = 0
            if 'padding' not in aipp_config:
                aipp_config['padding'] = 0

            cause_desc = "%s not support csc_switch[%d], resize[%d] and padding[%d]" % \
                         (aipp_config.get("input_format"),
                          aipp_config.get('csc_switch'),
                          aipp_config.get('resize'),
                          aipp_config.get('padding'))
            raise_runtime_error(cause_desc)

    if ('padding' in aipp_config and aipp_config.get('padding') == 1) or \
       ('crop' in aipp_config and aipp_config.get('crop') == 1) or \
       ('resize' in aipp_config and aipp_config.get('resize') == 1):
        if 'crop' not in aipp_config:
            aipp_config['crop'] = 0
        if 'resize' not in aipp_config:
            aipp_config['resize'] = 0
        if 'padding' not in aipp_config:
            aipp_config['padding'] = 0

        if "src_image_size_w" not in aipp_config:
            aipp_config['src_image_size_w'] = 0
        if "src_image_size_h" not in aipp_config:
            aipp_config['src_image_size_h'] = 0

        if "src_image_size_w" not in aipp_config or \
           "src_image_size_h" not in aipp_config or \
            aipp_config.get('src_image_size_w') <= 0 or \
            aipp_config.get('src_image_size_h') <= 0:
            cause_desc = "when crop[%d] or resize[%d] or padding[%d] is enable, " \
                         "src_image_size_w[%d] and src_image_size_h[%d] " \
                         "should be greater than 0" % \
                         (aipp_config.get('crop'),
                          aipp_config.get('resize'),
                          aipp_config.get('padding'),
                          aipp_config.get('src_image_size_w'),
                          aipp_config.get('src_image_size_h'))
            raise_runtime_error(cause_desc)

        if aipp_config.get('input_format') == "YUV420SP_U8" and \
                (aipp_config.get('src_image_size_w') % 2 != 0 or
                 aipp_config.get('src_image_size_h') % 2 != 0):
            cause_desc = "src_image_size_w[%d] and src_image_size_h[%d] " \
                         "should be even for YUV420SP_U8" % \
                         (aipp_config.get('src_image_size_w'),
                          aipp_config.get('src_image_size_h'))
            raise_runtime_error(cause_desc)

        if aipp_config.get('input_format') in ["YUV422SP_U8", "YUYV_U8"] and \
                (aipp_config.get('src_image_size_w') % 2 != 0):
            cause_desc = "src_image_size_w[%d] should be even for %s" % \
                         (aipp_config.get('src_image_size_w'),
                          aipp_config.get('input_format'))
            raise_runtime_error(cause_desc)

        if cur_cce_product in ["Hi3796CV300ES", "Hi3796CV300CS", "SD3403"]:
            if aipp_config.get('input_format') in \
                    ["YUV420SP_U8", "YUV400_U8", "YUV422SP_U8", "RAW10",
                     "RAW12", "RAW16", "uint16"]:
                if aipp_config.get('src_image_size_w') % 16 != 0:
                    cause_desc = "src_image_size_w[%d] must be multiples of 16 for %s" % \
                                 (aipp_config.get('src_image_size_w'),
                                  aipp_config.get('input_format'))
                    raise_runtime_error(cause_desc)
            elif aipp_config.get('input_format') in ["RGB888_U8"]:
                if (aipp_config.get('src_image_size_w')*3) % 16 != 0:
                    cause_desc = "src_image_size_w[%d]*3 must be multiples of 16 for %s" % \
                                 (aipp_config.get('src_image_size_w'),
                                  aipp_config.get('input_format'))
                    raise_runtime_error(cause_desc)
            elif aipp_config.get('input_format') in ["XRGB8888_U8", "ARGB8888_U8", "AYUV444_U8"]:
                if (aipp_config.get('src_image_size_w')*4) % 16 != 0:
                    cause_desc = "src_image_size_w[%d]*4 must be multiples of 16 for %s" % \
                                 (aipp_config.get('src_image_size_w'),
                                  aipp_config.get('input_format'))
                    raise_runtime_error(cause_desc)
            elif aipp_config.get('input_format') in ["YUYV_U8"]:
                if (aipp_config.get('src_image_size_w')*2) % 16 != 0:
                    cause_desc = "src_image_size_w[%d]*2 must be multiples of 16 for %s" % \
                                 (aipp_config.get('src_image_size_w'),
                                  aipp_config.get('input_format'))
                    raise_runtime_error(cause_desc)

        if ('crop' in aipp_config and aipp_config.get('crop') == 1):
            if ("load_start_pos_h") not in aipp_config:
                aipp_config["load_start_pos_h"] = 0
            if ("load_start_pos_w") not in aipp_config:
                aipp_config["load_start_pos_w"] = 0
            if ("crop_size_h") not in aipp_config:
                aipp_config["crop_size_h"] = 0
            if ("crop_size_w") not in aipp_config:
                aipp_config["crop_size_w"] = 0

            if 'resize' not in aipp_config:
                aipp_config['resize'] = 0
            if 'padding' not in aipp_config:
                aipp_config['padding'] = 0

            if aipp_config.get("crop_size_h") == 0 or \
                    aipp_config.get("crop_size_w") == 0:
                if ('resize' in aipp_config and \
                    aipp_config.get('resize') == 1) or \
                    ('padding' in aipp_config and \
                     aipp_config.get('padding') == 1):
                    cause_desc = "when crop_size_h[%d] is equal to 0 or " \
                                 "crop_size_w[%d] is equal to 0, " \
                                 "resize[%d] and padding[%d] " \
                                 "should be not enable" % \
                                 (aipp_config.get("crop_size_h"),
                                  aipp_config.get("crop_size_w"),
                                  aipp_config.get('resize'),
                                  aipp_config.get('padding'))
                    raise_runtime_error(cause_desc)

                if aipp_config.get("crop_size_h") == 0:
                    aipp_config["crop_size_h"] = h

                if aipp_config.get("crop_size_w") == 0:
                    aipp_config["crop_size_w"] = w

            if aipp_config.get("load_start_pos_h") + \
                    aipp_config.get("crop_size_h") > \
                    aipp_config.get("src_image_size_h"):
                cause_desc = "when crop is enable, " \
                             "load_start_pos_h[%d] + crop_size_h[%d] " \
                             "should be less than or equal to " \
                             "src_image_size_h[%d]" % \
                             (aipp_config.get("load_start_pos_h"),
                              aipp_config.get("crop_size_h"),
                              aipp_config.get("src_image_size_h"))
                raise_runtime_error(cause_desc)

            if aipp_config.get("load_start_pos_w") + \
                    aipp_config.get("crop_size_w") > \
                    aipp_config.get("src_image_size_w"):
                cause_desc = "when crop is enable, " \
                             "load_start_pos_w[%d] + crop_size_w[%d] " \
                             "should be less than or equal to " \
                             "src_image_size_w[%d]" % \
                             (aipp_config.get("load_start_pos_w"),
                              aipp_config.get("crop_size_w"),
                              aipp_config.get("src_image_size_w"))
                raise_runtime_error(cause_desc)

            if aipp_config.get('input_format') in ["YUV420SP_U8"]:
                if cur_cce_product not in Const.ODD_CROP_SUPPORT_SOC_VERSION_LIST and \
                   (aipp_config.get("load_start_pos_h") % 2 != 0 or
                    aipp_config.get("load_start_pos_w") % 2 != 0):
                    cause_desc = "when input_format is YUV420SP_U8, " \
                                 "load_start_pos_h[%d], load_start_pos_w[%d] " \
                                 "must be even" % \
                                 (aipp_config.get("load_start_pos_h"),
                                  aipp_config.get("load_start_pos_w"))
                    raise_runtime_error(cause_desc)

                if cur_cce_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403") and \
                       (aipp_config.get("crop_size_h") % 2 != 0 or aipp_config.get("crop_size_w") % 2 != 0):
                    cause_desc = "when input_format is YUV420SP_U8, " \
                                 "crop_size_h[%d] and crop_size_w[%d] must be even" % \
                                 (aipp_config.get("crop_size_h"), aipp_config.get("crop_size_w"))
                    raise_runtime_error(cause_desc)

            if cur_cce_product not in ("Ascend310B", "AS31XM1", "Ascend610Lite", "BS9SX2A", "MC61AM21A") and \
                aipp_config.get('input_format') in ["YUYV_U8", "YUV422SP_U8"]:
                if aipp_config.get("load_start_pos_w") % 2 != 0:
                    cause_desc = "when input_format is %s, " \
                                 "load_start_pos_w[%d] must be even" % \
                                 (aipp_config.get('input_format'),
                                  aipp_config.get("load_start_pos_w"))
                    raise_runtime_error(cause_desc)

                if cur_cce_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403") and \
                       aipp_config.get("crop_size_w") % 2 != 0:
                    cause_desc = "when input_format is %s, crop_size_w[%d] must be even" % \
                                 (aipp_config.get('input_format'), aipp_config.get("crop_size_w"))
                    raise_runtime_error(cause_desc)

        if ('resize' in aipp_config and aipp_config.get('resize') == 1):
            if cur_cce_product not in ["Hi3796CV300ES", "Hi3796CV300CS", "SD3403"]:
                cause_desc = "resize only support " \
                             "Hi3796CV300ES, Hi3796CV300CS and SD3403"
                raise_runtime_error(cause_desc)

            if ("resize_input_h") not in aipp_config:
                if "crop" in aipp_config and aipp_config.get("crop") == 1:
                    if aipp_config.get("crop_size_h") <= 0:
                        cause_desc = "when crop and resize is enable, " \
                                     "crop_size_h[%d] must be larger than 0" % \
                                     (aipp_config.get("crop_size_h"))
                        raise_runtime_error(cause_desc)

                    aipp_config["resize_input_h"] = aipp_config.get("crop_size_h")
                else:
                    aipp_config["resize_input_h"] = \
                        aipp_config.get("src_image_size_h")

            if aipp_config["resize_input_h"] < 16 or \
                aipp_config["resize_input_h"] > 4096:
                cause_desc = "when resize is enable, resize_input_h[%d] should " \
                             "be within [16, 4096]" % \
                             aipp_config["resize_input_h"]
                raise_runtime_error(cause_desc)

            if ("resize_input_w") not in aipp_config:
                if "crop" in aipp_config and aipp_config.get("crop") == 1:
                    if aipp_config.get("crop_size_w") <= 0:
                        cause_desc = "when crop and resize is enable, " \
                                     "crop_size_w[%d] must be larger than 0" % \
                                     (aipp_config.get("crop_size_w"))
                        raise_runtime_error(cause_desc)

                    aipp_config["resize_input_w"] = aipp_config.get("crop_size_w")
                else:
                    aipp_config["resize_input_w"] = \
                        aipp_config.get("src_image_size_w")

            if aipp_config["resize_input_w"] < 16 or \
                    aipp_config["resize_input_w"] > 4096:
                cause_desc = "when resize is enable, resize_input_w[%d] should " \
                             "be within [16, 4096]" % \
                             aipp_config["resize_input_w"]
                raise_runtime_error(cause_desc)

            if ("resize_output_h") not in aipp_config:
                aipp_config["resize_output_h"] = 0

            if ("resize_output_w") not in aipp_config:
                aipp_config["resize_output_w"] = 0

            if aipp_config.get("resize_input_w") > 4096:
                cause_desc = "resize_input_w[%d] should " \
                             "be less than or equal to 4096" % \
                             aipp_config.get("resize_input_w")
                raise_runtime_error(cause_desc)

            if aipp_config.get("resize_output_h") == 0 or \
                    aipp_config.get("resize_output_w") == 0:
                if 'padding' in aipp_config and aipp_config.get('padding') == 1:
                    cause_desc = "when resize_output_h[%d] is equal to 0 or " \
                                 "resize_output_w[%d] is equal to 0, " \
                                 "padding[%d] should be not enable" % \
                                 (aipp_config.get("resize_output_h"),
                                  aipp_config.get("resize_output_w"),
                                  aipp_config.get('padding'))
                    raise_runtime_error(cause_desc)

            if aipp_config.get("resize_output_h") == 0:
                aipp_config["resize_output_h"] = h

            if aipp_config.get("resize_output_w") == 0:
                aipp_config["resize_output_w"] = w

            if aipp_config.get("resize_output_w") < 16 or \
                    aipp_config.get("resize_output_w") > 1920:
                cause_desc = "resize_output_w[%d] should be within [16, 1920]" \
                             % aipp_config.get("resize_output_w")
                raise_runtime_error(cause_desc)

            if aipp_config.get("resize_output_h") < 16 or \
                    aipp_config.get("resize_output_h") > 4096:
                cause_desc = "resize_output_h[%d] should be within [16, 4096]" \
                             % aipp_config.get("resize_output_h")
                raise_runtime_error(cause_desc)

            resize_output_w = aipp_config.get("resize_output_w")
            resize_input_w = aipp_config.get("resize_input_w")
            resize_output_h = aipp_config.get("resize_output_h")
            resize_input_h = aipp_config.get("resize_input_h")

            if resize_output_w/resize_input_w < 1/16 or \
               resize_output_w/resize_input_w > 16:
                cause_desc = "resize_output_w[%d]/resize_input_w[%d] " \
                             "should be within [1/16, 16]" % \
                             (resize_output_w, resize_input_w)
                raise_runtime_error(cause_desc)

            if resize_output_h/resize_input_h < 1/16 or \
               resize_output_h/resize_input_h > 16:
                cause_desc = "resize_output_h[%d]/resize_input_h[%d] " \
                             "should be within [1/16, 16]" % \
                             (resize_output_h, resize_input_h)
                raise_runtime_error(cause_desc)

        if ('padding' in aipp_config and aipp_config.get('padding') == 1):
            if cur_cce_product in ("Ascend310", "Ascend910", "Ascend610", "Ascend310P",
                                   "Ascend910B", "Ascend310B", "AS31XM1", "Ascend610Lite",
                                   "BS9SX2A", "MC61AM21A"):
                if w > 1080:
                    cause_desc = "after padding, aipp output w[%d] should " \
                                 "be less than or eaqual to 1080" % w
                    raise_runtime_error(cause_desc)
            if cur_cce_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
                if w > 4096:
                    cause_desc = "after padding, aipp output w[%d] should " \
                                 "be less than or eaqual to 4096" % w
                    raise_runtime_error(cause_desc)

            # set default padding value if padding_value not in config
            if 'padding_value' not in aipp_config:
                aipp_config['padding_value'] = 0
            if output_dtype != "float16" and isinstance(aipp_config.get('padding_value'), float):
                cause_desc = "output dtype is %s, the dtype of padding_value should be the same as that of output" \
                             % output_dtype
                raise_runtime_error(cause_desc)

        crop_size_w = 0
        crop_size_h = 0
        if "crop_size_w" in aipp_config:
            crop_size_w = aipp_config.get("crop_size_w")
        if "crop_size_h" in aipp_config:
            crop_size_h = aipp_config.get("crop_size_h")

        left_padding_size, right_padding_size, top_padding_size, \
        bottom_padding_size = get_padding_size(aipp_config)
        if left_padding_size > 32 or right_padding_size > 32 or \
                top_padding_size > 32 or bottom_padding_size > 32:
            cause_desc = "The max padding size is 32, " \
                         "and left_padding_size is %d, " \
                         "right_padding_size is %d, " \
                         "top_padding_size is %d, " \
                         "bottom_padding_size is %d" % \
                         (left_padding_size, right_padding_size,
                          top_padding_size, bottom_padding_size)
            raise_runtime_error(cause_desc)

        if ('crop' in aipp_config and aipp_config.get('crop') == 1) and \
           ('resize' not in aipp_config or aipp_config.get('resize') == 0) and \
           ('padding' not in aipp_config or aipp_config.get('padding') == 0):
            if crop_size_h != h or crop_size_w != w:
                cause_desc = "when crop is enable, resize is disable and " \
                             "padding is disable, " \
                             "crop_size_w[%d] should be equal to output w[%d] and " \
                             "crop_size_h[%d] should be equal to output h[%d]" % \
                             (crop_size_w, w, crop_size_h, h)
                raise_runtime_error(cause_desc)

        if ('resize' in aipp_config and aipp_config.get('resize') == 1) and \
           ('padding' not in aipp_config or aipp_config.get('padding') == 0):
            if aipp_config.get("resize_output_h") != h or \
                    aipp_config.get("resize_output_w") != w:
                cause_desc = "when resize is enable and " \
                             "padding is disable, " \
                             "resize_output_w[%d] should " \
                             "be equal to output w[%d] " \
                             "and resize_output_h[%d] should " \
                             "be equal to output h[%d]" % \
                             (aipp_config.get("resize_output_w"), w,
                              aipp_config.get("resize_output_h"), h)
                raise_runtime_error(cause_desc)

        if ('crop' in aipp_config and aipp_config.get('crop') == 1) and \
           ('resize' not in aipp_config or aipp_config.get('resize') == 0) and \
           ('padding' in aipp_config and aipp_config.get('padding') == 1):
            if crop_size_w + left_padding_size + right_padding_size != w or \
               crop_size_h + top_padding_size + bottom_padding_size != h:
                cause_desc = "when crop is enable, resize is disable and " \
                             "padding is enable, " \
                             "crop_size_w[%d] + left_padding_size[%d] + " \
                             "right_padding_size[%d] " \
                             "should be equal to output w[%d] " \
                             "and crop_size_h[%d] + top_padding_size[%d] + " \
                             "bottom_padding_size[%d] should " \
                             "be equal to output h[%d]" % \
                             (crop_size_w, left_padding_size,
                              right_padding_size, w,
                              crop_size_h, top_padding_size,
                              bottom_padding_size, h)
                raise_runtime_error(cause_desc)

        if ('crop' not in aipp_config or aipp_config.get('crop') == 0) and \
           ('resize' in aipp_config and aipp_config.get('resize') == 1) and \
           ('padding' in aipp_config and aipp_config.get('padding') == 1):
            if aipp_config.get("resize_output_w") + \
                    left_padding_size + right_padding_size != w or \
                    aipp_config.get("resize_output_h") + \
                    top_padding_size + bottom_padding_size != h:
                cause_desc = "when crop is disable, resize is enable and " \
                             "padding is enable, " \
                             "resize_output_w[%d] + left_padding_size[%d] + " \
                             "right_padding_size[%d] should " \
                             "be equal to output w[%d] " \
                             "and resize_output_h[%d] + top_padding_size[%d] + " \
                             "bottom_padding_size[%d] should " \
                             "be equal to output h[%d]" % \
                             (aipp_config.get("resize_output_w"),
                              left_padding_size, right_padding_size, w,
                              aipp_config.get("resize_output_h"),
                              top_padding_size, bottom_padding_size, h)
                raise_runtime_error(cause_desc)

        if ('crop' not in aipp_config or aipp_config.get('crop') == 0) and \
           ('resize' not in aipp_config or aipp_config.get('resize') == 0) and \
           ('padding' in aipp_config and aipp_config.get('padding') == 1):
            if aipp_config.get("src_image_size_w") + \
                    left_padding_size + right_padding_size != w or \
               aipp_config.get("src_image_size_h") + \
                    top_padding_size + bottom_padding_size != h:
                cause_desc = "when crop is disable, resize is disable, " \
                             "padding is enable, " \
                             "src_image_size_w[%d] + left_padding_size[%d] + " \
                             "right_padding_size[%d] should " \
                             "be equal to output w[%d] and " \
                             "src_image_size_h[%d] + top_padding_size[%d] + " \
                             "bottom_padding_size[%d] should " \
                             "be equal to output h[%d]" % \
                             (aipp_config.get("src_image_size_w"),
                              left_padding_size, right_padding_size, w,
                              aipp_config.get("src_image_size_h"),
                              top_padding_size, bottom_padding_size, h)
                raise_runtime_error(cause_desc)

    if cur_cce_product in ["Ascend310"]:
        if aipp_config.get('input_format') not in Const.SUPPORT_IMAGE_FORMAT_MAP.get('Ascend310'):
            cause_desc = "Ascend310 only support " + ", ".join(Const.SUPPORT_IMAGE_FORMAT_MAP.get('Ascend310')) + \
                         ", current input format is %s" % aipp_config.get('input_format')
            raise_runtime_error(cause_desc)

    if cur_cce_product in ["Ascend310B", "AS31XM1", "Ascend610Lite", "BS9SX2A", "MC61AM21A"]:
        if aipp_config.get('input_format') not in Const.SUPPORT_IMAGE_FORMAT_MAP.get(cur_cce_product):
            cause_desc = "Ascend310B/AS31XM1/Ascend610Lite/BS9SX2A/MC61AM21A only support " + \
                         ", ".join(Const.SUPPORT_IMAGE_FORMAT_MAP.get(cur_cce_product)) + \
                         ", current input format is %s" % aipp_config.get('input_format')
            raise_runtime_error(cause_desc)

    if cur_cce_product in ["Ascend910"]:
        if aipp_config.get('input_format') not in Const.SUPPORT_IMAGE_FORMAT_MAP.get('Ascend910'):
            cause_desc = "Ascend910 only support " + ", ".join(Const.SUPPORT_IMAGE_FORMAT_MAP.get('Ascend910')) + \
                         ", current input format is %s" % aipp_config.get('input_format')
            raise_runtime_error(cause_desc)

    if cur_cce_product in ["Ascend910B", "Ascend910_93"]:
        if aipp_config.get('input_format') not in Const.SUPPORT_IMAGE_FORMAT_MAP.get('Ascend910B'):
            cause_desc = "Ascend910B only support " + ", ".join(Const.SUPPORT_IMAGE_FORMAT_MAP.get('Ascend910B')) + \
                         ", current input format is %s" % aipp_config.get('input_format')
            raise_runtime_error(cause_desc)

    if cur_cce_product in ["Ascend610"]:
        if aipp_config.get('input_format') not in Const.SUPPORT_IMAGE_FORMAT_MAP.get('Ascend610'):
            cause_desc = "Ascend610 only support " + ", ".join(Const.SUPPORT_IMAGE_FORMAT_MAP.get('Ascend610')) + \
                         ", current input format is %s" % aipp_config.get('input_format')
            raise_runtime_error(cause_desc)

    if cur_cce_product in ["Ascend310P"]:
        if aipp_config.get('input_format') not in Const.SUPPORT_IMAGE_FORMAT_MAP.get('Ascend310P'):
            cause_desc = "Ascend310P only support " + ", ".join(Const.SUPPORT_IMAGE_FORMAT_MAP.get('Ascend310P')) + \
                         ", current input format is %s" % aipp_config.get('input_format')
            raise_runtime_error(cause_desc)

    if cur_cce_product in ["BS9SX1A"]:
        if aipp_config.get('input_format') not in Const.SUPPORT_IMAGE_FORMAT_MAP.get('BS9SX1A'):
            cause_desc = "BS9SX1A only support " + ", ".join(Const.SUPPORT_IMAGE_FORMAT_MAP.get('BS9SX1A')) + \
                         ", current input format is %s" % aipp_config.get('input_format')
            raise_runtime_error(cause_desc)

    if cur_cce_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        if aipp_config.get('input_format') not in Const.SUPPORT_IMAGE_FORMAT_MAP.get(
                'Hi3796CV300ES-Hi3796CV300CS-SD3403'):
            cause_desc = "Hi3796CV300ES, Hi3796CV300CS and SD3403 only support " + \
                         ", ".join(Const.SUPPORT_IMAGE_FORMAT_MAP.get('Hi3796CV300ES-Hi3796CV300CS-SD3403')) + \
                         ", current input format is %s" % aipp_config.get('input_format')
            raise_runtime_error(cause_desc)

    if (("crop") not in aipp_config or aipp_config.get("crop") == 0) and \
       (("resize") not in aipp_config or aipp_config.get("resize") == 0) and \
       (("padding") not in aipp_config or aipp_config.get("padding") == 0):
        if (('src_image_size_h' in aipp_config) and \
            aipp_config.get('src_image_size_h') != 0) and \
                h != aipp_config.get('src_image_size_h'):
            cause_desc = "input h[%d] of network should be equal to " \
                         "src_image_size_h[%d]" % \
                         (h, aipp_config.get('src_image_size_h'))
            raise_runtime_error(cause_desc)

        if (('src_image_size_w' in aipp_config) and \
            aipp_config.get('src_image_size_w') != 0) and \
            w != aipp_config.get('src_image_size_w'):
            cause_desc = "input w[%d] of network should be equal to " \
                         "src_image_size_w[%d]" % \
                         (w, aipp_config.get('src_image_size_w'))
            raise_runtime_error(cause_desc)

        if cur_cce_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
            if aipp_config.get('input_format') in \
                ("YUV420SP_U8", "YUV400_U8", "YUV422SP_U8", "RAW10",
                 "RAW12", "RAW16", "uint16"):
                if (('src_image_size_w' in aipp_config) and \
                        aipp_config.get('src_image_size_w') % 16 != 0) or w % 16 != 0:
                    cause_desc = "src_image_size_w[%d] of %s must " \
                                 "be multiples of 16" % \
                                 (aipp_config.get('src_image_size_w') if ('src_image_size_w' in aipp_config) else w,
                                  aipp_config.get('input_format'))
                    raise_runtime_error(cause_desc)
            elif aipp_config.get('input_format') in ["RGB888_U8"]:
                if (('src_image_size_w' in aipp_config) and \
                        (aipp_config.get('src_image_size_w')*3) % 16 != 0) or (w*3) % 16 != 0:
                    cause_desc = "src_image_size_w[%d]*3 of %s must " \
                                 "be multiples of 16" % \
                                 (aipp_config.get('src_image_size_w') if ('src_image_size_w' in aipp_config) else w,
                                  aipp_config.get('input_format'))
                    raise_runtime_error(cause_desc)
            elif aipp_config.get('input_format') in ["XRGB8888_U8", "ARGB8888_U8", "AYUV444_U8"]:
                if (('src_image_size_w' in aipp_config) and \
                        (aipp_config.get('src_image_size_w')*4) % 16 != 0) or (w*4) % 16 != 0:
                    cause_desc = "src_image_size_w[%d]*4 of %s must " \
                                 "be multiples of 16" % \
                                 (aipp_config.get('src_image_size_w') if ('src_image_size_w' in aipp_config) else w,
                                  aipp_config.get('input_format'))
                    raise_runtime_error(cause_desc)
            elif aipp_config.get('input_format') in ["YUYV_U8"]:
                if (('src_image_size_w' in aipp_config) and \
                        (aipp_config.get('src_image_size_w')*2) % 16 != 0) or (w*2) % 16 != 0:
                    cause_desc = "src_image_size_w[%d]*2 of %s must " \
                                 "be multiples of 16" % \
                                 (aipp_config.get('src_image_size_w') if ('src_image_size_w' in aipp_config) else w,
                                  aipp_config.get('input_format'))
                    raise_runtime_error(cause_desc)

        if aipp_config.get('input_format') == "YUV420SP_U8" and \
            (w % 2 != 0 or h % 2 != 0):
            cause_desc = "src_image_size_h[%d] and src_image_size_w[%d] should " \
                         "be even for YUV420SP_U8" % (h, w)
            raise_runtime_error(cause_desc)

        if aipp_config.get('input_format') in ["YUV422SP_U8", "YUYV_U8"] and \
                (w % 2 != 0):
            cause_desc = "src_image_size_w[%d] should be even for %s" % \
                         (w, aipp_config.get('input_format'))
            raise_runtime_error(cause_desc)

    if 'src_image_size_h' not in aipp_config:
        aipp_config['src_image_size_h'] = 0
    if 'src_image_size_w' not in aipp_config:
        aipp_config['src_image_size_w'] = 0

    if 'src_image_size_h' in aipp_config:
        if ((aipp_config["src_image_size_h"] < 0 or
             aipp_config["src_image_size_h"] > 4096)):
            cause_desc = "src_image_size_h[%d] should " \
                         "be within [0, 4096]" % \
                         aipp_config["src_image_size_h"]
            raise_runtime_error(cause_desc)

        if aipp_config["src_image_size_h"] == 0:
            if h < 1 or h > 4096:
                cause_desc = "when src_image_size_h is 0, " \
                             "src_image_size_h is equal to " \
                             "input_H of network, " \
                             "and input_H[%d] of network should be " \
                             "within [1, 4096]" % h
                raise_runtime_error(cause_desc)

    if 'src_image_size_w' in aipp_config:
        if ((aipp_config["src_image_size_w"] < 2 or aipp_config["src_image_size_w"] > 4096) and
                aipp_config["src_image_size_w"] != 0):
            cause_desc = "src_image_size_w[%d] should " \
                         "be within [2, 4096] or equal to 0" % \
                         aipp_config["src_image_size_w"]
            raise_runtime_error(cause_desc)

        if aipp_config["src_image_size_w"] == 0:
            if w < 2 or w > 4096:
                cause_desc = "when src_image_size_w is 0, " \
                             "src_image_size_w is equal to " \
                             "input_W of network, " \
                             "and input_W[%d] of network should be " \
                             "within [2, 4096]" % w
                raise_runtime_error(cause_desc)

    if 'mean_chn_0' in aipp_config:
        check_mean(aipp_config, 'mean_chn_0', aipp_config.get('mean_chn_0'))
    if 'mean_chn_1' in aipp_config:
        check_mean(aipp_config, 'mean_chn_1', aipp_config.get('mean_chn_1'))
    if 'mean_chn_2' in aipp_config:
        check_mean(aipp_config, 'mean_chn_2', aipp_config.get('mean_chn_2'))
    if 'mean_chn_3' in aipp_config:
        check_mean(aipp_config, 'mean_chn_3', aipp_config.get('mean_chn_3'))

    if 'min_chn_0' in aipp_config:
        check_param_range('min_chn_0', aipp_config.get('min_chn_0'), 0, 255)
    if 'min_chn_1' in aipp_config:
        check_param_range('min_chn_1', aipp_config.get('min_chn_1'), 0, 255)
    if 'min_chn_2' in aipp_config:
        check_param_range('min_chn_2', aipp_config.get('min_chn_2'), 0, 255)
    if 'min_chn_3' in aipp_config:
        check_param_range('min_chn_3', aipp_config.get('min_chn_3'), 0, 255)

    if 'var_reci_chn_0' in aipp_config:
        check_param_range('var_reci_chn_0', aipp_config.get('var_reci_chn_0'),
                          -65504, 65504)
    if 'var_reci_chn_1' in aipp_config:
        check_param_range('var_reci_chn_1', aipp_config.get('var_reci_chn_1'),
                          -65504, 65504)
    if 'var_reci_chn_2' in aipp_config:
        check_param_range('var_reci_chn_2', aipp_config.get('var_reci_chn_2'),
                          -65504, 65504)
    if 'var_reci_chn_3' in aipp_config:
        check_param_range('var_reci_chn_3', aipp_config.get('var_reci_chn_3'),
                          -65504, 65504)

    if 'csc_switch' in aipp_config and \
        aipp_config.get('csc_switch') == 1:
        if 'matrix_r0c0' in aipp_config:
            check_param_range('matrix_r0c0', aipp_config.get('matrix_r0c0'),
                              -32768, 32767)
        if 'matrix_r0c1' in aipp_config:
            check_param_range('matrix_r0c1', aipp_config.get('matrix_r0c1'),
                              -32768, 32767)
        if 'matrix_r0c2' in aipp_config:
            check_param_range('matrix_r0c2', aipp_config.get('matrix_r0c2'),
                              -32768, 32767)
        if 'matrix_r1c0' in aipp_config:
            check_param_range('matrix_r1c0', aipp_config.get('matrix_r1c0'),
                              -32768, 32767)
        if 'matrix_r1c1' in aipp_config:
            check_param_range('matrix_r1c1', aipp_config.get('matrix_r1c1'),
                              -32768, 32767)
        if 'matrix_r1c2' in aipp_config:
            check_param_range('matrix_r1c2', aipp_config.get('matrix_r2c2'),
                              -32768, 32767)
        if 'matrix_r2c0' in aipp_config:
            check_param_range('matrix_r2c0', aipp_config.get('matrix_r2c0'),
                              -32768, 32767)
        if 'matrix_r2c1' in aipp_config:
            check_param_range('matrix_r2c1', aipp_config.get('matrix_r2c1'),
                              -32768, 32767)
        if 'matrix_r2c2' in aipp_config:
            check_param_range('matrix_r2c2', aipp_config.get('matrix_r2c2'),
                              -32768, 32767)

        if 'output_bias_0' in aipp_config:
            check_param_range('output_bias_0', aipp_config.get('output_bias_0'),
                              0, 255)
        if 'output_bias_1' in aipp_config:
            check_param_range('output_bias_1', aipp_config.get('output_bias_1'),
                              0, 255)
        if 'output_bias_2' in aipp_config:
            check_param_range('output_bias_2', aipp_config.get('output_bias_2'),
                              0, 255)
        if 'output_bias_3' in aipp_config:
            check_param_range('output_bias_3', aipp_config.get('output_bias_3'),
                              0, 255)

        if 'input_bias_0' in aipp_config:
            check_param_range('input_bias_0', aipp_config.get('input_bias_0'),
                              0, 255)
        if 'input_bias_1' in aipp_config:
            check_param_range('input_bias_1', aipp_config.get('input_bias_1'),
                              0, 255)
        if 'input_bias_2' in aipp_config:
            check_param_range('input_bias_2', aipp_config.get('input_bias_2'),
                              0, 255)
        if 'input_bias_3' in aipp_config:
            check_param_range('input_bias_3', aipp_config.get('input_bias_3'),
                              0, 255)

    if 'raw_rgbir_to_f16_n' in aipp_config:
        if cur_cce_product in ["Hi3796CV300ES", "Hi3796CV300CS", "SD3403"]:
            check_param_range('raw_rgbir_to_f16_n',
                              aipp_config.get('raw_rgbir_to_f16_n'), 0, 15)
        elif cur_cce_product in ["BS9SX1A"]:
            check_param_range('raw_rgbir_to_f16_n',
                              aipp_config.get('raw_rgbir_to_f16_n'), 0, 31)


def new_alloc(ib, dtype, size):
    """
    :param ib:
    :param dtype:
    :param size:
    :return:
    """

    if dtype == "float16":
        size = (((size*2 + 31) // 32)*32) // 2
    else:
        size = (((size + 31) // 32)*32)

    output_cb = ib.allocate(dtype, (size,), "output_cb", scope=tbe_platform.scope_cbuf)
    output_cb_buf = tvm.decl_buffer((size,), dtype, "output_cb_buf",
                                    scope=tbe_platform.scope_cbuf, data=output_cb)

    output_ub = ib.allocate(dtype, (size,), "output_ub", scope=tbe_platform.scope_ubuf)
    output_ub_buf = tvm.decl_buffer((size,), dtype, "output_ub_buf",
                                    scope=tbe_platform.scope_ubuf, data=output_ub)

    return output_cb_buf, output_ub_buf


def get_crop_info(aipp_config, src_h, src_w):
    """
    :param aipp_config:
    :return:
    """
    image_h = src_h
    image_w = src_w
    load_start_pos_h = 0
    load_start_pos_w = 0
    crop_size_h = src_h
    crop_size_w = src_w
    if "src_image_size_h" in aipp_config and \
        aipp_config.get("src_image_size_h") > 0:
        image_h = aipp_config.get("src_image_size_h")

    if "src_image_size_w" in aipp_config and \
            aipp_config.get("src_image_size_w") > 0:
        image_w = aipp_config.get("src_image_size_w")

    if 'crop' in aipp_config and aipp_config.get('crop') == 1:
        if "load_start_pos_h" in aipp_config:
            load_start_pos_h = aipp_config.get("load_start_pos_h")

        if "load_start_pos_w" in aipp_config:
            load_start_pos_w = aipp_config.get("load_start_pos_w")

        if "crop_size_h" in aipp_config and aipp_config.get("crop_size_h") > 0:
            crop_size_h = aipp_config.get("crop_size_h")

        if "crop_size_w" in aipp_config and aipp_config.get("crop_size_w") > 0:
            crop_size_w = aipp_config.get("crop_size_w")

    return image_h, image_w, load_start_pos_h, load_start_pos_w, \
           crop_size_h, crop_size_w


def get_actual_col_size(aipp_config, h, w, support_vertical_padding=False):
    """
    :param aipp_config:
    :param h:
    :param w:
    :return:
    """
    actual_col_size = h * w

    if "crop" in aipp_config and aipp_config.get("crop") == 1:
        crop_size_h = aipp_config.get("crop_size_h")
        crop_size_w = aipp_config.get("crop_size_w")
        actual_col_size = crop_size_h * crop_size_w

    if "resize" in aipp_config and aipp_config.get("resize") == 1:
        actual_col_size = \
            aipp_config.get("resize_output_h") * aipp_config.get("resize_output_w")

    if "padding" in aipp_config and aipp_config.get("padding") == 1:
        top_padding_size = 0
        bottom_padding_size = 0
        if "top_padding_size" in aipp_config:
            top_padding_size = aipp_config.get("top_padding_size")
        if "bottom_padding_size" in aipp_config:
            bottom_padding_size = aipp_config.get("bottom_padding_size")
        if not support_vertical_padding:
            output_h = h - top_padding_size - bottom_padding_size
        else:
            output_h = h
        output_w = w
        actual_col_size = output_h * output_w

    return actual_col_size


def get_padding_size(aipp_config):
    """
    :param aipp_config:
    :return:
    """

    left_padding_size = 0
    right_padding_size = 0
    top_padding_size = 0
    bottom_padding_size = 0
    if "padding" in aipp_config and aipp_config.get("padding") == 1:
        if "left_padding_size" in aipp_config:
            left_padding_size = aipp_config.get("left_padding_size")

        if "right_padding_size" in aipp_config:
            right_padding_size = aipp_config.get("right_padding_size")

        if "top_padding_size" in aipp_config:
            top_padding_size = aipp_config.get("top_padding_size")

        if "bottom_padding_size" in aipp_config:
            bottom_padding_size = aipp_config.get("bottom_padding_size")

    return left_padding_size, right_padding_size, \
           top_padding_size, bottom_padding_size


def vector_dup(ib, dtype, buf, size, padding_value=0):
    """
    set all buffer to padding_value.
    """
    one_cnt = 128
    repeat = size // one_cnt
    remainder = size % one_cnt
    loop_repeat = repeat // 255
    loop_remainder = repeat % 255

    with ib.if_scope(loop_repeat > 0):
        with ib.for_range(0, loop_repeat) as i:
            mask = one_cnt
            offset_repeat = i*one_cnt*255
            tbe_platform.reset_mask_insn(ib, dtype, bits=mask)
            ib.emit(
                tvm.call_extern(dtype, "vector_dup",
                                buf.access_ptr("w", offset=offset_repeat),
                                tvm.const(padding_value, dtype=dtype), 255, 1, 1,
                                8, 8))

    offset_remainder = loop_repeat*one_cnt*255
    with ib.if_scope(loop_remainder > 0):
        mask = one_cnt
        tbe_platform.reset_mask_insn(ib, dtype, bits=mask)
        ib.emit(
            tvm.call_extern(dtype, "vector_dup",
                            buf.access_ptr("w", offset=offset_remainder),
                            tvm.const(padding_value, dtype=dtype), loop_remainder,
                            1, 1, 8, 8))

    offset = one_cnt*loop_remainder + loop_repeat*one_cnt*255
    with ib.if_scope(remainder > 0):
        mask = remainder
        tbe_platform.reset_mask_insn(ib, dtype, bits=128)
        ib.emit(
            tvm.call_extern(dtype, "vector_dup",
                            buf.access_ptr("w", offset=offset),
                            tvm.const(padding_value, dtype=dtype), 1, 1, 1, 8, 8))


def vadds_zero(ib, dtype, dst_ub_buf, src_ub_buf, src_blk_stride, elems_count):
    """
    do vadds, float16
    """
    one_cnt = 128
    repeat = elems_count // one_cnt
    remainder = elems_count % one_cnt
    loop_repeat = repeat // 255
    loop_remainder = repeat % 255

    with ib.if_scope(loop_repeat > 0):
        with ib.for_range(0, loop_repeat) as i:
            mask = one_cnt
            offset_repeat = i*one_cnt*255
            tbe_platform.reset_mask_insn(ib, dtype, bits=mask)
            ib.emit(tvm.call_extern(dtype, "vadds",
                                    dst_ub_buf.access_ptr("w", offset=offset_repeat),
                                    src_ub_buf.access_ptr("r", offset=0),
                                    0, 255, 1, src_blk_stride, 8, 0))

    offset_remainder = loop_repeat*one_cnt*255
    with ib.if_scope(loop_remainder > 0):
        mask = one_cnt
        tbe_platform.reset_mask_insn(ib, dtype, bits=mask)
        ib.emit(tvm.call_extern(dtype, "vadds",
                                dst_ub_buf.access_ptr("w", offset=offset_remainder),
                                src_ub_buf.access_ptr("r", offset=0),
                                0, loop_remainder, 1, src_blk_stride, 8, 0))

    offset = one_cnt*loop_remainder + loop_repeat*one_cnt*255
    with ib.if_scope(remainder > 0):
        mask = remainder
        tbe_platform.reset_mask_insn(ib, dtype, bits=mask)
        ib.emit(tvm.call_extern(dtype, "vadds",
                                dst_ub_buf.access_ptr("w", offset=offset),
                                src_ub_buf.access_ptr("r", offset=0),
                                0, 1, 1, src_blk_stride, 8, 0))


def conv(ib, output_dtype, src_dtype, output_buf, src_buf, size):
    """set buffer to all zero."""

    one_cnt = 128
    repeat = size // one_cnt
    remainder = size % one_cnt
    loop_repeat = repeat // 255
    loop_remainder = repeat % 255

    if output_dtype == "int8":
        inst = "vconv_f162s8"
    else:
        inst = "vconv_f162u8"

    with ib.if_scope(loop_repeat > 0):
        with ib.for_range(0, loop_repeat) as i:
            mask = one_cnt
            offset_repeat = i*one_cnt*255
            tbe_platform.reset_mask_insn(ib, src_dtype, bits=mask)
            ib.emit(
                tvm.call_extern(output_dtype, inst,
                                output_buf.access_ptr("w", offset=offset_repeat),
                                src_buf.access_ptr("rw", offset=offset_repeat),
                                255, 1, 1, 4, 8))

    offset_remainder = loop_repeat*one_cnt*255
    with ib.if_scope(loop_remainder > 0):
        mask = one_cnt
        tbe_platform.reset_mask_insn(ib, src_dtype, bits=mask)
        ib.emit(
            tvm.call_extern(output_dtype, inst,
                            output_buf.access_ptr("w", offset=offset_remainder),
                            src_buf.access_ptr("rw", offset=offset_remainder),
                            loop_remainder, 1, 1, 4, 8))

    offset = one_cnt*loop_remainder + loop_repeat*one_cnt*255
    with ib.if_scope(remainder > 0):
        mask = 128
        tbe_platform.reset_mask_insn(ib, src_dtype, bits=mask)
        ib.emit(
            tvm.call_extern(output_dtype, inst,
                            output_buf.access_ptr("w", offset=offset),
                            src_buf.access_ptr("rw", offset=offset),
                            1, 1, 1, 4, 8))


def copy_ubuf_to_gm_tail(ib, dtype, dst, src, tail_ub, count,
                         dst_offset=0, src_offset=0):
    """

    :param ib:
    :param dtype:
    :param dst:
    :param src:
    :param tail_ub:
    :param count:
    :param dst_offset:
    :param src_offset:
    :return:
    """

    if dtype == "float16":
        block_len = 16
    else:
        block_len = 32

    for i in range(block_len):
        ib.emit(
            tvm.call_extern(
                dtype, 'reg_mov', tail_ub.access_ptr('w', offset=i),
                src.access_ptr('r',
                               offset=src_offset + count - block_len + i)))
    ib.emit(tvm.call_extern(dtype, 'copy_ubuf_to_gm',
                            dst.access_ptr('w', offset=dst_offset + count - block_len),
                            tail_ub.access_ptr('r'), 0, 1, 1, 0, 0))
