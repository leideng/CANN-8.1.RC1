"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

transpose
"""
# 'pylint: disable=too-many-lines

from impl.util import util_common
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.util_tik_comm_func import ceil_align
from impl.util.util_tik_comm_func import floor_align
from impl.util.util_tik_comm_func import ceil_div
from tbe.common.platform.platform_info import api_check_support
from impl.util.util_soc_common import is_v200

# 'pylint: disable=too-few-public-methods
ACCU_BLOCK_SIZE = 128  # should less than 240 for both 310 and 910
ROW_UNIT = 128

# scenario_0
S0_FIXED_PART_SCALA_MAX_NUM = 100

# scenario_1
S1_FIXED_PART_SCALA_MAX_NUM = 100
S1_PERCORE_PART_SCALA_MAX_NUM = 100

# scenario_2
S2_FIXED_PART_SCALA_MAX_NUM = 100
S2_PERCORE_PART_SCALA_MAX_NUM = 100

# scenario_3
S3_FIXED_PART_SCALA_MAX_NUM = 100
S3_PERCORE_PART_SCALA_MAX_NUM = 100

# scenario_4
S4_FIXED_PART_SCALA_MAX_NUM = 100
S4_PERCORE_PART_SCALA_MAX_NUM = 100

# scenario_7
S7_FIXED_PART_SCALA_MAX_NUM = 100
S7_PERCORE_PART_SCALA_MAX_NUM = 100

# scenario_9
S9_FIXED_PART_SCALA_MAX_NUM = 100
S9_PERCORE_PART_SCALA_MAX_NUM = 100

# scenario_10
S10_FIXED_PART_SCALA_MAX_NUM = 100
S10_PERCORE_PART_SCALA_MAX_NUM = 100

TILING_MAX_PARAM_NUM = 512
TILING_MAX_SIZE_GM = 2048  # 16KB
MAX_INT64_VALUE = 2**64 - 1
BLOCK_SIZE = 32
TRANSPOSE_MAX_AXIS_NUM = 8
BORROW_SRC_AXIS_NUM = 2
BORROW_DST_AXIS_NUM = 2
BORROW_SRC_AXIS_LT_NUM = 3
BORROW_DST_AXIS_LT_NUM = 3
UB_REORDER_COMBINATION = 4
REORDER_MAX_AXIS_NUM = 5
RESERVED_UB = 4  # 4KB
EPB2 = 2
EPB4 = 4
EPB8 = 8
EPB16 = 16
EPB32 = 32
ELE_NUM_PER_BLOCK_INT64 = 4
TILING_HEAD_LEN = 5
TILING_FIXED_MAX_LEN = 2048
MTE_PAD_GAP_LIMIT = 2**32 - 1
VG_MASK = 256
MOVE_MAX_STRIDE = 65535
MAX_DATA_MOVE_PAD_REPEATS = 4095
MAX_DATA_MOVE_PAD_BYTES = 2097151


def get_ub_size():
    """
    get_ub_size, get ub size
    """
    return tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)


def get_core_num():
    """
    get_core_num, get core num
    """
    return tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)


def get_cache_line():
    """
    get_cache_line, get cache line
    """
    soc_version = tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION)
    if soc_version in ("Ascend910B", "Ascend910_93"):
        return 512
    else:
        return 256


def is_soc_310p():
    soc_version = tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION)
    return soc_version == "Ascend310P"


def _fuzzy_match(shape_t):
    """
    temporary function, for dynamic & static union version not fully verified
    """
    white_list_shape_fuzzy =  [
                               [-1, 12, 197, 64], [-1, 197, 12, 64], [-1, 197, 768], [-1, 768, 196],
                               [-1, 768, 197], [768, -1, 197], [128, 197, 12, -1], [128, 12, 197, -1],
                               [-1, 3, 300, 18, 2], [-1, 2, 18, 3, 300], [-1, 3, 64, 300, 18], [-1, 3, 128, 300, 18],
                               [-1, 3, 128, 150, 18], [-1, 3, 256, 75, 18], [-1, 3, 256, 150, 18], [-1, 1, 1, 256],
                               [-1, 167, 1], [92, -1, 1], [75, -1, 1], [3, 3, 1, -1], [5, 5, 1, -1]
                              ]
    for shape_w in white_list_shape_fuzzy:
        if len(shape_t) != len(shape_w):
            continue
        count = 0
        for i, _ in enumerate(shape_t):
            if shape_w[i] == -1 or shape_t[i] == shape_w[i]:
                count = count + 1
                continue
            break
        if count == len(shape_t):
            return True
    return False


def _fuzzy_match_black(shape_t):
    """
    temporary function, for dynamic & static union version not fully verified
    """
    black_list_shape_fuzzy =  [
                               #SSDRESNET34 1BATCH
                               [-1, 10, 10, 6, 4], [-1, 10, 10, 6, 81], [-1, 19, 19, 6, 4], [-1, 19, 19, 6, 81],
                               [-1, 38, 38, 4, 4], [-1, 38, 38, 4, 81], [-1, 5, 5, 6, 81], [-1, 1, 1, 4, 81],
                               [-1, 3, 3, 4, 81], [-1, 1, 1, 4, 4], [-1, 3, 3, 4, 4], [-1, 5, 5, 6, 4],
                               #ALBERT
                               [-1, 30, 12, 26], [-1, 12, 30, 26],
                               #FACEBOXES
                               [-1, 8525, 4], [-1, 8525, 2], [-1, 2, 8525], [-1, 4, 200],
                               #RETINANETNEW
                               [-1, 147312, 1, 4], [-1, 147312, 80], [-1, 4, 300],
                               #SSDMOBILENETV1FPNNEW
                               [-1, 392832, 1, 4], [-1, 1, 4, 392832], [-1, 392832, 6],
                               #VGGSSD
                               [-1, 8732, 21], [-1, 21, 8732],
                               [-1, 10, 10, 126], [-1, 10, 10, 24], [-1, 1, 1, 16], [-1, 1, 1, 84], [-1, 126, 10, 10],
                               [-1, 126, 19, 19], [-1, 126, 5, 5], [-1, 16, 1, 1], [-1, 16, 3, 3], [-1, 16, 38, 38],
                               [-1, 19, 19, 126], [-1, 19, 19, 24], [-1, 21, 8732], [-1, 24, 10, 10], [-1, 24, 19, 19],
                               [-1, 24, 5, 5], [-1, 3, 3, 16], [-1, 3, 3, 84], [-1, 38, 38, 16], [-1, 38, 38, 84],
                               [-1, 5, 5, 126], [-1, 5, 5, 24], [-1, 84, 1, 1], [-1, 84, 3, 3], [-1, 84, 38, 38],
                               #TRANSFORMER
                               [145, -1, 8], [21, -1, 8], [25, -1, 8], [273, -1, 8], [-1, 8, 1, 64], [-1, 1, 8, 64],
                               [33, -1, 8], [49, -1, 8], [81, -1, 8], [-1, 4, 8, 64], [-1, 8, 145], [-1, 8, 21],
                               [-1, 8, 25], [-1, 8, 273], [-1, 8, 33], [-1, 8, 8, 64], [-1, 8, 49], [-1, 8, 81],
                               [-1, 8, 8, 64], [-1, 8, 4, 64],  [257, -1, 8], [-1, 8, 257],
                               #Industrial_graph_fusion
                               [-1, 128, 128, 32], [-1, 128, 128, 64], [-1, 16, 256, 256], [-1, 16, 512, 512],
                               [-1, 256, 256, 16], [-1, 256, 256, 32], [-1, 256, 64, 64],
                               [-1, 32, 128, 128], [-1, 512, 512, 16], [-1, 32, 32, 256],
                               [-1, 64, 128, 128], [-1, 64, 64, 256], [-1, 512, 512, 1],
                              ]
    for shape_w in black_list_shape_fuzzy:
        if len(shape_t) != len(shape_w):
            continue
        count = 0
        for i, _ in enumerate(shape_t):
            if shape_w[i] == -1 or shape_t[i] == shape_w[i]:
                count = count + 1
                continue
            break
        if count == len(shape_t):
            return True
    return False


def _static_scenario_goto_old_version(shape, core_num):
    black_list_shape = [
                         [1, 128, 128, 3], [1, 128, 128, 512], [1, 128, 256, 256],
                         [1, 16, 16, 3], [1, 16, 16, 512], [1, 256, 128, 128],
                         [1, 3, 128, 128], [1, 3, 16, 16],
                         [1, 32, 32, 3], [1, 32, 32, 512],  [1, 3, 256, 256],
                         [1, 3, 32, 32], [1, 3, 4, 4], [1, 3, 64, 64],
                         [1, 3, 8, 8], [1, 512, 16, 16], [1, 512, 32, 32],
                         [1, 512, 4, 4], [1, 512, 512, 128], [1, 512, 512, 3],
                         [1, 512, 64, 64], [1, 512, 8, 8], [1, 64, 64, 3],
                         [1, 64, 64, 512], [1, 8, 8, 3], [1, 8, 8, 512],
                         [24, 512, 1024], [24, 512, 4096], [16, 512, 4096], [16, 512, 1024],
                         [256, 224, 224, 3], [64, 224, 224, 3],
                         #SSDMOBILENETV1FPNNEW
                         [4, 392832], [392832, 4], [4, 3142656], [3142656, 4], [4, 1571328], [1571328, 4],
                         [4, 12570624], [12570624, 4], [4, 6, 392832], [4, 6285312], [6285312, 4],
                         [4, 25141248], [25141248, 4],
                         #SSDRESNET34 1BATCH
                         [4, 8732], [8732, 4],
                         #GNMT125
                         [1, 50],
                         #MASKRCNN
                         [1, 100, 90, 4], [1, 100, 91], [1, 160800, 1, 4], [1, 160800, 1], [1, 4, 100], [160800, 2],
                         [160800, 4], [2, 160800], [4, 160800], [4, 9000], [9000, 4],
                         #CTCREVIVEV1024
                         [1, 128, 6410], [1, 16, 6410], [1, 256, 6410], [1, 32, 6410], [1, 64, 6410],
                         #FASTERRCNNRESNET60
                         [1867776, 4], [4, 1867776], [4, 576000], [576000, 4], [64, 100, 90, 4], [64, 100, 91],
                         [64, 29184, 1, 4], [64, 29184, 1], [64, 4, 100],
                         #MTCNNPNET
                         [1, 157, 2, 283], [1, 157, 283, 2],
                         #RCNN
                         [25, 64, 512], [64, 25, 512],
                         #IMAGE_SEG
                         [1, 256, 256, 9], [1, 256, 9, 256],
                         [4, 4, 64, 114], [4, 4, 128, 228], [2, 4, 64, 114], [2, 4, 128, 228],
                         #PG2_PRECISION
                         [10, 4, 64, 114], [1, 12, 28, 28], [1, 21, 28, 28], [1, 24, 1, 1],  [1, 24, 14, 14],
                         [1, 24, 2, 2], [1, 24, 4, 4], [1, 24, 7, 7],  [1, 3948, 7],  [1, 42, 1, 1], [1, 42, 14, 14],
                         [1, 42, 2, 2], [1, 42, 4, 4], [1, 42, 7, 7], [1, 7, 3948], [2, 384], [24, 4, 64, 114],
                         [384, 2], [6, 4, 128, 228],
                      ]
    shape_t = list(shape)
    if shape_t in black_list_shape:
        return True

    if _fuzzy_match_black(shape_t):
        return True

    return False

def _nd_to_nz_shape_mismatch(input_x, output_y):
    x_shape = input_x.get("shape")
    x_format = input_x.get("format")
    y_shape = output_y.get("shape")
    y_format = output_y.get("format")
    x_shape_t = list(x_shape)
    y_shape_t = list(y_shape)
    if (x_format == "ND") and (y_format == "FRACTAL_NZ"):
        if (len(x_shape_t) == 5) and (len(y_shape_t) == 4):
            return True
    return False

def _by_dynamic_static_union_version(shape, core_num):
    """
    temporary function, for dynamic & static union version not fully verified
    """
    if core_num == 1:
        white_list_shape_lhisi = [[1, 24, 3, 20], [8, 128, 3, 21]]
        shape_t_lhisi = list(shape)
        if shape_t_lhisi in white_list_shape_lhisi:
            return True
        return False

    white_list_shape = [
                         [2, 512, 1024], [1024, 91], [2, 512, 1024], [256, 784, 91],
                         [1024, 364], [2, 128, 91, 28, 28], [2, 128, 28, 28, 91],
                         [1024, 1024], [2, 512, 1024], [12544, 1024], [2, 512, 12544],
                         [4, 2, 4, 2, 3, 64], [1100, 1100], [2, 100, 1], [200, 116, 116, 4],
                         [1100], [1100, 512], [1, 512, 1, 24], [1, 512, 24], [38, 67, 512], [67, 38, 512],
                         [1, 24, 5, 5], [1, 486, 5, 5], [1, 24, 10, 10], [1, 486, 10, 10],
                         [1, 24, 20, 20], [1, 486, 20, 20], [1, 24, 40, 40], [1, 486, 40, 40],
                         [1, 24, 80, 80], [1, 486, 80, 80], [12, 8, 8, 36, 120],
                         [1, 100, 28, 28, 91], [4, 100, 28, 28, 91], [8, 100, 28, 28, 91], [16, 100, 28, 28, 91],
                         [80, 8, 1, 240], [80, 240, 8], [80, 240, 1, 8], [8, 80, 240], [240, 8, 64], [80, 8, 84],
                         [8, 80, 64], [1, 4, 1080, 1920, 3], [2, 100, 28, 28, 91], [2560, 26, 512],
                         [16, 40, 3, 14, 14], [16, 80, 3, 7, 7], [16, 20, 3, 28, 28],
                         [32, 3, 76, 76, 85], [32, 3, 38, 38, 85], [32, 3, 19, 19, 85], [32, 3, 85, 76, 76],
                         [32, 3, 85, 38, 38], [32, 3, 85, 19, 19], [512, 512, 9], [16, 12, 512, 64],
                         [21, 1, 3000], [21, 16, 3000], [768, 1, 256], [768, 8, 32], [768, 4, 256],
                         [768, 32, 32], [768, 8, 256], [768, 64, 32], [768, 32, 256], [768, 256, 32],
                         [3136, 8, 2, 4, 6], [3136, 4, 2, 4, 6], [3136, 1, 2, 4, 6], [3136, 32, 2, 4, 6],
                         [3, 256, 1024], [48, 56, 64], [3, 16, 256, 64], [3, 1024, 256], [3, 3, 16, 16, 16, 16],
                         [3, 256, 16, 64], [48, 256, 64], [48, 256, 256], [768, 768], [3072, 768], [768, 197, 197],
                         [768, 3072], [512, 512, 3, 3], [8, 8732, 81], [8, 81, 8732], [2, 1, 1, 256],
                         [640, 320, 3, 3], [1280, 640, 3, 3], [640, 640, 3, 3], [256, 256, 3, 3], [128, 128, 3, 3],
                         [256, 256, 2, 2], [160, 160, 3, 3], [320, 320, 3, 3], [320, 160, 3, 3],
                         [1, 224, 224, 160, 4], [4, 224, 224, 160, 4], [1, 448, 448, 2, 2], [1, 224, 224, 2, 2],
                         [1, 64, 2, 2, 1020, 1020], [1, 64, 2, 2, 256, 256], [32, 256, 2, 2, 16, 12],
                         [32, 128, 2, 2, 32, 64], [32, 128, 2, 2, 16, 12], [256, 256, 2, 2],
                         [4, 3, 2, 2, 1020, 1020], [8, 3, 2, 2, 1020, 1020], [128, 224, 224, 3],
                         [1, 3, 2, 2, 1020, 1020], [4, 3, 2, 2, 1020, 1020], [8, 3, 2, 2, 1020, 1020],
                         [195, 128, 32, 61], [128, 32, 195, 61], [1, 64, 2, 2, 114, 114], [24, 128, 64, 131],
                         [24, 131, 64, 128], [195, 128], [192, 16, 8, 1, 64],
                         [8, 3, 40, 40, 85],
                         [768, 16, 256], [768, 128, 32], [3136, 16, 2, 4, 6], [16, 3, 16, 14, 16, 14],
                         [1024, 49, 3, 49], [1024, 3, 49, 49], [1024, 3, 49, 32], [1024, 49, 3, 3, 32],
                         [256, 49, 6, 49], [256, 6, 49, 49], [256, 6, 49, 32], [64, 12, 49, 49],
                         [64, 49, 12, 49], [64, 12, 49, 32],
                         [5120, 5, 31, 31], [5120, 32, 2, 80], [5120, 2, 2, 1280], [5120, 4, 32],
                         [5120, 32, 4], [1650, 5, 31, 31], [1650, 4, 128], [1650, 128, 4],
                         [1, 21, 8732], [1, 8732, 21], [16, 64, 2, 2, 48, 48], [16, 64, 48, 2, 48, 2],
                         [8, 3, 160, 160, 85], [8, 3, 80, 80, 85], [8, 3, 40, 40, 85], [8, 3, 20, 20, 85],
                         [8, 3, 85, 160, 160], [8, 3, 85, 80, 80],
                         [1, 512, 1, 2], [1, 256, 1, 2], [1, 128, 1, 2], [1, 64, 1, 2],
                         [1, 128, 1, 2, 4], [1, 64, 1, 2, 4],
                         [256, 16, 257, 64], [50, 96, 64], [50, 192, 64], [50, 384, 64],
                         [16, 512, 1, 2], [16, 256, 1, 2], [16, 128, 1, 2], [16, 64, 1, 2], [16, 128, 1, 2, 4],
                         [16, 64, 1, 2, 4], [24, 131, 64, 128], [16, 3, 72, 72, 85], [16, 3, 68, 68, 85],
                         [16, 3, 52, 52, 85], [16, 3, 48, 48, 85], [16, 3, 44, 44, 85], [16, 3, 36, 36, 85],
                         [16, 3, 34, 34, 85], [16, 3, 26, 26, 85], [16, 3, 24, 24, 85], [16, 3, 22, 22, 85],
                         [16, 3, 18, 18, 85], [16, 3, 17, 17, 85], [16, 3, 13, 13, 85], [16, 3, 12, 12, 85],
                         [16, 3, 11, 11, 85], [16, 3, 85, 72, 72], [16, 3, 85, 68, 68], [16, 3, 85, 48, 48],
                         [16, 3, 85, 34, 34], [16, 3, 85, 24, 24], [16, 3, 85, 18, 18],
                         [1000, 5, 64, 64], [1000, 12, 48, 32],
                         [16, 2, 4, 4, 32, 32], [16, 1, 4, 4, 32, 32], [16, 1, 32, 4, 32, 4], [16, 2, 32, 4, 32, 4],
                         [16, 3, 85, 80, 80], [16, 3, 85, 40, 40], [16, 3, 85, 20, 20],
                         [16, 3, 20, 20, 85], [16, 3, 40, 40, 85], [16, 3, 80, 80, 85],
                         [128, 1, 32, 32], [128, 5], [16, 16, 1, 256], [256, 1, 16, 16],
                         [32, 32, 1, 128], [512, 1, 8, 8], [8, 8, 1, 512],
                         [64, 64, 2, 2, 44, 44], [190, 3, 56, 4, 56, 4],
                         [8, 128, 64, 484], [190, 3, 224, 224], [190, 4, 4, 3, 56, 56], [64, 256, 44, 44],
                         [1, 960, 960, 3], [4, 32, 512, 3], [4, 64, 512, 3],
                         [8, 51150, 91], [16, 51150, 91], [18, 51150, 91],
                         [64, 64, 44, 2, 44, 2], [8, 3, 160, 160, 85], [8, 3, 80, 80, 85], [8, 3, 20, 20, 85],
                         [64, 64, 2, 2, 64, 64], [64, 64, 64, 2, 64, 2],
                         [64, 64, 3, 3, 64, 64], [64, 64, 64, 3, 64, 3],
                         [64, 64, 2, 2, 128, 128], [64, 64, 128, 2, 128, 2],
                         [32, 64, 4, 4, 30, 90], [32, 64, 30, 4, 90, 4],
                         [4, 3, 68, 4, 116, 4], [4, 3, 4, 4, 68, 116],
                         [128, 256, 8, 16], [128, 16, 8, 128], [128, 128, 8, 16], [8, 128, 2, 16],
                         [128, 8, 16, 2], [16, 8, 128, 2], [8, 16, 128, 256], [256, 8, 16, 64],
                         [128, 128, 8, 2], [8, 16, 128, 128],
                         [16, 64, 2, 2, 64, 64], [16, 64, 64, 2, 64, 2], [16, 64, 2, 2, 32, 32],
                         [16, 64, 32, 2, 32, 2], [4, 32, 64, 512, 14], [512, 5, 31, 31],
                         [105, 1, 105, 20], [105, 1, 105, 8], [53, 1, 53, 20], [105, 1, 105, 4],
                         [53, 1, 53, 8], [27, 1, 27, 20], [51, 3, 53, 4],
                         [4, 3, 52, 52, 25], [4, 3, 26, 26, 25], [4, 3, 13, 13, 25],
                         [2190, 40, 2190], [2190, 32, 2190], [1484, 40, 1484], [1484, 32, 1484],
                         [644, 40, 644], [644, 32, 644], [14, 1, 14, 768],
                         [644, 40, 644], [644, 32, 644], [14, 1, 14, 768],
                         [16, 77, 12, 64], [16, 12, 77, 64], [44, 77, 12, 64], [44, 12, 77, 64],
                         [16, 12, 64, 77],
                         [32, 1, 3, 3], [1, 32, 3, 3], [96, 1, 3, 3], [1, 96, 3, 3],
                         [144, 1, 3, 3], [1, 144, 3, 3], [192, 1, 3, 3], [1, 192, 3, 3], [384, 1, 3, 3],
                         [1, 384, 3, 3], [576, 1, 3, 3], [1, 576, 3, 3], [960, 1, 3, 3], [1, 960, 3, 3],
                         [8, 128, 20, 20],
                         [8, 8, 20, 20, 16], [8, 64, 40, 40], [8, 4, 40, 40, 16], [8, 32, 80, 80],
                         [8, 2, 80, 80, 16], [8, 16, 160, 160], [8, 1, 160, 160, 16], [8, 320, 320, 3],
                         [8, 3, 320, 320], [3, 3, 32, 1], [3, 3, 96, 1], [3, 3, 144, 1], [3, 3, 192, 1],
                         [3, 3, 384, 1], [3, 3, 576, 1], [3, 3, 960, 1], [1, 960, 3, 3],
                         [64, 3, 7, 32, 7, 32], [64, 50, 8, 1, 16], [960, 384, 64], [960, 64, 384],
                         [960, 384, 384],  [5120, 5, 31, 31], [128, 128, 62], [128, 62, 256],
                         [62, 128, 256], [62, 256, 128], [16, 128, 128, 1], [8, 128, 128, 1],
                         [4, 128, 128, 1], [2, 128, 128, 1], [62, 128, 128],
                         [98, 8, 8, 64]
                       ]
    shape_t = list(shape)
    if shape_t in white_list_shape:
        return True

    if _fuzzy_match(shape_t):
        return True

    return False


def change_dtype(x_dtype, y_dtype):
    """
    change dtype to support bfloat16 etc.
    """
    if x_dtype == "bool":
        x_dtype = "uint8"
    if y_dtype == "bool":
        y_dtype = "uint8"
    if x_dtype == "bfloat16":
        x_dtype = "float16"
    if y_dtype == "bfloat16":
        y_dtype = "float16"
    return x_dtype, y_dtype


# 'pylint: disable=unused-argument,too-many-return-statements
def check_supported(input_x, perm, output_y, kernel_name="dynamic_transpose"):
    """
    dynamic transpose is selected when any condition is true: \n
        -1 in input_x shape \n
        -1 in output_y shape \n
        -2 in input_x shape \n
    """
    x_shape = input_x.get("ori_shape")
    x_dtype = input_x.get("dtype")

    check_list = ["int8", "uint8", "bool", "float", "float32", "int32", "uint32", "int16", "uint16", "float16", "int64",
                  "uint64", "bfloat16"]
    if x_dtype not in check_list:
        reason = "x_dtype [%s] not in %s" % (x_dtype, str(check_list))
        return False, reason

    if util_common.is_unknown([input_x, perm, output_y]):
        return True, ""

    if _nd_to_nz_shape_mismatch(input_x, output_y):
        return False, "nd_to_nz_shape_mismatch"

    if _by_dynamic_static_union_version(x_shape, get_core_num()):
        return True, ""

    if tbe_context.get_context():
        if hasattr(tbe_context.get_context(), "get_build_type"):
            if tbe_context.get_context().get_build_type() == "fuzzily_build":
                return True, ""

    if tbe_platform.api_check_support("tik.vcopy"):
        return True, ""

    if _static_scenario_goto_old_version(x_shape, get_core_num()):
        return False, ""

    return True, ""


# 'pylint: disable=too-many-statements, too-many-arguments
def _set_param_python_arr(tiling_reg_list, reg_base, ub_input, ub_offset, param, actual_num, max_num):
    for i in range(max_num):
        tiling_reg_list[reg_base[0] + i].set_as(ub_input[ub_offset + i])
        param.append(tiling_reg_list[reg_base[0] + i])
    ub_offset.set_as(ub_offset + actual_num)
    reg_base[0] = reg_base[0] + max_num


# 'pylint: disable=too-many-statements, too-many-arguments
def _set_param_scalar_arr(tiling_reg_list, ub_input, ub_offset, param, actual_num, max_num):
    for i in range(max_num):
        param[i].set_as(ub_input[ub_offset + i])
    ub_offset.set_as(ub_offset + actual_num)


def _get_half_ub():
    # first div 2 means half the ub, second div 2 means b16
    return (get_ub_size() - RESERVED_UB * 1024) // 2 // 2


# 'pylint: disable=unused-argument,invalid-name, too-many-arguments, unused-variable, too-many-locals
# 'pylint: disable=too-many-statements, invalid-name, no-self-use, protected-access
# 'pylint: disable=too-many-instance-attributes, too-few-public-methods
class Transpose:
    """
    Transpose
    """

    class TilingParamS0:
        """
        TilingParamS0
        """
        def __init__(self, tiling_reg_list, ub_input_64_t, ub_input_64):
            """
            get tiling parameters
            """
            # part 2: fixed

            for i in range(2):
                tiling_reg_list[i].set_as(ub_input_64_t[TILING_HEAD_LEN + i])
            self.core_num = tiling_reg_list[0]
            self.ub_size = tiling_reg_list[1]

            #part 3 : percore
            reg_base = S0_FIXED_PART_SCALA_MAX_NUM
            self.base = tiling_reg_list[reg_base + 0]
            self.ele_num = tiling_reg_list[reg_base + 1]
            self.major_loop = tiling_reg_list[reg_base + 2]
            self.major_num = tiling_reg_list[reg_base + 3]
            self.tail_num = tiling_reg_list[reg_base + 4]
            self.not_align_ele = tiling_reg_list[reg_base + 5]

            self.base.set_as(ub_input_64[0])
            self.ele_num.set_as(ub_input_64[1])
            self.major_loop.set_as(ub_input_64[2])
            self.major_num.set_as(ub_input_64[3])
            self.tail_num.set_as(ub_input_64[4])
            self.not_align_ele.set_as(ub_input_64[5])

    class TilingParamS1:
        """
        TilingParamS1
        """
        def __init__(self, tiling_reg_list, ub_input_64_t, ub_input_64, tik_inst):
            """
            get tiling parameters
            """
            # part 2: fixed
            ub_offset = tik_inst.Scalar("int32", init_value=TILING_HEAD_LEN)

            for i in range(6):
                tiling_reg_list[i].set_as(ub_input_64_t[ub_offset + i])
            self.core_num = tiling_reg_list[0]
            self.ub_size = tiling_reg_list[1]
            self.last_axis_len = tiling_reg_list[2]
            self.last_axis_burst_len = tiling_reg_list[3]
            self.align_ele = tiling_reg_list[4]
            self.trans_axis_num = tiling_reg_list[5]

            reg_base = 6
            ub_offset.set_as(TILING_HEAD_LEN + 6)
            cycle = 3
            self.src_jump_stride = []
            self.dst_jump_stride = []
            self.dst_jump_factor = []
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.src_jump_stride.append(tiling_reg_list[reg_base + i * cycle + 0])
                self.dst_jump_stride.append(tiling_reg_list[reg_base + i * cycle + 1])
                self.dst_jump_factor.append(tiling_reg_list[reg_base + i * cycle + 2])

            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.src_jump_stride[i].set_as(ub_input_64_t[ub_offset + i + 0 * self.trans_axis_num])
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.dst_jump_stride[i].set_as(ub_input_64_t[ub_offset + i + 1 * self.trans_axis_num])
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.dst_jump_factor[i].set_as(ub_input_64_t[ub_offset + i + 2 * self.trans_axis_num])

            # part 3 : percore
            ub_offset.set_as(0)
            reg_base = S1_FIXED_PART_SCALA_MAX_NUM
            self.loop_num = tiling_reg_list[reg_base]
            self.aggregate_loop_unit = tiling_reg_list[reg_base + 1]
            self.aggregate_loop_num = tiling_reg_list[reg_base + 2]
            self.aggregate_loop_tail = tiling_reg_list[reg_base + 3]

            self.loop_num.set_as(ub_input_64[ub_offset])
            self.aggregate_loop_unit.set_as(ub_input_64[ub_offset + 1])
            self.aggregate_loop_num.set_as(ub_input_64[ub_offset + 2])
            self.aggregate_loop_tail.set_as(ub_input_64[ub_offset + 3])

            ub_offset.set_as(4)
            reg_base = reg_base + 4

            self.init_tuple = []
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.init_tuple.append(tiling_reg_list[reg_base + i])
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.init_tuple[i].set_as(ub_input_64[ub_offset + i])

            # part 4: variable
            reg_base = S1_FIXED_PART_SCALA_MAX_NUM + S1_PERCORE_PART_SCALA_MAX_NUM
            cycle = 1
            self.rt_tuple = []
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.rt_tuple.append(tiling_reg_list[reg_base + i * cycle + 0])

            reg_base = S1_FIXED_PART_SCALA_MAX_NUM + S1_PERCORE_PART_SCALA_MAX_NUM + cycle * TRANSPOSE_MAX_AXIS_NUM
            self.src_addr = tiling_reg_list[reg_base]
            self.dst_addr = tiling_reg_list[reg_base + 1]

    class TilingParamS2:
        """
        TilingParamS2
        """
        def __init__(self, tiling_reg_list, ub_input_64_t, ub_input_64, tik_inst):
            """
            get tiling parameters
            """
            # part 2: fixed
            for i in range(9):
                tiling_reg_list[i].set_as(ub_input_64_t[TILING_HEAD_LEN + i])
            self.core_num = tiling_reg_list[0]
            self.ub_size = tiling_reg_list[1]
            self.last_axis_len = tiling_reg_list[2]
            self.last_axis_burst_len = tiling_reg_list[3]
            self.align_ele = tiling_reg_list[4]
            self.trans_axis_num = tiling_reg_list[5]
            self.src_stride = tiling_reg_list[6]
            self.back_num = tiling_reg_list[7]
            self.skip_ele = tiling_reg_list[8]

            reg_base = 9
            ub_offset = tik_inst.Scalar("int32", init_value=TILING_HEAD_LEN + 9)
            cycle = 4
            self.src_jump_stride = []
            self.dst_jump_stride = []
            self.dst_jump_factor = []
            self.dst_jump_factor_mod = []
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.src_jump_stride.append(tiling_reg_list[reg_base + i * cycle + 0])
                self.dst_jump_stride.append(tiling_reg_list[reg_base + i * cycle + 1])
                self.dst_jump_factor.append(tiling_reg_list[reg_base + i * cycle + 2])
                self.dst_jump_factor_mod.append(tiling_reg_list[reg_base + i * cycle + 3])

            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.src_jump_stride[i].set_as(ub_input_64_t[ub_offset + i])
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.dst_jump_stride[i].set_as(ub_input_64_t[ub_offset + i + 1 * self.trans_axis_num])
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.dst_jump_factor[i].set_as(ub_input_64_t[ub_offset + i + 2 * self.trans_axis_num])
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.dst_jump_factor_mod[i].set_as(ub_input_64_t[ub_offset + i + 3 * self.trans_axis_num])

            # part 3 : percore
            ub_offset.set_as(0)
            reg_base = S7_FIXED_PART_SCALA_MAX_NUM

            self.base = tiling_reg_list[reg_base]
            self.base.set_as(ub_input_64[ub_offset])
            ub_offset.set_as(ub_offset + 1)
            reg_base = reg_base + 1

            self.loop_num = tiling_reg_list[reg_base]
            self.loop_num.set_as(ub_input_64[ub_offset])
            ub_offset.set_as(ub_offset + 1)
            reg_base = reg_base + 1

            self.init_tuple = []
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.init_tuple.append(tiling_reg_list[reg_base + i])
            reg_base = reg_base + TRANSPOSE_MAX_AXIS_NUM
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.init_tuple[i].set_as(ub_input_64[ub_offset + i])
            ub_offset.set_as(ub_offset + self.trans_axis_num)
            reg_base = reg_base + TRANSPOSE_MAX_AXIS_NUM

            self.head_major_loop = tiling_reg_list[reg_base + 0]
            self.head_major_num = tiling_reg_list[reg_base + 1]
            self.head_tail_num = tiling_reg_list[reg_base + 2]
            self.body_loop = tiling_reg_list[reg_base + 3]
            self.body_major_loop = tiling_reg_list[reg_base + 4]
            self.body_major_num = tiling_reg_list[reg_base + 5]
            self.body_tail_num = tiling_reg_list[reg_base + 6]
            self.tail_major_loop = tiling_reg_list[reg_base + 7]
            self.tail_major_num = tiling_reg_list[reg_base + 8]
            self.tail_tail_num = tiling_reg_list[reg_base + 9]

            self.head_major_loop.set_as(ub_input_64[ub_offset + 0])
            self.head_major_num.set_as(ub_input_64[ub_offset + 1])
            self.head_tail_num.set_as(ub_input_64[ub_offset + 2])
            self.body_loop.set_as(ub_input_64[ub_offset + 3])
            self.body_major_loop.set_as(ub_input_64[ub_offset + 4])
            self.body_major_num.set_as(ub_input_64[ub_offset + 5])
            self.body_tail_num.set_as(ub_input_64[ub_offset + 6])
            self.tail_major_loop.set_as(ub_input_64[ub_offset + 7])
            self.tail_major_num.set_as(ub_input_64[ub_offset + 8])
            self.tail_tail_num.set_as(ub_input_64[ub_offset + 9])
            ub_offset.set_as(ub_offset + 10)

            # part 4: variable
            reg_base = S2_FIXED_PART_SCALA_MAX_NUM + S2_PERCORE_PART_SCALA_MAX_NUM
            cycle = 1
            self.rt_tuple = []
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.rt_tuple.append(tiling_reg_list[reg_base + i * cycle + 0])

            reg_base = S2_FIXED_PART_SCALA_MAX_NUM + S2_PERCORE_PART_SCALA_MAX_NUM + cycle * TRANSPOSE_MAX_AXIS_NUM
            self.src_addr = tiling_reg_list[reg_base]
            self.dst_addr = tiling_reg_list[reg_base + 1]

    class TilingParamS3:
        """
        TilingParamS3
        """
        def __init__(self, tiling_reg_list, ub_input_64_t, ub_input_64, tik_inst):
            """
            get tiling parameters
            """
            # part 2: fixed
            ub_offset = tik_inst.Scalar("int32", init_value=TILING_HEAD_LEN)

            for i in range(10):
                tiling_reg_list[i].set_as(ub_input_64_t[ub_offset + i])
            self.core_num = tiling_reg_list[0]
            self.ub_size = tiling_reg_list[1]
            self.last_axis_len = tiling_reg_list[2]
            self.last_axis_burst_len = tiling_reg_list[3]
            self.align_ele = tiling_reg_list[4]
            self.trans_axis_num = tiling_reg_list[5]
            self.major_loop_num = tiling_reg_list[6]
            self.major_blocks = tiling_reg_list[7]
            self.tail_blocks = tiling_reg_list[8]
            self.back_ele = tiling_reg_list[9]

            reg_base = 10
            ub_offset.set_as(ub_offset + reg_base)
            cycle = 3
            self.src_jump_stride = []
            self.dst_jump_stride = []
            self.dst_jump_factor = []
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.src_jump_stride.append(tiling_reg_list[reg_base + i * cycle + 0])
                self.dst_jump_stride.append(tiling_reg_list[reg_base + i * cycle + 1])
                self.dst_jump_factor.append(tiling_reg_list[reg_base + i * cycle + 2])

            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.src_jump_stride[i].set_as(ub_input_64_t[ub_offset + i + 0 * self.trans_axis_num])
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.dst_jump_stride[i].set_as(ub_input_64_t[ub_offset + i + 1 * self.trans_axis_num])
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.dst_jump_factor[i].set_as(ub_input_64_t[ub_offset + i + 2 * self.trans_axis_num])

            # part 3 : percore
            ub_offset.set_as(0)
            reg_base = S3_FIXED_PART_SCALA_MAX_NUM
            self.loop_num = tiling_reg_list[reg_base]
            self.loop_num.set_as(ub_input_64[ub_offset])

            ub_offset.set_as(1)
            reg_base = S3_FIXED_PART_SCALA_MAX_NUM + 1
            self.init_tuple = []
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.init_tuple.append(tiling_reg_list[reg_base + i])
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.init_tuple[i].set_as(ub_input_64[ub_offset + i])

            # part 4: variable
            reg_base = S3_FIXED_PART_SCALA_MAX_NUM + S3_PERCORE_PART_SCALA_MAX_NUM
            cycle = 1
            self.rt_tuple = []
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.rt_tuple.append(tiling_reg_list[reg_base + i * cycle + 0])

            reg_base = S3_FIXED_PART_SCALA_MAX_NUM + S3_PERCORE_PART_SCALA_MAX_NUM + cycle * TRANSPOSE_MAX_AXIS_NUM
            self.src_addr = tiling_reg_list[reg_base]
            self.dst_addr = tiling_reg_list[reg_base + 1]

    class TilingParamS4:
        """
        TilingParamS4
        """
        def __init__(self, tiling_reg_list, ub_input_64_t, ub_input_64, tik_inst):
            """
            get tiling parameters
            """
            # part 2: fixed
            ub_offset = tik_inst.Scalar("int32", init_value=TILING_HEAD_LEN)

            reg_base = [31]
            for i in range(reg_base[0]):
                tiling_reg_list[i].set_as(ub_input_64_t[ub_offset + i])
            self.last_axis_len = tiling_reg_list[0]
            self.last_axis_burst_len = tiling_reg_list[1]
            self.align_ele = tiling_reg_list[2]
            self.logic_axis_num = tiling_reg_list[3]
            self.other_axis_num = tiling_reg_list[4]
            self.src_axis_num_no_dup = tiling_reg_list[5]
            self.dst_axis_num_no_dup = tiling_reg_list[6]
            self.major_burst_len_in = tiling_reg_list[7]
            self.tail_burst_len_in = tiling_reg_list[8]
            self.major_burst_len_out = tiling_reg_list[9]
            self.tail_burst_len_out = tiling_reg_list[10]
            self.major_dst_loop_in = tiling_reg_list[11]
            self.tail_dst_loop_in = tiling_reg_list[12]
            self.major_src_loop_out = tiling_reg_list[13]
            self.tail_src_loop_out = tiling_reg_list[14]
            self.major_in_ele = tiling_reg_list[15]
            self.tail_in_ele = tiling_reg_list[16]
            self.major_in_tail_ele = tiling_reg_list[17]
            self.tail_in_tail_ele = tiling_reg_list[18]
            self.major_out_ele = tiling_reg_list[19]
            self.tail_out_ele = tiling_reg_list[20]
            self.major_out_tail_ele = tiling_reg_list[21]
            self.tail_out_tail_ele = tiling_reg_list[22]
            self.dst_jump_major_step = tiling_reg_list[23]
            self.src_jump_major_step = tiling_reg_list[24]
            self.dup_axis = tiling_reg_list[25]
            self.src_axis_perm = tiling_reg_list[26]
            self.dst_axis_perm = tiling_reg_list[27]
            self.ub_axis_perm = tiling_reg_list[28]
            self.pivot_src_axis_dup = tiling_reg_list[29]
            self.pivot_dst_axis_dup = tiling_reg_list[30]

            ub_offset.set_as(ub_offset + reg_base[0])

            self.loop_1 = tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION)
            self.loop_2 = tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION)
            self.loop_3 = tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION)

            self.repeat_1 = tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION)
            self.repeat_2 = tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION)
            self.repeat_3 = tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION)

            self.burst_len_1 = tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION)
            self.burst_len_2 = tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION)
            self.burst_len_3 = tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION)

            self.src_stride_1 = tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION)
            self.src_stride_2 = tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION)
            self.src_stride_3 = tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION)

            self.dst_stride_1 = tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION)
            self.dst_stride_2 = tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION)
            self.dst_stride_3 = tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION)

            self.src_offset_1 = tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION)
            self.src_offset_2 = tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION)
            self.src_offset_3 = tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION)

            self.dst_offset_1 = tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION)
            self.dst_offset_2 = tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION)
            self.dst_offset_3 = tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION)

            cycle = 21
            for i in range(UB_REORDER_COMBINATION):
                self.loop_1[i].set_as(ub_input_64_t[ub_offset + i * cycle + 0])
                self.loop_2[i].set_as(ub_input_64_t[ub_offset + i * cycle + 1])
                self.loop_3[i].set_as(ub_input_64_t[ub_offset + i * cycle + 2])
                self.repeat_1[i].set_as(ub_input_64_t[ub_offset + i * cycle + 3])
                self.repeat_2[i].set_as(ub_input_64_t[ub_offset + i * cycle + 4])
                self.repeat_3[i].set_as(ub_input_64_t[ub_offset + i * cycle + 5])
                self.src_stride_1[i].set_as(ub_input_64_t[ub_offset + i * cycle + 6])
                self.src_stride_2[i].set_as(ub_input_64_t[ub_offset + i * cycle + 7])
                self.src_stride_3[i].set_as(ub_input_64_t[ub_offset + i * cycle + 8])
                self.dst_stride_1[i].set_as(ub_input_64_t[ub_offset + i * cycle + 9])
                self.dst_stride_2[i].set_as(ub_input_64_t[ub_offset + i * cycle + 10])
                self.dst_stride_3[i].set_as(ub_input_64_t[ub_offset + i * cycle + 11])
                self.burst_len_1[i].set_as(ub_input_64_t[ub_offset + i * cycle + 12])
                self.burst_len_2[i].set_as(ub_input_64_t[ub_offset + i * cycle + 13])
                self.burst_len_3[i].set_as(ub_input_64_t[ub_offset + i * cycle + 14])
                self.src_offset_1[i].set_as(ub_input_64_t[ub_offset + i * cycle + 15])
                self.src_offset_2[i].set_as(ub_input_64_t[ub_offset + i * cycle + 16])
                self.src_offset_3[i].set_as(ub_input_64_t[ub_offset + i * cycle + 17])
                self.dst_offset_1[i].set_as(ub_input_64_t[ub_offset + i * cycle + 18])
                self.dst_offset_2[i].set_as(ub_input_64_t[ub_offset + i * cycle + 19])
                self.dst_offset_3[i].set_as(ub_input_64_t[ub_offset + i * cycle + 20])

            ub_offset.set_as(ub_offset + UB_REORDER_COMBINATION * cycle)

            self.dst_jump_factor_in = []
            self.src_jump_factor_out = []
            self.logic_jump_factor = []
            self.src_stride_out = []

            self.dst_stride_in = tik_inst.ScalarArray("int64", length=BORROW_DST_AXIS_NUM)
            self.src_stride_out = tik_inst.ScalarArray("int64", length=BORROW_SRC_AXIS_NUM)
            self.logic_stride_in = tik_inst.ScalarArray("int64", length=TRANSPOSE_MAX_AXIS_NUM)
            self.logic_stride_out = tik_inst.ScalarArray("int64", length=TRANSPOSE_MAX_AXIS_NUM)

            _set_param_python_arr(tiling_reg_list, reg_base, ub_input_64_t, ub_offset, self.dst_jump_factor_in,
                                  self.dst_axis_num_no_dup, BORROW_DST_AXIS_NUM)
            _set_param_python_arr(tiling_reg_list, reg_base, ub_input_64_t, ub_offset, self.src_jump_factor_out,
                                  self.src_axis_num_no_dup, BORROW_SRC_AXIS_NUM)
            _set_param_python_arr(tiling_reg_list, reg_base, ub_input_64_t, ub_offset, self.logic_jump_factor,
                                  self.logic_axis_num, TRANSPOSE_MAX_AXIS_NUM)
            _set_param_scalar_arr(tiling_reg_list, ub_input_64_t, ub_offset, self.dst_stride_in,
                                  self.dst_axis_num_no_dup, BORROW_DST_AXIS_NUM)
            _set_param_scalar_arr(tiling_reg_list, ub_input_64_t, ub_offset, self.src_stride_out,
                                  self.src_axis_num_no_dup, BORROW_SRC_AXIS_NUM)
            _set_param_scalar_arr(tiling_reg_list, ub_input_64_t, ub_offset, self.logic_stride_in,
                                  self.logic_axis_num, TRANSPOSE_MAX_AXIS_NUM)
            _set_param_scalar_arr(tiling_reg_list, ub_input_64_t, ub_offset, self.logic_stride_out,
                                  self.logic_axis_num, TRANSPOSE_MAX_AXIS_NUM)

            # part 3: percore
            tiling_reg_list[reg_base[0]].set_as(ub_input_64[0])
            self.loop_per_core = tiling_reg_list[reg_base[0]]
            reg_base[0] = reg_base[0] + 1
            ub_offset.set_as(1)

            self.init_src_tuple_out = []
            self.init_dst_tuple_in = []
            self.init_logic_tuple = []

            _set_param_python_arr(tiling_reg_list, reg_base, ub_input_64, ub_offset, self.init_src_tuple_out,
                                  self.src_axis_num_no_dup, BORROW_SRC_AXIS_NUM)
            _set_param_python_arr(tiling_reg_list, reg_base, ub_input_64, ub_offset, self.init_dst_tuple_in,
                                  self.dst_axis_num_no_dup, BORROW_DST_AXIS_NUM)
            _set_param_python_arr(tiling_reg_list, reg_base, ub_input_64, ub_offset, self.init_logic_tuple,
                                  self.other_axis_num + 2, TRANSPOSE_MAX_AXIS_NUM)

            # part 4: variable
            reg_base[0] = S4_FIXED_PART_SCALA_MAX_NUM + S4_PERCORE_PART_SCALA_MAX_NUM

            self.rt_dst_tuple_in = tik_inst.ScalarArray("int64", length=BORROW_DST_AXIS_NUM)
            self.rt_src_tuple_out = tik_inst.ScalarArray("int64", length=BORROW_SRC_AXIS_NUM)
            self.rt_logic_tuple = tik_inst.ScalarArray("int64", length=TRANSPOSE_MAX_AXIS_NUM)

            self.rt_src_tuple_logic = tiling_reg_list[reg_base[0] + 0]
            self.rt_dst_tuple_logic = tiling_reg_list[reg_base[0] + 1]
            self.src_addr = tiling_reg_list[reg_base[0] + 2]
            self.dst_addr = tiling_reg_list[reg_base[0] + 3]
            self.offset_1 = tiling_reg_list[reg_base[0] + 4]
            self.offset_2 = tiling_reg_list[reg_base[0] + 5]
            self.offset_a = tiling_reg_list[reg_base[0] + 6]  # always hold thre result
            self.offset_b = tiling_reg_list[reg_base[0] + 7]
            self.offset_t = tiling_reg_list[reg_base[0] + 8]
            self.ub_res_addr = tiling_reg_list[reg_base[0] + 9]
            self.ub_offset = tiling_reg_list[reg_base[0] + 10]

            self.ub_offset.set_as(0)
            self.offset_1.set_as(0)
            self.offset_2.set_as(_get_half_ub())
            self.ub_res_addr.set_as(self.offset_1)

    class TilingParamS5:
        """
        TilingParamS5
        """

        # 'pylint:disable=invalid-name
        def __init__(self, tiling_reg_list, ub_input_64_t, ub_input_64, tik_inst):
            """
            get tiling parameters
            """
            # part 2: fixed
            ub_offset = tik_inst.Scalar("int32", init_value=TILING_HEAD_LEN)

            reg_base = [41]
            for i in range(reg_base[0]):
                tiling_reg_list[i].set_as(ub_input_64_t[ub_offset + i])
            self.last_axis_len = tiling_reg_list[0]
            self.second_to_last_axis_len = tiling_reg_list[1]
            self.last_axis_burst_len = tiling_reg_list[2]
            self.align_ele = tiling_reg_list[3]
            self.logic_axis_num = tiling_reg_list[4]
            self.other_axis_num = tiling_reg_list[5]
            self.src_axis_num_no_dup = tiling_reg_list[6]
            self.dst_axis_num_no_dup = tiling_reg_list[7]
            self.major_burst_len_in = tiling_reg_list[8]
            self.tail_burst_len_in = tiling_reg_list[9]
            self.major_burst_len_out = tiling_reg_list[10]
            self.tail_burst_len_out = tiling_reg_list[11]
            self.major_dst_loop_in = tiling_reg_list[12]
            self.tail_dst_loop_in = tiling_reg_list[13]
            self.major_src_loop_out = tiling_reg_list[14]
            self.tail_src_loop_out = tiling_reg_list[15]
            self.major_in_ele = tiling_reg_list[16]
            self.tail_in_ele = tiling_reg_list[17]
            self.major_in_tail_ele = tiling_reg_list[18]
            self.tail_in_tail_ele = tiling_reg_list[19]
            self.major_out_ele = tiling_reg_list[20]
            self.tail_out_ele = tiling_reg_list[21]
            self.major_out_tail_ele = tiling_reg_list[22]
            self.tail_out_tail_ele = tiling_reg_list[23]
            self.dst_jump_major_step = tiling_reg_list[24]
            self.src_jump_major_step = tiling_reg_list[25]
            self.dup_axis = tiling_reg_list[26]
            self.src_axis_perm = tiling_reg_list[27]
            self.dst_axis_perm = tiling_reg_list[28]
            self.ub_axis_perm = tiling_reg_list[29]
            self.pivot_src_axis_dup = tiling_reg_list[30]
            self.pivot_dst_axis_dup = tiling_reg_list[31]
            self.last_two_loop = tiling_reg_list[32]
            self.last_two_repeat = tiling_reg_list[33]
            self.last_two_s_stride = tiling_reg_list[34]
            self.last_two_d_stride = tiling_reg_list[35]
            self.last_two_s_list_repeat = tiling_reg_list[36]
            self.last_two_d_list_repeat = tiling_reg_list[37]
            self.is_last_two_aligned_and_trans = tiling_reg_list[38]
            self.is_last_axis_transpose = tiling_reg_list[39]
            self.second_to_out_last_axis_len = tiling_reg_list[40]

            ub_offset.set_as(ub_offset + reg_base[0])

            self.n = []
            self.vol = []
            self.loop = []
            self.repeat = []
            self.burst_len = []
            self.src_stride = []
            self.dst_stride = []
            self.src_offset = []
            self.dst_offset = []

            self.n.append(tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION))
            self.n.append(tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION))
            self.n.append(tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION))

            self.vol.append(tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION))
            self.vol.append(tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION))
            self.vol.append(tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION))

            self.loop.append(tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION))
            self.loop.append(tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION))
            self.loop.append(tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION))

            self.repeat.append(tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION))
            self.repeat.append(tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION))
            self.repeat.append(tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION))

            self.burst_len.append(tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION))
            self.burst_len.append(tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION))
            self.burst_len.append(tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION))

            self.src_stride.append(tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION))
            self.src_stride.append(tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION))
            self.src_stride.append(tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION))

            self.dst_stride.append(tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION))
            self.dst_stride.append(tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION))
            self.dst_stride.append(tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION))

            self.src_offset.append(tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION))
            self.src_offset.append(tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION))
            self.src_offset.append(tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION))

            self.dst_offset.append(tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION))
            self.dst_offset.append(tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION))
            self.dst_offset.append(tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION))

            self.xdxsVol = tik_inst.ScalarArray("int64", length=UB_REORDER_COMBINATION)

            cycle = 28
            for i in range(UB_REORDER_COMBINATION):
                self.n[0][i].set_as(ub_input_64_t[ub_offset + i * cycle + 0])
                self.n[1][i].set_as(ub_input_64_t[ub_offset + i * cycle + 1])
                self.n[2][i].set_as(ub_input_64_t[ub_offset + i * cycle + 2])
                self.vol[0][i].set_as(ub_input_64_t[ub_offset + i * cycle + 3])
                self.vol[1][i].set_as(ub_input_64_t[ub_offset + i * cycle + 4])
                self.vol[2][i].set_as(ub_input_64_t[ub_offset + i * cycle + 5])
                self.loop[0][i].set_as(ub_input_64_t[ub_offset + i * cycle + 6])
                self.loop[1][i].set_as(ub_input_64_t[ub_offset + i * cycle + 7])
                self.loop[2][i].set_as(ub_input_64_t[ub_offset + i * cycle + 8])
                self.repeat[0][i].set_as(ub_input_64_t[ub_offset + i * cycle + 9])
                self.repeat[1][i].set_as(ub_input_64_t[ub_offset + i * cycle + 10])
                self.repeat[2][i].set_as(ub_input_64_t[ub_offset + i * cycle + 11])
                self.src_stride[0][i].set_as(ub_input_64_t[ub_offset + i * cycle + 12])
                self.src_stride[1][i].set_as(ub_input_64_t[ub_offset + i * cycle + 13])
                self.src_stride[2][i].set_as(ub_input_64_t[ub_offset + i * cycle + 14])
                self.dst_stride[0][i].set_as(ub_input_64_t[ub_offset + i * cycle + 15])
                self.dst_stride[1][i].set_as(ub_input_64_t[ub_offset + i * cycle + 16])
                self.dst_stride[2][i].set_as(ub_input_64_t[ub_offset + i * cycle + 17])
                self.burst_len[0][i].set_as(ub_input_64_t[ub_offset + i * cycle + 18])
                self.burst_len[1][i].set_as(ub_input_64_t[ub_offset + i * cycle + 19])
                self.burst_len[2][i].set_as(ub_input_64_t[ub_offset + i * cycle + 20])
                self.src_offset[0][i].set_as(ub_input_64_t[ub_offset + i * cycle + 21])
                self.src_offset[1][i].set_as(ub_input_64_t[ub_offset + i * cycle + 22])
                self.src_offset[2][i].set_as(ub_input_64_t[ub_offset + i * cycle + 23])
                self.dst_offset[0][i].set_as(ub_input_64_t[ub_offset + i * cycle + 24])
                self.dst_offset[1][i].set_as(ub_input_64_t[ub_offset + i * cycle + 25])
                self.dst_offset[2][i].set_as(ub_input_64_t[ub_offset + i * cycle + 26])
                self.xdxsVol[i].set_as(ub_input_64_t[ub_offset + i * cycle + 27])

            ub_offset.set_as(ub_offset + UB_REORDER_COMBINATION * cycle)

            self.dst_jump_factor_in = []
            self.src_jump_factor_out = []
            self.logic_jump_factor = []
            self.src_stride_out = []

            self.dst_stride_in = tik_inst.ScalarArray("int64", length=BORROW_DST_AXIS_LT_NUM)
            self.src_stride_out = tik_inst.ScalarArray("int64", length=BORROW_SRC_AXIS_LT_NUM)
            self.logic_stride_in = tik_inst.ScalarArray("int64", length=TRANSPOSE_MAX_AXIS_NUM)
            self.logic_stride_out = tik_inst.ScalarArray("int64", length=TRANSPOSE_MAX_AXIS_NUM)

            _set_param_python_arr(tiling_reg_list, reg_base, ub_input_64_t, ub_offset, self.dst_jump_factor_in,
                                  self.dst_axis_num_no_dup, BORROW_DST_AXIS_LT_NUM)
            _set_param_python_arr(tiling_reg_list, reg_base, ub_input_64_t, ub_offset, self.src_jump_factor_out,
                                  self.src_axis_num_no_dup, BORROW_SRC_AXIS_LT_NUM)
            _set_param_python_arr(tiling_reg_list, reg_base, ub_input_64_t, ub_offset, self.logic_jump_factor,
                                  self.logic_axis_num, TRANSPOSE_MAX_AXIS_NUM)
            _set_param_scalar_arr(tiling_reg_list, ub_input_64_t, ub_offset, self.dst_stride_in,
                                  self.dst_axis_num_no_dup, BORROW_DST_AXIS_LT_NUM)
            _set_param_scalar_arr(tiling_reg_list, ub_input_64_t, ub_offset, self.src_stride_out,
                                  self.src_axis_num_no_dup, BORROW_SRC_AXIS_LT_NUM)
            _set_param_scalar_arr(tiling_reg_list, ub_input_64_t, ub_offset, self.logic_stride_in,
                                  self.logic_axis_num, TRANSPOSE_MAX_AXIS_NUM)
            _set_param_scalar_arr(tiling_reg_list, ub_input_64_t, ub_offset, self.logic_stride_out,
                                  self.logic_axis_num, TRANSPOSE_MAX_AXIS_NUM)

            # part 3: percore
            tiling_reg_list[reg_base[0]].set_as(ub_input_64[0])
            self.loop_per_core = tiling_reg_list[reg_base[0]]
            reg_base[0] = reg_base[0] + 1
            ub_offset.set_as(1)

            self.init_src_tuple_out = []
            self.init_dst_tuple_in = []
            self.init_logic_tuple = []

            _set_param_python_arr(tiling_reg_list, reg_base, ub_input_64, ub_offset, self.init_src_tuple_out,
                                  self.src_axis_num_no_dup, BORROW_SRC_AXIS_LT_NUM)
            _set_param_python_arr(tiling_reg_list, reg_base, ub_input_64, ub_offset, self.init_dst_tuple_in,
                                  self.dst_axis_num_no_dup, BORROW_DST_AXIS_LT_NUM)
            _set_param_python_arr(tiling_reg_list, reg_base, ub_input_64, ub_offset, self.init_logic_tuple,
                                  self.other_axis_num + 2, TRANSPOSE_MAX_AXIS_NUM)

            # part 4: variable
            reg_base[0] = S4_FIXED_PART_SCALA_MAX_NUM + S4_PERCORE_PART_SCALA_MAX_NUM

            self.rt_dst_tuple_in = tik_inst.ScalarArray("int64", length=BORROW_DST_AXIS_LT_NUM)
            self.rt_src_tuple_out = tik_inst.ScalarArray("int64", length=BORROW_SRC_AXIS_LT_NUM)
            self.rt_logic_tuple = tik_inst.ScalarArray("int64", length=TRANSPOSE_MAX_AXIS_NUM)

            self.rt_src_tuple_logic = tiling_reg_list[reg_base[0] + 0]
            self.rt_dst_tuple_logic = tiling_reg_list[reg_base[0] + 1]
            self.src_addr = tiling_reg_list[reg_base[0] + 2]
            self.dst_addr = tiling_reg_list[reg_base[0] + 3]
            self.offset_1 = tiling_reg_list[reg_base[0] + 4]
            self.offset_2 = tiling_reg_list[reg_base[0] + 5]
            self.offset_a = tiling_reg_list[reg_base[0] + 6]  # always hold the result
            self.offset_b = tiling_reg_list[reg_base[0] + 7]
            self.offset_t = tiling_reg_list[reg_base[0] + 8]
            self.ub_res_addr = tiling_reg_list[reg_base[0] + 9]
            self.ub_offset = tiling_reg_list[reg_base[0] + 10]
            self.ub_src_offset = tiling_reg_list[reg_base[0] + 11]
            self.src_stride_reorder = tiling_reg_list[reg_base[0] + 12]
            self.dst_stride_reorder = tiling_reg_list[reg_base[0] + 13]

            self.ub_offset.set_as(0)
            self.offset_1.set_as(0)
            self.offset_2.set_as(_get_half_ub())
            self.ub_res_addr.set_as(self.offset_1)

            self.src_axis_perm_len = tik_inst.Scalar("uint8", "src_axis_perm_len")
            self.dst_axis_perm_len = tik_inst.Scalar("uint8", "dst_axis_perm_len")
            self.ub_axis_perm_len = tik_inst.Scalar("uint8", "ub_axis_perm_len")
            self.tail_src_axis_0 = tiling_reg_list[reg_base[0] + 16]
            self.in_ub_layout = tik_inst.ScalarArray(length=REORDER_MAX_AXIS_NUM, init_value=1)
            self.out_ub_layout = tik_inst.ScalarArray(length=REORDER_MAX_AXIS_NUM, init_value=1)
            self.in_axis_off = tik_inst.ScalarArray(length=REORDER_MAX_AXIS_NUM, init_value=0)
            self.scenario = None
            self.src_offset_base_addr = None
            self.src_offset_addr_gap = None

    # 'pylint: disable=unused-variable
    class TilingParamS7:
        """
        TilingParamS7
        """

        # 'pylint: disable=too-many-branches
        def __init__(self, tiling_reg_list, ub_input_64_t, ub_input_64, tik_inst):
            """
            get tiling parameters
            """
            # part 2: fixed
            ub_offset = tik_inst.Scalar("int32", init_value=TILING_HEAD_LEN)

            for i in range(5):
                tiling_reg_list[i].set_as(ub_input_64_t[ub_offset + i])

            self.core_num = tiling_reg_list[0]
            self.ub_size = tiling_reg_list[1]
            self.n_axis_num = tiling_reg_list[2]
            self.dst_axis_num = tiling_reg_list[3]
            self.src_axis_num = tiling_reg_list[4]

            self.n_jump_factor = []
            self.n_jump_stride_in = []
            self.n_jump_stride_out = []
            self.dst_jump_factor = []
            self.dst_jump_stride = []
            self.src_jump_factor = []
            self.src_jump_stride = []

            reg_base = 5
            ub_offset.set_as(ub_offset + reg_base)
            cycle = 7
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.n_jump_factor.append(tiling_reg_list[reg_base + i * cycle + 0])
                self.n_jump_stride_in.append(tiling_reg_list[reg_base + i * cycle + 1])
                self.n_jump_stride_out.append(tiling_reg_list[reg_base + i * cycle + 2])
                self.dst_jump_factor.append(tiling_reg_list[reg_base + i * cycle + 3])
                self.dst_jump_stride.append(tiling_reg_list[reg_base + i * cycle + 4])
                self.src_jump_factor.append(tiling_reg_list[reg_base + i * cycle + 5])
                self.src_jump_stride.append(tiling_reg_list[reg_base + i * cycle + 6])

            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.n_jump_factor[i].set_as(ub_input_64_t[ub_offset + i])

            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.n_jump_stride_in[i].set_as(ub_input_64_t[ub_offset + i + \
                                                              1 * self.n_axis_num])
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.n_jump_stride_out[i].set_as(ub_input_64_t[ub_offset + i + \
                                                               2 * self.n_axis_num])
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.dst_jump_factor[i].set_as(ub_input_64_t[ub_offset + i + \
                                                             3 * self.n_axis_num])
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.dst_jump_stride[i].set_as(ub_input_64_t[ub_offset + i + \
                                                             3 * self.n_axis_num + \
                                                             1 * self.dst_axis_num])
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.src_jump_factor[i].set_as(ub_input_64_t[ub_offset + i + \
                                                             3 * self.n_axis_num + \
                                                             2 * self.dst_axis_num])
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.src_jump_stride[i].set_as(ub_input_64_t[ub_offset + i + \
                                                             3 * self.n_axis_num + \
                                                             2 * self.dst_axis_num + \
                                                             1 * self.src_axis_num])

            # part 3: per core
            per_core_front = 11
            ub_offset.set_as(0)
            reg_base = S7_FIXED_PART_SCALA_MAX_NUM
            for i in range(per_core_front):
                tiling_reg_list[reg_base + i].set_as(ub_input_64[ub_offset + i])
            ub_offset.set_as(per_core_front)

            self.loop_on_n = tiling_reg_list[reg_base + 0]
            self.col_per_mc = tiling_reg_list[reg_base + 1]
            self.loop_on_mc = tiling_reg_list[reg_base + 2]
            self.col_tc = tiling_reg_list[reg_base + 3]
            self.col_offset = tiling_reg_list[reg_base + 4]
            self.back_step_left = tiling_reg_list[reg_base + 5]
            self.row_per_mr = tiling_reg_list[reg_base + 6]
            self.loop_on_mr = tiling_reg_list[reg_base + 7]
            self.row_tr = tiling_reg_list[reg_base + 8]
            self.row_offset = tiling_reg_list[reg_base + 9]
            self.back_step_up = tiling_reg_list[reg_base + 10]
            #if add line here, should change "per_core_front"

            self.init_n_tuple = []
            self.init_dst_tuple = []
            self.tail_dst_tuple = []
            self.init_src_tuple = []
            self.tail_src_tuple = []

            reg_base = S7_FIXED_PART_SCALA_MAX_NUM + per_core_front
            cycle = 5
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.init_n_tuple.append(tiling_reg_list[reg_base + i * cycle + 0])
                self.init_dst_tuple.append(tiling_reg_list[reg_base + i * cycle + 1])
                self.tail_dst_tuple.append(tiling_reg_list[reg_base + i * cycle + 2])
                self.init_src_tuple.append(tiling_reg_list[reg_base + i * cycle + 3])
                self.tail_src_tuple.append(tiling_reg_list[reg_base + i * cycle + 4])

            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.init_n_tuple[i].set_as(ub_input_64[ub_offset + 0 * self.n_axis_num + i])

            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.init_dst_tuple[i].set_as(ub_input_64[i + ub_offset + self.n_axis_num + 0 * self.dst_axis_num])

            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.tail_dst_tuple[i].set_as(ub_input_64[i + ub_offset + self.n_axis_num + 1 * self.dst_axis_num])

            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.init_src_tuple[i].set_as(ub_input_64[i + ub_offset + self.n_axis_num + \
                                                          2 * self.dst_axis_num + \
                                                          0 * self.src_axis_num])

            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.tail_src_tuple[i].set_as(ub_input_64[i + ub_offset + self.n_axis_num + \
                                                          2 * self.dst_axis_num + \
                                                          1 * self.src_axis_num])

            # part 4: variable
            self.rt_n_tuple = []
            self.rt_src_tuple = []
            self.rt_dst_tuple = []
            self.rt_dst_tuple_backup = []

            reg_base = S7_FIXED_PART_SCALA_MAX_NUM + S7_PERCORE_PART_SCALA_MAX_NUM
            cycle = 4
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.rt_n_tuple.append(tiling_reg_list[reg_base + i * cycle + 0])
                self.rt_dst_tuple.append(tiling_reg_list[reg_base + i * cycle + 1])
                self.rt_src_tuple.append(tiling_reg_list[reg_base + i * cycle + 2])
                self.rt_dst_tuple_backup.append(tiling_reg_list[reg_base + i * cycle + 3])
            reg_base = S7_FIXED_PART_SCALA_MAX_NUM + S7_PERCORE_PART_SCALA_MAX_NUM + cycle * TRANSPOSE_MAX_AXIS_NUM

            self.offset_a = tiling_reg_list[reg_base + 0]
            self.offset_b = tiling_reg_list[reg_base + 1]
            self.offset_t = tiling_reg_list[reg_base + 2]
            self.src_stride_reorder = tiling_reg_list[reg_base + 3]
            self.dst_stride_reorder = tiling_reg_list[reg_base + 4]
            self.col_reorder = tiling_reg_list[reg_base + 5]
            self.row_reorder = tiling_reg_list[reg_base + 6]
            self.rt_dst_addr = tiling_reg_list[reg_base + 7]
            self.src_addr = tiling_reg_list[reg_base + 8]
            self.dst_addr = tiling_reg_list[reg_base + 9]
            self.n_src_offset = tiling_reg_list[reg_base + 10]
            self.n_dst_offset = tiling_reg_list[reg_base + 11]
            self.dst_vnc_pad = tiling_reg_list[reg_base + 12]
            self.offset_a.set_as(0)
            self.offset_b.set_as(_get_half_ub())
            self.dst_vnc_pad.set_as(ROW_UNIT)
            self.scenario = None
            self.src_offset_base_addr = None
            self.src_offset_addr_gap = None

    class TilingParamS13:
        """
        TilingParamS13: customized shapeinfo defined in InitSpecificShape func of tiling file
        """

        def __init__(self, tiling_reg_list, ub_input_64_t, block_idx, tik_inst):
            """
            get tiling parameters
            """
            for i in range(28):
                tiling_reg_list[i].set_as(ub_input_64_t[TILING_HEAD_LEN + i])
            self.n_num = tiling_reg_list[0]
            self.row_num = tiling_reg_list[1]
            self.col_num = tiling_reg_list[2]
            self.use_twice_vnc = tiling_reg_list[3]
            self.n_major_loop_core_num = tiling_reg_list[4]
            self.n_last_loop_core_num = tiling_reg_list[5]
            self.n_major_core_num = tiling_reg_list[6]
            self.n_last_core_num = tiling_reg_list[7]
            self.split_n = tiling_reg_list[8]

            self.tiling_used_core_num = []
            self.tiling_split_by_row = []
            self.tiling_major_core_loop_cnt = []
            self.tiling_last_core_loop_cnt = []
            self.tiling_loop_unit = []
            self.tiling_last_core_last_loop_unit = []
            self.tiling_inner_loop_unit = []
            self.tiling_inner_loop_cnt = []
            self.tiling_inner_last_loop_unit = []
            for i in range(2):
                self.tiling_used_core_num.append(tiling_reg_list[9 + i * 9])
                self.tiling_split_by_row.append(tiling_reg_list[10 + i * 9])
                self.tiling_major_core_loop_cnt.append(tiling_reg_list[11 + i * 9])
                self.tiling_last_core_loop_cnt.append(tiling_reg_list[12 + i * 9])
                self.tiling_loop_unit.append(tiling_reg_list[13 + i * 9])
                self.tiling_last_core_last_loop_unit.append(tiling_reg_list[14 + i * 9])
                self.tiling_inner_loop_unit.append(tiling_reg_list[15 + i * 9])
                self.tiling_inner_loop_cnt.append(tiling_reg_list[16 + i * 9])
                self.tiling_inner_last_loop_unit.append(tiling_reg_list[17 + i * 9])

            self._init_other_param(tiling_reg_list)
            with tik_inst.if_scope(self.split_n == 0):
                self._init_not_split_by_n(block_idx, tik_inst)
            with tik_inst.else_scope():
                self._init_split_by_n(block_idx, tik_inst)

        def _init_not_split_by_n(self, block_idx, tik_inst):
            with tik_inst.if_scope(block_idx < self.n_major_loop_core_num * self.n_major_core_num):
                self.n_offset.set_as(block_idx / self.n_major_loop_core_num)
                self.inner_core_offset.set_as(block_idx % self.n_major_loop_core_num)
                self._init_loop_param(0)
            with tik_inst.else_scope():
                self.n_offset.set_as((block_idx - self.n_major_loop_core_num * self.n_major_core_num) /
                                        self.n_last_loop_core_num + self.n_major_core_num)
                self.inner_core_offset.set_as((block_idx - self.n_major_loop_core_num * self.n_major_core_num) %
                                                self.n_last_loop_core_num)
                self._init_loop_param(1)

            self.n_loop_num.set_as(1)
            with tik_inst.if_scope(self.split_by_row == 1):
                core_offset = self.n_offset * self.col_num * self.row_num
                inner_offset = self.inner_core_offset * self.major_core_loop_cnt * self.loop_unit
                self.data_out_offset.set_as(core_offset + inner_offset)
                self.data_in_offset.set_as(core_offset + inner_offset * self.col_num)
            with tik_inst.else_scope():
                core_offset = self.n_offset * self.col_num * self.row_num
                inner_offset = self.inner_core_offset * self.major_core_loop_cnt * self.loop_unit
                self.data_in_offset.set_as(core_offset + inner_offset)
                self.data_out_offset.set_as(core_offset + inner_offset * self.row_num)

            with tik_inst.if_scope(self.inner_core_offset + 1 == self.used_core_num):
                self.major_loop_cnt.set_as(self.last_core_loop_cnt)
                self.last_loop_unit.set_as(self.last_core_last_loop_unit)
            with tik_inst.elif_scope(self.inner_core_offset < self.used_core_num):
                self.major_loop_cnt.set_as(self.major_core_loop_cnt)
                self.last_loop_unit.set_as(0)
            with tik_inst.else_scope():
                self.major_loop_cnt.set_as(0)
                self.last_loop_unit.set_as(0)

        def _init_split_by_n(self, block_idx, tik_inst):
            with tik_inst.if_scope(block_idx < self.n_major_core_num):
                self.n_offset.set_as(block_idx * self.n_major_loop_core_num)
                self._init_loop_param(0)
                self.n_loop_num.set_as(self.n_major_loop_core_num)
            with tik_inst.else_scope():
                self.n_offset.set_as(self.n_major_loop_core_num * self.n_major_core_num +
                                     (block_idx - self.n_major_core_num) * self.n_last_loop_core_num)
                self._init_loop_param(1)
                self.n_loop_num.set_as(self.n_last_loop_core_num)

            core_offset = self.n_offset * self.row_num * self.col_num
            self.data_out_offset.set_as(core_offset)
            self.data_in_offset.set_as(core_offset)
            self.major_loop_cnt.set_as(self.last_core_loop_cnt)
            self.last_loop_unit.set_as(self.last_core_last_loop_unit)

        def _init_loop_param(self, index):
            self.used_core_num.set_as(self.tiling_used_core_num[index])
            self.split_by_row.set_as(self.tiling_split_by_row[index])
            self.major_core_loop_cnt.set_as(self.tiling_major_core_loop_cnt[index])
            self.last_core_loop_cnt.set_as(self.tiling_last_core_loop_cnt[index])
            self.loop_unit.set_as(self.tiling_loop_unit[index])
            self.last_core_last_loop_unit.set_as(self.tiling_last_core_last_loop_unit[index])
            self.inner_loop_unit.set_as(self.tiling_inner_loop_unit[index])
            self.inner_loop_cnt.set_as(self.tiling_inner_loop_cnt[index])
            self.inner_last_loop_unit.set_as(self.tiling_inner_last_loop_unit[index])

        def _init_other_param(self, tiling_reg_list):
            offset_base = 100
            self.n_offset = tiling_reg_list[offset_base]
            self.inner_core_offset = tiling_reg_list[offset_base + 1]
            self.n_loop_num = tiling_reg_list[offset_base + 2]
            self.data_in_offset = tiling_reg_list[offset_base + 3]
            self.data_out_offset = tiling_reg_list[offset_base + 4]
            self.major_loop_cnt = tiling_reg_list[offset_base + 5]
            self.last_loop_unit = tiling_reg_list[offset_base + 6]
            self.nburst_in_ub = tiling_reg_list[offset_base + 7]
            self.src_ub_gap = tiling_reg_list[offset_base + 8]
            self.dst_ub_gap = tiling_reg_list[offset_base + 9]

            self.used_core_num = tiling_reg_list[offset_base + 10]
            self.split_by_row = tiling_reg_list[offset_base + 11]
            self.major_core_loop_cnt = tiling_reg_list[offset_base + 12]
            self.last_core_loop_cnt = tiling_reg_list[offset_base + 13]
            self.loop_unit = tiling_reg_list[offset_base + 14]
            self.last_core_last_loop_unit = tiling_reg_list[offset_base + 15]
            self.inner_loop_unit = tiling_reg_list[offset_base + 16]
            self.inner_loop_cnt = tiling_reg_list[offset_base + 17]
            self.inner_last_loop_unit = tiling_reg_list[offset_base + 18]

            self.vg_stride = tiling_reg_list[offset_base + 19]
            self.back_step = tiling_reg_list[offset_base + 20]
            self.vncov_step = tiling_reg_list[offset_base + 21]
            self.reorder_src_gap = tiling_reg_list[offset_base + 22]
            self.reorder_dst_gap = tiling_reg_list[offset_base + 23]

    class TilingParamS9:
        """
        TilingParamS9
        """

        def __init__(self, tiling_reg_list, ub_input_64_t, ub_input_64, tik_inst):
            """
            get tiling parameters
            """
            # part 2: fixed
            ub_offset = tik_inst.Scalar("int32", init_value=TILING_HEAD_LEN)

            for i in range(8):
                tiling_reg_list[i].set_as(ub_input_64_t[ub_offset + i])
            self.core_num = tiling_reg_list[0]
            self.ub_size = tiling_reg_list[1]
            self.last_axis_len = tiling_reg_list[2]
            self.last_axis_burst_len = tiling_reg_list[3]
            self.trans_axis_num = tiling_reg_list[4]
            self.repeat = tiling_reg_list[5]
            self.src_repeat_stride = tiling_reg_list[6]
            self.dst_repeat_stride = tiling_reg_list[7]

            reg_base = 8
            ub_offset.set_as(TILING_HEAD_LEN + reg_base)
            cycle = 4
            self.src_jump_stride = []
            self.dst_jump_stride = []
            self.src_jump_factor = []
            self.dst_jump_factor = []
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.src_jump_stride.append(tiling_reg_list[reg_base + i * cycle + 0])
                self.dst_jump_stride.append(tiling_reg_list[reg_base + i * cycle + 1])
                self.src_jump_factor.append(tiling_reg_list[reg_base + i * cycle + 2])
                self.dst_jump_factor.append(tiling_reg_list[reg_base + i * cycle + 3])

            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.src_jump_stride[i].set_as(ub_input_64_t[ub_offset + i + 0 * self.trans_axis_num])
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.dst_jump_stride[i].set_as(ub_input_64_t[ub_offset + i + 1 * self.trans_axis_num])
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.src_jump_factor[i].set_as(ub_input_64_t[ub_offset + i + 2 * self.trans_axis_num])
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.dst_jump_factor[i].set_as(ub_input_64_t[ub_offset + i + 3 * self.trans_axis_num])

            # part 3 : percore
            ub_offset.set_as(0)
            reg_base = S9_FIXED_PART_SCALA_MAX_NUM
            self.loop_num = tiling_reg_list[reg_base]
            self.loop_num.set_as(ub_input_64[ub_offset])

            ub_offset.set_as(1)
            reg_base = reg_base + 1
            self.init_tuple = []
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.init_tuple.append(tiling_reg_list[reg_base + i])
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.init_tuple[i].set_as(ub_input_64[ub_offset + i])

            # part 4: variable
            reg_base = S9_FIXED_PART_SCALA_MAX_NUM + S9_PERCORE_PART_SCALA_MAX_NUM
            cycle = 1
            self.rt_tuple = []
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.rt_tuple.append(tiling_reg_list[reg_base + i * cycle + 0])

            reg_base = S1_FIXED_PART_SCALA_MAX_NUM + S1_PERCORE_PART_SCALA_MAX_NUM + cycle * TRANSPOSE_MAX_AXIS_NUM
            self.src_addr = tiling_reg_list[reg_base]
            self.dst_addr = tiling_reg_list[reg_base + 1]

    class TilingParamS11:
        """
        TilingParamS11
        """

        def __init__(self, tiling_reg_list, ub_input_64_t, ub_input_64, tik_inst):
            """
            get tiling parameters
            """

            # part 2: fixed
            ub_offset = tik_inst.Scalar("int32", init_value=TILING_HEAD_LEN)

            reg_base = 16
            for i in range(reg_base):
                tiling_reg_list[i].set_as(ub_input_64_t[ub_offset + i])

            self.core_num = tiling_reg_list[0]
            self.ub_size = tiling_reg_list[1]
            self.n_axis_num = tiling_reg_list[2]
            self.col_per_mc = tiling_reg_list[3]
            self.col_block_per_mc = tiling_reg_list[4]
            self.col_block_tc = tiling_reg_list[5]
            self.row_per_mr = tiling_reg_list[6]
            self.row_block_per_mr = tiling_reg_list[7]
            self.row_block_tr = tiling_reg_list[8]
            self.src_stride_in = tiling_reg_list[9]
            self.src_stride_in_tail = tiling_reg_list[10]
            self.dst_stride_out = tiling_reg_list[11]
            self.dst_stride_out_tail = tiling_reg_list[12]
            self.n_unit = tiling_reg_list[13]
            self.col_vol = tiling_reg_list[14]
            self.row_vol = tiling_reg_list[15]

            ub_offset.set_as(ub_offset + reg_base)

            self.n_jump_factor = []
            self.n_src_jump_stride = []
            self.n_dst_jump_stride = []
            cycle = 3
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.n_jump_factor.append(tiling_reg_list[reg_base + i * cycle + 0])
                self.n_src_jump_stride.append(tiling_reg_list[reg_base + i * cycle + 1])
                self.n_dst_jump_stride.append(tiling_reg_list[reg_base + i * cycle + 2])

            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.n_jump_factor[i].set_as(ub_input_64_t[ub_offset + i + 0 * self.n_axis_num])

            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.n_src_jump_stride[i].set_as(ub_input_64_t[ub_offset + i + 1 * self.n_axis_num])

            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.n_dst_jump_stride[i].set_as(ub_input_64_t[ub_offset + i + 2 * self.n_axis_num])

            # part 3: per core
            per_core_front = 9
            ub_offset.set_as(0)
            reg_base = S10_FIXED_PART_SCALA_MAX_NUM
            for i in range(per_core_front):
                tiling_reg_list[reg_base + i].set_as(ub_input_64[ub_offset + i])
            ub_offset.set_as(per_core_front)

            self.loop_on_n = tiling_reg_list[reg_base + 0]
            self.init_n = tiling_reg_list[reg_base + 1]
            self.loop_on_mc = tiling_reg_list[reg_base + 2]
            self.col_tc = tiling_reg_list[reg_base + 3]
            self.col_offset = tiling_reg_list[reg_base + 4]
            self.loop_on_mr = tiling_reg_list[reg_base + 5]
            self.row_tr = tiling_reg_list[reg_base + 6]
            self.row_offset = tiling_reg_list[reg_base + 7]
            self.back_step_up = tiling_reg_list[reg_base + 8]

            self.init_n_tuple = []

            reg_base = S10_FIXED_PART_SCALA_MAX_NUM + per_core_front
            cycle = 1
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.init_n_tuple.append(tiling_reg_list[reg_base + i * cycle + 0])

            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.init_n_tuple[i].set_as(ub_input_64[ub_offset + i])
            ub_offset.set_as(ub_offset + self.n_axis_num)

            # part 4: variable
            reg_base = S10_FIXED_PART_SCALA_MAX_NUM + S10_PERCORE_PART_SCALA_MAX_NUM

            self.rt_n_tuple = []
            cycle = 1
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.rt_n_tuple.append(tiling_reg_list[reg_base + i * cycle + 0])

            reg_base = S10_FIXED_PART_SCALA_MAX_NUM + S10_PERCORE_PART_SCALA_MAX_NUM + cycle * TRANSPOSE_MAX_AXIS_NUM
            self.src_addr = tiling_reg_list[reg_base + 0]
            self.dst_addr = tiling_reg_list[reg_base + 1]
            self.offset_a = tiling_reg_list[reg_base + 2]
            self.offset_b = tiling_reg_list[reg_base + 3]
            self.col_reorder = tiling_reg_list[reg_base + 4]
            self.row_reorder = tiling_reg_list[reg_base + 5]
            self.rt_dst_addr = tiling_reg_list[reg_base + 6]
            self.src_stride_reorder = tiling_reg_list[reg_base + 7]
            self.dst_stride_reorder = tiling_reg_list[reg_base + 8]
            self.dst_vnc_pad = tiling_reg_list[reg_base + 9]
            self.offset_a.set_as(0)
            self.offset_b.set_as(_get_half_ub())
            self.dst_vnc_pad.set_as(ROW_UNIT)

    class TilingParamS12:
        """
        TilingParamS12
        """

        def __init__(self, tiling_reg_list, ub_input_64_t, ub_input_64, tik_inst):
            """
            get tiling parameters
            """
            # part 2: fixed
            ub_offset = tik_inst.Scalar("int32", init_value=TILING_HEAD_LEN)

            for i in range(12):
                tiling_reg_list[i].set_as(ub_input_64_t[ub_offset + i])
            self.core_num = tiling_reg_list[0]
            self.ub_size = tiling_reg_list[1]
            self.last_axis_len = tiling_reg_list[2]
            self.last_axis_burst_len = tiling_reg_list[3]
            self.trans_axis_num = tiling_reg_list[4]
            self.loop_unit = tiling_reg_list[5]
            self.loop_main = tiling_reg_list[6]
            self.loop_tail = tiling_reg_list[7]
            self.loop_src_stride = tiling_reg_list[8]
            self.loop_dst_stride = tiling_reg_list[9]
            self.src_repeat_stride = tiling_reg_list[10]
            self.dst_repeat_stride = tiling_reg_list[11]

            reg_base = 12
            ub_offset.set_as(TILING_HEAD_LEN + reg_base)
            cycle = 4
            self.src_jump_stride = []
            self.dst_jump_stride = []
            self.src_jump_factor = []
            self.dst_jump_factor = []
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.src_jump_stride.append(tiling_reg_list[reg_base + i * cycle + 0])
                self.dst_jump_stride.append(tiling_reg_list[reg_base + i * cycle + 1])
                self.src_jump_factor.append(tiling_reg_list[reg_base + i * cycle + 2])
                self.dst_jump_factor.append(tiling_reg_list[reg_base + i * cycle + 3])

            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.src_jump_stride[i].set_as(ub_input_64_t[ub_offset + i + 0 * self.trans_axis_num])
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.dst_jump_stride[i].set_as(ub_input_64_t[ub_offset + i + 1 * self.trans_axis_num])
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.src_jump_factor[i].set_as(ub_input_64_t[ub_offset + i + 2 * self.trans_axis_num])
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.dst_jump_factor[i].set_as(ub_input_64_t[ub_offset + i + 3 * self.trans_axis_num])

            # part 3 : percore
            ub_offset.set_as(0)
            reg_base = S9_FIXED_PART_SCALA_MAX_NUM
            self.loop_num = tiling_reg_list[reg_base]
            self.loop_num.set_as(ub_input_64[ub_offset])

            ub_offset.set_as(1)
            reg_base = reg_base + 1
            self.init_tuple = []
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.init_tuple.append(tiling_reg_list[reg_base + i])
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.init_tuple[i].set_as(ub_input_64[ub_offset + i])

            # part 4: variable
            reg_base = S9_FIXED_PART_SCALA_MAX_NUM + S9_PERCORE_PART_SCALA_MAX_NUM
            cycle = 1
            self.rt_tuple = []
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.rt_tuple.append(tiling_reg_list[reg_base + i * cycle + 0])

            reg_base = S1_FIXED_PART_SCALA_MAX_NUM + S1_PERCORE_PART_SCALA_MAX_NUM + cycle * TRANSPOSE_MAX_AXIS_NUM
            self.src_addr = tiling_reg_list[reg_base]
            self.dst_addr = tiling_reg_list[reg_base + 1]

    def _init_mem_make_ub_allocated(self, ub_input):
        ub_input_b16 = ub_input.reinterpret_cast_to("int16")
        self.tik_inst.vector_dup(128, ub_input_b16, 0, 1, 1, 0)

    def __init__(self, tik_inst, x_dtype, tensor_list, kernel_name):
        self.tik_inst = tik_inst
        self.x_dtype = x_dtype
        self.kernel_name = kernel_name
        self.data_in, self.data_perm, self.data_out, self.data_workspace, self.data_tiling = tensor_list
        self.ub_size = self._get_ub_size_by_dtype()
        self.ub_size_64 = self._get_ub_size_by_int64()
        self.ub_input_64_t = None
        self.ub_input_b16_vor = None
        self.ub_input_b64_helper = None
        self.ub_input_64 = None
        self.tiling_reg_list = [self.tik_inst.Scalar("int64") for i in range(TILING_MAX_PARAM_NUM)]
        self.element_per_block = self._element_per_block(self.x_dtype)
        self.fp16_times = (self._sizeof_dtype(x_dtype) + 1) // self._sizeof_dtype("float16") # fp32/int32:2 fp16/int16:1
        self.b8_times = self._sizeof_dtype(x_dtype)
        self.ele_per_block = BLOCK_SIZE // self._sizeof_dtype(x_dtype)
        self.is_mte_pad = tbe_platform.api_check_support("tik.data_move_pad")
        self.vg_lp_unit = VG_MASK // self.b8_times
        self.vg_base_addr = self.tik_inst.Scalar(dtype="uint32", init_value=0)
        # vgather 310B precision error
        self.tail_repeat = self.tik_inst.Scalar("int64", name="tail_repeat", init_value=0)
        self.head_repeat = self.tik_inst.Scalar("int64", name="head_repeat", init_value=0)
        self.mid_repeats = self.tik_inst.Scalar("int64", name="mid_repeats", init_value=0)
        # while repeat greater than max repeat limit split it into loops 
        self.repeat_loops = self.tik_inst.Scalar("int64", name="repeats_loops", init_value=0)
        self.repeat_tail = self.tik_inst.Scalar("int64", name="repeats_tail", init_value=0)
        self.m_data_in = self.data_in
        self.m_data_out = self.data_out
        self.m_ub_input = None
        self.m_offset_factor = 1
        self.vg_cond = tbe_platform.api_check_support("tik.vgather", self.x_dtype) & (is_v200() | is_soc_310p())
        self.vnc_split = self.ele_per_block
        self.half_ub_num = _get_half_ub() // self.fp16_times

    @staticmethod
    def _sizeof_dtype(dtype):
        if dtype in ("int8", "uint8", "bool"):
            return 1
        if dtype in ("float16", "int16", "uint16", "bfloat16"):
            return 2
        if dtype in ("float", "float32", "int32", "uint32"):
            return 4
        if dtype in ("int64", "uint64", "double"):
            return 8
        return 8

    @staticmethod
    def _element_per_block(dtype):
        if dtype in ("int8", "uint8", "bool"):
            return 32
        if dtype in ("float16", "int16", "uint16", "bfloat16"):
            return 16
        if dtype in ("float", "float32", "int32", "uint32"):
            return 8
        if dtype in ("int64", "uint64", "double"):
            return 4
        return 4

    def _get_ub_size_by_dtype(self):
        ub_size_not_int64 = (get_ub_size() - RESERVED_UB * 2048) // self._sizeof_dtype(self.x_dtype)
        return ub_size_not_int64

    def _get_ub_size_by_int64(self):
        ub_size_int64 = (get_ub_size() - RESERVED_UB * 1024) // self._sizeof_dtype("int64")
        return ub_size_int64

    @staticmethod
    def _get_src_size():
        if get_ub_size() == 256 * 1024:
            return 3968 - 16  # 910
        if get_ub_size() == 248 * 1024:
            return 3968 - 16  # 310
        if get_ub_size() == 192 * 1024:
            return 2848  # cs & a100
        if get_ub_size() == 128 * 1024:
            return 1861  # es
        return 3968 - 16

    @staticmethod
    def _get_dst_size():
        if get_ub_size() == 256 * 1024:
            return 247  # 910, 247 to avoid bank conflict
        if get_ub_size() == 248 * 1024:
            lite_core_tup = (1, 3, 4)
            if get_core_num() in lite_core_tup:
                return 243  # 310B, 243 to avoid ub conflict
            return 245  # 310, 245 to avoid bank conflict
        if get_ub_size() == 192 * 1024:
            return 187  # cs & a100
        if get_ub_size() == 128 * 1024:
            return 123  # es
        return 247

    def _ele_per_block(self):
        return BLOCK_SIZE // self._sizeof_dtype(self.x_dtype)

    def _do_data_move_no_stride(self, dst, src, nburst, burst_len, actual_len):
        if self.is_mte_pad:
            if self.b8_times == EPB8:
                dst = dst.reinterpret_cast_to("int8")
                src = src.reinterpret_cast_to("int8")
            self.tik_inst.data_move_pad(dst, src, nburst, actual_len * self.b8_times, 0, 0)
        else:
            self.tik_inst.data_move(dst, src, 0, nburst, burst_len, 0, 0)

    def _do_data_move_in_with_stride(self, dst, src, nburst, burst_len, actual_len, dst_stride, src_stride):
        if self.is_mte_pad:
            if self.b8_times == EPB8:
                dst = dst.reinterpret_cast_to("int8")
                src = src.reinterpret_cast_to("int8")
            self.tik_inst.data_move_pad(dst, src, nburst, actual_len * self.b8_times,
                                        dst_stride,
                                        src_stride * BLOCK_SIZE + burst_len * BLOCK_SIZE - actual_len * self.b8_times)
        else:
            self.tik_inst.data_move(dst, src, 0, nburst, burst_len, src_stride, dst_stride)

    # -------------------------------------------------------------------------------------------------
    #                                    scenario_0
    # -------------------------------------------------------------------------------------------------
    # 'pylint:disable=invalid-name
    def _major_loop_move(self, tp, ub_input, loop_num, major_loop_new):
        with self.tik_inst.for_range(0, loop_num) as i:
            self.tik_inst.data_move_pad(ub_input, self.data_in[tp.base + \
                                        i * major_loop_new * tp.major_num * self.ele_per_block], 
                                        major_loop_new, tp.major_num * BLOCK_SIZE, 0, 0)
            self.tik_inst.data_move_pad(self.data_out[tp.base + \
                                        i * major_loop_new * tp.major_num * self.ele_per_block], ub_input, 
                                        major_loop_new, tp.major_num * BLOCK_SIZE, 0, 0)

    # 'pylint:disable=invalid-name
    def _major_loop_tail_move(self, tp, ub_input, loop_num, major_loop_new):
        with self.tik_inst.if_scope(tp.major_loop - major_loop_new * loop_num != 0):
            major_loop_tail = tp.major_loop - major_loop_new * loop_num
            self.tik_inst.data_move_pad(ub_input, self.data_in[tp.base + \
                                        loop_num * major_loop_new * tp.major_num * self.ele_per_block], 
                                        major_loop_tail, tp.major_num * BLOCK_SIZE, 0, 0)
            self.tik_inst.data_move_pad(self.data_out[tp.base + \
                                        loop_num * major_loop_new * tp.major_num * self.ele_per_block], ub_input,
                                        major_loop_tail, tp.major_num * BLOCK_SIZE, 0, 0)
    
    # 'pylint:disable=invalid-name
    def _move_data_s0(self, tp, ub_input_64):
        ub_input = ub_input_64.reinterpret_cast_to(self.x_dtype)
        if self.is_mte_pad and self.b8_times != EPB8:
            with self.tik_inst.if_scope(tp.major_loop != 0):
                available_ub_size = get_ub_size() - RESERVED_UB * 1024
                loop_num = (tp.major_loop * tp.major_num * BLOCK_SIZE + available_ub_size - 1) // available_ub_size
                major_loop_new = tp.major_loop // loop_num
                loop_num = tp.major_loop // major_loop_new
                self._major_loop_move(tp, ub_input, loop_num, major_loop_new)
                self._major_loop_tail_move(tp, ub_input, loop_num, major_loop_new)
        else:
            with self.tik_inst.for_range(0, tp.major_loop) as i:
                self.tik_inst.data_move(ub_input, self.data_in[tp.base + i * tp.major_num * self.ele_per_block], 0, 1,
                                        tp.major_num, 0, 0)
                self.tik_inst.data_move(self.data_out[tp.base + i * tp.major_num * self.ele_per_block], ub_input, 0, 1,
                                        tp.major_num, 0, 0)

        # use data_move_pad when ele not align
        if self.is_mte_pad:
            with self.tik_inst.if_scope(tik.any(tp.tail_num != 0, tp.not_align_ele != 0)):
                self.m_ub_input = ub_input
                if self.b8_times == EPB8:
                    self.m_ub_input = ub_input.reinterpret_cast_to("int8")
                    self.m_data_in = self.data_in.reinterpret_cast_to("int8")
                    self.m_data_out = self.data_out.reinterpret_cast_to("int8")
                    self.m_offset_factor = EPB8
                move_offset = (tp.base + tp.major_loop * tp.major_num * self.ele_per_block) * self.m_offset_factor
                move_bytes = tp.tail_num * BLOCK_SIZE + tp.not_align_ele * self.b8_times
                self.tik_inst.data_move_pad(self.m_ub_input, self.m_data_in[move_offset], 1, move_bytes, 0, 0)
                self.tik_inst.data_move_pad(self.m_data_out[move_offset], self.m_ub_input, 1, move_bytes, 0, 0)
        else:
            with self.tik_inst.if_scope(tp.tail_num != 0):
                self.tik_inst.data_move(ub_input,
                                        self.data_in[tp.base + tp.major_loop * tp.major_num * self.ele_per_block],
                                        0, 1, tp.tail_num, 0, 0)
                self.tik_inst.data_move(self.data_out[tp.base + tp.major_loop * tp.major_num * self.ele_per_block],
                                        ub_input, 0, 1, tp.tail_num, 0, 0)
            with self.tik_inst.if_scope(tp.not_align_ele != 0):
                with self.tik_inst.if_scope(tik.all(tp.major_loop == 0, tp.tail_num == 0)):
                    self.tik_inst.data_move(ub_input, self.data_in[tp.base], 0, 1, 1, 0, 0)
                    self.tik_inst.data_move(self.data_out[tp.base], ub_input, 0, 1, 1, 0, 0)
                with self.tik_inst.else_scope():
                    self.tik_inst.data_move(ub_input, self.data_in[tp.base + tp.ele_num - self.ele_per_block],
                                            0, 1, 1, 0, 0)
                    self.tik_inst.data_move(self.data_out[tp.base + tp.ele_num - self.ele_per_block], ub_input,
                                            0, 1, 1, 0, 0)

    # -------------------------------------------------------------------------------------------------
    #                                    scenario_1
    # -------------------------------------------------------------------------------------------------
    # 'pylint:disable=invalid-name
    def _get_src_addr_s1(self, tp):
        with self.tik_inst.if_scope(tp.trans_axis_num == 1):
            tp.src_addr.set_as(tp.rt_tuple[0] * tp.src_jump_stride[0])
        with self.tik_inst.else_scope():
            with self.tik_inst.if_scope(tp.trans_axis_num == 2):
                tp.src_addr.set_as(tp.rt_tuple[0] * tp.src_jump_stride[0] + \
                                   tp.rt_tuple[1] * tp.src_jump_stride[1])
            with self.tik_inst.else_scope():
                with self.tik_inst.if_scope(tp.trans_axis_num == 3):
                    tp.src_addr.set_as(tp.rt_tuple[0] * tp.src_jump_stride[0] + \
                                       tp.rt_tuple[1] * tp.src_jump_stride[1] + \
                                       tp.rt_tuple[2] * tp.src_jump_stride[2])
                with self.tik_inst.else_scope():
                    with self.tik_inst.if_scope(tp.trans_axis_num == 4):
                        tp.src_addr.set_as(tp.rt_tuple[0] * tp.src_jump_stride[0] + \
                                           tp.rt_tuple[1] * tp.src_jump_stride[1] + \
                                           tp.rt_tuple[2] * tp.src_jump_stride[2] + \
                                           tp.rt_tuple[3] * tp.src_jump_stride[3])
                    with self.tik_inst.else_scope():
                        with self.tik_inst.if_scope(tp.trans_axis_num == 5):
                            tp.src_addr.set_as(tp.rt_tuple[0] * tp.src_jump_stride[0] + \
                                               tp.rt_tuple[1] * tp.src_jump_stride[1] + \
                                               tp.rt_tuple[2] * tp.src_jump_stride[2] + \
                                               tp.rt_tuple[3] * tp.src_jump_stride[3] + \
                                               tp.rt_tuple[4] * tp.src_jump_stride[4])

                        with self.tik_inst.if_scope(tp.trans_axis_num == 6):
                            tp.src_addr.set_as(tp.rt_tuple[0] * tp.src_jump_stride[0] + \
                                               tp.rt_tuple[1] * tp.src_jump_stride[1] + \
                                               tp.rt_tuple[2] * tp.src_jump_stride[2] + \
                                               tp.rt_tuple[3] * tp.src_jump_stride[3] + \
                                               tp.rt_tuple[4] * tp.src_jump_stride[4] + \
                                               tp.rt_tuple[5] * tp.src_jump_stride[5])
                        
                        with self.tik_inst.if_scope(tp.trans_axis_num == 7):
                            tp.src_addr.set_as(tp.rt_tuple[0] * tp.src_jump_stride[0] + \
                                               tp.rt_tuple[1] * tp.src_jump_stride[1] + \
                                               tp.rt_tuple[2] * tp.src_jump_stride[2] + \
                                               tp.rt_tuple[3] * tp.src_jump_stride[3] + \
                                               tp.rt_tuple[4] * tp.src_jump_stride[4] + \
                                               tp.rt_tuple[5] * tp.src_jump_stride[5] + \
                                               tp.rt_tuple[6] * tp.src_jump_stride[6])
                        
                        with self.tik_inst.if_scope(tp.trans_axis_num == 8):
                            tp.src_addr.set_as(tp.rt_tuple[0] * tp.src_jump_stride[0] + \
                                               tp.rt_tuple[1] * tp.src_jump_stride[1] + \
                                               tp.rt_tuple[2] * tp.src_jump_stride[2] + \
                                               tp.rt_tuple[3] * tp.src_jump_stride[3] + \
                                               tp.rt_tuple[4] * tp.src_jump_stride[4] + \
                                               tp.rt_tuple[5] * tp.src_jump_stride[5] + \
                                               tp.rt_tuple[6] * tp.src_jump_stride[6] + \
                                               tp.rt_tuple[7] * tp.src_jump_stride[7])

    # 'pylint:disable=invalid-name
    def _get_dst_addr_s1(self, tp):
        with self.tik_inst.if_scope(tp.trans_axis_num == 1):
            tp.dst_addr.set_as(tp.rt_tuple[0] * tp.dst_jump_stride[0])
        with self.tik_inst.else_scope():
            with self.tik_inst.if_scope(tp.trans_axis_num == 2):
                tp.dst_addr.set_as(tp.rt_tuple[0] * tp.dst_jump_stride[0] + \
                                   tp.rt_tuple[1] * tp.dst_jump_stride[1])
            with self.tik_inst.else_scope():
                with self.tik_inst.if_scope(tp.trans_axis_num == 3):
                    tp.dst_addr.set_as(tp.rt_tuple[0] * tp.dst_jump_stride[0] + \
                                       tp.rt_tuple[1] * tp.dst_jump_stride[1] + \
                                       tp.rt_tuple[2] * tp.dst_jump_stride[2])
                with self.tik_inst.else_scope():
                    with self.tik_inst.if_scope(tp.trans_axis_num == 4):
                        tp.dst_addr.set_as(tp.rt_tuple[0] * tp.dst_jump_stride[0] + \
                                           tp.rt_tuple[1] * tp.dst_jump_stride[1] + \
                                           tp.rt_tuple[2] * tp.dst_jump_stride[2] + \
                                           tp.rt_tuple[3] * tp.dst_jump_stride[3])
                    with self.tik_inst.else_scope():

                        with self.tik_inst.if_scope(tp.trans_axis_num == 5):
                            tp.dst_addr.set_as(tp.rt_tuple[0] * tp.dst_jump_stride[0] + \
                                               tp.rt_tuple[1] * tp.dst_jump_stride[1] + \
                                               tp.rt_tuple[2] * tp.dst_jump_stride[2] + \
                                               tp.rt_tuple[3] * tp.dst_jump_stride[3] + \
                                               tp.rt_tuple[4] * tp.dst_jump_stride[4])

                        with self.tik_inst.if_scope(tp.trans_axis_num == 6):
                            tp.dst_addr.set_as(tp.rt_tuple[0] * tp.dst_jump_stride[0] + \
                                               tp.rt_tuple[1] * tp.dst_jump_stride[1] + \
                                               tp.rt_tuple[2] * tp.dst_jump_stride[2] + \
                                               tp.rt_tuple[3] * tp.dst_jump_stride[3] + \
                                               tp.rt_tuple[4] * tp.dst_jump_stride[4] + \
                                               tp.rt_tuple[5] * tp.dst_jump_stride[5])

                        with self.tik_inst.if_scope(tp.trans_axis_num == 7):
                            tp.dst_addr.set_as(tp.rt_tuple[0] * tp.dst_jump_stride[0] + \
                                               tp.rt_tuple[1] * tp.dst_jump_stride[1] + \
                                               tp.rt_tuple[2] * tp.dst_jump_stride[2] + \
                                               tp.rt_tuple[3] * tp.dst_jump_stride[3] + \
                                               tp.rt_tuple[4] * tp.dst_jump_stride[4] + \
                                               tp.rt_tuple[5] * tp.dst_jump_stride[5] + \
                                               tp.rt_tuple[6] * tp.dst_jump_stride[6])
                        
                        with self.tik_inst.if_scope(tp.trans_axis_num == 8):
                            tp.dst_addr.set_as(tp.rt_tuple[0] * tp.dst_jump_stride[0] + \
                                               tp.rt_tuple[1] * tp.dst_jump_stride[1] + \
                                               tp.rt_tuple[2] * tp.dst_jump_stride[2] + \
                                               tp.rt_tuple[3] * tp.dst_jump_stride[3] + \
                                               tp.rt_tuple[4] * tp.dst_jump_stride[4] + \
                                               tp.rt_tuple[5] * tp.dst_jump_stride[5] + \
                                               tp.rt_tuple[6] * tp.dst_jump_stride[6] + \
                                               tp.rt_tuple[7] * tp.dst_jump_stride[7])

    # 'pylint:disable=invalid-name
    @staticmethod
    def _init_tuple_common(tp):
        for i in range(TRANSPOSE_MAX_AXIS_NUM):
            tp.rt_tuple[i].set_as(tp.init_tuple[i])

    # 'pylint:disable=invalid-name
    def _convert_data_type(self, ub_input):
        if self.b8_times == EPB8:
            self.m_ub_input = ub_input.reinterpret_cast_to("int8")
            self.m_data_in = self.data_in.reinterpret_cast_to("int8")
            self.m_data_out = self.data_out.reinterpret_cast_to("int8")
            self.m_offset_factor = EPB8

    # 'pylint:disable=invalid-name
    def _copy_in_s1_aggregate_aligned(self, tp, ub_input, burst_len, ub_offset):
        if self.is_mte_pad and self.b8_times != EPB8:
            self.m_ub_input = ub_input
            self.m_data_in = self.data_in
            with self.tik_inst.if_scope(tp.aggregate_loop_unit > tp.dst_jump_factor[0]):
                self._copy_in_s1_pad_loop_greater(tp, ub_offset, tp.aggregate_loop_unit, burst_len)
            with self.tik_inst.else_scope():
                self._copy_in_s1_pad_loop_smaller(tp, ub_offset, tp.aggregate_loop_unit, burst_len)
        else:
            with self.tik_inst.new_stmt_scope(disable_sync=True):
                with self.tik_inst.for_range(0, tp.aggregate_loop_unit) as loop:
                    self._get_src_addr_s1(tp)
                    self._update_tuple(tp.trans_axis_num, tp.rt_tuple, tp.dst_jump_factor)
                    self.tik_inst.data_move(ub_input[loop * burst_len * self.ele_per_block], 
                                            self.data_in[tp.src_addr], 
                                            0, 1, burst_len, 0, 0)

    # 'pylint:disable=invalid-name
    def _copy_out_s1_aggregate_aligned(self, tp, ub_input, burst_len, ub_offset):
        if self.is_mte_pad and self.b8_times != EPB8:
            self.m_ub_input = ub_input
            self.m_data_out = self.data_out
            with self.tik_inst.if_scope(tp.aggregate_loop_unit > tp.dst_jump_factor[0]):
                self._copy_out_s1_pad_loop_greater(tp, ub_offset, tp.aggregate_loop_unit, burst_len)
            with self.tik_inst.else_scope():
                self._copy_out_s1_pad_loop_smaller(tp, ub_offset, tp.aggregate_loop_unit, burst_len)
        else:
            with self.tik_inst.for_range(0, tp.aggregate_loop_unit) as loop:
                self.tik_inst.data_move(self.data_out[tp.dst_addr + loop * tp.last_axis_len],
                                        ub_input[loop * burst_len * self.ele_per_block], 0, 1, burst_len, 0, 0)

    # 'pylint:disable=invalid-name
    def _copy_in_s1_pad_loop_greater(self, tp, ub_offset, loops, burst_len):
        ub_offset.set_as(0)
        self.head_repeat.set_as((tp.dst_jump_factor[0] - (tp.rt_tuple[0] % \
                                tp.dst_jump_factor[0])) % tp.dst_jump_factor[0])
        self.mid_repeats.set_as((loops - self.head_repeat) // tp.dst_jump_factor[0])
        self.tail_repeat.set_as((loops - self.mid_repeats * \
                                 tp.dst_jump_factor[0] - self.head_repeat) % tp.dst_jump_factor[0])
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.if_scope(self.head_repeat > 0):
                self._get_src_addr_s1(tp)
                self._update_tuple(tp.trans_axis_num, tp.rt_tuple, tp.dst_jump_factor, self.head_repeat)
                data_move_pad_info = (self.m_ub_input, self.m_data_in, ub_offset, 
                                      tp.src_addr * self.m_offset_factor, self.head_repeat, 
                                      tp.last_axis_len, 0, 
                                      tp.src_jump_stride[0] - tp.last_axis_len)
                self._data_move_pad_in_process(data_move_pad_info)
                ub_offset.set_as(ub_offset + self.head_repeat * burst_len * self.ele_per_block * self.m_offset_factor)
            with self.tik_inst.for_range(0, self.mid_repeats):
                self._get_src_addr_s1(tp)
                self._update_tuple(tp.trans_axis_num, tp.rt_tuple, tp.dst_jump_factor, tp.dst_jump_factor[0])
                data_move_pad_info = (self.m_ub_input, self.m_data_in, ub_offset, 
                                      tp.src_addr * self.m_offset_factor, tp.dst_jump_factor[0], 
                                      tp.last_axis_len, 0, 
                                      tp.src_jump_stride[0] - tp.last_axis_len)
                self._data_move_pad_in_process(data_move_pad_info)
                ub_offset.set_as(ub_offset + tp.dst_jump_factor[0] * \
                                 burst_len * self.ele_per_block * self.m_offset_factor)
            with self.tik_inst.if_scope(self.tail_repeat > 0):
                self._get_src_addr_s1(tp)
                self._update_tuple(tp.trans_axis_num, tp.rt_tuple, tp.dst_jump_factor, self.tail_repeat)
                data_move_pad_info = (self.m_ub_input, self.m_data_in, ub_offset, 
                                      tp.src_addr * self.m_offset_factor, self.tail_repeat, 
                                      tp.last_axis_len, 0, 
                                      tp.src_jump_stride[0] - tp.last_axis_len)
                self._data_move_pad_in_process(data_move_pad_info)
                ub_offset.set_as(ub_offset + self.tail_repeat * burst_len * self.ele_per_block * self.m_offset_factor)

    # 'pylint:disable=invalid-name
    def _copy_in_s1_pad_loop_smaller(self, tp, ub_offset, loops, burst_len):
        ub_offset.set_as(0)
        self.head_repeat.set_as(tp.dst_jump_factor[0] - (tp.rt_tuple[0] % tp.dst_jump_factor[0]))
        with self.tik_inst.if_scope(self.head_repeat >= loops):
            self.head_repeat.set_as(loops)
            self.tail_repeat.set_as(0)
        with self.tik_inst.else_scope():
            self.tail_repeat.set_as(loops - self.head_repeat)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.if_scope(self.head_repeat > 0):
                self._get_src_addr_s1(tp)
                self._update_tuple(tp.trans_axis_num, tp.rt_tuple, tp.dst_jump_factor, self.head_repeat)
                data_move_pad_info = (self.m_ub_input, self.m_data_in, ub_offset, 
                                      tp.src_addr * self.m_offset_factor, self.head_repeat, 
                                      tp.last_axis_len, 0, 
                                      tp.src_jump_stride[0] - tp.last_axis_len)
                self._data_move_pad_in_process(data_move_pad_info)
                ub_offset.set_as(ub_offset + self.head_repeat * burst_len * self.ele_per_block * self.m_offset_factor)
            with self.tik_inst.if_scope(self.tail_repeat > 0):
                self._get_src_addr_s1(tp)
                self._update_tuple(tp.trans_axis_num, tp.rt_tuple, tp.dst_jump_factor, self.tail_repeat)
                data_move_pad_info = (self.m_ub_input, self.m_data_in, ub_offset, 
                                      tp.src_addr * self.m_offset_factor, self.tail_repeat, 
                                      tp.last_axis_len, 0, 
                                      tp.src_jump_stride[0] - tp.last_axis_len)
                self._data_move_pad_in_process(data_move_pad_info)
                ub_offset.set_as(ub_offset + self.tail_repeat * burst_len * self.ele_per_block * self.m_offset_factor)

    # 'pylint:disable=invalid-name
    def _copy_out_s1_pad_loop_greater(self, tp, ub_offset, loops, burst_len):
        ub_offset.set_as(0)
        self.head_repeat.set_as((tp.dst_jump_factor[0] - (tp.rt_tuple[0] % \
                                tp.dst_jump_factor[0])) % tp.dst_jump_factor[0])
        self.mid_repeats.set_as((loops - self.head_repeat) // tp.dst_jump_factor[0])
        self.tail_repeat.set_as((loops - self.mid_repeats * \
                                 tp.dst_jump_factor[0] - self.head_repeat) % tp.dst_jump_factor[0])
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.if_scope(self.head_repeat > 0):
                data_move_pad_info = (self.m_data_out, self.m_ub_input, tp.dst_addr, 
                                      ub_offset, self.head_repeat, 
                                      tp.last_axis_len, 0, 0)
                self._data_move_pad_out_process(data_move_pad_info)
                ub_offset.set_as(ub_offset + self.head_repeat * burst_len * self.ele_per_block * self.m_offset_factor)
                tp.dst_addr.set_as((tp.dst_addr + self.head_repeat * tp.last_axis_len) * self.m_offset_factor)
            with self.tik_inst.for_range(0, self.mid_repeats):
                data_move_pad_info = (self.m_data_out, self.m_ub_input, tp.dst_addr, 
                                      ub_offset, tp.dst_jump_factor[0], 
                                      tp.last_axis_len, 0, 0)
                self._data_move_pad_out_process(data_move_pad_info)
                ub_offset.set_as(ub_offset + tp.dst_jump_factor[0] * \
                                 burst_len * self.ele_per_block * self.m_offset_factor)
                tp.dst_addr.set_as((tp.dst_addr + tp.dst_jump_factor[0] * tp.last_axis_len) * self.m_offset_factor)
            with self.tik_inst.if_scope(self.tail_repeat > 0):
                data_move_pad_info = (self.m_data_out, self.m_ub_input, tp.dst_addr, 
                                      ub_offset, self.tail_repeat, 
                                      tp.last_axis_len, 0, 0)
                self._data_move_pad_out_process(data_move_pad_info)
                ub_offset.set_as(ub_offset + self.tail_repeat * burst_len * self.ele_per_block * self.m_offset_factor)

    # 'pylint:disable=invalid-name
    def _copy_out_s1_pad_loop_smaller(self, tp, ub_offset, loops, burst_len):
        ub_offset.set_as(0)
        self.head_repeat.set_as(tp.dst_jump_factor[0] - (tp.rt_tuple[0] % tp.dst_jump_factor[0]))
        with self.tik_inst.if_scope(self.head_repeat >= loops):
            self.head_repeat.set_as(loops)
            self.tail_repeat.set_as(0)
        with self.tik_inst.else_scope():
            self.tail_repeat.set_as(loops - self.head_repeat)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.if_scope(self.head_repeat > 0):
                data_move_pad_info = (self.m_data_out, self.m_ub_input, tp.dst_addr, 
                                      ub_offset, self.head_repeat, 
                                      tp.last_axis_len, 0, 0)
                self._data_move_pad_out_process(data_move_pad_info)
                ub_offset.set_as(ub_offset + self.head_repeat * burst_len * self.ele_per_block * self.m_offset_factor)
                tp.dst_addr.set_as((tp.dst_addr + self.head_repeat * tp.last_axis_len) * self.m_offset_factor)
            with self.tik_inst.if_scope(self.tail_repeat > 0):
                data_move_pad_info = (self.m_data_out, self.m_ub_input, tp.dst_addr, 
                                      ub_offset, self.tail_repeat, 
                                      tp.last_axis_len, 0, 0)
                self._data_move_pad_out_process(data_move_pad_info)
                ub_offset.set_as(ub_offset + self.tail_repeat * burst_len * self.ele_per_block * self.m_offset_factor)

    # 'pylint:disable=invalid-name
    def _copy_in_s1_aggregate_n_aligned(self, tp, ub_input, burst_len, ub_offset):
        if self.is_mte_pad and self.b8_times != EPB8:
            self.m_ub_input = ub_input
            self.m_data_in = self.data_in
            with self.tik_inst.if_scope(tp.aggregate_loop_unit > tp.dst_jump_factor[0]):
                self._copy_in_s1_pad_loop_greater(tp, ub_offset, tp.aggregate_loop_unit, burst_len)
            with self.tik_inst.else_scope():
                self._copy_in_s1_pad_loop_smaller(tp, ub_offset, tp.aggregate_loop_unit, burst_len)
        else:
            self._copy_in_s1_n_pad(tp, ub_input, burst_len)

    # 'pylint:disable=invalid-name
    def _copy_out_s1_n_pad(self, tp, ub_input, burst_len):
        if self.is_mte_pad:
            self.m_ub_input = ub_input
            if self.b8_times == EPB8:
                self.m_ub_input = ub_input.reinterpret_cast_to("int8")
                self.m_data_out = self.data_out.reinterpret_cast_to("int8")
                self.m_offset_factor = EPB8
            with self.tik_inst.new_stmt_scope(disable_sync=True):
                with self.tik_inst.for_range(0, tp.aggregate_loop_unit) as loop:
                    self.tik_inst.data_move_pad(self.m_data_out[(tp.dst_addr + loop * tp.last_axis_len) * 
                                                                    self.m_offset_factor],
                                                    self.m_ub_input[loop * burst_len * self.ele_per_block *
                                                                    self.m_offset_factor],
                                                    1, tp.last_axis_len * self.b8_times, 0, 0)
        else:
            with self.tik_inst.new_stmt_scope(disable_sync=True):
                with self.tik_inst.for_range(0, tp.aggregate_loop_unit) as loop:
                    self.tik_inst.data_move(self.data_out[tp.dst_addr + loop * tp.last_axis_len],
                                                ub_input[loop * burst_len * self.ele_per_block], 
                                                0, 1, burst_len - 1, 0, 0)
            with self.tik_inst.new_stmt_scope(disable_sync=True):
                with self.tik_inst.for_range(0, tp.aggregate_loop_unit) as loop:
                    self.tik_inst.data_move(self.data_out[tp.dst_addr + (loop + 1) * \
                                                tp.last_axis_len - self.ele_per_block],
                                                ub_input[(loop + 1) * burst_len * \
                                                self.ele_per_block - self.ele_per_block],
                                                0, 1, 1, 0, 0)

    # 'pylint:disable=invalid-name
    def _copy_in_s1_n_pad(self, tp, ub_input, burst_len):
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, tp.aggregate_loop_unit) as loop:
                self._get_src_addr_s1(tp)
                self._update_tuple(tp.trans_axis_num, tp.rt_tuple, tp.dst_jump_factor)
                if self.is_mte_pad:
                    self.m_ub_input = ub_input
                    self._convert_data_type(ub_input)
                    self.tik_inst.data_move_pad(self.m_ub_input[loop * burst_len * self.ele_per_block *
                                                                self.m_offset_factor],
                                                    self.m_data_in[tp.src_addr * self.m_offset_factor],
                                                    1, tp.last_axis_len * self.b8_times, 0, 0)
                else:
                    self.tik_inst.data_move(ub_input[loop * burst_len * self.ele_per_block],
                                                self.data_in[tp.src_addr], 0, 1, burst_len - 1, 0, 0)
                    self.tik_inst.data_move(ub_input[(loop + 1) * burst_len * self.ele_per_block - self.ele_per_block],
                                                self.data_in[tp.src_addr + tp.last_axis_len - self.ele_per_block],
                                                0, 1, 1, 0, 0)

    # 'pylint:disable=invalid-name
    def _copy_out_s1_aggregate_n_aligned(self, tp, ub_input, burst_len, ub_offset):
        if self.is_mte_pad and self.b8_times != EPB8:
            self.m_ub_input = ub_input
            self.m_data_out = self.data_out
            with self.tik_inst.if_scope(tp.aggregate_loop_unit > tp.dst_jump_factor[0]):
                self._copy_out_s1_pad_loop_greater(tp, ub_offset, tp.aggregate_loop_unit, burst_len)
            with self.tik_inst.else_scope():
                self._copy_out_s1_pad_loop_smaller(tp, ub_offset, tp.aggregate_loop_unit, burst_len)
        else:
            self._copy_out_s1_n_pad(tp, ub_input, burst_len)

    # 'pylint:disable=invalid-name
    def _copy_in_s1(self, tp, ub_input, burst_len, actual_len):
        self._get_src_addr_s1(tp)
        self._do_data_move_no_stride(ub_input, self.data_in[tp.src_addr], 1, burst_len, actual_len)

    # 'pylint:disable=invalid-name
    def _copy_out_s1(self, tp, ub_input, burst_len):
        self._get_dst_addr_s1(tp)
        self.tik_inst.data_move(self.data_out[tp.dst_addr], ub_input, 0, 1, burst_len, 0, 0)

    # 'pylint:disable=invalid-name
    def _copy_anti_overlap_s1(self, tp, ub_input):
        skip_offset = self.tik_inst.Scalar("int32")
        skip_offset.set_as((tp.last_axis_burst_len - 1) * self.ele_per_block)
        skip_offset.set_as(skip_offset - (self.ele_per_block - (tp.last_axis_len - skip_offset)))
        scalar_value = self.tik_inst.Scalar(self.x_dtype)
        with self.tik_inst.for_range(0, self.ele_per_block) as i:
            scalar_value.set_as(ub_input[skip_offset + i])
            ub_input[i] = scalar_value
        self.tik_inst.data_move(self.data_out[tp.dst_addr + skip_offset], ub_input, 0, 1, 1, 0, 0)

    # 'pylint:disable=invalid-name
    def _move_data_s1(self, tp, ub_input_64):
        ub_offset = self.tik_inst.Scalar("int32")  # unit : block
        ub_input = ub_input_64.reinterpret_cast_to(self.x_dtype)

        self._init_tuple_common(tp)
        with self.tik_inst.if_scope(tp.loop_num > 0):
            with self.tik_inst.if_scope(tp.align_ele == 0):

                with self.tik_inst.for_range(0, tp.aggregate_loop_num):
                    self._get_dst_addr_s1(tp)
                    self._copy_in_s1_aggregate_aligned(tp, ub_input, tp.last_axis_burst_len, ub_offset)
                    self._copy_out_s1_aggregate_aligned(tp, ub_input, tp.last_axis_burst_len, ub_offset)

                with self.tik_inst.for_range(0, tp.aggregate_loop_tail):
                    self._copy_in_s1(tp, ub_input, tp.last_axis_burst_len, tp.last_axis_len)
                    self._copy_out_s1(tp, ub_input, tp.last_axis_burst_len)
                    self._update_tuple(tp.trans_axis_num, tp.rt_tuple, tp.dst_jump_factor)

            with self.tik_inst.else_scope():

                with self.tik_inst.for_range(0, tp.aggregate_loop_num):
                    self._get_dst_addr_s1(tp)
                    self._copy_in_s1_aggregate_n_aligned(tp, ub_input, tp.last_axis_burst_len, ub_offset)
                    self._copy_out_s1_aggregate_n_aligned(tp, ub_input, tp.last_axis_burst_len, ub_offset)

                with self.tik_inst.for_range(0, tp.aggregate_loop_tail):
                    self._copy_in_s1(tp, ub_input, tp.last_axis_burst_len, tp.last_axis_len)
                    self._copy_out_s1(tp, ub_input, tp.last_axis_burst_len)
                    self._update_tuple(tp.trans_axis_num, tp.rt_tuple, tp.dst_jump_factor)

                self._copy_in_s1(tp, ub_input, tp.last_axis_burst_len, tp.last_axis_len)
                if self.is_mte_pad:
                    self._get_dst_addr_s1(tp)
                    self.m_ub_input = ub_input
                    self._convert_data_type(ub_input)
                    self.tik_inst.data_move_pad(self.m_data_out[tp.dst_addr * self.m_offset_factor], self.m_ub_input,
                                                1, tp.last_axis_len * self.b8_times, 0, 0)
                else:
                    self._copy_out_s1(tp, ub_input, tp.last_axis_burst_len - 1)
                    self._copy_anti_overlap_s1(tp, ub_input)

    # -------------------------------------------------------------------------------------------------
    #                                    scenario_2
    # -------------------------------------------------------------------------------------------------
    # 'pylint: disable=too-many-arguments, unused-argument, invalid-name
    def _reorder_s2(self, tp, ub_input, ub_offset, ub_offset_exclude_pad):
        if self.x_dtype in ("int8", "uint8", "bool"):
            b8_offset = _get_half_ub() * 2
            ub_input_b8 = ub_input.reinterpret_cast_to("int8")
            src_ele_num_in_b8 = self._get_src_size() * 2  # avoid bank conflict
            src_list = [ub_input_b8[src_ele_num_in_b8 * i] for i in range(EPB16)]
            dst_list_low = [ub_input_b8[b8_offset + EPB32 * i] for i in range(EPB16)]
            dst_list_high = [ub_input_b8[b8_offset + EPB32 * i + EPB32 * EPB16] for i in range(EPB16)]

            with self.tik_inst.if_scope(ub_offset == 1):
                self.tik_inst.vnchwconv(False, False, dst_list_low, src_list, 1, 0, 0)
                self.tik_inst.vnchwconv(False, True, dst_list_high, src_list, 1, 0, 0)
            with self.tik_inst.if_scope(ub_offset != 1):
                self.tik_inst.vnchwconv(False, False, dst_list_low, src_list, ub_offset, EPB32, 1)
                self.tik_inst.vnchwconv(False, True, dst_list_high, src_list, ub_offset, EPB32, 1)

            # step2. erase unused elements aligned
            all_line_number = tp.last_axis_burst_len * EPB32
            pad_line_number = tp.align_ele * self.fp16_times
            nburst = ub_offset // tp.last_axis_burst_len
            burst_len = all_line_number - pad_line_number
            self.tik_inst.data_move(ub_input_b8, ub_input_b8[b8_offset], 0, nburst, burst_len, pad_line_number, 0)

            # step3. make all elements in the first col be in memory of contiguous
            ub_offset_exclude_pad.set_as(((all_line_number - pad_line_number) * nburst + EPB32 - 1) // EPB32)
            src_list_low = [ub_input_b8[EPB32 * i] for i in range(EPB16)]
            src_list_high = [ub_input_b8[EPB32 * i + EPB32 * EPB16] for i in range(EPB16)]
            dst_list = [ub_input_b8[b8_offset + self._get_dst_size() * EPB32 * i] for i in range(EPB16)]

            with self.tik_inst.if_scope(ub_offset_exclude_pad == 1):
                self.tik_inst.vnchwconv(False, False, dst_list, src_list_low, 1, 0, 0)
                self.tik_inst.vnchwconv(True, False, dst_list, src_list_high, 1, 0, 0)
            with self.tik_inst.if_scope(ub_offset_exclude_pad > 1):
                self.tik_inst.vnchwconv(False, False, dst_list, src_list_low, ub_offset_exclude_pad, 1, EPB32)
                self.tik_inst.vnchwconv(True, False, dst_list, src_list_high, ub_offset_exclude_pad, 1, EPB32)
            self.tik_inst.data_move(ub_input_b8, ub_input_b8[b8_offset], 0, 1, ub_offset_exclude_pad, 0, 0)
        else:
            # step1. make all elements in the first col
            fp16_offset_1 = ACCU_BLOCK_SIZE * 32
            fp16_offset_2 = ACCU_BLOCK_SIZE * 32 + ACCU_BLOCK_SIZE * 32 * 16
            ub_input_fp16 = ub_input.reinterpret_cast_to("float16")
            src_ele_num_in_fp16 = self._get_src_size()  # avoid bank conflict
            src_list = [ub_input_fp16[src_ele_num_in_fp16 * i] for i in range(EPB16)]
            dst_list = [ub_input_fp16[fp16_offset_1 + EPB16 * i] for i in range(EPB16)]
            with self.tik_inst.if_scope(ub_offset == 1):
                self.tik_inst.vnchwconv(False, False, dst_list, src_list, 1, 0, 0)
            with self.tik_inst.if_scope(ub_offset != 1):
                self.tik_inst.vnchwconv(False, False, dst_list, src_list, ub_offset, EPB16, 1)

            # step2. erase unused elements aligned
            all_line_number = tp.last_axis_burst_len * EPB16
            pad_line_number = tp.align_ele * self.fp16_times
            nburst = ub_offset // tp.last_axis_burst_len
            burst_len = all_line_number - pad_line_number
            self.tik_inst.data_move(ub_input_fp16[fp16_offset_2], ub_input_fp16[fp16_offset_1], 0, nburst, burst_len,
                                    pad_line_number, 0)

            # step3. make all elements in the first col be in memory of contiguous
            ub_offset_exclude_pad.set_as(((all_line_number - pad_line_number) * nburst + EPB16 - 1) // EPB16)
            src_list = [ub_input_fp16[fp16_offset_2 + EPB16 * i] for i in range(EPB16)]
            # 247 avoid bank conflict
            dst_list = [ub_input_fp16[self._get_dst_size() * EPB16 * i] for i in range(EPB16)]

            with self.tik_inst.if_scope(ub_offset_exclude_pad == 1):
                self.tik_inst.vnchwconv(False, False, dst_list, src_list, 1, 0, 0)
            with self.tik_inst.if_scope(ub_offset_exclude_pad > 1):
                self.tik_inst.vnchwconv(False, False, dst_list, src_list, ub_offset_exclude_pad, 1, EPB16)

    def _get_src_addr_s2(self, tp):
        self._get_src_addr_s1(tp)

    @staticmethod
    def _get_dst_addr_s2(tp, steps):
        tp.dst_addr.set_as((tp.base + steps) * tp.last_axis_len)

    def _copy_out_s2(self, tp, ub_input, accu_blocks, backup_steps, steps):
        ub_offset_exclude_pad = self.tik_inst.Scalar("int32")  # unit : block
        ub_offset_exclude_pad.set_as(accu_blocks)
        with self.tik_inst.if_scope(tp.align_ele != 0):
            self._reorder_s2(tp, ub_input, accu_blocks, ub_offset_exclude_pad)
        self._get_dst_addr_s2(tp, backup_steps)
        self.tik_inst.data_move(self.data_out[tp.dst_addr], ub_input, 0, 1, ub_offset_exclude_pad, 0, 0)
        backup_steps.set_as(steps)
        accu_blocks.set_as(0)

    def _copy_common_s2(self, tp, ub_input, steps, accu_blocks, major_loop, major_num, tail_num):
        backup_steps = self.tik_inst.Scalar("int64", init_value=0)
        backup_steps.set_as(steps)
        tik_inst = self.tik_inst
        accu_block_size = ACCU_BLOCK_SIZE
        if self.fp16_times == 1:
            accu_block_size = ACCU_BLOCK_SIZE // 2
        with tik_inst.for_range(0, major_loop):
            self._get_src_addr_s2(tp)
            self._do_data_move_in_with_stride(ub_input[accu_blocks * self.ele_per_block], self.data_in[tp.src_addr],
                                              major_num, tp.last_axis_burst_len, tp.last_axis_len, 0, tp.src_stride)
            steps.set_as(steps + major_num)
            self._update_tuple_with_steps(tp.trans_axis_num, tp.rt_tuple, tp.dst_jump_factor, tp.dst_jump_factor_mod,
                                          tp.base, steps)
            accu_blocks.set_as(accu_blocks + major_num * tp.last_axis_burst_len)
            with self.tik_inst.if_scope(accu_blocks >= accu_block_size):  # 64=2KB, 128=4KB, 200=6.4KB
                self._copy_out_s2(tp, ub_input, accu_blocks, backup_steps, steps)

        with tik_inst.if_scope(tail_num != 0):
            self._get_src_addr_s2(tp)
            self._do_data_move_in_with_stride(ub_input[accu_blocks * self.ele_per_block], self.data_in[tp.src_addr],
                                              tail_num, tp.last_axis_burst_len, tp.last_axis_len, 0, tp.src_stride)
            steps.set_as(steps + tail_num)
            self._update_tuple_with_steps(tp.trans_axis_num, tp.rt_tuple, tp.dst_jump_factor, tp.dst_jump_factor_mod,
                                          tp.base, steps)
            accu_blocks.set_as(accu_blocks + tail_num * tp.last_axis_burst_len)
            with self.tik_inst.if_scope(accu_blocks >= accu_block_size):  # 64=2KB, 128=4KB, 200=6.4KB
                self._copy_out_s2(tp, ub_input, accu_blocks, backup_steps, steps)

        with self.tik_inst.if_scope(accu_blocks != 0):
            self._copy_out_s2(tp, ub_input, accu_blocks, backup_steps, steps)

    def _copy_head_s2_aligned(self, tp, ub_input, steps, accu_blocks):
        self._copy_common_s2(tp, ub_input, steps, accu_blocks, tp.head_major_loop, tp.head_major_num, tp.head_tail_num)

    def _copy_body_s2_aligned(self, tp, ub_input, steps, accu_blocks):
        with self.tik_inst.for_range(0, tp.body_loop):
            self._copy_common_s2(tp, ub_input, steps, accu_blocks, tp.body_major_loop, tp.body_major_num,
                                 tp.body_tail_num)

    def _copy_tail_s2_aligned(self, tp, ub_input, steps, accu_blocks):
        self._copy_common_s2(tp, ub_input, steps, accu_blocks, tp.tail_major_loop, tp.tail_major_num, tp.tail_tail_num)

    def _get_shape_size_s2(self, tp):
        shape_size = self.tik_inst.Scalar()
        with self.tik_inst.if_scope(tp.trans_axis_num == 0):
            shape_size.set_as(0)
        with self.tik_inst.elif_scope(tp.trans_axis_num == 1):
            shape_size.set_as(tp.dst_jump_factor[0])
        with self.tik_inst.elif_scope(tp.trans_axis_num == 2):
            shape_size.set_as(tp.dst_jump_factor[0] * tp.dst_jump_factor[1])
        with self.tik_inst.elif_scope(tp.trans_axis_num == 3):
            shape_size.set_as(tp.dst_jump_factor[0] * tp.dst_jump_factor[1] * tp.dst_jump_factor[2])
        with self.tik_inst.elif_scope(tp.trans_axis_num == 4):
            shape_size.set_as(tp.dst_jump_factor[0] * tp.dst_jump_factor[1] * tp.dst_jump_factor[2] * \
                              tp.dst_jump_factor[3])
        with self.tik_inst.elif_scope(tp.trans_axis_num == 5):
            shape_size.set_as(tp.dst_jump_factor[0] * tp.dst_jump_factor[1] * tp.dst_jump_factor[2] * \
                              tp.dst_jump_factor[3] * tp.dst_jump_factor[4])
        with self.tik_inst.elif_scope(tp.trans_axis_num == 6):
            shape_size.set_as(tp.dst_jump_factor[0] * tp.dst_jump_factor[1] * tp.dst_jump_factor[2] * \
                              tp.dst_jump_factor[3] * tp.dst_jump_factor[4] * tp.dst_jump_factor[5])
        with self.tik_inst.elif_scope(tp.trans_axis_num == 7):
            shape_size.set_as(tp.dst_jump_factor[0] * tp.dst_jump_factor[1] * tp.dst_jump_factor[2] * \
                              tp.dst_jump_factor[3] * tp.dst_jump_factor[4] * tp.dst_jump_factor[5] * \
                              tp.dst_jump_factor[6])
        with self.tik_inst.elif_scope(tp.trans_axis_num == 8):
            shape_size.set_as(tp.dst_jump_factor[0] * tp.dst_jump_factor[1] * tp.dst_jump_factor[2] * \
                              tp.dst_jump_factor[3] * tp.dst_jump_factor[4] * tp.dst_jump_factor[5] * \
                              tp.dst_jump_factor[6] * tp.dst_jump_factor[7])
        return shape_size

    def _copy_tiny_data_lt_blk_s2(self, tp, ub_input, steps, accu_blocks):
        with self.tik_inst.if_scope(tp.loop_num != 0):
            ub_offset_exclude_pad = self.tik_inst.Scalar("int32")  # unit : block
            scalar_value = self.tik_inst.Scalar(self.x_dtype)
            steps.set_as(0)
            accu_blocks.set_as(0)
            shape_size = self._get_shape_size_s2(tp)
            # this set_as will nerver excute but is necessary
            # because the small shape bytes always < MAX_DATA_MOVE_PAD_BYTES
            # compile failed under the big shape case, becase pass has bugs
            with self.tik_inst.if_scope(shape_size * self.b8_times > MAX_DATA_MOVE_PAD_BYTES):
                shape_size.set_as(0)
            self._get_dst_addr_s2(tp, steps)
            self._update_tuple_with_steps(tp.trans_axis_num, tp.rt_tuple, tp.dst_jump_factor, tp.dst_jump_factor_mod,
                                          tp.base, steps)
            with self.tik_inst.for_range(0, tp.loop_num):
                self._get_src_addr_s2(tp)
                self._do_data_move_no_stride(ub_input[accu_blocks * self.ele_per_block], self.data_in[tp.src_addr],
                                             1, 1, 1)
                steps.set_as(steps + 1)
                self._update_tuple_with_steps(tp.trans_axis_num, tp.rt_tuple, tp.dst_jump_factor,
                                              tp.dst_jump_factor_mod, tp.base, steps)
                accu_blocks.set_as(accu_blocks + 1)
            self._reorder_s2(tp, ub_input, accu_blocks, ub_offset_exclude_pad)
            with self.tik_inst.for_range(0, tp.loop_num) as i:
                scalar_value.set_as(ub_input[i])
                ub_input[i] = scalar_value
            if self.is_mte_pad:
                self.m_ub_input = ub_input
                self._convert_data_type(ub_input)
                with self.tik_inst.if_scope(shape_size * self.b8_times >= BLOCK_SIZE):
                    self.tik_inst.data_move_pad(self.m_data_out[tp.dst_addr * self.m_offset_factor], self.m_ub_input,
                                                1, tp.loop_num * self.b8_times, 0, 0)
                with self.tik_inst.else_scope():
                    self.tik_inst.data_move_pad(self.m_data_out[tp.dst_addr * self.m_offset_factor], self.m_ub_input,
                                                1, shape_size * self.b8_times, 0, 0)
            else:
                self.tik_inst.data_move(self.data_out[tp.dst_addr], ub_input, 0, 1, 1, 0, 0)
            # for int32 skip_ele = 0 if last_axis is 1,2,4
            with self.tik_inst.if_scope(tp.skip_ele != 0):
                with self.tik_inst.for_range(0, self.ele_per_block) as i:
                    scalar_value.set_as(ub_input[tp.skip_ele + i])
                    ub_input[i] = scalar_value
                self.tik_inst.data_move(self.data_out[tp.dst_addr + tp.skip_ele], ub_input, 0, 1, 1, 0, 0)

    def _copy_anti_overlap_lt_blk_s2(self, tp, ub_input, steps, accu_blocks):
        with self.tik_inst.if_scope(tp.loop_num != 0):
            ub_offset_exclude_pad = self.tik_inst.Scalar("int32")  # unit : block
            scalar_value = self.tik_inst.Scalar(self.x_dtype)
            steps.set_as(tp.loop_num - tp.back_num)
            accu_blocks.set_as(0)
            self._get_dst_addr_s2(tp, steps)
            self._update_tuple_with_steps(tp.trans_axis_num, tp.rt_tuple, tp.dst_jump_factor, tp.dst_jump_factor_mod,
                                          tp.base, steps)
            with self.tik_inst.for_range(0, tp.back_num):
                self._get_src_addr_s2(tp)
                self._do_data_move_no_stride(ub_input[accu_blocks * self.ele_per_block], self.data_in[tp.src_addr],
                                             1, 1, tp.last_axis_len)
                steps.set_as(steps + 1)
                self._update_tuple_with_steps(tp.trans_axis_num, tp.rt_tuple, tp.dst_jump_factor,
                                              tp.dst_jump_factor_mod, tp.base, steps)
                accu_blocks.set_as(accu_blocks + 1)
            self._reorder_s2(tp, ub_input, accu_blocks, ub_offset_exclude_pad)
            with self.tik_inst.for_range(0, self.ele_per_block) as i:
                scalar_value.set_as(ub_input[i])
                ub_input[i] = scalar_value
            self.tik_inst.data_move(self.data_out[tp.dst_addr], ub_input, 0, 1, 1, 0, 0)
            # for int32 skip_ele = 0 if last_axis is 1,2,4
            with self.tik_inst.if_scope(tp.skip_ele != 0):
                with self.tik_inst.for_range(0, self.ele_per_block) as i:
                    scalar_value.set_as(ub_input[tp.skip_ele + i])
                    ub_input[i] = scalar_value
                self.tik_inst.data_move(self.data_out[tp.dst_addr + tp.skip_ele], ub_input, 0, 1, 1, 0, 0)

    def _copy_anti_overlap_gt_blk_s2(self, tp, ub_input, steps, accu_blocks):
        with self.tik_inst.if_scope(tp.loop_num != 0):
            scalar_value = self.tik_inst.Scalar(self.x_dtype)
            steps.set_as(tp.loop_num - 1)
            self._get_dst_addr_s2(tp, steps)
            self._get_src_addr_s2(tp)

            self._update_tuple_with_steps(tp.trans_axis_num, tp.rt_tuple, tp.dst_jump_factor, tp.dst_jump_factor_mod,
                                          tp.base, steps)

            if self.is_mte_pad:
                self.m_ub_input = ub_input
                if self.b8_times == EPB8:
                    self.m_ub_input = ub_input.reinterpret_cast_to("int8")
                    self.m_data_in = self.data_in.reinterpret_cast_to("int8")
                    self.m_data_out = self.data_out.reinterpret_cast_to("int8")
                    self.m_offset_factor = EPB8
                self.tik_inst.data_move_pad(self.m_ub_input, self.m_data_in[tp.src_addr * self.m_offset_factor],
                                            1, tp.last_axis_len * self.b8_times, 0, 0)
                self.tik_inst.data_move_pad(self.m_data_out[tp.dst_addr * self.m_offset_factor], self.m_ub_input,
                                            1, tp.last_axis_len * self.b8_times, 0, 0)
            else:
                self.tik_inst.data_move(ub_input, self.data_in[tp.src_addr], 0, 1, tp.last_axis_burst_len, 0, 0)
                self.tik_inst.data_move(self.data_out[tp.dst_addr], ub_input, 0, 1, tp.last_axis_burst_len - 1, 0, 0)
                with self.tik_inst.for_range(0, self.ele_per_block) as i:
                    scalar_value.set_as(ub_input[tp.last_axis_len - self.ele_per_block + i])
                    ub_input[i] = scalar_value
                self.tik_inst.data_move(self.data_out[tp.dst_addr + tp.last_axis_len - self.ele_per_block], ub_input,
                                        0, 1, 1, 0, 0)

    def _move_data_s2(self, tp, ub_input_64):
        ub_offset = self.tik_inst.Scalar("int32")  # unit : block
        steps = self.tik_inst.Scalar("int64", init_value=0)
        accu_blocks = self.tik_inst.Scalar("int32", init_value=0)  # unit : block
        ub_input = ub_input_64.reinterpret_cast_to(self.x_dtype)

        #                   <----------------this core data---------------->
        #   <----------------------->|<----------------------->|<----------------------->

        #   -----------------------------------------------------------------------------
        #   |               |        |                         |           |            |
        #   |               | head   |      body               |      tail |            |
        #   |               |        |                         |           |            |
        #   -----------------------------------------------------------------------------

        self._init_tuple_common(tp)
        self._copy_head_s2_aligned(tp, ub_input, steps, accu_blocks)
        self._copy_body_s2_aligned(tp, ub_input, steps, accu_blocks)
        self._copy_tail_s2_aligned(tp, ub_input, steps, accu_blocks)

        with self.tik_inst.if_scope(tik.all(tp.head_major_loop == 0,
                                            tp.body_loop == 0,
                                            tp.tail_major_loop == 0,
                                            tp.last_axis_len < self.ele_per_block)):
            self._copy_tiny_data_lt_blk_s2(tp, ub_input, steps, accu_blocks)
        with self.tik_inst.else_scope():
            with self.tik_inst.if_scope(tp.last_axis_len < self.ele_per_block):
                self._copy_anti_overlap_lt_blk_s2(tp, ub_input, steps, accu_blocks)
            with self.tik_inst.if_scope(tp.align_ele != 0):
                with self.tik_inst.if_scope(tp.last_axis_len > self.ele_per_block):
                    self._copy_anti_overlap_gt_blk_s2(tp, ub_input, steps, accu_blocks)

    # -------------------------------------------------------------------------------------------------
    #                                    scenario_3
    # -------------------------------------------------------------------------------------------------
    def _copy_in_major_s3(self, tp, ub_input, last_axis_offset):
        self.tik_inst.data_move(ub_input, self.data_in[tp.src_addr + last_axis_offset], 0, 1, tp.major_blocks, 0, 0)

    def _copy_out_major_s3(self, tp, ub_input, last_axis_offset):
        self.tik_inst.data_move(self.data_out[tp.dst_addr + last_axis_offset], ub_input, 0, 1, tp.major_blocks, 0, 0)

    def _update_last_axis_offset(self, tp, last_axis_offset):
        last_axis_offset.set_as(last_axis_offset + tp.major_blocks * self.ele_per_block)

    def _copy_in_tail_s3(self, tp, ub_input, last_axis_offset):
        self.tik_inst.data_move(ub_input, self.data_in[tp.src_addr + last_axis_offset - tp.back_ele],
                                0, 1, tp.tail_blocks, 0, 0)

    def _copy_out_tail_s3(self, tp, ub_input, last_axis_offset):
        self.tik_inst.data_move(self.data_out[tp.dst_addr + last_axis_offset - tp.back_ele], ub_input,
                                0, 1, tp.tail_blocks, 0, 0)

    def _get_src_addr_s3(self, tp):
        self._get_src_addr_s1(tp)

    def _get_dst_addr_s3(self, tp):
        self._get_dst_addr_s1(tp)

    def _move_data_s3(self, tp, ub_input_64):
        ub_offset = self.tik_inst.Scalar("int32")  # unit : block
        last_axis_offset = self.tik_inst.Scalar("int32")  # unit : ele
        ub_input = ub_input_64.reinterpret_cast_to(self.x_dtype)
        tik_inst = self.tik_inst

        self._init_tuple_common(tp)
        with tik_inst.for_range(0, tp.loop_num):
            last_axis_offset.set_as(0)
            self._get_src_addr_s3(tp)
            self._get_dst_addr_s3(tp)
            with tik_inst.for_range(0, tp.major_loop_num):
                self._copy_in_major_s3(tp, ub_input, last_axis_offset)
                self._copy_out_major_s3(tp, ub_input, last_axis_offset)
                self._update_last_axis_offset(tp, last_axis_offset)
            with tik_inst.if_scope(tp.tail_blocks != 0):
                self._copy_in_tail_s3(tp, ub_input, last_axis_offset)
                self._copy_out_tail_s3(tp, ub_input, last_axis_offset)
            self._update_tuple(tp.trans_axis_num, tp.rt_tuple, tp.dst_jump_factor)

    # -------------------------------------------------------------------------------------------------
    #                                    scenario_4
    # -------------------------------------------------------------------------------------------------
    def _update_tuple_major_s4(self, axis_num, rt_tuple, step, logic_tuple, jump_factor, axis_perm):
        tik_inst = self.tik_inst
        with tik_inst.if_scope(axis_num == 2):

            with tik_inst.if_scope(axis_perm == 0x10):
                with tik_inst.if_scope((rt_tuple[0] + 1) % step == 0):
                    rt_tuple[0].set_as(logic_tuple * step)
                    rt_tuple[1].set_as(rt_tuple[1] + 1)
                with tik_inst.else_scope():
                    rt_tuple[0].set_as(rt_tuple[0] + 1)

            with tik_inst.if_scope(axis_perm == 0x01):
                with tik_inst.if_scope((rt_tuple[0] + 1) == jump_factor[0]):
                    rt_tuple[0].set_as(0)
                    rt_tuple[1].set_as(rt_tuple[1] + 1)
                with tik_inst.else_scope():
                    rt_tuple[0].set_as(rt_tuple[0] + 1)

        with tik_inst.if_scope(axis_num == 1):
            rt_tuple[0].set_as(rt_tuple[0] + 1)

    def _update_tuple_tail_s4(self, axis_num, rt_tuple, step, logic_tuple, jump_factor, axis_perm):
        tik_inst = self.tik_inst
        with self.tik_inst.if_scope(axis_num == 2):

            with tik_inst.if_scope(axis_perm == 0x10):
                with self.tik_inst.if_scope((rt_tuple[0] + 1) == jump_factor[0]):
                    rt_tuple[0].set_as((logic_tuple + 1) * step)
                    rt_tuple[1].set_as(rt_tuple[1] + 1)
                with self.tik_inst.else_scope():
                    rt_tuple[0].set_as(rt_tuple[0] + 1)

            with tik_inst.if_scope(axis_perm == 0x01):
                with self.tik_inst.if_scope((rt_tuple[0] + 1) == jump_factor[0]):
                    rt_tuple[0].set_as(0)
                    rt_tuple[1].set_as(rt_tuple[1] + 1)
                with self.tik_inst.else_scope():
                    rt_tuple[0].set_as(rt_tuple[0] + 1)

        with self.tik_inst.if_scope(axis_num == 1):
            rt_tuple[0].set_as(rt_tuple[0] + 1)

    def _init_major_src_tuple_copy_out_s4(self, tp):
        tik_inst = self.tik_inst
        with self.tik_inst.if_scope(tp.src_axis_num_no_dup == 2):

            with tik_inst.if_scope(tp.src_axis_perm == 0x10):
                tp.rt_src_tuple_out[0].set_as(tp.rt_logic_tuple[0] * tp.src_jump_major_step)
                tp.rt_src_tuple_out[1].set_as(0)

            with tik_inst.if_scope(tp.src_axis_perm == 0x01):
                tp.rt_src_tuple_out[0].set_as(0)
                tp.rt_src_tuple_out[1].set_as(tp.rt_logic_tuple[0] * tp.src_jump_major_step)

        with self.tik_inst.if_scope(tp.src_axis_num_no_dup == 1):

            with tik_inst.if_scope(tp.src_axis_perm == 0x0):
                tp.rt_src_tuple_out[0].set_as(tp.rt_logic_tuple[0] * tp.src_jump_major_step)

            with tik_inst.if_scope(tp.src_axis_perm == 0x01):
                tp.rt_src_tuple_out[0].set_as(tp.rt_logic_tuple[0] * tp.src_jump_major_step)

            with tik_inst.if_scope(tp.src_axis_perm == 0x10):
                tp.rt_src_tuple_out[0].set_as(0)

    def _init_tail_src_tuple_copy_out_s4(self, tp):
        tik_inst = self.tik_inst
        with self.tik_inst.if_scope(tp.src_axis_num_no_dup == 2):

            with tik_inst.if_scope(tp.src_axis_perm == 0x10):
                tp.rt_src_tuple_out[0].set_as((tp.rt_logic_tuple[0] + 1) * tp.src_jump_major_step)
                tp.rt_src_tuple_out[1].set_as(0)

            with tik_inst.if_scope(tp.src_axis_perm == 0x01):
                tp.rt_src_tuple_out[0].set_as(0)
                tp.rt_src_tuple_out[1].set_as((tp.rt_logic_tuple[0] + 1) * tp.src_jump_major_step)

        with self.tik_inst.if_scope(tp.src_axis_num_no_dup == 1):
            with self.tik_inst.if_scope(tp.pivot_src_axis_dup == 1):
                tp.rt_src_tuple_out[0].set_as(0)
            with self.tik_inst.if_scope(tp.pivot_src_axis_dup == 0):
                tp.rt_src_tuple_out[0].set_as((tp.rt_logic_tuple[0] + 1) * tp.src_jump_major_step)

    def _init_major_dst_tuple_copy_in_s4(self, tp):
        tik_inst = self.tik_inst
        with tik_inst.if_scope(tp.dst_axis_num_no_dup == 2):

            with tik_inst.if_scope(tp.dst_axis_perm == 0x10):
                tp.rt_dst_tuple_in[0].set_as(tp.rt_logic_tuple[1] * tp.dst_jump_major_step)
                tp.rt_dst_tuple_in[1].set_as(0)

            with tik_inst.if_scope(tp.dst_axis_perm == 0x01):
                tp.rt_dst_tuple_in[0].set_as(0)
                tp.rt_dst_tuple_in[1].set_as(tp.rt_logic_tuple[1] * tp.dst_jump_major_step)

        with tik_inst.if_scope(tp.dst_axis_num_no_dup == 1):
            tp.rt_dst_tuple_in[0].set_as(tp.rt_logic_tuple[1] * tp.dst_jump_major_step)

    def _init_tail_dst_tuple_copy_in_s4(self, tp):
        tik_inst = self.tik_inst
        with self.tik_inst.if_scope(tp.dst_axis_num_no_dup == 2):

            with tik_inst.if_scope(tp.dst_axis_perm == 0x10):
                tp.rt_dst_tuple_in[0].set_as((tp.rt_logic_tuple[1] + 1) * tp.dst_jump_major_step)
                tp.rt_dst_tuple_in[1].set_as(0)

            with tik_inst.if_scope(tp.dst_axis_perm == 0x01):
                tp.rt_dst_tuple_in[0].set_as(0)
                tp.rt_dst_tuple_in[1].set_as((tp.rt_logic_tuple[1] + 1) * tp.dst_jump_major_step)

        with self.tik_inst.if_scope(tp.dst_axis_num_no_dup == 1):
            with self.tik_inst.if_scope(tp.pivot_dst_axis_dup == 1):
                tp.rt_dst_tuple_in[0].set_as(0)
            with self.tik_inst.if_scope(tp.pivot_dst_axis_dup == 0):
                tp.rt_dst_tuple_in[0].set_as((tp.rt_logic_tuple[1] + 1) * tp.dst_jump_major_step)

    @staticmethod
    def _init_major_src_tuple_copy_out_dup_0x210_s4(tp):
        tp.rt_src_tuple_out[0].set_as(0)

    @staticmethod
    def _init_major_dst_tuple_copy_in_dup_0x210_s4(tp):
        tp.rt_dst_tuple_in[0].set_as(0)

    @staticmethod
    def _init_tail_src_tuple_copy_out_dup_0x210_s4(tp):
        tp.rt_src_tuple_out[0].set_as(0)

    @staticmethod
    def _init_tail_dst_tuple_copy_in_dup_0x210_s4(tp):
        tp.rt_dst_tuple_in[0].set_as(0)

    @staticmethod
    def _init_major_src_tuple_copy_out_dup_0x201_s4(tp):
        tp.rt_src_tuple_out[0].set_as(0)

    @staticmethod
    def _init_major_dst_tuple_copy_in_dup_0x201_s4(tp):
        tp.rt_dst_tuple_in[0].set_as(tp.rt_logic_tuple[1] * tp.dst_jump_major_step)

    @staticmethod
    def _init_tail_dst_tuple_copy_in_dup_0x201_s4(tp):
        tp.rt_dst_tuple_in[0].set_as((tp.rt_logic_tuple[1] + 1) * tp.dst_jump_major_step)

    @staticmethod
    def _init_major_src_tuple_copy_out_dup_0x120_s4(tp):
        tp.rt_src_tuple_out[0].set_as(tp.rt_logic_tuple[0] * tp.src_jump_major_step)

    @staticmethod
    def _init_major_dst_tuple_copy_in_dup_0x120_s4(tp):
        tp.rt_dst_tuple_in[0].set_as(0)

    @staticmethod
    def _init_tail_src_tuple_copy_out_dup_0x120_s4(tp):
        tp.rt_src_tuple_out[0].set_as((tp.rt_logic_tuple[0] + 1) * tp.src_jump_major_step)

    @staticmethod
    def _init_major_dst_tuple_copy_in_dup_0x10_s1d2_s4(tp):
        tp.rt_dst_tuple_in[0].set_as(0)

    @staticmethod
    def _init_major_src_tuple_copy_out_dup_0x10_s2d1_s4(tp):
        tp.rt_src_tuple_out[0].set_as(0)

    @staticmethod
    def _init_major_tuple_copy_in_dup_s4(tp):
        tp.rt_dst_tuple_in[0].set_as(tp.init_dst_tuple_in[0])

    def _init_major_tuple_copy_out_dup_s4(self, tp):
        tik_inst = self.tik_inst

        with self.tik_inst.if_scope(tp.src_axis_perm == 0x0):
            tp.rt_src_tuple_out[0].set_as(tp.rt_logic_tuple[0] * tp.src_jump_major_step)

        with self.tik_inst.if_scope(tp.src_axis_perm == 0x01):
            tp.rt_src_tuple_out[0].set_as(tp.rt_logic_tuple[0] * tp.src_jump_major_step)

        with self.tik_inst.if_scope(tp.src_axis_perm == 0x10):
            tp.rt_src_tuple_out[0].set_as(0)

    @staticmethod
    def _init_tail_tuple_copy_in_dup_s4(tp):
        # mul round % axis  as tail
        tp.rt_dst_tuple_in[0].set_as(tp.init_dst_tuple_in[0])

    def _init_tail_tuple_copy_out_dup_s4(self, tp):
        with self.tik_inst.if_scope(tp.src_axis_perm == 0x01):
            tp.rt_src_tuple_out[0].set_as((tp.rt_logic_tuple[0] + 1) * tp.src_jump_major_step)

        with self.tik_inst.if_scope(tik.any(tp.src_axis_perm == 0x10, tp.src_axis_perm == 0x0)):
            tp.rt_src_tuple_out[0].set_as(tp.init_src_tuple_out[0])

    @staticmethod
    def _init_logic_tuple_s4(tp):
        for i in range(TRANSPOSE_MAX_AXIS_NUM):
            tp.rt_logic_tuple[i].set_as(tp.init_logic_tuple[i])

    def _update_major_dst_tuple_in_s4(self, tp):
        self._update_tuple_major_s4(tp.dst_axis_num_no_dup,
                                    tp.rt_dst_tuple_in,
                                    tp.dst_jump_major_step,
                                    tp.rt_logic_tuple[1],
                                    tp.dst_jump_factor_in,
                                    tp.dst_axis_perm)

    def _update_major_src_tuple_out_s4(self, tp):
        self._update_tuple_major_s4(tp.src_axis_num_no_dup,
                                    tp.rt_src_tuple_out,
                                    tp.src_jump_major_step,
                                    tp.rt_logic_tuple[0],
                                    tp.src_jump_factor_out,
                                    tp.src_axis_perm)

    def _update_tail_dst_tuple_in_s4(self, tp):
        self._update_tuple_tail_s4(tp.dst_axis_num_no_dup,
                                   tp.rt_dst_tuple_in,
                                   tp.dst_jump_major_step,
                                   tp.rt_logic_tuple[1],
                                   tp.dst_jump_factor_in,
                                   tp.dst_axis_perm)

    def _update_tail_src_tuple_out_s4(self, tp):
        self._update_tuple_tail_s4(tp.src_axis_num_no_dup,
                                   tp.rt_src_tuple_out,
                                   tp.src_jump_major_step,
                                   tp.rt_logic_tuple[0],
                                   tp.src_jump_factor_out,
                                   tp.src_axis_perm)

    @staticmethod
    def _update_tail_tuple_in_dup_s4(tp):
        tp.rt_dst_tuple_in[0].set_as(tp.rt_dst_tuple_in[0] + 1)

    @staticmethod
    def _update_tail_tuple_out_dup_s4(tp):
        tp.rt_src_tuple_out[0].set_as(tp.rt_src_tuple_out[0] + 1)

    def _update_logic_tuple_s4(self, tp):
        self._update_tuple(tp.logic_axis_num, tp.rt_logic_tuple, tp.logic_jump_factor)

    def _detect_tail_flag(self, tp, is_src_tail_in, is_dst_tail_in):
        with self.tik_inst.if_scope(tp.rt_logic_tuple[0] == tp.logic_jump_factor[0] - 1):
            is_src_tail_in.set_as(1)

        with self.tik_inst.if_scope(tp.rt_logic_tuple[1] == tp.logic_jump_factor[1] - 1):
            is_dst_tail_in.set_as(1)

    def _get_src_addr_s4(self, tp, src_addr, is_tail):
        tik_inst = self.tik_inst

        src_addr.set_as(0)

        with tik_inst.if_scope(is_tail == 0):
            src_addr.set_as(tp.rt_logic_tuple[0] * tp.major_in_ele)
        with tik_inst.else_scope():
            src_addr.set_as((tp.rt_logic_tuple[0] + 1) * tp.major_in_ele)

        with tik_inst.for_range(0, tp.dst_axis_num_no_dup) as i:
            src_addr.set_as(src_addr + tp.rt_dst_tuple_in[i] * tp.dst_stride_in[i])

        with tik_inst.for_range(2, tp.other_axis_num + 2) as i:
            src_addr.set_as(src_addr + tp.rt_logic_tuple[i] * tp.logic_stride_in[i])

    def _get_dst_addr_s4(self, tp, dst_addr, is_tail):
        tik_inst = self.tik_inst
        dst_addr.set_as(0)

        with tik_inst.if_scope(tp.dup_axis == 0):
            with tik_inst.if_scope(is_tail == 0):
                dst_addr.set_as(tp.rt_logic_tuple[1] * tp.major_out_ele)
            with tik_inst.else_scope():
                dst_addr.set_as((tp.rt_logic_tuple[1] + 1) * tp.major_out_ele)
        with tik_inst.else_scope():
            with tik_inst.if_scope(tik.any(tp.src_axis_perm == 0x10, tp.src_axis_perm == 0x0)):
                with tik_inst.if_scope(tp.pivot_src_axis_dup == 0):
                    with tik_inst.if_scope(is_tail == 0):
                        dst_addr.set_as(tp.rt_logic_tuple[1] * tp.major_out_ele)
                    with tik_inst.else_scope():
                        dst_addr.set_as((tp.rt_logic_tuple[1] + 1) * tp.major_out_ele)
                with tik_inst.if_scope(tp.pivot_src_axis_dup == 1):
                    with tik_inst.if_scope(is_tail == 0):
                        dst_addr.set_as(tp.rt_logic_tuple[0] * tp.major_out_ele)
                    with tik_inst.else_scope():
                        dst_addr.set_as((tp.rt_logic_tuple[0] + 1) * tp.major_out_ele)

        with tik_inst.for_range(0, tp.src_axis_num_no_dup) as i:
            dst_addr.set_as(dst_addr + tp.rt_src_tuple_out[i] * tp.src_stride_out[i])

        with tik_inst.for_range(2, tp.other_axis_num + 2) as i:
            dst_addr.set_as(dst_addr + tp.rt_logic_tuple[i] * tp.logic_stride_out[i])

    def _get_src_addr_dup_0x210_s4(self, tp, src_addr, is_tail):
        tik_inst = self.tik_inst
        src_addr.set_as(0)

        with tik_inst.if_scope(is_tail == 0):
            src_addr.set_as(tp.rt_logic_tuple[0] * tp.major_in_ele)
        with tik_inst.else_scope():
            src_addr.set_as((tp.rt_logic_tuple[0] + 1) * tp.major_in_ele)

        with tik_inst.for_range(0, tp.dst_axis_num_no_dup) as i:
            src_addr.set_as(src_addr + tp.rt_dst_tuple_in[i] * tp.dst_stride_in[i])

        with tik_inst.for_range(2, tp.other_axis_num + 2) as i:
            src_addr.set_as(src_addr + tp.rt_logic_tuple[i] * tp.logic_stride_in[i])

    def _get_dst_addr_dup_0x210_s4(self, tp, dst_addr, is_tail):
        tik_inst = self.tik_inst
        dst_addr.set_as(0)

        with tik_inst.if_scope(is_tail == 0):
            dst_addr.set_as(tp.rt_logic_tuple[0] * tp.major_out_ele)
        with tik_inst.else_scope():
            dst_addr.set_as((tp.rt_logic_tuple[0] + 1) * tp.major_out_ele)

        with tik_inst.for_range(0, tp.src_axis_num_no_dup) as i:
            dst_addr.set_as(dst_addr + tp.rt_src_tuple_out[i] * tp.src_stride_out[i])

        with tik_inst.for_range(2, tp.other_axis_num + 2) as i:
            dst_addr.set_as(dst_addr + tp.rt_logic_tuple[i] * tp.logic_stride_out[i])

    def _get_src_addr_dup_0x201_s4(self, tp, src_addr):
        tik_inst = self.tik_inst
        src_addr.set_as(0)

        with tik_inst.for_range(0, tp.dst_axis_num_no_dup) as i:
            src_addr.set_as(src_addr + tp.rt_dst_tuple_in[i] * tp.dst_stride_in[i])

        with tik_inst.for_range(2, tp.other_axis_num + 2) as i:
            src_addr.set_as(src_addr + tp.rt_logic_tuple[i] * tp.logic_stride_in[i])

    def _get_dst_addr_dup_0x201_s4(self, tp, dst_addr, is_tail):
        tik_inst = self.tik_inst
        dst_addr.set_as(0)

        with tik_inst.if_scope(is_tail == 0):
            dst_addr.set_as(tp.rt_logic_tuple[1] * tp.major_out_ele)
        with tik_inst.else_scope():
            dst_addr.set_as((tp.rt_logic_tuple[1] + 1) * tp.major_out_ele)

        with tik_inst.for_range(0, tp.src_axis_num_no_dup) as i:
            dst_addr.set_as(dst_addr + tp.rt_src_tuple_out[i] * tp.src_stride_out[i])

        with tik_inst.for_range(2, tp.other_axis_num + 2) as i:
            dst_addr.set_as(dst_addr + tp.rt_logic_tuple[i] * tp.logic_stride_out[i])

    def _get_src_addr_dup_0x10_s1d2_s4(self, tp, src_addr, is_tail):
        tik_inst = self.tik_inst
        src_addr.set_as(0)

        with tik_inst.if_scope(is_tail == 0):
            src_addr.set_as(tp.rt_logic_tuple[0] * tp.major_in_ele)
        with tik_inst.else_scope():
            src_addr.set_as((tp.rt_logic_tuple[0] + 1) * tp.major_in_ele)

        with tik_inst.for_range(0, tp.dst_axis_num_no_dup) as i:
            src_addr.set_as(src_addr + tp.rt_dst_tuple_in[i] * tp.dst_stride_in[i])

        with tik_inst.for_range(2, tp.other_axis_num + 2) as i:
            src_addr.set_as(src_addr + tp.rt_logic_tuple[i] * tp.logic_stride_in[i])

    def _get_dst_addr_dup_0x10_s1d2_s4(self, tp, dst_addr, is_tail):
        tik_inst = self.tik_inst
        dst_addr.set_as(0)

        with tik_inst.if_scope(is_tail == 0):
            dst_addr.set_as(tp.rt_logic_tuple[0] * tp.major_out_ele)
        with tik_inst.else_scope():
            dst_addr.set_as((tp.rt_logic_tuple[0] + 1) * tp.major_out_ele)

        with tik_inst.for_range(2, tp.other_axis_num + 2) as i:
            dst_addr.set_as(dst_addr + tp.rt_logic_tuple[i] * tp.logic_stride_out[i])

    def _get_src_addr_dup_0x10_s2d1_s4(self, tp, src_addr, is_tail):
        tik_inst = self.tik_inst
        src_addr.set_as(0)

        with tik_inst.if_scope(is_tail == 0):
            src_addr.set_as(tp.rt_logic_tuple[0] * tp.major_in_ele)
        with tik_inst.else_scope():
            src_addr.set_as((tp.rt_logic_tuple[0] + 1) * tp.major_in_ele)

        with tik_inst.for_range(2, tp.other_axis_num + 2) as i:
            src_addr.set_as(src_addr + tp.rt_logic_tuple[i] * tp.logic_stride_in[i])

    def _get_dst_addr_dup_0x10_s2d1_s4(self, tp, dst_addr, is_tail):
        tik_inst = self.tik_inst
        dst_addr.set_as(0)

        with tik_inst.if_scope(is_tail == 0):
            dst_addr.set_as(tp.rt_logic_tuple[0] * tp.major_out_ele)
        with tik_inst.else_scope():
            dst_addr.set_as((tp.rt_logic_tuple[0] + 1) * tp.major_out_ele)

        with tik_inst.for_range(0, tp.src_axis_num_no_dup) as i:
            dst_addr.set_as(dst_addr + tp.rt_src_tuple_out[i] * tp.src_stride_out[i])

        with tik_inst.for_range(2, tp.other_axis_num + 2) as i:
            dst_addr.set_as(dst_addr + tp.rt_logic_tuple[i] * tp.logic_stride_out[i])

    def _copy_out_common_s4(self, tp, ub_input, loop, burst_len, x_out_ele, x_out_tail_ele):
        tik_inst = self.tik_inst
        ub_tail_offset = tik_inst.Scalar("int32")
        ub_tail_offset.set_as(loop % 32 * self.ele_per_block)
        with tik_inst.if_scope(tik.any(tp.align_ele == 0, x_out_tail_ele == 0)):
            tik_inst.data_move(self.data_out[tp.dst_addr],
                               ub_input[tp.ub_res_addr + loop * burst_len * EPB16 // self.fp16_times],
                               0, 1, burst_len, 0, 0)
        with tik_inst.else_scope():
            tik_inst.data_move(self.data_out[tp.dst_addr],
                               ub_input[tp.ub_res_addr + loop * burst_len * EPB16 // self.fp16_times],
                               0, 1, burst_len - 1, 0, 0)
            ub_input_tail = self.ub_input_b64_helper.reinterpret_cast_to(self.x_dtype)
            scalar_value = self.tik_inst.Scalar(self.x_dtype)
            with self.tik_inst.for_range(0, self.ele_per_block) as i:
                scalar_value.set_as(ub_input[tp.ub_res_addr + i +\
                                    loop * burst_len * EPB16 // self.fp16_times +\
                                    x_out_ele -\
                                    self.ele_per_block])
                ub_input_tail[ub_tail_offset + i] = scalar_value
            tik_inst.data_move(self.data_out[tp.dst_addr + x_out_ele - self.ele_per_block],
                               ub_input_tail[ub_tail_offset], 0, 1, 1, 0, 0)

    def _copy_in_major_src_major_dst_s4(self, tp, ub_input):
        tik_inst = self.tik_inst
        tp.ub_offset.set_as(0)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.major_dst_loop_in) as i:
                self._get_src_addr_s4(tp, tp.src_addr, 0)
                tik_inst.data_move(ub_input[i * tp.major_burst_len_in * EPB16 // self.fp16_times],
                                   self.data_in[tp.src_addr],
                                   0,
                                   1,
                                   tp.major_burst_len_in,
                                   0,
                                   0)
                self._update_major_dst_tuple_in_s4(tp)
                tp.ub_offset.set_as(tp.ub_offset + tp.major_burst_len_in)

    def _copy_out_major_src_major_dst_s4(self, tp, ub_input):
        tik_inst = self.tik_inst
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.major_src_loop_out) as i:
                self._get_dst_addr_s4(tp, tp.dst_addr, 0)
                self._copy_out_common_s4(tp, ub_input, i, tp.major_burst_len_out,
                                         tp.major_out_ele, tp.major_out_tail_ele)
                self._update_major_src_tuple_out_s4(tp)

    def _copy_in_tail_src_major_dst_s4(self, tp, ub_input):
        tik_inst = self.tik_inst
        tp.ub_offset.set_as(0)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.major_dst_loop_in) as i:
                self._get_src_addr_s4(tp, tp.src_addr, 1)
                tik_inst.data_move(ub_input[i * tp.tail_burst_len_in * EPB16 // self.fp16_times],
                                   self.data_in[tp.src_addr], 0, 1, tp.tail_burst_len_in, 0, 0)
                self._update_major_dst_tuple_in_s4(tp)
                tp.ub_offset.set_as(tp.ub_offset + tp.tail_burst_len_in)

    def _copy_out_tail_src_major_dst_s4(self, tp, ub_input):
        tik_inst = self.tik_inst
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.tail_src_loop_out) as i:
                self._get_dst_addr_s4(tp, tp.dst_addr, 0)
                self._copy_out_common_s4(tp, ub_input, i, tp.major_burst_len_out,
                                         tp.major_out_ele, tp.major_out_tail_ele)
                self._update_tail_src_tuple_out_s4(tp)

    def _copy_in_major_src_tail_dst_s4(self, tp, ub_input):
        tik_inst = self.tik_inst
        tp.ub_offset.set_as(0)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.tail_dst_loop_in) as i:
                self._get_src_addr_s4(tp, tp.src_addr, 0)
                tik_inst.data_move(ub_input[i * tp.major_burst_len_in * EPB16 // self.fp16_times],
                                   self.data_in[tp.src_addr], 0, 1, tp.major_burst_len_in, 0, 0)
                self._update_tail_dst_tuple_in_s4(tp)
                tp.ub_offset.set_as(tp.ub_offset + tp.major_burst_len_in)

    def _copy_out_major_src_tail_dst_s4(self, tp, ub_input):
        tik_inst = self.tik_inst
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.major_src_loop_out) as i:
                self._get_dst_addr_s4(tp, tp.dst_addr, 1)
                self._copy_out_common_s4(tp, ub_input, i, tp.tail_burst_len_out, tp.tail_out_ele, tp.tail_out_tail_ele)
                self._update_major_src_tuple_out_s4(tp)

    def _copy_in_tail_src_tail_dst_s4(self, tp, ub_input):
        tik_inst = self.tik_inst
        tp.ub_offset.set_as(0)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.tail_dst_loop_in) as i:
                self._get_src_addr_s4(tp, tp.src_addr, 1)
                tik_inst.data_move(ub_input[i * tp.tail_burst_len_in * EPB16 // self.fp16_times],
                                   self.data_in[tp.src_addr], 0, 1, tp.tail_burst_len_in, 0, 0)
                self._update_tail_dst_tuple_in_s4(tp)
                tp.ub_offset.set_as(tp.ub_offset + tp.tail_burst_len_in)

    def _copy_out_tail_src_tail_dst_s4(self, tp, ub_input):
        tik_inst = self.tik_inst
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.tail_src_loop_out) as i:
                self._get_dst_addr_s4(tp, tp.dst_addr, 1)
                self._copy_out_common_s4(tp, ub_input, i, tp.tail_burst_len_out, tp.tail_out_ele, tp.tail_out_tail_ele)
                self._update_tail_src_tuple_out_s4(tp)

    def _copy_in_major_src_major_dst_dup_0x210_s4(self, tp, ub_input):
        tik_inst = self.tik_inst
        tp.ub_offset.set_as(0)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.major_dst_loop_in) as i:
                self._get_src_addr_dup_0x210_s4(tp, tp.src_addr, 0)
                tik_inst.data_move(ub_input[i * tp.major_burst_len_in * EPB16 // self.fp16_times],
                                   self.data_in[tp.src_addr],
                                   0, 1, tp.major_burst_len_in, 0, 0)
                self._update_major_dst_tuple_in_s4(tp)
                tp.ub_offset.set_as(tp.ub_offset + tp.major_burst_len_in)

    def _copy_out_major_src_major_dst_dup_0x210_s4(self, tp, ub_input):
        tik_inst = self.tik_inst
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.major_src_loop_out) as i:
                self._get_dst_addr_dup_0x210_s4(tp, tp.dst_addr, 0)
                self._copy_out_common_s4(tp, ub_input, i, tp.major_burst_len_out,
                                         tp.major_out_ele, tp.major_out_tail_ele)
                self._update_tail_src_tuple_out_s4(tp)

    def _copy_in_tail_src_tail_dst_dup_0x210_s4(self, tp, ub_input):
        tik_inst = self.tik_inst
        tp.ub_offset.set_as(0)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.major_dst_loop_in) as i:
                self._get_src_addr_dup_0x210_s4(tp, tp.src_addr, 1)
                tik_inst.data_move(ub_input[i * tp.tail_burst_len_in * EPB16 // self.fp16_times],
                                   self.data_in[tp.src_addr], 0, 1, tp.tail_burst_len_in, 0, 0)
                self._update_tail_dst_tuple_in_s4(tp)
                tp.ub_offset.set_as(tp.ub_offset + tp.tail_burst_len_in)

    def _copy_out_tail_src_tail_dst_dup_0x210_s4(self, tp, ub_input):
        tik_inst = self.tik_inst
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.major_src_loop_out) as i:
                self._get_dst_addr_dup_0x210_s4(tp, tp.dst_addr, 1)
                self._copy_out_common_s4(tp, ub_input, i, tp.tail_burst_len_out, tp.tail_out_ele, tp.tail_out_tail_ele)
                self._update_tail_src_tuple_out_s4(tp)

    def _copy_in_major_src_major_dst_dup_0x201_s4(self, tp, ub_input):
        tik_inst = self.tik_inst
        tp.ub_offset.set_as(0)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.major_dst_loop_in) as i:
                self._get_src_addr_dup_0x201_s4(tp, tp.src_addr)
                tik_inst.data_move(ub_input[i * tp.major_burst_len_in * EPB16 // self.fp16_times],
                                   self.data_in[tp.src_addr],
                                   0, 1, tp.major_burst_len_in, 0, 0)
                self._update_major_dst_tuple_in_s4(tp)
                tp.ub_offset.set_as(tp.ub_offset + tp.major_burst_len_in)

    def _copy_out_major_src_major_dst_dup_0x201_s4(self, tp, ub_input):
        tik_inst = self.tik_inst
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.major_src_loop_out) as i:
                self._get_dst_addr_dup_0x201_s4(tp, tp.dst_addr, 0)
                self._copy_out_common_s4(tp, ub_input, i, tp.major_burst_len_out,
                                         tp.major_out_ele, tp.major_out_tail_ele)
                self._update_tail_src_tuple_out_s4(tp)

    def _copy_in_major_src_tail_dst_dup_0x201_s4(self, tp, ub_input):
        tik_inst = self.tik_inst
        tp.ub_offset.set_as(0)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.tail_dst_loop_in) as i:
                self._get_src_addr_dup_0x201_s4(tp, tp.src_addr)
                tik_inst.data_move(ub_input[i * tp.major_burst_len_in * EPB16 // self.fp16_times],
                                   self.data_in[tp.src_addr], 0, 1, tp.major_burst_len_in, 0, 0)
                self._update_tail_dst_tuple_in_s4(tp)
                tp.ub_offset.set_as(tp.ub_offset + tp.major_burst_len_in)

    def _copy_out_major_src_tail_dst_dup_0x201_s4(self, tp, ub_input):
        tik_inst = self.tik_inst
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.major_src_loop_out) as i:
                self._get_dst_addr_dup_0x201_s4(tp, tp.dst_addr, 1)
                self._copy_out_common_s4(tp, ub_input, i, tp.tail_burst_len_out, tp.tail_out_ele, tp.tail_out_tail_ele)
                self._update_tail_src_tuple_out_s4(tp)

    def _copy_in_major_src_major_dst_dup_0x10_s1d2_s4(self, tp, ub_input):
        tik_inst = self.tik_inst
        tp.ub_offset.set_as(0)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.major_dst_loop_in) as i:
                self._get_src_addr_dup_0x10_s1d2_s4(tp, tp.src_addr, 0)
                tik_inst.data_move(ub_input[i * tp.major_burst_len_in * EPB16 // self.fp16_times],
                                   self.data_in[tp.src_addr],
                                   0, 1, tp.major_burst_len_in, 0, 0)
                self._update_major_dst_tuple_in_s4(tp)
                tp.ub_offset.set_as(tp.ub_offset + tp.major_burst_len_in)

    def _copy_out_major_src_major_dst_dup_0x10_s1d2_s4(self, tp, ub_input):
        tik_inst = self.tik_inst
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            self._get_dst_addr_dup_0x10_s1d2_s4(tp, tp.dst_addr, 0)
            self._copy_out_common_s4(tp, ub_input, 0, tp.major_burst_len_out, tp.major_out_ele, tp.major_out_tail_ele)
            self._update_tail_src_tuple_out_s4(tp)

    def _copy_in_tail_src_major_dst_dup_0x10_s1d2_s4(self, tp, ub_input):
        tik_inst = self.tik_inst
        tp.ub_offset.set_as(0)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.major_dst_loop_in) as i:
                self._get_src_addr_dup_0x10_s1d2_s4(tp, tp.src_addr, 1)
                tik_inst.data_move(ub_input[i * tp.tail_burst_len_in * EPB16 // self.fp16_times],
                                   self.data_in[tp.src_addr],
                                   0, 1, tp.tail_burst_len_in, 0, 0)
                self._update_major_dst_tuple_in_s4(tp)
                tp.ub_offset.set_as(tp.ub_offset + tp.tail_burst_len_in)

    def _copy_out_tail_src_major_dst_dup_0x10_s1d2_s4(self, tp, ub_input):
        tik_inst = self.tik_inst
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            self._get_dst_addr_dup_0x10_s1d2_s4(tp, tp.dst_addr, 1)
            self._copy_out_common_s4(tp, ub_input, 0, tp.tail_burst_len_out, tp.tail_out_ele, tp.tail_out_tail_ele)
            self._update_tail_src_tuple_out_s4(tp)

    def _copy_in_major_src_major_dst_dup_0x10_s2d1_s4(self, tp, ub_input):
        tik_inst = self.tik_inst
        tp.ub_offset.set_as(0)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            self._get_src_addr_dup_0x10_s2d1_s4(tp, tp.src_addr, 0)
            tik_inst.data_move(ub_input[0],
                               self.data_in[tp.src_addr],
                               0, 1, tp.major_burst_len_in, 0, 0)
            self._update_major_dst_tuple_in_s4(tp)
            tp.ub_offset.set_as(tp.ub_offset + tp.major_burst_len_in)

    def _copy_out_major_src_major_dst_dup_0x10_s2d1_s4(self, tp, ub_input):
        tik_inst = self.tik_inst
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.major_src_loop_out) as i:
                self._get_dst_addr_dup_0x10_s2d1_s4(tp, tp.dst_addr, 0)
                self._copy_out_common_s4(tp, ub_input, i, tp.major_burst_len_out,
                                         tp.major_out_ele, tp.major_out_tail_ele)
                self._update_tail_src_tuple_out_s4(tp)

    def _copy_in_tail_src_major_dst_dup_0x10_s2d1_s4(self, tp, ub_input):
        tik_inst = self.tik_inst
        tp.ub_offset.set_as(0)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            self._get_src_addr_dup_0x10_s2d1_s4(tp, tp.src_addr, 1)
            tik_inst.data_move(ub_input[0],
                               self.data_in[tp.src_addr],
                               0, 1, tp.tail_burst_len_in, 0, 0)
            self._update_major_dst_tuple_in_s4(tp)
            tp.ub_offset.set_as(tp.ub_offset + tp.tail_burst_len_in)

    def _copy_out_tail_src_major_dst_dup_0x10_s2d1_s4(self, tp, ub_input):
        tik_inst = self.tik_inst
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.major_src_loop_out) as i:
                self._get_dst_addr_dup_0x10_s2d1_s4(tp, tp.dst_addr, 1)
                self._copy_out_common_s4(tp, ub_input, i, tp.tail_burst_len_out, tp.tail_out_ele, tp.tail_out_tail_ele)
                self._update_tail_src_tuple_out_s4(tp)

    def _copy_in_tail_dup_s4(self, tp, ub_input):
        tik_inst = self.tik_inst
        tp.ub_offset.set_as(0)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.major_dst_loop_in) as i:
                self._get_src_addr_s4(tp, tp.src_addr, 1)
                tik_inst.data_move(ub_input[i * tp.tail_burst_len_in * EPB16 // self.fp16_times],
                                   self.data_in[tp.src_addr],
                                   0,
                                   1,
                                   tp.tail_burst_len_in,
                                   0,
                                   0)
                self._update_tail_tuple_in_dup_s4(tp)
                tp.ub_offset.set_as(tp.ub_offset + tp.tail_burst_len_in)

    def _copy_out_tail_dup_0x10_s1d2_s4(self, tp, ub_input):
        tik_inst = self.tik_inst
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.major_src_loop_out) as i:
                self._get_dst_addr_s4(tp, tp.dst_addr, 1)
                self._copy_out_common_s4(tp, ub_input, i, tp.tail_burst_len_out, tp.tail_out_ele, tp.tail_out_tail_ele)
                self._update_tail_tuple_out_dup_s4(tp)

    def _copy_out_tail_dup_0x01_s4(self, tp, ub_input):
        tik_inst = self.tik_inst
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.tail_src_loop_out) as i:
                self._get_dst_addr_s4(tp, tp.dst_addr, 1)
                self._copy_out_common_s4(tp, ub_input, i, tp.major_burst_len_out,
                                         tp.major_out_ele, tp.major_out_tail_ele)
                self._update_tail_tuple_out_dup_s4(tp)

    def _copy_out_tail_dup_0x0_s4(self, tp, ub_input):
        tik_inst = self.tik_inst
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.major_src_loop_out) as i:
                self._get_dst_addr_s4(tp, tp.dst_addr, 1)
                self._copy_out_common_s4(tp, ub_input, i, tp.tail_burst_len_out, tp.tail_out_ele, tp.tail_out_tail_ele)
                self._update_tail_tuple_out_dup_s4(tp)

    @staticmethod
    def _swap(tp):
        tp.offset_t.set_as(tp.offset_a)
        tp.offset_a.set_as(tp.offset_b)
        tp.offset_b.set_as(tp.offset_t)

    def _make_ualigned_be_head_of_block(self, tp, ub_input, x_in_ele, x_in_tail_ele, x_dst_loop_in):
        tik_inst = self.tik_inst
        with tik_inst.if_scope(tp.align_ele != 0):
            ub_input_b16 = ub_input.reinterpret_cast_to("int16")
            src_ele_num_in_b16 = self._get_src_size()  # avoid bank conflict
            src_list = [ub_input_b16[tp.offset_a * self.fp16_times + src_ele_num_in_b16 * i] for i in range(EPB16)]
            dst_list = [ub_input_b16[tp.offset_b * self.fp16_times + EPB16 * i] for i in range(EPB16)]
            with tik_inst.if_scope(tp.ub_offset != 1):
                tik_inst.vnchwconv(False, False, dst_list, src_list, tp.ub_offset, EPB16, 1)
            with tik_inst.else_scope():
                tik_inst.vnchwconv(False, False, dst_list, src_list, tp.ub_offset, 0, 0)
            self._swap(tp)

            #eliminate dirty data between two in_blocks
            with tik_inst.if_scope(x_in_tail_ele != 0):
                tik_inst.data_move(ub_input_b16[tp.offset_b * self.fp16_times],
                                   ub_input_b16[tp.offset_a * self.fp16_times],
                                   0,
                                   x_dst_loop_in,
                                   x_in_ele * self.fp16_times,
                                   (self.ele_per_block - x_in_tail_ele) * self.fp16_times, # src_stride
                                   0) # dst_stride
                self._swap(tp)

    def _make_block_head_be_contiguous(self, tp, ub_input, x_out_ele, x_out_tail_ele, burst_len_out, x_src_loop_out):
        tik_inst = self.tik_inst
        with tik_inst.if_scope(tp.align_ele != 0):
            ub_input_b16 = ub_input.reinterpret_cast_to("int16")
            # insert data between two in_blocks to make each out_blocks be started with block align
            with tik_inst.if_scope(x_out_tail_ele != 0):
                tik_inst.data_move(ub_input_b16[tp.offset_b * self.fp16_times],
                                   ub_input_b16[tp.offset_a * self.fp16_times],
                                   0,
                                   x_src_loop_out,
                                   x_out_ele * self.fp16_times,
                                   0,
                                   (self.ele_per_block - x_out_tail_ele) * self.fp16_times)
                self._swap(tp)

            # make block head be line
            src_list = [ub_input_b16[tp.offset_a * self.fp16_times + EPB16 * i] for i in range(EPB16)]
            dst_list = [ub_input_b16[tp.offset_b * self.fp16_times + self._get_dst_size() * EPB16 * i] \
                        for i in range(EPB16)]
            with tik_inst.if_scope(x_src_loop_out * burst_len_out != 1):
                tik_inst.vnchwconv(False, False, dst_list, src_list, x_src_loop_out * burst_len_out, 1, EPB16)
            with tik_inst.else_scope():
                tik_inst.vnchwconv(False, False, dst_list, src_list, x_src_loop_out * burst_len_out, 0, 0)
            self._swap(tp)

    def _get_reorder_idx(self, is_src_tail_in, is_dst_tail_in, idx):
        tik_inst = self.tik_inst
        with tik_inst.if_scope(tik.all(is_dst_tail_in == 0, is_src_tail_in == 0)):
            idx.set_as(0)
        with tik_inst.if_scope(tik.all(is_dst_tail_in == 0, is_src_tail_in == 1)):
            idx.set_as(1)
        with tik_inst.if_scope(tik.all(is_dst_tail_in == 1, is_src_tail_in == 0)):
            idx.set_as(2)
        with tik_inst.if_scope(tik.all(is_dst_tail_in == 1, is_src_tail_in == 1)):
            idx.set_as(3)

    def _reorder_s4_data_move(self, tp, ub_input, idx):
        tik_inst = self.tik_inst
        ub_input_b16 = ub_input.reinterpret_cast_to("int16")
        with tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.loop_1[idx]) as i:
                tik_inst.data_move(ub_input_b16[(tp.offset_b + i * tp.dst_offset_1[idx]) * self.fp16_times],
                                   ub_input_b16[(tp.offset_a + i * tp.src_offset_1[idx]) * self.fp16_times],
                                   0,
                                   tp.repeat_1[idx],
                                   tp.burst_len_1[idx],
                                   tp.src_stride_1[idx],
                                   tp.dst_stride_1[idx])
        with tik_inst.if_scope(tp.loop_1[idx] > 0):
            self._swap(tp)

        with tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.loop_2[idx]) as i:
                tik_inst.data_move(ub_input_b16[(tp.offset_b + i * tp.dst_offset_2[idx]) * self.fp16_times],
                                   ub_input_b16[(tp.offset_a + i * tp.src_offset_2[idx]) * self.fp16_times],
                                   0,
                                   tp.repeat_2[idx],
                                   tp.burst_len_2[idx],
                                   tp.src_stride_2[idx],
                                   tp.dst_stride_2[idx])
        with tik_inst.if_scope(tp.loop_2[idx] > 0):
            self._swap(tp)

        with tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.loop_3[idx]) as i:
                tik_inst.data_move(ub_input_b16[(tp.offset_b + i * tp.dst_offset_3[idx]) * self.fp16_times],
                                   ub_input_b16[(tp.offset_a + i * tp.src_offset_3[idx]) * self.fp16_times],
                                   0,
                                   tp.repeat_3[idx],
                                   tp.burst_len_3[idx],
                                   tp.src_stride_3[idx],
                                   tp.dst_stride_3[idx])
        with tik_inst.if_scope(tp.loop_3[idx] > 0):
            self._swap(tp)

    def _reorder_s4_vcopy(self, tp, ub_input, idx):
        tik_inst = self.tik_inst
        ub_input_b16 = ub_input.reinterpret_cast_to("int16")
        with tik_inst.for_range(0, 4) as i:
            with tik_inst.for_range(0, 8) as j:
                with tik_inst.new_stmt_scope(disable_sync=True):
                    tik_inst.vcopy(6 * 16, ub_input_b16[tp.offset_b + i * 6 * 16 + j * 10 * 4 * 6 * 16],
                                   ub_input_b16[tp.offset_a + i * 10 * 8 * 6 * 16 + j * 6 * 16], 10, 1, 1, 4 * 6, 8 * 6,
                                   "counter")
        self._swap(tp)

    def _reorder_s4(self, tp, ub_input, is_src_tail_in, is_dst_tail_in, x_in_ele, x_in_tail_ele, burst_len_in,
                    x_dst_loop_in, x_out_ele, x_out_tail_ele, burst_len_out, x_src_loop_out):

        tik_inst = self.tik_inst
        idx = tik_inst.Scalar("int32")

        tp.offset_a.set_as(tp.offset_1 // self.fp16_times)
        tp.offset_b.set_as(tp.offset_2 // self.fp16_times)

        self._get_reorder_idx(is_src_tail_in, is_dst_tail_in, idx)

        self._make_ualigned_be_head_of_block(tp, ub_input, x_in_ele, x_in_tail_ele, x_dst_loop_in)

        self._reorder_s4_data_move(tp, ub_input, idx)

        self._make_block_head_be_contiguous(tp, ub_input, x_out_ele, x_out_tail_ele, burst_len_out, x_src_loop_out)

        tp.ub_res_addr.set_as(tp.offset_a)

    def _init_all_tuple_s4(self, tp):
        self._init_logic_tuple_s4(tp)

    def _move_data_no_dup_s4(self, tp, ub_input, is_src_tail_in, is_dst_tail_in):
        tik_inst = self.tik_inst
        with tik_inst.for_range(0, tp.loop_per_core):
            is_src_tail_in.set_as(0)
            is_dst_tail_in.set_as(0)

            self._init_major_src_tuple_copy_out_s4(tp)
            self._init_major_dst_tuple_copy_in_s4(tp)
            self._copy_in_major_src_major_dst_s4(tp, ub_input)
            self._reorder_s4(tp, ub_input, 0, 0,
                             tp.major_in_ele,
                             tp.major_in_tail_ele,
                             tp.major_burst_len_in,
                             tp.major_dst_loop_in,
                             tp.major_out_ele,
                             tp.major_out_tail_ele,
                             tp.major_burst_len_out,
                             tp.major_src_loop_out)
            self._copy_out_major_src_major_dst_s4(tp, ub_input)

            self._detect_tail_flag(tp, is_src_tail_in, is_dst_tail_in)

            with tik_inst.if_scope(is_src_tail_in == 1):
                with tik_inst.if_scope(tp.tail_burst_len_in != 0):
                    self._init_tail_src_tuple_copy_out_s4(tp)
                    self._init_major_dst_tuple_copy_in_s4(tp)
                    self._copy_in_tail_src_major_dst_s4(tp, ub_input)
                    self._reorder_s4(tp, ub_input, 1, 0,
                                     tp.tail_in_ele,
                                     tp.tail_in_tail_ele,
                                     tp.tail_burst_len_in,
                                     tp.major_dst_loop_in,
                                     tp.major_out_ele,
                                     tp.major_out_tail_ele,
                                     tp.major_burst_len_out,
                                     tp.major_src_loop_out)
                    self._copy_out_tail_src_major_dst_s4(tp, ub_input)

            with tik_inst.if_scope(tik.all(is_dst_tail_in == 1, tp.pivot_dst_axis_dup == 0)):
                with tik_inst.if_scope(tp.tail_burst_len_out != 0):
                    self._init_major_src_tuple_copy_out_s4(tp)
                    self._init_tail_dst_tuple_copy_in_s4(tp)
                    self._copy_in_major_src_tail_dst_s4(tp, ub_input)
                    self._reorder_s4(tp, ub_input, 0, 1,
                                     tp.major_in_ele,
                                     tp.major_in_tail_ele,
                                     tp.major_burst_len_in,
                                     tp.tail_dst_loop_in,
                                     tp.tail_out_ele,
                                     tp.tail_out_tail_ele,
                                     tp.tail_burst_len_out,
                                     tp.major_src_loop_out)
                    self._copy_out_major_src_tail_dst_s4(tp, ub_input)

            with tik_inst.if_scope(tik.all(is_dst_tail_in == 1, is_src_tail_in == 1)):
                with tik_inst.if_scope(tik.all(tp.tail_burst_len_in != 0, tp.tail_burst_len_out != 0)):
                    self._init_tail_src_tuple_copy_out_s4(tp)
                    self._init_tail_dst_tuple_copy_in_s4(tp)
                    self._copy_in_tail_src_tail_dst_s4(tp, ub_input)
                    self._reorder_s4(tp, ub_input, 1, 1,
                                     tp.tail_in_ele,
                                     tp.tail_in_tail_ele,
                                     tp.tail_burst_len_in,
                                     tp.tail_dst_loop_in,
                                     tp.tail_out_ele,
                                     tp.tail_out_tail_ele,
                                     tp.tail_burst_len_out,
                                     tp.tail_src_loop_out)
                    self._copy_out_tail_src_tail_dst_s4(tp, ub_input)

            self._update_logic_tuple_s4(tp)

    def _move_data_dup_0x210_s4(self, tp, ub_input, is_src_tail_in, is_dst_tail_in):
        tik_inst = self.tik_inst
        with tik_inst.for_range(0, tp.loop_per_core):
            is_src_tail_in.set_as(0)
            is_dst_tail_in.set_as(0)

            self._init_major_src_tuple_copy_out_dup_0x210_s4(tp)
            self._init_major_dst_tuple_copy_in_dup_0x210_s4(tp)
            self._copy_in_major_src_major_dst_dup_0x210_s4(tp, ub_input)
            self._reorder_s4(tp, ub_input, 0, 0,
                             tp.major_in_ele,
                             tp.major_in_tail_ele,
                             tp.major_burst_len_in,
                             tp.major_dst_loop_in,
                             tp.major_out_ele,
                             tp.major_out_tail_ele,
                             tp.major_burst_len_out,
                             tp.major_src_loop_out)
            self._copy_out_major_src_major_dst_dup_0x210_s4(tp, ub_input)

            self._detect_tail_flag(tp, is_src_tail_in, is_dst_tail_in)

            with tik_inst.if_scope(tik.all(is_src_tail_in == 1, is_dst_tail_in == 1)):
                with tik_inst.if_scope(tik.all(tp.tail_burst_len_in != 0, tp.tail_burst_len_out != 0)):
                    self._init_tail_src_tuple_copy_out_dup_0x210_s4(tp)
                    self._init_tail_dst_tuple_copy_in_dup_0x210_s4(tp)
                    self._copy_in_tail_src_tail_dst_dup_0x210_s4(tp, ub_input)
                    self._reorder_s4(tp, ub_input, 1, 1,
                                     tp.tail_in_ele,
                                     tp.tail_in_tail_ele,
                                     tp.tail_burst_len_in,
                                     tp.major_dst_loop_in,
                                     tp.tail_out_ele,
                                     tp.tail_out_tail_ele,
                                     tp.tail_burst_len_out,
                                     tp.major_src_loop_out)
                    self._copy_out_tail_src_tail_dst_dup_0x210_s4(tp, ub_input)

            self._update_logic_tuple_s4(tp)

    def _move_data_dup_0x201_s4(self, tp, ub_input, is_src_tail_in, is_dst_tail_in):
        # major_src_major_dst + major_src_tail_dst
        tik_inst = self.tik_inst
        with tik_inst.for_range(0, tp.loop_per_core):
            is_src_tail_in.set_as(0)
            is_dst_tail_in.set_as(0)
            tik_inst = self.tik_inst

            self._init_major_src_tuple_copy_out_dup_0x201_s4(tp)
            self._init_major_dst_tuple_copy_in_dup_0x201_s4(tp)
            self._copy_in_major_src_major_dst_dup_0x201_s4(tp, ub_input)
            self._reorder_s4(tp, ub_input, 0, 0,
                            tp.major_in_ele,
                            tp.major_in_tail_ele,
                            tp.major_burst_len_in,
                            tp.major_dst_loop_in,
                            tp.major_out_ele,
                            tp.major_out_tail_ele,
                            tp.major_burst_len_out,
                            tp.major_src_loop_out)
            self._copy_out_major_src_major_dst_dup_0x201_s4(tp, ub_input)

            self._detect_tail_flag(tp, is_src_tail_in, is_dst_tail_in)

            with tik_inst.if_scope(is_dst_tail_in == 1):
                with tik_inst.if_scope(tp.tail_burst_len_out != 0):
                    self._init_major_src_tuple_copy_out_dup_0x201_s4(tp)
                    self._init_tail_dst_tuple_copy_in_dup_0x201_s4(tp)
                    self._copy_in_major_src_tail_dst_dup_0x201_s4(tp, ub_input)
                    self._reorder_s4(tp, ub_input, 0, 1,
                                     tp.major_in_ele,
                                     tp.major_in_tail_ele,
                                     tp.major_burst_len_in,
                                     tp.tail_dst_loop_in,
                                     tp.tail_out_ele,
                                     tp.tail_out_tail_ele,
                                     tp.tail_burst_len_out,
                                     tp.major_src_loop_out)
                    self._copy_out_major_src_tail_dst_dup_0x201_s4(tp, ub_input)
            self._update_logic_tuple_s4(tp)

    def _move_data_dup_0x120_s4(self, tp, ub_input, is_src_tail_in, is_dst_tail_in):
        tik_inst = self.tik_inst
        with tik_inst.for_range(0, tp.loop_per_core):
            is_src_tail_in.set_as(0)
            is_dst_tail_in.set_as(0)

            self._init_major_dst_tuple_copy_in_dup_0x120_s4(tp)
            self._init_major_src_tuple_copy_out_dup_0x120_s4(tp)
            self._copy_in_major_src_major_dst_s4(tp, ub_input)
            self._reorder_s4(tp, ub_input, 0, 0,
                             tp.major_in_ele,
                             tp.major_in_tail_ele,
                             tp.major_burst_len_in,
                             tp.major_dst_loop_in,
                             tp.major_out_ele,
                             tp.major_out_tail_ele,
                             tp.major_burst_len_out,
                             tp.major_src_loop_out)
            self._copy_out_major_src_major_dst_s4(tp, ub_input)

            self._detect_tail_flag(tp, is_src_tail_in, is_dst_tail_in)

            with tik_inst.if_scope(is_src_tail_in == 1):
                with tik_inst.if_scope(tp.tail_burst_len_in != 0):
                    self._init_major_dst_tuple_copy_in_dup_0x120_s4(tp)
                    self._init_tail_src_tuple_copy_out_dup_0x120_s4(tp)
                    self._copy_in_tail_dup_s4(tp, ub_input)
                    self._reorder_s4(tp, ub_input, 1, 0,
                                     tp.tail_in_ele,
                                     tp.tail_in_tail_ele,
                                     tp.tail_burst_len_in,
                                     tp.major_dst_loop_in,
                                     tp.major_out_ele,
                                     tp.major_out_tail_ele,
                                     tp.major_burst_len_out,
                                     tp.tail_src_loop_out)
                    self._copy_out_tail_dup_0x01_s4(tp, ub_input)

            self._update_logic_tuple_s4(tp)

    def _move_data_dup_0x10_s1d2_s4(self, tp, ub_input, is_src_tail_in, is_dst_tail_in):
        tik_inst = self.tik_inst
        with tik_inst.for_range(0, tp.loop_per_core):
            is_src_tail_in.set_as(0)
            is_dst_tail_in.set_as(0)

            self._init_major_dst_tuple_copy_in_dup_0x10_s1d2_s4(tp)
            self._copy_in_major_src_major_dst_dup_0x10_s1d2_s4(tp, ub_input)
            self._reorder_s4(tp, ub_input, 0, 0,
                             tp.major_in_ele,
                             tp.major_in_tail_ele,
                             tp.major_burst_len_in,
                             tp.major_dst_loop_in,
                             tp.major_out_ele,
                             tp.major_out_tail_ele,
                             tp.major_burst_len_out,
                             tp.major_src_loop_out)
            self._copy_out_major_src_major_dst_dup_0x10_s1d2_s4(tp, ub_input)

            self._detect_tail_flag(tp, is_src_tail_in, is_dst_tail_in)

            with tik_inst.if_scope(is_src_tail_in == 1):
                with tik_inst.if_scope(tp.tail_burst_len_in != 0):
                    self._init_major_dst_tuple_copy_in_dup_0x10_s1d2_s4(tp)
                    self._copy_in_tail_src_major_dst_dup_0x10_s1d2_s4(tp, ub_input)
                    self._reorder_s4(tp, ub_input, 1, 0,
                                     tp.tail_in_ele,
                                     tp.tail_in_tail_ele,
                                     tp.tail_burst_len_in,
                                     tp.major_dst_loop_in,
                                     tp.tail_out_ele,
                                     tp.tail_out_tail_ele,
                                     tp.tail_burst_len_out,
                                     tp.major_src_loop_out)
                    self._copy_out_tail_src_major_dst_dup_0x10_s1d2_s4(tp, ub_input)

            self._update_logic_tuple_s4(tp)

    def _move_data_dup_0x10_s2d1_s4(self, tp, ub_input, is_src_tail_in, is_dst_tail_in):
        # major_src_major_dst + tail_src_major_dst
        tik_inst = self.tik_inst
        with tik_inst.for_range(0, tp.loop_per_core):
            is_src_tail_in.set_as(0)
            is_dst_tail_in.set_as(0)

            self._init_major_src_tuple_copy_out_dup_0x10_s2d1_s4(tp)
            self._copy_in_major_src_major_dst_dup_0x10_s2d1_s4(tp, ub_input)
            self._reorder_s4(tp, ub_input, 0, 0,
                             tp.major_in_ele,
                             tp.major_in_tail_ele,
                             tp.major_burst_len_in,
                             tp.major_dst_loop_in,
                             tp.major_out_ele,
                             tp.major_out_tail_ele,
                             tp.major_burst_len_out,
                             tp.major_src_loop_out)
            self._copy_out_major_src_major_dst_dup_0x10_s2d1_s4(tp, ub_input)

            self._detect_tail_flag(tp, is_src_tail_in, is_dst_tail_in)

            with tik_inst.if_scope(is_dst_tail_in == 1):
                with tik_inst.if_scope(tp.tail_burst_len_in != 0):
                    self._init_major_src_tuple_copy_out_dup_0x10_s2d1_s4(tp)
                    self._copy_in_tail_src_major_dst_dup_0x10_s2d1_s4(tp, ub_input)
                    self._reorder_s4(tp, ub_input, 1, 0,
                                     tp.tail_in_ele,
                                     tp.tail_in_tail_ele,
                                     tp.tail_burst_len_in,
                                     tp.major_dst_loop_in,
                                     tp.tail_out_ele,
                                     tp.tail_out_tail_ele,
                                     tp.tail_burst_len_out,
                                     tp.major_src_loop_out)
                    self._copy_out_tail_src_major_dst_dup_0x10_s2d1_s4(tp, ub_input)

            self._update_logic_tuple_s4(tp)

    def _move_data_s4(self, tp, ub_input_64):
        is_src_tail_in = self.tik_inst.Scalar("int32")
        is_dst_tail_in = self.tik_inst.Scalar("int32")

        ub_input = ub_input_64.reinterpret_cast_to(self.x_dtype)
        tik_inst = self.tik_inst

        self._init_all_tuple_s4(tp)

        with tik_inst.if_scope(tp.dup_axis == 0):
            self._move_data_no_dup_s4(tp, ub_input, is_src_tail_in, is_dst_tail_in)

        with tik_inst.if_scope(tp.dup_axis == 2):
            self._move_data_no_dup_s4(tp, ub_input, is_src_tail_in, is_dst_tail_in)

        with tik_inst.if_scope(tik.all(tp.dup_axis == 1, tp.ub_axis_perm == 0x210)):
            self._move_data_dup_0x210_s4(tp, ub_input, is_src_tail_in, is_dst_tail_in)

        with tik_inst.if_scope(tik.all(tp.dup_axis == 1, tp.ub_axis_perm == 0x201)):
            self._move_data_dup_0x201_s4(tp, ub_input, is_src_tail_in, is_dst_tail_in)

        with tik_inst.if_scope(tik.all(tp.dup_axis == 1, tp.ub_axis_perm == 0x120)):
            self._move_data_dup_0x120_s4(tp, ub_input, is_src_tail_in, is_dst_tail_in)

        with tik_inst.if_scope(tik.all(tp.dup_axis == 1, tp.ub_axis_perm == 0x10)):

            # 2, 10000, x -> 10000, 2, x
            with tik_inst.if_scope(tp.dst_axis_num_no_dup == 1):
                self._move_data_dup_0x10_s1d2_s4(tp, ub_input, is_src_tail_in, is_dst_tail_in)

            # 10000, 2, x -> 2, 10000, x
            with tik_inst.if_scope(tp.src_axis_num_no_dup == 1):
                self._move_data_dup_0x10_s2d1_s4(tp, ub_input, is_src_tail_in, is_dst_tail_in)

    # -------------------------------------------------------------------------------------------------
    #                                    scenario_5
    # -------------------------------------------------------------------------------------------------
    @staticmethod
    def _init_logic_tuple_s5(tp):
        for i in range(TRANSPOSE_MAX_AXIS_NUM):
            tp.rt_logic_tuple[i].set_as(tp.init_logic_tuple[i])

    def _init_all_tuple_s5(self, tp):
        self._init_logic_tuple_s5(tp)

    def _update_logic_tuple_s5(self, tp):
        self._update_tuple(tp.logic_axis_num, tp.rt_logic_tuple, tp.logic_jump_factor)

    def _update_dst_tuple_major_s5(self, tp, repeat):
        tik_inst = self.tik_inst
        axis_num = tp.dst_axis_num_no_dup
        rt_tuple = tp.rt_dst_tuple_in
        step = tp.dst_jump_major_step
        logic_tuple = tp.rt_logic_tuple[1]
        jump_factor = tp.dst_jump_factor_in
        axis_perm = tp.dst_axis_perm
        with self.tik_inst.if_scope(axis_num == 3):
            with tik_inst.if_scope(tik.any(axis_perm == 0x012, axis_perm == 0x021)):
                with self.tik_inst.if_scope((rt_tuple[0] + repeat) == jump_factor[0]):
                    rt_tuple[0].set_as(0)
                    with self.tik_inst.if_scope((rt_tuple[1] + 1) == jump_factor[1]):
                        rt_tuple[1].set_as(0)
                        rt_tuple[2].set_as(rt_tuple[2] + 1)
                    with self.tik_inst.else_scope():
                        rt_tuple[1].set_as(rt_tuple[1] + 1)
                with self.tik_inst.else_scope():
                    rt_tuple[0].set_as(rt_tuple[0] + repeat)

            with tik_inst.if_scope(tik.any(axis_perm == 0x120, axis_perm == 0x102)):
                with self.tik_inst.if_scope((rt_tuple[0] + repeat) == jump_factor[0]):
                    rt_tuple[0].set_as(0)
                    with self.tik_inst.if_scope((rt_tuple[1] + 1) % step == 0):
                        rt_tuple[1].set_as(logic_tuple * step)
                        rt_tuple[2].set_as(rt_tuple[2] + 1)
                    with self.tik_inst.else_scope():
                        rt_tuple[1].set_as(rt_tuple[1] + 1)
                with self.tik_inst.else_scope():
                    rt_tuple[0].set_as(rt_tuple[0] + repeat)

            with tik_inst.if_scope(tik.any(axis_perm == 0x210, axis_perm == 0x201)):
                with self.tik_inst.if_scope((rt_tuple[0] + 1) % step == 0):
                    rt_tuple[0].set_as(logic_tuple * step)
                    with self.tik_inst.if_scope((rt_tuple[1] + 1) == jump_factor[1]):
                        rt_tuple[1].set_as(0)
                        rt_tuple[2].set_as(rt_tuple[2] + 1)
                    with self.tik_inst.else_scope():
                        rt_tuple[1].set_as(rt_tuple[1] + 1)
                with self.tik_inst.else_scope():
                    rt_tuple[0].set_as(rt_tuple[0] + 1)

        with tik_inst.if_scope(axis_num == 2):

            with tik_inst.if_scope(tik.any(axis_perm == 0x10, axis_perm == 0x102, axis_perm == 0x120)):
                with tik_inst.if_scope((rt_tuple[0] + 1) % step == 0):
                    rt_tuple[0].set_as(logic_tuple * step)
                    rt_tuple[1].set_as(rt_tuple[1] + 1)
                with tik_inst.else_scope():
                    rt_tuple[0].set_as(rt_tuple[0] + 1)
            with tik_inst.else_scope():
                with tik_inst.if_scope((rt_tuple[0] + repeat) == jump_factor[0]):
                    rt_tuple[0].set_as(0)
                    rt_tuple[1].set_as(rt_tuple[1] + 1)
                with tik_inst.else_scope():
                    rt_tuple[0].set_as(rt_tuple[0] + repeat)

        with tik_inst.if_scope(axis_num == 1):
            rt_tuple[0].set_as(rt_tuple[0] + repeat)

    def _update_src_tuple_major_s5(self, axis_num, rt_tuple, step, logic_tuple, jump_factor, axis_perm):
        tik_inst = self.tik_inst

        with self.tik_inst.if_scope(axis_num == 3):
            with tik_inst.if_scope(tik.any(axis_perm == 0x012, axis_perm == 0x021)):
                with self.tik_inst.if_scope((rt_tuple[0] + 1) == jump_factor[0]):
                    rt_tuple[0].set_as(0)
                    with self.tik_inst.if_scope((rt_tuple[1] + 1) == jump_factor[1]):
                        rt_tuple[1].set_as(0)
                        rt_tuple[2].set_as(rt_tuple[2] + 1)
                    with self.tik_inst.else_scope():
                        rt_tuple[1].set_as(rt_tuple[1] + 1)
                with self.tik_inst.else_scope():
                    rt_tuple[0].set_as(rt_tuple[0] + 1)

            with tik_inst.if_scope(tik.any(axis_perm == 0x102, axis_perm == 0x201)):
                with self.tik_inst.if_scope((rt_tuple[0] + 1) == jump_factor[0]):
                    rt_tuple[0].set_as(0)
                    with self.tik_inst.if_scope((rt_tuple[1] + 1) % step == 0):
                        rt_tuple[1].set_as(logic_tuple * step)
                        rt_tuple[2].set_as(rt_tuple[2] + 1)
                    with self.tik_inst.else_scope():
                        rt_tuple[1].set_as(rt_tuple[1] + 1)
                with self.tik_inst.else_scope():
                    rt_tuple[0].set_as(rt_tuple[0] + 1)

            with tik_inst.if_scope(tik.any(axis_perm == 0x120, axis_perm == 0x210)):
                with self.tik_inst.if_scope((rt_tuple[0] + 1) % step == 0):
                    rt_tuple[0].set_as(logic_tuple * step)
                    with self.tik_inst.if_scope((rt_tuple[1] + 1) == jump_factor[1]):
                        rt_tuple[1].set_as(0)
                        rt_tuple[2].set_as(rt_tuple[2] + 1)
                    with self.tik_inst.else_scope():
                        rt_tuple[1].set_as(rt_tuple[1] + 1)
                with self.tik_inst.else_scope():
                    rt_tuple[0].set_as(rt_tuple[0] + 1)

        with tik_inst.if_scope(axis_num == 2):

            with tik_inst.if_scope(tik.any(axis_perm == 0x10, axis_perm == 0x102, axis_perm == 0x201)):
                with tik_inst.if_scope((rt_tuple[0] + 1) % step == 0):
                    rt_tuple[0].set_as(logic_tuple * step)
                    rt_tuple[1].set_as(rt_tuple[1] + 1)
                with tik_inst.else_scope():
                    rt_tuple[0].set_as(rt_tuple[0] + 1)
            with tik_inst.else_scope():
                with tik_inst.if_scope((rt_tuple[0] + 1) == jump_factor[0]):
                    rt_tuple[0].set_as(0)
                    rt_tuple[1].set_as(rt_tuple[1] + 1)
                with tik_inst.else_scope():
                    rt_tuple[0].set_as(rt_tuple[0] + 1)

        with tik_inst.if_scope(axis_num == 1):
            rt_tuple[0].set_as(rt_tuple[0] + 1)

    def _update_dst_tuple_tail_s5(self, axis_num, rt_tuple, step, logic_tuple, jump_factor, axis_perm):
        tik_inst = self.tik_inst

        with self.tik_inst.if_scope(axis_num == 3):
            with tik_inst.if_scope(tik.any(axis_perm == 0x012, axis_perm == 0x021)):
                with self.tik_inst.if_scope((rt_tuple[0] + 1) == jump_factor[0]):
                    rt_tuple[0].set_as(0)
                    with self.tik_inst.if_scope((rt_tuple[1] + 1) == jump_factor[1]):
                        rt_tuple[1].set_as(0)
                        rt_tuple[2].set_as(rt_tuple[2] + 1)
                    with self.tik_inst.else_scope():
                        rt_tuple[1].set_as(rt_tuple[1] + 1)
                with self.tik_inst.else_scope():
                    rt_tuple[0].set_as(rt_tuple[0] + 1)

            with tik_inst.if_scope(tik.any(axis_perm == 0x120, axis_perm == 0x102)):
                with self.tik_inst.if_scope((rt_tuple[0] + 1) == jump_factor[0]):
                    rt_tuple[0].set_as(0)
                    with self.tik_inst.if_scope((rt_tuple[1] + 1) == jump_factor[1]):
                        rt_tuple[1].set_as((logic_tuple + 1) * step)
                        rt_tuple[2].set_as(rt_tuple[2] + 1)
                    with self.tik_inst.else_scope():
                        rt_tuple[1].set_as(rt_tuple[1] + 1)
                with self.tik_inst.else_scope():
                    rt_tuple[0].set_as(rt_tuple[0] + 1)

            with tik_inst.if_scope(tik.any(axis_perm == 0x210, axis_perm == 0x201)):
                with self.tik_inst.if_scope((rt_tuple[0] + 1) == jump_factor[0]):
                    rt_tuple[0].set_as((logic_tuple + 1) * step)
                    with self.tik_inst.if_scope((rt_tuple[1] + 1) == jump_factor[1]):
                        rt_tuple[1].set_as(0)
                        rt_tuple[2].set_as(rt_tuple[2] + 1)
                    with self.tik_inst.else_scope():
                        rt_tuple[1].set_as(rt_tuple[1] + 1)
                with self.tik_inst.else_scope():
                    rt_tuple[0].set_as(rt_tuple[0] + 1)

        with self.tik_inst.if_scope(axis_num == 2):

            with tik_inst.if_scope(tik.any(axis_perm == 0x10, axis_perm == 0x102, axis_perm == 0x120)):
                with self.tik_inst.if_scope((rt_tuple[0] + 1) == jump_factor[0]):
                    rt_tuple[0].set_as((logic_tuple + 1) * step)
                    rt_tuple[1].set_as(rt_tuple[1] + 1)
                with self.tik_inst.else_scope():
                    rt_tuple[0].set_as(rt_tuple[0] + 1)
            with tik_inst.else_scope():
                with tik_inst.if_scope((rt_tuple[0] + 1) == jump_factor[0]):
                    rt_tuple[0].set_as(0)
                    rt_tuple[1].set_as(rt_tuple[1] + 1)
                with tik_inst.else_scope():
                    rt_tuple[0].set_as(rt_tuple[0] + 1)

        with self.tik_inst.if_scope(axis_num == 1):
            rt_tuple[0].set_as(rt_tuple[0] + 1)

    def _update_src_tuple_tail_s5(self, axis_num, rt_tuple, step, logic_tuple, jump_factor, axis_perm):
        tik_inst = self.tik_inst

        with self.tik_inst.if_scope(axis_num == 3):
            with tik_inst.if_scope(tik.any(axis_perm == 0x012, axis_perm == 0x021)):
                with self.tik_inst.if_scope((rt_tuple[0] + 1) == jump_factor[0]):
                    rt_tuple[0].set_as(0)
                    with self.tik_inst.if_scope((rt_tuple[1] + 1) == jump_factor[1]):
                        rt_tuple[1].set_as(0)
                        rt_tuple[2].set_as(rt_tuple[2] + 1)
                    with self.tik_inst.else_scope():
                        rt_tuple[1].set_as(rt_tuple[1] + 1)
                with self.tik_inst.else_scope():
                    rt_tuple[0].set_as(rt_tuple[0] + 1)

            with tik_inst.if_scope(tik.any(axis_perm == 0x102, axis_perm == 0x201)):
                with self.tik_inst.if_scope((rt_tuple[0] + 1) == jump_factor[0]):
                    rt_tuple[0].set_as(0)
                    with self.tik_inst.if_scope((rt_tuple[1] + 1) == jump_factor[1]):
                        rt_tuple[1].set_as((logic_tuple + 1) * step)
                        rt_tuple[2].set_as(rt_tuple[2] + 1)
                    with self.tik_inst.else_scope():
                        rt_tuple[1].set_as(rt_tuple[1] + 1)
                with self.tik_inst.else_scope():
                    rt_tuple[0].set_as(rt_tuple[0] + 1)

            with tik_inst.if_scope(tik.any(axis_perm == 0x120, axis_perm == 0x210)):
                with self.tik_inst.if_scope((rt_tuple[0] + 1) == jump_factor[0]):
                    rt_tuple[0].set_as((logic_tuple + 1) * step)
                    with self.tik_inst.if_scope((rt_tuple[1] + 1) == jump_factor[1]):
                        rt_tuple[1].set_as(0)
                        rt_tuple[2].set_as(rt_tuple[2] + 1)
                    with self.tik_inst.else_scope():
                        rt_tuple[1].set_as(rt_tuple[1] + 1)
                with self.tik_inst.else_scope():
                    rt_tuple[0].set_as(rt_tuple[0] + 1)

        with self.tik_inst.if_scope(axis_num == 2):
            with tik_inst.if_scope(tik.any(axis_perm == 0x10, axis_perm == 0x102, axis_perm == 0x201)):
                with self.tik_inst.if_scope((rt_tuple[0] + 1) == jump_factor[0]):
                    rt_tuple[0].set_as((logic_tuple + 1) * step)
                    rt_tuple[1].set_as(rt_tuple[1] + 1)
                with self.tik_inst.else_scope():
                    rt_tuple[0].set_as(rt_tuple[0] + 1)
            with tik_inst.else_scope():
                with self.tik_inst.if_scope((rt_tuple[0] + 1) == jump_factor[0]):
                    rt_tuple[0].set_as(0)
                    rt_tuple[1].set_as(rt_tuple[1] + 1)
                with self.tik_inst.else_scope():
                    rt_tuple[0].set_as(rt_tuple[0] + 1)

        with self.tik_inst.if_scope(axis_num == 1):
            rt_tuple[0].set_as(rt_tuple[0] + 1)

    def _update_major_dst_tuple_in_s5(self, tp, repeat=1):
        self._update_dst_tuple_major_s5(tp, repeat)

    def _update_major_src_tuple_out_s5(self, tp):
        self._update_src_tuple_major_s5(tp.src_axis_num_no_dup,
                                        tp.rt_src_tuple_out,
                                        tp.src_jump_major_step,
                                        tp.rt_logic_tuple[0],
                                        tp.src_jump_factor_out,
                                        tp.src_axis_perm)

    def _update_tail_dst_tuple_in_s5(self, tp):
        self._update_dst_tuple_tail_s5(tp.dst_axis_num_no_dup,
                                       tp.rt_dst_tuple_in,
                                       tp.dst_jump_major_step,
                                       tp.rt_logic_tuple[1],
                                       tp.dst_jump_factor_in,
                                       tp.dst_axis_perm)

    def _update_tail_src_tuple_out_s5(self, tp):
        self._update_src_tuple_tail_s5(tp.src_axis_num_no_dup,
                                       tp.rt_src_tuple_out,
                                       tp.src_jump_major_step,
                                       tp.rt_logic_tuple[0],
                                       tp.src_jump_factor_out,
                                       tp.src_axis_perm)

    def _get_src_addr_s5(self, tp, src_addr, is_tail):
        tik_inst = self.tik_inst

        src_addr.set_as(0)

        with tik_inst.if_scope(is_tail == 0):
            src_addr.set_as(tp.rt_logic_tuple[0] * tp.major_in_ele)
        with tik_inst.else_scope():
            src_addr.set_as((tp.rt_logic_tuple[0] + 1) * tp.major_in_ele)

        with tik_inst.for_range(0, tp.dst_axis_num_no_dup) as i:
            src_addr.set_as(src_addr + tp.rt_dst_tuple_in[i] * tp.dst_stride_in[i])

        with tik_inst.for_range(2, tp.other_axis_num + 2) as i:
            src_addr.set_as(src_addr + tp.rt_logic_tuple[i] * tp.logic_stride_in[i])

    def _get_dst_addr_s5(self, tp, is_tail):
        tik_inst = self.tik_inst
        dst_addr = tp.dst_addr
        dst_addr.set_as(0)

        with tik_inst.if_scope(tik.all(tp.pivot_src_axis_dup == 1, tp.pivot_dst_axis_dup == 1)):
            with tik_inst.if_scope(is_tail == 0):
                dst_addr.set_as(tp.rt_logic_tuple[0] * tp.major_out_ele)
            with tik_inst.else_scope():
                dst_addr.set_as((tp.rt_logic_tuple[0] + 1) * tp.major_out_ele)
        with tik_inst.else_scope():
            with tik_inst.if_scope(is_tail == 0):
                dst_addr.set_as(tp.rt_logic_tuple[1] * tp.major_out_ele)
            with tik_inst.else_scope():
                dst_addr.set_as((tp.rt_logic_tuple[1] + 1) * tp.major_out_ele)

        with tik_inst.for_range(0, tp.src_axis_num_no_dup) as i:
            dst_addr.set_as(dst_addr + tp.rt_src_tuple_out[i] * tp.src_stride_out[i])

        with tik_inst.for_range(2, tp.other_axis_num + 2) as i:
            dst_addr.set_as(dst_addr + tp.rt_logic_tuple[i] * tp.logic_stride_out[i])

    def _get_ub_src_offset_s5(self, tp):
        with self.tik_inst.if_scope(tp.src_axis_num_no_dup == 1):

            with self.tik_inst.if_scope(tik.any(tp.src_axis_perm == 0x0,
                                                tp.src_axis_perm == 0x01,
                                                tp.src_axis_perm == 0x021,
                                                tp.src_axis_perm == 0x012)):
                tp.ub_src_offset.set_as(tp.rt_src_tuple_out[0] % tp.src_jump_major_step)
            with self.tik_inst.else_scope():
                tp.ub_src_offset.set_as(tp.rt_src_tuple_out[0])

        with self.tik_inst.if_scope(tp.src_axis_num_no_dup == 2):

            with self.tik_inst.if_scope(tp.src_axis_perm == 0x01):
                tp.ub_src_offset.set_as(tp.rt_src_tuple_out[0])
                tp.ub_src_offset.set_as(tp.ub_src_offset + \
                                        tp.rt_src_tuple_out[1] % tp.src_jump_major_step * tp.src_jump_factor_out[0])

            with self.tik_inst.if_scope(tp.src_axis_perm == 0x10):
                tp.ub_src_offset.set_as(tp.rt_src_tuple_out[0] % tp.src_jump_major_step * tp.src_jump_factor_out[1])
                tp.ub_src_offset.set_as(tp.ub_src_offset + tp.rt_src_tuple_out[1])

            with self.tik_inst.if_scope(tp.src_axis_perm == 0x021):
                tp.ub_src_offset.set_as(tp.rt_src_tuple_out[0])
                tp.ub_src_offset.set_as(tp.ub_src_offset + \
                                        tp.rt_src_tuple_out[1] % tp.src_jump_major_step * tp.src_jump_factor_out[0])

            with self.tik_inst.if_scope(tp.src_axis_perm == 0x012):
                tp.ub_src_offset.set_as(tp.rt_src_tuple_out[0])
                tp.ub_src_offset.set_as(tp.ub_src_offset + \
                                        tp.rt_src_tuple_out[1] % tp.src_jump_major_step * tp.src_jump_factor_out[0])

            with self.tik_inst.if_scope(tp.src_axis_perm == 0x102):
                tp.ub_src_offset.set_as(tp.rt_src_tuple_out[0] % tp.src_jump_major_step * tp.src_jump_factor_out[1])
                tp.ub_src_offset.set_as(tp.ub_src_offset + tp.rt_src_tuple_out[1])

            with self.tik_inst.if_scope(tp.src_axis_perm == 0x120):
                tp.ub_src_offset.set_as(tp.rt_src_tuple_out[0])
                tp.ub_src_offset.set_as(tp.ub_src_offset + tp.rt_src_tuple_out[1] * tp.src_jump_factor_out[0])

            with self.tik_inst.if_scope(tp.src_axis_perm == 0x201):
                tp.ub_src_offset.set_as(tp.rt_src_tuple_out[0] % tp.src_jump_major_step * tp.src_jump_factor_out[1])
                tp.ub_src_offset.set_as(tp.ub_src_offset + tp.rt_src_tuple_out[1])

            with self.tik_inst.if_scope(tp.src_axis_perm == 0x210):
                tp.ub_src_offset.set_as(tp.rt_src_tuple_out[0] * tp.src_jump_factor_out[1])
                tp.ub_src_offset.set_as(tp.ub_src_offset + tp.rt_src_tuple_out[1])

        with self.tik_inst.if_scope(tp.src_axis_num_no_dup == 3):

            with self.tik_inst.if_scope(tp.src_axis_perm == 0x012):
                tp.ub_src_offset.set_as(tp.rt_src_tuple_out[0])
                tp.ub_src_offset.set_as(tp.ub_src_offset + tp.rt_src_tuple_out[1] * tp.src_jump_factor_out[0])
                tp.ub_src_offset.set_as(tp.ub_src_offset + \
                                        tp.rt_src_tuple_out[2] % tp.src_jump_major_step * \
                                        tp.src_jump_factor_out[0] * tp.src_jump_factor_out[1])

            with self.tik_inst.if_scope(tp.src_axis_perm == 0x021):
                tp.ub_src_offset.set_as(tp.rt_src_tuple_out[0] * tp.src_jump_factor_out[1])
                tp.ub_src_offset.set_as(tp.ub_src_offset + tp.rt_src_tuple_out[1])
                tp.ub_src_offset.set_as(tp.ub_src_offset + tp.rt_src_tuple_out[2] % tp.src_jump_major_step * \
                                        tp.src_jump_factor_out[0] * tp.src_jump_factor_out[1])

            with self.tik_inst.if_scope(tp.src_axis_perm == 0x102):
                tp.ub_src_offset.set_as(tp.rt_src_tuple_out[0])
                tp.ub_src_offset.set_as(tp.ub_src_offset + tp.rt_src_tuple_out[1] % tp.src_jump_major_step * \
                                        tp.src_jump_factor_out[0] * tp.src_jump_factor_out[2])
                tp.ub_src_offset.set_as(tp.ub_src_offset + tp.rt_src_tuple_out[2] * tp.src_jump_factor_out[0])

            with self.tik_inst.if_scope(tp.src_axis_perm == 0x120):
                tp.ub_src_offset.set_as(tp.rt_src_tuple_out[0] % tp.src_jump_major_step * \
                                        tp.src_jump_factor_out[1] * tp.src_jump_factor_out[2])
                tp.ub_src_offset.set_as(tp.ub_src_offset + tp.rt_src_tuple_out[1])
                tp.ub_src_offset.set_as(tp.ub_src_offset + tp.rt_src_tuple_out[2] * tp.src_jump_factor_out[1])

            with self.tik_inst.if_scope(tp.src_axis_perm == 0x210):
                tp.ub_src_offset.set_as(tp.rt_src_tuple_out[0] % tp.src_jump_major_step * \
                                        tp.src_jump_factor_out[1] * tp.src_jump_factor_out[2])
                tp.ub_src_offset.set_as(tp.ub_src_offset + tp.rt_src_tuple_out[1] * tp.src_jump_factor_out[2])
                tp.ub_src_offset.set_as(tp.ub_src_offset + tp.rt_src_tuple_out[2])

            with self.tik_inst.if_scope(tp.src_axis_perm == 0x201):
                tp.ub_src_offset.set_as(tp.rt_src_tuple_out[0] * tp.src_jump_factor_out[2])
                tp.ub_src_offset.set_as(tp.ub_src_offset + tp.rt_src_tuple_out[1] % tp.src_jump_major_step * \
                                        tp.src_jump_factor_out[0] * tp.src_jump_factor_out[2])
                tp.ub_src_offset.set_as(tp.ub_src_offset + tp.rt_src_tuple_out[2])

    def _init_major_src_tuple_copy_out_s5(self, tp):
        tik_inst = self.tik_inst

        with self.tik_inst.if_scope(tp.src_axis_num_no_dup == 3):

            with tik_inst.if_scope(tik.any(tp.src_axis_perm == 0x210, tp.src_axis_perm == 0x120)):
                tp.rt_src_tuple_out[0].set_as(tp.rt_logic_tuple[0] * tp.src_jump_major_step)
                tp.rt_src_tuple_out[1].set_as(0)
                tp.rt_src_tuple_out[2].set_as(0)

            with tik_inst.if_scope(tik.any(tp.src_axis_perm == 0x102, tp.src_axis_perm == 0x201)):
                tp.rt_src_tuple_out[0].set_as(0)
                tp.rt_src_tuple_out[1].set_as(tp.rt_logic_tuple[0] * tp.src_jump_major_step)
                tp.rt_src_tuple_out[2].set_as(0)

            with tik_inst.if_scope(tik.any(tp.src_axis_perm == 0x012, tp.src_axis_perm == 0x021)):
                tp.rt_src_tuple_out[0].set_as(0)
                tp.rt_src_tuple_out[1].set_as(0)
                tp.rt_src_tuple_out[2].set_as(tp.rt_logic_tuple[0] * tp.src_jump_major_step)

        with self.tik_inst.if_scope(tp.src_axis_num_no_dup == 2):

            with tik_inst.if_scope(tik.any(tp.src_axis_perm == 0x10,
                                           tp.src_axis_perm == 0x102,
                                           tp.src_axis_perm == 0x201)):
                tp.rt_src_tuple_out[0].set_as(tp.rt_logic_tuple[0] * tp.src_jump_major_step)
                tp.rt_src_tuple_out[1].set_as(0)

            with tik_inst.if_scope(tik.any(tp.src_axis_perm == 0x01,
                                           tp.src_axis_perm == 0x021,
                                           tp.src_axis_perm == 0x012)):
                tp.rt_src_tuple_out[0].set_as(0)
                tp.rt_src_tuple_out[1].set_as(tp.rt_logic_tuple[0] * tp.src_jump_major_step)

            with tik_inst.if_scope(tik.any(tp.src_axis_perm == 0x120, tp.src_axis_perm == 0x210)):
                tp.rt_src_tuple_out[0].set_as(0)
                tp.rt_src_tuple_out[1].set_as(0)

        with self.tik_inst.if_scope(tp.src_axis_num_no_dup == 1):

            with self.tik_inst.if_scope(tik.any(tp.src_axis_perm == 0x0,
                                                tp.src_axis_perm == 0x01,
                                                tp.src_axis_perm == 0x021)):
                tp.rt_src_tuple_out[0].set_as(tp.rt_logic_tuple[0] * tp.src_jump_major_step)
            with self.tik_inst.else_scope():
                tp.rt_src_tuple_out[0].set_as(0)

    def _init_tail_src_tuple_copy_out_s5(self, tp):
        tik_inst = self.tik_inst
        with self.tik_inst.if_scope(tp.src_axis_num_no_dup == 3):

            with tik_inst.if_scope(tik.any(tp.src_axis_perm == 0x120, tp.src_axis_perm == 0x210)):
                tp.rt_src_tuple_out[0].set_as((tp.rt_logic_tuple[0] + 1) * tp.src_jump_major_step)
                tp.rt_src_tuple_out[1].set_as(0)
                tp.rt_src_tuple_out[2].set_as(0)

            with tik_inst.if_scope(tik.any(tp.src_axis_perm == 0x102, tp.src_axis_perm == 0x201)):
                tp.rt_src_tuple_out[0].set_as(0)
                tp.rt_src_tuple_out[1].set_as((tp.rt_logic_tuple[0] + 1) * tp.src_jump_major_step)
                tp.rt_src_tuple_out[2].set_as(0)

            with tik_inst.if_scope(tik.any(tp.src_axis_perm == 0x021, tp.src_axis_perm == 0x012)):
                tp.rt_src_tuple_out[0].set_as(0)
                tp.rt_src_tuple_out[1].set_as(0)
                tp.rt_src_tuple_out[2].set_as((tp.rt_logic_tuple[0] + 1) * tp.src_jump_major_step)

        with self.tik_inst.if_scope(tp.src_axis_num_no_dup == 2):

            with tik_inst.if_scope(tik.any(tp.src_axis_perm == 0x10,
                                           tp.src_axis_perm == 0x102,
                                           tp.src_axis_perm == 0x201)):
                tp.rt_src_tuple_out[0].set_as((tp.rt_logic_tuple[0] + 1) * tp.src_jump_major_step)
                tp.rt_src_tuple_out[1].set_as(0)

            with tik_inst.if_scope(tik.any(tp.src_axis_perm == 0x01,
                                           tp.src_axis_perm == 0x021,
                                           tp.src_axis_perm == 0x012)):
                tp.rt_src_tuple_out[0].set_as(0)
                tp.rt_src_tuple_out[1].set_as((tp.rt_logic_tuple[0] + 1) * tp.src_jump_major_step)

            with tik_inst.if_scope(tik.any(tp.src_axis_perm == 0x120, tp.src_axis_perm == 0x210)):
                tp.rt_src_tuple_out[0].set_as(0)
                tp.rt_src_tuple_out[1].set_as(0)

        with self.tik_inst.if_scope(tp.src_axis_num_no_dup == 1):

            with self.tik_inst.if_scope(tik.any(tp.src_axis_perm == 0x0,
                                                tp.src_axis_perm == 0x01,
                                                tp.src_axis_perm == 0x021)):
                tp.rt_src_tuple_out[0].set_as((tp.rt_logic_tuple[0] + 1) * tp.src_jump_major_step)
            with self.tik_inst.else_scope():
                tp.rt_src_tuple_out[0].set_as(0)

    def _init_major_dst_tuple_copy_in_s5(self, tp):
        tik_inst = self.tik_inst

        with tik_inst.if_scope(tp.dst_axis_num_no_dup == 3):

            with tik_inst.if_scope(tik.any(tp.dst_axis_perm == 0x012, tp.dst_axis_perm == 0x021)):
                tp.rt_dst_tuple_in[0].set_as(0)
                tp.rt_dst_tuple_in[1].set_as(0)
                tp.rt_dst_tuple_in[2].set_as(tp.rt_logic_tuple[1] * tp.dst_jump_major_step)

            with tik_inst.if_scope(tik.any(tp.dst_axis_perm == 0x102, tp.dst_axis_perm == 0x120)):
                tp.rt_dst_tuple_in[0].set_as(0)
                tp.rt_dst_tuple_in[1].set_as(tp.rt_logic_tuple[1] * tp.dst_jump_major_step)
                tp.rt_dst_tuple_in[2].set_as(0)

            with tik_inst.if_scope(tik.any(tp.dst_axis_perm == 0x210, tp.dst_axis_perm == 0x201)):
                tp.rt_dst_tuple_in[0].set_as(tp.rt_logic_tuple[1] * tp.dst_jump_major_step)
                tp.rt_dst_tuple_in[1].set_as(0)
                tp.rt_dst_tuple_in[2].set_as(0)

        with tik_inst.if_scope(tp.dst_axis_num_no_dup == 2):

            with tik_inst.if_scope(tik.any(tp.dst_axis_perm == 0x10,
                                           tp.dst_axis_perm == 0x102,
                                           tp.dst_axis_perm == 0x120)):
                tp.rt_dst_tuple_in[0].set_as(tp.rt_logic_tuple[1] * tp.dst_jump_major_step)
                tp.rt_dst_tuple_in[1].set_as(0)

            with tik_inst.if_scope(tik.any(tp.dst_axis_perm == 0x01,
                                           tp.dst_axis_perm == 0x012,
                                           tp.dst_axis_perm == 0x021)):
                tp.rt_dst_tuple_in[0].set_as(0)
                tp.rt_dst_tuple_in[1].set_as(tp.rt_logic_tuple[1] * tp.dst_jump_major_step)

            with tik_inst.if_scope(tik.any(tp.dst_axis_perm == 0x201, tp.dst_axis_perm == 0x210)):
                tp.rt_dst_tuple_in[0].set_as(0)
                tp.rt_dst_tuple_in[1].set_as(0)

        with tik_inst.if_scope(tp.dst_axis_num_no_dup == 1):

            with self.tik_inst.if_scope(tik.any(tp.dst_axis_perm == 0x0,
                                                tp.dst_axis_perm == 0x01,
                                                tp.dst_axis_perm == 0x012,
                                                tp.dst_axis_perm == 0x021)):
                tp.rt_dst_tuple_in[0].set_as(tp.rt_logic_tuple[1] * tp.dst_jump_major_step)
            with self.tik_inst.else_scope():
                tp.rt_dst_tuple_in[0].set_as(0)

    def _init_tail_dst_tuple_copy_in_s5(self, tp):
        tik_inst = self.tik_inst
        with tik_inst.if_scope(tp.dst_axis_num_no_dup == 3):

            with tik_inst.if_scope(tik.any(tp.dst_axis_perm == 0x012, tp.dst_axis_perm == 0x021)):
                tp.rt_dst_tuple_in[0].set_as(0)
                tp.rt_dst_tuple_in[1].set_as(0)
                tp.rt_dst_tuple_in[2].set_as((tp.rt_logic_tuple[1] + 1) * tp.dst_jump_major_step)

            with tik_inst.if_scope(tik.any(tp.dst_axis_perm == 0x102, tp.dst_axis_perm == 0x120)):
                tp.rt_dst_tuple_in[0].set_as(0)
                tp.rt_dst_tuple_in[1].set_as((tp.rt_logic_tuple[1] + 1) * tp.dst_jump_major_step)
                tp.rt_dst_tuple_in[2].set_as(0)

            with tik_inst.if_scope(tik.any(tp.dst_axis_perm == 0x210, tp.dst_axis_perm == 0x201)):
                tp.rt_dst_tuple_in[0].set_as((tp.rt_logic_tuple[1] + 1) * tp.dst_jump_major_step)
                tp.rt_dst_tuple_in[1].set_as(0)
                tp.rt_dst_tuple_in[2].set_as(0)

        with self.tik_inst.if_scope(tp.dst_axis_num_no_dup == 2):

            with tik_inst.if_scope(tik.any(tp.dst_axis_perm == 0x10,
                                           tp.dst_axis_perm == 0x102,
                                           tp.dst_axis_perm == 0x120)):
                tp.rt_dst_tuple_in[0].set_as((tp.rt_logic_tuple[1] + 1) * tp.dst_jump_major_step)
                tp.rt_dst_tuple_in[1].set_as(0)

            with tik_inst.if_scope(tik.any(tp.dst_axis_perm == 0x01,
                                           tp.dst_axis_perm == 0x012,
                                           tp.dst_axis_perm == 0x021)):
                tp.rt_dst_tuple_in[0].set_as(0)
                tp.rt_dst_tuple_in[1].set_as((tp.rt_logic_tuple[1] + 1) * tp.dst_jump_major_step)

            with tik_inst.if_scope(tik.any(tp.dst_axis_perm == 0x201, tp.dst_axis_perm == 0x210)):
                tp.rt_dst_tuple_in[0].set_as(0)
                tp.rt_dst_tuple_in[1].set_as(0)

        with self.tik_inst.if_scope(tp.dst_axis_num_no_dup == 1):

            with self.tik_inst.if_scope(tik.any(tp.dst_axis_perm == 0x0,
                                                tp.dst_axis_perm == 0x01,
                                                tp.dst_axis_perm == 0x012,
                                                tp.dst_axis_perm == 0x021)):
                tp.rt_dst_tuple_in[0].set_as((tp.rt_logic_tuple[1] + 1) * tp.dst_jump_major_step)
            with self.tik_inst.else_scope():
                tp.rt_dst_tuple_in[0].set_as(0)

    def _copy_in_major_src_major_dst_s5(self, tp, ub_input):
        tik_inst = self.tik_inst
        tp.ub_offset.set_as(0)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.major_dst_loop_in) as i:
                self._get_src_addr_s5(tp, tp.src_addr, 0)
                ub_offset = i * tp.major_burst_len_in * EPB32 // self.b8_times
                self._do_data_move_no_stride(ub_input[ub_offset], self.data_in[tp.src_addr],
                                             1, tp.major_burst_len_in, tp.major_in_ele)
                self._update_major_dst_tuple_in_s5(tp)
                tp.ub_offset.set_as(tp.ub_offset + tp.major_burst_len_in)

    def _copy_in_tail_src_major_dst_s5(self, tp, ub_input):
        tik_inst = self.tik_inst
        tp.ub_offset.set_as(0)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.major_dst_loop_in) as i:
                self._get_src_addr_s5(tp, tp.src_addr, 1)
                ub_offset = i * tp.tail_burst_len_in * EPB32 // self.b8_times
                self._do_data_move_no_stride(ub_input[ub_offset], self.data_in[tp.src_addr],
                                             1, tp.tail_burst_len_in, tp.tail_in_ele)
                self._update_major_dst_tuple_in_s5(tp)
                tp.ub_offset.set_as(tp.ub_offset + tp.tail_burst_len_in)

    def _copy_in_major_src_tail_dst_s5(self, tp, ub_input):
        tik_inst = self.tik_inst
        tp.ub_offset.set_as(0)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.tail_dst_loop_in) as i:
                self._get_src_addr_s5(tp, tp.src_addr, 0)
                ub_offset = i * tp.major_burst_len_in * EPB32 // self.b8_times
                self._do_data_move_no_stride(ub_input[ub_offset], self.data_in[tp.src_addr],
                                             1, tp.major_burst_len_in, tp.major_in_ele)
                self._update_tail_dst_tuple_in_s5(tp)
                tp.ub_offset.set_as(tp.ub_offset + tp.major_burst_len_in)

    def _copy_in_tail_src_tail_dst_s5(self, tp, ub_input):
        tik_inst = self.tik_inst
        tp.ub_offset.set_as(0)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.tail_dst_loop_in) as i:
                self._get_src_addr_s5(tp, tp.src_addr, 1)
                ub_offset = i * tp.tail_burst_len_in * EPB32 // self.b8_times
                self._do_data_move_no_stride(ub_input[ub_offset], self.data_in[tp.src_addr],
                                             1, tp.tail_burst_len_in, tp.tail_in_ele)
                self._update_tail_dst_tuple_in_s5(tp)
                tp.ub_offset.set_as(tp.ub_offset + tp.tail_burst_len_in)


    def _copy_out_common_s5(self, tp, ub_input, loop, burst_len, x_out_ele, x_out_tail_ele):
        tik_inst = self.tik_inst
        ub_tail_offset = tik_inst.Scalar("int32")
        ub_offset_tmp = tik_inst.Scalar("int32")
        ub_offset_tmp.set_as(tp.ub_src_offset * burst_len * EPB32 // self.b8_times)
        ub_tail_offset.set_as(loop % 32 * self.ele_per_block)
        with tik_inst.if_scope(x_out_tail_ele == 0):
            tik_inst.data_move(self.data_out[tp.dst_addr],
                               ub_input[tp.ub_res_addr + ub_offset_tmp],
                               0, 1, burst_len, 0, 0)
        with tik_inst.else_scope():
            if self.is_mte_pad:
                self.m_ub_input = ub_input
                if self.b8_times == EPB8:
                    self.m_ub_input = ub_input.reinterpret_cast_to("int8")
                    self.m_data_out = self.data_out.reinterpret_cast_to("int8")
                    self.m_offset_factor = EPB8
                tik_inst.data_move_pad(self.m_data_out[tp.dst_addr * self.m_offset_factor],
                                       self.m_ub_input[(tp.ub_res_addr + ub_offset_tmp) * self.m_offset_factor],
                                       1, (burst_len - 1) * BLOCK_SIZE + x_out_tail_ele * self.b8_times, 0, 0)
            else:
                tik_inst.data_move(self.data_out[tp.dst_addr],
                                   ub_input[tp.ub_res_addr + ub_offset_tmp], 0, 1, burst_len - 1, 0, 0)
                ub_input_tail = self.ub_input_b64_helper.reinterpret_cast_to(self.x_dtype)
                svalue = self.tik_inst.Scalar(self.x_dtype)
                with self.tik_inst.for_range(0, self.ele_per_block) as i:
                    svalue.set_as(ub_input[tp.ub_res_addr + i + ub_offset_tmp + x_out_ele - self.ele_per_block])
                    ub_input_tail[ub_tail_offset + i] = svalue
                tik_inst.data_move(self.data_out[tp.dst_addr + x_out_ele - self.ele_per_block],
                                   ub_input_tail[ub_tail_offset], 0, 1, 1, 0, 0)

    def _copy_out_major_src_major_dst_s5(self, tp, ub_input):
        tik_inst = self.tik_inst
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.major_src_loop_out) as i:
                self._get_dst_addr_s5(tp, 0)
                self._get_ub_src_offset_s5(tp)
                self._copy_out_common_s5(tp, ub_input, i, tp.major_burst_len_out,
                                         tp.major_out_ele, tp.major_out_tail_ele)
                self._update_major_src_tuple_out_s5(tp)

    def _copy_out_tail_src_major_dst_s5(self, tp, ub_input):
        tik_inst = self.tik_inst
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.tail_src_loop_out) as i:
                self._get_dst_addr_s5(tp, 0)
                self._get_ub_src_offset_s5(tp)
                self._copy_out_common_s5(tp, ub_input, i, tp.major_burst_len_out,
                                         tp.major_out_ele, tp.major_out_tail_ele)
                self._update_tail_src_tuple_out_s5(tp)

    def _copy_out_major_src_tail_dst_s5(self, tp, ub_input):
        tik_inst = self.tik_inst
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.major_src_loop_out) as i:
                self._get_dst_addr_s5(tp, 1)
                self._get_ub_src_offset_s5(tp)
                self._copy_out_common_s5(tp, ub_input, i, tp.tail_burst_len_out, tp.tail_out_ele, tp.tail_out_tail_ele)
                self._update_major_src_tuple_out_s5(tp)

    def _copy_out_tail_src_tail_dst_s5(self, tp, ub_input):
        tik_inst = self.tik_inst
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with tik_inst.for_range(0, tp.tail_src_loop_out) as i:
                self._get_dst_addr_s5(tp, 1)
                self._get_ub_src_offset_s5(tp)
                self._copy_out_common_s5(tp, ub_input, i, tp.tail_burst_len_out, tp.tail_out_ele, tp.tail_out_tail_ele)
                self._update_tail_src_tuple_out_s5(tp)

    def _reorder_b8_s5_data_move(self, tp, ub_input, idx):
        tik_inst = self.tik_inst
        ub_input_b8 = ub_input.reinterpret_cast_to("int8")
        with tik_inst.for_range(0, tp.n[0][idx]) as i:
            with tik_inst.new_stmt_scope(disable_sync=True):
                with tik_inst.for_range(0, tp.loop[0][idx]) as j:
                    tik_inst.data_move(ub_input_b8[(tp.offset_b * 2 + \
                                                     i * tp.vol[0][idx] + j * tp.dst_offset[0][idx]) * self.fp16_times],
                                       ub_input_b8[(tp.offset_a * 2 + \
                                                     i * tp.vol[0][idx] + j * tp.src_offset[0][idx]) * self.fp16_times],
                                       0,
                                       tp.repeat[0][idx],
                                       tp.burst_len[0][idx],
                                       tp.src_stride[0][idx],
                                       tp.dst_stride[0][idx])
        with tik_inst.if_scope(tik.all(tp.n[0][idx] > 0, tp.loop[0][idx] > 0)):
            self._swap(tp)

        with tik_inst.for_range(0, tp.n[1][idx]) as i:
            with tik_inst.new_stmt_scope(disable_sync=True):
                with tik_inst.for_range(0, tp.loop[1][idx]) as j:
                    tik_inst.data_move(ub_input_b8[(tp.offset_b * 2 + \
                                                     i * tp.vol[1][idx] + j * tp.dst_offset[1][idx]) * self.fp16_times],
                                       ub_input_b8[(tp.offset_a * 2 + \
                                                     i * tp.vol[1][idx] + j * tp.src_offset[1][idx]) * self.fp16_times],
                                       0,
                                       tp.repeat[1][idx],
                                       tp.burst_len[1][idx],
                                       tp.src_stride[1][idx],
                                       tp.dst_stride[1][idx])
        with tik_inst.if_scope(tik.all(tp.n[1][idx] > 0, tp.loop[1][idx] > 0)):
            self._swap(tp)

        with tik_inst.for_range(0, tp.n[2][idx]) as i:
            with tik_inst.new_stmt_scope(disable_sync=True):
                with tik_inst.for_range(0, tp.loop[2][idx]) as j:
                    tik_inst.data_move(ub_input_b8[(tp.offset_b * 2 + \
                                                     i * tp.vol[2][idx] + j * tp.dst_offset[2][idx]) * self.fp16_times],
                                       ub_input_b8[(tp.offset_a * 2 + \
                                                     i * tp.vol[2][idx] + j * tp.src_offset[2][idx]) * self.fp16_times],
                                       0,
                                       tp.repeat[2][idx],
                                       tp.burst_len[2][idx],
                                       tp.src_stride[2][idx],
                                       tp.dst_stride[2][idx])
        with tik_inst.if_scope(tik.all(tp.n[2][idx] > 0, tp.loop[2][idx] > 0)):
            self._swap(tp)

        tp.ub_res_addr.set_as(tp.offset_a)

    def _reorder_s5_loop1_dst_stride_0_vor_main(self, tp, ub_input_b16, idx, i, j, k):
        tik_inst = self.tik_inst
        tik_inst.vor(128,
                     ub_input_b16[tp.offset_b + i * tp.vol[k][idx] + j * tp.repeat[k][idx] * self.ele_per_block],
                     ub_input_b16[tp.offset_a + i * tp.vol[k][idx] + j * self.ele_per_block],
                     self.ub_input_b16_vor,
                     tp.repeat[k][idx] // 8,
                     1, tp.loop[k][idx], 0, 8, 8 * tp.loop[k][idx], 0)

    def _reorder_s5_loop1_dst_stride_0_ub2ub_tail(self, tp, ub_input_b16, idx, tail_num, i, j, k):
        tik_inst = self.tik_inst
        tail_num.set_as(tp.repeat[k][idx] % 8)
        with tik_inst.if_scope(tail_num != 0):
            tik_inst.data_move(ub_input_b16[tp.offset_b + i * tp.vol[k][idx] + j * tp.dst_offset[k][idx] + \
                                            (tp.repeat[k][idx] - tail_num) * self.ele_per_block],
                               ub_input_b16[tp.offset_a + i * tp.vol[k][idx] + j * tp.src_offset[k][idx] + \
                                            (tp.repeat[k][idx] - tail_num) * tp.loop[k][idx] * self.ele_per_block],
                               0,
                               tp.repeat[k][idx] % 8,
                               1,
                               tp.src_stride[k][idx],
                               tp.dst_stride[k][idx])

    def _reorder_s5_loop1_src_stride_0_vor_main(self, tp, ub_input_b16, idx, i, j, k):
        tik_inst = self.tik_inst
        tik_inst.vor(128,
                     ub_input_b16[tp.offset_b + i * tp.vol[k][idx] + j * self.ele_per_block],
                     ub_input_b16[tp.offset_a + i * tp.vol[k][idx] + j * tp.repeat[k][idx] * self.ele_per_block],
                     self.ub_input_b16_vor,
                     tp.repeat[k][idx] // 8,
                     tp.loop[k][idx], 1, 0, 8 * tp.loop[k][idx], 8, 0)

    def _reorder_s5_loop1_src_stride_0_ub2ub_tail(self, tp, ub_input_b16, idx, tail_num, i, j, k):
        tik_inst = self.tik_inst
        tail_num.set_as(tp.repeat[k][idx] % 8)
        with tik_inst.if_scope(tail_num != 0):
            tik_inst.data_move(ub_input_b16[tp.offset_b + i * tp.vol[k][idx] + j * tp.dst_offset[k][idx] + \
                                            (tp.repeat[k][idx] - tail_num) * tp.loop[k][idx] * self.ele_per_block],
                               ub_input_b16[tp.offset_a + i * tp.vol[k][idx] + j * tp.src_offset[k][idx] + \
                                            (tp.repeat[k][idx] - tail_num) * self.ele_per_block],
                               0,
                               tail_num,
                               1,
                               tp.src_stride[k][idx],
                               tp.dst_stride[k][idx])

    def _reorder_s5_loop1_ub2ub(self, tp, ub_input_b16, idx, i, j, k):
        tik_inst = self.tik_inst
        tik_inst.data_move(ub_input_b16[(tp.offset_b + \
                                        i * tp.vol[k][idx] + j * tp.dst_offset[k][idx]) * self.fp16_times],
                           ub_input_b16[(tp.offset_a + \
                                        i * tp.vol[k][idx] + j * tp.src_offset[k][idx]) * self.fp16_times],
                           0,
                           tp.repeat[k][idx],
                           tp.burst_len[k][idx],
                           tp.src_stride[k][idx],
                           tp.dst_stride[k][idx])

    def _reorder_s5_data_move(self, tp, ub_input, idx):
        tik_inst = self.tik_inst
        ub_input_b16 = ub_input.reinterpret_cast_to("int16")
        tail_num = tik_inst.Scalar("int32")

        for k in range(BORROW_SRC_AXIS_LT_NUM):
            with tik_inst.if_scope(tik.any(self.fp16_times != 1, tp.burst_len[k][idx] != 1,
                                           tp.repeat[k][idx] <= 24, tp.loop[k][idx] >= 32)):
                # repeat <= 24 ub2ub is better than vor
                # loop >= 32 will result in stride gt 255
                with tik_inst.new_stmt_scope(disable_sync=True):
                    with tik_inst.for_range(0, tp.n[k][idx]) as i:
                        with tik_inst.for_range(0, tp.loop[k][idx]) as j:
                            self._reorder_s5_loop1_ub2ub(tp, ub_input_b16, idx, i, j, k)
            with tik_inst.else_scope():
                with tik_inst.new_stmt_scope(disable_sync=True):
                    with tik_inst.for_range(0, tp.n[k][idx]) as i:
                        with tik_inst.for_range(0, tp.loop[k][idx]) as j:
                            with tik_inst.if_scope(tp.dst_stride[k][idx] == 0):
                                self._reorder_s5_loop1_dst_stride_0_vor_main(tp, ub_input_b16,
                                                                             idx, i, j, k)
                                self._reorder_s5_loop1_dst_stride_0_ub2ub_tail(tp, ub_input_b16,
                                                                               idx, tail_num, i, j, k)
                            with tik_inst.else_scope():
                                self._reorder_s5_loop1_src_stride_0_vor_main(tp, ub_input_b16, idx, i, j, k)
                                self._reorder_s5_loop1_src_stride_0_ub2ub_tail(tp, ub_input_b16,
                                                                               idx, tail_num, i, j, k)

            with tik_inst.if_scope(tik.all(tp.n[k][idx] > 0, tp.loop[k][idx] > 0)):
                self._swap(tp)

        tp.ub_res_addr.set_as(tp.offset_a)

    def _make_ualigned_be_head_of_block_b8_s5(self, tp, ub_input, x_in_ele, x_in_tail_ele, x_dst_loop_in):
        tik_inst = self.tik_inst
        with tik_inst.if_scope(tik.any(tp.is_last_axis_transpose != 0, tp.align_ele != 0)):
            ub_input_b8 = ub_input.reinterpret_cast_to("int8")
            src_ele_num_in_b8 = self._get_src_size() * 2  # avoid bank conflict
            src_list = [ub_input_b8[src_ele_num_in_b8 * i] for i in range(EPB16)]
            dst_list_low = [ub_input_b8[tp.offset_b * 2 + EPB32 * i] for i in range(EPB16)]
            dst_list_high = [ub_input_b8[tp.offset_b * 2 + EPB32 * i + EPB32 * EPB16] for i in range(EPB16)]

            with tik_inst.if_scope(tp.ub_offset != 1):
                tik_inst.vnchwconv(False, False, dst_list_low, src_list, tp.ub_offset, EPB32, 1)
                tik_inst.vnchwconv(False, True, dst_list_high, src_list, tp.ub_offset, EPB32, 1)
            with tik_inst.else_scope():
                tik_inst.vnchwconv(False, False, dst_list_low, src_list, 1, 0, 0)
                tik_inst.vnchwconv(False, True, dst_list_high, src_list, 1, 0, 0)

            self._swap(tp)

            # eliminate dirty data between two in_blocks
            with tik_inst.if_scope(x_in_tail_ele != 0):
                tik_inst.data_move(ub_input_b8[tp.offset_b * 2],
                                   ub_input_b8[tp.offset_a * 2],
                                   0,
                                   x_dst_loop_in,
                                   x_in_ele,
                                   (self.ele_per_block - x_in_tail_ele), # src_stride
                                   0) # dst_stride
                self._swap(tp)

    def _make_ualigned_be_head_of_block_s5(self, tp, ub_input, x_in_ele, x_in_tail_ele, x_dst_loop_in):
        tik_inst = self.tik_inst
        with tik_inst.if_scope(tik.any(tp.is_last_axis_transpose == 1, tp.align_ele != 0)):
            ub_input_b16 = ub_input.reinterpret_cast_to("int16")
            src_ele_num_in_b16 = self._get_src_size()  # avoid bank conflict
            src_list = [ub_input_b16[tp.offset_a * self.fp16_times + src_ele_num_in_b16 * i] for i in range(EPB16)]
            dst_list = [ub_input_b16[tp.offset_b * self.fp16_times + EPB16 * i] for i in range(EPB16)]
            tik_inst.vnchwconv(False, False, dst_list, src_list, tp.ub_offset, EPB16, 1)
            self._swap(tp)

            #eliminate dirty data between two in_blocks
            with tik_inst.if_scope(x_in_tail_ele != 0):
                tik_inst.data_move(ub_input_b16[tp.offset_b * self.fp16_times],
                                   ub_input_b16[tp.offset_a * self.fp16_times],
                                   0,
                                   x_dst_loop_in,
                                   x_in_ele * self.fp16_times,
                                   (self.ele_per_block - x_in_tail_ele) * self.fp16_times, # src_stride
                                   0) # dst_stride
                self._swap(tp)

    def _make_block_head_be_contiguous_b8_s5(self, tp, ub_input, x_out_ele, x_out_tail_ele,
                                             burst_len_out, x_src_loop_out):
        tik_inst = self.tik_inst
        with tik_inst.if_scope(tik.any(tp.is_last_axis_transpose != 0, tp.align_ele != 0)):
            ub_input_b8 = ub_input.reinterpret_cast_to("int8")
            # 1. insert data between two in_blocks to make each out_blocks be started with block align
            with tik_inst.if_scope(x_out_tail_ele != 0):
                tik_inst.data_move(ub_input_b8[tp.offset_b * 2],
                                   ub_input_b8[tp.offset_a * 2],
                                   0,
                                   x_src_loop_out,
                                   x_out_ele,
                                   0,
                                   (self.ele_per_block - x_out_tail_ele))
                self._swap(tp)

            # 2. make block head be line
            ub_offset_exclude_pad = 60
            src_list_low = [ub_input_b8[tp.offset_a * 2 + EPB32 * i] for i in range(EPB16)]
            src_list_high = [ub_input_b8[tp.offset_a * 2 + EPB32 * i + EPB32 * EPB16] for i in range(EPB16)]
            dst_list = [ub_input_b8[tp.offset_b * 2 + self._get_dst_size() * EPB32 * i] for i in range(EPB16)]
            with tik_inst.if_scope(x_src_loop_out * burst_len_out != 1):
                self.tik_inst.vnchwconv(False, False, dst_list, src_list_low, x_src_loop_out * burst_len_out, 1, EPB32)
                self.tik_inst.vnchwconv(True, False, dst_list, src_list_high, x_src_loop_out * burst_len_out, 1, EPB32)
            with tik_inst.else_scope():
                self.tik_inst.vnchwconv(False, False, dst_list, src_list_low, 1, 0, 0)
                self.tik_inst.vnchwconv(True, False, dst_list, src_list_high, 1, 0, 0)
            self._swap(tp)

    def _make_block_head_be_contiguous_s5(self, tp, ub_input, x_out_ele, x_out_tail_ele, burst_len_out, x_src_loop_out):
        tik_inst = self.tik_inst
        with tik_inst.if_scope(tik.any(tp.is_last_axis_transpose == 1, tp.align_ele != 0)):
            ub_input_b16 = ub_input.reinterpret_cast_to("int16")
            # insert data between two in_blocks to make each out_blocks be started with block align
            with tik_inst.if_scope(tik.all(x_out_tail_ele != 0, x_src_loop_out > 1)):
                tik_inst.data_move(ub_input_b16[tp.offset_b * self.fp16_times],
                                   ub_input_b16[tp.offset_a * self.fp16_times],
                                   0,
                                   x_src_loop_out,
                                   x_out_ele * self.fp16_times,
                                   0,
                                   (self.ele_per_block - x_out_tail_ele) * self.fp16_times)
                self._swap(tp)

            # make block head be line
            src_list = [ub_input_b16[tp.offset_a * self.fp16_times + EPB16 * i] for i in range(EPB16)]
            dst_list = [ub_input_b16[tp.offset_b * self.fp16_times + self._get_dst_size() * EPB16 * i] \
                        for i in range(EPB16)]
            tik_inst.vnchwconv(False, False, dst_list, src_list, x_src_loop_out * burst_len_out, 1, EPB16)
            self._swap(tp)

    def _reorder_by_vgather_s5_s7(self, tp, params):
        """
        reorder data to target layout by vgather for scenario 5
        """
        ub_input, dst_offset, idx_offset, idx, x_out_ele = params
        ub_input_int32 = ub_input.reinterpret_cast_to("int32")
        self.tik_inst.vgather(x_out_ele,
                              ub_input[dst_offset],
                              ub_input,
                              ub_input_int32[tp.src_offset_base_addr + idx * idx_offset],
                              1, 8, 0, mask_mode="counter")

    def _reorder_s5(self, tp, ub_input, is_src_tail_in, is_dst_tail_in,
                    x_in_ele, x_in_tail_ele, burst_len_in, x_dst_loop_in,
                    x_out_ele, x_out_tail_ele, burst_len_out, x_src_loop_out):
        tik_inst = self.tik_inst
        idx = tik_inst.Scalar("int32")
        tp.offset_a.set_as(tp.offset_1 // self.fp16_times)
        tp.offset_b.set_as(tp.offset_2 // self.fp16_times)
        self._get_reorder_idx(is_src_tail_in, is_dst_tail_in, idx)

        if self.x_dtype in ("int8", "uint8", "bool"):
            self._make_ualigned_be_head_of_block_b8_s5(tp, ub_input, x_in_ele, x_in_tail_ele, x_dst_loop_in)
            self._reorder_b8_s5_data_move(tp, ub_input, idx)
            self._make_block_head_be_contiguous_b8_s5(tp, ub_input, x_out_ele, x_out_tail_ele,
                                                      burst_len_out, x_src_loop_out)
            tp.ub_res_addr.set_as(tp.offset_a * 2)
        else:

            def _reorder_by_vnchwconv_process():
                self._make_ualigned_be_head_of_block_s5(tp, ub_input, x_in_ele, x_in_tail_ele, x_dst_loop_in)
                self._reorder_s5_data_move(tp, ub_input, idx)
                self._make_block_head_be_contiguous_s5(tp, ub_input, x_out_ele, x_out_tail_ele,
                                                    burst_len_out, x_src_loop_out)
                tp.ub_res_addr.set_as(tp.offset_a)
            if self.vg_cond is True:
                with tik_inst.if_scope(tp.scenario == 14):
                    params = (ub_input, tp.offset_b,
                              tp.src_offset_addr_gap,
                              idx, x_src_loop_out * ceil_align(x_out_ele, self.ele_per_block))
                    self._reorder_by_vgather_s5_s7(tp, params)
                    tp.ub_res_addr.set_as(tp.offset_b)
                with tik_inst.else_scope():
                    _reorder_by_vnchwconv_process()
            else:
                _reorder_by_vnchwconv_process()

    def _gen_src_offset_process_s5(self, tp, ub_input, reorder_idx, conv_params):
        """
        process for generating src_offset for scenario 5
        """
        concur_cnt = 8
        tik_inst = self.tik_inst
        idx = tik_inst.Scalar("int32", init_value=0)
        tmp_off = tik_inst.Scalar("int32")
        ub_input_int32 = ub_input.reinterpret_cast_to("int32")
        major_dst_major_src_base_addr = tp.src_offset_base_addr
        major_dst_tail_src_base_addr = major_dst_major_src_base_addr + tp.src_offset_addr_gap
        tail_dst_major_src_base_addr = major_dst_tail_src_base_addr + tp.src_offset_addr_gap
        tail_dst_tail_src_base_addr = tail_dst_major_src_base_addr + tp.src_offset_addr_gap
        base_addr_tuple = (major_dst_major_src_base_addr, major_dst_tail_src_base_addr,
                           tail_dst_major_src_base_addr, tail_dst_tail_src_base_addr)

        def _inner_process(base_addr, params):
            tgt_axis_1, tgt_axis_2, tgt_axis_3, tgt_axis_4, tgt_axis_5 = params[0]
            tgt_axis_1_off, tgt_axis_2_off, tgt_axis_3_off, tgt_axis_4_off, tgt_axis_5_off = params[1]

            with tik_inst.for_range(0, 1) as axis_0_idx:
                with tik_inst.for_range(0, tgt_axis_1) as axis_1_idx:
                    with tik_inst.for_range(0, tgt_axis_2) as axis_2_idx:
                        with tik_inst.for_range(0, tgt_axis_3) as axis_3_idx:
                            with tik_inst.for_range(0, tgt_axis_4) as axis_4_idx:
                                with tik_inst.for_range(0, tgt_axis_5) as axis_5_idx:
                                    lp_cnt = tp.last_axis_len // concur_cnt
                                    with tik_inst.for_range(0, lp_cnt) as last_dim_lp_idx:
                                        tmp_off.set_as((last_dim_lp_idx * concur_cnt +
                                                        axis_5_idx * tgt_axis_5_off +
                                                        axis_4_idx * tgt_axis_4_off +
                                                        axis_3_idx * tgt_axis_3_off +
                                                        axis_2_idx * tgt_axis_2_off +
                                                        axis_1_idx * tgt_axis_1_off +
                                                        axis_0_idx * 0) * self.b8_times)
                                        for i in range(concur_cnt):
                                            ub_input_int32[base_addr + idx + i].set_as(tmp_off + i * self.b8_times)
                                        idx.set_as(idx + concur_cnt)
                                    lp_left = tp.last_axis_len % concur_cnt
                                    with tik_inst.if_scope(lp_left > 0):
                                        tmp_off.set_as((lp_cnt * concur_cnt +
                                                        axis_5_idx * tgt_axis_5_off +
                                                        axis_4_idx * tgt_axis_4_off +
                                                        axis_3_idx * tgt_axis_3_off +
                                                        axis_2_idx * tgt_axis_2_off +
                                                        axis_1_idx * tgt_axis_1_off +
                                                        axis_0_idx * 0) * self.b8_times)
                                        with tik_inst.for_range(0, lp_left) as left_idx:
                                            ub_input_int32[base_addr + idx].set_as(tmp_off + left_idx * self.b8_times)
                                            idx.set_as(idx + 1)

                                    # pad src_offset to block align
                                    with tik_inst.if_scope(tp.dst_axis_perm_len == 1):
                                        out_ele = tgt_axis_5 * tp.last_axis_len
                                        pad_cnt = self.ele_per_block - out_ele % self.ele_per_block
                                        with tik_inst.if_scope(tik.all(pad_cnt < self.ele_per_block,
                                                                    axis_5_idx == tgt_axis_5 - 1)):
                                            with tik_inst.for_range(0, pad_cnt):
                                                ub_input_int32[base_addr + idx].set_as(tmp_off)
                                                idx.set_as(idx + 1)
                                    with tik_inst.elif_scope(tp.dst_axis_perm_len == 2):
                                        out_ele = tgt_axis_4 * tgt_axis_5 * tp.last_axis_len
                                        pad_cnt = self.ele_per_block - out_ele % self.ele_per_block
                                        with tik_inst.if_scope(tik.all(pad_cnt < self.ele_per_block,
                                                                    axis_5_idx == tgt_axis_5 - 1,
                                                                    axis_4_idx == tgt_axis_4 - 1)):
                                            with tik_inst.for_range(0, pad_cnt):
                                                ub_input_int32[base_addr + idx].set_as(tmp_off)
                                                idx.set_as(idx + 1)
                                    with tik_inst.else_scope():
                                        out_ele = tgt_axis_3 * tgt_axis_4 * tgt_axis_5 * tp.last_axis_len
                                        pad_cnt = self.ele_per_block - out_ele % self.ele_per_block
                                        with tik_inst.if_scope(tik.all(pad_cnt < self.ele_per_block,
                                                                    axis_5_idx == tgt_axis_5 - 1,
                                                                    axis_4_idx == tgt_axis_4 - 1,
                                                                    axis_3_idx == tgt_axis_3 - 1)):
                                            with tik_inst.for_range(0, pad_cnt):
                                                ub_input_int32[base_addr + idx].set_as(tmp_off)
                                                idx.set_as(idx + 1)
        _inner_process(base_addr_tuple[reorder_idx], conv_params)

    def _get_src_dst_axis_perm_len_s5(self, tp):
        tik_inst = self.tik_inst

        def _inner_process(axis_perm_len, axis_perm):
            with tik_inst.if_scope(axis_perm == 0x0):
                axis_perm_len.set_as(1)
            with tik_inst.elif_scope(axis_perm <= 0x10):
                axis_perm_len.set_as(2)
            with tik_inst.elif_scope(axis_perm <= 0x210):
                axis_perm_len.set_as(3)
        _inner_process(tp.src_axis_perm_len, tp.src_axis_perm)
        _inner_process(tp.dst_axis_perm_len, tp.dst_axis_perm)

    def _get_ub_axis_perm_len_s5(self, tp):
        tik_inst = self.tik_inst

        with tik_inst.if_scope(tp.ub_axis_perm == 0x10):
            tp.ub_axis_perm_len.set_as(2)
        with tik_inst.elif_scope(tp.ub_axis_perm <= 0x210):
            tp.ub_axis_perm_len.set_as(3)
        with tik_inst.elif_scope(tp.ub_axis_perm <= 0x3210):
            tp.ub_axis_perm_len.set_as(4)
        with tik_inst.elif_scope(tp.ub_axis_perm <= 0x43210):
            tp.ub_axis_perm_len.set_as(5)
        with tik_inst.elif_scope(tp.ub_axis_perm <= 0x543210):
            tp.ub_axis_perm_len.set_as(6)

    @staticmethod
    def _get_axis_idx_s5(perm, mask, rshift):
        return (perm & mask) >> rshift

    def _get_input_axis_offset_s5(self, tp, axis_offset, begin_idx, src_jump_idx):
        tik_inst = self.tik_inst

        axis_offset.set_as(tp.last_axis_len)
        with tik_inst.if_scope(begin_idx >= src_jump_idx):
            with tik_inst.for_range(begin_idx + 1, REORDER_MAX_AXIS_NUM) as b_idx:
                axis_offset.set_as(axis_offset * tp.in_ub_layout[b_idx])
        with tik_inst.else_scope():
            tp.offset_t.set_as(tp.last_axis_len)
            with tik_inst.for_range(src_jump_idx, REORDER_MAX_AXIS_NUM) as j_idx:
                tp.offset_t.set_as(tp.offset_t * tp.in_ub_layout[j_idx])
            axis_offset.set_as(ceil_align(tp.offset_t, self.ele_per_block))
            with tik_inst.for_range(begin_idx + 1, src_jump_idx) as b_idx:
                axis_offset.set_as(axis_offset * tp.in_ub_layout[b_idx])

    def _gen_vgather_src_offset_4_ub_axis_perm_len_2_s5(self, tp, ub_input):
        tik_inst = self.tik_inst

        base_idx = 3
        self._get_src_dst_axis_perm_len_s5(tp)
        src_jump_idx = base_idx + tp.ub_axis_perm_len - tp.src_axis_perm_len
        ub_perm_tup = (self._get_axis_idx_s5(tp.ub_axis_perm, 0xf0, 4),
                       self._get_axis_idx_s5(tp.ub_axis_perm, 0x0f, 0))

        def _gen_in_ub_layout():
            with tik_inst.if_scope(tp.src_axis_perm_len == 1):
                tp.in_ub_layout[base_idx + 1].set_as(tp.src_jump_major_step)
            with tik_inst.if_scope(tp.src_axis_perm_len == 2):
                src_axis_tup = (tp.src_jump_major_step, tp.second_to_last_axis_len)
                for index, axis in enumerate(src_axis_tup):
                    tp.in_ub_layout[base_idx + index].set_as(axis)

            with tik_inst.if_scope(tp.dst_axis_perm_len == 1):
                tp.in_ub_layout[base_idx].set_as(tp.dst_jump_major_step)
            with tik_inst.if_scope(tp.dst_axis_perm_len == 2):
                dst_axis_tup = (tp.dst_jump_major_step, tp.second_to_out_last_axis_len)
                for index, axis in enumerate(dst_axis_tup):
                    tp.in_ub_layout[base_idx + ub_perm_tup[index]].set_as(axis)

        def _gen_dst_offset():
            for index, perm in enumerate(ub_perm_tup):
                self._get_input_axis_offset_s5(tp, tp.in_axis_off[base_idx + index], base_idx + perm, src_jump_idx)

        def _set_tail_in_ele():
            with tik_inst.if_scope(tp.src_axis_perm_len == 1):
                tp.in_ub_layout[base_idx + 1].set_as(tp.tail_in_ele // tp.last_axis_len)
            with tik_inst.elif_scope(tp.src_axis_perm_len == 2):
                tp.in_ub_layout[base_idx].set_as(tp.tail_in_ele // tp.last_axis_len // tp.second_to_last_axis_len)

        # for case [is_src_tail == 0, is_dst_tail == 0]
        _gen_in_ub_layout()
        _gen_dst_offset()
        conv_params = [[1, 1, 1,
                        tp.in_ub_layout[base_idx + ub_perm_tup[0]], tp.in_ub_layout[base_idx + ub_perm_tup[1]]],
                       [0, 0, 0,
                        tp.in_axis_off[base_idx], tp.in_axis_off[base_idx + 1]]]
        self._gen_src_offset_process_s5(tp, ub_input, 0, conv_params)

        # for case [is_src_tail == 1, is_dst_tail == 0]
        with tik_inst.if_scope(tik.all(tp.tail_in_ele > 0, tp.pivot_src_axis_dup == 0)):
            _set_tail_in_ele()
            _gen_dst_offset()
            self._gen_src_offset_process_s5(tp, ub_input, 1, conv_params)

        # for case [is_src_tail == 0, is_dst_tail == 1]
        with tik_inst.if_scope(tik.all(tp.tail_out_ele > 0, tp.pivot_dst_axis_dup == 0)):
            with tik_inst.if_scope(tik.all(tp.tail_in_ele > 0, tp.pivot_src_axis_dup == 0)):
                with tik_inst.if_scope(tp.src_axis_perm_len == 1):
                    tp.in_ub_layout[base_idx + 1].set_as(tp.src_jump_major_step)
                with tik_inst.elif_scope(tp.src_axis_perm_len == 2):
                    tp.in_ub_layout[base_idx].set_as(tp.src_jump_major_step)
            with tik_inst.if_scope(tp.dst_axis_perm_len == 1):
                tp.in_ub_layout[base_idx + ub_perm_tup[1]].set_as(tp.tail_out_ele // tp.last_axis_len)
            with tik_inst.elif_scope(tp.dst_axis_perm_len == 2):
                tp.in_ub_layout[base_idx + ub_perm_tup[0]].set_as(
                    tp.tail_out_ele // tp.last_axis_len // tp.second_to_out_last_axis_len)
            _gen_dst_offset()
            self._gen_src_offset_process_s5(tp, ub_input, 2, conv_params)

        # for case [is_src_tail == 1, is_dst_tail == 1]
        with tik_inst.if_scope(tik.all(tp.tail_in_ele > 0, tp.tail_out_ele > 0)):
            _set_tail_in_ele()
            _gen_dst_offset()
            self._gen_src_offset_process_s5(tp, ub_input, 3, conv_params)

    def _gen_vgather_src_offset_4_ub_axis_perm_len_3_s5(self, tp, ub_input):
        tik_inst = self.tik_inst

        base_idx = 2
        self._get_src_dst_axis_perm_len_s5(tp)
        src_jump_idx = base_idx + tp.ub_axis_perm_len - tp.src_axis_perm_len
        ub_perm_tup = (self._get_axis_idx_s5(tp.ub_axis_perm, 0x000f00, 8),
                       self._get_axis_idx_s5(tp.ub_axis_perm, 0x0000f0, 4),
                       self._get_axis_idx_s5(tp.ub_axis_perm, 0x00000f, 0))

        def _gen_in_ub_layout():
            major_src_axis_0 = tp.src_jump_major_step
            major_dst_axis_0 = tp.dst_jump_major_step
            with tik_inst.if_scope(tp.src_axis_perm_len == 1):
                tp.in_ub_layout[base_idx + 2].set_as(major_src_axis_0)
            with tik_inst.if_scope(tp.src_axis_perm_len == 2):
                major_src_axis_1 = tp.second_to_last_axis_len
                src_axis_tup = (major_src_axis_0, major_src_axis_1)
                for index, axis in enumerate(src_axis_tup):
                    tp.in_ub_layout[base_idx + 1 + index].set_as(axis)
            with tik_inst.if_scope(tp.src_axis_perm_len == 3):
                major_src_axis_2 = tp.second_to_last_axis_len
                major_src_axis_1 = tp.major_in_ele // major_src_axis_0 // major_src_axis_2 // tp.last_axis_len
                src_axis_tup = (major_src_axis_0, major_src_axis_1, major_src_axis_2)
                for index, axis in enumerate(src_axis_tup):
                    tp.in_ub_layout[base_idx + index].set_as(axis)

            with tik_inst.if_scope(tp.dst_axis_perm_len == 1):
                tp.in_ub_layout[base_idx + ub_perm_tup[2]].set_as(major_dst_axis_0)
            with tik_inst.if_scope(tp.dst_axis_perm_len == 2):
                major_dst_axis_1 = tp.second_to_out_last_axis_len
                dst_axis_tup = (major_dst_axis_0, major_dst_axis_1)
                for index, axis in enumerate(dst_axis_tup):
                    tp.in_ub_layout[base_idx + ub_perm_tup[1 + index]].set_as(axis)
            with tik_inst.if_scope(tp.dst_axis_perm_len == 3):
                major_dst_axis_2 = tp.second_to_out_last_axis_len
                major_dst_axis_1 = tp.major_out_ele // major_dst_axis_0 // major_dst_axis_2 // tp.last_axis_len
                dst_axis_tup = (major_dst_axis_0, major_dst_axis_1, major_dst_axis_2)
                for index, axis in enumerate(dst_axis_tup):
                    tp.in_ub_layout[base_idx + ub_perm_tup[index]].set_as(axis)

        def _gen_dst_offset():
            with tik_inst.if_scope(tp.ub_axis_perm_len - tp.dst_axis_perm_len <= 1):
                for index, perm in enumerate(ub_perm_tup):
                    self._get_input_axis_offset_s5(tp, tp.in_axis_off[base_idx + index],
                                                   base_idx + perm, src_jump_idx)
                    tp.out_ub_layout[base_idx + index].set_as(tp.in_ub_layout[base_idx + perm])
            with tik_inst.elif_scope(tp.ub_axis_perm_len - tp.dst_axis_perm_len == 2):
                with tik_inst.if_scope(ub_perm_tup[0] > ub_perm_tup[1]):
                    self._get_input_axis_offset_s5(tp, tp.in_axis_off[base_idx],
                                                   base_idx + ub_perm_tup[1], src_jump_idx)
                    self._get_input_axis_offset_s5(tp, tp.in_axis_off[base_idx + 1],
                                                   base_idx + ub_perm_tup[0], src_jump_idx)
                    tp.out_ub_layout[base_idx].set_as(tp.in_ub_layout[base_idx + ub_perm_tup[1]])
                    tp.out_ub_layout[base_idx + 1].set_as(tp.in_ub_layout[base_idx + ub_perm_tup[0]])
                with tik_inst.else_scope():
                    self._get_input_axis_offset_s5(tp, tp.in_axis_off[base_idx],
                                                   base_idx + ub_perm_tup[0], src_jump_idx)
                    self._get_input_axis_offset_s5(tp, tp.in_axis_off[base_idx + 1],
                                                   base_idx + ub_perm_tup[1], src_jump_idx)
                    tp.out_ub_layout[base_idx].set_as(tp.in_ub_layout[base_idx + ub_perm_tup[0]])
                    tp.out_ub_layout[base_idx + 1].set_as(tp.in_ub_layout[base_idx + ub_perm_tup[1]])
                for index, perm in enumerate(ub_perm_tup[2:]):
                    self._get_input_axis_offset_s5(tp, tp.in_axis_off[base_idx + 2 + index],
                                                   base_idx + perm, src_jump_idx)
                    tp.out_ub_layout[base_idx + 2 + index].set_as(tp.in_ub_layout[base_idx + perm])

        def _set_tail_in_ele():
            with tik_inst.if_scope(tp.src_axis_perm_len == 1):
                tp.in_ub_layout[base_idx + 2].set_as(tp.tail_in_ele // tp.last_axis_len)
            with tik_inst.elif_scope(tp.src_axis_perm_len == 2):
                major_src_axis_1 = tp.second_to_last_axis_len
                tp.in_ub_layout[base_idx + 1].set_as(tp.tail_in_ele // major_src_axis_1 // tp.last_axis_len)
            with tik_inst.else_scope():
                major_src_axis_2 = tp.second_to_last_axis_len
                major_src_axis_1 = tp.major_in_ele // major_src_axis_2 // tp.last_axis_len // tp.src_jump_major_step
                tp.in_ub_layout[base_idx].set_as(
                    tp.tail_in_ele // major_src_axis_1 // major_src_axis_2 // tp.last_axis_len)

        # for case [is_src_tail == 0, is_dst_tail == 0]
        _gen_in_ub_layout()
        _gen_dst_offset()
        conv_params = [[1, 1, tp.out_ub_layout[2], tp.out_ub_layout[3], tp.out_ub_layout[4]],
                       [0, 0, tp.in_axis_off[2], tp.in_axis_off[3], tp.in_axis_off[4]]]
        self._gen_src_offset_process_s5(tp, ub_input, 0, conv_params)

        # for case [is_src_tail == 1, is_dst_tail == 0]
        with tik_inst.if_scope(tik.all(tp.tail_in_ele > 0, tp.pivot_src_axis_dup == 0)):
            _set_tail_in_ele()
            _gen_dst_offset()
            self._gen_src_offset_process_s5(tp, ub_input, 1, conv_params)

        # for case [is_src_tail == 0, is_dst_tail == 1]
        with tik_inst.if_scope(tik.all(tp.tail_out_ele > 0, tp.pivot_dst_axis_dup == 0)):
            with tik_inst.if_scope(tik.all(tp.tail_in_ele > 0, tp.pivot_src_axis_dup == 0)):
                with tik_inst.if_scope(tp.src_axis_perm_len == 1):
                    tp.in_ub_layout[base_idx + 2].set_as(tp.src_jump_major_step)
                with tik_inst.elif_scope(tp.src_axis_perm_len == 2):
                    tp.in_ub_layout[base_idx + 1].set_as(tp.src_jump_major_step)
                with tik_inst.else_scope():
                    tp.in_ub_layout[base_idx].set_as(tp.src_jump_major_step)
            with tik_inst.if_scope(tp.dst_axis_perm_len == 1):
                tp.in_ub_layout[base_idx + ub_perm_tup[2]].set_as(tp.tail_out_ele // tp.last_axis_len)
            with tik_inst.elif_scope(tp.dst_axis_perm_len == 2):
                major_dst_axis_1 = tp.second_to_out_last_axis_len
                tp.in_ub_layout[base_idx + ub_perm_tup[1]].set_as(
                    tp.tail_out_ele // major_dst_axis_1 // tp.last_axis_len)
            with tik_inst.elif_scope(tp.dst_axis_perm_len == 3):
                major_dst_axis_2 = tp.second_to_out_last_axis_len
                major_dst_axis_1 = tp.major_out_ele // major_dst_axis_2 // tp.last_axis_len // tp.dst_jump_major_step
                tp.in_ub_layout[base_idx + ub_perm_tup[0]].set_as(
                    tp.tail_out_ele // major_dst_axis_1 // major_dst_axis_2 // tp.last_axis_len)
            _gen_dst_offset()
            self._gen_src_offset_process_s5(tp, ub_input, 2, conv_params)

        # for case [is_src_tail == 1, is_dst_tail == 1]
        with tik_inst.if_scope(tik.all(tp.tail_in_ele > 0, tp.tail_out_ele > 0)):
            _set_tail_in_ele()
            _gen_dst_offset()
            self._gen_src_offset_process_s5(tp, ub_input, 3, conv_params)

    def _gen_vgather_src_offset_4_ub_axis_perm_len_4_s5(self, tp, ub_input):
        tik_inst = self.tik_inst

        base_idx = 1
        self._get_src_dst_axis_perm_len_s5(tp)
        src_jump_idx = base_idx + tp.ub_axis_perm_len - tp.src_axis_perm_len
        ub_perm_tup = (self._get_axis_idx_s5(tp.ub_axis_perm, 0x00f000, 12),
                       self._get_axis_idx_s5(tp.ub_axis_perm, 0x000f00, 8),
                       self._get_axis_idx_s5(tp.ub_axis_perm, 0x0000f0, 4),
                       self._get_axis_idx_s5(tp.ub_axis_perm, 0x00000f, 0))

        def _gen_in_ub_layout():
            major_src_axis_0 = tp.src_jump_major_step
            major_dst_axis_0 = tp.dst_jump_major_step
            with tik_inst.if_scope(tp.src_axis_perm_len == 1):
                tp.in_ub_layout[base_idx + 3].set_as(major_src_axis_0)
            with tik_inst.if_scope(tp.src_axis_perm_len == 2):
                major_src_axis_1 = tp.second_to_last_axis_len
                src_axis_tup = (major_src_axis_0, major_src_axis_1)
                for index, axis in enumerate(src_axis_tup):
                    tp.in_ub_layout[base_idx + 2 + index].set_as(axis)
            with tik_inst.if_scope(tp.src_axis_perm_len == 3):
                major_src_axis_2 = tp.second_to_last_axis_len
                major_src_axis_1 = tp.major_in_ele // major_src_axis_0 // major_src_axis_2 // tp.last_axis_len
                src_axis_tup = (major_src_axis_0, major_src_axis_1, major_src_axis_2)
                for index, axis in enumerate(src_axis_tup):
                    tp.in_ub_layout[base_idx + 1 + index].set_as(axis)

            with tik_inst.if_scope(tp.dst_axis_perm_len == 1):
                tp.in_ub_layout[base_idx + ub_perm_tup[3]].set_as(major_dst_axis_0)
            with tik_inst.if_scope(tp.dst_axis_perm_len == 2):
                major_dst_axis_1 = tp.second_to_out_last_axis_len
                dst_axis_tup = (major_dst_axis_0, major_dst_axis_1)
                for index, axis in enumerate(dst_axis_tup):
                    tp.in_ub_layout[base_idx + ub_perm_tup[2 + index]].set_as(axis)
            with tik_inst.if_scope(tp.dst_axis_perm_len == 3):
                major_dst_axis_2 = tp.second_to_out_last_axis_len
                major_dst_axis_1 = tp.major_out_ele // major_dst_axis_0 // major_dst_axis_2 // tp.last_axis_len
                dst_axis_tup = (major_dst_axis_0, major_dst_axis_1, major_dst_axis_2)
                for index, axis in enumerate(dst_axis_tup):
                    tp.in_ub_layout[base_idx + ub_perm_tup[1 + index]].set_as(axis)

        def _gen_dst_offset():
            with tik_inst.if_scope(tp.ub_axis_perm_len - tp.dst_axis_perm_len == 1):
                for index, perm in enumerate(ub_perm_tup):
                    self._get_input_axis_offset_s5(tp, tp.in_axis_off[base_idx + index],
                                                   base_idx + perm, src_jump_idx)
                    tp.out_ub_layout[base_idx + index].set_as(tp.in_ub_layout[base_idx + perm])
            with tik_inst.elif_scope(tp.ub_axis_perm_len - tp.dst_axis_perm_len == 2):
                with tik_inst.if_scope(ub_perm_tup[0] > ub_perm_tup[1]):
                    self._get_input_axis_offset_s5(tp, tp.in_axis_off[base_idx],
                                                   base_idx + ub_perm_tup[1], src_jump_idx)
                    self._get_input_axis_offset_s5(tp, tp.in_axis_off[base_idx + 1],
                                                   base_idx + ub_perm_tup[0], src_jump_idx)
                    tp.out_ub_layout[base_idx].set_as(tp.in_ub_layout[base_idx + ub_perm_tup[1]])
                    tp.out_ub_layout[base_idx + 1].set_as(tp.in_ub_layout[base_idx + ub_perm_tup[0]])
                with tik_inst.else_scope():
                    self._get_input_axis_offset_s5(tp, tp.in_axis_off[base_idx],
                                                   base_idx + ub_perm_tup[0], src_jump_idx)
                    self._get_input_axis_offset_s5(tp, tp.in_axis_off[base_idx + 1],
                                                   base_idx + ub_perm_tup[1], src_jump_idx)
                    tp.out_ub_layout[base_idx].set_as(tp.in_ub_layout[base_idx + ub_perm_tup[0]])
                    tp.out_ub_layout[base_idx + 1].set_as(tp.in_ub_layout[base_idx + ub_perm_tup[1]])
                for index, perm in enumerate(ub_perm_tup[2:]):
                    self._get_input_axis_offset_s5(tp, tp.in_axis_off[base_idx + 2 + index],
                                                   base_idx + perm, src_jump_idx)
                    tp.out_ub_layout[base_idx + 2 + index].set_as(tp.in_ub_layout[base_idx + perm])
            # dup axis is 0
            with tik_inst.elif_scope(tp.ub_axis_perm_len - tp.dst_axis_perm_len == 3):
                for idx in range(3):
                    self._get_input_axis_offset_s5(tp, tp.in_axis_off[base_idx + idx],
                                                   src_jump_idx + idx, src_jump_idx)
                    tp.out_ub_layout[base_idx + idx].set_as(tp.in_ub_layout[src_jump_idx + idx])
                for index, perm in enumerate(ub_perm_tup[3:]):
                    self._get_input_axis_offset_s5(tp, tp.in_axis_off[base_idx + 3 + index],
                                                   base_idx + perm, src_jump_idx)
                    tp.out_ub_layout[base_idx + 3 + index].set_as(tp.in_ub_layout[base_idx + perm])

        def _set_tail_in_ele():
            with tik_inst.if_scope(tp.src_axis_perm_len == 1):
                tp.in_ub_layout[base_idx + 3].set_as(tp.tail_in_ele // tp.last_axis_len)
            with tik_inst.elif_scope(tp.src_axis_perm_len == 2):
                major_src_axis_1 = tp.second_to_last_axis_len
                tp.in_ub_layout[base_idx + 2].set_as(tp.tail_in_ele // major_src_axis_1 // tp.last_axis_len)
            with tik_inst.else_scope():
                major_src_axis_2 = tp.second_to_last_axis_len
                major_src_axis_1 = tp.major_in_ele // major_src_axis_2 // tp.last_axis_len // tp.src_jump_major_step
                tp.in_ub_layout[base_idx + 1].set_as(
                    tp.tail_in_ele // major_src_axis_1 // major_src_axis_2 // tp.last_axis_len)

        # for case [is_src_tail == 0, is_dst_tail == 0]
        _gen_in_ub_layout()
        _gen_dst_offset()
        conv_params = [[1, tp.out_ub_layout[1], tp.out_ub_layout[2], tp.out_ub_layout[3], tp.out_ub_layout[4]],
                       [0, tp.in_axis_off[1], tp.in_axis_off[2], tp.in_axis_off[3], tp.in_axis_off[4]]]
        self._gen_src_offset_process_s5(tp, ub_input, 0, conv_params)

        # for case [is_src_tail == 1, is_dst_tail == 0]
        with tik_inst.if_scope(tik.all(tp.tail_in_ele > 0, tp.pivot_src_axis_dup == 0)):
            _set_tail_in_ele()
            _gen_dst_offset()
            self._gen_src_offset_process_s5(tp, ub_input, 1, conv_params)

        # for case [is_src_tail == 0, is_dst_tail == 1]
        with tik_inst.if_scope(tik.all(tp.tail_out_ele > 0, tp.pivot_dst_axis_dup == 0)):
            with tik_inst.if_scope(tik.all(tp.tail_in_ele > 0, tp.pivot_src_axis_dup == 0)):
                with tik_inst.if_scope(tp.src_axis_perm_len == 1):
                    tp.in_ub_layout[base_idx + 3].set_as(tp.src_jump_major_step)
                with tik_inst.elif_scope(tp.src_axis_perm_len == 2):
                    tp.in_ub_layout[base_idx + 2].set_as(tp.src_jump_major_step)
                with tik_inst.else_scope():
                    tp.in_ub_layout[base_idx + 1].set_as(tp.src_jump_major_step)
            with tik_inst.if_scope(tp.dst_axis_perm_len == 1):
                tp.in_ub_layout[base_idx + ub_perm_tup[3]].set_as(tp.tail_out_ele // tp.last_axis_len)
            with tik_inst.elif_scope(tp.dst_axis_perm_len == 2):
                major_dst_axis_1 = tp.second_to_out_last_axis_len
                tp.in_ub_layout[base_idx + ub_perm_tup[2]].set_as(
                    tp.tail_out_ele // major_dst_axis_1 // tp.last_axis_len)
            with tik_inst.elif_scope(tp.dst_axis_perm_len == 3):
                major_dst_axis_2 = tp.second_to_out_last_axis_len
                major_dst_axis_1 = tp.major_out_ele // major_dst_axis_2 // tp.last_axis_len // tp.dst_jump_major_step
                tp.in_ub_layout[base_idx + ub_perm_tup[1]].set_as(
                    tp.tail_out_ele // major_dst_axis_1 // major_dst_axis_2 // tp.last_axis_len)
            _gen_dst_offset()
            self._gen_src_offset_process_s5(tp, ub_input, 2, conv_params)

        # for case [is_src_tail == 1, is_dst_tail == 1]
        with tik_inst.if_scope(tik.all(tp.tail_in_ele > 0, tp.tail_out_ele > 0)):
            _set_tail_in_ele()
            _gen_dst_offset()
            self._gen_src_offset_process_s5(tp, ub_input, 3, conv_params)

    def _gen_vgather_src_offset_4_ub_axis_perm_len_5_s5(self, tp, ub_input):
        tik_inst = self.tik_inst

        base_idx = 0
        self._get_src_dst_axis_perm_len_s5(tp)
        src_jump_idx = base_idx + tp.ub_axis_perm_len - tp.src_axis_perm_len
        ub_perm_tup = (self._get_axis_idx_s5(tp.ub_axis_perm, 0x0f0000, 16),
                       self._get_axis_idx_s5(tp.ub_axis_perm, 0x00f000, 12),
                       self._get_axis_idx_s5(tp.ub_axis_perm, 0x000f00, 8),
                       self._get_axis_idx_s5(tp.ub_axis_perm, 0x0000f0, 4),
                       self._get_axis_idx_s5(tp.ub_axis_perm, 0x00000f, 0))

        def _gen_in_ub_layout():
            major_src_axis_0 = tp.src_jump_major_step
            major_dst_axis_0 = tp.dst_jump_major_step
            with tik_inst.if_scope(tp.src_axis_perm_len == 2):
                major_src_axis_1 = tp.second_to_last_axis_len
                src_axis_tup = (major_src_axis_0, major_src_axis_1)
                for index, axis in enumerate(src_axis_tup):
                    tp.in_ub_layout[base_idx + 3 + index].set_as(axis)
            with tik_inst.if_scope(tp.src_axis_perm_len == 3):
                major_src_axis_2 = tp.second_to_last_axis_len
                major_src_axis_1 = tp.major_in_ele // major_src_axis_0 // major_src_axis_2 // tp.last_axis_len
                src_axis_tup = (major_src_axis_0, major_src_axis_1, major_src_axis_2)
                for index, axis in enumerate(src_axis_tup):
                    tp.in_ub_layout[base_idx + 2 + index].set_as(axis)

            with tik_inst.if_scope(tp.dst_axis_perm_len == 2):
                major_dst_axis_1 = tp.second_to_out_last_axis_len
                dst_axis_tup = (major_dst_axis_0, major_dst_axis_1)
                for index, axis in enumerate(dst_axis_tup):
                    tp.in_ub_layout[base_idx + ub_perm_tup[3 + index]].set_as(axis)
            with tik_inst.if_scope(tp.dst_axis_perm_len == 3):
                major_dst_axis_2 = tp.second_to_out_last_axis_len
                major_dst_axis_1 = tp.major_out_ele // major_dst_axis_0 // major_dst_axis_2 // tp.last_axis_len
                dst_axis_tup = (major_dst_axis_0, major_dst_axis_1, major_dst_axis_2)
                for index, axis in enumerate(dst_axis_tup):
                    tp.in_ub_layout[base_idx + ub_perm_tup[2 + index]].set_as(axis)

        def _gen_dst_offset():
            with tik_inst.if_scope(tp.ub_axis_perm_len - tp.dst_axis_perm_len == 2):
                with tik_inst.if_scope(ub_perm_tup[0] > ub_perm_tup[1]):
                    self._get_input_axis_offset_s5(tp, tp.in_axis_off[base_idx],
                                                   base_idx + ub_perm_tup[1], src_jump_idx)
                    self._get_input_axis_offset_s5(tp, tp.in_axis_off[base_idx + 1],
                                                   base_idx + ub_perm_tup[0], src_jump_idx)
                    tp.out_ub_layout[base_idx].set_as(tp.in_ub_layout[base_idx + ub_perm_tup[1]])
                    tp.out_ub_layout[base_idx + 1].set_as(tp.in_ub_layout[base_idx + ub_perm_tup[0]])
                with tik_inst.else_scope():
                    self._get_input_axis_offset_s5(tp, tp.in_axis_off[base_idx],
                                                   base_idx + ub_perm_tup[0], src_jump_idx)
                    self._get_input_axis_offset_s5(tp, tp.in_axis_off[base_idx + 1],
                                                   base_idx + ub_perm_tup[1], src_jump_idx)
                    tp.out_ub_layout[base_idx].set_as(tp.in_ub_layout[base_idx + ub_perm_tup[0]])
                    tp.out_ub_layout[base_idx + 1].set_as(tp.in_ub_layout[base_idx + ub_perm_tup[1]])
                for index, perm in enumerate(ub_perm_tup[2:]):
                    self._get_input_axis_offset_s5(tp, tp.in_axis_off[base_idx + 2 + index],
                                                   base_idx + perm, src_jump_idx)
                    tp.out_ub_layout[base_idx + 2 + index].set_as(tp.in_ub_layout[base_idx + perm])
            # dup axis is 0
            with tik_inst.elif_scope(tp.ub_axis_perm_len - tp.dst_axis_perm_len == 3):
                for idx in range(3):
                    self._get_input_axis_offset_s5(tp, tp.in_axis_off[base_idx + idx],
                                                   src_jump_idx + idx, src_jump_idx)
                    tp.out_ub_layout[base_idx + idx].set_as(tp.in_ub_layout[src_jump_idx + idx])
                for index, perm in enumerate(ub_perm_tup[3:]):
                    self._get_input_axis_offset_s5(tp, tp.in_axis_off[base_idx + 3 + index],
                                                   base_idx + perm, src_jump_idx)
                    tp.out_ub_layout[base_idx + 3 + index].set_as(tp.in_ub_layout[base_idx + perm])

        def _set_tail_in_ele():
            with tik_inst.if_scope(tp.src_axis_perm_len == 2):
                tp.in_ub_layout[base_idx + 3].set_as(tp.tail_in_ele // tp.last_axis_len // tp.second_to_last_axis_len)
            with tik_inst.elif_scope(tp.src_axis_perm_len == 3):
                major_src_axis_2 = tp.second_to_last_axis_len
                major_src_axis_1 = tp.major_in_ele // major_src_axis_2 // tp.last_axis_len // tp.src_jump_major_step
                tp.in_ub_layout[base_idx + 2].set_as(
                    tp.tail_in_ele // major_src_axis_1 // major_src_axis_2 // tp.last_axis_len)

        # for case [is_src_tail == 0, is_dst_tail == 0]
        _gen_in_ub_layout()
        _gen_dst_offset()
        conv_params = [[tp.out_ub_layout[0], tp.out_ub_layout[1], tp.out_ub_layout[2],
                        tp.out_ub_layout[3], tp.out_ub_layout[4]],
                       [tp.in_axis_off[0], tp.in_axis_off[1], tp.in_axis_off[2],
                        tp.in_axis_off[3], tp.in_axis_off[4]]]
        self._gen_src_offset_process_s5(tp, ub_input, 0, conv_params)

        # for case [is_src_tail == 1, is_dst_tail == 0]
        with tik_inst.if_scope(tik.all(tp.tail_in_ele > 0, tp.pivot_src_axis_dup == 0)):
            _set_tail_in_ele()
            _gen_dst_offset()
            self._gen_src_offset_process_s5(tp, ub_input, 1, conv_params)

        # for case [is_src_tail == 0, is_dst_tail == 1]
        with tik_inst.if_scope(tik.all(tp.tail_out_ele > 0, tp.pivot_dst_axis_dup == 0)):
            with tik_inst.if_scope(tik.all(tp.tail_in_ele > 0, tp.pivot_src_axis_dup == 0)):
                with tik_inst.if_scope(tp.src_axis_perm_len == 2):
                    tp.in_ub_layout[base_idx + 3].set_as(tp.src_jump_major_step)
                with tik_inst.elif_scope(tp.src_axis_perm_len == 3):
                    tp.in_ub_layout[base_idx + 2].set_as(tp.src_jump_major_step)
            with tik_inst.if_scope(tp.dst_axis_perm_len == 2):
                tp.in_ub_layout[base_idx + ub_perm_tup[3]].set_as(
                    tp.tail_out_ele // tp.last_axis_len // tp.second_to_out_last_axis_len)
            with tik_inst.elif_scope(tp.dst_axis_perm_len == 3):
                major_dst_axis_2 = tp.second_to_out_last_axis_len
                major_dst_axis_1 = tp.major_out_ele // major_dst_axis_2 // tp.last_axis_len // tp.dst_jump_major_step
                tp.in_ub_layout[base_idx + ub_perm_tup[2]].set_as(
                    tp.tail_out_ele // major_dst_axis_1 // major_dst_axis_2 // tp.last_axis_len)
            _gen_dst_offset()
            self._gen_src_offset_process_s5(tp, ub_input, 2, conv_params)

        # for case [is_src_tail == 1, is_dst_tail == 1]
        with tik_inst.if_scope(tik.all(tp.tail_in_ele > 0, tp.tail_out_ele > 0)):
            _set_tail_in_ele()
            _gen_dst_offset()
            self._gen_src_offset_process_s5(tp, ub_input, 3, conv_params)

    def _gen_vgather_src_offset_4_ub_axis_perm_len_6_s5(self, tp, ub_input):
        """
        src and dst axis perm size are all 3 and dup axis is 0
        """
        tik_inst = self.tik_inst

        base_idx = 1
        self._get_src_dst_axis_perm_len_s5(tp)
        src_jump_idx = base_idx + tp.ub_axis_perm_len - tp.src_axis_perm_len
        ub_perm_tup = (3,
                       self._get_axis_idx_s5(tp.ub_axis_perm, 0x000f00, 8),
                       self._get_axis_idx_s5(tp.ub_axis_perm, 0x0000f0, 4),
                       self._get_axis_idx_s5(tp.ub_axis_perm, 0x00000f, 0))

        def _gen_in_ub_layout():
            tp.in_ub_layout[base_idx + 3].set_as(tp.major_in_ele // tp.last_axis_len)
            major_dst_axis_0 = tp.dst_jump_major_step
            major_dst_axis_2 = tp.second_to_out_last_axis_len
            major_dst_axis_1 = tp.major_out_ele // major_dst_axis_0 // major_dst_axis_2 // tp.last_axis_len
            dst_axis_tup = (major_dst_axis_0, major_dst_axis_1, major_dst_axis_2)
            for index, axis in enumerate(dst_axis_tup):
                tp.in_ub_layout[base_idx + ub_perm_tup[1 + index]].set_as(axis)

        def _gen_dst_offset():
            for index, perm in enumerate(ub_perm_tup):
                self._get_input_axis_offset_s5(tp, tp.in_axis_off[base_idx + index], base_idx + perm, src_jump_idx)

        def _set_tail_in_ele():
            tp.in_ub_layout[base_idx + 3].set_as(tp.tail_in_ele // tp.last_axis_len)

        # for case [is_src_tail == 0, is_dst_tail == 0]
        _gen_in_ub_layout()
        _gen_dst_offset()
        conv_params = [[1, tp.in_ub_layout[base_idx + ub_perm_tup[0]], tp.in_ub_layout[base_idx + ub_perm_tup[1]],
                        tp.in_ub_layout[base_idx + ub_perm_tup[2]], tp.in_ub_layout[base_idx + ub_perm_tup[3]]],
                       [0, tp.in_axis_off[1], tp.in_axis_off[2], tp.in_axis_off[3], tp.in_axis_off[4]]]
        self._gen_src_offset_process_s5(tp, ub_input, 0, conv_params)

        # for case [is_src_tail == 1, is_dst_tail == 0]
        with tik_inst.if_scope(tik.all(tp.tail_in_ele > 0, tp.pivot_src_axis_dup == 0)):
            _set_tail_in_ele()
            _gen_dst_offset()
            self._gen_src_offset_process_s5(tp, ub_input, 1, conv_params)

        # for case [is_src_tail == 0, is_dst_tail == 1]
        with tik_inst.if_scope(tik.all(tp.tail_out_ele > 0, tp.pivot_dst_axis_dup == 0)):
            tp.in_ub_layout[base_idx + 3].set_as(tp.major_in_ele // tp.last_axis_len)
            major_dst_axis_2 = tp.second_to_out_last_axis_len
            major_dst_axis_1 = tp.major_out_ele // major_dst_axis_2 // tp.last_axis_len // tp.dst_jump_major_step
            tp.in_ub_layout[base_idx + ub_perm_tup[1]].set_as(
                tp.tail_out_ele // major_dst_axis_1 // major_dst_axis_2 // tp.last_axis_len)
            _gen_dst_offset()
            self._gen_src_offset_process_s5(tp, ub_input, 2, conv_params)

        # for case [is_src_tail == 1, is_dst_tail == 1]
        with tik_inst.if_scope(tik.all(tp.tail_in_ele > 0, tp.tail_out_ele > 0)):
            _set_tail_in_ele()
            _gen_dst_offset()
            self._gen_src_offset_process_s5(tp, ub_input, 3, conv_params)

    def _gen_vgather_src_offset_with_dup_axis_3_s5(self, tp, ub_input):
        tik_inst = self.tik_inst
        major_dst_axis_0 = tp.dst_jump_major_step
        major_dst_axis_2 = tp.second_to_out_last_axis_len
        major_dst_axis_1 = tp.major_out_ele // tp.last_axis_len // major_dst_axis_0 // major_dst_axis_2

        with tik_inst.if_scope(tp.dst_axis_perm == 0x021):
            # the order in input ub is: dst_axis_0 dst_axis_2 dst_axis_1 and the reorder index 1, 2 are empty
            input_axis_1_off = tp.last_axis_len
            input_axis_2_off = major_dst_axis_1 * tp.last_axis_len
            input_axis_0_off = major_dst_axis_2 * input_axis_2_off
            conv_params = [[1, 1, major_dst_axis_0, major_dst_axis_1, major_dst_axis_2],
                           [0, 0, input_axis_0_off, input_axis_1_off, input_axis_2_off]]
            self._gen_src_offset_process_s5(tp, ub_input, 0, conv_params)
            with tik_inst.if_scope(tp.tail_out_ele > 0):
                tail_dst_axis_0 = tp.tail_out_ele // tp.last_axis_len // major_dst_axis_1 // major_dst_axis_2
                conv_params = [[1, 1, tail_dst_axis_0, major_dst_axis_1, major_dst_axis_2],
                            [0, 0, input_axis_0_off, input_axis_1_off, input_axis_2_off]]
                self._gen_src_offset_process_s5(tp, ub_input, 3, conv_params)
        with tik_inst.if_scope(tp.dst_axis_perm == 0x210):
            # the order in input ub is: dst_axis_2 dst_axis_1 dst_axis_0 and the reorder index 1, 2 and 3 are empty
            input_axis_0_off = tp.last_axis_len
            input_axis_1_off = major_dst_axis_0 * tp.last_axis_len
            input_axis_2_off = major_dst_axis_1 * input_axis_1_off
            conv_params = [[1, 1, major_dst_axis_0, major_dst_axis_1, major_dst_axis_2],
                           [0, 0, input_axis_0_off, input_axis_1_off, input_axis_2_off]]
            self._gen_src_offset_process_s5(tp, ub_input, 0, conv_params)

    def _gen_vgather_src_offset_s5(self, tp, ub_input):
        """
        to generate src_offset in vgather for scenario 5
        """
        tik_inst = self.tik_inst

        self._get_ub_axis_perm_len_s5(tp)
        with tik_inst.if_scope(tp.dup_axis != 3):
            with tik_inst.if_scope(tp.ub_axis_perm_len == 2):
                self._gen_vgather_src_offset_4_ub_axis_perm_len_2_s5(tp, ub_input)
            with tik_inst.elif_scope(tp.ub_axis_perm_len == 3):
                self._gen_vgather_src_offset_4_ub_axis_perm_len_3_s5(tp, ub_input)
            with tik_inst.elif_scope(tp.ub_axis_perm_len == 4):
                self._gen_vgather_src_offset_4_ub_axis_perm_len_4_s5(tp, ub_input)
            with tik_inst.elif_scope(tp.ub_axis_perm_len == 5):
                self._gen_vgather_src_offset_4_ub_axis_perm_len_5_s5(tp, ub_input)
            with tik_inst.elif_scope(tp.ub_axis_perm_len == 6):
                self._gen_vgather_src_offset_4_ub_axis_perm_len_6_s5(tp, ub_input)
        with tik_inst.elif_scope(tp.dup_axis == 3):
            self._gen_vgather_src_offset_with_dup_axis_3_s5(tp, ub_input)

    def _major_process_s5(self, tp, ub_input):
        self._copy_in_major_src_major_dst_s5(tp, ub_input)
        self._reorder_s5(tp, ub_input, 0, 0,
                         tp.major_in_ele,
                         tp.major_in_tail_ele,
                         tp.major_burst_len_in,
                         tp.major_dst_loop_in,
                         tp.major_out_ele,
                         tp.major_out_tail_ele,
                         tp.major_burst_len_out,
                         tp.major_src_loop_out)
        self._copy_out_major_src_major_dst_s5(tp, ub_input)

    def _src_tail_in_process_s5(self, tp, ub_input):
        tik_inst = self.tik_inst
        with tik_inst.if_scope(tp.tail_burst_len_in != 0):
            self._init_tail_src_tuple_copy_out_s5(tp)
            self._init_major_dst_tuple_copy_in_s5(tp)
            self._copy_in_tail_src_major_dst_s5(tp, ub_input)
            self._reorder_s5(tp, ub_input, 1, 0,
                             tp.tail_in_ele,
                             tp.tail_in_tail_ele,
                             tp.tail_burst_len_in,
                             tp.major_dst_loop_in,
                             tp.major_out_ele,
                             tp.major_out_tail_ele,
                             tp.major_burst_len_out,
                             tp.tail_src_loop_out)
            self._copy_out_tail_src_major_dst_s5(tp, ub_input)

    def _dst_tail_in_process_s5(self, tp, ub_input):
        tik_inst = self.tik_inst
        with tik_inst.if_scope(tp.tail_burst_len_out != 0):
            self._init_major_src_tuple_copy_out_s5(tp)
            self._init_tail_dst_tuple_copy_in_s5(tp)
            self._copy_in_major_src_tail_dst_s5(tp, ub_input)
            self._reorder_s5(tp, ub_input, 0, 1,
                             tp.major_in_ele,
                             tp.major_in_tail_ele,
                             tp.major_burst_len_in,
                             tp.tail_dst_loop_in,
                             tp.tail_out_ele,
                             tp.tail_out_tail_ele,
                             tp.tail_burst_len_out,
                             tp.major_src_loop_out)
            self._copy_out_major_src_tail_dst_s5(tp, ub_input)

    def _both_tail_in_process_s5(self, tp, ub_input):
        tik_inst = self.tik_inst
        with tik_inst.if_scope(tik.all(tp.tail_burst_len_in != 0, tp.tail_burst_len_out != 0)):
            self._init_tail_src_tuple_copy_out_s5(tp)
            self._init_tail_dst_tuple_copy_in_s5(tp)
            self._copy_in_tail_src_tail_dst_s5(tp, ub_input)
            self._reorder_s5(tp, ub_input, 1, 1,
                             tp.tail_in_ele,
                             tp.tail_in_tail_ele,
                             tp.tail_burst_len_in,
                             tp.tail_dst_loop_in,
                             tp.tail_out_ele,
                             tp.tail_out_tail_ele,
                             tp.tail_burst_len_out,
                             tp.tail_src_loop_out)
            self._copy_out_tail_src_tail_dst_s5(tp, ub_input)

    def _move_data_s5(self, tp, ub_input_64):
        tik_inst = self.tik_inst
        is_src_tail_in = self.tik_inst.Scalar("int32")
        is_dst_tail_in = self.tik_inst.Scalar("int32")
        ub_input = ub_input_64.reinterpret_cast_to(self.x_dtype)
        self._init_all_tuple_s5(tp)
        if self.vg_cond is True:
            with tik_inst.if_scope(tp.scenario == 14):
                tp.src_offset_base_addr = tp.major_dst_loop_in * tp.major_burst_len_in * self.ele_per_block
                tp.src_offset_addr_gap = tp.major_src_loop_out * tp.major_burst_len_out * self.ele_per_block
                self._gen_vgather_src_offset_s5(tp, ub_input)

        with tik_inst.for_range(0, tp.loop_per_core):
            is_src_tail_in.set_as(0)
            is_dst_tail_in.set_as(0)

            self._init_major_src_tuple_copy_out_s5(tp)
            self._init_major_dst_tuple_copy_in_s5(tp)
            self._major_process_s5(tp, ub_input)
            self._detect_tail_flag(tp, is_src_tail_in, is_dst_tail_in)

            with tik_inst.if_scope(tik.all(is_src_tail_in == 1, tp.pivot_src_axis_dup == 0)):
                self._src_tail_in_process_s5(tp, ub_input)

            with tik_inst.if_scope(tik.all(is_dst_tail_in == 1, tp.pivot_dst_axis_dup == 0)):
                self._dst_tail_in_process_s5(tp, ub_input)

            with tik_inst.if_scope(tik.all(is_dst_tail_in == 1, is_src_tail_in == 1)):
                self._both_tail_in_process_s5(tp, ub_input)

            self._update_logic_tuple_s5(tp)

    # -------------------------------------------------------------------------------------------------
    #                                    scenario_7
    # -------------------------------------------------------------------------------------------------
    @staticmethod
    def _init_n_tuple(tp):
        for i in range(TRANSPOSE_MAX_AXIS_NUM - 1):
            tp.rt_n_tuple[i].set_as(tp.init_n_tuple[i])

    @staticmethod
    def _init_dst_tuple(tp):
        for i in range(TRANSPOSE_MAX_AXIS_NUM - 1):
            tp.rt_dst_tuple[i].set_as(tp.init_dst_tuple[i])
        for i in range(TRANSPOSE_MAX_AXIS_NUM - 1):
            tp.rt_dst_tuple_backup[i].set_as(tp.init_dst_tuple[i])

    @staticmethod
    def _restore_dst_tuple(tp):
        for i in range(TRANSPOSE_MAX_AXIS_NUM - 1):
            tp.rt_dst_tuple[i].set_as(tp.rt_dst_tuple_backup[i])

    @staticmethod
    def _backup_dst_tuple(tp):
        for i in range(TRANSPOSE_MAX_AXIS_NUM - 1):
            tp.rt_dst_tuple_backup[i].set_as(tp.rt_dst_tuple[i])

    @staticmethod
    def _tail_dst_tuple(tp):
        for i in range(TRANSPOSE_MAX_AXIS_NUM - 1):
            tp.rt_dst_tuple[i].set_as(tp.tail_dst_tuple[i])

    @staticmethod
    def _init_src_tuple(tp):
        for i in range(TRANSPOSE_MAX_AXIS_NUM - 1):
            tp.rt_src_tuple[i].set_as(tp.init_src_tuple[i])

    @staticmethod
    def _tail_src_tuple(tp):
        for i in range(TRANSPOSE_MAX_AXIS_NUM - 1):
            tp.rt_src_tuple[i].set_as(tp.tail_src_tuple[i])

    def _update_tuple(self, axis_num, rt_tuple, jump_factor, repeat=1):
        with self.tik_inst.if_scope(axis_num == 1):
            rt_tuple[0].set_as(rt_tuple[0] + repeat)
        with self.tik_inst.else_scope():
            with self.tik_inst.if_scope(axis_num == 2):
                with self.tik_inst.if_scope(rt_tuple[0] == jump_factor[0] - repeat):
                    rt_tuple[0].set_as(0)
                    rt_tuple[1].set_as(rt_tuple[1] + 1)
                with self.tik_inst.else_scope():
                    rt_tuple[0].set_as(rt_tuple[0] + repeat)
            with self.tik_inst.else_scope():
                with self.tik_inst.if_scope(axis_num == 3):
                    with self.tik_inst.if_scope(rt_tuple[0] == jump_factor[0] - repeat):
                        rt_tuple[0].set_as(0)
                        with self.tik_inst.if_scope(rt_tuple[1] == jump_factor[1] - 1):
                            rt_tuple[1].set_as(0)
                            rt_tuple[2].set_as(rt_tuple[2] + 1)
                        with self.tik_inst.else_scope():
                            rt_tuple[1].set_as(rt_tuple[1] + 1)
                    with self.tik_inst.else_scope():
                        rt_tuple[0].set_as(rt_tuple[0] + repeat)
                with self.tik_inst.else_scope():
                    with self.tik_inst.if_scope(axis_num == 4):
                        with self.tik_inst.if_scope(rt_tuple[0] == jump_factor[0] - repeat):
                            rt_tuple[0].set_as(0)
                            with self.tik_inst.if_scope(rt_tuple[1] == jump_factor[1] - 1):
                                rt_tuple[1].set_as(0)
                                with self.tik_inst.if_scope(rt_tuple[2] == jump_factor[2] - 1):
                                    rt_tuple[2].set_as(0)
                                    rt_tuple[3].set_as(rt_tuple[3] + 1)
                                with self.tik_inst.else_scope():
                                    rt_tuple[2].set_as(rt_tuple[2] + 1)
                            with self.tik_inst.else_scope():
                                rt_tuple[1].set_as(rt_tuple[1] + 1)
                        with self.tik_inst.else_scope():
                            rt_tuple[0].set_as(rt_tuple[0] + repeat)
                    with self.tik_inst.else_scope():
                        with self.tik_inst.if_scope(axis_num == 5):
                            with self.tik_inst.if_scope(rt_tuple[0] == jump_factor[0] - repeat):
                                rt_tuple[0].set_as(0)
                                with self.tik_inst.if_scope(rt_tuple[1] == jump_factor[1] - 1):
                                    rt_tuple[1].set_as(0)
                                    with self.tik_inst.if_scope(rt_tuple[2] == jump_factor[2] - 1):
                                        rt_tuple[2].set_as(0)
                                        with self.tik_inst.if_scope(rt_tuple[3] == jump_factor[3] - 1):
                                            rt_tuple[3].set_as(0)
                                            rt_tuple[4].set_as(rt_tuple[4] + 1)
                                        with self.tik_inst.else_scope():
                                            rt_tuple[3].set_as(rt_tuple[3] + 1)
                                    with self.tik_inst.else_scope():
                                        rt_tuple[2].set_as(rt_tuple[2] + 1)
                                with self.tik_inst.else_scope():
                                    rt_tuple[1].set_as(rt_tuple[1] + 1)
                            with self.tik_inst.else_scope():
                                rt_tuple[0].set_as(rt_tuple[0] + repeat)

                        with self.tik_inst.if_scope(axis_num == 6):
                            with self.tik_inst.if_scope(rt_tuple[0] == jump_factor[0] - repeat):
                                rt_tuple[0].set_as(0)
                                with self.tik_inst.if_scope(rt_tuple[1] == jump_factor[1] - 1):
                                    rt_tuple[1].set_as(0)
                                    with self.tik_inst.if_scope(rt_tuple[2] == jump_factor[2] - 1):
                                        rt_tuple[2].set_as(0)
                                        with self.tik_inst.if_scope(rt_tuple[3] == jump_factor[3] - 1):
                                            rt_tuple[3].set_as(0)
                                            with self.tik_inst.if_scope(rt_tuple[4] == jump_factor[4] - 1):
                                                rt_tuple[4].set_as(0)
                                                rt_tuple[5].set_as(rt_tuple[5] + 1)
                                            with self.tik_inst.else_scope():
                                                rt_tuple[4].set_as(rt_tuple[4] + 1)
                                        with self.tik_inst.else_scope():
                                            rt_tuple[3].set_as(rt_tuple[3] + 1)
                                    with self.tik_inst.else_scope():
                                        rt_tuple[2].set_as(rt_tuple[2] + 1)
                                with self.tik_inst.else_scope():
                                    rt_tuple[1].set_as(rt_tuple[1] + 1)
                            with self.tik_inst.else_scope():
                                rt_tuple[0].set_as(rt_tuple[0] + repeat)

                        with self.tik_inst.if_scope(axis_num == 7):
                            with self.tik_inst.if_scope(rt_tuple[0] == jump_factor[0] - repeat):
                                rt_tuple[0].set_as(0)
                                with self.tik_inst.if_scope(rt_tuple[1] == jump_factor[1] - 1):
                                    rt_tuple[1].set_as(0)
                                    with self.tik_inst.if_scope(rt_tuple[2] == jump_factor[2] - 1):
                                        rt_tuple[2].set_as(0)
                                        with self.tik_inst.if_scope(rt_tuple[3] == jump_factor[3] - 1):
                                            rt_tuple[3].set_as(0)
                                            with self.tik_inst.if_scope(rt_tuple[4] == jump_factor[4] - 1):
                                                rt_tuple[4].set_as(0)
                                                with self.tik_inst.if_scope(rt_tuple[5] == jump_factor[5] - 1):
                                                    rt_tuple[5].set_as(0)
                                                    rt_tuple[6].set_as(rt_tuple[6] + 1)
                                                with self.tik_inst.else_scope():
                                                    rt_tuple[5].set_as(rt_tuple[5] + 1)
                                            with self.tik_inst.else_scope():
                                                rt_tuple[4].set_as(rt_tuple[4] + 1)
                                        with self.tik_inst.else_scope():
                                            rt_tuple[3].set_as(rt_tuple[3] + 1)
                                    with self.tik_inst.else_scope():
                                        rt_tuple[2].set_as(rt_tuple[2] + 1)
                                with self.tik_inst.else_scope():
                                    rt_tuple[1].set_as(rt_tuple[1] + 1)
                            with self.tik_inst.else_scope():
                                rt_tuple[0].set_as(rt_tuple[0] + repeat)

                        with self.tik_inst.if_scope(axis_num == 8):
                            with self.tik_inst.if_scope(rt_tuple[0] == jump_factor[0] - repeat):
                                rt_tuple[0].set_as(0)
                                with self.tik_inst.if_scope(rt_tuple[1] == jump_factor[1] - 1):
                                    rt_tuple[1].set_as(0)
                                    with self.tik_inst.if_scope(rt_tuple[2] == jump_factor[2] - 1):
                                        rt_tuple[2].set_as(0)
                                        with self.tik_inst.if_scope(rt_tuple[3] == jump_factor[3] - 1):
                                            rt_tuple[3].set_as(0)
                                            with self.tik_inst.if_scope(rt_tuple[4] == jump_factor[4] - 1):
                                                rt_tuple[4].set_as(0)
                                                with self.tik_inst.if_scope(rt_tuple[5] == jump_factor[5] - 1):
                                                    rt_tuple[5].set_as(0)
                                                    with self.tik_inst.if_scope(rt_tuple[6] == jump_factor[6] - 1):
                                                        rt_tuple[6].set_as(0)
                                                        rt_tuple[7].set_as(rt_tuple[7] + 1)
                                                    with self.tik_inst.else_scope():
                                                        rt_tuple[6].set_as(rt_tuple[6] + 1)
                                                with self.tik_inst.else_scope():
                                                    rt_tuple[5].set_as(rt_tuple[5] + 1)
                                            with self.tik_inst.else_scope():
                                                rt_tuple[4].set_as(rt_tuple[4] + 1)
                                        with self.tik_inst.else_scope():
                                            rt_tuple[3].set_as(rt_tuple[3] + 1)
                                    with self.tik_inst.else_scope():
                                        rt_tuple[2].set_as(rt_tuple[2] + 1)
                                with self.tik_inst.else_scope():
                                    rt_tuple[1].set_as(rt_tuple[1] + 1)
                            with self.tik_inst.else_scope():
                                rt_tuple[0].set_as(rt_tuple[0] + repeat)

    def _update_tuple_with_steps(self, axis_num, rt_tuple, jump_factor, jump_factor_mod, base, steps):
        with self.tik_inst.if_scope(axis_num == 1):
            rt_tuple[0].set_as((base + steps) / jump_factor_mod[0] % jump_factor[0])
        with self.tik_inst.else_scope():
            with self.tik_inst.if_scope(axis_num == 2):
                rt_tuple[0].set_as((base + steps) / jump_factor_mod[0] % jump_factor[0])
                rt_tuple[1].set_as((base + steps) / jump_factor_mod[1] % jump_factor[1])
            with self.tik_inst.else_scope():
                with self.tik_inst.if_scope(axis_num == 3):
                    rt_tuple[0].set_as((base + steps) / jump_factor_mod[0] % jump_factor[0])
                    rt_tuple[1].set_as((base + steps) / jump_factor_mod[1] % jump_factor[1])
                    rt_tuple[2].set_as((base + steps) / jump_factor_mod[2] % jump_factor[2])
                with self.tik_inst.else_scope():
                    with self.tik_inst.if_scope(axis_num == 4):
                        rt_tuple[0].set_as((base + steps) / jump_factor_mod[0] % jump_factor[0])
                        rt_tuple[1].set_as((base + steps) / jump_factor_mod[1] % jump_factor[1])
                        rt_tuple[2].set_as((base + steps) / jump_factor_mod[2] % jump_factor[2])
                        rt_tuple[3].set_as((base + steps) / jump_factor_mod[3] % jump_factor[3])
                    with self.tik_inst.else_scope():
                        with self.tik_inst.if_scope(axis_num == 5):
                            rt_tuple[0].set_as((base + steps) / jump_factor_mod[0] % jump_factor[0])
                            rt_tuple[1].set_as((base + steps) / jump_factor_mod[1] % jump_factor[1])
                            rt_tuple[2].set_as((base + steps) / jump_factor_mod[2] % jump_factor[2])
                            rt_tuple[3].set_as((base + steps) / jump_factor_mod[3] % jump_factor[3])
                            rt_tuple[4].set_as((base + steps) / jump_factor_mod[4] % jump_factor[4])

                        with self.tik_inst.if_scope(axis_num == 6):
                            rt_tuple[0].set_as((base + steps) / jump_factor_mod[0] % jump_factor[0])
                            rt_tuple[1].set_as((base + steps) / jump_factor_mod[1] % jump_factor[1])
                            rt_tuple[2].set_as((base + steps) / jump_factor_mod[2] % jump_factor[2])
                            rt_tuple[3].set_as((base + steps) / jump_factor_mod[3] % jump_factor[3])
                            rt_tuple[4].set_as((base + steps) / jump_factor_mod[4] % jump_factor[4])
                            rt_tuple[5].set_as((base + steps) / jump_factor_mod[5] % jump_factor[5])

                        with self.tik_inst.if_scope(axis_num == 7):
                            rt_tuple[0].set_as((base + steps) / jump_factor_mod[0] % jump_factor[0])
                            rt_tuple[1].set_as((base + steps) / jump_factor_mod[1] % jump_factor[1])
                            rt_tuple[2].set_as((base + steps) / jump_factor_mod[2] % jump_factor[2])
                            rt_tuple[3].set_as((base + steps) / jump_factor_mod[3] % jump_factor[3])
                            rt_tuple[4].set_as((base + steps) / jump_factor_mod[4] % jump_factor[4])
                            rt_tuple[5].set_as((base + steps) / jump_factor_mod[5] % jump_factor[5])
                            rt_tuple[6].set_as((base + steps) / jump_factor_mod[6] % jump_factor[6])
                        
                        with self.tik_inst.if_scope(axis_num == 8):
                            rt_tuple[0].set_as((base + steps) / jump_factor_mod[0] % jump_factor[0])
                            rt_tuple[1].set_as((base + steps) / jump_factor_mod[1] % jump_factor[1])
                            rt_tuple[2].set_as((base + steps) / jump_factor_mod[2] % jump_factor[2])
                            rt_tuple[3].set_as((base + steps) / jump_factor_mod[3] % jump_factor[3])
                            rt_tuple[4].set_as((base + steps) / jump_factor_mod[4] % jump_factor[4])
                            rt_tuple[5].set_as((base + steps) / jump_factor_mod[5] % jump_factor[5])
                            rt_tuple[6].set_as((base + steps) / jump_factor_mod[6] % jump_factor[6])
                            rt_tuple[7].set_as((base + steps) / jump_factor_mod[7] % jump_factor[7])

    def _get_n_offset_s7(self, tp):

        with self.tik_inst.if_scope(tp.n_axis_num == 0):
            tp.n_src_offset.set_as(0)
            tp.n_dst_offset.set_as(0)
        with self.tik_inst.else_scope():
            with self.tik_inst.if_scope(tp.n_axis_num == 1):
                tp.n_src_offset.set_as(tp.rt_n_tuple[0] * tp.n_jump_stride_in[0])
                tp.n_dst_offset.set_as(tp.rt_n_tuple[0] * tp.n_jump_stride_out[0])
            with self.tik_inst.else_scope():
                with self.tik_inst.if_scope(tp.n_axis_num == 2):
                    tp.n_src_offset.set_as(tp.rt_n_tuple[0] * tp.n_jump_stride_in[0] + \
                                           tp.rt_n_tuple[1] * tp.n_jump_stride_in[1])
                    tp.n_dst_offset.set_as(tp.rt_n_tuple[0] * tp.n_jump_stride_out[0] + \
                                           tp.rt_n_tuple[1] * tp.n_jump_stride_out[1])
                with self.tik_inst.else_scope():
                    with self.tik_inst.if_scope(tp.n_axis_num == 3):
                        tp.n_src_offset.set_as(tp.rt_n_tuple[0] * tp.n_jump_stride_in[0] + \
                                               tp.rt_n_tuple[1] * tp.n_jump_stride_in[1] + \
                                               tp.rt_n_tuple[2] * tp.n_jump_stride_in[2])
                        tp.n_dst_offset.set_as(tp.rt_n_tuple[0] * tp.n_jump_stride_out[0] + \
                                               tp.rt_n_tuple[1] * tp.n_jump_stride_out[1] + \
                                               tp.rt_n_tuple[2] * tp.n_jump_stride_out[2])
                    with self.tik_inst.else_scope():
                        with self.tik_inst.if_scope(tp.n_axis_num == 4):
                            tp.n_src_offset.set_as(tp.rt_n_tuple[0] * tp.n_jump_stride_in[0] + \
                                                   tp.rt_n_tuple[1] * tp.n_jump_stride_in[1] + \
                                                   tp.rt_n_tuple[2] * tp.n_jump_stride_in[2] + \
                                                   tp.rt_n_tuple[3] * tp.n_jump_stride_in[3])
                            tp.n_dst_offset.set_as(tp.rt_n_tuple[0] * tp.n_jump_stride_out[0] + \
                                                   tp.rt_n_tuple[1] * tp.n_jump_stride_out[1] + \
                                                   tp.rt_n_tuple[2] * tp.n_jump_stride_out[2] + \
                                                   tp.rt_n_tuple[3] * tp.n_jump_stride_out[3])
                        with self.tik_inst.if_scope(tp.n_axis_num == 5):
                            tp.n_src_offset.set_as(tp.rt_n_tuple[0] * tp.n_jump_stride_in[0] + \
                                                   tp.rt_n_tuple[1] * tp.n_jump_stride_in[1] + \
                                                   tp.rt_n_tuple[2] * tp.n_jump_stride_in[2] + \
                                                   tp.rt_n_tuple[3] * tp.n_jump_stride_in[3] + \
                                                   tp.rt_n_tuple[4] * tp.n_jump_stride_in[4])
                            tp.n_dst_offset.set_as(tp.rt_n_tuple[0] * tp.n_jump_stride_out[0] + \
                                                   tp.rt_n_tuple[1] * tp.n_jump_stride_out[1] + \
                                                   tp.rt_n_tuple[2] * tp.n_jump_stride_out[2] + \
                                                   tp.rt_n_tuple[3] * tp.n_jump_stride_out[3] + \
                                                   tp.rt_n_tuple[4] * tp.n_jump_stride_out[4])

    def _get_src_addr_s7(self, tp, ln, lc, lr, bsl):
        src_addr = self.tik_inst.Scalar("int64")

        with self.tik_inst.if_scope(tp.src_axis_num == 1):
            src_addr.set_as(tp.col_offset + lc * tp.col_per_mc + \
                            tp.rt_src_tuple[0] * tp.src_jump_stride[0] - \
                            bsl + tp.n_src_offset)
        with self.tik_inst.else_scope():
            with self.tik_inst.if_scope(tp.src_axis_num == 2):
                src_addr.set_as(tp.col_offset + lc * tp.col_per_mc + \
                                tp.rt_src_tuple[0] * tp.src_jump_stride[0] + \
                                tp.rt_src_tuple[1] * tp.src_jump_stride[1] - \
                                bsl + tp.n_src_offset)
            with self.tik_inst.else_scope():
                with self.tik_inst.if_scope(tp.src_axis_num == 3):
                    src_addr.set_as(tp.col_offset + lc * tp.col_per_mc + \
                                    tp.rt_src_tuple[0] * tp.src_jump_stride[0] + \
                                    tp.rt_src_tuple[1] * tp.src_jump_stride[1] + \
                                    tp.rt_src_tuple[2] * tp.src_jump_stride[2] - \
                                    bsl + tp.n_src_offset)
                with self.tik_inst.else_scope():
                    with self.tik_inst.if_scope(tp.src_axis_num == 4):
                        src_addr.set_as(tp.col_offset + lc * tp.col_per_mc + \
                                        tp.rt_src_tuple[0] * tp.src_jump_stride[0] + \
                                        tp.rt_src_tuple[1] * tp.src_jump_stride[1] + \
                                        tp.rt_src_tuple[2] * tp.src_jump_stride[2] + \
                                        tp.rt_src_tuple[3] * tp.src_jump_stride[3] - \
                                        bsl + tp.n_src_offset)
                    with self.tik_inst.else_scope():
                        with self.tik_inst.if_scope(tp.src_axis_num == 5):
                            src_addr.set_as(tp.col_offset + lc * tp.col_per_mc + \
                                            tp.rt_src_tuple[0] * tp.src_jump_stride[0] + \
                                            tp.rt_src_tuple[1] * tp.src_jump_stride[1] + \
                                            tp.rt_src_tuple[2] * tp.src_jump_stride[2] + \
                                            tp.rt_src_tuple[3] * tp.src_jump_stride[3] + \
                                            tp.rt_src_tuple[4] * tp.src_jump_stride[4] - \
                                            bsl + tp.n_src_offset)

                        with self.tik_inst.if_scope(tp.src_axis_num == 6):
                            src_addr.set_as(tp.col_offset + lc * tp.col_per_mc + \
                                            tp.rt_src_tuple[0] * tp.src_jump_stride[0] + \
                                            tp.rt_src_tuple[1] * tp.src_jump_stride[1] + \
                                            tp.rt_src_tuple[2] * tp.src_jump_stride[2] + \
                                            tp.rt_src_tuple[3] * tp.src_jump_stride[3] + \
                                            tp.rt_src_tuple[4] * tp.src_jump_stride[4] + \
                                            tp.rt_src_tuple[5] * tp.src_jump_stride[5] - \
                                            bsl + tp.n_src_offset)

                        with self.tik_inst.if_scope(tp.src_axis_num == 7):
                            src_addr.set_as(tp.col_offset + lc * tp.col_per_mc + \
                                            tp.rt_src_tuple[0] * tp.src_jump_stride[0] + \
                                            tp.rt_src_tuple[1] * tp.src_jump_stride[1] + \
                                            tp.rt_src_tuple[2] * tp.src_jump_stride[2] + \
                                            tp.rt_src_tuple[3] * tp.src_jump_stride[3] + \
                                            tp.rt_src_tuple[4] * tp.src_jump_stride[4] + \
                                            tp.rt_src_tuple[5] * tp.src_jump_stride[5] + \
                                            tp.rt_src_tuple[6] * tp.src_jump_stride[6] - \
                                            bsl + tp.n_src_offset)
                            
                        with self.tik_inst.if_scope(tp.src_axis_num == 8):
                            src_addr.set_as(tp.col_offset + lc * tp.col_per_mc + \
                                            tp.rt_src_tuple[0] * tp.src_jump_stride[0] + \
                                            tp.rt_src_tuple[1] * tp.src_jump_stride[1] + \
                                            tp.rt_src_tuple[2] * tp.src_jump_stride[2] + \
                                            tp.rt_src_tuple[3] * tp.src_jump_stride[3] + \
                                            tp.rt_src_tuple[4] * tp.src_jump_stride[4] + \
                                            tp.rt_src_tuple[5] * tp.src_jump_stride[5] + \
                                            tp.rt_src_tuple[6] * tp.src_jump_stride[6] + \
                                            tp.rt_src_tuple[7] * tp.src_jump_stride[7] - \
                                            bsl + tp.n_src_offset)
        return src_addr

    def _get_dst_addr_s7(self, tp, ln, lc, lr, col_id, bsl, bsu):
        dst_addr = self.tik_inst.Scalar("int64")

        with self.tik_inst.if_scope(tp.dst_axis_num == 1):
            dst_addr.set_as(tp.row_offset + lr * tp.row_per_mr + \
                            tp.rt_dst_tuple[0] * tp.dst_jump_stride[0] - \
                            bsu + tp.n_dst_offset)
        with self.tik_inst.else_scope():
            with self.tik_inst.if_scope(tp.dst_axis_num == 2):
                dst_addr.set_as(tp.row_offset + lr * tp.row_per_mr + \
                                tp.rt_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                tp.rt_dst_tuple[1] * tp.dst_jump_stride[1] - \
                                bsu + tp.n_dst_offset)
            with self.tik_inst.else_scope():
                with self.tik_inst.if_scope(tp.dst_axis_num == 3):
                    dst_addr.set_as(tp.row_offset + lr * tp.row_per_mr + \
                                    tp.rt_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                    tp.rt_dst_tuple[1] * tp.dst_jump_stride[1] + \
                                    tp.rt_dst_tuple[2] * tp.dst_jump_stride[2] - \
                                    bsu + tp.n_dst_offset)
                with self.tik_inst.else_scope():
                    with self.tik_inst.if_scope(tp.dst_axis_num == 4):
                        dst_addr.set_as(tp.row_offset + lr * tp.row_per_mr + \
                                        tp.rt_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                        tp.rt_dst_tuple[1] * tp.dst_jump_stride[1] + \
                                        tp.rt_dst_tuple[2] * tp.dst_jump_stride[2] + \
                                        tp.rt_dst_tuple[3] * tp.dst_jump_stride[3] - \
                                        bsu + tp.n_dst_offset)
                    with self.tik_inst.else_scope():
                        with self.tik_inst.if_scope(tp.dst_axis_num == 5):
                            dst_addr.set_as(tp.row_offset + lr * tp.row_per_mr + \
                                            tp.rt_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                            tp.rt_dst_tuple[1] * tp.dst_jump_stride[1] + \
                                            tp.rt_dst_tuple[2] * tp.dst_jump_stride[2] + \
                                            tp.rt_dst_tuple[3] * tp.dst_jump_stride[3] + \
                                            tp.rt_dst_tuple[4] * tp.dst_jump_stride[4] - \
                                            bsu + tp.n_dst_offset)

                        with self.tik_inst.if_scope(tp.dst_axis_num == 6):
                            dst_addr.set_as(tp.row_offset + lr * tp.row_per_mr + \
                                            tp.rt_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                            tp.rt_dst_tuple[1] * tp.dst_jump_stride[1] + \
                                            tp.rt_dst_tuple[2] * tp.dst_jump_stride[2] + \
                                            tp.rt_dst_tuple[3] * tp.dst_jump_stride[3] + \
                                            tp.rt_dst_tuple[4] * tp.dst_jump_stride[4] + \
                                            tp.rt_dst_tuple[5] * tp.dst_jump_stride[5] - \
                                            bsu + tp.n_dst_offset)

                        with self.tik_inst.if_scope(tp.dst_axis_num == 7):
                            dst_addr.set_as(tp.row_offset + lr * tp.row_per_mr + \
                                            tp.rt_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                            tp.rt_dst_tuple[1] * tp.dst_jump_stride[1] + \
                                            tp.rt_dst_tuple[2] * tp.dst_jump_stride[2] + \
                                            tp.rt_dst_tuple[3] * tp.dst_jump_stride[3] + \
                                            tp.rt_dst_tuple[4] * tp.dst_jump_stride[4] + \
                                            tp.rt_dst_tuple[5] * tp.dst_jump_stride[5] + \
                                            tp.rt_dst_tuple[6] * tp.dst_jump_stride[6] - \
                                            bsu + tp.n_dst_offset)
                        
                        with self.tik_inst.if_scope(tp.dst_axis_num == 8):
                            dst_addr.set_as(tp.row_offset + lr * tp.row_per_mr + \
                                            tp.rt_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                            tp.rt_dst_tuple[1] * tp.dst_jump_stride[1] + \
                                            tp.rt_dst_tuple[2] * tp.dst_jump_stride[2] + \
                                            tp.rt_dst_tuple[3] * tp.dst_jump_stride[3] + \
                                            tp.rt_dst_tuple[4] * tp.dst_jump_stride[4] + \
                                            tp.rt_dst_tuple[5] * tp.dst_jump_stride[5] + \
                                            tp.rt_dst_tuple[6] * tp.dst_jump_stride[6] + \
                                            tp.rt_dst_tuple[7] * tp.dst_jump_stride[7] - \
                                            bsu + tp.n_dst_offset)
        return dst_addr

    def _init_dst_addr_s7(self, tp, ln):
        with self.tik_inst.if_scope(tp.dst_axis_num == 8):
            tp.rt_dst_addr.set_as(tp.init_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                  tp.init_dst_tuple[1] * tp.dst_jump_stride[1] + \
                                  tp.init_dst_tuple[2] * tp.dst_jump_stride[2] + \
                                  tp.init_dst_tuple[3] * tp.dst_jump_stride[3] + \
                                  tp.init_dst_tuple[4] * tp.dst_jump_stride[4] + \
                                  tp.init_dst_tuple[5] * tp.dst_jump_stride[5] + \
                                  tp.init_dst_tuple[6] * tp.dst_jump_stride[6] + \
                                  tp.init_dst_tuple[7] * tp.dst_jump_stride[7] + \
                                  tp.row_offset + tp.n_dst_offset)

        with self.tik_inst.if_scope(tp.dst_axis_num == 7):
            tp.rt_dst_addr.set_as(tp.init_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                  tp.init_dst_tuple[1] * tp.dst_jump_stride[1] + \
                                  tp.init_dst_tuple[2] * tp.dst_jump_stride[2] + \
                                  tp.init_dst_tuple[3] * tp.dst_jump_stride[3] + \
                                  tp.init_dst_tuple[4] * tp.dst_jump_stride[4] + \
                                  tp.init_dst_tuple[5] * tp.dst_jump_stride[5] + \
                                  tp.init_dst_tuple[6] * tp.dst_jump_stride[6] + \
                                  tp.row_offset + tp.n_dst_offset)

        with self.tik_inst.if_scope(tp.dst_axis_num == 6):
            tp.rt_dst_addr.set_as(tp.init_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                  tp.init_dst_tuple[1] * tp.dst_jump_stride[1] + \
                                  tp.init_dst_tuple[2] * tp.dst_jump_stride[2] + \
                                  tp.init_dst_tuple[3] * tp.dst_jump_stride[3] + \
                                  tp.init_dst_tuple[4] * tp.dst_jump_stride[4] + \
                                  tp.init_dst_tuple[5] * tp.dst_jump_stride[5] + \
                                  tp.row_offset + tp.n_dst_offset)

        with self.tik_inst.if_scope(tp.dst_axis_num == 5):
            tp.rt_dst_addr.set_as(tp.init_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                  tp.init_dst_tuple[1] * tp.dst_jump_stride[1] + \
                                  tp.init_dst_tuple[2] * tp.dst_jump_stride[2] + \
                                  tp.init_dst_tuple[3] * tp.dst_jump_stride[3] + \
                                  tp.init_dst_tuple[4] * tp.dst_jump_stride[4] + \
                                  tp.row_offset + tp.n_dst_offset)

        with self.tik_inst.if_scope(tp.dst_axis_num == 4):
            tp.rt_dst_addr.set_as(tp.init_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                  tp.init_dst_tuple[1] * tp.dst_jump_stride[1] + \
                                  tp.init_dst_tuple[2] * tp.dst_jump_stride[2] + \
                                  tp.init_dst_tuple[3] * tp.dst_jump_stride[3] + \
                                  tp.row_offset + tp.n_dst_offset)

        with self.tik_inst.if_scope(tp.dst_axis_num == 3):
            tp.rt_dst_addr.set_as(tp.init_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                  tp.init_dst_tuple[1] * tp.dst_jump_stride[1] + \
                                  tp.init_dst_tuple[2] * tp.dst_jump_stride[2] + \
                                  tp.row_offset + tp.n_dst_offset)

        with self.tik_inst.if_scope(tp.dst_axis_num == 2):
            tp.rt_dst_addr.set_as(tp.init_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                  tp.init_dst_tuple[1] * tp.dst_jump_stride[1] + \
                                  tp.row_offset + tp.n_dst_offset)

        with self.tik_inst.if_scope(tp.dst_axis_num == 1):
            tp.rt_dst_addr.set_as(tp.init_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                  tp.row_offset + tp.n_dst_offset)

    def _tail_dst_addr_f2t(self, tp, ln):  # need merge
        with self.tik_inst.if_scope(tp.dst_axis_num == 8):
            tp.rt_dst_addr.set_as(tp.tail_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                  tp.tail_dst_tuple[1] * tp.dst_jump_stride[1] + \
                                  tp.tail_dst_tuple[2] * tp.dst_jump_stride[2] + \
                                  tp.tail_dst_tuple[3] * tp.dst_jump_stride[3] + \
                                  tp.tail_dst_tuple[4] * tp.dst_jump_stride[4] + \
                                  tp.tail_dst_tuple[5] * tp.dst_jump_stride[5] + \
                                  tp.tail_dst_tuple[6] * tp.dst_jump_stride[6] + \
                                  tp.tail_dst_tuple[7] * tp.dst_jump_stride[7] + tp.n_dst_offset)

        with self.tik_inst.if_scope(tp.dst_axis_num == 7):
            tp.rt_dst_addr.set_as(tp.tail_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                  tp.tail_dst_tuple[1] * tp.dst_jump_stride[1] + \
                                  tp.tail_dst_tuple[2] * tp.dst_jump_stride[2] + \
                                  tp.tail_dst_tuple[3] * tp.dst_jump_stride[3] + \
                                  tp.tail_dst_tuple[4] * tp.dst_jump_stride[4] + \
                                  tp.tail_dst_tuple[5] * tp.dst_jump_stride[5] + \
                                  tp.tail_dst_tuple[6] * tp.dst_jump_stride[6] + tp.n_dst_offset)

        with self.tik_inst.if_scope(tp.dst_axis_num == 6):
            tp.rt_dst_addr.set_as(tp.tail_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                  tp.tail_dst_tuple[1] * tp.dst_jump_stride[1] + \
                                  tp.tail_dst_tuple[2] * tp.dst_jump_stride[2] + \
                                  tp.tail_dst_tuple[3] * tp.dst_jump_stride[3] + \
                                  tp.tail_dst_tuple[4] * tp.dst_jump_stride[4] + \
                                  tp.tail_dst_tuple[5] * tp.dst_jump_stride[5] + tp.n_dst_offset)

        with self.tik_inst.if_scope(tp.dst_axis_num == 5):
            tp.rt_dst_addr.set_as(tp.tail_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                  tp.tail_dst_tuple[1] * tp.dst_jump_stride[1] + \
                                  tp.tail_dst_tuple[2] * tp.dst_jump_stride[2] + \
                                  tp.tail_dst_tuple[3] * tp.dst_jump_stride[3] + \
                                  tp.tail_dst_tuple[4] * tp.dst_jump_stride[4] + tp.n_dst_offset)

        with self.tik_inst.if_scope(tp.dst_axis_num == 4):
            tp.rt_dst_addr.set_as(tp.tail_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                  tp.tail_dst_tuple[1] * tp.dst_jump_stride[1] + \
                                  tp.tail_dst_tuple[2] * tp.dst_jump_stride[2] + \
                                  tp.tail_dst_tuple[3] * tp.dst_jump_stride[3] + tp.n_dst_offset)

        with self.tik_inst.if_scope(tp.dst_axis_num == 3):
            tp.rt_dst_addr.set_as(tp.tail_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                  tp.tail_dst_tuple[1] * tp.dst_jump_stride[1] + \
                                  tp.tail_dst_tuple[2] * tp.dst_jump_stride[2] + tp.n_dst_offset)

        with self.tik_inst.if_scope(tp.dst_axis_num == 2):
            tp.rt_dst_addr.set_as(tp.tail_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                  tp.tail_dst_tuple[1] * tp.dst_jump_stride[1] + tp.n_dst_offset)

        with self.tik_inst.if_scope(tp.dst_axis_num == 1):
            tp.rt_dst_addr.set_as(tp.tail_dst_tuple[0] * tp.dst_jump_stride[0] + tp.n_dst_offset)

    @staticmethod
    def _update_dst_addr_f2t(tp):
        tp.rt_dst_addr.set_as(tp.rt_dst_addr + tp.col_per_mc * tp.row_per_mr)

    def _update_src_tuple_t2f(self, tp, lr):
        with self.tik_inst.if_scope(tp.src_axis_num == 4):
            tp.rt_src_tuple[0].set_as((tp.row_offset + lr * tp.row_per_mr) % tp.src_jump_stride[0])
            tp.rt_src_tuple[1].set_as(((tp.row_offset + lr * tp.row_per_mr) // tp.src_jump_stride[0]) % \
                                      tp.src_jump_stride[0])
            tp.rt_src_tuple[2].set_as(((tp.row_offset + lr * tp.row_per_mr) // \
                                       (tp.src_jump_stride[0] * tp.src_jump_stride[1])) % tp.src_jump_stride[1])
            tp.rt_src_tuple[3].set_as((tp.row_offset + lr * tp.row_per_mr) // \
                                      (tp.src_jump_stride[0] * tp.src_jump_stride[1] * tp.src_jump_stride[2]))

        with self.tik_inst.if_scope(tp.src_axis_num == 3):
            tp.rt_src_tuple[0].set_as((tp.row_offset + lr * tp.row_per_mr) % tp.src_jump_stride[0])
            tp.rt_src_tuple[1].set_as(((tp.row_offset + lr * tp.row_per_mr) // tp.src_jump_stride[0]) % \
                                      tp.src_jump_stride[0])
            tp.rt_src_tuple[2].set_as((tp.row_offset + lr * tp.row_per_mr) // \
                                      (tp.src_jump_stride[0] * tp.src_jump_stride[1]))

        with self.tik_inst.if_scope(tp.src_axis_num == 2):
            tp.rt_src_tuple[0].set_as((tp.row_offset + lr * tp.row_per_mr) % tp.src_jump_stride[0])
            tp.rt_src_tuple[1].set_as(((tp.row_offset + lr * tp.row_per_mr) // tp.src_jump_stride[0]))

        with self.tik_inst.if_scope(tp.src_axis_num == 1):
            tp.rt_src_tuple[0].set_as(tp.row_offset + lr * tp.row_per_mr)

    @staticmethod
    def _update_dst_addr_t2f(tp, lr, bsu):
        tp.rt_dst_addr.set_as(tp.rt_dst_addr + lr * tp.row_per_mr - bsu)

    # --------------------------------------------------------
    #                         |                        |
    #             A           |          A             |  B
    # --------------------------------------------------------
    #                         |                        |
    #             A           |          A             |  B
    # --------------------------------------------------------
    #             C           |          C             |  D
    # --------------------------------------------------------

    # `A: major_col_major_batch`
    # `B: tail_col_major_batch`
    # `C: major_col_tail_batch`
    # `D: tail_col_tail_batch`

    def _reorder_s7_b8(self, tp, ub_input, ub_offset, is_tc=False, is_tr=False):
        ub_input_b8 = ub_input.reinterpret_cast_to("int8")

        with self.tik_inst.if_scope(is_tc):
            tp.col_reorder.set_as(tp.col_tc)
        with self.tik_inst.else_scope():
            tp.col_reorder.set_as(tp.col_per_mc)

        with self.tik_inst.if_scope(is_tr):
            tp.row_reorder.set_as(tp.row_tr)
        with self.tik_inst.else_scope():
            tp.row_reorder.set_as(tp.row_per_mr)

        repeat_cnt = tp.col_reorder // EPB32

        with self.tik_inst.for_range(0, tp.row_reorder // EPB32) as loop:
            src_addr_list = [ub_input_b8[loop * tp.col_reorder * EPB32 + tp.col_reorder * i] for i in range(EPB16)]

            dst_list_low = [ub_input_b8[tp.offset_b * 2 + loop * EPB32 + i * ROW_UNIT] for i in range(EPB16)]
            dst_list_high = [ub_input_b8[tp.offset_b * 2 + loop * EPB32 + (i + EPB16) * ROW_UNIT] for i in range(EPB16)]

            tp.src_stride_reorder.set_as(1)
            tp.dst_stride_reorder.set_as(ROW_UNIT)

            self.tik_inst.vnchwconv(False, False, dst_list_low, src_addr_list, repeat_cnt,
                                    tp.dst_stride_reorder, tp.src_stride_reorder)
            self.tik_inst.vnchwconv(False, True, dst_list_high, src_addr_list, repeat_cnt,
                                    tp.dst_stride_reorder, tp.src_stride_reorder)

        with self.tik_inst.for_range(0, tp.row_reorder // EPB32) as loop:
            src_addr_list = [ub_input_b8[loop * tp.col_reorder * EPB32 + \
                                         tp.col_reorder * (EPB16 + i)] for i in range(EPB16)]

            dst_list_low = [ub_input_b8[tp.offset_b * 2 + loop * EPB32 + i * ROW_UNIT] for i in range(EPB16)]
            dst_list_high = [ub_input_b8[tp.offset_b * 2 + loop * EPB32 + (i + EPB16) * ROW_UNIT] for i in range(EPB16)]

            tp.src_stride_reorder.set_as(1)
            tp.dst_stride_reorder.set_as(ROW_UNIT)

            self.tik_inst.vnchwconv(True, False, dst_list_low, src_addr_list, repeat_cnt,
                                    tp.dst_stride_reorder, tp.src_stride_reorder)
            self.tik_inst.vnchwconv(True, True, dst_list_high, src_addr_list, repeat_cnt,
                                    tp.dst_stride_reorder, tp.src_stride_reorder)


    def _reorder_s7_b16(self, tp, ub_input, ub_offset, is_tc=False, is_tr=False, is_s11=False):
        ub_input_fp16 = ub_input.reinterpret_cast_to("float16")

        with self.tik_inst.if_scope(is_tc):
            tp.col_reorder.set_as(tp.col_tc)
        with self.tik_inst.else_scope():
            tp.col_reorder.set_as(tp.col_per_mc)

        with self.tik_inst.if_scope(is_tr):
            tp.row_reorder.set_as(tp.row_tr)
        with self.tik_inst.else_scope():
            tp.row_reorder.set_as(tp.row_per_mr)

        with self.tik_inst.if_scope(is_s11):
            tp.dst_vnc_pad.set_as(ROW_UNIT + EPB16)
        with self.tik_inst.else_scope():
            tp.dst_vnc_pad.set_as(ROW_UNIT)

        repeat_cnt = tp.col_reorder // EPB16

        with self.tik_inst.for_range(0, tp.row_reorder // EPB16) as loop:
            src_addr_list = [ub_input_fp16[loop * tp.col_reorder * EPB16 + tp.col_reorder * i] for i in range(EPB16)]
            dst_addr_list = [ub_input_fp16[tp.offset_b + loop * EPB16 + tp.dst_vnc_pad * i] for i in range(EPB16)]

            with self.tik_inst.if_scope(repeat_cnt == 1):
                tp.src_stride_reorder.set_as(0)
                tp.dst_stride_reorder.set_as(0)
            with self.tik_inst.else_scope():
                tp.src_stride_reorder.set_as(1)
                tp.dst_stride_reorder.set_as(tp.dst_vnc_pad)

            self.tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt,
                                    tp.dst_stride_reorder, tp.src_stride_reorder)

    def _reorder_s7_b32(self, tp, ub_input, ub_offset, is_tc=False, is_tr=False):

        with self.tik_inst.if_scope(is_tc):
            tp.col_reorder.set_as(tp.col_tc)
        with self.tik_inst.else_scope():
            tp.col_reorder.set_as(tp.col_per_mc)

        with self.tik_inst.if_scope(is_tr):
            tp.row_reorder.set_as(tp.row_tr)
        with self.tik_inst.else_scope():
            tp.row_reorder.set_as(tp.row_per_mr)

        # do hwc to chw transfer
        inner_hw_len = 16 // self.fp16_times
        fp16_inner_hwc_len = 8 * tp.col_reorder * self.fp16_times
        ub_input_fp16 = ub_input.reinterpret_cast_to("float16")

        # first vnchwconv
        src_addr_list = [ub_input_fp16[fp16_inner_hwc_len * i] for i in range(EPB16)]
        dst_addr_list = [ub_input_fp16[tp.offset_b + EPB16 * i] for i in range(EPB16)]
        repeat_cnt = tp.col_reorder
        with self.tik_inst.if_scope(repeat_cnt == 1):
            tp.src_stride_reorder.set_as(0)
            tp.dst_stride_reorder.set_as(0)
        with self.tik_inst.else_scope():
            tp.src_stride_reorder.set_as(1)
            tp.dst_stride_reorder.set_as(16)

        self.tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt,
                                tp.dst_stride_reorder, tp.src_stride_reorder)

        # do hwc to chw transfer
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, inner_hw_len) as i:
                self.tik_inst.data_move(ub_input_fp16[i * self.fp16_times * EPB16],
                                        ub_input_fp16[tp.offset_b + i * tp.col_reorder * self.fp16_times * EPB16],
                                        0, tp.col_reorder, self.fp16_times, 0, (inner_hw_len - 1) * self.fp16_times)

        # second vnchwconv
        src_addr_list = [ub_input_fp16[EPB16 * i] for i in range(EPB16)]
        dst_addr_list = [ub_input_fp16[tp.offset_b + EPB16 * i] for i in range(EPB16)]
        with self.tik_inst.if_scope(repeat_cnt == 1):
            tp.src_stride_reorder.set_as(0)
            tp.dst_stride_reorder.set_as(0)
        with self.tik_inst.else_scope():
            tp.src_stride_reorder.set_as(16)
            tp.dst_stride_reorder.set_as(16)
        self.tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt,
                                tp.dst_stride_reorder, tp.src_stride_reorder)

    def _reorder_s7_b64(self, tp, ub_input, ub_offset, is_tc=False, is_tr=False):

        with self.tik_inst.if_scope(is_tc is True):
            tp.col_reorder.set_as(tp.col_tc)
        with self.tik_inst.else_scope():
            tp.col_reorder.set_as(tp.col_per_mc)

        with self.tik_inst.if_scope(is_tr is True):
            tp.row_reorder.set_as(tp.row_tr)
        with self.tik_inst.else_scope():
            tp.row_reorder.set_as(tp.row_per_mr)

        # do hwc to chw transfer
        inner_hw_len = 16 // self.fp16_times
        fp16_inner_hwc_len = 4 * tp.col_reorder * self.fp16_times
        ub_input_fp16 = ub_input.reinterpret_cast_to("float16")

        # first vnchwconv - up part
        src_addr_list = [ub_input_fp16[fp16_inner_hwc_len * i] for i in range(EPB16)]
        dst_addr_list = [ub_input_fp16[tp.offset_b + EPB16 * i] for i in range(EPB16)]
        repeat_cnt = tp.col_reorder

        with self.tik_inst.if_scope(repeat_cnt == 1):
            tp.src_stride_reorder.set_as(0)
            tp.dst_stride_reorder.set_as(0)
        with self.tik_inst.else_scope():
            tp.src_stride_reorder.set_as(1)
            tp.dst_stride_reorder.set_as(16)

        self.tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt,
                                tp.dst_stride_reorder, tp.src_stride_reorder)

        # first vnchwconv - down part
        src_addr_list = [ub_input_fp16[tp.col_reorder * ROW_UNIT // 2 * self.fp16_times + fp16_inner_hwc_len * i] \
                         for i in range(EPB16)]
        dst_addr_list = [ub_input_fp16[tp.offset_b + tp.col_reorder * ROW_UNIT // 2 * self.fp16_times + EPB16 * i] \
                         for i in range(EPB16)]
        repeat_cnt = tp.col_reorder

        with self.tik_inst.if_scope(repeat_cnt == 1):
            tp.src_stride_reorder.set_as(0)
            tp.dst_stride_reorder.set_as(0)
        with self.tik_inst.else_scope():
            tp.src_stride_reorder.set_as(1)
            tp.dst_stride_reorder.set_as(16)

        self.tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt,
                                tp.dst_stride_reorder, tp.src_stride_reorder)

        # do hwc to chw transfer - up part
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, 4) as i:
                self.tik_inst.data_move(ub_input_fp16[i * self.fp16_times * EPB16],
                                        ub_input_fp16[tp.offset_b + i * tp.col_reorder * self.fp16_times * EPB16],
                                        0, tp.col_reorder, self.fp16_times, 0, (2 * inner_hw_len - 1) * self.fp16_times)
            with self.tik_inst.for_range(0, 4) as i:
                self.tik_inst.data_move(ub_input_fp16[i * self.fp16_times * EPB16 + EPB16 * EPB16],
                                        ub_input_fp16[tp.offset_b + \
                                                      tp.col_reorder * EPB16 * EPB16 + \
                                                      i * tp.col_reorder * self.fp16_times * EPB16],
                                        0, tp.col_reorder, self.fp16_times, 0, (2 * inner_hw_len - 1) * self.fp16_times)

        # second vnchwconv
        src_addr_list = [ub_input_fp16[EPB16 * i] for i in range(EPB16)]
        dst_addr_list = [ub_input_fp16[tp.offset_b + EPB16 * i] for i in range(EPB16)]
        with self.tik_inst.if_scope(repeat_cnt == 1):
            tp.src_stride_reorder.set_as(0)
            tp.dst_stride_reorder.set_as(0)
        with self.tik_inst.else_scope():
            tp.src_stride_reorder.set_as(16)
            tp.dst_stride_reorder.set_as(16)
        self.tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt * 2,
                                tp.dst_stride_reorder, tp.src_stride_reorder)

    def _reorder_s7(self, tp, ub_input, ub_offset, is_tc=False, is_tr=False):
        with self.tik_inst.if_scope(self.fp16_times == 1):
            self._reorder_s7_b16(tp, ub_input, ub_offset, is_tc, is_tr)
        with self.tik_inst.if_scope(self.fp16_times == 2):
            self._reorder_s7_b32(tp, ub_input, ub_offset, is_tc, is_tr)
        with self.tik_inst.if_scope(self.fp16_times == 4):
            self._reorder_s7_b64(tp, ub_input, ub_offset, is_tc, is_tr)

    def _data_move_pad_in_process(self, data_move_pad_info):
        # burst_len: unit dtype
        # dst_gap: unit dtype
        # src_gap: unit dtype
        # so we need convert above parameters into Byte or block
        dst, src, dst_addr, src_addr, repeat, burst_len, dst_gap, src_gap = data_move_pad_info
        with self.tik_inst.if_scope(repeat <= MAX_DATA_MOVE_PAD_REPEATS):
            self.tik_inst.data_move_pad(dst[dst_addr],
                                        src[src_addr],
                                        repeat,
                                        burst_len * self.b8_times,
                                        dst_gap // self.ele_per_block,
                                        src_gap * self.b8_times)

    def _data_move_pad_out_process(self, data_move_pad_info):
        # burst_len: unit dtype
        # dst_gap: unit dtype
        # src_gap: unit dtype
        # so we need convert above parameters into Byte or block
        dst, src, dst_addr, src_addr, repeat, burst_len, dst_gap, src_gap = data_move_pad_info
        with self.tik_inst.if_scope(repeat <= MAX_DATA_MOVE_PAD_REPEATS):
            self.tik_inst.data_move_pad(dst[dst_addr],
                                        src[src_addr],
                                        repeat,
                                        burst_len * self.b8_times,
                                        dst_gap * self.b8_times,
                                        src_gap // self.ele_per_block)

    def _copy_in_pad_loop_greater_s7(self, tp, ub_input, data_move_info):
        ub_offset, lc, loops, burst_len, bsl = data_move_info
        ub_offset.set_as(0)
        self.head_repeat.set_as((tp.src_jump_factor[0] - (tp.rt_src_tuple[0] % tp.src_jump_factor[0])) %\
                                tp.src_jump_factor[0])
        self.mid_repeats.set_as((loops - self.head_repeat) // tp.src_jump_factor[0])
        self.tail_repeat.set_as((loops - self.head_repeat - self.mid_repeats * tp.src_jump_factor[0]) %\
                                tp.src_jump_factor[0])
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.if_scope(self.head_repeat > 0):
                data_move_pad_info = (ub_input, self.data_in, ub_offset, self._get_src_addr_s7(tp, 0, lc, 0, bsl),
                                      self.head_repeat, burst_len, 0, tp.src_jump_stride[0] - burst_len)
                self._data_move_pad_in_process(data_move_pad_info)
                self._update_tuple(tp.src_axis_num, tp.rt_src_tuple, tp.src_jump_factor, self.head_repeat)
                ub_offset.set_as(ub_offset + burst_len * self.head_repeat)
            with self.tik_inst.for_range(0, self.mid_repeats):
                data_move_pad_info = (ub_input, self.data_in, ub_offset, self._get_src_addr_s7(tp, 0, lc, 0, bsl),
                                      tp.src_jump_factor[0], burst_len, 0, tp.src_jump_stride[0] - burst_len)
                self._data_move_pad_in_process(data_move_pad_info)
                self._update_tuple(tp.src_axis_num, tp.rt_src_tuple, tp.src_jump_factor, tp.src_jump_factor[0])
                ub_offset.set_as(ub_offset + burst_len * tp.src_jump_factor[0])
            with self.tik_inst.if_scope(self.tail_repeat > 0):
                data_move_pad_info = (ub_input, self.data_in, ub_offset, self._get_src_addr_s7(tp, 0, lc, 0, bsl),
                                      self.tail_repeat, burst_len, 0, tp.src_jump_stride[0] - burst_len)
                self._data_move_pad_in_process(data_move_pad_info)
                self._update_tuple(tp.src_axis_num, tp.rt_src_tuple, tp.src_jump_factor, self.tail_repeat)
                ub_offset.set_as(ub_offset + burst_len * self.tail_repeat)

    def _copy_in_pad_loop_smaller_s7(self, tp, ub_input, data_move_info):
        ub_offset, lc, loops, burst_len, bsl = data_move_info
        ub_offset.set_as(0)
        # head_repeat in [0, tp.src_jump_factor[0])
        self.head_repeat.set_as((tp.src_jump_factor[0] - (tp.rt_src_tuple[0] % tp.src_jump_factor[0])) %\
                                tp.src_jump_factor[0])
        # head_repeat >= loops means after loops times data move, it will not reach jump point,
        # so we can set loops as the repeat. Otherwise, the loops spans the jump point, we need split it by jump point.
        with self.tik_inst.if_scope(self.head_repeat >= loops):
            self.head_repeat.set_as(loops)
            self.tail_repeat.set_as(0)
        with self.tik_inst.else_scope():
            self.tail_repeat.set_as(loops - self.head_repeat)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.if_scope(self.head_repeat > 0):
                data_move_pad_info = (ub_input, self.data_in, ub_offset, self._get_src_addr_s7(tp, 0, lc, 0, bsl),
                                      self.head_repeat, burst_len, 0, tp.src_jump_stride[0] - burst_len)
                self._data_move_pad_in_process(data_move_pad_info)
                self._update_tuple(tp.src_axis_num, tp.rt_src_tuple, tp.src_jump_factor, self.head_repeat)
                ub_offset.set_as(ub_offset + burst_len * self.head_repeat)
            with self.tik_inst.if_scope(self.tail_repeat > 0):
                data_move_pad_info = (ub_input, self.data_in, ub_offset, self._get_src_addr_s7(tp, 0, lc, 0, bsl),
                                      self.tail_repeat, burst_len, 0, tp.src_jump_stride[0] - burst_len)
                self._data_move_pad_in_process(data_move_pad_info)
                self._update_tuple(tp.src_axis_num, tp.rt_src_tuple, tp.src_jump_factor, self.tail_repeat)
                ub_offset.set_as(ub_offset + burst_len * self.tail_repeat)

    def _copy_in_major_col_major_row_pad(self, tp, ub_input, ub_offset, iters):
        ln, lc, lr = iters
        data_move_info = (ub_offset, lc, tp.row_per_mr, tp.col_per_mc, 0)
        with self.tik_inst.if_scope(tp.row_per_mr >= tp.src_jump_factor[0]):
            self._copy_in_pad_loop_greater_s7(tp, ub_input, data_move_info)
        with self.tik_inst.else_scope():
            self._copy_in_pad_loop_smaller_s7(tp, ub_input, data_move_info)

    def _copy_in_major_col_major_row(self, tp, ub_input, ub_offset, ln, lc, lr):
        ub_offset.set_as(0)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, tp.row_per_mr) as line:
                self.tik_inst.data_move(ub_input[ub_offset],
                                        self.data_in[self._get_src_addr_s7(tp, ln, lc, lr, 0)],
                                        0,
                                        1,
                                        tp.col_per_mc // self.ele_per_block,
                                        0,
                                        0)
                self._update_tuple(tp.src_axis_num, tp.rt_src_tuple, tp.src_jump_factor)
                ub_offset.set_as(ub_offset + tp.col_per_mc)
            ub_offset.set_as(ub_offset // self.ele_per_block)

    def _copy_in_major_col_tail_row_pad(self, tp, ub_input, ub_offset, iters):
        ln, lc, lr = iters
        data_move_info = (ub_offset, lc, tp.row_tr, tp.col_per_mc, 0)
        with self.tik_inst.if_scope(tp.row_tr >= tp.src_jump_factor[0]):
            self._copy_in_pad_loop_greater_s7(tp, ub_input, data_move_info)
        with self.tik_inst.else_scope():
            self._copy_in_pad_loop_smaller_s7(tp, ub_input, data_move_info)

    def _copy_in_major_col_tail_row(self, tp, ub_input, ub_offset, ln, lc, lr):
        ub_offset.set_as(0)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, tp.row_tr) as line:
                self.tik_inst.data_move(ub_input[ub_offset],
                                        self.data_in[self._get_src_addr_s7(tp, ln, lc, lr, 0)],
                                        0,
                                        1,
                                        tp.col_per_mc // self.ele_per_block,
                                        0,
                                        0)
                self._update_tuple(tp.src_axis_num, tp.rt_src_tuple, tp.src_jump_factor)
                ub_offset.set_as(ub_offset + tp.col_per_mc)
            ub_offset.set_as(ub_offset // self.ele_per_block)

    def _copy_in_tail_col_major_row_pad(self, tp, ub_input, ub_offset, iters):
        ln, lc, lr = iters
        data_move_info = (ub_offset, lc, tp.row_per_mr, tp.col_tc, tp.back_step_left)
        with self.tik_inst.if_scope(tp.row_per_mr >= tp.src_jump_factor[0]):
            self._copy_in_pad_loop_greater_s7(tp, ub_input, data_move_info)
        with self.tik_inst.else_scope():
            self._copy_in_pad_loop_smaller_s7(tp, ub_input, data_move_info)
            
    def _copy_in_tail_col_major_row(self, tp, ub_input, ub_offset, ln, lc, lr):
        ub_offset.set_as(0)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, tp.row_per_mr) as line:
                self.tik_inst.data_move(ub_input[ub_offset],
                                        self.data_in[self._get_src_addr_s7(tp, ln, lc, lr, tp.back_step_left)],
                                        0,
                                        1,
                                        tp.col_tc // self.ele_per_block,
                                        0,
                                        0)
                self._update_tuple(tp.src_axis_num, tp.rt_src_tuple, tp.src_jump_factor)
                ub_offset.set_as(ub_offset + tp.col_tc)
            ub_offset.set_as(ub_offset // self.ele_per_block)

    def _copy_in_tail_col_tail_row_pad(self, tp, ub_input, ub_offset, iters):
        ln, lc, lr = iters
        data_move_info = (ub_offset, lc, tp.row_tr, tp.col_tc, tp.back_step_left)
        with self.tik_inst.if_scope(tp.row_tr >= tp.src_jump_factor[0]):
            self._copy_in_pad_loop_greater_s7(tp, ub_input, data_move_info)
        with self.tik_inst.else_scope():
            self._copy_in_pad_loop_smaller_s7(tp, ub_input, data_move_info)

    def _copy_in_tail_col_tail_row(self, tp, ub_input, ub_offset, ln, lc, lr):
        ub_offset.set_as(0)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, tp.row_tr) as line:
                self.tik_inst.data_move(ub_input[ub_offset],
                                        self.data_in[self._get_src_addr_s7(tp, ln, lc, lr, tp.back_step_left)],
                                        0,
                                        1,
                                        tp.col_tc // self.ele_per_block,
                                        0,
                                        0)
                self._update_tuple(tp.src_axis_num, tp.rt_src_tuple, tp.src_jump_factor)
                ub_offset.set_as(ub_offset + tp.col_tc)
            ub_offset.set_as(ub_offset // self.ele_per_block)

    def _copy_out_pad_loop_greater_s7(self, tp, ub_input, data_move_info):
        ub_offset, lr, loops, burst_len, bsu, ub_stride = data_move_info
        ub_offset.set_as(tp.offset_b // self.fp16_times)
        self.head_repeat.set_as((tp.dst_jump_factor[0] - (tp.rt_dst_tuple[0] % tp.dst_jump_factor[0])) %\
                                tp.dst_jump_factor[0])
        self.mid_repeats.set_as((loops - self.head_repeat) // tp.dst_jump_factor[0])
        self.tail_repeat.set_as((loops - self.head_repeat - self.mid_repeats * tp.dst_jump_factor[0]) %\
                                tp.dst_jump_factor[0])
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.if_scope(self.head_repeat > 0):
                data_move_pad_info = (self.data_out, ub_input, self._get_dst_addr_s7(tp, 0, 0, lr, 0, 0, bsu),
                                      ub_offset, self.head_repeat, burst_len, tp.dst_jump_stride[0] - burst_len,
                                      ub_stride - burst_len)
                self._data_move_pad_out_process(data_move_pad_info)
                ub_offset.set_as(ub_offset + ub_stride * self.head_repeat)
                self._update_tuple(tp.dst_axis_num, tp.rt_dst_tuple, tp.dst_jump_factor, self.head_repeat)
            with self.tik_inst.for_range(0, self.mid_repeats):
                data_move_pad_info = (self.data_out, ub_input, self._get_dst_addr_s7(tp, 0, 0, lr, 0, 0, bsu),
                                      ub_offset, tp.dst_jump_factor[0], burst_len, tp.dst_jump_stride[0] - burst_len,
                                      ub_stride - burst_len)
                self._data_move_pad_out_process(data_move_pad_info)
                ub_offset.set_as(ub_offset + ub_stride * tp.dst_jump_factor[0])
                self._update_tuple(tp.dst_axis_num, tp.rt_dst_tuple, tp.dst_jump_factor, tp.dst_jump_factor[0])
            with self.tik_inst.if_scope(self.tail_repeat > 0):
                data_move_pad_info = (self.data_out, ub_input, self._get_dst_addr_s7(tp, 0, 0, lr, 0, 0, bsu),
                                      ub_offset, self.tail_repeat, burst_len, tp.dst_jump_stride[0] - burst_len,
                                      ub_stride - burst_len)
                self._data_move_pad_out_process(data_move_pad_info)
                ub_offset.set_as(ub_offset + ub_stride * self.tail_repeat)
                self._update_tuple(tp.dst_axis_num, tp.rt_dst_tuple, tp.dst_jump_factor, self.tail_repeat)

    def _copy_out_pad_loop_smaller_s7(self, tp, ub_input, data_move_info):
        ub_offset, lr, loops, burst_len, bsu, ub_stride = data_move_info
        ub_offset.set_as(tp.offset_b // self.fp16_times)
        # head_repeat in [0, tp.dst_jump_factor[0])
        self.head_repeat.set_as((tp.dst_jump_factor[0] - (tp.rt_dst_tuple[0] % tp.dst_jump_factor[0])) %\
                                tp.dst_jump_factor[0])
        # head_repeat >= loops means after loops times data move, it will not reach jump point,
        # so we can set loops as the repeat. Otherwise, the loops spans the jump point, we need split it by jump point.
        with self.tik_inst.if_scope(self.head_repeat >= loops):
            self.head_repeat.set_as(loops)
            self.tail_repeat.set_as(0)
        with self.tik_inst.else_scope():
            self.tail_repeat.set_as(loops - self.head_repeat)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.if_scope(self.head_repeat > 0):
                data_move_pad_info = (self.data_out, ub_input, self._get_dst_addr_s7(tp, 0, 0, lr, 0, 0, bsu),
                                      ub_offset, self.head_repeat, burst_len, tp.dst_jump_stride[0] - burst_len,
                                      ub_stride - burst_len)
                self._data_move_pad_out_process(data_move_pad_info)
                ub_offset.set_as(ub_offset + ub_stride * self.head_repeat)
                self._update_tuple(tp.dst_axis_num, tp.rt_dst_tuple, tp.dst_jump_factor, self.head_repeat)
            with self.tik_inst.if_scope(self.tail_repeat > 0):
                data_move_pad_info = (self.data_out, ub_input, self._get_dst_addr_s7(tp, 0, 0, lr, 0, 0, bsu),
                                      ub_offset, self.tail_repeat, burst_len, tp.dst_jump_stride[0] - burst_len,
                                      ub_stride - burst_len)
                self._data_move_pad_out_process(data_move_pad_info)
                ub_offset.set_as(ub_offset + ub_stride * self.tail_repeat)
                self._update_tuple(tp.dst_axis_num, tp.rt_dst_tuple, tp.dst_jump_factor, self.tail_repeat)

    def _copy_out_major_col_major_row_pad(self, tp, ub_input, ub_offset, iters):
        ln, lc, lr = iters
        data_move_info = (ub_offset, lr, tp.col_per_mc, tp.row_per_mr, 0, ROW_UNIT)
        with self.tik_inst.if_scope(tp.col_per_mc >= tp.dst_jump_factor[0]):
            self._copy_out_pad_loop_greater_s7(tp, ub_input, data_move_info)
        with self.tik_inst.else_scope():
            self._copy_out_pad_loop_smaller_s7(tp, ub_input, data_move_info)

    def _copy_out_major_col_major_row(self, tp, ub_input, ub_offset, ln, lc, lr):
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, tp.col_per_mc) as col_id:
                self.tik_inst.data_move(self.data_out[self._get_dst_addr_s7(tp, ln, lc, lr, col_id, 0, 0)],
                                        ub_input[tp.offset_b // self.fp16_times + col_id * ROW_UNIT],
                                        0, 1, tp.row_per_mr // self.ele_per_block, 0, 0)
                self._update_tuple(tp.dst_axis_num, tp.rt_dst_tuple, tp.dst_jump_factor)

    def _copy_out_major_col_tail_row_pad(self, tp, ub_input, ub_offset, iters):
        ln, lc, lr = iters
        data_move_info = (ub_offset, lr, tp.col_per_mc, tp.row_tr, tp.back_step_up, ROW_UNIT)
        with self.tik_inst.if_scope(tp.col_per_mc >= tp.dst_jump_factor[0]):
            self._copy_out_pad_loop_greater_s7(tp, ub_input, data_move_info)
        with self.tik_inst.else_scope():
            self._copy_out_pad_loop_smaller_s7(tp, ub_input, data_move_info)

    def _copy_out_major_col_tail_row(self, tp, ub_input, ub_offset, ln, lc, lr):
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, tp.col_per_mc) as col_id:
                self.tik_inst.data_move(self.data_out[self._get_dst_addr_s7(tp, ln, lc, lr, col_id,
                                                      0, tp.back_step_up)],
                                        ub_input[tp.offset_b // self.fp16_times + col_id * ROW_UNIT],
                                        0, 1, tp.row_tr // self.ele_per_block, 0, 0)
                self._update_tuple(tp.dst_axis_num, tp.rt_dst_tuple, tp.dst_jump_factor)

    def _copy_out_tail_col_major_row_pad(self, tp, ub_input, ub_offset, iters):
        ln, lc, lr = iters
        data_move_info = (ub_offset, lr, tp.col_tc, tp.row_per_mr, 0, ROW_UNIT)
        with self.tik_inst.if_scope(tp.col_tc >= tp.dst_jump_factor[0]):
            self._copy_out_pad_loop_greater_s7(tp, ub_input, data_move_info)
        with self.tik_inst.else_scope():
            self._copy_out_pad_loop_smaller_s7(tp, ub_input, data_move_info)

    def _copy_out_tail_col_major_row(self, tp, ub_input, ub_offset, ln, lc, lr):
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, tp.col_tc) as col_id:
                with self.tik_inst.if_scope(col_id >= tp.back_step_left):
                    self.tik_inst.data_move(self.data_out[self._get_dst_addr_s7(tp, ln, lc, lr, col_id,
                                                                             tp.back_step_left, 0)],
                                            ub_input[tp.offset_b // self.fp16_times + col_id * ROW_UNIT],
                                            0, 1, tp.row_per_mr // self.ele_per_block, 0, 0)
                self._update_tuple(tp.dst_axis_num, tp.rt_dst_tuple, tp.dst_jump_factor)

    def _copy_out_tail_col_tail_row_pad(self, tp, ub_input, ub_offset, iters):
        ln, lc, lr = iters
        data_move_info = (ub_offset, lr, tp.col_tc, tp.row_tr, tp.back_step_up, ROW_UNIT)
        with self.tik_inst.if_scope(tp.col_tc >= tp.dst_jump_factor[0]):
            self._copy_out_pad_loop_greater_s7(tp, ub_input, data_move_info)
        with self.tik_inst.else_scope():
            self._copy_out_pad_loop_smaller_s7(tp, ub_input, data_move_info)

    def _copy_out_tail_col_tail_row(self, tp, ub_input, ub_offset, ln, lc, lr):
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, tp.col_tc) as col_id:
                with self.tik_inst.if_scope(col_id >= tp.back_step_left):
                    self.tik_inst.data_move(self.data_out[self._get_dst_addr_s7(tp, ln, lc, lr, col_id,
                                                                             tp.back_step_left, tp.back_step_up)],
                                            ub_input[tp.offset_b // self.fp16_times + col_id * ROW_UNIT],
                                            0, 1, tp.row_tr // self.ele_per_block, 0, 0)
                self._update_tuple(tp.dst_axis_num, tp.rt_dst_tuple, tp.dst_jump_factor)

    def _reorder_university_f2t(self, tp, ub_input, ub_offset, col_ele_num, row_ele_num, mode):
        # step1. make all elements in the first col
        ub_input_b16 = ub_input.reinterpret_cast_to("int16")
        src_ele_num_in_fp16 = self._get_src_size()  # minus 16 avoid bank conflict
        src_list = [ub_input_b16[src_ele_num_in_fp16 * i] for i in range(EPB16)]
        dst_list_intermediate = [ub_input_b16[tp.offset_b + EPB16 * i] for i in range(EPB16)]
        with self.tik_inst.if_scope(ub_offset == 1):
            self.tik_inst.vnchwconv(False, False, dst_list_intermediate, src_list, 1, 0, 0)
        with self.tik_inst.if_scope(ub_offset != 1):
            self.tik_inst.vnchwconv(False, False, dst_list_intermediate, src_list, ub_offset, EPB16, 1)

        # step2. move output elements together
        with self.tik_inst.if_scope(mode == 0):
            #f2t
            with self.tik_inst.if_scope(tik.all(self.fp16_times == 1, row_ele_num < 32)):
                with self.tik_inst.new_stmt_scope(disable_sync=True):
                    with self.tik_inst.for_range(0, row_ele_num) as i:
                        self.tik_inst.vor(128,
                                          ub_input_b16[i * EPB16],
                                          ub_input_b16[tp.offset_b + i * col_ele_num * EPB16],
                                          self.ub_input_b16_vor,
                                          col_ele_num // 8,
                                          row_ele_num,
                                          1,
                                          1,
                                          row_ele_num * 8,
                                          8,
                                          0)

            with self.tik_inst.if_scope(tik.all(self.fp16_times == 1, row_ele_num >= 32)):
                loop_num = self.tik_inst.Scalar("int32")
                tail_num = self.tik_inst.Scalar("int32")
                loop_num.set_as(row_ele_num // 8)
                tail_num.set_as(row_ele_num % 8)
                with self.tik_inst.new_stmt_scope(disable_sync=True):
                    with self.tik_inst.for_range(0, loop_num) as i:
                        self.tik_inst.vor(128,
                                          ub_input_b16[i * 8 * EPB16],
                                          ub_input_b16[tp.offset_b + i * col_ele_num * 8 * EPB16],
                                          self.ub_input_b16_vor,
                                          col_ele_num,
                                          1,
                                          col_ele_num,
                                          1,
                                          row_ele_num,
                                          1,
                                          0)
                    with self.tik_inst.if_scope(tail_num != 0):
                        self.tik_inst.vor(tail_num * 16,
                                          ub_input_b16[loop_num * 8 * EPB16],
                                          ub_input_b16[tp.offset_b + loop_num * col_ele_num * 8 * EPB16],
                                          self.ub_input_b16_vor,
                                          col_ele_num,
                                          1,
                                          col_ele_num,
                                          1,
                                          row_ele_num,
                                          1,
                                          0)

            with self.tik_inst.if_scope(self.fp16_times == 2):
                with self.tik_inst.new_stmt_scope(disable_sync=True):
                    with self.tik_inst.for_range(0, row_ele_num) as i:
                        self.tik_inst.data_move(ub_input_b16[i * self.fp16_times * EPB16],
                                                ub_input_b16[tp.offset_b + i * col_ele_num * self.fp16_times * EPB16],
                                                0,
                                                col_ele_num,
                                                self.fp16_times,
                                                0,
                                                row_ele_num * self.fp16_times - self.fp16_times)
        with self.tik_inst.else_scope():
            # t2f
            with self.tik_inst.new_stmt_scope(disable_sync=True):
                with self.tik_inst.for_range(0, col_ele_num) as i:

                    self.tik_inst.data_move(ub_input_b16[i * row_ele_num * self.fp16_times * EPB16],
                                            ub_input_b16[tp.offset_b + i * self.fp16_times * EPB16],
                                            0,
                                            row_ele_num,
                                            self.fp16_times,
                                            col_ele_num * self.fp16_times - self.fp16_times,
                                            0)

        # step3. make all elements in the first col be in memory of contiguous
        src_list_intermediate = [ub_input_b16[EPB16 * i] for i in range(EPB16)]
        dst_list_finally = [ub_input_b16[tp.offset_b + self._get_dst_size() * 16 * i] for i in range(EPB16)]

        with self.tik_inst.if_scope(ub_offset == 1):
            self.tik_inst.vnchwconv(False, False, dst_list_finally, src_list_intermediate, 1, 0, 0)
        with self.tik_inst.if_scope(ub_offset != 1):
            self.tik_inst.vnchwconv(False, False, dst_list_finally, src_list_intermediate, ub_offset, 1, EPB16)

    def _copy_in_major_col_f2t_pad(self, tp, ub_input, ub_offset, ln, lc):
        ub_offset.set_as(0)
        self._init_src_tuple(tp)
        data_move_info = (ub_offset, lc, tp.row_per_mr, tp.col_per_mc, 0)
        with self.tik_inst.if_scope(tp.row_per_mr >= tp.src_jump_factor[0]):
            self._copy_in_pad_loop_greater_s7(tp, ub_input, data_move_info)
        with self.tik_inst.else_scope():
            self._copy_in_pad_loop_smaller_s7(tp, ub_input, data_move_info)
        ub_offset.set_as(ub_offset // self.ele_per_block)

    def _copy_in_major_col_f2t(self, tp, ub_input, ub_offset, ln, lc):
        ub_offset.set_as(0)
        self._init_src_tuple(tp)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, tp.row_per_mr) as line:
                self.tik_inst.data_move(ub_input[ub_offset * self.ele_per_block],
                                        self.data_in[self._get_src_addr_s7(tp, ln, lc, 0, 0)],
                                        0,
                                        1,
                                        tp.col_per_mc // self.ele_per_block,
                                        0,
                                        0)
                self._update_tuple(tp.src_axis_num, tp.rt_src_tuple, tp.src_jump_factor)
                ub_offset.set_as(ub_offset + tp.col_per_mc // self.ele_per_block)

    def _copy_out_major_col_f2t(self, tp, ub_input, ub_offset, lc):
        self.tik_inst.data_move(self.data_out[tp.rt_dst_addr],
                                ub_input[tp.offset_b // self.fp16_times],
                                0,
                                1,
                                (tp.col_per_mc * tp.row_per_mr) // self.ele_per_block,
                                0,
                                0)
        self._update_dst_addr_f2t(tp)

    def _copy_in_tail_col_f2t_pad(self, tp, ub_input, ub_offset, ln, lc):
        ub_offset.set_as(0)
        self._init_src_tuple(tp)
        data_move_info = (ub_offset, lc, tp.row_per_mr, tp.col_tc, tp.back_step_left)
        with self.tik_inst.if_scope(tp.row_per_mr >= tp.src_jump_factor[0]):
            self._copy_in_pad_loop_greater_s7(tp, ub_input, data_move_info)
        with self.tik_inst.else_scope():
            self._copy_in_pad_loop_smaller_s7(tp, ub_input, data_move_info)
        ub_offset.set_as(ub_offset // self.ele_per_block)

    def _copy_in_tail_col_f2t(self, tp, ub_input, ub_offset, ln, lc):
        ub_offset.set_as(0)
        self._init_src_tuple(tp)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, tp.row_per_mr) as line:
                self.tik_inst.data_move(ub_input[ub_offset * self.ele_per_block],
                                        self.data_in[self._get_src_addr_s7(tp, ln, lc, 0, tp.back_step_left)],
                                        0,
                                        1,
                                        tp.col_tc // self.ele_per_block,
                                        0,
                                        0)
                self._update_tuple(tp.src_axis_num, tp.rt_src_tuple, tp.src_jump_factor)
                ub_offset.set_as(ub_offset + tp.col_tc // self.ele_per_block)

    def _copy_out_tail_col_f2t(self, tp, ub_input, ub_offset, ln, lc):
        self._tail_dst_addr_f2t(tp, ln)
        self.tik_inst.data_move(self.data_out[tp.rt_dst_addr],
                                ub_input[tp.offset_b // self.fp16_times],
                                0,
                                1,
                                (tp.col_tc * tp.row_per_mr) // self.ele_per_block,
                                0,
                                0)

    def _copy_in_major_row_t2f(self, tp, ub_input, ub_offset, ln, lr):
        ub_offset.set_as(0)
        self._update_src_tuple_t2f(tp, lr)
        self.tik_inst.data_move(ub_input[ub_offset * self.ele_per_block],
                                self.data_in[self._get_src_addr_s7(tp, ln, 0, lr, 0)],
                                0,
                                1,
                                (tp.col_per_mc * tp.row_per_mr) // self.ele_per_block,
                                0,
                                0)
        ub_offset.set_as(ub_offset + (tp.col_per_mc * tp.row_per_mr) // self.ele_per_block)

    def _copy_out_major_row_t2f_pad(self, tp, ub_input, ub_offset, ln, lr):
        self._init_dst_tuple(tp)
        data_move_info = (ub_offset, lr, tp.col_per_mc, tp.row_per_mr, 0, tp.row_per_mr)
        with self.tik_inst.if_scope(tp.col_per_mc >= tp.dst_jump_factor[0]):
            self._copy_out_pad_loop_greater_s7(tp, ub_input, data_move_info)
        with self.tik_inst.else_scope():
            self._copy_out_pad_loop_smaller_s7(tp, ub_input, data_move_info)

    def _copy_out_major_row_t2f(self, tp, ub_input, ub_offset, ln, lr):
        self._init_dst_tuple(tp)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, tp.col_per_mc) as col_id:
                self.tik_inst.data_move(self.data_out[self._get_dst_addr_s7(tp, ln, 0, lr, col_id, 0, 0)],
                                        ub_input[tp.offset_b // self.fp16_times + col_id * tp.row_per_mr],
                                        0,
                                        1,
                                        tp.row_per_mr // self.ele_per_block,
                                        0,
                                        0)
                self._update_tuple(tp.dst_axis_num, tp.rt_dst_tuple, tp.dst_jump_factor)

    def _copy_in_tail_row_t2f(self, tp, ub_input, ub_offset, ln, lr):
        ub_offset.set_as(0)
        self._tail_src_tuple(tp)
        self.tik_inst.data_move(ub_input[ub_offset * self.ele_per_block],
                                self.data_in[self._get_src_addr_s7(tp, ln, 0, lr, 0)],
                                0,
                                1,
                                (tp.col_per_mc * tp.row_tr) // self.ele_per_block,
                                0,
                                0)
        ub_offset.set_as(ub_offset + (tp.col_per_mc * tp.row_tr) // self.ele_per_block)

    def _copy_out_tail_row_t2f_pad(self, tp, ub_input, ub_offset, ln, lr):
        self._init_dst_tuple(tp)
        data_move_info = (ub_offset, lr, tp.col_per_mc, tp.row_tr, tp.back_step_up, tp.row_tr)
        with self.tik_inst.if_scope(tp.col_per_mc >= tp.dst_jump_factor[0]):
            self._copy_out_pad_loop_greater_s7(tp, ub_input, data_move_info)
        with self.tik_inst.else_scope():
            self._copy_out_pad_loop_smaller_s7(tp, ub_input, data_move_info)

    def _copy_out_tail_row_t2f(self, tp, ub_input, ub_offset, ln, lr):
        self._init_dst_tuple(tp)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, tp.col_per_mc) as col_id:
                self.tik_inst.data_move(self.data_out[self._get_dst_addr_s7(tp, ln, 0, lr, col_id, 0, tp.back_step_up)],
                                        ub_input[tp.offset_b // self.fp16_times + col_id * tp.row_tr],
                                        0,
                                        1,
                                        tp.row_tr // self.ele_per_block,
                                        0,
                                        0)
                self._update_tuple(tp.dst_axis_num, tp.rt_dst_tuple, tp.dst_jump_factor)

    def _major_col_process_s7(self, tp, ub_input, ub_offset, ln):
        with self.tik_inst.for_range(0, tp.loop_on_mc) as lc:
            self._init_src_tuple(tp)
            with self.tik_inst.for_range(0, tp.loop_on_mr) as lr:
                self._restore_dst_tuple(tp)
                if self.is_mte_pad and self.b8_times != EPB8:
                    iters = (ln, lc, lr)
                    self._copy_in_major_col_major_row_pad(tp, ub_input, ub_offset, iters)
                    self._reorder_s7(tp, ub_input, ub_offset, False, False)
                    self._copy_out_major_col_major_row_pad(tp, ub_input, ub_offset, iters)
                else:
                    self._copy_in_major_col_major_row(tp, ub_input, ub_offset, ln, lc, lr)
                    self._reorder_s7(tp, ub_input, ub_offset, False, False)
                    self._copy_out_major_col_major_row(tp, ub_input, ub_offset, ln, lc, lr)

            with self.tik_inst.if_scope(tp.row_tr != 0):
                self._tail_src_tuple(tp)
                self._restore_dst_tuple(tp)
                if self.is_mte_pad and self.b8_times != EPB8:
                    iters = (ln, lc, tp.loop_on_mr)
                    self._copy_in_major_col_tail_row_pad(tp, ub_input, ub_offset, iters)
                    self._reorder_s7(tp, ub_input, ub_offset, False, True)
                    self._copy_out_major_col_tail_row_pad(tp, ub_input, ub_offset, iters)
                else:
                    self._copy_in_major_col_tail_row(tp, ub_input, ub_offset, ln, lc, tp.loop_on_mr)
                    self._reorder_s7(tp, ub_input, ub_offset, False, True)
                    self._copy_out_major_col_tail_row(tp, ub_input, ub_offset, ln, lc, tp.loop_on_mr)
            self._backup_dst_tuple(tp)

    def _major_row_process_s7(self, tp, ub_input, ub_offset, ln):
        with self.tik_inst.if_scope(tp.col_tc != 0):
            self._init_src_tuple(tp)
            with self.tik_inst.for_range(0, tp.loop_on_mr) as lr:
                self._tail_dst_tuple(tp)
                if self.is_mte_pad and self.b8_times != EPB8:
                    iters = (ln, tp.loop_on_mc, lr)
                    self._copy_in_tail_col_major_row_pad(tp, ub_input, ub_offset, iters)
                    self._reorder_s7(tp, ub_input, ub_offset, True, False)
                    self._copy_out_tail_col_major_row_pad(tp, ub_input, ub_offset, iters)
                else:
                    self._copy_in_tail_col_major_row(tp, ub_input, ub_offset, ln, tp.loop_on_mc, lr)
                    self._reorder_s7(tp, ub_input, ub_offset, True, False)
                    self._copy_out_tail_col_major_row(tp, ub_input, ub_offset, ln, tp.loop_on_mc, lr)

        with self.tik_inst.if_scope(tp.col_tc != 0):
            with self.tik_inst.if_scope(tp.row_tr != 0):
                self._tail_src_tuple(tp)
                self._tail_dst_tuple(tp)
                if self.is_mte_pad and self.b8_times != EPB8:
                    iters = (ln, tp.loop_on_mc, tp.loop_on_mr)
                    self._copy_in_tail_col_tail_row_pad(tp, ub_input, ub_offset, iters)
                    self._reorder_s7(tp, ub_input, ub_offset, True, True)
                    self._copy_out_tail_col_tail_row_pad(tp, ub_input, ub_offset, iters)
                else:
                    self._copy_in_tail_col_tail_row(tp, ub_input, ub_offset, ln, tp.loop_on_mc, tp.loop_on_mr)
                    self._reorder_s7(tp, ub_input, ub_offset, True, True)
                    self._copy_out_tail_col_tail_row(tp, ub_input, ub_offset, ln, tp.loop_on_mc, tp.loop_on_mr)

    def _move_data_s7_university(self, tp, ub_input_64):
        ub_offset = self.tik_inst.Scalar("int32")  # unit : block
        ub_input = ub_input_64.reinterpret_cast_to(self.x_dtype)

        self._init_n_tuple(tp)

        with self.tik_inst.for_range(0, tp.loop_on_n) as ln:
            self._init_dst_tuple(tp)
            self._get_n_offset_s7(tp)

            self._major_col_process_s7(tp, ub_input, ub_offset, ln)
            self._major_row_process_s7(tp, ub_input, ub_offset, ln)

            self._update_tuple(tp.n_axis_num, tp.rt_n_tuple, tp.n_jump_factor)

    def _gen_src_offset_process_s7(self, tp, ub_input, reorder_idx, conv_params):
        """
        process for generating src_offset for scenario 7
        """
        concur_cnt = 8
        tik_inst = self.tik_inst
        idx = tik_inst.Scalar("int32", init_value=0)
        tmp_off = tik_inst.Scalar("int32")
        ub_input_int32 = ub_input.reinterpret_cast_to("int32")
        major_base_addr = tp.src_offset_base_addr
        tail_base_addr = major_base_addr + tp.src_offset_addr_gap
        base_addr_tuple = (major_base_addr, tail_base_addr)

        def _inner_process(base_addr, params):
            tgt_axis_0, tgt_axis_1 = params[0]
            tgt_axis_0_off, tgt_axis_1_off = params[1]

            with tik_inst.for_range(0, tgt_axis_0) as axis_0_idx:
                lp_cnt = tgt_axis_1 // concur_cnt
                with tik_inst.for_range(0, lp_cnt) as axis_1_lp_idx:
                    tmp_off.set_as((axis_1_lp_idx * concur_cnt * tgt_axis_1_off +
                                    axis_0_idx * tgt_axis_0_off) * self.b8_times)
                    for i in range(concur_cnt):
                        ub_input_int32[base_addr + idx + i].set_as(tmp_off + i * tgt_axis_1_off * self.b8_times)
                    idx.set_as(idx + concur_cnt)
                lp_left = tgt_axis_1 % concur_cnt
                with tik_inst.if_scope(lp_left > 0):
                    tmp_off.set_as((lp_cnt * concur_cnt * tgt_axis_1_off +
                                    axis_0_idx * tgt_axis_0_off) * self.b8_times)
                    with tik_inst.for_range(0, lp_left) as left_idx:
                        ub_input_int32[base_addr + idx].set_as(tmp_off + left_idx * tgt_axis_1_off * self.b8_times)
                        idx.set_as(idx + 1)
        _inner_process(base_addr_tuple[reorder_idx], conv_params)

    def _gen_vgather_src_offset_4_s7_fat_2_thin(self, tp, ub_input):
        """
        multiple in single out
        """
        major_row = tp.row_per_mr
        major_col = tp.col_per_mc
        row_offset = major_col
        col_offset = 1
        conv_params = [[major_col, major_row], [col_offset, row_offset]]
        self._gen_src_offset_process_s7(tp, ub_input, 0, conv_params)
        with self.tik_inst.if_scope(tp.col_tc > 0):
            tail_col = tp.col_tc
            row_offset = tail_col
            conv_params = [[tail_col, major_row], [col_offset, row_offset]]
            self._gen_src_offset_process_s7(tp, ub_input, 1, conv_params)

    def _move_data_s7_fat_2_thin(self, tp, ub_input_64):
        ub_offset = self.tik_inst.Scalar("int32")  # unit : block
        ub_input = ub_input_64.reinterpret_cast_to(self.x_dtype)
        self._init_n_tuple(tp)
        if self.vg_cond is True:
            with self.tik_inst.if_scope(tp.scenario == 15):
                tp.src_offset_base_addr = tp.col_per_mc * tp.row_per_mr
                tp.src_offset_addr_gap = tp.col_per_mc * tp.row_per_mr
                self._gen_vgather_src_offset_4_s7_fat_2_thin(tp, ub_input)

        with self.tik_inst.for_range(0, tp.loop_on_n) as ln:
            self._init_dst_tuple(tp)
            self._get_n_offset_s7(tp)
            self._init_dst_addr_s7(tp, ln)
            with self.tik_inst.for_range(0, tp.loop_on_mc) as lc:
                if self.is_mte_pad and self.b8_times != EPB8:
                    self._copy_in_major_col_f2t_pad(tp, ub_input, ub_offset, ln, lc)
                else:
                    self._copy_in_major_col_f2t(tp, ub_input, ub_offset, ln, lc)
                if self.vg_cond is True:
                    with self.tik_inst.if_scope(tp.scenario == 15):
                        params = (ub_input, tp.offset_b // self.fp16_times, tp.src_offset_addr_gap,
                                  0, tp.col_per_mc * tp.row_per_mr)
                        self._reorder_by_vgather_s5_s7(tp, params)
                    with self.tik_inst.else_scope():
                        self._reorder_university_f2t(tp, ub_input, ub_offset, tp.col_per_mc, tp.row_per_mr, 0)
                else:
                    self._reorder_university_f2t(tp, ub_input, ub_offset, tp.col_per_mc, tp.row_per_mr, 0)
                self._copy_out_major_col_f2t(tp, ub_input, ub_offset, lc)

            with self.tik_inst.if_scope(tp.col_tc != 0):
                if self.is_mte_pad and self.b8_times != EPB8:
                    self._copy_in_tail_col_f2t_pad(tp, ub_input, ub_offset, ln, tp.loop_on_mc)
                else:
                    self._copy_in_tail_col_f2t(tp, ub_input, ub_offset, ln, tp.loop_on_mc)
                if self.vg_cond is True:
                    with self.tik_inst.if_scope(tp.scenario == 15):
                        params = (ub_input, tp.offset_b // self.fp16_times, tp.src_offset_addr_gap,
                                  1, tp.col_tc * tp.row_per_mr)
                        self._reorder_by_vgather_s5_s7(tp, params)
                    with self.tik_inst.else_scope():
                        self._reorder_university_f2t(tp, ub_input, ub_offset, tp.col_tc, tp.row_per_mr, 0)
                else:
                    self._reorder_university_f2t(tp, ub_input, ub_offset, tp.col_tc, tp.row_per_mr, 0)
                self._copy_out_tail_col_f2t(tp, ub_input, ub_offset, ln, tp.loop_on_mc)
            self._update_tuple(tp.n_axis_num, tp.rt_n_tuple, tp.n_jump_factor)

    def _gen_vgather_src_offset_4_s7_thin_2_fat(self, tp, ub_input):
        """
        single in multiple out
        """
        major_row = tp.row_per_mr
        major_col = tp.col_per_mc
        row_offset = major_col
        col_offset = 1
        conv_params = [[major_col, major_row], [col_offset, row_offset]]
        self._gen_src_offset_process_s7(tp, ub_input, 0, conv_params)
        with self.tik_inst.if_scope(tp.row_tr > 0):
            tail_row = tp.row_tr
            conv_params = [[major_col, tail_row], [col_offset, row_offset]]
            self._gen_src_offset_process_s7(tp, ub_input, 1, conv_params)

    def _move_data_s7_thin_2_fat(self, tp, ub_input_64):
        ub_offset = self.tik_inst.Scalar("int32")  # unit : block
        ub_input = ub_input_64.reinterpret_cast_to(self.x_dtype)
        self._init_n_tuple(tp)
        if self.vg_cond is True:
            with self.tik_inst.if_scope(tp.scenario == 15):
                tp.src_offset_base_addr = tp.col_per_mc * tp.row_per_mr
                tp.src_offset_addr_gap = tp.col_per_mc * tp.row_per_mr
                self._gen_vgather_src_offset_4_s7_thin_2_fat(tp, ub_input)

        with self.tik_inst.for_range(0, tp.loop_on_n) as ln:
            self._init_src_tuple(tp)
            self._get_n_offset_s7(tp)
            with self.tik_inst.for_range(0, tp.loop_on_mr) as lr:
                self._copy_in_major_row_t2f(tp, ub_input, ub_offset, ln, lr)
                if self.vg_cond is True:
                    with self.tik_inst.if_scope(tp.scenario == 15):
                        params = (ub_input, tp.offset_b // self.fp16_times, tp.src_offset_addr_gap,
                                  0, tp.col_per_mc * tp.row_per_mr)
                        self._reorder_by_vgather_s5_s7(tp, params)
                    with self.tik_inst.else_scope():
                        self._reorder_university_f2t(tp, ub_input, ub_offset, tp.col_per_mc, tp.row_per_mr, 1)
                else:
                    self._reorder_university_f2t(tp, ub_input, ub_offset, tp.col_per_mc, tp.row_per_mr, 1)
                if self.is_mte_pad and self.b8_times != EPB8:
                    self._copy_out_major_row_t2f_pad(tp, ub_input, ub_offset, ln, lr)
                else:
                    self._copy_out_major_row_t2f(tp, ub_input, ub_offset, ln, lr)
            with self.tik_inst.if_scope(tp.row_tr != 0):
                self._copy_in_tail_row_t2f(tp, ub_input, ub_offset, ln, tp.loop_on_mr)
                if self.vg_cond is True:
                    with self.tik_inst.if_scope(tp.scenario == 15):
                        params = (ub_input, tp.offset_b // self.fp16_times, tp.src_offset_addr_gap,
                                  1, tp.col_per_mc * tp.row_tr)
                        self._reorder_by_vgather_s5_s7(tp, params)
                    with self.tik_inst.else_scope():
                        self._reorder_university_f2t(tp, ub_input, ub_offset, tp.col_per_mc, tp.row_tr, 1)
                else:
                    self._reorder_university_f2t(tp, ub_input, ub_offset, tp.col_per_mc, tp.row_tr, 1)
                if self.is_mte_pad and self.b8_times != EPB8:
                    self._copy_out_tail_row_t2f_pad(tp, ub_input, ub_offset, ln, tp.loop_on_mr)
                else:
                    self._copy_out_tail_row_t2f(tp, ub_input, ub_offset, ln, tp.loop_on_mr)
            self._update_tuple(tp.n_axis_num, tp.rt_n_tuple, tp.n_jump_factor)

    def _move_data_s7(self, tp, sub_scenario):
        with self.tik_inst.if_scope(sub_scenario == 0):
            self._move_data_s7_university(tp, self.ub_input_64)
        with self.tik_inst.else_scope():
            with self.tik_inst.if_scope(sub_scenario == 1):
                self._move_data_s7_fat_2_thin(tp, self.ub_input_64)
            with self.tik_inst.else_scope():
                self._move_data_s7_thin_2_fat(tp, self.ub_input_64)

    # -------------------------------------------------------------------------------------------------
    #                                    scenario_8
    # -------------------------------------------------------------------------------------------------
    def _move_data_s8(self, ub_input_64):
        if self.x_dtype in ("bool", "uint8", "int8", "int64", "uint64"):
            return
        # 4 255 3 8 -> 3 255 4 8
        if api_check_support("tik.vcopy", "int16"):
            ub_input = ub_input_64.reinterpret_cast_to(self.x_dtype)
            tik_inst = self.tik_inst
            tik_inst.data_move(ub_input, self.data_in, 0, 1, 4 * 255 * 3, 0, 0)
            with tik_inst.for_range(0, 3) as i:
                tik_inst.vcopy(4 * 8,
                               ub_input[32 * 1024 + i * 255 * 4 * 8],
                               ub_input[0 + i * 8],
                               255,
                               1,
                               255 * 3,
                               4,
                               3,
                               "counter")
            tik_inst.data_move(self.data_out, ub_input[32 * 1024], 0, 1, 4 * 255 * 3, 0, 0)

    # -------------------------------------------------------------------------------------------------
    #                                    scenario_9
    # -------------------------------------------------------------------------------------------------
    def _get_src_addr_s9(self, tp):
        tp.src_addr.set_as(0)
        with self.tik_inst.for_range(0, tp.trans_axis_num) as i:
            tp.src_addr.set_as(tp.src_addr + tp.rt_tuple[i] * tp.src_jump_stride[i])

    def _get_dst_addr_s9(self, tp):
        tp.dst_addr.set_as(0)
        with self.tik_inst.for_range(0, tp.trans_axis_num) as i:
            tp.dst_addr.set_as(tp.dst_addr + tp.rt_tuple[i] * tp.dst_jump_stride[i])

    def _copy_in_src_mode_s9(self, tp, ub_input, repeat, burst_len, ub_offset):
        self._get_src_addr_s1(tp)
        self.tik_inst.data_move(ub_input, self.data_in[tp.src_addr], 0, 1, repeat * burst_len, 0, 0)

    def _copy_out_src_mode_s9(self, tp, ub_input, repeat, burst_len, dst_stride, ub_offset):
        self._get_dst_addr_s1(tp)
        self.tik_inst.data_move(self.data_out[tp.dst_addr], ub_input, 0, repeat, burst_len, 0, dst_stride)

    def _copy_in_dst_mode_s9(self, tp, ub_input, repeat, burst_len, src_stride, ub_offset):
        self._get_src_addr_s1(tp)
        self.tik_inst.data_move(ub_input, self.data_in[tp.src_addr], 0, repeat, burst_len, src_stride, 0)

    def _copy_out_dst_mode_s9(self, tp, ub_input, repeat, burst_len, ub_offset):
        self._get_dst_addr_s1(tp)
        self.tik_inst.data_move(self.data_out[tp.dst_addr], ub_input, 0, 1, repeat * burst_len, 0, 0)

    def _move_data_s9(self, tp, ub_input_64, sub_scenario):
        ub_offset = self.tik_inst.Scalar("int32")  # unit : block
        ub_input = ub_input_64.reinterpret_cast_to(self.x_dtype)

        self._init_tuple_common(tp)
        with self.tik_inst.if_scope(sub_scenario == 1):  # dst mode
            with self.tik_inst.for_range(0, tp.loop_num) as ln:
                self._copy_in_dst_mode_s9(tp, ub_input, tp.repeat, tp.last_axis_burst_len,
                                          tp.src_repeat_stride, ub_offset)
                self._copy_out_dst_mode_s9(tp, ub_input, tp.repeat, tp.last_axis_burst_len, ub_offset)
                self._update_tuple(tp.trans_axis_num, tp.rt_tuple, tp.dst_jump_factor)
        with self.tik_inst.else_scope():  # src mode
            with self.tik_inst.for_range(0, tp.loop_num) as ln:
                self._copy_in_src_mode_s9(tp, ub_input, tp.repeat, tp.last_axis_burst_len, ub_offset)
                self._copy_out_src_mode_s9(tp, ub_input, tp.repeat, tp.last_axis_burst_len,
                                           tp.dst_repeat_stride, ub_offset)
                self._update_tuple(tp.trans_axis_num, tp.rt_tuple, tp.src_jump_factor)

    # -------------------------------------------------------------------------------------------------
    #                                    scenario_10
    # -------------------------------------------------------------------------------------------------
    def _reorder_s10_last_two_axis_16x16_b16(self, tp, ub_input, idx):
        ub_input_fp16 = ub_input.reinterpret_cast_to("float16")
        repeat_cnt = tp.xdxsVol[idx]
        src_addr_list = [ub_input_fp16[EPB16 * i] for i in range(EPB16)]
        dst_addr_list = [ub_input_fp16[tp.offset_b + EPB16 * i] for i in range(EPB16)]

        with self.tik_inst.if_scope(repeat_cnt == 1):
            tp.src_stride_reorder.set_as(0)
            tp.dst_stride_reorder.set_as(0)
        with self.tik_inst.else_scope():
            tp.src_stride_reorder.set_as(EPB16)
            tp.dst_stride_reorder.set_as(EPB16)
        self.tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt,
                                tp.dst_stride_reorder, tp.src_stride_reorder)

    def _reorder_s10_last_two_axis_b16(self, tp, ub_input, idx):
        ub_input_fp16 = ub_input.reinterpret_cast_to("float16")

        with self.tik_inst.if_scope(tik.all(tp.last_axis_len == EPB16, tp.second_to_last_axis_len == EPB16)):
            self._reorder_s10_last_two_axis_16x16_b16(tp, ub_input, idx)

        with self.tik_inst.else_scope():
            with self.tik_inst.for_range(0, tp.xdxsVol[idx]) as i:
                with self.tik_inst.for_range(0, tp.last_two_loop) as j:
                    repeat_cnt = tp.last_two_repeat
                    src_addr_list = [ub_input_fp16[i * tp.last_axis_len * tp.second_to_last_axis_len + \
                                                   j * tp.last_two_s_list_repeat + \
                                                   tp.last_axis_len * k] for k in range(EPB16)]
                    dst_addr_list = [ub_input_fp16[tp.offset_b + i * tp.last_axis_len * tp.second_to_last_axis_len +\
                                                   j * tp.last_two_d_list_repeat + \
                                                   tp.second_to_last_axis_len * k] for k in range(EPB16)]

                    tp.src_stride_reorder.set_as(tp.last_two_s_stride)
                    tp.dst_stride_reorder.set_as(tp.last_two_d_stride)
                    self.tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt,
                                            tp.dst_stride_reorder, tp.src_stride_reorder)
        self._swap(tp)

    def _reorder_s10_last_two_axis_b32(self, tp, ub_input, idx):

        col_reorder = tp.last_axis_len
        # `repeat_cnt = tp.last_axis_len * tp.second_to_last_axis_len // EPB8`
        repeat_cnt = 80

        # do hwc to chw transfer
        inner_hw_len = EPB8
        fp16_inner_hwc_len = EPB16 * col_reorder
        offset_unit = tp.last_axis_len * tp.second_to_last_axis_len * self.fp16_times
        ub_input_fp16 = ub_input.reinterpret_cast_to("float16")

        with self.tik_inst.for_range(0, tp.xdxsVol[idx]) as i:

            with self.tik_inst.if_scope(repeat_cnt == 1):
                tp.src_stride_reorder.set_as(0)
                tp.dst_stride_reorder.set_as(0)
            with self.tik_inst.else_scope():
                tp.src_stride_reorder.set_as(1)
                tp.dst_stride_reorder.set_as(EPB16)

            # first vnchwconv
            src_addr_list = [ub_input_fp16[i * offset_unit + fp16_inner_hwc_len * k] for k in range(EPB16)]
            dst_addr_list = [ub_input_fp16[i * offset_unit + tp.offset_b * self.fp16_times + EPB16 * k] \
                             for k in range(EPB16)]
            self.tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt,
                                    tp.dst_stride_reorder, tp.src_stride_reorder)

            # do hwc to chw transfer
            with self.tik_inst.new_stmt_scope(disable_sync=True):
                with self.tik_inst.for_range(0, inner_hw_len) as j:
                    self.tik_inst.data_move(ub_input_fp16[i * offset_unit + j * EPB32],
                                            ub_input_fp16[i * offset_unit + tp.offset_b * self.fp16_times + \
                                                          j * col_reorder * EPB32],
                                            0, col_reorder, self.fp16_times, 0, (inner_hw_len - 1) * self.fp16_times)

            # second vnchwconv
            src_addr_list = [ub_input_fp16[i * offset_unit + EPB16 * k] for k in range(EPB16)]
            dst_addr_list = [ub_input_fp16[i * offset_unit + tp.offset_b * self.fp16_times + EPB16 * k] \
                            for k in range(EPB16)]

            with self.tik_inst.if_scope(repeat_cnt == 1):
                tp.src_stride_reorder.set_as(0)
                tp.dst_stride_reorder.set_as(0)
            with self.tik_inst.else_scope():
                tp.src_stride_reorder.set_as(EPB16)
                tp.dst_stride_reorder.set_as(EPB16)

            self.tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt,
                                    tp.dst_stride_reorder, tp.src_stride_reorder)
        self._swap(tp)

    def _reorder_s10(self, tp, ub_input, is_src_tail_in, is_dst_tail_in,
                    x_in_ele, x_in_tail_ele, burst_len_in, x_dst_loop_in,
                    x_out_ele, x_out_tail_ele, burst_len_out, x_src_loop_out):
        tik_inst = self.tik_inst
        idx = tik_inst.Scalar("int32")
        tp.offset_a.set_as(tp.offset_1 // self.fp16_times)
        tp.offset_b.set_as(tp.offset_2 // self.fp16_times)
        self._get_reorder_idx(is_src_tail_in, is_dst_tail_in, idx)
        if self.fp16_times == 1:
            self._reorder_s10_last_two_axis_b16(tp, ub_input, idx)
        elif self.fp16_times == 2:
            self._reorder_s10_last_two_axis_b32(tp, ub_input, idx)
        self._reorder_s5_data_move(tp, ub_input, idx)

    def _major_process_s10(self, tp, ub_input):
        self._copy_in_major_src_major_dst_s5(tp, ub_input)
        self._reorder_s10(tp, ub_input, 0, 0,
                             tp.major_in_ele,
                             tp.major_in_tail_ele,
                             tp.major_burst_len_in,
                             tp.major_dst_loop_in,
                             tp.major_out_ele,
                             tp.major_out_tail_ele,
                             tp.major_burst_len_out,
                             tp.major_src_loop_out)
        self._copy_out_major_src_major_dst_s5(tp, ub_input)

    def _src_tail_in_process_s10(self, tp, ub_input):
        tik_inst = self.tik_inst
        with tik_inst.if_scope(tp.tail_burst_len_in != 0):
            self._init_tail_src_tuple_copy_out_s5(tp)
            self._init_major_dst_tuple_copy_in_s5(tp)
            self._copy_in_tail_src_major_dst_s5(tp, ub_input)
            self._reorder_s10(tp, ub_input, 1, 0,
                                     tp.tail_in_ele,
                                     tp.tail_in_tail_ele,
                                     tp.tail_burst_len_in,
                                     tp.major_dst_loop_in,
                                     tp.major_out_ele,
                                     tp.major_out_tail_ele,
                                     tp.major_burst_len_out,
                                     tp.major_src_loop_out)
            self._copy_out_tail_src_major_dst_s5(tp, ub_input)

    def _dst_tail_in_process_s10(self, tp, ub_input):
        tik_inst = self.tik_inst
        with tik_inst.if_scope(tp.tail_burst_len_out != 0):
            self._init_major_src_tuple_copy_out_s5(tp)
            self._init_tail_dst_tuple_copy_in_s5(tp)
            self._copy_in_major_src_tail_dst_s5(tp, ub_input)
            self._reorder_s10(tp, ub_input, 0, 1,
                                     tp.major_in_ele,
                                     tp.major_in_tail_ele,
                                     tp.major_burst_len_in,
                                     tp.tail_dst_loop_in,
                                     tp.tail_out_ele,
                                     tp.tail_out_tail_ele,
                                     tp.tail_burst_len_out,
                                     tp.major_src_loop_out)
            self._copy_out_major_src_tail_dst_s5(tp, ub_input)

    def _both_tail_in_process_s10(self, tp, ub_input):
        tik_inst = self.tik_inst
        with tik_inst.if_scope(tik.all(tp.tail_burst_len_in != 0, tp.tail_burst_len_out != 0)):
            self._init_tail_src_tuple_copy_out_s5(tp)
            self._init_tail_dst_tuple_copy_in_s5(tp)
            self._copy_in_tail_src_tail_dst_s5(tp, ub_input)
            self._reorder_s10(tp, ub_input, 1, 1,
                                     tp.tail_in_ele,
                                     tp.tail_in_tail_ele,
                                     tp.tail_burst_len_in,
                                     tp.tail_dst_loop_in,
                                     tp.tail_out_ele,
                                     tp.tail_out_tail_ele,
                                     tp.tail_burst_len_out,
                                     tp.tail_src_loop_out)
            self._copy_out_tail_src_tail_dst_s5(tp, ub_input)

    def _move_data_s10(self, tp, ub_input_64):
        tik_inst = self.tik_inst
        is_src_tail_in = self.tik_inst.Scalar("int32")
        is_dst_tail_in = self.tik_inst.Scalar("int32")
        ub_input = ub_input_64.reinterpret_cast_to(self.x_dtype)
        self._init_all_tuple_s5(tp)

        with tik_inst.for_range(0, tp.loop_per_core):
            is_src_tail_in.set_as(0)
            is_dst_tail_in.set_as(0)

            self._init_major_src_tuple_copy_out_s5(tp)
            self._init_major_dst_tuple_copy_in_s5(tp)
            self._major_process_s10(tp, ub_input)
            
            self._detect_tail_flag(tp, is_src_tail_in, is_dst_tail_in)

            with tik_inst.if_scope(tik.all(is_src_tail_in == 1, tp.pivot_src_axis_dup == 0)):
                self._src_tail_in_process_s10(tp, ub_input)

            with tik_inst.if_scope(tik.all(is_dst_tail_in == 1, tp.pivot_dst_axis_dup == 0)):
                self._dst_tail_in_process_s10(tp, ub_input)

            with tik_inst.if_scope(tik.all(is_dst_tail_in == 1, is_src_tail_in == 1)):
                self._both_tail_in_process_s10(tp, ub_input)

            self._update_logic_tuple_s5(tp)

    # -------------------------------------------------------------------------------------------------
    #                                    scenario_11
    # -------------------------------------------------------------------------------------------------
    @staticmethod
    def _get_src_addr_s11(tp, ln, lc, lr):
        tp.src_addr.set_as(tp.col_offset + lc * tp.col_per_mc + tp.row_offset * tp.col_vol \
                           + lr * tp.row_per_mr * tp.col_vol)
        return tp.src_addr

    @staticmethod
    def _get_dst_addr_s11(tp, ln, lc, lr):
        tp.dst_addr.set_as(tp.row_offset + lr * tp.row_per_mr + tp.col_offset * tp.row_vol \
                           + lc * tp.col_per_mc * tp.row_vol)
        return tp.dst_addr

    def _copy_in_s11_mcmr(self, tp, ub_input, ub_offset, ln, lc, lr):
        self.tik_inst.data_move(ub_input[0],
                                self.data_in[self._get_src_addr_s11(tp, ln, lc, lr)],
                                0,
                                tp.row_per_mr,
                                tp.col_block_per_mc,
                                tp.src_stride_in,
                                0)
        ub_offset.set_as(tp.row_per_mr * tp.col_per_mc)

    def _copy_in_s11_mctr(self, tp, ub_input, ub_offset, ln, lc, lr):
        self.tik_inst.data_move(ub_input[0],
                                self.data_in[self._get_src_addr_s11(tp, ln, lc, lr)],
                                0,
                                tp.row_tr,
                                tp.col_block_per_mc,
                                tp.src_stride_in,
                                0)
        ub_offset.set_as(tp.row_tr * tp.col_per_mc)

    def _copy_in_s11_tcmr(self, tp, ub_input, ub_offset, ln, lc, lr):
        self.tik_inst.data_move(ub_input[0],
                                self.data_in[self._get_src_addr_s11(tp, ln, lc, lr)],
                                0,
                                tp.row_per_mr,
                                tp.col_block_tc,
                                tp.src_stride_in_tail,
                                0)
        ub_offset.set_as(tp.row_per_mr * tp.col_tc)

    def _copy_in_s11_tctr(self, tp, ub_input, ub_offset, ln, lc, lr):
        self.tik_inst.data_move(ub_input[0],
                                self.data_in[self._get_src_addr_s11(tp, ln, lc, lr)],
                                0,
                                tp.row_tr,
                                tp.col_block_tc,
                                tp.src_stride_in_tail,
                                0)
        ub_offset.set_as(tp.row_tr * tp.col_tc)

    def _copy_out_s11_mcmr(self, tp, ub_input, ub_offset, ln, lc, lr):
        with self.tik_inst.if_scope(self.ele_per_block == EPB32):
            self.tik_inst.data_move(self.data_out[self._get_dst_addr_s11(tp, ln, lc, lr)],
                                    ub_input[tp.offset_b * 2],
                                    0,
                                    tp.col_per_mc,
                                    tp.row_block_per_mr,
                                    ROW_UNIT // EPB32 - tp.row_block_per_mr,
                                    tp.dst_stride_out)
        with self.tik_inst.else_scope():
            self.tik_inst.data_move(self.data_out[self._get_dst_addr_s11(tp, ln, lc, lr)],
                                    ub_input[tp.offset_b // self.fp16_times],
                                    0,
                                    tp.col_per_mc,
                                    tp.row_block_per_mr,
                                    tp.dst_vnc_pad // (EPB16 // self.fp16_times) - tp.row_block_per_mr,
                                    tp.dst_stride_out)

    def _copy_out_s11_mctr(self, tp, ub_input, ub_offset, ln, lc, lr):
        with self.tik_inst.if_scope(self.ele_per_block == EPB32):
            self.tik_inst.data_move(self.data_out[self._get_dst_addr_s11(tp, ln, lc, lr)],
                                    ub_input[tp.offset_b * 2],
                                    0,
                                    tp.col_per_mc,
                                    tp.row_block_tr,
                                    ROW_UNIT // EPB32 - tp.row_block_tr,
                                    tp.dst_stride_out_tail)
        with self.tik_inst.else_scope():
            self.tik_inst.data_move(self.data_out[self._get_dst_addr_s11(tp, ln, lc, lr)],
                                    ub_input[tp.offset_b // self.fp16_times],
                                    0,
                                    tp.col_per_mc,
                                    tp.row_block_tr,
                                    tp.dst_vnc_pad // (EPB16 // self.fp16_times) - tp.row_block_tr,
                                    tp.dst_stride_out_tail)

    def _copy_out_s11_tcmr(self, tp, ub_input, ub_offset, ln, lc, lr):
        with self.tik_inst.if_scope(self.ele_per_block == EPB32):
            self.tik_inst.data_move(self.data_out[self._get_dst_addr_s11(tp, ln, lc, lr)],
                                    ub_input[tp.offset_b * 2],
                                    0,
                                    tp.col_tc,
                                    tp.row_block_per_mr,
                                    ROW_UNIT // EPB32 - tp.row_block_per_mr,
                                    tp.dst_stride_out)
        with self.tik_inst.else_scope():
            self.tik_inst.data_move(self.data_out[self._get_dst_addr_s11(tp, ln, lc, lr)],
                                    ub_input[tp.offset_b // self.fp16_times],
                                    0,
                                    tp.col_tc,
                                    tp.row_block_per_mr,
                                    tp.dst_vnc_pad // (EPB16 // self.fp16_times) - tp.row_block_per_mr,
                                    tp.dst_stride_out)

    def _copy_out_s11_tctr(self, tp, ub_input, ub_offset, ln, lc, lr):
        with self.tik_inst.if_scope(self.ele_per_block == EPB32):
            self.tik_inst.data_move(self.data_out[self._get_dst_addr_s11(tp, ln, lc, lr)],
                                    ub_input[tp.offset_b * 2],
                                    0,
                                    tp.col_tc,
                                    tp.row_block_tr,
                                    ROW_UNIT // EPB32 - tp.row_block_tr,
                                    tp.dst_stride_out_tail)
        with self.tik_inst.else_scope():
            self.tik_inst.data_move(self.data_out[self._get_dst_addr_s11(tp, ln, lc, lr)],
                                    ub_input[tp.offset_b // self.fp16_times],
                                    0,
                                    tp.col_tc,
                                    tp.row_block_tr,
                                    tp.dst_vnc_pad // (EPB16 // self.fp16_times) - tp.row_block_tr,
                                    tp.dst_stride_out_tail)

    def _reorder_s11(self, tp, ub_input, ub_offset, is_tc=False, is_tr=False):
        with self.tik_inst.if_scope(self.fp16_times == 2):  # b32
            self._reorder_s7_b32(tp, ub_input, ub_offset, is_tc, is_tr)
        with self.tik_inst.else_scope():
            with self.tik_inst.if_scope(self.ele_per_block == EPB16):  # b16
                self._reorder_s7_b16(tp, ub_input, ub_offset, is_tc, is_tr, True)
            with self.tik_inst.else_scope():  # b8
                self._reorder_s7_b8(tp, ub_input, ub_offset, is_tc, is_tr)

    def _move_data_s11(self, tp, ub_input_64):
        ub_offset = self.tik_inst.Scalar("int32")  # unit : block
        ub_input = ub_input_64.reinterpret_cast_to(self.x_dtype)

        self._init_n_tuple(tp)

        with self.tik_inst.for_range(0, tp.loop_on_n) as ln:
            with self.tik_inst.for_range(0, tp.loop_on_mc) as lc:
                with self.tik_inst.for_range(0, tp.loop_on_mr) as lr:
                    self._copy_in_s11_mcmr(tp, ub_input, ub_offset, ln, lc, lr)
                    self._reorder_s11(tp, ub_input, ub_offset, False, False)
                    self._copy_out_s11_mcmr(tp, ub_input, ub_offset, ln, lc, lr)

                with self.tik_inst.if_scope(tp.row_tr != 0):
                    self._copy_in_s11_mctr(tp, ub_input, ub_offset, ln, lc, tp.loop_on_mr)
                    self._reorder_s11(tp, ub_input, ub_offset, False, True)
                    self._copy_out_s11_mctr(tp, ub_input, ub_offset, ln, lc, tp.loop_on_mr)

            with self.tik_inst.if_scope(tp.col_tc != 0):
                with self.tik_inst.for_range(0, tp.loop_on_mr) as lr:
                    self._copy_in_s11_tcmr(tp, ub_input, ub_offset, ln, tp.loop_on_mc, lr)
                    self._reorder_s11(tp, ub_input, ub_offset, True, False)
                    self._copy_out_s11_tcmr(tp, ub_input, ub_offset, ln, tp.loop_on_mc, lr)

            with self.tik_inst.if_scope(tp.col_tc != 0):
                with self.tik_inst.if_scope(tp.row_tr != 0):
                    self._copy_in_s11_tctr(tp, ub_input, ub_offset, ln, tp.loop_on_mc, tp.loop_on_mr)
                    self._reorder_s11(tp, ub_input, ub_offset, True, True)
                    self._copy_out_s11_tctr(tp, ub_input, ub_offset, ln, tp.loop_on_mc, tp.loop_on_mr)
            self._update_tuple(tp.n_axis_num, tp.rt_n_tuple, tp.n_jump_factor)

    # -------------------------------------------------------------------------------------------------
    #                                    scenario_12
    # -------------------------------------------------------------------------------------------------
    def _copy_in_src_mode_s12(self, tp, ub_input, repeat, burst_len, ub_offset):
        self._get_src_addr_s1(tp)
        self.tik_inst.data_move(ub_input, self.data_in[tp.src_addr], 0, 1, repeat * burst_len, 0, 0)

    def _copy_out_src_mode_s12(self, tp, ub_input, repeat, burst_len, dst_stride, ub_offset):
        self._get_dst_addr_s1(tp)
        self.tik_inst.data_move(self.data_out[tp.dst_addr], ub_input, 0, repeat, burst_len, 0, dst_stride)

    def _copy_in_src_mode_tail_s12(self, tp, ub_input, repeat, burst_len, ub_offset):
        self._get_src_addr_s1(tp)
        tp.src_addr.set_as(tp.src_addr + tp.loop_src_stride)
        self.tik_inst.data_move(ub_input, self.data_in[tp.src_addr], 0, 1, repeat * burst_len, 0, 0)

    def _copy_out_src_mode_tail_s12(self, tp, ub_input, repeat, burst_len, dst_stride, ub_offset):
        self._get_dst_addr_s1(tp)
        tp.dst_addr.set_as(tp.dst_addr + tp.loop_dst_stride)
        self.tik_inst.data_move(self.data_out[tp.dst_addr], ub_input, 0, repeat, burst_len, 0, dst_stride)

    def _copy_in_dst_mode_s12(self, tp, ub_input, repeat, burst_len, src_stride, ub_offset):
        self._get_src_addr_s1(tp)
        self.tik_inst.data_move(ub_input, self.data_in[tp.src_addr], 0, repeat, burst_len, src_stride, 0)

    def _copy_out_dst_mode_s12(self, tp, ub_input, repeat, burst_len, ub_offset):
        self._get_dst_addr_s1(tp)
        self.tik_inst.data_move(self.data_out[tp.dst_addr], ub_input, 0, 1, repeat * burst_len, 0, 0)

    def _copy_in_dst_mode_tail_s12(self, tp, ub_input, repeat, burst_len, src_stride, ub_offset):
        self._get_src_addr_s1(tp)
        tp.src_addr.set_as(tp.src_addr + tp.loop_src_stride)
        self.tik_inst.data_move(ub_input, self.data_in[tp.src_addr], 0, repeat, burst_len, src_stride, 0)

    def _copy_out_dst_mode_tail_s12(self, tp, ub_input, repeat, burst_len, ub_offset):
        self._get_dst_addr_s1(tp)
        tp.dst_addr.set_as(tp.dst_addr + tp.loop_dst_stride)
        self.tik_inst.data_move(self.data_out[tp.dst_addr], ub_input, 0, 1, repeat * burst_len, 0, 0)

    def _is_scenario_12_tail(self, tp):
        with self.tik_inst.if_scope(tp.trans_axis_num < 4):
            with self.tik_inst.if_scope(tp.trans_axis_num == 1):
                with self.tik_inst.if_scope(tp.rt_tuple[0] ==  (tp.loop_main - 1)):
                    return 1
            with self.tik_inst.if_scope(tp.trans_axis_num == 2):
                with self.tik_inst.if_scope(tp.rt_tuple[1] ==  (tp.loop_main - 1)):
                    return 1
            with self.tik_inst.if_scope(tp.trans_axis_num == 3):
                with self.tik_inst.if_scope(tp.rt_tuple[2] ==  (tp.loop_main - 1)):
                    return 1
        with self.tik_inst.else_scope():
            with self.tik_inst.if_scope(tp.trans_axis_num == 4):
                with self.tik_inst.if_scope(tp.rt_tuple[3] ==  (tp.loop_main - 1)):
                    return 1
            with self.tik_inst.if_scope(tp.trans_axis_num == 5):
                with self.tik_inst.if_scope(tp.rt_tuple[4] ==  (tp.loop_main - 1)):
                    return 1
            with self.tik_inst.if_scope(tp.trans_axis_num == 6):
                with self.tik_inst.if_scope(tp.rt_tuple[5] ==  (tp.loop_main - 1)):
                    return 1
        return 0

    def _move_data_s12(self, tp, ub_input_64, sub_scenario):
        ub_offset = self.tik_inst.Scalar("int32")  # unit : block
        ub_input = ub_input_64.reinterpret_cast_to(self.x_dtype)

        self._init_tuple_common(tp)
        with self.tik_inst.if_scope(sub_scenario == 1):  # dst mode
            with self.tik_inst.for_range(0, tp.loop_num) as ln:
                self._copy_in_dst_mode_s12(tp, ub_input, tp.loop_unit, tp.last_axis_burst_len,
                                           tp.src_repeat_stride, ub_offset)
                self._copy_out_dst_mode_s12(tp, ub_input, tp.loop_unit, tp.last_axis_burst_len, ub_offset)

                with self.tik_inst.if_scope(tp.loop_tail != 0):
                    with self.tik_inst.if_scope(self._is_scenario_12_tail(tp) == 1):
                        self._copy_in_dst_mode_tail_s12(tp, ub_input, tp.loop_tail, tp.last_axis_burst_len,
                                                   tp.src_repeat_stride, ub_offset)
                        self._copy_out_dst_mode_tail_s12(tp, ub_input, tp.loop_tail, tp.last_axis_burst_len, ub_offset)

                self._update_tuple(tp.trans_axis_num, tp.rt_tuple, tp.dst_jump_factor)

        with self.tik_inst.else_scope():  # src mode
            with self.tik_inst.for_range(0, tp.loop_num) as ln:
                self._copy_in_src_mode_s12(tp, ub_input, tp.loop_unit, tp.last_axis_burst_len, ub_offset)
                self._copy_out_src_mode_s12(tp, ub_input, tp.loop_unit, tp.last_axis_burst_len,
                                            tp.dst_repeat_stride, ub_offset)

                with self.tik_inst.if_scope(tp.loop_tail != 0):
                    with self.tik_inst.if_scope(self._is_scenario_12_tail(tp) == 1):
                        self._copy_in_src_mode_tail_s12(tp, ub_input, tp.loop_tail, tp.last_axis_burst_len, ub_offset)
                        self._copy_out_src_mode_tail_s12(tp, ub_input, tp.loop_tail, tp.last_axis_burst_len,
                                                         tp.dst_repeat_stride, ub_offset)

                self._update_tuple(tp.trans_axis_num, tp.rt_tuple, tp.src_jump_factor)

    # -------------------------------------------------------------------------------------------------
    #                                    scenario_13
    # -------------------------------------------------------------------------------------------------
    def _move_data_s13_move_in_not_mte_pad(self, move_in_info):
        tp, ub_input, src_gap, start_offset, nburst, bursts, src_ub_gap = move_in_info
        burst_in_ub = ceil_align(bursts, self.ele_per_block)
        with self.tik_inst.if_scope(tik.all(tp.col_num % self.ele_per_block == 0,
                                            (tp.col_num - burst_in_ub) // self.ele_per_block <= MOVE_MAX_STRIDE)):
            self.tik_inst.data_move(ub_input, self.data_in[start_offset],
                                    0, nburst,
                                    burst_in_ub // self.ele_per_block,
                                    (tp.col_num - burst_in_ub) // self.ele_per_block,
                                    src_ub_gap)
        with self.tik_inst.else_scope():
            with self.tik_inst.new_stmt_scope(disable_sync=True):
                with self.tik_inst.for_range(0, nburst) as loop:
                    self.tik_inst.data_move(ub_input[loop * (burst_in_ub + src_ub_gap * self.ele_per_block)],
                                            self.data_in[start_offset + loop * tp.col_num],
                                            0, 1,
                                            burst_in_ub // self.ele_per_block,
                                            0, 0)

    def _move_data_s13_move_in_mte_pad(self, move_in_info):
        tp, ub_input, src_gap, start_offset, nburst, bursts, src_ub_gap = move_in_info
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.if_scope(src_gap <= MTE_PAD_GAP_LIMIT):
                self.tik_inst.data_move_pad(ub_input,
                                            self.data_in[start_offset],
                                            nburst,
                                            bursts * self.b8_times,
                                            src_ub_gap,
                                            src_gap)
            with self.tik_inst.else_scope():
                burst_in_ub = ceil_align(bursts, self.ele_per_block)
                with self.tik_inst.for_range(0, nburst) as loop:
                    self.tik_inst.data_move_pad(ub_input[loop * (burst_in_ub + src_ub_gap * self.ele_per_block)],
                                                self.data_in[start_offset + loop * tp.col_num],
                                                1,
                                                bursts * self.b8_times,
                                                0,
                                                0)

    def _move_data_s13_move_in(self, move_in_info):
        if self.is_mte_pad:
            self._move_data_s13_move_in_mte_pad(move_in_info)
        else:
            self._move_data_s13_move_in_not_mte_pad(move_in_info)

    def _move_data_s13_move_out_not_mte_pad(self, move_out_info):
        tp, ub_input, dst_gap, start_offset, nburst, bursts, dst_ub_gap = move_out_info
        burst_in_ub = ceil_align(bursts, self.ele_per_block)
        with self.tik_inst.if_scope(tik.all(tp.row_num % self.ele_per_block == 0,
                                            (tp.row_num - burst_in_ub) // self.ele_per_block <= MOVE_MAX_STRIDE)):
            self.tik_inst.data_move(self.data_out[start_offset],
                                    ub_input[self.half_ub_num],
                                    0, nburst,
                                    burst_in_ub // self.ele_per_block,
                                    dst_ub_gap,
                                    (tp.row_num - burst_in_ub) // self.ele_per_block)
        with self.tik_inst.else_scope():
            with self.tik_inst.new_stmt_scope(disable_sync=True):
                with self.tik_inst.for_range(0, nburst) as loop:
                    self.tik_inst.data_move(self.data_out[start_offset + loop * tp.row_num],
                                            ub_input[self.half_ub_num +
                                                    loop * (burst_in_ub + dst_ub_gap * self.ele_per_block)],
                                            0, 1,
                                            burst_in_ub // self.ele_per_block,
                                            0, 0)

    def _move_data_s13_move_out_mte_pad(self, move_out_info):
        tp, ub_input, dst_gap, start_offset, nburst, bursts, dst_ub_gap = move_out_info
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.if_scope(dst_gap <= MTE_PAD_GAP_LIMIT):
                self.tik_inst.data_move_pad(self.data_out[start_offset],
                                            ub_input[self.half_ub_num],
                                            nburst,
                                            bursts * self.b8_times,
                                            dst_gap,
                                            dst_ub_gap)
            with self.tik_inst.else_scope():
                burst_in_ub = ceil_align(bursts, self.ele_per_block)
                with self.tik_inst.for_range(0, nburst) as loop:
                    self.tik_inst.data_move_pad(self.data_out[start_offset + loop * tp.row_num],
                                                ub_input[self.half_ub_num +
                                                        loop * (burst_in_ub + dst_ub_gap * self.ele_per_block)],
                                                1,
                                                bursts * self.b8_times,
                                                0,
                                                0)

    def _move_data_s13_move_out(self, move_out_info):
        if self.is_mte_pad:
            self._move_data_s13_move_out_mte_pad(move_out_info)
        else:
            self._move_data_s13_move_out_not_mte_pad(move_out_info)

    def _move_data_s13_each_loop(self, tp, ub_input, src_gap, src_start_offset,
                                 dst_gap, dst_start_offset, nburst, bursts):
        bursts_in_ub = ceil_align(bursts, self.ele_per_block)
        # to avoid bank conflict
        tp.nburst_in_ub.set_as(ceil_align(nburst, EPB16))
        tp.src_ub_gap.set_as(0)
        tp.dst_ub_gap.set_as(0)
        with self.tik_inst.if_scope(bursts_in_ub // self.ele_per_block % 2 == 0):
            tp.src_ub_gap.set_as(1)
        with self.tik_inst.if_scope(tp.nburst_in_ub // self.ele_per_block % 2 == 0):
            tp.dst_ub_gap.set_as(1)

        move_in_info = (tp, ub_input, src_gap, src_start_offset, nburst, bursts, tp.src_ub_gap)
        self._move_data_s13_move_in(move_in_info)

        bursts_num = bursts_in_ub + tp.src_ub_gap * self.ele_per_block
        repeat_cnt = bursts_num // self.ele_per_block
        nburst_num = tp.nburst_in_ub + tp.dst_ub_gap * self.ele_per_block

        if self.b8_times == EPB2:
            with self.tik_inst.for_range(0, tp.nburst_in_ub // EPB16) as loop:
                src_addr_list = [ub_input[loop * bursts_num * EPB16 + bursts_num * i] for i in range(EPB16)]
                dst_addr_list = [ub_input[self.half_ub_num +
                                          loop * EPB16 + nburst_num * i] for i in range(EPB16)]
                with self.tik_inst.if_scope(repeat_cnt == 1):
                    self.tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, 1, 0, 0)
                with self.tik_inst.else_scope():
                    self.tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, nburst_num, 1)

            # after vnchwconv, nburst changed to bursts, and bursts changed to nburst
            move_out_info = (tp, ub_input, dst_gap, dst_start_offset, bursts, nburst, tp.dst_ub_gap)
            self._move_data_s13_move_out(move_out_info)
        else:
            with self.tik_inst.for_range(0, tp.nburst_in_ub // EPB16) as loop:
                src_addr_list = [ub_input[loop * bursts_num * EPB16 + bursts_num * i] for i in range(EPB16)]
                dst_start_in_ub = self.half_ub_num + loop * EPB16
                dst_addr_list = [ub_input[dst_start_in_ub],
                                 ub_input[dst_start_in_ub + EPB8],
                                 ub_input[dst_start_in_ub + nburst_num],
                                 ub_input[dst_start_in_ub + nburst_num + EPB8],
                                 ub_input[dst_start_in_ub + nburst_num * 2],
                                 ub_input[dst_start_in_ub + nburst_num * 2 + EPB8],
                                 ub_input[dst_start_in_ub + nburst_num * 3],
                                 ub_input[dst_start_in_ub + nburst_num * 3 + EPB8],
                                 ub_input[dst_start_in_ub + nburst_num * 4],
                                 ub_input[dst_start_in_ub + nburst_num * 4 + EPB8],
                                 ub_input[dst_start_in_ub + nburst_num * 5],
                                 ub_input[dst_start_in_ub + nburst_num * 5 + EPB8],
                                 ub_input[dst_start_in_ub + nburst_num * 6],
                                 ub_input[dst_start_in_ub + nburst_num * 6 + EPB8],
                                 ub_input[dst_start_in_ub + nburst_num * 7],
                                 ub_input[dst_start_in_ub + nburst_num * 7 + EPB8]]
                with self.tik_inst.if_scope(repeat_cnt == 1):
                    self.tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, 1, 0, 0)
                with self.tik_inst.else_scope():
                    self.tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, nburst_num, 1)

            # after vnchwconv, nburst changed to bursts, and bursts changed to nburst
            with self.tik_inst.if_scope(tik.all(nburst % EPB16 <= self.ele_per_block, nburst % EPB16 != 0)):
                move_out_info = (tp, ub_input, dst_gap, dst_start_offset, bursts, nburst, tp.dst_ub_gap + 1)
                self._move_data_s13_move_out(move_out_info)
            with self.tik_inst.else_scope():
                move_out_info = (tp, ub_input, dst_gap, dst_start_offset, bursts, nburst, tp.dst_ub_gap)
                self._move_data_s13_move_out(move_out_info)

    def _move_data_s13_by_row_tail_loop_pad(self, tp, n_idx, ub_input, outer, bursts):
        src_start_offset = (tp.data_in_offset + n_idx * tp.col_num * tp.row_num +
                            tp.major_loop_cnt * tp.col_num * tp.loop_unit +
                            outer * tp.inner_loop_unit)
        dst_start_offset = (tp.data_out_offset + n_idx * tp.col_num * tp.row_num +
                            tp.major_loop_cnt * tp.loop_unit +
                            outer * tp.row_num * tp.inner_loop_unit)
        self._move_data_s13_each_loop(tp, ub_input,
                                      (tp.col_num - bursts) * self.b8_times,
                                      src_start_offset,
                                      (tp.row_num - tp.last_loop_unit) * self.b8_times,
                                      dst_start_offset,
                                      tp.last_loop_unit, bursts)

    def _move_data_s13_by_row_tail_loop_no_pad(self, tp, n_idx, ub_input, outer, bursts):
        with self.tik_inst.if_scope(tp.row_num >= self.ele_per_block):
            last_loop_unit_align = floor_align(tp.last_loop_unit, self.ele_per_block)
            with self.tik_inst.if_scope(last_loop_unit_align > 0):
                src_start_offset = (tp.data_in_offset + n_idx * tp.col_num * tp.row_num +
                                    tp.major_loop_cnt * tp.col_num * tp.loop_unit +
                                    outer * tp.inner_loop_unit)
                dst_start_offset = (tp.data_out_offset + n_idx * tp.col_num * tp.row_num +
                                    tp.major_loop_cnt * tp.loop_unit +
                                    outer * tp.row_num * tp.inner_loop_unit)
                self._move_data_s13_each_loop(tp, ub_input,
                                              (tp.col_num - bursts) * self.b8_times,
                                              src_start_offset,
                                              (tp.row_num - last_loop_unit_align) * self.b8_times,
                                              dst_start_offset,
                                              last_loop_unit_align, bursts)
            # step back copy in and copy out
            with self.tik_inst.if_scope(tp.last_loop_unit % self.ele_per_block != 0):
                tp.back_step.set_as(self.ele_per_block - tp.last_loop_unit % self.ele_per_block)
                src_start_offset = (tp.data_in_offset + n_idx * tp.col_num * tp.row_num +
                                    tp.major_loop_cnt * tp.col_num * tp.loop_unit +
                                    outer * tp.inner_loop_unit +
                                    last_loop_unit_align * tp.col_num - tp.back_step * tp.col_num)
                dst_start_offset = (tp.data_out_offset + n_idx * tp.col_num * tp.row_num +
                                    tp.major_loop_cnt * tp.loop_unit +
                                    outer * tp.row_num * tp.inner_loop_unit +
                                    last_loop_unit_align - tp.back_step)
                self._move_data_s13_each_loop(tp, ub_input,
                                              (tp.col_num - bursts) * self.b8_times,
                                              src_start_offset,
                                              (tp.row_num - self.ele_per_block) * self.b8_times,
                                              dst_start_offset,
                                              self.ele_per_block, bursts)

    def _move_data_s13_by_row_loop(self, tp, n_idx, ub_input, outer, bursts):
        # this if_scope is neccesary
        # compile failed when major_loop_cnt is 0 but tp.row_num < tp.loop_unit, because pass has bugs
        with self.tik_inst.if_scope(tik.all(tp.col_num >= bursts, tp.row_num >= tp.loop_unit)):
            with self.tik_inst.for_range(0, tp.major_loop_cnt) as inner:
                src_start_offset = (tp.data_in_offset + n_idx * tp.col_num * tp.row_num +
                                    inner * tp.col_num * tp.loop_unit + outer * tp.inner_loop_unit)
                dst_start_offset = (tp.data_out_offset + n_idx * tp.col_num * tp.row_num +
                                    inner * tp.loop_unit + outer * tp.row_num * tp.inner_loop_unit)
                self._move_data_s13_each_loop(tp, ub_input,
                                            (tp.col_num - bursts) * self.b8_times,
                                            src_start_offset,
                                            (tp.row_num - tp.loop_unit) * self.b8_times,
                                            dst_start_offset,
                                            tp.loop_unit, bursts)
        with self.tik_inst.if_scope(tik.all(tp.last_loop_unit != 0,
                                            tp.col_num >= bursts, tp.row_num >= tp.last_loop_unit)):
            if self.is_mte_pad:
                self._move_data_s13_by_row_tail_loop_pad(tp, n_idx, ub_input, outer, bursts)
            else:
                self._move_data_s13_by_row_tail_loop_no_pad(tp, n_idx, ub_input, outer, bursts)

    def _move_data_s13_by_col_tail_loop_pad(self, tp, n_idx, ub_input, outer, bursts):
        src_start_offset = (tp.data_in_offset + outer * tp.loop_unit +
                            n_idx * tp.col_num * tp.row_num +
                            tp.inner_loop_cnt * tp.inner_loop_unit * tp.col_num)
        dst_start_offset = (tp.data_out_offset + outer * tp.row_num * tp.loop_unit +
                            n_idx * tp.col_num * tp.row_num +
                            tp.inner_loop_cnt * tp.inner_loop_unit)
        self._move_data_s13_each_loop(tp, ub_input,
                                      (tp.col_num - bursts) * self.b8_times,
                                      src_start_offset,
                                      (tp.row_num - tp.inner_last_loop_unit) * self.b8_times,
                                      dst_start_offset,
                                      tp.inner_last_loop_unit, bursts)

    def _move_data_s13_by_col_tail_loop_no_pad(self, tp, n_idx, ub_input, outer, bursts):
        with self.tik_inst.if_scope(tp.row_num >= self.ele_per_block):
            inner_last_loop_align = floor_align(tp.inner_last_loop_unit, self.ele_per_block)
            with self.tik_inst.if_scope(inner_last_loop_align > 0):
                src_start_offset = (tp.data_in_offset + outer * tp.loop_unit +
                                    n_idx * tp.col_num * tp.row_num +
                                    tp.inner_loop_cnt * tp.inner_loop_unit * tp.col_num)
                dst_start_offset = (tp.data_out_offset + outer * tp.row_num * tp.loop_unit +
                                    n_idx * tp.col_num * tp.row_num +
                                    tp.inner_loop_cnt * tp.inner_loop_unit)
                self._move_data_s13_each_loop(tp, ub_input,
                                              (tp.col_num - bursts) * self.b8_times,
                                              src_start_offset,
                                              (tp.row_num - inner_last_loop_align) * self.b8_times,
                                              dst_start_offset,
                                              inner_last_loop_align, bursts)
            # step back copy in and copy out
            with self.tik_inst.if_scope(tp.inner_last_loop_unit % self.ele_per_block != 0):
                tp.back_step.set_as(self.ele_per_block - tp.inner_last_loop_unit % self.ele_per_block)
                src_start_offset = (tp.data_in_offset + outer * tp.loop_unit +
                                    n_idx * tp.col_num * tp.row_num +
                                    tp.inner_loop_cnt * tp.inner_loop_unit * tp.col_num +
                                    inner_last_loop_align * tp.col_num - tp.back_step * tp.col_num)
                dst_start_offset = (tp.data_out_offset + outer * tp.row_num * tp.loop_unit +
                                    n_idx * tp.col_num * tp.row_num +
                                    tp.inner_loop_cnt * tp.inner_loop_unit +
                                    inner_last_loop_align - tp.back_step)
                self._move_data_s13_each_loop(tp, ub_input,
                                              (tp.col_num - bursts) * self.b8_times,
                                              src_start_offset,
                                              (tp.row_num - self.ele_per_block) * self.b8_times,
                                              dst_start_offset,
                                              self.ele_per_block, bursts)

    def _move_data_s13_by_col_loop(self, tp, n_idx, ub_input, outer, bursts):
        # this if_scope is neccesary
        # compile failed when major_loop_cnt is 0 but tp.row_num < tp.inner_loop_unit, because pass has bugs
        with self.tik_inst.if_scope(tik.all(tp.col_num >= bursts, tp.row_num >= tp.inner_loop_unit)):
            with self.tik_inst.for_range(0, tp.inner_loop_cnt) as inner:
                src_start_offset = (tp.data_in_offset + n_idx * tp.col_num * tp.row_num +
                                    outer * tp.loop_unit + inner * tp.inner_loop_unit * tp.col_num)
                dst_start_offset = (tp.data_out_offset + n_idx * tp.col_num * tp.row_num +
                                    outer * tp.row_num * tp.loop_unit + inner * tp.inner_loop_unit)
                self._move_data_s13_each_loop(tp, ub_input,
                                            (tp.col_num - bursts) * self.b8_times,
                                            src_start_offset,
                                            (tp.row_num - tp.inner_loop_unit) * self.b8_times,
                                            dst_start_offset,
                                            tp.inner_loop_unit, bursts)
        with self.tik_inst.if_scope(tik.all(tp.inner_last_loop_unit != 0,
                                            tp.col_num >= bursts, tp.row_num >= tp.inner_last_loop_unit)):
            if self.is_mte_pad:
                self._move_data_s13_by_col_tail_loop_pad(tp, n_idx, ub_input, outer, bursts)
            else:
                self._move_data_s13_by_col_tail_loop_no_pad(tp, n_idx, ub_input, outer, bursts)

    def _move_data_s13_by_row_twice_vnc(self, tp, ub_input, src_start_offset, dst_start_offset):
        # 16*(n*16)*col -> 16*vncov_step*col
        # 1. copy in
        with self.tik_inst.if_scope(tp.src_ub_gap == 1):
            self.tik_inst.data_move(ub_input, self.data_in[src_start_offset], 0, EPB16,
                                    tp.vncov_step * tp.inner_loop_unit // self.ele_per_block, 0, 1)
        with self.tik_inst.else_scope():
            self.tik_inst.data_move(ub_input, self.data_in[src_start_offset], 0, 1,
                                    EPB16 * tp.vncov_step * tp.inner_loop_unit // self.ele_per_block, 0, 0)
        # 2. first vnchwconv: 16*((n*16)*col) -> ((n*16)*col)*16
        vnc_info = (ub_input, 1, # loop_cnt
                    tp.vncov_step * tp.inner_loop_unit + tp.src_ub_gap * self.ele_per_block, # src_stride
                    EPB16, # dst_stride
                    tp.vncov_step * tp.inner_loop_unit // self.ele_per_block, # repeat_cnt
                    1, # src_repeat_stride: blocks
                    EPB16, # dst_repeat_stride: blocks
                    0, self.half_ub_num)
        self._move_data_s13_do_vnchwconv(vnc_info)
        # 3. reorder ((n*16)*col)*16 -> (n*16)*(col*16)
        with self.tik_inst.if_scope(tp.reorder_src_gap == 0):
            self.tik_inst.data_move(ub_input, ub_input[self.half_ub_num],
                                    0, 1,
                                    tp.vncov_step * tp.inner_loop_unit * EPB16 // self.ele_per_block,
                                    0, 0)
        with self.tik_inst.else_scope():
            self.tik_inst.data_move(ub_input, ub_input[self.half_ub_num],
                                    0, tp.vncov_step,
                                    tp.inner_loop_unit * EPB16 // self.ele_per_block,
                                    0, 1)
        # 4. second vnchwconv: (n*16)*(col*16) -> (col*16)*(n*16)
        src_stride = tp.inner_loop_unit * EPB16 + tp.reorder_src_gap * self.ele_per_block
        dst_stride = tp.vncov_step + tp.reorder_dst_gap * self.ele_per_block
        with self.tik_inst.if_scope(tp.vncov_step // EPB16 >= tp.inner_loop_unit):
            vnc_info = (ub_input, tp.inner_loop_unit * EPB16 // self.ele_per_block, # loop_cnt
                        src_stride, # src_stride
                        dst_stride, # dst_stride
                        tp.vncov_step // EPB16, # repeat_cnt
                        EPB16 * src_stride // self.ele_per_block, # src_repeat_stride: blocks
                        EPB16 // self.ele_per_block, # dst_repeat_stride: blocks
                        0, self.half_ub_num)
            self._move_data_s13_do_vnchwconv_col_loop(vnc_info)
        with self.tik_inst.else_scope():
            vnc_info = (ub_input, tp.vncov_step // EPB16, # loop_cnt
                        src_stride, # src_stride
                        dst_stride, # dst_stride
                        tp.inner_loop_unit * EPB16 // self.ele_per_block, # repeat_cnt
                        1, # src_repeat_stride: blocks
                        dst_stride, # dst_repeat_stride: blocks
                        0, self.half_ub_num)
            self._move_data_s13_do_vnchwconv(vnc_info)
        # 5. copy out
        self._move_data_s13_by_row_twice_vnc_out(tp, ub_input, dst_start_offset, dst_stride)

    def _move_data_s13_by_row_twice_vnc_out(self, tp, ub_input, dst_start_offset, dst_stride):
        with self.tik_inst.if_scope(tp.reorder_dst_gap == 1):
            with self.tik_inst.new_stmt_scope(disable_sync=True):
                with self.tik_inst.for_range(0, tp.inner_loop_unit) as loop:
                    self.tik_inst.data_move(self.data_out[dst_start_offset + loop * tp.row_num],
                                            ub_input[self.half_ub_num + loop * EPB16 * dst_stride],
                                            0, EPB16, tp.vncov_step // self.ele_per_block, 1, 0)
        with self.tik_inst.else_scope():
            if self.is_mte_pad:
                self._move_data_s13_by_row_twice_vnc_out_pad(tp, ub_input, dst_start_offset, dst_stride)
            else:
                self._move_data_s13_by_row_twice_vnc_out_npad(tp, ub_input, dst_start_offset, dst_stride)

    def _move_data_s13_by_row_twice_vnc_out_pad(self, tp, ub_input, dst_start_offset, dst_stride):
        dst_gap = (tp.row_num - EPB16 * tp.vncov_step) * self.b8_times
        with self.tik_inst.if_scope(dst_gap <= MTE_PAD_GAP_LIMIT):
            self.tik_inst.data_move_pad(self.data_out[dst_start_offset],
                                        ub_input[self.half_ub_num],
                                        tp.inner_loop_unit,
                                        EPB16 * tp.vncov_step * self.b8_times,
                                        dst_gap,
                                        0)
        with self.tik_inst.else_scope():
            self._move_data_s13_by_row_twice_vnc_out_npad(tp, ub_input, dst_start_offset, dst_stride)

    def _move_data_s13_by_row_twice_vnc_out_npad(self, tp, ub_input, dst_start_offset, dst_stride):
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, tp.inner_loop_unit) as loop:
                self.tik_inst.data_move(self.data_out[dst_start_offset + loop * tp.row_num],
                                        ub_input[self.half_ub_num + loop * EPB16 * dst_stride],
                                        0, 1, EPB16 * tp.vncov_step // self.ele_per_block, 0, 0)

    #             loop0             loop1
    # repeat0 ---------------- ----------------
    # repeat1 ---------------- ----------------
    def _move_data_s13_do_vnchwconv_col_loop(self, vnc_info):
        ub_input, loop_cnt, src_stride, dst_stride, repeat_cnt, src_repeat_stride, dst_repeat_stride, \
            offset_src, offset_dst = vnc_info
        if self.b8_times == EPB2:
            with self.tik_inst.for_range(0, loop_cnt) as loop:
                src_addr_list = [ub_input[offset_src + loop * self.ele_per_block +
                                          src_stride * i] for i in range(EPB16)]
                dst_addr_list = [ub_input[offset_dst + loop * dst_stride * self.ele_per_block +
                                          dst_stride * i] for i in range(EPB16)]
                with self.tik_inst.if_scope(repeat_cnt == 1):
                    self.tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, 1, 0, 0)
                with self.tik_inst.else_scope():
                    self.tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt,
                                            dst_repeat_stride, src_repeat_stride)
        else:
            with self.tik_inst.for_range(0, loop_cnt) as loop:
                src_addr_list = [ub_input[offset_src + loop * self.ele_per_block +
                                          src_stride * i] for i in range(EPB16)]
                dst_addr_list = [ub_input[offset_dst + loop * dst_stride * self.ele_per_block],
                                 ub_input[offset_dst + loop * dst_stride * self.ele_per_block + self.ele_per_block],
                                 ub_input[offset_dst + loop * dst_stride * self.ele_per_block + dst_stride],
                                 ub_input[offset_dst + loop * dst_stride * self.ele_per_block + dst_stride +
                                          self.ele_per_block],
                                 ub_input[offset_dst + loop * dst_stride * self.ele_per_block + dst_stride * 2],
                                 ub_input[offset_dst + loop * dst_stride * self.ele_per_block + dst_stride * 2 +
                                          self.ele_per_block],
                                 ub_input[offset_dst + loop * dst_stride * self.ele_per_block + dst_stride * 3],
                                 ub_input[offset_dst + loop * dst_stride * self.ele_per_block + dst_stride * 3 +
                                          self.ele_per_block],
                                 ub_input[offset_dst + loop * dst_stride * self.ele_per_block + dst_stride * 4],
                                 ub_input[offset_dst + loop * dst_stride * self.ele_per_block + dst_stride * 4 +
                                          self.ele_per_block],
                                 ub_input[offset_dst + loop * dst_stride * self.ele_per_block + dst_stride * 5],
                                 ub_input[offset_dst + loop * dst_stride * self.ele_per_block + dst_stride * 5 +
                                          self.ele_per_block],
                                 ub_input[offset_dst + loop * dst_stride * self.ele_per_block + dst_stride * 6],
                                 ub_input[offset_dst + loop * dst_stride * self.ele_per_block + dst_stride * 6 +
                                          self.ele_per_block],
                                 ub_input[offset_dst + loop * dst_stride * self.ele_per_block + dst_stride * 7],
                                 ub_input[offset_dst + loop * dst_stride * self.ele_per_block + dst_stride * 7 +
                                          self.ele_per_block]]
                with self.tik_inst.if_scope(repeat_cnt == 1):
                    self.tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, 1, 0, 0)
                with self.tik_inst.else_scope():
                    self.tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt,
                                            dst_repeat_stride, src_repeat_stride)

    #           repeat0           repeat1
    # loop0 ---------------- ----------------
    # loop1 ---------------- ----------------
    def _move_data_s13_do_vnchwconv(self, vnc_info):
        ub_input, loop_cnt, src_stride, dst_stride, repeat_cnt, src_repeat_stride, dst_repeat_stride, \
            offset_src, offset_dst = vnc_info
        if self.b8_times == EPB2:
            with self.tik_inst.for_range(0, loop_cnt) as loop:
                src_addr_list = [ub_input[offset_src + loop * src_stride * EPB16 +
                                          src_stride * i] for i in range(EPB16)]
                dst_addr_list = [ub_input[offset_dst + loop * EPB16 +
                                          dst_stride * i] for i in range(EPB16)]
                with self.tik_inst.if_scope(repeat_cnt == 1):
                    self.tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, 1, 0, 0)
                with self.tik_inst.else_scope():
                    self.tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt,
                                            dst_repeat_stride, src_repeat_stride)
        else:
            with self.tik_inst.for_range(0, loop_cnt) as loop:
                src_addr_list = [ub_input[offset_src + loop * src_stride * EPB16 +
                                          src_stride * i] for i in range(EPB16)]
                dst_addr_list = [ub_input[offset_dst + loop * EPB16],
                                 ub_input[offset_dst + loop * EPB16 + EPB8],
                                 ub_input[offset_dst + loop * EPB16 + dst_stride],
                                 ub_input[offset_dst + loop * EPB16 + dst_stride + EPB8],
                                 ub_input[offset_dst + loop * EPB16 + dst_stride * 2],
                                 ub_input[offset_dst + loop * EPB16 + dst_stride * 2 + EPB8],
                                 ub_input[offset_dst + loop * EPB16 + dst_stride * 3],
                                 ub_input[offset_dst + loop * EPB16 + dst_stride * 3 + EPB8],
                                 ub_input[offset_dst + loop * EPB16 + dst_stride * 4],
                                 ub_input[offset_dst + loop * EPB16 + dst_stride * 4 + EPB8],
                                 ub_input[offset_dst + loop * EPB16 + dst_stride * 5],
                                 ub_input[offset_dst + loop * EPB16 + dst_stride * 5 + EPB8],
                                 ub_input[offset_dst + loop * EPB16 + dst_stride * 6],
                                 ub_input[offset_dst + loop * EPB16 + dst_stride * 6 + EPB8],
                                 ub_input[offset_dst + loop * EPB16 + dst_stride * 7],
                                 ub_input[offset_dst + loop * EPB16 + dst_stride * 7 + EPB8]]
                with self.tik_inst.if_scope(repeat_cnt == 1):
                    self.tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, 1, 0, 0)
                with self.tik_inst.else_scope():
                    self.tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt,
                                            dst_repeat_stride, src_repeat_stride)

    def _move_data_s13_row_twice_vnc_gap(self, tp, outer_loops, inner_loops):
        # 16*(n*16)*col -> 16*vncov_step*col
        tp.vncov_step.set_as(outer_loops // EPB16)
        tp.src_ub_gap.set_as(0)
        tp.reorder_src_gap.set_as(0) # vnchwconv read bank conflict
        tp.reorder_dst_gap.set_as(0) # vnchwconv write bank conflict
        with self.tik_inst.if_scope(tp.vncov_step * inner_loops // self.ele_per_block % 2 == 0):
            tp.src_ub_gap.set_as(1)
        with self.tik_inst.if_scope(inner_loops * EPB16 // self.ele_per_block % 2 == 0):
            tp.reorder_src_gap.set_as(1)
        if self.b8_times == EPB2:
            with self.tik_inst.if_scope(tp.vncov_step // self.ele_per_block % 2 == 0):
                tp.reorder_dst_gap.set_as(1)
        else:
            with self.tik_inst.if_scope(tp.vncov_step // self.ele_per_block % 4 == 0):
                tp.reorder_dst_gap.set_as(1)

    def _move_data_s13_use_row_twice_vnc(self, tp, n_idx, ub_input):
        with self.tik_inst.if_scope(tp.major_loop_cnt != 0):
            self._move_data_s13_row_twice_vnc_gap(tp, tp.loop_unit, tp.inner_loop_unit)
            with self.tik_inst.for_range(0, tp.major_loop_cnt) as outer:
                src_start_offset = (tp.data_in_offset + n_idx * tp.col_num * tp.row_num +
                                    outer * tp.loop_unit * tp.inner_loop_unit)
                dst_start_offset = tp.data_out_offset + n_idx * tp.col_num * tp.row_num + outer * tp.loop_unit
                self._move_data_s13_by_row_twice_vnc(tp, ub_input, src_start_offset, dst_start_offset)
        with self.tik_inst.if_scope(tp.last_loop_unit != 0):
            nburst_cnt = floor_align(tp.last_loop_unit, 256)
            with self.tik_inst.if_scope(nburst_cnt > 0):
                self._move_data_s13_row_twice_vnc_gap(tp, nburst_cnt, tp.inner_loop_unit)
                src_start_offset = (tp.data_in_offset + n_idx * tp.col_num * tp.row_num +
                                    tp.major_loop_cnt * tp.loop_unit * tp.inner_loop_unit)
                dst_start_offset = (tp.data_out_offset + n_idx * tp.col_num * tp.row_num +
                                    tp.major_loop_cnt * tp.loop_unit)
                self._move_data_s13_by_row_twice_vnc(tp, ub_input, src_start_offset, dst_start_offset)
            with self.tik_inst.if_scope(tik.all(tp.last_loop_unit % 256 != 0, tp.row_num >= 256)):
                tp.back_step.set_as(256 - tp.last_loop_unit % 256)
                self._move_data_s13_row_twice_vnc_gap(tp, 256, tp.inner_loop_unit)
                src_start_offset = (tp.data_in_offset + n_idx * tp.col_num * tp.row_num +
                                    tp.major_loop_cnt * tp.loop_unit * tp.inner_loop_unit +
                                    nburst_cnt * tp.inner_loop_unit - tp.back_step * tp.inner_loop_unit)
                dst_start_offset = (tp.data_out_offset + n_idx * tp.col_num * tp.row_num +
                                    tp.major_loop_cnt * tp.loop_unit + nburst_cnt - tp.back_step)
                self._move_data_s13_by_row_twice_vnc(tp, ub_input, src_start_offset, dst_start_offset)

    def _move_data_s13_use_col_twice_vnc(self, tp, n_idx, ub_input):
        with self.tik_inst.if_scope(tp.major_loop_cnt != 0):
            # for fp32 we use vnc_split with 8,
            # if use vnc_split with 16 we need much more blocks for padding to avoid bank conflict
            # fp16: row*16*(n*16) -> row*vnc_split*vncov_step
            # fp32: row*8*(n*16)  -> row*vnc_split*vncov_step
            self._move_data_s13_col_twice_vnc_gap(tp, tp.loop_unit, tp.inner_loop_unit)
            with self.tik_inst.for_range(0, tp.major_loop_cnt) as outer:
                src_start_offset = tp.data_in_offset + n_idx * tp.col_num * tp.row_num + outer * tp.loop_unit
                dst_start_offset = (tp.data_out_offset + n_idx * tp.col_num * tp.row_num +
                                    outer * tp.loop_unit * tp.inner_loop_unit)
                self._move_data_s13_by_col_twice_vnc(tp, ub_input, src_start_offset, dst_start_offset)
        with self.tik_inst.if_scope(tp.last_loop_unit != 0):
            vnc_unit = EPB16 * self.vnc_split
            bursts_cnt = floor_align(tp.last_loop_unit, vnc_unit)
            with self.tik_inst.if_scope(bursts_cnt > 0):
                self._move_data_s13_col_twice_vnc_gap(tp, bursts_cnt, tp.inner_loop_unit)
                src_start_offset = (tp.data_in_offset + n_idx * tp.col_num * tp.row_num +
                                    tp.major_loop_cnt * tp.loop_unit)
                dst_start_offset = (tp.data_out_offset + n_idx * tp.col_num * tp.row_num +
                                    tp.major_loop_cnt * tp.loop_unit * tp.inner_loop_unit)
                self._move_data_s13_by_col_twice_vnc(tp, ub_input, src_start_offset, dst_start_offset)
            with self.tik_inst.if_scope(tik.all(tp.last_loop_unit % vnc_unit != 0, tp.col_num >= vnc_unit)):
                tp.back_step.set_as(vnc_unit - tp.last_loop_unit % vnc_unit)
                self._move_data_s13_col_twice_vnc_gap(tp, vnc_unit, tp.inner_loop_unit)
                src_start_offset = (tp.data_in_offset + n_idx * tp.col_num * tp.row_num +
                                    tp.major_loop_cnt * tp.loop_unit +
                                    bursts_cnt - tp.back_step)
                dst_start_offset = (tp.data_out_offset + n_idx * tp.col_num * tp.row_num +
                                    tp.major_loop_cnt * tp.loop_unit * tp.inner_loop_unit +
                                    bursts_cnt * tp.inner_loop_unit - tp.back_step * tp.inner_loop_unit)
                self._move_data_s13_by_col_twice_vnc(tp, ub_input, src_start_offset, dst_start_offset)

    def _move_data_s13_col_twice_vnc_gap(self, tp, outer_loops, inner_loops):
        tp.vncov_step.set_as(outer_loops // self.vnc_split)
        tp.src_ub_gap.set_as(0)
        tp.dst_ub_gap.set_as(0)
        tp.reorder_dst_gap.set_as(0)
        with self.tik_inst.if_scope(tp.vncov_step // self.ele_per_block % 2 == 0):
            tp.src_ub_gap.set_as(1)
        with self.tik_inst.if_scope(inner_loops * tp.vncov_step // self.ele_per_block % 2 == 0):
            tp.reorder_dst_gap.set_as(1)
        if self.b8_times == EPB2:
            with self.tik_inst.if_scope(inner_loops % 2 == 0):
                tp.dst_ub_gap.set_as(1)
        else:
            with self.tik_inst.if_scope(inner_loops * self.vnc_split % EPB16 != 0):
                tp.dst_ub_gap.set_as(1)
            with self.tik_inst.if_scope((inner_loops + tp.dst_ub_gap) % 4 == 0):
                tp.dst_ub_gap.set_as(tp.dst_ub_gap + 1)

    def _move_data_s13_by_col_twice_vnc_move_in(self, tp, ub_input, src_start_offset, src_stride):
        with self.tik_inst.if_scope(tp.src_ub_gap != 0):
            with self.tik_inst.new_stmt_scope(disable_sync=True):
                with self.tik_inst.for_range(0, tp.inner_loop_unit) as loop:
                    self.tik_inst.data_move(ub_input[loop * src_stride * self.vnc_split],
                                            self.data_in[src_start_offset + loop * tp.col_num],
                                            0, self.vnc_split,
                                            tp.vncov_step // self.ele_per_block,
                                            0, 1)
        with self.tik_inst.else_scope():
            if self.is_mte_pad:
                self._move_data_s13_by_col_twice_vnc_move_in_pad(tp, ub_input, src_start_offset, src_stride)
            else:
                self._move_data_s13_by_col_twice_vnc_move_in_npad(tp, ub_input, src_start_offset, src_stride)

    def _move_data_s13_by_col_twice_vnc_move_in_pad(self, tp, ub_input, src_start_offset, src_stride):
        src_gap = (tp.col_num - self.vnc_split * tp.vncov_step) * self.b8_times
        with self.tik_inst.if_scope(src_gap <= MTE_PAD_GAP_LIMIT):
            self.tik_inst.data_move_pad(ub_input,
                                        self.data_in[src_start_offset],
                                        tp.inner_loop_unit,
                                        self.vnc_split * tp.vncov_step * self.b8_times,
                                        0,
                                        src_gap)
        with self.tik_inst.else_scope():
            self._move_data_s13_by_col_twice_vnc_move_in_npad(tp, ub_input, src_start_offset, src_stride)

    def _move_data_s13_by_col_twice_vnc_move_in_npad(self, tp, ub_input, src_start_offset, src_stride):
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, tp.inner_loop_unit) as loop:
                self.tik_inst.data_move(ub_input[loop * src_stride * self.vnc_split],
                                        self.data_in[src_start_offset + loop * tp.col_num],
                                        0, 1,
                                        self.vnc_split * tp.vncov_step // self.ele_per_block,
                                        0, 0)

    def _move_data_s13_by_col_twice_vnc(self, tp, ub_input, src_start_offset, dst_start_offset):
        # for fp32 we use vnc_split with 8,
        # if use vnc_split with 16 we need much more blocks for padding to avoid bank conflict
        # fp16: row*16*(n*16) -> row*vnc_split*vncov_step
        # fp32: row*8*(n*16)  -> row*vnc_split*vncov_step
        # 1. copy in
        src_stride = tp.vncov_step + tp.src_ub_gap * self.ele_per_block
        self._move_data_s13_by_col_twice_vnc_move_in(tp, ub_input, src_start_offset, src_stride)

        # 2. first vnchwconv: (row*vnc_split)*(n*16) -> (n*16)*(row*vnc_split)
        dst_stride = tp.inner_loop_unit * self.vnc_split + tp.dst_ub_gap * self.vnc_split
        vnc_info = (ub_input, ceil_div(self.vnc_split * tp.inner_loop_unit, EPB16), # loop_cnt
                    src_stride,
                    dst_stride, # dst_stride
                    tp.vncov_step // self.ele_per_block, # repeat_cnt
                    1, # src_repeat_stride: blocks
                    dst_stride, # dst_repeat_stride: blocks
                    0, self.half_ub_num)
        self._move_data_s13_do_vnchwconv(vnc_info)
        # 3. reorder: (n*16)*(row*vnc_split) -> ((n*16)*row)*vnc_split
        with self.tik_inst.if_scope(tp.dst_ub_gap == 0):
            self.tik_inst.data_move(ub_input, ub_input[self.half_ub_num],
                                    0, 1,
                                    tp.inner_loop_unit * tp.vncov_step * self.vnc_split // self.ele_per_block,
                                    0, 0)
        with self.tik_inst.else_scope():
            self.tik_inst.data_move(ub_input, ub_input[self.half_ub_num],
                                    0, tp.vncov_step,
                                    tp.inner_loop_unit * self.vnc_split // self.ele_per_block,
                                    tp.dst_ub_gap, 0) # tp.dst_ub_gap maybe is 2
        # 4. second vnchwconv: ((n*16)*row)*vnc_split -> vnc_split*((n*16)*row)
        vnc_info = (ub_input, 1, # loop_cnt
                    self.vnc_split, # src_stride
                    tp.vncov_step * tp.inner_loop_unit + tp.reorder_dst_gap * self.ele_per_block, # dst_stride
                    tp.vncov_step * tp.inner_loop_unit // EPB16, # repeat_cnt
                    EPB16 * self.vnc_split // self.ele_per_block, # src_repeat_stride: blocks
                    EPB16 // self.ele_per_block, # dst_repeat_stride: blocks
                    0, self.half_ub_num)
        self._move_data_s13_do_vnchwconv_col_loop(vnc_info)
        # 5. copy out
        with self.tik_inst.if_scope(tp.reorder_dst_gap == 0):
            self.tik_inst.data_move(self.data_out[dst_start_offset],
                                    ub_input[self.half_ub_num],
                                    0, 1,
                                    self.vnc_split * tp.vncov_step * tp.inner_loop_unit // self.ele_per_block,
                                    0, 0)
        with self.tik_inst.else_scope():
            self.tik_inst.data_move(self.data_out[dst_start_offset],
                                    ub_input[self.half_ub_num],
                                    0, self.vnc_split,
                                    tp.vncov_step * tp.inner_loop_unit // self.ele_per_block,
                                    1, 0)

    def _move_data_s13_supported(self):
        return self.is_mte_pad or self.vg_cond

    def _move_data_s13(self, tp, ub_input_64):
        if self._move_data_s13_supported() and (self.b8_times == EPB2 or self.b8_times == EPB4):
            ub_input = ub_input_64.reinterpret_cast_to(self.x_dtype)

            # row loop should be inner
            with self.tik_inst.if_scope(tp.use_twice_vnc == 1):
                with self.tik_inst.if_scope(tp.split_by_row == 1):
                    with self.tik_inst.for_range(0, tp.n_loop_num) as n_idx:
                        self._move_data_s13_use_row_twice_vnc(tp, n_idx, ub_input)
                with self.tik_inst.else_scope():
                    with self.tik_inst.for_range(0, tp.n_loop_num) as n_idx:
                        self._move_data_s13_use_col_twice_vnc(tp, n_idx, ub_input)
            with self.tik_inst.else_scope():
                with self.tik_inst.if_scope(tp.split_by_row == 1):
                    with self.tik_inst.for_range(0, tp.n_loop_num) as n_idx:
                        with self.tik_inst.for_range(0, tp.inner_loop_cnt) as outer:
                            self._move_data_s13_by_row_loop(tp, n_idx, ub_input, outer, tp.inner_loop_unit)
                        with self.tik_inst.if_scope(tp.inner_last_loop_unit != 0):
                            self._move_data_s13_by_row_loop(tp, n_idx, ub_input, tp.inner_loop_cnt,
                                                            tp.inner_last_loop_unit)
                with self.tik_inst.else_scope():
                    with self.tik_inst.for_range(0, tp.n_loop_num) as n_idx:
                        with self.tik_inst.for_range(0, tp.major_loop_cnt) as outer:
                            self._move_data_s13_by_col_loop(tp, n_idx, ub_input, outer, tp.loop_unit)
                        with self.tik_inst.if_scope(tp.last_loop_unit != 0):
                            self._move_data_s13_by_col_loop(tp, n_idx, ub_input, tp.major_loop_cnt, tp.last_loop_unit)

    # -------------------------------------------------------------------------------------------------
    #                                    scenario_end
    # -------------------------------------------------------------------------------------------------

    def _do_tiling_s0(self, block_idx, tiling_reg_list, ub_input_64_t, ub_input_64, fixed_len, per_core_len):
        if self.is_mte_pad:
            ub_input_b8 = ub_input_64.reinterpret_cast_to("int8")
            data_tiling_b8 = self.data_tiling.reinterpret_cast_to("int8")
            self.tik_inst.data_move_pad(ub_input_b8[0],
                                        data_tiling_b8[(TILING_HEAD_LEN + fixed_len + block_idx * per_core_len) * EPB8],
                                        1, per_core_len * EPB8, 0, 0)
        else:
            self.tik_inst.data_move(ub_input_64[0],
                                    self.data_tiling[TILING_HEAD_LEN + fixed_len + block_idx * per_core_len],
                                    0, 1, ceil_div(per_core_len, ELE_NUM_PER_BLOCK_INT64), 0, 0)

        tp = self.TilingParamS0(tiling_reg_list, ub_input_64_t, ub_input_64)
        return tp

    def _do_tiling_s1(self, block_idx, tiling_reg_list, ub_input_64_t, ub_input_64, fixed_len, per_core_len):
        return self._do_tiling_common(block_idx, fixed_len, per_core_len, self.TilingParamS1)

    def _do_tiling_s2(self, block_idx, tiling_reg_list, ub_input_64_t, ub_input_64, fixed_len, per_core_len):
        return self._do_tiling_common(block_idx, fixed_len, per_core_len, self.TilingParamS2)

    def _do_tiling_s3(self, block_idx, tiling_reg_list, ub_input_64_t, ub_input_64, fixed_len, per_core_len):
        return self._do_tiling_common(block_idx, fixed_len, per_core_len, self.TilingParamS3)

    def _do_tiling_s4(self, block_idx, fixed_len, per_core_len):
        return self._do_tiling_common(block_idx, fixed_len, per_core_len, self.TilingParamS4)

    def _do_tiling_s5(self, block_idx, fixed_len, per_core_len):
        return self._do_tiling_common(block_idx, fixed_len, per_core_len, self.TilingParamS5)

    def _do_tiling_s7(self, block_idx, tiling_reg_list, ub_input_64_t, ub_input_64, fixed_len, per_core_len):
        return self._do_tiling_common(block_idx, fixed_len, per_core_len, self.TilingParamS7)

    def _do_tiling_s9(self, block_idx, fixed_len, per_core_len):
        return self._do_tiling_common(block_idx, fixed_len, per_core_len, self.TilingParamS9)

    def _do_tiling_s10(self, block_idx, fixed_len, per_core_len):
        return self._do_tiling_common(block_idx, fixed_len, per_core_len, self.TilingParamS5)

    def _do_tiling_s11(self, block_idx, fixed_len, per_core_len):
        return self._do_tiling_common(block_idx, fixed_len, per_core_len, self.TilingParamS11)

    def _do_tiling_s12(self, block_idx, fixed_len, per_core_len):
        return self._do_tiling_common(block_idx, fixed_len, per_core_len, self.TilingParamS12)

    def _do_tiling_common(self, block_idx, fixed_len, per_core_len, tp_class):
        tiling_reg_list = self.tiling_reg_list
        ub_input_64_t = self.ub_input_64_t
        ub_input_64 = self.ub_input_64
        if self.is_mte_pad:
            ub_input_b8 = ub_input_64.reinterpret_cast_to("int8")
            data_tiling_b8 = self.data_tiling.reinterpret_cast_to("int8")
            self.tik_inst.data_move_pad(ub_input_b8[0],
                                        data_tiling_b8[(TILING_HEAD_LEN + fixed_len + block_idx * per_core_len) * EPB8],
                                        1, per_core_len * EPB8, 0, 0)
        else:
            self.tik_inst.data_move(ub_input_64[0],
                                    self.data_tiling[TILING_HEAD_LEN + fixed_len + block_idx * per_core_len],
                                    0, 1, ceil_div(per_core_len, ELE_NUM_PER_BLOCK_INT64), 0, 0)
        tp = tp_class(tiling_reg_list, ub_input_64_t, ub_input_64, self.tik_inst)
        return tp

    def _do_tiling_s13(self, block_idx, tiling_reg_list, ub_input_64_t):
        tp = self.TilingParamS13(tiling_reg_list, ub_input_64_t, block_idx, self.tik_inst)
        return tp

    def _decode_tiling_head(self):
        core_num_var = self.tik_inst.Scalar("int64")
        scenario = self.tik_inst.Scalar("int64")
        fixed_len = self.tik_inst.Scalar("int64")
        per_core_len = self.tik_inst.Scalar("int64")
        sub_scenario = self.tik_inst.Scalar("int64")
        ub_input_64_core = self.tik_inst.Tensor("int64", (EPB8,), tik.scope_ubuf, "ub_input_64_core")
        self.tik_inst.data_move(ub_input_64_core, self.data_tiling, 0, 1, 2, 0, 0)
        scenario.set_as(ub_input_64_core[0])
        fixed_len.set_as(ub_input_64_core[1])
        per_core_len.set_as(ub_input_64_core[2])
        sub_scenario.set_as(ub_input_64_core[3])
        core_num_var.set_as(ub_input_64_core[4])

        back_params = (core_num_var, scenario, fixed_len, per_core_len, sub_scenario)
        return back_params

    def compute_tiling(self):
        """
        execution function
        """

        core_num_var, scenario, fixed_len, per_core_len, sub_scenario = self._decode_tiling_head()
        with self.tik_inst.for_range(0, core_num_var, block_num=core_num_var) as block_idx:
            self.ub_input_64_t = self.tik_inst.Tensor("int64", (256, ), tik.scope_ubuf, "ub_input_64_t")  # 2048B
            self.ub_input_b16_vor = self.tik_inst.Tensor("int16", (128, ), tik.scope_ubuf, "ub_input_b16_vor")  # 256B
            self.ub_input_b64_helper = self.tik_inst.Tensor("int64", (128, ), tik.scope_ubuf,
                                                            "ub_input_b64_helper")  # 1024B
            self._init_mem_make_ub_allocated(self.ub_input_b16_vor)
            self.ub_input_64 = self.tik_inst.Tensor("int64", (self.ub_size_64, ), tik.scope_ubuf, "ub_input_64")
            self._init_mem_make_ub_allocated(self.ub_input_64)

            valid_block = ((fixed_len + TILING_HEAD_LEN) * EPB8 + BLOCK_SIZE - 1) // BLOCK_SIZE
            self.tik_inst.data_move(self.ub_input_64_t, self.data_tiling, 0, 1, valid_block, 0, 0)

            with self.tik_inst.if_scope(tik.any(scenario == 7, scenario == 15)):
                tp = self._do_tiling_s7(block_idx, self.tiling_reg_list, self.ub_input_64_t,
                                        self.ub_input_64, fixed_len, per_core_len)
                tp.scenario = scenario
                self._move_data_s7(tp, sub_scenario)
            with self.tik_inst.else_scope():
                with self.tik_inst.if_scope(scenario == 1):
                    tp = self._do_tiling_s1(block_idx, self.tiling_reg_list, self.ub_input_64_t,
                                            self.ub_input_64, fixed_len, per_core_len)
                    self._move_data_s1(tp, self.ub_input_64)
                with self.tik_inst.else_scope():
                    with self.tik_inst.if_scope(tik.any(scenario == 2, scenario == 6)):
                        tp = self._do_tiling_s2(block_idx, self.tiling_reg_list, self.ub_input_64_t,
                                                self.ub_input_64, fixed_len, per_core_len)
                        self._move_data_s2(tp, self.ub_input_64)
                    with self.tik_inst.else_scope():
                        with self.tik_inst.if_scope(scenario == 3):
                            tp = self._do_tiling_s3(block_idx, self.tiling_reg_list, self.ub_input_64_t,
                                                    self.ub_input_64, fixed_len, per_core_len)
                            self._move_data_s3(tp, self.ub_input_64)
                        with self.tik_inst.else_scope():
                            with self.tik_inst.if_scope(scenario == 0):
                                tp = self._do_tiling_s0(block_idx, self.tiling_reg_list, self.ub_input_64_t,
                                                        self.ub_input_64, fixed_len, per_core_len)
                                self._move_data_s0(tp, self.ub_input_64)
                            with self.tik_inst.elif_scope(tik.any(scenario == 5, scenario == 14)):
                                tp = self._do_tiling_s5(block_idx, fixed_len, per_core_len)
                                tp.scenario = scenario
                                self._move_data_s5(tp, self.ub_input_64)
                            with self.tik_inst.elif_scope(scenario == 8):
                                self._move_data_s8(self.ub_input_64)
                            with self.tik_inst.elif_scope(scenario == 9):
                                tp = self._do_tiling_s9(block_idx, fixed_len, per_core_len)
                                self._move_data_s9(tp, self.ub_input_64, sub_scenario)
                            with self.tik_inst.elif_scope(scenario == 10):
                                tp = self._do_tiling_s10(block_idx, fixed_len, per_core_len)
                                self._move_data_s10(tp, self.ub_input_64)
                            with self.tik_inst.elif_scope(scenario == 11):
                                tp = self._do_tiling_s11(block_idx, fixed_len, per_core_len)
                                self._move_data_s11(tp, self.ub_input_64)
                            with self.tik_inst.elif_scope(scenario == 12):
                                tp = self._do_tiling_s12(block_idx, fixed_len, per_core_len)
                                self._move_data_s12(tp, self.ub_input_64, sub_scenario)
                            with self.tik_inst.elif_scope(scenario == 13):
                                tp = self._do_tiling_s13(block_idx, self.tiling_reg_list, self.ub_input_64_t)
                                self._move_data_s13(tp, self.ub_input_64)
        return scenario

    def compute(self, input_list, is_dynamic=False):
        """
        entrance function
        """
        tbe_context.get_context().add_compile_info("vars", {
            "ub_size": get_ub_size() // BLOCK_SIZE,
            "core_num": get_core_num(),
            "dtype": self.x_dtype,
            "cache_line": get_cache_line(),
            "vg_cond": int(self.vg_cond),
            "mte_pad": int(self.is_mte_pad)
        })
        self.tik_inst.set_tiling_params(input_list, self.data_tiling)
        scenario = self.compute_tiling()

        # this "global_variable_link" flag suggest ccec.py do link without "-r" option
        # which will result in global variable in cce file with wrong address
        tbe_context.get_context().add_compile_info("global_variable_link", True)
        opt_config = {"enable_const_fold": True, "enable_cce_remat_higher_weight": True}
        if is_dynamic:
            tiling_key_list = [[0], [1], [2], [3], [4],
                               [5], [6], [7], [8], [9],
                               [10], [11], [12], [13]]
            if self.vg_cond is True:
                tiling_key_list.extend([[14], [15]])
            tiling_params = {"build_multi_kernels" : {"tiling_key" : [scenario],
                                                      "tiling_key_value": tiling_key_list}}
            self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=input_list,
                                   outputs=[self.data_out],
                                   flowtable=[self.data_tiling],
                                   extend_params = tiling_params,
                                   config=opt_config)
            return {"compile_info": tbe_context.get_context().get_compile_info()}
        else:
            self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=input_list,
                                   outputs=[self.data_out],
                                   flowtable=[self.data_tiling],
                                   config=opt_config)
            return {"compile_info": tbe_context.get_context().get_compile_info()}


@register_operator("Transpose")
def transpose(x, perm, y, kernel_name="transpose"):
    """
    do transpose by perm attribute
    Parameters
    ----------
    x : dict
        shape and dtype of input
    perm : list or tuple
        permutation of the dimension of tensor
    y : dict
        shape and dtype of output, the dtype should be same as input
    kernel_name : str
        kernel name, default value is "transpose"
    Returns
    -------
    compile info
    """
    x_dtype = x.get("dtype").lower()
    p_dtype = perm.get("dtype").lower()
    y_dtype = y.get("dtype").lower()
    is_dynamic = util_common.is_unknown([x, perm, y])
    x_dtype, y_dtype = change_dtype(x_dtype, y_dtype)
    tik_inst = tik.Tik()
    data_in = tik_inst.Tensor(x_dtype, (MAX_INT64_VALUE,), tik.scope_gm, "x")
    data_perm = tik_inst.Tensor(p_dtype, (MAX_INT64_VALUE,), tik.scope_gm, "perm")
    data_out = tik_inst.Tensor(y_dtype, (MAX_INT64_VALUE,), tik.scope_gm, "y")
    data_workspace = tik_inst.Tensor(y_dtype, (1024,), tik.scope_gm, "data_workspace", is_workspace=True)
    data_tiling = tik_inst.Tensor("int64", (TILING_MAX_SIZE_GM,), tik.scope_gm, "data_tiling")
    tensor_list = [data_in, data_perm, data_out, data_workspace, data_tiling]
    input_list = [data_in, data_perm]
    transpose_instance = Transpose(tik_inst, x_dtype, tensor_list, kernel_name)
    return transpose_instance.compute(input_list, is_dynamic)
