#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

from enum import Enum
from enum import unique


@unique
class ChipModel(Enum):
    """
    Define the chip type
    """
    CHIP_V1_1_0 = 0
    CHIP_V2_1_0 = 1
    CHIP_V3_1_0 = 2
    CHIP_V3_2_0 = 3
    CHIP_V3_3_0 = 4
    CHIP_V4_1_0 = 5
    CHIP_V1_1_1 = 7
    CHIP_V1_1_2 = 8
    CHIP_V1_1_3 = 11


class ChipCoreNum(Enum):
    """
    Define the ai core num of stars chip
    """
    CHIP_V4_1_0 = 24
    CHIP_V1_1_1 = 0
    CHIP_V1_1_2 = 0
    CHIP_V1_1_3 = 0
