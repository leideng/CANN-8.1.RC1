#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from enum import Enum
from enum import unique


@unique
class LevelDataType(Enum):
    """
    DataType enum for level data
    """

    PYTORCH_TX = 30000
    PTA = 25000
    MSPROFTX = 20500
    ACL = 20000
    MODEL = 15000
    NODE = 10000
    AICPU = 6000
    COMMUNICATION = 5500
    RUNTIME = 5000

    @classmethod
    def member_map(cls: any) -> dict:
        """
        enum map for DataFormat value and data format member
        :return:
        """
        return cls._value2member_map_
