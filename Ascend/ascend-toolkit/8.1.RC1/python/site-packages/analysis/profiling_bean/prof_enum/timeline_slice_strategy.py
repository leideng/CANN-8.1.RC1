#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

from enum import Enum
from enum import unique


@unique
class TimeLineSliceStrategy(Enum):
    """
    Define the tag for timeline slice
    """
    FILE_NUM_PRIORITY = 0
    LOADING_TIME_PRIORITY = 1


@unique
class LoadingTimeLevel(Enum):
    """
    Define the level for timeline loading time
    """
    EXCELLENT_LEVEL = 1
    FINE_LEVEL = 10
    BAD_LEVEL = 30
