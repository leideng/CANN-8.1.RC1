#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

from common_func.utils import Utils
from profiling_bean.stars.ffts_log import FftsLogDecoder
from profiling_bean.stars.stars_common import StarsCommon


class FftsPlusLogDecoder(FftsLogDecoder):
    """
    class used to decode binary data
    """

    def __init__(self: any, *args: tuple) -> None:
        filed = args[0]
        # get lower 6 bit
        self._func_type = Utils.get_func_type(filed[0])
        # get the most significant six bits
        self._task_type = filed[0] >> 10
        self._stars_common = StarsCommon(filed[3], filed[2], filed[4])
        self._subtask_id = filed[6]
        self._thread_id = filed[8]
        # get lower 8 bit
        self._subtask_type = filed[5] & 255
        # get 5-7 bit
        self._ffts_type = filed[7] >> 13
