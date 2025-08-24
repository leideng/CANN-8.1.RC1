#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

from common_func.info_conf_reader import InfoConfReader
from common_func.ms_constant.number_constant import NumberConstant
from common_func.ms_constant.str_constant import StrConstant


class MemoryCopyConstant:
    """
    Constant for memory copy
    """
    DEFAULT_TIMESTAMP = -1
    DEFAULT_VIEWER_VALUE = "N/A"

    RECEIVE_TAG = 0
    START_TAG = 1
    END_TAG = 2
    STATES_TIMESTAMPS_RECEIVE_INDEX = 0
    STATES_TIMESTAMPS_START_INDEX = 1
    STATES_TIMESTAMPS_END_INDEX = 2

    # before reshape
    STREAM_INDEX = 0
    TASK_INDEX = 1
    TIMESTAMP_INDEX = 2
    TASK_STATE_INDEX = 3
    BATCH_INDEX = 4
    # after reshape
    RECEIVE_INDEX = 2
    START_INDEX = 3
    END_INDEX = 4

    ASYNC_MEMCPY_NAME = "MemcopyAsync"
    TYPE = "other"

    H2H_TAG = "0"
    H2D_TAG = "1"
    D2H_TAG = "2"
    D2D_TAG = "3"
    H2H_NAME = "host to host"
    H2D_NAME = "host to device"
    D2H_NAME = "device to host"
    D2D_NAME = "device to device"
    DEFAULTE_NAME = "other"

    @staticmethod
    def syscnt_to_micro() -> int:
        """
        syscnt to micro multiplication factor
        :return: int
        """
        return NumberConstant.MICRO_SECOND / InfoConfReader().get_freq(StrConstant.HWTS)

    @classmethod
    def get_tag_index(cls: any, tag: int) -> str:
        """
        class name
        """
        tag_index_dict = {
            cls.RECEIVE_TAG: cls.STATES_TIMESTAMPS_RECEIVE_INDEX,
            cls.START_TAG: cls.STATES_TIMESTAMPS_START_INDEX,
            cls.END_TAG: cls.STATES_TIMESTAMPS_END_INDEX
        }

        return tag_index_dict.get(tag)

    @classmethod
    def get_direction(cls: any, tag: str) -> str:
        """
        class name
        """
        direction_dict = {
            cls.H2D_TAG: cls.H2D_NAME,
            cls.D2H_TAG: cls.D2H_NAME,
            cls.D2D_TAG: cls.D2D_NAME,
            cls.H2H_TAG: cls.H2H_NAME
        }
        return direction_dict.get(tag, cls.DEFAULTE_NAME)
