#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

from common_func.utils import Utils
from profiling_bean.stars.stars_common import StarsCommon
from profiling_bean.struct_info.struct_decoder import StructDecoder


class FftsLogDecoder(StructDecoder):
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
        self._thread_id = filed[9]
        # get lower 4 bit
        self._subtask_type = filed[8] & 15
        # get 5-7 bit
        self._ffts_type = (filed[8] & 112) >> 4

    @property
    def func_type(self: any) -> str:
        """
        get func_type
        :return: func_type
        """
        return self._func_type

    @property
    def task_type(self: any) -> int:
        """
        get task_type
        :return: task_type
        """
        return self._task_type

    @property
    def stars_common(self: any) -> object:
        """
        get stars_common info
        :return: stars_common
        """
        return self._stars_common

    @property
    def subtask_id(self: any) -> int:
        """
        get subtask_id
        :return: subtask_id
        """
        return self._subtask_id

    @property
    def thread_id(self: any) -> int:
        """
        get thread_id
        :return: thread_id
        """
        return self._thread_id

    @property
    def subtask_type(self: any) -> list:
        """
        get subtask_type
        :return: subtask_type
        """
        return self._subtask_type

    @property
    def ffts_type(self: any) -> list:
        """
        get ffts_type
        :return: ffts_type
        """
        return self._ffts_type
