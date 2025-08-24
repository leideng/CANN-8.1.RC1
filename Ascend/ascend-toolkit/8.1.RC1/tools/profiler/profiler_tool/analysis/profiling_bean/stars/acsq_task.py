#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

from common_func.info_conf_reader import InfoConfReader
from common_func.utils import Utils
from profiling_bean.stars.stars_common import StarsCommon
from profiling_bean.struct_info.struct_decoder import StructDecoder


class AcsqTask(StructDecoder):
    """
    class used to decode acsq task
    """

    def __init__(self: any, *args: any) -> None:
        args = args[0]
        # total 16 bit, get lower 6 bit
        self._func_type = Utils.get_func_type(args[0])
        # total 16 bit, get high 6 bit
        self._task_type = args[0] >> 10
        self._stream_id = StarsCommon.set_stream_id(args[2], args[3])
        self._task_id = StarsCommon.set_task_id(args[2], args[3])
        self._sys_cnt = args[4]
        # [acsq_id, acc_id] is total 16 bit, acc_id is the lower 6 bit
        self._acc_id = args[6] & 63
        # acsq_id is higher 10 bit where only lower 7 bit is valid
        self._acsq_id = (args[6] >> 6) & 127

    @property
    def func_type(self: any) -> str:
        """
        get func type
        :return: func type
        """
        return self._func_type

    @property
    def task_type(self: any) -> int:
        """
        get task type
        :return: task type
        """
        return self._task_type

    @property
    def stream_id(self: any) -> int:
        """
        get stream id
        :return: stream id
        """
        return self._stream_id

    @property
    def task_id(self: any) -> int:
        """
        get task id
        :return: task id
        """
        return self._task_id

    @property
    def sys_cnt(self: any) -> float:
        """
        get task sys cnt
        :return: sys cnt
        """
        return InfoConfReader().time_from_syscnt(self._sys_cnt)

    @property
    def acc_id(self: any) -> int:
        """
        get acc id
        :return: acc_id
        """
        return self._acc_id
