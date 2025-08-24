#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

from common_func.utils import Utils
from common_func.info_conf_reader import InfoConfReader


class StarsCommon:
    """
    class for decode binary data
    """

    def __init__(self: any, task_id: int, stream_id: int, timestamp: int or float) -> None:
        self._stream_id = self.set_stream_id(stream_id, task_id)
        self._task_id = self.set_task_id(stream_id, task_id)
        self._timestamp = timestamp

    @property
    def task_id(self: any) -> int:
        """
        get task id
        :return: task id
        """
        return self._task_id

    @property
    def stream_id(self: any) -> int:
        """
        get stream id
        :return: stream id
        """
        return self._stream_id

    @property
    def timestamp(self: any) -> float:
        """
        get timestamp
        :return: timestamp
        """
        return InfoConfReader().time_from_syscnt(self._timestamp)

    @staticmethod
    def set_stream_id(stream_id, task_id):
        """
        In the ffts scenario, when the 14th bit of the stream id is set,
        the lower 12 bits of the stream id need to be exchanged with the task id.
        when the 13th bit of the stream id is set, get the lower 12 bits of the stream id.
        """
        if stream_id & 0x1000 != 0:
            return Utils.get_stream_id(stream_id)
        if stream_id & 0x2000 != 0:
            stream_id = task_id & 0x0FFF
        return Utils.get_stream_id(stream_id)

    @staticmethod
    def set_task_id(stream_id, task_id):
        """
        In the ffts scenario,
        when the 14th bit of the stream id is set,
        the lower 12 bits of the stream id need to be exchanged with the task id.
        when only the 13th bit of the stream id is set,
        the high 4 bits of the stream id need to be exchanged with the task id.
        """
        if stream_id & 0x1000 != 0:
            task_id = task_id & 0x1FFF
            task_id |= (stream_id & 0xE000)
        elif stream_id & 0x2000 != 0:
            task_id = (stream_id & 0x0FFF) | (task_id & 0xF000)
        return task_id
