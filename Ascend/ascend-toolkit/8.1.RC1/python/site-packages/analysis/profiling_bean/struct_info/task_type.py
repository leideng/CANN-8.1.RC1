#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

from common_func.utils import Utils
from profiling_bean.struct_info.struct_decoder import StructDecoder


class TaskTypeBean(StructDecoder):
    """
    step trace
    """

    def __init__(self: any, *args: any) -> None:
        task_type_data = args[0]
        self._timestamp = task_type_data[4]
        self._stream_id = Utils.get_stream_id(task_type_data[5])
        self._task_id = task_type_data[6]
        self._task_type = task_type_data[7]
        self._task_state = task_type_data[8]

    @property
    def timestamp(self: any) -> int:
        """
        get timestamp
        :return: timestamp
        """
        return self._timestamp

    @property
    def stream_id(self: any) -> int:
        """
        get stream_id
        :return: stream_id
        """
        return self._stream_id

    @property
    def task_id(self: any) -> int:
        """
        get task_id
        :return: task_id
        """
        return self._task_id

    @property
    def task_type(self: any) -> int:
        """
        get task_type
        :return: task_type
        """
        return self._task_type

    @property
    def task_state(self: any) -> int:
        """
        get task state
        :return: task state
        """
        return self._task_state
