#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.


class TaskExecuteBean:
    def __init__(self: any, *args: any) -> None:
        self._stream_id = args[0]
        self._task_id = args[1]
        self._start_time = args[2]
        self._end_time = args[3]
        self._task_type = args[4]

    @property
    def stream_id(self: any) -> int:
        """
        for stream_id
        """
        return self._stream_id

    @property
    def task_id(self: any) -> int:
        """
        for task_id
        """
        return self._task_id

    @property
    def start_time(self: any) -> int:
        """
        for start time
        """
        return self._start_time

    @property
    def end_time(self: any) -> int:
        """
        for end time
        """
        return self._end_time

    @property
    def task_type(self: any) -> int:
        """
        for task type
        """
        return self._task_type