#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.


class TaskTimeline:
    """
    contain start and end time of task
    """

    def __init__(self: any, stream_id: int, task_id: int) -> None:
        self._stream_id = stream_id
        self._task_id = task_id
        self.start_time = None
        self.end_time = None

    @property
    def stream_id(self: any) -> int:
        """
        stream id
        :return: stream id
        """
        return self._stream_id

    @property
    def task_id(self: any) -> int:
        """
        task id
        :return: task id
        """
        return self._task_id


class TaskStateHandler:
    """
    handle task state
    """
    RECEIVE_TAG = 0
    START_TAG = 1
    END_TAG = 2

    def __init__(self: any, stream_id: int, task_id: int) -> None:
        self._stream_id = stream_id
        self._task_id = task_id
        self.new_task = None
        self.task_timeline_list = []

    @property
    def stream_id(self: any) -> int:
        """
        stream id
        :return: stream id
        """
        return self._stream_id

    @property
    def task_id(self: any) -> int:
        """
        task id
        :return: task id
        """
        return self._task_id

    def process_record(self: any, timestamp: int, task_state: int) -> None:
        """
        process record
        :param timestamp: timestamp
        :param task_state: task state
        """
        if task_state == self.START_TAG:
            self.new_task = TaskTimeline(self.stream_id, self.task_id)
            self.new_task.start_time = timestamp

        if task_state == self.END_TAG and self.new_task:
            self.new_task.end_time = timestamp
            self.task_timeline_list.append(self.new_task)
