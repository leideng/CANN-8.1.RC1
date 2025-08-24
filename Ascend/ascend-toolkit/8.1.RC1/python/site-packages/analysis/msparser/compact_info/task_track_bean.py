#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from msparser.compact_info.compact_info_bean import CompactInfoBean


class TaskTrackBean(CompactInfoBean):
    """
    task track bean
    """
    def __init__(self: any, *args) -> None:
        super().__init__(*args)
        data = args[0]
        self._device_id = data[6]
        self._stream_id = data[7]
        self._task_id = data[8]
        self._batch_id = data[9]
        self._task_type = data[10]

    @property
    def device_id(self: any) -> int:
        """
        task track device_id
        """
        return self._device_id

    @property
    def stream_id(self: any) -> int:
        """
        task track stream_id
        """
        return self._stream_id

    @property
    def task_id(self: any) -> int:
        """
        task track task_id
        """
        return self._task_id

    @property
    def batch_id(self: any) -> int:
        """
        task track batch_id
        """
        return self._batch_id

    @property
    def task_type(self: any) -> int:
        """
        task track task_type
        """
        return str(self._task_type)

    @batch_id.setter
    def batch_id(self: any, batch_id) -> None:
        """
        task track batch_id
        """
        self._batch_id = batch_id
