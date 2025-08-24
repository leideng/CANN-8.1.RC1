#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from dataclasses import dataclass
from common_func.ms_constant.number_constant import NumberConstant


@dataclass
class HostTask:
    """
    This class represents a runtime host task that adapts to all chips.
    """
    model_id: int
    index_id: int
    stream_id: int
    task_id: int
    context_id: int
    batch_id: int
    task_type: str
    device_id: int
    host_timestamp: int
    connection_id: int


@dataclass
class DeviceTask:
    """
    This class represents a device task that adapts to all chips.
    """
    stream_id: int
    task_id: int
    context_id: int
    timestamp: int
    duration: int
    task_type: str
    batch_id: int = NumberConstant.DEFAULT_BATCH_ID


@dataclass
class TopDownTask:
    """
    this class represents an top-down task which
    contains complete ascend software and hardware info.
    The instance of this class represents the smallest
    schedulable unit running on a device.
    """
    model_id: int
    index_id: int
    stream_id: int
    task_id: int
    context_id: int
    batch_id: int
    start_time: int
    duration: int
    host_task_type: str
    device_task_type: str
    connection_id: int
