#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from dataclasses import dataclass
from profiling_bean.db_dto.dto_meta_class import InstanceCheckMeta


@dataclass
class TaskTrackDto(metaclass=InstanceCheckMeta):
    batch_id: int = None
    data_len: int = None
    device_id: int = None
    level: str = None
    stream_id: int = None
    struct_type: str = None
    task_id: int = None
    task_type: str = None
    thread_id: int = None
    timestamp: float = None
