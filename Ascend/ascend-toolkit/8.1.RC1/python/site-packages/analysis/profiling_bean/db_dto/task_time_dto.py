#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
from dataclasses import dataclass
from profiling_bean.db_dto.dto_meta_class import InstanceCheckMeta


@dataclass
class TaskTimeDto(metaclass=InstanceCheckMeta):
    """
    Dto for stars data or hwts data
    """
    batch_id: int = None
    dur_time: float = None
    end_time: float = None
    ffts_type: int = None
    op_name: str = None
    start_time: float = None
    stream_id: int = None
    subtask_id: int = None
    subtask_type: str = None
    task_id: int = None
    task_time: float = None
    task_type: str = None
    thread_id: int = None
