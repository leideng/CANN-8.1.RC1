#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from dataclasses import dataclass
from profiling_bean.db_dto.dto_meta_class import InstanceCheckMeta


@dataclass
class NpuOpMemDto(metaclass=InstanceCheckMeta):
    """
    Dto for npu op mem data
    """
    addr: str = None
    device_type: str = None
    level: str = None
    operator: str = None
    size: int = None
    thread_id: int = None
    timestamp: float = None
    total_allocate_memory: int = None
    total_reserve_memory: int = None
    type_: str = None
