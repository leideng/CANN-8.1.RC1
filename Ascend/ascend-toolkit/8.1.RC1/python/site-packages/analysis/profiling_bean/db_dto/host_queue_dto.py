#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.


from dataclasses import dataclass
from profiling_bean.db_dto.dto_meta_class import InstanceCheckMeta


@dataclass
class HostQueueDto(metaclass=InstanceCheckMeta):
    get_time: int = None
    index_id: int = None
    mode: str = None
    queue_capacity: int = None
    queue_size: int = None
    send_time: float = None
    total_time: float = None
