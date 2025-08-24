#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.


from dataclasses import dataclass
from profiling_bean.db_dto.dto_meta_class import InstanceCheckMeta


@dataclass
class DataQueueDto(metaclass=InstanceCheckMeta):
    duration: float = None
    end_time: float = None
    node_name: str = None
    queue_size: int = None
    start_time: float = None
