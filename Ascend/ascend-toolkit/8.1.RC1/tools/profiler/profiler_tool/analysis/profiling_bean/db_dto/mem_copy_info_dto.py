#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.


from dataclasses import dataclass
from profiling_bean.db_dto.dto_meta_class import InstanceCheckMeta


@dataclass
class MemCopyInfoDto(metaclass=InstanceCheckMeta):
    data_len: int = None
    data_size: int = None
    level: str = None
    memcpy_direction: str = None
    struct_type: str = None
    thread_id: int = None
    timestamp: float = None
