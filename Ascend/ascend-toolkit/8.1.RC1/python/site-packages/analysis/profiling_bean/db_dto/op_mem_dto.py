#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.


from dataclasses import dataclass
from profiling_bean.db_dto.dto_meta_class import InstanceCheckMeta


@dataclass
class OpMemDto(metaclass=InstanceCheckMeta):
    """
    Dto for npu op mem data
    """
    allocation_time: int = None
    allocation_total_allocated: int = None
    allocation_total_reserved: int = None
    device_type: str = None
    duration: int = None
    name: str = None
    operator: str = None
    release_time: int = None
    release_total_allocated: int = None
    release_total_reserved: int = None
    size: int = None
