#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
from dataclasses import dataclass
from profiling_bean.db_dto.dto_meta_class import InstanceCheckMeta


@dataclass
class AccPmuOriDto(metaclass=InstanceCheckMeta):
    """
    Dto for acc pmu data
    """
    acc_id: int = None
    block_id: int = None
    read_bandwidth: int = None
    read_ost: int = None
    stream_id: int = None
    task_id: int = None
    timestamp: float = None
    write_bandwidth: int = None
    write_ost: int = None
