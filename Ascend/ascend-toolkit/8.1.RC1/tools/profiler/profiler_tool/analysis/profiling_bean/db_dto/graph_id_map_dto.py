#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.


from dataclasses import dataclass
from profiling_bean.db_dto.dto_meta_class import InstanceCheckMeta


@dataclass
class GraphIdMapDto(metaclass=InstanceCheckMeta):
    data_len: int = None
    graph_id: int = None
    level: str = None
    model_name: str = None
    struct_type: str = None
    thread_id: int = None
    timestamp: float = None
