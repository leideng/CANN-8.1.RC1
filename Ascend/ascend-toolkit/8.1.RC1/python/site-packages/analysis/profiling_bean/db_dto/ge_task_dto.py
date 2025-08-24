#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.


from dataclasses import dataclass
from profiling_bean.db_dto.dto_meta_class import InstanceCheckMeta


@dataclass
class GeTaskDto(metaclass=InstanceCheckMeta):
    batch_id: int = None
    block_dim: int = None
    context_id: int = None
    index_id: int = None
    mix_block_dim: int = None
    model_id: int = None
    op_name: str = None
    op_state: str = None
    op_type: str = None
    stream_id: int = None
    task_id: str = None
    task_type: int = None
    thread_id: int = None
    timestamp: float = None
