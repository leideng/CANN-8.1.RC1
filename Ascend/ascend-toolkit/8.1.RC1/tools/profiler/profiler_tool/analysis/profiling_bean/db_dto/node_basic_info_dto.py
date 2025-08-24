#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from dataclasses import dataclass
from profiling_bean.db_dto.dto_meta_class import InstanceCheckMeta


@dataclass
class NodeBasicInfoDto(metaclass=InstanceCheckMeta):
    block_dim: int = None
    data_len: int = None
    is_dynamic: int = None
    level: str = None
    mix_block_dim: int = None
    op_flag: str = None
    op_name: str = None
    op_type: str = None
    struct_type: str = None
    task_type: str = None
    thread_id: int = None
    timestamp: float = None
