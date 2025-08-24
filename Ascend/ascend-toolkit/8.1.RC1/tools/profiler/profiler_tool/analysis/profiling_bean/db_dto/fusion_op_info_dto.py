#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.


from dataclasses import dataclass
from profiling_bean.db_dto.dto_meta_class import InstanceCheckMeta


@dataclass
class FusionOpInfoDto(metaclass=InstanceCheckMeta):
    data_len: int = None
    fusion_op_names: str = None
    fusion_op_num: int = None
    level: str = None
    memory_input: str = None
    memory_output: str = None
    memory_total: str = None
    memory_weight: str = None
    memory_workspace: str = None
    op_name: str = None
    struct_type: str = None
    thread_id: int = None
    timestamp: float = None
