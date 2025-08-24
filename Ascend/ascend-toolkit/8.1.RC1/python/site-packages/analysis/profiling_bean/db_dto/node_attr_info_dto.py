#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

from dataclasses import dataclass
from profiling_bean.db_dto.dto_meta_class import InstanceCheckMeta


@dataclass
class NodeAttrInfoDto(metaclass=InstanceCheckMeta):
    level: str = None
    struct_type: str = None
    thread_id: int = None
    timestamp: float = None
    op_name: str = None
    hashid: str = None

