#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.


from dataclasses import dataclass
from profiling_bean.db_dto.dto_meta_class import InstanceCheckMeta


@dataclass
class GeHashDto(metaclass=InstanceCheckMeta):
    hash_key: str = None
    hash_value: str = None
    level: str = None
