#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.


from dataclasses import dataclass
from profiling_bean.db_dto.dto_meta_class import InstanceCheckMeta


@dataclass
class NpuModuleMemDto(metaclass=InstanceCheckMeta):
    """
    Dto for npu module mem data
    """
    device_type: str = None
    module_id: int = None
    syscnt: int = None
    total_size: int = None
