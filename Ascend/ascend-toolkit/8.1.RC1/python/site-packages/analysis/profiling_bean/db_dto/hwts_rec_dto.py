#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

from dataclasses import dataclass
from profiling_bean.db_dto.dto_meta_class import InstanceCheckMeta


@dataclass
class HwtsRecDto(metaclass=InstanceCheckMeta):
    """
    hwts rec dto
    """
    ai_core_num: int = None
    iter_id: int = None
    sys_cnt: int = None
    task_count: int = None
