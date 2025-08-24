#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.


from dataclasses import dataclass
from profiling_bean.db_dto.dto_meta_class import InstanceCheckMeta


@dataclass
class ClusterRankDto(metaclass=InstanceCheckMeta):
    device_id: int = None
    dir_name: str = None
    job_info: str = None
    rank_id: str = None
