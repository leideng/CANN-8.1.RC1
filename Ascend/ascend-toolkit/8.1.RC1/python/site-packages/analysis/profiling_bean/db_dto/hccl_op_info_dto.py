#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.


from dataclasses import dataclass
from common_func.constant import Constant
from profiling_bean.db_dto.dto_meta_class import InstanceCheckMeta


@dataclass
class HCCLOpInfoDto(metaclass=InstanceCheckMeta):
    alg_type: str = Constant.NA
    count: int = Constant.DEFAULT_INVALID_VALUE
    data_type: str = Constant.NA
    group_name: str = Constant.NA
    level: str = Constant.NA
    relay: int = Constant.DEFAULT_INVALID_VALUE
    retry: int = Constant.DEFAULT_INVALID_VALUE
    struct_type: str = Constant.NA
    thread_id: int = Constant.DEFAULT_INVALID_VALUE
    timestamp: float = Constant.DEFAULT_INVALID_VALUE
