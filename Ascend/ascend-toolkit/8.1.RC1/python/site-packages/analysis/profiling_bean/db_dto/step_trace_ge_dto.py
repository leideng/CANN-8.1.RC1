#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

from dataclasses import dataclass
from common_func.constant import Constant
from profiling_bean.db_dto.dto_meta_class import InstanceCheckMeta


@dataclass
class StepTraceGeDto(metaclass=InstanceCheckMeta):
    index_id: int = Constant.DEFAULT_INVALID_VALUE
    model_id: int = Constant.DEFAULT_INVALID_VALUE
    op_name: str = Constant.NA
    op_type: str = Constant.NA
    stream_id: int = Constant.DEFAULT_INVALID_VALUE
    tag_id: int = Constant.DEFAULT_INVALID_VALUE
    task_id: int = Constant.DEFAULT_INVALID_VALUE
    timestamp: float = Constant.DEFAULT_INVALID_VALUE
