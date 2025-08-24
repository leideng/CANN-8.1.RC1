#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

from dataclasses import dataclass
from common_func.constant import Constant
from profiling_bean.db_dto.dto_meta_class import InstanceCheckMeta


@dataclass
class TimeSectionDto(metaclass=InstanceCheckMeta):
    duration_time: int = Constant.DEFAULT_INVALID_VALUE
    end_time: int = Constant.DEFAULT_INVALID_VALUE
    index_id: int = Constant.DEFAULT_INVALID_VALUE
    model_id: int = Constant.DEFAULT_INVALID_VALUE
    op_name: str = None
    overlap_time: int = Constant.DEFAULT_INVALID_VALUE
    start_time: int = Constant.DEFAULT_INVALID_VALUE
    stream_id: int = Constant.DEFAULT_INVALID_VALUE
    task_id: int = Constant.DEFAULT_INVALID_VALUE
    task_type: str = None


class CommunicationTimeSection(TimeSectionDto):
    def __init__(self):
        super().__init__()
