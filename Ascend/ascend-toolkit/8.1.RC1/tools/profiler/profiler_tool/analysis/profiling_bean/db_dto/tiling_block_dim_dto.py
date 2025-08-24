#!/usr/bin/python3
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from dataclasses import dataclass

from common_func.constant import Constant
from profiling_bean.db_dto.dto_meta_class import InstanceCheckMeta


@dataclass
class TilingBlockDimDto(metaclass=InstanceCheckMeta):
    stream_id: int = Constant.DEFAULT_INVALID_VALUE
    task_id: int = Constant.DEFAULT_INVALID_VALUE
    timestamp: float = Constant.DEFAULT_INVALID_VALUE
    block_dim: int = Constant.DEFAULT_INVALID_VALUE
    batch_id: int = Constant.DEFAULT_INVALID_VALUE
