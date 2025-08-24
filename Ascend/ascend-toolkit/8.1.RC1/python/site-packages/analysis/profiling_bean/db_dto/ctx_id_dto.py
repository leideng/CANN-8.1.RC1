#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.


from dataclasses import dataclass
from profiling_bean.db_dto.dto_meta_class import InstanceCheckMeta

INVALID_CONTEXT_ID = "4294967295"


@dataclass
class CtxIdDto(metaclass=InstanceCheckMeta):
    ctx_id: str = INVALID_CONTEXT_ID
    ctx_id_num: int = None
    data_len: int = None
    level: str = None
    op_name: str = None
    struct_type: str = None
    thread_id: int = None
    timestamp: float = None
