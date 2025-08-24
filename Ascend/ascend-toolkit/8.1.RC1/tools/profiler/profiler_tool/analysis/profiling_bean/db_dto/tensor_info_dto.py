#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from dataclasses import dataclass
from profiling_bean.db_dto.dto_meta_class import InstanceCheckMeta


@dataclass
class TensorInfoDto(metaclass=InstanceCheckMeta):
    data_len: int = None
    input_data_types: str = None
    input_formats: str = None
    input_shapes: str = None
    level: str = None
    op_name: str = None
    output_data_types: str = None
    output_formats: str = None
    output_shapes: str = None
    struct_type: str = None
    tensor_num: int = None
    thread_id: int = None
    timestamp: float = None
