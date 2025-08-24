#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

from dataclasses import dataclass
from common_func.constant import Constant
from profiling_bean.db_dto.dto_meta_class import InstanceCheckMeta


@dataclass
class GeTimeDto(metaclass=InstanceCheckMeta):
    infer_end: float = Constant.DEFAULT_VALUE
    infer_start: float = Constant.DEFAULT_VALUE
    input_end: float = Constant.DEFAULT_VALUE
    input_start: float = Constant.DEFAULT_VALUE
    model_id: int = Constant.DEFAULT_VALUE
    model_name: str = None
    output_end: float = Constant.DEFAULT_VALUE
    output_start: float = Constant.DEFAULT_VALUE
    request_id: int = Constant.DEFAULT_VALUE
    stage_num: int = Constant.DEFAULT_VALUE
    thread_id: int = Constant.DEFAULT_VALUE
