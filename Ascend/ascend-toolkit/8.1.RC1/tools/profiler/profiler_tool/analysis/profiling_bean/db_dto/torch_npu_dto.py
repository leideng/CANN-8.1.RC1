#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from dataclasses import dataclass
from profiling_bean.db_dto.dto_meta_class import InstanceCheckMeta


@dataclass
class TorchNpuDto(metaclass=InstanceCheckMeta):
    """
    Dto for relationship between torch op and npu kernel
    """
    acl_compile_time: int = None
    acl_end_time: int = None
    acl_start_time: int = None
    acl_tid: int = None
    batch_id: int = None
    context_id: int = None
    op_name: str = None
    stream_id: int = None
    task_id: int = None
    torch_op_pid: int = None
    torch_op_start_time: int = None
    torch_op_tid: int = None
