#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from dataclasses import dataclass
from common_func.constant import Constant
from common_func.ms_constant.number_constant import NumberConstant
from profiling_bean.db_dto.dto_meta_class import InstanceCheckMeta


@dataclass
class HCCLInfoDto(metaclass=InstanceCheckMeta):
    ccl_tag: str = Constant.NA
    context_id: int = NumberConstant.DEFAULT_GE_CONTEXT_ID
    data_len: int = Constant.DEFAULT_INVALID_VALUE
    data_type: str = Constant.NA
    dst_addr: str = Constant.DEFAULT_INVALID_VALUE
    duration_estimated: int = Constant.DEFAULT_INVALID_VALUE
    group_name: str = Constant.NA
    level: str = Constant.NA
    link_type: str = Constant.NA
    local_rank: int = Constant.DEFAULT_INVALID_VALUE
    notify_id: int = Constant.DEFAULT_INVALID_VALUE
    op_name: str = Constant.NA
    op_type: str = Constant.NA
    plane_id: int = Constant.DEFAULT_INVALID_VALUE
    rank_size: int = Constant.DEFAULT_INVALID_VALUE
    rdma_type: str = Constant.NA
    remote_rank: int = Constant.DEFAULT_INVALID_VALUE
    role: str = Constant.NA
    size: int = Constant.DEFAULT_INVALID_VALUE
    src_addr: str = Constant.DEFAULT_INVALID_VALUE
    stage: str = Constant.DEFAULT_INVALID_VALUE
    struct_type: str = Constant.NA
    thread_id: int = Constant.DEFAULT_INVALID_VALUE
    timestamp: float = Constant.DEFAULT_INVALID_VALUE
    transport_type: str = Constant.NA
    work_flow_mode: str = Constant.NA
