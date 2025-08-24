#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from dataclasses import dataclass
from common_func.constant import Constant
from common_func.hccl_info_common import DeviceHcclSource


@dataclass
class HcclOps:
    device_id: int = Constant.DEFAULT_VALUE
    model_id: int = Constant.DEFAULT_INVALID_VALUE
    index_id: int = Constant.DEFAULT_INVALID_VALUE
    thread_id: int = Constant.DEFAULT_INVALID_VALUE
    op_name: str = Constant.NA
    task_type: str = Constant.NA
    op_type: str = Constant.NA
    timestamp: int = Constant.DEFAULT_VALUE
    duration: int = Constant.DEFAULT_VALUE
    begin: int = Constant.DEFAULT_VALUE
    end: int = Constant.DEFAULT_VALUE
    is_dynamic: int = Constant.DEFAULT_INVALID_VALUE
    connection_id: int = Constant.DEFAULT_INVALID_VALUE
    relay: int = Constant.DEFAULT_INVALID_VALUE
    retry: int = Constant.DEFAULT_INVALID_VALUE
    data_type: str = Constant.NA
    alg_type: str = Constant.NA
    count: int = Constant.DEFAULT_INVALID_VALUE
    group_name: str = Constant.NA
    source: int = DeviceHcclSource.INVALID.value
    kfc_connection_id: int = Constant.DEFAULT_INVALID_VALUE


@dataclass
class HcclTask:
    model_id: int = Constant.DEFAULT_INVALID_VALUE
    index_id: int = Constant.DEFAULT_INVALID_VALUE
    name: str = Constant.NA
    group_name: str = Constant.NA
    plane_id: int = Constant.DEFAULT_VALUE
    timestamp: int = Constant.DEFAULT_VALUE
    duration: int = Constant.DEFAULT_VALUE
    stream_id: int = Constant.DEFAULT_VALUE
    task_id: int = Constant.DEFAULT_VALUE
    context_id: int = Constant.DEFAULT_VALUE
    batch_id: int = Constant.DEFAULT_VALUE
    iteration: int = Constant.DEFAULT_VALUE
    hccl_name: str = Constant.NA
    first_timestamp: int = Constant.DEFAULT_VALUE
    host_timestamp: int = Constant.DEFAULT_INVALID_VALUE
    iter_id: int = Constant.DEFAULT_VALUE
    device_id: int = Constant.DEFAULT_VALUE
    is_dynamic: int = Constant.DEFAULT_INVALID_VALUE
    is_master: int = Constant.DEFAULT_INVALID_VALUE
    op_name: str = Constant.NA
    op_type: str = Constant.NA
    task_type: str = Constant.NA
    connection_id: int = Constant.DEFAULT_INVALID_VALUE
    duration_estimated: int = Constant.DEFAULT_INVALID_VALUE
    local_rank: int = Constant.DEFAULT_INVALID_VALUE
    remote_rank: int = Constant.DEFAULT_INVALID_VALUE
    transport_type: str = Constant.NA
    data_type: str = Constant.NA
    link_type: str = Constant.NA
    size: int = Constant.DEFAULT_INVALID_VALUE
    bandwidth: int = Constant.DEFAULT_INVALID_VALUE
    notify_id: int = Constant.DEFAULT_INVALID_VALUE
    rdma_type: str = Constant.NA
    thread_id: int = Constant.DEFAULT_INVALID_VALUE
