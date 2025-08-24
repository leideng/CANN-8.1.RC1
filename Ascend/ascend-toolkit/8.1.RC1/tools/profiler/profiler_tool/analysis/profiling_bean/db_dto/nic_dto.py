#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

from dataclasses import dataclass
from profiling_bean.db_dto.dto_meta_class import InstanceCheckMeta


@dataclass
class NicDto(metaclass=InstanceCheckMeta):
    """
    Dto for ge model time data
    """
    bandwidth: float = None
    rx_dropped_rate: float = None
    rx_error_rate: float = None
    rx_packet: float = None
    rxbyte: float = None
    timestamp: float = None
    tx_dropped_rate: float = None
    tx_error_rate: float = None
    tx_packet: float = None
    txbyte: float = None
