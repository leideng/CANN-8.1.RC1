#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from dataclasses import dataclass


@dataclass
class NpuMemDto:
    """
    Dto for npu mem data
    """

    event: str = None
    ddr: int = None
    hbm: int = None
    memory: int = None
    timestamp: float = None
