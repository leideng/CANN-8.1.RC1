#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from enum import Enum


class RoleType(Enum):
    DST = 0
    SRC = 1
    RESERVED = 255
    INVALID_TYPE = 4294967295


class OpType(Enum):
    SUM = 0
    MUL = 1
    MAX = 2
    MIN = 3
    RESERVED = 255
    INVALID_TYPE = 4294967295


class DataType(Enum):
    INT8 = 0
    INT16 = 1
    INT32 = 2
    FP16 = 3
    FP32 = 4
    INT64 = 5
    UINT64 = 6
    UINT8 = 7
    UINT16 = 8
    UINT32 = 9
    FP64 = 10
    BFP16 = 11
    INT128 = 12
    RESERVED = 255
    INVALID_TYPE = 4294967295


class LinkType(Enum):
    ON_CHIP = 0
    HCCS = 1
    PCIE = 2
    ROCE = 3
    SIO = 4
    HCCS_SW = 5
    STANDARD_ROCE = 6
    RESERVED = 255
    INVALID_TYPE = 4294967295


class TransPortType(Enum):
    SDMA = 0
    RDMA = 1
    LOCAL = 2
    RESERVED = 255
    INVALID_TYPE = 4294967295


class RdmaType(Enum):
    RDMA_SEND_NOTIFY = 0
    RDMA_SEND_PAYLOAD = 1
    RESERVED = 255
    INVALID_TYPE = 4294967295


class AlgType(Enum):
    NONE = 0
    MESH = 1
    RING = 2
    NB = 3
    HD = 4
    NHR = 5
    PIPELINE = 6
    PAIRWISE = 7
    STAR = 8
    INVALID = 4294967295


class DeviceHcclSource(Enum):
    HCCL = 0
    MC2 = 1
    INVALID = 65535


def trans_enum_name(enum_class, value):
    try:
        return enum_class(int(value)).name
    except ValueError:
        return value
