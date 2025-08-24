#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

from .constant import (
    MsptiActivityKind,
    MsptiActivityFlag,
    MsptiActivitySourceKind
)


class KernelData:
    KIND = "kind"
    START = "start"
    END = "end"
    DEVICE_ID = "deviceId"
    STREAM_ID = "streamId"
    CORRELATION_ID = "correlationId"
    TYPE = "type"
    NAME = "name"

    def __init__(self, origin_data: dict):
        # Activity record kind, must be MSPTI_ACTIVITY_KIND_KERNEL
        self.kind: MsptiActivityKind = MsptiActivityKind(origin_data.get(self.KIND, 0))
        # Start timestamp for the kernel, in ns
        self.start: int = origin_data.get(self.START, 0)
        # End timestamp for the kernel, in ns
        self.end: int = origin_data.get(self.END, 0)
        # Device id
        self.device_id: int = origin_data.get(self.DEVICE_ID, 0)
        # Stream id
        self.stream_id: int = origin_data.get(self.STREAM_ID, 0)
        # Correlation id
        self.correlation_id: int = origin_data.get(self.CORRELATION_ID, 0)
        # Kernel type
        self.type: str = origin_data.get(self.TYPE, "")
        # Kernel name
        self.name: str = origin_data.get(self.NAME, "")


class MsptiObjectId:
    PROCESS_ID = "processId"
    THREAD_ID = "threadId"
    DEVICE_ID = "deviceId"
    STREAM_ID = "streamId"

    def __init__(self, origin_data):
        # A thread object requires that we identify process and thread ID
        self.process_id: int = origin_data.get(self.PROCESS_ID, -1)
        self.thread_id: int = origin_data.get(self.THREAD_ID, -1)

        # A stream object requires that we identify device and stream ID
        self.device_id: int = origin_data.get(self.DEVICE_ID, 0)
        self.stream_id: int = origin_data.get(self.STREAM_ID, 0)


class MarkerData:
    KIND = "kind"
    FLAG = "flag"
    SOURCE_KIND = "sourceKind"
    TIMESTAMP = "timestamp"
    ID = "id"
    NAME = "name"
    DOMAIN = "domain"

    def __init__(self, origin_data: dict):
        # Activity record kind, must be MSPTI_ACTIVITY_KIND_MARKER
        self.kind: MsptiActivityKind = MsptiActivityKind(origin_data.get(self.KIND, 0))
        # Flags associated with the marker
        self.flag: MsptiActivityFlag = MsptiActivityFlag(origin_data.get(self.FLAG, 1))
        # Source kinds of mark data
        self.source_kind: MsptiActivitySourceKind = MsptiActivitySourceKind(origin_data.get(self.SOURCE_KIND, 0))
        # Timestamp for the marker, in ns.
        self.timestamp: int = origin_data.get(self.TIMESTAMP, 0)
        # Marker id
        self.id: int = origin_data.get(self.ID, 0)
        # Identifier for the activity object associated with the marker
        self.object_id: MsptiObjectId = MsptiObjectId(origin_data)
        # Marker name for an instantaneous or start marker
        self.name: str = origin_data.get(self.NAME, "")
        # Name of the domain to which this marker belongs to
        self.domain: str = origin_data.get(self.DOMAIN, "")


class RangeMarkerData:
    def __init__(self):
        # Activity record kind, must be MSPTI_ACTIVITY_KIND_MARKER
        self.kind: MsptiActivityKind = MsptiActivityKind.MSPTI_ACTIVITY_KIND_MARKER
        # Source kinds of mark data
        self.source_kind: MsptiActivitySourceKind = MsptiActivitySourceKind.MSPTI_ACTIVITY_SOURCE_KIND_HOST
        # Marker id
        self.id: int = 0
        # Identifier for the activity object associated with the marker
        self.object_id: MsptiObjectId = MsptiObjectId({})
        # Marker name for an instantaneous or start marker
        self.name: str = ""
        # Name of the domain to which this marker belongs to
        self.domain: str = ""
        # range start timestamp(ns)
        self.start: int = 0
        # range end timestamp(ns)
        self.end: int = 0


class HcclData:
    KIND = "kind"
    START = "start"
    END = "end"
    DEVICE_ID = "deviceId"
    STREAM_ID = "streamId"
    BANDWIDTH = "bandWidth"
    NAME = "name"
    COMMNAME = "commName"

    def __init__(self, origin_data: dict):
        # Activity record kind, must be MSPTI_ACTIVITY_KIND_HCCL
        self.kind: MsptiActivityKind = MsptiActivityKind(origin_data.get(self.KIND, 0))
        # Start timestamp for the hccl, in ns
        self.start: int = origin_data.get(self.START, 0)
        # End timestamp for the hccl, in ns
        self.end: int = origin_data.get(self.END, 0)
        # device_id
        self.device_id: int = origin_data.get(self.DEVICE_ID, 0)
        # stream_id
        self.stream_id: int = origin_data.get(self.STREAM_ID, 0)
        # bandwidth for hccl, in GB/S
        self.bandwidth: float = origin_data.get(self.BANDWIDTH, 0)
        # name
        self.name: str = origin_data.get(self.NAME, "")
        # commName
        self.comm_name: str = origin_data.get(self.COMMNAME, "")