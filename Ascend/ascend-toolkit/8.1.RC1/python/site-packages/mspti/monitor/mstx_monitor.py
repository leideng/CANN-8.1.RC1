#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

from typing import Callable
from collections import defaultdict
from ..activity_data import MarkerData
from ..activity_data import RangeMarkerData
from ..constant import MsptiResult
from ..utils import print_error_msg
from .base_monitor import BaseMonitor
from ._mspti_c import (
    _mstx_register_cb,
    _mstx_unregister_cb
)
from ..constant import (
    MsptiActivityFlag
)


class MstxMonitor(BaseMonitor):
    def __init__(self):
        super().__init__()
        self.mark_user_cb = None
        self.range_user_cb = None
        self.data_dict = defaultdict(lambda: {'start': None, 'end': None})

    def empty_callback(self):
        pass

    def start(self,
              mark_cb : Callable[[MarkerData], None] = empty_callback,
              range_cb : Callable[[RangeMarkerData], None] = empty_callback) -> MsptiResult:
        if not callable(mark_cb) and not callable(range_cb):
            print_error_msg("Mstx callback is invalid")
            return MsptiResult.MSPTI_ERROR_INVALID_PARAMETER
        ret = BaseMonitor.start_monitor()
        if ret == MsptiResult.MSPTI_SUCCESS:
            self.mark_user_cb = mark_cb
            self.range_user_cb = range_cb
            return MsptiResult(_mstx_register_cb(self.callback))
        return ret

    def stop(self) -> MsptiResult:
        ret = BaseMonitor.stop_monitor()
        if ret == MsptiResult.MSPTI_SUCCESS:
            self.mark_user_cb = None
            self.range_user_cb = None
            return MsptiResult(_mstx_unregister_cb())
        return ret

    def callback(self, origin_data: dict):
        try:
            self.append_data(MarkerData(origin_data))
        except Exception as ex:
            print_error_msg(f"Call mstx callback failed. Exception: {str(ex)}")

    def assemble_start_end(self, start_data: MarkerData, end_data: MarkerData):
        range_mark_data = RangeMarkerData()
        range_mark_data.source_kind = start_data.source_kind
        range_mark_data.id = start_data.id
        range_mark_data.object_id.device_id = start_data.object_id.device_id
        range_mark_data.object_id.stream_id = start_data.object_id.stream_id
        range_mark_data.object_id.process_id = start_data.object_id.process_id
        range_mark_data.object_id.thread_id = start_data.object_id.thread_id
        range_mark_data.name = start_data.name
        range_mark_data.domain = start_data.domain
        range_mark_data.start = start_data.timestamp
        range_mark_data.end = end_data.timestamp
        return range_mark_data

    def append_data(self, mark_data: MarkerData):
        if (mark_data.flag == MsptiActivityFlag.MSPTI_ACTIVITY_FLAG_MARKER_INSTANTANEOUS or
                mark_data.flag == MsptiActivityFlag.MSPTI_ACTIVITY_FLAG_MARKER_INSTANTANEOUS_WITH_DEVICE):
            self.mark_user_cb(mark_data)

        unique_id = (mark_data.id, mark_data.source_kind)
        if (mark_data.flag == MsptiActivityFlag.MSPTI_ACTIVITY_FLAG_MARKER_END or mark_data.flag ==
                MsptiActivityFlag.MSPTI_ACTIVITY_FLAG_MARKER_END_WITH_DEVICE):
            self.data_dict[unique_id]['end'] = mark_data
        if (mark_data.flag == MsptiActivityFlag.MSPTI_ACTIVITY_FLAG_MARKER_START or mark_data.flag ==
                MsptiActivityFlag.MSPTI_ACTIVITY_FLAG_MARKER_START_WITH_DEVICE):
            self.data_dict[unique_id]['start'] = mark_data
        if self.data_dict[unique_id]['end'] is not None and self.data_dict[unique_id]['start'] is not None:
            range_mark_data = self.assemble_start_end(
                self.data_dict[unique_id]['start'], self.data_dict[unique_id]['end'])
            self.range_user_cb(range_mark_data)
            self.data_dict.pop(unique_id)
