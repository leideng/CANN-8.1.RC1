#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

from abc import ABC

from common_func.info_conf_reader import InfoConfReader
from common_func.trace_view_header_constant import TraceViewHeaderConstant
from common_func.trace_view_manager import TraceViewManager
from msmodel.stars.acc_pmu_model import AccPmuModel
from viewer.interface.base_viewer import BaseViewer


class AccPmuViewer(BaseViewer, ABC):
    """
    class for get acc_pmu data
    """

    DATA_TYPE = 'Acc PMU'
    SAMPLE_BASED = 'sample_based'

    def __init__(self: any, configs: dict, params: dict) -> None:
        super().__init__(configs, params)
        self.pid = InfoConfReader().get_json_pid_data()
        self.tid = InfoConfReader().get_json_tid_data()
        self.model_list = {
            'acc_pmu': AccPmuModel,
        }

    def get_timeline_header(self: any) -> list:
        """
        to get chrome trace json header
        """
        acc_header = [
            [
                "process_name",
                self.pid,
                self.tid,
                self.DATA_TYPE
            ]
        ]
        return acc_header

    def get_trace_timeline(self: any, datas: list) -> list:
        """
        to format data to chrome trace json
        """
        if not datas:
            return []
        result = []
        for data in datas:
            local_time = InfoConfReader().trans_into_local_time(raw_timestamp=data.timestamp, use_us=True)
            result.append(["read_bandwidth", local_time,
                           {'value': data.read_bandwidth, 'acc_id': data.acc_id}])
            result.append(["write_bandwidth", local_time,
                           {'value': data.write_bandwidth, 'acc_id': data.acc_id}])
            result.append(["read_ost", local_time,
                           {'value': data.read_ost, 'acc_id': data.acc_id}])
            result.append(["write_ost", local_time,
                           {'value': data.write_ost, 'acc_id': data.acc_id}])
        for item in result:
            item[2:2] = [self.pid, self.tid]
        _trace = TraceViewManager.column_graph_trace(TraceViewHeaderConstant.COLUMN_GRAPH_HEAD_LEAST, result)
        result = TraceViewManager.metadata_event(self.get_timeline_header())
        result.extend(_trace)
        return result
