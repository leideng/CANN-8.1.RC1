#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

from abc import ABC

from common_func.info_conf_reader import InfoConfReader
from common_func.ms_constant.number_constant import NumberConstant
from common_func.trace_view_header_constant import TraceViewHeaderConstant
from common_func.trace_view_manager import TraceViewManager
from msmodel.stars.sio_model import SioModel
from viewer.interface.base_viewer import BaseViewer


class SioViewer(BaseViewer, ABC):
    """
    class for get sio data
    """
    DATA_TYPE = 'SIO'
    ARGS_INDEX = 4
    # 标号为在sio.db中的列索引
    ACC_ID = 0
    REQ_RX = 1
    TIME_STAMP = 9
    BANDWIDTH_TYPE = ["req_rx", "rsp_rx", "snp_rx", "dat_rx", "req_tx", "rsp_tx", "snp_tx", "dat_tx"]
    ACC_ID_KEY = {
        0: "die 0",
        1: "die 1",
    }

    def __init__(self: any, configs: dict, params: dict) -> None:
        super().__init__(configs, params)
        self.model_list = {
            'sio': SioModel
        }
        self.pid = InfoConfReader().get_json_pid_data()
        self.tid = InfoConfReader().get_json_tid_data()

    def get_trace_timeline(self: any, datas: list) -> list:
        """
        format data to standard timeline format
        :return: list
        """
        if not datas:
            return []
        timestamp_dict = {}
        event_dict = {}
        for data in datas:
            self._compute_bandwidth(data, event_dict, timestamp_dict)
        result = list(event_dict.values())
        _trace = TraceViewManager.column_graph_trace(TraceViewHeaderConstant.COLUMN_GRAPH_HEAD_LEAST, result)
        result = TraceViewManager.metadata_event([["process_name", self.pid, self.tid, self.DATA_TYPE]])
        result.extend(_trace)
        return result

    def _compute_bandwidth(self, data: list, event_dict: dict, timestamp_dict: dict):
        acc_id = data[self.ACC_ID]
        timestamp = data[self.TIME_STAMP]
        if acc_id in timestamp_dict and timestamp_dict[acc_id] != timestamp:
            data_size_list = data[self.REQ_RX: self.TIME_STAMP]
            bandwidth_data = [data_size / (NumberConstant.BYTES_TO_KB ** 2) / (
                (timestamp - timestamp_dict[acc_id]) / NumberConstant.NANO_SECOND) for data_size in data_size_list]
            local_time = InfoConfReader().trans_into_local_time(raw_timestamp=data[self.TIME_STAMP], use_us=True)
            for key, value in zip(self.BANDWIDTH_TYPE, bandwidth_data):
                tmp_key = "{}_{}".format(key, local_time)
                if tmp_key not in event_dict:
                    event_dict[tmp_key] = [key, local_time, self.pid, self.tid, {self.ACC_ID_KEY.get(acc_id): value}]
                else:
                    event_dict.get(tmp_key)[self.ARGS_INDEX][self.ACC_ID_KEY.get(acc_id)] = value
        timestamp_dict[acc_id] = timestamp
