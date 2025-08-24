#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

from abc import ABC

from common_func.info_conf_reader import InfoConfReader
from common_func.trace_view_header_constant import TraceViewHeaderConstant
from common_func.trace_view_manager import TraceViewManager
from msmodel.stars.inter_soc_model import InterSocModel
from msmodel.stars.lowpower_model import LowPowerModel
from viewer.interface.base_viewer import BaseViewer


class LowPowerViewer(BaseViewer, ABC):
    """
    class for get lowpower samplen data
    """

    DATA_TYPE = 'data_type'
    TIME_STAMP = 0
    VF_SAMPLEING_TIMES = 1
    PWR_SAMPLEING_TIMES = 2
    TEMP_SAMPLEING_TIMES = 3
    TEM_OF_AI_CORE = 4
    TEM_OF_HBM = 5
    TEM_OF_HBM_GRANULARITY = 6
    TEM_OF_CPU = 8
    TEM_OF_3DSRAM = 7
    TEM_OF_PERIPHERALS = 9
    TEM_OF_L2_BUFF = 10
    AIC_CURRENT_DPM = 11
    POWER_COS_DPM = 12
    AIC_CURRENT_SD5003 = 13
    POWER_COS_SD5003 = 14
    AIC_FREQUENCY = 16
    IMON = 15
    WARN_CNT0 = 17
    WARN_CNT1 = 18
    WARN_CNT2 = 19
    WARN_CNT3 = 20
    VOLTAGE = 31

    def __init__(self: any, configs: dict, params: dict) -> None:
        super().__init__(configs, params)
        self.pid = 0
        self.model_list = {
            'inter_soc_time': InterSocModel,
            'inter_soc_transmission': InterSocModel,
            'low_power': LowPowerModel,
        }

    def get_timeline_header(self: any) -> list:
        """
        get timeline trace header
        :return: list
        """
        low_power_header = [
            [
                "process_name", self.pid,
                InfoConfReader().get_json_tid_data(), self.params.get(self.DATA_TYPE)
            ]
        ]
        return low_power_header

    def get_trace_timeline(self: any, datas: list) -> list:
        """
        format data to standard timeline format
        :return: list
        """
        if not datas:
            return []
        result = []
        for data in datas:
            self.update_trace_data1(data, result)
            self.update_trace_data2(data, result)
        _trace = TraceViewManager.column_graph_trace(TraceViewHeaderConstant.COLUMN_GRAPH_HEAD_LEAST, result)
        result = TraceViewManager.metadata_event(self.get_timeline_header())
        result.extend(_trace)
        return result

    def update_trace_data1(self: any, data: dict, result: list):
        result.append(["VF Sampling cnt", InfoConfReader().trans_into_local_time(data[self.TIME_STAMP],
                                                                                 use_us=True),
                       self.pid, self.VF_SAMPLEING_TIMES, {'Value': data[self.VF_SAMPLEING_TIMES]}])
        result.append(["Pwr Sampling cnt", InfoConfReader().trans_into_local_time(data[self.TIME_STAMP],
                                                                                  use_us=True),
                       self.pid, self.PWR_SAMPLEING_TIMES, {'Value': data[self.PWR_SAMPLEING_TIMES]}])
        result.append(["Temp Sampling cnt", InfoConfReader().trans_into_local_time(data[self.TIME_STAMP],
                                                                                   use_us=True),
                       self.pid, self.TEMP_SAMPLEING_TIMES, {'Value': data[self.TEMP_SAMPLEING_TIMES]}])
        result.append(["AIC TEMP (℃)", InfoConfReader().trans_into_local_time(data[self.TIME_STAMP],
                                                                              use_us=True),
                       self.pid, self.TEM_OF_AI_CORE, {'Value': data[self.TEM_OF_AI_CORE]}])
        result.append(
            ["HBM Controller TEMP (℃)", InfoConfReader().trans_into_local_time(data[self.TIME_STAMP],
                                                                               use_us=True),
             self.pid, self.TEM_OF_HBM, {'Value': data[self.TEM_OF_HBM]}])
        result.append(["HBM TEMP (℃)", InfoConfReader().trans_into_local_time(data[self.TIME_STAMP],
                                                                              use_us=True),
                       self.pid, self.TEM_OF_HBM, {'Value': data[self.TEM_OF_HBM_GRANULARITY]}])
        result.append(
            ["CPU TEMP (℃)", InfoConfReader().trans_into_local_time(data[self.TIME_STAMP], use_us=True),
             self.pid, self.TEM_OF_CPU, {'Value': data[self.TEM_OF_CPU]}])
        result.append(["Peripherals TEMP (℃)", InfoConfReader().trans_into_local_time(data[self.TIME_STAMP],
                                                                                      use_us=True),
                       self.pid, self.TEM_OF_PERIPHERALS, {'Value': data[self.TEM_OF_PERIPHERALS]}])
        result.append(["L2 Buffer TEMP (℃)", InfoConfReader().trans_into_local_time(data[self.TIME_STAMP],
                                                                                    use_us=True),
                       self.pid, self.TEM_OF_L2_BUFF, {'Value': data[self.TEM_OF_L2_BUFF]}])
        result.append(["DPM AIC Current (A)", InfoConfReader().trans_into_local_time(data[self.TIME_STAMP],
                                                                                     use_us=True),
                       self.pid, self.AIC_CURRENT_DPM, {'Value': data[self.AIC_CURRENT_DPM]}])
        
    def update_trace_data2(self: any, data: dict, result: list):
        result.append(["DPM Power (W)", InfoConfReader().trans_into_local_time(data[self.TIME_STAMP],
                                                                               use_us=True),
                       self.pid, self.POWER_COS_DPM, {'Value': data[self.POWER_COS_DPM]}])
        result.append(["AIC Current (A)", InfoConfReader().trans_into_local_time(data[self.TIME_STAMP],
                                                                                 use_us=True),
                       self.pid, self.AIC_CURRENT_SD5003, {'Value': data[self.AIC_CURRENT_SD5003]}])
        result.append(["AIC Power (W)", InfoConfReader().trans_into_local_time(data[self.TIME_STAMP],
                                                                               use_us=True),
                       self.pid, self.POWER_COS_SD5003, {'Value': data[self.POWER_COS_SD5003]}])
        result.append(["AIC Voltage (V)", InfoConfReader().trans_into_local_time(data[self.TIME_STAMP],
                                                                                 use_us=True),
                       self.pid, self.VOLTAGE, {'Value': data[self.VOLTAGE]}])
        result.append(["AIC Frequency (MHz)", InfoConfReader().trans_into_local_time(data[self.TIME_STAMP],
                                                                                     use_us=True),
                       self.pid, self.AIC_FREQUENCY, {'Value': data[self.AIC_FREQUENCY]}])
        result.append(["Imon", InfoConfReader().trans_into_local_time(data[self.TIME_STAMP], use_us=True),
                       self.pid, self.IMON, {'Value': data[self.IMON]}])
        result.append(["TEMP Warning 0", InfoConfReader().trans_into_local_time(data[self.TIME_STAMP],
                                                                                use_us=True),
                       self.pid, self.WARN_CNT0, {'Value': data[self.WARN_CNT0]}])
        result.append(["TEMP Warning 1", InfoConfReader().trans_into_local_time(data[self.TIME_STAMP],
                                                                                use_us=True),
                       self.pid, self.WARN_CNT1, {'Value': data[self.WARN_CNT1]}])
        result.append(["TEMP Warning 2", InfoConfReader().trans_into_local_time(data[self.TIME_STAMP],
                                                                                use_us=True),
                       self.pid, self.WARN_CNT2, {'Value': data[self.WARN_CNT2]}])
        result.append(["TEMP Warning 3", InfoConfReader().trans_into_local_time(data[self.TIME_STAMP],
                                                                                use_us=True),
                       self.pid, self.WARN_CNT3, {'Value': data[self.WARN_CNT3]}])
