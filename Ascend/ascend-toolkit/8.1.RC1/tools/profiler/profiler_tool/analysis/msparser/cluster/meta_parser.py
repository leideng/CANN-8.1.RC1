#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import logging
from abc import abstractmethod
from collections import defaultdict

from common_func.constant import Constant
from common_func.ms_constant.number_constant import NumberConstant
from common_func.ms_constant.str_constant import OpAnalysisType
from common_func.ms_constant.str_constant import OpBandWidthType
from common_func.ms_constant.str_constant import StrConstant
from common_func.ms_constant.str_constant import TransportType
from common_func.platform.chip_manager import ChipManager
from profiling_bean.prof_enum.chip_model import ChipModel


class MetaParser:
    """
    abstract class for cluster communication and optimization suggestion
    """
    @abstractmethod
    def run(self):
        self.parse()

    @abstractmethod
    def parse(self):
        return


class HcclAnalysisTool:
    """
    support hccl parse
    """
    StandardBandWidth = {
        ChipModel.CHIP_V2_1_0: {
            StrConstant.RDMA: NumberConstant.RDMA_BANDWIDTH_V2_1_0,
            StrConstant.HCCS: NumberConstant.HCCS_BANDWIDTH_V2_1_0,
            StrConstant.PCIE: NumberConstant.PCIE_BANDWIDTH_V2_1_0
        },
        ChipModel.CHIP_V4_1_0: {
            StrConstant.RDMA: NumberConstant.RDMA_BANDWIDTH_V4_1_0,
            StrConstant.HCCS: NumberConstant.HCCS_BANDWIDTH_V4_1_0,
        }
    }

    MessageSizeThreshold = {
        StrConstant.RDMA: NumberConstant.RDMA_MESSAGE_SIZE_THRESHOLD,
        StrConstant.HCCS: NumberConstant.HCCS_MESSAGE_SIZE_THRESHOLD,
        StrConstant.PCIE: NumberConstant.PCIE_MESSAGE_SIZE_THRESHOLD,
        StrConstant.SIO: NumberConstant.SIO_MESSAGE_SIZE_THRESHOLD,
    }

    @classmethod
    def get_standard_bandwidth(cls):
        return cls.StandardBandWidth.get(ChipManager().get_chip_id(), {})

    @classmethod
    def get_value(cls: any, value: any, value_msg: str) -> float:
        if isinstance(value, int) or isinstance(value, float):
            return value
        if isinstance(value, str):
            logging.warning('%s is a string, not a int or float, please check', value_msg)
        if value is None:
            logging.warning('%s is a None value, please check', value_msg)
        return 0

    @classmethod
    def determine_rdma(cls: any, events: list, idx: int, rdma_transit_op_num: int) -> bool:
        if idx > len(events) - rdma_transit_op_num:
            return False
        second_task_type = events[idx + 1].hccl_name
        third_task_type = events[idx + 2].hccl_name
        if second_task_type == StrConstant.RDMA_SEND and third_task_type == StrConstant.NOTIFY_WAIT:
            return True
        else:
            return False

    @classmethod
    def get_rdma_time_info(cls: any, events: list, idx: int, rdma_transit_op_num: int) -> list:
        transit_size = HcclAnalysisTool.get_value(events[idx].size, 'size') / NumberConstant.COMMUNICATION_B_to_MB
        transit_time = HcclAnalysisTool.get_value(events[idx + rdma_transit_op_num - 1].duration +
                                                  events[idx + rdma_transit_op_num - 1].timestamp -
                                                  events[idx].timestamp,
                                                  'duration') / NumberConstant.NS_TO_MS
        return [transit_time, transit_size]

    @classmethod
    def find_consecutive_payload_tasks_count(cls: any, events: list, idx: int) -> int:
        count = 0
        while idx < len(events) and events[idx].rdma_type == 'RDMA_SEND_PAYLOAD':
            idx += 1
            count += 1
        return count

    @classmethod
    def calculate_consecutive_payload_tasks_info(cls: any, events: list, idx: int, payload_cnt: int, idx_jump: int):
        if (idx + payload_cnt + idx_jump - 2) >= len(events):
            op_name = events[idx].op_name
            logging.warning("Bandwidth calculation abnormal. Index out of range, missing closure tasks. op_name:%s",
                            op_name)
            return []
        saved_size = 0
        first_payload_time = events[idx].timestamp
        for i in range(idx, idx + payload_cnt):
            saved_size += events[i].size
        transit_size = saved_size / NumberConstant.COMMUNICATION_B_to_MB
        transit_time = HcclAnalysisTool.get_value(events[idx + payload_cnt + idx_jump - 2].duration +
                                                  events[idx + payload_cnt + idx_jump - 2].timestamp -
                                                  first_payload_time, 'duration') / NumberConstant.NS_TO_MS
        return [transit_time, transit_size]

    @classmethod
    def is_send_or_recv_op(cls, events: list, idx: int) -> bool:
        return 'send' in events[idx].op_name.lower() or 'receive' in events[idx].op_name.lower()

    @classmethod
    def init_dict(cls: any, keys: list) -> dict:
        return {key: 0 for key in keys}

    @classmethod
    def init_bandwidth_dict(cls) -> dict:
        dic = dict()
        # get public variables from OpAnalysisType
        values = [value for key, value in OpBandWidthType.__dict__.items() if '__' not in key]
        for trans_type in StrConstant.TRANSIT_TYPE:
            dic[trans_type] = HcclAnalysisTool.init_dict(values)
            dic[trans_type][OpBandWidthType.SIZE_DISTRIBUTION] = defaultdict(lambda: [0, 0])
        return dic

    @classmethod
    def update_time_ratio(cls: any, op_time_dict: dict, op_name: str) -> None:
        try:
            op_time_dict[OpAnalysisType.WAIT_TIME_RATIO] = \
                round(op_time_dict.get(OpAnalysisType.WAIT_TIME) /
                      (op_time_dict.get(OpAnalysisType.WAIT_TIME) + op_time_dict.get(OpAnalysisType.TRANSIT_TIME)), 4)
        except ZeroDivisionError as err:
            logging.warning('%s Transit Time and Wait Time is 0 %s', op_name, err)
        try:
            op_time_dict[OpAnalysisType.SYNCHRONIZATION_TIME_RATIO] = \
                round(op_time_dict.get(OpAnalysisType.SYNCHRONIZATION_TIME)
                      / (op_time_dict.get(OpAnalysisType.SYNCHRONIZATION_TIME) +
                         op_time_dict.get(OpAnalysisType.TRANSIT_TIME)), 4)
        except ZeroDivisionError as err:
            logging.warning('%s Transit Time and Synchronization Time Time is 0 %s', op_name, err)

    @classmethod
    def update_bandwidth_record(cls: any, bandwidth_dict: dict, bandwidth_info_type: str, size: float,
                                dur: float) -> None:
        bandwidth_dict[bandwidth_info_type][OpBandWidthType.TRANSIT_SIZE_MB] += size
        bandwidth_dict[bandwidth_info_type][OpBandWidthType.TRANSIT_TIME_MS] += dur
        bandwidth_dict[bandwidth_info_type][OpBandWidthType.SIZE_DISTRIBUTION][size][0] += 1
        bandwidth_dict[bandwidth_info_type][OpBandWidthType.SIZE_DISTRIBUTION][size][1] += dur

    @classmethod
    def combine_sdma_info(cls: any, bandwidth_dict: dict) -> None:
        bandwidth_dict[StrConstant.SDMA][OpBandWidthType.TRANSIT_SIZE_MB] += \
            bandwidth_dict[StrConstant.HCCS][OpBandWidthType.TRANSIT_SIZE_MB] + \
            bandwidth_dict[StrConstant.PCIE][OpBandWidthType.TRANSIT_SIZE_MB] + \
            bandwidth_dict[StrConstant.SIO][OpBandWidthType.TRANSIT_SIZE_MB]
        bandwidth_dict[StrConstant.SDMA][OpBandWidthType.TRANSIT_TIME_MS] += \
            bandwidth_dict[StrConstant.HCCS][OpBandWidthType.TRANSIT_TIME_MS] + \
            bandwidth_dict[StrConstant.PCIE][OpBandWidthType.TRANSIT_TIME_MS] + \
            bandwidth_dict[StrConstant.SIO][OpBandWidthType.TRANSIT_TIME_MS]
        if bandwidth_dict[StrConstant.SDMA][OpBandWidthType.TRANSIT_TIME_MS] != 0:
            bandwidth_dict[StrConstant.SDMA][OpBandWidthType.BANDWIDTH_GB_S] = round(
                (bandwidth_dict[StrConstant.SDMA][OpBandWidthType.TRANSIT_SIZE_MB] /
                 NumberConstant.COMMUNICATION_MB_to_GB) /
                (bandwidth_dict[StrConstant.SDMA][OpBandWidthType.TRANSIT_TIME_MS] / NumberConstant.CONVERSION_TIME), 4
            )

    @classmethod
    def analyze_bandwidth_info(cls: any, bandwidth_dict: dict, transport_type: str) -> None:
        if bandwidth_dict[transport_type][OpBandWidthType.TRANSIT_TIME_MS] != 0:
            bandwidth_dict[transport_type][OpBandWidthType.BANDWIDTH_GB_S] = round(
                (bandwidth_dict[transport_type][OpBandWidthType.TRANSIT_SIZE_MB] /
                 NumberConstant.COMMUNICATION_MB_to_GB) /
                (bandwidth_dict[transport_type][OpBandWidthType.TRANSIT_TIME_MS] / NumberConstant.CONVERSION_TIME), 4
            )
        bandwidth_dict[transport_type][OpBandWidthType.BANDWIDTH_UTILIZATION] = round(
            bandwidth_dict[transport_type][OpBandWidthType.BANDWIDTH_GB_S] /
            cls.get_standard_bandwidth().get(transport_type, -1), 4
        )
        packet_num = 0
        large_packet_num = 0
        for size, size_info in bandwidth_dict[transport_type][OpBandWidthType.SIZE_DISTRIBUTION].items():
            if size > cls.MessageSizeThreshold.get(transport_type, 0):
                large_packet_num += size_info[0]
            packet_num += size_info[0]
        if packet_num > 0:
            bandwidth_dict[transport_type][OpBandWidthType.LARGE_PACKET_RATIO] = \
                round(large_packet_num / packet_num, 4)

    @classmethod
    def is_valid_link(cls: any, event) -> bool:
        src_rank_valid = event.local_rank is not None and event.local_rank != int("0xffffffff", 16)
        dst_rank_valid = event.remote_rank is not None
        if src_rank_valid and dst_rank_valid:
            return True
        else:
            return False

    @classmethod
    def divide(cls: any, dividend: float, divisor: float):
        try:
            quotient = round(dividend / divisor, 4)
        except ZeroDivisionError as err:
            logging.error(str(err), exc_info=Constant.TRACE_BACK_SWITCH)
            return 0
        return quotient

    @classmethod
    def convert_to_enum(cls: any, trans_type: str) -> int:
        if trans_type == StrConstant.HCCS or trans_type == StrConstant.HCCS_SW:
            return TransportType.HCCS
        if trans_type == StrConstant.PCIE:
            return TransportType.PCIE
        if trans_type == StrConstant.RDMA:
            return TransportType.RDMA
        if trans_type == StrConstant.LOCAL:
            return TransportType.LOCAL
        if trans_type == StrConstant.SIO:
            return TransportType.SIO
        logging.warning("trans_type is not normal, which is %s", trans_type)
        return -1

    @classmethod
    def convert_to_str(cls: any, trans_data_type: int) -> str:
        if trans_data_type == TransportType.HCCS:
            return StrConstant.HCCS
        if trans_data_type == TransportType.PCIE:
            return StrConstant.PCIE
        if trans_data_type == TransportType.RDMA:
            return StrConstant.RDMA
        if trans_data_type == TransportType.LOCAL:
            return StrConstant.LOCAL
        if trans_data_type == TransportType.SIO:
            return StrConstant.SIO
        logging.warning("trans_data_type is not normal, which is %d", trans_data_type)
        return 'Unknown transport type'
