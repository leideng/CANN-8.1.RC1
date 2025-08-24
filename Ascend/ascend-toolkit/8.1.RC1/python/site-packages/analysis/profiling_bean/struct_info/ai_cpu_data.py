#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

import logging
import struct

from common_func.info_conf_reader import InfoConfReader
from common_func.ms_constant.number_constant import NumberConstant
from common_func.utils import Utils
from msparser.data_struct_size_constant import StructFmt
from profiling_bean.struct_info.struct_decoder import StructDecoder


class AiCpuTimeConsuming:
    """
    time consuming for ai cpu
    """

    def __init__(self: any, *args: any) -> None:
        ai_cpu_data = args[0]
        self._ai_cpu_task_start = ai_cpu_data[1]
        self._compute_time = (ai_cpu_data[3] - ai_cpu_data[2]) / NumberConstant.MILLI_SECOND
        self._mem_copy_time = (ai_cpu_data[4] - ai_cpu_data[3]) / NumberConstant.MILLI_SECOND
        self._ai_cpu_task_end = ai_cpu_data[6]
        self._ai_cpu_task_time = ai_cpu_data[6] - ai_cpu_data[1]
        self._dispatch_time = ai_cpu_data[14] / NumberConstant.MILLI_SECOND
        self._total_time = ai_cpu_data[15] / NumberConstant.MILLI_SECOND

    @property
    def compute_time(self: any) -> float:
        """
        ai cpu compute time
        :return: ai cpu compute time
        """
        return self._compute_time

    @property
    def memory_copy_time(self: any) -> float:
        """
        ai cpu memory copy time
        :return: ai cpu memory copy time
        """
        return self._mem_copy_time

    @property
    def ai_cpu_task_start_time(self: any) -> any:
        """
        ai cpu task start time
        :return: ai cpu task start time
        """
        if self._ai_cpu_task_start != 0:
            return InfoConfReader().time_from_syscnt(self._ai_cpu_task_start, NumberConstant.MILLI_SECOND)
        return 0

    @property
    def ai_cpu_task_start_syscnt(self: any) -> any:
        """
        ai cpu task start time
        :return: ai cpu task start time
        """
        return self._ai_cpu_task_start

    @property
    def ai_cpu_task_end_time(self: any) -> any:
        """
        ai cpu task end time
        :return: ai cpu task end time
        """
        if self._ai_cpu_task_end != 0:
            return InfoConfReader().time_from_syscnt(self._ai_cpu_task_end, NumberConstant.MILLI_SECOND)
        return 0

    @property
    def ai_cpu_task_end_syscnt(self: any) -> any:
        """
        ai cpu task end time
        :return: ai cpu task end time
        """
        return self._ai_cpu_task_end

    @property
    def ai_cpu_task_time(self: any) -> float:
        """
        ai cpu task time
        :return: ai cpu task time
        """
        return self.ai_cpu_task_end_time - self.ai_cpu_task_start_time

    @property
    def dispatch_time(self: any) -> float:
        """
        ai cpu dispatch time
        :return: ai cpu dispatch time
        """
        return self._dispatch_time

    @property
    def total_time(self: any) -> float:
        """
        ai cpu total time
        :return: ai cpu total time
        """
        return self._total_time


class AiCpuData(StructDecoder):
    """
    struct for ai cpu
    """

    AI_CPU_DATA_TAG = 60

    def __init__(self: any) -> None:
        self._stream_id = None
        self._task_id = None
        self._ai_cpu_time_consuming = None

    @property
    def stream_id(self: any) -> any:
        """
        stream id for ai cpu
        :return: stream id for ai cpu
        """
        return self._stream_id

    @property
    def task_id(self: any) -> any:
        """
        task id for ai cpu
        :return: task id for ai cpu
        """
        return self._task_id

    @property
    def ai_cpu_time_consuming(self: any) -> AiCpuTimeConsuming:
        """
        bean time for ai cpu
        :return: bean time for ai cpu
        """
        return self._ai_cpu_time_consuming

    def ai_cpu_decode(self: any, bin_data: any) -> any:
        """
        decode the aicpu bin data
        :param bin_data: aicpu bin data
        :return: instance of aicpu
        """
        if self.construct_bean(struct.unpack(StructFmt.AI_CPU_FMT, bin_data)):
            return self
        return {}

    def construct_bean(self: any, *args: dict) -> bool:
        """
        refresh the aicpu data
        :param args: aicpu bin data
        :return: True or False
        """
        _ai_cpu_data = args[0]
        _magic_num, _data_tag = _ai_cpu_data[:2]
        if _magic_num == NumberConstant.MAGIC_NUM and _data_tag == self.AI_CPU_DATA_TAG:
            self._stream_id = Utils.get_stream_id(_ai_cpu_data[2])
            self._task_id = str(_ai_cpu_data[3])
            self._ai_cpu_time_consuming = AiCpuTimeConsuming(_ai_cpu_data[4:])
            return True
        logging.error("AICPU data struct is incomplete: %s, please check the AICPU file.", hex(_magic_num))
        return False
