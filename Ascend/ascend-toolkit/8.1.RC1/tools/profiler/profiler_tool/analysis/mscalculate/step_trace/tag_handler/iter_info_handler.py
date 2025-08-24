#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

from collections import defaultdict

from common_func.step_trace_constant import StepTraceConstant
from mscalculate.interface.step_trace_tag_handler import StepTraceTagHandler


class AllReduceStreamHandler(StepTraceTagHandler):
    """
    get all reduce data
    """
    def __init__(self: any) -> None:
        self.collect_data = []
        self.next_handler_group = defaultdict(AllReduceTagHandler)

    def receive_record(self: any, record: dict) -> None:
        """
        receive record of step trace
        :param record: contain model_id, tag_id, timestamp, stream id
        :return: void
        """
        stream_id = record.get(StepTraceConstant.STREAM_ID)
        self.next_handler_group[stream_id].receive_record(record)

    def get_data(self: any) -> list:
        """
        return data of this handler
        :return: list
        """
        self.collect_data = []
        for next_handler in self.next_handler_group.values():
            self.collect_data.extend(next_handler.get_data())
        return self.collect_data

    def clear(self: any) -> None:
        """
        clear next handler
        :return: void
        """
        self.next_handler_group.clear()


class AllReduceTagHandler(StepTraceTagHandler):
    """
    get all reduce data
    """
    def __init__(self: any) -> None:
        self.collect_data = []

    def receive_record(self: any, record: dict) -> None:
        """
        receive record of step trace
        :param record: contain model_id, tag_id, timestamp
        :return: void
        """
        self.process_record(record)

    def get_data(self: any) -> list:
        """
        return data of this handler
        :return: dict
        """
        return self.collect_data

    def process_record(self: any, record: dict) -> None:
        """
        get reduce start, reduce end from record
        :param record: contain model_id, tag_id, timestamp
        :return: void
        """
        if not record[StepTraceConstant.TAG_ID] % 2:
            self.collect_data.append(
                {StepTraceConstant.REDUCE_START: record[StepTraceConstant.TIME_STAMP],
                 StepTraceConstant.REDUCE_END: None})
        if record[StepTraceConstant.TAG_ID] % 2 and self.collect_data:
            self.collect_data[-1][StepTraceConstant.REDUCE_END] = record[StepTraceConstant.TIME_STAMP]

    def clear(self: any) -> None:
        """
        clear collect data
        :return: void
        """
        self.collect_data.clear()


class TrainingTraceTagHandler(StepTraceTagHandler):
    """
    get training trace data
    """
    def __init__(self: any) -> None:
        self.collect_data = {}

    def receive_record(self: any, record: dict) -> None:
        """
        receive record of step trace
        :param record: contain model_id, tag_id, timestamp
        :return: void
        """
        self.process_record(record)

    def get_data(self: any) -> list:
        """
        get reduce start, reduce end from record
        :param record: contain model_id, tag_id, timestamp
        :return: void
        """
        return self.collect_data

    def process_record(self: any, record: dict) -> None:
        """
        get bp, fp from record
        :param record: contain model_id, tag_id, timestamp
        :return: void
        """
        user_set_flag = False
        if record[StepTraceConstant.TAG_ID] == StepTraceConstant.FP_TAG:
            user_set_flag = True
            self.collect_data[StepTraceConstant.FORWARD_PROPAGATION] = record.get(StepTraceConstant.TIME_STAMP)

        if record[StepTraceConstant.TAG_ID] == StepTraceConstant.BP_TAG:
            self.collect_data[StepTraceConstant.BACK_PROPAGATION] = record.get(StepTraceConstant.TIME_STAMP)

    def clear(self: any) -> None:
        """
        clear collect data
        :return: void
        """
        self.collect_data.clear()


class GetNextTagHandler(StepTraceTagHandler):
    """
    get_next tag data handler
    """
    def __init__(self: any) -> None:
        self.collect_data = {}

    def receive_record(self: any, record: dict) -> None:
        """
        receive record of step trace
        :param record: contain model_id, tag_id, timestamp
        :return: void
        """
        self.process_record(record)

    def get_data(self: any) -> dict:
        """
        return data of this handler
        :return: dict
        """
        return self.collect_data

    def process_record(self: any, record: dict) -> None:
        """
        get get_next start, get_next end from record
        :param record: contain model_id, tag_id, timestamp
        :return: void
        """
        # contineous 2 tags represent getnext start and end
        self.collect_data.setdefault(record[StepTraceConstant.TAG_ID] // 2, []).append(record)

    def clear(self: any) -> None:
        """
        clear collect data
        :return: void
        """
        self.collect_data.clear()
