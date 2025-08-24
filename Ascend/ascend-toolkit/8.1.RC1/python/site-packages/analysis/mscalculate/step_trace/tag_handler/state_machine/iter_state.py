#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

from common_func.step_trace_constant import StepTraceConstant


class BaseTagState:
    """
    states of IndexTagHandler
    """

    def __init__(self: any, handler: any) -> None:
        self.handler = handler
        self.tag_to_func = {
            StepTraceConstant.MODEL_START_TAG: self.process_model_start_tag,
            StepTraceConstant.MODEL_END_TAG: self.process_model_end_tag,
            StepTraceConstant.ITER_END_TAG: self.process_iter_end_tag
        }

    @staticmethod
    def do_nothing() -> None:
        """
        do nothing
        :return: void
        """

    def process_record(self: any, record: any) -> None:
        """
        the way to process record
        :param record: contain model_id, tag_id, timestamp
        :return: void
        """
        tag_id = record.get(StepTraceConstant.TAG_ID)
        if tag_id in self.tag_to_func:
            self.tag_to_func.get(tag_id)(record.get(StepTraceConstant.TIME_STAMP), tag_id)

    def start_iter(self: any, time: any) -> None:
        """
        start iter
        :param time: timestamp of record
        :return: void
        """
        self.handler.next_handler_group[self.handler.index_id].clear()
        self.handler.collect_data[self.handler.index_id][StepTraceConstant.STEP_START] = time

    def finish_iter(self: any, time: any) -> None:
        """
        finish iter
        :param time: timestamp of record
        :return: void
        """
        self.handler.collect_data.get(self.handler.index_id)[StepTraceConstant.STEP_END] = time
        self.handler.index_id += 1

    def process_model_start_tag(self: any, time: int, tag_id: int) -> tuple:
        """
        process model start tag
        :param time: timestamp of record
        :param tag_id: tag id
        :return: void
        """
        self.do_nothing()
        return time, tag_id

    def process_model_end_tag(self: any, time: int, tag_id: int) -> tuple:
        """
        process model end tag
        :param time: timestamp of record
        :param tag_id: tag id
        :return: void
        """
        self.do_nothing()
        return time, tag_id

    def process_iter_end_tag(self: any, time: int, tag_id: int) -> tuple:
        """
        process iter end tag
        :param time: timestamp of record
        :param tag_id: tag id
        :return: void
        """
        self.do_nothing()
        return time, tag_id


class ModelStart(BaseTagState):
    """
    state of IndexTagHandler receiving model start tag
    """
    def process_model_end_tag(self: any, time: int, tag_id: int) -> None:
        self.finish_iter(time)
        self.handler.cur_state = self.handler.step_state.get(tag_id)

    def process_iter_end_tag(self: any, time: int, tag_id: int) -> None:
        self.finish_iter(time)
        self.start_iter(time)
        self.handler.cur_state = self.handler.step_state.get(tag_id)


class ModelEnd(BaseTagState):
    """
    state of IndexTagHandler receiving model end tag
    """
    def process_model_start_tag(self: any, time: int, tag_id: int) -> None:
        self.start_iter(time)
        self.handler.cur_state = self.handler.step_state.get(tag_id)


class IterEnd(BaseTagState):
    """
    state of IndexTagHandler receiving iter end tag
    """
    def process_iter_end_tag(self: any, time: int, tag_id: int) -> None:
        self.finish_iter(time)
        self.start_iter(time)
        self.handler.cur_state = self.handler.step_state.get(tag_id)

    def process_model_start_tag(self: any, time: int, tag_id: int) -> None:
        self.start_iter(time)
        self.handler.cur_state = self.handler.step_state.get(tag_id)
