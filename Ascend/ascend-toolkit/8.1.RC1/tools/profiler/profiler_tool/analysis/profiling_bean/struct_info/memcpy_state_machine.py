#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

import logging
from abc import abstractmethod

from common_func.info_conf_reader import InfoConfReader
from common_func.memcpy_constant import MemoryCopyConstant
from common_func.ms_constant.number_constant import NumberConstant


class MemcpyRecorder:
    """
    class for reading memory copy
    """

    def __init__(self: any, stream_id: int, task_id: int) -> None:
        self.tag_to_state = {
            MemoryCopyConstant.RECEIVE_TAG: RecieveState(self),
            MemoryCopyConstant.START_TAG: StartState(self),
            MemoryCopyConstant.END_TAG: EndState(self)
        }

        self.stream_id = stream_id
        self.task_id = task_id
        self.each_batch_timestamp = []
        self._current_state = EndState(self)

    def start_new_batch(self: any) -> None:
        """
        start a new batch
        :return: None
        """
        self.each_batch_timestamp.append([MemoryCopyConstant.DEFAULT_TIMESTAMP,
                                          MemoryCopyConstant.DEFAULT_TIMESTAMP,
                                          MemoryCopyConstant.DEFAULT_TIMESTAMP])

    def process_state_tag(self: any, tag: int, timestamp: int) -> None:
        """
        process state tag
        :param tag: recieve tag, start tag and end tag
        :param timestamp: timestamp
        :return: None
        """
        self._current_state.process_state_tag(tag, timestamp)
        self._current_state = self.tag_to_state.get(tag, self._current_state)


class MemcpyState:
    """
    class for reading memory copy state, including recieving, starting, ending
    """

    NEWEST_BATCH_INDEX = -1

    def __init__(self: any, memcpy_recorder: any) -> None:
        self.tag_to_func = {
            MemoryCopyConstant.RECEIVE_TAG: self.process_recieve_tag,
            MemoryCopyConstant.START_TAG: self.process_start_tag,
            MemoryCopyConstant.END_TAG: self.process_end_tag
        }
        self.memcpy_recorder = memcpy_recorder

    def process_state_tag(self: any, tag: int, timestamp: int) -> None:
        """
        process state tag according to tag_to_func
        :param tag: recieve tag, start tag and end tag
        :param timestamp: timestamp
        :return: None
        """
        self.tag_to_func.get(tag, self.process_unknown_tag)(timestamp)

    def process_unknown_tag(self: any, timestamp: int) -> None:
        """
        process other tag
        :param timestamp: timestamp
        """
        logging.warning("The state tag of stream %d task %d timestamp %d is unknown",
                      self.memcpy_recorder.stream_id,
                      self.memcpy_recorder.task_id,
                      timestamp)

    @abstractmethod
    def process_recieve_tag(self: any, timestamp: int) -> None:
        """
        process recieve tag
        :param timestamp: timestamp
        """

    @abstractmethod
    def process_start_tag(self: any, timestamp: int) -> None:
        """
        process start tag
        :param timestamp: timestamp
        """

    @abstractmethod
    def process_end_tag(self: any, timestamp: int) -> None:
        """
        process end tag
        :param timestamp: timestamp
        """


class RecieveState(MemcpyState):
    """
    class for recieving state
    """

    def process_recieve_tag(self: any, timestamp: int) -> None:
        logging.warning("The state tag %d of stream %d task %d is repeating.",
                        MemoryCopyConstant.RECEIVE_TAG,
                        self.memcpy_recorder.stream_id,
                        self.memcpy_recorder.task_id)

    def process_start_tag(self: any, timestamp: int) -> None:
        self.memcpy_recorder.each_batch_timestamp[self.NEWEST_BATCH_INDEX][
            MemoryCopyConstant.STATES_TIMESTAMPS_START_INDEX] = InfoConfReader().time_from_syscnt(
                timestamp, NumberConstant.MICRO_SECOND)

    def process_end_tag(self: any, timestamp: int) -> None:
        logging.warning("Miss state tag %d of stream %d task %d.",
                        MemoryCopyConstant.START_TAG,
                        self.memcpy_recorder.stream_id,
                        self.memcpy_recorder.task_id)
        self.memcpy_recorder.each_batch_timestamp[self.NEWEST_BATCH_INDEX][
            MemoryCopyConstant.STATES_TIMESTAMPS_END_INDEX] = InfoConfReader().time_from_syscnt(
                timestamp, NumberConstant.MICRO_SECOND)


class StartState(MemcpyState):
    """
    class for starting state
    """

    def process_recieve_tag(self: any, timestamp: int) -> None:
        logging.warning("Miss state tag %d of stream %d task %d.",
                        MemoryCopyConstant.END_TAG,
                        self.memcpy_recorder.stream_id,
                        self.memcpy_recorder.task_id)
        self.memcpy_recorder.start_new_batch()
        self.memcpy_recorder.each_batch_timestamp[self.NEWEST_BATCH_INDEX][
            MemoryCopyConstant.STATES_TIMESTAMPS_RECEIVE_INDEX] = InfoConfReader().time_from_syscnt(
                timestamp, NumberConstant.MICRO_SECOND)

    def process_start_tag(self: any, timestamp: int) -> None:
        logging.warning("The state tag %d of stream %d task %d is repeating.",
                        MemoryCopyConstant.START_TAG,
                        self.memcpy_recorder.stream_id,
                        self.memcpy_recorder.task_id)

    def process_end_tag(self: any, timestamp: int) -> None:
        self.memcpy_recorder.each_batch_timestamp[self.NEWEST_BATCH_INDEX][
            MemoryCopyConstant.STATES_TIMESTAMPS_END_INDEX] = InfoConfReader().time_from_syscnt(
                timestamp, NumberConstant.MICRO_SECOND)


class EndState(MemcpyState):
    """
    class for ending state
    """

    def process_recieve_tag(self: any, timestamp: int) -> None:
        self.memcpy_recorder.start_new_batch()
        self.memcpy_recorder.each_batch_timestamp[self.NEWEST_BATCH_INDEX][
            MemoryCopyConstant.STATES_TIMESTAMPS_RECEIVE_INDEX] = InfoConfReader().time_from_syscnt(
                timestamp, NumberConstant.MICRO_SECOND)

    def process_start_tag(self: any, timestamp: int) -> None:
        logging.warning("Miss state tag %d of stream %d task %d.",
                        MemoryCopyConstant.RECEIVE_TAG,
                        self.memcpy_recorder.stream_id,
                        self.memcpy_recorder.task_id)
        self.memcpy_recorder.start_new_batch()
        self.memcpy_recorder.each_batch_timestamp[self.NEWEST_BATCH_INDEX][
            MemoryCopyConstant.STATES_TIMESTAMPS_START_INDEX] = InfoConfReader().time_from_syscnt(
                timestamp, NumberConstant.MICRO_SECOND)

    def process_end_tag(self: any, timestamp: int) -> None:
        logging.warning("The state tag %d of stream %d task %d is repeating.",
                        MemoryCopyConstant.END_TAG,
                        self.memcpy_recorder.stream_id,
                        self.memcpy_recorder.task_id)
