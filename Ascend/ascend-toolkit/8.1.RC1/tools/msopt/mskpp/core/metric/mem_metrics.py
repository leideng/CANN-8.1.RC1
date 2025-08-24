#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import logging
from ..common import Singleton
from .metrics_summary import PipeMetricsSummary, InstructionMetricsSummary


@Singleton
class MemMetrics:
    def __init__(self):
        self._summary = {}
        self._total_summary = {
            "size": 0  # 搬运量， 单位：B
        }

    def add_event(self, pipe_name, instr_obj):
        if pipe_name not in self._summary.keys():
            self._summary[pipe_name] = MemPipeMetrics(pipe_name)
        size = instr_obj.move_size()
        self._summary[pipe_name].add_event(MemEvent(instr_obj.task_name, size))
        self._total_summary["size"] += size

    def summary(self):
        for pipe_name in self._summary.keys():
            self._summary[pipe_name].summary()
        PipeMetricsSummary().update(
            "Total", "Size(B)", self._total_summary["size"])


class MemEvent:
    def __init__(self, name, size):
        self.name = name
        self.move_size = size


class MemEvents:
    '''
    统计每类事件的累积数据
    '''

    def __init__(self, name):
        self.name = name
        self.move_size = 0

    def __str__(self):
        msg = "{} : size(B) {}".format(self.name, self.move_size)
        return msg

    def add_event(self, event):
        self.move_size += event.move_size


class MemPipeMetrics:
    '''
    统计每个pipe的信息
    '''

    def __init__(self, name):
        self.name = name
        self._detail = {}
        self._summary = {
            "size": 0  # 搬运量， 单位：B
        }

    def add_event(self, event):
        if event.name not in self._detail.keys():
            self._detail[event.name] = MemEvents(event.name)
        self._detail[event.name].add_event(event)
        self._summary["size"] += event.move_size

    def summary(self):
        PipeMetricsSummary().update(
            self.name, "Size(B)", self._summary["size"])
        for inst, event in self._detail.items():
            InstructionMetricsSummary().update(inst, "Size(B)", event.move_size)
