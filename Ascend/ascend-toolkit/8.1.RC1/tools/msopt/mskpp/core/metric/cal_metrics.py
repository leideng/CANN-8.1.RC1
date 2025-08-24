#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import logging
from ..common import Singleton
from .metrics_summary import PipeMetricsSummary, InstructionMetricsSummary


@Singleton
class CalMetrics:
    def __init__(self):
        self._summary = {}
        self._total_summary = {
            "size": 0  # 计算量， 单位：OPS
        }

    def add_event(self, pipe_name, instr_obj):
        if pipe_name not in self._summary.keys():
            self._summary[pipe_name] = CalPipeMetrics(pipe_name)
        size = instr_obj.cal_size()
        self._summary[pipe_name].add_event(CalEvent(instr_obj.task_name, size))
        self._total_summary["size"] += size

    def summary(self):
        for pipe_name in self._summary.keys():
            self._summary[pipe_name].summary()
        PipeMetricsSummary().update(
            "Total", "Ops", self._total_summary["size"])


class CalEvent:
    def __init__(self, name, size):
        self.name = name
        self.cal_size = size


class CalEvents:
    '''
    统计每类事件的累积数据
    '''

    def __init__(self, name):
        self.name = name
        self.cal_size = 0

    def __str__(self):
        msg = "{} : size(OPS) {}".format(self.name, self.cal_size)
        return msg

    def add_event(self, event):
        self.cal_size += event.cal_size


class CalPipeMetrics:
    '''
    统计每个pipe的信息
    '''

    def __init__(self, name):
        self.name = name
        self._detail = {}
        self._summary = {
            "size": 0  # 计算量， 单位：OPS
        }

    def add_event(self, event):
        if event.name not in self._detail.keys():
            self._detail[event.name] = CalEvents(event.name)
        self._detail[event.name].add_event(event)
        self._summary["size"] += event.cal_size

    def summary(self):
        PipeMetricsSummary().update(self.name, "Ops", self._summary["size"])
        for inst, event in self._detail.items():
            InstructionMetricsSummary().update(inst, "Ops", event.cal_size)
