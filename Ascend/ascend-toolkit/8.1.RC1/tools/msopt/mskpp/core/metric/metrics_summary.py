# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import os
from collections import OrderedDict
from abc import ABC
import logging
from mskpp._C import arch
from ..common.singleton import Singleton
from .output_tool import TableOutputWrapper
from .prof import ProfSummary
from .file_system import FileChecker


class MetricsSummary(ABC):
    def __init__(self):
        self._summary = OrderedDict()
        self._keys = []
        self._title = "Title"

    def update(self, name, key, value):
        if name not in self._summary.keys():
            self._summary[name] = OrderedDict()
        self._summary[name][key] = value
        if key not in self._keys:
            self._keys.append(key)

    def output(self, filepath=None):
        with TableOutputWrapper(filepath=filepath) as out:
            header = [self._title] + self._keys
            out.write(header)
            for pipe, stat_info in self._summary.items():
                row = [pipe]
                for key in self._keys:
                    row.append(stat_info.get(key, "-"))
                out.write(row)
        if filepath:
            logging.info("The %s summary is save at %s", self._title, filepath)


@Singleton
class PipeMetricsSummary(MetricsSummary):
    def __init__(self):
        super().__init__()
        self._title = "Pipe"
        self.prof_summary_path = None
        
    def get_summary(self):
        return self._summary

    def set_total_duration(self, duration):
        self.update("Total", "Duration(us)", round(arch.cal_duration(duration), 4))

    def set_prof_summary_path(self, file_path):
        file_checker = FileChecker(file_path, "csv")
        if not file_checker.check_input_file():
            raise Exception("set prof summary path failed.")
        self.prof_summary_path = file_path
        
    def output(self, filepath):
        if self.prof_summary_path:
            self.add_prof_summary()
        super().output(filepath)
        
    def add_prof_summary(self):
        prof_summary = ProfSummary().parse(self.prof_summary_path)
        for index, prof in prof_summary.items():
            duration = f"ProfDuration(us)_{index}"
            ratio = f"ProfRatio_{index}"
            for name, summary in self._summary.items():
                self.update(name, duration, prof.get(name, "-"))
                ratio_value = None
                if summary.get("Duration(us)", 0) != 0:
                    ratio_value = round(prof.get(name, 0) / summary["Duration(us)"], 4)
                self.update(name, ratio, ratio_value if ratio_value else "-")
            

@Singleton
class InstructionMetricsSummary(MetricsSummary):
    def __init__(self):
        super().__init__()
        self._title = "Instruction"
