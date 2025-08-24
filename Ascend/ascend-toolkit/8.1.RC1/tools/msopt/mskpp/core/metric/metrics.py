#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import logging
import os
from ..common.singleton import Singleton
from ..common import checker
from .mem_metrics import MemMetrics
from .cycle_metrics import CycleMetrics
from .cal_metrics import CalMetrics
from .metrics_summary import PipeMetricsSummary, InstructionMetricsSummary


@Singleton
class Metrics:
    def __init__(self):
        self.is_enable = False

    @staticmethod
    def add_event(task_obj):
        if task_obj.instr_obj.name == "MOV":
            MemMetrics().add_event(task_obj.owner, task_obj.instr_obj)
        else:
            CalMetrics().add_event(task_obj.owner, task_obj.instr_obj)
        CycleMetrics().add_event(task_obj.owner, task_obj.instr_obj)

    @staticmethod
    def summary(output_dir):
        logging.info("Metrics:")
        CycleMetrics().summary(output_dir)
        MemMetrics().summary()
        CalMetrics().summary()
        pipe_csv = os.path.join(output_dir, "Pipe_statistic.csv")
        if checker.check_path_exists(pipe_csv):
            raise Exception("The file {} already exists, cannot generate, please remove it first".format(pipe_csv))
        PipeMetricsSummary().output(pipe_csv)
        instruction_csv = os.path.join(output_dir, "Instruction_statistic.csv")
        if checker.check_path_exists(instruction_csv):
            raise Exception("The file {} already exists, cannot generate, please remove it first".
                            format(instruction_csv))
        InstructionMetricsSummary().output(instruction_csv)
    
    @staticmethod
    def set_total_duration(duration):
        PipeMetricsSummary().set_total_duration(duration)
    
    @staticmethod
    def set_prof_summary_path(file_path):
        PipeMetricsSummary().set_prof_summary_path(file_path)

    def set_enable(self, enable):
        self.is_enable = enable