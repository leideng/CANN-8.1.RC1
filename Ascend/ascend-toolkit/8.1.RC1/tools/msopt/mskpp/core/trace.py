#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import os
import stat
import json
import logging
from mskpp._C import arch
from .common.singleton import Singleton
from .common import checker
from .metric.output_tool import SAVE_DATA_FILE_AUTHORITY, OPEN_FLAGS


pipes = {
    "PIPE-MTE1": 0,
    "PIPE-MTE2": 1,
    "PIPE-MTE3": 2,
    "PIPE-FIX": 3,
    "PIPE-M": 4,
    "PIPE-V": 5,
    "PIPE-S": 6,
}

# core type的集合，包含了不重复core类型，如aic0、aiv1等
core_type_list = list()


class TraceEvent:
    def __init__(self, pipe_type, pipe_name, task_name, start_time, end_time, size):
        self.pipe_type = pipe_type
        self.start_time = start_time
        self.dur = end_time - start_time
        self.size = size
        self.pipe_name = pipe_name
        self.task_name = task_name
        self.args = self.gen_args()
        self.core_type = str()
        self.pipe_id = int()
        self.gen_pipe_info()

    def gen_pipe_info(self):
        def parse_core_type(core_type_input):
            pipe_name_list = core_type_input.split("-")
            if len(pipe_name_list) < 2:  # pipe name按照-分割,至少有2个基本单元
                raise Exception("Invalid pipe(The pipe name({}) is not support)".format(core_type_input))
            if pipe_name_list[0] == "PIPE":
                return [None, core_type_input]
            if len(pipe_name_list) != 3:  # 带core信息的pipe name按照-分割,至少有3个基本单元
                raise Exception("Invalid pipe(The pipe name({}) is not support)".format(core_type_input))
            return [pipe_name_list[0], pipe_name_list[1] + "-" + pipe_name_list[2]]
        self.core_type = parse_core_type(self.pipe_name)[0]
        self.pipe_name = parse_core_type(self.pipe_name)[1]
        pipe_id = pipes.get(self.pipe_name, None)
        if pipe_id is None:
            raise Exception(
                "Unsupport event(The pipe({}) is not support)".format(self.pipe_name))
        self.pipe_id = pipe_id

    def gen_args(self):
        args_map = {"Cycle": self.dur}
        if self.pipe_type == "MOV":
            size_gbyte = self.size / 1024 / 1024 / 1024
            bandwidth_gbyte_per_second = 0 if arch.cal_duration(self.dur) == 0 else \
                (size_gbyte / arch.cal_duration(self.dur) * 1000000)
            args_map["Size(B)"] = self.size
            args_map["Bandwidth(GB/s)"] = round(bandwidth_gbyte_per_second, 2)
        else:
            args_map["Task Type"] = "AI_CORE"
            args_map["Ops"] = self.size
            args_map["Ops/Cycle"] = 0 if (self.dur == 0) else (self.size / self.dur)
        return args_map


@Singleton
class Trace:
    time_unit = "ns"  # only ms or ns

    def __init__(self):
        self.is_enable = False
        self.context_before = []
        self.context_after = []

    def set_enable(self, enable):
        self.is_enable = enable

    def add_event(self, trace_event):
        index = 0
        if len(core_type_list) != 0:
            index = core_type_list.index(trace_event.core_type)
        event = {
            "pid": index,
            "ph": "X",
            "ts": arch.cal_duration(trace_event.start_time),
            "dur": arch.cal_duration(trace_event.dur),
            "tid": trace_event.pipe_id,
            "name": trace_event.task_name,
            "args": trace_event.args
        }
        self.context_after.append(event)

    def dump(self, output_dir):
        self._add_head()
        self.context_after = self.context_before + self.context_after
        trace_file = os.path.join(output_dir, "trace.json")
        if checker.check_path_exists(trace_file):
            raise Exception("The file {} already exists, cannot generate, please remove it first".format(trace_file))
        trace_obj = {
            "displayTimeUnit": self.time_unit,
            "traceEvents": self.context_after
        }
        with os.fdopen(os.open(trace_file, OPEN_FLAGS, SAVE_DATA_FILE_AUTHORITY), 'w') as f:
            data = json.dumps(trace_obj)
            f.truncate()
            f.write(data)
            logging.info("The trace is save at %s", trace_file)

    def gen_head(self, base_args, core_type_index):
        p_head = {
            "args": {"name": base_args},
            "name": "process_name",
            "ph": "M",
            "pid": core_type_index
        }
        self.context_before.append(p_head)
        for name, tid in pipes.items():
            t_head = {
                "args": {"name": name},
                "name": "thread_name",
                "ph": "M",
                "pid": core_type_index,
                "tid": tid
            }
            self.context_before.append(t_head)

    def _add_head(self):
        if len(core_type_list) == 0:
            core_type_list.append("")  # 添加空串保证for循环至少可以进入一次
        base_args = "kernel perf prediction"
        for index, core_type in enumerate(core_type_list):
            base_args = base_args if (core_type == "") else core_type
            self.gen_head(base_args, index)
            base_args = ""
        core_type_list.clear()  # 生成trace.json前清空，避免老的Chip信息影响下一次的with Chip()
