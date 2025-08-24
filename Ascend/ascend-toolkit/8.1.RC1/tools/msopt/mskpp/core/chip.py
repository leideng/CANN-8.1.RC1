#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import logging
import time
import os
import shutil
from datetime import datetime, timezone
from mskpp._C import arch, task_schedule
from .metric import Metrics
from .trace import Trace
from .common import checker
from .metric.file_system import FileChecker, DATA_DIRECTORY_AUTHORITY


class Chip:
    support_list = ["Ascend910B1", "Ascend910B2", "Ascend910B3", "Ascend910B4", "Ascend910B4-1", "Ascend910B2C"]

    def __init__(self, name, debug_mode=False):
        self.chip_name = name
        self.need_trace = False
        self.need_metrics = False
        self.debug_mode = debug_mode
        self.output_dir = ''
        self.param_transfer()
        task_schedule.Schedule().set_debug_mode(debug_mode)

    def __enter__(self):
        self.create_output_dir()
        arch.set(self.chip_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        time.sleep(0)  # 通过cpu的等待，保证打印等信息的显示时序。
        if self.debug_mode:
            Trace().set_enable(False)
            Metrics().set_enable(False)
            task_schedule.Schedule().run()
            return
        duration = task_schedule.Schedule().run()
        if self.need_trace:
            Trace().dump(self.output_dir)
        if self.need_metrics:
            Metrics().set_total_duration(duration)
            Metrics().summary(self.output_dir)

    @staticmethod
    def set_cache_hit_ratio(config):
        arch.set_cache_hit_ratio(config["cache_hit_ratio"])

    @staticmethod
    def set_prof_summary_path(file_path):
        Metrics().set_prof_summary_path(file_path)

    def enable_trace(self):
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        self.need_trace = True
        Trace().set_enable(True)

    def enable_metrics(self):
        self.need_metrics = True
        Metrics().set_enable(True)

    def param_transfer(self):
        if not checker.is_required_type(self.chip_name, str) or self.chip_name not in Chip.support_list:
            raise Exception("Parameter chip_name in Chip is unsupported")
        self.chip_name = "Ascend910B1"
        if not checker.is_required_type(self.debug_mode, bool):
            raise Exception("Parameter debug_mode in Chip should be bool, but got: {}".format(type(self.debug_mode)))

    def create_output_dir(self):
        time_stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M%S")
        cur_dir = os.getcwd()
        self.output_dir = os.path.join(cur_dir, "MSKPP" + time_stamp)
        file_checker = FileChecker(self.output_dir, "dir")
        if not file_checker.check_output_file():
            raise Exception("Fail to Create output folder")
        os.makedirs(self.output_dir)
        os.chmod(self.output_dir, DATA_DIRECTORY_AUTHORITY)

        