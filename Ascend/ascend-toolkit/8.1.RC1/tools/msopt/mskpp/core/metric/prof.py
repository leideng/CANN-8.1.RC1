#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import csv
import logging
from .file_system import FileChecker
from ..common.singleton import Singleton


@Singleton
class ProfSummary:
    pipe_map = {
        "aicore_time(us)": "Total",
        "aic_mac_time(us)": "PIPE-M",
        "aic_mte1_time(us)": "PIPE-MTE1",
        "aic_mte2_time(us)": "PIPE-MTE2",
        "aic_fixpipe_time(us)": "PIPE-FIX",
        "aiv_time(us)": "Total",
        "aiv_vec_time(us)": "PIPE-V",
        "aiv_mte2_time(us)": "PIPE-MTE2",
        "aiv_mte3_time(us)": "PIPE-MTE3",
        "mac_time(us)": "PIPE-M",
        "mte1_time(us)": "PIPE-MTE1",
        "mte2_time(us)": "PIPE-MTE2",
        "mte3_time(us)": "PIPE-MTE3",
        "fixpipe_time(us)": "PIPE-FIX",
        "vec_time(us)": "PIPE-V",
    }

    def __init__(self):
        self._summary = {}

    @staticmethod
    def _check_can_transfer_float(value):
        try:
            float(value)
        except ValueError:
            return False
        return True

    def parse(self, prof_summary_path):
        file_checker = FileChecker(prof_summary_path, "csv")
        if not file_checker.check_input_file():
            logging.error("Profiling Summary file check fail")
            return self._summary
        with open(prof_summary_path, "r", encoding="utf-8") as fd:
            reader = csv.reader(fd)
            try:
                head = next(reader)
            except StopIteration as ex:
                logging.error("%s is empty", prof_summary_path)
                return self._summary
            fd.seek(0)
            self._parse_csv(csv.DictReader(fd))
        return self._summary

    def _parse_csv(self, csv_reader):
        for index, row in enumerate(csv_reader):
            if not self._check_row(row):
                continue
            self._summary[index] = {}
            for k, v in row.items():
                if self._check_cell(k, v):
                    pipe_name = self.pipe_map.get(k, "")
                    self._summary[index].setdefault(pipe_name, 0)
                    self._summary[index][pipe_name] += float(v)

    def _check_cell(self, key, content):
        if key not in self.pipe_map:
            return False
        if content in ["0", "N/A", "UNKNOWN", "", None]:
            return False
        return self._check_can_transfer_float(content)

    def _check_row(self, row):
        return any([self._check_cell(k, row.get(k)) for k in self.pipe_map])