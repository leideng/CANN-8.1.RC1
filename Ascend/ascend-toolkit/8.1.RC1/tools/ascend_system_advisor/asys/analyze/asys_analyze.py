#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import os
import stat
import time

from analyze.coredump_analyze import CoreDump
from common import log_error, log_warning
from common import FileOperate as f
from common.cmd_run import check_command
from common.task_common import get_target_cnt
from collect.trace import ParseTrace
from collect.stackcore import ParseStackCore
from params import ParamDict


class AsysAnalyze:
    def __init__(self):
        self.file = self.get_param_arg('file')
        self.path = self.get_param_arg('path')
        self.exe_file = self.get_param_arg("exe_file")
        self.core_file = self.get_param_arg("core_file")
        self.symbol = self.get_param_arg('symbol')
        self.symbol_path = self.get_param_arg('symbol_path')
        self.output = ParamDict().asys_output_timestamp_dir
        self.run_mode = self.get_param_arg('run_mode')

    @staticmethod
    def clean_output():
        f.remove_dir(ParamDict().asys_output_timestamp_dir)

    @staticmethod
    def get_param_arg(mode):
        if mode == "symbol":
            return ParamDict().get_arg(mode)
        return ParamDict().get_arg(mode) if ParamDict().get_arg(mode) else None

    def __copy_dir(self):
        if self.run_mode == "trace":
            return f.copy_dir(self.path, self.output)
        # stackcore
        for root, _, files in os.walk(self.path):
            for file in files:
                if not file.startswith("stackcore"):
                    continue
                root_path = os.path.relpath(root, self.path)
                if not f.copy_file_to_dir(os.path.join(root, file), os.path.join(self.output, root_path)):
                    return False
        return True

    def __atrace_analyze(self):
        """
        parse the trace file. If the file exists, parse the file. If the directory exists, parse the directory.
        """
        if self.run_mode == "trace":
            parse_struct = ParseTrace(True)
        elif self.run_mode == "stackcore":
            parse_struct = ParseStackCore(self.symbol_path, self.file)
            if not self.symbol_path:
                log_warning("'--symbol_path' is not set, the default path will be used to analyze.")
        else:
            return False
        if self.file:
            f.copy_file_to_dir(self.file, self.output)
            return parse_struct.start_parse_file(os.path.join(self.output, os.path.basename(self.file)))
        elif self.path:
            self.path = os.path.abspath(self.path)
            self.output = os.path.join(self.output, self.path.split(os.sep)[-1])
            copy_res = self.__copy_dir()
            if not copy_res:
                return False
            count = get_target_cnt(self.output)
            return parse_struct.run(self.output, count=count)
        else:
            log_error("analyze requires either the --file or --path argument")
            return False

    def __core_dump_analyze(self):
        stack_txt = "[process]\n"
        if not check_command("gdb"):
            log_error("gdb does not exist. Install gdb before using it.")
            return False
        if not self.exe_file:
            log_error("The --exe_file parameter must exist for analyze coredump.")
            return False
        if not self.core_file:
            log_error("The --core_file parameter must exist for analyze coredump.")
            return False
        core_dump = CoreDump(self.exe_file, self.core_file, self.symbol, self.output)
        stack_txt, pid = core_dump.start_gdb(stack_txt)
        if pid == 0:
            return False
        file_name = f"stackcore_{os.path.basename(self.exe_file)}_{pid}_{int(round(time.time() * 1000))}.txt"
        self.write_res_file(file_name, stack_txt)
        return True

    def write_res_file(self, file_name, file_content):
        try:
            flags = os.O_WRONLY | os.O_CREAT
            modes = stat.S_IWUSR | stat.S_IRUSR
            with os.fdopen(os.open(f"{self.output}/{file_name}", flags, modes), 'w') as fw:
                fw.write(file_content)
        except Exception as e:
            log_error(e)

    def analyze(self):
        if self.run_mode == "trace":
            return self.__atrace_analyze()
        elif self.run_mode == "stackcore":
            return self.__atrace_analyze()
        elif self.run_mode == "coredump":
            return self.__core_dump_analyze()
        else:
            log_error("only trace, stackcore and coredump are supported for analyze.")
            return False
