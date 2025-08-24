#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import os
import re
import subprocess
import stat
from threading import Thread, Lock

from common import FileOperate as f
from common import log_error, log_warning
from common.cmd_run import check_command, run_linux_cmd
from common.task_common import out_progress_bar, str_to_hex, is_hexadecimal
from common.const import ADDR_LEN_HEX


class ParseStackCore:
    def __init__(self, symbol, file=None):
        self.file = file
        self.symbol_path = symbol
        self.__readelf = "readelf"
        self.__addr2line = "addr2line"
        self.output_logs = {}
        self.maps_addr_binary_path = {}
        self.lock = Lock()

    def check_tool_exists(self):
        if not check_command(self.__readelf):
            log_error("The readelf tool does not exist. Install it before using it.")
            return False
        if not check_command(self.__addr2line):
            log_error("The addr2line tool does not exist. Install it before using it.")
            return False
        return True

    @staticmethod
    def write_res_file(file_name, file_lines):
        try:
            with open(file_name, "w") as fw:
                fw.writelines(file_lines)
        except Exception as e:
            log_error(e)
            return False
        return True

    def error(self, msg):
        if self.file:
            log_error(msg)

    def warning(self, so_name, msg):
        if self.file:
            self.lock.acquire()  # Exclusive Locking
            error_info = self.output_logs.get(so_name)
            if not (error_info and msg == error_info):
                log_warning(msg)
                self.output_logs[so_name] = msg
            self.lock.release()  # Unlock

    def get_source_location(self, so_name, address):
        """Run the addr2line command to obtain the function name and the line where the function is located."""
        try:
            output = subprocess.check_output([self.__addr2line, hex(address), '-e', so_name, '-f', '-C', '-s', '-i'],
                                             stderr=subprocess.STDOUT)
            output_lines = output.decode().strip().split("\n")
            result_lines = []
            for line in output_lines:
                if line.startswith(self.__addr2line):
                    warning_info = f"{so_name} {line.split(':')[-1]}"
                    self.warning(so_name, warning_info)
                    continue
                result_lines.append(line)
            return result_lines
        except Exception as e:
            self.warning(so_name, f"{so_name} is not permitted to read.")
            return []

    def file_lines_add_stack_num(self, file_lines):
        # stack add line num
        file_lines_with_stack_num = []
        stack_num = 0
        for line in file_lines:

            if line.endswith("Ignore\n"):
                stack_str = f"#0{stack_num}" if stack_num < 10 else f"#{stack_num}"
                file_lines_with_stack_num.append(f"{stack_str} {' ' * ADDR_LEN_HEX} Ignore\n")
                stack_num += 1
                continue
            if line.startswith("Thread "):
                stack_num = 0
            if not line.startswith("### "):
                file_lines_with_stack_num.append(line)
                continue

            for _line in line.split("\n"):
                if not _line:
                    continue
                stack_str = f"#0{stack_num}" if stack_num < 10 else f"#{stack_num}"
                file_lines_with_stack_num.append(f"{stack_str}{_line[3:]}\n")
                stack_num += 1
        return file_lines_with_stack_num

    def _get_line_with_addr2line(self, binary_path, address, stack_addr, so_name):
        all_func = self.get_source_location(binary_path, address)
        file_line = ""
        if not all_func:
            return file_line
        for i in range(0, len(all_func), 2):
            func_name, func_file = all_func[i], all_func[i + 1]
            if i == 0:
                file_line += f"### {stack_addr} {func_name} in {func_file} from {so_name}\n"
            else:
                file_line += f"### {' ' * len(stack_addr)} {func_name} in {func_file} from {so_name}\n"
        return file_line

    def parse_line(self, index, line, file_lines):
        line_num, stack_addr, delta_addr, binary_path = line.strip("\n").split(" ")[:4]
        so_name = os.path.basename(binary_path)
        # Obtain the actual binary file.
        if self.symbol_path:
            for path in self.symbol_path:
                so_path = os.path.join(path, so_name)
                binary_path = ""
                if os.path.exists(so_path):
                    binary_path = so_path
                    break
        else:
            maps_binary_path = self.maps_addr_binary_path.get(str_to_hex(delta_addr))
            if maps_binary_path:
                binary_path = maps_binary_path

        # if it does not exist or is not a file
        if binary_path == "" or not os.path.exists(binary_path):
            is_file = False
        else:
            _mode = os.stat(binary_path).st_mode
            is_file = any([os.path.isfile(binary_path), stat.S_ISBLK(_mode), stat.S_ISCHR(_mode), stat.S_ISSOCK(_mode)])
        if not is_file:
            warning_info = f"{so_name} not found in symbol_path directory." if self.symbol_path \
                else f"{binary_path} is not exists."
            self.warning(so_name, warning_info)
            file_lines[index] = line.replace(line_num, "###")
            return False
        if run_linux_cmd(f"{self.__readelf} -h {binary_path} | grep EXEC"):
            address = str_to_hex(stack_addr)
        else:
            address = str_to_hex(stack_addr) - str_to_hex(delta_addr)

        line_with_addr = self._get_line_with_addr2line(binary_path, address, stack_addr, so_name)
        file_lines[index] = line_with_addr if line_with_addr else line.replace(line_num, "###")
        return True

    def set_maps_addr_binary_path(self, file_lines):
        if self.symbol_path:
            return
        start_up = False
        for line in file_lines:
            if line.startswith("["):
                # get [maps] info
                if not start_up and line.startswith("[map"):
                    start_up = True
                else:
                    start_up = False
                continue
            if not start_up:
                continue
            line_list = [i.strip() for i in line.strip("\n").split(" ") if i.strip()]
            if len(line_list) != 6 or not line_list[-1].startswith("/"):
                continue
            addr, _, _, _, _, binary_path = line_list
            start_addr = addr.split("-")[0]
            if not is_hexadecimal(start_addr):
                continue
            start_addr_int = str_to_hex(start_addr)
            self.maps_addr_binary_path[start_addr_int] = binary_path

    def start_parse_file(self, stackcore_file, count=0):
        """Parsing a single file"""
        stackcore_file_name = stackcore_file.split(os.sep)[-1]
        if not stackcore_file_name.startswith("stackcore"):
            log_error(f"The {stackcore_file} file is not in stackcore format.")
            return False
        # Check whether the readelf and addr2line tools exist.
        if not self.check_tool_exists():
            return False
        try:
            with open(stackcore_file, "r") as fp:
                file_lines = fp.readlines()
        except Exception as e:
            self.error(e)
            return False
        if not file_lines:
            self.error(f"The {stackcore_file_name} file is empty.")
            return False

        # not symbol_path, get binary_path from maps
        self.set_maps_addr_binary_path(file_lines)

        start_up = False
        threads = []
        if self.file:
            count = len(file_lines)
        for index, line in enumerate(file_lines):
            if self.file:
                out_progress_bar(count, index)
            if line.startswith("["):
                # If [stack] is found, the parsing of the next line starts. Otherwise, the parsing ends.
                if not start_up and line.startswith("[stack]"):
                    start_up = True
                else:
                    start_up = False
                continue
            # If it is not started and is not in stackcore format, it is not processed.
            if not start_up or not re.match("#[0-9]+?", line) or len(line.split(" ")) < 4:
                continue
            line_num, stack_addr, delta_addr, binary_path = line.strip("\n").split(" ")[:4]
            if not (is_hexadecimal(stack_addr) and is_hexadecimal(delta_addr)):
                continue
            t = Thread(target=self.parse_line, args=(index, line, file_lines), daemon=True)
            t.start()
            threads.append(t)
        # wait for all threads to end.
        for t in threads:
            t.join()
        file_lines = self.file_lines_add_stack_num(file_lines)
        return self.write_res_file(stackcore_file, file_lines)

    def save_file_result(self, stackcore_file, count, num, results):
        ret = self.start_parse_file(stackcore_file, count)
        out_progress_bar(count, num)
        if not ret:
            log_error(f"Failed to analyze the '{stackcore_file}' file.")
        results.append(ret)

    def run(self, stack_core_path, count=0):
        stackcore_dirs = f.walk_dir(stack_core_path)
        if not stackcore_dirs:
            return False
        if not self.check_tool_exists():
            return False
        num = 0
        threads = []
        results = []
        for dirs, _, files in stackcore_dirs:
            for file in files:
                stackcore_file = os.path.join(dirs, file)
                num += 1
                t = Thread(target=self.save_file_result, args=(stackcore_file, count, num, results), daemon=True)
                t.start()
                threads.append(t)
        # wait for all threads to end.
        for t in threads:
            t.join()
        out_progress_bar(count, count)
        if not self.symbol_path:
            return any(results)
        return True
