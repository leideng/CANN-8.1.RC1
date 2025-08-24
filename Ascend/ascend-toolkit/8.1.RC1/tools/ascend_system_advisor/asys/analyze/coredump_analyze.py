#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import re
import subprocess
from platform import machine
from multiprocessing import Pool, cpu_count, Manager

from common import log_error, log_warning
from common.task_common import is_hexadecimal, int_to_hex
from common.const import REG_OFF, REG_THREAD, REG_STACK, ADDR_LEN_HEX, ADDR_BIT_LEN, GDB_LAYER_MAX
from params import ParamDict


def _get_reg_info_cmd():
    sys_type = machine()
    if sys_type == "x86_64":
        reg_cmd = f"info reg rbp rsp rip\n"
    elif sys_type == "aarch64":
        reg_cmd = f"info reg x29 sp pc\n"
    else:
        reg_cmd = ""
    return reg_cmd


def thread_stacks_reg_info(cmd, thread, stacks, queue_reg_info):
    """
    get thread all stacks reg value
    """
    thread_id = thread.split(" ")[1]
    gdb_process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   bufsize=1, encoding='utf-8', errors='ignore', text=True, shell=True)
    _reg_info = {thread: dict()}
    gdb_process.stdin.write(f"thread {thread_id}\n")

    reg_cmd = _get_reg_info_cmd()
    if not reg_cmd:
        return False

    for stack in stacks:
        stack_id = stack.split(" ")[0][1:]
        reg_list = [None, None, None]
        gdb_process.stdin.write(f"frame {stack_id}\n")
        gdb_process.stdin.write(reg_cmd)
        while True:
            line = gdb_process.stdout.readline().rstrip()
            gdb_process.stdout.flush()
            data_list = line.strip("\n").strip(" ").split()
            if line.startswith("(gdb) rbp ") or line.startswith("(gdb) x29 "):
                reg_list[0] = data_list[2]
                continue
            if line.startswith("rsp ") or line.startswith("sp "):
                reg_list[1] = data_list[1]
                continue
            if line.startswith("rip ") or line.startswith("pc "):
                reg_list[2] = data_list[1]

            if all(reg_list):
                line_num = f"#{stack_id}"
                # #0 -> #00
                if len(line_num) == 2:
                    line_num = f"#0{stack_id}"
                _reg_info[thread][line_num] = reg_list
                break
    gdb_process.stdin.write("quit\n")
    gdb_process.stdin.write("y\n")

    queue_reg_info.put(_reg_info)
    return True


class CoreDump:
    def __init__(self, exe_file, core_file, symbol, output):
        self.exe_file = exe_file
        self.core_file = core_file
        self.symbol = symbol
        self.output = output
        self.bt_info = dict()
        self.map_info = list()
        self.map_str = "[maps]\n"
        self.reg_level = ParamDict().get_arg("reg")

    @staticmethod
    def check_map_line(data_list):
        """
        Check whether the data row is in the mapping format.
        """
        if len(data_list) < 5:
            return False
        if not (is_hexadecimal(data_list[0]) and is_hexadecimal(data_list[1]) and is_hexadecimal(data_list[2])):
            return False
        return True

    def collect_info(self, thread_name, data_list, line, before_line):
        """
        Collect stack information and map table information.
        """
        if re.match(r'#(\d+)', data_list[0]) and thread_name:
            if self.bt_info.get(thread_name):
                self.bt_info[thread_name].append(line)
            else:
                self.bt_info[thread_name] = [line]
        elif data_list[0] == "Start" and data_list[-1] == "objfile":
            self.map_str += (line[6:] + "\n")
        elif self.check_map_line(data_list):
            self.map_str += (line[6:] + "\n")
            if data_list[-1] != before_line:
                self.map_info.append([int_to_hex(data_list[0]), int_to_hex(data_list[1]), data_list[-1]])
            else:
                if int_to_hex(data_list[0]) < self.map_info[-1][0]:
                    self.map_info[-1][0] = int_to_hex(data_list[0])
                if int_to_hex(data_list[1]) > self.map_info[-1][1]:
                    self.map_info[-1][1] = int_to_hex(data_list[1])
            before_line = data_list[-1]
        return before_line

    def view_map(self, stack_txt, bt_line, reg_info):
        """
        Obtain the stack start address and dynamic library from the mapping table.
        """
        bt_list = bt_line.strip("\n").strip(" ").split()
        # #0 -> #00
        if len(bt_list[0]) == 2:
            bt_list[0] = f"#0{bt_list[0][1]}"
        address = int_to_hex(bt_list[1])
        for values in self.map_info:
            start_address, end_address, key = values
            if int(start_address, ADDR_BIT_LEN) < int(address, ADDR_BIT_LEN) < int(end_address, ADDR_BIT_LEN):
                stack_txt += f"{bt_list[0]} 0x{address[2:].rjust(ADDR_BIT_LEN, '0')} " \
                             f"0x{start_address[2:].rjust(ADDR_BIT_LEN, '0')} {key}\n"
                stack_txt = self._stack_add_reg(stack_txt, bt_list[0], reg_info)
                return stack_txt
        if self.symbol:
            stack_txt += " ".join(bt_list)
            stack_txt += "\n"
            stack_txt = self._stack_add_reg(stack_txt, bt_list[0], reg_info)

        return stack_txt

    @staticmethod
    def _get_reg_str(start_str, reg_values):
        if not reg_values or len(reg_values) != 3:
            return ""

        reg_cmd = _get_reg_info_cmd()
        line_start = " " * (len(start_str) + 1)
        if "x29 sp pc" in reg_cmd:
            reg_str = f"{line_start}fp = {reg_values[0]}    sp = {reg_values[1]}\n{line_start}pc = {reg_values[2]}\n"
        elif "rbp rsp rip" in reg_cmd:
            reg_str = f"{line_start}rbp = {reg_values[0]}    rsp = {reg_values[1]}\n{line_start}rip = {reg_values[2]}\n"
        else:
            reg_str = ""
        return reg_str

    def _stack_add_reg(self, stack_txt, stack_id, reg_info):
        # stack_txt add stack reg_data
        if not reg_info or isinstance(reg_info, list):
            return stack_txt
        reg_values = reg_info.get(stack_id)
        stack_txt += self._get_reg_str(stack_id, reg_values)

        return stack_txt

    def parse_stackcore(self, stack_txt, bt_lines, reg_info=None):
        if self.reg_level == REG_THREAD:
            stack_txt += self._get_reg_str("", reg_info)
        for bt_line in bt_lines:
            bt_list = bt_line.strip("\n").strip(" ").split()
            # #0 -> #00
            if len(bt_list[0]) == 2:
                bt_list[0] = f"#0{bt_list[0][1]}"
            if self.symbol and "in ??" not in bt_line:
                stack_txt += " ".join(bt_list)
                stack_txt += "\n"
                stack_txt = self._stack_add_reg(stack_txt, bt_list[0], reg_info)
                continue
            if not is_hexadecimal(bt_list[1]):
                stack_txt += f"{bt_list[0]} {' ' * ADDR_LEN_HEX} {' ' * ADDR_LEN_HEX} Ignore\n"
                continue
            stack_txt = self.view_map(stack_txt, bt_line, reg_info)
        return stack_txt

    def _get_reg_info_level_stack(self):
        queue_reg_info = Manager().Queue()
        cmd = f'gdb {self.exe_file} {self.core_file}'
        p = Pool(cpu_count() - 1)
        for thread, stacks in self.bt_info.items():
            p.apply_async(thread_stacks_reg_info, args=(cmd, thread, stacks, queue_reg_info))
        p.close()
        p.join()

        threads_stacks_reg_info = dict()
        while queue_reg_info.qsize() > 0:
            for thread, stacks in queue_reg_info.get().items():
                threads_stacks_reg_info[thread] = stacks
        return threads_stacks_reg_info

    def _get_reg_info_level_thread(self):
        gdb_process = subprocess.Popen(f'gdb {self.exe_file} {self.core_file}',
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       encoding='utf-8', errors='ignore', text=True, bufsize=1, shell=True)
        _reg_info = dict()
        reg_cmd = _get_reg_info_cmd()
        if not reg_cmd:
            return _reg_info

        for thread in self.bt_info.keys():
            thread_id = thread.split(" ")[1]
            gdb_process.stdin.write(f"thread {thread_id}\n")
            gdb_process.stdin.write(reg_cmd)
            reg_list = [None, None, None]
            while True:
                line = gdb_process.stdout.readline().rstrip()
                gdb_process.stdout.flush()
                data_list = line.strip("\n").strip(" ").split()
                if line.startswith("(gdb) rbp ") or line.startswith("(gdb) x29 "):
                    reg_list[0] = data_list[2]
                    continue
                if line.startswith("rsp ") or line.startswith("sp "):
                    reg_list[1] = data_list[1]
                    continue
                if line.startswith("rip ") or line.startswith("pc "):
                    reg_list[2] = data_list[1]

                if all(reg_list):
                    _reg_info[thread] = reg_list
                    break
        gdb_process.stdin.write("quit\n")
        gdb_process.stdin.write("y\n")

        return _reg_info

    def get_threads_stacks_reg_info(self):
        if self.reg_level == REG_OFF:
            return {}
        elif self.reg_level == REG_THREAD:
            return self._get_reg_info_level_thread()
        elif self.reg_level == REG_STACK:
            return self._get_reg_info_level_stack()
        else:
            return {}

    def start_gdb(self, stack_txt):
        gdb_process = subprocess.Popen(f'gdb {self.exe_file} {self.core_file}',
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       encoding='utf-8', errors='ignore', text=True, shell=True)
        gdb_process.stdin.write("info inferiors\n")
        gdb_process.stdin.write("info sharedlibrary\n")
        gdb_process.stdin.write(f"thread apply all bt {GDB_LAYER_MAX}\n")
        gdb_process.stdin.write("info proc mappings\n")
        gdb_process.stdin.write("quit\n")
        gdb_process.stdin.write("y\n")
        console_output, _ = gdb_process.communicate()
        out_lines = console_output.split("\n")
        thread_name = None
        pid = 0
        before_line = ""
        crash_info = {"crash reason": "", "crash pid": "", "crash tid": ""}
        for line in out_lines:
            if f"{self.exe_file}: No such file or directory" in line:
                log_error(line)
                return stack_txt, pid
            if "core file may not match specified executable file" in line:
                log_warning("core file may not match specified executable file")
                continue
            if line.startswith("Program terminated"):
                crash_info["crash reason"] = f"{line.split('signal')[-1].strip(' ').split(',')[0]}"
                continue
            data_list = line.strip("\n").strip(" ").split()
            if "Current thread" in line:
                crash_info["crash tid"] = re.match(r"(\d+)", data_list[-1])[0]
                continue
            if data_list and data_list[0] == "No":
                log_warning(f'could not load shared library symbols for "{data_list[-1]}", parsing errors may occur.')
                continue
            if len(data_list) < 2:
                continue
            if line.startswith("*") and "process" == data_list[2]:
                pid = data_list[3]
                continue
            if line.startswith("Thread") and "LWP" in line:
                tid = re.search(r"LWP (\d+)", line).group(1)
                thread_name = f"Thread {data_list[1]} ({tid})"
                continue
            before_line = self.collect_info(thread_name, data_list, line, before_line)
        stack_txt, pid = self._process_stack_txt(crash_info, pid, stack_txt)
        return stack_txt, pid

    def _process_stack_txt(self, crash_info, pid, stack_txt):
        if not self.map_info or not self.bt_info:
            log_error("Failed to obtain the core dump information.")
            return stack_txt, pid

        crash_info["crash pid"] = pid

        for crash_key, crash_value in crash_info.items():
            stack_txt += f"{crash_key}: {crash_value}\n"

        stack_txt += "\n"
        stack_txt += "[stack]\n"
        threads_stacks_reg_info = self.get_threads_stacks_reg_info()
        for stack_name, bt_lines in self.bt_info.items():
            stack_txt += (stack_name + "\n")
            stacks_reg_info = threads_stacks_reg_info.get(stack_name)
            stack_txt = self.parse_stackcore(stack_txt, bt_lines, stacks_reg_info)
            stack_txt += "\n"
        stack_txt += self.map_str
        return stack_txt, pid
