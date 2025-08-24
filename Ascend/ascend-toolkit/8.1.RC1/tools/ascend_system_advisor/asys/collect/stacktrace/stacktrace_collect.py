#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import os
import re
import time

from common import get_cann_log_path
from common import log_error, popen_run_cmd, log_warning
from common.const import ATRACE_LOG_NAME, RetCode
from params import ParamDict
from collect.stacktrace import AscendTraceDll

EVERY_ROUND_TIME = 0.5
CHECK_BIN_TIMEOUT = 10


class AsysStackTrace(AscendTraceDll):
    """
    Send signal to export stackcore
    """
    def __init__(self):
        super(AsysStackTrace, self).__init__()
        self.run_mode = ParamDict().get_arg("run_mode")
        self.remote_id = ParamDict().get_arg("remote")
        self.is_all_task = ParamDict().get_arg("all")
        self.quiet = ParamDict().get_arg("quiet")

    def _get_bin_file_path(self, all_exists_bin):
        trace_path, _ = get_cann_log_path(ATRACE_LOG_NAME)
        for path, _, files in os.walk(os.path.abspath(trace_path)):
            for file in files:
                if not (file.startswith(f"stackcore_tracer_35_{self.remote_id}_") and file.endswith(".bin")):
                    continue
                bin_file_path = os.path.join(path, file)
                if bin_file_path in all_exists_bin:
                    continue
                return bin_file_path

    def _get_exists_bin_file_num(self):
        trace_path, _ = get_cann_log_path(ATRACE_LOG_NAME)
        cmd = f"ls -lt {os.path.abspath(trace_path)}/trace_*/stackcore_event_{self.remote_id}_*/" \
              f"stackcore_tracer_35_{self.remote_id}_*.bin | wc -l"
        ret = popen_run_cmd(cmd).replace("\n", "")
        if not ret.isdigit():
            return 0
        return int(ret)

    def _get_last_bin_file_name(self):
        trace_path, _ = get_cann_log_path(ATRACE_LOG_NAME)
        cmd = f"ls -lt {os.path.abspath(trace_path)}/trace_*/stackcore_event_{self.remote_id}_*/" \
              f"stackcore_tracer_35_{self.remote_id}_*.bin | head -n 1 | awk \'{{print $9}}\'"
        return popen_run_cmd(cmd).replace("\n", "")

    def _wait_bin_file_generate(self, exists_bin_file_num):
        bin_file_name = None
        for _ in range(int(CHECK_BIN_TIMEOUT // EVERY_ROUND_TIME)):  # 20 * 0.5 = 10s
            if not bin_file_name:
                current_bin_file_num = self._get_exists_bin_file_num()
                if current_bin_file_num == exists_bin_file_num:
                    time.sleep(EVERY_ROUND_TIME)
                    continue
                if current_bin_file_num > exists_bin_file_num:
                    bin_file_name = self._get_last_bin_file_name()
                    continue

            if popen_run_cmd(f"lsof {bin_file_name}"):
                time.sleep(EVERY_ROUND_TIME)
                continue
            return bin_file_name
        log_error("Generating the stackcore bin file timeout. "
                  "For details, see the related description in the document.")
        return None

    @staticmethod
    def _check_other_param():
        output = ParamDict().get_arg("output")
        task_dir = ParamDict().get_arg("task_dir")
        tar = ParamDict().get_arg("tar")
        if output or task_dir or tar:
            log_error(
                "'--output', '--task_dir', and '--tar' can be used only when '-r' is not used."
            )
            return False
        return True

    def _check_remote_id_validity(self):
        if self.remote_id < 2:
            log_error(f"the value of '--remote' must be greater than 1, input: {self.remote_id}.")
            return False

        try:
            os.kill(self.remote_id, 0)
        except Exception:
            log_error(f"no such process, id: {self.remote_id}.")
            return False

        # check remote pid ?
        cmd = f"ps -p {self.remote_id}"
        ret = popen_run_cmd(cmd)[:-1].split("\n")
        if len(ret) != 2:
            log_error("The remote parameter must be set to the PID of the process.")
            return False
        return True

    def _get_all_tid_of_process(self, current_pid):
        cmd = fr"ps -efT | grep ' {self.remote_id} ' | grep -v {current_pid} | awk '{{print $2}}' | xargs ps -Lf \
                 | awk '{{print $4}}'"
        ret = popen_run_cmd(cmd).split("\n")
        ret = [i for i in ret if i.isdigit()]
        if len(ret) < 2:
            log_error(f"get pid fail by remote: {self.remote_id}.")
            return []
        return ret

    @staticmethod
    def _get_other_stacktrace_remote_id(current_pid):
        all_remote_id = []
        cmd = rf"ps -ef | grep -E asys[\.py]{{0\,3}}\ collect | grep stacktrace | grep -v ' {current_pid} '"
        ret = popen_run_cmd(cmd).split("\n")
        ret = [i for i in ret if i]
        if not ret:
            return all_remote_id

        p_pid = os.getppid()
        for process in ret:
            process_info_list = [i for i in process.split(" ") if i]
            _pid = process_info_list[1]
            # exclude current process ppid is other process pid
            if _pid.isdigit() and int(_pid) == p_pid:
                continue
            _remote_id = re.search(r" --remote[ =](\d+)", process)
            if _remote_id:
                all_remote_id.append(_remote_id.group(1))
        return all_remote_id

    def _check_collect_stacktrace_parallel(self):
        current_pid = os.getpid()
        all_remote_id = self._get_other_stacktrace_remote_id(current_pid)
        if not all_remote_id:
            return True
        # other running remote_id contains the current remote_id.
        if str(self.remote_id) in all_remote_id:
            return False

        all_tid_of_process = self._get_all_tid_of_process(current_pid)
        # abnormal state
        if not all_tid_of_process:
            return False

        all_tid_remote_id = all_remote_id + all_tid_of_process
        # tid contained in the current remote_id is running
        if len(all_tid_remote_id) > len(set(all_tid_remote_id)):
            return False
        return True

    def collect_stacktrace(self):
        """
        send signals to export stackcore files.
        """
        param_ret = self._check_other_param()
        if not param_ret:
            return False

        if self.remote_id is False or not self.is_all_task:
            log_error(f"'-r=stacktrace' must be used together with '--remote' and '--all'.")
            return False

        if self.trace_dll == RetCode.FAILED:
            return False

        if not self._check_remote_id_validity():
            return False
        log_warning(f"This command sends signal 35 to the process:{self.remote_id}. "
                    "If the process is executed to disable signal receiving through the environment variable "
                    f"ASCEND_COREDUMP_SIGNAL=none, the process:{self.remote_id} will be killed. "
                    "Are you sure that signal reception is not disabled? (Y/N)")
        if not self.quiet and input().upper() != "Y":
            return True

        if not self._check_collect_stacktrace_parallel():
            log_error("collect stacktrace not support Parallelism.")
            return False

        exists_bin_num = self._get_exists_bin_file_num()
        signal_ret = self.send_signal_to_pid(self.is_all_task, self.remote_id)
        if not signal_ret:
            return False

        bin_file_path = self._wait_bin_file_generate(exists_bin_num)
        if not bin_file_path:
            return False

        parse_ret = self.parse_stackcore_bin_to_txt(bin_file_path)
        if not parse_ret:
            return False

        return True
