#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
import os
import sys
from datetime import datetime, timezone

from params import ParamDict
from common.const import RetCode, CANN_LOG_NAME, consts, STACKTRACE
from common.log import log_error, log_info
from common.file_operate import FileOperate as f
from drv import EnvVarName

__all__ = [
    "create_out_timestamp_dir", "get_asys_output_path", "get_target_cnt", "out_progress_bar", "is_hexadecimal",
    "str_to_hex", "int_to_hex", "get_cann_log_path"
]


_asys_output_path = None


def get_asys_output_path():
    return _asys_output_path


def create_out_timestamp_dir():
    def init_output_dir_parent():
        output_arg = ParamDict().get_arg("output")
        return EnvVarName().current_path if not output_arg else output_arg

    if ParamDict().get_command() not in [consts.collect_cmd, consts.launch_cmd, consts.analyze_cmd]:
        return RetCode.SUCCESS
    if ParamDict().get_command() == consts.collect_cmd and ParamDict().get_arg("run_mode") == STACKTRACE:
        return RetCode.SUCCESS

    output_dir = init_output_dir_parent()
    if not os.access(output_dir, os.W_OK):
        log_error("no write permission to asys output root directory: {}.".format(output_dir))
        return RetCode.PERMISSION_FAILED

    utc_dt = datetime.now(timezone.utc) # UTC time
    dir_name = 'asys_output_' + utc_dt.astimezone().strftime('%Y%m%d%H%M%S%f')[:-3]

    global _asys_output_path
    _asys_output_path = os.path.abspath(os.path.join(output_dir, dir_name))
    if not f.create_dir(_asys_output_path):
        return RetCode.ARG_CREATE_DIR_FAILED
    ParamDict().asys_output_timestamp_dir = _asys_output_path
    log_info("asys output directory: {0}".format(_asys_output_path))
    return RetCode.SUCCESS


def is_hexadecimal(value):
    try:
        int(value, 16)
        return True
    except ValueError:
        return False


def str_to_hex(str_number):
    return int(str_number, 16)


def int_to_hex(value):
    return hex(int(value, 16))


def get_target_cnt(dir_path):
    """
    Counts the number of bin files in the trace folder.
    """
    count = 0
    atrace_dirs = f.walk_dir(dir_path)
    if not atrace_dirs:
        return count
    for dirs, _, files in atrace_dirs:
        for _ in files:
            count += 1
    return count


def out_progress_bar(count, num):
    """
    Show progress bar
    """
    if count == 0:
        return
    sys.stdout.write("\r")
    sys.stdout.write("Parse progress: {:.2f}%: ".format(num/count * 100))
    sys.stdout.write("\r")
    sys.stdout.flush()


def get_cann_log_path(log_type):
    """
    get trace or cann log path from env
    """
    env_var = EnvVarName()
    if log_type == CANN_LOG_NAME:
        if env_var.process_log_path:
            return env_var.process_log_path, "${ASCEND_PROCESS_LOG_PATH}"

    if env_var.work_path:
        return os.path.join(env_var.work_path, log_type), f"${{ASCEND_WORK_PATH}}/{log_type}"
    return os.path.join(env_var.home_path, "ascend", log_type), f"${{HOME}}/ascend/{log_type}"
