#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import os

from common import FileOperate as f
from params import ParamDict
from common import consts, get_cann_log_path
from common.const import CANN_LOG_NAME, ATRACE_LOG_NAME
from common import log_warning
from common.file_operate import COPY_MODE, MOVE_MODE

__all__ = ["collect_host_logs"]


def collect_messages(output_root_dir):
    root_path = os.sep
    message_root_dir = os.path.join(root_path, "var", "log")
    message_file_name = "syslog" if f.check_file(os.path.join(message_root_dir, "syslog")) else "messages"
    message_source = os.path.join(message_root_dir, message_file_name)
    message_target = os.path.join(output_root_dir, "dfx", "log", "host", "message")
    return f.collect_file_to_dir(message_source, message_target, COPY_MODE)


def collect_atrace_logs(output_root_dir):
    atrace_log_path, _ = get_cann_log_path(ATRACE_LOG_NAME)
    if f.check_dir(atrace_log_path):
        atrace_target = os.path.join(output_root_dir, "dfx", "atrace")
        return f.collect_dir(atrace_log_path, atrace_target, COPY_MODE)
    return False


def collect_cann_logs(output_root_dir):
    cann_log_path, env_path_name = get_cann_log_path(CANN_LOG_NAME)
    log_types = ["debug", "run", "security"]
    ret = True
    for log_type in log_types:
        cann_source = os.path.join(cann_log_path, log_type)
        cann_target = os.path.join(output_root_dir, "dfx", "log", "host", "cann", log_type)
        if not os.path.exists(cann_source):
            log_warning("No file exists in %s/%s" % (env_path_name, log_type))
        if ParamDict().get_command() == consts.launch_cmd:
            ret = ret and f.collect_dir(cann_source, cann_target, MOVE_MODE)
        else:
            ret = ret and f.collect_dir(cann_source, cann_target, COPY_MODE)
    return ret


def collect_install_logs(output_root_dir):
    root_path = os.sep
    install_source = os.path.join(root_path, "var", "log", "ascend_seclog", "ascend_install.log")
    install_target = os.path.join(output_root_dir, "dfx", "log", "host", "install")
    return f.collect_file_to_dir(install_source, install_target, COPY_MODE)


def collect_host_logs(output_root_dir):
    err = 0 # err equals to zero denotes no errors
    message_ret = collect_messages(output_root_dir)
    if not message_ret:
        log_warning("collect host messages failed.")
        err += 1
    cann_ret = collect_cann_logs(output_root_dir)
    if not cann_ret:
        log_warning("collect host cann logs failed.")
        err += 1
    atrace_ret = collect_atrace_logs(output_root_dir)
    if not atrace_ret:
        log_warning("collect host atrace failed.")
        err += 1
    install_ret = collect_install_logs(output_root_dir)
    if not install_ret:
        log_warning("collect install logs failed.")
        err += 1
    return err == 0
