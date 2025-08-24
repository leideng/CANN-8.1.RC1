#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import os

from params import ParamDict
from common import consts
from common import log_warning
from common import FileOperate as f
from common.file_operate import COPY_MODE
from common.path import get_log_conf_path

__all__ = ["collect_rc_logs"]


def collect_messages(output_root_dir):
    message_root_dir = os.path.join(os.sep, "var", "log")
    message_file_name = "syslog" if f.check_file(os.path.join(message_root_dir, "syslog")) else "messages"
    message_source = os.path.join(message_root_dir, message_file_name)
    message_target = os.path.join(output_root_dir, "dfx", "log", "message")
    return f.collect_file_to_dir(message_source, message_target, COPY_MODE)


def collect_stackcore(output_root_dir):
    coredump_path = os.path.join(os.sep, "var", "log", "npu", "coredump")
    stackcore_target = os.path.join(output_root_dir, "dfx", "stackcore")
    ret = True
    if not f.list_dir(coredump_path):
        log_warning("no files or dirs in {0}".format(coredump_path))
        return False
    stackcore_source_dirs = os.walk(coredump_path)
    for stackcore_path, _, stackcore_files in stackcore_source_dirs:
        for stackcore_file in stackcore_files:
            stackcore_source_name = os.path.join(stackcore_path, stackcore_file)
            if ParamDict().get_command() == consts.launch_cmd and \
                    f".{str(ParamDict().get_task_pid())}." not in stackcore_file:
                continue
            ret = ret and f.collect_file_to_dir(stackcore_source_name,
                                                stackcore_path.replace(coredump_path, stackcore_target), COPY_MODE)
    return ret


def collect_install(output_root_dir):
    install_source = os.path.join(os.sep, "var", "log", "ascend_seclog", "ascend_install.log")
    install_target = os.path.join(output_root_dir, "dfx", "log", "install")
    return f.collect_file_to_dir(install_source, install_target, COPY_MODE)


def collect_bbox(output_root_dir):
    bbox_source = get_log_conf_path("bbox")
    bbox_target = os.path.join(output_root_dir, "dfx", "bbox")
    return f.collect_dir(bbox_source, bbox_target, COPY_MODE)


def collect_device_os(slog_path, output_root_dir):
    ret = True
    log_types = ["debug", "run", "security"]
    task_pid = ParamDict().get_task_pid()
    for log_type in log_types:
        dev_os_path = os.path.join(slog_path, log_type)
        device_os_dirs = f.list_dir(dev_os_path)
        if not device_os_dirs:
            log_warning("no files or dirs in {0}".format(dev_os_path))
            return False
        for device_os_dir_name in device_os_dirs:
            if device_os_dir_name == "device-os":
                device_os_log_source = os.path.join(slog_path, log_type, device_os_dir_name)
                device_os_log_target = os.path.join(output_root_dir, "dfx", "log", "system",
                                                    log_type, device_os_dir_name)
                ret = ret and f.collect_dir(device_os_log_source, device_os_log_target, COPY_MODE)
            # collect: task_pid is None; launch: task_pid is subprocess pid.
            elif (task_pid is None and device_os_dir_name.startswith("device-app")) or \
                    device_os_dir_name.endswith(str(task_pid)):
                device_os_log_source = os.path.join(slog_path, log_type, device_os_dir_name)
                device_os_log_target = os.path.join(output_root_dir, "dfx", "log", "cann",
                                                    log_type, device_os_dir_name)
                ret = ret and f.collect_dir(device_os_log_source, device_os_log_target, COPY_MODE)
    return ret


def collect_device_id(slog_path, output_root_dir):
    dev_os_dirs = f.list_dir(slog_path)
    if not dev_os_dirs:
        log_warning("no files or dirs in {0}".format(slog_path))
        return False
    ret = True
    for dev_os_id in dev_os_dirs:
        if "device-" not in dev_os_id:
            continue
        device_id_log_source = os.path.join(slog_path, dev_os_id)
        device_id_log_target = os.path.join(output_root_dir, "dfx", "log", "firmware", dev_os_id)
        ret = ret and f.collect_dir(device_id_log_source, device_id_log_target, COPY_MODE)
    return ret


def collect_rc_logs(output_root_dir):
    all_func = [collect_messages, collect_install, collect_bbox, collect_stackcore]
    for func in all_func:
        ret = func(output_root_dir)
        if not ret:
            msg_name = func.__name__.split('_')[-1]
            log_warning("collect {} logs failed.".format(msg_name))

    slog_path = get_log_conf_path("slog")
    device_os_ret = collect_device_os(slog_path, output_root_dir)
    if not device_os_ret:
        log_warning("collect device device_os logs failed.")
    device_id_ret = collect_device_id(slog_path, output_root_dir)
    if not device_id_ret:
        log_warning("collect device device_id logs failed.")