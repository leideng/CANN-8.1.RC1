#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import os
import itertools

from common import FileOperate as f
from common import log_warning, log_error
from common.file_operate import MOVE_MODE


__all__ = ["collect_device_logs"]


def collect_device(source_root_dir, output_root_dir, log_types, feature):
    slog_path = os.path.join(source_root_dir, "slog")
    dev_os_dirs = f.list_dir(slog_path)
    if not dev_os_dirs:
        return False
    target_dict = {
        "device-app": "application",
        "device-os": "system",
        "device-": "firmware",
        "event": "system"
    }
    ret = True
    for dev_os_id, log_type in itertools.product(dev_os_dirs, log_types):
        device_type_path = os.path.join(slog_path, dev_os_id, log_type)
        device_dirs = f.list_dir(device_type_path)
        if not device_dirs:
            continue
        for device_dir in device_dirs:
            if not device_dir.startswith(feature):
                continue
            device_source = os.path.join(slog_path, dev_os_id, log_type, device_dir)
            if log_type:
                device_target = os.path.join(output_root_dir, "dfx", "log", "device",
                                             dev_os_id, target_dict[feature],  log_type, device_dir)
            else:
                device_target = os.path.join(output_root_dir, "dfx", "log", "device",
                                             dev_os_id, target_dict[feature],  "debug", device_dir)
            ret = ret and f.collect_dir(device_source, device_target, MOVE_MODE)
    return ret


def collect_messages(source_root_dir, output_root_dir):
    message_dir = os.path.join(source_root_dir, "message")
    dev_os_dirs = f.list_dir(message_dir)
    if not dev_os_dirs:
        return False
    ret = True
    for dev_os_id in dev_os_dirs:
        message_target = os.path.join(output_root_dir, "dfx", "log", "device", dev_os_id, "system", "message")
        message_source =  os.path.join(message_dir, dev_os_id)
        ret = ret and f.collect_dir(message_source, message_target, MOVE_MODE)
    return ret


def collect_stackcore(source_root_dir, output_root_dir):
    stackcore_source = os.path.join(source_root_dir, "stackcore")
    stackcore_target = os.path.join(output_root_dir, "dfx", "stackcore")
    return f.collect_dir(stackcore_source, stackcore_target, MOVE_MODE)


def collect_bbox(source_root_dir, output_root_dir):
    bbox_source = os.path.join(source_root_dir, "hisi_logs")
    bbox_target = os.path.join(output_root_dir, "dfx", "bbox")
    return f.collect_dir(bbox_source, bbox_target, MOVE_MODE)


def collect_host_driver(source_root_dir, output_root_dir):
    host_driver_path = os.path.join(source_root_dir, "slog", "host")
    host_driver_dirs = f.list_dir(host_driver_path)
    if not host_driver_dirs:
        return False
    ret = True
    host_driver_log_target = os.path.join(output_root_dir, "dfx", "log", "host", "driver")
    for host_driver_log in host_driver_dirs:
        host_driver_log_source = os.path.join(host_driver_path, host_driver_log)
        ret = ret and f.collect_file_to_dir(host_driver_log_source, host_driver_log_target, MOVE_MODE)
    f.remove_dir(host_driver_path)
    return ret


def collect_slogd(source_root_dir, output_root_dir):
    slog_path = os.path.join(source_root_dir, "slog")
    slog_dirs = f.list_dir(slog_path)
    if not slog_dirs:
        return False
    ret = True
    for slog_dir in slog_dirs:
        slogd_log_source = os.path.join(slog_path, slog_dir, "slogd")
        if f.check_dir(slogd_log_source):
            slogd_log_target = os.path.join(output_root_dir, "dfx", "log", "device", slog_dir, "slogd")
            ret = ret and f.collect_dir(slogd_log_source, slogd_log_target, MOVE_MODE)
    return ret


def collect_device_app(source_root_dir, output_root_dir):
    log_types = ["debug", "run", "security"]
    feature = "device-app"
    return collect_device(source_root_dir, output_root_dir, log_types, feature) 


def collect_device_os(source_root_dir, output_root_dir):
    log_types = ["debug", "run", "security"]
    feature = "device-os"
    return collect_device(source_root_dir, output_root_dir, log_types, feature)


def collect_device_id(source_root_dir, output_root_dir):
    log_types = ["debug", "run", "security", ""]
    feature = "device-"
    return collect_device(source_root_dir, output_root_dir, log_types, feature)


def collect_event(source_root_dir, output_root_dir):
    log_types = ["run"]
    feature = "event"
    return collect_device(source_root_dir, output_root_dir, log_types, feature)


def collect_device_logs(source_root_dir, output_root_dir):
    if not source_root_dir:
        log_error("msnpureport output directory: {} is not exist".format(source_root_dir))
        return False
    err = 0  # err equals to zero denotes no errors
    message_ret = collect_messages(source_root_dir, output_root_dir)
    if not message_ret:
        log_warning("collect device messages failed.")
        err += 1
    stackcore_ret = collect_stackcore(source_root_dir, output_root_dir)
    if not stackcore_ret:
        log_warning("collect device stackcore failed.")
        err += 1
    bbox_ret = collect_bbox(source_root_dir, output_root_dir)
    if not bbox_ret:
        log_warning("collect device bbox failed.")
        err += 1
    host_driver_ret = collect_host_driver(source_root_dir, output_root_dir)
    if not host_driver_ret:
        log_warning("collect host driver log failed.")
        err += 1
    slogd_ret = collect_slogd(source_root_dir, output_root_dir)
    if not slogd_ret:
        log_warning("collect device slogd logs failed.")
        err += 1   
    device_app_ret = collect_device_app(source_root_dir, output_root_dir)
    if not device_app_ret:
        log_warning("collect device device_app logs failed.")
        err += 1
    device_os_ret = collect_device_os(source_root_dir, output_root_dir)
    if not device_os_ret:
        log_warning("collect device device_os logs failed.")
        err += 1
    device_id_ret = collect_device_id(source_root_dir, output_root_dir)
    if not device_id_ret:
        log_warning("collect device device_id logs failed.")
        err += 1
    event_ret = collect_event(source_root_dir, output_root_dir)
    if not event_ret:
        log_warning("collect device event logs failed.")
        err += 1
    return err == 0
