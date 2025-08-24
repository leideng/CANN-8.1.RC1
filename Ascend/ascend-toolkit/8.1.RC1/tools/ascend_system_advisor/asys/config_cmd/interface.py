#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import ctypes
import sys

from common import log_info, log_error
from common.const import NOT_SUPPORT
from view import generate_report


def get_stress_detect_config(device_id, device_obj):
    """config get stress_detect, aic & bus volt"""

    config_data = []
    aic_info = device_obj.get_device_aic_info(device_id)
    if aic_info[1] != NOT_SUPPORT:
        config_data.append(["AI Core Voltage (MV)", aic_info[1]])
    bus_info = device_obj.get_device_bus_info(device_id)
    if bus_info[0] != NOT_SUPPORT:
        config_data.append(["Bus Voltage (MV)", bus_info[0]])
    # get ai core volt & bus volt, all failed
    if not config_data:
        log_error(f"Configuration unsuccessfully get, on device {device_id}.")
        return False

    table_header = [[f"Device ID: {device_id}", "CURRENT CONFIGURATION"]]
    table_data = {"none": config_data}
    ret_str = generate_report(table_header, table_data)
    sys.stdout.write(ret_str)  # print screen
    return True


def restore_stress_detect_config(device_id, device_obj):
    """config restore stress_detect, aic & bus volt"""
    try:
        ret_code = device_obj.ascend_ml.AmlStressRestore(ctypes.c_int32(device_id))
    except AttributeError:
        ret_code = None

    if ret_code != 0:
        log_error(f"Configuration unsuccessfully restore, on device {device_id}.")
        return False

    log_info(f"Configuration successfully restore, on device {device_id}.")
    return True
