#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import os
import sys

from common import DeviceInfo
from common import log_error
from common import consts
from common.const import UNKNOWN, NONE
from params import ParamDict
from view import generate_report


class AsysHealth:
    """device health check"""
    def __init__(self):
        self.device = DeviceInfo()

    @staticmethod
    def _get_highest_status(ret):
        critical = "Critical"
        alarm = "Alarm"
        warn = "Warning"
        healthy = "Healthy"
        all_device_status = [info[0] for info in ret.values()]

        if UNKNOWN in all_device_status:
            return UNKNOWN
        if critical in all_device_status:
            return critical
        if alarm in all_device_status:
            return alarm
        if warn in all_device_status:
            return warn
        if healthy in all_device_status:
            return healthy
        return UNKNOWN

    @staticmethod
    def _save_file(ret):
        """save file"""
        save_str = ""
        # Example: err_info = [health, [[error_code, error_msg], [error_code, error_msg] ...]]
        for device_id in sorted(ret.keys()):
            err_info = ret[device_id]
            table_header = [
                [f"Device ID: {device_id}", f"Overall Health: {err_info[0]}"],
                ["", f"ErrorCode Num: {len(err_info[1])}"]
            ]
            table_data = {NONE: err_info[1]}
            save_str += generate_report(table_header, table_data, split_line=True)
        try:
            output_file = os.path.join(ParamDict().asys_output_timestamp_dir, "health_result.txt")
            with open(output_file, "w", encoding="utf8") as file:
                file.write(save_str)
        except Exception as e:
            log_error(f"Failed to save result: {e}.")

    def run_health_check(self, diagnose_devices):
        """Multi-thread parallel execution"""
        ret = {}
        for device_id in diagnose_devices:
            # status -> "Healthy"/"Warning"/"Alarm"/"Critical"/"Unknown"
            status = self.device.get_device_health(device_id)
            # Example: err_info = [[error_code, error_msg], [error_code, error_msg] ...]
            err_info = self.device.get_device_errorcode(device_id)
            ret[device_id] = [status, err_info]
        return ret

    def _print_screen(self, device_id, ret):
        """print screen"""
        if device_id is False:
            # without '-d', displays brief information about all devices.
            highest_status = self._get_highest_status(ret)
            table_header = [[f"Group of {len(ret)} Device", f"Overall Health: {highest_status}"]]
            table_data = {NONE: [[f"Device ID: {idx}", ret[idx][0]] for idx in sorted(ret.keys())]}
        else:
            # with '-d', displays detailed information about the input device.
            table_header = [
                [f"Device ID: {device_id}", f"Overall Health: {ret[device_id][0]}"],
                ["", f"ErrorCode Num: {len(ret[device_id][1])}"]
            ]
            table_data = {NONE: ret[device_id][1]}
            # Only the first five records are displayed on the screen.
            if len(table_data[NONE]) > 5:
                table_data[NONE] = table_data[NONE][:5]
                table_data[NONE].append(["......", "......"])
        print_str = generate_report(table_header, table_data, split_line=True)
        sys.stdout.write(print_str)

    def health(self):
        """health check main"""
        devices_num = self.device.get_device_count()
        if devices_num is None:
            return False

        device_id = ParamDict().get_arg("device_id")
        if device_id is False:
            diagnose_devices = [i for i in range(devices_num)]
        else:
            if device_id >= devices_num:
                log_error("'-d' value should be in [0, {}), input {}".format(devices_num, device_id))
                return False
            diagnose_devices = [device_id]

        # Example: ret = {device_id: [health, [[error_code, error_msg], [error_code, error_msg] ...]]}
        ret = self.run_health_check(diagnose_devices)

        if ParamDict().get_command() == consts.health_cmd:
            self._print_screen(device_id, ret)
        else:
            # only collect or launch save file
            self._save_file(ret)

        return True
