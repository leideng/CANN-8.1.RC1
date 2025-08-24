#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import os
import re
import sys
import threading
from datetime import datetime, timezone

from common import log_error, log_warning, log_info, open_log, close_log
from common.const import RetCode, UNKNOWN
from common.const import HBM_MIN_TIMEOUT, CPU_MIN_TIMEOUT, DETECT_MAX_TIMEOUT
from common import DeviceInfo
from common.cmd_run import run_linux_cmd
from params.param_dict import ParamDict
from diagnose.interface import run_diagnose_910b, run_diagnose_910_93
from view.table import generate_report
from view.progress_display import waiting
from drv import EnvVarName


HBM_MODE = "hbm_detect"
CPU_MODE = "cpu_detect"
WARN_STATUS = "Warn"


class AsysDiagnose():
    """"""
    def __init__(self):
        self.finish_flag = False

    @staticmethod
    def __get_hbm_table_data(device_id, ret):
        link_symbol = ", "
        devices_ecc = []

        if device_id is False:
            devices_ret = [ret[key][0] for key in sorted(ret.keys())]
            devices_ecc = [ret[key][1] for key in sorted(ret.keys())]
            if ("Pass" in devices_ret and "Warn" in devices_ret) or len(devices_ret) == 1:
                devices_str = ", ".join(devices_ret)
            else:
                devices_str = f"{devices_ret[0]} - All"
        else:
            devices_str = ret[device_id][0]

        hbm_table = "HBM Detect"
        if device_id is False and len(devices_ecc) != 1:
            devices_hbm = ("(" + link_symbol.join(devices_ecc) + ")")
            ret_data_str = [[hbm_table, devices_str], ["", devices_hbm]]
        elif len(devices_ecc) == 1:
            devices_str += ("(" + link_symbol.join(devices_ecc) + ")")
            ret_data_str = [[hbm_table, devices_str]]
        else:
            devices_str += ("(" + ret[device_id][1] + ")")
            ret_data_str = [[hbm_table, devices_str]]

        return ret_data_str

    @staticmethod
    def __get_other_table_data(device_id, ret):
        if device_id is False:
            # without '-d', displays information about all devices.
            devices_ret = [ret[key] for key in sorted(ret.keys())]
            if len(set(devices_ret)) > 1 or len(devices_ret) == 1:
                devices_str = ", ".join(devices_ret)
            else:
                devices_str = f"{devices_ret[0]} - All"
        else:
            # with '-d', displays information about the input device.
            devices_str = ret[device_id]

        return devices_str

    def print_save(self, device_id, ret, run_mode):
        """print screen & save file"""
        if device_id is False:
            # without '-d', displays information about all devices.
            table_header = [[f"Group of {len(ret)} Device", "Diagnostic Result"]]
        else:
            # with '-d', displays information about the input device.
            table_header = [[f"Device ID: {device_id}", "Diagnostic Result"]]

        if run_mode == HBM_MODE:
            ret_data_str = self.__get_hbm_table_data(device_id, ret)
            table_data = {" Hardware ": ret_data_str}
        elif run_mode == CPU_MODE:
            ret_data_str = self.__get_other_table_data(device_id, ret)
            table_data = {" Hardware ": [["CPU Detect", ret_data_str]]}
        else:
            ret_data_str = self.__get_other_table_data(device_id, ret)
            table_data = {" Performance ": [["Stress Detect", ret_data_str]]}
        ret_str = generate_report(table_header, table_data)
        sys.stdout.write(ret_str)  # print screen

        # save result to file
        output_path = ParamDict().get_arg("output")
        utc_dt = datetime.now(timezone.utc)  # UTC time
        dir_name = utc_dt.astimezone().strftime('%Y%m%d%H%M%S%f')[:-3]
        if output_path is not False:
            try:
                output_file = os.path.join(ParamDict().get_arg("output"), f"diagnose_result_{dir_name}.txt")
                with open(output_file, "w", encoding="utf8") as file:
                    file.write(ret_str)
                open_log()
                log_info(f"output file: {os.path.abspath(output_file)}")
                close_log()
            except Exception as e:
                log_error(f"Failed to save result: {e}.")

    @staticmethod
    def _check_support():
        # check VMs and docker
        if not run_linux_cmd("systemd-detect-virt", "none"):
            log_error("The diagnose command cannot be executed on VMs and docker.")
            return False

        # username
        if os.getuid() != 0:  # 0 -> administrator
            log_error("The diagnose command must be executed as the root user.")
            return False

        run_mode = ParamDict().get_arg("run_mode")
        if run_mode == "stress_detect":
            # check opp_kernel, ${install_path}/latest/opp_kernel
            opp_path = EnvVarName().opp_path
            if not (opp_path and os.path.isfile(os.path.join(opp_path, "..", "opp_kernel", "version.info"))):
                log_error("The diagnose command can be executed only after the opp_kernel is installed.")
                return False

        timeout = ParamDict().get_arg("timeout")
        if run_mode == HBM_MODE and timeout is not False:
            if timeout < HBM_MIN_TIMEOUT or timeout > DETECT_MAX_TIMEOUT:
                log_error(f"The value of timeout must be in the range of [{HBM_MIN_TIMEOUT}, {DETECT_MAX_TIMEOUT}].")
                return False

        if run_mode == CPU_MODE and timeout is not False:
            if timeout < CPU_MIN_TIMEOUT or timeout > DETECT_MAX_TIMEOUT:
                log_error(f"The value of timeout must be in the range of [{CPU_MIN_TIMEOUT}, {DETECT_MAX_TIMEOUT}].")
                return False
        return True

    @staticmethod
    def get_diagnose_devices_chip_info(device_obj, device_id, devices_num):
        diagnose_devices = []
        chip_info = UNKNOWN
        if device_id is False:
            for i in range(devices_num):
                _chip_info = device_obj.get_chip_info(i)
                if not (re.search(r"910B\d", _chip_info) or "910_93" in _chip_info):
                    log_error(f"The diagnose command does not support on device_{i}: {_chip_info}.")
                    continue
                diagnose_devices.append(i)
                chip_info = _chip_info
        else:
            _chip_info = device_obj.get_chip_info(device_id)
            if not (re.search(r"910B\d", _chip_info) or "910_93" in _chip_info):
                log_error(f"The diagnose command does not support {_chip_info}.")
            else:
                chip_info = _chip_info
                diagnose_devices = [device_id]
        return diagnose_devices, chip_info

    def diagnose(self):
        """diagnose cmd main"""
        if not self._check_support():
            return False

        device_obj = DeviceInfo()
        devices_num = device_obj.get_device_count()
        if devices_num == 0:
            return False

        device_id = ParamDict().get_arg("device_id")
        if int(device_id) >= devices_num:
            log_error("'-d' value should be in [0, {}), input {}".format(devices_num, device_id))
            return False

        # load dll: libascend_ml.so
        if device_obj.ascend_ml == RetCode.FAILED:
            return False

        diagnose_devices, chip_info = self.get_diagnose_devices_chip_info(device_obj, device_id, devices_num)
        if not diagnose_devices or chip_info == UNKNOWN:
            return False

        # collect launch without "-r", run_mode is False
        run_mode = ParamDict().get_arg("run_mode")
        t = threading.Thread(target=self.wait_view, daemon=True)
        t.start()
        # Multi-thread parallel execution
        if "910_93" in chip_info:
            ret = run_diagnose_910_93(device_obj, diagnose_devices, run_mode)
        else:
            ret = run_diagnose_910b(device_obj, diagnose_devices, run_mode)

        # ret add not support device
        if device_id is False:
            for i in range(devices_num):
                if i in diagnose_devices:
                    continue
                ret[i] = [WARN_STATUS, "0"] if run_mode == HBM_MODE else WARN_STATUS

        if WARN_STATUS in ret.values():
            log_warning("Diagnosis results have failed, please analyze aml logs")
        self.finish_flag = True
        t.join()
        # screen print & save ret to file
        self.print_save(device_id, ret, run_mode)
        return True

    def wait_view(self):
        while not self.finish_flag:
            waiting()
            continue
