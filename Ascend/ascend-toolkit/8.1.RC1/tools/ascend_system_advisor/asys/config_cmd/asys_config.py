#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import os
import re

from common import log_error
from common import DeviceInfo
from common.cmd_run import run_linux_cmd
from params import ParamDict
from config_cmd.interface import get_stress_detect_config, restore_stress_detect_config


class AsysConfig:
    def __init__(self):
        self.is_get_mode = ParamDict().get_arg("get")
        self.is_restore_mode = ParamDict().get_arg("restore")
        self.device_obj = DeviceInfo()
        self.device_id = int(ParamDict().get_arg("device_id"))
        self.options = []

    def _check_support(self):
        # not support VMs and docker
        if not run_linux_cmd("systemd-detect-virt", "none"):
            log_error("The config command cannot be executed on VMs and docker.")
            return False
        # restore mode only support root
        if os.getuid() != 0:
            log_error("The config --restore command must be executed as the root user.")
            return False

        # without device
        devices_num = self.device_obj.get_device_count()
        if devices_num == 0:
            return False
        # device_id out of range
        if self.device_id >= devices_num:
            log_error(f"'-d' value should be in [0, {devices_num}), input {self.device_id}.")
            return False
        # only support 910B & 910_93
        chip_info = self.device_obj.get_chip_info(self.device_id)
        if not (re.search(r"910B\d", chip_info) or "910_93" in chip_info):
            log_error(f"The config command does not support {chip_info}.")
            return False

        # check run mode
        if not (self.is_get_mode or self.is_restore_mode):
            log_error("The config command requires either the --get or --restore argument.")
            return False
        # check options
        all_options = ["stress_detect"]
        current_options = []
        for opt in all_options:
            if ParamDict().get_arg(opt):
                current_options.append(opt)
        if not current_options:
            log_error("At least one configuration option is required.")
            return False
        self.options = current_options

        return True

    def _run_get_mode(self):
        ret = True
        for opt in self.options:
            if opt == "stress_detect":
                ret = ret and get_stress_detect_config(self.device_id, self.device_obj)
        return ret

    def _run_restore_mode(self):
        ret = True
        for opt in self.options:
            if opt == "stress_detect":
                ret = ret and restore_stress_detect_config(self.device_id, self.device_obj)
        return ret

    def config(self):
        if not self._check_support():
            return False

        if self.is_get_mode:
            return self._run_get_mode()
        if self.is_restore_mode:
            return self._run_restore_mode()

        return False
