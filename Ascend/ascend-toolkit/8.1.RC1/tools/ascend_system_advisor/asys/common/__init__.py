#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

from common.log import log_debug, log_info, log_warning, log_error, close_log, open_log
from common.const import consts, RetCode, Singleton, STACKTRACE
from common.file_operate import FileOperate
from common.cmd_run import run_command, run_msnpureport_cmd, run_linux_cmd, popen_run_cmd
from common.path import get_project_conf, get_ascend_home, get_log_conf_path
from common.device import DeviceInfo
from common.compress_output_dir import compress_output_dir_tar
from common.task_common import get_cann_log_path
