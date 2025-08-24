#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import os
import re
import sys

from common import log_error


__all__ = ["get_project_conf", "get_ascend_home", "get_log_conf_path"]


def get_project_conf():
    project_conf_path = os.path.join(os.path.dirname(sys.argv[0]), "conf")
    if not os.path.exists(project_conf_path):
        project_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        project_conf_path = os.path.join(project_path, "conf")
    return project_conf_path


def get_ascend_home():
    # TOOLCHAIN_HOME -> ${install_path}/latest/toolkit
    toolchain_home = os.getenv("TOOLCHAIN_HOME")
    if toolchain_home:
        latest_path = toolchain_home.split(":")[0]
        return os.path.abspath(os.path.join(latest_path, "../.."))
    else:
        return "/usr/local/Ascend"


def get_log_conf_path(name):
    log_path = None
    if name == "slog":
        log_conf_paths = ["/etc/slog.conf", "/var/log/npu/conf/slog.conf", "/var/log/npu/conf/slog/slog.conf"]
        search_re = "logAgentFileDir=(.+?)\n"
        log_path = "/var/log/npu/slog/"
    elif name == "bbox":
        log_conf_paths = ["/var/bbox.conf", "/etc/bbox.conf", "/var/log/npu/conf/bbox.conf",
                          "/var/log/npu/conf/bbox/bbox.conf"]
        search_re = "MNTN_PATH=(.+?)\n"
        log_path = "/var/log/npu/hisi_logs/"
    else:
        log_error("The input parameter is incorrect.")
        return log_path

    for conf_path in log_conf_paths:
        if not os.path.exists(conf_path):
            continue
        with open(conf_path, "r", encoding="utf8") as conf_f:
            conf_info = conf_f.read()

        log_path_f = re.search(search_re, conf_info)
        if log_path_f is not None:
            log_path = log_path_f.groups()[0]

    return log_path
