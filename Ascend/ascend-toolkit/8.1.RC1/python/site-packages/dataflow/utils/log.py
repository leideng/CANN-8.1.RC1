#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright 2024-2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

__all__ = [
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "get_logger",
    "log",
    "debug",
    "info",
    "warning",
    "error",
    "set_log_level",
    "get_log_level",
]

import logging
import os
import sys
import threading

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR

_global_logger = None
_global_logger_lock = threading.Lock()

DATA_FLOW_LOG_FORMAT = (
    "[%(levelname)s] %(name)s(%(process)d,%(processName)s):%(asctime)s.%(msecs)03d "
    "[%(filename)s:%(lineno)d]%(thread)d %(funcName)s: %(message)s"
)

_log_level = {
    "0": logging.DEBUG,
    "1": logging.INFO,
    "2": logging.WARNING,
    "3": logging.ERROR,
}


def get_logger():
    global _global_logger
    if _global_logger:
        return _global_logger
    with _global_logger_lock:
        if _global_logger:
            return _global_logger
        logger = logging.getLogger("DATAFLOW")
        log_level = os.getenv("ASCEND_GLOBAL_LOG_LEVEL")
        logger.setLevel(_log_level.get(log_level, logging.ERROR))
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(DATA_FLOW_LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
        _global_logger = logger
        if log_level is not None:
            logger.debug("get ASCEND_GLOBAL_LOG_LEVEL=%s", log_level)
        return _global_logger


def log(level, msg, *args, **kwargs):
    get_logger().log(level, msg, *args, **kwargs)


def debug(msg, *args, **kwargs):
    get_logger().debug(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    get_logger().error(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    get_logger().info(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    get_logger().warning(msg, *args, **kwargs)


def set_log_level(log_level):
    get_logger().setLevel(log_level)


def get_log_level():
    return get_logger().getEffectiveLevel()
