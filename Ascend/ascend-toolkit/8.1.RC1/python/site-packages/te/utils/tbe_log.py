#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019-2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
TBE common log
"""
import os


class TbeLog:
    """
    TBE common log
    """
    def __init__(self, name):
        self.use_slog = False
        try:
            from te.utils.AscendLog import LOGGER
            self.slog = LOGGER
            self.use_slog = True
        except ImportError:
            import logging
            self.log = logging.getLogger(name)
            level = (int(os.getenv("GLOBAL_LOG_LEVEL", "3")) + 1) * 10
            self.log.setLevel(level)

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = object.__new__(cls)
        return cls._instance

    def info(self, log_msg):
        """
        info level
        """
        if self.use_slog:
            self.slog.info(self.slog.module.tbe, log_msg)
        else:
            self.log.info(log_msg)

    def debug(self, log_msg):
        """
        debug level
        """
        if self.use_slog:
            self.slog.debug(self.slog.module.tbe, log_msg)
        else:
            self.log.debug(log_msg)

    def warning(self, log_msg):
        """
        warn level
        """
        if self.use_slog:
            self.slog.warn(self.slog.module.tbe, log_msg)
        else:
            self.log.warning(log_msg)

    def error(self, log_msg):
        """
        error level
        """
        if self.use_slog:
            self.slog.error(self.slog.module.tbe, log_msg)
        else:
            self.log.error(log_msg)
