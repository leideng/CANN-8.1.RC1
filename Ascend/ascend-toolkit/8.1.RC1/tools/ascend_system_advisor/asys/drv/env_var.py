# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import os

from common import FileOperate as f


class EnvVarName:
    def __init__(self):
        self.process_log_path = self._set_env_path("ASCEND_PROCESS_LOG_PATH")
        self.npu_collect_path = self._set_env_path("NPU_COLLECT_PATH")
        self.dump_graph_path = self._set_env_path("DUMP_GRAPH_PATH")
        self.work_path = self._set_env_path("ASCEND_WORK_PATH")
        self.cache_path = self._set_env_path("ASCEND_CACHE_PATH")
        self.opp_path = self._set_env_path("ASCEND_OPP_PATH")
        self.custom_opp_path = self._set_env_path("ASCEND_CUSTOM_OPP_PATH")
        self.current_path = os.getcwd()
        self.home_path = os.path.expanduser("~")

    @staticmethod
    def _set_env_path(env_var):
        env_path = os.getenv(env_var)
        if env_path and f.check_dir(env_path):
            ret = os.path.abspath(env_path)
        else:
            ret = None
        return ret
