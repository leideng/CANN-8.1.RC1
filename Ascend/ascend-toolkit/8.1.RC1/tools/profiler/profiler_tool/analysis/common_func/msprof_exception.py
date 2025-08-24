#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.

from common_func.common import error


class ProfException(Exception):
    """
    The class for Profiling Exception
    """

    # error code for user:success
    PROF_NONE_ERROR = 0
    # error code for user: error
    PROF_INVALID_PARAM_ERROR = 1
    PROF_INVALID_PATH_ERROR = 2
    PROF_INVALID_CONNECT_ERROR = 3
    PROF_INVALID_EXECUTE_CMD_ERROR = 4
    PROF_INVALID_CONFIG_ERROR = 5
    PROF_INVALID_DISK_SHORT_ERROR = 6
    PROF_INVALID_JSON_ERROR = 7
    PROF_INVALID_DATA_ERROR = 8
    PROF_INVALID_STEP_TRACE_ERROR = 9
    PROF_SYSTEM_EXIT = 10
    PROF_CLUSTER_DIR_ERROR = 11
    PROF_CLUSTER_INVALID_DB = 12
    PROF_DB_RECORD_EXCEED_LIMIT = 13

    def __init__(self: any, code: int, message: str = '', callback: any = error) -> None:
        super().__init__(code, message)
        self.code = code
        self.message = message
        self.callback = callback

    def __str__(self):
        return self.message
