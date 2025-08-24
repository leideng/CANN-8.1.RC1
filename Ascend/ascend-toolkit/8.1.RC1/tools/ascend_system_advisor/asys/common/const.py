#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import enum

PROCESSES_NUMBER = 4
DEVICE_ID_MIN = 0
DEVICE_ID_MAX = 63
MAX_CHAR_LINE = 100

MEMORY_FREQUENCY = 1
HBM_FREQUENCY = 6
CONTROL_CPU_FREQUENCY = 2
AI_CORE_FREQUENCY = 7

MEM_BANDWIDTH_USE = 5
AI_CORE_USE = 2
AI_CPU_USE = 3
CONTROL_CPU_USE = 4

NOT_SUPPORT = "-"
UNKNOWN = "Unknown"
NONE = "none"

CANN_LOG_NAME = "log"
ATRACE_LOG_NAME = "atrace"

REG_OFF = 0
REG_THREAD = 1
REG_STACK = 2

HBM_MIN_TIMEOUT = 0  # detection side
CPU_MIN_TIMEOUT = 1  # detection 1s
DETECT_MAX_TIMEOUT = 604800  # one week
DETECT_DEFAULT_TIMEOUT = 600

CPU_DETECT_ERROR_CODE_MIN = 500000
CPU_DETECT_ERROR_CODE_MAX = 599999

ADDR_LEN_HEX = 18  # 0x 0000 0000 0000 0000
ADDR_BIT_LEN = 16  # 0000 0000 0000 0000

GDB_LAYER_MAX = 32

# sigrtmin
SIGRTMIN = 34
STACKTRACE = "stacktrace"

GET_DEVICES_INFO_TIMEOUT = 10


class CannPkg:
    firmware = "firmware"
    driver = "driver"
    runtime = "runtime"
    compiler = "compiler"
    fwkplugin = "fwkplugin"
    opp = "opp"
    toolkit = "toolkit"
    aoe = "aoe"
    hccl = "hccl"
    ncs = "ncs"
    opp_kernel = "opp_kernel"

    @classmethod
    def get_all_pkg_list(cls):
        return [cls.firmware, cls.driver, cls.runtime, cls.compiler, cls.fwkplugin, cls.opp, cls.toolkit,
                cls.aoe, cls.hccl, cls.ncs, cls.opp_kernel]


class Singleton(type):
    """ Singleton class """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in Singleton._instances:
            Singleton._instances[cls] = super().__call__(*args, **kwargs)
        return Singleton._instances[cls]

    def clear(cls):
        try:
            del Singleton._instances[cls]
        except KeyError:
            pass


class RetCode(enum.Enum):
    SUCCESS = 0
    FAILED = 1
    ARG_PATH_INVALID = 2
    ARG_EMPTY_STRING = 3
    ARG_SAPCE_STRING = 4
    ARG_ILLEGAL_STRING = 5
    ARG_NO_EXIST_DIR = 6
    ARG_NO_EXECUTABLE = 7
    ARG_NO_WRITABLE_PERMISSION = 8
    ARG_CREATE_DIR_FAILED = 9
    READ_FILE_FAILED = 10
    PERMISSION_FAILED = 11


class Constants:
    @property
    def help_cmd(self):
        return 'help'

    @property
    def collect_cmd(self):
        return 'collect'

    @property
    def launch_cmd(self):
        return 'launch'

    @property
    def info_cmd(self):
        return 'info'

    @property
    def diagnose_cmd(self):
        return 'diagnose'

    @property
    def health_cmd(self):
        return 'health'

    @property
    def analyze_cmd(self):
        return 'analyze'

    @property
    def config_cmd(self):
        return 'config'

    @property
    def cmd_set(self):
        return [self.help_cmd, self.collect_cmd, self.launch_cmd, self.info_cmd, self.diagnose_cmd, self.health_cmd,
                self.analyze_cmd, self.config_cmd]


consts = Constants()
