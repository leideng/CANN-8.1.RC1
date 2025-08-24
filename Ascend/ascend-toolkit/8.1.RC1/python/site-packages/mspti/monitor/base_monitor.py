#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

from abc import ABCMeta
from ..constant import MsptiResult
from ..utils import print_error_msg
from ._mspti_c import (
    _start,
    _stop,
    _flush_all,
    _flush_period
)


class BaseMonitor(metaclass=ABCMeta):

    def __init__(self):
        super().__init__()

    @classmethod
    def start_monitor(cls) -> MsptiResult:
        return MsptiResult(_start())

    @classmethod
    def stop_monitor(cls) -> MsptiResult:
        try:
            ret = MsptiResult(_stop())
            if ret == MsptiResult.MSPTI_SUCCESS:
                return MsptiResult(_flush_all())
            return ret
        except Exception as ex:
            print_error_msg(f"Call stop failed. Exception: {str(ex)}")
            return MsptiResult.MSPTI_ERROR_INNER

    @classmethod
    def flush_all(cls) -> MsptiResult:
        try:
            return MsptiResult(_flush_all())
        except Exception as ex:
            print_error_msg(f"Call flush_all failed. Exception: {str(ex)}")
            return MsptiResult.MSPTI_ERROR_INNER

    @classmethod
    def flush_period(cls, time_ms: int) -> MsptiResult:
        return MsptiResult(_flush_period(time_ms))
