#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

from typing import Callable
from ..activity_data import HcclData
from ..constant import MsptiResult
from ..utils import print_error_msg
from .base_monitor import BaseMonitor
from ._mspti_c import (
    _hccl_register_cb,
    _hccl_unregister_cb
)


class HcclMonitor(BaseMonitor):

    def __init__(self):
        super().__init__()
        self.user_cb = None

    def start(self, cb: Callable[[HcclData], None]) -> MsptiResult:
        if not callable(cb):
            print_error_msg("Hccl callback is invalid")
            return MsptiResult.MSPTI_ERROR_INVALID_PARAMETER
        ret = BaseMonitor.start_monitor()
        if ret == MsptiResult.MSPTI_SUCCESS:
            self.user_cb = cb
            return MsptiResult(_hccl_register_cb(self.callback))
        return ret

    def stop(self) -> MsptiResult:
        ret = BaseMonitor.stop_monitor()
        if ret == MsptiResult.MSPTI_SUCCESS:
            self.user_cb = None
            return MsptiResult(_hccl_unregister_cb())
        return ret

    def callback(self, origin_data: dict):
        try:
            if callable(self.user_cb):
                self.user_cb(HcclData(origin_data))
        except Exception as ex:
            print_error_msg(f"Call hccl callback failed. Exception: {str(ex)}")
