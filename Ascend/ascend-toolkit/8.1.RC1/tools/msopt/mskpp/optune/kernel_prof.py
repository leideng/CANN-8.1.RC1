#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from mspti import KernelData, KernelMonitor, MsptiResult
import acl


class Monitor:
    """A context manager for monitoring kernel execution via MSPTI."""

    def __init__(self):
        self._monitor = KernelMonitor()
        self._durations = []

    def get_task_duration(self):
        return sum(self._durations)

    def start(self, device_id):
        self._durations.clear()
        acl.init()
        acl.rt.set_device(device_id)
        result = self._monitor.start(self._kernel_callback)
        if result != MsptiResult.MSPTI_SUCCESS:
            raise RuntimeError(f'failed to start mspti monitor, error code: {result}.')

    def stop(self, device_id):
        acl.finalize()
        self._monitor.flush_all()
        result = self._monitor.stop()
        if result != MsptiResult.MSPTI_SUCCESS:
            raise RuntimeError(f'failed to stop mspti monitor, error code: {result}')
        acl.rt.reset_device(device_id)

    def _kernel_callback(self, data: KernelData):
        duration = data.end - data.start
        self._durations.append(duration)
