#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
import time

__all__ = ["waiting"]


def waiting(cycle=1, delay=0.1):
    for i in range(cycle):
        for ch in ['-', '\\', '|', '/']:
            print('\r%s\r' % ch, end='', flush=True)
            time.sleep(delay)