#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import datetime
import os


def print_info_msg(message: str):
    time_str = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{time_str} [INFO] [MSPTI] [{os.getpid()}]: {message}")


def print_warn_msg(message: str):
    time_str = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{time_str} [WARNING] [MSPTI] [{os.getpid()}]: {message}")


def print_error_msg(message: str):
    time_str = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{time_str} [ERROR] [MSPTI] [{os.getpid()}]: {message}")
