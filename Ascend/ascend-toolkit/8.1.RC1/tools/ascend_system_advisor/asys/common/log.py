#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import logging

__all__ = ["log_debug", "log_info", "log_warning", "log_error", "close_log", "open_log"]

LOG_FORMAT = "%(asctime)s [ASYS] [%(levelname)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def log_debug(log_str):
    logging.debug(log_str)


def log_info(log_str):
    logging.info(log_str)


def log_warning(log_str):
    logging.warning(log_str)


def log_error(log_str):
    logging.error(log_str)


def close_log():
    logging.disable(logging.INFO)
    logging.disable(logging.DEBUG)
    logging.disable(logging.WARNING)


def open_log():
    logging.disable(logging.NOTSET)
