#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import os

from mskpp.utils import logger, safe_check


def check_params(configs, warmup, repeat, device_ids):
    check_configs(configs)
    check_warmup(warmup)
    check_repeat(repeat)
    check_device_ids(device_ids)


def check_configs(configs):
    if not configs or not isinstance(configs, list):
        raise ValueError('The autotune configs is not a valid list.')
    for config in configs:
        if not isinstance(config, dict):
            raise ValueError(f'The config {config} is not a valid dict.')
        for key, val in config.items():
            if not key or not isinstance(key, str):
                raise ValueError(f'The key {key} is not a valid str.')
            if not val or not isinstance(val, str):
                raise ValueError(f'The val {val} is not a valid str.')


def check_warmup(warmup):
    if not isinstance(warmup, int) or warmup <= 0:
        raise ValueError(f'The warmup value is not a valid positive integer.')
    if warmup < 300:
        logger.warning('The device requires 300μs to reach full frequency, '
                       'but the warmup value you provided is less than 300μs.')
    if warmup > 10 ** 5:
        raise ValueError(f'The warmup value {warmup} is too large.')


def check_repeat(repeat):
    if not isinstance(repeat, int) or repeat <= 0:
        raise ValueError('The repeat value is not a valid positive integer.')
    if repeat > 10 ** 4:
        raise ValueError(f'The warmup value {repeat} is too large.')


def check_device_ids(device_ids):
    if not isinstance(device_ids, list):
        raise ValueError(f'The device_ids: {device_ids} is not a list.')
    if not device_ids:
        raise ValueError('The device id list is empty.')
    if len(device_ids) > 10 ** 2:
        raise ValueError(f'The device id list is too large.')
    for device_id in device_ids:
        if not isinstance(device_id, int) or device_id < 0:
            raise ValueError(f'The device id {device_id} is not valid.')
    if len(device_ids) > 1:
        logger.warning(
            'Multi-device parallel execution is not yet supported. '
            'Only the first device id in the device id list will be used currently.')


def get_file_lines(file):
    if not file or not os.path.isfile(file):
        raise OSError(f'The kernel file {file} is not valid.')
    safe_check.check_input_file(file)
    with open(file, 'r', encoding='utf-8') as file_handler:
        lines = file_handler.readlines()
    return lines
