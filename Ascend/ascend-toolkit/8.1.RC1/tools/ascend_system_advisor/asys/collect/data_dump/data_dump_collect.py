#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import os

from params import ParamDict
from common import consts
from common import FileOperate as f
from common.file_operate import MOVE_MODE, COPY_MODE
from common import log_debug, log_warning
from drv import EnvVarName

__all__ = ["collect_data_dump"]


def _get_collect_path():
    collect_data_dump_path_list = []
    # asys launch
    if ParamDict().get_command() == consts.launch_cmd:
        collect_data_dump_path_list.append(
            os.path.join(ParamDict().asys_output_timestamp_dir, "npu_collect_intermediates")
        )
        return collect_data_dump_path_list

    # asys collect
    task_dir = ParamDict().get_arg("task_dir")
    if task_dir:
        collect_data_dump_path_list.append(task_dir)
    env_var = EnvVarName()
    if env_var.npu_collect_path:
        log_debug("get env NPU_COLLECT_PATH success, add NPU_COLLECT_PATH to dump collect path.")
        collect_data_dump_path_list.append(env_var.npu_collect_path)

    if env_var.work_path:
        log_debug("get env ASCEND_WORK_PATH success, add ASCEND_WORK_PATH to dump collect path.")
        collect_data_dump_path_list.append(env_var.work_path)

    collect_data_dump_path_list.append(env_var.current_path)

    return collect_data_dump_path_list


def get_source_dir():
    for data_dump_path in _get_collect_path():
        data_dump_source_dir = os.path.join(data_dump_path, "extra-info", "data-dump")
        if data_dump_source_dir and f.check_dir(data_dump_source_dir):
            log_debug("data-dump source check success, path={}.".format(data_dump_source_dir))
            return data_dump_source_dir
    return None


def collect_data_dump(output_root_path):
    data_dump_source_dir = get_source_dir()
    if data_dump_source_dir is None:
        log_warning("data_dump collect failed.")
        return

    data_dump_target_dir = os.path.join(output_root_path, "dfx", "data-dump")
    if ParamDict().get_command() == consts.launch_cmd:
        ret = f.collect_dir(data_dump_source_dir, data_dump_target_dir, MOVE_MODE)
    else:
        ret = f.collect_dir(data_dump_source_dir, data_dump_target_dir, COPY_MODE)
    if not ret:
        log_warning("data_dump collect failed, copy {} dir failed.".format(data_dump_source_dir))
