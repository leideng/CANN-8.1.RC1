#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import os

from params import ParamDict
from common import consts
from common import log_debug, log_warning
from common import FileOperate as f
from common.file_operate import COPY_MODE, MOVE_MODE
from drv import EnvVarName

__all__ = ["collect_graph"]


def collect_graph_files(source_dir, target_dir):
    ge_res = []
    tf_res = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if (file.startswith("ge_onnx") and file.endswith(".pbtxt")) or \
                    (file.startswith("ge_proto") and file.endswith(".txt")):
                ge_res.append(os.path.join(root, file))
            if file.startswith("TF_GeOp") and file.endswith(".pbtxt"):
                tf_res.append(os.path.join(root, file))

    if not (ge_res or tf_res):
        return False

    ge_target_dir_path = os.path.join(target_dir, "ge")
    tf_target_dir_path = os.path.join(target_dir, "tf")

    ret = True
    for file_path in ge_res:
        ge_file_ret = f.collect_file_to_dir(file_path, ge_target_dir_path, COPY_MODE)
        ret = ret and ge_file_ret

    for file_path in tf_res:
        tf_file_ret = f.collect_file_to_dir(file_path, tf_target_dir_path, COPY_MODE)
        ret = ret and tf_file_ret
    return ret


def collect_cmd_graph_files(graph_target_dir):
    collect_graph_path_list = []

    task_dir_path = ParamDict().get_arg("task_dir")  # task_dir checked in set_arg
    if task_dir_path and f.check_dir(task_dir_path):
        collect_graph_path_list.append(task_dir_path)

    env_var = EnvVarName()
    if env_var.npu_collect_path:
        log_debug("get env NPU_COLLECT_PATH success, add NPU_COLLECT_PATH to graph collect path.")
        collect_graph_path_list.append(env_var.npu_collect_path)
    if env_var.dump_graph_path:
        log_debug("get env DUMP_GRAPH_PATH success, add DUMP_GRAPH_PATH to graph collect path.")
        collect_graph_path_list.append(env_var.dump_graph_path)
    if env_var.work_path:
        log_debug("get env ASCEND_WORK_PATH success, add ASCEND_WORK_PATH to graph collect path.")
        collect_graph_path_list.append(env_var.work_path)
    collect_graph_path_list.append(env_var.current_path)

    for collect_graph_path in collect_graph_path_list:
        ret = collect_graph_files(collect_graph_path, graph_target_dir)
        if ret:
            return True
    return False


def collect_graph(output_root_path):
    if (ParamDict().get_command() == consts.launch_cmd) and (not ParamDict().get_ini("graph") == "1"):  # 1: open
        log_debug("graph is not set on, not collect graph files")
        return

    ret = False
    graph_target_dir = os.path.join(output_root_path, "dfx", "graph")
    if ParamDict().get_command() == consts.launch_cmd:
        npu_collect_path = os.path.join(ParamDict().asys_output_timestamp_dir, "npu_collect_intermediates")
        graph_source_dir = os.path.join(npu_collect_path, "extra-info", "graph")
        if f.check_dir(graph_source_dir):
            log_debug("graph source check success, path={}.".format(graph_source_dir))
            ret = f.collect_dir(graph_source_dir, graph_target_dir, MOVE_MODE)
    else:
        ret = collect_cmd_graph_files(graph_target_dir)

    if not ret:
        log_warning("graph collect failed.")
