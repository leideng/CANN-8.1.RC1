#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import os
import subprocess
import signal
import sys
import threading

from params import ParamDict
from common import log_info, log_error, log_debug, log_warning
from common import FileOperate as f
from common.const import RetCode
from view.progress_display import waiting
from collect import AsysCollect

__all__ = ["AsysLaunch"]


class AsysLaunch:
    def __init__(self):
        self.output_root_path = ParamDict().asys_output_timestamp_dir
        self.finish_flag = False
        self.user_cmd = ParamDict().get_arg("task")
        self.console_output = ""
        npu_collect_path = os.path.join(ParamDict().asys_output_timestamp_dir, "npu_collect_intermediates")
        self.env_prepare = {
            "DUMP_GE_GRAPH": ParamDict().get_ini("DUMP_GE_GRAPH"),
            "DUMP_GRAPH_LEVEL": ParamDict().get_ini("DUMP_GRAPH_LEVEL"),
            "ASCEND_GLOBAL_LOG_LEVEL": ParamDict().get_ini("ASCEND_GLOBAL_LOG_LEVEL"),
            "ASCEND_GLOBAL_EVENT_ENABLE": ParamDict().get_ini("ASCEND_GLOBAL_EVENT_ENABLE"),
            "ASCEND_SLOG_PRINT_TO_STDOUT": ParamDict().get_ini("ASCEND_SLOG_PRINT_TO_STDOUT"),
            "ASCEND_HOST_LOG_FILE_NUM": "1000",
            "ASCEND_PROCESS_LOG_PATH": os.path.join(npu_collect_path, "task_launch_host_log"),
            "ASCEND_WORK_PATH": os.path.join(npu_collect_path, "task_launch_host_log"),
            "NPU_COLLECT_PATH": npu_collect_path,
        }

    def prepare_for_launch(self):
        # prepare environment variable
        for env_name, env_val in self.env_prepare.items():
            log_debug("env_name: {}, env_val: {}".format(env_name, env_val))
            os.environ[env_name] = env_val
        # prepare npu collect path dir
        if not f.create_dir(os.environ["NPU_COLLECT_PATH"]):
            log_error(f"create npu collect path failed, NPU_COLLECT_PATH={self.env_prepare['NPU_COLLECT_PATH']}")
            return RetCode.FAILED
        # collect atrace log
        if not f.create_dir(os.environ["ASCEND_WORK_PATH"]):
            log_error(f"create ascend work path failed, ASCEND_WORK_PATH={self.env_prepare['ASCEND_WORK_PATH']}")
            return RetCode.FAILED

        log_debug("prepare for launch finished.")
        return RetCode.SUCCESS

    def execute_task(self):
        def interrupt_handler(signum, frame):
            os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            # exit the current main process group.
            os.killpg(os.getpgid(0), signal.SIGINT)

        log_info('launch task start, running:')
        signal.signal(signal.SIGINT, interrupt_handler)
        pro = subprocess.Popen(self.user_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               encoding='utf-8', preexec_fn=os.setsid, env=os.environ, errors='ignore')
        t = threading.Thread(target=self.wait_view, daemon=True)
        t.start()

        ParamDict().set_task_pid(pid=pro.pid)

        self.console_output, _ = pro.communicate()
        # subprocess end, restore the default SIGINT signal
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        self.finish_flag = True
        t.join()    # wait print process_display end

        if pro.returncode == 0:
            log_info('task execute finished, output:\n{0}'.format(self.console_output))
        else:
            log_warning("task occurred error, output:\n{0}".format(self.console_output))

    def task_out_collect(self, output_root_path):
        def collect_task_output_info():
            if ParamDict().get_env_type() == "EP":
                dir_path = os.path.join(output_root_path, "dfx", "log", "host")
            else:
                dir_path = os.path.join(output_root_path, "dfx", "log")
            if not f.check_dir(dir_path):
                f.create_dir(dir_path)
            user_cmd_path = os.path.join(dir_path, "user_cmd")
            screen_print_path = os.path.join(dir_path, "screen.txt")
            f.write_file(user_cmd_path, self.user_cmd)
            f.write_file(screen_print_path, self.console_output)
            log_debug("collect user cmd and task print success.")

        collect_task_output_info()
        task_collector = AsysCollect()
        if not task_collector.collect():
            log_error("collect information after task failed.")
        else:
            log_info("collect information after task success.")

    def launch(self):
        ret = self.prepare_for_launch()
        if ret != RetCode.SUCCESS:
            log_error("prepare for launch failed.")
            return False

        self.execute_task()
        self.task_out_collect(self.output_root_path)

        return True

    def wait_view(self):
        while not self.finish_flag:
            waiting()
            continue

    def clean_work(self):
        npu_collect_path = os.path.join(self.output_root_path, "npu_collect_intermediates")
        msnpureport_export_path = os.path.join(self.output_root_path, "export_tmp")
        dir_list = [npu_collect_path, msnpureport_export_path]
        f.delete_dirs(dir_list)