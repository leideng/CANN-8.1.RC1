#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
import os.path
import threading
from multiprocessing import Process

from common.const import GET_DEVICES_INFO_TIMEOUT
from params import ParamDict
from common import log_error, log_info
from common import run_msnpureport_cmd
from common import FileOperate as f
from drv.env_type import LoadSoType
from info import AsysInfo
from health import AsysHealth

from view.progress_display import waiting

from collect.log import collect_device_logs
from collect.log import collect_host_logs
from collect.log import collect_rc_logs
from collect.graph import collect_graph
from collect.ops import collect_ops
from collect.trace import collect_trace
from collect.data_dump import collect_data_dump

__all__ = ["AsysCollect"]


def collect_health_status():
    """
    no redundant plog temporary directory is generated when the status and health information is collected.
    """
    if ParamDict().get_env_type() == "EP":
        # collect software, hardware and status information
        AsysInfo().write_info()
        # collect health check result
        AsysHealth().health()
    else:
        AsysInfo().get_software_info(write_file=True)


class AsysCollect:
    def __init__(self):
        self.output_root_path = ParamDict().asys_output_timestamp_dir
        self.finish_flag = False

    def _device_file_export(self):
        def run_msnpureport(export_dir_path):
            f.create_dir(export_dir_path)
            export_dir_cmd = "cd " + export_dir_path
            export_tool = "msnpureport -f"
            export_cmd = "{0};{1}".format(export_dir_cmd, export_tool)
            cmd_res = run_msnpureport_cmd(export_cmd)
            if not cmd_res[0]:
                log_error("call msnpureport tool failed, sys.stderr: \"{}\"".format(cmd_res[1].strip()))
                f.remove_dir(export_dir_path)
                return False
            return True

        export_dir_path = os.path.join(self.output_root_path, "export_tmp")
        if not run_msnpureport(export_dir_path):
            return False
        # export success
        in_export_dir = f.list_dir(export_dir_path)
        if not in_export_dir:
            log_error("no files or directories in {0}".format(export_dir_path))
            return False
        msnpureport_output_dir = os.path.join(export_dir_path, in_export_dir[0])
        return msnpureport_output_dir

    def collect(self):
        # check params
        if ParamDict().get_arg("remote") is not False or ParamDict().get_arg("all") or ParamDict().get_arg("quiet"):
            log_error("'--remote', '--all' and '--quiet' can be used only when '-r=stacktrace'.")
            return False

        log_info('collect task start, running:')
        # When the main program exits, the system checks whether there is a sub-thread whose daemon value is False.
        # If it exists, the main program exits after the sub-thread exits. Default daemon value is False.
        t = threading.Thread(target=self.wait_view, daemon=True)
        t.start()

        if ParamDict().get_env_type() == "EP":
            # collect log files
            collect_host_logs(self.output_root_path)
            # export device files
            msnpureport_output_dir = self._device_file_export()
            if msnpureport_output_dir:
                collect_device_logs(msnpureport_output_dir, self.output_root_path)
            else:
                log_error("msnpureport tool export device files failed.")
        else:
            collect_rc_logs(self.output_root_path)

        # collect graph
        collect_graph(self.output_root_path)

        # collect_data_dump
        collect_data_dump(self.output_root_path)

        # collect ops
        collect_ops(self.output_root_path)

        # collect trace
        collect_trace(self.output_root_path)

        #  plog are generated when health and status are collected.
        LoadSoType().dll_close()
        p = Process(target=collect_health_status)
        p.daemon = True
        p.start()
        # wait 10s
        p.join(GET_DEVICES_INFO_TIMEOUT)
        # terminate the process, use SIGTERM
        p.terminate()

        self.finish_flag = True
        t.join()    # wait print process_display end
        return True

    def wait_view(self):
        while not self.finish_flag:
            waiting()
            continue

    def clean_work(self):
        msnpureport_export_path = os.path.join(self.output_root_path, "export_tmp")
        dir_list = [msnpureport_export_path]
        f.delete_dirs(dir_list)
