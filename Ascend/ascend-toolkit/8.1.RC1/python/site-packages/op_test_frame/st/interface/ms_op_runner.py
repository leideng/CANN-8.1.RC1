#!/usr/bin/env python
# coding=utf-8
"""
Function:
This class mainly involves compile and run mindspore op.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved Â© 2020-201
"""

import os
import subprocess

from op_test_frame.common import op_status
from op_test_frame.st.interface.const_manager import ConstManager
from op_test_frame.st.interface import utils
from op_test_frame.st.interface import op_st_case_info


class MsOpRunner:
    """
    Class for compile and run mindspore op test code.
    """

    def __init__(self, path, op_name, soc_version, report):
        self.path = path
        self.soc_version = soc_version
        self.report = report
        self.op_name = op_name

    def run(self):
        """
        Run mindspore op
        """
        # test_data
        test_file = ConstManager.TEST_PY.format(op_name=self.op_name)
        test_py_path = os.path.join(self.path, 'src', test_file)
        utils.print_info_log('Start to run %s.' % test_py_path)
        if not os.path.exists(test_py_path):
            utils.print_error_log(
                'There is no execute file "%s" in %s. Please check the path '
                'for running.' % (test_file, os.path.dirname(test_py_path)))
            raise utils.OpTestGenException(
                ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
        src_path = os.path.dirname(test_py_path)
        utils.check_path_valid(src_path, True)
        output_path = os.path.dirname(self.path)
        os.chdir(output_path)
        run_cmd = ['python3', '-m', 'pytest', '-s', test_py_path]
        utils.print_info_log("Run command line: cd %s && %s " % (
            output_path, " ".join(run_cmd)))
        self._execute_command(run_cmd)
        utils.print_info_log('Finish to run %s.' % test_py_path)
        self.add_op_st_stage_result(op_status.SUCCESS, "run_ms_op_test_code",
                                    None, " ".join(run_cmd))

    def add_op_st_stage_result(self, status=op_status.FAILED,
                               stage_name=None, result=None, cmd=None):
        """
        add op st stage_result
        """
        stage_result = op_st_case_info.OpSTStageResult(
            status, stage_name, result, cmd)
        for case_report in self.report.report_list:
            case_report.trace_detail.add_stage_result(stage_result)

    def process(self):
        """
        run mindspore op
        """
        self.run()

    def _execute_command(self, cmd):
        utils.print_info_log('Execute command: %s' % cmd)
        ms_process = subprocess.Popen(cmd, shell=False,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT)
        while ms_process.poll() is None:
            line = ms_process.stdout.readline()
            line = line.strip()
            if line:
                print(line)
        if ms_process.returncode != 0:
            self.add_op_st_stage_result(op_status.FAILED, "run_ms_op_test_code",
                                        None, " ".join(cmd))
            utils.print_error_log('Failed to execute command: %s' % cmd)
            raise utils.OpTestGenException(
                ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
