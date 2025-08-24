#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import os
from ..utils import safe_check
from ..core.metric.file_system import FileChecker


class KernelInvokeConfig:
    """
    An configuration descriptor for a possible kernel developed based on an Act example
    """

    def __init__(self, kernel_src_file : str, kernel_name : str):
        safe_check.check_variable_type(kernel_src_file, str)
        safe_check.check_variable_type(kernel_name, str)
        checker = FileChecker(kernel_src_file, "file")
        if not checker.check_input_file():
            raise Exception("Check kernel_src_file {} permission failed.".format(kernel_src_file))

        self.kernel_src_file = os.path.abspath(kernel_src_file)
        self.kernel_name = kernel_name
        self.type = "Act"
