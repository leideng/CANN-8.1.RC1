#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import os
import shutil

from analyzer.communication_analyzer import CommunicationAnalyzer
from analyzer.communication_matrix_analyzer import CommunicationMatrixAnalyzer
from common_func.data_check_manager import DataCheckManager
from common_func.msprof_common import get_path_dir
from common_func.msvp_common import check_dir_writable
from common_func.common import warn


class AnalyzeCommand:
    """
    The class for handle analyze command.
    """

    FILE_NAME = os.path.basename(__file__)

    def __init__(self: any, args: any) -> None:
        self.collection_path = os.path.realpath(args.collection_path)
        self.rule = args.rule
        self.export_type = getattr(args, "export_type", 'text')
        self.clear_mode = getattr(args, "clear_mode", False)

    def process(self: any) -> None:
        if DataCheckManager.check_prof_level0(self.collection_path):
            warn(self.FILE_NAME, "Analyze will do nothing in prof level 0.")
            self._clear_dir(self.collection_path)
            return
        analyze_handler = {
            'communication': CommunicationAnalyzer,
            'communication_matrix': CommunicationMatrixAnalyzer
        }
        rules = set(self.rule.split(','))
        for rule in rules:
            analyze_command = analyze_handler.get(rule)(self.collection_path, self.export_type)
            analyze_command.process()
        self._clear_dir(self.collection_path)

    def _clear_dir(self, result_dir: str) -> None:
        if not self.clear_mode:
            return
        sub_dirs = sorted(get_path_dir(result_dir), reverse=True)
        clear_dir = 'sqlite'
        for sub_dir in sub_dirs:  # result_dir
            dir_name = os.path.join(result_dir, sub_dir, clear_dir)
            if os.path.exists(dir_name):
                check_dir_writable(dir_name)
                shutil.rmtree(dir_name)
