#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import os
from copy import deepcopy

from mskpp.utils import logger, autotune_utils


class Replacer:
    def __init__(self, kernel_file):
        self.origin_lines = autotune_utils.get_file_lines(kernel_file)

    @staticmethod
    def _replace_content_for_alias_name(line_index, line, replacement):
        if '=' in line:
            logger.warning(f"Line {line_index + 1} of the kernel file contains '=', please confirm whether it "
                           f"is within expectations.")
        index = len(line) - len(line.lstrip())
        new_line = line[:index] + replacement
        if line.endswith('\n'):
            new_line += '\n'
        return new_line

    @staticmethod
    def _replace_content_for_tunable_name(line_index, line, replacement):
        return line[:line.index('=') + 1] + ' ' + replacement + ';\n'

    @staticmethod
    def _write_to_file(lines, path):
        with open(path, 'w', encoding='utf-8') as file_handler:
            file_handler.writelines(lines)
        os.chmod(path, 0o640)

    def replace_config(self, node: dict, output_file_path):
        lines = deepcopy(self.origin_lines)
        for key, value in node.items():
            if not self._replace_param(key, value, lines):
                raise RuntimeError(f"the key: {key} doesn't match any line in the kernel file, replace failed.")
        self._write_to_file(lines, output_file_path)

    def _replace_param(self, key, val, lines):
        if not lines:
            raise OSError('The kernel src file is empty.')
        replace_param_success = False
        alias_key = f'tunable:{key}'
        for index, line in enumerate(lines):
            if not line:
                continue
            line_without_space = line.strip().replace(' ', '')
            if line_without_space.startswith('//'):
                continue
            # mode 1, match alias name
            if line_without_space.endswith('//' + alias_key):
                lines[index] = self._replace_content_for_alias_name(index, line, val)
                replace_param_success = True
                break
            # mode 2, match tunable name
            if line_without_space.endswith('//' + 'tunable') and key in line:
                if '=' not in line:
                    raise ValueError(f"Line {index + 1} of the kernel file doesn't contain '='.")
                if line.count('=') > 1:
                    logger.warning(
                        f'There is more than one "=" in line {index + 1}. The first "=" will be matched.')
                variale_name = line.split('=')[0].strip().split(' ')[-1]
                if key == variale_name:
                    lines[index] = self._replace_content_for_tunable_name(index, line, val)
                    replace_param_success = True
                    break
        return replace_param_success
