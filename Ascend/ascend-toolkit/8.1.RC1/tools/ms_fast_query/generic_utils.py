#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
import os
import re
import platform


MAX_HEADER_FILE_SIZE = 2 * 1024**2  # 2MB
MAX_JSON_FILE_SIZE = 50 * 1024**2  # 50MB
MAX_PATH_LENGTH_LIMIT = 4096
MAX_FILE_COUNT = 5000
MAX_FILENAME_LENGTH_LIMIT = 200



def is_path_length_valid(path):
    if len(os.path.basename(path)) > MAX_FILENAME_LENGTH_LIMIT:
        return False
    path = os.path.realpath(path)
    return len(path) <= MAX_PATH_LENGTH_LIMIT


def check_path_length_valid(path):
    if not is_path_length_valid(path):
        raise ValueError('The real path or file name is too long.')


def check_path_name_valid(path):
    if platform.system().lower() == 'windows':
        pattern = re.compile(r'[\s./\\:_\-~0-9a-zA-Z]+')
        if not pattern.fullmatch(path):
            raise ValueError('Invalid path, only the following '
                    'characters are allowed in the path: A-Z a-z 0-9 - _ . / \\ :')
    else:
        pattern = re.compile(r'[\s./:_\-~0-9a-zA-Z]+')
        if not pattern.fullmatch(path):
            raise ValueError('Invalid path, only the following '
                    'characters are allowed in the path: A-Z a-z 0-9 - _ . / :')


def check_path_not_empty(path):
    if not path:
        raise ValueError('The path is empty.')


def is_path_owner_consistent(path):
    if platform.system().lower() == 'windows':
        return True
    file_owner = os.stat(path).st_uid
    return file_owner == os.getuid()


def check_output_file(output_path):
    check_path_not_empty(output_path)
    check_path_length_valid(output_path)
    check_path_name_valid(output_path)
    check_path_not_exist(output_path)
    output_dir = os.path.dirname(output_path)
    check_output_dir(output_dir)


def check_file_size(input_path, max_file_size=MAX_HEADER_FILE_SIZE):
    if os.path.getsize(input_path) > max_file_size:
        raise ValueError(f'The size of file exceeds {max_file_size // 1024 ** 2}MB.')


def check_input_file(input_path, max_file_size=MAX_HEADER_FILE_SIZE):
    check_path_not_empty(input_path)
    check_path_length_valid(input_path)
    check_path_name_valid(input_path)
    check_path_exist(input_path)
    input_abs_path = os.path.realpath(input_path)
    if os.path.isfile(input_path):
        check_file_size(input_path, max_file_size)
    if not os.access(input_abs_path, os.R_OK):
        raise PermissionError('The input path is not readable!')
    if not is_path_owner_consistent(input_abs_path):
        raise ValueError(f'The input path may be insecure because it does not belong to you.')


def check_path_not_exist(path):
    path = os.path.abspath(path)
    abs_path = os.path.realpath(path)
    if os.path.exists(abs_path) or os.path.islink(path):
        raise ValueError(f'The path already exist, please remove it first.')


def check_path_exist(path):
    abs_path = os.path.realpath(path)
    if not os.path.exists(abs_path):
        raise ValueError("The path doesn't exist.")


def check_output_dir(output_dir):
    check_path_exist(output_dir)
    output_abs_dir = os.path.realpath(output_dir)
    if not os.access(output_abs_dir, os.W_OK):
        raise PermissionError(f'The output directory is not writable!')
