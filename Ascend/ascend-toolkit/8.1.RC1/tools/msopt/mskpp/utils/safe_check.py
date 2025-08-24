#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import os
import stat


MAX_FILE_SIZE = 10 * 1024 ** 2
SAVE_DATA_FILE_AUTHORITY = stat.S_IWUSR | stat.S_IRUSR
OPEN_FLAGS = os.O_WRONLY | os.O_CREAT


def check_input_file(path):
    if not os.path.isfile(path):
        raise OSError(f'{path} is not a valid file path.')
    path = os.path.abspath(path)
    if os.path.islink(path):
        raise OSError(f"The file {path} shouldn't be a soft link.")
    if not os.access(path, os.R_OK):
        raise PermissionError(f'The file {path} is not readable.')
    if os.path.getsize(path) >= MAX_FILE_SIZE:
        raise ValueError(f'The file {path} is too large.')
    if not check_path_owner_consistent(path):
        raise PermissionError(f'The file {path} is insecure because it does not belong to you.')
    check_others_w_permission(path)


def check_path_owner_consistent(path):
    # st_uid:user ID of owner, os.getuid: Return the current process's user id, root's uid is 0
    uid = os.stat(path).st_uid
    return uid == os.getuid() or uid == 0


def check_others_w_permission(path):
    mode = os.stat(path).st_mode
    if mode & stat.S_IWGRP:
        raise PermissionError(f'Path {path} cannot have write permission of group.')
    if mode & stat.S_IWOTH:
        raise PermissionError(f'Path {path} cannot have write permission of other users.')


def check_variable_type(var, expected_type):
    if not isinstance(var, expected_type):
        raise TypeError(f'The variable {var} is not of type {expected_type}.')