#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import logging
import os
import re
import sys

DIR_NAME_LENGTH_LIMIT = 1024
FILE_NAME_LENGTH_LIMIT = 200
INPUT_BINARY_FILE_MAX_SIZE = 100 * 1024 * 1024
DATA_DIRECTORY_AUTHORITY = 0o750


class FileChecker:
    def __init__(self, path, file_type, threshold=INPUT_BINARY_FILE_MAX_SIZE):
        self.path = path
        self.absolute_path = os.path.abspath(path)
        self.threshold = threshold
        self.file_type = file_type

    def check_input_file(self):
        if not self.is_string_char_valid():
            return False
        if not os.path.exists(self.absolute_path):
            logging.error("path:%s not exist.", self.absolute_path)
            return False
        if not self.path_len_check_valid():
            logging.error("path:%s length is too long.", self.absolute_path)
            return False
        if self.is_soft_link_recusively():
            logging.error("path:%s contains soft link which may cause security problems, please check",
                self.absolute_path)
            return False
        if self.file_type != "dir" and os.path.isdir(self.absolute_path):
            logging.error("path:%s is dir, not a file.", self.absolute_path)
            return False
        path_permission = {"csv" : os.R_OK, "dir" : os.W_OK, "file" : os.R_OK}
        if self.file_type not in path_permission:
            logging.error("path:%s, the file type is unsupport.", self.absolute_path)
            return False
        file_mode = path_permission[self.file_type]
        if not self.check_path_permission(file_mode):
            return False
        if self.file_type != "dir" and file_mode & os.R_OK is os.R_OK and not self.check_file_size_valid:
            logging.error("path:%s, file size is too large, max file size:%s", self.absolute_path, self.threshold)
            return False
        return True

    def check_output_file(self):
        if os.path.exists(self.absolute_path):
            logging.error("path:%s already exists", self.absolute_path)
            return False
        self.absolute_path = os.path.dirname(self.absolute_path)
        if (not self.check_input_file()):
            return False
        return True


    def is_string_char_valid(self):
        invalid_chars = {'\n':'\\n', '\f':'\\f', '\r':'\\r', '\b':'\\b', '\t':'\\t', '\v':'\\v', '\u007F':'\\u007F'}
        for key in invalid_chars:
            if key in self.absolute_path:
                logging.error("path:%s contains %s, which is invalid", self.absolute_path, invalid_chars[key])
                return False
        return True
    
    def is_soft_link_recusively(self):
        while self.absolute_path.endswith('/'):
            self.absolute_path = self.absolute_path[:-1]
        if os.path.islink(self.absolute_path):
            return True
        dirs = self.absolute_path.split('/')
        curpath = ""
        for dir_name in dirs:
            if dir_name == "":
                continue
            curpath = curpath + '/' + dir_name
            if os.path.islink(curpath):
                return True
        return False

    def path_len_check_valid(self):
        if len(self.absolute_path) > DIR_NAME_LENGTH_LIMIT:
            return False
        dirs = self.absolute_path.split('/')
        for dir_name in dirs:
            if len(dir_name) > FILE_NAME_LENGTH_LIMIT:
                return False
        return True
    
    def check_path_permission(self, file_mode):
        # 读写执行权限校验
        file_stat = os.stat(self.absolute_path)
        file_permission = oct(file_stat.st_mode)[-3:]
        ower_permission = int(file_permission[0])
        other_permission = int(file_permission[2])
        uid = os.getuid()
        if file_mode & os.R_OK is os.R_OK and ower_permission & os.R_OK is not os.R_OK:
            logging.error("path:%s is not readable", self.absolute_path)
            return False
        if file_mode & os.W_OK is os.W_OK and ower_permission & os.W_OK is not os.W_OK:
            logging.error("path:%s is not writable", self.absolute_path)
            return False
        if file_mode & os.X_OK is os.X_OK and ower_permission & os.X_OK is not os.X_OK:
            logging.error("path:%s is not executable", self.absolute_path)
            return False
        if other_permission & os.W_OK is os.W_OK:
            logging.error("path:%s is writable by any other users.", self.absolute_path)
            return False
        if uid != 0 and uid != file_stat.st_uid:
            logging.error("path:%s. the current owner have inconsistent permission.", self.absolute_path)
            return False
        return True

    def check_file_size_valid(self):
        file_stat = os.stat(self.absolute_path)
        file_size = file_stat.st_size
        return file_size <= self.threshold