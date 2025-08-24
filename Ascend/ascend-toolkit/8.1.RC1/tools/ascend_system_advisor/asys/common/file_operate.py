#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import csv
import os
import shutil
import configparser

from common.log import log_error, log_warning

__all__ = ["FileOperate", "MOVE_MODE", "COPY_MODE"]


MOVE_MODE = 'm'
COPY_MODE = 'c'
ENCODE_UTF_8 = "utf-8"


class FileOperate:

    @staticmethod
    def check_file(file_path):
        return os.path.isfile(file_path)

    @staticmethod
    def check_dir(dir_path):
        return os.path.isdir(dir_path)

    @staticmethod
    def check_exists(path):
        return os.path.exists(path)

    @staticmethod
    def check_access(path, mode=os.F_OK):
        return os.access(path, mode)    # mode: F_OK, R_OK, W_OK, X_OK

    @staticmethod
    def remove_file(file_path):
        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def create_file(file_path):
        if os.path.exists(file_path):
            log_warning('file {} already exists.'.format(file_path))
            return False
        else:
            with open(file_path, 'w') as fp:
                fp.close()
            return True

    @staticmethod
    def create_dir(dir_path, exist_ok=False):
        try:
            os.makedirs(dir_path, mode=0o750, exist_ok=exist_ok)
            return True
        except OSError:
            log_error("make directory {0} failed.".format(dir_path))
            return False

    @staticmethod
    def remove_dir(dir_path):
        if not os.access(dir_path, os.F_OK):
            log_warning("dir: {0} is not exist, do not need to remove.".format(dir_path))
            return False
        if not os.access(dir_path, os.W_OK):
            log_warning("dir: {0} is not access to write, can not remove.".format(dir_path))
            return False
        shutil.rmtree(dir_path)
        return True

    @staticmethod
    def walk_dir(dir_path):
        if not os.access(dir_path, os.R_OK):
            return False
        f = os.walk(dir_path)
        return f

    @staticmethod
    def list_dir(dir_path):
        if not os.access(dir_path, os.R_OK):
            return False
        f = os.listdir(dir_path)
        return f

    @staticmethod
    def write_file(file_path, info):
        file_dir = os.path.split(file_path)[0]
        if not os.path.exists(file_dir) and not FileOperate.create_dir(file_dir):
            log_error("create path directory: \"{}\" failed in write file.".format(file_dir))
            return
        with open(file_path, mode="w", encoding=ENCODE_UTF_8) as f:
            f.write(info)

    @staticmethod
    def append_write_file(file_path, info):
        file_dir = os.path.split(file_path)[0]
        if not os.path.exists(file_dir) and not FileOperate.create_dir(file_dir):
            log_error("create path directory: \"{}\" failed in write file.".format(file_dir))
            return
        with open(file_path, mode="a", encoding=ENCODE_UTF_8) as f:
            f.write(info)

    @staticmethod
    def read_file(file_path):
        if file_path.endswith(".ini"):
            cf = configparser.ConfigParser()
            cf.read(file_path, encoding=ENCODE_UTF_8)
            return cf
        elif file_path.endswith(".csv"):
            csv_buf = []
            with open(file_path, mode="r", encoding=ENCODE_UTF_8) as f:
                reader = csv.reader(f)
                for row in reader:
                    csv_buf.append(row)
            return csv_buf
        else:
            file_buf = str()
            with open(file_path, mode="r", encoding=ENCODE_UTF_8) as f:
                file_buf = f.read()
            return file_buf

    @staticmethod
    def delete_dirs(dir_list):
        if not dir_list:
            return
        for inter_dir in dir_list:
            if inter_dir and os.path.exists(inter_dir):
                if not FileOperate.remove_dir(inter_dir):
                    log_error("delete intermediate: \"{}\" failed in asys clean work.".format(inter_dir))

    @staticmethod
    def copy_file_to_dir(source_file_path, target_dir_path):
        if not os.path.exists(source_file_path) or not os.access(source_file_path, os.R_OK) or \
                not os.path.isfile(source_file_path):
            return False
        if not os.path.exists(target_dir_path):
            os.makedirs(target_dir_path)
        shutil.copy(source_file_path, target_dir_path)
        return True

    @staticmethod
    def copy_dir(source_dir_path, target_dir_path):
        if not os.path.exists(source_dir_path) or not os.access(source_dir_path, os.R_OK) or \
                not os.path.isdir(source_dir_path):
            return False
        if os.path.relpath(source_dir_path, target_dir_path).endswith(".."):
            log_error("the output directory cannot be in the data directory.")
            return False
        shutil.copytree(source_dir_path, target_dir_path)
        return True

    @staticmethod
    def move_file_to_dir(source_file_path, target_dir_path):
        if not os.path.exists(source_file_path) or not os.access(source_file_path, os.R_OK) or \
                not os.path.isfile(source_file_path):
            return False
        if not os.path.exists(target_dir_path):
            os.makedirs(target_dir_path)
        shutil.move(source_file_path, target_dir_path)
        return True

    @staticmethod
    def move_dir(source_dir_path, target_dir_path):
        if not os.path.exists(source_dir_path) or not os.access(source_dir_path, os.R_OK) or \
                not os.path.isdir(source_dir_path):
            return False
        if os.path.exists(target_dir_path):
            shutil.rmtree(target_dir_path)
        shutil.move(source_dir_path, target_dir_path)
        return True

    @staticmethod
    def collect_file_to_dir(source_file_path, target_dir_path, mode):
        if mode == MOVE_MODE: # move mode
            return FileOperate.move_file_to_dir(source_file_path, target_dir_path)
        elif mode == COPY_MODE: # copy mode
            return FileOperate.copy_file_to_dir(source_file_path, target_dir_path)
        else:
            log_error("unknown mode in collect file.")
            return False

    @staticmethod
    def collect_dir(source_dir_path, target_dir_path, mode):
        if mode == MOVE_MODE: # move mode
            return FileOperate.move_dir(source_dir_path, target_dir_path)
        elif mode == COPY_MODE: # copy mode
            return FileOperate.copy_dir(source_dir_path, target_dir_path)
        else:
            log_error("unknown mode in collect directory.")
            return False

    @staticmethod
    def check_valid_dir(dir_path):
        if not (os.path.exists(dir_path) and os.path.isdir(dir_path) and os.access(dir_path, os.R_OK)):
            return False
        if len(os.listdir(dir_path)) == 0:
            return False
        return True
