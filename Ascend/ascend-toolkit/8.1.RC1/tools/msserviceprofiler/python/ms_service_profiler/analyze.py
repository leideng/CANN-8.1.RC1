# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.

import argparse
from ms_service_profiler.utils.file_open_check import FileStat


def check_input_path_valid(path):
    try:
        file_stat = FileStat(path)
    except Exception as err:
        raise argparse.ArgumentTypeError(f"input path:{path} is illegal. Please check.")
    if not file_stat.is_dir:
        raise argparse.ArgumentTypeError(f"Path is not a valid directory: {path}")
    return path
