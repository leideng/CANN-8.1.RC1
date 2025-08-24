# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
import os
import shutil
import sqlite3
import argparse
from pathlib import Path
import multiprocessing

import pandas as pd

from ms_service_profiler.utils.file_open_check import FileStat
from ms_service_profiler.utils.check.rule import Rule
from ms_service_profiler.utils.error import DatabaseError
from ms_service_profiler.utils.file_open_check import ms_open
from ms_service_profiler.utils.log import logger
from ms_service_profiler.utils.sec import traverse_dir_common_check, read_file_common_check


visual_db_fp = ''
db_write_lock = multiprocessing.Lock()
MAX_ITERATIONS = 10000


def create_sqlite_db(output):
    global visual_db_fp

    if visual_db_fp != '':
        return

    if not os.path.exists(output):
        os.makedirs(output)
    visual_db_fp = os.path.join(output, 'profiler.db')
    try:
        conn = sqlite3.connect(visual_db_fp)
        conn.isolation_level = None
        cursor = conn.cursor()
        conn.close()
    except Exception as ex:
        raise DatabaseError("Cannot create sqlite database.") from ex


def add_table_into_visual_db(df, table_name):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        logger.warning("Writing table %r failed due to invalid dateframe:\n\t%s", table_name, df)
        return
    with db_write_lock:
        with ms_open(visual_db_fp, "a") as f:
            try:
                conn = sqlite3.connect(visual_db_fp)
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                conn.commit()
                conn.close()
            except Exception as ex:
                raise DatabaseError("Cannot update sqlite database.") from ex


def save_dataframe_to_csv(filtered_df, output, file_name):
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_path = output_path / file_name
        file_path = str(file_path)
        with ms_open(file_path, "w") as f:
            filtered_df.to_csv(f, index=False)


def _check_directory(dir_path, iteration_count):
    """检查单个文件夹的安全性"""
    try:
        # 检查循环数
        iteration_count += 1
        if iteration_count > MAX_ITERATIONS:
            raise argparse.ArgumentTypeError(f"Maximum iteration count ({MAX_ITERATIONS}) exceeded.")

        # 校验文件夹
        traverse_dir_common_check(dir_path)
    except argparse.ArgumentTypeError as e:
        raise argparse.ArgumentTypeError(f"Directory is NOT safe")


def _check_file(file_path, iteration_count):
    """检查单个文件的安全性"""
    try:
        # 检查循环数
        iteration_count += 1
        if iteration_count > MAX_ITERATIONS:
            raise argparse.ArgumentTypeError(f"Maximum iteration count ({MAX_ITERATIONS}) exceeded.")

        # 校验文件
        read_file_common_check(file_path)
    except argparse.ArgumentTypeError as e:
        raise argparse.ArgumentTypeError(f"File is NOT safe")


def check_input_path_valid(path):
    try:
        # 首先校验传入路径是否为目录，并确保目录可遍历
        safe_path = traverse_dir_common_check(path)

        # 初始化计数器
        iteration_count = 0

        # 递归检查目录下的所有文件和文件夹
        for root, dirs, files in os.walk(safe_path):
            # 检查文件夹
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                _check_directory(dir_path, iteration_count)

            # 检查文件
            for file_name in files:
                file_path = os.path.join(root, file_name)
                _check_file(file_path, iteration_count)

        return safe_path

    except argparse.ArgumentTypeError as e:
        raise argparse.ArgumentTypeError(f"Input path is illegal. {str(e)}")


def check_output_path_valid(path):
    path = os.path.abspath(path)
    if not os.path.isdir(path):  # not exists or exists but not directory
        os.makedirs(path, mode=0o750)  # create
    else:
        check_path = Rule.output_dir().check(path)
        if not check_path:
            raise argparse.ArgumentTypeError("Output path %r is incorrect due to %s, please check", path, check_path)
    return path


def find_file_in_dir(directory, filename):
    count = 0
    max_iter = 10000

    for _, _, files in os.walk(directory):
        count += len(files)
        if count > max_iter:
            break
        if filename in files:
            return True
    return False


def delete_dir_safely(path):
    # 删除文件安全校验
    try:
        check_input_path_valid(path)
    except Exception as e:
        logger.error(f'check_input_path_valid {path} failed, due to {e}')
        return

    try:
        shutil.rmtree(path)
        logger.warning(f"Delete {path}")
    except Exception as e:
        logger.error(f"Delete failed: {path}, error: {e}")
