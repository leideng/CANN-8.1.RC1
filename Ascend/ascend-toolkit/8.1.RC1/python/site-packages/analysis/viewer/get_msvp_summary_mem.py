#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2018-2019. All rights reserved.

import sqlite3

from common_func.db_manager import DBManager
from common_func.ms_constant.number_constant import NumberConstant
from common_func.ms_constant.str_constant import StrConstant
from common_func.msvp_constant import MsvpConstant


def _get_sys_mem_sql(table_name: str) -> str:
    sql = 'select sum(memtotal)/?, sum(memfree)/?, sum(buffers)/?, sum(cached)/?, ' \
          'sum(shmem)/?, sum(commitlimit)/?, sum(committed_as)/?, sum(hugepages_total)/?,' \
          ' sum(hugepages_free)/? from {}'.format(table_name)
    return sql


def _get_sys_mem_res(configs: dict, curs: any, table_name: str) -> tuple:
    data_count = curs.execute('select count(*) from {}'.format(table_name)).fetchone()[0]
    if data_count == NumberConstant.NULL_NUMBER:
        return MsvpConstant.MSVP_EMPTY_DATA
    data = curs.execute(_get_sys_mem_sql(table_name), (data_count,) * NumberConstant.COLUMN_COUNT).fetchall()
    unit = curs.execute('select unit from {} limit 1'.format(table_name)).fetchone()[0]
    headers = []
    for header in configs.get(StrConstant.CONFIG_HEADERS):
        if header not in ['Huge Pages Total(pages)', 'Huge Pages Free(pages)']:
            headers.append(header + '(' + unit + ')')
        else:
            headers.append(header)
    return headers, data, len(data)


def get_sys_mem_data(db_path: str, table_name: str, configs: dict) -> tuple:
    """
    get system memory data
    """
    conn, curs = DBManager.check_connect_db_path(db_path)
    if not (conn and curs):
        return MsvpConstant.MSVP_EMPTY_DATA
    try:
        if not DBManager.judge_table_exist(curs, table_name):
            return MsvpConstant.MSVP_EMPTY_DATA
        return _get_sys_mem_res(configs, curs, table_name)
    except sqlite3.Error:
        return MsvpConstant.MSVP_EMPTY_DATA
    finally:
        DBManager.destroy_db_connect(conn, curs)


def get_process_mem_data(db_path: str, table_name: str, configs: dict) -> tuple:
    """
    get process memory data
    """
    conn, curs = DBManager.check_connect_db_path(db_path)
    try:
        if not (conn and curs and DBManager.judge_table_exist(curs, table_name)):
            return MsvpConstant.MSVP_EMPTY_DATA
    except sqlite3.Error:
        return MsvpConstant.MSVP_EMPTY_DATA
    else:
        sql = 'select pid,name,sum(size)/count(*),sum(resident)/count(*),sum(shared)/count(*) ' \
              'from {} group by pid order by ' \
              '(sum(size)/count(*)+sum(resident)/count(*)+sum(shared)/count(*)) ' \
              'desc'.format(table_name)
        data = DBManager.fetch_all_data(curs, sql)
        return configs.get(StrConstant.CONFIG_HEADERS), data, len(data)
    finally:
        DBManager.destroy_db_connect(conn, curs)
