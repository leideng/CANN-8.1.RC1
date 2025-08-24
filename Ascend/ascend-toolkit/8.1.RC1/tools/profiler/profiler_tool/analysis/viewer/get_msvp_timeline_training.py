#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2018-2019. All rights reserved.

import sqlite3
from collections import OrderedDict

from common_func.db_manager import DBManager
from common_func.db_name_constant import DBNameConstant
from common_func.ms_constant.number_constant import NumberConstant
from common_func.msvp_common import float_calculate
from common_func.utils import Utils


def get_dvpp_engine_id(dvpp_type_name: str, conn: any) -> dict:
    """
    Get dvpp engine id for distinct engine type.
    """
    _result = OrderedDict(zip(dvpp_type_name, Utils.generator_to_list([] for _ in dvpp_type_name)))
    try:
        curs = conn.cursor() if isinstance(conn, sqlite3.Connection) else None
    except sqlite3.Error:
        return _result
    if not curs:
        return _result
    for _engine in list(_result.keys()):
        sql = "SELECT DISTINCT engineid FROM DvppOriginalData " \
              "WHERE enginetype is {}".format(dvpp_type_name.index(_engine))
        _result[_engine].extend(Utils.generator_to_list(x[0] for x in DBManager.fetch_all_data(curs, sql)))
    return _result


def get_dvpp_ids(conn: any) -> list:
    """
    Get all dvpp id list.
    """
    try:
        curs = conn.cursor() if isinstance(conn, sqlite3.Connection) else None
    except sqlite3.Error:
        return []
    dvpp_list = []
    if not curs:
        return dvpp_list
    res = DBManager.fetch_all_data(curs, "SELECT DISTINCT dvppid FROM {}".format(DBNameConstant.TABLE_DVPP_ORIGIN))
    return Utils.generator_to_list(x[0] for x in res) if res else dvpp_list


def get_dvpp_total_data(param: dict, conn: any) -> tuple:
    """
    provides data for get_dvpp_timeline method
    """
    curs = conn.cursor()
    # If the end time is equal to 0, output data for all time
    if abs(param['end_time'] - 0) <= NumberConstant.FLT_EPSILON:
        res = curs.execute("select timestamp, proctime/1, lasttime/1, "
                           "procframe/1, lastframe/1, procutilization, "
                           "allutilization from DvppOriginalData "
                           "where enginetype=? and engineid=? "
                           "and device_id=? and dvppId=?",
                           (param['engine_type'], param['engine_id'],
                            param['device_id'], param['dvppid'])).fetchall()
    else:
        res = curs.execute("select timestamp, proctime/1, lasttime/1, "
                           "procframe/1, lastframe/1, procutilization, "
                           "allutilization from DvppOriginalData "
                           "where enginetype=? and engineid=? "
                           "and timestamp between ? and ? and device_id=? and dvppId=?",
                           (param['engine_type'], param['engine_id'],
                            param['start_time'],
                            param['end_time'],
                            param['device_id'], param['dvppid'])).fetchall()
    data_time, data_util = get_result_data_for_dvpp(res, 0)
    return data_time, data_util


def get_result_data_for_dvpp(res: list, delta: float) -> tuple:
    """
    method that provides resulta for get_dvpp_total_data
    """
    data_time = []
    data_util = []
    for i, _ in enumerate(res):
        _timestamp = float_calculate([res[i][0], delta])
        data_time_piece = [_timestamp, res[i][1], res[i][2], res[i][3], res[i][4]]
        data_time.append(tuple(data_time_piece))
        # change proc_util to percentage
        per1 = float(res[i][5].split("%")[0])
        per1 = round(per1 / NumberConstant.PERCENTAGE, 3)
        per2 = float(res[i][6].split("%")[0])
        per2 = round(per2 / NumberConstant.PERCENTAGE, 3)
        data_util_piece = [_timestamp, per1, per2]
        data_util.append(tuple(data_util_piece))

    return data_time, data_util


def get_dvpp_legend() -> list:
    """
    Get dvpp data legend.
    """
    return ['proc_time', 'last_time', 'proc_frame', 'last_frame'], \
           ['proc_utilization', 'all_utilization']
