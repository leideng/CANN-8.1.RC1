#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

from common_func.db_manager import DBManager
from common_func.db_name_constant import DBNameConstant
from common_func.ms_constant.str_constant import StrConstant
from common_func.path_manager import PathManager
from msmodel.interface.view_model import ViewModel


class FreqDataViewModel(ViewModel):
    '''
    class for query aicore freq.db
    '''

    
    def __init__(self, params: dict) -> None:
        self._result_dir = params.get(StrConstant.PARAM_RESULT_DIR)
        self._iter_range = params.get(StrConstant.PARAM_ITER_ID)
        super().__init__(self._result_dir, DBNameConstant.DB_FREQ,
                         [DBNameConstant.TABLE_FREQ_PARSE])

    def get_data(self) -> list:
        '''
        query data from freq.db
        '''
        if not DBManager.check_tables_in_db(
            PathManager.get_db_path(self._result_dir, self.db_name), DBNameConstant.TABLE_FREQ_PARSE):
            return []
        sql = "select syscnt, freq from {}".format(DBNameConstant.TABLE_FREQ_PARSE)
        return DBManager.fetch_all_data(self.cur, sql)
