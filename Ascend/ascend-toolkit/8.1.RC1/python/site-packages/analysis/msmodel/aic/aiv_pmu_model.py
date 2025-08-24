#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

from common_func.db_manager import DBManager
from common_func.db_name_constant import DBNameConstant
from common_func.ms_constant.str_constant import StrConstant
from common_func.path_manager import PathManager
from mscalculate.aic.aic_utils import AicPmuUtils
from msmodel.interface.parser_model import ParserModel
from viewer.calculate_rts_data import create_metric_table
from viewer.calculate_rts_data import get_metrics_from_sample_config


class AivPmuModel(ParserModel):
    """
    ffts pmu model.
    """

    def __init__(self: any, result_dir: str) -> None:
        super().__init__(result_dir, DBNameConstant.DB_METRICS_SUMMARY, DBNameConstant.TABLE_AIV_METRIC_SUMMARY)

    def create_table(self: any) -> None:
        """
        create aic and aiv table by sample.json
        :return:
        """
        self.clear()
        aic_profiling_events = get_metrics_from_sample_config(self.result_dir, StrConstant.AIV_PROFILING_METRICS)
        column_list = AicPmuUtils.remove_unused_column(aic_profiling_events)
        create_metric_table(self.conn, column_list, DBNameConstant.TABLE_AIV_METRIC_SUMMARY)

    def flush(self: any, data_list: list) -> None:
        """
        insert data into database
        :param data_list: ffts pmu data list
        :return: None
        """
        self.insert_data_to_db(DBNameConstant.TABLE_AIV_METRIC_SUMMARY, data_list)

    def clear(self: any) -> None:
        """
        clear ai core metric table
        :return: None
        """
        db_path = PathManager.get_db_path(self.result_dir, DBNameConstant.DB_METRICS_SUMMARY)
        if DBManager.check_tables_in_db(db_path, DBNameConstant.TABLE_AIV_METRIC_SUMMARY):
            DBManager.drop_table(self.conn, DBNameConstant.TABLE_AIV_METRIC_SUMMARY)
