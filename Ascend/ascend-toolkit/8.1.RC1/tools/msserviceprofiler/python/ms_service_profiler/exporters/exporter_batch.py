# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.

from enum import Enum
from pathlib import Path
import json
import pandas as pd

from ms_service_profiler.exporters.base import ExporterBase
from ms_service_profiler.exporters.utils import save_dataframe_to_csv
from ms_service_profiler.utils.log import logger
from ms_service_profiler.exporters.utils import add_table_into_visual_db


class ExporterBatchData(ExporterBase):
    name = "batch_data"

    @classmethod
    def initialize(cls, args):
        cls.args = args

    @classmethod
    def export(cls, data) -> None:
        df = data.get('tx_data_df')
        if df is None:
            logger.warning("The data is empty, please check")
            return
        # mindie 330将BatchScheduler打点修改为batchFrameworkProcessing，此处做新旧版本的兼容处理
        batch_df = df[(df['name'] == 'BatchSchedule') | (df['name'] == 'modelExec') | \
            (df['name'] == 'batchFrameworkProcessing')]
        if batch_df.empty:
            logger.warning("No batch data found. Please check msproftx.db.")
            return
        try:
            model_df = batch_df[['name', 'res_list', 'start_time', 'end_time', 'batch_size', \
                'batch_type', 'during_time',]]
            model_df = model_df.rename(columns={
            'start_time': 'start_time(microsecond)',
            'end_time': 'end_time(microsecond)',
            'during_time': 'during_time(microsecond)'
        })
        except KeyError as e:
            logger.warning(f"Field '{e.args[0]}' not found in msproftx.db.")
        output = cls.args.output_path

        save_dataframe_to_csv(model_df, output, "batch.csv")

        for col in model_df:
            if model_df[col].dtype == 'object':
                model_df[col] = model_df[col].astype(str)
            if col == 'batch_size':
                model_df[col] = model_df[col].astype(float)

        add_table_into_visual_db(model_df, 'batch')
