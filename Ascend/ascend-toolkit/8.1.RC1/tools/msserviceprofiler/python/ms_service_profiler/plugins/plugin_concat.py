# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.

from collections import defaultdict

import pandas as pd

from ms_service_profiler.plugins.base import PluginBase


class PluginConcat(PluginBase):
    name = "plugin_concat"
    depends = ["plugin_timestamp"]

    @staticmethod
    def _merge_msprof_data(data):
        """合并 msprof_data_df 数据"""
        msprof_merged = []
        for data_single in data:
            value = data_single.get("msprof_data_df")
            if isinstance(value, list):
                msprof_merged.extend(value)
            elif value is not None:
                msprof_merged.append(value)
        return msprof_merged

    @classmethod
    def parse(cls, data):
        merged_data = defaultdict(pd.DataFrame)
        for data_single in data:
            for key, value in data_single.items():
                if isinstance(value, pd.DataFrame):
                    merged_data[key] = pd.concat([merged_data[key], value], ignore_index=True)

        msprof_merged = cls._merge_msprof_data(data)

        if msprof_merged:
            merged_data["msprof_data"] = msprof_merged

        for key, value in merged_data.items():
            if isinstance(value, pd.DataFrame):
                merged_data[key] = value.sort_values(by='start_time', ascending=True).reset_index(drop=True)

        return merged_data

