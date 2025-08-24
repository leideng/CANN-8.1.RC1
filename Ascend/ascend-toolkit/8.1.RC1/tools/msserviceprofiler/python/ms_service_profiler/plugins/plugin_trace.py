# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.

import pandas as pd
from ms_service_profiler.plugins.base import PluginBase


class PluginTrace(PluginBase):
    name = "plugin_trace"
    depends = ["plugin_common", "plugin_req_status"]

    @classmethod
    def parse(cls, data):
        tx_data_df = data.get('tx_data_df')
        if tx_data_df is None:
            raise ValueError("tx_data_df is None")

        # 解析batch，vllm数据解析modelexec自带batch_type字段
        if 'batch_type' not in tx_data_df.columns:
            tx_data_df['batch_type'] = None
        tx_data_df['batch_type'] = [
            extract_batch_type(token_list, batch_type)
            for token_list, batch_type in zip(tx_data_df['token_id_list'], tx_data_df['batch_type'])
        ]

        tx_data_df['batch_size'] = [extract_batch_size(x) for x in tx_data_df['rid_list']]

        # PD混跑场景拆解batch size
        tx_data_df['prefill_batch_size'], tx_data_df['decode_batch_size'] = zip(
            *[
                extract_batch_size_when_pd_mixed(token_list) if batch_type == 'Prefill, Decode' else (None, None)
                for token_list, batch_type in zip(tx_data_df['token_id_list'], tx_data_df['batch_type'])
            ]
        )

        data['tx_data_df'] = tx_data_df
        return data


def extract_batch_type(token_list, batch_type):
    if batch_type is not None:
        return batch_type
    if token_list is None:
        return None
    if all(token == 0 for token in token_list):
        return 'Prefill'
    elif 0 in token_list and len(set(token_list)) > 1:
        return 'Prefill, Decode'
    else:
        return 'Decode'


def extract_batch_size(rid_list):
    if rid_list is None:
        return None
    return str(int(len(rid_list)))


def extract_batch_size_when_pd_mixed(token_list):
    prefill_batch_size = token_list.count(0)
    decode_batch_size = len(token_list) - prefill_batch_size
    return prefill_batch_size, decode_batch_size


