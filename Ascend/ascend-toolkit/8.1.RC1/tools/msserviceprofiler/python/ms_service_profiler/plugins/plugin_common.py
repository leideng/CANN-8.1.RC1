# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.

import json
import pandas as pd
import numpy as np
from ms_service_profiler.plugins.base import PluginBase
from ms_service_profiler.plugins.plugin_vllm_helper import VllmHelper
from ms_service_profiler.utils.log import logger
from ms_service_profiler.utils.error import ParseError, DataFrameMissingError, KeyMissingError, ValidationError


class PluginCommon(PluginBase):
    name = "plugin_common"
    depends = ["plugin_concat"]

    @classmethod
    def parse(cls, data):
        tx_data_df = data.get("tx_data_df")
        if tx_data_df is None:
            raise DataFrameMissingError("tx_data_df")

        tx_data_df = tx_data_df.replace(to_replace=np.nan, value=None)
        data["tx_data_df"], data["rid_link_map"] = parse_rid(tx_data_df)
        
        return data


# 只有vllm框架数据解析会走这部分流程，从batchSchdule中的iter_size中获取迭代信息
def extract_iter_from_batch(req):
    rid = req.get('rid')
    iter_size = req.get('iter_size')
    return VllmHelper.add_req_batch_iter(rid, iter_size)


def extract_ids_from_reslist(rid_from_message, rid_map):
    if not rid_from_message:
        return [], []

    rid = []
    token_id = []

    for req in rid_from_message:
        if isinstance(req, int) or isinstance(req, float):
            rid.append(int(req))
            token_id.append(None)
        elif isinstance(req, dict):
            rid.append(rid_map.get(req.get('rid', None), req.get('rid', None)))
            # iter_size 为vllm数据采集特有字段
            if req.get('iter_size'):
                token_id.append(extract_iter_from_batch(req))
            else:
                token_id.append(req.get('iter', None))
        elif isinstance(req, str):
            rid.append(req)
            token_id.append(None)

    return rid, token_id


def extract_rid(rid_from_message, rid_map):
    rid, rid_list, token_id_list = None, None, None
    if rid_from_message is not None:
        if isinstance(rid_from_message, str):
            rid = str(rid_map.get(rid_from_message, rid_from_message))
        elif isinstance(rid_from_message, list):
            rid_list, token_id_list = extract_ids_from_reslist(rid_from_message, rid_map)
            rid = ','.join(map(str, rid_list))
        else:
            rid = str(rid_from_message)

    return rid, rid_list, token_id_list


def parse_rid_map(all_data_df):
    df = all_data_df[all_data_df["type"] == 3]  # already checked 'type' in all_data_df
    if "from" in df.columns and "to" in df.columns:
        rid_link_map = dict(zip(df['from'], df['to']))
    else:
        rid_link_map = {}

    try:
        rid_link_map = {k: int(v) for k, v in rid_link_map.items()}
    except Exception as ex:
        logger.error(f'rid must be integer. {ex}')
        raise

    return rid_link_map


def parse_rid(tx_data_df):
    if "type" not in tx_data_df.columns or "rid" not in tx_data_df.columns:
        logger.error(f'Missing columns "type" or "rid". Skip parsing')
        return tx_data_df, None
    tx_data_df['res_list'] = tx_data_df['rid']
    rid_link_map = parse_rid_map(tx_data_df)

    df = tx_data_df['rid'].apply(lambda x: extract_rid(x, rid_link_map))
    tx_data_df[['rid', 'rid_list', 'token_id_list']] = pd.DataFrame(df.tolist(), index=tx_data_df.index)

    return tx_data_df, rid_link_map
