# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.

from enum import Enum
from pathlib import Path
import json
import pandas as pd

from ms_service_profiler.exporters.utils import save_dataframe_to_csv
from ms_service_profiler.exporters.base import ExporterBase

from ms_service_profiler.utils.log import logger


def update_name(row):
    if row['RUNNING+'] == 1:
        row['name'] = 'RUNNING'
    elif row['PENDING+'] == 1:
        row['name'] = 'PENDING'
    return row


def process_data(req_en_queue_df, req_running_df, pending_df):
    """
    处理数据，计算等待时间和执行时间。

    参数:
    req_en_queue_df (pd.DataFrame): 请求队列的数据
    req_running_df (pd.DataFrame): 正在运行的请求的数据
    pending_df (pd.DataFrame): 等待中的请求的数据

    返回:
    wait_df (pd.DataFrame): 包含等待时间和执行时间的DataFrame
    """
    # 分组并取第一个
    decode_first_df = req_en_queue_df.groupby('rid').head(1)
    running_first_df = req_running_df.groupby('rid').head(1)


    # 计算prefill阶段的等待时间
    if decode_first_df.shape[0] != running_first_df.shape[0]:
        logger.warning("The number of 'Enqueue' is different from 'RUNNING' in the prefill phase, please check")
    prefill_df = pd.merge(decode_first_df, running_first_df, on=['rid'], how='left', suffixes=('_enque', '_running'))
    prefill_df['waiting_time'] = prefill_df["start_time_running"] - prefill_df["end_time_enque"]

    # 计算decode阶段的等待时间
    decode_running_df = req_running_df.groupby('rid').apply(lambda x: x.iloc[1:]).reset_index(drop=True)
    pending_df = pending_df.reset_index(drop=True)
    pending_df = pending_df[['start_time', 'end_time', 'rid']]
    decode_running_df = decode_running_df[['start_time', 'end_time', 'rid']]

    pending_df = pending_df.groupby('rid').agg({'start_time': ["sum", "count"]}).reset_index()
    pending_df = pending_df.sort_values("rid")
    pending_df = pd.merge(pending_df["start_time"], pending_df["rid"], left_index=True, right_index=True)
    pending_df.columns = ['start_time', 'count', 'rid']
 
    decode_running_df = decode_running_df.groupby('rid').agg({'start_time': ["sum", "count"]}).reset_index()
    decode_running_df = decode_running_df.sort_values("rid")
    decode_running_df = pd.merge(decode_running_df["start_time"], decode_running_df["rid"], \
        left_index=True, right_index=True)
    decode_running_df.columns = ['start_time', 'count', 'rid']

    rows_pending = pending_df.shape[0]
    rows_running = decode_running_df.shape[0]
    if rows_pending != rows_running:
        logger.warning("The number of 'PENDING' is different from 'RUNNING' in the decode phase , please check")
    decode_merge = pd.concat([pending_df, decode_running_df], ignore_index=True, axis=1)

    decode_merge.columns = ['start_time_pending', 'end_time_pending', 'rid', 'start_time_running', \
        'end_time_running', 'rid_running']
    decode_merge["pending_time"] = decode_merge['start_time_running'] - decode_merge['start_time_pending']

    decode_merge = decode_merge.drop(columns=['start_time_running', 'end_time_running'])
    pending_time_sum = decode_merge.groupby('rid')['pending_time'].sum().reset_index()

    # 计算总的等待时间
    if prefill_df.shape[0] != pending_time_sum.shape[0]:
        logger.warning("The waiting time length in the prefill phase is different from that in the decode phase.")

    pending_time_sum.set_index('rid', inplace=True)
    wait_df = pd.merge(prefill_df, pending_time_sum, on='rid', how='left')

    wait_df['queue_wait_time'] = wait_df['waiting_time'] + wait_df['pending_time']
    wait_df['rid'] = wait_df['rid'].apply(str)
    wait_df = wait_df[['rid', 'queue_wait_time']]
    return wait_df


def get_wait_df(df):
    req_en_queue_df = df[df['name'] == 'Enqueue']
    req_running_df = df[df['name'] == 'RUNNING']
    pending_df = df[df['name'] == 'PENDING']
    if pending_df.empty:
        # vllm采集不到PENDING，用WAITING替代
        pending_df = df[df['name'] == 'WAITING']
    wait_df = process_data(req_en_queue_df, req_running_df, pending_df)
    return wait_df


def is_invaild_rid(rid):
    return ',' in rid or '{' in rid or ':' in rid


def get_req_base_info(df):
    req_group_df = df.groupby('rid')
    req_base_info = []
    for rid, pre_req_data in req_group_df:
        rid = str(rid)
        if rid == "" or is_invaild_rid(rid):
            continue
        new_req = {
            'rid': rid,
            'start_time': '',
            'end_time': '',
            'recvTokenSize=': '',
            'replyTokenSize=': '',
            'execution_time': ''
        }

        # 获取httpReq
        http_req_df = pre_req_data[pre_req_data['name'] == 'httpReq']
        if not http_req_df.empty:
            first_row = http_req_df.iloc[0]
            new_req['start_time'] = first_row.get("start_time", 0)

        # 获取 httpRes
        # 由于存在httpRes提前被调用，导致请求结束时间过早的情况，所以当前取httpRes和DecodeEnd中最晚一个点作为请求结束时间
        http_res_df = pre_req_data[pre_req_data['name'].isin(['httpRes', 'DecodeEnd'])]
        if not http_res_df.empty:
            last_row = http_res_df.iloc[-1]
            new_req['end_time'] = last_row.get("end_time", 0)

        # 获取replyTokenSize
        if 'replyTokenSize=' in pre_req_data.columns and pre_req_data['replyTokenSize='].notna().any():
            # 获取当replyTokenSize列中值不为空时，获取其中的第一个值
            new_req['replyTokenSize='] = pre_req_data['replyTokenSize='].dropna().iloc[0]

        # 获取 recvTokenSize=
        if 'recvTokenSize=' in pre_req_data.columns and pre_req_data['recvTokenSize='].notna().any():
            # 获取当replyTokenSize列中值不为空时，获取其中的第一个值
            new_req['recvTokenSize='] = pre_req_data['recvTokenSize='].dropna().iloc[0]

        # 计算 execution_time
        if new_req['start_time'] != '' and new_req['end_time'] != '':
            new_req['execution_time'] = new_req['end_time'] - new_req['start_time']

        req_base_info.append(new_req)
    return pd.DataFrame(req_base_info)


class ExporterReqData(ExporterBase):
    name = "req_data"

    @classmethod
    def initialize(cls, args):
        cls.args = args

    @classmethod
    def export(cls, data) -> None:
        df = data.get('tx_data_df')
        if df is None:
            logger.error("The data is empty, please check")
            return
        output = cls.args.output_path

        df = df[df['domain'] != 'KVCache']
        req_base_info = get_req_base_info(df)
        try:
            df = df.rename(columns={"RUNNING+": "RUNNING", "PENDING+": "PENDING"})
            wait_df = get_wait_df(df)
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return

        # 使用merge操作将req_base_info和wait_df的数据进行匹配
        if req_base_info.shape[0] != wait_df.shape[0]:
            logger.warning("The shape of 'req_base_info' is different from 'wait_df', please check.")

        df_merged = pd.merge(req_base_info, wait_df, on='rid', how='outer', indicator=True)
        df_merged.drop(columns=['_merge'], inplace=True)

        filtered_df = df_merged[['rid', 'start_time', 'recvTokenSize=', 'replyTokenSize=', \
            'execution_time', 'queue_wait_time']]
        filtered_df = filtered_df.rename(columns={
            'rid': 'http_rid',
            'recvTokenSize=': 'recv_token_size',
            'replyTokenSize=': 'reply_token_size',
            'start_time': 'start_time_httpReq(microsecond)',
            'execution_time': 'execution_time(microsecond)',
            'queue_wait_time': 'queue_wait_time(microsecond)'
        })

        save_dataframe_to_csv(filtered_df, output, "request.csv")
