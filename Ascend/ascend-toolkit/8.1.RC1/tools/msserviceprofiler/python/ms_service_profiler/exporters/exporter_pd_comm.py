# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.

import pandas as pd
from ms_service_profiler.exporters.base import ExporterBase
from ms_service_profiler.utils.log import logger
from ms_service_profiler.exporters.utils import save_dataframe_to_csv


def process_each_req_group(req_group_df):
    http_req_time = 0
    request_send_time = 0
    request_send_succ_time = 0
    prefill_res_time = 0
    requset_end_time = 0

    for _, record in req_group_df.iterrows():
        name = record.get('name')
        if name is None:
            continue

        event_time = record.get('start_datetime')
        if name == 'receiveReq':
            http_req_time = event_time
        elif name == 'sendReqToD':
            request_send_time = event_time
        elif name == 'sendReqToDSucc':
            request_send_succ_time = event_time
        elif name == 'prefillRes':
            prefill_res_time = event_time
        elif name == 'decodeRes':
            requset_end_time = event_time
    return http_req_time, request_send_time, request_send_succ_time, prefill_res_time, requset_end_time


class ExporterPDComm(ExporterBase):
    name = "pd_comm"
    req_result_list = []

    @classmethod
    def initialize(cls, args):
        cls.args = args

    @classmethod
    def export(cls, data) -> None:
        all_data_df = data['tx_data_df']
        output = cls.args.output_path

        if all_data_df is None:
            logger.warning("The tx_data_df is empty, please check")
            return

        pd_split_df = all_data_df[(all_data_df['domain'] == 'PDSplit')]
        if pd_split_df.empty:
            return

        # 按照rid进行分组
        req_group_df = pd_split_df.groupby('rid')
        for rid, pre_req_data in req_group_df:
            http_req, request_send, request_send_succ, prefill_res, requset_end = process_each_req_group(pre_req_data)
            cls.req_result_list.append({'rid': rid, 'httpReqTime': http_req, 'requestSendTime': request_send, \
                'requestSendSuccTime': request_send_succ, 'prefillResTime': prefill_res, 'requsetEndTime': requset_end})

        save_dataframe_to_csv(pd.DataFrame(cls.req_result_list), output, "pdSplitComm.csv")
