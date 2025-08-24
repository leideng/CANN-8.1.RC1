# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.

import os
import datetime
from decimal import Decimal
import sqlite3
import numpy as np
import pandas as pd
from ms_service_profiler.exporters.base import ExporterBase
from ms_service_profiler.utils.log import logger
from ms_service_profiler.exporters.utils import add_table_into_visual_db


def is_contained_vaild_iter_info(rid_list, token_id_list):
    if rid_list is None or token_id_list is None or len(rid_list) != len(token_id_list):
        return False

    return True


def print_warning_log(log_name):
    if not ExporterLatency.get_err_log_flag(log_name):
        logger.warning(f"The '{log_name}' field info is missing, please check.")
        ExporterLatency.set_err_log_flag(log_name, True)


def process_each_record(req_map, record):
    name = record.get('name')
    rid = record.get('rid')
    if rid is None or name is None:
        return

    if name == 'httpReq':
        req_map[rid] = {}
        req_map[rid]['start_time'] = record.get('start_time')
        return

    if req_map.get(rid) is not None:
        if name == 'httpRes':
            req_map[rid]['end_time'] = record.get('end_time')
        req_map[rid]['req_exec_time'] = record.get('end_time')

    rid_list = record.get('rid_list')
    token_id_list = record.get('token_id_list')
    if not is_contained_vaild_iter_info(rid_list, token_id_list):
        return

    for i, value in enumerate(rid_list):
        req_rid = str(value)
        if req_map.get(req_rid) is None:
            print_warning_log('httpReq')
            continue

        req_map[req_rid]['req_exec_time'] = record.get('end_time')

        # 更新请求首token时延
        cur_iter = token_id_list[i]
        if cur_iter is None:
            continue

        if cur_iter == 0:
            if req_map[req_rid].get('first_token_latency') is None:
                req_map[req_rid]['first_token_latency'] = record.get('during_time')
            else:
                req_map[req_rid]['first_token_latency'] += record.get('during_time')

        # 更新请求生成token数量
        gen_token_num = cur_iter + 1
        if record.get('batch_type') == 'Prefill':
            if req_map[req_rid].get('prefill_token_num') is None or \
                req_map[req_rid]['prefill_token_num'] < gen_token_num:
                req_map[req_rid]['prefill_token_num'] = gen_token_num
        elif record.get('batch_type') == 'Decode':
            if req_map[req_rid].get('decode_token_num') is None or \
                req_map[req_rid]['decode_token_num'] < gen_token_num:
                req_map[req_rid]['decode_token_num'] = gen_token_num


def get_percentile_results(metric):
    if not metric or any(not isinstance(value, (int, float)) for value in metric):
        return np.nan, np.nan, np.nan, np.nan
    avg = round(np.average(metric), 4)
    p50, p90, p99 = np.round(np.percentile(metric, [50, 90, 99]), 4)
    return avg, p50, p90, p99


def calculate_first_token_latency(req_map):
    first_token_latency = []
    for _, req_info in req_map.items():
        # 计算首token时延，µs级
        if req_info.get('first_token_latency') is not None:
            first_token_latency.append(round(req_info['first_token_latency'], 4))
    
    return get_percentile_results(first_token_latency)


def calculate_req_latency(req_map):
    req_latency = []
    for _, req_info in req_map.items():
        if req_info.get('start_time') is None:
            print_warning_log('start_time')
            continue
        cur_req_start_time = req_info['start_time']

        # 计算请求端到端时延，µs级
        if req_info.get('end_time') is not None:
            cur_req_end_time = req_info['end_time']
            cur_req_latency = cur_req_end_time - cur_req_start_time
            req_latency.append(round(cur_req_latency, 4))
    return get_percentile_results(req_latency)


def calculate_gen_token_speed_latency(req_map, is_prefill):
    gen_token_speed = []
    for _, req_info in req_map.items():
        if req_info.get('start_time') is None:
            print_warning_log('start_time')
            continue
        cur_req_start_time = req_info['start_time']

        cur_req_gen_token_num = 0
        try:
            if is_prefill:
                # 计算prefill token平均时延
                cur_req_gen_token_num = req_info['prefill_token_num']
            else:
                # 计算decode token平均时延
                cur_req_gen_token_num = req_info['decode_token_num']

            # 计算生成token执行时间
            gen_last_token_time = req_info['req_exec_time']
            if gen_last_token_time <= cur_req_start_time:
                raise ValueError("The execution time for generating the token is a negative number.")
            diff_time = gen_last_token_time - cur_req_start_time

            # 计算生成token平均时延，s级
            cur_gen_speed = round(cur_req_gen_token_num / (diff_time / 1000000), 4) # 1000000:换算为秒级
            gen_token_speed.append(cur_gen_speed)
        except KeyError as e:
            # 并发场景下，若请求到达后还未生成token，则跳过当前请求不计算
            continue

    return get_percentile_results(gen_token_speed)


def gen_exporter_results(all_data_df):
    req_map = {}
    first_token_latency_views = []
    req_latency_views = []
    prefill_gen_speed_views = []
    decode_gen_speed_views = []

    for _, record in all_data_df.iterrows():
        process_each_record(req_map, record)

        # 生成首token时延
        if record.get('batch_type') == 'Prefill':
            avg, p50, p90, p99 = calculate_first_token_latency(req_map)
            cur_timestamp = record.get('end_datetime')
            first_token_latency_views.append({'timestamp': cur_timestamp, \
                'avg': avg, 'p99': p99, 'p90': p90, 'p50': p50})

        # 生成请求端到端时延
        if record.get('name') == 'httpRes':
            avg, p50, p90, p99 = calculate_req_latency(req_map)
            cur_timestamp = record.get('end_datetime')
            req_latency_views.append({'timestamp': cur_timestamp, \
                'avg': avg, 'p99': p99, 'p90': p90, 'p50': p50})

        # 生成token平均时延
        if is_contained_vaild_iter_info(record.get('rid_list'), record.get('token_id_list')):
            cur_timestamp = record.get('end_datetime')
            if record.get('batch_type') == 'Prefill':
                avg, p50, p90, p99 = calculate_gen_token_speed_latency(req_map, True)
                prefill_gen_speed_views.append({'timestamp': cur_timestamp, \
                    'avg': avg, 'p99': p99, 'p90': p90, 'p50': p50})
            if record.get('batch_type') == 'Decode':
                avg, p50, p90, p99 = calculate_gen_token_speed_latency(req_map, False)
                decode_gen_speed_views.append({'timestamp': cur_timestamp, \
                    'avg': avg, 'p99': p99, 'p90': p90, 'p50': p50})

    return first_token_latency_views, req_latency_views, prefill_gen_speed_views, decode_gen_speed_views


class ExporterLatency(ExporterBase):
    name = "latency"
    err_log = {'rid or name': False, 'start_time': False, 'httpReq': False, 'token_id_list': False}

    @classmethod
    def initialize(cls, args):
        cls.args = args
        cls.err_log = {'rid or name': False, 'start_time': False, 'httpReq': False, 'token_id_list': False}

    @classmethod
    def set_err_log_flag(cls, index, value):
        cls.err_log[index] = value

    @classmethod
    def get_err_log_flag(cls, index):
        return cls.err_log[index]

    @classmethod
    def export(cls, data) -> None:
        all_data_df = data['tx_data_df']
        output = cls.args.output_path

        first_token_latency_views, req_latency_views, prefill_gen_speed_views, decode_gen_speed_views = \
            gen_exporter_results(all_data_df)

        add_table_into_visual_db(pd.DataFrame(first_token_latency_views), 'first_token_latency')
        add_table_into_visual_db(pd.DataFrame(req_latency_views), 'req_latency')
        add_table_into_visual_db(pd.DataFrame(prefill_gen_speed_views), 'prefill_gen_speed')
        add_table_into_visual_db(pd.DataFrame(decode_gen_speed_views), 'decode_gen_speed')
