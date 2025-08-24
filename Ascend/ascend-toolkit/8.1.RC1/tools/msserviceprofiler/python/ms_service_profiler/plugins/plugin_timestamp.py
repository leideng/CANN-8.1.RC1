# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.

import datetime
import psutil

from ms_service_profiler.constant import US_PER_SECOND, NS_PER_US, NS_PER_SECOND
from ms_service_profiler.plugins.base import PluginBase


class PluginTimeStampHelper(PluginBase):
    name = "plugin_timestamp_helper"
    depends = []

    @classmethod
    def parse(cls, data):
        tx_data_df = data.get('tx_data_df')
        cpu_data_df = data.get('cpu_data_df')
        memory_data_df = data.get('memory_data_df')
        time_info = data.get('time_info')
        msprof_data_df = data.get('msprof_data')

        if time_info is None:
            raise ValueError("There is no time information, please check data.")

        calculate_timestamp(tx_data_df, time_info, prof_type='system_count')
        calculate_timestamp(cpu_data_df, time_info, prof_type='system_timestamp')
        calculate_timestamp(memory_data_df, time_info, prof_type='system_timestamp')

        data = {
            'tx_data_df': tx_data_df,
            'cpu_data_df': cpu_data_df,
            'memory_data_df': memory_data_df,
            'msprof_data_df': msprof_data_df
        }
        return data


class PluginTimeStamp(PluginBase):
    name = "plugin_timestamp"
    depends = []
    helper = PluginTimeStampHelper()

    @classmethod
    def parse(cls, data):
        res = []
        for data_single in data:
            res.append(cls.helper.parse(data_single))
        return res


# system_count, 用于tx_data_df中时间戳转换计算
def convert_syscnt_to_ts(cnt, time_info):
    cpu_frequency = time_info.get('cpu_frequency')
    collection_time_begin = time_info.get('collection_time_begin')
    collection_cnt_begin = time_info.get('cntvct')
    host_clock_monotonic_raw = time_info.get('host_clock_monotonic_raw')
    start_clock_monotonic_raw = time_info.get('start_clock_monotonic_raw')

    try:
        '''
            频率不为空，获取的是计数，需要除以频率；
            频率为空，获取的是monotonic，可以直接计算
            collection_time_begin单位为us，其他数据单位为ns
            (cnt - collection_cnt_begin) / cpu_frequency 单位为s
        '''
        if cpu_frequency != 0:
            return collection_time_begin + ((cnt - collection_cnt_begin) / cpu_frequency * NS_PER_SECOND + \
                host_clock_monotonic_raw - start_clock_monotonic_raw) / NS_PER_US
        else:
            return collection_time_begin + (cnt - start_clock_monotonic_raw) / NS_PER_US
    except Exception as ex:
        raise AttributeError("Timestamp format error.") from ex


# system_timestamp, 用于cpu_data_df, memory_data_df中时间戳装换计算
def convert_systs_to_ts(systs, time_info):
    cpu_frequency = time_info.get('cpu_frequency')
    collection_time_begin = time_info.get('collection_time_begin')
    collection_systs_begin = time_info.get('start_clock_monotonic_raw')
    try:
        # collection_time_begin 以微秒us为单位
        # systs/start_systs 以纳秒ns为单位
        # 返回值单位为微秒us
        return collection_time_begin + (systs - collection_systs_begin) / NS_PER_US
    except Exception as ex:
        raise AttributeError("Timestamp format error.") from ex


def timestamp_converter(timestamp):
    date_time = datetime.datetime.fromtimestamp(timestamp / US_PER_SECOND)
    return date_time.strftime("%Y-%m-%d %H:%M:%S:%f")


def calculate_timestamp(df, time_info, prof_type='system_count'):
    if df is None:
        return

    for column_name in ['start_time', 'end_time']:
        if column_name not in df.columns:
            raise KeyError(f'{column_name} not found. Timestamp parsing failed.')

    if prof_type == 'system_count':
        df['start_time'] = convert_syscnt_to_ts(df['start_time'], time_info)
        df['end_time'] = convert_syscnt_to_ts(df['end_time'], time_info)
    else:
        df['start_time'] = convert_systs_to_ts(df['start_time'], time_info)
        df['end_time'] = convert_systs_to_ts(df['end_time'], time_info)

    df['during_time'] = df['end_time'] - df['start_time']
    df['start_datetime'] = df['start_time'].apply(timestamp_converter)
    df['end_datetime'] = df['end_time'].apply(timestamp_converter)
