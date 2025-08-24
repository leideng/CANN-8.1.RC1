#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

from msconfig.meta_config import MetaConfig


class TablesTrainingConfig(MetaConfig):
    DATA = {
        'TaskTimeMap': [
            ('replayid', 'INTEGER,null'),
            ('device_id', 'INTEGER,null'),
            ('api', 'INTEGER,null'),
            ('apirowid', 'INTEGER,null'),
            ('tasktype', 'INTEGER,null'),
            ('task_id', 'INTEGER,null'),
            ('stream_id', 'INTEGER,null'),
            ('waittime', 'TEXT,null'),
            ('pendingtime', 'Text,null'),
            ('runtime', 'TEXT,null'),
            ('complete', 'TEXT,null')
        ],
        'TsOriginalDataMap': [
            ('replayid', 'INTEGER,null'),
            ('timestamp', 'numeric,null'),
            ('pc', 'TEXT,null'),
            ('callstack', 'TEXT,null'),
            ('event', 'TEXT,null'),
            ('count', 'INTEGER,null'),
            ('function', 'TEXT,null')
        ],
        'RoceOriginalDataMap': [
            ('device_id', 'INTEGER,null'),
            ('replayid', 'INTEGER,null'),
            ('timestamp', 'REAL,null'),
            ('bandwidth', 'INTEGER,null'),
            ('rxpacket', 'REAL,null'),
            ('rxbyte', 'REAL,null'),
            ('rxpackets', 'REAL,null'),
            ('rxbytes', 'REAL,null'),
            ('rxerrors', 'REAL,null'),
            ('rxdropped', 'REAL,null'),
            ('txpacket', 'REAL,null'),
            ('txbyte', 'REAL,null'),
            ('txpackets', 'REAL,null'),
            ('txbytes', 'REAL,null'),
            ('txerrors', 'REAL,null'),
            ('txdropped', 'REAL,null'),
            ('funcid', 'INTEGER,null')
        ],
        'RoceReportDataMap': [
            ('device_id', 'INTEGER,null'),
            ('duration', 'TEXT,null'),
            ('bandwidth', 'TEXT,null'),
            ('rxbandwidth', 'TEXT,null'),
            ('txbandwidth', 'TEXT,null'),
            ('rxpacket', 'TEXT,null'),
            ('rxerrorrate', 'TEXT,null'),
            ('rxdroppedrate', 'TEXT,null'),
            ('txpacket', 'TEXT,null'),
            ('txerrorrate', 'TEXT,null'),
            ('txdroppedrate', 'TEXT,null'),
            ('funcid', 'INTEGER,null')
        ],
        'StreamMap': [
            ('replayid', 'INTEGER, null'),
            ('device_id', 'INTEGER, null'),
            ('stream_id', 'INTEGER, null'),
            ('task_id', 'INTEGER, null'),
            ('tasktype', 'INTEGER, null'),
            ('waittime', 'INTEGER, null'),
            ('pendingtime', 'INTEGER,null'),
            ('runtime', 'INTEGER,null'),
            ('completetime', 'INTEGER, null'),
            ('api', 'INTEGER, null'),
            ('apirowid', 'INTEGER, null'),
            ('eventid', 'INTEGER, null'),
            ('streamname', 'TEXT, null')
        ],
        'LLCOriginalDataMap': [
            ('device_id', 'INT,null'),
            ('timestamp', 'REAL,null'),
            ('counts', 'INT,null'),
            ('event', 'INT,null'),
            ('l3tid', 'INT,null')
        ],
        'LLCEventsMap': [
            ('device_id', 'INT,null'),
            ('l3tid', 'INT,null'),
            ('timestamp', 'REAL,null'),
            ('event0', 'INT,null'),
            ('event1', 'INT,null'),
            ('event2', 'INT,null'),
            ('event3', 'INT,null'),
            ('event4', 'INT,null'),
            ('event5', 'INT,null'),
            ('event6', 'INT,null'),
            ('event7', 'INT,null')
        ],
        'LLCMetricsMap': [
            ('device_id', 'INT,null'),
            ('l3tid', 'INT,null'),
            ('timestamp', 'REAL,null'),
            ('hitrate', 'REAL,null'),
            ('throughput', 'REAL,null')
        ],
        'HBMOriginalDataMap': [
            ('device_id', 'INT,null'),
            ('replayid', 'INT,null'),
            ('timestamp', 'REAL,null'),
            ('counts', 'INT,null'),
            ('event_type', 'TEXT,null'),
            ('hbmid', 'INT,null')
        ],
        'HBMbwDataMap': [
            ('device_id', 'INT,null'),
            ('timestamp', 'REAL,null'),
            ('bandwidth', 'REAL,null'),
            ('hbmid', 'INT,null'),
            ('event_type', 'TEXT,null')
        ],
        'HCCSOriginalDataMap': [
            ('device_id', 'INT,null'),
            ('timestamp', 'REAL,null'),
            ('txamount', 'INT,null'),
            ('rxamount', 'INT,null')
        ],
        'HCCSEventsDataMap': [
            ('device_id', 'INT,null'),
            ('timestamp', 'REAL,null'),
            ('txthroughput', 'INT,null'),
            ('rxthroughput', 'INT,null')
        ],
        'HWTSTaskTimeMap': [
            ('device_id', 'INTEGER,null'),
            ('stream_id', 'INTEGER,null'),
            ('task_id', 'INTEGER,null'),
            ('running', 'INTEGER,null'),
            ('complete', 'INTEGER,null'),
            ('index_id', 'INTEGER,null')
        ],
        'PCIeDataMap': [
            ('timestamp', 'INT,null'),
            ('device_id', 'INT,null'),
            ('tx_p_bandwidth_min', 'REAL,null'),
            ('tx_p_bandwidth_max', 'REAL,null'),
            ('tx_p_bandwidth_avg', 'REAL,null'),
            ('tx_np_bandwidth_min', 'REAL,null'),
            ('tx_np_bandwidth_max', 'REAL,null'),
            ('tx_np_bandwidth_avg', 'REAL,null'),
            ('tx_cpl_bandwidth_min', 'REAL,null'),
            ('tx_cpl_bandwidth_max', 'REAL,null'),
            ('tx_cpl_bandwidth_avg', 'REAL,null'),
            ('tx_np_lantency_min', 'REAL,null'),
            ('tx_np_lantency_max', 'REAL,null'),
            ('tx_np_lantency_avg', 'REAL,null'),
            ('rx_p_bandwidth_min', 'REAL,null'),
            ('rx_p_bandwidth_max', 'REAL,null'),
            ('rx_p_bandwidth_avg', 'REAL,null'),
            ('rx_np_bandwidth_min', 'REAL,null'),
            ('rx_np_bandwidth_max', 'REAL,null'),
            ('rx_np_bandwidth_avg', 'REAL,null'),
            ('rx_cpl_bandwidth_min', 'REAL,null'),
            ('rx_cpl_bandwidth_max', 'REAL,null'),
            ('rx_cpl_bandwidth_avg', 'REAL,null')
        ],
        'ModifiedTaskTimeMap': [
            ('task_id', 'INTEGER, null'),
            ('stream_id', 'INTEGER, null'),
            ('start_time', 'INTEGER,null'),
            ('duration_time', 'INTEGER, null'),
            ('wait_time', 'INTEGER, null'),
            ('task_type', 'INTEGER,null'),
            ('index_id', 'INTEGER,null'),
            ('batch_id', 'INTEGER,null')
        ],
        'GeMergeMap': [
            ('model_id', 'INTEGER,null'),
            ('op_name', 'text,null'),
            ('op_type', 'text,null'),
            ('task_type', 'text,null'),
            ('task_id', 'INTEGER,null'),
            ('stream_id', 'INTEGER,null'),
            ('device_id', 'INTEGER,null'),
            ('batch_id', 'INTEGER,null')
        ],
        'RtsTaskMap': [
            ('task_id', 'INTEGER,null'),
            ('stream_id', 'INTEGER,null'),
            ('start_time', 'INTEGER,null'),
            ('duration', 'INTEGER,null'),
            ('task_type', 'text,null'),
            ('index_id', 'INTEGER,null'),
            ('batch_id', 'INTEGER,null')
        ],
        'OpReportMap': [
            ('op_type', 'text,null'),
            ('core_type', 'text,null'),
            ('occurrences', 'text,null'),
            ('total_time', 'REAL,null'),
            ('min', 'REAL,null'),
            ('avg', 'REAL,null'),
            ('max', 'REAL,null'),
            ('ratio', 'text,null'),
            ('device_id', 'INTEGER,null')
        ]
    }
