#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import os
import logging
from collections import namedtuple
from collections import defaultdict

from common_func.ms_constant.str_constant import StrConstant
from common_func.ms_multi_process import MsMultiProcess
from common_func.db_name_constant import DBNameConstant
from common_func.path_manager import PathManager
from common_func.profiling_scene import ProfilingScene
from common_func.db_manager import DBManager
from common_func.ms_constant.number_constant import NumberConstant
from common_func.info_conf_reader import InfoConfReader
from common_func.msprof_object import CustomizedNamedtupleFactory
from common_func.hccl_info_common import DeviceHcclSource
from msmodel.task_time.ascend_task_model import AscendTaskViewModel
from msmodel.ge.ge_info_model import GeInfoViewModel
from msmodel.add_info.kfc_info_model import KfcInfoViewModel
from msmodel.add_info.mc2_comm_info_model import Mc2CommInfoViewModel
from msmodel.hccl.hccl_model import HCCLModel
from mscalculate.interface.icalculator import ICalculator
from mscalculate.flip.flip_calculator import FlipCalculator
from mscalculate.hccl.hccl_calculator import HcclCalculator
from msmodel.hccl.hccl_model import HcclViewModel


class KfcCalculator(ICalculator, MsMultiProcess):
    KFC_OP_DATA = CustomizedNamedtupleFactory.enhance_namedtuple(
        namedtuple("KfcOpData",
                   ["ascend_data", "group_name", "op_name", "first_timestamp", "iter_id", "op_type",
                    "relay", "retry", "data_type", "alg_type", "count", "rank_size", "source"]),
        {})
    BLACK_KFC_OP_TYPE = ["NOTIFY_WAIT", "MEMCPY_ASYNC"]
    BLACK_KFC_OP_NAME = ["hcomAicpuInit"]
    MC2_MASTER_STREAM_TASK_TYPE = "C_CORE_SQE"
    FIRST_TASK_TYPE = 0
    LAST_TASK_TYPE = 1

    def __init__(self, file_list, sample_config):
        super().__init__(sample_config)
        self._file_list = file_list
        self._project_path = sample_config.get(StrConstant.SAMPLE_CONFIG_PROJECT_PATH)
        self._kfc_op_data = {}
        self._kfc_task_data = {}
        self._kfc_small_task = {}
        self._plane_id = {}
        self._master_stream = {}
        self._master_stream_first_task = set()
        self._source = {}

    @staticmethod
    def make_default_kfc_info() -> KfcInfoViewModel.KFC_HCCL_INFO_TYPE:
        default_kfc_info = KfcInfoViewModel.KFC_HCCL_INFO_TYPE(
            0, "N/A", 'N/A', 'N/A', 4294967295, 4294967295, -1, 0, -1, 4294967295, 'N/A', 'N/A', 'N/A', -1,
            'N/A', 'N/A', -1, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', -1, -1, 0, 0, 0, 0, 'N/A', -1
        )
        return default_kfc_info

    @staticmethod
    def get_hccl_op_info(op_data, idx, hccl_op_info):
        while idx < len(hccl_op_info) and hccl_op_info[idx].timestamp < op_data.ascend_data.timestamp:
            idx += 1
        curr_hccl_op_info = None
        if idx < len(hccl_op_info) and \
                op_data.ascend_data.timestamp <= hccl_op_info[idx].timestamp <= \
                op_data.ascend_data.timestamp + op_data.ascend_data.duration:
            curr_hccl_op_info = hccl_op_info[idx]
            idx += 1
        return curr_hccl_op_info, idx

    @staticmethod
    def update_op_data_time(op_data, time_idx, timestamp_master_stream_task_table):
        timestamp_list = list(timestamp_master_stream_task_table.keys())
        while time_idx < len(timestamp_list) and timestamp_list[time_idx] < op_data.ascend_data.timestamp:
            time_idx += 1
        aicpu_timestamp = op_data.ascend_data.timestamp
        aicpu_duration = op_data.ascend_data.duration
        while time_idx < len(timestamp_list) and \
                aicpu_timestamp <= timestamp_list[time_idx] <= aicpu_timestamp + aicpu_duration:
            data_type, task = timestamp_master_stream_task_table.get(timestamp_list[time_idx])
            if data_type == KfcCalculator.FIRST_TASK_TYPE:
                # first task
                ascend_data = op_data.ascend_data.replace(timestamp=task.start_time)
                op_data = op_data.replace(ascend_data=ascend_data)
            # last task
            duration = task.start_time + task.duration - op_data.ascend_data.timestamp
            ascend_data = op_data.ascend_data.replace(duration=duration)
            op_data = op_data.replace(ascend_data=ascend_data)
            time_idx += 1
        return op_data, time_idx

    @staticmethod
    def update_op_name_by_group(kfc_op_data: KFC_OP_DATA):
        group_dict = defaultdict(lambda: {"first_timestamp": 0, "count": 0})
        for num, data in enumerate(kfc_op_data):
            if data.group_name in group_dict:
                if data.first_timestamp > group_dict[data.group_name]["first_timestamp"]:
                    group_dict[data.group_name]["first_timestamp"] = data.first_timestamp
                    group_dict[data.group_name]["count"] += 1
                elif data.first_timestamp < group_dict[data.group_name]["first_timestamp"]:
                    group_dict[data.group_name]["first_timestamp"] = data.first_timestamp
                    group_dict[data.group_name]["count"] = 0
            else:
                group_dict[data.group_name]["first_timestamp"] = data.first_timestamp
                group_dict[data.group_name]["count"] = 0
            index = group_dict[data.group_name]["count"]
            kfc_op_data[num] = data.replace(
                op_name=f"{data.op_name}_{data.group_name[-3:]}_{str(index)}_{str(data.iter_id)}")

    def calculate(self: any) -> None:
        self.calculate_kfc_op()
        self.calculate_kfc_task()

    def save(self: any) -> None:
        if self._kfc_op_data:
            kfc_op_data = []
            for data in self._kfc_op_data.values():
                kfc_op_data.extend(data)
            with HCCLModel(self._project_path, [DBNameConstant.TABLE_KFC_OP]) as model:
                model.flush(kfc_op_data, DBNameConstant.TABLE_KFC_OP)
        if self._kfc_task_data:
            kfc_task_data = []
            for data in self._kfc_task_data.values():
                kfc_task_data.extend(data)
            with HCCLModel(self._project_path, [DBNameConstant.TABLE_KFC_TASK]) as model:
                model.flush(kfc_task_data, DBNameConstant.TABLE_KFC_TASK)

    def get_mc2_comm_info_data(self: any) -> tuple:
        with Mc2CommInfoViewModel(self._project_path, [DBNameConstant.TABLE_MC2_COMM_INFO]) as model:
            comm_info = model.get_kfc_stream(DBNameConstant.TABLE_MC2_COMM_INFO)
        kfc_stream_id_group_table = {}
        comm_stream_id_group_table = {}
        for info in comm_info:
            kfc_stream_id_group_table[info.aicpu_kfc_stream_id] = info.group_name
            comm_stream_list = []
            try:
                comm_stream_list = list(map(int, info.comm_stream_ids.split(",")))
            except Exception:
                logging.error("The comm_stream_ids is not number")
            for stream_id in comm_stream_list:
                comm_stream_id_group_table.setdefault(stream_id, set()).add(info.group_name)
        return kfc_stream_id_group_table, comm_stream_id_group_table

    def get_kfc_data(self:any, kfc_stream_id: dict, comm_stream_ids: dict) -> tuple:
        kfc_op_data = []
        kfc_comm_task_data = []
        conn, curs = DBManager.check_connect_db(self._project_path, DBNameConstant.DB_ASCEND_TASK)
        if conn and curs:
            DBManager.destroy_db_connect(conn, curs)
            with AscendTaskViewModel(self._project_path, [DBNameConstant.TABLE_ASCEND_TASK]) as model:
                # 由于图模式下，aicpu kernel不在aicpu流上，所以约定aicpu的通信下发算子名以AicpuKernel结尾
                # 通过算子名和aicpu流筛选aicpu算子
                kfc_op_data = model.get_ascend_task_data_with_op_name_pattern_and_stream_id(
                    InfoConfReader().get_device_id(),
                    StrConstant.AICPU_KERNEL,
                    tuple(kfc_stream_id.keys())
                )
                kfc_comm_task_data = model.get_ascend_task_data_with_stream_id(tuple(comm_stream_ids.keys()))
        return kfc_op_data, kfc_comm_task_data

    def get_kfc_op_data(self: any) -> tuple:
        kfc_stream_id_group_table, comm_stream_id_group_table = self.get_mc2_comm_info_data()
        kfc_op_data, kfc_comm_task_data = self.get_kfc_data(kfc_stream_id_group_table, comm_stream_id_group_table)
        kfc_comm_task_data = FlipCalculator.set_device_batch_id(kfc_comm_task_data,
                                                                self._project_path, is_flip_num=True)
        with KfcInfoViewModel(self._project_path, [DBNameConstant.TABLE_AICPU_TASK_FLIP]) as kfc_info_model:
            aicpu_task_flip = kfc_info_model.get_aicpu_task_flip()
        with KfcInfoViewModel(self._project_path,
                              [DBNameConstant.TABLE_KFC_INFO]) as kfc_info_model:
            kfc_info = kfc_info_model.get_kfc_info_data()
            kfc_info = FlipCalculator.compute_batch_id(kfc_info, aicpu_task_flip, is_flip_num=True)
        kfc_comm_task_data = self.process_kfc_info_data(kfc_comm_task_data, kfc_info)
        kfc_comm_task_data.sort(key=lambda x: x.start_time + x.duration)
        kfc_op_data_stream_id_table = {}
        for data in kfc_op_data:
            # kfc大算子流
            if data.host_task_type in self.BLACK_KFC_OP_TYPE or data.op_name in self.BLACK_KFC_OP_NAME:
                continue
            group_name = kfc_stream_id_group_table.get(data.stream_id, "N/A")
            kfc_op_data_stream_id_table.setdefault(data.stream_id, []).append(
                self.KFC_OP_DATA(data, group_name, "N/A", 0, 1, "N/A",
                                 -1, -1, "N/A", "N/A", -1, -1, DeviceHcclSource.INVALID.value)
            )
        master_stream_task = self.get_master_stream_task_in_hccl_op(kfc_comm_task_data, aicpu_task_flip)
        for data in kfc_comm_task_data:
            # kfc小算子流
            group_name_set = comm_stream_id_group_table.get(data.stream_id, set())
            for group_name in group_name_set:
                self._kfc_small_task.setdefault(group_name, []).append(data)
        return kfc_op_data_stream_id_table, master_stream_task

    def get_host_task_info(self: any, kfc_op_data) -> dict:
        with GeInfoViewModel(self._project_path, [DBNameConstant.TABLE_GE_TASK]) as model:
            ge_data = model.get_ge_info_by_device_id(DBNameConstant.TABLE_GE_TASK, InfoConfReader().get_device_id())
        node_info = {}
        for data in ge_data:
            if data.stream_id not in kfc_op_data:
                continue
            node_key = "{0}-{1}-{2}-{3}".format(data.stream_id, data.task_id, data.context_id, data.batch_id)
            node_info[node_key] = [data.op_name, data.op_type, data.timestamp]
        return node_info

    def get_kfc_host_hccl_op(self: any) -> dict:
        iter_range = self.sample_config.get(StrConstant.PARAM_ITER_ID)
        with HcclViewModel(self._project_path, DBNameConstant.DB_HCCL, [DBNameConstant.TABLE_HCCL_OP]) as model:
            if not model.check_table():
                return dict()
            hccl_ops = model.get_hccl_ops(iter_range.model_id, iter_range.iteration_id)
        kfc_hccl_op_map = {}
        for data in hccl_ops:
            kfc_hccl_op_map[data.kfc_connection_id] = data
        return kfc_hccl_op_map

    def get_hccl_op_info_data(self: any) -> dict:
        with KfcInfoViewModel(self._project_path,
                              [DBNameConstant.TABLE_DEVICE_HCCL_OP_INFO]) as kfc_info_model:
            hccl_op_info = kfc_info_model.get_hccl_op_info_data()
        hccl_op_info_dict = {}
        for hccl_op in hccl_op_info:
            hccl_op_info_dict.setdefault(hccl_op.stream_id, []).append(hccl_op)
            self._source[hccl_op.stream_id] = hccl_op.source
        return hccl_op_info_dict

    def get_master_stream_task_in_hccl_op(self: any, comm_task_data: list, aicpu_task_flip: list) -> dict:
        '''
        1. hccl aicpu场景下，通过唯一Id关联每个通信大算子的主流首尾task
        2. 将主流放入_master_stream中
        3. 每个通信大算子的主流首task不作为master，将其唯一Id放入_master_stream_first_task中
        '''
        with KfcInfoViewModel(self._project_path,
                              [DBNameConstant.TABLE_AICPU_MASTER_STREAM_HCCL_TASK]) as kfc_info_model:
            master_stream_hccl_task = kfc_info_model.get_aicpu_master_stream_hccl_task()
        if not master_stream_hccl_task:
            return {}
        master_stream_hccl_task = FlipCalculator.compute_batch_id(master_stream_hccl_task,
                                                                  aicpu_task_flip, is_flip_num=True)
        hccl_small_task = {}
        for data in comm_task_data:
            unique_id = "{0}-{1}-{2}-{3}".format(data.stream_id, data.task_id, data.context_id, data.batch_id)
            hccl_small_task[unique_id] = data
        master_stream_task = {}
        for data in master_stream_hccl_task:
            if data.task_type != self.FIRST_TASK_TYPE and data.task_type != self.LAST_TASK_TYPE:
                logging.warning('The main stream task type is invalid, type is %d', data.task_type)
                continue
            unique_id = "{0}-{1}-{2}-{3}".format(data.stream_id, data.task_id,
                                                 NumberConstant.DEFAULT_GE_CONTEXT_ID, data.batch_id)
            if data.task_type == self.FIRST_TASK_TYPE:
                # 第一个通信子任务（notify wait）可能会从通信算子实际执行时间之前开始等待,
                # 为了避免通信时间计算不准,所以第一个通信子任务不作为主流算子,is_master字段设置为0
                self._master_stream_first_task.add(unique_id)
            small_task = hccl_small_task.get(unique_id)
            if not small_task:
                continue
            master_stream_task.setdefault(data.aicpu_stream_id, {})
            master_stream_task[data.aicpu_stream_id][data.timestamp] = [data.task_type, small_task]
        return master_stream_task

    def calculate_kfc_op(self: any) -> None:
        kfc_op_data_stream_id_table, master_stream_task = self.get_kfc_op_data()
        node_info = self.get_host_task_info(kfc_op_data_stream_id_table)
        kfc_hccl_op_map = self.get_kfc_host_hccl_op()
        hccl_op_info_data = self.get_hccl_op_info_data()
        # 考虑到不同流可能出现并行，所以在每条流上卡时间关联DeviceHcclOpInfo
        for stream_id, kfc_op_data in kfc_op_data_stream_id_table.items():
            idx = 0
            time_idx = 0
            hccl_op_info = hccl_op_info_data.get(stream_id, [])
            timestamp_master_stream_task_table = master_stream_task.get(stream_id, {})
            kfc_op_with_task = set()
            kfc_op_with_task_index = {}
            for i, op_data in enumerate(kfc_op_data):
                curr_hccl_op_info, idx = self.get_hccl_op_info(op_data, idx, hccl_op_info)
                op_data, time_idx = self.update_op_data_time(op_data, time_idx, timestamp_master_stream_task_table)
                data = op_data.ascend_data
                iter_id = 1
                node_key = "{0}-{1}-{2}-{3}".format(data.stream_id, data.task_id, data.context_id, data.batch_id)
                if node_key in kfc_op_with_task:
                    iter_id = kfc_op_with_task_index.get(node_key, 1) + 1
                    kfc_op_with_task_index[node_key] = iter_id
                kfc_op_with_task.add(node_key)
                op_name, op_type, first_timestamp = node_info.get(node_key, [None, None, None])
                if not op_name:
                    logging.error("The kfc op name is None with task type %s", data.host_task_type)
                    continue
                source = self._source.get(stream_id, DeviceHcclSource.INVALID.value)
                host_hccl_op = kfc_hccl_op_map.get(data.connection_id, None)
                if source == DeviceHcclSource.HCCL.value and host_hccl_op:
                    op_name = host_hccl_op.op_name
                if curr_hccl_op_info:
                    kfc_op_data[i] = op_data.replace(op_name=op_name, first_timestamp=first_timestamp,
                                                     iter_id=iter_id, op_type=op_type,
                                                     relay=curr_hccl_op_info.relay, retry=curr_hccl_op_info.retry,
                                                     data_type=curr_hccl_op_info.data_type,
                                                     alg_type=curr_hccl_op_info.alg_type, count=curr_hccl_op_info.count,
                                                     rank_size=curr_hccl_op_info.rank_size,
                                                     source=source, group_name=curr_hccl_op_info.group_name)
                else:
                    kfc_op_data[i] = op_data.replace(op_name=op_name, group_name=op_data.group_name, source=source,
                                                     first_timestamp=first_timestamp, iter_id=iter_id, op_type=op_type)
            self.update_op_name_by_group(kfc_op_data)
            for data in kfc_op_data:
                self._kfc_op_data.setdefault(data.group_name, []).append(
                    [data.ascend_data.model_id, data.ascend_data.index_id, data.op_name,
                     data.ascend_data.timestamp, data.ascend_data.duration, data.group_name,
                     data.ascend_data.connection_id, data.op_type, data.relay, data.retry,
                     data.data_type, data.alg_type, data.count, data.rank_size, data.source])

    def process_kfc_info_data(self: any, kfc_task: list, kfc_info_data: list) -> list:
        '''
        通过唯一Id关联hccl info和通信小task
        '''
        if not kfc_info_data:
            # L0场景
            kfc_info_data = [None] * len(kfc_task)
            for idx, data in enumerate(kfc_task):
                kfc_info_data[idx] = self.make_default_kfc_info()
                kfc_info_data[idx] = kfc_info_data[idx].replace(start_time=data.timestamp, duration=data.duration,
                                                                stream_id=data.stream_id, task_id=data.task_id,
                                                                context_id=data.context_id, batch_id=data.batch_id,
                                                                device_task_type=data.device_task_type,
                                                                ts_virtual_batch_id=data.ts_virtual_batch_id)
            return kfc_info_data
        task_time = {}
        for data in kfc_task:
            node_key = "{0}-{1}-{2}-{3}".format(data.stream_id, data.task_id, data.context_id, data.batch_id)
            task_time[node_key] = data
        for idx, data in enumerate(kfc_info_data):
            unique_id = "{0}-{1}-{2}-{3}".format(data.stream_id, data.task_id, data.context_id, data.batch_id)
            task_data = task_time.get(unique_id)
            if not task_data:
                continue
            kfc_info_data[idx] = data.replace(start_time=task_data.timestamp, duration=task_data.duration,
                                              device_task_type=task_data.device_task_type,
                                              ts_virtual_batch_id=task_data.ts_virtual_batch_id)
        HcclCalculator.update_bandwidth(kfc_info_data)
        return kfc_info_data

    def get_plane_id(self: any, data: KfcInfoViewModel.KFC_HCCL_INFO_TYPE,
                     group_name: str, stream_id: int, source: int, is_master: bool):
        if data.plane_id == -1:
            return -1
        unique_id = "{0}-{1}-{2}-{3}".format(data.stream_id, data.task_id, data.context_id, data.batch_id)
        if is_master or unique_id in self._master_stream_first_task:
            plane_id = 1 if source == DeviceHcclSource.HCCL.value else 0
            return plane_id
        if stream_id not in self._plane_id.get(group_name, {}):
            self._plane_id.setdefault(group_name, {})
            self._plane_id[group_name][stream_id] = len(self._plane_id[group_name]) + 1
        plane_id = self._plane_id.get(group_name, {}).get(stream_id, 1)
        if source == DeviceHcclSource.HCCL.value:
            plane_id += 1  # hccl aicpu场景, 由于plane 0是控制流的task, aicpu展开的通信task是从1开始,所以需要+1
        return plane_id

    def get_task_is_master(self: any, data: KfcInfoViewModel.KFC_HCCL_INFO_TYPE, group_name: str, source: int):
        unique_id = "{0}-{1}-{2}-{3}".format(data.stream_id, data.task_id, data.context_id, data.batch_id)
        if data.device_task_type == self.MC2_MASTER_STREAM_TASK_TYPE or unique_id in self._master_stream_first_task:
            self._master_stream[group_name] = data.stream_id
            return source != DeviceHcclSource.HCCL.value
        return self._master_stream.get(group_name) == data.stream_id

    def calculate_kfc_task(self: any) -> None:
        for group_name, hccl_small_task in self._kfc_small_task.items():
            idx = 0
            match_num = 0
            kfc_op_data = self._kfc_op_data.get(group_name, [])
            self._kfc_task_data[group_name] = [None] * len(hccl_small_task)
            for kfc_op in kfc_op_data:
                # kfc_op[3]: kfc op start time
                while idx < len(hccl_small_task) and \
                        hccl_small_task[idx].start_time + hccl_small_task[idx].duration < kfc_op[3]:
                    idx += 1
                # kfc_op[3]: kfc op start time; kfc_op[4]: kfc op duration
                while idx < len(hccl_small_task) and \
                        hccl_small_task[idx].start_time + hccl_small_task[idx].duration <= kfc_op[3] + kfc_op[4]:
                    data = hccl_small_task[idx]
                    # kfc_op[14]是source
                    source = kfc_op[14]
                    is_master = int(self.get_task_is_master(data, group_name, source))
                    plane_id = self.get_plane_id(data, group_name, data.stream_id, source, is_master)
                    self._kfc_task_data[group_name][match_num] = [
                        *kfc_op[:4], 0, data.op_name, group_name, plane_id, data.start_time, data.duration, is_master,
                        data.stream_id, data.task_id, data.duration_estimated, data.local_rank, data.remote_rank,
                        data.transport_type, data.size, data.data_type, data.link_type, data.bandwidth, data.context_id,
                        # kfc_op[6]是connection_id
                        data.notify_id, data.ts_virtual_batch_id, data.rdma_type, kfc_op[6], source
                    ]
                    match_num += 1
                    idx += 1
            if idx > match_num:
                logging.warning("The kfc info is mismatched with kfc op, mismatch num is %d", idx - match_num)
            self._kfc_task_data[group_name] = self._kfc_task_data[group_name][:match_num]

    def ms_run(self: any) -> None:
        """
        entry
        :return: None`
        """
        if not os.path.exists(PathManager.get_db_path(self._project_path, DBNameConstant.DB_MC2_COMM_INFO)) \
                or not self._judge_calculate_again():
            return
        self._drop_table()
        self.calculate()
        self.save()

    def _drop_table(self):
        with HCCLModel(self._project_path, [DBNameConstant.TABLE_KFC_OP, DBNameConstant.TABLE_KFC_TASK]) as model:
            model.drop_table(DBNameConstant.TABLE_KFC_OP)
            model.drop_table(DBNameConstant.TABLE_KFC_TASK)

    def _judge_calculate_again(self):
        if not ProfilingScene().is_all_export():
            logging.info("In graph scene, to generate table %s", DBNameConstant.TABLE_KFC_OP)
            return True
        else:
            hccl_db_path = PathManager.get_db_path(self._project_path, DBNameConstant.DB_HCCL_SINGLE_DEVICE)
            if DBManager.check_tables_in_db(hccl_db_path, DBNameConstant.TABLE_KFC_OP):
                logging.info("Found table %s in operator scene, no need to generate again",
                             DBNameConstant.TABLE_KFC_OP)
                return False
            logging.info("No table %s found, to generate it", DBNameConstant.TABLE_KFC_OP)
            return True
