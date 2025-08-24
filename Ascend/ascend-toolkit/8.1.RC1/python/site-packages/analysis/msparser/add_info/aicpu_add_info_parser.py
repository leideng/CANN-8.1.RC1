#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import logging

from common_func.db_name_constant import DBNameConstant
from common_func.ms_multi_process import MsMultiProcess
from common_func.ms_constant.str_constant import StrConstant
from common_func.info_conf_reader import InfoConfReader
from common_func.hash_dict_constant import HashDictData
from common_func.hccl_info_common import trans_enum_name
from common_func.hccl_info_common import RoleType
from common_func.hccl_info_common import OpType
from common_func.hccl_info_common import DataType
from common_func.hccl_info_common import LinkType
from common_func.hccl_info_common import TransPortType
from common_func.hccl_info_common import RdmaType
from common_func.hccl_info_common import DeviceHcclSource
from msmodel.ai_cpu.ai_cpu_model import AiCpuModel
from msmodel.step_trace.ts_track_model import TsTrackModel
from msmodel.ai_cpu.data_preparation_model import DataPreparationModel
from msmodel.add_info.kfc_info_model import KfcInfoModel
from msparser.add_info.aicpu_add_info_bean import AicpuAddInfoBean
from msparser.data_struct_size_constant import StructFmt
from msparser.interface.data_parser import DataParser
from profiling_bean.prof_enum.data_tag import DataTag


class AicpuAddInfoParser(DataParser, MsMultiProcess):
    """
    aicpu data parser
    """
    NONE_NODE_NAME = ''
    INVALID_CONTEXT_ID = 4294967295

    def __init__(self: any, file_list: dict, sample_config: dict) -> None:
        super().__init__(sample_config)
        super(DataParser, self).__init__(sample_config)
        self._file_list = file_list
        self.project_path = sample_config.get(StrConstant.SAMPLE_CONFIG_PROJECT_PATH)
        self.hash_data = {}
        self._aicpu_data = {
            AicpuAddInfoBean.AICPU_NODE: [],
            AicpuAddInfoBean.AICPU_DP: [],
            AicpuAddInfoBean.AICPU_MODEL: [],
            AicpuAddInfoBean.AICPU_MI: [],
            AicpuAddInfoBean.KFC_COMM_TURN: [],
            AicpuAddInfoBean.KFC_COMPUTE_TURN: [],
            AicpuAddInfoBean.KFC_HCCL_INFO: [],
            AicpuAddInfoBean.HCCL_OP_INFO: [],
            AicpuAddInfoBean.AICPU_FLIP_TASK: [],
            AicpuAddInfoBean.AICPU_MASTER_STREAM_HCCL_TASK: [],
        }
        self._get_data_func = {
            AicpuAddInfoBean.AICPU_NODE: self.get_aicpu_node_data,
            AicpuAddInfoBean.AICPU_DP: self.get_aicpu_dp_data,
            AicpuAddInfoBean.AICPU_MODEL: self.get_aicpu_model_data,
            AicpuAddInfoBean.AICPU_MI: self.get_aicpu_mi_data,
            AicpuAddInfoBean.KFC_COMM_TURN: self.get_kfc_comm_turn_data,
            AicpuAddInfoBean.KFC_COMPUTE_TURN: self.get_kfc_compute_turn_data,
            AicpuAddInfoBean.KFC_HCCL_INFO: self.get_kfc_hccl_info_data,
            AicpuAddInfoBean.HCCL_OP_INFO: self.get_hccl_op_info_data,
            AicpuAddInfoBean.AICPU_FLIP_TASK: self.get_aicpu_flip_task_data,
            AicpuAddInfoBean.AICPU_MASTER_STREAM_HCCL_TASK: self.get_aicpu_master_stream_hccl_tank_data,
        }
        self._aicpu_table = {
            AicpuAddInfoBean.KFC_COMM_TURN: DBNameConstant.TABLE_KFC_COMM_TURN,
            AicpuAddInfoBean.KFC_COMPUTE_TURN: DBNameConstant.TABLE_KFC_COMPUTE_TURN,
            AicpuAddInfoBean.KFC_HCCL_INFO: DBNameConstant.TABLE_KFC_INFO,
            AicpuAddInfoBean.HCCL_OP_INFO: DBNameConstant.TABLE_DEVICE_HCCL_OP_INFO,
            AicpuAddInfoBean.AICPU_FLIP_TASK: DBNameConstant.TABLE_AICPU_TASK_FLIP,
            AicpuAddInfoBean.AICPU_MASTER_STREAM_HCCL_TASK: DBNameConstant.TABLE_AICPU_MASTER_STREAM_HCCL_TASK,
        }

    @staticmethod
    def get_aicpu_node_data(aicpu_info: AicpuAddInfoBean) -> list:
        return [
            aicpu_info.data.stream_id,
            aicpu_info.data.task_id,
            aicpu_info.data.ai_cpu_task_start_time,
            aicpu_info.data.ai_cpu_task_end_time,
            AicpuAddInfoParser.NONE_NODE_NAME,
            aicpu_info.data.compute_time,
            aicpu_info.data.memory_copy_time,
            aicpu_info.data.ai_cpu_task_time,
            aicpu_info.data.dispatch_time,
            aicpu_info.data.total_time
        ]

    @staticmethod
    def get_aicpu_dp_data(aicpu_info: AicpuAddInfoBean) -> list:
        return [
            InfoConfReader().trans_into_local_time(float(aicpu_info.timestamp)),
            aicpu_info.data.action,
            aicpu_info.data.source,
            aicpu_info.data.buffer_size,
        ]

    @staticmethod
    def get_aicpu_model_data(aicpu_info: AicpuAddInfoBean) -> list:
        return [
            aicpu_info.data.index_id,
            aicpu_info.data.model_id,
            aicpu_info.timestamp,
            aicpu_info.data.tag_id,
            aicpu_info.data.event_id,
        ]

    @staticmethod
    def get_aicpu_mi_data(aicpu_info: AicpuAddInfoBean) -> list:
        return [
            aicpu_info.data.node_name,
            aicpu_info.data.queue_size,
            aicpu_info.data.start_time,
            aicpu_info.data.end_time,
            aicpu_info.data.duration,
        ]

    @staticmethod
    def get_kfc_comm_turn_data(aicpu_info: AicpuAddInfoBean) -> list:
        return [
            aicpu_info.data.device_id,
            aicpu_info.data.stream_id,
            aicpu_info.data.task_id,
            aicpu_info.data.comm_turn,
            aicpu_info.data.current_turn,
            aicpu_info.data.server_start_time,
            aicpu_info.data.wait_msg_start_time,
            aicpu_info.data.kfc_alg_exe_start_time,
            aicpu_info.data.send_task_start_time,
            aicpu_info.data.send_sqe_finish_time,
            aicpu_info.data.rtsq_exe_end_time,
            aicpu_info.data.server_end_time,
        ]

    @staticmethod
    def get_kfc_compute_turn_data(aicpu_info: AicpuAddInfoBean) -> list:
        return [
            aicpu_info.data.device_id,
            aicpu_info.data.stream_id,
            aicpu_info.data.task_id,
            aicpu_info.data.compute_turn,
            aicpu_info.data.current_turn,
            aicpu_info.data.wait_compute_start_time,
            aicpu_info.data.compute_start_time,
            aicpu_info.data.compute_exe_end_time,
        ]

    @staticmethod
    def get_aicpu_flip_task_data(aicpu_info: AicpuAddInfoBean) -> list:
        return [
            aicpu_info.data.stream_id,
            InfoConfReader().time_from_syscnt(aicpu_info.timestamp),
            aicpu_info.data.task_id,
            aicpu_info.data.flip_num,
        ]

    @staticmethod
    def get_aicpu_master_stream_hccl_tank_data(aicpu_info: AicpuAddInfoBean) -> list:
        return [
            InfoConfReader().time_from_syscnt(aicpu_info.timestamp),
            aicpu_info.data.aicpu_stream_id,
            aicpu_info.data.aicpu_task_id,
            aicpu_info.data.stream_id,
            aicpu_info.data.task_id,
            aicpu_info.data.type,
        ]

    def get_hccl_op_info_data(self: any, aicpu_info: AicpuAddInfoBean) -> list:
        source = DeviceHcclSource.INVALID
        if aicpu_info.struct_type == str(AicpuAddInfoBean.HCCL_OP_INFO):
            source = DeviceHcclSource.HCCL
        return [
            InfoConfReader().time_from_syscnt(aicpu_info.timestamp),
            aicpu_info.data.relay,
            aicpu_info.data.retry,
            trans_enum_name(DataType, aicpu_info.data.data_type),
            self.hash_data.get(aicpu_info.data.alg_type, aicpu_info.data.alg_type),
            aicpu_info.data.count,
            aicpu_info.data.group_name,
            aicpu_info.data.stream_id,
            aicpu_info.data.task_id,
            aicpu_info.data.rank_size,
            source.value,
        ]

    def get_kfc_hccl_info_data(self: any, aicpu_info: AicpuAddInfoBean) -> list:
        result = []
        for kfc_hccl_info in [aicpu_info.data.first_hccl_info, aicpu_info.data.second_hccl_info]:
            if kfc_hccl_info.group_name == "0":
                continue
            role = trans_enum_name(RoleType, kfc_hccl_info.role)
            op_type = trans_enum_name(OpType, kfc_hccl_info.op_type)
            data_type = trans_enum_name(DataType, kfc_hccl_info.data_type)
            link_type = trans_enum_name(LinkType, kfc_hccl_info.link_type)
            transport_type = trans_enum_name(TransPortType, kfc_hccl_info.transport_type)
            rdma_type = trans_enum_name(RdmaType, kfc_hccl_info.rdma_type)
            result.append([
                InfoConfReader().time_from_syscnt(kfc_hccl_info.timestamp),
                self.hash_data.get(kfc_hccl_info.item_id, kfc_hccl_info.item_id),
                kfc_hccl_info.ccl_tag,
                kfc_hccl_info.group_name,
                kfc_hccl_info.local_rank,
                kfc_hccl_info.remote_rank,
                kfc_hccl_info.rank_size,
                kfc_hccl_info.work_flow_mode,
                kfc_hccl_info.plane_id,
                self.INVALID_CONTEXT_ID,
                kfc_hccl_info.notify_id,
                kfc_hccl_info.stage,
                role,
                kfc_hccl_info.duration_estimated,
                kfc_hccl_info.src_addr,
                kfc_hccl_info.dst_addr,
                kfc_hccl_info.data_size,
                op_type,
                data_type,
                link_type,
                transport_type,
                rdma_type,
                kfc_hccl_info.stream_id,
                kfc_hccl_info.task_id,
            ])
        return result

    def parse(self: any) -> None:
        """
        parse ai cpu
        """
        aicpu_files = self._file_list.get(DataTag.AICPU_ADD_INFO, [])
        aicpu_info = self.parse_bean_data(
            aicpu_files,
            StructFmt.AI_CPU_ADD_FMT_SIZE,
            AicpuAddInfoBean,
            check_func=self.check_magic_num
        )
        self.hash_data = HashDictData(self._project_path).get_ge_hash_dict()
        self.set_aicpu_data(aicpu_info)

    def save(self: any) -> None:
        """
        save data to db
        :return:
        """
        aicpu_node_data = self._aicpu_data.get(AicpuAddInfoBean.AICPU_NODE, [])
        aicpu_dp_data = self._aicpu_data.get(AicpuAddInfoBean.AICPU_DP, [])
        aicpu_model_data = self._aicpu_data.get(AicpuAddInfoBean.AICPU_MODEL, [])
        aicpu_mi_data = self._aicpu_data.get(AicpuAddInfoBean.AICPU_MI, [])
        if aicpu_node_data:
            with AiCpuModel(self.project_path, [DBNameConstant.TABLE_AI_CPU]) as model:
                model.flush(aicpu_node_data, DBNameConstant.TABLE_AI_CPU)
        if aicpu_dp_data:
            with AiCpuModel(self.project_path, [DBNameConstant.TABLE_AI_CPU_DP]) as model:
                model.flush(aicpu_dp_data, DBNameConstant.TABLE_AI_CPU_DP)
        if aicpu_model_data:
            with TsTrackModel(self.project_path,
                              DBNameConstant.DB_STEP_TRACE, [DBNameConstant.TABLE_MODEL_WITH_Q]) as model:
                model.create_table(DBNameConstant.TABLE_MODEL_WITH_Q)
                model.flush(DBNameConstant.TABLE_MODEL_WITH_Q, aicpu_model_data)
        if aicpu_mi_data:
            with DataPreparationModel(self.project_path, [DBNameConstant.TABLE_DATA_QUEUE]) as model:
                model.flush(aicpu_mi_data)
        for aicpu_type, aicpu_table in self._aicpu_table.items():
            aicpu_data = self._aicpu_data.get(aicpu_type, [])
            if not aicpu_data:
                continue
            with KfcInfoModel(self.project_path, [aicpu_table]) as model:
                model.flush(aicpu_data, aicpu_table)

    def ms_run(self: any) -> None:
        """
        parse and save ge fusion data
        :return:
        """
        if not self._file_list.get(DataTag.AICPU_ADD_INFO, []):
            return
        logging.info("start parsing aicpu data, files: %s", str(self._file_list.get(DataTag.AICPU_ADD_INFO)))
        self.parse()
        self.save()

    def set_aicpu_data(self: any, aicpu_data: list) -> None:
        for aicpu_info in aicpu_data:
            struct_type = int(aicpu_info.struct_type)
            get_data_func = self._get_data_func.get(struct_type)
            if not get_data_func:
                logging.error("The aicpu type %d is invalid.", aicpu_info.struct_type)
                continue
            if struct_type == AicpuAddInfoBean.AICPU_NODE and \
                    (aicpu_info.data.ai_cpu_task_start_time == 0 or aicpu_info.data.ai_cpu_task_end_time == 0):
                continue
            if struct_type == AicpuAddInfoBean.KFC_HCCL_INFO:
                self._aicpu_data.get(struct_type).extend(get_data_func(aicpu_info))
            else:
                self._aicpu_data.get(struct_type).append(get_data_func(aicpu_info))
