#!/usr/bin/python3
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

from msconfig.meta_config import MetaConfig


class DataParsersConfig(MetaConfig):
    DATA = {
        'GeLogicStreamParser': [
            ('path', 'msparser.ge.ge_logic_stream_parser'),
            ('chip_model', '0,1,2,3,4,5,7,8,11'),
            ('position', 'H')
        ],
        'ParsingRuntimeData': [
            ('path', 'common_func.create_runtime_db'),
            ('chip_model', '0,1,2,3,4,5,7,8,11'),
            ('position', 'D')
        ],
        'L2CacheParser': [
            ('path', 'msparser.l2_cache.l2_cache_parser'),
            ('chip_model', '1,2,3,4,5,7,8,11'),
            ('position', 'D')
        ],
        'TsTimelineRecParser': [
            ('path', 'msparser.iter_rec.ts_timeline_parser'),
            ('chip_model', '0'),
            ('level', '3'),
            ('position', 'D')
        ],
        'ParsingDDRData': [
            ('path', 'msparser.hardware.ddr_parser'),
            ('chip_model', '0,1,2,3,4,7,8,11'),
            ('position', 'D')
        ],
        'ParsingPeripheralData': [
            ('path', 'msparser.hardware.dvpp_parser'),
            ('chip_model', '0,1,5,7,8,11'),
            ('position', 'D')
        ],
        'ParsingNicData': [
            ('path', 'msparser.hardware.nic_parser'),
            ('chip_model', '0,1,5,7,8,11'),
            ('position', 'D')
        ],
        'ParsingRoceData': [
            ('path', 'msparser.hardware.roce_parser'),
            ('chip_model', '0,1,5'),
            ('position', 'D')
        ],
        'ParsingTSData': [
            ('path', 'msparser.hardware.tscpu_parser'),
            ('chip_model', '0,1,2,3,4,5,7,8,11'),
            ('position', 'D')
        ],
        'ParsingAICPUData': [
            ('path', 'msparser.hardware.ai_cpu_parser'),
            ('chip_model', '0,1,2,3,4,5,7,8,11'),
            ('position', 'D')
        ],
        'ParsingCtrlCPUData': [
            ('path', 'msparser.hardware.ctrl_cpu_parser'),
            ('chip_model', '0,1,2,3,4,5,7,8,11'),
            ('position', 'D')
        ],
        'ParsingMemoryData': [
            ('path', 'msparser.hardware.sys_mem_parser'),
            ('chip_model', '0,1,2,3,4,5,7,8,11')
        ],
        'ParsingCpuUsageData': [
            ('path', 'msparser.hardware.sys_usage_parser'),
            ('chip_model', '0,1,2,3,4,5,7,8,11')
        ],
        'ParsingPcieData': [
            ('path', 'msparser.hardware.pcie_parser'),
            ('chip_model', '0,1,2,3,4,5'),
            ('position', 'D')
        ],
        'ParsingHBMData': [
            ('path', 'msparser.hardware.hbm_parser'),
            ('chip_model', '0,1,2,3,4,5,7,8,11'),
            ('position', 'D')
        ],
        'ParsingQosData': [
            ('path', 'msparser.hardware.qos_parser'),
            ('chip_model', '5'),
            ('position', 'D')
        ],
        'NonMiniLLCParser': [
            ('path', 'msparser.hardware.llc_parser'),
            ('chip_model', '1,2,3,4,5,7,8,11'),
            ('position', 'D')
        ],
        'MiniLLCParser': [
            ('path', 'msparser.hardware.mini_llc_parser'),
            ('chip_model', '0'),
            ('position', 'D')
        ],
        'ParsingHCCSData': [
            ('path', 'msparser.hardware.hccs_parser'),
            ('chip_model', '0,1,2,3,4,5'),
            ('position', 'D')
        ],
        'TstrackParser': [
            ('path', 'msparser.step_trace.ts_track_parser'),
            ('chip_model', '1,2,3,4,5,7,8,11'),
            ('level', '2'),
            ('position', 'D'),
        ],
        'ParsingAICoreSampleData': [
            ('path', 'msparser.aic_sample.ai_core_sample_parser'),
            ('chip_model', '0,1,2,3,4'),
            ('position', 'D')
        ],
        'ParsingAIVectorCoreSampleData': [
            ('path', 'msparser.aic_sample.ai_core_sample_parser'),
            ('chip_model', '2,3,4'),
            ('position', 'D')
        ],
        'MsprofTxParser': [
            ('path', 'msparser.msproftx.msproftx_parser'),
            ('chip_model', '0,1,2,3,4,5,7,8,11'),
            ('level', '4'),
            ('position', 'H')
        ],
        'AicpuBinDataParser': [
            ('path', 'msparser.aicpu.aicpu_bin_data_parser'),
            ('chip_model', '0,1,2,3,4,5,7,8,11'),
            ('level', '3'),
            ('position', 'D')
        ],
        'ParsingFftsAICoreSampleData': [
            ('path', 'msparser.aic_sample.ai_core_sample_parser'),
            ('chip_model', '5,7,8,11'),
            ('position', 'D')
        ],
        'BiuPerfParser': [
            ('path', 'msparser.biu_perf.biu_perf_parser'),
            ('chip_model', '5,7,8,11'),
            ('position', 'D')
        ],
        'SocProfilerParser': [
            ('path', 'msparser.stars.soc_profiler_parser'),
            ('chip_model', '5,7,8,11'),
            ('position', 'D')
        ],
        'MsTimeParser': [
            ('path', 'msparser.ms_timer.ms_time_parser'),
            ('chip_model', '0,1,2,3,4,5')
        ],
        'DataPreparationParser': [
            ('path', 'msparser.aicpu.data_preparation_parser'),
            ('chip_model', '0,1,2,3,4,5')
        ],
        'HCCLOperatiorParser': [
            ('path', 'msparser.parallel.hccl_operator_parser'),
            ('chip_model', '1,2,3,4,5'),
            ('level', '3')
        ],
        'ParallelStrategyParser': [
            ('path', 'msparser.parallel.parallel_strategy_parser'),
            ('chip_model', '1,2,3,4,5'),
            ('position', 'D')
        ],
        'ParallelParser': [
            ('path', 'msparser.parallel.parallel_parser'),
            ('chip_model', '1,2,3,4,5'),
            ('level', '4'),
            ('position', 'D')
        ],
        'NpuMemParser': [
            ('path', 'msparser.npu_mem.npu_mem_parser'),
            ('chip_model', '0,1,3,4,5,7,8,11'),
            ('position', 'D')
        ],
        'NpuModuleMemParser': [
            ('path', 'msparser.npu_mem.npu_module_mem_parser'),
            ('chip_model', '1,2,3,4,5,7,8,11'),
            ('position', 'D')
        ],
        'NpuOpMemParser': [
            ('path', 'msparser.npu_mem.npu_op_mem_parser'),
            ('chip_model', '0,1,2,3,4,5,7,8,11'),
            ('position', 'H')
        ],
        'FreqParser': [
            ('path', 'msparser.freq.freq_parser'),
            ('chip_model', '5, 7'),
            ('position', 'D')
        ],
        'ApiEventParser': [
            ('path', 'msparser.api_event.api_event_parser'),
            ('chip_model', '0,1,2,3,4,5,7,8,11'),
            ('level', '2'),
            ('position', 'H')
        ],
        'HashDicParser': [
            ('path', 'msparser.hash_dic.hash_dic_parser'),
            ('chip_model', '0,1,2,3,4,5,7,8,11'),
            ('position', 'H')
        ],
        'TaskTrackParser': [
            ('path', 'msparser.compact_info.task_track_parser'),
            ('chip_model', '0,1,2,3,4,5,7,8,11'),
            ('level', '2'),
            ('position', 'H')
        ],
        'MemcpyInfoParser': [
            ('path', 'msparser.compact_info.memcpy_info_parser'),
            ('chip_model', '0,1,2,3,4,5,7,8,11'),
            ('level', '2'),
            ('position', 'H')
        ],
        'HcclInfoParser': [
            ('path', 'msparser.add_info.hccl_info_parser'),
            ('chip_model', '0,1,2,3,4,5,7,8,11'),
            ('level', '2'),
            ('position', 'H')
        ],
        'MultiThreadParser': [
            ('path', 'msparser.add_info.multi_thread_parser'),
            ('chip_model', '0,1,2,3,4,5,7,8,11'),
            ('level', '2'),
            ('position', 'H')
        ],
        'AicpuAddInfoParser': [
            ('path', 'msparser.add_info.aicpu_add_info_parser'),
            ('chip_model', '0,1,2,3,4,5,7,8,11'),
            ('position', 'D'),
        ],
        'TensorAddInfoParser': [
            ('path', 'msparser.add_info.tensor_add_info_parser'),
            ('chip_model', '0,1,2,3,4,5,7,8,11'),
            ('level', '2'),
            ('position', 'H')
        ],
        'FusionAddInfoParser': [
            ('path', 'msparser.add_info.fusion_add_info_parser'),
            ('chip_model', '0,1,2,3,4,5,7,8,11'),
            ('level', '2'),
            ('position', 'H')
        ],
        'GraphAddInfoParser': [
            ('path', 'msparser.add_info.graph_add_info_parser'),
            ('chip_model', '0,1,2,3,4,5,7,8,11'),
            ('level', '2'),
            ('position', 'H')
        ],
        'NodeBasicInfoParser': [
            ('path', 'msparser.compact_info.node_basic_info_parser'),
            ('chip_model', '0,1,2,3,4,5,7,8,11'),
            ('level', '2'),
            ('position', 'H')
        ],
        'NodeAttrInfoParser': [
            ('path', 'msparser.compact_info.node_attr_info_parser'),
            ('chip_model', '0,1,2,3,4,5,7,8,11'),
            ('level', '2'),
            ('position', 'H')
        ],
        'MemoryApplicationParser': [
            ('path', 'msparser.add_info.memory_application_parser'),
            ('chip_model', '0,1,2,3,4,5,7,8,11'),
            ('level', '2'),
            ('position', 'H')
        ],
        'StaticOpMemParser': [
            ('path', 'msparser.add_info.static_op_mem_parser'),
            ('chip_model', '0,1,2,3,4,5,7,8,11'),
            ('level', '4'),
            ('position', 'H')
        ],
        'CtxIdParser': [
            ('path', 'msparser.add_info.ctx_id_parser'),
            ('chip_model', '5,7,8,11'),
            ('level', '2'),
            ('position', 'H')
        ],
        'CANNCalculator': [
            ('path', 'mscalculate.cann.cann_calculator'),
            ('chip_model', '0,1,2,3,4,5,7,8,11'),
            ('level', '3'),
            ('position', 'H')
        ],
        'IterRecParser': [
            ('path', 'msparser.iter_rec.iter_rec_parser'),
            ('chip_model', '1,2,3,4'),
            ('level', '4'),
            ('position', 'D')
        ],
        'NoGeIterRecParser': [
            ('path', 'msparser.iter_rec.iter_rec_parser'),
            ('chip_model', '1,2,3,4'),
            ('level', '4'),
            ('position', 'D')
        ],
        'HcclOpInfoParser': [
            ('path', 'msparser.compact_info.hccl_op_info_parser'),
            ('chip_model', '0,1,2,3,4,5,7,8,11'),
            ('level', '2'),
            ('position', 'H')
        ],
        'Mc2CommInfoParser': [
            ('path', 'msparser.add_info.mc2_comm_info_parser'),
            ('chip_model', '4,5'),
            ('position', 'H')
        ],
    }
