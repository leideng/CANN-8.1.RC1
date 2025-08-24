/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef METADEF_INC_GRAPH_PARALLELISM_GRAPH_PARALLEL_OPTION_H_
#define METADEF_INC_GRAPH_PARALLELISM_GRAPH_PARALLEL_OPTION_H_

#include <cstdint>
#include <string>

namespace ge {
struct PipelineParallelOption {
  bool is_enabled = false;
  bool is_auto = false;
  std::string pipeline_strategy;
  int32_t pipe_stage_num = -1;
  int32_t schedule_opt_virtual_stage_num = -1;
};

struct TensorParallelOption {
  bool is_enabled = false;
  bool is_auto = false;
  int32_t tensor_parallel_size = -1;
  int32_t inter_batch_flow_num = 1;
};

struct DataParallelOption {
  bool is_enabled = false;
  bool is_auto = false;
  // to be deleted below
  bool optimizer_state_sharding = false;
  bool gradient_sharding = false;
  bool model_weight_sharding = false;
  bool model_weight_prefetch = true;
  int32_t data_parallel_size = -1;
  // model weight prefetch buffer size(MB)
  uint32_t model_weight_prefetch_buffer_size = 0U;
};

struct TensorShardingOption {
  bool is_enabled = false;
  bool optimizer_state_sharding = false;
  bool gradient_sharding = false;
  bool model_weight_sharding = false;
  bool model_weight_prefetch = true;
  // model weight prefetch buffer size(MB)
  uint32_t model_weight_prefetch_buffer_size = 0U;
};

struct OptimizerOffloadGraphOption {
  bool is_enabled = false;
  std::string offload; // cpu or NVME, NVME is reserved
  std::string offload_path; // NVME path, reserved
};

struct EngineParallelOption {
  bool is_enabled = false;
  bool is_auto = false;
  std::string config_path;  // used if is_auto == true
};

struct GraphParallelOption {
  bool auto_deploy = false;
  std::string mode;      // AOE mode, search_strategy/search_and_shard_graph/load_strategy/load_and_eval_strategy
  std::string work_dir;  // AOE dump/load path for strategies
  std::string opt_level;
  int32_t global_batch_size = -1;
  DataParallelOption data_parallel_option;
  TensorParallelOption tensor_parallel_option;
  TensorShardingOption tensor_sharding_option;
  PipelineParallelOption pipeline_parallel_option;
  OptimizerOffloadGraphOption optimizer_offload_option;
  EngineParallelOption engine_parallel_option;
};
}  // namespace ge

#endif  // METADEF_INC_GRAPH_PARALLELISM_GRAPH_PARALLEL_OPTION_H_
