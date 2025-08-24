/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "parallelism/tensor_parallel_attrs.h"
#include "common/ge_common/util.h"
#include "graph/debug/ge_util.h"
#include "nlohmann/json.hpp"
#include "parallelism/comm_task_builder.h"

#define USED_BY_JSON __attribute__((unused)) static

namespace ge {
namespace tp {
namespace {
using Json = nlohmann::json;

constexpr size_t kValidDimSliceItemNum = 2U;
constexpr size_t kIndexStepId = 0U;
constexpr size_t kIndexOutputIndex = 1U;

Status StringToJson(const std::string &json_str, Json &json) {
  std::stringstream ss;
  ss << json_str;
  try {
    ss >> json;
  } catch (const nlohmann::json::exception &e) {
    GELOGE(PARAM_INVALID, "Failed to init json object, err = %s, json_str = %s", e.what(), json_str.c_str());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

template<typename T>
Status ParseFromJson(const std::string &type, const std::string &json_str, T &value) {
  Json json;
  GE_CHK_STATUS_RET_NOLOG(StringToJson(json_str, json));
  try {
    value = json.get<T>();
  } catch (const nlohmann::json::exception &e) {
    GELOGE(PARAM_INVALID,
           "Failed to parse json object, type = %s, err = %s, json_str = %s",
           type.c_str(),
           e.what(),
           json_str.c_str());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

template<typename T>
std::shared_ptr<T> CreateReshardTaskInfo(const Json &j) {
  return ComGraphMakeShared<T>(j.get<T>());
}

template<typename T>
std::string ToJsonString(const T &obj) {
  try {
    const Json j = obj;
    return j.dump();
  } catch (const nlohmann::json::exception &e) {
    GELOGE(FAILED, "Failed to dump object, err = %s", e.what());
    return "";
  }
}

template<typename T>
void GetValue(const Json &j, const std::string &key, T &value) {
  value = j.at(key).template get<T>();
}
template<typename T>
void TryGetValue(const Json &j, const std::string &key, T &value) {
  if (j.contains(key)) {
    value = j.at(key).template get<T>();
  }
}
}  // namespace

void CommTaskBuilder::BuildCommTask(const Json &j, CommTask &comm_task) {
  auto task_type = j.at("task_type").get<std::string>();
  const decltype(builders_)::const_iterator it = builders_.find(task_type);
  if (it == builders_.cend()) {
    GELOGE(PARAM_INVALID, "unsupported op type %s", comm_task.task_type.c_str());
    return;
  }
  it->second(j, comm_task);  // exception caught by caller
  comm_task.task_type = std::move(task_type);
}

Status CommTaskBuilder::ConvertToJson(const CommTask &comm_task, nlohmann::json &j) {
  const decltype(json_converters_)::const_iterator it = json_converters_.find(comm_task.task_type);
  GE_CHK_BOOL_RET_STATUS(it != json_converters_.cend(),
                         PARAM_INVALID,
                         "unsupported op type %s",
                         comm_task.task_type.c_str());
  return it->second(comm_task, j);  // exception caught by caller
}

std::string DeviceIndex::DebugString() const {
  return engine_type + ToString(indices);
}

USED_BY_JSON void to_json(Json &j, const DimSlice &dim_slice) {
  j = std::vector<int64_t>{dim_slice.begin, dim_slice.end};
}

USED_BY_JSON void from_json(const Json &j, DimSlice &dim_slice) {
  const auto &range = j.get<std::vector<int64_t>>();
  if (range.size() == kValidDimSliceItemNum) {
    dim_slice.begin = range.front();
    dim_slice.end = range.back();
  } else {
    dim_slice.begin = -1;
    dim_slice.end = -1;
    GELOGE(PARAM_INVALID, "invalid DimSlice: %s", j.dump().c_str());
  }
}

USED_BY_JSON void to_json(Json &j, const DeviceIndex &device_index) {
  j = Json();
  j["engine_type"] = device_index.engine_type;
  j["index"] = device_index.indices;
}

USED_BY_JSON void from_json(const Json &j, DeviceIndex &device_index) {
  GetValue(j, "engine_type", device_index.engine_type);
  GetValue(j, "index", device_index.indices);
}

USED_BY_JSON void to_json(Json &j, const ModelIndex &model_index) {
  j = Json();
  j["device_index"] = model_index.device_index;
  j["virtual_stage_id"] = model_index.virtual_stage_id;
  j["stage_id"] = model_index.stage_id;
}

USED_BY_JSON void from_json(const Json &j, ModelIndex &model_index) {
  GetValue(j, "device_index", model_index.device_index);
  GetValue(j, "virtual_stage_id", model_index.virtual_stage_id);
  GetValue(j, "stage_id", model_index.stage_id);
}

USED_BY_JSON void to_json(Json &j, const PipelineConfig &pipeline_config) {
  j = Json();
  j["micro_batch"] = pipeline_config.micro_batch;
  j["stage_id"] = pipeline_config.stage_id;
  j["virtual_stage_id"] = pipeline_config.virtual_stage_id;
}

USED_BY_JSON void from_json(const Json &j, PipelineConfig &pipeline_config) {
  GetValue(j, "micro_batch", pipeline_config.micro_batch);
  GetValue(j, "stage_id", pipeline_config.stage_id);
  GetValue(j, "virtual_stage_id", pipeline_config.virtual_stage_id);
}

USED_BY_JSON void to_json(Json &j, const TensorSliceDeployment &tensor_slice_deployment) {
  j = Json();
  j["device_indices_each_slice"] = tensor_slice_deployment.device_indices_each_slice;
  j["axis_slices"] = tensor_slice_deployment.axis_slices;
}

USED_BY_JSON void from_json(const Json &j, TensorSliceDeployment &tensor_slice_deployment) {
  GetValue(j, "device_indices_each_slice", tensor_slice_deployment.device_indices_each_slice);
  GetValue(j, "axis_slices", tensor_slice_deployment.axis_slices);
}

USED_BY_JSON void to_json(Json &j, const TensorDeployment &tensor_deployment) {
  j = Json();
  j["shard_deployment"] = tensor_deployment.shard_deployment;
  if (!tensor_deployment.verbose.empty()) {
    j["verbose"] = tensor_deployment.verbose;
  }
}

USED_BY_JSON void from_json(const Json &j, TensorDeployment &tensor_deployment) {
  GetValue(j, "shard_deployment", tensor_deployment.shard_deployment);
  TryGetValue(j, "verbose", tensor_deployment.verbose);
}

USED_BY_JSON void to_json(Json &j, const TensorDeployments &tensor_deployments) {
  j = Json();
  j["deployments"] = tensor_deployments.deployments;
}

USED_BY_JSON void from_json(const Json &j, NodeDeployments &node_deployments) {
  GetValue(j, "deployments", node_deployments.deployments);
}

USED_BY_JSON void from_json(const Json &j, TensorDeployments &tensor_deployments) {
  GetValue(j, "deployments", tensor_deployments.deployments);
}

USED_BY_JSON void to_json(Json &j, const NodeDeployment &node_deployment) {
  j = Json();
  j["devices"] = node_deployment.devices;
  j["pipeline_config"] = node_deployment.pipeline_config;
}

USED_BY_JSON void from_json(const Json &j, NodeDeployment &node_deployment) {
  GetValue(j, "devices", node_deployment.devices);
  TryGetValue(j, "pipeline_config", node_deployment.pipeline_config);
}

USED_BY_JSON void to_json(Json &j, const NodeDeployments &node_deployments) {
  j = Json();
  j["deployments"] = node_deployments.deployments;
}


USED_BY_JSON void to_json(Json &j, const CommPair &comm_pair) {
  j = Json();
  j["src_device_index"] = comm_pair.src_device_index;
  j["dst_device_index"] = comm_pair.dst_device_index;
  j["src_virtual_stage_id"] = comm_pair.src_virtual_stage_id;
  j["dst_virtual_stage_id"] = comm_pair.dst_virtual_stage_id;
}

USED_BY_JSON void from_json(const Json &j, CommPair &comm_pair) {
  GetValue(j, "src_device_index", comm_pair.src_device_index);
  GetValue(j, "dst_device_index", comm_pair.dst_device_index);
  TryGetValue(j, "src_virtual_stage_id", comm_pair.src_virtual_stage_id);
  TryGetValue(j, "dst_virtual_stage_id", comm_pair.dst_virtual_stage_id);
}

USED_BY_JSON void to_json(Json &j, const FlowAttr &comm_group) {
  j = Json();
  j["depth"] = comm_group.depth;
  j["enqueue_policy"] = comm_group.enqueue_policy;
}

USED_BY_JSON void from_json(const Json &j, FlowAttr &comm_group) {
  GetValue(j, "depth", comm_group.depth);
  GetValue(j, "enqueue_policy", comm_group.enqueue_policy);
}

USED_BY_JSON void to_json(Json &j, const CommGroup &comm_group) {
  j = comm_group.device_indices;
}

USED_BY_JSON void from_json(const Json &j, CommGroup &comm_group) {
  comm_group.device_indices = j.get<std::vector<DeviceIndex>>();
}

USED_BY_JSON void to_json(Json &j, const SendRecvReshardTask &task_info) {
  j = Json();
  j["task_type"] = kCommTaskTypeSendReceive;
  j["comm_pairs"] = task_info.comm_pairs;
  j["parallel_group"] = task_info.parallel_group;
  j["comm_type"] = task_info.comm_type;
  j["flow_attr"] = task_info.flow_attr;
}

USED_BY_JSON void from_json(const Json &j, SendRecvReshardTask &task_info) {
  GetValue(j, "comm_pairs", task_info.comm_pairs);
  TryGetValue(j, "comm_type", task_info.comm_type);
  TryGetValue(j, "parallel_group", task_info.parallel_group);
  TryGetValue(j, "flow_attr", task_info.flow_attr);
}

USED_BY_JSON void to_json(Json &j, const AllGatherReshardTask &task_info) {
  j = Json();
  j["task_type"] = kCommTaskTypeHcomAllGather;
  j["axis"] = task_info.axis;
  j["comm_groups"] = task_info.comm_groups;
  j["parallel_group"] = task_info.parallel_group;
  j["output_allocator"] = task_info.output_allocator;
}

USED_BY_JSON void from_json(const Json &j, AllGatherReshardTask &all_gather_task_info) {
  GetValue(j, "comm_groups", all_gather_task_info.comm_groups);
  GetValue(j, "axis", all_gather_task_info.axis);
  TryGetValue(j, "parallel_group", all_gather_task_info.parallel_group);
  TryGetValue(j, "output_allocator", all_gather_task_info.output_allocator);
}

USED_BY_JSON void to_json(Json &j, const AllToAllReshardTask &task_info) {
  j = Json();
  j["task_type"] = kCommTaskTypeHcomAllToAll;
  j["comm_groups"] = task_info.comm_groups;
  j["parallel_group"] = task_info.parallel_group;
}

USED_BY_JSON void from_json(const Json &j, AllToAllReshardTask &all_to_all_task_info) {
  GetValue(j, "comm_groups", all_to_all_task_info.comm_groups);
  TryGetValue(j, "parallel_group", all_to_all_task_info.parallel_group);
}

USED_BY_JSON void to_json(Json &j, const AllReduceReshardTask &task_info) {
  j = Json();
  j["task_type"] = kCommTaskTypeHcomAllReduce;
  j["comm_groups"] = task_info.comm_groups;
  j["reduction"] = task_info.reduction;
  j["parallel_group"] = task_info.parallel_group;
}

USED_BY_JSON void from_json(const Json &j, AllReduceReshardTask &all_reduce_task_info) {
  GetValue(j, "reduction", all_reduce_task_info.reduction);
  GetValue(j, "comm_groups", all_reduce_task_info.comm_groups);
  TryGetValue(j, "parallel_group", all_reduce_task_info.parallel_group);
}

USED_BY_JSON void to_json(Json &j, const AllReduceMeanReshardTask &task_info) {
  j = Json();
  j["task_type"] = kCommTaskTypeHcomAllReduceMean;
  j["comm_groups"] = task_info.comm_groups;
  j["axis"] = task_info.axis;
  j["value"] = task_info.value;
  j["parallel_group"] = task_info.parallel_group;
}

USED_BY_JSON void from_json(const Json &j, AllReduceMeanReshardTask &task_info) {
  GetValue(j, "comm_groups", task_info.comm_groups);
  GetValue(j, "axis", task_info.axis);
  GetValue(j, "value", task_info.value);
  TryGetValue(j, "parallel_group", task_info.parallel_group);
}

USED_BY_JSON void to_json(Json &j, const ReduceScatterReshardTask &task_info) {
  j = Json();
  j["task_type"] = kCommTaskTypeHcomReduceScatter;
  j["comm_groups"] = task_info.comm_groups;
  j["reduction"] = task_info.reduction;
  j["parallel_group"] = task_info.parallel_group;
}

USED_BY_JSON void from_json(const Json &j, ReduceScatterReshardTask &reduce_scatter_task_info) {
  GetValue(j, "reduction", reduce_scatter_task_info.reduction);
  GetValue(j, "comm_groups", reduce_scatter_task_info.comm_groups);
  TryGetValue(j, "parallel_group", reduce_scatter_task_info.parallel_group);
}

USED_BY_JSON void to_json(Json &j, const BroadcastReshardTask &task_info) {
  j = Json();
  j["task_type"] = kCommTaskTypeHcomBroadcast;
  j["comm_groups"] = task_info.comm_groups;
  j["roots"] = task_info.root_device_indices;
  j["parallel_group"] = task_info.parallel_group;
}

USED_BY_JSON void from_json(const Json &j, BroadcastReshardTask &broadcast_task_info) {
  GetValue(j, "roots", broadcast_task_info.root_device_indices);
  GetValue(j, "comm_groups", broadcast_task_info.comm_groups);
  TryGetValue(j, "parallel_group", broadcast_task_info.parallel_group);
}

USED_BY_JSON void to_json(Json &j, const SliceReshardTask &task_info) {
  j = Json();
  j["task_type"] = kCommTaskTypeSlice;
  j["axes"] = task_info.axes;
  j["offsets"] = task_info.offsets;
  j["size"] = task_info.sizes;
  j["device_index"] = task_info.device_index;
}

USED_BY_JSON void from_json(const Json &j, SliceReshardTask &task_info) {
  TryGetValue(j, "axes", task_info.axes);
  GetValue(j, "offsets", task_info.offsets);
  GetValue(j, "size", task_info.sizes);
  TryGetValue(j, "device_index", task_info.device_index);
}

USED_BY_JSON void to_json(Json &j, const SliceByAxisReshardTask &task_info) {
  j = Json();
  j["task_type"] = kCommTaskTypeSliceByAxis;
  j["axis_to_slice_deployments"] = task_info.axis_to_slice_deployments;
}

USED_BY_JSON void from_json(const Json &j, SliceByAxisReshardTask &task_info) {
  GetValue(j, "axis_to_slice_deployments", task_info.axis_to_slice_deployments);
}

USED_BY_JSON void to_json(Json &j, const ConcatReshardTask &task_info) {
  j = Json();
  j["task_type"] = kCommTaskTypeConcat;
  j["concat_dim"] = task_info.concat_dim;
}

USED_BY_JSON void from_json(const Json &j, ConcatReshardTask &task_info) {
  GetValue(j, "concat_dim", task_info.concat_dim);
}

USED_BY_JSON void to_json(Json &j, const UniqueConcatReshardTask &task_info) {
  j = Json();
  j["task_type"] = kCommTaskTypeUniqueConcat;
  j["unique_id"] = task_info.unique_id;
  j["concat_dim"] = task_info.concat_dim;
  j["src_device_indices"] = task_info.src_device_indices;
  j["dst_device_index"] = task_info.dst_device_index;
}

USED_BY_JSON void from_json(const Json &j, UniqueConcatReshardTask &task_info) {
  TryGetValue(j, "unique_id", task_info.unique_id);
  GetValue(j, "concat_dim", task_info.concat_dim);
  GetValue(j, "src_device_indices", task_info.src_device_indices);
  GetValue(j, "dst_device_index", task_info.dst_device_index);
}

USED_BY_JSON void to_json(Json &j, const SplitReshardTask &task_info) {
  j = Json();
  j["task_type"] = kCommTaskTypeSplit;
  j["num_split"] = task_info.num_split;
  j["split_dim"] = task_info.split_dim;
}

USED_BY_JSON void from_json(const Json &j, SplitReshardTask &task_info) {
  GetValue(j, "num_split", task_info.num_split);
  GetValue(j, "split_dim", task_info.split_dim);
}

USED_BY_JSON void to_json(Json &j, const TransposeReshardTask &task_info) {
  j = Json();
  j["task_type"] = kCommTaskTypeTranspose;
  j["perm"] = task_info.perm;
}

USED_BY_JSON void from_json(const Json &j, TransposeReshardTask &task_info) {
  GetValue(j, "perm", task_info.perm);
}

USED_BY_JSON void to_json(Json &j, const ReshapeReshardTask &task_info) {
  j = Json();
  j["task_type"] = kCommTaskTypeReshape;
  j["shape"] = task_info.shape;
}

USED_BY_JSON void from_json(const Json &j, ReshapeReshardTask &task_info) {
  GetValue(j, "shape", task_info.shape);
}

USED_BY_JSON void to_json(Json &j, const ModifyValueReshardTask &task_info) {
  j = Json();
  j["task_type"] = kCommTaskTypeModifyValue;
  j["op_type"] = task_info.op_type;
  j["value"] = task_info.value;
}

USED_BY_JSON void from_json(const Json &j, ModifyValueReshardTask &task_info) {
  GetValue(j, "op_type", task_info.op_type);
  GetValue(j, "value", task_info.value);
}

USED_BY_JSON void to_json(Json &j, const CastReshardTask &task_info) {
  j = Json();
  j["task_type"] = kCommTaskTypeCast;
  j["dst_type"] = static_cast<int32_t>(task_info.dst_type);
}

USED_BY_JSON void from_json(const Json &j, CastReshardTask &task_info) {
  int32_t dst_type = -1;
  GetValue(j, "dst_type", dst_type);
  task_info.dst_type = static_cast<DataType>(dst_type);
}

USED_BY_JSON void to_json(Json &j, const CommTask &comm_task) {
  GE_CHK_STATUS(CommTaskBuilder::GetInstance().ConvertToJson(comm_task, j));
}

USED_BY_JSON void from_json(const Json &j, CommTask &comm_task) {
  CommTaskBuilder::GetInstance().BuildCommTask(j, comm_task);
}

USED_BY_JSON void to_json(Json &j, const CommStepInput &step_input) {
  j = std::vector<int32_t>{step_input.step_id, step_input.output_index};
}

USED_BY_JSON void from_json(const Json &j, CommStepInput &step_input) {
  const auto step_id_and_out_index = j.get<std::vector<int32_t>>();
  const size_t num_items = step_id_and_out_index.size();
  if (num_items > kIndexStepId) {
    step_input.step_id = step_id_and_out_index[kIndexStepId];
  }
  if (num_items > kIndexOutputIndex) {
    step_input.output_index = step_id_and_out_index[kIndexOutputIndex];
  }
}

USED_BY_JSON void to_json(Json &j, const CommStep &comm_step) {
  j = Json();
  j["id"] = comm_step.id;
  if (!comm_step.inputs.empty()) {
    j["input_ids"] = comm_step.inputs;
  }
  j["comm_task"] = comm_step.comm_task;
}

USED_BY_JSON void from_json(const Json &j, CommStep &comm_step) {
  comm_step.id = j.at("id").get<int32_t>();
  if (j.contains("input_ids")) {
    comm_step.inputs = j.at("input_ids").get<std::vector<CommStepInput>>();
  }
  comm_step.comm_task = j.at("comm_task").get<CommTask>();
}

USED_BY_JSON void to_json(Json &j, const PeerInput &peer_input) {
  j = Json();
  j["step_id"] = peer_input.step_id;
  j["node_name"] = peer_input.node_name;
  j["input_index"] = peer_input.input_index;
  j["stage_id"] = peer_input.stage_id;
  j["virtual_stage_id"] = peer_input.virtual_stage_id;
}

USED_BY_JSON void from_json(const Json &j, PeerInput &peer_input) {
  GetValue(j, "step_id", peer_input.step_id);
  GetValue(j, "node_name", peer_input.node_name);
  GetValue(j, "input_index", peer_input.input_index);
  TryGetValue(j, "stage_id", peer_input.stage_id);
  TryGetValue(j, "virtual_stage_id", peer_input.virtual_stage_id);
}

USED_BY_JSON void to_json(Json &j, const OutputReshardRes &reshard_res) {
  j = Json();
  j["comm_steps"] = reshard_res.comm_steps;
  j["peer_inputs"] = reshard_res.peer_inputs;
  j["device_list"] = reshard_res.device_indices;
  j["stage_id"] = reshard_res.stage_id;
  j["virtual_stage_id"] = reshard_res.virtual_stage_id;
}

USED_BY_JSON void from_json(const Json &j, OutputReshardRes &reshard_res) {
  GetValue(j, "comm_steps", reshard_res.comm_steps);
  GetValue(j, "peer_inputs", reshard_res.peer_inputs);
  GetValue(j, "device_list", reshard_res.device_indices);
  TryGetValue(j, "stage_id", reshard_res.stage_id);
  TryGetValue(j, "virtual_stage_id", reshard_res.virtual_stage_id);
}

USED_BY_JSON void to_json(Json &j, const ReshardAttr &reshard_attr) {
  j = reshard_attr.reshard_infos;
}

USED_BY_JSON void to_json(Json &j, const ShardGraphExtAttrs &shard_graph_ext_attrs) {
  j = Json();
  j["dev_index_to_logic_dev_id"] = shard_graph_ext_attrs.dev_index_to_logic_dev_id;
  j["graph_name_to_endpoints"] = shard_graph_ext_attrs.graph_name_to_endpoints;
  j["group_name_to_dev_ids"] = shard_graph_ext_attrs.group_name_to_dev_ids;
}

USED_BY_JSON void from_json(const Json &j, ShardGraphExtAttrs &shard_graph_ext_attrs) {
  shard_graph_ext_attrs.dev_index_to_logic_dev_id =
      j.at("dev_index_to_logic_dev_id").get<std::map<DeviceIndex, std::vector<int32_t>>>();
  shard_graph_ext_attrs.graph_name_to_endpoints =
      j.at("graph_name_to_endpoints").get<std::map<std::string, std::map<std::string, std::vector<std::string>>>>();
  shard_graph_ext_attrs.group_name_to_dev_ids =
      j.at("group_name_to_dev_ids").get<std::map<std::string, std::vector<std::string>>>();
}

USED_BY_JSON void from_json(const Json &j, ReshardAttr &reshard_attr) {
  reshard_attr.reshard_infos = j.get<std::vector<std::vector<OutputReshardRes>>>();
}

Status TensorParallelAttrs::FromJson(const std::string &json_str, ShardGraphExtAttrs &shard_graph_ext_attrs) {
  return ParseFromJson("ShardGraphExtAttrs", json_str, shard_graph_ext_attrs);
}

Status TensorParallelAttrs::FromJson(const std::string &json_str, DeviceIndex &device_index) {
  return ParseFromJson("DeviceIndex", json_str, device_index);
}

Status TensorParallelAttrs::FromJson(const std::string &json_str, ModelIndex &model_index) {
  return ParseFromJson("ModelIndex", json_str, model_index);
}

Status TensorParallelAttrs::FromJson(const std::string &json_str, PipelineConfig &pipeline_config) {
  return ParseFromJson("PipelineConfig", json_str, pipeline_config);
}


Status TensorParallelAttrs::FromJson(const std::string &json_str,
                                     ReshardAttr &reshard_attr) {
  return ParseFromJson("ReshardRes", json_str, reshard_attr);
}

Status TensorParallelAttrs::FromJson(const std::string &json_str,
                                     TensorDeployment &tensor_deployment) {
  return ParseFromJson("TensorDeployment", json_str, tensor_deployment);
}

Status TensorParallelAttrs::FromJson(const std::string &json_str,
                                     TensorDeployments &tensor_deployments) {
  return ParseFromJson("TensorDeployments", json_str, tensor_deployments);
}

Status TensorParallelAttrs::FromJson(const std::string &json_str,
                                     NodeDeployments &node_deployments) {
  return ParseFromJson("NodeDeployments", json_str, node_deployments);
}

Status TensorParallelAttrs::FromJson(const std::string &json_str, CommTask &comm_task) {
  return ParseFromJson("CommTask", json_str, comm_task);
}

Status TensorParallelAttrs::FromJson(const std::string &json_str, CommStep &comm_step) {
  return ParseFromJson("CommStep", json_str, comm_step);
}

Status TensorParallelAttrs::FromJson(const std::string &json_str,
                                     OutputReshardRes &output_reshard_res) {
  return ParseFromJson("TensorReshardInfo", json_str, output_reshard_res);
}

Status TensorParallelAttrs::FromJson(const std::string &json_str, NodeDeployment &node_deployment) {
  return ParseFromJson("NodeDeployment", json_str, node_deployment);
}

std::string TensorParallelAttrs::ToJson(const ShardGraphExtAttrs &shard_graph_ext_attrs) {
  return ToJsonString(shard_graph_ext_attrs);
}

std::string TensorParallelAttrs::ToJson(const DeviceIndex &device_index) {
  return ToJsonString(device_index);
}

std::string TensorParallelAttrs::ToJson(const ModelIndex &model_index) {
  return ToJsonString(model_index);
}

std::string TensorParallelAttrs::ToJson(const PipelineConfig &pipeline_config) {
  return ToJsonString(pipeline_config);
}

std::string TensorParallelAttrs::ToJson(const NodeDeployment &node_deployment) {
  return ToJsonString(node_deployment);
}

std::string TensorParallelAttrs::ToJson(const TensorDeployment &tensor_deployment) {
  return ToJsonString(tensor_deployment);
}

std::string TensorParallelAttrs::ToJson(const ReshardAttr &reshard_attr) {
  return ToJsonString(reshard_attr);
}

std::string TensorParallelAttrs::ToJson(const TensorDeployments &tensor_deployments) {
  return ToJsonString(tensor_deployments);
}

std::string TensorParallelAttrs::ToJson(const NodeDeployments &node_deployments) {
  return ToJsonString(node_deployments);
}

CommTaskBuilder::CommTaskBuilder() {
  InitCommTaskBuilders();
  InitJsonConverters();
}

void CommTaskBuilder::InitCommTaskBuilders() {
  builders_[kCommTaskTypeSlice] = [](const Json &j, CommTask &comm_task) {
    comm_task.slice_reshard_task = CreateReshardTaskInfo<SliceReshardTask>(j);
  };
  builders_[kCommTaskTypeSliceByAxis] = [](const Json &j, CommTask &comm_task) {
    comm_task.slice_by_axis_reshard_task = CreateReshardTaskInfo<SliceByAxisReshardTask>(j);
  };
  builders_[kCommTaskTypeSplit] = [](const Json &j, CommTask &comm_task) {
    comm_task.split_reshard_task = CreateReshardTaskInfo<SplitReshardTask>(j);
  };
  builders_[kCommTaskTypeConcat] = [](const Json &j, CommTask &comm_task) {
    comm_task.concat_reshard_task = CreateReshardTaskInfo<ConcatReshardTask>(j);
  };
  builders_[kCommTaskTypeUniqueConcat] = [](const Json &j, CommTask &comm_task) {
    comm_task.unique_concat_reshard_task = CreateReshardTaskInfo<UniqueConcatReshardTask>(j);
  };
  builders_[kCommTaskTypeTranspose] = [](const Json &j, CommTask &comm_task) {
    comm_task.transpose_reshard_task = CreateReshardTaskInfo<TransposeReshardTask>(j);
  };
  builders_[kCommTaskTypeHcomAllGather] = [](const Json &j, CommTask &comm_task) {
    comm_task.all_gather_reshard_task = CreateReshardTaskInfo<AllGatherReshardTask>(j);
  };
  builders_[kCommTaskTypeHcomAllReduce] = [](const Json &j, CommTask &comm_task) {
    comm_task.all_reduce_reshard_task = CreateReshardTaskInfo<AllReduceReshardTask>(j);
  };
  builders_[kCommTaskTypeHcomAllReduceMean] = [](const Json &j, CommTask &comm_task) {
    comm_task.all_reduce_mean_reshard_task = CreateReshardTaskInfo<AllReduceMeanReshardTask>(j);
  };
  builders_[kCommTaskTypeHcomReduceScatter] = [](const Json &j, CommTask &comm_task) {
    comm_task.reduce_scatter_reshard_task = CreateReshardTaskInfo<ReduceScatterReshardTask>(j);
  };
  builders_[kCommTaskTypeHcomBroadcast] = [](const Json &j, CommTask &comm_task) {
    comm_task.broadcast_reshard_task = CreateReshardTaskInfo<BroadcastReshardTask>(j);
  };
  builders_[kCommTaskTypeHcomAllToAll] = [](const Json &j, CommTask &comm_task) {
    comm_task.all_to_all_reshard_task = CreateReshardTaskInfo<AllToAllReshardTask>(j);
  };
  builders_[kCommTaskTypeSendReceive] = [](const Json &j, CommTask &comm_task) {
    comm_task.send_recv_reshard_task = CreateReshardTaskInfo<SendRecvReshardTask>(j);
  };
  builders_[kCommTaskTypeModifyValue] = [](const Json &j, CommTask &comm_task) {
    comm_task.modify_value_reshard_task = CreateReshardTaskInfo<ModifyValueReshardTask>(j);
  };
  builders_[kCommTaskTypeReshape] = [](const Json &j, CommTask &comm_task) {
    comm_task.reshape_reshard_task = CreateReshardTaskInfo<ReshapeReshardTask>(j);
  };
  builders_[kCommTaskTypeCast] = [](const Json &j, CommTask &comm_task) {
    comm_task.cast_reshard_task = CreateReshardTaskInfo<CastReshardTask>(j);
  };
}

template<typename T>
Status CommTaskBuilder::ConvertToJson(const T *reshard_task, nlohmann::json &j) {
  GE_CHECK_NOTNULL(reshard_task);
  j = *reshard_task;
  return SUCCESS;
}

void CommTaskBuilder::InitJsonConverters() {
  json_converters_[kCommTaskTypeSlice] = [](const CommTask &comm_task, nlohmann::json &j) {
    return ConvertToJson(comm_task.slice_reshard_task.get(), j);
  };
  json_converters_[kCommTaskTypeSliceByAxis] = [](const CommTask &comm_task, nlohmann::json &j) {
    return ConvertToJson(comm_task.slice_by_axis_reshard_task.get(), j);
  };
  json_converters_[kCommTaskTypeSplit] = [](const CommTask &comm_task, nlohmann::json &j) {
    return ConvertToJson(comm_task.split_reshard_task.get(), j);
  };
  json_converters_[kCommTaskTypeConcat] = [](const CommTask &comm_task, nlohmann::json &j) {
    return ConvertToJson(comm_task.concat_reshard_task.get(), j);
  };
  json_converters_[kCommTaskTypeUniqueConcat] = [](const CommTask &comm_task, nlohmann::json &j) {
    return ConvertToJson(comm_task.unique_concat_reshard_task.get(), j);
  };
  json_converters_[kCommTaskTypeTranspose] = [](const CommTask &comm_task, nlohmann::json &j) {
    return ConvertToJson(comm_task.transpose_reshard_task.get(), j);
  };
  json_converters_[kCommTaskTypeHcomAllGather] = [](const CommTask &comm_task, nlohmann::json &j) {
    return ConvertToJson(comm_task.all_gather_reshard_task.get(), j);
  };
  json_converters_[kCommTaskTypeHcomAllReduce] = [](const CommTask &comm_task, nlohmann::json &j) {
    return ConvertToJson(comm_task.all_reduce_reshard_task.get(), j);
  };
  json_converters_[kCommTaskTypeHcomAllReduceMean] = [](const CommTask &comm_task, nlohmann::json &j) {
    return ConvertToJson(comm_task.all_reduce_mean_reshard_task.get(), j);
  };
  json_converters_[kCommTaskTypeHcomReduceScatter] = [](const CommTask &comm_task, nlohmann::json &j) {
    return ConvertToJson(comm_task.reduce_scatter_reshard_task.get(), j);
  };
  json_converters_[kCommTaskTypeHcomBroadcast] = [](const CommTask &comm_task, nlohmann::json &j) {
    return ConvertToJson(comm_task.broadcast_reshard_task.get(), j);
  };
  json_converters_[kCommTaskTypeHcomAllToAll] = [](const CommTask &comm_task, nlohmann::json &j) {
    return ConvertToJson(comm_task.all_to_all_reshard_task.get(), j);
  };
  json_converters_[kCommTaskTypeSendReceive] = [](const CommTask &comm_task, nlohmann::json &j) {
    return ConvertToJson(comm_task.send_recv_reshard_task.get(), j);
  };
  json_converters_[kCommTaskTypeModifyValue] = [](const CommTask &comm_task, nlohmann::json &j) {
    return ConvertToJson(comm_task.modify_value_reshard_task.get(), j);
  };
  json_converters_[kCommTaskTypeReshape] = [](const CommTask &comm_task, nlohmann::json &j) {
    return ConvertToJson(comm_task.reshape_reshard_task.get(), j);
  };
  json_converters_[kCommTaskTypeCast] = [](const CommTask &comm_task, nlohmann::json &j) {
    return ConvertToJson(comm_task.cast_reshard_task.get(), j);
  };
}

bool operator==(const DeviceIndex &lhs, const DeviceIndex &rhs) {
  return lhs.engine_type == rhs.engine_type &&
      lhs.indices == rhs.indices;
}

bool operator!=(const DeviceIndex &lhs, const DeviceIndex &rhs) {
  return !(rhs == lhs);
}

bool operator<(const DeviceIndex &lhs, const DeviceIndex &rhs) {
  if (lhs.engine_type < rhs.engine_type) {
    return true;
  }
  if (rhs.engine_type < lhs.engine_type) {
    return false;
  }
  return lhs.indices < rhs.indices;
}

bool operator==(const ModelIndex &lhs, const ModelIndex &rhs) {
  return (lhs.device_index == rhs.device_index) && (lhs.virtual_stage_id == rhs.virtual_stage_id);
}

bool operator!=(const ModelIndex &lhs, const ModelIndex &rhs) {
  return !(rhs == lhs);
}

bool operator<(const ModelIndex &lhs, const ModelIndex &rhs) {
  if (lhs.virtual_stage_id < rhs.virtual_stage_id) {
    return true;
  }
  if (rhs.virtual_stage_id < lhs.virtual_stage_id) {
    return false;
  }
  return lhs.device_index < rhs.device_index;
}

bool operator==(const CommStepInput &lhs, const CommStepInput &rhs) {
  return (lhs.step_id == rhs.step_id) && (lhs.output_index == rhs.output_index);
}

bool operator<(const CommStepInput &lhs, const CommStepInput &rhs) {
  if (lhs.step_id < rhs.step_id) {
    return true;
  }
  if (rhs.step_id < lhs.step_id) {
    return false;
  }
  return lhs.output_index < rhs.output_index;
}

bool operator==(const SrcNodeInfo &lhs, const SrcNodeInfo &rhs) {
  return (lhs.inserted_node_id == rhs.inserted_node_id) && (lhs.output_index == rhs.output_index);
}
bool operator<(const SrcNodeInfo &lhs, const SrcNodeInfo &rhs) {
  if (lhs.inserted_node_id < rhs.inserted_node_id) {
    return true;
  }
  if (rhs.inserted_node_id < lhs.inserted_node_id) {
    return false;
  }
  return lhs.output_index < rhs.output_index;
}

bool operator==(const OrigNodeInfo &lhs, const OrigNodeInfo &rhs) {
  return (lhs.node_name == rhs.node_name) && (lhs.sliced_id == rhs.sliced_id);
}

bool operator<(const OrigNodeInfo &lhs, const OrigNodeInfo &rhs) {
  if (lhs.node_name < rhs.node_name) {
    return true;
  }
  if (rhs.node_name < lhs.node_name) {
    return false;
  }
  return lhs.sliced_id < rhs.sliced_id;
}

bool operator==(const DstNodeInfo &lhs, const DstNodeInfo &rhs) {
  return (lhs.orig_node_info == rhs.orig_node_info) && (lhs.input_indexes == rhs.input_indexes);
}

bool operator<(const DstNodeInfo &lhs, const DstNodeInfo &rhs) {
  if (lhs.orig_node_info < rhs.orig_node_info) {
    return true;
  }
  if (rhs.orig_node_info < lhs.orig_node_info) {
    return false;
  }
  return lhs.InputIndexesToString() < rhs.InputIndexesToString();
}

bool operator==(const InsertedNodeInput &lhs, const InsertedNodeInput &rhs) {
  if ((lhs.input_info.inserted_node_id >= 0) && (rhs.input_info.inserted_node_id >= 0)) {
    return (lhs.input_info == rhs.input_info);
  }
  if ((lhs.input_info.inserted_node_id < 0) && (rhs.input_info.inserted_node_id < 0)) {
    return (lhs.input_info == rhs.input_info) && (lhs.orig_node_info == rhs.orig_node_info);
  }
  return false;
}
bool operator<(const InsertedNodeInput &lhs, const InsertedNodeInput &rhs) {
  if (lhs.input_info < rhs.input_info) {
    return true;
  }
  if (rhs.input_info < lhs.input_info) {
    return false;
  }
  return lhs.orig_node_info < rhs.orig_node_info;
}

bool operator==(const PeerOutNodeInfo &lhs, const PeerOutNodeInfo &rhs) {
  return (lhs.input_info == rhs.input_info) && (lhs.node_info == rhs.node_info);
}

bool operator<(const PeerOutNodeInfo &lhs, const PeerOutNodeInfo &rhs) {
  if (lhs.input_info < rhs.input_info) {
    return true;
  }
  if (rhs.input_info < lhs.input_info) {
    return false;
  }
  return lhs.node_info < rhs.node_info;
}

std::string ModelIndex::DebugString() const {
  return device_index.DebugString() + "[S" + std::to_string(stage_id) + ", V" + std::to_string(virtual_stage_id) + "]";
}
}  // namespace tp
}  // namespace ge
