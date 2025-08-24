/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "external/register/register_custom_pass.h"

#include "common/checker.h"

#include "register/custom_pass_helper.h"
#include "graph/debug/ge_log.h"
#include "graph/debug/ge_util.h"
#include "common/plugin/plugin_manager.h"
#include "register/custom_pass_context_impl.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/graph_utils.h"

namespace ge {
const std::set<CustomPassStage> kConstGraphStages = {CustomPassStage::kAfterAssignLogicStream};
const std::map<CustomPassStage, std::string> kCustomPassStageToStringMap = {
    {CustomPassStage::kBeforeInferShape, "BeforeInferShape"},
    {CustomPassStage::kAfterInferShape, "AfterInferShape"},
    {CustomPassStage::kAfterAssignLogicStream, "AfterAssignLogicStream"},
    {CustomPassStage::kInvalid, "InvalidStage"}
};

namespace {
std::string CustomPassStageToString(CustomPassStage stage) {
  GE_ASSERT_TRUE(stage <= CustomPassStage::kInvalid);
  return kCustomPassStageToStringMap.find(stage)->second;
}

Status RunAllocateStreamPass(const PassRegistrationData &reg_data, const GraphPtr &graph,
                             CustomPassContext &custom_pass_context) {
  GE_ASSERT_NOTNULL(graph);
  const auto allocate_stream_pass_func = reg_data.GetCustomAllocateStreamPass();
  if (allocate_stream_pass_func == nullptr) {
    GE_LOGE("[Check][Param] It is required CustomAllocateStreamPassFunc of [%s] at stage[%s] but got nullptr.",
            reg_data.GetPassName().c_str(), CustomPassStageToString(reg_data.GetStage()).c_str());
    std::stringstream reason;
    reason << "It is required CustomAllocateStreamPassFunc in stage " << CustomPassStageToString(reg_data.GetStage()) <<
        ", but got nullptr";
    REPORT_INPUT_ERROR("E19030", std::vector<std::string>({"passname", "reason"}),
                       std::vector<std::string>({reg_data.GetPassName(), reason.str()}));
    return FAILED;
  }

  const auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph);
  GE_ASSERT_NOTNULL(compute_graph);
  const auto root_graph = GraphUtils::FindRootGraph(compute_graph);
  GE_ASSERT_NOTNULL(root_graph);

  GE_DUMP(root_graph, "RunCustomPass_BeforeAssignLogicStream");
  // 此处框架保证传入的custom_pass_context实例为stream_pass_context
  auto *stream_pass_context = dynamic_cast<StreamPassContext *>(&custom_pass_context);
  GE_ASSERT_NOTNULL(stream_pass_context, "Failed to transfer CustomPassContext to StreamPassContext");
  const auto ret = allocate_stream_pass_func(graph, *stream_pass_context);
  GE_DUMP(root_graph, "RunCustomPass_AfterAssignLogicStream");
  if (ret != SUCCESS) {
    GE_LOGE("Execution of custom pass [%s] failed! Reason: %s.", reg_data.GetPassName().c_str(),
            custom_pass_context.GetErrorMessage().GetString());
    REPORT_INPUT_ERROR("E19028", std::vector<std::string>({"passname", "retcode", "reason"}),
                       std::vector<std::string>({reg_data.GetPassName(), std::to_string(ret),
                         std::string(custom_pass_context.GetErrorMessage().GetString())}));
    return FAILED;
  }
  return SUCCESS;
}

Status RunCustomPass(const PassRegistrationData &reg_data, GraphPtr &graph, CustomPassContext &custom_pass_context) {
  const auto custom_pass_fn = reg_data.GetCustomPassFn();
  if (custom_pass_fn == nullptr) {
    GELOGW("[Check][Param] Failed to retrieve custom_pass_fn for custom pass %s failed",
           reg_data.GetPassName().c_str());
    return SUCCESS;
  }
  const auto ret = custom_pass_fn(graph, custom_pass_context);
  if (ret != SUCCESS) {
    GE_LOGE("Execution of custom pass [%s] failed! Reason: %s.", reg_data.GetPassName().c_str(),
            custom_pass_context.GetErrorMessage().GetString());
    REPORT_INPUT_ERROR("E19028", std::vector<std::string>({"passname", "retcode", "reason"}),
                       std::vector<std::string>({reg_data.GetPassName(), std::to_string(ret),
                         std::string(custom_pass_context.GetErrorMessage().GetString())}));
    return FAILED;
  }
  return SUCCESS;
}
} // namespace
PassReceiver::PassReceiver(PassRegistrationData &reg_data) {
  CustomPassHelper::Instance().Insert(reg_data);
}

class PassRegistrationDataImpl {
 public:
  PassRegistrationDataImpl() = default;
  ~PassRegistrationDataImpl() = default;

  explicit PassRegistrationDataImpl(const std::string &pass_name);

private:
  friend class PassRegistrationData;
  std::string pass_name_;
  CustomPassFunc custom_pass_ = nullptr;
  CustomAllocateStreamPassFunc allocate_stream_pass_ = nullptr;
  CustomPassStage stage_ = CustomPassStage::kBeforeInferShape;
};

PassRegistrationDataImpl::PassRegistrationDataImpl(const std::string &pass_name)
    : pass_name_(pass_name), custom_pass_(nullptr) {}

PassRegistrationData::PassRegistrationData(std::string pass_name) {
  impl_ = ge::ComGraphMakeShared<PassRegistrationDataImpl>(pass_name);
  if (impl_ == nullptr) {
    GELOGW("[Check][Param] make impl failed, pass_name:%s", pass_name.c_str());
  }
}

std::string PassRegistrationData::GetPassName() const {
  if (impl_ == nullptr) {
    return "";
  }
  return impl_->pass_name_;
}

PassRegistrationData &PassRegistrationData::CustomPassFn(const CustomPassFunc &custom_pass_fn) {
  if (impl_ != nullptr) {
    impl_->custom_pass_ = custom_pass_fn;
  }
  return *this;
}

PassRegistrationData &PassRegistrationData::CustomAllocateStreamPassFn(
    const CustomAllocateStreamPassFunc &allocate_stream_pass_fn) {
  if (impl_ != nullptr) {
    impl_->allocate_stream_pass_ = allocate_stream_pass_fn;
    impl_->stage_ = CustomPassStage::kAfterAssignLogicStream;
  }
  return *this;
}

CustomPassFunc PassRegistrationData::GetCustomPassFn() const {
  if (impl_ == nullptr) {
    return nullptr;
  }
  return impl_->custom_pass_;
}

CustomAllocateStreamPassFunc PassRegistrationData::GetCustomAllocateStreamPass() const {
  GE_ASSERT_NOTNULL(impl_);
  return impl_->allocate_stream_pass_;
}

PassRegistrationData &PassRegistrationData::Stage(const CustomPassStage stage) {
  if ((impl_ != nullptr) && (stage < CustomPassStage::kInvalid)) {
    impl_->stage_ = stage;
    GELOGD("Setting pass [%s] stage to [%s]", impl_->pass_name_.c_str(), CustomPassStageToString(stage).c_str());
  }
  return *this;
}

CustomPassStage PassRegistrationData::GetStage() const {
  if (impl_ == nullptr) {
    return CustomPassStage::kInvalid;
  }
  return impl_->stage_;
}

CustomPassContext::CustomPassContext() {
  impl_ = ComGraphMakeUnique<CustomPassContextImpl>();
  if (impl_ == nullptr) {
    GELOGW("[Check][Param] make impl failed");
  }
}

void CustomPassContext::SetErrorMessage(const AscendString &error_message) {
  if (impl_ != nullptr) {
    impl_->SetErrorMessage(error_message);
  }
}

AscendString CustomPassContext::GetErrorMessage() const {
  if (impl_ != nullptr) {
    return impl_->GetErrorMessage();
  }
  return "";
}

StreamPassContext::StreamPassContext(int64_t current_max_stream_id) {
  impl_ = ge::ComGraphMakeUnique<StreamPassContextImpl>(current_max_stream_id);
  if (impl_ == nullptr) {
    GELOGW("[Check][Param] make StreamPassContextImpl failed");
  }
}

graphStatus StreamPassContext::SetStreamId(const GNode &node, int64_t stream_id) {
  GE_ASSERT_NOTNULL(impl_);
  return impl_->SetStreamId(node, stream_id);
}

int64_t StreamPassContext::GetStreamId(const GNode &node) const {
  int64_t stream_id;
  if ((impl_ == nullptr) || impl_->GetStreamId(node, stream_id) != SUCCESS) {
    return INVALID_STREAM_ID;
  }
  return stream_id;
}

int64_t StreamPassContext::GetCurrMaxStreamId() const {
  GE_ASSERT_NOTNULL(impl_);
  return impl_->GetCurrentMaxStreamId();
}

int64_t StreamPassContext::AllocateNextStreamId() {
  if (impl_ != nullptr) {
    return impl_->AllocateNextStreamId();
  }
  return INT64_MAX;
}

CustomPassHelper &CustomPassHelper::Instance() {
  static CustomPassHelper instance;
  return instance;
}

void CustomPassHelper::Insert(const PassRegistrationData &reg_data) {
  (void)registration_datas_.emplace_back(reg_data);
}

Status CustomPassHelper::Load() {
  GELOGD("[Load][CustomPassLibs] Start to load custom pass libs");
  std::string opp_path;
  ge::PluginManager::GetOppPath(opp_path);
  std::string vendors_path = opp_path + "/vendors";

  // 存储所有的 .so 文件路径
  std::vector<std::string> so_files;
  ge::PluginManager::FindSoFilesInCustomPassDirs(vendors_path, so_files);

  if (so_files.empty()) {
    GELOGD("No custom pass libs found in %s, skip loading custom pass libs", vendors_path.c_str());
    return ge::SUCCESS;
  }

  // 逐个 dlopen
  for (const auto &so_file : so_files) {
    void *handle = dlopen(so_file.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
      const char* error = dlerror();
      REPORT_INPUT_ERROR("E19029", std::vector<std::string>({"passlibname", "reason"}),
                         std::vector<std::string>({so_file, error}));
      GELOGE(ge::FAILED, "Failed to load %s: %s", so_file.c_str(), error);
      return ge::FAILED;
    }
    handles_.emplace_back(handle);
    GELOGI("Load custom pass lib %s success", so_file.c_str());
  }
  return ge::SUCCESS;
}

Status CustomPassHelper::Unload() {
  registration_datas_.clear();
  for (auto &handle : handles_) {
    if (handle != nullptr && dlclose(handle) != 0) {
      GELOGE(ge::FAILED, "[Unload][CustomPassLibs] Failed to unload custom pass lib: %s", dlerror());
      return ge::FAILED;
    }
    GELOGI("Unload custom pass lib success");
  }
  handles_.clear();
  return ge::SUCCESS;
}

Status CustomPassHelper::Run(GraphPtr &graph, CustomPassContext &custom_pass_context) const {
  return Run(graph, custom_pass_context, CustomPassStage::kBeforeInferShape);
}

Status CustomPassHelper::Run(GraphPtr &graph, CustomPassContext &custom_pass_context,
                             const CustomPassStage stage) const {
  for (auto &item : registration_datas_) {
    if (item.GetStage() != stage) {
      continue;
    }
    GELOGD("Starting custom pass [%s] in stage [%s]!", item.GetPassName().c_str(), CustomPassStageToString(stage).c_str());
    if (stage == CustomPassStage::kAfterAssignLogicStream) {
      GE_ASSERT_SUCCESS(RunAllocateStreamPass(item, graph, custom_pass_context));
    } else {
      GE_ASSERT_SUCCESS(RunCustomPass(item, graph, custom_pass_context));
    }
    GELOGD("Run custom pass [%s] success!", item.GetPassName().c_str());
  }
  return SUCCESS;
}
}  // namespace ge
