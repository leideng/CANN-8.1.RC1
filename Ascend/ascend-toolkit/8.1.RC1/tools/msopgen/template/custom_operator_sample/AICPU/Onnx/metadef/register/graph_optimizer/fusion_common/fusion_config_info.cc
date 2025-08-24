/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "register/graph_optimizer/fusion_common/fusion_config_info.h"
#include "mmpa/mmpa_api.h"
#include "graph/debug/ge_log.h"

namespace fe {
namespace {
const std::string kEnableNetworkAnalysis = "ENABLE_NETWORK_ANALYSIS_DEBUG";
}

FusionConfigInfo& FusionConfigInfo::Instance() {
  static FusionConfigInfo fusion_config_info;
  return fusion_config_info;
}

Status FusionConfigInfo::Initialize() {
  if (is_init_) {
    return SUCCESS;
  }

  InitEnvParam();
  is_init_ = true;
  return SUCCESS;
}

void FusionConfigInfo::InitEnvParam() {
  char env_value[MMPA_MAX_PATH];
  const INT32 ret = mmGetEnv(kEnableNetworkAnalysis.c_str(), env_value, MMPA_MAX_PATH);
  if (ret == EN_OK) {
    std::string env_str_value = std::string(env_value);
    GELOGD("The value of env[%s] is [%s].", kEnableNetworkAnalysis.c_str(), env_str_value.c_str());
    is_enable_network_analysis_ = static_cast<bool> (std::stoi(env_str_value.c_str()));
  }
  GELOGD("Enable network analysis is [%d].", is_enable_network_analysis_);
}

Status FusionConfigInfo::Finalize() {
  is_init_ = false;
  is_enable_network_analysis_ = false;
  return SUCCESS;
}

bool FusionConfigInfo::IsEnableNetworkAnalysis() const {
  return is_enable_network_analysis_;
}
}
