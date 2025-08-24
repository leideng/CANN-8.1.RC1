/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "register/op_lib_register_impl.h"
#include "graph/debug/ge_util.h"

#include "mmpa/mmpa_api.h"
#include "common/ge_common/debug/ge_log.h"
#include "common/ge_common/string_util.h"
#include "common/checker.h"
#include "common/plugin/plugin_manager.h"
#include "graph/utils/file_utils.h"

namespace {
  const ge::char_t *const custom_so_name = "libcust_opapi.so";
}

namespace ge {
OpLibRegister::OpLibRegister(const char_t *vendor_name) : impl_(ComGraphMakeUnique<OpLibRegisterImpl>()) {
  if (impl_ != nullptr) {
    impl_->MutableVendorName() = vendor_name;
  }
}

OpLibRegister::OpLibRegister(const OpLibRegister &other) {
  if (other.impl_ != nullptr) {
    OpLibRegistry::GetInstance().RegisterInitFunc(*other.impl_);
  }
}

OpLibRegister::OpLibRegister(OpLibRegister &&other) noexcept {
  if (other.impl_ != nullptr) {
    OpLibRegistry::GetInstance().RegisterInitFunc(*other.impl_);
  }
}

OpLibRegister::~OpLibRegister() = default;

OpLibRegister &OpLibRegister::RegOpLibInit(OpLibRegister::OpLibInitFunc func) {
  if (impl_ != nullptr) {
    impl_->MutableInitFunc() = func;
  }
  return *this;
}

OpLibRegistry &OpLibRegistry::GetInstance() {
  static OpLibRegistry instance;
  return instance;
}

const char_t* OpLibRegistry::GetCustomOpLibPath() {
  GELOGI("get op lib path is %s", op_lib_paths_.c_str());
  return op_lib_paths_.c_str();
}

void OpLibRegistry::RegisterInitFunc(OpLibRegisterImpl &register_impl) {
  const std::string vendor_name = register_impl.MutableVendorName();
  auto func = register_impl.MutableInitFunc();
  const std::lock_guard<std::mutex> lk(mu_);
  const auto it = vendor_names_set_.insert(vendor_name);
  // ignore same vendor_name op lib when register secondly
  if (it.second) {
    if (func != nullptr) {
      vendor_funcs_.emplace_back(vendor_name, func);
    }
    GELOGI("%s op lib register successfully", vendor_name.c_str());
  } else {
    GELOGW("%s op lib has already registered", vendor_name.c_str());
  }
}

/**
 * @brief 对环境变量下ASCEND_CUSTOM_OPP_PATH新so交付的自定义算子目录作预处理，需要保证在获取自定义算子目录前调用，
 * @brief 当前提供metadef接口, air仓各个流程初始化靠前的位置调用
 *
 * 当前最新的自定义算子工程交付分为run包交付和so交付（新做的）两种形式：
 * 新的so交付的形式下：export ASCEND_CUSTOM_OPP_PATH=/path/to/customize:/path/to/mdc:/path/to/lhisi
 * 三个目录下都只有一个libcust_opapi.so
 *
 * 老的run包交付的形式下：export ASCEND_CUSTOM_OPP_PATH=/path/to/customize:/path/to/mdc:/path/to/lhisi
 * 三个目录下都有完整的算子子目录，如op_proto,op_impl子目录等
 *
 * 当前支持两种方式混用。混用优先级以新的so交付方式优先。
 * 例如export ASCEND_CUSTOM_OPP_PATH=/home/a:/home/b:/home/c,其中只有/home/b是新so交付的方式
 * 则最终优先级别顺序为b,a,c
 * @return
 */
graphStatus OpLibRegistry::PreProcessForCustomOp() {
  if (is_processed_) {
    GELOGD("pre process for custom op has already been called");
    return GRAPH_SUCCESS;
  }
  std::string custom_opp_path;
  const char *const custom_opp_path_env = std::getenv("ASCEND_CUSTOM_OPP_PATH");
  if (custom_opp_path_env != nullptr) {
    custom_opp_path = custom_opp_path_env;
  }
  std::vector<std::string> so_real_paths;
  GE_ASSERT_GRAPH_SUCCESS(GetAllCustomOpApiSoPaths(custom_opp_path, so_real_paths));
  GE_ASSERT_GRAPH_SUCCESS(CallInitFunc(custom_opp_path, so_real_paths));
  is_processed_ = true;
  return GRAPH_SUCCESS;
}

graphStatus OpLibRegistry::GetAllCustomOpApiSoPaths(const std::string &custom_opp_path,
                                                    std::vector<std::string> &so_real_paths) const {
  if (custom_opp_path.empty()) {
    GELOGI("custom_opp_path is empty, no need to get custom op so");
    return GRAPH_SUCCESS;
  }
  GELOGI("value of env ASCEND_CUSTOM_OPP_PATH is %s.", custom_opp_path.c_str());
  std::vector<std::string> current_custom_opp_path = StringUtils::Split(custom_opp_path, ':');

  if (current_custom_opp_path.empty()) {
    GELOGI("find no custom opp path, just return");
    return GRAPH_SUCCESS;
  }

  for (const auto &path : current_custom_opp_path) {
    if (path.empty()) {
      continue;
    }
    const std::string so_path = path + "/" + custom_so_name;
    std::string so_real_path = RealPath(so_path.c_str());
    if (!so_real_path.empty()) {
      GELOGI("find so_real_path %s", so_real_path.c_str());
      so_real_paths.emplace_back(so_real_path);
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus OpLibRegistry::CallInitFunc(const std::string &custom_opp_path,
                                        const std::vector<std::string> &so_real_paths) {
  // dlopen so orderly
  const int32_t mode = static_cast<int32_t>(static_cast<uint32_t>(MMPA_RTLD_NOW));
  for (const auto &so_path : so_real_paths) {
    GELOGI("begin dlopen %s", so_path.c_str());
    void* const handle = mmDlopen(so_path.c_str(), mode);
    GE_ASSERT_NOTNULL(handle, "Failed to dlopen %s! errmsg:%s", so_path.c_str(), mmDlerror());
    handles_.emplace_back(handle);
  }

  // call init func orderly
  const std::lock_guard<std::mutex> lk(mu_);
  for (auto &vendor_func : vendor_funcs_) {
    GELOGI("begin to call %s init func", vendor_func.first.c_str());
    AscendString tmp_dir("");
    GE_ASSERT_GRAPH_SUCCESS(vendor_func.second(tmp_dir));
    GELOGI("end to call %s init func, tmp_dir is %s", vendor_func.first.c_str(), tmp_dir.GetString());
    op_lib_paths_ += (std::string(tmp_dir.GetString()) + ":");
  }
  if (custom_opp_path.empty()) { // ignore the end :
    op_lib_paths_ = op_lib_paths_.substr(0, op_lib_paths_.find_last_of(':'));
  } else {
    op_lib_paths_ += custom_opp_path; // add origin env path to ensure priority(so mode first, runbag mode second)
  }
  PluginManager::SetCustomOpLibPath(op_lib_paths_);
  GELOGI("CallInitFunc %zu successfully, op_lib_paths_ is %s", vendor_funcs_.size(), op_lib_paths_.c_str());
  return GRAPH_SUCCESS;
}

void OpLibRegistry::ClearHandles() {
  for (auto handle : handles_) {
    (void)mmDlclose(handle);
  }
  handles_.clear();
}

OpLibRegistry::~OpLibRegistry() {
  ClearHandles();
}
} // namespace ge
