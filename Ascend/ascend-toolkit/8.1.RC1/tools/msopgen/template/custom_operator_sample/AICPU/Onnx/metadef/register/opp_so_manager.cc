/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "register/opp_so_manager.h"

#include <string>

#include "common/plugin/plugin_manager.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/debug/ge_log.h"
#include "graph/opsproto_manager.h"
#include "mmpa/mmpa_api.h"
#include "register/op_impl_space_registry.h"
#include "register/op_impl_registry_holder_manager.h"
#include "register/shape_inference.h"

namespace ge {
namespace {
const char_t *const kBuiltIn = "built-in";
const char_t *const kSoSuffix = ".so";
const char_t *const kRt2SoSuffix = "rt2.0.so";
const char_t *const kCtSoSuffix = "ct.so";
const char_t *const kRtSoSuffix = "rt.so";
void GetOppSoList(const std::string &opp_so_path, std::vector<std::string> &so_list, bool is_split) {
  if (opp_so_path.find(kBuiltIn) != std::string::npos) {
    // *_ct.so all need to be loaded
    // *_rt.so* is mutually exclusive with *rt2.0.so*
    PluginManager::GetFileListWithSuffix(opp_so_path, kCtSoSuffix, so_list);
    const auto &rt_so_name = is_split ? kRtSoSuffix : kRt2SoSuffix;
    PluginManager::GetFileListWithSuffix(opp_so_path, rt_so_name, so_list);
  } else {
    PluginManager::GetFileListWithSuffix(opp_so_path, kSoSuffix, so_list);
  }
  for (const auto &so_name : so_list) {
    GELOGD("GetOppSoList from path %s, so_name is %s", opp_so_path.c_str(), so_name.c_str());
  }
}
void LoadSoAndInitDefault(const std::vector<std::string> &so_list, OppImplVersion opp_impl_version) {
  GELOGI("Start to LoadSoAndInitDefault, opp version:%d", opp_impl_version);
  for (const auto &so_path : so_list) {
    if (gert::OpImplSpaceRegistry::ConvertSoToRegistry(so_path, opp_impl_version) != ge::SUCCESS) {
      GELOGW("Save so failed, path is %s", so_path.c_str());
    }
  }
}
}  // namespace

void OppSoManager::LoadOpsProtoPackage() const {
  auto is_split = PluginManager::IsSplitOpp();
  GELOGI("Start load ops proto package, is_split:[%d].", is_split);
  LoadOpsProtoSo(is_split);
  if (is_split) {
    GELOGD("Start load upgraded ops proto package.");
    LoadUpgradedOpsProtoSo();
  }
}

void OppSoManager::LoadOpMasterPackage() const {
  auto is_split = PluginManager::IsSplitOpp();
  GELOGI("Start load op master package, is_split:[%d].", is_split);
  LoadOpMasterSo(is_split);
  if (is_split) {
    GELOGD("Start load upgraded op master package.");
    LoadUpgradedOpMasterSo();
  }
}

void OppSoManager::LoadOppPackage() const {
  LoadOpsProtoPackage();
  LoadOpMasterPackage();
}

void OppSoManager::LoadOpsProtoSo(bool is_split) const {
  std::string ops_proto_path;
  const Status ret = PluginManager::GetOpsProtoPath(ops_proto_path);
  if (ret != SUCCESS) {
    return;
  }

  std::string os_type;
  std::string cpu_type;
  PluginManager::GetCurEnvPackageOsAndCpuType(os_type, cpu_type);

  std::vector<std::string> v_path;
  PluginManager::SplitPath(ops_proto_path, v_path);
  for (size_t i = 0UL; i < v_path.size(); ++i) {
    std::string path = v_path[i] + "lib/" + os_type + "/" + cpu_type + "/";
    char_t resolved_path[MMPA_MAX_PATH] = {};
    const INT32 result = mmRealPath(path.c_str(), &(resolved_path[0U]), MMPA_MAX_PATH);
    if (result != EN_OK) {
      GELOGW("[FindSo][Check] Get path with os&cpu type [%s] failed, reason:%s", path.c_str(), strerror(errno));
      continue;
    }
    std::vector<std::string> so_list;
    GetOppSoList(path, so_list, is_split);
    LoadSoAndInitDefault(so_list, OppImplVersion::kOpp);
  }
}

void OppSoManager::LoadOpMasterSo(bool is_split) const {
  std::string op_tiling_path;
  const Status ret = PluginManager::GetOpTilingForwardOrderPath(op_tiling_path);
  if (ret != SUCCESS) {
    return;
  }

  std::string os_type;
  std::string cpu_type;
  PluginManager::GetCurEnvPackageOsAndCpuType(os_type, cpu_type);

  std::vector<std::string> path_vec;
  PluginManager::SplitPath(op_tiling_path, path_vec);
  for (const auto &path : path_vec) {
    std::string root_path = path + "op_master/lib/" + os_type + "/" + cpu_type + "/";
    char_t resolved_path[MMPA_MAX_PATH] = {};
    if (mmRealPath(root_path.c_str(), &(resolved_path[0U]), MMPA_MAX_PATH) != EN_OK) {
      GELOGW("Get path with op_master path [%s] failed, reason:%s", root_path.c_str(), strerror(errno));
      root_path = path + "op_tiling/lib/" + os_type + "/" + cpu_type + "/";
      if (mmRealPath(root_path.c_str(), &(resolved_path[0U]), MMPA_MAX_PATH) != EN_OK) {
        GELOGW("Get path with op_tiling path [%s] failed, reason:%s", root_path.c_str(), strerror(errno));
        continue;
      }
    }
    std::vector<std::string> so_list;
    GetOppSoList(root_path, so_list, is_split);
    LoadSoAndInitDefault(so_list, OppImplVersion::kOpp);
  }
}

void OppSoManager::LoadUpgradedOpsProtoSo() const {
  std::string ops_proto_path;
  if (PluginManager::GetUpgradedOpsProtoPath(ops_proto_path) != ge::SUCCESS) {
    return;
  }

  std::string os_type;
  std::string cpu_type;
  PluginManager::GetCurEnvPackageOsAndCpuType(os_type, cpu_type);

  std::vector<std::string> path_vec;
  PluginManager::SplitPath(ops_proto_path, path_vec);
  for (const auto &path : path_vec) {
    std::string root_path = path + "/lib/" + os_type + "/" + cpu_type + "/";
    char_t resolved_path[MMPA_MAX_PATH] = {};
    if (mmRealPath(root_path.c_str(), &(resolved_path[0U]), MMPA_MAX_PATH) != EN_OK) {
      GELOGW("[FindSo][Check] Get upgraded path with os&cpu type [%s] failed, reason:%s", root_path.c_str(),
             strerror(errno));
      continue;
    }

    std::vector<std::string> so_list;
    GetOppSoList(root_path, so_list, true);
    LoadSoAndInitDefault(so_list, OppImplVersion::kOppKernel);
  }
}

void OppSoManager::LoadUpgradedOpMasterSo() const {
  std::string op_tiling_path;
  if (PluginManager::GetUpgradedOpMasterPath(op_tiling_path) != ge::SUCCESS) {
    return;
  }

  std::string os_type;
  std::string cpu_type;
  PluginManager::GetCurEnvPackageOsAndCpuType(os_type, cpu_type);

  std::vector<std::string> path_vec;
  PluginManager::SplitPath(op_tiling_path, path_vec);

  for (const auto &path : path_vec) {
    std::string root_path = path + "/op_master/lib" + os_type + "/" + cpu_type + "/";
    char_t resolved_path[MMPA_MAX_PATH] = {};
    if (mmRealPath(root_path.c_str(), &(resolved_path[0U]), MMPA_MAX_PATH) != EN_OK) {
      GELOGW("[FindSo][Check] Get upgraded path with op_master path [%s] failed, reason:%s", root_path.c_str(),
             strerror(errno));
      root_path = path + "/op_tiling/lib/" + os_type + "/" + cpu_type + "/";
      if (mmRealPath(root_path.c_str(), &(resolved_path[0U]), MMPA_MAX_PATH) != EN_OK) {
        GELOGW("[FindSo][Check] Get upgraded path with op_tiling path [%s] failed, reason:%s", root_path.c_str(),
               strerror(errno));
        continue;
      }
    }

    std::vector<std::string> so_list;
    GetOppSoList(root_path, so_list, true);
    LoadSoAndInitDefault(so_list, OppImplVersion::kOppKernel);
  }
}

OppSoManager &OppSoManager::GetInstance() {
  static OppSoManager instance;
  return instance;
}
}  // namespace ge
