/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef AIR_CXX_HOST_RESOURCE_CENTER_H
#define AIR_CXX_HOST_RESOURCE_CENTER_H
#include <array>
#include "common/host_resource_center/host_resource_manager.h"
namespace ge {
using HostResourceMgrPtr = std::unique_ptr<HostResourceManager>;
enum class HostResourceType : uint32_t {
  kWeight = 0,  // including const/constant
  kTilingData,
  kNum,
};
class HostResourceCenter {
 public:
  HostResourceCenter();
  const HostResourceManager *GetHostResourceMgr(HostResourceType type) const;
  Status TakeOverHostResources(const ComputeGraphPtr &root_graph);
  ~HostResourceCenter() = default;

 private:
  std::array<HostResourceMgrPtr, static_cast<size_t>(HostResourceType::kNum)> resource_managers_{nullptr};
};
using HostResourceCenterPtr = std::shared_ptr<HostResourceCenter>;
}  // namespace ge
#endif  // AIR_CXX_HOST_RESOURCE_CENTER_H