/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef AIR_CXX_HOST_RESOURCE_SERIALIZER_H
#define AIR_CXX_HOST_RESOURCE_SERIALIZER_H

#include <vector>
#include <map>
#include "framework/common/host_resource_center/host_resource_center.h"
namespace ge {
struct ResourceSubPartition {
  size_t offset_;
  size_t size_;
};
struct HostResourcePartition {
  std::vector<uint8_t> buffer_;
  size_t total_size_;
  std::map<HostResourceType, ResourceSubPartition> resource_types_2_subpartition_;
};

class HostResourceSerializer {
 public:
  graphStatus SerializeTilingData(HostResourceCenter &host_resource_center, uint8_t *&data, size_t &len);
  static graphStatus SerializeTilingData(const HostResourceCenter &host_resource_center,
                                         HostResourcePartition &partition);
  static graphStatus DeSerializeTilingData(const HostResourceCenter &host_resource_center, const uint8_t *const data,
                                           size_t length);
  static graphStatus RecoverOpRunInfoToExtAttrs(const HostResourceCenter &host_resource_center,
                                                const ComputeGraphPtr &graph);

 private:
  std::vector<HostResourcePartition> partitions_;
};
}  // namespace ge
#endif  // AIR_CXX_HOST_RESOURCE_SERIALIZER_H
