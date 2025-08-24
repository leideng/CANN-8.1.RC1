/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "external/graph/graph_buffer.h"
#include "graph/buffer.h"
#include "common/util/mem_utils.h"

namespace ge {
GraphBuffer::GraphBuffer() : buffer_(MakeShared<Buffer>()) {}

GraphBuffer::~GraphBuffer() {}

const std::uint8_t *GraphBuffer::GetData() const {
  return buffer_->GetData();
}

std::size_t GraphBuffer::GetSize() const {
  return buffer_->GetSize();
}
}  // namespace ge
