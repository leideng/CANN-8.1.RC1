/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file atomic_clean_tiling_context.h
 * \brief
 */
#ifndef OPS_COMMON_INC_ATOMIC_CLEAN_TILING_CONTEXT_H
#define OPS_COMMON_INC_ATOMIC_CLEAN_TILING_CONTEXT_H
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/continuous_vector.h"
namespace ops {
class AtomicCleanTilingContext : public gert::TilingContext {
 public:
  /**
   * Get workspaceSize list
   * @return workspaceSize list
   */
  const gert::ContinuousVector *GetCleanWorkspaceSizes() const {
    return GetInputPointer<gert::ContinuousVector>(0);
  }

  /**
   * Get the size of the output memory to be cleared by the output index
   * @param index output index
   * @return Size of output memory to be cleaned
   */
  uint64_t GetCleanOutputSize(size_t index) const {
    return GetInputValue<uint64_t>(index + 1U);
  }
};

static_assert(std::is_standard_layout<AtomicCleanTilingContext>::value,
              "The class AtomicCleanTilingContext must be a POD.");
}  // namespace ops
#endif  // OPS_COMMON_INC_ATOMIC_CLEAN_TILING_CONTEXT_H
