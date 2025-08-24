/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_EXE_GRAPH_LOWERING_EXE_RES_GENERATION_CTX_BUILDER_H
#define INC_EXE_GRAPH_LOWERING_EXE_RES_GENERATION_CTX_BUILDER_H

#include "external/exe_graph/runtime/exe_res_generation_context.h"
#include "graph/node.h"
#include "exe_graph/lowering/kernel_run_context_builder.h"

namespace gert {
using ExeResGenerationCtxHolderPtr = std::shared_ptr<gert::KernelContextHolder>;
class ExeResGenerationCtxBuilder {
 public:
  ExeResGenerationCtxHolderPtr CreateOpExeContext(ge::Node &node);
  ExeResGenerationCtxHolderPtr CreateOpCheckContext(ge::Node &node);
 private:
  void CreateShapesInputs(const ge::Node &node, std::vector<void *> &inputs);
 private:
  ExeResGenerationCtxHolderPtr ctx_holder_ptr_;
  std::vector<StorageShape> input_shapes_;
  std::vector<StorageShape> output_shapes_;
};
} // namespace fe

#endif
