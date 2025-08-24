/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef GE_RUNTIME_TILING_PARSE_CONTEXT_BUILDER_H_
#define GE_RUNTIME_TILING_PARSE_CONTEXT_BUILDER_H_

#include "exe_graph/runtime/kernel_context.h"
#include "graph/operator.h"
#include "exe_graph/lowering/kernel_run_context_builder.h"
#include "register/op_impl_registry.h"

namespace gert {
class TilingParseContextBuilder {
 public:
  TilingParseContextBuilder &CompileJson(const ge::char_t *compile_json);
  TilingParseContextBuilder &PlatformInfo(void *platform_info);
  TilingParseContextBuilder &CompileInfoCreatorFunc(OpImplRegisterV2::CompileInfoCreatorFunc create_func);
  TilingParseContextBuilder &CompileInfoDeleterFunc(OpImplRegisterV2::CompileInfoDeleterFunc delete_func);
  KernelContextHolder Build(const ge::Operator &op);

 private:
  void *compile_json_{ nullptr };
  void *platform_info_{ nullptr };
  OpImplRegisterV2::CompileInfoCreatorFunc create_func_{ nullptr };
  OpImplRegisterV2::CompileInfoDeleterFunc delete_func_{ nullptr };
};
}  // namespace gert
#endif // GE_RUNTIME_TILING_PARSE_CONTEXT_BUILDER_H_
