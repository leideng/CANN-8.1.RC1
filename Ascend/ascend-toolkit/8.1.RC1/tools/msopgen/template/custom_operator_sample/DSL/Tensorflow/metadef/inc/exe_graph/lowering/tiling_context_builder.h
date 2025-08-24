/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef GE_COMMMON_RUNTIME_TILING_KERNEL_CONTEXT_BUILDER_H_
#define GE_COMMMON_RUNTIME_TILING_KERNEL_CONTEXT_BUILDER_H_

#include "graph/node.h"
#include "exe_graph/runtime/compute_node_info.h"
#include "exe_graph/runtime/kernel_context.h"
#include "exe_graph/lowering/buffer_pool.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/lowering/kernel_run_context_builder.h"
#include "register/op_impl_space_registry.h"

namespace gert {
class TilingContextBuilder {
 public:
  TilingContextBuilder &CompileInfo(void *compile_info);
  TilingContextBuilder &Deterministic(int32_t deterministic);
  TilingContextBuilder &PlatformInfo(void *platform_info);
  TilingContextBuilder &TilingData(void *tiling_data);
  TilingContextBuilder &Workspace(ContinuousVector *workspace);
  // 兼容air, 随后air合入后删除
  TilingContextBuilder &SpaceRegistry(const gert::OpImplSpaceRegistryPtr &space_registry);
  TilingContextBuilder &SpaceRegistries(const gert::OpImplSpaceRegistryArray &space_registries);
  KernelContextHolder Build(const ge::Operator &op); // deprecated later
  KernelContextHolder Build(const ge::Operator &op, ge::graphStatus &ret);

 private:
  ge::graphStatus GetDependInputTensorAddr(const ge::Operator &op, const size_t input_idx, TensorAddress &address);
  ge::graphStatus BuildRtTensor(const ge::GeTensorDesc &tensor_desc, ConstTensorAddressPtr address,
                                std::unique_ptr<uint8_t[]> &rt_tensor_holder) const;
  ge::graphStatus BuildRTInputTensors(const ge::Operator &op);
  ge::graphStatus BuildRTOutputShapes(const ge::Operator &op);

  void *compile_info_{nullptr};
  void *platform_info_{nullptr};
  int32_t deterministic_;
  std::vector<std::unique_ptr<ge::Tensor>> depend_ge_tensor_holders_;
  std::vector<std::unique_ptr<uint8_t[]>> rt_tensor_holders_;
  std::vector<void *> outputs_ {TilingContext::kOutputNum};
  KernelRunContextBuilder base_builder_;
  OpImplSpaceRegistryArray space_registries_;
};

class AtomicTilingContextBuilder {
 public:
  AtomicTilingContextBuilder &CompileInfo(void *compile_info);
  AtomicTilingContextBuilder &CleanWorkspaceSizes(ContinuousVector *workspace_sizes);
  AtomicTilingContextBuilder &CleanOutputSizes(const std::vector<int64_t> &output_sizes);
  AtomicTilingContextBuilder &TilingData(void *tiling_data);
  AtomicTilingContextBuilder &Workspace(ContinuousVector *workspace);
  KernelContextHolder Build(const ge::Operator &op); // deprecated later
  KernelContextHolder Build(const ge::Operator &op, ge::graphStatus &ret);

 private:
  void *compile_info_{nullptr};
  void *worksapce_sizes_{nullptr};
  std::vector<int64_t> clean_output_sizes_;
  std::vector<void *> outputs_ {TilingContext::kOutputNum};
  KernelRunContextBuilder base_builder_;
};
}  // namespace gert
#endif // GE_COMMMON_RUNTIME_TILING_KERNEL_CONTEXT_BUILDER_H_
