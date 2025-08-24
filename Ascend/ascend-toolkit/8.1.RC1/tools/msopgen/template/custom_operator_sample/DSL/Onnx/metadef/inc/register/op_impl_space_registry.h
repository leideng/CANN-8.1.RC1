/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_OP_IMPL_SPACE_REGISTRY_H_
#define INC_OP_IMPL_SPACE_REGISTRY_H_

#include "register/op_impl_registry_holder_manager.h"
#include "graph/op_desc.h"

namespace gert {
using OpTypesToImplMap = std::map<OpImplRegisterV2::OpType, OpImplKernelRegistry::OpImplFunctionsV2>;
using OpTypesToCtImplMap = std::map<OpCtImplKernelRegistry::OpType, OpCtImplKernelRegistry::OpCtImplFunctions>;
class OpImplSpaceRegistry : public std::enable_shared_from_this<OpImplSpaceRegistry> {
 public:
  OpImplSpaceRegistry() = default;

  ~OpImplSpaceRegistry() = default;

  ge::graphStatus GetOrCreateRegistry(const std::vector<ge::OpSoBinPtr> &bins, const ge::SoInOmInfo &so_info);
  // 兼容air，待air合入后删除
  ge::graphStatus GetOrCreateRegistry(const std::vector<ge::OpSoBinPtr> &bins, const ge::SoInOmInfo &so_info,
                                      const std::string &opp_path_identifier);

  ge::graphStatus AddRegistry(const std::shared_ptr<OpImplRegistryHolder> &registry_holder);

  const OpImplKernelRegistry::OpImplFunctionsV2 *GetOpImpl(const std::string &op_type) const;

  OpImplKernelRegistry::OpImplFunctionsV2 *CreateOrGetOpImpl(const std::string &op_type);

  const OpCtImplKernelRegistry::OpCtImplFunctions *GetOpCtImpl(const std::string &op_type) const;

  const OpImplRegisterV2::PrivateAttrList &GetPrivateAttrs(const std::string &op_type) const;

  static ge::graphStatus LoadSoAndSaveToRegistry(const std::string &so_path);

  static ge::graphStatus ConvertSoToRegistry(const std::string &so_path, ge::OppImplVersion opp_impl_version);

 private:
  void MergeTypesToImpl(OpTypesToImplMap &merged_impl, OpTypesToImplMap &src_impl) const;
  void MergeTypesToCtImpl(OpTypesToCtImplMap &merged_impl, OpTypesToCtImplMap &src_impl) const;
  void MergeFunctions(OpImplKernelRegistry::OpImplFunctionsV2 &merged_funcs,
                      const OpImplKernelRegistry::OpImplFunctionsV2 &src_funcs, const std::string &op_type) const;
  void MergeCtFunctions(OpCtImplKernelRegistry::OpCtImplFunctions &merged_funcs,
                        const OpCtImplKernelRegistry::OpCtImplFunctions &src_funcs, const std::string &op_type) const;
 private:
  std::vector<std::shared_ptr<OpImplRegistryHolder>> op_impl_registries_;
  OpTypesToImplMap merged_types_to_impl_;
  OpTypesToCtImplMap  merged_types_to_ct_impl_;
};
using OpImplSpaceRegistryPtr = std::shared_ptr<OpImplSpaceRegistry>;
using OpImplSpaceRegistryArray =
  std::array<OpImplSpaceRegistryPtr, static_cast<size_t>(ge::OppImplVersion::kVersionEnd)>;

class DefaultOpImplSpaceRegistry {
 public:
  DefaultOpImplSpaceRegistry() = default;

  ~DefaultOpImplSpaceRegistry() = default;

  static DefaultOpImplSpaceRegistry &GetInstance();

  const OpImplSpaceRegistryArray &GetDefaultSpaceRegistries() const {
    return space_registries_;
  }

  OpImplSpaceRegistryPtr &GetDefaultSpaceRegistry(ge::OppImplVersion opp_impl_version = ge::OppImplVersion::kOpp) {
    if (opp_impl_version >= ge::OppImplVersion::kVersionEnd) {
      opp_impl_version = ge::OppImplVersion::kOpp;
    }
    return space_registries_[static_cast<size_t>(opp_impl_version)];
  }

  void SetDefaultSpaceRegistry(const OpImplSpaceRegistryPtr &space_registry,
                               ge::OppImplVersion opp_impl_version = ge::OppImplVersion::kOpp) {
    space_registries_[static_cast<size_t>(opp_impl_version)] = space_registry;
  }

 private:
  OpImplSpaceRegistryArray space_registries_;
};
}  // namespace gert
#endif  // INC_OP_IMPL_SPACE_REGISTRY_H_
