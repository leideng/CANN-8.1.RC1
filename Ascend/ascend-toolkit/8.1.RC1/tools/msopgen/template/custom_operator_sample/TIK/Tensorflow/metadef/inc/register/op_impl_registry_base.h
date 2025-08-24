/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_EXTERNAL_REGISTER_OP_IMPL_REGISTRY_BASE_H_
#define INC_EXTERNAL_REGISTER_OP_IMPL_REGISTRY_BASE_H_
#include "register/op_impl_kernel_registry.h"
namespace gert {
struct OpImplRegistryBase : public OpImplKernelRegistry {
  virtual ~OpImplRegistryBase() = default;
  virtual const OpImplFunctions *GetOpImpl(const ge::char_t *op_type) const = 0;
  virtual const OpImplRegisterV2::PrivateAttrList &GetPrivateAttrs(const ge::char_t *op_type) const = 0;
};

class OpImplRegistry : public OpImplRegistryBase {
 public:
  static OpImplRegistry &GetInstance();
  OpImplRegistry::OpImplFunctionsV2 &CreateOrGetOpImpl(const ge::char_t *op_type);
  const OpImplRegistry::OpImplFunctionsV2 *GetOpImpl(const ge::char_t *op_type) const override;
  const OpImplRegisterV2::PrivateAttrList &GetPrivateAttrs(const ge::char_t *op_type) const override;
  const std::map<OpImplRegisterV2::OpType, OpImplRegistry::OpImplFunctionsV2> &GetAllTypesToImpl() const;
  std::map<OpImplRegisterV2::OpType, OpImplRegistry::OpImplFunctionsV2> &GetAllTypesToImpl();

 private:
  std::map<OpImplRegisterV2::OpType, OpImplRegistry::OpImplFunctionsV2> types_to_impl_;
  uint8_t reserved_[40] = {0U};  // Reserved field, 32+8, do not directly use when only 8-byte left
};
}  // namespace gert

#endif
