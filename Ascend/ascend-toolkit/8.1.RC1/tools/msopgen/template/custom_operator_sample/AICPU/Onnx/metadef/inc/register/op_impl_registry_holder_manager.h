/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_OP_IMPL_REGISTRY_HOLDER_MANAGER_H_
#define INC_OP_IMPL_REGISTRY_HOLDER_MANAGER_H_

#include <string>
#include <unordered_map>
#include <map>
#include <mutex>
#include "graph/op_so_bin.h"
#include "register/op_impl_registry_api.h"
#include "register/op_ct_impl_registry_api.h"

namespace gert {

enum class ImplType {
  RT_TYPE = 0,
  CT_TYPE = 1,
  RT_V2_TYPE = 2,
  END_TYPE
};

class OpImplRegistryHolder {
 public:
  OpImplRegistryHolder() = default;

  virtual ~OpImplRegistryHolder();

  std::map<OpImplRegisterV2::OpType, OpImplKernelRegistry::OpImplFunctionsV2> &GetTypesToImpl() {
    return types_v2_to_impl_;
  }

  std::map<OpCtImplKernelRegistry::OpType, OpCtImplKernelRegistry::OpCtImplFunctions> &GetTypesToCtImpl() {
    return types_to_ct_impl_;
  }

  void SetHandle(const void *handle) { handle_ = const_cast<void *>(handle); }

  ge::graphStatus GetOpImplFunctionsByHandle(const void *handle, const std::string &so_path);

  std::unique_ptr<TypesToImpl[]> GetOpImplFunctionsByHandle(const void *handle,
                                                            const std::string &so_path,
                                                            size_t &impl_num) const;
  void AddTypesToImpl(const gert::OpImplRegisterV2::OpType op_type,
                      const gert::OpImplKernelRegistry::OpImplFunctionsV2 funcs);
 protected:
  std::map<OpImplRegisterV2::OpType, OpImplKernelRegistry::OpImplFunctions> types_to_impl_;
  std::map<OpImplRegisterV2::OpType, OpImplKernelRegistry::OpImplFunctionsV2> types_v2_to_impl_;
  std::map<OpCtImplKernelRegistry::OpType, OpCtImplKernelRegistry::OpCtImplFunctions> types_to_ct_impl_;
  std::vector<void*> impl_map_vec_{&types_v2_to_impl_, &types_to_impl_, &types_to_ct_impl_};
  void *handle_ = nullptr;
};
using OpImplRegistryHolderPtr = std::shared_ptr<OpImplRegistryHolder>;

class OmOpImplRegistryHolder : public OpImplRegistryHolder {
 public:
  OmOpImplRegistryHolder() = default;

  virtual ~OmOpImplRegistryHolder() override = default;

  ge::graphStatus LoadSo(const std::shared_ptr<ge::OpSoBin> &so_bin);

 private:
  ge::graphStatus CreateOmOppDir(std::string &opp_dir) const;

  ge::graphStatus RmOmOppDir(const std::string &opp_dir) const;

  ge::graphStatus SaveToFile(const std::shared_ptr<ge::OpSoBin> &so_bin, const std::string &opp_path) const;
};

class OpImplRegistryHolderManager {
 public:
  OpImplRegistryHolderManager() = default;

  ~OpImplRegistryHolderManager();

  static OpImplRegistryHolderManager &GetInstance();

  void AddRegistry(std::string &so_data, const std::shared_ptr<OpImplRegistryHolder> &registry_holder);

  void UpdateOpImplRegistries();

  const OpImplRegistryHolderPtr GetOpImplRegistryHolder(std::string &so_data);

  OpImplRegistryHolderPtr GetOrCreateOpImplRegistryHolder(std::string &so_data,
                                                          const std::string &so_name,
                                                          const ge::SoInOmInfo &so_info,
                                                          const std::function<OpImplRegistryHolderPtr()> create_func);

  size_t GetOpImplRegistrySize() {
    const std::lock_guard<std::mutex> lock(map_mutex_);
    return op_impl_registries_.size();
  }

  void ClearOpImplRegistries() {
    const std::lock_guard<std::mutex> lock(map_mutex_);
    op_impl_registries_.clear();
  }

 private:
  /**
   * 背景：当前加载的so里边包含了算子原型等自注册(如OperatorFactoryImpl::operator_infer_axis_type_info_funcs等static变量，在进程退出前析构)。
   * 原方案使用weak_ptr管理OpImplRegistryHolder，不影响生命周期，引用计数一旦减为0，将触发析构，并关闭so的句柄；
   * 如果OpImplRegistryHolder在比较早的时机析构，那么进程退出时，operator_infer_axis_type_info_funcs这些static变量将无法正常析构。
   * 此处临时改为shared_ptr，使OpImplRegistryHolder与本类单例的生命周期一致，从而能够确保在进程退出前才触发析构，规避上述问题。
   * todo 此处是规避方案，后续将继续使用weak_ptr来管理OpImplRegistryHolder，后续将梳理上述自注册机制，修改成space_registry注册机制
   * */
  std::unordered_map<std::string, std::shared_ptr<OpImplRegistryHolder>> op_impl_registries_;
  std::mutex map_mutex_;
};
}  // namespace gert
#endif  // INC_OP_IMPL_REGISTRY_HOLDER_MANAGER_H_
