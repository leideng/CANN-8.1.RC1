/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "common/ge_common/debug/ge_log.h"
#include "register/op_ct_impl_registry_api.h"
#include "register/op_ct_impl_registry.h"

namespace gert {
namespace {
void RegisterOpImplToRegistry(const OpCtImplRegisterV2Impl *rd) {
  if (rd == nullptr) {
    GELOGW("The register data is invalid, the ct impl is nullptr");
    return;
  }
  auto &funcs = OpCtImplRegistry::GetInstance().CreateOrGetOpImpl(rd->op_type.GetString());
  if (rd->functions.calc_op_param != nullptr) {
    GELOGD("Op type:%s reg calc func.", rd->op_type.GetString());
    funcs.calc_op_param = rd->functions.calc_op_param;
  }
  if (rd->functions.gen_task != nullptr) {
    GELOGD("Op type:%s reg gen task func.", rd->op_type.GetString());
    funcs.gen_task = rd->functions.gen_task;
  }
  if (rd->functions.check_support != nullptr) {
    GELOGD("Op type:%s reg check support func.", rd->op_type.GetString());
    funcs.check_support = rd->functions.check_support;
  }
  if (rd->functions.op_select_format != nullptr) {
    GELOGD("Op type:%s reg op select format func.", rd->op_type.GetString());
    funcs.op_select_format = rd->functions.op_select_format;
  }
  if (rd->functions.get_op_support_info != nullptr) {
    GELOGD("Op type:%s reg get op support info func.", rd->op_type.GetString());
    funcs.get_op_support_info = rd->functions.get_op_support_info;
  }
  if (rd->functions.get_op_specific_info != nullptr) {
    GELOGD("Op type:%s reg get op specific info func.", rd->op_type.GetString());
    funcs.get_op_specific_info = rd->functions.get_op_specific_info;
  }
}
}

OpCtImplRegistry &OpCtImplRegistry::GetInstance() {
  static OpCtImplRegistry instance;
  return instance;
}

OpCtImplRegistry::OpCtImplFunctions &OpCtImplRegistry::CreateOrGetOpImpl(const ge::char_t *op_type) {
  (void)reserved_;
  return types_to_impl_[op_type];
}
const OpCtImplRegistry::OpCtImplFunctions *OpCtImplRegistry::GetOpImpl(const ge::char_t *op_type) const {
  const auto iter = types_to_impl_.find(op_type);
  if (iter == types_to_impl_.end()) {
    return nullptr;
  }
  return &iter->second;
}
const std::map<OpCtImplRegistry::OpType, OpCtImplRegistry::OpCtImplFunctions> &OpCtImplRegistry::GetAllTypesToImpl() const {
  return types_to_impl_;
}
std::map<OpCtImplRegistry::OpType, OpCtImplRegistry::OpCtImplFunctions> &OpCtImplRegistry::GetAllTypesToImpl() {
  return types_to_impl_;
}

OpCtImplRegisterV2::OpCtImplRegisterV2(const ge::char_t *op_type) : impl_(new(std::nothrow) OpCtImplRegisterV2Impl) {
  if (impl_ == nullptr) {
    return;
  }
  GELOGD("Op type: %s", op_type);
  impl_->op_type = op_type;
  impl_->functions.calc_op_param = nullptr;
  impl_->functions.gen_task = nullptr;
  impl_->functions.check_support = nullptr;
  impl_->functions.op_select_format = nullptr;
  impl_->functions.get_op_support_info = nullptr;
  impl_->functions.get_op_specific_info = nullptr;

  // private attr controlled by is_private_attr_registered
  (void)OpCtImplRegistry::GetInstance().CreateOrGetOpImpl(op_type);
}

OpCtImplRegisterV2::~OpCtImplRegisterV2() = default;

OpCtImplRegisterV2::OpCtImplRegisterV2(const OpCtImplRegisterV2 &register_data) {
  RegisterOpImplToRegistry(register_data.impl_.get());
}

OpCtImplRegisterV2::OpCtImplRegisterV2(OpCtImplRegisterV2 &&register_data) noexcept {
  RegisterOpImplToRegistry(register_data.impl_.get());
}

OpCtImplRegisterV2 &OpCtImplRegisterV2::CalcOpParam(OpCtImplKernelRegistry::OpCalcParamKernelFunc calc_op_param_func) {
  if (impl_ != nullptr) {
    GELOGD("Reg calc func.");
    impl_->functions.calc_op_param = calc_op_param_func;
  }
  return *this;
}

OpCtImplRegisterV2 &OpCtImplRegisterV2::GenerateTask(OpCtImplKernelRegistry::OpGenTaskKernelFunc gen_task_func) {
  if (impl_ != nullptr) {
    GELOGD("Reg gen task func.");
    impl_->functions.gen_task = gen_task_func;
  }
  return *this;
}

OpCtImplRegisterV2 &OpCtImplRegisterV2::CheckSupport(OpCtImplKernelRegistry::OP_CHECK_FUNC_V2 check_support_func) {
  if (impl_ != nullptr) {
    GELOGD("Reg check support func.");
    impl_->functions.check_support = check_support_func;
  }
  return *this;
}
OpCtImplRegisterV2 &OpCtImplRegisterV2::OpSelectFormat(OpCtImplKernelRegistry::OP_CHECK_FUNC_V2 op_select_format_func) {
  if (impl_ != nullptr) {
    GELOGD("Reg op select format func.");
    impl_->functions.op_select_format = op_select_format_func;
  }
  return *this;
}
OpCtImplRegisterV2 &OpCtImplRegisterV2::GetOpSupportInfo(OpCtImplKernelRegistry::OP_CHECK_FUNC_V2 get_op_support_info_func) {
  if (impl_ != nullptr) {
    GELOGD("Reg op support info func.");
    impl_->functions.get_op_support_info = get_op_support_info_func;
  }
  return *this;
}
OpCtImplRegisterV2 &OpCtImplRegisterV2::GetOpSpecificInfo(OpCtImplKernelRegistry::OP_CHECK_FUNC_V2 get_op_specific_info_func) {
  if (impl_ != nullptr) {
    GELOGD("Reg op specific info func.");
    impl_->functions.get_op_specific_info = get_op_specific_info_func;
  }
  return *this;
}
}  // namespace gert

#ifdef __cplusplus
extern "C" {
#endif

size_t GetRegisteredOpCtNum(void) {
  return gert::OpCtImplRegistry::GetInstance().GetAllTypesToImpl().size();
}
int32_t GetOpCtImplFunctions(TypesToCtImpl *impl, size_t impl_num) {
  const auto types_to_impl = gert::OpCtImplRegistry::GetInstance().GetAllTypesToImpl();
  if (impl_num != types_to_impl.size()) {
    GELOGE(ge::FAILED, "Get types_to_impl_ failed, impl_num[%zu] and map size[%zu] not match",
           impl_num, types_to_impl.size());
    return static_cast<int32_t>(ge::GRAPH_FAILED);
  }
  const auto first_st = types_to_impl.cbegin();
  const size_t real_size = impl[0].funcs.st_size;
  const size_t op_size = first_st->second.st_size;
  GELOGD("Cann version[%d]/size[%zu] with opp version[%d]/size[%zu].", impl[0].funcs.version, real_size,
         first_st->second.version, op_size);
  if (real_size != op_size) {
    const size_t real_offset = real_size + sizeof(char*);
    const size_t copy_size = std::min(real_size, op_size);
    uint8_t *real_impl_base = reinterpret_cast<uint8_t *>(impl);
    size_t i = 0;
    for (auto &it : types_to_impl) {
      auto cur_impl = reinterpret_cast<TypesToCtImpl *>(real_impl_base + i * real_offset);
      cur_impl->op_type = it.first.GetString();
      const auto mem_ret = memcpy_s(&cur_impl->funcs, copy_size, &it.second, copy_size);
      if (mem_ret != EOK) {
        GELOGE(ge::FAILED, "Copy func failed with size:%zu.", copy_size);
        return static_cast<int32_t>(ge::GRAPH_FAILED);
      }
      ++i;
    }
    return static_cast<int32_t>(ge::GRAPH_SUCCESS);
  }
  size_t cnt = 0U;
  for (auto &it : types_to_impl) {
    impl[cnt].op_type = it.first.GetString();
    impl[cnt].funcs = it.second;
    cnt++;
  }
  return static_cast<int32_t>(ge::GRAPH_SUCCESS);
}
#ifdef __cplusplus
}
#endif
