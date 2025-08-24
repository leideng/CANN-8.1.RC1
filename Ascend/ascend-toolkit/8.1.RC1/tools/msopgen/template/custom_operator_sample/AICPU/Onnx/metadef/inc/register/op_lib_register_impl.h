/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef METADEF_CXX_REGISTER_OP_LIB_REGISTER_IMPL_H_
#define METADEF_CXX_REGISTER_OP_LIB_REGISTER_IMPL_H_

#include <mutex>
#include <set>
#include <string>
#include "register/op_lib_register.h"
#include "graph/ge_error_codes.h"

namespace ge {
class OpLibRegisterImpl {
 public:
  std::string &MutableVendorName() { return vendor_name_; }
  OpLibRegister::OpLibInitFunc &MutableInitFunc() { return init_func_; }

 private:
  std::string vendor_name_;
  OpLibRegister::OpLibInitFunc init_func_ = nullptr;
};

class OpLibRegistry {
 public:
  static OpLibRegistry &GetInstance();
  ~OpLibRegistry();
  void RegisterInitFunc(OpLibRegisterImpl &register_impl);
  graphStatus PreProcessForCustomOp();
  const char_t* GetCustomOpLibPath();

 private:
  void ClearHandles();
  graphStatus GetAllCustomOpApiSoPaths(const std::string &custom_opp_path,
                                       std::vector<std::string> &so_real_paths) const;
  graphStatus CallInitFunc(const std::string &custom_opp_path,
                           const std::vector<std::string> &so_real_paths);

  std::mutex mu_;
  std::vector<std::pair<std::string, OpLibRegister::OpLibInitFunc>> vendor_funcs_;
  std::set<std::string> vendor_names_set_;
  std::vector<void *> handles_;
  bool is_processed_ = false;
  std::string op_lib_paths_;
};
}  // namespace ge
#endif  // METADEF_CXX_REGISTER_OP_LIB_REGISTER_IMPL_H_
