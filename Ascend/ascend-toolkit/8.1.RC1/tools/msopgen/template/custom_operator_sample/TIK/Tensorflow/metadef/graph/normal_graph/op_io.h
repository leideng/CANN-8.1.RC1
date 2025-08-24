/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef METADEF_CXX_OP_IO_H
#define METADEF_CXX_OP_IO_H
namespace ge {

class OpIO {
 public:
  OpIO(const std::string &name, const int32_t index, const OperatorImplPtr &owner)
      : name_(name), index_(index), owner_(owner) {}

  ~OpIO() = default;

  std::string GetName() const { return name_; }

  int32_t GetIndex() const { return index_; }

  OperatorImplPtr GetOwner() const { return owner_; }

 private:
  std::string name_;
  int32_t index_;
  std::shared_ptr<OperatorImpl> owner_;
};
}
#endif  // METADEF_CXX_OP_IO_H
