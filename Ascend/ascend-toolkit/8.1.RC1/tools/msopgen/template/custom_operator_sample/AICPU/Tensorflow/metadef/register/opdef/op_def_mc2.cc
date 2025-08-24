/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * [graph-engine] is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include <algorithm>
#include "op_def_impl.h"

namespace ops {
OpMC2Def::OpMC2Def() : impl_(new(std::nothrow) OpMC2DefImpl) {}

OpMC2Def::OpMC2Def(const OpMC2Def &mc2_def) : impl_(new(std::nothrow) OpMC2DefImpl) {
  this->impl_->group_list = mc2_def.impl_->group_list;
}

OpMC2Def::~OpMC2Def() = default;

OpMC2Def &OpMC2Def::operator=(const OpMC2Def &mc2_def) {
  if (this != &mc2_def) {
    *this->impl_ = *mc2_def.impl_;
  }
  return *this;
}

OpMC2Def &OpMC2Def::HcclGroup(const char *value) {
  if (std::find(this->impl_->group_list.begin(), this->impl_->group_list.end(), value) ==
      this->impl_->group_list.end()) {
    this->impl_->group_list.emplace_back(value);
  }
  return *this;
}

OpMC2Def &OpMC2Def::HcclGroup(std::vector<const char *> value) {
  for (const char *val : value) {
    if (std::find(this->impl_->group_list.begin(), this->impl_->group_list.end(), val) ==
        this->impl_->group_list.end()) {
      this->impl_->group_list.emplace_back(val);
    }
  }
  return *this;
}

std::vector<ge::AscendString> &OpMC2Def::GetHcclGroups(void) const {
  return this->impl_->group_list;
}

}  // namespace ops
