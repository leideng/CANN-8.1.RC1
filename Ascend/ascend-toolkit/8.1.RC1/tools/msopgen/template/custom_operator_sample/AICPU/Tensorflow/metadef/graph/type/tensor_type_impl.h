/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef METADEF_CXX_TENSOR_TYPE_IMPL_H
#define METADEF_CXX_TENSOR_TYPE_IMPL_H
#include <set>
#include "graph/types.h"
namespace ge {
class TensorTypeImpl {
 public:
  TensorTypeImpl() = default;
  ~TensorTypeImpl() = default;

  std::set<DataType> &GetMutableDateTypeSet() {
      return dt_set_;
  }
  bool IsDataTypeInRange(const DataType data_type) const {
    return (dt_set_.count(data_type) > 0);
  }
 private:
  std::set<DataType> dt_set_;
};
}  // namespace ge

#endif  // METADEF_CXX_TENSOR_TYPE_IMPL_H
