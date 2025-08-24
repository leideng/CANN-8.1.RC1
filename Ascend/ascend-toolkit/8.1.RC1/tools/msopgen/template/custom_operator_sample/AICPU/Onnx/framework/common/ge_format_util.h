/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_FRAMEWORK_COMMON_GE_FORMAT_UTIL_H_
#define INC_FRAMEWORK_COMMON_GE_FORMAT_UTIL_H_

#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "graph/tensor.h"

namespace ge {
class GE_FUNC_VISIBILITY GeFormatUtil {
 public:
  ///
  /// @name   TransShape
  /// @brief  transform the shape of tensor according to destination format
  /// @param  [in] src_desc       source tensor desc
  /// @param  [in] dst_format     destination format
  /// @param  [out] dst_shape     destination shape
  /// @return Status
  ///
  static Status TransShape(const TensorDesc &src_desc, const Format dst_format, std::vector<int64_t> &dst_shape);
};
}  // namespace ge

#endif  // INC_FRAMEWORK_COMMON_GE_FORMAT_UTIL_H_
