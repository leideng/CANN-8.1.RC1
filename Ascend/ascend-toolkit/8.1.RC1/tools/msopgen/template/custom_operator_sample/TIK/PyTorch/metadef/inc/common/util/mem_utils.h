/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef COMMON_GRAPH_UTILS_MEM_UTILS_H_
#define COMMON_GRAPH_UTILS_MEM_UTILS_H_

#include <memory>
#include <utility>

namespace ge {
template <typename _Tp, typename... _Args>
static inline std::shared_ptr<_Tp> MakeShared(_Args &&... __args) {
  using _Tp_nc = typename std::remove_const<_Tp>::type;
  const std::shared_ptr<_Tp> ret(new (std::nothrow) _Tp_nc(std::forward<_Args>(__args)...));
  return ret;
}
}

#endif  // COMMON_GRAPH_UTILS_MEM_UTILS_H_
