/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef EXECUTE_GRAPH_OBJECT_POOL_H
#define EXECUTE_GRAPH_OBJECT_POOL_H

#include <memory>
#include <queue>
namespace ge {
constexpr size_t kDefaultPoolSize = 100UL;

template<class T, size_t N = kDefaultPoolSize>
class ObjectPool {
 public:
  ObjectPool() = default;
  ~ObjectPool() = default;
  ObjectPool(ObjectPool &) = delete;
  ObjectPool(ObjectPool &&) = delete;
  ObjectPool &operator=(const ObjectPool &) = delete;
  ObjectPool &operator=(ObjectPool &&) = delete;

  template<typename... Args>
  std::unique_ptr<T> Acquire(Args &&...args) {
    if (!handlers_.empty()) {
      std::unique_ptr<T> tmp(std::move(handlers_.front()));
      handlers_.pop();
      return tmp;
    }
    return std::unique_ptr<T>(new (std::nothrow) T(args...));
  }

  void Release(std::unique_ptr<T> ptr) {
    if ((handlers_.size() < N) && (ptr != nullptr)) {
      handlers_.push(std::move(ptr));
    }
  }

  bool IsEmpty() const {
    return handlers_.empty();
  }

  bool IsFull() const {
    return handlers_.size() >= N;
  }

 private:
  std::queue<std::unique_ptr<T>> handlers_;
};
}  // namespace ge
#endif  // EXECUTE_GRAPH_OBJECT_POOL_H
