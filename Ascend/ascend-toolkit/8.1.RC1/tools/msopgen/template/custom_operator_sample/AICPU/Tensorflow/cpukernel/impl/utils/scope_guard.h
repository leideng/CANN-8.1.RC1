/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CPU_KERNEL_UTIL_SCOPE_GUARD_H_
#define CPU_KERNEL_UTIL_SCOPE_GUARD_H_

#include <functional>

namespace aicpu {
  class ScopeGuard {
    public:
      explicit ScopeGuard(const std::function<void()> exitScope) : exitScope_(exitScope) {}
      ~ScopeGuard() {
        if (exitScope_ == nullptr) {
          return;
        }

        try {
          exitScope_();
        } catch (...) {
          // pass
        }
      }

    private:
      ScopeGuard(const ScopeGuard&) =  delete;
      ScopeGuard(ScopeGuard&&) = delete;
      ScopeGuard& operator=(const ScopeGuard&) = delete;
      ScopeGuard& operator=(ScopeGuard&&) = delete;

      std::function<void()> exitScope_;
  };
}

#endif