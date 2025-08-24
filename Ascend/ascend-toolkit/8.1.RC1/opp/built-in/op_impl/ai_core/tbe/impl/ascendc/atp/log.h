/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

/*!
 * \file log.h
 * \brief
 */
#ifndef ATP_LOG_H_
#define ATP_LOG_H_

#define DO_JOIN_SYMBOL(symbol1, symbol2) symbol1##symbol2
#define JOIN_SYMBOL(symbol1, symbol2) DO_JOIN_SYMBOL(symbol1, symbol2)

#ifdef __COUNTER__
#define UNIQUE_ID __COUNTER__
#else
#define UNIQUE_ID __LINE__
#endif

#define UNIQUE_NAME(prefix) JOIN_SYMBOL(prefix, UNIQUE_ID)

#if defined __CCE_KT_TEST__ || defined __ATP_UT__
#ifdef __CCE_KT_TEST__
#define RUN_LOG_BASE(...)   \
  if (GetBlockIdx() == 0) { \
    printf(__VA_ARGS__);    \
  }
#else
#define RUN_LOG_BASE(...)   \
  do { \
    printf(__VA_ARGS__);    \
  } while(0)
#endif

#include <cxxabi.h>
#include <securec.h>
#define PRINT_TYPE(T) abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, nullptr)

#define RUN_LOG_ONE_BLOCK(...)  \
  do {                                                                     \
    const char* filename = strrchr(__FILE__, '/');                         \
    if (!filename) filename = strrchr(__FILE__, '\\');                     \
    filename = filename ? filename + 1 : __FILE__;                         \
    if constexpr (sizeof(#__VA_ARGS__) <= 1) {                             \
      printf("[INFO][Core0:%s:%d] (empty log)\n", filename, __LINE__);     \
    } else {                                                               \
      char buffer[1024];                                                   \
      snprintf_s(buffer, sizeof(buffer), sizeof(buffer) - 1, __VA_ARGS__); \
      if (buffer[strlen(buffer) - 1] != '\n') {                            \
        printf("[INFO][Core0:%s:%d] %s\n", filename, __LINE__, buffer);    \
      } else {                                                             \
        printf("[INFO][Core0:%s:%d] %s", filename, __LINE__, buffer);      \
      }                                                                    \
    }                                                                      \
  } while (0)

#else
#define RUN_LOG_BASE(...)
#define PRINT_TYPE(T)
#define RUN_LOG_ONE_BLOCK(...)
#endif

/////////////////////////////////////////////////////////////////////////////

#ifdef __CCE_KT_TEST__
// CPU孪生调试时支持打印
#define RUN_LOG(...)                                                         \
  if (GetBlockIdx() == 0) {                                                  \
    RUN_LOG_ONE_BLOCK(__VA_ARGS__);                                          \
  }

template <typename T, T v>
struct Print2 {
  constexpr operator char() { return 1 + 0xFF; }
};
#define BUILD_LOG(...) char UNIQUE_NAME(print_value_) = Print2<__VA_ARGS__>()

#else  // __CCE_KT_TEST__
#ifdef __ATP_UT__
#define RUN_LOG(...)   RUN_LOG_ONE_BLOCK(__VA_ARGS__)
#else  // __ATP_UT__
// 实际编译Kernel时不打印
#define RUN_LOG(...)
#endif // __ATP_UT__
#define BUILD_LOG(...)
#endif  // __CCE_KT_TEST__

#endif  // ATP_LOG_H_
