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
 * \file elems.h
 * \brief
 */
#ifndef ATP_ELEMS_H_
#define ATP_ELEMS_H_

#ifndef __CCE_AICORE__
#define __aicore__
#endif

#include "aux1.h"
namespace AscendC {
template <typename... Ts>
struct Elems {
  using Type = Elems<Ts...>;
  constexpr static size_t Size = sizeof...(Ts);

  template <typename... T>
  using Append = Elems<Ts..., T...>;

  template <template <class...> class T>
  using Export = T<Ts...>;

  template <typename Es>
  using Concat = typename Es::template Export<Append>;

  template <int pos>
  using At = __aux::GetElemAt<pos, 0, Ts...>;

  template <typename T, int start = 0>
  constexpr static bool IsExist() {
    if constexpr (start < Size) {
      if constexpr (__aux::IsSameType<T, At<start>>::Value) {
        return true;
      }
      return IsExist<T, start + 1>();
    }
    return false;
  }

  template <typename T, int start = 0>
  __aicore__ constexpr static int GetIndex() {
    if constexpr (__aux::IsSameType<T, At<start>>::Value) {
      return start;
    } else if constexpr (start + 1 < Size) {
      return GetIndex<T, start + 1>();
    }
    return -1;
  }

  template <template <class, class> class Check, typename Filtered = Elems<>>
  using Filter = typename __aux::FilterAux<Elems<Ts...>, Check, Filtered>::Type;

  using Unique = __aux::UniqAuxT<Elems<Ts...>>;

  template <class Es>
  using Union = typename Concat<Es>::Unique;

 private:
  template <class Es>
  struct _RemoveAux {
    template <typename E, typename T>
    struct Check {
      constexpr static bool Value = !Es::template IsExist<T>();
    };

    using Type = typename __aux::FilterAux<Elems<Ts...>, Check>::Type;
  };

 public:
  template <class Es>
  using Remove = typename _RemoveAux<Es>::Type;

  template <template <class...> class F>
  using ForEach = Elems<typename F<Ts>::Type...>;

  template <typename ToPop>
  using PopFront = typename __aux::PopFrontAux<Elems<Ts...>, ToPop>::Type;
};
}  // namespace AscendC

#endif  // ATP_ELEMS_H_
