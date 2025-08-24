/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef D_INC_GRAPH_LIST_NODE_H
#define D_INC_GRAPH_LIST_NODE_H

namespace ge {
enum class ListMode { kWorkMode = 0, kFreeMode };

template <class T>
class QuickList;

template <class T>
struct ListElement {
  ListElement<T> *next;
  ListElement<T> *prev;
  QuickList<T> *owner;
  ListMode mode;
  T data;
  explicit ListElement(const T &x) : data(x), next(nullptr), prev(nullptr), owner(nullptr), mode(ListMode::kFreeMode) {}
  bool operator==(const ListElement<T> &r_ListElement) const {
    return data == r_ListElement.data;
  }
  ListElement() : next(nullptr), prev(nullptr), owner(nullptr), mode(ListMode::kFreeMode) {}
  void SetOwner(QuickList<T> *new_owner) {
    owner = new_owner;
  }
};
}  // namespace ge
#endif
