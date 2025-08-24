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

/*!
 * \file kernel_operator_list_tensor_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_LIST_TENSOR_IMPL_H
#define ASCENDC_MODULE_OPERATOR_LIST_TENSOR_IMPL_H

#include "kernel_operator.h"

namespace AscendC {
template<bool Sequential = true> class ListTensorDescImpl {
public:
    __aicore__ inline ListTensorDescImpl();
    __aicore__ inline void ListTensorDecode(__gm__ void* data, uint32_t length, uint32_t shapeSize);
    template<class T> __aicore__ inline __gm__ T* GetDataPtr(uint32_t index);
    __aicore__ inline uint32_t GetSize();
private:
    __aicore__ inline void GetShape(uint64_t* shapeInfo, __gm__ uint64_t* target);

    __gm__ uint64_t* dataPtr_ = nullptr;
    __gm__ uint64_t* shapeStart_ = nullptr;
    uint32_t dim_ = 0;
    uint32_t descStructSize_ = 0;
    uint32_t curOffset_ = 0;
    uint32_t count_ = 0;
    int64_t preloadOffset = 0;
};

template<bool Sequential>
__aicore__ inline ListTensorDescImpl<Sequential>::ListTensorDescImpl() {}

template<bool Sequential>
__aicore__ inline void ListTensorDescImpl<Sequential>::
ListTensorDecode(__gm__ void* data, uint32_t length, uint32_t shapeSize)
{
    ASSERT(data != nullptr);
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t *>(data);
    uint64_t dataPtrOffset = *dataAddr;  // 数据地址相对于首地址的偏移量

    shapeStart_ = dataAddr + 1;  // 指向存放shape的头指针
    dataPtr_ = dataAddr + (dataPtrOffset >> 3);  // 右移 3 位表示除以sizeof(uint64_t)

    dim_ = *shapeStart_ & 0xffffffff;  // 0xffffffff 表示取低32位
    uint32_t offset = *shapeStart_ >> 32;  // 右移 32 位表示取高32位

    // dim_ = 0时表示输入数据仅为数据指针, shape使用一个0xffffffff填充, 因此descStructSize_为 2
    descStructSize_ = (dim_ == 0) ? 2 : 1 + dim_;
    curOffset_ = 0;  // 当前shape相对于shape头指针的偏移量
    count_ = (offset != 0) ? offset : (*(dataPtr_ - descStructSize_) >> 32) + 1;  // 右移 32 位表示取高32位
    // dim != 0 时, 不支持相同shape压缩储存
    // 数据地址相对于首地址的偏移量为固定值(2 + 1) * sizeof(uint64_t) = 24
    ASSERT(((dim_ != 0) && (count_ == ((dataPtrOffset - sizeof(uint64_t)) / (descStructSize_ * sizeof(uint64_t))))) ||
           ((dim_ == 0) && (*(shapeStart_ + 1) == 0xffffffff) && (dataPtrOffset == 24)));

    if ((length != 0xffffffff) && (shapeSize != 0xffffffff)) {
        ASSERT(count_ == shapeSize);
        ASSERT(length == (dataPtrOffset + (count_ << 3)));  // 左移 3 位表示乘sizeof(uint64_t)
    }
}

template<bool Sequential> template<class T>
__aicore__ inline __gm__ T* ListTensorDescImpl<Sequential>::GetDataPtr(uint32_t index)
{
    ASSERT(index < count_);
    return reinterpret_cast<__gm__ T *>(*(dataPtr_ + index));
}

template<bool Sequential>
__aicore__ inline uint32_t ListTensorDescImpl<Sequential>::GetSize()
{
    return count_;
}

} // namespace AscendC

#endif // ASCENDC_MODULE_OPERATOR_LIST_TENSOR_IMPL_H