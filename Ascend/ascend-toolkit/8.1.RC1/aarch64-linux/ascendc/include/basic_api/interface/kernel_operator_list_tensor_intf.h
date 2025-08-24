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
 * \file kernel_operator_list_tensor_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_LIST_TENSOR_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_LIST_TENSOR_INTERFACE_H

#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif

#if __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_list_tensor_impl.h"
#elif (__CCE_AICORE__ == 200)
#include "dav_m200/kernel_operator_list_tensor_impl.h"
#elif (__CCE_AICORE__ == 100)
#include "dav_c100/kernel_operator_list_tensor_impl.h"
#endif

namespace AscendC {
#if __CCE_AICORE__ >= 200
class ListTensorDesc;
template<class T> class TensorDesc {
public:
    __aicore__ inline TensorDesc() {}
    __aicore__ inline ~TensorDesc() {}
    __aicore__ inline void SetShapeAddr(uint64_t* shapePtr)
    {
        tensorDesc.SetShapeAddr(shapePtr);
    }

    __aicore__ inline uint64_t GetDim()
    {
        return tensorDesc.GetDim();
    }
    __aicore__ inline uint64_t GetIndex()
    {
        return tensorDesc.GetIndex();
    }
    __aicore__ inline uint64_t GetShape(uint32_t offset)
    {
        return tensorDesc.GetShape(offset);
    }
    __aicore__ inline __gm__ T* GetDataPtr()
    {
        return tensorDesc.GetDataPtr();
    }

    __aicore__ inline GlobalTensor<T> GetDataObj()
    {
        return tensorDesc.GetDataObj();
    }

    friend class ListTensorDesc;
private:
    TensorDescImpl<T> tensorDesc;
};
#endif

class ListTensorDesc {
public:
    __aicore__ inline ListTensorDesc() {}
    __aicore__ inline ~ListTensorDesc() {}
    __aicore__ inline ListTensorDesc(__gm__ void* data, uint32_t length = 0xffffffff, uint32_t shapeSize = 0xffffffff)
    {
        listTensorDesc.ListTensorDecode(data, length, shapeSize);
    }
    __aicore__ inline void Init(__gm__ void* data, uint32_t length = 0xffffffff, uint32_t shapeSize = 0xffffffff)
    {
        listTensorDesc.ListTensorDecode(data, length, shapeSize);
    }
#if __CCE_AICORE__ >= 200
    template<class T> __aicore__ inline void GetDesc(TensorDesc<T>& desc, uint32_t index)
    {
        listTensorDesc.GetDesc(&desc.tensorDesc, index);
    }
#endif
    template<class T> __aicore__ inline __gm__ T* GetDataPtr(uint32_t index)
    {
        return listTensorDesc.GetDataPtr<T>(index);
    }
    __aicore__ inline uint32_t GetSize()
    {
        return listTensorDesc.GetSize();
    }
private:
#ifdef __CCE_AICORE__
    ListTensorDescImpl<true> listTensorDesc;
#endif
};
}
#endif // ASCENDC_MODULE_OPERATOR_LIST_TENSOR_INTERFACE_H