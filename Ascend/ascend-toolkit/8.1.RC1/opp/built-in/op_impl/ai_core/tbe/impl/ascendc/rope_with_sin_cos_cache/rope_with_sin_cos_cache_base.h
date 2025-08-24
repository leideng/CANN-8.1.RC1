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
 * \file rope_with_sin_cos_cache_base.h
 * \brief rope_with_sin_cos_cache_base.h
 */
#ifndef ROPE_WITH_SIN_COS_CACHE_BASE_H
#define ROPE_WITH_SIN_COS_CACHE_BASE_H

#include "kernel_operator.h"

namespace RopeWithSinCosCache {
using namespace AscendC;
using AscendC::HardEvent;
using AscendC::Duplicate;

template <typename T>
class RopeWithSinCosCacheBase {
public:
    //构造函数
    __aicore__ inline RopeWithSinCosCacheBase(){};
    __aicore__ inline void InitData(const RopeWithSinCosCacheTilingData& tilingData);
    __aicore__ inline void SToMTE2Sync();
    __aicore__ inline void MTE2ToSSync();
    __aicore__ inline void SToMTE3Sync();
    __aicore__ inline void MTE3ToSSync();
    __aicore__ inline void SToVSync();
    __aicore__ inline void MTE3ToVSync();

protected:
    uint32_t blockIdx_;
    uint64_t core_num_use;
    uint64_t num_tokens;
    uint64_t num_q_heads;
    uint64_t num_kv_heads;
    uint64_t rotary_dim;
    uint64_t mrope_section0;
    uint64_t mrope_section1;
    uint64_t mrope_section2;
    uint64_t head_size;
    uint64_t q_leading_dimension;
    uint64_t k_leading_dimension;
    uint64_t front_core;
    uint64_t tail_core;
    uint64_t num_tokens_each_front_core;
    uint64_t num_tokens_each_tail_core;
    uint64_t is_neox_style;

    uint64_t loop_time_current_core{0};   // 当前核批处理数据轮数
    uint64_t num_tokens_each_loop_current_core{0};  // 当前核每轮处理的token数
    uint64_t num_tokens_last_loop_current_core{0};  // 当前核最后一轮轮处理的token数
};

template <typename T>
__aicore__ inline void RopeWithSinCosCacheBase<T>::InitData(const RopeWithSinCosCacheTilingData& tilingData){
    blockIdx_ = AscendC::GetBlockIdx();

    core_num_use = tilingData.core_num_use;
    num_tokens = tilingData.num_tokens;
    num_q_heads = tilingData.num_q_heads;
    num_kv_heads = tilingData.num_kv_heads;
    rotary_dim = tilingData.rotary_dim;
    mrope_section0 = tilingData.mrope_section0;
    mrope_section1 = tilingData.mrope_section1;
    mrope_section2 = tilingData.mrope_section2;
    head_size = tilingData.head_size;
    q_leading_dimension = tilingData.q_leading_dimension;
    k_leading_dimension = tilingData.k_leading_dimension;
    front_core = tilingData.front_core;
    tail_core = tilingData.tail_core;
    num_tokens_each_front_core = tilingData.num_tokens_each_front_core;
    num_tokens_each_tail_core = tilingData.num_tokens_each_tail_core;
    is_neox_style = tilingData.isNeoxStyle;

    loop_time_current_core = (blockIdx_ < front_core) ? tilingData.loop_time_each_front_core : tilingData.loop_time_each_tail_core;
    num_tokens_each_loop_current_core = (blockIdx_ < front_core) ? tilingData.num_tokens_front_core_each_loop : tilingData.num_tokens_tail_core_each_loop;
    num_tokens_last_loop_current_core = (blockIdx_ < front_core) ? tilingData.num_tokens_front_core_last_loop : tilingData.num_tokens_tail_core_last_loop;
}

template <typename T>
__aicore__ inline void RopeWithSinCosCacheBase<T>::SToMTE2Sync() {
    event_t eventIDSToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    SetFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
    WaitFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
}

template <typename T>
__aicore__ inline void RopeWithSinCosCacheBase<T>::MTE2ToSSync() {
    event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
}

template <typename T>
__aicore__ inline void RopeWithSinCosCacheBase<T>::SToMTE3Sync() {
    event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
}

template <typename T>
__aicore__ inline void RopeWithSinCosCacheBase<T>::MTE3ToSSync() {
    event_t eventIDMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
    WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
}

template <typename T>
__aicore__ inline void RopeWithSinCosCacheBase<T>::SToVSync() {
    event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIDSToV);
    WaitFlag<HardEvent::S_V>(eventIDSToV);
}

template <typename T>
__aicore__ inline void RopeWithSinCosCacheBase<T>::MTE3ToVSync() {
    event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
}

} 

#endif  //namespace RopeWithSinCosCache