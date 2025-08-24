/* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
 * \file kernel_tquesync_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_TQUESYNC_IMPL_H
#define ASCENDC_MODULE_TQUESYNC_IMPL_H

namespace AscendC {
template <pipe_t src, pipe_t dst>
__aicore__ inline void TQueSync<src, dst>::SetFlag(TEventID id)
{
    static_assert((src != dst), "src/dst pipe cannot be same.");
    static_assert(IsSupportedPipe(src), "src pipe not supported");
    static_assert(IsSupportedPipe(dst), "dst pipe not supported");
    ASCENDC_ASSERT((id < QUE_MAX_EVENT), {
        KERNEL_LOG(KERNEL_ERROR, "event id input is %d, which should be less than %d", id, QUE_MAX_EVENT);
    });
    set_flag(src, dst, id);
}

template <pipe_t src, pipe_t dst>
__aicore__ inline void TQueSync<src, dst>::WaitFlag(TEventID id)
{
    static_assert((src != dst), "src/dst pipe cannot be same.");
    static_assert(IsSupportedPipe(src), "src pipe not supported");
    static_assert(IsSupportedPipe(dst), "dst pipe not supported");
    ASCENDC_ASSERT((id < QUE_MAX_EVENT), {
        KERNEL_LOG(KERNEL_ERROR, "event id input is %d, which should be less than %d", id, QUE_MAX_EVENT);
    });
    wait_flag(src, dst, id);
}
}
#endif // ASCENDC_MODULE_TQUESYNC_IMPL_H