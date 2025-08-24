/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
 * \file apply_rotary_pos_emb_base.h
 * \brief
 */
#ifndef APPLY_ROTARY_POS_EMB_BASE_H
#define APPLY_ROTARY_POS_EMB_BASE_H

#include "kernel_operator.h"
#include "../inc/platform.h"

namespace ApplyRotaryPosEmb {
using namespace AscendC;

template <typename T>
class ApplyRotaryPosEmbBase {
public:
    __aicore__ inline ApplyRotaryPosEmbBase(){};

protected:
    template <typename T1, typename T2>
    __aicore__ inline void LocalTensor2NewTensor(LocalTensor<T1> &tensor_new, const LocalTensor<T2> &tensor_old)
    {
        tensor_new = tensor_old.template ReinterpretCast<T1>();
    };
};
}  // namespace ApplyRotaryPosEmb

#endif  // APPLY_ROTARY_POS_EMB_BASE_H
