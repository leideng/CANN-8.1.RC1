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
 * \file feeds_repeat.cpp
 * \brief
 */
#include "feeds_repeat.h"
using namespace AscendC;
using namespace FeedsRepeat;
extern "C" __global__ __aicore__ void feeds_repeat(GM_ADDR feeds, GM_ADDR feeds_repeat_times, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    if(TILING_KEY_IS(1)){ //<fp32, int32>
        FeedsRepeatND<float, int32_t> op;
        op.Init(feeds, feeds_repeat_times, y, &tiling_data);
        op.Process();
    }
    else if(TILING_KEY_IS(2)){ //<fp16, int32>
        FeedsRepeatND<half, int32_t> op;
        op.Init(feeds, feeds_repeat_times, y, &tiling_data);
        op.Process();
    }
    else if(TILING_KEY_IS(3)){ //<bf16, int32>
        FeedsRepeatND<bfloat16_t, int32_t> op;
        op.Init(feeds, feeds_repeat_times, y, &tiling_data);
        op.Process();
    }
    else if(TILING_KEY_IS(101)){ //<fp32, int64>
        FeedsRepeatND<float, int64_t> op;
        op.Init(feeds, feeds_repeat_times, y, &tiling_data);
        op.Process();
    }
    else if(TILING_KEY_IS(102)){ //<fp16, int64>
        FeedsRepeatND<half, int64_t> op;
        op.Init(feeds, feeds_repeat_times, y, &tiling_data);
        op.Process();
    }
    else if(TILING_KEY_IS(103)){ //<bf16, int64>
        FeedsRepeatND<bfloat16_t, int64_t> op;
        op.Init(feeds, feeds_repeat_times, y, &tiling_data);
        op.Process();
    }
}

