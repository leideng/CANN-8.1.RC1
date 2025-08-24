/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
 * \file dynamic_rnn_v2_910b.cpp
 * \brief
 */
#include "../dynamic_r_n_n/LstmFP16.cpp"
#include "../dynamic_r_n_n/LstmFP32.cpp"

extern "C" __global__ __aicore__ void dynamic_rnn_v2(GM_ADDR inputX, GM_ADDR weightInput, GM_ADDR weightHidden,
                                                  GM_ADDR bias, GM_ADDR seqLength,
                                                  GM_ADDR initH, GM_ADDR initC, GM_ADDR wCi, GM_ADDR wCf, GM_ADDR wCo,
                                                  GM_ADDR mask, GM_ADDR outputY, GM_ADDR outputH, GM_ADDR outputC,
                                                  GM_ADDR outputI, GM_ADDR outputJ, GM_ADDR outputF, GM_ADDR outputO,
                                                  GM_ADDR outputTanhC, GM_ADDR workspace, GM_ADDR rnnTiling) {
  set_mask_norm();
  GET_TILING_DATA(tiling_data, rnnTiling);
  const DynamicRNNTilingData* __restrict tilingData = &tiling_data;
  const TCubeTiling* __restrict inputMMTiling = &(tilingData->inputMMParam);
  const TCubeTiling* __restrict hiddenMMTiling = &(tilingData->hiddenMMParam);

  if (TILING_KEY_IS(10000001)) {
    LstmMmSplitNDNDFP16<half> lstmOp;
    REGIST_MATMUL_OBJ(&lstmOp.pipe, GetSysWorkSpacePtr(), lstmOp.inputMM, inputMMTiling,
                       lstmOp.hiddenMM, hiddenMMTiling);

    lstmOp.InitV2(inputX, weightInput, weightHidden, bias, seqLength, initH, initC, wCi, wCf, wCo, mask,
                  outputY, outputH, outputC, outputI, outputJ, outputF, outputO, outputTanhC, &tiling_data, workspace);
    lstmOp.Process();
  } else if (TILING_KEY_IS(10000002)) {
    LstmMmSplitNDNDFP32<float> lstmOp;
    REGIST_MATMUL_OBJ(&lstmOp.pipe, GetSysWorkSpacePtr(), lstmOp.inputMM, inputMMTiling,
                      lstmOp.hiddenMM, hiddenMMTiling);

    lstmOp.InitV2(inputX, weightInput, weightHidden, bias, seqLength, initH, initC, wCi, wCf, wCo, mask,
                  outputY, outputH, outputC, outputI, outputJ, outputF, outputO, outputTanhC, &tiling_data, workspace);
    lstmOp.Process();
  }
}
