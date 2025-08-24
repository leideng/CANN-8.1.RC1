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
 * \file bidirection_lstm.cpp
 * \brief
 */

#include "lstm_bidir_fp16.cpp"

extern "C" __global__ __aicore__ void bidirection_lstm(GM_ADDR x, GM_ADDR init_h, GM_ADDR init_c, GM_ADDR w_ih, GM_ADDR w_hh,
                                                  GM_ADDR b_ih, GM_ADDR b_hh, GM_ADDR w_ih_reverse, GM_ADDR w_hh_reverse,
                                                  GM_ADDR b_ih_reverse, GM_ADDR b_hh_reverse, GM_ADDR y, GM_ADDR output_h,
                                                  GM_ADDR output_c, GM_ADDR usrworkspace, GM_ADDR lstmTiling) {
  SetAtomicNone();
  GET_TILING_DATA(tiling_data, lstmTiling);
  const BidirectionLSTMTilingData* __restrict tilingData = &tiling_data;
  if (TILING_KEY_IS(10000001)) {
    LstmBidirFP16 lstm_handle;
    lstm_handle.Init(x, init_h, init_c, w_ih, w_hh, b_ih, b_hh, w_ih_reverse, w_hh_reverse,
                    b_ih_reverse, b_hh_reverse, nullptr, y, output_h, output_c, usrworkspace, tilingData);
    lstm_handle.Process();
  }
}
