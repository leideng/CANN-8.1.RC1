/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef GET_NPU_MEMORY_USAGE_H
#define GET_NPU_MEMORY_USAGE_H

#include <vector>


namespace msServiceProfiler {
struct dcmi_get_memory_info_stru {
    unsigned long long memory_size;      /* unit:MB */
    unsigned long long memory_available; /* free + hugepages_free * hugepagesize */
    unsigned int freq;
    unsigned long hugepagesize;          /* unit:KB */
    unsigned long hugepages_total;
    unsigned long hugepages_free;
    unsigned int utiliza;                /* ddr memory info usages */
    unsigned char reserve[60];           /* the size of dcmi_memory_info is 96 */
};

struct dsmi_hbm_info_stru {
    unsigned long long memory_size;  /**< HBM total size, MB */
    unsigned int freq;               /**< HBM freq, MHz */
    unsigned long long memory_usage; /**< HBM memory_usage, MB */
    int temp;                        /**< HBM temperature */
    unsigned int bandwith_util_rate;
};

const int EXITCODE_SUCCESS = 0;
const int EXITCODE_EMPTY_DCMI_HANDLER = 1;
const int PERCENTAGE_SCALE = 100;

struct CardDevice {
    int cardId;
    int deviceId;
};

class NpuMemoryUsage {
public:
    NpuMemoryUsage();
    ~NpuMemoryUsage();
    int InitDcmiCardAndDevices();
    int GetByDcmi(std::vector<int> &memoryUsed, std::vector<int> &memoryUtiliza);

private:
    void *handleDcmi = nullptr;
    bool isHbmDevice = false;
    std::vector<CardDevice> cardDevices;

    int DcmiInit() const;
    int DcmiGetCardList(int *cardNum, int *cardList, int listLen) const;
    int DcmiGetDeviceIdInCard(int cardId, int *deviceIdMax) const;
    int DcmiGetDeviceMemoryInfoV3(
        int cardId, int deviceId, struct dcmi_get_memory_info_stru *memoryInfo) const;
    int DcmiGetDeviceHbmInfo(int cardId, int deviceId, struct dsmi_hbm_info_stru *hbmInfo) const;
};
}  // namespace msServiceProfiler
#endif  // GET_NPU_MEMORY_USAGE_H
