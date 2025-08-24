/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
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

#ifndef MS_SERVER_PROFILER_MARKER_H
#define MS_SERVER_PROFILER_MARKER_H

#include <string>
#include <thread>

#include <nlohmann/json.hpp>

#include "ServiceProfilerInterface.h"

using Json = nlohmann::json;

namespace msServiceProfiler {
    using AclprofConfig = struct aclprofConfig;
    class ServiceProfilerManager {
    public:
        ServiceProfilerManager(const ServiceProfilerManager &) = delete;
        ServiceProfilerManager& operator=(const ServiceProfilerManager &) = delete;

        static ServiceProfilerManager &GetInstance();

        inline bool IsEnable(uint32_t level) const
        {
            return enable_ && level_ >= level;
        }

        void StartProfiler();

        void StopProfiler();

        static std::string ToSemName(const std::string &oriSemName);

        std::string &GetConfigPath()
        {
            return configPath_;
        }

    private:
        ServiceProfilerManager();

        ~ServiceProfilerManager();

        Json ReadConfig();

        void ReadEnable(const Json &config);

        void ReadProfPath(const Json &config);

        void ReadLevel(const Json &config);

        void ReadAclTaskTime(const Json &config);

        bool ReadCollectConfig(const Json &config);

        bool ReadHostConfig(const Json &config);

        bool ReadNpuConfig(const Json &config);

        void SetAclProfHostSysConfig() const;

        void DynamicControl();

        void LaunchThread();

        void ThreadFunction();

        void ReadConfigPath();

        void MarkFirstProcessAsMain();

        void InitProfPathDateTail(bool forceReinit = false);

        AclprofConfig* ProfCreateConfig();

    private:
        bool isMaster_ = true;
        bool enable_ = false;
        bool started_ = false;
        bool isAclInit_ = false;
        std::string configPath_;
        std::string profPath_;
        std::string profPathDateTail_;
        uint32_t level_ = Level::INFO;
        bool enableAclTaskTime_ = false;
        void *configHandle_ = nullptr;
        int lastUpdate_ = 0;

        bool hostCpuUsage_ = false;
        bool hostMemoryUsage_ = false;
        uint32_t hostFreq_ = 10;
        uint32_t hostFreqMin_ = 1;
        uint32_t hostFreqMax_ = 50;

        bool npuMemoryUsage_ = false;
        uint32_t npuMemoryFreq_ = 1;
        uint32_t npuMemoryFreqMin_ = 1;
        uint32_t npuMemoryFreqMax_ = 50;
        uint32_t npuMemorySleepMilliseconds_ = 1000;

        std::thread thread_;
    };
}  // namespace msServiceProfiler

#endif
