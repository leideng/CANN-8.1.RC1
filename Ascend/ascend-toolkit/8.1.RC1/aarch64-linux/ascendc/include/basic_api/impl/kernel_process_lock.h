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
 * \file kernel_process_lock.h
 * \brief
 */
#ifndef __KERNEL_PROCESS_LOCK_H__
#define __KERNEL_PROCESS_LOCK_H__
#ifdef ASCENDC_CPU_DEBUG
#include <pthread.h>
#include <sys/types.h>
#include <cstdlib>

namespace AscendC {
class ProcessLock {
public:
    int Read();
    int Write();
    int Unlock();
    void UnInit();
    static void FreeLock();
    ProcessLock();
    static ProcessLock* CreateLock();

    static ProcessLock* GetProcessLock()
    {
        if (processLock == nullptr) {
            processLock = CreateLock();
        }
        return processLock;
    }
    ~ProcessLock();

private:
    pthread_rwlock_t lock;
    pthread_rwlockattr_t attr;
    inline void Init();
    static ProcessLock* processLock;
};
} // namespace AscendC
#endif
#endif // __KERNEL_PROCESS_LOCK_H__
