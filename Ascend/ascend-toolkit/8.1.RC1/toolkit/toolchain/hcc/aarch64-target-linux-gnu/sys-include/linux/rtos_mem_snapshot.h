/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024.
 * Description: External data interface type of the real-time snapshot persist memory.
 * Author: heyuqiang
 * Create: 2024-04-07
 */
#ifndef _LINUX_RTOS_MEM_SNAPSHOT_H
#define _LINUX_RTOS_MEM_SNAPSHOT_H

#include <linux/types.h>

enum {
	PMEM_BUILD_PHYMAP,
	PMEM_CLEAR_PHYMAP,
	PMEM_RELEASE,
	PMEM_GET_SIZE,
	PMEM_RESET_ON_REBOOT,
	PMEM_DETACH,
	PMEM_GET_STATUS,
	PMEM_GET_BITMAP,
};

struct persist_mem_ioctl_args {
	__s32 id;
	__u64 addr;
	__u64 length;
	__s32 prot;
	__s32 flag;
	__s32 status;
	__s32 mark;
	__u64 bitmap;
	__u64 bitmap_size;
};

#define PERSIST_MEM_IOC_MAGIC 'P'
#define PMEM_IOWR(nr, type) _IOWR(PERSIST_MEM_IOC_MAGIC, nr, type)

#define PERSIST_MEM_IOC_BUILD_PHYMAP    PMEM_IOWR(PMEM_BUILD_PHYMAP, struct persist_mem_ioctl_args)
#define PERSIST_MEM_IOC_CLEAR_PHYMAP    PMEM_IOWR(PMEM_CLEAR_PHYMAP, struct persist_mem_ioctl_args)
#define PERSIST_MEM_IOC_RELEASE         PMEM_IOWR(PMEM_RELEASE, struct persist_mem_ioctl_args)
#define PERSIST_MEM_IOC_GET_SIZE        PMEM_IOWR(PMEM_GET_SIZE, struct persist_mem_ioctl_args)
#define PERSIST_MEM_IOC_RESET_ON_REBOOT PMEM_IOWR(PMEM_RESET_ON_REBOOT, struct persist_mem_ioctl_args)
#define PERSIST_MEM_IOC_DETACH          PMEM_IOWR(PMEM_DETACH, struct persist_mem_ioctl_args)
#define PERSIST_MEM_IOC_GET_STATUS      PMEM_IOWR(PMEM_GET_STATUS, struct persist_mem_ioctl_args)
#define PERSIST_MEM_IOC_GET_BITMAP      PMEM_IOWR(PMEM_GET_BITMAP, struct persist_mem_ioctl_args)
#endif
