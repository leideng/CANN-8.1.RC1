/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022.
 * Description: ioctl of fast notification for hpvisor
 * Author: lilinjie8 <lilinjie8@huawei.com>
 * Create: 2022-03-25
 */

#ifndef LINUX_HPVISOR_H
#define LINUX_HPVISOR_H
#include <stdint.h>
#include <linux/types.h>
#include <linux/ioctl.h>

#define HPVISORIO 0xAF

struct hpvisor_ioctl_arg {
	long int arg1;
	uintptr_t arg2;
	uintptr_t arg3;
};

struct hpvisor_chn_xid {
	int32_t pid; /* process id */
	int32_t uid; /* user id of task */
	int32_t gid; /* group id of task */
};

struct hpvisor_vnotify_setting {
	unsigned int max_stride_nr;
	unsigned int stride_chn_nr;
	unsigned int ctrl_mem_size;
	unsigned int strides_mem_size;
};

#define TRANSIT_DIRECTION 2
#define HPVISOR_VNOTIFY_CTRL_MEM_PAGE_OFFSET 0
#define HPVISOR_VNOTIFY_STRIDE_MEM_PAGE_OFFSET 4
#define RX_RQ_HEADER_OFFSET(idx) (idx * TRANSIT_DIRECTION)
#define TX_RQ_HEADER_OFFSET(idx) (idx * TRANSIT_DIRECTION + 1)

struct hpvisor_chn_fd {
	int fd0;  /* wait fd */
	int fd1;  /* send fd */
};

#define HPVISOR_VNOTIFY_CHECK _IO(HPVISORIO, 0x00)
#define HPVISOR_VNOTIFY_QUERY _IOW(HPVISORIO, 0x01, unsigned long)

/* called by hpvisor.so */
#define HPVISOR_VNOTIFY_CTRL_ATTACH _IO(HPVISORIO, 0x02)
#define HPVISOR_VNOTIFY_CTRL_DETACH _IOW(HPVISORIO, 0x03, unsigned long)
#define HPVISOR_VNOTIFY_CTRL_SEND _IOW(HPVISORIO, 0x04, unsigned long)
#define HPVISOR_VNOTIFY_GUEST_ATTACH _IOW(HPVISORIO, 0x05, unsigned long)
#define HPVISOR_VNOTIFY_GUEST_DETACH _IOW(HPVISORIO, 0x06, unsigned long)

/* called by vnotify.so */
#define HPVISOR_VNOTIFY_HOST_SEND _IOW(HPVISORIO, 0x07, unsigned long)
#define HPVISOR_VNOTIFY_HOST_WAKE _IOW(HPVISORIO, 0x08, unsigned long)
#define HPVISOR_VNOTIFY_HOST_FIND _IOW(HPVISORIO, 0x09, unsigned long)
#define HPVISOR_VNOTIFY_HOST_ATTACH _IOW(HPVISORIO, 0x0A, unsigned long)
#define HPVISOR_VNOTIFY_HOST_DETACH _IOW(HPVISORIO, 0x0B, unsigned long)
#define HPVISOR_VNOTIFY_PRINT_RECORD _IOW(HPVISORIO, 0x0C, unsigned long)
#define HPVISOR_VNOTIFY_GET_STRIDES_MEM_SIZE _IOW(HPVISORIO, 0x0D, unsigned long)
#define HPVISOR_VNOTIFY_GET_CHN_XID _IOW(HPVISORIO, 0x0E, unsigned long)
#define HPVISOR_VNOTIFY_GET_SETTING _IOW(HPVISORIO, 0x0F, unsigned long)

/* called by um mode */
#define UM_VNOTIFY_MEM_INIT _IOW(HPVISORIO, 0x10, unsigned long)

#define HPVISOR_ATTACH_CREATE_BIT	(1 << 0)
#define HPVISOR_ATTACH_SHARED_BIT	(1 << 1)
#define HPVISOR_ATTACH_PERSISTENT_BIT	(1 << 2)
#define HPVISOR_ATTACH_FAST_BIT		(1 << 3)
#define HPVISOR_ATTACH_RECEIVE_BIT	(1 << 4)

#define HPVISOR_CHN_HOST_ATTACH		(1 << 0)
#define HPVISOR_CHN_GUEST_ATTACH	(1 << 1)
#define HPVISOR_CHN_HOST_DETACH		(1 << 2)
#define HPVISOR_CHN_GUEST_DETACH	(1 << 3)

#endif
