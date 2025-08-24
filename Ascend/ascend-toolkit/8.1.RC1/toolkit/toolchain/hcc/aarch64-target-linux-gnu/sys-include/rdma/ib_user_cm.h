/* SPDX-License-Identifier: ((GPL-2.0 WITH Linux-syscall-note) OR Linux-OpenIB) */

#ifndef IB_USER_CM_H
#define IB_USER_CM_H

#include <linux/types.h>

#define MAX_FD_NUMBER 2

enum {
	IB_EXT_FD_CMD_HOLD,
	IB_EXT_FD_CMD_REMOVE,
	IB_EXT_FD_CMD_RECOVER,
	IB_EXT_FD_CMD_CLEAN,
	IB_EXT_FD_CMD_RSVD1,
	IB_EXT_FD_CMD_RSVD2,
	IB_EXT_FD_CMD_RSVD3,
	IB_EXT_FD_CMD_RSVD4,
};

struct ib_ext_fd_hdr {
	__u32 cmd;
	__u16 in;
	__u16 out;
};

struct ib_ext_hold_fd_resp {
	__s32 handle[MAX_FD_NUMBER];
};

struct ib_ext_hold_fd {
	__u32 fd_num;
	__u32 port;
	__s32 fd[MAX_FD_NUMBER];
	__aligned_u64 response;
};

struct ib_ext_recover_fd_resp {
	__s32 fd[MAX_FD_NUMBER];
};

struct ib_ext_recover_fd {
	__u32 fd_num;
	__u32 rsvd;
	__s32 handle[MAX_FD_NUMBER];
	__aligned_u64 response;
};

struct ib_ext_remove_fd_resp {
	__s32 handle[MAX_FD_NUMBER];
};

struct ib_ext_remove_fd {
	__u32 fd_num;
	__u32 rsvd;
	__s32 handle[MAX_FD_NUMBER];
	__aligned_u64 response;
};

struct ib_ext_clean_fd {
	__u32 port;
	__u32 rsvd;
};

#endif /* IB_USER_CM_H */

