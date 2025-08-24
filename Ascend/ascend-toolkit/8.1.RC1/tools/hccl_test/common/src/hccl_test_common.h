#ifndef __HCCL_TEST_COMMON_H_
#define __HCCL_TEST_COMMON_H_
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <unistd.h>
#include <vector>
#include "hccl/hccl.h"
#include <hccl/hccl_types.h>
#include <limits.h>
#include <ctype.h>
#include "acl/acl.h"
#include "acl/acl_prof.h"

#undef INT_MAX
#define INT_MAX __INT_MAX__

typedef signed char s8;
typedef signed short s16;
typedef signed int s32;
typedef signed long long s64;
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

struct DataSize {
    u64 min_bytes;
    u64 max_bytes;
    u64 step_bytes = 0;
    double step_factor;
    u64 count;
    u64 data_size;
    u64 type_size;
};

constexpr int SERVER_MAX_DEV_NUM = 8;
constexpr int BUF_ALGIN_SIZE = 512;
constexpr int BUF_ALGIN_LINE = 1024 * 1024;

#define ACLCHECK(cmd)                                                                                       \
    do {                                                                                                    \
        aclError ret = cmd;                                                                                 \
        if (ret != ACL_SUCCESS) {                                                                           \
            printf("acl interface return err %s:%d, retcode: %d.\n", __FILE__, __LINE__, ret);              \
            if (ret == ACL_ERROR_RT_MEMORY_ALLOCATION) {                                                    \
                printf("memory allocation error, check whether the current memory space is sufficient.\n"); \
            }                                                                                               \
            return ret;                                                                                     \
        }                                                                                                   \
    } while (0)

#define HCCLCHECK(cmd)                                                                                    \
    do {                                                                                                  \
        HcclResult ret = cmd;                                                                             \
        if (ret != HCCL_SUCCESS) {                                                                        \
            printf("hccl interface return errreturn err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret); \
            return ret;                                                                                   \
        }                                                                                                 \
    } while (0)

#define HCCLROOTRANKCHECK(cmd)                                                                            \
    do {                                                                                                  \
        HcclResult ret = cmd;                                                                             \
        if (ret != HCCL_SUCCESS && ret != HCCL_E_PARA) {                                                  \
            printf("hccl interface return errreturn err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret); \
            return ret;                                                                                   \
        }                                                                                                 \
    } while (0)
namespace hccl {
class HcclTest {
public:
    HcclTest();
    virtual ~HcclTest();

    void print_help();
    static struct option longopts[];

    int parse_opt(int opt);
    int parse_cmd_line(int argc, char *argv[]);

    int check_data_count();
    int check_cmd_line();

    // 计算当前进程rank号, 同一个服务器内的rank从0开始编号[0,nDev-1]
    int get_mpi_proc();

    int getAviDevs(const char *devs, std::vector<int> &dev_ids);

    virtual int hccl_op_base_test();
    virtual void init_data_count()
    {}
    virtual int destory_alloc_buf(); // 销毁集合通信内存资源
    virtual int init_hcclComm();

    int opbase_test_by_data_size();

    virtual int destory_hcclComm();

    int get_env_resource();
    int release_env_resource();

    int start_test();

protected:
    virtual size_t init_malloc_Ksize_by_data()
    {
        return 0;
    }
    virtual void init_send_recv_size_by_data(size_t &send_bytes, size_t &recv_bytes)
    {
        send_bytes = 0;
        recv_bytes = 0;
    }
    void get_buff_size(size_t &send_bytes, size_t &recv_bytes);
    // 如果不需要初始化send or recv，则返回0
    int prepare_zero_copy(const size_t &send_bytes, const size_t &recv_bytes);
    int alloc_hccl_send_recv_buffer(
        void *&send_buff, const size_t &send_bytes, void *&recv_buff, const size_t &recv_bytes);
    int free_send_recv_buff_and_disable_local_buffer();

private:
    int set_device_sat_mode();
    int device_init();

public:
    DataSize *data{nullptr};
    void *vir_ptr{nullptr};
    bool enable_zero_copy{false};
    size_t malloc_kSize{0};
    void *send_buff{nullptr};
    void *recv_buff{nullptr};
    std::vector<std::pair<void *, aclrtDrvMemHandle>> phy_alloc_mem_handle;
    aclrtPhysicalMemProp prop;
    size_t granularity{128};                      // 默认128 Byte
    size_t physicalGranularity{2 * 1024 * 1024};  // zero_copy默认对齐2M

    long data_parsed_begin = 64 * 1024 * 1024;
    long data_parsed_end = 64 * 1024 * 1024;
    int64_t temp_step_bytes = 0;
    int iters = 20;
    int op_type = HCCL_REDUCE_SUM;
    int dtype = HCCL_DATA_TYPE_FP32;
    int warmup_iters = 10;
    int check = 1;
    u32 dev_count = 0;
    int stepfactor_flag = 0;
    int stepbytes_flag = 0;
    int op_flag = 0;
    int root_rank = 0;
    uint32_t dev_id = 0;
    int rank_id = 0;
    int rank_size = 0;

    aclrtStream stream;
    HcclComm hccl_comm;
    HcclRootInfo comm_id;

    aclrtEvent start_event, end_event;

    bool print_header = true;
    bool print_dump = true;
    bool need_ranksize_alignment = false;

private:
    // 当前进程在通信域(MPI_COMM_WORLD)内的进程号
    int proc_rank = 0;
    // 通信域(MPI_COMM_WORLD)中的总进程数
    int proc_size = 0;
    // 当前进程在服务器内的rank号，每个服务器内的rank号都是从0开始索引
    int local_rank = 0;
    int npus = -1;
    int profiling_flag = 0;
    aclprofConfig *profiling_config = NULL;
    int npus_flag = 0;
};
}  // namespace hccl

#endif
