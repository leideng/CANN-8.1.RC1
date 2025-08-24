#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include "mpi.h"
#include "hccl_opbase_rootinfo_base.h"
#include "hccl_allgather_rootinfo_test.h"
#include "hccl_test_common.h"
#include <algorithm>

constexpr s32 HCCL_TEST_REDUCE_RESERVED = 4;
constexpr s32 HCCL_TEST_DATA_TYPE_RESERVED = 12;
HcclReduceOp test_ops[HCCL_TEST_REDUCE_RESERVED] = {
    HCCL_REDUCE_SUM, HCCL_REDUCE_PROD, HCCL_REDUCE_MAX, HCCL_REDUCE_MIN};
const char *test_opnames[HCCL_TEST_REDUCE_RESERVED] = {"sum", "prod", "max", "min"};

HcclDataType test_types[HCCL_TEST_DATA_TYPE_RESERVED] = {HCCL_DATA_TYPE_INT8, /**< int8 */
    HCCL_DATA_TYPE_INT16,                                                     /**< int16 */
    HCCL_DATA_TYPE_INT32,                                                     /**< int32 */
    HCCL_DATA_TYPE_FP16,                                                      /**< fp16 */
    HCCL_DATA_TYPE_FP32,                                                      /**< fp32 */
    HCCL_DATA_TYPE_INT64,                                                     /**< int64 */
    HCCL_DATA_TYPE_UINT64,                                                    /**< uint64 */
    HCCL_DATA_TYPE_UINT8,                                                     /**< uint8 */
    HCCL_DATA_TYPE_UINT16,                                                    /**< uint16 */
    HCCL_DATA_TYPE_UINT32,                                                    /**< uint32 */
    HCCL_DATA_TYPE_FP64,                                                      /**< fp64 */
    HCCL_DATA_TYPE_BFP16};
const char *test_typenames[HCCL_TEST_DATA_TYPE_RESERVED] = {
    "int8", "int16", "int32", "fp16", "fp32", "int64", "uint64", "uint8", "uint16", "uint32", "fp64", "bfp16"};

int get_hccl_op_from_str(char *str)
{
    for (int op = 0; op < HCCL_TEST_REDUCE_RESERVED; op++) {
        if (strcmp(str, test_opnames[op]) == 0) {
            return op;
        }
    }

    return -1;
}

int get_hccl_dtype_from_str(char *str)
{
    for (int t = 0; t < HCCL_TEST_DATA_TYPE_RESERVED; t++) {
        if (strcmp(str, test_typenames[t]) == 0) {
            return t;
        }
    }
    return -1;
}

static void get_host_name(char *hostName, int maxlen)
{
    gethostname(hostName, maxlen);
    for (int i = 0; i < maxlen; i++) {
        if (hostName[i] == '.') {
            hostName[i] = '\0';
            return;
        }
    }
    return;
}

static uint64_t get_host_hash(const char *string)
{
    // Based on DJB2, result = result * 33 + char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++) {
        result = ((result << 5) + result) + string[c];
    }
    return result;
}

static long parsesize(const char *value)
{
    long long int units;
    long size;
    char *size_lit;

    size = strtol(value, &size_lit, 0);
    if (strlen(size_lit) == 1) {
        switch (*size_lit) {
            case 'G':
            case 'g':
                units = 1024 * 1024 * 1024;
                break;
            case 'M':
            case 'm':
                units = 1024 * 1024;
                break;
            case 'K':
            case 'k':
                units = 1024;
                break;
            default:
                return -1;
        }
    } else if (strlen(size_lit) == 0) {
        units = 1;
    } else {
        return -1;
    }

    return size * units;
}

u32 sal_str_len(const char *s, u32 maxLen = INT_MAX)
{
    return strnlen(s, maxLen);
}

int is_all_digit(const char *strNum)
{
    // 参数有效性检查
    if (strNum == NULL) {
        printf("Error: ptr [%s] is NULL\n", strNum);
        return -1;
    }

    u32 nLength = sal_str_len(strNum);
    for (u32 index = 0; index < nLength; index++) {
        if (!isdigit(strNum[index])) {
            printf("Error:In judge all digit, Check whether the value of [-i -n -r -w -c -p] is a positive integer.\n");
            return -1;
        }
    }
    return 0;
}

long strtol_alldigit(const char *optarg)
{
    long ret = is_all_digit(optarg);
    if (ret != 0) {
        return ret;
    }

    return strtol(optarg, NULL, 0);
}

aclrtMemAttr use_ddr_mem()
{
    constexpr char ASCEND910[] = "Ascend910";
    constexpr char ASCEND910B1[] = "Ascend910B1";
    constexpr char ASCEND910B2[] = "Ascend910B2";
    constexpr char ASCEND910B2C[] = "Ascend910B2C";
    constexpr char ASCEND910B3[] = "Ascend910B3";
    constexpr char ASCEND910B4[] = "Ascend910B4";
    constexpr char ASCEND910B4_1[] = "Ascend910B4-1";
    constexpr char ASCEND910_9391[] = "Ascend910_9391";
    constexpr char ASCEND910_9381[] = "Ascend910_9381";
    constexpr char ASCEND910_9392[] = "Ascend910_9392";
    constexpr char ASCEND910_9382[] = "Ascend910_9382";
    constexpr char ASCEND910_9372[] = "Ascend910_9372";
    constexpr char ASCEND910_9361[] = "Ascend910_9361";
    constexpr char ASCEND610[] = "Ascend610";
    constexpr char ASCEND610B1[] = "Ascend610B1";
    constexpr char ASCEND610B2[] = "Ascend610B2";
    constexpr char ASCEND610LITE[] = "Ascend610Lite";
    constexpr char BS9SX1A[] = "BS9SX1A";
    constexpr char BS9SX1AA[] = "BS9SX1AA";
    constexpr char BS9SX1AB[] = "BS9SX1AB";
    constexpr char BS9SX1AC[] = "BS9SX1AC";
    constexpr char BS9SX2A[] = "BS9SX2A";
    constexpr char BS9SX2AA[] = "BS9SX2AA";
    constexpr char BS9SX2AB[] = "BS9SX2AB";
    constexpr char MC61AM21A[] = "MC61AM21A";
    constexpr char MC61AM21AA[] = "MC61AM21AA";
    constexpr char MC61AM21AB[] = "MC61AM21AB";
    constexpr char ASCEND310P[] = "Ascend310P";
    constexpr char ASCEND310P1[] = "Ascend310P1";
    constexpr char ASCEND310P2[] = "Ascend310P2";
    constexpr char ASCEND310P3[] = "Ascend310P3";
    constexpr char ASCEND310P4[] = "Ascend310P4";
    constexpr char ASCEND310B1[] = "Ascend310B1";
    constexpr char ASCEND310B2[] = "Ascend310B2";
    constexpr char ASCEND310B3[] = "Ascend310B3";
    constexpr char ASCEND310B4[] = "Ascend310B4";
    constexpr char ASCEND310B[] = "Ascend310B";
    constexpr char AS31XM1X[] = "AS31XM1X";
    std::string socVersion = aclrtGetSocName();
    // printf("socVersion %s\n", socVersion.c_str());
    if (socVersion.compare(0, strlen(ASCEND310P), ASCEND310P) == 0 ||
        socVersion.compare(0, strlen(BS9SX1A), BS9SX1A) == 0 ||
        socVersion.compare(0, strlen(ASCEND610), ASCEND610) == 0 ||
        socVersion.compare(0, strlen(ASCEND610LITE), ASCEND610LITE) == 0 ||
        socVersion.compare(0, strlen(BS9SX2A), BS9SX2A) == 0 ||
        socVersion.compare(0, strlen(MC61AM21A), MC61AM21A) == 0 ||
        socVersion.compare(0, strlen(ASCEND310B), ASCEND310B) == 0 ||
        socVersion.compare(0, strlen(AS31XM1X), AS31XM1X) == 0) {
        return aclrtMemAttr::ACL_DDR_MEM_HUGE;
    }
    return aclrtMemAttr::ACL_HBM_MEM_HUGE;
}

int check_alloc_memory_size(size_t &alloc_size)
{
    size_t freeMem = 0;
    size_t totalMem = 0;
    size_t reservedMem;
    aclError ret = aclrtGetMemInfo(use_ddr_mem(), &freeMem, &totalMem);
    if (ret != ACL_ERROR_NONE) {
        printf("Get ddr memory info failed.");
        return HCCL_E_MEMORY;
    }
    if (alloc_size > freeMem) {
        printf("Alloc memory size is larger than free memory, allocSize: %zu, freeMem: %zu, totalMem: %zu\n",
            alloc_size,
            freeMem,
            totalMem);
        return HCCL_E_MEMORY;
    }
    return HCCL_SUCCESS;
}

namespace hccl {
HcclTest::HcclTest()
{
    data = new DataSize;
    data->step_factor = 1;
    proc_rank = 0;
    proc_size = 0;
}

HcclTest::~HcclTest()
{
    delete data;
    data = nullptr;
}

struct option HcclTest::longopts[] = {{"op", required_argument, 0, 'o'},
    {"datatype", required_argument, 0, 'd'},
    {"minbytes", required_argument, 0, 'b'},
    {"maxbytes", required_argument, 0, 'e'},
    {"stepbytes", required_argument, 0, 'i'},
    {"stepfactor", required_argument, 0, 'f'},
    {"root", required_argument, 0, 'r'},
    {"iters", required_argument, 0, 'n'},
    {"warmup_iters", required_argument, 0, 'w'},
    {"check", required_argument, 0, 'c'},
    {"npus", required_argument, 0, 'p'},
    {"help", no_argument, 0, 'h'},
    {"zero_copy", required_argument, 0, 'z'}};

void HcclTest::print_help()
{
    printf("USAGE: ./test \n\t"
           "[-b,--minbytes <min size in bytes>] \n\t"
           "[-e,--maxbytes <max size in bytes>] \n\t"
           "[-i,--stepbytes <increment size>] \n\t"
           "[-f,--stepfactor <increment factor>] \n\t"
           "[-n,--iters <iteration count>] \n\t"
           "[-o,--op <sum/prod/min/max>] \n\t"
           "[-d,--datatype <int8/int16/int32/fp16/fp32/int64/uint64/uint8/uint16/uint32/fp64/bfp16>] \n\t"
           "[-r,--root <root>] \n\t"
           "[-w,--warmup_iters <warmup iteration count>] \n\t"
           "[-c,--check <result verification> 0:disabled 1:enabled.] \n\t"
           "[-p,--npus <npus used for one node>] \n\t"
           "[-z,--zero_copy  0:disabled 1:enabled.] \n\t"
           "[-h,--help]\n");
    return;
}

int HcclTest::check_data_count()
{
    if (data_parsed_begin < 0 || data_parsed_end < 0) {
        printf("invalid size specified for [-b,--minbytes] or [-e,--maxbytes]\n");
        return -1;
    }
    data->min_bytes = (u64)data_parsed_begin;
    data->max_bytes = (u64)data_parsed_end;

    if (stepbytes_flag != 0 && temp_step_bytes < 0) {
        printf("Error: [-i,--stepbytes] must be greater than or equal to 0.\n");
        return -1;
    }

    if (data->max_bytes < data->min_bytes) {
        printf("invalid option: maxbytes < minbytes, Check the [-b,--minbytes] and [-e,--maxbytes] options.\n");
        return -1;
    } else {
        if (stepbytes_flag != 0) {  // 用户配置了增量步长
            data->step_bytes = temp_step_bytes;
        } else {  // 用户未配置增量步长
            if (data->max_bytes == data->min_bytes) {
                data->step_bytes =
                    1;  // 用户配置数据量的起始值和结束值相同，但未配置增量步长，为防止进入死循环，设置增量步长为1
            }
            if (data->max_bytes > data->min_bytes) {
                data->step_bytes = (data->max_bytes - data->min_bytes) / 10;
            }
        }
    }

    // 数据量需要被ranksize整除
    if (need_ranksize_alignment) {
        u64 algin_size = rank_size * BUF_ALGIN_SIZE;
        if (data->min_bytes < rank_size) {
            data->min_bytes = rank_size;
        }
        if (data->min_bytes > BUF_ALGIN_LINE) {
            data->min_bytes = data->min_bytes / algin_size * algin_size;
        } else {
            data->min_bytes = data->min_bytes / rank_size * rank_size;
        }

        if (data->max_bytes < rank_size) {
            data->max_bytes = rank_size;
        }
        if (data->max_bytes > BUF_ALGIN_LINE) {
            data->max_bytes = data->max_bytes / algin_size * algin_size;
        } else {
            data->max_bytes = data->max_bytes / rank_size * rank_size;
        }

        if (data->step_bytes > BUF_ALGIN_LINE) {
            data->step_bytes = (data->step_bytes + algin_size - 1) / algin_size * algin_size;
        } else {
            data->step_bytes = (data->step_bytes + rank_size - 1) / rank_size * rank_size;
        }
    }

    if (stepfactor_flag != 0 && data->step_factor <= 1.0) {
        printf("Error: [-f,--stepfactor] Must be greater than 1.0f, Start step mod.\n");
        return -1;
    }

    if (stepfactor_flag != 0 && stepbytes_flag != 0) {
        printf("Warning: [-f,--stepfactor] and [-i,--stepbytes] are set, [-f,--stepfactor] is enabled by default.\n");
    }

    return 0;
}

int HcclTest::check_cmd_line()
{
    int ret = 0;
    ret = check_data_count();
    if (ret != 0) {
        return ret;
    }

    if (dtype == -1) {
        printf("Error: [-d,--datatype] is invalid, Use [-h,--help] to check the correct input parameter.\n");
        return -1;
    }

    if (op_type == -1) {
        printf("Error: [-o,--op] is invalid, Use [-h,--help] to check the correct input parameter.\n");
        return -1;
    }

    if (warmup_iters < 0) {
        printf("Error: [-w,--warmup_iters] is invalid, warmup_iters must be greater than or equal to 0.\n");
        return -1;
    }

    if (iters < 0) {
        printf("Error: [-n,--iters] is invalid, iters must be greater than or equal to 0.\n");
        return -1;
    }

    if (root_rank >= rank_size || root_rank < 0)  // 如果指定的root rank大于等于rank_size
    {
        printf("Error: [-r,--root <root>] is invalid, root rank must be greater than or equal to 0 and less than or "
               "equal to %d.\n",
            rank_size - 1);
        return -1;
    }

    if (check != 1 && check != 0) {
        printf("Error: [-c,--check] is invalid, check should be 0 or 1\n");
        return -1;
    }

    if (dev_count == 0) {
        printf("Error: The number of device is 0.Check whether the package is correct.\n");
        return -1;
    }

    if (npus < 1 || npus > dev_count) {
        printf("Error: [-p,--npus <npus used for one node>] is invalid, npus must be greater than or equal to 1 and "
               "less than or equal to %d.\n",
            dev_count);
        return -1;
    }
    
    return 0;
}

int HcclTest::get_env_resource()
{
    // 支持profiling
    const char *profiling_env = getenv("HCCL_TEST_PROFILING");
    if (profiling_env != NULL) {
        profiling_flag = atoi(profiling_env);
        u32 nLength = sal_str_len(profiling_env);
        // 校验：入参为字符
        for (u32 index = 0; index < nLength; index++) {
            if (!isdigit(profiling_env[index])) {
                printf("Check whether HCCL_TEST_PROFILING is 0 or 1.\n");
                return -1;
            }
        }
        // 校验：入参非0非1
        if (profiling_flag != 0 && profiling_flag != 1) {
            printf("Check whether HCCL_TEST_PROFILING is 0 or 1.\n");
            return -1;
        }
    }

    // 开启profiling
    if (profiling_flag == 1) {
        std::string prof_path = "/var/log/npu/profiling";
        const char *profiling_env_path = getenv("HCCL_TEST_PROFILING_PATH");
        if (profiling_env_path != NULL) {
            prof_path = profiling_env_path;
        }
        aclprofInit(prof_path.c_str(), prof_path.size());
        uint32_t profSwitch = ACL_PROF_ACL_API | ACL_PROF_TASK_TIME | ACL_PROF_AICORE_METRICS | ACL_PROF_AICPU |
                              ACL_PROF_HCCL_TRACE | ACL_PROF_MSPROFTX | ACL_PROF_RUNTIME_API;
        uint32_t deviceIdList = dev_id;
        int devNum = 1;
        profiling_config = aclprofCreateConfig(&deviceIdList, devNum, ACL_AICORE_PIPE_UTILIZATION, nullptr, profSwitch);
        ACLCHECK(aclprofStart(profiling_config));
    }

    return 0;
}

int HcclTest::release_env_resource()
{
    if (profiling_flag == 1) {
        ACLCHECK(aclprofStop(profiling_config));
        aclprofFinalize();
    }

    return 0;
}

int HcclTest::parse_opt(int opt)
{
    switch (opt) {
        case 'b':
            data_parsed_begin = parsesize(optarg);
            break;
        case 'e':
            data_parsed_end = parsesize(optarg);
            break;
        case 'i':
            stepbytes_flag++;
            temp_step_bytes = strtol_alldigit(optarg);
            break;
        case 'f':
            stepfactor_flag++;
            char *temp;
            data->step_factor = strtof(optarg, &temp);
            break;
        case 'n':
            iters = strtol_alldigit(optarg);
            break;
        case 'o':
            op_flag++;
            op_type = get_hccl_op_from_str(optarg);
            break;
        case 'd':
            dtype = get_hccl_dtype_from_str(optarg);
            break;
        case 'r':
            root_rank = strtol_alldigit(optarg);
            break;
        case 'w':
            warmup_iters = strtol_alldigit(optarg);
            break;
        case 'c':
            check = strtol_alldigit(optarg);
            break;
        case 'p':
            npus = strtol_alldigit(optarg);
            npus_flag = 1;
            break;
        case 'z':
            enable_zero_copy = std::stoi(optarg);
            break;
        case 'h':
            print_help();
            return 1;
        default:
            printf("invalid option \n");
            printf("Try [-h --help] for more information.\n");
            return -1;
    }
    return 0;
}

int HcclTest::parse_cmd_line(int argc, char *argv[])
{
    int opt = -1;
    int longindex = 0;
    int ret = 0;
    long parsed;
    while (-1 != (opt = getopt_long(argc, argv, "o:d:b:e:i:f:r:n:w:c:p:z:h", longopts, &longindex))) {
        ret = parse_opt(opt);
        if (ret != 0) {
            return ret;
        }
    }

    if (optind < argc) {
        printf("non-option ARGV-elements: ");
        while (optind < argc) {
            printf("%s ", argv[optind++]);
        }
        printf("\n");
        return -1;
    }

    return 0;
}

int HcclTest::getAviDevs(const char *devs, std::vector<int> &dev_ids)
{

    std::string use_devs(devs);
    std::string pattern = ",";
    std::string::size_type pos;
    use_devs += pattern;
    size_t val_size = use_devs.size();
    for (size_t i = 0; i < val_size; ++i) {
        pos = use_devs.find(pattern, i);
        if (pos < val_size) {
            std::string s = use_devs.substr(i, pos);
            int tmp_rank = atoi(s.c_str());
            dev_ids.push_back(tmp_rank);
            i = pos + pattern.size() - 1;
        }
    }

    return 0;
}

int HcclTest::get_mpi_proc()
{
    // 获取当前进程在所属进程组的编号
    MPI_Comm_size(MPI_COMM_WORLD, &proc_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

    ACLCHECK(aclrtGetDeviceCount(&dev_count));

    // 入参没有-p参数，npus默认值为device count
    if (npus_flag == 0) {
        npus = dev_count;
    }

    rank_id = proc_rank;
    rank_size = proc_size;

    const char *devs = getenv("HCCL_TEST_USE_DEVS");
    if (devs != NULL) {
        std::vector<int> dev_ids;
        int ret = getAviDevs(devs, dev_ids);
        sort(dev_ids.begin(), dev_ids.end());

        int local_rank;
        local_rank = proc_rank % dev_ids.size();
        for (int i = 0; i < dev_ids.size(); i++) {
            if (local_rank == i) {
                dev_id = dev_ids[i];
                break;
            }
        }
    } else {
        dev_id = proc_rank % npus;
    }

    return 0;
}

int HcclTest::hccl_op_base_test()
{
    return 0;
}

int HcclTest::destory_alloc_buf()
{
    return 0;
}

int HcclTest::set_device_sat_mode()
{
    const char *soc_name_ptr = aclrtGetSocName();
    if (soc_name_ptr == nullptr) {
        printf("aclrtGetSocName failed");
        return -1;
    }

    std::string support_soc_name = "Ascend910B";
    if (support_soc_name.length() != strlen(soc_name_ptr) &&
        support_soc_name.compare(0, support_soc_name.length(), soc_name_ptr, 0, support_soc_name.length()) == 0) {
        ACLCHECK(aclrtSetDeviceSatMode(ACL_RT_OVERFLOW_MODE_INFNAN));
    }
    return 0;
}

size_t get_reseved_size(size_t send_bytes, size_t recv_bytes, size_t physicalGranularity)
{
   return (((send_bytes + physicalGranularity - 1) / physicalGranularity) * physicalGranularity) +
                       (((recv_bytes + physicalGranularity - 1) / physicalGranularity) * physicalGranularity);
}

int HcclTest::start_test()
{
    int ret = 0;
    ret = device_init();
    if (ret != 0) {
        return ret;
    }
    size_t reserve_mem = 0;
    if (enable_zero_copy)
    {
        // 获取send recv内存大小
        data->data_size = data->min_bytes;
        if (!(stepbytes_flag && !data->step_bytes)) {
            for (; data->data_size <= data->max_bytes;
                 (data->step_factor <= 1.0 ? data->data_size += data->step_bytes
                                           : data->data_size *= data->step_factor)) {
                if (need_ranksize_alignment && data->data_size > BUF_ALGIN_LINE) {
                    u64 algin_size = rank_size * BUF_ALGIN_SIZE;
                    data->data_size = (data->data_size + algin_size - 1) / algin_size * algin_size;
                }
            }
        }

        size_t send_bytes = 0;
        size_t recv_bytes = 0;
        // 设置内存对齐大小 默认对齐128字节，开启zero_copy，默认对齐2M
        prop = {aclrtMemHandleType::ACL_MEM_HANDLE_TYPE_NONE,
            aclrtMemAllocationType::ACL_MEM_ALLOCATION_TYPE_PINNED,
            use_ddr_mem(),
            {dev_id, aclrtMemLocationType::ACL_MEM_LOCATION_TYPE_DEVICE},
            0};
        auto status =
            aclrtMemGetAllocationGranularity(&prop, ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM, &physicalGranularity);
        if (status != ACL_SUCCESS) {
            physicalGranularity = 2 * 1024 * 1024;
        }
        get_buff_size(send_bytes, recv_bytes);
        reserve_mem = get_reseved_size(send_bytes, recv_bytes, physicalGranularity);
        //printf("reserve_mem: %zu, send_bytes: %zu, recv_bytes: %zu, physicalGranularity: %zu\n", reserve_mem,
        //    send_bytes, recv_bytes, physicalGranularity);
        HCCLCHECK(static_cast<HcclResult>(check_alloc_memory_size(reserve_mem)));
        status = aclrtReserveMemAddress(&vir_ptr, reserve_mem, 0, NULL, 1);
        if (status != ACL_SUCCESS) {
            printf("[%s][%d] aclrtReserveMemAddress failed.\n", __FUNCTION__, __LINE__);
            goto error_device_init;
        }
    }
    ret = init_hcclComm();
    if (ret != 0) {
        goto error_reserve_memAddress;
    }
    if (enable_zero_copy) {
        auto status = HcclCommSetMemoryRange(hccl_comm, vir_ptr, reserve_mem, 0, 0);
        if (status != HCCL_SUCCESS) {
            printf("[%s][%d] HcclCommSetMemoryRange failed.\n", __FUNCTION__, __LINE__);
            goto error_reserve_lpcMemory;
        }
    }
    ret = opbase_test_by_data_size();

error_reserve_lpcMemory:
    if (enable_zero_copy) {
        ret = HcclCommUnsetMemoryRange(hccl_comm, vir_ptr);
    }
error_destory_comm:
    ret = HcclCommDestroy(hccl_comm);
error_reserve_memAddress:
    if (enable_zero_copy) {
        ret = aclrtReleaseMemAddress(vir_ptr);
    }
error_device_init:
    ret = destory_hcclComm();
    return ret;
}

int HcclTest::device_init()
{
    // 设备资源初始化
    ACLCHECK(aclInit(NULL));
    // 指定集合通信操作使用的设备
    ACLCHECK(aclrtSetDevice(dev_id));
    // 关闭溢出检测
    int ret = set_device_sat_mode();
    if (ret != 0) {
        printf("set_device_sat_mode execute failed, Detailed logs are stored in path: /root/ascend/log/\n");
        return ret;
    }
    ACLCHECK(aclrtCreateEvent(&start_event));
    ACLCHECK(aclrtCreateEvent(&end_event));
    // 创建任务stream
    ACLCHECK(aclrtCreateStream(&stream));
    // 设置遇错即停
    ACLCHECK(aclrtSetStreamFailureMode(stream, 1));
    return 0;
}

int HcclTest::init_hcclComm()
{
    // 将root_info广播到通信域内的其他rank
    MPI_Request request;
    MPI_Status status;
    // 在root_rank获取root_info
    if (rank_id == root_rank) {
        printf("the minbytes is %llu, maxbytes is %llu, iters is %d, warmup_iters is %d\n",
            data->min_bytes,
            data->max_bytes,
            iters,
            warmup_iters);
        HcclResult getRootInfo = HcclGetRootInfo(&comm_id);
        if (getRootInfo == HCCL_SUCCESS) {
            // 将root_info广播到通信域内的其他rank
            MPI_Ibcast(&comm_id, HCCL_ROOT_INFO_BYTES, MPI_CHAR, root_rank, MPI_COMM_WORLD, &request);
        } else {
            // 通知其他rank获取root_info失败
            char send_str[] = "invalid";
            MPI_Ibcast(send_str, HCCL_ROOT_INFO_BYTES, MPI_CHAR, root_rank, MPI_COMM_WORLD, &request);
        }
        MPI_Wait(&request, &status);
        if (getRootInfo != HCCL_SUCCESS) {
            printf("Process %d HcclGetRootInfo failed, notify other process to exit\n", root_rank);
            return getRootInfo;
        }
    } else {
        // 非根进程接收数据
        MPI_Ibcast(&comm_id, HCCL_ROOT_INFO_BYTES, MPI_CHAR, root_rank, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, &status);
        if (strcmp(reinterpret_cast<const char *>(&comm_id), "invalid") == 0) {
            // 检测到无效数据
            printf("Process %d received invalid data from root process %d\n", rank_id, root_rank);
            return -1;
        }
    }
    // 初始化集合通信域
    HCCLCHECK(HcclCommInitRootInfo(rank_size, &comm_id, rank_id, &hccl_comm));
    return 0;
}

int HcclTest::opbase_test_by_data_size()
{
    int ret = 0;
    for (data->data_size = data->min_bytes; data->data_size <= data->max_bytes;
         (data->step_factor <= 1.0 ? data->data_size += data->step_bytes : data->data_size *= data->step_factor)) {
        if (need_ranksize_alignment && data->data_size > BUF_ALGIN_LINE) {
            u64 algin_size = rank_size * BUF_ALGIN_SIZE;
            data->data_size = (data->data_size + algin_size - 1) / algin_size * algin_size;
        }

        size_t send_bytes = 0;
        size_t recv_bytes = 0;
        get_buff_size(send_bytes, recv_bytes);
        HCCLCHECK(static_cast<HcclResult>(prepare_zero_copy(send_bytes, recv_bytes)));
        HCCLCHECK(static_cast<HcclResult>(alloc_hccl_send_recv_buffer(send_buff, send_bytes, recv_buff, recv_bytes)));
        ret = hccl_op_base_test();
        HCCLCHECK(static_cast<HcclResult>(free_send_recv_buff_and_disable_local_buffer()));
        if (ret != 0) {
            printf("hccl_op_base execute failed, Detailed logs are stored in path: /root/ascend/log/\n");
            break;
        }
    }
    if (enable_zero_copy) {
        for (auto &&mem_handle : phy_alloc_mem_handle) {
            // printf("releasr virtual_ptr:0x%p, phy_ptr:0x%p\n",mem_handle.first, mem_handle.second);
            HCCLCHECK(HcclCommDeactivateCommMemory(hccl_comm, mem_handle.first));
            ACLCHECK(aclrtUnmapMem(mem_handle.first));
            ACLCHECK(aclrtFreePhysical(mem_handle.second));
        }
        phy_alloc_mem_handle.clear();
    }
    return ret;
}

int HcclTest::destory_hcclComm()
{
    // 销毁任务流
    ACLCHECK(aclrtDestroyStream(stream));
    ACLCHECK(aclrtDestroyEvent(start_event));
    ACLCHECK(aclrtDestroyEvent(end_event));
    // 重置设备
    ACLCHECK(aclrtResetDevice(dev_id));
    return 0;
}

void HcclTest::get_buff_size(size_t &send_bytes, size_t &recv_bytes)
{
    init_data_count();
    malloc_kSize = init_malloc_Ksize_by_data();
    init_send_recv_size_by_data(send_bytes, recv_bytes);
    return;
}

int HcclTest::prepare_zero_copy(const size_t &send_bytes, const size_t &recv_bytes)
{
    if (!enable_zero_copy || ((send_bytes + recv_bytes) == 0)) {
        return HCCL_SUCCESS;
    }
    static size_t current_alloc_phy_mem_byte = 0;
    auto malloc_mem = get_reseved_size(send_bytes, recv_bytes, physicalGranularity);
    if (current_alloc_phy_mem_byte >= malloc_mem) {
        return HCCL_SUCCESS;
    }
    //printf("current_alloc_phy_mem_byte:%zu, malloc_mem:%zu\n", current_alloc_phy_mem_byte, malloc_mem);
    int status = 0;
    aclrtDrvMemHandle mem_handle;
    status = aclrtMallocPhysical(&mem_handle, malloc_mem - current_alloc_phy_mem_byte, &prop, 0);
    if (status != ACL_SUCCESS) {
        printf("[%s][%d] aclrtMallocPhysical failed.\n", __FUNCTION__, __LINE__);
        goto enableerr0;
    }
    status = aclrtMapMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(vir_ptr) + current_alloc_phy_mem_byte),
        malloc_mem - current_alloc_phy_mem_byte,
        0,
        mem_handle,
        0);
    if (status != ACL_SUCCESS) {
        printf("[%s][%d] aclrtMapMem failed.\n", __FUNCTION__, __LINE__);
        goto enableerr0;
    }
    status = HcclCommActivateCommMemory(hccl_comm,
        reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(vir_ptr) + current_alloc_phy_mem_byte),
        malloc_mem - current_alloc_phy_mem_byte,
        0,
        mem_handle,
        0);
    if (status != 0) {
        printf("[%s][%d] HcclCommValidMemory failed.\n", __FUNCTION__, __LINE__);
        goto enableerr0;
    }
    phy_alloc_mem_handle.push_back(std::make_pair(
        reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(vir_ptr) + current_alloc_phy_mem_byte), mem_handle));
    // printf("alloic virtual_ptr:0x%p, phy_ptr:0x%p\n",phy_alloc_mem_handle.back().first,
    // phy_alloc_mem_handle.back().second);
    current_alloc_phy_mem_byte = malloc_mem;
    return 0;

enableerr0:
    for (auto &&mem_handle : phy_alloc_mem_handle) {
        // std::printf("release virtual_ptr:0x%p, phy_ptr:0x%p\n",mem_handle.first, mem_handle.second);
        HCCLCHECK(HcclCommDeactivateCommMemory(hccl_comm, mem_handle.first));
        ACLCHECK(aclrtUnmapMem(mem_handle.first));
        ACLCHECK(aclrtFreePhysical(mem_handle.second));
    }
    phy_alloc_mem_handle.clear();
    return status;
}

int HcclTest::alloc_hccl_send_recv_buffer(
    void *&send_buff, const size_t &send_bytes, void *&recv_buff, const size_t &recv_bytes)
{
    if (!enable_zero_copy) {
        // 申请集合通信操作的内存
        if (send_bytes) {
            ACLCHECK(aclrtMalloc((void **)&send_buff, send_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        if (recv_bytes) {
            ACLCHECK(aclrtMalloc((void **)&recv_buff, recv_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        }
    } else {
        // 2M对齐
        send_buff = vir_ptr;
        recv_buff = reinterpret_cast<void *>(
            reinterpret_cast<uintptr_t>(vir_ptr) +
            (((send_bytes + physicalGranularity - 1) / physicalGranularity) * physicalGranularity));
        //uintptr_t diff =
        //    (reinterpret_cast<uintptr_t>(send_buff) - reinterpret_cast<uintptr_t>(recv_buff)) % (2 * 1024 * 1024);
        //printf("recv_buff:%p, send_buff:%p, diff:%llu\n", recv_buff, send_buff, diff);
    }
    return 0;
}

int HcclTest::free_send_recv_buff_and_disable_local_buffer()
{
    if (!enable_zero_copy) {
        // 申请集合通信操作的内存
        if (send_buff) {
            ACLCHECK(aclrtFree(send_buff));
        }
        if (recv_buff) {
            ACLCHECK(aclrtFree(recv_buff));
        }
    }

    int ret = destory_alloc_buf();
    if (ret != 0) {
        printf("hccl_op_base destory_alloc_buf failed, ret[%d]", ret);
        return HCCL_E_MEMORY;
    }
    return HCCL_SUCCESS;
}
}  // namespace hccl
