/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: IP Address Resource Management
 */

#ifndef HCCL_IP_ADDRESS_H
#define HCCL_IP_ADDRESS_H

#include <securec.h>
#include <arpa/inet.h>
#include <hccl/base.h>


namespace hccl {
constexpr u32 IP_ADDRESS_BUFFER_LEN = 64;
struct HcclSocketInfo {
    void *socketHandle; /**< socket handle */
    void *fdHandle; /**< fd handle */
};

union HcclInAddr {
    struct in_addr addr;
    struct in6_addr addr6;
};

class HcclIpAddress {
public:
    explicit HcclIpAddress()
    {
        scopeID = 0;
        family = AF_INET;
        binaryAddr.addr.s_addr = 0;
        readableIP = "0.0.0.0";
        readableAddr = readableIP;
    }
    explicit HcclIpAddress(u32 address)
    {
        union HcclInAddr ipAddr;
        ipAddr.addr.s_addr = address;
        (void)SetBianryAddress(AF_INET, ipAddr);
    }
    explicit HcclIpAddress(s32 family, const union HcclInAddr &address)
    {
        (void)SetBianryAddress(family, address);
    }
    explicit HcclIpAddress(const struct in_addr &address)
    {
        union HcclInAddr ipAddr;
        ipAddr.addr = address;
        (void)SetBianryAddress(AF_INET, ipAddr);
    }
    explicit HcclIpAddress(const struct in6_addr &address)
    {
        union HcclInAddr ipAddr;
        ipAddr.addr6 = address;
        (void)SetBianryAddress(AF_INET6, ipAddr);
    }
    explicit HcclIpAddress(const std::string &address)
    {
        (void)SetReadableAddress(address);
    }
    ~HcclIpAddress() {}

    std::string GetIfName() const
    {
        return ifname;
    }
    HcclResult SetScopeID(s32 scopeID)
    {
        this->scopeID = scopeID;
        return HCCL_SUCCESS;
    }
    s32 GetScopeID() const
    {
        return scopeID;
    }
    s32 GetFamily() const
    {
        return family;
    }

    const char *GetReadableIP() const
    {
        // return "IP adddress (string)"
        return readableIP.c_str();
    }
    const char *GetReadableAddress() const
    {
        // return "IP adddress (string) % ifname"
        return readableAddr.c_str();
    }
    union HcclInAddr GetBinaryAddress() const
    {
        return binaryAddr;
    }
    bool IsIPv6() const
    {
        return (family == AF_INET6);
    }
    void clear()
    {
        family = AF_INET;
        scopeID = 0;
        binaryAddr.addr.s_addr = 0;
        readableAddr.clear();
        readableIP.clear();
        ifname.clear();
    }
    bool IsInvalid() const
    {
        return ((family == AF_INET) && (binaryAddr.addr.s_addr == 0));
    }
    bool operator == (const HcclIpAddress &that) const
    {
        if (this->family != that.family) {
            return false;
        }
        if (this->family == AF_INET) {
            return (this->binaryAddr.addr.s_addr == that.binaryAddr.addr.s_addr);
        } else {
            if (memcmp(&this->binaryAddr.addr6, &that.binaryAddr.addr6, sizeof(this->binaryAddr.addr6)) != 0) {
                return false;
            } else {
                if (this->ifname.empty() || that.ifname.empty()) {
                    return true;
                } else {
                    return (this->ifname == that.ifname);
                }
            }
        }
    }
    bool operator != (const HcclIpAddress &that) const
    {
        return !(*this == that);
    }
    bool operator < (const HcclIpAddress &that) const
    {
        if (this->family < that.family) {
            return true;
        }
        if (this->family > that.family) {
            return false;
        }
        return (this->family == AF_INET) ? (this->binaryAddr.addr.s_addr < that.binaryAddr.addr.s_addr) :
                                           (this->readableAddr < that.readableAddr);
    }

    HcclResult SetReadableAddress(const std::string &address);
    HcclResult SetIfName(const std::string &name);

private:
    HcclResult SetBianryAddress(s32 family, const union HcclInAddr &address);

    union HcclInAddr binaryAddr{};   // 二进制IP地址
    std::string readableAddr{};      // 字符串IP地址 + % + 网卡名
    std::string readableIP{};        // 字符串IP地址
    std::string ifname{};            // 网卡名
    s32 family{};
    s32 scopeID{};
};
}
#endif // HCCL_IP_ADDRESS_H
