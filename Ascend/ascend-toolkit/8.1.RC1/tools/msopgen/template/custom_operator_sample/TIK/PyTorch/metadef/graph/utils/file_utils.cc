/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "graph/utils/file_utils.h"

#include <cerrno>
#include <fstream>
#include "graph/types.h"
#include "graph/debug/ge_log.h"
#include "mmpa/mmpa_api.h"
#include "graph/def_types.h"
#include "common/checker.h"
#include "debug/ge_util.h"

namespace {
  const int32_t kFileSuccess = 0;
  const uint32_t kMaxWriteSize = 1 * 1024 * 1024 * 1024U; // 1G
  constexpr const ge::char_t *kAscendWorkPathEnvName = "ASCEND_WORK_PATH";
}
namespace ge {
std::string RealPath(const char_t *path) {
  if (path == nullptr) {
    REPORT_INNER_ERROR("E18888", "path is nullptr, check invalid");
    GELOGE(FAILED, "[Check][Param] path pointer is NULL.");
    return "";
  }
  if (strnlen(path, static_cast<size_t>(MMPA_MAX_PATH)) >= static_cast<size_t>(MMPA_MAX_PATH)) {
    ErrorManager::GetInstance().ATCReportErrMessage("E19002", {"filepath", "size"},
                                                    {path, std::to_string(MMPA_MAX_PATH)});
    GELOGE(FAILED, "[Check][Param]Path[%s] len is too long, it must be less than %d", path, MMPA_MAX_PATH);
    return "";
  }

  // Nullptr is returned when the path does not exist or there is no permission
  // Return absolute path when path is accessible
  std::string res;
  char_t resolved_path[MMPA_MAX_PATH] = {};
  if (mmRealPath(path, &(resolved_path[0U]), MMPA_MAX_PATH) == EN_OK) {
    res = &(resolved_path[0]);
  } else {
    GELOGW("[Util][realpath] Can not get real_path for %s, reason:%s", path, strerror(errno));
  }
  return res;
}

std::string GetRegulatedName(const std::string name) {
  std::string regulate_name = name;
  replace(regulate_name.begin(), regulate_name.end(), '/', '_');
  replace(regulate_name.begin(), regulate_name.end(), '\\', '_');
  replace(regulate_name.begin(), regulate_name.end(), '.', '_');
  GELOGD("Get regulated name[%s] success", regulate_name.c_str());
  return regulate_name;
}

std::string GetSanitizedName(const std::string &input) {
  const std::string illegal_chars = "/\\:*?\"<>|";
  std::string sanitized;
  sanitized.reserve(input.size());
  for (char c : input) {
    if (illegal_chars.find(c) != std::string::npos) {
      sanitized += '_';
    } else {
      sanitized += c;
    }
  }
  GELOGD("Get regulated name[%s] from [%s]success", sanitized.c_str(), input.c_str());
  return sanitized;
}

inline int32_t CheckAndMkdir(const char_t *tmp_dir_path, mmMode_t mode) {
  if (mmAccess2(tmp_dir_path, M_F_OK) != EN_OK) {
    const int32_t ret = mmMkdir(tmp_dir_path, mode);
    if (ret != 0) {
      REPORT_CALL_ERROR("E18888",
                        "Can not create directory %s. Make sure the directory "
                        "exists and writable. errmsg:%s",
                        tmp_dir_path, strerror(errno));
      GELOGW(
          "[Util][mkdir] Create directory %s failed, reason:%s. Make sure the "
          "directory exists and writable.",
          tmp_dir_path, strerror(errno));
      return ret;
    }
  }
  return 0;
}

/**
 *  @ingroup domi_common
 *  @brief Create directory, support to create multi-level directory
 *  @param [in] directory_path  Path, can be multi-level directory
 *  @return -1 fail
 *  @return 0 success
 */
int32_t CreateDir(const std::string &directory_path, uint32_t mode) {
  GE_CHK_BOOL_EXEC(!directory_path.empty(),
                   REPORT_INNER_ERROR("E18888", "directory path is empty, check invalid");
                       return -1, "[Check][Param] directory path is empty.");
  const auto dir_path_len = directory_path.length();
  if (dir_path_len >= static_cast<size_t>(MMPA_MAX_PATH)) {
    ErrorManager::GetInstance().ATCReportErrMessage("E19002", {"filepath", "size"},
                                                    {directory_path, std::to_string(MMPA_MAX_PATH)});
    GELOGW("[Util][mkdir] Path %s len is too long, it must be less than %d", directory_path.c_str(), MMPA_MAX_PATH);
    return -1;
  }
  char_t tmp_dir_path[MMPA_MAX_PATH] = {};
  const auto mkdir_mode = static_cast<mmMode_t>(mode);
  for (size_t i = 0U; i < dir_path_len; i++) {
    tmp_dir_path[i] = directory_path[i];
    if ((tmp_dir_path[i] == '\\') || (tmp_dir_path[i] == '/')) {
      const int32_t ret = CheckAndMkdir(&(tmp_dir_path[0U]), mkdir_mode);
      if (ret != 0) {
        return ret;
      }
    }
  }
  return CheckAndMkdir(directory_path.c_str(), mkdir_mode);
}

/**
 *  @ingroup domi_common
 *  @brief Create directory, support to create multi-level directory
 *  @param [in] directory_path  Path, can be multi-level directory
 *  @return -1 fail
 *  @return 0 success
 */
int32_t CreateDir(const std::string &directory_path) {
  constexpr auto mkdir_mode = static_cast<uint32_t>(M_IRUSR | M_IWUSR | M_IXUSR);
  return CreateDir(directory_path, mkdir_mode);
}

/**
 *  @ingroup domi_common
 *  @brief Create directory, support to create multi-level directory
 *  @param [in] directory_path  Path, can be multi-level directory
 *  @return -1 fail
 *  @return 0 success
 */
int32_t CreateDirectory(const std::string &directory_path) {
  return CreateDir(directory_path);
}

std::unique_ptr<char_t[]> GetBinFromFile(std::string &path, uint32_t &data_len) {
  return GetBinDataFromFile(path, data_len);
}

std::unique_ptr<char_t[]> GetBinDataFromFile(const std::string &path, uint32_t &data_len) {
  GE_ASSERT_TRUE(!path.empty());

  std::ifstream ifs(path, std::ifstream::binary);
  if (!ifs.is_open()) {
    GELOGW("path:%s not open", path.c_str());
    return nullptr;
  }

  (void)ifs.seekg(0, std::ifstream::end);
  const uint32_t len = static_cast<uint32_t>(ifs.tellg());
  (void)ifs.seekg(0, std::ifstream::beg);
  auto bin_data = ComGraphMakeUnique<char_t[]>(len);
  if (bin_data == nullptr) {
    GELOGE(FAILED, "[Allocate][Mem]Allocate mem failed");
    ifs.close();
    return nullptr;
  }
  (void)ifs.read(reinterpret_cast<char_t*>(bin_data.get()), static_cast<std::streamsize>(len));
  data_len = len;
  ifs.close();
  return bin_data;
}

std::unique_ptr<char[]> GetBinFromFile(const std::string &path, size_t offset, size_t data_len) {
  const std::string real_path = RealPath(path.c_str());
  GE_ASSERT_TRUE(!real_path.empty(), "Failed to get real path of %s", path.c_str());
  std::ifstream ifs(real_path, std::ifstream::binary);
  GE_ASSERT_TRUE(ifs.is_open(), "Read file %s failed.", real_path.c_str());
  GE_MAKE_GUARD(close_ifs, [&ifs]() { ifs.close(); });
  (void)ifs.seekg(0, std::ifstream::end);
  const size_t act_file_len = static_cast<size_t>(ifs.tellg());
  GE_ASSERT_TRUE((offset <= (SIZE_MAX - data_len)) && ((offset + data_len) <= act_file_len),
                 "Offset add length overflow size_t or file length");
  ifs.clear();
  (void)ifs.seekg(static_cast<int64_t>(offset), std::ifstream::beg);
  auto bin_data = ComGraphMakeUnique<char_t[]>(data_len);
  GE_ASSERT_NOTNULL(bin_data, "[Allocate][Mem] Allocate mem failed");
  (void)ifs.read(reinterpret_cast<char_t *>(bin_data.get()), static_cast<int64_t>(data_len));
  ifs.close();
  return bin_data;
}

graphStatus GetBinFromFile(const std::string &path, char_t *buffer, size_t &data_len) {
  GE_ASSERT_TRUE(!path.empty());
  GE_ASSERT_TRUE(buffer != nullptr);
  std::string real_path = RealPath(path.c_str());
  GE_ASSERT_TRUE(!real_path.empty(),
      "Path: %s is invalid, file or directory is not exist", path.c_str());
  std::ifstream ifs(real_path, std::ifstream::binary);
  if (!ifs.is_open()) {
    GELOGE(GRAPH_FAILED, "path:%s not open", real_path.c_str());
    return GRAPH_FAILED;
  }

  (void)ifs.seekg(0, std::ifstream::end);
  const size_t len = static_cast<size_t>(ifs.tellg());
  (void)ifs.seekg(0, std::ifstream::beg);
  if (len != data_len) {
    REPORT_CALL_ERROR("E18888",
        "Bin length[%zu] is not equal to defined length[%zu], file_path[%s].",
        len, data_len, path.c_str());
    GELOGE(GRAPH_FAILED,
        "Bin length[%zu] is not equal to defined length[%zu], file_path[%s].",
        len, data_len, path.c_str());
    ifs.close();
    return GRAPH_FAILED;
  }
  (void)ifs.read(buffer, static_cast<std::streamsize>(len));
  ifs.close();
  return GRAPH_SUCCESS;
}

graphStatus WriteBinToFile(std::string &path, char_t *data, uint32_t &data_len) {
  GE_ASSERT_TRUE(!path.empty());
  std::ofstream ofs(path, std::ios::out | std::ifstream::binary);
  GE_ASSERT_TRUE(ofs.is_open(), "path:%s open failed", path.c_str());
  (void)ofs.write(data, static_cast<std::streamsize>(data_len));
  ofs.close();
  return GRAPH_SUCCESS;
}

graphStatus WriteBinToFile(const int32_t fd, const char_t * const data, size_t data_len) {
  if ((data == nullptr) || (data_len == 0UL)) {
    GELOGE(GRAPH_FAILED, "check param failed, data is nullptr or length is zero.");
    return GRAPH_FAILED;
  }
  int64_t write_count = 0;
  size_t remain_size = data_len;
  auto seek = static_cast<void *>(const_cast<char_t *>(data));
  do {
    size_t copy_size = remain_size > kMaxWriteSize ? kMaxWriteSize : remain_size;
    write_count = mmWrite(fd, seek, static_cast<uint32_t>(copy_size));
    GE_ASSERT_TRUE(((write_count != EN_INVALID_PARAM) && (write_count != EN_ERROR)),
        "Write data failed, data_len: %llu", data_len);
    seek = PtrAdd<uint8_t>(PtrToPtr<void, uint8_t>(seek), remain_size, static_cast<size_t>(write_count));
    remain_size -= static_cast<size_t>(write_count);
  } while (remain_size > 0U);
  return GRAPH_SUCCESS;
}

void SplitFilePath(const std::string &file_path, std::string &dir_path, std::string &file_name) {
  if (file_path.empty()) {
    GELOGD("file_path is empty, no need split");
    return;
  }
  int32_t split_pos = static_cast<int32_t>(file_path.length() - 1UL);
  for (; split_pos >= 0; split_pos--) {
    if ((file_path[static_cast<size_t>(split_pos)] == '\\') ||
        (file_path[static_cast<size_t>(split_pos)] == '/')) {
      break;
    }
  }
  if (split_pos < 0) {
    file_name = file_path;
    return;
  }
  dir_path = std::string(file_path).substr(0U, static_cast<size_t>(split_pos));
  file_name = std::string(file_path.substr(split_pos + 1U, file_path.length()));
  return;
}

graphStatus SaveBinToFile(const char * const data, size_t length, const std::string &file_path) {
  if (data == nullptr || length == 0UL) {
    GELOGE(GRAPH_FAILED, "check param failed, data is nullptr or length is zero.");
    return GRAPH_FAILED;
  }
  std::string dir_path;
  std::string file_name;
  SplitFilePath(file_path, dir_path, file_name);
  const bool meta_file_exist = (mmAccess(dir_path.c_str()) == EN_OK);
  if ((!dir_path.empty()) && (!meta_file_exist)) {
    GE_ASSERT_TRUE((CreateDir(dir_path) == kFileSuccess),
                   "Create direct failed, path: %s.", file_path.c_str());
  }
  std::string real_path = RealPath(dir_path.c_str());
  GE_ASSERT_TRUE(!real_path.empty(), "Path: %s is empty", file_path.c_str());
  real_path = real_path + "/" + file_name;
  // Open file
  const mmMode_t mode = static_cast<mmMode_t>(static_cast<uint32_t>(M_IRUSR) | static_cast<uint32_t>(M_IWUSR));
  const int32_t open_flag = static_cast<int32_t>(static_cast<uint32_t>(M_RDWR) |
                                                 static_cast<uint32_t>(M_CREAT) |
                                                 static_cast<uint32_t>(O_TRUNC));
  int32_t fd = mmOpen2(&real_path[0UL], open_flag, mode);
  GE_ASSERT_TRUE(((fd != EN_INVALID_PARAM) && (fd != EN_ERROR)), "Open file failed, path: %s", real_path.c_str());
  Status ret = GRAPH_SUCCESS;
  if (WriteBinToFile(fd, data, length) != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "Write data to file: %s failed.", real_path.c_str());
    ret = GRAPH_FAILED;
  }
  if (mmClose(fd) != 0) { // mmClose:0 success
    GELOGE(GRAPH_FAILED, "Close file failed.");
    return GRAPH_FAILED;
  }
  return ret;
}

Status GetAscendWorkPath(std::string &ascend_work_path) {
  char_t work_path[MMPA_MAX_PATH] = {'\0'};
  const int32_t work_path_ret = mmGetEnv(kAscendWorkPathEnvName, work_path, MMPA_MAX_PATH);
  if (work_path_ret == EN_OK) {
    if (mmAccess(work_path) != EN_OK) {
      if (ge::CreateDir(work_path) != 0) {
        std::string reason = "The path doesn't exist, create path failed.";
        REPORT_INPUT_ERROR("E10001", std::vector<std::string>({"parameter", "value", "reason"}),
                           std::vector<std::string>({kAscendWorkPathEnvName, work_path, reason}));
        return FAILED;
      }
    }
    ascend_work_path = RealPath(work_path);
    if (ascend_work_path.empty()) {
      GELOGE(FAILED, "[Call][RealPath] File path %s is invalid.", work_path);
      return FAILED;
    }
    GELOGD("Get ASCEND_WORK_PATH success, path = %s, real path = %s", work_path, ascend_work_path.c_str());
    return SUCCESS;
  }
  ascend_work_path = "";
  GELOGD("Get ASCEND_WORK_PATH fail");
  return SUCCESS;
}

int32_t Scandir(const CHAR *path, mmDirent ***entry_list, mmFilter filter_func, mmSort sort) {
  const auto count = mmScandir(path, entry_list, filter_func, sort);
  if ((count < EN_OK) || (entry_list == nullptr) || (*entry_list == nullptr)) {
    return count;
  }
  for (size_t i = 0; i < static_cast<size_t>(count); ++i) {
    mmDirent *const dir_ent = (*entry_list)[i];
    if (dir_ent != nullptr) {
      std::string dir_path = std::string(path) + "/" + std::string(dir_ent->d_name);
      if ((static_cast<int32_t>(dir_ent->d_type) == DT_UNKNOWN) && (mmIsDir(dir_path.c_str()) == EN_OK)) {
        dir_ent->d_type = DT_DIR;
      }
    }
  }
  return count;
}
}
