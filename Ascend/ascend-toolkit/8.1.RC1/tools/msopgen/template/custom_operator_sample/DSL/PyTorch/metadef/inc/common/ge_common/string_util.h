/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_COMMON_STRING_UTIL_H_
#define INC_COMMON_STRING_UTIL_H_

#include <cctype>
#include <securec.h>

#include <algorithm>
#include <functional>
#include <sstream>
#include <string>
#include <vector>
#include "graph/types.h"
#include "external/ge_common/ge_error_codes.h"

namespace ge {
class GE_FUNC_VISIBILITY StringUtils {
 public:
  static std::string &Ltrim(std::string &s) {
#if __cplusplus >= 201103L
    (void)s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                                          [](const int32_t c) { return std::isspace(c) == 0; }));
#else
    (void)s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                                          std::not1(std::ptr_fun<int32_t, int32_t>(std::isspace))));
#endif
    return s;
  }
  // lint -esym(551,*)
  static std::string &Rtrim(std::string &s) {  /*lint !e618*/
#if __cplusplus >= 201103L
    (void)s.erase(std::find_if(s.rbegin(), s.rend(),
                               [](const int32_t c) { return std::isspace(c) == 0; }).base(), s.end());
#else
    (void)s.erase(std::find_if(s.rbegin(), s.rend(),
                               std::not1(std::ptr_fun<int32_t, int32_t>(std::isspace))).base(), s.end());
#endif
    return s;
  }
  // lint -esym(551,*)
  ///
  ///  @ingroup domi_common
  ///  @brief delete spaces at the beginning and end of a string
  ///  @param [in] string to be trimmed
  ///  @return string after trim
  ///
  static std::string &Trim(std::string &s) { return Ltrim(Rtrim(s)); }

  ///
  ///  @ingroup domi_common
  ///  @brief string splitting
  ///  @param [in] str string to be trimmed
  ///  @param [in] delim  separator
  ///  @return string array after segmentation
  ///
  static std::vector<std::string, std::allocator<std::string>> Split(const std::string &str, const char_t delim) {
    std::vector<std::string, std::allocator<std::string>> elems;

    if (str.empty()) {
      (void) elems.emplace_back("");
      return elems;
    }

    std::stringstream ss(str);
    std::string item;

    while (getline(ss, item, delim)) {
      (void) elems.push_back(item);
    }

    const auto str_size = str.size();
    if ((str_size > 0U) && (str[str_size - 1U] == delim)) {
      (void) elems.emplace_back("");
    }

    return elems;
  }

  template<typename Iterator>
  static std::string Join(Iterator begin, Iterator end, const std::string &separator) {
    if (begin == end) {
      return "";
    }
    std::stringstream str_stream;
    str_stream << *begin;
    for (Iterator it = std::next(begin); it != end; ++it) {
      str_stream << separator << *it;
    }
    return str_stream.str();
  }

  ///
  ///  @ingroup domi_common
  ///  @brief obtain the file name
  ///  @param [in] s path name
  ///  @return file name
  ///
  static std::string GetFileName(const std::string &s) {
    if (s.empty()) {
      return "";
    }
    const std::vector<std::string> files = StringUtils::Split(s, '/');

    return files.empty() ? "" : files[files.size() - 1U];
  }
  ///
  ///  @ingroup domi_common
  ///  @brief full replacement
  ///  @link
  ///  @param [in] str str string to be replaced
  ///  @param [in] old_value  old Characters Before Replacement
  ///  @param [in] new_value  new Characters Before Replacement
  ///  @return string after replacement
  ///
  static std::string ReplaceAll(std::string str, const std::string &old_value, const std::string &new_value) {
    std::string::size_type cur_pos = 0U;
    const std::string::size_type old_length = old_value.length();
    const std::string::size_type new_length = new_value.length();
    // cycle replace
    for (; cur_pos != std::string::npos; cur_pos += new_length) {
      cur_pos = str.find(old_value, cur_pos);
      if (cur_pos != std::string::npos) {
        (void)str.replace(cur_pos, old_length, new_value);
      } else {
        break;
      }
    }
    return str;
  }

  ///
  ///  @ingroup domi_common
  ///  @brief checks whether a character string starts with a character string (prefix)
  ///  @link
  ///  @param [in] str string to be compared
  ///  @param [in] str_x prefix
  ///  @return if the value is a prefix, true is returned. Otherwise, false is returned
  ///
  static bool StartWith(const std::string &str, const std::string &str_x) {
    return ((str.size() >= str_x.size()) && (str.compare(0U, str_x.size(), str_x) == 0));
  }

  ///
  ///  @ingroup domi_common
  ///  @brief format string
  ///  @link
  ///  @param [in] format specifies the character string format
  ///  @param [in] ... format Filling Content
  ///  @return formatted string
  ///
  static std::string FormatString(const char_t * const format, ...) {
    const uint32_t MAX_BUFFER_LEN = 1024U;  // the stack memory plint check result must be less than 1024
    va_list args;
    va_start(args, format);
    char_t buffer[MAX_BUFFER_LEN] = {};
    const int32_t ret = vsnprintf_s(&buffer[0], MAX_BUFFER_LEN, MAX_BUFFER_LEN - 1U, format, args);
    va_end(args);
    return (ret > 0) ? buffer : "";
  }

  ///
  ///  @ingroup domi_common
  ///  @brief check whether the input str is signed int32
  ///  @link
  ///  @param [in] str input string
  ///  @return true/false
  ///
  static bool IsSignedInt32(const std::string &str) {
    if (str.empty()) {
      return false;
    }

    try {
      (void)std::stoi(str);
    } catch (...) {
      return false;
    }

    for (size_t i = 1; i < str.size(); i++) {
      if (isdigit(static_cast<int32_t>(str.at(i))) == 0) {
        return false;
      }
    }

    return true;
  }

  ///  @ingroup domi_common
  ///  @brief replace substring
  ///  @link
  ///  @param [in] str: string to be replaced
  ///  @param [in] to_replace: substring to be replaced
  ///  @param [in] replace_with: substring to replace with
  ///  @return void
  ///
  static void ReplaceSubstring(std::string &str, const std::string &to_replace, const std::string &replace_with) {
    if (to_replace.empty()) {
      return;
    }
    // 查找和替换所有的子串
    std::string::size_type pos = str.find(to_replace);
    while (pos != std::string::npos) {
      (void) str.replace(pos, to_replace.length(), replace_with);  // 用新的子串替换
      pos += replace_with.length();                        // 移动到替换后的位置继续查找
      pos = str.find(to_replace, pos);
    }
  }
};
}  // namespace ge

#endif  // INC_COMMON_STRING_UTIL_H_
