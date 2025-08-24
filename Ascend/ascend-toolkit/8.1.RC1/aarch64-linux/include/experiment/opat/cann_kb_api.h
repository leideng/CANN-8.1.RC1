/**
 * @file cann_kb_api.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.\n
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n
 *
 */

#ifndef CANN_KB_API_H
#define CANN_KB_API_H

#include<map>
#include<string>
#include<vector>
#include "cann_kb_status.h"


namespace CannKb {
/**
 * @brief       : initialize cann knowledge bank service
 * @param [in]  : map<std::string, std::string> &sysConfig          sys config
                    sysConfig["soc_version"] = "Ascend310B1";
                    sysConfig["core_num"] = "1";
 * @param [in]  : map<std::string, std::string> &loadConfig         load config
 * @return      : == CANN_KB_SUCC : success, != CANN_KB_SUCC : failed
 */
extern "C" CANN_KB_STATUS CannKbInit(const std::map<std::string, std::string> &sysConfig,
    const std::map<std::string, std::string> &loadConfig);


/**
 * @brief       : Finalize Cann Knowledge Bank Service
 * @return      : == CANN_KB_SUCC : success, != CANN_KB_SUCC : failed
 */
extern "C" CANN_KB_STATUS CannKbFinalize();


/**
 * @brief       : Cann Knowledge Search
 * @param [in]  : string &infoDict                                        info dict
 * @param [in]  : map<std::string, std::string> &search_config             search config
                    search_config["op_type"] = "impl"
 * @param [in]  : std::vector<std::map<std::string, std::map<std::string, std::string>>> &search_result
                    search_result[0]["knowledge"] = {"dynamic_compile_static":"True", "op_impl":"tik"}
 * @return      : == CANN_KB_SUCC : success, != CANN_KB_SUCC : failed
 */
extern "C" CANN_KB_STATUS CannKbSearch(const std::string &infoDict,
    const std::map<std::string, std::string> &searchConfig,
    std::vector<std::map<std::string, std::string>> &searchResult);


/**
 * @brief       : Cann Knowledge write
 * @param [in]  : std::string &infoDict                                   info dict
 * @param [in]  : std::string &knowledge                                   knowledge
 * @param [in]  : std::map<std::string, std::string> &writeConfig         write config
 * @param [in]  : bool &flush                                              flush
 * @return      : == AOE_SUCCESS : success, != AOE_SUCCESS : failed
 */
extern "C" CANN_KB_STATUS CannKbWrite(const std::string &infoDict, const std::string &knowledge,
    const std::map<std::string, std::string> &writeConfig, const bool &flush);


/**
 * @brief       : Cann Knowledge delete
 * @param [in]  : std::string &infoDict                                   info dict
 * @param [in]  : std::map<std::string, std::string> &deleteConfig        delete config
 * @param [in]  : bool &flush                                              flush
 * @return      : == AOE_SUCCESS : success, != AOE_SUCCESS : failed
 */
extern "C" CANN_KB_STATUS CannKbDelete(const std::string &infoDict,
    const std::map<std::string, std::string> &deleteConfig, const bool &flush);

}  // namespace CannKb


#endif
