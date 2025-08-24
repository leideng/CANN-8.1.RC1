#!/bin/sh

# 内部函数
# 在指定路径下搜索文件
__find_filename_in_paths() {
    local paths="$1"
    local filename="$2"
    local out_varname="$3"
    local result
    local total_result=""

    for path in ${paths}; do
        if [ "${path}" != "" ] && [ -d "${path}" ]; then
            result="$(find -L "${path}" -name "${filename}" -print0 | tr '\0' ';')"
            total_result="${total_result}${result}"
        fi
    done
    # 移除末尾的;
    eval "${out_varname}=\"${total_result%%;}\""
}

# 搜索用户资产文件
# 搜索优先级为: 1.环境变量下的路径 2.本次安装路径 3.默认安装路径
# 即优先搜索环境变量下的路径，如果找不到，则再搜索本次安装路径，如果还找不到，则再搜索默认安装路径
# env_vars : 环境变量名列表（用:符号隔开的变量名）
# install_path : 本次安装路径
# default_install_path : 默认安装路径
# filenames : 用户资产文件名列表（用:符号隔开的文件名，注意必须是文件名，不可带目录前缀）
# out_varname[输出变量] : 搜索结果输出到的变量名，注意不能与本函数中的变量名相同
# 搜索的结果，设置到${out_varname}变量中，如果同一个文件名搜索到多个路径，则不同的路径使用;（分号）分隔。
# 如果入参filenames为多个文件名，则搜索的结果中，不同文件的搜索结果以:（冒号）分隔。
#
# [例1]
# 系统中存在如下文件
# /usr/local/Ascend/user1/asset01.txt
# 系统环境变量 ASCEND_HOME 设置为 /usr/local/Ascend
# 调用: search_user_assets "ASCEND_HOME" "" "" "asset01.txt" "output"
# 则最后output变量中的值为 /usr/local/Ascend/user1/asset01.txt
#
# [例2]
# 系统中存在如下文件
# /usr/local/Ascend/user1/asset02.txt
# /usr/local/Ascend/user2/asset01.txt
# 系统环境变量 ASCEND_HOME 设置为 /usr/local/Ascend/user1
# 调用: search_user_assets "ASCEND_HOME" "/usr/local/Ascend/user2" "" "asset01.txt" "output"
# 则最后output变量中的值为 /usr/local/Ascend/user2/asset01.txt
# 注: 环境变量路径中找不到 asset01.txt 文件，则再搜索安装路径。
#
# [例3]
# 系统中存在如下文件
# /usr/local/Ascend/user1/asset02.txt
# /usr/local/Ascend/user2/asset01.txt
# 调用: search_user_assets "" "/usr/local/Ascend/user1" "/usr/local/Ascend/user2" "asset01.txt" "output"
# 则最后output变量中的值为 /usr/local/Ascend/user2/asset01.txt
# 注: 安装路径下找不到 asset01.txt 文件，则再搜索默认安装路径。
#
# [例4]
# 系统中存在如下文件
# /usr/local/Ascend/user1/asset01.txt
# /usr/local/Ascend/user2/asset01.txt
# /usr/local/Ascend/user2/asset03.txt
# 调用: search_user_assets "" "/usr/local/Ascend" "" "asset01.txt:asset02.txt:asset03.txt" "output"
# 则最后output变量中的值为 /usr/local/Ascend/user2/asset01.txt;/usr/local/Ascend/user1/asset01.txt::/usr/local/Ascend/user2/asset03.txt
# 注:
# 1. 搜索路径下存在多个asset01.txt，所以结果中asset02.txt的搜索结果为;分隔的字符串。各文件路径出现的顺序不保证。
# 2. 搜索路径下不存在asset02.txt，所以结果中asset02.txt的搜索结果为空。
#
# [例5]
# 系统中存在如下文件
# /usr/local/Ascend/user1/asset01.txt
# /usr/local/Ascend/user2/asset01.txt
# 系统环境变量 ASCEND_HOME1 设置为 /usr/local/Ascend/user1
# 系统环境变量 ASCEND_HOME2 设置为 /usr/local/Ascend/user2
# 调用: search_user_assets "ASCEND_HOME1:ASCEND_HOME2" "" "" "asset01.txt" "output"
# 则最后output变量中的值为 /usr/local/Ascend/user1/asset01.txt;/usr/local/Ascend/user2/asset01.txt
# 注: 对于同一个文件，设置搜索的环境变量会依次搜索，取结果的并集。
search_user_assets() {
    local env_vars="$1"
    local install_path="$2"
    local default_install_path="$3"
    local filenames="$4"
    local out_varname="$5"
    # 使用短变量名，防止与外部变量名冲突
    local r # 搜索结果
    local e # 环境变量解析值
    local ep # 环境变量解析出的路径列表
    local f # 迭代文件名
    local o # 输出结果
    local i # 迭代环境变量名

    OLD_IFS="${IFS}"
    IFS=":"
    for f in ${filenames}; do
        # 跳过空文件名
        if [ -z "${f}" ]; then
            o="${o}:"
            continue
        fi

        if [ "${env_vars}" != "" ]; then
            ep=""
            for i in ${env_vars}; do
                # 跳过空环境变量
                if [ -z "${i}" ]; then
                    continue
                fi

                eval e="\${${i}}"
                ep="${ep}${e}:"
            done

            __find_filename_in_paths "${ep%%:}" "${f}" "r"
            if [ "${r}" != "" ]; then
                o="${o}${r}:"
                continue
            fi
        fi

        __find_filename_in_paths "${install_path}" "${f}" "r"

        if [ "${r}" != "" ]; then
            o="${o}${r}:"
            continue
        fi

        __find_filename_in_paths "${default_install_path}" "${f}" "r"

        if [ "${r}" != "" ]; then
            o="${o}${r}:"
            continue
        fi

        # 没有找到该文件
        o="${o}:"
    done
    eval "${out_varname}=\"${o%%:}\""

    return 0
}
