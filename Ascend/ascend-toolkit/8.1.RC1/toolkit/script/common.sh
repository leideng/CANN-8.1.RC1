#!/bin/bash
install_info_key_array=("Install_Type" "Chip_Type" "Feature_Type" "UserName" "UserGroup" "Install_Path_Param" "Docker_Root_Path_Param")
install_feature_type_arr=("all" "comm" "ascendc")

updateInstallParam() {
    local _key=$1
    local _val=$2
    local _file=$3
    local _param

    if [ ! -f "${_file}" ]; then
        exit 1
    fi

    for key_param in "${install_info_key_array[@]}"; do
        if [ ${key_param} == ${_key} ]; then
            _param=`grep -r "${_key}=" "${_file}"`
            if [ "x${_param}" = "x" ]; then
                echo "${_key}=${_val}" >> "${_file}"
            else
                sed -i "/^${_key}=/c ${_key}=${_val}" "${_file}"
            fi
            break
        fi
    done
}

getInstallParam() {
    local _key=$1
    local _file=$2
    local _param

    if [ ! -f "${_file}" ];then
        exit 1
    fi

    for key_param in "${install_info_key_array[@]}"; do
        if [ ${key_param} == ${_key} ]; then
            _param=`grep -r "${_key}=" "${_file}" | cut -d"=" -f2-`
            break;
        fi
    done
    echo "${_param}"
}

checkFeature() {
    local _feature_type=$1
    local _full_feature_arr=`echo $@ | cut -d" " -f2-`

    if [ -z ${_feature_type} ]; then
        return 1
    fi

    for type in ${_full_feature_arr[@]}; do
        if [ ${type} = ${_feature_type} ]; then
            return 0
        fi
    done
    return 1
}

checkFeatureType() {
    local _feature_type=$1

    local _feature_type_arr=`echo ${_feature_type} | tr -s "," " "`
    for type in ${_feature_type_arr[@]}; do
        checkFeature ${type} ${install_feature_type_arr[@]}
        [ $? -ne 0 ] && return 1
    done
    return 0
}

checkAscendcFeature() {
    local _feature_type=$1
    local _op_feature=("comm" "ascendc" "stress_detect")

    local _feature_type_arr=`echo ${_feature_type} | tr -s "," " "`
    for type in ${_feature_type_arr[@]}; do
        checkFeature ${type} ${_op_feature[@]}
        if [ $? -eq 0 ]; then
            return 0
        fi
    done
    return 1
}

checkAllFeature() {
    local _feature_type=$1
    local _all_feature=("all")

    local _feature_type_arr=`echo ${_feature_type} | tr -s "," " "`
    for type in ${_feature_type_arr[@]}; do
        checkFeature ${type} ${_all_feature[@]}
        if [ $? -eq 0 ]; then
            return 0
        fi
    done
    return 1
}

# check username and usergroup relationship
checkGroup(){
    _ugroup="$1"
    _uname_param="$2"
    if [ $(groups "${_uname_param}" | grep "${_uname_param} :" -c) -eq 1 ]; then
        _related=$(groups "${_uname_param}" 2> /dev/null |awk -F":" '{print $2}'|grep -w "${_ugroup}")
    else
        _related=$(groups "${_uname_param}" 2> /dev/null |grep -w "${_ugroup}")
    fi
    if [ "${_related}" != "" ];then
        return 0
    else
        return 1
    fi
}

changeMode() {
    local _mode=$1
    local _path=$2
    local _type=$3
    if [ ! x"${install_for_all}" = "x" ] && [ ${install_for_all} = y ]; then
        _mode="$(expr substr $_mode 1 2)$(expr substr $_mode 2 1)"
    fi
    if [ "${_type}" = "dir" ]; then
        find "${_path}" -type d -exec chmod ${_mode} {} \; 2> /dev/null
    elif [ "${_type}" = "file" ]; then
        find "${_path}" -type f -exec chmod ${_mode} {} \; 2> /dev/null
    fi
}

changeFileMode() {
    local _mode=$1
    local _path=$2
    changeMode ${_mode} "${_path}" file
}

changeDirMode() {
    local _mode=$1
    local _path=$2
    changeMode ${_mode} "${_path}" dir
}

createFile() {
    local _file=$1

    if [ ! -f "${_file}" ]; then
        touch "${_file}"
        if [ ! -f "${_file}" ]; then
            return 1
        fi
    fi

    chown -hf "$2" "${_file}"
    changeFileMode "$3" "${_file}"
    if [ $? -ne 0 ]; then
        return 1
    fi
    return 0
}

createFolder() {
    local _path=$1

    _path=`echo ${_path} | sed "s/\/*$//g"`
    if [ -z "${_path}" ]; then
        return 1
    fi

    if [ ! -d "${_path}" ]; then
        mkdir -p "${_path}"
        if [ ! -d "${_path}" ]; then
            return 1
        fi
    fi

    chown -hf "$2" "${_path}"
    changeDirMode "$3" "${_path}"
    if [ $? -ne 0 ]; then
        return 1
    fi
    return 0
}

# check directory is empty
isDirEmpty() {
    local _path=$1
    local _file_num

    if [ ! -d "${_path}" ]; then
        return 1
    fi

    _file_num=`ls "${_path}" | wc -l`
    if [ ${_file_num} -eq 0 ]; then
        return 0
    fi
    return 1
}

readShellInfo() {
    local shell_info=$1
    local start_flag=$2
    local end_flag=$3
    local match_state=0
    local shell_array=()
    local item=0
    while read _line; do
        if [ x"${_line}" == "x" ]; then
            continue
        fi
        if [ ${_line} == ${start_flag} ]; then
            match_state=1
            continue
        fi
        if [ $match_state -eq 0 ]; then
            continue
        fi
        if [ ${_line} == ${end_flag} ]; then
            break
        fi
        echo "${_line}" | grep -n '\.sh$' > /dev/null 2>&1
        if [ $? -ne 0 ]; then
            continue
        fi
        shell_array[${item}]="${_line}"
        let item+=1
    done < "$shell_info"
    echo ${shell_array[@]}
}

megerConfig() {
    local _old_file=$1
    local _new_file=$2
    local _tmp

    if [ "${_old_file}" == "${_new_file}" ]; then
        return 1
    fi
    if [ ! -f "${_old_file}" ] || [ ! -f "${_new_file}" ]; then
        return 1
    fi

    diff "${_old_file}" "${_new_file}" > /dev/null
    if [ $? -eq 0 ]; then
        return 0 # old file content equal new file content
    fi

    # check file permission
    if [ ! -r "${_old_file}" ] && [ ! -w "${_new_file}" ]; then
        return 1
    fi

    cp -f "${_old_file}" "${_old_file}.old"
    cp -f "${_new_file}" "${_old_file}"
    _new_file="${_old_file}"
    _old_file="${_old_file}.old"

    while read _line; do
        _tmp=`echo "${_line}" | sed "s/ //g"`
        if [ x"${_tmp}" == "x" ]; then
            continue # null line
        fi
        _tmp=`echo "${_tmp}" | cut -d"#" -f1`
        if [ x"${_tmp}" == "x" ]; then
            continue # the line is comment
        fi

        _tmp=`echo "${_line}" | grep "="`
        if [ x"${_tmp}" == "x" ]; then
            continue
        fi

        local _key=`echo "${_line}" | cut -d"=" -f1`
        local _value=`echo "${_line}" | cut -d"=" -f2-`
        if [ x"${_key}" == "x" ]; then
            echo "the config format is unrecognized, line=${_line}"
            continue
        fi
        if [ x"${_value}" == "x" ]; then
            continue
        fi
        # replace config value to new file
        sed -i "/^${_key}=/c ${_key}=${_value}" "${_new_file}"
    done < "${_old_file}"

    rm -f "${_old_file}"
}
