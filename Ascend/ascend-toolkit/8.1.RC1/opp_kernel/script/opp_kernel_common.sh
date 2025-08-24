#!/bin/bash

if [ $(id -u) -ne 0 ]; then
    log_dir="${HOME}/var/log/ascend_seclog"
    OPERATION_LOGDIR="${HOME}/var/log/ascend_seclog"
else
    log_dir="/var/log/ascend_seclog"
    OPERATION_LOGDIR="/var/log/ascend_seclog"
fi
logFile="${log_dir}/ascend_install.log"
OPERATION_LOGPATH="${OPERATION_LOGDIR}/operation.log"

log() {
    cur_date=`date +"%Y-%m-%d %H:%M:%S"`
    echo "[Opp_Kernel] [$cur_date] "$1 >> $logFile
}

logPrint() {
    cur_date=`date +"%Y-%m-%d %H:%M:%S"`
    echo "[Opp_Kernel] [$cur_date] ""$1"
}

logAndPrint() {
    cur_date=`date +"%Y-%m-%d %H:%M:%S"`
    echo "[Opp_Kernel] [$cur_date] ""$1"
    echo "[Opp_Kernel] [$cur_date] "$1 >> $logFile
}

colorPrint() {
    cur_date=`date +"%Y-%m-%d %H:%M:%S"`
    echo -e  "[Opp_Kernel] [$cur_date] $1"
}

remove_cann_ops(){
    local targetdir=$1
    if [[ -L "${targetdir}/ops" && -e "${targetdir}/ops" ]]; then
        logPrint "[WARNING]: ${targetdir} have content" 
        log "[WARNING]: ${targetdir} have content"  
    else  
        rm -rf ${targetdir}/ops > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            logPrint "[INFO]: Remove real ops softlinks successfully!"
            log "[INFO]: Remove real ops softlinks successfully!"
            local targetdir_sub=$(ls "${targetdir}/" 2> /dev/null)
            if [ "${targetdir_sub}x" = "x" ]; then
                rm -fr ${targetdir}
                if [ $? -eq 0 ]; then
                    logPrint "[INFO]: Remove ${targetdir} successfully!"
                    log "[INFO]: Remove ${targetdir} successfully!"
                else
                    logPrint "[WARNING]: Remove ${targetdir} failed!"
                    log "[WARNING]: Remove ${targetdir} failed!"
                fi
            fi
        else
            logPrint "[WARNING]: Remove ops softlinks failed!"
            log "[WARNING]: Remove ops softlinks failed!"
        fi  
    fi
}

createrelativelysoftlink() {
    local src_path_="$1"
    local dst_path_="$2"
    local dst_parent_path_=$(dirname ${dst_path_})
    # echo "dst_parent_path_: ${dst_parent_path_}"
    local relative_path_=$(realpath --relative-to="$dst_parent_path_" "$src_path_")
    # echo "relative_path_: ${relative_path_}"
    if [ -L "$2" ]; then
        logPrint "[WARNING]: Soft link for ops is existed. Cannot create new soft link."
        log "[WARNING]: Soft link for ops is existed. Cannot create new soft link."
        return 0
    fi
    ln -s "${relative_path_}" "${dst_path_}" 2> /dev/null
    if [ "$?" != "0" ]; then
        return 1
    else
        return 0
    fi
}
create_opapi_latest_softlink() {
    targetPkg=$2
    if [ "${targetPkg}x" = "x" ]; then
            targetPkg=opp_kernel
    fi
    osName=""
    if [ -f "$1/$targetPkg/scene.info" ]; then
        . $1/$targetPkg/scene.info
        osName=${os}
    fi
    opapi_lib_path="$1/opp/built-in/op_impl/ai_core/tbe/op_api/lib/${osName}/${architecture}"
    opapi_include_level1_path="$1/opp/built-in/op_impl/ai_core/tbe/op_api/include/aclnnop"
    opapi_include_level2_path="${opapi_include_level1_path}/level2"
    if [ ! -d  ${opapi_lib_path} ] || [ ! -d  ${opapi_include_level1_path} ] || [ ! -d  ${opapi_include_level2_path} ]; then
        return 3
    fi    
    if [ -d $(dirname $1)/latest/${architectureDir}/lib64 ]; then
        for file_so in $(ls -1 $1/${architectureDir}/lib64 | egrep "libaclnn_|libopapi.so")
        do
            latest_arch_lib64_src_path="$1/${architectureDir}/lib64/${file_so}"
            latest_arch_lib64_dst_path="$(dirname $1)/latest/${architectureDir}/lib64/${file_so}"
            if [ -f $latest_arch_lib64_dst_path ] || [ -L $latest_arch_lib64_dst_path ]; then
                rm -fr "$latest_arch_lib64_dst_path"
            fi
            createrelativelysoftlink ${latest_arch_lib64_src_path} ${latest_arch_lib64_dst_path}
        done
    fi
    

    # second the headfiles with 1 and 2 level
    if [ -d $1/${architectureDir}/include/aclnnop ]; then
        for file_level1 in $(ls -1 -F ${opapi_include_level1_path} | grep -v [/$] | sed 's/\*$//')
        do
            latest_arch_include_src_path="${opapi_include_level1_path}/${file_level1}"
            latest_arch_include_dst_path="$(dirname $1)/latest/${architectureDir}/include/aclnnop/${file_level1}"
            if [ -f $latest_arch_include_dst_path ] || [ -L $latest_arch_include_dst_path ]; then
                rm -fr "$latest_arch_include_dst_path"
            fi
            createrelativelysoftlink ${latest_arch_include_src_path} ${latest_arch_include_dst_path}
        done

    fi
    
    if [ -d $1/${architectureDir}/include/aclnnop/level2 ]; then
        for file_level2 in $(ls -1 -F ${opapi_include_level2_path} | grep -v [/$] | sed 's/\*$//')
        do  
            latest_arch_include_src_path="${opapi_include_level2_path}/${file_level2}"
            latest_arch_include_dst_path="$(dirname $1)/latest/${architectureDir}/include/aclnnop/level2/${file_level2}"
            if [ -f $latest_arch_include_dst_path ] || [ -L $latest_arch_include_dst_path ]; then
                rm -fr "$latest_arch_include_dst_path"
            fi
            createrelativelysoftlink ${latest_arch_include_src_path} ${latest_arch_include_dst_path}
        done
    fi
}
create_opapi_softlink() {
    targetPkg=$2
    if [ "${targetPkg}x" = "x" ]; then
            targetPkg=opp_kernel
    fi
    osName=""
    if [ -f "$1/$targetPkg/scene.info" ]; then
        . $1/$targetPkg/scene.info
        osName=${os}
    fi
    opapi_lib_path="$1/opp/built-in/op_impl/ai_core/tbe/op_api/lib/${osName}/${architecture}"
    opapi_include_level1_path="$1/opp/built-in/op_impl/ai_core/tbe/op_api/include/aclnnop"
    opapi_include_level2_path="${opapi_include_level1_path}/level2"

    if [ ! -d  ${opapi_lib_path} ] || [ ! -d  ${opapi_include_level1_path} ] || [ ! -d  ${opapi_include_level2_path} ]; then
        return 3
    fi    
    # first the libopapi.so
    if [ -d $1/${architectureDir}/lib64 ]; then
        for file_so in $(ls -1  ${opapi_lib_path} | grep "so"$)
        do
            arch_lib64_src_path="${opapi_lib_path}/${file_so}"
            arch_lib64_dst_path="$1/${architectureDir}/lib64/${file_so}"
            if [ -f $arch_lib64_dst_path ] || [ -L $arch_lib64_dst_path ]; then
                rm -fr "$arch_lib64_dst_path"
            fi
            createrelativelysoftlink ${arch_lib64_src_path} ${arch_lib64_dst_path}
        done
    fi
    
    if [ -d $1/opp/lib64 ]; then
        for file_so in $(ls -1 $1/${architectureDir}/lib64 | egrep "libaclnn_|libopapi.so")
        do
            opp_lib64_src_path="$1/${architectureDir}/lib64/${file_so}"
            opp_lib64_dst_path="$1/opp/lib64/${file_so}"
            if [ -f $opp_lib64_dst_path ] || [ -L $opp_lib64_dst_path ]; then
                rm -fr "$opp_lib64_dst_path"
            fi
            createrelativelysoftlink ${opp_lib64_src_path} ${opp_lib64_dst_path}
        done
    fi

    # second the headfiles with 1 and 2 level
    if [ -d $1/${architectureDir}/include/aclnnop ]; then
        for file_level1 in $(ls -1 -F ${opapi_include_level1_path} | grep -v [/$] | sed 's/\*$//')
        do
            arch_include_src_path="${opapi_include_level1_path}/${file_level1}"
            arch_include_dst_path="$1/${architectureDir}/include/aclnnop/${file_level1}"
            if [ -f $arch_include_dst_path ] || [ -L $arch_include_dst_path ]; then
                rm -fr "$arch_include_dst_path"
            fi
            createrelativelysoftlink ${arch_include_src_path} ${arch_include_dst_path}
    
            opp_include_src_path="${arch_include_dst_path}"
            opp_include_dst_path="$1/opp/include/aclnnop/${file_level1}"
            if [ -f $opp_include_dst_path ] || [ -L $opp_include_dst_path ]; then
                rm -fr "$opp_include_dst_path"
            fi
            createrelativelysoftlink ${opp_include_src_path} ${opp_include_dst_path}
        done
    fi
    
    if [ -d $1/${architectureDir}/include/aclnnop/level2 ]; then
        for file_level2 in $(ls -1 -F ${opapi_include_level2_path} | grep -v [/$] | sed 's/\*$//')
        do  
            arch_include_src_path="${opapi_include_level2_path}/${file_level2}"
            arch_include_dst_path="$1/${architectureDir}/include/aclnnop/level2/${file_level2}"
            if [ -f $arch_include_dst_path ] || [ -L $arch_include_dst_path ]; then
                rm -fr "$arch_include_dst_path"
            fi
            createrelativelysoftlink ${arch_include_src_path} ${arch_include_dst_path}
    
            opp_include_src_path="${arch_include_dst_path}"
            opp_include_dst_path="$1/opp/include/aclnnop/level2/${file_level2}"
            if [ -f $opp_include_dst_path ] || [ -L $opp_include_dst_path ]; then
                rm -fr "$opp_include_dst_path"
            fi
            createrelativelysoftlink ${opp_include_src_path} ${opp_include_dst_path}
        done
    fi
}

latest_softlinks_remove() {
    targetdir=$1
    osName=""
    if [ -f "$targetdir/opp_kernel/scene.info" ]; then
        . $targetdir/opp_kernel/scene.info
        osName=${os}
    fi
    opapi_lib_path="$targetdir/opp/built-in/op_impl/ai_core/tbe/op_api/lib/${osName}/${architecture}"
    opapi_include_level1_path="$1/opp/built-in/op_impl/ai_core/tbe/op_api/include/aclnnop"
    opapi_include_level2_path="${opapi_include_level1_path}/level2"

    if [ -d $(dirname $targetdir)/latest/${architectureDir}/lib64 ]; then
        for file_so in $(ls -l "$(dirname $targetdir)/latest/${architectureDir}/lib64/"  | egrep "libaclnn_|libopapi.so")
        do         
            latest_arch_lib64_path="$(dirname $targetdir)/latest/${architectureDir}/lib64/${file_so}"
            remove_opapi_softlink ${latest_arch_lib64_path}
        done
    fi
    
    # second the headfiles with 1 and 2 level
    if [ -d $(dirname $targetdir)/latest/${architectureDir}/include/aclnnop ]; then
        for file_level1 in $(ls -1 -F ${opapi_include_level1_path} | grep -v [/$] | sed 's/\*$//')
        do
            latest_arch_include_path="$(dirname $targetdir)/latest/${architectureDir}/include/aclnnop/${file_level1}"
            remove_opapi_softlink ${latest_arch_include_path}
        done
    fi

    if [ -d $(dirname $targetdir)/latest/${architectureDir}/include/aclnnop/level2 ]; then
        for file_level2 in $(ls -1 -F ${opapi_include_level2_path} | grep -v [/$] | sed 's/\*$//')
        do
            latest_arch_include_path="$(dirname $targetdir)/latest/${architectureDir}/include/aclnnop/level2/${file_level2}"
            remove_opapi_softlink ${latest_arch_include_path}
        done
    fi
}

remove_opapi_softlink() {
    path="$1"
    if [ -L "$1" ]; then
        rm -fr ${path}
        return 0
    else
        return 1
    fi
}
 
softlinks_remove() {
    targetdir=$1
    osName=""
    if [ -f "$targetdir/opp_kernel/scene.info" ]; then
        . $targetdir/opp_kernel/scene.info
        osName=${os}
    fi
    opapi_lib_path="$targetdir/opp/built-in/op_impl/ai_core/tbe/op_api/lib/${osName}/${architecture}"
    opapi_include_level1_path="$targetdir/opp/built-in/op_impl/ai_core/tbe/op_api/include/aclnnop"
    opapi_include_level2_path="${opapi_include_level1_path}/level2"

    # first the libopapi.so
    if [ -d $targetdir/${architectureDir}/lib64 ]; then
        for file_so in $(ls -1 $targetdir/${architectureDir}/lib64 | egrep "libaclnn_|libopapi.so")
        do
            arch_lib64_path="$targetdir/${architectureDir}/lib64/${file_so}"
            remove_opapi_softlink ${arch_lib64_path}
        done
    fi
    
    if [ -d $targetdir/opp/lib64 ]; then
        for file_so in $(ls -l $targetdir/opp/lib64 | egrep "libaclnn_|libopapi.so")
        do 
            opp_lib64_path="$targetdir/opp/lib64/${file_so}"
            remove_opapi_softlink ${opp_lib64_path}
        done
    fi
    
    # second the headfiles with 1 and 2 level
    if [ -d $targetdir/${architectureDir}/include/aclnnop ]; then
        for file_level1 in $(ls -1 -F ${opapi_include_level1_path} | grep -v [/$] | sed 's/\*$//')
        do
            arch_include_path="$targetdir/${architectureDir}/include/aclnnop/${file_level1}"
            remove_opapi_softlink ${arch_include_path}

            opp_include_path="$targetdir/opp/include/aclnnop/${file_level1}"
            remove_opapi_softlink ${opp_include_path}
        done
    fi

    if [ -d $targetdir/${architectureDir}/include/aclnnop/level2 ]; then
        for file_level2 in $(ls -1 -F ${opapi_include_level2_path} | grep -v [/$] | sed 's/\*$//')
        do  
            arch_include_path="$targetdir/${architectureDir}/include/aclnnop/level2/${file_level2}"
            remove_opapi_softlink ${arch_include_path}

            opp_include_path="$targetdir/opp/include/aclnnop/level2/${file_level2}"
            remove_opapi_softlink ${opp_include_path}
        done
    fi
}

deleteDiffernetArchs() {
    if [ -f "$1/opp/version.info" ]; then
        logPrint "[DEBUG]: Opp exists in the environment, do not delete different architectures."
        log "[DEBUG]: Opp exists in the environment, do not delete different architectures."        
        return 0
    fi
    osName=""
    if [ -f "$1/opp_kernel/scene.info" ]; then
        . $1/opp_kernel/scene.info
        osName=${os}
    fi
    op_proto_lib_base_path="$1/opp/built-in/op_proto/lib/${osName}"
    op_tiling_lib_base_path="$1/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/${osName}"
    if [ ! -d  ${op_proto_lib_base_path} ] || [ ! -d  ${op_tiling_lib_base_path} ]; then
        return 3
    fi
    op_proto_lib_base_path_mod_num=$(stat -c %a "${op_proto_lib_base_path}") > /dev/null 2>&1
    op_tiling_lib_base_path_mod_num=$(stat -c %a "${op_tiling_lib_base_path}") > /dev/null 2>&1
    chmod -R 755 "$op_proto_lib_base_path" > /dev/null 2>&1
    chmod -R 755 "$op_tiling_lib_base_path" > /dev/null 2>&1
    for op_proto_folder in ${op_proto_lib_base_path}/*; do
        if [ "${op_proto_folder##*/}" != "${architecture}" ]; then
          rm -rf "$op_proto_folder" > /dev/null 2>&1
        fi
    done
 
    for op_tiling_folder in ${op_tiling_lib_base_path}/*; do
        if [ "${op_tiling_folder##*/}" != "${architecture}" ]; then
          rm -rf "$op_tiling_folder" > /dev/null 2>&1
        fi
    done
    chmod -R ${op_proto_lib_base_path_mod_num} "$op_proto_lib_base_path" > /dev/null 2>&1
    chmod -R ${op_tiling_lib_base_path_mod_num} "$op_tiling_lib_base_path" > /dev/null 2>&1
}
