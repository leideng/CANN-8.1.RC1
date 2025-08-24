#!/usr/bin/env csh

set PYTHON_VERSION=3.7.5
set SHELL_PATH=`readlink -f $0`
set SHELL_DIR=`dirname ${SHELL_PATH}`

set python_version=`python3 --version | cut -d" " -f2`
foreach num (1 2 3)
    set version_num=`echo ${python_version} | cut -d"." -f${num}`
    set min_version_num=`echo ${PYTHON_VERSION} | cut -d"." -f${num}`
    if (${version_num} < ${min_version_num}) then
        echo "python version ${python_version} low than ${PYTHON_VERSION}."
        break
    endif
    if (${version_num} > ${min_version_num}) then
        break
    endif
end

set python_module_infos=(sqlite3:)
set install_info="${SHELL_DIR}/../ascend_install.info"
if (-f "${install_info}") then
    set feature_type=`cat ${install_info} | grep "Feature_Type"`
    set feature_type_matched=`echo ${feature_type} | grep -Eo "op|all|model"`
    if ("${feature_type_matched}" != "") then
        set python_module_infos=(${python_module_infos} numpy:1.13.3)
    endif
    set feature_type_matched=`echo ${feature_type} | grep -Eo "op|all"`
    if ("${feature_type_matched}" != "") then
        set python_module_infos=(${python_module_infos} scipy:1.4.1 psutil:5.7.0 protobuf:3.13.0)
    endif
endif

foreach module_info (${python_module_infos})
    set module=`echo ${module_info} | cut -d":" -f1`
    if ("${module}" == "protobuf") then
        python3 -c "import google.protobuf" >&/dev/null
        set check_result=${status}
    else
        python3 -c "import ${module}" >&/dev/null
        set check_result=${status}
    endif
    if (${check_result} != 0) then
        echo "python module ${module} doesn't exist."
        continue
    endif

    set min_version=`echo ${module_info} | cut -d":" -f2`
    if ("${min_version}" == "") then
        continue
    endif

    set installed_version=`pip3 list | &grep "^${module} " | cut -d" " -f2- | grep -Eo "[0-9.]+"`
    if ("${installed_version}" == "") then
        echo "search $module version failed."
        continue
    endif
    set temp_installed_version=$installed_version
    set min_version_arr=`echo ${min_version} | tr "." " "`
    foreach min_version_num (${min_version_arr})
        set installed_version_num=`echo ${temp_installed_version} | cut -d"." -f1`
        set temp_installed_version=`echo ${temp_installed_version} | cut -d"." -f2-`
        if (${installed_version_num} == "") then
            break
        endif
        if (${installed_version_num} < ${min_version_num}) then
            echo "python module ${module} version ${installed_version} low than ${min_version}."
            break
        endif
        if (${installed_version_num} > ${min_version_num}) then
            break
        endif
    end
end
