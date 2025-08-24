#!/bin/csh
set param_mult_ver = $argv[1]
set REAL_SHELL_PATH = `realpath $0`
set CANN_PATH = `cd $(dirname $REAL_SHELL_PATH)/../../ && pwd`
if (-d "$CANN_PATH/opp" && -d "$CANN_PATH/../latest") then
    set INSATLL_PATH = `cd $(dirname $REAL_SHELL_PATH)/../../../ && pwd`
    if (-L "$INSATLL_PATH/latest/opp") then
        set _ASCEND_OPP_PATH = `cd $CANN_PATH/opp && pwd`
        if ($param_mult_ver == "multi_version") then
            set _ASCEND_OPP_PATH = `cd $INSATLL_PATH/latest/opp && pwd`
        endif
    elseif (! -L "$INSATLL_PATH/latest/opp" && -L "$INSATLL_PATH/latest/opp_kernel") then
        set _ASCEND_OPP_PATH = `cd $CANN_PATH/opp && pwd`
    endif
elseif (-d "$CANN_PATH/opp") then
    set _ASCEND_OPP_PATH = `cd $CANN_PATH/opp && pwd`
endif

set _ASCEND_AICPU_PATH=`cd $_ASCEND_OPP_PATH/../;pwd`
setenv ASCEND_OPP_PATH ${_ASCEND_OPP_PATH}
setenv ASCEND_AICPU_PATH ${_ASCEND_AICPU_PATH}
if ( $?PYTHONPATH == 0 ) then
    setenv PYTHONPATH ${ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe/
else
    setenv PYTHONPATH ${PYTHONPATH}:${ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe/
endif
set CURFILE=`readlink -f {$1}`
set owner="`stat -c %U $CURFILE`"
set custom_path_file="$_ASCEND_OPP_PATH/../conf/path.cfg"
set common_interface="$_ASCEND_OPP_PATH/script/common_interface.csh"
set dst_dir="`grep -w "data" "$custom_path_file" | cut --only-delimited -d"=" -f2-`"
set dst_dir="`eval echo $dst_dir`"
set vendors_dir="$_ASCEND_OPP_PATH/vendors"
set vendor_dir=`ls $vendors_dir`
if ( "`id -u`" != 0 && "`id -un`" != "$owner" && -f "$custom_path_file" && -f "$common_interface" ) then
    csh -f "$common_interface" mk_custom_path "$custom_path_file"
    foreach dir_name ("data")
        if ( -d "$_ASCEND_OPP_PATH/built-in/$dir_name" && -d "$dst_dir" ) then
            chmod -R u+w $dst_dir/* >& /dev/null
            cp -rfL $_ASCEND_OPP_PATH/built-in/$dir_name/* "$dst_dir"
        endif
        if ( -d "$vendors_dir/$vendor_dir/$dir_name" && -d "$dst_dir" ) then
            chmod -R u+w $dst_dir/* >& /dev/null
            cp -rfL $vendors_dir/$vendor_dir/$dir_name/* "$dst_dir"
        else if ( ! -d "$vendors_dir/$vendor_dir/$dir_name" && -d "$dst_dir" && -d "$_ASCEND_OPP_PATH/$dir_name") then
            chmod -R u+w $dst_dir/* >& /dev/null
            cp -rfL $_ASCEND_OPP_PATH/$dir_name/* "$dst_dir"
        endif
    end
endif
if ( "`id -u`" != 0 && "`id -un`" != "$owner" ) then
    set opp_custom_list="op_impl op_proto fusion_pass fusion_rules framework"
    foreach i ($opp_custom_list)
        set dst_file=$dst_dir$i/custom
        mkdir -p "$dst_file"
        chmod -R u+w $dst_file >& /dev/null
        set custom_file="`find $_ASCEND_OPP_PATH/ -name "custom" |grep $i`"
        if ( "$custom_file" != "" && ! -d "$vendors_dir/$vendor_dir/$i" ) then
            set opp_custom_file=`ls $custom_file`
        else if ( -d "$vendors_dir/$vendor_dir/$i" ) then
            set opp_custom_file=`ls $vendors_dir/$vendor_dir/$i`
        endif
        if ( "$opp_custom_file" != "" ) then
            cp -rfL $custom_file/*  $dst_file
        else
            echo "[INFO]: $custom_file/ is empty"
        endif
    end
endif

