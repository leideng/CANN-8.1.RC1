#!/bin/csh

set func_name = "$1"
switch ( "$func_name" )
    case "mk_custom_path":
        if ( "`id -u`" == 0 ) then
            exit 0
        endif
        set file_path = "$2"
        foreach line ("` cat $file_path `")
            set custom_path = "`echo '$line' | cut --only-delimited -d= -f2`"
            if ( "$custom_path" == "" ) then
                continue
            endif
            set custom_path = "` eval echo $custom_path `"
            if ( ! -d "$custom_path" ) then
                mkdir -p "$custom_path"
                if ( $status != 0 ) then
                    set cur_date = "`date +'%Y-%m-%d %H:%M:%S'`"
                    echo "[Common] [$cur_date] [ERROR]: create $custom_path failed."
                    exit 1
                endif
            endif
        end
        breaksw
    default:
        breaksw
endsw
