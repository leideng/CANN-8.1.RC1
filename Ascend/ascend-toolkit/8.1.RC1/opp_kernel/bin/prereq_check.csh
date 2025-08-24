#!/bin/csh

set FILE_NOT_EXIST="0x0080"

set cur_date=`date +"%Y-%m-%d %H:%M:%S"`
echo "[Opp][${cur_date}][INFO]: Start pre installation check of opp module."
python3 --version >/dev/null 2>&1
if ($status != "0") then
set cur_date=`date +"%Y-%m-%d %H:%M:%S"`
exit 0
endif
set python_version=`python3 --version`
set idx=`echo ${python_version} | awk '{print index($0, "3.7.5")}'`
if ($idx > 0) then
set cur_date=`date +"%Y-%m-%d %H:%M:%S"`
exit 0
else
set cur_date=`date +"%Y-%m-%d %H:%M:%S"`
exit 0
endif
