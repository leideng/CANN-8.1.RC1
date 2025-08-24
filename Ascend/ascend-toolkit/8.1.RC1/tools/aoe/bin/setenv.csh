#!/usr/bin/env csh

set CURFILE=`readlink -f ${1}`
set CURPATH=`dirname ${CURFILE}`

if ( "$2" == "multi_version" ) then
    set INSTALL_DIR="`realpath ${CURPATH}/../../../../latest`/tools/aoe"
else
    set INSTALL_DIR="`realpath ${CURPATH}/..`"
endif

set BIN="$INSTALL_DIR/bin"
set WORDS=`echo $PATH:q | sed 's/:/ /g'`
foreach bin ($WORDS:q)
    if (${bin} != ${CURPATH}) then
        set BIN=$BIN":"${bin}
    endif
end
set PATH=$BIN

set REAL="$INSTALL_DIR/lib64"
set LD=$REAL
set WORDS=`echo $LD_LIBRARY_PATH:q | sed 's/:/ /g'`
foreach bin ($WORDS:q)
    if (${bin} != ${LD}) then
        set LD=$LD":"${bin}
    endif
end
set LD_LIBRARY_PATH=$LD
