#!/bin/sh
#

# Basedir on device
basedir=/data/local/tmp/llama.cpp

cli_opts=

branch=.
[ "$B" != "" ] && branch=$B

adbserial=
[ "$S" != "" ] && adbserial="-s $S"

model="Llama-3.2-1B-Instruct-Q4_0.gguf"
[ "$M" != "" ] && model="$M"

device="HTP0"
[ "$D" != "" ] && device="$D"

verbose=
[ "$V" != "" ] && verbose="GGML_HEXAGON_VERBOSE=$V"

experimental=
[ "$E" != "" ] && experimental="GGML_HEXAGON_EXPERIMENTAL=$E"

sched=
[ "$SCHED" != "" ] && sched="GGML_SCHED_DEBUG=2" cli_opts="$cli_opts -v"

profile=
[ "$PROF" != "" ] && profile="GGML_HEXAGON_PROFILE=$PROF GGML_HEXAGON_OPSYNC=1"

opmask=
[ "$OPMASK" != "" ] && opmask="GGML_HEXAGON_OPMASK=$OPMASK"

nhvx=
[ "$NHVX" != "" ] && nhvx="GGML_HEXAGON_NHVX=$NHVX"

ndev=
[ "$NDEV" != "" ] && ndev="GGML_HEXAGON_NDEV=$NDEV"

set -x

# Windows: replace adb with the absolute path of your adb.exe executable
adb $adbserial shell " \
  cd $basedir; ulimit -c unlimited;        \
    LD_LIBRARY_PATH=$basedir/$branch/lib   \
    ADSP_LIBRARY_PATH=$basedir/$branch/lib \
    $verbose $experimental $sched $opmask $profile $nhvx $ndev           \
      ./$branch/bin/llama-cli -m $basedir/../gguf/$model       \
         -t 4 --mlock --ctx-size 32768 --batch-size 1 -ctk q8_0 -ctv q8_0 --temp 1.0 --seed 42 --no-display-prompt -fa on \
         -ngl 99 --device $device $cli_opts $@ \
"
