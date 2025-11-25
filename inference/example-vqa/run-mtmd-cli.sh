#!/bin/sh
#

# Basedir on device
basedir=/data/local/tmp/llama.cpp

cli_opts=

branch=.
[ "$B" != "" ] && branch=$B

adbserial=
[ "$S" != "" ] && adbserial="-s $S"

model="Llama-3.2-3B-Instruct-Q4_0.gguf"
[ "$M" != "" ] && model="$M"

mmproj="something.gguf"
[ "$M" != "" ] && mmproj="$MM"

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

# --- new: parse command-line args for image and prompt ---
while [ "$1" != "" ]; do
  case $1 in
    --image )  shift; image=$1 ;;
    --prompt ) shift; prompt=$1 ;;
    * ) cli_opts="$cli_opts $1" ;;
  esac
  shift
done

set -x



adb $adbserial shell " \
  cd $basedir; ulimit -c unlimited;        \
    LD_LIBRARY_PATH=$basedir/$branch/lib   \
    ADSP_LIBRARY_PATH=$basedir/$branch/lib \
    $experimental $sched $opmask $profile $nhvx $ndev           \
      ./$branch/bin/llama-mtmd-cli -m $basedir/../gguf/$model       \
         --mmproj $basedir/../gguf/$mmproj --no-mmproj-offload    \
         --batch-size 1 -fa on -n 30 --no-mmap -ctk q4_0 -ctv q4_0\
         --device $device --temp 0\
         --chat-template deepseek \
         --image \"$image\" -p \"$prompt\" \
"

#read -p "Press Enter to exit"
