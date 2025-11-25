#!/bin/sh
#
# Runner script for llama.cpp on Snapdragon via adb.
#

########################################
# SELECT BACKEND HERE:
#   MODE=CPU  → CPU only (no --device, -ngl 0)
#   MODE=NPU  → Hexagon HTP0 NPU (--device HTP0, -ngl 99)
#   MODE=GPU  → Adreno GPU via OpenCL (--device GPUOpenCL, -ngl 99)
########################################
MODE="${MODE:-CPU}"   # default if not set

########################################

# Basedir on device (matches where you pushed pkg-snapdragon payload)
basedir=/data/local/tmp/llama.cpp

# Extra CLI options (gets -v appended when SCHED is set)
# Branch / subdir name on device (usually ".")
branch=.
[ "$B" != "" ] && branch="$B"

# Optional: select device by serial number via env S
adbserial=
[ "$S" != "" ] && adbserial="-s $S"

# Default model on device; override with env M if needed
model="granite-3.3-8b-instruct-gguf"
[ "$M" != "" ] && model="$M"

########################################
# Resolve MODE → device + ngl
########################################

device=""
ngl=0

case "$MODE" in
    CPU|cpu)
        device=""
        ngl=0
        ;;
    NPU|npu|HTP0)
        device="HTP0"
        ngl=99
        ;;
    GPU|gpu|GPUOpenCL)
        device="GPUOpenCL"
        ngl=99
        ;;
    *)
        echo "Unknown MODE='$MODE'. Use CPU, NPU, or GPU." >&2
        exit 1
        ;;
esac

# Build optional --device argument (CPU mode → no flag)
dev_arg=
if [ "$device" != "" ]; then
    dev_arg="--device $device"
fi

########################################
# Optional Hexagon / scheduler / profiling env vars
########################################

verbose=
[ "$V" != "" ] && verbose="GGML_HEXAGON_VERBOSE=$V"

experimental=
[ "$E" != "" ] && experimental="GGML_HEXAGON_EXPERIMENTAL=$E"

sched=
if [ "$SCHED" != "" ]; then
    sched="GGML_SCHED_DEBUG=2"
    cli_opts="$cli_opts -v"
fi

profile=
[ "$PROF" != "" ] && profile="GGML_HEXAGON_PROFILE=$PROF GGML_HEXAGON_OPSYNC=1"

opmask=
[ "$OPMASK" != "" ] && opmask="GGML_HEXAGON_OPMASK=$OPMASK"

nhvx=
[ "$NHVX" != "" ] && nhvx="GGML_HEXAGON_NHVX=$NHVX"

ndev=
[ "$NDEV" != "" ] && ndev="GGML_HEXAGON_NDEV=$NDEV"

set -x

# Note: To list available accelerator devices (GPUOpenCL, HTP0) on the phone:
#   adb shell 'cd /data/local/tmp/llama.cpp; \
#       LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib \
#       ./bin/llama-cli --list-devices'
# CPU does NOT appear there; CPU mode is used when no --device flag is provided.

adb $adbserial shell " \
  cd $basedir; ulimit -c unlimited;        \
    LD_LIBRARY_PATH=$basedir/$branch/lib   \
    ADSP_LIBRARY_PATH=$basedir/$branch/lib \
    $verbose $experimental $sched $opmask $profile $nhvx $ndev           \
      ./$branch/bin/llama-cli -m $basedir/../gguf/$model       \
        -t 4 --mlock --ctx-size 32768 --batch-size 1 \
        -ctk q8_0 -ctv q8_0 --temp 1.0 --seed 42 \
        --no-display-prompt -fa on \
        -ngl $ngl $dev_arg -n 250 $cli_opts $@ \
"


