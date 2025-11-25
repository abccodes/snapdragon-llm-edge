#!/bin/sh
#
# Runner script for llama.cpp on Snapdragon via adb, with StreamLLM-style sinks.
#

########################################
# SELECT BACKEND HERE:
#   MODE=CPU  → CPU only (no --device, -ngl 0)
#   MODE=NPU  → Hexagon HTP0 NPU (--device HTP0, -ngl 99)
#   MODE=GPU  → Adreno GPU via OpenCL (--device GPUOpenCL, -ngl 99)
########################################
MODE="${MODE:-CPU}"   # default if not set

########################################
# SINK / STREAMING SETTINGS
#
# These control the StreamLLM-style behavior:
#   ENABLE_SINKS    → 1 to enable context-shift + sinks, 0 to disable
#   SINK_KEEP       → how many tokens from the start to keep as sinks
#   CTX_SIZE        → context window size (must be <= model's max ctx)
#   N_PRED          → max tokens to generate (can exceed CTX_SIZE to force shifting)
########################################
ENABLE_SINKS="${ENABLE_SINKS:-1}"   # 1 = enable sinks + sliding window, 0 = no sinks
SINK_KEEP="${SINK_KEEP:-4}"        # number of sink tokens to keep pinned at left
CTX_SIZE="${CTX_SIZE:-4096}"       # llama-cli --ctx-size
N_PRED="${N_PRED:-250}"            # llama-cli -n (n_predict)



basedir=/data/local/tmp/llama.cpp

# Extra CLI options (gets -v appended when SCHED is set)
cli_opts=

# Branch / subdir name on device (usually ".")
branch=.
[ "$B" != "" ] && branch="$B"

# Optional: select device by serial number via env S
adbserial=
[ "$S" != "" ] && adbserial="-s $S"

# Default model on device; override with env M if needed
model="qwen2-7b-tinytron-Q4_K_M.gguf"
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

########################################
# Build sink / context-shift arguments
########################################

sink_args=
if [ "$ENABLE_SINKS" != "0" ]; then
    # --context-shift turns on KV sliding
    # --keep $SINK_KEEP pins the first SINK_KEEP tokens as sinks
    sink_args="--context-shift --keep $SINK_KEEP"
fi

set -x

# Note: To list available accelerator devices (GPUOpenCL, HTP0, CPU) on the phone:
#   adb shell 'cd /data/local/tmp/llama.cpp; \
#       LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib \
#       ./bin/llama-cli --list-devices'

adb $adbserial shell " \
  cd $basedir; ulimit -c unlimited;        \
    LD_LIBRARY_PATH=$basedir/$branch/lib   \
    ADSP_LIBRARY_PATH=$basedir/$branch/lib \
    $verbose $experimental $sched $opmask $profile $nhvx $ndev           \
      ./$branch/bin/llama-cli -m $basedir/../gguf/$model       \
        -t 8 --ctx-size $CTX_SIZE --batch-size 128 \
        -ctk f16 -ctv f16 --temp 1.0 --seed 42 \
        --no-display-prompt -fa on \
        -ngl $ngl $dev_arg -n $N_PRED $sink_args \
	-sys 'You are a helpful assistant. Provide truthful, accurate, and concise answers.' \
	$cli_opts $@ \
"

