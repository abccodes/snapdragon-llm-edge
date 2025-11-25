## Build Instructions
https://github.com/qualcomm/llama.cpp/blob/hexagon/docs/backend/hexagon/README.md#how-to-build

Clone Llama.cpp repo:
```bash
git clone https://github.com/qualcomm/llama.cpp
```

Switch to Hexagon branch:
```bash
git checkout hexagon
```

### Setup

The easiest way to build llama.cpp for a Snapdragon-based Android device is using the toolchain Docker image (see github.com/snapdragon-toolchain). This image includes Android NDK, OpenCL SDK, Hexagon SDK, CMake, etc.

For developing on the S25 with Windows you will need specific drivers installed in order to use adb on the device: [Samsung Android USB Driver | Samsung Developer](https://developer.samsung.com/android-usb-driver)

On Ubuntu or Mac you only need ADB
Ubuntu:
```
apt install android-platform-tools
```

Mac:
```
brew install android-platform-tools
```

Windows:
Install ADB from here: 
[SDK Platform Tools release notes  |  Android Studio  |  Android Developers](https://developer.android.com/tools/releases/platform-tools)

After download the package, update the `PATH` environment variable.

### Build

This method works on Linux, macOS, and Windows. macOS and Windows users should install Docker Desktop.

Linux/Mac: 
```bash
docker run -it -u $(id -u):$(id -g) --volume $(pwd):/workspace --platform linux/amd64 ghcr.io/snapdragon-toolchain/arm64-android:v0.1
[d]/> cd /workspace
```

Windows
```powershell
docker run -it --platform linux/amd64 --volume "${PWD}:/workspace" ghcr.io/snapdragon-toolchain/arm64-android:v0.1
```

When in docker container cd to `workspace`

Run Build commands:
```bash
cp docs/backend/hexagon/CMakeUserPresets.json .

cmake --preset arm64-android-snapdragon-release -B build-snapdragon
cmake --build build-snapdragon
```

### Pushing to device
For this step, your device needs to be configured for on-device development.
See https://developer.android.com/studio/debug/dev-options for details:

- After you turn on your mobile phone, go to `Setting -> About Phone -> Software Information -> Build Number`. Tap it 7 times until you see the dialog "you are now a developer!".
- Return back to the prior page and find the `Developer Options` tab at the bottom. Toggle this option on. Toggle the USB debugging and Disable ADB timeout option on.
- Connect your device to your laptop. Make sure your laptop can detect the mobile phone.
- Try `adb shell` on your laptop. For the first time you try to login, it will say permission denied. Approve it on your mobile phone and try it again. 


Create a device installable package:
```bash
cmake --install build-snapdragon --prefix pkg-snapdragon
```

Exit container and push package contents to android device
```bash
exit
adb push pkg-snapdragon/* /data/local/tmp/llama.cpp
```

On windows `pkg-snapdragon/*` notation doesn't work just push whole folder `pkg-snapdragon`: `adb push pkg-snapdragon/ /data/local/tmp/llama.cpp`
then 
```
adb shell
cd data/local/tmp/llama.cpp
mv pkg-snapdragon/* .
rm -rf pkg-snapdragon
```

Check that the package contents are on the device, should see this from `adb shell`
```
pa2q:/data/local/tmp/llama.cpp $ ls
bin  include  lib
```

Push model to device:
```bash
adb push Llama-3.2-1B-Instruct-Q4_0.gguf /data/local/tmp/gguf/Llama-3.2-1B-Instruct-Q4_0.gguf
```

Check model is on device:
```
adb shell
cd data/local/tmp/gguf
ls
```
Update permissions of executables in `llama.cpp/bin`
```
adb shell
cd data/local/tmp/llama.cpp
chmod +x bin/*
```

### Setup on Qualcomm Device Cloud
Go to [QDC](https://qdc.qualcomm.com/) and login with your Qualcomm ID. Request for free time of the Snapdragon 8 Elite devices (you will get 1000 mins). Start an interactive session. Do not turn on any options and upload any files.

Once your session is initiated, you can login by following the tutorial given on QDC to start the SSH session and login through `adb` as illustrated previously.

> [!NOTE]
> QDC is not stable for development in a long period. It will disconnect your session and shutdown with error sometimes. You can use it as an alternative for short-term development. Long period benchmarking (> 2 hours) should be executed on your phone.

> [!TIP]
> Do `netstat` to check if any ports are occupied and change the SSH forwarding command correspondingly.

---

## Model Selection Guide

Choosing the right model depends on your specific use case, device memory constraints, and performance requirements. This guide helps you select the best model for your needs on Snapdragon edge devices.

### Quick Recommendations by Use Case

| Use Case | Recommended Model | Size | Backend | Notes |
|----------|------------------|------|---------|-------|
| **General Q&A / Chat** | Llama-3.2-1B-Instruct | 1B | HTP0 (NPU) | Fast, efficient, good for basic conversations |
| **Code Assistance** | Granite-3.3-8B-Instruct | 8B | HTP0,HTP1 | Better reasoning for coding tasks |
| **Long Document Analysis** | Llama-3.2-1B-Instruct | 1B | HTP0 | Good context handling with 32K context |
| **Knowledge Tasks** | Granite-3.3-8B-Instruct | 8B | HTP0,HTP1 | Higher accuracy for factual responses |
| **Vision + Language (VQA)** | MobileVLM_V2-1.7B | 1.7B | HTP0 | Multimodal capabilities |
| **Edge Efficiency** | LFM2-1.2B | 1.2B | HTP0 | Optimized for edge deployment |
| **Mixture of Experts** | OLMoE-1B-7B | 1B active / 7B total | HTP0,HTP1 | Efficient sparse model |

### Model Size and Memory Requirements

| Model Size | Quantization | Approx. Memory | HTP Sessions Required |
|------------|--------------|----------------|----------------------|
| 1-2B | Q4_0 | ~1-2 GB | 1 (HTP0) |
| 3-4B | Q4_0 | ~2-3 GB | 1 (HTP0) |
| 7-8B | Q4_0 | ~4-5 GB | 2 (HTP0,HTP1) |
| 13B | Q4_0 | ~7-8 GB | 2+ (HTP0,HTP1) |

### Backend Selection Guide

| Backend | Best For | Tradeoffs |
|---------|----------|-----------|
| **NPU (HTP0)** | Production use, power efficiency | Best performance/watt, limited to quantized models |
| **GPU (GPUOpenCL)** | FP16 models, larger models | Good performance, higher power consumption |
| **CPU** | Development, debugging | Slowest but most compatible |

### Choosing Based on Your Priorities

**If speed is your priority:**
- Use smaller models (1-2B parameters)
- Use NPU backend (HTP0)
- Use Q4_0 quantization

**If accuracy is your priority:**
- Use larger models (7-8B parameters)  
- Use Q8_0 quantization if memory allows
- Consider Granite or Llama models

**If you need multimodal capabilities:**
- Use MobileVLM_V2-1.7B for vision tasks
- See the "Run Vision Language Models" section below

**If you're benchmarking for competitions:**
- Start with Llama-3.2-1B-Instruct for baseline
- Test with LFM2-1.2B for efficiency
- Use the TruthfulQA and LongBench benchmarks provided

### Getting Started

For most users, we recommend starting with **Llama-3.2-1B-Instruct** as it provides a good balance of performance and quality:

```bash
wget https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf
```

---

## Running Our First Model

Pull a model to test (run this on native machine in `llama.cpp`):
```
wget https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf
```

The easiest way to run llama.cpp cli tools is using provided wrapper scripts that properly set up all required environment variables.

llama.cpp supports three backends on Snapdragon-based devices: CPU, Adreno GPU (GPUOpenCL), and Hexagon NPU (HTP0-4). You can select which backend to run the model on using the `D=` variable, which maps to the `--device` option.

Set variables with these options:
Windows
```powershell
$env:D = "HTP0" # Sets device to execute on NPU backend
$env:M = "Llama-3.2-1B-Instruct-Q4_0.gguf" # Sets desired model
```

Linux:
```bash
export D="HTP0"  # Sets device to execute on NPU backend
export M="Llama-3.2-1B-Instruct-Q4_0.gguf"  # Sets desired model
```

Available Devices:
```bash
D="none" CPU  (or --device=none in llama-cli command)
D="GPUOpenCL" (for Adreno)
D="HTP0"   signle Hexagon session (models up to 4B)
D="HTP0,HTP1"  two or more Hexagon sessions (models up to 13B)
```

To run the scripts navigate to this directory for access to the wrapper scripts
```bash
cd docs\backend\hexagon
```

Ask the model a question:
```bash
./run-cli.sh -no-cnv -p "'what is the most popular cookie in the world?'"
```

> [!NOTE]
> Windows: It is recommended to use WSL, since the testing script is Linux-based. Please update the `adb` command in `run-cli.sh` to the absolute path of your adb.exe executable.

---

## Run Testbench

For LLM benchmarking, we will use subsets of [LongBench](https://github.com/THUDM/LongBench) for long-context processing evaluation and [TruthfulQA](https://github.com/sylinrl/TruthfulQA) for short-context knowledge evaluation. These datasets are used in [NeurIPS 2024 EdgeLLM competition](https://edge-llms-challenge.github.io/edge-llm-challenge.github.io/challenge). 

> [!IMPORTANT]
> You are only allowed to use datasets that are not LongBench or TruthfulQA to train or finetune your model. For example, you can use [C4](https://huggingface.co/datasets/allenai/c4) for model pretraining and [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) for instruction tuning. The final evaluation will include hidden test cases to prevent overfitting the benchmark.

1. Unzip `prompt_files.zip` to the folder `prompt_files`.
2. Run `python -m pip install -r requirement.txt` (or manually install these packages one by one). 
3. Replace `run-cli.sh` under `/docs/backend/hexagon`.
4. TruthfulQA: run `python truthful_qa_eval.py`. After running the script, it will show the `max_score` and `accuracy`. Both metrics are higher the better. We use BLEURT, which is a model-based metric recommended in the TruthfulQA paper.
> [!WARNING]
> TruthfulQA takes around 3 hours to finish. Make sure your mobile phone is connected during the evaluation. We do not recommend using QDC for benchmarking.
5. LongBench: As an example, we provided 50 samples from the QMSum subset. Run `python longbench_test.py`. After it is finished, run `python longbench_eval.py`. It will log the average RougeL score of your model.
6. For both benchmark, the script will product a `debug.log` file. Run
```
python parse_log.py debug.log
```
This will print the average generation speed in tokens/s.

---

## Convert and Run a Huggingface model

LFM2 is a new generation of hybrid models developed by [Liquid AI](https://www.liquid.ai/), specifically designed for edge AI and on-device deployment. It sets a new standard in terms of quality, speed, and memory efficiency.

[LiquidAI/LFM2-1.2B · Hugging Face](https://huggingface.co/LiquidAI/LFM2-1.2B)

From `llama.cpp` root, clone this model repo:
```bash
git lfs install # Make sure git-lfs is installed
git clone https://huggingface.co/LiquidAI/LFM2-1.2B
```

Run `convert_hf_to_gguf_update.py` script to convert the `model.safetensors` file to `gguf` format
```bash
python convert_hf_to_gguf.py --outtype f32 --outfile ./lfm2-1.3b.gguf ./LFM2-1.2B
```

Set new model to run:
```powershell
$env:M = "lfm2-1.3b.gguf"
```
or for linux
```bash
export M="lfm2-1.3b.gguf"
```

Navigate to hexagon directory:
```
cd docs\backend\hexagon
```

Run the new model:
```bash
./run-cli.sh -no-cnv -p "'what is the most popular cookie in the world?'"
```

The above process can be adapted to work for any suitable model you find on huggingface. 

---

# Additional Notes

## Environment variables

- `GGML_HEXAGON_NDEV=1`
  Controls the number of devices/sessions to allocate. The default is 1.
  Most quantized models under 4B fit into a single session; an 8B model needs two, and a 20B model needs four.

- `GGML_HEXAGON_NHVX=0`
  Controls the number of HVX hardware threads to use. The default is all (actual number varies depending on the hardware version).

- `GGML_HEXAGON_HOSTBUF=1`
  Controls whether the Hexagon backend allocates host buffers. By default, all buffers except for REPACK are host buffers.
  This option is required for testing Ops that require REPACK buffers (MUL_MAT and MUL_MAT_ID).

- `GGML_HEXAGON_VERBOSE=1`
  Enables verbose logging of Ops from the backend. Example output:

  ```
  ggml-hex: HTP0 graph-compute n_nodes 2
  ggml-hex: HTP0 matmul : blk.27.ffn_up.weight x ffn_norm-27 -> ffn_up-27 : 3072:8192 x 3072:1 -> 8192:1 : q4_0 x f32 -> f32 : HTP0 x HTP0 -> HTP0 : flags 0x1
  ggml-hex: HTP0 matmul : blk.27.ffn_gate.weight x ffn_norm-27 -> ffn_gate-27 : 3072:8192 x 3072:1 -> 8192:1 : q4_0 x f32 -> f32 : HTP0 x HTP0 -> HTP0 : flags 0x3
  ggml-hex: HTP0 graph-compute n_nodes 1
  ggml-hex: HTP0 matmul : blk.27.ffn_down.weight x ffn_gate_par-27 -> ffn_out-27 : 8192:3072 x 8192:1 -> 3072:1 : q4_0 x f32 -> f32 : HTP0 x HTP0 -> HTP0 : flags 0x0
  ggml-hex: HTP0 get-tensor result_output : data 0x7592487000 offset 0 size 513024
  ```

- `GGML_HEXAGON_PROFILE=1`
  Generates a host-side profile for the ggml-hexagon Ops.

- `GGML_HEXAGON_OPMASK=0x0`
  Allows enabling specific stages of the processing pipeline:

  - `0x1` Enable Op Queue (i.e., queuing Ops into NPU)
  - `0x2` Enable Dynamic Quantizer (if needed for the Op)
  - `0x4` Enable Op Compute (MUL_MAT, etc.)

  Examples:

      `GGML_HEXAGON_OPMASK=0x1 llama-cli ...` - Ops are enqueued but NPU-side processing is stubbed out
      `GGML_HEXAGON_OPMASK=0x3 llama-cli ...` - NPU performs dynamic quantization and skips the rest
      `GGML_HEXAGON_OPMASK=0x7 llama-cli ...` - Full queuing and processing of Ops (default)


---

## Additional Chapters and Examples

# llama.cpp on Snapdragon — Quick Manual

This guide explains how to run **llama.cpp** on Snapdragon-based Android devices, convert Hugging Face models to GGUF, quantize them, and update model architectures. It includes **CPU** and **GPU (Adreno/OpenCL)** examples alongside Hexagon NPU usage.

---

## 1. Examples: llama.cpp on Snapdragon (CPU & GPU)

Your README already covers Hexagon (`D=HTP0`). For CPU and GPU, just change the `D=` variable:

- **CPU** → `D=none`
- **GPU (Adreno/OpenCL)** → `D=GPUOpenCL`

### 1.1 Simple prompt with Llama-3.2-1B

**CPU:**
```bash
~/src/llama.cpp$ M=Llama-3.2-1B-Instruct-Q4_0.gguf   D=none ./docs/backend/snapdragon/run-cli.sh -no-cnv   -p "what is the most popular cookie in the world?"
```

**GPU (Adreno/OpenCL):**
```bash
~/src/llama.cpp$ M=Llama-3.2-1B-Instruct-Q4_0.gguf   D=GPUOpenCL ./docs/backend/snapdragon/run-cli.sh -no-cnv   -p "what is the most popular cookie in the world?"
```

---

### 1.2 Larger model (OLMoE-1B-7B)

**CPU:**
```bash
M=../gguf/OLMoE-1B-7B-0125-Instruct-Q4_0.gguf D=none   docs/backend/hexagon/run-cli.sh -f surfing.txt -no-cnv
```

**GPU:**
```bash
M=../gguf/OLMoE-1B-7B-0125-Instruct-Q4_0.gguf D=GPUOpenCL   docs/backend/hexagon/run-cli.sh -f surfing.txt -no-cnv
```

---

### 1.3 Benchmark with llama-bench

**CPU:**
```bash
~/src/llama.cpp$ M=Llama-3.2-1B-Instruct-Q4_0.gguf   D=none docs/backend/hexagon/run-bench.sh -p 128 -n 64
```

**GPU:**
```bash
~/src/llama.cpp$ M=Llama-3.2-1B-Instruct-Q4_0.gguf   D=GPUOpenCL docs/backend/hexagon/run-bench.sh -p 128 -n 64
```

---

## 2. Convert Hugging Face Models → GGUF FP16

### 2.1 Download HF model
```bash
git clone https://huggingface.co/meta-llama/Llama-3.1-8B
```

### 2.2 Convert to GGUF FP16
```bash
python3 convert_hf_to_gguf.py ./Llama-3.1-8B --outfile ./llama-hf/Llama-3.1-8B-F16.gguf --outtype f16
```

---

## 3. Quantize GGUF Models

### 3.1 Q4_0 and Q8_0
```bash
llama-quantize Llama-3.1-8B-F16.gguf.gguf Llama-3.1-8B_Q4_0.gguf q4_0
llama-quantize Llama-3.1-8B-F16.gguf.gguf Llama-3.1-8B_Q8_0.gguf q8_0
```

### 3.2 MXFP4
- MXFP4 is a 4-bit floating-point format.
- Use **prebuilt MXFP4 GGUFs** or specialized PTQ toolchains (e.g., AMD Quark).
- Snapdragon backends mainly support Q4_0/Q8_0 for now.

---

## 4. Updating Model Architecture

1. **Conversion (Python)**  
   - Add a new `Model` subclass in `convert_hf_to_gguf.py`.
   - Map HF tensors → GGUF names, write metadata.

   ```python
   @Model.register("MyModelForCausalLM")
   class MyModel(Model):
       model_arch = gguf.MODEL_ARCH.MYMODEL
       # implement tensor mapping and metadata
   ```

2. **Runtime (C/C++)**  
   - Add architecture enum and hparams.
   - Implement GGML graph for forward pass.

3. **Test**  
   - Verify with `llama-cli`, `llama-bench`, and quantization tools.

---

## Run Vision Language Models

### 1. Download Models
Download the following files from [ZiangWu/MobileVLM_V2-1.7B-GGUF](https://huggingface.co/ZiangWu/MobileVLM_V2-1.7B-GGUF):

- `ggml-model-q4_k.gguf` (Language Model)
- `mmproj-model-f16.gguf` (Projector)

### 2. Upload to Phone via ADB
```bash
adb push ggml-model-q4_k.gguf /data/local/tmp/gguf
adb push mmproj-model-f16.gguf /data/local/tmp/gguf
```

### 3. Run the Model
Use `llama-mtmd-cli` to run the model. 

In the adb shell, check usage of `llama-mtmd-cli` with:
```bash
LD_LIBRARY_PATH=/data/local/tmp/llama.cpp/lib \
ADSP_LIBRARY_PATH=/data/local/tmp/llama.cpp/lib \
/data/local/tmp/llama.cpp/bin/llama-mtmd-cli --help
````

Refer to `run-mtmd-cli.sh` for an example script.  
A simplest example would be:
```bash
llama-mtmd-cli -m $path_to_LM --mmproj $path_to_projector --image $path_to_image -p "$input_text"
```



### 4. Run Benchmark
Note: you might need to modify the `run_llava.py` line 113-116 (code below) to correctly call the bash script
```python
out = subprocess.run(
    [
        "C:\Program Files\Git\\bin\\bash.exe", # you need to change this to the correct bash of your machine
        "./run-mtmd-cli.sh", 
        "--image", 
        REMOTE_IMG, 
        "--prompt", 
        prompt],
    text=True, capture_output=True, check=True
)
```
```bash
cd example-vqa;
python run_llava.py
```

### 5. Additional Tutorial for VLLM

More detailed information can be found in
https://github.com/qualcomm/llama.cpp/blob/hexagon/docs/multimodal.md
https://github.com/qualcomm/llama.cpp/blob/hexagon/docs/multimodal/MobileVLM.md
https://github.com/qualcomm/llama.cpp/blob/hexagon/docs/multimodal/llava.md
https://github.com/qualcomm/llama.cpp/tree/hexagon/docs/multimodal

---

## Quick Reference

```bash
# Convert HF → GGUF FP16
python3 convert_hf_to_gguf.py ./llama-hf --outfile model-F16.gguf --outtype f16

# Quantize
llama-quantize model-F16.gguf model-Q4_0.gguf q4_0
llama-quantize model-F16.gguf model-Q8_0.gguf q8_0

# Run on CPU
M=model-Q4_0.gguf D=none ./docs/backend/snapdragon/run-cli.sh -no-cnv -p "hi"

# Run on GPU
M=model-Q4_0.gguf D=GPUOpenCL ./docs/backend/snapdragon/run-cli.sh -no-cnv -p "hi"
```

---

### Notes
- Build with `GGML_OPENCL=ON` for GPU support.
- Use ADB to push binaries and models to `/data/local/tmp`.
- Wrapper scripts handle environment setup; just change `D=`.

