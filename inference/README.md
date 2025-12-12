#### Env setup
```
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirement.txt
unzip -o prompt_files.zip -d prompt_files

Mac issue. If you run into any issues, make sure versions match:
tensorflow-macos==2.15.*
tensorflow-metal==1.1.*
bleurt
protobuf<4
absl-py<2
```

#### How to reproduce

The optimal hyperparameters are already included in each script, and each script runs on the full sample of the correlated benchmark only one time by default.

#### TruthfulQA
```
./hyperparameter_search.sh
```

#### Longbench
```
./hyperparameter_search_longbench.sh
```

#### Llamabench
```
./hyperparameter_search_llama_bench.sh
```

#### Energy measurement

Script with correct parameters located in ./run_llama.sh

```
adb start-server

adb devices

adb shell

adb push run_llama.sh /data/local/tmp/llama.cpp

On phone:
cd data/local/tmp/llama.cpp
for i in $(seq 1 100); do ./run_llama.sh -no-cnv -p 'hello?' -n 250; done
```

#### Hardware Test

Extremely basic script to test initial hardware performance of CPU vs NPU vs GPU

```
./speed_test.sh
```

