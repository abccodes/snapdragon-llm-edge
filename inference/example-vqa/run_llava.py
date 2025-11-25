import torch
import os
import sys
import random

import numpy as np
from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets

from argparse import ArgumentParser

from utils.data_utils import load_yaml, construct_prompt, save_json, process_single_sample, CAT_SHORT2LONG
from utils.model_utils import call_llava_engine_df, llava_image_processor
from utils.eval_utils import parse_multi_choice_response, parse_open_response

import subprocess
import re
import json
import numpy as np
from tqdm import tqdm
import shlex

def parse_output(output):
    # TODO you can implement your own parse function
    # save output
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    output = output.split("\n\n")[1:]
    return "".join([s.strip() for s in output])

def main():
    parser = ArgumentParser()
    parser.add_argument('--output_path', type=str, default='llava1.5_13b_val.json',
                        help='name of saved json')
    parser.add_argument('--data_path', type=str, default="lmms-lab/textvqa") # hf dataset path. # "lmms-lab/textvqa"
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    print('llava_initializing...')
    processor = None
    call_model_engine = call_llava_engine_df
    vis_process_func = llava_image_processor

    # load config and process to one value
    #args.config = load_yaml(args.config_path)
    #for key, value in args.config.items():
    #    if key != 'eval_params' and type(value) == list:
    #        assert len(value) == 1, 'key {} has more than one value'.format(key)
    #        args.config[key] = value[0]


    if "MMMU" in args.data_path:
        # run for each subject
        sub_dataset_list = []
        for subject in CAT_SHORT2LONG.values():
            sub_dataset = load_dataset(args.data_path, subject, split=args.split)
            sub_dataset_list.append(sub_dataset)
            break

        # merge all dataset
        dataset = concatenate_datasets(sub_dataset_list)
        question = "final_input_prompt"
    elif "textvqa" in args.data_path:
        dataset = load_dataset("lmms-lab/textvqa", split="validation")
        dataset = dataset.select(range(100))
        question = "question"
    else:
        raise NotImplementedError

    # load model
#    model_name = get_model_name_from_path(args.model_path)
#    tokenizer, model, vis_processors, _ = load_pretrained_model(args.model_path, None,
#                                                                model_name)
    REMOTE_IMG = "/data/local/tmp/image.jpg"
    samples = []
    patterns = {
        "load_time_ms": r"load time\s*=\s*([\d.]+)\s*ms",
        "prompt_eval_time_per_token_ms": r"prompt eval time\s*=\s*[\d.]+\s*ms\s*/\s*\d+\s*tokens\s*\(\s*([\d.]+)\s*ms per token",
        "eval_time_per_token_ms": r"eval time\s*=\s*[\d.]+\s*ms\s*/\s*\d+\s*runs\s*\(\s*([\d.]+)\s*ms per token",
        "total_time_ms": r"total time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens"
    }
    load_time_ms = []
    prompt_eval_time_per_token_ms = []
    eval_time_per_token_ms = []
    total_time_ms = []
    total_token = []
    correct = []

    for sample in tqdm(dataset):

        if "MMMU" in args.data_path:
            sample = process_single_sample(sample)
            sample = construct_prompt(sample, args.config)

        if sample['image']:
            image = sample['image'].convert("RGB")
            image.save("image.jpg", "JPEG")
            prompt = sample[question]
            print(prompt)
            prompt = prompt.replace('"', '\\"').replace("`", "\\`").replace("$", "\\$")
            # use bash command "adb push image.jpg /data/local/tmp/image.jpg" to upload image to file
            subprocess.run(["adb", "push", "image.jpg", REMOTE_IMG], check=True)
            try:
                # you might need to change the command below if you use ubuntu or mac or other bash
                out = subprocess.run(
                    ["C:\Program Files\Git\\bin\\bash.exe", "./run-mtmd-cli.sh", "--image", REMOTE_IMG, "--prompt", prompt],
                    text=True, capture_output=True, check=True
                )
                output = parse_output(out.stdout)
                print(output)
                match = 0
                for answer in sample["answers"]:
                    if answer == output.lower():
                        match = 1
                        break
                correct.append(match)
                # parse runtime profile information
                results = {}
                for key, pattern in patterns.items():
                    match = re.search(pattern, out.stderr)
                    if match:
                        if key == "total_time_ms":
                            results["total_time_ms"] = float(match.group(1))
                            results["total_tokens"] = int(match.group(2))
                        else:
                            results[key] = float(match.group(1))
                load_time_ms.append(results["load_time_ms"])
                prompt_eval_time_per_token_ms.append(results["prompt_eval_time_per_token_ms"])
                eval_time_per_token_ms.append(results["eval_time_per_token_ms"])
                total_time_ms.append(results["total_time_ms"])
                total_token.append(results["total_tokens"])

            except subprocess.CalledProcessError as e:
                print("--- SCRIPT FAILED ---", file=sys.stderr)
                print(f"Exit Code: {e.returncode}", file=sys.stderr)

                print("\n--- SCRIPT STDOUT ---", file=sys.stderr)
                print(e.stdout, file=sys.stderr)

                print("\n--- SCRIPT STDERR (Error Message) ---", file=sys.stderr)
                print(e.stderr, file=sys.stderr)  # This is the most important part!
                sys.exit(1)

    print("average correct")
    print(np.mean(correct))
    print("average load time")
    print(np.mean(load_time_ms))
    print("average prompt eval time")
    print(np.mean(prompt_eval_time_per_token_ms))
    print("average eval time")
    print(np.mean(eval_time_per_token_ms))
    print("average total time")
    print(np.mean(total_time_ms))
    print("average total token")
    print(np.mean(total_token))



if __name__ == '__main__':
    main()

