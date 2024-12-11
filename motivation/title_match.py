import gzip
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer
)


def data():
    metadata = []
    with gzip.open("/data0/liuyuting/CoLLM/dataset/amazon_book/meta_Books.json.gz") as f:
        for l in f:
            metadata.append(eval(l.strip()))
    df_meta = pd.DataFrame.from_dict(metadata)
    titles = df_meta['title'].dropna()
    samples = titles.sample(n=10000).reset_index(drop=True)
    shuffled_index = np.random.permutation(samples.index)
    groups = np.array_split(samples.loc[shuffled_index], 1000)
    # prompt = """
    #     Given the following ten words, which one is the most relevant to the target word <target_word>? Please select the most appropriate word. Options: 1. <option>, 2. <option>, 3. <option>, 4. <option>, 5. <option>, 6. <option>, 7. <option>, 8. <option>, 9. <option>. Please respond with only one word.
    #     """
    with open("../../ChatGPT/title.json", 'w') as f:
        for i, group in tqdm(enumerate(groups)):
            df = pd.DataFrame(group, columns=['title'])
            options = df['title'].tolist()
            line = f"""Given the following nine book titles, which one is the most relevant to the target book title "{options[-1]}"? Please select the most appropriate word. Options: 1. {options[0]}, 2. {options[1]}, 3. {options[2]}, 4. {options[3]}, 5. {options[4]}, 6. {options[5]}, 7. {options[6]}, 8. {options[7]}, 9. {options[8]}. Please respond with only one book title."""
            f.write(line + '\n')


def load_model(
        model_path: str,
        device: str,
):
    if device == "cpu":
        kwargs = {"torch_dtype": torch.float32}
    elif "cuda" in device:
        kwargs = {"torch_dtype": torch.float16}
    else:
        raise ValueError(f"Invalid device: {device}")

    # tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,use_fast=True)
    tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True, padding_side='left')
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **kwargs,
        )
    except ValueError:
        model = AutoModel.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **kwargs,
        )
    except Exception as e:
        print(e)
        return None

    return model, tokenizer


def get_input(question_file):
    question_jsons = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            question_jsons.append(line)

    return question_jsons


def run_eval(args, test_data):
    # Evaluate the model for answers
    model, tokenizer = load_model(
        args.model_path, args.device
    )
    tokenizer.pad_token = tokenizer.eos_token
    if "cuda" in args.device or args.device == "mps":
        model.to(args.device)
    # model = model.to(args.device)

    inputs = []
    references = []
    for i, line in enumerate(tqdm(test_data)):
        test = json.loads(line)
        input = test['input']
        reference = test["ground_truth"]
        if (i + 1) % 40 != 0:
            inputs.append(input)
            references.append(reference)
            continue
        batch_inputs = tokenizer(inputs, return_tensors='pt', padding='longest').to(args.device)
        batch_outputs = model.generate(
            **batch_inputs,
            max_new_tokens=100,
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = []
        for input_ids, output_ids in zip(batch_inputs.input_ids, batch_outputs):
            decoded_text = tokenizer.decode(output_ids[len(input_ids):], skip_special_tokens=True).strip()
            response.append(decoded_text)
        for j in range(len(inputs)):
            dump_jsonl({"input": inputs[j], "ground_truth": references[j], "generation": response[j]},
                       "matching/" + args.model_path.split("/")[-1] + ".json", append=True)
        inputs = []
        references = []


def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, 'a+', encoding='utf-8') as f:
            json_record = json.dumps(data, ensure_ascii=False)
            f.write(json_record + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data0/liuyuting/CoLLM/vicuna_weight_working",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="ITMatching.json"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="The device type",
    )

    args = parser.parse_args()

    input = get_input(args.test_file)
    run_eval(
        args,
        input
    )