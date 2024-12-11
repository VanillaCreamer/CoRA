import json
import argparse

from tqdm import tqdm
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer
)


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
    # Load questions file
    # prompt = (
    #     "Please summarize the user's preferences, the user has given high ratings to the following books: <ItemTitleList>."
    # )
    # annotation = pd.read_pickle("/data0/liuyuting/CoLLM/dataset/amazon_book/train_ood2.pkl").reset_index(drop=True)
    # annotation['his_title'] = annotation['his_title'].map(list)
    # annotation = annotation[annotation['his_title'].apply(lambda x: 2 < len(x) < 10)]
    # samples = annotation.sample(n=1000)
    #
    # with open(question_file, 'w') as f:
    #     for index, row in samples.iterrows():
    #         line = prompt.replace("<ItemTitleList>", ",".join(row["his_title"]))
    #         f.write(line + '\n')
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

    # prompt = (
    #     "Please summarize the user's preferences, the user has given high ratings to the following books: <ItemTitleList>."
    # )

    inputs = []
    references = []
    for i, line in enumerate(tqdm(test_data)):
        test = json.loads(line)
        input = test['input']
        reference = test["ground_truth"]
        if (i + 1) % 10 != 0:
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
                       "profile/" + args.model_path.split("/")[-1] + ".json", append=True)
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
        default="/data0/liuyuting/CoLLM/vicuna_rec",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="profile.json"
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
