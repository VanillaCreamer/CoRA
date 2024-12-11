from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from tqdm import tqdm
import json
import numpy as np
import evaluate
from transformers import LlamaTokenizer, AutoModelForCausalLM


# 计算困惑度
if __name__ == "__main__":
    test_file = "/data2/liuyuting/logs/reviews1.txt"
    perplexities = []

    # 加载 GPT-2 模型和分词器
    # model_name = "/data0/liuyuting/CoLLM/vicuna_weight_working"  #"/data2/liuyuting/models/gpt2"  # 你也可以选择 gpt2-medium, gpt2-large 等模型
    # model = GPT2LMHeadModel.from_pretrained(model_name)
    # tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # model = AutoModelForCausalLM.from_pretrained(model_name)
    # tokenizer = LlamaTokenizer.from_pretrained(model_name, padding_side='left')
    calculator = evaluate.load("/data2/liuyuting/evaluate/metrics/perplexity", module_type="metric")

    # with open(test_file) as f:
    #     for i, line in tqdm(enumerate(f)):
    #         sentence = json.loads(line)['review']
    #         inputs = tokenizer(sentence, return_tensors="pt", max_length=1024, truncation=True)
    #         input_ids = inputs.input_ids
    #
    #         # 使用语言模型计算 logits 并提取 log-probabilities
    #         with torch.no_grad():
    #             outputs = model(input_ids, labels=input_ids)
    #             log_likelihood = outputs.loss * input_ids.size(1)
    #         perplexity = torch.exp(log_likelihood / input_ids.size(1))
    #         perplexities.append(perplexity.item())
    # print('avg perplexity score:', np.sum(perplexities))
    sentence = "a6a"
    # inputs = tokenizer(sentence, return_tensors="pt")
    # input_ids = inputs.input_ids
    #
    # # 使用语言模型计算 logits 并提取 log-probabilities
    # with torch.no_grad():
    #     outputs = model(input_ids, labels=input_ids)
    #     log_likelihood = outputs.loss * input_ids.size(1)
    # perplexity = torch.exp(log_likelihood / input_ids.size(1))
    perplexity = calculator.compute(model_id='/data2/liuyuting/models/gpt2', predictions=sentence)
    print(perplexity['mean_perplexity'])

#  ours: 234.44969940185547  CoLLM: 173.73327684402466