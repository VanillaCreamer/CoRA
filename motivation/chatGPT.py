import os
from openai import OpenAI
import json
from tqdm import tqdm

def get_input(question_file):
  question_jsons = []
  with open(question_file, "r") as ques_file:
    for line in ques_file:
      question_jsons.append(line)
  return question_jsons


def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, 'a+', encoding='utf-8') as f:
            json_record = json.dumps(data, ensure_ascii=False)
            f.write(json_record + '\n')


# os.environ["OPENAI_API_KEY"] = "sk-Mgq5e922e8b37be51a47d7511a645eefc07e2756b6e9Qxpk"
# os.environ["OPENAI_API_BASE"] = "https://api.gptsapi.net"
client = OpenAI(api_key="sk-Mgq5e922e8b37be51a47d7511a645eefc07e2756b6e9Qxpk", base_url="https://api.gptsapi.net/v1")  # api_key="sk-Mgq5e922e8b37be51a47d7511a645eefc07e2756b6e9Qxpk", api_base="https://api.gptsapi.net"
# response = client.chat.completions.create(
#     model="gpt-3.5-turbo",  # 你可以选择其他合适的模型
#     messages=[{"role": "user", "content": "Hello!"}],
# )
# print(response.choices[0].message.content.strip())

results = []
test_data = get_input("user_history.json")
for i, line in enumerate(tqdm(test_data)):
    prompt = line
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # 你可以选择其他合适的模型
        messages=[{"role": "user", "content": line}]
    )
    result = response.choices[0].message.content.strip()
    dump_jsonl({"input": prompt, "ground_truth": result},
               "gpt.json", append=True)