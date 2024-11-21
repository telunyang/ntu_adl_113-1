import json
from transformers import AutoTokenizer

base_model_path = 'zake7749/gemma-2-2b-it-chinese-kyara-dpo'
tokenizer = AutoTokenizer.from_pretrained(base_model_path)


with open('./data/train.json', "r", encoding='utf-8') as file:
    tokens = 0
    li_data = json.loads(file.read())
    for index, d in enumerate(li_data):
        prompt = d['instruction']
        inputs = tokenizer.encode(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).to('cuda:0')
        tokens += len(inputs[0])

    print("總題數:", len(li_data))
    print("平均 tokens 數:", tokens / len(li_data))