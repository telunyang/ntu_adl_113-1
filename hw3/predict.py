# 自訂模組
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))
from utils import (
    get_prompt, get_bnb_config
)
import json, re, argparse
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# 取得 cmd 引數
parser = argparse.ArgumentParser(description='ADL HW3')
parser.add_argument('--base_model_path', default='zake7749/gemma-2-2b-it-chinese-kyara-dpo', help='base_model_path')
parser.add_argument('--adapter_model_path', default='./adapter_checkpoint', help='adapter_model_path')
parser.add_argument('--input', default='./data/public_test', help='輸入 test data 的路徑 (jsonl 格式)')
parser.add_argument('--output', default='./d12944007_output.json', help='輸出 prediction 的路徑 (jsonl 格式)')
args = parser.parse_args()

# base model
base_model_path = args.base_model_path
bnb_config = get_bnb_config()
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# peft model
adapter_model_path = args.adapter_model_path
model = PeftModel.from_pretrained(model, adapter_model_path)

# 使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 儲存結果
li_prediction = []

# 讀檔
input_json_path = args.input
with open(input_json_path, "r", encoding='utf-8') as file:
    # 將每一行資料以 list 型態回傳
    li_data = json.loads(file.read())

    for d in li_data:
        prompt = get_prompt(d['instruction'])
        inputs = tokenizer.encode(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
        outputs = model.generate(
            inputs=inputs,
            # max_length=2048,  # 可以根據需要調整生成摘要的最大長度 (要視訓練/微調時的 max_length 而定)
            max_new_tokens=100, # 限制生成的 token 數量
            do_sample=True, # 使用 sampling 生成摘要，如果是 False 則使用 Greedy Decoding
            temperature=0.7, # 控制生成的 token 的多樣性，關注用字分布的熵，值越大生成的 token 越多樣性
            top_k=50, # 從 logits 中取 top-k 個 token，值越大，選擇的 token 越多，生成的 token 越多樣性
            top_p=0.9, # 只保留累積機率至 top-p 的 token，值越小，生成的 token 越多樣性
            # num_beams=5,  # 使用 Beam Search 增強摘要質量，用在 do_sample=False 時
            # no_repeat_ngram_size=2, # 如果使用 Beam Search，可以設置這個參數，用於避免生成重複的 ngram，2 代表避免重複 2-gram
            # early_stopping=True # 如果使用 Beam Search，可以設置這個參數，當生成的 token 都是結束 token 時提前停止
        )
        output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output = output[len(prompt):].strip() # 移除 prompt，只取生成的部分
        li_prediction.append({
            "id": d["id"],
            "output": output
        })

        print("=" * 80)
        print("instruction:", d['instruction'])
        print("output:", output)

# 存檔
output_json_path = args.output
with open(output_json_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(li_prediction, ensure_ascii=False, indent=4))