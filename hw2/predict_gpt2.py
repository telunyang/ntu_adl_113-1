import torch
import json
import re
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizerFast, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import argparse

# 取得 cmd 引數
parser = argparse.ArgumentParser(description='ADL HW2')
parser.add_argument('--input', default='./input.jsonl', help='輸入 test data 的路徑 (jsonl 格式)')
parser.add_argument('--output', default='./output.jsonl', help='輸出 prediction 的路徑 (jsonl 格式)')
args = parser.parse_args()


# 讀取測試資料
# path_test_data = './tmp/public.jsonl'
# path_test_data = './tmp/sample_test.jsonl'
path_test_data = args.input


# 讀取資料
def load_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(eval(line))
    return data

# 將資料轉換為 DataFrame
df_test = pd.DataFrame(load_data(path_test_data))

# 讀取微調後的模型
model_name = './models_gpt2-base-chinese'
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 生成摘要
def generate(maintexts):
    # 儲存生成的摘要
    summaries = []

    # 逐筆處理 maintexts
    for text in maintexts:
        # 將 maintexts 轉換為模型的輸入格式
        text = text[:500]  # 限制最大長度為 500
        input_text = f"{text}\n<|sep|>\n"
        inputs = tokenizer.encode(input_text, return_tensors='pt', truncation=True, max_length=512).to(device)

        # 生成摘要
        outputs = model.generate(
            inputs=inputs,
            # max_length=128,  # 可以根據需要調整生成摘要的最大長度 (要視訓練/微調時的 max_length 而定)
            max_new_tokens=128, # 限制生成的 token 數量
            do_sample=False, # 使用 sampling 生成摘要，如果是 False 則使用 Greedy Decoding
            # temperature=0.1, # 控制生成的 token 的多樣性，關注用字分布的熵，值越大生成的 token 越多樣性
            # top_k=50, # 從 logits 中取 top-k 個 token，值越大，選擇的 token 越多，生成的 token 越多樣性
            # top_p=0.9, # 只保留累積機率至 top-p 的 token，值越小，生成的 token 越多樣性
            num_beams=2,  # 使用 Beam Search 增強摘要質量，用在 do_sample=False 時
            early_stopping=True, # 如果使用 Beam Search，可以設置這個參數，當生成的 token 都是結束 token 時提前停止
            no_repeat_ngram_size=5, # 如果使用 Beam Search，可以設置這個參數，用於避免生成重複的 ngram，2 代表避免重複 2-gram
            # pad_token_id=tokenizer.eos_token_id # 將生成的 token 以 eos_token_id 結束
        )

        # 將生成的 token docode 為文本
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # 取得生成的摘要 (透過 special token <|sep|> 進行分隔)
        if '<|sep|>' in generated_text:
            summary = generated_text.split('<|sep|>')[-1].strip()
            summary = re.sub(r'\s|\[[A-Za-z0-9]+\]', '', summary)
        else:
            # 如果分隔符號不在生成的文本中，可能需要從原始文本長度開始取得後面生成的文本
            summary = generated_text[len(text):].strip()

        # 將生成的摘要加入 summaries
        summaries.append(re.sub(r'\s', '', summary))

    return summaries

# 生成摘要
df_test['generated'] = generate(df_test['maintext'].tolist())

# 將生成的摘要儲存為 JSONL 檔案
output_file = args.output
with open(output_file, 'w', encoding='utf-8') as f:
    for _, row in df_test.iterrows():
        result = {
            "title": row['generated'],
            "id": row['id']
        }
        json.dump(result, f, ensure_ascii=False)
        f.write('\n')

print(f"gpt2 生成的摘要已儲存至: {output_file}")
