import pandas as pd
import json, re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
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

# 只取出 maintext 和 id 欄位
df_test = df_test[['maintext', 'id']]

# 加載已經微調的模型和 tokenizer
model_name = './models_mt5-small'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 將輸入資料進行 tokenize 並生成摘要
def generate(maintexts):
    # 儲存生成的摘要
    summaries = []

    # 逐筆處理 maintexts
    for text in maintexts:
        inputs = tokenizer.encode(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        # 生成摘要 (內容格式是生成文字對應的 token id)
        outputs = model.generate(
            inputs=inputs,
            max_length=128,  # 可以根據需要調整生成摘要的最大長度 (要視訓練/微調時的 max_length 而定)
            # max_new_tokens=max_new_tokens, # 限制生成的 token 數量
            do_sample=True, # 使用 sampling 生成摘要，如果是 False 則使用 Greedy Decoding
            temperature=0.5, # 控制生成的 token 的多樣性，關注用字分布的熵，值越大生成的 token 越多樣性
            top_k=50, # 從 logits 中取 top-k 個 token，值越大，選擇的 token 越多，生成的 token 越多樣性
            top_p=0.9, # 只保留累積機率至 top-p 的 token，值越小，生成的 token 越多樣性
            # num_beams=5,  # 使用 Beam Search 增強摘要質量，用在 do_sample=False 時
            # no_repeat_ngram_size=2, # 如果使用 Beam Search，可以設置這個參數，用於避免生成重複的 ngram，2 代表避免重複 2-gram
            # early_stopping=True # 如果使用 Beam Search，可以設置這個參數，當生成的 token 都是結束 token 時提前停止
        )
        
        # 對生成摘要進行解碼
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 將生成的摘要加入 summaries
        summaries.append(re.sub(r'\s', '', summary))

    return summaries

# 生成摘要並儲存在 DataFrame 當中
df_test['generated'] = generate(df_test['maintext'].tolist())

# 將資料格式轉換為字典
results = []
for _, row in df_test.iterrows():
    result = {
        "title": row['generated'],
        "id": row['id']
    }
    results.append(result)

# 將結果保存為 JSON Lines 格式
output_file = args.output
with open(output_file, 'w', encoding='utf-8') as f:
    for result in results:
        json.dump(result, f, ensure_ascii=False)
        f.write('\n')  # 每條記錄換行

print(f"mt5 生成的摘要已儲存至: {output_file}")
