import torch
import json
import pandas as pd
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# 判斷是否有 GPU 可以使用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 讀取 model 和 tokenizer
model_path = './models_span_selection'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)

# 放置模型到 GPU 上
model.to(device)

# 設定評估模式
model.eval()

# 讀取資料
with open('./context.json', "r", encoding="utf-8") as f:
    context = json.loads(f.read())
with open('./valid.json', "r", encoding="utf-8") as f:
    valid_data = json.loads(f.read())
with open('./test.json', "r", encoding="utf-8") as f:
    test_data = json.loads(f.read())

# 取得答案
def get_answer(question, context_text, tokenizer, model, max_length):
    # 對問題與段落進行編碼
    encoding = tokenizer.encode_plus(
        question,
        context_text,
        max_length=max_length,
        truncation='only_second',
        padding='max_length',
        return_offsets_mapping=True,
        return_tensors='pt'
    )

    # 將輸入 tensors 放到裝置中
    inputs = {k: v.to(device) for k, v in encoding.items() if k != 'offset_mapping'}

    # 取得模型輸出
    with torch.no_grad():
        outputs = model(**inputs)

    # 取得模型預測的答案
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # 計算答案的機率
    start_probs = torch.softmax(start_logits, dim=1)
    end_probs = torch.softmax(end_logits, dim=1)

    # 取得最有可能的起始與結束位置
    start_index = torch.argmax(start_probs, dim=1).item()
    end_index = torch.argmax(end_probs, dim=1).item()

    # # 確認起始不會大於結束的位置
    # if end_index < start_index:
    #     end_index = start_index

    # 取得答案 (透過 offset_mapping 將 token 轉換回字元)
    offset_mapping = encoding['offset_mapping'][0]
    start_char = offset_mapping[start_index][0].item()
    end_char = offset_mapping[end_index][1].item()
    answer = context_text[start_char:end_char]

    return answer, start_char, end_char

# 預測並保存結果
results = []

# 預測每一筆 test data
for index, sample in enumerate(valid_data):
    # if index == 3:
    #     break
    
    # 取得 question
    question = sample['question']

    # 取得 paragraph_text
    relevant_paragraph_id = sample['relevant']
    paragraph_text = context[relevant_paragraph_id]

    # 設定最大長度
    max_seq_length = 512

    # 對段落進行編碼
    tokenized_tokens = tokenizer.encode(
        paragraph_text, 
        padding=True, 
        truncation=True, 
        max_length=max_seq_length, 
        add_special_tokens=True
    )

    # 取得答案
    answer, start_char, end_char= get_answer(question, paragraph_text, tokenizer, model, max_seq_length)

    # 輸出結果
    print("=" * 50)
    print(f"[{index}] question: {question}")
    print(f"[{index}] paragraph: {paragraph_text}")
    print(f"[{index}] tokenized_tokens: {len(tokenized_tokens)}")
    print(f"[{index}] answer: {answer}")
    print(f"[{index}] start_char: {start_char}, end_char: {end_char}")