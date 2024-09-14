import json, logging
import torch
import pandas as pd
from transformers import (
    AutoModelForMultipleChoice, 
    AutoModelForQuestionAnswering, 
    AutoTokenizer
)

# 判斷是否有 GPU 可以使用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 讀取模型與分詞器
model_path = './models_paragraph_selection'
tokenizer_paragraph = AutoTokenizer.from_pretrained(model_path)
model_paragraph = AutoModelForMultipleChoice.from_pretrained(model_path)
model_path = './models_span_selection'
tokenizer_span = AutoTokenizer.from_pretrained(model_path)
model_span = AutoModelForQuestionAnswering.from_pretrained(model_path)

# 放置兩個模型到相同的裝置上
model_paragraph.to(device)
model_span.to(device)

# 設定成評估模式
model_paragraph.eval()
model_span.eval()

# 讀取作業提供的段落資料
with open('./context.json', "r", encoding="utf-8") as f:
    context = json.loads(f.read())
with open('./test.json', "r", encoding="utf-8") as f:
    test_data = json.loads(f.read())

# 取得段落
def get_paragraph(sample, tokenizer, context, model, max_length):
    # 取得問題
    question = sample['question']

    # 取得段落 (有多個段落，都是 id)
    paragraphs = sample['paragraphs']

    # 將問題與每個段落合併，形成多選項格式
    choices = []
    for paragraph_id in paragraphs:
        paragraph_text = context[paragraph_id]
        encoding = tokenizer(
            question,
            paragraph_text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}
        choices.append(encoding)

    # 將多選項格式轉換成模型可以接受的格式
    input_ids = torch.stack([x['input_ids'][0] for x in choices]).unsqueeze(0)
    attention_mask = torch.stack([x['attention_mask'][0] for x in choices]).unsqueeze(0)
    token_type_ids = torch.stack([x['token_type_ids'][0] for x in choices]).unsqueeze(0)

    # 準備模型輸入
    model_inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
    }

    # 取得模型輸出
    with torch.no_grad():
        outputs = model(**model_inputs)
        logits = outputs.logits

    # 選擇最高分數的段落
    selected_index = torch.argmax(logits, dim=1).item()
    selected_paragraph = context[paragraphs[selected_index]]

    return selected_paragraph

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

    # 確認起始不會大於結束的位置
    if end_index < start_index:
        end_index = start_index

    # 取得答案 (透過 offset_mapping 將 token 轉換回字元)
    offset_mapping = encoding['offset_mapping'][0]
    start_char = offset_mapping[start_index][0].item()
    end_char = offset_mapping[end_index][1].item()
    answer = context_text[start_char:end_char]

    return answer


# 放置 id 與 answer 的結果
results = []

# 進行評估
for index, sample in enumerate(test_data):
    # 取得合適的段落
    max_seq_length = 512
    paragraph_text = get_paragraph(sample, tokenizer_paragraph, context, model_paragraph, max_seq_length)

    # 取得 question
    question = sample['question']

    print("=" * 50)
    print(f"[{index}] question: ", question)
    print(f"[{index}] paragraph: ", paragraph_text)
    

    # # 預測答案
    # answer = get_answer(question, paragraph_text, tokenizer_span, model_span, max_seq_length)

    # # 儲存結果
    # result = {
    #     'id': sample['id'],
    #     'answer': answer
    # }
    # results.append(result)

    # # 輸出結果
    # print(f"question: {question}")
    # print(f"prediction: {answer}")
    # print("=" * 10)

# # 將結果轉換為 DataFrame 並儲存為 CSV 檔
# df = pd.DataFrame(results, columns=['id', 'answer'])
# df.to_csv('./prediction.csv', index=False)