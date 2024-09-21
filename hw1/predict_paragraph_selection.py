import json
import torch
import argparse
from transformers import AutoModelForMultipleChoice, AutoTokenizer

# 取得 cmd 引數
parser = argparse.ArgumentParser(description='批次建立索引')
parser.add_argument('--model_path', default='./bert-base-chinese/models_paragraph_selection', help='微調後的模型路徑')
parser.add_argument('--context_path', default='./context.json', help='段落資料集的路徑')
parser.add_argument('--test_path', default='./test.json', help='測試資料的路徑')
args = parser.parse_args()

# 判斷是否有 GPU 可以使用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 讀取模型與分詞器
model_path = args.model_path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForMultipleChoice.from_pretrained(model_path)

# 放置模型到 GPU 上
model.to(device)

# 設定成評估模式
model.eval()

# 讀取資料
with open(args.context_path, "r", encoding="utf-8") as f:
    context = json.loads(f.read())
with open(args.test_path, "r", encoding="utf-8") as f:
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

# 進行評估
for index, sample in enumerate(test_data):
    # if index == 3:
    #     break
    
    # 取得合適的段落
    max_seq_length = 512
    paragraph_text = get_paragraph(sample, tokenizer, context, model, max_seq_length)

    # 取得 question
    question = sample['question']

    print("=" * 50)
    print(f"[{index}] question: ", question)
    print(f"[{index}] paragraph: ", paragraph_text)
    