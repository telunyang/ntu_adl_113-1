import json
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# 讀取 model 和 tokenizer
model_name = './models_end_to_end'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval() # 設定為評估模式

# 基本設定
max_length = 1024
path_context = './context.json'
path_test_data = './test.json'

# 取得預測結果
def predict(test_data, context, model, tokenizer, max_length):
    predictions = []
    
    # 取得每個測試樣本
    for sample in test_data:
        # 取得問題和段落
        question = sample['question']
        paragraphs = sample['paragraphs']
        
        # 初始化最佳答案
        best_answer = ''

        # 分數設定為負無窮
        best_score = float('-inf')

        # 對每個段落進行預測
        for paragraph_id in paragraphs:
            # 取得段落文字
            paragraph_text = context[paragraph_id]

            # 將問題和段落轉換成模型可以接受的格式
            encoding = tokenizer(
                question,
                paragraph_text,
                truncation="only_second",
                max_length=max_length,
                padding="max_length",
                return_offsets_mapping=True,
                return_tensors="pt"
            )

            # 將資料放到 GPU 上
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            token_type_ids = encoding.get('token_type_ids')

            # 如果有 token_type_ids，也放到 GPU 上
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            # 進行預測
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )

            # 取得起始和結束位置的機率
            start_logits = outputs.start_logits[0]
            end_logits = outputs.end_logits[0]

            # 將機率轉換成機率分佈
            start_probs = torch.softmax(start_logits, dim=-1)
            end_probs = torch.softmax(end_logits, dim=-1)

            # 設定最大答案長度
            max_answer_length = 50

            # 將機率由大到小排序
            start_indexes = torch.argsort(start_probs, descending=True).cpu().numpy()
            end_indexes = torch.argsort(end_probs, descending=True).cpu().numpy()

            # 尋找最佳答案
            found_answer = False
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index <= end_index
                        and end_index - start_index + 1 <= max_answer_length
                        and encoding['offset_mapping'][0][start_index][0] != 0
                        and encoding['offset_mapping'][0][end_index][1] != 0
                    ):
                        # 計算答案分數
                        answer_score = start_probs[start_index] + end_probs[end_index]

                        # 如果分數比最佳分數高，就更新最佳分數
                        if answer_score > best_score:
                            offset_mapping = encoding['offset_mapping'][0]
                            start_char = offset_mapping[start_index][0]
                            end_char = offset_mapping[end_index][1]
                            answer_text = paragraph_text[start_char:end_char]
                            best_score = answer_score
                            best_answer = answer_text.strip()
                            found_answer = True

                        # 如果找到答案，就跳出 for loop
                        break
                
                # 如果找到答案，就跳出 for loop
                if found_answer:
                    break

        # 將預測結果加到 predictions
        predictions.append({
            'id': sample['id'],
            'question': question,
            'answer': best_answer
        })

    return predictions



if __name__ == '__main__':
    # 讀取資料
    with open(path_context, "r", encoding="utf-8") as f:
        context = json.loads(f.read())
    with open(path_test_data, "r", encoding="utf-8") as f:
        test_data = json.loads(f.read())

    # 執行預測
    predictions = predict(test_data, context, model, tokenizer, max_length)

    # 輸出預測結果
    for pred in predictions:
        print("=" * 50)
        print(f"Question: {pred['question']}")
        print(f"Answer: {pred['answer']}")