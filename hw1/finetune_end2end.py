import json
import numpy as np
from time import time
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForQuestionAnswering, AutoTokenizer,
    Trainer, TrainingArguments, DataCollatorWithPadding
)

# 讀取 model 和 tokenizer
model_name = 'schen/longformer-chinese-base-4096' # 'google-bert/bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# 基本設定
max_length = 1024
path_context = './context.json'
path_train_data = './train.json'
path_valid_data = './valid.json'
save_to_path = './models_end_to_end'
output_dir = f'{save_to_path}/checkpoints'
run_name = f'{model_name}___end_to_end_qa_with_validation'

# 建立適用於目前任務的資料集
class EndToEndQADataset(Dataset):
    def __init__(self, data, tokenizer, context, max_length):
        self.examples = []

        # 將資料轉換成模型可以接受的格式
        for sample in data:
            question = sample['question']
            paragraphs = sample['paragraphs']
            relevant = sample.get('relevant')
            answer = sample.get('answer')

            # 對每個段落建立一個樣本
            for paragraph_id in paragraphs:
                paragraph_text = context[paragraph_id]
                if relevant is not None and paragraph_id == relevant:
                    # 找出答案在段落中的位置
                    answer_text = answer['text']
                    answer_start_char = answer['start']
                    answer_end_char = answer_start_char + len(answer_text)
                else:
                    # 沒有相關段落，就設定答案為空
                    answer_text = ""
                    answer_start_char = None
                    answer_end_char = None

                # 建立訓練資料
                self.examples.append({
                    'question': question,
                    'context': paragraph_text,
                    'answer_text': answer_text,
                    'answer_start_char': answer_start_char,
                    'answer_end_char': answer_end_char
                })

        # 設定 tokenizer 和最大長度
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        sample = self.examples[idx]
        question = sample['question']
        context = sample['context']
        # answer_text = sample['answer_text']
        answer_start_char = sample['answer_start_char']
        answer_end_char = sample['answer_end_char']

        encoding = self.tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=self.max_length,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        # 將答案轉換成 token 的起始和結束位置
        offset_mapping = encoding['offset_mapping'][0]
        sequence_ids = encoding.sequence_ids(0)
        cls_index = encoding['input_ids'][0].tolist().index(self.tokenizer.cls_token_id)

        # 如果答案是空的，就設定答案為 [CLS] 標記
        if answer_start_char is None:
            start_position = cls_index
            end_position = cls_index
        else:
            # 查找答案在token中的起始和结束位置
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            token_end_index = len(encoding['input_ids'][0]) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # 初始化 token 的起始和結束位置
            start_position = token_start_index
            end_position = token_end_index

            # 找出答案在 token 中的位置
            for i in range(token_start_index, token_end_index + 1):
                if offset_mapping[i][0] <= answer_start_char < offset_mapping[i][1]:
                    start_position = i
                if offset_mapping[i][0] < answer_end_char <= offset_mapping[i][1]:
                    end_position = i
                    break

            # 如果答案不在 token 中，就設定答案為 [CLS] 標記
            if start_position == token_start_index and end_position == token_end_index:
                start_position = cls_index
                end_position = cls_index

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'token_type_ids': encoding['token_type_ids'].squeeze(),
            'start_positions': torch.tensor(start_position),
            'end_positions': torch.tensor(end_position)
        }

# 計算 exact match (em)
def exact_math(p):
    start_logits, end_logits = p.predictions
    start_labels, end_labels = p.label_ids
    total = len(start_labels)
    exact_matches = 0
    for i in range(total):
        pred_start = np.argmax(start_logits[i])
        pred_end = np.argmax(end_logits[i])
        if pred_start == start_labels[i] and pred_end == end_labels[i]:
            exact_matches += 1
    return {'exact_match': exact_matches / total}

# 訓練模型
def train(train_dataset, eval_dataset=None):
    # 訓練參數
    training_args = TrainingArguments(
        run_name=run_name,
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        eval_strategy="steps", # epoch, steps, no
        eval_steps=200,
        save_strategy="steps", # epoch, steps, no
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        # logging_dir='./logs',
        # logging_steps=500,
        # logging_strategy="epoch",
        # prediction_loss_only=True,
        report_to='wandb',
        gradient_accumulation_steps=2,
        warmup_steps=25,
        # fp16=True,
        learning_rate=3e-5,
        max_steps=-1,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        seed=42
    )

    # 建立 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=exact_math
    )

    # 訓練 model
    trainer.train()

    # 保存 model 和 tokenizer
    trainer.save_model(save_to_path)
    tokenizer.save_pretrained(save_to_path)



if __name__ == '__main__':
    t1 = time()

    # 讀取資料
    with open('./context.json', "r", encoding="utf-8") as f:
        context = json.loads(f.read())
    with open('./train.json', "r", encoding="utf-8") as f:
        train_data = json.loads(f.read())
    with open('./valid.json', "r", encoding="utf-8") as f:
        eval_data = json.loads(f.read())

    # 建立資料集
    train_dataset = EndToEndQADataset(train_data, tokenizer, context, max_length)
    valid_dataset = EndToEndQADataset(eval_data, tokenizer, context, max_length)

    # 訓練模型
    train(train_dataset, valid_dataset)
    
    t2 = time()
    print(f"[Finetuning for building end-to-end model] 程式結束，一共花費 {t2 - t1} 秒 ({(t2 - t1) / 60} 分鐘) ({(t2 - t1) / 3600} 小時)")