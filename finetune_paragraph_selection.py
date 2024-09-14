import json, argparse
from time import time
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForMultipleChoice, AutoTokenizer,
    Trainer, TrainingArguments, TrainerCallback
)
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

# 清除 CUDA 快取
# torch.cuda.empty_cache()

# 設定 CUDA_LAUNCH_BLOCKING=1，讓程式在運行時遇到錯誤時，立即停止，方便 debug
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 取得 cmd 引數
# parser = argparse.ArgumentParser(description='批次建立索引')
# parser.add_argument('--db_path', default='./baso.db', help='資料庫檔案路徑名稱')
# args = parser.parse_args()


# 讀取模型
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
model = AutoModelForMultipleChoice.from_pretrained('bert-base-chinese')

# 基本設定
max_length = 512
path_context = './context.json'
path_train_data = './train.json'
path_valid_data = './valid.json'
save_to_path = './models_paragraph_selection'
output_dir = f'{save_to_path}/checkpoints'



# 建立 Early Stopping 機制
class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=3, delta=0.0):
        '''
        patience: 損失不更新的限制次數，超過就停止訓練，代表訓練損失可能最小
        delta: 緩衝用的增量值
        best_metric: 最佳損失值(訓練期間所記錄的較小損失)
        epochs_no_improve: 用來計算 patience 次數，來決定是否結束訓練
        '''
        self.patience = patience
        self.delta = delta
        self.best_metric = None
        self.epochs_no_improve = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # 取得 callback 執行當前的損失
        current_metric = metrics['eval_loss']

        # 初始化最佳損失值
        if self.best_metric is None:
            self.best_metric = current_metric

        # 取得執行當前的員失，若小於「最佳損失值+緩衝用的增量值」
        if current_metric < self.best_metric + self.delta:
            # 將 patience 歸零，並更新最佳損失值
            self.epochs_no_improve = 0
            self.best_metric = current_metric
        else:
            # 當前損失大於「最佳損失值+緩衝用的增量值」，則進行 patience 累計
            self.epochs_no_improve += 1

        # 損失數值不再變小，且大於等於 patience，則停止訓練
        if self.epochs_no_improve >= self.patience:
            control.should_training_stop = True

        return control

# 計算 Accuracy
class AccuracyCallback(TrainerCallback):
    def compute_accuracy(predictions, labels):
        # 將 prediction 結果和 label 從 tensor 轉換 numpy 格式
        predictions = np.argmax(predictions, axis=2)
        labels = labels.numpy()

        # 計算 accuracy
        valid_mask = labels != -100  # -100 是 masked 的 label，不該被拿來計算
        acc = np.sum((predictions == labels) * valid_mask) / valid_mask.sum()
        return acc

    def on_log(self, args, state, control, logs=None, **kwargs):
        # logs 包括了 training 和 validation 的資訊，可以從裡面拿到 prediction 和 labels
        if 'eval_logits' in logs:
            eval_accuracy = self.compute_accuracy(logs['eval_logits'], logs['eval_labels'])
            print(f"Validation Accuracy: {eval_accuracy}")

# 自訂預處理資料的格式
class ParagraphSelectionDataset(Dataset):
    def __init__(self, data, tokenizer, context, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.context = context
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        question = sample['question']
        paragraphs = sample['paragraphs']
        relevant = sample['relevant']

        # 將問題與每個段落合併，形成多選項格式
        choices = []
        for paragraph_id in paragraphs:
            paragraph_text = self.context[paragraph_id]
            encoding = self.tokenizer(
                question,
                paragraph_text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            choices.append(encoding)

        # 找出正確答案的 index
        index_label = paragraphs.index(relevant)

        # 將多個選項的 input_ids, attention_mask, token_type_ids 整理成 (num_choices, seq_length) 格式
        input_ids = torch.cat([d["input_ids"] for d in choices], dim=0).unsqueeze(0)
        attention_mask = torch.cat([d["attention_mask"] for d in choices], dim=0).unsqueeze(0)
        token_type_ids = torch.cat([d["token_type_ids"] for d in choices], dim=0).unsqueeze(0)

        return {
            'input_ids': input_ids.squeeze(0),
            'attention_mask': attention_mask.squeeze(0),
            'token_type_ids': token_type_ids.squeeze(0),
            'labels': torch.tensor(index_label)
        }
    

# 訓練模型
def train(train_dataset, eval_dataset):
    # 定義訓練參數
    training_args = TrainingArguments(
        run_name='finetune_paragraph_selection',
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        # save_steps=500,
        save_strategy="steps", # epoch
        save_total_limit=2,
        load_best_model_at_end=True,
        eval_strategy="steps", # epoch
        eval_steps=10,
        # logging_dir='./logs',
        # logging_steps=500,
        # logging_strategy="epoch",
        # prediction_loss_only=True,
        gradient_accumulation_steps=2,
        # fp16=True,
        learning_rate=3e-5,
        max_steps=-1,
        lr_scheduler_type="linear",
        seed=42
    )

    # 使用 Hugging Face 的 Trainer API 進行 Finetune
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[AccuracyCallback()]
    )

    # 開始訓練
    trainer.train()

    # 儲存 model 與 tokenizer
    trainer.save_model(save_to_path)
    tokenizer.save_pretrained(save_to_path)





if __name__ == '__main__':
    t1 = time()

    with open('./context.json', "r", encoding="utf-8") as f:
        context = json.loads(f.read())
    with open('./train.json', "r", encoding="utf-8") as f:
        train_data = json.loads(f.read())
    with open('./valid.json', "r", encoding="utf-8") as f:
        eval_data = json.loads(f.read())

    train_data = ParagraphSelectionDataset(data=train_data, tokenizer=tokenizer, context=context)
    eval_data = ParagraphSelectionDataset(data=eval_data, tokenizer=tokenizer, context=context)

    train(train_data, eval_data)
    
    t2 = time()
    print(f"[Finetuning for paragraph selection] 程式結束，一共花費 {t2 - t1} 秒 ({(t2 - t1) / 60} 分鐘) ({(t2 - t1) / 3600} 小時)")