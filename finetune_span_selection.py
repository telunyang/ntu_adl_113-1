import json, argparse
from time import time
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForQuestionAnswering, AutoTokenizer,
    DataCollatorWithPadding,
    Trainer, TrainingArguments, TrainerCallback
)

# 清除 CUDA 快取
# torch.cuda.empty_cache()

# 設定 CUDA_LAUNCH_BLOCKING=1，讓程式在運行時遇到錯誤時，立即停止，方便 debug
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 取得 cmd 引數
# parser = argparse.ArgumentParser(description='批次建立索引')
# parser.add_argument('--db_path', default='./baso.db', help='資料庫檔案路徑名稱')
# args = parser.parse_args()


# 讀取模型
model_name = 'google-bert/bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
for param in model.parameters(): 
    param.data = param.data.contiguous()

# 基本設定
max_length = 512
path_context = './context.json'
path_train_data = './train.json'
path_valid_data = './valid.json'
save_to_path = 'models_final' # './models_span_selection'
output_dir = f'{save_to_path}/checkpoints'
run_name = f'{model_name}___final' # f'{model_name}___span_selection_with_validation'

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

        # 取得執行當前的損失，若小於「最佳損失值+緩衝用的增量值」
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

# 自訂資料格式
class DataCollatorForQA(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        if 'offset_mapping' in features[0]:
            batch['offset_mapping'] = [f['offset_mapping'] for f in features]
        if 'sequence_ids' in features[0]:
            batch['sequence_ids'] = [f['sequence_ids'] for f in features]
        return batch

# 自訂預處理資料的格式
class SpanSelectionDataset(Dataset):
    def __init__(self, data, tokenizer, context, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.context = context
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        question = sample['question']
        relevant_paragraph_id = sample['relevant']
        context_text = self.context[relevant_paragraph_id]
        answer_text = sample['answer']['text']
        answer_start_char = sample['answer']['start']
        answer_end_char = answer_start_char + len(answer_text)

        # 編碼問題和段落
        encoding = self.tokenizer(
            question,
            context_text,
            max_length=self.max_length,
            truncation="only_second",
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        # 獲取 offset mapping 和 sequence ids
        offset_mapping = encoding['offset_mapping'][0]
        sequence_ids = encoding.sequence_ids(0)

        start_position = next((i for i, (start, end) in enumerate(offset_mapping) if sequence_ids[i] == 1 and start <= answer_start_char < end), 0)

        end_position = next((i for i, (start, end) in enumerate(offset_mapping) if sequence_ids[i] == 1 and start < answer_end_char <= end), 0)

        return {
            'input_ids': encoding['input_ids'][0],
            'attention_mask': encoding['attention_mask'][0],
            'token_type_ids': encoding['token_type_ids'][0],
            'start_positions': torch.tensor(start_position),
            'end_positions': torch.tensor(end_position),
            'offset_mapping': encoding['offset_mapping'][0],
            'sequence_ids': sequence_ids,
        }

# 計算 exact match
def exact_match(pred):
    # 取得預測的 start 和 end logits，以及對應位置的 label ids
    start_logits, end_logits = pred.predictions
    start_labels, end_labels = pred.label_ids

    # 取得最大機率的起始和結束位置
    start_preds = np.argmax(start_logits, axis=1)
    end_preds = np.argmax(end_logits, axis=1)

    # 用來計算答對的數量
    exact_match_count = 0
    total = len(start_labels)

    for i in range(total):
        # 取得測試後的起始和結束位置
        pred_start = start_preds[i]
        pred_end = end_preds[i]

        # 取得實際答案
        true_start = start_labels[i]
        true_end = end_labels[i]

        # 比較預測和實際答案的位置是否相同，答對則 exact_match_count + 1
        if pred_start == true_start and pred_end == true_end:
            exact_match_count += 1

    # 計算 exact match 的比例
    exact_match = exact_match_count / total
    return {'exact_match': exact_match}

# 訓練模型
def train(train_dataset, eval_dataset=None):
    '''
    註:
    1. 若 eval_dataset 為 None，則只進行訓練，不進行驗證
    2. eval_strategy="no", save_strategy="no"
    '''
    # 定義訓練參數
    training_args = TrainingArguments(
        run_name=run_name,
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        eval_strategy="steps", # epoch, steps, no
        eval_steps=50,
        save_strategy="steps", # epoch, steps, no
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        # logging_dir='./logs',
        # logging_steps=500,
        # logging_strategy="epoch",
        # prediction_loss_only=True,
        report_to='wandb',
        gradient_accumulation_steps=2,
        # warmup_steps=25,
        # fp16=True,
        learning_rate=3e-5,
        max_steps=-1,
        lr_scheduler_type="linear",
        include_inputs_for_metrics=None,
        seed=42
    )

    # 使用 Hugging Face 的 Trainer API 進行 Finetune
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=exact_match,
        data_collator=DataCollatorForQA(tokenizer),
        callbacks=[AccuracyCallback()]
    )

    # 開始訓練
    trainer.train()

    # 儲存 model 與 tokenizer
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

    # 資料前處理
    train_data = SpanSelectionDataset(data=train_data, tokenizer=tokenizer, context=context, max_length=max_length)
    eval_data = SpanSelectionDataset(data=eval_data, tokenizer=tokenizer, context=context, max_length=max_length)

    # 訓練模型
    train(train_data, eval_data)
    
    t2 = time()
    print(f"[Finetuning for span selection] 程式結束，一共花費 {t2 - t1} 秒 ({(t2 - t1) / 60} 分鐘) ({(t2 - t1) / 3600} 小時)")