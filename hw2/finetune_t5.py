import pandas as pd
import wandb
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
from rouge_score import rouge_scorer
import torch
torch.cuda.set_per_process_memory_fraction(0.9)  # 每個 Process 最多佔用 90% 的 GPU memory

# 下載預訓練模型與 tokenizer
model_name = "google/mt5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
for param in model.parameters(): 
    param.data = param.data.contiguous()

# 初始化 data collator，用於生成摘要時的資料處理 (填充、截斷等)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# ROUGE 計算工具，包含 ROUGE-1, ROUGE-2, ROUGE-L
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# 基本設定
# max_length = 512
path_train_data = './tmp/train.jsonl'
path_valid_data = './tmp/public.jsonl'
postfix = model_name.split('/')[-1]
save_to_path = f'./models_{postfix}'
output_dir = f'{save_to_path}/checkpoints'
run_name = f'{postfix}_finetune'

# 讀取資料
def load_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(eval(line))
    return data

# 計算 metrics 的函數
def compute_metrics(eval_pred):
    # 取出預測與標籤
    preds, labels = eval_pred

    # 將預測與標籤解碼
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # 如果 labels 有 -100 (specail token)，將其移除以便計算 ROUGE 分數
    # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 初始化 ROUGE 分數的總和
    rouge1_total = 0
    rouge2_total = 0
    rougeL_total = 0

    # 計算每個預測與標籤的 ROUGE 分數
    for pred, label in zip(decoded_preds, decoded_labels):
        scores = scorer.score(pred, label)
        rouge1_total += scores['rouge1'].fmeasure
        rouge2_total += scores['rouge2'].fmeasure
        rougeL_total += scores['rougeL'].fmeasure

    # 計算 ROUGE-1, ROUGE-2, ROUGE-L 的平均值
    num_samples = len(decoded_preds)
    result = {
        'rouge1': rouge1_total / num_samples * 100,
        'rouge2': rouge2_total / num_samples * 100,
        'rougeL': rougeL_total / num_samples * 100
    }

    del decoded_preds, decoded_labels, preds, labels, num_samples

    return result

# 將資料轉換為 DataFrame
data = load_data(path_train_data) + load_data(path_valid_data)
df_train = pd.DataFrame(data)
# df_train = pd.DataFrame(load_data(path_train_data))
# df_valid = pd.DataFrame(load_data(path_valid_data))

# 選擇需要的欄位 maintext 為輸入，title 為輸出
df_train = df_train[['maintext', 'title']]
# df_valid = df_valid[['maintext', 'title']]

# 定義資料集格式
train_dataset = Dataset.from_pandas(df_train)
# eval_dataset = Dataset.from_pandas(df_valid)

# 刪除不必要的變數
del df_train
# del df_valid

# 預處理資料 (將 maintext 與 title 進行 tokenization)
def preprocess_data(examples):
    # 如果 tokenizer 沒有設定 pad_token，將其設定為 eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 將 maintext 進行 tokenization
    inputs = [example for example in examples['maintext']]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding=True)

    # 將 title 進行 tokenization
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['title'], max_length=128, truncation=True, padding=True)

    # 更新 model_inputs，加入 labels
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

# 準備資料集
tokenized_train_datasets = train_dataset.map(preprocess_data, batched=True)
# tokenized_eval_datasets = eval_dataset.map(preprocess_data, batched=True)

# 刪除不必要的變數
del train_dataset
# del eval_dataset

# 定義訓練參數
training_args = TrainingArguments(
    run_name=run_name,
    report_to='wandb',
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=21,
    # per_device_eval_batch_size=1,
    # eval_strategy="steps", # epoch, steps, no
    # eval_steps=300,
    save_strategy="epoch", # epoch, steps, no
    save_steps=500,
    save_total_limit=3,
    # load_best_model_at_end=True,
    gradient_accumulation_steps=1,
    # eval_accumulation_steps=2,
    warmup_steps=200,
    weight_decay=0.01,
    learning_rate=3e-5,
    max_steps=-1,
    lr_scheduler_type="linear",
    # fp16=True,
    # gradient_checkpointing=True,
    # optim="adafactor",
    # metric_for_best_model='eval_loss',
    # dataloader_num_workers=4,
    seed=42
)

# 使用 Trainer 進行微調
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_datasets,
    data_collator=data_collator,
    # eval_dataset=tokenized_eval_datasets,
    # compute_metrics=compute_metrics
)

# 刪除不必要的變數
del tokenized_train_datasets

# 開始訓練
trainer.train()

# 儲存 model 與 tokenizer
trainer.save_model(save_to_path)
tokenizer.save_pretrained(save_to_path)

# 刪除不必要的變數
del trainer, tokenizer, model