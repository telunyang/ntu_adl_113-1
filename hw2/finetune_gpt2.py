import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizerFast, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import torch
torch.cuda.set_per_process_memory_fraction(0.9)  # 每個 Process 最多佔用 90% 的 GPU memory

# 下載預訓練模型與 tokenizer (這裡需要使用 GPT2 相關的模型)
model_name = "ckiplab/gpt2-base-chinese"
model = GPT2LMHeadModel.from_pretrained(model_name)
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
for param in model.parameters(): 
    param.data = param.data.contiguous()

# 初始化 data collator，用於生成摘要時的資料處理 (填充、截斷等)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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

# 將資料轉換為 DataFrame
data = load_data(path_train_data) + load_data(path_valid_data)
df_train = pd.DataFrame(data)
# df_train = pd.DataFrame(load_data(path_train_data))
# df_valid = pd.DataFrame(load_data(path_valid_data))

# 定義資料集格式
train_dataset = Dataset.from_pandas(df_train)
# eval_dataset = Dataset.from_pandas(df_valid)

# 刪除不必要的變數
del df_train
# del df_valid

# 新增特殊 token
special_tokens = {'sep_token': '<|sep|>'}
tokenizer.add_special_tokens(special_tokens)

# 調整模型的 token embeddings 大小
model.resize_token_embeddings(len(tokenizer))

# 預處理資料 (將 maintext 與 title 併接後，進行 tokenization)
def preprocess_data(examples):
    # 用於儲存處理後的資料
    inputs = []

    # 對每一筆資料進行處理
    for maintext, title in zip(examples['maintext'], examples['title']):
        # 將 maintext 與 title 併接 (透過自訂的特殊 token 進行分隔)
        text = f"{maintext}\n<|sep|>\n{title}"
        inputs.append(text)

    # 使用 tokenizer 對資料進行 tokenization
    model_inputs = tokenizer(
        inputs,
        truncation=True, # 決定是否截斷
        padding=True, # 決定是否 padding
        max_length=1024, # 設定最大長度，超過的部分會被截斷
    )

    # 將處理後的資料返回 (在 GPT2 當中，labels 就是 input_ids)
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    
    return model_inputs

# 準備資料集
tokenized_train_datasets = train_dataset.map(preprocess_data, batched=True, remove_columns=train_dataset.column_names)
# tokenized_eval_datasets = eval_dataset.map(preprocess_data, batched=True, remove_columns=train_dataset.column_names)

# 刪除不必要的變數
del train_dataset
# del eval_dataset

# 定義訓練參數
training_args = TrainingArguments(
    run_name=run_name,
    report_to='wandb',
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=30,
    per_device_train_batch_size=6,
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