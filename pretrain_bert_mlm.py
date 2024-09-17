import os, argparse, time
from transformers import (
    BertTokenizer, 
    BertConfig, 
    BertForMaskedLM,
    LineByLineTextDataset,
    DataCollatorForLanguageModeling, 
    Trainer, 
    TrainingArguments,
    TrainerCallback
)
import numpy as np

def main(tokenizer_path, config, train_data, eval_data, save_to_path):
    # 建立放置模型的資料夾路徑
    if not os.path.exists(save_to_path):
        os.makedirs(save_to_path)

    '''
    # 進行 continual learning (further pretraining)，取得 checkpoint 的模型名稱
    model_name = "bert-base-multilingual-cased"
    model = BertForPreTraining.from_pretrained(model_name)

    # 可能要另外建立 tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    '''
    
    # 從頭開始建立自己的 BERT 預訓練模型
    cfg = BertConfig.from_json_file(config)
    # model = BertForPreTraining(cfg) # 用在同時訓練 NSP 和 MLM
    model = BertForMaskedLM(cfg) # 僅用於訓練 MLM
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    # 獲取模型的總參數量
    total_parameters = model.num_parameters()
    print(f"Total parameters: {total_parameters}")

    # 序列的最大 tokens 數量
    max_seq_length = 510 # 1022

    # 僅用於 MLM 的訓練與評估資料格式
    train_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=train_data,
        block_size=max_seq_length,
    )
    eval_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=eval_data,
        block_size=max_seq_length,
    )

    # 建立 Masked Language Model 的校正器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True,
        mlm_probability= 0.15
    )

    # 訓練參數
    training_args = TrainingArguments(
        use_cpu=False,
        do_train=True,
        do_eval=True,
        output_dir=save_to_path,
        overwrite_output_dir=True,
        num_train_epochs=30,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        # save_steps=500,
        save_strategy="epoch",
        save_total_limit=5,
        load_best_model_at_end=True,
        evaluation_strategy="epoch",
        # eval_steps=500,，
        logging_dir=save_to_path,
        # logging_steps=500,
        logging_strategy="epoch",
        prediction_loss_only=True,
        gradient_accumulation_steps=1,
        fp16=True,
        learning_rate=5e-5,
        max_steps=-1,
        lr_scheduler_type="linear",
        seed=42
    )

    # 訓練器
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(patience=3, delta=0.0)],
        callbacks=[AccuracyCallback()]
    )

    # 開始訓練
    trainer.train()

    # 儲存模型
    trainer.save_model(save_to_path)

    # 儲存 tokenizer 設定
    tokenizer.save_pretrained(save_to_path)


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




# 主程式
if __name__ == "__main__":
    # 取得指令引數
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", help="進行分詞功能的資料夾路徑", type=str)
    parser.add_argument("--config", help="模型設定的檔案路徑", type=str)
    parser.add_argument("--train_data", help="訓練資料的檔案路徑", type=str)
    parser.add_argument("--eval_data", help="評估資料的檔案路徑", type=str)
    parser.add_argument("--save_to_path", help="訓練後的模型資料夾路徑", type=str)
    args = parser.parse_args()

    # 執行主程式
    t_begin = time.time()
    main(
        args.tokenizer_path, 
        args.config, 
        args.train_data,
        args.eval_data,
        args.save_to_path
    )
    print(f"整體執行花費時間: {time.time() - t_begin} 秒")