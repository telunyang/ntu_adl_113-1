# ntu_adl_hw1
113-1 深度學習之應用 - Homework 1
d12944007 楊德倫

---

## 1. 檔案簡介
- 微調
  - finetune_paragraph_selection.py: 微調 bert model for paragraph selection
  - finetune_span_selection.py: 微調 bert model for span seletion
  - finetune_end2end.py: 完成 Q5 的微調程式
- 預測 (測試用)
  - predict_paragraph_selection.py: 檢視 paragraph selection 的微調結果
  - predict_span_selection.py: 檢視 span selection 的微調結果
  - predict_end2end.py: 檢視 Q5 的微調結果
- 預訓練
  - pretrain_bert_mlm.py: 完成 Q4 的自訂預訓練程式
- run.py: 匯出 prediction.csv 的程式
- requirements.txt: 套件列表
- run.sh、download.sh: 作業要求的主要執行程式
- sample_submission.csv、train.json、valid.json、test.json: 作業提供的範例和資料

---

## 2. 安裝工具

## 2.1 需要 unzip
```bash
sudo apt-get install unzip
```

## 2.2 套件安裝 (請參考 requirements.txt)
```log
torch==2.1.0
scikit-learn==1.5.1
nltk==3.9.1
tqdm
numpy==1.26.4
pandas
transformers==4.44.2
datasets==2.21.0
accelerate==0.34.2
evaluate
matplotlib
gdown
wandb
```
註: 安裝指令 `pip install -r requirements.txt`

---

## 3. 執行指令

## 3.1 下載模型並解壓縮模型
```bash
bash ./download.sh
```
註 1: 會下載 `bert-base-chinese/models_paragraph_selection` 和 `bert-base-chinese/models_span_selection` 兩個模型
註 2: 合計約 780 MB

## 3.2 輸出 prediction.csv
```bash
bash ./run.sh ./context.json ./test.json ./prediction.csv
```

---

## 4. 其它 (Optional)

### 4.1 預覽 paragraph selection 的測試結果
```bash
python ./predict_paragraph_selection.py \
--model_path=./bert-base-chinese/models_paragraph_selection \
--context_path=./context.json \
--test_path=./test.json
```

### 4.2 預覽 span selection 的測試結果
```bash
python ./predict_span_selection.py \
--model_path=./bert-base-chinese/models_span_selection \
--context_path=./context.json \
--valid_path=./valid.json
```

### 4.3 手動輸出 prediction.csv
```bash
python ./run.py \
--model_paragraph_selection_path=./bert-base-chinese/models_paragraph_selection \
--model_span_selection_path=./bert-base-chinese/models_span_selection \
--context_path=./context.json \
--test_path=./test.json \
--output_csv_path=./prediction.csv
```

