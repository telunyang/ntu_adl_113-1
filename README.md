# ntu_adl_hw1
113-1 深度學習之應用 - 作業 1
d12944007 楊德倫

## 1. 需要 unzip
```bash
sudo apt-get install unzip
```

## 2. 套件安裝 (請參考 requirements.txt)
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

## 3. 下載模型並解壓縮模型
```bash
bash ./download.sh
```

## 4. 輸出 prediction.csv
```bash
bash ./run.sh ./context.json ./test.json ./prediction.csv
```

## 5. 其它

### 5.1 預覽 paragraph selection 的測試結果
```bash
python ./predict_paragraph_selection.py \
--model_path=./bert-base-chinese/models_paragraph_selection \
--context_path=./context.json \
--test_path=./test.json
```

### 5.2 預覽 span selection 的測試結果
```bash
python ./predict_span_selection.py \
--model_path=./bert-base-chinese/models_span_selection \
--context_path=./context.json \
--valid_path=./valid.json
```

### 5.3 手動輸出 prediction.csv
```bash
python ./run.py \
--model_paragraph_selection_path=./bert-base-chinese/models_paragraph_selection \
--model_span_selection_path=./bert-base-chinese/models_span_selection \
--context_path=./context.json \
--test_path=./test.json \
--output_csv_path=./prediction.csv
```