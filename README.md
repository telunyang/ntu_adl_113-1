# ntu_adl_hw1
113-1 深度學習之應用 - 作業 1

## 需要 unzip
```bash
sudo apt-get install unzip
```

## 下載模型並解壓縮模型
```bash
bash ./download.sh
```

## 輸出 prediction.csv
```bash
bash ./run.sh ./context.json ./test.json ./prediction.csv
```

## 其它

### 預覽 paragraph selection 的測試結果
```bash
python ./predict_paragraph_selection.py \
--model_path=./bert-base-chinese/models_paragraph_selection \
--context_path=./context.json \
--test_path=./test.json
```

### 預覽 span selection 的測試結果
```bash
python ./predict_span_selection.py \
--model_path=./bert-base-chinese/models_span_selection \
--context_path=./context.json \
--valid_path=./valid.json
```

### 手動輸出 prediction.csv
```bash
python ./run.py \
--model_paragraph_selection_path=./bert-base-chinese/models_paragraph_selection \
--model_span_selection_path=./bert-base-chinese/models_span_selection \
--context_path=./context.json \
--test_path=./test.json \
--output_csv_path=./prediction.csv
```