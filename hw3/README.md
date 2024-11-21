# ntu_adl_hw3
113-1 深度學習之應用 - Homework 3
d12944007 楊德倫

---

## 1. 安裝工具

## 1.1 需要 unzip
```bash
sudo apt-get install unzip
```

## 1.2 套件安裝 (請參考 requirements.txt)
```log
# 助教基本要求
torch==2.4.1
transformers==4.45.1
bitsandbytes==0.44.1
peft==0.13.0

# 額外套件
safetensors
datasets
trl
wandb
accelerate
```
註：安裝指令 `pip install -r requirements.txt`

---

## 2. 執行指令

## 2.1 下載模型並解壓縮模型
```bash
bash ./download.sh
```
註 1: 會下載 `adapter_checkpoint`
註 2: 約 2.4 GB

## 2.2 輸出推論後的 JSON 檔案
```bash
time bash ./run.sh 'zake7749/gemma-2-2b-it-chinese-kyara-dpo' ./adapter_checkpoint ./data/public_test.json ./output/d12944007_output.json
```

## 2.3 Grading - ppl.py
```bash
time python ppl.py --base_model_path 'zake7749/gemma-2-2b-it-chinese-kyara-dpo' --peft_path ./adapter_checkpoint --test_data_path ./data/public_test.json
```