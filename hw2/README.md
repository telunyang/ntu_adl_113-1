# ntu_adl_hw2
113-1 深度學習之應用 - Homework 2
d12944007 楊德倫

---

## 1. 安裝工具

## 1.1 需要 unzip
```bash
sudo apt-get install unzip
```

## 1.2 套件安裝 (請參考 requirements.txt)
```log
transformers==4.44.2
datasets==2.21.0
accelerate==0.34.2
sentencepiece==0.2.0
evaluate==0.4.3
rouge==1.0.1
spacy==3.7.6
nltk==3.9.1
ckiptagger==0.2.1
tqdm==4.66.5
pandas==2.0.3
jsonlines==4.0.0
protobuf==4.25.5
```
註 1: 安裝指令 `pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121`
註 2: 安裝指令 `pip install -r requirements.txt`

---

## 2. 執行指令

## 2.1 下載模型並解壓縮模型
```bash
bash ./download.sh
```
註 1: 會下載 `models_mt5-small` 模型
註 2: 約 1.14 GB

## 2.2 輸出 
```bash
bash ./run.sh ./data/public.jsonl ./submission.jsonl
```
註: 用 GPU 花費時間約 19 分鐘，用 CPU 花費時間約 45 分鐘