'''
套件/模組匯入
'''
# 自訂模組
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))
from utils import (
    get_prompt, get_bnb_config
)

# 顯示較詳細的 cuda 錯誤訊息
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 匯入套件模組
import os, time, random, pprint, json
import torch
from datasets import load_dataset, Dataset, DatasetDict
import bitsandbytes as bnb
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training, 
    AutoPeftModelForCausalLM
)
from trl import SFTTrainer, SFTConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)



'''
參數設定
'''
################################################################################
# 模型設定
################################################################################

# 用來讀取和微調的模型名稱
model_name = "zake7749/gemma-2-2b-it-chinese-kyara-dpo"
# model_name = './models/checkpoint-2500'

################################################################################
# bitsandbytes 參數
################################################################################

# # Activate 4-bit precision base model loading
# load_in_4bit = True

# # Activate nested quantization for 4-bit base models (double quantization)
# bnb_4bit_use_double_quant = False

# # Quantization type (fp4 or nf4)
# bnb_4bit_quant_type = "nf4"

# # Compute data type for 4-bit base models
# bnb_4bit_compute_dtype = torch.float32 # torch.float16 # torch.bfloat16

################################################################################
# QLoRA 參數
################################################################################

# LoRA attention dimension
lora_r = 4

# Alpha parameter for LoRA scaling
lora_alpha = 8

# Dropout probability for LoRA layers
lora_dropout = 0.05

# Bias
bias = "none"

# Task type
task_type = "CAUSAL_LM"

################################################################################
# 訓練參數
################################################################################
# 只使用 CPU 嗎?
use_cpu = False

# 是否進行訓練?
do_train = True

# 是否進行評估?
do_eval = True

# 儲存模型預測結果以及 checkpoints 的路徑
output_dir = "./models"

# 是否要覆蓋過原先的輸出資料夾
overwrite_output_dir = True

# 儲存策略
save_strategy = "steps"

# 評估策略
eval_strategy = "steps"

# 訓練回合數 (預設 3.0)
num_train_epochs = 2.4

# 每個 GPU 當中有多少 Batch size 用來訓練
per_device_train_batch_size = 2

# 每個 GPU 當中有多少 Batch size 用來評估
per_device_eval_batch_size = 2

# 累積多少 steps 才進行梯度更新
gradient_accumulation_steps = 1

# 多少 steps 儲存一次 checkpoints
save_steps = 250

# 模型儲存最後幾個作為 checkpoints
save_total_limit = 5

# 最後是否要讀取最好的模型
load_best_model_at_end = False

# 初始化 learning rate (AdamW optimizer)
learning_rate = 2e-4

# Optimizer to use
optim = "paged_adamw_32bit"

# Training steps 的次數 (會覆蓋 num_train_epochs 設定)
max_steps = -1

# 學習率從 0 到 指定 learning rate，要花多少 steps
warmup_steps = 100

# 開啟 fp16/bf16 訓縲模式
fp16 = False

# 每幾個 step 進行一次 logging (會覆蓋 num_train_epochs 設定)
logging_steps = 500

# 日誌儲存路徑
logging_dir = './logs'

# 日誌記錄策略
logging_strategy = "epoch"

# 是否在評估與預測時，只回傳 loss 就好?
prediction_loss_only = False

# 種子
seed = 42

# 模型支援序列最大長度
max_seq_length = 8192

################################################################################
# 額外設定
################################################################################
# 切分資料，訓練資料要佔多少比例
# train_size = 0.8

# 是否存成 safetensor 格式? (False 則是存成 pytorch.bin 格式)
save_safetensors = True

# 任務名稱
run_name='translation_finetuning'

# 是否要使用 wandb 來記錄訓練過程
report_to='wandb'


'''
函式
'''
# 讀取 .txt 文件
def load_dataset_from_file(file_path):
    # 讀檔
    with open(file_path, "r", encoding='utf-8') as file:
        # 將每一行資料以 list 型態回傳
        li_data = json.loads(file.read())

        # 洗牌
        random.seed(42)
        random.shuffle(li_data)

        # 整合訓練資料
        instruction = []
        output = []
        for d in li_data:
            instruction.append(d['instruction'])
            output.append(d['output'])
        return instruction, output

# 轉換成 huggingface trainer 可以使用的 datasets
def convert_to_dataset(inst, out, tokenizer, max_length):
    # 暫放訓練資料的變數
    data = []

    # data = [{"input": i, "output": o} for i, o in zip(inst, out)]
    # return Dataset.from_pandas(pd.DataFrame(data))

    # 設定 prompt + answer 的字串，再將其 tokenize，以 embedding 格式儲存
    for i in range(len(inst)):
        # 定義系統提示字 (system prompt 隨機取得)
        prompt = get_prompt(inst[i])
        instruction = f'{prompt}{out[i]}'

        # 格式化訓練資料
        x = tokenizer(instruction, truncation=True, padding=True, return_tensors='pt', max_length = max_length)
        x['input_ids'] = x['input_ids'][0]
        x['attention_mask'] = x['attention_mask'][0]
        data.append(x)

    return data
    

# 取得 NLG 的預訓練模型
def load_model(model_name, bnb_config):
    """
    Loads model and model tokenizer

    :param model_name: Hugging Face model name
    :param bnb_config: Bitsandbytes configuration
    """
    # 取得 GPU 數量與設定最大的 GPU 記憶體用量
    n_gpus = torch.cuda.device_count()
    max_memory = '24500MiB'

    # 最大記憶體使用設定
    max_memory = {i: max_memory for i in range(n_gpus)}
    max_memory['cpu'] = '100GiB'
    # max_memory = {
    #     0: "24GiB", 
    #     # 1: "24GiB",
    #     # 2: "24GiB",
    #     # 3: "24GiB",
    # }

    # 讀取量化後的預訓練模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_safetensors = True,
        quantization_config = bnb_config,
        device_map={'':torch.cuda.current_device()}, # device_map = "auto",
        max_memory = max_memory,
    )

    # 讀取 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 將 EOS token 作為 padding token
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

# 取得模型支援的最大序列長度
def get_max_length(model):
    """
    Extracts maximum token length from the model configuration

    :param model: Hugging Face model
    """
    # 初始化儲存模型最長序列資訊的變數
    max_length = None
    
    # 尋找最大序列長度，然後將其儲存在 max_length 裡
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    
    # 如果沒有找到模型的最大序列長度資訊，就先暫時設定個 1024
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    
    return max_length

# 建立 PEFT 組態設定 (使用 LoRA)
def create_peft_config(r, lora_alpha, target_modules, lora_dropout, bias, task_type):
    """
    Creates Parameter-Efficient Fine-Tuning configuration for the model

    :param r: LoRA attention dimension
    :param lora_alpha: Alpha parameter for LoRA scaling
    :param modules: Names of the modules to apply LoRA to
    :param lora_dropout: Dropout Probability for LoRA layers
    :param bias: Specifies if the bias parameters should be trained
    """
    config = LoraConfig(
        r = r,
        lora_alpha = lora_alpha,
        target_modules = target_modules,
        lora_dropout = lora_dropout,
        bias = bias,
        task_type = task_type,
    )

    return config

# 取得 LoRA 模組列表
def find_all_linear_names(model):
    """
    Find modules to apply LoRA to.

    :param model: PEFT model
    """

    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    # lora_module_names.add("lm_head")

    lora_module_names = list(lora_module_names)
    lora_module_names.append("lm_head")

    print(f"LoRA module names: {lora_module_names}")
    return list(lora_module_names)

# 輸出用來訓練的參數量
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.

    :param model: PEFT model
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"All Parameters: {all_param:,d} || Trainable Parameters: {trainable_params:,d} || Trainable Parameters %: {100 * trainable_params / all_param}")

# 微調
def fine_tune(
        model, tokenizer, dataset, lora_r, lora_alpha,
        lora_dropout, bias, task_type, per_device_train_batch_size, per_device_eval_batch_size, 
        save_strategy, eval_strategy, gradient_accumulation_steps, warmup_steps, save_steps, 
        max_steps, learning_rate, fp16, output_dir, optim
    ):

    # 開啟梯度檢查點，以便微調過程中，減少記憶體使用量
    model.gradient_checkpointing_enable()

    # 讓模型能夠以 k bits 的規格來訓練
    model = prepare_model_for_kbit_training(model)

    # 取得 LoRA 模組名稱列表
    target_modules = find_all_linear_names(model)

    # 透過 LoRA 來建立 PEFT 組態，同時整合預訓練模型
    peft_config = create_peft_config(lora_r, lora_alpha, target_modules, lora_dropout, bias, task_type)
    model = get_peft_model(model, peft_config)

    # 輸出訓練參數相關的訊息
    print_trainable_parameters(model)

    # 建立儲存模型的路徑(資料夾)
    os.makedirs(output_dir, exist_ok = True)

    # 訓練參數
    train_args = TrainingArguments(
        run_name=run_name,
        use_cpu = use_cpu,

        do_train = do_train,
        num_train_epochs = num_train_epochs,
        per_device_train_batch_size = per_device_train_batch_size,
        save_strategy = save_strategy,
        save_total_limit = save_total_limit,
        learning_rate = learning_rate,

        do_eval = do_eval,
        load_best_model_at_end = load_best_model_at_end,
        per_device_eval_batch_size = per_device_eval_batch_size,
        eval_strategy = eval_strategy,

        save_steps = save_steps,
        gradient_accumulation_steps = gradient_accumulation_steps,
        warmup_steps = warmup_steps,
        max_steps = max_steps,
        # logging_steps = logging_steps,

        output_dir = output_dir,
        overwrite_output_dir = overwrite_output_dir,
        save_safetensors = save_safetensors,
        report_to=report_to,
        optim = optim,
        fp16 = fp16,
        seed=seed
    )

    # 建立 Huggingface 的 Trainer (可以考慮改用 SFTTrainer)
    trainer = Trainer(
        model = model,
        train_dataset = dataset['train'],
        eval_dataset = dataset['test'],
        args = train_args,
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False),
        # callbacks=[EarlyStoppingCallback(patience=3, delta=0.0)]
    )
    
    # # SFTTrainer 範例
    # trainer = SFTTrainer(
    #     model = model,
    #     train_dataset = dataset['train'],
    #     eval_dataset = dataset['test'],
    #     args = train_args,
    #     peft_config=peft_config,
    #     dataset_text_field="text",
    #     tokenizer=tokenizer,
    #     max_seq_length=max_seq_length
    # )
    
    model.config.use_cache = False

    print("Training...")

    # 進行訓練
    if do_train:
        # 迭代地訓練與記錄訓練過程(log metrics)
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print(metrics)

    # 儲存模型
    print("Saving last checkpoint of the model...")
    trainer.model.save_pretrained(output_dir)

    # 釋放記憶體
    del model
    del trainer
    torch.cuda.empty_cache()

# 建立 Early Stopping 機制
class EarlyStoppingCallback(EarlyStoppingCallback):
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


'''
主程式
'''
if __name__ == "__main__":
    # 計算單次程式執行時間
    ts = time.time()

    # 取得 bitsandbytes 設定，同時整合設定來讀取模型
    print("1. 取得 bitsandbytes 設定，同時整合設定來讀取模型")
    bnb_config = get_bnb_config()
    model, tokenizer = load_model(model_name, bnb_config)

    # 檢視量化設定
    print("2. 檢視量化設定")
    pprint.pprint(model.config.quantization_config.to_dict())

    # 取得模型支援的最大序列
    print("3. 取得模型支援的最大序列")
    max_length = get_max_length(model)

    # 讀取訓練/評估資料集
    print("4. 讀取訓練/評估資料集")
    file_path_train = './data/train.json'
    file_path_eval = './data/public_test.json'
    inst_train, out_train = load_dataset_from_file(file_path_train)
    inst_eval, out_eval = load_dataset_from_file(file_path_eval)
    preprocessed_data_train = convert_to_dataset(inst_train, out_train, tokenizer, max_length)
    preprocessed_data_eval = convert_to_dataset(inst_eval, out_eval, tokenizer, max_length)
    data = DatasetDict({
        'train': preprocessed_data_train,
        'test': preprocessed_data_eval
    })

    # 開始徵調
    print("5. 開始徵調")
    fine_tune(
        model, tokenizer, data, lora_r, lora_alpha, 
        lora_dropout, bias, task_type, per_device_train_batch_size, per_device_eval_batch_size,
        save_strategy, eval_strategy, gradient_accumulation_steps, warmup_steps, save_steps, 
        max_steps, learning_rate, fp16, output_dir, optim
    )

    # 讀取微調後的模型與權重
    print("6. 讀取微調後的模型與權重")
    model = AutoPeftModelForCausalLM.from_pretrained(
        output_dir, 
        device_map = "auto", 
        torch_dtype = torch.float32
    ) 

    # 將模型與 LoRA layers 合併
    # print("7. 將模型與 LoRA layers 合併")
    # model = model.merge_and_unload()

    # 儲存 (量化) 微調後的模型在新的路徑
    print("8. 儲存 (量化) 微調後的模型在新的路徑")
    output_adapter_dir = f"{output_dir}"
    os.makedirs(output_adapter_dir, exist_ok = True)
    model.save_pretrained(output_adapter_dir, safe_serialization = True)

    # 儲存 tokenizer
    print("10. 儲存 tokenizer")
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)

    # 輸出單次程式執行時間
    t = time.time() - ts
    print(f'Finetuning:\nIt took {t} seconds. ({t / 60} minutes) ({t/ 3600} hours)')
