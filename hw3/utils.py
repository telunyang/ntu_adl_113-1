from transformers import BitsAndBytesConfig
import torch


def get_prompt(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: {instruction} ASSISTANT:"


def get_bnb_config() -> BitsAndBytesConfig:
    """
    {
        '_load_in_4bit': True,
        '_load_in_8bit': False,
        'bnb_4bit_compute_dtype': 'float32',
        'bnb_4bit_quant_storage': 'uint8',
        'bnb_4bit_quant_type': 'nf4',
        'bnb_4bit_use_double_quant': True,
        'llm_int8_enable_fp32_cpu_offload': False,
        'llm_int8_has_fp16_weight': False,
        'llm_int8_skip_modules': None,
        'llm_int8_threshold': 6.0,
        'load_in_4bit': True,
        'load_in_8bit': False,
        'quant_method': <QuantizationMethod.BITS_AND_BYTES: 'bitsandbytes'>
    }
    """
    ################################################################################
    # bitsandbytes 參數
    ################################################################################
    load_in_4bit = True
    bnb_4bit_use_double_quant = True
    bnb_4bit_quant_type = "nf4"
    bnb_4bit_compute_dtype = torch.bfloat16 # torch.float32 # torch.float16 # torch.bfloat16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit = load_in_4bit,
        bnb_4bit_use_double_quant = bnb_4bit_use_double_quant,
        bnb_4bit_quant_type = bnb_4bit_quant_type,
        bnb_4bit_compute_dtype = bnb_4bit_compute_dtype,
    )

    return bnb_config
