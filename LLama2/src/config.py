import torch 
import bitsandbytes as bnb
from transformers import (
    BitsAndBytesConfig, # 양자화 라이브러리 
    GenerationConfig,
    TrainingArguments
)
from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType, 
    prepare_model_for_kbit_training 
)


def get_4bit_configure() -> BitsAndBytesConfig:
    """
    4bit를 사용하기 위한 설정을 반환합니다. 
    
    Args:
        None
    
    Returns:
        int4_config (BitsAndBytesConfig): 4bit를 사용하기 위한 설정
    """
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    if compute_dtype == torch.bfloat16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)
        else: 
            raise ValueError("현재 사용하고 계신 GPU는 bfloat16 혹은 Os에서 int 4bit 지원하지 않습니다. ")
        
    bnb_4bit_compute_dtype = "bfloat16"
    use_4bit = True

    int4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return int4_config

def get_lora_config() -> LoraConfig:
    """
    lora를 사용하기 위한 설정을 반환합니다.
    
    Args:
        None

    Returns:
        LoraConfig: lora를 사용하기 위한 설정
    
    """
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False, # 학습하는지  
        r=16, # 작을 수록 trainable 한 파라미터의 개수가 낮아진ㄷ.ㅏ  
        lora_alpha=16,  # scaling factor 
        lora_dropout=0.1
    ) # dropout 
    
    return peft_config


def get_training_arg(
    output_dir: str,
    num_train_epochs: int,
    batch_size: int,
    lr: float,
    logging_steps: int    
) -> TrainingArguments:
    """
    Trainer config를 반환합니다.
    
    Args:
        None
        
    Returns:
        TrainingArguments: Trainer config를 반환합니다. 
        
    """
    training_args = TrainingArguments(
        output_dir=output_dir, 
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2*batch_size,
        learning_rate=lr,
        weight_decay=0.1,
        lr_scheduler_type='cosine',
        evaluation_strategy='epoch',
        logging_steps=logging_steps,
        optim='adamw_torch',
        save_strategy='epoch',
        save_total_limit=2
        fp16=True,
        push_to_hub=False,
    )

    return training_args