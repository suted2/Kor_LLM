## 데이터 관련 라이브러리 로드 
import re
import numpy as np
import pandas as pd 
from tqdm import tqdm
from typing import Any, Dict, List, Tuple  
from datasets import (
    Dataset, 
    DatasetDict, 
    load_dataset, 
    concatenate_datasets
)

## LLM, 딥러닝  관련 라이브러리 로드 
import torch 
import accelerate # 가속화 라이브러리
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from transformers import (
    AutoTokenizer, #토크나이저 
    AutoModelForCausalLM,
    LlamaForCausalLM,
    DataCollatorForLanguageModeling
) 

## config 관련 라이브러리 로드 
from config import *

# 학습 관련된 모델 
from peft import PeftModel

# 효율적 학습을 위한 라이브러리 , LORA 관련 라이브러리 
from transformers import Trainer, TrainingArguments 

# Evaluation 관련 라이브러리 
from metric import (
    acc, 
    compute_metrics, 
    find_all_linear_names
)

def load_model(
    model_path: str, 
    accelerator: bool = True, 
    use_n_bit: int = 32
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """  
    지정된 로컬 경로나, huggingface model hub에서 모델을 다운로드하거나 로드합니다.
    
    Args:
        model_path (str): 모델의 경로 혹은 huggingface model repo의 이름. 
        accelerator (bool): 가속화 모듈을 사용할지 여부를 지정합니다.
        use_n_bit (int): precision 정도를 지정합니다.
        
    Returns:
        Tuple[tokenizer, Model]: 지정한 모델과 토크나이저를 반환합니다. 

    Raises:
        ValueError: OS가 양자화를 지원하지 않거나, GPU 가 지원하지 않는 torch.type을 사용하려고 하는 경우 발생한다. 
        Exception: 학습을 하기 위해 CPU를 사용하는 경우 발생한다. 

    """
    if use_n_bit == 4: 
        int4_config = get_4bit_configure()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if accelerator:
            model = LlamaForCausalLM.from_pretrained(model_path, quantization_config=int4_config, device_map = 'auto')
        else:
            model = LlamaForCausalLM.from_pretrained(model_path, quantization_config=int4_config, device_map = 'cuda')

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        peft_config = get_lora_config()
        peft_model = get_peft_model(model, peft_config)

    elif use_n_bit == 8: 
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if accelerator:
            model = LlamaForCausalLM.from_pretrained(model_path, load_in_8bit=True, device_map = 'auto')
        else:
            model = LlamaForCausalLM.from_pretrained(model_path, load_in_8bit=True, device_map = 'cuda')

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        peft_config = get_lora_config()
        peft_model = get_peft_model(model, peft_config)


    return tokenizer, model


def train_model_pytorch(
    model : AutoModelForCausalLM, 
    train_data_loader, 
    val_data_loader, 
    num_train_epochs : int, 
    lr : float = 2e-7, 
    device : str = 'cuda'
    ):
    """ 
    모델을 pytorch 기본 코드로 학습을 진행하는 함수입니다. 
    
    Args:
        model (class): 모델 객체 
        train_data_loader (DataLoader) : 학습을 위한 데이터 로더입니다. 
        val_data_loader (DataLoader) : 검증을 위한 데이터 로더입니다. 
        num_train_epochs (int) : 학습을 진행할 epoch 숫자를 지정합니다.  
        lr (float) : 학습 레이트를 지정합니다. 
        device (str) : 학습을 진행할 디바이스를 지정합니다.  

    """
    ## LLM criterion, optimizer 설정 
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    ## 지정된 epoch만큼 학습을 진행한다. 
    for epoch in range(num_train_epochs):
        train_losses = []
        train_acc = 0.0
        model.train()
        
        for step, batch in enumerate(tqdm(train_data_loader)):
            label = batch['label'].to(device)
            input_id = batch['input_ids'].to(device) 
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
 
            model.zero_grad()
            pred = model(input_id, token_type_ids, attention_mask)
            
            loss = criterion(torch.sigmoid(pred.logits.t()[1]), label.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item().cpu())
            train_acc += acc(pred.logits.argmax(dim=1), label)
            
    #         if (step+1)%100==0:
    #             print("train loss: ", np.mean(train_losses))
    #             print("train acc: ", train_acc/(step*batch_size))
                
        print("train loss: ", np.mean(train_losses))
        print("train acc: ", train_acc/len(train_data_loader.dataset))
        
        val_losses = []
        val_acc = 0
        model.eval()
        
        for step, batch in enumerate(tqdm(val_data_loader)):
            label = batch['label'].to(device)
            input_id = batch['input_ids'].to(device) 
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
         
            pred = model(input_id, token_type_ids, attention_mask)
            loss = criterion(torch.sigmoid(pred.logits.t()[1]), label.float())
            
            val_losses.append(loss.item().cpu())
            val_acc += acc(pred.logits.argmax(dim=1), label)
            
        print("val loss: ", np.mean(val_losses))
        print("val acc: ", val_acc/len(val_data_loader.dataset))

    model.save_pretrained('final_model')

def train_model_hf(
    tokenizer: AutoTokenizer, 
    model: AutoModelForCausalLM, 
    output_dir: str, 
    dataset: DatasetDict, 
    model_checkpoint: str, 
    num_train_epochs: int = 1, 
    batch_size: int = 1, 
    lr: float = 2e-7, 
    device: str = 'cuda'
    ):

    """
    huggingface 의 trainer 함수를 이용한 학습 과정을 진행하기 위한 방법이다. 
    
    
    Args:
        model_path (str): 모델의 경로 혹은 huggingface model repo의 이름. 
        device (str): A small value added to the denominator for numerical stability. Default is 1e-6.

        tokenizer (transformers.Autotokenizer) : tokenizer 객체
        model (transformers.AutoModelForCausalLM): LLM 모델 객체
        output_dir (str) : 모델을 저장할 경로
        dataset (DatasetDict) : 학습을 위한 데이터셋 (train, test로 구성된. )
        num_train_epochs (int) : 학습을 진행할 epoch 숫자를 지정
        batch_size (int) : 배치 사이즈
        lr (float) : learning rate
        device (str) : 학습을 진행할 디바이스를 지정한다.

    Returns:

    
    """
    logging_steps = len(dataset['train']) // batch_size
       
    # 아래의 TrainingArguments에 대한 설명 
    # https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments 에서 자세한 사항 확인 가능. 


    training_args = get_training_arg(output_dir, num_train_epochs, batch_size, lr, logging_steps) 
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
   

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=tokenizer
        )

    if model_checkpoint:
        print('모델 체크포인트가 존재합니다. 해당 체크포인트로부터 학습을 시작합니다. ')
        trainer.train(resume_from_checkpoint = model_checkpoint) ## 학습시작 
    else:
        trainer.train() ## 학습시작 

    print("학습이 완료 되었습니다.")
    final_save_folder = output_dir + '/final_model'

    model.save_pretrained(final_save_folder)
    tokenizer.save_pretrained(final_save_folder)

    print("모델 저장이 완료되었습니다. 저장된 폴더는 다음과 같습니다. ", final_save_folder)
        
