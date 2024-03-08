import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def compute_metrics(pred):
    """
    평가를 위한 metric을 계산하는 함수이다.
    
    Args:
        pred (List[List[int], List[float]]) : 라벨값과 예측된 logits값을 가지고 있는 리스트이다.
        
    Returns:
        metrics (Dict) : acc, f1-score, precision, recall의 평가 지표가 담겨있는 Dict을 반환한다. 
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    
    metrics = {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
   
    return metrics

def acc(pred,label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()

def find_all_linear_names(model):
    """
    Qlora 방식으로 finetuning 할 때, 학습을 진행할 수 있는 layer(모듈)을 찾고, 이름을 확인하는 함수임. 
    
    Args:
        model (transformers.AutoModelForCausalLM) : 라벨값과 예측된 logits값을 가지고 있는 리스트이다.
        
    Returns:
        metrics (List[str]) : 학습이 가능한 layer(모듈의) 이름이 들어있는 list 반환
    """

    cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # lm_head is often excluded.
    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')

    return list(lora_module_names)    

