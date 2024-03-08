## 시스템 관련 라이브러리 로드
import os
import gc
import sys
import argparse
from utils import *

parser = argparse.ArgumentParser(description="Training LLM")

parser.add_argument("-m", "--model_name", type=str, default="meta-llama/Llama-2-7b", help="Model path, local or huggingface")
parser.add_argument("-d", "--dataset", type=str, default="mata", help="Dataset for csv or huggingface name")
parser.add_argument("-o", "--output_dir", type=str, default="output", help="Output directory")
parser.add_argument("-n", "--num", type=int, default=1, help="Number of epochs")
parser.add_argument("-b", "--batch", type=int, default=1, help="Batch size")
parser.add_argumnet("-a", "--accelator", type=bool, default=False, help="Use accelator for training") 
parser.add_argument("--train", type=str, default="hf", help="Train method for LLM, pytorch or hf")
parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")

args = parser.parse_args()


def main():
    ## dataset이 csv파일인지, huggingface 에서 가지고 오는지 확인한다.
    if args.dataset.endswith(".csv"):
        dataset = Dataset.from_pandas(pd.read_csv(args.dataset))
    elif args.dataset.endswith(".json"):
        raise ValueError("json 파일 양식은 아직 지원하지 않습니다. ")
    else:
        dataset = load_dataset(args.dataset)

    ## 모델, 토크나이저 로드
    tokenizer, model = load_model(args.model_name, device=args.device)

    if args.trian == "hf":
        train_model_hf(
            tokenizer,
            model,
            args.output_dir,
            dataset,
            num_train_epochs=args.num,
            batch_size=args.batch,
        )
    elif args.trian == "pytorch":
        train_model_pytorch(
            model, 
            train_data_loader, 
            val_data_loader, 
            num_epochs)
    else:
        raise ValueError(
            "현재 지원하는 학습 방법은 huggingface의 trainer 함수와, pytorch 뿐입니다. "
        )


if __name__ == "__main__":
    main()
