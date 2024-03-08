import os
import gc
import sys
import argparse
from utils import *
from transformers import GenerationConfig

parser = argparse.ArgumentParser(description="Training LLM")

parser.add_argument("-m", "--model_name", type=str, default="meta-llama/Llama-2-7b", help="Model path, local or huggingface")
parser.add_argument("-p", "--prompt", type=str, default="안녕 너는 누구야?", help="Prompt for model generation")
parser.add_argumnet("--int4", type=bool, default=False, help="Use 4Bit inference for training")
parser.add_argument("--device", type=str, default="gpu", help="Device to use for training")

args = parser.parse_args()


def get_config(top_k: int, temperature: float, max_new_tokens: int) -> GenerationConfig:
    """
    LLM의 Generation Config를 정하는 함수.

    Args:
        top_k (int): 다음 단어의 확률 분포에서 상위 K개만 선택헤서 그중 랜덤하게 선택하게 된다.  default는 10
        top_p (int, optional ): 가장 가능성이 높은 분포에서만 선택하게 된다.
        temperature (float): 다음 토큰 확률의 분포에 변화를 주게된다.

    Returns:
        generate_config (GenerationConfig): 생성을 위한 config를 반환한다.

    """
    generate_config = GenerationConfig(
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=10,
        top_p=0.9,
        num_return_sequences=1,
        repetition_penalty=1.1,
        max_new_tokens=500,
        temperature=0.8,
    )
    return generate_config


def generate(
    tokenizer: AutoTokenizer, model: AutoModelForCausalLM, input: str, max_length: int
) -> str:
    """
    LLM에서 input prompt 를 받아 답변을 생성하는 함수

    Args:
        tokenizer () : tokenizer
        model () : model
        input (str) : LLM 에게 요청하기 위한 input prompt .
        max_length (int) : token 단위로 생성할 최대 길이를 의미한다.

    Returns:
        generate_config (GenerationConfig): 생성을 위한 config를 반환한다.

    Raise:
        Exception: max_token_lenght 가 모델이 지정한 max_length 보다 길 경우 에러를 발생시킨다.

    """

    assert (
        model.max_length < max_length
    ), "현재 모델이 생성할 수 있는 최대 길이를 넘어 성능의 저하를 부를 수 있다. "

    outputs = tokenizer(
        input,
        truncation=True,
        max_length=max_length,
        return_overflowing_tokens=True,  # 자른 나머지를 버리는 것이 아닌 추후에 다시 사용하는 것을 의미한다.
        return_length=True,  # 길이를 같이 return 하게 된다.
        return_tensors="pt",
    )

    generate_config = get_config(10, 0.9)

    generate_ids = model.generate(
        input_ids=outputs.input_ids.to("cuda"),
        attention_mask=outputs.attention_mask.to("cuda"),
        generation_config=generate_config,
    )

    result = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return result


def peft_generate():
    pass


def main():
    model_path = args.model_name
    prompt = args.prompt
    int4 = args.int4
    device = args.device

    # 모델 로드
    tokenizer, model = load_train_model(model_path, device=device, DO_PEFT=int4)

    assert tokenizer.padding_side == "left", "패딩 사이드가 왼쪽이 아닙니다."

    generation_config = get_config(10, 0.9, 500)  # config 설정
    outputs = generate(tokenizer, model, prompt, 1000)  # 결과 생성


if __name__ == "__main__":
    main()
