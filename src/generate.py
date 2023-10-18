from transformers import AutoModelForCausalLM, AutoTokenizer
import timeit
from transformers.generation import GenerationConfig
import os

from icecream import ic
ic.configureOutput(includeContext=True)

# os.getenv(Environment Variable Name, Default Value)
MAX_TOKENS = int(os.getenv('MAX_TOKENS', '20'))
MIN_LENGTH = int(os.getenv('MIN_LENGTH', '1'))
DO_SAMPLE = bool(os.getenv('DO_SAMPLE', 'True'))
TEMPERATURE = float(os.getenv('TEMPERATURE', '0.7'))
TOP_K = int(os.getenv('TOP_K', '50'))
TOP_P = float(os.getenv('TOP_P', '0.95'))


def get_config(model: any):
    gen_cfg = GenerationConfig.from_model_config(model.config)

    gen_cfg.max_new_tokens = MAX_TOKENS
    gen_cfg.min_length = MIN_LENGTH
    gen_cfg.do_sample = DO_SAMPLE
    gen_cfg.temperature = TEMPERATURE
    gen_cfg.top_k = TOP_K
    gen_cfg.top_p = TOP_P
    return gen_cfg


def get_respose_length(input, output):
    return len(list(output[0].tolist())) - len(list(input[0].tolist()))


models = [
    "TheBloke/Llama-2-7b-Chat-GPTQ",
    "TheBloke/Llama-2-7b-GPTQ",
]

hf_model = models[0]


def generate(prompt: str):
    model = AutoModelForCausalLM.from_pretrained(
        hf_model, device_map="auto", trust_remote_code=True, revision="main"
    )

    model.config.max_new_tokens = MAX_TOKENS
    model.config.min_length = MIN_LENGTH

    gen_cfg = get_config(model)

    tokenizer = AutoTokenizer.from_pretrained(hf_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    prompt_template = f'''[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>
User: {prompt}[/INST]

'''

    input_ids = tokenizer(prompt_template, padding=True,
                          return_tensors='pt').input_ids.cuda()
    generation = model.generate(inputs=input_ids, generation_config=gen_cfg)

    ic(get_respose_length(input_ids, generation))

    output = tokenizer.decode(generation[0])
    output = output.replace(prompt_template, "")
    output = f"User: {prompt}\n{output}"

    return output


def main():
    prompt = "Tell me about AI"
    print(generate(prompt))


if __name__ == "__main__":
    ic(timeit.timeit(main, number=1))
