from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, TextIteratorStreamer
import timeit
from transformers.generation import GenerationConfig
import os
from datetime import datetime
from threading import Thread

from icecream import ic
ic.configureOutput(includeContext=True)

# os.getenv(Environment Variable Name, Default Value)
MAX_TOKENS = int(os.getenv('MAX_TOKENS', '400'))
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
    "TheBloke/Llama-2-13B-chat-GPTQ"
]

hf_model = models[2]


def main():
    model = AutoModelForCausalLM.from_pretrained(
        hf_model, device_map="auto", trust_remote_code=True, revision="main"
    )

    model.config.max_new_tokens = MAX_TOKENS
    model.config.min_length = MIN_LENGTH

    gen_cfg = get_config(model)

    tokenizer = AutoTokenizer.from_pretrained(
        hf_model, use_fast=True, device_map="auto")
    tokenizer.pad_token = tokenizer.eos_token

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

#   prompt = """

# """
#   prompt_template=f'''[INST] <<SYS>>
# You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Respond briefly and concisely, and don't use too many words.
# <</SYS>>
# {prompt}[/INST]

# '''
    prompt = "What is one plus one?"
    prompt_template = f"""[INST] <<SYS>>
You are a deterministic, honest and truthful assistant program. You are being asked to generate a single multiple choice question based off of a given prompt.
The machine that reads your responses is extremely strict, and will only accept the following XML-like format with no extraneous text:

<QN>
<T>"The question"</T>
<OPT1>"The first option"</OPT1>
<OPT2>"The second option"</OPT2>
<OPT3>"The third option"</OPT3>
<OPT4>"The fourth option"</OPT4>
<ANS>n</ANS>
</QN>

(where n is the number of the correct answer, 1-4)

<</SYS>>
{prompt}[/INST]"""

    start_time = datetime.now()
    inputs = tokenizer(prompt_template, padding=True,
                       return_tensors='pt').input_ids.cuda()
    tokenize_time = datetime.now()
    generation_kwargs = dict(
        inputs=inputs, streamer=streamer, generation_config=gen_cfg)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)

    # generation = model.generate(inputs=inputs, generation_config=gen_cfg, streamer=streamer)
    generation_time = datetime.now()

    print(f"User: {prompt}\n")

    thread.start()
    output = ""
    for new_text in streamer:
        print(new_text, end="", flush=True)
        output += new_text
    print()

    # print(f"Tokenize time: {tokenize_time - start_time}")
    # print(f"Generation time: {generation_time - tokenize_time}")

    # ic(get_respose_length(inputs.input_ids.cuda(), generation))

    # output = tokenizer.decode(generation[0])
    # output = output.replace(prompt_template, "")
    # output = f"User: {prompt}\n{output}"


if __name__ == "__main__":
    ic(timeit.timeit(main, number=1))
