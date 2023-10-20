import runpod
from icecream import ic
from .llama import Llama, MODELS

MODEL = "./models/Llama-2-13B-chat-GPTQ"

SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."


def handler(job):
    chat = job["input"]["chat"]
    llama = Llama.instance(
        system_prompt=SYSTEM_PROMPT,
        hf_model=MODEL,
        local_model=True
    )

    ic(chat)


runpod.serverless.start({"handler": handler})
