import runpod
import json
import os

# local imports
from llama import Llama

MODEL = os.getenv("MODEL", "/runpod-volume/models/Llama-2-70B-chat-GPTQ")

SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."


def handler(job: dict[str, dict[str, list[dict[str, str]]]]):
    if "input" not in job:
        return {"success": False, "error": "No input provided"}

    if "chat" not in job["input"]:
        return {"success": False, "error": "No chat request provided"}

    chat = job["input"]["chat"]

    if len(chat) == 0:
        return {"success": False, "error": "No chat provided"}

    llama = Llama.instance(
        system_prompt=SYSTEM_PROMPT,
        hf_model=MODEL,
        local_model=True
    )

    streamer = llama.generate(chat)

    for new_text in streamer:
        if (new_text == ""):
            continue

        yield new_text


runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True
})
