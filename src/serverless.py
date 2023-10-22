import runpod
import json

# local imports
from llama import Llama

# MODEL = "./models/Llama-2-13B-chat-GPTQ"
MODEL = "/runpod-volume/models/Llama-2-70B-chat-GPTQ"

SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."


def handler(job):
    chat = job["input"]["chat"]

    llama = Llama.instance(
        system_prompt=SYSTEM_PROMPT,
        hf_model=MODEL,
        local_model=True
    )

    streamer = llama.generate(chat)

    for new_text in streamer:
        yield json.dumps({
            "data": new_text
        })
        # print(new_text, end="", flush=True)
        # output += new_text

    # return output

# def handler(job):
#     directory = job["input"]["directory"]
#     # ls / and return the result as an array

#     stuff = os.listdir(directory)
#     return {"output": stuff}


runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True
})
