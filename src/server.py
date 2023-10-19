from fastapi import FastAPI, Request, Response
from .llama import Llama, MODELS
from transformers import AutoTokenizer
from .sse.sse import EventSourceResponse
import json

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

STREAM_DELAY = 1  # second
RETRY_TIMEOUT = 15000  # milisecond

with open("system.txt", "r") as f:
    system_prompt = f.read()


@app.get('/chat')
async def chat(request: Request):
    llama = Llama.instance(system_prompt=system_prompt,
                           hf_model=MODELS.LLAMA_2_13B_CHAT_GPTQ)
    prompt = request.query_params.get("prompt")

    if prompt is None:
        return Response("Please provide a prompt", status_code=400)

    chat = [
        # {"role": "system", "content": f"<<SYS>>{system_prompt}<</SYS>>"},
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hello there, young mathematician! I'm Cosmo, your friendly and intelligent assistant here to help you learn all about limits and continuity in calculus!"},
        {"role": "user", "content": prompt},
    ]

    async def event_generator():
        streamer = llama.generate(chat)

        for new_text in streamer:
            print(new_text, end="", flush=True)

            if (new_text == ""):
                continue

            yield json.dumps({
                "data": new_text
            })

    return EventSourceResponse(event_generator(), sep="\n")

tokenizer = AutoTokenizer.from_pretrained(
    MODELS.LLAMA_2_13B_CHAT_GPTQ, use_fast=True, device_map="auto")
tokenizer.pad_token = tokenizer.eos_token


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)
