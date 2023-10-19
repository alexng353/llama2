from fastapi import FastAPI, Request, Response
import json

# Local imports
from .llama import Llama, MODELS
from .sse.sse import EventSourceResponse

app = FastAPI()

with open("system.txt", "r") as f:
    SYSTEM_PROMPT = f.read()

MODEL = MODELS.LLAMA_2_13B_CHAT_GPTQ
STREAM_DELAY = 1  # second
RETRY_TIMEOUT = 15000  # milisecond


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get('/chat')
async def chat(request: Request):
    llama = Llama.instance(
        system_prompt=SYSTEM_PROMPT,
        hf_model=MODEL
    )

    prompt = request.query_params.get("prompt")

    if prompt is None:
        return Response("Please provide a prompt", status_code=400)

    chat = [
        # {"role": "system", "content": f"<<SYS>>{system_prompt}<</SYS>>"},
        {"role": "system", "content": SYSTEM_PROMPT},
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)
