import asyncio
from fastapi import FastAPI, Request
from .llama import Llama
# from .sse.sse import EventSourceResponse

from sse_starlette import EventSourceResponse
import json

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

STREAM_DELAY = 1  # second
RETRY_TIMEOUT = 15000  # milisecond


@app.get('/1plus1')
async def one_plus_one(request: Request):
    llama = Llama.instance()

    async def event_generator():
        prompt = request.query_params.get("prompt", "What is one plus one?")
        streamer = llama.generate(prompt)

        for new_text in streamer:
            print(new_text, end="", flush=True)
            # if (new_text != ""):
            #     # yield new_text

            if (new_text == ""):
                continue

            yield json.dumps({
                "data": new_text
            })

        print()

    return EventSourceResponse(event_generator(), sep="\n")
